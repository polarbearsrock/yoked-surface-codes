"""The supplied native MPP confidence estimator with correlated L1 and plain L2.

The native extension is optional: importing this module neither builds it nor
downloads dependencies. Its build manifest pins the engine and identifies the
binary used for each experiment. Baseline decoding does not depend on MPP.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pymatching
import stim

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
from yoked.hierarchical._patch_graphs import PatchGraphs
from yoked.hierarchical._provenance import REPOSITORY_ROOT, sha256_file

MPP_CONFIGURATION = 'correlated_mwpm_mpp_cluster_score__plain_mwpm'


def load_mpp_native():
    """Load only an explicitly selected, verified build of the optional extension."""
    directory = os.environ.get('YOKED_MPP_BUILD')
    if not directory:
        raise RuntimeError('Build MPP with tools/build_mpp, then set YOKED_MPP_BUILD to its output directory')
    return _load_build(str(Path(directory).resolve()))


@lru_cache(maxsize=1)
def _load_build(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'build.json').read_text())
    if metadata['schema'] != 'mpp-native-build/1':
        raise ValueError('Unsupported MPP build manifest')
    for source in ('native/mpp/mpp_fast.cc', 'native/mpp/CMakeLists.txt', 'tools/build_mpp'):
        if sha256_file(REPOSITORY_ROOT / source) != metadata['source_sha256'][source]:
            raise ValueError(f'Stale MPP build: {source} changed; rerun tools/build_mpp')
    module_path = directory / Path(metadata['module']).name
    if sha256_file(module_path) != metadata['module_sha256']:
        raise ValueError('MPP binary hash disagrees with build manifest')
    spec = importlib.util.spec_from_file_location('_mpp_fast', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, metadata


@dataclass(frozen=True)
class MppDecodeResult:
    """Patch-major arrays (shots, 2P); native weights are (shots, P).

    ``cluster_score`` uses the native engine's normalization, with larger values
    indicating greater confidence. Its units do not make it an exact log-odds.
    This initial experiment uses it directly as a heuristic L2 edge weight.
    """
    reference: np.ndarray
    cluster_score: np.ndarray
    native_weight: np.ndarray
    residual: np.ndarray
    prediction: np.ndarray
    sigma: np.ndarray
    outer_tied: np.ndarray

    def __post_init__(self):
        reference = np.asarray(self.reference)
        if reference.ndim != 2 or reference.shape[1] < 2 or reference.shape[1] % 2:
            raise ValueError('Expected reference shape (shots, 2 * patches)')
        shots, columns = reference.shape
        shapes = {'native_weight': (shots, columns // 2), 'sigma': (shots, 2), 'outer_tied': (shots, 2)}
        for field in fields(self):
            value = np.asarray(getattr(self, field.name))
            if value.shape != shapes.get(field.name, reference.shape):
                raise ValueError(f'Invalid shape for {field.name}')
            floating = field.name in ('cluster_score', 'native_weight')
            if floating:
                if not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f'{field.name} must be finite and nonnegative')
            elif not np.isin(value, (0, 1)).all():
                raise ValueError(f'{field.name} must be binary')
            object.__setattr__(self, field.name, readonly_array(value, dtype=float if floating else bool))
        if not np.array_equal(self.prediction, self.reference ^ self.residual):
            raise ValueError('Prediction must equal reference XOR residual')
        if not np.array_equal(self.residual.reshape(shots, columns // 2, 2).sum(axis=1) % 2, self.sigma):
            raise ValueError('Residual parity must equal the frame-adjusted yoke syndrome')

    def arrays(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


class MppHierarchicalDecoder:
    """Native correlated MWPM + MPP cluster score -> plain MWPM across patches.

    Correlation mode is deliberately fixed to ``all``: ``none`` and ``cross``
    change L1 itself and would confound this confidence-only comparison. The
    native adapter still exposes these modes for separate validation studies.
    ``verify=True`` enables the slow independent score oracle for every shot.
    """
    def __init__(self, dem: stim.DetectorErrorModel, *, num_patches: int | None = None,
                 method: str = 'dijkstra', verify: bool = False):
        if method not in ('dijkstra', 'incremental'):
            raise ValueError('method must be dijkstra or incremental')
        if pymatching.__version__ != '2.4.0':
            raise RuntimeError('MPP comparison requires the validated PyMatching 2.4.0 wheel')
        module, self.build_metadata = load_mpp_native()
        self.method, self.verify = method, verify
        self.patches = PatchGraphs.from_yoked_dem(
            dem, num_patches=dem.num_observables // 2 if num_patches is None else num_patches)
        self.decoders = tuple(module.SoftDecoder(str(patch.local_dem), 'all', method) for patch in self.patches)

    def decode_with_scores_batch(self, detectors) -> MppDecodeResult:
        detectors = np.asarray(detectors)
        if (detectors.ndim != 2 or detectors.shape[1] != self.patches.num_detectors
                or not np.isin(detectors, (0, 1)).all()):
            raise ValueError(f'Expected binary detectors of shape (shots, {self.patches.num_detectors})')
        shots, patches = len(detectors), len(self.patches)
        reference = np.zeros((shots, 2 * patches), dtype=bool)
        scores = np.zeros(reference.shape)
        native_weights = np.zeros((shots, patches))
        # Local score computation sees neither ideal yokes nor truth labels.
        for index, decoder in enumerate(self.decoders):
            local = self.patches[index].local_syndromes(detectors).astype(np.uint8)
            prediction, score, weight = decoder.decode_batch(local, verify=self.verify)
            reference[:, 2*index:2*index+2] = prediction
            scores[:, 2*index:2*index+2] = score
            native_weights[:, index] = weight

        yoke = detectors[:, self.patches.yoke_detector_ids]
        sigma = frame_adjusted_syndrome(yoke, reference)
        residual = np.zeros_like(reference)
        tied = np.zeros((shots, 2), dtype=bool)
        for sector in range(2):
            # Exactly the same L2 solver and tie convention as the baseline.
            result = mwpm_outer_log_odds_batch(scores[:, sector::2], sigma[:, sector])
            residual[:, sector::2] = result.patterns
            tied[:, sector] = result.tied
        prediction = reference ^ residual
        if not np.array_equal(prediction.reshape(shots, patches, 2).sum(axis=1) % 2, yoke):
            raise ValueError('Final predictions violate the measured yoke checks')
        return MppDecodeResult(reference, scores, native_weights, residual, prediction, sigma, tied)

    def decode_batch(self, detectors):
        return self.decode_with_scores_batch(detectors).prediction

    def decode(self, detectors):
        return self.decode_batch(np.asarray(detectors)[None])[0]
