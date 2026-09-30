"""Correlated UF with full cluster-gap confidence, followed by plain MWPM.

Cluster gaps follow the construction of Meister, Pattison and Preskill
(arXiv:2405.07433), applied here to the final correlation-reweighted UF graph.
They are confidence heuristics, not exact complementary gaps or log odds.
No confidence calibration, search cap, or score quantization is applied.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import stim

from yoked.decoders._correlated_union_find import CorrelatedUnionFindDecoder
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
from yoked.hierarchical._patch_graphs import PatchGraphs

CORRELATED_UF_CONFIGURATION = 'correlated_uf_full_cluster_gap__plain_mwpm'


@dataclass(frozen=True)
class CorrelatedUFDecodeResult:
    """Patch-major arrays (shots, 2P); syndrome and ties are (shots, 2).

    ``reference`` is the final correlated-UF prediction. ``cluster_gap`` is
    computed from that pass's clusters and weights. ``dijkstra_states`` counts
    settled search states, a work measure rather than a latency estimate.
    """
    reference: np.ndarray
    cluster_gap: np.ndarray
    dijkstra_states: np.ndarray
    residual: np.ndarray
    prediction: np.ndarray
    sigma: np.ndarray
    outer_tied: np.ndarray

    def __post_init__(self):
        reference = np.asarray(self.reference)
        if reference.ndim != 2 or reference.shape[1] < 2 or reference.shape[1] % 2:
            raise ValueError('Expected reference shape (shots, 2 * patches)')
        shots, columns = reference.shape
        for field in fields(self):
            value = np.asarray(getattr(self, field.name))
            shape = (shots, 2) if field.name in ('sigma', 'outer_tied') else reference.shape
            if value.shape != shape:
                raise ValueError(f'Invalid shape for {field.name}')
            if field.name == 'cluster_gap':
                if not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError('Cluster gaps must be finite and nonnegative')
                dtype = np.float64
            elif field.name == 'dijkstra_states':
                if value.dtype.kind not in 'iu' or (value < 0).any():
                    raise ValueError('Search work must be a nonnegative integer count')
                dtype = np.int64
            else:
                if not np.isin(value, (0, 1)).all():
                    raise ValueError(f'{field.name} must be binary')
                dtype = bool
            object.__setattr__(self, field.name, readonly_array(value, dtype=dtype))
        if not np.array_equal(self.prediction, self.reference ^ self.residual):
            raise ValueError('Prediction must equal reference XOR residual')
        parity = self.residual.reshape(shots, columns // 2, 2).sum(axis=1) % 2
        if not np.array_equal(parity, self.sigma):
            raise ValueError('Residual parity must equal the frame-adjusted syndrome')

    def arrays(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


class CorrelatedUFHierarchicalDecoder:
    """Two-pass correlated UF + full per-sector cluster gaps -> plain MWPM.

    Correlations are conditioned on the first UF correction. The second UF
    pass supplies the logical reference and residual growth costs. The gap
    search only reuses graph topology from the original patch; all per-shot
    costs come from the final UF pass. Ideal yokes enter only the L2 stage.
    """

    def __init__(self, dem: stim.DetectorErrorModel, *, num_patches: int | None = None):
        self.patches = PatchGraphs.from_yoked_dem(
            dem, num_patches=dem.num_observables // 2 if num_patches is None else num_patches)
        self.decoders = tuple(
            CorrelatedUnionFindDecoder(
                patch.graph, correlation_rules=correlation_rules_from_dem(patch.graph, patch.local_dem))
            for patch in self.patches)
        self.gap_searches = tuple(ClusterGapUnionFindDecoder(patch.graph) for patch in self.patches)

    def decode_with_gaps_batch(self, detectors) -> CorrelatedUFDecodeResult:
        detectors = np.asarray(detectors)
        if (detectors.ndim != 2 or detectors.shape[1] != self.patches.num_detectors
                or not np.isin(detectors, (0, 1)).all()):
            raise ValueError(f'Expected binary detectors of shape (shots, {self.patches.num_detectors})')
        shots, patches = len(detectors), len(self.patches)
        reference = np.zeros((shots, 2 * patches), dtype=bool)
        gaps = np.zeros(reference.shape, dtype=np.float64)
        states = np.zeros(reference.shape, dtype=np.int64)
        for index, (decoder, search) in enumerate(zip(self.decoders, self.gap_searches)):
            local = self.patches[index].local_syndromes(detectors)
            columns = slice(2 * index, 2 * index + 2)
            for shot, syndrome in enumerate(local):
                decoded = decoder.decode_with_growth_costs(syndrome)
                reference[shot, columns] = [(decoded.observable_mask >> sector) & 1 for sector in range(2)]
                # No max_gap is supplied: retain the full confidence ordering.
                gaps[shot, columns], states[shot, columns] = search.gaps_from_costs(decoded.remaining_costs)

        yoke = detectors[:, self.patches.yoke_detector_ids]
        sigma = frame_adjusted_syndrome(yoke, reference)
        residual = np.zeros_like(reference)
        tied = np.zeros((shots, 2), dtype=bool)
        for sector in range(2):
            decision = mwpm_outer_log_odds_batch(gaps[:, sector::2], sigma[:, sector])
            residual[:, sector::2] = decision.patterns
            tied[:, sector] = decision.tied
        prediction = reference ^ residual
        if not np.array_equal(prediction.reshape(shots, patches, 2).sum(axis=1) % 2, yoke):
            raise ValueError('Final predictions violate the measured yoke checks')
        return CorrelatedUFDecodeResult(reference, gaps, states, residual, prediction, sigma, tied)

    def decode_batch(self, detectors):
        return self.decode_with_gaps_batch(detectors).prediction

    def decode(self, detectors):
        return self.decode_batch(np.asarray(detectors)[None])[0]
