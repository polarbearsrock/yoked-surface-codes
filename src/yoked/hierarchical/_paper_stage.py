"""Run and retain the paper-style configuration on generated or verified saved shots."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import stim

from yoked.hierarchical._collect import SampleSet, _positions
from yoked.hierarchical._paper_decoder import PAPER_CONFIGURATION, PaperHierarchicalDecoder
from yoked.hierarchical._provenance import (
    atomic_replacement, canonical_json, git_commit, package_versions, sha256_bytes,
    sha256_file, source_hashes, utc_now, write_json_atomic,
)


_SOURCES = (
    'src/yoked/decoders/_graph.py',
    'src/yoked/hierarchical/_arrays.py',
    'src/yoked/hierarchical/_patch_graphs.py',
    'src/yoked/hierarchical/_matching_gaps.py',
    'src/yoked/hierarchical/_correlated_matching_gap.py',
    'src/yoked/hierarchical/_outer_decoder.py',
    'src/yoked/hierarchical/_outer_mwpm.py',
    'src/yoked/hierarchical/_paper_decoder.py',
    'src/yoked/hierarchical/_paper_stage.py',
    'src/yoked/hierarchical/_collect.py',
    'src/yoked/hierarchical/_provenance.py',
    'tools/hierarchical_experiment',
)


def stage_paper_decode(sample: SampleSet, out_dir, *, rows=None, gap_scale: float = 1.0,
                       batch_size: int = 64) -> dict:
    """Decode with native correlated L1 and raw complementary-gap MWPM L2.

    A fresh output directory is required. Saved samples are referenced with their
    content identities; new samples are saved in ``sample/``. ``manifest.json``
    is published last, after predictions and metrics. This direct configuration
    uses neither legacy L1 records nor a fitted calibration artifact.
    """
    if not isinstance(sample, SampleSet):
        raise TypeError('sample must be a generated or verified SampleSet')
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    rows = np.arange(sample.shots) if rows is None else _positions(rows, 'rows', limit=sample.shots)
    if not len(rows):
        raise ValueError('At least one sample row is required')
    directory = Path(out_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f'{directory} is not an empty output directory; choose a new path')
    start = time.perf_counter()
    decoder = PaperHierarchicalDecoder(stim.DetectorErrorModel(sample.dem_text),
                                      num_patches=sample.parameters.patches, gap_scale=gap_scale)
    directory.mkdir(parents=True, exist_ok=True)
    if sample.source['directory'] is None:
        sample = SampleSet.load(sample.save(directory / 'sample'))
    chunks: dict[str, list[np.ndarray]] = {}
    for offset in range(0, len(rows), batch_size):
        detectors, actual = sample.rows(rows[offset:offset + batch_size])
        result = decoder.decode_with_gaps_batch(detectors)
        for key, array in {**result.arrays(), 'actual': actual}.items():
            chunks.setdefault(key, []).append(array)
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    arrays['row_ids'] = np.asarray(rows, dtype=np.int64)
    failures = np.any(arrays['prediction'] != arrays['actual'], axis=1)
    results = {
        'configuration': PAPER_CONFIGURATION,
        'gap_scale': decoder.gap_scale,
        'shots': len(rows),
        'block_failures': int(failures.sum()),
        'block_failure_rate': float(failures.mean()),
        'reference_block_failures': int(np.any(arrays['reference'] != arrays['actual'], axis=1).sum()),
        'outer_tied_sectors': int(arrays['outer_tied'].sum()),
        'elapsed_seconds': time.perf_counter() - start,
    }
    with atomic_replacement(directory / 'predictions.npz') as temporary:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    write_json_atomic(directory / 'results.json', results)
    provenance = {
        'schema': 'paper-correlated-mwpm/1',
        'configuration': PAPER_CONFIGURATION,
        'gap_scale': decoder.gap_scale,
        'gap_units': 'nats',
        'l1': 'PyMatching native enable_correlations=True',
        'gap': 'forced logical classes on one frozen native integer-weight graph',
        'l2': 'plain PyMatching MWPM with scaled complementary gaps as edge weights',
        'calibration': 'none; direct gap weights',
        'sample': {'identities': dict(sample.identities), 'inputs': sample.identity_inputs(),
                   'source': dict(sample.source)},
        'rows_sha256': sha256_bytes(np.ascontiguousarray(rows, dtype='<i8')),
        'source_sha256': source_hashes(_SOURCES),
        'versions': package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')),
    }
    write_json_atomic(directory / 'manifest.json', {
        **provenance,
        'identity': sha256_bytes(canonical_json(provenance).encode()),
        'code_commit': git_commit(),
        'created_utc': utc_now(),
        'artifacts': {name: sha256_file(directory / name) for name in ('predictions.npz', 'results.json')},
    })
    return results
