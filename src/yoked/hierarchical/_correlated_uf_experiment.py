"""Collect correlated-UF confidence and compare against verified saved MWPM runs.

The L1 decoder changes in this experiment. Each hierarchy therefore owns its
reference bits; only final block outcomes are paired across configurations.
Truth labels are used for metrics and artifact checks, never for decoding.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import binomtest
import stim

from yoked.hierarchical._collect import SampleSet, _positions
from yoked.hierarchical._correlated_uf import CORRELATED_UF_CONFIGURATION, CorrelatedUFHierarchicalDecoder
from yoked.hierarchical._provenance import (
    atomic_replacement, canonical_json, git_commit, package_versions, sha256_bytes,
    sha256_file, source_hashes, utc_now, write_json_atomic,
)

SOURCES = (
    'src/yoked/decoders/_graph.py', 'src/yoked/decoders/_union_find.py',
    'src/yoked/decoders/_correlated_union_find.py', 'src/yoked/decoders/_correlations.py',
    'src/yoked/hierarchical/_arrays.py', 'src/yoked/hierarchical/_patch_graphs.py',
    'src/yoked/hierarchical/_cluster_gap.py', 'src/yoked/hierarchical/_outer_decoder.py',
    'src/yoked/hierarchical/_outer_mwpm.py', 'src/yoked/hierarchical/_correlated_uf.py',
    'src/yoked/hierarchical/_correlated_uf_experiment.py',
    'src/yoked/hierarchical/_collect.py', 'src/yoked/hierarchical/_provenance.py',
    'tools/correlated_uf_experiment',
)
PREDICTIONS = {
    'correlated_uf_cluster_gap': 'prediction',
    'correlated_mwpm_complementary_gap': 'gap_prediction',
    'correlated_mwpm_mpp': 'mpp_prediction',
}


def load_baseline(sample: SampleSet, directory):
    """Verify a completed distance from the earlier sharded MWPM comparison.

    Full sample coverage is required here. New experiments may select any
    subset of those rows, but matching by row number alone is insufficient:
    the model, sample identity, saved truth, and artifact hashes must agree.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != 'mpp-sharded-comparison/1':
        raise ValueError('Expected a completed sharded MPP comparison distance')
    if manifest['sample_identities'] != dict(sample.identities):
        raise ValueError('Baseline and candidate must use the identical saved sample')
    for name in ('predictions.npz', 'results.json'):
        if sha256_file(directory / name) != manifest['artifacts'][name]:
            raise ValueError(f'Baseline artifact hash mismatch: {name}')
    settings = json.loads((directory / 'results.json').read_text())
    if settings['parameters'] != sample.parameters.to_json():
        raise ValueError('Baseline circuit parameters disagree with the sample')
    with np.load(directory / 'predictions.npz') as stored:
        arrays = {name: stored[name] for name in
                  ('row_ids', 'actual', 'reference', 'gap_prediction', 'mpp_prediction')}
    np.testing.assert_array_equal(arrays.pop('row_ids'), np.arange(sample.shots))
    for name, array in arrays.items():
        if array.shape != (sample.shots, sample.num_observables) or not np.isin(array, (0, 1)).all():
            raise ValueError(f'Invalid baseline array: {name}')
    metadata = {'directory': str(directory), 'manifest_sha256': sha256_file(directory / 'manifest.json'),
                'artifacts': manifest['artifacts'], 'provenance': manifest['provenance']}
    return arrays, metadata


def summarize_accuracy(arrays, *, replicates=10000, seed=43):
    """Report whole-block LER and paired differences without sharing L1 strata."""
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError('replicates must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    actual = arrays['actual']
    shots = len(actual)
    if not shots:
        raise ValueError('At least one shot is required')
    metrics, failures = {}, {}
    for name, key in PREDICTIONS.items():
        if key not in arrays:
            continue
        failed = np.any(arrays[key] != actual, axis=1)
        failures[name] = failed
        interval = binomtest(int(failed.sum()), shots).proportion_ci(method='exact')
        reference = arrays['reference' if key == 'prediction' else 'mwpm_reference']
        metrics[name] = {
            'block_failures': int(failed.sum()), 'block_failure_rate': float(failed.mean()),
            'block_failure_ci95_exact': [float(interval.low), float(interval.high)],
            'l1_block_failures': int(np.any(reference != actual, axis=1).sum()),
        }

    candidate = failures['correlated_uf_cluster_gap']
    comparisons = {}
    for name, baseline in failures.items():
        if name == 'correlated_uf_cluster_gap':
            continue
        repairs = int((baseline & ~candidate).sum())
        regressions = int((~baseline & candidate).sum())
        cells = [int((~baseline & ~candidate).sum()), repairs, regressions,
                 int((baseline & candidate).sum())]
        draws = np.random.default_rng(seed).multinomial(shots, np.array(cells) / shots, size=replicates)
        base_counts, candidate_counts = draws[:, 1] + draws[:, 3], draws[:, 2] + draws[:, 3]
        usable = base_counts > 0
        difference = (candidate_counts - base_counts) / shots
        ratio = float(candidate.sum() / baseline.sum()) if baseline.any() else None
        comparisons[name] = {
            'difference_direction': 'correlated UF minus this MWPM baseline; positive is worse',
            'block_failure_difference': float(candidate.mean() - baseline.mean()),
            'difference_ci95_paired_bootstrap': np.percentile(difference, [2.5, 97.5]).tolist(),
            'candidate_to_baseline_ratio': ratio,
            'relative_increase_percent': None if ratio is None else 100 * (ratio - 1),
            'ratio_ci95_paired_bootstrap': (
                np.percentile(candidate_counts[usable] / base_counts[usable], [2.5, 97.5]).tolist()
                if usable.any() else None),
            'zero_baseline_bootstrap_replicates': int((~usable).sum()),
            'candidate_repairs': repairs, 'candidate_regressions': regressions,
            'both_fail': cells[3], 'both_succeed': cells[0],
            'mcnemar_exact_two_sided_p': (
                float(binomtest(regressions, repairs + regressions, 0.5).pvalue)
                if repairs + regressions else 1.0),
            'l1_reference_bit_disagreements': int(np.count_nonzero(
                arrays['reference'] != arrays['mwpm_reference'])),
        }
    return {
        'shots': shots, 'configurations': metrics, 'paired_against': comparisons,
        'candidate_outer_tied_sectors': int(arrays['outer_tied'].sum()),
        'candidate_dijkstra_states': int(arrays['dijkstra_states'].sum()),
        'bootstrap': {'replicates': int(replicates), 'seed': int(seed), 'unit': 'whole paired shot'},
    }


def stage_correlated_uf(sample: SampleSet, out_dir, *, baseline_run=None, rows=None,
                        batch_size=64, replicates=10000, seed=43, progress=None):
    """Decode selected rows and publish a manifest only after all checks pass."""
    if not isinstance(sample, SampleSet):
        raise TypeError('sample must be a verified SampleSet')
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    rows = np.arange(sample.shots) if rows is None else _positions(rows, 'rows', limit=sample.shots)
    if not len(rows):
        raise ValueError('At least one sample row is required')
    directory = Path(out_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f'{directory} is not an empty output directory')
    sources = source_hashes(SOURCES)
    baseline, baseline_metadata = (None, None) if baseline_run is None else load_baseline(sample, baseline_run)
    decoder = CorrelatedUFHierarchicalDecoder(stim.DetectorErrorModel(sample.dem_text),
                                            num_patches=sample.parameters.patches)
    directory.mkdir(parents=True, exist_ok=True)
    if sample.source['directory'] is None:
        sample = SampleSet.load(sample.save(directory / 'sample'))
    started = time.perf_counter()
    chunks = {}
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset:offset + batch_size]
        detectors, actual = sample.rows(selected)
        result = decoder.decode_with_gaps_batch(detectors)
        values = {**result.arrays(), 'actual': actual}
        if baseline is not None:
            np.testing.assert_array_equal(baseline['actual'][selected], actual)
            values['mwpm_reference'] = baseline['reference'][selected]
            for name in ('gap_prediction', 'mpp_prediction'):
                prediction = baseline[name][selected]
                parity = prediction.reshape(len(selected), sample.parameters.patches, 2).sum(axis=1) % 2
                np.testing.assert_array_equal(parity, detectors[:, decoder.patches.yoke_detector_ids])
                values[name] = prediction
        for name, value in values.items():
            chunks.setdefault(name, []).append(value)
        if progress is not None:
            progress(min(offset + batch_size, len(rows)), len(rows))
    arrays = {name: np.concatenate(values) for name, values in chunks.items()}
    arrays['row_ids'] = np.asarray(rows, dtype=np.int64)
    results = summarize_accuracy(arrays, replicates=replicates, seed=seed)
    results.update(parameters=sample.parameters.to_json(), elapsed_seconds=time.perf_counter() - started)
    if source_hashes(SOURCES) != sources:
        raise ValueError('Decoder sources changed during collection')
    with atomic_replacement(directory / 'predictions.npz') as temporary:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    write_json_atomic(directory / 'results.json', results)
    provenance = {
        'schema': 'correlated-uf-cluster-gap/1', 'configuration': CORRELATED_UF_CONFIGURATION,
        'l1_reference': 'final correlated-UF correction; first UF pass only conditions weights',
        'confidence': 'full unbounded cluster gap on final UF growth state; raw L2 weights',
        'l2': 'plain PyMatching MWPM; same solver and tie convention as saved baselines',
        'calibration': 'none', 'baseline': baseline_metadata,
        'sample': {'identities': dict(sample.identities), 'inputs': sample.identity_inputs(),
                   'source': dict(sample.source)},
        'rows_sha256': sha256_bytes(np.ascontiguousarray(rows, dtype='<i8')),
        'source_sha256': sources,
        'versions': package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')),
        'bootstrap': results['bootstrap'],
    }
    write_json_atomic(directory / 'manifest.json', {
        **provenance, 'identity': sha256_bytes(canonical_json(provenance).encode()),
        'code_commit': git_commit(), 'created_utc': utc_now(),
        'artifacts': {name: sha256_file(directory / name) for name in ('predictions.npz', 'results.json')},
    })
    return results
