"""Paired accuracy comparison: complementary gaps versus MPP cluster scores.

Both sides see the same local syndromes, use the same correlated-MWPM reference
bits, and call the same plain-MWPM outer decoder. Truth is used only for metrics.
No parameters or calibration maps are fitted on these evaluation shots.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import stim

from yoked.hierarchical._collect import SampleSet, _positions
from yoked.hierarchical._metrics import (
    block_failures, failure_by_stratum, misattribution, residual_failure_counts, sector_failures,
)
from yoked.hierarchical._mpp import MPP_CONFIGURATION, MppHierarchicalDecoder
from yoked.hierarchical._paper_decoder import PAPER_CONFIGURATION, PaperHierarchicalDecoder
from yoked.hierarchical._paper_stage import _SOURCES as PAPER_SOURCES
from yoked.hierarchical._provenance import (
    atomic_replacement, canonical_json, git_commit, package_versions, sha256_bytes,
    sha256_file, source_hashes, utc_now, write_json_atomic,
)

SOURCES = (*PAPER_SOURCES, 'src/yoked/hierarchical/_mpp.py',
           'src/yoked/hierarchical/_mpp_experiment.py', 'src/yoked/hierarchical/_metrics.py',
           'native/mpp/mpp_fast.cc', 'native/mpp/CMakeLists.txt', 'tools/build_mpp', 'tools/mpp_experiment')


def compare_accuracy(reference, actual, gap_prediction, mpp_prediction, *, replicates=10000, seed=43):
    """Count failures and resample paired whole shots, including both sectors.

    Compress identical outcome rows before multinomial resampling. This is the
    same empirical bootstrap as sampling individual shot indices, with bounded
    memory even for large runs. It preserves all dependence between methods
    and between a shot's sectors. Intervals are percentile 95% intervals.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError('replicates must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    counts = residual_failure_counts(reference, actual)
    eligible = counts == 1
    blocks, wrong_sectors, metrics = [], [], {}
    for name, prediction in (('complementary_gap', gap_prediction), ('mpp', mpp_prediction)):
        block = block_failures(prediction, actual)
        wrong = sector_failures(prediction, actual)
        blocks.append(block)
        wrong_sectors.append(wrong)
        metrics[name] = {
            'block_failures': int(block.sum()), 'block_failure_rate': float(block.mean()),
            'single_error_misattribution': {k: v.to_json() for k, v in misattribution(wrong, counts).items()},
            'failure_by_l1_error_count': {
                k: {s: rate.to_json() for s, rate in rates.items()}
                for k, rates in failure_by_stratum(wrong, counts).items()},
        }
    columns = np.column_stack((blocks[0], blocks[1],
                               (wrong_sectors[0] & eligible).sum(axis=1),
                               (wrong_sectors[1] & eligible).sum(axis=1), eligible.sum(axis=1)))
    patterns, frequencies = np.unique(columns, axis=0, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(actual), frequencies / len(actual), size=replicates)
    totals = draws @ patterns
    block_difference = (totals[:, 1] - totals[:, 0]) / len(actual)
    usable = totals[:, 4] > 0
    mis_difference = (totals[usable, 3] - totals[usable, 2]) / totals[usable, 4]
    return {
        'shots': len(actual), **metrics,
        'paired': {
            'difference_direction': 'mpp minus complementary_gap; positive is worse',
            'block_failure_difference': float(blocks[1].mean() - blocks[0].mean()),
            'block_failure_difference_ci95': np.percentile(block_difference, [2.5, 97.5]).tolist(),
            'mpp_repairs': int((blocks[0] & ~blocks[1]).sum()),
            'mpp_regressions': int((~blocks[0] & blocks[1]).sum()),
            'both_fail': int((blocks[0] & blocks[1]).sum()),
            'both_succeed': int((~blocks[0] & ~blocks[1]).sum()),
            'pooled_misattribution_difference': (
                float(((wrong_sectors[1] & eligible).sum() - (wrong_sectors[0] & eligible).sum())
                      / eligible.sum()) if eligible.any() else None),
            'pooled_misattribution_difference_ci95': (
                np.percentile(mis_difference, [2.5, 97.5]).tolist() if usable.any() else None),
            'zero_eligible_bootstrap_replicates': int((~usable).sum()),
        },
        'bootstrap': {'replicates': int(replicates), 'seed': int(seed), 'unit': 'whole paired shot'},
    }


def stage_mpp_compare(sample: SampleSet, out_dir, *, rows=None, method='dijkstra',
                      batch_size=64, verify_scores=False, replicates=10000, seed=43,
                      progress=None):
    """Publish predictions, paired metrics, sample identity, and native build provenance.

    A fresh output directory is required. Each batch checks identical reference
    bits and native matching weights before it can contribute to a report.
    ``verify_scores`` additionally checks exact integer score agreement with
    the slow C++ oracle; omit it for production evaluation after validation.
    """
    if not isinstance(sample, SampleSet):
        raise TypeError('sample must be a verified or generated SampleSet')
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    rows = np.arange(sample.shots) if rows is None else _positions(rows, 'rows', limit=sample.shots)
    if not len(rows):
        raise ValueError('At least one sample row is required')
    directory = Path(out_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f'{directory} is not an empty output directory')
    start = time.perf_counter()
    dem = stim.DetectorErrorModel(sample.dem_text)
    baseline = PaperHierarchicalDecoder(dem, num_patches=sample.parameters.patches)
    candidate = MppHierarchicalDecoder(dem, num_patches=sample.parameters.patches,
                                      method=method, verify=verify_scores)
    directory.mkdir(parents=True, exist_ok=True)
    if sample.source['directory'] is None:
        sample = SampleSet.load(sample.save(directory / 'sample'))
    chunks: dict[str, list[np.ndarray]] = {}
    for offset in range(0, len(rows), batch_size):
        detectors, actual = sample.rows(rows[offset:offset + batch_size])
        gap = baseline.decode_with_gaps_batch(detectors)
        mpp = candidate.decode_with_scores_batch(detectors)
        if not np.array_equal(gap.reference, mpp.reference):
            raise ValueError('MPP changed L1 reference predictions; confidence-only comparison rejected')
        # Compare the chosen logical class cost as well as its label. A score
        # match cannot hide a different graph normalization or reweighting rule.
        chosen = gap.reference.reshape(len(detectors), -1, 2).astype(int)
        weights = gap.forced_weights[np.arange(len(detectors))[:, None],
                                     np.arange(sample.parameters.patches)[None, :],
                                     chosen[:, :, 0], chosen[:, :, 1]]
        if not np.allclose(weights, mpp.native_weight, atol=1e-10, rtol=0):
            raise ValueError('MPP native matching weights disagree with baseline')
        values = {'actual': actual, 'reference': gap.reference,
                  'complementary_gap': gap.complementary_gap, 'forced_weights': gap.forced_weights,
                  'cluster_score': mpp.cluster_score, 'native_weight': mpp.native_weight, 'sigma': gap.sigma,
                  'gap_prediction': gap.prediction, 'mpp_prediction': mpp.prediction,
                  'gap_residual': gap.residual, 'mpp_residual': mpp.residual,
                  'gap_outer_tied': gap.outer_tied, 'mpp_outer_tied': mpp.outer_tied}
        for key, array in values.items():
            chunks.setdefault(key, []).append(array)
        if progress is not None:
            progress(min(offset + batch_size, len(rows)), len(rows))
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    arrays['row_ids'] = np.asarray(rows, dtype=np.int64)
    results = compare_accuracy(arrays['reference'], arrays['actual'], arrays['gap_prediction'],
                               arrays['mpp_prediction'], replicates=replicates, seed=seed)
    results.update(method=method, reference_disagreements=0, native_weight_disagreements=0,
                   score_oracle_checked=bool(verify_scores), elapsed_seconds=time.perf_counter() - start)
    for name in ('gap', 'mpp'):
        results['complementary_gap' if name == 'gap' else name]['outer_tied_sectors'] = int(
            arrays[f'{name}_outer_tied'].sum())
    with atomic_replacement(directory / 'predictions.npz') as temporary:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    write_json_atomic(directory / 'results.json', results)
    provenance = {
        'schema': 'mpp-confidence-comparison/1',
        'configurations': {'complementary_gap': PAPER_CONFIGURATION, 'mpp': MPP_CONFIGURATION},
        'l1': 'PyMatching 2.4.0 native correlated MWPM; all correlations; identical predictions required',
        'l2': 'plain PyMatching MWPM; identical tie convention',
        'confidence': 'direct complementary gaps versus direct MPP cluster scores; no calibration',
        'method': method, 'score_oracle_checked': bool(verify_scores),
        'sample': {'identities': dict(sample.identities), 'inputs': sample.identity_inputs(),
                   'source': dict(sample.source)},
        'rows_sha256': sha256_bytes(np.ascontiguousarray(rows, dtype='<i8')),
        'source_sha256': source_hashes(SOURCES), 'native_build': candidate.build_metadata,
        'versions': package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')),
        'bootstrap': results['bootstrap'],
    }
    write_json_atomic(directory / 'manifest.json', {
        **provenance, 'identity': sha256_bytes(canonical_json(provenance).encode()),
        'code_commit': git_commit(), 'created_utc': utc_now(),
        'artifacts': {name: sha256_file(directory / name) for name in ('predictions.npz', 'results.json')},
    })
    return results
