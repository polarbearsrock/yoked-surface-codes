"""Independent calibration and paired evaluation of UF confidence candidates."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sinter

from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._outer_mwpm import mwpm_outer_map_batch
from yoked.hierarchical._provenance import (
    REPLAY_SOURCES, CALIBRATION_SOURCES, package_versions, sha256_file, source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import by_sector, load_record, to_columns
from yoked.hierarchical._uf_soft import SCORE_NAMES, TIME_NAMES, quantize_gap
from yoked.hierarchical._uf_soft_collect import FEATURE_SOURCES, VERSIONS, load_features, _write_arrays

ANALYSIS_SOURCES = tuple(sorted(set(FEATURE_SOURCES) | set(REPLAY_SOURCES) | set(CALIBRATION_SOURCES) | {
    'src/yoked/hierarchical/_uf_soft_analysis.py',
}))


def _all_scores(record, features):
    scores = {name: features['scores'][:, k] for k, name in enumerate(SCORE_NAMES)}
    scores['cluster_gap'] = record.cluster_gap
    scores['correlated_uf_gap_4bit'] = quantize_gap(scores['correlated_uf_gap'])
    # Accuracy endpoint reuses the saved expensive scores; no matching confidence is recomputed.
    scores['correlated_matching_gap'] = signed_gaps(
        record.forced_correlated, record.uf_reference.reshape(record.shots, record.num_patches, 2)
    ).reshape(record.shots, -1)
    return scores


def _ler(rate, pieces):
    return sinter.shot_error_rate_to_piece_error_rate(float(rate), pieces=pieces, values=8)


def compare_predictions(predictions, actual, *, pieces, replicates=10_000, seed=43):
    """Whole-shot paired bootstrap, compressed to observed joint failure patterns."""
    names = list(predictions)
    failures = np.stack([np.any(predictions[name] != actual, axis=1) for name in names], axis=1)
    patterns, counts = np.unique(failures, axis=0, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(actual), counts / len(actual), size=replicates)
    boot_rates = (draws @ patterns.astype(float)) / len(actual)
    boot_ler = np.array([[_ler(r, pieces) for r in column] for column in boot_rates.T]).T
    rates = failures.mean(axis=0)
    summaries = {}
    baseline = names.index('joint_correlated_uf') if 'joint_correlated_uf' in names else None
    for k, name in enumerate(names):
        row = dict(failures=int(failures[:, k].sum()), shots=len(actual), block_rate=float(rates[k]),
                   normalized_ler=_ler(rates[k], pieces),
                   ler_ci95=np.percentile(boot_ler[:, k], [2.5, 97.5]).tolist())
        if baseline is not None:
            denominator = _ler(rates[baseline], pieces)
            row['ler_ratio_to_correlated_uf'] = (row['normalized_ler'] / denominator if denominator else None)
            valid = boot_ler[:, baseline] > 0
            row['ratio_undefined_replicates'] = int((~valid).sum())
            row['paired_ratio_ci95'] = (np.percentile(
                boot_ler[valid, k] / boot_ler[valid, baseline], [2.5, 97.5]).tolist() if valid.any() else None)
            row['paired_block_difference_ci95'] = np.percentile(
                boot_rates[:, k] - boot_rates[:, baseline], [2.5, 97.5]).tolist()
        summaries[name] = row
    return summaries


def analyze(calibration_record, evaluation_record, calibration_features, evaluation_features, output_dir):
    """Fit on calibration shots only, then decode the held-out saved evaluation."""
    cal, evaluation = load_record(calibration_record), load_record(evaluation_record)
    if cal.manifest['role'] != 'calibration' or evaluation.manifest['role'] != 'evaluation':
        raise ValueError('Expected calibration and evaluation roles')
    if cal.identities['model'] != evaluation.identities['model']:
        raise ValueError('Calibration and evaluation model mismatch')
    for key in ('parent_sample', 'sampling_family'):
        if cal.identities[key] == evaluation.identities[key]:
            raise ValueError(f'Calibration/evaluation must not share {key}')
    ca, cr = load_features(calibration_features, cal)
    ea, er = load_features(evaluation_features, evaluation)
    for key in ('source_sha256', 'versions', 'gap_cap', 'score_names'):
        if cr[key] != er[key]:
            raise ValueError(f'Calibration/evaluation feature {key} mismatch')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cal_scores, eval_scores = _all_scores(cal.record, ca), _all_scores(evaluation.record, ea)
    errors = cal.record.uf_reference ^ cal.record.actual
    # Fit every map and publish it before accessing evaluation labels for metrics.
    models = {name: [IsotonicCalibrator.fit(scores[:, sector::2], errors[:, sector::2], direction='decreasing')
                     for sector in range(2)] for name, scores in cal_scores.items()}
    write_json_atomic(output / 'calibrators.json', {name: [m.to_json() for m in maps] for name, maps in models.items()})
    bins = np.arange(-8, 8)
    rom = {sector: model.probability(bins) for sector, model in
           zip(('X', 'Z'), models['correlated_uf_gap_4bit'])}
    write_json_atomic(output / 'confidence_rom.json', dict(
        score_bins=bins.tolist(), bin_width_nats=2,
        probabilities={sector: q.tolist() for sector, q in rom.items()},
        log_odds={sector: (np.log1p(-q) - np.log(q)).tolist() for sector, q in rom.items()}))
    reference = by_sector(evaluation.record.uf_reference)
    parity = reference.sum(axis=2) % 2 ^ evaluation.record.yoke
    predictions = dict(evaluation.record.baselines)
    diagnostics, all_q = {}, {}
    for name, scores in eval_scores.items():
        q = np.empty_like(scores)
        for sector, model in enumerate(models[name]):
            q[:, sector::2] = model.probability(scores[:, sector::2])
        sectors = by_sector(q)
        start = time.perf_counter()
        decisions = [mwpm_outer_map_batch(sectors[:, sector], parity[:, sector]) for sector in range(2)]
        flips = np.stack([d.patterns for d in decisions], axis=1)
        final = to_columns(reference ^ flips)
        if not np.array_equal(by_sector(final).sum(axis=2) % 2, evaluation.record.yoke):
            raise ValueError('L2 prediction violates yoke parity')
        predictions[name] = final
        all_q[name] = q
        errors_eval = evaluation.record.uf_reference ^ evaluation.record.actual
        one_error = by_sector(errors_eval).sum(axis=2) == 1
        remaining_error = by_sector(final ^ evaluation.record.actual).any(axis=2)
        diagnostics[name] = dict(above_half_fraction=float((q > .5).mean()),
                                 tied_sector_fraction=float(np.stack([d.tied for d in decisions]).mean()),
                                 brier=float(np.mean((q - errors_eval) ** 2)),
                                 one_error_sectors=int(one_error.sum()),
                                 one_error_misattributions=int((one_error & remaining_error).sum()),
                                 l2_seconds=time.perf_counter() - start)
    params = dict(evaluation.manifest['parameters'])
    summaries = compare_predictions(predictions, evaluation.record.actual,
                                     pieces=params['patches'] * params['rounds'])
    _write_arrays(output / 'predictions.npz', predictions)
    _write_arrays(output / 'probabilities.npz', all_q)
    work = dict(patch_rows=int(ea['rules_fired'].size), reweighted_patch_rows=int(ea['rules_fired'].sum()),
                component_seconds=dict(zip(TIME_NAMES, ea['seconds'].tolist())),
                bounded_states=int(ea['states'][:, :, 0].sum()),
                correlated_bounded_states=int(ea['states'][:, :, 1].sum()),
                full_cluster_states=int(evaluation.record.dijkstra_states.sum()),
                forced_calls_per_weight_model_per_patch=2,
                l1_matching_solves=0)
    result = dict(parameters=params, calibration_shots=cal.record.shots, evaluation_shots=evaluation.record.shots,
                  bootstrap=dict(replicates=10000, seed=43, unit='whole shot', calibration='fixed fitted maps'),
                  methods=summaries, diagnostics=diagnostics, work=work,
                  calibration_record=str(cal.directory), evaluation_record=str(evaluation.directory),
                  calibration_features=str(Path(calibration_features).resolve()),
                  evaluation_features=str(Path(evaluation_features).resolve()),
                  feature_sources=er['source_sha256'], analysis_sources=source_hashes(ANALYSIS_SOURCES),
                  versions=package_versions(VERSIONS))
    write_json_atomic(output / 'results.json', result)
    artifacts = {name: sha256_file(output / name) for name in (
        'calibrators.json', 'confidence_rom.json', 'predictions.npz', 'probabilities.npz', 'results.json')}
    inputs = {str(Path(path).resolve()): sha256_file(path) for path in (
        cal.directory / 'manifest.json', evaluation.directory / 'manifest.json',
        Path(calibration_features) / 'manifest.json', Path(evaluation_features) / 'manifest.json')}
    write_json_atomic(output / 'manifest.json', dict(schema='UFSoftAnalysis/1', inputs=inputs, artifacts=artifacts))
    return result
