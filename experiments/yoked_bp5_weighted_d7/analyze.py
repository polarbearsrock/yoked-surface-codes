"""Correct the fourth row while preserving the three paired baseline results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys

NAMES = ('correlated_mwpm_complementary_gap', 'correlated_mwpm_cluster_gap',
         'correlated_uf_cluster_gap', 'bp5_weighted_uf_cluster_gap')
LABELS = ('Correlated MWPM + complementary gap', 'Correlated MWPM + cluster gap',
          'Correlated UF + cluster gap', 'BP5 + weighted UF + cluster gap')
OLD_BP = 'bp5_correlated_uf_cluster_gap'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    launch = json.loads((output/'launch.json').read_text())
    os.environ['DANTE_REPO'] = launch['snapshot_repository']
    sys.path.insert(0, str(Path(launch['snapshot_repository'])/'src'))
    import numpy as np
    from scipy.stats import binomtest
    import sinter
    import stim
    from weighted import sha, write_json
    from yoked.hierarchical._collect import SampleSet
    from yoked.hierarchical._patch_graphs import PatchGraphs

    source_manifest = json.loads((output/'source_manifest.json').read_text())
    for name, expected in source_manifest.items():
        if sha(output/'source_snapshot'/name) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    completed = json.loads((output/'data/weighted/completion.json').read_text())
    assert completed['shots'] == launch['shots'] == 100000
    assert completed['predictions_sha256'] == sha(output/'data/weighted/predictions.npz')
    assert completed['configuration_sha256'] == sha(output/'data/weighted/configuration.json')
    assert completed['validation_sha256'] == sha(output/'verification.json')
    baseline_manifest = json.loads((output/'baseline/artifact_manifest.json').read_text())
    assert baseline_manifest['data/paired.npz'] == sha(output/'data/baseline_paired.npz')
    for name, expected in baseline_manifest.items():
        if name.startswith('data/sample/'):
            assert sha(output/name) == expected, name
    with np.load(output/'data/baseline_paired.npz') as saved:
        baseline = {k:saved[k] for k in saved.files}
    with np.load(output/'data/weighted/predictions.npz') as saved:
        weighted = {k:saved[k] for k in saved.files}
    sample = SampleSet.load(output/'data/sample')
    shots = sample.shots
    rows = np.arange(shots)
    np.testing.assert_array_equal(baseline['row_ids'], rows)
    np.testing.assert_array_equal(weighted['row_ids'], rows)
    np.testing.assert_array_equal(baseline['variant_names'], [*NAMES[:3], OLD_BP])
    actual = np.unpackbits(sample.actual_packed, axis=1, count=12, bitorder='little').astype(bool)
    np.testing.assert_array_equal(actual, baseline['actual'])
    np.testing.assert_array_equal(actual, weighted['actual'])
    prediction = np.concatenate((baseline['predictions'][:, :3], weighted['prediction'][:, None]), axis=1)
    reference = np.concatenate((baseline['references'][:, :3], weighted['reference'][:, None]), axis=1)
    confidence = np.concatenate((baseline['confidence'][:, :3], weighted['gaps'][:, None]), axis=1)
    assert np.isfinite(confidence).all() and (confidence >= 0).all()
    np.testing.assert_array_equal(weighted['statistics'][:, :, 5:8],
        np.broadcast_to([5, 1, 0], (shots, 6, 3)))
    patches = PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text), num_patches=6)
    yoke = np.column_stack([(sample.detectors_packed[:, k//8] >> (k%8)) & 1
                           for k in patches.yoke_detector_ids]).astype(bool)
    for k in range(4):
        np.testing.assert_array_equal(prediction[:, k].reshape(shots, 6, 2).sum(axis=1)%2, yoke)
    failure = np.any(prediction ^ actual[:, None], axis=2)
    l1_failure = np.any(reference ^ actual[:, None], axis=2)
    np.testing.assert_array_equal(failure[:, :3], baseline['block_failures'][:, :3])
    old_failure = np.any(baseline['predictions'][:, 3] ^ actual, axis=1)
    all_failure = np.column_stack((failure, old_failure))
    patterns, frequencies = np.unique(all_failure, axis=0, return_counts=True)
    seed, replicates = 202609290709, 10000
    draws = np.random.default_rng(seed).multinomial(shots, frequencies/shots, size=replicates)
    rates = draws @ patterns.astype(np.int64)/shots
    normalized = lambda p:sinter.shot_error_rate_to_piece_error_rate(p, pieces=168, values=8)
    result = dict(shots=shots, parameters=sample.parameters.to_json(), seed=sample.seed,
        sample=dict(sample.identities), stim_fork=launch['stim_fork'], variants={}, paired={},
        baseline_experiment=launch['baseline'], unchanged_baselines=list(NAMES[:3]),
        failure_definition='Any of the twelve tracked patch-observable bits is wrong after L2',
        bootstrap=dict(unit='whole paired six-patch shot', seed=seed, replicates=replicates),
        inference='All six corrected pairwise comparisons and the prior BP variant are reported; p-values are exploratory and unadjusted.',
        normalization='sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=168, values=8); effective per physical patch per round, not independently measured patch failures')
    for k, name in enumerate(NAMES):
        fails = int(failure[:, k].sum())
        rate = fails/shots
        interval = binomtest(fails, shots).proportion_ci(method='exact')
        sectors = (prediction[:, k]^actual).reshape(shots, 6, 2).any(axis=1)
        l1_wrong = (reference[:, k]^actual).reshape(shots, 6, 2)
        eligible = l1_wrong.sum(axis=1) == 1
        result['variants'][name] = dict(label=LABELS[k], block_failures=fails,
            block_failure_rate=rate, block_failure_ci95_exact=[interval.low, interval.high],
            effective_ler_per_patch_round=float(normalized(rate)),
            effective_ler_ci95=[float(normalized(interval.low)), float(normalized(interval.high))],
            l1_block_failures=int(l1_failure[:, k].sum()), l1_failed_patch_sectors=int(l1_wrong.sum()),
            final_sector_failures=sectors.sum(axis=0).astype(int).tolist(),
            single_error_misattribution=dict(eligible_sector_cases=int(eligible.sum()),
                failed_sector_cases=int((eligible & sectors).sum())),
            l2_repairs=int((l1_failure[:, k] & ~failure[:, k]).sum()),
            l2_regressions=int((~l1_failure[:, k] & failure[:, k]).sum()))
    original = json.loads((output/'baseline/results.json').read_text())
    for name in NAMES[:3]:
        assert result['variants'][name] == original['variants'][name], name

    def paired(a, b, baseline_name, candidate_name):
        repairs = int((all_failure[:, a] & ~all_failure[:, b]).sum())
        regressions = int((~all_failure[:, a] & all_failure[:, b]).sum())
        return dict(baseline=baseline_name, candidate=candidate_name, repairs=repairs,
            regressions=regressions, both_fail=int((all_failure[:, a] & all_failure[:, b]).sum()),
            both_succeed=int((~all_failure[:, a] & ~all_failure[:, b]).sum()),
            block_failure_difference=float(all_failure[:, b].mean()-all_failure[:, a].mean()),
            difference_ci95=np.percentile(rates[:, b]-rates[:, a], [2.5, 97.5]).tolist(),
            relative_change=float(all_failure[:, b].mean()/all_failure[:, a].mean()-1),
            mcnemar_exact_p=float(binomtest(regressions, repairs+regressions).pvalue) if repairs+regressions else 1.)
    for a, b in itertools.combinations(range(4), 2):
        result['paired'][NAMES[b]+'__versus__'+NAMES[a]] = paired(a, b, NAMES[a], NAMES[b])
    result['versus_previous_bp_correlated'] = paired(4, 3, OLD_BP, NAMES[3])
    result['execution'] = dict(openmp_threads=32, cpu_affinity=launch['cpu_affinity'],
        collection_seconds=completed['elapsed_seconds_this_collection'],
        native_and_packing_seconds=completed['native_and_packing_seconds'],
        outer_seconds=completed['outer_seconds'], reused_baseline_shots=shots,
        newly_decoded_variants=[NAMES[3]],
        timing_note='Collection wall time including all six patches; not a single-shot latency benchmark.')
    np.savez_compressed(output/'data/paired.npz', row_ids=rows, actual=actual,
        predictions=prediction, references=reference, confidence=confidence,
        block_failures=failure, l1_block_failures=l1_failure, variant_names=np.array(NAMES))
    write_json(output/'results.json', result)
    lines = ['# Corrected four-way yoked surface-code comparison', '',
        '100,000 identical paired shots; six d=7 patches, two ideal yokes, 28 noisy CZ rounds, '
        'SI1000 p=0.003, ideal time boundaries, and plain MWPM at L2.', '',
        f'The first three configurations reuse the completed results in `{launch["baseline"]}`. '
        'Only BP5 + weighted UF + cluster gap was newly decoded. The previous experiment remains preserved.', '',
        f'Stim {launch["stim_fork"]["version"]}, fork commit `{launch["stim_fork"]["source_commit"]}`; '
        f'sample seed `{sample.seed}`.', '',
        '| Configuration | Block failures | Block LER | Exact 95% CI | Effective LER / patch / round |',
        '|---|---:|---:|---:|---:|']
    for name in NAMES:
        r = result['variants'][name]
        low, high = r['block_failure_ci95_exact']
        lines.append(f'| {r["label"]} | {r["block_failures"]:,} / {shots:,} | '
            f'{100*r["block_failure_rate"]:.3f}% | {100*low:.3f}%–{100*high:.3f}% | '
            f'{r["effective_ler_per_patch_round"]:.6g} |')
    lines += ['', 'A block failure means any of the twelve tracked patch observables is wrong after L2. '
        'The last column uses the existing Sinter normalization (pieces=168, values=8); '
        'it is an effective normalization, not an independently measured patch failure rate.', '',
        '| Candidate versus baseline | Repairs | Regressions | Difference in percentage points (paired 95% CI) | Relative change |',
        '|---|---:|---:|---:|---:|']
    for r in result['paired'].values():
        a, b = NAMES.index(r['baseline']), NAMES.index(r['candidate'])
        low, high = r['difference_ci95']
        lines.append(f'| {LABELS[b]} vs {LABELS[a]} | {r["repairs"]:,} | {r["regressions"]:,} | '
            f'{100*r["block_failure_difference"]:+.3f} ({100*low:+.3f}, {100*high:+.3f}) | '
            f'{100*r["relative_change"]:+.2f}% |')
    r = result['versus_previous_bp_correlated']
    low, high = r['difference_ci95']
    lines += ['', 'Negative differences favor the candidate. Intervals use 10,000 paired whole-shot '
        'bootstrap replicates; all six comparisons are reported. Exploratory unadjusted McNemar '
        'p-values are in results.json.', '',
        f'Compared with the previous BP5 + correlated UF result ({int(old_failure.sum()):,} failures), '
        f'the corrected weighted-UF variant changes block LER by {100*r["block_failure_difference"]:+.3f} '
        f'percentage points (paired 95% CI {100*low:+.3f}, {100*high:+.3f}), '
        f'with {r["repairs"]:,} repairs and {r["regressions"]:,} regressions.', '',
        '## Decoder and validation', '',
        'The corrected variant runs exactly five sum-product flooding BP iterations (damping 0.5, '
        'LLR clip 30). The existing projection sets edge weights to '
        '`-log(clip(sum of supporting fault posteriors, 1e-15, 1))`. It then runs weighted UF once '
        'on the original syndrome and obtains the full uncapped cluster gap from that final growth state. '
        'There are no subsequent correlation discounts or second UF pass. Confidence calibration and '
        'evaluation-set tuning are absent. MWPM cluster gap remains the existing MPP matching-radius score.', '',
        'The single-pass correction matches the existing native BP5 + weighted UF implementation. '
        'Independent Python BP, weighted UF and gap oracles verified all 16 synthetic syndromes and '
        'eight fresh d=7 cases. One-versus-32-thread outputs agree on 16 fresh shots for two patches. '
        'All 600,000 collected patch decodes use exactly BP5 and one UF pass, and their corrections '
        'are checked against the physical syndrome. All final predictions satisfy the ideal yoke '
        'checks. The first three configurations match every previous prediction and metric.', '',
        '## Execution and reproduction', '',
        f'The corrected variant used 32 verified OpenMP threads on the same 32 physical cores; '
        f'collection took {completed["elapsed_seconds_this_collection"]:.1f} seconds. '
        'The existing baselines used 32 MWPM worker processes and 32 UF OpenMP threads as recorded '
        'in baseline/. Collection wall time is not a single-shot latency benchmark.', '',
        'The full sample, baseline predictions, corrected predictions and combined paired arrays '
        'are retained under data/. Source snapshots, manifests, native build commands, verification '
        'results and original baseline provenance are retained alongside this report. '
        'See protocol.md for reproduction commands.', '']
    (output/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, ax = plt.subplots(figsize=(8.8, 3.4), constrained_layout=True)
    for k, name in enumerate(NAMES):
        r = result['variants'][name]
        value = 100*r['block_failure_rate']
        low, high = 100*np.array(r['block_failure_ci95_exact'])
        ax.errorbar(value, k, xerr=[[value-low], [high-value]], fmt='o', capsize=4,
            color=('#304B78', '#597CA4', '#BC7944', '#387B66')[k], markersize=7)
        ax.annotate(f'{value:.3f}%', (high, k), xytext=(7, 0), textcoords='offset points', va='center')
    ax.set_yticks(range(4), ('cMWPM + complementary gap', 'cMWPM + cluster gap',
                           'cUF + cluster gap', 'BP5 + weighted UF + cluster gap'))
    ax.invert_yaxis()
    ax.set_xlabel('Final block logical error rate (%)')
    ax.set_title('Six yoked d=7 patches · SI1000 p=0.003 · 28 rounds\n'
                 '100,000 paired shots · exact 95% binomial intervals', fontsize=11)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=.25, y=.2)
    fig.savefig(output/'comparison.png', dpi=180)
    fig.savefig(output/'comparison.svg')
    plt.close(fig)
    files = [p for p in output.rglob('*') if p.is_file()
             and p not in (output/'artifact_manifest.json', output/'progress.json')]
    write_json(output/'artifact_manifest.json', {str(p.relative_to(output)):sha(p) for p in sorted(files)})
    write_json(output/'progress.json', dict(state='complete', shots=shots,
        updated_utc=datetime.now(timezone.utc).isoformat(), report=str(output/'report.md'),
        results_sha256=sha(output/'results.json')))
    print(json.dumps(dict(variants=result['variants'], paired=result['paired'],
        versus_previous_bp_correlated=result['versus_previous_bp_correlated']), indent=2), flush=True)
    print(f'Retained report: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
