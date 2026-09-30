"""Retain paired four-way results and report uncertainty and provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

NAMES = ('correlated_mwpm_complementary_gap', 'correlated_mwpm_cluster_gap',
         'correlated_uf_cluster_gap', 'bp5_correlated_uf_cluster_gap')
LABELS = ('Correlated MWPM + complementary gap', 'Correlated MWPM + cluster gap',
          'Correlated UF + cluster gap', 'BP5 + correlated UF + cluster gap')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    launch = json.loads((output/'launch.json').read_text())
    work = Path(launch['work'])
    os.environ['DANTE_REPO'] = launch['snapshot_repository']
    sys.path.insert(0, str(Path(launch['snapshot_repository'])/'src'))
    import numpy as np
    from scipy.stats import binomtest
    import sinter
    import stim
    from native import sha, write_json
    from yoked.hierarchical._collect import SampleSet
    from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
    from yoked.hierarchical._patch_graphs import PatchGraphs

    mwpm_completion = json.loads((work/'mwpm/completion.json').read_text())
    uf_completion = json.loads((work/'uf/completion.json').read_text())
    if uf_completion['predictions_sha256'] != sha(work/'uf/predictions.npz'):
        raise ValueError('UF prediction checksum mismatch')
    mwpm_manifest = json.loads((work/'mwpm/d7/manifest.json').read_text())
    if mwpm_completion['distance_manifests']['7'] != sha(work/'mwpm/d7/manifest.json'):
        raise ValueError('MWPM completion manifest mismatch')
    for name, expected in mwpm_manifest['artifacts'].items():
        if sha(work/'mwpm/d7'/name) != expected:
            raise ValueError(f'MWPM artifact checksum mismatch: {name}')
    with np.load(work/'mwpm/d7/predictions.npz') as saved:
        mwpm = {name:saved[name] for name in saved.files}
    with np.load(work/'uf/predictions.npz') as saved:
        uf = {name:saved[name] for name in saved.files}
    sample = SampleSet.load(work/'mwpm/d7/sample')
    shots = sample.shots
    rows = np.arange(shots)
    np.testing.assert_array_equal(mwpm['row_ids'], rows)
    np.testing.assert_array_equal(uf['row_ids'], rows)
    np.testing.assert_array_equal(mwpm['actual'], uf['actual'])
    actual = np.unpackbits(sample.actual_packed, axis=1, count=12, bitorder='little').astype(bool)
    np.testing.assert_array_equal(actual, mwpm['actual'])
    prediction = np.stack((mwpm['gap_prediction'], mwpm['mpp_prediction'],
                           uf['prediction'][:, 0], uf['prediction'][:, 1]), axis=1)
    reference = np.stack((mwpm['reference'], mwpm['reference'],
                          uf['reference'][:, 0], uf['reference'][:, 1]), axis=1)
    confidence = np.stack((mwpm['complementary_gap'], mwpm['cluster_score'],
                           uf['gaps'][:, 0], uf['gaps'][:, 1]), axis=1)
    patches = PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text), num_patches=6)
    # Only unpack the two ideal yoke columns from the bit-packed sample.
    yoke = np.column_stack([(sample.detectors_packed[:, k//8] >> (k%8)) & 1
                            for k in patches.yoke_detector_ids]).astype(bool)
    for variant in range(4):
        np.testing.assert_array_equal(prediction[:, variant].reshape(shots, 6, 2).sum(axis=1)%2, yoke)
    assert np.isfinite(confidence).all() and (confidence >= 0).all()
    failure = np.any(prediction ^ actual[:, None, :], axis=2)
    l1_failure = np.any(reference ^ actual[:, None, :], axis=2)
    patterns, frequencies = np.unique(failure, axis=0, return_counts=True)
    bootstrap_seed, replicates = 202609290709, 10000
    draws = np.random.default_rng(bootstrap_seed).multinomial(shots, frequencies/shots, size=replicates)
    rates = (draws @ patterns.astype(np.int64))/shots
    normalized = lambda rate:sinter.shot_error_rate_to_piece_error_rate(rate, pieces=6*28, values=8)
    result = dict(shots=shots, parameters=sample.parameters.to_json(), seed=sample.seed,
        sample=dict(sample.identities), stim_fork=launch['stim_fork'], variants={}, paired={},
        failure_definition='Any of the twelve jointly tracked patch-observable bits is wrong after L2',
        bootstrap=dict(unit='whole paired six-patch shot', seed=bootstrap_seed, replicates=replicates),
        inference='All six pairwise comparisons are reported; p-values are exploratory and unadjusted.',
        normalization='sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=168, values=8); effective per physical patch per round; not independently measured patch failures')
    for index, name in enumerate(NAMES):
        fails = int(failure[:, index].sum())
        rate = fails/shots
        interval = binomtest(fails, shots).proportion_ci(method='exact')
        sector_failure = (prediction[:, index]^actual).reshape(shots, 6, 2).any(axis=1)
        l1_wrong = (reference[:, index]^actual).reshape(shots, 6, 2)
        eligible = l1_wrong.sum(axis=1) == 1
        result['variants'][name] = dict(label=LABELS[index], block_failures=fails,
            block_failure_rate=rate, block_failure_ci95_exact=[float(interval.low), float(interval.high)],
            effective_ler_per_patch_round=float(normalized(rate)),
            effective_ler_ci95=[float(normalized(interval.low)),float(normalized(interval.high))],
            l1_block_failures=int(l1_failure[:, index].sum()),
            l1_failed_patch_sectors=int(l1_wrong.sum()),
            final_sector_failures=sector_failure.sum(axis=0).astype(int).tolist(),
            single_error_misattribution=dict(eligible_sector_cases=int(eligible.sum()),
                failed_sector_cases=int((eligible & sector_failure).sum())),
            l2_repairs=int((l1_failure[:, index] & ~failure[:, index]).sum()),
            l2_regressions=int((~l1_failure[:, index] & failure[:, index]).sum()))
    for a, b in itertools.combinations(range(4), 2):
        repairs = int((failure[:, a] & ~failure[:, b]).sum())
        regressions = int((~failure[:, a] & failure[:, b]).sum())
        difference = failure[:, b].mean()-failure[:, a].mean()
        key = NAMES[b]+'__versus__'+NAMES[a]
        result['paired'][key] = dict(baseline=NAMES[a], candidate=NAMES[b], repairs=repairs,
            regressions=regressions, both_fail=int((failure[:, a] & failure[:, b]).sum()),
            both_succeed=int((~failure[:, a] & ~failure[:, b]).sum()),
            block_failure_difference=float(difference),
            difference_ci95=np.percentile(rates[:, b]-rates[:, a],[2.5,97.5]).tolist(),
            relative_change=float(failure[:, b].mean()/failure[:, a].mean()-1),
            mcnemar_exact_p=float(binomtest(regressions, repairs+regressions).pvalue) if repairs+regressions else 1.)
    shard_meta = [json.loads(path.read_text()) for path in sorted((work/'uf/shards').glob('*.json'))]
    result['execution'] = dict(cpu_affinity=launch['cpu_affinity'],
        mwpm_worker_processes=32, uf_bp_openmp_threads=32,
        mwpm_phase_seconds=mwpm_completion['elapsed_seconds'],
        uf_native_and_packing_seconds=sum(x['native_seconds'] for x in shard_meta),
        uf_outer_collection_seconds=sum(x['outer_seconds'] for x in shard_meta),
        uf_phase_seconds=uf_completion['elapsed_seconds_this_collection'],
        timing_note='Collection wall times; not single-shot latency or FPGA estimates. Includes validation checks where stated.')
    # Retain exact samples, all prediction arrays, confidence values, and checks.
    retained = output/'data'
    retained.mkdir(exist_ok=True)
    shutil.copytree(work/'mwpm/d7/sample', retained/'sample', dirs_exist_ok=True)
    for subdir, paths in {
        'mwpm': ['d7/predictions.npz','d7/results.json','d7/manifest.json','run.json','completion.json'],
        'uf': ['predictions.npz','configuration.json','completion.json'],
    }.items():
        dest = retained/subdir
        dest.mkdir(exist_ok=True)
        for path in paths:
            shutil.copyfile(work/subdir/path, dest/Path(path).name)
    np.savez_compressed(retained/'paired.npz', row_ids=rows, actual=actual,
        predictions=prediction, references=reference, confidence=confidence,
        block_failures=failure, l1_block_failures=l1_failure, variant_names=np.array(NAMES))
    source_root = output/'source_snapshot'
    source_manifest = json.loads((output/'source_manifest.json').read_text())
    for source in Path(__file__).resolve().parent.iterdir():
        if source.suffix not in ('.py', '.cc', '.md'):
            continue
        relative = Path('experiments/yoked_fourway_d7')/source.name
        destination = source_root/relative
        if source != destination:
            shutil.copyfile(source, destination)
        source_manifest[str(relative)] = sha(destination)
    write_json(output/'source_manifest.json', source_manifest)
    write_json(output/'results.json', result)
    lines = ['# Four-way yoked surface-code comparison', '',
        '100,000 paired shots; six d=7 patches; two ideal yokes; 28 CZ rounds; '
        'SI1000 p=0.003; ideal time boundaries; plain MWPM at L2.', '',
        f'Stim {launch["stim_fork"]["version"]}, fork commit `{launch["stim_fork"]["source_commit"]}`. '
        f'Sampling seed: `{sample.seed}`. All four configurations use the same sampled circuit and detector rows.', '',
        '| L1 decoder + confidence | Block failures | Block LER | Exact 95% CI | Effective LER / patch / round |',
        '|---|---:|---:|---:|---:|']
    for name in NAMES:
        r = result['variants'][name]
        low, high = r['block_failure_ci95_exact']
        lines.append(f'| {r["label"]} | {r["block_failures"]:,} / {shots:,} | '
            f'{100*r["block_failure_rate"]:.3f}% | {100*low:.3f}%–{100*high:.3f}% | '
            f'{r["effective_ler_per_patch_round"]:.6g} |')
    lines += ['', 'A block failure means any of the twelve tracked patch-observable bits is wrong after L2. '
        'The last column follows the existing Sinter normalization with pieces=168 and values=8; '
        'it is not an independent measurement of one physical patch.', '',
        '| Candidate versus baseline | Repairs | Regressions | LER difference, percentage points (paired 95% CI) | Relative change |',
        '|---|---:|---:|---:|---:|']
    for r in result['paired'].values():
        b, a = NAMES.index(r['candidate']), NAMES.index(r['baseline'])
        low, high = r['difference_ci95']
        lines.append(f'| {LABELS[b]} vs {LABELS[a]} | {r["repairs"]:,} | {r["regressions"]:,} | '
            f'{100*r["block_failure_difference"]:+.3f} ({100*low:+.3f}, {100*high:+.3f}) | '
            f'{100*r["relative_change"]:+.2f}% |')
    lines += ['', 'Negative differences favor the candidate. Paired 95% intervals use 10,000 bootstrap '
        'replicates of whole six-patch shots. All six comparisons are shown; exploratory McNemar '
        'p-values are in results.json and have no multiple-comparison adjustment.', '',
        '## Definitions and checks', '',
        '- MWPM cluster gap is the existing MPP matching-radius score. Both MWPM configurations '
        'have identical L1 predictions and native chosen-class matching costs, checked on every shot.',
        '- BP5 is exactly five sum-product iterations, damping 0.5 and LLR limit 30, followed by '
        'the existing posterior-to-edge projection and two-pass correlated UF. Its correlation '
        'discounts use the original DEM-prior rules; this explicit composition differs from BP5 + plain UF.',
        '- UF confidence is the full, uncapped shortest odd logical boundary walk on the final '
        'UF growth costs. No calibration or evaluation-set tuning was performed.',
        '- Native corrections, remaining edge costs and gaps match independent Python oracles on '
        'all sixteen synthetic syndromes and eight d=7 cases, including zero syndrome. '
        'One-versus-32-thread outputs agree on sixteen shots of two patches. '
        'Every final native correction is checked against its input physical syndrome.',
        '- All final predictions obey the two ideal yoke checks; all retained row IDs and truth '
        'arrays match the same saved sample.', '',
        '## Execution', '',
        'The MWPM phase used 32 worker processes. The UF/BP phase used 32 verified OpenMP threads '
        'across shots. Phases ran separately on the same 32 physical cores. Stim sampling uses '
        'the fork’s native vectorized detector sampler.', '',
        f'MWPM phase wall time: {result["execution"]["mwpm_phase_seconds"]:.1f} s. '
        f'UF/BP collection wall time: {result["execution"]["uf_phase_seconds"]:.1f} s. '
        'These are collection times, not single-shot latency benchmarks.', '',
        '## Reproduction', '',
        'Source revisions, binary hashes and the MWPM command are in launch.json; original local '
        'changes are preserved in repository.diff and the complete source_snapshot. '
        'The initial MWPM preparation was restarted before sampling to add an omitted provenance '
        'file; mwpm_restart.json records this. UF/BP source and build commands are in '
        'data/uf/configuration.json. The full data sample and predictions are retained in data/.', '',
        '```bash', 'source /data2/s2chitni/projects/dante/env/workspace.sh',
        'source "$DANTE_REPO/.venv/bin/activate"',
        'python experiments/yoked_fourway_d7/launch.py --wait',
        '# Use the output directory printed by launch.py:',
        'python experiments/yoked_fourway_d7/run_uf.py --output RESULTS_DIRECTORY --threads 32',
        'python experiments/yoked_fourway_d7/analyze.py --output RESULTS_DIRECTORY', '```', '']
    (output/'report.md').write_text('\n'.join(lines))
    # Standard plotting tools provide an exportable comparison figure.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, ax = plt.subplots(figsize=(8.3, 3.4), constrained_layout=True)
    plot_labels = ('cMWPM + complementary gap', 'cMWPM + cluster gap',
                   'cUF + cluster gap', 'BP5 + cUF + cluster gap')
    for i, name in enumerate(NAMES):
        r = result['variants'][name]
        value = 100*r['block_failure_rate']
        low, high = 100*np.array(r['block_failure_ci95_exact'])
        ax.errorbar(value, i, xerr=[[value-low],[high-value]], fmt='o', capsize=4,
                    color=('#304B78','#597CA4','#BC7944','#387B66')[i], markersize=7)
        ax.annotate(f'{value:.3f}%', (high, i), xytext=(7,0), textcoords='offset points', va='center')
    ax.set_yticks(range(4), plot_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Final block logical error rate (%)')
    ax.set_title('Six yoked d=7 patches · SI1000 p=0.003 · 28 rounds\n100,000 paired shots · exact 95% binomial intervals', fontsize=11)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.margins(x=.25, y=.2)
    fig.savefig(output/'comparison.png', dpi=180)
    fig.savefig(output/'comparison.svg')
    plt.close(fig)
    files = [path for path in output.rglob('*') if path.is_file()
             and path.name not in ('artifact_manifest.json','progress.json')]
    write_json(output/'artifact_manifest.json', {str(p.relative_to(output)):sha(p) for p in sorted(files)})
    write_json(output/'progress.json', dict(state='complete', shots=shots,
        updated_utc=datetime.now(timezone.utc).isoformat(), report=str(output/'report.md'),
        results_sha256=sha(output/'results.json')))
    print(json.dumps({name:result['variants'][name] for name in NAMES}, indent=2), flush=True)
    print(f'Retained report: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
