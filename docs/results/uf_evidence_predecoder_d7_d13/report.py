"""Verify and summarize the paired evidence experiment without rerunning decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import sinter
from scipy.stats import binomtest

from evidence import HERE, VARIANTS, sha256, write_json


def paired(failure, reference):
    n = len(failure)
    repairs = int(np.count_nonzero(reference & ~failure))
    regressions = int(np.count_nonzero(~reference & failure))
    improvement = (repairs-regressions)/n
    se = float(np.sqrt(max(0., (repairs+regressions)/n-improvement**2)/n))
    return dict(repairs=repairs, regressions=regressions, shot_failure_reduction=improvement,
        shot_failure_reduction_ci95=[improvement-1.96*se, improvement+1.96*se],
        mcnemar_p=binomtest(repairs, repairs+regressions).pvalue if repairs+regressions else 1.)


def summarize(directory):
    request = json.loads((directory/'request.json').read_text())
    verification = json.loads((directory/'verification.json').read_text())
    if sha256(directory/'results.npz') != verification['result_sha256']:
        raise ValueError('Results checksum changed')
    if sha256(directory/'diagnostic.json') != verification['diagnostic_sha256']:
        raise ValueError('Diagnostic checksum changed')
    with np.load(directory/'results.npz', allow_pickle=False) as saved:
        arrays = dict(saved)
    d, n = request['distance'], request['shots']
    if arrays['predictions'].shape != (n, len(VARIANTS)) or len(np.unique(arrays['rows'])) != n:
        raise ValueError('Result shape or row uniqueness changed')
    convert = lambda p: float(sinter.shot_error_rate_to_piece_error_rate(p, pieces=24*d, values=8))
    failures = arrays['predictions'] != arrays['actual'][:, None]
    mwpm = arrays['matching'] != arrays['actual']
    rows = []
    for k, name in enumerate(VARIANTS):
        count = int(failures[:, k].sum())
        ci = binomtest(count, n).proportion_ci(method='wilson')
        uf = arrays['serial_uf_seconds'][:, k]
        evidence = arrays['serial_evidence_seconds'][:, k]
        rows.append(dict(variant=name, failures=count, shots=n, shot_failure_rate=count/n,
            shot_failure_ci95=[ci.low, ci.high], normalized_ler=convert(count/n),
            normalized_ler_ci95=[convert(ci.low), convert(ci.high)],
            vs_correlated_uf=paired(failures[:, k], failures[:, 1]),
            vs_prior_projection=paired(failures[:, k], failures[:, 2]),
            vs_correlated_mwpm=paired(failures[:, k], mwpm),
            serial_kernel_timing=dict(shots=len(uf), evidence_mean_ms=float(evidence.mean()*1000),
                uf_mean_ms=float(uf.mean()*1000), total_mean_ms=float((uf+evidence).mean()*1000),
                total_median_ms=float(np.median(uf+evidence)*1000)),
            mean_largest_cluster=float(arrays['max_cluster'][:, k].mean()),
            mean_zero_weight_edges=float(arrays['zero_weight_edges'][:, k].mean())))
    count = int(mwpm.sum())
    diagnosis = json.loads((directory/'diagnostic.json').read_text())
    diagnostic_summary = {}
    for name in (VARIANTS[1], VARIANTS[3], VARIANTS[4]):
        variants = [r['variants'][name] for r in diagnosis['rows']]
        diagnostic_summary[name] = dict(shots=len(variants), correct=sum(v['correct'] for v in variants),
            forest_allows_truth=sum(v['forest_allows_truth'] for v in variants),
            partition_allows_truth=sum(v['partition_allows_truth'] for v in variants),
            residual_failures_excluding_truth=sum(not v['correct'] and not v['partition_allows_truth'] for v in variants))
    return dict(distance=d, shots=n, rows=rows, model=request['model'],
        correlated_mwpm=dict(failures=count, shot_failure_rate=count/n, normalized_ler=convert(count/n)),
        diagnostic=diagnostic_summary, validation=verification,
        intervals='Wilson marginal intervals; paired normal intervals for raw failure-rate differences; '
                  'exact McNemar p-values. Exploratory and not adjusted for multiple comparisons.')


def report(root):
    reports = [summarize(p) for p in sorted(root.glob('d*'), key=lambda p: int(p.name[1:]))
               if p.is_dir() and (p/'verification.json').exists()]
    if not reports:
        raise ValueError('No completed distances')
    if root.resolve() != HERE.resolve():
        shutil.copyfile(HERE/'README.md', root/'README.md')
    for result in reports:
        write_json(root/f"d{result['distance']}"/'summary.json', result)
    lines = ['# Soft evidence before unchanged UF', '',
        'This paired pilot changes the evidence supplied to UF while preserving its '
        'growth, stopping, and peeling rules. It uses SI1000 p=0.003, six patches, two '
        'ideal yokes, CZ circuits, and rounds=4d.', '',
        'The BP variants run 5 or 20 fixed damped sum-product iterations over joint '
        'DEM fault mechanisms, then supply nonnegative weights to UF. No syndrome '
        'or graph pruning occurs, and BP never returns the final correction. '
        'Prior projection applies the same weight convention without syndrome evidence.', '',
        '## Paired failure counts', '',
        '| d | Shots | Plain UF | Correlated UF | Prior projection | BP5 + UF | BP20 + UF | Correlated MWPM |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['distance']} | {r['shots']:,} | " + ' | '.join(f"{row['failures']:,}" for row in r['rows'])
                     + f" | {r['correlated_mwpm']['failures']:,} |")
    lines += ['', '## Evidence effect relative to current correlated UF', '',
        'Positive reduction means fewer whole-shot failures. Intervals preserve '
        'the paired shot comparison. Both budgets were specified before evaluation.', '',
        '| d | Evidence | Repairs | Regressions | Failure-rate reduction, percentage points (95% CI) |',
        '|---:|---|---:|---:|---:|']
    for r in reports:
        for row in r['rows'][3:]:
            p = row['vs_correlated_uf']; lo, hi = p['shot_failure_reduction_ci95']
            lines.append(f"| {r['distance']} | {row['variant']} | {p['repairs']} | {p['regressions']} | "
                         f"{100*p['shot_failure_reduction']:.2f} [{100*lo:.2f}, {100*hi:.2f}] |")
    lines += ['', '## Evidence effect relative to the prior-projection control', '',
        '| d | Evidence | Repairs | Regressions | Failure-rate reduction, percentage points (95% CI) |',
        '|---:|---|---:|---:|---:|']
    for r in reports:
        for row in r['rows'][3:]:
            p = row['vs_prior_projection']; lo, hi = p['shot_failure_reduction_ci95']
            lines.append(f"| {r['distance']} | {row['variant']} | {p['repairs']} | {p['regressions']} | "
                         f"{100*p['shot_failure_reduction']:.2f} [{100*lo:.2f}, {100*hi:.2f}] |")
    lines += ['', '## Normalized LER and distance trend', '',
        'Normalization is `sinter.shot_error_rate_to_piece_error_rate(failures/shots, '
        'pieces=24*d, values=8)`, matching the earlier reports. This finite-distance '
        'pilot is not a threshold or asymptotic scaling measurement.', '',
        '| d | Correlated UF | BP5 + UF | BP20 + UF | Correlated MWPM | BP5/MWPM | BP20/MWPM |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        base, bp5, bp20 = [r['rows'][k]['normalized_ler'] for k in (1,3,4)]
        mwpm = r['correlated_mwpm']['normalized_ler']
        ratios = [f'{p/mwpm:.3f}' if mwpm else 'undefined' for p in (bp5,bp20)]
        lines.append(f"| {r['distance']} | {base:.7g} | {bp5:.7g} | {bp20:.7g} | {mwpm:.7g} | "
                     + ' | '.join(ratios) + ' |')
    lines += ['', '## Conditional logical-accessibility diagnostic', '',
        'At each distance, uniformly select up to 32 pilot shots on which correlated '
        'UF fails and correlated MWPM succeeds. These are conditional diagnostics, '
        'not full-population rates. Restore every original edge internal to each '
        'final UF component and test whether the true logical class is reachable.', '',
        '| d | Variant | Correct / diagnostic shots | Partition admits truth | Remaining failures excluding truth |',
        '|---:|---|---:|---:|---:|']
    for r in reports:
        for name, q in r['diagnostic'].items():
            lines.append(f"| {r['distance']} | {name} | {q['correct']} / {q['shots']} | "
                         f"{q['partition_allows_truth']} / {q['shots']} | {q['residual_failures_excluding_truth']} |")
    lines += ['', '## Software preprocessing cost', '',
        'Means over 16 predetermined serial shots per distance, measured inside the '
        'native kernel. Correlated-UF evidence time includes its first UF solve and '
        'reweighting; BP evidence time includes initialization, all message iterations '
        'and projection. The common model setup, Python orchestration, packing and '
        'post-solve syndrome audit are excluded. Shared BP work is charged in full '
        'to each budget. These small samples are not a tail-latency study or a hardware benchmark.', '',
        '| d | Variant | Evidence ms | Final UF ms | Total ms |',
        '|---:|---|---:|---:|---:|']
    for r in reports:
        for row in r['rows']:
            t=row['serial_kernel_timing']
            lines.append(f"| {r['distance']} | {row['variant']} | {t['evidence_mean_ms']:.3f} | "
                         f"{t['uf_mean_ms']:.3f} | {t['total_mean_ms']:.3f} |")
    lines += ['', '## Validation and limits', '',
        'The complete historical 100,000-shot sampling calls were regenerated and '
        'their packed payload hashes verified; model text matches exactly or differs '
        'only by one explicitly verified trailing newline. Pilot rows were selected '
        'uniformly by a fixed seed before outcomes, and archived truth matches the '
        'reproduced sample on all parent rows. These are historical evaluation shots, '
        'not a fresh independent confirmation set.', '',
        'Every pilot correlated-UF prediction reproduces its saved baseline. Every '
        'new correction satisfies the full syndrome and every prediction satisfies '
        'the two ideal-yoke parities. Four predetermined real shots per distance '
        'compare all five physical forests and corrections with production Python UF, '
        'compare BP posteriors with an independent implementation, and reproduce '
        'correlated MWPM. Native outputs and physical corrections agree across '
        'thread counts. Synthetic checks include exact acyclic posteriors, zero '
        'messages, high-degree checks and correlated detector cancellations.', '',
        'BP estimates on this loopy graph and the projection into edge weights are '
        'approximations. Improvements or regressions test this particular evidence '
        'layer and budget; they do not establish that all soft evidence helps or '
        'fails. This joint-decoder experiment also does not measure a separate L1/L2 '
        'hierarchy or confidence calibration. Intervals are exploratory, without '
        'multiplicity adjustment.', '',
        '![Accuracy and preprocessing cost](comparison.png)', '',
        'See the [protocol and reproduction instructions](README.md), the per-distance '
        '`request.json`, `verification.json`, `summary.json`, `diagnostic.json`, and '
        '`results.npz`. Source and artifact hashes are in `artifact_manifest.json`.', '']
    (root/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    distances = [r['distance'] for r in reports]
    colors = ['#7b8794', '#d55e00', '#8b6bb1', '#0072b2', '#009e73']
    for k in (1, 3, 4):
        values = np.array([r['rows'][k]['normalized_ler'] for r in reports])
        bounds = np.array([r['rows'][k]['normalized_ler_ci95'] for r in reports]).T
        axes[0].errorbar(distances, values, yerr=[values-bounds[0], bounds[1]-values],
                         marker='o', capsize=3, color=colors[k], label=VARIANTS[k])
        axes[1].plot(distances, [r['rows'][k]['serial_kernel_timing']['total_mean_ms'] for r in reports],
                      marker='o', color=colors[k], label=VARIANTS[k])
    axes[0].plot(distances, [r['correlated_mwpm']['normalized_ler'] for r in reports],
                  marker='s', color='black', label='correlated_mwpm')
    axes[0].set(ylabel='Normalized LER (95% marginal CI)', yscale='log', title='Paired accuracy pilot')
    axes[1].set(ylabel='Mean serial kernel time (ms)', yscale='log', title='Evidence + final UF; 16 shots')
    for ax in axes:
        ax.set_xlabel('Distance d')
        ax.set_xticks(distances)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(root/'comparison.png', dpi=180)
    fig.savefig(root/'comparison.svg')
    plt.close(fig)
    # A report manifest is published last, over completed durable artifacts.
    paths = [p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    write_json(root/'artifact_manifest.json', dict(files={str(p.relative_to(root)): sha256(p) for p in sorted(paths)},
               distances=[r['distance'] for r in reports]))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=HERE)
    args = parser.parse_args()
    summaries = report(args.directory)
    print(args.directory/'report.md')
    for r in summaries:
        print(r['distance'], {row['variant']: row['failures'] for row in r['rows']},
              'correlated_mwpm', r['correlated_mwpm']['failures'])


if __name__ == '__main__':
    main()
