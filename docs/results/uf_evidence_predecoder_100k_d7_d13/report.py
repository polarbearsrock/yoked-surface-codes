"""Summarize verified 100k paired results, including rows outside the BP pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import sinter
from scipy.stats import binomtest

from common import HERE, VARIANTS, sha256, write_json

NAMES = ('correlated_uf', *VARIANTS, 'correlated_mwpm')
LABELS = ('Correlated UF', 'BP5 + UF', 'BP5 + correlated UF', 'Correlated MWPM')


def paired(failure, reference):
    n = len(failure)
    repairs = int(np.count_nonzero(reference & ~failure))
    regressions = int(np.count_nonzero(~reference & failure))
    excess = (regressions-repairs)/n
    se = float(np.sqrt(max(0., (repairs+regressions)/n-excess**2)/n))
    return dict(repairs=repairs, regressions=regressions, failure_rate_excess=excess,
        failure_rate_excess_ci95=[excess-1.96*se, excess+1.96*se],
        mcnemar_p=binomtest(repairs, repairs+regressions).pvalue if repairs+regressions else 1.)


def cohort(predictions, actual, distance):
    n = len(actual)
    failure = predictions != actual[:, None]
    convert = lambda p: float(sinter.shot_error_rate_to_piece_error_rate(p, pieces=24*distance, values=8))
    result = {}
    mwpm_rate = float(failure[:, 3].mean())
    mwpm_ler = convert(mwpm_rate)
    for k, name in enumerate(NAMES):
        count = int(failure[:, k].sum())
        rate = count/n
        ci = binomtest(count, n).proportion_ci(method='wilson')
        result[name] = dict(failures=count, shots=n, shot_failure_rate=rate,
            shot_failure_ci95=[ci.low, ci.high], normalized_ler=convert(rate),
            normalized_ler_ci95=[convert(ci.low), convert(ci.high)],
            raw_failure_ratio_to_mwpm=rate/mwpm_rate if mwpm_rate else None,
            normalized_ler_ratio_to_mwpm=convert(rate)/mwpm_ler if mwpm_ler else None,
            vs_correlated_mwpm=paired(failure[:, k], failure[:, 3]),
            vs_correlated_uf=paired(failure[:, k], failure[:, 0]),
            vs_bp5_uf=paired(failure[:, k], failure[:, 1]))
    return dict(shots=n, decoders=result)


def summarize(directory):
    request = json.loads((directory/'request.json').read_text())
    verification = json.loads((directory/'verification.json').read_text())
    if sha256(directory/'results.npz') != verification['result_sha256']:
        raise ValueError('Results checksum differs')
    with np.load(directory/'results.npz', allow_pickle=False) as saved:
        a = dict(saved)
    n = request['shots']
    assert n == 100000 and a['predictions'].shape == (n, 2)
    assert request['variants'] == list(VARIANTS)
    np.testing.assert_array_equal(a['rows'], np.arange(n))
    assert verification['full_syndrome_checks'] == 2*n
    assert verification['recomputed_correlated_mwpm_matches'] == n
    assert verification['pilot_bp5_predictions_and_all_nontiming_diagnostics_reproduced'] == 5000
    pilot = np.zeros(n, dtype=bool)
    pilot[a['pilot_rows']] = True
    assert pilot.sum() == 5000
    predictions = np.column_stack((a['correlated_uf'], a['predictions'], a['matching']))
    d = request['distance']
    timing = {}
    for k, name in enumerate(VARIANTS):
        evidence, uf = a['serial_evidence_seconds'][:, k], a['serial_uf_seconds'][:, k]
        timing[name] = dict(shots=len(evidence), evidence_mean_ms=float(evidence.mean()*1000),
            final_uf_mean_ms=float(uf.mean()*1000), total_mean_ms=float((evidence+uf).mean()*1000))
    return dict(distance=d, all_rows=cohort(predictions, a['actual'], d),
        non_pilot_rows=cohort(predictions[~pilot], a['actual'][~pilot], d),
        pilot_rows=cohort(predictions[pilot], a['actual'][pilot], d),
        serial_native_timing=timing, model=request['model'], validation=verification,
        intervals='Wilson marginal 95% intervals; paired normal 95% intervals for raw failure-rate excess; '
                  'exact McNemar tests. Exploratory, no multiplicity adjustment.')


def report(root):
    distances = [7,9,11,13]
    reports = [summarize(root/f'd{d}') for d in distances]
    for r in reports:
        write_json(root/f"d{r['distance']}"/'summary.json', r)
    if root.resolve() != HERE.resolve():
        shutil.copyfile(HERE/'README.md', root/'README.md')
    lines = ['# BP5 and correlated UF versus correlated MWPM: 100,000 paired shots', '',
        'Each distance uses all 100,000 rows of the same archived SI1000 p=0.003 '
        'sample: six patches, two ideal yokes, CZ circuits, rounds=4d, seed 42. '
        'All decoders see identical syndromes. A whole-shot failure means at least '
        'one of the 12 logical-observable predictions differs from truth.', '',
        '**BP5 + UF** reproduces the original pilot exactly. **BP5 + correlated UF** '
        'adds the repository\'s prior-derived correlation discounts, based on the '
        'BP-weighted first correction, followed by a second unchanged UF pass. '
        'Both passes use the original syndrome. BP already includes correlated '
        'faults, so the additional hard-evidence pass can double-count information. '
        'The variants and parameters were fixed before this expanded run.', '',
        'Correlated MWPM was recomputed on every row and verified against the '
        'archived predictions. Current correlated-UF predictions are reused from '
        'the identical verified samples and checked on predetermined rows. '
        'The BP pilot\'s 5,000 rows are included; the remaining 95,000 rows are '
        'reported separately below. These are historical evaluation samples, '
        'not newly seeded independent confirmation samples.', '',
        '## Whole-shot failures out of 100,000', '',
        '| d | Correlated UF | BP5 + UF | BP5 + correlated UF | Correlated MWPM |',
        '|---:|---:|---:|---:|---:|']
    for r in reports:
        values = r['all_rows']['decoders']
        lines.append(f"| {r['distance']} | " + ' | '.join(f"{values[name]['failures']:,}" for name in NAMES) + ' |')
    lines += ['', '## Failure rates and marginal uncertainty', '',
        '| d | Decoder | Whole-shot failure % (95% CI) | Ratio to MWPM, raw failure rate |',
        '|---:|---|---:|---:|']
    for r in reports:
        for name, label in zip(NAMES, LABELS):
            v = r['all_rows']['decoders'][name]
            lo, hi = v['shot_failure_ci95']
            ratio = v['raw_failure_ratio_to_mwpm']
            lines.append(f"| {r['distance']} | {label} | {100*v['shot_failure_rate']:.3f} "
                f"[{100*lo:.3f}, {100*hi:.3f}] | {ratio:.4f} |")
    lines += ['', '## Paired comparison against correlated MWPM', '',
        'A repair is an MWPM failure corrected by the BP variant. A regression '
        'is an MWPM success that the BP variant fails. Positive excess means '
        'more failures than MWPM. Differences use pairing, not overlapping '
        'marginal confidence intervals.', '',
        '| d | Decoder | Repairs | Regressions | Excess failure rate, pp (95% CI) | McNemar p |',
        '|---:|---|---:|---:|---:|---:|']
    for r in reports:
        for name, label in zip(VARIANTS, LABELS[1:3]):
            v = r['all_rows']['decoders'][name]['vs_correlated_mwpm']
            lo, hi = v['failure_rate_excess_ci95']
            lines.append(f"| {r['distance']} | {label} | {v['repairs']:,} | {v['regressions']:,} | "
                f"{100*v['failure_rate_excess']:.3f} [{100*lo:.3f}, {100*hi:.3f}] | {v['mcnemar_p']:.4g} |")
    lines += ['', '## Effect of the additional correlation pass', '',
        'Here repairs and regressions compare BP5 + correlated UF with BP5 + UF. '
        'Positive excess means that the extra pass hurts accuracy.', '',
        '| d | Repairs | Regressions | Excess failure rate, pp (95% CI) | McNemar p |',
        '|---:|---:|---:|---:|---:|']
    for r in reports:
        v = r['all_rows']['decoders']['bp5_correlated_uf']['vs_bp5_uf']
        lo, hi = v['failure_rate_excess_ci95']
        lines.append(f"| {r['distance']} | {v['repairs']:,} | {v['regressions']:,} | "
            f"{100*v['failure_rate_excess']:.3f} [{100*lo:.3f}, {100*hi:.3f}] | {v['mcnemar_p']:.4g} |")
    lines += ['', '## Improvement over current correlated UF', '',
        '| d | Decoder | Failure-count reduction | Repairs | Regressions |',
        '|---:|---|---:|---:|---:|']
    for r in reports:
        base = r['all_rows']['decoders']['correlated_uf']['failures']
        for name, label in zip(VARIANTS, LABELS[1:3]):
            v = r['all_rows']['decoders'][name]
            p = v['vs_correlated_uf']
            lines.append(f"| {r['distance']} | {label} | {100*(1-v['failures']/base):.2f}% | "
                f"{p['repairs']:,} | {p['regressions']:,} |")
    lines += ['', '## Historical normalized LER', '',
        'Use `sinter.shot_error_rate_to_piece_error_rate(failures/shots, '
        'pieces=24*d, values=8)`, matching earlier reports. This is a reporting '
        'convention, not an independently measured per-round failure rate. '
        'The transformation is nonlinear, so its ratios differ from raw '
        'whole-shot failure-rate ratios.', '',
        '| d | Correlated UF | BP5 + UF | BP5 + correlated UF | Correlated MWPM | BP5/MWPM | BP5+cUF/MWPM |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        v = r['all_rows']['decoders']
        lines.append(f"| {r['distance']} | " + ' | '.join(f"{v[name]['normalized_ler']:.7g}" for name in NAMES)
            + ' | ' + ' | '.join(f"{v[name]['normalized_ler_ratio_to_mwpm']:.4f}" for name in VARIANTS) + ' |')
    lines += ['', '## The 95,000 rows outside the original BP pilot', '',
        'This subset excludes all original pilot rows. It is reported without '
        'changing decoder parameters or selecting the better decoder per shot.', '',
        '| d | Correlated UF | BP5 + UF | BP5 + correlated UF | Correlated MWPM |',
        '|---:|---:|---:|---:|---:|']
    for r in reports:
        v = r['non_pilot_rows']['decoders']
        lines.append(f"| {r['distance']} | " + ' | '.join(f"{v[name]['failures']:,}" for name in NAMES) + ' |')
    lines += ['', 'All cohort-level confidence intervals and paired comparisons '
        'are available in the per-distance `summary.json` files.', '',
        '## Software cost', '',
        'Means over 16 predetermined serial shots per distance, measured inside '
        'the native kernel after parallel decoding and matching have finished. '
        'Each variant is charged all of its shared evidence work. The second '
        'variant\'s evidence time includes the first UF solve and correlation '
        'discounts. These are software kernel timings, not FPGA or ASIC estimates. '
        'See `environment.json` for scope and hardware.', '',
        '| d | Decoder | Evidence ms | Final UF ms | Total ms |',
        '|---:|---|---:|---:|---:|']
    for r in reports:
        for name, label in zip(VARIANTS, LABELS[1:3]):
            t = r['serial_native_timing'][name]
            lines.append(f"| {r['distance']} | {label} | {t['evidence_mean_ms']:.3f} | "
                f"{t['final_uf_mean_ms']:.3f} | {t['total_mean_ms']:.3f} |")
    lines += ['', '## Validation and limitations', '',
        'All 800,000 new UF corrections reproduce the full syndrome and satisfy '
        'the ideal-yoke prediction parities. All 400,000 freshly recomputed '
        'correlated-MWPM predictions match their archived references. All 20,000 '
        'overlapping BP5 predictions and every non-timing diagnostic reproduce '
        'the frozen pilot exactly. Independent Python checks validate BP '
        'posteriors, physical correction edges, merge forests, the extra '
        'correlation pass, and thread invariance. Exact synthetic BP checks '
        'cover trees, zero messages, high-degree constraints and correlated '
        'detector cancellations.', '',
        'BP and the projection into graph weights are approximate. Adding '
        'prior-derived hard correlation discounts after BP is also a heuristic. '
        'This is joint decoding with ideal yokes; it does not separately test '
        'the L1/L2 hierarchy. One physical noise rate and four finite distances '
        'do not establish an asymptotic threshold or suppression law. Intervals '
        'and paired tests are exploratory, without multiplicity adjustment.', '',
        '![Accuracy and paired difference from correlated MWPM](comparison.png)', '',
        'See the [fixed protocol](README.md), per-distance request/sample '
        'identities, verification records, and per-shot `results.npz` arrays. '
        'Source identities are in each request; durable artifact hashes are '
        'in `artifact_manifest.json`.', '']
    (root/'report.md').write_text('\n'.join(lines))
    plot(root, reports)
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json')
    write_json(root/'artifact_manifest.json', dict(distances=distances,
        files={str(p.relative_to(root)):sha256(p) for p in paths}))
    return reports


def plot(root, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5), constrained_layout=True)
    ds = [r['distance'] for r in reports]
    colors = ('#d55e00', '#0072b2', '#009e73', '#222222')
    for name, label, color in zip(NAMES, LABELS, colors):
        values = np.array([r['all_rows']['decoders'][name]['normalized_ler'] for r in reports])
        bounds = np.array([r['all_rows']['decoders'][name]['normalized_ler_ci95'] for r in reports]).T
        axes[0].errorbar(ds, values, yerr=[values-bounds[0], bounds[1]-values],
            marker='o', capsize=3, label=label, color=color)
    for name, label, color in zip(VARIANTS, LABELS[1:3], colors[1:3]):
        records = [r['all_rows']['decoders'][name]['vs_correlated_mwpm'] for r in reports]
        values = np.array([v['failure_rate_excess'] for v in records])*100
        bounds = np.array([v['failure_rate_excess_ci95'] for v in records]).T*100
        axes[1].errorbar(ds, values, yerr=[values-bounds[0], bounds[1]-values],
            marker='o', capsize=3, label=label, color=color)
    axes[0].set(yscale='log', ylabel='Historical normalized LER (95% marginal CI)',
                title='100,000 paired shots per distance')
    axes[1].axhline(0, color='#444444', linestyle='--', linewidth=1)
    axes[1].set(ylabel='Whole-shot failure excess over MWPM (pp)',
                title='Paired difference (95% CI); lower is better')
    for ax in axes:
        ax.set_xlabel('Distance d')
        ax.set_xticks(ds)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(root/'comparison.png', dpi=180)
    fig.savefig(root/'comparison.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for r in report(args.directory):
        print(r['distance'], {n:v['failures'] for n,v in r['all_rows']['decoders'].items()})


if __name__ == '__main__':
    main()
