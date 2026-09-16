"""Report the p=0.001 comparison with exact bounds for small failure counts."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sinter
from scipy.stats import binomtest

import collect
from experiment import summary, write_json
from yoked.hierarchical._provenance import sha256_file

HERE, REPO = collect.HERE, collect.REPO
DISTANCES = (7, 9, 11, 13)
LABELS = ('Correlated UF', 'UF + forest-cost control', 'UF + bridges',
          'Frontier UF', 'Frontier UF + forest-cost control', 'Frontier UF + bridges')


def convert(rate, distance):
    return float(sinter.shot_error_rate_to_piece_error_rate(float(rate), pieces=24*distance, values=8))


def rate_stats(count, n, d):
    ci = binomtest(count, n).proportion_ci(confidence_level=.95, method='exact')
    result = dict(failures=count, shots=n, normalized_ler=convert(count/n, d),
                  normalized_ler_ci95_exact=[convert(ci.low, d), convert(ci.high, d)])
    if count == 0:
        upper = -math.expm1(math.log(.05)/n)
        result.update(shot_failure_rate_upper95_one_sided=upper,
                      normalized_ler_upper95_one_sided=convert(upper, d))
    return result


def comparison(reference, changed, d):
    n = len(reference)
    a, b = int(changed.sum()), int(reference.sum())
    # Two 97.5% Clopper-Pearson intervals jointly cover with >=95%
    # probability by Bonferroni, regardless of within-shot dependence.
    ca = binomtest(a, n).proportion_ci(confidence_level=.975, method='exact')
    cb = binomtest(b, n).proportion_ci(confidence_level=.975, method='exact')
    lower = convert(ca.low, d)/convert(cb.high, d)
    upper = convert(ca.high, d)/convert(cb.low, d) if cb.low > 0 else None
    ratio = convert(a/n, d)/convert(b/n, d) if b else None
    repaired = int(np.count_nonzero(reference & ~changed))
    regressed = int(np.count_nonzero(~reference & changed))
    return dict(normalized_ler_ratio=ratio, relative_gap_percent=100*(ratio-1) if ratio is not None else None,
                ratio_ci95_conservative=[lower, upper], repaired=repaired, regressed=regressed,
                exact_mcnemar_p=binomtest(repaired, repaired+regressed).pvalue if repaired+regressed else 1.,
                reference_failures=b, decoder_failures=a)


def load(d):
    root = HERE/f'd{d}'
    request = json.loads((root/'request.json').read_text())
    saved = json.loads((root/'summary.json').read_text())
    assert request['shots'] == 100000
    assert request['parameters'] == dict(distance=d, p=.001, patches=6, rounds=4*d,
                                          yokes=2, noise='si1000', style='cz')
    for name, digest in request['sources'].items():
        assert sha256_file(REPO/name) == digest, name
    assert sha256_file(root/'sample.json') == request['sample']['manifest_sha256']
    assert sha256_file(root/'results.npz') == saved['validation']['results_sha256']
    with np.load(root/'results.npz') as file:
        arrays = dict(file)
    for key, value in summary(arrays, d).items():
        assert saved[key] == value, (d, key)
    assert arrays['predictions'].shape == (100000, 6)
    for before, after in ((1, 2), (4, 5)):
        assert np.all(arrays['costs'][:, after] <= arrays['costs'][:, before]+1e-7)
    return saved, arrays


def format_gap(row):
    value = row['relative_gap_percent']
    return 'N/A (zero MWPM failures)' if value is None else f'{value:+.1f}%'


def format_ci(ci):
    lo, hi = ci
    return f'[{lo:.3f}, {hi:.3f}]' if hi is not None else f'[{lo:.3f}, unbounded]'


def plot(analysis):
    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    for offset, (key, label, color, marker) in zip((-.08, 0, .08), (
            ('correlated_mwpm', 'Correlated MWPM', '#2563a6', 'o'),
            ('correlated_uf', 'Correlated UF', '#b45309', 's'),
            ('frontier_bridges', 'Frontier UF + bridges', '#19845b', '^'))):
        labelled = False
        for d in DISTANCES:
            row = analysis[d]['rates'][key]
            if row['failures']:
                value = row['normalized_ler']
                lo, hi = row['normalized_ler_ci95_exact']
                ax.errorbar([d+offset], [value], yerr=[[value-lo], [hi-value]], fmt=marker,
                            color=color, capsize=3, label=label if not labelled else None)
            else:
                upper = row['normalized_ler_upper95_one_sided']
                ax.scatter([d+offset], [upper], marker='v', facecolors='none', edgecolors=color,
                           label=label if not labelled else None)
            labelled = True
    ax.set(yscale='log', xticks=DISTANCES, xlabel='Distance d', ylabel='Normalized LER',
           title='p=0.001 · 100,000 shared shots per distance · n=6 · rounds=4d')
    ax.grid(alpha=.2, which='both')
    ax.legend(frameon=False)
    fig.supxlabel('Open downward triangles: 95% upper bounds for zero observed failures.\n'
                  'Error bars: exact two-sided 95% intervals. Markers offset slightly for visibility.', fontsize=9)
    fig.savefig(HERE/'low_noise_ler.png', dpi=180)
    svg = HERE/'low_noise_ler.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    audit = json.loads((HERE/'failure_audit.json').read_text())
    for name, digest in audit['sources'].items():
        assert sha256_file(REPO/name) == digest, name
    analysis, reports = {}, {}
    for d in DISTANCES:
        saved, arrays = load(d)
        reports[d] = saved
        assert audit['distances'][str(d)]['results_sha256'] == saved['validation']['results_sha256']
        failures = arrays['predictions'] != arrays['actual'][:, None]
        mwpm_failed = arrays['matching'] != arrays['actual']
        rates = {'correlated_mwpm': rate_stats(int(mwpm_failed.sum()), 100000, d)}
        rates.update({name: rate_stats(int(failures[:, k].sum()), 100000, d)
                      for k, name in enumerate(saved['variants'])})
        analysis[d] = dict(rates=rates, uf_vs_mwpm=comparison(mwpm_failed, failures[:, 0], d),
                           frontier_vs_mwpm=comparison(mwpm_failed, failures[:, 5], d),
                           frontier_vs_uf=comparison(failures[:, 0], failures[:, 5], d),
                           bridges_vs_frontier=comparison(failures[:, 3], failures[:, 5], d))
    write_json(HERE/'analysis.json', dict(distances=analysis,
        normalization='sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=24*d, values=8)',
        intervals='Exact two-sided 95% Clopper-Pearson LER intervals; exact one-sided 95% upper bounds for zero counts. Ratio intervals conservatively combine two 97.5% marginal exact intervals (Bonferroni), valid for paired observations. A null ratio is undefined because zero reference failures were observed; a null upper endpoint is unbounded. Comparisons are exploratory and unadjusted across distances/decoders.'))
    plot(analysis)
    unresolved = ', '.join(str(d) for d in DISTANCES
                           if analysis[d]['rates']['correlated_mwpm']['failures'] == 0)
    lines = ['# Frozen UF clustering comparison at p=0.001', '',
             f'The observed relative LER gaps cannot be calculated at d={unresolved}: '
             'MWPM has zero failures in those 100,000-shot samples. The small counts at '
             'the other distances also limit precision. These runs do not establish that '
             'the decoders have equal true LER or that the p=0.003 gap has disappeared.', '',
             '100,000 new shots at each of d=7,9,11,13, shared by every decoder at that distance. '
             'SI1000 p=0.001, n=6 patches, two ideal yokes, CZ circuits, rounds=4d. Seed 42 and '
             'one full packed Stim sampling call per distance. Decoder rules are unchanged from p=0.003; '
             'the model and correlation weights are rebuilt at the new physical noise probability.', '',
             '## Failure counts and relative gaps', '',
             'Relative gap is `(LER_decoder / LER_correlated_MWPM - 1) * 100%`, using the previous '
             'normalization. A zero MWPM count makes the observed ratio undefined. Equal zero counts '
             'do not establish equal true error rates.', '',
             '| d | MWPM failures | Correlated UF failures | Frontier + bridges failures | UF gap vs MWPM | Frontier + bridges gap vs MWPM |',
             '|---:|---:|---:|---:|---:|---:|']
    for d, a in analysis.items():
        r = a['rates']
        lines.append(f"| {d} | {r['correlated_mwpm']['failures']} | {r['correlated_uf']['failures']} | "
                     f"{r['frontier_bridges']['failures']} | {format_gap(a['uf_vs_mwpm'])} | {format_gap(a['frontier_vs_mwpm'])} |")
    lines += ['', '## Normalized LER and uncertainty', '',
              'The normalization is `sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=24*d, values=8)`. '
              'Intervals below are exact binomial intervals transformed through this monotone normalization. '
              'For zero failures, `< bound` denotes a one-sided 95% upper confidence bound, not a measured nonzero LER.', '',
              '| d | Decoder | Observed normalized LER | Exact 95% interval or zero-count upper bound |',
              '|---:|---|---:|---:|']
    for d, a in analysis.items():
        for key, label in [('correlated_mwpm', 'Correlated MWPM'), ('correlated_uf', 'Correlated UF'),
                           ('frontier_bridges', 'Frontier UF + bridges')]:
            r = a['rates'][key]
            lo, hi = r['normalized_ler_ci95_exact']
            interval = f'[{lo:.4e}, {hi:.4e}]' if r['failures'] else f"< {r['normalized_ler_upper95_one_sided']:.4e} (one-sided)"
            lines.append(f"| {d} | {label} | {r['normalized_ler']:.4e} | {interval} |")
    lines += ['', f'![Low-noise LER estimates and bounds]({HERE.name}/low_noise_ler.png)', '',
              'Zero failures in 100,000 shots gives the same whole-shot 95% upper bound '
              f'of {-math.expm1(math.log(.05)/100000):.6g} at every distance. Differences between normalized '
              'zero-count bounds are due to the rounds normalization; they do not measure distance suppression.', '',
              '## Uncertainty in comparisons with MWPM', '',
              'The ratio bounds conservatively combine two exact 97.5% marginal intervals, giving at least '
              '95% simultaneous coverage by Bonferroni regardless of pairing. They are deliberately not a '
              'bootstrap that would treat an observed zero count as a known zero probability. Exact McNemar '
              'tests use only discordant paired outcomes. These are exploratory comparisons without '
              'multiplicity adjustment across distances and decoders.', '',
              '| d | Decoder | LER-ratio 95% conservative interval | MWPM failures repaired | MWPM successes spoiled | Exact McNemar p |',
              '|---:|---|---:|---:|---:|---:|']
    for d, a in analysis.items():
        for key, label in [('uf_vs_mwpm', 'Correlated UF'), ('frontier_vs_mwpm', 'Frontier UF + bridges')]:
            row = a[key]
            lines.append(f"| {d} | {label} | {format_ci(row['ratio_ci95_conservative'])} | "
                         f"{row['repaired']} | {row['regressed']} | {row['exact_mcnemar_p']:.4g} |")
    lines += ['', '## All six UF variants', '', '| Decoder | d=7 failures | d=9 failures | d=11 failures | d=13 failures |',
              '|---|---:|---:|---:|---:|']
    for k, label in enumerate(LABELS):
        lines.append('| '+label+' | '+' | '.join(str(reports[d]['rows'][k]['failures']) for d in DISTANCES)+' |')
    targeted = sum(row['shots'] for row in audit['distances'].values())
    lines += ['', '## Validation and reproducibility', '',
              'The original native kernel, growth rule, 0.5-nat bridge eligibility cap, two bridge rounds '
              'and first-pass UF correlation rules are frozen. No truth or MWPM output enters UF inference. '
              'Source hashes, full sample identities, per-shot predictions and operation counters are retained. '
              'The experiment makes no hardware timing or resource claims.', '',
              'All 2,400,000 new UF corrections pass full-syndrome validation; all seven outputs pass yoke parity. '
              'At each distance, 32 predetermined random rows agree with production Python physical corrections '
              'and independent tree costs; two rows also match a full-edge frontier-growth scan. Native outputs '
              'agree between one and 64 threads, and MWPM agrees between packed parallel and unpacked serial decoding.', '',
              f'All {targeted} rows where any decoder failed were then independently rechecked against Python, '
              'including full-edge scans, and exactly reproduced the saved results. This outcome-selected '
              'validation changes neither decoder settings nor predictions. The failure-audit manifest records '
              'the exact rows and input hashes.', '',
              'The archived `summary.json` files retain the legacy summary format for reproducibility. '
              'The small-count conclusions and intervals in this report use `analysis.json` and the exact '
              'procedures above, not the legacy normal intervals on paired changes.', '',
              f'[Reproduction instructions]({HERE.name}/README.md). '
              f'[Exact statistics]({HERE.name}/analysis.json). '
              f'[Failure-case validation]({HERE.name}/failure_audit.json).', '']
    path = HERE.with_suffix('.md')
    path.write_text('\n'.join(lines))
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json')
    write_json(HERE/'artifact_manifest.json', dict(sha256={str(p.relative_to(HERE)): sha256_file(p) for p in files},
                                                 report_sha256=sha256_file(path)))
    print(path)
    print(json.dumps({d: {key: value for key, value in a.items() if key != 'rates'} for d, a in analysis.items()}, indent=2))


if __name__ == '__main__':
    main()
