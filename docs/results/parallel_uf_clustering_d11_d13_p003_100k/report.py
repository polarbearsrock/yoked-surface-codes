"""Audit the saved extension and report paired LER scaling at d=7,9,11,13."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sinter
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN = HERE.parent / 'parallel_uf_clustering_d7_d9_p003_100k'
sys.path.insert(0, str(FROZEN))
from experiment import summary, write_json
from yoked.hierarchical._provenance import sha256_file

DISTANCES = (7, 9, 11, 13)
LABELS = ('Correlated UF', 'UF + forest-cost control', 'UF + bridges',
          'Frontier UF', 'Frontier UF + forest-cost control', 'Frontier UF + bridges')


def convert(rate, distance):
    return float(sinter.shot_error_rate_to_piece_error_rate(float(rate), pieces=24*distance, values=8))


def paired_counts(reference, changed):
    repaired = int(np.count_nonzero(reference & ~changed))
    regressed = int(np.count_nonzero(~reference & changed))
    reduction = (repaired-regressed) / len(reference)
    se = np.sqrt(max(0., (repaired+regressed)/len(reference)-reduction**2)/len(reference))
    return dict(repaired=repaired, regressed=regressed, net_repairs=repaired-regressed,
                shot_failure_reduction=reduction,
                reduction_ci95=[reduction-1.96*se, reduction+1.96*se])


def paired_ratio(matching_failed, frontier_failed, distance):
    # Multinomial draws of the four joint outcomes are exactly the count-level
    # bootstrap of paired shots. Both decoder predictions stay paired.
    joint = 2*frontier_failed.astype(np.int64) + matching_failed
    counts = np.bincount(joint, minlength=4)
    rng = np.random.default_rng(2026091600 + distance)
    draws = rng.multinomial(len(joint), counts/len(joint), size=20000)
    frontier_rate = (draws[:, 2]+draws[:, 3])/len(joint)
    matching_rate = (draws[:, 1]+draws[:, 3])/len(joint)
    assert np.all(matching_rate > 0)
    ratios = np.array([convert(a, distance)/convert(b, distance)
                       for a, b in zip(frontier_rate, matching_rate)])
    return dict(joint_counts_both_correct_mwpm_only_fails_frontier_only_fails_both_fail=counts.tolist(),
                normalized_ler_ratio_ci95=np.quantile(ratios, [.025, .975]).tolist(),
                bootstrap_draws=len(draws), bootstrap_seed=2026091600+distance)


def load(distance):
    root = (FROZEN if distance < 11 else HERE) / f'd{distance}'
    request = json.loads((root/'request.json').read_text())
    saved = json.loads((root/'summary.json').read_text())
    assert request['shots'] == 100000
    assert request['parameters'] == dict(distance=distance, p=.003, patches=6,
                                          rounds=4*distance, yokes=2, noise='si1000', style='cz')
    for name, digest in request['sources'].items():
        path = FROZEN/name if distance < 11 else REPO/name
        assert sha256_file(path) == digest, path
    assert sha256_file(root/'results.npz') == saved['validation']['results_sha256']
    if distance >= 11:
        assert sha256_file(root/'sample.json') == request['sample']['manifest_sha256']
    with np.load(root/'results.npz') as archive:
        arrays = dict(archive)
    computed = summary(arrays, distance)
    for key, value in computed.items():
        assert saved[key] == value, (distance, key)
    assert arrays['predictions'].shape == (100000, 6)
    assert arrays['actual'].shape == arrays['matching'].shape == (100000,)
    for before, after in ((1, 2), (4, 5)):
        assert np.all(arrays['costs'][:, after] <= arrays['costs'][:, before]+1e-7)
    return saved, arrays


def plot(reports, analysis):
    fig, (ax, ratio_ax) = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    series = [('Correlated MWPM', None, '#2563a6', 'o'),
              ('Correlated UF', 0, '#b45309', 's'),
              ('Frontier UF + bridges', 5, '#19845b', '^')]
    for label, k, color, marker in series:
        values, intervals = [], []
        for d in DISTANCES:
            row = reports[d]['correlated_mwpm'] if k is None else reports[d]['rows'][k]
            values.append(row['normalized_ler'])
            ci = binomtest(row['failures'], 100000).proportion_ci(method='wilson')
            intervals.append([convert(ci.low, d), convert(ci.high, d)])
        values, intervals = np.array(values), np.array(intervals)
        ax.errorbar(DISTANCES, values, yerr=np.abs(intervals.T-values),
                    color=color, marker=marker, capsize=3, linewidth=1.7, label=label)
    ax.set(yscale='log', xlabel='Distance d', ylabel='Normalized logical error rate',
           title='100,000 shared shots per distance', xticks=DISTANCES)
    ax.grid(alpha=.2, which='both')
    ax.legend(frameon=False, fontsize=9)
    ratios = np.array([analysis[d]['frontier_to_mwpm_ratio'] for d in DISTANCES])
    intervals = np.array([analysis[d]['paired_ratio']['normalized_ler_ratio_ci95'] for d in DISTANCES])
    ratio_ax.errorbar(DISTANCES, ratios, yerr=np.abs(intervals.T-ratios),
                      color='#19845b', marker='^', capsize=3, linewidth=1.7)
    ratio_ax.axhline(1, color='#2563a6', linestyle='--', linewidth=1)
    ratio_ax.set(xlabel='Distance d', ylabel='Frontier UF + bridges / correlated MWPM',
                 title='Relative LER gap (paired 95% intervals)', xticks=DISTANCES)
    ratio_ax.grid(alpha=.2)
    fig.suptitle('SI1000 p=0.003 · 6 patches · 2 ideal yokes · rounds=4d', fontsize=12)
    fig.savefig(HERE/'distance_scaling.png', dpi=180)
    svg = HERE/'distance_scaling.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    reports, analysis = {}, {}
    for d in DISTANCES:
        report, arrays = load(d)
        reports[d] = report
        failures = arrays['predictions'] != arrays['actual'][:, None]
        matching_failed = arrays['matching'] != arrays['actual']
        mwpm = report['correlated_mwpm']['normalized_ler']
        uf = report['rows'][0]['normalized_ler']
        frontier = report['rows'][5]['normalized_ler']
        analysis[d] = dict(frontier_to_mwpm_ratio=frontier/mwpm,
                           frontier_minus_mwpm_ler=frontier-mwpm,
                           relative_excess_over_mwpm=frontier/mwpm-1,
                           relative_reduction_from_uf=1-frontier/uf,
                           fraction_of_uf_mwpm_gap_closed=(uf-frontier)/(uf-mwpm),
                           paired_ratio=paired_ratio(matching_failed, failures[:, 5], d),
                           compared_with_uf=paired_counts(failures[:, 0], failures[:, 5]),
                           compared_with_frontier=paired_counts(failures[:, 3], failures[:, 5]),
                           compared_with_mwpm=paired_counts(matching_failed, failures[:, 5]))
    scaling = []
    for before, after in zip(DISTANCES[:-1], DISTANCES[1:]):
        entry = dict(before=before, after=after)
        for label, key in [('correlated_mwpm', None), ('correlated_uf', 0), ('frontier_bridges', 5)]:
            a = reports[before]['correlated_mwpm'] if key is None else reports[before]['rows'][key]
            b = reports[after]['correlated_mwpm'] if key is None else reports[after]['rows'][key]
            entry[label] = dict(ler_suppression=a['normalized_ler']/b['normalized_ler'],
                               shot_failure_suppression=a['failures']/b['failures'])
        entry['relative_ratio_change'] = analysis[after]['frontier_to_mwpm_ratio']/analysis[before]['frontier_to_mwpm_ratio']-1
        entry['absolute_gap_change'] = analysis[after]['frontier_minus_mwpm_ler']/analysis[before]['frontier_minus_mwpm_ler']-1
        scaling.append(entry)
    write_json(HERE/'scaling.json', dict(distances=analysis, adjacent_distance_steps=scaling,
               normalization='sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=24*d, values=8)',
               intervals='Wilson marginal LER intervals; percentile paired-shot multinomial bootstrap for LER ratios (20,000 draws). Exploratory, unadjusted.'))
    plot(reports, analysis)
    full_mwpm_suppression = reports[7]['correlated_mwpm']['normalized_ler']/reports[13]['correlated_mwpm']['normalized_ler']
    full_frontier_suppression = reports[7]['rows'][5]['normalized_ler']/reports[13]['rows'][5]['normalized_ler']
    lines = ['# Frozen frontier UF experiment: distance scaling through d=13', '',
             f'From d=7 to d=13, normalized LER falls by {full_frontier_suppression:.2f}× for '
             f'Frontier UF + bridges and {full_mwpm_suppression:.2f}× for correlated MWPM. '
             f'The Frontier/MWPM LER ratio changes from {analysis[7]["frontier_to_mwpm_ratio"]:.3f} '
             f'to {analysis[13]["frontier_to_mwpm_ratio"]:.3f}.', '',
             'The d=11 and d=13 runs each use 100,000 newly sampled shots, shared by all decoders at that distance. '
             'The d=7/9 results are reused unchanged. SI1000 p=0.003, six patches, two ideal yokes, '
             'CZ circuits, and rounds=4d throughout.', '',
             'The native decoder and all settings are byte-for-byte the prior experiment: unchanged first-pass UF '
             'and correlation rules; frontier rates rounded down to powers of two; bridge eligibility at 0.5 nats '
             'remaining growth; two rounds of mutually best improving bridges. Neither truth nor MWPM predictions '
             'enter UF decoding. The new distances were evaluated without outcome-based parameter changes.', '',
             '## LER and the remaining gap', '',
             'Normalized LER uses `sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=24*d, values=8)`, '
             'the same convention as the prior results. It is a normalization of whole-shot failure probability; '
             'raw counts below avoid relying on the normalization to assess the trend.', '',
             '| d | Rounds | Correlated MWPM | Correlated UF | Frontier UF + bridges | Frontier/MWPM (95% paired CI) | Excess over MWPM |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for d in DISTANCES:
        r, a = reports[d], analysis[d]
        lo, hi = a['paired_ratio']['normalized_ler_ratio_ci95']
        lines.append(f"| {d} | {4*d} | {r['correlated_mwpm']['normalized_ler']:.9f} | "
                     f"{r['rows'][0]['normalized_ler']:.9f} | {r['rows'][5]['normalized_ler']:.9f} | "
                     f"{a['frontier_to_mwpm_ratio']:.3f} [{lo:.3f}, {hi:.3f}] | {100*a['relative_excess_over_mwpm']:.1f}% |")
    lines += ['', f'![Distance scaling]({HERE.name}/distance_scaling.png)', '',
              'A suppression factor above one means LER fell when distance increased. '
              'The relative gap and absolute gap answer different questions:', '',
              '| Distance step | MWPM suppression | UF suppression | Frontier + bridges suppression | Change in Frontier/MWPM ratio | Change in absolute LER gap |',
              '|---|---:|---:|---:|---:|---:|']
    for row in scaling:
        lines.append(f"| {row['before']} → {row['after']} | {row['correlated_mwpm']['ler_suppression']:.3f}× | "
                     f"{row['correlated_uf']['ler_suppression']:.3f}× | {row['frontier_bridges']['ler_suppression']:.3f}× | "
                     f"{100*row['relative_ratio_change']:+.1f}% | {100*row['absolute_gap_change']:+.1f}% |")
    lines += ['', '## All decoder variants', '',
              'Each cell is whole-shot failures out of 100,000, followed by normalized LER. '
              'A shot fails if any of its 12 predicted observable bits differs from the sampled truth.', '',
              '| Decoder | d=7 | d=9 | d=11 | d=13 |', '|---|---:|---:|---:|---:|']
    for k, label in [(None, 'Correlated MWPM'), *enumerate(LABELS)]:
        cells = []
        for d in DISTANCES:
            row = reports[d]['correlated_mwpm'] if k is None else reports[d]['rows'][k]
            cells.append(f"{row['failures']:,}; {row['normalized_ler']:.9f}")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '## Paired effect of the combined algorithm', '',
              '| d | LER reduction from correlated UF | Original UF–MWPM LER gap closed | Repairs / regressions vs UF | Additional repairs / regressions vs frontier alone |',
              '|---:|---:|---:|---:|---:|']
    for d, a in analysis.items():
        base, inc = a['compared_with_uf'], a['compared_with_frontier']
        lines.append(f"| {d} | {100*a['relative_reduction_from_uf']:.1f}% | "
                     f"{100*a['fraction_of_uf_mwpm_gap_closed']:.1f}% | "
                     f"{base['repaired']:,} / {base['regressed']:,} | {inc['repaired']:,} / {inc['regressed']:,} |")
    lines += ['', '## Algorithmic work at the new distances', '',
              'These are second-pass algorithmic counters, not hardware cycles or FPGA latency. '
              'The event simulator uses floating-point weights and a heap. Tree message counts cover only '
              'the min-sum/traceback stage; they exclude growth and arbitration communication.', '',
              '| d | Growth | Mean epochs | Mean edge evaluations | Mean frontier visits | Mean largest cluster | Mean bridges | p99 final tree depth |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for d in (11, 13):
        hw = reports[d]['hardware_proxies']
        for k, label in enumerate(('Original UF', 'Frontier UF')):
            lines.append(f"| {d} | {label} | {hw['epochs']['mean'][k]:.1f} | "
                         f"{hw['edge_evaluations']['mean'][k]:,.0f} | {hw['frontier_visits']['mean'][k]:,.0f} | "
                         f"{hw['max_cluster']['mean'][k]:.1f} | {hw['bridges']['mean'][k]:.2f} | "
                         f"{hw['tree_depth']['p99'][3*k+2]:.0f} |")
    lines += ['', '## Validation and limits', '',
              'The current generator reproduces the earlier d=9 circuit and detector-error-model hashes. '
              'New samples retain exact circuit, model, packed data and version identities. Every new UF '
              'correction passes full-syndrome validation: 1,200,000 checks across six variants and two distances. '
              'All seven outputs also pass the ideal-yoke parity checks.', '',
              'At each new distance, 32 predetermined random shots pass native/Python first-pass physical-edge '
              'and correlated-UF final-forest/correction comparisons. Four tree-cost outputs per shot agree '
              'with the independent Python forest solver. Two shots per distance additionally agree with an '
              'independent full-edge scan of frontier growth. Native outputs agree bit-for-bit between one '
              'and 64 threads; the same 32 MWPM predictions agree between serial unpacked and parallel packed inputs. '
              'The new 100k native baselines are not independently decoded in full by Python. The frozen kernel '
              'previously matched all 200k saved d=7/9 baseline predictions and passed its exhaustive small-graph checks.', '',
              'Intervals are exploratory and unadjusted for multiple comparisons. The ratio intervals resample '
              'paired decoder outcomes, preserving their dependence. Earlier d=7/9 samples informed algorithm '
              'development; d=11/13 use fresh samples with the algorithm frozen. Four distances at one p and '
              'rounds=4d describe this finite-distance trend, not an asymptotic scaling law or a threshold estimate.', '',
              f'[Reproduction instructions]({HERE.name}/README.md). '
              f'[Full scaling statistics]({HERE.name}/scaling.json).', '']
    report_path = HERE.with_suffix('.md')
    report_path.write_text('\n'.join(lines))
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json')
    write_json(HERE/'artifact_manifest.json', dict(
        sha256={str(p.relative_to(HERE)): sha256_file(p) for p in paths},
        report_sha256=sha256_file(report_path),
        reused_results={f'd{d}': sha256_file(FROZEN/f'd{d}'/'results.npz') for d in (7, 9)}))
    print(report_path)
    print(json.dumps(dict(distances=analysis, adjacent_distance_steps=scaling), indent=2))


if __name__ == '__main__':
    main()
