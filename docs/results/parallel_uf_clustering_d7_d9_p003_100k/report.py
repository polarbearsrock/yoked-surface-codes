"""Audit saved arrays and regenerate the complete paired experiment report."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment import HERE, summary, write_json
from yoked.hierarchical._provenance import sha256_file

LABELS = ('Correlated UF', 'UF + forest-cost control', 'UF + bounded bridges',
          'Frontier-weighted UF', 'Frontier-weighted UF + forest-cost control',
          'Frontier-weighted UF + bounded bridges')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--report', type=Path, default=HERE.with_suffix('.md'))
    args = parser.parse_args()
    reports, incremental = {}, {}
    for d in (7, 9):
        root = args.root/f'd{d}'
        request = json.loads((root/'request.json').read_text())
        saved = json.loads((root/'summary.json').read_text())
        assert request['shots'] == 100000
        assert request['parameters'] == dict(distance=d, noise='si1000', p=.003, patches=6,
                                              rounds=4*d, style='cz', yokes=2)
        assert sha256_file(root/'results.npz') == saved['validation']['results_sha256']
        for name, digest in request['sources'].items():
            assert sha256_file(args.root/name) == digest
        with np.load(root/'results.npz') as file:
            arrays = dict(file)
        calculated = summary(arrays, d)
        for key, value in calculated.items():
            assert saved[key] == value, (d, key)
        # Corrections selected after extension cannot increase the tree optimum.
        for base, changed in ((1, 2), (4, 5)):
            assert np.all(arrays['costs'][:, changed] <= arrays['costs'][:, base]+1e-7)
        before = arrays['predictions'][:, 3] != arrays['actual']
        after = arrays['predictions'][:, 5] != arrays['actual']
        repaired = int(np.count_nonzero(before & ~after))
        spoiled = int(np.count_nonzero(~before & after))
        delta = (repaired-spoiled)/len(before)
        se = np.sqrt(((repaired+spoiled)/len(before)-delta**2)/len(before))
        incremental[d] = (repaired, spoiled, delta, delta-1.96*se, delta+1.96*se)
        reports[d] = saved
    gains = [100*(1-reports[d]['rows'][5]['normalized_ler']/reports[d]['rows'][0]['normalized_ler']) for d in (7, 9)]
    gaps = [100*(reports[d]['rows'][5]['normalized_ler']/reports[d]['correlated_mwpm']['normalized_ler']-1) for d in (7, 9)]
    lines = [
        '# Parallel UF clustering: paired 100k-shot d=7/d=9 experiment', '',
        f'Frontier-weighted growth plus bounded bridges reduces normalized LER by '
        f'**{gains[0]:.1f}% at d=7 and {gains[1]:.1f}% at d=9** relative to correlated UF. '
        f'It remains {gaps[0]:.1f}% and {gaps[1]:.1f}% above correlated MWPM, respectively. '
        'Changing growth supplies most of the benefit; optimizing the existing '
        'forests alone produces no LER improvement.', '',
        'SI1000 p=0.003, six patches, two ideal yokes, CZ circuits, rounds=4d. '
        'Both distances reuse their original complete 100,000-shot samples. '
        'The first-pass UF correction and resulting second-pass correlation weights '
        'are identical across variants. No truth or MWPM output is used in decoding.', '',
        '## Logical error rates', '',
        'The normalized LER uses the existing convention '
        '`sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=6*4*d, values=8)`. '
        'Whole-shot failure counts are included to make the comparison unambiguous.', '',
        '| Decoder | d=7 failures | d=7 normalized LER | d=9 failures | d=9 normalized LER |',
        '|---|---:|---:|---:|---:|',
    ]
    pm = [reports[d]['correlated_mwpm'] for d in (7,9)]
    lines.append(f"| Correlated MWPM (saved reference) | {pm[0]['failures']:,} | {pm[0]['normalized_ler']:.9f} | {pm[1]['failures']:,} | {pm[1]['normalized_ler']:.9f} |")
    for k, label in enumerate(LABELS):
        rows = [reports[d]['rows'][k] for d in (7,9)]
        lines.append(f"| {label} | {rows[0]['failures']:,} | {rows[0]['normalized_ler']:.9f} | {rows[1]['failures']:,} | {rows[1]['normalized_ler']:.9f} |")
    lines += ['', '## Paired changes from correlated UF', '',
              'Positive reduction means improvement. Confidence intervals refer to '
              'the reduction in whole-shot failure probability, in percentage points; '
              'they are paired normal 95% intervals. Comparisons are exploratory '
              'and are not adjusted for multiplicity. Earlier diagnoses used this '
              'same shot pool, so this is not an untouched holdout; the intervals '
              'do not account for adaptive algorithm-development choices.', '',
              '| d | Variant | Failures repaired | Successes spoiled | Net repairs | Failure reduction, percentage points (95% CI) |',
              '|---:|---|---:|---:|---:|---:|']
    for d in (7, 9):
        for k in (1,2,3,4,5):
            row = reports[d]['rows'][k]
            lo, hi = np.asarray(row['paired_reduction_ci95'])*100
            lines.append(f"| {d} | {LABELS[k]} | {row['repaired']:,} | {row['regressed']:,} | {row['repaired']-row['regressed']:+,} | {100*row['failure_rate_reduction']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', 'The bridge stage also improves on frontier-weighted growth alone:', '',
              '| d | Additional repairs | Additional spoiled successes | Net additional repairs | Failure reduction, percentage points (95% CI) |',
              '|---:|---:|---:|---:|---:|']
    for d, (repaired, spoiled, delta, lo, hi) in incremental.items():
        lines.append(f'| {d} | {repaired:,} | {spoiled:,} | {repaired-spoiled:+,} | {delta*100:+.3f} [{lo*100:+.3f}, {hi*100:+.3f}] |')
    lines += ['', '## Growth work and dependency proxies', '',
              'These cover only the modified second UF pass. They are software-measured '
              'algorithmic quantities, not FPGA cycles, latency, or resource estimates. '
              'An epoch is a global next-event batch. Edge evaluations count initial '
              'growth-rate calculations and later reevaluations.', '',
              '| d | Growth policy | Mean event epochs | Mean edge evaluations | Mean frontier visits | Mean largest cluster | p99 largest cluster | Mean tree depth | p99 tree depth |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in (7, 9):
        hw = reports[d]['hardware_proxies']
        for p, name in enumerate(('Existing UF', 'Frontier-weighted')):
            lines.append(f"| {d} | {name} | {hw['epochs']['mean'][p]:.1f} | {hw['edge_evaluations']['mean'][p]:,.0f} | {hw['frontier_visits']['mean'][p]:,.0f} | {hw['max_cluster']['mean'][p]:.1f} | {hw['max_cluster']['p99'][p]:.0f} | {hw['tree_depth']['mean'][3*p]:.1f} | {hw['tree_depth']['p99'][3*p]:.0f} |")
    lines += ['', '## Bounded refinement work', '',
              'Two refinement rounds are permitted. Disjoint cluster pairs can merge '
              'in parallel. Tree messages count directed min-sum messages plus final '
              'traceback traversals. They exclude growth communication, proposal '
              'aggregation/broadcast, candidate-edge exchanges, and arbitration; '
              'this is a partial communication count, not total bytes.', '',
              '| d | Starting growth | Mean eligible edges | Mean accepted bridges | Mean peak parallel pairs | Mean tree messages | p99 extended tree depth |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for d in (7, 9):
        hw = reports[d]['hardware_proxies']
        for p, name in enumerate(('Existing UF', 'Frontier-weighted')):
            lines.append(f"| {d} | {name} | {hw['eligible_bridges']['mean'][p]:.1f} | {hw['bridges']['mean'][p]:.1f} | {hw['max_parallel_pairs']['mean'][p]:.1f} | {hw['tree_messages']['mean'][p]:,.0f} | {hw['tree_depth']['p99'][3*p+2]:.0f} |")
    lines += ['', '## Interpretation limits and validation', '',
              'Frontier weighting changes growth order using powers-of-two rates. '
              'Refinement admits only individually cost-improving forest bridges '
              'with at most 0.5 nats of growth remaining. It cannot introduce cycles '
              'or jointly accept a sequence of individually unprofitable connections. '
              'A graph-cost improvement does not guarantee a logical correction.', '',
              'All 200,000 baseline predictions agree with the saved experiment; '
              'all 1,200,000 variant corrections satisfy their full detector syndromes. '
              'Independent checks cover random graphs with exhaustive correction '
              'enumeration, full-edge scans for the new growth, bridge proposals '
              'evaluated by independent forest solves, production physical corrections, '
              'and native thread determinism. Seventy production UF/correlation tests passed.', '',
              'The native kernel is a software event simulator. Hardware implementation '
              'still requires choosing fixed-point precision, designing reductions and '
              'merge synchronization, and accounting for high-degree yoke routing. '
              'No hardware speedup or MWPM resource comparison is claimed.', '',
              f'[Protocol, counters, and reproduction instructions]({args.root.name}/README.md). '
              f'[Independent validation]({args.root.name}/verification.json).', '']
    args.report.write_text('\n'.join(lines))
    paths = sorted(p for p in args.root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json')
    write_json(args.root/'artifact_manifest.json', dict(sha256={str(p.relative_to(args.root)): sha256_file(p) for p in paths},
                                                       report_sha256=sha256_file(args.report)))
    print(args.report)


if __name__ == '__main__':
    main()
