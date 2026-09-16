"""Controlled checks on reduced and complete failure examples.

Alternative tie order and growth schedules are diagnostic interventions, not
production decoders. One-edge augmentation uses MWPM to identify an edge and
is therefore an oracle diagnostic, not an implementable edge-selection rule.
"""
import argparse
import inspect
import heapq
import json
import math
from pathlib import Path
import textwrap

import numpy as np
import pymatching

from trace import Analysis, PREVIOUS, TracedGrowth, bitmask, trace_syndrome, write
from algorithms import scan_growth, minimum_cost_forest
from yoked.decoders import UnionFindDecoder
from yoked.decoders._union_find import _Growth, _peel


def reordered_growth(priority):
    source = textwrap.dedent(inspect.getsource(_Growth._merge_group))
    assert 'for e in sorted(edges):' in source
    source = source.replace('for e in sorted(edges):',
                            'for e in sorted(edges, key=lambda e: priority[e]):')
    namespace = {'priority': priority, 'heapq': heapq}
    exec(source, namespace)
    return type('ReorderedGrowth', (_Growth,), {'_merge_group': namespace['_merge_group']})


def check_case(a, syndrome, actual, name, out, reduced=False):
    first = a.uf._decode(syndrome)
    weights = a.weights(first.selected_edges)
    graph = a.regraph(weights)
    base, growth = UnionFindDecoder(graph)._decode_state(syndrome)
    matching = a.matching_edges(syndrome, weights)
    assert a.mask(matching, syndrome) == actual != base.observable_mask
    trace = json.loads((out / f'{name}.json').read_text())
    missing = [r['edge'] for r in trace['summary']['blocked_MWPM_edges']]
    assert len(missing) == 1
    augmented = list(base.forest_edges) + missing
    selected, cost = minimum_cost_forest(graph, syndrome, augmented)
    peel, peel_mask = _peel(graph, syndrome, augmented)
    exact = pymatching.Matching.from_check_matrix(
        a.matrix.check_matrix[:, augmented], weights=weights[augmented],
        faults_matrix=a.matrix.faults_matrix[:, augmented], merge_strategy='disallow')
    exact_selected = a.matrix.edge_ids_of(exact.decode_to_edges_array(syndrome))
    exact_cost = math.fsum(weights[e] for e in exact_selected)
    assert math.isclose(exact_cost, cost, rel_tol=1e-9, abs_tol=1e-5)
    assert a.mask(peel, syndrome) == peel_mask
    row = dict(case=name, distance=a.distance, reduced=reduced, actual=actual,
               UF_mask=base.observable_mask, single_missing_edge=missing[0],
               augmented_peeling_mask=peel_mask, augmented_DP_mask=a.mask(selected, syndrome),
               augmented_DP_cost=cost, independent_matching_cost=exact_cost,
               total_MWPM_cost=math.fsum(weights[e] for e in matching))
    rng = np.random.default_rng(20260915)
    labels = np.array([growth.find(v) for v in range(len(graph.adjacency))])
    orders = [('reverse', -np.arange(len(graph.edges)))] + [
        (f'seeded_shuffle_{k}', rng.permutation(len(graph.edges))) for k in range(16)]
    tie_rows = []
    for label, priority in orders:
        g = reordered_growth(priority)(graph, syndrome)
        g.run()
        selected, mask = _peel(graph, syndrome, g.forest)
        a.mask(selected, syndrome)
        groups = {}
        for v in range(len(graph.adjacency)):
            groups.setdefault(g.find(v), set()).add(int(labels[v]))
        same_partition = len(groups) == len(set(labels)) and all(len(x) == 1 for x in groups.values())
        assert same_partition
        tie_rows.append(dict(order=label, mask=mask, same_partition=same_partition,
                             same_forest_edges=set(g.forest) == set(base.forest_edges)))
    row['tie_order_checks'] = tie_rows
    if reduced:
        row['schedules'] = {}
        for mode in ('standard', 'passive_yoke', 'frontier_normalized'):
            result = scan_growth(graph, syndrome, mode=mode)
            assert result['valid']
            if mode == 'standard':
                assert result['forest'] == list(base.forest_edges)
            chosen, mask = _peel(graph, syndrome, result['forest'])
            a.mask(chosen, syndrome)
            row['schedules'][mode] = dict(mask=mask,
                cost=math.fsum(weights[e] for e in chosen),
                time=result['time'], batches=result['batches'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a = Analysis(7)
    rows = []
    for shot in (44875, 76890):
        reduced = json.loads((PREVIOUS/f'reduced_{shot}.json').read_text())
        syndrome = np.zeros(a.graph.num_detectors, dtype=bool)
        syndrome[reduced['defect_ids']] = True
        rows.append(check_case(a, syndrome, reduced['reduced']['actual_mask'],
                               f'trace_{shot}', args.out, reduced=True))
        print(json.dumps(dict(completed=shot)), flush=True)
    # A valid single-yoke syndrome exposes a deadlock in the naive passive rule.
    syndrome = np.zeros(a.graph.num_detectors, dtype=bool)
    syndrome[-2] = True
    baseline = a.uf._decode(syndrome)
    a.mask(baseline.selected_edges, syndrome)
    passive = scan_growth(a.graph, syndrome, mode='passive_yoke')
    assert not passive['valid']
    passive_counterexample = dict(fired_detectors=[a.graph.num_detectors-2],
        standard_valid_correction=list(baseline.selected_edges), passive_result=passive)
    for distance, shot in ((7, 6463), (9, 34624)):
        if distance != a.distance:
            a = Analysis(distance)
        syndrome = a.syndrome(shot)
        actual = bitmask(a.record.actual[shot])
        prefix = f'full_d{distance}_'
        trace_syndrome(a, syndrome, actual, shot, args.out, prefix=prefix)
        rows.append(check_case(a, syndrome, actual, f'{prefix}{shot}', args.out))
        print(json.dumps(dict(completed=shot, distance=distance)), flush=True)
    write(args.out/'case_checks.json', dict(cases=rows,
        passive_yoke_counterexample=passive_counterexample))


if __name__ == '__main__':
    main()
