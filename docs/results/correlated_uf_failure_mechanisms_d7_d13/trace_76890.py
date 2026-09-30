"""Locate and intervene on blocked logical routes in complete d=7 shot 76890.

MWPM identifies candidate edges only for this offline diagnosis. No production
decoder or evaluation input is changed. This uses the full original shot,
unlike the older ten-fault reduction of the same source shot.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pymatching

from diagnose import HERE, RESULTS, helper, previous, sha256_file, write
from reachable_cases import forced_bit_costs, forced_matching_costs
from yoked.decoders import UnionFindDecoder
from yoked.decoders._union_find import _peel

GROWTH = RESULTS / 'correlated_uf_growth_diagnosis_d7_d9'
trace = helper('saved_growth_helpers', GROWTH / 'trace.py')
trees = helper('saved_tree_helpers', GROWTH / 'algorithms.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'full_shot_76890.json')
    args = parser.parse_args()
    a = previous.Analysis(7)
    shot = 76890
    syndrome = a.syndrome(shot)
    actual = previous.bitmask(a.record.actual[shot])
    first = a.uf._decode(syndrome)
    weights = a.weights(first.selected_edges)
    graph = a.regraph(weights)
    uf, growth = UnionFindDecoder(graph)._decode_state(syndrome)
    matching = a.matching_edges(syndrome, weights)
    assert a.mask(uf.selected_edges, syndrome) == 3116
    assert a.mask(matching, syndrome) == actual == 3236
    ufset, mset = set(uf.selected_edges), set(matching)
    components = [c for c in trace.difference_components(graph, ufset, mset) if c['mask']]
    cost = lambda edges: math.fsum(weights[e] for e in edges)

    def cluster(vertex):
        root = growth.find(vertex)
        return dict(root=root, vertices=growth.size[root], odd=bool(growth.parity[root]),
                    boundary=bool(growth.terminal[root]), active=bool(growth.active(root)),
                    yokes=[d for d in (graph.num_detectors-2, graph.num_detectors-1)
                           if growth.find(d) == root])

    def solve(allowed):
        allowed = sorted(allowed)
        matcher = pymatching.Matching.from_check_matrix(
            a.matrix.check_matrix[:, allowed], weights=weights[allowed],
            faults_matrix=a.matrix.faults_matrix[:, allowed], merge_strategy='disallow')
        selected = a.matrix.edge_ids_of(matcher.decode_to_edges_array(syndrome))
        return dict(prediction=a.mask(selected, syndrome), cost=cost(selected), selected_edges=selected)

    rows = []
    for component in components:
        m_edges = sorted(set(component['edges']) & mset)
        missing = []
        for e in m_edges:
            u, v = graph.endpoints[e]
            if growth.find(u) == growth.find(v):
                continue
            spent = min(weights[e], growth.grown[e] + growth.rate[e]*(growth.time-growth.settled_at[e]))
            missing.append(dict(edge=e, endpoints=[u, v], boundary_edge=graph.edges[e][1] is None,
                                weight=float(weights[e]), grown=float(spent), remaining=float(weights[e]-spent),
                                first_cluster=cluster(u), second_cluster=cluster(v)))
        assert len(missing) == 1 and missing[0]['edge'] == 9726
        augmented = list(growth.forest) + [missing[0]['edge']]
        selected, optimum = trees.minimum_cost_forest(graph, syndrome, augmented)
        independent = solve(augmented)
        assert math.isclose(optimum, independent['cost'], rel_tol=1e-12, abs_tol=1e-8)
        assert a.mask(selected, syndrome) == independent['prediction']
        peeled, peel_mask = _peel(graph, syndrome, augmented)
        assert a.mask(peeled, syndrome) == peel_mask
        augmented_basis = previous.logical_basis(graph, augmented)
        assert len(augmented_basis) == 1
        assert previous.contains(augmented_basis, uf.observable_mask ^ actual)
        bit = (uf.observable_mask ^ actual).bit_length() - 1
        class_costs = forced_bit_costs(graph, syndrome, weights, augmented, bit)
        independent_class_costs = forced_matching_costs(graph, syndrome, weights, augmented, bit)
        np.testing.assert_allclose(class_costs, independent_class_costs, rtol=1e-12, atol=1e-8)
        added = sorted(set(m_edges) - set(growth.forest))
        restored = solve(set(growth.forest) | set(m_edges))
        exchanged = ufset ^ set(component['edges'])
        assert a.mask(exchanged, syndrome) == restored['prediction'] == actual
        assert math.isclose(cost(exchanged), restored['cost'], rel_tol=1e-12, abs_tol=1e-8)
        rows.append(dict(logical_mask=component['mask'],
                         UF_component_edges=sorted(set(component['edges']) & ufset),
                         MWPM_component_edges=m_edges,
                         UF_component_cost=cost(set(component['edges']) & ufset),
                         MWPM_component_cost=cost(m_edges),
                         missing_intercluster_edges=missing,
                         component_edges_missing_from_forest=[
                             dict(edge=e, endpoints=list(graph.endpoints[e]),
                                  original_roots=[growth.find(v) for v in graph.endpoints[e]])
                             for e in added],
                         one_edge_ordinary_peeling=dict(prediction=peel_mask, cost=cost(peeled)),
                         one_edge_minimum_cost_forest=independent,
                         one_edge_forest_allows_truth=previous.contains(
                             augmented_basis, uf.observable_mask ^ actual),
                         one_edge_forest_class_costs=dict(
                             correct=class_costs[(actual >> bit) & 1],
                             wrong=class_costs[(uf.observable_mask >> bit) & 1]),
                         all_component_MWPM_edges_added=restored))
    original_optimum = solve(growth.forest)
    endpoints = np.asarray(graph.endpoints)
    owner = np.array([growth.find(v) for v in range(len(graph.adjacency))])
    allowed = np.flatnonzero(owner[endpoints[:, 0]] == owner[endpoints[:, 1]])
    basis = previous.logical_basis(graph, allowed)
    assert not previous.contains(basis, uf.observable_mask ^ actual)
    source_files = [Path(__file__), HERE / 'reachable_cases.py',
                    GROWTH / 'trace.py', GROWTH / 'algorithms.py',
                    RESULTS / 'correlated_uf_mwpm_failure_analysis_d7_d9/analyze.py',
                    HERE.parents[2] / 'src/yoked/decoders/_union_find.py']
    result = dict(shot=shot, distance=7, pattern='complete original saved shot',
                  actual=actual, UF_prediction=uf.observable_mask, MWPM_prediction=a.mask(matching),
                  UF_total_cost=cost(uf.selected_edges), MWPM_total_cost=cost(matching),
                  growth_end=growth.time, original_forest_optimum=original_optimum,
                  original_partition_logical_rank=len(basis), original_partition_allows_truth=False,
                  logical_components=rows, source=json.loads(json.dumps(a.run, default=dict)),
                  source_hashes={str(p.resolve()): sha256_file(p) for p in source_files},
                  every_correction_syndrome_checked=True,
                  boundary_edge_selection='MWPM oracle for diagnosis, with UF-conditioned weights fixed')
    write(args.output, result)
    print(json.dumps(dict(shot=shot, actual=actual, UF=uf.observable_mask,
                          one_edge_optima=[{k: v for k, v in row['one_edge_minimum_cost_forest'].items()
                                             if k != 'selected_edges'} for row in rows],
                          one_edge_peeling=[row['one_edge_ordinary_peeling'] for row in rows]), indent=2))


if __name__ == '__main__':
    main()
