"""Price both logical classes in the seven residual forests admitting truth.

Each selected forest has logical-change rank one. Constraining one observable
bit that distinguishes the two classes therefore fixes the full logical class.
Truth is used only for this diagnostic, never for decoding the evaluation set.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pymatching
import scipy.sparse
import stim

import diagnose as analysis
from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.hierarchical._matching_gaps import _CheckMatrixGraph


def forced_bit_costs(graph, syndrome, weights, forest, bit):
    """Tree DP carries parent-edge parity and one accumulated observable bit."""
    adjacency = {}
    for e in forest:
        u, v = graph.endpoints[e]
        adjacency.setdefault(u, []).append((v, e))
        adjacency.setdefault(v, []).append((u, e))
    parent, order, roots, children = {}, [], [], {}
    for root in adjacency:
        if root in parent:
            continue
        parent[root] = None
        roots.append(root)
        pending = [root]
        while pending:
            u = pending.pop()
            order.append(u)
            children[u] = []
            for v, e in adjacency[u]:
                if v == parent[u]:
                    continue
                assert v not in parent, 'Expected a forest'
                parent[v] = u
                children[u].append((v, e))
                pending.append(v)
    assert all(int(v) in parent for v in np.flatnonzero(syndrome))
    dp = {}
    for u in reversed(order):
        costs = [[0., math.inf], [math.inf, math.inf]]
        for v, e in children[u]:
            following = [[math.inf, math.inf], [math.inf, math.inf]]
            label = (graph.edges[e][3] >> bit) & 1
            for parity in (0, 1):
                for logical in (0, 1):
                    for selected in (0, 1):
                        for child_logical in (0, 1):
                            p = parity ^ selected
                            q = logical ^ child_logical ^ (selected * label)
                            cost = costs[parity][logical] + dp[v][selected][child_logical] + selected*weights[e]
                            following[p][q] = min(following[p][q], cost)
            costs = following
        if u < graph.num_detectors:
            dp[u] = [costs[int(syndrome[u]) ^ p] for p in (0, 1)]
        else:
            best = [min(costs[0][q], costs[1][q]) for q in (0, 1)]
            dp[u] = [best, best]
    total = [0., math.inf]
    for root in roots:
        total = [min(total[p] + dp[root][0][q ^ p] for p in (0, 1)) for q in (0, 1)]
    return total


def forced_matching_costs(graph, syndrome, weights, forest, bit):
    """Independent construction: move tree-edge labels onto boundary vertices.

    Path labels give vertex potentials. The observable then equals a fixed
    syndrome term XOR the total correction parity at label-one boundaries.
    Identify those boundaries as one extra constrained detector. Label-zero
    boundaries remain free. The result is still a graphlike matching problem.
    """
    adjacency = {}
    for e in forest:
        u, v = graph.endpoints[e]
        label = (graph.edges[e][3] >> bit) & 1
        adjacency.setdefault(u, []).append((v, label))
        adjacency.setdefault(v, []).append((u, label))
    potential = {}
    for root in adjacency:
        if root in potential:
            continue
        potential[root] = 0
        pending = [root]
        while pending:
            u = pending.pop()
            for v, label in adjacency[u]:
                value = potential[u] ^ label
                if v in potential:
                    assert potential[v] == value
                else:
                    potential[v] = value
                    pending.append(v)
    fixed = 0
    for u in np.flatnonzero(syndrome):
        fixed ^= potential[int(u)]
    rows, cols, edge_map = [], [], {}
    for col, e in enumerate(forest):
        mapped = []
        for vertex in graph.endpoints[e]:
            if vertex < graph.num_detectors:
                rows.append(vertex)
                cols.append(col)
                mapped.append(vertex)
            elif potential[vertex]:
                rows.append(graph.num_detectors)
                cols.append(col)
                mapped.append(graph.num_detectors)
        key = (mapped[0], -1) if len(mapped) == 1 else tuple(sorted(mapped))
        assert key not in edge_map
        edge_map[key] = e
    check = scipy.sparse.csc_matrix((np.ones(len(rows)), (rows, cols)),
                                   shape=(graph.num_detectors+1, len(forest)))
    matcher = pymatching.Matching.from_check_matrix(check, weights=weights[forest], merge_strategy='disallow')
    costs = []
    for target in (0, 1):
        returned = matcher.decode_to_edges_array(np.append(syndrome, target ^ fixed))
        selected = [edge_map[(int(u), -1) if v == -1 else tuple(sorted((int(u), int(v))))]
                    for u, v in returned]
        response = np.zeros(graph.num_detectors, dtype=np.uint8)
        observable = 0
        for e in selected:
            u, v, _, label = graph.edges[e]
            response[u] ^= 1
            if v is not None:
                response[v] ^= 1
            observable ^= (label >> bit) & 1
        np.testing.assert_array_equal(response, syndrome)
        assert observable == target
        # Recompute from original weights, avoiding PyMatching's quantized
        # return_weight total when comparing high precision path costs.
        costs.append(math.fsum(weights[e] for e in selected))
    return costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=analysis.HERE)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for distance in (7, 9, 11, 13):
        directory = args.root / f'd{distance}'
        rows = json.loads((directory / 'rows.json').read_text())
        cases = [r for r in rows if not r['variants']['frontier_bridges']['correct']
                 and r['variants']['frontier_bridges']['partition_allows_truth']]
        if not cases:
            continue
        source = json.loads((directory / 'verification.json').read_text())['source']
        sample = Path(source['directory'])
        dem = stim.DetectorErrorModel.from_file(sample / 'model.dem')
        graph = DecodingGraph.from_dem(dem)
        matrix = _CheckMatrixGraph(graph)
        flat_rules = correlation_rules_from_dem(graph, dem)
        rules = index_rules_by_source(graph, flat_rules)
        packed = np.load(sample / 'detectors_packed.npy', mmap_mode='r')
        native = analysis.experiment.Native(args.library, graph, flat_rules)
        try:
            out, flags = native.decode(packed[[r['shot'] for r in cases]], threads=1, audit=True)
        finally:
            native.close()
        for k, row in enumerate(cases):
            state = row['variants']['frontier_bridges']
            assert state['forest_allows_truth'] and state['forest_logical_rank'] == 1
            forest = np.flatnonzero(flags[k] & (1 << 10)).tolist()
            first = np.flatnonzero(flags[k] & 1).tolist()
            adjusted = apply_correlation_rules(matrix.weights, rules, first)
            weights = matrix.weights if adjusted is None else np.asarray(adjusted)
            syndrome = np.unpackbits(packed[row['shot']], count=graph.num_detectors, bitorder='little')
            bit = (state['prediction'] ^ row['actual']).bit_length() - 1
            costs = forced_bit_costs(graph, syndrome, weights, forest, bit)
            independent_costs = forced_matching_costs(graph, syndrome, weights, forest, bit)
            np.testing.assert_allclose(costs, independent_costs, rtol=1e-10, atol=1e-7)
            truth_cost = costs[(row['actual'] >> bit) & 1]
            wrong_cost = costs[(state['prediction'] >> bit) & 1]
            assert math.isclose(min(costs), float(out[k, 5, 1]), rel_tol=1e-12, abs_tol=1e-8)
            # Independent MWPM verifies the unrestricted optimum on this forest.
            restricted = pymatching.Matching.from_check_matrix(
                matrix.check_matrix[:, forest], weights=weights[forest],
                faults_matrix=matrix.faults_matrix[:, forest], merge_strategy='disallow')
            selected = matrix.edge_ids_of(restricted.decode_to_edges_array(syndrome))
            assert analysis.correction(graph, matrix, syndrome, selected) == state['prediction']
            assert math.isclose(math.fsum(weights[e] for e in selected), wrong_cost, rel_tol=1e-12, abs_tol=1e-8)
            assert row['same_weight_MWPM_correct']
            global_cost = float(out[k, 0, 1]) - row['UF_minus_MWPM_cost']
            results.append(dict(distance=distance, shot=row['shot'], distinguishing_observable=bit,
                                forest_wrong_class_cost=wrong_cost, forest_true_class_cost=truth_cost,
                                global_MWPM_true_class_cost=global_cost,
                                true_class_penalty_within_forest=truth_cost-wrong_cost,
                                true_class_penalty_from_missing_edges=truth_cost-global_cost))
    analysis.write(args.root / 'reachable_cases.json', dict(
        cases=results, script_sha256=analysis.sha256_file(__file__),
        independent_restricted_MWPM_checks=len(results),
        independent_forced_class_MWPM_checks=2*len(results)))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
