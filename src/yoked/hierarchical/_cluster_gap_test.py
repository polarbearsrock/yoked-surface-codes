"""Tests for the cluster-gap soft output (_cluster_gap.py).

Checks: the cluster gap on an empty syndrome equals the cost of the cheapest
odd logical walk (a brute-force enumeration, since no cluster has formed);
a cluster that already spans both boundary terminals has zero gap for the
observable it carries; partially grown edges are charged exactly their
remaining growth (`weight - grown`, floored at zero, zero within a cluster),
matching a hand-computed fixture and a brute-force walk over those same
costs; Dijkstra's shortest odd walk agrees with brute force on random small
graphs, with and without partial growth; a graph where an observable's
flipping edges do not lie in exactly one connected component is rejected;
and, on the six-patch fixture, the reference bits equal plain UF's
prediction while every cluster gap is finite and nonnegative and every
Dijkstra state count is positive.
"""
import collections
import math

import numpy as np
import pytest

from yoked.decoders import DecodingGraph, UnionFindDecoder
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._fixtures import yoked_fixture


def _brute_force_odd_walk(graph, costs, observable):
    """Minimum cost over all simple paths in the state graph from (B, 0) to (B, 1)."""
    boundary = graph.num_detectors
    adjacency = collections.defaultdict(list)
    for e, (u, v, _, _) in enumerate(graph.edges):
        other = boundary if v is None else v
        adjacency[u].append((e, other))
        adjacency[other].append((e, u))
    best = math.inf

    def walk(vertex, parity, cost, visited):
        nonlocal best
        if (vertex, parity) == (boundary, 1):
            best = min(best, cost)
            return
        for e, other in adjacency[vertex]:
            flips = (graph.edges[e][3] >> observable) & 1
            state = (other, parity ^ flips)
            if state not in visited:
                walk(other, parity ^ flips, cost + costs[e], visited | {state})

    walk(boundary, 0, 0.0, {(boundary, 0)})
    return best


def _path_graph(boundary_weights=(1.0, 1.0)):
    # B -(flips)- 0 - 1 - 2 -(plain)- B, unit interior weights.
    w0, w2 = boundary_weights
    return DecodingGraph(3, 1, [(0, 1, 1.0, 0), (1, 2, 1.0, 0), (0, None, w0, 1), (2, None, w2, 0)])


def _assert_valid_correction(graph, syndrome, result):
    reconstructed = np.zeros(graph.num_detectors, dtype=np.uint8)
    mask = 0
    for e in result.selected_edges:
        u, v, _, label = graph.edges[e]
        reconstructed[u] ^= 1
        if v is not None:
            reconstructed[v] ^= 1
        mask ^= label
    np.testing.assert_array_equal(reconstructed, syndrome)
    assert [(mask >> k) & 1 for k in range(graph.num_observables)] == result.prediction.tolist()


def test_empty_syndrome_gap_is_the_cheapest_logical_path():
    graph = _path_graph()
    result = ClusterGapUnionFindDecoder(graph).decode_with_gaps([0, 0, 0])
    assert result.prediction.tolist() == [False]
    assert result.selected_edges == ()
    # No cluster exists, so the walk B-0-1-2-B costs every edge in full.
    assert result.cluster_gap[0] == pytest.approx(4.0)
    assert result.dijkstra_states[0] > 0


def test_cluster_spanning_both_boundaries_has_zero_gap():
    graph = _path_graph()
    syndrome = np.array([0, 1, 0], dtype=bool)
    result = ClusterGapUnionFindDecoder(graph).decode_with_gaps(syndrome)
    _assert_valid_correction(graph, syndrome, result)
    # Growth from detector 1 reaches both boundary terminals in one tied batch,
    # so both logical classes are free inside the final cluster.
    assert result.cluster_gap[0] == pytest.approx(0.0)


def test_partially_grown_edges_are_charged_their_remaining_growth():
    graph = _path_graph(boundary_weights=(3.0, 0.5))
    syndrome = np.array([1, 0, 0], dtype=bool)
    costs = UnionFindDecoder(graph).decode_with_growth_costs(syndrome).remaining_costs
    # Growth from detector 0: edges 0-1 (t=1) and 1-2 (t=2) complete, then the
    # cheap boundary at 2 (t=2.5) stops the cluster. The flipping boundary edge at 0
    # has grown 2.5 of its weight 3, leaving 0.5; internal edges cost nothing.
    np.testing.assert_allclose(costs, [0.0, 0.0, 0.5, 0.0])
    result = ClusterGapUnionFindDecoder(graph).decode_with_gaps(syndrome)
    _assert_valid_correction(graph, syndrome, result)
    assert result.cluster_gap[0] == pytest.approx(0.5)
    assert result.cluster_gap[0] == pytest.approx(_brute_force_odd_walk(graph, costs, 0))


@pytest.mark.parametrize('seed', range(6))
def test_dijkstra_agrees_with_brute_force_on_random_small_graphs(seed):
    rng = np.random.default_rng(seed)
    n = 5
    # A path keeps every detector connected to a boundary; extra chords add cycles.
    edges = [(u, u + 1, float(rng.integers(1, 5)), 0) for u in range(n - 1)]
    edges += [(u, v, float(rng.integers(1, 5)), 0) for u in range(n) for v in range(u + 2, n) if rng.random() < 0.4]
    edges += [(0, None, float(rng.integers(1, 5)), 1), (n - 1, None, float(rng.integers(1, 5)), 0)]
    graph = DecodingGraph(n, 1, edges)
    decoder = ClusterGapUnionFindDecoder(graph)
    for _ in range(4):
        syndrome = rng.random(n) < 0.4
        costs = UnionFindDecoder(graph).decode_with_growth_costs(syndrome).remaining_costs
        result = decoder.decode_with_gaps(syndrome)
        _assert_valid_correction(graph, syndrome, result)
        assert result.cluster_gap[0] == pytest.approx(_brute_force_odd_walk(graph, costs, 0))


def test_observable_without_flipping_edges_is_rejected():
    with pytest.raises(ValueError, match='exactly one component'):
        ClusterGapUnionFindDecoder(DecodingGraph(2, 1, [(0, 1, 1.0, 0), (1, None, 1.0, 0)]))


def test_reference_bits_equal_plain_uf_on_the_fixture():
    fx = yoked_fixture(shots=32)
    patch = fx.patches[2]
    local = patch.local_syndromes(fx.detectors)
    expected = UnionFindDecoder(patch.graph).decode_batch(local)
    decoder = ClusterGapUnionFindDecoder(patch.graph)
    for shot in range(len(local)):
        result = decoder.decode_with_gaps(local[shot])
        _assert_valid_correction(patch.graph, local[shot], result)
        np.testing.assert_array_equal(result.prediction, expected[shot])
        assert np.isfinite(result.cluster_gap).all() and (result.cluster_gap >= 0).all()
        assert (result.dijkstra_states > 0).all()
