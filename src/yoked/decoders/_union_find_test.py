from collections import deque
import itertools
import sys

import numpy as np
import pytest

from yoked.decoders import DecodingGraph, InvalidSyndromeError, UnionFindDecoder
from yoked.decoders._union_find import GrowthDecodeResult, _Growth


def _assert_valid(graph, syndrome, result):
    reconstructed = np.zeros(graph.num_detectors, dtype=np.uint8)
    mask = 0
    assert len(set(result.selected_edges)) == len(result.selected_edges)
    assert set(result.selected_edges) <= set(result.forest_edges)
    for e in result.selected_edges:
        u, v, _, label = graph.edges[e]
        reconstructed[u] ^= 1
        if v is not None:
            reconstructed[v] ^= 1
        mask ^= label
    np.testing.assert_array_equal(reconstructed, syndrome)
    assert result.observable_mask == mask


def _scan_oracle(graph, syndrome):
    """Tiny-graph oracle using explicit member sets and full frontier scans.

    It shares neither DSU state nor scheduling/peeling code with the decoder.
    """
    nd = graph.num_detectors
    components = {v: {v} for v in range(len(graph.adjacency))}
    owner = list(components)
    grown = [0.0] * len(graph.edges)
    now = 0.0
    forest = []

    def active(component):
        members = components[component]
        return all(v < nd for v in members) and sum(int(syndrome[v]) for v in members) % 2 == 1

    def completions():
        result = {}
        for e, (u, v) in enumerate(graph.endpoints):
            if owner[u] == owner[v]:
                continue
            rate = int(active(owner[u])) + int(active(owner[v]))
            remaining = max(0.0, graph.edges[e][2] - grown[e])
            if remaining == 0:
                result[e] = now
            elif rate:
                result[e] = now + remaining / rate
        return result

    events = completions()
    while events:
        batch_time = min(events.values())
        limit = batch_time + 1e-12 + 1e-12 * abs(batch_time)
        while True:
            group = sorted(e for e, time in events.items() if time <= limit)
            if not group:
                break
            elapsed = batch_time - now
            for e, (u, v) in enumerate(graph.endpoints):
                if owner[u] != owner[v]:
                    rate = int(active(owner[u])) + int(active(owner[v]))
                    grown[e] = min(graph.edges[e][2], grown[e] + elapsed * rate)
            now = batch_time
            for e in group:
                grown[e] = graph.edges[e][2]
                u, v = graph.endpoints[e]
                a, b = owner[u], owner[v]
                if a == b:
                    continue
                members = components.pop(a) | components.pop(b)
                root = min(members)
                components[root] = members
                for member in members:
                    owner[member] = root
                forest.append(e)
            events = completions()
    if any(active(c) for c in components):
        raise InvalidSyndromeError('Unresolved odd component in scan oracle')

    # Remove unused terminal leaves, then eliminate detector leaves. With one
    # terminal per tree (or none), the remaining forest correction is unique.
    adjacency = [set() for _ in graph.adjacency]
    for e in forest:
        for v in graph.endpoints[e]:
            adjacency[v].add(e)
    for members in components.values():
        terminals = sorted(v for v in members if v >= nd)
        for terminal in terminals[1:]:
            for e in list(adjacency[terminal]):
                for v in graph.endpoints[e]:
                    adjacency[v].remove(e)
    parity = list(map(int, syndrome)) + [0] * (len(adjacency) - nd)
    leaves = deque(v for v in range(nd) if len(adjacency[v]) == 1)
    selected = set()
    mask = 0
    while leaves:
        v = leaves.popleft()
        if not adjacency[v]:
            continue
        e, = adjacency[v]
        a, b = graph.endpoints[e]
        neighbor = b if a == v else a
        if parity[v]:
            selected.add(e)
            mask ^= graph.edges[e][3]
            parity[neighbor] ^= 1
        parity[v] = 0
        adjacency[v].remove(e)
        adjacency[neighbor].remove(e)
        if neighbor < nd and len(adjacency[neighbor]) == 1:
            leaves.append(neighbor)
    assert not any(parity[:nd])
    return tuple(forest), selected, mask


def _check_oracle(graph, syndrome):
    decoder = UnionFindDecoder(graph)
    result = decoder._decode(syndrome)
    _assert_valid(graph, syndrome, result)
    forest, selected, mask = _scan_oracle(graph, syndrome)
    assert result.forest_edges == forest
    assert set(result.selected_edges) == selected
    assert result.observable_mask == mask
    np.testing.assert_array_equal(decoder.decode(syndrome), [(mask >> k) & 1 for k in range(graph.num_observables)])


def test_weighted_competition_pauses_and_does_not_absorb_other_neighbors():
    graph = DecodingGraph(4, 3, [(0, 1, 1, 1), (0, 2, 4, 2), (2, 3, 1, 4)])
    growth = _Growth(graph, np.ones(4, dtype=np.bool_))
    growth.run()
    assert growth.time == 0.5
    assert growth.forest == [0, 2]
    assert growth.find(0) != growth.find(2)
    assert growth.grown[1] == 1
    _check_oracle(graph, [1, 1, 1, 1])


def test_odd_cluster_absorbs_even_cluster_and_publishes_earlier_deadline():
    graph = DecodingGraph(4, 3, [(1, 2, 1, 1), (0, 1, 3, 2), (2, 3, 6, 4)])
    growth = _Growth(graph, np.ones(4, dtype=np.bool_))
    assert growth.step() and growth.time == 0.5
    assert not growth.active(growth.find(1))
    assert growth.step() and growth.time == 2.5
    assert growth.active(growth.find(0))
    assert growth.step() and growth.time == 4.0
    assert not growth.step()
    result = UnionFindDecoder(graph)._decode([1, 1, 1, 1])
    assert set(result.selected_edges) == {1, 2}
    _check_oracle(graph, [1, 1, 1, 1])


def test_unfired_hub_between_defects():
    graph = DecodingGraph(4, 2, [(0, 1, 1, 1), (0, 2, 4, 2), (0, 3, 100, 0)])
    growth = _Growth(graph, np.array([0, 1, 1, 0], dtype=np.bool_))
    assert growth.step() and growth.time == 1
    assert growth.step() and growth.time == 2.5
    assert growth.find(0) != growth.find(3)
    assert not growth.step()
    _check_oracle(graph, [0, 1, 1, 0])


@pytest.mark.parametrize('hub_bit,k', list(itertools.product([0, 1], [32, 33])))
def test_equal_weight_hub_fan_in(hub_bit, k):
    graph = DecodingGraph(k + 1, 6, [(0, v, 1, 1 << (v % 6)) for v in range(1, k + 1)] + [(0, None, 10, 0)])
    syndrome = np.array([hub_bit] + [1] * k, dtype=np.bool_)
    growth = _Growth(graph, syndrome)
    assert growth.step()
    assert growth.time == (0.5 if hub_bit else 1)
    assert growth.forest == list(range(k))
    root = growth.find(0)
    assert growth.size[root] == k + 1
    assert growth.parity[root] == bool(hub_bit ^ (k % 2))
    assert not growth.terminal[root]
    _check_oracle(graph, syndrome)


@pytest.mark.parametrize('k,m', list(itertools.product([32, 33], [1, 2])))
def test_hub_fan_in_includes_simultaneous_fired_partners(k, m):
    edges = [(0, v, 1, 1) for v in range(1, k + 1)]
    edges += [(v + 1, k + v + 1, 1, 2) for v in range(m)]
    np.random.default_rng(42).shuffle(edges)
    edges.append((0, None, 10, 0))
    graph = DecodingGraph(k + m + 1, 2, edges)
    syndrome = np.ones(k + m + 1, dtype=np.bool_)
    growth = _Growth(graph, syndrome)
    assert growth.step() and growth.time == 0.5
    assert growth.forest == list(range(k + m))
    root = growth.find(0)
    assert growth.size[root] == k + m + 1
    assert growth.parity[root] == bool(1 ^ ((k + m) % 2))
    _check_oracle(graph, syndrome)


def test_earlier_partner_pairing_can_leave_hub_untouched():
    k = 6
    edges = [(0, v, 2, 1) for v in range(1, k + 1)]
    edges += [(v, v + k, 1, 2) for v in range(1, k + 1)]
    graph = DecodingGraph(2 * k + 1, 2, edges)
    syndrome = [0] + [1] * (2 * k)
    growth = _Growth(graph, syndrome)
    growth.run()
    assert growth.time == 0.5
    assert growth.size[growth.find(0)] == 1
    assert growth.forest == list(range(k, 2 * k))
    _check_oracle(graph, syndrome)


def test_triangle_skips_newly_internal_edge_and_reports_invalid_syndrome():
    edges = [(0, 1, 1, 1), (1, 2, 1, 2), (2, 0, 1, 4)]
    graph = DecodingGraph(3, 3, edges)
    growth = _Growth(graph, [1, 1, 1])
    assert growth.step()
    assert growth.forest == [0, 1]
    with pytest.raises(InvalidSyndromeError):
        growth.step()
    with pytest.raises(InvalidSyndromeError):
        UnionFindDecoder(graph).decode([1, 1, 1])
    _check_oracle(DecodingGraph(3, 3, edges + [(2, None, 5, 0)]), [1, 1, 1])


@pytest.mark.parametrize('bits', list(itertools.product([0, 1], repeat=3)))
def test_zero_cost_closure_preserves_labels(bits):
    graph = DecodingGraph(3, 3, [(0, 1, 0, 1), (1, 2, 0, 2), (2, 0, 0, 4), (0, None, 1, 0)])
    growth = _Growth(graph, bits)
    assert growth.step() and growth.time == 0
    assert growth.forest == [0, 1]
    _check_oracle(graph, bits)


def test_tied_terminals_peel_toward_smallest_terminal():
    graph = DecodingGraph(1, 2, [(0, None, 1, 1), (0, None, 1, 2)])
    result = UnionFindDecoder(graph)._decode([1])
    assert result.forest_edges == (0, 1)
    assert result.selected_edges == (0,)
    assert result.observable_mask == 1
    _check_oracle(graph, [1])


def test_both_endpoint_activities_change_without_double_settlement():
    graph = DecodingGraph(6, 0, [(0, 1, 1, 0), (2, 3, 1, 0), (4, 0, 3, 0), (5, 2, 3, 0), (0, 2, 10, 0)])
    growth = _Growth(graph, [1] * 6)
    assert growth.step() and growth.time == 0.5
    assert growth.grown[4] == 1 and growth.rate[4] == 0
    assert growth.step() and growth.time == 2.5
    assert growth.grown[4] == 1 and growth.rate[4] == 2
    assert growth.step() and growth.time == 7
    _check_oracle(graph, [1] * 6)


def test_newly_published_completion_is_included_in_same_time_closure():
    graph = DecodingGraph(3, 1, [(0, 1, 1, 0), (1, 2, 1 + 2.5e-12, 1)])
    growth = _Growth(graph, [1, 0, 1])
    assert growth.step() and growth.time == 1
    assert growth.forest == [0, 1]
    assert not growth.step()
    _check_oracle(graph, [1, 0, 1])


def test_tolerance_window_does_not_expand_transitively():
    graph = DecodingGraph(6, 0, [(0, 1, 2, 0), (2, 3, 2 + 3e-12, 0), (4, 5, 2 + 6e-12, 0)])
    growth = _Growth(graph, [1] * 6)
    assert growth.step() and growth.time == 1
    assert growth.forest == [0, 1]
    assert growth.step() and growth.time > 1 + 2e-12
    _check_oracle(graph, [1] * 6)


def test_long_chain_uses_iterative_growth_and_peeling():
    n = sys.getrecursionlimit() + 100
    graph = DecodingGraph(n, 1, [(v, v + 1, 1, int(v == 0)) for v in range(n - 1)])
    syndrome = np.zeros(n, dtype=np.bool_)
    syndrome[[0, -1]] = True
    result = UnionFindDecoder(graph)._decode(syndrome)
    assert len(result.forest_edges) == n - 1
    assert len(result.selected_edges) == n - 1
    assert result.observable_mask == 1
    _assert_valid(graph, syndrome, result)


def test_small_realizable_syndromes_exhaustively():
    graph = DecodingGraph(4, 3, [(0, 1, 1, 1), (1, 2, 2, 2), (2, 3, 3, 4), (3, 0, 1, 3), (0, 2, 2, 6)])
    for edge_bits in itertools.product([0, 1], repeat=len(graph.edges)):
        syndrome = np.zeros(4, dtype=np.uint8)
        for bit, (u, v, _, _) in zip(edge_bits, graph.edges):
            syndrome[u] ^= bit
            syndrome[v] ^= bit
        _check_oracle(graph, syndrome)


def test_random_realizable_syndromes_match_independent_scan():
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(1, 9))
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.35:
                    edges.append((u, v, int(rng.integers(1, 4)), int(rng.integers(0, 8))))
            if rng.random() < 0.3:
                edges.append((u, None, int(rng.integers(1, 4)), int(rng.integers(0, 8))))
        if edges and rng.random() < 0.5:
            u, v, w, mask = edges[0]
            edges.append((u, v, w, mask ^ 1))
        rng.shuffle(edges)
        graph = DecodingGraph(n, 3, edges)
        for _ in range(5):
            syndrome = np.zeros(n, dtype=np.uint8)
            for u, v, _, _ in edges:
                if rng.integers(0, 2):
                    syndrome[u] ^= 1
                    if v is not None:
                        syndrome[v] ^= 1
            _check_oracle(graph, syndrome)


def test_public_batch_shapes_immutability_and_error_recovery():
    graph = DecodingGraph(3, 4, [(0, 1, 1, 2)])
    decoder = UnionFindDecoder(graph)
    syndromes = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=np.uint8)
    original = syndromes.copy()
    syndromes.flags.writeable = False
    expected = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.bool_)
    np.testing.assert_array_equal(decoder.decode_batch(syndromes), expected)
    np.testing.assert_array_equal(syndromes, original)
    with pytest.raises(InvalidSyndromeError):
        decoder.decode([0, 0, 1])
    with pytest.raises(InvalidSyndromeError):
        decoder.decode_batch([[1, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(decoder.decode_batch(syndromes), expected)
    empty = decoder.decode_batch(np.empty((0, 3), dtype=np.bool_))
    assert empty.dtype == np.bool_ and empty.shape == (0, 4)
    assert UnionFindDecoder(DecodingGraph(0, 2, [])).decode([]).shape == (2,)


@pytest.mark.parametrize('syndrome', [[1], [[1, 1]], [2, 0], [-1, 0], [float('nan'), 0], ['1', '0'], [1j, 0]])
def test_invalid_syndrome_input(syndrome):
    decoder = UnionFindDecoder(DecodingGraph(2, 0, [(0, 1, 1, 0)]))
    with pytest.raises(ValueError):
        decoder.decode(syndrome)


def test_invalid_batch_input():
    decoder = UnionFindDecoder(DecodingGraph(2, 0, []))
    for data in [[], [0, 0], [[0]], [[2, 0]]]:
        with pytest.raises(ValueError):
            decoder.decode_batch(data)


def test_growth_time_overflow_has_a_clear_error():
    graph = DecodingGraph(2, 0, [(0, 1, 1e308, 0), (1, None, 1e308, 0)])
    with pytest.raises(OverflowError, match='rescale the graph weights'):
        UnionFindDecoder(graph).decode([1, 0])


# --- decode_with_growth_costs: the shared _decode_state path ---------------
#
# These tests exercise the growth-cost entry point added for soft-output
# consumers (the cluster-gap decoder). Both `decode`/`_decode` and
# `decode_with_growth_costs` are built on the same private `_decode_state`,
# so they must agree on every correction and fail identically on the same
# malformed input; only `decode_with_growth_costs` additionally exposes the
# settled per-edge growth costs.


def test_growth_costs_entry_point_matches_plain_decode_on_random_graphs():
    rng = np.random.default_rng(123)
    for _ in range(20):
        n = int(rng.integers(1, 9))
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.35:
                    edges.append((u, v, int(rng.integers(1, 4)), int(rng.integers(0, 8))))
            if rng.random() < 0.3:
                edges.append((u, None, int(rng.integers(1, 4)), int(rng.integers(0, 8))))
        if edges and rng.random() < 0.5:
            u, v, w, mask = edges[0]
            edges.append((u, v, w, mask ^ 1))
        rng.shuffle(edges)
        graph = DecodingGraph(n, 3, edges)
        decoder = UnionFindDecoder(graph)
        for _ in range(3):
            syndrome = np.zeros(n, dtype=np.uint8)
            for u, v, _, _ in edges:
                if rng.integers(0, 2):
                    syndrome[u] ^= 1
                    if v is not None:
                        syndrome[v] ^= 1
            expected = decoder._decode(syndrome)
            growth_result = decoder.decode_with_growth_costs(syndrome)
            # Both entry points must choose the identical correction.
            assert growth_result.selected_edges == expected.selected_edges
            assert growth_result.observable_mask == expected.observable_mask
            assert len(growth_result.remaining_costs) == len(graph.edges)
            np.testing.assert_array_equal(
                decoder.decode(syndrome),
                [(growth_result.observable_mask >> k) & 1 for k in range(graph.num_observables)],
            )


def test_growth_decode_result_tuples_survive_a_later_decode_unchanged():
    graph = DecodingGraph(4, 3, [(0, 1, 1, 1), (1, 2, 2, 2), (2, 3, 3, 4), (3, 0, 1, 3), (0, 2, 2, 6)])
    decoder = UnionFindDecoder(graph)
    first = decoder.decode_with_growth_costs([1, 1, 0, 0])
    assert isinstance(first, GrowthDecodeResult)
    selected_snapshot = tuple(first.selected_edges)
    costs_snapshot = tuple(first.remaining_costs)
    mask_snapshot = first.observable_mask
    # Later calls on the same decoder must not reach back and mutate a
    # previously returned result; each call owns fresh growth state.
    decoder.decode_with_growth_costs([0, 1, 1, 0])
    decoder.decode([1, 0, 0, 1])
    decoder.decode_batch([[1, 1, 0, 0], [0, 0, 0, 0]])
    assert first.selected_edges == selected_snapshot
    assert first.remaining_costs == costs_snapshot
    assert first.observable_mask == mask_snapshot


@pytest.mark.parametrize('syndrome', [[1], [[1, 1]], [2, 0], [-1, 0], [float('nan'), 0], ['1', '0'], [1j, 0]])
def test_growth_costs_rejects_malformed_syndromes_like_decode(syndrome):
    decoder = UnionFindDecoder(DecodingGraph(2, 0, [(0, 1, 1, 0)]))
    with pytest.raises(ValueError) as via_decode:
        decoder.decode(syndrome)
    with pytest.raises(ValueError) as via_growth_costs:
        decoder.decode_with_growth_costs(syndrome)
    assert type(via_decode.value) is type(via_growth_costs.value)
    assert str(via_decode.value) == str(via_growth_costs.value)


def test_growth_costs_rejects_unrealizable_syndrome_like_decode():
    # A boundaryless triangle with an odd syndrome has no valid correction.
    edges = [(0, 1, 1, 1), (1, 2, 1, 2), (2, 0, 1, 4)]
    decoder = UnionFindDecoder(DecodingGraph(3, 3, edges))
    with pytest.raises(InvalidSyndromeError):
        decoder.decode([1, 1, 1])
    with pytest.raises(InvalidSyndromeError):
        decoder.decode_with_growth_costs([1, 1, 1])
