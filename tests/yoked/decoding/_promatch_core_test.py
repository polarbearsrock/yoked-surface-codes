from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding._promatch import FallbackReason, predecode
from yoked.decoding._promatch_graph import DomainGraph, Edge
from yoked.decoding._promatch_layout import L1FullHistoryDomain, YokeDetector


def _edge(
    edge_id: int,
    source: int,
    target: int | None,
    *,
    weight: float = 1,
    mask: bytes = b"",
) -> Edge:
    role = YokeDetector()
    return Edge(
        edge_id=edge_id,
        source=source,
        target=target,
        weight=weight,
        observable_mask=mask,
        source_role=role,
        target_role=None if target is None else role,
    )


def _domain_graph(
    patch_id: int, vertices: tuple[int, ...], edges: tuple[Edge, ...]
) -> DomainGraph:
    domain = L1FullHistoryDomain(patch_id=patch_id, check_basis="X")
    adjacency = {v: [] for v in vertices}
    neighbors = {v: set() for v in vertices}
    boundary = []
    for edge in edges:
        adjacency[edge.source].append(edge)
        if edge.target is None:
            boundary.append(edge)
        else:
            adjacency[edge.target].append(edge)
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
    return DomainGraph(
        domain=domain,
        detector_ids=vertices,
        edges=edges,
        adjacency={v: tuple(adjacency[v]) for v in vertices},
        neighbors={v: tuple(sorted(neighbors[v])) for v in vertices},
        boundary_edges=tuple(boundary),
        boundary_adjacency={
            v: tuple(edge for edge in adjacency[v] if edge.target is None)
            for v in vertices
        },
    )


def _compiled(
    *domain_graphs: DomainGraph,
    num_detectors: int,
    num_observables: int = 0,
):
    edges_by_id = {}
    for graph in domain_graphs:
        for edge in graph.edges:
            edges_by_id[edge.edge_id] = edge
    return SimpleNamespace(
        num_detectors=num_detectors,
        num_observables=num_observables,
        edges=tuple(edges_by_id[k] for k in sorted(edges_by_id)),
        domain_graphs={graph.domain: graph for graph in domain_graphs},
    )


def test_stage1_uses_deterministic_weight_order_and_stops_at_limit() -> None:
    heavy = _edge(0, 0, 1, weight=5)
    light = _edge(1, 2, 3, weight=1)
    graph = _domain_graph(0, (0, 1, 2, 3), (heavy, light))
    result = predecode(
        _compiled(graph, num_detectors=4),
        np.ones(4, dtype=np.uint8),
        residual_hw_limit=2,
    )

    assert np.array_equal(result.residual_syndrome, [1, 1, 0, 0])
    assert len(result.paths) == 1
    assert result.paths[0].stage == 1
    assert result.paths[0].edge_ids == (1,)
    assert result.domain_stats[graph.domain].committed_stage_counts == (1, 0, 0, 0)


def test_stage2_prefers_safe_degree_one_pair() -> None:
    edges = (
        _edge(0, 0, 1, weight=2),
        _edge(1, 1, 2, weight=0.1),
        _edge(2, 2, 3, weight=1),
    )
    graph = _domain_graph(0, (0, 1, 2, 3), edges)
    result = predecode(
        _compiled(graph, num_detectors=4),
        np.ones(4, dtype=np.uint8),
        residual_hw_limit=2,
    )

    # The tempting middle edge would create two singletons.  Of the two safe
    # end pairs, the lower-weight one is selected.
    assert result.paths[0].stage == 2
    assert result.paths[0].edge_ids == (2,)
    assert np.array_equal(result.residual_syndrome, [1, 1, 0, 0])


def test_stage3_path_may_cross_an_active_internal_detector() -> None:
    # Active graph: 0 is a singleton and 1 is the center of a two-leaf star.
    # The only safe singleton match is 0--2 (or 0--3), and its immutable-graph
    # shortest path passes through active detector 1.  Detector 1 must remain.
    edges = (
        _edge(0, 1, 2),
        _edge(1, 1, 3, weight=3),
        _edge(2, 0, 4),
        _edge(3, 4, 1),
    )
    graph = _domain_graph(0, (0, 1, 2, 3, 4), edges)
    syndrome = np.asarray([1, 1, 1, 1, 0], dtype=np.uint8)
    result = predecode(_compiled(graph, num_detectors=5), syndrome, residual_hw_limit=2)

    assert result.paths[0].stage == 3
    assert result.paths[0].endpoints == (0, 2)
    assert result.paths[0].edge_ids == (2, 3, 0)
    assert result.residual_syndrome[1] == 1
    assert np.array_equal(result.residual_syndrome, [0, 1, 0, 1, 0])


def test_stage4_is_used_only_when_every_adjacent_pair_is_risky() -> None:
    edges = (
        _edge(0, 0, 1, weight=3),
        _edge(1, 0, 2, weight=1),
        _edge(2, 0, 3, weight=2),
    )
    graph = _domain_graph(0, (0, 1, 2, 3), edges)
    result = predecode(
        _compiled(graph, num_detectors=4),
        np.ones(4, dtype=np.uint8),
        residual_hw_limit=2,
    )

    assert result.paths[0].stage == 4
    assert result.paths[0].edge_ids == (1,)
    assert np.array_equal(result.residual_syndrome, [0, 1, 0, 1])


def test_domain_rollback_is_transactional_and_does_not_leak_frame() -> None:
    successful = _domain_graph(0, (0, 1), (_edge(0, 0, 1, mask=b"\x00"),))
    failing = _domain_graph(
        1,
        (2, 3, 4, 5),
        (_edge(1, 2, 3, mask=b"\x01"),),
    )
    shot = np.ones(6, dtype=np.uint8)
    result = predecode(
        _compiled(successful, failing, num_detectors=6, num_observables=1),
        shot,
        residual_hw_limit=0,
        observable_policy="any",
    )

    assert np.array_equal(result.residual_syndrome, [0, 0, 1, 1, 1, 1])
    assert np.array_equal(result.observable_frame, [0])
    assert [path.edge_ids for path in result.paths] == [(0,)]
    stats = result.domain_stats[failing.domain]
    assert stats.status == "rollback"
    assert stats.attempted_matches == 1
    assert stats.committed_matches == 0
    assert stats.attempted_residual_hw == 2
    assert stats.final_residual_hw == 4
    assert stats.fallback_reason is FallbackReason.DISCONNECTED


def test_odd_parity_boundary_is_real_edge_and_toggles_one_detector() -> None:
    boundary = _edge(0, 0, None)
    graph = _domain_graph(0, (0, 1, 2), (boundary,))
    compiled = _compiled(graph, num_detectors=3)

    enabled = predecode(
        compiled,
        np.ones(3, dtype=np.uint8),
        residual_hw_limit=2,
        boundary_policy="odd-parity",
    )
    assert enabled.paths[0].endpoints == (0, None)
    assert enabled.paths[0].stage == 1
    assert np.array_equal(enabled.residual_syndrome, [0, 1, 1])
    assert enabled.domain_stats[graph.domain].boundary_was_used

    disabled = predecode(
        compiled,
        np.ones(3, dtype=np.uint8),
        residual_hw_limit=2,
        boundary_policy="disabled",
    )
    assert disabled.domain_stats[graph.domain].status == "rollback"
    assert np.array_equal(disabled.residual_syndrome, [1, 1, 1])


def test_unused_parity_boundary_is_recorded_and_has_no_correction() -> None:
    graph = _domain_graph(
        0,
        (0, 1, 2),
        (
            _edge(0, 0, 1, weight=0),
            _edge(1, 2, None, weight=4),
        ),
    )
    result = predecode(
        _compiled(graph, num_detectors=3),
        np.ones(3, dtype=np.uint8),
        residual_hw_limit=1,
        boundary_policy="odd-parity",
    )

    # Both 0--1 and 2--boundary are isolated; deterministic weight ordering
    # reaches the target through 0--1 and discards the unused virtual vertex.
    assert result.paths[0].edge_ids == (0,)
    stats = result.domain_stats[graph.domain]
    assert stats.boundary_was_added
    assert not stats.boundary_was_used
    assert stats.boundary_discarded_unused


def test_observable_policies_and_gf2_frame_cancellation() -> None:
    nonzero = _edge(0, 0, 1, mask=b"\x01")
    graph = _domain_graph(0, (0, 1), (nonzero,))
    compiled = _compiled(graph, num_detectors=2, num_observables=1)

    strict = predecode(
        compiled,
        np.ones(2, dtype=np.uint8),
        residual_hw_limit=0,
        observable_policy="edge-zero",
    )
    assert strict.domain_stats[graph.domain].status == "rollback"

    frame_bearing = predecode(
        compiled,
        np.ones(2, dtype=np.uint8),
        residual_hw_limit=0,
        observable_policy="any",
    )
    assert np.array_equal(frame_bearing.observable_frame, [1])

    # Two independently selected masks for the same observable cancel over
    # GF(2), even though both path decision costs remain in the telemetry.
    cancel_graph = _domain_graph(
        0,
        (0, 1, 2, 3),
        (
            _edge(0, 0, 1, mask=b"\x01"),
            _edge(1, 2, 3, mask=b"\x01"),
        ),
    )
    cancelled = predecode(
        _compiled(cancel_graph, num_detectors=4, num_observables=1),
        np.ones(4, dtype=np.uint8),
        residual_hw_limit=0,
        observable_policy="any",
    )
    assert np.array_equal(cancelled.observable_frame, [0])
    assert cancelled.decision_weight == 2


def test_path_zero_accepts_canceling_simple_path_but_edge_zero_does_not() -> None:
    graph = _domain_graph(
        0,
        (0, 1, 2),
        (
            _edge(0, 0, 1, mask=b"\x01"),
            _edge(1, 1, 2, mask=b"\x01"),
        ),
    )
    compiled = _compiled(graph, num_detectors=3, num_observables=1)
    shot = np.asarray([1, 0, 1], dtype=np.uint8)

    strict = predecode(
        compiled,
        shot,
        residual_hw_limit=0,
        observable_policy="edge-zero",
    )
    assert strict.domain_stats[graph.domain].status == "rollback"

    path_zero = predecode(
        compiled,
        shot,
        residual_hw_limit=0,
        observable_policy="path-zero",
    )
    assert path_zero.domain_stats[graph.domain].status == "success"
    assert path_zero.paths[0].stage == 3
    assert path_zero.paths[0].edge_ids == (0, 1)
    assert path_zero.paths[0].observable_mask == b"\x00"
    assert np.array_equal(path_zero.observable_frame, [0])


def test_frame_bearing_path_rejects_foreign_patch_observable() -> None:
    graph = _domain_graph(
        1,
        (0, 1),
        (_edge(0, 0, 1, mask=b"\x01"),),
    )
    with pytest.raises(ValueError, match="owned by patch 0"):
        predecode(
            _compiled(graph, num_detectors=2, num_observables=4),
            np.ones(2, dtype=np.uint8),
            residual_hw_limit=0,
            observable_policy="any",
        )


def test_result_is_independent_of_edge_enumeration_and_input_is_immutable() -> None:
    edge_a = _edge(0, 0, 1)
    edge_b = _edge(1, 2, 3)
    graph_forward = _domain_graph(0, (0, 1, 2, 3), (edge_a, edge_b))
    graph_reverse = _domain_graph(0, (0, 1, 2, 3), (edge_b, edge_a))
    shot = np.ones(4, dtype=np.uint8)
    original = shot.copy()

    forward = predecode(
        _compiled(graph_forward, num_detectors=4), shot, residual_hw_limit=2
    )
    reverse = predecode(
        _compiled(graph_reverse, num_detectors=4), shot, residual_hw_limit=2
    )

    assert np.array_equal(shot, original)
    assert forward.paths == reverse.paths
    assert np.array_equal(forward.residual_syndrome, reverse.residual_syndrome)


def test_equal_routes_and_zero_weight_cycle_terminate_deterministically() -> None:
    graph = _domain_graph(
        0,
        (0, 1, 2, 3),
        (
            _edge(0, 0, 1, weight=0),
            _edge(1, 1, 2, weight=0),
            _edge(2, 0, 2, weight=0),
            _edge(3, 1, 3, weight=1),
            _edge(4, 2, 3, weight=1),
        ),
    )
    result = predecode(
        _compiled(graph, num_detectors=4),
        np.asarray([1, 0, 0, 1], dtype=np.uint8),
        residual_hw_limit=0,
    )
    assert result.paths[0].stage == 3
    assert result.paths[0].edge_ids == (0, 3)


def test_overlapping_paths_cancel_repeated_edges_over_gf2() -> None:
    edges = (
        _edge(0, 0, 5, weight=1),
        _edge(1, 0, 7, weight=0),
        _edge(2, 1, 4, weight=1),
        _edge(3, 2, 4, weight=1),
        _edge(4, 2, 6, weight=1),
        _edge(5, 2, 7, weight=2),
        _edge(6, 3, 5, weight=1),
        _edge(7, 4, 7, weight=1),
    )
    graph = _domain_graph(0, tuple(range(8)), edges)
    shot = np.asarray([1, 1, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    result = predecode(
        _compiled(graph, num_detectors=8),
        shot,
        residual_hw_limit=0,
    )
    assert [path.edge_ids for path in result.paths] == [
        (1,),
        (2,),
        (6, 0, 1, 5, 4),
    ]
    edge_parity = np.zeros(len(edges), dtype=np.uint8)
    detector_boundary = np.zeros(8, dtype=np.uint8)
    for path in result.paths:
        for edge_id in path.edge_ids:
            edge_parity[edge_id] ^= 1
    assert edge_parity[1] == 0
    for edge, parity in zip(edges, edge_parity):
        if parity:
            detector_boundary[edge.source] ^= 1
            detector_boundary[edge.target] ^= 1
    np.testing.assert_array_equal(detector_boundary, shot)
    assert not np.any(result.residual_syndrome)


def _slow_first_candidate(vertices, edges, active):
    active = set(active)
    neighbors = {v: set() for v in active}
    direct = []
    adjacency = {v: [] for v in vertices}
    for edge in edges:
        adjacency[edge.source].append((edge.target, edge))
        adjacency[edge.target].append((edge.source, edge))
        if edge.source in active and edge.target in active:
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
            direct.append(((edge.source, edge.target), edge))

    def creates_singleton(endpoints):
        removed = set(endpoints)
        return any(
            old and not (old - removed)
            for node, old in neighbors.items()
            if node not in removed
        )

    stage1 = []
    for endpoints, edge in direct:
        a, b = endpoints
        if neighbors[a] == {b} and neighbors[b] == {a}:
            stage1.append(
                ((edge.weight, min(a, b), max(a, b), edge.edge_id), endpoints, edge)
            )
    if stage1:
        _, endpoints, edge = min(stage1)
        return 1, endpoints, (edge.edge_id,)

    stage2 = []
    for endpoints, edge in direct:
        if not creates_singleton(endpoints):
            a, b = endpoints
            key = (
                0 if min(len(neighbors[a]), len(neighbors[b])) == 1 else 1,
                edge.weight,
                min(a, b),
                max(a, b),
                edge.edge_id,
            )
            stage2.append((key, endpoints, edge))
    if stage2:
        _, endpoints, edge = min(stage2)
        return 2, endpoints, (edge.edge_id,)

    def all_simple_paths(source, target):
        found = []

        def visit(node, seen, path):
            if node == target:
                ids = tuple(edge.edge_id for edge in path)
                found.append((sum(edge.weight for edge in path), len(path), ids))
                return
            for other, edge in adjacency[node]:
                if other not in seen:
                    visit(other, seen | {other}, (*path, edge))

        visit(source, {source}, ())
        return found

    stage3 = []
    for singleton in sorted(v for v in active if not neighbors[v]):
        for other in sorted(active):
            if other == singleton or creates_singleton((singleton, other)):
                continue
            paths = all_simple_paths(singleton, other)
            if paths:
                weight, edge_count, edge_ids = min(paths)
                stage3.append(
                    (
                        (weight, singleton, other, edge_count, edge_ids),
                        (singleton, other),
                        edge_ids,
                    )
                )
    if stage3:
        _, endpoints, edge_ids = min(stage3)
        return 3, endpoints, edge_ids

    stage4 = []
    for endpoints, edge in direct:
        if creates_singleton(endpoints):
            a, b = endpoints
            key = (
                0 if min(len(neighbors[a]), len(neighbors[b])) == 1 else 1,
                edge.weight,
                min(a, b),
                max(a, b),
                edge.edge_id,
            )
            stage4.append((key, endpoints, edge))
    if stage4:
        _, endpoints, edge = min(stage4)
        return 4, endpoints, (edge.edge_id,)
    return None


def test_randomized_first_candidate_matches_independent_slow_oracle() -> None:
    rng = random.Random(20260817)
    role = YokeDetector()
    for _ in range(250):
        vertex_count = rng.randint(2, 7)
        vertices = tuple(range(vertex_count))
        edges = []
        for a in vertices:
            for b in range(a + 1, vertex_count):
                if rng.random() < 0.38:
                    edges.append(
                        Edge(
                            edge_id=len(edges),
                            source=a,
                            target=b,
                            weight=float(rng.randrange(3)),
                            observable_mask=b"",
                            source_role=role,
                            target_role=role,
                        )
                    )
        active = {v for v in vertices if rng.random() < 0.7}
        if len(active) < 2:
            active = set(rng.sample(vertices, 2))
        graph = _domain_graph(0, vertices, tuple(edges))
        shot = np.asarray([v in active for v in vertices], dtype=np.uint8)
        expected = _slow_first_candidate(vertices, tuple(edges), active)
        actual = predecode(
            _compiled(graph, num_detectors=vertex_count),
            shot,
            residual_hw_limit=len(active) - 2,
        )
        if expected is None:
            assert not actual.paths
            assert actual.domain_stats[graph.domain].status == "rollback"
        else:
            stage, endpoints, edge_ids = expected
            assert len(actual.paths) == 1
            assert actual.paths[0].stage == stage
            assert actual.paths[0].endpoints == endpoints
            assert actual.paths[0].edge_ids == edge_ids


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), -1])
def test_invalid_eligible_weights_are_fatal_not_rollbacks(bad_weight: float) -> None:
    graph = _domain_graph(0, (0, 1), (_edge(0, 0, 1, weight=bad_weight),))
    with pytest.raises(ValueError, match="invalid weight"):
        predecode(
            _compiled(graph, num_detectors=2),
            np.ones(2, dtype=np.uint8),
            residual_hw_limit=0,
        )
