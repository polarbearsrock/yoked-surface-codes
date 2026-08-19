from __future__ import annotations

import json
from typing import cast

import numpy as np
import pymatching
import pytest

from yoked.decoding._promatch import PrematchedPath
from yoked.decoding._promatch_graph import CompiledPromatchGraph, DomainGraph, Edge
from yoked.decoding._promatch_layout import (
    L1FullHistoryDomain,
    PromatchLayout,
    YokeDetector,
)
from yoked.decoding._promatch_oracle_replay import (
    run_oracle_trajectory,
    run_phase_a_replay,
    shadow_score_paths,
)


def _mask(fault_ids: set[int], *, num_observables: int) -> bytes:
    result = bytearray((num_observables + 7) // 8)
    for fault_id in fault_ids:
        result[fault_id // 8] |= 1 << (fault_id % 8)
    return bytes(result)


def _compiled(
    matcher: pymatching.Matching,
    *,
    num_observables: int,
    domain_edge_pairs: tuple[tuple[int, int | None], ...],
) -> CompiledPromatchGraph:
    matcher.ensure_num_fault_ids(num_observables)
    role = YokeDetector()
    edges: list[Edge] = []
    by_pair: dict[tuple[int, int | None], Edge] = {}
    for edge_id, (raw_source, raw_target, data) in enumerate(matcher.edges()):
        source = int(raw_source)
        target = None if raw_target is None else int(raw_target)
        if target is not None and target < source:
            source, target = target, source
        edge = Edge(
            edge_id=edge_id,
            source=source,
            target=target,
            weight=float(data["weight"]),
            observable_mask=_mask(
                set(data.get("fault_ids", ())),
                num_observables=num_observables,
            ),
            source_role=role,
            target_role=None if target is None else role,
        )
        edges.append(edge)
        by_pair[(source, target)] = edge

    domain = L1FullHistoryDomain(patch_id=0, check_basis="X")
    domain_edges = tuple(by_pair[pair] for pair in domain_edge_pairs)
    detector_ids = tuple(range(matcher.num_detectors))
    adjacency: dict[int, list[Edge]] = {v: [] for v in detector_ids}
    neighbors: dict[int, set[int]] = {v: set() for v in detector_ids}
    boundary_edges: list[Edge] = []
    for edge in domain_edges:
        adjacency[edge.source].append(edge)
        if edge.target is None:
            boundary_edges.append(edge)
        else:
            adjacency[edge.target].append(edge)
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
    domain_graph = DomainGraph(
        domain=domain,
        detector_ids=detector_ids,
        edges=domain_edges,
        adjacency={v: tuple(adjacency[v]) for v in detector_ids},
        neighbors={v: tuple(sorted(neighbors[v])) for v in detector_ids},
        boundary_edges=tuple(boundary_edges),
        boundary_adjacency={
            v: tuple(edge for edge in adjacency[v] if edge.target is None)
            for v in detector_ids
        },
    )
    return CompiledPromatchGraph(
        layout=cast(PromatchLayout, None),
        matcher=matcher,
        edges=tuple(edges),
        domain_graphs={domain: domain_graph},
        fingerprint="oracle-replay-test",
        require_zero_frame=False,
        num_detectors=matcher.num_detectors,
        num_observables=num_observables,
    )


def _edge(graph: CompiledPromatchGraph, pair: tuple[int, int | None]) -> Edge:
    for edge in graph.edges:
        if (edge.source, edge.target) == pair:
            return edge
    raise AssertionError(f"missing edge {pair!r}")


def _rollback_graph() -> CompiledPromatchGraph:
    matcher = pymatching.Matching()
    matcher.add_edge(0, 1, weight=1, fault_ids={0})
    matcher.add_boundary_edge(0, weight=10)
    matcher.add_boundary_edge(1, weight=10)
    matcher.add_boundary_edge(2, weight=1)
    matcher.add_boundary_edge(3, weight=1)
    return _compiled(
        matcher,
        num_observables=1,
        domain_edge_pairs=((0, 1),),
    )


def _frame_veto_graph() -> CompiledPromatchGraph:
    matcher = pymatching.Matching()
    matcher.add_edge(0, 1, weight=2, fault_ids={0})
    matcher.add_boundary_edge(0, weight=1)
    matcher.add_boundary_edge(1, weight=1, fault_ids={1})
    matcher.add_boundary_edge(2, weight=1)
    return _compiled(
        matcher,
        num_observables=2,
        domain_edge_pairs=((0, None),),
    )


def test_transaction_rolls_back_an_accepted_prefix_but_partial_retains_it() -> None:
    graph = _rollback_graph()
    syndrome = np.ones(4, dtype=np.uint8)

    tx = run_oracle_trajectory(
        graph,
        syndrome,
        policy="frame",
        transaction_policy="tx",
        residual_hw_limit=0,
        observable_policy="any",
    )
    partial = run_oracle_trajectory(
        graph,
        syndrome,
        policy="frame",
        transaction_policy="partial",
        residual_hw_limit=0,
        observable_policy="any",
    )

    assert tx.domains[0].outcome.status == "rollback"
    assert tx.proposals[0].accepted
    assert tx.proposals[0].rolled_back
    assert not tx.proposals[0].durable
    assert tx.final_residual_syndrome == (1, 1, 1, 1)
    assert tx.final_observable_frame == b"\x00"

    assert partial.domains[0].outcome.status == "partial-exhausted"
    assert partial.proposals[0].accepted
    assert partial.proposals[0].durable
    assert not partial.proposals[0].rolled_back
    assert partial.final_residual_syndrome == (0, 0, 1, 1)
    assert partial.final_observable_frame == b"\x01"
    assert tx.final_prediction == tx.initial_u0_prediction
    assert partial.final_prediction == partial.initial_u0_prediction


def test_frame_veto_does_not_mutate_global_syndrome_or_frame() -> None:
    graph = _frame_veto_graph()
    syndrome = np.ones(3, dtype=np.uint8)
    result = run_oracle_trajectory(
        graph,
        syndrome,
        policy="frame",
        transaction_policy="partial",
        residual_hw_limit=0,
        boundary_policy="odd-parity",
        observable_policy="any",
    )

    assert result.proposals[0].vetoed
    assert not result.proposals[0].evaluation.frame_compatible
    assert result.final_residual_syndrome == (1, 1, 1)
    assert result.final_observable_frame == b"\x00"
    assert result.final_prediction == result.initial_u0_prediction


def test_shadow_scores_an_incompatible_legacy_path_but_still_follows_it() -> None:
    graph = _frame_veto_graph()
    candidate = _edge(graph, (0, None))
    domain = next(iter(graph.domain_graphs))
    path = PrematchedPath(
        domain=domain,
        stage=1,
        endpoints=(0, None),
        edge_ids=(candidate.edge_id,),
        decision_weight=candidate.weight,
        observable_mask=candidate.observable_mask,
    )
    result = shadow_score_paths(graph, np.ones(3, dtype=np.uint8), (path,))

    assert result.proposals[0].oracle_cost_accepts
    assert not result.proposals[0].oracle_frame_accepts
    assert result.final_residual_syndrome == (0, 1, 1)
    assert result.final_prediction != result.initial_u0_prediction


def test_shadow_reconstructs_legacy_path_boundary_independently() -> None:
    graph = _frame_veto_graph()
    candidate = _edge(graph, (0, None))
    domain = next(iter(graph.domain_graphs))
    malformed = PrematchedPath(
        domain=domain,
        stage=1,
        endpoints=(1, None),
        edge_ids=(candidate.edge_id,),
        decision_weight=candidate.weight,
        observable_mask=candidate.observable_mask,
    )
    with pytest.raises(AssertionError, match="canonical boundary"):
        shadow_score_paths(graph, np.ones(3, dtype=np.uint8), (malformed,))


def test_phase_a_bundle_is_deterministically_json_serializable() -> None:
    graph = _frame_veto_graph()
    candidate = _edge(graph, (0, None))
    domain = next(iter(graph.domain_graphs))
    legacy_path = PrematchedPath(
        domain=domain,
        stage=1,
        endpoints=(0, None),
        edge_ids=(candidate.edge_id,),
        decision_weight=candidate.weight,
        observable_mask=candidate.observable_mask,
    )
    result = run_phase_a_replay(
        graph,
        np.ones(3, dtype=np.uint8),
        (legacy_path,),
        residual_hw_limit=0,
        boundary_policy="odd-parity",
        observable_policy="any",
    )

    first = json.dumps(result.to_json(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(result.to_json(), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert '"rolled_back"' in first
    assert '"candidate_weight_hex":"0x1.0000000000000p+0"' in first
    assert '"actual_observables"' not in first
    assert result.cache_stats.hits > 0
    assert result.cache_stats.misses > 0
    assert result.cache_stats.entries > 0
    assert result.initial_uncached_repeatability_verified
    assert result.initial_cached_equivalence_verified
    assert result.frame_tx.final_prediction == result.frame_tx.initial_u0_prediction
    assert (
        result.frame_partial.final_prediction
        == result.frame_partial.initial_u0_prediction
    )
