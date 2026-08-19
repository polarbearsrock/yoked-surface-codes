from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding._promatch import (
    CommitProposal,
    DomainProposalStepper,
    FallbackReason,
    apply_detector_boundary,
    predecode,
)
from yoked.decoding._promatch_graph import DomainGraph, Edge
from yoked.decoding._promatch_layout import L1FullHistoryDomain, YokeDetector


def _edge(
    edge_id: int,
    source: int,
    target: int | None,
    *,
    weight: float,
    observable_mask: bytes = b"",
) -> Edge:
    role = YokeDetector()
    return Edge(
        edge_id=edge_id,
        source=source,
        target=target,
        weight=weight,
        observable_mask=observable_mask,
        source_role=role,
        target_role=None if target is None else role,
    )


def _graph(vertex_count: int, edges: tuple[Edge, ...]) -> DomainGraph:
    domain = L1FullHistoryDomain(patch_id=0, check_basis="X")
    adjacency = {v: [] for v in range(vertex_count)}
    neighbors = {v: set() for v in range(vertex_count)}
    boundary_edges = []
    for edge in edges:
        adjacency[edge.source].append(edge)
        if edge.target is None:
            boundary_edges.append(edge)
        else:
            adjacency[edge.target].append(edge)
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
    return DomainGraph(
        domain=domain,
        detector_ids=tuple(range(vertex_count)),
        edges=edges,
        adjacency={v: tuple(adjacency[v]) for v in range(vertex_count)},
        neighbors={v: tuple(sorted(neighbors[v])) for v in range(vertex_count)},
        boundary_edges=tuple(boundary_edges),
        boundary_adjacency={
            v: tuple(edge for edge in adjacency[v] if edge.target is None)
            for v in range(vertex_count)
        },
    )


def _stepper(
    graph: DomainGraph,
    *,
    hw_limit: int = 0,
    veto_budget: int | None = None,
) -> DomainProposalStepper:
    return DomainProposalStepper(
        graph,
        graph.detector_ids,
        num_detectors=len(graph.detector_ids),
        num_observables=0,
        residual_hw_limit=hw_limit,
        veto_budget=veto_budget,
    )


def _accept_all(stepper: DomainProposalStepper):
    while (proposal := stepper.next_proposal()) is not None:
        stepper.accept(proposal)
    return stepper.outcome("tx")


def test_all_accept_stepper_matches_legacy_path_order_and_residual() -> None:
    graph = _graph(
        4,
        (
            _edge(0, 0, 1, weight=5),
            _edge(1, 2, 3, weight=1),
        ),
    )
    legacy = predecode(
        SimpleNamespace(
            num_detectors=4,
            num_observables=0,
            edges=graph.edges,
            domain_graphs={graph.domain: graph},
        ),
        np.ones(4, dtype=np.uint8),
        residual_hw_limit=0,
    )
    stepped = _accept_all(_stepper(graph))

    assert stepped.status == "success"
    assert stepped.provisional_paths == legacy.paths
    assert stepped.durable_paths == legacy.paths
    assert stepped.durable_active == frozenset(
        np.flatnonzero(legacy.residual_syndrome)
    )
    assert stepped.proposal_stage_counts == (2, 0, 0, 0)


def test_proposal_is_immutable_and_does_not_mutate_before_accept() -> None:
    graph = _graph(2, (_edge(0, 0, 1, weight=2),))
    stepper = _stepper(graph)
    before_active = stepper.active
    proposal = stepper.next_proposal()

    assert isinstance(proposal, CommitProposal)
    assert proposal is stepper.next_proposal()
    assert proposal.detector_boundary == (0, 1)
    assert stepper.active == before_active
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.stage = 4  # type: ignore[misc]

    shot = np.ones(2, dtype=np.uint8)
    corrected = apply_detector_boundary(shot, proposal.detector_boundary)
    np.testing.assert_array_equal(shot, [1, 1])
    np.testing.assert_array_equal(corrected, [0, 0])


def test_candidate_and_stage3_enumeration_telemetry_is_monotone() -> None:
    graph = _graph(2, (_edge(0, 0, 1, weight=2),))
    stepper = _stepper(graph)

    assert stepper.total_candidate_enumeration_ns == 0
    assert stepper.total_stage3_enumeration_ns == 0
    proposal = stepper.next_proposal()
    assert proposal is not None
    assert stepper.last_candidate_enumeration_ns >= stepper.last_stage3_enumeration_ns >= 0
    assert stepper.total_candidate_enumeration_ns == stepper.last_candidate_enumeration_ns
    assert stepper.total_stage3_enumeration_ns == stepper.last_stage3_enumeration_ns

    # Reading an already-pending proposal must not perform or charge another
    # candidate enumeration.
    before = (
        stepper.total_candidate_enumeration_ns,
        stepper.total_stage3_enumeration_ns,
    )
    assert stepper.next_proposal() is proposal
    assert before == (
        stepper.total_candidate_enumeration_ns,
        stepper.total_stage3_enumeration_ns,
    )


def test_veto_is_state_local_and_accept_restarts_stage_order() -> None:
    graph = _graph(
        4,
        (
            _edge(0, 0, 1, weight=5),
            _edge(1, 2, 3, weight=1),
        ),
    )
    stepper = _stepper(graph)
    first = stepper.next_proposal()
    assert first is not None and first.edge_ids == (1,)
    state = stepper.active
    stepper.veto(first)
    assert stepper.active == state
    assert stepper.accepted_paths == ()

    second = stepper.next_proposal()
    assert second is not None and second.edge_ids == (0,)
    stepper.accept(second)

    # Acceptance changed the fingerprint and discarded the old blacklist.
    third = stepper.next_proposal()
    assert third is not None and third.edge_ids == (1,)
    assert third.active_state_fingerprint != first.active_state_fingerprint
    stepper.accept(third)
    outcome = stepper.outcome("tx")
    assert outcome.status == "success"
    assert outcome.attempted_proposals == 3
    assert outcome.accepted_proposals == 2
    assert outcome.vetoed_proposals == 1


def test_exact_bth_veto_budget_rolls_back_tx_and_keeps_partial_prefix() -> None:
    graph = _graph(
        6,
        (
            _edge(0, 0, 1, weight=1),
            _edge(1, 2, 3, weight=2),
            _edge(2, 4, 5, weight=3),
        ),
    )
    stepper = _stepper(graph, veto_budget=2)
    accepted = stepper.next_proposal()
    assert accepted is not None
    stepper.accept(accepted)
    rejected1 = stepper.next_proposal()
    assert rejected1 is not None
    stepper.veto(rejected1)
    rejected2 = stepper.next_proposal()
    assert rejected2 is not None
    stepper.veto(rejected2)

    assert stepper.is_finished
    assert stepper.next_proposal() is None
    tx = stepper.outcome("tx")
    partial = stepper.outcome("partial")
    assert tx.exhaustion_kind == "veto-budget"
    assert tx.veto_budget_hit
    assert tx.attempted_proposals == 3
    assert tx.accepted_proposals == 1
    assert tx.vetoed_proposals == 2
    assert tx.status == "rollback"
    assert tx.durable_paths == ()
    assert tx.durable_active == frozenset(range(6))
    assert partial.status == "partial-exhausted"
    assert len(partial.provisional_paths) == 1
    accepted_path = partial.provisional_paths[0]
    assert partial.durable_paths == partial.provisional_paths
    assert accepted_path.edge_ids == accepted.edge_ids
    assert partial.durable_active == frozenset((2, 3, 4, 5))


def test_proposal_exhaustion_has_tx_and_partial_durable_semantics() -> None:
    graph = _graph(4, (_edge(0, 0, 1, weight=1),))
    stepper = _stepper(graph)
    proposal = stepper.next_proposal()
    assert proposal is not None
    stepper.accept(proposal)
    assert stepper.next_proposal() is None

    tx = stepper.outcome("tx")
    partial = stepper.outcome("partial")
    assert tx.exhaustion_kind == "proposal"
    assert tx.fallback_reason is FallbackReason.DISCONNECTED
    assert tx.status == "rollback"
    assert tx.provisional_active == frozenset((2, 3))
    assert tx.durable_active == frozenset(range(4))
    assert not tx.durable_paths
    assert partial.status == "partial-exhausted"
    assert partial.durable_active == frozenset((2, 3))
    assert partial.durable_paths == partial.provisional_paths


def test_all_accept_preserves_legacy_accumulated_disconnected_reason() -> None:
    graph = _graph(
        5,
        (
            _edge(0, 0, 2, weight=1, observable_mask=b"\x01"),
            _edge(1, 0, 4, weight=4, observable_mask=b"\x01"),
            _edge(2, 0, None, weight=4, observable_mask=b"\x00"),
            _edge(3, 1, 3, weight=4, observable_mask=b"\x00"),
            _edge(4, 1, 4, weight=2, observable_mask=b"\x01"),
            _edge(5, 2, None, weight=3, observable_mask=b"\x00"),
            _edge(6, 3, 4, weight=2, observable_mask=b"\x00"),
        ),
    )
    syndrome = np.ones(5, dtype=np.uint8)
    legacy = predecode(
        SimpleNamespace(
            num_detectors=5,
            num_observables=1,
            edges=graph.edges,
            domain_graphs={graph.domain: graph},
        ),
        syndrome,
        residual_hw_limit=0,
        observable_policy="path-zero",
    )
    stepper = DomainProposalStepper(
        graph,
        graph.detector_ids,
        num_detectors=5,
        num_observables=1,
        residual_hw_limit=0,
        observable_policy="path-zero",
    )
    stepped = _accept_all(stepper)

    assert tuple(path.edge_ids for path in stepped.provisional_paths) == (
        (0, 1),
        (3,),
    )
    # Both are transactional rollbacks, so compare the attempted trace via
    # the stepper and the legacy telemetry's accumulated terminal reason.
    stats = legacy.domain_stats[graph.domain]
    assert stats.attempted_stage_counts == stepped.accepted_stage_counts
    assert stats.fallback_reason is FallbackReason.DISCONNECTED
    assert stepped.fallback_reason is FallbackReason.DISCONNECTED


@pytest.mark.parametrize("bad_budget", [True, 0, -1, 1.5])
def test_veto_budget_must_be_a_positive_integer(bad_budget: object) -> None:
    graph = _graph(2, (_edge(0, 0, 1, weight=1),))
    with pytest.raises((TypeError, ValueError), match="veto_budget"):
        DomainProposalStepper(
            graph,
            graph.detector_ids,
            num_detectors=2,
            num_observables=0,
            residual_hw_limit=0,
            veto_budget=bad_budget,  # type: ignore[arg-type]
        )
