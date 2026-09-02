from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from yoked.decoding._patch_uf_reference import (
    BudgetLimits,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
    forest_diameter_hops,
    run_reference_lane,
)


def _policy(tau: Fraction | int = 0, **semantic_caps: int | None) -> UFPolicy:
    unbounded = BudgetLimits.unbounded_for_testing()
    values = {field.name: None for field in dataclasses.fields(BudgetLimits)}
    values.update(semantic_caps)
    return UFPolicy(tau, BudgetLimits(**values), unbounded)


def test_two_active_components_grow_at_rate_two_and_peel() -> None:
    graph = UFLaneGraph(
        2,
        (UFEdge(7, 0, 1, 2.0, "correction"),),
    )
    result = run_reference_lane(graph, [0, 1], _policy())

    assert result.status == "completed"
    assert result.terminal_event_time == Fraction(1)
    assert result.counters.growth_event_count == 1
    assert result.counters.union_attempt_count == 1
    assert result.counters.successful_union_count == 1
    assert result.counters.peel_operation_count == 1
    component = result.completed_components[0]
    assert component.absorbed_vertices == (0, 1)
    assert component.original_defects == (0, 1)
    assert component.forest_edge_ids == (7,)
    assert component.peeled_support_edge_ids == (7,)
    assert component.forest_diameter_hops == 1
    assert component.exact_margin is None
    assert component.gate_decision == "eligible"
    assert component.event_batch_ids == (1,)
    assert component.event_batch_times == (Fraction(1),)
    assert component.last_membership_event_time == Fraction(1)
    assert component.maximum_incident_half_edge_charge == Fraction(1)
    assert result.last_complete_batch_id == 1


def test_boundary_tie_keeps_canonical_min_and_defers_at_zero_margin() -> None:
    graph = UFLaneGraph(
        1,
        (
            UFEdge(5, 0, None, 1.0, "boundary"),
            UFEdge(3, 0, None, 1.0, "boundary"),
        ),
    )
    result = run_reference_lane(graph, [0], _policy(0))

    assert result.status == "completed"
    assert result.counters.growth_event_count == 2
    assert result.counters.forest_edge_count == 1
    component = result.completed_components[0]
    assert component.forest_edge_ids == (3,)
    assert component.peeled_support_edge_ids == (3,)
    assert component.exact_margin == 0
    assert component.gate_decision == "deferred"
    assert component.primary_gate_reason == "below-threshold"


def test_merged_boundary_components_retain_one_canonical_incidence() -> None:
    graph = UFLaneGraph(
        3,
        (
            UFEdge(5, 0, None, 1.0, "boundary"),
            UFEdge(3, 2, None, 1.0, "boundary"),
            UFEdge(8, 0, 1, 4.0, "correction"),
            UFEdge(9, 1, 2, 4.0, "correction"),
        ),
    )
    result = run_reference_lane(graph, [0, 1, 2], _policy())

    assert result.status == "completed"
    assert result.terminal_event_time == 3
    component = result.completed_components[0]
    assert component.forest_edge_ids == (3, 8, 9)
    assert 5 not in component.forest_edge_ids
    assert component.exact_margin == 0
    assert component.peeled_support_edge_ids in ((3, 8), (3, 9))


def test_port_contact_neutralizes_component_and_skips_peeling() -> None:
    graph = UFLaneGraph(
        2,
        (
            UFEdge(1, 0, None, 1.0, "port", port_kind="yoke"),
            UFEdge(2, 0, 1, 3.0, "correction"),
            UFEdge(3, 1, None, 2.0, "boundary"),
        ),
    )
    result = run_reference_lane(graph, [0], _policy())

    assert result.status == "completed"
    # The port saturates at t=1 and the component stops there: the correction
    # never closes and the true boundary is never reached.
    assert result.terminal_event_time == Fraction(1)
    assert result.counters.growth_event_count == 1
    assert result.counters.union_attempt_count == 0
    assert result.counters.forest_edge_count == 0
    assert result.counters.peel_operation_count == 0
    component = result.completed_components[0]
    assert component.absorbed_vertices == (0,)
    assert component.original_defects == (0,)
    assert component.forest_edge_ids == ()
    assert component.peeled_support_edge_ids == ()
    assert not component.boundary_reached
    assert component.port_tainted
    assert component.port_kind_set == ("yoke",)
    assert component.saturated_port_count == 1
    assert component.exact_margin == 0
    assert component.gate_decision == "deferred"
    assert component.primary_gate_reason == "port-contact"
    assert set(component.gate_reason_set) == {
        "below-threshold",
        "port-contact",
        "port-yoke",
    }
    assert component.maximum_incident_half_edge_charge == Fraction(1)


def test_port_contact_is_inherited_through_union_and_stops_growth() -> None:
    graph = UFLaneGraph(
        3,
        (
            UFEdge(0, 0, None, 1.0, "port", port_kind="yoke"),
            UFEdge(1, 0, 1, 2.0, "correction"),
            UFEdge(2, 1, 2, 2.0, "correction"),
            UFEdge(3, 2, None, 10.0, "boundary"),
        ),
    )
    result = run_reference_lane(graph, [0, 2], _policy())

    assert result.status == "completed"
    # Vertex 0 stops at its port at t=1.  Vertex 2 keeps growing, absorbs
    # vertex 1 at t=2, and reaches the stopped component at t=3; the union
    # inherits the port contact and is neutral immediately.
    assert result.terminal_event_time == Fraction(3)
    assert result.counters.growth_event_count == 3
    assert result.counters.successful_union_count == 2
    assert result.counters.peel_operation_count == 0
    component = result.completed_components[0]
    assert component.absorbed_vertices == (0, 1, 2)
    assert component.original_defects == (0, 2)
    assert component.forest_edge_ids == (1, 2)
    assert component.peeled_support_edge_ids == ()
    assert not component.boundary_reached
    assert component.port_tainted
    assert component.event_batch_ids == (1, 2, 3)
    assert component.gate_decision == "deferred"
    assert component.primary_gate_reason == "port-contact"


def test_strict_threshold_uses_exact_post_growth_slack() -> None:
    graph = UFLaneGraph(
        2,
        (
            UFEdge(1, 0, 1, 1.0, "correction"),
            UFEdge(2, 0, 1, 2.0, "correction"),
        ),
    )
    equal = run_reference_lane(graph, [0, 1], _policy(Fraction(1)))
    below = run_reference_lane(graph, [0, 1], _policy(Fraction(1, 2)))

    assert equal.completed_components[0].exact_margin == 1
    assert equal.completed_components[0].gate_decision == "deferred"
    assert below.completed_components[0].gate_decision == "eligible"


def test_local_incomplete_returns_last_complete_snapshot() -> None:
    graph = UFLaneGraph(1, ())
    result = run_reference_lane(graph, [0], _policy())

    assert result.status == "censored"
    assert result.censor_reason == "local-incomplete-neutralization"
    assert not result.budget_exceeded_set
    assert result.censored_components[0].current_defects == (0,)
    assert result.censored_components[0].partial_cluster_defect_lower_bound == 1


def test_semantic_budget_rejects_whole_batch_without_counting_it() -> None:
    graph = UFLaneGraph(2, (UFEdge(0, 0, 1, 1.0, "correction"),))
    result = run_reference_lane(
        graph,
        [0, 1],
        _policy(growth_event_count=0, union_attempt_count=0),
    )

    assert result.status == "censored"
    assert result.censor_reason == "budget-exhaustion"
    assert result.counters.growth_event_count == 0
    assert result.counters.union_attempt_count == 0
    assert [item.cap_name for item in result.budget_exceeded_set] == [
        "growth_event_count",
        "union_attempt_count",
    ]
    assert result.primary_budget_cap == "growth_event_count"


def test_graph_and_policy_validation_fail_closed() -> None:
    limits = BudgetLimits.unbounded_for_testing()
    with pytest.raises(ValueError, match="strictly positive"):
        UFEdge(0, 0, 1, 0.0, "correction")
    with pytest.raises(ValueError, match="duplicate edge_id"):
        UFLaneGraph(
            2,
            (
                UFEdge(0, 0, 1, 1.0, "correction"),
                UFEdge(0, 0, None, 1.0, "boundary"),
            ),
        )
    with pytest.raises(ValueError, match="tau"):
        UFPolicy(Fraction(-1), limits, limits)


def test_forest_diameter_helper_measures_hops_and_fails_closed() -> None:
    assert forest_diameter_hops([5], []) == 0
    assert forest_diameter_hops([0, 1], [(0, 1)]) == 1
    # chain 0-1-2-3
    assert forest_diameter_hops([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]) == 3
    # star centred on 0
    assert forest_diameter_hops([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)]) == 2
    # forest must span the vertex set
    with pytest.raises(ValueError, match="disconnected"):
        forest_diameter_hops([0, 1, 2], [(0, 1)])
    with pytest.raises(ValueError, match="cycle"):
        forest_diameter_hops([0, 1, 2], [(0, 1), (1, 2), (2, 0)])


def test_final_components_record_forest_diameter_for_chain_star_and_singleton() -> None:
    # Chain: defects at both ends of 0-1-2-3, equal weights.
    chain = UFLaneGraph(
        4,
        (
            UFEdge(0, 0, 1, 2.0, "correction"),
            UFEdge(1, 1, 2, 2.0, "correction"),
            UFEdge(2, 2, 3, 2.0, "correction"),
        ),
    )
    result = run_reference_lane(chain, [0, 3], _policy())
    assert result.completed_components[0].absorbed_vertices == (0, 1, 2, 3)
    assert result.completed_components[0].forest_diameter_hops == 3

    # Star: a lone defect at the centre absorbs three leaves at once, then
    # reaches the boundary through leaf 1.
    star = UFLaneGraph(
        4,
        (
            UFEdge(0, 0, 1, 1.0, "correction"),
            UFEdge(1, 0, 2, 1.0, "correction"),
            UFEdge(2, 0, 3, 1.0, "correction"),
            UFEdge(3, 1, None, 10.0, "boundary"),
        ),
    )
    result = run_reference_lane(star, [0], _policy())
    component = result.completed_components[0]
    assert component.absorbed_vertices == (0, 1, 2, 3)
    assert component.boundary_reached
    assert component.forest_diameter_hops == 2

    # Singleton reaching a boundary directly.
    single = UFLaneGraph(1, (UFEdge(0, 0, None, 1.0, "boundary"),))
    result = run_reference_lane(single, [0], _policy())
    assert result.completed_components[0].forest_diameter_hops == 0

    # A port-contact component still reports its forest diameter.
    port = UFLaneGraph(
        3,
        (
            UFEdge(0, 0, None, 1.0, "port", port_kind="yoke"),
            UFEdge(1, 0, 1, 2.0, "correction"),
            UFEdge(2, 1, 2, 2.0, "correction"),
            UFEdge(3, 2, None, 10.0, "boundary"),
        ),
    )
    result = run_reference_lane(port, [0, 2], _policy())
    component = result.completed_components[0]
    assert component.port_tainted
    assert component.forest_diameter_hops == 2
