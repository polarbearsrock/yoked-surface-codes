from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from yoked.decoding._patch_uf import Dyadic, run_lane
from yoked.decoding._patch_uf_reference import (
    BudgetLimits,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
)


def _limits(**changes: int | None) -> BudgetLimits:
    values = {field.name: None for field in dataclasses.fields(BudgetLimits)}
    values.update(changes)
    return BudgetLimits(**values)


def _policy(
    tau: object = 0,
    *,
    semantic: BudgetLimits | None = None,
    production: BudgetLimits | None = None,
) -> UFPolicy:
    return UFPolicy(
        tau,
        semantic if semantic is not None else _limits(),
        production if production is not None else _limits(),
    )


def test_dyadic_is_canonical_and_exact() -> None:
    assert Dyadic(8, -4) == Dyadic(1, -1)
    assert Dyadic.from_value(0.1).as_fraction() == Fraction.from_float(0.1)
    assert Dyadic.from_value(Fraction(3, 8)) + Dyadic.from_value(Fraction(1, 4)) == Fraction(5, 8)
    assert Dyadic.from_value(Fraction(3, 8)).divide_int(2) == Fraction(3, 16)
    assert Dyadic(-3, -2) < Dyadic(-1, -2) < Dyadic(0) < Dyadic(1, 20)
    with pytest.raises(ValueError, match="dyadic"):
        Dyadic.from_value(Fraction(1, 3))


def test_production_heap_lifecycle_has_exact_golden_counters() -> None:
    graph = UFLaneGraph(
        3,
        (
            UFEdge(0, 0, 1, 2.0, "correction"),
            UFEdge(1, 1, 2, 4.0, "correction"),
            UFEdge(2, 0, None, 5.0, "boundary"),
        ),
    )
    result = run_lane(graph, [0, 1], _policy())

    assert result.status == "completed"
    # At the first and only batch, all three incidences are schedulable, one is
    # popped at the minimum, and the rebuilt queue is then discarded.
    assert result.counters.heap_push_count == 3
    assert result.counters.heap_pop_count == 1
    assert result.counters.stale_heap_pop_count == 0
    assert result.counters.heap_operation_count == 4
    assert result.counters.peak_heap_size == 3
    assert result.counters.temporary_memory_units == 3


def test_production_heap_budget_is_transactional_and_production_only() -> None:
    graph = UFLaneGraph(
        2,
        (
            UFEdge(0, 0, 1, 1.0, "correction"),
            UFEdge(1, 0, None, 4.0, "boundary"),
        ),
    )
    result = run_lane(
        graph,
        [0, 1],
        _policy(production=_limits(heap_operation_count=2, peak_heap_size=1)),
    )

    assert result.status == "censored"
    assert result.counters.heap_operation_count == 0
    assert result.counters.growth_event_count == 0
    # The persistent exact-tick lifecycle inserts two initial entries; two
    # operations meet the cap exactly, while queue size two exceeds its cap.
    assert [item.cap_name for item in result.budget_exceeded_set] == [
        "peak_heap_size",
    ]
    assert result.budget_exceeded_set[0].rejected_next_value == 2


def test_simultaneous_cycle_counts_failed_union_and_zero_margin() -> None:
    graph = UFLaneGraph(
        3,
        (
            UFEdge(0, 0, 1, 2.0, "correction"),
            UFEdge(1, 1, 2, 2.0, "correction"),
            UFEdge(2, 0, 2, 2.0, "correction"),
            UFEdge(3, 0, None, 4.0, "boundary"),
        ),
    )
    result = run_lane(graph, [0, 1, 2], _policy())

    assert result.status == "completed"
    assert result.counters.union_attempt_count == 3
    assert result.counters.successful_union_count == 2
    assert result.counters.failed_union_count == 1
    component = result.completed_components[0]
    assert component.exact_margin == 0
    assert component.gate_decision == "deferred"
    assert len(component.forest_edge_ids) == 3  # two corrections + one boundary


def test_semantic_component_size_and_peel_caps_match_exact_next_value() -> None:
    graph = UFLaneGraph(2, (UFEdge(0, 0, 1, 1.0, "correction"),))
    size_limited = run_lane(
        graph,
        [0, 1],
        _policy(semantic=_limits(absorbed_vertex_count=1)),
    )
    peel_limited = run_lane(
        graph,
        [0, 1],
        _policy(semantic=_limits(peel_operation_count=0)),
    )

    assert size_limited.status == "censored"
    assert size_limited.primary_budget_cap == "absorbed_vertex_count"
    assert size_limited.budget_exceeded_set[0].rejected_next_value == 2
    assert size_limited.counters.growth_event_count == 0
    assert size_limited.terminal_event_time == 0
    assert all(
        component.maximum_incident_half_edge_charge == 0
        for component in size_limited.censored_components
    )
    assert peel_limited.status == "censored"
    assert peel_limited.primary_budget_cap == "peel_operation_count"
    assert peel_limited.counters.growth_event_count == 1
    assert peel_limited.counters.peel_operation_count == 0


def test_graph_protocol_adapter_does_not_require_compiler_type() -> None:
    class View:
        num_vertices = 2
        edges = (UFEdge(4, 0, 1, 1.0, "correction"),)

    result = run_lane(View(), [0, 1], _policy())
    assert result.status == "completed"
    assert result.completed_components[0].peeled_support_edge_ids == (4,)


def test_post_reschedule_budget_rejection_rolls_back_entire_batch() -> None:
    graph = UFLaneGraph(
        3,
        (
            UFEdge(0, 0, 1, 2, "correction"),
            UFEdge(1, 1, 2, 4, "correction"),
            UFEdge(2, 2, None, 10, "boundary"),
        ),
    )
    result = run_lane(
        graph,
        [0, 1, 2],
        _policy(production=_limits(heap_push_count=3)),
    )

    assert result.status == "censored"
    assert result.primary_budget_cap == "heap_push_count"
    assert result.budget_exceeded_set[0].rejected_next_value == 4
    assert result.terminal_event_time == 0
    assert result.last_complete_batch_id is None
    assert result.counters.heap_push_count == 3
    assert result.counters.heap_pop_count == 0
    assert result.counters.growth_event_count == 0
    assert result.counters.simultaneous_event_batch_count == 0
    assert tuple(c.absorbed_vertices for c in result.censored_components) == (
        (0,),
        (1,),
        (2,),
    )
    assert all(not c.forest_edge_ids for c in result.censored_components)
    assert all(not c.event_batch_ids for c in result.censored_components)


def test_stale_queue_drain_is_counted_before_local_incompleteness() -> None:
    graph = UFLaneGraph(
        3,
        (
            UFEdge(0, 0, 1, 2, "correction"),
            UFEdge(1, 0, 1, 4, "correction"),
        ),
    )
    result = run_lane(graph, [0, 1, 2], _policy())

    assert result.status == "censored"
    assert result.censor_reason == "local-incomplete-neutralization"
    assert result.counters.heap_push_count == 2
    assert result.counters.heap_pop_count == 2
    assert result.counters.stale_heap_pop_count == 1
    assert result.counters.heap_operation_count == 4
    assert result.counters.growth_event_count == 1
    assert result.counters.simultaneous_event_batch_count == 1
    assert result.last_complete_batch_id == 1
