from __future__ import annotations

import dataclasses
import random
from fractions import Fraction

from yoked.decoding._patch_uf import Dyadic, compile_lane, run_lane
from yoked.decoding._patch_uf_reference import (
    BudgetLimits,
    LaneOutcome,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
    run_reference_lane,
)


def _policy(tau: Fraction = Fraction(0)) -> UFPolicy:
    limits = BudgetLimits.unbounded_for_testing()
    return UFPolicy(tau, limits, limits)


def _exact(value: object | None) -> Fraction | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    assert isinstance(value, Dyadic)
    return value.as_fraction()


def _semantic(outcome: LaneOutcome) -> object:
    components = tuple(
        (
            c.absorbed_vertices,
            c.original_defects,
            c.forest_edge_ids,
            c.peeled_support_edge_ids,
            _exact(c.exact_margin),
            c.gate_decision,
            c.gate_reason_set,
            c.primary_gate_reason,
            c.boundary_reached,
            c.port_tainted,
            c.port_kind_set,
            c.saturated_port_count,
            c.merge_count,
            c.simultaneous_event_batch_count,
            c.event_batch_ids,
            tuple(_exact(value) for value in c.event_batch_times),
            _exact(c.last_membership_event_time),
            _exact(c.maximum_incident_half_edge_charge),
        )
        for c in outcome.completed_components
    )
    censored = tuple(
        (
            c.absorbed_vertices,
            c.current_defects,
            c.forest_edge_ids,
            c.boundary_reached,
            c.port_tainted,
            c.port_kind_set,
            c.saturated_port_count,
            c.merge_count,
            c.simultaneous_event_batch_count,
            c.event_batch_ids,
            tuple(_exact(value) for value in c.event_batch_times),
            _exact(c.last_membership_event_time),
            _exact(c.maximum_incident_half_edge_charge),
        )
        for c in outcome.censored_components
    )
    counters = outcome.counters
    semantic_counters = (
        counters.growth_event_count,
        counters.simultaneous_event_batch_count,
        counters.union_attempt_count,
        counters.successful_union_count,
        counters.failed_union_count,
        counters.forest_edge_count,
        counters.peel_operation_count,
        counters.peak_live_component_count,
    )
    return (
        outcome.status,
        components,
        censored,
        semantic_counters,
        outcome.censor_reason,
        outcome.budget_exceeded_set,
        outcome.primary_budget_cap,
        _exact(outcome.terminal_event_time),
        outcome.last_complete_batch_id,
    )


def _random_graph(rng: random.Random, n: int) -> UFLaneGraph:
    edges: list[UFEdge] = []
    edge_id = 0
    # A chain makes every vertex reachable while optional boundaries permit odd
    # components to terminate.  Extra edges deliberately create cycles/ties.
    for v in range(n - 1):
        edges.append(
            UFEdge(
                edge_id,
                v,
                v + 1,
                Fraction(rng.choice((1, 2, 3, 4)), 2),
                "correction",
            )
        )
        edge_id += 1
    for _ in range(rng.randrange(n + 2)):
        a, b = rng.sample(range(n), 2)
        edges.append(
            UFEdge(
                edge_id,
                a,
                b,
                Fraction(rng.choice((1, 2, 3, 4, 6, 8)), 4),
                "correction",
            )
        )
        edge_id += 1
    for v in range(n):
        if rng.random() < 0.45:
            edges.append(
                UFEdge(
                    edge_id,
                    v,
                    None,
                    Fraction(rng.choice((1, 2, 3, 4)), 2),
                    "boundary",
                )
            )
            edge_id += 1
        if rng.random() < 0.35:
            edges.append(
                UFEdge(
                    edge_id,
                    v,
                    None,
                    Fraction(rng.choice((1, 2, 3, 4)), 2),
                    "port",
                    port_kind=rng.choice(("yoke", "cross-lane")),
                )
            )
            edge_id += 1
    return UFLaneGraph(n, tuple(edges))


def test_random_dyadic_graphs_match_fraction_reference() -> None:
    rng = random.Random(0xC0FFEE)
    for _ in range(250):
        n = rng.randrange(2, 8)
        graph = _random_graph(rng, n)
        defects = [v for v in range(n) if rng.random() < 0.45]
        tau = Fraction(rng.choice((0, 1, 2, 3)), 4)
        reference = run_reference_lane(graph, defects, _policy(tau))
        production = run_lane(graph, defects, _policy(tau))
        assert _semantic(production) == _semantic(reference)


def test_edge_iteration_and_root_ties_do_not_change_semantics() -> None:
    edges = (
        UFEdge(10, 0, 1, 1.0, "correction"),
        UFEdge(11, 1, 2, 1.0, "correction"),
        UFEdge(12, 2, 3, 1.0, "correction"),
        UFEdge(13, 3, 0, 1.0, "correction"),
        UFEdge(14, 0, None, 2.0, "boundary"),
        UFEdge(15, 2, None, 2.0, "port", port_kind="yoke"),
    )
    expected = _semantic(run_lane(UFLaneGraph(4, edges), [0, 1, 2], _policy()))
    for permutation in (
        tuple(reversed(edges)),
        (edges[2], edges[0], edges[5], edges[3], edges[1], edges[4]),
    ):
        assert _semantic(run_lane(UFLaneGraph(4, permutation), [0, 1, 2], _policy())) == expected


def test_compiled_lane_reuses_graph_invariants_without_changing_semantics() -> None:
    graph = _random_graph(random.Random(123), 7)
    policy = _policy(Fraction(1, 4))
    compiled = compile_lane(graph, policy)

    assert compiled.graph is graph
    assert len(compiled.tick_weights) == len(graph.edges)
    for defects in ((0, 2, 5), (1, 3), ()):
        assert _semantic(compiled.run(defects)) == _semantic(
            run_reference_lane(graph, defects, policy)
        )
