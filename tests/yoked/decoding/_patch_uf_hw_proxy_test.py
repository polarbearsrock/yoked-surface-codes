from __future__ import annotations

import copy
import dataclasses
from fractions import Fraction
from types import SimpleNamespace

import pytest

from yoked.decoding._patch_uf import run_lane
from yoked.decoding._patch_uf_hw_proxy import (
    UFParallelDepthAssumptions,
    derive_uf_lane_hardware_proxy,
    derive_uf_shot_hardware_proxy,
    iteration_index,
)
from yoked.decoding._patch_uf_reference import (
    BudgetLimits,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
)


def _assumptions(**changes: object) -> UFParallelDepthAssumptions:
    values: dict[str, object] = {"growth_quantum_weight": Fraction(1, 2)}
    values.update(changes)
    return UFParallelDepthAssumptions(**values)


def _counters(
    *,
    batches: int,
    growth: int,
    attempts: int,
    successes: int,
    failures: int,
    forest: int,
    peel: int,
) -> dict[str, int]:
    return {
        "growth_event_count": growth,
        "simultaneous_event_batch_count": batches,
        "union_attempt_count": attempts,
        "successful_union_count": successes,
        "failed_union_count": failures,
        "forest_edge_count": forest,
        "peel_operation_count": peel,
    }


def _dyadic(mantissa: int, exponent: int = 0) -> dict[str, int]:
    return {"mantissa": mantissa, "exponent": exponent}


def _completed_component(
    *,
    size: int,
    absorbed: int,
    times: list[object],
    diameter: int,
) -> dict[str, object]:
    return {
        "state_collection": "completed_components",
        "adapter": {
            "cluster_defect_count": size,
            "absorbed_vertex_count": absorbed,
            "simultaneous_event_batch_count": len(times),
            "event_batch_ids": list(range(1, len(times) + 1)),
            "event_batch_times": list(times),
            "forest_diameter_hops": diameter,
        },
    }


def _lane(
    *,
    lane_offset: int,
    shot_id: int,
    terminal: object,
    counters: dict[str, int],
    status: str = "completed",
) -> dict[str, object]:
    return {
        "global_shot_id": shot_id,
        "lane_offset": lane_offset,
        "adapter": {
            "status": status,
            "counters": counters,
            "terminal_event_time": terminal,
        },
    }


def _serialized_bundle() -> dict[str, object]:
    lanes = [
        _lane(
            lane_offset=0,
            shot_id=7,
            terminal=_dyadic(3),
            counters=_counters(
                batches=3, growth=7, attempts=5, successes=4, failures=1, forest=4, peel=4
            ),
        ),
        _lane(
            lane_offset=1,
            shot_id=7,
            terminal=_dyadic(5, -2),
            counters=_counters(
                batches=2, growth=2, attempts=1, successes=1, failures=0, forest=1, peel=1
            ),
        ),
    ]
    components = [
        {
            "global_shot_id": 7,
            "lane_offset": 0,
            **_completed_component(
                size=5, absorbed=7, times=[_dyadic(1), _dyadic(3)], diameter=4
            ),
        },
        {
            "global_shot_id": 7,
            "lane_offset": 0,
            **_completed_component(size=2, absorbed=2, times=[_dyadic(1)], diameter=1),
        },
        {
            "global_shot_id": 7,
            "lane_offset": 1,
            **_completed_component(
                size=2, absorbed=3, times=[_dyadic(5, -2)], diameter=2
            ),
        },
    ]
    return {
        "shot": {
            "global_shot_id": 7,
            "adapter_metrics": {
                "original_detector_count": 12,
                "residual_detector_count": 6,
                "committed_defect_count": 6,
                "durable_boundary_count": 6,
            },
        },
        "lanes": lanes,
        "components": components,
    }


def test_iteration_index_is_an_exact_ceiling() -> None:
    q = Fraction(1, 2)
    assert iteration_index(Fraction(0), q) == 0
    assert iteration_index(Fraction(1), q) == 2
    assert iteration_index(Fraction(5, 4), q) == 3
    assert iteration_index(_dyadic(5, -2), q) == 3
    assert iteration_index(3, q) == 6
    with pytest.raises(ValueError, match="negative"):
        iteration_index(Fraction(-1), q)


def test_serialized_rows_produce_raw_work_and_helios_depth_proxy() -> None:
    result = derive_uf_shot_hardware_proxy(_serialized_bundle(), assumptions=_assumptions())

    assert result.global_shot_id == 7
    assert (result.lane_count, result.patch_count, result.active_lane_count) == (2, 1, 2)
    assert result.synchronous_event_batch_work == 5
    assert result.saturated_growth_event_work == 9
    assert result.union_merge_attempt_work == 6
    assert result.successful_union_merge_work == 5
    assert result.redundant_union_merge_work == 1
    assert result.forest_edge_work == result.peel_operation_work == 5
    assert result.completed_component_count == 3
    assert result.maximum_lane_synchronous_event_batches == 3
    assert result.maximum_completed_component_defect_count == 5
    assert result.maximum_absorbed_vertex_count == 7
    assert result.maximum_component_event_batch_count == 2

    lane0 = result.lanes[0]
    # Lane 0 grew for 3 weight units; at a quantum of 1/2 that is 6 iterations.
    assert lane0.growth_depth_weight == Fraction(3)
    assert lane0.growth_iteration_count == 6
    assert lane0.maximum_forest_diameter_hops == 4
    # Each iteration: 1 growing + 2 controller + 1 merge settle = 4 cycles.
    assert lane0.growth_depth_cycles == 24
    # Merge flooding: iterations 2 and 6 both see the diameter-4 component.
    assert lane0.merge_depth_cycles == 8
    # 1 load + 24 + 8 + 4 serial peel ops + 2 confidence checks.
    assert lane0.conservative_parallel_depth_cycles == 39

    lane1 = result.lanes[1]
    assert lane1.growth_depth_weight == Fraction(5, 4)
    assert lane1.growth_iteration_count == 3
    assert lane1.merge_depth_cycles == 2
    assert lane1.conservative_parallel_depth_cycles == 17

    assert result.growth_depth_weight == Fraction(3)
    assert result.growth_iteration_count == 6
    assert result.maximum_forest_diameter_hops == 4
    assert result.lane_core_critical_path_cycles == 39
    assert result.residual_boundary_update_work == 6
    assert result.residual_update_depth_cycles == 6
    assert result.conservative_parallel_depth_cycles == 46
    assert result.per_patch_residual_boundary_update_work is None
    assert result.parallel_lane_cores_per_patch_depth_cycles is None
    assert result.serial_basis_patch_engines_depth_cycles is None
    assert result.fully_shared_frontend_engine_depth_cycles == 39 + 17 + 1 + 6


def test_merge_cost_is_charged_per_iteration_by_largest_merging_diameter() -> None:
    bundle = _serialized_bundle()
    # Charge two cycles per hop and no fixed per-iteration cost so the merge
    # term is isolated: iterations 2 and 6 each cost 2*4 in lane 0.
    assumptions = _assumptions(
        syndrome_load_cycles=0,
        growing_cycles_per_iteration=0,
        controller_cycles_per_iteration=0,
        merge_settle_cycles_per_iteration=0,
        merge_cycles_per_hop=2,
        peel_cycles_per_operation=0,
        confidence_cycles_per_completed_component=0,
    )
    result = derive_uf_shot_hardware_proxy(bundle, assumptions=assumptions)
    assert result.lanes[0].merge_depth_cycles == 16
    assert result.lanes[0].conservative_parallel_depth_cycles == 16
    assert result.lanes[1].merge_depth_cycles == 4


def test_depth_assumptions_make_basis_and_residual_serialization_explicit() -> None:
    assumptions = _assumptions(
        basis_lanes_parallel=False,
        residual_update_cycles_per_boundary_event=2,
    )
    result = derive_uf_shot_hardware_proxy(_serialized_bundle(), assumptions=assumptions)

    assert result.lane_core_critical_path_cycles == 39 + 17
    assert result.patch_transaction_depth_cycles == 1
    assert result.residual_update_depth_cycles == 12
    assert result.conservative_parallel_depth_cycles == 69


def _six_patch_bundle() -> dict[str, object]:
    lanes: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    for lane_offset in range(12):
        depth = lane_offset + 1
        lanes.append(
            _lane(
                lane_offset=lane_offset,
                shot_id=9,
                terminal=depth,
                counters=_counters(
                    batches=depth, growth=depth, attempts=0, successes=0,
                    failures=0, forest=0, peel=0,
                ),
            )
        )
        components.append(
            {
                "global_shot_id": 9,
                "lane_offset": lane_offset,
                **_completed_component(size=2, absorbed=2, times=[1], diameter=1),
            }
        )
    patch_outcomes: list[dict[str, object]] = []
    for patch_id in range(6):
        boundary_count = patch_id + 1
        boundary_start = patch_id * 100
        patch_outcomes.append(
            {
                "patch_id": patch_id,
                "lane_outcomes": [{}, {}],
                "durable_detector_boundary": list(
                    range(boundary_start, boundary_start + boundary_count)
                ),
            }
        )
    return {
        "shot": {
            "global_shot_id": 9,
            "adapter_metrics": {
                "durable_boundary_count": 21,
                "committed_defect_count": 21,
                "patch_outcomes": patch_outcomes,
            },
        },
        "lanes": lanes,
        "components": components,
    }


def _growth_only_assumptions() -> UFParallelDepthAssumptions:
    # One cycle per iteration and nothing else, so lane depth == iterations.
    return UFParallelDepthAssumptions(
        growth_quantum_weight=1,
        syndrome_load_cycles=0,
        growing_cycles_per_iteration=1,
        controller_cycles_per_iteration=0,
        merge_settle_cycles_per_iteration=0,
        merge_cycles_per_hop=0,
        peel_cycles_per_operation=0,
        confidence_cycles_per_completed_component=0,
        patch_transaction_cycles=2,
        residual_update_cycles_per_boundary_event=1,
        lanes_per_patch=2,
        basis_lanes_parallel=True,
    )


def test_three_architecture_depths_use_exact_per_patch_boundary_work() -> None:
    result = derive_uf_shot_hardware_proxy(
        _six_patch_bundle(), assumptions=_growth_only_assumptions()
    )

    assert tuple(lane.growth_iteration_count for lane in result.lanes) == tuple(range(1, 13))
    assert result.per_patch_residual_boundary_update_work == (1, 2, 3, 4, 5, 6)
    # Per patch: max(X-depth, Z-depth) + transaction + local residual updates.
    assert result.per_patch_parallel_basis_depth_cycles == (5, 8, 11, 14, 17, 20)
    # Per patch: X-depth + Z-depth + transaction + local residual updates.
    assert result.per_patch_serial_basis_depth_cycles == (6, 11, 16, 21, 26, 31)
    assert result.parallel_lane_cores_per_patch_depth_cycles == 20
    assert result.serial_basis_patch_engines_depth_cycles == 31
    # Shared: sum(1..12) lane cycles + six transactions*2 + sum(1..6).
    assert result.fully_shared_frontend_engine_depth_cycles == 111


def test_patch_architecture_telemetry_requires_dense_six_by_two_layout() -> None:
    assumptions = _growth_only_assumptions()
    missing_patch = copy.deepcopy(_six_patch_bundle())
    missing_patch["shot"]["adapter_metrics"]["patch_outcomes"].pop()
    with pytest.raises(ValueError, match="dense"):
        derive_uf_shot_hardware_proxy(missing_patch, assumptions=assumptions)

    wrong_lane_count = copy.deepcopy(_six_patch_bundle())
    wrong_lane_count["shot"]["adapter_metrics"]["patch_outcomes"][0]["lane_outcomes"] = [{}]
    with pytest.raises(ValueError, match="exactly 2 lanes"):
        derive_uf_shot_hardware_proxy(wrong_lane_count, assumptions=assumptions)

    inconsistent_total = copy.deepcopy(_six_patch_bundle())
    inconsistent_total["shot"]["adapter_metrics"]["durable_boundary_count"] = 20
    inconsistent_total["shot"]["adapter_metrics"]["committed_defect_count"] = 20
    with pytest.raises(ValueError, match="does not reconcile with shot total"):
        derive_uf_shot_hardware_proxy(inconsistent_total, assumptions=assumptions)


def _unbounded_policy() -> UFPolicy:
    limits = BudgetLimits.unbounded_for_testing()
    return UFPolicy(Fraction(0), limits, limits)


def test_live_lane_outcome_and_asdict_shot_are_supported() -> None:
    graph = UFLaneGraph(2, (UFEdge(0, 0, 1, Fraction(1), "correction"),))
    lane = run_lane(graph, [0, 1], _unbounded_policy())
    shot = SimpleNamespace(
        lane_outcomes=(lane,),
        original_detector_count=2,
        residual_detector_count=0,
        durable_boundary_count=2,
    )

    live = derive_uf_shot_hardware_proxy(shot, assumptions=_assumptions())
    serialized = derive_uf_shot_hardware_proxy(
        {
            "lane_outcomes": [dataclasses.asdict(lane)],
            "original_detector_count": 2,
            "residual_detector_count": 0,
        },
        assumptions=_assumptions(),
    )

    # The pair met at t = 1/2: one iteration at quantum 1/2, diameter 1.
    assert live.lanes[0].growth_depth_weight == Fraction(1, 2)
    assert live.lanes[0].growth_iteration_count == 1
    assert live.lanes[0].maximum_forest_diameter_hops == 1
    assert live.lanes[0].merge_depth_cycles == 1
    assert live.maximum_completed_component_defect_count == 2
    assert live.lanes[0].synchronous_event_batch_count == 1
    assert live.lanes[0].union_merge_attempt_count == 1
    # 1 load + 1*(1+2+1) growth + 1 merge + 1 peel + 1 confidence.
    assert live.lanes[0].conservative_parallel_depth_cycles == 8
    assert live.conservative_parallel_depth_cycles == 11
    assert dataclasses.asdict(serialized) == dataclasses.asdict(live)


def test_censored_component_uses_absorbed_vertices_as_diameter_bound() -> None:
    lane = {
        "status": "censored",
        "counters": _counters(
            batches=2, growth=2, attempts=1, successes=1, failures=0, forest=1, peel=0
        ),
        "terminal_event_time": 1,
        "censored_components": [
            {
                "current_defects": [0, 2, 3],
                "absorbed_vertices": [0, 1, 2, 3],
                "event_batch_ids": [1],
                "event_batch_times": [1],
                "simultaneous_event_batch_count": 1,
            }
        ],
    }
    result = derive_uf_lane_hardware_proxy(lane, assumptions=_assumptions(growth_quantum_weight=1))

    assert result.maximum_completed_component_defect_count == 0
    assert result.maximum_censored_component_defect_lower_bound == 3
    assert result.maximum_observed_component_defect_count == 3
    assert result.confidence_depth_cycles == 0
    # No forest diameter is retained for a censored snapshot; the chain bound
    # of absorbed_vertices - 1 stands in for it.
    assert result.maximum_forest_diameter_hops == 3
    assert result.growth_iteration_count == 1
    assert result.merge_depth_cycles == 3


def test_proxy_validation_fails_closed_on_inconsistent_or_incomplete_rows() -> None:
    assumptions = _assumptions()
    completed = _counters(
        batches=1, growth=1, attempts=1, successes=1, failures=0, forest=1, peel=1
    )

    inconsistent = {
        "status": "completed",
        "counters": dict(completed, union_attempt_count=2),
        "terminal_event_time": 1,
        "completed_components": [
            _completed_component(size=2, absorbed=2, times=[1], diameter=1)["adapter"]
        ],
    }
    with pytest.raises(ValueError, match="union attempts"):
        derive_uf_lane_hardware_proxy(inconsistent, assumptions=assumptions)

    missing_components = {"status": "completed", "counters": completed, "terminal_event_time": 1}
    with pytest.raises(ValueError, match="component telemetry"):
        derive_uf_lane_hardware_proxy(missing_components, assumptions=assumptions)

    missing_terminal = {
        "status": "completed",
        "counters": completed,
        "completed_components": [
            _completed_component(size=2, absorbed=2, times=[1], diameter=1)["adapter"]
        ],
    }
    with pytest.raises(ValueError, match="terminal_event_time"):
        derive_uf_lane_hardware_proxy(missing_terminal, assumptions=assumptions)

    missing_diameter = copy.deepcopy(missing_terminal)
    missing_diameter["terminal_event_time"] = 1
    del missing_diameter["completed_components"][0]["forest_diameter_hops"]
    with pytest.raises(ValueError, match="forest_diameter_hops"):
        derive_uf_lane_hardware_proxy(missing_diameter, assumptions=assumptions)

    event_after_terminal = copy.deepcopy(missing_terminal)
    event_after_terminal["terminal_event_time"] = Fraction(1, 2)
    with pytest.raises(ValueError, match="terminal"):
        derive_uf_lane_hardware_proxy(event_after_terminal, assumptions=assumptions)

    zero_size = {
        "status": "completed",
        "counters": _counters(batches=1, growth=1, attempts=0, successes=0, failures=0, forest=0, peel=0),
        "terminal_event_time": 1,
        "completed_components": [
            _completed_component(size=0, absorbed=1, times=[1], diameter=0)["adapter"]
        ],
    }
    with pytest.raises(ValueError, match="cluster_defect_count"):
        derive_uf_lane_hardware_proxy(zero_size, assumptions=assumptions)

    defects_exceed_vertices = copy.deepcopy(zero_size)
    defects_exceed_vertices["completed_components"][0]["cluster_defect_count"] = 2
    with pytest.raises(ValueError, match="absorbed_vertex_count"):
        derive_uf_lane_hardware_proxy(defects_exceed_vertices, assumptions=assumptions)

    empty_with_work = {
        "status": "empty",
        "counters": _counters(batches=1, growth=1, attempts=0, successes=0, failures=0, forest=0, peel=0),
        "terminal_event_time": 0,
    }
    with pytest.raises(ValueError, match="empty lane"):
        derive_uf_lane_hardware_proxy(empty_with_work, assumptions=assumptions)

    censored_with_completed = {
        "status": "censored",
        "counters": _counters(batches=1, growth=1, attempts=0, successes=0, failures=0, forest=0, peel=0),
        "terminal_event_time": 1,
        "completed_components": [
            _completed_component(size=1, absorbed=1, times=[1], diameter=0)["adapter"]
        ],
        "censored_components": [
            {
                "current_defects": [0],
                "absorbed_vertices": [0],
                "event_batch_ids": [1],
                "event_batch_times": [1],
                "simultaneous_event_batch_count": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="censored lane cannot"):
        derive_uf_lane_hardware_proxy(censored_with_completed, assumptions=assumptions)


def test_empty_lane_has_zero_depth_without_terminal_time() -> None:
    lane = {
        "status": "empty",
        "counters": _counters(batches=0, growth=0, attempts=0, successes=0, failures=0, forest=0, peel=0),
    }
    result = derive_uf_lane_hardware_proxy(lane, assumptions=_assumptions())
    assert result.growth_depth_weight == 0
    assert result.growth_iteration_count == 0
    assert result.merge_depth_cycles == 0
    assert result.conservative_parallel_depth_cycles == 1  # syndrome load only


def test_complete_depth_is_unavailable_when_residual_work_is_not_retained() -> None:
    bundle = _serialized_bundle()
    del bundle["shot"]["adapter_metrics"]
    result = derive_uf_shot_hardware_proxy(bundle, assumptions=_assumptions())

    assert result.lane_core_critical_path_cycles == 39
    assert result.residual_boundary_update_work is None
    assert result.residual_update_depth_cycles is None
    assert result.conservative_parallel_depth_cycles is None


def test_assumption_validation_fails_closed() -> None:
    with pytest.raises(TypeError, match="syndrome_load_cycles"):
        _assumptions(syndrome_load_cycles=True)
    with pytest.raises(ValueError, match="lanes_per_patch"):
        _assumptions(lanes_per_patch=0)
    with pytest.raises(ValueError, match="growth_quantum_weight"):
        _assumptions(growth_quantum_weight=0)
    with pytest.raises(TypeError):
        UFParallelDepthAssumptions()  # the quantum has no default
    assert _assumptions(growth_quantum_weight=0.25).growth_quantum_weight == Fraction(1, 4)
