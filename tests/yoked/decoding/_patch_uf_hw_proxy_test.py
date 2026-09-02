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
)
from yoked.decoding._patch_uf_reference import (
    BudgetLimits,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
)


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


def _completed_component(
    *, size: int, absorbed: int, batches: int
) -> dict[str, object]:
    return {
        "state_collection": "completed_components",
        "adapter": {
            "cluster_defect_count": size,
            "absorbed_vertex_count": absorbed,
            "simultaneous_event_batch_count": batches,
            "event_batch_ids": list(range(1, batches + 1)),
        },
    }


def _serialized_bundle() -> dict[str, object]:
    lanes = [
        {
            "global_shot_id": 7,
            "lane_offset": 0,
            "adapter": {
                "status": "completed",
                "counters": _counters(
                    batches=3,
                    growth=7,
                    attempts=5,
                    successes=4,
                    failures=1,
                    forest=4,
                    peel=4,
                ),
            },
        },
        {
            "global_shot_id": 7,
            "lane_offset": 1,
            "adapter": {
                "status": "completed",
                "counters": _counters(
                    batches=2,
                    growth=2,
                    attempts=1,
                    successes=1,
                    failures=0,
                    forest=1,
                    peel=1,
                ),
            },
        },
    ]
    components = [
        {
            "global_shot_id": 7,
            "lane_offset": 0,
            **_completed_component(size=5, absorbed=7, batches=2),
        },
        {
            "global_shot_id": 7,
            "lane_offset": 0,
            **_completed_component(size=2, absorbed=2, batches=1),
        },
        {
            "global_shot_id": 7,
            "lane_offset": 1,
            **_completed_component(size=2, absorbed=3, batches=1),
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


def test_serialized_rows_produce_raw_work_and_default_depth_proxy() -> None:
    result = derive_uf_shot_hardware_proxy(_serialized_bundle())

    assert result.global_shot_id == 7
    assert (result.lane_count, result.patch_count, result.active_lane_count) == (
        2,
        1,
        2,
    )
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

    # Default batch cost is growth + event resolution + merge + settle = 4.
    # Lane 0: load 1 + 3*4 + four serial peel ops + two confidence checks.
    assert result.lanes[0].conservative_parallel_depth_cycles == 19
    # The basis lanes run in parallel, followed by one patch transaction cycle
    # and six serialized residual-boundary updates.
    assert result.lane_core_critical_path_cycles == 19
    assert result.residual_boundary_update_work == 6
    assert result.residual_update_depth_cycles == 6
    assert result.conservative_parallel_depth_cycles == 26
    assert result.per_patch_residual_boundary_update_work is None
    assert result.parallel_lane_cores_per_patch_depth_cycles is None
    assert result.serial_basis_patch_engines_depth_cycles is None
    # A fully shared engine needs only the exact shot total, not a guessed
    # distribution of that total across patches.
    assert result.fully_shared_frontend_engine_depth_cycles == 37


def test_depth_assumptions_make_basis_and_residual_serialization_explicit() -> None:
    assumptions = dataclasses.replace(
        UFParallelDepthAssumptions(),
        basis_lanes_parallel=False,
        residual_update_cycles_per_boundary_event=2,
    )
    result = derive_uf_shot_hardware_proxy(
        _serialized_bundle(), assumptions=assumptions
    )

    assert result.lane_core_critical_path_cycles == 19 + 11
    assert result.patch_transaction_depth_cycles == 1
    assert result.residual_update_depth_cycles == 12
    assert result.conservative_parallel_depth_cycles == 43


def _six_patch_bundle() -> dict[str, object]:
    lanes: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    for lane_offset in range(12):
        batches = lane_offset + 1
        lanes.append(
            {
                "global_shot_id": 9,
                "lane_offset": lane_offset,
                "adapter": {
                    "status": "completed",
                    "counters": _counters(
                        batches=batches,
                        growth=batches,
                        attempts=0,
                        successes=0,
                        failures=0,
                        forest=0,
                        peel=0,
                    ),
                },
            }
        )
        components.append(
            {
                "global_shot_id": 9,
                "lane_offset": lane_offset,
                **_completed_component(size=2, absorbed=2, batches=1),
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


def test_three_architecture_depths_use_exact_per_patch_boundary_work() -> None:
    assumptions = UFParallelDepthAssumptions(
        syndrome_load_cycles=0,
        growth_cycles_per_synchronous_batch=1,
        event_resolution_cycles_per_synchronous_batch=0,
        merge_cycles_per_synchronous_batch=0,
        settle_cycles_per_synchronous_batch=0,
        peel_cycles_per_operation=0,
        confidence_cycles_per_completed_component=0,
        patch_transaction_cycles=2,
        residual_update_cycles_per_boundary_event=1,
        lanes_per_patch=2,
        basis_lanes_parallel=True,
    )
    result = derive_uf_shot_hardware_proxy(
        _six_patch_bundle(), assumptions=assumptions
    )

    assert result.per_patch_residual_boundary_update_work == (1, 2, 3, 4, 5, 6)
    # Per patch: max(X-depth, Z-depth) + transaction + local residual updates.
    assert result.per_patch_parallel_basis_depth_cycles == (
        5,
        8,
        11,
        14,
        17,
        20,
    )
    # Per patch: X-depth + Z-depth + transaction + local residual updates.
    assert result.per_patch_serial_basis_depth_cycles == (
        6,
        11,
        16,
        21,
        26,
        31,
    )
    assert result.parallel_lane_cores_per_patch_depth_cycles == 20
    assert result.serial_basis_patch_engines_depth_cycles == 31
    # Shared: sum(1..12) lane cycles + six transactions*2 + sum(1..6).
    assert result.fully_shared_frontend_engine_depth_cycles == 111


def test_patch_architecture_telemetry_requires_dense_six_by_two_layout() -> None:
    missing_patch = copy.deepcopy(_six_patch_bundle())
    missing_patch["shot"]["adapter_metrics"]["patch_outcomes"].pop()
    with pytest.raises(ValueError, match="dense"):
        derive_uf_shot_hardware_proxy(missing_patch)

    wrong_lane_count = copy.deepcopy(_six_patch_bundle())
    wrong_lane_count["shot"]["adapter_metrics"]["patch_outcomes"][0][
        "lane_outcomes"
    ] = [{}]
    with pytest.raises(ValueError, match="exactly 2 lanes"):
        derive_uf_shot_hardware_proxy(wrong_lane_count)

    inconsistent_total = copy.deepcopy(_six_patch_bundle())
    inconsistent_total["shot"]["adapter_metrics"]["durable_boundary_count"] = 20
    inconsistent_total["shot"]["adapter_metrics"]["committed_defect_count"] = 20
    with pytest.raises(ValueError, match="does not reconcile with shot total"):
        derive_uf_shot_hardware_proxy(inconsistent_total)


def _unbounded_policy() -> UFPolicy:
    limits = BudgetLimits.unbounded_for_testing()
    return UFPolicy(Fraction(0), limits, limits)


def test_live_lane_outcome_and_asdict_shot_are_supported() -> None:
    graph = UFLaneGraph(
        2, (UFEdge(0, 0, 1, Fraction(1), "correction"),)
    )
    lane = run_lane(graph, [0, 1], _unbounded_policy())
    shot = SimpleNamespace(
        lane_outcomes=(lane,),
        original_detector_count=2,
        residual_detector_count=0,
        durable_boundary_count=2,
    )

    live = derive_uf_shot_hardware_proxy(shot)
    serialized = derive_uf_shot_hardware_proxy(
        {
            "lane_outcomes": [dataclasses.asdict(lane)],
            "original_detector_count": 2,
            "residual_detector_count": 0,
        }
    )

    assert live.maximum_completed_component_defect_count == 2
    assert live.maximum_component_event_batch_count == 1
    assert live.lanes[0].synchronous_event_batch_count == 1
    assert live.lanes[0].union_merge_attempt_count == 1
    assert live.lanes[0].conservative_parallel_depth_cycles == 7
    assert live.conservative_parallel_depth_cycles == 10
    assert dataclasses.asdict(serialized) == dataclasses.asdict(live)


def test_censored_component_size_remains_named_as_a_lower_bound() -> None:
    lane = {
        "status": "censored",
        "counters": _counters(
            batches=2,
            growth=2,
            attempts=1,
            successes=1,
            failures=0,
            forest=1,
            peel=0,
        ),
        "censored_components": [
            {
                "current_defects": [0, 2, 3],
                "absorbed_vertices": [0, 1, 2, 3],
                "event_batch_ids": [1],
                "simultaneous_event_batch_count": 1,
            }
        ],
    }
    result = derive_uf_lane_hardware_proxy(lane)

    assert result.maximum_completed_component_defect_count == 0
    assert result.maximum_censored_component_defect_lower_bound == 3
    assert result.maximum_observed_component_defect_count == 3
    assert result.confidence_depth_cycles == 0


def test_proxy_validation_fails_closed_on_inconsistent_or_incomplete_rows() -> None:
    inconsistent = {
        "status": "completed",
        "counters": _counters(
            batches=1,
            growth=1,
            attempts=2,
            successes=1,
            failures=0,
            forest=1,
            peel=1,
        ),
        "completed_components": [
            _completed_component(size=2, absorbed=2, batches=1)["adapter"]
        ],
    }
    with pytest.raises(ValueError, match="union attempts"):
        derive_uf_lane_hardware_proxy(inconsistent)

    missing_components = {
        "status": "completed",
        "counters": _counters(
            batches=1,
            growth=1,
            attempts=1,
            successes=1,
            failures=0,
            forest=1,
            peel=1,
        ),
    }
    with pytest.raises(ValueError, match="component telemetry"):
        derive_uf_lane_hardware_proxy(missing_components)

    zero_size = {
        "status": "completed",
        "counters": _counters(
            batches=1,
            growth=1,
            attempts=0,
            successes=0,
            failures=0,
            forest=0,
            peel=0,
        ),
        "completed_components": [
            _completed_component(size=0, absorbed=1, batches=1)["adapter"]
        ],
    }
    with pytest.raises(ValueError, match="cluster_defect_count"):
        derive_uf_lane_hardware_proxy(zero_size)

    defects_exceed_vertices = {
        "status": "completed",
        "counters": _counters(
            batches=1,
            growth=1,
            attempts=0,
            successes=0,
            failures=0,
            forest=0,
            peel=0,
        ),
        "completed_components": [
            _completed_component(size=2, absorbed=1, batches=1)["adapter"]
        ],
    }
    with pytest.raises(ValueError, match="absorbed_vertex_count"):
        derive_uf_lane_hardware_proxy(defects_exceed_vertices)

    empty_with_work = {
        "status": "empty",
        "counters": _counters(
            batches=1,
            growth=1,
            attempts=0,
            successes=0,
            failures=0,
            forest=0,
            peel=0,
        ),
    }
    with pytest.raises(ValueError, match="empty lane"):
        derive_uf_lane_hardware_proxy(empty_with_work)

    censored_with_completed = {
        "status": "censored",
        "counters": _counters(
            batches=1,
            growth=1,
            attempts=0,
            successes=0,
            failures=0,
            forest=0,
            peel=0,
        ),
        "completed_components": [
            _completed_component(size=1, absorbed=1, batches=1)["adapter"]
        ],
        "censored_components": [
            {
                "current_defects": [0],
                "absorbed_vertices": [0],
                "event_batch_ids": [1],
                "simultaneous_event_batch_count": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="censored lane cannot"):
        derive_uf_lane_hardware_proxy(censored_with_completed)


def test_complete_depth_is_unavailable_when_residual_work_is_not_retained() -> None:
    bundle = _serialized_bundle()
    del bundle["shot"]["adapter_metrics"]
    result = derive_uf_shot_hardware_proxy(bundle)

    assert result.lane_core_critical_path_cycles == 19
    assert result.residual_boundary_update_work is None
    assert result.residual_update_depth_cycles is None
    assert result.conservative_parallel_depth_cycles is None


def test_assumption_validation_rejects_boolean_cycles_and_zero_lane_width() -> None:
    with pytest.raises(TypeError, match="syndrome_load_cycles"):
        UFParallelDepthAssumptions(syndrome_load_cycles=True)
    with pytest.raises(ValueError, match="lanes_per_patch"):
        UFParallelDepthAssumptions(lanes_per_patch=0)
