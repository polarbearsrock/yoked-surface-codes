from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding._matched_frontend_hardware import (
    ARCHITECTURES,
    CYCLE_BUDGETS,
    RAW_WORK_METRICS,
    _correctness_cube,
    _histogram,
    _lane_architectures,
    _pair_order_architectures,
    _pairwise_contingencies,
    _promatch_shot_metrics,
    analyze_hardware_replay,
    load_hardware_replay_npz,
    render_hardware_report,
    validate_reference_results,
    validate_hardware_npz_provenance,
    write_hardware_reanalysis_artifacts,
    write_hardware_replay_artifacts,
)
from yoked.decoding._promatch_layout import L1WindowDomain


def _arrays(shots: int = 8) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "shot_id": np.arange(shots, dtype=np.int64),
        "global_failed": np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64),
        "promatch_failed": np.asarray([0, 1, 0, 1, 0, 1, 1, 1], dtype=np.int64),
        "pinball_failed": np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64),
        "union_find_failed": np.asarray([0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int64),
        "original_detector_event_count": np.arange(10, 10 + shots, dtype=np.int64),
        "pinball_fired_primitive_count": np.arange(shots, dtype=np.int64),
        "union_find_maximum_observed_component_defect_count": np.asarray(
            [0, 2, 4, 4, 6, 8, 10, 12], dtype=np.int64
        ),
        "union_find_maximum_completed_component_defect_count": np.asarray(
            [0, 2, 4, 4, 6, 8, 8, 10], dtype=np.int64
        ),
        "union_find_maximum_censored_component_defect_lower_bound": np.asarray(
            [0, 0, 0, 0, 0, 0, 10, 12], dtype=np.int64
        ),
        "union_find_maximum_absorbed_vertex_count": np.asarray(
            [0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64
        ),
    }
    for arm, reduction in (
        ("global", 0),
        ("promatch", 4),
        ("pinball", 1),
        ("union_find", 5),
    ):
        result[f"{arm}_residual_detector_event_count"] = np.maximum(
            0, result["original_detector_event_count"] - reduction
        )
    for architecture_index, architecture in enumerate(ARCHITECTURES, start=1):
        result[f"promatch_cycles_{architecture}"] = (
            np.arange(shots, dtype=np.int64) * 100 + 32 * architecture_index
        )
        result[f"pinball_stream_cycles_{architecture}"] = np.full(
            shots, 32 * architecture_index, dtype=np.int64
        )
        result[f"pinball_stream_offline_or_cycles_{architecture}"] = np.full(
            shots, 64 * architecture_index, dtype=np.int64
        )
        result[f"pinball_post_final_input_tail_cycles_{architecture}"] = np.full(
            shots, 9 * architecture_index, dtype=np.int64
        )
        result[f"union_find_cycles_{architecture}"] = (
            np.arange(shots, dtype=np.int64) * 50 + 64 * architecture_index
        )
    for metrics in RAW_WORK_METRICS.values():
        for metric_index, (_, source_array, _, _) in enumerate(metrics, start=1):
            result.setdefault(
                source_array,
                np.arange(shots, dtype=np.int64) + metric_index,
            )
    return result


def _reference(arrays: dict[str, np.ndarray]) -> dict:
    arms = ("global", "promatch", "pinball", "union_find")
    return {
        "shots": len(arrays["shot_id"]),
        "accuracy": {
            "correctness_cube": _correctness_cube(arrays),
            "marginals": {
                arm: {"failures": int(np.sum(arrays[f"{arm}_failed"]))}
                for arm in arms
            }
        },
        "raw_inputs": {
            "pairwise_contingencies": _pairwise_contingencies(arrays),
            "telemetry": {
                arm: {
                    "residual_event_sum": int(
                        np.sum(arrays[f"{arm}_residual_detector_event_count"])
                    ),
                    "residual_hw_histogram": _histogram(
                        arrays[f"{arm}_residual_detector_event_count"]
                    ),
                }
                for arm in arms
            }
        },
    }


def test_lane_architecture_aggregation() -> None:
    lanes = {
        (patch, basis): 10 * patch + (1 if basis == "X" else 2)
        for patch in range(3)
        for basis in ("X", "Z")
    }
    assert _lane_architectures(lanes, patches=3) == {
        "fully_parallel_12_lane": 22,
        "patch_shared_6_engine": 43,
        "fully_shared_1_engine": 69,
    }
    assert _pair_order_architectures([1, 2, 11, 12, 21, 22], patches=3) == {
        "fully_parallel_12_lane": 22,
        "patch_shared_6_engine": 43,
        "fully_shared_1_engine": 69,
    }


def test_promatch_below_limit_none_proxy_is_zero_but_active_none_is_rejected() -> None:
    domain = L1WindowDomain(patch_id=0, check_basis="X", window_id=0)
    below = SimpleNamespace(
        domain_stats={
            domain: SimpleNamespace(status="below-limit", hardware_proxy=None)
        }
    )
    metrics = _promatch_shot_metrics(below, patches=1)
    assert metrics["promatch_cycles_fully_parallel_12_lane"] == 0
    assert metrics["promatch_selection_round_count"] == 0

    active = SimpleNamespace(
        domain_stats={domain: SimpleNamespace(status="success", hardware_proxy=None)}
    )
    with pytest.raises(ValueError, match="active ProMatch domain"):
        _promatch_shot_metrics(active, patches=1)


def test_analysis_keeps_pinball_proxies_separate_and_applies_deadlines() -> None:
    arrays = _arrays()
    analysis = analyze_hardware_replay(
        arrays,
        source_identity={"cell_id": "synthetic"},
        reference_analysis=_reference(arrays),
        enforce_frozen_expected=False,
    )
    pinball = analysis["cycle_proxy_distributions"]["pinball"]
    assert set(pinball) == {
        "ideal_stream_cycle_lower_bound",
        "stream_plus_offline_full_history_or_depth_proxy",
        "post_final_input_tail_lower_bound",
    }
    deadline = analysis["illustrative_deadline_sensitivity"]["promatch"][
        "fully_parallel_12_lane"
    ]["budget_rows"]["64"]
    assert deadline["timeout_shots"] == 7
    assert deadline["strict_failure_rate"] == pytest.approx(7 / 8)
    assert deadline["global_bypass_failure_rate"] == pytest.approx(4 / 8)
    assert deadline["global_bypass_residual_detector_events"] == 104
    assert deadline["global_bypass_detector_event_reduction_percentage"] == (
        pytest.approx(100 * 4 / 108)
    )
    quantiles = analysis["illustrative_deadline_sensitivity"]["union_find"][
        "fully_parallel_12_lane"
    ]["quantile_policy_rows"]
    assert set(quantiles) == {"p50", "p90", "p95", "p99", "max"}
    assert quantiles["max"]["timeout_shots"] == 0
    assert CYCLE_BUDGETS == (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    raw = analysis["raw_hardware_work_distributions"]["arms"]
    assert raw["promatch"]["selection_rounds"]["source_array"] == (
        "promatch_selection_round_count"
    )
    activation = raw["pinball"]["activation_operand_uses"]
    assert "not physical memory reads" in activation["label"]
    assert activation["total_over_shots"] == int(
        np.sum(arrays["pinball_activation_bit_read_count"])
    )
    assert analysis["union_find_cluster_sizes"][
        "maximum_observed_component_defect_count"
    ]["distribution"]["max"] == 12
    report = render_hardware_report(analysis)
    assert "Cycle fields are architecture proxies" in report
    assert "Pinball stream + offline full-history OR-depth proxy" in report
    assert report.index("## Raw hardware-work distributions") < report.index(
        "## Secondary synthetic cycle/depth models"
    )
    assert "Global-bypass residual detector-event" in report


def test_reference_validation_rejects_one_changed_decision() -> None:
    arrays = _arrays()
    reference = _reference(arrays)
    changed = dict(arrays)
    changed["promatch_failed"] = arrays["promatch_failed"].copy()
    changed["promatch_failed"][0] ^= 1
    with pytest.raises(ValueError, match="promatch failures"):
        validate_reference_results(
            changed, reference, enforce_frozen_expected=False
        )


def test_reference_validation_rejects_changed_cube_with_same_marginals() -> None:
    arrays = _arrays()
    reference = _reference(arrays)
    changed = dict(arrays)
    changed["promatch_failed"] = arrays["promatch_failed"].copy()
    changed["promatch_failed"][[0, 3]] = changed["promatch_failed"][[3, 0]]
    assert np.sum(changed["promatch_failed"]) == np.sum(arrays["promatch_failed"])
    with pytest.raises(ValueError, match="correctness cube"):
        validate_reference_results(
            changed, reference, enforce_frozen_expected=False
        )


def test_artifact_writer_is_fresh_only(tmp_path) -> None:
    arrays = _arrays()
    analysis = analyze_hardware_replay(
        arrays,
        source_identity={"cell_id": "synthetic"},
        reference_analysis=_reference(arrays),
        enforce_frozen_expected=False,
    )
    output = tmp_path / "hardware"
    paths = write_hardware_replay_artifacts(
        output,
        arrays=arrays,
        analysis=analysis,
        provenance={"test": True},
    )
    assert set(paths) == {"per_shot", "analysis", "report", "provenance"}
    with np.load(output / "per_shot.npz", allow_pickle=False) as loaded:
        assert np.array_equal(loaded["shot_id"], arrays["shot_id"])
    safely_loaded = load_hardware_replay_npz(output / "per_shot.npz")
    source_provenance = json.loads((output / "provenance.json").read_text())
    binding = validate_hardware_npz_provenance(
        output / "per_shot.npz", safely_loaded, source_provenance
    )
    source_before = (output / "per_shot.npz").read_bytes()
    reanalysis = tmp_path / "hardware-v2"
    reanalysis_paths = write_hardware_reanalysis_artifacts(
        reanalysis,
        analysis=analysis,
        provenance={"source_per_shot_npz": binding},
    )
    assert set(reanalysis_paths) == {"analysis", "report", "provenance"}
    assert not (reanalysis / "per_shot.npz").exists()
    assert (output / "per_shot.npz").read_bytes() == source_before
    with pytest.raises(FileExistsError, match="fresh absent"):
        write_hardware_replay_artifacts(
            output,
            arrays=arrays,
            analysis=analysis,
            provenance={"test": True},
        )


def test_safe_npz_loader_rejects_object_arrays(tmp_path) -> None:
    unsafe = tmp_path / "unsafe.npz"
    np.savez(unsafe, shot_id=np.asarray([object()], dtype=object))
    with pytest.raises(ValueError, match="unsafe NumPy array"):
        load_hardware_replay_npz(unsafe)


def _uf_shard_row() -> dict:
    from yoked.decoding._matched_frontend_hardware import _uf_shot_metrics  # noqa: F401

    zero = {
        "growth_event_count": 0,
        "simultaneous_event_batch_count": 0,
        "union_attempt_count": 0,
        "successful_union_count": 0,
        "failed_union_count": 0,
        "forest_edge_count": 0,
        "peel_operation_count": 0,
    }
    lane0 = {
        "status": "completed",
        "terminal_event_time": {"mantissa": 3, "exponent": 0},
        "counters": {
            **zero,
            "growth_event_count": 3,
            "simultaneous_event_batch_count": 2,
            "union_attempt_count": 2,
            "successful_union_count": 2,
            "forest_edge_count": 3,
            "peel_operation_count": 2,
        },
        "completed_components": [
            {
                "cluster_defect_count": 2,
                "absorbed_vertex_count": 3,
                "absorbed_vertices": [0, 1, 2],
                "simultaneous_event_batch_count": 2,
                "event_batch_ids": [1, 2],
                "event_batch_times": [
                    {"mantissa": 1, "exponent": 0},
                    {"mantissa": 3, "exponent": 0},
                ],
                # Two correction edges forming the chain 10-11-12 plus the
                # boundary incidence 9 at vertex 10, which is not a tree edge.
                "forest_edge_ids": [3, 5, 9],
            }
        ],
        "censored_components": [],
    }
    lane1 = {
        "status": "empty",
        "terminal_event_time": {"mantissa": 0, "exponent": 0},
        "counters": dict(zero),
        "completed_components": [],
        "censored_components": [],
    }
    return {
        "global_shot_id": 4,
        "adapter_metrics": {
            "original_detector_count": 2,
            "residual_detector_count": 0,
            "committed_defect_count": 2,
            "durable_boundary_count": 2,
            "patch_outcomes": [
                {
                    "patch_id": 0,
                    "lane_outcomes": [lane0, lane1],
                    "durable_detector_boundary": [10, 12],
                }
            ],
        },
    }


_ENDPOINTS = {3: (10, 11), 5: (11, 12), 9: (10, None)}


def test_forest_diameters_are_rebuilt_from_canonical_forest_edges() -> None:
    from yoked.decoding._matched_frontend_hardware import annotate_forest_diameters

    row = _uf_shard_row()
    annotated = annotate_forest_diameters(row, _ENDPOINTS)
    component = annotated["adapter_metrics"]["patch_outcomes"][0]["lane_outcomes"][0][
        "completed_components"
    ][0]
    assert component["forest_diameter_hops"] == 2
    # The source row is left untouched and an already-present value is kept.
    assert "forest_diameter_hops" not in row["adapter_metrics"]["patch_outcomes"][0][
        "lane_outcomes"
    ][0]["completed_components"][0]
    component["forest_diameter_hops"] = 7
    assert annotate_forest_diameters(annotated, _ENDPOINTS) == annotated

    # A lone defect whose forest is only its boundary incidence has diameter 0.
    single = _uf_shard_row()
    lane = single["adapter_metrics"]["patch_outcomes"][0]["lane_outcomes"][0]
    lane["completed_components"][0].update(
        {"absorbed_vertex_count": 1, "absorbed_vertices": [0], "forest_edge_ids": [9]}
    )
    assert annotate_forest_diameters(single, _ENDPOINTS)["adapter_metrics"][
        "patch_outcomes"
    ][0]["lane_outcomes"][0]["completed_components"][0]["forest_diameter_hops"] == 0

    # A multi-vertex component without correction forest edges is malformed.
    broken = _uf_shard_row()
    broken["adapter_metrics"]["patch_outcomes"][0]["lane_outcomes"][0][
        "completed_components"
    ][0]["forest_edge_ids"] = [9]
    with pytest.raises(ValueError, match="forest"):
        annotate_forest_diameters(broken, _ENDPOINTS)


def test_uf_shot_metrics_report_growth_depth_iterations_and_merge_cycles() -> None:
    from yoked.decoding._matched_frontend_hardware import _uf_shot_metrics
    from yoked.decoding._patch_uf_hw_proxy import UFParallelDepthAssumptions

    metrics = _uf_shot_metrics(
        _uf_shard_row(),
        patches=1,
        assumptions=UFParallelDepthAssumptions(growth_quantum_weight=1),
        edge_endpoints=_ENDPOINTS,
    )
    assert metrics["union_find_growth_iteration_count"] == 3
    assert metrics["union_find_growth_depth_milli_weight_units"] == 3000
    assert metrics["union_find_maximum_forest_diameter_hops"] == 2
    # Events in iterations 1 and 3 each flood a diameter-2 cluster.
    assert metrics["union_find_merge_depth_cycles"] == 4
    # Lane 0: 1 load + 3*(1+2+1) + 4 merge + 2 peel + 1 confidence = 20;
    # patch: 20 + 1 transaction + 2 residual updates.
    assert metrics["union_find_cycles_fully_parallel_12_lane"] == 23
    assert metrics["union_find_maximum_lane_synchronous_event_batches"] == 2
