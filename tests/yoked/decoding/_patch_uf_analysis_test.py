from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from yoked.decoding._patch_uf_analysis import (
    AnalysisConfig,
    analyze_verified_collection,
    write_analysis_bundle,
)
from yoked.decoding._patch_uf_experiment import VerifiedCollection
from yoked.decoding._patch_uf_stats import ShotClusterRecord


def _counters(**changes: int) -> dict[str, int]:
    result = {
        "growth_event_count": 0,
        "simultaneous_event_batch_count": 0,
        "union_attempt_count": 0,
        "successful_union_count": 0,
        "failed_union_count": 0,
        "forest_edge_count": 0,
        "peel_operation_count": 0,
        "heap_push_count": 0,
        "heap_pop_count": 0,
        "stale_heap_pop_count": 0,
        "heap_operation_count": 0,
        "peak_heap_size": 0,
        "peak_live_component_count": 0,
        "temporary_memory_units": 0,
    }
    result.update(changes)
    return result


def _verified() -> VerifiedCollection:
    summary = {
        "experiment_id": "11" * 32,
        "protocol_self_sha256": "22" * 32,
        "payload_sha256": "33" * 32,
        "stage": "characterization",
        "cell_id": "selected-d7",
        "provenance": {"num_observables": 2},
        "shots": 2,
        "lane_records": 4,
        "component_records": 3,
        "paired_contingency": {"a": 1, "b": 0, "c": 1, "d": 0},
        "prediction_agreements": 1,
        "hrlk_joint_histogram": [[1, 1, 1, 0, 1], [3, 1, 3, 2, 1]],
    }
    shots = (
        {
            "global_shot_id": 0,
            "global_prediction_hex": "00",
            "treatment_prediction_hex": "00",
            "actual_observables_hex": "00",
            "global_failed": False,
            "treatment_failed": False,
            "prediction_agreement": True,
            "lane_start": 0,
            "lane_count": 2,
            "component_start": 0,
            "component_count": 2,
            "adapter_metrics": {
                "original_detector_count": 3,
                "residual_detector_count": 1,
                "lane_owned_detector_count": 3,
                "committed_defect_count": 2,
                "cluster_summary_complete": True,
                "lane_original_detector_counts": [3, 0],
                "lane_residual_detector_counts": [1, 0],
                "original_body_detector_count": 2,
                "residual_body_detector_count": 0,
                "original_terminal_detector_count": 1,
                "residual_terminal_detector_count": 1,
                "original_yoke_detector_count": 0,
                "residual_yoke_detector_count": 0,
            },
        },
        {
            "global_shot_id": 1,
            "global_prediction_hex": "01",
            "treatment_prediction_hex": "00",
            "actual_observables_hex": "00",
            "global_failed": True,
            "treatment_failed": False,
            "prediction_agreement": False,
            "lane_start": 0,
            "lane_count": 2,
            "component_start": 0,
            "component_count": 1,
            "adapter_metrics": {
                "original_detector_count": 1,
                "residual_detector_count": 1,
                "lane_owned_detector_count": 1,
                "committed_defect_count": 0,
                "cluster_summary_complete": False,
                "lane_original_detector_counts": [1, 0],
                "lane_residual_detector_counts": [1, 0],
                "original_body_detector_count": 1,
                "residual_body_detector_count": 1,
                "original_terminal_detector_count": 0,
                "residual_terminal_detector_count": 0,
                "original_yoke_detector_count": 0,
                "residual_yoke_detector_count": 0,
            },
        },
    )
    lane0_counters = _counters(
        growth_event_count=2,
        simultaneous_event_batch_count=1,
        union_attempt_count=1,
        successful_union_count=1,
        forest_edge_count=1,
        peel_operation_count=1,
        heap_push_count=3,
        heap_pop_count=1,
        heap_operation_count=4,
        peak_heap_size=3,
        peak_live_component_count=2,
        temporary_memory_units=3,
    )
    empty = {
        "status": "empty",
        "censor_reason": None,
        "counters": _counters(),
        "completed_components": [],
        "censored_components": [],
        "last_complete_batch_id": None,
    }
    lanes = (
        {
            "global_shot_id": 0,
            "lane_offset": 0,
            "adapter": {
                "status": "completed",
                "censor_reason": None,
                "counters": lane0_counters,
                "completed_components": [{}, {}],
                "censored_components": [],
                "last_complete_batch_id": 1,
            },
        },
        {"global_shot_id": 0, "lane_offset": 1, "adapter": copy.deepcopy(empty)},
        {
            "global_shot_id": 1,
            "lane_offset": 0,
            "adapter": {
                "status": "censored",
                "censor_reason": "local-incomplete-neutralization",
                "counters": _counters(),
                "completed_components": [],
                "censored_components": [{}],
                "last_complete_batch_id": None,
            },
        },
        {"global_shot_id": 1, "lane_offset": 1, "adapter": copy.deepcopy(empty)},
    )
    components = (
        {
            "global_shot_id": 0,
            "lane_offset": 0,
            "state_collection": "completed_components",
            "durable_decision": [True, "committed"],
            "adapter": {
                "component_index": 0,
                "cluster_defect_count": 2,
                "merge_count": 1,
                "gate_decision": "eligible",
                "gate_reason_set": [],
                "primary_gate_reason": "eligible",
                "exact_margin": {"numerator": 1, "denominator": 1},
                "boundary_reached": True,
                "port_kind_set": [],
            },
        },
        {
            "global_shot_id": 0,
            "lane_offset": 0,
            "state_collection": "completed_components",
            "durable_decision": [False, "port-tie"],
            "adapter": {
                "component_index": 1,
                "cluster_defect_count": 1,
                "merge_count": 0,
                "gate_decision": "deferred",
                "gate_reason_set": ["below-threshold", "port-tie", "port-yoke"],
                "primary_gate_reason": "port-tie",
                "exact_margin": {"numerator": 0, "denominator": 1},
                "boundary_reached": False,
                "port_kind_set": ["yoke"],
            },
        },
        {
            "global_shot_id": 1,
            "lane_offset": 0,
            "state_collection": "censored_components",
            "durable_decision": None,
            "adapter": {
                "component_index": 0,
                "partial_cluster_defect_lower_bound": 1,
                "merge_count": 0,
            },
        },
    )
    records = (
        ShotClusterRecord(0, True, {1: 1, 2: 1}, 2),
        ShotClusterRecord(1, False, {}, None),
    )
    controls = {
        name: {"shots": 2, "equal": 2, "mismatches": 0}
        for name in (
            "ordinary_treatment_vs_telemetry",
            "global_vs_adapter_control",
            "global_vs_uf_shadow",
        )
    }
    return VerifiedCollection(
        summary=summary,
        shot_rows=shots,
        lane_rows=lanes,
        component_rows=components,
        cluster_records=records,
        control_equality=controls,
        corpus_identity={"index_payload_sha256": "44" * 32},
        detector_corpus_bytes=b"",
        detector_corpus_sha256=(
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
    )


def _config() -> AnalysisConfig:
    return AnalysisConfig(
        workload_bootstrap_replicates=40,
        cluster_bootstrap_replicates=40,
        workload_bootstrap_seed=123,
        cluster_bootstrap_seed=456,
        casebook_seed_root="ab" * 32,
        maximum_cases_per_category=1,
    )


def test_analysis_is_canonical_deterministic_and_reconciled() -> None:
    first = analyze_verified_collection(_verified(), config=_config())
    second = analyze_verified_collection(_verified(), config=_config())

    assert first.analysis_bytes == second.analysis_bytes
    assert first.report_bytes == second.report_bytes
    assert json.loads(first.analysis_bytes) == first.analysis
    assert first.analysis["reconciliation"]["status"] == "reconciled"
    assert first.analysis["paired_accuracy"]["c"] == 1
    workload = first.analysis["workload_coverage"]["summary"]
    assert workload["original_total"] == 4
    assert workload["residual_total"] == 2
    assert workload["committed_total"] == 2
    assert workload["frontend_coverage"]["value"] == pytest.approx(1 / 2)
    routing = first.analysis["routing"]
    assert routing["completed_components"] == 2
    assert routing["committed_components"] == 1
    assert routing["durable_deferred_components"] == 1
    assert routing["censored_components"] == 1
    assert first.analysis["casebook"]["recovery"]["retained_shots"] == 1
    assert first.analysis["casebook"]["threshold-tie"]["retained_shots"] == 1
    assert first.analysis["casebook"]["boundary-using-commit"]["retained_shots"] == 1
    observable = first.analysis["observable_accuracy"]
    assert observable["num_observables"] == 2
    assert observable["per_observable"][0]["global_error"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
        "status": "estimated",
    }
    assert observable["per_observable"][0]["prediction_disagreement"][
        "numerator"
    ] == 1
    conditional = first.analysis["conditional_accuracy"]
    assert conditional["by_any_uf_activation"] == [
        {
            "activated": True,
            "a": 1,
            "b": 0,
            "c": 1,
            "d": 0,
            "denominator": 2,
            "global_errors": 1,
            "treatment_errors": 0,
            "regressions": 0,
            "recoveries": 1,
        }
    ]
    assert first.analysis["routing_rates"]["shot_durable_commit"][
        "numerator"
    ] == 1
    assert first.analysis["routing_rates"]["patch_abort_given_activation"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
        "status": "estimated",
    }
    lane_x = first.analysis["lane_breakdown"]["by_patch_basis"][0]
    assert (lane_x["patch_id"], lane_x["check_basis"]) == (0, "X")
    assert lane_x["original_detector_events"] == 4
    assert lane_x["residual_detector_events"] == 2
    roles = first.analysis["detector_role_workload"]
    assert roles["terminal"]["status"] == "recorded-v1"
    assert roles["terminal"]["original_detector_events"] == 1
    assert roles["body"]["removed_detector_events"] == 2
    confidence = first.analysis["confidence"]
    bin_one = next(
        row
        for row in confidence["component_acceptance_by_bin"]
        if row["bin_index"] == 5
    )
    assert bin_one["durable_commit"]["numerator"] == 1
    downstream = next(
        row
        for row in confidence[
            "downstream_regression_recovery_by_accepted_confidence_bin"
        ]
        if row["bin_index"] == 5
    )
    assert downstream["accepted_shot_outcomes"]["a"] == 1
    zero_threshold = first.analysis["risk_coverage"]["rows"][0]
    assert zero_threshold["comparison"] == "strict-greater-than"
    assert zero_threshold["frontend_defect_coverage"]["numerator"] == 2
    assert zero_threshold["frontend_defect_coverage"]["denominator"] == 4
    assert "raw numerators and denominators" in first.report_markdown


def test_analysis_bundle_is_write_once_and_tamper_evident(tmp_path) -> None:
    artifacts = analyze_verified_collection(_verified(), config=_config())
    out = tmp_path / "analysis"
    write_analysis_bundle(out, artifacts)
    write_analysis_bundle(out, artifacts)
    assert (out / "analysis.json").read_bytes() == artifacts.analysis_bytes
    assert (out / "report.md").read_bytes() == artifacts.report_bytes

    (out / "report.md").write_text("tampered")
    with pytest.raises(ValueError, match="differs"):
        write_analysis_bundle(out, artifacts)


def test_inconsistent_lane_counters_fail_closed() -> None:
    verified = _verified()
    bad_lanes = copy.deepcopy(list(verified.lane_rows))
    bad_lanes[0]["adapter"]["counters"]["heap_operation_count"] = 99
    bad = dataclasses.replace(verified, lane_rows=tuple(bad_lanes))
    with pytest.raises(ValueError, match="heap counter reconciliation"):
        analyze_verified_collection(bad, config=_config())


def test_zero_workload_denominators_are_null_not_estimable() -> None:
    verified = _verified()
    shots = (copy.deepcopy(verified.shot_rows[1]),)
    shots[0]["global_shot_id"] = 0
    shots[0]["lane_count"] = 2
    shots[0]["component_count"] = 0
    shots[0]["global_prediction_hex"] = "00"
    shots[0]["treatment_prediction_hex"] = "00"
    shots[0]["actual_observables_hex"] = "00"
    shots[0]["global_failed"] = False
    shots[0]["treatment_failed"] = False
    shots[0]["prediction_agreement"] = True
    shots[0]["adapter_metrics"].update(
        {
            "original_detector_count": 0,
            "residual_detector_count": 0,
            "lane_owned_detector_count": 0,
            "committed_defect_count": 0,
            "cluster_summary_complete": True,
            "lane_original_detector_counts": [0, 0],
            "lane_residual_detector_counts": [0, 0],
            "original_body_detector_count": 0,
            "residual_body_detector_count": 0,
            "original_terminal_detector_count": 0,
            "residual_terminal_detector_count": 0,
            "original_yoke_detector_count": 0,
            "residual_yoke_detector_count": 0,
        }
    )
    lanes = tuple(copy.deepcopy(verified.lane_rows[3]) for _ in range(2))
    for lane_offset, lane in enumerate(lanes):
        lane["global_shot_id"] = 0
        lane["lane_offset"] = lane_offset
    record = ShotClusterRecord(0, True, {}, 0)
    summary = {
        **verified.summary,
        "shots": 1,
        "lane_records": 2,
        "component_records": 0,
        "paired_contingency": {"a": 1, "b": 0, "c": 0, "d": 0},
        "prediction_agreements": 1,
        "hrlk_joint_histogram": [[0, 0, 0, 0, 1]],
    }
    controls = {
        name: {"shots": 1, "equal": 1, "mismatches": 0}
        for name in verified.control_equality
    }
    zero = VerifiedCollection(
        summary=summary,
        shot_rows=shots,
        lane_rows=lanes,
        component_rows=(),
        cluster_records=(record,),
        control_equality=controls,
        corpus_identity=verified.corpus_identity,
        detector_corpus_bytes=verified.detector_corpus_bytes,
        detector_corpus_sha256=verified.detector_corpus_sha256,
    )

    result = analyze_verified_collection(zero, config=_config()).analysis
    workload = result["workload_coverage"]
    assert workload["summary"]["workload_ratio"] == {
        "value": None,
        "status": "not-estimable",
    }
    assert workload["bootstrap"]["workload_ratio"]["interval"] is None
    assert workload["bootstrap"]["workload_ratio"]["status"] == "not-estimable"


def test_confidence_binning_is_exact_at_dyadic_edges_and_preserves_infinity() -> None:
    verified = _verified()
    components = copy.deepcopy(list(verified.component_rows))
    components[1]["adapter"]["exact_margin"] = {
        "mantissa": 1,
        "exponent": -4,
    }
    components[0]["adapter"]["exact_margin"] = "infinity"
    changed = dataclasses.replace(verified, component_rows=tuple(components))

    result = analyze_verified_collection(changed, config=_config()).analysis
    bins = result["confidence"]["component_acceptance_by_bin"]
    edge_bin = next(row for row in bins if row["bin_index"] == 1)
    infinity = next(row for row in bins if row["bin_index"] == "infinity")
    assert edge_bin["components"] == 1
    assert edge_bin["lower_exact"] == {"numerator": 1, "denominator": 16}
    assert infinity["components"] == 1
    assert infinity["durable_commit"]["numerator"] == 1
    shot_infinity = next(
        row
        for row in result["confidence"][
            "shot_acceptance_by_minimum_durable_margin_bin"
        ]
        if row["bin_index"] == "infinity"
    )
    assert shot_infinity["accepted_shot_outcomes"]["a"] == 1
    assert result["risk_coverage"]["rows"][-1]["accepted_shots"][
        "numerator"
    ] == 1


def test_censored_lower_bounds_are_not_treated_as_exact_lane_workload() -> None:
    verified = _verified()
    shots = copy.deepcopy(list(verified.shot_rows))
    lanes = copy.deepcopy(list(verified.lane_rows))
    components = copy.deepcopy(list(verified.component_rows))
    shots[1]["component_count"] = 2
    shots[1]["adapter_metrics"].update(
        {
            "original_detector_count": 3,
            "residual_detector_count": 3,
            "lane_owned_detector_count": 3,
            "lane_original_detector_counts": [3, 0],
            "lane_residual_detector_counts": [3, 0],
            "original_body_detector_count": 3,
            "residual_body_detector_count": 3,
        }
    )
    lanes[2]["adapter"]["completed_components"] = [{}]
    completed_before_censor = {
        "global_shot_id": 1,
        "lane_offset": 0,
        "state_collection": "completed_components",
        "durable_decision": [
            False,
            "local-incomplete-neutralization-patch-abort",
        ],
        "adapter": {
            "component_index": 1,
            "cluster_defect_count": 1,
            "merge_count": 0,
            "gate_decision": "eligible",
            "gate_reason_set": [],
            "primary_gate_reason": "eligible",
            "exact_margin": "infinity",
            "boundary_reached": False,
            "port_kind_set": [],
        },
    }
    components.append(completed_before_censor)
    summary = {
        **verified.summary,
        "component_records": 4,
        "hrlk_joint_histogram": [[3, 1, 3, 2, 1], [3, 3, 3, 0, 1]],
    }
    records = (
        verified.cluster_records[0],
        ShotClusterRecord(1, False, {1: 1}, None),
    )
    changed = VerifiedCollection(
        summary=summary,
        shot_rows=tuple(shots),
        lane_rows=tuple(lanes),
        component_rows=tuple(components),
        cluster_records=records,
        control_equality=verified.control_equality,
        corpus_identity=verified.corpus_identity,
        detector_corpus_bytes=verified.detector_corpus_bytes,
        detector_corpus_sha256=verified.detector_corpus_sha256,
    )

    result = analyze_verified_collection(changed, config=_config()).analysis
    lane_x = result["lane_breakdown"]["by_patch_basis"][0]
    assert lane_x["original_detector_events"] == 6
    assert lane_x["residual_detector_events"] == 4
    assert result["routing"]["completed_components"] == 3
    assert result["routing"]["censored_components"] == 1


def test_lane_and_role_workload_identity_mismatch_fails_closed() -> None:
    verified = _verified()
    shots = copy.deepcopy(list(verified.shot_rows))
    shots[0]["adapter_metrics"]["lane_original_detector_counts"] = [2, 0]
    changed = dataclasses.replace(verified, shot_rows=tuple(shots))
    with pytest.raises(ValueError, match="lane/role workload telemetry"):
        analyze_verified_collection(changed, config=_config())
