import copy
import json

import pytest

from yoked.decoding._pinball_promatch_analysis import (
    aggregate_cell_ledgers,
    analyze_cell,
    build_json_report,
    cell_csv_row,
    report_csv_rows,
    report_json_bytes,
    validate_cell_aggregate,
)


def _cell(*, cell_id: str = "d5-p6-r20", scale: int = 1) -> dict:
    # Failure bits are ordered (u0, promatch, pinball).
    cube = {
        "000": 80 * scale,
        "001": 5 * scale,
        "010": 4 * scale,
        "011": 1 * scale,
        "100": 3 * scale,
        "101": 2 * scale,
        "110": 1 * scale,
        "111": 4 * scale,
    }
    return {
        "cell_id": cell_id,
        "shots": 100 * scale,
        "correctness_cube": cube,
        "pairwise_contingencies": {
            "pinball_minus_promatch": {
                "both_correct": 83 * scale,
                "regressions": 7 * scale,
                "recoveries": 5 * scale,
                "both_wrong": 5 * scale,
            },
            "pinball_minus_u0": {
                "both_correct": 84 * scale,
                "regressions": 6 * scale,
                "recoveries": 4 * scale,
                "both_wrong": 6 * scale,
            },
            "promatch_minus_u0": {
                "both_correct": 85 * scale,
                "regressions": 5 * scale,
                "recoveries": 5 * scale,
                "both_wrong": 5 * scale,
            },
        },
        "prediction_agreement": {
            "pinball_minus_promatch": {
                "agree": 88 * scale,
                "disagree": 12 * scale,
            },
            "pinball_minus_u0": {
                "agree": 87 * scale,
                "disagree": 13 * scale,
            },
            "promatch_minus_u0": {
                "agree": 90 * scale,
                "disagree": 10 * scale,
            },
        },
        "telemetry": {
            "common": {
                "shots": 100 * scale,
                "original_event_sum": 1_000 * scale,
                "original_hw_histogram": {"10": 100 * scale},
            },
            "promatch": {
                "shots": 100 * scale,
                "residual_event_sum": 700 * scale,
                "activated_shots": 60 * scale,
                "rollback_shots": 10 * scale,
                "attempted_stage_counts": [1 * scale, 2 * scale, 3 * scale, 4 * scale],
                "domain_status_counts": {
                    "below-limit": 40 * scale,
                    "success": 60 * scale,
                },
            },
            "pinball": {
                "shots": 100 * scale,
                "residual_event_sum": 500 * scale,
                "commit_shots": 80 * scale,
                "complex_domains": 30 * scale,
                "simple_domains": 170 * scale,
                "stage_match_counts": [scale] * 9,
            },
        },
    }


def test_analysis_reconciles_cube_marginals_pairs_and_workload() -> None:
    result = analyze_cell(_cell())
    assert result["marginals"]["u0"]["failures"] == 10
    assert result["marginals"]["promatch"]["failures"] == 10
    assert result["marginals"]["pinball"]["failures"] == 12
    assert result["marginals"]["pinball"]["any_observable_logical_error_rate"] == 0.12
    assert (
        result["pairwise"]["pinball_minus_promatch"]["paired_risk_difference"]
        == 0.02
    )
    interval = result["pairwise"]["pinball_minus_promatch"]["confidence_interval"]
    assert interval["lower"] < 0.02 < interval["upper"]
    assert interval["confidence_level"] == 0.95
    assert (
        result["pairwise"]["pinball_minus_promatch"]["prediction_agreement"][
            "fraction"
        ]
        == 0.88
    )
    ratios = result["telemetry_summary"]["workload_ratios"]
    assert ratios == {
        "promatch_residual_over_original": 0.7,
        "pinball_residual_over_original": 0.5,
        "pinball_residual_over_promatch": 5 / 7,
    }
    assert (
        result["telemetry_summary"]["per_shot_scalar_telemetry"]["pinball"][
            "complex_domains"
        ]
        == 0.3
    )


def test_pair_orientation_and_two_sided_swap_symmetry() -> None:
    first = analyze_cell(_cell())["pairwise"]["pinball_minus_promatch"]
    changed = _cell()
    pair = changed["pairwise_contingencies"]["pinball_minus_promatch"]
    pair["regressions"], pair["recoveries"] = pair["recoveries"], pair["regressions"]
    # Swap only P/PB failure bits while keeping U0 fixed.
    changed["correctness_cube"] = {
        bits[0] + bits[2] + bits[1]: count
        for bits, count in changed["correctness_cube"].items()
    }
    pb_u0 = changed["pairwise_contingencies"]["pinball_minus_u0"]
    pm_u0 = changed["pairwise_contingencies"]["promatch_minus_u0"]
    changed["pairwise_contingencies"]["pinball_minus_u0"] = pm_u0
    changed["pairwise_contingencies"]["promatch_minus_u0"] = pb_u0
    second = analyze_cell(changed)["pairwise"]["pinball_minus_promatch"]
    assert second["paired_risk_difference"] == -first["paired_risk_difference"]
    assert second["confidence_interval"]["lower"] == pytest.approx(
        -first["confidence_interval"]["upper"]
    )
    assert second["confidence_interval"]["upper"] == pytest.approx(
        -first["confidence_interval"]["lower"]
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda c: c.update(extra=1), "incorrect fields"),
        (lambda c: c["correctness_cube"].pop("111"), "exactly"),
        (lambda c: c["correctness_cube"].__setitem__("000", 79), "reconcile"),
        (
            lambda c: c["pairwise_contingencies"][
                "pinball_minus_promatch"
            ].__setitem__("regressions", 6),
            "reconcile|disagrees",
        ),
        (
            lambda c: c["prediction_agreement"][
                "pinball_minus_u0"
            ].__setitem__("agree", 86),
            "reconcile",
        ),
        (lambda c: c["telemetry"].update(extra={}), "exactly"),
        (
            lambda c: c["telemetry"]["pinball"].__setitem__("shots", True),
            "boolean|reconcile",
        ),
        (lambda c: c["telemetry"]["common"].pop("original_event_sum"), "missing"),
    ],
)
def test_validation_fails_closed(mutation, message: str) -> None:
    cell = _cell()
    mutation(cell)
    with pytest.raises(ValueError, match=message):
        validate_cell_aggregate(cell)


def test_aggregate_ledgers_sums_all_additive_shapes_and_checks_completion() -> None:
    aggregate = aggregate_cell_ledgers(
        [_cell(scale=1), _cell(scale=2)],
        expected_cell_id="d5-p6-r20",
        expected_shots=300,
    )
    assert aggregate["shots"] == 300
    assert aggregate["correctness_cube"]["000"] == 240
    assert aggregate["telemetry"]["pinball"]["stage_match_counts"] == [3] * 9
    assert aggregate["telemetry"]["promatch"]["domain_status_counts"] == {
        "below-limit": 120,
        "success": 180,
    }
    with pytest.raises(ValueError, match="expected"):
        aggregate_cell_ledgers([_cell()], expected_shots=1_000_000)


def test_aggregate_rejects_telemetry_shape_drift() -> None:
    second = _cell(scale=2)
    second["telemetry"]["pinball"]["stage_match_counts"].pop()
    with pytest.raises(ValueError, match="different lengths"):
        aggregate_cell_ledgers([_cell(), second])
    second = _cell(scale=2)
    second["telemetry"]["pinball"]["new_count"] = 0
    with pytest.raises(ValueError, match="fields change"):
        aggregate_cell_ledgers([_cell(), second])


def test_json_report_is_canonical_hashed_and_complete() -> None:
    other = _cell(cell_id="d7-p6-r28")
    report = build_json_report(
        [other, _cell()],
        campaign_id="campaign-1",
        expected_cell_ids=["d5-p6-r20", "d7-p6-r28"],
    )
    assert [cell["cell_id"] for cell in report["cells"]] == [
        "d5-p6-r20",
        "d7-p6-r28",
    ]
    encoded = report_json_bytes(report)
    assert json.loads(encoded) == report
    tampered = copy.deepcopy(report)
    tampered["cells"][0]["shots"] += 1
    with pytest.raises(ValueError, match="does not reconcile"):
        report_json_bytes(tampered)
    with pytest.raises(ValueError, match="expected_cell_ids"):
        build_json_report(
            [_cell()], campaign_id="campaign-1", expected_cell_ids=["missing"]
        )


def test_csv_helpers_flatten_primary_endpoints_and_telemetry() -> None:
    analyzed = analyze_cell(_cell())
    row = cell_csv_row(analyzed)
    assert row["pinball_ler"] == 0.12
    assert row["pinball_minus_promatch_delta"] == 0.02
    assert row["workload_pinball_residual_over_original"] == 0.5
    assert json.loads(row["pinball_telemetry_json"])["commit_shots"] == 80
    report = build_json_report([_cell()], campaign_id="campaign-1")
    assert report_csv_rows(report) == [row]


def test_zero_original_and_promatch_workload_have_explicit_undefined_ratios() -> None:
    cell = _cell()
    cell["telemetry"]["common"]["original_event_sum"] = 0
    cell["telemetry"]["promatch"]["residual_event_sum"] = 0
    summary = analyze_cell(cell)["telemetry_summary"]
    assert summary["workload_ratios"] == {
        "promatch_residual_over_original": None,
        "pinball_residual_over_original": None,
        "pinball_residual_over_promatch": None,
    }
