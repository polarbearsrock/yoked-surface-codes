from __future__ import annotations

import copy

import pytest

import yoked.decoding._pinball_promatch_matched_analysis as matched_analysis
from yoked.decoding._pinball_promatch_matched_accuracy import (
    ARM_ORDER,
    PAIR_DEFINITIONS,
)
from yoked.decoding._pinball_promatch_matched_analysis import (
    ARM_STANDARD_NAMES,
    MatchedAnalysisConfig,
    analyze_matched_aggregate,
    render_matched_analysis_markdown,
)


def _aggregate() -> dict[str, object]:
    shots = 20
    failed = {
        "global": {0, 1, 2, 3},
        "promatch": {1, 2, 4},
        "pinball": {2, 3, 5, 6},
        "union_find": {2, 7},
    }
    cube = {f"{value:04b}": 0 for value in range(16)}
    for shot in range(shots):
        bits = "".join("1" if shot in failed[arm] else "0" for arm in ARM_ORDER)
        cube[bits] += 1

    tables: dict[str, dict[str, int]] = {}
    agreements: dict[str, dict[str, int]] = {}
    for pair, (baseline, treatment) in PAIR_DEFINITIONS.items():
        baseline_failed = failed[baseline]
        treatment_failed = failed[treatment]
        table = {
            "both_correct": sum(
                shot not in baseline_failed and shot not in treatment_failed
                for shot in range(shots)
            ),
            "regressions": sum(
                shot not in baseline_failed and shot in treatment_failed
                for shot in range(shots)
            ),
            "recoveries": sum(
                shot in baseline_failed and shot not in treatment_failed
                for shot in range(shots)
            ),
            "both_wrong": sum(
                shot in baseline_failed and shot in treatment_failed
                for shot in range(shots)
            ),
        }
        tables[pair] = table
        agreements[pair] = {
            "agree": table["both_correct"] + table["both_wrong"],
            "disagree": table["regressions"] + table["recoveries"],
        }

    telemetry = {
        "common": {
            "shots": shots,
            "original_event_sum": 30,
            "original_hw_histogram": {"1": 10, "2": 10},
            "terminal_event_sum": 3,
            "yoke_event_sum": 2,
            "native_common_counter": 17,
        },
        "global": {
            "shots": shots,
            "residual_event_sum": 30,
            "residual_hw_histogram": {"1": 10, "2": 10},
            "original_residual_hw_joint_histogram": {"1,1": 10, "2,2": 10},
        },
        "promatch": {
            "shots": shots,
            "residual_event_sum": 20,
            "residual_hw_histogram": {"0": 5, "1": 10, "2": 5},
            "original_residual_hw_joint_histogram": {
                "1,0": 5,
                "1,1": 5,
                "2,1": 5,
                "2,2": 5,
            },
            "native_promatch_counter": 11,
        },
        "pinball": {
            "shots": shots,
            "residual_event_sum": 15,
            "residual_hw_histogram": {"0": 10, "1": 5, "2": 5},
            "original_residual_hw_joint_histogram": {
                "1,0": 10,
                "2,1": 5,
                "2,2": 5,
            },
            "native_pinball_counter": 13,
        },
        "union_find": {
            "shots": shots,
            "residual_hw_available_shots": shots,
            "residual_event_sum": 12,
            "residual_hw_histogram": {"0": 14, "2": 6},
            "original_residual_hw_joint_histogram": {
                "1,0": 10,
                "2,0": 4,
                "2,2": 6,
            },
            "native_union_find_counter": 19,
        },
    }
    return {
        "schema": "yoked.pinball-promatch-uf-matched-aggregate-v1",
        "source_identity": {"shots": shots, "cell_id": "matched-d7-p003"},
        "prepared_provenance": {"decoder_code_sha256": "11" * 32},
        "complete": True,
        "ranges": 32,
        "shots": shots,
        "ordered_range_payload_sha256": ["22" * 32] * 32,
        "arm_order": list(ARM_ORDER),
        "correctness_cube": cube,
        "pairwise_contingencies": tables,
        "prediction_agreement": agreements,
        "telemetry": telemetry,
        "payload_sha256": "33" * 32,
    }


@pytest.fixture
def bypass_upstream_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        matched_analysis,
        "validate_matched_aggregate",
        lambda value, **kwargs: None,
    )


def _config(*, seed: int = 123) -> MatchedAnalysisConfig:
    return MatchedAnalysisConfig(
        workload_bootstrap_replicates=1000,
        workload_bootstrap_seed=seed,
        workload_bootstrap_chunk_size=64,
    )


def test_accuracy_workload_raw_telemetry_and_markdown_are_complete(
    bypass_upstream_validation: None,
) -> None:
    aggregate = _aggregate()
    before = copy.deepcopy(aggregate)
    result = analyze_matched_aggregate(
        aggregate,
        corpus=object(),  # upstream validator is isolated by the fixture
        config=_config(),
    )

    assert aggregate == before
    assert set(result["accuracy"]["marginals"]) == set(ARM_ORDER)
    assert set(result["accuracy"]["pairwise"]) == set(PAIR_DEFINITIONS)
    assert len(result["accuracy"]["pairwise"]) == 6
    assert result["accuracy"]["marginals"]["global"]["failures"] == 4
    assert result["accuracy"]["marginals"]["union_find"]["failures"] == 2
    interval = result["accuracy"]["marginals"]["global"]["confidence_interval"]
    assert interval["confidence_level"] == 0.95
    assert interval["lower"] < 0.2 < interval["upper"]

    comparison = result["accuracy"]["pairwise"]["promatch_minus_global"]
    assert comparison["baseline"] == "global"
    assert comparison["treatment"] == "promatch"
    assert comparison["failure_risk_difference"]["fraction"] == {
        "numerator": -1,
        "denominator": 20,
        "text": "-1/20",
    }
    assert comparison["failure_risk_difference"]["percentage_points"] == -5
    assert comparison["tango_confidence_interval"]["confidence_level"] == 0.95
    assert 0 <= comparison["exact_two_sided_mcnemar_p_value"] <= 1
    assert comparison["prediction_agreement"] == {
        "agree": 17,
        "disagree": 3,
        "fraction": 0.85,
        "percentage": 85,
    }

    workload = result["workload"]
    assert workload["common_original_detector_events"] == 30
    assert workload["arms"]["global"]["residual_over_original_ratio"] == 1
    assert workload["arms"]["promatch"]["residual_over_original_ratio"] == pytest.approx(2 / 3)
    assert workload["arms"]["pinball"]["detector_event_reduction_percentage"] == 50
    assert workload["arms"]["union_find"]["residual_over_original_ratio"] == 0.4
    for arm in ARM_ORDER:
        bootstrap = workload["arms"][arm][
            "residual_over_original_ratio_confidence_interval"
        ]
        assert bootstrap["confidence_level"] == 0.95
        assert bootstrap["replicates"] == 1000
        assert bootstrap["lower"] <= bootstrap["estimate"] <= bootstrap["upper"]
    direct = workload["pairwise_ratio_differences"]["pinball_minus_promatch"]
    assert direct["residual_over_original_ratio_difference"]["fraction"] == {
        "numerator": -5,
        "denominator": 30,
        "text": "-5/30",
    }
    assert direct["paired_confidence_interval"] is None
    assert workload["cross_arm_paired_inference"]["available"] is False

    assert result["raw_inputs"]["telemetry"]["promatch"][
        "native_promatch_counter"
    ] == 11
    markdown = render_matched_analysis_markdown(result)
    for standard_name in ARM_STANDARD_NAMES.values():
        assert standard_name in markdown
    assert "percentage points" in markdown
    assert "not a latency measurement" in markdown
    assert " pp " not in markdown


def test_bootstrap_and_payload_are_deterministic(
    bypass_upstream_validation: None,
) -> None:
    first = analyze_matched_aggregate(
        _aggregate(), corpus=object(), config=_config(seed=7)
    )
    second = analyze_matched_aggregate(
        _aggregate(), corpus=object(), config=_config(seed=7)
    )
    assert first == second
    assert first["payload_sha256"] == second["payload_sha256"]

    changed = copy.deepcopy(first)
    changed["accuracy"]["marginals"]["global"]["failures"] += 1
    with pytest.raises(ValueError, match="payload digest mismatch"):
        render_matched_analysis_markdown(changed)


def test_public_entrypoint_invokes_upstream_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def validate(value, **kwargs):
        seen["value"] = value
        seen.update(kwargs)

    monkeypatch.setattr(matched_analysis, "validate_matched_aggregate", validate)
    aggregate = _aggregate()
    corpus = object()
    provenance = {"decoder_code_sha256": "11" * 32}
    analyze_matched_aggregate(
        aggregate,
        corpus=corpus,
        config=_config(),
        expected_prepared_provenance=provenance,
        require_complete=False,
    )
    assert seen == {
        "value": aggregate,
        "corpus": corpus,
        "expected_prepared_provenance": provenance,
        "require_complete": False,
    }


def test_missing_or_nonreconciling_joint_workload_fails_clearly(
    bypass_upstream_validation: None,
) -> None:
    missing = _aggregate()
    del missing["telemetry"]["union_find"][
        "original_residual_hw_joint_histogram"
    ]
    with pytest.raises(
        ValueError,
        match=r"telemetry\.union_find\.original_residual_hw_joint_histogram is required",
    ):
        analyze_matched_aggregate(missing, corpus=object(), config=_config())

    mismatch = _aggregate()
    mismatch["telemetry"]["pinball"]["residual_event_sum"] = 16
    with pytest.raises(ValueError, match="residual total differs"):
        analyze_matched_aggregate(mismatch, corpus=object(), config=_config())

    different_original = _aggregate()
    different_original["telemetry"]["promatch"][
        "original_residual_hw_joint_histogram"
    ] = {"1,0": 5, "1,1": 5, "3,1": 5, "3,2": 5}
    with pytest.raises(ValueError, match="original marginal differs"):
        analyze_matched_aggregate(
            different_original, corpus=object(), config=_config()
        )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (
            {"workload_bootstrap_replicates": 0, "workload_bootstrap_seed": 1},
            "replicates must be positive",
        ),
        (
            {"workload_bootstrap_replicates": 1, "workload_bootstrap_seed": -1},
            "seed must be nonnegative",
        ),
        (
            {"workload_bootstrap_replicates": True, "workload_bootstrap_seed": 1},
            "replicates must be an integer",
        ),
    ],
)
def test_analysis_config_is_strict(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        MatchedAnalysisConfig(**kwargs)
