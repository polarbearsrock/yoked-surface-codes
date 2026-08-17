from __future__ import annotations

import json
from pathlib import Path

import pytest

import yoked.decoding._promatch_analysis as promatch_analysis
from yoked.decoding._promatch_analysis import (
    ANALYSIS_SCHEMA,
    analyze_cell,
    analyze_summary,
    construct_confirmatory_draft_from_pilot,
    render_markdown,
    select_pilot_cell,
)
from yoked.decoding._promatch_experiment import PROTOCOL_SCHEMA, SUMMARY_SCHEMA


def test_confirmatory_derivation_copies_the_verified_pilot_seed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[3]
    draft = json.loads(
        (root / "docs" / "PROMATCH_FIRST_ROUND_PROTOCOL.json").read_text()
    )
    draft["sampler_seed_roots"]["pilot"] = "ff" * 32
    pilot_seed_root = "aa" * 32
    pilot_manifest = {
        "schema": PROTOCOL_SCHEMA,
        "phase": "pilot",
        "experiment_id": "1" * 64,
        "sampler_seed_roots": {"pilot": pilot_seed_root},
        "source_hashes": {"source": "2" * 64},
        "cells": [
            {
                "cell_id": "selected",
                "d": 5,
                "patches": 6,
                "yokes": 2,
                "r": 20,
                "p": 0.003,
                "circuit_sha256": "3" * 64,
                "dem_sha256": "4" * 64,
                "graph_fingerprint": "5" * 64,
            }
        ],
    }
    selection_log = {
        "selection_sha256": "6" * 64,
        "selected": {
            "cell_id": "selected",
            "confirmatory_shots": 20_000,
            "activation_fraction": 0.5,
            "u0_failures": 100,
            "discordant_pairs": 100,
            "integrity_checks_passed": True,
            "p_u0_design": 0.1,
            "delta_noninferiority": 0.01,
            "discordance_upper": 0.2,
            "normal_rule_raw_shots": 15_000,
        },
    }
    monkeypatch.setattr(
        promatch_analysis, "load_verified_summary", lambda **_: {"verified": True}
    )
    monkeypatch.setattr(
        promatch_analysis,
        "analyze_summary",
        lambda summary, *, manifest: {
            "analysis_sha256": "7" * 64,
            "blinded_selection": selection_log,
        },
    )
    monkeypatch.setattr(
        promatch_analysis, "_pilot_result_digest", lambda _: "8" * 64
    )
    pilot_protocol_path = tmp_path / "pilot.json"
    pilot_protocol_path.write_text("{}")
    result = construct_confirmatory_draft_from_pilot(
        draft,
        pilot_manifest=pilot_manifest,
        pilot_protocol_path=pilot_protocol_path,
        pilot_input_directory=tmp_path,
    )
    assert result["sampler_seed_roots"]["pilot"] == pilot_seed_root


def _telemetry(*, shots: int, activated: int, rollback: int = 0):
    return {
        "shots": shots,
        "original_event_sum": shots * 2,
        "residual_event_sum": shots,
        "original_residual_hw_joint_histogram": {"2,1": shots},
        "activated_shots": activated,
        "rollback_shots": rollback,
    }


def _summary_cell(
    cell_id: str,
    *,
    both_correct: int,
    regressions: int,
    recoveries: int,
    both_wrong: int,
    activated: int,
):
    shots = both_correct + regressions + recoveries + both_wrong
    return {
        "cell_id": cell_id,
        "shots": shots,
        "paired_contingency": {
            "both_correct": both_correct,
            "regressions": regressions,
            "recoveries": recoveries,
            "both_wrong": both_wrong,
        },
        "telemetry": _telemetry(shots=shots, activated=activated),
    }


def _confirm_manifest():
    return {
        "schema": PROTOCOL_SCHEMA,
        "phase": "confirm",
        "experiment_id": "a" * 64,
        "sampler_seed_roots": {"timing_bootstrap": "11" * 32},
        "analysis_config": {
            "selection": {"delta_noninferiority": 0.02},
            "accuracy_protocol": {"alpha_one_sided": 0.025},
            "workload_protocol": {
                "paired_bootstrap_replicates": 200,
                "bootstrap_alpha_one_sided": 0.025,
                "workload_ratio_upper_threshold": 0.9,
                "quantile_method": "empirical_type_7",
                "bootstrap_unit": "paired_shot_via_exact_joint_histogram_multinomial",
            },
        },
    }


def test_confirmatory_analysis_reports_ordered_accuracy_and_workload_gates() -> None:
    manifest = _confirm_manifest()
    cell = _summary_cell(
        "selected",
        both_correct=800,
        regressions=20,
        recoveries=100,
        both_wrong=80,
        activated=500,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": manifest["experiment_id"],
        "phase": "confirm",
        "cells": [cell],
    }
    analysis = analyze_summary(summary, manifest=manifest)
    assert analysis["schema"] == ANALYSIS_SCHEMA
    result = analysis["cells"][0]
    assert result["delta_pu_minus_u0"] == pytest.approx(-0.08)
    assert result["tango_upper_one_sided"] < 0
    assert result["noninferiority_passed"] is True
    assert result["ordered_superiority_passed"] is True
    assert result["workload_ratio"] == 0.5
    assert result["workload_ratio_upper_one_sided"] == 0.5
    assert result["workload_improvement_passed"] is True
    assert result["exact_mcnemar_superiority_p"] < 0.001
    assert len(analysis["analysis_sha256"]) == 64


def test_zero_discordance_analysis_uses_finite_profile_boundary() -> None:
    manifest = _confirm_manifest()
    cell = _summary_cell(
        "selected",
        both_correct=100,
        regressions=0,
        recoveries=0,
        both_wrong=0,
        activated=0,
    )
    result = analyze_cell(
        cell,
        manifest=manifest,
        delta_noninferiority=0.05,
    )
    z = 1.959963984540054
    assert result["tango_upper_one_sided"] == pytest.approx(
        z * z / (100 + z * z)
    )
    assert result["tango_upper_one_sided"] < 1


def _pilot_manifest(cell_ids):
    return {
        "schema": PROTOCOL_SCHEMA,
        "phase": "pilot",
        "experiment_id": "b" * 64,
        "cells": [{"cell_id": cell_id} for cell_id in cell_ids],
        "sampler_seed_roots": {"timing_bootstrap": "22" * 32},
        "analysis_config": {
            "selection_gates": {
                "minimum_activation_fraction": 0.1,
                "minimum_u0_direct_failures": 50,
                "minimum_discordant_pairs": 50,
                "require_integrity_checks": True,
                "maximum_confirmatory_paired_shots": 100_000,
            },
            "statistical_design": {
                "noninferiority_alpha_one_sided": 0.025,
                "power_verification": {
                    "replicates": 200,
                    "seed": 12345,
                    "power_bound_alpha": 0.05,
                    "accept_if_lower_bound_at_least": 0.5,
                },
            },
            "workload_protocol": {
                "paired_bootstrap_replicates": 100,
                "bootstrap_alpha_one_sided": 0.025,
                "workload_ratio_upper_threshold": 0.9,
                "quantile_method": "empirical_type_7",
                "bootstrap_unit": "paired_shot_via_exact_joint_histogram_multinomial",
            },
        },
    }


def test_pilot_selection_is_fixed_order_and_does_not_emit_signed_difference() -> None:
    ids = ["first", "second"]
    manifest = _pilot_manifest(ids)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": manifest["experiment_id"],
        "phase": "pilot",
        "cells": [
            _summary_cell(
                "first",
                both_correct=500,
                regressions=100,
                recoveries=200,
                both_wrong=200,
                activated=400,
            ),
            _summary_cell(
                "second",
                both_correct=450,
                regressions=150,
                recoveries=200,
                both_wrong=200,
                activated=500,
            ),
        ],
    }
    selection = select_pilot_cell(summary, manifest=manifest)
    assert selection["selected"]["cell_id"] == "first"
    assert selection["selection_used_signed_difference"] is False
    serialized = json.dumps(selection)
    assert "regressions" not in serialized
    assert "recoveries" not in serialized
    assert "signed_accuracy" not in serialized

    analysis = analyze_summary(summary, manifest=manifest)
    assert analysis["blinded_selection"] == selection
    markdown = render_markdown(analysis)
    assert "did not compute or emit the signed paired difference" in markdown


def test_analysis_rejects_missing_or_nonreconciling_paired_workload() -> None:
    manifest = _confirm_manifest()
    cell = _summary_cell(
        "selected",
        both_correct=90,
        regressions=2,
        recoveries=3,
        both_wrong=5,
        activated=10,
    )
    del cell["telemetry"]["original_residual_hw_joint_histogram"]
    with pytest.raises(ValueError, match="paired workload inference is impossible"):
        analyze_cell(cell, manifest=manifest, delta_noninferiority=0.02)

    cell = _summary_cell(
        "selected",
        both_correct=90,
        regressions=2,
        recoveries=3,
        both_wrong=5,
        activated=10,
    )
    cell["telemetry"]["original_residual_hw_joint_histogram"] = {"2,1": 99}
    with pytest.raises(ValueError, match="does not reconcile"):
        analyze_cell(cell, manifest=manifest, delta_noninferiority=0.02)
