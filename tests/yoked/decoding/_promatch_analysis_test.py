from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import REPO_ROOT
import yoked.decoding._promatch_analysis as promatch_analysis
from yoked.decoding._promatch_analysis import (
    ANALYSIS_SCHEMA,
    analyze_cell,
    analyze_summary,
    construct_confirmatory_draft_from_pilot,
    load_verified_summary,
    render_markdown,
    select_pilot_cell,
    validate_generated_analysis_artifact,
)
from yoked.decoding._promatch_experiment import (
    PROTOCOL_SCHEMA,
    SUMMARY_SCHEMA,
    default_smoke_protocol,
    summarize_ledgers,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


def test_confirmatory_derivation_copies_the_verified_pilot_seed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = REPO_ROOT
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
    monkeypatch.setattr(promatch_analysis, "_pilot_result_digest", lambda _: "8" * 64)
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

    extra = copy.deepcopy(analysis)
    extra["implementation_defined"] = None
    with pytest.raises(ValueError, match="incorrect top-level fields"):
        validate_generated_analysis_artifact(extra)

    missing_cell_field = copy.deepcopy(analysis)
    del missing_cell_field["cells"][0]["workload_ratio"]
    with pytest.raises(ValueError, match="cells have incorrect fields"):
        validate_generated_analysis_artifact(missing_cell_field)


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
    assert result["tango_upper_one_sided"] == pytest.approx(z * z / (100 + z * z))
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
    with pytest.raises(ValueError, match="scientifically verified summary"):
        select_pilot_cell(summary, manifest=manifest)
    verified = promatch_analysis._VerifiedSummary(
        summary,
        provenance=promatch_analysis._VerificationProvenance(
            experiment_id=manifest["experiment_id"],
            phase="pilot",
            deterministic_regeneration_passed=True,
            summary_sha256=promatch_analysis.hashlib.sha256(
                canonical_json_bytes(summary)
            ).hexdigest(),
            manifest_sha256=promatch_analysis.hashlib.sha256(
                canonical_json_bytes(manifest)
            ).hexdigest(),
        ),
    )
    selection = select_pilot_cell(verified, manifest=manifest)
    assert selection["selected"]["cell_id"] == "first"
    assert selection["selection_used_signed_difference"] is False
    serialized = json.dumps(selection)
    assert "regressions" not in serialized
    assert "recoveries" not in serialized
    assert "signed_accuracy" not in serialized

    mutated = promatch_analysis._VerifiedSummary(
        copy.deepcopy(dict(verified)),
        provenance=verified.provenance,
    )
    mutated["cells"][0]["shots"] += 1
    with pytest.raises(ValueError, match="scientifically verified summary"):
        select_pilot_cell(mutated, manifest=manifest)

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["analysis_config"]["selection_gates"][
        "minimum_u0_direct_failures"
    ] += 1
    with pytest.raises(ValueError, match="scientifically verified summary"):
        select_pilot_cell(verified, manifest=changed_manifest)

    analysis = analyze_summary(verified, manifest=manifest)
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


def _raw_collection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path, list[tuple[str, int]]]:
    manifest = default_smoke_protocol(processes=1, shots=1)
    cell = manifest["cells"][0]
    batch = manifest["cell_batch_schedules"][cell["cell_id"]][0]
    row = {
        "schema": "promatch-l1-paired-batch-v1",
        "experiment_id": manifest["experiment_id"],
        "phase": "smoke",
        "cell_id": cell["cell_id"],
        "batch": batch,
        "stim_seed": 1,
        "detectors": {},
        "observables": {},
        "provenance": {},
        "paired_contingency": {
            "both_correct": 1,
            "regressions": 0,
            "recoveries": 0,
            "both_wrong": 0,
        },
        "telemetry": {"shots": 1},
        "replay_samples": [],
    }
    summary = summarize_ledgers(
        [row],
        experiment_id=manifest["experiment_id"],
        phase="smoke",
        replay_policy=manifest["replay_policy"],
    )
    collection = tmp_path / "collection"
    ledger = collection / "batches" / cell["cell_id"] / "batch-00000000.json"
    ledger.parent.mkdir(parents=True)
    (collection / "experiment.json").write_text(
        json.dumps(
            {
                "schema": PROTOCOL_SCHEMA,
                "experiment_id": manifest["experiment_id"],
                "phase": "smoke",
            }
        )
    )
    (collection / "protocol.json").write_text(json.dumps(manifest))
    ledger.write_text(json.dumps(row))
    (collection / "summary.json").write_text(json.dumps(summary))

    regeneration_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        promatch_analysis,
        "validate_experiment_protocol",
        lambda *_, **__: manifest["experiment_id"],
    )
    monkeypatch.setattr(
        promatch_analysis,
        "prepare_cell",
        lambda *_, **__: SimpleNamespace(provenance={}),
    )
    monkeypatch.setattr(
        promatch_analysis, "_validate_ledger_row", lambda *_, **__: None
    )

    def record_regeneration(*_, phase: str, processes: int, **__) -> None:
        regeneration_calls.append((phase, processes))

    monkeypatch.setattr(
        promatch_analysis,
        "_verify_scientific_regeneration_without_writes",
        record_regeneration,
    )
    return manifest, collection, regeneration_calls


def _artifact_snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_verified_summary_reconciles_before_read_only_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, collection, regeneration_calls = _raw_collection_fixture(
        tmp_path, monkeypatch
    )
    before = _artifact_snapshot(collection)
    summary = load_verified_summary(
        manifest=manifest,
        input_directory=collection,
        scientific=True,
    )
    assert isinstance(summary, promatch_analysis._VerifiedSummary)
    assert summary.provenance.deterministic_regeneration_passed is True
    assert regeneration_calls == [("smoke", 1)]
    assert _artifact_snapshot(collection) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-summary",
        "missing-ledger",
        "tampered-summary",
        "corrupt-ledger",
        "duplicate-summary-key",
        "unexpected-artifact",
    ],
)
def test_verified_summary_preflight_failures_do_not_mutate_or_regenerate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifest, collection, regeneration_calls = _raw_collection_fixture(
        tmp_path, monkeypatch
    )
    ledger = next((collection / "batches").glob("*/*.json"))
    summary_path = collection / "summary.json"
    if mutation == "missing-summary":
        summary_path.unlink()
    elif mutation == "missing-ledger":
        ledger.unlink()
    elif mutation == "tampered-summary":
        summary = json.loads(summary_path.read_text())
        summary["cells"][0]["shots"] = 2
        summary_path.write_text(json.dumps(summary))
    elif mutation == "corrupt-ledger":
        ledger.write_text("{")
    elif mutation == "duplicate-summary-key":
        summary_path.write_text(
            '{"schema":"promatch-l1-paired-summary-v1","schema":"duplicate"}'
        )
    elif mutation == "unexpected-artifact":
        (collection / "analysis.json").write_text("{}")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mutation)
    before = _artifact_snapshot(collection)
    with pytest.raises(ValueError):
        load_verified_summary(
            manifest=manifest,
            input_directory=collection,
            scientific=True,
        )
    assert regeneration_calls == []
    assert _artifact_snapshot(collection) == before


@pytest.mark.parametrize(
    "artifact_kind",
    [
        "experiment",
        "protocol",
        "summary",
        "batch-directory",
        "batch-ledger",
    ],
)
def test_verified_summary_rejects_symlinked_collection_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    manifest, collection, regeneration_calls = _raw_collection_fixture(
        tmp_path, monkeypatch
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    if artifact_kind == "batch-directory":
        artifact = next((collection / "batches").iterdir())
        target = outside / artifact.name
        artifact.rename(target)
        artifact.symlink_to(target, target_is_directory=True)
    else:
        artifact = {
            "experiment": collection / "experiment.json",
            "protocol": collection / "protocol.json",
            "summary": collection / "summary.json",
            "batch-ledger": next((collection / "batches").glob("*/*.json")),
        }[artifact_kind]
        target = outside / artifact.name
        target.write_bytes(artifact.read_bytes())
        artifact.unlink()
        artifact.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        load_verified_summary(
            manifest=manifest,
            input_directory=collection,
            scientific=True,
        )
    assert regeneration_calls == []
