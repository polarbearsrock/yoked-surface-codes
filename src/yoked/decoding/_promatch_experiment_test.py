from __future__ import annotations

import json
from pathlib import Path

import pytest

import yoked.decoding._promatch_experiment as promatch_experiment
from yoked.decoding._promatch_experiment import (
    LEDGER_SCHEMA,
    SUMMARY_SCHEMA,
    build_batch_schedule,
    collect_prepared_batch,
    default_smoke_protocol,
    normalize_protocol,
    prepare_cell,
    run_collection,
    summarize_ledgers,
    validate_experiment_protocol,
    freeze_protocol,
    _retain_lowest_hash_samples,
    _verify_regenerated_scientific_ledger,
)
from yoked.decoding._promatch_stats import (
    BatchSpec,
    manifest_experiment_id,
    validate_process_count,
)


def test_batch_schedule_is_fixed_and_processes_are_capped() -> None:
    assert build_batch_schedule(20_001) == [
        {"batch_id": 0, "shot_start": 0, "shots": 10_000},
        {"batch_id": 1, "shot_start": 10_000, "shots": 10_000},
        {"batch_id": 2, "shot_start": 20_000, "shots": 1},
    ]
    assert validate_process_count(32) == 32
    with pytest.raises(ValueError, match="exceeds"):
        validate_process_count(33)


def test_freeze_hashes_the_input_template_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[dict, bool]] = []

    def stop_after_template_hash(draft: dict, *, for_freeze: bool) -> dict:
        seen.append((draft, for_freeze))
        raise RuntimeError("normalization reached")

    monkeypatch.setattr(promatch_experiment, "normalize_protocol", stop_after_template_hash)
    with pytest.raises(RuntimeError, match="normalization reached"):
        freeze_protocol({"phase": "pilot"})
    assert seen == [({"phase": "pilot"}, True)]


def test_documented_pilot_template_normalizes_to_disjoint_cell_schedules() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "PROMATCH_PILOT_PROTOCOL.json"
    )
    normalized = normalize_protocol(json.loads(path.read_text()))
    assert normalized["schema"] == "promatch-l1-paired-protocol-v1"
    assert normalized["kind"] == "promatch-l1-paired-fixed-shot"
    assert normalized["phase"] == "pilot"
    assert normalized["claim_bearing"] is False
    assert len(normalized["cells"]) == 5
    ranges = []
    for cell in normalized["cells"]:
        schedule = normalized["cell_batch_schedules"][cell["cell_id"]]
        assert len(schedule) == 20
        ranges.append((schedule[0]["batch_id"], schedule[-1]["batch_id"]))
    assert ranges == [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]

    candidate = json.loads(json.dumps(normalized))
    candidate["status"] = "FROZEN"
    candidate["frozen"] = True
    for cell in candidate["cells"]:
        cell.update(
            circuit_sha256="0" * 64,
            dem_sha256="0" * 64,
            layout_fingerprint="0" * 64,
            graph_fingerprint="0" * 64,
        )
    # It reaches ordinary frozen-provenance validation (missing cell hashes),
    # proving the exact-grid check has a bound, validated cells collection.
    with pytest.raises(ValueError, match="null is forbidden"):
        validate_experiment_protocol(
            candidate, phase="pilot", scientific=True, processes=32
        )


def test_documented_confirm_template_is_inspectable_but_not_freezable() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "PROMATCH_FIRST_ROUND_PROTOCOL.json"
    )
    documented = json.loads(path.read_text())
    normalized = normalize_protocol(documented)
    assert normalized["kind"] == "promatch-l1-paired-fixed-shot"
    assert normalized["phase"] == "confirm"
    assert normalized["cells"][0]["cell_id"] == "target-d11-n6-y2-r44-p0.001"
    assert normalized["performance_cells"] == normalized["cells"]
    target_id = "target-d11-n6-y2-r44-p0.001"
    target_schedule = normalized["performance_cell_batch_schedules"][target_id]
    assert normalized["performance_expected_shots_by_cell"] == {
        target_id: 1_000_000
    }
    assert len(target_schedule) == 100
    assert [row["batch_id"] for row in target_schedule] == list(range(100))
    assert all(row["shots"] == 10_000 for row in target_schedule)
    with_hashes = json.loads(json.dumps(normalized))
    with_hashes["performance_cells"][0].update(
        circuit_sha256="0" * 64,
        dem_sha256="0" * 64,
        layout_fingerprint="0" * 64,
        graph_fingerprint="0" * 64,
    )
    before = manifest_experiment_id(with_hashes)
    changed = json.loads(json.dumps(with_hashes))
    changed["performance_cells"][0]["graph_fingerprint"] = "1" * 64
    assert manifest_experiment_id(changed) != before
    with pytest.raises(ValueError, match="selection.selected_cell"):
        normalize_protocol(documented, for_freeze=True)


def test_target_validation_enforces_exact_performance_geometry_before_provenance() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "PROMATCH_FIRST_ROUND_PROTOCOL.json"
    )
    candidate = normalize_protocol(json.loads(path.read_text()))
    candidate["status"] = "FROZEN"
    candidate["frozen"] = True
    candidate["claim_bearing"] = True
    for key in ("cells", "performance_cells"):
        for cell in candidate[key]:
            cell.update(
                circuit_sha256="0" * 64,
                dem_sha256="0" * 64,
                layout_fingerprint="0" * 64,
                graph_fingerprint="0" * 64,
            )
    candidate["performance_cells"][0]["d"] = 10
    with pytest.raises(ValueError, match="exact frozen d11 target geometry"):
        validate_experiment_protocol(
            candidate, phase="target", scientific=True, processes=32
        )
    candidate["performance_cells"][0]["d"] = 11
    # Target performance shares the same frozen first-round protocol, so it
    # also refuses an underived/manual pilot selection.
    with pytest.raises(ValueError, match="pilot selection"):
        validate_experiment_protocol(
            candidate, phase="target", scientific=True, processes=32
        )


def test_target_summary_is_explicitly_performance_only() -> None:
    summary = summarize_ledgers(
        [],
        experiment_id="0" * 64,
        phase="target",
        disagreement_cap=0,
    )
    assert summary["collection_scope"] == (
        "target-performance-only-no-accuracy-confirmation"
    )


def test_target_collection_routes_to_performance_cells_and_seed(tmp_path) -> None:
    protocol = default_smoke_protocol(processes=1, shots=7)
    performance_cell = dict(protocol["cells"][0])
    performance_cell["cell_id"] = "performance-smoke"
    protocol["performance_cells"] = [performance_cell]
    protocol["performance_expected_shots_by_cell"] = {"performance-smoke": 7}
    protocol["performance_cell_batch_schedules"] = {
        "performance-smoke": build_batch_schedule(7)
    }
    protocol["sampler_seed_roots"]["target_workload"] = "4b" * 32
    protocol["experiment_id"] = manifest_experiment_id(protocol)
    summary = run_collection(
        protocol,
        phase="target",
        out=tmp_path / "target",
        processes=1,
        scientific=False,
    )
    assert summary["phase"] == "target"
    assert summary["cells"][0]["cell_id"] == "performance-smoke"
    assert summary["cells"][0]["shots"] == 7
    ledger = next((tmp_path / "target" / "batches").glob("*/*.json"))
    row = json.loads(ledger.read_text())
    assert row["cell_id"] == "performance-smoke"
    assert row["phase"] == "target"


def test_replay_retention_uses_lowest_hash_not_arrival_order() -> None:
    samples = [
        {"selection_sha256": "f" * 64, "value": 1},
        {"selection_sha256": "0" * 64, "value": 2},
        {"selection_sha256": "8" * 64, "value": 3},
    ]
    assert [row["value"] for row in _retain_lowest_hash_samples(samples, 2)] == [2, 3]


def test_scientific_resume_rejects_any_regenerated_payload_difference() -> None:
    _verify_regenerated_scientific_ledger({"digest": "a"}, {"digest": "a"}, path="x")
    with pytest.raises(ValueError, match="regenerated scientific batch"):
        _verify_regenerated_scientific_ledger(
            {"digest": "claimed"}, {"digest": "actual"}, path="batch.json"
        )


def test_scientific_collection_rejects_draft_before_repository_checks() -> None:
    protocol = default_smoke_protocol(processes=32, shots=17)
    protocol["phase"] = "pilot"
    protocol["sampler_seed_roots"] = {"pilot": "3a" * 32}
    protocol["experiment_id"] = manifest_experiment_id(protocol)
    with pytest.raises(ValueError, match="status='FROZEN'"):
        validate_experiment_protocol(
            protocol, phase="pilot", scientific=True, processes=32
        )


def test_real_paired_batch_has_complete_reconciling_telemetry() -> None:
    protocol = default_smoke_protocol(processes=1, shots=64)
    cell = protocol["cells"][0]
    prepared = prepare_cell(
        cell,
        decoder_config=protocol["decoder"],
        dem_options=protocol["dem_options"],
        verify_hashes=False,
    )
    row = collect_prepared_batch(
        prepared,
        batch=BatchSpec.from_json(
            protocol["cell_batch_schedules"][cell["cell_id"]][0]
        ),
        seed_root=protocol["sampler_seed_roots"]["smoke"],
        experiment_id=protocol["experiment_id"],
        phase="smoke",
        disagreement_cap=protocol["disagreement_cap"],
    )

    assert row["schema"] == LEDGER_SCHEMA
    assert sum(row["paired_contingency"].values()) == 64
    assert row["detectors"]["shape"][0] == 64
    assert row["observables"]["shape"][0] == 64
    telemetry = row["telemetry"]
    assert telemetry["shots"] == 64
    assert telemetry["activated_shots"] == 0
    assert row["paired_contingency"]["regressions"] == 0
    assert row["paired_contingency"]["recoveries"] == 0
    assert sum(telemetry["domain_status_counts"].values()) == (
        64 * len(prepared.compiled_pu.graph.domain_graphs)
    )
    assert len(telemetry["attempted_stage_counts"]) == 4
    assert len(telemetry["committed_stage_counts"]) == 4
    assert "decision_weight_histogram_float_hex" in telemetry
    assert "xor_support_weight_histogram_float_hex" in telemetry
    assert "committed_path_length_histogram" in telemetry
    assert "terminal_withheld_event_sum" in telemetry
    assert "yoke_withheld_event_sum" in telemetry
    assert "domain_identity_counts" in telemetry
    assert sum(telemetry["original_residual_hw_joint_histogram"].values()) == 64
    assert all(
        len([s for s in row["replay_samples"] if s["category"] == category])
        <= protocol["disagreement_cap"]
        for category in ("regression", "recovery", "rollback")
    )


def test_smoke_collection_resumes_from_complete_ledger(tmp_path) -> None:
    protocol = default_smoke_protocol(processes=1, shots=16)
    out = tmp_path / "paired"
    first = run_collection(
        protocol, phase="smoke", out=out, processes=1, scientific=False
    )
    assert first["schema"] == SUMMARY_SCHEMA
    ledger = next((out / "batches").glob("*/*.json"))
    before = ledger.read_bytes()
    second = run_collection(
        protocol, phase="smoke", out=out, processes=1, scientific=False
    )
    assert second == first
    assert ledger.read_bytes() == before
    assert json.loads((out / "summary.json").read_text()) == first


def test_resume_rejects_output_experiment_mismatch(tmp_path) -> None:
    protocol = default_smoke_protocol(processes=1, shots=1)
    out = tmp_path / "wrong"
    out.mkdir()
    (out / "experiment.json").write_text(
        json.dumps(
            {
                "schema": protocol["schema"],
                "experiment_id": "0" * 64,
                "phase": "smoke",
            }
        )
    )
    with pytest.raises(ValueError, match="different experiment"):
        run_collection(
            protocol, phase="smoke", out=out, processes=1, scientific=False
        )
