from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import yoked.decoding._promatch_experiment as promatch_experiment
from yoked.decoding._promatch_experiment import (
    LEDGER_SCHEMA,
    REPLAY_CATEGORIES,
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
    _validate_ledger_row,
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
    assert normalized["schema"] == "promatch-l1-paired-protocol-v2"
    assert normalized["kind"] == "promatch-l1-paired-fixed-shot"
    assert normalized["phase"] == "pilot"
    assert normalized["claim_bearing"] is False
    assert normalized["replay_policy"] == {
        "categories": ["regression", "recovery", "rollback"],
        "maximum_candidate_rows_per_category_per_batch_ledger": 100,
        "maximum_retained_rows_per_category_per_cell_summary": 100,
        "selection_key": "SHA256_ASCII(cell_id:batch_id:shot_index:category)",
        "batch_ledger_selection": (
            "lowest_selection_sha256_within_batch_and_category"
        ),
        "cell_summary_selection": (
            "lowest_selection_sha256_across_batch_candidates_within_cell_and_category"
        ),
        "equal_cap_prefilter_equivalence": True,
        "invariant_violation_policy": "fatal_run_error_no_replay_row",
        "must_be_replayable": True,
    }
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
    assert tuple(normalized["replay_policy"]["categories"]) == REPLAY_CATEGORIES
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


def test_documented_replay_policy_fails_closed() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "PROMATCH_PILOT_PROTOCOL.json"
    )
    documented = json.loads(path.read_text())
    policy = documented["output_schema"]["bounded_replay_policy"]
    mutations = []

    changed = copy.deepcopy(documented)
    changed["output_schema"]["bounded_replay_policy"]["categories"].append(
        "invariant_debug"
    )
    mutations.append((changed, "replay_policy"))

    changed = copy.deepcopy(documented)
    changed["output_schema"]["bounded_replay_policy"][
        "maximum_candidate_rows_per_category_per_batch_ledger"
    ] = policy["maximum_retained_rows_per_category_per_cell_summary"] - 1
    mutations.append((changed, "replay_policy"))

    changed = copy.deepcopy(documented)
    changed["output_schema"]["bounded_replay_policy"]["selection_key"] = (
        "implementation_defined"
    )
    mutations.append((changed, "replay_policy"))

    changed = copy.deepcopy(documented)
    changed["output_schema"]["bounded_replay_policy"][
        "invariant_violation_policy"
    ] = "retain_invariant_debug"
    mutations.append((changed, "replay_policy"))

    changed = copy.deepcopy(documented)
    changed["output_schema"]["schema_version"] = 1
    mutations.append((changed, "output_schema.schema_version"))

    changed = copy.deepcopy(documented)
    changed["protocol_version"] = "promatch-l1-pilot-v1"
    mutations.append((changed, "protocol_version"))

    for candidate, message in mutations:
        with pytest.raises(ValueError, match=message):
            normalize_protocol(candidate)


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
        replay_policy=default_smoke_protocol(processes=1, shots=1)[
            "replay_policy"
        ],
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


def test_summary_replay_cap_selects_global_lowest_hashes_across_batches() -> None:
    replay_policy = copy.deepcopy(
        default_smoke_protocol(processes=1, shots=1)["replay_policy"]
    )
    replay_policy[
        "maximum_candidate_rows_per_category_per_batch_ledger"
    ] = 2
    replay_policy[
        "maximum_retained_rows_per_category_per_cell_summary"
    ] = 2

    def row(batch_id: int, hashes: list[str]) -> dict:
        return {
            "cell_id": "cell",
            "batch": {"batch_id": batch_id, "shot_start": batch_id, "shots": 1},
            "paired_contingency": {"both_correct": 1},
            "telemetry": {"shots": 1},
            "replay_samples": [
                {"category": "regression", "selection_sha256": value}
                for value in hashes
            ],
        }

    rows = [row(0, ["f" * 64, "0" * 64]), row(1, ["8" * 64, "1" * 64])]
    assert all(len(item["replay_samples"]) <= 2 for item in rows)
    summary = summarize_ledgers(
        rows,
        experiment_id="0" * 64,
        phase="pilot",
        replay_policy=replay_policy,
    )
    retained = summary["cells"][0]["replay_samples"]
    assert [item["selection_sha256"] for item in retained] == ["0" * 64, "1" * 64]


def test_ledger_validation_rejects_noncanonical_replay_categories() -> None:
    protocol = default_smoke_protocol(processes=1, shots=4)
    cell = protocol["cells"][0]
    prepared = prepare_cell(
        cell,
        decoder_config=protocol["decoder"],
        dem_options=protocol["dem_options"],
        verify_hashes=False,
    )
    batch = BatchSpec.from_json(protocol["cell_batch_schedules"][cell["cell_id"]][0])
    row = collect_prepared_batch(
        prepared,
        batch=batch,
        seed_root=protocol["sampler_seed_roots"]["smoke"],
        experiment_id=protocol["experiment_id"],
        phase="smoke",
        replay_policy=protocol["replay_policy"],
    )
    validation_args = {
        "experiment_id": protocol["experiment_id"],
        "phase": "smoke",
        "cell": cell,
        "batch": batch,
        "seed_root": protocol["sampler_seed_roots"]["smoke"],
        "expected_provenance": prepared.provenance,
        "replay_policy": protocol["replay_policy"],
    }
    _validate_ledger_row(row, **validation_args)

    malformed = copy.deepcopy(row)
    malformed["replay_samples"] = [None]
    with pytest.raises(ValueError, match="malformed replay"):
        _validate_ledger_row(malformed, **validation_args)

    detector_width = (prepared.dem.num_detectors + 7) // 8
    observable_width = (prepared.dem.num_observables + 7) // 8
    sample = {
        "selection_sha256": "0" * 64,
        "category": "",
        "batch_id": batch.batch_id,
        "shot_offset": 0,
        "shot_index": batch.shot_start,
        "stim_seed": row["stim_seed"],
        "detection_events_hex": "00" * detector_width,
        "observables_hex": "00" * observable_width,
        "u0_prediction_hex": "00" * observable_width,
        "pu_prediction_hex": "00" * observable_width,
    }
    for category in ("invariant_debug", "invariant-debug", "arbitrary"):
        invalid = copy.deepcopy(row)
        invalid_sample = dict(sample)
        invalid_sample["category"] = category
        invalid["replay_samples"] = [invalid_sample]
        with pytest.raises(ValueError, match="unsupported replay category"):
            _validate_ledger_row(invalid, **validation_args)

    complete = copy.deepcopy(row)
    complete["paired_contingency"] = {
        "both_correct": batch.shots - 1,
        "regressions": 1,
        "recoveries": 0,
        "both_wrong": 0,
    }
    complete["telemetry"]["rollback_shots"] = 0
    regression_sample = dict(sample)
    regression_sample["category"] = "regression"
    regression_sample["selection_sha256"] = (
        promatch_experiment._replay_selection_sha256(
            cell_id=cell["cell_id"],
            batch_id=batch.batch_id,
            shot_index=batch.shot_start,
            category="regression",
        )
    )
    regression_sample["pu_prediction_hex"] = "01" + "00" * (
        observable_width - 1
    )
    complete["replay_samples"] = [regression_sample]
    _validate_ledger_row(complete, **validation_args)
    complete["replay_samples"] = []
    with pytest.raises(ValueError, match="replay samples are incomplete"):
        _validate_ledger_row(complete, **validation_args)

    truncated_policy = copy.deepcopy(protocol["replay_policy"])
    truncated_policy[
        "maximum_candidate_rows_per_category_per_batch_ledger"
    ] = 2
    truncated_policy[
        "maximum_retained_rows_per_category_per_cell_summary"
    ] = 2
    truncated = copy.deepcopy(row)
    truncated["paired_contingency"] = {
        "both_correct": batch.shots - 3,
        "regressions": 3,
        "recoveries": 0,
        "both_wrong": 0,
    }
    truncated["telemetry"]["rollback_shots"] = 0
    truncated["replay_samples"] = [regression_sample]
    with pytest.raises(ValueError, match="replay samples are incomplete"):
        _validate_ledger_row(
            truncated,
            **{**validation_args, "replay_policy": truncated_policy},
        )


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
        replay_policy=protocol["replay_policy"],
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
        <= protocol["replay_policy"][
            "maximum_candidate_rows_per_category_per_batch_ledger"
        ]
        for category in REPLAY_CATEGORIES
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
