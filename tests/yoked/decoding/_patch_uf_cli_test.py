from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "benchmark_patch_uf_mwpm"


@pytest.fixture(scope="module")
def cli() -> dict[str, object]:
    return runpy.run_path(str(TOOL), run_name="patch_uf_cli_test_module")


def _draft(cli: dict[str, object]) -> dict[str, object]:
    semantic_limits = dict(cli["EXPECTED_SEMANTIC_LIMITS"])
    production_limits = dict(cli["EXPECTED_PRODUCTION_LIMITS"])
    stages = {
        cli["SHAKEOUT_STAGE"]: {"shots": 1_000, "seed_root": "11" * 32},
        cli["CHARACTERIZATION_STAGE"]: {
            "shots": 10_000,
            "seed_root": "22" * 32,
        },
    }
    return {
        "schema": cli["PROTOCOL_SCHEMA"],
        "schema_version": 1,
        "status": "DRAFT",
        "frozen": False,
        "claim_status": "non-claim-bearing",
        "source_paths": sorted(cli["MINIMUM_SOURCE_PATHS"]),
        "selected_cell": {
            "cell_id": "cguf-01-d7-n6-y2-r28-p0.003",
            "d": 7,
            "r": 28,
            "p": 0.003,
            "patches": 6,
            "yokes": 2,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
        },
        "dem_options": {
            "decompose_errors": True,
            "approximate_disjoint_errors": True,
        },
        "decoder": {
            "arms": dict(cli["EXPECTED_ARMS"]),
            "policy": {
                "tau": "0x0.0p+0",
                "comparison": "strict-greater-than",
                "semantic_limits": semantic_limits,
                "production_limits": production_limits,
            },
        },
        "collection_limits": {
            "expected_lanes_per_shot": 12,
            "maximum_component_records_per_shot": 4_096,
            "maximum_metric_bytes_per_range": 536_870_912,
        },
        "sampling": {
            "range_count": 32,
            "seed_derivation": cli["SEED_DERIVATION"],
            "stages": stages,
        },
        "diagnostics": {
            "confidence_margin_bin_edges": list(cli["EXPECTED_CONFIDENCE_EDGES"]),
            "cluster_defect_count_display_bins": list(cli["EXPECTED_CLUSTER_BINS"]),
            "bootstrap_seed_root": "44" * 32,
            "replay_selection_seed_root": "55" * 32,
            "bootstrap_replicates": 10_000,
            "alpha": 0.05,
            "maximum_cases_per_category": 100,
        },
        "latency": {
            "schedule_seed": "33" * 32,
            "batches": list(cli["EXPECTED_LATENCY_BATCHES"]),
            "host_policy": {
                "cpu_affinity": [31],
                "expected_host": {
                    "cpu_model": None,
                    "kernel": None,
                    "machine": None,
                    "microcode": None,
                    "os": None,
                },
                "expected_numa_nodes": [],
            },
        },
    }


def test_help_exposes_all_workflow_commands(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    for command in (
        "inspect",
        "smoke",
        "probe",
        "capacity-probe",
        "freeze",
        "verify-protocol",
        "collect",
        "verify-collection",
        "latency",
        "verify-latency",
        "replay",
    ):
        assert command in result.stdout


def test_exact_contract_and_scratch_identity(cli: dict[str, object]) -> None:
    draft = _draft(cli)
    latency = cli["_validate_protocol_contract"](
        draft, expected_status="DRAFT"
    )
    assert [item.batch_size for item in latency.batches] == [1, 64, 1024]
    assert latency.host_policy.cpu_affinity == (31,)
    scratch = cli["_scratch_protocol"](
        draft,
        stage="probe",
        shots=100,
        state={"implementation_commit": "a" * 40, "clean_worktree": True},
    )
    assert scratch["status"] == "SCRATCH"
    assert scratch["sampling"]["stages"]["probe"]["shots"] == 100
    assert "provenance" not in scratch["selected_cell"]
    assert scratch["protocol_self_sha256"] == cli[
        "canonical_protocol_self_sha256"
    ](scratch)
    assert scratch["experiment_id"] == cli["_experiment_id"](
        scratch, scratch=True
    )
    assert scratch["scratch_gate"]["campaign_lock_policy"] == cli[
        "CAMPAIGN_LOCK_POLICY"
    ]
    assert scratch["source_identity"]["campaign_lock_policy"] == cli[
        "CAMPAIGN_LOCK_POLICY"
    ]
    assert "src/yoked/decoding/_patch_uf_casebook_replay.py" in cli[
        "MINIMUM_SOURCE_PATHS"
    ]
    assert "tests/yoked/decoding/_patch_uf_casebook_replay_test.py" in cli[
        "MINIMUM_SOURCE_PATHS"
    ]


def test_metric_capacity_gate_scales_authenticated_probe_shots_to_max_range(
    cli: dict[str, object],
) -> None:
    shot_metrics = {"original_detector_count": 3}
    lane = {"global_shot_id": 7, "lane_offset": 0, "adapter": {"x": 1}}
    component = {
        "global_shot_id": 7,
        "lane_offset": 0,
        "state_collection": "completed_components",
        "adapter": {"cluster_defect_count": 2},
    }
    verified = SimpleNamespace(
        shot_rows=({"global_shot_id": 7, "adapter_metrics": shot_metrics},),
        lane_rows=(lane,),
        component_rows=(component,),
    )
    one_shot_bytes = len(
        cli["canonical_json_bytes"](
            {
                "shot_corrections": [shot_metrics],
                "lane_groups": [
                    [{k: v for k, v in lane.items() if k != "global_shot_id"}]
                ],
                "component_groups": [
                    [
                        {
                            k: v
                            for k, v in component.items()
                            if k != "global_shot_id"
                        }
                    ]
                ],
            }
        )
    )
    required = (
        one_shot_bytes
        * cli["MAX_CHARACTERIZATION_RANGE_SHOTS"]
        * cli["METRIC_CAPACITY_SAFETY_FACTOR"]
    )
    attestation = cli["_metric_capacity_attestation"](
        verified, configured_limit=required
    )
    assert attestation["observed_max_single_shot_metric_bytes"] == one_shot_bytes
    assert attestation["required_metric_bytes_per_range"] == required
    assert attestation["characterization_max_range_shots"] == 313
    with pytest.raises(ValueError, match="probe-derived capacity requirement"):
        cli["_metric_capacity_attestation"](
            verified, configured_limit=required - 1
        )


def test_exact_capacity_probe_requires_two_times_full_range_bytes(
    cli: dict[str, object],
) -> None:
    component_payload = {
        "range": {"range_id": 1, "shot_start": 312, "shot_stop": 625, "shots": 313},
        "provenance": {"graph": "fixed"},
        "payload_sha256": "aa" * 32,
        "shot_index": [[312, 0, 1, 0, 1]],
        "lanes": [
            {"global_shot_id": 312, "lane_offset": 0, "adapter": {"x": 1}}
        ],
        "components": [
            {
                "global_shot_id": 312,
                "lane_offset": 0,
                "state_collection": "completed_components",
                "adapter": {"cluster_defect_count": 2},
            }
        ],
    }
    shot_payload = {
        "payload_sha256": "bb" * 32,
        "shots": [
            {"global_shot_id": 312, "adapter_metrics": {"original_detector_count": 3}}
        ],
    }
    metric_bytes = cli["_range_metric_tree_bytes"](
        component_payload, shot_payload
    )
    protocol = {
        "experiment_id": "11" * 32,
        "protocol_self_sha256": "22" * 32,
        "source_identity": {"source": "fixed"},
        "scratch_gate": {"kind": "capacity-probe"},
        "collection_limits": {
            "maximum_metric_bytes_per_range": 2 * metric_bytes
        },
    }
    record = cli["_capacity_probe_record"](
        protocol=protocol,
        component_payload=component_payload,
        shot_payload=shot_payload,
        component_bytes=b"component",
        shot_bytes=b"shot",
    )
    assert record["canonical_metric_tree_bytes"] == metric_bytes
    assert record["required_metric_bytes_per_range"] == 2 * metric_bytes
    assert cli["_capacity_probe_range"]().shots == 313
    protocol["collection_limits"]["maximum_metric_bytes_per_range"] -= 1
    with pytest.raises(ValueError, match="exact 313-shot capacity-probe"):
        cli["_capacity_probe_record"](
            protocol=protocol,
            component_payload=component_payload,
            shot_payload=shot_payload,
            component_bytes=b"component",
            shot_bytes=b"shot",
        )


def test_capacity_probe_verifier_binds_exact_protocol_and_layout(
    cli: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _draft(cli)
    state = {"implementation_commit": "a" * 40, "clean_worktree": True}
    protocol = cli["_scratch_protocol"](
        draft, stage="capacity-probe", shots=10_000, state=state
    )
    shot_range = cli["_capacity_probe_range"]()
    component_payload = {
        "range": shot_range.as_json(),
        "provenance": {"graph": "fixed"},
        "payload_sha256": "aa" * 32,
        "shot_index": [[shot_range.shot_start, 0, 1, 0, 1]],
        "lanes": [
            {
                "global_shot_id": shot_range.shot_start,
                "lane_offset": 0,
                "adapter": {"x": 1},
            }
        ],
        "components": [
            {
                "global_shot_id": shot_range.shot_start,
                "lane_offset": 0,
                "state_collection": "completed_components",
                "adapter": {"cluster_defect_count": 2},
            }
        ],
    }
    shot_payload = {
        "payload_sha256": "bb" * 32,
        "shots": [
            {
                "global_shot_id": shot_range.shot_start,
                "adapter_metrics": {"original_detector_count": 3},
            }
        ],
    }
    component_bytes = b"component"
    shot_bytes = b"shot"
    record = cli["_capacity_probe_record"](
        protocol=protocol,
        component_payload=component_payload,
        shot_payload=shot_payload,
        component_bytes=component_bytes,
        shot_bytes=shot_bytes,
    )
    root = tmp_path / "capacity"
    (root / "collection" / "component_metrics").mkdir(parents=True)
    (root / "collection" / "shards").mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (root / cli["_component_probe_relative_path"]()).write_bytes(component_bytes)
    (root / cli["_shot_probe_relative_path"]()).write_bytes(shot_bytes)
    (root / "capacity.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setitem(
        cli["_verify_capacity_probe"].__globals__,
        "_read_range_pair",
        lambda **_kwargs: (
            component_payload,
            shot_payload,
            component_bytes,
            shot_bytes,
        ),
    )
    verified, provenance = cli["_verify_capacity_probe"](
        root, draft=draft, state=state
    )
    assert verified == record
    assert provenance == component_payload["provenance"]

    extra = root / "unexpected-empty-directory"
    extra.mkdir()
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        cli["_verify_capacity_probe"](root, draft=draft, state=state)
    extra.rmdir()

    shot_path = root / cli["_shot_probe_relative_path"]()
    shot_path.unlink()
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        cli["_verify_capacity_probe"](root, draft=draft, state=state)
    shot_path.write_bytes(shot_bytes)

    drifted = json.loads(json.dumps(protocol))
    drifted["collection_limits"]["maximum_metric_bytes_per_range"] *= 2
    drifted["experiment_id"] = cli["_experiment_id"](drifted, scratch=True)
    drifted["protocol_self_sha256"] = cli["canonical_protocol_self_sha256"](
        drifted
    )
    (root / "protocol.json").write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="exact draft-derived identity"):
        cli["_verify_capacity_probe"](root, draft=draft, state=state)


def test_replay_bundle_is_canonical_idempotent_and_tamper_evident(
    cli: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = {
        "schema": cli["REPLAY_SCHEMA"],
        "schema_version": 1,
        "status": "reconciled",
        "fresh_process": True,
        "worker_processes": 2,
        "replayed_cases": 1,
        "cases": [{"global_shot_id": 7}],
        "payload_sha256": "ab" * 32,
    }
    observed: dict[str, object] = {}

    def fake_replay(protocol, **kwargs):
        observed.update(kwargs)
        return summary

    monkeypatch.setitem(
        cli["_run_casebook_replay"].__globals__,
        "replay_casebook_in_fresh_process",
        fake_replay,
    )
    out = tmp_path / "replay"
    result = cli["_run_casebook_replay"](
        protocol={"experiment_id": "11" * 32},
        stage=cli["CHARACTERIZATION_STAGE"],
        collection=tmp_path / "collection",
        analysis=tmp_path / "analysis.json",
        out=out,
        worker_processes=2,
    )
    expected = cli["canonical_json_bytes"](summary) + b"\n"
    assert (out / "replay.json").read_bytes() == expected
    assert result["replayed_cases"] == 1
    assert observed["worker_processes"] == 2

    cli["_run_casebook_replay"](
        protocol={"experiment_id": "11" * 32},
        stage=cli["CHARACTERIZATION_STAGE"],
        collection=tmp_path / "collection",
        analysis=tmp_path / "analysis.json",
        out=out,
        worker_processes=2,
    )
    (out / "replay.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        cli["_run_casebook_replay"](
            protocol={"experiment_id": "11" * 32},
            stage=cli["CHARACTERIZATION_STAGE"],
            collection=tmp_path / "collection",
            analysis=tmp_path / "analysis.json",
            out=out,
            worker_processes=2,
        )


def test_contract_rejects_seed_reuse_and_policy_drift(
    cli: dict[str, object],
) -> None:
    reused = _draft(cli)
    reused["diagnostics"]["bootstrap_seed_root"] = reused["latency"][
        "schedule_seed"
    ]
    with pytest.raises(ValueError, match="pairwise distinct"):
        cli["_validate_protocol_contract"](reused, expected_status="DRAFT")

    drifted = _draft(cli)
    drifted["decoder"]["policy"]["tau"] = "0x1.0p-4"
    with pytest.raises(ValueError, match="literal Patch-UF V1"):
        cli["_validate_protocol_contract"](drifted, expected_status="DRAFT")

    cap_drifted = _draft(cli)
    cap_drifted["collection_limits"]["maximum_metric_bytes_per_range"] *= 2
    with pytest.raises(ValueError, match="frozen 512 MiB ceiling"):
        cli["_validate_protocol_contract"](
            cap_drifted, expected_status="DRAFT"
        )


def test_process_tmpdir_max_errors_and_new_json_guards(
    cli: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli["_validate_processes"](32, scientific=True) == 32
    with pytest.raises(ValueError, match="exactly 32"):
        cli["_validate_processes"](31, scientific=True)
    with pytest.raises(ValueError, match="maximum"):
        cli["_validate_processes"](33, scientific=False)

    monkeypatch.delenv("TMPDIR", raising=False)
    with pytest.raises(ValueError, match="TMPDIR"):
        cli["_require_tmpdir"]()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert cli["_require_tmpdir"]() == tmp_path.resolve()

    monkeypatch.setenv("MAX_ERRORS", "")
    with pytest.raises(ValueError, match="remain unset"):
        cli["_reject_max_errors"]()
    monkeypatch.delenv("MAX_ERRORS")

    path = tmp_path / "one.json"
    cli["_write_new_json"](path, {"finite": 1})
    assert json.loads(path.read_text()) == {"finite": 1}
    with pytest.raises(FileExistsError):
        cli["_write_new_json"](path, {"finite": 2})


def test_contract_is_machine_readable_without_protocol(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(TOOL), "verify-protocol", "--describe"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    value = json.loads(result.stdout)
    contract = value["protocol_contract"]
    assert contract["latency"]["host_policy"]["cpu_affinity"] == [31]
    assert len(contract["independent_seed_roots"]) == 5
    assert contract["diagnostics"]["bootstrap_replicates"] == 10_000


def test_campaign_lock_is_exclusive_and_only_recovers_dead_owner(
    cli: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquire = cli["_acquire_campaign_lock"]
    target = tmp_path / "campaign"
    lease = acquire(tmp_path, command="probe", target=target)
    lock_path = tmp_path / "patch-uf-mwpm-campaign-v1.lock"
    assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    with pytest.raises(RuntimeError, match="already active"):
        acquire(tmp_path, command="smoke", target=target)
    lease.release()
    assert not lock_path.exists()

    lock_path.write_text(
        json.dumps(
            {
                "schema": cli["CAMPAIGN_LOCK_SCHEMA"],
                "pid": 999_999_999,
                "command": "collect",
                "target": str(target),
                "policy": cli["CAMPAIGN_LOCK_POLICY"],
            }
        )
    )
    monkeypatch.setitem(
        acquire.__globals__, "_pid_is_alive", lambda _pid: False
    )
    recovered = acquire(tmp_path, command="probe", target=target)
    assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    recovered.release()

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert cli["main"](
        [
            "smoke",
            "--protocol",
            str(tmp_path / "missing.json"),
            "--out",
            str(target),
        ]
    ) == 2
    assert not lock_path.exists()
