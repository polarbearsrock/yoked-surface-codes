from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

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
            "maximum_metric_bytes_per_range": 134_217_728,
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
