from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from yoked.decoding import _patch_uf_latency as latency
from yoked.decoding._patch_uf_analysis import ANALYSIS_SCHEMA
from yoked.decoding._patch_uf_experiment import (
    PROTOCOL_SCHEMA,
    canonical_protocol_self_sha256,
)
from yoked.decoding._patch_uf_latency_analysis import (
    LATENCY_ANALYSIS_SCHEMA,
    analyze_latency_suite,
    write_latency_analysis_bundle,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "analyze_patch_uf_mwpm"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _call(
    *,
    protocol_id: str,
    suite_id: str,
    workload_id: str,
    pair_name: str,
    side: str,
    block: int,
    duration: int,
) -> dict[str, object]:
    identity = {
        "restart": 0,
        "batch_size": 1,
        "pair": pair_name,
        "side": side,
        "block": block,
        "call_index": 0,
    }
    keys = [["corpus", block]]
    return {
        "timing_call_id": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        **identity,
        "schedule_id": "11" * 32,
        "start_offset": block,
        "corpus_indices": [block],
        "workload_keys": keys,
        "workload_key_digest": latency._json_digest(keys),
        "detector_batch_digest": "22" * 32,
        "precomputed_summary_digest": "33" * 32,
        "duration_ns": duration,
    }


def _latency_suite(tmp_path: Path) -> Path:
    host = latency.HostPolicy(
        cpu_affinity=(0,),
        expected_host=(('machine', 'x86_64'),),
        expected_numa_nodes=(),
    )
    protocol = latency.LatencyProtocol(
        batches=(latency.BatchTiming(1, 1, 2, 1, 1),),
        schedule_seed=7,
        host_policy=host,
    )
    protocol_json = protocol.to_json()
    protocol_id = latency._json_digest(protocol_json)
    workload = {"corpus_digest": "44" * 32, "corpus_manifest_sha256": "55" * 32}
    workload_id = latency._json_digest(workload)
    suite_id = latency._json_digest(
        {
            "protocol_id": protocol_id,
            "workload_id": workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
        }
    )
    pairs = {}
    for spec in latency.FIXED_PAIRS:
        numerator = [
            [_call(
                protocol_id=protocol_id,
                suite_id=suite_id,
                workload_id=workload_id,
                pair_name=spec.name,
                side="numerator",
                block=block,
                duration=10 * (block + 1),
            )]
            for block in range(2)
        ]
        denominator = [
            [_call(
                protocol_id=protocol_id,
                suite_id=suite_id,
                workload_id=workload_id,
                pair_name=spec.name,
                side="denominator",
                block=block,
                duration=20 * (block + 1),
            )]
            for block in range(2)
        ]
        pairs[spec.name] = {
            "pair": spec.name,
            "numerator": spec.numerator,
            "denominator": spec.denominator,
            "order_by_block": ["AB", "BA"],
            "numerator_calls": numerator,
            "denominator_calls": denominator,
            "numerator_block_totals_ns": [10, 20],
            "denominator_block_totals_ns": [20, 40],
        }
    record = {
        "schema": latency.RESTART_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": workload,
        "restart_index": 0,
        "batch_size": 1,
        "corpus": {"row_count": 2},
        "untimed_prediction_check": {
            "checked_rows": 1,
            "corpus_index": 0,
            "global_shot_id": 0,
            "full_corpus_prediction_attestation_sha256": None,
            "prediction_digests": {
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in latency.VARIANT_NAMES
            },
            "equality_groups": {
                "global_control_shadow_backend_original": [
                    "global_mwpm",
                    "adapter_control",
                    "uf_shadow",
                    "backend_original",
                ],
                "treatment_backend_residual": [
                    "treatment",
                    "backend_residual",
                ],
            },
        },
        "pairs": pairs,
    }
    root = tmp_path / "latency"
    _write(root / "protocol.json", canonical_json_bytes(protocol_json) + b"\n")
    restart = root / "batch-1.restart-00.json"
    _write(restart, canonical_json_bytes(record) + b"\n")
    suite = {
        "schema": latency.SUITE_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": workload,
        "fresh_process_per_restart": True,
        "timed_restart_concurrency": 1,
        "affinity_policy": host.to_json(),
        "restart_ledgers": [restart.name],
        "restart_ledger_sha256": {
            restart.name: hashlib.sha256(restart.read_bytes()).hexdigest()
        },
    }
    _write(root / "suite.json", canonical_json_bytes(suite) + b"\n")
    return root


def _analysis_bundle(
    root: Path, *, shots: int, experiment_id: str, protocol_sha: str
) -> None:
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": 1,
        "source": {
            "experiment_id": experiment_id,
            "protocol_self_sha256": protocol_sha,
        },
        "reconciliation": {"shots": shots},
    }
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    _write(root / "analysis.json", canonical_json_bytes(payload))
    _write(root / "report.md", b"report\n")


def test_synthetic_latency_ledgers_are_authenticated_and_analyzed(tmp_path) -> None:
    suite = _latency_suite(tmp_path)
    first = analyze_latency_suite(suite, bootstrap_replicates=30)
    second = analyze_latency_suite(suite, bootstrap_replicates=30)

    assert first.analysis_bytes == second.analysis_bytes
    net = first.analysis["batches"]["1"]["pairs"]["net_total"]
    assert net["inference"]["geometric_paired_block_ratio"]["estimate"] == pytest.approx(0.5)
    assert net["inference"]["pooled_type7_p99_ratio"]["estimate"] == pytest.approx(0.5)
    assert net["numerator_summary"]["p99_ns"] == pytest.approx(19.9)
    out = tmp_path / "latency-analysis"
    write_latency_analysis_bundle(out, first)
    write_latency_analysis_bundle(out, first)
    assert json.loads((out / "latency_analysis.json").read_bytes())["schema"] == LATENCY_ANALYSIS_SCHEMA

    restart = suite / "batch-1.restart-00.json"
    restart.write_bytes(restart.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        analyze_latency_suite(suite, bootstrap_replicates=2)


def test_finalize_cli_binds_three_authenticated_analysis_identities(tmp_path) -> None:
    experiment_id = "66" * 32
    protocol_sha = "77" * 32
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "experiment_id": experiment_id,
    }
    protocol["protocol_self_sha256"] = canonical_protocol_self_sha256(protocol)
    protocol_sha = protocol["protocol_self_sha256"]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    shakeout = tmp_path / "shakeout"
    characterization = tmp_path / "characterization"
    _analysis_bundle(
        shakeout,
        shots=1_000,
        experiment_id=experiment_id,
        protocol_sha=protocol_sha,
    )
    _analysis_bundle(
        characterization,
        shots=10_000,
        experiment_id=experiment_id,
        protocol_sha=protocol_sha,
    )
    latency_root = tmp_path / "latency-analysis"
    latency_payload = {
        "schema": LATENCY_ANALYSIS_SCHEMA,
        "suite_id": "88" * 32,
        "protocol_id": "99" * 32,
        "workload_id": "aa" * 32,
        "workload_identity": {
            "experiment_id": experiment_id,
            "protocol_self_sha256": protocol_sha,
        },
    }
    latency_payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(latency_payload)
    ).hexdigest()
    _write(latency_root / "latency_analysis.json", canonical_json_bytes(latency_payload))
    _write(latency_root / "report.md", b"latency\n")
    out = tmp_path / "final"
    env = {
        **os.environ,
        "TMPDIR": os.environ["TMPDIR"],
        "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
    }
    command = [
        sys.executable,
        str(TOOL),
        "finalize",
        "--protocol",
        str(protocol_path),
        "--shakeout",
        str(shakeout),
        "--characterization",
        str(characterization),
        "--latency",
        str(latency_root),
        "--out",
        str(out),
        "--allow-non-scientific",
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    result = json.loads((out / "finalization.json").read_bytes())
    assert result["schema"] == "patch-uf-finalization-v1"
    assert result["sources"]["shakeout_1k"]["payload_sha256"]
    assert result["sources"]["latency"]["suite_id"] == "88" * 32
