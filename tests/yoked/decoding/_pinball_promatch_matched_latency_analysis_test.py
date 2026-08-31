from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding import _pinball_promatch_matched_latency as latency
from yoked.decoding._pinball_promatch_matched_latency import (
    BatchTiming,
    LatencyProtocol,
    LatencyRestartTask,
    LatencyWorkload,
    TimedVariant,
    VARIANT_NAMES,
    capture_host_policy,
    load_authenticated_detector_corpus,
    run_latency_restart,
    write_authenticated_detector_corpus,
)
from yoked.decoding._pinball_promatch_matched_latency_analysis import (
    ANALYSIS_SCHEMA,
    analyze_latency_suite,
    render_latency_markdown,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


class _Call:
    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __call__(self, packed: np.ndarray) -> np.ndarray:
        assert gc.isenabled() is False or len(packed) == 1
        result = np.array(packed[:, :1], copy=True)
        result ^= self.tag
        return result


class _Factory:
    def __init__(self, manifest: str, *, identity_tag: str = "fixture") -> None:
        self.manifest = manifest
        corpus = load_authenticated_detector_corpus(manifest)
        self.suite_identity = {
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "identity_tag": identity_tag,
        }

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        return LatencyWorkload(
            corpus=load_authenticated_detector_corpus(self.manifest),
            variants=tuple(
                TimedVariant(name, _Call(index))
                for index, name in enumerate(VARIANT_NAMES)
            ),
            provenance={
                "fixture": "analysis",
                "pid": os.getpid(),
                "restart_index": restart_index,
                "batch_size": batch_size,
            },
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1
        return self.value


def _write(path: Path, value: object) -> bytes:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return payload


def _build_suite(
    tmp_path: Path,
    *,
    execution_mode: str = "fork-preloaded",
) -> tuple[Path, LatencyProtocol, _Factory]:
    detectors = np.arange(80, dtype=np.uint8).reshape(80, 1) & 0b111
    manifest = write_authenticated_detector_corpus(
        tmp_path / "corpus",
        detectors=detectors,
        num_detectors=3,
        global_shot_ids=tuple(range(80)),
        provenance={"fixture": "analysis"},
    )
    factory = _Factory(str(manifest))
    protocol = LatencyProtocol(
        batches=(
            BatchTiming(1, 2, 2, 1, 1),
            BatchTiming(64, 2, 2, 1, 1),
        ),
        schedule_seed=456,
        host_policy=capture_host_policy(),
    )
    identity, protocol_id, workload_id, suite_id = latency._task_ids(
        factory,
        protocol,
        execution_mode=execution_mode,
    )
    root = tmp_path / "suite"
    root.mkdir()
    _write(root / "protocol.json", protocol.to_json())
    names = []
    hashes = {}
    ratios = {
        "promatch_vs_global": (2, 4),
        "pinball_vs_global": (4, 4),
        "pinball_vs_promatch": (2, 1),
    }
    for batch in protocol.batches:
        for restart in range(batch.restarts):
            _, _, _, spawn_suite_id = latency._task_ids(
                factory,
                protocol,
                execution_mode="spawn-factory",
            )
            task = LatencyRestartTask(
                factory=factory,
                protocol=protocol,
                restart_index=restart,
                batch_size=batch.batch_size,
                protocol_id=protocol_id,
                suite_id=spawn_suite_id,
                workload_id=workload_id,
                workload_identity=identity,
            )
            record = run_latency_restart(task, clock=_Clock())
            record["clock"] = "time.perf_counter_ns"
            record["suite_id"] = suite_id
            record["execution_mode"] = execution_mode
            record["process_start_method"] = (
                "fork" if execution_mode == "fork-preloaded" else "spawn"
            )
            for pair_name, pair in record["pairs"].items():
                ratio = ratios[pair_name][restart]
                for block in pair["numerator_calls"]:
                    block[0]["duration_ns"] = ratio * 10
                for block in pair["denominator_calls"]:
                    block[0]["duration_ns"] = 10
                pair["numerator_block_totals_ns"] = [
                    ratio * 10
                ] * batch.blocks_per_restart
                pair["denominator_block_totals_ns"] = [
                    10
                ] * batch.blocks_per_restart
            name = f"batch-{batch.batch_size}.restart-{restart:02d}.json"
            payload = _write(root / name, record)
            names.append(name)
            hashes[name] = latency._sha256(payload)
    suite = {
        "schema": latency.SUITE_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": identity,
        "fresh_process_per_restart": True,
        "timed_restart_concurrency": 1,
        "restart_concurrency_policy": "serialized-to-avoid-mutual-contention",
        "execution_mode": execution_mode,
        "process_start_method": (
            "fork" if execution_mode == "fork-preloaded" else "spawn"
        ),
        "parent_preload_once": execution_mode == "fork-preloaded",
        "affinity_policy": protocol.host_policy.to_json(),
        "native_threads": 1,
        "restart_ledgers": names,
        "restart_ledger_sha256": hashes,
    }
    _write(root / "suite.json", suite)
    return root, protocol, factory


def _rehash_ledger(root: Path, name: str, value: object) -> None:
    payload = _write(root / name, value)
    suite_path = root / "suite.json"
    suite = json.loads(suite_path.read_text())
    suite["restart_ledger_sha256"][name] = latency._sha256(payload)
    _write(suite_path, suite)


def test_analysis_reports_paired_ratios_quantiles_counts_and_throughput(
    tmp_path: Path,
) -> None:
    root, protocol, factory = _build_suite(tmp_path)

    artifacts = analyze_latency_suite(
        root,
        protocol=protocol,
        factory=factory,
        bootstrap_replicates=200,
        bootstrap_seed=123,
    )
    analysis = artifacts.analysis
    primary = analysis["batches"]["1"]["pairs"]["promatch_vs_global"]
    descriptive = analysis["batches"]["64"]["pairs"]["promatch_vs_global"]

    assert analysis["schema"] == ANALYSIS_SCHEMA
    assert analysis["batch_1_primary"] is True
    assert analysis["secondary_batches_descriptive"] == [64]
    assert primary["inference"]["geometric_paired_block_ratio"][
        "estimate"
    ] == pytest.approx(np.sqrt(8))
    assert primary["inference"]["restart_dispersion"]["values"] == [2.0, 4.0]
    assert primary["inference"]["restart_dispersion"]["median"] == 3.0
    assert primary["numerator_call_summary"]["p50_ns"] == 30.0
    assert primary["numerator_call_summary"]["p95_ns"] == 40.0
    assert primary["denominator_call_summary"]["p99_ns"] == 10.0
    assert primary["raw_counts"] == {
        "restarts": 2,
        "blocks_per_restart": 2,
        "calls_per_side_per_block": 1,
        "calls_per_side": 4,
        "timed_shots_per_side": 4,
    }
    assert primary["descriptive_throughput"] is None
    assert descriptive["raw_counts"]["timed_shots_per_side"] == 256
    assert descriptive["descriptive_throughput"][
        "denominator_shots_per_second"
    ] == 6.4e9
    assert analysis["timing_scope"]["decoder_compilation_excluded"] is True
    assert artifacts.analysis_bytes == canonical_json_bytes(analysis)


def test_bootstrap_is_deterministic_for_configurable_seed(tmp_path: Path) -> None:
    root, _, _ = _build_suite(tmp_path)

    first = analyze_latency_suite(
        root, bootstrap_replicates=100, bootstrap_seed=999
    ).analysis
    second = analyze_latency_suite(
        root, bootstrap_replicates=100, bootstrap_seed=999
    ).analysis
    third = analyze_latency_suite(
        root, bootstrap_replicates=100, bootstrap_seed=1000
    ).analysis

    assert first == second
    first_seed = first["batches"]["1"]["pairs"]["promatch_vs_global"][
        "inference"
    ]["seed"]
    third_seed = third["batches"]["1"]["pairs"]["promatch_vs_global"][
        "inference"
    ]["seed"]
    assert first_seed != third_seed


def test_markdown_uses_standard_arm_names_and_scope_caveats(tmp_path: Path) -> None:
    root, _, _ = _build_suite(tmp_path)
    artifacts = analyze_latency_suite(root, bootstrap_replicates=20)
    report = artifacts.report_markdown

    assert "Global MWPM" in report
    assert "ProMatch" in report
    assert "Pinball" in report
    assert "Batch 1 (primary)" in report
    assert "Batch 64 (descriptive)" in report
    assert "Compilation, corpus preparation, telemetry" in report
    assert "fork/COW" in report
    assert "not hardware" in report
    assert render_latency_markdown(artifacts.analysis) == report


def test_analysis_rejects_hash_tampering_and_unexpected_artifacts(
    tmp_path: Path,
) -> None:
    root, _, _ = _build_suite(tmp_path)
    ledger = root / "batch-1.restart-00.json"
    ledger.write_bytes(ledger.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        analyze_latency_suite(root, bootstrap_replicates=10)

    root2, _, _ = _build_suite(tmp_path / "second")
    (root2 / "unexpected.txt").write_text("not allowed")
    with pytest.raises(ValueError, match="artifact set"):
        analyze_latency_suite(root2, bootstrap_replicates=10)


def test_analysis_rejects_suite_field_drift_and_metadata_symlinks(
    tmp_path: Path,
) -> None:
    root, _, _ = _build_suite(tmp_path)
    suite_path = root / "suite.json"
    suite = json.loads(suite_path.read_text())
    suite["unrecognized"] = True
    _write(suite_path, suite)
    with pytest.raises(ValueError, match="suite fields"):
        analyze_latency_suite(root, bootstrap_replicates=10)

    root2, _, _ = _build_suite(tmp_path / "second")
    protocol_path = root2 / "protocol.json"
    protocol_copy = tmp_path / "protocol-copy.json"
    protocol_copy.write_bytes(protocol_path.read_bytes())
    protocol_path.unlink()
    protocol_path.symlink_to(protocol_copy)
    with pytest.raises(ValueError, match="metadata must be a regular file"):
        analyze_latency_suite(root2, bootstrap_replicates=10)


def test_analysis_rejects_misaligned_pair_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    root, _, _ = _build_suite(tmp_path)
    name = "batch-1.restart-00.json"
    record = json.loads((root / name).read_text())
    call = record["pairs"]["promatch_vs_global"]["denominator_calls"][0][0]
    call["corpus_indices"] = list(reversed(call["corpus_indices"])) + [79]
    _rehash_ledger(root, name, record)

    with pytest.raises(ValueError, match="corpus plan|different workloads"):
        analyze_latency_suite(root, bootstrap_replicates=10)


def test_analysis_rejects_supplied_protocol_or_factory_identity_drift(
    tmp_path: Path,
) -> None:
    root, protocol, factory = _build_suite(tmp_path)
    different_protocol = LatencyProtocol(
        batches=protocol.batches,
        schedule_seed=protocol.schedule_seed + 1,
        host_policy=protocol.host_policy,
    )
    with pytest.raises(ValueError, match="supplied latency protocol"):
        analyze_latency_suite(
            root,
            protocol=different_protocol,
            bootstrap_replicates=10,
        )

    wrong_factory = _Factory(factory.manifest, identity_tag="different")
    with pytest.raises(ValueError, match="factory identity"):
        analyze_latency_suite(
            root,
            factory=wrong_factory,
            bootstrap_replicates=10,
        )


def test_render_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="wrong matched latency schema"):
        render_latency_markdown({"schema": "wrong"})
