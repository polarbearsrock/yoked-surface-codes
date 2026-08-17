from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import yoked.decoding._promatch_latency_analysis as latency_analysis
from yoked.decoding._promatch_experiment import (
    default_smoke_protocol,
    normalize_protocol,
)
from yoked.decoding._promatch_latency import (
    BACKEND_RESIDUAL_VS_ORIGINAL,
    LATENCY_SUITE_SCHEMA,
    TOTAL_PU_VS_DIRECT,
    TOTAL_PU_VS_WRAP,
    LatencyWorkload,
    run_latency_restart,
)
from yoked.decoding._promatch_latency_analysis import (
    LATENCY_ANALYSIS_SCHEMA,
    TinyLatencyAnalysisConfig,
    analyze_latency_suite,
    render_latency_markdown,
)
from yoked.decoding._promatch_latency_integration import (
    TinyLatencySmokeConfig,
    YokedPromatchLatencyFactory,
    latency_protocol_from_manifest,
)
from yoked.decoding._promatch_stats import (
    canonical_json_bytes,
    derive_stim_batch_seed,
    manifest_experiment_id,
)


class _ManualClock:
    def __init__(self) -> None:
        self.now = 1_000

    def __call__(self) -> int:
        return self.now


class _SyntheticBoundFactory:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        identity: dict[str, Any],
        protocol: Any,
        timing_corpus_root: str,
    ) -> None:
        self.manifest = manifest
        self.identity = identity
        self.protocol = protocol
        self.timing_corpus_root = timing_corpus_root
        self.clock = _ManualClock()

    def _call(self, duration: int) -> Callable[[np.ndarray], np.ndarray]:
        def invoke(batch: np.ndarray) -> np.ndarray:
            self.clock.now += duration
            return np.zeros((len(batch), 1), dtype=np.uint8)

        return invoke

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        batch_index = self.protocol.batch_sizes.index(batch_size)
        corpus_batch_id = restart_index * len(self.protocol.batch_sizes) + batch_index
        stim_seed = derive_stim_batch_seed(
            seed_root=self.timing_corpus_root,
            batch_id=corpus_batch_id,
        )
        shots = batch_size * self.identity["corpus_batches"]
        total = np.arange(shots * 2, dtype=np.uint8).reshape(shots, 2)
        total = np.bitwise_xor(total, np.uint8(restart_index + batch_size))
        residual = np.bitwise_xor(total, np.uint8(0x55))
        provenance = {
            "experiment_id": self.manifest.get("experiment_id"),
            "cell_id": self.identity["cell_id"],
            **self.identity.get("cell_hashes", {}),
            "decoder_config_sha256": self.identity["decoder_config_sha256"],
            "decoder_config": self.manifest["decoder"],
            "dem_options": self.manifest["dem_options"],
            "restart_index": restart_index,
            "batch_size": batch_size,
            "corpus_batches": self.identity["corpus_batches"],
            "corpus_shots_per_restart": self.identity[
                "corpus_shots_per_restart"
            ],
            "complete_timing_batches": shots // batch_size,
            "corpus_batch_id": corpus_batch_id,
            "stim_seed": stim_seed,
            "timing_corpus_seed_root_sha256": hashlib.sha256(
                bytes.fromhex(self.timing_corpus_root)
            ).hexdigest(),
            "u0_backend": "pymatching-uncorrelated",
            "residual_backend": "pymatching-uncorrelated",
            "correlated_matching": False,
            "residual_generation_retained_shot_telemetry": False,
        }
        return LatencyWorkload(
            total_corpus=total,
            u0_direct=self._call(10),
            u0_wrap=self._call(20),
            pu_window=self._call(5),
            backend_original_corpus=total,
            backend_residual_corpus=residual,
            backend_original=self._call(8),
            backend_residual=self._call(3),
            provenance=provenance,
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _canonical_file_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _refresh_ledger_hash(output: Path, name: str) -> None:
    suite_path = output / "suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["restart_ledger_sha256"][name] = _canonical_file_sha256(output / name)
    _write_json(suite_path, suite)


def _ids(protocol: Any, identity: dict[str, Any]) -> tuple[str, str, str]:
    protocol_id = hashlib.sha256(
        canonical_json_bytes(protocol.to_json(scientific=False))
    ).hexdigest()
    workload_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    suite_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_id": protocol_id,
                "workload_id": workload_id,
                "claim_bearing": False,
                "configured_processes": 1,
                "timed_restart_concurrency": 1,
                "affinity_policy": "inherit-and-record",
            }
        )
    ).hexdigest()
    return protocol_id, workload_id, suite_id


def _smoke_inputs(tmp_path: Path) -> tuple[Path, dict[str, Any], TinyLatencyAnalysisConfig]:
    manifest = default_smoke_protocol(processes=1, shots=4)
    collection = TinyLatencySmokeConfig(
        restarts=2,
        blocks_per_restart=2,
        calls_per_block=3,
        warmup_calls_per_variant=1,
        batch_sizes=(1, 2),
        schedule_seed=0x123456,
        corpus_batches=2,
        timing_corpus_seed_root="91" * 32,
    )
    smoke = TinyLatencyAnalysisConfig(
        collection=collection,
        bootstrap_replicates=50,
        alpha_one_sided=0.025,
    )
    protocol = collection.protocol()
    cell_id = manifest["cells"][0]["cell_id"]
    yoked_factory = YokedPromatchLatencyFactory.from_manifest(
        manifest,
        cell_id=cell_id,
        scientific=False,
        smoke=collection,
    )
    identity = dict(yoked_factory.suite_identity)
    protocol_id, workload_id, suite_id = _ids(protocol, identity)
    factory = _SyntheticBoundFactory(
        manifest=manifest,
        identity=identity,
        protocol=protocol,
        timing_corpus_root=collection.timing_corpus_seed_root,
    )
    output = tmp_path / "latency"
    output.mkdir(parents=True)
    _write_json(output / "protocol.json", manifest)
    ledgers: list[str] = []
    for batch_size in protocol.batch_sizes:
        for restart_index in range(protocol.restarts):
            factory.clock = _ManualClock()
            record = run_latency_restart(
                factory,
                protocol=protocol,
                restart_index=restart_index,
                batch_size=batch_size,
                scientific=False,
                clock=factory.clock,
                workload_identity=identity,
                workload_id=workload_id,
                suite_id=suite_id,
            )
            name = f"batch-{batch_size}.restart-{restart_index:02d}.json"
            _write_json(output / name, record)
            ledgers.append(name)
    suite = {
        "schema": LATENCY_SUITE_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": identity,
        "claim_bearing": False,
        "protocol": protocol.to_json(scientific=False),
        "processes": 1,
        "process_cap": 32,
        "timed_restart_concurrency": 1,
        "restart_concurrency_policy": "serialized-to-avoid-mutual-contention",
        "affinity_policy": "inherit-and-record",
        "fresh_process_per_restart": True,
        "restart_ledgers": ledgers,
        "restart_ledger_sha256": {
            name: _canonical_file_sha256(output / name)
            for name in ledgers
        },
    }
    _write_json(output / "suite.json", suite)
    return output, manifest, smoke


def test_analyze_nonclaim_smoke_aggregates_frozen_shapes_and_gates(
    tmp_path: Path,
) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    cell_id = manifest["cells"][0]["cell_id"]
    result = analyze_latency_suite(
        output,
        manifest=manifest,
        cell_id=cell_id,
        scientific=False,
        smoke=smoke,
    )

    assert result["schema"] == LATENCY_ANALYSIS_SCHEMA
    assert result["claim_bearing"] is False
    assert result["cell_role"] == "explicit_nonclaim_smoke"
    assert set(result["batch_results"]) == {"1", "2"}
    direct = result["batch_results"]["1"]["pairs"][TOTAL_PU_VS_DIRECT]
    wrap = result["batch_results"]["1"]["pairs"][TOTAL_PU_VS_WRAP]
    backend = result["batch_results"]["1"]["pairs"][
        BACKEND_RESIDUAL_VS_ORIGINAL
    ]
    assert direct["array_shape"] == [2, 2, 3]
    assert direct["geometric_ratio"] == pytest.approx(0.5)
    assert direct["p99_ratio"] == pytest.approx(0.5)
    assert direct["numerator_raw_timing"]["median_ns"] == 5
    assert direct["denominator_raw_timing"]["p99_ns"] == 10
    assert wrap["geometric_ratio"] == pytest.approx(0.25)
    assert backend["geometric_ratio"] == pytest.approx(3 / 8)
    assert result["batch_1_gates"]["residual_backend_relief_passed"] is True
    assert (
        result["batch_1_gates"][
            "end_to_end_software_latency_improvement_passed"
        ]
        is True
    )
    assert result["batch_1_gates"]["claim_authorized"] is False
    assert result["claim_scope"]["hardware_latency_claim_authorized"] is False
    seeds = {
        pair["bootstrap_seed"]
        for batch in result["batch_results"].values()
        for pair in batch["pairs"].values()
    }
    assert len(seeds) == 6
    assert len(result["analysis_sha256"]) == 64


def test_render_markdown_is_explicitly_software_not_hardware(tmp_path: Path) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    result = analyze_latency_suite(
        output,
        manifest=manifest,
        cell_id=manifest["cells"][0]["cell_id"],
        scientific=False,
        smoke=smoke,
    )
    markdown = render_latency_markdown(result)
    assert "in-process software latency only" in markdown
    assert "not hardware latency" in markdown
    assert "real-time deadline" in markdown
    assert "U0-wrap is diagnostic only" in markdown
    assert "Claim authorized: `False`" in markdown


def test_missing_and_extra_restart_ledgers_fail_closed(tmp_path: Path) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    cell_id = manifest["cells"][0]["cell_id"]
    (output / "batch-1.restart-00.json").unlink()
    with pytest.raises(ValueError, match="artifact set mismatch"):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=cell_id,
            scientific=False,
            smoke=smoke,
        )

    output, manifest, smoke = _smoke_inputs(tmp_path / "second")
    _write_json(output / "batch-99.restart-99.json", {"extra": True})
    with pytest.raises(ValueError, match="artifact set mismatch"):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=manifest["cells"][0]["cell_id"],
            scientific=False,
            smoke=smoke,
        )


def test_tampered_restart_ledger_hash_fails_closed(tmp_path: Path) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    path = output / "batch-1.restart-00.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["pairs"][TOTAL_PU_VS_DIRECT]["numerator_calls_ns"][0][0] += 1
    _write_json(path, record)
    with pytest.raises(ValueError, match="restart ledger hash mismatch"):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=manifest["cells"][0]["cell_id"],
            scientific=False,
            smoke=smoke,
        )


@pytest.mark.parametrize("corruption", ["nonpositive", "dimension", "total", "mixed"])
def test_corrupt_or_mixed_restart_ledger_fails_closed(
    tmp_path: Path,
    corruption: str,
) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    path = output / "batch-1.restart-00.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    pair = record["pairs"][TOTAL_PU_VS_DIRECT]
    if corruption == "nonpositive":
        pair["numerator_calls_ns"][0][0] = 0
    elif corruption == "dimension":
        pair["denominator_calls_ns"][0].pop()
    elif corruption == "total":
        pair["numerator_block_totals_ns"][0] += 1
    else:
        record["workload_id"] = "f" * 64
    _write_json(path, record)
    _refresh_ledger_hash(output, path.name)
    with pytest.raises(ValueError):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=manifest["cells"][0]["cell_id"],
            scientific=False,
            smoke=smoke,
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    (output / "suite.json").write_text(
        '{"schema":"first","schema":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=manifest["cells"][0]["cell_id"],
            scientific=False,
            smoke=smoke,
        )


def _latency_artifact_snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-protocol",
        "tampered-protocol",
        "duplicate-protocol-key",
        "duplicate-restart-key",
        "unexpected-analysis",
    ],
)
def test_latency_analysis_preflight_is_exact_and_read_only(
    tmp_path: Path, mutation: str
) -> None:
    output, manifest, smoke = _smoke_inputs(tmp_path)
    protocol_path = output / "protocol.json"
    if mutation == "missing-protocol":
        protocol_path.unlink()
    elif mutation == "tampered-protocol":
        changed = json.loads(protocol_path.read_text())
        changed["processes"] = 2
        _write_json(protocol_path, changed)
    elif mutation == "duplicate-protocol-key":
        protocol_path.write_text('{"schema":"first","schema":"second"}')
    elif mutation == "duplicate-restart-key":
        restart = output / "batch-1.restart-00.json"
        restart.write_text('{"schema":"first","schema":"second"}')
    elif mutation == "unexpected-analysis":
        (output / "analysis.json").write_text("{}")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mutation)
    before = _latency_artifact_snapshot(output)
    with pytest.raises(ValueError):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=manifest["cells"][0]["cell_id"],
            scientific=False,
            smoke=smoke,
        )
    assert _latency_artifact_snapshot(output) == before


def _frozen_scientific_manifest() -> dict[str, Any]:
    document = Path(__file__).resolve().parents[3] / "docs" / "PROMATCH_FIRST_ROUND_PROTOCOL.json"
    manifest = normalize_protocol(json.loads(document.read_text(encoding="utf-8")))
    hashes = {
        "circuit_sha256": "11" * 32,
        "dem_sha256": "22" * 32,
        "layout_fingerprint": "33" * 32,
        "graph_fingerprint": "44" * 32,
    }
    selected = {
        "cell_id": "selected-d7-n6-y2-r28-p0.002",
        "generator": "yoked._yoked_memory_circuits:yoked_magic_memory_circuit",
        "d": 7,
        "r": 28,
        "p": 0.002,
        "patches": 6,
        "yokes": 2,
        "style": "cz",
        "noise": "si1000",
        "remove_x_yoke": False,
        **hashes,
    }
    target = dict(manifest["performance_cells"][0])
    target.update(hashes)
    manifest.update(
        status="FROZEN",
        frozen=True,
        claim_bearing=True,
        processes=32,
        cells=[selected],
        performance_cells=[target],
    )
    manifest["analysis_config"]["selection"]["selected_cell_id"] = selected[
        "cell_id"
    ]
    manifest["analysis_config"]["selection"]["selected_cell"] = {
        "cell_id": selected["cell_id"]
    }
    manifest["analysis_config"]["selection"]["n_confirm"] = 10_000
    manifest["expected_shots_by_cell"] = {selected["cell_id"]: 10_000}
    manifest["cell_batch_schedules"] = {
        selected["cell_id"]: [{"batch_id": 0, "shot_start": 0, "shots": 10_000}]
    }
    manifest.pop("experiment_id", None)
    manifest["experiment_id"] = manifest_experiment_id(manifest)
    return manifest


def test_scientific_suite_requires_exactly_32_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _frozen_scientific_manifest()
    cell_id = manifest["cells"][0]["cell_id"]
    protocol = latency_protocol_from_manifest(manifest, scientific=True)
    factory = YokedPromatchLatencyFactory.from_manifest(
        manifest,
        cell_id=cell_id,
        scientific=True,
    )
    identity = dict(factory.suite_identity)
    protocol_json = protocol.to_json(scientific=True)
    protocol_id = hashlib.sha256(canonical_json_bytes(protocol_json)).hexdigest()
    workload_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    suite_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_id": protocol_id,
                "workload_id": workload_id,
                "claim_bearing": True,
                "configured_processes": 32,
                "timed_restart_concurrency": 1,
                "affinity_policy": "inherit-and-record",
            }
        )
    ).hexdigest()
    output = tmp_path / "scientific"
    output.mkdir()
    _write_json(output / "protocol.json", manifest)
    restart_names = [
        f"batch-{batch}.restart-{restart:02d}.json"
        for batch in protocol.batch_sizes
        for restart in range(protocol.restarts)
    ]
    for name in restart_names:
        _write_json(output / name, {})
    _write_json(
        output / "suite.json",
        {
            "schema": LATENCY_SUITE_SCHEMA,
            "protocol_id": protocol_id,
            "suite_id": suite_id,
            "workload_id": workload_id,
            "workload_identity": identity,
            "claim_bearing": True,
            "protocol": protocol_json,
            "processes": 31,
            "process_cap": 32,
            "timed_restart_concurrency": 1,
            "restart_concurrency_policy": "serialized-to-avoid-mutual-contention",
            "affinity_policy": "inherit-and-record",
            "fresh_process_per_restart": True,
            "restart_ledgers": restart_names,
            "restart_ledger_sha256": {
                f"batch-{batch}.restart-{restart:02d}.json": "00" * 32
                for batch in protocol.batch_sizes
                for restart in range(protocol.restarts)
            },
        },
    )
    # The full scientific validator is exercised elsewhere against frozen
    # repository state; isolate this suite-level invariant in the unit test.
    validator_calls: list[dict[str, Any]] = []

    def fake_validator(*args: Any, **kwargs: Any) -> str:
        validator_calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(
        latency_analysis,
        "validate_experiment_protocol",
        fake_validator,
    )
    with pytest.raises(ValueError, match="process count differs"):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=cell_id,
            scientific=True,
        )
    assert validator_calls == [
        {"phase": "confirm", "scientific": True, "processes": 32}
    ]


def test_scientific_analysis_rejects_draft_or_nonconfirm_manifest(
    tmp_path: Path,
) -> None:
    output, manifest, _ = _smoke_inputs(tmp_path)
    with pytest.raises(ValueError):
        analyze_latency_suite(
            output,
            manifest=manifest,
            cell_id=manifest["cells"][0]["cell_id"],
            scientific=True,
        )
