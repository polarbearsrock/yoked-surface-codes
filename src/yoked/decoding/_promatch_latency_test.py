from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import yoked.decoding._promatch_latency as latency
from yoked.decoding._promatch_experiment import default_smoke_protocol
from yoked.decoding._promatch_latency import (
    BACKEND_RESIDUAL_VS_ORIGINAL,
    TOTAL_PU_VS_DIRECT,
    TOTAL_PU_VS_WRAP,
    LatencyProtocol,
    LatencyWorkload,
    balanced_pair_orders,
    run_latency_benchmark,
    run_latency_restart,
    write_restart_ledger_atomic,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


class _ManualClock:
    def __init__(self) -> None:
        self.now = 10_000

    def __call__(self) -> int:
        return self.now


class _SyntheticFactory:
    def __init__(self, clock: _ManualClock | None = None) -> None:
        self.clock = clock
        self.log: list[tuple[str, np.ndarray]] = []

    def _call(self, name: str, duration: int):
        def invoke(batch: np.ndarray) -> np.ndarray:
            assert not batch.flags.writeable
            self.log.append((name, batch.copy()))
            if self.clock is not None:
                self.clock.now += duration
            return np.zeros((len(batch), 1), dtype=np.uint8)

        return invoke

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        del restart_index, batch_size
        total = np.arange(24, dtype=np.uint8).reshape(8, 3)
        original = np.arange(24, 48, dtype=np.uint8).reshape(8, 3)
        residual = np.arange(48, 72, dtype=np.uint8).reshape(8, 3)
        return LatencyWorkload(
            total_corpus=total,
            u0_direct=self._call("u0_direct", 10),
            u0_wrap=self._call("u0_wrap", 20),
            pu_window=self._call("pu_window", 5),
            backend_original_corpus=original,
            backend_residual_corpus=residual,
            backend_original=self._call("backend_original", 8),
            backend_residual=self._call("backend_residual", 3),
            provenance={"test": "synthetic", "graph_fingerprint": "abc"},
        )


def _small_protocol(**changes: Any) -> LatencyProtocol:
    values = {
        "restarts": 1,
        "blocks_per_restart": 2,
        "calls_per_block": 2,
        "warmup_calls_per_variant": 1,
        "batch_sizes": (2,),
        "schedule_seed": 1234,
    }
    values.update(changes)
    return LatencyProtocol(**values)


def test_balanced_pair_orders_are_exact_and_deterministic() -> None:
    first = balanced_pair_orders(blocks=100, seed=123, pair_name="pair")
    second = balanced_pair_orders(blocks=100, seed=123, pair_name="pair")
    assert first == second
    assert first.count("AB") == 50
    assert first.count("BA") == 50
    assert balanced_pair_orders(blocks=100, seed=124, pair_name="pair") != first
    with pytest.raises(ValueError, match="even"):
        balanced_pair_orders(blocks=3, seed=1, pair_name="pair")


@pytest.mark.parametrize(
    "protocol,match",
    [
        (LatencyProtocol(restarts=9), "restarts=10"),
        (LatencyProtocol(blocks_per_restart=98), "blocks_per_restart=100"),
        (LatencyProtocol(calls_per_block=99), "calls_per_block=100"),
        (LatencyProtocol(warmup_calls_per_variant=999), "warmup_calls_per_variant=1000"),
        (LatencyProtocol(batch_sizes=(64,)), "batch size 1"),
        (LatencyProtocol(batch_sizes=(1, 2)), "only batch sizes"),
    ],
)
def test_scientific_mode_enforces_frozen_primary(protocol: LatencyProtocol, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        protocol.validate(scientific=True)


def test_nonclaim_mode_explicitly_allows_small_counts() -> None:
    protocol = _small_protocol()
    protocol.validate(scientific=False)
    assert protocol.to_json(scientific=False)["scientific"] is False
    with pytest.raises(ValueError, match="even"):
        replace(protocol, blocks_per_restart=1).validate(scientific=False)


def test_restart_collects_direct_raw_calls_balanced_blocks_and_digests() -> None:
    clock = _ManualClock()
    factory = _SyntheticFactory(clock)
    record = run_latency_restart(
        factory,
        protocol=_small_protocol(),
        restart_index=0,
        batch_size=2,
        scientific=False,
        clock=clock,
    )

    assert record["schema"] == latency.LATENCY_RESTART_SCHEMA
    assert record["claim_bearing"] is False
    assert record["timing_scope"]["input_generation_inside_timing"] is False
    assert record["timing_scope"]["telemetry_retained_inside_total"] is False
    assert set(record["pairs"]) == {
        TOTAL_PU_VS_DIRECT,
        TOTAL_PU_VS_WRAP,
        BACKEND_RESIDUAL_VS_ORIGINAL,
    }

    direct = record["pairs"][TOTAL_PU_VS_DIRECT]
    wrap = record["pairs"][TOTAL_PU_VS_WRAP]
    backend = record["pairs"][BACKEND_RESIDUAL_VS_ORIGINAL]
    assert direct["numerator_calls_ns"] == [[5, 5], [5, 5]]
    assert direct["denominator_calls_ns"] == [[10, 10], [10, 10]]
    assert direct["numerator_block_totals_ns"] == [10, 10]
    assert direct["denominator_block_totals_ns"] == [20, 20]
    assert wrap["numerator_calls_ns"] == [[5, 5], [5, 5]]
    assert wrap["denominator_calls_ns"] == [[20, 20], [20, 20]]
    assert backend["numerator_calls_ns"] == [[3, 3], [3, 3]]
    assert backend["denominator_calls_ns"] == [[8, 8], [8, 8]]
    for pair in record["pairs"].values():
        assert sorted(pair["order_by_block"]) == ["AB", "BA"]

    # One warmup/variant; PU is then timed against both controls.
    counts = {name: sum(logged_name == name for logged_name, _ in factory.log) for name in {
        "u0_direct", "u0_wrap", "pu_window", "backend_original", "backend_residual"
    }}
    assert counts == {
        "u0_direct": 5,
        "u0_wrap": 5,
        "pu_window": 9,
        "backend_original": 5,
        "backend_residual": 5,
    }
    assert record["corpus"]["total"]["shape"] == [8, 3]
    assert len(record["corpus"]["total"]["sha256"]) == 64
    assert all(
        value == "1"
        for value in record["provenance"]["runtime"]["native_thread_environment"].values()
    )
    # The result is strict portable JSON, including all raw calls.
    json.dumps(record, allow_nan=False)


def test_backend_corpora_must_be_paired() -> None:
    protocol = _small_protocol()
    clock = _ManualClock()
    factory = _SyntheticFactory(clock)
    workload = factory(0, 2)
    workload.backend_residual_corpus = workload.backend_residual_corpus[:-1]

    with pytest.raises(ValueError, match="equal shot counts"):
        run_latency_restart(
            lambda _restart, _batch: workload,
            protocol=protocol,
            restart_index=0,
            batch_size=2,
            scientific=False,
            clock=clock,
        )


def test_atomic_ledger_uses_tmpdir_and_refuses_accidental_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    destination = tmp_path / "results" / "restart.json"
    write_restart_ledger_atomic(destination, {"schema": "test", "value": 1})
    assert json.loads(destination.read_text(encoding="utf-8"))["value"] == 1
    assert list(scratch.iterdir()) == []
    with pytest.raises(FileExistsError):
        write_restart_ledger_atomic(destination, {"schema": "test", "value": 2})
    write_restart_ledger_atomic(
        destination,
        {"schema": "test", "value": 2},
        overwrite=True,
    )
    assert json.loads(destination.read_text(encoding="utf-8"))["value"] == 2


def test_orchestrator_hard_rejects_more_than_32_processes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds the frozen cap"):
        run_latency_benchmark(
            _SyntheticFactory(),
            manifest=default_smoke_protocol(processes=32, shots=1),
            protocol=_small_protocol(),
            out_dir=tmp_path / "must-not-be-created",
            processes=33,
            scientific=False,
        )
    assert not (tmp_path / "must-not-be-created").exists()


def test_orchestrator_persists_all_ledgers_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    created_executors: list[dict[str, Any]] = []

    class ImmediateFuture:
        def __init__(self, function, task) -> None:
            self.function = function
            self.task = task

        def result(self):
            return self.function(self.task)

    class ImmediateExecutor:
        def __init__(self, **kwargs) -> None:
            created_executors.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

        def submit(self, function, task):
            return ImmediateFuture(function, task)

    monkeypatch.setattr(latency, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(latency, "as_completed", lambda futures: list(futures))

    protocol = _small_protocol(restarts=2, calls_per_block=1, batch_sizes=(1,))
    manifest = default_smoke_protocol(processes=32, shots=1)
    output = tmp_path / "latency"
    suite = run_latency_benchmark(
        _SyntheticFactory(),
        manifest=manifest,
        protocol=protocol,
        out_dir=output,
        processes=32,
        scientific=False,
    )
    assert suite["processes"] == 32
    assert suite["process_cap"] == 32
    assert suite["fresh_process_per_restart"] is True
    assert suite["restart_ledgers"] == [
        "batch-1.restart-00.json",
        "batch-1.restart-01.json",
    ]
    assert created_executors[0]["max_workers"] == 1
    assert suite["timed_restart_concurrency"] == 1
    assert created_executors[0]["max_tasks_per_child"] == 1
    assert (output / "suite.json").exists()
    assert json.loads((output / "protocol.json").read_text()) == manifest
    for name in suite["restart_ledgers"]:
        ledger = json.loads((output / name).read_text(encoding="utf-8"))
        assert ledger["schema"] == latency.LATENCY_RESTART_SCHEMA
        assert suite["restart_ledger_sha256"][name] == hashlib.sha256(
            canonical_json_bytes(ledger)
        ).hexdigest()

    # A completed suite is a pure resume; no new executor is constructed.
    again = run_latency_benchmark(
        _SyntheticFactory(),
        manifest=manifest,
        protocol=protocol,
        out_dir=output,
        processes=32,
        scientific=False,
        resume=True,
    )
    assert again == suite
    assert len(created_executors) == 1


def _file_snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-protocol",
        "unexpected-analysis",
        "duplicate-protocol-key",
        "duplicate-suite-key",
        "duplicate-restart-key",
    ],
)
def test_latency_resume_preflight_rejects_without_mutation(
    tmp_path: Path, mutation: str
) -> None:
    output = tmp_path / "latency"
    output.mkdir()
    manifest = default_smoke_protocol(processes=1, shots=1)
    protocol = _small_protocol(restarts=1, calls_per_block=1, batch_sizes=(1,))
    protocol_path = output / "protocol.json"
    restart_path = output / "batch-1.restart-00.json"
    if mutation != "missing-protocol":
        protocol_path.write_text(json.dumps(manifest))
    if mutation == "unexpected-analysis":
        (output / "analysis.json").write_text("{}")
    elif mutation == "duplicate-protocol-key":
        protocol_path.write_text('{"schema":"first","schema":"second"}')
    elif mutation == "duplicate-suite-key":
        restart_path.write_text("{}")
        (output / "suite.json").write_text(
            '{"schema":"first","schema":"second"}'
        )
    elif mutation == "duplicate-restart-key":
        restart_path.write_text('{"schema":"first","schema":"second"}')
    before = _file_snapshot(output)
    with pytest.raises((ValueError, FileExistsError)):
        run_latency_benchmark(
            _SyntheticFactory(),
            manifest=manifest,
            protocol=protocol,
            out_dir=output,
            processes=1,
            scientific=False,
            resume=True,
        )
    assert _file_snapshot(output) == before
