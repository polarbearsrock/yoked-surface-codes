from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding import _pinball_promatch_matched_latency as latency
from yoked.decoding._pinball_promatch_matched_latency import (
    BatchTiming,
    FIXED_PAIRS,
    LatencyProtocol,
    LatencyRestartTask,
    LatencyWorkload,
    TimedVariant,
    VARIANT_NAMES,
    build_timed_variants,
    capture_host_policy,
    load_authenticated_detector_corpus,
    run_latency_restart,
    run_latency_suite,
    validate_restart_record,
    write_authenticated_detector_corpus,
)


def _corpus_manifest(tmp_path: Path, *, rows: int = 8) -> Path:
    detectors = (np.arange(rows, dtype=np.uint8).reshape(rows, 1) & 0b111)
    return write_authenticated_detector_corpus(
        tmp_path / "corpus",
        detectors=detectors,
        num_detectors=3,
        global_shot_ids=tuple(100 + index for index in range(rows)),
        provenance={"fixture": "matched"},
    )


class _RecordingCall:
    def __init__(self, name: str, events: list[tuple[str, bool, int]]) -> None:
        self.name = name
        self.events = events

    def __call__(self, packed: np.ndarray) -> np.ndarray:
        self.events.append((self.name, gc.isenabled(), len(packed)))
        return np.array(packed[:, :1], dtype=np.uint8, copy=True)


class _RecordingFactory:
    def __init__(self, manifest: str) -> None:
        self.manifest = manifest
        corpus = load_authenticated_detector_corpus(manifest)
        self.suite_identity = {
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "fixture": "recording",
        }
        self.events: list[tuple[str, bool, int]] = []

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        del restart_index, batch_size
        corpus = load_authenticated_detector_corpus(self.manifest)
        return LatencyWorkload(
            corpus=corpus,
            variants=tuple(
                TimedVariant(name, _RecordingCall(name, self.events))
                for name in VARIANT_NAMES
            ),
            provenance={"factory": "recording", "pid": os.getpid()},
        )


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 10
        return self.value


class _PackedDecoder:
    def __init__(self, tag: int = 0) -> None:
        self.tag = tag

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        result = np.array(bit_packed_detection_event_data[:, :1], copy=True)
        result ^= self.tag
        return result


class _SpawnFactory:
    def __init__(self, manifest: str) -> None:
        self.manifest = manifest
        corpus = load_authenticated_detector_corpus(manifest)
        self.suite_identity = {
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "fixture": "spawn",
        }
        self.preload_calls = 0

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        corpus = load_authenticated_detector_corpus(self.manifest)
        return LatencyWorkload(
            corpus=corpus,
            variants=build_timed_variants(
                global_mwpm=_PackedDecoder(),
                promatch=_PackedDecoder(1),
                pinball=_PackedDecoder(2),
            ),
            provenance={
                "factory": "spawn",
                "pid": os.getpid(),
                "restart_index": restart_index,
                "batch_size": batch_size,
            },
        )

    def preload(self) -> LatencyWorkload:
        self.preload_calls += 1
        return self(0, 1)


def _protocol(*, batch_size: int = 1, restarts: int = 1) -> LatencyProtocol:
    return LatencyProtocol(
        batches=(
            BatchTiming(
                batch_size=batch_size,
                restarts=restarts,
                blocks_per_restart=2,
                warmup_calls_per_variant=1,
                timed_calls_per_side_per_block=1,
            ),
        ),
        schedule_seed=12345,
        host_policy=capture_host_policy(),
    )


def _task(factory: object, protocol: LatencyProtocol) -> LatencyRestartTask:
    identity, protocol_id, workload_id, suite_id = latency._task_ids(factory, protocol)
    return LatencyRestartTask(
        factory=factory,
        protocol=protocol,
        restart_index=0,
        batch_size=protocol.batches[0].batch_size,
        protocol_id=protocol_id,
        suite_id=suite_id,
        workload_id=workload_id,
        workload_identity=identity,
    )


def test_detector_only_corpus_round_trip_is_immutable(tmp_path: Path) -> None:
    manifest = _corpus_manifest(tmp_path)
    corpus = load_authenticated_detector_corpus(manifest)

    assert corpus.detectors.shape == (8, 1)
    assert corpus.global_shot_ids == tuple(range(100, 108))
    assert not corpus.detectors.flags.writeable
    assert not hasattr(corpus, "observables")
    assert not hasattr(corpus, "residuals")
    with pytest.raises(ValueError):
        corpus.detectors[0, 0] = 4


def test_detector_corpus_rejects_manifest_and_payload_tampering(tmp_path: Path) -> None:
    manifest = _corpus_manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["observables"] = {"path": "forbidden.npy", "sha256": "0" * 64}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fields are malformed"):
        load_authenticated_detector_corpus(manifest)

    second = _corpus_manifest(tmp_path / "second")
    detector_path = second.parent / "detectors.npy"
    payload = bytearray(detector_path.read_bytes())
    payload[-1] ^= 1
    detector_path.write_bytes(payload)
    with pytest.raises(ValueError, match="artifact digest"):
        load_authenticated_detector_corpus(second)


def test_packed_variant_builder_uses_public_production_entrypoint() -> None:
    variants = build_timed_variants(
        global_mwpm=_PackedDecoder(1),
        promatch=_PackedDecoder(2),
        pinball=_PackedDecoder(3),
    )
    packed = np.array([[7]], dtype=np.uint8)

    assert tuple(variant.name for variant in variants) == VARIANT_NAMES
    assert [int(variant.function(packed)[0, 0]) for variant in variants] == [6, 5, 4]
    with pytest.raises(TypeError, match="packed production decode"):
        build_timed_variants(
            global_mwpm=object(), promatch=_PackedDecoder(), pinball=_PackedDecoder()
        )


def test_fake_clock_proves_pairing_warmup_gc_and_configurable_batch(tmp_path: Path) -> None:
    factory = _RecordingFactory(str(_corpus_manifest(tmp_path)))
    protocol = _protocol(batch_size=2)
    record = run_latency_restart(_task(factory, protocol), clock=_FakeClock())

    assert record["clock"] == "test-clock"
    assert record["warmup"]["calls_per_variant"] == 1
    assert set(record["warmup"]["variant_order"]) == set(VARIANT_NAMES)
    assert set(record["pairs"]) == {pair.name for pair in FIXED_PAIRS}
    # The first six calls are two deterministic preflight calls per variant. GC is
    # disabled for every subsequent warmup and measured call.
    assert [name for name, _, _ in factory.events[:6]] == [
        name for name in VARIANT_NAMES for _ in range(2)
    ]
    assert all(not gc_enabled for _, gc_enabled, _ in factory.events[6:])
    assert all(rows == 1 for _, _, rows in factory.events[:6])
    assert all(rows == 2 for _, _, rows in factory.events[6:])
    for pair in record["pairs"].values():
        assert sorted(pair["order_by_block"]) == ["AB", "BA"]
        for field in ("numerator_calls", "denominator_calls"):
            assert [block[0]["duration_ns"] for block in pair[field]] == [10, 10]
        for numerator, denominator in zip(
            pair["numerator_calls"], pair["denominator_calls"]
        ):
            assert numerator[0]["corpus_indices"] == denominator[0]["corpus_indices"]
            assert numerator[0]["detector_batch_digest"] == denominator[0]["detector_batch_digest"]


def test_restart_validation_rejects_unbalanced_or_misaligned_record(tmp_path: Path) -> None:
    factory = _RecordingFactory(str(_corpus_manifest(tmp_path)))
    protocol = _protocol()
    task = _task(factory, protocol)
    record = run_latency_restart(task, clock=_FakeClock())
    record["clock"] = "time.perf_counter_ns"
    validate_restart_record(record, task=task)

    pair = record["pairs"]["promatch_vs_global"]
    pair["order_by_block"] = ["AB", "AB"]
    with pytest.raises(ValueError, match="not balanced"):
        validate_restart_record(record, task=task)


def test_suite_uses_fresh_spawn_and_resumes_immutable_ledgers(tmp_path: Path) -> None:
    factory = _SpawnFactory(str(_corpus_manifest(tmp_path)))
    protocol = _protocol()
    output = tmp_path / "latency"
    parent_pid = os.getpid()

    suite = run_latency_suite(factory, protocol=protocol, out_dir=output)
    ledger_path = output / suite["restart_ledgers"][0]
    ledger = json.loads(ledger_path.read_text())

    assert suite["fresh_process_per_restart"] is True
    assert suite["execution_mode"] == "spawn-factory"
    assert suite["process_start_method"] == "spawn"
    assert suite["timed_restart_concurrency"] == 1
    assert suite["native_threads"] == 1
    assert ledger["provenance"]["workload"]["pid"] != parent_pid
    assert ledger["untimed_prediction_check"]["deterministic_repeats_per_variant"] == 2
    assert ledger["timing_scope"]["decoder_compilation_inside_timing"] is False
    assert all(
        value == "1"
        for value in ledger["provenance"]["runtime_start"][
            "native_thread_environment"
        ].values()
    )

    original = ledger_path.read_bytes()
    assert run_latency_suite(factory, protocol=protocol, out_dir=output) == suite
    assert ledger_path.read_bytes() == original


def test_fork_preloaded_compiles_once_and_uses_fresh_cow_child(tmp_path: Path) -> None:
    factory = _SpawnFactory(str(_corpus_manifest(tmp_path)))
    protocol = _protocol(restarts=2)
    output = tmp_path / "fork-latency"
    parent_pid = os.getpid()

    suite = run_latency_suite(
        factory,
        protocol=protocol,
        out_dir=output,
        execution_mode="fork-preloaded",
    )
    ledgers = [
        json.loads((output / name).read_text()) for name in suite["restart_ledgers"]
    ]
    ledger = ledgers[0]

    assert factory.preload_calls == 1
    assert suite["execution_mode"] == "fork-preloaded"
    assert suite["process_start_method"] == "fork"
    assert suite["parent_preload_once"] is True
    assert ledger["execution_mode"] == "fork-preloaded"
    assert ledger["process_start_method"] == "fork"
    assert ledger["provenance"]["workload"]["pid"] == parent_pid
    assert ledger["provenance"]["runtime_start"]["pid"] != parent_pid
    assert len({row["provenance"]["runtime_start"]["pid"] for row in ledgers}) == 2

    # A fully complete resume verifies ledgers without compiling again.
    resumed_factory = _SpawnFactory(factory.manifest)
    assert run_latency_suite(
        resumed_factory,
        protocol=protocol,
        out_dir=output,
        execution_mode="fork-preloaded",
    ) == suite
    assert resumed_factory.preload_calls == 0


def test_nonresume_rejects_existing_restart_artifacts(tmp_path: Path) -> None:
    factory = _SpawnFactory(str(_corpus_manifest(tmp_path)))
    protocol = _protocol()
    output = tmp_path / "latency"
    run_latency_suite(factory, protocol=protocol, out_dir=output)

    with pytest.raises(FileExistsError, match="already contains"):
        run_latency_suite(factory, protocol=protocol, out_dir=output, resume=False)
