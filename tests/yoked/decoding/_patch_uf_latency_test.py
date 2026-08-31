from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding import _patch_uf_latency as latency
from yoked.decoding._patch_uf_latency import (
    FIXED_PAIRS,
    VARIANT_NAMES,
    AuthenticatedLatencyCorpus,
    BatchTiming,
    LatencyProtocol,
    LatencyRestartTask,
    LatencyWorkload,
    TimedVariant,
    balanced_pair_orders,
    capture_host_policy,
    load_authenticated_latency_corpus,
    run_latency_restart,
    run_latency_suite,
    tiny_smoke_protocol,
    write_authenticated_latency_corpus,
)


def _corpus_dir(tmp_path: Path, *, rows: int = 6, residual_delta: int = 0) -> Path:
    detectors = np.arange(rows, dtype=np.uint8).reshape(rows, 1) & 0b111
    residuals = (detectors ^ residual_delta) & 0b111
    ids = tuple(100 + index for index in range(rows))
    summaries = tuple(
        {
            "global_shot_id": shot_id,
            "cluster_summary_complete": True,
            "residual_detector_count": int(residuals[index, 0].bit_count()),
        }
        for index, shot_id in enumerate(ids)
    )
    return write_authenticated_latency_corpus(
        tmp_path / "corpus",
        detectors=detectors,
        residuals=residuals,
        num_detectors=3,
        global_shot_ids=ids,
        summaries=summaries,
        provenance={"fixture": "tiny"},
    )


class _EventCallable:
    def __init__(
        self,
        name: str,
        events: list[str],
        row_counts: list[tuple[str, int]],
        *,
        xor: int = 0,
    ) -> None:
        self.name = name
        self.events = events
        self.row_counts = row_counts
        self.xor = xor

    def __call__(self, packed: np.ndarray) -> np.ndarray:
        self.events.append(f"call:{self.name}:gc={int(gc.isenabled())}")
        self.row_counts.append((self.name, len(packed)))
        result = np.array(packed[:, :1], dtype=np.uint8, copy=True)
        result ^= self.xor
        return result


class _SmokeFactory:
    def __init__(self, manifest: str, *, mismatch: str | None = None) -> None:
        self.manifest = manifest
        corpus = load_authenticated_latency_corpus(manifest)
        self.suite_identity = {
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "fixture": "smoke",
        }
        self.mismatch = mismatch
        self.events: list[str] = []
        self.row_counts: list[tuple[str, int]] = []

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        del restart_index, batch_size
        corpus = load_authenticated_latency_corpus(self.manifest)
        variants = []
        for name in VARIANT_NAMES:
            xor = 1 if name == self.mismatch else 0
            variants.append(
                TimedVariant(
                    name=name,
                    corpus_kind="residuals" if name == "backend_residual" else "detectors",
                    timer_scope=(
                        "matcher-only" if name.startswith("backend_") else "total-adapter"
                    ),
                    function=_EventCallable(
                        name, self.events, self.row_counts, xor=xor
                    ),
                )
            )
        return LatencyWorkload(
            corpus=corpus,
            variants=tuple(variants),
            provenance={"factory": "smoke"},
        )


class _FakeClock:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.value = 0

    def __call__(self) -> int:
        self.events.append("clock")
        self.value += 10
        return self.value


def _task(factory: _SmokeFactory, protocol: LatencyProtocol) -> LatencyRestartTask:
    identity = dict(factory.suite_identity)
    protocol_id = latency._json_digest(protocol.to_json())
    workload_id = latency._json_digest(identity)
    suite_id = latency._json_digest(
        {
            "protocol_id": protocol_id,
            "workload_id": workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
        }
    )
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


def test_generic_variants_fixed_pairs_and_protocol_have_no_promatch_counts() -> None:
    assert VARIANT_NAMES == (
        "global_mwpm",
        "adapter_control",
        "uf_shadow",
        "treatment",
        "backend_original",
        "backend_residual",
    )
    assert [(pair.numerator, pair.denominator) for pair in FIXED_PAIRS] == [
        ("treatment", "global_mwpm"),
        ("adapter_control", "global_mwpm"),
        ("uf_shadow", "adapter_control"),
        ("treatment", "uf_shadow"),
        ("backend_residual", "backend_original"),
    ]
    assert sorted(balanced_pair_orders(blocks=6, seed=2, pair_name="net_total")) == [
        "AB",
        "AB",
        "AB",
        "BA",
        "BA",
        "BA",
    ]
    with pytest.raises(ValueError, match="even"):
        BatchTiming(1, 1, 3, 1, 1)


def test_authenticated_loader_has_no_observable_slot_and_makes_arrays_read_only(
    tmp_path: Path,
) -> None:
    manifest = _corpus_dir(tmp_path)
    corpus = load_authenticated_latency_corpus(manifest)

    assert not hasattr(corpus, "actual_observables")
    assert not hasattr(corpus, "observables")
    assert corpus.detectors.shape == corpus.residuals.shape == (6, 1)
    assert not corpus.detectors.flags.writeable
    assert not corpus.residuals.flags.writeable
    assert corpus.workload_keys[0] == (corpus.corpus_digest, 100)
    with pytest.raises(ValueError):
        corpus.detectors[0, 0] = 1

    value = json.loads(manifest.read_text())
    value["actual_observables"] = {"path": "forbidden.npy", "sha256": "0" * 64}
    unsigned = dict(value)
    unsigned.pop("manifest_sha256")
    value["manifest_sha256"] = latency._json_digest(unsigned)
    manifest.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="fields are malformed"):
        load_authenticated_latency_corpus(manifest)


def test_authenticated_loader_rejects_duplicate_ids_and_tail_bits(tmp_path: Path) -> None:
    detectors = np.array([[0], [0b1000]], dtype=np.uint8)
    residuals = np.zeros((2, 1), dtype=np.uint8)
    summaries = (
        {"global_shot_id": 1},
        {"global_shot_id": 1},
    )
    with pytest.raises(ValueError, match="tail bits"):
        write_authenticated_latency_corpus(
            tmp_path / "bad",
            detectors=detectors,
            residuals=residuals,
            num_detectors=3,
            global_shot_ids=(1, 1),
            summaries=summaries,
            provenance={"fixture": "bad"},
        )


def test_fake_clock_proves_direct_timer_scope_warmup_gc_and_pair_balance(
    tmp_path: Path,
) -> None:
    factory = _SmokeFactory(str(_corpus_dir(tmp_path)))
    protocol = tiny_smoke_protocol(capture_host_policy())
    clock = _FakeClock(factory.events)

    record = run_latency_restart(_task(factory, protocol), clock=clock)

    assert record["warmup"]["calls_per_variant"] == 1
    assert set(record["warmup"]["variant_order"]) == set(VARIANT_NAMES)
    assert record["timing_scope"]["actual_observables_available"] is False
    assert record["untimed_prediction_check"]["checked_rows"] == 1
    assert 0 <= record["untimed_prediction_check"]["corpus_index"] < 6
    assert factory.row_counts[:6] == [(name, 1) for name in VARIANT_NAMES]
    assert set(record["pairs"]) == {pair.name for pair in FIXED_PAIRS}
    for pair in record["pairs"].values():
        assert sorted(pair["order_by_block"]) == ["AB", "BA"]
        for side in ("numerator_calls", "denominator_calls"):
            assert [[call["duration_ns"] for call in block] for block in pair[side]] == [
                [10],
                [10],
            ]
    # Six equality calls and six warmups occur outside every clock read.  The
    # final 20 timed calls are exactly clock -> callable -> clock.
    timed_events = factory.events[-60:]
    assert len(timed_events) == 60
    for offset in range(0, len(timed_events), 3):
        assert timed_events[offset] == "clock"
        assert timed_events[offset + 1].startswith("call:")
        assert timed_events[offset + 1].endswith("gc=0")
        assert timed_events[offset + 2] == "clock"
    assert all(event.endswith("gc=0") for event in factory.events[6:12])


def test_cyclic_plans_pair_sides_and_workload_keys_exactly(tmp_path: Path) -> None:
    factory = _SmokeFactory(str(_corpus_dir(tmp_path, rows=5)))
    host = capture_host_policy()
    protocol = LatencyProtocol(
        batches=(BatchTiming(3, 1, 2, 1, 2),),
        schedule_seed=123,
        host_policy=host,
    )

    record = run_latency_restart(_task(factory, protocol), clock=_FakeClock(factory.events))

    for pair in record["pairs"].values():
        for numerator_block, denominator_block in zip(
            pair["numerator_calls"], pair["denominator_calls"]
        ):
            for numerator, denominator in zip(numerator_block, denominator_block):
                assert numerator["corpus_indices"] == denominator["corpus_indices"]
                assert numerator["workload_keys"] == denominator["workload_keys"]
                assert numerator["precomputed_summary_digest"] == denominator[
                    "precomputed_summary_digest"
                ]
                assert len(numerator["corpus_indices"]) == 3
                assert all(0 <= index < 5 for index in numerator["corpus_indices"])
                expected_ids = [100 + index for index in numerator["corpus_indices"]]
                assert [key[1] for key in numerator["workload_keys"]] == expected_ids


def test_untimed_prediction_mismatch_rejects_before_any_clock(tmp_path: Path) -> None:
    factory = _SmokeFactory(
        str(_corpus_dir(tmp_path)), mismatch="adapter_control"
    )
    protocol = tiny_smoke_protocol(capture_host_policy())
    clock = _FakeClock(factory.events)

    with pytest.raises(ValueError, match="untimed prediction equality"):
        run_latency_restart(_task(factory, protocol), clock=clock)
    assert clock.value == 0


def test_strict_host_policy_rejects_before_factory(tmp_path: Path) -> None:
    factory = _SmokeFactory(str(_corpus_dir(tmp_path)))
    current = capture_host_policy()
    bad = latency.HostPolicy(
        cpu_affinity=current.cpu_affinity,
        expected_host=tuple(
            (key, "definitely-wrong" if key == "machine" else value)
            for key, value in current.expected_host
        ),
        expected_numa_nodes=current.expected_numa_nodes,
    )
    protocol = tiny_smoke_protocol(bad)

    with pytest.raises(RuntimeError, match="host field"):
        run_latency_restart(_task(factory, protocol), clock=_FakeClock(factory.events))
    assert factory.events == []


def test_spawned_suite_is_fresh_atomic_and_resumable(tmp_path: Path) -> None:
    factory = _SmokeFactory(str(_corpus_dir(tmp_path)))
    protocol = tiny_smoke_protocol(capture_host_policy())
    output = tmp_path / "latency"

    first = run_latency_suite(factory, protocol=protocol, out_dir=output)
    second = run_latency_suite(factory, protocol=protocol, out_dir=output)

    assert first == second
    assert first["fresh_process_per_restart"] is True
    assert first["timed_restart_concurrency"] == 1
    assert len(first["restart_ledgers"]) == 1
    restart_path = output / first["restart_ledgers"][0]
    restart = json.loads(restart_path.read_text())
    assert restart["provenance"]["runtime_start"]["pid"] != os.getpid()
    before = restart_path.read_bytes()
    run_latency_suite(factory, protocol=protocol, out_dir=output)
    assert restart_path.read_bytes() == before
