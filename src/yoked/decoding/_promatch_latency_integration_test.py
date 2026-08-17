from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from yoked.decoding._promatch_decoder import CompiledPromatchDecoder
from yoked.decoding._promatch_experiment import (
    default_smoke_protocol,
    normalize_protocol,
)
from yoked.decoding._promatch_latency_integration import (
    TinyLatencySmokeConfig,
    YokedPromatchLatencyFactory,
    latency_protocol_from_manifest,
)


def _documented_protocol(name: str) -> dict[str, object]:
    path = Path(__file__).resolve().parents[3] / "docs" / name
    return normalize_protocol(json.loads(path.read_text(encoding="utf-8")))


def _tiny_manifest() -> dict[str, object]:
    manifest = default_smoke_protocol(processes=1, shots=4)
    cell = manifest["cells"][0]
    cell.update(
        cell_id="latency-smoke-d3-n2-y2-r3-p0.01",
        d=3,
        r=3,
        p=0.01,
        patches=2,
        yokes=2,
    )
    manifest["cells"] = [cell]
    return manifest


def _smoke(*, corpus_batches: int = 2) -> TinyLatencySmokeConfig:
    return TinyLatencySmokeConfig(
        restarts=1,
        blocks_per_restart=2,
        calls_per_block=1,
        warmup_calls_per_variant=1,
        batch_sizes=(1, 2),
        schedule_seed=12345,
        corpus_batches=corpus_batches,
        timing_corpus_seed_root="91" * 32,
    )


def test_scientific_protocol_is_extracted_exactly_from_normalized_manifest() -> None:
    manifest = _documented_protocol("PROMATCH_FIRST_ROUND_PROTOCOL.json")
    protocol = latency_protocol_from_manifest(manifest, scientific=True)

    assert protocol.restarts == 10
    assert protocol.blocks_per_restart == 100
    assert protocol.calls_per_block == 100
    assert protocol.warmup_calls_per_variant == 1_000
    assert protocol.batch_sizes == (1, 64, 1024)
    assert protocol.schedule_seed == int(
        manifest["sampler_seed_roots"]["timing_bootstrap"], 16
    )
    protocol.validate(scientific=True)

    changed = json.loads(json.dumps(manifest))
    changed["analysis_config"]["timing_protocol"]["claim_gates"][
        "total_p99_ratio_upper"
    ] = 1.06
    with pytest.raises(ValueError, match="exact claim_gates"):
        latency_protocol_from_manifest(changed, scientific=True)

    changed = json.loads(json.dumps(manifest))
    changed["analysis_config"]["timing_protocol"]["diagnostic_intervals"] = [
        "total"
    ]
    with pytest.raises(ValueError, match="diagnostic_intervals"):
        latency_protocol_from_manifest(changed, scientific=True)

    changed = json.loads(json.dumps(manifest))
    changed["analysis_config"]["timing_protocol"]["primary_interval"] = (
        "implementation-defined"
    )
    with pytest.raises(ValueError, match="frozen adapter interval"):
        latency_protocol_from_manifest(changed, scientific=True)


def test_tiny_protocol_requires_explicit_nonclaim_configuration() -> None:
    manifest = _tiny_manifest()
    smoke = _smoke()
    protocol = latency_protocol_from_manifest(
        manifest,
        scientific=False,
        smoke=smoke,
    )
    assert protocol.batch_sizes == (1, 2)
    assert protocol.restarts == 1
    protocol.validate(scientific=False)

    with pytest.raises(ValueError, match="explicit TinyLatencySmokeConfig"):
        latency_protocol_from_manifest(manifest, scientific=False)
    with pytest.raises(ValueError, match="cannot be claim-bearing"):
        latency_protocol_from_manifest(manifest, scientific=True, smoke=smoke)


def test_real_tiny_factory_is_picklable_and_corpora_are_deterministic() -> None:
    manifest = _tiny_manifest()
    smoke = _smoke(corpus_batches=3)
    cell_id = manifest["cells"][0]["cell_id"]
    factory = YokedPromatchLatencyFactory.from_manifest(
        manifest,
        cell_id=cell_id,
        scientific=False,
        smoke=smoke,
    )
    factory = pickle.loads(pickle.dumps(factory))

    first = factory(0, 2)
    second = factory(0, 2)
    assert first.total_corpus.shape[0] == 6
    assert first.total_corpus.dtype == np.uint8
    assert not first.total_corpus.flags.writeable
    assert not first.backend_residual_corpus.flags.writeable
    np.testing.assert_array_equal(first.total_corpus, second.total_corpus)
    np.testing.assert_array_equal(
        first.backend_residual_corpus,
        second.backend_residual_corpus,
    )
    assert first.provenance["stim_seed"] == second.provenance["stim_seed"]
    assert first.provenance["correlated_matching"] is False
    assert first.provenance["u0_backend"] == "pymatching-uncorrelated"
    assert first.provenance["residual_backend"] == "pymatching-uncorrelated"


def test_real_tiny_callables_have_packed_shapes_and_wrap_equals_direct() -> None:
    manifest = _tiny_manifest()
    smoke = _smoke(corpus_batches=2)
    factory = YokedPromatchLatencyFactory.from_manifest(
        manifest,
        cell_id=manifest["cells"][0]["cell_id"],
        scientific=False,
        smoke=smoke,
    )
    workload = factory(0, 2)
    packed = workload.total_corpus
    expected = (packed.shape[0], 1)  # Four logical observables for two patches.

    direct = workload.u0_direct(packed)
    wrapped = workload.u0_wrap(packed)
    treatment = workload.pu_window(packed)
    backend_original = workload.backend_original(workload.backend_original_corpus)
    backend_residual = workload.backend_residual(workload.backend_residual_corpus)
    assert direct.shape == expected
    assert wrapped.shape == expected
    assert treatment.shape == expected
    assert backend_original.shape == expected
    assert backend_residual.shape == expected
    assert all(
        value.dtype == np.uint8
        for value in (
            direct,
            wrapped,
            treatment,
            backend_original,
            backend_residual,
        )
    )
    np.testing.assert_array_equal(wrapped, direct)
    np.testing.assert_array_equal(backend_original, direct)


def test_residual_corpus_uses_only_the_nonretaining_predecode_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _tiny_manifest()
    smoke = _smoke(corpus_batches=1)
    observed_retain_flags: list[bool] = []
    original = CompiledPromatchDecoder._predecode_shots

    def recording_predecode(
        self: CompiledPromatchDecoder,
        unpacked_detection_events: np.ndarray,
        *,
        retain_results: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[object, ...]]:
        observed_retain_flags.append(retain_results)
        return original(
            self,
            unpacked_detection_events,
            retain_results=retain_results,
        )

    monkeypatch.setattr(
        CompiledPromatchDecoder,
        "_predecode_shots",
        recording_predecode,
    )
    factory = YokedPromatchLatencyFactory.from_manifest(
        manifest,
        cell_id=manifest["cells"][0]["cell_id"],
        scientific=False,
        smoke=smoke,
    )
    workload = factory(0, 1)

    assert observed_retain_flags
    assert not any(observed_retain_flags)
    assert (
        workload.provenance["residual_generation_retained_shot_telemetry"]
        is False
    )
