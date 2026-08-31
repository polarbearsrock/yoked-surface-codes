from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding import _pinball_promatch_matched_latency_integration as integration
from yoked.decoding._patch_uf_latency import write_authenticated_latency_corpus
from yoked.decoding._pinball_promatch_experiment import (
    PINBALL_CONFIG,
    PROMATCH_CONFIG,
)
from yoked.decoding._pinball_promatch_matched_latency import (
    VARIANT_NAMES,
    load_authenticated_detector_corpus,
)
from yoked.decoding._pinball_promatch_matched_latency_integration import (
    CompiledDirectPyMatchingDecoder,
    YokedMatchedLatencyFactory,
    latency_protocol_from_matched_protocol,
    materialize_detector_corpus_from_patch_uf,
)


_HASHES = {
    "circuit_sha256": "1" * 64,
    "dem_sha256": "2" * 64,
    "promatch_layout_fingerprint": "3" * 64,
    "promatch_graph_fingerprint": "4" * 64,
    "pinball_layout_fingerprint": "5" * 64,
    "pinball_graph_fingerprint": "6" * 64,
    "pinball_schedule_fingerprint": "7" * 64,
}


def _source_manifest(tmp_path: Path, *, rows: int = 8) -> Path:
    detectors = np.arange(rows, dtype=np.uint8).reshape(rows, 1) & 0b111
    residuals = detectors ^ 1
    summaries = tuple(
        {"global_shot_id": index, "residual_detector_count": 0}
        for index in range(rows)
    )
    return write_authenticated_latency_corpus(
        tmp_path / "uf-source",
        detectors=detectors,
        residuals=residuals,
        num_detectors=3,
        global_shot_ids=tuple(range(rows)),
        summaries=summaries,
        provenance={
            "schema": "patch-uf-latency-materialization-v1",
            "cell_id": "toy-d1",
            "circuit_sha256": _HASHES["circuit_sha256"],
            "dem_sha256": _HASHES["dem_sha256"],
            "num_detectors": 3,
            "num_observables": 3,
        },
    )


def _materialized(tmp_path: Path) -> Path:
    return materialize_detector_corpus_from_patch_uf(
        _source_manifest(tmp_path),
        tmp_path / "matched-corpus",
    )


def _protocol(corpus_manifest: Path) -> dict[str, object]:
    corpus = load_authenticated_detector_corpus(corpus_manifest)
    imported = dict(corpus.provenance)
    return {
        "experiment_id": "8" * 64,
        "cell": {
            "cell_id": "toy-d1",
            "d": 1,
            "r": 2,
            "p": 0.003,
            "patches": 1,
            "yokes": 0,
            **_HASHES,
        },
        "promatch_config": copy.deepcopy(PROMATCH_CONFIG),
        "pinball_config": copy.deepcopy(PINBALL_CONFIG),
        "dem_options": {
            "decompose_errors": True,
            "approximate_disjoint_errors": True,
        },
        "corpus": {
            "manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "source_patch_uf_manifest_sha256": imported[
                "source_patch_uf_manifest_sha256"
            ],
            "source_patch_uf_corpus_digest": imported[
                "source_patch_uf_corpus_digest"
            ],
            "source_detector_array_digest": imported[
                "source_detector_array_digest"
            ],
        },
        "latency": {
            "batches": [
                {
                    "batch_size": 1,
                    "restarts": 10,
                    "blocks_per_restart": 20,
                    "warmup_calls_per_variant": 50,
                    "timed_calls_per_side_per_block": 10,
                },
                {
                    "batch_size": 64,
                    "restarts": 5,
                    "blocks_per_restart": 4,
                    "warmup_calls_per_variant": 5,
                    "timed_calls_per_side_per_block": 2,
                },
                {
                    "batch_size": 1024,
                    "restarts": 3,
                    "blocks_per_restart": 2,
                    "warmup_calls_per_variant": 1,
                    "timed_calls_per_side_per_block": 1,
                },
            ],
            "schedule_seed": "9" * 64,
        },
    }


class _FakeMatcher:
    def __init__(self, output: int = 0xFF) -> None:
        self.output = output
        self.calls: list[tuple[np.ndarray, bool, bool]] = []

    def decode_batch(
        self,
        packed: np.ndarray,
        *,
        bit_packed_shots: bool,
        bit_packed_predictions: bool,
    ) -> np.ndarray:
        self.calls.append(
            (np.array(packed, copy=True), bit_packed_shots, bit_packed_predictions)
        )
        return np.full((len(packed), 1), self.output, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class _FakePackedDecoder:
    tag: int

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        result = np.array(bit_packed_detection_event_data[:, :1], copy=True)
        result ^= self.tag
        return result


def _fake_prepared(*, wrong_graph: bool = False) -> SimpleNamespace:
    provenance = {
        **_HASHES,
        "promatch_graph_fingerprint": (
            "a" * 64 if wrong_graph else _HASHES["promatch_graph_fingerprint"]
        ),
        "num_detectors": 3,
        "num_observables": 3,
        "arms": {"fixture": "toy"},
    }
    return SimpleNamespace(
        provenance=provenance,
        matcher_u0=_FakeMatcher(output=0),
        compiled_promatch=_FakePackedDecoder(1),
        compiled_pinball=_FakePackedDecoder(2),
    )


def test_imports_exact_detector_array_and_excludes_source_residuals(
    tmp_path: Path,
) -> None:
    source_path = _source_manifest(tmp_path)
    source_json = json.loads(source_path.read_text())
    target_path = materialize_detector_corpus_from_patch_uf(
        source_path,
        tmp_path / "target",
        expected_source_identity={
            "manifest_sha256": source_json["manifest_sha256"],
            "corpus_digest": source_json["corpus_digest"],
            "num_detectors": 3,
            "row_count": 8,
        },
    )
    target = load_authenticated_detector_corpus(target_path)
    source_detectors = np.load(source_path.parent / "detectors.npy", allow_pickle=False)

    assert np.array_equal(target.detectors, source_detectors)
    assert target.global_shot_ids == tuple(range(8))
    assert not hasattr(target, "residuals")
    assert not hasattr(target, "observables")
    assert set(json.loads(target_path.read_text())) == {
        "schema",
        "num_detectors",
        "row_count",
        "corpus_digest",
        "detectors",
        "global_shot_ids",
        "provenance",
        "manifest_sha256",
    }
    with pytest.raises(ValueError, match="differs"):
        materialize_detector_corpus_from_patch_uf(
            source_path,
            tmp_path / "bad-target",
            expected_source_identity={"corpus_digest": "f" * 64},
        )


def test_import_rejects_observable_bearing_source_manifest(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path)
    value = json.loads(source.read_text())
    value["observables"] = {"path": "forbidden.bitpack", "sha256": "0" * 64}
    source.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="fields are malformed"):
        materialize_detector_corpus_from_patch_uf(
            source,
            tmp_path / "target",
        )


def test_direct_global_adapter_validates_input_and_masks_output_tail() -> None:
    matcher = _FakeMatcher()
    decoder = CompiledDirectPyMatchingDecoder(
        matcher=matcher,
        num_detectors=3,
        num_observables=3,
    )
    packed = np.array([[1], [2]], dtype=np.uint8)

    prediction = decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )

    assert prediction.tolist() == [[0b111], [0b111]]
    assert matcher.calls[0][1:] == (True, True)
    with pytest.raises(ValueError, match="tail bits"):
        decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.array([[0b1000]], dtype=np.uint8)
        )


def test_factory_preloads_once_and_binds_production_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_manifest = _materialized(tmp_path)
    protocol = _protocol(corpus_manifest)
    calls: list[dict[str, object]] = []

    def fake_prepare_cell(cell: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"cell": cell, **kwargs})
        return _fake_prepared()

    monkeypatch.setattr(integration, "prepare_cell", fake_prepare_cell)
    factory = YokedMatchedLatencyFactory.from_protocol(
        protocol,
        corpus_manifest_path=corpus_manifest,
    )

    identity_first = factory.suite_identity
    identity_second = factory.suite_identity
    assert identity_first == identity_second
    assert calls == []
    pickle.dumps(factory)

    workload = factory.preload()
    assert len(calls) == 1
    assert calls[0]["verify_hashes"] is True
    assert calls[0]["promatch_config"] == PROMATCH_CONFIG
    assert calls[0]["pinball_config"] == PINBALL_CONFIG
    assert tuple(variant.name for variant in workload.variants) == VARIANT_NAMES
    packed = workload.corpus.detectors[:2]
    predictions = [variant.function(packed) for variant in workload.variants]
    assert not np.array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[1], predictions[2])
    assert workload.provenance["prepared"]["circuit_sha256"] == "1" * 64
    assert workload.provenance["corpus_import"]["schema"] == integration.IMPORT_SCHEMA


def test_factory_rejects_prepared_fingerprint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_manifest = _materialized(tmp_path)
    factory = YokedMatchedLatencyFactory.from_protocol(
        _protocol(corpus_manifest),
        corpus_manifest_path=corpus_manifest,
    )
    monkeypatch.setattr(
        integration,
        "prepare_cell",
        lambda *args, **kwargs: _fake_prepared(wrong_graph=True),
    )

    with pytest.raises(ValueError, match="promatch_graph_fingerprint"):
        factory.preload()


def test_maps_exact_uf_batch_literals_to_latency_protocol(tmp_path: Path) -> None:
    protocol = _protocol(_materialized(tmp_path))
    result = latency_protocol_from_matched_protocol(protocol)

    assert [batch.to_json() for batch in result.batches] == protocol["latency"][
        "batches"
    ]
    assert result.schedule_seed == int("9" * 64, 16)
    assert len(result.host_policy.cpu_affinity) == 1

    protocol["latency"]["host_policy"] = result.host_policy.to_json()
    frozen_host = latency_protocol_from_matched_protocol(
        protocol,
        cpu=result.host_policy.cpu_affinity[0],
    )
    assert frozen_host.host_policy == result.host_policy

    bad = copy.deepcopy(protocol)
    del bad["latency"]["batches"][0]["warmup_calls_per_variant"]
    with pytest.raises(ValueError, match="batch fields"):
        latency_protocol_from_matched_protocol(bad)
