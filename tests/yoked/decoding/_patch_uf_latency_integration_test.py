from __future__ import annotations

import dataclasses
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding import _patch_uf_latency_integration as integration
from yoked.decoding._patch_uf_experiment import (
    CHARACTERIZATION_STAGE,
    PROTOCOL_SCHEMA,
    SEED_DERIVATION,
    PreparedCell,
    VerifiedCollection,
    canonical_protocol_self_sha256,
    prepare_selected_cell,
)
from yoked.decoding._patch_uf_latency import (
    VARIANT_NAMES,
    load_authenticated_latency_corpus,
    write_authenticated_latency_corpus,
)
from yoked.decoding._patch_uf_reference import BudgetLimits


def _protocol() -> dict[str, object]:
    limits = {field.name: None for field in dataclasses.fields(BudgetLimits)}
    value: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "status": "DRAFT",
        "frozen": False,
        "experiment_id": "11" * 32,
        "source_identity": {"fixture": "d3-latency-integration"},
        "selected_cell": {
            "cell_id": "latency-smoke-d3-n6-y2-r3-p0.001",
            "d": 3,
            "r": 3,
            "p": 0.001,
            "patches": 6,
            "yokes": 2,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
        },
        "sampling": {
            "range_count": 32,
            "seed_derivation": SEED_DERIVATION,
            "stages": {
                CHARACTERIZATION_STAGE: {"shots": 32, "seed_root": "22" * 32}
            },
        },
        "dem_options": {
            "decompose_errors": True,
            "approximate_disjoint_errors": True,
        },
        "decoder": {
            "policy": {
                "tau": "0",
                "semantic_limits": limits,
                "production_limits": limits,
            }
        },
        "collection_limits": {
            "expected_lanes_per_shot": 12,
            "maximum_component_records_per_shot": 4096,
            "maximum_metric_bytes_per_range": 1_000_000,
        },
        "latency": {
            "batches": [
                {
                    "batch_size": 1,
                    "restarts": 1,
                    "blocks_per_restart": 2,
                    "warmup_calls_per_variant": 1,
                    "timed_calls_per_side_per_block": 1,
                },
                {
                    "batch_size": 2,
                    "restarts": 2,
                    "blocks_per_restart": 4,
                    "warmup_calls_per_variant": 3,
                    "timed_calls_per_side_per_block": 2,
                },
            ],
            "schedule_seed": "33" * 32,
        },
    }
    value["protocol_self_sha256"] = canonical_protocol_self_sha256(value)
    return value


@dataclasses.dataclass(frozen=True)
class _FakeCompleteMatcher:
    predictions: tuple[int, ...]

    def invoke_backend_prevalidated(self, packed: np.ndarray) -> np.ndarray:
        if len(packed) != len(self.predictions):
            raise AssertionError("fake complete matcher received the wrong row count")
        return np.asarray(self.predictions, dtype=np.uint8).reshape(-1, 1)


@dataclasses.dataclass(frozen=True)
class _FakeTreatment(_FakeCompleteMatcher):
    def precompute_residual_batch(self, *args: object, **kwargs: object):
        raise AssertionError("latency materialization must not invoke the planner")


def _fake_provenance() -> dict[str, object]:
    return {
        "circuit_sha256": "41" * 32,
        "dem_sha256": "42" * 32,
        "layout_fingerprint": "layout",
        "graph_fingerprint": "graph",
        "validated_catalog_fingerprint": "catalog",
        "projection_fingerprint": "projection",
        "num_detectors": 9,
        "num_observables": 3,
    }


def _control_equality(shots: int) -> dict[str, object]:
    return {
        name: {
            "shots": shots,
            "equal": shots,
            "mismatches": 0,
            "ordered_range_evidence_sha256": character * 64,
        }
        for name, character in (
            ("ordinary_treatment_vs_telemetry", "1"),
            ("global_vs_adapter_control", "2"),
            ("global_vs_uf_shadow", "3"),
        )
    }


def _source_metric(*, original: int, residual: int) -> dict[str, object]:
    return {
        "projection_fingerprint": "projection",
        "capture": "metrics",
        "num_detectors": 9,
        "num_observables": 3,
        "durable_detector_boundary": [0],
        "durable_support_edge_ids": [5],
        "durable_observable_frame": "00",
        "durable_support_count": 1,
        "durable_boundary_count": 1,
        "durable_frame_weight": 0,
        "original_detector_count": original,
        "cluster_summary_complete": False,
        "maximum_final_component_defect_count": None,
        "completed_final_component_count": 1,
        "completed_component_size_histogram": [[1, 1]],
        "committed_defect_count": 1,
        "residual_detector_count": residual,
    }


def _authenticated_metric_rows(
    shots: int,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    lanes: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    for shot_id in range(shots):
        lanes.append(
            {
                "global_shot_id": shot_id,
                "lane_offset": 0,
                "adapter": {
                    "status": "censored",
                    "counters": {
                        "growth_event_count": 2,
                        "successful_union_count": 1,
                        "heap_operation_count": 4,
                        "peel_operation_count": 3,
                    },
                },
            }
        )
        components.extend(
            (
                {
                    "global_shot_id": shot_id,
                    "lane_offset": 0,
                    "state_collection": "completed_components",
                    "durable_decision": [True, "committed"],
                    "adapter": {"cluster_defect_count": 1},
                },
                {
                    "global_shot_id": shot_id,
                    "lane_offset": 0,
                    "state_collection": "censored_components",
                    "durable_decision": None,
                    "adapter": {"partial_cluster_defect_lower_bound": 3},
                },
            )
        )
    return tuple(lanes), tuple(components)


def test_materialization_uses_persisted_decisions_without_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    prepared = PreparedCell(
        cell=protocol["selected_cell"],
        circuit=object(),
        dem=object(),
        global_decoder=_FakeCompleteMatcher((0, 1)),
        treatment_decoder=_FakeTreatment((0, 1)),
        control_decoder=object(),
        shadow_decoder=object(),
        provenance=_fake_provenance(),
    )
    source = tmp_path / "characterization"
    (source / "corpus").mkdir(parents=True)
    detectors = np.array([[1, 0], [3, 0]], dtype=np.uint8)
    detector_bytes = detectors.tobytes()
    (source / "corpus" / "detectors.bitpack").write_bytes(detector_bytes)
    detector_digest = integration._sha256(detector_bytes)
    sentinel_observable_digest = "fe" * 32
    shot_rows = (
        {
            "global_shot_id": 0,
            "lane_count": 1,
            "component_count": 2,
            "adapter_metrics": _source_metric(original=1, residual=0),
            "global_prediction_hex": "00",
            "treatment_prediction_hex": "00",
            "actual_observables_hex": "07",
        },
        {
            "global_shot_id": 1,
            "lane_count": 1,
            "component_count": 2,
            "adapter_metrics": _source_metric(original=2, residual=1),
            "global_prediction_hex": "01",
            "treatment_prediction_hex": "01",
            "actual_observables_hex": "07",
        },
    )
    lane_rows, component_rows = _authenticated_metric_rows(2)
    verified = VerifiedCollection(
        summary={
            "payload_sha256": "51" * 32,
            "provenance": _fake_provenance(),
        },
        shot_rows=shot_rows,
        lane_rows=lane_rows,
        component_rows=component_rows,
        cluster_records=(),
        control_equality=_control_equality(2),
        corpus_identity={
            "detectors_sha256": detector_digest,
            "observables_sha256": sentinel_observable_digest,
        },
        detector_corpus_bytes=detector_bytes,
        detector_corpus_sha256=detector_digest,
    )
    monkeypatch.setattr(
        integration, "verify_collection", lambda *args, **kwargs: verified
    )

    manifest = integration.materialize_latency_corpus_from_characterization(
        protocol,
        characterization_dir=source,
        out_dir=tmp_path / "latency",
        scientific=False,
        processes=1,
        prepared=prepared,
    )
    corpus = load_authenticated_latency_corpus(manifest)
    np.testing.assert_array_equal(corpus.detectors, detectors)
    np.testing.assert_array_equal(
        corpus.residuals, np.array([[0, 0], [2, 0]], dtype=np.uint8)
    )
    assert json.loads(corpus.summary_json[0]) == {
        "cluster_summary_complete": False,
        "committed_defect_count": 1,
        "global_shot_id": 0,
        "growth_event_count": 2,
        "heap_operation_count": 4,
        "maximum_final_component_defect_count": None,
        "maximum_partial_component_defect_lower_bound": 3,
        "peel_operation_count": 3,
        "residual_detector_count": 0,
        "successful_union_count": 1,
    }
    emitted = manifest.read_text(encoding="utf-8") + (
        manifest.parent / "summaries.json"
    ).read_text(
        encoding="utf-8",
    )
    assert "actual_observables" not in emitted
    assert "observables_sha256" not in emitted
    assert sentinel_observable_digest not in emitted
    assert corpus.corpus_digest == detector_digest
    provenance = json.loads(corpus.provenance_json)
    attestation = provenance["full_corpus_prediction_attestation"]
    assert attestation["shot_count"] == 2
    assert attestation["original_prediction_equality"] == {
        "equal": 2,
        "mismatches": 0,
    }
    assert attestation["residual_prediction_equality"] == {
        "equal": 2,
        "mismatches": 0,
    }
    assert provenance["full_corpus_prediction_attestation_sha256"] == (
        integration._digest_mapping(attestation)
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("boundary", "residual detector count differs"),
        ("support_count", "durable support count differs"),
        ("projection", "projection fingerprint differs"),
    ),
)
def test_materialization_rejects_tampered_boundary_or_metric_rows(
    tamper: str,
    message: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    provenance = _fake_provenance()
    prepared = PreparedCell(
        cell=protocol["selected_cell"],
        circuit=object(),
        dem=object(),
        global_decoder=object(),
        treatment_decoder=_FakeTreatment((0,)),
        control_decoder=object(),
        shadow_decoder=object(),
        provenance=provenance,
    )
    source = tmp_path / "characterization"
    (source / "corpus").mkdir(parents=True)
    data = bytes([1, 0])
    (source / "corpus" / "detectors.bitpack").write_bytes(data)
    metric = _source_metric(original=1, residual=0)
    if tamper == "boundary":
        metric["durable_detector_boundary"] = [1]
    elif tamper == "support_count":
        metric["durable_support_count"] = 2
    elif tamper == "projection":
        metric["projection_fingerprint"] = "other"
    lane_rows, component_rows = _authenticated_metric_rows(1)
    verified = VerifiedCollection(
        summary={"payload_sha256": "51" * 32, "provenance": provenance},
        shot_rows=(
            {
                "global_shot_id": 0,
                "lane_count": 1,
                "component_count": 2,
                "adapter_metrics": metric,
            },
        ),
        lane_rows=lane_rows,
        component_rows=component_rows,
        cluster_records=(),
        control_equality=_control_equality(1),
        corpus_identity={"detectors_sha256": integration._sha256(data)},
        detector_corpus_bytes=data,
        detector_corpus_sha256=integration._sha256(data),
    )
    monkeypatch.setattr(
        integration, "verify_collection", lambda *args, **kwargs: verified
    )
    with pytest.raises(ValueError, match=message):
        integration.materialize_latency_corpus_from_characterization(
            protocol,
            characterization_dir=source,
            out_dir=tmp_path / "latency",
            scientific=False,
            processes=1,
            prepared=prepared,
        )


@pytest.mark.parametrize("tamper", ("control", "global", "treatment"))
def test_materialization_rejects_control_or_full_corpus_prediction_mismatch(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    provenance = _fake_provenance()
    global_prediction = 1 if tamper == "global" else 0
    treatment_prediction = 1 if tamper == "treatment" else 0
    prepared = PreparedCell(
        cell=protocol["selected_cell"],
        circuit=object(),
        dem=object(),
        global_decoder=_FakeCompleteMatcher((global_prediction,)),
        treatment_decoder=_FakeTreatment((treatment_prediction,)),
        control_decoder=object(),
        shadow_decoder=object(),
        provenance=provenance,
    )
    source = tmp_path / "characterization"
    (source / "corpus").mkdir(parents=True)
    data = bytes([1, 0])
    (source / "corpus" / "detectors.bitpack").write_bytes(data)
    lane_rows, component_rows = _authenticated_metric_rows(1)
    controls = _control_equality(1)
    if tamper == "control":
        controls["global_vs_uf_shadow"]["mismatches"] = 1
        controls["global_vs_uf_shadow"]["equal"] = 0
    verified = VerifiedCollection(
        summary={"payload_sha256": "51" * 32, "provenance": provenance},
        shot_rows=(
            {
                "global_shot_id": 0,
                "lane_count": 1,
                "component_count": 2,
                "adapter_metrics": _source_metric(original=1, residual=0),
                "global_prediction_hex": "00",
                "treatment_prediction_hex": "00",
            },
        ),
        lane_rows=lane_rows,
        component_rows=component_rows,
        cluster_records=(),
        control_equality=controls,
        corpus_identity={"detectors_sha256": integration._sha256(data)},
        detector_corpus_bytes=data,
        detector_corpus_sha256=integration._sha256(data),
    )
    monkeypatch.setattr(
        integration, "verify_collection", lambda *args, **kwargs: verified
    )
    message = (
        "control-equality ledger"
        if tamper == "control"
        else "full original-corpus matcher"
        if tamper == "global"
        else "full residual-corpus matcher"
    )
    with pytest.raises(ValueError, match=message):
        integration.materialize_latency_corpus_from_characterization(
            protocol,
            characterization_dir=source,
            out_dir=tmp_path / "latency",
            scientific=False,
            processes=1,
            prepared=prepared,
        )


def test_latency_protocol_maps_every_frozen_count_and_seed() -> None:
    protocol = _protocol()
    cpu = min(os.sched_getaffinity(0))
    timing = integration.latency_protocol_from_experiment(protocol, cpu=cpu)

    assert [batch.to_json() for batch in timing.batches] == protocol["latency"][
        "batches"
    ]
    assert timing.schedule_seed == int("33" * 32, 16)
    assert timing.host_policy.cpu_affinity == (cpu,)


def test_real_d3_factory_rebuild_is_pickleable_and_callable(tmp_path: Path) -> None:
    protocol = _protocol()
    prepared = prepare_selected_cell(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        processes=1,
        scientific=False,
    )
    num_detectors = int(prepared.provenance["num_detectors"])
    detectors = np.zeros((2, (num_detectors + 7) // 8), dtype=np.uint8)
    identity = integration._prepared_identity(protocol, prepared)
    detector_digest = "91" * 32
    recorded_global = prepared.global_decoder.invoke_backend_prevalidated(detectors)
    recorded_treatment = prepared.treatment_decoder.invoke_backend_prevalidated(detectors)
    source_rows = tuple(
        {
            "global_prediction_hex": bytes(recorded_global[index]).hex(),
            "treatment_prediction_hex": bytes(recorded_treatment[index]).hex(),
        }
        for index in range(len(detectors))
    )
    attestation = integration._full_corpus_prediction_attestation(
        prepared=prepared,
        detectors=detectors,
        residuals=detectors.copy(),
        source_rows=source_rows,
        control_equality=_control_equality(2),
        num_observables=int(prepared.provenance["num_observables"]),
    )
    manifest = write_authenticated_latency_corpus(
        tmp_path / "corpus",
        detectors=detectors,
        residuals=detectors.copy(),
        num_detectors=num_detectors,
        global_shot_ids=(0, 1),
        summaries=({"global_shot_id": 0}, {"global_shot_id": 1}),
        provenance={
            "schema": integration.MATERIALIZATION_SCHEMA,
            **identity,
            "source_detector_sha256": detector_digest,
            "full_corpus_prediction_attestation": attestation,
            "full_corpus_prediction_attestation_sha256": integration._digest_mapping(
                attestation
            ),
        },
        corpus_digest=detector_digest,
    )
    factory = integration.YokedPatchUFLatencyFactory(
        protocol=protocol,
        corpus_manifest_path=str(manifest),
        scientific=False,
        processes=1,
    )
    factory = pickle.loads(pickle.dumps(factory))
    workload = factory(0, 1)

    assert tuple(workload.variant_map()) == VARIANT_NAMES
    for variant in workload.variants:
        corpus = (
            workload.corpus.detectors
            if variant.corpus_kind == "detectors"
            else workload.corpus.residuals
        )
        result = variant.function(corpus[:1])
        assert np.asarray(result).shape[0] == 1
    assert integration.verify_latency_factory(factory) == factory.suite_identity

    loaded = load_authenticated_latency_corpus(manifest)
    tampered = json.loads(loaded.provenance_json)
    tampered_attestation = tampered["full_corpus_prediction_attestation"]
    tampered_attestation["original_prediction_equality"] = {
        "equal": 1,
        "mismatches": 1,
    }
    tampered["full_corpus_prediction_attestation_sha256"] = (
        integration._digest_mapping(tampered_attestation)
    )
    with pytest.raises(ValueError, match="attestation equality mismatch"):
        integration._validate_full_corpus_prediction_attestation(tampered, loaded)
