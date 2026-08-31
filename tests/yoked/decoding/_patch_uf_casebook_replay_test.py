from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding import _patch_uf_casebook_replay as replay
from yoked.decoding._patch_uf_analysis import CASEBOOK_CATEGORIES
from yoked.decoding._patch_uf_decoder import CaptureMode
from yoked.decoding._patch_uf_experiment import (
    CHARACTERIZATION_STAGE,
    PROTOCOL_SCHEMA,
    SEED_DERIVATION,
    SHAKEOUT_STAGE,
    VerifiedCollection,
    _normalize_metrics,
    canonical_protocol_self_sha256,
    prepare_selected_cell,
)
from yoked.decoding._patch_uf_reference import BudgetLimits
from yoked.decoding._promatch_stats import canonical_json_bytes


def _protocol() -> dict[str, object]:
    limits = {field.name: None for field in dataclasses.fields(BudgetLimits)}
    value: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "status": "DRAFT",
        "frozen": False,
        "experiment_id": "11" * 32,
        "source_identity": {"fixture": "d3-casebook-replay"},
        "selected_cell": {
            "cell_id": "casebook-smoke-d3-n6-y2-r3-p0.001",
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
    }
    value["protocol_self_sha256"] = canonical_protocol_self_sha256(value)
    return value


@pytest.fixture(scope="module")
def real_case():
    protocol = _protocol()
    prepared = prepare_selected_cell(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        processes=1,
        scientific=False,
    )
    width = (int(prepared.provenance["num_detectors"]) + 7) // 8
    packed = np.zeros((1, width), dtype=np.uint8)
    global_prediction = prepared.global_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    treatment_prediction, corrections = (
        prepared.treatment_decoder.decode_shots_bit_packed_with_capture(
            bit_packed_detection_event_data=packed,
            capture=CaptureMode.METRICS,
        )
    )
    shot_metrics, lane_groups, component_groups = _normalize_metrics(
        corrections,
        shots=1,
        expected_lanes=12,
        max_components_per_shot=4096,
        max_metric_bytes=1_000_000,
    )
    case = replay.CasebookReplayCase(
        global_shot_id=0,
        categories=("highest-heap-operation-count",),
        packed_detector=bytes(packed[0]),
        global_prediction_hex=bytes(global_prediction[0]).hex(),
        treatment_prediction_hex=bytes(treatment_prediction[0]).hex(),
        adapter_metrics=shot_metrics[0],
        lanes=tuple(lane_groups[0]),
        components=tuple(component_groups[0]),
    )
    return protocol, prepared, case


def _request(protocol, prepared, *cases) -> replay.CasebookReplayRequest:
    return replay.CasebookReplayRequest(
        protocol=protocol,
        stage=CHARACTERIZATION_STAGE,
        processes=1,
        scientific=False,
        parent_pid=os.getpid(),
        expected_provenance=prepared.provenance,
        collection_payload_sha256="31" * 32,
        detector_corpus_sha256="32" * 32,
        analysis_payload_sha256="33" * 32,
        casebook_selection_sha256="34" * 32,
        cases=tuple(cases),
    )


def test_real_d3_case_replays_bit_exactly_in_spawned_process(real_case) -> None:
    protocol, prepared, case = real_case
    second_case = dataclasses.replace(case, global_shot_id=1)
    first = replay.replay_casebook_request_fresh_process(
        _request(protocol, prepared, case, second_case), worker_processes=2
    )
    second = replay.replay_casebook_request_fresh_process(
        _request(protocol, prepared, case, second_case), worker_processes=2
    )

    assert first == second
    assert first["fresh_process"] is True
    assert first["status"] == "reconciled"
    assert first["replayed_cases"] == 2
    assert first["worker_processes"] == 2
    assert first["cases"][0]["global_shot_id"] == 0


def test_fresh_process_replay_fails_closed_on_prediction_mismatch(real_case) -> None:
    protocol, prepared, case = real_case
    wrong = dataclasses.replace(case, treatment_prediction_hex="ff")

    with pytest.raises(ValueError, match="treatment replay mismatch"):
        replay.replay_casebook_request_fresh_process(
            _request(protocol, prepared, wrong)
        )


def _analysis_casebook(
    *,
    shot: dict[str, object],
    lanes: list[dict[str, object]],
    components: list[dict[str, object]],
    root_hex: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    selected_category = "highest-heap-operation-count"
    for category in CASEBOOK_CATEGORIES:
        rows = []
        if category == selected_category:
            rows.append(
                {
                    "global_shot_id": 0,
                    "metric": 0,
                    "selection_sha256": replay._selection_digest(
                        bytes.fromhex(root_hex), category, 0
                    ),
                    "shot": shot,
                    "lanes": lanes,
                    "components": components,
                }
            )
        result[category] = {
            "candidate_shots": len(rows),
            "retained_shots": len(rows),
            "maximum_retained": 1,
            "selection": (
                "metric-descending-then-rooted-sha256"
                if category
                in {
                    "largest-final-component",
                    "largest-committed-component",
                    "largest-censored-partial-lower-bound",
                    "highest-heap-operation-count",
                }
                else "rooted-sha256-ascending"
            ),
            "rows": rows,
        }
    return result


def _write_analysis(path: Path, value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    value["payload_sha256"] = replay._sha256(canonical_json_bytes(unsigned))
    path.write_bytes(canonical_json_bytes(value))


@pytest.mark.parametrize(
    ("stage", "install_corpus"),
    (
        (SHAKEOUT_STAGE, False),
        (CHARACTERIZATION_STAGE, True),
    ),
)
def test_authenticated_request_uses_range_detectors_for_both_scientific_stages(
    real_case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    install_corpus: bool,
) -> None:
    protocol, prepared, case = real_case
    detector_bytes = case.packed_detector
    detector_digest = replay._sha256(detector_bytes)
    collection = tmp_path / "collection"
    collection.mkdir()
    if install_corpus:
        (collection / "corpus").mkdir()
        (collection / "corpus" / "detectors.bitpack").write_bytes(
            detector_bytes
        )
    lanes = [
        {"global_shot_id": 0, **copy.deepcopy(row)} for row in case.lanes
    ]
    components = [
        {"global_shot_id": 0, **copy.deepcopy(row)} for row in case.components
    ]
    shot = {
        "global_shot_id": 0,
        "global_prediction_hex": case.global_prediction_hex,
        "treatment_prediction_hex": case.treatment_prediction_hex,
        "actual_observables_hex": "00",
        "global_failed": False,
        "treatment_failed": False,
        "prediction_agreement": True,
        "lane_count": len(lanes),
        "component_count": len(components),
        "adapter_metrics": copy.deepcopy(case.adapter_metrics),
    }
    corpus_identity = (
        {
            "index_path": "corpus/index.json",
            "index_payload_sha256": "41" * 32,
            "detectors_sha256": detector_digest,
            "observables_sha256": "42" * 32,
        }
        if install_corpus
        else None
    )
    collection_digest = "43" * 32
    verified = VerifiedCollection(
        summary={
            "payload_sha256": collection_digest,
            "provenance": prepared.provenance,
        },
        shot_rows=(shot,),
        lane_rows=tuple(lanes),
        component_rows=tuple(components),
        cluster_records=(),
        control_equality={},
        corpus_identity=corpus_identity,
        detector_corpus_bytes=detector_bytes,
        detector_corpus_sha256=detector_digest,
    )
    monkeypatch.setattr(replay, "verify_collection", lambda *args, **kwargs: verified)
    root_hex = "51" * 32
    analysis = {
        "schema": "patch-uf-analysis-v1",
        "schema_version": 1,
        "source": {
            "experiment_id": protocol["experiment_id"],
            "protocol_self_sha256": protocol["protocol_self_sha256"],
            "collection_payload_sha256": collection_digest,
            "stage": stage,
            "cell_id": protocol["selected_cell"]["cell_id"],
            "corpus_identity": corpus_identity,
        },
        "config": {
            "casebook_seed_root": root_hex,
            "maximum_cases_per_category": 1,
        },
        "reconciliation": {"status": "reconciled", "shots": 1},
        "casebook": _analysis_casebook(
            shot=shot,
            lanes=lanes,
            components=components,
            root_hex=root_hex,
        ),
    }
    analysis_path = tmp_path / "analysis.json"
    _write_analysis(analysis_path, analysis)

    request = replay.build_authenticated_casebook_replay_request(
        protocol,
        collection_out=collection,
        analysis_path=analysis_path,
        stage=stage,
        processes=1,
        scientific=False,
    )

    assert len(request.cases) == 1
    assert request.cases[0].packed_detector == detector_bytes
    assert request.detector_corpus_sha256 == detector_digest
    assert "actual_observables" not in repr(request)

    if install_corpus:
        (collection / "corpus" / "detectors.bitpack").write_bytes(
            bytes([detector_bytes[0] ^ 1, *detector_bytes[1:]])
        )
        with pytest.raises(ValueError, match="differs from authenticated ranges"):
            replay.build_authenticated_casebook_replay_request(
                protocol,
                collection_out=collection,
                analysis_path=analysis_path,
                stage=stage,
                processes=1,
                scientific=False,
            )
        (collection / "corpus" / "detectors.bitpack").write_bytes(
            detector_bytes
        )

    tampered = copy.deepcopy(analysis)
    tampered["casebook"]["highest-heap-operation-count"]["rows"][0]["shot"][
        "treatment_prediction_hex"
    ] = "ff"
    _write_analysis(analysis_path, tampered)
    with pytest.raises(ValueError, match="shot row differs"):
        replay.build_authenticated_casebook_replay_request(
            protocol,
            collection_out=collection,
            analysis_path=analysis_path,
            stage=stage,
            processes=1,
            scientific=False,
        )


def test_authenticated_request_fails_closed_on_range_aggregate_tampering(
    real_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, prepared, case = real_case
    detector_bytes = case.packed_detector
    detector_digest = replay._sha256(detector_bytes)
    collection = tmp_path / "shakeout"
    collection.mkdir()
    lanes = [{"global_shot_id": 0, **copy.deepcopy(row)} for row in case.lanes]
    components = [
        {"global_shot_id": 0, **copy.deepcopy(row)} for row in case.components
    ]
    shot = {
        "global_shot_id": 0,
        "global_prediction_hex": case.global_prediction_hex,
        "treatment_prediction_hex": case.treatment_prediction_hex,
        "actual_observables_hex": "00",
        "global_failed": False,
        "treatment_failed": False,
        "prediction_agreement": True,
        "lane_count": len(lanes),
        "component_count": len(components),
        "adapter_metrics": copy.deepcopy(case.adapter_metrics),
    }
    collection_digest = "43" * 32
    verified = VerifiedCollection(
        summary={
            "payload_sha256": collection_digest,
            "provenance": prepared.provenance,
        },
        shot_rows=(shot,),
        lane_rows=tuple(lanes),
        component_rows=tuple(components),
        cluster_records=(),
        control_equality={},
        corpus_identity=None,
        detector_corpus_bytes=detector_bytes,
        detector_corpus_sha256=detector_digest,
    )
    # Simulate a compromised return value despite the frozen dataclass guard.
    object.__setattr__(verified, "detector_corpus_bytes", b"\xff" * len(detector_bytes))
    monkeypatch.setattr(replay, "verify_collection", lambda *args, **kwargs: verified)
    root_hex = "51" * 32
    analysis = {
        "schema": "patch-uf-analysis-v1",
        "schema_version": 1,
        "source": {
            "experiment_id": protocol["experiment_id"],
            "protocol_self_sha256": protocol["protocol_self_sha256"],
            "collection_payload_sha256": collection_digest,
            "stage": SHAKEOUT_STAGE,
            "cell_id": protocol["selected_cell"]["cell_id"],
            "corpus_identity": None,
        },
        "config": {
            "casebook_seed_root": root_hex,
            "maximum_cases_per_category": 1,
        },
        "reconciliation": {"status": "reconciled", "shots": 1},
        "casebook": _analysis_casebook(
            shot=shot,
            lanes=lanes,
            components=components,
            root_hex=root_hex,
        ),
    }
    analysis_path = tmp_path / "analysis.json"
    _write_analysis(analysis_path, analysis)

    with pytest.raises(ValueError, match="authenticated detector-corpus digest"):
        replay.build_authenticated_casebook_replay_request(
            protocol,
            collection_out=collection,
            analysis_path=analysis_path,
            stage=SHAKEOUT_STAGE,
            processes=1,
            scientific=False,
        )
