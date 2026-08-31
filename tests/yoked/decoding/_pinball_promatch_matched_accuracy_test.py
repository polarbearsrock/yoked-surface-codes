from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding._patch_uf_experiment import VerifiedCollection, fixed_worker_ranges
from yoked.decoding._pinball_promatch_matched_accuracy import (
    ARM_ORDER,
    PAIR_DEFINITIONS,
    aggregate_matched_ledgers,
    clear_matched_worker_preload,
    collect_matched_range,
    load_matched_corpus,
    matched_corpus_from_verified,
    matched_range_tasks,
    preload_matched_worker,
    validate_matched_aggregate,
    validate_matched_ledger,
    worker_collect_matched_range,
    write_matched_corpus,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verified() -> VerifiedCollection:
    shots = 32
    detectors = bytes((index * 3) & 0xFF for index in range(shots))
    actual = bytes(index & 1 for index in range(shots))
    uf_predictions = bytes((index // 2) & 1 for index in range(shots))
    rows = []
    for shot_id in range(shots):
        global_prediction = 0
        actual_value = actual[shot_id]
        uf_prediction = uf_predictions[shot_id]
        original_hw = int(detectors[shot_id]).bit_count()
        rows.append(
            {
                "global_shot_id": shot_id,
                "global_prediction_hex": bytes([global_prediction]).hex(),
                "treatment_prediction_hex": bytes([uf_prediction]).hex(),
                "actual_observables_hex": bytes([actual_value]).hex(),
                "global_failed": global_prediction != actual_value,
                "treatment_failed": uf_prediction != actual_value,
                "prediction_agreement": global_prediction == uf_prediction,
                "adapter_metrics": {
                    "original_detector_count": original_hw,
                    "residual_detector_count": max(0, original_hw - 1),
                    "residual_body_detector_count": max(0, original_hw - 1),
                    "residual_terminal_detector_count": 0,
                    "residual_yoke_detector_count": 0,
                },
            }
        )
    provenance = {
        "circuit_sha256": "11" * 32,
        "dem_sha256": "22" * 32,
        "num_detectors": 8,
        "num_observables": 2,
    }
    summary = {
        "experiment_id": "33" * 32,
        "protocol_self_sha256": "44" * 32,
        "payload_sha256": "55" * 32,
        "stage": "characterization",
        "cell_id": "matched-d3",
        "shots": shots,
        "provenance": provenance,
    }
    corpus_identity = {
        "detectors_sha256": _sha256(detectors),
        "observables_sha256": _sha256(actual),
        "index_payload_sha256": "66" * 32,
    }
    return VerifiedCollection(
        summary=summary,
        shot_rows=tuple(rows),
        lane_rows=(),
        component_rows=(),
        cluster_records=(),
        control_equality={},
        corpus_identity=corpus_identity,
        detector_corpus_bytes=detectors,
        detector_corpus_sha256=_sha256(detectors),
    )


class _ZeroMatcher:
    def decode_batch(
        self,
        packed: np.ndarray,
        *,
        bit_packed_shots: bool,
        bit_packed_predictions: bool,
    ) -> np.ndarray:
        assert bit_packed_shots and bit_packed_predictions
        return np.zeros((len(packed), 1), dtype=np.uint8)


def _promatch_result() -> SimpleNamespace:
    domain = SimpleNamespace(
        initial_hw=1,
        attempted_residual_hw=0,
        final_residual_hw=0,
        status="success",
        fallback_reason=None,
        attempted_stage_counts=(1, 0, 0, 0),
        committed_stage_counts=(1, 0, 0, 0),
        attempted_matches=1,
        committed_matches=1,
        boundary_was_added=False,
        boundary_was_used=False,
        boundary_discarded_unused=False,
    )
    return SimpleNamespace(
        decision_weight=1.0,
        xor_support_weight=1.0,
        paths=(),
        domain_stats={"domain": domain},
    )


def _pinball_result() -> SimpleNamespace:
    domain = SimpleNamespace(
        complex=False,
        initial_hw=1,
        tentative_residual_hw=0,
        final_residual_hw=0,
    )
    return SimpleNamespace(
        edge_support=(1,),
        tentative_edge_support=(1,),
        observable_frame=np.zeros(2, dtype=np.uint8),
        physical_correction=(1,),
        tentative_physical_correction=(1,),
        domain_results={"domain": domain},
        complex=False,
        stage_match_counts=(1,),
    )


class _Compiled:
    def __init__(self, *, pinball: bool) -> None:
        self.graph = SimpleNamespace(
            matcher=_ZeroMatcher(),
            layout=SimpleNamespace(
                terminal_detector_ids=np.asarray([], dtype=np.int64),
                yoke_detector_ids=np.asarray([], dtype=np.int64),
            ),
        )
        self.num_observables = 2
        self.pinball = pinball
        if pinball:
            self.schedule = SimpleNamespace(
                stages=(SimpleNamespace(stage="M"),)
            )

    def predecode_shots(self, unpacked: np.ndarray):
        residual = np.zeros_like(unpacked)
        frames = np.zeros((len(unpacked), 2), dtype=np.uint8)
        result = _pinball_result if self.pinball else _promatch_result
        return residual, frames, tuple(result() for _ in unpacked)


def _prepared(*, global_mismatch: bool = False) -> SimpleNamespace:
    matcher = _ZeroMatcher()
    if global_mismatch:
        class _MismatchMatcher(_ZeroMatcher):
            def decode_batch(self, packed, **kwargs):
                result = super().decode_batch(packed, **kwargs)
                result[0, 0] = 1
                return result

        matcher = _MismatchMatcher()
    return SimpleNamespace(
        cell={"cell_id": "matched-d3"},
        provenance={
            "circuit_sha256": "11" * 32,
            "dem_sha256": "22" * 32,
            "num_detectors": 8,
            "num_observables": 2,
            "promatch_layout_fingerprint": "77" * 32,
            "promatch_graph_fingerprint": "88" * 32,
            "pinball_layout_fingerprint": "99" * 32,
            "pinball_graph_fingerprint": "aa" * 32,
            "pinball_schedule_fingerprint": "bb" * 32,
            "arms": {},
        },
        matcher_u0=matcher,
        compiled_promatch=_Compiled(pinball=False),
        compiled_pinball=_Compiled(pinball=True),
    )


def test_exact_verified_projection_and_range_ledger_reconcile() -> None:
    corpus = matched_corpus_from_verified(_verified())
    assert corpus.shots == 32
    assert corpus.detectors.flags.writeable is False
    assert corpus.actual_observables.flags.writeable is False
    prepared = _prepared()
    shot_range = fixed_worker_ranges(corpus.shots)[0]
    ledger = collect_matched_range(
        prepared, corpus, shot_range=shot_range, microbatch_size=1
    )

    assert validate_matched_ledger(
        ledger,
        corpus=corpus,
        expected_prepared_provenance=prepared.provenance,
    ) == shot_range
    assert ledger["global_prediction_equality"] == {"equal": 1, "mismatches": 0}
    assert set(ledger["correctness_cube"]) == {
        f"{value:04b}" for value in range(16)
    }
    assert set(ledger["pairwise_contingencies"]) == set(PAIR_DEFINITIONS)
    assert len(PAIR_DEFINITIONS) == 6
    assert ledger["arm_order"] == list(ARM_ORDER)
    assert ledger["telemetry"]["global"]["residual_event_sum"] == ledger[
        "telemetry"
    ]["common"]["original_event_sum"]
    assert ledger["telemetry"]["union_find"][
        "residual_hw_available_shots"
    ] == 1
    assert sum(
        ledger["telemetry"]["global"][
            "original_residual_hw_joint_histogram"
        ].values()
    ) == 1
    assert sum(
        ledger["telemetry"]["union_find"][
            "original_residual_hw_joint_histogram"
        ].values()
    ) == 1
    assert ledger["telemetry"]["union_find"][
        "residual_role_available_shots"
    ] == 1


def test_global_recomputation_and_imported_rows_fail_closed() -> None:
    verified = _verified()
    corpus = matched_corpus_from_verified(verified)
    with pytest.raises(ValueError, match="recomputed Global prediction differs"):
        collect_matched_range(
            _prepared(global_mismatch=True),
            corpus,
            shot_range=fixed_worker_ranges(corpus.shots)[0],
        )

    changed_rows = list(copy.deepcopy(verified.shot_rows))
    changed_rows[0]["global_failed"] = True
    changed = VerifiedCollection(
        summary=verified.summary,
        shot_rows=tuple(changed_rows),
        lane_rows=verified.lane_rows,
        component_rows=verified.component_rows,
        cluster_records=verified.cluster_records,
        control_equality=verified.control_equality,
        corpus_identity=verified.corpus_identity,
        detector_corpus_bytes=verified.detector_corpus_bytes,
        detector_corpus_sha256=verified.detector_corpus_sha256,
    )
    with pytest.raises(ValueError, match="recorded global_failed differs"):
        matched_corpus_from_verified(changed)


def test_all_32_ranges_aggregate_and_digest_tampering_is_rejected() -> None:
    corpus = matched_corpus_from_verified(_verified())
    prepared = _prepared()
    rows = [
        collect_matched_range(prepared, corpus, shot_range=shot_range)
        for shot_range in fixed_worker_ranges(corpus.shots)
    ]
    result = aggregate_matched_ledgers(
        rows,
        corpus=corpus,
        expected_prepared_provenance=prepared.provenance,
    )
    validate_matched_aggregate(
        result,
        corpus=corpus,
        expected_prepared_provenance=prepared.provenance,
    )
    assert result["complete"] is True
    assert result["ranges"] == 32
    assert result["shots"] == 32
    assert sum(result["correctness_cube"].values()) == 32
    for pair in PAIR_DEFINITIONS:
        assert sum(result["pairwise_contingencies"][pair].values()) == 32
        assert sum(result["prediction_agreement"][pair].values()) == 32

    changed = copy.deepcopy(rows[0])
    changed["correctness_cube"]["0000"] += 1
    with pytest.raises(ValueError, match="payload digest"):
        validate_matched_ledger(changed, corpus=corpus)
    with pytest.raises(ValueError, match="duplicate range"):
        aggregate_matched_ledgers([rows[0], rows[0]], corpus=corpus)


def test_parent_preload_and_pickleable_exact_range_tasks() -> None:
    corpus = matched_corpus_from_verified(_verified())
    prepared = _prepared()
    tasks = matched_range_tasks(corpus, microbatch_size=1)
    assert len(tasks) == 32
    assert [task["range"]["range_id"] for task in tasks] == list(range(32))
    with pytest.raises(RuntimeError, match="did not inherit"):
        worker_collect_matched_range(tasks[0])
    preload_matched_worker(prepared, corpus)
    try:
        ledger = worker_collect_matched_range(tasks[0])
        assert ledger["range"] == tasks[0]["range"]
    finally:
        clear_matched_worker_preload()


def test_matched_corpus_persistence_round_trips_without_pickle(
    tmp_path: Path,
) -> None:
    corpus = matched_corpus_from_verified(_verified())
    output = tmp_path / "matched-corpus"
    manifest = write_matched_corpus(output, corpus)
    assert manifest == output / "manifest.json"
    loaded = load_matched_corpus(output)
    assert loaded.source_identity == corpus.source_identity
    assert loaded.source_provenance == corpus.source_provenance
    for name in (
        "detectors",
        "actual_observables",
        "global_predictions",
        "union_find_predictions",
        "global_failures",
        "union_find_failures",
        "union_find_residual_hw",
        "union_find_residual_body_hw",
        "union_find_residual_terminal_hw",
        "union_find_residual_yoke_hw",
    ):
        assert np.array_equal(getattr(loaded, name), getattr(corpus, name))
        assert getattr(loaded, name).flags.writeable is False
    with pytest.raises(ValueError, match="new absent path"):
        write_matched_corpus(output, corpus)


def test_matched_corpus_persistence_rejects_tampering_and_unexpected_files(
    tmp_path: Path,
) -> None:
    corpus = matched_corpus_from_verified(_verified())

    changed_array = tmp_path / "changed-array"
    write_matched_corpus(changed_array, corpus)
    detector_path = changed_array / "detectors.npy"
    payload = bytearray(detector_path.read_bytes())
    payload[-1] ^= 1
    detector_path.write_bytes(payload)
    with pytest.raises(ValueError, match="file digest mismatch"):
        load_matched_corpus(changed_array)

    changed_manifest = tmp_path / "changed-manifest"
    write_matched_corpus(changed_manifest, corpus)
    manifest_path = changed_manifest / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_identity"]["shots"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_matched_corpus(changed_manifest)

    unexpected = tmp_path / "unexpected"
    write_matched_corpus(unexpected, corpus)
    unexpected.joinpath("extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_matched_corpus(unexpected)
