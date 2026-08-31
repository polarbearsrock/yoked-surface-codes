"""Real YSC integration for the generic Patch-UF latency harness.

Materialization is the only code here that may inspect a verified paired
collection.  It emits a new detector/residual-only corpus.  Fresh timing
workers receive only that sanitized manifest plus the frozen experiment
protocol, rebuild the selected cell and four compiled adapters, and never open
the characterization observable corpus.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from yoked.decoding._patch_uf_experiment import (
    CHARACTERIZATION_STAGE,
    RANGE_COUNT,
    PreparedCell,
    VerifiedCollection,
    canonical_protocol_self_sha256,
    prepare_selected_cell,
    verify_collection,
)
from yoked.decoding._patch_uf_latency import (
    AuthenticatedLatencyCorpus,
    BatchTiming,
    HostPolicy,
    LatencyProtocol,
    LatencyWorkload,
    build_timed_variants,
    capture_host_policy,
    load_authenticated_latency_corpus,
    write_authenticated_latency_corpus,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


MATERIALIZATION_SCHEMA = "patch-uf-latency-materialization-v1"
FULL_CORPUS_ATTESTATION_SCHEMA = (
    "patch-uf-latency-full-corpus-prediction-attestation-v1"
)
_SUMMARY_FIELDS = (
    "global_shot_id",
    "cluster_summary_complete",
    "maximum_final_component_defect_count",
    "maximum_partial_component_defect_lower_bound",
    "committed_defect_count",
    "growth_event_count",
    "successful_union_count",
    "heap_operation_count",
    "peel_operation_count",
    "residual_detector_count",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_mapping(value: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(dict(value)))


def _canonical_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return json.loads(canonical_json_bytes(dict(value)))


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_protocol_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = _canonical_mapping(protocol, name="protocol")
    claimed = value.get("protocol_self_sha256")
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or claimed != canonical_protocol_self_sha256(value)
    ):
        raise ValueError("protocol self identity is invalid")
    experiment_id = value.get("experiment_id")
    if not isinstance(experiment_id, str) or len(experiment_id) != 64:
        raise ValueError("protocol experiment_id is invalid")
    cell = value.get("selected_cell")
    decoder = value.get("decoder")
    if not isinstance(cell, Mapping) or not isinstance(decoder, Mapping):
        raise ValueError("protocol selected_cell/decoder fields are malformed")
    return value


def _prepared_identity(
    protocol: Mapping[str, Any], prepared: PreparedCell
) -> dict[str, Any]:
    provenance = dict(prepared.provenance)
    required = (
        "circuit_sha256",
        "dem_sha256",
        "layout_fingerprint",
        "graph_fingerprint",
        "validated_catalog_fingerprint",
        "projection_fingerprint",
        "num_detectors",
        "num_observables",
    )
    if any(name not in provenance for name in required):
        raise ValueError("prepared selected-cell provenance is incomplete")
    return {
        "experiment_id": protocol["experiment_id"],
        "protocol_self_sha256": protocol["protocol_self_sha256"],
        "cell_id": protocol["selected_cell"]["cell_id"],
        "decoder_config_sha256": _digest_mapping(protocol["decoder"]),
        **{name: provenance[name] for name in required},
    }


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(
        _nonnegative_int(item, name=f"{name} item") for item in value
    )


def _group_authenticated_rows(
    rows: tuple[Mapping[str, Any], ...],
    *,
    shots: int,
    name: str,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    groups: list[list[Mapping[str, Any]]] = [[] for _ in range(shots)]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"verified {name} row is malformed")
        shot_id = _nonnegative_int(
            row.get("global_shot_id"), name=f"{name} global_shot_id"
        )
        if shot_id >= shots:
            raise ValueError(f"verified {name} row has an out-of-range shot ID")
        groups[shot_id].append(row)
    return tuple(tuple(group) for group in groups)


def _packed_bit_count(row: np.ndarray, *, bits: int) -> int:
    return int(
        np.count_nonzero(
            np.unpackbits(row, count=bits, bitorder="little")
        )
    )


def _zero_durable_frame(metrics: Mapping[str, Any], *, bits: int) -> None:
    encoded = metrics.get("durable_observable_frame")
    if not isinstance(encoded, str) or encoded.lower() != encoded:
        raise ValueError("durable observable frame must be lowercase hexadecimal")
    try:
        frame = bytes.fromhex(encoded)
    except ValueError as ex:
        raise ValueError("durable observable frame is not hexadecimal") from ex
    if len(frame) != (bits + 7) // 8:
        raise ValueError("durable observable frame has the wrong width")
    if bits % 8 and frame and frame[-1] >> (bits % 8):
        raise ValueError("durable observable frame has nonzero tail bits")
    recorded_weight = _nonnegative_int(
        metrics.get("durable_frame_weight"), name="durable_frame_weight"
    )
    if recorded_weight != sum(value.bit_count() for value in frame):
        raise ValueError("durable frame weight differs from persisted frame")
    if recorded_weight or any(frame):
        raise ValueError("V1 latency materialization requires zero durable frames")


def _row_adapter(row: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    adapter = row.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError(f"verified {name} adapter record is malformed")
    return adapter


def _materialize_authenticated_shot(
    *,
    detector: np.ndarray,
    shot: Mapping[str, Any],
    lanes: tuple[Mapping[str, Any], ...],
    components: tuple[Mapping[str, Any], ...],
    num_detectors: int,
    num_observables: int,
    projection_fingerprint: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    shot_id = _nonnegative_int(
        shot.get("global_shot_id"), name="shot global_shot_id"
    )
    if _nonnegative_int(shot.get("lane_count"), name="shot lane_count") != len(
        lanes
    ):
        raise ValueError("verified shot lane count differs from lane rows")
    if _nonnegative_int(
        shot.get("component_count"), name="shot component_count"
    ) != len(components):
        raise ValueError("verified shot component count differs from component rows")
    lane_offsets = tuple(
        _nonnegative_int(row.get("lane_offset"), name="lane_offset")
        for row in lanes
    )
    if lane_offsets != tuple(range(len(lanes))):
        raise ValueError("verified lane rows are not in dense lane order")

    metrics = shot.get("adapter_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("verified shot adapter_metrics is malformed")
    if metrics.get("capture") != "metrics":
        raise ValueError("shot correction was not retained in metrics capture mode")
    if metrics.get("projection_fingerprint") != projection_fingerprint:
        raise ValueError("shot projection fingerprint differs from selected cell")
    if _nonnegative_int(
        metrics.get("num_detectors"), name="shot num_detectors"
    ) != num_detectors:
        raise ValueError("shot detector count differs from selected cell")
    if _nonnegative_int(
        metrics.get("num_observables"), name="shot num_observables"
    ) != num_observables:
        raise ValueError("shot observable width differs from selected cell")

    boundary = _integer_tuple(
        metrics.get("durable_detector_boundary"),
        name="durable_detector_boundary",
    )
    if boundary != tuple(sorted(set(boundary))):
        raise ValueError("durable detector boundary must be sorted and unique")
    if boundary and boundary[-1] >= num_detectors:
        raise ValueError("durable detector boundary contains an invalid detector")
    support = _integer_tuple(
        metrics.get("durable_support_edge_ids"),
        name="durable_support_edge_ids",
    )
    if support != tuple(sorted(set(support))):
        raise ValueError("durable support edge IDs must be sorted and unique")
    if _nonnegative_int(
        metrics.get("durable_support_count"), name="durable_support_count"
    ) != len(support):
        raise ValueError("durable support count differs from persisted support")
    if _nonnegative_int(
        metrics.get("durable_boundary_count"), name="durable_boundary_count"
    ) != len(boundary):
        raise ValueError("durable boundary count differs from persisted boundary")
    _zero_durable_frame(metrics, bits=num_observables)

    residual = np.array(detector, dtype=np.uint8, copy=True)
    for detector_id in boundary:
        residual[detector_id // 8] ^= np.uint8(1 << (detector_id % 8))
    original_count = _packed_bit_count(detector, bits=num_detectors)
    if _nonnegative_int(
        metrics.get("original_detector_count"),
        name="original_detector_count",
    ) != original_count:
        raise ValueError("original detector count differs from detector corpus")
    residual_count = _packed_bit_count(residual, bits=num_detectors)
    if _nonnegative_int(
        metrics.get("residual_detector_count"),
        name="residual_detector_count",
    ) != residual_count:
        raise ValueError("residual detector count differs from persisted boundary")

    counter_names = (
        "growth_event_count",
        "successful_union_count",
        "heap_operation_count",
        "peel_operation_count",
    )
    counter_totals = {name: 0 for name in counter_names}
    lane_complete = True
    for lane in lanes:
        adapter = _row_adapter(lane, name="lane")
        status = adapter.get("status")
        if status not in ("empty", "completed", "censored"):
            raise ValueError("verified lane status is invalid")
        lane_complete &= status in ("empty", "completed")
        counters = adapter.get("counters")
        if not isinstance(counters, Mapping):
            raise ValueError("verified lane counters are malformed")
        for name in counter_names:
            counter_totals[name] += _nonnegative_int(
                counters.get(name), name=f"lane {name}"
            )

    completed_sizes: list[int] = []
    partial_sizes: list[int] = []
    committed_defects = 0
    for component in components:
        adapter = _row_adapter(component, name="component")
        state = component.get("state_collection")
        if state == "completed_components":
            size = _positive_int(
                adapter.get("cluster_defect_count"),
                name="completed cluster_defect_count",
            )
            completed_sizes.append(size)
            decision = component.get("durable_decision")
            if (
                not isinstance(decision, list)
                or len(decision) != 2
                or not isinstance(decision[0], bool)
                or not isinstance(decision[1], str)
            ):
                raise ValueError("completed component durable decision is malformed")
            if decision[0]:
                committed_defects += size
        elif state == "censored_components":
            partial_sizes.append(
                _nonnegative_int(
                    adapter.get("partial_cluster_defect_lower_bound"),
                    name="partial cluster defect lower bound",
                )
            )
        else:
            raise ValueError("component row has an unsupported state collection")

    recorded_complete = metrics.get("cluster_summary_complete")
    if not isinstance(recorded_complete, bool) or recorded_complete != lane_complete:
        raise ValueError("cluster-summary completeness differs from lane rows")
    maximum_completed = max(completed_sizes, default=0) if lane_complete else None
    if metrics.get("maximum_final_component_defect_count") != maximum_completed:
        raise ValueError("maximum completed component differs from component rows")
    if _nonnegative_int(
        metrics.get("completed_final_component_count"),
        name="completed_final_component_count",
    ) != len(completed_sizes):
        raise ValueError("completed component count differs from component rows")
    size_histogram: dict[int, int] = {}
    for size in completed_sizes:
        size_histogram[size] = size_histogram.get(size, 0) + 1
    recorded_histogram = metrics.get("completed_component_size_histogram")
    expected_histogram = [
        [size, count] for size, count in sorted(size_histogram.items())
    ]
    if recorded_histogram != expected_histogram:
        raise ValueError("completed component histogram differs from component rows")
    if _nonnegative_int(
        metrics.get("committed_defect_count"), name="committed_defect_count"
    ) != committed_defects:
        raise ValueError("committed defect count differs from component decisions")
    if residual_count != original_count - committed_defects:
        raise ValueError("persisted correction violates residual count conservation")

    summary = {
        "global_shot_id": shot_id,
        "cluster_summary_complete": lane_complete,
        "maximum_final_component_defect_count": maximum_completed,
        "maximum_partial_component_defect_lower_bound": max(
            partial_sizes, default=0
        ),
        "committed_defect_count": committed_defects,
        **counter_totals,
        "residual_detector_count": residual_count,
    }
    if tuple(summary) != _SUMMARY_FIELDS:
        raise AssertionError("latency summary emitter drifted")
    canonical_json_bytes(summary)
    residual.setflags(write=False)
    return residual, summary


def _read_verified_detectors(
    *,
    collection_dir: Path,
    verified: VerifiedCollection,
    num_detectors: int,
) -> tuple[np.ndarray, str]:
    path = collection_dir / "corpus" / "detectors.bitpack"
    if path.is_symlink() or not path.is_file():
        raise ValueError("verified detector corpus must be a regular file")
    data = path.read_bytes()
    digest = _sha256(data)
    identity = verified.corpus_identity
    if not isinstance(identity, Mapping) or identity.get("detectors_sha256") != digest:
        raise ValueError("verified detector corpus identity mismatch")
    rows = len(verified.shot_rows)
    width = (num_detectors + 7) // 8
    if len(data) != rows * width:
        raise ValueError("verified detector corpus byte length mismatch")
    detectors = np.frombuffer(data, dtype=np.uint8).reshape(rows, width)
    if num_detectors % 8 and rows and width:
        unused = 0xFF ^ ((1 << (num_detectors % 8)) - 1)
        if np.any(np.bitwise_and(detectors[:, -1], unused)):
            raise ValueError("verified detector corpus has nonzero tail bits")
    detectors.setflags(write=False)
    return detectors, digest


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot enter latency provenance")
    return {
        "sha256": _sha256(array.tobytes(order="C")),
        "shape": [int(size) for size in array.shape],
        "dtype": array.dtype.str,
    }


def _packed_prediction_array(
    value: object,
    *,
    rows: int,
    num_observables: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    expected_shape = (rows, (num_observables + 7) // 8)
    if array.dtype != np.uint8 or array.shape != expected_shape:
        raise ValueError(
            f"{name} must be packed uint8 with shape {expected_shape}, got "
            f"dtype={array.dtype}, shape={array.shape}"
        )
    array = np.ascontiguousarray(array)
    if num_observables % 8 and rows and expected_shape[1]:
        unused = 0xFF ^ ((1 << (num_observables % 8)) - 1)
        if np.any(np.bitwise_and(array[:, -1], unused)):
            raise ValueError(f"{name} has nonzero unused observable tail bits")
    array.setflags(write=False)
    return array


def _recorded_prediction_array(
    rows: tuple[Mapping[str, Any], ...],
    *,
    field: str,
    num_observables: int,
) -> np.ndarray:
    width = (num_observables + 7) // 8
    packed: list[bytes] = []
    for shot_id, row in enumerate(rows):
        encoded = row.get(field)
        if (
            not isinstance(encoded, str)
            or encoded.lower() != encoded
            or len(encoded) != 2 * width
        ):
            raise ValueError(f"recorded {field} is malformed at shot {shot_id}")
        try:
            value = bytes.fromhex(encoded)
        except ValueError as ex:
            raise ValueError(
                f"recorded {field} is not hexadecimal at shot {shot_id}"
            ) from ex
        packed.append(value)
    array = np.frombuffer(b"".join(packed), dtype=np.uint8).reshape(len(rows), width)
    return _packed_prediction_array(
        array,
        rows=len(rows),
        num_observables=num_observables,
        name=f"recorded {field}",
    )


def _validated_control_equality(
    value: Mapping[str, Any], *, shots: int
) -> dict[str, Any]:
    expected_names = {
        "ordinary_treatment_vs_telemetry",
        "global_vs_adapter_control",
        "global_vs_uf_shadow",
    }
    normalized = _canonical_mapping(value, name="collection control equality")
    if set(normalized) != expected_names:
        raise ValueError("collection control-equality ledger names are malformed")
    expected_fields = {
        "shots",
        "equal",
        "mismatches",
        "ordered_range_evidence_sha256",
    }
    for name, evidence in normalized.items():
        if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
            raise ValueError(f"collection control-equality ledger {name!r} is malformed")
        evidence_shots = _nonnegative_int(
            evidence.get("shots"), name=f"collection control {name} shots"
        )
        evidence_equal = _nonnegative_int(
            evidence.get("equal"), name=f"collection control {name} equal"
        )
        evidence_mismatches = _nonnegative_int(
            evidence.get("mismatches"),
            name=f"collection control {name} mismatches",
        )
        if (
            evidence_shots != shots
            or evidence_equal != shots
            or evidence_mismatches != 0
        ):
            raise ValueError(f"collection control-equality ledger {name!r} does not match")
        digest = evidence.get("ordered_range_evidence_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"collection control-equality ledger {name!r} digest is malformed"
            )
    return normalized


def _full_corpus_prediction_attestation(
    *,
    prepared: PreparedCell,
    detectors: np.ndarray,
    residuals: np.ndarray,
    source_rows: tuple[Mapping[str, Any], ...],
    control_equality: Mapping[str, Any],
    num_observables: int,
) -> dict[str, Any]:
    """Runs the two complete matchers once and binds them to collection rows."""

    shots = len(source_rows)
    controls = _validated_control_equality(control_equality, shots=shots)
    recorded_global = _recorded_prediction_array(
        source_rows,
        field="global_prediction_hex",
        num_observables=num_observables,
    )
    recorded_treatment = _recorded_prediction_array(
        source_rows,
        field="treatment_prediction_hex",
        num_observables=num_observables,
    )
    original_matcher = getattr(
        prepared.global_decoder, "invoke_backend_prevalidated", None
    )
    residual_matcher = getattr(
        prepared.treatment_decoder, "invoke_backend_prevalidated", None
    )
    if not callable(original_matcher) or not callable(residual_matcher):
        raise TypeError("prepared decoders omit complete matcher invocation")
    matched_original = _packed_prediction_array(
        original_matcher(detectors),
        rows=shots,
        num_observables=num_observables,
        name="full original-corpus matcher prediction",
    )
    matched_residual = _packed_prediction_array(
        residual_matcher(residuals),
        rows=shots,
        num_observables=num_observables,
        name="full residual-corpus matcher prediction",
    )
    original_equal = np.all(matched_original == recorded_global, axis=1)
    residual_equal = np.all(matched_residual == recorded_treatment, axis=1)
    if not bool(np.all(original_equal)):
        raise ValueError("full original-corpus matcher differs from recorded Global prediction")
    if not bool(np.all(residual_equal)):
        raise ValueError("full residual-corpus matcher differs from recorded treatment prediction")
    attestation = {
        "schema": FULL_CORPUS_ATTESTATION_SCHEMA,
        "shot_count": shots,
        "collection_control_equality": controls,
        "collection_control_equality_sha256": _digest_mapping(controls),
        "original_corpus": _array_identity(detectors),
        "residual_corpus": _array_identity(residuals),
        "recorded_global_prediction": _array_identity(recorded_global),
        "full_matcher_original_prediction": _array_identity(matched_original),
        "original_prediction_equality": {
            "equal": int(np.count_nonzero(original_equal)),
            "mismatches": int(shots - np.count_nonzero(original_equal)),
        },
        "recorded_treatment_prediction": _array_identity(recorded_treatment),
        "full_matcher_residual_prediction": _array_identity(matched_residual),
        "residual_prediction_equality": {
            "equal": int(np.count_nonzero(residual_equal)),
            "mismatches": int(shots - np.count_nonzero(residual_equal)),
        },
    }
    return _canonical_mapping(attestation, name="full-corpus prediction attestation")


def _validate_full_corpus_prediction_attestation(
    provenance: Mapping[str, Any], corpus: AuthenticatedLatencyCorpus
) -> str:
    attestation = provenance.get("full_corpus_prediction_attestation")
    if not isinstance(attestation, Mapping):
        raise ValueError("latency corpus omits full-corpus prediction attestation")
    normalized = _canonical_mapping(
        attestation, name="full-corpus prediction attestation"
    )
    claimed = provenance.get("full_corpus_prediction_attestation_sha256")
    actual = _digest_mapping(normalized)
    if claimed != actual:
        raise ValueError("full-corpus prediction attestation digest mismatch")
    expected_fields = {
        "schema",
        "shot_count",
        "collection_control_equality",
        "collection_control_equality_sha256",
        "original_corpus",
        "residual_corpus",
        "recorded_global_prediction",
        "full_matcher_original_prediction",
        "original_prediction_equality",
        "recorded_treatment_prediction",
        "full_matcher_residual_prediction",
        "residual_prediction_equality",
    }
    if set(normalized) != expected_fields or normalized.get("schema") != (
        FULL_CORPUS_ATTESTATION_SCHEMA
    ):
        raise ValueError("full-corpus prediction attestation fields are malformed")
    shots = corpus.row_count
    if normalized.get("shot_count") != shots:
        raise ValueError("full-corpus prediction attestation shot count mismatch")
    controls = _validated_control_equality(
        normalized["collection_control_equality"], shots=shots
    )
    if normalized.get("collection_control_equality_sha256") != _digest_mapping(
        controls
    ):
        raise ValueError("collection control-equality attestation digest mismatch")
    if normalized.get("original_corpus") != _array_identity(corpus.detectors):
        raise ValueError("full-corpus original detector attestation mismatch")
    if normalized.get("residual_corpus") != _array_identity(corpus.residuals):
        raise ValueError("full-corpus residual detector attestation mismatch")
    for recorded, matched, equality in (
        (
            "recorded_global_prediction",
            "full_matcher_original_prediction",
            "original_prediction_equality",
        ),
        (
            "recorded_treatment_prediction",
            "full_matcher_residual_prediction",
            "residual_prediction_equality",
        ),
    ):
        if normalized.get(recorded) != normalized.get(matched):
            raise ValueError("full-corpus prediction attestation digest disagreement")
        if normalized.get(equality) != {"equal": shots, "mismatches": 0}:
            raise ValueError("full-corpus prediction attestation equality mismatch")
    return actual


def materialize_latency_corpus_from_characterization(
    protocol: Mapping[str, Any],
    *,
    characterization_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    scientific: bool = True,
    processes: int = RANGE_COUNT,
    prepared: PreparedCell | None = None,
) -> Path:
    """Projects verified immutable decisions into detector/residual timing data."""

    normalized = _validate_protocol_identity(protocol)
    source = Path(characterization_dir)
    verified = verify_collection(
        normalized,
        stage=CHARACTERIZATION_STAGE,
        out=source,
        processes=processes,
        scientific=scientific,
    )
    if prepared is None:
        prepared = prepare_selected_cell(
            normalized,
            stage=CHARACTERIZATION_STAGE,
            processes=processes,
            scientific=scientific,
        )
    identity = _prepared_identity(normalized, prepared)
    if prepared.cell["cell_id"] != normalized["selected_cell"]["cell_id"]:
        raise ValueError("prepared selected cell differs from protocol")
    if int(prepared.provenance["num_detectors"]) <= 0:
        raise ValueError("prepared detector count is invalid")
    recorded_provenance = verified.summary.get("provenance")
    if not isinstance(recorded_provenance, Mapping):
        raise ValueError("verified collection provenance is malformed")
    for name in (
        "circuit_sha256",
        "dem_sha256",
        "layout_fingerprint",
        "graph_fingerprint",
        "validated_catalog_fingerprint",
        "projection_fingerprint",
        "num_detectors",
        "num_observables",
    ):
        if recorded_provenance.get(name) != prepared.provenance[name]:
            raise ValueError(
                f"prepared selected-cell provenance {name!r} differs from collection"
            )
    detectors, detector_digest = _read_verified_detectors(
        collection_dir=source,
        verified=verified,
        num_detectors=int(prepared.provenance["num_detectors"]),
    )
    source_rows = tuple(verified.shot_rows)
    global_ids = tuple(int(row["global_shot_id"]) for row in source_rows)
    if global_ids != tuple(range(len(source_rows))):
        raise ValueError(
            "characterization global shot IDs are not the corpus bijection"
        )
    if len(detectors) != len(source_rows):
        raise ValueError("verified detector/shot row counts differ")

    lane_groups = _group_authenticated_rows(
        verified.lane_rows,
        shots=len(source_rows),
        name="lane",
    )
    component_groups = _group_authenticated_rows(
        verified.component_rows,
        shots=len(source_rows),
        name="component",
    )
    residual_rows: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    num_detectors = int(prepared.provenance["num_detectors"])
    num_observables = int(prepared.provenance["num_observables"])
    projection_fingerprint = prepared.provenance["projection_fingerprint"]
    if not isinstance(projection_fingerprint, str) or not projection_fingerprint:
        raise ValueError("prepared projection fingerprint is invalid")
    for shot_id, shot in enumerate(source_rows):
        residual, summary = _materialize_authenticated_shot(
            detector=detectors[shot_id],
            shot=shot,
            lanes=lane_groups[shot_id],
            components=component_groups[shot_id],
            num_detectors=num_detectors,
            num_observables=num_observables,
            projection_fingerprint=projection_fingerprint,
        )
        if summary["global_shot_id"] != shot_id:
            raise ValueError("shot summary ID differs from detector corpus row")
        residual_rows.append(residual)
        summaries.append(summary)
    residuals = np.stack(residual_rows, axis=0)
    residuals.setflags(write=False)
    full_corpus_attestation = _full_corpus_prediction_attestation(
        prepared=prepared,
        detectors=detectors,
        residuals=residuals,
        source_rows=source_rows,
        control_equality=verified.control_equality,
        num_observables=num_observables,
    )
    provenance = {
        "schema": MATERIALIZATION_SCHEMA,
        **identity,
        "source_detector_sha256": detector_digest,
        "source_collection_summary_payload_sha256": verified.summary[
            "payload_sha256"
        ],
        "shot_count": len(detectors),
        "summary_fields": list(_SUMMARY_FIELDS),
        "full_corpus_prediction_attestation": full_corpus_attestation,
        "full_corpus_prediction_attestation_sha256": _digest_mapping(
            full_corpus_attestation
        ),
    }
    # Deliberately do not include VerifiedCollection.corpus_identity: it also
    # authenticates the observable file, which timing artifacts must not name.
    return write_authenticated_latency_corpus(
        out_dir,
        detectors=detectors,
        residuals=residuals,
        num_detectors=num_detectors,
        global_shot_ids=global_ids,
        summaries=summaries,
        provenance=provenance,
        corpus_digest=detector_digest,
    )


def _validate_materialized_identity(
    protocol: Mapping[str, Any], corpus: AuthenticatedLatencyCorpus
) -> dict[str, Any]:
    provenance = json.loads(corpus.provenance_json)
    if provenance.get("schema") != MATERIALIZATION_SCHEMA:
        raise ValueError("latency corpus materialization schema mismatch")
    expected = {
        "experiment_id": protocol["experiment_id"],
        "protocol_self_sha256": protocol["protocol_self_sha256"],
        "cell_id": protocol["selected_cell"]["cell_id"],
        "decoder_config_sha256": _digest_mapping(protocol["decoder"]),
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"latency corpus {name!r} differs from protocol")
    if provenance.get("source_detector_sha256") != corpus.corpus_digest:
        raise ValueError("latency corpus detector identity is inconsistent")
    _validate_full_corpus_prediction_attestation(provenance, corpus)
    return provenance


@dataclasses.dataclass(frozen=True)
class YokedPatchUFLatencyFactory:
    """Pickleable fresh-worker factory rebuilding the real selected YSC cell."""

    protocol: Mapping[str, Any]
    corpus_manifest_path: str
    scientific: bool = True
    processes: int = RANGE_COUNT
    stage: str = CHARACTERIZATION_STAGE

    def __post_init__(self) -> None:
        protocol = _validate_protocol_identity(self.protocol)
        if self.stage != CHARACTERIZATION_STAGE:
            raise ValueError("latency factory requires the characterization stage")
        processes = _positive_int(self.processes, name="processes")
        corpus = load_authenticated_latency_corpus(self.corpus_manifest_path)
        _validate_materialized_identity(protocol, corpus)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(
            self,
            "corpus_manifest_path",
            str(Path(self.corpus_manifest_path)),
        )
        object.__setattr__(self, "processes", processes)

    @property
    def suite_identity(self) -> dict[str, Any]:
        corpus = load_authenticated_latency_corpus(self.corpus_manifest_path)
        provenance = _validate_materialized_identity(self.protocol, corpus)
        return {
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "experiment_id": self.protocol["experiment_id"],
            "protocol_self_sha256": self.protocol["protocol_self_sha256"],
            "cell_id": self.protocol["selected_cell"]["cell_id"],
            "decoder_config_sha256": provenance["decoder_config_sha256"],
            "circuit_sha256": provenance["circuit_sha256"],
            "dem_sha256": provenance["dem_sha256"],
            "layout_fingerprint": provenance["layout_fingerprint"],
            "graph_fingerprint": provenance["graph_fingerprint"],
            "validated_catalog_fingerprint": provenance[
                "validated_catalog_fingerprint"
            ],
            "projection_fingerprint": provenance["projection_fingerprint"],
            "full_corpus_prediction_attestation_sha256": provenance[
                "full_corpus_prediction_attestation_sha256"
            ],
        }

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        if (
            isinstance(restart_index, bool)
            or not isinstance(restart_index, int)
            or restart_index < 0
        ):
            raise ValueError("restart_index must be a nonnegative integer")
        _positive_int(batch_size, name="batch_size")
        corpus = load_authenticated_latency_corpus(self.corpus_manifest_path)
        materialized = _validate_materialized_identity(self.protocol, corpus)
        prepared = prepare_selected_cell(
            self.protocol,
            stage=self.stage,
            processes=self.processes,
            scientific=self.scientific,
        )
        rebuilt = _prepared_identity(self.protocol, prepared)
        for name, value in rebuilt.items():
            if materialized.get(name) != value:
                raise ValueError(
                    f"rebuilt selected-cell identity {name!r} differs from corpus"
                )
        if corpus.num_detectors != int(prepared.provenance["num_detectors"]):
            raise ValueError("latency corpus detector count differs from rebuilt cell")
        variants = build_timed_variants(
            global_mwpm=prepared.global_decoder,
            adapter_control=prepared.control_decoder,
            uf_shadow=prepared.shadow_decoder,
            treatment=prepared.treatment_decoder,
        )
        return LatencyWorkload(
            corpus=corpus,
            variants=variants,
            provenance={
                "restart_index": restart_index,
                "batch_size": batch_size,
                "full_corpus_prediction_attestation_sha256": materialized[
                    "full_corpus_prediction_attestation_sha256"
                ],
                **rebuilt,
            },
        )


def latency_protocol_from_experiment(
    protocol: Mapping[str, Any],
    *,
    cpu: int = 31,
) -> LatencyProtocol:
    """Maps frozen experiment timing literals to the generic harness protocol."""

    normalized = _validate_protocol_identity(protocol)
    timing = normalized.get("latency")
    if not isinstance(timing, Mapping):
        raise ValueError("experiment protocol requires a latency object")
    rows = timing.get("batches")
    if not isinstance(rows, list) or not rows:
        raise ValueError("latency batches must be a nonempty list")
    expected_fields = {
        "batch_size",
        "restarts",
        "blocks_per_restart",
        "warmup_calls_per_variant",
        "timed_calls_per_side_per_block",
    }
    batches: list[BatchTiming] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("latency batch fields are malformed")
        batches.append(BatchTiming(**dict(row)))
    seed_root = timing.get("schedule_seed")
    if (
        not isinstance(seed_root, str)
        or len(seed_root) != 64
        or any(character not in "0123456789abcdef" for character in seed_root)
    ):
        raise ValueError("latency schedule_seed must be 64 lowercase hex digits")
    frozen_host = timing.get("host_policy")
    if frozen_host is None:
        host = capture_host_policy(cpu=cpu)
    else:
        if not isinstance(frozen_host, Mapping) or set(frozen_host) != {
            "cpu_affinity",
            "expected_host",
            "expected_numa_nodes",
        }:
            raise ValueError("latency host_policy fields are malformed")
        host = HostPolicy(
            cpu_affinity=tuple(frozen_host["cpu_affinity"]),
            expected_host=tuple(sorted(dict(frozen_host["expected_host"]).items())),
            expected_numa_nodes=tuple(frozen_host["expected_numa_nodes"]),
        )
        if host.cpu_affinity != (cpu,):
            raise ValueError("requested timing CPU differs from frozen host policy")
    return LatencyProtocol(
        batches=tuple(batches),
        schedule_seed=int(seed_root, 16),
        host_policy=host,
    )


def verify_latency_factory(factory: YokedPatchUFLatencyFactory) -> dict[str, Any]:
    """Rebuilds once and returns the reconciled detector/projection identity."""

    if not isinstance(factory, YokedPatchUFLatencyFactory):
        raise TypeError("factory must be YokedPatchUFLatencyFactory")
    workload = factory(0, 1)
    identity = factory.suite_identity
    if workload.corpus.manifest_sha256 != identity["corpus_manifest_sha256"]:
        raise ValueError("factory workload differs from its suite identity")
    return identity


__all__ = [
    "FULL_CORPUS_ATTESTATION_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "YokedPatchUFLatencyFactory",
    "latency_protocol_from_experiment",
    "materialize_latency_corpus_from_characterization",
    "verify_latency_factory",
]
