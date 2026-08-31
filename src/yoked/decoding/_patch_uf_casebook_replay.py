"""Authenticated deterministic casebook replay in a fresh spawned process."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import dataclasses
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from yoked.decoding._artifact_io import is_lowercase_hex, load_json_strict
from yoked.decoding._patch_uf_analysis import (
    ANALYSIS_SCHEMA,
    ANALYSIS_SCHEMA_VERSION,
    CASEBOOK_CATEGORIES,
)
from yoked.decoding._patch_uf_decoder import CaptureMode
from yoked.decoding._patch_uf_experiment import (
    CHARACTERIZATION_STAGE,
    PROTOCOL_SCHEMA,
    RANGE_COUNT,
    VerifiedCollection,
    _normalize_metrics,
    canonical_protocol_self_sha256,
    configure_single_thread_runtime,
    prepare_selected_cell,
    verify_collection,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


REPLAY_SCHEMA = "patch-uf-casebook-replay-v1"
REPLAY_SCHEMA_VERSION = 1
MAXIMUM_CASES_PER_CATEGORY = 100
MAXIMUM_REPLAY_CASES = len(CASEBOOK_CATEGORIES) * MAXIMUM_CASES_PER_CATEGORY
_METRIC_CATEGORIES = frozenset(
    {
        "largest-final-component",
        "largest-committed-component",
        "largest-censored-partial-lower-bound",
        "highest-heap-operation-count",
    }
)
_CATEGORY_FIELDS = frozenset(
    {
        "candidate_shots",
        "retained_shots",
        "maximum_retained",
        "selection",
        "rows",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "global_shot_id",
        "metric",
        "selection_sha256",
        "shot",
        "lanes",
        "components",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _canonical_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} is not canonical-JSON compatible") from ex


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = _canonical_mapping(protocol, name="protocol")
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("casebook replay protocol schema/version mismatch")
    claimed = value.get("protocol_self_sha256")
    if not is_lowercase_hex(claimed, length=64):
        raise ValueError("casebook replay protocol self hash is malformed")
    if claimed != canonical_protocol_self_sha256(value):
        raise ValueError("casebook replay protocol self hash mismatch")
    return value


def _payload_digest(value: Mapping[str, Any], *, name: str) -> str:
    claimed = value.get("payload_sha256")
    if not is_lowercase_hex(claimed, length=64):
        raise ValueError(f"{name} payload digest is malformed")
    unsigned = dict(value)
    del unsigned["payload_sha256"]
    if _sha256(canonical_json_bytes(unsigned)) != claimed:
        raise ValueError(f"{name} payload digest mismatch")
    return claimed


def _load_analysis(path: Path) -> dict[str, Any]:
    value = load_json_strict(path, description="Patch-UF casebook analysis")
    if path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("casebook analysis must use exact canonical JSON bytes")
    if (
        value.get("schema") != ANALYSIS_SCHEMA
        or value.get("schema_version") != ANALYSIS_SCHEMA_VERSION
    ):
        raise ValueError("casebook analysis schema/version mismatch")
    _payload_digest(value, name="casebook analysis")
    return value


def _read_detector_corpus(
    collection_out: Path,
    *,
    verified: VerifiedCollection,
    num_detectors: int,
) -> tuple[np.ndarray, str]:
    path = collection_out / "corpus" / "detectors.bitpack"
    if path.is_symlink() or not path.is_file():
        raise ValueError("casebook detector corpus must be a regular file")
    raw = path.read_bytes()
    digest = _sha256(raw)
    identity = _mapping(verified.corpus_identity, name="collection corpus identity")
    if identity.get("detectors_sha256") != digest:
        raise ValueError("casebook detector corpus digest mismatch")
    rows = len(verified.shot_rows)
    width = (num_detectors + 7) // 8
    if len(raw) != rows * width:
        raise ValueError("casebook detector corpus byte length mismatch")
    detectors = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width)
    if num_detectors % 8 and rows and width:
        unused = 0xFF ^ ((1 << (num_detectors % 8)) - 1)
        if np.any(np.bitwise_and(detectors[:, -1], unused)):
            raise ValueError("casebook detector corpus has nonzero tail bits")
    detectors.setflags(write=False)
    return detectors, digest


def _rows_by_shot(
    rows: tuple[Mapping[str, Any], ...],
    *,
    shots: int,
    name: str,
) -> dict[int, list[Mapping[str, Any]]]:
    result = {shot_id: [] for shot_id in range(shots)}
    for row in rows:
        row = _mapping(row, name=f"verified {name} row")
        shot_id = _integer(row.get("global_shot_id"), name=f"{name} shot ID")
        if shot_id >= shots:
            raise ValueError(f"verified {name} row shot ID is out of range")
        result[shot_id].append(row)
    return result


def _selection_digest(root: bytes, category: str, shot_id: int) -> str:
    return hashlib.sha256(
        root
        + b"patch-uf-casebook-v1\0"
        + category.encode("utf-8")
        + b"\0"
        + str(shot_id).encode("ascii")
    ).hexdigest()


@dataclasses.dataclass(frozen=True)
class CasebookReplayCase:
    global_shot_id: int
    categories: tuple[str, ...]
    packed_detector: bytes
    global_prediction_hex: str
    treatment_prediction_hex: str
    adapter_metrics: Mapping[str, Any]
    lanes: tuple[Mapping[str, Any], ...]
    components: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        shot_id = _integer(self.global_shot_id, name="replay global_shot_id")
        categories = tuple(self.categories)
        if (
            not categories
            or categories != tuple(sorted(set(categories)))
            or any(category not in CASEBOOK_CATEGORIES for category in categories)
        ):
            raise ValueError("replay case categories are malformed")
        detector = bytes(self.packed_detector)
        if not detector:
            raise ValueError("replay packed detector must be nonempty")
        for name in ("global_prediction_hex", "treatment_prediction_hex"):
            value = getattr(self, name)
            if not isinstance(value, str) or value.lower() != value:
                raise ValueError(f"replay {name} must be lowercase hexadecimal")
            try:
                bytes.fromhex(value)
            except ValueError as ex:
                raise ValueError(f"replay {name} is not hexadecimal") from ex
        metrics = _canonical_mapping(self.adapter_metrics, name="adapter metrics")
        lanes = tuple(
            _canonical_mapping(row, name="lane row") for row in self.lanes
        )
        components = tuple(
            _canonical_mapping(row, name="component row")
            for row in self.components
        )
        object.__setattr__(self, "global_shot_id", shot_id)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "packed_detector", detector)
        object.__setattr__(self, "adapter_metrics", metrics)
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "components", components)


@dataclasses.dataclass(frozen=True)
class CasebookReplayRequest:
    protocol: Mapping[str, Any]
    stage: str
    processes: int
    scientific: bool
    parent_pid: int
    expected_provenance: Mapping[str, Any]
    collection_payload_sha256: str
    detector_corpus_sha256: str
    analysis_payload_sha256: str
    casebook_selection_sha256: str
    cases: tuple[CasebookReplayCase, ...]

    def __post_init__(self) -> None:
        protocol = _validate_protocol(self.protocol)
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("replay stage must be a nonempty string")
        if not isinstance(self.scientific, bool):
            raise TypeError("replay scientific flag must be boolean")
        processes = _integer(self.processes, name="replay processes", minimum=1)
        parent_pid = _integer(self.parent_pid, name="replay parent PID", minimum=1)
        provenance = _canonical_mapping(
            self.expected_provenance, name="expected selected-cell provenance"
        )
        for name in (
            "collection_payload_sha256",
            "detector_corpus_sha256",
            "analysis_payload_sha256",
            "casebook_selection_sha256",
        ):
            if not is_lowercase_hex(getattr(self, name), length=64):
                raise ValueError(f"replay {name} is malformed")
        cases = tuple(self.cases)
        if (
            not cases
            or len(cases) > MAXIMUM_REPLAY_CASES
            or any(not isinstance(case, CasebookReplayCase) for case in cases)
        ):
            raise ValueError("replay request has an invalid bounded case count")
        ids = tuple(case.global_shot_id for case in cases)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("replay request cases must use sorted unique shot IDs")
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "processes", processes)
        object.__setattr__(self, "parent_pid", parent_pid)
        object.__setattr__(self, "expected_provenance", provenance)
        object.__setattr__(self, "cases", cases)


def _strip_global_id(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("global_shot_id", None)
    return result


def build_authenticated_casebook_replay_request(
    protocol: Mapping[str, Any],
    *,
    collection_out: str | os.PathLike[str],
    analysis_path: str | os.PathLike[str],
    stage: str = CHARACTERIZATION_STAGE,
    processes: int = RANGE_COUNT,
    scientific: bool = True,
) -> CasebookReplayRequest:
    """Authenticates collection/analysis and builds an observable-free request."""

    normalized = _validate_protocol(protocol)
    collection = Path(collection_out)
    verified = verify_collection(
        normalized,
        stage=stage,
        out=collection,
        processes=processes,
        scientific=scientific,
    )
    analysis = _load_analysis(Path(analysis_path))
    source = _mapping(analysis.get("source"), name="analysis source")
    expected_source = {
        "experiment_id": normalized["experiment_id"],
        "protocol_self_sha256": normalized["protocol_self_sha256"],
        "collection_payload_sha256": verified.summary["payload_sha256"],
        "stage": stage,
        "cell_id": normalized["selected_cell"]["cell_id"],
        "corpus_identity": verified.corpus_identity,
    }
    if dict(source) != expected_source:
        raise ValueError("casebook analysis source differs from verified collection")
    provenance = _mapping(
        verified.summary.get("provenance"), name="collection provenance"
    )
    num_detectors = _integer(
        provenance.get("num_detectors"), name="collection num_detectors", minimum=1
    )
    detectors, detector_digest = _read_detector_corpus(
        collection,
        verified=verified,
        num_detectors=num_detectors,
    )
    shots = len(verified.shot_rows)
    shot_by_id = {
        _integer(row.get("global_shot_id"), name="verified shot ID"): row
        for row in verified.shot_rows
    }
    if tuple(sorted(shot_by_id)) != tuple(range(shots)):
        raise ValueError("verified shot rows are not a dense corpus bijection")
    lanes_by_id = _rows_by_shot(verified.lane_rows, shots=shots, name="lane")
    components_by_id = _rows_by_shot(
        verified.component_rows, shots=shots, name="component"
    )

    config = _mapping(analysis.get("config"), name="analysis config")
    cap = _integer(
        config.get("maximum_cases_per_category"),
        name="maximum_cases_per_category",
        minimum=1,
    )
    if cap > MAXIMUM_CASES_PER_CATEGORY:
        raise ValueError("casebook per-category cap exceeds the replay bound")
    root_hex = config.get("casebook_seed_root")
    if not is_lowercase_hex(root_hex, length=64):
        raise ValueError("analysis casebook seed root is malformed")
    if scientific:
        diagnostics = _mapping(
            normalized.get("diagnostics"), name="protocol diagnostics"
        )
        if diagnostics.get("maximum_cases_per_category") != cap:
            raise ValueError("analysis casebook cap differs from frozen protocol")
        if diagnostics.get("replay_selection_seed_root") != root_hex:
            raise ValueError("analysis casebook seed differs from frozen protocol")
    reconciliation = _mapping(
        analysis.get("reconciliation"), name="analysis reconciliation"
    )
    if (
        reconciliation.get("status") != "reconciled"
        or reconciliation.get("shots") != shots
    ):
        raise ValueError("casebook analysis did not reconcile the collection")
    root = bytes.fromhex(root_hex)
    casebook = _mapping(analysis.get("casebook"), name="analysis casebook")
    if set(casebook) != set(CASEBOOK_CATEGORIES):
        raise ValueError("analysis casebook category set mismatch")

    categories_by_id: dict[int, set[str]] = {}
    for category in CASEBOOK_CATEGORIES:
        category_row = _mapping(casebook[category], name=f"casebook {category}")
        if set(category_row) != _CATEGORY_FIELDS:
            raise ValueError(f"casebook {category} fields are malformed")
        rows = category_row.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"casebook {category} rows must be a list")
        retained = _integer(
            category_row.get("retained_shots"), name=f"{category} retained shots"
        )
        maximum = _integer(
            category_row.get("maximum_retained"),
            name=f"{category} maximum retained",
            minimum=1,
        )
        if retained != len(rows) or maximum != cap or retained > cap:
            raise ValueError(f"casebook {category} bound/count mismatch")
        candidates = _integer(
            category_row.get("candidate_shots"),
            name=f"{category} candidate shots",
        )
        if candidates < retained:
            raise ValueError(f"casebook {category} candidates are below retained")
        expected_selection = (
            "metric-descending-then-rooted-sha256"
            if category in _METRIC_CATEGORIES
            else "rooted-sha256-ascending"
        )
        if category_row.get("selection") != expected_selection:
            raise ValueError(f"casebook {category} selection policy mismatch")
        seen: set[int] = set()
        for selected in rows:
            selected = _mapping(selected, name=f"casebook {category} selection")
            if set(selected) != _SELECTION_FIELDS:
                raise ValueError(f"casebook {category} selection fields are malformed")
            shot_id = _integer(
                selected.get("global_shot_id"), name=f"{category} shot ID"
            )
            if shot_id not in shot_by_id or shot_id in seen:
                raise ValueError(f"casebook {category} shot identity is invalid")
            seen.add(shot_id)
            expected_digest = _selection_digest(root, category, shot_id)
            if selected.get("selection_sha256") != expected_digest:
                raise ValueError(f"casebook {category} selection digest mismatch")
            if selected.get("shot") != shot_by_id[shot_id]:
                raise ValueError(
                    f"casebook {category} shot row differs from collection"
                )
            if selected.get("lanes") != lanes_by_id[shot_id]:
                raise ValueError(
                    f"casebook {category} lane rows differ from collection"
                )
            if selected.get("components") != components_by_id[shot_id]:
                raise ValueError(
                    f"casebook {category} component rows differ from collection"
                )
            categories_by_id.setdefault(shot_id, set()).add(category)
    if not categories_by_id:
        raise ValueError("analysis casebook selected no replay cases")
    if len(categories_by_id) > MAXIMUM_REPLAY_CASES:
        raise ValueError("analysis casebook exceeds the aggregate replay bound")

    cases = []
    for shot_id in sorted(categories_by_id):
        shot = shot_by_id[shot_id]
        metrics = _mapping(shot.get("adapter_metrics"), name="shot adapter metrics")
        cases.append(
            CasebookReplayCase(
                global_shot_id=shot_id,
                categories=tuple(sorted(categories_by_id[shot_id])),
                packed_detector=bytes(detectors[shot_id]),
                global_prediction_hex=str(shot.get("global_prediction_hex")),
                treatment_prediction_hex=str(shot.get("treatment_prediction_hex")),
                adapter_metrics=metrics,
                lanes=tuple(_strip_global_id(row) for row in lanes_by_id[shot_id]),
                components=tuple(
                    _strip_global_id(row) for row in components_by_id[shot_id]
                ),
            )
        )
    return CasebookReplayRequest(
        protocol=normalized,
        stage=stage,
        processes=processes,
        scientific=scientific,
        parent_pid=os.getpid(),
        expected_provenance=provenance,
        collection_payload_sha256=str(verified.summary["payload_sha256"]),
        detector_corpus_sha256=detector_digest,
        analysis_payload_sha256=str(analysis["payload_sha256"]),
        casebook_selection_sha256=_sha256(canonical_json_bytes(dict(casebook))),
        cases=tuple(cases),
    )


def _decode_hex(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("replay decoder returned malformed packed predictions")
    return bytes(array[0]).hex()


def _first_difference(expected: object, actual: object, *, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return f"{path} type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, Mapping):
        if set(expected) != set(actual):
            return f"{path} keys {sorted(expected)} != {sorted(actual)}"
        for key in sorted(expected):
            difference = _first_difference(
                expected[key], actual[key], path=f"{path}.{key}"
            )
            if difference:
                return difference
        return ""
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return f"{path} length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _first_difference(
                left, right, path=f"{path}[{index}]"
            )
            if difference:
                return difference
        return ""
    return "" if expected == actual else f"{path} {expected!r} != {actual!r}"


def run_casebook_replay_worker(request: CasebookReplayRequest) -> dict[str, Any]:
    """Spawn-worker entry point; callers should use the fresh-process wrapper."""

    if not isinstance(request, CasebookReplayRequest):
        raise TypeError("request must be CasebookReplayRequest")
    if os.getpid() == request.parent_pid:
        raise RuntimeError("casebook replay worker did not run in a fresh process")
    configure_single_thread_runtime()
    prepared = prepare_selected_cell(
        request.protocol,
        stage=request.stage,
        processes=request.processes,
        scientific=request.scientific,
    )
    if dict(prepared.provenance) != dict(request.expected_provenance):
        raise ValueError("replay selected-cell provenance mismatch")
    expected_lanes = _integer(
        request.protocol["collection_limits"].get("expected_lanes_per_shot"),
        name="expected lanes per shot",
        minimum=1,
    )
    max_components = _integer(
        request.protocol["collection_limits"].get(
            "maximum_component_records_per_shot"
        ),
        name="maximum components per shot",
        minimum=1,
    )
    max_metric_bytes = _integer(
        request.protocol["collection_limits"].get(
            "maximum_metric_bytes_per_range"
        ),
        name="maximum metric bytes",
        minimum=1,
    )
    width = (int(prepared.provenance["num_detectors"]) + 7) // 8
    case_summaries: list[dict[str, Any]] = []
    for case in request.cases:
        if len(case.packed_detector) != width:
            raise ValueError("replay detector width differs from rebuilt cell")
        packed = np.frombuffer(case.packed_detector, dtype=np.uint8).reshape(1, width)
        global_prediction = prepared.global_decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed
        )
        treatment_prediction, corrections = (
            prepared.treatment_decoder.decode_shots_bit_packed_with_capture(
                bit_packed_detection_event_data=packed,
                capture=CaptureMode.METRICS,
            )
        )
        if len(corrections) != 1:
            raise ValueError("replay treatment returned the wrong correction count")
        if _decode_hex(global_prediction) != case.global_prediction_hex:
            raise ValueError(
                f"Global MWPM replay mismatch for shot {case.global_shot_id}"
            )
        if _decode_hex(treatment_prediction) != case.treatment_prediction_hex:
            raise ValueError(
                f"treatment replay mismatch for shot {case.global_shot_id}"
            )
        correction = corrections[0]
        residual, frame = prepared.treatment_decoder.apply_shot_correction(
            packed[0], correction
        )
        boundary = tuple(case.adapter_metrics.get("durable_detector_boundary", ()))
        expected_residual = np.array(packed[0], dtype=np.uint8, copy=True)
        for detector_id in boundary:
            detector_id = _integer(detector_id, name="durable detector ID")
            if detector_id >= int(prepared.provenance["num_detectors"]):
                raise ValueError("persisted durable detector ID is out of range")
            expected_residual[detector_id // 8] ^= np.uint8(
                1 << (detector_id % 8)
            )
        if not np.array_equal(residual, expected_residual):
            raise ValueError(
                f"durable residual replay mismatch for shot {case.global_shot_id}"
            )
        if bytes(frame).hex() != case.adapter_metrics.get("durable_observable_frame"):
            raise ValueError(
                f"durable frame replay mismatch for shot {case.global_shot_id}"
            )
        if tuple(correction.durable_support_edge_ids) != tuple(
            case.adapter_metrics.get("durable_support_edge_ids", ())
        ):
            raise ValueError(
                f"durable support replay mismatch for shot {case.global_shot_id}"
            )
        shot_rows, lane_groups, component_groups = _normalize_metrics(
            (correction,),
            shots=1,
            expected_lanes=expected_lanes,
            max_components_per_shot=max_components,
            max_metric_bytes=max_metric_bytes,
        )
        if shot_rows[0] != case.adapter_metrics:
            difference = _first_difference(case.adapter_metrics, shot_rows[0])
            raise ValueError(
                f"shot telemetry replay mismatch for shot {case.global_shot_id}: "
                f"{difference}"
            )
        if lane_groups[0] != list(case.lanes):
            raise ValueError(
                f"lane telemetry replay mismatch for shot {case.global_shot_id}"
            )
        if component_groups[0] != list(case.components):
            raise ValueError(
                f"component telemetry replay mismatch for shot {case.global_shot_id}"
            )
        telemetry = {
            "shot": shot_rows[0],
            "lanes": lane_groups[0],
            "components": component_groups[0],
        }
        case_summaries.append(
            {
                "global_shot_id": case.global_shot_id,
                "categories": list(case.categories),
                "detector_sha256": _sha256(case.packed_detector),
                "global_prediction_sha256": _sha256(bytes(global_prediction[0])),
                "treatment_prediction_sha256": _sha256(
                    bytes(treatment_prediction[0])
                ),
                "residual_sha256": _sha256(bytes(residual)),
                "durable_support_sha256": _sha256(
                    canonical_json_bytes(
                        {"edge_ids": list(correction.durable_support_edge_ids)}
                    )
                ),
                "telemetry_sha256": _sha256(canonical_json_bytes(telemetry)),
            }
        )
    summary = {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "reconciled",
        "fresh_process": True,
        "experiment_id": request.protocol["experiment_id"],
        "protocol_self_sha256": request.protocol["protocol_self_sha256"],
        "stage": request.stage,
        "cell_id": request.protocol["selected_cell"]["cell_id"],
        "collection_payload_sha256": request.collection_payload_sha256,
        "detector_corpus_sha256": request.detector_corpus_sha256,
        "analysis_payload_sha256": request.analysis_payload_sha256,
        "casebook_selection_sha256": request.casebook_selection_sha256,
        "replayed_cases": len(case_summaries),
        "cases": case_summaries,
    }
    summary["payload_sha256"] = _sha256(canonical_json_bytes(summary))
    return json.loads(canonical_json_bytes(summary))


def replay_casebook_request_fresh_process(
    request: CasebookReplayRequest,
    *,
    worker_processes: int = 1,
) -> dict[str, Any]:
    """Runs bounded request shards in fresh spawn workers and validates them."""

    if not isinstance(request, CasebookReplayRequest):
        raise TypeError("request must be CasebookReplayRequest")
    if request.parent_pid != os.getpid():
        raise ValueError("replay request belongs to a different parent process")
    workers = _integer(
        worker_processes, name="replay worker_processes", minimum=1
    )
    if workers > RANGE_COUNT:
        raise ValueError("replay worker_processes exceeds the hard maximum of 32")
    workers = min(workers, len(request.cases))
    shards = tuple(
        dataclasses.replace(
            request,
            cases=tuple(request.cases[index::workers]),
        )
        for index in range(workers)
    )
    configure_single_thread_runtime()
    for shard in shards:
        pickle.dumps(shard)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        futures = tuple(
            pool.submit(run_casebook_replay_worker, shard) for shard in shards
        )
        partials = tuple(future.result() for future in futures)
    expected_identity = {
        "experiment_id": request.protocol["experiment_id"],
        "protocol_self_sha256": request.protocol["protocol_self_sha256"],
        "stage": request.stage,
        "cell_id": request.protocol["selected_cell"]["cell_id"],
        "collection_payload_sha256": request.collection_payload_sha256,
        "detector_corpus_sha256": request.detector_corpus_sha256,
        "analysis_payload_sha256": request.analysis_payload_sha256,
        "casebook_selection_sha256": request.casebook_selection_sha256,
    }
    case_rows: list[dict[str, Any]] = []
    for partial in partials:
        partial = _canonical_mapping(partial, name="casebook replay shard summary")
        if (
            partial.get("schema") != REPLAY_SCHEMA
            or partial.get("status") != "reconciled"
            or partial.get("fresh_process") is not True
        ):
            raise ValueError("fresh-process casebook replay shard is malformed")
        _payload_digest(partial, name="casebook replay shard summary")
        for name, expected in expected_identity.items():
            if partial.get(name) != expected:
                raise ValueError(f"fresh-process casebook replay {name} mismatch")
        rows = partial.get("cases")
        if not isinstance(rows, list) or partial.get("replayed_cases") != len(rows):
            raise ValueError("fresh-process casebook replay shard count mismatch")
        case_rows.extend(rows)
    case_rows.sort(key=lambda row: row["global_shot_id"])
    expected_cases = [
        {
            "global_shot_id": case.global_shot_id,
            "categories": list(case.categories),
        }
        for case in request.cases
    ]
    if [
        {
            "global_shot_id": row.get("global_shot_id"),
            "categories": row.get("categories"),
        }
        for row in case_rows
        if isinstance(row, Mapping)
    ] != expected_cases:
        raise ValueError("fresh-process casebook replay case identities mismatch")
    summary = {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "reconciled",
        "fresh_process": True,
        "worker_processes": workers,
        **expected_identity,
        "replayed_cases": len(case_rows),
        "cases": case_rows,
    }
    summary["payload_sha256"] = _sha256(canonical_json_bytes(summary))
    return summary


def replay_casebook_in_fresh_process(
    protocol: Mapping[str, Any],
    *,
    collection_out: str | os.PathLike[str],
    analysis_path: str | os.PathLike[str],
    stage: str = CHARACTERIZATION_STAGE,
    processes: int = RANGE_COUNT,
    worker_processes: int = 1,
    scientific: bool = True,
) -> dict[str, Any]:
    """Authenticates, rebuilds, and bit-exactly replays the analysis casebook."""

    request = build_authenticated_casebook_replay_request(
        protocol,
        collection_out=collection_out,
        analysis_path=analysis_path,
        stage=stage,
        processes=processes,
        scientific=scientific,
    )
    return replay_casebook_request_fresh_process(
        request, worker_processes=worker_processes
    )


__all__ = [
    "CasebookReplayCase",
    "CasebookReplayRequest",
    "MAXIMUM_CASES_PER_CATEGORY",
    "MAXIMUM_REPLAY_CASES",
    "REPLAY_SCHEMA",
    "REPLAY_SCHEMA_VERSION",
    "build_authenticated_casebook_replay_request",
    "replay_casebook_in_fresh_process",
    "replay_casebook_request_fresh_process",
    "run_casebook_replay_worker",
]
