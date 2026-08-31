"""Exact-corpus four-arm accuracy and workload characterization.

This module projects an already authenticated patch-UF characterization corpus
into additive range ledgers for Global MWPM, native ProMatch, native Pinball V2,
and the recorded Union-Find treatment.  It deliberately owns no CLI, protocol,
or artifact-directory policy.  A caller first obtains a :class:`VerifiedCollection`
from the frozen patch-UF verifier, materializes a :class:`MatchedCorpus`, compiles
the three local arms once in the parent, and forks workers over the corpus's
exact 32 logical ranges.

The supplied detector and observable arrays are never resampled.  Global MWPM
is recomputed on every range and must match the prediction imported from the UF
collection bit for bit before a ledger can be emitted.
"""

from __future__ import annotations

from collections import Counter
import copy
import dataclasses
import hashlib
import io
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from yoked.decoding._artifact_io import install_bytes_atomic, load_json_strict
from yoked.decoding._patch_uf_experiment import (
    RANGE_COUNT,
    ShotRange,
    VerifiedCollection,
    fixed_worker_ranges,
)
from yoked.decoding._pinball_promatch_experiment import (
    _COMMON_TELEMETRY_KEYS,
    _PINBALL_TELEMETRY_KEYS,
    _PROMATCH_TELEMETRY_KEYS,
    _Telemetry,
    _decode_residual,
)
from yoked.decoding._promatch_stats import canonical_json_bytes, digest_array


LEDGER_SCHEMA = "yoked.pinball-promatch-uf-matched-range-v1"
AGGREGATE_SCHEMA = "yoked.pinball-promatch-uf-matched-aggregate-v1"
CORPUS_SCHEMA = "yoked.pinball-promatch-uf-matched-corpus-v1"
CORPUS_MANIFEST_SCHEMA = "yoked.pinball-promatch-uf-matched-corpus-manifest-v1"
ARM_ORDER = ("global", "promatch", "pinball", "union_find")
PAIR_DEFINITIONS = {
    f"{treatment}_minus_{baseline}": (baseline, treatment)
    for baseline, treatment in combinations(ARM_ORDER, 2)
}

_LEDGER_FIELDS = {
    "schema",
    "source_identity",
    "prepared_provenance",
    "range",
    "microbatch_size",
    "arm_order",
    "array_digests",
    "global_prediction_equality",
    "correctness_cube",
    "pairwise_contingencies",
    "prediction_agreement",
    "telemetry",
    "payload_sha256",
}
_ARRAY_NAMES = {
    "detectors",
    "actual_observables",
    "imported_global_predictions",
    "computed_global_predictions",
    "promatch_predictions",
    "pinball_predictions",
    "union_find_predictions",
    "imported_global_failures",
    "imported_union_find_failures",
    "union_find_residual_hw",
    "union_find_residual_body_hw",
    "union_find_residual_terminal_hw",
    "union_find_residual_yoke_hw",
}
_PAIR_FIELDS = {"both_correct", "regressions", "recoveries", "both_wrong"}
_AGREEMENT_FIELDS = {"agree", "disagree"}
_CORPUS_ARRAY_ATTRIBUTES = (
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
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _canonical_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    raw = _mapping(value, name=name)
    try:
        normalized = copy.deepcopy(dict(raw))
        canonical_json_bytes(normalized)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} is not canonical JSON data") from ex
    return normalized


def _digest_json(value: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(value))


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    result = digest_array(np.ascontiguousarray(value))
    return {
        "sha256": result.sha256,
        "shape": [int(size) for size in result.shape],
        "dtype": result.dtype,
    }


def _packed_array(
    value: Any,
    *,
    rows: int,
    bits: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    expected = (rows, (bits + 7) // 8)
    if array.dtype != np.uint8 or array.shape != expected:
        raise ValueError(
            f"{name} must be packed uint8 with shape {expected}, got "
            f"dtype={array.dtype}, shape={array.shape}"
        )
    array = np.ascontiguousarray(array)
    if bits % 8 and rows and expected[1]:
        unused = 0xFF ^ ((1 << (bits % 8)) - 1)
        if np.any(np.bitwise_and(array[:, -1], unused)):
            raise ValueError(f"{name} has nonzero unused tail bits")
    array.setflags(write=False)
    return array


def _hex_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    width: int,
) -> np.ndarray:
    values: list[bytes] = []
    for shot_id, row in enumerate(rows):
        encoded = row.get(field)
        if (
            not isinstance(encoded, str)
            or encoded.lower() != encoded
            or len(encoded) != 2 * width
        ):
            raise ValueError(f"{field} is malformed at shot {shot_id}")
        try:
            values.append(bytes.fromhex(encoded))
        except ValueError as ex:
            raise ValueError(f"{field} is not hexadecimal at shot {shot_id}") from ex
    return np.frombuffer(b"".join(values), dtype=np.uint8).reshape(len(rows), width)


def _failure_array(prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if prediction.shape != actual.shape:
        raise ValueError("prediction and actual observable arrays have different shapes")
    result = np.any(np.bitwise_xor(prediction, actual) != 0, axis=1)
    result.setflags(write=False)
    return result


@dataclasses.dataclass(frozen=True)
class MatchedCorpus:
    """Authenticated immutable arrays projected from a UF collection."""

    source_identity: Mapping[str, Any]
    detectors: np.ndarray
    actual_observables: np.ndarray
    global_predictions: np.ndarray
    union_find_predictions: np.ndarray
    global_failures: np.ndarray
    union_find_failures: np.ndarray
    union_find_residual_hw: np.ndarray
    union_find_residual_body_hw: np.ndarray
    union_find_residual_terminal_hw: np.ndarray
    union_find_residual_yoke_hw: np.ndarray
    source_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        identity = _canonical_mapping(self.source_identity, name="source_identity")
        provenance = _canonical_mapping(
            self.source_provenance, name="source_provenance"
        )
        shots = _positive_int(identity.get("shots"), name="source shots")
        num_detectors = _positive_int(
            identity.get("num_detectors"), name="source num_detectors"
        )
        num_observables = _positive_int(
            identity.get("num_observables"), name="source num_observables"
        )
        detectors = _packed_array(
            self.detectors,
            rows=shots,
            bits=num_detectors,
            name="detectors",
        )
        arrays = {
            name: _packed_array(
                getattr(self, name),
                rows=shots,
                bits=num_observables,
                name=name,
            )
            for name in (
                "actual_observables",
                "global_predictions",
                "union_find_predictions",
            )
        }
        failures: dict[str, np.ndarray] = {}
        for name in ("global_failures", "union_find_failures"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name)))
            if array.dtype != np.bool_ or array.shape != (shots,):
                raise ValueError(f"{name} must be bool with shape {(shots,)}")
            array.setflags(write=False)
            failures[name] = array
        residual_arrays: dict[str, np.ndarray] = {}
        for name in (
            "union_find_residual_hw",
            "union_find_residual_body_hw",
            "union_find_residual_terminal_hw",
            "union_find_residual_yoke_hw",
        ):
            array = np.ascontiguousarray(np.asarray(getattr(self, name)))
            if array.dtype != np.int64 or array.shape != (shots,):
                raise ValueError(f"{name} must be int64 with one value per shot")
            if np.any(array < -1):
                raise ValueError(f"{name} may use only -1 for missing values")
            array.setflags(write=False)
            residual_arrays[name] = array
        object.__setattr__(self, "source_identity", identity)
        object.__setattr__(self, "source_provenance", provenance)
        object.__setattr__(self, "detectors", detectors)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        for name, array in failures.items():
            object.__setattr__(self, name, array)
        for name, array in residual_arrays.items():
            object.__setattr__(self, name, array)

    @property
    def shots(self) -> int:
        return int(self.detectors.shape[0])

    @property
    def num_detectors(self) -> int:
        return int(self.source_identity["num_detectors"])

    @property
    def num_observables(self) -> int:
        return int(self.source_identity["num_observables"])


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _corpus_file_descriptor(
    *, path: str, file_bytes: bytes, array: np.ndarray
) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": _sha256(file_bytes),
        "array": _array_digest(array),
    }


def write_matched_corpus(directory: str | Path, corpus: MatchedCorpus) -> Path:
    """Persist an authenticated matched projection in a new absent directory.

    Arrays are individual NumPy ``.npy`` files written with pickling disabled.
    The canonical manifest is installed last and binds every file byte digest,
    logical array digest, source identity, and source provenance.
    """

    if not isinstance(corpus, MatchedCorpus):
        raise TypeError("corpus must be a MatchedCorpus")
    output = Path(directory).absolute()
    if output.is_symlink() or output.exists():
        raise ValueError("matched corpus output must be a new absent path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("matched corpus parent must be a regular directory")
    output.mkdir()
    descriptors: dict[str, dict[str, Any]] = {}
    try:
        for name in _CORPUS_ARRAY_ATTRIBUTES:
            array = np.asarray(getattr(corpus, name))
            payload = _npy_bytes(array)
            relative = f"{name}.npy"
            install_bytes_atomic(
                output / relative,
                payload,
                prefix="matched-corpus-array-",
                overwrite=False,
            )
            descriptors[name] = _corpus_file_descriptor(
                path=relative, file_bytes=payload, array=array
            )
        unsigned = {
            "schema": CORPUS_MANIFEST_SCHEMA,
            "source_identity": dict(corpus.source_identity),
            "source_provenance": dict(corpus.source_provenance),
            "arrays": descriptors,
        }
        manifest = {**unsigned, "manifest_sha256": _digest_json(unsigned)}
        manifest_path = output / "manifest.json"
        install_bytes_atomic(
            manifest_path,
            canonical_json_bytes(manifest) + b"\n",
            prefix="matched-corpus-manifest-",
            overwrite=False,
        )
        loaded = load_matched_corpus(output)
        for name in _CORPUS_ARRAY_ATTRIBUTES:
            if not np.array_equal(getattr(loaded, name), getattr(corpus, name)):
                raise AssertionError(f"matched corpus round trip differs for {name}")
        return manifest_path
    except BaseException:
        # Preserve partial evidence for diagnosis.  The absent-path contract
        # prevents an interrupted import from being mistaken for resumable.
        raise


def _load_corpus_array(
    root: Path, *, name: str, descriptor: Any
) -> np.ndarray:
    raw = _mapping(descriptor, name=f"matched corpus {name} descriptor")
    if set(raw) != {"path", "file_sha256", "array"}:
        raise ValueError(f"matched corpus {name} descriptor has incorrect fields")
    relative = raw["path"]
    expected_relative = f"{name}.npy"
    if relative != expected_relative:
        raise ValueError(f"matched corpus {name} path differs")
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ValueError(f"matched corpus {name} must be a contained regular file")
    payload = path.read_bytes()
    file_digest = raw["file_sha256"]
    if (
        not isinstance(file_digest, str)
        or len(file_digest) != 64
        or any(character not in "0123456789abcdef" for character in file_digest)
        or _sha256(payload) != file_digest
    ):
        raise ValueError(f"matched corpus {name} file digest mismatch")
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as ex:
        raise ValueError(f"matched corpus {name} is not a safe NumPy array") from ex
    if not isinstance(array, np.ndarray):
        raise ValueError(f"matched corpus {name} did not load as one array")
    if _array_digest(array) != _validate_digest(
        raw["array"], name=f"matched corpus {name} array identity"
    ):
        raise ValueError(f"matched corpus {name} logical array digest mismatch")
    return array


def load_matched_corpus(directory: str | Path) -> MatchedCorpus:
    """Load a strict, hash-complete matched projection without using pickle."""

    candidate = Path(directory).absolute()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("matched corpus root must be a regular non-symlink directory")
    root = candidate.resolve()
    expected_names = {"manifest.json"} | {
        f"{name}.npy" for name in _CORPUS_ARRAY_ATTRIBUTES
    }
    entries = tuple(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("matched corpus entries may not be symlinks")
    if {entry.name for entry in entries} != expected_names or any(
        not entry.is_file() for entry in entries
    ):
        raise ValueError("matched corpus directory has missing or unexpected entries")
    manifest = load_json_strict(
        root / "manifest.json", description="matched corpus manifest"
    )
    expected_fields = {
        "schema",
        "source_identity",
        "source_provenance",
        "arrays",
        "manifest_sha256",
    }
    if set(manifest) != expected_fields or manifest.get("schema") != (
        CORPUS_MANIFEST_SCHEMA
    ):
        raise ValueError("matched corpus manifest has incorrect fields or schema")
    unsigned = dict(manifest)
    recorded_digest = unsigned.pop("manifest_sha256")
    if recorded_digest != _digest_json(unsigned):
        raise ValueError("matched corpus manifest digest mismatch")
    descriptors = _mapping(manifest["arrays"], name="matched corpus arrays")
    if set(descriptors) != set(_CORPUS_ARRAY_ATTRIBUTES):
        raise ValueError("matched corpus manifest has incorrect array fields")
    arrays = {
        name: _load_corpus_array(root, name=name, descriptor=descriptors[name])
        for name in _CORPUS_ARRAY_ATTRIBUTES
    }
    return MatchedCorpus(
        source_identity=manifest["source_identity"],
        source_provenance=manifest["source_provenance"],
        **arrays,
    )


def matched_corpus_from_verified(verified: VerifiedCollection) -> MatchedCorpus:
    """Strictly project exact arrays and imported decisions from UF V3 rows."""

    if not isinstance(verified, VerifiedCollection):
        raise TypeError("verified must be a VerifiedCollection")
    summary = _mapping(verified.summary, name="verified summary")
    if summary.get("stage") != "characterization":
        raise ValueError("matched accuracy requires a characterization collection")
    shots = _positive_int(summary.get("shots"), name="summary shots")
    rows = tuple(verified.shot_rows)
    if len(rows) != shots:
        raise ValueError("verified shot count differs from summary")
    shot_ids = [row.get("global_shot_id") for row in rows]
    if shot_ids != list(range(shots)):
        raise ValueError("verified shots are not in canonical global-shot order")
    provenance = _mapping(summary.get("provenance"), name="summary provenance")
    num_detectors = _positive_int(
        provenance.get("num_detectors"), name="provenance num_detectors"
    )
    num_observables = _positive_int(
        provenance.get("num_observables"), name="provenance num_observables"
    )
    detector_width = (num_detectors + 7) // 8
    if len(verified.detector_corpus_bytes) != shots * detector_width:
        raise ValueError("verified detector corpus has the wrong byte length")
    if _sha256(verified.detector_corpus_bytes) != verified.detector_corpus_sha256:
        raise ValueError("verified detector corpus digest changed")
    detectors = np.frombuffer(verified.detector_corpus_bytes, dtype=np.uint8).reshape(
        shots, detector_width
    )
    detectors = _packed_array(
        detectors, rows=shots, bits=num_detectors, name="verified detectors"
    )
    observable_width = (num_observables + 7) // 8
    actual = _packed_array(
        _hex_rows(rows, field="actual_observables_hex", width=observable_width),
        rows=shots,
        bits=num_observables,
        name="actual observables",
    )
    global_predictions = _packed_array(
        _hex_rows(rows, field="global_prediction_hex", width=observable_width),
        rows=shots,
        bits=num_observables,
        name="imported Global predictions",
    )
    union_find_predictions = _packed_array(
        _hex_rows(rows, field="treatment_prediction_hex", width=observable_width),
        rows=shots,
        bits=num_observables,
        name="imported UF predictions",
    )
    global_failures = _failure_array(global_predictions, actual)
    union_find_failures = _failure_array(union_find_predictions, actual)
    residual_hw = np.full(shots, -1, dtype=np.int64)
    residual_body_hw = np.full(shots, -1, dtype=np.int64)
    residual_terminal_hw = np.full(shots, -1, dtype=np.int64)
    residual_yoke_hw = np.full(shots, -1, dtype=np.int64)
    unpacked = np.unpackbits(
        detectors, axis=1, count=num_detectors, bitorder="little"
    )
    original_hw = np.count_nonzero(unpacked, axis=1)
    for shot_id, row in enumerate(rows):
        for field, derived in (
            ("global_failed", global_failures[shot_id]),
            ("treatment_failed", union_find_failures[shot_id]),
        ):
            recorded = row.get(field)
            if not isinstance(recorded, bool) or recorded != bool(derived):
                raise ValueError(f"recorded {field} differs at shot {shot_id}")
        metrics = row.get("adapter_metrics")
        if metrics is None:
            continue
        metrics = _mapping(metrics, name=f"shot {shot_id} adapter_metrics")
        if "original_detector_count" in metrics:
            recorded_original = _nonnegative_int(
                metrics["original_detector_count"],
                name=f"shot {shot_id} original_detector_count",
            )
            if recorded_original != int(original_hw[shot_id]):
                raise ValueError(
                    f"UF original detector count differs at shot {shot_id}"
                )
        if "residual_detector_count" in metrics:
            residual_hw[shot_id] = _nonnegative_int(
                metrics["residual_detector_count"],
                name=f"shot {shot_id} residual_detector_count",
            )
        role_fields = (
            "residual_body_detector_count",
            "residual_terminal_detector_count",
            "residual_yoke_detector_count",
        )
        present_roles = [field in metrics for field in role_fields]
        if any(present_roles) and not all(present_roles):
            raise ValueError(f"UF residual role counts are partial at shot {shot_id}")
        if all(present_roles):
            role_values = [
                _nonnegative_int(
                    metrics[field], name=f"shot {shot_id} {field}"
                )
                for field in role_fields
            ]
            if residual_hw[shot_id] < 0:
                raise ValueError(
                    f"UF residual role counts omit total at shot {shot_id}"
                )
            if sum(role_values) != int(residual_hw[shot_id]):
                raise ValueError(
                    f"UF residual role counts do not partition total at shot {shot_id}"
                )
            (
                residual_body_hw[shot_id],
                residual_terminal_hw[shot_id],
                residual_yoke_hw[shot_id],
            ) = role_values
    corpus_identity = _mapping(
        verified.corpus_identity, name="verified corpus identity"
    )
    if corpus_identity.get("detectors_sha256") != verified.detector_corpus_sha256:
        raise ValueError("verified corpus detector identity differs")
    observables_sha256 = _sha256(actual.tobytes(order="C"))
    if corpus_identity.get("observables_sha256") != observables_sha256:
        raise ValueError("verified corpus observable identity differs")
    identity = {
        "schema": CORPUS_SCHEMA,
        "experiment_id": summary.get("experiment_id"),
        "protocol_self_sha256": summary.get("protocol_self_sha256"),
        "collection_summary_payload_sha256": summary.get("payload_sha256"),
        "cell_id": summary.get("cell_id"),
        "shots": shots,
        "num_detectors": num_detectors,
        "num_observables": num_observables,
        "detector_corpus_sha256": verified.detector_corpus_sha256,
        "observable_corpus_sha256": observables_sha256,
        "corpus_index_payload_sha256": corpus_identity.get(
            "index_payload_sha256"
        ),
    }
    for name in (
        "experiment_id",
        "protocol_self_sha256",
        "collection_summary_payload_sha256",
        "detector_corpus_sha256",
        "observable_corpus_sha256",
        "corpus_index_payload_sha256",
    ):
        value = identity[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"matched source identity {name} is malformed")
    cell_id = identity["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("matched source cell_id is malformed")
    return MatchedCorpus(
        source_identity=identity,
        detectors=detectors,
        actual_observables=actual,
        global_predictions=global_predictions,
        union_find_predictions=union_find_predictions,
        global_failures=global_failures,
        union_find_failures=union_find_failures,
        union_find_residual_hw=residual_hw,
        union_find_residual_body_hw=residual_body_hw,
        union_find_residual_terminal_hw=residual_terminal_hw,
        union_find_residual_yoke_hw=residual_yoke_hw,
        source_provenance=provenance,
    )


def validate_prepared_cell(prepared: Any, corpus: MatchedCorpus) -> None:
    """Authenticate a compiled Global/ProMatch/Pinball cell against the UF cell."""

    if not isinstance(corpus, MatchedCorpus):
        raise TypeError("corpus must be a MatchedCorpus")
    provenance = _mapping(
        getattr(prepared, "provenance", None), name="prepared provenance"
    )
    for name in ("circuit_sha256", "dem_sha256", "num_detectors", "num_observables"):
        if provenance.get(name) != corpus.source_provenance.get(name):
            raise ValueError(f"prepared cell {name} differs from UF corpus")
    cell = _mapping(getattr(prepared, "cell", None), name="prepared cell")
    if cell.get("cell_id") != corpus.source_identity["cell_id"]:
        raise ValueError("prepared cell_id differs from UF corpus")
    for name in ("matcher_u0", "compiled_promatch", "compiled_pinball"):
        if getattr(prepared, name, None) is None:
            raise ValueError(f"prepared cell omits {name}")


def _complete_cube(failures: Mapping[str, np.ndarray]) -> dict[str, int]:
    shots = len(next(iter(failures.values())))
    cube = Counter(
        "".join("1" if bool(failures[arm][shot]) else "0" for arm in ARM_ORDER)
        for shot in range(shots)
    )
    return {f"{value:04b}": int(cube[f"{value:04b}"]) for value in range(16)}


def _table_from_cube(
    cube: Mapping[str, int], *, baseline: str, treatment: str
) -> dict[str, int]:
    baseline_index = ARM_ORDER.index(baseline)
    treatment_index = ARM_ORDER.index(treatment)
    return {
        "both_correct": sum(
            count
            for bits, count in cube.items()
            if bits[baseline_index] == "0" and bits[treatment_index] == "0"
        ),
        "regressions": sum(
            count
            for bits, count in cube.items()
            if bits[baseline_index] == "0" and bits[treatment_index] == "1"
        ),
        "recoveries": sum(
            count
            for bits, count in cube.items()
            if bits[baseline_index] == "1" and bits[treatment_index] == "0"
        ),
        "both_wrong": sum(
            count
            for bits, count in cube.items()
            if bits[baseline_index] == "1" and bits[treatment_index] == "1"
        ),
    }


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _range_source_arrays(corpus: MatchedCorpus, shot_range: ShotRange) -> dict[str, np.ndarray]:
    selection = slice(shot_range.shot_start, shot_range.shot_stop)
    return {
        "detectors": corpus.detectors[selection],
        "actual_observables": corpus.actual_observables[selection],
        "imported_global_predictions": corpus.global_predictions[selection],
        "union_find_predictions": corpus.union_find_predictions[selection],
        "imported_global_failures": corpus.global_failures[selection],
        "imported_union_find_failures": corpus.union_find_failures[selection],
        "union_find_residual_hw": corpus.union_find_residual_hw[selection],
        "union_find_residual_body_hw": corpus.union_find_residual_body_hw[selection],
        "union_find_residual_terminal_hw": corpus.union_find_residual_terminal_hw[
            selection
        ],
        "union_find_residual_yoke_hw": corpus.union_find_residual_yoke_hw[selection],
    }


def collect_matched_range(
    prepared: Any,
    corpus: MatchedCorpus,
    *,
    shot_range: ShotRange,
    microbatch_size: int = 32,
) -> dict[str, Any]:
    """Decode one exact UF range and emit one compact additive ledger."""

    validate_prepared_cell(prepared, corpus)
    if shot_range not in fixed_worker_ranges(corpus.shots):
        raise ValueError("shot_range is not one of the corpus's exact 32 ranges")
    microbatch_size = _positive_int(microbatch_size, name="microbatch_size")
    source = _range_source_arrays(corpus, shot_range)
    shots = shot_range.shots
    cube = Counter({f"{value:04b}": 0 for value in range(16)})
    agreements = {name: Counter() for name in PAIR_DEFINITIONS}
    telemetry = _Telemetry(prepared)
    global_hw = Counter()
    global_joint_hw = Counter()
    union_find_hw = Counter()
    union_find_joint_hw = Counter()
    global_residual_sum = 0
    union_find_residual_sum = 0
    union_find_available = 0
    union_find_role_available = 0
    union_find_residual_body_sum = 0
    union_find_residual_terminal_sum = 0
    union_find_residual_yoke_sum = 0
    prediction_parts: dict[str, list[np.ndarray]] = {
        "computed_global_predictions": [],
        "promatch_predictions": [],
        "pinball_predictions": [],
    }
    for start in range(0, shots, microbatch_size):
        stop = min(shots, start + microbatch_size)
        packed = source["detectors"][start:stop]
        actual = source["actual_observables"][start:stop]
        imported_global = source["imported_global_predictions"][start:stop]
        computed_global = np.asarray(
            prepared.matcher_u0.decode_batch(
                packed,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
        computed_global = _packed_array(
            computed_global,
            rows=stop - start,
            bits=corpus.num_observables,
            name="computed Global predictions",
        )
        if not np.array_equal(computed_global, imported_global):
            mismatch = int(
                np.flatnonzero(np.any(computed_global != imported_global, axis=1))[0]
            )
            raise ValueError(
                "recomputed Global prediction differs from imported Global at "
                f"shot {shot_range.shot_start + start + mismatch}"
            )
        unpacked = np.unpackbits(
            packed,
            axis=1,
            count=corpus.num_detectors,
            bitorder="little",
        )
        unpacked_before = _array_digest(unpacked)
        pm_residual, pm_frames, pm_results = (
            prepared.compiled_promatch.predecode_shots(unpacked)
        )
        if _array_digest(unpacked) != unpacked_before:
            raise AssertionError("ProMatch mutated the shared matched syndrome")
        if np.count_nonzero(pm_frames):
            raise AssertionError("frozen zero-frame ProMatch emitted a frame")
        promatch = _decode_residual(
            prepared.compiled_promatch, pm_residual, pm_frames
        )
        pb_residual, pb_frames, pb_results = (
            prepared.compiled_pinball.predecode_shots(unpacked)
        )
        if _array_digest(unpacked) != unpacked_before:
            raise AssertionError("Pinball mutated the shared matched syndrome")
        pinball = _decode_residual(
            prepared.compiled_pinball, pb_residual, pb_frames
        )
        union_find = source["union_find_predictions"][start:stop]
        predictions = {
            "global": computed_global,
            "promatch": promatch,
            "pinball": pinball,
            "union_find": union_find,
        }
        failures = {
            arm: _failure_array(prediction, actual)
            for arm, prediction in predictions.items()
        }
        if not np.array_equal(
            failures["global"], source["imported_global_failures"][start:stop]
        ):
            raise ValueError("recomputed Global failures differ from imported failures")
        if not np.array_equal(
            failures["union_find"],
            source["imported_union_find_failures"][start:stop],
        ):
            raise ValueError("imported UF failures do not reconcile")
        local_cube = _complete_cube(failures)
        cube.update(local_cube)
        for name, (baseline, treatment) in PAIR_DEFINITIONS.items():
            equal = np.all(predictions[baseline] == predictions[treatment], axis=1)
            agree = int(np.count_nonzero(equal))
            agreements[name]["agree"] += agree
            agreements[name]["disagree"] += len(equal) - agree
        telemetry.common[
            "promatch_pinball_both_wrong_prediction_disagreement_shots"
        ] += int(
            np.count_nonzero(
                failures["promatch"]
                & failures["pinball"]
                & np.any(promatch != pinball, axis=1)
            )
        )
        telemetry.add_common(unpacked)
        telemetry.add_promatch(unpacked, pm_residual, pm_results)
        telemetry.add_pinball(unpacked, pb_residual, pb_results)
        before_hw = np.count_nonzero(unpacked, axis=1)
        global_residual_sum += int(before_hw.sum())
        global_hw.update(map(int, before_hw))
        global_joint_hw.update(f"{int(value)},{int(value)}" for value in before_hw)
        uf_hw = source["union_find_residual_hw"][start:stop]
        available = uf_hw >= 0
        union_find_available += int(np.count_nonzero(available))
        union_find_residual_sum += int(uf_hw[available].sum())
        union_find_hw.update(map(int, uf_hw[available]))
        union_find_joint_hw.update(
            f"{int(before)},{int(after)}"
            for before, after in zip(before_hw[available], uf_hw[available])
        )
        uf_body = source["union_find_residual_body_hw"][start:stop]
        uf_terminal = source["union_find_residual_terminal_hw"][start:stop]
        uf_yoke = source["union_find_residual_yoke_hw"][start:stop]
        role_available = (uf_body >= 0) & (uf_terminal >= 0) & (uf_yoke >= 0)
        if not np.array_equal(role_available, uf_body >= 0) or not np.array_equal(
            role_available, uf_terminal >= 0
        ) or not np.array_equal(role_available, uf_yoke >= 0):
            raise ValueError("UF residual role availability differs within a range")
        union_find_role_available += int(np.count_nonzero(role_available))
        union_find_residual_body_sum += int(uf_body[role_available].sum())
        union_find_residual_terminal_sum += int(uf_terminal[role_available].sum())
        union_find_residual_yoke_sum += int(uf_yoke[role_available].sum())
        prediction_parts["computed_global_predictions"].append(computed_global)
        prediction_parts["promatch_predictions"].append(promatch)
        prediction_parts["pinball_predictions"].append(pinball)
    predictions = {
        name: np.concatenate(parts, axis=0)
        for name, parts in prediction_parts.items()
    }
    complete_cube = {key: int(cube[key]) for key in sorted(cube)}
    pairwise = {
        name: _table_from_cube(
            complete_cube, baseline=baseline, treatment=treatment
        )
        for name, (baseline, treatment) in PAIR_DEFINITIONS.items()
    }
    native = telemetry.finish()
    telemetry_payload = {
        "common": native["common"],
        "global": {
            "shots": shots,
            "residual_event_sum": global_residual_sum,
            "residual_hw_histogram": _counter_json(global_hw),
            "original_residual_hw_joint_histogram": _counter_json(
                global_joint_hw
            ),
        },
        "promatch": native["promatch"],
        "pinball": native["pinball"],
        "union_find": {
            "shots": shots,
            "residual_hw_available_shots": union_find_available,
            "residual_event_sum": union_find_residual_sum,
            "residual_hw_histogram": _counter_json(union_find_hw),
            "original_residual_hw_joint_histogram": _counter_json(
                union_find_joint_hw
            ),
            "residual_role_available_shots": union_find_role_available,
            "residual_body_event_sum": union_find_residual_body_sum,
            "residual_terminal_event_sum": union_find_residual_terminal_sum,
            "residual_yoke_event_sum": union_find_residual_yoke_sum,
        },
    }
    array_digests = {
        "detectors": _array_digest(source["detectors"]),
        "actual_observables": _array_digest(source["actual_observables"]),
        "imported_global_predictions": _array_digest(
            source["imported_global_predictions"]
        ),
        "computed_global_predictions": _array_digest(
            predictions["computed_global_predictions"]
        ),
        "promatch_predictions": _array_digest(predictions["promatch_predictions"]),
        "pinball_predictions": _array_digest(predictions["pinball_predictions"]),
        "union_find_predictions": _array_digest(source["union_find_predictions"]),
        "imported_global_failures": _array_digest(
            source["imported_global_failures"]
        ),
        "imported_union_find_failures": _array_digest(
            source["imported_union_find_failures"]
        ),
        "union_find_residual_hw": _array_digest(
            source["union_find_residual_hw"]
        ),
        "union_find_residual_body_hw": _array_digest(
            source["union_find_residual_body_hw"]
        ),
        "union_find_residual_terminal_hw": _array_digest(
            source["union_find_residual_terminal_hw"]
        ),
        "union_find_residual_yoke_hw": _array_digest(
            source["union_find_residual_yoke_hw"]
        ),
    }
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "source_identity": dict(corpus.source_identity),
        "prepared_provenance": _canonical_mapping(
            prepared.provenance, name="prepared provenance"
        ),
        "range": shot_range.as_json(),
        "microbatch_size": microbatch_size,
        "arm_order": list(ARM_ORDER),
        "array_digests": array_digests,
        "global_prediction_equality": {"equal": shots, "mismatches": 0},
        "correctness_cube": complete_cube,
        "pairwise_contingencies": pairwise,
        "prediction_agreement": {
            name: {
                "agree": int(agreements[name]["agree"]),
                "disagree": int(agreements[name]["disagree"]),
            }
            for name in PAIR_DEFINITIONS
        },
        "telemetry": telemetry_payload,
    }
    ledger["payload_sha256"] = _digest_json(ledger)
    validate_matched_ledger(
        ledger,
        corpus=corpus,
        expected_prepared_provenance=prepared.provenance,
    )
    return ledger


def _validate_digest(
    value: Any,
    *,
    name: str,
    shape: Sequence[int] | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, name=name)
    if set(raw) != {"sha256", "shape", "dtype"}:
        raise ValueError(f"{name} has incorrect fields")
    digest = raw["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} has malformed SHA-256")
    if (
        not isinstance(raw["shape"], list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw["shape"]
        )
        or not isinstance(raw["dtype"], str)
    ):
        raise ValueError(f"{name} has malformed array metadata")
    if shape is not None and raw["shape"] != list(shape):
        raise ValueError(f"{name} has the wrong shape")
    if dtype is not None and raw["dtype"] != dtype:
        raise ValueError(f"{name} has the wrong dtype")
    return dict(raw)


def _validate_additive(value: Any, *, path: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{path} contains a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{path} contains a negative count")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_additive(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            _validate_additive(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-additive value")


def _shot_range_from_json(value: Any) -> ShotRange:
    raw = _mapping(value, name="range")
    if set(raw) != {"range_id", "shot_start", "shot_stop", "shots"}:
        raise ValueError("range has incorrect fields")
    shot_range = ShotRange(
        range_id=raw["range_id"],
        shot_start=raw["shot_start"],
        shot_stop=raw["shot_stop"],
    )
    if raw["shots"] != shot_range.shots:
        raise ValueError("range shot count does not reconcile")
    return shot_range


def validate_matched_ledger(
    row: Mapping[str, Any],
    *,
    corpus: MatchedCorpus,
    expected_prepared_provenance: Mapping[str, Any] | None = None,
) -> ShotRange:
    """Fail closed on one immutable matched-corpus range ledger."""

    if not isinstance(corpus, MatchedCorpus):
        raise TypeError("corpus must be a MatchedCorpus")
    raw = _mapping(row, name="matched ledger")
    if set(raw) != _LEDGER_FIELDS or raw.get("schema") != LEDGER_SCHEMA:
        raise ValueError("matched ledger has incorrect fields or schema")
    payload = dict(raw)
    recorded_digest = payload.pop("payload_sha256")
    if recorded_digest != _digest_json(payload):
        raise ValueError("matched ledger payload digest mismatch")
    if raw["source_identity"] != dict(corpus.source_identity):
        raise ValueError("matched ledger source identity mismatch")
    provenance = _canonical_mapping(
        raw["prepared_provenance"], name="prepared provenance"
    )
    if expected_prepared_provenance is not None and provenance != _canonical_mapping(
        expected_prepared_provenance, name="expected prepared provenance"
    ):
        raise ValueError("matched ledger prepared provenance mismatch")
    for name in ("circuit_sha256", "dem_sha256", "num_detectors", "num_observables"):
        if provenance.get(name) != corpus.source_provenance.get(name):
            raise ValueError(f"matched ledger prepared {name} differs")
    shot_range = _shot_range_from_json(raw["range"])
    if shot_range not in fixed_worker_ranges(corpus.shots):
        raise ValueError("matched ledger range is not in the exact UF partition")
    shots = shot_range.shots
    _positive_int(raw["microbatch_size"], name="microbatch_size")
    if raw["arm_order"] != list(ARM_ORDER):
        raise ValueError("matched ledger arm order differs")
    digests = _mapping(raw["array_digests"], name="array_digests")
    if set(digests) != _ARRAY_NAMES:
        raise ValueError("matched ledger array digests have incorrect fields")
    observable_width = (corpus.num_observables + 7) // 8
    detector_width = (corpus.num_detectors + 7) // 8
    source = _range_source_arrays(corpus, shot_range)
    source_digest_names = {
        "detectors": "detectors",
        "actual_observables": "actual_observables",
        "imported_global_predictions": "imported_global_predictions",
        "union_find_predictions": "union_find_predictions",
        "imported_global_failures": "imported_global_failures",
        "imported_union_find_failures": "imported_union_find_failures",
        "union_find_residual_hw": "union_find_residual_hw",
        "union_find_residual_body_hw": "union_find_residual_body_hw",
        "union_find_residual_terminal_hw": "union_find_residual_terminal_hw",
        "union_find_residual_yoke_hw": "union_find_residual_yoke_hw",
    }
    for digest_name, source_name in source_digest_names.items():
        if digests[digest_name] != _array_digest(source[source_name]):
            raise ValueError(f"matched ledger source digest differs for {digest_name}")
    for name in (
        "imported_global_predictions",
        "computed_global_predictions",
        "promatch_predictions",
        "pinball_predictions",
        "union_find_predictions",
        "actual_observables",
    ):
        _validate_digest(
            digests[name], name=f"array_digests.{name}", shape=(shots, observable_width)
        )
    _validate_digest(
        digests["detectors"],
        name="array_digests.detectors",
        shape=(shots, detector_width),
    )
    for name in ("imported_global_failures", "imported_union_find_failures"):
        _validate_digest(
            digests[name], name=f"array_digests.{name}", shape=(shots,), dtype="|b1"
        )
    for name in (
        "union_find_residual_hw",
        "union_find_residual_body_hw",
        "union_find_residual_terminal_hw",
        "union_find_residual_yoke_hw",
    ):
        _validate_digest(
            digests[name],
            name=f"array_digests.{name}",
            shape=(shots,),
            dtype="<i8",
        )
    if digests["computed_global_predictions"] != digests[
        "imported_global_predictions"
    ]:
        raise ValueError("computed and imported Global prediction digests differ")
    if raw["global_prediction_equality"] != {"equal": shots, "mismatches": 0}:
        raise ValueError("Global prediction equality does not reconcile")
    cube = _mapping(raw["correctness_cube"], name="correctness_cube")
    expected_cube_keys = {f"{value:04b}" for value in range(16)}
    if set(cube) != expected_cube_keys:
        raise ValueError("correctness cube has incorrect fields")
    for key, value in cube.items():
        _nonnegative_int(value, name=f"correctness_cube.{key}")
    if sum(cube.values()) != shots:
        raise ValueError("correctness cube does not reconcile")
    tables = _mapping(raw["pairwise_contingencies"], name="pairwise contingencies")
    if set(tables) != set(PAIR_DEFINITIONS):
        raise ValueError("pairwise contingencies have incorrect pairs")
    agreements = _mapping(raw["prediction_agreement"], name="prediction agreement")
    if set(agreements) != set(PAIR_DEFINITIONS):
        raise ValueError("prediction agreement has incorrect pairs")
    for name, (baseline, treatment) in PAIR_DEFINITIONS.items():
        expected_table = _table_from_cube(
            cube, baseline=baseline, treatment=treatment
        )
        if tables[name] != expected_table:
            raise ValueError(f"pairwise contingency {name} differs from cube")
        agreement = _mapping(agreements[name], name=f"prediction agreement {name}")
        if set(agreement) != _AGREEMENT_FIELDS:
            raise ValueError(f"prediction agreement {name} has incorrect fields")
        for field in _AGREEMENT_FIELDS:
            _nonnegative_int(agreement[field], name=f"{name}.{field}")
        if sum(agreement.values()) != shots:
            raise ValueError(f"prediction agreement {name} does not reconcile")
        if agreement["disagree"] < (
            expected_table["regressions"] + expected_table["recoveries"]
        ):
            raise ValueError(f"prediction agreement {name} misses discordant outcomes")
    telemetry = _mapping(raw["telemetry"], name="telemetry")
    if set(telemetry) != set(ARM_ORDER) | {"common"}:
        raise ValueError("matched ledger telemetry has incorrect arms")
    _validate_additive(telemetry, path="telemetry")
    expected_keys = {
        "common": _COMMON_TELEMETRY_KEYS,
        "global": {
            "shots",
            "residual_event_sum",
            "residual_hw_histogram",
            "original_residual_hw_joint_histogram",
        },
        "promatch": _PROMATCH_TELEMETRY_KEYS,
        "pinball": _PINBALL_TELEMETRY_KEYS,
        "union_find": {
            "shots",
            "residual_hw_available_shots",
            "residual_event_sum",
            "residual_hw_histogram",
            "original_residual_hw_joint_histogram",
            "residual_role_available_shots",
            "residual_body_event_sum",
            "residual_terminal_event_sum",
            "residual_yoke_event_sum",
        },
    }
    for arm, keys in expected_keys.items():
        branch = _mapping(telemetry[arm], name=f"telemetry.{arm}")
        if set(branch) != keys:
            raise ValueError(f"telemetry.{arm} has incorrect fields")
        if branch.get("shots") != shots:
            raise ValueError(f"telemetry.{arm} shot count differs")
    if sum(telemetry["common"]["original_hw_histogram"].values()) != shots:
        raise ValueError("common original HW histogram does not reconcile")
    if telemetry["global"]["residual_event_sum"] != telemetry["common"][
        "original_event_sum"
    ]:
        raise ValueError("Global residual workload must equal original workload")
    if telemetry["global"]["residual_hw_histogram"] != telemetry["common"][
        "original_hw_histogram"
    ]:
        raise ValueError("Global residual histogram must equal original histogram")
    if sum(
        telemetry["global"]["original_residual_hw_joint_histogram"].values()
    ) != shots:
        raise ValueError("Global joint workload histogram does not reconcile")
    for arm in ("global", "promatch", "pinball"):
        if sum(telemetry[arm]["residual_hw_histogram"].values()) != shots:
            raise ValueError(f"telemetry.{arm} residual histogram does not reconcile")
    uf = telemetry["union_find"]
    available = _nonnegative_int(
        uf["residual_hw_available_shots"], name="UF available shots"
    )
    if available > shots or sum(uf["residual_hw_histogram"].values()) != available:
        raise ValueError("UF residual histogram does not reconcile")
    if sum(uf["original_residual_hw_joint_histogram"].values()) != available:
        raise ValueError("UF joint workload histogram does not reconcile")
    role_available = _nonnegative_int(
        uf["residual_role_available_shots"], name="UF role available shots"
    )
    if role_available > available:
        raise ValueError("UF role availability exceeds residual availability")
    if role_available == available and (
        uf["residual_body_event_sum"]
        + uf["residual_terminal_event_sum"]
        + uf["residual_yoke_event_sum"]
        != uf["residual_event_sum"]
    ):
        raise ValueError("UF residual role sums do not partition total workload")
    return shot_range


def _sum_additive(left: Any, right: Any, *, path: str) -> Any:
    if isinstance(left, int) and not isinstance(left, bool):
        if not isinstance(right, int) or isinstance(right, bool):
            raise ValueError(f"{path} changes additive type")
        return left + right
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError(f"{path} changes additive vector width")
        return [
            _sum_additive(a, b, path=f"{path}[{index}]")
            for index, (a, b) in enumerate(zip(left, right))
        ]
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping):
            raise ValueError(f"{path} changes additive type")
        result = copy.deepcopy(dict(left))
        for key, value in right.items():
            result[key] = (
                copy.deepcopy(value)
                if key not in result
                else _sum_additive(result[key], value, path=f"{path}.{key}")
            )
        return dict(sorted(result.items()))
    raise ValueError(f"{path} is not additive")


def aggregate_matched_ledgers(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus: MatchedCorpus,
    expected_prepared_provenance: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate, order, and add matched range ledgers without replaying shots."""

    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        shot_range = validate_matched_ledger(
            row,
            corpus=corpus,
            expected_prepared_provenance=expected_prepared_provenance,
        )
        if shot_range.range_id in indexed:
            raise ValueError("matched ledgers contain a duplicate range")
        indexed[shot_range.range_id] = copy.deepcopy(dict(row))
    expected_ranges = fixed_worker_ranges(corpus.shots)
    if require_complete and set(indexed) != set(range(RANGE_COUNT)):
        raise ValueError(
            f"matched collection has {len(indexed)} of {RANGE_COUNT} exact ranges"
        )
    if not indexed:
        raise ValueError("at least one matched ledger is required")
    ordered = [indexed[index] for index in sorted(indexed)]
    first = ordered[0]
    aggregate = {
        "correctness_cube": {key: 0 for key in first["correctness_cube"]},
        "pairwise_contingencies": {
            name: {field: 0 for field in _PAIR_FIELDS} for name in PAIR_DEFINITIONS
        },
        "prediction_agreement": {
            name: {field: 0 for field in _AGREEMENT_FIELDS}
            for name in PAIR_DEFINITIONS
        },
        "telemetry": {
            arm: {
                key: (
                    0
                    if isinstance(value, int)
                    else [0] * len(value)
                    if isinstance(value, list)
                    else {}
                )
                for key, value in first["telemetry"][arm].items()
            }
            for arm in first["telemetry"]
        },
    }
    shots = 0
    for row in ordered:
        shots += int(row["range"]["shots"])
        aggregate["correctness_cube"] = _sum_additive(
            aggregate["correctness_cube"],
            row["correctness_cube"],
            path="correctness_cube",
        )
        aggregate["pairwise_contingencies"] = _sum_additive(
            aggregate["pairwise_contingencies"],
            row["pairwise_contingencies"],
            path="pairwise_contingencies",
        )
        aggregate["prediction_agreement"] = _sum_additive(
            aggregate["prediction_agreement"],
            row["prediction_agreement"],
            path="prediction_agreement",
        )
        aggregate["telemetry"] = _sum_additive(
            aggregate["telemetry"], row["telemetry"], path="telemetry"
        )
    result: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "source_identity": dict(corpus.source_identity),
        "prepared_provenance": first["prepared_provenance"],
        "complete": len(indexed) == RANGE_COUNT,
        "ranges": len(indexed),
        "shots": shots,
        "ordered_range_payload_sha256": [
            row["payload_sha256"] for row in ordered
        ],
        "arm_order": list(ARM_ORDER),
        **aggregate,
    }
    result["payload_sha256"] = _digest_json(result)
    validate_matched_aggregate(
        result,
        corpus=corpus,
        expected_prepared_provenance=expected_prepared_provenance,
        require_complete=require_complete,
    )
    return result


def validate_matched_aggregate(
    value: Mapping[str, Any],
    *,
    corpus: MatchedCorpus,
    expected_prepared_provenance: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> None:
    """Validate the reconciled additive projection emitted by the aggregator."""

    raw = _mapping(value, name="matched aggregate")
    expected_fields = {
        "schema",
        "source_identity",
        "prepared_provenance",
        "complete",
        "ranges",
        "shots",
        "ordered_range_payload_sha256",
        "arm_order",
        "correctness_cube",
        "pairwise_contingencies",
        "prediction_agreement",
        "telemetry",
        "payload_sha256",
    }
    if set(raw) != expected_fields or raw.get("schema") != AGGREGATE_SCHEMA:
        raise ValueError("matched aggregate has incorrect fields or schema")
    payload = dict(raw)
    digest = payload.pop("payload_sha256")
    if digest != _digest_json(payload):
        raise ValueError("matched aggregate payload digest mismatch")
    if raw["source_identity"] != dict(corpus.source_identity):
        raise ValueError("matched aggregate source identity differs")
    provenance = _canonical_mapping(
        raw["prepared_provenance"], name="aggregate prepared provenance"
    )
    if expected_prepared_provenance is not None and provenance != _canonical_mapping(
        expected_prepared_provenance, name="expected prepared provenance"
    ):
        raise ValueError("matched aggregate prepared provenance differs")
    ranges = _positive_int(raw["ranges"], name="aggregate ranges")
    if ranges > RANGE_COUNT or raw["complete"] != (ranges == RANGE_COUNT):
        raise ValueError("matched aggregate range completeness differs")
    if require_complete and not raw["complete"]:
        raise ValueError("matched aggregate is incomplete")
    shots = _positive_int(raw["shots"], name="aggregate shots")
    if raw["complete"] and shots != corpus.shots:
        raise ValueError("complete matched aggregate shot count differs")
    digests = raw["ordered_range_payload_sha256"]
    if not isinstance(digests, list) or len(digests) != ranges:
        raise ValueError("aggregate ordered range digest list differs")
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in digests
    ):
        raise ValueError("aggregate ordered range digest is malformed")
    if raw["arm_order"] != list(ARM_ORDER):
        raise ValueError("aggregate arm order differs")
    cube = _mapping(raw["correctness_cube"], name="aggregate cube")
    if set(cube) != {f"{value:04b}" for value in range(16)} or sum(
        cube.values()
    ) != shots:
        raise ValueError("aggregate correctness cube does not reconcile")
    tables = _mapping(raw["pairwise_contingencies"], name="aggregate pairs")
    agreements = _mapping(raw["prediction_agreement"], name="aggregate agreement")
    if set(tables) != set(PAIR_DEFINITIONS) or set(agreements) != set(
        PAIR_DEFINITIONS
    ):
        raise ValueError("aggregate pair fields differ")
    for name, (baseline, treatment) in PAIR_DEFINITIONS.items():
        if not isinstance(tables[name], Mapping) or set(tables[name]) != _PAIR_FIELDS:
            raise ValueError(f"aggregate pair {name} has incorrect fields")
        if tables[name] != _table_from_cube(
            cube, baseline=baseline, treatment=treatment
        ):
            raise ValueError(f"aggregate pair {name} differs from cube")
        if (
            not isinstance(agreements[name], Mapping)
            or set(agreements[name]) != _AGREEMENT_FIELDS
        ):
            raise ValueError(f"aggregate agreement {name} has incorrect fields")
        if sum(agreements[name].values()) != shots:
            raise ValueError(f"aggregate agreement {name} does not reconcile")
    telemetry = _mapping(raw["telemetry"], name="aggregate telemetry")
    if set(telemetry) != set(ARM_ORDER) | {"common"}:
        raise ValueError("aggregate telemetry arms differ")
    _validate_additive(telemetry, path="aggregate.telemetry")
    expected_telemetry_keys = {
        "common": _COMMON_TELEMETRY_KEYS,
        "global": {
            "shots",
            "residual_event_sum",
            "residual_hw_histogram",
            "original_residual_hw_joint_histogram",
        },
        "promatch": _PROMATCH_TELEMETRY_KEYS,
        "pinball": _PINBALL_TELEMETRY_KEYS,
        "union_find": {
            "shots",
            "residual_hw_available_shots",
            "residual_event_sum",
            "residual_hw_histogram",
            "original_residual_hw_joint_histogram",
            "residual_role_available_shots",
            "residual_body_event_sum",
            "residual_terminal_event_sum",
            "residual_yoke_event_sum",
        },
    }
    for arm in telemetry:
        if set(telemetry[arm]) != expected_telemetry_keys[arm]:
            raise ValueError(f"aggregate telemetry {arm} fields differ")
        if telemetry[arm].get("shots") != shots:
            raise ValueError(f"aggregate telemetry {arm} shots differ")
    if sum(telemetry["common"]["original_hw_histogram"].values()) != shots:
        raise ValueError("aggregate original histogram does not reconcile")
    for arm in ("global", "promatch", "pinball"):
        if sum(telemetry[arm]["residual_hw_histogram"].values()) != shots:
            raise ValueError(f"aggregate {arm} residual histogram does not reconcile")
    if sum(
        telemetry["global"]["original_residual_hw_joint_histogram"].values()
    ) != shots:
        raise ValueError("aggregate Global joint histogram does not reconcile")
    if sum(telemetry["union_find"]["residual_hw_histogram"].values()) != telemetry[
        "union_find"
    ]["residual_hw_available_shots"]:
        raise ValueError("aggregate UF residual histogram does not reconcile")
    uf = telemetry["union_find"]
    if sum(uf["original_residual_hw_joint_histogram"].values()) != uf[
        "residual_hw_available_shots"
    ]:
        raise ValueError("aggregate UF joint histogram does not reconcile")
    if uf["residual_role_available_shots"] > uf["residual_hw_available_shots"]:
        raise ValueError("aggregate UF role availability exceeds residual availability")
    if uf["residual_role_available_shots"] == uf[
        "residual_hw_available_shots"
    ] and (
        uf["residual_body_event_sum"]
        + uf["residual_terminal_event_sum"]
        + uf["residual_yoke_event_sum"]
        != uf["residual_event_sum"]
    ):
        raise ValueError("aggregate UF role sums do not partition residual workload")


_WORKER_PRELOAD: tuple[Any, MatchedCorpus] | None = None


def preload_matched_worker(prepared: Any, corpus: MatchedCorpus) -> None:
    """Install parent-compiled state for inheritance by explicit-fork workers."""

    global _WORKER_PRELOAD
    validate_prepared_cell(prepared, corpus)
    _WORKER_PRELOAD = (prepared, corpus)


def clear_matched_worker_preload() -> None:
    """Release parent-owned matched state after the range pool exits."""

    global _WORKER_PRELOAD
    _WORKER_PRELOAD = None


def matched_range_tasks(
    corpus: MatchedCorpus, *, microbatch_size: int = 32
) -> tuple[dict[str, Any], ...]:
    """Return pickleable tasks for the exact 32 authenticated UF ranges."""

    if not isinstance(corpus, MatchedCorpus):
        raise TypeError("corpus must be a MatchedCorpus")
    microbatch_size = _positive_int(microbatch_size, name="microbatch_size")
    return tuple(
        {
            "source_identity": dict(corpus.source_identity),
            "range": shot_range.as_json(),
            "microbatch_size": microbatch_size,
            "require_preload": True,
        }
        for shot_range in fixed_worker_ranges(corpus.shots)
    )


def worker_collect_matched_range(task: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry point; fail rather than compile or copy a missing cell."""

    if _WORKER_PRELOAD is None:
        raise RuntimeError("matched worker did not inherit parent-precompiled state")
    prepared, corpus = _WORKER_PRELOAD
    raw = _mapping(task, name="matched worker task")
    expected_fields = {
        "source_identity",
        "range",
        "microbatch_size",
        "require_preload",
    }
    if set(raw) != expected_fields or raw.get("require_preload") is not True:
        raise ValueError("matched worker task has incorrect fields")
    if raw["source_identity"] != dict(corpus.source_identity):
        raise ValueError("matched worker task source identity differs")
    return collect_matched_range(
        prepared,
        corpus,
        shot_range=_shot_range_from_json(raw["range"]),
        microbatch_size=raw["microbatch_size"],
    )


__all__ = [
    "AGGREGATE_SCHEMA",
    "ARM_ORDER",
    "CORPUS_MANIFEST_SCHEMA",
    "CORPUS_SCHEMA",
    "LEDGER_SCHEMA",
    "MatchedCorpus",
    "PAIR_DEFINITIONS",
    "aggregate_matched_ledgers",
    "clear_matched_worker_preload",
    "collect_matched_range",
    "load_matched_corpus",
    "matched_corpus_from_verified",
    "matched_range_tasks",
    "preload_matched_worker",
    "validate_matched_aggregate",
    "validate_matched_ledger",
    "validate_prepared_cell",
    "worker_collect_matched_range",
    "write_matched_corpus",
]
