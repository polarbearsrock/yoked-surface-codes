"""Production YSC integration for the matched three-arm latency harness.

The source of timing shots is an already authenticated Patch-UF sanitized
latency corpus.  Import copies only its packed detector array and row IDs into
the additive matched-latency corpus format.  The production factory then
compiles the common YSC cell once for fork/COW execution (or once per restart
when used through the fallback spawn factory interface).
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from yoked.decoding._artifact_io import load_json_strict
from yoked.decoding._patch_uf_latency import (
    BatchTiming,
    HostPolicy,
    LatencyProtocol,
    capture_host_policy,
    load_authenticated_latency_corpus,
)
from yoked.decoding._pinball_promatch_experiment import (
    PINBALL_CONFIG,
    PROMATCH_CONFIG,
    PreparedCell,
    prepare_cell,
)
from yoked.decoding._pinball_promatch_matched_latency import (
    AuthenticatedDetectorCorpus,
    LatencyWorkload,
    build_timed_variants,
    load_authenticated_detector_corpus,
    write_authenticated_detector_corpus,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


IMPORT_SCHEMA = "pinball-promatch-matched-latency-import-v1"

_CORPUS_IDENTITY_FIELDS = {
    "manifest_sha256",
    "corpus_digest",
    "source_patch_uf_manifest_sha256",
    "source_patch_uf_corpus_digest",
    "source_detector_array_digest",
}
_PREPARED_HASH_FIELDS = (
    "circuit_sha256",
    "dem_sha256",
    "promatch_layout_fingerprint",
    "promatch_graph_fingerprint",
    "pinball_layout_fingerprint",
    "pinball_graph_fingerprint",
    "pinball_schedule_fingerprint",
)
_REQUIRED_CELL_FIELDS = {
    "cell_id",
    "d",
    "r",
    "p",
    "patches",
    "yokes",
    *_PREPARED_HASH_FIELDS,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_mapping(value: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(dict(value)))


def _canonical_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} must contain canonical JSON data") from ex
    if not isinstance(normalized, dict):  # pragma: no cover - mapping guarantees this
        raise AssertionError("canonical mapping normalization changed top-level type")
    return normalized


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return _sha256(header + b"\0" + array.tobytes(order="C"))


def _source_detector_sha256(
    source_manifest_path: Path, source_manifest: Mapping[str, Any]
) -> str:
    descriptor = source_manifest.get("detectors")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("Patch-UF detector descriptor is malformed")
    relative = descriptor.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or Path(relative).parts != (relative,)
    ):
        raise ValueError("Patch-UF detector artifact path is unsafe")
    path = source_manifest_path.parent / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("Patch-UF detector artifact must be a regular file")
    digest = _sha256(path.read_bytes())
    if digest != _validate_digest(descriptor.get("sha256"), name="detectors sha256"):
        raise ValueError("Patch-UF detector artifact digest mismatch")
    return digest


def materialize_detector_corpus_from_patch_uf(
    source_manifest_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    expected_source_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Authenticates a UF sanitized corpus and copies only its detector rows.

    ``load_authenticated_latency_corpus`` validates the exact source manifest,
    all source artifact hashes, and recursively rejects observable-bearing
    fields before this function examines the detector array.
    """

    source_path = Path(source_manifest_path)
    source_manifest = load_json_strict(
        source_path, description="source Patch-UF latency corpus manifest"
    )
    source = load_authenticated_latency_corpus(source_path)
    detector_artifact_sha256 = _source_detector_sha256(source_path, source_manifest)
    detector_array_digest = _array_digest(source.detectors)
    source_provenance = json.loads(source.provenance_json)
    expected = {} if expected_source_identity is None else _canonical_mapping(
        expected_source_identity, name="expected_source_identity"
    )
    actual = {
        "manifest_sha256": source.manifest_sha256,
        "corpus_digest": source.corpus_digest,
        "detectors_sha256": detector_artifact_sha256,
        "detector_array_digest": detector_array_digest,
        "num_detectors": source.num_detectors,
        "row_count": source.row_count,
    }
    unknown = set(expected) - set(actual)
    if unknown:
        raise ValueError(f"unknown expected source identity fields: {sorted(unknown)}")
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(f"source Patch-UF identity field {name!r} differs")
    provenance = {
        "schema": IMPORT_SCHEMA,
        "source_patch_uf_manifest_sha256": source.manifest_sha256,
        "source_patch_uf_corpus_digest": source.corpus_digest,
        "source_detector_artifact_sha256": detector_artifact_sha256,
        "source_detector_array_digest": detector_array_digest,
        "source_cell_id": source_provenance.get("cell_id"),
        "source_circuit_sha256": source_provenance.get("circuit_sha256"),
        "source_dem_sha256": source_provenance.get("dem_sha256"),
        "num_detectors": source.num_detectors,
        "row_count": source.row_count,
    }
    target_path = write_authenticated_detector_corpus(
        out_dir,
        detectors=source.detectors,
        num_detectors=source.num_detectors,
        global_shot_ids=source.global_shot_ids,
        provenance=provenance,
    )
    target = load_authenticated_detector_corpus(target_path)
    if (
        target.global_shot_ids != source.global_shot_ids
        or target.num_detectors != source.num_detectors
        or target.row_count != source.row_count
        or _array_digest(target.detectors) != detector_array_digest
        or not np.array_equal(target.detectors, source.detectors)
    ):
        raise ValueError("imported matched detector corpus differs from Patch-UF source")
    return target_path


def _validate_packed_input(value: np.ndarray, *, num_detectors: int) -> np.ndarray:
    packed = np.asarray(value)
    expected_width = (num_detectors + 7) // 8
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != expected_width:
        raise ValueError(
            "packed detector input must be uint8 with width "
            f"{expected_width}, got dtype={packed.dtype}, shape={packed.shape}"
        )
    if num_detectors % 8 and len(packed) and expected_width:
        unused = 0xFF ^ ((1 << (num_detectors % 8)) - 1)
        if np.any(np.bitwise_and(packed[:, -1], unused)):
            raise ValueError("packed detector input has nonzero tail bits")
    return packed


@dataclasses.dataclass(frozen=True)
class CompiledDirectPyMatchingDecoder:
    """Direct uncorrelated PyMatching behind the common public packed API."""

    matcher: Any
    num_detectors: int
    num_observables: int

    def __post_init__(self) -> None:
        _positive_int(self.num_detectors, name="num_detectors")
        if (
            isinstance(self.num_observables, bool)
            or not isinstance(self.num_observables, int)
            or self.num_observables < 0
        ):
            raise ValueError("num_observables must be a nonnegative integer")
        if not callable(getattr(self.matcher, "decode_batch", None)):
            raise TypeError("matcher must expose decode_batch")

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        packed = _validate_packed_input(
            bit_packed_detection_event_data,
            num_detectors=self.num_detectors,
        )
        prediction = np.asarray(
            self.matcher.decode_batch(
                packed,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
        expected = (len(packed), (self.num_observables + 7) // 8)
        if prediction.shape != expected:
            raise ValueError(
                f"direct PyMatching returned shape {prediction.shape}, expected {expected}"
            )
        if self.num_observables % 8 and prediction.shape[1]:
            prediction[:, -1] &= (1 << (self.num_observables % 8)) - 1
        return prediction


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = _canonical_mapping(protocol, name="matched latency protocol")
    experiment_id = value.get("experiment_id")
    _validate_digest(experiment_id, name="experiment_id")
    cell = value.get("cell")
    if not isinstance(cell, Mapping) or not _REQUIRED_CELL_FIELDS <= set(cell):
        raise ValueError("matched latency protocol cell is incomplete")
    for field in _PREPARED_HASH_FIELDS:
        _validate_digest(cell.get(field), name=f"cell {field}")
    if not isinstance(cell.get("cell_id"), str) or not cell["cell_id"]:
        raise ValueError("matched latency cell_id must be nonempty")
    if value.get("promatch_config") != PROMATCH_CONFIG:
        raise ValueError("matched latency protocol has the wrong ProMatch config")
    if value.get("pinball_config") != PINBALL_CONFIG:
        raise ValueError("matched latency protocol has the wrong Pinball config")
    dem_options = value.get("dem_options")
    if (
        not isinstance(dem_options, Mapping)
        or not dem_options
        or any(
            not isinstance(key, str) or not isinstance(flag, bool)
            for key, flag in dem_options.items()
        )
    ):
        raise ValueError("matched latency DEM options must be a nonempty bool mapping")
    corpus = value.get("corpus")
    if not isinstance(corpus, Mapping) or not _CORPUS_IDENTITY_FIELDS <= set(corpus):
        raise ValueError("matched latency protocol corpus identity is incomplete")
    for field in _CORPUS_IDENTITY_FIELDS:
        _validate_digest(corpus.get(field), name=f"corpus {field}")
    if not isinstance(value.get("latency"), Mapping):
        raise ValueError("matched latency protocol requires latency literals")
    return value


def _validate_corpus_against_protocol(
    protocol: Mapping[str, Any], corpus: AuthenticatedDetectorCorpus
) -> dict[str, Any]:
    expected = protocol["corpus"]
    provenance = dict(corpus.provenance)
    actual = {
        "manifest_sha256": corpus.manifest_sha256,
        "corpus_digest": corpus.corpus_digest,
        "source_patch_uf_manifest_sha256": provenance.get(
            "source_patch_uf_manifest_sha256"
        ),
        "source_patch_uf_corpus_digest": provenance.get(
            "source_patch_uf_corpus_digest"
        ),
        "source_detector_array_digest": provenance.get(
            "source_detector_array_digest"
        ),
    }
    for name, value in actual.items():
        if expected.get(name) != value:
            raise ValueError(f"matched detector corpus identity field {name!r} differs")
    if provenance.get("source_detector_array_digest") != corpus.corpus_digest:
        raise ValueError("matched detector corpus array differs from imported source")
    if provenance.get("schema") != IMPORT_SCHEMA:
        raise ValueError("matched detector corpus has the wrong import provenance")
    cell = protocol["cell"]
    if (
        provenance.get("source_cell_id") != cell["cell_id"]
        or provenance.get("source_circuit_sha256") != cell["circuit_sha256"]
        or provenance.get("source_dem_sha256") != cell["dem_sha256"]
        or provenance.get("num_detectors") != corpus.num_detectors
        or provenance.get("row_count") != corpus.row_count
    ):
        raise ValueError("matched detector corpus source provenance differs from cell")
    return provenance


def _validate_prepared(
    protocol: Mapping[str, Any],
    corpus: AuthenticatedDetectorCorpus,
    prepared: PreparedCell,
) -> dict[str, Any]:
    provenance = _canonical_mapping(prepared.provenance, name="prepared provenance")
    cell = protocol["cell"]
    for field in _PREPARED_HASH_FIELDS:
        if provenance.get(field) != cell[field]:
            raise ValueError(f"prepared cell {field!r} differs from protocol")
    if provenance.get("num_detectors") != corpus.num_detectors:
        raise ValueError("prepared detector count differs from timing corpus")
    num_observables = provenance.get("num_observables")
    _positive_int(num_observables, name="prepared num_observables")
    if not callable(getattr(prepared.compiled_promatch, "decode_shots_bit_packed", None)):
        raise TypeError("prepared ProMatch decoder lacks packed production decode")
    if not callable(getattr(prepared.compiled_pinball, "decode_shots_bit_packed", None)):
        raise TypeError("prepared Pinball decoder lacks packed production decode")
    return provenance


@dataclasses.dataclass(frozen=True)
class YokedMatchedLatencyFactory:
    """Pickleable production factory with an explicit compile-once preload hook."""

    protocol_json: str
    corpus_manifest_path: str

    @classmethod
    def from_protocol(
        cls,
        protocol: Mapping[str, Any],
        *,
        corpus_manifest_path: str | os.PathLike[str],
    ) -> "YokedMatchedLatencyFactory":
        normalized = _validate_protocol(protocol)
        corpus = load_authenticated_detector_corpus(corpus_manifest_path)
        _validate_corpus_against_protocol(normalized, corpus)
        return cls(
            protocol_json=canonical_json_bytes(normalized).decode("utf-8"),
            corpus_manifest_path=str(Path(corpus_manifest_path).resolve()),
        )

    def __post_init__(self) -> None:
        try:
            parsed = json.loads(self.protocol_json)
        except (TypeError, json.JSONDecodeError) as ex:
            raise ValueError("protocol_json must be canonical JSON") from ex
        protocol = _validate_protocol(parsed)
        if canonical_json_bytes(protocol).decode("utf-8") != self.protocol_json:
            raise ValueError("protocol_json is not canonical")
        corpus = load_authenticated_detector_corpus(self.corpus_manifest_path)
        _validate_corpus_against_protocol(protocol, corpus)

    def _protocol(self) -> dict[str, Any]:
        return _validate_protocol(json.loads(self.protocol_json))

    @property
    def suite_identity(self) -> dict[str, Any]:
        protocol = self._protocol()
        corpus = load_authenticated_detector_corpus(self.corpus_manifest_path)
        import_provenance = _validate_corpus_against_protocol(protocol, corpus)
        cell = protocol["cell"]
        return {
            "experiment_id": protocol["experiment_id"],
            "protocol_sha256": _sha256(self.protocol_json.encode("utf-8")),
            "cell_id": cell["cell_id"],
            "circuit_sha256": cell["circuit_sha256"],
            "dem_sha256": cell["dem_sha256"],
            "promatch_layout_fingerprint": cell["promatch_layout_fingerprint"],
            "promatch_graph_fingerprint": cell["promatch_graph_fingerprint"],
            "pinball_layout_fingerprint": cell["pinball_layout_fingerprint"],
            "pinball_graph_fingerprint": cell["pinball_graph_fingerprint"],
            "pinball_schedule_fingerprint": cell["pinball_schedule_fingerprint"],
            "promatch_config_sha256": _digest_mapping(protocol["promatch_config"]),
            "pinball_config_sha256": _digest_mapping(protocol["pinball_config"]),
            "dem_options_sha256": _digest_mapping(protocol["dem_options"]),
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "source_patch_uf_manifest_sha256": import_provenance[
                "source_patch_uf_manifest_sha256"
            ],
            "source_patch_uf_corpus_digest": import_provenance[
                "source_patch_uf_corpus_digest"
            ],
            "source_detector_array_digest": import_provenance[
                "source_detector_array_digest"
            ],
        }

    def _build_workload(
        self, *, restart_index: int | None, batch_size: int | None
    ) -> LatencyWorkload:
        protocol = self._protocol()
        corpus = load_authenticated_detector_corpus(self.corpus_manifest_path)
        import_provenance = _validate_corpus_against_protocol(protocol, corpus)
        prepared = prepare_cell(
            protocol["cell"],
            promatch_config=protocol["promatch_config"],
            pinball_config=protocol["pinball_config"],
            dem_options=protocol["dem_options"],
            verify_hashes=True,
        )
        prepared_provenance = _validate_prepared(protocol, corpus, prepared)
        global_decoder = CompiledDirectPyMatchingDecoder(
            matcher=prepared.matcher_u0,
            num_detectors=corpus.num_detectors,
            num_observables=prepared_provenance["num_observables"],
        )
        provenance = {
            "factory": "yoked-matched-latency-production-v1",
            "compile_pid": os.getpid(),
            "restart_index": restart_index,
            "batch_size": batch_size,
            "protocol_sha256": _sha256(self.protocol_json.encode("utf-8")),
            "promatch_config": protocol["promatch_config"],
            "pinball_config": protocol["pinball_config"],
            "dem_options": protocol["dem_options"],
            "prepared": prepared_provenance,
            "corpus_import": import_provenance,
        }
        return LatencyWorkload(
            corpus=corpus,
            variants=build_timed_variants(
                global_mwpm=global_decoder,
                promatch=prepared.compiled_promatch,
                pinball=prepared.compiled_pinball,
            ),
            provenance=provenance,
        )

    def preload(self) -> LatencyWorkload:
        """Compiles all three production arms exactly once in the fork parent."""

        return self._build_workload(restart_index=None, batch_size=None)

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        """Fallback spawn interface; rebuilds within that fresh restart."""

        if (
            isinstance(restart_index, bool)
            or not isinstance(restart_index, int)
            or restart_index < 0
        ):
            raise ValueError("restart_index must be a nonnegative integer")
        _positive_int(batch_size, name="batch_size")
        return self._build_workload(
            restart_index=restart_index,
            batch_size=batch_size,
        )


def latency_protocol_from_matched_protocol(
    protocol: Mapping[str, Any],
    *,
    cpu: int | None = None,
) -> LatencyProtocol:
    """Maps exact per-batch timing literals into the generic latency protocol."""

    normalized = _validate_protocol(protocol)
    timing = normalized["latency"]
    rows = timing.get("batches")
    if not isinstance(rows, list) or not rows:
        raise ValueError("matched latency batches must be a nonempty list")
    expected_fields = {
        "batch_size",
        "restarts",
        "blocks_per_restart",
        "warmup_calls_per_variant",
        "timed_calls_per_side_per_block",
    }
    batches = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("matched latency batch fields are malformed")
        batches.append(BatchTiming(**dict(row)))
    seed = timing.get("schedule_seed")
    if isinstance(seed, str):
        _validate_digest(seed, name="latency schedule_seed")
        seed_value = int(seed, 16)
    elif isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**256:
        seed_value = seed
    else:
        raise ValueError("latency schedule_seed must be a uint256 or 64 hex digits")
    raw_host = timing.get("host_policy")
    if raw_host is None:
        host = capture_host_policy(cpu=cpu)
    else:
        if not isinstance(raw_host, Mapping) or set(raw_host) != {
            "cpu_affinity",
            "expected_host",
            "expected_numa_nodes",
        }:
            raise ValueError("matched latency host_policy fields are malformed")
        expected_host = raw_host["expected_host"]
        if not isinstance(expected_host, Mapping):
            raise ValueError("matched latency expected_host must be a mapping")
        host = HostPolicy(
            cpu_affinity=tuple(raw_host["cpu_affinity"]),
            expected_host=tuple(sorted(expected_host.items())),
            expected_numa_nodes=tuple(raw_host["expected_numa_nodes"]),
        )
        if cpu is not None and host.cpu_affinity != (cpu,):
            raise ValueError("requested timing CPU differs from matched host policy")
    return LatencyProtocol(
        batches=tuple(batches),
        schedule_seed=seed_value,
        host_policy=host,
    )


__all__ = [
    "CompiledDirectPyMatchingDecoder",
    "IMPORT_SCHEMA",
    "YokedMatchedLatencyFactory",
    "latency_protocol_from_matched_protocol",
    "materialize_detector_corpus_from_patch_uf",
]
