"""Deterministic paired collector for confidence-gated patch Union-Find.

The collector deliberately knows very little about the Union-Find engine.  A
compiled adapter receives only packed detector samples and returns immutable
predictions plus bounded telemetry.  This module normalizes that telemetry
into shot, lane, and component ledgers, joins actual observables only after
every prediction is fixed, and owns resumable deterministic artifacts.

Each logical worker range has two files.  The component/lane file is installed
first.  The shot shard, which cross-authenticates that file, is installed last
and is the only commit marker.  An orphan component file is regenerated and
reused only when its complete deterministic bytes match.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
from enum import Enum
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    install_bytes_atomic,
    is_lowercase_hex,
    unique_json_object,
    validate_resumable_output_root,
)
from yoked.decoding._promatch_stats import canonical_json_bytes, digest_array
from yoked.decoding._patch_uf_stats import ShotClusterRecord


PROTOCOL_SCHEMA = "patch-uf-mwpm-protocol-v1"
COMPONENT_SCHEMA = "patch-uf-component-range-v1"
SHOT_SCHEMA = "patch-uf-shot-range-v1"
SUMMARY_SCHEMA = "patch-uf-collection-summary-v1"
CORPUS_SCHEMA = "patch-uf-characterization-corpus-v1"
SCHEMA_VERSION = 1
RANGE_COUNT = 32
SHAKEOUT_STAGE = "engineering-shakeout"
CHARACTERIZATION_STAGE = "characterization"
SCIENTIFIC_STAGE_SHOTS = {
    SHAKEOUT_STAGE: 1_000,
    CHARACTERIZATION_STAGE: 10_000,
}
SEED_DERIVATION = (
    "sha256(root-bytes || ASCII(patch-uf-named-seed-v1\\0) || "
    "canonical-json(identity))-first8-uint64le"
)
SEED_SCHEMA = "patch-uf-named-seed-v1"
DEFAULT_MAX_COMPONENT_RECORDS_PER_SHOT = 4_096
DEFAULT_MAX_METRIC_BYTES_PER_RANGE = 512 * 1024 * 1024


__all__ = [
    "CHARACTERIZATION_STAGE",
    "COMPONENT_SCHEMA",
    "CORPUS_SCHEMA",
    "DEFAULT_MAX_COMPONENT_RECORDS_PER_SHOT",
    "DEFAULT_MAX_METRIC_BYTES_PER_RANGE",
    "PROTOCOL_SCHEMA",
    "PreparedCell",
    "RANGE_COUNT",
    "RangeArtifacts",
    "SCHEMA_VERSION",
    "SEED_DERIVATION",
    "SHAKEOUT_STAGE",
    "SHOT_SCHEMA",
    "SUMMARY_SCHEMA",
    "ShotRange",
    "VerifiedCollection",
    "canonical_protocol_self_sha256",
    "collect_prepared_range",
    "configure_single_thread_runtime",
    "derive_named_seed",
    "fixed_worker_ranges",
    "prepare_selected_cell",
    "run_collection",
    "verify_collection",
]


def configure_single_thread_runtime() -> None:
    """Pin every native numerical runtime to one thread."""

    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _safe_name(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{name} must be a safe nonempty path component")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclasses.dataclass(frozen=True, order=True)
class ShotRange:
    """One of the experiment's fixed 32 half-open logical ranges."""

    range_id: int
    shot_start: int
    shot_stop: int

    def __post_init__(self) -> None:
        range_id = _nonnegative_int(self.range_id, name="range_id")
        start = _nonnegative_int(self.shot_start, name="shot_start")
        stop = _nonnegative_int(self.shot_stop, name="shot_stop")
        if range_id >= RANGE_COUNT:
            raise ValueError(f"range_id must be below {RANGE_COUNT}")
        if stop <= start:
            raise ValueError("a collection range must contain at least one shot")
        object.__setattr__(self, "range_id", range_id)
        object.__setattr__(self, "shot_start", start)
        object.__setattr__(self, "shot_stop", stop)

    @property
    def shots(self) -> int:
        return self.shot_stop - self.shot_start

    def as_json(self) -> dict[str, int]:
        return {
            "range_id": self.range_id,
            "shot_start": self.shot_start,
            "shot_stop": self.shot_stop,
            "shots": self.shots,
        }


def fixed_worker_ranges(shots: int) -> tuple[ShotRange, ...]:
    """Return the exact 32-range floor partition frozen by the experiment."""

    shots = _positive_int(shots, name="shots")
    if shots < RANGE_COUNT:
        raise ValueError(f"shots must be at least {RANGE_COUNT} for 32 nonempty ranges")
    result = tuple(
        ShotRange(
            range_id=worker,
            shot_start=shots * worker // RANGE_COUNT,
            shot_stop=shots * (worker + 1) // RANGE_COUNT,
        )
        for worker in range(RANGE_COUNT)
    )
    if result[0].shot_start != 0 or result[-1].shot_stop != shots:
        raise AssertionError("fixed range partition does not cover the corpus")
    for left, right in zip(result, result[1:]):
        if left.shot_stop != right.shot_start:
            raise AssertionError("fixed range partition has a gap or overlap")
    return result


def canonical_protocol_self_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash canonical protocol JSON with its self-hash field omitted."""

    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    payload = dict(protocol)
    payload.pop("protocol_self_sha256", None)
    return _sha256(canonical_json_bytes(payload))


def derive_named_seed(
    *,
    seed_root: str,
    experiment_id: str,
    stage: str,
    cell_id: str,
    shot_range: ShotRange,
    purpose: str,
) -> int:
    """Derive a range seed from every frozen identity and a purpose label."""

    if not is_lowercase_hex(seed_root, length=64):
        raise ValueError("seed_root must be 64 lowercase hexadecimal characters")
    if not is_lowercase_hex(experiment_id, length=64):
        raise ValueError("experiment_id must be 64 lowercase hexadecimal characters")
    stage = _safe_name(stage, name="stage")
    cell_id = _safe_name(cell_id, name="cell_id")
    purpose = _safe_name(purpose, name="purpose")
    if not isinstance(shot_range, ShotRange):
        raise TypeError("shot_range must be a ShotRange")
    identity = {
        "schema": SEED_SCHEMA,
        "experiment_id": experiment_id,
        "stage": stage,
        "cell_id": cell_id,
        "range": shot_range.as_json(),
        "purpose": purpose,
    }
    digest = hashlib.sha256(
        bytes.fromhex(seed_root)
        + b"patch-uf-named-seed-v1\0"
        + canonical_json_bytes(identity)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@dataclasses.dataclass(frozen=True)
class PreparedCell:
    """One selected physical cell and its four compiled adapter paths."""

    cell: Mapping[str, Any]
    circuit: Any
    dem: Any
    global_decoder: Any
    treatment_decoder: Any
    control_decoder: Any
    shadow_decoder: Any
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        cell = dict(self.cell)
        _safe_name(cell.get("cell_id"), name="cell_id")
        provenance = _jsonable(self.provenance, path="provenance")
        if not isinstance(provenance, dict):
            raise TypeError("provenance must normalize to a JSON object")
        for name in ("num_detectors", "num_observables"):
            _positive_int(provenance.get(name), name=f"provenance {name}")
        for name in (
            "global_decoder",
            "treatment_decoder",
            "control_decoder",
            "shadow_decoder",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"prepared cell requires {name}")
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "provenance", provenance)


@dataclasses.dataclass(frozen=True)
class RangeArtifacts:
    """Deterministic payloads and gzip bytes for one logical range."""

    shot_range: ShotRange
    component_payload: Mapping[str, Any]
    shot_payload: Mapping[str, Any]
    component_bytes: bytes
    shot_bytes: bytes


@dataclasses.dataclass(frozen=True)
class VerifiedCollection:
    """Read-only, fully authenticated rows supplied to characterization analysis."""

    summary: Mapping[str, Any]
    shot_rows: tuple[Mapping[str, Any], ...]
    lane_rows: tuple[Mapping[str, Any], ...]
    component_rows: tuple[Mapping[str, Any], ...]
    cluster_records: tuple[ShotClusterRecord, ...]
    control_equality: Mapping[str, Any]
    corpus_identity: Mapping[str, Any] | None
    detector_corpus_bytes: bytes
    detector_corpus_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.detector_corpus_bytes, (bytes, bytearray, memoryview)
        ):
            raise TypeError("verified detector-corpus bytes must be bytes-like")
        detector_bytes = bytes(self.detector_corpus_bytes)
        detector_sha256 = self.detector_corpus_sha256
        if not is_lowercase_hex(detector_sha256, length=64):
            raise ValueError("verified detector-corpus digest is malformed")
        if _sha256(detector_bytes) != detector_sha256:
            raise ValueError("verified detector-corpus bytes/digest mismatch")
        object.__setattr__(self, "detector_corpus_bytes", detector_bytes)


@dataclasses.dataclass(frozen=True)
class _ProtocolContext:
    protocol: dict[str, Any]
    experiment_id: str
    protocol_self_sha256: str
    source_identity: dict[str, Any]
    stage: str
    shots: int
    seed_root: str
    cell: dict[str, Any]
    dem_options: dict[str, Any]
    decoder_config: dict[str, Any]
    expected_lanes_per_shot: int
    max_components_per_shot: int
    max_metric_bytes_per_range: int


def _protocol_context(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    processes: int,
    scientific: bool,
) -> _ProtocolContext:
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    value = dict(protocol)
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("protocol schema/version mismatch")
    experiment_id = value.get("experiment_id")
    if not is_lowercase_hex(experiment_id, length=64):
        raise ValueError("protocol experiment_id must be a SHA-256 digest")
    self_hash = value.get("protocol_self_sha256")
    if not is_lowercase_hex(self_hash, length=64):
        raise ValueError("protocol_self_sha256 must be a SHA-256 digest")
    if self_hash != canonical_protocol_self_sha256(value):
        raise ValueError("protocol self hash mismatch")
    source_identity = _jsonable(value.get("source_identity"), path="source_identity")
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("protocol source_identity must be a nonempty object")
    stage = _safe_name(stage, name="stage")
    sampling = value.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("protocol sampling must be an object")
    if sampling.get("range_count") != RANGE_COUNT:
        raise ValueError(f"protocol sampling range_count must be {RANGE_COUNT}")
    if sampling.get("seed_derivation") != SEED_DERIVATION:
        raise ValueError("protocol seed derivation mismatch")
    stages = sampling.get("stages")
    if not isinstance(stages, Mapping) or stage not in stages:
        raise ValueError("requested stage is absent from protocol sampling stages")
    stage_config = stages[stage]
    if not isinstance(stage_config, Mapping):
        raise ValueError("protocol stage configuration must be an object")
    shots = _positive_int(stage_config.get("shots"), name="stage shots")
    seed_root = stage_config.get("seed_root")
    if not is_lowercase_hex(seed_root, length=64):
        raise ValueError("stage seed_root must be a SHA-256-sized lowercase hex root")
    fixed_worker_ranges(shots)
    processes = _positive_int(processes, name="processes")
    if processes > RANGE_COUNT:
        raise ValueError("engineering process count exceeds 32")
    if scientific:
        if processes != RANGE_COUNT:
            raise ValueError("scientific collection requires exactly 32 processes")
        expected_shots = SCIENTIFIC_STAGE_SHOTS.get(stage)
        if expected_shots is None or shots != expected_shots:
            raise ValueError("scientific stage/name has the wrong fixed shot count")
        if value.get("status") != "FROZEN" or value.get("frozen") is not True:
            raise ValueError("scientific collection requires a frozen protocol")
    cell = value.get("selected_cell")
    if not isinstance(cell, Mapping):
        raise ValueError("protocol selected_cell must be an object")
    cell = dict(cell)
    _safe_name(cell.get("cell_id"), name="selected cell_id")
    for name in ("d", "r", "patches"):
        _positive_int(cell.get(name), name=f"selected cell {name}")
    if cell.get("patches") != 6:
        raise ValueError("V1 selected cell must contain exactly six patches")
    if cell.get("yokes") not in (0, 1, 2):
        raise ValueError("selected cell yokes is unsupported")
    p = cell.get("p")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("selected cell p must lie in [0, 1]")
    dem_options = value.get("dem_options")
    decoder_config = value.get("decoder")
    if not isinstance(dem_options, Mapping) or not isinstance(decoder_config, Mapping):
        raise ValueError("protocol dem_options and decoder must be objects")
    if scientific:
        fixed_cell = {
            "d": 7,
            "r": 28,
            "patches": 6,
            "yokes": 2,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
        }
        for name, expected in fixed_cell.items():
            if cell.get(name) != expected:
                raise ValueError(
                    f"scientific V1 selected cell {name} must be {expected!r}"
                )
        if not isinstance(cell.get("p"), float) or cell["p"].hex() != (0.003).hex():
            raise ValueError("scientific V1 selected cell p must be exact binary64 0.003")
        if dict(dem_options) != {
            "decompose_errors": True,
            "approximate_disjoint_errors": True,
        }:
            raise ValueError("scientific V1 DEM options differ from the frozen pair")
        expected_provenance = cell.get("provenance")
        if not isinstance(expected_provenance, Mapping) or not expected_provenance:
            raise ValueError("scientific V1 requires complete selected-cell provenance")
    limits = value.get("collection_limits")
    if not isinstance(limits, Mapping):
        raise ValueError("protocol collection_limits must be an object")
    expected_lanes = _positive_int(
        limits.get("expected_lanes_per_shot"), name="expected_lanes_per_shot"
    )
    if expected_lanes != 2 * int(cell["patches"]):
        raise ValueError("expected_lanes_per_shot must equal two lanes per patch")
    max_components = _positive_int(
        limits.get("maximum_component_records_per_shot"),
        name="maximum_component_records_per_shot",
    )
    max_metric_bytes = _positive_int(
        limits.get("maximum_metric_bytes_per_range"),
        name="maximum_metric_bytes_per_range",
    )
    return _ProtocolContext(
        protocol=value,
        experiment_id=experiment_id,
        protocol_self_sha256=self_hash,
        source_identity=source_identity,
        stage=stage,
        shots=shots,
        seed_root=seed_root,
        cell=cell,
        dem_options=dict(dem_options),
        decoder_config=dict(decoder_config),
        expected_lanes_per_shot=expected_lanes,
        max_components_per_shot=max_components,
        max_metric_bytes_per_range=max_metric_bytes,
    )


def _jsonable(value: Any, *, path: str = "$") -> Any:
    """Normalize immutable adapter records without inventing telemetry."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return _jsonable(value.value, path=path)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"nonfinite adapter metric at {path}")
        return result
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), path=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string adapter metric key at {path}")
            result[key] = _jsonable(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item, path=f"{path}[]") for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f"unsupported adapter metric {type(value).__name__} at {path}")


def _policy_from_decoder_config(decoder_config: Mapping[str, Any]) -> Any:
    from fractions import Fraction

    from yoked.decoding._patch_uf_reference import BudgetLimits, UFPolicy

    raw = decoder_config.get("policy", decoder_config)
    if not isinstance(raw, Mapping):
        raise ValueError("decoder policy must be an object")
    tau = raw.get("tau")
    if isinstance(tau, str):
        if tau.startswith(("0x", "+0x", "-0x")):
            tau = Fraction.from_float(float.fromhex(tau))
        elif "/" in tau:
            numerator, denominator = tau.split("/", 1)
            tau = Fraction(int(numerator), int(denominator))
        else:
            tau = Fraction(tau)
    semantic = raw.get("semantic_limits")
    production = raw.get("production_limits")
    if not isinstance(semantic, Mapping) or not isinstance(production, Mapping):
        raise ValueError("decoder policy requires semantic_limits/production_limits")
    return UFPolicy(
        tau=tau,
        semantic_limits=BudgetLimits(**dict(semantic)),
        production_limits=BudgetLimits(**dict(production)),
    )


def _default_adapter_factory(*, dem: Any, decoder_config: Mapping[str, Any]) -> Any:
    """Compile the production adapter bundle lazily as its module evolves."""

    try:
        from yoked.decoding import _patch_uf_decoder as decoder_module
    except ImportError as ex:
        raise RuntimeError(
            "patch-UF decoder adapters are unavailable; pass adapter_factory"
        ) from ex
    compile_bundle = getattr(decoder_module, "compile_patch_uf_adapters", None)
    if compile_bundle is not None:
        return compile_bundle(dem=dem, decoder_config=dict(decoder_config))
    policy = _policy_from_decoder_config(decoder_config)
    treatment_factory = getattr(decoder_module, "PatchUFTreatmentDecoder", None)
    control_type = getattr(decoder_module, "CompiledAdapterControlDecoder", None)
    shadow_type = getattr(decoder_module, "CompiledUFShadowDecoder", None)
    global_type = getattr(decoder_module, "CompiledGlobalMWPMDecoder", None)
    if None in (treatment_factory, control_type, shadow_type, global_type):
        raise RuntimeError("_patch_uf_decoder does not expose the V1 adapter classes")
    treatment = treatment_factory(policy=policy).compile_decoder_for_dem(dem=dem)
    shared = {
        "graph": treatment.graph,
        "projection": treatment.projection,
        "compiled_lanes": treatment.compiled_lanes,
        "policy": treatment.policy,
        "num_detectors": treatment.num_detectors,
        "num_observables": treatment.num_observables,
    }
    return {
        "treatment": treatment,
        "control": control_type(**shared),
        "shadow": shadow_type(**shared),
        "global": global_type(
            graph=treatment.graph,
            num_detectors=treatment.num_detectors,
            num_observables=treatment.num_observables,
        ),
    }


def _adapter_bundle(value: Any) -> tuple[Any, Any, Any, Any | None]:
    if isinstance(value, Mapping):
        treatment = value.get("treatment")
        control = value.get("control")
        shadow = value.get("shadow")
        global_decoder = value.get("global")
    else:
        treatment = getattr(value, "treatment", value)
        control = getattr(value, "control", None)
        shadow = getattr(value, "shadow", None)
        global_decoder = getattr(value, "global_decoder", None)
    if treatment is None or control is None or shadow is None:
        raise ValueError("adapter factory must provide treatment, control, and shadow")
    return treatment, control, shadow, global_decoder


def prepare_selected_cell(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    processes: int = RANGE_COUNT,
    scientific: bool = True,
    adapter_factory: Callable[..., Any] | None = None,
) -> PreparedCell:
    """Build and authenticate the single physical cell selected by a protocol."""

    configure_single_thread_runtime()
    context = _protocol_context(
        protocol, stage=stage, processes=processes, scientific=scientific
    )
    import pymatching

    import gen
    from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

    cell = context.cell
    circuit = yoked_magic_memory_circuit(
        patch_diameter=int(cell["d"]),
        rounds=int(cell["r"]),
        noise=gen.NoiseModel.si1000(float(cell["p"])),
        style="cz",
        yokes=int(cell["yokes"]),
        num_patches=int(cell["patches"]),
        remove_x_yoke=False,
    )
    dem = circuit.detector_error_model(**context.dem_options)
    global_decoder = pymatching.Matching.from_detector_error_model(dem)
    global_decoder.ensure_num_fault_ids(dem.num_observables)
    factory = _default_adapter_factory if adapter_factory is None else adapter_factory
    treatment, control, shadow, bundled_global = _adapter_bundle(
        factory(dem=dem, decoder_config=context.decoder_config)
    )
    if bundled_global is not None:
        global_decoder = bundled_global
    graph = getattr(treatment, "graph", None)
    projection = getattr(treatment, "projection", None)
    if graph is None or projection is None:
        if scientific:
            raise ValueError("scientific treatment adapter omits graph/projection")
        graph_edges = projection_support_edges = projection_lanes = None
        layout_fingerprint = graph_fingerprint = None
        catalog_fingerprint = projection_fingerprint = None
        exact_weight_count = compiled_lane_count = None
    else:
        graph_edges = len(graph.edges)
        projection_support_edges = len(projection.support_edges)
        projection_lanes = len(projection.lanes)
        layout_fingerprint = graph.layout.fingerprint
        graph_fingerprint = graph.fingerprint
        catalog_fingerprint = projection.validated_catalog_fingerprint
        projection_fingerprint = projection.fingerprint
        exact_weight_count = len(projection.exact_weights)
        compiled_lane_count = len(getattr(treatment, "compiled_lanes", ()))
    software_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for package in ("stim", "pymatching", "sinter"):
        try:
            software_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software_versions[package] = None
    provenance: dict[str, Any] = {
        "circuit_sha256": _sha256(str(circuit).encode()),
        "dem_sha256": _sha256(str(dem).encode()),
        "num_detectors": int(dem.num_detectors),
        "num_observables": int(dem.num_observables),
        "layout_fingerprint": layout_fingerprint,
        "graph_fingerprint": graph_fingerprint,
        "validated_catalog_fingerprint": catalog_fingerprint,
        "projection_fingerprint": projection_fingerprint,
        "graph_edge_count": graph_edges,
        "projection_support_edge_count": projection_support_edges,
        "projection_lane_count": projection_lanes,
        "projection_exact_weight_count": exact_weight_count,
        "compiled_lane_count": compiled_lane_count,
        "software_versions": software_versions,
    }
    expected = cell.get("provenance")
    if scientific:
        if not isinstance(expected, Mapping) or dict(expected) != provenance:
            raise ValueError("selected-cell provenance differs from frozen protocol")
    return PreparedCell(
        cell=cell,
        circuit=circuit,
        dem=dem,
        global_decoder=global_decoder,
        treatment_decoder=treatment,
        control_decoder=control,
        shadow_decoder=shadow,
        provenance=provenance,
    )


def _canonical_gzip(value: Mapping[str, Any]) -> bytes:
    raw = canonical_json_bytes(value)
    destination = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=destination,
        compresslevel=9,
        mtime=0,
    ) as output:
        output.write(raw)
    return destination.getvalue()


def _payload_with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "payload_sha256" in result:
        raise ValueError("payload digest field is reserved")
    result["payload_sha256"] = _sha256(canonical_json_bytes(result))
    return result


def _validate_payload_digest(value: Mapping[str, Any]) -> None:
    digest = value.get("payload_sha256")
    if not is_lowercase_hex(digest, length=64):
        raise ValueError("artifact payload_sha256 is malformed")
    unsigned = dict(value)
    del unsigned["payload_sha256"]
    if _sha256(canonical_json_bytes(unsigned)) != digest:
        raise ValueError("artifact canonical payload digest mismatch")


def _load_canonical_gzip_bytes(data: bytes, *, description: str) -> dict[str, Any]:
    try:
        raw = gzip.decompress(data)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as ex:
        raise ValueError(f"cannot decode {description}") from ex
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object")
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"{description} is not canonical JSON")
    if _canonical_gzip(value) != data:
        raise ValueError(f"{description} is not deterministic gzip")
    _validate_payload_digest(value)
    return value


def _digest_json(array: np.ndarray) -> dict[str, Any]:
    value = digest_array(array)
    return {
        "sha256": value.sha256,
        "shape": list(value.shape),
        "dtype": value.dtype,
    }


def _validate_packed_array(
    value: np.ndarray,
    *,
    rows: int,
    bits: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    expected_shape = (rows, (bits + 7) // 8)
    if array.dtype != np.uint8 or array.shape != expected_shape:
        raise ValueError(
            f"{name} must have dtype uint8 and shape {expected_shape}, got "
            f"{array.dtype} {array.shape}"
        )
    result = np.ascontiguousarray(array)
    if bits % 8 and result.shape[1]:
        forbidden = np.uint8(0xFF ^ ((1 << (bits % 8)) - 1))
        if np.any(np.bitwise_and(result[:, -1], forbidden)):
            raise ValueError(f"{name} has nonzero unused tail bits")
    result.setflags(write=False)
    return result


def _ordinary_decode(adapter: Any, packed: np.ndarray) -> np.ndarray:
    method = getattr(adapter, "decode_shots_bit_packed", None)
    if method is not None:
        try:
            result = method(bit_packed_detection_event_data=packed)
        except TypeError:
            result = method(packed)
        return np.asarray(result, dtype=np.uint8)
    method = getattr(adapter, "decode_batch", None)
    if method is None:
        raise TypeError("compiled decoder has no packed batch decode method")
    try:
        return np.asarray(
            method(
                packed,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
    except TypeError:
        return np.asarray(method(packed), dtype=np.uint8)


def _telemetry_decode(adapter: Any, packed: np.ndarray) -> tuple[np.ndarray, Any]:
    for name in (
        "decode_shots_bit_packed_with_telemetry",
        "decode_shots_with_metrics",
    ):
        method = getattr(adapter, name, None)
        if method is None:
            continue
        try:
            result = method(bit_packed_detection_event_data=packed)
        except TypeError:
            result = method(packed)
        if isinstance(result, tuple) and len(result) == 2:
            return np.asarray(result[0], dtype=np.uint8), result[1]
        predictions = getattr(result, "predictions", None)
        metrics = getattr(result, "metrics", None)
        if predictions is not None and metrics is not None:
            return np.asarray(predictions, dtype=np.uint8), metrics
        raise TypeError(f"{name} must return (predictions, metrics)")
    raise TypeError("treatment decoder has no telemetry-enabled packed method")


_COMPONENT_COLLECTION_FIELDS = (
    "completed_components",
    "censored_components",
    "component_records",
    "components",
)
_NORMATIVE_COMPONENT_FIELDS = (
    "event_batch_ids",
    "event_batch_times",
    "last_membership_event_time",
    "maximum_incident_half_edge_charge",
)


def _metric_shot_index(value: Mapping[str, Any], *, name: str) -> int:
    for key in ("shot_index", "shot_offset", "local_shot_index"):
        if key in value:
            return _nonnegative_int(value[key], name=f"{name} {key}")
    raise ValueError(f"{name} must expose a local shot_index/shot_offset")


def _normalize_nested_metrics(
    metrics: Sequence[Any],
    *,
    shots: int,
    expected_lanes: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    if len(metrics) != shots:
        raise ValueError("telemetry must return exactly one ShotCorrection per shot")
    shot_rows: list[dict[str, Any]] = []
    lane_groups: list[list[dict[str, Any]]] = []
    component_groups: list[list[dict[str, Any]]] = []
    for shot_offset, raw_shot in enumerate(metrics):
        shot = _jsonable(raw_shot, path=f"metrics.shot_corrections[{shot_offset}]")
        if not isinstance(shot, dict):
            raise TypeError("ShotCorrection must normalize to an object")
        for property_name in (
            "original_detector_count",
            "residual_detector_count",
            "lane_owned_detector_count",
            "committed_defect_count",
            "cluster_summary_complete",
            "durable_support_count",
            "durable_boundary_count",
            "durable_frame_weight",
            "completed_final_component_count",
            "completed_component_size_histogram",
            "maximum_final_component_defect_count",
        ):
            if property_name not in shot and hasattr(raw_shot, property_name):
                shot[property_name] = _jsonable(
                    getattr(raw_shot, property_name),
                    path=f"metrics.shot_corrections[{shot_offset}].{property_name}",
                )
        raw_lanes = shot.pop("lane_outcomes", None)
        raw_lane_objects = getattr(raw_shot, "lane_outcomes", None)
        if not isinstance(raw_lanes, list) or len(raw_lanes) != expected_lanes:
            raise ValueError("ShotCorrection must expose the exact dense lane_outcomes")
        lanes: list[dict[str, Any]] = []
        components: list[dict[str, Any]] = []
        for lane_offset, raw_lane in enumerate(raw_lanes):
            if not isinstance(raw_lane, dict):
                raise TypeError("lane outcome must normalize to an object")
            lane = dict(raw_lane)
            lane_object = (
                raw_lane_objects[lane_offset]
                if isinstance(raw_lane_objects, Sequence)
                and lane_offset < len(raw_lane_objects)
                else None
            )
            if "last_complete_batch_id" not in lane:
                raise ValueError("lane outcome omits normative last_complete_batch_id")
            for collection_name in _COMPONENT_COLLECTION_FIELDS:
                raw_components = lane.pop(collection_name, None)
                if raw_components is None:
                    continue
                if not isinstance(raw_components, list):
                    raise TypeError(f"lane {collection_name} must be a sequence")
                component_objects = (
                    getattr(lane_object, collection_name, ())
                    if lane_object is not None
                    else ()
                )
                for component_offset, component in enumerate(raw_components):
                    if not isinstance(component, dict):
                        raise TypeError("component record must normalize to an object")
                    component = dict(component)
                    component_object = (
                        component_objects[component_offset]
                        if isinstance(component_objects, Sequence)
                        and component_offset < len(component_objects)
                        else None
                    )
                    for property_name in (
                        "cluster_defect_count",
                        "absorbed_vertex_count",
                        "partial_cluster_defect_lower_bound",
                    ):
                        if (
                            property_name not in component
                            and component_object is not None
                            and hasattr(component_object, property_name)
                        ):
                            component[property_name] = _jsonable(
                                getattr(component_object, property_name),
                                path=(
                                    f"metrics.shot_corrections[{shot_offset}]"
                                    f".lane_outcomes[{lane_offset}]"
                                    f".{collection_name}[{component_offset}]"
                                    f".{property_name}"
                                ),
                            )
                    if (
                        collection_name == "completed_components"
                        and "exact_margin" in component
                        and component["exact_margin"] is None
                    ):
                        # The semantic core uses None for an empty competing
                        # set.  Persist the protocol's explicit, unambiguous
                        # positive-infinity token instead of JSON null.
                        component["exact_margin"] = "infinity"
                    missing = set(_NORMATIVE_COMPONENT_FIELDS) - set(component)
                    if missing:
                        raise ValueError(
                            "component record omits normative fields "
                            f"{sorted(missing)}"
                        )
                    components.append(
                        {
                            "lane_offset": lane_offset,
                            "state_collection": collection_name,
                            "durable_decision": (
                                _jsonable(
                                    raw_shot.component_durable_decision(
                                        lane_offset,
                                        int(component["component_index"]),
                                    ),
                                    path="metrics.component_durable_decision",
                                )
                                if collection_name == "completed_components"
                                and hasattr(raw_shot, "component_durable_decision")
                                else None
                            ),
                            "adapter": component,
                        }
                    )
            lanes.append({"lane_offset": lane_offset, "adapter": lane})
        shot_rows.append(shot)
        lane_groups.append(lanes)
        component_groups.append(components)
    return shot_rows, lane_groups, component_groups


def _normalize_flat_metrics(
    metrics: Mapping[str, Any],
    *,
    shots: int,
    expected_lanes: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    if set(metrics) != {"shot_corrections", "lane_outcomes", "components"}:
        raise ValueError(
            "flat telemetry fields must be shot_corrections/lane_outcomes/components"
        )
    raw_shots = _jsonable(metrics["shot_corrections"], path="metrics.shot_corrections")
    raw_lanes = _jsonable(metrics["lane_outcomes"], path="metrics.lane_outcomes")
    raw_components = _jsonable(metrics["components"], path="metrics.components")
    if not all(isinstance(value, list) for value in (raw_shots, raw_lanes, raw_components)):
        raise TypeError("flat telemetry arrays must be sequences")
    if len(raw_shots) != shots:
        raise ValueError("flat telemetry shot count mismatch")
    lane_groups: list[list[dict[str, Any]]] = [[] for _ in range(shots)]
    component_groups: list[list[dict[str, Any]]] = [[] for _ in range(shots)]
    for raw in raw_lanes:
        if not isinstance(raw, dict):
            raise TypeError("flat lane outcome must be an object")
        offset = _metric_shot_index(raw, name="lane outcome")
        if offset >= shots:
            raise ValueError("lane outcome shot index exceeds range")
        if "last_complete_batch_id" not in raw:
            raise ValueError("lane outcome omits normative last_complete_batch_id")
        lane_groups[offset].append({"lane_offset": len(lane_groups[offset]), "adapter": raw})
    for raw in raw_components:
        if not isinstance(raw, dict):
            raise TypeError("flat component record must be an object")
        offset = _metric_shot_index(raw, name="component record")
        if offset >= shots:
            raise ValueError("component record shot index exceeds range")
        missing = set(_NORMATIVE_COMPONENT_FIELDS) - set(raw)
        if missing:
            raise ValueError(f"component record omits normative fields {sorted(missing)}")
        state_kind = raw.get("state_kind")
        state_collection = {
            "completed": "completed_components",
            "censored": "censored_components",
        }.get(state_kind, "components")
        component_groups[offset].append(
            {
                "lane_offset": raw.get("lane_offset"),
                "state_collection": state_collection,
                "adapter": raw,
            }
        )
    for lanes in lane_groups:
        if len(lanes) != expected_lanes:
            raise ValueError("flat telemetry does not contain the exact lane count")
    return [dict(item) for item in raw_shots], lane_groups, component_groups


def _normalize_metrics(
    metrics: Any,
    *,
    shots: int,
    expected_lanes: int,
    max_components_per_shot: int,
    max_metric_bytes: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    if isinstance(metrics, Mapping):
        result = _normalize_flat_metrics(
            metrics, shots=shots, expected_lanes=expected_lanes
        )
    elif isinstance(metrics, Sequence) and not isinstance(
        metrics, (str, bytes, bytearray)
    ):
        result = _normalize_nested_metrics(
            metrics, shots=shots, expected_lanes=expected_lanes
        )
    else:
        raise TypeError("telemetry must be a ShotCorrection sequence or flat mapping")
    shot_rows, lane_groups, component_groups = result
    for offset, components in enumerate(component_groups):
        if len(components) > max_components_per_shot:
            raise ValueError(
                f"shot {offset} exceeds maximum_component_records_per_shot"
            )
    metric_tree = {
        "shot_corrections": shot_rows,
        "lane_groups": lane_groups,
        "component_groups": component_groups,
    }
    if len(canonical_json_bytes(metric_tree)) > max_metric_bytes:
        raise ValueError("range telemetry exceeds maximum_metric_bytes_per_range")
    return result


def _required_hrlk(shot: Mapping[str, Any]) -> tuple[int, int, int, int]:
    names = (
        "original_detector_count",
        "residual_detector_count",
        "lane_owned_detector_count",
        "committed_defect_count",
    )
    values = tuple(_nonnegative_int(shot.get(name), name=name) for name in names)
    original, residual, lane_owned, committed = values
    if not 0 <= committed <= lane_owned <= original:
        raise ValueError("shot metrics violate 0 <= K <= L <= H")
    if residual != original - committed:
        raise ValueError("shot metrics violate R = H - K")
    if not isinstance(shot.get("cluster_summary_complete"), bool):
        raise ValueError("shot metrics require boolean cluster_summary_complete")
    return values


def _prediction_checks(
    *, global_prediction: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    equal_rows = np.all(global_prediction == candidate, axis=1)
    return {
        "shots": len(global_prediction),
        "equal": int(np.count_nonzero(equal_rows)),
        "mismatches": int(len(equal_rows) - np.count_nonzero(equal_rows)),
        "global_prediction": _digest_json(global_prediction),
        "candidate_prediction": _digest_json(candidate),
    }


def _identity_fields(
    *, context: _ProtocolContext, prepared: PreparedCell, shot_range: ShotRange
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": context.experiment_id,
        "protocol_self_sha256": context.protocol_self_sha256,
        "source_identity": context.source_identity,
        "stage": context.stage,
        "cell_id": prepared.cell["cell_id"],
        "range": shot_range.as_json(),
        "provenance": dict(prepared.provenance),
    }


def _range_stem(shot_range: ShotRange) -> str:
    return (
        f"range-{shot_range.range_id:02d}-"
        f"{shot_range.shot_start:08d}-{shot_range.shot_stop:08d}"
    )


def _component_relative_path(shot_range: ShotRange) -> str:
    return f"collection/component_metrics/{_range_stem(shot_range)}.json.gz"


def _shot_relative_path(shot_range: ShotRange) -> str:
    return f"collection/shards/{_range_stem(shot_range)}.json.gz"


def collect_prepared_range(
    prepared: PreparedCell,
    *,
    protocol: Mapping[str, Any],
    stage: str,
    shot_range: ShotRange,
    scientific: bool = False,
    processes: int = 1,
) -> RangeArtifacts:
    """Sample and collect one deterministic paired range directly."""

    if "MAX_ERRORS" in os.environ:
        raise ValueError("MAX_ERRORS must remain unset for fixed-N collection")
    if not isinstance(prepared, PreparedCell):
        raise TypeError("prepared must be a PreparedCell")
    context = _protocol_context(
        protocol, stage=stage, processes=processes, scientific=scientific
    )
    if shot_range not in fixed_worker_ranges(context.shots):
        raise ValueError("shot_range is not one of the protocol's fixed 32 ranges")
    if prepared.cell["cell_id"] != context.cell["cell_id"]:
        raise ValueError("prepared cell does not match selected protocol cell")
    seed = derive_named_seed(
        seed_root=context.seed_root,
        experiment_id=context.experiment_id,
        stage=context.stage,
        cell_id=str(prepared.cell["cell_id"]),
        shot_range=shot_range,
        purpose="stim-sample",
    )
    sampler = prepared.circuit.compile_detector_sampler(seed=seed)
    sampled = sampler.sample(
        shots=shot_range.shots,
        separate_observables=True,
        bit_packed=True,
    )
    if not isinstance(sampled, tuple) or len(sampled) != 2:
        raise TypeError("detector sampler must return (detectors, observables)")
    num_detectors = int(prepared.provenance["num_detectors"])
    num_observables = int(prepared.provenance["num_observables"])
    detectors = _validate_packed_array(
        sampled[0],
        rows=shot_range.shots,
        bits=num_detectors,
        name="sampled detectors",
    )
    # Keep actual observables immutable and out of every decoder call.  The
    # first row-level read occurs only after all predictions below are fixed.
    actual_observables = _validate_packed_array(
        sampled[1],
        rows=shot_range.shots,
        bits=num_observables,
        name="sampled observables",
    )
    detector_digest = digest_array(detectors)
    observable_digest = digest_array(actual_observables)

    global_prediction = _validate_packed_array(
        _ordinary_decode(prepared.global_decoder, detectors),
        rows=shot_range.shots,
        bits=num_observables,
        name="Global MWPM prediction",
    )
    ordinary_treatment = _validate_packed_array(
        _ordinary_decode(prepared.treatment_decoder, detectors),
        rows=shot_range.shots,
        bits=num_observables,
        name="ordinary treatment prediction",
    )
    telemetry_prediction_raw, raw_metrics = _telemetry_decode(
        prepared.treatment_decoder, detectors
    )
    telemetry_treatment = _validate_packed_array(
        telemetry_prediction_raw,
        rows=shot_range.shots,
        bits=num_observables,
        name="telemetry treatment prediction",
    )
    control_prediction = _validate_packed_array(
        _ordinary_decode(prepared.control_decoder, detectors),
        rows=shot_range.shots,
        bits=num_observables,
        name="Adapter-Control prediction",
    )
    shadow_prediction = _validate_packed_array(
        _ordinary_decode(prepared.shadow_decoder, detectors),
        rows=shot_range.shots,
        bits=num_observables,
        name="UF-Shadow prediction",
    )
    if not np.array_equal(ordinary_treatment, telemetry_treatment):
        raise AssertionError("ordinary and telemetry treatment predictions differ")
    if not np.array_equal(global_prediction, control_prediction):
        raise AssertionError("Adapter-Control does not bit-match Global MWPM")
    if not np.array_equal(global_prediction, shadow_prediction):
        raise AssertionError("UF-Shadow does not bit-match Global MWPM")
    for prediction in (
        global_prediction,
        ordinary_treatment,
        telemetry_treatment,
        control_prediction,
        shadow_prediction,
    ):
        prediction.setflags(write=False)
    if digest_array(detectors) != detector_digest:
        raise AssertionError("a decoder mutated the paired detector corpus")

    shot_metrics, lane_groups, component_groups = _normalize_metrics(
        raw_metrics,
        shots=shot_range.shots,
        expected_lanes=context.expected_lanes_per_shot,
        max_components_per_shot=context.max_components_per_shot,
        max_metric_bytes=context.max_metric_bytes_per_range,
    )

    # Everything above is outcome-blind.  Join immutable actual observables
    # only now, after both scientific predictions and all controls are fixed.
    shot_rows: list[dict[str, Any]] = []
    flat_lanes: list[dict[str, Any]] = []
    flat_components: list[dict[str, Any]] = []
    cross_index: list[list[int]] = []
    contingency = Counter({"a": 0, "b": 0, "c": 0, "d": 0})
    hrlk = Counter()
    for local_shot in range(shot_range.shots):
        global_shot_id = shot_range.shot_start + local_shot
        lane_start = len(flat_lanes)
        component_start = len(flat_components)
        for lane in lane_groups[local_shot]:
            flat_lanes.append(
                {"global_shot_id": global_shot_id, **lane}
            )
        for component in component_groups[local_shot]:
            flat_components.append(
                {"global_shot_id": global_shot_id, **component}
            )
        lane_count = len(flat_lanes) - lane_start
        component_count = len(flat_components) - component_start
        cross_index.append(
            [
                global_shot_id,
                lane_start,
                lane_count,
                component_start,
                component_count,
            ]
        )
        actual = actual_observables[local_shot]
        global_failed = bool(np.any(global_prediction[local_shot] != actual))
        treatment_failed = bool(
            np.any(ordinary_treatment[local_shot] != actual)
        )
        if not global_failed and not treatment_failed:
            contingency["a"] += 1
        elif not global_failed and treatment_failed:
            contingency["b"] += 1
        elif global_failed and not treatment_failed:
            contingency["c"] += 1
        else:
            contingency["d"] += 1
        h, r, l, k = _required_hrlk(shot_metrics[local_shot])
        hrlk[(h, r, l, k)] += 1
        shot_rows.append(
            {
                "global_shot_id": global_shot_id,
                "global_prediction_hex": bytes(global_prediction[local_shot]).hex(),
                "treatment_prediction_hex": bytes(
                    ordinary_treatment[local_shot]
                ).hex(),
                "actual_observables_hex": bytes(actual).hex(),
                "global_failed": global_failed,
                "treatment_failed": treatment_failed,
                "prediction_agreement": bool(
                    np.array_equal(
                        global_prediction[local_shot],
                        ordinary_treatment[local_shot],
                    )
                ),
                "lane_start": lane_start,
                "lane_count": lane_count,
                "component_start": component_start,
                "component_count": component_count,
                "adapter_metrics": shot_metrics[local_shot],
            }
        )
    if digest_array(actual_observables) != observable_digest:
        raise AssertionError("actual observables changed during paired collection")

    cross_digest = _sha256(canonical_json_bytes({"rows": cross_index}))
    common = _identity_fields(
        context=context, prepared=prepared, shot_range=shot_range
    )
    component_payload = _payload_with_digest(
        {
            "schema": COMPONENT_SCHEMA,
            **common,
            "shot_index": cross_index,
            "cross_digest": cross_digest,
            "lane_count": len(flat_lanes),
            "component_count": len(flat_components),
            "lanes": flat_lanes,
            "components": flat_components,
        }
    )
    component_bytes = _canonical_gzip(component_payload)
    component_ref = {
        "path": _component_relative_path(shot_range),
        "sha256": _sha256(component_bytes),
        "bytes": len(component_bytes),
        "canonical_payload_sha256": component_payload["payload_sha256"],
        "lane_count": len(flat_lanes),
        "component_count": len(flat_components),
        "cross_digest": cross_digest,
    }
    outside_checks = {
        "ordinary_treatment_vs_telemetry": _prediction_checks(
            global_prediction=ordinary_treatment,
            candidate=telemetry_treatment,
        ),
        "global_vs_adapter_control": _prediction_checks(
            global_prediction=global_prediction,
            candidate=control_prediction,
        ),
        "global_vs_uf_shadow": _prediction_checks(
            global_prediction=global_prediction,
            candidate=shadow_prediction,
        ),
    }
    shot_payload = _payload_with_digest(
        {
            "schema": SHOT_SCHEMA,
            **common,
            "stim_seed": seed,
            "component_file": component_ref,
            "shot_index": cross_index,
            "cross_digest": cross_digest,
            "packed_corpus": {
                "detectors": {
                    **_digest_json(detectors),
                    "data_hex": detectors.tobytes(order="C").hex(),
                },
                "observables": {
                    **_digest_json(actual_observables),
                    "data_hex": actual_observables.tobytes(order="C").hex(),
                },
            },
            "paired_contingency": dict(contingency),
            "prediction_agreements": sum(
                int(row["prediction_agreement"]) for row in shot_rows
            ),
            "hrlk_joint_histogram": [
                [h, r, l, k, count]
                for (h, r, l, k), count in sorted(hrlk.items())
            ],
            "outside_timer_checks": outside_checks,
            "shots": shot_rows,
        }
    )
    shot_bytes = _canonical_gzip(shot_payload)
    return RangeArtifacts(
        shot_range=shot_range,
        component_payload=component_payload,
        shot_payload=shot_payload,
        component_bytes=component_bytes,
        shot_bytes=shot_bytes,
    )


_COMMON_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "protocol_self_sha256",
    "source_identity",
    "stage",
    "cell_id",
    "range",
    "provenance",
}
_COMPONENT_FIELDS = {
    "schema",
    *_COMMON_ARTIFACT_FIELDS,
    "shot_index",
    "cross_digest",
    "lane_count",
    "component_count",
    "lanes",
    "components",
    "payload_sha256",
}
_SHOT_FIELDS = {
    "schema",
    *_COMMON_ARTIFACT_FIELDS,
    "stim_seed",
    "component_file",
    "shot_index",
    "cross_digest",
    "packed_corpus",
    "paired_contingency",
    "prediction_agreements",
    "hrlk_joint_histogram",
    "outside_timer_checks",
    "shots",
    "payload_sha256",
}


def _validate_common_identity(
    payload: Mapping[str, Any],
    *,
    context: _ProtocolContext,
    shot_range: ShotRange,
    provenance: Mapping[str, Any] | None,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": context.experiment_id,
        "protocol_self_sha256": context.protocol_self_sha256,
        "source_identity": context.source_identity,
        "stage": context.stage,
        "cell_id": context.cell["cell_id"],
        "range": shot_range.as_json(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"range artifact {key} identity mismatch")
    if provenance is not None and payload.get("provenance") != dict(provenance):
        raise ValueError("range artifact provenance mismatch")
    if not isinstance(payload.get("provenance"), Mapping):
        raise ValueError("range artifact provenance is malformed")


def _validate_cross_index(
    rows: Any,
    *,
    shot_range: ShotRange,
    lane_count: int,
    component_count: int,
    expected_lanes: int,
) -> list[list[int]]:
    if not isinstance(rows, list) or len(rows) != shot_range.shots:
        raise ValueError("cross index has the wrong shot count")
    lane_cursor = component_cursor = 0
    normalized: list[list[int]] = []
    for offset, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError("cross index row must contain exactly five integers")
        values = [_nonnegative_int(item, name="cross index value") for item in row]
        shot_id, lane_start, lanes, component_start, components = values
        if shot_id != shot_range.shot_start + offset:
            raise ValueError("cross index shot IDs are not exact and contiguous")
        if lane_start != lane_cursor or component_start != component_cursor:
            raise ValueError("cross index offsets are not exact and contiguous")
        if lanes != expected_lanes:
            raise ValueError("cross index lane count does not match selected cell")
        lane_cursor += lanes
        component_cursor += components
        normalized.append(values)
    if lane_cursor != lane_count or component_cursor != component_count:
        raise ValueError("cross index totals do not reconcile")
    return normalized


def _validate_component_payload(
    payload: Mapping[str, Any],
    *,
    context: _ProtocolContext,
    shot_range: ShotRange,
    provenance: Mapping[str, Any] | None,
) -> None:
    if set(payload) != _COMPONENT_FIELDS or payload.get("schema") != COMPONENT_SCHEMA:
        raise ValueError("component range schema/fields mismatch")
    _validate_payload_digest(payload)
    _validate_common_identity(
        payload,
        context=context,
        shot_range=shot_range,
        provenance=provenance,
    )
    lanes = payload["lanes"]
    components = payload["components"]
    lane_count = _nonnegative_int(payload["lane_count"], name="lane_count")
    component_count = _nonnegative_int(
        payload["component_count"], name="component_count"
    )
    if not isinstance(lanes, list) or len(lanes) != lane_count:
        raise ValueError("component range lane count mismatch")
    if not isinstance(components, list) or len(components) != component_count:
        raise ValueError("component range component count mismatch")
    index = _validate_cross_index(
        payload["shot_index"],
        shot_range=shot_range,
        lane_count=lane_count,
        component_count=component_count,
        expected_lanes=context.expected_lanes_per_shot,
    )
    if _sha256(canonical_json_bytes({"rows": index})) != payload["cross_digest"]:
        raise ValueError("component range cross digest mismatch")
    for shot_id, lane_start, lanes_for_shot, component_start, components_for_shot in index:
        for row in lanes[lane_start : lane_start + lanes_for_shot]:
            if not isinstance(row, Mapping) or row.get("global_shot_id") != shot_id:
                raise ValueError("lane row ownership disagrees with cross index")
            adapter = row.get("adapter")
            if not isinstance(adapter, Mapping) or "last_complete_batch_id" not in adapter:
                raise ValueError("lane row omits normative telemetry")
        for row in components[
            component_start : component_start + components_for_shot
        ]:
            if not isinstance(row, Mapping) or row.get("global_shot_id") != shot_id:
                raise ValueError("component row ownership disagrees with cross index")
            adapter = row.get("adapter")
            if not isinstance(adapter, Mapping):
                raise ValueError("component adapter record is malformed")
            missing = set(_NORMATIVE_COMPONENT_FIELDS) - set(adapter)
            if missing:
                raise ValueError("component row omits normative telemetry")


def _decode_corpus_array(
    value: Any, *, rows: int, bits: int, name: str
) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != {
        "sha256",
        "shape",
        "dtype",
        "data_hex",
    }:
        raise ValueError(f"packed corpus {name} fields are malformed")
    if value["shape"] != [rows, (bits + 7) // 8] or value["dtype"] != "|u1":
        raise ValueError(f"packed corpus {name} shape/dtype mismatch")
    data_hex = value["data_hex"]
    if not isinstance(data_hex, str) or data_hex.lower() != data_hex:
        raise ValueError(f"packed corpus {name} data is not lowercase hex")
    try:
        data = bytes.fromhex(data_hex)
    except ValueError as ex:
        raise ValueError(f"packed corpus {name} data is not hexadecimal") from ex
    expected_bytes = rows * ((bits + 7) // 8)
    if len(data) != expected_bytes or _sha256(data) != value["sha256"]:
        raise ValueError(f"packed corpus {name} bytes/digest mismatch")
    array = np.frombuffer(data, dtype=np.uint8).reshape(value["shape"])
    return _validate_packed_array(array, rows=rows, bits=bits, name=name)


def _decode_prediction_hex(
    value: Any, *, width: int, bits: int, name: str
) -> bytes:
    if not isinstance(value, str) or value.lower() != value or len(value) != 2 * width:
        raise ValueError(f"{name} prediction/observable hex has wrong width")
    try:
        result = bytes.fromhex(value)
    except ValueError as ex:
        raise ValueError(f"{name} is not hexadecimal") from ex
    if bits % 8 and result and result[-1] >> (bits % 8):
        raise ValueError(f"{name} has nonzero unused tail bits")
    return result


def _validate_outside_checks(value: Any, *, shots: int) -> None:
    expected_names = {
        "ordinary_treatment_vs_telemetry",
        "global_vs_adapter_control",
        "global_vs_uf_shadow",
    }
    if not isinstance(value, Mapping) or set(value) != expected_names:
        raise ValueError("outside-timer checks have incorrect fields")
    for name, item in value.items():
        if not isinstance(item, Mapping) or set(item) != {
            "shots",
            "equal",
            "mismatches",
            "global_prediction",
            "candidate_prediction",
        }:
            raise ValueError(f"outside-timer check {name} is malformed")
        if item["shots"] != shots or item["equal"] != shots or item["mismatches"] != 0:
            raise ValueError(f"outside-timer check {name} does not bit-match")
        if item["global_prediction"] != item["candidate_prediction"]:
            raise ValueError(f"outside-timer check {name} prediction digests differ")


def _validate_shot_payload(
    payload: Mapping[str, Any],
    *,
    component_payload: Mapping[str, Any],
    component_bytes: bytes,
    context: _ProtocolContext,
    shot_range: ShotRange,
    provenance: Mapping[str, Any] | None,
) -> None:
    if set(payload) != _SHOT_FIELDS or payload.get("schema") != SHOT_SCHEMA:
        raise ValueError("shot range schema/fields mismatch")
    _validate_payload_digest(payload)
    _validate_common_identity(
        payload,
        context=context,
        shot_range=shot_range,
        provenance=provenance,
    )
    for key in _COMMON_ARTIFACT_FIELDS:
        if payload[key] != component_payload[key]:
            raise ValueError("shot/component identity fields disagree")
    index = _validate_cross_index(
        payload["shot_index"],
        shot_range=shot_range,
        lane_count=int(component_payload["lane_count"]),
        component_count=int(component_payload["component_count"]),
        expected_lanes=context.expected_lanes_per_shot,
    )
    if index != component_payload["shot_index"]:
        raise ValueError("shot/component cross-index rows disagree")
    if payload["cross_digest"] != component_payload["cross_digest"]:
        raise ValueError("shot/component cross digests disagree")
    reference = payload["component_file"]
    expected_reference = {
        "path": _component_relative_path(shot_range),
        "sha256": _sha256(component_bytes),
        "bytes": len(component_bytes),
        "canonical_payload_sha256": component_payload["payload_sha256"],
        "lane_count": component_payload["lane_count"],
        "component_count": component_payload["component_count"],
        "cross_digest": component_payload["cross_digest"],
    }
    if reference != expected_reference:
        raise ValueError("shot component-file reference is inconsistent")
    expected_seed = derive_named_seed(
        seed_root=context.seed_root,
        experiment_id=context.experiment_id,
        stage=context.stage,
        cell_id=context.cell["cell_id"],
        shot_range=shot_range,
        purpose="stim-sample",
    )
    if payload["stim_seed"] != expected_seed:
        raise ValueError("shot range Stim seed mismatch")
    provenance_value = payload["provenance"]
    num_detectors = _positive_int(
        provenance_value.get("num_detectors"), name="num_detectors"
    )
    num_observables = _positive_int(
        provenance_value.get("num_observables"), name="num_observables"
    )
    corpus = payload["packed_corpus"]
    if not isinstance(corpus, Mapping) or set(corpus) != {"detectors", "observables"}:
        raise ValueError("packed corpus fields are malformed")
    detectors = _decode_corpus_array(
        corpus["detectors"],
        rows=shot_range.shots,
        bits=num_detectors,
        name="detectors",
    )
    observables = _decode_corpus_array(
        corpus["observables"],
        rows=shot_range.shots,
        bits=num_observables,
        name="observables",
    )
    del detectors
    shots = payload["shots"]
    if not isinstance(shots, list) or len(shots) != shot_range.shots:
        raise ValueError("shot ledger row count mismatch")
    contingency = Counter({"a": 0, "b": 0, "c": 0, "d": 0})
    agreements = 0
    hrlk = Counter()
    width = (num_observables + 7) // 8
    for offset, (row, cross) in enumerate(zip(shots, index)):
        if not isinstance(row, Mapping) or row.get("global_shot_id") != cross[0]:
            raise ValueError("shot row identity mismatch")
        for key, expected in zip(
            ("lane_start", "lane_count", "component_start", "component_count"),
            cross[1:],
        ):
            if row.get(key) != expected:
                raise ValueError("shot row cross offsets/counts disagree")
        actual = _decode_prediction_hex(
            row.get("actual_observables_hex"),
            width=width,
            bits=num_observables,
            name="actual observables",
        )
        if actual != bytes(observables[offset]):
            raise ValueError("shot actual observables disagree with packed corpus")
        global_prediction = _decode_prediction_hex(
            row.get("global_prediction_hex"),
            width=width,
            bits=num_observables,
            name="global",
        )
        treatment_prediction = _decode_prediction_hex(
            row.get("treatment_prediction_hex"),
            width=width,
            bits=num_observables,
            name="treatment",
        )
        global_failed = global_prediction != actual
        treatment_failed = treatment_prediction != actual
        if row.get("global_failed") is not global_failed:
            raise ValueError("shot global correctness flag mismatch")
        if row.get("treatment_failed") is not treatment_failed:
            raise ValueError("shot treatment correctness flag mismatch")
        agreement = global_prediction == treatment_prediction
        if row.get("prediction_agreement") is not agreement:
            raise ValueError("shot prediction agreement flag mismatch")
        agreements += int(agreement)
        if not global_failed and not treatment_failed:
            contingency["a"] += 1
        elif not global_failed and treatment_failed:
            contingency["b"] += 1
        elif global_failed and not treatment_failed:
            contingency["c"] += 1
        else:
            contingency["d"] += 1
        metrics = row.get("adapter_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("shot adapter_metrics is malformed")
        hrlk[_required_hrlk(metrics)] += 1
    if dict(contingency) != payload["paired_contingency"]:
        raise ValueError("shot paired contingency does not reconcile")
    if agreements != payload["prediction_agreements"]:
        raise ValueError("shot prediction agreement count does not reconcile")
    expected_hrlk = [
        [h, r, l, k, count]
        for (h, r, l, k), count in sorted(hrlk.items())
    ]
    if payload["hrlk_joint_histogram"] != expected_hrlk:
        raise ValueError("shot H/R/L/K histogram does not reconcile")
    _validate_outside_checks(payload["outside_timer_checks"], shots=shot_range.shots)


def _read_range_pair(
    *,
    out: Path,
    context: _ProtocolContext,
    shot_range: ShotRange,
    provenance: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    component_path = out / _component_relative_path(shot_range)
    shot_path = out / _shot_relative_path(shot_range)
    for path, name in ((component_path, "component"), (shot_path, "shot")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"committed {name} range must be a regular non-symlink file")
    component_bytes = component_path.read_bytes()
    shot_bytes = shot_path.read_bytes()
    component = _load_canonical_gzip_bytes(
        component_bytes, description="component range"
    )
    shot = _load_canonical_gzip_bytes(shot_bytes, description="shot range")
    _validate_component_payload(
        component,
        context=context,
        shot_range=shot_range,
        provenance=provenance,
    )
    _validate_shot_payload(
        shot,
        component_payload=component,
        component_bytes=component_bytes,
        context=context,
        shot_range=shot_range,
        provenance=provenance,
    )
    return component, shot, component_bytes, shot_bytes


def _validate_component_orphan(
    *,
    path: Path,
    context: _ProtocolContext,
    shot_range: ShotRange,
    provenance: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("orphan component range must be a regular non-symlink file")
    data = path.read_bytes()
    payload = _load_canonical_gzip_bytes(data, description="orphan component range")
    _validate_component_payload(
        payload,
        context=context,
        shot_range=shot_range,
        provenance=provenance,
    )
    return payload, data


_WORKER_PRELOAD: PreparedCell | None = None


def _preload_worker_cell(prepared: PreparedCell) -> None:
    global _WORKER_PRELOAD
    if not isinstance(prepared, PreparedCell):
        raise TypeError("worker preload must be a PreparedCell")
    _WORKER_PRELOAD = prepared


def _clear_worker_preload() -> None:
    global _WORKER_PRELOAD
    _WORKER_PRELOAD = None


def _worker_collect(task: Mapping[str, Any]) -> RangeArtifacts:
    configure_single_thread_runtime()
    if _WORKER_PRELOAD is None:
        raise RuntimeError("fork worker did not inherit the prepared selected cell")
    return collect_prepared_range(
        _WORKER_PRELOAD,
        protocol=task["protocol"],
        stage=task["stage"],
        shot_range=ShotRange(**task["shot_range"]),
        scientific=False,
        processes=1,
    )


def _validate_output_tree(
    out: Path,
    *,
    ranges: Sequence[ShotRange],
) -> Path:
    out, names = validate_resumable_output_root(
        out,
        allowed_entries={"protocol.json", "collection", "corpus"},
        description="patch-UF paired collection",
    )
    if names and "protocol.json" not in names:
        raise ValueError("partial collection is missing protocol.json")
    collection = out / "collection"
    allowed_component_paths = {
        out / _component_relative_path(shot_range) for shot_range in ranges
    }
    allowed_shot_paths = {out / _shot_relative_path(shot_range) for shot_range in ranges}
    if collection.exists():
        if collection.is_symlink() or not collection.is_dir():
            raise ValueError("collection must be a regular directory")
        entries = {entry.name: entry for entry in collection.iterdir()}
        if set(entries) - {"component_metrics", "shards", "summary.json"}:
            raise ValueError("collection contains unexpected entries")
        for directory_name, allowed in (
            ("component_metrics", allowed_component_paths),
            ("shards", allowed_shot_paths),
        ):
            directory = collection / directory_name
            if not directory.exists():
                continue
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"collection/{directory_name} is not a safe directory")
            for artifact in directory.iterdir():
                if artifact.is_symlink() or not artifact.is_file() or artifact not in allowed:
                    raise ValueError(f"unexpected or unsafe range artifact {artifact}")
        summary = collection / "summary.json"
        if summary.exists() and (summary.is_symlink() or not summary.is_file()):
            raise ValueError("collection summary must be a regular file")
    corpus = out / "corpus"
    if corpus.exists():
        if corpus.is_symlink() or not corpus.is_dir():
            raise ValueError("corpus must be a regular directory")
        allowed_names = {"detectors.bitpack", "observables.bitpack", "index.json"}
        entries = tuple(corpus.iterdir())
        if any(
            item.is_symlink()
            or not item.is_file()
            or item.name not in allowed_names
            for item in entries
        ):
            raise ValueError("corpus contains unexpected or unsafe artifacts")
    return out


def _install_or_compare(path: Path, data: bytes, *, prefix: str) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"existing deterministic artifact differs: {path}")
        return
    install_bytes_atomic(path, data, prefix=prefix, overwrite=False)


def _install_range(
    *, out: Path, artifacts: RangeArtifacts, orphan_component_bytes: bytes | None
) -> None:
    component_path = out / _component_relative_path(artifacts.shot_range)
    shot_path = out / _shot_relative_path(artifacts.shot_range)
    if orphan_component_bytes is not None:
        if (
            orphan_component_bytes != artifacts.component_bytes
            or component_path.is_symlink()
            or not component_path.is_file()
            or component_path.read_bytes() != artifacts.component_bytes
        ):
            raise ValueError(
                "regenerated component bytes differ from installed orphan; "
                "output root is not repairable in place"
            )
    else:
        install_bytes_atomic(
            component_path,
            artifacts.component_bytes,
            prefix="patch-uf-components-",
            suffix=".json.gz",
            overwrite=False,
        )
    # Installed last: this is the deterministic range commit marker.
    install_bytes_atomic(
        shot_path,
        artifacts.shot_bytes,
        prefix="patch-uf-shots-",
        suffix=".json.gz",
        overwrite=False,
    )


def _aggregate_control_equality(
    shot_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = (
        "ordinary_treatment_vs_telemetry",
        "global_vs_adapter_control",
        "global_vs_uf_shadow",
    )
    result: dict[str, Any] = {}
    for name in names:
        evidence = [payload["outside_timer_checks"][name] for payload in shot_payloads]
        result[name] = {
            "shots": sum(int(item["shots"]) for item in evidence),
            "equal": sum(int(item["equal"]) for item in evidence),
            "mismatches": sum(int(item["mismatches"]) for item in evidence),
            "ordered_range_evidence_sha256": _sha256(
                canonical_json_bytes({"ranges": evidence})
            ),
        }
    return result


def _cluster_records_from_range_rows(
    range_rows: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], bytes, bytes]
    ],
) -> tuple[ShotClusterRecord, ...]:
    records: list[ShotClusterRecord] = []
    for component_payload, shot_payload, _, _ in range_rows:
        components = component_payload["components"]
        for shot in shot_payload["shots"]:
            start = int(shot["component_start"])
            stop = start + int(shot["component_count"])
            histogram: Counter[int] = Counter()
            for component in components[start:stop]:
                if component.get("state_collection") != "completed_components":
                    continue
                adapter = component.get("adapter")
                if not isinstance(adapter, Mapping):
                    raise ValueError("component adapter record is malformed")
                size = _positive_int(
                    adapter.get("cluster_defect_count"),
                    name="completed component cluster_defect_count",
                )
                histogram[size] += 1
            complete = shot["adapter_metrics"].get("cluster_summary_complete")
            if not isinstance(complete, bool):
                raise ValueError("shot cluster-summary completeness is malformed")
            maximum = (max(histogram) if histogram else 0) if complete else None
            records.append(
                ShotClusterRecord(
                    global_shot_id=int(shot["global_shot_id"]),
                    cluster_summary_complete=complete,
                    completed_component_size_histogram=dict(histogram),
                    maximum_final_component_defect_count=maximum,
                )
            )
    records.sort(key=lambda item: item.global_shot_id)
    if len({item.global_shot_id for item in records}) != len(records):
        raise ValueError("verified cluster records contain duplicate shot IDs")
    return tuple(records)


def _cluster_summary_json(records: Sequence[ShotClusterRecord]) -> dict[str, Any]:
    rows = [
        {
            "global_shot_id": record.global_shot_id,
            "cluster_summary_complete": record.cluster_summary_complete,
            "completed_component_size_histogram": [
                list(item) for item in record.completed_component_size_histogram
            ],
            "maximum_final_component_defect_count": (
                record.maximum_final_component_defect_count
            ),
        }
        for record in records
    ]
    return {
        "complete_shots": sum(int(item.cluster_summary_complete) for item in records),
        "censored_shots": sum(int(not item.cluster_summary_complete) for item in records),
        "completed_components": sum(
            count
            for record in records
            for _, count in record.completed_component_size_histogram
        ),
        "shot_cluster_records_sha256": _sha256(
            canonical_json_bytes({"records": rows})
        ),
    }


def _aggregate_summary(
    *,
    context: _ProtocolContext,
    range_rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], bytes, bytes]],
) -> dict[str, Any]:
    contingency = Counter({"a": 0, "b": 0, "c": 0, "d": 0})
    hrlk = Counter()
    agreements = lanes = components = shots = 0
    component_digests: list[str] = []
    shot_digests: list[str] = []
    provenance: Mapping[str, Any] | None = None
    for component, shot, component_bytes, shot_bytes in range_rows:
        if provenance is None:
            provenance = component["provenance"]
        elif component["provenance"] != provenance:
            raise ValueError("range provenances disagree during aggregation")
        contingency.update(shot["paired_contingency"])
        for h, r, l, k, count in shot["hrlk_joint_histogram"]:
            hrlk[(h, r, l, k)] += count
        agreements += int(shot["prediction_agreements"])
        shots += int(shot["range"]["shots"])
        lanes += int(component["lane_count"])
        components += int(component["component_count"])
        component_digests.append(_sha256(component_bytes))
        shot_digests.append(_sha256(shot_bytes))
    shot_payloads = [row[1] for row in range_rows]
    cluster_records = _cluster_records_from_range_rows(range_rows)
    payload: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": context.experiment_id,
        "protocol_self_sha256": context.protocol_self_sha256,
        "source_identity": context.source_identity,
        "stage": context.stage,
        "cell_id": context.cell["cell_id"],
        "provenance": provenance,
        "ranges": len(range_rows),
        "shots": shots,
        "lane_records": lanes,
        "component_records": components,
        "paired_contingency": dict(contingency),
        "prediction_agreements": agreements,
        "hrlk_joint_histogram": [
            [h, r, l, k, count]
            for (h, r, l, k), count in sorted(hrlk.items())
        ],
        "cluster_summary": _cluster_summary_json(cluster_records),
        "control_equality": _aggregate_control_equality(shot_payloads),
        "ordered_component_files_sha256": _sha256(
            canonical_json_bytes({"digests": component_digests})
        ),
        "ordered_shot_files_sha256": _sha256(
            canonical_json_bytes({"digests": shot_digests})
        ),
        "corpus": None,
    }
    return _payload_with_digest(payload)


def _expected_corpus_artifacts(
    *,
    context: _ProtocolContext,
    shot_rows: Sequence[Mapping[str, Any]],
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    """Purely derive characterization corpus bytes, index, and identity."""

    detector_parts: list[bytes] = []
    observable_parts: list[bytes] = []
    ranges: list[dict[str, Any]] = []
    detector_width = observable_width = None
    for shot in shot_rows:
        packed = shot["packed_corpus"]
        detector = packed["detectors"]
        observable = packed["observables"]
        detector_parts.append(bytes.fromhex(detector["data_hex"]))
        observable_parts.append(bytes.fromhex(observable["data_hex"]))
        detector_width = int(detector["shape"][1])
        observable_width = int(observable["shape"][1])
        ranges.append(
            {
                "range": shot["range"],
                "shot_payload_sha256": shot["payload_sha256"],
                "detectors_sha256": detector["sha256"],
                "observables_sha256": observable["sha256"],
            }
        )
    detector_bytes = b"".join(detector_parts)
    observable_bytes = b"".join(observable_parts)
    index = _payload_with_digest(
        {
            "schema": CORPUS_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "experiment_id": context.experiment_id,
            "protocol_self_sha256": context.protocol_self_sha256,
            "source_identity": context.source_identity,
            "stage": context.stage,
            "cell_id": context.cell["cell_id"],
            "shots": context.shots,
            "detectors": {
                "path": "corpus/detectors.bitpack",
                "sha256": _sha256(detector_bytes),
                "bytes": len(detector_bytes),
                "shape": [context.shots, detector_width],
                "dtype": "|u1",
            },
            "observables": {
                "path": "corpus/observables.bitpack",
                "sha256": _sha256(observable_bytes),
                "bytes": len(observable_bytes),
                "shape": [context.shots, observable_width],
                "dtype": "|u1",
            },
            "ranges": ranges,
        }
    )
    identity = {
        "index_path": "corpus/index.json",
        "index_payload_sha256": index["payload_sha256"],
        "detectors_sha256": _sha256(detector_bytes),
        "observables_sha256": _sha256(observable_bytes),
    }
    return detector_bytes, observable_bytes, index, identity


def _aggregate_detector_corpus_bytes(
    range_rows: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], bytes, bytes]
    ],
) -> bytes:
    """Concatenate detector bytes from already-authenticated range shards."""

    return b"".join(
        bytes.fromhex(shot_payload["packed_corpus"]["detectors"]["data_hex"])
        for _, shot_payload, _, _ in range_rows
    )


def _install_corpus(
    *,
    out: Path,
    context: _ProtocolContext,
    shot_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    detector_bytes, observable_bytes, index, identity = _expected_corpus_artifacts(
        context=context, shot_rows=shot_rows
    )
    corpus = out / "corpus"
    _install_or_compare(
        corpus / "detectors.bitpack",
        detector_bytes,
        prefix="patch-uf-corpus-detectors-",
    )
    _install_or_compare(
        corpus / "observables.bitpack",
        observable_bytes,
        prefix="patch-uf-corpus-observables-",
    )
    _install_or_compare(
        corpus / "index.json",
        canonical_json_bytes(index),
        prefix="patch-uf-corpus-index-",
    )
    return identity


def _read_plain_canonical(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as ex:
        raise ValueError(f"cannot decode {description}") from ex
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{description} is not one canonical JSON object")
    return value


def _expected_provenance(context: _ProtocolContext) -> Mapping[str, Any] | None:
    value = context.cell.get("provenance")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("selected-cell provenance must be an object")
    return dict(value)


def _verify_complete_rows(
    *, out: Path, context: _ProtocolContext
) -> list[tuple[dict[str, Any], dict[str, Any], bytes, bytes]]:
    provenance = _expected_provenance(context)
    return [
        _read_range_pair(
            out=out,
            context=context,
            shot_range=shot_range,
            provenance=provenance,
        )
        for shot_range in fixed_worker_ranges(context.shots)
    ]


def _verify_or_write_final_artifacts(
    *,
    out: Path,
    context: _ProtocolContext,
    range_rows: Sequence[tuple[dict[str, Any], dict[str, Any], bytes, bytes]],
) -> dict[str, Any]:
    summary = _aggregate_summary(context=context, range_rows=range_rows)
    if context.stage == CHARACTERIZATION_STAGE:
        summary["corpus"] = _install_corpus(
            out=out,
            context=context,
            shot_rows=[row[1] for row in range_rows],
        )
        unsigned = dict(summary)
        unsigned.pop("payload_sha256")
        summary["payload_sha256"] = _sha256(canonical_json_bytes(unsigned))
    elif (out / "corpus").exists():
        raise ValueError("non-characterization collection may not contain a corpus")
    summary_path = out / "collection" / "summary.json"
    _install_or_compare(
        summary_path,
        canonical_json_bytes(summary),
        prefix="patch-uf-summary-",
    )
    return summary


def verify_collection(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    out: Path,
    processes: int = RANGE_COUNT,
    scientific: bool = True,
) -> VerifiedCollection:
    """Validate every committed range, cross-digest, summary, and corpus byte."""

    context = _protocol_context(
        protocol, stage=stage, processes=processes, scientific=scientific
    )
    ranges = fixed_worker_ranges(context.shots)
    out = _validate_output_tree(Path(out), ranges=ranges)
    protocol_path = out / "protocol.json"
    if _read_plain_canonical(protocol_path, description="collection protocol") != dict(
        protocol
    ):
        raise ValueError("collection protocol differs from runtime protocol")
    rows = _verify_complete_rows(out=out, context=context)
    detector_corpus_bytes = _aggregate_detector_corpus_bytes(rows)
    detector_corpus_sha256 = _sha256(detector_corpus_bytes)
    expected = _aggregate_summary(context=context, range_rows=rows)
    if context.stage == CHARACTERIZATION_STAGE:
        for path, name in (
            (out / "corpus" / "detectors.bitpack", "corpus detectors"),
            (out / "corpus" / "observables.bitpack", "corpus observables"),
            (out / "corpus" / "index.json", "corpus index"),
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{name} must be a regular non-symlink file")
        corpus_index = _read_plain_canonical(
            out / "corpus" / "index.json", description="corpus index"
        )
        _validate_payload_digest(corpus_index)
        detectors = (out / "corpus" / "detectors.bitpack").read_bytes()
        observables = (out / "corpus" / "observables.bitpack").read_bytes()
        if _sha256(detectors) != corpus_index["detectors"]["sha256"]:
            raise ValueError("corpus detector digest mismatch")
        if _sha256(observables) != corpus_index["observables"]["sha256"]:
            raise ValueError("corpus observable digest mismatch")
        # Regenerate exact aggregate bytes/index and compare without mutation.
        expected_detectors, expected_observables, expected_index, expected_corpus = (
            _expected_corpus_artifacts(
                context=context,
                shot_rows=[row[1] for row in rows],
            )
        )
        if detectors != detector_corpus_bytes:
            raise ValueError(
                "characterization detector corpus differs from authenticated ranges"
            )
        if detectors != expected_detectors or observables != expected_observables:
            raise ValueError("characterization corpus bytes differ from range shards")
        if corpus_index != expected_index:
            raise ValueError("characterization corpus index differs from range shards")
        expected["corpus"] = expected_corpus
        unsigned = dict(expected)
        unsigned.pop("payload_sha256")
        expected["payload_sha256"] = _sha256(canonical_json_bytes(unsigned))
    elif (out / "corpus").exists():
        raise ValueError("non-characterization collection contains a corpus")
    recorded = _read_plain_canonical(
        out / "collection" / "summary.json", description="collection summary"
    )
    _validate_payload_digest(recorded)
    if recorded != expected:
        raise ValueError("collection summary does not reconcile exactly")
    shot_rows = tuple(shot for _, payload, _, _ in rows for shot in payload["shots"])
    lane_rows = tuple(lane for payload, _, _, _ in rows for lane in payload["lanes"])
    component_rows = tuple(
        component
        for payload, _, _, _ in rows
        for component in payload["components"]
    )
    cluster_records = _cluster_records_from_range_rows(rows)
    return VerifiedCollection(
        summary=recorded,
        shot_rows=shot_rows,
        lane_rows=lane_rows,
        component_rows=component_rows,
        cluster_records=cluster_records,
        control_equality=recorded["control_equality"],
        corpus_identity=recorded["corpus"],
        detector_corpus_bytes=detector_corpus_bytes,
        detector_corpus_sha256=detector_corpus_sha256,
    )


def run_collection(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    out: Path,
    processes: int = RANGE_COUNT,
    scientific: bool = True,
    prepared: PreparedCell | None = None,
    adapter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Collect or resume all fixed ranges for the selected protocol cell."""

    if "MAX_ERRORS" in os.environ:
        raise ValueError("MAX_ERRORS must remain unset for fixed-N collection")
    configure_single_thread_runtime()
    context = _protocol_context(
        protocol, stage=stage, processes=processes, scientific=scientific
    )
    ranges = fixed_worker_ranges(context.shots)
    out = _validate_output_tree(Path(out), ranges=ranges)
    if prepared is None:
        prepared = prepare_selected_cell(
            protocol,
            stage=stage,
            processes=processes,
            scientific=scientific,
            adapter_factory=adapter_factory,
        )
    if prepared.cell["cell_id"] != context.cell["cell_id"]:
        raise ValueError("prepared cell differs from selected protocol cell")
    expected_provenance = _expected_provenance(context)
    if expected_provenance is not None and dict(prepared.provenance) != dict(
        expected_provenance
    ):
        raise ValueError("prepared selected-cell provenance mismatch")

    protocol_bytes = canonical_json_bytes(dict(protocol))
    protocol_path = out / "protocol.json"
    _install_or_compare(
        protocol_path, protocol_bytes, prefix="patch-uf-protocol-"
    )
    completed: dict[int, tuple[dict[str, Any], dict[str, Any], bytes, bytes]] = {}
    missing: list[ShotRange] = []
    orphan_bytes: dict[int, bytes] = {}
    for shot_range in ranges:
        component_path = out / _component_relative_path(shot_range)
        shot_path = out / _shot_relative_path(shot_range)
        if shot_path.exists() and not component_path.exists():
            raise ValueError("shot commit marker exists without its component file")
        if shot_path.exists():
            completed[shot_range.range_id] = _read_range_pair(
                out=out,
                context=context,
                shot_range=shot_range,
                provenance=dict(prepared.provenance),
            )
            continue
        if component_path.exists():
            _, data = _validate_component_orphan(
                path=component_path,
                context=context,
                shot_range=shot_range,
                provenance=dict(prepared.provenance),
            )
            orphan_bytes[shot_range.range_id] = data
        missing.append(shot_range)
    if (out / "collection" / "summary.json").exists() and missing:
        raise ValueError("collection summary exists before every range is committed")
    if (out / "corpus").exists() and missing:
        raise ValueError("corpus exists before every range is committed")

    if missing:
        _preload_worker_cell(prepared)
        tasks = [
            {
                "protocol": context.protocol,
                "stage": context.stage,
                "shot_range": {
                    "range_id": shot_range.range_id,
                    "shot_start": shot_range.shot_start,
                    "shot_stop": shot_range.shot_stop,
                },
            }
            for shot_range in missing
        ]
        try:
            with ProcessPoolExecutor(
                max_workers=processes,
                initializer=configure_single_thread_runtime,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                futures = {
                    executor.submit(_worker_collect, task): task for task in tasks
                }
                for future in as_completed(futures):
                    artifacts = future.result()
                    _install_range(
                        out=out,
                        artifacts=artifacts,
                        orphan_component_bytes=orphan_bytes.get(
                            artifacts.shot_range.range_id
                        ),
                    )
                    completed[artifacts.shot_range.range_id] = _read_range_pair(
                        out=out,
                        context=context,
                        shot_range=artifacts.shot_range,
                        provenance=dict(prepared.provenance),
                    )
        finally:
            _clear_worker_preload()
    ordered = [completed[index] for index in range(RANGE_COUNT)]
    return _verify_or_write_final_artifacts(
        out=out,
        context=context,
        range_rows=ordered,
    )
