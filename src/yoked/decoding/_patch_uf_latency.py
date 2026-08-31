"""Controlled fixed-corpus latency harness for confidence-gated Patch UF.

The harness is intentionally policy-literal agnostic.  Restart, block, warmup,
and call counts are supplied by an authenticated protocol object.  Timing
workers receive only detector and precomputed residual arrays; there is no
field, loader path, or callable argument for actual observables.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
import pickle
import platform
import time
from typing import Any, Literal, Protocol

import numpy as np

from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    install_bytes_atomic,
    load_json_artifact,
    load_json_strict,
    validate_resumable_output_root,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


CORPUS_SCHEMA = "patch-uf-latency-corpus-v1"
RESTART_SCHEMA = "patch-uf-latency-restart-v1"
SUITE_SCHEMA = "patch-uf-latency-suite-v1"

VARIANT_NAMES = (
    "global_mwpm",
    "adapter_control",
    "uf_shadow",
    "treatment",
    "backend_original",
    "backend_residual",
)

NET_TOTAL = "net_total"
ADAPTER_COST = "adapter_cost"
UF_GATE_COST = "uf_gate_cost"
RESIDUAL_APPLICATION = "residual_application"
BACKEND_RELIEF = "backend_relief"


@dataclasses.dataclass(frozen=True)
class TimedPair:
    name: str
    numerator: str
    denominator: str


FIXED_PAIRS = (
    TimedPair(NET_TOTAL, "treatment", "global_mwpm"),
    TimedPair(ADAPTER_COST, "adapter_control", "global_mwpm"),
    TimedPair(UF_GATE_COST, "uf_shadow", "adapter_control"),
    TimedPair(RESIDUAL_APPLICATION, "treatment", "uf_shadow"),
    TimedPair(BACKEND_RELIEF, "backend_residual", "backend_original"),
)
PAIR_NAMES = tuple(pair.name for pair in FIXED_PAIRS)


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_value_bytes(value: object) -> bytes:
    """Canonical JSON for arrays as well as mappings; rejects nonfinite data."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_digest(value: object) -> str:
    return _sha256(_canonical_json_value_bytes(value))


def _array_digest(value: np.ndarray) -> str:
    header = canonical_json_bytes(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    )
    return _sha256(header + b"\0" + value.tobytes(order="C"))


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclasses.dataclass(frozen=True)
class HostPolicy:
    """Exact affinity and stable host fields required for every restart."""

    cpu_affinity: tuple[int, ...]
    expected_host: tuple[tuple[str, str | None], ...]
    expected_numa_nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        affinity = tuple(
            _nonnegative_int(value, name="affinity CPU") for value in self.cpu_affinity
        )
        if not affinity or affinity != tuple(sorted(set(affinity))):
            raise ValueError("cpu_affinity must be a nonempty sorted unique tuple")
        expected = tuple(self.expected_host)
        if not expected or tuple(key for key, _ in expected) != tuple(
            sorted({key for key, _ in expected})
        ):
            raise ValueError("expected_host keys must be nonempty, sorted, and unique")
        supported = {"cpu_model", "microcode", "os", "kernel", "machine"}
        if any(key not in supported for key, _ in expected):
            raise ValueError("expected_host contains an unsupported field")
        if any(value is not None and not isinstance(value, str) for _, value in expected):
            raise TypeError("expected_host values must be strings or null")
        nodes = tuple(
            _nonnegative_int(value, name="NUMA node")
            for value in self.expected_numa_nodes
        )
        if nodes != tuple(sorted(set(nodes))):
            raise ValueError("expected_numa_nodes must be sorted and unique")
        object.__setattr__(self, "cpu_affinity", affinity)
        object.__setattr__(self, "expected_host", expected)
        object.__setattr__(self, "expected_numa_nodes", nodes)

    def to_json(self) -> dict[str, Any]:
        return {
            "cpu_affinity": list(self.cpu_affinity),
            "expected_host": {key: value for key, value in self.expected_host},
            "expected_numa_nodes": list(self.expected_numa_nodes),
        }


@dataclasses.dataclass(frozen=True)
class BatchTiming:
    """Protocol-supplied counts for one batch size."""

    batch_size: int
    restarts: int
    blocks_per_restart: int
    warmup_calls_per_variant: int
    timed_calls_per_side_per_block: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            object.__setattr__(
                self,
                field.name,
                _positive_int(getattr(self, field.name), name=field.name),
            )
        if self.blocks_per_restart % 2:
            raise ValueError("blocks_per_restart must be even for balanced AB/BA")

    def to_json(self) -> dict[str, int]:
        return {field.name: int(getattr(self, field.name)) for field in dataclasses.fields(self)}


@dataclasses.dataclass(frozen=True)
class LatencyProtocol:
    """Complete generic latency schedule with no scientific count defaults."""

    batches: tuple[BatchTiming, ...]
    schedule_seed: int
    host_policy: HostPolicy

    def __post_init__(self) -> None:
        batches = tuple(self.batches)
        if not batches or any(not isinstance(value, BatchTiming) for value in batches):
            raise TypeError("batches must be a nonempty tuple of BatchTiming")
        sizes = tuple(value.batch_size for value in batches)
        if sizes != tuple(sorted(set(sizes))):
            raise ValueError("batch sizes must be sorted and unique")
        seed = _nonnegative_int(self.schedule_seed, name="schedule_seed")
        if seed >= 2**256:
            raise ValueError("schedule_seed must lie in [0, 2**256)")
        if not isinstance(self.host_policy, HostPolicy):
            raise TypeError("host_policy must be HostPolicy")
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "schedule_seed", seed)

    def batch(self, batch_size: int) -> BatchTiming:
        for value in self.batches:
            if value.batch_size == batch_size:
                return value
        raise ValueError(f"batch_size={batch_size} is absent from the protocol")

    def to_json(self) -> dict[str, Any]:
        return {
            "batches": [value.to_json() for value in self.batches],
            "schedule_seed": f"{self.schedule_seed:064x}",
            "host_policy": self.host_policy.to_json(),
            "variant_names": list(VARIANT_NAMES),
            "pairs": [dataclasses.asdict(value) for value in FIXED_PAIRS],
        }


def tiny_smoke_protocol(
    host_policy: HostPolicy,
    *,
    batch_sizes: tuple[int, ...] = (1,),
    schedule_seed: int = 1,
) -> LatencyProtocol:
    """Returns a deliberately tiny, explicitly non-scientific test protocol."""

    return LatencyProtocol(
        batches=tuple(
            BatchTiming(
                batch_size=batch_size,
                restarts=1,
                blocks_per_restart=2,
                warmup_calls_per_variant=1,
                timed_calls_per_side_per_block=1,
            )
            for batch_size in sorted(batch_sizes)
        ),
        schedule_seed=schedule_seed,
        host_policy=host_policy,
    )


@dataclasses.dataclass(frozen=True)
class AuthenticatedLatencyCorpus:
    """Detector-only aligned original/residual corpus loaded before timing."""

    detectors: np.ndarray
    residuals: np.ndarray
    num_detectors: int
    corpus_digest: str
    global_shot_ids: tuple[int, ...]
    summary_json: tuple[bytes, ...]
    manifest_sha256: str
    provenance_json: bytes

    def __post_init__(self) -> None:
        num_detectors = _nonnegative_int(self.num_detectors, name="num_detectors")
        width = (num_detectors + 7) // 8
        arrays: list[np.ndarray] = []
        for name, raw in (("detectors", self.detectors), ("residuals", self.residuals)):
            value = np.asarray(raw)
            if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != width:
                raise ValueError(
                    f"{name} must be packed uint8 with width {width}, got "
                    f"dtype={value.dtype}, shape={value.shape}"
                )
            value = np.ascontiguousarray(value)
            if num_detectors % 8 and len(value) and width:
                unused = 0xFF ^ ((1 << (num_detectors % 8)) - 1)
                if np.any(np.bitwise_and(value[:, -1], unused)):
                    raise ValueError(f"{name} has nonzero unused detector tail bits")
            value.setflags(write=False)
            arrays.append(value)
        if arrays[0].shape != arrays[1].shape or not len(arrays[0]):
            raise ValueError("detector and residual corpora must be nonempty and aligned")
        ids = tuple(
            _nonnegative_int(value, name="global_shot_id")
            for value in self.global_shot_ids
        )
        if len(ids) != len(arrays[0]) or len(set(ids)) != len(ids):
            raise ValueError("global_shot_ids must be a corpus-row bijection")
        summaries = tuple(bytes(value) for value in self.summary_json)
        if len(summaries) != len(ids):
            raise ValueError("summary row count differs from the detector corpus")
        for shot_id, payload in zip(ids, summaries):
            try:
                row = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                raise ValueError("summary_json contains invalid JSON") from ex
            if _canonical_json_value_bytes(row) != payload:
                raise ValueError("summary_json rows must be canonical JSON")
            if not isinstance(row, Mapping) or row.get("global_shot_id") != shot_id:
                raise ValueError("summary row ID differs from corpus index")
        provenance = bytes(self.provenance_json)
        try:
            provenance_value = json.loads(provenance)
        except (UnicodeDecodeError, json.JSONDecodeError) as ex:
            raise ValueError("provenance_json contains invalid JSON") from ex
        if canonical_json_bytes(provenance_value) != provenance:
            raise ValueError("provenance_json must be canonical JSON")
        object.__setattr__(self, "detectors", arrays[0])
        object.__setattr__(self, "residuals", arrays[1])
        object.__setattr__(self, "num_detectors", num_detectors)
        object.__setattr__(
            self, "corpus_digest", _validate_sha256(self.corpus_digest, name="corpus_digest")
        )
        object.__setattr__(self, "global_shot_ids", ids)
        object.__setattr__(self, "summary_json", summaries)
        object.__setattr__(
            self,
            "manifest_sha256",
            _validate_sha256(self.manifest_sha256, name="manifest_sha256"),
        )
        object.__setattr__(self, "provenance_json", provenance)

    @property
    def row_count(self) -> int:
        return len(self.detectors)

    @property
    def workload_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple((self.corpus_digest, shot_id) for shot_id in self.global_shot_ids)


_CORPUS_FIELDS = frozenset(
    {
        "schema",
        "num_detectors",
        "corpus_digest",
        "detectors",
        "residuals",
        "global_shot_ids",
        "summaries",
        "provenance",
        "manifest_sha256",
    }
)
_FILE_FIELDS = frozenset({"path", "sha256"})
_FORBIDDEN_OBSERVABLE_KEYS = frozenset(
    {
        "actual_observables",
        "actual_observables_hex",
        "observables",
        "packed_actual_observables",
    }
)


def _reject_observable_payload(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FORBIDDEN_OBSERVABLE_KEYS:
                raise ValueError(f"actual observables are forbidden at {path}.{key}")
            _reject_observable_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_observable_payload(child, path=f"{path}[{index}]")


def _safe_artifact_path(root: Path, descriptor: Mapping[str, Any], *, name: str) -> Path:
    if set(descriptor) != _FILE_FIELDS:
        raise ValueError(f"{name} descriptor fields are malformed")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} path must be a nonempty relative path")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    candidate = unresolved.resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError(f"{name} path escapes the corpus directory")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} must be a regular file")
    expected = _validate_sha256(descriptor.get("sha256"), name=f"{name} sha256")
    if _sha256(candidate.read_bytes()) != expected:
        raise ValueError(f"{name} file digest mismatch")
    return candidate


def load_authenticated_latency_corpus(
    manifest_path: str | os.PathLike[str],
) -> AuthenticatedLatencyCorpus:
    """Loads an exact-field detector/residual manifest; observables are rejected."""

    path = Path(manifest_path)
    manifest = load_json_strict(path, description="Patch-UF latency corpus manifest")
    if set(manifest) != _CORPUS_FIELDS:
        raise ValueError("latency corpus manifest fields are malformed")
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError("latency corpus manifest has the wrong schema")
    _reject_observable_payload(manifest)
    claimed = _validate_sha256(manifest.get("manifest_sha256"), name="manifest_sha256")
    unsigned = dict(manifest)
    del unsigned["manifest_sha256"]
    if _json_digest(unsigned) != claimed:
        raise ValueError("latency corpus manifest digest mismatch")
    root = path.resolve().parent
    detector_path = _safe_artifact_path(root, manifest["detectors"], name="detectors")
    residual_path = _safe_artifact_path(root, manifest["residuals"], name="residuals")
    summary_path = _safe_artifact_path(root, manifest["summaries"], name="summaries")
    detectors = np.load(detector_path, allow_pickle=False)
    residuals = np.load(residual_path, allow_pickle=False)
    summary_rows = load_json_artifact(summary_path)
    if not isinstance(summary_rows, list):
        raise ValueError("latency summaries must be a JSON list")
    _reject_observable_payload(summary_rows)
    provenance = manifest.get("provenance")
    _reject_observable_payload(provenance)
    canonical_json_bytes(provenance)
    ids = manifest.get("global_shot_ids")
    if not isinstance(ids, list):
        raise ValueError("global_shot_ids must be a JSON list")
    return AuthenticatedLatencyCorpus(
        detectors=detectors,
        residuals=residuals,
        num_detectors=manifest["num_detectors"],
        corpus_digest=manifest["corpus_digest"],
        global_shot_ids=tuple(ids),
        summary_json=tuple(canonical_json_bytes(row) for row in summary_rows),
        manifest_sha256=claimed,
        provenance_json=canonical_json_bytes(provenance),
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def write_authenticated_latency_corpus(
    out_dir: str | os.PathLike[str],
    *,
    detectors: np.ndarray,
    residuals: np.ndarray,
    num_detectors: int,
    global_shot_ids: Sequence[int],
    summaries: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    corpus_digest: str | None = None,
) -> Path:
    """Atomically writes the detector-only corpus format used by workers."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or any(output.iterdir()):
        raise ValueError("latency corpus output must be an empty regular directory")
    detector_array = np.asarray(detectors)
    residual_array = np.asarray(residuals)
    detector_bytes = _npy_bytes(detector_array)
    residual_bytes = _npy_bytes(residual_array)
    summary_value = [dict(row) for row in summaries]
    summary_bytes = _canonical_json_value_bytes(summary_value) + b"\n"
    install_bytes_atomic(
        output / "detectors.npy",
        detector_bytes,
        prefix="patch-uf-lat-det-",
        overwrite=False,
    )
    install_bytes_atomic(
        output / "residuals.npy",
        residual_bytes,
        prefix="patch-uf-lat-res-",
        overwrite=False,
    )
    install_bytes_atomic(
        output / "summaries.json",
        summary_bytes,
        prefix="patch-uf-lat-sum-",
        overwrite=False,
    )
    unsigned = {
        "schema": CORPUS_SCHEMA,
        "num_detectors": int(num_detectors),
        "corpus_digest": (
            _array_digest(detector_array) if corpus_digest is None else corpus_digest
        ),
        "detectors": {"path": "detectors.npy", "sha256": _sha256(detector_bytes)},
        "residuals": {"path": "residuals.npy", "sha256": _sha256(residual_bytes)},
        "global_shot_ids": [int(value) for value in global_shot_ids],
        "summaries": {"path": "summaries.json", "sha256": _sha256(summary_bytes)},
        "provenance": dict(provenance),
    }
    manifest = {**unsigned, "manifest_sha256": _json_digest(unsigned)}
    manifest_path = output / "manifest.json"
    install_bytes_atomic(
        manifest_path,
        canonical_json_bytes(manifest) + b"\n",
        prefix="patch-uf-lat-manifest-",
        overwrite=False,
    )
    # Round-trip validation makes the writer fail closed before returning.
    load_authenticated_latency_corpus(manifest_path)
    return manifest_path


CorpusKind = Literal["detectors", "residuals"]
TimerScope = Literal["total-adapter", "matcher-only"]


@dataclasses.dataclass(frozen=True)
class TimedVariant:
    name: str
    corpus_kind: CorpusKind
    timer_scope: TimerScope
    function: Callable[[np.ndarray], Any]

    def __post_init__(self) -> None:
        if self.name not in VARIANT_NAMES:
            raise ValueError(f"unknown timed variant {self.name!r}")
        if self.corpus_kind not in ("detectors", "residuals"):
            raise ValueError("corpus_kind must be detectors or residuals")
        if self.timer_scope not in ("total-adapter", "matcher-only"):
            raise ValueError("timer_scope is invalid")
        if not callable(self.function):
            raise TypeError("timed variant function must be callable")


@dataclasses.dataclass(frozen=True)
class _PackedDecoderCall:
    compiled_decoder: Any

    def __call__(self, packed: np.ndarray) -> Any:
        return self.compiled_decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed
        )


@dataclasses.dataclass(frozen=True)
class _PrevalidatedBackendCall:
    compiled_decoder: Any

    def __call__(self, packed: np.ndarray) -> Any:
        return self.compiled_decoder.invoke_backend_prevalidated(packed)


def build_timed_variants(
    *,
    global_mwpm: Any,
    adapter_control: Any,
    uf_shadow: Any,
    treatment: Any,
) -> tuple[TimedVariant, ...]:
    """Binds compiled adapters to the six generic timing contracts."""

    total = {
        "global_mwpm": global_mwpm,
        "adapter_control": adapter_control,
        "uf_shadow": uf_shadow,
        "treatment": treatment,
    }
    for name, compiled in total.items():
        if not callable(getattr(compiled, "decode_shots_bit_packed", None)):
            raise TypeError(f"compiled adapter {name!r} lacks packed decode")
    for name, compiled in (
        ("backend_original", global_mwpm),
        ("backend_residual", treatment),
    ):
        if not callable(getattr(compiled, "invoke_backend_prevalidated", None)):
            raise TypeError(f"compiled adapter {name!r} lacks matcher-only invocation")
    return (
        *(
            TimedVariant(
                name=name,
                corpus_kind="detectors",
                timer_scope="total-adapter",
                function=_PackedDecoderCall(total[name]),
            )
            for name in VARIANT_NAMES[:4]
        ),
        TimedVariant(
            "backend_original",
            "detectors",
            "matcher-only",
            _PrevalidatedBackendCall(global_mwpm),
        ),
        TimedVariant(
            "backend_residual",
            "residuals",
            "matcher-only",
            _PrevalidatedBackendCall(treatment),
        ),
    )


@dataclasses.dataclass(frozen=True)
class LatencyWorkload:
    """All restart-local state, structurally excluding actual observables."""

    corpus: AuthenticatedLatencyCorpus
    variants: tuple[TimedVariant, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, AuthenticatedLatencyCorpus):
            raise TypeError("corpus must be AuthenticatedLatencyCorpus")
        variants = tuple(self.variants)
        if tuple(value.name for value in variants) != VARIANT_NAMES:
            raise ValueError("variants must contain the six names in canonical order")
        expected = {
            "global_mwpm": ("detectors", "total-adapter"),
            "adapter_control": ("detectors", "total-adapter"),
            "uf_shadow": ("detectors", "total-adapter"),
            "treatment": ("detectors", "total-adapter"),
            "backend_original": ("detectors", "matcher-only"),
            "backend_residual": ("residuals", "matcher-only"),
        }
        for value in variants:
            if (value.corpus_kind, value.timer_scope) != expected[value.name]:
                raise ValueError(f"variant {value.name!r} has the wrong timing contract")
        canonical_json_bytes(dict(self.provenance))
        object.__setattr__(self, "variants", variants)

    def variant_map(self) -> dict[str, TimedVariant]:
        return {value.name: value for value in self.variants}


class LatencyRestartFactory(Protocol):
    suite_identity: Mapping[str, Any] | Callable[[], Mapping[str, Any]]

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload: ...


@dataclasses.dataclass(frozen=True)
class LatencyRestartTask:
    factory: LatencyRestartFactory
    protocol: LatencyProtocol
    restart_index: int
    batch_size: int
    protocol_id: str
    suite_id: str
    workload_id: str
    workload_identity: Mapping[str, Any]


def _seed_bytes(value: int) -> bytes:
    return int(value).to_bytes(32, "little", signed=False)


def _derive_int(seed: int, *parts: object) -> int:
    payload = bytearray(_seed_bytes(seed))
    for part in parts:
        encoded = str(part).encode("utf-8")
        payload.extend(len(encoded).to_bytes(4, "little"))
        payload.extend(encoded)
    return int.from_bytes(hashlib.sha256(payload).digest(), "little")


def balanced_pair_orders(*, blocks: int, seed: int, pair_name: str) -> tuple[str, ...]:
    blocks = _positive_int(blocks, name="blocks")
    if blocks % 2:
        raise ValueError("blocks must be even")
    if pair_name not in PAIR_NAMES:
        raise ValueError("unknown pair name")
    labels = ["AB"] * (blocks // 2) + ["BA"] * (blocks // 2)
    return tuple(
        label
        for _, label in sorted(
            enumerate(labels),
            key=lambda item: hashlib.sha256(
                _seed_bytes(seed)
                + pair_name.encode("utf-8")
                + item[0].to_bytes(8, "little")
            ).digest(),
        )
    )


def _deterministic_order(names: Sequence[str], *, seed: int, purpose: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            names,
            key=lambda name: hashlib.sha256(
                _seed_bytes(seed) + purpose.encode() + name.encode()
            ).digest(),
        )
    )


def _first_cpu_field(field: str) -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == field:
                return value.strip()
    except OSError:
        pass
    return None


def _read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _numa_nodes(affinity: Sequence[int]) -> tuple[int, ...]:
    result: set[int] = set()
    for cpu in affinity:
        try:
            entries = Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node*")
            for entry in entries:
                suffix = entry.name.removeprefix("node")
                if suffix.isdigit():
                    result.add(int(suffix))
        except OSError:
            continue
    return tuple(sorted(result))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> dict[str, Any]:
    try:
        affinity = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity = ()
    uname = platform.uname()
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = []
    return {
        "pid": os.getpid(),
        "cpu_model": _first_cpu_field("model name"),
        "microcode": _first_cpu_field("microcode"),
        "cpu_affinity": list(affinity),
        "numa_nodes": list(_numa_nodes(affinity)),
        "os": uname.system,
        "kernel": uname.release,
        "machine": uname.machine,
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("stim", "sinter", "pymatching", "numpy", "scipy")
        },
        "governor_cpu0": _read_optional(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "frequency_khz_cpu0": _read_optional(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        ),
        "intel_pstate_no_turbo": _read_optional(
            "/sys/devices/system/cpu/intel_pstate/no_turbo"
        ),
        "cpufreq_boost": _read_optional("/sys/devices/system/cpu/cpufreq/boost"),
        "load_average": load_average,
        "native_thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
    }


def capture_host_policy(*, cpu: int | None = None) -> HostPolicy:
    """Freezes the current host and one allowed CPU for a later spawned task."""

    provenance = runtime_provenance()
    affinity = tuple(provenance["cpu_affinity"])
    if not affinity:
        raise RuntimeError("CPU affinity is unavailable on this host")
    selected = min(affinity) if cpu is None else _nonnegative_int(cpu, name="cpu")
    if selected not in affinity:
        raise ValueError(f"CPU {selected} is outside the current affinity")
    expected = tuple(
        (key, provenance[key])
        for key in ("cpu_model", "kernel", "machine", "microcode", "os")
    )
    return HostPolicy(
        cpu_affinity=(selected,),
        expected_host=tuple(sorted(expected)),
        expected_numa_nodes=_numa_nodes((selected,)),
    )


def _validate_host(policy: HostPolicy, provenance: Mapping[str, Any]) -> None:
    if tuple(provenance.get("cpu_affinity", ())) != policy.cpu_affinity:
        raise RuntimeError("latency CPU affinity differs from the frozen policy")
    if tuple(provenance.get("numa_nodes", ())) != policy.expected_numa_nodes:
        raise RuntimeError("latency NUMA placement differs from the frozen policy")
    for key, expected in policy.expected_host:
        if provenance.get(key) != expected:
            raise RuntimeError(f"latency host field {key!r} differs from policy")


@contextmanager
def _host_controls(policy: HostPolicy):
    try:
        previous_affinity = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError) as ex:
        raise RuntimeError("strict CPU affinity is unavailable") from ex
    previous_threads = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    os.sched_setaffinity(0, set(policy.cpu_affinity))
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        try:
            from threadpoolctl import threadpool_limits

            limiter = threadpool_limits(limits=1)
        except ImportError:
            limiter = nullcontext()
        with limiter:
            start = runtime_provenance()
            _validate_host(policy, start)
            yield start
            _validate_host(policy, runtime_provenance())
    finally:
        os.sched_setaffinity(0, set(previous_affinity))
        for name, value in previous_threads.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclasses.dataclass(frozen=True)
class _ExtendedCorpus:
    original: np.ndarray
    extended: np.ndarray
    digest: str


def _extend_corpus(value: np.ndarray, *, batch_size: int) -> _ExtendedCorpus:
    if len(value) < batch_size:
        raise ValueError("corpus has fewer rows than the protocol batch size")
    extension = batch_size - 1
    extended = np.concatenate((value, value[:extension]), axis=0)
    extended.setflags(write=False)
    return _ExtendedCorpus(value, extended, _array_digest(extended))


def _slice_indices(start: int, *, batch_size: int, row_count: int) -> tuple[int, ...]:
    return tuple((start + offset) % row_count for offset in range(batch_size))


def _summary_digest(corpus: AuthenticatedLatencyCorpus, indices: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        row = corpus.summary_json[index]
        digest.update(len(row).to_bytes(8, "little"))
        digest.update(row)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class _ScheduledCall:
    pair: str
    block: int
    call_index: int
    start_offset: int
    indices: tuple[int, ...]
    schedule_id: str
    summary_digest: str


def _schedule_calls(
    corpus: AuthenticatedLatencyCorpus,
    *,
    seed: int,
    restart_index: int,
    batch: BatchTiming,
    pair: TimedPair,
) -> tuple[tuple[_ScheduledCall, ...], ...]:
    blocks: list[tuple[_ScheduledCall, ...]] = []
    for block in range(batch.blocks_per_restart):
        base = _derive_int(
            seed, restart_index, batch.batch_size, pair.name, block, "cyclic-offset"
        ) % corpus.row_count
        schedule_id = hashlib.sha256(
            canonical_json_bytes(
                {
                    "restart": restart_index,
                    "batch_size": batch.batch_size,
                    "pair": pair.name,
                    "block": block,
                    "base_offset": base,
                }
            )
        ).hexdigest()
        calls: list[_ScheduledCall] = []
        for call_index in range(batch.timed_calls_per_side_per_block):
            start = (base + call_index * batch.batch_size) % corpus.row_count
            indices = _slice_indices(
                start, batch_size=batch.batch_size, row_count=corpus.row_count
            )
            calls.append(
                _ScheduledCall(
                    pair.name,
                    block,
                    call_index,
                    start,
                    indices,
                    schedule_id,
                    _summary_digest(corpus, indices),
                )
            )
        blocks.append(tuple(calls))
    return tuple(blocks)


def _prediction_digest(value: object, *, name: str, rows: int) -> tuple[np.ndarray, str]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(f"variant {name!r} returned malformed packed predictions")
    return array, _array_digest(array)


def _untimed_prediction_checks(
    workload: LatencyWorkload,
    *,
    row_index: int,
    expected_full_corpus_attestation_sha256: object,
) -> dict[str, Any]:
    variants = workload.variant_map()
    corpus = workload.corpus
    row_index = _nonnegative_int(row_index, name="untimed prediction row index")
    if row_index >= corpus.row_count:
        raise ValueError("untimed prediction row index is outside the corpus")
    actual_attestation = workload.provenance.get(
        "full_corpus_prediction_attestation_sha256"
    )
    if expected_full_corpus_attestation_sha256 is not None:
        expected_attestation = _validate_sha256(
            expected_full_corpus_attestation_sha256,
            name="full-corpus materialization attestation",
        )
        if actual_attestation != expected_attestation:
            raise ValueError(
                "restart full-corpus materialization attestation mismatch"
            )
    elif actual_attestation is not None:
        expected_attestation = _validate_sha256(
            actual_attestation,
            name="full-corpus materialization attestation",
        )
    else:
        expected_attestation = None
    inputs = {
        "detectors": corpus.detectors[row_index : row_index + 1],
        "residuals": corpus.residuals[row_index : row_index + 1],
    }
    predictions: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in VARIANT_NAMES:
        variant = variants[name]
        value, digest = _prediction_digest(
            variant.function(inputs[variant.corpus_kind]),
            name=name,
            rows=1,
        )
        predictions[name] = value
        digests[name] = digest
    for name in ("adapter_control", "uf_shadow", "backend_original"):
        if not np.array_equal(predictions[name], predictions["global_mwpm"]):
            raise ValueError(f"untimed prediction equality failed for {name}")
    if not np.array_equal(predictions["treatment"], predictions["backend_residual"]):
        raise ValueError("untimed treatment/backend-residual prediction equality failed")
    return {
        "checked_rows": 1,
        "corpus_index": row_index,
        "global_shot_id": corpus.global_shot_ids[row_index],
        "full_corpus_prediction_attestation_sha256": expected_attestation,
        "prediction_digests": digests,
        "equality_groups": {
            "global_control_shadow_backend_original": [
                "global_mwpm",
                "adapter_control",
                "uf_shadow",
                "backend_original",
            ],
            "treatment_backend_residual": [
                "treatment",
                "backend_residual",
            ],
        },
    }


def _timing_call_id(
    *,
    restart_index: int,
    batch_size: int,
    call: _ScheduledCall,
    side: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "restart": restart_index,
                "batch_size": batch_size,
                "pair": call.pair,
                "side": side,
                "block": call.block,
                "call_index": call.call_index,
            }
        )
    ).hexdigest()


def _measure_side(
    *,
    variant: TimedVariant,
    side: str,
    calls: tuple[_ScheduledCall, ...],
    extended: _ExtendedCorpus,
    corpus: AuthenticatedLatencyCorpus,
    restart_index: int,
    batch_size: int,
    clock: Callable[[], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call in calls:
        batch_view = extended.extended[
            call.start_offset : call.start_offset + batch_size
        ]
        if not np.array_equal(
            batch_view, extended.original[np.asarray(call.indices, dtype=np.intp)]
        ):
            raise AssertionError("extended cyclic corpus slice is inconsistent")
        detector_digest = _array_digest(batch_view)
        before = int(clock())
        result = variant.function(batch_view)
        after = int(clock())
        del result
        duration = after - before
        if duration <= 0:
            raise RuntimeError("latency clock produced a non-positive duration")
        workload_keys = [list(corpus.workload_keys[index]) for index in call.indices]
        rows.append(
            {
                "timing_call_id": _timing_call_id(
                    restart_index=restart_index,
                    batch_size=batch_size,
                    call=call,
                    side=side,
                ),
                "restart": restart_index,
                "batch_size": batch_size,
                "pair": call.pair,
                "side": side,
                "block": call.block,
                "call_index": call.call_index,
                "schedule_id": call.schedule_id,
                "start_offset": call.start_offset,
                "corpus_indices": list(call.indices),
                "workload_keys": workload_keys,
                "workload_key_digest": _json_digest(workload_keys),
                "detector_batch_digest": detector_digest,
                "precomputed_summary_digest": call.summary_digest,
                "duration_ns": duration,
            }
        )
    return rows


def _warmup(
    variant: TimedVariant,
    extended: _ExtendedCorpus,
    *,
    calls: int,
    batch_size: int,
    start: int,
    row_count: int,
) -> None:
    for call_index in range(calls):
        offset = (start + call_index * batch_size) % row_count
        result = variant.function(extended.extended[offset : offset + batch_size])
        del result


def _measure_restart(
    workload: LatencyWorkload,
    *,
    task: LatencyRestartTask,
    clock: Callable[[], int],
    runtime_start: Mapping[str, Any],
) -> dict[str, Any]:
    batch = task.protocol.batch(task.batch_size)
    corpus = workload.corpus
    if corpus.row_count < batch.batch_size:
        raise ValueError("authenticated corpus is smaller than the batch size")
    variants = workload.variant_map()
    before_digests = {
        "detectors": _array_digest(corpus.detectors),
        "residuals": _array_digest(corpus.residuals),
    }
    seed = _derive_int(
        task.protocol.schedule_seed,
        task.restart_index,
        task.batch_size,
        "restart",
    )
    preflight_row = _derive_int(
        task.protocol.schedule_seed,
        task.restart_index,
        task.batch_size,
        "untimed-equality-row",
    ) % corpus.row_count
    equality_check = _untimed_prediction_checks(
        workload,
        row_index=preflight_row,
        expected_full_corpus_attestation_sha256=task.workload_identity.get(
            "full_corpus_prediction_attestation_sha256"
        ),
    )
    extended = {
        "detectors": _extend_corpus(corpus.detectors, batch_size=batch.batch_size),
        "residuals": _extend_corpus(corpus.residuals, batch_size=batch.batch_size),
    }
    warmup_order = _deterministic_order(
        VARIANT_NAMES, seed=seed, purpose="six-variant-warmup"
    )
    pair_order = _deterministic_order(PAIR_NAMES, seed=seed, purpose="pair-order")
    pair_by_name = {value.name: value for value in FIXED_PAIRS}
    schedules = {
        name: _schedule_calls(
            corpus,
            seed=task.protocol.schedule_seed,
            restart_index=task.restart_index,
            batch=batch,
            pair=pair_by_name[name],
        )
        for name in PAIR_NAMES
    }
    orders = {
        name: balanced_pair_orders(
            blocks=batch.blocks_per_restart,
            seed=_derive_int(seed, name, "ABBA"),
            pair_name=name,
        )
        for name in PAIR_NAMES
    }

    previous_gc = gc.isenabled()
    pair_records: dict[str, Any] = {}
    gc.disable()
    try:
        for name in warmup_order:
            variant = variants[name]
            start = _derive_int(seed, name, "warmup-offset") % corpus.row_count
            _warmup(
                variant,
                extended[variant.corpus_kind],
                calls=batch.warmup_calls_per_variant,
                batch_size=batch.batch_size,
                start=start,
                row_count=corpus.row_count,
            )
        for pair_name in pair_order:
            pair = pair_by_name[pair_name]
            numerator = variants[pair.numerator]
            denominator = variants[pair.denominator]
            numerator_rows: list[list[dict[str, Any]]] = []
            denominator_rows: list[list[dict[str, Any]]] = []
            for block_index, order in enumerate(orders[pair_name]):
                calls = schedules[pair_name][block_index]

                def measure_numerator() -> list[dict[str, Any]]:
                    return _measure_side(
                        variant=numerator,
                        side="numerator",
                        calls=calls,
                        extended=extended[numerator.corpus_kind],
                        corpus=corpus,
                        restart_index=task.restart_index,
                        batch_size=batch.batch_size,
                        clock=clock,
                    )

                def measure_denominator() -> list[dict[str, Any]]:
                    return _measure_side(
                        variant=denominator,
                        side="denominator",
                        calls=calls,
                        extended=extended[denominator.corpus_kind],
                        corpus=corpus,
                        restart_index=task.restart_index,
                        batch_size=batch.batch_size,
                        clock=clock,
                    )

                if order == "AB":
                    numerator_rows.append(measure_numerator())
                    denominator_rows.append(measure_denominator())
                elif order == "BA":
                    denominator_rows.append(measure_denominator())
                    numerator_rows.append(measure_numerator())
                else:  # pragma: no cover
                    raise AssertionError("invalid internal pair order")
            pair_records[pair_name] = {
                "pair": pair_name,
                "numerator": pair.numerator,
                "denominator": pair.denominator,
                "order_by_block": list(orders[pair_name]),
                "numerator_calls": numerator_rows,
                "denominator_calls": denominator_rows,
                "numerator_block_totals_ns": [
                    sum(call["duration_ns"] for call in row) for row in numerator_rows
                ],
                "denominator_block_totals_ns": [
                    sum(call["duration_ns"] for call in row) for row in denominator_rows
                ],
            }
    finally:
        if previous_gc:
            gc.enable()

    after_digests = {
        "detectors": _array_digest(corpus.detectors),
        "residuals": _array_digest(corpus.residuals),
    }
    if after_digests != before_digests:
        raise RuntimeError("a timed variant mutated the authenticated corpus")
    for name, value in extended.items():
        if _array_digest(value.extended) != value.digest:
            raise RuntimeError(f"a timed variant mutated the extended {name} corpus")
    runtime_end = runtime_provenance()
    _validate_host(task.protocol.host_policy, runtime_end)
    record = {
        "schema": RESTART_SCHEMA,
        "protocol_id": task.protocol_id,
        "suite_id": task.suite_id,
        "workload_id": task.workload_id,
        "workload_identity": dict(task.workload_identity),
        "restart_index": task.restart_index,
        "batch_size": task.batch_size,
        "clock": (
            "time.perf_counter_ns"
            if clock is time.perf_counter_ns
            else "explicit-test-clock"
        ),
        "timing_scope": {
            "total": "public-adapter-entry-to-packed-prediction-return",
            "backend": "matcher-decode-batch-invocation-to-return",
            "gc_disabled_during_warmup_and_timing": True,
            "actual_observables_available": False,
            "native_threads": 1,
        },
        "warmup": {
            "variant_order": list(warmup_order),
            "calls_per_variant": batch.warmup_calls_per_variant,
        },
        "pair_execution_order": list(pair_order),
        "corpus": {
            "manifest_sha256": corpus.manifest_sha256,
            "corpus_digest": corpus.corpus_digest,
            "row_count": corpus.row_count,
            "num_detectors": corpus.num_detectors,
            "original_array_digest": before_digests["detectors"],
            "residual_array_digest": before_digests["residuals"],
            "extended_original_digest": extended["detectors"].digest,
            "extended_residual_digest": extended["residuals"].digest,
        },
        "untimed_prediction_check": equality_check,
        "provenance": {
            "runtime_start": dict(runtime_start),
            "runtime_end": runtime_end,
            "workload": dict(workload.provenance),
            "corpus": json.loads(corpus.provenance_json),
        },
        "pairs": pair_records,
    }
    canonical_json_bytes(record)
    return record


def run_latency_restart(
    task: LatencyRestartTask,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Runs one restart in-process; suite collection uses a fresh spawn."""

    if not isinstance(task, LatencyRestartTask):
        raise TypeError("task must be LatencyRestartTask")
    batch = task.protocol.batch(task.batch_size)
    restart = _nonnegative_int(task.restart_index, name="restart_index")
    if restart >= batch.restarts:
        raise ValueError("restart_index is outside the batch protocol")
    if not callable(task.factory):
        raise TypeError("task factory must be callable")
    expected_protocol_id = _json_digest(task.protocol.to_json())
    if task.protocol_id != expected_protocol_id:
        raise ValueError("restart task protocol_id is inconsistent")
    if task.workload_id != _json_digest(dict(task.workload_identity)):
        raise ValueError("restart task workload_id is inconsistent")
    expected_suite_id = _json_digest(
        {
            "protocol_id": task.protocol_id,
            "workload_id": task.workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
        }
    )
    if task.suite_id != expected_suite_id:
        raise ValueError("restart task suite_id is inconsistent")
    if _factory_identity(task.factory) != dict(task.workload_identity):
        raise ValueError("restart factory identity differs from the serialized task")
    with _host_controls(task.protocol.host_policy) as runtime_start:
        workload = task.factory(restart, task.batch_size)
        if not isinstance(workload, LatencyWorkload):
            raise TypeError("restart factory must return LatencyWorkload")
        corpus_identity = {
            "corpus_manifest_sha256": workload.corpus.manifest_sha256,
            "corpus_digest": workload.corpus.corpus_digest,
        }
        for key, actual in corpus_identity.items():
            if task.workload_identity.get(key) != actual:
                raise ValueError(f"restart corpus identity {key!r} differs from suite")
        return _measure_restart(
            workload,
            task=task,
            clock=clock,
            runtime_start=runtime_start,
        )


def run_latency_restart_worker(task: LatencyRestartTask) -> dict[str, Any]:
    """Top-level spawned-process entry point."""

    return run_latency_restart(task)


def write_restart_ledger_atomic(
    path: str | os.PathLike[str], record: Mapping[str, Any]
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    install_bytes_atomic(
        destination,
        canonical_json_bytes(dict(record)) + b"\n",
        prefix="patch-uf-latency-restart-",
        overwrite=False,
    )


def _factory_identity(factory: LatencyRestartFactory) -> dict[str, Any]:
    raw = getattr(factory, "suite_identity", None)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("latency factory requires a nonempty suite_identity")
    value = dict(raw)
    canonical_json_bytes(value)
    required = {"corpus_manifest_sha256", "corpus_digest"}
    if required - set(value):
        raise ValueError(
            "latency factory suite_identity is missing "
            f"{sorted(required - set(value))}"
        )
    for key in required:
        _validate_sha256(value[key], name=f"suite_identity {key}")
    return value


def _restart_name(batch_size: int, restart: int) -> str:
    return f"batch-{batch_size}.restart-{restart:02d}.json"


def _validate_existing_restart(
    record: Mapping[str, Any],
    *,
    protocol_id: str,
    suite_id: str,
    workload_id: str,
    workload_identity: Mapping[str, Any] | None = None,
    protocol: LatencyProtocol | None = None,
    batch_size: int | None = None,
    batch: BatchTiming | None = None,
    restart: int,
) -> None:
    if protocol is not None:
        if batch is not None:
            raise ValueError("restart validation received duplicate batch contracts")
        if batch_size is None:
            raise ValueError("restart validation requires batch_size with protocol")
        batch = protocol.batch(batch_size)
    elif batch is None:
        raise ValueError("restart validation requires a protocol or batch contract")
    record_identity = record.get("workload_identity")
    if not isinstance(record_identity, Mapping) or _json_digest(
        dict(record_identity)
    ) != workload_id:
        raise ValueError("existing latency restart workload identity is malformed")
    expected_workload_identity = (
        dict(record_identity)
        if workload_identity is None
        else dict(workload_identity)
    )
    if (
        record.get("schema") != RESTART_SCHEMA
        or record.get("protocol_id") != protocol_id
        or record.get("suite_id") != suite_id
        or record.get("workload_id") != workload_id
        or record_identity != expected_workload_identity
        or record.get("batch_size") != batch.batch_size
        or record.get("restart_index") != restart
    ):
        raise ValueError("existing latency restart identity mismatch")
    pairs = record.get("pairs")
    if not isinstance(pairs, Mapping) or set(pairs) != set(PAIR_NAMES):
        raise ValueError("existing latency restart has malformed pairs")
    call_count = batch.timed_calls_per_side_per_block
    row_count = record.get("corpus", {}).get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < batch.batch_size
    ):
        raise ValueError("existing latency restart has malformed corpus dimensions")
    preflight = record.get("untimed_prediction_check")
    if not isinstance(preflight, Mapping) or set(preflight) != {
        "checked_rows",
        "corpus_index",
        "global_shot_id",
        "full_corpus_prediction_attestation_sha256",
        "prediction_digests",
        "equality_groups",
    }:
        raise ValueError("existing latency restart has malformed untimed preflight")
    expected_preflight_index = (
        None
        if protocol is None
        else _derive_int(
            protocol.schedule_seed,
            restart,
            batch.batch_size,
            "untimed-equality-row",
        )
        % row_count
    )
    if (
        preflight.get("checked_rows") != 1
        or isinstance(preflight.get("corpus_index"), bool)
        or not isinstance(preflight.get("corpus_index"), int)
        or not 0 <= preflight["corpus_index"] < row_count
        or (
            expected_preflight_index is not None
            and preflight.get("corpus_index") != expected_preflight_index
        )
        or isinstance(preflight.get("global_shot_id"), bool)
        or not isinstance(preflight.get("global_shot_id"), int)
        or preflight["global_shot_id"] < 0
    ):
        raise ValueError("existing latency restart untimed preflight is not one-row bounded")
    prediction_digests = preflight.get("prediction_digests")
    if not isinstance(prediction_digests, Mapping) or set(prediction_digests) != set(
        VARIANT_NAMES
    ):
        raise ValueError("existing latency restart preflight digests are malformed")
    for name, digest in prediction_digests.items():
        _validate_sha256(digest, name=f"untimed prediction {name}")
    if preflight.get("equality_groups") != {
        "global_control_shadow_backend_original": [
            "global_mwpm",
            "adapter_control",
            "uf_shadow",
            "backend_original",
        ],
        "treatment_backend_residual": ["treatment", "backend_residual"],
    }:
        raise ValueError("existing latency restart preflight equality groups are malformed")
    expected_attestation = expected_workload_identity.get(
        "full_corpus_prediction_attestation_sha256"
    )
    if expected_attestation is not None:
        _validate_sha256(
            expected_attestation,
            name="workload full-corpus materialization attestation",
        )
    if preflight.get("full_corpus_prediction_attestation_sha256") != expected_attestation:
        raise ValueError("existing latency restart materialization attestation mismatch")
    pair_specs = {value.name: value for value in FIXED_PAIRS}
    call_fields = {
        "timing_call_id",
        "restart",
        "batch_size",
        "pair",
        "side",
        "block",
        "call_index",
        "schedule_id",
        "start_offset",
        "corpus_indices",
        "workload_keys",
        "workload_key_digest",
        "detector_batch_digest",
        "precomputed_summary_digest",
        "duration_ns",
    }
    for pair_name, pair in pairs.items():
        if not isinstance(pair, Mapping):
            raise ValueError("existing latency pair is malformed")
        spec = pair_specs[pair_name]
        if (
            pair.get("pair") != pair_name
            or pair.get("numerator") != spec.numerator
            or pair.get("denominator") != spec.denominator
        ):
            raise ValueError("existing latency pair identity mismatch")
        orders = pair.get("order_by_block", ())
        if len(orders) != batch.blocks_per_restart:
            raise ValueError("existing latency pair block count mismatch")
        if sorted(orders) != ["AB"] * (batch.blocks_per_restart // 2) + [
            "BA"
        ] * (batch.blocks_per_restart // 2):
            raise ValueError("existing latency pair order is not balanced")
        for field in ("numerator_calls", "denominator_calls"):
            blocks = pair.get(field)
            if (
                not isinstance(blocks, list)
                or len(blocks) != batch.blocks_per_restart
                or any(
                    not isinstance(row, list)
                    or len(row) != call_count
                    or any(
                        not isinstance(call, Mapping)
                        or not isinstance(call.get("duration_ns"), int)
                        or call["duration_ns"] <= 0
                        for call in row
                    )
                    for row in blocks
                )
            ):
                raise ValueError("existing latency pair call dimensions mismatch")
        numerator_blocks = pair["numerator_calls"]
        denominator_blocks = pair["denominator_calls"]
        for block_index, (numerator_row, denominator_row) in enumerate(
            zip(numerator_blocks, denominator_blocks)
        ):
            for call_index, (numerator, denominator) in enumerate(
                zip(numerator_row, denominator_row)
            ):
                for side, call in (
                    ("numerator", numerator),
                    ("denominator", denominator),
                ):
                    if set(call) != call_fields:
                        raise ValueError("existing latency call fields are malformed")
                    expected_identity = {
                        "restart": restart,
                        "batch_size": batch.batch_size,
                        "pair": pair_name,
                        "side": side,
                        "block": block_index,
                        "call_index": call_index,
                    }
                    if any(call.get(key) != value for key, value in expected_identity.items()):
                        raise ValueError("existing latency call identity mismatch")
                    expected_id = _sha256(canonical_json_bytes(expected_identity))
                    if call.get("timing_call_id") != expected_id:
                        raise ValueError("existing latency timing_call_id mismatch")
                    start = call.get("start_offset")
                    indices = call.get("corpus_indices")
                    if (
                        isinstance(start, bool)
                        or not isinstance(start, int)
                        or not 0 <= start < row_count
                        or indices
                        != list(
                            _slice_indices(
                                start,
                                batch_size=batch.batch_size,
                                row_count=row_count,
                            )
                        )
                    ):
                        raise ValueError("existing latency cyclic call plan mismatch")
                    keys = call.get("workload_keys")
                    if not isinstance(keys, list) or len(keys) != batch.batch_size:
                        raise ValueError("existing latency workload keys are malformed")
                    if call.get("workload_key_digest") != _json_digest(keys):
                        raise ValueError("existing latency workload-key digest mismatch")
                    for digest_field in (
                        "schedule_id",
                        "detector_batch_digest",
                        "precomputed_summary_digest",
                    ):
                        _validate_sha256(call.get(digest_field), name=digest_field)
                shared = {
                    "schedule_id",
                    "start_offset",
                    "corpus_indices",
                    "workload_keys",
                    "workload_key_digest",
                    "precomputed_summary_digest",
                }
                if any(numerator[field] != denominator[field] for field in shared):
                    raise ValueError("paired latency sides have different workload plans")
        for rows_field, totals_field in (
            ("numerator_calls", "numerator_block_totals_ns"),
            ("denominator_calls", "denominator_block_totals_ns"),
        ):
            expected_totals = [
                sum(call["duration_ns"] for call in row) for row in pair[rows_field]
            ]
            if pair.get(totals_field) != expected_totals:
                raise ValueError("existing latency block totals are inconsistent")
    canonical_json_bytes(record)


def _run_latency_suite_locked(
    factory: LatencyRestartFactory,
    *,
    protocol: LatencyProtocol,
    out_dir: str | os.PathLike[str],
    resume: bool = True,
) -> dict[str, Any]:
    """Implementation of a serialized suite while its exclusive lock is held."""

    if not isinstance(protocol, LatencyProtocol):
        raise TypeError("protocol must be LatencyProtocol")
    if not callable(factory):
        raise TypeError("factory must be callable")
    identity = _factory_identity(factory)
    protocol_json = protocol.to_json()
    protocol_id = _json_digest(protocol_json)
    workload_id = _json_digest(identity)
    suite_id = _json_digest(
        {
            "protocol_id": protocol_id,
            "workload_id": workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
        }
    )
    names = {
        "protocol.json",
        "suite.json",
        *(
            _restart_name(batch.batch_size, restart)
            for batch in protocol.batches
            for restart in range(batch.restarts)
        ),
    }
    output, actual = validate_resumable_output_root(
        out_dir, allowed_entries=names, description="Patch-UF latency output"
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing = load_json_strict(protocol_path, description="latency protocol")
        if existing != protocol_json:
            raise ValueError("existing latency protocol differs")
    else:
        install_bytes_atomic(
            protocol_path,
            canonical_json_bytes(protocol_json) + b"\n",
            prefix="patch-uf-latency-protocol-",
            overwrite=False,
        )
    if not resume and actual - {"protocol.json"}:
        raise FileExistsError("latency output already contains restart artifacts")

    ledgers: list[str] = []
    ledger_digests: dict[str, str] = {}
    spawn = multiprocessing.get_context("spawn")
    for batch in protocol.batches:
        for restart in range(batch.restarts):
            name = _restart_name(batch.batch_size, restart)
            path = output / name
            if path.exists():
                record = load_json_strict(path, description="latency restart")
                _validate_existing_restart(
                    record,
                    protocol_id=protocol_id,
                    suite_id=suite_id,
                    workload_id=workload_id,
                    workload_identity=identity,
                    protocol=protocol,
                    batch_size=batch.batch_size,
                    restart=restart,
                )
            else:
                task = LatencyRestartTask(
                    factory=factory,
                    protocol=protocol,
                    restart_index=restart,
                    batch_size=batch.batch_size,
                    protocol_id=protocol_id,
                    suite_id=suite_id,
                    workload_id=workload_id,
                    workload_identity=identity,
                )
                pickle.dumps(task)
                with ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=spawn,
                    max_tasks_per_child=1,
                ) as pool:
                    record = pool.submit(run_latency_restart_worker, task).result()
                _validate_existing_restart(
                    record,
                    protocol_id=protocol_id,
                    suite_id=suite_id,
                    workload_id=workload_id,
                    workload_identity=identity,
                    protocol=protocol,
                    batch_size=batch.batch_size,
                    restart=restart,
                )
                write_restart_ledger_atomic(path, record)
            ledgers.append(name)
            ledger_digests[name] = _sha256(path.read_bytes())

    suite = {
        "schema": SUITE_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": identity,
        "fresh_process_per_restart": True,
        "timed_restart_concurrency": 1,
        "affinity_policy": protocol.host_policy.to_json(),
        "restart_ledgers": ledgers,
        "restart_ledger_sha256": ledger_digests,
    }
    suite_path = output / "suite.json"
    if suite_path.exists():
        if load_json_strict(suite_path, description="latency suite") != suite:
            raise ValueError("existing latency suite differs")
    else:
        install_bytes_atomic(
            suite_path,
            canonical_json_bytes(suite) + b"\n",
            prefix="patch-uf-latency-suite-",
            overwrite=False,
        )
    return suite


def run_latency_suite(
    factory: LatencyRestartFactory,
    *,
    protocol: LatencyProtocol,
    out_dir: str | os.PathLike[str],
    resume: bool = True,
) -> dict[str, Any]:
    """Runs serialized fresh restarts under an exclusive overlap guard."""

    destination = Path(out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.latency.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as ex:
        raise RuntimeError("another latency suite overlaps this output") from ex
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            os.fsync(lock.fileno())
        return _run_latency_suite_locked(
            factory,
            protocol=protocol,
            out_dir=destination,
            resume=resume,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "ADAPTER_COST",
    "BACKEND_RELIEF",
    "CORPUS_SCHEMA",
    "FIXED_PAIRS",
    "NET_TOTAL",
    "PAIR_NAMES",
    "RESTART_SCHEMA",
    "RESIDUAL_APPLICATION",
    "SUITE_SCHEMA",
    "UF_GATE_COST",
    "VARIANT_NAMES",
    "AuthenticatedLatencyCorpus",
    "BatchTiming",
    "HostPolicy",
    "LatencyProtocol",
    "LatencyRestartFactory",
    "LatencyRestartTask",
    "LatencyWorkload",
    "TimedPair",
    "TimedVariant",
    "balanced_pair_orders",
    "build_timed_variants",
    "capture_host_policy",
    "load_authenticated_latency_corpus",
    "run_latency_restart",
    "run_latency_restart_worker",
    "run_latency_suite",
    "runtime_provenance",
    "tiny_smoke_protocol",
    "write_authenticated_latency_corpus",
    "write_restart_ledger_atomic",
]
