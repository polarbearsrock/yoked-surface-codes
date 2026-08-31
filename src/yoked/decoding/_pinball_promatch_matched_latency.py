"""Matched fixed-corpus latency for Global MWPM, ProMatch, and Pinball.

This module is deliberately additive.  It does not import characterization
observables and does not alter the frozen Patch-UF latency implementation.
Corpus materialization, decoder compilation, provenance construction, and
serialization all occur outside measured intervals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import dataclasses
import gc
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import time
from typing import Any, Protocol

import numpy as np

from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    install_bytes_atomic,
    load_json_artifact,
    load_json_strict,
    validate_resumable_output_root,
)
from yoked.decoding._patch_uf_latency import (
    BatchTiming,
    HostPolicy,
    LatencyProtocol,
    capture_host_policy,
    runtime_provenance,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


CORPUS_SCHEMA = "pinball-promatch-matched-latency-corpus-v1"
RESTART_SCHEMA = "pinball-promatch-matched-latency-restart-v1"
SUITE_SCHEMA = "pinball-promatch-matched-latency-suite-v1"

VARIANT_NAMES = ("global_mwpm", "promatch", "pinball")


@dataclasses.dataclass(frozen=True)
class TimedPair:
    name: str
    numerator: str
    denominator: str


FIXED_PAIRS = (
    TimedPair("promatch_vs_global", "promatch", "global_mwpm"),
    TimedPair("pinball_vs_global", "pinball", "global_mwpm"),
    TimedPair("pinball_vs_promatch", "pinball", "promatch"),
)
PAIR_NAMES = tuple(pair.name for pair in FIXED_PAIRS)

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
                raise ValueError(f"detector-only corpus contains forbidden field {path}.{key}")
            _reject_observable_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_observable_payload(child, path=f"{path}[{index}]")


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


def _json_digest(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _array_digest(value: np.ndarray) -> str:
    header = canonical_json_bytes(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    )
    return _sha256(header + b"\0" + value.tobytes(order="C"))


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()


def _safe_child(root: Path, descriptor: Mapping[str, Any], *, name: str) -> Path:
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError(f"{name} descriptor fields are malformed")
    relative = descriptor["path"]
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or Path(relative).parts != (relative,)
    ):
        raise ValueError(f"{name} path is unsafe")
    _validate_digest(descriptor["sha256"], name=f"{name} sha256")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path


@dataclasses.dataclass(frozen=True)
class AuthenticatedDetectorCorpus:
    """Read-only packed detector rows with no observable-data field."""

    detectors: np.ndarray
    num_detectors: int
    global_shot_ids: tuple[int, ...]
    corpus_digest: str
    manifest_sha256: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        num_detectors = _nonnegative_int(self.num_detectors, name="num_detectors")
        value = np.asarray(self.detectors)
        width = (num_detectors + 7) // 8
        if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != width:
            raise ValueError("packed detector corpus has malformed dtype or shape")
        if not value.flags.c_contiguous:
            raise ValueError("packed detector corpus must be C-contiguous")
        ids = tuple(
            _nonnegative_int(item, name="global_shot_id")
            for item in self.global_shot_ids
        )
        if len(ids) != len(value) or len(set(ids)) != len(ids):
            raise ValueError("global shot IDs must be unique and align with the corpus")
        if num_detectors % 8 and len(value) and width:
            unused = 0xFF ^ ((1 << (num_detectors % 8)) - 1)
            if np.any(np.bitwise_and(value[:, -1], unused)):
                raise ValueError("packed detector corpus has nonzero tail bits")
        if _array_digest(value) != _validate_digest(
            self.corpus_digest, name="corpus_digest"
        ):
            raise ValueError("detector corpus digest mismatch")
        _validate_digest(self.manifest_sha256, name="manifest_sha256")
        canonical_json_bytes(dict(self.provenance))
        value.setflags(write=False)
        object.__setattr__(self, "detectors", value)
        object.__setattr__(self, "num_detectors", num_detectors)
        object.__setattr__(self, "global_shot_ids", ids)

    @property
    def row_count(self) -> int:
        return len(self.detectors)

    @property
    def workload_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple((self.corpus_digest, shot_id) for shot_id in self.global_shot_ids)


def write_authenticated_detector_corpus(
    out_dir: str | os.PathLike[str],
    *,
    detectors: np.ndarray,
    num_detectors: int,
    global_shot_ids: Sequence[int],
    provenance: Mapping[str, Any],
) -> Path:
    """Atomically creates a detector-only corpus; existing output is rejected."""

    _reject_observable_payload(provenance)
    output = Path(out_dir).absolute()
    if output.is_symlink() or output.exists():
        raise FileExistsError(output)
    value = np.ascontiguousarray(np.asarray(detectors))
    value.setflags(write=False)
    ids = tuple(global_shot_ids)
    provisional = AuthenticatedDetectorCorpus(
        detectors=value,
        num_detectors=num_detectors,
        global_shot_ids=ids,
        corpus_digest=_array_digest(value),
        manifest_sha256="0" * 64,
        provenance=dict(provenance),
    )
    output.mkdir(parents=True)
    detector_bytes = _npy_bytes(value)
    ids_bytes = canonical_json_bytes({"global_shot_ids": list(ids)}) + b"\n"
    install_bytes_atomic(
        output / "detectors.npy",
        detector_bytes,
        prefix="matched-latency-detectors-",
        overwrite=False,
    )
    install_bytes_atomic(
        output / "global_shot_ids.json",
        ids_bytes,
        prefix="matched-latency-shot-ids-",
        overwrite=False,
    )
    unsigned = {
        "schema": CORPUS_SCHEMA,
        "num_detectors": provisional.num_detectors,
        "row_count": provisional.row_count,
        "corpus_digest": provisional.corpus_digest,
        "detectors": {"path": "detectors.npy", "sha256": _sha256(detector_bytes)},
        "global_shot_ids": {
            "path": "global_shot_ids.json",
            "sha256": _sha256(ids_bytes),
        },
        "provenance": dict(provenance),
    }
    manifest = {**unsigned, "manifest_sha256": _json_digest(unsigned)}
    path = output / "manifest.json"
    install_bytes_atomic(
        path,
        canonical_json_bytes(manifest) + b"\n",
        prefix="matched-latency-manifest-",
        overwrite=False,
    )
    load_authenticated_detector_corpus(path)
    return path


def load_authenticated_detector_corpus(
    manifest_path: str | os.PathLike[str],
) -> AuthenticatedDetectorCorpus:
    path = Path(manifest_path)
    manifest = load_json_strict(path, description="matched latency corpus manifest")
    expected = {
        "schema",
        "num_detectors",
        "row_count",
        "corpus_digest",
        "detectors",
        "global_shot_ids",
        "provenance",
        "manifest_sha256",
    }
    if set(manifest) != expected or manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError("matched latency corpus manifest fields are malformed")
    _reject_observable_payload(manifest)
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256")
    if claimed != _json_digest(unsigned):
        raise ValueError("matched latency corpus manifest digest mismatch")
    root = path.parent
    detector_path = _safe_child(root, manifest["detectors"], name="detectors")
    ids_path = _safe_child(root, manifest["global_shot_ids"], name="global_shot_ids")
    detector_bytes = detector_path.read_bytes()
    ids_bytes = ids_path.read_bytes()
    if _sha256(detector_bytes) != manifest["detectors"]["sha256"]:
        raise ValueError("detector artifact digest mismatch")
    if _sha256(ids_bytes) != manifest["global_shot_ids"]["sha256"]:
        raise ValueError("shot-ID artifact digest mismatch")
    try:
        detectors = np.load(io.BytesIO(detector_bytes), allow_pickle=False)
    except (OSError, ValueError) as ex:
        raise ValueError("cannot load detector artifact") from ex
    ids_payload = load_json_artifact(ids_path)
    if not isinstance(ids_payload, Mapping) or set(ids_payload) != {"global_shot_ids"}:
        raise ValueError("shot-ID artifact fields are malformed")
    corpus = AuthenticatedDetectorCorpus(
        detectors=detectors,
        num_detectors=manifest["num_detectors"],
        global_shot_ids=tuple(ids_payload["global_shot_ids"]),
        corpus_digest=manifest["corpus_digest"],
        manifest_sha256=claimed,
        provenance=manifest["provenance"],
    )
    if manifest["row_count"] != corpus.row_count:
        raise ValueError("matched latency corpus row count mismatch")
    return corpus


@dataclasses.dataclass(frozen=True)
class TimedVariant:
    name: str
    function: Callable[[np.ndarray], Any]

    def __post_init__(self) -> None:
        if self.name not in VARIANT_NAMES:
            raise ValueError(f"unknown latency variant {self.name!r}")
        if not callable(self.function):
            raise TypeError("timed variant function must be callable")


@dataclasses.dataclass(frozen=True)
class PackedProductionCall:
    """Direct public packed-adapter invocation used by every total endpoint."""

    compiled_decoder: Any

    def __call__(self, packed: np.ndarray) -> Any:
        return self.compiled_decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed
        )


def build_timed_variants(
    *, global_mwpm: Any, promatch: Any, pinball: Any
) -> tuple[TimedVariant, ...]:
    compiled = {
        "global_mwpm": global_mwpm,
        "promatch": promatch,
        "pinball": pinball,
    }
    for name, decoder in compiled.items():
        if not callable(getattr(decoder, "decode_shots_bit_packed", None)):
            raise TypeError(f"compiled decoder {name!r} lacks packed production decode")
    return tuple(
        TimedVariant(name, PackedProductionCall(compiled[name]))
        for name in VARIANT_NAMES
    )


@dataclasses.dataclass(frozen=True)
class LatencyWorkload:
    corpus: AuthenticatedDetectorCorpus
    variants: tuple[TimedVariant, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, AuthenticatedDetectorCorpus):
            raise TypeError("corpus must be AuthenticatedDetectorCorpus")
        variants = tuple(self.variants)
        if tuple(value.name for value in variants) != VARIANT_NAMES:
            raise ValueError("variants must use canonical matched-latency order")
        canonical_json_bytes(dict(self.provenance))
        object.__setattr__(self, "variants", variants)

    def variant_map(self) -> dict[str, TimedVariant]:
        return {variant.name: variant for variant in self.variants}


class LatencyRestartFactory(Protocol):
    suite_identity: Mapping[str, Any] | Callable[[], Mapping[str, Any]]

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload: ...


class ForkPreloadFactory(Protocol):
    """Factory extension required by the compile-once fork/COW mode."""

    suite_identity: Mapping[str, Any] | Callable[[], Mapping[str, Any]]

    def preload(self) -> LatencyWorkload: ...


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
    execution_mode: str = "spawn-factory"


@dataclasses.dataclass(frozen=True)
class PreloadedLatencyRestartTask:
    """Restart identity whose workload is inherited through a raw fork."""

    protocol: LatencyProtocol
    restart_index: int
    batch_size: int
    protocol_id: str
    suite_id: str
    workload_id: str
    workload_identity: Mapping[str, Any]
    execution_mode: str = "fork-preloaded"


def _seed_bytes(value: int) -> bytes:
    return int(value).to_bytes(32, "little", signed=False)


def _derive_int(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(_seed_bytes(seed))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def balanced_pair_orders(*, blocks: int, seed: int, pair_name: str) -> tuple[str, ...]:
    blocks = _positive_int(blocks, name="blocks")
    if blocks % 2:
        raise ValueError("blocks must be even")
    values = ["AB"] * (blocks // 2) + ["BA"] * (blocks // 2)
    rng = np.random.Generator(np.random.PCG64(_derive_int(seed, pair_name, "ABBA")))
    rng.shuffle(values)
    return tuple(values)


def _deterministic_order(names: Sequence[str], *, seed: int, purpose: str) -> tuple[str, ...]:
    values = list(names)
    rng = np.random.Generator(np.random.PCG64(_derive_int(seed, purpose)))
    rng.shuffle(values)
    return tuple(values)


@contextmanager
def _native_thread_environment():
    previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_host(policy: HostPolicy, provenance: Mapping[str, Any]) -> None:
    if tuple(provenance.get("cpu_affinity", ())) != policy.cpu_affinity:
        raise RuntimeError("latency CPU affinity differs from frozen policy")
    if tuple(provenance.get("numa_nodes", ())) != policy.expected_numa_nodes:
        raise RuntimeError("latency NUMA placement differs from frozen policy")
    for key, expected in policy.expected_host:
        if provenance.get(key) != expected:
            raise RuntimeError(f"latency host field {key!r} differs from policy")
    environment = provenance.get("native_thread_environment", {})
    if any(environment.get(name) != "1" for name in THREAD_ENVIRONMENT):
        raise RuntimeError("latency native thread environment is not pinned to one")


@contextmanager
def _host_controls(policy: HostPolicy):
    try:
        previous_affinity = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError) as ex:
        raise RuntimeError("strict CPU affinity is unavailable") from ex
    os.sched_setaffinity(0, set(policy.cpu_affinity))
    try:
        with _native_thread_environment():
            try:
                from threadpoolctl import threadpool_limits

                limiter = threadpool_limits(limits=1)
            except ImportError:
                from contextlib import nullcontext

                limiter = nullcontext()
            with limiter:
                start = runtime_provenance()
                _validate_host(policy, start)
                yield start
                _validate_host(policy, runtime_provenance())
    finally:
        os.sched_setaffinity(0, set(previous_affinity))


@dataclasses.dataclass(frozen=True)
class _ExtendedCorpus:
    original: np.ndarray
    extended: np.ndarray
    digest: str


def _extend_corpus(value: np.ndarray, *, batch_size: int) -> _ExtendedCorpus:
    if len(value) < batch_size:
        raise ValueError("detector corpus has fewer rows than the batch size")
    extended = np.concatenate((value, value[: batch_size - 1]), axis=0)
    extended.setflags(write=False)
    return _ExtendedCorpus(value, extended, _array_digest(extended))


def _slice_indices(start: int, *, batch_size: int, row_count: int) -> tuple[int, ...]:
    return tuple((start + offset) % row_count for offset in range(batch_size))


@dataclasses.dataclass(frozen=True)
class _ScheduledCall:
    block: int
    call_index: int
    start_offset: int
    indices: tuple[int, ...]
    schedule_id: str


def _schedule_calls(
    *,
    corpus: AuthenticatedDetectorCorpus,
    protocol: LatencyProtocol,
    batch: BatchTiming,
    restart_index: int,
    pair_name: str,
) -> tuple[tuple[_ScheduledCall, ...], ...]:
    rows: list[tuple[_ScheduledCall, ...]] = []
    for block in range(batch.blocks_per_restart):
        start = _derive_int(
            protocol.schedule_seed,
            restart_index,
            batch.batch_size,
            pair_name,
            block,
            "cyclic-offset",
        ) % corpus.row_count
        schedule_id = _json_digest(
            {
                "restart_index": restart_index,
                "batch_size": batch.batch_size,
                "pair": pair_name,
                "block": block,
                "start_offset": start,
            }
        )
        calls = []
        for call_index in range(batch.timed_calls_per_side_per_block):
            offset = (start + call_index * batch.batch_size) % corpus.row_count
            calls.append(
                _ScheduledCall(
                    block,
                    call_index,
                    offset,
                    _slice_indices(
                        offset,
                        batch_size=batch.batch_size,
                        row_count=corpus.row_count,
                    ),
                    schedule_id,
                )
            )
        rows.append(tuple(calls))
    return tuple(rows)


def _prediction_digest(value: object, *, name: str, rows: int) -> str:
    prediction = np.asarray(value)
    if prediction.dtype != np.uint8 or prediction.ndim != 2 or len(prediction) != rows:
        raise ValueError(f"variant {name!r} returned malformed packed predictions")
    return _array_digest(prediction)


def _untimed_preflight(workload: LatencyWorkload, *, row_index: int) -> dict[str, Any]:
    packed = workload.corpus.detectors[row_index : row_index + 1]
    digests: dict[str, str] = {}
    for name, variant in workload.variant_map().items():
        first = _prediction_digest(variant.function(packed), name=name, rows=1)
        second = _prediction_digest(variant.function(packed), name=name, rows=1)
        if first != second:
            raise ValueError(f"variant {name!r} failed deterministic repeat preflight")
        digests[name] = first
    return {
        "checked_rows": 1,
        "deterministic_repeats_per_variant": 2,
        "corpus_index": row_index,
        "global_shot_id": workload.corpus.global_shot_ids[row_index],
        "prediction_digests": digests,
    }


def _timing_call_id(
    *, restart: int, batch_size: int, pair: str, side: str, block: int, call_index: int
) -> str:
    return _json_digest(
        {
            "restart": restart,
            "batch_size": batch_size,
            "pair": pair,
            "side": side,
            "block": block,
            "call_index": call_index,
        }
    )


def _measure_side(
    *,
    variant: TimedVariant,
    side: str,
    pair_name: str,
    calls: tuple[_ScheduledCall, ...],
    extended: _ExtendedCorpus,
    corpus: AuthenticatedDetectorCorpus,
    restart_index: int,
    batch_size: int,
    clock: Callable[[], int],
) -> list[dict[str, Any]]:
    result_rows = []
    for call in calls:
        packed = extended.extended[call.start_offset : call.start_offset + batch_size]
        before = int(clock())
        result = variant.function(packed)
        after = int(clock())
        del result
        duration = after - before
        if duration <= 0:
            raise RuntimeError("latency clock produced a non-positive duration")
        workload_keys = [list(corpus.workload_keys[index]) for index in call.indices]
        result_rows.append(
            {
                "timing_call_id": _timing_call_id(
                    restart=restart_index,
                    batch_size=batch_size,
                    pair=pair_name,
                    side=side,
                    block=call.block,
                    call_index=call.call_index,
                ),
                "restart": restart_index,
                "batch_size": batch_size,
                "pair": pair_name,
                "side": side,
                "block": call.block,
                "call_index": call.call_index,
                "schedule_id": call.schedule_id,
                "start_offset": call.start_offset,
                "corpus_indices": list(call.indices),
                "workload_keys": workload_keys,
                "workload_key_digest": _json_digest(workload_keys),
                "detector_batch_digest": _array_digest(packed),
                "duration_ns": duration,
            }
        )
    return result_rows


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
    corpus = workload.corpus
    batch = task.protocol.batch(task.batch_size)
    extended = _extend_corpus(corpus.detectors, batch_size=batch.batch_size)
    before_digest = _array_digest(corpus.detectors)
    seed = _derive_int(
        task.protocol.schedule_seed, task.restart_index, task.batch_size, "restart"
    )
    preflight_index = _derive_int(seed, "untimed-preflight") % corpus.row_count
    preflight = _untimed_preflight(workload, row_index=preflight_index)
    variants = workload.variant_map()
    warmup_order = _deterministic_order(
        VARIANT_NAMES, seed=seed, purpose="warmup-order"
    )
    pair_order = _deterministic_order(PAIR_NAMES, seed=seed, purpose="pair-order")
    pair_specs = {pair.name: pair for pair in FIXED_PAIRS}
    schedules = {
        name: _schedule_calls(
            corpus=corpus,
            protocol=task.protocol,
            batch=batch,
            restart_index=task.restart_index,
            pair_name=name,
        )
        for name in PAIR_NAMES
    }
    orders = {
        name: balanced_pair_orders(
            blocks=batch.blocks_per_restart,
            seed=_derive_int(seed, name),
            pair_name=name,
        )
        for name in PAIR_NAMES
    }
    previous_gc = gc.isenabled()
    pair_records: dict[str, Any] = {}
    gc.disable()
    try:
        for name in warmup_order:
            _warmup(
                variants[name],
                extended,
                calls=batch.warmup_calls_per_variant,
                batch_size=batch.batch_size,
                start=_derive_int(seed, name, "warmup") % corpus.row_count,
                row_count=corpus.row_count,
            )
        for pair_name in pair_order:
            pair = pair_specs[pair_name]
            numerator_blocks: list[list[dict[str, Any]]] = []
            denominator_blocks: list[list[dict[str, Any]]] = []
            for block_index, order in enumerate(orders[pair_name]):
                calls = schedules[pair_name][block_index]

                def numerator() -> list[dict[str, Any]]:
                    return _measure_side(
                        variant=variants[pair.numerator],
                        side="numerator",
                        pair_name=pair_name,
                        calls=calls,
                        extended=extended,
                        corpus=corpus,
                        restart_index=task.restart_index,
                        batch_size=batch.batch_size,
                        clock=clock,
                    )

                def denominator() -> list[dict[str, Any]]:
                    return _measure_side(
                        variant=variants[pair.denominator],
                        side="denominator",
                        pair_name=pair_name,
                        calls=calls,
                        extended=extended,
                        corpus=corpus,
                        restart_index=task.restart_index,
                        batch_size=batch.batch_size,
                        clock=clock,
                    )

                if order == "AB":
                    numerator_blocks.append(numerator())
                    denominator_blocks.append(denominator())
                else:
                    denominator_blocks.append(denominator())
                    numerator_blocks.append(numerator())
            pair_records[pair_name] = {
                "pair": pair_name,
                "numerator": pair.numerator,
                "denominator": pair.denominator,
                "order_by_block": list(orders[pair_name]),
                "numerator_calls": numerator_blocks,
                "denominator_calls": denominator_blocks,
                "numerator_block_totals_ns": [
                    sum(call["duration_ns"] for call in row)
                    for row in numerator_blocks
                ],
                "denominator_block_totals_ns": [
                    sum(call["duration_ns"] for call in row)
                    for row in denominator_blocks
                ],
            }
    finally:
        if previous_gc:
            gc.enable()
    if _array_digest(corpus.detectors) != before_digest:
        raise RuntimeError("a timed decoder mutated the authenticated corpus")
    if _array_digest(extended.extended) != extended.digest:
        raise RuntimeError("a timed decoder mutated the extended corpus")
    runtime_end = runtime_provenance()
    _validate_host(task.protocol.host_policy, runtime_end)
    return {
        "schema": RESTART_SCHEMA,
        "protocol_id": task.protocol_id,
        "suite_id": task.suite_id,
        "workload_id": task.workload_id,
        "workload_identity": dict(task.workload_identity),
        "restart_index": task.restart_index,
        "batch_size": task.batch_size,
        "execution_mode": task.execution_mode,
        "process_start_method": (
            "fork" if task.execution_mode == "fork-preloaded" else "spawn"
        ),
        "clock": "time.perf_counter_ns" if clock is time.perf_counter_ns else "test-clock",
        "timing_scope": {
            "total": "public-packed-adapter-entry-to-packed-prediction-return",
            "input_generation_inside_timing": False,
            "decoder_compilation_inside_timing": False,
            "telemetry_retained_inside_timing": False,
            "gc_disabled_during_warmup_and_timing": True,
            "native_threads": 1,
            "actual_observables_available": False,
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
            "original_array_digest": before_digest,
            "extended_array_digest": extended.digest,
        },
        "untimed_prediction_check": preflight,
        "provenance": {
            "runtime_start": dict(runtime_start),
            "runtime_end": runtime_end,
            "workload": dict(workload.provenance),
            "corpus": dict(corpus.provenance),
        },
        "pairs": pair_records,
    }


def _factory_identity(factory: LatencyRestartFactory) -> dict[str, Any]:
    raw = getattr(factory, "suite_identity", None)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("latency factory requires a nonempty suite_identity")
    result = json.loads(canonical_json_bytes(dict(raw)))
    for field in ("corpus_manifest_sha256", "corpus_digest"):
        _validate_digest(result.get(field), name=f"suite_identity {field}")
    return result


def _task_ids(
    factory: LatencyRestartFactory,
    protocol: LatencyProtocol,
    *,
    execution_mode: str = "spawn-factory",
) -> tuple[dict[str, Any], str, str, str]:
    if execution_mode not in ("spawn-factory", "fork-preloaded"):
        raise ValueError("execution_mode must be spawn-factory or fork-preloaded")
    identity = _factory_identity(factory)
    protocol_id = _json_digest(protocol.to_json())
    workload_id = _json_digest(identity)
    suite_id = _json_digest(
        {
            "protocol_id": protocol_id,
            "workload_id": workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
            "execution_mode": execution_mode,
        }
    )
    return identity, protocol_id, workload_id, suite_id


def run_latency_restart(
    task: LatencyRestartTask,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    if not isinstance(task, LatencyRestartTask):
        raise TypeError("task must be LatencyRestartTask")
    batch = task.protocol.batch(task.batch_size)
    restart = _nonnegative_int(task.restart_index, name="restart_index")
    if restart >= batch.restarts:
        raise ValueError("restart_index is outside the protocol")
    if task.execution_mode != "spawn-factory":
        raise ValueError("ordinary restart task must use spawn-factory mode")
    identity, protocol_id, workload_id, suite_id = _task_ids(
        task.factory,
        task.protocol,
        execution_mode=task.execution_mode,
    )
    if (
        identity != dict(task.workload_identity)
        or protocol_id != task.protocol_id
        or workload_id != task.workload_id
        or suite_id != task.suite_id
    ):
        raise ValueError("latency restart task identity mismatch")
    with _host_controls(task.protocol.host_policy) as runtime_start:
        workload = task.factory(restart, task.batch_size)
        if not isinstance(workload, LatencyWorkload):
            raise TypeError("latency factory must return LatencyWorkload")
        if workload.corpus.manifest_sha256 != identity["corpus_manifest_sha256"]:
            raise ValueError("latency workload corpus manifest differs from suite")
        if workload.corpus.corpus_digest != identity["corpus_digest"]:
            raise ValueError("latency workload corpus digest differs from suite")
        return _measure_restart(
            workload, task=task, clock=clock, runtime_start=runtime_start
        )


def run_latency_restart_worker(task: LatencyRestartTask) -> dict[str, Any]:
    return run_latency_restart(task)


_FORK_PRELOADED_WORKLOAD: LatencyWorkload | None = None


def run_preloaded_latency_restart(
    task: PreloadedLatencyRestartTask,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Runs one restart against a workload inherited from the fork parent."""

    if not isinstance(task, PreloadedLatencyRestartTask):
        raise TypeError("task must be PreloadedLatencyRestartTask")
    if task.execution_mode != "fork-preloaded":
        raise ValueError("preloaded restart task must use fork-preloaded mode")
    batch = task.protocol.batch(task.batch_size)
    restart = _nonnegative_int(task.restart_index, name="restart_index")
    if restart >= batch.restarts:
        raise ValueError("restart_index is outside the protocol")
    expected_protocol_id = _json_digest(task.protocol.to_json())
    expected_workload_id = _json_digest(dict(task.workload_identity))
    expected_suite_id = _json_digest(
        {
            "protocol_id": expected_protocol_id,
            "workload_id": expected_workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
            "execution_mode": task.execution_mode,
        }
    )
    if (
        task.protocol_id != expected_protocol_id
        or task.workload_id != expected_workload_id
        or task.suite_id != expected_suite_id
    ):
        raise ValueError("preloaded latency restart task identity mismatch")
    workload = _FORK_PRELOADED_WORKLOAD
    if not isinstance(workload, LatencyWorkload):
        raise RuntimeError("fork child has no inherited preloaded workload")
    if (
        workload.corpus.manifest_sha256
        != task.workload_identity.get("corpus_manifest_sha256")
        or workload.corpus.corpus_digest
        != task.workload_identity.get("corpus_digest")
    ):
        raise ValueError("preloaded workload corpus differs from suite identity")
    with _host_controls(task.protocol.host_policy) as runtime_start:
        return _measure_restart(
            workload,
            task=task,
            clock=clock,
            runtime_start=runtime_start,
        )


def _fork_worker_entry(connection: Any, task: PreloadedLatencyRestartTask) -> None:
    try:
        record = run_preloaded_latency_restart(task)
        payload = {"ok": True, "record": record}
    except BaseException as ex:
        payload = {
            "ok": False,
            "error_type": type(ex).__name__,
            "error": str(ex),
        }
    try:
        connection.send_bytes(canonical_json_bytes(payload))
    finally:
        connection.close()


def _run_fork_preloaded_once(task: PreloadedLatencyRestartTask) -> dict[str, Any]:
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("fork-preloaded mode is unavailable on this platform")
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_fork_worker_entry, args=(child, task))
    process.start()
    child.close()
    try:
        try:
            raw = parent.recv_bytes()
        except EOFError as ex:
            process.join()
            raise RuntimeError(
                f"fork-preloaded restart exited without a ledger (exit={process.exitcode})"
            ) from ex
    finally:
        parent.close()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"fork-preloaded restart process failed with exit code {process.exitcode}"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as ex:
        raise RuntimeError("fork-preloaded restart returned malformed payload") from ex
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        error_type = payload.get("error_type", "Error") if isinstance(payload, Mapping) else "Error"
        error = (
            payload.get("error", "unknown fork child failure")
            if isinstance(payload, Mapping)
            else "unknown fork child failure"
        )
        raise RuntimeError(f"fork-preloaded restart failed: {error_type}: {error}")
    record = payload.get("record")
    if not isinstance(record, Mapping):
        raise RuntimeError("fork-preloaded restart omitted its ledger")
    return dict(record)


def _restart_name(batch_size: int, restart: int) -> str:
    return f"batch-{batch_size}.restart-{restart:02d}.json"


def validate_restart_record(
    record: Mapping[str, Any],
    *,
    task: LatencyRestartTask | PreloadedLatencyRestartTask,
) -> None:
    batch = task.protocol.batch(task.batch_size)
    required = {
        "schema", "protocol_id", "suite_id", "workload_id", "workload_identity",
        "restart_index", "batch_size", "execution_mode", "process_start_method",
        "clock", "timing_scope", "warmup",
        "pair_execution_order", "corpus", "untimed_prediction_check", "provenance",
        "pairs",
    }
    if set(record) != required:
        raise ValueError("latency restart fields are malformed")
    expected_identity = {
        "schema": RESTART_SCHEMA,
        "protocol_id": task.protocol_id,
        "suite_id": task.suite_id,
        "workload_id": task.workload_id,
        "workload_identity": dict(task.workload_identity),
        "restart_index": task.restart_index,
        "batch_size": task.batch_size,
        "execution_mode": task.execution_mode,
        "process_start_method": (
            "fork" if task.execution_mode == "fork-preloaded" else "spawn"
        ),
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("latency restart identity mismatch")
    if record.get("clock") != "time.perf_counter_ns":
        raise ValueError("durable latency restart must use time.perf_counter_ns")
    if record.get("timing_scope") != {
        "total": "public-packed-adapter-entry-to-packed-prediction-return",
        "input_generation_inside_timing": False,
        "decoder_compilation_inside_timing": False,
        "telemetry_retained_inside_timing": False,
        "gc_disabled_during_warmup_and_timing": True,
        "native_threads": 1,
        "actual_observables_available": False,
    }:
        raise ValueError("latency restart timing scope is malformed")
    warmup = record.get("warmup")
    if (
        not isinstance(warmup, Mapping)
        or set(warmup) != {"variant_order", "calls_per_variant"}
        or sorted(warmup.get("variant_order", ())) != sorted(VARIANT_NAMES)
        or warmup.get("calls_per_variant") != batch.warmup_calls_per_variant
    ):
        raise ValueError("latency restart warmup is malformed")
    if sorted(record.get("pair_execution_order", ())) != sorted(PAIR_NAMES):
        raise ValueError("latency restart pair execution order is malformed")
    corpus_record = record.get("corpus")
    if not isinstance(corpus_record, Mapping) or set(corpus_record) != {
        "manifest_sha256",
        "corpus_digest",
        "row_count",
        "num_detectors",
        "original_array_digest",
        "extended_array_digest",
    }:
        raise ValueError("latency restart corpus record is malformed")
    if (
        corpus_record.get("manifest_sha256")
        != task.workload_identity.get("corpus_manifest_sha256")
        or corpus_record.get("corpus_digest")
        != task.workload_identity.get("corpus_digest")
    ):
        raise ValueError("latency restart corpus identity mismatch")
    for field in (
        "manifest_sha256",
        "corpus_digest",
        "original_array_digest",
        "extended_array_digest",
    ):
        _validate_digest(corpus_record.get(field), name=f"corpus {field}")
    preflight = record.get("untimed_prediction_check")
    if not isinstance(preflight, Mapping) or set(preflight) != {
        "checked_rows",
        "deterministic_repeats_per_variant",
        "corpus_index",
        "global_shot_id",
        "prediction_digests",
    }:
        raise ValueError("latency restart prediction preflight is malformed")
    prediction_digests = preflight.get("prediction_digests")
    if (
        preflight.get("checked_rows") != 1
        or preflight.get("deterministic_repeats_per_variant") != 2
        or isinstance(preflight.get("corpus_index"), bool)
        or not isinstance(preflight.get("corpus_index"), int)
        or isinstance(preflight.get("global_shot_id"), bool)
        or not isinstance(preflight.get("global_shot_id"), int)
        or not isinstance(prediction_digests, Mapping)
        or set(prediction_digests) != set(VARIANT_NAMES)
    ):
        raise ValueError("latency restart prediction preflight is malformed")
    for name, digest in prediction_digests.items():
        _validate_digest(digest, name=f"prediction digest {name}")
    pairs = record.get("pairs")
    if not isinstance(pairs, Mapping) or set(pairs) != set(PAIR_NAMES):
        raise ValueError("latency restart pairs are malformed")
    specs = {pair.name: pair for pair in FIXED_PAIRS}
    row_count = corpus_record.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < batch.batch_size
    ):
        raise ValueError("latency restart corpus dimensions are malformed")
    expected_preflight = _derive_int(
        _derive_int(
            task.protocol.schedule_seed,
            task.restart_index,
            task.batch_size,
            "restart",
        ),
        "untimed-preflight",
    ) % row_count
    if not (
        preflight["corpus_index"] == expected_preflight
        and 0 <= preflight["global_shot_id"]
    ):
        raise ValueError("latency restart prediction preflight index is malformed")
    pair_fields = {
        "pair",
        "numerator",
        "denominator",
        "order_by_block",
        "numerator_calls",
        "denominator_calls",
        "numerator_block_totals_ns",
        "denominator_block_totals_ns",
    }
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
        "duration_ns",
    }
    for pair_name, pair in pairs.items():
        spec = specs[pair_name]
        if (
            not isinstance(pair, Mapping)
            or set(pair) != pair_fields
            or pair.get("pair") != pair_name
            or pair.get("numerator") != spec.numerator
            or pair.get("denominator") != spec.denominator
        ):
            raise ValueError("latency pair identity mismatch")
        orders = pair.get("order_by_block")
        restart_seed = _derive_int(
            task.protocol.schedule_seed,
            task.restart_index,
            task.batch_size,
            "restart",
        )
        expected_orders = list(
            balanced_pair_orders(
                blocks=batch.blocks_per_restart,
                seed=_derive_int(restart_seed, pair_name),
                pair_name=pair_name,
            )
        )
        if not isinstance(orders, list) or orders != expected_orders:
            raise ValueError("latency pair order is not balanced")
        for calls_field, totals_field, side in (
            ("numerator_calls", "numerator_block_totals_ns", "numerator"),
            ("denominator_calls", "denominator_block_totals_ns", "denominator"),
        ):
            blocks = pair.get(calls_field)
            if (
                not isinstance(blocks, list)
                or len(blocks) != batch.blocks_per_restart
                or any(
                    not isinstance(row, list)
                    or len(row) != batch.timed_calls_per_side_per_block
                    for row in blocks
                )
            ):
                raise ValueError("latency pair call dimensions mismatch")
            expected_totals = []
            for block_index, calls in enumerate(blocks):
                total = 0
                for call_index, call in enumerate(calls):
                    if (
                        not isinstance(call, Mapping)
                        or set(call) != call_fields
                        or call.get("restart") != task.restart_index
                        or call.get("batch_size") != task.batch_size
                        or call.get("pair") != pair_name
                        or call.get("side") != side
                        or call.get("block") != block_index
                        or call.get("call_index") != call_index
                        or not isinstance(call.get("duration_ns"), int)
                        or call["duration_ns"] <= 0
                    ):
                        raise ValueError("latency timing call identity is malformed")
                    indices = call.get("corpus_indices")
                    start = call.get("start_offset")
                    block_start = _derive_int(
                        task.protocol.schedule_seed,
                        task.restart_index,
                        task.batch_size,
                        pair_name,
                        block_index,
                        "cyclic-offset",
                    ) % row_count
                    expected_start = (
                        block_start + call_index * task.batch_size
                    ) % row_count
                    expected_schedule = _json_digest(
                        {
                            "restart_index": task.restart_index,
                            "batch_size": task.batch_size,
                            "pair": pair_name,
                            "block": block_index,
                            "start_offset": block_start,
                        }
                    )
                    if (
                        start != expected_start
                        or call.get("schedule_id") != expected_schedule
                        or indices
                        != list(
                            _slice_indices(
                                expected_start,
                                batch_size=task.batch_size,
                                row_count=row_count,
                            )
                        )
                    ):
                        raise ValueError("latency timing call corpus plan is malformed")
                    keys = call.get("workload_keys")
                    if (
                        not isinstance(keys, list)
                        or len(keys) != task.batch_size
                        or any(
                            not isinstance(key, list)
                            or len(key) != 2
                            or key[0] != corpus_record["corpus_digest"]
                            or isinstance(key[1], bool)
                            or not isinstance(key[1], int)
                            or key[1] < 0
                            for key in keys
                        )
                        or call.get("workload_key_digest") != _json_digest(keys)
                    ):
                        raise ValueError("latency timing call workload keys are malformed")
                    for field in ("timing_call_id", "schedule_id", "detector_batch_digest"):
                        _validate_digest(call.get(field), name=field)
                    expected_timing_id = _timing_call_id(
                        restart=task.restart_index,
                        batch_size=task.batch_size,
                        pair=pair_name,
                        side=side,
                        block=block_index,
                        call_index=call_index,
                    )
                    if call["timing_call_id"] != expected_timing_id:
                        raise ValueError("latency timing call ID is malformed")
                    total += call["duration_ns"]
                expected_totals.append(total)
            if pair.get(totals_field) != expected_totals:
                raise ValueError("latency pair block totals are inconsistent")
        for numerator, denominator in zip(pair["numerator_calls"], pair["denominator_calls"]):
            for left, right in zip(numerator, denominator):
                for field in (
                    "schedule_id", "start_offset", "corpus_indices", "workload_keys",
                    "workload_key_digest", "detector_batch_digest",
                ):
                    if left[field] != right[field]:
                        raise ValueError("paired latency calls use different workloads")
    canonical_json_bytes(dict(record))


def write_restart_ledger_atomic(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    install_bytes_atomic(
        path,
        canonical_json_bytes(dict(record)) + b"\n",
        prefix="matched-latency-restart-",
        overwrite=False,
    )


@contextmanager
def _spawn_controls(policy: HostPolicy):
    previous_affinity = tuple(sorted(os.sched_getaffinity(0)))
    os.sched_setaffinity(0, set(policy.cpu_affinity))
    try:
        with _native_thread_environment():
            _validate_host(policy, runtime_provenance())
            yield
    finally:
        os.sched_setaffinity(0, set(previous_affinity))


def _run_latency_suite_locked(
    factory: LatencyRestartFactory,
    *,
    protocol: LatencyProtocol,
    out_dir: str | os.PathLike[str],
    resume: bool,
    execution_mode: str,
) -> dict[str, Any]:
    if not isinstance(protocol, LatencyProtocol):
        raise TypeError("protocol must be LatencyProtocol")
    identity, protocol_id, workload_id, suite_id = _task_ids(
        factory,
        protocol,
        execution_mode=execution_mode,
    )
    protocol_json = protocol.to_json()
    names = {
        "protocol.json", "suite.json",
        *(
            _restart_name(batch.batch_size, restart)
            for batch in protocol.batches
            for restart in range(batch.restarts)
        ),
    }
    output, existing_names = validate_resumable_output_root(
        out_dir,
        allowed_entries=names,
        description="matched decoder latency output",
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if load_json_strict(protocol_path, description="matched latency protocol") != protocol_json:
            raise ValueError("existing matched latency protocol differs")
    else:
        install_bytes_atomic(
            protocol_path,
            canonical_json_bytes(protocol_json) + b"\n",
            prefix="matched-latency-protocol-",
            overwrite=False,
        )
    if not resume and existing_names - {"protocol.json"}:
        raise FileExistsError("matched latency output already contains artifacts")
    ledgers = []
    hashes = {}
    missing = any(
        not (output / _restart_name(batch.batch_size, restart)).exists()
        for batch in protocol.batches
        for restart in range(batch.restarts)
    )
    global _FORK_PRELOADED_WORKLOAD
    if execution_mode == "fork-preloaded" and missing:
        if _FORK_PRELOADED_WORKLOAD is not None:
            raise RuntimeError("another fork-preloaded latency suite is active")
        preload = getattr(factory, "preload", None)
        if not callable(preload):
            raise TypeError("fork-preloaded mode requires factory.preload()")
        with _host_controls(protocol.host_policy):
            workload = preload()
        if not isinstance(workload, LatencyWorkload):
            raise TypeError("factory.preload() must return LatencyWorkload")
        if (
            workload.corpus.manifest_sha256
            != identity["corpus_manifest_sha256"]
            or workload.corpus.corpus_digest != identity["corpus_digest"]
        ):
            raise ValueError("preloaded workload corpus differs from suite identity")
        _FORK_PRELOADED_WORKLOAD = workload
    try:
        for batch in protocol.batches:
            for restart in range(batch.restarts):
                if execution_mode == "fork-preloaded":
                    task: LatencyRestartTask | PreloadedLatencyRestartTask = (
                        PreloadedLatencyRestartTask(
                            protocol=protocol,
                            restart_index=restart,
                            batch_size=batch.batch_size,
                            protocol_id=protocol_id,
                            suite_id=suite_id,
                            workload_id=workload_id,
                            workload_identity=identity,
                        )
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
                name = _restart_name(batch.batch_size, restart)
                path = output / name
                if path.exists():
                    record = load_json_strict(path, description="matched latency restart")
                    validate_restart_record(record, task=task)
                else:
                    if execution_mode == "fork-preloaded":
                        with _spawn_controls(protocol.host_policy):
                            record = _run_fork_preloaded_once(task)
                    else:
                        pickle.dumps(task)
                        spawn = multiprocessing.get_context("spawn")
                        with _spawn_controls(protocol.host_policy):
                            with ProcessPoolExecutor(
                                max_workers=1,
                                mp_context=spawn,
                                max_tasks_per_child=1,
                            ) as executor:
                                record = executor.submit(
                                    run_latency_restart_worker, task
                                ).result()
                    validate_restart_record(record, task=task)
                    write_restart_ledger_atomic(path, record)
                ledgers.append(name)
                hashes[name] = _sha256(path.read_bytes())
    finally:
        _FORK_PRELOADED_WORKLOAD = None
    suite = {
        "schema": SUITE_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": identity,
        "fresh_process_per_restart": True,
        "timed_restart_concurrency": 1,
        "restart_concurrency_policy": "serialized-to-avoid-mutual-contention",
        "execution_mode": execution_mode,
        "process_start_method": (
            "fork" if execution_mode == "fork-preloaded" else "spawn"
        ),
        "parent_preload_once": execution_mode == "fork-preloaded",
        "affinity_policy": protocol.host_policy.to_json(),
        "native_threads": 1,
        "restart_ledgers": ledgers,
        "restart_ledger_sha256": hashes,
    }
    suite_path = output / "suite.json"
    if suite_path.exists():
        if load_json_strict(suite_path, description="matched latency suite") != suite:
            raise ValueError("existing matched latency suite differs")
    else:
        install_bytes_atomic(
            suite_path,
            canonical_json_bytes(suite) + b"\n",
            prefix="matched-latency-suite-",
            overwrite=False,
        )
    return suite


def run_latency_suite(
    factory: LatencyRestartFactory,
    *,
    protocol: LatencyProtocol,
    out_dir: str | os.PathLike[str],
    resume: bool = True,
    execution_mode: str = "spawn-factory",
) -> dict[str, Any]:
    """Runs serialized fresh restarts using spawn or compile-once fork/COW.

    ``fork-preloaded`` requires ``factory.preload()`` and is intended for
    expensive, effectively immutable compiled decoders.  The parent calls it
    once, then every missing restart runs in a new raw fork child.  No timer is
    read until child-side preflight, warmup, and the direct packed calls.
    """

    if execution_mode not in ("spawn-factory", "fork-preloaded"):
        raise ValueError("execution_mode must be spawn-factory or fork-preloaded")

    destination = Path(out_dir).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.latency.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as ex:
        raise RuntimeError("another matched latency suite overlaps this output") from ex
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
            execution_mode=execution_mode,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "AuthenticatedDetectorCorpus",
    "BatchTiming",
    "CORPUS_SCHEMA",
    "FIXED_PAIRS",
    "ForkPreloadFactory",
    "HostPolicy",
    "LatencyProtocol",
    "LatencyRestartTask",
    "LatencyWorkload",
    "PAIR_NAMES",
    "PackedProductionCall",
    "PreloadedLatencyRestartTask",
    "RESTART_SCHEMA",
    "SUITE_SCHEMA",
    "TimedPair",
    "TimedVariant",
    "VARIANT_NAMES",
    "balanced_pair_orders",
    "build_timed_variants",
    "capture_host_policy",
    "load_authenticated_detector_corpus",
    "run_latency_restart",
    "run_latency_restart_worker",
    "run_preloaded_latency_restart",
    "run_latency_suite",
    "validate_restart_record",
    "write_authenticated_detector_corpus",
    "write_restart_ledger_atomic",
]
