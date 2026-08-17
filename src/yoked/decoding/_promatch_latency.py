"""Controlled latency collection for the L1 ProMatch experiment.

This module implements the frozen timing design in Section 17 of
``docs/PROMATCH_IMPLEMENTATION_PLAN.md``.  In particular, it keeps corpus
generation, decoder compilation, provenance collection, and serialization
outside every measured interval.  Primary ``T_total`` calls use the production
decoder entry points supplied by the caller; this module never asks the
ProMatch adapter to retain per-shot telemetry.

The unit of restart isolation is :class:`LatencyRestartTask`.  It is a
top-level, pickle-friendly object so :func:`run_latency_benchmark` can execute
each restart in a fresh spawned process (``max_tasks_per_child=1``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Protocol

import numpy as np

from yoked.decoding._promatch_experiment import normalize_protocol
from yoked.decoding._promatch_stats import (
    canonical_json_bytes,
    manifest_experiment_id,
    validate_process_count,
)


__all__ = [
    "LATENCY_RESTART_SCHEMA",
    "LATENCY_RESTART_FIELDS",
    "LATENCY_SUITE_SCHEMA",
    "LATENCY_SUITE_FIELDS",
    "LATENCY_PAIR_FIELDS",
    "LatencyProtocol",
    "LatencyRestartFactory",
    "LatencyRestartTask",
    "LatencyWorkload",
    "balanced_pair_orders",
    "run_latency_benchmark",
    "run_latency_restart",
    "run_latency_restart_worker",
    "write_restart_ledger_atomic",
]


LATENCY_RESTART_SCHEMA = "promatch-l1-latency-restart-v1"
LATENCY_SUITE_SCHEMA = "promatch-l1-latency-suite-v1"
LATENCY_RESTART_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "suite_id",
        "workload_id",
        "workload_identity",
        "claim_bearing",
        "protocol",
        "restart_index",
        "restart_seed",
        "batch_size",
        "clock",
        "timing_scope",
        "warmup",
        "pair_execution_order",
        "corpus",
        "provenance",
        "pairs",
    }
)
LATENCY_SUITE_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "suite_id",
        "workload_id",
        "workload_identity",
        "claim_bearing",
        "protocol",
        "processes",
        "process_cap",
        "timed_restart_concurrency",
        "restart_concurrency_policy",
        "affinity_policy",
        "fresh_process_per_restart",
        "restart_ledgers",
        "restart_ledger_sha256",
    }
)
LATENCY_PAIR_FIELDS = frozenset(
    {
        "pair",
        "numerator",
        "denominator",
        "order_by_block",
        "numerator_calls_ns",
        "denominator_calls_ns",
        "numerator_block_totals_ns",
        "denominator_block_totals_ns",
        "block_total_definition",
    }
)

PRIMARY_RESTARTS = 10
PRIMARY_BLOCKS_PER_RESTART = 100
PRIMARY_CALLS_PER_BLOCK = 100
PRIMARY_WARMUP_CALLS = 1_000
PRIMARY_BATCH_SIZE = 1

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

TOTAL_PU_VS_DIRECT = "total_pu_vs_u0_direct"
TOTAL_PU_VS_WRAP = "total_pu_vs_u0_wrap"
BACKEND_RESIDUAL_VS_ORIGINAL = "backend_residual_vs_original"
PAIR_NAMES = (
    TOTAL_PU_VS_DIRECT,
    TOTAL_PU_VS_WRAP,
    BACKEND_RESIDUAL_VS_ORIGINAL,
)


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class LatencyProtocol:
    """Fixed call counts and schedule seed for a latency suite.

    Scientific collection fixes the four count fields to the primary Section
    17 values.  It may add the documented secondary batch sizes (64 and 1024),
    but batch size 1 must remain present.  Smaller counts are accepted only
    when ``scientific=False`` is passed to a runner, and are explicitly marked
    non-claim-bearing in every ledger.
    """

    restarts: int = PRIMARY_RESTARTS
    blocks_per_restart: int = PRIMARY_BLOCKS_PER_RESTART
    calls_per_block: int = PRIMARY_CALLS_PER_BLOCK
    warmup_calls_per_variant: int = PRIMARY_WARMUP_CALLS
    batch_sizes: tuple[int, ...] = (PRIMARY_BATCH_SIZE,)
    schedule_seed: int = 0x50_52_4F_4D_41_54_43_48

    def validate(self, *, scientific: bool) -> None:
        _positive_int(self.restarts, name="restarts")
        blocks = _positive_int(self.blocks_per_restart, name="blocks_per_restart")
        _positive_int(self.calls_per_block, name="calls_per_block")
        _positive_int(
            self.warmup_calls_per_variant,
            name="warmup_calls_per_variant",
        )
        if blocks % 2:
            raise ValueError("blocks_per_restart must be even for exact AB/BA balance")
        if not isinstance(self.batch_sizes, tuple) or not self.batch_sizes:
            raise TypeError("batch_sizes must be a nonempty tuple")
        batch_sizes = tuple(_positive_int(v, name="batch_size") for v in self.batch_sizes)
        if len(set(batch_sizes)) != len(batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates")
        if isinstance(self.schedule_seed, bool) or not isinstance(
            self.schedule_seed, (int, np.integer)
        ):
            raise TypeError("schedule_seed must be an integer")
        if int(self.schedule_seed) < 0 or int(self.schedule_seed) >= 2**256:
            raise ValueError("schedule_seed must lie in [0, 2**256)")
        if scientific:
            expected = {
                "restarts": PRIMARY_RESTARTS,
                "blocks_per_restart": PRIMARY_BLOCKS_PER_RESTART,
                "calls_per_block": PRIMARY_CALLS_PER_BLOCK,
                "warmup_calls_per_variant": PRIMARY_WARMUP_CALLS,
            }
            for field, value in expected.items():
                if getattr(self, field) != value:
                    raise ValueError(
                        f"scientific latency requires {field}={value}; "
                        f"got {getattr(self, field)}"
                    )
            if PRIMARY_BATCH_SIZE not in batch_sizes:
                raise ValueError("scientific latency must include primary batch size 1")
            unsupported = set(batch_sizes) - {1, 64, 1024}
            if unsupported:
                raise ValueError(
                    "scientific latency supports only batch sizes 1, 64, and 1024; "
                    f"got {sorted(unsupported)}"
                )

    def to_json(self, *, scientific: bool) -> dict[str, Any]:
        self.validate(scientific=scientific)
        return {
            "restarts": int(self.restarts),
            "blocks_per_restart": int(self.blocks_per_restart),
            "calls_per_block": int(self.calls_per_block),
            "warmup_calls_per_variant": int(self.warmup_calls_per_variant),
            "batch_sizes": [int(v) for v in self.batch_sizes],
            "schedule_seed": f"{int(self.schedule_seed):064x}",
            "scientific": bool(scientific),
        }


@dataclass
class LatencyWorkload:
    """Pregenerated corpora and uninstrumented production call paths.

    Each corpus is shot-major.  A timing call receives a read-only slice with
    shape ``(batch_size, ...)``.  ``backend_original_corpus`` and
    ``backend_residual_corpus`` must be aligned paired corpora.  Constructing
    the workload (including generation of residual syndromes) occurs before
    warmup and before any timer is read.
    """

    total_corpus: np.ndarray
    u0_direct: Callable[[np.ndarray], Any]
    u0_wrap: Callable[[np.ndarray], Any]
    pu_window: Callable[[np.ndarray], Any]
    backend_original_corpus: np.ndarray
    backend_residual_corpus: np.ndarray
    backend_original: Callable[[np.ndarray], Any]
    backend_residual: Callable[[np.ndarray], Any]
    provenance: Mapping[str, Any]


class LatencyRestartFactory(Protocol):
    """Pickle-friendly factory called once inside a fresh restart worker."""

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload: ...


@dataclass(frozen=True)
class LatencyRestartTask:
    """Complete pickle-friendly input to one fresh worker process."""

    restart_factory: LatencyRestartFactory
    protocol: LatencyProtocol
    restart_index: int
    batch_size: int
    scientific: bool
    workload_identity: Mapping[str, Any] | None = None
    workload_id: str | None = None
    suite_id: str | None = None


def _factory_suite_identity(
    restart_factory: LatencyRestartFactory, *, scientific: bool
) -> dict[str, Any]:
    raw = getattr(restart_factory, "suite_identity", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        if scientific:
            raise ValueError(
                "scientific latency requires restart_factory.suite_identity"
            )
        raw = {
            "nonclaim_factory_type": (
                f"{type(restart_factory).__module__}."
                f"{type(restart_factory).__qualname__}"
            )
        }
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("restart_factory.suite_identity must be a nonempty mapping")
    identity = dict(raw)
    canonical_json_bytes(identity)
    if scientific:
        required = {"experiment_id", "cell_id", "cell_hashes", "decoder_config_sha256"}
        missing = required - set(identity)
        if missing or any(identity.get(key) is None for key in required):
            raise ValueError(
                "scientific latency suite identity is missing "
                f"{sorted(missing or {key for key in required if identity.get(key) is None})}"
            )
    return identity


def _seed_bytes(seed: int) -> bytes:
    return int(seed).to_bytes(32, "little", signed=False)


def _derived_seed(
    seed: int,
    *,
    restart_index: int,
    batch_size: int,
    purpose: str,
) -> int:
    payload = (
        _seed_bytes(seed)
        + int(restart_index).to_bytes(8, "little", signed=False)
        + int(batch_size).to_bytes(8, "little", signed=False)
        + purpose.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "little")


def balanced_pair_orders(*, blocks: int, seed: int, pair_name: str) -> tuple[str, ...]:
    """Return a deterministic, exactly balanced randomized AB/BA schedule."""

    blocks = _positive_int(blocks, name="blocks")
    if blocks % 2:
        raise ValueError("blocks must be even for exact AB/BA balance")
    if not isinstance(pair_name, str) or not pair_name:
        raise ValueError("pair_name must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0 or int(seed) >= 2**256:
        raise ValueError("seed must lie in [0, 2**256)")

    # Hash ranking avoids dependence on NumPy RNG implementation details.
    labels = ["AB"] * (blocks // 2) + ["BA"] * (blocks // 2)
    seed_material = _seed_bytes(int(seed)) + pair_name.encode("utf-8")
    ranked = sorted(
        enumerate(labels),
        key=lambda entry: hashlib.sha256(
            seed_material + entry[0].to_bytes(8, "little")
        ).digest(),
    )
    return tuple(label for _, label in ranked)


def _deterministic_name_order(
    names: tuple[str, ...],
    *,
    seed: int,
    purpose: str,
) -> tuple[str, ...]:
    material = _seed_bytes(seed) + purpose.encode("utf-8")
    return tuple(
        sorted(
            names,
            key=lambda name: hashlib.sha256(
                material + name.encode("utf-8")
            ).digest(),
        )
    )


def _prepare_corpus(
    value: np.ndarray,
    *,
    name: str,
    batch_size: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    corpus = np.asarray(value)
    if corpus.ndim < 2:
        raise ValueError(f"{name} must be a shot-major array with at least 2 dimensions")
    if corpus.shape[0] < batch_size:
        raise ValueError(
            f"{name} has {corpus.shape[0]} shots, fewer than batch_size={batch_size}"
        )
    if corpus.dtype.hasobject:
        raise TypeError(f"{name} cannot have object dtype")
    corpus = np.ascontiguousarray(corpus)
    corpus.setflags(write=False)
    full_batches = corpus.shape[0] // batch_size
    batches = tuple(
        corpus[k * batch_size : (k + 1) * batch_size]
        for k in range(full_batches)
    )
    return corpus, batches


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()
    return {
        "sha256": digest,
        "shape": list(value.shape),
        "dtype": value.dtype.str,
    }


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


def _numa_nodes(affinity: list[int]) -> list[int]:
    nodes: set[int] = set()
    for cpu in affinity:
        cpu_path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        try:
            entries = tuple(cpu_path.glob("node*"))
        except OSError:
            continue
        for entry in entries:
            suffix = entry.name.removeprefix("node")
            if suffix.isdigit():
                nodes.add(int(suffix))
    return sorted(nodes)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_provenance() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = []
    uname = platform.uname()
    return {
        "pid": os.getpid(),
        "cpu_model": _first_cpu_field("model name"),
        "microcode": _first_cpu_field("microcode"),
        "cpu_affinity": affinity,
        "numa_nodes": _numa_nodes(affinity),
        "os": uname.system,
        "kernel": uname.release,
        "machine": uname.machine,
        "cpu_governor_cpu0": _read_optional(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpu_current_frequency_khz_cpu0": _read_optional(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        ),
        "intel_pstate_no_turbo": _read_optional(
            "/sys/devices/system/cpu/intel_pstate/no_turbo"
        ),
        "cpufreq_boost": _read_optional("/sys/devices/system/cpu/cpufreq/boost"),
        "load_average": load_average,
        "python": platform.python_version(),
        "packages": {
            "stim": _package_version("stim"),
            "sinter": _package_version("sinter"),
            "pymatching": _package_version("pymatching"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
        },
        "native_thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
    }


@contextmanager
def _one_native_thread() -> Any:
    previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = nullcontext()
    try:
        with limiter:
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _call_untimed(
    function: Callable[[np.ndarray], Any],
    batches: tuple[np.ndarray, ...],
    *,
    calls: int,
) -> None:
    for call_index in range(calls):
        result = function(batches[call_index % len(batches)])
        # Do not retain predictions across calls.  Ref-counted arrays are
        # released before the next call while GC remains fixed/disabled.
        del result


def _time_calls(
    function: Callable[[np.ndarray], Any],
    batches: tuple[np.ndarray, ...],
    *,
    calls: int,
    call_offset: int,
    clock: Callable[[], int],
) -> list[int]:
    durations: list[int] = []
    for local_index in range(calls):
        batch = batches[(call_offset + local_index) % len(batches)]
        before = int(clock())
        result = function(batch)
        after = int(clock())
        del result
        duration = after - before
        if duration <= 0:
            raise RuntimeError(
                "latency clock produced a non-positive call duration; "
                "a nanosecond monotonic clock is required"
            )
        durations.append(duration)
    return durations


def _measure_pair(
    *,
    pair_name: str,
    numerator_name: str,
    numerator: Callable[[np.ndarray], Any],
    numerator_batches: tuple[np.ndarray, ...],
    denominator_name: str,
    denominator: Callable[[np.ndarray], Any],
    denominator_batches: tuple[np.ndarray, ...],
    orders: tuple[str, ...],
    calls_per_block: int,
    clock: Callable[[], int],
) -> dict[str, Any]:
    numerator_calls: list[list[int]] = []
    denominator_calls: list[list[int]] = []
    for block_index, order in enumerate(orders):
        call_offset = block_index * calls_per_block
        if order == "AB":
            a = _time_calls(
                numerator,
                numerator_batches,
                calls=calls_per_block,
                call_offset=call_offset,
                clock=clock,
            )
            b = _time_calls(
                denominator,
                denominator_batches,
                calls=calls_per_block,
                call_offset=call_offset,
                clock=clock,
            )
        elif order == "BA":
            b = _time_calls(
                denominator,
                denominator_batches,
                calls=calls_per_block,
                call_offset=call_offset,
                clock=clock,
            )
            a = _time_calls(
                numerator,
                numerator_batches,
                calls=calls_per_block,
                call_offset=call_offset,
                clock=clock,
            )
        else:  # Defensive: orders are internally generated.
            raise AssertionError(f"invalid pair order {order!r}")
        numerator_calls.append(a)
        denominator_calls.append(b)
    return {
        "pair": pair_name,
        "numerator": numerator_name,
        "denominator": denominator_name,
        "order_by_block": list(orders),
        "numerator_calls_ns": numerator_calls,
        "denominator_calls_ns": denominator_calls,
        "numerator_block_totals_ns": [sum(row) for row in numerator_calls],
        "denominator_block_totals_ns": [sum(row) for row in denominator_calls],
        "block_total_definition": "sum_of_direct_per_call_perf_counter_ns_intervals",
    }


def _measure_latency_workload(
    workload: LatencyWorkload,
    *,
    protocol: LatencyProtocol,
    restart_index: int,
    batch_size: int,
    scientific: bool,
    clock: Callable[[], int],
    workload_identity: Mapping[str, Any] | None = None,
    workload_id: str | None = None,
    suite_id: str | None = None,
) -> dict[str, Any]:
    protocol.validate(scientific=scientific)
    restart_index = _positive_int(restart_index + 1, name="restart_index_plus_one") - 1
    if restart_index >= protocol.restarts:
        raise ValueError(
            f"restart_index={restart_index} is outside [0, {protocol.restarts})"
        )
    batch_size = _positive_int(batch_size, name="batch_size")
    if batch_size not in protocol.batch_sizes:
        raise ValueError(f"batch_size={batch_size} is not declared by the protocol")
    if not isinstance(workload, LatencyWorkload):
        raise TypeError("restart_factory must return LatencyWorkload")
    for name in (
        "u0_direct",
        "u0_wrap",
        "pu_window",
        "backend_original",
        "backend_residual",
    ):
        if not callable(getattr(workload, name)):
            raise TypeError(f"workload {name} must be callable")

    total_corpus, total_batches = _prepare_corpus(
        workload.total_corpus,
        name="total_corpus",
        batch_size=batch_size,
    )
    original_corpus, original_batches = _prepare_corpus(
        workload.backend_original_corpus,
        name="backend_original_corpus",
        batch_size=batch_size,
    )
    residual_corpus, residual_batches = _prepare_corpus(
        workload.backend_residual_corpus,
        name="backend_residual_corpus",
        batch_size=batch_size,
    )
    if original_corpus.shape[0] != residual_corpus.shape[0]:
        raise ValueError("backend original/residual corpora must contain equal shot counts")

    workload_provenance = dict(workload.provenance)
    # Provenance must be serializable before the first warmup or timer read.
    canonical_json_bytes(workload_provenance)
    if scientific:
        required_provenance = {
            "circuit_sha256",
            "dem_sha256",
            "layout_fingerprint",
            "graph_fingerprint",
            "decoder_config_sha256",
        }
        missing = required_provenance - set(workload_provenance)
        if missing:
            raise ValueError(
                "scientific latency workload provenance is missing "
                f"{sorted(missing)}"
            )
    if workload_identity is not None:
        expected_identity = dict(workload_identity)
        actual_identity = {
            "experiment_id": workload_provenance.get("experiment_id"),
            "cell_id": workload_provenance.get("cell_id"),
            "cell_hashes": {
                key: workload_provenance.get(key)
                for key in (
                    "circuit_sha256",
                    "dem_sha256",
                    "layout_fingerprint",
                    "graph_fingerprint",
                )
            },
            "decoder_config_sha256": workload_provenance.get(
                "decoder_config_sha256"
            ),
        }
        for key, actual in actual_identity.items():
            if key in expected_identity and expected_identity[key] != actual:
                raise ValueError(
                    f"latency workload identity field {key!r} differs from suite identity"
                )
        canonical_json_bytes(expected_identity)
    runtime_provenance = _runtime_provenance()
    corpus_provenance = {
        "total": _array_digest(total_corpus),
        "backend_original": _array_digest(original_corpus),
        "backend_residual": _array_digest(residual_corpus),
        "batch_size": batch_size,
        "total_complete_batches": len(total_batches),
        "backend_complete_batches": len(original_batches),
    }

    restart_seed = _derived_seed(
        int(protocol.schedule_seed),
        restart_index=restart_index,
        batch_size=batch_size,
        purpose="restart",
    )
    pair_orders = {
        pair_name: balanced_pair_orders(
            blocks=protocol.blocks_per_restart,
            seed=_derived_seed(
                restart_seed,
                restart_index=restart_index,
                batch_size=batch_size,
                purpose=pair_name,
            ),
            pair_name=pair_name,
        )
        for pair_name in PAIR_NAMES
    }
    pair_execution_order = _deterministic_name_order(
        PAIR_NAMES,
        seed=restart_seed,
        purpose="pair-execution-order",
    )
    warmup_variants = (
        "u0_direct",
        "u0_wrap",
        "pu_window",
        "backend_original",
        "backend_residual",
    )
    warmup_order = _deterministic_name_order(
        warmup_variants,
        seed=restart_seed,
        purpose="warmup-order",
    )
    total_functions = {
        "u0_direct": (workload.u0_direct, total_batches),
        "u0_wrap": (workload.u0_wrap, total_batches),
        "pu_window": (workload.pu_window, total_batches),
        "backend_original": (workload.backend_original, original_batches),
        "backend_residual": (workload.backend_residual, residual_batches),
    }

    pair_specs = {
        TOTAL_PU_VS_DIRECT: (
            "pu_window",
            workload.pu_window,
            total_batches,
            "u0_direct",
            workload.u0_direct,
            total_batches,
        ),
        TOTAL_PU_VS_WRAP: (
            "pu_window",
            workload.pu_window,
            total_batches,
            "u0_wrap",
            workload.u0_wrap,
            total_batches,
        ),
        BACKEND_RESIDUAL_VS_ORIGINAL: (
            "backend_residual",
            workload.backend_residual,
            residual_batches,
            "backend_original",
            workload.backend_original,
            original_batches,
        ),
    }

    previous_gc_enabled = gc.isenabled()
    pairs: dict[str, Any] = {}
    gc.disable()
    try:
        for name in warmup_order:
            function, batches = total_functions[name]
            _call_untimed(
                function,
                batches,
                calls=protocol.warmup_calls_per_variant,
            )
        for pair_name in pair_execution_order:
            (
                numerator_name,
                numerator,
                numerator_batches,
                denominator_name,
                denominator,
                denominator_batches,
            ) = pair_specs[pair_name]
            pairs[pair_name] = _measure_pair(
                pair_name=pair_name,
                numerator_name=numerator_name,
                numerator=numerator,
                numerator_batches=numerator_batches,
                denominator_name=denominator_name,
                denominator=denominator,
                denominator_batches=denominator_batches,
                orders=pair_orders[pair_name],
                calls_per_block=protocol.calls_per_block,
                clock=clock,
            )
    finally:
        if previous_gc_enabled:
            gc.enable()

    protocol_json = protocol.to_json(scientific=scientific)
    protocol_id = hashlib.sha256(canonical_json_bytes(protocol_json)).hexdigest()
    record = {
        "schema": LATENCY_RESTART_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": (
            None if workload_identity is None else dict(workload_identity)
        ),
        "claim_bearing": bool(scientific),
        "protocol": protocol_json,
        "restart_index": restart_index,
        "restart_seed": f"{restart_seed:064x}",
        "batch_size": batch_size,
        "clock": (
            "time.perf_counter_ns"
            if clock is time.perf_counter_ns
            else "explicit_nonclaim_test_clock"
        ),
        "timing_scope": {
            "total": "direct adapter-entry-to-packed-prediction-return",
            "backend": "direct matcher-call-only on pregenerated packed corpora",
            "input_generation_inside_timing": False,
            "telemetry_retained_inside_total": False,
            "gc_policy": "disabled_during_warmup_and_timing",
            "native_threads": 1,
        },
        "warmup": {
            "calls_per_variant": protocol.warmup_calls_per_variant,
            "variant_order": list(warmup_order),
        },
        "pair_execution_order": list(pair_execution_order),
        "corpus": corpus_provenance,
        "provenance": {
            "runtime": runtime_provenance,
            "workload": workload_provenance,
        },
        "pairs": pairs,
    }
    if set(record) != LATENCY_RESTART_FIELDS or any(
        not isinstance(pair, Mapping) or set(pair) != LATENCY_PAIR_FIELDS
        for pair in pairs.values()
    ):
        raise AssertionError("latency restart emitter drifted from its exact schema")
    # Catch accidental NumPy scalars/nonfinite data before returning to a
    # parent process or writing a ledger.
    canonical_json_bytes(record)
    return record


def run_latency_restart(
    restart_factory: LatencyRestartFactory,
    *,
    protocol: LatencyProtocol,
    restart_index: int,
    batch_size: int,
    scientific: bool,
    clock: Callable[[], int] = time.perf_counter_ns,
    workload_identity: Mapping[str, Any] | None = None,
    workload_id: str | None = None,
    suite_id: str | None = None,
) -> dict[str, Any]:
    """Build an input corpus, then collect one restart entirely in-process."""

    protocol.validate(scientific=scientific)
    if not callable(restart_factory):
        raise TypeError("restart_factory must be callable")
    if scientific and clock is not time.perf_counter_ns:
        raise ValueError("scientific latency requires time.perf_counter_ns")
    with _one_native_thread():
        # Factory construction is deliberately before warmup and all timing.
        workload = restart_factory(restart_index, batch_size)
        return _measure_latency_workload(
            workload,
            protocol=protocol,
            restart_index=restart_index,
            batch_size=batch_size,
            scientific=scientific,
            clock=clock,
            workload_identity=workload_identity,
            workload_id=workload_id,
            suite_id=suite_id,
        )


def run_latency_restart_worker(task: LatencyRestartTask) -> dict[str, Any]:
    """ProcessPool entry point; one task is one fresh process restart."""

    if not isinstance(task, LatencyRestartTask):
        raise TypeError("worker input must be LatencyRestartTask")
    return run_latency_restart(
        task.restart_factory,
        protocol=task.protocol,
        restart_index=task.restart_index,
        batch_size=task.batch_size,
        scientific=task.scientific,
        workload_identity=task.workload_identity,
        workload_id=task.workload_id,
        suite_id=task.suite_id,
    )


def _tmpdir() -> Path:
    raw = os.environ.get("TMPDIR")
    if not raw:
        raise RuntimeError("TMPDIR must be set for atomic latency-ledger staging")
    result = Path(raw)
    result.mkdir(parents=True, exist_ok=True)
    return result


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _load_json_strict(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"latency artifact must be a regular non-symlink file: {path}")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(
                source,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as ex:
        raise ValueError(f"cannot read strict JSON latency artifact {path}") from ex
    if not isinstance(value, dict):
        raise ValueError(f"latency artifact must contain one JSON object: {path}")
    canonical_json_bytes(value)
    return value


def write_restart_ledger_atomic(
    path: str | os.PathLike[str],
    record: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write canonical JSON via ``$TMPDIR`` and atomically replace ``path``."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(record)) + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix="promatch-latency-ledger-",
        suffix=".json",
        dir=_tmpdir(),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _restart_filename(*, batch_size: int, restart_index: int) -> str:
    return f"batch-{batch_size}.restart-{restart_index:02d}.json"


def _load_existing_restart(
    path: Path,
    *,
    protocol_id: str,
    suite_id: str,
    workload_id: str,
    workload_identity: Mapping[str, Any],
    protocol_json: Mapping[str, Any],
    scientific: bool,
    batch_size: int,
    restart_index: int,
) -> dict[str, Any]:
    record = _load_json_strict(path)
    if set(record) != LATENCY_RESTART_FIELDS:
        raise ValueError(f"existing latency ledger has incorrect fields: {path}")
    if record.get("schema") != LATENCY_RESTART_SCHEMA:
        raise ValueError(f"existing latency ledger has wrong schema: {path}")
    if record.get("protocol_id") != protocol_id:
        raise ValueError(f"existing latency ledger has wrong protocol_id: {path}")
    if record.get("suite_id") != suite_id or record.get("workload_id") != workload_id:
        raise ValueError(f"existing latency ledger has wrong workload/suite identity: {path}")
    if record.get("workload_identity") != dict(workload_identity):
        raise ValueError(f"existing latency ledger has changed workload identity: {path}")
    if record.get("claim_bearing") is not bool(scientific):
        raise ValueError(f"existing latency ledger has wrong claim scope: {path}")
    if record.get("batch_size") != batch_size or record.get("restart_index") != restart_index:
        raise ValueError(f"existing latency ledger has wrong task identity: {path}")
    protocol = record.get("protocol")
    if protocol != dict(protocol_json):
        raise ValueError(f"existing latency ledger has wrong protocol object: {path}")
    calls = protocol.get("calls_per_block")
    blocks = protocol.get("blocks_per_restart")
    pairs = record.get("pairs")
    if not isinstance(calls, int) or not isinstance(blocks, int) or not isinstance(pairs, Mapping):
        raise ValueError(f"existing latency ledger has malformed timing dimensions: {path}")
    if set(pairs) != set(PAIR_NAMES):
        raise ValueError(f"existing latency ledger has wrong timing pairs: {path}")
    for pair_name, pair in pairs.items():
        if not isinstance(pair, Mapping) or set(pair) != LATENCY_PAIR_FIELDS:
            raise ValueError(f"existing latency pair {pair_name!r} is malformed: {path}")
        if len(pair.get("order_by_block", [])) != blocks:
            raise ValueError(f"existing latency pair has wrong block count: {path}")
        for field in ("numerator_calls_ns", "denominator_calls_ns"):
            rows = pair.get(field)
            if (
                not isinstance(rows, list)
                or len(rows) != blocks
                or any(
                    not isinstance(row, list)
                    or len(row) != calls
                    or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in row)
                    for row in rows
                )
            ):
                raise ValueError(f"existing latency pair has malformed {field}: {path}")
    canonical_json_bytes(record)
    return record


def run_latency_benchmark(
    restart_factory: LatencyRestartFactory,
    *,
    manifest: Mapping[str, Any],
    protocol: LatencyProtocol = LatencyProtocol(),
    out_dir: str | os.PathLike[str],
    processes: int = 32,
    scientific: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Run restart-isolated timing tasks and atomically persist every ledger.

    ``processes`` is validated by the shared experiment guard and can never
    exceed 32.  ``max_tasks_per_child=1`` ensures each scheduled restart gets a
    freshly spawned process instead of a reused pool worker.
    """

    processes = validate_process_count(processes)
    normalized_manifest = normalize_protocol(manifest)
    canonical_json_bytes(normalized_manifest)
    protocol.validate(scientific=scientific)
    if not callable(restart_factory):
        raise TypeError("restart_factory must be callable")
    protocol_json = protocol.to_json(scientific=scientific)
    protocol_id = hashlib.sha256(canonical_json_bytes(protocol_json)).hexdigest()
    workload_identity = _factory_suite_identity(
        restart_factory, scientific=scientific
    )
    if scientific and (
        processes != 32
        or normalized_manifest.get("processes") != processes
        or manifest_experiment_id(normalized_manifest)
        != normalized_manifest.get("experiment_id")
        or workload_identity.get("experiment_id")
        != normalized_manifest.get("experiment_id")
    ):
        raise ValueError(
            "scientific latency manifest, process count, and workload identity differ"
        )
    workload_id = hashlib.sha256(
        canonical_json_bytes(workload_identity)
    ).hexdigest()
    suite_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_id": protocol_id,
                "workload_id": workload_id,
                "claim_bearing": bool(scientific),
                "configured_processes": processes,
                "timed_restart_concurrency": 1,
                "affinity_policy": "inherit-and-record",
            }
        )
    ).hexdigest()
    output = Path(out_dir).resolve()
    ordered_ledgers = [
        _restart_filename(batch_size=batch_size, restart_index=restart_index)
        for batch_size in protocol.batch_sizes
        for restart_index in range(protocol.restarts)
    ]
    expected_names = {"protocol.json", "suite.json", *ordered_ledgers}
    existing_suite: dict[str, Any] | None = None
    existing_suite_path = output / "suite.json"
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ValueError("latency output must be a regular directory")
        actual_names = {entry.name for entry in output.iterdir()}
        extras = sorted(actual_names - expected_names)
        if extras:
            raise ValueError(f"latency output contains unexpected artifacts: {extras}")
        protocol_path = output / "protocol.json"
        if "protocol.json" not in actual_names:
            raise ValueError("existing latency output is missing protocol.json")
        if _load_json_strict(protocol_path) != normalized_manifest:
            raise ValueError(
                "existing latency protocol.json differs from the runtime manifest"
            )
        if existing_suite_path.exists():
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                raise ValueError(
                    "completed latency suite has missing artifacts: " + str(missing)
                )
            existing_suite = _load_json_strict(existing_suite_path)
            if set(existing_suite) != LATENCY_SUITE_FIELDS:
                raise ValueError("existing latency suite has incorrect fields")
            expected_suite_identity = {
                "schema": LATENCY_SUITE_SCHEMA,
                "protocol_id": protocol_id,
                "suite_id": suite_id,
                "workload_id": workload_id,
                "workload_identity": workload_identity,
                "claim_bearing": bool(scientific),
            }
            if any(
                existing_suite.get(key) != value
                for key, value in expected_suite_identity.items()
            ):
                raise ValueError(
                    "existing latency output belongs to a different protocol or workload"
                )
        if not resume and actual_names - {"protocol.json"}:
            raise FileExistsError("latency output already contains collection artifacts")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_restart_ledger_atomic(
            output / "protocol.json",
            normalized_manifest,
        )

    records: dict[tuple[int, int], dict[str, Any]] = {}
    tasks: list[LatencyRestartTask] = []
    for batch_size in protocol.batch_sizes:
        for restart_index in range(protocol.restarts):
            key = (batch_size, restart_index)
            path = output / _restart_filename(
                batch_size=batch_size,
                restart_index=restart_index,
            )
            if path.exists():
                if not resume:
                    raise FileExistsError(path)
                records[key] = _load_existing_restart(
                    path,
                    protocol_id=protocol_id,
                    suite_id=suite_id,
                    workload_id=workload_id,
                    workload_identity=workload_identity,
                    protocol_json=protocol_json,
                    scientific=scientific,
                    batch_size=batch_size,
                    restart_index=restart_index,
                )
            else:
                tasks.append(
                    LatencyRestartTask(
                        restart_factory=restart_factory,
                        protocol=protocol,
                        restart_index=restart_index,
                        batch_size=batch_size,
                        scientific=scientific,
                        workload_identity=workload_identity,
                        workload_id=workload_id,
                        suite_id=suite_id,
                    )
                )

    if tasks:
        spawn_context = multiprocessing.get_context("spawn")
        # Spawn children while the environment is fixed to one native thread;
        # this takes effect before their NumPy/PyMatching imports.  Each worker
        # also reapplies the guard around workload construction and timing.
        with _one_native_thread():
            # Timing restarts are deliberately serialized.  Running independent
            # latency trials concurrently would turn mutual CPU/cache contention
            # into part of the treatment effect.  ``processes`` remains the
            # global experiment cap/configuration and is recorded, but measured
            # restarts use one fresh child at a time.
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=spawn_context,
                max_tasks_per_child=1,
            ) as executor:
                future_to_task = {
                    executor.submit(run_latency_restart_worker, task): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    record = future.result()
                    key = (task.batch_size, task.restart_index)
                    path = output / _restart_filename(
                        batch_size=task.batch_size,
                        restart_index=task.restart_index,
                    )
                    write_restart_ledger_atomic(path, record)
                    records[key] = record

    if len(records) != len(ordered_ledgers):
        raise RuntimeError("latency suite completed without all restart ledgers")
    suite = {
        "schema": LATENCY_SUITE_SCHEMA,
        "protocol_id": protocol_id,
        "suite_id": suite_id,
        "workload_id": workload_id,
        "workload_identity": workload_identity,
        "claim_bearing": bool(scientific),
        "protocol": protocol_json,
        "processes": processes,
        "process_cap": 32,
        "timed_restart_concurrency": 1,
        "restart_concurrency_policy": "serialized-to-avoid-mutual-contention",
        "affinity_policy": "inherit-and-record",
        "fresh_process_per_restart": True,
        "restart_ledgers": ordered_ledgers,
        "restart_ledger_sha256": {
            _restart_filename(batch_size=batch_size, restart_index=restart_index):
            hashlib.sha256(
                canonical_json_bytes(records[(batch_size, restart_index)])
            ).hexdigest()
            for batch_size in protocol.batch_sizes
            for restart_index in range(protocol.restarts)
        },
    }
    if set(suite) != LATENCY_SUITE_FIELDS:
        raise AssertionError("latency suite emitter drifted from its exact schema")
    if existing_suite is not None:
        if existing_suite != suite:
            raise ValueError("existing latency suite does not exactly reconcile")
    else:
        write_restart_ledger_atomic(output / "suite.json", suite)
    return suite
