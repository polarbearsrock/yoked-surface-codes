"""Fail-closed inference over controlled ProMatch latency ledgers.

The collector intentionally writes raw restart ledgers instead of an analysis
object.  This module is the only claim-bearing bridge from those ledgers to
the Section 17 hierarchical bootstrap.  It reconstructs the exact
``YokedPromatchLatencyFactory`` identity from the normalized manifest, rejects
incomplete or mixed suites, and validates every timing value and dimension
before calling the frozen statistical routine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from yoked.decoding._artifact_io import load_json_strict
from yoked.decoding._promatch_experiment import (
    PROTOCOL_SCHEMA,
    normalize_protocol,
    validate_experiment_protocol,
)
from yoked.decoding._promatch_latency import (
    _SCIENTIFIC_GATES,
    _TIMING_SCOPE,
    BACKEND_RESIDUAL_VS_ORIGINAL,
    LATENCY_PAIR_FIELDS,
    LATENCY_RESTART_FIELDS,
    LATENCY_RESTART_SCHEMA,
    LATENCY_SUITE_FIELDS,
    LATENCY_SUITE_SCHEMA,
    PAIR_NAMES,
    THREAD_ENVIRONMENT,
    TOTAL_PU_VS_DIRECT,
    TOTAL_PU_VS_WRAP,
    LatencyProtocol,
    balanced_pair_orders,
)
from yoked.decoding._promatch_latency_integration import (
    TinyLatencySmokeConfig,
    YokedPromatchLatencyFactory,
    latency_protocol_from_manifest,
)
from yoked.decoding._promatch_stats import (
    canonical_json_bytes,
    derive_stim_batch_seed,
    empirical_type7_quantile,
    hierarchical_timing_bootstrap,
    manifest_experiment_id,
    validate_process_count,
)


__all__ = [
    "LATENCY_ANALYSIS_SCHEMA",
    "TinyLatencyAnalysisConfig",
    "analyze_latency_suite",
    "render_latency_markdown",
]


LATENCY_ANALYSIS_SCHEMA = "promatch-l1-latency-analysis-v1"

_PAIR_VARIANTS = {
    TOTAL_PU_VS_DIRECT: ("pu_window", "u0_direct"),
    TOTAL_PU_VS_WRAP: ("pu_window", "u0_wrap"),
    BACKEND_RESIDUAL_VS_ORIGINAL: ("backend_residual", "backend_original"),
}
_WARMUP_VARIANTS = (
    "u0_direct",
    "u0_wrap",
    "pu_window",
    "backend_original",
    "backend_residual",
)
# Mirrors the fixed d=11 target geometry frozen in the first-round confirm
# protocol (docs/PROMATCH_FIRST_ROUND_PROTOCOL.json, generator_contract.
# target_cell_metadata) and independently pinned by
# _promatch_experiment.validate_experiment_protocol's target-phase check.
_TARGET_CELL = {
    "cell_id": "target-d11-n6-y2-r44-p0.001",
    "d": 11,
    "patches": 6,
    "yokes": 2,
    "r": 44,
    "p": 0.001,
    "style": "cz",
    "noise": "si1000",
    "remove_x_yoke": False,
}


@dataclass(frozen=True)
class TinyLatencyAnalysisConfig:
    """Explicit, non-claim-bearing timing and bootstrap configuration."""

    collection: TinyLatencySmokeConfig
    bootstrap_replicates: int = 100
    alpha_one_sided: float = 0.025

    def validate(self) -> None:
        if not isinstance(self.collection, TinyLatencySmokeConfig):
            raise TypeError("smoke collection must be TinyLatencySmokeConfig")
        self.collection.validate()
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates <= 0
        ):
            raise ValueError("smoke bootstrap_replicates must be a positive integer")
        if (
            isinstance(self.alpha_one_sided, bool)
            or not isinstance(self.alpha_one_sided, (int, float))
            or not math.isfinite(float(self.alpha_one_sided))
            or not 0 < float(self.alpha_one_sided) < 1
        ):
            raise ValueError("smoke alpha_one_sided must lie strictly between 0 and 1")


def _load_json(path: Path) -> dict[str, Any]:
    value = load_json_strict(path, description="latency input")
    canonical_json_bytes(value)
    return value


def _sha256_hex(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal string")
    try:
        bytes.fromhex(value)
    except ValueError as ex:
        raise ValueError(f"{name} must be hexadecimal") from ex
    return value.lower()


# Sync note: _protocol_id/_workload_id/_suite_id below deliberately re-derive
# the identities computed by _promatch_latency.run_latency_benchmark as an
# independent cross-check; drift between the two implementations fails loudly
# against the ids recorded in every suite and restart ledger.


def _protocol_id(protocol_json: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(protocol_json)).hexdigest()


def _workload_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _suite_id(
    *,
    protocol_id: str,
    workload_id: str,
    scientific: bool,
    configured_processes: int,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_id": protocol_id,
                "workload_id": workload_id,
                "claim_bearing": bool(scientific),
                "configured_processes": configured_processes,
                "timed_restart_concurrency": 1,
                "affinity_policy": "inherit-and-record",
            }
        )
    ).hexdigest()


def _schedule_seed(
    seed: int,
    *,
    restart_index: int,
    batch_size: int,
    purpose: str,
) -> int:
    # Sync note: deliberate independent re-derivation of
    # _promatch_latency._derived_seed; drift fails loudly against the
    # restart/pair seeds recorded in every ledger.
    payload = (
        int(seed).to_bytes(32, "little", signed=False)
        + restart_index.to_bytes(8, "little", signed=False)
        + batch_size.to_bytes(8, "little", signed=False)
        + purpose.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "little")


def _name_order(names: tuple[str, ...], *, seed: int, purpose: str) -> tuple[str, ...]:
    # Sync note: deliberate independent re-derivation of
    # _promatch_latency._deterministic_name_order; drift fails loudly against
    # the warmup/pair orders recorded in every ledger.
    material = int(seed).to_bytes(32, "little") + purpose.encode("utf-8")
    return tuple(
        sorted(
            names,
            key=lambda name: hashlib.sha256(material + name.encode("utf-8")).digest(),
        )
    )


def _normalized_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be an object")
    normalized = normalize_protocol(manifest)
    if normalized.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("latency analysis requires the normalized protocol schema")
    canonical_json_bytes(normalized)
    return normalized


def _manifest_cells(
    manifest: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    for field in ("cells", "performance_cells"):
        raw = manifest.get(field, [])
        if not isinstance(raw, list) or any(
            not isinstance(cell, Mapping) for cell in raw
        ):
            raise ValueError(f"normalized manifest {field} must be an array of objects")
        groups.append(raw)
    accuracy, target = groups
    ids = [cell.get("cell_id") for cell in (*accuracy, *target)]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in ids):
        raise ValueError("every latency cell must have a nonempty string cell_id")
    if len(ids) != len(set(ids)):
        raise ValueError("accuracy and target latency cell IDs must be disjoint")
    return accuracy, target


def _validate_fixed_schedule(
    rows: Any,
    *,
    shots: int,
    require_batch_ids: list[int] | None = None,
) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("scientific latency cell must have a nonempty batch schedule")
    cursor = 0
    batch_ids: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "batch_id",
            "shot_start",
            "shots",
        }:
            raise ValueError("scientific latency batch schedule has incorrect fields")
        batch_id = row["batch_id"]
        shot_start = row["shot_start"]
        count = row["shots"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (batch_id, shot_start, count)
        ):
            raise ValueError(
                "scientific latency batch schedule values must be integers"
            )
        if batch_id < 0 or shot_start != cursor or count <= 0 or count > 10_000:
            raise ValueError(
                "scientific latency batch schedule is not a fixed partition"
            )
        if index + 1 < len(rows) and count != 10_000:
            raise ValueError(
                "only the final scientific batch may contain fewer than 10000 shots"
            )
        cursor += count
        batch_ids.append(batch_id)
    if cursor != shots:
        raise ValueError("scientific latency batch schedule does not cover fixed shots")
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("scientific latency batch schedule has duplicate batch IDs")
    if batch_ids != list(range(batch_ids[0], batch_ids[0] + len(batch_ids))):
        raise ValueError("scientific latency batch IDs must be contiguous")
    if require_batch_ids is not None and batch_ids != require_batch_ids:
        raise ValueError("scientific target schedule must use exact batch IDs 0..99")


def _scientific_cell_role(manifest: Mapping[str, Any], *, cell_id: str) -> str:
    if manifest.get("phase") != "confirm":
        raise ValueError("scientific latency analysis requires phase='confirm'")
    if manifest.get("status") != "FROZEN" or manifest.get("frozen") is not True:
        raise ValueError("scientific latency analysis requires a frozen manifest")
    if manifest.get("claim_bearing") is not True:
        raise ValueError("scientific confirm/target latency must be claim-bearing")
    if manifest.get("processes") != 32:
        raise ValueError("scientific confirm/target manifest requires processes=32")
    embedded_id = manifest.get("experiment_id")
    if embedded_id != manifest_experiment_id(manifest):
        raise ValueError("scientific latency manifest has a stale experiment_id")

    accuracy, targets = _manifest_cells(manifest)
    if len(accuracy) != 1:
        raise ValueError(
            "scientific confirm manifest must contain one selected accuracy cell"
        )
    analysis = manifest.get("analysis_config")
    if not isinstance(analysis, Mapping):
        raise ValueError("scientific confirm manifest is missing analysis_config")
    selection = analysis.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("scientific confirm manifest is missing its frozen selection")
    if selection.get("selected_cell_id") != accuracy[0].get("cell_id"):
        raise ValueError("frozen selected_cell_id does not identify the accuracy cell")
    selected_cell = selection.get("selected_cell")
    if not isinstance(selected_cell, Mapping) or selected_cell.get(
        "cell_id"
    ) != accuracy[0].get("cell_id"):
        raise ValueError(
            "frozen selection.selected_cell does not identify the accuracy cell"
        )
    n_confirm = selection.get("n_confirm")
    if (
        isinstance(n_confirm, bool)
        or not isinstance(n_confirm, int)
        or n_confirm <= 0
        or n_confirm % 10_000
    ):
        raise ValueError(
            "scientific confirm manifest requires fixed n_confirm in 10000-shot units"
        )

    if len(targets) != 1 or any(
        targets[0].get(key) != expected for key, expected in _TARGET_CELL.items()
    ):
        raise ValueError(
            "scientific manifest is missing the exact fixed d=11 target cell"
        )
    required_hashes = {
        "circuit_sha256",
        "dem_sha256",
        "layout_fingerprint",
        "graph_fingerprint",
    }
    for cell in (*accuracy, *targets):
        missing = required_hashes - set(cell)
        if missing:
            raise ValueError(
                f"scientific latency cell {cell['cell_id']!r} is missing {sorted(missing)}"
            )
        for field in required_hashes:
            _sha256_hex(cell[field], name=f"{cell['cell_id']}.{field}")

    accuracy_expected = manifest.get("expected_shots_by_cell")
    accuracy_schedules = manifest.get("cell_batch_schedules")
    if (
        not isinstance(accuracy_expected, Mapping)
        or set(accuracy_expected) != {accuracy[0]["cell_id"]}
        or accuracy_expected[accuracy[0]["cell_id"]] != n_confirm
        or not isinstance(accuracy_schedules, Mapping)
        or set(accuracy_schedules) != {accuracy[0]["cell_id"]}
    ):
        raise ValueError("scientific confirm fixed-shot state does not match n_confirm")
    _validate_fixed_schedule(
        accuracy_schedules[accuracy[0]["cell_id"]],
        shots=n_confirm,
    )
    target_expected = manifest.get("performance_expected_shots_by_cell")
    target_schedules = manifest.get("performance_cell_batch_schedules")
    if (
        not isinstance(target_expected, Mapping)
        or target_expected != {targets[0]["cell_id"]: 1_000_000}
        or not isinstance(target_schedules, Mapping)
        or set(target_schedules) != {targets[0]["cell_id"]}
    ):
        raise ValueError("scientific target must declare exactly 1000000 fixed shots")
    _validate_fixed_schedule(
        target_schedules[targets[0]["cell_id"]],
        shots=1_000_000,
        require_batch_ids=list(range(100)),
    )

    if cell_id == accuracy[0]["cell_id"]:
        return "confirm_selected_accuracy_cell"
    if cell_id == targets[0]["cell_id"]:
        return "fixed_target_performance_cell"
    raise ValueError(f"cell_id {cell_id!r} is outside the confirm/target timing scope")


def _bootstrap_configuration(
    manifest: Mapping[str, Any],
    *,
    scientific: bool,
    smoke: TinyLatencyAnalysisConfig | None,
) -> tuple[int, float, bytes, str, Mapping[str, float]]:
    if scientific:
        if smoke is not None:
            raise ValueError("scientific latency analysis cannot use smoke settings")
        analysis = manifest.get("analysis_config")
        timing = (
            analysis.get("timing_protocol") if isinstance(analysis, Mapping) else None
        )
        if not isinstance(timing, Mapping):
            raise ValueError("scientific manifest is missing timing_protocol")
        replicates = timing.get("bootstrap_replicates")
        percentile = timing.get("upper_confidence_percentile")
        if replicates != 10_000:
            raise ValueError(
                "scientific timing bootstrap requires exactly 10000 replicates"
            )
        if percentile != 97.5:
            raise ValueError(
                "scientific timing upper confidence percentile must be 97.5"
            )
        # The frozen first-round confirm protocol names the thresholds
        # "claim_gates"; the earlier frozen pilot protocols used the legacy
        # key "gates".  Both stay accepted so frozen artifacts remain
        # readable.
        gates_key = "claim_gates" if "claim_gates" in timing else "gates"
        gates = timing.get(gates_key)
        if gates != _SCIENTIFIC_GATES:
            raise ValueError(
                "scientific timing claim gates differ from the frozen values"
            )
        roots = manifest.get("sampler_seed_roots")
        root = roots.get("timing_bootstrap") if isinstance(roots, Mapping) else None
        _sha256_hex(root, name="sampler_seed_roots.timing_bootstrap")
        alpha = (100.0 - float(percentile)) / 100.0
        return (
            replicates,
            alpha,
            bytes.fromhex(root),
            "manifest.timing_bootstrap",
            gates,
        )

    if smoke is None:
        raise ValueError(
            "non-claim latency analysis requires explicit TinyLatencyAnalysisConfig"
        )
    smoke.validate()
    seed_bytes = int(smoke.collection.schedule_seed).to_bytes(32, "big")
    return (
        smoke.bootstrap_replicates,
        float(smoke.alpha_one_sided),
        seed_bytes,
        "explicit_smoke_schedule_seed",
        _SCIENTIFIC_GATES,
    )


def _bootstrap_seed(
    root: bytes,
    *,
    cell_id: str,
    pair_name: str,
    batch_size: int,
) -> int:
    digest = hashlib.sha256(
        root
        + b"promatch-l1-latency-bootstrap-v1\0"
        + cell_id.encode("utf-8")
        + b"\0"
        + pair_name.encode("utf-8")
        + b"\0"
        + batch_size.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _expected_restart_names(protocol: LatencyProtocol) -> list[str]:
    # Sync note: deliberate independent re-derivation of
    # _promatch_latency._restart_filename; drift fails loudly against the
    # ledger names declared in every suite.
    return [
        f"batch-{batch_size}.restart-{restart_index:02d}.json"
        for batch_size in protocol.batch_sizes
        for restart_index in range(protocol.restarts)
    ]


def _validate_suite(
    suite: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    protocol: LatencyProtocol,
    identity: Mapping[str, Any],
    scientific: bool,
) -> tuple[str, str, str, list[str], dict[str, str]]:
    if set(suite) != LATENCY_SUITE_FIELDS:
        raise ValueError("suite.json fields do not exactly match latency suite v1")
    if suite.get("schema") != LATENCY_SUITE_SCHEMA:
        raise ValueError("suite.json has the wrong latency schema")
    protocol_json = protocol.to_json(scientific=scientific)
    expected_protocol_id = _protocol_id(protocol_json)
    expected_workload_id = _workload_id(identity)
    expected_suite_id = _suite_id(
        protocol_id=expected_protocol_id,
        workload_id=expected_workload_id,
        scientific=scientific,
        configured_processes=int(manifest["processes"]),
    )
    expected = {
        "protocol_id": expected_protocol_id,
        "workload_id": expected_workload_id,
        "suite_id": expected_suite_id,
        "workload_identity": dict(identity),
        "protocol": protocol_json,
        "claim_bearing": bool(scientific),
        "process_cap": 32,
        "timed_restart_concurrency": 1,
        "restart_concurrency_policy": "serialized-to-avoid-mutual-contention",
        "affinity_policy": "inherit-and-record",
        "fresh_process_per_restart": True,
    }
    for field, value in expected.items():
        if suite.get(field) != value:
            raise ValueError(
                f"suite.json {field} does not match manifest/factory state"
            )
    processes = validate_process_count(suite.get("processes"))
    if processes != manifest.get("processes"):
        raise ValueError("suite process count differs from the normalized manifest")
    if scientific and processes != 32:
        raise ValueError("scientific latency suite must use exactly 32 processes")
    ledgers = suite.get("restart_ledgers")
    expected_ledgers = _expected_restart_names(protocol)
    if ledgers != expected_ledgers:
        raise ValueError("suite restart_ledgers is not the exact declared restart set")
    if len(set(ledgers)) != len(ledgers):
        raise ValueError("suite restart ledger names must be unique")
    ledger_hashes = suite.get("restart_ledger_sha256")
    if not isinstance(ledger_hashes, Mapping) or set(ledger_hashes) != set(ledgers):
        raise ValueError(
            "suite restart_ledger_sha256 is not the exact declared restart set"
        )
    normalized_hashes = {
        name: _sha256_hex(ledger_hashes[name], name=f"restart ledger hash {name}")
        for name in ledgers
    }
    return (
        expected_protocol_id,
        expected_workload_id,
        expected_suite_id,
        ledgers,
        normalized_hashes,
    )


def _timing_array(
    value: Any,
    *,
    blocks: int,
    calls: int,
    name: str,
) -> tuple[np.ndarray, list[list[int]]]:
    if not isinstance(value, list) or len(value) != blocks:
        raise ValueError(f"{name} must contain exactly {blocks} blocks")
    normalized: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != calls:
            raise ValueError(f"{name} blocks must contain exactly {calls} calls")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in row):
            raise ValueError(f"{name} timings must be positive integer nanoseconds")
        normalized.append(row)
    try:
        array = np.asarray(normalized, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as ex:
        raise ValueError(f"{name} contains unrepresentable timings") from ex
    if array.shape != (blocks, calls) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite or unrepresentable timings")
    return array, normalized


def _validate_digest(
    value: Any, *, name: str, expected_shots: int
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "shape", "dtype"}:
        raise ValueError(f"{name} has an invalid corpus digest object")
    _sha256_hex(value.get("sha256"), name=f"{name}.sha256")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
        or shape[0] != expected_shots
    ):
        raise ValueError(f"{name}.shape is not the expected shot-major corpus shape")
    if value.get("dtype") != "|u1":
        raise ValueError(f"{name}.dtype must be packed uint8")
    return value


def _runtime_invariants(
    runtime: Mapping[str, Any], *, scientific: bool
) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise ValueError("restart runtime provenance must be an object")
    environment = runtime.get("native_thread_environment")
    if not isinstance(environment, Mapping) or set(environment) != set(
        THREAD_ENVIRONMENT
    ):
        raise ValueError("restart native-thread environment is incomplete")
    if any(environment[name] != "1" for name in THREAD_ENVIRONMENT):
        raise ValueError("every latency restart must use one native thread")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != {
        "stim",
        "sinter",
        "pymatching",
        "numpy",
        "scipy",
    }:
        raise ValueError("restart package provenance is missing")
    invariant_fields = (
        "cpu_model",
        "microcode",
        "cpu_affinity",
        "numa_nodes",
        "os",
        "kernel",
        "machine",
        "cpu_governor_cpu0",
        "intel_pstate_no_turbo",
        "cpufreq_boost",
        "python",
    )
    result = {field: runtime.get(field) for field in invariant_fields}
    result["packages"] = dict(packages)
    result["native_thread_environment"] = dict(environment)
    if scientific and any(
        result[field] is None for field in ("cpu_model", "os", "kernel", "python")
    ):
        raise ValueError("scientific latency runtime provenance is incomplete")
    canonical_json_bytes(result)
    return result


def _validate_workload_provenance(
    provenance: Any,
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    protocol: LatencyProtocol,
    cell_id: str,
    batch_size: int,
    restart_index: int,
    timing_corpus_root: str,
) -> Mapping[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("restart workload provenance must be an object")
    batch_index = protocol.batch_sizes.index(batch_size)
    corpus_batch_id = restart_index * len(protocol.batch_sizes) + batch_index
    expected_seed = derive_stim_batch_seed(
        seed_root=timing_corpus_root,
        batch_id=corpus_batch_id,
    )
    expected = {
        "experiment_id": manifest.get("experiment_id"),
        "cell_id": cell_id,
        "decoder_config_sha256": identity["decoder_config_sha256"],
        "decoder_config": manifest.get("decoder"),
        "dem_options": manifest.get("dem_options"),
        "restart_index": restart_index,
        "batch_size": batch_size,
        "corpus_batches": identity["corpus_batches"],
        "corpus_shots_per_restart": identity["corpus_shots_per_restart"],
        "complete_timing_batches": (
            max(batch_size, identity["corpus_shots_per_restart"]) // batch_size
            if identity["corpus_shots_per_restart"] is not None
            else identity["corpus_batches"]
        ),
        "corpus_batch_id": corpus_batch_id,
        "stim_seed": expected_seed,
        "timing_corpus_seed_root_sha256": identity["timing_corpus_seed_root_sha256"],
        "u0_backend": "pymatching-uncorrelated",
        "residual_backend": "pymatching-uncorrelated",
        "correlated_matching": False,
        "residual_generation_retained_shot_telemetry": False,
    }
    expected.update(dict(identity.get("cell_hashes", {})))
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"restart workload provenance field {field!r} does not match the factory"
            )
    return provenance


def _validate_restart(
    record: Mapping[str, Any],
    *,
    protocol: LatencyProtocol,
    protocol_id: str,
    workload_id: str,
    suite_id: str,
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cell_id: str,
    scientific: bool,
    batch_size: int,
    restart_index: int,
    timing_corpus_root: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if (
        set(record) != LATENCY_RESTART_FIELDS
        or record.get("schema") != LATENCY_RESTART_SCHEMA
    ):
        raise ValueError("restart ledger fields/schema do not exactly match latency v1")
    expected_header = {
        "protocol_id": protocol_id,
        "workload_id": workload_id,
        "suite_id": suite_id,
        "workload_identity": dict(identity),
        "claim_bearing": bool(scientific),
        "protocol": protocol.to_json(scientific=scientific),
        "restart_index": restart_index,
        "batch_size": batch_size,
        "timing_scope": _TIMING_SCOPE,
    }
    for field, value in expected_header.items():
        if record.get(field) != value:
            raise ValueError(
                f"restart ledger {field} differs from suite/manifest state"
            )
    expected_clock = "time.perf_counter_ns" if scientific else record.get("clock")
    if record.get("clock") != expected_clock or expected_clock not in {
        "time.perf_counter_ns",
        "explicit_nonclaim_test_clock",
    }:
        raise ValueError("restart ledger has an unsupported timing clock")

    restart_seed = _schedule_seed(
        protocol.schedule_seed,
        restart_index=restart_index,
        batch_size=batch_size,
        purpose="restart",
    )
    if record.get("restart_seed") != f"{restart_seed:064x}":
        raise ValueError("restart schedule seed is inconsistent with the protocol")
    expected_pair_execution = _name_order(
        PAIR_NAMES,
        seed=restart_seed,
        purpose="pair-execution-order",
    )
    if record.get("pair_execution_order") != list(expected_pair_execution):
        raise ValueError(
            "restart pair execution order is not the deterministic schedule"
        )
    expected_warmup_order = _name_order(
        _WARMUP_VARIANTS,
        seed=restart_seed,
        purpose="warmup-order",
    )
    warmup = record.get("warmup")
    if warmup != {
        "calls_per_variant": protocol.warmup_calls_per_variant,
        "variant_order": list(expected_warmup_order),
    }:
        raise ValueError("restart warmup schedule differs from the protocol")

    corpus = record.get("corpus")
    if not isinstance(corpus, Mapping) or set(corpus) != {
        "total",
        "backend_original",
        "backend_residual",
        "batch_size",
        "total_complete_batches",
        "backend_complete_batches",
    }:
        raise ValueError("restart corpus provenance has incorrect fields")
    corpus_batches = identity.get("corpus_batches")
    corpus_shots_per_restart = identity.get("corpus_shots_per_restart")
    expected_shots = (
        batch_size * corpus_batches
        if corpus_shots_per_restart is None
        else max(batch_size, corpus_shots_per_restart)
    )
    complete_timing_batches = expected_shots // batch_size
    if (
        corpus.get("batch_size") != batch_size
        or corpus.get("total_complete_batches") != complete_timing_batches
        or corpus.get("backend_complete_batches") != complete_timing_batches
    ):
        raise ValueError("restart corpus batch counts differ from the factory")
    total_digest = _validate_digest(
        corpus.get("total"), name="corpus.total", expected_shots=expected_shots
    )
    original_digest = _validate_digest(
        corpus.get("backend_original"),
        name="corpus.backend_original",
        expected_shots=expected_shots,
    )
    _validate_digest(
        corpus.get("backend_residual"),
        name="corpus.backend_residual",
        expected_shots=expected_shots,
    )
    if total_digest != original_digest:
        raise ValueError(
            "total and original-backend corpora must be the identical input"
        )

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "runtime",
        "workload",
    }:
        raise ValueError("restart provenance has incorrect fields")
    runtime_invariants = _runtime_invariants(
        provenance["runtime"], scientific=scientific
    )
    _validate_workload_provenance(
        provenance["workload"],
        manifest=manifest,
        identity=identity,
        protocol=protocol,
        cell_id=cell_id,
        batch_size=batch_size,
        restart_index=restart_index,
        timing_corpus_root=timing_corpus_root,
    )

    pairs = record.get("pairs")
    if not isinstance(pairs, Mapping) or set(pairs) != set(PAIR_NAMES):
        raise ValueError("restart ledger has missing or extra timing pairs")
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pair_name in PAIR_NAMES:
        pair = pairs[pair_name]
        if not isinstance(pair, Mapping) or set(pair) != LATENCY_PAIR_FIELDS:
            raise ValueError(f"latency pair {pair_name!r} has incorrect fields")
        numerator_name, denominator_name = _PAIR_VARIANTS[pair_name]
        if (
            pair.get("pair") != pair_name
            or pair.get("numerator") != numerator_name
            or pair.get("denominator") != denominator_name
            or pair.get("block_total_definition")
            != "sum_of_direct_per_call_perf_counter_ns_intervals"
        ):
            raise ValueError(f"latency pair {pair_name!r} has incorrect semantics")
        pair_seed = _schedule_seed(
            restart_seed,
            restart_index=restart_index,
            batch_size=batch_size,
            purpose=pair_name,
        )
        expected_orders = balanced_pair_orders(
            blocks=protocol.blocks_per_restart,
            seed=pair_seed,
            pair_name=pair_name,
        )
        if pair.get("order_by_block") != list(expected_orders):
            raise ValueError(f"latency pair {pair_name!r} has a corrupt AB/BA schedule")
        numerator, numerator_rows = _timing_array(
            pair.get("numerator_calls_ns"),
            blocks=protocol.blocks_per_restart,
            calls=protocol.calls_per_block,
            name=f"{pair_name}.numerator_calls_ns",
        )
        denominator, denominator_rows = _timing_array(
            pair.get("denominator_calls_ns"),
            blocks=protocol.blocks_per_restart,
            calls=protocol.calls_per_block,
            name=f"{pair_name}.denominator_calls_ns",
        )
        for field, rows in (
            ("numerator_block_totals_ns", numerator_rows),
            ("denominator_block_totals_ns", denominator_rows),
        ):
            totals = pair.get(field)
            expected_totals = [sum(row) for row in rows]
            if totals != expected_totals:
                raise ValueError(f"{pair_name}.{field} does not equal its raw calls")
        arrays[pair_name] = numerator, denominator
    return arrays, runtime_invariants


def _quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return {
        "median_ns": empirical_type7_quantile(flat, 0.50),
        "p90_ns": empirical_type7_quantile(flat, 0.90),
        "p99_ns": empirical_type7_quantile(flat, 0.99),
    }


def analyze_latency_suite(
    suite_path: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any],
    cell_id: str,
    scientific: bool,
    smoke: TinyLatencyAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Validate a complete suite and run the frozen hierarchical bootstrap."""

    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a nonempty string")
    normalized = _normalized_manifest(manifest)
    if scientific:
        # This intentionally rechecks current repository/source hashes,
        # software versions, execution environment, fixed selection
        # derivation, and exact confirm/target schedules before any confidence
        # interval can be interpreted as a claim.
        validate_experiment_protocol(
            normalized,
            phase="confirm",
            scientific=True,
            processes=32,
        )
        cell_role = _scientific_cell_role(normalized, cell_id=cell_id)
        smoke_collection = None
    else:
        if smoke is None:
            raise ValueError(
                "non-claim latency analysis requires explicit TinyLatencyAnalysisConfig"
            )
        smoke.validate()
        cell_role = "explicit_nonclaim_smoke"
        smoke_collection = smoke.collection

    protocol = latency_protocol_from_manifest(
        normalized,
        scientific=scientific,
        smoke=smoke_collection,
    )
    factory = YokedPromatchLatencyFactory.from_manifest(
        normalized,
        cell_id=cell_id,
        scientific=scientific,
        smoke=smoke_collection,
    )
    identity = dict(factory.suite_identity)
    replicates, alpha, bootstrap_root, seed_source, gates = _bootstrap_configuration(
        normalized,
        scientific=scientific,
        smoke=smoke,
    )

    path = Path(suite_path)
    suite_file = path if path.name == "suite.json" else path / "suite.json"
    root = suite_file.parent
    expected_names = {
        "protocol.json",
        "suite.json",
        *_expected_restart_names(protocol),
    }
    if not root.is_dir() or root.is_symlink():
        raise ValueError("latency input must be a regular directory")
    actual_names = {entry.name for entry in root.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"latency artifact set mismatch: missing={missing}, extra={extra}"
        )
    stored_manifest = _load_json(root / "protocol.json")
    if stored_manifest != normalized:
        raise ValueError(
            "latency protocol.json differs from the supplied normalized manifest"
        )
    suite = _load_json(suite_file)
    protocol_id, workload_id, suite_id, ledger_names, ledger_hashes = _validate_suite(
        suite,
        manifest=normalized,
        protocol=protocol,
        identity=identity,
        scientific=scientific,
    )
    declared = set(ledger_names)
    actual = {candidate.name for candidate in root.glob("batch-*.restart-*.json")}
    if actual != declared:
        missing = sorted(declared - actual)
        extra = sorted(actual - declared)
        raise ValueError(
            f"latency restart set mismatch: missing={missing}, extra={extra}"
        )

    roots = normalized.get("sampler_seed_roots")
    if scientific:
        timing_corpus_root = (
            roots.get("timing_corpus") if isinstance(roots, Mapping) else None
        )
    else:
        timing_corpus_root = smoke.collection.timing_corpus_seed_root
    _sha256_hex(timing_corpus_root, name="timing_corpus_seed_root")

    collected: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        batch_size: {pair: [] for pair in PAIR_NAMES}
        for batch_size in protocol.batch_sizes
    }
    common_runtime: dict[str, Any] | None = None
    for batch_size in protocol.batch_sizes:
        for restart_index in range(protocol.restarts):
            name = f"batch-{batch_size}.restart-{restart_index:02d}.json"
            record = _load_json(root / name)
            actual_hash = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
            if actual_hash != ledger_hashes[name]:
                raise ValueError(f"latency restart ledger hash mismatch: {name}")
            arrays, runtime = _validate_restart(
                record,
                protocol=protocol,
                protocol_id=protocol_id,
                workload_id=workload_id,
                suite_id=suite_id,
                identity=identity,
                manifest=normalized,
                cell_id=cell_id,
                scientific=scientific,
                batch_size=batch_size,
                restart_index=restart_index,
                timing_corpus_root=timing_corpus_root,
            )
            if common_runtime is None:
                common_runtime = runtime
            elif runtime != common_runtime:
                raise ValueError("latency restart ledgers mix runtime environments")
            for pair_name in PAIR_NAMES:
                collected[batch_size][pair_name].append(arrays[pair_name])

    batch_results: dict[str, Any] = {}
    for batch_size in protocol.batch_sizes:
        pair_results: dict[str, Any] = {}
        for pair_name in PAIR_NAMES:
            rows = collected[batch_size][pair_name]
            numerator = np.stack([row[0] for row in rows])
            denominator = np.stack([row[1] for row in rows])
            expected_shape = (
                protocol.restarts,
                protocol.blocks_per_restart,
                protocol.calls_per_block,
            )
            if numerator.shape != expected_shape or denominator.shape != expected_shape:
                raise ValueError("aggregated timing arrays have incorrect dimensions")
            seed = _bootstrap_seed(
                bootstrap_root,
                cell_id=cell_id,
                pair_name=pair_name,
                batch_size=batch_size,
            )
            inference = hierarchical_timing_bootstrap(
                numerator_calls=numerator,
                denominator_calls=denominator,
                replicates=replicates,
                seed=seed,
                alpha=alpha,
            )
            pair_results[pair_name] = {
                "numerator": _PAIR_VARIANTS[pair_name][0],
                "denominator": _PAIR_VARIANTS[pair_name][1],
                "array_shape": list(expected_shape),
                "timing_unit": "batch",
                "batch_size": batch_size,
                "bootstrap_seed": seed,
                "bootstrap_replicates": replicates,
                "alpha_one_sided": alpha,
                "geometric_ratio": inference.geometric_ratio,
                "geometric_ratio_upper_one_sided": inference.geometric_ratio_upper,
                "p99_ratio": inference.p99_ratio,
                "p99_ratio_upper_one_sided": inference.p99_ratio_upper,
                "numerator_raw_timing": _quantiles(numerator),
                "denominator_raw_timing": _quantiles(denominator),
            }
        batch_results[str(batch_size)] = {
            "batch_size": batch_size,
            "calls_per_variant": protocol.restarts
            * protocol.blocks_per_restart
            * protocol.calls_per_block,
            "pairs": pair_results,
        }

    primary_pairs = batch_results["1"]["pairs"]
    backend = primary_pairs[BACKEND_RESIDUAL_VS_ORIGINAL]
    direct = primary_pairs[TOTAL_PU_VS_DIRECT]
    wrap = primary_pairs[TOTAL_PU_VS_WRAP]
    backend_passed = (
        backend["geometric_ratio_upper_one_sided"]
        < gates["backend_geometric_ratio_upper"]
    )
    total_geometric_passed = (
        direct["geometric_ratio_upper_one_sided"] < gates["total_geometric_ratio_upper"]
    )
    total_p99_passed = (
        direct["p99_ratio_upper_one_sided"] < gates["total_p99_ratio_upper"]
    )
    total_passed = total_geometric_passed and total_p99_passed
    result = {
        "schema": LATENCY_ANALYSIS_SCHEMA,
        "suite_id": suite_id,
        "protocol_id": protocol_id,
        "workload_id": workload_id,
        "experiment_id": normalized.get("experiment_id"),
        "cell_id": cell_id,
        "cell_role": cell_role,
        "claim_bearing": bool(scientific),
        "processes": suite["processes"],
        "timed_restart_concurrency": suite["timed_restart_concurrency"],
        "restart_concurrency_policy": suite["restart_concurrency_policy"],
        "affinity_policy": suite["affinity_policy"],
        "bootstrap": {
            "method": "hierarchical_restart_then_paired_block_percentile",
            "seed_derivation": (
                "sha256(root || promatch-l1-latency-bootstrap-v1 || "
                "cell_id || pair || uint64le(batch_size)); first uint64le"
            ),
            "seed_source": seed_source,
            "replicates": replicates,
            "alpha_one_sided": alpha,
            "quantile_method": "empirical_type_7",
        },
        "batch_results": batch_results,
        "batch_1_gates": {
            "thresholds": dict(gates),
            "residual_backend_relief_passed": bool(backend_passed),
            "total_geometric_gate_passed": bool(total_geometric_passed),
            "total_p99_gate_passed": bool(total_p99_passed),
            "end_to_end_software_latency_improvement_passed": bool(total_passed),
            "u0_wrap_diagnostic": {
                "claim_gate": None,
                "geometric_ratio": wrap["geometric_ratio"],
                "geometric_ratio_upper_one_sided": wrap[
                    "geometric_ratio_upper_one_sided"
                ],
                "p99_ratio": wrap["p99_ratio"],
                "p99_ratio_upper_one_sided": wrap["p99_ratio_upper_one_sided"],
            },
            "claim_authorized": bool(scientific),
        },
        "claim_scope": {
            "scope": "in_process_batch_1_software_latency",
            "total_interval": "adapter_entry_through_packed_prediction_return",
            "backend_interval": "matcher_call_only_on_pregenerated_corpora",
            "hardware_latency_claim_authorized": False,
            "real_time_deadline_claim_authorized": False,
            "secondary_batches_are_throughput_diagnostics": True,
            "nonclaim_smoke": not scientific,
        },
    }
    result["analysis_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def render_latency_markdown(analysis: Mapping[str, Any]) -> str:
    """Render a compact report while preserving the software-only scope."""

    if (
        not isinstance(analysis, Mapping)
        or analysis.get("schema") != LATENCY_ANALYSIS_SCHEMA
    ):
        raise ValueError("analysis is not a ProMatch latency analysis v1 object")
    scope = analysis.get("claim_scope")
    batches = analysis.get("batch_results")
    gates = analysis.get("batch_1_gates")
    if (
        not isinstance(scope, Mapping)
        or not isinstance(batches, Mapping)
        or not isinstance(gates, Mapping)
    ):
        raise ValueError("latency analysis is missing report sections")
    if scope.get("hardware_latency_claim_authorized") is not False:
        raise ValueError("latency report must remain explicitly non-hardware")

    claim_label = (
        "claim-bearing frozen analysis"
        if analysis.get("claim_bearing") is True
        else "non-claim smoke diagnostic"
    )
    lines = [
        "# ProMatch L1 latency analysis",
        "",
        f"Cell: `{analysis.get('cell_id')}` ({claim_label}).",
        "",
        (
            "Scope: in-process software latency only. These measurements are "
            "not hardware latency and do not establish a real-time deadline."
        ),
        (
            "Timing restarts were serialized with concurrency "
            f"`{analysis.get('timed_restart_concurrency')}`; affinity policy was "
            f"`{analysis.get('affinity_policy')}`."
        ),
        "",
        "| Batch | Pair | Geometric ratio | Upper bound | p99 ratio | p99 upper |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for batch_key in sorted(batches, key=lambda item: int(item)):
        batch = batches[batch_key]
        for pair_name in PAIR_NAMES:
            pair = batch["pairs"][pair_name]
            lines.append(
                "| {batch} | {pair} | {g:.6g} | {gu:.6g} | {p:.6g} | {pu:.6g} |".format(
                    batch=batch["batch_size"],
                    pair=pair_name,
                    g=pair["geometric_ratio"],
                    gu=pair["geometric_ratio_upper_one_sided"],
                    p=pair["p99_ratio"],
                    pu=pair["p99_ratio_upper_one_sided"],
                )
            )
    lines.extend(
        [
            "",
            "## Raw timing quantiles",
            "",
            "| Batch | Pair | Side | Median ns | p90 ns | p99 ns |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for batch_key in sorted(batches, key=lambda item: int(item)):
        batch = batches[batch_key]
        for pair_name in PAIR_NAMES:
            pair = batch["pairs"][pair_name]
            for side in ("numerator", "denominator"):
                values = pair[f"{side}_raw_timing"]
                lines.append(
                    "| {batch} | {pair} | {side} | {median:.6g} | {p90:.6g} | {p99:.6g} |".format(
                        batch=batch["batch_size"],
                        pair=pair_name,
                        side=pair[side],
                        median=values["median_ns"],
                        p90=values["p90_ns"],
                        p99=values["p99_ns"],
                    )
                )
    lines.extend(
        [
            "",
            "## Batch-1 gates",
            "",
            f"- Residual-backend relief: `{gates['residual_backend_relief_passed']}`",
            (
                "- End-to-end software-latency improvement: "
                f"`{gates['end_to_end_software_latency_improvement_passed']}`"
            ),
            f"- Claim authorized: `{gates['claim_authorized']}`",
            "- U0-wrap is diagnostic only and has no independent claim gate.",
            "",
        ]
    )
    return "\n".join(lines)
