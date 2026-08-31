"""Fail-closed analysis for matched Global/ProMatch/Pinball latency suites."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from yoked.decoding._artifact_io import load_json_strict
from yoked.decoding._patch_uf_latency import BatchTiming, HostPolicy, LatencyProtocol
from yoked.decoding import _pinball_promatch_matched_latency as latency
from yoked.decoding._promatch_stats import (
    canonical_json_bytes,
    empirical_type7_quantile,
)


ANALYSIS_SCHEMA = "pinball-promatch-matched-latency-analysis-v1"

_ARM_LABELS = {
    "global_mwpm": "Global MWPM",
    "promatch": "ProMatch",
    "pinball": "Pinball",
}
_SUITE_FIELDS = {
    "schema",
    "protocol_id",
    "suite_id",
    "workload_id",
    "workload_identity",
    "fresh_process_per_restart",
    "timed_restart_concurrency",
    "restart_concurrency_policy",
    "execution_mode",
    "process_start_method",
    "parent_preload_once",
    "affinity_policy",
    "native_threads",
    "restart_ledgers",
    "restart_ledger_sha256",
}


@dataclasses.dataclass(frozen=True)
class MatchedLatencyAnalysisArtifacts:
    analysis: Mapping[str, Any]
    report_markdown: str
    analysis_bytes: bytes
    report_bytes: bytes


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


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _protocol_from_json(raw: Mapping[str, Any]) -> LatencyProtocol:
    expected = {"batches", "schedule_seed", "host_policy", "variant_names", "pairs"}
    if set(raw) != expected:
        raise ValueError("matched latency protocol fields are malformed")
    host = raw.get("host_policy")
    if not isinstance(host, Mapping) or set(host) != {
        "cpu_affinity",
        "expected_host",
        "expected_numa_nodes",
    }:
        raise ValueError("matched latency protocol host policy is malformed")
    expected_host = host["expected_host"]
    if not isinstance(expected_host, Mapping):
        raise ValueError("matched latency expected host is malformed")
    batches = raw.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("matched latency batches are malformed")
    seed = raw.get("schedule_seed")
    _validate_digest(seed, name="schedule_seed")
    try:
        protocol = LatencyProtocol(
            batches=tuple(BatchTiming(**dict(item)) for item in batches),
            schedule_seed=int(seed, 16),
            host_policy=HostPolicy(
                cpu_affinity=tuple(host["cpu_affinity"]),
                expected_host=tuple(sorted(expected_host.items())),
                expected_numa_nodes=tuple(host["expected_numa_nodes"]),
            ),
        )
    except (TypeError, ValueError) as ex:
        raise ValueError("matched latency protocol literals are invalid") from ex
    if protocol.to_json() != dict(raw):
        raise ValueError("matched latency protocol does not round-trip canonically")
    if 1 not in tuple(batch.batch_size for batch in protocol.batches):
        raise ValueError("matched latency analysis requires primary batch size 1")
    return protocol


def _factory_identity(factory: object) -> dict[str, Any]:
    raw = getattr(factory, "suite_identity", None)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("supplied latency factory has no suite identity")
    try:
        result = json.loads(canonical_json_bytes(dict(raw)))
    except (TypeError, ValueError) as ex:
        raise ValueError("supplied latency factory identity is not canonical") from ex
    return result


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    flat = [int(value) for value in np.asarray(values).ravel()]
    if not flat or any(value <= 0 for value in flat):
        raise ValueError("latency durations must be finite and strictly positive")
    return {
        "calls": len(flat),
        "p50_ns": empirical_type7_quantile(flat, 0.50),
        "p95_ns": empirical_type7_quantile(flat, 0.95),
        "p99_ns": empirical_type7_quantile(flat, 0.99),
        "total_ns": sum(flat),
    }


def _ratio_summary(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if not len(flat) or np.any(~np.isfinite(flat)) or np.any(flat <= 0):
        raise ValueError("restart ratios must be finite and strictly positive")
    logs = np.log(flat)
    return {
        "values": [float(value) for value in flat],
        "minimum": float(np.min(flat)),
        "p05": empirical_type7_quantile(flat.tolist(), 0.05),
        "median": empirical_type7_quantile(flat.tolist(), 0.50),
        "p95": empirical_type7_quantile(flat.tolist(), 0.95),
        "maximum": float(np.max(flat)),
        "log_standard_deviation": (
            float(np.std(logs, ddof=1)) if len(logs) > 1 else 0.0
        ),
    }


def _bootstrap_seed(
    root_seed: int, *, suite_id: str, batch_size: int, pair_name: str
) -> int:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    digest = hashlib.sha256(
        root_seed.to_bytes(max(1, (root_seed.bit_length() + 7) // 8), "little")
        + bytes.fromhex(suite_id)
        + b"pinball-promatch-matched-latency-bootstrap-v1\0"
        + batch_size.to_bytes(8, "little")
        + pair_name.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _hierarchical_bootstrap(
    numerator_blocks: np.ndarray,
    denominator_blocks: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    numerator = np.asarray(numerator_blocks, dtype=np.float64)
    denominator = np.asarray(denominator_blocks, dtype=np.float64)
    if (
        numerator.ndim != 2
        or numerator.shape != denominator.shape
        or 0 in numerator.shape
        or np.any(numerator <= 0)
        or np.any(denominator <= 0)
    ):
        raise ValueError("paired block totals must be aligned positive 2D arrays")
    ratios = numerator / denominator
    estimate = float(np.exp(np.mean(np.log(ratios))))
    restart_estimates = np.exp(np.mean(np.log(ratios), axis=1))
    rng = np.random.default_rng(seed)
    restarts, blocks = ratios.shape
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        restart_ids = rng.integers(0, restarts, size=restarts)
        sampled: list[np.ndarray] = []
        for restart_id in restart_ids:
            block_ids = rng.integers(0, blocks, size=blocks)
            sampled.append(ratios[restart_id, block_ids])
        bootstrap[replicate] = np.exp(np.mean(np.log(np.concatenate(sampled))))
    return {
        "method": "restart-then-paired-block-percentile",
        "replicates": replicates,
        "alpha_two_sided": alpha,
        "seed": seed,
        "geometric_paired_block_ratio": {
            "estimate": estimate,
            "interval": {
                "lower": empirical_type7_quantile(bootstrap.tolist(), alpha / 2),
                "upper": empirical_type7_quantile(
                    bootstrap.tolist(), 1 - alpha / 2
                ),
            },
        },
        "restart_dispersion": _ratio_summary(restart_estimates),
    }


def _runtime_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("latency restart provenance is malformed")
    runtime = provenance.get("runtime_start")
    if not isinstance(runtime, Mapping):
        raise ValueError("latency restart runtime provenance is malformed")
    return {
        field: runtime.get(field)
        for field in (
            "cpu_model",
            "microcode",
            "cpu_affinity",
            "numa_nodes",
            "os",
            "kernel",
            "machine",
            "python",
            "packages",
            "native_thread_environment",
        )
    }


def _task_for_record(
    *,
    suite: Mapping[str, Any],
    protocol: LatencyProtocol,
    batch_size: int,
    restart: int,
) -> latency.LatencyRestartTask | latency.PreloadedLatencyRestartTask:
    common = {
        "protocol": protocol,
        "restart_index": restart,
        "batch_size": batch_size,
        "protocol_id": suite["protocol_id"],
        "suite_id": suite["suite_id"],
        "workload_id": suite["workload_id"],
        "workload_identity": suite["workload_identity"],
    }
    if suite["execution_mode"] == "fork-preloaded":
        return latency.PreloadedLatencyRestartTask(**common)
    return latency.LatencyRestartTask(factory=object(), **common)


def _validate_suite(
    root: Path,
    *,
    recorded_protocol: Mapping[str, Any],
    protocol: LatencyProtocol,
    supplied_protocol: LatencyProtocol | None,
    factory: object | None,
) -> tuple[dict[str, Any], list[str]]:
    if supplied_protocol is not None:
        if not isinstance(supplied_protocol, LatencyProtocol):
            raise TypeError("protocol must be LatencyProtocol")
        if supplied_protocol.to_json() != dict(recorded_protocol):
            raise ValueError("supplied latency protocol differs from recorded protocol")
    suite = load_json_strict(root / "suite.json", description="matched latency suite")
    if set(suite) != _SUITE_FIELDS or suite.get("schema") != latency.SUITE_SCHEMA:
        raise ValueError("matched latency suite fields or schema are malformed")
    protocol_id = _json_digest(recorded_protocol)
    if suite.get("protocol_id") != protocol_id:
        raise ValueError("matched latency suite protocol identity mismatch")
    workload_identity = suite.get("workload_identity")
    if not isinstance(workload_identity, Mapping):
        raise ValueError("matched latency suite workload identity is malformed")
    workload_id = _json_digest(dict(workload_identity))
    if suite.get("workload_id") != workload_id:
        raise ValueError("matched latency suite workload identity mismatch")
    mode = suite.get("execution_mode")
    if mode not in ("spawn-factory", "fork-preloaded"):
        raise ValueError("matched latency suite execution mode is malformed")
    expected_suite_id = _json_digest(
        {
            "protocol_id": protocol_id,
            "workload_id": workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
            "execution_mode": mode,
        }
    )
    if suite.get("suite_id") != expected_suite_id:
        raise ValueError("matched latency suite identity derivation mismatch")
    expected_process = "fork" if mode == "fork-preloaded" else "spawn"
    if (
        suite.get("fresh_process_per_restart") is not True
        or suite.get("timed_restart_concurrency") != 1
        or suite.get("restart_concurrency_policy")
        != "serialized-to-avoid-mutual-contention"
        or suite.get("process_start_method") != expected_process
        or suite.get("parent_preload_once") is not (mode == "fork-preloaded")
        or suite.get("native_threads") != 1
        or suite.get("affinity_policy") != protocol.host_policy.to_json()
    ):
        raise ValueError("matched latency suite execution controls differ")
    if factory is not None and _factory_identity(factory) != dict(workload_identity):
        raise ValueError("supplied latency factory identity differs from suite")
    names = [
        f"batch-{batch.batch_size}.restart-{restart:02d}.json"
        for batch in protocol.batches
        for restart in range(batch.restarts)
    ]
    if suite.get("restart_ledgers") != names:
        raise ValueError("matched latency restart ledger order differs")
    hashes = suite.get("restart_ledger_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(names):
        raise ValueError("matched latency restart digest map is malformed")
    for name, digest in hashes.items():
        _validate_digest(digest, name=f"restart digest {name}")
    actual = {path.name for path in root.iterdir()}
    if actual != {"protocol.json", "suite.json", *names}:
        raise ValueError("matched latency artifact set is incomplete or unexpected")
    return dict(suite), names


def analyze_latency_suite(
    latency_out: str | Path,
    *,
    protocol: LatencyProtocol | None = None,
    factory: object | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0x4D_41_54_43_48_45_44,
    alpha: float = 0.05,
) -> MatchedLatencyAnalysisArtifacts:
    """Authenticates a complete suite and analyzes paired block totals."""

    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise ValueError("bootstrap_replicates must be positive")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    alpha = float(alpha)
    root = Path(latency_out)
    if root.name == "suite.json":
        root = root.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("matched latency input must be a regular directory")
    for metadata_name in ("protocol.json", "suite.json"):
        metadata_path = root / metadata_name
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError(
                f"matched latency metadata must be a regular file: {metadata_name}"
            )
    recorded_protocol = load_json_strict(
        root / "protocol.json", description="matched latency protocol"
    )
    parsed_protocol = _protocol_from_json(recorded_protocol)
    suite, names = _validate_suite(
        root,
        recorded_protocol=recorded_protocol,
        protocol=parsed_protocol,
        supplied_protocol=protocol,
        factory=factory,
    )
    del names
    collected: dict[int, dict[str, list[dict[str, Any]]]] = {
        batch.batch_size: {pair: [] for pair in latency.PAIR_NAMES}
        for batch in parsed_protocol.batches
    }
    runtime_signature: dict[str, Any] | None = None
    for batch in parsed_protocol.batches:
        for restart in range(batch.restarts):
            name = f"batch-{batch.batch_size}.restart-{restart:02d}.json"
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"matched latency restart must be a regular file: {name}")
            if _sha256(path.read_bytes()) != suite["restart_ledger_sha256"][name]:
                raise ValueError(f"matched latency restart digest mismatch: {name}")
            record = load_json_strict(path, description=f"matched latency restart {name}")
            task = _task_for_record(
                suite=suite,
                protocol=parsed_protocol,
                batch_size=batch.batch_size,
                restart=restart,
            )
            latency.validate_restart_record(record, task=task)
            signature = _runtime_signature(record)
            if runtime_signature is None:
                runtime_signature = signature
            elif signature != runtime_signature:
                raise ValueError("matched latency restart runtime environments differ")
            for pair_name in latency.PAIR_NAMES:
                pair = record["pairs"][pair_name]
                collected[batch.batch_size][pair_name].append(
                    {
                        "numerator_calls": np.asarray(
                            [
                                [call["duration_ns"] for call in block]
                                for block in pair["numerator_calls"]
                            ],
                            dtype=np.float64,
                        ),
                        "denominator_calls": np.asarray(
                            [
                                [call["duration_ns"] for call in block]
                                for block in pair["denominator_calls"]
                            ],
                            dtype=np.float64,
                        ),
                        "numerator_blocks": np.asarray(
                            pair["numerator_block_totals_ns"], dtype=np.float64
                        ),
                        "denominator_blocks": np.asarray(
                            pair["denominator_block_totals_ns"], dtype=np.float64
                        ),
                    }
                )
    if runtime_signature is None:  # pragma: no cover - protocols cannot be empty
        raise AssertionError("complete suite yielded no runtime provenance")
    pair_specs = {pair.name: pair for pair in latency.FIXED_PAIRS}
    batches_json: dict[str, Any] = {}
    for batch in parsed_protocol.batches:
        pairs_json: dict[str, Any] = {}
        for pair_name in latency.PAIR_NAMES:
            rows = collected[batch.batch_size][pair_name]
            numerator_calls = np.stack([row["numerator_calls"] for row in rows])
            denominator_calls = np.stack([row["denominator_calls"] for row in rows])
            numerator_blocks = np.stack([row["numerator_blocks"] for row in rows])
            denominator_blocks = np.stack([row["denominator_blocks"] for row in rows])
            expected_call_shape = (
                batch.restarts,
                batch.blocks_per_restart,
                batch.timed_calls_per_side_per_block,
            )
            expected_block_shape = (batch.restarts, batch.blocks_per_restart)
            if (
                numerator_calls.shape != expected_call_shape
                or denominator_calls.shape != expected_call_shape
                or numerator_blocks.shape != expected_block_shape
                or denominator_blocks.shape != expected_block_shape
            ):
                raise ValueError("matched latency aggregate dimensions differ")
            numerator_summary = _quantile_summary(numerator_calls)
            denominator_summary = _quantile_summary(denominator_calls)
            for summary in (numerator_summary, denominator_summary):
                summary["median_batch_latency_ns"] = summary["p50_ns"]
            throughput = None
            if batch.batch_size in (64, 1024):
                throughput = {
                    "numerator_shots_per_second": (
                        batch.batch_size
                        * numerator_summary["calls"]
                        * 1_000_000_000
                        / numerator_summary["total_ns"]
                    ),
                    "denominator_shots_per_second": (
                        batch.batch_size
                        * denominator_summary["calls"]
                        * 1_000_000_000
                        / denominator_summary["total_ns"]
                    ),
                    "numerator_median_per_shot_ns": (
                        numerator_summary["p50_ns"] / batch.batch_size
                    ),
                    "denominator_median_per_shot_ns": (
                        denominator_summary["p50_ns"] / batch.batch_size
                    ),
                }
            bootstrap_pair_seed = _bootstrap_seed(
                bootstrap_seed,
                suite_id=suite["suite_id"],
                batch_size=batch.batch_size,
                pair_name=pair_name,
            )
            spec = pair_specs[pair_name]
            pairs_json[pair_name] = {
                "numerator": spec.numerator,
                "denominator": spec.denominator,
                "raw_counts": {
                    "restarts": batch.restarts,
                    "blocks_per_restart": batch.blocks_per_restart,
                    "calls_per_side_per_block": batch.timed_calls_per_side_per_block,
                    "calls_per_side": int(numerator_calls.size),
                    "timed_shots_per_side": int(
                        numerator_calls.size * batch.batch_size
                    ),
                },
                "numerator_call_summary": numerator_summary,
                "denominator_call_summary": denominator_summary,
                "descriptive_throughput": throughput,
                "inference": _hierarchical_bootstrap(
                    numerator_blocks,
                    denominator_blocks,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_pair_seed,
                    alpha=alpha,
                ),
            }
        batches_json[str(batch.batch_size)] = {
            "batch_size": batch.batch_size,
            "endpoint_role": "primary" if batch.batch_size == 1 else "descriptive",
            "pairs": pairs_json,
        }
    payload: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "protocol_id": suite["protocol_id"],
        "workload_id": suite["workload_id"],
        "workload_identity": dict(suite["workload_identity"]),
        "execution": {
            "execution_mode": suite["execution_mode"],
            "process_start_method": suite["process_start_method"],
            "parent_preload_once": suite["parent_preload_once"],
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
            "native_threads": 1,
            "runtime_signature": runtime_signature,
        },
        "timing_scope": {
            "software_latency_only": True,
            "public_packed_production_calls": True,
            "decoder_compilation_excluded": True,
            "corpus_preparation_excluded": True,
            "telemetry_excluded": True,
        },
        "bootstrap": {
            "method": "restart-then-paired-block-percentile",
            "replicates": bootstrap_replicates,
            "root_seed": bootstrap_seed,
            "alpha_two_sided": alpha,
            "quantile": "empirical-type-7",
        },
        "batch_1_primary": True,
        "secondary_batches_descriptive": [
            batch.batch_size
            for batch in parsed_protocol.batches
            if batch.batch_size != 1
        ],
        "batches": batches_json,
    }
    payload["payload_sha256"] = _sha256(canonical_json_bytes(payload))
    analysis_bytes = canonical_json_bytes(payload)
    report = render_latency_markdown(payload)
    return MatchedLatencyAnalysisArtifacts(
        analysis=payload,
        report_markdown=report,
        analysis_bytes=analysis_bytes,
        report_bytes=report.encode("utf-8"),
    )


def _milliseconds(nanoseconds: float) -> str:
    return f"{nanoseconds / 1_000_000:.3f}"


def render_latency_markdown(analysis: Mapping[str, Any]) -> str:
    """Renders a compact human report from a validated analysis payload."""

    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("analysis has the wrong matched latency schema")
    lines = [
        "# Matched frontend software latency",
        "",
        "Batch 1 is the primary response-latency endpoint. Larger batches are "
        "descriptive throughput measurements.",
        "",
        "Compilation, corpus preparation, telemetry, and serialization are excluded "
        "from all timed intervals. These are software measurements, not hardware "
        "decoder latency or a real-time deadline claim.",
        "",
    ]
    execution = analysis["execution"]
    if execution["execution_mode"] == "fork-preloaded":
        lines.extend(
            [
                "The production workload was compiled once in the parent and inherited "
                "by fresh serialized fork/COW children. Warmup and timing occurred in "
                "each child; compile cost is not represented.",
                "",
            ]
        )
    for batch_key, batch in analysis["batches"].items():
        role = "primary" if batch["endpoint_role"] == "primary" else "descriptive"
        lines.extend(
            [
                f"## Batch {batch_key} ({role})",
                "",
                "| Pair | Geometric ratio (95% CI) | Numerator p50/p95/p99 (ms) | "
                "Denominator p50/p95/p99 (ms) |",
                "|---|---:|---:|---:|",
            ]
        )
        for pair in batch["pairs"].values():
            numerator = _ARM_LABELS[pair["numerator"]]
            denominator = _ARM_LABELS[pair["denominator"]]
            ratio = pair["inference"]["geometric_paired_block_ratio"]
            num = pair["numerator_call_summary"]
            den = pair["denominator_call_summary"]
            lines.append(
                f"| {numerator} / {denominator} | {ratio['estimate']:.4f} "
                f"[{ratio['interval']['lower']:.4f}, "
                f"{ratio['interval']['upper']:.4f}] | "
                f"{_milliseconds(num['p50_ns'])} / {_milliseconds(num['p95_ns'])} / "
                f"{_milliseconds(num['p99_ns'])} | "
                f"{_milliseconds(den['p50_ns'])} / {_milliseconds(den['p95_ns'])} / "
                f"{_milliseconds(den['p99_ns'])} |"
            )
        if batch["batch_size"] in (64, 1024):
            lines.extend(["", "Effective throughput (shots/s):"])
            for pair in batch["pairs"].values():
                throughput = pair["descriptive_throughput"]
                lines.append(
                    f"- {_ARM_LABELS[pair['numerator']]} / "
                    f"{_ARM_LABELS[pair['denominator']]}: "
                    f"{throughput['numerator_shots_per_second']:.1f} / "
                    f"{throughput['denominator_shots_per_second']:.1f}."
                )
        lines.append("")
    lines.append(f"Analysis digest: `{analysis['payload_sha256']}`.")
    return "\n".join(lines) + "\n"


__all__ = [
    "ANALYSIS_SCHEMA",
    "MatchedLatencyAnalysisArtifacts",
    "analyze_latency_suite",
    "render_latency_markdown",
]
