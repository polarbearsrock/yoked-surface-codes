"""Read-only analysis of authenticated Patch-UF latency suites."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from yoked.decoding._artifact_io import install_bytes_atomic, load_json_strict
from yoked.decoding import _patch_uf_latency as latency
from yoked.decoding._promatch_stats import (
    canonical_json_bytes,
    empirical_type7_quantile,
    paired_geometric_mean_ratio,
)


LATENCY_ANALYSIS_SCHEMA = "patch-uf-latency-analysis-v1"


__all__ = [
    "LATENCY_ANALYSIS_SCHEMA",
    "LatencyAnalysisArtifacts",
    "analyze_latency_suite",
    "write_latency_analysis_bundle",
]


@dataclasses.dataclass(frozen=True)
class LatencyAnalysisArtifacts:
    analysis: Mapping[str, Any]
    report_markdown: str
    analysis_bytes: bytes
    report_bytes: bytes


def _protocol_from_json(raw: Mapping[str, Any]) -> latency.LatencyProtocol:
    if set(raw) != {
        "batches",
        "schedule_seed",
        "host_policy",
        "variant_names",
        "pairs",
    }:
        raise ValueError("latency protocol fields are malformed")
    if raw["variant_names"] != list(latency.VARIANT_NAMES):
        raise ValueError("latency protocol variant names differ")
    if raw["pairs"] != [dataclasses.asdict(item) for item in latency.FIXED_PAIRS]:
        raise ValueError("latency protocol pair definitions differ")
    host = raw["host_policy"]
    if not isinstance(host, Mapping):
        raise TypeError("latency host_policy must be an object")
    policy = latency.HostPolicy(
        cpu_affinity=tuple(host.get("cpu_affinity", ())),
        expected_host=tuple(sorted(dict(host.get("expected_host", {})).items())),
        expected_numa_nodes=tuple(host.get("expected_numa_nodes", ())),
    )
    batches = raw["batches"]
    if not isinstance(batches, list):
        raise TypeError("latency batches must be an array")
    try:
        schedule_seed = int(str(raw["schedule_seed"]), 16)
    except ValueError as ex:
        raise ValueError("latency schedule_seed must be hexadecimal") from ex
    result = latency.LatencyProtocol(
        batches=tuple(latency.BatchTiming(**dict(item)) for item in batches),
        schedule_seed=schedule_seed,
        host_policy=policy,
    )
    if result.to_json() != dict(raw):
        raise ValueError("latency protocol does not round-trip canonically")
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    flat = [float(item) for item in np.asarray(values).ravel()]
    if not flat or any(not math.isfinite(item) or item <= 0 for item in flat):
        raise ValueError("latency durations must be finite and strictly positive")
    return {
        "calls": len(flat),
        "median_ns": empirical_type7_quantile(flat, 0.5),
        "p90_ns": empirical_type7_quantile(flat, 0.9),
        "p95_ns": empirical_type7_quantile(flat, 0.95),
        "p99_ns": empirical_type7_quantile(flat, 0.99),
        "total_ns": int(sum(flat)),
    }


def _bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    a = np.asarray(numerator, dtype=np.float64)
    b = np.asarray(denominator, dtype=np.float64)
    if a.ndim != 3 or a.shape != b.shape or 0 in a.shape:
        raise ValueError("latency arrays must have identical nonempty 3D shapes")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("latency arrays must be strictly positive")
    block_a = np.sum(a, axis=2)
    block_b = np.sum(b, axis=2)
    geometric = paired_geometric_mean_ratio(block_a.ravel(), block_b.ravel())
    p99_ratio = empirical_type7_quantile(a.ravel(), 0.99) / empirical_type7_quantile(
        b.ravel(), 0.99
    )
    rng = np.random.default_rng(seed)
    geometric_values: list[float] = []
    p99_values: list[float] = []
    restarts, blocks, _ = a.shape
    for _ in range(replicates):
        selected_a = []
        selected_b = []
        for source_restart in rng.integers(0, restarts, size=restarts):
            block_ids = rng.integers(0, blocks, size=blocks)
            selected_a.append(a[source_restart, block_ids, :])
            selected_b.append(b[source_restart, block_ids, :])
        sample_a = np.stack(selected_a)
        sample_b = np.stack(selected_b)
        geometric_values.append(
            paired_geometric_mean_ratio(
                np.sum(sample_a, axis=2).ravel(),
                np.sum(sample_b, axis=2).ravel(),
            )
        )
        p99_values.append(
            empirical_type7_quantile(sample_a.ravel(), 0.99)
            / empirical_type7_quantile(sample_b.ravel(), 0.99)
        )
    return {
        "method": "restart-then-paired-block-percentile",
        "replicates": replicates,
        "alpha_two_sided": alpha,
        "seed": seed,
        "geometric_paired_block_ratio": {
            "estimate": geometric,
            "interval": {
                "lower": empirical_type7_quantile(geometric_values, alpha / 2),
                "upper": empirical_type7_quantile(
                    geometric_values, 1 - alpha / 2
                ),
            },
        },
        "pooled_type7_p99_ratio": {
            "estimate": p99_ratio,
            "interval": {
                "lower": empirical_type7_quantile(p99_values, alpha / 2),
                "upper": empirical_type7_quantile(p99_values, 1 - alpha / 2),
            },
        },
    }


def _seed(suite_id: str, *, batch_size: int, pair_name: str) -> int:
    digest = hashlib.sha256(
        bytes.fromhex(suite_id)
        + b"patch-uf-latency-analysis-bootstrap-v1\0"
        + batch_size.to_bytes(8, "little")
        + pair_name.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _report(data: Mapping[str, Any]) -> str:
    lines = [
        "# Patch-UF software latency analysis",
        "",
        f"Analysis digest: `{data['payload_sha256']}`.",
        "",
        "Inference scope: fixed characterization corpus on the recorded host. "
        "Batch 1 is primary; batches 64 and 1,024 are descriptive throughput diagnostics.",
        "",
    ]
    for batch, result in data["batches"].items():
        lines.extend([f"## Batch {batch}", ""])
        for pair_name, pair in result["pairs"].items():
            ratio = pair["inference"]["geometric_paired_block_ratio"]
            lines.append(
                f"- `{pair_name}`: geometric paired-block ratio "
                f"{ratio['estimate']}; numerator p99 "
                f"{pair['numerator_summary']['p99_ns']} ns, denominator p99 "
                f"{pair['denominator_summary']['p99_ns']} ns."
            )
        lines.append("")
    return "\n".join(lines)


def analyze_latency_suite(
    latency_out: Path,
    *,
    bootstrap_replicates: int = 10_000,
    alpha: float = 0.05,
) -> LatencyAnalysisArtifacts:
    """Authenticate every latency ledger and compute fixed-context summaries."""

    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise ValueError("bootstrap_replicates must be positive")
    if not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    alpha = float(alpha)
    root = Path(latency_out)
    if root.name == "suite.json":
        root = root.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("latency input must be a regular directory")
    protocol_json = load_json_strict(
        root / "protocol.json", description="Patch-UF latency protocol"
    )
    protocol = _protocol_from_json(protocol_json)
    suite = load_json_strict(root / "suite.json", description="Patch-UF latency suite")
    if suite.get("schema") != latency.SUITE_SCHEMA:
        raise ValueError("latency suite schema mismatch")
    protocol_id = latency._json_digest(protocol_json)
    if suite.get("protocol_id") != protocol_id:
        raise ValueError("latency suite protocol identity mismatch")
    suite_id = suite.get("suite_id")
    workload_id = suite.get("workload_id")
    if not isinstance(suite_id, str) or len(suite_id) != 64:
        raise ValueError("latency suite_id is malformed")
    try:
        bytes.fromhex(suite_id)
    except ValueError as ex:
        raise ValueError("latency suite_id is malformed") from ex
    workload_identity = suite.get("workload_identity")
    if not isinstance(workload_identity, Mapping):
        raise ValueError("latency suite workload identity is malformed")
    if workload_id != latency._json_digest(dict(workload_identity)):
        raise ValueError("latency suite workload identity mismatch")
    expected_suite_id = latency._json_digest(
        {
            "protocol_id": protocol_id,
            "workload_id": workload_id,
            "fresh_process_per_restart": True,
            "timed_restart_concurrency": 1,
        }
    )
    if suite_id != expected_suite_id:
        raise ValueError("latency suite identity derivation mismatch")
    if suite.get("fresh_process_per_restart") is not True or suite.get(
        "timed_restart_concurrency"
    ) != 1:
        raise ValueError("latency suite restart policy mismatch")
    if suite.get("affinity_policy") != protocol.host_policy.to_json():
        raise ValueError("latency suite affinity policy mismatch")
    expected_names = [
        f"batch-{batch.batch_size}.restart-{restart:02d}.json"
        for batch in protocol.batches
        for restart in range(batch.restarts)
    ]
    if suite.get("restart_ledgers") != expected_names:
        raise ValueError("latency suite restart ledger order mismatch")
    digests = suite.get("restart_ledger_sha256")
    if not isinstance(digests, Mapping) or set(digests) != set(expected_names):
        raise ValueError("latency suite restart digest map mismatch")
    actual_names = {item.name for item in root.iterdir()}
    if actual_names != {"protocol.json", "suite.json", *expected_names}:
        raise ValueError("latency input artifact set is incomplete or unexpected")

    collected: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        batch.batch_size: {name: [] for name in latency.PAIR_NAMES}
        for batch in protocol.batches
    }
    workload_keys: dict[tuple[int, str, str], set[str]] = {}
    for batch in protocol.batches:
        for restart in range(batch.restarts):
            name = f"batch-{batch.batch_size}.restart-{restart:02d}.json"
            path = root / name
            if _sha256_file(path) != digests[name]:
                raise ValueError(f"latency restart digest mismatch: {name}")
            record = load_json_strict(path, description=f"latency restart {name}")
            latency._validate_existing_restart(
                record,
                protocol_id=protocol_id,
                suite_id=suite_id,
                workload_id=workload_id,
                batch=batch,
                restart=restart,
            )
            for pair_name in latency.PAIR_NAMES:
                pair = record["pairs"][pair_name]
                arrays = []
                for side in ("numerator", "denominator"):
                    rows = pair[f"{side}_calls"]
                    arrays.append(
                        np.asarray(
                            [
                                [call["duration_ns"] for call in block]
                                for block in rows
                            ],
                            dtype=np.float64,
                        )
                    )
                    key_set = workload_keys.setdefault(
                        (batch.batch_size, pair_name, side), set()
                    )
                    for block in rows:
                        for call in block:
                            key_set.update(
                                json.dumps(key, sort_keys=True, separators=(",", ":"))
                                for key in call["workload_keys"]
                            )
                collected[batch.batch_size][pair_name].append(
                    (arrays[0], arrays[1])
                )

    batches_json: dict[str, Any] = {}
    pair_specs = {item.name: item for item in latency.FIXED_PAIRS}
    for batch in protocol.batches:
        pair_json: dict[str, Any] = {}
        for pair_name in latency.PAIR_NAMES:
            rows = collected[batch.batch_size][pair_name]
            numerator = np.stack([item[0] for item in rows])
            denominator = np.stack([item[1] for item in rows])
            expected_shape = (
                batch.restarts,
                batch.blocks_per_restart,
                batch.timed_calls_per_side_per_block,
            )
            if numerator.shape != expected_shape or denominator.shape != expected_shape:
                raise ValueError("latency aggregate array shape mismatch")
            num_summary = _quantiles(numerator)
            den_summary = _quantiles(denominator)
            for side, summary_row in (
                ("numerator", num_summary),
                ("denominator", den_summary),
            ):
                summary_row["throughput_shots_per_second"] = (
                    batch.batch_size
                    * summary_row["calls"]
                    * 1_000_000_000
                    / summary_row["total_ns"]
                )
                summary_row["unique_workload_key_count"] = len(
                    workload_keys[(batch.batch_size, pair_name, side)]
                )
            spec = pair_specs[pair_name]
            pair_json[pair_name] = {
                "numerator": spec.numerator,
                "denominator": spec.denominator,
                "array_shape": list(expected_shape),
                "numerator_summary": num_summary,
                "denominator_summary": den_summary,
                "inference": _bootstrap(
                    numerator,
                    denominator,
                    replicates=bootstrap_replicates,
                    seed=_seed(
                        suite_id, batch_size=batch.batch_size, pair_name=pair_name
                    ),
                    alpha=alpha,
                ),
            }
        batches_json[str(batch.batch_size)] = {
            "batch_size": batch.batch_size,
            "endpoint_role": "primary" if batch.batch_size == 1 else "descriptive",
            "pairs": pair_json,
        }
    payload: dict[str, Any] = {
        "schema": LATENCY_ANALYSIS_SCHEMA,
        "schema_version": 1,
        "suite_id": suite_id,
        "protocol_id": protocol_id,
        "workload_id": workload_id,
        "workload_identity": dict(workload_identity),
        "inference_scope": "fixed_characterization_corpus_recorded_host",
        "bootstrap": {
            "method": "restart-then-paired-block-percentile",
            "replicates": bootstrap_replicates,
            "alpha_two_sided": alpha,
            "quantile": "empirical-type-7",
        },
        "batch_1_primary": "1" in batches_json,
        "secondary_batches_descriptive": [
            int(key) for key in batches_json if key != "1"
        ],
        "batches": batches_json,
    }
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    analysis_bytes = canonical_json_bytes(payload)
    report = _report(payload)
    return LatencyAnalysisArtifacts(
        analysis=payload,
        report_markdown=report,
        analysis_bytes=analysis_bytes,
        report_bytes=report.encode("utf-8"),
    )


def write_latency_analysis_bundle(
    out: Path, artifacts: LatencyAnalysisArtifacts
) -> None:
    if not isinstance(artifacts, LatencyAnalysisArtifacts):
        raise TypeError("artifacts must be LatencyAnalysisArtifacts")
    out = Path(out)
    if out.exists() and (out.is_symlink() or not out.is_dir()):
        raise ValueError("latency analysis output must be a regular directory")
    out.mkdir(parents=True, exist_ok=True)
    allowed = {"latency_analysis.json", "report.md"}
    if {item.name for item in out.iterdir()} - allowed:
        raise ValueError("latency analysis output contains unexpected entries")
    for path, data, prefix in (
        (
            out / "latency_analysis.json",
            artifacts.analysis_bytes,
            "patch-uf-latency-analysis-",
        ),
        (out / "report.md", artifacts.report_bytes, "patch-uf-latency-report-"),
    ):
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"existing latency analysis differs: {path}")
        else:
            install_bytes_atomic(path, data, prefix=prefix, overwrite=False)
