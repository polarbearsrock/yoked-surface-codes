"""Offline, deterministic analysis for the ProMatch B1 policy audit.

This module is intentionally downstream-only.  It reads immutable canonical
``*.jsonl.gz`` worker shards, and it never imports circuit generation, sampling,
matching, or decoding code.  All oracle and context facts used here must have
already been committed to the collection ledgers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from yoked.decoding._promatch_stats import (
    PairedContingency,
    tango_paired_risk_difference_upper,
)


ANALYSIS_SCHEMA = "promatch-l1-policy-audit-analysis-v1"
ANALYSIS_MANIFEST_SCHEMA = "promatch-l1-policy-audit-analysis-manifest-v1"
ANALYSIS_READY_SCHEMA = "promatch-l1-policy-audit-analysis-ready-v1"
HUMAN_REPORT_FORMAT = "promatch-l1-policy-audit-human-report-v1"
HUMAN_REPORT_FILE = "report.md"
PLOT_TABLE_SCHEMA = "promatch-l1-policy-audit-plot-table-v1"
CASEBOOK_SCHEMA = "promatch-l1-policy-audit-casebook-selection-v1"
SPARSE_UNSAFE_STATES = 100
CASEBOOK_MIN_STATES = 20
CONTEXT_PRIORITY = (
    "yoke",
    "true-boundary",
    "terminal",
    "cross-window",
    "cross-patch-or-basis",
    "support-cancellation",
    "in-domain",
)
CONTEXT_LABELS = frozenset(CONTEXT_PRIORITY)
DEGENERACY_DIAGNOSTICS = frozenset(
    {
        "same-pair-different-path-or-frame",
        "equal-weight-logical-class",
        "disconnected-support-reconfiguration",
        "unclassified",
    }
)
COLLECTOR_GATE_ATTESTATION_SCHEMA = (
    "promatch-l1-policy-audit-fatal-gate-attestation-v1"
)
COLLECTOR_GATE_CHECKS = {
    3: ("scalar-batch-u0",),
    4: ("shadow-frozen-v3-equivalence",),
    7: (
        "backend-support-fsum",
        "decimal-4096",
        "uncached-repeatability",
        "tolerance-grid",
    ),
    8: ("actual-observable-invariance",),
    9: ("veto-state-frame-prefix-invariance",),
    14: ("matching-pair-and-support-reconciliation",),
    16: ("cached-uncached-oracle-repeatability",),
    18: ("execution-and-source-provenance",),
}
TERMINAL_ACTIONS = (
    "same-stage-alternative",
    "later-stage-alternative",
    "abstain-true-exhaustion",
    "censored-invalid",
)
CERTIFICATE_CLASSES = (
    "positive-cost-excess",
    "cost-compatible-frame-conflict",
    "O-frame-safe",
)
SHARD_FILES = {
    "shots": "shots.jsonl.gz",
    "proposals": "proposals.jsonl.gz",
    "counterfactuals": "counterfactuals.jsonl.gz",
    "domains": "domains.jsonl.gz",
}
EXPECTED_LEDGER_SCHEMAS = {
    "shots": "promatch-l1-policy-audit-shot-v1",
    "proposals": "promatch-l1-policy-audit-proposal-v1",
    "counterfactuals": "promatch-l1-policy-audit-counterfactual-v1",
    "domains": "promatch-l1-policy-audit-domain-v1",
}


class PolicyAnalysisError(ValueError):
    """The immutable audit corpus violates its analysis contract."""


def _collector_gate_attestations(
    manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    raw = manifest.get("fatal_gate_attestations")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PolicyAnalysisError("fatal_gate_attestations must be an object")
    expected_keys = {str(gate) for gate in COLLECTOR_GATE_CHECKS}
    if set(raw) != expected_keys:
        raise PolicyAnalysisError(
            "fatal_gate_attestations does not cover the exact collector-only gate set"
        )
    result: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            raise PolicyAnalysisError(f"fatal gate {key} attestation must be an object")
        gate = int(key)
        expected = {
            "schema": COLLECTOR_GATE_ATTESTATION_SCHEMA,
            "gate": gate,
            "status": "passed",
            "scope": "frozen-protocol-required-scope",
            "checks": list(COLLECTOR_GATE_CHECKS[gate]),
            "failures": 0,
        }
        if dict(value) != expected:
            raise PolicyAnalysisError(f"fatal gate {gate} attestation is not exact")
        result[gate] = dict(value)
    return result


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyAnalysisError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as ex:
        raise PolicyAnalysisError(f"cannot read canonical JSON {path}") from ex
    if not isinstance(value, dict):
        raise PolicyAnalysisError(f"{path} must contain one JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise PolicyAnalysisError(f"{path} is not canonical JSON")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if any(key in value for key in ("path", "relative_path", "artifact_path")):
            yield value
        for child in value.values():
            yield from _manifest_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_records(child)


def _manifest_record(manifest: Mapping[str, Any], relative: str) -> Mapping[str, Any]:
    matches = []
    for row in _manifest_records(manifest):
        path = row.get("path", row.get("relative_path", row.get("artifact_path")))
        if isinstance(path, str) and Path(path).as_posix() == relative:
            matches.append(row)
    if len(matches) != 1:
        raise PolicyAnalysisError(
            f"manifest must contain exactly one record for {relative!r}; got {len(matches)}"
        )
    return matches[0]


def _record_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _load_jsonl_gzip(
    path: Path, *, manifest: Mapping[str, Any], root: Path
) -> list[dict[str, Any]]:
    try:
        compressed = path.read_bytes()
        uncompressed = gzip.decompress(compressed)
    except (OSError, EOFError, gzip.BadGzipFile) as ex:
        raise PolicyAnalysisError(f"cannot read gzip ledger {path}") from ex
    if len(compressed) < 10 or compressed[4:8] != b"\x00\x00\x00\x00" or compressed[3] & 0x08:
        raise PolicyAnalysisError(f"gzip ledger {path} is not deterministic mtime=0/name-empty")
    relative = path.relative_to(root).as_posix()
    record = _manifest_record(manifest, relative)
    expected_compressed = _record_value(
        record, "compressed_sha256", "gzip_sha256", "sha256"
    )
    expected_uncompressed = _record_value(
        record, "uncompressed_sha256", "content_sha256", "jsonl_sha256"
    )
    if not isinstance(expected_compressed, str) or _sha256(compressed) != expected_compressed:
        raise PolicyAnalysisError(f"compressed digest mismatch for {relative}")
    if not isinstance(expected_uncompressed, str) or _sha256(uncompressed) != expected_uncompressed:
        raise PolicyAnalysisError(f"uncompressed digest mismatch for {relative}")
    if uncompressed and not uncompressed.endswith(b"\n"):
        raise PolicyAnalysisError(f"canonical JSONL {relative} lacks its final newline")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(uncompressed.splitlines(), 1):
        try:
            row = json.loads(raw, object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError) as ex:
            raise PolicyAnalysisError(
                f"invalid JSON in {relative}:{line_number}"
            ) from ex
        if not isinstance(row, dict):
            raise PolicyAnalysisError(f"{relative}:{line_number} is not an object")
        if canonical_json_bytes(row) != raw:
            raise PolicyAnalysisError(f"{relative}:{line_number} is not canonical JSON")
        rows.append(row)
    expected_rows = _record_value(record, "row_count", "rows", "count")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise PolicyAnalysisError(f"manifest row count missing for {relative}")
    if expected_rows != len(rows):
        raise PolicyAnalysisError(f"row-count mismatch for {relative}")
    return rows


def _deep_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(_deep_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_deep_values(child, key))
    return found


def _one_deep(value: Any, names: Sequence[str], *, required: bool = False) -> Any:
    for name in names:
        found = _deep_values(value, name)
        if found:
            first = found[0]
            if any(item != first for item in found[1:]):
                raise PolicyAnalysisError(f"ambiguous values for semantic field {name!r}")
            return first
    if required:
        raise PolicyAnalysisError(f"missing semantic field; expected one of {list(names)}")
    return None


def _at(row: Mapping[str, Any], *paths: str, required: bool = False) -> Any:
    for path in paths:
        current: Any = row
        ok = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    if required:
        raise PolicyAnalysisError(f"missing semantic field; expected one of {paths}")
    return None


def _as_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PolicyAnalysisError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise PolicyAnalysisError(f"{name} must be nonnegative")
    return value


def _as_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyAnalysisError(f"{name} must be Boolean")
    return value


def _float_value(row: Mapping[str, Any], name: str) -> float | None:
    raw = _at(row, name, f"evaluation.{name}", f"oracle.{name}")
    raw_hex = _at(row, f"{name}_hex", f"evaluation.{name}_hex", f"oracle.{name}_hex")
    if raw is None and raw_hex is None:
        return None
    if raw_hex is not None:
        if not isinstance(raw_hex, str):
            raise PolicyAnalysisError(f"{name}_hex must be a string")
        try:
            exact = float.fromhex(raw_hex)
        except ValueError as ex:
            raise PolicyAnalysisError(f"invalid {name}_hex") from ex
        if raw is not None and float(raw) != exact:
            raise PolicyAnalysisError(f"{name} and {name}_hex disagree")
        result = exact
    else:
        result = float(raw)
    if not math.isfinite(result):
        raise PolicyAnalysisError(f"{name} must be finite")
    return result


def _required_float(row: Mapping[str, Any], name: str) -> float:
    value = _float_value(row, name)
    if value is None:
        raise PolicyAnalysisError(f"missing required finite float {name!r}")
    return value


def empirical_type7(values: Sequence[float], probability: float) -> float | None:
    """Exact public wrapper for the protocol's empirical type-7 quantile."""

    if not 0 <= probability <= 1 or not math.isfinite(probability):
        raise ValueError("probability must be finite and in [0, 1]")
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("quantile values must be a finite one-dimensional sequence")
    return float(np.quantile(array, probability, method="linear"))


def exact_ecdf(values: Sequence[float]) -> list[dict[str, Any]]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("ECDF values must be finite and one-dimensional")
    unique, counts = np.unique(array, return_counts=True)
    cumulative = np.cumsum(counts)
    denominator = len(array)
    return [
        {
            "value": float(value),
            "count": int(count),
            "cumulative_count": int(cum),
            "cumulative_fraction": float(cum / denominator),
            "denominator": denominator,
        }
        for value, count, cum in zip(unique, counts, cumulative)
    ]


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    data = [float(v) for v in values]
    return {
        "denominator": len(data),
        "median": empirical_type7(data, 0.5),
        "p10": empirical_type7(data, 0.1),
        "p90": empirical_type7(data, 0.9),
        "p99": empirical_type7(data, 0.99) if len(data) >= 1000 else None,
        "maximum": max(data) if data else None,
        "ecdf": exact_ecdf(data),
    }


def derive_bootstrap_seed(seed_root: str, *, cell_id: str, estimand: str) -> int:
    if not isinstance(seed_root, str) or not seed_root:
        raise PolicyAnalysisError("bootstrap seed root must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(b"promatch-b1-shot-bootstrap-v1\0")
    for value in (seed_root, cell_id, estimand):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:16], "little")


def clustered_bootstrap_ratios(
    contributions: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Bootstraps ratio-of-sums columns while preserving complete shots.

    ``contributions`` has columns ``num0, den0, num1, den1, ...``.  One row is
    one physical shot and therefore carries every proposal/domain contribution.
    """

    array = np.asarray(contributions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] % 2:
        raise ValueError("contributions must have an even number of columns")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("contributions must contain finite rows")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    metrics = array.shape[1] // 2
    draws: list[list[float]] = [[] for _ in range(metrics)]
    undefined = [0] * metrics
    # For one ratio, an ordinary complete-shot bootstrap depends only on the
    # empirical histogram of that shot's (numerator, denominator) pair.  A
    # multinomial draw from this histogram is exactly the same marginal
    # bootstrap distribution as gathering N sampled shot rows, while avoiding
    # O(replicates * shots * metrics) index and gather work.  Metrics are drawn
    # independently because the public product contains only marginal
    # intervals; no joint/covariance claim is made from these replicates.
    for metric in range(metrics):
        pairs = array[:, 2 * metric : 2 * metric + 2]
        unique_pairs, frequencies = np.unique(pairs, axis=0, return_counts=True)
        probabilities = frequencies.astype(np.float64) / array.shape[0]
        remaining = replicates
        while remaining:
            max_chunk = max(1, 4_000_000 // max(1, len(unique_pairs)))
            chunk = min(256, max_chunk, remaining)
            multiplicities = rng.multinomial(
                array.shape[0], probabilities, size=chunk
            )
            totals = multiplicities @ unique_pairs
            denominator = totals[:, 1]
            defined = denominator != 0
            undefined[metric] += int((~defined).sum())
            draws[metric].extend(
                (totals[defined, 0] / denominator[defined]).astype(float).tolist()
            )
            remaining -= chunk
    result = []
    for metric, samples in enumerate(draws):
        result.append(
            {
                "replicates": replicates,
                "defined_replicates": len(samples),
                "undefined_replicates": undefined[metric],
                "lower": empirical_type7(samples, alpha / 2),
                "upper": empirical_type7(samples, 1 - alpha / 2),
                "upper_one_sided": empirical_type7(samples, 1 - alpha / 2),
                "quantile_method": "empirical-type-7",
            }
        )
    return result


@dataclass(frozen=True)
class PolicyAuditCorpus:
    root: Path
    experiment: dict[str, Any]
    config: dict[str, Any]
    manifest: dict[str, Any]
    rows: dict[str, tuple[dict[str, Any], ...]]
    source_hashes: dict[str, str]


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    cell = _at(row, "cell_id", required=True)
    worker = _as_nonnegative_int(_at(row, "worker_id", required=True), name="worker_id")
    shot = _as_nonnegative_int(
        _at(row, "global_shot_id", "physical_identity.global_shot_id", required=True),
        name="global_shot_id",
    )
    if not isinstance(cell, str) or not cell:
        raise PolicyAnalysisError("cell_id must be a nonempty string")
    return cell, worker, shot


def _merge_authenticated_timing(value: Any, timing: Any) -> Any:
    """Reattaches a separately authenticated timing projection in memory."""

    if not isinstance(value, Mapping) or not isinstance(timing, Mapping):
        raise PolicyAnalysisError("captured core timing does not match a ledger object")
    result = dict(value)
    for key, item in timing.items():
        if key in result:
            if not isinstance(result[key], Mapping) or not isinstance(item, Mapping):
                raise PolicyAnalysisError(f"captured timing collides with ledger field {key}")
            result[key] = _merge_authenticated_timing(result[key], item)
        else:
            result[key] = item
    return result


def load_policy_audit(root: Path) -> PolicyAuditCorpus:
    root = Path(root).resolve()
    ready_path = root / "COLLECTION_READY"
    if not ready_path.is_file():
        raise PolicyAnalysisError("audit root has no COLLECTION_READY marker")
    experiment = _load_json(root / "experiment.json")
    config = _load_json(root / "config.json")
    manifest = _load_json(root / "manifest.json")
    ready = _load_json(ready_path)
    experiment_id = _at(experiment, "experiment_id", required=True)
    if _at(config, "experiment_id", required=True) != experiment_id:
        raise PolicyAnalysisError("experiment/config identities differ")
    if ready.get("schema") != "promatch-l1-policy-audit-collection-ready-v1":
        raise PolicyAnalysisError("COLLECTION_READY has the wrong schema")
    if ready.get("experiment_id") != experiment_id:
        raise PolicyAnalysisError("COLLECTION_READY has the wrong experiment identity")
    if ready.get("manifest_sha256") != _sha256((root / "manifest.json").read_bytes()):
        raise PolicyAnalysisError("COLLECTION_READY manifest digest mismatch")
    if manifest.get("experiment_id") != experiment_id:
        raise PolicyAnalysisError("manifest has the wrong experiment identity")

    worker_dirs = sorted((root / "shards").glob("worker-*"))
    if not worker_dirs:
        raise PolicyAnalysisError("audit root contains no worker shards")
    rows: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SHARD_FILES}
    source_hashes = {
        name: _sha256((root / name).read_bytes())
        for name in ("experiment.json", "config.json", "manifest.json", "COLLECTION_READY")
    }
    for worker_dir in worker_dirs:
        try:
            worker_id = int(worker_dir.name.removeprefix("worker-"))
        except ValueError as ex:
            raise PolicyAnalysisError(f"invalid worker directory {worker_dir.name}") from ex
        worker_rows: dict[str, list[dict[str, Any]]] = {}
        for kind, filename in SHARD_FILES.items():
            path = worker_dir / filename
            if not path.is_file():
                raise PolicyAnalysisError(f"missing worker artifact {path}")
            source_hashes[path.relative_to(root).as_posix()] = _sha256(path.read_bytes())
            loaded = _load_jsonl_gzip(path, manifest=manifest, root=root)
            for row in loaded:
                if _at(row, "experiment_id", required=True) != experiment_id:
                    raise PolicyAnalysisError(f"{kind} row has wrong experiment_id")
                _, row_worker, _ = _identity(row)
                if row_worker != worker_id:
                    raise PolicyAnalysisError(f"{kind} row is in the wrong worker shard")
                schema = _at(row, "schema", required=True)
                if schema != EXPECTED_LEDGER_SCHEMAS[kind]:
                    raise PolicyAnalysisError(f"{kind} row has invalid schema")
            worker_rows[kind] = loaded

        timing_path = worker_dir / "timing.json"
        if not timing_path.is_file():
            raise PolicyAnalysisError(f"missing authenticated worker timing {timing_path}")
        timing = _load_json(timing_path)
        manifest_shards = manifest.get("shards")
        matches = [
            shard for shard in manifest_shards
            if isinstance(shard, Mapping)
            and isinstance(shard.get("worker"), Mapping)
            and shard["worker"].get("worker_id") == worker_id
        ] if isinstance(manifest_shards, list) else []
        relative_timing = timing_path.relative_to(root).as_posix()
        if (
            len(matches) != 1
            or matches[0].get("timing_path") != relative_timing
            or matches[0].get("timing_sha256") != _sha256(timing_path.read_bytes())
            or matches[0].get("nondeterministic_telemetry_paths") != [relative_timing]
            or timing.get("schema") != "promatch-l1-policy-audit-timing-v2"
            or timing.get("worker_id") != worker_id
            or timing.get("scientifically_deterministic") is not False
            or timing.get("excluded_from_bit_exact_ledger_contract") is not True
        ):
            raise PolicyAnalysisError("worker timing is not authenticated by its manifest shard")
        captured = timing.get("core_timing_by_ledger")
        if not isinstance(captured, Mapping) or set(captured) != set(SHARD_FILES):
            raise PolicyAnalysisError("worker timing lacks the exact core ledger projection")
        for kind, records in captured.items():
            if not isinstance(records, list):
                raise PolicyAnalysisError("worker core timing projection must be an array")
            seen: set[int] = set()
            for record in records:
                index = record.get("row_index") if isinstance(record, Mapping) else None
                if (
                    not isinstance(record, Mapping)
                    or set(record) != {
                        "row_index", "worker_shot_index", "global_shot_id", "timing"
                    }
                    or isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(worker_rows[kind])
                    or index in seen
                    or record.get("worker_shot_index")
                    != worker_rows[kind][index].get("worker_shot_index")
                    or record.get("global_shot_id")
                    != worker_rows[kind][index].get("global_shot_id")
                ):
                    raise PolicyAnalysisError("worker core timing projection identity is invalid")
                worker_rows[kind][index] = _merge_authenticated_timing(
                    worker_rows[kind][index], record.get("timing")
                )
                seen.add(index)
        source_hashes[relative_timing] = _sha256(timing_path.read_bytes())
        for kind in SHARD_FILES:
            rows[kind].extend(worker_rows[kind])

    shots = rows["shots"]
    shot_ids = [_identity(row) for row in shots]
    if len(shot_ids) != len(set(shot_ids)):
        raise PolicyAnalysisError("physical shot rows are not unique")
    known = set(shot_ids)
    shot_provenance = {
        _identity(row): (
            _as_nonnegative_int(
                _at(row, "worker_shot_index", required=True), name="worker_shot_index"
            ),
            _at(row, "physical_input_sha256", required=True),
        )
        for row in shots
    }
    for kind in ("proposals", "counterfactuals", "domains"):
        unknown = sorted({_identity(row) for row in rows[kind]} - known)
        if unknown:
            raise PolicyAnalysisError(f"{kind} rows reference unknown physical shots")
        for row in rows[kind]:
            provenance = (
                _as_nonnegative_int(
                    _at(row, "worker_shot_index", required=True), name="worker_shot_index"
                ),
                _at(row, "physical_input_sha256", required=True),
            )
            if provenance != shot_provenance[_identity(row)]:
                raise PolicyAnalysisError(f"{kind} row has inconsistent shot provenance")

    configured_scientific_shots = _one_deep(config, ("total_shots", "shot_count"))
    if configured_scientific_shots is not None and _as_nonnegative_int(
        configured_scientific_shots, name="configured scientific shot count"
    ) != 20_000:
        raise PolicyAnalysisError("B1 protocol must retain its 20,000-shot scientific schedule")
    mode = manifest.get("mode")
    mode_counts = {"scientific": 20_000, "probe": 100, "smoke": 32}
    if mode not in mode_counts:
        raise PolicyAnalysisError(f"invalid collection mode {mode!r}")
    expected_shots = mode_counts[str(mode)]
    if manifest.get("workers") != 32 or manifest.get("shots") != expected_shots:
        raise PolicyAnalysisError("manifest campaign schedule differs from its frozen mode")
    expected_tail_attestation = {
        "uncapped_counterfactuals": True,
        "censored_states": 0,
        "repeated_same_state_proposal_signatures": 0,
        "worker_timeouts": 0,
        "output_truncations": 0,
    }
    if manifest.get("tail_censor_attestation") != expected_tail_attestation:
        raise PolicyAnalysisError("manifest lacks the exact clean tail/censor attestation")
    if len(worker_dirs) != 32 or len(shots) != expected_shots:
        raise PolicyAnalysisError("ledger campaign schedule differs from its manifest")
    if mode == "scientific":
        schedule = [(worker, 625 * worker, 625) for worker in range(32)]
    elif mode == "probe":
        schedule = [
            (worker, 4 * worker if worker < 4 else 16 + 3 * (worker - 4), 4 if worker < 4 else 3)
            for worker in range(32)
        ]
    else:
        schedule = [(worker, worker, 1) for worker in range(32)]
    counts = Counter(worker for _, worker, _ in shot_ids)
    for worker, start, count in schedule:
        if counts[worker] != count:
            raise PolicyAnalysisError("worker shot count differs from the frozen mode schedule")
        expected = set(range(start, start + count))
        actual = {shot for _, w, shot in shot_ids if w == worker}
        if actual != expected:
            raise PolicyAnalysisError("worker global-shot range is not canonical")
        indices = {
            _as_nonnegative_int(
                _at(row, "worker_shot_index", required=True), name="worker_shot_index"
            )
            for row in shots
            if _identity(row)[1] == worker
        }
        if indices != set(range(count)):
            raise PolicyAnalysisError("worker-local shot indices are not canonical")

    return PolicyAuditCorpus(
        root=root,
        experiment=experiment,
        config=config,
        manifest=manifest,
        rows={kind: tuple(value) for kind, value in rows.items()},
        source_hashes=dict(sorted(source_hashes.items())),
    )


def _arm_role(arm_id: str) -> str:
    value = arm_id.lower()
    if "u0-joint" in value or value in {"u0", "u0-joint-y2"}:
        return "u0"
    if "shadow" in value or "legacy" in value or "v3" in value:
        return "shadow"
    if "ocost" in value or "o-cost" in value:
        return "o-cost-tx"
    if ("oframe" in value or "o-frame" in value) and "partial" in value:
        return "o-frame-partial"
    if "oframe" in value or "o-frame" in value:
        return "o-frame-tx"
    return arm_id


def _arm_results(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    predictions = row.get("arm_predictions_hex")
    failures = row.get("arm_failures")
    if isinstance(predictions, Mapping) and isinstance(failures, Mapping):
        if set(predictions) != set(failures):
            raise PolicyAnalysisError("shot prediction/failure arm sets differ")
        summaries: Mapping[str, Any] = {}
        for key in ("arm_summaries", "arm_workload", "arm_results", "arms", "trajectories"):
            candidate = row.get(key)
            if isinstance(candidate, Mapping):
                summaries = candidate
                break
        result: dict[str, Mapping[str, Any]] = {}
        for arm_id in sorted(predictions):
            role = _arm_role(str(arm_id))
            if role in result:
                raise PolicyAnalysisError(f"shot contains duplicate semantic arm {role!r}")
            summary = summaries.get(arm_id, summaries.get(role, {}))
            if not isinstance(summary, Mapping):
                raise PolicyAnalysisError(f"arm summary for {arm_id!r} must be an object")
            result[role] = {
                **summary,
                "prediction_hex": predictions[arm_id],
                "logical_failure": failures[arm_id],
                "source_arm_id": arm_id,
            }
        return result
    for key in ("arms", "arm_results", "decoder_results", "results", "trajectories"):
        raw = row.get(key)
        if isinstance(raw, Mapping) and all(isinstance(value, Mapping) for value in raw.values()):
            result: dict[str, Mapping[str, Any]] = {}
            for arm_id, value in raw.items():
                role = _arm_role(str(arm_id))
                if role in result:
                    raise PolicyAnalysisError(f"shot contains duplicate semantic arm {role!r}")
                result[role] = value
            return result
    arm_id = row.get("arm_id")
    if isinstance(arm_id, str):
        return {_arm_role(arm_id): row}
    raise PolicyAnalysisError("shot row has no arm result mapping")


def _failure(result: Mapping[str, Any]) -> bool:
    value = _at(
        result,
        "logical_failure",
        "failed",
        "any_observable_failure",
        "posthoc.logical_failure",
        "posthoc_ground_truth.logical_failure",
    )
    return _as_bool(value, name="logical_failure")


def _prediction_token(result: Mapping[str, Any]) -> str | None:
    value = _at(result, "prediction_hex", "prediction", "logical_prediction_hex")
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyAnalysisError("prediction token must be a string")
    return value


def _original_hw(row: Mapping[str, Any]) -> int:
    return _as_nonnegative_int(
        _at(
            row,
            "original_detector_hw",
            "original_detector_events",
            "global_detector_hw",
            "detector_hw",
            "workload.original_detector_events",
            required=True,
        ),
        name="original detector HW",
    )


def _sorted_labels(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PolicyAnalysisError(f"{name} must be an array of strings")
    normalized = tuple(sorted(set(value)))
    if list(normalized) != value:
        raise PolicyAnalysisError(f"{name} must be sorted and unique")
    return normalized


def _component_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    def canonical_edge_ids(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(type(edge_id) is int and edge_id >= 0 for edge_id in value)
            and value == sorted(set(value))
        )

    direct_labels = _sorted_labels(
        _at(row, "support_difference_component_labels", required=True),
        name="support_difference_component_labels",
    )
    components = _at(row, "support_difference_components", required=True)
    if not isinstance(components, list):
        raise PolicyAnalysisError("support_difference_components must be an array")
    if _at(row, "support_difference_representation_version", required=True) != (
        "promatch-support-difference-v2"
    ):
        raise PolicyAnalysisError("unsupported support-difference representation")
    labels: set[str] = set()
    if not canonical_edge_ids(_at(row, "detector_boundary_ids", required=True)):
        raise PolicyAnalysisError("detector_boundary_ids is not a canonical detector set")
    real_edge_union: set[int] = set()
    cancellation_union: set[int] = set()
    saw_disconnected = False
    real_components: list[Mapping[str, Any]] = []
    cancellation_components: list[Mapping[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping):
            raise PolicyAnalysisError("support difference component must be an object")
        if set(component) != {
            "certificate_kind", "canonical_edge_ids",
            "support_cancellation_edge_ids", "component_detector_ids",
            "candidate_support_witness_edge_ids",
            "candidate_boundary_witness_detector_ids",
            "labels", "candidate_relevant",
            "candidate_relevance_reasons",
        }:
            raise PolicyAnalysisError("support difference component fields are not exact")
        tags = _at(component, "labels", "tags", required=True)
        component_labels = _sorted_labels(tags, name="support difference component tags")
        if set(component_labels) - CONTEXT_LABELS:
            raise PolicyAnalysisError("support difference component contains unknown labels")
        if "in-domain" in component_labels and len(component_labels) != 1:
            raise PolicyAnalysisError("in-domain is not exclusive in support component labels")
        kind = _at(component, "certificate_kind", required=True)
        relevant = _as_bool(
            _at(component, "candidate_relevant", required=True),
            name="support component candidate_relevant",
        )
        reasons = _at(component, "candidate_relevance_reasons", required=True)
        if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
            raise PolicyAnalysisError("support component relevance reasons are not canonical")
        if set(reasons) - {
            "candidate-support-edge",
            "candidate-boundary-detector",
            "candidate-residual-support-cancellation",
        }:
            raise PolicyAnalysisError("support component has an unknown relevance reason")
        edge_ids = _at(component, "canonical_edge_ids", required=True)
        cancellation_ids = _at(component, "support_cancellation_edge_ids", required=True)
        detector_ids = _at(component, "component_detector_ids", required=True)
        support_witness = _at(component, "candidate_support_witness_edge_ids", required=True)
        boundary_witness = _at(
            component, "candidate_boundary_witness_detector_ids", required=True
        )
        if not all(canonical_edge_ids(value) for value in (
            edge_ids, cancellation_ids, detector_ids, support_witness, boundary_witness
        )):
            raise PolicyAnalysisError("support component edge IDs are not canonical sets")
        if kind == "real-x-component":
            candidate_support = _at(row, "P_candidate_support_edge_ids", required=True)
            detector_boundary = _at(row, "detector_boundary_ids", required=True)
            expected_support_witness = sorted(set(edge_ids).intersection(candidate_support))
            expected_boundary_witness = sorted(set(detector_ids).intersection(detector_boundary))
            expected_reasons = sorted(
                (["candidate-support-edge"] if expected_support_witness else [])
                + (["candidate-boundary-detector"] if expected_boundary_witness else [])
            )
            if not edge_ids or cancellation_ids or "support-cancellation" in component_labels:
                raise PolicyAnalysisError("real X component has invalid certificate support")
            if (
                support_witness != expected_support_witness
                or boundary_witness != expected_boundary_witness
                or reasons != expected_reasons
                or bool(reasons) != relevant
            ):
                raise PolicyAnalysisError("real X component relevance disagrees with its reasons")
            if real_edge_union.intersection(edge_ids):
                raise PolicyAnalysisError("real X components overlap")
            real_edge_union.update(edge_ids)
            saw_disconnected |= not relevant
            real_components.append(component)
        elif kind == "support-cancellation":
            if (
                edge_ids or not cancellation_ids or detector_ids or support_witness
                or boundary_witness or not relevant or reasons != [
                    "candidate-residual-support-cancellation"
                ] or "support-cancellation" not in component_labels
            ):
                raise PolicyAnalysisError("support-cancellation certificate is malformed")
            if cancellation_union:
                raise PolicyAnalysisError("multiple support-cancellation certificates")
            cancellation_union.update(cancellation_ids)
            cancellation_components.append(component)
        else:
            raise PolicyAnalysisError(f"unknown support certificate kind {kind!r}")
        if relevant:
            labels.update(component_labels)
    if components != sorted(
        real_components, key=lambda component: component["canonical_edge_ids"]
    ) + cancellation_components:
        raise PolicyAnalysisError("support certificates are not in canonical order")
    supports = {}
    for field in (
        "B_base_support_edge_ids", "P_candidate_support_edge_ids",
        "R_residual_support_edge_ids", "Q_forced_parity_support_edge_ids",
        "X_support_difference_edge_ids", "P_intersection_R_edge_ids",
    ):
        value = _at(row, field, required=True)
        if not canonical_edge_ids(value):
            raise PolicyAnalysisError(f"{field} is not a canonical square-free support")
        supports[field] = value
    b = set(supports["B_base_support_edge_ids"])
    p = set(supports["P_candidate_support_edge_ids"])
    r = set(supports["R_residual_support_edge_ids"])
    for alias, canonical in (
        ("base_support_edge_ids", supports["B_base_support_edge_ids"]),
        ("candidate_support_edge_ids", supports["P_candidate_support_edge_ids"]),
        ("residual_support_edge_ids", supports["R_residual_support_edge_ids"]),
    ):
        if _at(row, alias, required=True) != canonical:
            raise PolicyAnalysisError(f"{alias} disagrees with its named B/P/R support")
    if supports["Q_forced_parity_support_edge_ids"] != sorted(p ^ r):
        raise PolicyAnalysisError("Q support does not reconcile P xor R")
    if supports["X_support_difference_edge_ids"] != sorted(b ^ p ^ r):
        raise PolicyAnalysisError("X support does not reconcile B xor P xor R")
    if supports["P_intersection_R_edge_ids"] != sorted(p & r):
        raise PolicyAnalysisError("cancellation support does not reconcile P intersection R")
    for flag in (
        "supports_square_free", "B_base_support_square_free",
        "P_candidate_support_square_free", "R_residual_support_square_free",
        "Q_forced_parity_support_square_free", "X_support_difference_square_free",
    ):
        if _at(row, flag, required=True) is not True:
            raise PolicyAnalysisError(f"{flag} must be true")
    x_support = supports["X_support_difference_edge_ids"]
    cancellation_support = supports["P_intersection_R_edge_ids"]
    top_cancellation = _at(row, "support_cancellation_edge_ids", required=True)
    if top_cancellation != cancellation_support:
        raise PolicyAnalysisError("top-level cancellation support is inconsistent")
    if sorted(real_edge_union) != x_support:
        raise PolicyAnalysisError("real support components do not partition X exactly")
    if sorted(cancellation_union) != cancellation_support:
        raise PolicyAnalysisError("cancellation certificates do not reconcile P intersection R")
    disconnected_flag = _as_bool(
        _at(row, "disconnected_support_reconfiguration", required=True),
        name="disconnected_support_reconfiguration",
    )
    if disconnected_flag != saw_disconnected:
        raise PolicyAnalysisError("disconnected support-reconfiguration flag is inconsistent")
    if labels - {"in-domain"}:
        labels.discard("in-domain")
    component_labels = tuple(sorted(labels))
    if direct_labels != component_labels:
        raise PolicyAnalysisError(
            "support-difference component labels disagree with their component union"
        )
    return component_labels


def _validate_context_labels(labels: tuple[str, ...], *, name: str) -> None:
    unknown = set(labels) - CONTEXT_LABELS
    if unknown:
        raise PolicyAnalysisError(f"{name} contains unknown labels: {sorted(unknown)}")
    if "in-domain" in labels and len(labels) != 1:
        raise PolicyAnalysisError(f"in-domain is not exclusive in {name}")


def _context(row: Mapping[str, Any]) -> dict[str, Any]:
    matched_raw = _at(
        row,
        "matched_partner_labels",
        "base_matched_partner_labels",
        "context.matched_partner_labels",
        required=True,
    )
    matched = _sorted_labels(
        matched_raw,
        name="matched_partner_labels",
    )
    support_path_raw = _at(
        row,
        "support_path_labels",
        "base_support_path_labels",
        "context.support_path_labels",
        required=True,
    )
    support_path = _sorted_labels(
        support_path_raw,
        name="support_path_labels",
    )
    difference = _component_labels(row)
    for name, labels in (
        ("matched_partner_labels", matched),
        ("support_path_labels", support_path),
        ("support_difference_component_labels", difference),
    ):
        _validate_context_labels(labels, name=name)
    omitted = _sorted_labels(
        _at(row, "omitted_context_labels", "context.omitted_context_labels", required=True),
        name="omitted_context_labels",
    )
    _validate_context_labels(omitted, name="omitted_context_labels")
    expected_omitted_set = set(matched) | set(support_path)
    if expected_omitted_set - {"in-domain"}:
        expected_omitted_set.discard("in-domain")
    expected_omitted = tuple(sorted(expected_omitted_set))
    if omitted != expected_omitted:
        raise PolicyAnalysisError(
            "omitted_context_labels disagrees with matched/support-path union"
        )
    degeneracy = _sorted_labels(
        _at(row, "degeneracy_diagnostics", "context.degeneracy_diagnostics", required=True),
        name="degeneracy_diagnostics",
    )
    unknown_degeneracy = set(degeneracy) - DEGENERACY_DIAGNOSTICS
    if unknown_degeneracy:
        raise PolicyAnalysisError(
            f"unknown degeneracy diagnostics: {sorted(unknown_degeneracy)}"
        )
    diagnostic_flags = {
        "same-pair-different-path-or-frame": _as_bool(
            _at(row, "same_pair_different_path_or_frame", required=True),
            name="same_pair_different_path_or_frame",
        ),
        "equal-weight-logical-class": _as_bool(
            _at(row, "equal_weight_logical_class", required=True),
            name="equal_weight_logical_class",
        ),
        "disconnected-support-reconfiguration": _as_bool(
            _at(row, "disconnected_support_reconfiguration", required=True),
            name="disconnected_support_reconfiguration",
        ),
        "unclassified": _as_bool(
            _at(row, "degeneracy_unclassified", required=True),
            name="degeneracy_unclassified",
        ),
    }
    expected_from_flags = tuple(
        sorted(label for label, present in diagnostic_flags.items() if present)
    )
    if degeneracy != expected_from_flags:
        raise PolicyAnalysisError(
            "degeneracy_diagnostics disagrees with its explicit diagnostic flags"
        )
    endpoints = _at(row, "ordered_endpoints", required=True)
    pairs = _at(row, "base_matched_active_pairs", required=True)
    base_support = _at(row, "base_support_edge_ids", required=True)
    candidate_support = _at(row, "candidate_support_edge_ids", required=True)
    base_frame = _at(row, "base_frame", required=True)
    candidate_frame = _at(row, "candidate_frame", required=True)
    if (
        not isinstance(endpoints, list)
        or not isinstance(pairs, list)
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
        or not isinstance(base_support, list)
        or not isinstance(candidate_support, list)
        or not isinstance(base_frame, str)
        or not isinstance(candidate_frame, str)
    ):
        raise PolicyAnalysisError("degeneracy source fields have invalid types")
    same_pair = (
        len(endpoints) == 2
        and endpoints[0] is not None
        and any(set(pair) == set(endpoints) for pair in pairs)
    )
    expected_same_pair_diagnostic = same_pair and (
        set(base_support) != set(candidate_support) or base_frame != candidate_frame
    )
    if (
        diagnostic_flags["same-pair-different-path-or-frame"]
        != expected_same_pair_diagnostic
    ):
        raise PolicyAnalysisError(
            "same-pair-different-path-or-frame is not structurally reconciled"
        )
    expected_equal_weight = _cost_compatible(row) and not _frame_compatible(row)
    if diagnostic_flags["equal-weight-logical-class"] != expected_equal_weight:
        raise PolicyAnalysisError(
            "equal-weight-logical-class disagrees with the oracle certificate"
        )
    oracle_accepts = _as_bool(
        _at(row, "oracle_policy_accepts", required=True), name="oracle_policy_accepts"
    )
    if oracle_accepts != (_cost_compatible(row) and _frame_compatible(row)):
        raise PolicyAnalysisError("oracle_policy_accepts disagrees with cost/frame compatibility")
    expected_unclassified = (
        not oracle_accepts
        and not diagnostic_flags["same-pair-different-path-or-frame"]
        and not diagnostic_flags["equal-weight-logical-class"]
        and not diagnostic_flags["disconnected-support-reconfiguration"]
        and not difference
    )
    if diagnostic_flags["unclassified"] != expected_unclassified:
        raise PolicyAnalysisError("unclassified degeneracy residual is not reconciled")
    exclusive = _at(row, "exclusive_support_component_context", "exclusive_context_label", "context.exclusive")
    if exclusive is None:
        exclusive = next((label for label in CONTEXT_PRIORITY if label in difference), None)
    if exclusive is not None and exclusive not in CONTEXT_PRIORITY:
        raise PolicyAnalysisError(f"unknown exclusive context label {exclusive!r}")
    if exclusive != next((label for label in CONTEXT_PRIORITY if label in difference), None):
        raise PolicyAnalysisError("exclusive context label violates frozen display priority")
    # These views are deliberately not unioned or aliased.  A downstream table
    # can compare them, but each keeps its own denominator and label set.
    return {
        "matched_partner_labels": matched,
        "support_path_labels": support_path,
        "support_difference_component_labels": difference,
        "exclusive_support_component_context": exclusive,
        "omitted_context_labels": omitted,
        "degeneracy_diagnostics": degeneracy,
    }


def _cost_compatible(row: Mapping[str, Any]) -> bool:
    explicit = _at(row, "cost_compatible", "evaluation.cost_compatible", "oracle.cost_compatible")
    if explicit is not None:
        return _as_bool(explicit, name="cost_compatible")
    classification = _at(
        row,
        "cost_classification",
        "evaluation.cost_classification",
        "oracle.cost_classification",
        required=True,
    )
    if classification in {"numerically-cost-compatible", "cost-compatible", "compatible"}:
        return True
    if classification in {"positive-cost-excess", "positive-excess"}:
        return False
    raise PolicyAnalysisError(f"invalid/fatal cost classification {classification!r}")


def _frame_compatible(row: Mapping[str, Any]) -> bool:
    return _as_bool(
        _at(row, "frame_compatible", "evaluation.frame_compatible", "oracle.frame_compatible", required=True),
        name="frame_compatible",
    )


def certificate_class(row: Mapping[str, Any]) -> str:
    cost = _cost_compatible(row)
    frame = _frame_compatible(row)
    excess = _required_float(row, "cost_excess")
    tolerance = _required_float(row, "tau_k")
    if tolerance < 0:
        raise PolicyAnalysisError("tau_k must be nonnegative")
    if cost != (excess <= tolerance):
        raise PolicyAnalysisError("cost classification disagrees with cost_excess/tau_k")
    if excess < -tolerance:
        raise PolicyAnalysisError("negative cost excess exceeds the frozen tolerance")
    if not cost:
        return "positive-cost-excess"
    if not frame:
        return "cost-compatible-frame-conflict"
    return "O-frame-safe"


def _origin(row: Mapping[str, Any]) -> str:
    value = _at(row, "trajectory_origin", required=True)
    if not isinstance(value, str):
        raise PolicyAnalysisError("trajectory_origin must be a string")
    return value


def _stage(row: Mapping[str, Any]) -> int:
    stage = _as_nonnegative_int(_at(row, "stage", "proposal.stage", required=True), name="stage")
    if stage not in {1, 2, 3, 4}:
        raise PolicyAnalysisError("ProMatch stage must be in 1..4")
    return stage


def _domain_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _at(row, "patch_id", "domain.patch_id"),
        _at(row, "basis", "domain.basis", "domain.check_basis"),
        _at(row, "window_id", "domain.window_id"),
    )


def _state_key(row: Mapping[str, Any]) -> str:
    explicit = _at(
        row,
        "original_state_sha256",
        "counterfactual_state_sha256",
        "complete_pre_state_fingerprint",
        "pre_state_fingerprint",
    )
    if isinstance(explicit, str) and explicit:
        base = explicit
    else:
        raise PolicyAnalysisError("counterfactual/proposal row lacks state identity")
    cell, worker, shot = _identity(row)
    return _sha256(canonical_json_bytes([cell, worker, shot, _domain_key(row), base]))


def _terminal_action(row: Mapping[str, Any]) -> str | None:
    value = _at(row, "terminal_action", "counterfactual.terminal_action")
    if value is None:
        return None
    if value not in TERMINAL_ACTIONS:
        raise PolicyAnalysisError(f"invalid terminal action {value!r}")
    return str(value)


def _counterfactual_states(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        origin = _origin(row)
        if origin == "casebook-exhaustive":
            continue
        if origin != "shadow-original-state-counterfactual":
            raise PolicyAnalysisError(f"invalid all-shot counterfactual origin {origin!r}")
        grouped[_state_key(row)].append(row)
    states = []
    for state_id, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: _as_nonnegative_int(
                _at(row, "operational_veto_chain_rank", "counterfactual_rank", required=True),
                name="operational_veto_chain_rank",
            ),
        )
        ranks = [
            _as_nonnegative_int(
                _at(row, "operational_veto_chain_rank", "counterfactual_rank", required=True),
                name="operational_veto_chain_rank",
            )
            for row in ordered
        ]
        if ranks != list(range(1, len(ranks) + 1)):
            raise PolicyAnalysisError("counterfactual ranks are not contiguous from one")
        terminal_values = [_terminal_action(row) for row in ordered]
        if any(value is not None for value in terminal_values[:-1]) or terminal_values[-1] is None:
            raise PolicyAnalysisError("counterfactual terminal action must occur only on the final row")
        action = terminal_values[-1]
        assert action is not None
        stages = [_stage(row) for row in ordered]
        if stages != sorted(stages):
            raise PolicyAnalysisError("unchanged-state counterfactual moved to an earlier stage")
        signatures = []
        proposal_digests = []
        for row in ordered:
            signature = _at(row, "proposal_signature", required=True)
            if not isinstance(signature, list):
                raise PolicyAnalysisError("counterfactual proposal_signature must be an array")
            expected_signature = [
                _stage(row),
                _at(row, "ordered_endpoints", required=True),
                _at(row, "canonical_edge_ids", required=True),
            ]
            if signature != expected_signature:
                raise PolicyAnalysisError(
                    "counterfactual proposal_signature disagrees with its proposal fields"
                )
            signatures.append(canonical_json_bytes(signature))
            proposal_digests.append(_proposal_sha(row))
            if "censored" not in row or _as_bool(row["censored"], name="censored"):
                raise PolicyAnalysisError("all-shot counterfactual row is censored or unmarked")
            if "veto_budget" not in row or row["veto_budget"] is not None:
                raise PolicyAnalysisError("all-shot counterfactual is not explicitly uncapped")
        if len(set(signatures)) != len(signatures):
            raise PolicyAnalysisError("counterfactual state repeats a proposal signature")
        if len(set(proposal_digests)) != len(proposal_digests):
            raise PolicyAnalysisError("counterfactual state repeats a proposal digest")
        row_contexts = [_context(row) for row in ordered]
        original = ordered[0]
        original_reference = _at(original, "original_proposal_sha256", required=True)
        if not isinstance(original_reference, str) or original_reference != _proposal_sha(original):
            raise PolicyAnalysisError("rank-one counterfactual is not the referenced original proposal")
        for field in (
            "original_proposal_sha256",
            "original_state_sha256",
            "complete_pre_state_fingerprint",
            "local_active_state_fingerprint",
        ):
            expected = _at(original, field, required=True)
            if any(_at(row, field, required=True) != expected for row in ordered[1:]):
                raise PolicyAnalysisError(
                    f"counterfactual chain changed its original-state field {field!r}"
                )
        original_class = certificate_class(original)
        if original_class == "O-frame-safe":
            # Safe states may carry only cheap competitor inspection and are not
            # part of the unsafe-state counterfactual endpoint.
            continue
        safe_rows = [row for row in ordered[1:] if certificate_class(row) == "O-frame-safe"]
        first_safe = safe_rows[0] if safe_rows else None
        expected_first_safe_rank = None if first_safe is None else ordered.index(first_safe) + 1
        expected_first_safe_digest = None if first_safe is None else _proposal_sha(first_safe)
        for row in ordered:
            recorded_rank = _at(row, "first_safe_rank", required=True)
            if recorded_rank is not None:
                recorded_rank = _as_nonnegative_int(recorded_rank, name="first_safe_rank")
            recorded_digest = _at(
                row, "first_safe_alternative_proposal_sha256", required=True
            )
            if recorded_rank != expected_first_safe_rank or recorded_digest != expected_first_safe_digest:
                raise PolicyAnalysisError("counterfactual first-safe references do not reconcile")
        decisions = [_at(row, "decision", required=True) for row in ordered]
        if any(not isinstance(value, str) for value in decisions):
            raise PolicyAnalysisError("counterfactual decisions must be strings")
        for row, decision in zip(ordered, decisions):
            expected_decision = (
                "inspect-only" if certificate_class(row) == "O-frame-safe" else "veto"
            )
            if decision != expected_decision:
                raise PolicyAnalysisError("counterfactual decision disagrees with its certificate")
        if action in {"same-stage-alternative", "later-stage-alternative"}:
            if first_safe is None:
                raise PolicyAnalysisError("alternative action has no safe proposal")
            if ordered.index(first_safe) != len(ordered) - 1:
                raise PolicyAnalysisError("all-shot counterfactual continued after first safe proposal")
            expected = (
                "same-stage-alternative"
                if _stage(first_safe) == _stage(original)
                else "later-stage-alternative"
            )
            if _stage(first_safe) < _stage(original) or action != expected:
                raise PolicyAnalysisError("counterfactual terminal action/stage is inconsistent")
            if any(_at(row, "exhaustion_kind", required=True) is not None for row in ordered):
                raise PolicyAnalysisError("safe-alternative chain claims proposal exhaustion")
        elif action == "abstain-true-exhaustion":
            if first_safe is not None:
                raise PolicyAnalysisError("abstention state contains a safe proposal")
            exhaustion = [_at(row, "exhaustion_kind", required=True) for row in ordered]
            if any(value is not None for value in exhaustion[:-1]) or exhaustion[-1] != "proposal":
                raise PolicyAnalysisError("abstention is not final true proposal exhaustion")
        else:
            raise PolicyAnalysisError("uncapped all-shot chain cannot terminate as censored-invalid")
        state_oracle_calls = [
            _as_nonnegative_int(
                _at(row, "state_oracle_call_count", required=True),
                name="state_oracle_call_count",
            )
            for row in ordered
        ]
        if len(set(state_oracle_calls)) != 1 or state_oracle_calls[0] != len(ordered):
            raise PolicyAnalysisError("counterfactual state oracle-call count does not reconcile")
        context = row_contexts[0]
        candidate_count = _at(original, "state_total_candidate_count")
        if candidate_count is not None:
            candidate_count = _as_nonnegative_int(
                candidate_count, name="state_total_candidate_count"
            )
        original_events = _at(original, "events_removed_if_committed")
        if original_events is not None:
            original_events = _as_nonnegative_int(
                original_events, name="events_removed_if_committed"
            )
        first_safe_events = (
            None if first_safe is None else _at(first_safe, "events_removed_if_committed")
        )
        if first_safe_events is not None:
            first_safe_events = _as_nonnegative_int(
                first_safe_events, name="first-safe events_removed_if_committed"
            )
        states.append(
            {
                "state_id": state_id,
                "cell_id": _identity(original)[0],
                "global_shot_id": _identity(original)[2],
                "original_proposal_sha256": _proposal_sha(original),
                "trajectory_commit_index": _at(
                    original,
                    "trajectory_commit_index",
                    "commitment_index",
                    "accepted_prefix_length",
                ),
                "original_stage": _stage(original),
                "original_certificate_class": original_class,
                "original_cost_excess": _required_float(original, "cost_excess"),
                "exclusive_context": context["exclusive_support_component_context"],
                "terminal_action": action,
                "first_safe_rank": expected_first_safe_rank,
                "first_safe_stage": None if first_safe is None else _stage(first_safe),
                "veto_chain_length": len(ordered) if first_safe is None else len(ordered) - 1,
                "candidate_count": candidate_count,
                "original_decision_weight": _required_float(original, "decision_weight"),
                "original_path_length": _as_nonnegative_int(
                    _at(original, "canonical_edge_count", required=True),
                    name="canonical_edge_count",
                ),
                "original_weight_margin": _float_value(original, "absolute_weight_margin"),
                "original_events_removed": original_events,
                "first_safe_decision_weight": None
                if first_safe is None
                else _required_float(first_safe, "decision_weight"),
                "first_safe_path_length": None
                if first_safe is None
                else _as_nonnegative_int(
                    _at(first_safe, "canonical_edge_count", required=True),
                    name="canonical_edge_count",
                ),
                "first_safe_weight_margin": None
                if first_safe is None
                else _float_value(first_safe, "absolute_weight_margin"),
                "first_safe_events_removed": first_safe_events,
                "rows": ordered,
            }
        )
    return states


def _paired_table(baseline: Sequence[bool], treatment: Sequence[bool]) -> dict[str, Any]:
    if len(baseline) != len(treatment):
        raise PolicyAnalysisError("paired outcome vectors differ in length")
    both_correct = regressions = recoveries = both_wrong = 0
    for base_failed, treatment_failed in zip(baseline, treatment):
        if not base_failed and not treatment_failed:
            both_correct += 1
        elif not base_failed and treatment_failed:
            regressions += 1
        elif base_failed and not treatment_failed:
            recoveries += 1
        else:
            both_wrong += 1
    table = PairedContingency(both_correct, regressions, recoveries, both_wrong)
    return {
        "shots": table.shots,
        "both_correct": both_correct,
        "regressions": regressions,
        "recoveries": recoveries,
        "both_wrong": both_wrong,
        "paired_risk_difference": table.delta,
        "tango_upper_one_sided_97_5": tango_paired_risk_difference_upper(table, alpha=0.025),
    }


def _bootstrap_config(
    config: Mapping[str, Any], *, family: Literal["proposal", "workload"]
) -> tuple[int, str]:
    replicates = _one_deep(config, ("bootstrap_replicates", "replicates"))
    if replicates is None:
        replicates = 10_000
    replicates = _as_nonnegative_int(replicates, name="bootstrap replicates")
    if replicates == 0:
        raise PolicyAnalysisError("bootstrap replicates must be positive")
    seed_root = _at(config, f"bootstrap.seed_roots.{family}")
    if seed_root is None:
        names = (
            f"{family}_bootstrap_seed_root",
            "policy_bootstrap_seed_root",
            "bootstrap_seed_root",
        )
        seed_root = _one_deep(config, names)
    if not isinstance(seed_root, str) or not seed_root:
        raise PolicyAnalysisError("config lacks a bootstrap seed root")
    return replicates, seed_root


def _shot_key(row: Mapping[str, Any]) -> tuple[str, int]:
    cell, _, shot = _identity(row)
    return cell, shot


def _proposal_sha(row: Mapping[str, Any]) -> str:
    value = _at(row, "proposal_sha256", required=True)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PolicyAnalysisError("proposal_sha256 must be a 64-character digest")
    return value


def _arm_metric(
    shot: Mapping[str, Any],
    arm: Mapping[str, Any],
    *names: str,
) -> Any:
    value = _at(arm, *names)
    if value is not None:
        return value
    source_arm = arm.get("source_arm_id")
    role = _arm_role(str(source_arm)) if source_arm is not None else None
    for name in names:
        for container_name in (
            f"{name}_by_arm",
            f"arm_{name}",
            "workload_by_arm",
            "transaction_by_arm",
        ):
            container = shot.get(container_name)
            if not isinstance(container, Mapping):
                continue
            candidate = container.get(source_arm, container.get(role))
            if isinstance(candidate, Mapping):
                candidate = _at(candidate, name)
            if candidate is not None:
                return candidate
    return None


def _normalize_shots(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in sorted(rows, key=_identity):
        arms = _arm_results(row)
        required = {"u0", "shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial"}
        if set(arms) != required:
            raise PolicyAnalysisError(
                f"shot arms differ from frozen B1 set: {sorted(arms)}"
            )
        original_hw = _original_hw(row)
        predictions = {name: _prediction_token(value) for name, value in arms.items()}
        differs = {
            name: _at(value, "differs_from_u0", "prediction_differs_from_u0")
            for name, value in arms.items()
        }
        for name in ("o-frame-tx", "o-frame-partial"):
            if predictions["u0"] is not None and predictions[name] is not None:
                equal = predictions[name] == predictions["u0"]
            elif differs[name] is not None:
                equal = not _as_bool(differs[name], name="differs_from_u0")
            else:
                raise PolicyAnalysisError(
                    f"cannot verify exact U0 equality for {name}"
                )
            if not equal:
                raise PolicyAnalysisError(f"fatal {name}/U0 prediction mismatch")
        failures = {name: _failure(value) for name, value in arms.items()}
        if any(
            predictions[name] == predictions["u0"]
            and failures[name] != failures["u0"]
            for name in ("shadow", "o-frame-tx", "o-frame-partial")
        ):
            raise PolicyAnalysisError("equal prediction tokens have inconsistent failure labels")
        normalized.append(
            {
                "cell_id": _identity(row)[0],
                "worker_id": _identity(row)[1],
                "global_shot_id": _identity(row)[2],
                "original_hw": original_hw,
                "zero_event": original_hw == 0,
                "predictions": predictions,
                "failures": failures,
                "final_hw": {
                    name: _as_nonnegative_int(
                        original_hw
                        if name == "u0"
                        else _arm_metric(
                            row,
                            value,
                            "final_residual_detector_hw",
                            "final_residual_detector_events",
                            "residual_detector_hw",
                        ),
                        name=f"{name} final residual detector HW",
                    )
                    for name, value in arms.items()
                },
                "provisional_removed": {
                    name: _arm_metric(
                        row,
                        value,
                        "provisional_events_removed",
                        "workload.provisional_events_removed",
                    )
                    for name, value in arms.items()
                },
                "rollback_lost": {
                    name: _arm_metric(
                        row,
                        value,
                        "events_lost_to_rollback",
                        "workload.events_lost_to_rollback",
                    )
                    for name, value in arms.items()
                },
                "raw": row,
            }
        )
    return normalized


def _normalize_shadow_proposals(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        if _origin(row) not in {"shadow-original", "shadow"}:
            continue
        proposal_sha = _proposal_sha(row)
        identity = (_identity(row)[0], _identity(row)[2], proposal_sha)
        if identity in seen:
            raise PolicyAnalysisError("duplicate shadow proposal digest")
        seen.add(identity)
        context = _context(row)
        durable_raw = _at(row, "durable", "decision.durable", required=True)
        durable = _as_bool(durable_raw, name="durable")
        if not durable:
            raise PolicyAnalysisError("shadow-original proposal is not durable")
        decision = _at(row, "decision", "decision.kind", required=True)
        if isinstance(decision, Mapping):
            decision = decision.get("kind")
        if decision != "shadow-commit":
            raise PolicyAnalysisError("shadow-original proposal is not a shadow commitment")
        sequence = _as_nonnegative_int(
            _at(
                row,
                "trajectory_commit_index",
                "commitment_index",
                "accepted_prefix_length",
                required=True,
            ),
            name="trajectory_commit_index",
        )
        proposal_signature = _at(row, "proposal_signature", required=True)
        expected_signature = [
            _stage(row),
            _at(row, "ordered_endpoints", required=True),
            _at(row, "canonical_edge_ids", required=True),
        ]
        if proposal_signature != expected_signature:
            raise PolicyAnalysisError(
                "shadow-original proposal_signature disagrees with its proposal fields"
            )
        competitor_exists = _as_bool(
            _at(row, "same_stage_competitor_exists", required=True),
            name="same_stage_competitor_exists",
        )
        competitor_weight = _float_value(row, "same_stage_competitor_weight")
        weight_margin = _float_value(row, "absolute_weight_margin")
        relative_margin = _float_value(row, "relative_weight_margin")
        if competitor_exists is False and any(
            value is not None for value in (competitor_weight, weight_margin, relative_margin)
        ):
            raise PolicyAnalysisError("missing competitor is paired with competitor metrics")
        if competitor_exists is True and (competitor_weight is None or weight_margin is None):
            raise PolicyAnalysisError("recorded competitor lacks weight/margin")
        events_removed = _at(row, "events_removed_if_committed")
        if events_removed is not None:
            events_removed = _as_nonnegative_int(
                events_removed, name="events_removed_if_committed"
            )
        result.append(
            {
                "proposal_sha256": proposal_sha,
                "cell_id": _identity(row)[0],
                "global_shot_id": _identity(row)[2],
                "stage": _stage(row),
                "domain": _domain_key(row),
                "certificate_class": certificate_class(row),
                "frame_compatible": _frame_compatible(row),
                "durable": durable,
                "trajectory_commit_index": sequence,
                "cost_excess": _required_float(row, "cost_excess"),
                "tau_k": _required_float(row, "tau_k"),
                "decision_weight": _required_float(row, "decision_weight"),
                "path_length": _as_nonnegative_int(
                    _at(row, "canonical_edge_count", "path_length", required=True),
                    name="canonical_edge_count",
                ),
                "local_weight_margin": weight_margin,
                "relative_weight_margin": relative_margin,
                "same_stage_competitor_exists": competitor_exists,
                "same_stage_competitor_weight": competitor_weight,
                "events_removed": events_removed,
                "window_offset": _at(
                    row,
                    "window_offset",
                    "round_offset_from_window_start",
                    "local.window_offset",
                ),
                "static_boundary_competition": _at(
                    row,
                    "static_boundary_competition",
                    "local.static_boundary_competition",
                ),
                "domain_hw": _at(row, "domain_current_hw", "state.domain_current_hw"),
                "candidate_multiplicity": _at(
                    row,
                    "candidate_multiplicity",
                    "state_total_candidate_count",
                ),
                **context,
                "raw": row,
            }
        )
    return sorted(result, key=lambda item: (item["cell_id"], item["global_shot_id"], item["proposal_sha256"]))


def _domain_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        arm_id = _at(row, "arm_id", required=True)
        if not isinstance(arm_id, str):
            raise PolicyAnalysisError("domain arm_id must be a string")
        role = _arm_role(arm_id)
        cell = _identity(row)[0]
        status = _at(row, "domain_terminal_status", "status", "outcome.status", required=True)
        if status == "partial-exhausted":
            status = "partial-exhaustion"
        if status not in {"below-limit", "success", "rollback", "partial-exhaustion"}:
            raise PolicyAnalysisError(f"invalid domain terminal status {status!r}")
        key = (cell, role)
        counters[key]["domains"] += 1
        counters[key][f"status:{status}"] += 1
        initial_hw = _at(row, "domain_initial_hw", "initial_hw")
        target = _at(row, "residual_hw_target", "residual_hw_limit")
        if initial_hw is not None and target is not None:
            active = int(_as_nonnegative_int(initial_hw, name="domain_initial_hw") > _as_nonnegative_int(target, name="residual_hw_target"))
            counters[key]["activated_domains"] += active
            counters[key]["activation_known_domains"] += 1
        for field in (
            "provisional_events_removed",
            "durable_events_removed",
            "events_lost_to_rollback",
            "accepted_prefix_length",
        ):
            value = _at(row, field, f"outcome.{field}")
            if value is not None:
                counters[key][field] += _as_nonnegative_int(value, name=field)
    result = []
    for (cell, arm), values in sorted(counters.items()):
        row: dict[str, Any] = {"cell_id": cell, "arm": arm, **dict(sorted(values.items()))}
        denominator = values["domains"]
        for status in ("below-limit", "success", "rollback", "partial-exhaustion"):
            count = values[f"status:{status}"]
            row[f"status:{status}"] = count
            row[f"rate:{status}"] = count / denominator
        row["activation_rate"] = (
            values["activated_domains"] / denominator
            if values["activation_known_domains"] == denominator
            else None
        )
        result.append(row)
    return result


def _casebook_rank_digest(state_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", state_id):
        raise PolicyAnalysisError("casebook state identity must be its canonical SHA-256 digest")
    return state_id


def select_casebook(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Selects outcome-blind states using only frozen oracle/policy fields."""

    selected: dict[str, dict[str, Any]] = {}
    strata: dict[tuple[str | None, int], list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        strata[(state.get("exclusive_context"), int(state["original_stage"]))].append(state)
    audit_rows = []
    for (context, stage), group in sorted(strata.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        slot = f"context={context}|stage={stage}"
        if len(group) < CASEBOOK_MIN_STATES:
            audit_rows.append({"slot": slot, "eligible": len(group), "selected_state_id": None})
            continue
        values = [float(state["original_cost_excess"]) for state in group if state.get("original_cost_excess") is not None]
        if len(values) != len(group):
            raise PolicyAnalysisError("casebook cost-excess stratum contains missing values")
        median = empirical_type7(values, 0.5)
        assert median is not None
        chosen = min(
            group,
            key=lambda state: (
                abs(float(state["original_cost_excess"]) - median),
                _casebook_rank_digest(str(state["state_id"])),
            ),
        )
        selected[str(chosen["state_id"])] = {
            "state_id": chosen["state_id"],
            "selection_reasons": [slot],
            "original_proposal_sha256": chosen["original_proposal_sha256"],
        }
        audit_rows.append(
            {
                "slot": slot,
                "eligible": len(group),
                "median_metric": median,
                "selected_state_id": chosen["state_id"],
            }
        )

    for action in TERMINAL_ACTIONS[:3]:
        group = [state for state in states if state.get("terminal_action") == action]
        slot = f"terminal_action={action}"
        if not group:
            audit_rows.append({"slot": slot, "eligible": 0, "selected_state_id": None})
            continue
        median = empirical_type7([float(state["veto_chain_length"]) for state in group], 0.5)
        assert median is not None
        chosen = min(
            group,
            key=lambda state: (
                abs(float(state["veto_chain_length"]) - median),
                _casebook_rank_digest(str(state["state_id"])),
            ),
        )
        state_id = str(chosen["state_id"])
        if state_id in selected:
            selected[state_id]["selection_reasons"].append(slot)
        else:
            selected[state_id] = {
                "state_id": state_id,
                "selection_reasons": [slot],
                "original_proposal_sha256": chosen["original_proposal_sha256"],
            }
        audit_rows.append(
            {
                "slot": slot,
                "eligible": len(group),
                "median_metric": median,
                "selected_state_id": state_id,
            }
        )
    result = {
        "schema": CASEBOOK_SCHEMA,
        "selection_uses_actual_observables_or_correctness": False,
        "min_context_stage_states": CASEBOOK_MIN_STATES,
        "states": [
            {**row, "selection_reasons": sorted(row["selection_reasons"])}
            for _, row in sorted(selected.items())
        ],
        "selection_audit": audit_rows,
    }
    result["selection_sha256"] = _sha256(canonical_json_bytes(result))
    return result


def _bootstrap_fraction_table(
    *,
    shot_keys: Sequence[tuple[str, int]],
    categories: Sequence[str],
    observations: Sequence[tuple[tuple[str, int], str, bool]],
    replicates: int,
    seed_root: str,
    cell_id: str,
    estimand: str,
) -> list[dict[str, Any]]:
    index = {key: offset for offset, key in enumerate(shot_keys)}
    category_index = {value: offset for offset, value in enumerate(categories)}
    matrix = np.zeros((len(shot_keys), 2 * len(categories)), dtype=np.float64)
    for shot_key, category, numerator in observations:
        if shot_key not in index or category not in category_index:
            raise PolicyAnalysisError("bootstrap observation references an unknown key")
        row = index[shot_key]
        column = category_index[category]
        matrix[row, 2 * column + 1] += 1
        matrix[row, 2 * column] += int(numerator)
    intervals = clustered_bootstrap_ratios(
        matrix,
        replicates=replicates,
        seed=derive_bootstrap_seed(seed_root, cell_id=cell_id, estimand=estimand),
    )
    result = []
    for position, category in enumerate(categories):
        numerator = int(matrix[:, 2 * position].sum())
        denominator = int(matrix[:, 2 * position + 1].sum())
        result.append(
            {
                "category": category,
                "numerator": numerator,
                "denominator": denominator,
                "fraction": None if denominator == 0 else numerator / denominator,
                "bootstrap": intervals[position],
                "stratum_status": (
                    "insufficient-for-rule-formulation"
                    if numerator < SPARSE_UNSAFE_STATES
                    else "descriptive-only"
                ),
            }
        )
    return result


def _outcome_name(shot: Mapping[str, Any]) -> str:
    u0 = shot["failures"]["u0"]
    pu = shot["failures"]["shadow"]
    prediction_agreement = shot["predictions"]["u0"] == shot["predictions"]["shadow"]
    if prediction_agreement:
        return "prediction-agreement"
    if not u0 and pu:
        return "regression"
    if u0 and not pu:
        return "recovery"
    if u0 and pu:
        return "prediction-discordant-both-wrong"
    raise PolicyAnalysisError(
        "prediction-discordant arms cannot both be correct against one actual observable"
    )


def _unsafe_bin(count: int) -> str:
    return str(count) if count < 4 else "4+"


def _count_table(
    rows: Iterable[Mapping[str, Any]], *, keys: Sequence[str]
) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter(tuple(row.get(key) for key in keys) for row in rows)
    return [
        {**dict(zip(keys, values)), "count": count}
        for values, count in sorted(counts.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def _sum_optional(values: Iterable[Any], *, name: str) -> int | None:
    materialized = list(values)
    if not materialized or any(value is None for value in materialized):
        return None
    return sum(_as_nonnegative_int(value, name=name) for value in materialized)


def _risk_count_table(
    rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]
) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        counts[key][1] += 1
        counts[key][0] += row["certificate_class"] != "O-frame-safe"
    return [
        {
            **dict(zip(keys, values)),
            "unsafe": count[0],
            "denominator": count[1],
            "unsafe_fraction": count[0] / count[1],
            "stratum_status": (
                "insufficient-for-rule-formulation"
                if count[0] < SPARSE_UNSAFE_STATES
                else "descriptive-only"
            ),
        }
        for values, count in sorted(counts.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def analyze_policy_audit(corpus: PolicyAuditCorpus) -> dict[str, Any]:
    """Validates and analyzes a frozen B1 corpus without reconstructing decoding."""

    shots = _normalize_shots(corpus.rows["shots"])
    cells = sorted({shot["cell_id"] for shot in shots})
    if len(cells) != 1:
        raise PolicyAnalysisError("B1 analysis requires exactly one physical cell")
    cell_id = cells[0]
    shot_keys = [(shot["cell_id"], shot["global_shot_id"]) for shot in shots]
    if len(shot_keys) != len(set(shot_keys)):
        raise PolicyAnalysisError("cell/global shot identities are not unique")

    proposals = [row for row in _normalize_shadow_proposals(corpus.rows["proposals"]) if row["durable"]]
    states = _counterfactual_states(corpus.rows["counterfactuals"])
    if any(state["terminal_action"] == "censored-invalid" for state in states):
        raise PolicyAnalysisError("uncapped counterfactual ledger contains a censored state")
    unsafe = [row for row in proposals if row["certificate_class"] != "O-frame-safe"]
    unsafe_by_sha = {
        (row["cell_id"], row["global_shot_id"], row["proposal_sha256"]): row
        for row in unsafe
    }
    states_by_sha = {
        (row["cell_id"], row["global_shot_id"], str(row["original_proposal_sha256"])): row
        for row in states
    }
    if len(unsafe_by_sha) != len(unsafe) or len(states_by_sha) != len(states):
        raise PolicyAnalysisError("unsafe proposal/state identity is not unique")
    if set(unsafe_by_sha) != set(states_by_sha):
        raise PolicyAnalysisError("unsafe original commitments do not reconcile one-to-one")
    for proposal_sha, state in states_by_sha.items():
        # The counterfactual sidecar references the original proposal digest;
        # trajectory order is owned by that original shadow commitment.
        state["trajectory_commit_index"] = unsafe_by_sha[proposal_sha][
            "trajectory_commit_index"
        ]

    shot_by_key = {(shot["cell_id"], shot["global_shot_id"]): shot for shot in shots}
    proposals_by_shot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    states_by_shot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        proposals_by_shot[(proposal["cell_id"], proposal["global_shot_id"])].append(proposal)
    for state in states:
        states_by_shot[(state["cell_id"], state["global_shot_id"])].append(state)
    for key, values in proposals_by_shot.items():
        sequence = [value["trajectory_commit_index"] for value in values]
        if len(values) > 1 and any(value is None for value in sequence):
            raise PolicyAnalysisError(f"shot {key} lacks original-trajectory commitment order")
        if all(value is not None for value in sequence):
            values.sort(key=lambda value: (int(value["trajectory_commit_index"]), value["proposal_sha256"]))
    for key, values in states_by_shot.items():
        sequence = [value["trajectory_commit_index"] for value in values]
        if len(values) > 1 and any(value is None for value in sequence):
            raise PolicyAnalysisError(f"shot {key} lacks unsafe-state trajectory order")
        if all(value is not None for value in sequence):
            values.sort(
                key=lambda value: (
                    int(value["trajectory_commit_index"]),
                    value["original_proposal_sha256"],
                )
            )

    discordant = []
    for key, shot in shot_by_key.items():
        u0_prediction = shot["predictions"]["u0"]
        pu_prediction = shot["predictions"]["shadow"]
        if u0_prediction is None or pu_prediction is None:
            raise PolicyAnalysisError("shot ledger lacks exact U0/PU predictions")
        if u0_prediction != pu_prediction:
            discordant.append(key)
            if not any(not row["frame_compatible"] for row in proposals_by_shot[key]):
                raise PolicyAnalysisError(
                    "U0/PU-discordant shot has no durable original frame conflict"
                )

    proposal_replicates, proposal_seed = _bootstrap_config(corpus.config, family="proposal")
    workload_replicates, workload_seed = _bootstrap_config(corpus.config, family="workload")

    paired = {
        "u0_vs_shadow": _paired_table(
            [shot["failures"]["u0"] for shot in shots],
            [shot["failures"]["shadow"] for shot in shots],
        ),
        "u0_vs_o_cost_tx": _paired_table(
            [shot["failures"]["u0"] for shot in shots],
            [shot["failures"]["o-cost-tx"] for shot in shots],
        ),
    }
    activated_shot_keys: set[tuple[str, int]] = set()
    for row in corpus.rows["domains"]:
        arm_id = _at(row, "arm_id", required=True)
        status = _at(row, "domain_terminal_status", "status", "outcome.status", required=True)
        if isinstance(arm_id, str) and _arm_role(arm_id) == "shadow" and status != "below-limit":
            activated_shot_keys.add(_shot_key(row))
    overview = {
        "cell_id": cell_id,
        "physical_cell": corpus.config.get("cell"),
        "arm_definitions": corpus.config.get("arms"),
        "source_hashes": corpus.source_hashes,
        "shots": len(shots),
        "workers": len({_identity(row)[1] for row in corpus.rows["shots"]}),
        "zero_event_shots": sum(shot["zero_event"] for shot in shots),
        "nonzero_event_shots": sum(not shot["zero_event"] for shot in shots),
        "activated_shots": len(activated_shot_keys),
        "u0_failures": sum(shot["failures"]["u0"] for shot in shots),
        "shadow_failures": sum(shot["failures"]["shadow"] for shot in shots),
        "o_cost_tx_failures": sum(shot["failures"]["o-cost-tx"] for shot in shots),
        "u0_shadow_prediction_disagreements": len(discordant),
        "shots_with_unsafe_durable_original": sum(bool(states_by_shot[key]) for key in shot_keys),
        "unsafe_durable_original_commitments": len(states),
        "o_frame_tx_equals_u0_predictions": True,
        "o_frame_partial_equals_u0_predictions": True,
    }
    overview["predecoder_activation_rate"] = overview["activated_shots"] / len(shots)
    overview["u0_shadow_prediction_disagreement_rate"] = len(discordant) / len(shots)
    overview["shots_with_unsafe_durable_original_rate"] = (
        overview["shots_with_unsafe_durable_original"] / len(shots)
    )

    certificate_rows = []
    for stage in range(1, 5):
        stage_rows = [row for row in proposals if row["stage"] == stage]
        for certificate in CERTIFICATE_CLASSES:
            certificate_rows.append(
                {
                    "stage": stage,
                    "certificate_class": certificate,
                    "count": sum(row["certificate_class"] == certificate for row in stage_rows),
                    "stage_denominator": len(stage_rows),
                }
            )
    certificate_categories = [
        f"stage={stage}|class={certificate}"
        for stage in range(1, 5)
        for certificate in CERTIFICATE_CLASSES
    ]
    certificate_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=certificate_categories,
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                f"stage={row['stage']}|class={certificate}",
                row["certificate_class"] == certificate,
            )
            for row in proposals
            for certificate in CERTIFICATE_CLASSES
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="certificate-fraction-by-stage",
    )
    interval_by_category = {row["category"]: row for row in certificate_intervals}
    for row in certificate_rows:
        category = f"stage={row['stage']}|class={row['certificate_class']}"
        row["fraction"] = interval_by_category[category]["fraction"]
        row["bootstrap"] = interval_by_category[category]["bootstrap"]
        unsafe_in_stage = sum(
            proposal["stage"] == row["stage"]
            and proposal["certificate_class"] != "O-frame-safe"
            for proposal in proposals
        )
        row["unsafe_state_count_for_sparse_rule"] = unsafe_in_stage
        row["stratum_status"] = (
            "insufficient-for-rule-formulation"
            if unsafe_in_stage < SPARSE_UNSAFE_STATES
            else "descriptive-only"
        )
    domain_certificate_counts = Counter(
        (*row["domain"], row["stage"], row["certificate_class"])
        for row in proposals
    )
    observed_domain_stages = sorted(
        {(*row["domain"], row["stage"]) for row in proposals},
        key=lambda values: tuple(str(value) for value in values),
    )
    certificate_by_domain = [
        {
            "patch_id": domain_stage[0],
            "basis": domain_stage[1],
            "window_id": domain_stage[2],
            "stage": domain_stage[3],
            "certificate_class": certificate,
            "count": domain_certificate_counts[(*domain_stage, certificate)],
        }
        for domain_stage in observed_domain_stages
        for certificate in CERTIFICATE_CLASSES
    ]
    domain_stage_denominators = Counter(
        (row["domain"][0], row["domain"][1], row["domain"][2], row["stage"])
        for row in proposals
    )
    domain_stage_unsafe = Counter(
        (row["domain"][0], row["domain"][1], row["domain"][2], row["stage"])
        for row in proposals
        if row["certificate_class"] != "O-frame-safe"
    )
    domain_categories = sorted(
        {
            canonical_json_bytes(
                [row["domain"][0], row["domain"][1], row["domain"][2], row["stage"], certificate]
            ).decode("utf-8")
            for row in proposals
            for certificate in CERTIFICATE_CLASSES
        }
    )
    domain_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=domain_categories,
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                canonical_json_bytes(
                    [row["domain"][0], row["domain"][1], row["domain"][2], row["stage"], certificate]
                ).decode("utf-8"),
                row["certificate_class"] == certificate,
            )
            for row in proposals
            for certificate in CERTIFICATE_CLASSES
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="certificate-fraction-by-domain-and-stage",
    )
    domain_interval = {row["category"]: row for row in domain_intervals}
    for row in certificate_by_domain:
        key = (row["patch_id"], row["basis"], row["window_id"], row["stage"])
        denominator = domain_stage_denominators[key]
        row["domain_stage_denominator"] = denominator
        row["fraction"] = row["count"] / denominator
        category = canonical_json_bytes(
            [*key, row["certificate_class"]]
        ).decode("utf-8")
        row["bootstrap"] = domain_interval[category]["bootstrap"]
        row["unsafe_state_count_for_sparse_rule"] = domain_stage_unsafe[key]
        row["stratum_status"] = (
            "insufficient-for-rule-formulation"
            if domain_stage_unsafe[key] < SPARSE_UNSAFE_STATES
            else "descriptive-only"
        )
    unsafe_stage = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=[str(stage) for stage in range(1, 5)],
        observations=[
            ((row["cell_id"], row["global_shot_id"]), str(row["stage"]), row["certificate_class"] != "O-frame-safe")
            for row in proposals
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="unsafe-fraction-by-stage",
    )

    terminal_actions = []
    action_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=list(TERMINAL_ACTIONS),
        observations=[
            (
                (state["cell_id"], state["global_shot_id"]),
                action,
                state["terminal_action"] == action,
            )
            for state in states
            for action in TERMINAL_ACTIONS
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="counterfactual-terminal-action",
    )
    action_interval = {row["category"]: row for row in action_intervals}
    for action in TERMINAL_ACTIONS:
        group = [state for state in states if state["terminal_action"] == action]
        terminal_actions.append(
            {
                "terminal_action": action,
                "count": len(group),
                "denominator": len(states),
                "fraction": None if not states else len(group) / len(states),
                "bootstrap": action_interval[action]["bootstrap"],
                "stratum_status": (
                    "insufficient-for-rule-formulation"
                    if len(group) < SPARSE_UNSAFE_STATES
                    else "descriptive-only"
                ),
            }
        )
    first_safe_rank = _count_table(
        ({"first_safe_rank": state["first_safe_rank"]} for state in states),
        keys=("first_safe_rank",),
    )
    rank_values = sorted(
        {state["first_safe_rank"] for state in states},
        key=lambda value: (value is None, -1 if value is None else int(value)),
    )
    rank_categories = ["abstain" if value is None else f"rank={value}" for value in rank_values]
    rank_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=rank_categories,
        observations=[
            (
                (state["cell_id"], state["global_shot_id"]),
                "abstain" if candidate is None else f"rank={candidate}",
                state["first_safe_rank"] == candidate,
            )
            for state in states
            for candidate in rank_values
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="first-safe-rank-fractions",
    )
    rank_interval = {row["category"]: row for row in rank_intervals}
    for row in first_safe_rank:
        category = (
            "abstain"
            if row["first_safe_rank"] is None
            else f"rank={row['first_safe_rank']}"
        )
        row["denominator"] = len(states)
        row["fraction"] = None if not states else row["count"] / len(states)
        row["bootstrap"] = rank_interval[category]["bootstrap"]
        row["stratum_status"] = (
            "insufficient-for-rule-formulation"
            if row["count"] < SPARSE_UNSAFE_STATES
            else "descriptive-only"
        )
    transitions = _count_table(
        (
            {
                "original_stage": state["original_stage"],
                "terminal_stage": state["first_safe_stage"]
                if state["first_safe_stage"] is not None
                else "abstain",
            }
            for state in states
        ),
        keys=("original_stage", "terminal_stage"),
    )

    context_fields = {
        "matched_partner_labels": "matched_partner_labels",
        "support_path_labels": "support_path_labels",
        "support_difference_component_labels": "support_difference_component_labels",
        "omitted_context_labels": "omitted_context_labels",
        "degeneracy_diagnostics": "degeneracy_diagnostics",
    }
    context_vocabularies = {
        "matched_partner_labels": CONTEXT_PRIORITY,
        "support_path_labels": CONTEXT_PRIORITY,
        "support_difference_component_labels": CONTEXT_PRIORITY,
        "omitted_context_labels": CONTEXT_PRIORITY,
        "degeneracy_diagnostics": tuple(sorted(DEGENERACY_DIAGNOSTICS)),
    }
    context_tables = {
        view: [
            {
                "label": label,
                "count": sum(label in row[field] for row in unsafe),
            }
            for label in vocabulary
        ]
        for view, (field, vocabulary) in {
            view: (field, context_vocabularies[view])
            for view, field in context_fields.items()
        }.items()
    }
    context_tables["exclusive_support_component_context"] = [
        {
            "label": None if label == "none" else label,
            "count": sum(
                (row["exclusive_support_component_context"] or "none") == label
                for row in unsafe
            ),
        }
        for label in (*CONTEXT_PRIORITY, "none")
    ]
    context_categories = [
        f"{view}|{label}"
        for view, vocabulary in context_vocabularies.items()
        for label in vocabulary
    ] + [
        f"exclusive_support_component_context|{label}"
        for label in (*CONTEXT_PRIORITY, "none")
    ]
    context_observations = []
    for row in unsafe:
        shot_key = (row["cell_id"], row["global_shot_id"])
        for view, field in context_fields.items():
            labels = row[field]
            for label in context_vocabularies[view]:
                context_observations.append((shot_key, f"{view}|{label}", label in labels))
        exclusive = row["exclusive_support_component_context"] or "none"
        for label in (*CONTEXT_PRIORITY, "none"):
            context_observations.append(
                (shot_key, f"exclusive_support_component_context|{label}", exclusive == label)
            )
    context_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=context_categories,
        observations=context_observations,
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="unsafe-context-fractions",
    )
    context_interval = {row["category"]: row for row in context_intervals}
    for name, table in context_tables.items():
        for row in table:
            row["unsafe_state_denominator"] = len(unsafe)
            interval = context_interval.get(f"{name}|{row['label'] or 'none'}")
            if interval is not None:
                row["fraction"] = interval["fraction"]
                row["bootstrap"] = interval["bootstrap"]
            row["stratum_status"] = (
                "insufficient-for-rule-formulation"
                if row["count"] < SPARSE_UNSAFE_STATES
                else "descriptive-only"
            )
    visibility_field_counts: Counter[str] = Counter()
    visibility_state_counts: Counter[str] = Counter()
    for row in unsafe:
        raw_visibility = _at(row["raw"], "feature_visibility")
        present = {"oracle-only"}
        if raw_visibility is not None:
            if not isinstance(raw_visibility, Mapping) or any(
                not isinstance(value, str) for value in raw_visibility.values()
            ):
                raise PolicyAnalysisError("feature_visibility must map fields to classes")
            visibility_field_counts.update(raw_visibility.values())
            present.update(raw_visibility.values())
        # The certificate and all three support views are explicitly oracle-only.
        visibility_field_counts["oracle-only"] += 4
        visibility_state_counts.update(present)
    visibility_summary = [
        {
            "visibility_class": visibility,
            "unsafe_states_with_class": visibility_state_counts[visibility],
            "unsafe_state_denominator": len(unsafe),
            "recorded_field_occurrences": visibility_field_counts[visibility],
        }
        for visibility in (
            "L1-local-dynamic",
            "L1-static-boundary",
            "temporal-neighbor-dynamic",
            "nonlocal-yoke-dynamic",
            "oracle-only",
            "posthoc-ground-truth",
        )
    ]

    arm_order = ("shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial")
    workload_matrix = np.empty((len(shots), 2 * len(arm_order)), dtype=np.float64)
    for row_index, shot in enumerate(shots):
        for arm_index, arm in enumerate(arm_order):
            workload_matrix[row_index, 2 * arm_index] = shot["final_hw"][arm]
            workload_matrix[row_index, 2 * arm_index + 1] = shot["original_hw"]
    workload_intervals = clustered_bootstrap_ratios(
        workload_matrix,
        replicates=workload_replicates,
        seed=derive_bootstrap_seed(workload_seed, cell_id=cell_id, estimand="R-event"),
    )
    original_total = sum(shot["original_hw"] for shot in shots)
    shadow_relief = original_total - sum(shot["final_hw"]["shadow"] for shot in shots)
    event_summary = []
    for arm_index, arm in enumerate(arm_order):
        final_total = sum(shot["final_hw"][arm] for shot in shots)
        relief = original_total - final_total
        provisional = _sum_optional(
            (shot["provisional_removed"][arm] for shot in shots),
            name="provisional_events_removed",
        )
        rollback_lost = _sum_optional(
            (shot["rollback_lost"][arm] for shot in shots),
            name="events_lost_to_rollback",
        )
        if (
            provisional is not None
            and rollback_lost is not None
            and provisional - rollback_lost != relief
        ):
            raise PolicyAnalysisError(f"{arm} aggregate transaction/event accounting disagrees")
        event_summary.append(
            {
                "arm": arm,
                "shots": len(shots),
                "sum_original_detector_hw": original_total,
                "sum_final_residual_detector_hw": final_total,
                "R_event": None if original_total == 0 else final_total / original_total,
                "R_event_bootstrap": workload_intervals[arm_index],
                "durable_events_removed": relief,
                "provisional_events_removed": provisional,
                "events_lost_to_rollback": rollback_lost,
                "transaction_telemetry_complete": (
                    provisional is not None and rollback_lost is not None
                ),
                "R_relief_retained_relative_to_shadow": (
                    None if shadow_relief == 0 else relief / shadow_relief
                ),
            }
        )
    residual_hw_distributions = {
        arm: distribution_summary([shot["final_hw"][arm] for shot in shots])
        for arm in arm_order
    }

    association = []
    for key in shot_keys:
        shot = shot_by_key[key]
        ordered_states = states_by_shot[key]
        first = ordered_states[0] if ordered_states else None
        association.append(
            {
                "cell_id": key[0],
                "global_shot_id": key[1],
                "unsafe_count_bin": _unsafe_bin(len(ordered_states)),
                "first_unsafe_stage": None if first is None else first["original_stage"],
                "first_unsafe_context": None if first is None else first["exclusive_context"],
                "terminal_action": None if first is None else first["terminal_action"],
                "paired_outcome": _outcome_name(shot),
                "prediction_disagreement": shot["predictions"]["u0"] != shot["predictions"]["shadow"],
            }
        )
    association_by_unsafe_count = []
    for bin_name in ("0", "1", "2", "3", "4+"):
        group = [row for row in association if row["unsafe_count_bin"] == bin_name]
        counts = Counter(row["paired_outcome"] for row in group)
        association_by_unsafe_count.append(
            {
                "unsafe_count_bin": bin_name,
                "shots": len(group),
                "prediction_agreement": counts["prediction-agreement"],
                "regression": counts["regression"],
                "recovery": counts["recovery"],
                "prediction_discordant_both_wrong": counts[
                    "prediction-discordant-both-wrong"
                ],
                "prediction_disagreement": sum(row["prediction_disagreement"] for row in group),
            }
        )
    unsafe_counts = {
        key: len(states_by_shot[key])
        for key in shot_keys
    }
    unsafe_count_values = sorted(set(unsafe_counts.values()))
    unsafe_count_categories = [str(value) for value in unsafe_count_values]
    unsafe_count_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=unsafe_count_categories,
        observations=[
            (key, str(candidate), count == candidate)
            for key, count in unsafe_counts.items()
            for candidate in unsafe_count_values
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="full-unsafe-count-distribution",
    )
    unsafe_count_interval = {row["category"]: row for row in unsafe_count_intervals}
    unsafe_count_distribution = [
        {
            "unsafe_count": count,
            "shots": sum(value == count for value in unsafe_counts.values()),
            "denominator": len(shots),
            "fraction": unsafe_count_interval[str(count)]["fraction"],
            "bootstrap": unsafe_count_interval[str(count)]["bootstrap"],
        }
        for count in unsafe_count_values
    ]
    association_by_first = {}
    for field in ("first_unsafe_stage", "first_unsafe_context", "terminal_action"):
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in association:
            grouped[row[field]].append(row)
        association_by_first[field] = [
            {
                field: value,
                "shots": len(group),
                **dict(Counter(row["paired_outcome"] for row in group)),
                "prediction_disagreement": sum(row["prediction_disagreement"] for row in group),
            }
            for value, group in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
    first_conflict_discordant = _count_table(
        (
            {
                "first_unsafe_stage": row["first_unsafe_stage"],
                "first_unsafe_context": row["first_unsafe_context"] or "none",
            }
            for row in association
            if row["prediction_disagreement"]
        ),
        keys=("first_unsafe_stage", "first_unsafe_context"),
    )
    if sum(row["count"] for row in first_conflict_discordant) != sum(
        row["prediction_disagreement"] for row in association
    ):
        raise PolicyAnalysisError("first-conflict context totals do not reconcile")

    distributions = {
        "cost_excess": distribution_summary(
            [row["cost_excess"] for row in proposals if row["cost_excess"] is not None]
        ),
        "local_weight_margin": distribution_summary(
            [row["local_weight_margin"] for row in proposals if row["local_weight_margin"] is not None]
        ),
        "veto_chain_length": distribution_summary([state["veto_chain_length"] for state in states]),
        "candidate_count": distribution_summary(
            [state["candidate_count"] for state in states if state["candidate_count"] is not None]
        ),
        "events_removed": distribution_summary(
            [row["events_removed"] for row in proposals if row["events_removed"] is not None]
        ),
        "cost_tolerance_tau_k": distribution_summary(
            [row["tau_k"] for row in proposals if row["tau_k"] is not None]
        ),
    }
    competitor_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=["available", "unavailable"],
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                category,
                row["same_stage_competitor_exists"] is (category == "available"),
            )
            for row in proposals
            for category in ("available", "unavailable")
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="same-stage-competitor-availability",
    )
    competitor_summary = {
        "shadow_commitments": len(proposals),
        "available": sum(row["same_stage_competitor_exists"] is True for row in proposals),
        "unavailable": sum(row["same_stage_competitor_exists"] is False for row in proposals),
        "unrecorded": 0,
        "margin_denominator": sum(row["local_weight_margin"] is not None for row in proposals),
        "availability_fractions": [
            {
                "availability": row["category"],
                "count": row["numerator"],
                "denominator": row["denominator"],
                "fraction": row["fraction"],
                "bootstrap": row["bootstrap"],
            }
            for row in competitor_intervals
        ],
    }
    cost_ecdf_by_stage = {
        str(stage): exact_ecdf(
            [row["cost_excess"] for row in proposals if row["stage"] == stage and row["cost_excess"] is not None]
        )
        for stage in range(1, 5)
    }
    cost_ecdf_by_certificate = {
        certificate: exact_ecdf(
            [
                row["cost_excess"]
                for row in proposals
                if row["certificate_class"] == certificate and row["cost_excess"] is not None
            ]
        )
        for certificate in CERTIFICATE_CLASSES
    }
    cost_ecdf_by_context = {
        label: exact_ecdf(
            [
                row["cost_excess"]
                for row in proposals
                if row["exclusive_support_component_context"] == context_value
                and row["cost_excess"] is not None
            ]
        )
        for label in (*CONTEXT_PRIORITY, "none")
        for context_value in [None if label == "none" else label]
    }
    if sum(points[-1]["denominator"] for points in cost_ecdf_by_context.values() if points) != sum(
        row["cost_excess"] is not None for row in proposals
    ):
        raise PolicyAnalysisError("cost ECDF context totals do not reconcile")
    original_vs_alternative = [
        {
            key: state[key]
            for key in (
                "state_id", "original_stage", "first_safe_stage",
                "original_decision_weight", "first_safe_decision_weight",
                "original_path_length", "first_safe_path_length",
                "original_weight_margin", "first_safe_weight_margin",
                "original_events_removed", "first_safe_events_removed",
            )
        }
        for state in states
    ]
    margins = sorted(
        row["local_weight_margin"]
        for row in proposals
        if row["local_weight_margin"] is not None
    )
    margin_edges = [empirical_type7(margins, q / 10) for q in range(11)] if margins else []
    risk_rows = []
    for row in proposals:
        copy = dict(row)
        margin = row["local_weight_margin"]
        if margin is None or not margin_edges:
            copy["margin_decile"] = None
        else:
            copy["margin_decile"] = min(
                10,
                1 + sum(float(margin) > float(edge) for edge in margin_edges[1:-1]),
            )
        risk_rows.append(copy)
    risk_heatmaps = {
        "stage_by_window_offset": _risk_count_table(
            risk_rows, keys=("stage", "window_offset")
        ),
        "stage_by_static_boundary_competition": _risk_count_table(
            risk_rows, keys=("stage", "static_boundary_competition")
        ),
        "domain_hw_by_candidate_multiplicity": _risk_count_table(
            risk_rows, keys=("domain_hw", "candidate_multiplicity")
        ),
        "stage_by_margin_decile": _risk_count_table(
            risk_rows, keys=("stage", "margin_decile")
        ),
        "margin_decile_edges_type7": margin_edges,
        "threshold_use_permitted": False,
    }
    tail_metrics: dict[str, list[float]] = {
        "proposals_per_state": [float(len(state["rows"])) for state in states],
        "vetoes_per_state": [float(state["veto_chain_length"]) for state in states],
    }
    incomplete_tail_metrics: set[str] = set()
    for output_name, field_names in {
        "oracle_calls_per_state": (
            "state_oracle_call_count",
            "oracle_calls",
            "state_oracle_calls",
        ),
        "stage3_enumeration_ns_per_state": (
            "state_total_stage3_enumeration_wall_ns",
            "stage3_enumeration_ns",
            "state_stage3_enumeration_ns",
        ),
    }.items():
        values = []
        complete = True
        for state in states:
            value = _at(state["rows"][0], *field_names)
            if value is None:
                complete = False
                break
            values.append(float(_as_nonnegative_int(value, name=output_name)))
        if not complete:
            mode = corpus.manifest.get("mode")
            if mode != "smoke":
                raise PolicyAnalysisError(
                    f"{mode} corpus lacks required {output_name} tail telemetry"
                )
            incomplete_tail_metrics.add(output_name)
        tail_metrics[output_name] = values if complete else []
    for metric in tuple(tail_metrics):
        if not metric.endswith("_per_state"):
            continue
        shot_values: dict[tuple[str, int], float] = defaultdict(float)
        complete = True
        for state_index, state in enumerate(states):
            values = tail_metrics[metric]
            if state_index >= len(values):
                complete = False
                break
            shot_values[(state["cell_id"], state["global_shot_id"])] += values[state_index]
        tail_metrics[metric.removesuffix("_per_state") + "_per_shot"] = (
            [shot_values[key] for key in shot_keys] if complete else []
        )
        if not complete:
            incomplete_tail_metrics.add(metric.removesuffix("_per_state") + "_per_shot")
    veto_chain_tails = {
        name: {
            **distribution_summary(values),
            "telemetry_status": (
                "incomplete-smoke" if name in incomplete_tail_metrics else "complete"
            ),
        }
        for name, values in tail_metrics.items()
    }

    domain_summary = _domain_table(corpus.rows["domains"])
    casebook = select_casebook(states)
    recomputed_gate_evidence: dict[int, str] = {
        1: "loader recomputed exact mode schedule, shot IDs, worker ranges, and row counts",
        2: "loader reconciled every sidecar row to one authenticated physical-shot provenance",
        5: "analyzer compared exact U0, O-frame-transactional, and O-frame-partial prediction tokens",
        6: "analyzer recomputed certificate classes from exact-hex cost excess, tolerance, and frame compatibility",
        10: "analyzer reconciled counterfactual rank one by proposal digest and canonical proposal signature to the durable shadow original",
        11: "analyzer recomputed uncapped unchanged-state ranks, signatures, decisions, stage monotonicity, first-safe stop, and true exhaustion",
        12: "analyzer reconciled aggregate provisional, rollback, durable, and residual event accounting",
        13: "analyzer reconciled every exact U0/PU prediction disagreement to a durable frame conflict",
        15: "analyzer validated the closed context vocabulary and reconciled omitted-context and degeneracy diagnostics",
        17: "loader authenticated canonical gzip/JSONL bytes, digests, schemas, identities, and COLLECTION_READY",
    }
    collector_attestations = _collector_gate_attestations(corpus.manifest)
    gates = [
        {
            "gate": gate,
            "status": (
                "passed-ledger-recomputed"
                if gate in recomputed_gate_evidence
                else (
                    "collector-attested"
                    if gate in collector_attestations
                    else "missing-collector-attestation"
                )
            ),
            "evidence": {
                "kind": (
                    "ledger-recomputation"
                    if gate in recomputed_gate_evidence
                    else (
                        "authenticated-collector-attestation"
                        if gate in collector_attestations
                        else "missing"
                    )
                ),
                "claim": (
                    recomputed_gate_evidence[gate]
                    if gate in recomputed_gate_evidence
                    else collector_attestations.get(gate)
                ),
                "authenticated_by": (
                    ["canonical-ledgers", "manifest.json", "COLLECTION_READY"]
                    if gate in recomputed_gate_evidence
                    else (
                        ["manifest.json", "COLLECTION_READY"]
                        if gate in collector_attestations
                        else []
                    )
                ),
            },
        }
        for gate in range(1, 19)
    ]
    interpretation_checkpoints = [
        {
            "checkpoint": "few-final-disagreements",
            "fatal": False,
            "observed": len(discordant),
        },
        {
            "checkpoint": "little-certified-event-relief",
            "fatal": False,
            "observed_shadow_R_event": next(
                row["R_event"] for row in event_summary if row["arm"] == "shadow"
            ),
        },
        {
            "checkpoint": "many-true-abstentions",
            "fatal": False,
            "observed": sum(
                state["terminal_action"] == "abstain-true-exhaustion" for state in states
            ),
        },
        {
            "checkpoint": "nonlocal-yoke-context-dominates",
            "fatal": False,
            "observed": sum(state["exclusive_context"] == "yoke" for state in states),
        },
        {
            "checkpoint": "shadow-effect-magnitude",
            "fatal": False,
            "observed_paired_risk_difference": paired["u0_vs_shadow"][
                "paired_risk_difference"
            ],
        },
    ]
    tables = {
        "overview": overview,
        "paired_outcomes": paired,
        "event_and_transaction_summary": event_summary,
        "residual_hw_distributions": residual_hw_distributions,
        "domain_terminal_summary": domain_summary,
        "certificate_by_stage": certificate_rows,
        "certificate_by_domain": certificate_by_domain,
        "unsafe_fraction_by_stage": unsafe_stage,
        "counterfactual_terminal_action": terminal_actions,
        "first_safe_rank": first_safe_rank,
        "stage_transition": transitions,
        "context_views": context_tables,
        "visibility_summary": visibility_summary,
        "association_by_unsafe_count": association_by_unsafe_count,
        "unsafe_count_distribution": unsafe_count_distribution,
        "association_by_first": association_by_first,
        "first_conflict_discordant": first_conflict_discordant,
        "continuous_distributions": distributions,
        "local_competitor_summary": competitor_summary,
        "cost_excess_ecdf_by_stage": cost_ecdf_by_stage,
        "cost_excess_ecdf_by_certificate": cost_ecdf_by_certificate,
        "cost_excess_ecdf_by_context": cost_ecdf_by_context,
        "original_vs_alternative": original_vs_alternative,
        "risk_heatmaps": risk_heatmaps,
        "veto_chain_tails": veto_chain_tails,
        "fatal_gates": gates,
        "interpretation_checkpoints": interpretation_checkpoints,
    }
    result = {
        "schema": ANALYSIS_SCHEMA,
        "experiment_id": _at(corpus.experiment, "experiment_id", required=True),
        "cell_id": cell_id,
        "analysis_contract": {
            "source": "immutable-canonical-gzip-jsonl-only",
            "sampling_or_decoding_reconstruction": False,
            "bootstrap_unit": "complete-physical-shot",
            "bootstrap_quantile": "empirical-type-7",
            "proposal_bootstrap_replicates": proposal_replicates,
            "workload_bootstrap_replicates": workload_replicates,
            "casebook_outcome_blind": True,
            "casebook_exhaustive_rows_excluded": True,
            "support_context_views_kept_distinct": True,
            "required_tail_telemetry": (
                "complete"
                if not incomplete_tail_metrics
                else "incomplete-smoke-only"
            ),
            "complete_written_by_analyzer": False,
            "complete_accepted_as_analysis_substitute": False,
            "next_required_stage": "casebook-expansion-and-finalization-external",
        },
        "source_hashes": corpus.source_hashes,
        "tables": tables,
        "casebook_selection": casebook,
    }
    result["analysis_sha256"] = _sha256(canonical_json_bytes(result))
    return result


def _write_canonical_json(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    return _sha256(data)


def _report_object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyAnalysisError(f"human report requires {name} to be an object")
    return value


def _report_rows(value: Any, *, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise PolicyAnalysisError(f"human report requires {name} to be an array of objects")
    return value


def _report_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyAnalysisError(f"human report requires {name} to be a nonnegative integer")
    return value


def _report_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyAnalysisError(f"human report requires {name} to be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyAnalysisError(f"human report requires {name} to be a finite number")
    return result


def _report_single_line(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n|`"):
        raise PolicyAnalysisError(f"human report requires {name} to be safe single-line text")
    return value


def _report_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator:,} / 0 (undefined)"
    return f"{numerator:,} / {denominator:,} ({numerator / denominator:.3%})"


def _report_ratio(value: Any, *, name: str) -> str:
    if value is None:
        return "undefined"
    return f"{_report_number(value, name=name):.6g}"


def policy_human_report_bytes(analysis: Mapping[str, Any]) -> bytes:
    """Renders the frozen human report solely from downstream analysis tables."""

    if not isinstance(analysis, Mapping):
        raise PolicyAnalysisError("human report requires an analysis object")
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise PolicyAnalysisError("human report requires the exact analysis schema")
    experiment_id = _report_single_line(
        analysis.get("experiment_id"), name="experiment_id"
    )
    cell_id = _report_single_line(analysis.get("cell_id"), name="cell_id")
    tables = _report_object(analysis.get("tables"), name="tables")
    required_tables = {
        "overview",
        "paired_outcomes",
        "event_and_transaction_summary",
        "certificate_by_stage",
        "counterfactual_terminal_action",
        "context_views",
        "visibility_summary",
        "local_competitor_summary",
    }
    if not required_tables <= set(tables):
        missing = sorted(required_tables - set(tables))
        raise PolicyAnalysisError(f"human report is missing required tables: {missing}")

    overview = _report_object(tables["overview"], name="tables.overview")
    shots = _report_count(overview.get("shots"), name="overview.shots")
    if shots == 0:
        raise PolicyAnalysisError("human report requires at least one shot")
    workers = _report_count(overview.get("workers"), name="overview.workers")
    zero_event = _report_count(
        overview.get("zero_event_shots"), name="overview.zero_event_shots"
    )
    nonzero_event = _report_count(
        overview.get("nonzero_event_shots"), name="overview.nonzero_event_shots"
    )
    if zero_event + nonzero_event != shots:
        raise PolicyAnalysisError("human report shot denominators do not reconcile")
    unsafe_shots = _report_count(
        overview.get("shots_with_unsafe_durable_original"),
        name="overview.shots_with_unsafe_durable_original",
    )
    unsafe_states = _report_count(
        overview.get("unsafe_durable_original_commitments"),
        name="overview.unsafe_durable_original_commitments",
    )
    disagreements = _report_count(
        overview.get("u0_shadow_prediction_disagreements"),
        name="overview.u0_shadow_prediction_disagreements",
    )
    activated = _report_count(
        overview.get("activated_shots"), name="overview.activated_shots"
    )
    failure_counts = {
        "U0": _report_count(overview.get("u0_failures"), name="overview.u0_failures"),
        "shadow": _report_count(
            overview.get("shadow_failures"), name="overview.shadow_failures"
        ),
        "O-cost transactional": _report_count(
            overview.get("o_cost_tx_failures"), name="overview.o_cost_tx_failures"
        ),
    }
    if (
        overview.get("o_frame_tx_equals_u0_predictions") is not True
        or overview.get("o_frame_partial_equals_u0_predictions") is not True
    ):
        raise PolicyAnalysisError("human report requires authenticated O-frame/U0 equality")
    failure_counts["O-frame transactional"] = failure_counts["U0"]
    failure_counts["O-frame partial"] = failure_counts["U0"]

    workload_rows = _report_rows(
        tables["event_and_transaction_summary"],
        name="tables.event_and_transaction_summary",
    )
    workload_by_arm: dict[str, Mapping[str, Any]] = {}
    for row in workload_rows:
        arm = row.get("arm")
        if arm not in {"shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial"}:
            raise PolicyAnalysisError("human report encountered an unknown workload arm")
        if arm in workload_by_arm:
            raise PolicyAnalysisError("human report encountered a duplicate workload arm")
        if _report_count(row.get("shots"), name=f"workload.{arm}.shots") != shots:
            raise PolicyAnalysisError("human report workload shot denominator disagrees")
        workload_by_arm[str(arm)] = row
    if set(workload_by_arm) != {
        "shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial"
    }:
        raise PolicyAnalysisError("human report requires the exact workload arm set")

    paired = _report_object(tables["paired_outcomes"], name="tables.paired_outcomes")
    u0_shadow = _report_object(paired.get("u0_vs_shadow"), name="paired.u0_vs_shadow")
    if _report_count(u0_shadow.get("shots"), name="paired.u0_vs_shadow.shots") != shots:
        raise PolicyAnalysisError("human report paired denominator disagrees")
    regressions = _report_count(
        u0_shadow.get("regressions"), name="paired.u0_vs_shadow.regressions"
    )
    recoveries = _report_count(
        u0_shadow.get("recoveries"), name="paired.u0_vs_shadow.recoveries"
    )

    competitor = _report_object(
        tables["local_competitor_summary"], name="tables.local_competitor_summary"
    )
    shadow_commitments = _report_count(
        competitor.get("shadow_commitments"), name="local.shadow_commitments"
    )
    local_available = _report_count(
        competitor.get("available"), name="local.available"
    )
    local_unavailable = _report_count(
        competitor.get("unavailable"), name="local.unavailable"
    )
    local_unrecorded = _report_count(
        competitor.get("unrecorded"), name="local.unrecorded"
    )
    if local_available + local_unavailable + local_unrecorded != shadow_commitments:
        raise PolicyAnalysisError("human report local-competitor denominator disagrees")

    visibility_rows = _report_rows(
        tables["visibility_summary"], name="tables.visibility_summary"
    )
    visibility: dict[str, Mapping[str, Any]] = {}
    expected_visibility = {
        "L1-local-dynamic",
        "L1-static-boundary",
        "temporal-neighbor-dynamic",
        "nonlocal-yoke-dynamic",
        "oracle-only",
        "posthoc-ground-truth",
    }
    for row in visibility_rows:
        label = row.get("visibility_class")
        if label not in expected_visibility or label in visibility:
            raise PolicyAnalysisError("human report visibility classes are not exact")
        denominator = _report_count(
            row.get("unsafe_state_denominator"), name=f"visibility.{label}.denominator"
        )
        if denominator != unsafe_states:
            raise PolicyAnalysisError("human report visibility denominator disagrees")
        visibility[str(label)] = row
    if set(visibility) != expected_visibility:
        raise PolicyAnalysisError("human report requires the exact visibility taxonomy")

    context_views = _report_object(tables["context_views"], name="tables.context_views")
    exclusive_rows = _report_rows(
        context_views.get("exclusive_support_component_context"),
        name="context_views.exclusive_support_component_context",
    )
    exclusive_counts: dict[str, int] = {}
    for row in exclusive_rows:
        label_value = row.get("label")
        label = "none" if label_value is None else str(label_value)
        if label not in {*CONTEXT_PRIORITY, "none"} or label in exclusive_counts:
            raise PolicyAnalysisError("human report exclusive contexts are not exact")
        if _report_count(
            row.get("unsafe_state_denominator"), name=f"context.{label}.denominator"
        ) != unsafe_states:
            raise PolicyAnalysisError("human report context denominator disagrees")
        exclusive_counts[label] = _report_count(
            row.get("count"), name=f"context.{label}.count"
        )
    if set(exclusive_counts) != {*CONTEXT_PRIORITY, "none"}:
        raise PolicyAnalysisError("human report requires the exact exclusive-context vocabulary")
    if sum(exclusive_counts.values()) != unsafe_states:
        raise PolicyAnalysisError("human report exclusive contexts do not partition unsafe states")

    degeneracy_rows = _report_rows(
        context_views.get("degeneracy_diagnostics"),
        name="context_views.degeneracy_diagnostics",
    )
    degeneracy_counts: dict[str, int] = {}
    for row in degeneracy_rows:
        label = row.get("label")
        if label not in DEGENERACY_DIAGNOSTICS or label in degeneracy_counts:
            raise PolicyAnalysisError("human report degeneracy diagnostics are not exact")
        if _report_count(
            row.get("unsafe_state_denominator"), name=f"degeneracy.{label}.denominator"
        ) != unsafe_states:
            raise PolicyAnalysisError("human report degeneracy denominator disagrees")
        degeneracy_counts[str(label)] = _report_count(
            row.get("count"), name=f"degeneracy.{label}.count"
        )
    if set(degeneracy_counts) != DEGENERACY_DIAGNOSTICS:
        raise PolicyAnalysisError("human report requires exact degeneracy diagnostics")

    certificate_stage_rows = _report_rows(
        tables["certificate_by_stage"], name="tables.certificate_by_stage"
    )
    stage_statuses: dict[int, str] = {}
    if any(
        row.get("stratum_status") not in {
            "insufficient-for-rule-formulation", "descriptive-only"
        }
        for row in certificate_stage_rows
    ):
        raise PolicyAnalysisError("human report encountered an unknown stage stratum status")
    for row in certificate_stage_rows:
        stage = _report_count(row.get("stage"), name="certificate.stage")
        if stage not in {1, 2, 3, 4}:
            raise PolicyAnalysisError("human report encountered an invalid stage")
        status = str(row["stratum_status"])
        if stage in stage_statuses and stage_statuses[stage] != status:
            raise PolicyAnalysisError("human report stage stratum statuses disagree")
        stage_statuses[stage] = status
    if set(stage_statuses) != {1, 2, 3, 4}:
        raise PolicyAnalysisError("human report requires all four stage strata")
    sparse_stages = sum(
        status == "insufficient-for-rule-formulation"
        for status in stage_statuses.values()
    )

    terminal_rows = _report_rows(
        tables["counterfactual_terminal_action"],
        name="tables.counterfactual_terminal_action",
    )
    terminal_counts: dict[str, int] = {}
    for row in terminal_rows:
        action = row.get("terminal_action")
        if action not in TERMINAL_ACTIONS or action in terminal_counts:
            raise PolicyAnalysisError("human report terminal actions are not exact")
        if _report_count(row.get("denominator"), name=f"terminal.{action}.denominator") != unsafe_states:
            raise PolicyAnalysisError("human report terminal-action denominator disagrees")
        terminal_counts[str(action)] = _report_count(
            row.get("count"), name=f"terminal.{action}.count"
        )
    if set(terminal_counts) != set(TERMINAL_ACTIONS):
        raise PolicyAnalysisError("human report requires exact terminal actions")

    lines = [
        f"<!-- format: {HUMAN_REPORT_FORMAT} -->",
        "# ProMatch L1 B1 policy-audit report",
        "",
        f"Experiment `{experiment_id}`; cell `{cell_id}`.",
        "",
        (
            "This deterministic report is generated only from the authenticated downstream "
            "analysis. It does not reconstruct sampling or decoding. All associations and "
            "explanations below are hypothesis-generating, not causal proof."
        ),
        "",
        "## Population and denominators",
        "",
        f"- Physical shots: {shots:,} across {workers:,} workers.",
        f"- Nonzero-event shots: {_report_fraction(nonzero_event, shots)}; zero-event shots: {_report_fraction(zero_event, shots)}.",
        f"- Predecoder-activated shots: {_report_fraction(activated, shots)}.",
        f"- Shots with at least one unsafe durable original commitment: {_report_fraction(unsafe_shots, shots)}.",
        f"- Unsafe durable original commitments (the denominator for context, visibility, and counterfactual summaries): {unsafe_states:,}.",
        f"- U0/shadow prediction-discordant shots: {_report_fraction(disagreements, shots)}.",
        "",
        "## Arm errors and detector-event workload",
        "",
        "Logical-error denominators are physical shots. Workload is a ratio of sums over the same physical shots; its denominator is the summed original detector-event count.",
        "",
        "| Arm | Logical errors | Final/original detector events | Durable / provisional / rollback-lost events |",
        "| --- | ---: | ---: | ---: |",
        f"| U0 | {_report_fraction(failure_counts['U0'], shots)} | reference; not separately tabulated | not applicable |",
    ]
    display_arms = (
        ("shadow", "shadow"),
        ("o-cost-tx", "O-cost transactional"),
        ("o-frame-tx", "O-frame transactional"),
        ("o-frame-partial", "O-frame partial"),
    )
    for arm, display in display_arms:
        row = workload_by_arm[arm]
        original = _report_count(
            row.get("sum_original_detector_hw"), name=f"workload.{arm}.original"
        )
        final = _report_count(
            row.get("sum_final_residual_detector_hw"), name=f"workload.{arm}.final"
        )
        durable = _report_count(
            row.get("durable_events_removed"), name=f"workload.{arm}.durable"
        )
        if original - final != durable:
            raise PolicyAnalysisError("human report workload event totals do not reconcile")
        provisional = row.get("provisional_events_removed")
        rollback = row.get("events_lost_to_rollback")
        transaction = (
            "unavailable"
            if provisional is None or rollback is None
            else f"{durable:,} / {_report_count(provisional, name=f'workload.{arm}.provisional'):,} / {_report_count(rollback, name=f'workload.{arm}.rollback'):,}"
        )
        lines.append(
            f"| {display} | {_report_fraction(failure_counts[display], shots)} | "
            f"{final:,} / {original:,} = {_report_ratio(row.get('R_event'), name=f'workload.{arm}.R_event')} | {transaction} |"
        )
    lines.extend(
        [
            "",
            f"For U0 versus shadow, regressions were {_report_fraction(regressions, shots)} and recoveries were {_report_fraction(recoveries, shots)}.",
            "O-frame transactional and partial predictions were authenticated as exactly equal to U0 predictions.",
            "",
            "## Locally observable policy clues",
            "",
            (
                "These are candidate clues available at the L1 decision surface, not labels "
                "showing whether a commitment was truly safe."
            ),
            "",
            f"- A same-stage local competitor was recorded for {_report_fraction(local_available, shadow_commitments)} shadow commitments; {_report_fraction(local_unavailable, shadow_commitments)} had none and {_report_fraction(local_unrecorded, shadow_commitments)} were unrecorded.",
        ]
    )
    for label in ("L1-local-dynamic", "L1-static-boundary"):
        row = visibility[label]
        present = _report_count(
            row.get("unsafe_states_with_class"), name=f"visibility.{label}.states"
        )
        occurrences = _report_count(
            row.get("recorded_field_occurrences"), name=f"visibility.{label}.fields"
        )
        lines.append(
            f"- `{label}` fields appeared on {_report_fraction(present, unsafe_states)} unsafe states ({occurrences:,} recorded field occurrences)."
        )
    lines.extend(
        [
            "",
            "No threshold or decision rule is licensed by these descriptive local associations.",
            "",
            "## Nonlocal and oracle-only explanations",
            "",
            (
                "Certificates, matched-partner paths, support paths, and support-difference "
                "components are oracle-only explanations. Their context labels must not be "
                "treated as locally observable policy inputs, even when the label is `in-domain`."
            ),
            "",
        ]
    )
    for label in CONTEXT_PRIORITY:
        lines.append(
            f"- Exclusive support context `{label}`: {_report_fraction(exclusive_counts[label], unsafe_states)} unsafe commitments."
        )
    lines.append(
        f"- No exclusive support context: {_report_fraction(exclusive_counts['none'], unsafe_states)} unsafe commitments."
    )
    for label in (
        "temporal-neighbor-dynamic",
        "nonlocal-yoke-dynamic",
        "oracle-only",
        "posthoc-ground-truth",
    ):
        row = visibility[label]
        present = _report_count(
            row.get("unsafe_states_with_class"), name=f"visibility.{label}.states"
        )
        occurrences = _report_count(
            row.get("recorded_field_occurrences"), name=f"visibility.{label}.fields"
        )
        lines.append(
            f"- Visibility `{label}`: {_report_fraction(present, unsafe_states)} unsafe states ({occurrences:,} recorded field occurrences)."
        )
    lines.extend(
        [
            "",
            "The exclusive context is only a display-priority partition; the distinct multi-label support views remain authoritative.",
            "",
            "## Counterfactual outcomes and limitations",
            "",
        ]
    )
    for action in TERMINAL_ACTIONS:
        lines.append(
            f"- `{action}`: {_report_fraction(terminal_counts[action], unsafe_states)} unsafe commitments."
        )
    lines.extend(
        [
            "",
            f"Sparse-stratum rule: fewer than {SPARSE_UNSAFE_STATES} unsafe states is insufficient for rule formulation. {sparse_stages:,} / {len(stage_statuses):,} stage strata are marked insufficient; all displayed strata remain descriptive only.",
            f"Tied-support diagnostic `equal-weight-logical-class`: {_report_fraction(degeneracy_counts['equal-weight-logical-class'], unsafe_states)} unsafe commitments.",
            f"Other support diagnostics are `same-pair-different-path-or-frame` {_report_fraction(degeneracy_counts['same-pair-different-path-or-frame'], unsafe_states)}, `disconnected-support-reconfiguration` {_report_fraction(degeneracy_counts['disconnected-support-reconfiguration'], unsafe_states)}, and `unclassified` {_report_fraction(degeneracy_counts['unclassified'], unsafe_states)}. Disconnected component graph roles are retained as structural evidence but excluded from policy-visible candidate context. Diagnostics can overlap and therefore do not form a partition.",
            "",
            "Sparse or tied support can make apparent context and margin patterns unstable. This audit can prioritize follow-up hypotheses; it cannot identify a causal policy rule.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _plot_payloads(tables: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "certificate-flow": {
            "schema": PLOT_TABLE_SCHEMA,
            "certificate_by_stage": tables["certificate_by_stage"],
            "terminal_action": tables["counterfactual_terminal_action"],
        },
        "unsafe-fraction-by-stage": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["unsafe_fraction_by_stage"],
        },
        "first-conflict-stage-context": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["first_conflict_discordant"],
            "context_views": tables["context_views"],
        },
        "cost-excess-ecdf": {
            "schema": PLOT_TABLE_SCHEMA,
            "by_stage": tables["cost_excess_ecdf_by_stage"],
            "by_certificate": tables["cost_excess_ecdf_by_certificate"],
            "by_context": tables["cost_excess_ecdf_by_context"],
            "tolerance_tau_k": tables["continuous_distributions"]["cost_tolerance_tau_k"],
        },
        "first-safe-action-rank": {
            "schema": PLOT_TABLE_SCHEMA,
            "actions": tables["counterfactual_terminal_action"],
            "ranks": tables["first_safe_rank"],
        },
        "stage-transition-matrix": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["stage_transition"],
        },
        "original-versus-alternative": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["original_vs_alternative"],
        },
        "risk-heatmaps": {
            "schema": PLOT_TABLE_SCHEMA,
            "tables": tables["risk_heatmaps"],
        },
        "disagreement-association": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["association_by_unsafe_count"],
        },
        "event-relief": {
            "schema": PLOT_TABLE_SCHEMA,
            "summary": tables["event_and_transaction_summary"],
            "distributions": tables["residual_hw_distributions"],
        },
        "veto-chain-tails": {
            "schema": PLOT_TABLE_SCHEMA,
            "metrics": tables["veto_chain_tails"],
        },
    }


def _save_plot(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "yoked-policy-audit"})
    import matplotlib.pyplot as plt

    plt.close(figure)


def _render_plots(plot_dir: Path, payloads: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rendered: list[str] = []

    payload = payloads["certificate-flow"]
    rows = payload["certificate_by_stage"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(4)
    for certificate in CERTIFICATE_CLASSES:
        values = [
            next(row["count"] for row in rows if row["stage"] == stage and row["certificate_class"] == certificate)
            for stage in range(1, 5)
        ]
        ax.bar(range(1, 5), values, bottom=bottom, label=certificate)
        bottom += values
    ax.set(xlabel="ProMatch stage", ylabel="durable shadow commitments", title="Certificate flow by original stage")
    ax.legend(fontsize=8)
    _save_plot(fig, plot_dir / "certificate-flow.png")
    rendered.append("certificate-flow.png")

    rows = payloads["unsafe-fraction-by-stage"]["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(1, 5)
    y = np.array([row["fraction"] if row["fraction"] is not None else np.nan for row in rows])
    lower = np.array([row["bootstrap"]["lower"] if row["bootstrap"]["lower"] is not None else np.nan for row in rows])
    upper = np.array([row["bootstrap"]["upper"] if row["bootstrap"]["upper"] is not None else np.nan for row in rows])
    ax.errorbar(
        x,
        y,
        yerr=np.maximum(0, np.vstack((y - lower, upper - y))),
        marker="o",
        capsize=4,
    )
    ax.set(xlabel="ProMatch stage", ylabel="O-frame-unsafe fraction", title="Unsafe durable commitments by stage", xticks=x, ylim=(0, 1))
    _save_plot(fig, plot_dir / "unsafe-fraction-by-stage.png")
    rendered.append("unsafe-fraction-by-stage.png")

    rows = payloads["first-conflict-stage-context"]["rows"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    contexts = [
        label for label in (*CONTEXT_PRIORITY, "none")
        if any(row["first_unsafe_context"] == label for row in rows)
    ]
    bottom = np.zeros(4)
    for context in contexts:
        values = [sum(row["count"] for row in rows if row["first_unsafe_stage"] == stage and row["first_unsafe_context"] == context) for stage in range(1, 5)]
        ax.bar(range(1, 5), values, bottom=bottom, label=context)
        bottom += values
    if int(bottom.sum()) != sum(row["count"] for row in rows):
        raise PolicyAnalysisError("rendered first-conflict stacks do not reconcile")
    ax.set(xlabel="first unsafe stage", ylabel="U0/PU-discordant shots", title="First conflict context (exclusive display label)", xticks=range(1, 5))
    if contexts:
        ax.legend(fontsize=8)
    _save_plot(fig, plot_dir / "first-conflict-stage-context.png")
    rendered.append("first-conflict-stage-context.png")

    payload = payloads["cost-excess-ecdf"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for stage, points in payload["by_stage"].items():
        if points:
            ax.step([point["value"] for point in points], [point["cumulative_fraction"] for point in points], where="post", label=f"stage {stage}")
    tolerance_points = payload["tolerance_tau_k"]["ecdf"]
    max_tolerance = max((point["value"] for point in tolerance_points), default=0.0)
    ax.axvspan(-max_tolerance, max_tolerance, color="0.8", alpha=0.5, label="recorded ±tau_k band")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1e-9)
    ax.set(xlabel="cost excess (symlog)", ylabel="empirical CDF", title="Cost-excess distribution by stage", ylim=(0, 1))
    ax.legend(fontsize=8)
    _save_plot(fig, plot_dir / "cost-excess-ecdf.png")
    rendered.append("cost-excess-ecdf.png")

    payload = payloads["first-safe-action-rank"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    action_rows = [row for row in payload["actions"] if row["terminal_action"] != "censored-invalid"]
    axes[0].bar(range(len(action_rows)), [row["count"] for row in action_rows])
    axes[0].set_xticks(range(len(action_rows)), [row["terminal_action"] for row in action_rows], rotation=25, ha="right")
    axes[0].set(ylabel="unsafe states", title="Terminal action")
    rank_rows = [row for row in payload["ranks"] if row["first_safe_rank"] is not None]
    axes[1].bar([row["first_safe_rank"] for row in rank_rows], [row["count"] for row in rank_rows])
    axes[1].set(xlabel="operational first-safe rank", ylabel="unsafe states", title="First safe alternative")
    _save_plot(fig, plot_dir / "first-safe-action-rank.png")
    rendered.append("first-safe-action-rank.png")

    rows = payloads["stage-transition-matrix"]["rows"]
    columns: list[Any] = [1, 2, 3, 4, "abstain"]
    matrix = np.zeros((4, len(columns)))
    for row in rows:
        matrix[int(row["original_stage"]) - 1, columns.index(row["terminal_stage"])] = row["count"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(columns)), [str(value) for value in columns])
    ax.set_yticks(range(4), [str(stage) for stage in range(1, 5)])
    ax.set(xlabel="first-safe stage / terminal", ylabel="original unsafe stage", title="Original-state counterfactual transition")
    fig.colorbar(image, ax=ax, label="states")
    _save_plot(fig, plot_dir / "stage-transition-matrix.png")
    rendered.append("stage-transition-matrix.png")

    rows = payloads["original-versus-alternative"]["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    metrics = (("decision_weight", "decision weight"), ("path_length", "path length"), ("weight_margin", "local weight margin"), ("events_removed", "immediate events removed"))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for row in rows:
            left, right = row[f"original_{metric}"], row[f"first_safe_{metric}"]
            if left is not None and right is not None:
                ax.plot((0, 1), (left, right), color="0.6", alpha=0.2)
        ax.set_xticks((0, 1), ("original", "first safe"))
        ax.set_ylabel(label)
    fig.suptitle("Original unsafe candidate versus first safe alternative")
    _save_plot(fig, plot_dir / "original-versus-alternative.png")
    rendered.append("original-versus-alternative.png")

    risk = payloads["risk-heatmaps"]["tables"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, rows) in zip(axes.flat, [(name, risk[name]) for name in ("stage_by_window_offset", "stage_by_static_boundary_competition", "domain_hw_by_candidate_multiplicity", "stage_by_margin_decile")]):
        if not rows:
            ax.set_axis_off()
            continue
        keys = [key for key in rows[0] if key not in {"unsafe", "denominator", "unsafe_fraction", "stratum_status"}]
        ys = sorted({row[keys[0]] for row in rows}, key=str)
        xs = sorted({row[keys[1]] for row in rows}, key=str)
        values = np.full((len(ys), len(xs)), np.nan)
        for row in rows:
            values[ys.index(row[keys[0]]), xs.index(row[keys[1]])] = row["unsafe_fraction"]
        ax.imshow(values, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(xs)), [str(value) for value in xs], rotation=45, ha="right")
        ax.set_yticks(range(len(ys)), [str(value) for value in ys])
        ax.set(xlabel=keys[1], ylabel=keys[0], title=name.replace("_", " "))
    fig.suptitle("Descriptive unsafe-fraction heatmaps (not policy thresholds)")
    _save_plot(fig, plot_dir / "risk-heatmaps.png")
    rendered.append("risk-heatmaps.png")

    rows = payloads["disagreement-association"]["rows"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(rows))
    denominator = np.array([row["shots"] for row in rows], dtype=float)
    for name in (
        "prediction_disagreement",
        "regression",
        "recovery",
        "prediction_discordant_both_wrong",
    ):
        values = np.divide([row[name] for row in rows], denominator, out=np.full(len(rows), np.nan), where=denominator != 0)
        ax.plot(x, values, marker="o", label=name)
    ax.set_xticks(x, [row["unsafe_count_bin"] for row in rows])
    ax.set(xlabel="unsafe durable commitments per shot", ylabel="unconditional shot fraction", title="Association with final U0/PU outcome")
    ax.legend(fontsize=8)
    _save_plot(fig, plot_dir / "disagreement-association.png")
    rendered.append("disagreement-association.png")

    rows = payloads["event-relief"]["summary"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    y = np.array([row["R_event"] if row["R_event"] is not None else np.nan for row in rows])
    lower = np.array([row["R_event_bootstrap"]["lower"] if row["R_event_bootstrap"]["lower"] is not None else np.nan for row in rows])
    upper = np.array([row["R_event_bootstrap"]["upper"] if row["R_event_bootstrap"]["upper"] is not None else np.nan for row in rows])
    ax.errorbar(
        x,
        y,
        yerr=np.maximum(0, np.vstack((y - lower, upper - y))),
        fmt="o",
        capsize=4,
    )
    ax.set_xticks(x, [row["arm"] for row in rows], rotation=25, ha="right")
    ax.set(ylabel="R_event (ratio of sums)", title="Durable detector-event relief")
    _save_plot(fig, plot_dir / "event-relief.png")
    rendered.append("event-relief.png")

    metrics = payloads["veto-chain-tails"]["metrics"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, summary in metrics.items():
        if summary["ecdf"]:
            ax.step([row["value"] for row in summary["ecdf"]], [row["cumulative_fraction"] for row in summary["ecdf"]], where="post", label=name)
    ax.set(xlabel="count / elapsed ns", ylabel="empirical CDF", title="Counterfactual and Stage-3 tails", ylim=(0, 1))
    ax.legend(fontsize=8)
    _save_plot(fig, plot_dir / "veto-chain-tails.png")
    rendered.append("veto-chain-tails.png")
    return rendered


def write_policy_analysis(
    corpus: PolicyAuditCorpus,
    analysis: Mapping[str, Any],
    *,
    render_plots: bool = True,
) -> dict[str, Any]:
    """Installs analysis products and writes ANALYSIS_READY last.

    ``COMPLETE`` is deliberately not written here.  It belongs to the later
    casebook-expansion/finalization stage after exhaustive sidecars verify.
    """

    root = corpus.root
    gate_rows = _at(analysis, "tables.fatal_gates", required=True)
    passing_statuses = {"passed-ledger-recomputed", "collector-attested"}
    if (
        not isinstance(gate_rows, list)
        or [row.get("gate") for row in gate_rows if isinstance(row, Mapping)]
        != list(range(1, 19))
        or any(
            not isinstance(row, Mapping)
            or row.get("status") not in passing_statuses
            or not isinstance(row.get("evidence"), Mapping)
            or not row["evidence"].get("authenticated_by")
            for row in gate_rows
        )
    ):
        raise PolicyAnalysisError(
            "analysis has a missing or unauthenticated fatal gate; ANALYSIS_READY refused"
        )
    if (root / "analysis").exists() or (root / "ANALYSIS_READY").exists():
        raise PolicyAnalysisError("analysis output already exists; immutable rerun refused")
    scratch_root = os.environ.get("TMPDIR")
    if not scratch_root:
        raise PolicyAnalysisError("TMPDIR must be set for atomic analysis installation")
    if os.stat(scratch_root).st_dev != os.stat(root).st_dev:
        raise PolicyAnalysisError("TMPDIR and audit root must share a filesystem")
    temporary = Path(tempfile.mkdtemp(prefix="promatch-policy-analysis-", dir=scratch_root))
    installed = False
    try:
        tables_dir = temporary / "tables"
        plot_data_dir = temporary / "plot-data"
        plot_dir = temporary / "plots"
        tables_dir.mkdir()
        plot_data_dir.mkdir()
        plot_dir.mkdir()
        table_hashes = {}
        tables = analysis["tables"]
        if not isinstance(tables, Mapping):
            raise PolicyAnalysisError("analysis tables must be an object")
        for name, table in sorted(tables.items()):
            table_hashes[f"tables/{name}.json"] = _write_canonical_json(
                tables_dir / f"{name}.json", table
            )
        plot_payloads = _plot_payloads(tables)
        plot_data_hashes = {}
        for name, payload in sorted(plot_payloads.items()):
            plot_data_hashes[f"plot-data/{name}.json"] = _write_canonical_json(
                plot_data_dir / f"{name}.json", payload
            )
        rendered = _render_plots(plot_dir, plot_payloads) if render_plots else []
        summary_hash = _write_canonical_json(temporary / "summary.json", analysis)
        report_data = policy_human_report_bytes(analysis)
        (temporary / HUMAN_REPORT_FILE).write_bytes(report_data)
        report_hash = _sha256(report_data)
        manifest = {
            "schema": ANALYSIS_MANIFEST_SCHEMA,
            "experiment_id": analysis["experiment_id"],
            "analysis_sha256": analysis["analysis_sha256"],
            "summary_file_sha256": summary_hash,
            "report_file_sha256": report_hash,
            "source_hashes": corpus.source_hashes,
            "table_file_hashes": dict(sorted(table_hashes.items())),
            "plot_data_file_hashes": dict(sorted(plot_data_hashes.items())),
            "plot_images": [f"plots/{name}" for name in sorted(rendered)],
            "plot_images_scientifically_digested": False,
        }
        manifest_hash = _write_canonical_json(temporary / "manifest.json", manifest)

        casebook_dir = root / "casebook"
        casebook_dir.mkdir(exist_ok=True)
        selection_path = casebook_dir / "selection.json"
        selection_data = canonical_json_bytes(analysis["casebook_selection"]) + b"\n"
        if selection_path.exists():
            if selection_path.read_bytes() != selection_data:
                raise PolicyAnalysisError("existing casebook selection differs")
        else:
            selection_fd, selection_name = tempfile.mkstemp(
                prefix="promatch-policy-casebook-", dir=scratch_root
            )
            os.close(selection_fd)
            selection_temp = Path(selection_name)
            try:
                with selection_temp.open("wb") as file:
                    file.write(selection_data)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(selection_temp, selection_path)
            finally:
                if selection_temp.exists():
                    selection_temp.unlink()

        os.replace(temporary, root / "analysis")
        installed = True
        ready = {
            "schema": ANALYSIS_READY_SCHEMA,
            "experiment_id": analysis["experiment_id"],
            "analysis_manifest_sha256": manifest_hash,
            "casebook_selection_sha256": _sha256(selection_data),
            "report_file_sha256": report_hash,
            "plots_rendered": render_plots,
            "casebook_exhaustive_expansion_required_before_complete": True,
        }
        ready_fd, ready_name = tempfile.mkstemp(
            prefix="promatch-policy-analysis-ready-", dir=scratch_root
        )
        os.close(ready_fd)
        ready_temp = Path(ready_name)
        try:
            with ready_temp.open("wb") as file:
                file.write(canonical_json_bytes(ready) + b"\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(ready_temp, root / "ANALYSIS_READY")
        finally:
            if ready_temp.exists():
                ready_temp.unlink()
        return manifest
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def verify_existing_policy_analysis(
    corpus: PolicyAuditCorpus, analysis: Mapping[str, Any]
) -> dict[str, Any]:
    """Recomputes and authenticates an already-installed offline analysis."""

    root = corpus.root
    analysis_dir = root / "analysis"
    if not analysis_dir.is_dir():
        raise PolicyAnalysisError("audit root has no installed analysis")
    marker = root / "ANALYSIS_READY"
    if not marker.is_file():
        raise PolicyAnalysisError(
            "installed analysis has no ANALYSIS_READY marker; COMPLETE is not an analysis substitute"
        )
    expected_summary_bytes = canonical_json_bytes(analysis) + b"\n"
    summary_path = analysis_dir / "summary.json"
    if not summary_path.is_file() or summary_path.read_bytes() != expected_summary_bytes:
        raise PolicyAnalysisError("installed summary bytes differ from recomputed canonical analysis")
    analysis_without_digest = dict(analysis)
    claimed_analysis_digest = analysis_without_digest.pop("analysis_sha256", None)
    if claimed_analysis_digest != _sha256(canonical_json_bytes(analysis_without_digest)):
        raise PolicyAnalysisError("recomputed analysis self digest is invalid")
    manifest = _load_json(analysis_dir / "manifest.json")
    manifest_fields = {
        "schema",
        "experiment_id",
        "analysis_sha256",
        "summary_file_sha256",
        "report_file_sha256",
        "source_hashes",
        "table_file_hashes",
        "plot_data_file_hashes",
        "plot_images",
        "plot_images_scientifically_digested",
    }
    if set(manifest) != manifest_fields:
        raise PolicyAnalysisError("installed analysis manifest has unexpected fields")
    if manifest.get("schema") != ANALYSIS_MANIFEST_SCHEMA:
        raise PolicyAnalysisError("installed analysis manifest has the wrong schema")
    if manifest.get("experiment_id") != analysis.get("experiment_id"):
        raise PolicyAnalysisError("installed analysis manifest has the wrong experiment identity")
    if manifest.get("analysis_sha256") != claimed_analysis_digest:
        raise PolicyAnalysisError("installed manifest has the wrong analysis digest")
    if manifest.get("summary_file_sha256") != _sha256(expected_summary_bytes):
        raise PolicyAnalysisError("installed analysis summary digest mismatch")
    expected_report_bytes = policy_human_report_bytes(analysis)
    report_path = analysis_dir / HUMAN_REPORT_FILE
    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or report_path.read_bytes() != expected_report_bytes
    ):
        raise PolicyAnalysisError("installed human report bytes differ from recomputation")
    if manifest.get("report_file_sha256") != _sha256(expected_report_bytes):
        raise PolicyAnalysisError("installed human report digest mismatch")
    if manifest.get("source_hashes") != corpus.source_hashes:
        raise PolicyAnalysisError("installed analysis manifest has the wrong source hashes")
    if manifest.get("plot_images_scientifically_digested") is not False:
        raise PolicyAnalysisError("plot images must remain explicitly non-scientific")

    tables = analysis.get("tables")
    if not isinstance(tables, Mapping):
        raise PolicyAnalysisError("recomputed analysis tables must be an object")
    expected_tables = {
        f"tables/{name}.json": canonical_json_bytes(table) + b"\n"
        for name, table in sorted(tables.items())
    }
    plot_payloads = _plot_payloads(tables)
    expected_plot_data = {
        f"plot-data/{name}.json": canonical_json_bytes(payload) + b"\n"
        for name, payload in sorted(plot_payloads.items())
    }
    for group, expected_files in (
        ("table_file_hashes", expected_tables),
        ("plot_data_file_hashes", expected_plot_data),
    ):
        records = manifest.get(group)
        expected_hashes = {
            relative: _sha256(data) for relative, data in expected_files.items()
        }
        if records != expected_hashes:
            raise PolicyAnalysisError(f"installed {group} differs from recomputed artifacts")
        for relative, expected_bytes in expected_files.items():
            path = analysis_dir / relative
            if path.is_symlink() or not path.is_file() or path.read_bytes() != expected_bytes:
                raise PolicyAnalysisError(
                    f"installed analysis bytes differ from recomputation: {relative}"
                )
    plot_images = manifest.get("plot_images")
    if not isinstance(plot_images, list) or any(not isinstance(item, str) for item in plot_images):
        raise PolicyAnalysisError("installed analysis manifest has invalid plot image paths")
    for relative in plot_images:
        raw_path = analysis_dir / relative
        path = raw_path.resolve()
        if analysis_dir.resolve() not in path.parents or raw_path.is_symlink() or not path.is_file():
            raise PolicyAnalysisError(f"installed plot image is missing or unsafe: {relative}")

    ready = _load_json(marker)
    expected_ready_fields = {
        "schema",
        "experiment_id",
        "analysis_manifest_sha256",
        "casebook_selection_sha256",
        "report_file_sha256",
        "plots_rendered",
        "casebook_exhaustive_expansion_required_before_complete",
    }
    if set(ready) != expected_ready_fields:
        raise PolicyAnalysisError("ANALYSIS_READY has unexpected fields")
    if ready.get("schema") != ANALYSIS_READY_SCHEMA:
        raise PolicyAnalysisError("ANALYSIS_READY has the wrong schema")
    if ready.get("experiment_id") != analysis.get("experiment_id"):
        raise PolicyAnalysisError("ANALYSIS_READY has the wrong experiment identity")
    if ready.get("analysis_manifest_sha256") != _sha256(
        (analysis_dir / "manifest.json").read_bytes()
    ):
        raise PolicyAnalysisError("ANALYSIS_READY manifest digest mismatch")
    if ready.get("plots_rendered") is not bool(plot_images):
        raise PolicyAnalysisError("ANALYSIS_READY plot-rendering flag is inconsistent")
    if ready.get("casebook_exhaustive_expansion_required_before_complete") is not True:
        raise PolicyAnalysisError("ANALYSIS_READY omitted the required expansion contract")
    selection = root / "casebook" / "selection.json"
    expected_selection = canonical_json_bytes(analysis["casebook_selection"]) + b"\n"
    if not selection.is_file() or selection.read_bytes() != expected_selection:
        raise PolicyAnalysisError("installed casebook selection differs")
    if ready.get("casebook_selection_sha256") != _sha256(expected_selection):
        raise PolicyAnalysisError("ANALYSIS_READY selection digest mismatch")
    if ready.get("report_file_sha256") != _sha256(expected_report_bytes):
        raise PolicyAnalysisError("ANALYSIS_READY report digest mismatch")
    return manifest
