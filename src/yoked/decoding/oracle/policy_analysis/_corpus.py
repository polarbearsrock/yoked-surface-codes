"""Canonical artifact loading and corpus authentication.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` loads a completed
B1 collection directory, authenticating the ready marker, manifest, canonical
gzip/JSONL shard bytes, worker timing sidecars, and row relationships before
any analysis runs.  It inherits the package's downstream-only contract: it
never imports circuit generation, sampling, matching, or decoding code.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._contract import (
    COLLECTOR_GATE_ATTESTATION_SCHEMA,
    COLLECTOR_GATE_CHECKS,
    EXPECTED_LEDGER_SCHEMAS,
    SHARD_FILES,
    PolicyAnalysisError,
    _sha256,
    _unique_object,
    canonical_json_bytes,
)
from ._fields import _as_nonnegative_int, _at, _one_deep, _record_value


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


def _load_jsonl_gzip(
    path: Path, *, manifest: Mapping[str, Any], root: Path
) -> list[dict[str, Any]]:
    try:
        compressed = path.read_bytes()
        uncompressed = gzip.decompress(compressed)
    except (OSError, EOFError, gzip.BadGzipFile) as ex:
        raise PolicyAnalysisError(f"cannot read gzip ledger {path}") from ex
    if (
        len(compressed) < 10
        or compressed[4:8] != b"\x00\x00\x00\x00"
        or compressed[3] & 0x08
    ):
        raise PolicyAnalysisError(
            f"gzip ledger {path} is not deterministic mtime=0/name-empty"
        )
    relative = path.relative_to(root).as_posix()
    record = _manifest_record(manifest, relative)
    expected_compressed = _record_value(
        record, "compressed_sha256", "gzip_sha256", "sha256"
    )
    expected_uncompressed = _record_value(
        record, "uncompressed_sha256", "content_sha256", "jsonl_sha256"
    )
    if (
        not isinstance(expected_compressed, str)
        or _sha256(compressed) != expected_compressed
    ):
        raise PolicyAnalysisError(f"compressed digest mismatch for {relative}")
    if (
        not isinstance(expected_uncompressed, str)
        or _sha256(uncompressed) != expected_uncompressed
    ):
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


@dataclass(frozen=True)
class PolicyAuditCorpus:
    """Authenticated collection metadata and immutable rows grouped by ledger."""

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
                raise PolicyAnalysisError(
                    f"captured timing collides with ledger field {key}"
                )
            result[key] = _merge_authenticated_timing(result[key], item)
        else:
            result[key] = item
    return result


def load_policy_audit(root: Path) -> PolicyAuditCorpus:
    """Loads and fully authenticates a completed B1 collection directory."""

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
        for name in (
            "experiment.json",
            "config.json",
            "manifest.json",
            "COLLECTION_READY",
        )
    }
    for worker_dir in worker_dirs:
        try:
            worker_id = int(worker_dir.name.removeprefix("worker-"))
        except ValueError as ex:
            raise PolicyAnalysisError(
                f"invalid worker directory {worker_dir.name}"
            ) from ex
        worker_rows: dict[str, list[dict[str, Any]]] = {}
        for kind, filename in SHARD_FILES.items():
            path = worker_dir / filename
            if not path.is_file():
                raise PolicyAnalysisError(f"missing worker artifact {path}")
            source_hashes[path.relative_to(root).as_posix()] = _sha256(
                path.read_bytes()
            )
            loaded = _load_jsonl_gzip(path, manifest=manifest, root=root)
            for row in loaded:
                if _at(row, "experiment_id", required=True) != experiment_id:
                    raise PolicyAnalysisError(f"{kind} row has wrong experiment_id")
                _, row_worker, _ = _identity(row)
                if row_worker != worker_id:
                    raise PolicyAnalysisError(
                        f"{kind} row is in the wrong worker shard"
                    )
                schema = _at(row, "schema", required=True)
                if schema != EXPECTED_LEDGER_SCHEMAS[kind]:
                    raise PolicyAnalysisError(f"{kind} row has invalid schema")
            worker_rows[kind] = loaded

        timing_path = worker_dir / "timing.json"
        if not timing_path.is_file():
            raise PolicyAnalysisError(
                f"missing authenticated worker timing {timing_path}"
            )
        timing = _load_json(timing_path)
        manifest_shards = manifest.get("shards")
        matches = (
            [
                shard
                for shard in manifest_shards
                if isinstance(shard, Mapping)
                and isinstance(shard.get("worker"), Mapping)
                and shard["worker"].get("worker_id") == worker_id
            ]
            if isinstance(manifest_shards, list)
            else []
        )
        relative_timing = timing_path.relative_to(root).as_posix()
        if len(matches) != 1:
            raise PolicyAnalysisError(
                f"manifest must contain exactly one shard for worker {worker_id}"
            )
        shard = matches[0]
        if shard.get("timing_path") != relative_timing:
            raise PolicyAnalysisError(
                f"manifest shard timing_path does not name {relative_timing}"
            )
        if shard.get("timing_sha256") != _sha256(timing_path.read_bytes()):
            raise PolicyAnalysisError(
                f"manifest shard timing_sha256 does not match {relative_timing}"
            )
        if shard.get("nondeterministic_telemetry_paths") != [relative_timing]:
            raise PolicyAnalysisError(
                "manifest shard nondeterministic_telemetry_paths must list exactly "
                "the worker timing file"
            )
        if timing.get("schema") != "promatch-l1-policy-audit-timing-v2":
            raise PolicyAnalysisError(
                "worker timing schema is not promatch-l1-policy-audit-timing-v2"
            )
        if timing.get("worker_id") != worker_id:
            raise PolicyAnalysisError(
                "worker timing worker_id does not match its shard directory"
            )
        if timing.get("scientifically_deterministic") is not False:
            raise PolicyAnalysisError(
                "worker timing scientifically_deterministic must be false"
            )
        if timing.get("excluded_from_bit_exact_ledger_contract") is not True:
            raise PolicyAnalysisError(
                "worker timing excluded_from_bit_exact_ledger_contract must be true"
            )
        captured = timing.get("core_timing_by_ledger")
        if not isinstance(captured, Mapping) or set(captured) != set(SHARD_FILES):
            raise PolicyAnalysisError(
                "worker timing lacks the exact core ledger projection"
            )
        for kind, records in captured.items():
            if not isinstance(records, list):
                raise PolicyAnalysisError(
                    "worker core timing projection must be an array"
                )
            seen: set[int] = set()
            for record in records:
                if not isinstance(record, Mapping):
                    raise PolicyAnalysisError(
                        "worker core timing record must be an object"
                    )
                if set(record) != {
                    "row_index",
                    "worker_shot_index",
                    "global_shot_id",
                    "timing",
                }:
                    raise PolicyAnalysisError(
                        "worker core timing record fields are not exact"
                    )
                index = record.get("row_index")
                if isinstance(index, bool) or not isinstance(index, int):
                    raise PolicyAnalysisError(
                        "worker core timing row_index must be an integer"
                    )
                if index < 0 or index >= len(worker_rows[kind]):
                    raise PolicyAnalysisError(
                        "worker core timing row_index is out of range"
                    )
                if index in seen:
                    raise PolicyAnalysisError(
                        "worker core timing row_index is duplicated"
                    )
                if record.get("worker_shot_index") != worker_rows[kind][index].get(
                    "worker_shot_index"
                ):
                    raise PolicyAnalysisError(
                        "worker core timing worker_shot_index does not match its "
                        "ledger row"
                    )
                if record.get("global_shot_id") != worker_rows[kind][index].get(
                    "global_shot_id"
                ):
                    raise PolicyAnalysisError(
                        "worker core timing global_shot_id does not match its "
                        "ledger row"
                    )
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
                    _at(row, "worker_shot_index", required=True),
                    name="worker_shot_index",
                ),
                _at(row, "physical_input_sha256", required=True),
            )
            if provenance != shot_provenance[_identity(row)]:
                raise PolicyAnalysisError(
                    f"{kind} row has inconsistent shot provenance"
                )

    configured_scientific_shots = _one_deep(config, ("total_shots", "shot_count"))
    if (
        configured_scientific_shots is not None
        and _as_nonnegative_int(
            configured_scientific_shots, name="configured scientific shot count"
        )
        != 20_000
    ):
        raise PolicyAnalysisError(
            "B1 protocol must retain its 20,000-shot scientific schedule"
        )
    mode = manifest.get("mode")
    mode_counts = {"scientific": 20_000, "probe": 100, "smoke": 32}
    if mode not in mode_counts:
        raise PolicyAnalysisError(f"invalid collection mode {mode!r}")
    expected_shots = mode_counts[str(mode)]
    if manifest.get("workers") != 32 or manifest.get("shots") != expected_shots:
        raise PolicyAnalysisError(
            "manifest campaign schedule differs from its frozen mode"
        )
    expected_tail_attestation = {
        "uncapped_counterfactuals": True,
        "censored_states": 0,
        "repeated_same_state_proposal_signatures": 0,
        "worker_timeouts": 0,
        "output_truncations": 0,
    }
    if manifest.get("tail_censor_attestation") != expected_tail_attestation:
        raise PolicyAnalysisError(
            "manifest lacks the exact clean tail/censor attestation"
        )
    if len(worker_dirs) != 32 or len(shots) != expected_shots:
        raise PolicyAnalysisError("ledger campaign schedule differs from its manifest")
    if mode == "scientific":
        schedule = [(worker, 625 * worker, 625) for worker in range(32)]
    elif mode == "probe":
        schedule = [
            (
                worker,
                4 * worker if worker < 4 else 16 + 3 * (worker - 4),
                4 if worker < 4 else 3,
            )
            for worker in range(32)
        ]
    else:
        schedule = [(worker, worker, 1) for worker in range(32)]
    counts = Counter(worker for _, worker, _ in shot_ids)
    for worker, start, count in schedule:
        if counts[worker] != count:
            raise PolicyAnalysisError(
                "worker shot count differs from the frozen mode schedule"
            )
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
