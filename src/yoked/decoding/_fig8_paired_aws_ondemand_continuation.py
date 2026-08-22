"""Audited On-Demand continuation of a frozen AWS Spot Figure-8 campaign.

The original campaign manifest and all existing ledgers remain immutable.  A
separate sibling record freezes the operational boundary and authenticates the
On-Demand host, continuation checkout, and exact set of pre-existing ledgers.
Missing batches continue to use the original experiment ID, seed root, batch
schedule, decoder, and atomic single-writer ledger installation path.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    is_lowercase_hex,
    load_json_strict,
    repo_root,
)
from yoked.decoding import _fig8_paired_aws_sweep as spot
from yoked.decoding._promatch_experiment import (
    _atomic_json_write,
    _canonical_file_hash,
    current_execution_environment,
    current_software_versions,
    repository_state,
)
from yoked.decoding._promatch_stats import BatchSpec, canonical_json_bytes


CONTINUATION_SCHEMA = "yoked.fig8-paired-aws192-ondemand-continuation-v1"
CONTINUATION_KIND = "yoked-fig8b-paired-fixed-shot-aws192-ondemand-continuation"
CONTINUATION_STATUS_SCHEMA = (
    "yoked.fig8-paired-aws192-ondemand-continuation-status-v1"
)
EXPECTED_LIFECYCLE = "on-demand"

CONTINUATION_SOURCE_PATHS = {
    "src/yoked/decoding/_fig8_paired_aws_ondemand_continuation.py",
    "tools/benchmark_fig8_paired_aws_ondemand",
    "aws/run_fig8_paired_ondemand",
}

CONTINUATION_FIELDS = {
    "schema",
    "kind",
    "status",
    "frozen",
    "claim_bearing",
    "created_utc",
    "campaign_root",
    "campaign_sha256",
    "campaign_experiment_id",
    "campaign_repository_commit",
    "continuation_repository_commit",
    "clean_worktree",
    "software_versions",
    "execution_environment",
    "source_hashes",
    "spot_aws_identity",
    "ondemand_aws_identity",
    "ondemand_instance_id",
    "ondemand_ami_id",
    "spot_numa_topology",
    "ondemand_numa_topology",
    "worker_layout",
    "processes",
    "threads_per_process",
    "baseline_completed_batches",
    "baseline_completed_shots",
    "missing_batches_at_prepare",
    "baseline_ledger_hashes",
    "continuation_id",
}


def continuation_record_path(directory: Path) -> Path:
    campaign = directory.absolute()
    return campaign.parent / f".{campaign.name}.ondemand-continuation-v1.json"


def _continuation_id(record: Mapping[str, Any]) -> str:
    content = dict(record)
    content.pop("continuation_id", None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _validate_original_sources(campaign: Mapping[str, Any], root: Path) -> None:
    for relative, expected in campaign["source_hashes"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"invalid frozen campaign source path {relative!r}")
        if _canonical_file_hash(path) != expected:
            raise ValueError(f"frozen campaign source hash differs for {relative}")


def _continuation_source_hashes(root: Path) -> dict[str, str]:
    missing = sorted(
        relative
        for relative in CONTINUATION_SOURCE_PATHS
        if not (root / relative).is_file()
    )
    if missing:
        raise ValueError(f"continuation source paths are missing: {missing}")
    return {
        relative: _canonical_file_hash((root / relative).resolve())
        for relative in sorted(CONTINUATION_SOURCE_PATHS)
    }


def _validate_repository_for_prepare(
    campaign: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("On-Demand continuation requires a clean worktree")
    if not _git_is_ancestor(
        root, campaign["repository_commit"], state["repository_commit"]
    ):
        raise ValueError(
            "frozen campaign commit is not an ancestor of continuation HEAD"
        )
    if current_software_versions() != campaign["software_versions"]:
        raise ValueError("continuation software versions differ from frozen campaign")
    _validate_original_sources(campaign, root)
    return state


def _validate_ondemand_identity(
    identity: Mapping[str, Any], *, frozen: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or set(identity) != spot.AWS_IDENTITY_FIELDS:
        raise ValueError("On-Demand AWS identity has incorrect fields")
    if identity.get("lifecycle") != EXPECTED_LIFECYCLE:
        raise ValueError("On-Demand continuation requires lifecycle='on-demand'")
    for key in ("instance_type", "region", "availability_zone"):
        if identity.get(key) != frozen.get(key):
            raise ValueError(f"On-Demand continuation changed AWS identity field {key}")
    return dict(identity)


def _validate_compatible_topology(
    topology: Mapping[str, Any], *, frozen: Mapping[str, Any]
) -> dict[str, Any]:
    current = spot._validate_numa_topology(topology)
    for key in ("visible_cpus", "physical_core_count", "threads_per_core"):
        if current.get(key) != frozen.get(key):
            raise ValueError(
                f"On-Demand continuation changed NUMA topology field {key}"
            )
    old_nodes = frozen.get("nodes")
    new_nodes = current.get("nodes")
    if not isinstance(old_nodes, list) or len(old_nodes) != len(new_nodes):
        raise ValueError("On-Demand continuation changed NUMA node count")
    for old, new in zip(old_nodes, new_nodes, strict=True):
        for key in ("node_id", "cpus"):
            if old.get(key) != new.get(key):
                raise ValueError(
                    "On-Demand continuation changed NUMA node "
                    f"{new.get('node_id')} {key}"
                )
    # Total and per-node usable-memory reports may differ slightly between two
    # otherwise identical c8a.48xlarge hosts.  _validate_numa_topology has
    # already enforced the frozen 350/175 GiB safety minima.
    return current


def _current_host(campaign: Mapping[str, Any]) -> dict[str, Any]:
    identity = _validate_ondemand_identity(
        spot.current_aws_identity(), frozen=campaign["aws_identity"]
    )
    topology = _validate_compatible_topology(
        spot.current_numa_topology(), frozen=campaign["numa_topology"]
    )
    return {
        "aws_identity": identity,
        "instance_id": spot._imds_get("instance-id"),
        "ami_id": spot._imds_get("ami-id"),
        "numa_topology": topology,
        "worker_layout": spot._worker_layout(topology),
    }


def _baseline_ledger_hashes(
    directory: Path,
    campaign: Mapping[str, Any],
) -> tuple[dict[str, str], int, int]:
    rows, collection = spot._validate_existing_ledgers(directory, campaign)
    hashes: dict[str, str] = {}
    shots = 0
    for (cell_id, batch_id), row in sorted(rows.items()):
        path = spot.base._ledger_path(
            collection, cell_id=cell_id, batch_id=batch_id
        )
        relative = path.relative_to(directory.absolute()).as_posix()
        hashes[relative] = _canonical_file_hash(path)
        shots += int(row["batch"]["shots"])
    return hashes, len(rows), shots


def _validate_baseline_ledgers(directory: Path, record: Mapping[str, Any]) -> None:
    declared = record["baseline_ledger_hashes"]
    for relative, expected in declared.items():
        candidate = directory.absolute() / relative
        path = candidate.resolve()
        if (
            candidate.is_symlink()
            or directory.absolute() not in path.parents
            or not path.is_file()
        ):
            raise ValueError(f"baseline ledger is missing: {relative}")
        if _canonical_file_hash(path) != expected:
            raise ValueError(
                f"baseline ledger changed after continuation freeze: {relative}"
            )


def validate_continuation(
    record: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    directory: Path,
) -> str:
    canonical_json_bytes(record)
    if set(record) != CONTINUATION_FIELDS:
        raise ValueError("On-Demand continuation record has incorrect fields")
    fixed = {
        "schema": CONTINUATION_SCHEMA,
        "kind": CONTINUATION_KIND,
        "status": "FROZEN",
        "frozen": True,
        "claim_bearing": False,
        "clean_worktree": True,
        "processes": spot.REQUIRED_PROCESSES,
        "threads_per_process": 1,
    }
    for key, expected in fixed.items():
        if record.get(key) != expected:
            raise ValueError(f"On-Demand continuation has invalid frozen {key}")
    if record.get("campaign_root") != str(directory.absolute()):
        raise ValueError("On-Demand continuation campaign root changed")
    if record.get("campaign_sha256") != _canonical_file_hash(
        directory.absolute() / "campaign.json"
    ):
        raise ValueError("frozen campaign.json changed after continuation freeze")
    if record.get("campaign_experiment_id") != campaign["experiment_id"]:
        raise ValueError("On-Demand continuation campaign experiment ID changed")
    if record.get("campaign_repository_commit") != campaign["repository_commit"]:
        raise ValueError("On-Demand continuation campaign commit changed")
    if record.get("spot_aws_identity") != campaign["aws_identity"]:
        raise ValueError("On-Demand continuation changed frozen Spot identity")
    if record.get("spot_numa_topology") != campaign["numa_topology"]:
        raise ValueError("On-Demand continuation changed frozen Spot topology")
    _validate_ondemand_identity(
        record["ondemand_aws_identity"], frozen=campaign["aws_identity"]
    )
    topology = _validate_compatible_topology(
        record["ondemand_numa_topology"], frozen=campaign["numa_topology"]
    )
    if record.get("worker_layout") != spot._worker_layout(topology):
        raise ValueError("On-Demand continuation worker layout is invalid")
    if record.get("software_versions") != campaign["software_versions"]:
        raise ValueError("On-Demand continuation software differs from campaign")
    if record.get("execution_environment") != campaign["execution_environment"]:
        raise ValueError("On-Demand continuation execution environment differs")
    for name in ("created_utc", "ondemand_instance_id", "ondemand_ami_id"):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f"On-Demand continuation {name} must be nonempty")
    commit = record.get("continuation_repository_commit")
    if not isinstance(commit, str) or not is_lowercase_hex(commit):
        raise ValueError("On-Demand continuation repository commit is invalid")
    source_hashes = record.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != CONTINUATION_SOURCE_PATHS
    ):
        raise ValueError("On-Demand continuation source hashes are incomplete")
    for relative, digest in source_hashes.items():
        if not is_lowercase_hex(digest, length=64):
            raise ValueError(f"invalid continuation source hash for {relative}")
    ledger_hashes = record.get("baseline_ledger_hashes")
    if not isinstance(ledger_hashes, Mapping) or not ledger_hashes:
        raise ValueError("On-Demand continuation baseline ledger hashes are empty")
    for relative, digest in ledger_hashes.items():
        parts = Path(relative).parts
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in parts
            or parts[:2] != ("collection", "batches")
            or not is_lowercase_hex(digest, length=64)
        ):
            raise ValueError("On-Demand continuation has an invalid baseline ledger")
    count = record.get("baseline_completed_batches")
    shots = record.get("baseline_completed_shots")
    missing = record.get("missing_batches_at_prepare")
    for name, value in (("batches", count), ("shots", shots), ("missing", missing)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"On-Demand continuation {name} must be positive")
    if count != len(ledger_hashes):
        raise ValueError("On-Demand continuation baseline batch count is inconsistent")
    allowed_ledgers: dict[str, int] = {}
    for cell_id, schedule in campaign["cell_batch_schedules"].items():
        for raw_batch in schedule:
            batch = BatchSpec.from_json(raw_batch)
            relative = (
                Path("collection")
                / "batches"
                / cell_id
                / f"batch-{batch.batch_id:08d}.json"
            ).as_posix()
            allowed_ledgers[relative] = batch.shots
    if not set(ledger_hashes) <= set(allowed_ledgers):
        raise ValueError("On-Demand continuation baseline includes unknown ledgers")
    if shots != sum(allowed_ledgers[path] for path in ledger_hashes):
        raise ValueError("On-Demand continuation baseline shot count is inconsistent")
    expected_batches = sum(
        len(rows) for rows in campaign["cell_batch_schedules"].values()
    )
    if count + missing != expected_batches:
        raise ValueError("On-Demand continuation missing-batch count is inconsistent")
    identifier = _continuation_id(record)
    if record.get("continuation_id") != identifier:
        raise ValueError("On-Demand continuation ID does not match canonical content")
    return identifier


def load_continuation(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = spot.load_campaign(directory)
    record = load_json_strict(
        continuation_record_path(directory),
        description="AWS On-Demand continuation record",
    )
    validate_continuation(record, campaign=campaign, directory=directory)
    return campaign, record


def prepare_continuation(
    directory: Path, *, root: Path | None = None
) -> dict[str, Any]:
    directory = directory.absolute()
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    record_path = continuation_record_path(directory)
    with spot._exclusive_campaign_lock(directory):
        if record_path.exists() or record_path.is_symlink():
            raise ValueError(f"continuation record already exists: {record_path}")
        campaign = spot.load_campaign(directory)
        state = _validate_repository_for_prepare(campaign, root=root)
        environment = current_execution_environment()
        if environment != campaign["execution_environment"]:
            raise ValueError("continuation execution environment differs from campaign")
        host = _current_host(campaign)
        ledger_hashes, completed_batches, completed_shots = _baseline_ledger_hashes(
            directory, campaign
        )
        expected_batches = sum(
            len(rows) for rows in campaign["cell_batch_schedules"].values()
        )
        missing = expected_batches - completed_batches
        if completed_batches <= 0 or missing <= 0:
            raise ValueError(
                "On-Demand continuation requires a nonempty partial campaign"
            )
        record: dict[str, Any] = {
            "schema": CONTINUATION_SCHEMA,
            "kind": CONTINUATION_KIND,
            "status": "FROZEN",
            "frozen": True,
            "claim_bearing": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_root": str(directory),
            "campaign_sha256": _canonical_file_hash(directory / "campaign.json"),
            "campaign_experiment_id": campaign["experiment_id"],
            "campaign_repository_commit": campaign["repository_commit"],
            "continuation_repository_commit": state["repository_commit"],
            "clean_worktree": True,
            "software_versions": current_software_versions(),
            "execution_environment": environment,
            "source_hashes": _continuation_source_hashes(root),
            "spot_aws_identity": campaign["aws_identity"],
            "ondemand_aws_identity": host["aws_identity"],
            "ondemand_instance_id": host["instance_id"],
            "ondemand_ami_id": host["ami_id"],
            "spot_numa_topology": campaign["numa_topology"],
            "ondemand_numa_topology": host["numa_topology"],
            "worker_layout": host["worker_layout"],
            "processes": spot.REQUIRED_PROCESSES,
            "threads_per_process": 1,
            "baseline_completed_batches": completed_batches,
            "baseline_completed_shots": completed_shots,
            "missing_batches_at_prepare": missing,
            "baseline_ledger_hashes": ledger_hashes,
        }
        record["continuation_id"] = _continuation_id(record)
        validate_continuation(record, campaign=campaign, directory=directory)
        _atomic_json_write(record_path, record)
    return record


def _validate_runtime(
    directory: Path,
    *,
    processes: int,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if processes != spot.REQUIRED_PROCESSES:
        raise ValueError("On-Demand continuation requires exactly 192 processes")
    if os.environ.get("MAX_ERRORS") is not None:
        raise ValueError("MAX_ERRORS is forbidden for fixed-shot continuation")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise ValueError(f"{name} must be exactly 1 before continuation")
    campaign, record = load_continuation(directory)
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("On-Demand continuation requires a clean worktree")
    if state["repository_commit"] != record["continuation_repository_commit"]:
        raise ValueError("continuation repository commit differs from frozen record")
    if current_software_versions() != record["software_versions"]:
        raise ValueError("continuation software versions differ from frozen record")
    if current_execution_environment() != record["execution_environment"]:
        raise ValueError(
            "continuation execution environment differs from frozen record"
        )
    _validate_original_sources(campaign, root)
    if _continuation_source_hashes(root) != record["source_hashes"]:
        raise ValueError("continuation source hashes differ from frozen record")
    host = _current_host(campaign)
    if host["aws_identity"] != record["ondemand_aws_identity"]:
        raise ValueError("continuation AWS identity differs from frozen record")
    if host["instance_id"] != record["ondemand_instance_id"]:
        raise ValueError("continuation EC2 instance differs from frozen record")
    if host["ami_id"] != record["ondemand_ami_id"]:
        raise ValueError("continuation AMI differs from frozen record")
    if host["numa_topology"] != record["ondemand_numa_topology"]:
        raise ValueError("continuation NUMA topology differs from frozen record")
    _validate_baseline_ledgers(directory, record)
    spot._validate_existing_ledgers(directory, campaign)
    return campaign, record


def _continuation_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "continuation_id": record["continuation_id"],
        "baseline_completed_batches": record["baseline_completed_batches"],
        "baseline_completed_shots": record["baseline_completed_shots"],
        "missing_batches_at_prepare": record["missing_batches_at_prepare"],
        "spot_lifecycle": record["spot_aws_identity"]["lifecycle"],
        "continuation_lifecycle": record["ondemand_aws_identity"]["lifecycle"],
        "ondemand_instance_id": record["ondemand_instance_id"],
    }


def continuation_status(directory: Path) -> dict[str, Any]:
    campaign, record = load_continuation(directory)
    _validate_baseline_ledgers(directory, record)
    status = spot.campaign_status(directory)
    completed = status["totals"]["completed_batches"]
    return {
        "schema": CONTINUATION_STATUS_SCHEMA,
        "experiment_id": campaign["experiment_id"],
        "complete": status["complete"],
        "totals": status["totals"],
        "cells": status["cells"],
        "continuation": {
            **_continuation_summary(record),
            "completed_after_prepare": completed
            - record["baseline_completed_batches"],
        },
    }


def run_continuation(
    directory: Path,
    *,
    processes: int,
    root: Path | None = None,
) -> dict[str, Any]:
    directory = directory.absolute()
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    with spot._exclusive_campaign_lock(directory):
        campaign, record = _validate_runtime(
            directory, processes=processes, root=root
        )
        original_validator = spot._validate_runtime

        def continuation_guard(
            candidate: Mapping[str, Any], *, processes: int, root: Path
        ) -> None:
            if processes != spot.REQUIRED_PROCESSES:
                raise ValueError("continuation worker count changed")
            if candidate.get("experiment_id") != campaign["experiment_id"]:
                raise ValueError("continuation campaign changed during collection")

        # The original collector is intentionally reused byte-for-byte.  Its
        # Spot-only runtime gate is replaced only after the stronger
        # continuation validation above, while the campaign lock is held.
        spot._validate_runtime = continuation_guard
        try:
            status = spot._run_campaign_locked(
                directory, processes=processes, root=root
            )
        finally:
            spot._validate_runtime = original_validator
    return {
        **status,
        "continuation": _continuation_summary(record),
    }


__all__ = [
    "CONTINUATION_KIND",
    "CONTINUATION_SCHEMA",
    "CONTINUATION_STATUS_SCHEMA",
    "continuation_record_path",
    "continuation_status",
    "load_continuation",
    "prepare_continuation",
    "run_continuation",
    "validate_continuation",
]
