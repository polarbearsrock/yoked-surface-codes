"""AWS c8a.48xlarge execution profile for the paired Figure-8b sweep.

The scientific cell grid, paired sampling, seeds, ledger rows, and residual
decoder are the same as :mod:`_fig8_paired_sweep`.  This module freezes a
separate campaign type because it deliberately uses 192 workers: two pools of
96 workers, each pinned to one NUMA node.  Only the parent process validates
and installs ledgers, preserving the original single-writer resume contract.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import multiprocessing
import os
from pathlib import Path
import secrets
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.request

from yoked.decoding import _fig8_paired_sweep as base
from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    load_json_strict,
    repo_root,
)
from yoked.decoding._promatch_experiment import (
    _WORKER_CACHE,
    _atomic_json_write,
    _canonical_file_hash,
    _collection_task,
    _validate_ledger_row,
    _worker_collect,
    configure_single_thread_runtime,
    current_execution_environment,
    current_software_versions,
    prepare_cell,
    repository_state,
)
from yoked.decoding._promatch_stats import (
    BatchSpec,
    canonical_json_bytes,
    manifest_experiment_id,
)


CAMPAIGN_SCHEMA = "yoked.fig8-paired-aws192-sweep-campaign-v1"
CAMPAIGN_KIND = "yoked-fig8b-paired-fixed-shot-aws192-sweep"
STATUS_SCHEMA = "yoked.fig8-paired-aws192-sweep-status-v1"
HOST_CHECK_SCHEMA = "yoked.aws-c8a-48xlarge-host-check-v1"
PHASE = base.PHASE
SAMPLE_BATCH_SIZE = base.SAMPLE_BATCH_SIZE
MAX_SHOTS_PER_CELL = base.MAX_SHOTS_PER_CELL
MAX_SI1000_P = base.MAX_SI1000_P
GRID = base.GRID
DEM_OPTIONS = base.DEM_OPTIONS
DECODER = base.DECODER
REPLAY_POLICY = base.REPLAY_POLICY

REQUIRED_PROCESSES = 192
NUMA_POOL_COUNT = 2
PROCESSES_PER_POOL = 96
MAX_PENDING_PER_POOL = 2 * PROCESSES_PER_POOL
EXPECTED_INSTANCE_TYPE = "c8a.48xlarge"
EXPECTED_REGION = "us-east-1"
EXPECTED_LIFECYCLE = "spot"
MIN_MEMORY_KIB = 350 * 1024 * 1024
MIN_NODE_MEMORY_KIB = 175 * 1024 * 1024

REQUIRED_AWS_SOURCE_PATHS = {
    "src/yoked/decoding/_fig8_paired_aws_sweep.py",
    "tools/benchmark_fig8_paired_aws",
    "tools/plot_fig8_paired",
    "aws/activate_environment",
    "aws/run_fig8_paired",
}
CAMPAIGN_FIELDS = base.CAMPAIGN_FIELDS | {
    "aws_identity",
    "numa_topology",
    "worker_layout",
}
AWS_IDENTITY_FIELDS = {
    "instance_type",
    "region",
    "availability_zone",
    "lifecycle",
}
NUMA_TOPOLOGY_FIELDS = {
    "visible_cpus",
    "physical_core_count",
    "threads_per_core",
    "memory_total_kib",
    "nodes",
}
NUMA_NODE_FIELDS = {"node_id", "cpus", "memory_total_kib"}


def _imds_get(path: str, *, timeout: float = 2.0) -> str:
    """Read one EC2 identity value using IMDSv2, failing closed."""

    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        data=b"",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(token_request, timeout=timeout) as response:
            token = response.read().decode("utf-8").strip()
        value_request = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(value_request, timeout=timeout) as response:
            value = response.read().decode("utf-8").strip()
    except (OSError, urllib.error.URLError) as ex:
        raise RuntimeError(f"unable to read EC2 IMDSv2 metadata {path!r}") from ex
    if not token or not value:
        raise RuntimeError(f"EC2 IMDSv2 metadata {path!r} is empty")
    return value


def current_aws_identity() -> dict[str, str]:
    return {
        "instance_type": _imds_get("instance-type"),
        "region": _imds_get("placement/region"),
        "availability_zone": _imds_get("placement/availability-zone"),
        "lifecycle": _imds_get("instance-life-cycle"),
    }


def _parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for piece in value.strip().split(","):
        if not piece:
            continue
        bounds = piece.split("-", 1)
        try:
            start = int(bounds[0])
            stop = int(bounds[-1])
        except ValueError as ex:
            raise ValueError(f"invalid Linux CPU list {value!r}") from ex
        if start < 0 or stop < start:
            raise ValueError(f"invalid Linux CPU range {piece!r}")
        cpus.update(range(start, stop + 1))
    if not cpus:
        raise ValueError("Linux CPU list must not be empty")
    return sorted(cpus)


def _read_memory_total_kib(path: Path, *, prefix: str | None = None) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as ex:
        raise RuntimeError(f"unable to read memory topology from {path}") from ex
    for line in lines:
        candidate = line
        if prefix is not None:
            marker = f"{prefix} MemTotal:"
            if marker not in line:
                continue
            candidate = line.split(marker, 1)[1]
        elif not line.startswith("MemTotal:"):
            continue
        fields = candidate.replace("MemTotal:", "", 1).split()
        if len(fields) >= 2 and fields[1] == "kB":
            try:
                result = int(fields[0])
            except ValueError as ex:
                raise RuntimeError(f"invalid MemTotal in {path}") from ex
            if result > 0:
                return result
    raise RuntimeError(f"missing positive MemTotal in {path}")


def current_numa_topology(
    *, sys_root: Path = Path("/sys"), proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as ex:
        raise RuntimeError("AWS runner requires Linux sched_getaffinity") from ex

    nodes: list[dict[str, Any]] = []
    node_root = sys_root / "devices/system/node"
    for path in sorted(
        node_root.glob("node[0-9]*"), key=lambda item: int(item.name[4:])
    ):
        node_id = int(path.name[4:])
        try:
            cpus = _parse_cpu_list(
                path.joinpath("cpulist").read_text(encoding="utf-8")
            )
        except OSError as ex:
            raise RuntimeError(f"unable to read CPU list for NUMA node {node_id}") from ex
        nodes.append(
            {
                "node_id": node_id,
                "cpus": cpus,
                "memory_total_kib": _read_memory_total_kib(
                    path / "meminfo", prefix=f"Node {node_id}"
                ),
            }
        )

    core_counts: dict[tuple[int, int], int] = {}
    for cpu in affinity:
        topology = sys_root / f"devices/system/cpu/cpu{cpu}/topology"
        try:
            package = int(
                topology.joinpath("physical_package_id").read_text().strip()
            )
            core = int(topology.joinpath("core_id").read_text().strip())
        except (OSError, ValueError) as ex:
            raise RuntimeError(f"unable to read topology for CPU {cpu}") from ex
        key = (package, core)
        core_counts[key] = core_counts.get(key, 0) + 1
    thread_counts = set(core_counts.values())
    threads_per_core = next(iter(thread_counts)) if len(thread_counts) == 1 else 0
    return {
        "visible_cpus": affinity,
        "physical_core_count": len(core_counts),
        "threads_per_core": threads_per_core,
        "memory_total_kib": _read_memory_total_kib(proc_root / "meminfo"),
        "nodes": nodes,
    }


def _worker_layout(topology: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total_workers": REQUIRED_PROCESSES,
        "pool_count": NUMA_POOL_COUNT,
        "workers_per_pool": PROCESSES_PER_POOL,
        "task_partition": "global_batch_id_mod_pool_count",
        "worker_affinity": "sched_setaffinity_before_cell_compile",
        "memory_policy": "linux_default_first_touch_after_worker_affinity",
        "pools": [
            {
                "pool_id": node["node_id"],
                "node_id": node["node_id"],
                "workers": PROCESSES_PER_POOL,
                "cpus": list(node["cpus"]),
            }
            for node in topology["nodes"]
        ],
    }


def _validate_aws_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or set(identity) != AWS_IDENTITY_FIELDS:
        raise ValueError("AWS identity has incorrect fields")
    fixed = {
        "instance_type": EXPECTED_INSTANCE_TYPE,
        "region": EXPECTED_REGION,
        "lifecycle": EXPECTED_LIFECYCLE,
    }
    for key, expected in fixed.items():
        if identity.get(key) != expected:
            raise ValueError(f"AWS runner requires {key}={expected!r}")
    zone = identity.get("availability_zone")
    if not isinstance(zone, str) or not zone.startswith(EXPECTED_REGION):
        raise ValueError("AWS availability zone must belong to us-east-1")
    return dict(identity)


def _validate_numa_topology(topology: Any) -> dict[str, Any]:
    if not isinstance(topology, Mapping) or set(topology) != NUMA_TOPOLOGY_FIELDS:
        raise ValueError("NUMA topology has incorrect fields")
    cpus = topology.get("visible_cpus")
    if cpus != list(range(REQUIRED_PROCESSES)):
        raise ValueError("AWS runner requires exclusive visibility of CPUs 0-191")
    if topology.get("physical_core_count") != REQUIRED_PROCESSES:
        raise ValueError("AWS runner requires exactly 192 physical cores")
    if topology.get("threads_per_core") != 1:
        raise ValueError("AWS runner requires exactly one hardware thread per core")
    memory = topology.get("memory_total_kib")
    if (
        isinstance(memory, bool)
        or not isinstance(memory, int)
        or memory < MIN_MEMORY_KIB
    ):
        raise ValueError("AWS runner requires at least 350 GiB of visible memory")
    nodes = topology.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != NUMA_POOL_COUNT:
        raise ValueError("AWS runner requires exactly two NUMA nodes")
    union: set[int] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping) or set(node) != NUMA_NODE_FIELDS:
            raise ValueError("NUMA node has incorrect fields")
        if node.get("node_id") != index:
            raise ValueError("NUMA nodes must be ordered and numbered 0, 1")
        node_cpus = node.get("cpus")
        expected = list(
            range(index * PROCESSES_PER_POOL, (index + 1) * PROCESSES_PER_POOL)
        )
        if node_cpus != expected:
            raise ValueError(f"NUMA node {index} must expose its contiguous 96-CPU set")
        node_memory = node.get("memory_total_kib")
        if (
            isinstance(node_memory, bool)
            or not isinstance(node_memory, int)
            or node_memory < MIN_NODE_MEMORY_KIB
        ):
            raise ValueError("each NUMA node must report at least 175 GiB of memory")
        if union.intersection(node_cpus):
            raise ValueError("NUMA CPU sets overlap")
        union.update(node_cpus)
    if sorted(union) != cpus:
        raise ValueError("NUMA CPU sets do not cover the visible affinity")
    return dict(topology)


def validate_aws_host(
    *,
    identity: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = current_aws_identity() if identity is None else identity
    topology = current_numa_topology() if topology is None else topology
    valid_identity = _validate_aws_identity(identity)
    valid_topology = _validate_numa_topology(topology)
    return {
        "schema": HOST_CHECK_SCHEMA,
        "valid": True,
        "aws_identity": valid_identity,
        "numa_topology": valid_topology,
        "worker_layout": _worker_layout(valid_topology),
    }


def _source_paths(root: Path) -> list[str]:
    paths = set(base._source_paths(root))
    paths.update(REQUIRED_AWS_SOURCE_PATHS)
    missing = sorted(relative for relative in paths if not (root / relative).is_file())
    if missing:
        raise ValueError(f"AWS campaign source paths are missing: {missing}")
    return sorted(paths)


def create_campaign(
    out: Path, *, p: float, shots_per_cell: int, root: Path | None = None
) -> dict[str, Any]:
    p = base._validate_probability(p)
    shots_per_cell = base._validate_shots(shots_per_cell)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("campaign creation requires a clean worktree")
    host = validate_aws_host()
    destination = base._validate_new_campaign_root(out)
    cells = base._unfrozen_cells(p)
    populated: list[dict[str, Any]] = []
    for cell in cells:
        prepared = prepare_cell(
            cell, decoder_config=DECODER, dem_options=DEM_OPTIONS, verify_hashes=False
        )
        populated.append({**cell, **prepared.provenance})
        del prepared
    expected, schedules = base._build_schedules(
        populated, shots_per_cell=shots_per_cell
    )
    source_hashes = {
        relative: _canonical_file_hash((root / relative).resolve())
        for relative in _source_paths(root)
    }
    campaign: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "kind": CAMPAIGN_KIND,
        "status": "FROZEN",
        "frozen": True,
        "claim_bearing": False,
        "phase": PHASE,
        "repository_commit": state["repository_commit"],
        "clean_worktree": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": current_software_versions(),
        "execution_environment": current_execution_environment(),
        "source_hashes": source_hashes,
        "processes": REQUIRED_PROCESSES,
        "threads_per_process": 1,
        "sample_batch_size": SAMPLE_BATCH_SIZE,
        "shots_per_cell": shots_per_cell,
        "p": p,
        "seed_derivation": base.SEED_DERIVATION,
        "sampler_seed_root": secrets.token_hex(32),
        "dem_options": dict(DEM_OPTIONS),
        "decoder": dict(DECODER),
        "grid": dict(GRID),
        "expected_shots_by_cell": expected,
        "cell_batch_schedules": schedules,
        "cells": populated,
        "aws_identity": host["aws_identity"],
        "numa_topology": host["numa_topology"],
        "worker_layout": host["worker_layout"],
    }
    campaign["experiment_id"] = manifest_experiment_id(campaign)
    validate_campaign(campaign)
    destination.mkdir(parents=True)
    (destination / "collection").mkdir()
    _atomic_json_write(destination / "campaign.json", campaign)
    return campaign


def _base_projection(campaign: Mapping[str, Any]) -> dict[str, Any]:
    projected = {key: campaign[key] for key in base.CAMPAIGN_FIELDS if key in campaign}
    projected["schema"] = base.CAMPAIGN_SCHEMA
    projected["kind"] = base.CAMPAIGN_KIND
    projected["processes"] = base.REQUIRED_PROCESSES
    projected["experiment_id"] = manifest_experiment_id(projected)
    return projected


def validate_campaign(campaign: Mapping[str, Any]) -> str:
    canonical_json_bytes(campaign)
    if set(campaign) != CAMPAIGN_FIELDS:
        raise ValueError("AWS campaign.json has incorrect top-level fields")
    fixed = {
        "schema": CAMPAIGN_SCHEMA,
        "kind": CAMPAIGN_KIND,
        "processes": REQUIRED_PROCESSES,
        "threads_per_process": 1,
    }
    for key, expected in fixed.items():
        if campaign.get(key) != expected:
            raise ValueError(f"AWS campaign has invalid frozen {key}")
    base.validate_campaign(_base_projection(campaign))
    _validate_aws_identity(campaign.get("aws_identity"))
    topology = _validate_numa_topology(campaign.get("numa_topology"))
    if campaign.get("worker_layout") != _worker_layout(topology):
        raise ValueError("AWS campaign worker layout does not match NUMA topology")
    missing = REQUIRED_AWS_SOURCE_PATHS - set(campaign["source_hashes"])
    if missing:
        raise ValueError(
            "AWS campaign source_hashes omits required sources: "
            f"{sorted(missing)}"
        )
    experiment_id = manifest_experiment_id(campaign)
    if campaign.get("experiment_id") != experiment_id:
        raise ValueError("AWS campaign experiment_id does not match canonical content")
    return experiment_id


def load_campaign(directory: Path) -> dict[str, Any]:
    campaign = load_json_strict(
        directory.absolute() / "campaign.json", description="AWS Figure-8 campaign"
    )
    validate_campaign(campaign)
    return campaign


def validate_analysis_runtime(
    campaign: Mapping[str, Any], *, root: Path | None = None
) -> None:
    validate_campaign(campaign)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("AWS Figure-8 analysis requires a clean worktree")
    if state["repository_commit"] != campaign["repository_commit"]:
        raise ValueError("analysis repository commit differs from campaign")
    if current_software_versions() != campaign["software_versions"]:
        raise ValueError("analysis software versions differ from campaign")
    for relative, expected in campaign["source_hashes"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"invalid analysis source path {relative!r}")
        if _canonical_file_hash(path) != expected:
            raise ValueError(f"analysis source hash differs for {relative}")


# Descriptive alias used by tooling that treats host-independent analysis as an
# environment check.  Keep the established runtime name for API parity.
validate_analysis_environment = validate_analysis_runtime


def _validate_runtime(
    campaign: Mapping[str, Any], *, processes: int, root: Path
) -> None:
    if processes != REQUIRED_PROCESSES:
        raise ValueError(
            f"AWS Figure-8 campaign requires exactly {REQUIRED_PROCESSES} processes"
        )
    if os.environ.get("MAX_ERRORS") is not None:
        raise ValueError("MAX_ERRORS is forbidden for fixed-shot Figure-8 collection")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise ValueError(f"{name} must be exactly 1 before Figure-8 collection")
    validate_analysis_runtime(campaign, root=root)
    host = validate_aws_host()
    if host["aws_identity"] != campaign["aws_identity"]:
        raise ValueError("runtime AWS identity differs from campaign")
    if host["numa_topology"] != campaign["numa_topology"]:
        raise ValueError("runtime NUMA topology differs from campaign")
    if current_execution_environment() != campaign["execution_environment"]:
        raise ValueError("runtime execution environment differs from campaign")


def _validate_existing_ledgers(
    directory: Path, campaign: Mapping[str, Any]
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path]:
    return base._validate_existing_ledgers(directory, campaign)


def load_validated_collection(
    directory: Path, *, require_complete: bool = False
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    campaign = load_campaign(directory)
    indexed, _ = _validate_existing_ledgers(directory, campaign)
    schedules = base._schedules(campaign)
    rows = tuple(
        indexed[(cell_id, batch.batch_id)]
        for cell in campaign["cells"]
        for cell_id in (str(cell["cell_id"]),)
        for batch in schedules[cell_id]
        if (cell_id, batch.batch_id) in indexed
    )
    expected_batches = sum(len(batches) for batches in schedules.values())
    if require_complete and len(rows) != expected_batches:
        raise ValueError(
            "AWS Figure-8 collection is incomplete: "
            f"found {len(rows)} of {expected_batches} validated batches"
        )
    return campaign, rows


def campaign_status(directory: Path) -> dict[str, Any]:
    campaign = load_campaign(directory)
    rows, _ = _validate_existing_ledgers(directory, campaign)
    schedules = base._schedules(campaign)
    cells = []
    for cell in campaign["cells"]:
        cell_id = str(cell["cell_id"])
        completed = [
            rows[(cell_id, batch.batch_id)]
            for batch in schedules[cell_id]
            if (cell_id, batch.batch_id) in rows
        ]
        cells.append(
            {
                "cell_id": cell_id,
                "expected_shots": campaign["shots_per_cell"],
                "completed_shots": sum(row["batch"]["shots"] for row in completed),
                "expected_batches": len(schedules[cell_id]),
                "completed_batches": len(completed),
            }
        )
    totals = {
        key: sum(row[key] for row in cells)
        for key in (
            "expected_shots",
            "completed_shots",
            "expected_batches",
            "completed_batches",
        )
    }
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": campaign["experiment_id"],
        "complete": totals["completed_batches"] == totals["expected_batches"],
        "totals": totals,
        "cells": cells,
    }


def _initialize_numa_worker(cpus: tuple[int, ...]) -> None:
    try:
        os.sched_setaffinity(0, set(cpus))
    except (AttributeError, OSError) as ex:
        raise RuntimeError(
            "unable to pin AWS collection worker to its NUMA node"
        ) from ex
    if tuple(sorted(os.sched_getaffinity(0))) != cpus:
        raise RuntimeError("AWS worker affinity does not match its declared NUMA pool")
    configure_single_thread_runtime()
    _WORKER_CACHE.clear()


def _partition_tasks(
    tasks: Iterable[dict[str, Any]], *, pool_count: int = NUMA_POOL_COUNT
) -> tuple[list[dict[str, Any]], ...]:
    if pool_count != NUMA_POOL_COUNT:
        raise ValueError("AWS task partition requires exactly two pools")
    partitions: tuple[list[dict[str, Any]], ...] = tuple(
        [] for _ in range(pool_count)
    )
    for task in tasks:
        batch_id = BatchSpec.from_json(task["batch"]).batch_id
        partitions[batch_id % pool_count].append(task)
    return partitions


def _run_numa_pools(
    tasks: Iterable[dict[str, Any]], *, layout: Mapping[str, Any], install: Any
) -> None:
    partitions = _partition_tasks(tasks)
    if not isinstance(layout, Mapping):
        raise ValueError("runtime worker layout must be an object")
    pools = layout.get("pools")
    if not isinstance(pools, list) or len(pools) != NUMA_POOL_COUNT:
        raise ValueError("runtime worker layout must declare exactly two pools")
    if any(not isinstance(row, Mapping) for row in pools):
        raise ValueError("runtime worker pools must be objects")
    reconstructed = {
        "nodes": [
            {"node_id": row["node_id"], "cpus": row["cpus"]} for row in pools
        ]
    }
    if layout != _worker_layout(reconstructed):
        raise ValueError("runtime worker layout is not the frozen 2x96 layout")
    executors: list[ProcessPoolExecutor] = []
    try:
        for pool in pools:
            executors.append(
                ProcessPoolExecutor(
                    max_workers=PROCESSES_PER_POOL,
                    initializer=_initialize_numa_worker,
                    initargs=(tuple(pool["cpus"]),),
                    # Two independent executors are started by one parent.
                    # ``spawn`` avoids forking the second pool after the first
                    # executor has created management threads.  The worker
                    # initializer still pins affinity before any cell graph is
                    # compiled.
                    mp_context=multiprocessing.get_context("spawn"),
                )
            )
        iterators = [iter(rows) for rows in partitions]
        pending: dict[Any, tuple[int, dict[str, Any]]] = {}
        pending_by_pool = [0] * NUMA_POOL_COUNT

        def fill(pool_id: int) -> None:
            while pending_by_pool[pool_id] < MAX_PENDING_PER_POOL:
                try:
                    task = next(iterators[pool_id])
                except StopIteration:
                    return
                future = executors[pool_id].submit(_worker_collect, task)
                pending[future] = (pool_id, task)
                pending_by_pool[pool_id] += 1

        for pool_id in range(NUMA_POOL_COUNT):
            fill(pool_id)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            first_error: BaseException | None = None
            for future in done:
                pool_id, task = pending.pop(future)
                pending_by_pool[pool_id] -= 1
                try:
                    row = future.result()
                    install(task, row)
                except BaseException as ex:
                    if first_error is None:
                        first_error = ex
            if first_error is not None:
                raise first_error
            for pool_id in range(NUMA_POOL_COUNT):
                fill(pool_id)
    except BaseException:
        # Keep already-installed ledgers, but do not spend time executing the
        # hundreds of queued batches after a worker/validation/interruption
        # failure.  Running calls finish in their worker processes; their rows
        # are intentionally not installed and are retried on resume.
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        for executor in executors:
            executor.shutdown(wait=True)


@contextmanager
def _exclusive_campaign_lock(directory: Path):
    """Prevent two coordinators from scheduling the same campaign batches."""

    campaign = directory.absolute()
    if (
        campaign.is_symlink()
        or not campaign.is_dir()
        or campaign.resolve() != campaign
    ):
        raise ValueError("campaign root must be a regular non-symlink directory")
    lock_path = campaign.parent / f".{campaign.name}.aws192-collection.lock"
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as ex:
            raise RuntimeError(
                "another coordinator is already collecting this AWS campaign"
            ) from ex
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def run_campaign(
    directory: Path, *, processes: int, root: Path | None = None
) -> dict[str, Any]:
    if processes != REQUIRED_PROCESSES:
        raise ValueError(
            f"AWS Figure-8 campaign requires exactly {REQUIRED_PROCESSES} processes"
        )
    with _exclusive_campaign_lock(directory):
        return _run_campaign_locked(directory, processes=processes, root=root)


def _run_campaign_locked(
    directory: Path, *, processes: int, root: Path | None = None
) -> dict[str, Any]:
    campaign = load_campaign(directory)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    _validate_runtime(campaign, processes=processes, root=root)
    configure_single_thread_runtime()
    existing, collection = _validate_existing_ledgers(directory, campaign)
    schedules = base._schedules(campaign)

    for cell in campaign["cells"]:
        prepared = prepare_cell(
            cell,
            decoder_config=campaign["decoder"],
            dem_options=campaign["dem_options"],
            verify_hashes=True,
        )
        del prepared

    tasks = []
    for cell in campaign["cells"]:
        cell_id = str(cell["cell_id"])
        for batch in schedules[cell_id]:
            if (cell_id, batch.batch_id) in existing:
                continue
            tasks.append(
                _collection_task(
                    cell=cell,
                    batch=batch,
                    decoder=campaign["decoder"],
                    dem_options=campaign["dem_options"],
                    verify_hashes=True,
                    seed_root=campaign["sampler_seed_root"],
                    experiment_id=campaign["experiment_id"],
                    phase=PHASE,
                    replay_policy=REPLAY_POLICY,
                )
            )

    def install(task: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        cell_id = str(task["cell"]["cell_id"])
        batch = BatchSpec.from_json(task["batch"])
        _validate_ledger_row(
            row,
            experiment_id=campaign["experiment_id"],
            phase=PHASE,
            cell=task["cell"],
            batch=batch,
            seed_root=campaign["sampler_seed_root"],
            expected_provenance=base._provenance(task["cell"]),
            replay_policy=REPLAY_POLICY,
        )
        _atomic_json_write(
            base._ledger_path(collection, cell_id=cell_id, batch_id=batch.batch_id), row
        )

    if tasks:
        _run_numa_pools(tasks, layout=campaign["worker_layout"], install=install)
    return campaign_status(directory)


__all__ = [
    "CAMPAIGN_KIND",
    "CAMPAIGN_SCHEMA",
    "HOST_CHECK_SCHEMA",
    "NUMA_POOL_COUNT",
    "PROCESSES_PER_POOL",
    "REQUIRED_PROCESSES",
    "STATUS_SCHEMA",
    "campaign_status",
    "create_campaign",
    "current_aws_identity",
    "current_numa_topology",
    "load_campaign",
    "load_validated_collection",
    "run_campaign",
    "validate_analysis_environment",
    "validate_analysis_runtime",
    "validate_aws_host",
    "validate_campaign",
]
