"""Frozen, paired Figure-8b sweep creation, collection, and status.

This workflow is deliberately separate from the first-round V3 protocols.  It
uses their paired U0/PU batch payload and ledger validator, but freezes its own
exact 16-cell exploratory grid and 1,000-shot batch schedule.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import math
import multiprocessing
import os
from pathlib import Path
import secrets
from typing import Any, Iterable, Mapping

from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    is_lowercase_hex,
    load_json_strict,
    repo_root,
)
from yoked.decoding._promatch_experiment import (
    GENERATOR,
    SEED_DERIVATION,
    _atomic_json_write,
    _canonical_file_hash,
    _canonical_replay_policy,
    _collection_task,
    _default_source_paths,
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
    validate_batch_schedule,
)


CAMPAIGN_SCHEMA = "yoked.fig8-paired-sweep-campaign-v1"
CAMPAIGN_KIND = "yoked-fig8b-paired-fixed-shot-sweep"
STATUS_SCHEMA = "yoked.fig8-paired-sweep-status-v1"
PHASE = "fig8-sweep"
SAMPLE_BATCH_SIZE = 1_000
REQUIRED_PROCESSES = 32
MAX_SHOTS_PER_CELL = 1_000_000
MAX_SI1000_P = 0.2
DISTANCES = (5, 7, 9, 11)
PATCH_COUNTS = (6, 10)
ROUND_MULTIPLIERS = (4, 8)
GRID = {
    "distances": list(DISTANCES),
    "patches": list(PATCH_COUNTS),
    "round_multipliers": list(ROUND_MULTIPLIERS),
    "yokes": 2,
    "style": "cz",
    "noise": "si1000",
}
DEM_OPTIONS = {
    "decompose_errors": True,
    "approximate_disjoint_errors": True,
}
DECODER = {
    "residual_hw_limit": 10,
    "domain_mode": "windowd",
    "boundary_policy": "disabled",
    "observable_policy": "zero-frame",
}
REPLAY_POLICY = _canonical_replay_policy(0)
REQUIRED_SWEEP_SOURCE_PATHS = {
    "src/yoked/decoding/_fig8_paired_sweep.py",
    "tools/benchmark_fig8_paired",
    "tools/plot_fig8_paired",
}

CAMPAIGN_FIELDS = {
    "schema",
    "kind",
    "status",
    "frozen",
    "claim_bearing",
    "phase",
    "repository_commit",
    "clean_worktree",
    "created_utc",
    "software_versions",
    "execution_environment",
    "source_hashes",
    "processes",
    "threads_per_process",
    "sample_batch_size",
    "shots_per_cell",
    "p",
    "seed_derivation",
    "sampler_seed_root",
    "dem_options",
    "decoder",
    "grid",
    "expected_shots_by_cell",
    "cell_batch_schedules",
    "cells",
    "experiment_id",
}
CELL_FIELDS = {
    "cell_id",
    "generator",
    "d",
    "r",
    "p",
    "patches",
    "yokes",
    "style",
    "noise",
    "remove_x_yoke",
    "circuit_sha256",
    "dem_sha256",
    "layout_fingerprint",
    "graph_fingerprint",
    "num_detectors",
    "num_observables",
}
PROVENANCE_FIELDS = {
    "circuit_sha256",
    "dem_sha256",
    "layout_fingerprint",
    "graph_fingerprint",
    "num_detectors",
    "num_observables",
}


def _validate_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("p must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= MAX_SI1000_P:
        raise ValueError("SI1000 probability-model domain requires finite 0 < p <= 0.2")
    return result


def _validate_shots(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("shots_per_cell must be an integer")
    if not 1 <= value <= MAX_SHOTS_PER_CELL:
        raise ValueError(f"shots_per_cell must be in [1, {MAX_SHOTS_PER_CELL}]")
    return value


def _cell_id(*, d: int, patches: int, rounds: int, p: float) -> str:
    return f"fig8-d{d}-n{patches}-y2-r{rounds}-p{format(p, '.17g')}"


def _unfrozen_cells(p: float) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": _cell_id(d=d, patches=patches, rounds=multiplier * d, p=p),
            "generator": GENERATOR,
            "d": d,
            "r": multiplier * d,
            "p": p,
            "patches": patches,
            "yokes": 2,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
        }
        for d in DISTANCES
        for patches in PATCH_COUNTS
        for multiplier in ROUND_MULTIPLIERS
    ]


def _build_schedules(
    cells: Iterable[Mapping[str, Any]], *, shots_per_cell: int
) -> tuple[dict[str, int], dict[str, list[dict[str, int]]]]:
    expected: dict[str, int] = {}
    schedules: dict[str, list[dict[str, int]]] = {}
    next_batch_id = 0
    for cell in cells:
        cell_id = str(cell["cell_id"])
        expected[cell_id] = shots_per_cell
        shot_start = 0
        rows = []
        while shot_start < shots_per_cell:
            shots = min(SAMPLE_BATCH_SIZE, shots_per_cell - shot_start)
            rows.append(
                {
                    "batch_id": next_batch_id,
                    "shot_start": shot_start,
                    "shots": shots,
                }
            )
            next_batch_id += 1
            shot_start += shots
        schedules[cell_id] = rows
    return expected, schedules


def _source_paths(root: Path) -> list[str]:
    paths = set(_default_source_paths(root))
    paths.update(REQUIRED_SWEEP_SOURCE_PATHS)
    missing = sorted(relative for relative in paths if not (root / relative).is_file())
    if missing:
        raise ValueError(f"campaign source paths are missing: {missing}")
    return sorted(paths)


def _validate_new_campaign_root(out: Path) -> Path:
    candidate = out.absolute()
    if candidate.is_symlink():
        raise ValueError("campaign root may not be a symlink")
    if candidate.exists():
        raise ValueError("campaign output must not already exist")
    return candidate.resolve()


def create_campaign(
    out: Path,
    *,
    p: float,
    shots_per_cell: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Create one immutable, provenance-complete Figure-8b campaign.

    Every cell is compiled before the manifest is written.  Some values inside
    SI1000's probability-model domain can still be rejected when they produce
    graph-incompatible matching weights for this particular sweep.
    """

    p = _validate_probability(p)
    shots_per_cell = _validate_shots(shots_per_cell)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("campaign creation requires a clean worktree")
    destination = _validate_new_campaign_root(out)
    cells = _unfrozen_cells(p)
    populated: list[dict[str, Any]] = []
    for cell in cells:
        prepared = prepare_cell(
            cell,
            decoder_config=DECODER,
            dem_options=DEM_OPTIONS,
            verify_hashes=False,
        )
        populated.append({**cell, **prepared.provenance})
        del prepared
    expected, schedules = _build_schedules(populated, shots_per_cell=shots_per_cell)
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
        "seed_derivation": SEED_DERIVATION,
        "sampler_seed_root": secrets.token_hex(32),
        "dem_options": dict(DEM_OPTIONS),
        "decoder": dict(DECODER),
        "grid": dict(GRID),
        "expected_shots_by_cell": expected,
        "cell_batch_schedules": schedules,
        "cells": populated,
    }
    campaign["experiment_id"] = manifest_experiment_id(campaign)
    validate_campaign(campaign)
    destination.mkdir(parents=True)
    (destination / "collection").mkdir()
    _atomic_json_write(destination / "campaign.json", campaign)
    return campaign


def _expected_cell_rows(p: float) -> list[dict[str, Any]]:
    return _unfrozen_cells(p)


def validate_campaign(campaign: Mapping[str, Any]) -> str:
    """Validate the complete immutable campaign without consulting runtime state."""

    canonical_json_bytes(campaign)
    if set(campaign) != CAMPAIGN_FIELDS:
        raise ValueError("campaign.json has incorrect top-level fields")
    fixed = {
        "schema": CAMPAIGN_SCHEMA,
        "kind": CAMPAIGN_KIND,
        "status": "FROZEN",
        "frozen": True,
        "claim_bearing": False,
        "phase": PHASE,
        "clean_worktree": True,
        "processes": REQUIRED_PROCESSES,
        "threads_per_process": 1,
        "sample_batch_size": SAMPLE_BATCH_SIZE,
        "seed_derivation": SEED_DERIVATION,
        "dem_options": DEM_OPTIONS,
        "decoder": DECODER,
        "grid": GRID,
    }
    for key, value in fixed.items():
        if campaign.get(key) != value:
            raise ValueError(f"campaign has invalid frozen {key}")
    p = _validate_probability(campaign["p"])
    shots_per_cell = _validate_shots(campaign["shots_per_cell"])
    if not is_lowercase_hex(campaign.get("sampler_seed_root"), length=64):
        raise ValueError("campaign sampler_seed_root must be 256-bit lowercase hex")
    if not isinstance(campaign.get("repository_commit"), str) or not is_lowercase_hex(
        campaign["repository_commit"]
    ):
        raise ValueError("campaign repository_commit must be lowercase hex")
    if not isinstance(campaign.get("created_utc"), str) or not campaign["created_utc"]:
        raise ValueError("campaign created_utc must be a nonempty string")
    for name in ("software_versions", "execution_environment", "source_hashes"):
        if not isinstance(campaign.get(name), Mapping) or not campaign[name]:
            raise ValueError(f"campaign {name} must be a nonempty object")
    missing_sources = REQUIRED_SWEEP_SOURCE_PATHS - set(campaign["source_hashes"])
    if missing_sources:
        raise ValueError(
            f"campaign source_hashes omits required sweep sources: {sorted(missing_sources)}"
        )
    for relative, digest in campaign["source_hashes"].items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not is_lowercase_hex(digest, length=64)
        ):
            raise ValueError("campaign source_hashes contains an invalid entry")

    cells = campaign.get("cells")
    if not isinstance(cells, list) or len(cells) != 16:
        raise ValueError("campaign must contain exactly 16 cells")
    expected_inputs = _expected_cell_rows(p)
    for cell, expected_input in zip(cells, expected_inputs, strict=True):
        if not isinstance(cell, Mapping) or set(cell) != CELL_FIELDS:
            raise ValueError("campaign cell has incorrect fields")
        for key, expected_value in expected_input.items():
            if cell.get(key) != expected_value:
                raise ValueError(f"campaign has incorrect Figure-8 cell {key}")
        for key in PROVENANCE_FIELDS - {"num_detectors", "num_observables"}:
            if not is_lowercase_hex(cell.get(key), length=64):
                raise ValueError(f"campaign cell has invalid {key}")
        for key in ("num_detectors", "num_observables"):
            if (
                isinstance(cell.get(key), bool)
                or not isinstance(cell.get(key), int)
                or cell[key] <= 0
            ):
                raise ValueError(f"campaign cell has invalid {key}")

    cell_ids = [str(cell["cell_id"]) for cell in cells]
    expected = campaign.get("expected_shots_by_cell")
    schedules = campaign.get("cell_batch_schedules")
    if not isinstance(expected, Mapping) or set(expected) != set(cell_ids):
        raise ValueError("campaign expected shots do not exactly cover cells")
    if not isinstance(schedules, Mapping) or set(schedules) != set(cell_ids):
        raise ValueError("campaign schedules do not exactly cover cells")
    all_batch_ids: set[int] = set()
    for cell_id in cell_ids:
        if expected[cell_id] != shots_per_cell:
            raise ValueError("campaign expected shots differ from shots_per_cell")
        parsed = validate_batch_schedule(
            schedules[cell_id],
            expected_shots=shots_per_cell,
            batch_size=SAMPLE_BATCH_SIZE,
        )
        ids = {batch.batch_id for batch in parsed}
        if all_batch_ids & ids:
            raise ValueError("campaign batch IDs must be globally unique")
        all_batch_ids |= ids
    experiment_id = manifest_experiment_id(campaign)
    if campaign.get("experiment_id") != experiment_id:
        raise ValueError("campaign experiment_id does not match canonical content")
    return experiment_id


def load_campaign(directory: Path) -> dict[str, Any]:
    path = directory.absolute() / "campaign.json"
    campaign = load_json_strict(path, description="Figure-8 campaign")
    validate_campaign(campaign)
    return campaign


def validate_analysis_runtime(
    campaign: Mapping[str, Any], *, root: Path | None = None
) -> None:
    """Authenticate code and software for read-only analysis of a campaign.

    Unlike collection validation, this intentionally does not constrain worker
    count, native-thread variables, ``MAX_ERRORS``, or execution hardware.
    """

    validate_campaign(campaign)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("Figure-8 analysis requires a clean worktree")
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


def _validate_runtime(
    campaign: Mapping[str, Any], *, processes: int, root: Path
) -> None:
    if processes != REQUIRED_PROCESSES:
        raise ValueError(
            f"Figure-8 campaign requires exactly {REQUIRED_PROCESSES} processes"
        )
    if os.environ.get("MAX_ERRORS") is not None:
        raise ValueError("MAX_ERRORS is forbidden for fixed-shot Figure-8 collection")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise ValueError(f"{name} must be exactly 1 before Figure-8 collection")
    validate_analysis_runtime(campaign, root=root)
    if current_execution_environment() != campaign["execution_environment"]:
        raise ValueError("runtime execution environment differs from campaign")


def _schedules(campaign: Mapping[str, Any]) -> dict[str, tuple[BatchSpec, ...]]:
    return {
        str(cell["cell_id"]): validate_batch_schedule(
            campaign["cell_batch_schedules"][cell["cell_id"]],
            expected_shots=campaign["shots_per_cell"],
            batch_size=SAMPLE_BATCH_SIZE,
        )
        for cell in campaign["cells"]
    }


def _provenance(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {key: cell[key] for key in PROVENANCE_FIELDS}


def _ledger_path(collection: Path, *, cell_id: str, batch_id: int) -> Path:
    return collection / "batches" / cell_id / f"batch-{batch_id:08d}.json"


def _validate_collection_root(
    campaign_dir: Path,
    *,
    schedules: Mapping[str, tuple[BatchSpec, ...]],
) -> Path:
    root = campaign_dir.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign root must be a regular non-symlink directory")
    entries = tuple(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("campaign entries may not be symlinks")
    names = {entry.name for entry in entries}
    if (
        names - {"campaign.json", "collection", "plots"}
        or not {
            "campaign.json",
            "collection",
        }
        <= names
    ):
        raise ValueError(
            "campaign root must contain campaign.json, collection, and optionally plots"
        )
    plots = root / "plots"
    if "plots" in names and (plots.is_symlink() or not plots.is_dir()):
        raise ValueError("campaign plots must be a regular non-symlink directory")
    collection = root / "collection"
    if collection.is_symlink() or not collection.is_dir():
        raise ValueError("campaign collection must be a regular directory")
    collection_entries = tuple(collection.iterdir())
    if any(entry.is_symlink() for entry in collection_entries):
        raise ValueError("collection entries may not be symlinks")
    names = {entry.name for entry in collection_entries}
    if names - {"batches"}:
        raise ValueError("collection contains unexpected artifacts")
    if "batches" not in names:
        return collection
    batch_root = collection / "batches"
    if batch_root.is_symlink() or not batch_root.is_dir():
        raise ValueError("collection batches must be a regular directory")
    allowed = {
        _ledger_path(collection, cell_id=cell_id, batch_id=batch.batch_id)
        for cell_id, batches in schedules.items()
        for batch in batches
    }
    allowed_dirs = {path.parent for path in allowed}
    for cell_dir in batch_root.iterdir():
        if (
            cell_dir.is_symlink()
            or not cell_dir.is_dir()
            or cell_dir not in allowed_dirs
        ):
            raise ValueError(
                f"collection contains unexpected cell directory {cell_dir}"
            )
        for ledger in cell_dir.iterdir():
            if ledger.is_symlink() or not ledger.is_file() or ledger not in allowed:
                raise ValueError(f"collection contains unexpected ledger {ledger}")
    return collection


def _validate_existing_ledgers(
    campaign_dir: Path, campaign: Mapping[str, Any]
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path]:
    schedules = _schedules(campaign)
    collection = _validate_collection_root(campaign_dir, schedules=schedules)
    cells = {str(cell["cell_id"]): cell for cell in campaign["cells"]}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for cell_id, batches in schedules.items():
        cell = cells[cell_id]
        for batch in batches:
            path = _ledger_path(collection, cell_id=cell_id, batch_id=batch.batch_id)
            if not path.exists():
                continue
            row = load_json_strict(path, description="Figure-8 batch ledger")
            _validate_ledger_row(
                row,
                experiment_id=campaign["experiment_id"],
                phase=PHASE,
                cell=cell,
                batch=batch,
                seed_root=campaign["sampler_seed_root"],
                expected_provenance=_provenance(cell),
                replay_policy=REPLAY_POLICY,
            )
            rows[(cell_id, batch.batch_id)] = row
    return rows, collection


def load_validated_collection(
    directory: Path, *, require_complete: bool = False
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Load a campaign and its validated ledger rows in frozen schedule order."""

    campaign = load_campaign(directory)
    indexed_rows, _ = _validate_existing_ledgers(directory, campaign)
    schedules = _schedules(campaign)
    rows = tuple(
        indexed_rows[(cell_id, batch.batch_id)]
        for cell in campaign["cells"]
        for cell_id in (str(cell["cell_id"]),)
        for batch in schedules[cell_id]
        if (cell_id, batch.batch_id) in indexed_rows
    )
    expected_batches = sum(len(batches) for batches in schedules.values())
    if require_complete and len(rows) != expected_batches:
        raise ValueError(
            "Figure-8 collection is incomplete: "
            f"found {len(rows)} of {expected_batches} validated batches"
        )
    return campaign, rows


def campaign_status(directory: Path) -> dict[str, Any]:
    """Return compact, fully validated per-cell batch/shot progress."""

    campaign = load_campaign(directory)
    rows, _ = _validate_existing_ledgers(directory, campaign)
    schedules = _schedules(campaign)
    cell_rows = []
    for cell in campaign["cells"]:
        cell_id = str(cell["cell_id"])
        completed = [
            rows[(cell_id, batch.batch_id)]
            for batch in schedules[cell_id]
            if (cell_id, batch.batch_id) in rows
        ]
        cell_rows.append(
            {
                "cell_id": cell_id,
                "expected_shots": campaign["shots_per_cell"],
                "completed_shots": sum(row["batch"]["shots"] for row in completed),
                "expected_batches": len(schedules[cell_id]),
                "completed_batches": len(completed),
            }
        )
    totals = {
        "expected_shots": sum(row["expected_shots"] for row in cell_rows),
        "completed_shots": sum(row["completed_shots"] for row in cell_rows),
        "expected_batches": sum(row["expected_batches"] for row in cell_rows),
        "completed_batches": sum(row["completed_batches"] for row in cell_rows),
    }
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": campaign["experiment_id"],
        "complete": totals["completed_batches"] == totals["expected_batches"],
        "totals": totals,
        "cells": cell_rows,
    }


def _run_bounded_pool(
    tasks: Iterable[dict[str, Any]],
    *,
    processes: int,
    install: Any,
) -> None:
    """Run tasks with no more than twice the worker count submitted at once."""

    iterator = iter(tasks)
    limit = 2 * processes
    with ProcessPoolExecutor(
        max_workers=processes,
        initializer=configure_single_thread_runtime,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        pending: dict[Any, dict[str, Any]] = {}

        def fill() -> None:
            while len(pending) < limit:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
                pending[executor.submit(_worker_collect, task)] = task

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                install(task, future.result())
            fill()


def run_campaign(
    directory: Path,
    *,
    processes: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Strictly validate and resume the frozen paired Figure-8b collection."""

    if processes != REQUIRED_PROCESSES:
        raise ValueError(
            f"Figure-8 campaign requires exactly {REQUIRED_PROCESSES} processes"
        )
    campaign = load_campaign(directory)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    _validate_runtime(campaign, processes=processes, root=root)
    configure_single_thread_runtime()
    existing, collection = _validate_existing_ledgers(directory, campaign)
    schedules = _schedules(campaign)

    # Rebuild every circuit/DEM/graph before collection, rejecting provenance
    # drift before any new ledger is installed.  Do not retain PreparedCells in
    # the parent; workers cache at most their current cell.
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
            expected_provenance=_provenance(task["cell"]),
            replay_policy=REPLAY_POLICY,
        )
        _atomic_json_write(
            _ledger_path(collection, cell_id=cell_id, batch_id=batch.batch_id), row
        )

    if tasks:
        _run_bounded_pool(tasks, processes=processes, install=install)
    return campaign_status(directory)


__all__ = [
    "CAMPAIGN_KIND",
    "CAMPAIGN_SCHEMA",
    "DECODER",
    "DEM_OPTIONS",
    "GRID",
    "MAX_SI1000_P",
    "MAX_SHOTS_PER_CELL",
    "PHASE",
    "REQUIRED_SWEEP_SOURCE_PATHS",
    "REQUIRED_PROCESSES",
    "SAMPLE_BATCH_SIZE",
    "STATUS_SCHEMA",
    "campaign_status",
    "create_campaign",
    "load_campaign",
    "load_validated_collection",
    "run_campaign",
    "validate_analysis_runtime",
    "validate_campaign",
]
