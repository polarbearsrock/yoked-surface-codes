"""Frozen three-arm Pinball/ProMatch Figure-8b campaign lifecycle.

This module owns campaign creation, validation, resumption, and status.  Shot
sampling and the U0/ProMatch/Pinball batch payload live in
``_pinball_promatch_experiment``.  Keeping those layers separate makes the
remote 32-worker collection auditable without changing either the earlier
two-arm ProMatch campaign or any of its immutable artifacts.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import ctypes
from datetime import datetime, timezone
import errno
import multiprocessing
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any, Iterable, Mapping

from yoked.decoding._artifact_io import (
    THREAD_ENVIRONMENT,
    is_lowercase_hex,
    load_json_strict,
    repo_root,
)
from yoked.decoding._pinball_promatch_experiment import (
    GENERATOR,
    PINBALL_CONFIG,
    PROMATCH_CONFIG,
    REPLAY_POLICY,
    SEED_DERIVATION,
    _clear_worker_preload,
    _collection_task,
    _preload_worker_cell,
    _worker_collect,
    prepare_cell,
    validate_ledger_row,
)
from yoked.decoding._promatch_experiment import (
    _atomic_json_write,
    _canonical_file_hash,
    _default_source_paths,
    configure_single_thread_runtime,
    current_execution_environment,
    current_software_versions,
    repository_state,
)
from yoked.decoding._promatch_stats import (
    BatchSpec,
    canonical_json_bytes,
    manifest_experiment_id,
    validate_batch_schedule,
)


CAMPAIGN_SCHEMA = "yoked.pinball-promatch-fig8-sweep-campaign-v1"
CAMPAIGN_KIND = "yoked-fig8b-u0-promatch-pinball-fixed-shot-sweep"
STATUS_SCHEMA = "yoked.pinball-promatch-fig8-sweep-status-v1"
PHASE = "pinball-promatch-fig8-sweep"
FIXED_P = 0.002
P = FIXED_P
SAMPLE_BATCH_SIZE = 1_000
REQUIRED_PROCESSES = 32
MAX_SHOTS_PER_CELL = 1_000_000
DISTANCES = (5, 7, 9, 11)
PATCH_COUNTS = (6, 10)
ROUND_MULTIPLIERS = (4, 8)
DEM_OPTIONS = {
    "decompose_errors": True,
    "approximate_disjoint_errors": True,
}
PROMATCH_DECODER = PROMATCH_CONFIG
PINBALL_DECODER = PINBALL_CONFIG
GRID = {
    "distances": list(DISTANCES),
    "patches": list(PATCH_COUNTS),
    "round_multipliers": list(ROUND_MULTIPLIERS),
    "yokes": 2,
    "style": "cz",
    "noise": "si1000",
}
REQUIRED_SWEEP_SOURCE_PATHS = {
    "aws/run_pinball_promatch_fig8",
    "experiments/PINBALL_PROMATCH_FIG8_PAIRED_32.md",
    "gcp/run_pinball_promatch_fig8",
    "src/yoked/_yoked_memory_circuits.py",
    "src/yoked/decoding/_artifact_io.py",
    "src/yoked/decoding/_pinball_promatch_analysis.py",
    "src/yoked/decoding/_pinball_promatch_fig8_sweep.py",
    "src/yoked/decoding/_pinball_promatch_experiment.py",
    "src/yoked/decoding/_pinball_reference.py",
    "src/yoked/decoding/_pinball_v2.py",
    "src/yoked/decoding/_pinball_v2_decoder.py",
    "src/yoked/decoding/_promatch.py",
    "src/yoked/decoding/_promatch_decoder.py",
    "src/yoked/decoding/_promatch_layout.py",
    "src/yoked/decoding/_promatch_stats.py",
    "tools/benchmark_pinball_promatch_fig8",
    "tools/plot_pinball_promatch_fig8",
}

PROVENANCE_FIELDS = {
    "circuit_sha256",
    "dem_sha256",
    "promatch_layout_fingerprint",
    "promatch_graph_fingerprint",
    "pinball_layout_fingerprint",
    "pinball_graph_fingerprint",
    "pinball_schedule_fingerprint",
    "num_detectors",
    "num_observables",
    "arms",
}
HASH_PROVENANCE_FIELDS = PROVENANCE_FIELDS - {
    "num_detectors",
    "num_observables",
    "arms",
}
CELL_INPUT_FIELDS = {
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
}
CELL_FIELDS = CELL_INPUT_FIELDS | PROVENANCE_FIELDS
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
    "promatch_decoder",
    "pinball_decoder",
    "replay_policy",
    "grid",
    "expected_shots_by_cell",
    "cell_batch_schedules",
    "cells",
    "experiment_id",
}


def _validate_shots(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("shots_per_cell must be an integer")
    if not 1 <= value <= MAX_SHOTS_PER_CELL:
        raise ValueError(
            f"shots_per_cell must be in [1, {MAX_SHOTS_PER_CELL}]"
        )
    return value


def _cell_id(*, d: int, patches: int, rounds: int) -> str:
    return f"fig8-d{d}-n{patches}-y2-r{rounds}-p0.002"


def _unfrozen_cells() -> list[dict[str, Any]]:
    return [
        {
            "cell_id": _cell_id(
                d=d, patches=patches, rounds=round_multiplier * d
            ),
            "generator": GENERATOR,
            "d": d,
            "r": round_multiplier * d,
            "p": P,
            "patches": patches,
            "yokes": 2,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
        }
        for d in DISTANCES
        for patches in PATCH_COUNTS
        for round_multiplier in ROUND_MULTIPLIERS
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
        rows: list[dict[str, int]] = []
        for shot_start in range(0, shots_per_cell, SAMPLE_BATCH_SIZE):
            shots = min(SAMPLE_BATCH_SIZE, shots_per_cell - shot_start)
            rows.append(
                {
                    "batch_id": next_batch_id,
                    "shot_start": shot_start,
                    "shots": shots,
                }
            )
            next_batch_id += 1
        schedules[cell_id] = rows
    return expected, schedules


def _source_paths(root: Path) -> list[str]:
    paths = set(_default_source_paths(root)) | REQUIRED_SWEEP_SOURCE_PATHS
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


def _install_new_campaign(
    destination: Path, campaign: Mapping[str, Any]
) -> None:
    """Atomically publish a complete campaign directory at a fresh path."""

    parent, scratch = _validate_campaign_install_location(destination)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"pinball-campaign-{destination.name}-",
            dir=scratch,
        )
    )
    installed = False
    try:
        collection = staging / "collection"
        manifest = staging / "campaign.json"
        collection.mkdir()
        _atomic_json_write(manifest, campaign)
        staged_campaign = load_json_strict(
            manifest, description="staged Pinball/ProMatch Figure-8 campaign"
        )
        validate_campaign(staged_campaign)
        if staged_campaign != campaign:
            raise ValueError("staged campaign differs from constructed campaign")
        if {entry.name for entry in staging.iterdir()} != {
            "campaign.json",
            "collection",
        }:
            raise ValueError("staged campaign contains unexpected entries")
        if any(collection.iterdir()):
            raise ValueError("staged campaign collection must be empty")
        _fsync_directory(collection)
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, destination)
        installed = True
        try:
            _fsync_directory(parent)
        except OSError:
            pass
    finally:
        if not installed and staging.exists():
            try:
                (staging / "campaign.json").unlink(missing_ok=True)
                (staging / "collection").rmdir()
                staging.rmdir()
            except OSError:
                # Never remove unexpected contents recursively. A hard kill
                # can leave only this unique, mode-0700 path under TMPDIR.
                pass


def _validate_campaign_install_location(destination: Path) -> tuple[Path, Path]:
    """Validate atomic-publication paths before expensive cell compilation."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve()
    if parent != destination.parent or destination.is_symlink():
        raise ValueError("campaign parent and output must not traverse symlinks")
    if destination.exists():
        raise ValueError("campaign output must not already exist")

    raw_scratch = os.environ.get("TMPDIR")
    if not raw_scratch:
        raise ValueError("TMPDIR must be set for atomic campaign creation")
    scratch = Path(raw_scratch)
    if not scratch.is_absolute() or scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("TMPDIR must be an absolute, real directory")
    if scratch.resolve() != scratch:
        raise ValueError("TMPDIR must not traverse symbolic links")
    if not _same_filesystem(scratch, parent):
        raise ValueError(
            "TMPDIR and campaign parent must share a filesystem for atomic creation"
        )
    return parent, scratch


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_filesystem(left: Path, right: Path) -> bool:
    return os.stat(left).st_dev == os.stat(right).st_dev


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish ``source`` with Linux ``RENAME_NOREPLACE`` semantics."""

    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("atomic no-replace campaign publication requires Linux")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as ex:
        raise RuntimeError(
            "the C library does not provide renameat2 for atomic campaign creation"
        ) from ex
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    flags = os.O_RDONLY | os.O_DIRECTORY
    source_parent_fd = os.open(source.parent, flags)
    try:
        destination_parent_fd = os.open(destination.parent, flags)
        try:
            ctypes.set_errno(0)
            result = renameat2(
                source_parent_fd,
                os.fsencode(source.name),
                destination_parent_fd,
                os.fsencode(destination.name),
                1,  # RENAME_NOREPLACE
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(
                    error,
                    "campaign output appeared during atomic publication",
                    destination,
                )
            if error in {
                errno.ENOSYS,
                errno.EINVAL,
                errno.EXDEV,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                errno.ENOTSUP,
            }:
                raise RuntimeError(
                    "atomic no-replace campaign publication is unavailable: "
                    f"{os.strerror(error)}"
                )
            raise OSError(error, os.strerror(error), destination)
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_parent_fd)


def create_campaign(
    out: Path,
    *,
    shots_per_cell: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compile and freeze a fresh fixed-p three-arm campaign."""

    shots_per_cell = _validate_shots(shots_per_cell)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("campaign creation requires a clean worktree")
    destination = _validate_new_campaign_root(out)
    _validate_campaign_install_location(destination)

    populated: list[dict[str, Any]] = []
    for cell in _unfrozen_cells():
        prepared = prepare_cell(
            cell,
            promatch_config=PROMATCH_DECODER,
            pinball_config=PINBALL_DECODER,
            dem_options=DEM_OPTIONS,
            verify_hashes=False,
        )
        provenance = dict(prepared.provenance)
        if set(provenance) != PROVENANCE_FIELDS:
            raise ValueError("prepared cell returned an incorrect provenance schema")
        populated.append({**cell, **provenance})
        del prepared

    expected, schedules = _build_schedules(
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
        "p": P,
        "seed_derivation": SEED_DERIVATION,
        "sampler_seed_root": secrets.token_hex(32),
        "dem_options": dict(DEM_OPTIONS),
        "promatch_decoder": copy.deepcopy(PROMATCH_DECODER),
        "pinball_decoder": copy.deepcopy(PINBALL_DECODER),
        "replay_policy": copy.deepcopy(REPLAY_POLICY),
        "grid": copy.deepcopy(GRID),
        "expected_shots_by_cell": expected,
        "cell_batch_schedules": schedules,
        "cells": populated,
    }
    campaign["experiment_id"] = manifest_experiment_id(campaign)
    validate_campaign(campaign)
    _install_new_campaign(destination, campaign)
    return campaign


def validate_campaign(campaign: Mapping[str, Any]) -> str:
    """Validate immutable content without consulting current runtime state."""

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
        "p": P,
        "seed_derivation": SEED_DERIVATION,
        "dem_options": DEM_OPTIONS,
        "promatch_decoder": PROMATCH_DECODER,
        "pinball_decoder": PINBALL_DECODER,
        "replay_policy": REPLAY_POLICY,
        "grid": GRID,
    }
    for key, expected in fixed.items():
        if campaign.get(key) != expected:
            raise ValueError(f"campaign has invalid frozen {key}")
    shots_per_cell = _validate_shots(campaign["shots_per_cell"])
    if not is_lowercase_hex(campaign.get("sampler_seed_root"), length=64):
        raise ValueError("campaign sampler_seed_root must be 256-bit lowercase hex")
    if not is_lowercase_hex(campaign.get("repository_commit"), length=40):
        raise ValueError("campaign repository_commit must be 40-character lowercase hex")
    if not isinstance(campaign.get("created_utc"), str) or not campaign["created_utc"]:
        raise ValueError("campaign created_utc must be a nonempty string")
    for key in ("software_versions", "execution_environment", "source_hashes"):
        if not isinstance(campaign.get(key), Mapping) or not campaign[key]:
            raise ValueError(f"campaign {key} must be a nonempty object")
    missing_sources = REQUIRED_SWEEP_SOURCE_PATHS - set(campaign["source_hashes"])
    if missing_sources:
        raise ValueError(
            "campaign source_hashes omits required sweep sources: "
            f"{sorted(missing_sources)}"
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
    for cell, expected_input in zip(cells, _unfrozen_cells(), strict=True):
        if not isinstance(cell, Mapping) or set(cell) != CELL_FIELDS:
            raise ValueError("campaign cell has incorrect fields")
        for key, expected in expected_input.items():
            if cell.get(key) != expected:
                raise ValueError(f"campaign has incorrect Figure-8 cell {key}")
        for key in HASH_PROVENANCE_FIELDS:
            if not is_lowercase_hex(cell.get(key), length=64):
                raise ValueError(f"campaign cell has invalid {key}")
        for key in ("num_detectors", "num_observables"):
            if (
                isinstance(cell.get(key), bool)
                or not isinstance(cell.get(key), int)
                or cell[key] <= 0
            ):
                raise ValueError(f"campaign cell has invalid {key}")
        expected_arms = {
            "u0": {
                "decoder": "uncorrelated-pymatching-from-common-dem",
                "construction": "pymatching.Matching.from_detector_error_model",
            },
            "promatch": copy.deepcopy(PROMATCH_DECODER),
            "pinball": copy.deepcopy(PINBALL_DECODER),
        }
        if cell.get("arms") != expected_arms:
            raise ValueError("campaign cell has invalid arms provenance")

    cell_ids = [str(cell["cell_id"]) for cell in cells]
    expected_shots = campaign.get("expected_shots_by_cell")
    schedules = campaign.get("cell_batch_schedules")
    if not isinstance(expected_shots, Mapping) or set(expected_shots) != set(cell_ids):
        raise ValueError("campaign expected shots do not exactly cover cells")
    if not isinstance(schedules, Mapping) or set(schedules) != set(cell_ids):
        raise ValueError("campaign schedules do not exactly cover cells")
    all_batch_ids: set[int] = set()
    for cell_id in cell_ids:
        if expected_shots[cell_id] != shots_per_cell:
            raise ValueError("campaign expected shots differ from shots_per_cell")
        batches = validate_batch_schedule(
            schedules[cell_id],
            expected_shots=shots_per_cell,
            batch_size=SAMPLE_BATCH_SIZE,
        )
        ids = {batch.batch_id for batch in batches}
        if all_batch_ids & ids:
            raise ValueError("campaign batch IDs must be globally unique")
        all_batch_ids |= ids
    experiment_id = manifest_experiment_id(campaign)
    if campaign.get("experiment_id") != experiment_id:
        raise ValueError("campaign experiment_id does not match canonical content")
    return experiment_id


def load_campaign(directory: Path) -> dict[str, Any]:
    campaign = load_json_strict(
        directory.absolute() / "campaign.json",
        description="Pinball/ProMatch Figure-8 campaign",
    )
    validate_campaign(campaign)
    return campaign


def validate_analysis_runtime(
    campaign: Mapping[str, Any], *, root: Path | None = None
) -> None:
    """Authenticate code and software for a read-only campaign analysis."""

    validate_campaign(campaign)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("campaign analysis requires a clean worktree")
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
        raise ValueError(f"campaign requires exactly {REQUIRED_PROCESSES} processes")
    if os.environ.get("MAX_ERRORS") is not None:
        raise ValueError("MAX_ERRORS is forbidden for fixed-shot collection")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise ValueError(f"{name} must be exactly 1 before collection")
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
    campaign_dir: Path, *, schedules: Mapping[str, tuple[BatchSpec, ...]]
) -> Path:
    root = campaign_dir.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign root must be a regular non-symlink directory")
    entries = tuple(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("campaign entries may not be symlinks")
    names = {entry.name for entry in entries}
    if names - {"campaign.json", "collection", "analysis", "plots"} or not {
        "campaign.json",
        "collection",
    } <= names:
        raise ValueError("campaign root contains unexpected artifacts")
    for optional in ("analysis", "plots"):
        path = root / optional
        if optional in names and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"campaign {optional} must be a regular directory")
    collection = root / "collection"
    if collection.is_symlink() or not collection.is_dir():
        raise ValueError("campaign collection must be a regular directory")
    entries = tuple(collection.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("collection entries may not be symlinks")
    if {entry.name for entry in entries} - {"batches"}:
        raise ValueError("collection contains unexpected artifacts")
    if not entries:
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
        if cell_dir.is_symlink() or not cell_dir.is_dir() or cell_dir not in allowed_dirs:
            raise ValueError(f"collection contains unexpected cell directory {cell_dir}")
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
            row = load_json_strict(path, description="three-arm batch ledger")
            validate_ledger_row(
                row,
                experiment_id=campaign["experiment_id"],
                phase=PHASE,
                cell=cell,
                batch=batch,
                seed_root=campaign["sampler_seed_root"],
                expected_provenance=_provenance(cell),
                replay_policy=campaign["replay_policy"],
            )
            rows[(cell_id, batch.batch_id)] = row
    return rows, collection


def load_validated_collection(
    directory: Path, *, require_complete: bool = False
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    campaign = load_campaign(directory)
    indexed, _ = _validate_existing_ledgers(directory, campaign)
    schedules = _schedules(campaign)
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
            f"collection is incomplete: found {len(rows)} of {expected_batches} "
            "validated batches"
        )
    return campaign, rows


def campaign_status(directory: Path) -> dict[str, Any]:
    campaign = load_campaign(directory)
    rows, _ = _validate_existing_ledgers(directory, campaign)
    schedules = _schedules(campaign)
    cells = []
    for cell in campaign["cells"]:
        cell_id = str(cell["cell_id"])
        complete_rows = [
            rows[(cell_id, batch.batch_id)]
            for batch in schedules[cell_id]
            if (cell_id, batch.batch_id) in rows
        ]
        cells.append(
            {
                "cell_id": cell_id,
                "expected_shots": campaign["shots_per_cell"],
                "completed_shots": sum(row["batch"]["shots"] for row in complete_rows),
                "expected_batches": len(schedules[cell_id]),
                "completed_batches": len(complete_rows),
            }
        )
    totals = {
        "expected_shots": sum(cell["expected_shots"] for cell in cells),
        "completed_shots": sum(cell["completed_shots"] for cell in cells),
        "expected_batches": sum(cell["expected_batches"] for cell in cells),
        "completed_batches": sum(cell["completed_batches"] for cell in cells),
    }
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": campaign["experiment_id"],
        "p": campaign["p"],
        "complete": totals["completed_batches"] == totals["expected_batches"],
        "totals": totals,
        "cells": cells,
    }


def _require_single_threaded_parent(
    task_directory: Path = Path("/proc/self/task"),
    *,
    expected_tid: int | None = None,
) -> None:
    """Fail unless the Linux parent has only its main task before ``fork``.

    CPython 3.14's process-pool implementation launches every explicit-fork
    worker on the first submission, before starting its executor-management
    thread.  This check is deliberately adjacent to that first submission.
    Forking a process that already has another native thread could inherit a
    native-library mutex with no surviving owner in the child.

    ``task_directory`` and ``expected_tid`` are injectable only so focused
    tests can exercise the Linux task-directory contract without threads.
    """

    expected = os.getpid() if expected_tid is None else expected_tid
    try:
        entries = tuple(task_directory.iterdir())
    except OSError as ex:
        raise RuntimeError(
            f"cannot inspect Linux task directory {task_directory} before fork"
        ) from ex
    try:
        tids = sorted(int(entry.name) for entry in entries)
    except ValueError as ex:
        raise RuntimeError(
            f"Linux task directory {task_directory} contains a non-task entry"
        ) from ex
    if tids != [expected]:
        raise RuntimeError(
            "fork-preloaded collection requires exactly one parent native task "
            f"with tid {expected}; found {tids}"
        )


def _run_bounded_pool(
    tasks: Iterable[dict[str, Any]], *, processes: int, install: Any
) -> None:
    """Run one preloaded cell while bounding queued work to twice the pool."""

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

        _require_single_threaded_parent()
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                install(task, future.result())
            fill()


def run_campaign(
    directory: Path, *, processes: int, root: Path | None = None
) -> dict[str, Any]:
    """Validate and resume only the missing batches of a frozen campaign."""

    if processes != REQUIRED_PROCESSES:
        raise ValueError(f"campaign requires exactly {REQUIRED_PROCESSES} processes")
    campaign = load_campaign(directory)
    root = repo_root(Path(__file__)) if root is None else root.resolve()
    _validate_runtime(campaign, processes=processes, root=root)
    configure_single_thread_runtime()
    existing, collection = _validate_existing_ledgers(directory, campaign)
    schedules = _schedules(campaign)

    def install(task: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        cell_id = str(task["cell"]["cell_id"])
        batch = BatchSpec.from_json(task["batch"])
        validate_ledger_row(
            row,
            experiment_id=campaign["experiment_id"],
            phase=PHASE,
            cell=task["cell"],
            batch=batch,
            seed_root=campaign["sampler_seed_root"],
            expected_provenance=_provenance(task["cell"]),
            replay_policy=campaign["replay_policy"],
        )
        _atomic_json_write(
            _ledger_path(collection, cell_id=cell_id, batch_id=batch.batch_id), row
        )

    # Compile and authenticate exactly one cell in the single-threaded parent,
    # then fork its short-lived worker pool.  Children inherit the large graph
    # and Pinball schedule through copy-on-write instead of compiling 32
    # independent copies.  A fully completed cell is not compiled on resume.
    for cell in campaign["cells"]:
        cell_id = str(cell["cell_id"])
        tasks = [
            _collection_task(
                cell=cell,
                batch=batch,
                promatch_config=campaign["promatch_decoder"],
                pinball_config=campaign["pinball_decoder"],
                dem_options=campaign["dem_options"],
                verify_hashes=True,
                seed_root=campaign["sampler_seed_root"],
                experiment_id=campaign["experiment_id"],
                phase=PHASE,
                replay_policy=campaign["replay_policy"],
                require_preload=True,
            )
            for batch in schedules[cell_id]
            if (cell_id, batch.batch_id) not in existing
        ]
        if not tasks:
            continue
        prepared = prepare_cell(
            cell,
            promatch_config=campaign["promatch_decoder"],
            pinball_config=campaign["pinball_decoder"],
            dem_options=campaign["dem_options"],
            verify_hashes=True,
        )
        _preload_worker_cell(prepared)
        try:
            _run_bounded_pool(tasks, processes=processes, install=install)
        finally:
            _clear_worker_preload()
            del prepared
    return campaign_status(directory)


__all__ = [
    "CAMPAIGN_KIND",
    "CAMPAIGN_SCHEMA",
    "DEM_OPTIONS",
    "FIXED_P",
    "GRID",
    "MAX_SHOTS_PER_CELL",
    "P",
    "PHASE",
    "PINBALL_DECODER",
    "PROMATCH_DECODER",
    "REQUIRED_PROCESSES",
    "REQUIRED_SWEEP_SOURCE_PATHS",
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
