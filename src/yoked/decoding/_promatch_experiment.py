"""Paired, fixed-shot experiment harness for the L1 ProMatch study.

The collector intentionally does not use Sinter's error-count stopping.  A
protocol declares immutable 10,000-shot batches, each batch is sampled once,
and the identical bit-packed detector/observable arrays are decoded by U0 and
PU.  Completed batches are independent JSON ledger records, making resume a
set difference over predeclared batch IDs instead of a continuation of an RNG
stream.

Frozen protocol schema (``promatch-l1-paired-protocol-v2``):

* ``phase`` is ``pilot`` or ``confirm``, ``status`` is ``FROZEN``, and
  ``frozen`` is true; a frozen confirm manifest also owns the separate
  ``target`` performance corpus;
* ``repository_commit``, ``clean_worktree``, ``software_versions`` and
  ``source_hashes`` freeze executable provenance;
* ``sample_batch_size`` is 10000 and ``processes`` is in [1, 32];
* ``sampler_seed_roots[phase]`` is a literal 256-bit hexadecimal root;
* ``expected_shots[phase]`` and ``batch_schedules[phase]`` declare a gap-free
  fixed-shot schedule;
* each ``cells`` row declares generator inputs and the circuit, DEM, layout,
  and graph hashes; and
* ``decoder`` freezes the PU policy while ``replay_policy`` separately bounds
  per-batch replay candidates and per-cell summary examples.  Only regression,
  recovery, and rollback shots are replay categories; invariant violations are
  fatal run errors.

Draft protocols may be inspected/frozen and used by the ``smoke`` command, but
never by scientific ``pilot``/``confirm`` collection.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._promatch_decoder import PromatchDecoder
from yoked.decoding._promatch_stats import (
    ROUND_ONE_BATCH_SIZE,
    ArrayDigest,
    BatchSpec,
    PairedContingency,
    canonical_json_bytes,
    derive_stim_batch_seed,
    digest_array,
    manifest_experiment_id,
    validate_batch_schedule,
    validate_process_count,
    validate_protocol_manifest,
)


PROTOCOL_SCHEMA = "promatch-l1-paired-protocol-v2"
PROTOCOL_KIND = "promatch-l1-paired-fixed-shot"
LEDGER_SCHEMA = "promatch-l1-paired-batch-v1"
SUMMARY_SCHEMA = "promatch-l1-paired-summary-v1"
SEED_DERIVATION = "sha256-root+stim-batch+uint64le-first8-uint64le"
GENERATOR = "yoked._yoked_memory_circuits:yoked_magic_memory_circuit"
REPLAY_CATEGORIES = ("regression", "recovery", "rollback")
REPLAY_SELECTION_KEY = "SHA256_ASCII(cell_id:batch_id:shot_index:category)"
REPLAY_BATCH_SELECTION = (
    "lowest_selection_sha256_within_batch_and_category"
)
REPLAY_CELL_SELECTION = (
    "lowest_selection_sha256_across_batch_candidates_within_cell_and_category"
)
INVARIANT_VIOLATION_POLICY = "fatal_run_error_no_replay_row"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
DEFAULT_SOURCE_PATHS = (
    "src/yoked/decoding/__init__.py",
    "src/yoked/decoding/_promatch_analysis.py",
    "src/yoked/decoding/_promatch.py",
    "src/yoked/decoding/_promatch_decoder.py",
    "src/yoked/decoding/_promatch_experiment.py",
    "src/yoked/decoding/_promatch_graph.py",
    "src/yoked/decoding/_promatch_layout.py",
    "src/yoked/decoding/_promatch_latency.py",
    "src/yoked/decoding/_promatch_latency_analysis.py",
    "src/yoked/decoding/_promatch_latency_integration.py",
    "src/yoked/decoding/_promatch_stats.py",
    "src/yoked/_yoked_memory_circuits.py",
    "tools/analyze_promatch_l1",
    "tools/benchmark_promatch_l1",
)


def _default_source_paths(root: Path) -> list[str]:
    paths = set(DEFAULT_SOURCE_PATHS)
    for base in (root / "src" / "gen", root / "src" / "yoked" / "decoding"):
        if base.is_dir():
            for path in base.rglob("*.py"):
                if not path.name.endswith("_test.py"):
                    paths.add(path.relative_to(root).as_posix())
    for relative in (
        "src/yoked/__init__.py",
        "src/yoked/_patch_rotation.py",
        "src/yoked/_yoked_memory_circuits.py",
        "requirements.txt",
        "reproduce_fig8_1d",
    ):
        if (root / relative).is_file():
            paths.add(relative)
    return sorted(paths)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_replay_policy(cap: int) -> dict[str, Any]:
    return {
        "categories": list(REPLAY_CATEGORIES),
        "maximum_candidate_rows_per_category_per_batch_ledger": cap,
        "maximum_retained_rows_per_category_per_cell_summary": cap,
        "selection_key": REPLAY_SELECTION_KEY,
        "batch_ledger_selection": REPLAY_BATCH_SELECTION,
        "cell_summary_selection": REPLAY_CELL_SELECTION,
        "equal_cap_prefilter_equivalence": True,
        "invariant_violation_policy": INVARIANT_VIOLATION_POLICY,
        "must_be_replayable": True,
    }


def _validate_replay_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("replay_policy must be an object")
    batch_cap = value.get(
        "maximum_candidate_rows_per_category_per_batch_ledger"
    )
    summary_cap = value.get(
        "maximum_retained_rows_per_category_per_cell_summary"
    )
    for name, cap in (("batch", batch_cap), ("summary", summary_cap)):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ValueError(f"replay_policy {name} cap must be a nonnegative integer")
    if batch_cap != summary_cap:
        raise ValueError(
            "replay_policy batch and summary caps must be equal so the batch "
            "prefilter preserves the globally lowest per-cell hashes"
        )
    expected = _canonical_replay_policy(batch_cap)
    if dict(value) != expected:
        raise ValueError(f"replay_policy must be exactly {expected}")
    return expected


def _jsonable_digest(value: ArrayDigest) -> dict[str, Any]:
    return {"sha256": value.sha256, "shape": list(value.shape), "dtype": value.dtype}


def _canonical_file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(args: list[str], *, root: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def repository_state(root: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if root is None else root.resolve()
    return {
        "repository_commit": _git(["rev-parse", "HEAD"], root=root),
        "clean_worktree": _git(
            ["status", "--porcelain", "--untracked-files=all"], root=root
        )
        == "",
    }


def _validate_post_freeze_protocol_commit(
    *, manifest: Mapping[str, Any], root: Path, frozen_base: str, current_head: str
) -> None:
    """Allow exactly one protocol-only commit after the frozen implementation.

    A manifest cannot contain the hash of the commit that contains itself.  The
    supported two-commit sequence is therefore: freeze against a clean
    implementation commit, then add that exact JSON as one ``docs/*FROZEN*.json``
    file.  No source or second documentation change is admitted between the
    recorded implementation commit and collection.
    """

    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_base, current_head],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("frozen implementation commit is not an ancestor of HEAD")
    changed = _git(
        ["diff", "--name-status", "--find-renames", f"{frozen_base}..{current_head}"],
        root=root,
    ).splitlines()
    if len(changed) != 1:
        raise ValueError(
            "after freezing, HEAD may differ only by one committed frozen protocol file"
        )
    pieces = changed[0].split("\t")
    if len(pieces) != 2 or pieces[0] not in {"A", "M"}:
        raise ValueError("post-freeze commit must only add the frozen protocol file")
    relative = pieces[1]
    candidate = (root / relative).resolve()
    if (
        root not in candidate.parents
        or candidate.suffix != ".json"
        or "FROZEN" not in candidate.name
        or candidate.parent != (root / "docs").resolve()
        or not candidate.is_file()
    ):
        raise ValueError(
            "post-freeze file must be docs/*FROZEN*.json"
        )
    try:
        committed_manifest = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        raise ValueError("committed frozen protocol is not valid JSON") from ex
    if committed_manifest != dict(manifest):
        raise ValueError(
            "post-freeze protocol commit does not contain the exact runtime manifest"
        )


def current_software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "stim": stim.__version__,
        "sinter": importlib.metadata.version("sinter"),
        "pymatching": pymatching.__version__,
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
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


def current_execution_environment() -> dict[str, Any]:
    """Stable machine fields frozen for claim-bearing reproducibility."""

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    uname = platform.uname()
    return {
        "os": uname.system,
        "kernel": uname.release,
        "machine": uname.machine,
        "cpu_model": _first_cpu_field("model name"),
        "microcode": _first_cpu_field("microcode"),
        "cpu_affinity": affinity,
        "native_thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
    }


def configure_single_thread_runtime() -> None:
    """Force one numerical/native thread in the parent and every worker."""

    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        # Environment variables cover the pinned runtime; threadpoolctl is an
        # optional extra and is deliberately not a runtime dependency.
        pass


def build_batch_schedule(shots: int, *, batch_id_start: int = 0) -> list[dict[str, int]]:
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    rows = []
    shot_start = 0
    if isinstance(batch_id_start, bool) or not isinstance(batch_id_start, int) or batch_id_start < 0:
        raise ValueError("batch_id_start must be a nonnegative integer")
    batch_id = batch_id_start
    while shot_start < shots:
        count = min(ROUND_ONE_BATCH_SIZE, shots - shot_start)
        rows.append(
            {"batch_id": batch_id, "shot_start": shot_start, "shots": count}
        )
        shot_start += count
        batch_id += 1
    return rows


def _decoder_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("decoder")
    if not isinstance(raw, Mapping):
        raise ValueError("protocol decoder must be an object")
    expected = {
        "residual_hw_limit",
        "domain_mode",
        "boundary_policy",
        "observable_policy",
    }
    if set(raw) != expected:
        raise ValueError(f"decoder fields must be exactly {sorted(expected)}")
    config = dict(raw)
    # Construction is also the authoritative enum/range validation.
    PromatchDecoder(**config)
    return config


def _dem_options(manifest: Mapping[str, Any]) -> dict[str, bool]:
    raw = manifest.get("dem_options")
    expected = {
        "decompose_errors": True,
        "approximate_disjoint_errors": True,
    }
    if raw != expected:
        raise ValueError(f"dem_options must be exactly {expected}")
    return expected


def _validate_cell_inputs(cell: Mapping[str, Any], *, require_hashes: bool) -> None:
    required = {
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
    if require_hashes:
        required |= {
            "circuit_sha256",
            "dem_sha256",
            "layout_fingerprint",
            "graph_fingerprint",
        }
    missing = required - set(cell)
    if missing:
        raise ValueError(f"cell is missing fields {sorted(missing)}")
    if not isinstance(cell["cell_id"], str) or not cell["cell_id"]:
        raise ValueError("cell_id must be a nonempty string")
    if "/" in cell["cell_id"] or "\\" in cell["cell_id"]:
        raise ValueError("cell_id cannot contain path separators")
    if cell["generator"] != GENERATOR:
        raise ValueError(f"unsupported circuit generator {cell['generator']!r}")
    if cell["noise"] != "si1000" or cell["style"] != "cz":
        raise ValueError("round one supports only SI1000 noise with style='cz'")
    if cell["remove_x_yoke"] is not False:
        raise ValueError("remove_x_yoke must be false")
    for field in ("d", "r", "patches"):
        if isinstance(cell[field], bool) or not isinstance(cell[field], int) or cell[field] <= 0:
            raise ValueError(f"cell {field} must be a positive integer")
    if cell["r"] < 2:
        raise ValueError("cell r must be at least 2")
    if cell["yokes"] not in (0, 1, 2):
        raise ValueError("cell yokes must be 0, 1, or 2")
    if not isinstance(cell["p"], (int, float)) or not 0 <= cell["p"] <= 1:
        raise ValueError("cell p must lie in [0, 1]")
    if require_hashes:
        for field in (
            "circuit_sha256",
            "dem_sha256",
            "layout_fingerprint",
            "graph_fingerprint",
        ):
            value = cell[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(f"cell {field} must be a lowercase SHA-256 hex digest")


@dataclasses.dataclass(frozen=True)
class PreparedCell:
    cell: dict[str, Any]
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    compiled_pu: Any
    provenance: dict[str, Any]


def prepare_cell(
    cell: Mapping[str, Any],
    *,
    decoder_config: Mapping[str, Any],
    dem_options: Mapping[str, bool],
    verify_hashes: bool,
) -> PreparedCell:
    _validate_cell_inputs(cell, require_hashes=verify_hashes)
    circuit = yoked_magic_memory_circuit(
        patch_diameter=int(cell["d"]),
        rounds=int(cell["r"]),
        noise=gen.NoiseModel.si1000(float(cell["p"])),
        style="cz",
        yokes=int(cell["yokes"]),
        num_patches=int(cell["patches"]),
        remove_x_yoke=False,
    )
    dem = circuit.detector_error_model(**dict(dem_options))
    compiled = PromatchDecoder(**dict(decoder_config)).compile_decoder_for_dem(dem=dem)
    provenance = {
        "circuit_sha256": _sha256_bytes(str(circuit).encode("utf-8")),
        "dem_sha256": _sha256_bytes(str(dem).encode("utf-8")),
        "layout_fingerprint": compiled.graph.layout.fingerprint,
        "graph_fingerprint": compiled.graph.fingerprint,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
    }
    if verify_hashes:
        for key in (
            "circuit_sha256",
            "dem_sha256",
            "layout_fingerprint",
            "graph_fingerprint",
        ):
            if cell[key] != provenance[key]:
                raise ValueError(
                    f"cell {cell['cell_id']!r} {key} mismatch: "
                    f"protocol={cell[key]!r}, actual={provenance[key]!r}"
                )
    return PreparedCell(dict(cell), circuit, dem, compiled, provenance)


def _required_protocol_fields() -> tuple[str, ...]:
    return (
        "schema",
        "kind",
        "status",
        "frozen",
        "claim_bearing",
        "phase",
        "repository_commit",
        "clean_worktree",
        "software_versions",
        "execution_environment",
        "created_utc",
        "frozen_utc",
        "template_sha256",
        "source_hashes",
        "sample_batch_size",
        "processes",
        "seed_derivation",
        "sampler_seed_roots",
        "expected_shots_by_cell",
        "cell_batch_schedules",
        "dem_options",
        "decoder",
        "replay_policy",
        "cells",
        "scientific_contract",
        "analysis_config",
        "reference_provenance",
        "experiment_id",
    )


def _protocol_split(manifest: Mapping[str, Any], phase: str) -> str:
    roots = manifest.get("sampler_seed_roots")
    if not isinstance(roots, Mapping):
        raise ValueError("sampler_seed_roots must be an object")
    split = {
        "confirm": "confirmatory_holdout",
        "target": "target_workload",
    }.get(phase, phase)
    if split not in roots:
        raise ValueError(f"protocol has no seed root for split {split!r}")
    return split


def _cell_schedules(
    manifest: Mapping[str, Any], cells: Iterable[Mapping[str, Any]]
) -> dict[str, tuple[BatchSpec, ...]]:
    schedules = manifest.get("cell_batch_schedules")
    expected = manifest.get("expected_shots_by_cell")
    if not isinstance(schedules, Mapping) or not isinstance(expected, Mapping):
        raise ValueError(
            "cell_batch_schedules and expected_shots_by_cell must be objects"
        )
    cell_ids = {str(cell["cell_id"]) for cell in cells}
    if set(schedules) != cell_ids or set(expected) != cell_ids:
        raise ValueError("per-cell schedules must exactly cover the declared cells")
    result = {}
    all_batch_ids: set[int] = set()
    for cell_id in sorted(cell_ids):
        parsed = validate_batch_schedule(
            schedules[cell_id],
            expected_shots=expected[cell_id],
            batch_size=ROUND_ONE_BATCH_SIZE,
        )
        ids = {batch.batch_id for batch in parsed}
        overlap = all_batch_ids & ids
        if overlap:
            raise ValueError(
                f"batch IDs must be disjoint across cells; repeated {sorted(overlap)[:8]}"
            )
        all_batch_ids |= ids
        result[cell_id] = parsed
    return result


def _phase_cells(manifest: Mapping[str, Any], phase: str) -> list[Mapping[str, Any]]:
    key = "performance_cells" if phase == "target" else "cells"
    cells = manifest.get(key)
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{key} must be a nonempty array")
    return cells


def _phase_schedules(
    manifest: Mapping[str, Any], phase: str, cells: Iterable[Mapping[str, Any]]
) -> dict[str, tuple[BatchSpec, ...]]:
    if phase != "target":
        return _cell_schedules(manifest, cells)
    view = {
        "cell_batch_schedules": manifest.get("performance_cell_batch_schedules"),
        "expected_shots_by_cell": manifest.get("performance_expected_shots_by_cell"),
    }
    return _cell_schedules(view, cells)


def _nested_cell(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": raw["cell_id"],
        "generator": GENERATOR,
        "d": raw["d"],
        "r": raw["rounds"],
        "p": raw["si1000_p"],
        "patches": raw["patches"],
        "yokes": raw["yokes"],
        "style": "cz",
        "noise": "si1000",
        "remove_x_yoke": False,
    }


def normalize_protocol(manifest: Mapping[str, Any], *, for_freeze: bool = False) -> dict[str, Any]:
    """Normalize the documented nested template into the collector schema."""

    if manifest.get("schema") == PROTOCOL_SCHEMA:
        return json.loads(json.dumps(manifest))
    if (
        manifest.get("schema") != "yoked.promatch.l1.protocol"
        or manifest.get("schema_version") != 2
    ):
        raise ValueError("unsupported documented protocol schema")
    kind = manifest.get("protocol_kind")
    if kind == "pilot":
        if manifest.get("protocol_version") != "promatch-l1-pilot-v2":
            raise ValueError("pilot protocol_version must be promatch-l1-pilot-v2")
        phase = "pilot"
        raw_cells = manifest.get("pilot", {}).get("cells")
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ValueError("documented pilot template has no cells")
        shots_per_cell = manifest.get("pilot", {}).get("shots_per_cell")
        cells = [_nested_cell(cell) for cell in raw_cells]
        expected = {cell["cell_id"]: shots_per_cell for cell in raw_cells}
        schedules = {
            cell["cell_id"]: build_batch_schedule(
                shots_per_cell, batch_id_start=cell["batch_id_start"]
            )
            for cell in raw_cells
        }
        for cell in raw_cells:
            if schedules[cell["cell_id"]][-1]["batch_id"] != cell["batch_id_end_inclusive"]:
                raise ValueError("documented pilot batch ID range does not match its shot count")
    elif kind == "first_round_confirmatory":
        if manifest.get("protocol_version") != "promatch-l1-first-round-v2":
            raise ValueError(
                "confirmatory protocol_version must be promatch-l1-first-round-v2"
            )
        phase = "confirm"
        selected = manifest.get("selection", {}).get("selected_cell")
        if selected is None:
            if for_freeze:
                raise ValueError("confirmatory freeze requires selection.selected_cell")
            selected = manifest.get("generator_contract", {}).get("target_cell_metadata")
        if not isinstance(selected, Mapping):
            raise ValueError("documented confirmatory template has no inspectable cell")
        cells = [_nested_cell(selected)]
        cell_id = cells[0]["cell_id"]
        shots = manifest.get("expected_shots", {}).get("confirmatory_holdout")
        raw_schedule = manifest.get("batch_schedules", {}).get("confirmatory_holdout")
        if shots is None or raw_schedule is None:
            if for_freeze:
                raise ValueError("confirmatory freeze requires holdout shots and schedule")
            shots = ROUND_ONE_BATCH_SIZE
            raw_schedule = {"batch_id_start": 0, "batch_id_end_inclusive": 0}
        schedules = {
            cell_id: build_batch_schedule(shots, batch_id_start=raw_schedule["batch_id_start"])
        }
        if schedules[cell_id][-1]["batch_id"] != raw_schedule["batch_id_end_inclusive"]:
            raise ValueError("confirmatory batch ID range does not match n_confirm")
        expected = {cell_id: shots}

        raw_target = manifest.get("generator_contract", {}).get(
            "target_cell_metadata"
        )
        if not isinstance(raw_target, Mapping):
            raise ValueError("confirmatory protocol requires target_cell_metadata")
        performance_cells = [_nested_cell(raw_target)]
        performance_cell_id = performance_cells[0]["cell_id"]
        target_shots = manifest.get("expected_shots", {}).get("target_workload")
        target_schedule = manifest.get("batch_schedules", {}).get("target_workload")
        if target_shots != 1_000_000:
            raise ValueError("target_workload must declare exactly 1000000 shots")
        if not isinstance(target_schedule, Mapping):
            raise ValueError("target_workload must declare an exact batch schedule")
        if target_schedule.get("shots_per_batch") != ROUND_ONE_BATCH_SIZE:
            raise ValueError("target_workload shots_per_batch must be 10000")
        performance_schedules = {
            performance_cell_id: build_batch_schedule(
                target_shots, batch_id_start=target_schedule["batch_id_start"]
            )
        }
        if (
            performance_schedules[performance_cell_id][-1]["batch_id"]
            != target_schedule.get("batch_id_end_inclusive")
        ):
            raise ValueError("target_workload batch ID range must cover exactly 100 batches")
        performance_expected = {performance_cell_id: target_shots}
    else:
        raise ValueError(f"unsupported documented protocol_kind {kind!r}")
    config = manifest.get("decoders", {}).get("pu_window", {}).get("configuration", {})
    output_schema = manifest.get("output_schema")
    if not isinstance(output_schema, Mapping):
        raise ValueError("documented output_schema must be an object")
    if output_schema.get("schema_version") != 2:
        raise ValueError("documented output_schema.schema_version must be 2")
    replay_policy = _validate_replay_policy(output_schema.get("bounded_replay_policy"))
    normalized = {
        "schema": PROTOCOL_SCHEMA,
        "kind": PROTOCOL_KIND,
        "status": manifest.get("status"),
        "frozen": manifest.get("frozen"),
        "claim_bearing": manifest.get("claim_bearing"),
        "phase": phase,
        "repository_commit": manifest.get("repository_commit"),
        "clean_worktree": manifest.get("clean_worktree"),
        "created_utc": manifest.get("created_utc"),
        "sample_batch_size": manifest.get("sample_batch_size"),
        "processes": manifest.get("processes", 32),
        "seed_derivation": SEED_DERIVATION,
        "sampler_seed_roots": manifest.get("sampler_seed_roots"),
        "expected_shots_by_cell": expected,
        "cell_batch_schedules": schedules,
        "dem_options": manifest.get("dem", {}).get("options"),
        "decoder": {
            "residual_hw_limit": config.get("residual_hw_limit"),
            "domain_mode": config.get("domain_mode"),
            "boundary_policy": config.get("boundary_policy"),
            "observable_policy": config.get("observable_policy"),
        },
        "replay_policy": replay_policy,
        "cells": cells,
        "scientific_contract": (
            {
                "protocol_version": manifest.get("protocol_version"),
                "pilot_cells": raw_cells,
            }
            if phase == "pilot"
            else {
                "protocol_version": manifest.get("protocol_version"),
                "pilot_provenance": manifest.get("pilot_provenance"),
                "selection": manifest.get("selection"),
                "accuracy_protocol": manifest.get("accuracy_protocol"),
                "power_protocol": manifest.get("power_protocol"),
            }
        ),
        "analysis_config": (
            {
                "selection_gates": manifest.get("selection_gates"),
                "statistical_design": manifest.get("statistical_design"),
                "workload_protocol": manifest.get("workload_protocol"),
                "timing_protocol": manifest.get("timing_protocol"),
                "output_schema": manifest.get("output_schema"),
            }
            if phase == "pilot"
            else {
                "pilot_provenance": manifest.get("pilot_provenance"),
                "selection": manifest.get("selection"),
                "accuracy_protocol": manifest.get("accuracy_protocol"),
                "power_protocol": manifest.get("power_protocol"),
                "workload_protocol": manifest.get("workload_protocol"),
                "timing_protocol": manifest.get("timing_protocol"),
                "output_schema": manifest.get("output_schema"),
                "claim_scope": manifest.get("claim_scope"),
            }
        ),
        "reference_provenance": manifest.get("content_hashes"),
    }
    if manifest.get("experiment_id") is not None:
        normalized["experiment_id"] = manifest["experiment_id"]
    if phase == "confirm":
        normalized.update(
            performance_cells=performance_cells,
            performance_expected_shots_by_cell=performance_expected,
            performance_cell_batch_schedules=performance_schedules,
        )
    return normalized


def _reject_null_tree(value: Any, *, path: str) -> None:
    if value is None:
        raise ValueError(f"null is forbidden in frozen protocol field {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_null_tree(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_null_tree(item, path=f"{path}[{index}]")


def _validate_confirmatory_derivation(manifest: Mapping[str, Any]) -> None:
    """Recheck every pilot-derived literal embedded by the freeze command."""

    analysis = manifest.get("analysis_config")
    contract = manifest.get("scientific_contract")
    if not isinstance(analysis, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("confirmatory protocol is missing its scientific contract")
    selection = analysis.get("selection")
    provenance = analysis.get("pilot_provenance")
    if selection != contract.get("selection") or provenance != contract.get(
        "pilot_provenance"
    ):
        raise ValueError("confirmatory analysis and scientific contract disagree")
    if not isinstance(selection, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("confirmatory selection/pilot provenance must be objects")
    if provenance.get("selection_did_not_read_signed_difference") is not True:
        raise ValueError("pilot selection must certify that no signed contrast was read")
    for key in (
        "pilot_protocol_experiment_id",
        "pilot_protocol_sha256",
        "raw_pilot_result_sha256",
        "pilot_analysis_sha256",
        "selection_log_sha256",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"pilot provenance {key} must be a SHA-256 hex string")
        try:
            bytes.fromhex(value)
        except ValueError as ex:
            raise ValueError(f"pilot provenance {key} is not hexadecimal") from ex
    if provenance.get("pilot_source_hashes") != manifest.get("source_hashes"):
        raise ValueError(
            "confirmatory implementation hashes differ from the verified pilot; "
            "rerun the pilot"
        )

    log = provenance.get("selection_log")
    if not isinstance(log, Mapping) or log.get("schema") != "promatch-l1-pilot-selection-v1":
        raise ValueError("pilot provenance is missing the complete selection log")
    without_hash = dict(log)
    recorded_selection_hash = without_hash.pop("selection_sha256", None)
    computed_selection_hash = hashlib.sha256(
        json.dumps(
            without_hash,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        recorded_selection_hash != computed_selection_hash
        or provenance.get("selection_log_sha256") != computed_selection_hash
    ):
        raise ValueError("embedded pilot selection log hash mismatch")
    if log.get("selection_used_signed_difference") is not False:
        raise ValueError("pilot selection log used a forbidden signed contrast")
    if log.get("experiment_id") != provenance.get("pilot_protocol_experiment_id"):
        raise ValueError("pilot selection/protocol experiment IDs differ")
    rows = log.get("cells")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("pilot selection log must contain all five ordered cells")
    expected_ids = [
        "pilot-01-d7-n6-y2-r28-p0.001",
        "pilot-02-d7-n6-y2-r28-p0.002",
        "pilot-03-d7-n6-y2-r28-p0.003",
        "pilot-04-d5-n6-y2-r20-p0.003",
        "pilot-05-d5-n6-y2-r20-p0.005",
    ]
    if [row.get("cell_id") for row in rows if isinstance(row, Mapping)] != expected_ids:
        raise ValueError("pilot selection log does not cover the frozen ordered grid")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("pilot selection rows must be objects")
        computed_pass = (
            isinstance(row.get("activation_fraction"), (int, float))
            and row["activation_fraction"] >= 0.05
            and isinstance(row.get("u0_failures"), int)
            and not isinstance(row.get("u0_failures"), bool)
            and row["u0_failures"] >= 200
            and isinstance(row.get("discordant_pairs"), int)
            and not isinstance(row.get("discordant_pairs"), bool)
            and row["discordant_pairs"] >= 100
            and row.get("integrity_checks_passed") is True
            and isinstance(row.get("confirmatory_shots"), int)
            and not isinstance(row.get("confirmatory_shots"), bool)
            and 0 < row["confirmatory_shots"] <= 10_000_000
            and isinstance(row.get("power_lower_bound"), (int, float))
            and row["power_lower_bound"] >= 0.9
        )
        if row.get("passed") is not bool(computed_pass):
            raise ValueError("pilot selection row pass flag disagrees with frozen gates")
    first_passing = next(
        (row for row in rows if isinstance(row, Mapping) and row.get("passed") is True),
        None,
    )
    if first_passing is None or log.get("selected") != first_passing:
        raise ValueError("selected cell is not the first passing pilot cell")

    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("confirmatory protocol must contain one selected accuracy cell")
    cell = cells[0]
    if selection.get("selected_cell_id") != cell.get("cell_id") or first_passing.get(
        "cell_id"
    ) != cell.get("cell_id"):
        raise ValueError("selected cell identity differs across pilot and holdout")
    nested = selection.get("selected_cell")
    expected_nested = {
        "cell_id": cell.get("cell_id"),
        "d": cell.get("d"),
        "patches": cell.get("patches"),
        "yokes": cell.get("yokes"),
        "rounds": cell.get("r"),
        "si1000_p": cell.get("p"),
    }
    if nested != expected_nested:
        raise ValueError("selected cell metadata differs from the frozen circuit cell")

    p_design = first_passing.get("p_u0_design")
    delta = first_passing.get("delta_noninferiority")
    n_confirm = first_passing.get("confirmatory_shots")
    gates = selection.get("gate_inputs")
    if not isinstance(gates, Mapping):
        raise ValueError("confirmatory gate_inputs must be an object")
    if (
        gates.get("integrity_checks_passed") is not True
        or gates.get("resource_gate_passed") is not True
        or gates.get("activation_fraction") != first_passing.get("activation_fraction")
        or gates.get("u0_direct_failures") != first_passing.get("u0_failures")
        or gates.get("discordant_pairs") != first_passing.get("discordant_pairs")
    ):
        raise ValueError("confirmatory gate inputs differ from the verified pilot")
    if (
        not isinstance(p_design, (int, float))
        or not isinstance(delta, (int, float))
        or not math.isclose(delta, 0.05 * p_design, rel_tol=0, abs_tol=1e-15)
        or selection.get("p_u0_design") != p_design
        or selection.get("delta_noninferiority") != delta
        or selection.get("discordance_clopper_pearson_upper")
        != first_passing.get("discordance_upper")
        or selection.get("normal_rule_raw_shots")
        != first_passing.get("normal_rule_raw_shots")
        or selection.get("power_verified_confirmatory_shots") != n_confirm
        or selection.get("n_confirm") != n_confirm
    ):
        raise ValueError("confirmatory design literals do not derive from the pilot")
    expected = manifest.get("expected_shots_by_cell", {}).get(cell["cell_id"])
    schedules = _phase_schedules(manifest, "confirm", cells)[cell["cell_id"]]
    if expected != n_confirm or sum(batch.shots for batch in schedules) != n_confirm:
        raise ValueError("confirmatory n_confirm does not equal the frozen schedule")


def validate_experiment_protocol(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    scientific: bool,
    processes: int | None = None,
    root: Path | None = None,
) -> str:
    """Fail-closed validation performed before creating an output directory."""

    if manifest.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema must be {PROTOCOL_SCHEMA!r}")
    if manifest.get("kind") != PROTOCOL_KIND:
        raise ValueError(f"protocol kind must be {PROTOCOL_KIND!r}")
    if phase not in {"pilot", "confirm", "target", "smoke"}:
        raise ValueError(f"unsupported phase {phase!r}")
    protocol_phase = manifest.get("phase")
    expected_protocol_phase = "confirm" if phase == "target" else phase
    if scientific and protocol_phase != expected_protocol_phase:
        raise ValueError(
            f"protocol phase {protocol_phase!r} does not match command {phase!r}"
        )
    if scientific and manifest.get("status") != "FROZEN":
        raise ValueError("pilot/confirm/target collection requires status='FROZEN'")
    if scientific and manifest.get("frozen") is not True:
        raise ValueError("pilot/confirm/target collection requires frozen=true")
    expected_claim_bearing = phase in {"confirm", "target"}
    if scientific and manifest.get("claim_bearing") is not expected_claim_bearing:
        raise ValueError(
            "claim_bearing must be false for pilot and true for the frozen "
            "first-round confirm/target protocol"
        )
    if manifest.get("sample_batch_size") != ROUND_ONE_BATCH_SIZE:
        raise ValueError(f"sample_batch_size must be {ROUND_ONE_BATCH_SIZE}")
    protocol_processes = validate_process_count(manifest.get("processes", 32))
    if processes is not None and protocol_processes != validate_process_count(processes):
        raise ValueError("CLI processes must exactly match frozen protocol processes")
    _decoder_config(manifest)
    if scientific and _decoder_config(manifest) != {
        "residual_hw_limit": 10,
        "domain_mode": "windowd",
        "boundary_policy": "disabled",
        "observable_policy": "zero-frame",
    }:
        raise ValueError("scientific collection requires the frozen primary PU-window decoder")
    _dem_options(manifest)
    replay_policy = _validate_replay_policy(manifest.get("replay_policy"))
    if scientific:
        analysis_config = manifest.get("analysis_config")
        if not isinstance(analysis_config, Mapping):
            raise ValueError("frozen analysis_config must be an object")
        output_schema = analysis_config.get("output_schema")
        if not isinstance(output_schema, Mapping):
            raise ValueError("frozen output_schema must be an object")
        if output_schema.get("schema_version") != 2:
            raise ValueError("frozen output_schema.schema_version must be 2")
        documented_policy = output_schema.get("bounded_replay_policy")
        if documented_policy != replay_policy:
            raise ValueError(
                "frozen output-schema replay policy differs from the executable policy"
            )
        scientific_contract = manifest.get("scientific_contract")
        if not isinstance(scientific_contract, Mapping):
            raise ValueError("frozen scientific_contract must be an object")
        expected_protocol_version = (
            "promatch-l1-pilot-v2"
            if phase == "pilot"
            else "promatch-l1-first-round-v2"
        )
        if scientific_contract.get("protocol_version") != expected_protocol_version:
            raise ValueError(
                "frozen scientific_contract has the wrong protocol_version"
            )
    cells = _phase_cells(manifest, phase)
    if len({cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)}) != len(cells):
        raise ValueError("cell IDs must be unique")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise TypeError("each cell must be an object")
        _validate_cell_inputs(cell, require_hashes=scientific)
    if scientific and phase == "pilot":
        expected_grid = [
            ("pilot-01-d7-n6-y2-r28-p0.001", 7, 6, 2, 28, 0.001, 0, 19),
            ("pilot-02-d7-n6-y2-r28-p0.002", 7, 6, 2, 28, 0.002, 20, 39),
            ("pilot-03-d7-n6-y2-r28-p0.003", 7, 6, 2, 28, 0.003, 40, 59),
            ("pilot-04-d5-n6-y2-r20-p0.003", 5, 6, 2, 20, 0.003, 60, 79),
            ("pilot-05-d5-n6-y2-r20-p0.005", 5, 6, 2, 20, 0.005, 80, 99),
        ]
        actual_grid = []
        for cell in cells:
            schedule = manifest.get("cell_batch_schedules", {}).get(cell["cell_id"], [])
            if not schedule:
                raise ValueError("pilot cell has no batch schedule")
            actual_grid.append(
                (
                    cell["cell_id"], cell["d"], cell["patches"], cell["yokes"],
                    cell["r"], cell["p"], schedule[0]["batch_id"],
                    schedule[-1]["batch_id"],
                )
            )
            if manifest.get("expected_shots_by_cell", {}).get(cell["cell_id"]) != 200_000:
                raise ValueError("each scientific pilot cell must have exactly 200000 shots")
        if actual_grid != expected_grid:
            raise ValueError("scientific pilot must use the exact ordered five-cell grid")
    if scientific and phase == "confirm":
        analysis = manifest.get("analysis_config")
        if not isinstance(analysis, Mapping):
            raise ValueError("confirmatory protocol requires analysis_config")
        selection = analysis.get("selection", {})
        accuracy = analysis.get("accuracy_protocol", {})
        power = analysis.get("power_protocol", {})
        timing = analysis.get("timing_protocol", {})
        required_positive = {
            "selection.p_u0_design": selection.get("p_u0_design"),
            "selection.delta_noninferiority": selection.get("delta_noninferiority"),
            "selection.n_confirm": selection.get("n_confirm"),
            "accuracy.alpha_one_sided": accuracy.get("alpha_one_sided"),
            "power.simulation_replicates": power.get("simulation_replicates"),
            "power.simulation_seed": power.get("simulation_seed"),
            "timing.bootstrap_replicates": timing.get("bootstrap_replicates"),
        }
        for name, value in required_positive.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"confirmatory protocol requires positive literal {name}")
        if not isinstance(timing.get("claim_gates"), Mapping):
            raise ValueError("confirmatory protocol requires literal timing claim_gates")
    if scientific and protocol_phase == "confirm":
        performance_fields = (
            "performance_cells",
            "performance_expected_shots_by_cell",
            "performance_cell_batch_schedules",
        )
        for field in performance_fields:
            if field not in manifest:
                raise ValueError(f"first-round protocol is missing {field}")
            _reject_null_tree(manifest[field], path=field)
        performance_cells = _phase_cells(manifest, "target")
        for cell in performance_cells:
            _validate_cell_inputs(cell, require_hashes=True)
        performance_schedules = _phase_schedules(
            manifest, "target", performance_cells
        )
        expected_target = {
            "cell_id": "target-d11-n6-y2-r44-p0.001",
            "d": 11,
            "patches": 6,
            "yokes": 2,
            "r": 44,
            "p": 0.001,
            "generator": GENERATOR,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
        }
        if len(performance_cells) != 1 or any(
            performance_cells[0].get(key) != value
            for key, value in expected_target.items()
        ):
            raise ValueError("target phase requires the exact frozen d11 target geometry")
        target_schedule_rows = performance_schedules[expected_target["cell_id"]]
        if (
            manifest["performance_expected_shots_by_cell"].get(expected_target["cell_id"])
            != 1_000_000
            or len(target_schedule_rows) != 100
            or [row.batch_id for row in target_schedule_rows] != list(range(100))
            or any(row.shots != ROUND_ONE_BATCH_SIZE for row in target_schedule_rows)
        ):
            raise ValueError(
                "target phase requires exactly 1000000 shots in batch IDs 0..99"
            )
        _validate_confirmatory_derivation(manifest)
    required = _required_protocol_fields() if scientific else ()
    if scientific:
        for field in required:
            if field in manifest:
                _reject_null_tree(manifest[field], path=field)
    experiment_id = validate_protocol_manifest(manifest, required_fields=required)
    split = _protocol_split(manifest, phase)
    _phase_schedules(manifest, phase, cells)

    if scientific:
        if os.environ.get("MAX_ERRORS") is not None:
            raise ValueError("MAX_ERRORS is forbidden for fixed-shot scientific collection")
        root = _repo_root() if root is None else root.resolve()
        state = repository_state(root)
        if manifest["clean_worktree"] is not True or not state["clean_worktree"]:
            raise ValueError("scientific collection requires a clean worktree")
        if state["repository_commit"] != manifest["repository_commit"]:
            _validate_post_freeze_protocol_commit(
                manifest=manifest,
                root=root,
                frozen_base=manifest["repository_commit"],
                current_head=state["repository_commit"],
            )
        versions = current_software_versions()
        if manifest["software_versions"] != versions:
            raise ValueError(
                f"software version mismatch: protocol={manifest['software_versions']!r}, "
                f"actual={versions!r}"
            )
        environment = current_execution_environment()
        if manifest.get("execution_environment") != environment:
            raise ValueError(
                "execution environment differs from the frozen protocol: "
                f"protocol={manifest.get('execution_environment')!r}, "
                f"actual={environment!r}"
            )
        if manifest.get("seed_derivation") != SEED_DERIVATION:
            raise ValueError("unsupported seed derivation rule")
        hashes = manifest["source_hashes"]
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError("source_hashes must be a nonempty object")
        for relative, expected in hashes.items():
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"invalid source-hash path {relative!r}")
            if _canonical_file_hash(path) != expected:
                raise ValueError(f"source hash mismatch for {relative}")
    return experiment_id


def freeze_protocol(
    draft: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Populate all derived provenance without sampling any shots."""

    root = _repo_root() if root is None else root.resolve()
    template_sha256 = hashlib.sha256(canonical_json_bytes(dict(draft))).hexdigest()
    frozen = normalize_protocol(draft, for_freeze=True)
    if frozen.get("schema") not in (None, PROTOCOL_SCHEMA):
        raise ValueError("cannot freeze an unsupported protocol schema")
    phase = frozen.get("phase")
    if phase not in {"pilot", "confirm"}:
        raise ValueError("draft phase must be 'pilot' or 'confirm'")
    frozen["schema"] = PROTOCOL_SCHEMA
    frozen["kind"] = PROTOCOL_KIND
    frozen["status"] = "FROZEN"
    frozen["frozen"] = True
    frozen["claim_bearing"] = phase == "confirm"
    frozen["sample_batch_size"] = ROUND_ONE_BATCH_SIZE
    frozen["processes"] = validate_process_count(frozen.get("processes", 32))
    frozen["seed_derivation"] = SEED_DERIVATION
    frozen["clean_worktree"] = True
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("freeze requires a clean worktree")
    frozen["repository_commit"] = state["repository_commit"]
    frozen["software_versions"] = current_software_versions()
    frozen["execution_environment"] = current_execution_environment()
    frozen["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    if frozen.get("created_utc") is None:
        frozen["created_utc"] = frozen["frozen_utc"]
    frozen["template_sha256"] = template_sha256
    source_paths = frozen.pop("source_paths", None)
    if source_paths is None:
        source_paths = _default_source_paths(root)
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("source_paths must be a nonempty array")
    frozen["source_hashes"] = {
        relative: _canonical_file_hash((root / relative).resolve())
        for relative in source_paths
    }
    _decoder_config(frozen)
    dem_options = _dem_options(frozen)
    cells = frozen.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cells must be a nonempty array")
    populated = []
    for raw_cell in cells:
        cell = dict(raw_cell)
        prepared = prepare_cell(
            cell,
            decoder_config=_decoder_config(frozen),
            dem_options=dem_options,
            verify_hashes=False,
        )
        cell.update(prepared.provenance)
        populated.append(cell)
    frozen["cells"] = populated
    _cell_schedules(frozen, populated)
    if phase == "confirm":
        performance = frozen.get("performance_cells")
        if not isinstance(performance, list) or not performance:
            raise ValueError("confirmatory freeze requires performance_cells")
        populated_performance = []
        for raw_cell in performance:
            cell = dict(raw_cell)
            prepared = prepare_cell(
                cell,
                decoder_config=_decoder_config(frozen),
                dem_options=dem_options,
                verify_hashes=False,
            )
            cell.update(prepared.provenance)
            populated_performance.append(cell)
        frozen["performance_cells"] = populated_performance
        _phase_schedules(frozen, "target", populated_performance)
    frozen.pop("experiment_id", None)
    frozen["experiment_id"] = manifest_experiment_id(frozen)
    validate_experiment_protocol(
        frozen,
        phase=phase,
        scientific=True,
        processes=frozen["processes"],
        root=root,
    )
    return frozen


def inspect_protocol(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return actual cell provenance for a draft/frozen protocol."""

    manifest = normalize_protocol(manifest)
    decoder = _decoder_config(manifest)
    dem_options = _dem_options(manifest)
    result: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "repository": repository_state(),
        "software_versions": current_software_versions(),
    }
    cache: dict[str, dict[str, Any]] = {}
    for key in ("cells", "performance_cells"):
        if key not in manifest:
            continue
        inspected = []
        for cell in manifest[key]:
            cache_key = json.dumps(cell, sort_keys=True, separators=(",", ":"))
            provenance = cache.get(cache_key)
            if provenance is None:
                prepared = prepare_cell(
                    cell,
                    decoder_config=decoder,
                    dem_options=dem_options,
                    verify_hashes=False,
                )
                provenance = prepared.provenance
                cache[cache_key] = provenance
            inspected.append({"cell_id": cell["cell_id"], **provenance})
        result[key] = inspected
    return result


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda e: str(e[0]))}


def _telemetry(
    results: Iterable[Any], original: np.ndarray, residual: np.ndarray, layout: Any
) -> dict[str, Any]:
    original_hw = np.count_nonzero(original, axis=1)
    residual_hw = np.count_nonzero(residual, axis=1)
    joint_hw = Counter(
        f"{int(before)},{int(after)}"
        for before, after in zip(original_hw, residual_hw)
    )
    status = Counter()
    fallback = Counter()
    domain_initial = Counter()
    domain_attempted = Counter()
    domain_final = Counter()
    attempted_stages = np.zeros(4, dtype=np.int64)
    committed_stages = np.zeros(4, dtype=np.int64)
    attempted_matches = committed_matches = 0
    boundary_added = boundary_used = boundary_discarded = 0
    rollback_shots = success_shots = activated_shots = 0
    decision_weight = Counter()
    xor_support_weight = Counter()
    path_lengths = Counter()
    domain_identity = Counter()
    for result in results:
        decision_weight[float(result.decision_weight).hex()] += 1
        xor_support_weight[float(result.xor_support_weight).hex()] += 1
        for path in result.paths:
            path_lengths[len(path.edge_ids)] += 1
        shot_activated = shot_rollback = shot_success = False
        for domain, item in result.domain_stats.items():
            domain_key = repr(domain)
            domain_identity[f"{domain_key}|domains"] += 1
            domain_identity[f"{domain_key}|initial_events"] += item.initial_hw
            domain_identity[f"{domain_key}|attempted_residual_events"] += item.attempted_residual_hw
            domain_identity[f"{domain_key}|final_residual_events"] += item.final_residual_hw
            domain_identity[f"{domain_key}|status={item.status}"] += 1
            status[item.status] += 1
            domain_initial[item.initial_hw] += 1
            domain_attempted[item.attempted_residual_hw] += 1
            domain_final[item.final_residual_hw] += 1
            attempted_stages += np.asarray(item.attempted_stage_counts)
            committed_stages += np.asarray(item.committed_stage_counts)
            attempted_matches += item.attempted_matches
            committed_matches += item.committed_matches
            boundary_added += int(item.boundary_was_added)
            boundary_used += int(item.boundary_was_used)
            boundary_discarded += int(item.boundary_discarded_unused)
            shot_activated |= item.status != "below-limit"
            shot_rollback |= item.status == "rollback"
            shot_success |= item.status == "success"
            if item.fallback_reason is not None:
                fallback[item.fallback_reason.value] += 1
        activated_shots += int(shot_activated)
        rollback_shots += int(shot_rollback)
        success_shots += int(shot_success)
    return {
        "shots": int(original.shape[0]),
        "original_event_sum": int(original_hw.sum()),
        "residual_event_sum": int(residual_hw.sum()),
        "original_hw_histogram": _counter_json(Counter(map(int, original_hw))),
        "residual_hw_histogram": _counter_json(Counter(map(int, residual_hw))),
        "original_residual_hw_joint_histogram": _counter_json(joint_hw),
        "domain_initial_hw_histogram": _counter_json(domain_initial),
        "domain_attempted_hw_histogram": _counter_json(domain_attempted),
        "domain_final_hw_histogram": _counter_json(domain_final),
        "domain_status_counts": _counter_json(status),
        "domain_identity_counts": _counter_json(domain_identity),
        "fallback_reason_counts": _counter_json(fallback),
        "decision_weight_histogram_float_hex": _counter_json(decision_weight),
        "xor_support_weight_histogram_float_hex": _counter_json(xor_support_weight),
        "committed_path_length_histogram": _counter_json(path_lengths),
        "terminal_withheld_event_sum": int(
            np.count_nonzero(original[:, layout.terminal_detector_ids])
        ),
        "yoke_withheld_event_sum": int(
            np.count_nonzero(original[:, layout.yoke_detector_ids])
        ),
        "attempted_stage_counts": [int(v) for v in attempted_stages],
        "committed_stage_counts": [int(v) for v in committed_stages],
        "attempted_matches": int(attempted_matches),
        "committed_matches": int(committed_matches),
        "boundary_added_domains": boundary_added,
        "boundary_used_domains": boundary_used,
        "boundary_discarded_domains": boundary_discarded,
        "activated_shots": activated_shots,
        "rollback_shots": rollback_shots,
        "success_shots": success_shots,
    }


def _prediction_failed(prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if prediction.shape != actual.shape:
        raise AssertionError(
            f"prediction/observable shape mismatch {prediction.shape} != {actual.shape}"
        )
    return np.any(np.bitwise_xor(prediction, actual) != 0, axis=1)


def _replay_sample(
    *,
    cell_id: str,
    category: str,
    batch: BatchSpec,
    offset: int,
    seed: int,
    dets: np.ndarray,
    obs: np.ndarray,
    u0: np.ndarray,
    pu: np.ndarray,
) -> dict[str, Any]:
    shot_index = batch.shot_start + offset
    return {
        "selection_sha256": _replay_selection_sha256(
            cell_id=cell_id,
            batch_id=batch.batch_id,
            shot_index=shot_index,
            category=category,
        ),
        "category": category,
        "batch_id": batch.batch_id,
        "shot_offset": offset,
        "shot_index": shot_index,
        "stim_seed": seed,
        "detection_events_hex": bytes(dets[offset]).hex(),
        "observables_hex": bytes(obs[offset]).hex(),
        "u0_prediction_hex": bytes(u0[offset]).hex(),
        "pu_prediction_hex": bytes(pu[offset]).hex(),
    }


def _replay_selection_sha256(
    *, cell_id: str, batch_id: int, shot_index: int, category: str
) -> str:
    if category not in REPLAY_CATEGORIES:
        raise ValueError(f"unsupported replay category {category!r}")
    selection_key = f"{cell_id}:{batch_id}:{shot_index}:{category}".encode(
        "ascii"
    )
    return _sha256_bytes(selection_key)


def _retain_lowest_hash_samples(
    samples: Iterable[dict[str, Any]], cap: int
) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda sample: sample["selection_sha256"])[:cap]


def collect_prepared_batch(
    prepared: PreparedCell,
    *,
    batch: BatchSpec,
    seed_root: str,
    experiment_id: str,
    phase: str,
    replay_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect one immutable paired batch (also useful for focused tests)."""

    replay_policy = _validate_replay_policy(replay_policy)
    batch_candidate_cap = replay_policy[
        "maximum_candidate_rows_per_category_per_batch_ledger"
    ]
    seed = derive_stim_batch_seed(seed_root=seed_root, batch_id=batch.batch_id)
    sampler = prepared.circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(
        shots=batch.shots, separate_observables=True, bit_packed=True
    )
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    dets.setflags(write=False)
    obs.setflags(write=False)
    det_digest = digest_array(dets)
    obs_digest = digest_array(obs)

    # U0 is ordinary *uncorrelated* PyMatching on the original syndrome.
    u0 = np.asarray(
        prepared.compiled_pu.graph.matcher.decode_batch(
            dets, bit_packed_shots=True, bit_packed_predictions=True
        ),
        dtype=np.uint8,
    )
    unpacked = np.unpackbits(
        dets,
        axis=1,
        count=prepared.dem.num_detectors,
        bitorder="little",
    )
    residual, frames, prematch_results = prepared.compiled_pu.predecode_shots(unpacked)
    residual_packed = np.packbits(residual, axis=1, bitorder="little")
    pu = np.asarray(
        prepared.compiled_pu.graph.matcher.decode_batch(
            residual_packed, bit_packed_shots=True, bit_packed_predictions=True
        ),
        dtype=np.uint8,
    )
    if prepared.dem.num_observables:
        packed_frames = np.packbits(frames, axis=1, bitorder="little")
        pu ^= packed_frames
    for shot_index, result in enumerate(prematch_results):
        if all(
            item.initial_hw <= prepared.compiled_pu.residual_hw_limit
            for item in result.domain_stats.values()
        ) and not np.array_equal(pu[shot_index], u0[shot_index]):
            raise AssertionError(
                "inactive/below-limit shot differs between PU and U0"
            )
    # Explicit immutability check catches accidental in-place decoder behavior.
    if digest_array(dets) != det_digest or digest_array(obs) != obs_digest:
        raise AssertionError("a decoder mutated the shared paired-shot corpus")

    baseline_failed = _prediction_failed(u0, obs)
    treatment_failed = _prediction_failed(pu, obs)
    table = PairedContingency.from_failures(
        baseline_failed=baseline_failed, treatment_failed=treatment_failed
    )
    replay = []
    category_offsets = {
        "regression": np.flatnonzero(~baseline_failed & treatment_failed),
        "recovery": np.flatnonzero(baseline_failed & ~treatment_failed),
        "rollback": np.asarray(
            [
                i
                for i, result in enumerate(prematch_results)
                if any(s.status == "rollback" for s in result.domain_stats.values())
            ],
            dtype=np.int64,
        ),
    }
    if tuple(category_offsets) != REPLAY_CATEGORIES:
        raise AssertionError("collector replay categories drifted from the protocol")
    for category in REPLAY_CATEGORIES:
        offsets = category_offsets[category]
        candidates = [
            _replay_sample(
                cell_id=prepared.cell["cell_id"],
                category=category,
                batch=batch,
                offset=int(offset),
                seed=seed,
                dets=dets,
                obs=obs,
                u0=u0,
                pu=pu,
            )
            for offset in offsets
        ]
        replay.extend(_retain_lowest_hash_samples(candidates, batch_candidate_cap))
    return {
        "schema": LEDGER_SCHEMA,
        "experiment_id": experiment_id,
        "phase": phase,
        "cell_id": prepared.cell["cell_id"],
        "batch": dataclasses.asdict(batch),
        "stim_seed": seed,
        "detectors": _jsonable_digest(det_digest),
        "observables": _jsonable_digest(obs_digest),
        "provenance": prepared.provenance,
        "paired_contingency": dataclasses.asdict(table),
        "telemetry": _telemetry(
            prematch_results, unpacked, residual, prepared.compiled_pu.graph.layout
        ),
        "replay_samples": replay,
    }


_WORKER_CACHE: dict[str, PreparedCell] = {}


def _worker_collect(task: dict[str, Any]) -> dict[str, Any]:
    configure_single_thread_runtime()
    cell = task["cell"]
    cell_id = cell["cell_id"]
    prepared = _WORKER_CACHE.get(cell_id)
    if prepared is None:
        prepared = prepare_cell(
            cell,
            decoder_config=task["decoder"],
            dem_options=task["dem_options"],
            verify_hashes=task["verify_hashes"],
        )
        _WORKER_CACHE[cell_id] = prepared
    return collect_prepared_batch(
        prepared,
        batch=BatchSpec.from_json(task["batch"]),
        seed_root=task["seed_root"],
        experiment_id=task["experiment_id"],
        phase=task["phase"],
        replay_policy=task["replay_policy"],
    )


def _ledger_path(out: Path, *, cell_id: str, batch_id: int) -> Path:
    return out / "batches" / cell_id / f"batch-{batch_id:08d}.json"


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = os.environ.get("TMPDIR")
    if not scratch_root:
        raise RuntimeError("TMPDIR must be set for atomic ledger writes")
    fd, temporary_name = tempfile.mkstemp(prefix="promatch-ledger-", dir=scratch_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(temporary_name, path)
        except OSError:
            # Cross-device output: copy a completed immutable scratch file.
            with open(temporary_name, "rb") as source, open(path, "xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            os.unlink(temporary_name)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _verify_regenerated_scientific_ledger(
    existing: Mapping[str, Any], regenerated: Mapping[str, Any], *, path: str
) -> None:
    if existing != regenerated:
        raise ValueError(
            f"regenerated scientific batch differs from ledger {path}"
        )


def _validate_ledger_row(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    phase: str,
    cell: Mapping[str, Any],
    batch: BatchSpec,
    seed_root: str,
    expected_provenance: Mapping[str, Any],
    replay_policy: Mapping[str, Any],
) -> None:
    replay_policy = _validate_replay_policy(replay_policy)
    batch_candidate_cap = replay_policy[
        "maximum_candidate_rows_per_category_per_batch_ledger"
    ]
    if row.get("schema") != LEDGER_SCHEMA:
        raise ValueError("existing ledger has an unsupported schema")
    if row.get("experiment_id") != experiment_id or row.get("phase") != phase:
        raise ValueError("existing ledger protocol identity mismatch")
    if row.get("cell_id") != cell["cell_id"] or row.get("batch") != dataclasses.asdict(batch):
        raise ValueError("existing ledger cell/batch identity mismatch")
    expected_seed = derive_stim_batch_seed(
        seed_root=seed_root, batch_id=batch.batch_id
    )
    if row.get("stim_seed") != expected_seed:
        raise ValueError("existing ledger Stim seed mismatch")
    if row.get("provenance") != dict(expected_provenance):
        raise ValueError("existing ledger circuit/DEM/graph provenance mismatch")
    table = row.get("paired_contingency")
    expected_table_fields = {
        "both_correct", "regressions", "recoveries", "both_wrong"
    }
    if not isinstance(table, Mapping) or set(table) != expected_table_fields:
        raise ValueError("existing ledger has invalid contingency fields")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in table.values()):
        raise ValueError("existing ledger has invalid contingency counts")
    if sum(table.values()) != batch.shots:
        raise ValueError("existing ledger contingency does not reconcile to batch shots")
    expected_widths = {
        "detectors": (int(expected_provenance["num_detectors"]) + 7) // 8,
        "observables": (int(expected_provenance["num_observables"]) + 7) // 8,
    }
    for key, expected_width in expected_widths.items():
        digest = row.get(key)
        if not isinstance(digest, Mapping) or set(digest) != {"sha256", "shape", "dtype"}:
            raise ValueError(f"existing ledger has invalid {key} digest")
        if (
            not isinstance(digest["sha256"], str)
            or len(digest["sha256"]) != 64
            or digest["shape"] != [batch.shots, expected_width]
            or digest["dtype"] != "|u1"
        ):
            raise ValueError(f"existing ledger has mismatched {key} digest metadata")
    telemetry = row.get("telemetry")
    if not isinstance(telemetry, Mapping) or telemetry.get("shots") != batch.shots:
        raise ValueError("existing ledger telemetry does not reconcile to batch shots")
    rollback_shots = telemetry.get("rollback_shots")
    if (
        isinstance(rollback_shots, bool)
        or not isinstance(rollback_shots, int)
        or not 0 <= rollback_shots <= batch.shots
    ):
        raise ValueError("existing ledger telemetry has invalid rollback_shots")
    replay = row.get("replay_samples")
    if not isinstance(replay, list):
        raise ValueError("existing ledger replay_samples must be an array")
    replay_fields = {
        "selection_sha256",
        "category",
        "batch_id",
        "shot_offset",
        "shot_index",
        "stim_seed",
        "detection_events_hex",
        "observables_hex",
        "u0_prediction_hex",
        "pu_prediction_hex",
    }
    replay_counts: Counter[str] = Counter()
    selection_hashes: set[str] = set()
    for sample in replay:
        if not isinstance(sample, Mapping) or set(sample) != replay_fields:
            raise ValueError("existing ledger has a malformed replay sample")
        category = sample["category"]
        if category not in REPLAY_CATEGORIES:
            raise ValueError(
                f"existing ledger has unsupported replay category {category!r}"
            )
        replay_counts[category] += 1
        offset = sample["shot_offset"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < batch.shots
        ):
            raise ValueError("existing ledger replay shot_offset is outside its batch")
        shot_index = batch.shot_start + offset
        if (
            sample["batch_id"] != batch.batch_id
            or sample["shot_index"] != shot_index
            or sample["stim_seed"] != expected_seed
        ):
            raise ValueError("existing ledger replay identity does not match its batch")
        expected_selection = _replay_selection_sha256(
            cell_id=str(cell["cell_id"]),
            batch_id=batch.batch_id,
            shot_index=shot_index,
            category=category,
        )
        if sample["selection_sha256"] != expected_selection:
            raise ValueError("existing ledger replay selection hash mismatch")
        if expected_selection in selection_hashes:
            raise ValueError("existing ledger contains a duplicate replay sample")
        selection_hashes.add(expected_selection)

        decoded: dict[str, bytes] = {}
        replay_widths = {
            "detection_events_hex": expected_widths["detectors"],
            "observables_hex": expected_widths["observables"],
            "u0_prediction_hex": expected_widths["observables"],
            "pu_prediction_hex": expected_widths["observables"],
        }
        for key, expected_width in replay_widths.items():
            value = sample[key]
            if not isinstance(value, str) or len(value) != expected_width * 2:
                raise ValueError(f"existing ledger replay {key} has the wrong width")
            if any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(
                    f"existing ledger replay {key} is not canonical lowercase hex"
                )
            decoded[key] = bytes.fromhex(value)
        observable = decoded["observables_hex"]
        u0_failed = any(
            actual ^ predicted
            for actual, predicted in zip(observable, decoded["u0_prediction_hex"])
        )
        pu_failed = any(
            actual ^ predicted
            for actual, predicted in zip(observable, decoded["pu_prediction_hex"])
        )
        # Correctness categories are self-verifying from the retained logical
        # predictions. Rollback is an internal predecoder control-flow state:
        # its detector corpus is replayable here, while scientific collection
        # authenticates the status by regenerating and comparing the complete
        # deterministic batch payload before analysis.
        if category == "regression" and (u0_failed or not pu_failed):
            raise ValueError("existing ledger replay is not a regression")
        if category == "recovery" and (not u0_failed or pu_failed):
            raise ValueError("existing ledger replay is not a recovery")
    if any(value > batch_candidate_cap for value in replay_counts.values()):
        raise ValueError("existing ledger exceeds the bounded replay-sample policy")
    expected_replay_counts = {
        "regression": min(batch_candidate_cap, int(table["regressions"])),
        "recovery": min(batch_candidate_cap, int(table["recoveries"])),
        "rollback": min(batch_candidate_cap, rollback_shots),
    }
    if any(
        replay_counts[category] != expected_replay_counts[category]
        for category in REPLAY_CATEGORIES
    ):
        raise ValueError(
            "existing ledger replay samples are incomplete for the frozen "
            "deterministic retention policy"
        )


def _sum_counter_dict(target: Counter[str], value: Mapping[str, int]) -> None:
    for key, count in value.items():
        target[key] += int(count)


def summarize_ledgers(
    rows: Iterable[Mapping[str, Any]],
    *,
    experiment_id: str,
    phase: str,
    replay_policy: Mapping[str, Any],
) -> dict[str, Any]:
    replay_policy = _validate_replay_policy(replay_policy)
    summary_cell_cap = replay_policy[
        "maximum_retained_rows_per_category_per_cell_summary"
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    cell_summaries = []
    for cell_id in sorted(grouped):
        cell_rows = sorted(grouped[cell_id], key=lambda e: e["batch"]["shot_start"])
        contingency = Counter()
        scalar_telemetry = Counter()
        dict_telemetry: dict[str, Counter[str]] = {}
        stage_telemetry: dict[str, np.ndarray] = {}
        replay: dict[str, list[dict[str, Any]]] = {
            category: [] for category in REPLAY_CATEGORIES
        }
        for row in cell_rows:
            contingency.update(row["paired_contingency"])
            for key, value in row["telemetry"].items():
                if isinstance(value, int):
                    scalar_telemetry[key] += value
                elif isinstance(value, list):
                    stage_telemetry.setdefault(key, np.zeros(len(value), dtype=np.int64))
                    stage_telemetry[key] += np.asarray(value, dtype=np.int64)
                elif isinstance(value, Mapping):
                    _sum_counter_dict(dict_telemetry.setdefault(key, Counter()), value)
            for sample in row["replay_samples"]:
                category = sample["category"]
                replay[category].append(sample)
        telemetry: dict[str, Any] = dict(scalar_telemetry)
        telemetry.update({k: _counter_json(v) for k, v in dict_telemetry.items()})
        telemetry.update({k: [int(x) for x in v] for k, v in stage_telemetry.items()})
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "batches": len(cell_rows),
                "shots": sum(r["batch"]["shots"] for r in cell_rows),
                "paired_contingency": dict(contingency),
                "telemetry": telemetry,
                "replay_samples": [
                    sample
                    for category in replay.values()
                    for sample in _retain_lowest_hash_samples(category, summary_cell_cap)
                ],
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": experiment_id,
        "phase": phase,
        "collection_scope": (
            "target-performance-only-no-accuracy-confirmation"
            if phase == "target"
            else "paired-accuracy-and-workload"
        ),
        "cells": cell_summaries,
    }


def run_collection(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    out: Path,
    processes: int = 32,
    scientific: bool = True,
) -> dict[str, Any]:
    processes = validate_process_count(processes)
    manifest = normalize_protocol(manifest)
    experiment_id = validate_experiment_protocol(
        manifest, phase=phase, scientific=scientific, processes=processes
    )
    configure_single_thread_runtime()
    split = _protocol_split(manifest, phase)
    collection_cells = _phase_cells(manifest, phase)
    schedules = _phase_schedules(manifest, phase, collection_cells)
    decoder = _decoder_config(manifest)
    dem_options = _dem_options(manifest)

    # Validate every expensive provenance object before creating output state.
    prepared_provenance: dict[str, dict[str, Any]] = {}
    for cell in collection_cells:
        prepared = prepare_cell(
            cell,
            decoder_config=decoder,
            dem_options=dem_options,
            verify_hashes=scientific,
        )
        prepared_provenance[cell["cell_id"]] = prepared.provenance
    out = out.resolve()
    identity_path = out / "experiment.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity != {
            "schema": PROTOCOL_SCHEMA,
            "experiment_id": experiment_id,
            "phase": phase,
        }:
            raise ValueError("output directory belongs to a different experiment")
    else:
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(
            identity_path,
            {"schema": PROTOCOL_SCHEMA, "experiment_id": experiment_id, "phase": phase},
        )
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("output protocol.json differs from the runtime manifest")
    else:
        _atomic_json_write(protocol_path, manifest)

    completed: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    allowed_ledger_paths = {
        _ledger_path(out, cell_id=cell["cell_id"], batch_id=batch.batch_id)
        for cell in collection_cells
        for batch in schedules[cell["cell_id"]]
    }
    existing_ledger_paths = set((out / "batches").glob("*/*.json"))
    unexpected = existing_ledger_paths - allowed_ledger_paths
    if unexpected:
        raise ValueError(
            "output contains ledger rows outside the frozen schedule: "
            + ", ".join(str(path) for path in sorted(unexpected))
        )
    for cell in collection_cells:
        for batch in schedules[cell["cell_id"]]:
            path = _ledger_path(out, cell_id=cell["cell_id"], batch_id=batch.batch_id)
            if path.exists():
                row = json.loads(path.read_text())
                _validate_ledger_row(
                    row,
                    experiment_id=experiment_id,
                    phase=phase,
                    cell=cell,
                    batch=batch,
                    seed_root=manifest["sampler_seed_roots"][split],
                    expected_provenance=prepared_provenance[cell["cell_id"]],
                    replay_policy=manifest["replay_policy"],
                )
                if not scientific:
                    completed.append(row)
                    continue
            task = {
                    "cell": dict(cell),
                    "batch": dataclasses.asdict(batch),
                    "decoder": decoder,
                    "dem_options": dem_options,
                    "verify_hashes": scientific,
                    "seed_root": manifest["sampler_seed_roots"][split],
                    "experiment_id": experiment_id,
                    "phase": phase,
                    "replay_policy": manifest["replay_policy"],
                }
            if path.exists():
                # A scientific resume never trusts strings in an old ledger:
                # regenerate the immutable batch from its counter seed and
                # compare the complete deterministic scientific payload.
                task["verify_existing_path"] = str(path)
            tasks.append(task)

    if tasks:
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=configure_single_thread_runtime,
            # Python 3.14 defaults to forkserver on POSIX.  An explicit fork
            # context avoids an untracked Unix-domain forkserver and retains
            # the frozen parent environment in each short-lived worker.
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            future_to_task = {executor.submit(_worker_collect, task): task for task in tasks}
            for future in as_completed(future_to_task):
                row = future.result()
                task = future_to_task[future]
                path = _ledger_path(
                    out,
                    cell_id=task["cell"]["cell_id"],
                    batch_id=task["batch"]["batch_id"],
                )
                existing_path = task.get("verify_existing_path")
                if existing_path is not None:
                    existing = json.loads(Path(existing_path).read_text())
                    _verify_regenerated_scientific_ledger(
                        existing, row, path=existing_path
                    )
                else:
                    _atomic_json_write(path, row)
                completed.append(row)

    summary = summarize_ledgers(
        completed,
        experiment_id=experiment_id,
        phase=phase,
        replay_policy=manifest["replay_policy"],
    )
    _atomic_json_write(out / "summary.json", summary)
    return summary


def default_smoke_protocol(*, processes: int = 32, shots: int = 10_000) -> dict[str, Any]:
    processes = validate_process_count(processes)
    manifest: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "kind": PROTOCOL_KIND,
        "status": "DRAFT",
        "frozen": False,
        "claim_bearing": False,
        "phase": "smoke",
        "sample_batch_size": ROUND_ONE_BATCH_SIZE,
        "processes": processes,
        "seed_derivation": SEED_DERIVATION,
        "sampler_seed_roots": {"smoke": "3a" * 32},
        "expected_shots_by_cell": {"smoke-d3-n6-r12-p0.001-y2": shots},
        "cell_batch_schedules": {
            "smoke-d3-n6-r12-p0.001-y2": build_batch_schedule(shots)
        },
        "dem_options": {
            "decompose_errors": True,
            "approximate_disjoint_errors": True,
        },
        "decoder": {
            "residual_hw_limit": 10,
            "domain_mode": "windowd",
            "boundary_policy": "disabled",
            "observable_policy": "zero-frame",
        },
        "replay_policy": _canonical_replay_policy(4),
        "cells": [
            {
                "cell_id": "smoke-d3-n6-r12-p0.001-y2",
                "generator": GENERATOR,
                "d": 3,
                "r": 12,
                "p": 0.001,
                "patches": 6,
                "yokes": 2,
                "style": "cz",
                "noise": "si1000",
                "remove_x_yoke": False,
            }
        ],
    }
    manifest["experiment_id"] = manifest_experiment_id(manifest)
    return manifest


__all__ = [
    "GENERATOR",
    "LEDGER_SCHEMA",
    "PROTOCOL_KIND",
    "PROTOCOL_SCHEMA",
    "REPLAY_CATEGORIES",
    "SUMMARY_SCHEMA",
    "build_batch_schedule",
    "collect_prepared_batch",
    "configure_single_thread_runtime",
    "current_software_versions",
    "default_smoke_protocol",
    "freeze_protocol",
    "inspect_protocol",
    "normalize_protocol",
    "prepare_cell",
    "repository_state",
    "run_collection",
    "summarize_ledgers",
    "validate_experiment_protocol",
]
