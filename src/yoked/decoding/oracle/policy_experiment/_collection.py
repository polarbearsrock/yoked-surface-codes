"""Multi-worker campaign orchestration for one fixed B1 collection.

This slice owns output-root hygiene (immutable-corpus and symlink refusal),
the fork-based 32-process worker pool, whole-shard resume, campaign manifest
assembly with probe launch-gate projection, and the COLLECTION_READY marker.
It inherits the package isolation contract (see ``__init__``): the parent
process never touches per-shot policy logic or ground truth; it only
schedules workers and reconciles their authenticated shard manifests.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from yoked.decoding._promatch_experiment import (
    PreparedCell,
    _canonical_file_hash,
    configure_single_thread_runtime,
    prepare_cell,
    repository_state,
)

from yoked.decoding.oracle.policy_experiment._attestation import (
    _aggregate_collector_gate_attestations,
    _clean_tail_censor_attestation,
    _validate_collection_manifest,
)
from yoked.decoding.oracle.policy_experiment._identity import (
    EXPERIMENT_SCHEMA,
    MANIFEST_SCHEMA,
    SCIENTIFIC_SHOTS,
    SCIENTIFIC_WORKERS,
    WorkerSpec,
    _atomic_json,
    _strict_json_load,
    policy_worker_schedule,
)
from yoked.decoding.oracle.policy_experiment._protocol import (
    materialize_policy_draft,
    validate_policy_protocol,
)
from yoked.decoding.oracle.policy_experiment._shards import (
    _peak_rss_bytes,
    _shard_dir,
)


_WORKER_PREPARED: PreparedCell | None = None


def _worker_task(
    task: dict[str, Any],
) -> tuple[int, dict[str, Any], int, int]:
    # All collaborators are resolved through the package namespace at call
    # time, exactly like the pre-package module-global lookups (the worker
    # cache in particular is shared package state).
    from yoked.decoding.oracle import policy_experiment as _package

    configure_single_thread_runtime()
    compile_ns = 0
    if _package._WORKER_PREPARED is None:
        compile_start_ns = time.perf_counter_ns()
        _package._WORKER_PREPARED = prepare_cell(
            task["config"]["cell"],
            decoder_config=task["config"]["decoder"],
            dem_options=task["config"]["dem_options"],
            verify_hashes=task["scientific"],
        )
        compile_ns = time.perf_counter_ns() - compile_start_ns
    spec = WorkerSpec(**task["spec"])
    shard, payloads = _package.collect_policy_worker_shard(
        _package._WORKER_PREPARED, config=task["config"], mode=task["mode"], spec=spec
    )
    # Each worker owns a disjoint final shard directory.  Authenticate and
    # atomically install that shard here so compression transfer plus a second,
    # serial parent-side verification cannot dominate the 32-way campaign.
    _package.install_worker_shard(
        Path(task["out"]),
        shard=shard,
        payloads=payloads,
        config=task["config"],
        mode=task["mode"],
        spec=spec,
    )
    return spec.worker_id, shard, os.getpid(), compile_ns


# Path substrings of this workstation's historical corpora (AGENTS.md:
# out/promatch_l1_round1* are immutable audit artifacts) that B1 collection
# must never write into.
IMMUTABLE_OUTPUT_PATTERNS = ("promatch_l1_round1",)


def _reject_immutable_output(out: Path) -> None:
    text = out.resolve().as_posix()
    if any(pattern in text for pattern in IMMUTABLE_OUTPUT_PATTERNS):
        raise ValueError("B1 may not write into an immutable round-one corpus")


def _validate_output_root(out: Path) -> Path:
    """Rejects symlinks and any non-B1 material before creating artifacts."""

    candidate = out.absolute()
    if candidate.is_symlink():
        raise ValueError("B1 output root may not be a symlink")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("B1 output root must be a directory")
    if candidate.exists():
        allowed = {
            "experiment.json",
            "config.json",
            "manifest.json",
            "COLLECTION_READY",
            "shards",
        }
        entries = tuple(candidate.iterdir())
        unexpected = {entry.name for entry in entries} - allowed
        if unexpected:
            raise ValueError(
                f"B1 output root contains unexpected entries: {sorted(unexpected)}"
            )
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"B1 output entry may not be a symlink: {entry.name}")
        shard_root = candidate / "shards"
        if shard_root.exists() and not shard_root.is_dir():
            raise ValueError("B1 shards entry must be a directory")
    return candidate.resolve()


def run_policy_collection(
    config: Mapping[str, Any],
    *,
    mode: str,
    out: Path,
    processes: int = SCIENTIFIC_WORKERS,
    scientific: bool,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Runs or verifies one fixed B1 collection; COMPLETE remains analyzer-owned."""

    from yoked.decoding.oracle import policy_experiment as _package

    if processes != SCIENTIFIC_WORKERS:
        raise ValueError(
            "B1 smoke, probe, and scientific collection require exactly 32 processes"
        )
    if scientific and mode != "scientific":
        raise ValueError("scientific B1 collection requires mode='scientific'")
    if not scientific and mode not in {"smoke", "probe"}:
        raise ValueError("non-scientific B1 collection must be smoke or probe")
    out = _validate_output_root(out)
    _reject_immutable_output(out)
    campaign_start_ns = time.perf_counter_ns()
    experiment_id = validate_policy_protocol(
        config, scientific=scientific, protocol_path=protocol_path
    )
    if not scientific and "circuit_sha256" not in config["cell"]:
        config = materialize_policy_draft(config)
        experiment_id = str(config["experiment_id"])
    elif config.get("experiment_id") is None:
        config = {**dict(config), "experiment_id": experiment_id}
    configure_single_thread_runtime()
    if not scientific:
        scratch = Path(os.environ.get("TMPDIR", "")).resolve()
        if scratch not in out.parents:
            raise ValueError("smoke/probe output must live under TMPDIR")
    schedule = policy_worker_schedule(mode)

    # Validate expensive cell provenance before creating output state.
    setup_start_ns = time.perf_counter_ns()
    prepared = prepare_cell(
        config["cell"],
        decoder_config=config["decoder"],
        dem_options=config["dem_options"],
        verify_hashes=scientific,
    )
    del prepared
    setup_ns = time.perf_counter_ns() - setup_start_ns
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "implementation_commit": config.get("implementation_commit"),
        "config_commit": repository_state()["repository_commit"],
    }
    for name, value in (("experiment.json", identity), ("config.json", dict(config))):
        path = out / name
        if path.exists():
            if _strict_json_load(path) != value:
                raise ValueError(f"existing {name} belongs to another B1 experiment")
        else:
            _atomic_json(path, value)

    completed: dict[int, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    shard_root = out / "shards"
    if shard_root.exists():
        expected_directories = {f"worker-{spec.worker_id:02d}" for spec in schedule}
        unexpected = {path.name for path in shard_root.iterdir()} - expected_directories
        if unexpected:
            raise ValueError(
                f"B1 output contains unexpected shard entries: {sorted(unexpected)}"
            )
    for spec in schedule:
        path = _shard_dir(out, spec.worker_id)
        if path.exists():
            if path.is_symlink():
                raise ValueError(f"B1 worker shard may not be a symlink: {path}")
            completed[spec.worker_id] = _package.verify_worker_shard(
                path, config=config, mode=mode, spec=spec
            )
        else:
            tasks.append(
                {
                    "config": dict(config),
                    "mode": mode,
                    "spec": spec.to_json(),
                    "scientific": scientific,
                    "out": str(out),
                }
            )
    manifest_path = out / "manifest.json"
    ready_path = out / "COLLECTION_READY"
    if not tasks and manifest_path.exists() and ready_path.exists():
        previous_manifest = _strict_json_load(manifest_path)
        previous_ready = _strict_json_load(ready_path)
        _validate_collection_manifest(
            previous_manifest,
            config=config,
            mode=mode,
            schedule=schedule,
            shards=[completed[k] for k in sorted(completed)],
        )
        if (
            previous_manifest.get("schema") != MANIFEST_SCHEMA
            or previous_manifest.get("experiment_id") != config["experiment_id"]
            or previous_manifest.get("mode") != mode
            or previous_manifest.get("shards")
            != [completed[k] for k in sorted(completed)]
            or previous_ready
            != {
                "schema": "promatch-l1-policy-audit-collection-ready-v1",
                "experiment_id": config["experiment_id"],
                "mode": mode,
                "manifest_sha256": _canonical_file_hash(manifest_path),
                "verified_worker_shards": SCIENTIFIC_WORKERS,
                "verified_shots": sum(spec.shots for spec in schedule),
            }
        ):
            raise ValueError("existing B1 manifest/COLLECTION_READY identity mismatch")
        return previous_manifest
    if ready_path.exists():
        raise ValueError(
            "COLLECTION_READY exists but one or more worker shards are missing"
        )
    worker_pids: set[int] = set()
    worker_compile_ns: list[dict[str, int]] = []
    worker_phase_start_ns = time.perf_counter_ns()
    if tasks:
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=configure_single_thread_runtime,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            future_to_task = {
                executor.submit(_worker_task, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                scheduled = WorkerSpec(**future_to_task[future]["spec"])
                try:
                    worker_id, shard, pid, compile_ns = future.result()
                except Exception as ex:
                    raise RuntimeError(
                        f"B1 worker {scheduled.worker_id} shot range "
                        f"[{scheduled.shot_start}, {scheduled.shot_stop}) failed: {ex}"
                    ) from ex
                worker_pids.add(pid)
                worker_compile_ns.append(
                    {"worker_id": worker_id, "compile_ns": compile_ns}
                )
                completed[worker_id] = shard
    worker_phase_ns = time.perf_counter_ns() - worker_phase_start_ns
    if len(tasks) == SCIENTIFIC_WORKERS and len(worker_pids) != SCIENTIFIC_WORKERS:
        raise RuntimeError(
            "fresh B1 run did not observe exactly 32 distinct worker processes"
        )
    if set(completed) != set(range(SCIENTIFIC_WORKERS)):
        raise AssertionError("B1 collection is missing worker shards")
    tail_censor = _clean_tail_censor_attestation()
    tail_censor["censored_states"] = sum(
        int(shard["tail_censor_attestation"]["censored_states"])
        for shard in completed.values()
    )
    tail_censor["repeated_same_state_proposal_signatures"] = sum(
        int(shard["tail_censor_attestation"]["repeated_same_state_proposal_signatures"])
        for shard in completed.values()
    )
    tail_censor["output_truncations"] = sum(
        int(shard["tail_censor_attestation"]["output_truncations"])
        for shard in completed.values()
    )
    if tail_censor != _clean_tail_censor_attestation():
        raise RuntimeError(
            "B1 tail/censor gate failed; COLLECTION_READY was not written"
        )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "workers": SCIENTIFIC_WORKERS,
        "shots": sum(spec.shots for spec in schedule),
        "new_worker_processes_observed": len(worker_pids),
        "tail_censor_attestation": tail_censor,
        "shards": [completed[k] for k in sorted(completed)],
        "fatal_gate_attestations": _aggregate_collector_gate_attestations(
            [completed[k] for k in sorted(completed)]
        ),
        "performance_telemetry": {
            "schema": "promatch-l1-policy-audit-campaign-performance-v1",
            "parent_setup_ns": setup_ns,
            "worker_phase_ns": worker_phase_ns,
            "parent_peak_rss_bytes": _peak_rss_bytes(),
            "parent_peak_rss_source": (
                "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib"
            ),
            "new_worker_compile_ns": sorted(
                worker_compile_ns, key=lambda row: row["worker_id"]
            ),
            "scientifically_deterministic": False,
            "excluded_from_scientific_decisions": True,
        },
    }
    if mode == "probe":
        compressed_bytes = sum(
            path.stat().st_size
            for spec in schedule
            for path in _shard_dir(out, spec.worker_id).iterdir()
        )
        parallel_worker_compile_ns = max(
            (row["compile_ns"] for row in worker_compile_ns), default=0
        )
        fixed_setup_ns = setup_ns + parallel_worker_compile_ns
        variable_ns = max(0, worker_phase_ns - parallel_worker_compile_ns)
        gate = config["launch_gates"]
        # Scale the 100-shot probe up to the 20k scientific campaign, padded
        # by the frozen launch-gate headroom factor (1.5).
        probe_headroom_factor = gate["probe_headroom_factor"]
        probe_to_scientific_scale = SCIENTIFIC_SHOTS // sum(
            spec.shots for spec in schedule
        )
        projected_wall_seconds = fixed_setup_ns / 1e9 + (
            probe_headroom_factor * probe_to_scientific_scale * (variable_ns / 1e9)
        )
        projected_artifact_bytes = math.ceil(
            probe_headroom_factor * probe_to_scientific_scale * compressed_bytes
        )
        free_bytes = shutil.disk_usage(out).free
        manifest["probe_projection"] = {
            "parent_setup_seconds": setup_ns / 1e9,
            "parent_setup_seconds_hex": (setup_ns / 1e9).hex(),
            "parallel_worker_compile_seconds": parallel_worker_compile_ns / 1e9,
            "parallel_worker_compile_seconds_hex": (
                parallel_worker_compile_ns / 1e9
            ).hex(),
            "fixed_setup_seconds": fixed_setup_ns / 1e9,
            "fixed_setup_seconds_hex": (fixed_setup_ns / 1e9).hex(),
            "variable_100_shot_seconds": variable_ns / 1e9,
            "variable_100_shot_seconds_hex": (variable_ns / 1e9).hex(),
            "compressed_probe_bytes": compressed_bytes,
            "projected_wall_seconds": projected_wall_seconds,
            "projected_wall_seconds_hex": projected_wall_seconds.hex(),
            "projected_artifact_bytes": projected_artifact_bytes,
            "free_output_bytes": free_bytes,
            "wall_gate_passed": projected_wall_seconds
            <= gate["projected_wall_seconds_max"],
            "artifact_gate_passed": projected_artifact_bytes
            <= gate["projected_artifact_bytes_max"],
            "free_space_gate_passed": free_bytes
            >= max(gate["free_bytes_min"], 2 * projected_artifact_bytes),
        }
        manifest["probe_projection"]["all_launch_gates_passed"] = all(
            manifest["probe_projection"][name]
            for name in (
                "wall_gate_passed",
                "artifact_gate_passed",
                "free_space_gate_passed",
            )
        )
    manifest["campaign_wall_ns"] = time.perf_counter_ns() - campaign_start_ns
    _validate_collection_manifest(
        manifest,
        config=config,
        mode=mode,
        schedule=schedule,
        shards=[completed[k] for k in sorted(completed)],
    )
    _atomic_json(manifest_path, manifest)
    ready = {
        "schema": "promatch-l1-policy-audit-collection-ready-v1",
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "manifest_sha256": _canonical_file_hash(manifest_path),
        "verified_worker_shards": SCIENTIFIC_WORKERS,
        "verified_shots": sum(spec.shots for spec in schedule),
    }
    _atomic_json(ready_path, ready)
    return manifest
