"""B1 fixed-shot policy-audit protocol and deterministic shard collector.

This module is intentionally separate from the immutable V3 paired collector.
It owns only protocol/provenance, fixed worker schedules, sampling-once
orchestration, and artifact integrity.  The scientific per-shot policy logic
is reached through one narrow adapter, :func:`_audit_policy_shot`, so the core
implementation can evolve without leaking ground truth into its API.
"""

from __future__ import annotations

# These limits must be in force before importing NumPy/PyMatching indirectly.
import os

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_name in THREAD_ENVIRONMENT:
    os.environ[_thread_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
from decimal import Decimal, localcontext
import gzip
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from yoked.decoding._promatch_experiment import (
    GENERATOR,
    PreparedCell,
    _canonical_file_hash,
    _validate_post_freeze_protocol_commit,
    configure_single_thread_runtime,
    current_execution_environment,
    current_software_versions,
    prepare_cell,
    repository_state,
)
from yoked.decoding._promatch_oracle import (
    FullGraphOracle,
    OracleTolerance,
    classify_cost_excess,
)
from yoked.decoding._promatch_stats import canonical_json_bytes, digest_array


PROTOCOL_SCHEMA = "promatch-l1-policy-audit-protocol-v1"
EXPERIMENT_SCHEMA = "promatch-l1-policy-audit-experiment-v1"
MANIFEST_SCHEMA = "promatch-l1-policy-audit-manifest-v1"
SHARD_SCHEMA = "promatch-l1-policy-audit-shard-v1"
SHOT_SCHEMA = "promatch-l1-policy-audit-shot-v1"
PROPOSAL_SCHEMA = "promatch-l1-policy-audit-proposal-v1"
COUNTERFACTUAL_SCHEMA = "promatch-l1-policy-audit-counterfactual-v1"
DOMAIN_SCHEMA = "promatch-l1-policy-audit-domain-v1"
PROBE_ATTESTATION_SCHEMA = "promatch-l1-policy-audit-probe-attestation-v1"
SEED_DERIVATION = (
    "sha256-root+promatch-policy-worker-v1+experiment-id+cell-id+worker-id-"
    "first8-uint64le"
)
SCIENTIFIC_SHOTS = 20_000
SCIENTIFIC_WORKERS = 32
SCIENTIFIC_SHOTS_PER_WORKER = 625
GZIP_LEVEL = 9
ARM_IDS = (
    "u0-joint-y2",
    "pu-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2-shadow",
    "pu-ocost-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2",
    "pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2",
    "pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2",
)
POLICY_SOURCE_PATHS = (
    "requirements.txt",
    "src/yoked/_yoked_memory_circuits.py",
    "src/yoked/decoding/__init__.py",
    "src/yoked/decoding/_promatch.py",
    "src/yoked/decoding/_promatch_decoder.py",
    "src/yoked/decoding/_promatch_experiment.py",
    "src/yoked/decoding/_promatch_graph.py",
    "src/yoked/decoding/_promatch_layout.py",
    "src/yoked/decoding/_promatch_oracle.py",
    "src/yoked/decoding/_promatch_oracle_test.py",
    "src/yoked/decoding/_promatch_oracle_replay.py",
    "src/yoked/decoding/_promatch_oracle_replay_test.py",
    "src/yoked/decoding/_promatch_stats.py",
    "src/yoked/decoding/_promatch_stepper_test.py",
    "src/yoked/decoding/_promatch_policy_audit.py",
    "src/yoked/decoding/_promatch_policy_audit_test.py",
    "src/yoked/decoding/_promatch_policy_casebook.py",
    "src/yoked/decoding/_promatch_policy_casebook_test.py",
    "src/yoked/decoding/_promatch_policy_experiment.py",
    "src/yoked/decoding/_promatch_policy_experiment_test.py",
    "src/yoked/decoding/_promatch_policy_analysis.py",
    "src/yoked/decoding/_promatch_policy_analysis_test.py",
    "experiments/PROMATCH_L1_POLICY_AUDIT_20K.md",
    "tools/benchmark_promatch_policy_audit",
    "tools/analyze_promatch_policy_audit",
)
ANALYSIS_TABLE_NAMES = (
    "overview", "paired_outcomes", "event_and_transaction_summary",
    "residual_hw_distributions", "domain_terminal_summary",
    "certificate_by_stage", "certificate_by_domain", "unsafe_fraction_by_stage",
    "counterfactual_terminal_action", "first_safe_rank", "stage_transition",
    "context_views", "visibility_summary", "association_by_unsafe_count",
    "unsafe_count_distribution", "association_by_first",
    "first_conflict_discordant", "continuous_distributions",
    "local_competitor_summary", "cost_excess_ecdf_by_stage",
    "cost_excess_ecdf_by_certificate", "cost_excess_ecdf_by_context",
    "original_vs_alternative", "risk_heatmaps", "veto_chain_tails",
    "fatal_gates", "interpretation_checkpoints",
)
ANALYSIS_PLOT_NAMES = (
    "certificate-flow", "unsafe-fraction-by-stage",
    "first-conflict-stage-context", "cost-excess-ecdf",
    "first-safe-action-rank", "stage-transition-matrix",
    "original-versus-alternative", "risk-heatmaps",
    "disagreement-association", "event-relief", "veto-chain-tails",
)
_ARM_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SIDECARS = (
    ("shots", SHOT_SCHEMA),
    ("proposals", PROPOSAL_SCHEMA),
    ("counterfactuals", COUNTERFACTUAL_SCHEMA),
    ("domains", DOMAIN_SCHEMA),
)
_GROUND_TRUTH_FORBIDDEN_FIELDS = (
    "actual_observables",
    "actual_observables_hex",
    "packed_actual_observables_hex",
    "posthoc_ground_truth",
    "arm_failures",
    "correct",
    "correctness",
    "failure",
    "regression",
    "recovery",
)
_COLLECTOR_OWNED_FIELDS = frozenset(
    {
        "schema", "experiment_id", "cell_id", "worker_id",
        "worker_shot_index", "global_shot_id", "stim_seed",
        "physical_input_sha256", "detector_input_sha256", "circuit_sha256",
        "packed_detectors_hex", "packed_detector_bits",
        "packed_detectors_sha256", "packed_actual_observables_hex",
        "packed_actual_observable_bits", "packed_actual_observables_sha256",
        "arm_predictions_hex", "arm_failures",
    }
)


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    shot_start: int
    shots: int

    @property
    def shot_stop(self) -> int:
        return self.shot_start + self.shots

    def to_json(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NormalizedPolicyShot:
    arm_predictions: dict[str, bytes]
    shot: dict[str, Any]
    proposals: tuple[dict[str, Any], ...]
    counterfactuals: tuple[dict[str, Any], ...]
    domains: tuple[dict[str, Any], ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json_load(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f, object_pairs_hook=unique)
    except json.JSONDecodeError as ex:
        raise ValueError(f"invalid JSON file {path}: {ex}") from ex
    if not isinstance(value, dict):
        raise ValueError(f"JSON file {path} must contain one object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    scratch = os.environ.get("TMPDIR")
    if not scratch:
        raise RuntimeError("TMPDIR must be set for atomic artifact writes")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="promatch-policy-json-", dir=scratch)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_json_bytes(value))
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def policy_worker_schedule(mode: str) -> tuple[WorkerSpec, ...]:
    """Returns the exact deterministic worker-owned shot ranges."""

    if mode == "scientific":
        counts = [SCIENTIFIC_SHOTS_PER_WORKER] * SCIENTIFIC_WORKERS
    elif mode == "smoke":
        counts = [1] * SCIENTIFIC_WORKERS
    elif mode == "probe":
        counts = [4] * 4 + [3] * 28
    else:
        raise ValueError(f"unsupported policy-audit mode {mode!r}")
    rows: list[WorkerSpec] = []
    cursor = 0
    for worker_id, shots in enumerate(counts):
        rows.append(WorkerSpec(worker_id, cursor, shots))
        cursor += shots
    expected = {"scientific": 20_000, "smoke": 32, "probe": 100}[mode]
    if cursor != expected or len(rows) != SCIENTIFIC_WORKERS:
        raise AssertionError("internal B1 worker schedule is inconsistent")
    return tuple(rows)


def derive_policy_worker_seed(
    *, seed_root: str, experiment_id: str, cell_id: str, worker_id: int
) -> int:
    """Derives a schedule-independent uint64 Stim seed for one worker."""

    if not re.fullmatch(r"[0-9a-f]{64}", seed_root):
        raise ValueError("sampling seed root must be 64 lowercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{64}", experiment_id):
        raise ValueError("experiment_id must be a lowercase SHA-256 digest")
    if not isinstance(cell_id, str) or not cell_id or not cell_id.isascii():
        raise ValueError("cell_id must be nonempty ASCII")
    if isinstance(worker_id, bool) or not isinstance(worker_id, int):
        raise TypeError("worker_id must be an integer")
    if worker_id < 0 or worker_id >= SCIENTIFIC_WORKERS:
        raise ValueError("worker_id must be in [0, 32)")
    digest = hashlib.sha256(
        bytes.fromhex(seed_root)
        + b"promatch-policy-worker-v1\0"
        + bytes.fromhex(experiment_id)
        + b"\0"
        + cell_id.encode("ascii")
        + b"\0"
        + struct.pack("<Q", worker_id)
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _semantic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(config)))
    result.pop("experiment_id", None)
    result.pop("config_self_sha256", None)
    return result


def policy_config_self_sha256(config: Mapping[str, Any]) -> str:
    return _sha256(b"promatch-policy-config-v1\0" + canonical_json_bytes(_semantic_config(config)))


def policy_experiment_id(config: Mapping[str, Any]) -> str:
    return _sha256(b"promatch-policy-experiment-v1\0" + canonical_json_bytes(_semantic_config(config)))


def _expected_cell() -> dict[str, Any]:
    return {
        "cell_id": "b1-d7-n6-y2-r28-p0.002",
        "generator": GENERATOR,
        "d": 7,
        "r": 28,
        "p": 0.002,
        "patches": 6,
        "yokes": 2,
        "style": "cz",
        "noise": "si1000",
        "remove_x_yoke": False,
    }


def _expected_decoder() -> dict[str, Any]:
    return {
        "residual_hw_limit": 10,
        "domain_mode": "windowd",
        "boundary_policy": "disabled",
        "observable_policy": "zero-frame",
    }


def _expected_dem_options() -> dict[str, bool]:
    return {"decompose_errors": True, "approximate_disjoint_errors": True}


def _expected_arms() -> list[dict[str, Any]]:
    return [
        {"arm_id": ARM_IDS[0], "policy": "u0", "transaction": None},
        {"arm_id": ARM_IDS[1], "policy": "v3-shadow", "transaction": "tx"},
        {"arm_id": ARM_IDS[2], "policy": "cost", "transaction": "tx"},
        {"arm_id": ARM_IDS[3], "policy": "frame", "transaction": "tx"},
        {"arm_id": ARM_IDS[4], "policy": "frame", "transaction": "partial"},
    ]


def _expected_counterfactual() -> dict[str, Any]:
    return {
        "ordering": "DomainProposalStepper-stage-then-candidate-key-v1",
        "veto_budget": None,
        "stop": "first-o-frame-safe-or-true-exhaustion",
        "casebook_full_slate_only": True,
    }


def _expected_oracle() -> dict[str, Any]:
    return {
        "tolerance": {"absolute": 1e-9, "relative": 1e-6},
        "tolerance_sensitivity_relative": [1e-7, 1e-6, 1e-5],
        "decimal_precision_digits": 4096,
        "weight_sum": "math.fsum-canonical-edge-weights",
        "actual_observables_forbidden": True,
    }


def _expected_context_taxonomy() -> dict[str, Any]:
    labels = [
        "yoke", "true-boundary", "terminal", "cross-window",
        "cross-patch-or-basis", "support-cancellation", "in-domain",
    ]
    return {
        "version": "promatch-support-context-v2",
        "multi_labels": labels,
        "exclusive_display_priority": labels,
        "no_candidate_context_display_label": "none",
        "degeneracy_diagnostics": [
            "same-pair-different-path-or-frame",
            "equal-weight-logical-class",
            "disconnected-support-reconfiguration", "unclassified",
        ],
    }


def _expected_visibility_taxonomy() -> dict[str, Any]:
    return {
        "version": "promatch-policy-visibility-v1",
        "classes": [
            "L1-local-dynamic", "L1-static-boundary",
            "temporal-neighbor-dynamic", "nonlocal-yoke-dynamic",
            "oracle-only", "posthoc-ground-truth",
        ],
    }


def _expected_casebook_selection() -> dict[str, Any]:
    return {
        "algorithm": "context-stage-strata-nge20-median-distance-then-state-sha256-v1",
        "minimum_stratum_states": 20,
        "terminal_action_supplement": True,
        "uses_actual_observables": False,
        "expansion": "separate-post-selection-full-slate",
    }


def _expected_report_contract() -> dict[str, Any]:
    return {
        "human_report": {
            "path": "analysis/report.md",
            "format": "promatch-l1-policy-audit-human-report-v1",
            "encoding": "utf-8",
            "line_endings": "lf",
            "interpretation": "hypothesis-generating-not-causal-proof",
        },
        "tables": [
            "physical-cell-and-provenance", "paired-outcomes",
            "event-and-transaction-summary", "certificate-by-stage",
            "counterfactual-action-and-rank", "context-and-visibility",
            "fatal-and-interpretation-gates",
        ],
        "plots": [
            "certificate-flow", "unsafe-fraction-by-stage",
            "first-conflict-stage-context", "cost-excess-ecdf",
            "first-safe-action-rank", "stage-transition-matrix",
            "original-versus-alternative", "risk-heatmaps",
            "disagreement-association", "event-relief", "veto-chain-tails",
        ],
        "bins": [
            "window-offset-integers", "domain-hw-integers",
            "candidate-count-integers", "unsafe-count-0-1-2-3-4plus",
            "continuous-unbinned-ecdf",
        ],
    }


def _graph_hashes(prepared: PreparedCell) -> dict[str, str]:
    graph = prepared.compiled_pu.graph
    edge_rows = [
        {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": edge.target,
            "weight": edge.weight,
            "weight_hex": edge.weight.hex(),
            "observable_mask": edge.observable_mask.hex(),
            "source_role": dataclasses.asdict(edge.source_role),
            "source_role_type": type(edge.source_role).__name__,
            "target_role": (
                None if edge.target_role is None else dataclasses.asdict(edge.target_role)
            ),
            "target_role_type": (
                None if edge.target_role is None else type(edge.target_role).__name__
            ),
        }
        for edge in graph.edges
    ]
    domain_rows = []
    for key in sorted(graph.domain_graphs):
        domain = graph.domain_graphs[key]
        domain_rows.append(
            {
                "domain_type": type(key).__name__,
                "domain": dataclasses.asdict(key),
                "detector_ids": list(domain.detector_ids),
                "edge_ids": [edge.edge_id for edge in domain.edges],
            }
        )
    undecomposed = prepared.circuit.detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=True
    )
    return {
        "undecomposed_dem_sha256": _sha256(str(undecomposed).encode()),
        "matcher_edge_table_sha256": _sha256(canonical_json_bytes({"edges": edge_rows})),
        "domain_graphs_sha256": _sha256(canonical_json_bytes({"domains": domain_rows})),
    }


def inspect_policy_protocol(
    config: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Computes the cell and repository provenance without sampling shots."""

    validate_policy_protocol(config, scientific=False)
    root = _repo_root() if root is None else root.resolve()
    prepared = prepare_cell(
        config["cell"],
        decoder_config=config["decoder"],
        dem_options=config["dem_options"],
        verify_hashes=False,
    )
    return {
        "repository": repository_state(root),
        "software_versions": current_software_versions(),
        "execution_environment": current_execution_environment(),
        "cell_provenance": {**prepared.provenance, **_graph_hashes(prepared)},
    }


def freeze_policy_protocol(
    draft: Mapping[str, Any],
    *,
    protocol_relative_path: str,
    probe_root: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Freezes a draft only after its exact 100-shot probe passed analysis."""

    validate_policy_protocol(draft, scientific=False)
    root = _repo_root() if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("policy protocol freeze requires a clean worktree")
    candidate = (root / protocol_relative_path).resolve()
    if (
        root not in candidate.parents
        or candidate.parent != (root / "docs").resolve()
        or candidate.suffix != ".json"
        or "FROZEN" not in candidate.name
    ):
        raise ValueError("frozen protocol path must be docs/*FROZEN*.json")
    prepared = prepare_cell(
        draft["cell"],
        decoder_config=draft["decoder"],
        dem_options=draft["dem_options"],
        verify_hashes=False,
    )
    materialized_probe_config = json.loads(json.dumps(dict(draft)))
    materialized_probe_config["cell"] = {
        **dict(draft["cell"]),
        **prepared.provenance,
        **_graph_hashes(prepared),
    }
    materialized_probe_config.pop("experiment_id", None)
    materialized_probe_config.pop("config_self_sha256", None)
    materialized_probe_config["experiment_id"] = policy_experiment_id(
        materialized_probe_config
    )
    materialized_probe_config["config_self_sha256"] = policy_config_self_sha256(
        materialized_probe_config
    )
    probe_attestation = attest_completed_policy_probe(
        probe_root,
        expected_config=materialized_probe_config,
        implementation_commit=state["repository_commit"],
    )
    frozen = json.loads(json.dumps(dict(draft)))
    frozen["status"] = "FROZEN"
    frozen["frozen"] = True
    frozen["implementation_commit"] = state["repository_commit"]
    frozen["config_commit"] = "verified-runtime-head"
    frozen["protocol_relative_path"] = protocol_relative_path
    frozen["software_versions"] = current_software_versions()
    frozen["execution_environment"] = current_execution_environment()
    frozen["probe_attestation"] = probe_attestation
    frozen["cell"] = {
        **dict(draft["cell"]),
        **prepared.provenance,
        **_graph_hashes(prepared),
    }
    source_paths = tuple(frozen.get("source_paths", ()))
    if not source_paths:
        raise ValueError("draft source_paths must be a nonempty array")
    frozen["source_hashes"] = {
        relative: _canonical_file_hash((root / relative).resolve())
        for relative in source_paths
    }
    frozen["requirements_sha256"] = _canonical_file_hash(root / "requirements.txt")
    frozen.pop("experiment_id", None)
    frozen.pop("config_self_sha256", None)
    frozen["experiment_id"] = policy_experiment_id(frozen)
    frozen["config_self_sha256"] = policy_config_self_sha256(frozen)
    validate_policy_protocol(frozen, scientific=False)
    return frozen


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_probe_attestation(
    value: Any, *, implementation_commit: str | None = None
) -> Mapping[str, Any]:
    required = {
        "schema", "probe_experiment_id", "implementation_commit",
        "probe_config_self_sha256", "probe_experiment_sha256",
        "probe_config_sha256", "probe_manifest_sha256",
        "collection_ready_sha256", "analysis_ready_sha256",
        "analysis_manifest_sha256", "analysis_summary_sha256",
        "casebook_selection_sha256", "verified_workers", "verified_shots",
        "all_launch_gates_passed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen B1 probe attestation has invalid fields")
    if value.get("schema") != PROBE_ATTESTATION_SCHEMA:
        raise ValueError("frozen B1 probe attestation has the wrong schema")
    for name in required - {
        "schema", "implementation_commit", "verified_workers",
        "verified_shots", "all_launch_gates_passed",
    }:
        _require_sha256(value.get(name), name=f"probe_attestation.{name}")
    commit = value.get("implementation_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("probe attestation implementation_commit is invalid")
    if implementation_commit is not None and commit != implementation_commit:
        raise ValueError("probe attestation is for a different implementation commit")
    if value.get("verified_workers") != SCIENTIFIC_WORKERS or value.get(
        "verified_shots"
    ) != 100:
        raise ValueError("probe attestation must certify exactly 32 workers and 100 shots")
    if value.get("all_launch_gates_passed") is not True:
        raise ValueError("probe attestation did not pass every launch gate")
    return value


def _clean_tail_censor_attestation() -> dict[str, Any]:
    return {
        "uncapped_counterfactuals": True,
        "censored_states": 0,
        "repeated_same_state_proposal_signatures": 0,
        "worker_timeouts": 0,
        "output_truncations": 0,
    }


def _authenticated_analysis_file(
    analysis_root: Path, relative: Any, expected_hash: Any, *, group: str
) -> None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"probe analysis {group} contains an invalid path")
    candidate = analysis_root / relative
    path = candidate.resolve()
    if (
        candidate.is_symlink()
        or analysis_root.resolve() not in path.parents
        or not path.is_file()
    ):
        raise ValueError(f"probe analysis {group} artifact is missing or unsafe: {relative}")
    _require_sha256(expected_hash, name=f"analysis.{group}.{relative}")
    if _canonical_file_hash(path) != expected_hash:
        raise ValueError(f"probe analysis {group} digest mismatch: {relative}")


def _validate_probe_projection(value: Any, *, gates: Mapping[str, Any]) -> None:
    fields = {
        "parent_setup_seconds", "parent_setup_seconds_hex",
        "parallel_worker_compile_seconds", "parallel_worker_compile_seconds_hex",
        "fixed_setup_seconds", "fixed_setup_seconds_hex",
        "variable_100_shot_seconds", "variable_100_shot_seconds_hex",
        "compressed_probe_bytes", "projected_wall_seconds",
        "projected_wall_seconds_hex", "projected_artifact_bytes",
        "free_output_bytes", "wall_gate_passed", "artifact_gate_passed",
        "free_space_gate_passed", "all_launch_gates_passed",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("probe projection fields are incomplete")
    for name in (
        "parent_setup_seconds", "parallel_worker_compile_seconds",
        "fixed_setup_seconds", "variable_100_shot_seconds",
        "projected_wall_seconds",
    ):
        raw = value[name]
        if (
            not isinstance(raw, float)
            or not math.isfinite(raw)
            or raw < 0
            or value[f"{name}_hex"] != raw.hex()
        ):
            raise ValueError(f"probe projection has invalid exact float {name}")
    for name in ("compressed_probe_bytes", "projected_artifact_bytes", "free_output_bytes"):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"probe projection has invalid count {name}")
    if not math.isclose(
        value["fixed_setup_seconds"],
        value["parent_setup_seconds"] + value["parallel_worker_compile_seconds"],
        rel_tol=1e-15, abs_tol=1e-12,
    ):
        raise ValueError("probe fixed setup projection does not reconcile")
    expected_wall = value["fixed_setup_seconds"] + gates["probe_headroom_factor"] * 200 * value[
        "variable_100_shot_seconds"
    ]
    expected_bytes = math.ceil(
        gates["probe_headroom_factor"] * 200 * value["compressed_probe_bytes"]
    )
    expected_bools = {
        "wall_gate_passed": value["projected_wall_seconds"]
        <= gates["projected_wall_seconds_max"],
        "artifact_gate_passed": value["projected_artifact_bytes"]
        <= gates["projected_artifact_bytes_max"],
        "free_space_gate_passed": value["free_output_bytes"]
        >= max(gates["free_bytes_min"], 2 * value["projected_artifact_bytes"]),
    }
    if (
        value["projected_wall_seconds"] != expected_wall
        or value["projected_artifact_bytes"] != expected_bytes
        or any(value[name] is not expected for name, expected in expected_bools.items())
        or value["all_launch_gates_passed"] is not all(expected_bools.values())
    ):
        raise ValueError("probe wall/storage/free-space launch projection is inconsistent")


def _aggregate_collector_gate_attestations(
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from yoked.decoding._promatch_policy_analysis import (
        COLLECTOR_GATE_ATTESTATION_SCHEMA,
        COLLECTOR_GATE_CHECKS,
    )

    for shard in shards:
        evidence = shard.get("collector_gate_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("real_graph") is not True:
            raise ValueError("campaign fatal gates require real-graph worker evidence")
        checks = evidence.get("checks")
        expected = {
            check for gate_checks in COLLECTOR_GATE_CHECKS.values()
            for check in gate_checks
        }
        if not isinstance(checks, Mapping) or set(checks) != expected:
            raise ValueError("campaign worker fatal-gate evidence is incomplete")
        for check, record in checks.items():
            if (
                not isinstance(record, Mapping)
                or set(record) != {"status", "observations"}
                or record.get("status") != "passed"
                or isinstance(record.get("observations"), bool)
                or not isinstance(record.get("observations"), int)
                or record["observations"] <= 0
            ):
                raise ValueError(f"campaign worker fatal-gate check failed: {check}")
    return {
        str(gate): {
            "schema": COLLECTOR_GATE_ATTESTATION_SCHEMA,
            "gate": gate,
            "status": "passed",
            "scope": "frozen-protocol-required-scope",
            "checks": list(checks),
            "failures": 0,
        }
        for gate, checks in COLLECTOR_GATE_CHECKS.items()
    }


def _validate_campaign_performance_telemetry(value: Any) -> None:
    fields = {
        "schema", "parent_setup_ns", "worker_phase_ns", "parent_peak_rss_bytes",
        "parent_peak_rss_source", "new_worker_compile_ns",
        "scientifically_deterministic", "excluded_from_scientific_decisions",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema")
        != "promatch-l1-policy-audit-campaign-performance-v1"
        or value.get("parent_peak_rss_source")
        != "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib"
        or value.get("scientifically_deterministic") is not False
        or value.get("excluded_from_scientific_decisions") is not True
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), int)
            or value[name] < 0
            for name in ("parent_setup_ns", "worker_phase_ns", "parent_peak_rss_bytes")
        )
        or not isinstance(value.get("new_worker_compile_ns"), list)
    ):
        raise ValueError("campaign performance telemetry is invalid")
    seen: set[int] = set()
    for record in value["new_worker_compile_ns"]:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"worker_id", "compile_ns"}
            or isinstance(record.get("worker_id"), bool)
            or not isinstance(record.get("worker_id"), int)
            or record["worker_id"] < 0
            or record["worker_id"] >= SCIENTIFIC_WORKERS
            or record["worker_id"] in seen
            or isinstance(record.get("compile_ns"), bool)
            or not isinstance(record.get("compile_ns"), int)
            or record["compile_ns"] < 0
        ):
            raise ValueError("campaign worker compilation telemetry is invalid")
        seen.add(record["worker_id"])


def _validate_collection_manifest(
    manifest: Mapping[str, Any], *, config: Mapping[str, Any], mode: str,
    schedule: Sequence[WorkerSpec], shards: Sequence[Mapping[str, Any]],
) -> None:
    fields = {
        "schema", "experiment_id", "mode", "workers", "shots",
        "new_worker_processes_observed", "tail_censor_attestation", "shards",
        "campaign_wall_ns",
        "fatal_gate_attestations", "performance_telemetry",
    }
    if mode == "probe":
        fields.add("probe_projection")
    observed = manifest.get("new_worker_processes_observed")
    campaign_ns = manifest.get("campaign_wall_ns")
    if (
        set(manifest) != fields
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("experiment_id") != config["experiment_id"]
        or manifest.get("mode") != mode
        or manifest.get("workers") != SCIENTIFIC_WORKERS
        or manifest.get("shots") != sum(spec.shots for spec in schedule)
        or manifest.get("shards") != list(shards)
        or manifest.get("tail_censor_attestation") != _clean_tail_censor_attestation()
        or manifest.get("fatal_gate_attestations")
        != _aggregate_collector_gate_attestations(shards)
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < 0
        or observed > SCIENTIFIC_WORKERS
        or isinstance(campaign_ns, bool)
        or not isinstance(campaign_ns, int)
        or campaign_ns < 0
    ):
        raise ValueError("B1 campaign manifest does not reconcile")
    _validate_campaign_performance_telemetry(manifest["performance_telemetry"])
    if mode == "probe":
        if observed != SCIENTIFIC_WORKERS:
            raise ValueError("probe manifest did not observe exactly 32 worker processes")
        _validate_probe_projection(manifest["probe_projection"], gates=config["launch_gates"])


def attest_completed_policy_probe(
    probe_root: Path,
    *,
    expected_config: Mapping[str, Any],
    implementation_commit: str,
) -> dict[str, Any]:
    """Authenticates a completed exact-arm probe and its installed analysis."""

    root = Path(probe_root).resolve()
    required_files = {
        "experiment": root / "experiment.json",
        "config": root / "config.json",
        "manifest": root / "manifest.json",
        "collection_ready": root / "COLLECTION_READY",
        "analysis_ready": root / "ANALYSIS_READY",
        "analysis_manifest": root / "analysis" / "manifest.json",
        "analysis_summary": root / "analysis" / "summary.json",
        "analysis_report": root / "analysis" / "report.md",
        "casebook_selection": root / "casebook" / "selection.json",
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"probe root is incomplete; missing {sorted(missing)}")
    loaded = {
        name: _strict_json_load(path)
        for name, path in required_files.items()
        if name != "analysis_report"
    }
    expected = json.loads(json.dumps(dict(expected_config)))
    validate_policy_protocol(expected, scientific=False)
    if loaded["config"] != expected:
        raise ValueError("probe config is not the materialized supplied draft")
    experiment_id = str(expected["experiment_id"])
    expected_experiment = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": experiment_id,
        "mode": "probe",
        "implementation_commit": expected.get("implementation_commit"),
        "config_commit": implementation_commit,
    }
    if loaded["experiment"] != expected_experiment:
        raise ValueError("probe experiment is not from the current implementation commit")

    manifest = loaded["manifest"]
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("mode") != "probe"
        or manifest.get("workers") != SCIENTIFIC_WORKERS
        or manifest.get("shots") != 100
        or not isinstance(manifest.get("shards"), list)
        or len(manifest["shards"]) != SCIENTIFIC_WORKERS
    ):
        raise ValueError("probe manifest is not the exact 32-worker/100-shot probe")
    projection = manifest.get("probe_projection")
    _validate_probe_projection(projection, gates=expected["launch_gates"])
    if projection.get("all_launch_gates_passed") is not True:
        raise ValueError("probe did not pass every wall/storage/free-space launch gate")

    schedule = policy_worker_schedule("probe")
    shard_root = root / "shards"
    expected_dirs = {f"worker-{spec.worker_id:02d}" for spec in schedule}
    if not shard_root.is_dir() or {p.name for p in shard_root.iterdir()} != expected_dirs:
        raise ValueError("probe shard directories do not match the exact 32-worker schedule")
    verified = [
        verify_worker_shard(
            shard_root / f"worker-{spec.worker_id:02d}",
            config=expected,
            mode="probe",
            spec=spec,
        )
        for spec in schedule
    ]
    if manifest["shards"] != verified:
        raise ValueError("probe manifest does not match the authenticated worker shards")
    _validate_collection_manifest(
        manifest, config=expected, mode="probe", schedule=schedule, shards=verified
    )

    hashes = {name: _canonical_file_hash(path) for name, path in required_files.items()}
    collection_ready = loaded["collection_ready"]
    if collection_ready != {
        "schema": "promatch-l1-policy-audit-collection-ready-v1",
        "experiment_id": experiment_id,
        "mode": "probe",
        "manifest_sha256": hashes["manifest"],
        "verified_worker_shards": SCIENTIFIC_WORKERS,
        "verified_shots": 100,
    }:
        raise ValueError("probe COLLECTION_READY does not authenticate the exact manifest")

    analysis_ready = loaded["analysis_ready"]
    if (
        set(analysis_ready) != {
            "schema", "experiment_id", "analysis_manifest_sha256",
            "casebook_selection_sha256", "report_file_sha256", "plots_rendered",
            "casebook_exhaustive_expansion_required_before_complete",
        }
        or analysis_ready.get("schema")
        != "promatch-l1-policy-audit-analysis-ready-v1"
        or analysis_ready.get("experiment_id") != experiment_id
        or analysis_ready.get("analysis_manifest_sha256")
        != hashes["analysis_manifest"]
        or analysis_ready.get("casebook_selection_sha256")
        != hashes["casebook_selection"]
        or analysis_ready.get("report_file_sha256") != hashes["analysis_report"]
        or analysis_ready.get(
            "casebook_exhaustive_expansion_required_before_complete"
        ) is not True
        or analysis_ready.get("plots_rendered") is not True
    ):
        raise ValueError("probe ANALYSIS_READY does not authenticate a full analysis")

    analysis_manifest = loaded["analysis_manifest"]
    summary = loaded["analysis_summary"]
    analysis_contract = summary.get("analysis_contract")
    source_hashes = analysis_manifest.get("source_hashes")
    expected_source_hashes = {
        "experiment.json": hashes["experiment"],
        "config.json": hashes["config"],
        "manifest.json": hashes["manifest"],
        "COLLECTION_READY": hashes["collection_ready"],
    }
    for spec in schedule:
        for name, _ in _SIDECARS:
            relative = f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz"
            expected_source_hashes[relative] = _canonical_file_hash(root / relative)
        timing_relative = f"shards/worker-{spec.worker_id:02d}/timing.json"
        expected_source_hashes[timing_relative] = _canonical_file_hash(
            root / timing_relative
        )
    analysis_manifest_fields = {
        "schema", "experiment_id", "analysis_sha256", "summary_file_sha256",
        "report_file_sha256",
        "source_hashes", "table_file_hashes", "plot_data_file_hashes",
        "plot_images", "plot_images_scientifically_digested",
    }
    if (
        set(analysis_manifest) != analysis_manifest_fields
        or analysis_manifest.get("schema")
        != "promatch-l1-policy-audit-analysis-manifest-v1"
        or analysis_manifest.get("experiment_id") != experiment_id
        or analysis_manifest.get("summary_file_sha256")
        != hashes["analysis_summary"]
        or analysis_manifest.get("report_file_sha256")
        != hashes["analysis_report"]
        or source_hashes != expected_source_hashes
        or summary.get("schema") != "promatch-l1-policy-audit-analysis-v1"
        or set(summary) != {
            "schema", "experiment_id", "cell_id", "analysis_contract",
            "source_hashes", "tables", "casebook_selection", "analysis_sha256",
        }
        or summary.get("experiment_id") != experiment_id
        or summary.get("cell_id") != expected["cell"]["cell_id"]
        or summary.get("source_hashes") != expected_source_hashes
        or summary.get("casebook_selection") != loaded["casebook_selection"]
        or not isinstance(summary.get("tables"), Mapping)
        or set(summary["tables"]) != set(ANALYSIS_TABLE_NAMES)
        or not isinstance(analysis_contract, Mapping)
        or set(analysis_contract) != {
            "source", "sampling_or_decoding_reconstruction", "bootstrap_unit",
            "bootstrap_quantile", "proposal_bootstrap_replicates",
            "workload_bootstrap_replicates", "casebook_outcome_blind",
            "casebook_exhaustive_rows_excluded", "support_context_views_kept_distinct",
            "required_tail_telemetry", "complete_written_by_analyzer",
            "complete_accepted_as_analysis_substitute", "next_required_stage",
        }
        or analysis_contract.get("source") != "immutable-canonical-gzip-jsonl-only"
        or analysis_contract.get("sampling_or_decoding_reconstruction") is not False
        or analysis_contract.get("bootstrap_unit") != "complete-physical-shot"
        or analysis_contract.get("bootstrap_quantile") != "empirical-type-7"
        or analysis_contract.get("proposal_bootstrap_replicates")
        != expected["bootstrap"]["replicates"]
        or analysis_contract.get("workload_bootstrap_replicates")
        != expected["bootstrap"]["replicates"]
        or analysis_contract.get("casebook_outcome_blind") is not True
        or analysis_contract.get("casebook_exhaustive_rows_excluded") is not True
        or analysis_contract.get("support_context_views_kept_distinct") is not True
        or analysis_contract.get("required_tail_telemetry") != "complete"
        or analysis_contract.get("complete_written_by_analyzer") is not False
        or analysis_contract.get("complete_accepted_as_analysis_substitute") is not False
        or analysis_contract.get("next_required_stage")
        != "casebook-expansion-and-finalization-external"
        or analysis_manifest.get("analysis_sha256") != summary.get("analysis_sha256")
        or analysis_manifest.get("plot_images_scientifically_digested") is not False
    ):
        raise ValueError("probe analysis manifest/summary authentication failed")
    analysis_root = root / "analysis"
    expected_analysis_files = {
        "table_file_hashes": {f"tables/{name}.json" for name in ANALYSIS_TABLE_NAMES},
        "plot_data_file_hashes": {
            f"plot-data/{name}.json" for name in ANALYSIS_PLOT_NAMES
        },
    }
    for group in ("table_file_hashes", "plot_data_file_hashes"):
        records = analysis_manifest.get(group)
        if not isinstance(records, Mapping) or set(records) != expected_analysis_files[group]:
            raise ValueError(f"probe full analysis has incomplete {group}")
        for relative, expected_hash in records.items():
            _authenticated_analysis_file(
                analysis_root, relative, expected_hash, group=group
            )
            if group == "table_file_hashes":
                table_name = Path(relative).stem
                with (analysis_root / relative).open(encoding="utf-8") as stream:
                    table_value = json.load(stream)
                if table_value != summary["tables"][table_name]:
                    raise ValueError(f"probe analysis table differs from summary: {relative}")
    plot_images = analysis_manifest.get("plot_images")
    expected_images = [f"plots/{name}.png" for name in sorted(ANALYSIS_PLOT_NAMES)]
    if plot_images != expected_images:
        raise ValueError("probe full analysis did not render plot images")
    for relative in plot_images:
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("probe analysis contains an invalid plot image path")
        candidate = analysis_root / relative
        image = candidate.resolve()
        if (
            candidate.is_symlink()
            or analysis_root.resolve() not in image.parents
            or not image.is_file()
        ):
            raise ValueError(f"probe plot image is missing or unsafe: {relative}")
    summary_without_digest = dict(summary)
    summary_digest = summary_without_digest.pop("analysis_sha256", None)
    if summary_digest != _sha256(canonical_json_bytes(summary_without_digest)):
        raise ValueError("probe analysis summary self digest is invalid")
    fatal_gates = summary.get("tables", {}).get("fatal_gates")
    if (
        not isinstance(fatal_gates, list)
        or [row.get("gate") for row in fatal_gates if isinstance(row, Mapping)]
        != list(range(1, 19))
        or any(
            not isinstance(row, Mapping)
            or row.get("status") not in {"collector-attested", "passed-ledger-recomputed"}
            for row in fatal_gates
        )
    ):
        raise ValueError("probe full analysis did not certify all 18 fatal gates")

    result = {
        "schema": PROBE_ATTESTATION_SCHEMA,
        "probe_experiment_id": experiment_id,
        "implementation_commit": implementation_commit,
        "probe_config_self_sha256": str(expected["config_self_sha256"]),
        "probe_experiment_sha256": hashes["experiment"],
        "probe_config_sha256": hashes["config"],
        "probe_manifest_sha256": hashes["manifest"],
        "collection_ready_sha256": hashes["collection_ready"],
        "analysis_ready_sha256": hashes["analysis_ready"],
        "analysis_manifest_sha256": hashes["analysis_manifest"],
        "analysis_summary_sha256": hashes["analysis_summary"],
        "casebook_selection_sha256": hashes["casebook_selection"],
        "verified_workers": SCIENTIFIC_WORKERS,
        "verified_shots": 100,
        "all_launch_gates_passed": True,
    }
    _validate_probe_attestation(result, implementation_commit=implementation_commit)
    return result


def materialize_policy_draft(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adds real-graph provenance to a scratch draft without freezing it."""

    validate_policy_protocol(config, scientific=False)
    prepared = prepare_cell(
        config["cell"],
        decoder_config=config["decoder"],
        dem_options=config["dem_options"],
        verify_hashes=False,
    )
    result = json.loads(json.dumps(dict(config)))
    result["cell"] = {
        **dict(config["cell"]),
        **prepared.provenance,
        **_graph_hashes(prepared),
    }
    result.pop("experiment_id", None)
    result.pop("config_self_sha256", None)
    result["experiment_id"] = policy_experiment_id(result)
    result["config_self_sha256"] = policy_config_self_sha256(result)
    validate_policy_protocol(result, scientific=False)
    return result


def _require_exact_keys(value: Any, required: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != required:
        raise ValueError(
            f"{name} fields must be exactly {sorted(required)}; "
            f"got {sorted(value)}"
        )
    return value


def validate_policy_protocol(
    config: Mapping[str, Any],
    *,
    scientific: bool,
    protocol_path: Path | None = None,
    root: Path | None = None,
) -> str:
    """Validates B1 semantics and, when scientific, two-commit provenance."""

    canonical_json_bytes(config)
    base_fields = {
        "schema", "status", "frozen", "claim_bearing", "cell", "dem_options",
        "decoder", "sampling", "arms", "oracle", "counterfactual",
        "context_taxonomy", "visibility_taxonomy", "casebook_selection",
        "bootstrap", "report_contract", "fatal_gates", "artifact", "source_paths",
        "launch_gates",
    }
    is_frozen = config.get("status") == "FROZEN" and config.get("frozen") is True
    if is_frozen:
        if "probe_attestation" not in config:
            raise ValueError("frozen B1 protocol has no probe attestation")
        expected_fields = base_fields | {
            "implementation_commit", "config_commit", "protocol_relative_path",
            "software_versions", "execution_environment", "probe_attestation",
            "source_hashes", "requirements_sha256", "experiment_id",
            "config_self_sha256",
        }
    else:
        if config.get("status") != "DRAFT" or config.get("frozen") is not False:
            raise ValueError("policy status/frozen pair must be DRAFT/false or FROZEN/true")
        digest_fields = {"experiment_id", "config_self_sha256"}.intersection(config)
        if digest_fields not in (set(), {"experiment_id", "config_self_sha256"}):
            raise ValueError("materialized draft must carry both experiment and self digests")
        expected_fields = base_fields | digest_fields
    _require_exact_keys(config, expected_fields, name="policy protocol")
    if config.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"policy protocol schema must be {PROTOCOL_SCHEMA!r}")
    if config.get("claim_bearing") is not False:
        raise ValueError("B1 must remain exploratory and non-claim-bearing")
    cell = dict(config["cell"])
    base_cell = _expected_cell()
    provenance_fields = {
        "circuit_sha256", "dem_sha256", "layout_fingerprint", "graph_fingerprint",
        "num_detectors", "num_observables", "undecomposed_dem_sha256",
        "matcher_edge_table_sha256", "domain_graphs_sha256",
    }
    expected_cell_fields = set(base_cell)
    if provenance_fields.intersection(cell):
        expected_cell_fields |= provenance_fields
    if set(cell) != expected_cell_fields or any(cell.get(k) != v for k, v in base_cell.items()):
        raise ValueError("B1 requires the exact fixed d7/p0.002/y2 cell")
    if provenance_fields.issubset(cell):
        for name in provenance_fields - {"num_detectors", "num_observables"}:
            _require_sha256(cell[name], name=f"cell.{name}")
        for name in ("num_detectors", "num_observables"):
            if isinstance(cell[name], bool) or not isinstance(cell[name], int) or cell[name] <= 0:
                raise ValueError(f"cell.{name} must be a positive integer")
    elif is_frozen:
        raise ValueError("frozen B1 cell is missing complete graph provenance")
    if config.get("dem_options") != _expected_dem_options():
        raise ValueError("B1 DEM options differ from the fixed decomposed model")
    if config.get("decoder") != _expected_decoder():
        raise ValueError("B1 decoder must use the exact frozen V3 configuration")
    sampling = _require_exact_keys(
        config["sampling"],
        {"total_shots", "workers", "shots_per_worker", "worker_ranges", "seed_derivation", "seed_roots"},
        name="sampling",
    )
    if (
        sampling["total_shots"] != SCIENTIFIC_SHOTS
        or sampling["workers"] != SCIENTIFIC_WORKERS
        or sampling["shots_per_worker"] != SCIENTIFIC_SHOTS_PER_WORKER
        or sampling["seed_derivation"] != SEED_DERIVATION
    ):
        raise ValueError("scientific B1 sampling must be exactly 20000=32x625")
    if sampling["worker_ranges"] != [row.to_json() for row in policy_worker_schedule("scientific")]:
        raise ValueError("B1 worker_ranges must explicitly encode all 32 fixed ranges")
    if sampling["seed_roots"] != {
        "scientific": "17" * 32, "smoke": "2b" * 32, "probe": "c4" * 32,
    }:
        raise ValueError("sampling seed roots differ from the frozen disjoint roots")
    if config["arms"] != _expected_arms():
        raise ValueError("B1 arms must be the exact ordered five-arm registry")
    if config["oracle"] != _expected_oracle():
        raise ValueError("B1 oracle numerical contract differs from the frozen contract")
    OracleTolerance(**config["oracle"]["tolerance"])
    if config["counterfactual"] != _expected_counterfactual():
        raise ValueError("B1 counterfactual contract differs from the uncapped frozen contract")
    if config["context_taxonomy"] != _expected_context_taxonomy():
        raise ValueError("B1 context taxonomy differs from the frozen catalog")
    if config["visibility_taxonomy"] != _expected_visibility_taxonomy():
        raise ValueError("B1 visibility taxonomy differs from the frozen catalog")
    if config["casebook_selection"] != _expected_casebook_selection():
        raise ValueError("B1 casebook selection/expansion contract differs")
    bootstrap = config["bootstrap"]
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("replicates") != 10_000
        or bootstrap.get("unit") != "complete-physical-shot"
        or bootstrap.get("quantile") != "empirical-type-7"
        or bootstrap.get("seed_roots") != {"proposal": "6d" * 32, "workload": "a1" * 32}
        or set(bootstrap) != {"replicates", "unit", "quantile", "seed_roots"}
    ):
        raise ValueError("B1 complete-shot bootstrap contract is incomplete")
    if config["report_contract"] != _expected_report_contract():
        raise ValueError("B1 table/plot/binning contract differs from the frozen contract")
    if config["fatal_gates"] != list(range(1, 19)):
        raise ValueError("B1 must freeze fatal correctness gates 1 through 18")
    if config["launch_gates"] != {
        "projected_wall_seconds_max": 7200,
        "projected_artifact_bytes_max": 20 * 1024**3,
        "free_bytes_min": 40 * 1024**3,
        "probe_headroom_factor": 1.5,
    }:
        raise ValueError("B1 wall/storage launch gates are not frozen exactly")
    artifact = config["artifact"]
    if artifact != {
        "format": "canonical-jsonl-gzip",
        "gzip_level": GZIP_LEVEL,
        "gzip_mtime": 0,
        "gzip_filename": "",
        "temporary_root": "TMPDIR",
        "whole_worker_shard_resume": True,
    }:
        raise ValueError("B1 artifact settings differ from deterministic v1 storage")
    if config["source_paths"] != list(POLICY_SOURCE_PATHS):
        raise ValueError("B1 source_paths differ from the complete frozen implementation scope")
    experiment_id = policy_experiment_id(config)
    if config.get("experiment_id") not in (None, experiment_id):
        raise ValueError("policy experiment_id does not match semantic config")
    if config.get("config_self_sha256") not in (
        None,
        policy_config_self_sha256(config),
    ):
        raise ValueError("policy config_self_sha256 mismatch")
    if is_frozen:
        if config.get("config_commit") != "verified-runtime-head":
            raise ValueError("B1 config_commit policy must be verified-runtime-head")
        if not isinstance(config.get("protocol_relative_path"), str):
            raise ValueError("frozen B1 protocol path is missing")
        source_hashes = config.get("source_hashes")
        if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(POLICY_SOURCE_PATHS):
            raise ValueError("frozen B1 source_hashes do not cover exact source_paths")
        for relative, digest in source_hashes.items():
            _require_sha256(digest, name=f"source_hashes.{relative}")
        _require_sha256(config.get("requirements_sha256"), name="requirements_sha256")
        _validate_probe_attestation(
            config.get("probe_attestation"),
            implementation_commit=config.get("implementation_commit"),
        )
    elif "probe_attestation" in config:
        raise ValueError("draft B1 protocol may not carry a frozen probe attestation")
    if not scientific:
        return experiment_id

    if not is_frozen:
        raise ValueError("scientific B1 collection requires a frozen protocol")
    if os.environ.get("MAX_ERRORS") is not None:
        raise ValueError("MAX_ERRORS is forbidden for fixed-shot B1 collection")
    configure_single_thread_runtime()
    if any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT):
        raise ValueError("all B1 native numerical thread limits must equal one")
    if protocol_path is None:
        raise ValueError("scientific validation requires the committed protocol path")
    root = _repo_root() if root is None else root.resolve()
    state = repository_state(root)
    if not state["clean_worktree"]:
        raise ValueError("scientific B1 collection requires a clean worktree")
    if config.get("config_commit") != "verified-runtime-head":
        raise ValueError("B1 config_commit policy must be verified-runtime-head")
    implementation_commit = config.get("implementation_commit")
    if not isinstance(implementation_commit, str):
        raise ValueError("B1 frozen protocol has no implementation commit")
    _validate_post_freeze_protocol_commit(
        manifest=config,
        root=root,
        frozen_base=implementation_commit,
        current_head=state["repository_commit"],
    )
    expected_protocol = (root / str(config.get("protocol_relative_path"))).resolve()
    if protocol_path.resolve() != expected_protocol:
        raise ValueError("runtime protocol path differs from the frozen path")
    if config.get("software_versions") != current_software_versions():
        raise ValueError("B1 software versions differ from the frozen protocol")
    if config.get("execution_environment") != current_execution_environment():
        raise ValueError("B1 execution environment differs from the frozen protocol")
    source_hashes = config.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("B1 frozen protocol has no source hashes")
    for relative, expected_hash in source_hashes.items():
        if _canonical_file_hash((root / relative).resolve()) != expected_hash:
            raise ValueError(f"B1 source hash mismatch for {relative}")
    if _canonical_file_hash(root / "requirements.txt") != config.get("requirements_sha256"):
        raise ValueError("B1 requirements hash mismatch")
    return experiment_id


def canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def deterministic_gzip(data: bytes, *, level: int = GZIP_LEVEL) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", compresslevel=level, fileobj=target, mtime=0
    ) as f:
        f.write(data)
    return target.getvalue()


def _artifact_metadata(
    *, compressed: bytes, uncompressed: bytes, rows: int, schema: str,
    path: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema": schema,
        "rows": rows,
        "compressed_sha256": _sha256(compressed),
        "compressed_bytes": len(compressed),
        "uncompressed_sha256": _sha256(uncompressed),
        "uncompressed_bytes": len(uncompressed),
    }
    if path is not None:
        result["path"] = path
    return result


def _peak_rss_bytes() -> int:
    """Returns Linux ``ru_maxrss`` in bytes for authenticated telemetry."""

    # This experiment is frozen to the Linux execution environment recorded in
    # the protocol.  Linux reports ru_maxrss in KiB (unlike macOS, which reports
    # bytes), so keep the conversion and its source explicit in timing.json.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _type7_quantiles_ns(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("timing quantiles require at least one physical shot")
    result: dict[str, Any] = {}
    for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        value = float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))
        result[name] = value
        result[f"{name}_hex"] = value.hex()
    result["max"] = max(values)
    return result


def _shot_performance_telemetry(
    audited: NormalizedPolicyShot,
    *,
    worker_shot_index: int,
    global_shot_id: int,
    audit_wall_ns: int,
) -> dict[str, Any]:
    """Extracts concrete, non-decision timing/call telemetry from one audit."""

    shot = audited.shot
    reported = shot.get("timing_telemetry", {})
    if not isinstance(reported, Mapping):
        raise ValueError("policy-core timing_telemetry must be an object")
    candidate_rows = (*audited.proposals, *audited.counterfactuals)

    def integer(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"policy-core performance counter {name} is invalid")
        return value

    def optional_integer(name: str, value: Any) -> int:
        return integer(name, 0 if value is None else value)

    return {
        "worker_shot_index": worker_shot_index,
        "global_shot_id": global_shot_id,
        "audit_wall_ns": integer("audit_wall_ns", audit_wall_ns),
        "oracle_cache_hits": optional_integer(
            "oracle_cache_hits", shot.get("oracle_cache_hits")
        ),
        "oracle_cache_misses": optional_integer(
            "oracle_cache_misses", shot.get("oracle_cache_misses")
        ),
        "oracle_evaluation_call_count": optional_integer(
            "oracle_evaluation_call_count", shot.get("oracle_evaluation_call_count")
        ),
        "full_mwpm_cache_miss_count": optional_integer(
            "total_full_mwpm_cache_miss_count_all_arms",
            shot.get("total_full_mwpm_cache_miss_count_all_arms"),
        ),
        "matched_active_pair_backend_call_count": optional_integer(
            "matched_active_pair_backend_call_count",
            shot.get("matched_active_pair_backend_call_count"),
        ),
        "matched_active_pair_backend_wall_ns": optional_integer(
            "matched_active_pair_backend_wall_ns",
            shot.get("matched_active_pair_backend_wall_ns"),
        ),
        "counterfactual_wall_ns": optional_integer(
            "counterfactual_wall_ns", reported.get("counterfactual_wall_ns")
        ),
        "support_classification_wall_ns": optional_integer(
            "support_classification_wall_ns",
            reported.get("support_classification_wall_ns"),
        ),
        "candidate_enumeration_wall_ns": sum(
            optional_integer(
                "candidate_enumeration_wall_ns",
                row.get("candidate_enumeration_wall_ns"),
            )
            for row in candidate_rows
        ),
        "stage3_enumeration_wall_ns": optional_integer(
            "stage3_specific_wall_ns", reported.get("stage3_specific_wall_ns")
        ),
    }


def _separate_nondeterministic_timing(value: Any) -> tuple[Any, Any | None]:
    """Splits wall-clock fields out of a scientific ledger value recursively."""

    if isinstance(value, Mapping):
        scientific: dict[str, Any] = {}
        timing: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.endswith("_wall_ns") or name == "timing_telemetry":
                timing[name] = item
                continue
            cleaned, nested_timing = _separate_nondeterministic_timing(item)
            scientific[name] = cleaned
            if nested_timing is not None:
                timing[name] = nested_timing
        return scientific, timing or None
    if isinstance(value, (tuple, list)):
        scientific_items: list[Any] = []
        timing_items: list[Any | None] = []
        found = False
        for item in value:
            cleaned, nested_timing = _separate_nondeterministic_timing(item)
            scientific_items.append(cleaned)
            timing_items.append(nested_timing)
            found |= nested_timing is not None
        return scientific_items, timing_items if found else None
    return value, None


def _real_graph_numerical_preflight(
    graph: Any, syndrome: np.ndarray, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Runs the frozen numerical repeatability subset on one real worker state."""

    tolerance = OracleTolerance(**config["oracle"]["tolerance"])
    first_oracle = FullGraphOracle(graph, tolerance=tolerance)
    first = first_oracle.decode_state(syndrome, use_cache=False)
    repeated = first_oracle.decode_state(syndrome, use_cache=False)
    if first != repeated:
        raise AssertionError("uncached full-graph oracle decode is not repeatable")

    cached_oracle = FullGraphOracle(graph, tolerance=tolerance)
    uncached = cached_oracle.decode_state(syndrome, use_cache=False)
    cached_seed = cached_oracle.decode_state(syndrome, use_cache=True)
    cached_repeat = cached_oracle.decode_state(syndrome, use_cache=True)
    if uncached != cached_seed or cached_seed != cached_repeat:
        raise AssertionError("cached and uncached full-graph oracle results differ")
    if cached_oracle.cache_stats.hits != 1:
        raise AssertionError("oracle repeatability preflight did not exercise a cache hit")

    weights = [graph.edges[edge_id].weight for edge_id in first.support_edge_ids]
    fsum_weight = math.fsum(weights)
    if fsum_weight != first.support_weight:
        raise AssertionError("oracle support weight is not the canonical math.fsum")
    precision = config["oracle"]["decimal_precision_digits"]
    with localcontext() as context:
        context.prec = precision
        decimal_weight = sum((Decimal.from_float(value) for value in weights), Decimal(0))
        decimal_fsum = Decimal.from_float(fsum_weight)
        rounding_bound = Decimal.from_float(
            math.ulp(fsum_weight) if fsum_weight != 0 else math.ulp(0.0)
        )
        if abs(decimal_weight - decimal_fsum) > rounding_bound:
            raise AssertionError("math.fsum differs from the 4096-digit Decimal reference")

    grid_observations = 0
    for relative in config["oracle"]["tolerance_sensitivity_relative"]:
        grid_tolerance = OracleTolerance(
            absolute=tolerance.absolute, relative=relative
        )
        grid_solution = FullGraphOracle(
            graph, tolerance=grid_tolerance
        ).decode_state(syndrome, use_cache=False)
        if (
            grid_solution.prediction != first.prediction
            or grid_solution.support_edge_ids != first.support_edge_ids
            or grid_solution.support_weight != first.support_weight
            or grid_solution.backend_weight != first.backend_weight
        ):
            raise AssertionError("oracle support changed across the tolerance grid")
        excess = grid_solution.support_weight - grid_solution.backend_weight
        tau = grid_tolerance.tau_k(
            base_weight=grid_solution.backend_weight,
            composite_weight=grid_solution.support_weight,
        )
        classification = classify_cost_excess(cost_excess=excess, tau_k=tau)
        expected = (
            "positive-cost-excess" if excess > tau
            else "numeric-accounting-anomaly" if excess < -tau
            else "numerically-cost-compatible"
        )
        if classification.value != expected:
            raise AssertionError("oracle tolerance-grid classification is inconsistent")
        grid_observations += 1

    return {
        "schema": "promatch-l1-policy-audit-numerical-preflight-v1",
        "real_graph": True,
        "syndrome_sha256": _sha256(np.asarray(syndrome, dtype=np.uint8).tobytes()),
        "decimal_precision_digits": precision,
        "tolerance_grid_observations": grid_observations,
        "backend_support_fsum_passed": True,
        "decimal_reference_passed": True,
        "uncached_repeatability_passed": True,
        "cached_uncached_repeatability_passed": True,
        "tolerance_grid_passed": True,
    }


def _worker_gate_evidence(
    *,
    spec: WorkerSpec,
    numerical_preflight: Mapping[str, Any],
    shots: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from yoked.decoding._promatch_policy_analysis import COLLECTOR_GATE_CHECKS

    oracle_calls = sum(int(row.get("oracle_evaluation_call_count", 0)) for row in shots)
    observations = {
        "scalar-batch-u0": spec.shots,
        "shadow-frozen-v3-equivalence": spec.shots,
        "backend-support-fsum": 1,
        "decimal-4096": 1,
        "uncached-repeatability": 1,
        "tolerance-grid": int(numerical_preflight["tolerance_grid_observations"]),
        "actual-observable-invariance": spec.shots,
        "veto-state-frame-prefix-invariance": max(spec.shots, len(counterfactuals)),
        "matching-pair-and-support-reconciliation": max(spec.shots, oracle_calls),
        "cached-uncached-oracle-repeatability": 1,
        "execution-and-source-provenance": 1,
    }
    expected = {check for checks in COLLECTOR_GATE_CHECKS.values() for check in checks}
    if set(observations) != expected or any(value <= 0 for value in observations.values()):
        raise AssertionError("worker collector gate evidence is incomplete")
    return {
        "schema": "promatch-l1-policy-audit-worker-gate-evidence-v1",
        "real_graph": numerical_preflight.get("real_graph") is True,
        "checks": {
            check: {"status": "passed", "observations": observations[check]}
            for check in sorted(expected)
        },
        "numerical_preflight": dict(numerical_preflight),
    }


def _normalize_prediction(value: Any, *, width: int) -> bytes:
    if isinstance(value, str):
        try:
            result = bytes.fromhex(value)
        except ValueError as ex:
            raise ValueError("arm prediction is not hexadecimal") from ex
    elif isinstance(value, bytes):
        result = value
    else:
        array = np.asarray(value, dtype=np.uint8)
        if array.ndim != 1:
            raise ValueError("arm prediction must be one-dimensional")
        result = bytes(np.packbits(array, bitorder="little"))
    if len(result) != width:
        raise ValueError(f"arm prediction has {len(result)} bytes; expected {width}")
    return result


def _forbid_ground_truth(value: Any, *, path: str = "audit") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in _GROUND_TRUTH_FORBIDDEN_FIELDS or lowered.startswith(
                ("actual_observables_", "packed_actual_observables_")
            ):
                raise ValueError(f"ground-truth-like field {path}.{key} entered policy core output")
            _forbid_ground_truth(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _forbid_ground_truth(item, path=f"{path}[{index}]")


def _require_float_hex_companions(value: Any, *, path: str = "audit") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, float):
                companion = value.get(f"{key}_hex")
                if companion != item.hex():
                    raise ValueError(
                        f"float field {path}.{key} lacks its exact *_hex companion"
                    )
            _require_float_hex_companions(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_float_hex_companions(item, path=f"{path}[{index}]")


def _validate_support_difference_ledger(
    row: Mapping[str, Any], *, path: str, graph: Any | None = None
) -> None:
    """Fail-closed validation of the v2 support certificate before persistence."""

    def canonical_edge_ids(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(type(edge_id) is int and edge_id >= 0 for edge_id in value)
            and value == sorted(set(value))
        )

    if row.get("support_difference_representation_version") != "promatch-support-difference-v2":
        raise ValueError(f"{path} lacks the exact v2 support-difference representation")
    components = row.get("support_difference_components")
    if not isinstance(components, list):
        raise ValueError(f"{path} support_difference_components must be an array")
    exact_keys = {
        "certificate_kind", "canonical_edge_ids", "support_cancellation_edge_ids",
        "component_detector_ids", "candidate_support_witness_edge_ids",
        "candidate_boundary_witness_detector_ids",
        "labels", "candidate_relevant", "candidate_relevance_reasons",
    }
    labels_vocab = set(_expected_context_taxonomy()["multi_labels"])
    reason_vocab = {
        "candidate-support-edge", "candidate-boundary-detector",
        "candidate-residual-support-cancellation",
    }
    detector_boundary_ids = row.get("detector_boundary_ids")
    if not canonical_edge_ids(detector_boundary_ids):
        raise ValueError(f"{path} detector_boundary_ids is not a canonical detector set")
    real: set[int] = set()
    cancellations: set[int] = set()
    candidate_labels: set[str] = set()
    disconnected = False
    real_components: list[Mapping[str, Any]] = []
    cancellation_components: list[Mapping[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping) or set(component) != exact_keys:
            raise ValueError(f"{path} support component fields are not exact")
        edges = component["canonical_edge_ids"]
        cancel = component["support_cancellation_edge_ids"]
        detector_ids = component["component_detector_ids"]
        support_witness = component["candidate_support_witness_edge_ids"]
        boundary_witness = component["candidate_boundary_witness_detector_ids"]
        labels = component["labels"]
        reasons = component["candidate_relevance_reasons"]
        relevant = component["candidate_relevant"]
        if (
            not canonical_edge_ids(edges)
            or not canonical_edge_ids(cancel)
            or not canonical_edge_ids(detector_ids)
            or not canonical_edge_ids(support_witness)
            or not canonical_edge_ids(boundary_witness)
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or labels != sorted(set(labels))
            or set(labels) - labels_vocab
            or ("in-domain" in labels and len(labels) != 1)
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) for reason in reasons)
            or reasons != sorted(set(reasons))
            or set(reasons) - reason_vocab
            or not isinstance(relevant, bool)
        ):
            raise ValueError(f"{path} support component is not canonical")
        if component["certificate_kind"] == "real-x-component":
            expected_support_witness = sorted(
                set(edges).intersection(row.get("P_candidate_support_edge_ids", []))
            )
            expected_boundary_witness = sorted(
                set(detector_ids).intersection(row.get("detector_boundary_ids", []))
            )
            expected_reasons = sorted(
                (["candidate-support-edge"] if expected_support_witness else [])
                + (["candidate-boundary-detector"] if expected_boundary_witness else [])
            )
            if graph is not None:
                expected_detector_ids: set[int] = set()
                for edge_id in edges:
                    if edge_id >= len(graph.edges):
                        raise ValueError(f"{path} component references an unknown graph edge")
                    edge = graph.edges[edge_id]
                    expected_detector_ids.add(int(edge.source))
                    if edge.target is not None:
                        expected_detector_ids.add(int(edge.target))
                if detector_ids != sorted(expected_detector_ids):
                    raise ValueError(f"{path} component detector witness disagrees with graph")
            if (
                not edges
                or cancel
                or support_witness != expected_support_witness
                or boundary_witness != expected_boundary_witness
                or reasons != expected_reasons
                or "support-cancellation" in labels
                or bool(reasons) != relevant
                or real.intersection(edges)
            ):
                raise ValueError(f"{path} real X component is malformed or overlapping")
            real.update(edges)
            disconnected |= not relevant
            real_components.append(component)
        elif component["certificate_kind"] == "support-cancellation":
            if (
                edges
                or not cancel
                or detector_ids or support_witness or boundary_witness
                or cancellations
                or "support-cancellation" not in labels
                or not relevant
                or reasons != ["candidate-residual-support-cancellation"]
            ):
                raise ValueError(f"{path} cancellation certificate is malformed")
            cancellations.update(cancel)
            cancellation_components.append(component)
        else:
            raise ValueError(f"{path} has an unknown support certificate kind")
        if relevant:
            candidate_labels.update(labels)
    if components != sorted(
        real_components, key=lambda component: component["canonical_edge_ids"]
    ) + cancellation_components:
        raise ValueError(f"{path} support certificates are not in canonical order")
    supports: dict[str, list[int]] = {}
    for field in (
        "B_base_support_edge_ids", "P_candidate_support_edge_ids",
        "R_residual_support_edge_ids", "Q_forced_parity_support_edge_ids",
        "X_support_difference_edge_ids", "P_intersection_R_edge_ids",
    ):
        value = row.get(field)
        if not canonical_edge_ids(value):
            raise ValueError(f"{path} {field} is not a canonical support")
        supports[field] = value
    b, p, r = (set(supports[name]) for name in (
        "B_base_support_edge_ids", "P_candidate_support_edge_ids",
        "R_residual_support_edge_ids",
    ))
    for alias, canonical in (
        ("base_support_edge_ids", supports["B_base_support_edge_ids"]),
        ("candidate_support_edge_ids", supports["P_candidate_support_edge_ids"]),
        ("residual_support_edge_ids", supports["R_residual_support_edge_ids"]),
    ):
        if row.get(alias) != canonical:
            raise ValueError(f"{path} {alias} disagrees with its named B/P/R support")
    if supports["Q_forced_parity_support_edge_ids"] != sorted(p ^ r):
        raise ValueError(f"{path} Q support does not reconcile")
    if supports["X_support_difference_edge_ids"] != sorted(b ^ p ^ r):
        raise ValueError(f"{path} X support does not reconcile")
    if supports["P_intersection_R_edge_ids"] != sorted(p & r):
        raise ValueError(f"{path} P intersection R does not reconcile")
    if sorted(real) != supports["X_support_difference_edge_ids"]:
        raise ValueError(f"{path} real components do not partition X")
    if sorted(cancellations) != supports["P_intersection_R_edge_ids"]:
        raise ValueError(f"{path} cancellation certificates do not reconcile")
    if row.get("support_cancellation_edge_ids") != sorted(cancellations):
        raise ValueError(f"{path} top-level cancellation support does not reconcile")
    if row.get("support_difference_component_labels") != sorted(candidate_labels):
        raise ValueError(f"{path} candidate-context labels do not reconcile")
    if row.get("disconnected_support_reconfiguration") is not disconnected:
        raise ValueError(f"{path} disconnected-support flag does not reconcile")
    expected_exclusive = next(
        (
            label
            for label in _expected_context_taxonomy()["exclusive_display_priority"]
            if label in candidate_labels
        ),
        None,
    )
    if row.get("exclusive_support_component_context") != expected_exclusive:
        raise ValueError(f"{path} exclusive candidate context does not reconcile")
    diagnostics = row.get("degeneracy_diagnostics")
    if (
        not isinstance(diagnostics, list)
        or any(not isinstance(label, str) for label in diagnostics)
        or diagnostics != sorted(set(diagnostics))
        or set(diagnostics)
        - set(_expected_context_taxonomy()["degeneracy_diagnostics"])
        or ("disconnected-support-reconfiguration" in diagnostics) != disconnected
    ):
        raise ValueError(f"{path} degeneracy diagnostics do not reconcile")
    if not supports["X_support_difference_edge_ids"]:
        if row.get("frame_compatible") is not True:
            raise ValueError(f"{path} has an algebraically impossible X-empty frame conflict")
        if (
            row.get("oracle_policy_accepts") is False
            and not supports["P_intersection_R_edge_ids"]
        ):
            raise ValueError(f"{path} unsafe row has no X or cancellation certificate")
    for flag in (
        "supports_square_free", "B_base_support_square_free",
        "P_candidate_support_square_free", "R_residual_support_square_free",
        "Q_forced_parity_support_square_free", "X_support_difference_square_free",
    ):
        if row.get(flag) is not True:
            raise ValueError(f"{path} {flag} must be true")


def _normalized_context_union(*groups: Any, path: str) -> list[str]:
    """Returns the frozen context union with ``in-domain`` exclusivity."""

    vocabulary = set(_expected_context_taxonomy()["multi_labels"])
    labels: set[str] = set()
    for index, group in enumerate(groups):
        if (
            not isinstance(group, list)
            or any(not isinstance(label, str) for label in group)
            or group != sorted(set(group))
            or set(group) - vocabulary
            or ("in-domain" in group and len(group) != 1)
        ):
            raise ValueError(f"{path} context label group {index} is not canonical")
        labels.update(group)
    if labels - {"in-domain"}:
        labels.discard("in-domain")
    return sorted(labels)


def _validate_context_union_ledger(row: Mapping[str, Any], *, path: str) -> None:
    """Reconciles the three independently persisted omitted-context views."""

    matched = row.get("matched_partner_labels")
    support_path = row.get("support_path_labels")
    omitted = row.get("omitted_context_labels")
    expected = _normalized_context_union(matched, support_path, path=path)
    if _normalized_context_union(omitted, path=path) != expected or omitted != expected:
        raise ValueError(
            f"{path} omitted_context_labels disagrees with the normalized "
            "matched/support-path union"
        )


def _audit_policy_shot(
    graph: Any,
    syndrome: np.ndarray,
    *,
    tolerance: OracleTolerance,
    audit_fn: Callable[..., Any] | None = None,
) -> NormalizedPolicyShot:
    """Ground-truth-free adapter to the independently implemented B1 core."""

    if audit_fn is None:
        try:
            from yoked.decoding._promatch_policy_audit import audit_policy_shot
        except ImportError as ex:
            raise RuntimeError(
                "B1 policy core is unavailable; expected "
                "yoked.decoding._promatch_policy_audit:audit_policy_shot"
            ) from ex
        audit_fn = audit_policy_shot
    raw = audit_fn(graph, syndrome.copy(), tolerance=tolerance)
    if hasattr(raw, "to_json"):
        raw = raw.to_json()
    required = {"arm_predictions", "shot", "proposals", "counterfactuals", "domains"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(f"audit_policy_shot output fields must be exactly {sorted(required)}")
    _forbid_ground_truth(raw)
    _require_float_hex_companions(raw)
    width = (graph.num_observables + 7) // 8
    predictions_raw = raw["arm_predictions"]
    if not isinstance(predictions_raw, Mapping) or set(predictions_raw) != set(ARM_IDS):
        raise ValueError("policy core must return exactly one prediction for every B1 arm")
    predictions = {
        arm_id: _normalize_prediction(predictions_raw[arm_id], width=width)
        for arm_id in ARM_IDS
    }
    collections: dict[str, tuple[dict[str, Any], ...]] = {}
    for name in ("proposals", "counterfactuals", "domains"):
        values = raw[name]
        if not isinstance(values, (tuple, list)) or not all(isinstance(v, Mapping) for v in values):
            raise ValueError(f"policy core {name} must be an array of objects")
        collections[name] = tuple(dict(v) for v in values)
    for name in ("proposals", "counterfactuals"):
        for index, row in enumerate(collections[name]):
            _validate_support_difference_ledger(
                row, path=f"audit.{name}[{index}]", graph=graph
            )
            _validate_context_union_ledger(
                row, path=f"audit.{name}[{index}]"
            )
    if not isinstance(raw["shot"], Mapping):
        raise ValueError("policy core shot must be an object")
    return NormalizedPolicyShot(
        arm_predictions=predictions,
        shot=dict(raw["shot"]),
        proposals=collections["proposals"],
        counterfactuals=collections["counterfactuals"],
        domains=collections["domains"],
    )


def _row_identity(
    *, config: Mapping[str, Any], spec: WorkerSpec, worker_shot_index: int, stim_seed: int, physical_input_sha256: str
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "cell_id": config["cell"]["cell_id"],
        "worker_id": spec.worker_id,
        "worker_shot_index": worker_shot_index,
        "global_shot_id": spec.shot_start + worker_shot_index,
        "stim_seed": stim_seed,
        "physical_input_sha256": physical_input_sha256,
        "detector_input_sha256": physical_input_sha256,
        "circuit_sha256": config["cell"]["circuit_sha256"],
    }


def _merge_core_row(
    *, schema: str, identity: Mapping[str, Any], core: Mapping[str, Any]
) -> dict[str, Any]:
    overlap = _COLLECTOR_OWNED_FIELDS.intersection(core)
    if overlap:
        raise ValueError(
            "policy core attempted to override collector-owned fields: "
            f"{sorted(overlap)}"
        )
    return {"schema": schema, **identity, **core}


def collect_policy_worker_shard(
    prepared: PreparedCell,
    *,
    config: Mapping[str, Any],
    mode: str,
    spec: WorkerSpec,
    audit_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Samples and audits one immutable worker range entirely in memory."""

    configure_single_thread_runtime()
    seed = derive_policy_worker_seed(
        seed_root=config["sampling"]["seed_roots"][mode],
        experiment_id=config["experiment_id"],
        cell_id=config["cell"]["cell_id"],
        worker_id=spec.worker_id,
    )
    sampler = prepared.circuit.compile_detector_sampler(seed=seed)
    sample_start = time.perf_counter_ns()
    dets, obs = sampler.sample(shots=spec.shots, separate_observables=True, bit_packed=True)
    sample_ns = time.perf_counter_ns() - sample_start
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    dets.setflags(write=False)
    obs.setflags(write=False)
    det_digest = digest_array(dets)
    obs_digest = digest_array(obs)
    graph = prepared.compiled_pu.graph
    u0_start = time.perf_counter_ns()
    u0_batch = np.asarray(
        graph.matcher.decode_batch(
            dets, bit_packed_shots=True, bit_packed_predictions=True
        ),
        dtype=np.uint8,
    )
    u0_batch_decode_ns = time.perf_counter_ns() - u0_start
    unpacked = np.unpackbits(
        dets, axis=1, bitorder="little", count=graph.num_detectors
    ).astype(np.uint8, copy=False)
    if audit_fn is None:
        numerical_preflight = _real_graph_numerical_preflight(
            graph, unpacked[0], config=config
        )
    else:
        # Unit tests use a deliberately minimal matcher test double.  This
        # evidence is accepted for isolated shard-verifier tests but is
        # explicitly rejected by campaign aggregation.
        numerical_preflight = {
            "schema": "promatch-l1-policy-audit-numerical-preflight-v1",
            "real_graph": False,
            "syndrome_sha256": _sha256(unpacked[0].tobytes()),
            "decimal_precision_digits": config["oracle"]["decimal_precision_digits"],
            "tolerance_grid_observations": len(
                config["oracle"]["tolerance_sensitivity_relative"]
            ),
            "backend_support_fsum_passed": True,
            "decimal_reference_passed": True,
            "uncached_repeatability_passed": True,
            "cached_uncached_repeatability_passed": True,
            "tolerance_grid_passed": True,
        }
    tolerance = OracleTolerance(**config["oracle"]["tolerance"])
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _SIDECARS}
    per_shot_timing: list[dict[str, Any]] = []
    decode_start = time.perf_counter_ns()
    for offset in range(spec.shots):
        audit_start = time.perf_counter_ns()
        audited = _audit_policy_shot(
            graph, unpacked[offset], tolerance=tolerance, audit_fn=audit_fn
        )
        audit_wall_ns = time.perf_counter_ns() - audit_start
        if audited.arm_predictions[ARM_IDS[0]] != bytes(u0_batch[offset]):
            raise AssertionError("scalar policy-core U0 differs from bit-packed batch U0")
        detector_bytes = bytes(dets[offset])
        observable_bytes = bytes(obs[offset])
        # Proposal/casebook identity is detector-only.  Actual observables are
        # posthoc ground truth and must not influence any audit identity or
        # deterministic selection tie-break.
        physical_sha = _sha256(
            b"promatch-policy-detector-input-v1\0" + detector_bytes
        )
        identity = _row_identity(
            config=config,
            spec=spec,
            worker_shot_index=offset,
            stim_seed=seed,
            physical_input_sha256=physical_sha,
        )
        per_shot_timing.append(
            _shot_performance_telemetry(
                audited,
                worker_shot_index=offset,
                global_shot_id=identity["global_shot_id"],
                audit_wall_ns=audit_wall_ns,
            )
        )
        prediction_hex = {
            arm_id: audited.arm_predictions[arm_id].hex() for arm_id in ARM_IDS
        }
        failures = {
            arm_id: audited.arm_predictions[arm_id] != observable_bytes
            for arm_id in ARM_IDS
        }
        shot_overlap = _COLLECTOR_OWNED_FIELDS.intersection(audited.shot)
        if shot_overlap:
            raise ValueError(
                "policy core attempted to override collector-owned fields: "
                f"{sorted(shot_overlap)}"
            )
        rows["shots"].append(
            {
                "schema": SHOT_SCHEMA,
                **identity,
                "packed_detectors_hex": detector_bytes.hex(),
                "packed_detector_bits": graph.num_detectors,
                "packed_detectors_sha256": _sha256(detector_bytes),
                "packed_actual_observables_hex": observable_bytes.hex(),
                "packed_actual_observable_bits": graph.num_observables,
                "packed_actual_observables_sha256": _sha256(observable_bytes),
                "arm_predictions_hex": prediction_hex,
                "arm_failures": failures,
                **audited.shot,
            }
        )
        for name, schema in _SIDECARS[1:]:
            for item in getattr(audited, name):
                rows[name].append(
                    _merge_core_row(schema=schema, identity=identity, core=item)
                )
    decode_ns = time.perf_counter_ns() - decode_start
    if digest_array(dets) != det_digest or digest_array(obs) != obs_digest:
        raise AssertionError("a policy arm mutated the shared sampled corpus")

    payloads: dict[str, bytes] = {}
    artifact_rows: dict[str, Any] = {}
    serialization_by_artifact: dict[str, Any] = {}
    core_timing_by_ledger: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _ in _SIDECARS
    }
    serialization_start = time.perf_counter_ns()
    for name, schema in _SIDECARS:
        scientific_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows[name]):
            scientific_row, extracted_timing = _separate_nondeterministic_timing(row)
            if not isinstance(scientific_row, dict):
                raise AssertionError("ledger timing separation changed the row type")
            scientific_rows.append(scientific_row)
            if extracted_timing is not None:
                core_timing_by_ledger[name].append(
                    {
                        "row_index": row_index,
                        "worker_shot_index": row["worker_shot_index"],
                        "global_shot_id": row["global_shot_id"],
                        "timing": extracted_timing,
                    }
                )
        canonical_start = time.perf_counter_ns()
        raw = canonical_jsonl(scientific_rows)
        canonical_ns = time.perf_counter_ns() - canonical_start
        compression_start = time.perf_counter_ns()
        compressed = deterministic_gzip(raw)
        compression_ns = time.perf_counter_ns() - compression_start
        payloads[f"{name}.jsonl.gz"] = compressed
        artifact_rows[f"{name}.jsonl.gz"] = _artifact_metadata(
            compressed=compressed,
            uncompressed=raw,
            rows=len(rows[name]),
            schema=schema,
            path=f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz",
        )
        serialization_by_artifact[f"{name}.jsonl.gz"] = {
            "canonical_jsonl_ns": canonical_ns,
            "gzip_ns": compression_ns,
            "rows": len(scientific_rows),
            "compressed_bytes": len(compressed),
            "uncompressed_bytes": len(raw),
            "compressed_bytes_per_row": {
                "numerator": len(compressed), "denominator": len(scientific_rows)
            },
            "uncompressed_bytes_per_row": {
                "numerator": len(raw), "denominator": len(scientific_rows)
            },
            "compressed_bytes_per_physical_shot": {
                "numerator": len(compressed), "denominator": spec.shots
            },
            "uncompressed_bytes_per_physical_shot": {
                "numerator": len(raw), "denominator": spec.shots
            },
        }
    serialization_ns = time.perf_counter_ns() - serialization_start
    audit_values = [row["audit_wall_ns"] for row in per_shot_timing]
    timing = {
        "schema": "promatch-l1-policy-audit-timing-v2",
        "sampling_ns": sample_ns,
        "u0_batch_decode_ns": u0_batch_decode_ns,
        "shot_audit_loop_ns": decode_ns,
        "ledger_serialization_ns": serialization_ns,
        "shot_audit_wall_ns_quantiles": _type7_quantiles_ns(audit_values),
        "per_shot": per_shot_timing,
        "serialization_by_artifact": serialization_by_artifact,
        "core_timing_by_ledger": core_timing_by_ledger,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib",
        "per_arm_decode_wall_ns_available": False,
        "per_arm_decode_wall_ns_gap": (
            "audit_policy_shot currently returns all five arms atomically; "
            "per-shot audit and backend/enumeration timings are recorded, but "
            "wall time cannot yet be attributed to individual arms"
        ),
        "worker_id": spec.worker_id,
        "scientifically_deterministic": False,
        "excluded_from_bit_exact_ledger_contract": True,
    }
    payloads["timing.json"] = canonical_json_bytes(timing) + b"\n"
    shard = {
        "schema": SHARD_SCHEMA,
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "cell_id": config["cell"]["cell_id"],
        "worker": spec.to_json(),
        "stim_seed": seed,
        "detectors": {
            "sha256": det_digest.sha256,
            "shape": list(det_digest.shape),
            "dtype": det_digest.dtype,
        },
        "observables": {
            "sha256": obs_digest.sha256,
            "shape": list(obs_digest.shape),
            "dtype": obs_digest.dtype,
        },
        "artifacts": artifact_rows,
        "timing_sha256": _sha256(payloads["timing.json"]),
        "timing_path": f"shards/worker-{spec.worker_id:02d}/timing.json",
        "native_thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "tail_censor_attestation": _worker_tail_censor_attestation(
            rows["counterfactuals"]
        ),
        "collector_gate_evidence": _worker_gate_evidence(
            spec=spec,
            numerical_preflight=numerical_preflight,
            shots=rows["shots"],
            counterfactuals=rows["counterfactuals"],
        ),
        "nondeterministic_telemetry_paths": [
            f"shards/worker-{spec.worker_id:02d}/timing.json"
        ],
        "bit_exact_regeneration_paths": [
            f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz"
            for name, _ in _SIDECARS
        ],
    }
    payloads["shard.json"] = canonical_json_bytes(shard) + b"\n"
    return shard, payloads


def _worker_tail_censor_attestation(
    counterfactual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    censored_states: set[str] = set()
    seen: set[tuple[str, bytes]] = set()
    repeated = 0
    for row in counterfactual_rows:
        state = row.get(
            "original_state_sha256", row.get("complete_pre_state_fingerprint")
        )
        if not isinstance(state, str) or not state:
            raise ValueError("counterfactual row lacks complete original-state identity")
        if row.get("censored") is True:
            censored_states.add(state)
        if row.get("veto_budget") is not None:
            raise ValueError("B1 counterfactual row contains a forbidden veto budget")
        signature = row.get("proposal_signature", row.get("proposal_sha256"))
        if signature is None:
            raise ValueError("counterfactual row lacks a proposal signature")
        # Proposal signatures are structured JSON values (normally arrays),
        # while canonical_json_bytes intentionally accepts only a top-level
        # object.  Wrap the value in a version-stable object before comparing
        # exact bytes within one original state.
        key = state, canonical_json_bytes({"proposal_signature": signature})
        if key in seen:
            repeated += 1
        seen.add(key)
    return {
        "uncapped_counterfactuals": True,
        "censored_states": len(censored_states),
        "repeated_same_state_proposal_signatures": repeated,
        "output_truncations": 0,
    }


def _shard_dir(out: Path, worker_id: int) -> Path:
    return out / "shards" / f"worker-{worker_id:02d}"


def install_worker_shard(
    out: Path, *, shard: Mapping[str, Any], payloads: Mapping[str, bytes],
    config: Mapping[str, Any], mode: str, spec: WorkerSpec,
) -> Path:
    """Validates then atomically installs one completed shard from TMPDIR."""

    scratch = os.environ.get("TMPDIR")
    if not scratch:
        raise RuntimeError("TMPDIR must be set for shard installation")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if os.stat(scratch).st_dev != os.stat(out).st_dev:
        raise RuntimeError("TMPDIR and output must share a filesystem for atomic shard install")
    worker_id = int(shard["worker"]["worker_id"])
    final = _shard_dir(out, worker_id)
    if final.exists():
        raise FileExistsError(f"worker shard already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f"promatch-policy-worker-{worker_id:02d}-", dir=scratch))
    try:
        if set(payloads) != {"shots.jsonl.gz", "proposals.jsonl.gz", "counterfactuals.jsonl.gz", "domains.jsonl.gz", "timing.json", "shard.json"}:
            raise ValueError("worker payload set is incomplete")
        for name, data in payloads.items():
            path = temporary / name
            with path.open("xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        if int(shard["worker"]["worker_id"]) != spec.worker_id:
            raise ValueError("worker result ID differs from its scheduled shard")
        verified = verify_worker_shard(
            temporary, config=config, mode=mode, spec=spec
        )
        if verified != dict(shard):
            raise ValueError("worker shard manifest differs from returned worker metadata")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        # Authenticate the installed namespace, not only the staging directory.
        # This keeps worker-side installation parallel while closing the gap
        # between the bytes verified before rename and the path trusted by the
        # parent campaign manifest.
        installed = verify_worker_shard(
            final, config=config, mode=mode, spec=spec
        )
        if installed != verified:
            raise ValueError("installed worker shard differs from its staged verification")
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_worker_shard(
    path: Path,
    *,
    config: Mapping[str, Any],
    mode: str,
    spec: WorkerSpec,
) -> dict[str, Any]:
    """Authenticates a whole immutable worker shard before resume."""

    expected_names = {"shots.jsonl.gz", "proposals.jsonl.gz", "counterfactuals.jsonl.gz", "domains.jsonl.gz", "timing.json", "shard.json"}
    if not path.is_dir() or {p.name for p in path.iterdir()} != expected_names:
        raise ValueError(f"partial or unexpected worker shard {path}")
    if path.is_symlink() or any(entry.is_symlink() or not entry.is_file() for entry in path.iterdir()):
        raise ValueError(f"worker shard contains a symlink or non-file entry: {path}")
    shard = _strict_json_load(path / "shard.json")
    shard_fields = {
        "schema", "experiment_id", "mode", "cell_id", "worker", "stim_seed",
        "detectors", "observables", "artifacts", "timing_sha256", "timing_path",
        "native_thread_environment", "tail_censor_attestation",
        "nondeterministic_telemetry_paths", "bit_exact_regeneration_paths",
        "collector_gate_evidence",
    }
    if (
        set(shard) != shard_fields
        or shard.get("schema") != SHARD_SCHEMA
        or shard.get("experiment_id") != config["experiment_id"]
        or shard.get("mode") != mode
        or shard.get("cell_id") != config["cell"]["cell_id"]
        or shard.get("worker") != spec.to_json()
    ):
        raise ValueError("worker shard identity mismatch")
    expected_seed = derive_policy_worker_seed(
        seed_root=config["sampling"]["seed_roots"][mode],
        experiment_id=config["experiment_id"],
        cell_id=config["cell"]["cell_id"],
        worker_id=spec.worker_id,
    )
    if shard.get("stim_seed") != expected_seed:
        raise ValueError("worker shard seed mismatch")
    if shard.get("native_thread_environment") != {
        name: "1" for name in THREAD_ENVIRONMENT
    }:
        raise ValueError("worker shard native thread environment is not single-threaded")
    if shard.get("nondeterministic_telemetry_paths") != [
        f"shards/worker-{spec.worker_id:02d}/timing.json"
    ]:
        raise ValueError("worker shard does not isolate nondeterministic timing telemetry")
    if shard.get("bit_exact_regeneration_paths") != [
        f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz"
        for name, _ in _SIDECARS
    ]:
        raise ValueError("worker shard bit-exact ledger registry is incomplete")
    artifacts = shard.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        f"{name}.jsonl.gz" for name, _ in _SIDECARS
    }:
        raise ValueError("worker shard artifact registry is incomplete")
    parsed: dict[str, list[dict[str, Any]]] = {}
    for name, schema in _SIDECARS:
        compressed = (path / f"{name}.jsonl.gz").read_bytes()
        try:
            raw = gzip.decompress(compressed)
        except (gzip.BadGzipFile, EOFError) as ex:
            raise ValueError(f"invalid compressed sidecar {name}") from ex
        if deterministic_gzip(raw) != compressed:
            raise ValueError(f"sidecar gzip bytes are not deterministic for {name}")
        metadata = artifacts[f"{name}.jsonl.gz"]
        expected = _artifact_metadata(
            compressed=compressed,
            uncompressed=raw,
            rows=len(raw.splitlines()),
            schema=schema,
            path=f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz",
        )
        if metadata != expected:
            raise ValueError(f"sidecar digest/count mismatch for {name}")
        lines = raw.splitlines()
        parsed[name] = []
        for line in lines:
            def remember(items: list[tuple[str, Any]]) -> dict[str, Any]:
                seen: set[str] = set()
                for key, _ in items:
                    if key in seen:
                        raise ValueError(f"duplicate JSONL key {key!r} in {name}")
                    seen.add(key)
                return dict(items)

            row = json.loads(line, object_pairs_hook=remember)
            if canonical_json_bytes(row) != line:
                raise ValueError(f"sidecar row is not canonical JSON for {name}")
            _, leaked_timing = _separate_nondeterministic_timing(row)
            if leaked_timing is not None:
                raise ValueError(f"scientific ledger contains nondeterministic timing in {name}")
            if (
                row.get("schema") != schema
                or row.get("experiment_id") != config["experiment_id"]
                or row.get("cell_id") != config["cell"]["cell_id"]
                or row.get("worker_id") != spec.worker_id
                or row.get("stim_seed") != expected_seed
                or row.get("circuit_sha256") != config["cell"]["circuit_sha256"]
            ):
                raise ValueError(f"sidecar identity mismatch for {name}")
            worker_index = row.get("worker_shot_index")
            global_id = row.get("global_shot_id")
            if (
                isinstance(worker_index, bool)
                or not isinstance(worker_index, int)
                or worker_index < 0
                or worker_index >= spec.shots
                or global_id != spec.shot_start + worker_index
            ):
                raise ValueError(f"sidecar shot schedule mismatch for {name}")
            physical = row.get("physical_input_sha256")
            if (
                not isinstance(physical, str)
                or re.fullmatch(r"[0-9a-f]{64}", physical) is None
                or row.get("detector_input_sha256") != physical
            ):
                raise ValueError(f"sidecar detector identity mismatch for {name}")
            if name != "shots":
                if row.get("arm_id") not in ARM_IDS:
                    raise ValueError(f"sidecar contains an unknown arm for {name}")
                _forbid_ground_truth(row, path=name)
            if name in {"proposals", "counterfactuals"} and row.get(
                "graph_fingerprint"
            ) != config["cell"]["graph_fingerprint"]:
                raise ValueError(f"{name} row graph fingerprint mismatch")
            if name in {"proposals", "counterfactuals"}:
                _validate_support_difference_ledger(row, path=name)
                _validate_context_union_ledger(row, path=name)
            _require_float_hex_companions(row, path=name)
            parsed[name].append(row)
        if name == "shots":
            ids = [row["global_shot_id"] for row in parsed[name]]
            if ids != list(range(spec.shot_start, spec.shot_stop)):
                raise ValueError("shot sidecar does not exactly cover worker range")

    detector_width = (int(config["cell"]["num_detectors"]) + 7) // 8
    observable_width = (int(config["cell"]["num_observables"]) + 7) // 8
    detector_rows: list[bytes] = []
    observable_rows: list[bytes] = []
    shot_identity: dict[int, tuple[str, int]] = {}
    for row in parsed["shots"]:
        try:
            detector_bytes = bytes.fromhex(row["packed_detectors_hex"])
            observable_bytes = bytes.fromhex(row["packed_actual_observables_hex"])
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError("shot row has invalid packed detector/observable data") from ex
        if (
            len(detector_bytes) != detector_width
            or len(observable_bytes) != observable_width
            or row.get("packed_detector_bits") != config["cell"]["num_detectors"]
            or row.get("packed_actual_observable_bits") != config["cell"]["num_observables"]
            or row.get("packed_detectors_sha256") != _sha256(detector_bytes)
            or row.get("packed_actual_observables_sha256") != _sha256(observable_bytes)
            or row.get("physical_input_sha256")
            != _sha256(b"promatch-policy-detector-input-v1\0" + detector_bytes)
        ):
            raise ValueError("shot packed input digest/width reconciliation failed")
        predictions = row.get("arm_predictions_hex")
        failures = row.get("arm_failures")
        if not isinstance(predictions, Mapping) or set(predictions) != set(ARM_IDS):
            raise ValueError("shot prediction arm set differs from frozen arms")
        if not isinstance(failures, Mapping) or set(failures) != set(ARM_IDS):
            raise ValueError("shot failure arm set differs from frozen arms")
        for arm_id in ARM_IDS:
            try:
                prediction = bytes.fromhex(predictions[arm_id])
            except (TypeError, ValueError) as ex:
                raise ValueError("shot contains an invalid arm prediction") from ex
            if (
                len(prediction) != observable_width
                or not isinstance(failures[arm_id], bool)
                or failures[arm_id] != (prediction != observable_bytes)
            ):
                raise ValueError("shot arm prediction/failure reconciliation failed")
        detector_rows.append(detector_bytes)
        observable_rows.append(observable_bytes)
        shot_identity[row["worker_shot_index"]] = (
            row["physical_input_sha256"], row["global_shot_id"]
        )
    for name in ("proposals", "counterfactuals", "domains"):
        for row in parsed[name]:
            if shot_identity.get(row["worker_shot_index"]) != (
                row["physical_input_sha256"], row["global_shot_id"]
            ):
                raise ValueError(f"{name} row does not reference its retained shot")
    detector_array = np.asarray(
        [list(value) for value in detector_rows], dtype=np.uint8
    ).reshape(spec.shots, detector_width)
    observable_array = np.asarray(
        [list(value) for value in observable_rows], dtype=np.uint8
    ).reshape(spec.shots, observable_width)
    for name, array in (("detectors", detector_array), ("observables", observable_array)):
        digest = digest_array(array)
        if shard.get(name) != {
            "sha256": digest.sha256, "shape": list(digest.shape), "dtype": digest.dtype,
        }:
            raise ValueError(f"worker {name} array digest mismatch")
    if shard.get("tail_censor_attestation") != _worker_tail_censor_attestation(
        parsed["counterfactuals"]
    ):
        raise ValueError("worker tail/censor attestation does not match its ledger")
    gate_evidence = shard.get("collector_gate_evidence")
    numerical = (
        gate_evidence.get("numerical_preflight")
        if isinstance(gate_evidence, Mapping) else None
    )
    numerical_fields = {
        "schema", "real_graph", "syndrome_sha256", "decimal_precision_digits",
        "tolerance_grid_observations", "backend_support_fsum_passed",
        "decimal_reference_passed", "uncached_repeatability_passed",
        "cached_uncached_repeatability_passed", "tolerance_grid_passed",
    }
    first_syndrome = np.unpackbits(
        detector_array[0], bitorder="little", count=config["cell"]["num_detectors"]
    ).astype(np.uint8, copy=False)
    if (
        not isinstance(gate_evidence, Mapping)
        or set(gate_evidence) != {
            "schema", "real_graph", "checks", "numerical_preflight"
        }
        or gate_evidence.get("schema")
        != "promatch-l1-policy-audit-worker-gate-evidence-v1"
        or not isinstance(gate_evidence.get("real_graph"), bool)
        or not isinstance(numerical, Mapping)
        or set(numerical) != numerical_fields
        or numerical.get("schema")
        != "promatch-l1-policy-audit-numerical-preflight-v1"
        or numerical.get("real_graph") is not gate_evidence.get("real_graph")
        or numerical.get("syndrome_sha256") != _sha256(first_syndrome.tobytes())
        or numerical.get("decimal_precision_digits")
        != config["oracle"]["decimal_precision_digits"]
        or numerical.get("tolerance_grid_observations")
        != len(config["oracle"]["tolerance_sensitivity_relative"])
        or any(
            numerical.get(name) is not True
            for name in (
                "backend_support_fsum_passed", "decimal_reference_passed",
                "uncached_repeatability_passed",
                "cached_uncached_repeatability_passed", "tolerance_grid_passed",
            )
        )
        or gate_evidence
        != _worker_gate_evidence(
            spec=spec,
            numerical_preflight=numerical,
            shots=parsed["shots"],
            counterfactuals=parsed["counterfactuals"],
        )
    ):
        raise ValueError("worker collector fatal-gate evidence is invalid")

    timing_bytes = (path / "timing.json").read_bytes()
    if _sha256(timing_bytes) != shard.get("timing_sha256"):
        raise ValueError("worker timing digest mismatch")
    timing = _strict_json_load(path / "timing.json")
    timing_fields = {
        "schema", "sampling_ns", "u0_batch_decode_ns", "shot_audit_loop_ns",
        "ledger_serialization_ns", "shot_audit_wall_ns_quantiles", "per_shot",
        "serialization_by_artifact", "core_timing_by_ledger", "peak_rss_bytes", "peak_rss_source",
        "per_arm_decode_wall_ns_available", "per_arm_decode_wall_ns_gap",
        "worker_id", "scientifically_deterministic",
        "excluded_from_bit_exact_ledger_contract",
    }
    nonnegative_integer_fields = (
        "sampling_ns", "u0_batch_decode_ns", "shot_audit_loop_ns",
        "ledger_serialization_ns", "peak_rss_bytes",
    )
    per_shot_fields = {
        "worker_shot_index", "global_shot_id", "audit_wall_ns",
        "oracle_cache_hits", "oracle_cache_misses",
        "oracle_evaluation_call_count", "full_mwpm_cache_miss_count",
        "matched_active_pair_backend_call_count",
        "matched_active_pair_backend_wall_ns", "counterfactual_wall_ns",
        "support_classification_wall_ns", "candidate_enumeration_wall_ns",
        "stage3_enumeration_wall_ns",
    }
    timing_rows = timing.get("per_shot")
    if (
        set(timing) != timing_fields
        or timing.get("schema") != "promatch-l1-policy-audit-timing-v2"
        or timing.get("worker_id") != spec.worker_id
        or timing.get("scientifically_deterministic") is not False
        or timing.get("excluded_from_bit_exact_ledger_contract") is not True
        or timing.get("per_arm_decode_wall_ns_available") is not False
        or not isinstance(timing.get("per_arm_decode_wall_ns_gap"), str)
        or not timing["per_arm_decode_wall_ns_gap"]
        or timing.get("peak_rss_source")
        != "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib"
        or any(
            isinstance(timing.get(name), bool)
            or not isinstance(timing.get(name), int)
            or timing[name] < 0
            for name in nonnegative_integer_fields
        )
        or not isinstance(timing_rows, list)
        or len(timing_rows) != spec.shots
        or canonical_json_bytes(timing) + b"\n" != timing_bytes
        or shard.get("timing_path")
        != f"shards/worker-{spec.worker_id:02d}/timing.json"
    ):
        raise ValueError("worker timing telemetry is invalid")
    timing_values: list[int] = []
    for index, row in enumerate(timing_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != per_shot_fields
            or row.get("worker_shot_index") != index
            or row.get("global_shot_id") != spec.shot_start + index
            or any(
                isinstance(row.get(name), bool)
                or not isinstance(row.get(name), int)
                or row[name] < 0
                for name in per_shot_fields - {"worker_shot_index", "global_shot_id"}
            )
        ):
            raise ValueError("worker per-shot timing telemetry is invalid")
        timing_values.append(row["audit_wall_ns"])
    if (
        timing.get("shot_audit_wall_ns_quantiles") != _type7_quantiles_ns(timing_values)
        or timing["shot_audit_loop_ns"] < sum(timing_values)
    ):
        raise ValueError("worker shot timing distribution does not reconcile")
    core_timing = timing.get("core_timing_by_ledger")
    if not isinstance(core_timing, Mapping) or set(core_timing) != {
        name for name, _ in _SIDECARS
    }:
        raise ValueError("worker core timing ledger registry is incomplete")
    for name, records in core_timing.items():
        if not isinstance(records, list):
            raise ValueError("worker core timing records must be arrays")
        seen_indices: set[int] = set()
        for record in records:
            if (
                not isinstance(record, Mapping)
                or set(record) != {
                    "row_index", "worker_shot_index", "global_shot_id", "timing"
                }
                or isinstance(record.get("row_index"), bool)
                or not isinstance(record.get("row_index"), int)
                or record["row_index"] < 0
                or record["row_index"] >= len(parsed[name])
                or record["row_index"] in seen_indices
                or not isinstance(record.get("timing"), Mapping)
                or not record["timing"]
            ):
                raise ValueError("worker core timing record is invalid")
            source = parsed[name][record["row_index"]]
            if (
                record["worker_shot_index"] != source["worker_shot_index"]
                or record["global_shot_id"] != source["global_shot_id"]
            ):
                raise ValueError("worker core timing record identity mismatch")
            seen_indices.add(record["row_index"])
    serialization = timing.get("serialization_by_artifact")
    if not isinstance(serialization, Mapping) or set(serialization) != set(artifacts):
        raise ValueError("worker serialization telemetry is incomplete")
    serialization_component_ns = 0
    for filename, row in serialization.items():
        metadata = artifacts[filename]
        fields = {
            "canonical_jsonl_ns", "gzip_ns", "rows", "compressed_bytes",
            "uncompressed_bytes", "compressed_bytes_per_row",
            "uncompressed_bytes_per_row", "compressed_bytes_per_physical_shot",
            "uncompressed_bytes_per_physical_shot",
        }
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("worker serialization artifact telemetry is invalid")
        for name in ("canonical_jsonl_ns", "gzip_ns", "rows", "compressed_bytes", "uncompressed_bytes"):
            if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
                raise ValueError("worker serialization metric is invalid")
        expected_ratios = {
            "compressed_bytes_per_row": {
                "numerator": metadata["compressed_bytes"], "denominator": metadata["rows"]
            },
            "uncompressed_bytes_per_row": {
                "numerator": metadata["uncompressed_bytes"], "denominator": metadata["rows"]
            },
            "compressed_bytes_per_physical_shot": {
                "numerator": metadata["compressed_bytes"], "denominator": spec.shots
            },
            "uncompressed_bytes_per_physical_shot": {
                "numerator": metadata["uncompressed_bytes"], "denominator": spec.shots
            },
        }
        if (
            row["rows"] != metadata["rows"]
            or row["compressed_bytes"] != metadata["compressed_bytes"]
            or row["uncompressed_bytes"] != metadata["uncompressed_bytes"]
            or any(row[name] != expected for name, expected in expected_ratios.items())
        ):
            raise ValueError("worker serialization byte telemetry does not reconcile")
        serialization_component_ns += row["canonical_jsonl_ns"] + row["gzip_ns"]
    if timing["ledger_serialization_ns"] < serialization_component_ns:
        raise ValueError("worker serialization timing does not reconcile")
    if canonical_json_bytes(shard) + b"\n" != (path / "shard.json").read_bytes():
        raise ValueError("worker shard manifest is not canonical")
    return shard


_WORKER_PREPARED: PreparedCell | None = None


def _worker_task(
    task: dict[str, Any],
) -> tuple[int, dict[str, Any], int, int]:
    configure_single_thread_runtime()
    global _WORKER_PREPARED
    compile_ns = 0
    if _WORKER_PREPARED is None:
        compile_start_ns = time.perf_counter_ns()
        _WORKER_PREPARED = prepare_cell(
            task["config"]["cell"],
            decoder_config=task["config"]["decoder"],
            dem_options=task["config"]["dem_options"],
            verify_hashes=task["scientific"],
        )
        compile_ns = time.perf_counter_ns() - compile_start_ns
    spec = WorkerSpec(**task["spec"])
    shard, payloads = collect_policy_worker_shard(
        _WORKER_PREPARED, config=task["config"], mode=task["mode"], spec=spec
    )
    # Each worker owns a disjoint final shard directory.  Authenticate and
    # atomically install that shard here so compression transfer plus a second,
    # serial parent-side verification cannot dominate the 32-way campaign.
    install_worker_shard(
        Path(task["out"]),
        shard=shard,
        payloads=payloads,
        config=task["config"],
        mode=task["mode"],
        spec=spec,
    )
    return spec.worker_id, shard, os.getpid(), compile_ns


def _reject_immutable_output(out: Path) -> None:
    text = out.resolve().as_posix()
    if "promatch_l1_round1" in text:
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
            "experiment.json", "config.json", "manifest.json",
            "COLLECTION_READY", "shards",
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

    if processes != SCIENTIFIC_WORKERS:
        raise ValueError("B1 smoke, probe, and scientific collection require exactly 32 processes")
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
        expected_directories = {
            f"worker-{spec.worker_id:02d}" for spec in schedule
        }
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
            completed[spec.worker_id] = verify_worker_shard(
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
        raise ValueError("COLLECTION_READY exists but one or more worker shards are missing")
    worker_pids: set[int] = set()
    worker_compile_ns: list[dict[str, int]] = []
    worker_phase_start_ns = time.perf_counter_ns()
    if tasks:
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=configure_single_thread_runtime,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            future_to_task = {executor.submit(_worker_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                worker_id, shard, pid, compile_ns = future.result()
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
        raise RuntimeError("B1 tail/censor gate failed; COLLECTION_READY was not written")
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
        projected_wall_seconds = fixed_setup_ns / 1e9 + 1.5 * 200 * (variable_ns / 1e9)
        projected_artifact_bytes = math.ceil(1.5 * 200 * compressed_bytes)
        free_bytes = shutil.disk_usage(out).free
        gate = config["launch_gates"]
        manifest["probe_projection"] = {
            "parent_setup_seconds": setup_ns / 1e9,
            "parent_setup_seconds_hex": (setup_ns / 1e9).hex(),
            "parallel_worker_compile_seconds": parallel_worker_compile_ns / 1e9,
            "parallel_worker_compile_seconds_hex": (parallel_worker_compile_ns / 1e9).hex(),
            "fixed_setup_seconds": fixed_setup_ns / 1e9,
            "fixed_setup_seconds_hex": (fixed_setup_ns / 1e9).hex(),
            "variable_100_shot_seconds": variable_ns / 1e9,
            "variable_100_shot_seconds_hex": (variable_ns / 1e9).hex(),
            "compressed_probe_bytes": compressed_bytes,
            "projected_wall_seconds": projected_wall_seconds,
            "projected_wall_seconds_hex": projected_wall_seconds.hex(),
            "projected_artifact_bytes": projected_artifact_bytes,
            "free_output_bytes": free_bytes,
            "wall_gate_passed": projected_wall_seconds <= gate["projected_wall_seconds_max"],
            "artifact_gate_passed": projected_artifact_bytes <= gate["projected_artifact_bytes_max"],
            "free_space_gate_passed": free_bytes >= max(
                gate["free_bytes_min"], 2 * projected_artifact_bytes
            ),
        }
        manifest["probe_projection"]["all_launch_gates_passed"] = all(
            manifest["probe_projection"][name]
            for name in ("wall_gate_passed", "artifact_gate_passed", "free_space_gate_passed")
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


def default_policy_audit_draft() -> dict[str, Any]:
    """Returns the non-frozen B1 draft used by the checked-in JSON file."""

    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "DRAFT",
        "frozen": False,
        "claim_bearing": False,
        "cell": _expected_cell(),
        "dem_options": _expected_dem_options(),
        "decoder": _expected_decoder(),
        "sampling": {
            "total_shots": SCIENTIFIC_SHOTS,
            "workers": SCIENTIFIC_WORKERS,
            "shots_per_worker": SCIENTIFIC_SHOTS_PER_WORKER,
            "worker_ranges": [
                row.to_json() for row in policy_worker_schedule("scientific")
            ],
            "seed_derivation": SEED_DERIVATION,
            "seed_roots": {
                "scientific": "17" * 32,
                "smoke": "2b" * 32,
                "probe": "c4" * 32,
            },
        },
        "arms": _expected_arms(),
        "oracle": _expected_oracle(),
        "counterfactual": _expected_counterfactual(),
        "artifact": {
            "format": "canonical-jsonl-gzip",
            "gzip_level": GZIP_LEVEL,
            "gzip_mtime": 0,
            "gzip_filename": "",
            "temporary_root": "TMPDIR",
            "whole_worker_shard_resume": True,
        },
        "context_taxonomy": _expected_context_taxonomy(),
        "visibility_taxonomy": _expected_visibility_taxonomy(),
        "casebook_selection": _expected_casebook_selection(),
        "bootstrap": {
            "replicates": 10_000,
            "unit": "complete-physical-shot",
            "quantile": "empirical-type-7",
            "seed_roots": {"proposal": "6d" * 32, "workload": "a1" * 32},
        },
        "report_contract": _expected_report_contract(),
        "fatal_gates": list(range(1, 19)),
        "launch_gates": {
            "projected_wall_seconds_max": 7200,
            "projected_artifact_bytes_max": 20 * 1024**3,
            "free_bytes_min": 40 * 1024**3,
            "probe_headroom_factor": 1.5,
        },
        "source_paths": list(POLICY_SOURCE_PATHS),
    }


__all__ = [
    "ARM_IDS",
    "NormalizedPolicyShot",
    "PROTOCOL_SCHEMA",
    "SCIENTIFIC_SHOTS",
    "SCIENTIFIC_SHOTS_PER_WORKER",
    "SCIENTIFIC_WORKERS",
    "WorkerSpec",
    "canonical_jsonl",
    "collect_policy_worker_shard",
    "attest_completed_policy_probe",
    "default_policy_audit_draft",
    "derive_policy_worker_seed",
    "deterministic_gzip",
    "freeze_policy_protocol",
    "inspect_policy_protocol",
    "install_worker_shard",
    "materialize_policy_draft",
    "policy_config_self_sha256",
    "policy_experiment_id",
    "policy_worker_schedule",
    "run_policy_collection",
    "validate_policy_protocol",
    "verify_worker_shard",
]
