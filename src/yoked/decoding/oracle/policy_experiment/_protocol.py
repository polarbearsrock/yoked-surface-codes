"""Frozen B1 protocol semantics plus draft/freeze/scientific validation.

This slice owns the exact expected protocol content (cell, decoder, arms,
oracle, taxonomies, contracts), draft materialization, the freeze pipeline,
and :func:`validate_policy_protocol` with its two-commit scientific checks.
It inherits the package isolation contract (see ``__init__``): it verifies
provenance and semantics but never samples shots or touches ground truth.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

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
from yoked.decoding._promatch_stats import canonical_json_bytes
from yoked.decoding.oracle.full_graph import OracleTolerance

from yoked.decoding.oracle.policy_experiment._identity import (
    ARM_IDS,
    GZIP_LEVEL,
    LEGACY_POLICY_SOURCE_PATHS,
    POLICY_SOURCE_PATHS,
    PROBE_ATTESTATION_SCHEMA,
    PROTOCOL_SCHEMA,
    SCIENTIFIC_SHOTS,
    SCIENTIFIC_SHOTS_PER_WORKER,
    SCIENTIFIC_WORKERS,
    SEED_DERIVATION,
    THREAD_ENVIRONMENT,
    _repo_root,
    _require_exact_keys,
    _require_sha256,
    _sha256,
    policy_config_self_sha256,
    policy_experiment_id,
    policy_worker_schedule,
)


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
        "yoke",
        "true-boundary",
        "terminal",
        "cross-window",
        "cross-patch-or-basis",
        "support-cancellation",
        "in-domain",
    ]
    return {
        "version": "promatch-support-context-v2",
        "multi_labels": labels,
        "exclusive_display_priority": labels,
        "no_candidate_context_display_label": "none",
        "degeneracy_diagnostics": [
            "same-pair-different-path-or-frame",
            "equal-weight-logical-class",
            "disconnected-support-reconfiguration",
            "unclassified",
        ],
    }


def _expected_visibility_taxonomy() -> dict[str, Any]:
    return {
        "version": "promatch-policy-visibility-v1",
        "classes": [
            "L1-local-dynamic",
            "L1-static-boundary",
            "temporal-neighbor-dynamic",
            "nonlocal-yoke-dynamic",
            "oracle-only",
            "posthoc-ground-truth",
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
            "physical-cell-and-provenance",
            "paired-outcomes",
            "event-and-transaction-summary",
            "certificate-by-stage",
            "counterfactual-action-and-rank",
            "context-and-visibility",
            "fatal-and-interpretation-gates",
        ],
        "plots": [
            "certificate-flow",
            "unsafe-fraction-by-stage",
            "first-conflict-stage-context",
            "cost-excess-ecdf",
            "first-safe-action-rank",
            "stage-transition-matrix",
            "original-versus-alternative",
            "risk-heatmaps",
            "disagreement-association",
            "event-relief",
            "veto-chain-tails",
        ],
        "bins": [
            "window-offset-integers",
            "domain-hw-integers",
            "candidate-count-integers",
            "unsafe-count-0-1-2-3-4plus",
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
                None
                if edge.target_role is None
                else dataclasses.asdict(edge.target_role)
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
        "matcher_edge_table_sha256": _sha256(
            canonical_json_bytes({"edges": edge_rows})
        ),
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

    # Imported at call time: the attestation slice depends on this module for
    # protocol validation, so the reverse edge stays out of import time.
    from yoked.decoding.oracle.policy_experiment._attestation import (
        attest_completed_policy_probe,
    )

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


def _validate_probe_attestation(
    value: Any, *, implementation_commit: str | None = None
) -> Mapping[str, Any]:
    required = {
        "schema",
        "probe_experiment_id",
        "implementation_commit",
        "probe_config_self_sha256",
        "probe_experiment_sha256",
        "probe_config_sha256",
        "probe_manifest_sha256",
        "collection_ready_sha256",
        "analysis_ready_sha256",
        "analysis_manifest_sha256",
        "analysis_summary_sha256",
        "casebook_selection_sha256",
        "verified_workers",
        "verified_shots",
        "all_launch_gates_passed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen B1 probe attestation has invalid fields")
    if value.get("schema") != PROBE_ATTESTATION_SCHEMA:
        raise ValueError("frozen B1 probe attestation has the wrong schema")
    for name in required - {
        "schema",
        "implementation_commit",
        "verified_workers",
        "verified_shots",
        "all_launch_gates_passed",
    }:
        _require_sha256(value.get(name), name=f"probe_attestation.{name}")
    commit = value.get("implementation_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("probe attestation implementation_commit is invalid")
    if implementation_commit is not None and commit != implementation_commit:
        raise ValueError("probe attestation is for a different implementation commit")
    if (
        value.get("verified_workers") != SCIENTIFIC_WORKERS
        or value.get("verified_shots") != 100
    ):
        raise ValueError(
            "probe attestation must certify exactly 32 workers and 100 shots"
        )
    if value.get("all_launch_gates_passed") is not True:
        raise ValueError("probe attestation did not pass every launch gate")
    return value


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
        "schema",
        "status",
        "frozen",
        "claim_bearing",
        "cell",
        "dem_options",
        "decoder",
        "sampling",
        "arms",
        "oracle",
        "counterfactual",
        "context_taxonomy",
        "visibility_taxonomy",
        "casebook_selection",
        "bootstrap",
        "report_contract",
        "fatal_gates",
        "artifact",
        "source_paths",
        "launch_gates",
    }
    is_frozen = config.get("status") == "FROZEN" and config.get("frozen") is True
    if is_frozen:
        if "probe_attestation" not in config:
            raise ValueError("frozen B1 protocol has no probe attestation")
        expected_fields = base_fields | {
            "implementation_commit",
            "config_commit",
            "protocol_relative_path",
            "software_versions",
            "execution_environment",
            "probe_attestation",
            "source_hashes",
            "requirements_sha256",
            "experiment_id",
            "config_self_sha256",
        }
    else:
        if config.get("status") != "DRAFT" or config.get("frozen") is not False:
            raise ValueError(
                "policy status/frozen pair must be DRAFT/false or FROZEN/true"
            )
        digest_fields = {"experiment_id", "config_self_sha256"}.intersection(config)
        if digest_fields not in (set(), {"experiment_id", "config_self_sha256"}):
            raise ValueError(
                "materialized draft must carry both experiment and self digests"
            )
        expected_fields = base_fields | digest_fields
    _require_exact_keys(config, expected_fields, name="policy protocol")
    if config.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"policy protocol schema must be {PROTOCOL_SCHEMA!r}")
    if config.get("claim_bearing") is not False:
        raise ValueError("B1 must remain exploratory and non-claim-bearing")
    cell = dict(config["cell"])
    base_cell = _expected_cell()
    provenance_fields = {
        "circuit_sha256",
        "dem_sha256",
        "layout_fingerprint",
        "graph_fingerprint",
        "num_detectors",
        "num_observables",
        "undecomposed_dem_sha256",
        "matcher_edge_table_sha256",
        "domain_graphs_sha256",
    }
    expected_cell_fields = set(base_cell)
    if provenance_fields.intersection(cell):
        expected_cell_fields |= provenance_fields
    if set(cell) != expected_cell_fields or any(
        cell.get(k) != v for k, v in base_cell.items()
    ):
        raise ValueError("B1 requires the exact fixed d7/p0.002/y2 cell")
    if provenance_fields.issubset(cell):
        for name in provenance_fields - {"num_detectors", "num_observables"}:
            _require_sha256(cell[name], name=f"cell.{name}")
        for name in ("num_detectors", "num_observables"):
            if (
                isinstance(cell[name], bool)
                or not isinstance(cell[name], int)
                or cell[name] <= 0
            ):
                raise ValueError(f"cell.{name} must be a positive integer")
    elif is_frozen:
        raise ValueError("frozen B1 cell is missing complete graph provenance")
    if config.get("dem_options") != _expected_dem_options():
        raise ValueError("B1 DEM options differ from the fixed decomposed model")
    if config.get("decoder") != _expected_decoder():
        raise ValueError("B1 decoder must use the exact frozen V3 configuration")
    sampling = _require_exact_keys(
        config["sampling"],
        {
            "total_shots",
            "workers",
            "shots_per_worker",
            "worker_ranges",
            "seed_derivation",
            "seed_roots",
        },
        name="sampling",
    )
    if (
        sampling["total_shots"] != SCIENTIFIC_SHOTS
        or sampling["workers"] != SCIENTIFIC_WORKERS
        or sampling["shots_per_worker"] != SCIENTIFIC_SHOTS_PER_WORKER
        or sampling["seed_derivation"] != SEED_DERIVATION
    ):
        raise ValueError("scientific B1 sampling must be exactly 20000=32x625")
    if sampling["worker_ranges"] != [
        row.to_json() for row in policy_worker_schedule("scientific")
    ]:
        raise ValueError("B1 worker_ranges must explicitly encode all 32 fixed ranges")
    if sampling["seed_roots"] != {
        "scientific": "17" * 32,
        "smoke": "2b" * 32,
        "probe": "c4" * 32,
    }:
        raise ValueError("sampling seed roots differ from the frozen disjoint roots")
    if config["arms"] != _expected_arms():
        raise ValueError("B1 arms must be the exact ordered five-arm registry")
    if config["oracle"] != _expected_oracle():
        raise ValueError(
            "B1 oracle numerical contract differs from the frozen contract"
        )
    OracleTolerance(**config["oracle"]["tolerance"])
    if config["counterfactual"] != _expected_counterfactual():
        raise ValueError(
            "B1 counterfactual contract differs from the uncapped frozen contract"
        )
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
        raise ValueError(
            "B1 table/plot/binning contract differs from the frozen contract"
        )
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
    raw_source_paths = config["source_paths"]
    if not isinstance(raw_source_paths, list) or any(
        not isinstance(path, str) for path in raw_source_paths
    ):
        raise ValueError("B1 source_paths must be an ordered string array")
    source_paths = tuple(raw_source_paths)
    current_source_scope = source_paths == POLICY_SOURCE_PATHS
    legacy_frozen_scope = is_frozen and source_paths == LEGACY_POLICY_SOURCE_PATHS
    if not current_source_scope and not legacy_frozen_scope:
        raise ValueError(
            "B1 source_paths differ from the complete frozen implementation scope"
        )
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
        if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(
            source_paths
        ):
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
    if _canonical_file_hash(root / "requirements.txt") != config.get(
        "requirements_sha256"
    ):
        raise ValueError("B1 requirements hash mismatch")
    return experiment_id


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
