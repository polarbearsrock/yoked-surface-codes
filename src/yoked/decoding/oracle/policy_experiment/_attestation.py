"""Probe attestation and campaign-manifest authentication.

This slice authenticates a completed exact-arm 100-shot probe (collection,
launch-gate projection, installed analysis, and casebook selection) and owns
the campaign-level manifest reconciliation shared by the probe attestor and
the multi-worker orchestrator.  It inherits the package isolation contract
(see ``__init__``): it verifies recorded artifacts and gate evidence only,
and never recomputes or consumes ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from yoked.decoding._promatch_experiment import _canonical_file_hash
from yoked.decoding._promatch_stats import canonical_json_bytes

from yoked.decoding.oracle.policy_experiment._identity import (
    ANALYSIS_PLOT_NAMES,
    ANALYSIS_TABLE_NAMES,
    EXPERIMENT_SCHEMA,
    MANIFEST_SCHEMA,
    PROBE_ATTESTATION_SCHEMA,
    SCIENTIFIC_WORKERS,
    WorkerSpec,
    _SIDECARS,
    _require_sha256,
    _sha256,
    _strict_json_load,
    policy_worker_schedule,
)
from yoked.decoding.oracle.policy_experiment._protocol import (
    _validate_probe_attestation,
    validate_policy_protocol,
)


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
        raise ValueError(
            f"probe analysis {group} artifact is missing or unsafe: {relative}"
        )
    _require_sha256(expected_hash, name=f"analysis.{group}.{relative}")
    if _canonical_file_hash(path) != expected_hash:
        raise ValueError(f"probe analysis {group} digest mismatch: {relative}")


def _validate_probe_projection(value: Any, *, gates: Mapping[str, Any]) -> None:
    fields = {
        "parent_setup_seconds",
        "parent_setup_seconds_hex",
        "parallel_worker_compile_seconds",
        "parallel_worker_compile_seconds_hex",
        "fixed_setup_seconds",
        "fixed_setup_seconds_hex",
        "variable_100_shot_seconds",
        "variable_100_shot_seconds_hex",
        "compressed_probe_bytes",
        "projected_wall_seconds",
        "projected_wall_seconds_hex",
        "projected_artifact_bytes",
        "free_output_bytes",
        "wall_gate_passed",
        "artifact_gate_passed",
        "free_space_gate_passed",
        "all_launch_gates_passed",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("probe projection fields are incomplete")
    for name in (
        "parent_setup_seconds",
        "parallel_worker_compile_seconds",
        "fixed_setup_seconds",
        "variable_100_shot_seconds",
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
    for name in (
        "compressed_probe_bytes",
        "projected_artifact_bytes",
        "free_output_bytes",
    ):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"probe projection has invalid count {name}")
    if not math.isclose(
        value["fixed_setup_seconds"],
        value["parent_setup_seconds"] + value["parallel_worker_compile_seconds"],
        rel_tol=1e-15,
        abs_tol=1e-12,
    ):
        raise ValueError("probe fixed setup projection does not reconcile")
    expected_wall = (
        value["fixed_setup_seconds"]
        + gates["probe_headroom_factor"] * 200 * value["variable_100_shot_seconds"]
    )
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
        raise ValueError(
            "probe wall/storage/free-space launch projection is inconsistent"
        )


def _aggregate_collector_gate_attestations(
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from yoked.decoding.oracle.policy_analysis import (
        COLLECTOR_GATE_ATTESTATION_SCHEMA,
        COLLECTOR_GATE_CHECKS,
    )

    for shard in shards:
        evidence = shard.get("collector_gate_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("real_graph") is not True:
            raise ValueError("campaign fatal gates require real-graph worker evidence")
        checks = evidence.get("checks")
        expected = {
            check
            for gate_checks in COLLECTOR_GATE_CHECKS.values()
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
        "schema",
        "parent_setup_ns",
        "worker_phase_ns",
        "parent_peak_rss_bytes",
        "parent_peak_rss_source",
        "new_worker_compile_ns",
        "scientifically_deterministic",
        "excluded_from_scientific_decisions",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != "promatch-l1-policy-audit-campaign-performance-v1"
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
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    mode: str,
    schedule: Sequence[WorkerSpec],
    shards: Sequence[Mapping[str, Any]],
) -> None:
    fields = {
        "schema",
        "experiment_id",
        "mode",
        "workers",
        "shots",
        "new_worker_processes_observed",
        "tail_censor_attestation",
        "shards",
        "campaign_wall_ns",
        "fatal_gate_attestations",
        "performance_telemetry",
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
            raise ValueError(
                "probe manifest did not observe exactly 32 worker processes"
            )
        _validate_probe_projection(
            manifest["probe_projection"], gates=config["launch_gates"]
        )


def attest_completed_policy_probe(
    probe_root: Path,
    *,
    expected_config: Mapping[str, Any],
    implementation_commit: str,
) -> dict[str, Any]:
    """Authenticates a completed exact-arm probe and its installed analysis."""

    # Shard authentication is resolved through the package namespace at call
    # time, exactly like the pre-package module-global lookup.
    from yoked.decoding.oracle import policy_experiment as _package

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
        raise ValueError(
            "probe experiment is not from the current implementation commit"
        )

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
    if (
        not shard_root.is_dir()
        or {p.name for p in shard_root.iterdir()} != expected_dirs
    ):
        raise ValueError(
            "probe shard directories do not match the exact 32-worker schedule"
        )
    verified = [
        _package.verify_worker_shard(
            shard_root / f"worker-{spec.worker_id:02d}",
            config=expected,
            mode="probe",
            spec=spec,
        )
        for spec in schedule
    ]
    if manifest["shards"] != verified:
        raise ValueError(
            "probe manifest does not match the authenticated worker shards"
        )
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
        raise ValueError(
            "probe COLLECTION_READY does not authenticate the exact manifest"
        )

    analysis_ready = loaded["analysis_ready"]
    if (
        set(analysis_ready)
        != {
            "schema",
            "experiment_id",
            "analysis_manifest_sha256",
            "casebook_selection_sha256",
            "report_file_sha256",
            "plots_rendered",
            "casebook_exhaustive_expansion_required_before_complete",
        }
        or analysis_ready.get("schema") != "promatch-l1-policy-audit-analysis-ready-v1"
        or analysis_ready.get("experiment_id") != experiment_id
        or analysis_ready.get("analysis_manifest_sha256") != hashes["analysis_manifest"]
        or analysis_ready.get("casebook_selection_sha256")
        != hashes["casebook_selection"]
        or analysis_ready.get("report_file_sha256") != hashes["analysis_report"]
        or analysis_ready.get("casebook_exhaustive_expansion_required_before_complete")
        is not True
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
    if (
        set(analysis_manifest) != analysis_manifest_fields
        or analysis_manifest.get("schema")
        != "promatch-l1-policy-audit-analysis-manifest-v1"
        or analysis_manifest.get("experiment_id") != experiment_id
        or analysis_manifest.get("summary_file_sha256") != hashes["analysis_summary"]
        or analysis_manifest.get("report_file_sha256") != hashes["analysis_report"]
        or source_hashes != expected_source_hashes
        or summary.get("schema") != "promatch-l1-policy-audit-analysis-v1"
        or set(summary)
        != {
            "schema",
            "experiment_id",
            "cell_id",
            "analysis_contract",
            "source_hashes",
            "tables",
            "casebook_selection",
            "analysis_sha256",
        }
        or summary.get("experiment_id") != experiment_id
        or summary.get("cell_id") != expected["cell"]["cell_id"]
        or summary.get("source_hashes") != expected_source_hashes
        or summary.get("casebook_selection") != loaded["casebook_selection"]
        or not isinstance(summary.get("tables"), Mapping)
        or set(summary["tables"]) != set(ANALYSIS_TABLE_NAMES)
        or not isinstance(analysis_contract, Mapping)
        or set(analysis_contract)
        != {
            "source",
            "sampling_or_decoding_reconstruction",
            "bootstrap_unit",
            "bootstrap_quantile",
            "proposal_bootstrap_replicates",
            "workload_bootstrap_replicates",
            "casebook_outcome_blind",
            "casebook_exhaustive_rows_excluded",
            "support_context_views_kept_distinct",
            "required_tail_telemetry",
            "complete_written_by_analyzer",
            "complete_accepted_as_analysis_substitute",
            "next_required_stage",
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
        or analysis_contract.get("complete_accepted_as_analysis_substitute")
        is not False
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
        if (
            not isinstance(records, Mapping)
            or set(records) != expected_analysis_files[group]
        ):
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
                    raise ValueError(
                        f"probe analysis table differs from summary: {relative}"
                    )
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
            or row.get("status")
            not in {"collector-attested", "passed-ledger-recomputed"}
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
