"""B1 fixed-shot policy-audit protocol and deterministic shard collector.

This package is intentionally separate from the immutable V3 paired collector.
It owns only protocol/provenance, fixed worker schedules, sampling-once
orchestration, and artifact integrity.  The scientific per-shot policy logic
is reached through one narrow adapter, :func:`_audit_policy_shot`, so the core
implementation can evolve without leaking ground truth into its API.

The submodules follow protocol order and each inherits this contract:
:mod:`._identity` (deterministic schedules and identity), :mod:`._protocol`
(draft/freeze validation), :mod:`._ledger` (artifact encodings and the
ground-truth firewall), :mod:`._shards` (worker collection and shard
verification), :mod:`._attestation` (probe attestation), and
:mod:`._collection` (multi-worker orchestration).  Analysis and report
generation remain downstream in :mod:`yoked.decoding.oracle.policy_analysis`.
This ``__init__`` re-exports the complete surface of the pre-package single
module (public names and the underscore-private helpers that tests, tools,
and the casebook layer use directly).
"""

from __future__ import annotations

# _identity must load first: it pins the native-thread environment before any
# submodule (or its dependencies) can import NumPy/PyMatching.
from yoked.decoding.oracle.policy_experiment._identity import (
    ANALYSIS_PLOT_NAMES,
    ANALYSIS_TABLE_NAMES,
    ARM_IDS,
    COUNTERFACTUAL_SCHEMA,
    DOMAIN_SCHEMA,
    EXPERIMENT_SCHEMA,
    GZIP_LEVEL,
    LEGACY_POLICY_SOURCE_PATHS,
    MANIFEST_SCHEMA,
    POLICY_SOURCE_PATHS,
    PROBE_ATTESTATION_SCHEMA,
    PROPOSAL_SCHEMA,
    PROTOCOL_SCHEMA,
    SCIENTIFIC_SHOTS,
    SCIENTIFIC_SHOTS_PER_WORKER,
    SCIENTIFIC_WORKERS,
    SEED_DERIVATION,
    SHARD_SCHEMA,
    SHOT_SCHEMA,
    THREAD_ENVIRONMENT,
    WorkerSpec,
    _ARM_ID_PATTERN,
    _SIDECARS,
    _atomic_json,
    _repo_root,
    _require_exact_keys,
    _require_sha256,
    _semantic_config,
    _sha256,
    _strict_json_load,
    derive_policy_worker_seed,
    policy_config_self_sha256,
    policy_experiment_id,
    policy_worker_schedule,
)
from yoked.decoding.oracle.policy_experiment._protocol import (
    _expected_arms,
    _expected_casebook_selection,
    _expected_cell,
    _expected_context_taxonomy,
    _expected_counterfactual,
    _expected_decoder,
    _expected_dem_options,
    _expected_oracle,
    _expected_report_contract,
    _expected_visibility_taxonomy,
    _graph_hashes,
    _validate_probe_attestation,
    default_policy_audit_draft,
    freeze_policy_protocol,
    inspect_policy_protocol,
    materialize_policy_draft,
    validate_policy_protocol,
)
from yoked.decoding.oracle.policy_experiment._ledger import (
    CASEBOOK_EXTRA_GROUND_TRUTH_KEYS,
    GROUND_TRUTH_FORBIDDEN_KEYS,
    NormalizedPolicyShot,
    SUPPORT_COMPONENT_FIELDS_V2,
    _COLLECTOR_GROUND_TRUTH_FIELDS,
    _COLLECTOR_OWNED_FIELDS,
    _artifact_metadata,
    _audit_policy_shot,
    _normalize_prediction,
    _normalized_context_union,
    _require_float_hex_companions,
    _separate_nondeterministic_timing,
    _validate_context_union_ledger,
    _validate_support_difference_ledger,
    canonical_jsonl,
    deterministic_gzip,
    forbid_ground_truth_keys,
)
from yoked.decoding.oracle.policy_experiment._shards import (
    _merge_core_row,
    _peak_rss_bytes,
    _real_graph_numerical_preflight,
    _row_identity,
    _shard_dir,
    _shot_performance_telemetry,
    _type7_quantiles_ns,
    _worker_gate_evidence,
    _worker_tail_censor_attestation,
    collect_policy_worker_shard,
    install_worker_shard,
    verify_worker_shard,
)
from yoked.decoding.oracle.policy_experiment._attestation import (
    _aggregate_collector_gate_attestations,
    _authenticated_analysis_file,
    _clean_tail_censor_attestation,
    _validate_campaign_performance_telemetry,
    _validate_collection_manifest,
    _validate_probe_projection,
    attest_completed_policy_probe,
)
from yoked.decoding.oracle.policy_experiment._collection import (
    IMMUTABLE_OUTPUT_PATTERNS,
    _WORKER_PREPARED,
    _reject_immutable_output,
    _validate_output_root,
    _worker_task,
    run_policy_collection,
)


__all__ = [
    "ARM_IDS",
    "CASEBOOK_EXTRA_GROUND_TRUTH_KEYS",
    "GROUND_TRUTH_FORBIDDEN_KEYS",
    "NormalizedPolicyShot",
    "PROTOCOL_SCHEMA",
    "SCIENTIFIC_SHOTS",
    "SCIENTIFIC_SHOTS_PER_WORKER",
    "SCIENTIFIC_WORKERS",
    "SUPPORT_COMPONENT_FIELDS_V2",
    "WorkerSpec",
    "canonical_jsonl",
    "forbid_ground_truth_keys",
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
