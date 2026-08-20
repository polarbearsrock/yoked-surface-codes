"""Offline, deterministic analysis for the ProMatch B1 policy audit.

This package is intentionally downstream-only.  It reads immutable canonical
``*.jsonl.gz`` worker shards, and it never imports circuit generation, sampling,
matching, or decoding code.  All oracle and context facts used here must have
already been committed to the collection ledgers.

Loading first authenticates the ready marker, manifest, shards, and row
relationships. The main analysis then constructs frozen tables, a deterministic
casebook selection, and plot payloads. Rendering is a final, separate step so
no presentation code can influence the measured records.

Every submodule inherits this contract; this ``__init__`` re-exports the
complete surface of the pre-package single module (public names and the
underscore-private helpers that tests exercise directly).
"""

from __future__ import annotations

from ._artifacts import (
    _install_bytes_atomic,
    _write_canonical_json,
    verify_existing_policy_analysis,
    write_policy_analysis,
)
from ._casebook import (
    _casebook_rank_digest,
    select_casebook,
)
from ._contract import (
    ANALYSIS_MANIFEST_SCHEMA,
    ANALYSIS_READY_SCHEMA,
    ANALYSIS_SCHEMA,
    CASEBOOK_MIN_STATES,
    CASEBOOK_SCHEMA,
    CERTIFICATE_CLASSES,
    COLLECTOR_GATE_ATTESTATION_SCHEMA,
    COLLECTOR_GATE_CHECKS,
    CONTEXT_LABELS,
    CONTEXT_PRIORITY,
    DEGENERACY_DIAGNOSTICS,
    EXPECTED_LEDGER_SCHEMAS,
    HUMAN_REPORT_FILE,
    HUMAN_REPORT_FORMAT,
    PLOT_TABLE_SCHEMA,
    SHARD_FILES,
    SPARSE_UNSAFE_STATES,
    TERMINAL_ACTIONS,
    PolicyAnalysisError,
    _sha256,
    _unique_object,
    canonical_json_bytes,
)
from ._corpus import (
    PolicyAuditCorpus,
    _collector_gate_attestations,
    _identity,
    _load_json,
    _load_jsonl_gzip,
    _manifest_record,
    _manifest_records,
    _merge_authenticated_timing,
    load_policy_audit,
)
from ._fields import (
    _as_bool,
    _as_nonnegative_int,
    _at,
    _deep_values,
    _float_value,
    _one_deep,
    _record_value,
    _required_float,
)
from ._plots import (
    _plot_payloads,
    _render_plots,
    _save_plot,
)
from ._report import (
    _report_count,
    _report_fraction,
    _report_number,
    _report_object,
    _report_ratio,
    _report_rows,
    _report_single_line,
    policy_human_report_bytes,
)
from ._rows import (
    _arm_metric,
    _arm_results,
    _arm_role,
    _component_labels,
    _context,
    _cost_compatible,
    _counterfactual_states,
    _domain_key,
    _domain_table,
    _failure,
    _frame_compatible,
    _normalize_shadow_proposals,
    _normalize_shots,
    _origin,
    _original_hw,
    _prediction_token,
    _proposal_sha,
    _shot_key,
    _sorted_labels,
    _stage,
    _state_key,
    _terminal_action,
    _validate_context_labels,
    certificate_class,
)
from ._stats import (
    _bootstrap_config,
    _bootstrap_fraction_table,
    _paired_table,
    clustered_bootstrap_ratios,
    derive_bootstrap_seed,
    distribution_summary,
    empirical_type7,
    exact_ecdf,
)
from ._tables import (
    _count_table,
    _outcome_name,
    _risk_count_table,
    _sum_optional,
    _unsafe_bin,
    analyze_policy_audit,
)
