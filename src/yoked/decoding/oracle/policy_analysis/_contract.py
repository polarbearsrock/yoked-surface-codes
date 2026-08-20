"""Frozen schema names, closed vocabularies, and canonical JSON encoding.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` holds the frozen
identifier constants, the :class:`PolicyAnalysisError` contract-violation
type, and the canonical byte encoding and digest helpers shared by every
other slice.  It inherits the package's downstream-only contract: it never
imports circuit generation, sampling, matching, or decoding code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


ANALYSIS_SCHEMA = "promatch-l1-policy-audit-analysis-v1"
ANALYSIS_MANIFEST_SCHEMA = "promatch-l1-policy-audit-analysis-manifest-v1"
ANALYSIS_READY_SCHEMA = "promatch-l1-policy-audit-analysis-ready-v1"
HUMAN_REPORT_FORMAT = "promatch-l1-policy-audit-human-report-v1"
HUMAN_REPORT_FILE = "report.md"
PLOT_TABLE_SCHEMA = "promatch-l1-policy-audit-plot-table-v1"
CASEBOOK_SCHEMA = "promatch-l1-policy-audit-casebook-selection-v1"
SPARSE_UNSAFE_STATES = 100
CASEBOOK_MIN_STATES = 20
# Sync note: yoked.decoding.oracle.policy_audit.CONTEXT_PRIORITY is a
# deliberate twin.  This module's downstream-only contract forbids importing
# that decoding-adjacent producer, so the tuple is duplicated; keep both
# copies and their most-specific-first ordering (first match wins for the
# exclusive context) identical.
CONTEXT_PRIORITY = (
    "yoke",
    "true-boundary",
    "terminal",
    "cross-window",
    "cross-patch-or-basis",
    "support-cancellation",
    "in-domain",
)
CONTEXT_LABELS = frozenset(CONTEXT_PRIORITY)
DEGENERACY_DIAGNOSTICS = frozenset(
    {
        "same-pair-different-path-or-frame",
        "equal-weight-logical-class",
        "disconnected-support-reconfiguration",
        "unclassified",
    }
)
COLLECTOR_GATE_ATTESTATION_SCHEMA = "promatch-l1-policy-audit-fatal-gate-attestation-v1"
COLLECTOR_GATE_CHECKS = {
    3: ("scalar-batch-u0",),
    4: ("shadow-frozen-v3-equivalence",),
    7: (
        "backend-support-fsum",
        "decimal-4096",
        "uncached-repeatability",
        "tolerance-grid",
    ),
    8: ("actual-observable-invariance",),
    9: ("veto-state-frame-prefix-invariance",),
    14: ("matching-pair-and-support-reconciliation",),
    16: ("cached-uncached-oracle-repeatability",),
    18: ("execution-and-source-provenance",),
}
TERMINAL_ACTIONS = (
    "same-stage-alternative",
    "later-stage-alternative",
    "abstain-true-exhaustion",
    "censored-invalid",
)
CERTIFICATE_CLASSES = (
    "positive-cost-excess",
    "cost-compatible-frame-conflict",
    "O-frame-safe",
)
SHARD_FILES = {
    "shots": "shots.jsonl.gz",
    "proposals": "proposals.jsonl.gz",
    "counterfactuals": "counterfactuals.jsonl.gz",
    "domains": "domains.jsonl.gz",
}
EXPECTED_LEDGER_SCHEMAS = {
    "shots": "promatch-l1-policy-audit-shot-v1",
    "proposals": "promatch-l1-policy-audit-proposal-v1",
    "counterfactuals": "promatch-l1-policy-audit-counterfactual-v1",
    "domains": "promatch-l1-policy-audit-domain-v1",
}


class PolicyAnalysisError(ValueError):
    """The immutable audit corpus violates its analysis contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Returns the canonical UTF-8 encoding used by B1 artifact digests."""

    # Sync note: yoked.decoding._promatch_stats.canonical_json_bytes is a
    # deliberate near-twin kept separate by this module's downstream-only
    # contract.  It diverges on purpose: the stats variant requires a Mapping
    # and pre-validates the JSON tree, while this variant accepts any JSON
    # value (rows are lists too) and relies on allow_nan=False alone.  Keep
    # the dumps options (sort_keys/ensure_ascii/allow_nan/separators) in sync.
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyAnalysisError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
