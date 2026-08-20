"""Human report rendering from the frozen downstream analysis tables.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` renders the
deterministic markdown report solely from an already-computed analysis
object, failing closed on any missing or inconsistent table.  It inherits
the package's downstream-only contract: it never imports circuit generation,
sampling, matching, or decoding code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._contract import (
    ANALYSIS_SCHEMA,
    CONTEXT_PRIORITY,
    DEGENERACY_DIAGNOSTICS,
    HUMAN_REPORT_FORMAT,
    SPARSE_UNSAFE_STATES,
    TERMINAL_ACTIONS,
    PolicyAnalysisError,
)


def _report_object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyAnalysisError(f"human report requires {name} to be an object")
    return value


def _report_rows(value: Any, *, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping) for row in value
    ):
        raise PolicyAnalysisError(
            f"human report requires {name} to be an array of objects"
        )
    return value


def _report_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyAnalysisError(
            f"human report requires {name} to be a nonnegative integer"
        )
    return value


def _report_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyAnalysisError(f"human report requires {name} to be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyAnalysisError(f"human report requires {name} to be a finite number")
    return result


def _report_single_line(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in "\r\n|`")
    ):
        raise PolicyAnalysisError(
            f"human report requires {name} to be safe single-line text"
        )
    return value


def _report_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator:,} / 0 (undefined)"
    return f"{numerator:,} / {denominator:,} ({numerator / denominator:.3%})"


def _report_ratio(value: Any, *, name: str) -> str:
    if value is None:
        return "undefined"
    return f"{_report_number(value, name=name):.6g}"


def policy_human_report_bytes(analysis: Mapping[str, Any]) -> bytes:
    """Renders the frozen human report solely from downstream analysis tables."""

    if not isinstance(analysis, Mapping):
        raise PolicyAnalysisError("human report requires an analysis object")
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise PolicyAnalysisError("human report requires the exact analysis schema")
    experiment_id = _report_single_line(
        analysis.get("experiment_id"), name="experiment_id"
    )
    cell_id = _report_single_line(analysis.get("cell_id"), name="cell_id")
    tables = _report_object(analysis.get("tables"), name="tables")
    required_tables = {
        "overview",
        "paired_outcomes",
        "event_and_transaction_summary",
        "certificate_by_stage",
        "counterfactual_terminal_action",
        "context_views",
        "visibility_summary",
        "local_competitor_summary",
    }
    if not required_tables <= set(tables):
        missing = sorted(required_tables - set(tables))
        raise PolicyAnalysisError(f"human report is missing required tables: {missing}")

    overview = _report_object(tables["overview"], name="tables.overview")
    shots = _report_count(overview.get("shots"), name="overview.shots")
    if shots == 0:
        raise PolicyAnalysisError("human report requires at least one shot")
    workers = _report_count(overview.get("workers"), name="overview.workers")
    zero_event = _report_count(
        overview.get("zero_event_shots"), name="overview.zero_event_shots"
    )
    nonzero_event = _report_count(
        overview.get("nonzero_event_shots"), name="overview.nonzero_event_shots"
    )
    if zero_event + nonzero_event != shots:
        raise PolicyAnalysisError("human report shot denominators do not reconcile")
    unsafe_shots = _report_count(
        overview.get("shots_with_unsafe_durable_original"),
        name="overview.shots_with_unsafe_durable_original",
    )
    unsafe_states = _report_count(
        overview.get("unsafe_durable_original_commitments"),
        name="overview.unsafe_durable_original_commitments",
    )
    disagreements = _report_count(
        overview.get("u0_shadow_prediction_disagreements"),
        name="overview.u0_shadow_prediction_disagreements",
    )
    activated = _report_count(
        overview.get("activated_shots"), name="overview.activated_shots"
    )
    failure_counts = {
        "U0": _report_count(overview.get("u0_failures"), name="overview.u0_failures"),
        "shadow": _report_count(
            overview.get("shadow_failures"), name="overview.shadow_failures"
        ),
        "O-cost transactional": _report_count(
            overview.get("o_cost_tx_failures"), name="overview.o_cost_tx_failures"
        ),
    }
    if (
        overview.get("o_frame_tx_equals_u0_predictions") is not True
        or overview.get("o_frame_partial_equals_u0_predictions") is not True
    ):
        raise PolicyAnalysisError(
            "human report requires authenticated O-frame/U0 equality"
        )
    failure_counts["O-frame transactional"] = failure_counts["U0"]
    failure_counts["O-frame partial"] = failure_counts["U0"]

    workload_rows = _report_rows(
        tables["event_and_transaction_summary"],
        name="tables.event_and_transaction_summary",
    )
    workload_by_arm: dict[str, Mapping[str, Any]] = {}
    for row in workload_rows:
        arm = row.get("arm")
        if arm not in {"shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial"}:
            raise PolicyAnalysisError(
                "human report encountered an unknown workload arm"
            )
        if arm in workload_by_arm:
            raise PolicyAnalysisError(
                "human report encountered a duplicate workload arm"
            )
        if _report_count(row.get("shots"), name=f"workload.{arm}.shots") != shots:
            raise PolicyAnalysisError(
                "human report workload shot denominator disagrees"
            )
        workload_by_arm[str(arm)] = row
    if set(workload_by_arm) != {"shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial"}:
        raise PolicyAnalysisError("human report requires the exact workload arm set")

    paired = _report_object(tables["paired_outcomes"], name="tables.paired_outcomes")
    u0_shadow = _report_object(paired.get("u0_vs_shadow"), name="paired.u0_vs_shadow")
    if _report_count(u0_shadow.get("shots"), name="paired.u0_vs_shadow.shots") != shots:
        raise PolicyAnalysisError("human report paired denominator disagrees")
    regressions = _report_count(
        u0_shadow.get("regressions"), name="paired.u0_vs_shadow.regressions"
    )
    recoveries = _report_count(
        u0_shadow.get("recoveries"), name="paired.u0_vs_shadow.recoveries"
    )

    competitor = _report_object(
        tables["local_competitor_summary"], name="tables.local_competitor_summary"
    )
    shadow_commitments = _report_count(
        competitor.get("shadow_commitments"), name="local.shadow_commitments"
    )
    local_available = _report_count(competitor.get("available"), name="local.available")
    local_unavailable = _report_count(
        competitor.get("unavailable"), name="local.unavailable"
    )
    local_unrecorded = _report_count(
        competitor.get("unrecorded"), name="local.unrecorded"
    )
    if local_available + local_unavailable + local_unrecorded != shadow_commitments:
        raise PolicyAnalysisError("human report local-competitor denominator disagrees")

    visibility_rows = _report_rows(
        tables["visibility_summary"], name="tables.visibility_summary"
    )
    visibility: dict[str, Mapping[str, Any]] = {}
    expected_visibility = {
        "L1-local-dynamic",
        "L1-static-boundary",
        "temporal-neighbor-dynamic",
        "nonlocal-yoke-dynamic",
        "oracle-only",
        "posthoc-ground-truth",
    }
    for row in visibility_rows:
        label = row.get("visibility_class")
        if label not in expected_visibility or label in visibility:
            raise PolicyAnalysisError("human report visibility classes are not exact")
        denominator = _report_count(
            row.get("unsafe_state_denominator"), name=f"visibility.{label}.denominator"
        )
        if denominator != unsafe_states:
            raise PolicyAnalysisError("human report visibility denominator disagrees")
        visibility[str(label)] = row
    if set(visibility) != expected_visibility:
        raise PolicyAnalysisError("human report requires the exact visibility taxonomy")

    context_views = _report_object(tables["context_views"], name="tables.context_views")
    exclusive_rows = _report_rows(
        context_views.get("exclusive_support_component_context"),
        name="context_views.exclusive_support_component_context",
    )
    exclusive_counts: dict[str, int] = {}
    for row in exclusive_rows:
        label_value = row.get("label")
        label = "none" if label_value is None else str(label_value)
        if label not in {*CONTEXT_PRIORITY, "none"} or label in exclusive_counts:
            raise PolicyAnalysisError("human report exclusive contexts are not exact")
        if (
            _report_count(
                row.get("unsafe_state_denominator"), name=f"context.{label}.denominator"
            )
            != unsafe_states
        ):
            raise PolicyAnalysisError("human report context denominator disagrees")
        exclusive_counts[label] = _report_count(
            row.get("count"), name=f"context.{label}.count"
        )
    if set(exclusive_counts) != {*CONTEXT_PRIORITY, "none"}:
        raise PolicyAnalysisError(
            "human report requires the exact exclusive-context vocabulary"
        )
    if sum(exclusive_counts.values()) != unsafe_states:
        raise PolicyAnalysisError(
            "human report exclusive contexts do not partition unsafe states"
        )

    degeneracy_rows = _report_rows(
        context_views.get("degeneracy_diagnostics"),
        name="context_views.degeneracy_diagnostics",
    )
    degeneracy_counts: dict[str, int] = {}
    for row in degeneracy_rows:
        label = row.get("label")
        if label not in DEGENERACY_DIAGNOSTICS or label in degeneracy_counts:
            raise PolicyAnalysisError(
                "human report degeneracy diagnostics are not exact"
            )
        if (
            _report_count(
                row.get("unsafe_state_denominator"),
                name=f"degeneracy.{label}.denominator",
            )
            != unsafe_states
        ):
            raise PolicyAnalysisError("human report degeneracy denominator disagrees")
        degeneracy_counts[str(label)] = _report_count(
            row.get("count"), name=f"degeneracy.{label}.count"
        )
    if set(degeneracy_counts) != DEGENERACY_DIAGNOSTICS:
        raise PolicyAnalysisError("human report requires exact degeneracy diagnostics")

    certificate_stage_rows = _report_rows(
        tables["certificate_by_stage"], name="tables.certificate_by_stage"
    )
    stage_statuses: dict[int, str] = {}
    if any(
        row.get("stratum_status")
        not in {"insufficient-for-rule-formulation", "descriptive-only"}
        for row in certificate_stage_rows
    ):
        raise PolicyAnalysisError(
            "human report encountered an unknown stage stratum status"
        )
    for row in certificate_stage_rows:
        stage = _report_count(row.get("stage"), name="certificate.stage")
        if stage not in {1, 2, 3, 4}:
            raise PolicyAnalysisError("human report encountered an invalid stage")
        status = str(row["stratum_status"])
        if stage in stage_statuses and stage_statuses[stage] != status:
            raise PolicyAnalysisError("human report stage stratum statuses disagree")
        stage_statuses[stage] = status
    if set(stage_statuses) != {1, 2, 3, 4}:
        raise PolicyAnalysisError("human report requires all four stage strata")
    sparse_stages = sum(
        status == "insufficient-for-rule-formulation"
        for status in stage_statuses.values()
    )

    terminal_rows = _report_rows(
        tables["counterfactual_terminal_action"],
        name="tables.counterfactual_terminal_action",
    )
    terminal_counts: dict[str, int] = {}
    for row in terminal_rows:
        action = row.get("terminal_action")
        if action not in TERMINAL_ACTIONS or action in terminal_counts:
            raise PolicyAnalysisError("human report terminal actions are not exact")
        if (
            _report_count(row.get("denominator"), name=f"terminal.{action}.denominator")
            != unsafe_states
        ):
            raise PolicyAnalysisError(
                "human report terminal-action denominator disagrees"
            )
        terminal_counts[str(action)] = _report_count(
            row.get("count"), name=f"terminal.{action}.count"
        )
    if set(terminal_counts) != set(TERMINAL_ACTIONS):
        raise PolicyAnalysisError("human report requires exact terminal actions")

    lines = [
        f"<!-- format: {HUMAN_REPORT_FORMAT} -->",
        "# ProMatch L1 B1 policy-audit report",
        "",
        f"Experiment `{experiment_id}`; cell `{cell_id}`.",
        "",
        (
            "This deterministic report is generated only from the authenticated downstream "
            "analysis. It does not reconstruct sampling or decoding. All associations and "
            "explanations below are hypothesis-generating, not causal proof."
        ),
        "",
        "## Population and denominators",
        "",
        f"- Physical shots: {shots:,} across {workers:,} workers.",
        f"- Nonzero-event shots: {_report_fraction(nonzero_event, shots)}; zero-event shots: {_report_fraction(zero_event, shots)}.",
        f"- Predecoder-activated shots: {_report_fraction(activated, shots)}.",
        f"- Shots with at least one unsafe durable original commitment: {_report_fraction(unsafe_shots, shots)}.",
        f"- Unsafe durable original commitments (the denominator for context, visibility, and counterfactual summaries): {unsafe_states:,}.",
        f"- U0/shadow prediction-discordant shots: {_report_fraction(disagreements, shots)}.",
        "",
        "## Arm errors and detector-event workload",
        "",
        "Logical-error denominators are physical shots. Workload is a ratio of sums over the same physical shots; its denominator is the summed original detector-event count.",
        "",
        "| Arm | Logical errors | Final/original detector events | Durable / provisional / rollback-lost events |",
        "| --- | ---: | ---: | ---: |",
        f"| U0 | {_report_fraction(failure_counts['U0'], shots)} | reference; not separately tabulated | not applicable |",
    ]
    display_arms = (
        ("shadow", "shadow"),
        ("o-cost-tx", "O-cost transactional"),
        ("o-frame-tx", "O-frame transactional"),
        ("o-frame-partial", "O-frame partial"),
    )
    for arm, display in display_arms:
        row = workload_by_arm[arm]
        original = _report_count(
            row.get("sum_original_detector_hw"), name=f"workload.{arm}.original"
        )
        final = _report_count(
            row.get("sum_final_residual_detector_hw"), name=f"workload.{arm}.final"
        )
        durable = _report_count(
            row.get("durable_events_removed"), name=f"workload.{arm}.durable"
        )
        if original - final != durable:
            raise PolicyAnalysisError(
                "human report workload event totals do not reconcile"
            )
        provisional = row.get("provisional_events_removed")
        rollback = row.get("events_lost_to_rollback")
        transaction = (
            "unavailable"
            if provisional is None or rollback is None
            else f"{durable:,} / {_report_count(provisional, name=f'workload.{arm}.provisional'):,} / {_report_count(rollback, name=f'workload.{arm}.rollback'):,}"
        )
        lines.append(
            f"| {display} | {_report_fraction(failure_counts[display], shots)} | "
            f"{final:,} / {original:,} = {_report_ratio(row.get('R_event'), name=f'workload.{arm}.R_event')} | {transaction} |"
        )
    lines.extend(
        [
            "",
            f"For U0 versus shadow, regressions were {_report_fraction(regressions, shots)} and recoveries were {_report_fraction(recoveries, shots)}.",
            "O-frame transactional and partial predictions were authenticated as exactly equal to U0 predictions.",
            "",
            "## Locally observable policy clues",
            "",
            (
                "These are candidate clues available at the L1 decision surface, not labels "
                "showing whether a commitment was truly safe."
            ),
            "",
            f"- A same-stage local competitor was recorded for {_report_fraction(local_available, shadow_commitments)} shadow commitments; {_report_fraction(local_unavailable, shadow_commitments)} had none and {_report_fraction(local_unrecorded, shadow_commitments)} were unrecorded.",
        ]
    )
    for label in ("L1-local-dynamic", "L1-static-boundary"):
        row = visibility[label]
        present = _report_count(
            row.get("unsafe_states_with_class"), name=f"visibility.{label}.states"
        )
        occurrences = _report_count(
            row.get("recorded_field_occurrences"), name=f"visibility.{label}.fields"
        )
        lines.append(
            f"- `{label}` fields appeared on {_report_fraction(present, unsafe_states)} unsafe states ({occurrences:,} recorded field occurrences)."
        )
    lines.extend(
        [
            "",
            "No threshold or decision rule is licensed by these descriptive local associations.",
            "",
            "## Nonlocal and oracle-only explanations",
            "",
            (
                "Certificates, matched-partner paths, support paths, and support-difference "
                "components are oracle-only explanations. Their context labels must not be "
                "treated as locally observable policy inputs, even when the label is `in-domain`."
            ),
            "",
        ]
    )
    for label in CONTEXT_PRIORITY:
        lines.append(
            f"- Exclusive support context `{label}`: {_report_fraction(exclusive_counts[label], unsafe_states)} unsafe commitments."
        )
    lines.append(
        f"- No exclusive support context: {_report_fraction(exclusive_counts['none'], unsafe_states)} unsafe commitments."
    )
    for label in (
        "temporal-neighbor-dynamic",
        "nonlocal-yoke-dynamic",
        "oracle-only",
        "posthoc-ground-truth",
    ):
        row = visibility[label]
        present = _report_count(
            row.get("unsafe_states_with_class"), name=f"visibility.{label}.states"
        )
        occurrences = _report_count(
            row.get("recorded_field_occurrences"), name=f"visibility.{label}.fields"
        )
        lines.append(
            f"- Visibility `{label}`: {_report_fraction(present, unsafe_states)} unsafe states ({occurrences:,} recorded field occurrences)."
        )
    lines.extend(
        [
            "",
            "The exclusive context is only a display-priority partition; the distinct multi-label support views remain authoritative.",
            "",
            "## Counterfactual outcomes and limitations",
            "",
        ]
    )
    for action in TERMINAL_ACTIONS:
        lines.append(
            f"- `{action}`: {_report_fraction(terminal_counts[action], unsafe_states)} unsafe commitments."
        )
    lines.extend(
        [
            "",
            f"Sparse-stratum rule: fewer than {SPARSE_UNSAFE_STATES} unsafe states is insufficient for rule formulation. {sparse_stages:,} / {len(stage_statuses):,} stage strata are marked insufficient; all displayed strata remain descriptive only.",
            f"Tied-support diagnostic `equal-weight-logical-class`: {_report_fraction(degeneracy_counts['equal-weight-logical-class'], unsafe_states)} unsafe commitments.",
            f"Other support diagnostics are `same-pair-different-path-or-frame` {_report_fraction(degeneracy_counts['same-pair-different-path-or-frame'], unsafe_states)}, `disconnected-support-reconfiguration` {_report_fraction(degeneracy_counts['disconnected-support-reconfiguration'], unsafe_states)}, and `unclassified` {_report_fraction(degeneracy_counts['unclassified'], unsafe_states)}. Disconnected component graph roles are retained as structural evidence but excluded from policy-visible candidate context. Diagnostics can overlap and therefore do not form a partition.",
            "",
            "Sparse or tied support can make apparent context and margin patterns unstable. This audit can prioritize follow-up hypotheses; it cannot identify a causal policy rule.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")
