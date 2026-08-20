"""Frozen table construction for the B1 policy-audit analysis.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` prepares the
normalized corpus inputs and assembles the frozen analysis tables through
per-table-family builders orchestrated by :func:`analyze_policy_audit`.  It
inherits the package's downstream-only contract: it never imports circuit
generation, sampling, matching, or decoding code.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._casebook import select_casebook
from ._contract import (
    ANALYSIS_SCHEMA,
    CERTIFICATE_CLASSES,
    CONTEXT_PRIORITY,
    DEGENERACY_DIAGNOSTICS,
    SPARSE_UNSAFE_STATES,
    TERMINAL_ACTIONS,
    PolicyAnalysisError,
    _sha256,
    canonical_json_bytes,
)
from ._corpus import PolicyAuditCorpus, _collector_gate_attestations, _identity
from ._fields import _as_nonnegative_int, _at
from ._rows import (
    _arm_role,
    _counterfactual_states,
    _domain_table,
    _normalize_shadow_proposals,
    _normalize_shots,
    _shot_key,
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


def _outcome_name(shot: Mapping[str, Any]) -> str:
    u0 = shot["failures"]["u0"]
    pu = shot["failures"]["shadow"]
    prediction_agreement = shot["predictions"]["u0"] == shot["predictions"]["shadow"]
    if prediction_agreement:
        return "prediction-agreement"
    if not u0 and pu:
        return "regression"
    if u0 and not pu:
        return "recovery"
    if u0 and pu:
        return "prediction-discordant-both-wrong"
    raise PolicyAnalysisError(
        "prediction-discordant arms cannot both be correct against one actual observable"
    )


def _unsafe_bin(count: int) -> str:
    return str(count) if count < 4 else "4+"


def _count_table(
    rows: Iterable[Mapping[str, Any]], *, keys: Sequence[str]
) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter(
        tuple(row.get(key) for key in keys) for row in rows
    )
    return [
        {**dict(zip(keys, values)), "count": count}
        for values, count in sorted(
            counts.items(), key=lambda item: tuple(str(v) for v in item[0])
        )
    ]


def _sum_optional(values: Iterable[Any], *, name: str) -> int | None:
    materialized = list(values)
    if not materialized or any(value is None for value in materialized):
        return None
    return sum(_as_nonnegative_int(value, name=name) for value in materialized)


def _risk_count_table(
    rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]
) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        counts[key][1] += 1
        counts[key][0] += row["certificate_class"] != "O-frame-safe"
    return [
        {
            **dict(zip(keys, values)),
            "unsafe": count[0],
            "denominator": count[1],
            "unsafe_fraction": count[0] / count[1],
            "stratum_status": (
                "insufficient-for-rule-formulation"
                if count[0] < SPARSE_UNSAFE_STATES
                else "descriptive-only"
            ),
        }
        for values, count in sorted(
            counts.items(), key=lambda item: tuple(str(v) for v in item[0])
        )
    ]


@dataclass
class _PreparedInputs:
    """Normalized, reconciled corpus inputs shared by the family builders."""

    shots: list[dict[str, Any]]
    cell_id: str
    shot_keys: list[tuple[str, int]]
    proposals: list[dict[str, Any]]
    states: list[dict[str, Any]]
    unsafe: list[dict[str, Any]]
    shot_by_key: dict[tuple[str, int], dict[str, Any]]
    proposals_by_shot: Mapping[tuple[str, int], list[dict[str, Any]]]
    states_by_shot: Mapping[tuple[str, int], list[dict[str, Any]]]
    discordant: list[tuple[str, int]]
    proposal_replicates: int
    proposal_seed: str
    workload_replicates: int
    workload_seed: str


def _prepare_inputs(corpus: PolicyAuditCorpus) -> _PreparedInputs:
    """Normalizes and cross-reconciles the corpus rows for table building."""

    shots = _normalize_shots(corpus.rows["shots"])
    cells = sorted({shot["cell_id"] for shot in shots})
    if len(cells) != 1:
        raise PolicyAnalysisError("B1 analysis requires exactly one physical cell")
    cell_id = cells[0]
    shot_keys = [(shot["cell_id"], shot["global_shot_id"]) for shot in shots]
    if len(shot_keys) != len(set(shot_keys)):
        raise PolicyAnalysisError("cell/global shot identities are not unique")

    proposals = [
        row
        for row in _normalize_shadow_proposals(corpus.rows["proposals"])
        if row["durable"]
    ]
    states = _counterfactual_states(corpus.rows["counterfactuals"])
    if any(state["terminal_action"] == "censored-invalid" for state in states):
        raise PolicyAnalysisError(
            "uncapped counterfactual ledger contains a censored state"
        )
    unsafe = [row for row in proposals if row["certificate_class"] != "O-frame-safe"]
    unsafe_by_sha = {
        (row["cell_id"], row["global_shot_id"], row["proposal_sha256"]): row
        for row in unsafe
    }
    states_by_sha = {
        (
            row["cell_id"],
            row["global_shot_id"],
            str(row["original_proposal_sha256"]),
        ): row
        for row in states
    }
    if len(unsafe_by_sha) != len(unsafe) or len(states_by_sha) != len(states):
        raise PolicyAnalysisError("unsafe proposal/state identity is not unique")
    if set(unsafe_by_sha) != set(states_by_sha):
        raise PolicyAnalysisError(
            "unsafe original commitments do not reconcile one-to-one"
        )
    for proposal_sha, state in states_by_sha.items():
        # The counterfactual sidecar references the original proposal digest;
        # trajectory order is owned by that original shadow commitment.
        state["trajectory_commit_index"] = unsafe_by_sha[proposal_sha][
            "trajectory_commit_index"
        ]

    shot_by_key = {(shot["cell_id"], shot["global_shot_id"]): shot for shot in shots}
    proposals_by_shot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    states_by_shot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        proposals_by_shot[(proposal["cell_id"], proposal["global_shot_id"])].append(
            proposal
        )
    for state in states:
        states_by_shot[(state["cell_id"], state["global_shot_id"])].append(state)
    for key, values in proposals_by_shot.items():
        sequence = [value["trajectory_commit_index"] for value in values]
        if len(values) > 1 and any(value is None for value in sequence):
            raise PolicyAnalysisError(
                f"shot {key} lacks original-trajectory commitment order"
            )
        if all(value is not None for value in sequence):
            values.sort(
                key=lambda value: (
                    int(value["trajectory_commit_index"]),
                    value["proposal_sha256"],
                )
            )
    for key, values in states_by_shot.items():
        sequence = [value["trajectory_commit_index"] for value in values]
        if len(values) > 1 and any(value is None for value in sequence):
            raise PolicyAnalysisError(f"shot {key} lacks unsafe-state trajectory order")
        if all(value is not None for value in sequence):
            values.sort(
                key=lambda value: (
                    int(value["trajectory_commit_index"]),
                    value["original_proposal_sha256"],
                )
            )

    discordant = []
    for key, shot in shot_by_key.items():
        u0_prediction = shot["predictions"]["u0"]
        pu_prediction = shot["predictions"]["shadow"]
        if u0_prediction is None or pu_prediction is None:
            raise PolicyAnalysisError("shot ledger lacks exact U0/PU predictions")
        if u0_prediction != pu_prediction:
            discordant.append(key)
            if not any(not row["frame_compatible"] for row in proposals_by_shot[key]):
                raise PolicyAnalysisError(
                    "U0/PU-discordant shot has no durable original frame conflict"
                )

    proposal_replicates, proposal_seed = _bootstrap_config(
        corpus.config, family="proposal"
    )
    workload_replicates, workload_seed = _bootstrap_config(
        corpus.config, family="workload"
    )
    return _PreparedInputs(
        shots=shots,
        cell_id=cell_id,
        shot_keys=shot_keys,
        proposals=proposals,
        states=states,
        unsafe=unsafe,
        shot_by_key=shot_by_key,
        proposals_by_shot=proposals_by_shot,
        states_by_shot=states_by_shot,
        discordant=discordant,
        proposal_replicates=proposal_replicates,
        proposal_seed=proposal_seed,
        workload_replicates=workload_replicates,
        workload_seed=workload_seed,
    )


def _overview_and_paired_tables(
    corpus: PolicyAuditCorpus, prepared: _PreparedInputs
) -> dict[str, Any]:
    """Builds the ``overview`` and ``paired_outcomes`` tables."""

    shots = prepared.shots
    cell_id = prepared.cell_id
    shot_keys = prepared.shot_keys
    states = prepared.states
    states_by_shot = prepared.states_by_shot
    discordant = prepared.discordant
    paired = {
        "u0_vs_shadow": _paired_table(
            [shot["failures"]["u0"] for shot in shots],
            [shot["failures"]["shadow"] for shot in shots],
        ),
        "u0_vs_o_cost_tx": _paired_table(
            [shot["failures"]["u0"] for shot in shots],
            [shot["failures"]["o-cost-tx"] for shot in shots],
        ),
    }
    activated_shot_keys: set[tuple[str, int]] = set()
    for row in corpus.rows["domains"]:
        arm_id = _at(row, "arm_id", required=True)
        status = _at(
            row, "domain_terminal_status", "status", "outcome.status", required=True
        )
        if (
            isinstance(arm_id, str)
            and _arm_role(arm_id) == "shadow"
            and status != "below-limit"
        ):
            activated_shot_keys.add(_shot_key(row))
    overview = {
        "cell_id": cell_id,
        "physical_cell": corpus.config.get("cell"),
        "arm_definitions": corpus.config.get("arms"),
        "source_hashes": corpus.source_hashes,
        "shots": len(shots),
        "workers": len({_identity(row)[1] for row in corpus.rows["shots"]}),
        "zero_event_shots": sum(shot["zero_event"] for shot in shots),
        "nonzero_event_shots": sum(not shot["zero_event"] for shot in shots),
        "activated_shots": len(activated_shot_keys),
        "u0_failures": sum(shot["failures"]["u0"] for shot in shots),
        "shadow_failures": sum(shot["failures"]["shadow"] for shot in shots),
        "o_cost_tx_failures": sum(shot["failures"]["o-cost-tx"] for shot in shots),
        "u0_shadow_prediction_disagreements": len(discordant),
        "shots_with_unsafe_durable_original": sum(
            bool(states_by_shot[key]) for key in shot_keys
        ),
        "unsafe_durable_original_commitments": len(states),
        "o_frame_tx_equals_u0_predictions": True,
        "o_frame_partial_equals_u0_predictions": True,
    }
    overview["predecoder_activation_rate"] = overview["activated_shots"] / len(shots)
    overview["u0_shadow_prediction_disagreement_rate"] = len(discordant) / len(shots)
    overview["shots_with_unsafe_durable_original_rate"] = overview[
        "shots_with_unsafe_durable_original"
    ] / len(shots)
    return {"overview": overview, "paired_outcomes": paired}


def _certificate_tables(prepared: _PreparedInputs) -> dict[str, Any]:
    """Builds the certificate-by-stage/domain and unsafe-fraction tables."""

    proposals = prepared.proposals
    shot_keys = prepared.shot_keys
    cell_id = prepared.cell_id
    proposal_replicates = prepared.proposal_replicates
    proposal_seed = prepared.proposal_seed
    certificate_rows = []
    for stage in range(1, 5):
        stage_rows = [row for row in proposals if row["stage"] == stage]
        for certificate in CERTIFICATE_CLASSES:
            certificate_rows.append(
                {
                    "stage": stage,
                    "certificate_class": certificate,
                    "count": sum(
                        row["certificate_class"] == certificate for row in stage_rows
                    ),
                    "stage_denominator": len(stage_rows),
                }
            )
    certificate_categories = [
        f"stage={stage}|class={certificate}"
        for stage in range(1, 5)
        for certificate in CERTIFICATE_CLASSES
    ]
    certificate_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=certificate_categories,
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                f"stage={row['stage']}|class={certificate}",
                row["certificate_class"] == certificate,
            )
            for row in proposals
            for certificate in CERTIFICATE_CLASSES
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="certificate-fraction-by-stage",
    )
    interval_by_category = {row["category"]: row for row in certificate_intervals}
    for row in certificate_rows:
        category = f"stage={row['stage']}|class={row['certificate_class']}"
        row["fraction"] = interval_by_category[category]["fraction"]
        row["bootstrap"] = interval_by_category[category]["bootstrap"]
        unsafe_in_stage = sum(
            proposal["stage"] == row["stage"]
            and proposal["certificate_class"] != "O-frame-safe"
            for proposal in proposals
        )
        row["unsafe_state_count_for_sparse_rule"] = unsafe_in_stage
        row["stratum_status"] = (
            "insufficient-for-rule-formulation"
            if unsafe_in_stage < SPARSE_UNSAFE_STATES
            else "descriptive-only"
        )
    domain_certificate_counts = Counter(
        (*row["domain"], row["stage"], row["certificate_class"]) for row in proposals
    )
    observed_domain_stages = sorted(
        {(*row["domain"], row["stage"]) for row in proposals},
        key=lambda values: tuple(str(value) for value in values),
    )
    certificate_by_domain = [
        {
            "patch_id": domain_stage[0],
            "basis": domain_stage[1],
            "window_id": domain_stage[2],
            "stage": domain_stage[3],
            "certificate_class": certificate,
            "count": domain_certificate_counts[(*domain_stage, certificate)],
        }
        for domain_stage in observed_domain_stages
        for certificate in CERTIFICATE_CLASSES
    ]
    domain_stage_denominators = Counter(
        (row["domain"][0], row["domain"][1], row["domain"][2], row["stage"])
        for row in proposals
    )
    domain_stage_unsafe = Counter(
        (row["domain"][0], row["domain"][1], row["domain"][2], row["stage"])
        for row in proposals
        if row["certificate_class"] != "O-frame-safe"
    )
    domain_categories = sorted(
        {
            canonical_json_bytes(
                [
                    row["domain"][0],
                    row["domain"][1],
                    row["domain"][2],
                    row["stage"],
                    certificate,
                ]
            ).decode("utf-8")
            for row in proposals
            for certificate in CERTIFICATE_CLASSES
        }
    )
    domain_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=domain_categories,
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                canonical_json_bytes(
                    [
                        row["domain"][0],
                        row["domain"][1],
                        row["domain"][2],
                        row["stage"],
                        certificate,
                    ]
                ).decode("utf-8"),
                row["certificate_class"] == certificate,
            )
            for row in proposals
            for certificate in CERTIFICATE_CLASSES
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="certificate-fraction-by-domain-and-stage",
    )
    domain_interval = {row["category"]: row for row in domain_intervals}
    for row in certificate_by_domain:
        key = (row["patch_id"], row["basis"], row["window_id"], row["stage"])
        denominator = domain_stage_denominators[key]
        row["domain_stage_denominator"] = denominator
        row["fraction"] = row["count"] / denominator
        category = canonical_json_bytes([*key, row["certificate_class"]]).decode(
            "utf-8"
        )
        row["bootstrap"] = domain_interval[category]["bootstrap"]
        row["unsafe_state_count_for_sparse_rule"] = domain_stage_unsafe[key]
        row["stratum_status"] = (
            "insufficient-for-rule-formulation"
            if domain_stage_unsafe[key] < SPARSE_UNSAFE_STATES
            else "descriptive-only"
        )
    unsafe_stage = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=[str(stage) for stage in range(1, 5)],
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                str(row["stage"]),
                row["certificate_class"] != "O-frame-safe",
            )
            for row in proposals
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="unsafe-fraction-by-stage",
    )
    return {
        "certificate_by_stage": certificate_rows,
        "certificate_by_domain": certificate_by_domain,
        "unsafe_fraction_by_stage": unsafe_stage,
    }


def _counterfactual_tables(prepared: _PreparedInputs) -> dict[str, Any]:
    """Builds the terminal-action, first-safe-rank, and stage-transition tables."""

    states = prepared.states
    shot_keys = prepared.shot_keys
    cell_id = prepared.cell_id
    proposal_replicates = prepared.proposal_replicates
    proposal_seed = prepared.proposal_seed
    terminal_actions = []
    action_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=list(TERMINAL_ACTIONS),
        observations=[
            (
                (state["cell_id"], state["global_shot_id"]),
                action,
                state["terminal_action"] == action,
            )
            for state in states
            for action in TERMINAL_ACTIONS
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="counterfactual-terminal-action",
    )
    action_interval = {row["category"]: row for row in action_intervals}
    for action in TERMINAL_ACTIONS:
        group = [state for state in states if state["terminal_action"] == action]
        terminal_actions.append(
            {
                "terminal_action": action,
                "count": len(group),
                "denominator": len(states),
                "fraction": None if not states else len(group) / len(states),
                "bootstrap": action_interval[action]["bootstrap"],
                "stratum_status": (
                    "insufficient-for-rule-formulation"
                    if len(group) < SPARSE_UNSAFE_STATES
                    else "descriptive-only"
                ),
            }
        )
    first_safe_rank = _count_table(
        ({"first_safe_rank": state["first_safe_rank"]} for state in states),
        keys=("first_safe_rank",),
    )
    rank_values = sorted(
        {state["first_safe_rank"] for state in states},
        key=lambda value: (value is None, -1 if value is None else int(value)),
    )
    rank_categories = [
        "abstain" if value is None else f"rank={value}" for value in rank_values
    ]
    rank_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=rank_categories,
        observations=[
            (
                (state["cell_id"], state["global_shot_id"]),
                "abstain" if candidate is None else f"rank={candidate}",
                state["first_safe_rank"] == candidate,
            )
            for state in states
            for candidate in rank_values
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="first-safe-rank-fractions",
    )
    rank_interval = {row["category"]: row for row in rank_intervals}
    for row in first_safe_rank:
        category = (
            "abstain"
            if row["first_safe_rank"] is None
            else f"rank={row['first_safe_rank']}"
        )
        row["denominator"] = len(states)
        row["fraction"] = None if not states else row["count"] / len(states)
        row["bootstrap"] = rank_interval[category]["bootstrap"]
        row["stratum_status"] = (
            "insufficient-for-rule-formulation"
            if row["count"] < SPARSE_UNSAFE_STATES
            else "descriptive-only"
        )
    transitions = _count_table(
        (
            {
                "original_stage": state["original_stage"],
                "terminal_stage": state["first_safe_stage"]
                if state["first_safe_stage"] is not None
                else "abstain",
            }
            for state in states
        ),
        keys=("original_stage", "terminal_stage"),
    )
    return {
        "counterfactual_terminal_action": terminal_actions,
        "first_safe_rank": first_safe_rank,
        "stage_transition": transitions,
    }


def _context_view_tables(prepared: _PreparedInputs) -> dict[str, Any]:
    """Builds the distinct context views and the visibility summary."""

    unsafe = prepared.unsafe
    shot_keys = prepared.shot_keys
    cell_id = prepared.cell_id
    proposal_replicates = prepared.proposal_replicates
    proposal_seed = prepared.proposal_seed
    context_fields = {
        "matched_partner_labels": "matched_partner_labels",
        "support_path_labels": "support_path_labels",
        "support_difference_component_labels": "support_difference_component_labels",
        "omitted_context_labels": "omitted_context_labels",
        "degeneracy_diagnostics": "degeneracy_diagnostics",
    }
    context_vocabularies = {
        "matched_partner_labels": CONTEXT_PRIORITY,
        "support_path_labels": CONTEXT_PRIORITY,
        "support_difference_component_labels": CONTEXT_PRIORITY,
        "omitted_context_labels": CONTEXT_PRIORITY,
        "degeneracy_diagnostics": tuple(sorted(DEGENERACY_DIAGNOSTICS)),
    }
    context_tables = {
        view: [
            {
                "label": label,
                "count": sum(label in row[field] for row in unsafe),
            }
            for label in vocabulary
        ]
        for view, (field, vocabulary) in {
            view: (field, context_vocabularies[view])
            for view, field in context_fields.items()
        }.items()
    }
    context_tables["exclusive_support_component_context"] = [
        {
            "label": None if label == "none" else label,
            "count": sum(
                (row["exclusive_support_component_context"] or "none") == label
                for row in unsafe
            ),
        }
        for label in (*CONTEXT_PRIORITY, "none")
    ]
    context_categories = [
        f"{view}|{label}"
        for view, vocabulary in context_vocabularies.items()
        for label in vocabulary
    ] + [
        f"exclusive_support_component_context|{label}"
        for label in (*CONTEXT_PRIORITY, "none")
    ]
    context_observations = []
    for row in unsafe:
        shot_key = (row["cell_id"], row["global_shot_id"])
        for view, field in context_fields.items():
            labels = row[field]
            for label in context_vocabularies[view]:
                context_observations.append(
                    (shot_key, f"{view}|{label}", label in labels)
                )
        exclusive = row["exclusive_support_component_context"] or "none"
        for label in (*CONTEXT_PRIORITY, "none"):
            context_observations.append(
                (
                    shot_key,
                    f"exclusive_support_component_context|{label}",
                    exclusive == label,
                )
            )
    context_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=context_categories,
        observations=context_observations,
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="unsafe-context-fractions",
    )
    context_interval = {row["category"]: row for row in context_intervals}
    for name, table in context_tables.items():
        for row in table:
            row["unsafe_state_denominator"] = len(unsafe)
            interval = context_interval.get(f"{name}|{row['label'] or 'none'}")
            if interval is not None:
                row["fraction"] = interval["fraction"]
                row["bootstrap"] = interval["bootstrap"]
            row["stratum_status"] = (
                "insufficient-for-rule-formulation"
                if row["count"] < SPARSE_UNSAFE_STATES
                else "descriptive-only"
            )
    visibility_field_counts: Counter[str] = Counter()
    visibility_state_counts: Counter[str] = Counter()
    for row in unsafe:
        raw_visibility = _at(row["raw"], "feature_visibility")
        present = {"oracle-only"}
        if raw_visibility is not None:
            if not isinstance(raw_visibility, Mapping) or any(
                not isinstance(value, str) for value in raw_visibility.values()
            ):
                raise PolicyAnalysisError(
                    "feature_visibility must map fields to classes"
                )
            visibility_field_counts.update(raw_visibility.values())
            present.update(raw_visibility.values())
        # The certificate and all three support views are explicitly oracle-only.
        visibility_field_counts["oracle-only"] += 4
        visibility_state_counts.update(present)
    visibility_summary = [
        {
            "visibility_class": visibility,
            "unsafe_states_with_class": visibility_state_counts[visibility],
            "unsafe_state_denominator": len(unsafe),
            "recorded_field_occurrences": visibility_field_counts[visibility],
        }
        for visibility in (
            "L1-local-dynamic",
            "L1-static-boundary",
            "temporal-neighbor-dynamic",
            "nonlocal-yoke-dynamic",
            "oracle-only",
            "posthoc-ground-truth",
        )
    ]
    return {
        "context_views": context_tables,
        "visibility_summary": visibility_summary,
    }


def _workload_tables(prepared: _PreparedInputs) -> dict[str, Any]:
    """Builds the event/transaction summary and residual-HW distributions."""

    shots = prepared.shots
    cell_id = prepared.cell_id
    workload_replicates = prepared.workload_replicates
    workload_seed = prepared.workload_seed
    arm_order = ("shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial")
    workload_matrix = np.empty((len(shots), 2 * len(arm_order)), dtype=np.float64)
    for row_index, shot in enumerate(shots):
        for arm_index, arm in enumerate(arm_order):
            workload_matrix[row_index, 2 * arm_index] = shot["final_hw"][arm]
            workload_matrix[row_index, 2 * arm_index + 1] = shot["original_hw"]
    workload_intervals = clustered_bootstrap_ratios(
        workload_matrix,
        replicates=workload_replicates,
        seed=derive_bootstrap_seed(workload_seed, cell_id=cell_id, estimand="R-event"),
    )
    original_total = sum(shot["original_hw"] for shot in shots)
    shadow_relief = original_total - sum(shot["final_hw"]["shadow"] for shot in shots)
    event_summary = []
    for arm_index, arm in enumerate(arm_order):
        final_total = sum(shot["final_hw"][arm] for shot in shots)
        relief = original_total - final_total
        provisional = _sum_optional(
            (shot["provisional_removed"][arm] for shot in shots),
            name="provisional_events_removed",
        )
        rollback_lost = _sum_optional(
            (shot["rollback_lost"][arm] for shot in shots),
            name="events_lost_to_rollback",
        )
        if (
            provisional is not None
            and rollback_lost is not None
            and provisional - rollback_lost != relief
        ):
            raise PolicyAnalysisError(
                f"{arm} aggregate transaction/event accounting disagrees"
            )
        event_summary.append(
            {
                "arm": arm,
                "shots": len(shots),
                "sum_original_detector_hw": original_total,
                "sum_final_residual_detector_hw": final_total,
                "R_event": None
                if original_total == 0
                else final_total / original_total,
                "R_event_bootstrap": workload_intervals[arm_index],
                "durable_events_removed": relief,
                "provisional_events_removed": provisional,
                "events_lost_to_rollback": rollback_lost,
                "transaction_telemetry_complete": (
                    provisional is not None and rollback_lost is not None
                ),
                "R_relief_retained_relative_to_shadow": (
                    None if shadow_relief == 0 else relief / shadow_relief
                ),
            }
        )
    residual_hw_distributions = {
        arm: distribution_summary([shot["final_hw"][arm] for shot in shots])
        for arm in arm_order
    }
    return {
        "event_and_transaction_summary": event_summary,
        "residual_hw_distributions": residual_hw_distributions,
    }


def _association_tables(prepared: _PreparedInputs) -> dict[str, Any]:
    """Builds the unsafe-count and first-conflict association tables."""

    shots = prepared.shots
    cell_id = prepared.cell_id
    shot_keys = prepared.shot_keys
    shot_by_key = prepared.shot_by_key
    states_by_shot = prepared.states_by_shot
    proposal_replicates = prepared.proposal_replicates
    proposal_seed = prepared.proposal_seed
    association = []
    for key in shot_keys:
        shot = shot_by_key[key]
        ordered_states = states_by_shot[key]
        first = ordered_states[0] if ordered_states else None
        association.append(
            {
                "cell_id": key[0],
                "global_shot_id": key[1],
                "unsafe_count_bin": _unsafe_bin(len(ordered_states)),
                "first_unsafe_stage": None
                if first is None
                else first["original_stage"],
                "first_unsafe_context": None
                if first is None
                else first["exclusive_context"],
                "terminal_action": None if first is None else first["terminal_action"],
                "paired_outcome": _outcome_name(shot),
                "prediction_disagreement": shot["predictions"]["u0"]
                != shot["predictions"]["shadow"],
            }
        )
    association_by_unsafe_count = []
    for bin_name in ("0", "1", "2", "3", "4+"):
        group = [row for row in association if row["unsafe_count_bin"] == bin_name]
        counts = Counter(row["paired_outcome"] for row in group)
        association_by_unsafe_count.append(
            {
                "unsafe_count_bin": bin_name,
                "shots": len(group),
                "prediction_agreement": counts["prediction-agreement"],
                "regression": counts["regression"],
                "recovery": counts["recovery"],
                "prediction_discordant_both_wrong": counts[
                    "prediction-discordant-both-wrong"
                ],
                "prediction_disagreement": sum(
                    row["prediction_disagreement"] for row in group
                ),
            }
        )
    unsafe_counts = {key: len(states_by_shot[key]) for key in shot_keys}
    unsafe_count_values = sorted(set(unsafe_counts.values()))
    unsafe_count_categories = [str(value) for value in unsafe_count_values]
    unsafe_count_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=unsafe_count_categories,
        observations=[
            (key, str(candidate), count == candidate)
            for key, count in unsafe_counts.items()
            for candidate in unsafe_count_values
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="full-unsafe-count-distribution",
    )
    unsafe_count_interval = {row["category"]: row for row in unsafe_count_intervals}
    unsafe_count_distribution = [
        {
            "unsafe_count": count,
            "shots": sum(value == count for value in unsafe_counts.values()),
            "denominator": len(shots),
            "fraction": unsafe_count_interval[str(count)]["fraction"],
            "bootstrap": unsafe_count_interval[str(count)]["bootstrap"],
        }
        for count in unsafe_count_values
    ]
    association_by_first = {}
    for field in ("first_unsafe_stage", "first_unsafe_context", "terminal_action"):
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in association:
            grouped[row[field]].append(row)
        association_by_first[field] = [
            {
                field: value,
                "shots": len(group),
                **dict(Counter(row["paired_outcome"] for row in group)),
                "prediction_disagreement": sum(
                    row["prediction_disagreement"] for row in group
                ),
            }
            for value, group in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
    first_conflict_discordant = _count_table(
        (
            {
                "first_unsafe_stage": row["first_unsafe_stage"],
                "first_unsafe_context": row["first_unsafe_context"] or "none",
            }
            for row in association
            if row["prediction_disagreement"]
        ),
        keys=("first_unsafe_stage", "first_unsafe_context"),
    )
    if sum(row["count"] for row in first_conflict_discordant) != sum(
        row["prediction_disagreement"] for row in association
    ):
        raise PolicyAnalysisError("first-conflict context totals do not reconcile")
    return {
        "association_by_unsafe_count": association_by_unsafe_count,
        "unsafe_count_distribution": unsafe_count_distribution,
        "association_by_first": association_by_first,
        "first_conflict_discordant": first_conflict_discordant,
    }


def _distribution_tables(prepared: _PreparedInputs) -> dict[str, Any]:
    """Builds the continuous-distribution, competitor, ECDF, and risk tables."""

    proposals = prepared.proposals
    states = prepared.states
    shot_keys = prepared.shot_keys
    cell_id = prepared.cell_id
    proposal_replicates = prepared.proposal_replicates
    proposal_seed = prepared.proposal_seed
    distributions = {
        "cost_excess": distribution_summary(
            [row["cost_excess"] for row in proposals if row["cost_excess"] is not None]
        ),
        "local_weight_margin": distribution_summary(
            [
                row["local_weight_margin"]
                for row in proposals
                if row["local_weight_margin"] is not None
            ]
        ),
        "veto_chain_length": distribution_summary(
            [state["veto_chain_length"] for state in states]
        ),
        "candidate_count": distribution_summary(
            [
                state["candidate_count"]
                for state in states
                if state["candidate_count"] is not None
            ]
        ),
        "events_removed": distribution_summary(
            [
                row["events_removed"]
                for row in proposals
                if row["events_removed"] is not None
            ]
        ),
        "cost_tolerance_tau_k": distribution_summary(
            [row["tau_k"] for row in proposals if row["tau_k"] is not None]
        ),
    }
    competitor_intervals = _bootstrap_fraction_table(
        shot_keys=shot_keys,
        categories=["available", "unavailable"],
        observations=[
            (
                (row["cell_id"], row["global_shot_id"]),
                category,
                row["same_stage_competitor_exists"] is (category == "available"),
            )
            for row in proposals
            for category in ("available", "unavailable")
        ],
        replicates=proposal_replicates,
        seed_root=proposal_seed,
        cell_id=cell_id,
        estimand="same-stage-competitor-availability",
    )
    competitor_summary = {
        "shadow_commitments": len(proposals),
        "available": sum(
            row["same_stage_competitor_exists"] is True for row in proposals
        ),
        "unavailable": sum(
            row["same_stage_competitor_exists"] is False for row in proposals
        ),
        "unrecorded": 0,
        "margin_denominator": sum(
            row["local_weight_margin"] is not None for row in proposals
        ),
        "availability_fractions": [
            {
                "availability": row["category"],
                "count": row["numerator"],
                "denominator": row["denominator"],
                "fraction": row["fraction"],
                "bootstrap": row["bootstrap"],
            }
            for row in competitor_intervals
        ],
    }
    cost_ecdf_by_stage = {
        str(stage): exact_ecdf(
            [
                row["cost_excess"]
                for row in proposals
                if row["stage"] == stage and row["cost_excess"] is not None
            ]
        )
        for stage in range(1, 5)
    }
    cost_ecdf_by_certificate = {
        certificate: exact_ecdf(
            [
                row["cost_excess"]
                for row in proposals
                if row["certificate_class"] == certificate
                and row["cost_excess"] is not None
            ]
        )
        for certificate in CERTIFICATE_CLASSES
    }
    cost_ecdf_by_context = {
        label: exact_ecdf(
            [
                row["cost_excess"]
                for row in proposals
                if row["exclusive_support_component_context"] == context_value
                and row["cost_excess"] is not None
            ]
        )
        for label in (*CONTEXT_PRIORITY, "none")
        for context_value in [None if label == "none" else label]
    }
    if sum(
        points[-1]["denominator"] for points in cost_ecdf_by_context.values() if points
    ) != sum(row["cost_excess"] is not None for row in proposals):
        raise PolicyAnalysisError("cost ECDF context totals do not reconcile")
    original_vs_alternative = [
        {
            key: state[key]
            for key in (
                "state_id",
                "original_stage",
                "first_safe_stage",
                "original_decision_weight",
                "first_safe_decision_weight",
                "original_path_length",
                "first_safe_path_length",
                "original_weight_margin",
                "first_safe_weight_margin",
                "original_events_removed",
                "first_safe_events_removed",
            )
        }
        for state in states
    ]
    margins = sorted(
        row["local_weight_margin"]
        for row in proposals
        if row["local_weight_margin"] is not None
    )
    margin_edges = (
        [empirical_type7(margins, q / 10) for q in range(11)] if margins else []
    )
    risk_rows = []
    for row in proposals:
        copy = dict(row)
        margin = row["local_weight_margin"]
        if margin is None or not margin_edges:
            copy["margin_decile"] = None
        else:
            copy["margin_decile"] = min(
                10,
                1 + sum(float(margin) > float(edge) for edge in margin_edges[1:-1]),
            )
        risk_rows.append(copy)
    risk_heatmaps = {
        "stage_by_window_offset": _risk_count_table(
            risk_rows, keys=("stage", "window_offset")
        ),
        "stage_by_static_boundary_competition": _risk_count_table(
            risk_rows, keys=("stage", "static_boundary_competition")
        ),
        "domain_hw_by_candidate_multiplicity": _risk_count_table(
            risk_rows, keys=("domain_hw", "candidate_multiplicity")
        ),
        "stage_by_margin_decile": _risk_count_table(
            risk_rows, keys=("stage", "margin_decile")
        ),
        "margin_decile_edges_type7": margin_edges,
        "threshold_use_permitted": False,
    }
    return {
        "continuous_distributions": distributions,
        "local_competitor_summary": competitor_summary,
        "cost_excess_ecdf_by_stage": cost_ecdf_by_stage,
        "cost_excess_ecdf_by_certificate": cost_ecdf_by_certificate,
        "cost_excess_ecdf_by_context": cost_ecdf_by_context,
        "original_vs_alternative": original_vs_alternative,
        "risk_heatmaps": risk_heatmaps,
    }


def _tail_tables(
    corpus: PolicyAuditCorpus, prepared: _PreparedInputs
) -> tuple[dict[str, Any], set[str]]:
    """Builds the veto-chain tail table; also returns the incomplete metrics."""

    states = prepared.states
    shot_keys = prepared.shot_keys
    tail_metrics: dict[str, list[float]] = {
        "proposals_per_state": [float(len(state["rows"])) for state in states],
        "vetoes_per_state": [float(state["veto_chain_length"]) for state in states],
    }
    incomplete_tail_metrics: set[str] = set()
    for output_name, field_names in {
        "oracle_calls_per_state": (
            "state_oracle_call_count",
            "oracle_calls",
            "state_oracle_calls",
        ),
        "stage3_enumeration_ns_per_state": (
            "state_total_stage3_enumeration_wall_ns",
            "stage3_enumeration_ns",
            "state_stage3_enumeration_ns",
        ),
    }.items():
        values = []
        complete = True
        for state in states:
            value = _at(state["rows"][0], *field_names)
            if value is None:
                complete = False
                break
            values.append(float(_as_nonnegative_int(value, name=output_name)))
        if not complete:
            mode = corpus.manifest.get("mode")
            if mode != "smoke":
                raise PolicyAnalysisError(
                    f"{mode} corpus lacks required {output_name} tail telemetry"
                )
            incomplete_tail_metrics.add(output_name)
        tail_metrics[output_name] = values if complete else []
    for metric in tuple(tail_metrics):
        if not metric.endswith("_per_state"):
            continue
        shot_values: dict[tuple[str, int], float] = defaultdict(float)
        complete = True
        for state_index, state in enumerate(states):
            values = tail_metrics[metric]
            if state_index >= len(values):
                complete = False
                break
            shot_values[(state["cell_id"], state["global_shot_id"])] += values[
                state_index
            ]
        tail_metrics[metric.removesuffix("_per_state") + "_per_shot"] = (
            [shot_values[key] for key in shot_keys] if complete else []
        )
        if not complete:
            incomplete_tail_metrics.add(metric.removesuffix("_per_state") + "_per_shot")
    veto_chain_tails = {
        name: {
            **distribution_summary(values),
            "telemetry_status": (
                "incomplete-smoke" if name in incomplete_tail_metrics else "complete"
            ),
        }
        for name, values in tail_metrics.items()
    }
    return {"veto_chain_tails": veto_chain_tails}, incomplete_tail_metrics


def _gate_and_checkpoint_tables(
    corpus: PolicyAuditCorpus,
    prepared: _PreparedInputs,
    tables: Mapping[str, Any],
) -> dict[str, Any]:
    """Builds the fatal-gate and interpretation-checkpoint tables."""

    states = prepared.states
    discordant = prepared.discordant
    paired = tables["paired_outcomes"]
    event_summary = tables["event_and_transaction_summary"]
    recomputed_gate_evidence: dict[int, str] = {
        1: "loader recomputed exact mode schedule, shot IDs, worker ranges, and row counts",
        2: "loader reconciled every sidecar row to one authenticated physical-shot provenance",
        5: "analyzer compared exact U0, O-frame-transactional, and O-frame-partial prediction tokens",
        6: "analyzer recomputed certificate classes from exact-hex cost excess, tolerance, and frame compatibility",
        10: "analyzer reconciled counterfactual rank one by proposal digest and canonical proposal signature to the durable shadow original",
        11: "analyzer recomputed uncapped unchanged-state ranks, signatures, decisions, stage monotonicity, first-safe stop, and true exhaustion",
        12: "analyzer reconciled aggregate provisional, rollback, durable, and residual event accounting",
        13: "analyzer reconciled every exact U0/PU prediction disagreement to a durable frame conflict",
        15: "analyzer validated the closed context vocabulary and reconciled omitted-context and degeneracy diagnostics",
        17: "loader authenticated canonical gzip/JSONL bytes, digests, schemas, identities, and COLLECTION_READY",
    }
    collector_attestations = _collector_gate_attestations(corpus.manifest)
    gates = [
        {
            "gate": gate,
            "status": (
                "passed-ledger-recomputed"
                if gate in recomputed_gate_evidence
                else (
                    "collector-attested"
                    if gate in collector_attestations
                    else "missing-collector-attestation"
                )
            ),
            "evidence": {
                "kind": (
                    "ledger-recomputation"
                    if gate in recomputed_gate_evidence
                    else (
                        "authenticated-collector-attestation"
                        if gate in collector_attestations
                        else "missing"
                    )
                ),
                "claim": (
                    recomputed_gate_evidence[gate]
                    if gate in recomputed_gate_evidence
                    else collector_attestations.get(gate)
                ),
                "authenticated_by": (
                    ["canonical-ledgers", "manifest.json", "COLLECTION_READY"]
                    if gate in recomputed_gate_evidence
                    else (
                        ["manifest.json", "COLLECTION_READY"]
                        if gate in collector_attestations
                        else []
                    )
                ),
            },
        }
        for gate in range(1, 19)
    ]
    interpretation_checkpoints = [
        {
            "checkpoint": "few-final-disagreements",
            "fatal": False,
            "observed": len(discordant),
        },
        {
            "checkpoint": "little-certified-event-relief",
            "fatal": False,
            "observed_shadow_R_event": next(
                row["R_event"] for row in event_summary if row["arm"] == "shadow"
            ),
        },
        {
            "checkpoint": "many-true-abstentions",
            "fatal": False,
            "observed": sum(
                state["terminal_action"] == "abstain-true-exhaustion"
                for state in states
            ),
        },
        {
            "checkpoint": "nonlocal-yoke-context-dominates",
            "fatal": False,
            "observed": sum(state["exclusive_context"] == "yoke" for state in states),
        },
        {
            "checkpoint": "shadow-effect-magnitude",
            "fatal": False,
            "observed_paired_risk_difference": paired["u0_vs_shadow"][
                "paired_risk_difference"
            ],
        },
    ]
    return {
        "fatal_gates": gates,
        "interpretation_checkpoints": interpretation_checkpoints,
    }


def analyze_policy_audit(corpus: PolicyAuditCorpus) -> dict[str, Any]:
    """Validates and analyzes a frozen B1 corpus without reconstructing decoding."""

    prepared = _prepare_inputs(corpus)
    # Assemble the same frozen tables as the pre-split inline construction,
    # one family builder per group of the original ``tables`` dict literal.
    tables: dict[str, Any] = {}
    tables.update(_overview_and_paired_tables(corpus, prepared))
    tables.update(_workload_tables(prepared))
    tables["domain_terminal_summary"] = _domain_table(corpus.rows["domains"])
    tables.update(_certificate_tables(prepared))
    tables.update(_counterfactual_tables(prepared))
    tables.update(_context_view_tables(prepared))
    tables.update(_association_tables(prepared))
    tables.update(_distribution_tables(prepared))
    tail_tables, incomplete_tail_metrics = _tail_tables(corpus, prepared)
    tables.update(tail_tables)
    tables.update(_gate_and_checkpoint_tables(corpus, prepared, tables))
    casebook = select_casebook(prepared.states)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "experiment_id": _at(corpus.experiment, "experiment_id", required=True),
        "cell_id": prepared.cell_id,
        "analysis_contract": {
            "source": "immutable-canonical-gzip-jsonl-only",
            "sampling_or_decoding_reconstruction": False,
            "bootstrap_unit": "complete-physical-shot",
            "bootstrap_quantile": "empirical-type-7",
            "proposal_bootstrap_replicates": prepared.proposal_replicates,
            "workload_bootstrap_replicates": prepared.workload_replicates,
            "casebook_outcome_blind": True,
            "casebook_exhaustive_rows_excluded": True,
            "support_context_views_kept_distinct": True,
            "required_tail_telemetry": (
                "complete" if not incomplete_tail_metrics else "incomplete-smoke-only"
            ),
            "complete_written_by_analyzer": False,
            "complete_accepted_as_analysis_substitute": False,
            "next_required_stage": "casebook-expansion-and-finalization-external",
        },
        "source_hashes": corpus.source_hashes,
        "tables": tables,
        "casebook_selection": casebook,
    }
    result["analysis_sha256"] = _sha256(canonical_json_bytes(result))
    return result
