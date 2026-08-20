"""Deterministic, outcome-blind casebook selection.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` selects casebook
states from frozen oracle/policy fields only, never from actual observables
or correctness labels.  It inherits the package's downstream-only contract:
it never imports circuit generation, sampling, matching, or decoding code.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ._contract import (
    CASEBOOK_MIN_STATES,
    CASEBOOK_SCHEMA,
    TERMINAL_ACTIONS,
    PolicyAnalysisError,
    _sha256,
    canonical_json_bytes,
)
from ._stats import empirical_type7


def _casebook_rank_digest(state_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", state_id):
        raise PolicyAnalysisError(
            "casebook state identity must be its canonical SHA-256 digest"
        )
    return state_id


def select_casebook(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Selects outcome-blind states using only frozen oracle/policy fields."""

    selected: dict[str, dict[str, Any]] = {}
    strata: dict[tuple[str | None, int], list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        strata[(state.get("exclusive_context"), int(state["original_stage"]))].append(
            state
        )
    audit_rows = []
    for (context, stage), group in sorted(
        strata.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        slot = f"context={context}|stage={stage}"
        if len(group) < CASEBOOK_MIN_STATES:
            audit_rows.append(
                {"slot": slot, "eligible": len(group), "selected_state_id": None}
            )
            continue
        values = [
            float(state["original_cost_excess"])
            for state in group
            if state.get("original_cost_excess") is not None
        ]
        if len(values) != len(group):
            raise PolicyAnalysisError(
                "casebook cost-excess stratum contains missing values"
            )
        median = empirical_type7(values, 0.5)
        assert median is not None
        chosen = min(
            group,
            key=lambda state: (
                abs(float(state["original_cost_excess"]) - median),
                _casebook_rank_digest(str(state["state_id"])),
            ),
        )
        selected[str(chosen["state_id"])] = {
            "state_id": chosen["state_id"],
            "selection_reasons": [slot],
            "original_proposal_sha256": chosen["original_proposal_sha256"],
        }
        audit_rows.append(
            {
                "slot": slot,
                "eligible": len(group),
                "median_metric": median,
                "selected_state_id": chosen["state_id"],
            }
        )

    for action in TERMINAL_ACTIONS[:3]:
        group = [state for state in states if state.get("terminal_action") == action]
        slot = f"terminal_action={action}"
        if not group:
            audit_rows.append({"slot": slot, "eligible": 0, "selected_state_id": None})
            continue
        median = empirical_type7(
            [float(state["veto_chain_length"]) for state in group], 0.5
        )
        assert median is not None
        chosen = min(
            group,
            key=lambda state: (
                abs(float(state["veto_chain_length"]) - median),
                _casebook_rank_digest(str(state["state_id"])),
            ),
        )
        state_id = str(chosen["state_id"])
        if state_id in selected:
            selected[state_id]["selection_reasons"].append(slot)
        else:
            selected[state_id] = {
                "state_id": state_id,
                "selection_reasons": [slot],
                "original_proposal_sha256": chosen["original_proposal_sha256"],
            }
        audit_rows.append(
            {
                "slot": slot,
                "eligible": len(group),
                "median_metric": median,
                "selected_state_id": state_id,
            }
        )
    result = {
        "schema": CASEBOOK_SCHEMA,
        "selection_uses_actual_observables_or_correctness": False,
        "min_context_stage_states": CASEBOOK_MIN_STATES,
        "states": [
            {**row, "selection_reasons": sorted(row["selection_reasons"])}
            for _, row in sorted(selected.items())
        ],
        "selection_audit": audit_rows,
    }
    result["selection_sha256"] = _sha256(canonical_json_bytes(result))
    return result
