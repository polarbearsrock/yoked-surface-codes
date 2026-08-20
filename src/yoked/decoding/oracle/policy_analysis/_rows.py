"""Row normalization and context validation for the immutable ledgers.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` reconciles shot,
proposal, counterfactual, and domain rows into normalized records: semantic
arm roles, oracle certificate classes, the closed support-context vocabulary,
degeneracy diagnostics, and the uncapped counterfactual state chains.  It
inherits the package's downstream-only contract: it never imports circuit
generation, sampling, matching, or decoding code.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ._contract import (
    CONTEXT_LABELS,
    CONTEXT_PRIORITY,
    DEGENERACY_DIAGNOSTICS,
    TERMINAL_ACTIONS,
    PolicyAnalysisError,
    _sha256,
    canonical_json_bytes,
)
from ._corpus import _identity
from ._fields import (
    _as_bool,
    _as_nonnegative_int,
    _at,
    _float_value,
    _required_float,
)


def _arm_role(arm_id: str) -> str:
    value = arm_id.lower()
    if "u0-joint" in value or value in {"u0", "u0-joint-y2"}:
        return "u0"
    if "shadow" in value or "legacy" in value or "v3" in value:
        return "shadow"
    if "ocost" in value or "o-cost" in value:
        return "o-cost-tx"
    if ("oframe" in value or "o-frame" in value) and "partial" in value:
        return "o-frame-partial"
    if "oframe" in value or "o-frame" in value:
        return "o-frame-tx"
    return arm_id


def _arm_results(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    predictions = row.get("arm_predictions_hex")
    failures = row.get("arm_failures")
    if isinstance(predictions, Mapping) and isinstance(failures, Mapping):
        if set(predictions) != set(failures):
            raise PolicyAnalysisError("shot prediction/failure arm sets differ")
        summaries: Mapping[str, Any] = {}
        for key in (
            "arm_summaries",
            "arm_workload",
            "arm_results",
            "arms",
            "trajectories",
        ):
            candidate = row.get(key)
            if isinstance(candidate, Mapping):
                summaries = candidate
                break
        result: dict[str, Mapping[str, Any]] = {}
        for arm_id in sorted(predictions):
            role = _arm_role(str(arm_id))
            if role in result:
                raise PolicyAnalysisError(
                    f"shot contains duplicate semantic arm {role!r}"
                )
            summary = summaries.get(arm_id, summaries.get(role, {}))
            if not isinstance(summary, Mapping):
                raise PolicyAnalysisError(
                    f"arm summary for {arm_id!r} must be an object"
                )
            result[role] = {
                **summary,
                "prediction_hex": predictions[arm_id],
                "logical_failure": failures[arm_id],
                "source_arm_id": arm_id,
            }
        return result
    for key in ("arms", "arm_results", "decoder_results", "results", "trajectories"):
        raw = row.get(key)
        if isinstance(raw, Mapping) and all(
            isinstance(value, Mapping) for value in raw.values()
        ):
            result: dict[str, Mapping[str, Any]] = {}
            for arm_id, value in raw.items():
                role = _arm_role(str(arm_id))
                if role in result:
                    raise PolicyAnalysisError(
                        f"shot contains duplicate semantic arm {role!r}"
                    )
                result[role] = value
            return result
    arm_id = row.get("arm_id")
    if isinstance(arm_id, str):
        return {_arm_role(arm_id): row}
    raise PolicyAnalysisError("shot row has no arm result mapping")


def _failure(result: Mapping[str, Any]) -> bool:
    value = _at(
        result,
        "logical_failure",
        "failed",
        "any_observable_failure",
        "posthoc.logical_failure",
        "posthoc_ground_truth.logical_failure",
    )
    return _as_bool(value, name="logical_failure")


def _prediction_token(result: Mapping[str, Any]) -> str | None:
    value = _at(result, "prediction_hex", "prediction", "logical_prediction_hex")
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyAnalysisError("prediction token must be a string")
    return value


def _original_hw(row: Mapping[str, Any]) -> int:
    return _as_nonnegative_int(
        _at(
            row,
            "original_detector_hw",
            "original_detector_events",
            "global_detector_hw",
            "detector_hw",
            "workload.original_detector_events",
            required=True,
        ),
        name="original detector HW",
    )


def _sorted_labels(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PolicyAnalysisError(f"{name} must be an array of strings")
    normalized = tuple(sorted(set(value)))
    if list(normalized) != value:
        raise PolicyAnalysisError(f"{name} must be sorted and unique")
    return normalized


def _component_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    def canonical_edge_ids(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(type(edge_id) is int and edge_id >= 0 for edge_id in value)
            and value == sorted(set(value))
        )

    direct_labels = _sorted_labels(
        _at(row, "support_difference_component_labels", required=True),
        name="support_difference_component_labels",
    )
    components = _at(row, "support_difference_components", required=True)
    if not isinstance(components, list):
        raise PolicyAnalysisError("support_difference_components must be an array")
    if _at(row, "support_difference_representation_version", required=True) != (
        "promatch-support-difference-v2"
    ):
        raise PolicyAnalysisError("unsupported support-difference representation")
    labels: set[str] = set()
    if not canonical_edge_ids(_at(row, "detector_boundary_ids", required=True)):
        raise PolicyAnalysisError(
            "detector_boundary_ids is not a canonical detector set"
        )
    real_edge_union: set[int] = set()
    cancellation_union: set[int] = set()
    saw_disconnected = False
    real_components: list[Mapping[str, Any]] = []
    cancellation_components: list[Mapping[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping):
            raise PolicyAnalysisError("support difference component must be an object")
        if set(component) != {
            "certificate_kind",
            "canonical_edge_ids",
            "support_cancellation_edge_ids",
            "component_detector_ids",
            "candidate_support_witness_edge_ids",
            "candidate_boundary_witness_detector_ids",
            "labels",
            "candidate_relevant",
            "candidate_relevance_reasons",
        }:
            raise PolicyAnalysisError(
                "support difference component fields are not exact"
            )
        tags = _at(component, "labels", "tags", required=True)
        component_labels = _sorted_labels(
            tags, name="support difference component tags"
        )
        if set(component_labels) - CONTEXT_LABELS:
            raise PolicyAnalysisError(
                "support difference component contains unknown labels"
            )
        if "in-domain" in component_labels and len(component_labels) != 1:
            raise PolicyAnalysisError(
                "in-domain is not exclusive in support component labels"
            )
        kind = _at(component, "certificate_kind", required=True)
        relevant = _as_bool(
            _at(component, "candidate_relevant", required=True),
            name="support component candidate_relevant",
        )
        reasons = _at(component, "candidate_relevance_reasons", required=True)
        if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
            raise PolicyAnalysisError(
                "support component relevance reasons are not canonical"
            )
        if set(reasons) - {
            "candidate-support-edge",
            "candidate-boundary-detector",
            "candidate-residual-support-cancellation",
        }:
            raise PolicyAnalysisError(
                "support component has an unknown relevance reason"
            )
        edge_ids = _at(component, "canonical_edge_ids", required=True)
        cancellation_ids = _at(
            component, "support_cancellation_edge_ids", required=True
        )
        detector_ids = _at(component, "component_detector_ids", required=True)
        support_witness = _at(
            component, "candidate_support_witness_edge_ids", required=True
        )
        boundary_witness = _at(
            component, "candidate_boundary_witness_detector_ids", required=True
        )
        if not all(
            canonical_edge_ids(value)
            for value in (
                edge_ids,
                cancellation_ids,
                detector_ids,
                support_witness,
                boundary_witness,
            )
        ):
            raise PolicyAnalysisError(
                "support component edge IDs are not canonical sets"
            )
        if kind == "real-x-component":
            candidate_support = _at(row, "P_candidate_support_edge_ids", required=True)
            detector_boundary = _at(row, "detector_boundary_ids", required=True)
            expected_support_witness = sorted(
                set(edge_ids).intersection(candidate_support)
            )
            expected_boundary_witness = sorted(
                set(detector_ids).intersection(detector_boundary)
            )
            expected_reasons = sorted(
                (["candidate-support-edge"] if expected_support_witness else [])
                + (["candidate-boundary-detector"] if expected_boundary_witness else [])
            )
            if (
                not edge_ids
                or cancellation_ids
                or "support-cancellation" in component_labels
            ):
                raise PolicyAnalysisError(
                    "real X component has invalid certificate support"
                )
            if (
                support_witness != expected_support_witness
                or boundary_witness != expected_boundary_witness
                or reasons != expected_reasons
                or bool(reasons) != relevant
            ):
                raise PolicyAnalysisError(
                    "real X component relevance disagrees with its reasons"
                )
            if real_edge_union.intersection(edge_ids):
                raise PolicyAnalysisError("real X components overlap")
            real_edge_union.update(edge_ids)
            saw_disconnected |= not relevant
            real_components.append(component)
        elif kind == "support-cancellation":
            if (
                edge_ids
                or not cancellation_ids
                or detector_ids
                or support_witness
                or boundary_witness
                or not relevant
                or reasons != ["candidate-residual-support-cancellation"]
                or "support-cancellation" not in component_labels
            ):
                raise PolicyAnalysisError(
                    "support-cancellation certificate is malformed"
                )
            if cancellation_union:
                raise PolicyAnalysisError("multiple support-cancellation certificates")
            cancellation_union.update(cancellation_ids)
            cancellation_components.append(component)
        else:
            raise PolicyAnalysisError(f"unknown support certificate kind {kind!r}")
        if relevant:
            labels.update(component_labels)
    if (
        components
        != sorted(
            real_components, key=lambda component: component["canonical_edge_ids"]
        )
        + cancellation_components
    ):
        raise PolicyAnalysisError("support certificates are not in canonical order")
    supports = {}
    for field in (
        "B_base_support_edge_ids",
        "P_candidate_support_edge_ids",
        "R_residual_support_edge_ids",
        "Q_forced_parity_support_edge_ids",
        "X_support_difference_edge_ids",
        "P_intersection_R_edge_ids",
    ):
        value = _at(row, field, required=True)
        if not canonical_edge_ids(value):
            raise PolicyAnalysisError(f"{field} is not a canonical square-free support")
        supports[field] = value
    b = set(supports["B_base_support_edge_ids"])
    p = set(supports["P_candidate_support_edge_ids"])
    r = set(supports["R_residual_support_edge_ids"])
    for alias, canonical in (
        ("base_support_edge_ids", supports["B_base_support_edge_ids"]),
        ("candidate_support_edge_ids", supports["P_candidate_support_edge_ids"]),
        ("residual_support_edge_ids", supports["R_residual_support_edge_ids"]),
    ):
        if _at(row, alias, required=True) != canonical:
            raise PolicyAnalysisError(f"{alias} disagrees with its named B/P/R support")
    if supports["Q_forced_parity_support_edge_ids"] != sorted(p ^ r):
        raise PolicyAnalysisError("Q support does not reconcile P xor R")
    if supports["X_support_difference_edge_ids"] != sorted(b ^ p ^ r):
        raise PolicyAnalysisError("X support does not reconcile B xor P xor R")
    if supports["P_intersection_R_edge_ids"] != sorted(p & r):
        raise PolicyAnalysisError(
            "cancellation support does not reconcile P intersection R"
        )
    for flag in (
        "supports_square_free",
        "B_base_support_square_free",
        "P_candidate_support_square_free",
        "R_residual_support_square_free",
        "Q_forced_parity_support_square_free",
        "X_support_difference_square_free",
    ):
        if _at(row, flag, required=True) is not True:
            raise PolicyAnalysisError(f"{flag} must be true")
    x_support = supports["X_support_difference_edge_ids"]
    cancellation_support = supports["P_intersection_R_edge_ids"]
    top_cancellation = _at(row, "support_cancellation_edge_ids", required=True)
    if top_cancellation != cancellation_support:
        raise PolicyAnalysisError("top-level cancellation support is inconsistent")
    if sorted(real_edge_union) != x_support:
        raise PolicyAnalysisError("real support components do not partition X exactly")
    if sorted(cancellation_union) != cancellation_support:
        raise PolicyAnalysisError(
            "cancellation certificates do not reconcile P intersection R"
        )
    disconnected_flag = _as_bool(
        _at(row, "disconnected_support_reconfiguration", required=True),
        name="disconnected_support_reconfiguration",
    )
    if disconnected_flag != saw_disconnected:
        raise PolicyAnalysisError(
            "disconnected support-reconfiguration flag is inconsistent"
        )
    if labels - {"in-domain"}:
        labels.discard("in-domain")
    component_labels = tuple(sorted(labels))
    if direct_labels != component_labels:
        raise PolicyAnalysisError(
            "support-difference component labels disagree with their component union"
        )
    return component_labels


def _validate_context_labels(labels: tuple[str, ...], *, name: str) -> None:
    unknown = set(labels) - CONTEXT_LABELS
    if unknown:
        raise PolicyAnalysisError(f"{name} contains unknown labels: {sorted(unknown)}")
    if "in-domain" in labels and len(labels) != 1:
        raise PolicyAnalysisError(f"in-domain is not exclusive in {name}")


def _context(row: Mapping[str, Any]) -> dict[str, Any]:
    matched_raw = _at(
        row,
        "matched_partner_labels",
        "base_matched_partner_labels",
        "context.matched_partner_labels",
        required=True,
    )
    matched = _sorted_labels(
        matched_raw,
        name="matched_partner_labels",
    )
    support_path_raw = _at(
        row,
        "support_path_labels",
        "base_support_path_labels",
        "context.support_path_labels",
        required=True,
    )
    support_path = _sorted_labels(
        support_path_raw,
        name="support_path_labels",
    )
    difference = _component_labels(row)
    for name, labels in (
        ("matched_partner_labels", matched),
        ("support_path_labels", support_path),
        ("support_difference_component_labels", difference),
    ):
        _validate_context_labels(labels, name=name)
    omitted = _sorted_labels(
        _at(
            row,
            "omitted_context_labels",
            "context.omitted_context_labels",
            required=True,
        ),
        name="omitted_context_labels",
    )
    _validate_context_labels(omitted, name="omitted_context_labels")
    expected_omitted_set = set(matched) | set(support_path)
    if expected_omitted_set - {"in-domain"}:
        expected_omitted_set.discard("in-domain")
    expected_omitted = tuple(sorted(expected_omitted_set))
    if omitted != expected_omitted:
        raise PolicyAnalysisError(
            "omitted_context_labels disagrees with matched/support-path union"
        )
    degeneracy = _sorted_labels(
        _at(
            row,
            "degeneracy_diagnostics",
            "context.degeneracy_diagnostics",
            required=True,
        ),
        name="degeneracy_diagnostics",
    )
    unknown_degeneracy = set(degeneracy) - DEGENERACY_DIAGNOSTICS
    if unknown_degeneracy:
        raise PolicyAnalysisError(
            f"unknown degeneracy diagnostics: {sorted(unknown_degeneracy)}"
        )
    diagnostic_flags = {
        "same-pair-different-path-or-frame": _as_bool(
            _at(row, "same_pair_different_path_or_frame", required=True),
            name="same_pair_different_path_or_frame",
        ),
        "equal-weight-logical-class": _as_bool(
            _at(row, "equal_weight_logical_class", required=True),
            name="equal_weight_logical_class",
        ),
        "disconnected-support-reconfiguration": _as_bool(
            _at(row, "disconnected_support_reconfiguration", required=True),
            name="disconnected_support_reconfiguration",
        ),
        "unclassified": _as_bool(
            _at(row, "degeneracy_unclassified", required=True),
            name="degeneracy_unclassified",
        ),
    }
    expected_from_flags = tuple(
        sorted(label for label, present in diagnostic_flags.items() if present)
    )
    if degeneracy != expected_from_flags:
        raise PolicyAnalysisError(
            "degeneracy_diagnostics disagrees with its explicit diagnostic flags"
        )
    endpoints = _at(row, "ordered_endpoints", required=True)
    pairs = _at(row, "base_matched_active_pairs", required=True)
    base_support = _at(row, "base_support_edge_ids", required=True)
    candidate_support = _at(row, "candidate_support_edge_ids", required=True)
    base_frame = _at(row, "base_frame", required=True)
    candidate_frame = _at(row, "candidate_frame", required=True)
    if (
        not isinstance(endpoints, list)
        or not isinstance(pairs, list)
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
        or not isinstance(base_support, list)
        or not isinstance(candidate_support, list)
        or not isinstance(base_frame, str)
        or not isinstance(candidate_frame, str)
    ):
        raise PolicyAnalysisError("degeneracy source fields have invalid types")
    same_pair = (
        len(endpoints) == 2
        and endpoints[0] is not None
        and any(set(pair) == set(endpoints) for pair in pairs)
    )
    expected_same_pair_diagnostic = same_pair and (
        set(base_support) != set(candidate_support) or base_frame != candidate_frame
    )
    if (
        diagnostic_flags["same-pair-different-path-or-frame"]
        != expected_same_pair_diagnostic
    ):
        raise PolicyAnalysisError(
            "same-pair-different-path-or-frame is not structurally reconciled"
        )
    expected_equal_weight = _cost_compatible(row) and not _frame_compatible(row)
    if diagnostic_flags["equal-weight-logical-class"] != expected_equal_weight:
        raise PolicyAnalysisError(
            "equal-weight-logical-class disagrees with the oracle certificate"
        )
    oracle_accepts = _as_bool(
        _at(row, "oracle_policy_accepts", required=True), name="oracle_policy_accepts"
    )
    if oracle_accepts != (_cost_compatible(row) and _frame_compatible(row)):
        raise PolicyAnalysisError(
            "oracle_policy_accepts disagrees with cost/frame compatibility"
        )
    expected_unclassified = (
        not oracle_accepts
        and not diagnostic_flags["same-pair-different-path-or-frame"]
        and not diagnostic_flags["equal-weight-logical-class"]
        and not diagnostic_flags["disconnected-support-reconfiguration"]
        and not difference
    )
    if diagnostic_flags["unclassified"] != expected_unclassified:
        raise PolicyAnalysisError("unclassified degeneracy residual is not reconciled")
    exclusive = _at(
        row,
        "exclusive_support_component_context",
        "exclusive_context_label",
        "context.exclusive",
    )
    if exclusive is None:
        exclusive = next(
            (label for label in CONTEXT_PRIORITY if label in difference), None
        )
    if exclusive is not None and exclusive not in CONTEXT_PRIORITY:
        raise PolicyAnalysisError(f"unknown exclusive context label {exclusive!r}")
    if exclusive != next(
        (label for label in CONTEXT_PRIORITY if label in difference), None
    ):
        raise PolicyAnalysisError(
            "exclusive context label violates frozen display priority"
        )
    # These views are deliberately not unioned or aliased.  A downstream table
    # can compare them, but each keeps its own denominator and label set.
    return {
        "matched_partner_labels": matched,
        "support_path_labels": support_path,
        "support_difference_component_labels": difference,
        "exclusive_support_component_context": exclusive,
        "omitted_context_labels": omitted,
        "degeneracy_diagnostics": degeneracy,
    }


def _cost_compatible(row: Mapping[str, Any]) -> bool:
    explicit = _at(
        row, "cost_compatible", "evaluation.cost_compatible", "oracle.cost_compatible"
    )
    if explicit is not None:
        return _as_bool(explicit, name="cost_compatible")
    classification = _at(
        row,
        "cost_classification",
        "evaluation.cost_classification",
        "oracle.cost_classification",
        required=True,
    )
    if classification in {
        "numerically-cost-compatible",
        "cost-compatible",
        "compatible",
    }:
        return True
    if classification in {"positive-cost-excess", "positive-excess"}:
        return False
    raise PolicyAnalysisError(f"invalid/fatal cost classification {classification!r}")


def _frame_compatible(row: Mapping[str, Any]) -> bool:
    return _as_bool(
        _at(
            row,
            "frame_compatible",
            "evaluation.frame_compatible",
            "oracle.frame_compatible",
            required=True,
        ),
        name="frame_compatible",
    )


def certificate_class(row: Mapping[str, Any]) -> str:
    """Validates and returns the row's mutually exclusive oracle certificate."""

    cost = _cost_compatible(row)
    frame = _frame_compatible(row)
    excess = _required_float(row, "cost_excess")
    tolerance = _required_float(row, "tau_k")
    if tolerance < 0:
        raise PolicyAnalysisError("tau_k must be nonnegative")
    if cost != (excess <= tolerance):
        raise PolicyAnalysisError(
            "cost classification disagrees with cost_excess/tau_k"
        )
    if excess < -tolerance:
        raise PolicyAnalysisError("negative cost excess exceeds the frozen tolerance")
    if not cost:
        return "positive-cost-excess"
    if not frame:
        return "cost-compatible-frame-conflict"
    return "O-frame-safe"


def _origin(row: Mapping[str, Any]) -> str:
    value = _at(row, "trajectory_origin", required=True)
    if not isinstance(value, str):
        raise PolicyAnalysisError("trajectory_origin must be a string")
    return value


def _stage(row: Mapping[str, Any]) -> int:
    stage = _as_nonnegative_int(
        _at(row, "stage", "proposal.stage", required=True), name="stage"
    )
    if stage not in {1, 2, 3, 4}:
        raise PolicyAnalysisError("ProMatch stage must be in 1..4")
    return stage


def _domain_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _at(row, "patch_id", "domain.patch_id"),
        _at(row, "basis", "domain.basis", "domain.check_basis"),
        _at(row, "window_id", "domain.window_id"),
    )


def _state_key(row: Mapping[str, Any]) -> str:
    explicit = _at(
        row,
        "original_state_sha256",
        "counterfactual_state_sha256",
        "complete_pre_state_fingerprint",
        "pre_state_fingerprint",
    )
    if isinstance(explicit, str) and explicit:
        base = explicit
    else:
        raise PolicyAnalysisError("counterfactual/proposal row lacks state identity")
    cell, worker, shot = _identity(row)
    return _sha256(canonical_json_bytes([cell, worker, shot, _domain_key(row), base]))


def _terminal_action(row: Mapping[str, Any]) -> str | None:
    value = _at(row, "terminal_action", "counterfactual.terminal_action")
    if value is None:
        return None
    if value not in TERMINAL_ACTIONS:
        raise PolicyAnalysisError(f"invalid terminal action {value!r}")
    return str(value)


def _counterfactual_states(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        origin = _origin(row)
        if origin == "casebook-exhaustive":
            continue
        if origin != "shadow-original-state-counterfactual":
            raise PolicyAnalysisError(
                f"invalid all-shot counterfactual origin {origin!r}"
            )
        grouped[_state_key(row)].append(row)
    states = []
    for state_id, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: _as_nonnegative_int(
                _at(
                    row,
                    "operational_veto_chain_rank",
                    "counterfactual_rank",
                    required=True,
                ),
                name="operational_veto_chain_rank",
            ),
        )
        ranks = [
            _as_nonnegative_int(
                _at(
                    row,
                    "operational_veto_chain_rank",
                    "counterfactual_rank",
                    required=True,
                ),
                name="operational_veto_chain_rank",
            )
            for row in ordered
        ]
        if ranks != list(range(1, len(ranks) + 1)):
            raise PolicyAnalysisError(
                "counterfactual ranks are not contiguous from one"
            )
        terminal_values = [_terminal_action(row) for row in ordered]
        if (
            any(value is not None for value in terminal_values[:-1])
            or terminal_values[-1] is None
        ):
            raise PolicyAnalysisError(
                "counterfactual terminal action must occur only on the final row"
            )
        action = terminal_values[-1]
        assert action is not None
        stages = [_stage(row) for row in ordered]
        if stages != sorted(stages):
            raise PolicyAnalysisError(
                "unchanged-state counterfactual moved to an earlier stage"
            )
        signatures = []
        proposal_digests = []
        for row in ordered:
            signature = _at(row, "proposal_signature", required=True)
            if not isinstance(signature, list):
                raise PolicyAnalysisError(
                    "counterfactual proposal_signature must be an array"
                )
            expected_signature = [
                _stage(row),
                _at(row, "ordered_endpoints", required=True),
                _at(row, "canonical_edge_ids", required=True),
            ]
            if signature != expected_signature:
                raise PolicyAnalysisError(
                    "counterfactual proposal_signature disagrees with its proposal fields"
                )
            signatures.append(canonical_json_bytes(signature))
            proposal_digests.append(_proposal_sha(row))
            if "censored" not in row or _as_bool(row["censored"], name="censored"):
                raise PolicyAnalysisError(
                    "all-shot counterfactual row is censored or unmarked"
                )
            if "veto_budget" not in row or row["veto_budget"] is not None:
                raise PolicyAnalysisError(
                    "all-shot counterfactual is not explicitly uncapped"
                )
        if len(set(signatures)) != len(signatures):
            raise PolicyAnalysisError(
                "counterfactual state repeats a proposal signature"
            )
        if len(set(proposal_digests)) != len(proposal_digests):
            raise PolicyAnalysisError("counterfactual state repeats a proposal digest")
        row_contexts = [_context(row) for row in ordered]
        original = ordered[0]
        original_reference = _at(original, "original_proposal_sha256", required=True)
        if not isinstance(
            original_reference, str
        ) or original_reference != _proposal_sha(original):
            raise PolicyAnalysisError(
                "rank-one counterfactual is not the referenced original proposal"
            )
        for field in (
            "original_proposal_sha256",
            "original_state_sha256",
            "complete_pre_state_fingerprint",
            "local_active_state_fingerprint",
        ):
            expected = _at(original, field, required=True)
            if any(_at(row, field, required=True) != expected for row in ordered[1:]):
                raise PolicyAnalysisError(
                    f"counterfactual chain changed its original-state field {field!r}"
                )
        original_class = certificate_class(original)
        if original_class == "O-frame-safe":
            # Safe states may carry only cheap competitor inspection and are not
            # part of the unsafe-state counterfactual endpoint.
            continue
        safe_rows = [
            row for row in ordered[1:] if certificate_class(row) == "O-frame-safe"
        ]
        first_safe = safe_rows[0] if safe_rows else None
        expected_first_safe_rank = (
            None if first_safe is None else ordered.index(first_safe) + 1
        )
        expected_first_safe_digest = (
            None if first_safe is None else _proposal_sha(first_safe)
        )
        for row in ordered:
            recorded_rank = _at(row, "first_safe_rank", required=True)
            if recorded_rank is not None:
                recorded_rank = _as_nonnegative_int(
                    recorded_rank, name="first_safe_rank"
                )
            recorded_digest = _at(
                row, "first_safe_alternative_proposal_sha256", required=True
            )
            if (
                recorded_rank != expected_first_safe_rank
                or recorded_digest != expected_first_safe_digest
            ):
                raise PolicyAnalysisError(
                    "counterfactual first-safe references do not reconcile"
                )
        decisions = [_at(row, "decision", required=True) for row in ordered]
        if any(not isinstance(value, str) for value in decisions):
            raise PolicyAnalysisError("counterfactual decisions must be strings")
        for row, decision in zip(ordered, decisions):
            expected_decision = (
                "inspect-only" if certificate_class(row) == "O-frame-safe" else "veto"
            )
            if decision != expected_decision:
                raise PolicyAnalysisError(
                    "counterfactual decision disagrees with its certificate"
                )
        if action in {"same-stage-alternative", "later-stage-alternative"}:
            if first_safe is None:
                raise PolicyAnalysisError("alternative action has no safe proposal")
            if ordered.index(first_safe) != len(ordered) - 1:
                raise PolicyAnalysisError(
                    "all-shot counterfactual continued after first safe proposal"
                )
            expected = (
                "same-stage-alternative"
                if _stage(first_safe) == _stage(original)
                else "later-stage-alternative"
            )
            if _stage(first_safe) < _stage(original) or action != expected:
                raise PolicyAnalysisError(
                    "counterfactual terminal action/stage is inconsistent"
                )
            if any(
                _at(row, "exhaustion_kind", required=True) is not None
                for row in ordered
            ):
                raise PolicyAnalysisError(
                    "safe-alternative chain claims proposal exhaustion"
                )
        elif action == "abstain-true-exhaustion":
            if first_safe is not None:
                raise PolicyAnalysisError("abstention state contains a safe proposal")
            exhaustion = [_at(row, "exhaustion_kind", required=True) for row in ordered]
            if (
                any(value is not None for value in exhaustion[:-1])
                or exhaustion[-1] != "proposal"
            ):
                raise PolicyAnalysisError(
                    "abstention is not final true proposal exhaustion"
                )
        else:
            raise PolicyAnalysisError(
                "uncapped all-shot chain cannot terminate as censored-invalid"
            )
        state_oracle_calls = [
            _as_nonnegative_int(
                _at(row, "state_oracle_call_count", required=True),
                name="state_oracle_call_count",
            )
            for row in ordered
        ]
        if len(set(state_oracle_calls)) != 1 or state_oracle_calls[0] != len(ordered):
            raise PolicyAnalysisError(
                "counterfactual state oracle-call count does not reconcile"
            )
        context = row_contexts[0]
        candidate_count = _at(original, "state_total_candidate_count")
        if candidate_count is not None:
            candidate_count = _as_nonnegative_int(
                candidate_count, name="state_total_candidate_count"
            )
        original_events = _at(original, "events_removed_if_committed")
        if original_events is not None:
            original_events = _as_nonnegative_int(
                original_events, name="events_removed_if_committed"
            )
        first_safe_events = (
            None
            if first_safe is None
            else _at(first_safe, "events_removed_if_committed")
        )
        if first_safe_events is not None:
            first_safe_events = _as_nonnegative_int(
                first_safe_events, name="first-safe events_removed_if_committed"
            )
        states.append(
            {
                "state_id": state_id,
                "cell_id": _identity(original)[0],
                "global_shot_id": _identity(original)[2],
                "original_proposal_sha256": _proposal_sha(original),
                "trajectory_commit_index": _at(
                    original,
                    "trajectory_commit_index",
                    "commitment_index",
                    "accepted_prefix_length",
                ),
                "original_stage": _stage(original),
                "original_certificate_class": original_class,
                "original_cost_excess": _required_float(original, "cost_excess"),
                "exclusive_context": context["exclusive_support_component_context"],
                "terminal_action": action,
                "first_safe_rank": expected_first_safe_rank,
                "first_safe_stage": None if first_safe is None else _stage(first_safe),
                "veto_chain_length": len(ordered)
                if first_safe is None
                else len(ordered) - 1,
                "candidate_count": candidate_count,
                "original_decision_weight": _required_float(
                    original, "decision_weight"
                ),
                "original_path_length": _as_nonnegative_int(
                    _at(original, "canonical_edge_count", required=True),
                    name="canonical_edge_count",
                ),
                "original_weight_margin": _float_value(
                    original, "absolute_weight_margin"
                ),
                "original_events_removed": original_events,
                "first_safe_decision_weight": None
                if first_safe is None
                else _required_float(first_safe, "decision_weight"),
                "first_safe_path_length": None
                if first_safe is None
                else _as_nonnegative_int(
                    _at(first_safe, "canonical_edge_count", required=True),
                    name="canonical_edge_count",
                ),
                "first_safe_weight_margin": None
                if first_safe is None
                else _float_value(first_safe, "absolute_weight_margin"),
                "first_safe_events_removed": first_safe_events,
                "rows": ordered,
            }
        )
    return states


def _shot_key(row: Mapping[str, Any]) -> tuple[str, int]:
    cell, _, shot = _identity(row)
    return cell, shot


def _proposal_sha(row: Mapping[str, Any]) -> str:
    value = _at(row, "proposal_sha256", required=True)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PolicyAnalysisError("proposal_sha256 must be a 64-character digest")
    return value


def _arm_metric(
    shot: Mapping[str, Any],
    arm: Mapping[str, Any],
    *names: str,
) -> Any:
    value = _at(arm, *names)
    if value is not None:
        return value
    source_arm = arm.get("source_arm_id")
    role = _arm_role(str(source_arm)) if source_arm is not None else None
    for name in names:
        for container_name in (
            f"{name}_by_arm",
            f"arm_{name}",
            "workload_by_arm",
            "transaction_by_arm",
        ):
            container = shot.get(container_name)
            if not isinstance(container, Mapping):
                continue
            candidate = container.get(source_arm, container.get(role))
            if isinstance(candidate, Mapping):
                candidate = _at(candidate, name)
            if candidate is not None:
                return candidate
    return None


def _normalize_shots(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in sorted(rows, key=_identity):
        arms = _arm_results(row)
        required = {"u0", "shadow", "o-cost-tx", "o-frame-tx", "o-frame-partial"}
        if set(arms) != required:
            raise PolicyAnalysisError(
                f"shot arms differ from frozen B1 set: {sorted(arms)}"
            )
        original_hw = _original_hw(row)
        predictions = {name: _prediction_token(value) for name, value in arms.items()}
        differs = {
            name: _at(value, "differs_from_u0", "prediction_differs_from_u0")
            for name, value in arms.items()
        }
        for name in ("o-frame-tx", "o-frame-partial"):
            if predictions["u0"] is not None and predictions[name] is not None:
                equal = predictions[name] == predictions["u0"]
            elif differs[name] is not None:
                equal = not _as_bool(differs[name], name="differs_from_u0")
            else:
                raise PolicyAnalysisError(f"cannot verify exact U0 equality for {name}")
            if not equal:
                raise PolicyAnalysisError(f"fatal {name}/U0 prediction mismatch")
        failures = {name: _failure(value) for name, value in arms.items()}
        if any(
            predictions[name] == predictions["u0"] and failures[name] != failures["u0"]
            for name in ("shadow", "o-frame-tx", "o-frame-partial")
        ):
            raise PolicyAnalysisError(
                "equal prediction tokens have inconsistent failure labels"
            )
        normalized.append(
            {
                "cell_id": _identity(row)[0],
                "worker_id": _identity(row)[1],
                "global_shot_id": _identity(row)[2],
                "original_hw": original_hw,
                "zero_event": original_hw == 0,
                "predictions": predictions,
                "failures": failures,
                "final_hw": {
                    name: _as_nonnegative_int(
                        original_hw
                        if name == "u0"
                        else _arm_metric(
                            row,
                            value,
                            "final_residual_detector_hw",
                            "final_residual_detector_events",
                            "residual_detector_hw",
                        ),
                        name=f"{name} final residual detector HW",
                    )
                    for name, value in arms.items()
                },
                "provisional_removed": {
                    name: _arm_metric(
                        row,
                        value,
                        "provisional_events_removed",
                        "workload.provisional_events_removed",
                    )
                    for name, value in arms.items()
                },
                "rollback_lost": {
                    name: _arm_metric(
                        row,
                        value,
                        "events_lost_to_rollback",
                        "workload.events_lost_to_rollback",
                    )
                    for name, value in arms.items()
                },
                "raw": row,
            }
        )
    return normalized


def _normalize_shadow_proposals(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        if _origin(row) not in {"shadow-original", "shadow"}:
            continue
        proposal_sha = _proposal_sha(row)
        identity = (_identity(row)[0], _identity(row)[2], proposal_sha)
        if identity in seen:
            raise PolicyAnalysisError("duplicate shadow proposal digest")
        seen.add(identity)
        context = _context(row)
        durable_raw = _at(row, "durable", "decision.durable", required=True)
        durable = _as_bool(durable_raw, name="durable")
        if not durable:
            raise PolicyAnalysisError("shadow-original proposal is not durable")
        decision = _at(row, "decision", "decision.kind", required=True)
        if isinstance(decision, Mapping):
            decision = decision.get("kind")
        if decision != "shadow-commit":
            raise PolicyAnalysisError(
                "shadow-original proposal is not a shadow commitment"
            )
        sequence = _as_nonnegative_int(
            _at(
                row,
                "trajectory_commit_index",
                "commitment_index",
                "accepted_prefix_length",
                required=True,
            ),
            name="trajectory_commit_index",
        )
        proposal_signature = _at(row, "proposal_signature", required=True)
        expected_signature = [
            _stage(row),
            _at(row, "ordered_endpoints", required=True),
            _at(row, "canonical_edge_ids", required=True),
        ]
        if proposal_signature != expected_signature:
            raise PolicyAnalysisError(
                "shadow-original proposal_signature disagrees with its proposal fields"
            )
        competitor_exists = _as_bool(
            _at(row, "same_stage_competitor_exists", required=True),
            name="same_stage_competitor_exists",
        )
        competitor_weight = _float_value(row, "same_stage_competitor_weight")
        weight_margin = _float_value(row, "absolute_weight_margin")
        relative_margin = _float_value(row, "relative_weight_margin")
        if competitor_exists is False and any(
            value is not None
            for value in (competitor_weight, weight_margin, relative_margin)
        ):
            raise PolicyAnalysisError(
                "missing competitor is paired with competitor metrics"
            )
        if competitor_exists is True and (
            competitor_weight is None or weight_margin is None
        ):
            raise PolicyAnalysisError("recorded competitor lacks weight/margin")
        events_removed = _at(row, "events_removed_if_committed")
        if events_removed is not None:
            events_removed = _as_nonnegative_int(
                events_removed, name="events_removed_if_committed"
            )
        result.append(
            {
                "proposal_sha256": proposal_sha,
                "cell_id": _identity(row)[0],
                "global_shot_id": _identity(row)[2],
                "stage": _stage(row),
                "domain": _domain_key(row),
                "certificate_class": certificate_class(row),
                "frame_compatible": _frame_compatible(row),
                "durable": durable,
                "trajectory_commit_index": sequence,
                "cost_excess": _required_float(row, "cost_excess"),
                "tau_k": _required_float(row, "tau_k"),
                "decision_weight": _required_float(row, "decision_weight"),
                "path_length": _as_nonnegative_int(
                    _at(row, "canonical_edge_count", "path_length", required=True),
                    name="canonical_edge_count",
                ),
                "local_weight_margin": weight_margin,
                "relative_weight_margin": relative_margin,
                "same_stage_competitor_exists": competitor_exists,
                "same_stage_competitor_weight": competitor_weight,
                "events_removed": events_removed,
                "window_offset": _at(
                    row,
                    "window_offset",
                    "round_offset_from_window_start",
                    "local.window_offset",
                ),
                "static_boundary_competition": _at(
                    row,
                    "static_boundary_competition",
                    "local.static_boundary_competition",
                ),
                "domain_hw": _at(row, "domain_current_hw", "state.domain_current_hw"),
                "candidate_multiplicity": _at(
                    row,
                    "candidate_multiplicity",
                    "state_total_candidate_count",
                ),
                **context,
                "raw": row,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["cell_id"],
            item["global_shot_id"],
            item["proposal_sha256"],
        ),
    )


def _domain_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        arm_id = _at(row, "arm_id", required=True)
        if not isinstance(arm_id, str):
            raise PolicyAnalysisError("domain arm_id must be a string")
        role = _arm_role(arm_id)
        cell = _identity(row)[0]
        status = _at(
            row, "domain_terminal_status", "status", "outcome.status", required=True
        )
        if status == "partial-exhausted":
            status = "partial-exhaustion"
        if status not in {"below-limit", "success", "rollback", "partial-exhaustion"}:
            raise PolicyAnalysisError(f"invalid domain terminal status {status!r}")
        key = (cell, role)
        counters[key]["domains"] += 1
        counters[key][f"status:{status}"] += 1
        initial_hw = _at(row, "domain_initial_hw", "initial_hw")
        target = _at(row, "residual_hw_target", "residual_hw_limit")
        if initial_hw is not None and target is not None:
            active = int(
                _as_nonnegative_int(initial_hw, name="domain_initial_hw")
                > _as_nonnegative_int(target, name="residual_hw_target")
            )
            counters[key]["activated_domains"] += active
            counters[key]["activation_known_domains"] += 1
        for field in (
            "provisional_events_removed",
            "durable_events_removed",
            "events_lost_to_rollback",
            "accepted_prefix_length",
        ):
            value = _at(row, field, f"outcome.{field}")
            if value is not None:
                counters[key][field] += _as_nonnegative_int(value, name=field)
    result = []
    for (cell, arm), values in sorted(counters.items()):
        row: dict[str, Any] = {
            "cell_id": cell,
            "arm": arm,
            **dict(sorted(values.items())),
        }
        denominator = values["domains"]
        for status in ("below-limit", "success", "rollback", "partial-exhaustion"):
            count = values[f"status:{status}"]
            row[f"status:{status}"] = count
            row[f"rate:{status}"] = count / denominator
        row["activation_rate"] = (
            values["activated_domains"] / denominator
            if values["activation_known_domains"] == denominator
            else None
        )
        result.append(row)
    return result
