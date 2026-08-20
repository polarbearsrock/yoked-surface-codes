"""Ledger encodings, the ground-truth firewall, and the policy-core adapter.

This slice owns the deterministic artifact encodings (canonical JSONL, frozen
gzip), the shared ground-truth key firewall used by both the collector and
the casebook layer, fail-closed validation of the v2 support-difference and
context-union ledgers, and :func:`_audit_policy_shot` — the one narrow
adapter through which the scientific per-shot policy core is reached.  It
inherits the package isolation contract (see ``__init__``): sampled
observables never pass through any function defined here.
"""

from __future__ import annotations

import dataclasses
import gzip
import io
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from yoked.decoding._promatch_stats import canonical_json_bytes
from yoked.decoding.oracle.full_graph import OracleTolerance

from yoked.decoding.oracle.policy_experiment._identity import (
    ARM_IDS,
    GZIP_LEVEL,
    _sha256,
)
from yoked.decoding.oracle.policy_experiment._protocol import (
    _expected_context_taxonomy,
)


# Exact field set of one promatch-support-difference-v2 component; shared with
# the casebook layer (yoked.decoding.oracle.policy_casebook).
SUPPORT_COMPONENT_FIELDS_V2 = frozenset(
    {
        "certificate_kind",
        "canonical_edge_ids",
        "support_cancellation_edge_ids",
        "component_detector_ids",
        "candidate_support_witness_edge_ids",
        "candidate_boundary_witness_detector_ids",
        "labels",
        "candidate_relevant",
        "candidate_relevance_reasons",
    }
)
# One ground-truth firewall vocabulary for every enforcement layer.  The
# collector fields below are this module's historical set; the casebook layer
# (yoked.decoding.oracle.policy_casebook) historically added four extra
# spellings, kept explicit here so the shared union stays auditable.  Both
# layers reject the full union via :func:`forbid_ground_truth_keys`.
_COLLECTOR_GROUND_TRUTH_FIELDS = frozenset(
    {
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
    }
)
CASEBOOK_EXTRA_GROUND_TRUTH_KEYS = frozenset(
    {
        "arm_failure",
        "failed",
        "logical_failure",
        "posthoc",
    }
)
GROUND_TRUTH_FORBIDDEN_KEYS = (
    _COLLECTOR_GROUND_TRUTH_FIELDS | CASEBOOK_EXTRA_GROUND_TRUTH_KEYS
)
_COLLECTOR_OWNED_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "cell_id",
        "worker_id",
        "worker_shot_index",
        "global_shot_id",
        "stim_seed",
        "physical_input_sha256",
        "detector_input_sha256",
        "circuit_sha256",
        "packed_detectors_hex",
        "packed_detector_bits",
        "packed_detectors_sha256",
        "packed_actual_observables_hex",
        "packed_actual_observable_bits",
        "packed_actual_observables_sha256",
        "arm_predictions_hex",
        "arm_failures",
    }
)


@dataclasses.dataclass(frozen=True)
class NormalizedPolicyShot:
    """Validated per-shot records split by their artifact destination."""

    arm_predictions: dict[str, bytes]
    shot: dict[str, Any]
    proposals: tuple[dict[str, Any], ...]
    counterfactuals: tuple[dict[str, Any], ...]
    domains: tuple[dict[str, Any], ...]


def canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serializes records as canonical, newline-terminated JSON Lines."""

    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def deterministic_gzip(data: bytes, *, level: int = GZIP_LEVEL) -> bytes:
    """Compresses bytes with frozen level, empty name, and zero timestamp."""

    target = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", compresslevel=level, fileobj=target, mtime=0
    ) as f:
        f.write(data)
    return target.getvalue()


def _artifact_metadata(
    *,
    compressed: bytes,
    uncompressed: bytes,
    rows: int,
    schema: str,
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


def forbid_ground_truth_keys(
    value: Any, *, path: str = "audit", error: type[ValueError] = ValueError
) -> None:
    """Recursively rejects every ground-truth-like key in a JSON-like tree.

    This is the single firewall walker for both enforcement layers: the
    collector calls it with :class:`ValueError` and the casebook layer wraps
    it with its own error type.  The key set is the full frozen union.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in GROUND_TRUTH_FORBIDDEN_KEYS or lowered.startswith(
                ("actual_observable", "packed_actual_observable")
            ):
                raise error(f"ground-truth-like key is forbidden at {path}.{key}")
            forbid_ground_truth_keys(item, path=f"{path}.{key}", error=error)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            forbid_ground_truth_keys(item, path=f"{path}[{index}]", error=error)


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

    if (
        row.get("support_difference_representation_version")
        != "promatch-support-difference-v2"
    ):
        raise ValueError(f"{path} lacks the exact v2 support-difference representation")
    components = row.get("support_difference_components")
    if not isinstance(components, list):
        raise ValueError(f"{path} support_difference_components must be an array")
    exact_keys = SUPPORT_COMPONENT_FIELDS_V2
    labels_vocab = set(_expected_context_taxonomy()["multi_labels"])
    reason_vocab = {
        "candidate-support-edge",
        "candidate-boundary-detector",
        "candidate-residual-support-cancellation",
    }
    detector_boundary_ids = row.get("detector_boundary_ids")
    if not canonical_edge_ids(detector_boundary_ids):
        raise ValueError(
            f"{path} detector_boundary_ids is not a canonical detector set"
        )
    real: set[int] = set()
    cancellations: set[int] = set()
    candidate_label_groups: list[list[str]] = []
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
                        raise ValueError(
                            f"{path} component references an unknown graph edge"
                        )
                    edge = graph.edges[edge_id]
                    expected_detector_ids.add(int(edge.source))
                    if edge.target is not None:
                        expected_detector_ids.add(int(edge.target))
                if detector_ids != sorted(expected_detector_ids):
                    raise ValueError(
                        f"{path} component detector witness disagrees with graph"
                    )
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
                or detector_ids
                or support_witness
                or boundary_witness
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
            candidate_label_groups.append(labels)
    if (
        components
        != sorted(
            real_components, key=lambda component: component["canonical_edge_ids"]
        )
        + cancellation_components
    ):
        raise ValueError(f"{path} support certificates are not in canonical order")
    supports: dict[str, list[int]] = {}
    for field in (
        "B_base_support_edge_ids",
        "P_candidate_support_edge_ids",
        "R_residual_support_edge_ids",
        "Q_forced_parity_support_edge_ids",
        "X_support_difference_edge_ids",
        "P_intersection_R_edge_ids",
    ):
        value = row.get(field)
        if not canonical_edge_ids(value):
            raise ValueError(f"{path} {field} is not a canonical support")
        supports[field] = value
    b, p, r = (
        set(supports[name])
        for name in (
            "B_base_support_edge_ids",
            "P_candidate_support_edge_ids",
            "R_residual_support_edge_ids",
        )
    )
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
    candidate_labels = _normalized_context_union(
        *candidate_label_groups, path=f"{path}.support_difference_components"
    )
    if row.get("support_difference_component_labels") != candidate_labels:
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
            raise ValueError(
                f"{path} has an algebraically impossible X-empty frame conflict"
            )
        if (
            row.get("oracle_policy_accepts") is False
            and not supports["P_intersection_R_edge_ids"]
        ):
            raise ValueError(f"{path} unsafe row has no X or cancellation certificate")
    for flag in (
        "supports_square_free",
        "B_base_support_square_free",
        "P_candidate_support_square_free",
        "R_residual_support_square_free",
        "Q_forced_parity_support_square_free",
        "X_support_difference_square_free",
    ):
        if row.get(flag) is not True:
            raise ValueError(f"{path} {flag} must be true")


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
            from yoked.decoding.oracle.policy_audit import audit_policy_shot
        except ImportError as ex:
            raise RuntimeError(
                "B1 policy core is unavailable; expected "
                "yoked.decoding.oracle.policy_audit:audit_policy_shot"
            ) from ex
        audit_fn = audit_policy_shot
    raw = audit_fn(graph, syndrome.copy(), tolerance=tolerance)
    if hasattr(raw, "to_json"):
        raw = raw.to_json()
    required = {"arm_predictions", "shot", "proposals", "counterfactuals", "domains"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(
            f"audit_policy_shot output fields must be exactly {sorted(required)}"
        )
    forbid_ground_truth_keys(raw)
    _require_float_hex_companions(raw)
    width = (graph.num_observables + 7) // 8
    predictions_raw = raw["arm_predictions"]
    if not isinstance(predictions_raw, Mapping) or set(predictions_raw) != set(ARM_IDS):
        raise ValueError(
            "policy core must return exactly one prediction for every B1 arm"
        )
    predictions = {
        arm_id: _normalize_prediction(predictions_raw[arm_id], width=width)
        for arm_id in ARM_IDS
    }
    collections: dict[str, tuple[dict[str, Any], ...]] = {}
    for name in ("proposals", "counterfactuals", "domains"):
        values = raw[name]
        if not isinstance(values, (tuple, list)) or not all(
            isinstance(v, Mapping) for v in values
        ):
            raise ValueError(f"policy core {name} must be an array of objects")
        collections[name] = tuple(dict(v) for v in values)
    for name in ("proposals", "counterfactuals"):
        for index, row in enumerate(collections[name]):
            _validate_support_difference_ledger(
                row, path=f"audit.{name}[{index}]", graph=graph
            )
            _validate_context_union_ledger(row, path=f"audit.{name}[{index}]")
    if not isinstance(raw["shot"], Mapping):
        raise ValueError("policy core shot must be an object")
    return NormalizedPolicyShot(
        arm_predictions=predictions,
        shot=dict(raw["shot"]),
        proposals=collections["proposals"],
        counterfactuals=collections["counterfactuals"],
        domains=collections["domains"],
    )
