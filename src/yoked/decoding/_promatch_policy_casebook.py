"""Authenticated B1 casebook expansion and finalization.

This layer owns corpus authentication and artifact installation.  The core
expander remains detector-only and therefore cannot receive sampled logical
observables or post-hoc correctness labels.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

if os.environ.get("TMPDIR"):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(os.environ["TMPDIR"]) / "yoked-surface-codes-matplotlib"),
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from yoked.decoding._promatch_experiment import prepare_cell
from yoked.decoding._promatch_oracle import OracleTolerance
from yoked.decoding._promatch_policy_analysis import (
    analyze_policy_audit,
    canonical_json_bytes,
    load_policy_audit,
    verify_existing_policy_analysis,
)
from yoked.decoding._promatch_policy_audit import expand_policy_casebook_state
from yoked.decoding._promatch_policy_experiment import validate_policy_protocol


EXPANSION_SCHEMA = "promatch-l1-policy-audit-casebook-expansion-v1"
EXPANSION_MANIFEST_SCHEMA = "promatch-l1-policy-audit-casebook-expansion-manifest-v1"
EXPANSION_READY_SCHEMA = "promatch-l1-policy-audit-expansion-ready-v1"
COMPLETE_SCHEMA = "promatch-l1-policy-audit-complete-v1"
EXPANSION_ROW_SCHEMA = "promatch-l1-policy-audit-casebook-exhaustive-row-v1"
_GROUND_TRUTH_KEYS = {
    "actual_observables", "actual_observables_hex", "packed_actual_observables_hex",
    "arm_failure", "arm_failures", "failure", "failed", "logical_failure",
    "correctness", "correct", "regression", "recovery", "posthoc",
    "posthoc_ground_truth",
}
_SUPPORT_DIFFERENCE_REPRESENTATION = "promatch-support-difference-v2"
_SUPPORT_COMPONENT_FIELDS = {
    "certificate_kind", "canonical_edge_ids", "support_cancellation_edge_ids",
    "component_detector_ids", "candidate_support_witness_edge_ids",
    "candidate_boundary_witness_detector_ids",
    "labels", "candidate_relevant", "candidate_relevance_reasons",
}
class PolicyCasebookError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as ex:
        raise PolicyCasebookError(f"cannot read canonical JSON {path}") from ex
    if not isinstance(value, dict):
        raise PolicyCasebookError(f"{path} must contain one JSON object")
    if canonical_json_bytes(value) + b"\n" != path.read_bytes():
        raise PolicyCasebookError(f"{path} is not canonical JSON")
    return value


def _state_id(row: Mapping[str, Any]) -> str:
    state = row.get("original_state_sha256", row.get("complete_pre_state_fingerprint"))
    identity = [
        row.get("cell_id"), row.get("worker_id"), row.get("global_shot_id"),
        [row.get("patch_id"), row.get("basis"), row.get("window_id")], state,
    ]
    if not isinstance(state, str) or not state:
        raise PolicyCasebookError("rank-one counterfactual lacks original state identity")
    return _sha(canonical_json_bytes(identity))


def _strip_nondeterministic_timing(value: Any) -> Any:
    """Removes wall-clock telemetry from scientifically digested casebook rows."""
    if isinstance(value, Mapping):
        return {
            str(key): _strip_nondeterministic_timing(item)
            for key, item in value.items()
            if not str(key).endswith("_wall_ns") and str(key) != "timing_telemetry"
        }
    if isinstance(value, list):
        return [_strip_nondeterministic_timing(item) for item in value]
    return value


def _reject_ground_truth_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            text = str(key)
            normalized = text.lower().replace("-", "_")
            if (
                normalized in _GROUND_TRUTH_KEYS
                or normalized.startswith("actual_observable")
                or normalized.startswith("packed_actual_observable")
            ):
                raise PolicyCasebookError(f"ground-truth-like key is forbidden at {path}.{text}")
            _reject_ground_truth_keys(item, path=f"{path}.{text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_ground_truth_keys(item, path=f"{path}[{index}]")


def _verify_float_hex(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, float):
                companion = value.get(f"{key}_hex")
                if companion != item.hex():
                    raise PolicyCasebookError(f"float lacks exact hex companion at {path}.{key}")
            _verify_float_hex(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _verify_float_hex(item, path=f"{path}[{index}]")


def _deterministic_gzip(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as stream:
        stream.write(data)
    return output.getvalue()


def _canonical_edge_ids(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(edge_id, bool) or not isinstance(edge_id, int) or edge_id < 0
        for edge_id in value
    ):
        raise PolicyCasebookError(f"{field} must be an array of nonnegative integer edge IDs")
    if value != sorted(set(value)):
        raise PolicyCasebookError(f"{field} must be sorted and duplicate-free")
    return list(value)


def _canonical_detector_ids(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(detector_id, bool)
        or not isinstance(detector_id, int)
        or detector_id < 0
        for detector_id in value
    ):
        raise PolicyCasebookError(
            f"{field} must be an array of nonnegative integer detector IDs"
        )
    if value != sorted(set(value)):
        raise PolicyCasebookError(f"{field} must be sorted and duplicate-free")
    return list(value)


def _canonical_labels(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(label, str) or not label for label in value
    ):
        raise PolicyCasebookError(f"{field} must be an array of nonempty strings")
    if value != sorted(set(value)):
        raise PolicyCasebookError(f"{field} must be sorted and duplicate-free")
    return list(value)


def _support_evidence(row: Mapping[str, Any], *, graph: Any) -> dict[str, Any]:
    """Validates and normalizes the producer's exhaustive-row v2 evidence."""
    if row.get("support_difference_representation_version") != (
        _SUPPORT_DIFFERENCE_REPRESENTATION
    ):
        raise PolicyCasebookError(
            "exhaustive row requires support_difference_representation_version "
            f"{_SUPPORT_DIFFERENCE_REPRESENTATION!r}"
        )
    candidate = _canonical_edge_ids(
        row.get("P_candidate_support_edge_ids"), field="P_candidate_support_edge_ids"
    )
    difference = _canonical_edge_ids(
        row.get("X_support_difference_edge_ids"), field="X_support_difference_edge_ids"
    )
    cancellation = _canonical_edge_ids(
        row.get("P_intersection_R_edge_ids"), field="P_intersection_R_edge_ids"
    )
    cancellation_alias = _canonical_edge_ids(
        row.get("support_cancellation_edge_ids"), field="support_cancellation_edge_ids"
    )
    if cancellation_alias != cancellation:
        raise PolicyCasebookError(
            "support_cancellation_edge_ids must exactly equal P_intersection_R_edge_ids"
        )
    if not set(cancellation).issubset(candidate):
        raise PolicyCasebookError("P_intersection_R_edge_ids must be a subset of candidate P")
    candidate_boundary = _canonical_detector_ids(
        row.get("detector_boundary_ids"), field="detector_boundary_ids"
    )
    context_labels = _canonical_labels(
        row.get("support_difference_component_labels"),
        field="support_difference_component_labels",
    )
    components = row.get("support_difference_components")
    if not isinstance(components, list):
        raise PolicyCasebookError("support_difference_components must be an array")
    disconnected_flag = row.get("disconnected_support_reconfiguration")
    if not isinstance(disconnected_flag, bool):
        raise PolicyCasebookError("disconnected_support_reconfiguration must be boolean")

    real_edge_ids: set[int] = set()
    cancellation_edge_ids: set[int] = set()
    relevant_edge_ids: set[int] = set()
    disconnected_edge_ids: set[int] = set()
    derived_context: set[str] = set()
    remote_labels: set[str] = set()
    normalized_components: list[dict[str, Any]] = []
    seen_cancellation = False
    previous_real_key: tuple[int, ...] | None = None
    for index, raw_component in enumerate(components):
        if not isinstance(raw_component, Mapping) or set(raw_component) != _SUPPORT_COMPONENT_FIELDS:
            raise PolicyCasebookError(
                f"support_difference_components[{index}] must have the exact v2 component fields"
            )
        component = dict(raw_component)
        kind = component["certificate_kind"]
        canonical = _canonical_edge_ids(
            component["canonical_edge_ids"],
            field=f"support_difference_components[{index}].canonical_edge_ids",
        )
        component_cancellation = _canonical_edge_ids(
            component["support_cancellation_edge_ids"],
            field=(
                f"support_difference_components[{index}].support_cancellation_edge_ids"
            ),
        )
        component_detectors = _canonical_detector_ids(
            component["component_detector_ids"],
            field=f"support_difference_components[{index}].component_detector_ids",
        )
        support_witness = _canonical_edge_ids(
            component["candidate_support_witness_edge_ids"],
            field=(
                f"support_difference_components[{index}]."
                "candidate_support_witness_edge_ids"
            ),
        )
        boundary_witness = _canonical_detector_ids(
            component["candidate_boundary_witness_detector_ids"],
            field=(
                f"support_difference_components[{index}]."
                "candidate_boundary_witness_detector_ids"
            ),
        )
        labels = _canonical_labels(
            component["labels"], field=f"support_difference_components[{index}].labels"
        )
        candidate_relevant = component["candidate_relevant"]
        reasons = _canonical_labels(
            component["candidate_relevance_reasons"],
            field=f"support_difference_components[{index}].candidate_relevance_reasons",
        )
        if not isinstance(candidate_relevant, bool):
            raise PolicyCasebookError(
                f"support_difference_components[{index}].candidate_relevant must be boolean"
            )
        if kind == "real-x-component":
            if seen_cancellation:
                raise PolicyCasebookError("real X components must precede support-cancellation")
            if not canonical or component_cancellation:
                raise PolicyCasebookError(
                    "real-x-component requires canonical edges and no cancellation edges"
                )
            real_key = tuple(canonical)
            if previous_real_key is not None and real_key <= previous_real_key:
                raise PolicyCasebookError("real X components must be in canonical edge order")
            previous_real_key = real_key
            computed_detectors: set[int] = set()
            for edge_id in canonical:
                try:
                    edge = graph.edges[edge_id]
                except (AttributeError, IndexError) as ex:
                    raise PolicyCasebookError(
                        f"support component references unknown edge {edge_id}"
                    ) from ex
                computed_detectors.add(int(edge.source))
                if edge.target is not None:
                    computed_detectors.add(int(edge.target))
            expected_support_witness = sorted(set(canonical).intersection(candidate))
            expected_boundary_witness = sorted(
                computed_detectors.intersection(candidate_boundary)
            )
            if component_detectors != sorted(computed_detectors):
                raise PolicyCasebookError(
                    "real-x-component detector vertices disagree with graph topology"
                )
            if support_witness != expected_support_witness:
                raise PolicyCasebookError(
                    "real-x-component candidate support witness disagrees with component P"
                )
            if boundary_witness != expected_boundary_witness:
                raise PolicyCasebookError(
                    "real-x-component candidate boundary witness disagrees with component "
                    "detectors and detector_boundary_ids"
                )
            expected_reasons = []
            if expected_support_witness:
                expected_reasons.append("candidate-support-edge")
            if expected_boundary_witness:
                expected_reasons.append("candidate-boundary-detector")
            expected_reasons.sort()
            if reasons != expected_reasons:
                raise PolicyCasebookError(
                    "real-x-component relevance reasons disagree with authenticated witnesses"
                )
            if candidate_relevant != bool(expected_reasons):
                raise PolicyCasebookError(
                    "real-x-component candidate relevance must exactly match its reasons"
                )
            overlap = real_edge_ids.intersection(canonical)
            if overlap:
                raise PolicyCasebookError("real X components contain duplicate edge membership")
            real_edge_ids.update(canonical)
            if candidate_relevant:
                relevant_edge_ids.update(canonical)
                derived_context.update(labels)
            else:
                disconnected_edge_ids.update(canonical)
                remote_labels.update(labels)
        elif kind == "support-cancellation":
            if seen_cancellation:
                raise PolicyCasebookError("support-cancellation certificate must be unique")
            seen_cancellation = True
            if canonical or not component_cancellation:
                raise PolicyCasebookError(
                    "support-cancellation requires cancellation edges and no canonical edges"
                )
            if component_detectors or support_witness or boundary_witness:
                raise PolicyCasebookError(
                    "support-cancellation witness arrays must have the deterministic empty shape"
                )
            if not candidate_relevant or reasons != [
                "candidate-residual-support-cancellation"
            ]:
                raise PolicyCasebookError(
                    "support-cancellation has invalid candidate relevance"
                )
            if "support-cancellation" not in labels:
                raise PolicyCasebookError(
                    "support-cancellation certificate lacks its required label"
                )
            cancellation_edge_ids.update(component_cancellation)
            derived_context.update(labels)
        else:
            raise PolicyCasebookError(f"unsupported support component kind {kind!r}")
        normalized_components.append({
            "certificate_kind": kind,
            "canonical_edge_ids": canonical,
            "support_cancellation_edge_ids": component_cancellation,
            "component_detector_ids": component_detectors,
            "candidate_support_witness_edge_ids": support_witness,
            "candidate_boundary_witness_detector_ids": boundary_witness,
            "labels": labels,
            "candidate_relevant": candidate_relevant,
            "candidate_relevance_reasons": reasons,
        })

    if sorted(real_edge_ids) != difference:
        raise PolicyCasebookError("real X components must exactly partition X support")
    if sorted(cancellation_edge_ids) != cancellation:
        raise PolicyCasebookError(
            "support-cancellation certificate must exactly represent P intersection R"
        )
    if bool(disconnected_edge_ids) != disconnected_flag:
        raise PolicyCasebookError(
            "disconnected_support_reconfiguration disagrees with component relevance"
        )
    if sorted(derived_context) != context_labels:
        raise PolicyCasebookError(
            "support_difference_component_labels must contain exactly candidate-relevant "
            "component and cancellation labels"
        )
    return {
        "candidate_edge_ids": candidate,
        "support_difference_edge_ids": difference,
        "support_cancellation_edge_ids": cancellation,
        "candidate_relevant_support_difference_edge_ids": sorted(relevant_edge_ids),
        "disconnected_support_difference_edge_ids": sorted(disconnected_edge_ids),
        "context_labels": context_labels,
        "remote_support_role_labels": sorted(remote_labels),
        "components": normalized_components,
    }


def _support_graph_snapshot(
    graph: Any, rows: Sequence[Mapping[str, Any]], *, graph_fingerprint: str,
    layout_fingerprint: str,
) -> dict[str, Any]:
    active = {
        int(detector) for detector in rows[0].get("local_active_state_fingerprint", [])
        if int(detector) >= 0
    }
    candidate_ranks: dict[int, list[int]] = {}
    difference_ranks: dict[int, list[int]] = {}
    relevant_ranks: dict[int, list[int]] = {}
    disconnected_ranks: dict[int, list[int]] = {}
    cancellation_ranks: dict[int, list[int]] = {}
    for row in rows:
        rank = int(row["operational_veto_chain_rank"])
        evidence = _support_evidence(row, graph=graph)
        for edge_id in evidence["candidate_edge_ids"]:
            candidate_ranks.setdefault(edge_id, []).append(rank)
        for edge_id in evidence["support_difference_edge_ids"]:
            difference_ranks.setdefault(edge_id, []).append(rank)
        for edge_id in evidence["candidate_relevant_support_difference_edge_ids"]:
            relevant_ranks.setdefault(edge_id, []).append(rank)
        for edge_id in evidence["disconnected_support_difference_edge_ids"]:
            disconnected_ranks.setdefault(edge_id, []).append(rank)
        for edge_id in evidence["support_cancellation_edge_ids"]:
            cancellation_ranks.setdefault(edge_id, []).append(rank)
    edge_ids = sorted(
        set(candidate_ranks) | set(difference_ranks) | set(cancellation_ranks)
    )
    edges = []
    detector_ids = set(active)
    for edge_id in edge_ids:
        try:
            edge = graph.edges[edge_id]
        except (AttributeError, IndexError) as ex:
            raise PolicyCasebookError(f"support snapshot references unknown edge {edge_id}") from ex
        detector_ids.add(int(edge.source))
        if edge.target is not None:
            detector_ids.add(int(edge.target))
        membership = []
        if edge_id in candidate_ranks:
            membership.append("candidate-P")
        if edge_id in difference_ranks:
            membership.append("support-difference-X")
        if edge_id in relevant_ranks:
            membership.append("candidate-relevant-X")
        if edge_id in disconnected_ranks:
            membership.append("disconnected-X")
        if edge_id in cancellation_ranks:
            membership.append("support-cancellation-P-intersection-R")
        edges.append({
            "edge_id": edge_id, "source": int(edge.source),
            "target": None if edge.target is None else int(edge.target),
            "membership": membership,
            "candidate_ranks": candidate_ranks.get(edge_id, []),
            "support_difference_ranks": difference_ranks.get(edge_id, []),
            "candidate_relevant_ranks": relevant_ranks.get(edge_id, []),
            "disconnected_ranks": disconnected_ranks.get(edge_id, []),
            "support_cancellation_ranks": cancellation_ranks.get(edge_id, []),
        })
    coordinates = getattr(getattr(graph, "layout", None), "coordinates", ())
    roles = getattr(getattr(graph, "layout", None), "roles", ())
    detectors = []
    for detector_id in sorted(detector_ids):
        coordinate = coordinates[detector_id] if detector_id < len(coordinates) else ()
        role = roles[detector_id] if detector_id < len(roles) else None
        role_fields = {}
        if role is not None:
            for name in ("patch_id", "check_basis", "time", "window_id"):
                if hasattr(role, name):
                    role_fields[name] = getattr(role, name)
        detectors.append({
            "detector_id": detector_id, "active_local": detector_id in active,
            "coordinate_hex": [float(value).hex() for value in coordinate],
            "role": {
                "type": "unknown" if role is None else type(role).__name__,
                **role_fields,
            },
        })
    return {
        "schema": "promatch-l1-policy-audit-support-graph-snapshot-v2",
        "graph_fingerprint": graph_fingerprint,
        "layout_fingerprint": layout_fingerprint,
        "cropping": (
            "active-local-union-candidate-P-union-support-difference-X-union-"
            "support-cancellation-P-intersection-R"
        ),
        "detectors": detectors, "edges": edges,
    }


def _render_state_diagram(state: Mapping[str, Any], path: Path) -> None:
    state_id = str(state["state_id"])
    rows = state["rows"]
    ranks = [int(row["operational_veto_chain_rank"]) for row in rows]
    excess = [float(row["cost_excess"]) for row in rows]
    safe = [bool(row["oracle_policy_accepts"]) for row in rows]
    stages = [int(row["stage"]) for row in rows]
    first_safe = next((rank for rank, value in zip(ranks, safe) if value), None)
    contexts = sorted({
        label for row in rows for label in row["support_difference_component_labels"]
    })
    remote_roles = sorted({
        label for item in state["all_veto_timeline"]
        for label in item["remote_support_role_labels"]
    })
    visibility = sorted({
        label for row in rows for label in row.get("feature_visibility", {}).values()
    })
    fig, (ax, graph_ax) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = ["#2a7f62" if value else "#b3443f" for value in safe]
    markers = ["o" if value else "X" for value in safe]
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.plot(ranks, excess, color="#777777", linewidth=1, zorder=1)
    for rank, value, color, marker, stage in zip(ranks, excess, colors, markers, stages):
        ax.scatter([rank], [value], color=color, marker=marker, s=55, zorder=2)
        ax.annotate(f"S{stage}", (rank, value), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8)
        path_count = len(rows[rank - 1]["P_candidate_support_edge_ids"])
        difference_count = len(rows[rank - 1]["X_support_difference_edge_ids"])
        ax.annotate(f"P={path_count}, X={difference_count}", (rank, value),
                    xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7)
    ax.set(
        xlabel="Unchanged-state proposal rank",
        ylabel="Forced correction cost excess",
        title=f"Casebook {state_id[:12]}: exhaustive local slate",
        xticks=ranks,
    )
    ax.grid(axis="y", alpha=0.2)
    snapshot = state["support_graph_snapshot"]
    positions: dict[int, tuple[float, float]] = {}
    for index, detector in enumerate(snapshot["detectors"]):
        coordinate = [float.fromhex(value) for value in detector["coordinate_hex"]]
        positions[int(detector["detector_id"])] = (
            coordinate[0] if coordinate else float(index),
            coordinate[1] if len(coordinate) > 1 else 0.0,
        )
    for edge in snapshot["edges"]:
        source = positions[edge["source"]]
        target_id = edge["target"]
        target = (
            (source[0] + 0.35, source[1] + 0.35)
            if target_id is None else positions[target_id]
        )
        membership = set(edge["membership"])
        if "support-cancellation-P-intersection-R" in membership:
            color, style, width = "#b3443f", ":", 2.8
        elif "candidate-P" in membership and "candidate-relevant-X" in membership:
            color, style, width = "#7551a6", "-", 2.4
        elif "candidate-relevant-X" in membership:
            color, style, width = "#d07724", "-", 2.2
        elif "disconnected-X" in membership:
            color, style, width = "#888888", "--", 1.7
        elif "candidate-P" in membership:
            color, style, width = "#326fa8", "-", 2.0
        else:
            color, style, width = "#d07724", "--", 2.0
        graph_ax.plot([source[0], target[0]], [source[1], target[1]],
                      color=color, linestyle=style, linewidth=width, zorder=1)
        midpoint = ((source[0] + target[0]) / 2, (source[1] + target[1]) / 2)
        graph_ax.annotate(f"e{edge['edge_id']}", midpoint, fontsize=7)
    for detector in snapshot["detectors"]:
        detector_id = int(detector["detector_id"])
        x, y = positions[detector_id]
        graph_ax.scatter(
            [x], [y], s=48 if detector["active_local"] else 25,
            facecolors="#222222" if detector["active_local"] else "#dddddd",
            edgecolors="#222222", zorder=2,
        )
        graph_ax.annotate(str(detector_id), (x, y), xytext=(4, 4),
                          textcoords="offset points", fontsize=7)
    graph_ax.plot([], [], color="#326fa8", linewidth=2, label="candidate P")
    graph_ax.plot([], [], color="#d07724", linestyle="--", linewidth=2,
                  label="oracle difference X")
    graph_ax.plot([], [], color="#7551a6", linewidth=2.4, label="P and X")
    graph_ax.plot([], [], color="#b3443f", linestyle=":", linewidth=2.8,
                  label="P intersection R cancellation")
    graph_ax.plot([], [], color="#888888", linestyle="--", linewidth=1.7,
                  label="disconnected X")
    graph_ax.scatter([], [], color="#222222", s=48, label="active local detector")
    graph_ax.set(
        xlabel="detector coordinate x", ylabel="detector coordinate y",
        title="Authenticated local / complete-support footprint",
    )
    graph_ax.legend(fontsize=7, loc="best")
    graph_ax.set_aspect("equal", adjustable="datalim")
    caption = (
        f"Original rank 1; first safe rank {first_safe if first_safe is not None else 'none'}; "
        f"all {len(rows)} proposals were enumerated to true exhaustion. "
        f"Candidate-relevant context: {', '.join(contexts) if contexts else 'none recorded'}. "
        f"Remote disconnected roles: {', '.join(remote_roles) if remote_roles else 'none recorded'}; "
        "these are explanatory support evidence, not local candidate context. "
        f"Visibility: {', '.join(visibility) if visibility else 'none recorded'}."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150, metadata={"Software": "yoked-surface-codes"})
    plt.close(fig)


def _authenticate_inputs(
    root: Path, *, config: Mapping[str, Any], protocol_path: Path | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    corpus = load_policy_audit(root)
    supplied = json.loads(json.dumps(dict(config)))
    # Collection authentication is re-established by the corpus loader; this
    # call checks the exact protocol structure without re-running launch-time
    # clean-worktree checks during offline expansion.
    validate_policy_protocol(supplied, scientific=False)
    if corpus.config != supplied:
        raise PolicyCasebookError("supplied config differs from installed authenticated config")
    if supplied.get("frozen") is True and protocol_path is None:
        raise PolicyCasebookError("frozen expansion requires the exact protocol path")
    if protocol_path is not None:
        protocol_path = Path(protocol_path)
        if protocol_path.is_symlink():
            raise PolicyCasebookError("protocol path may not be a symlink")
        protocol = _read_json(protocol_path.resolve())
        if protocol != supplied:
            raise PolicyCasebookError("protocol bytes differ from installed config")
    if supplied.get("frozen") is True:
        repository_root = Path(__file__).resolve().parents[3]
        expected = (repository_root / str(supplied.get("protocol_relative_path"))).resolve()
        supplied_path = Path(protocol_path).resolve()  # guarded above
        if (
            supplied_path != expected
            or expected.parent != (repository_root / "docs").resolve()
            or repository_root not in expected.parents
        ):
            raise PolicyCasebookError("frozen protocol path is not the exact repository docs path")
        # Re-authenticates the current implementation/config commit, source
        # hashes, and requirements hash. COMPLETE is intentionally irrelevant.
        validate_policy_protocol(
            supplied, scientific=True, protocol_path=supplied_path,
            root=repository_root,
        )
    analysis = analyze_policy_audit(corpus)
    verify_existing_policy_analysis(corpus, analysis)
    selection_path = root / "casebook" / "selection.json"
    selection = _read_json(selection_path)
    if selection != analysis.get("casebook_selection"):
        raise PolicyCasebookError("installed selection differs from recomputed analysis")
    ready = _read_json(root / "ANALYSIS_READY")
    if ready.get("casebook_selection_sha256") != _sha(selection_path.read_bytes()):
        raise PolicyCasebookError("ANALYSIS_READY does not authenticate selection")
    return corpus, analysis, selection


def _selected_rank_one_rows(corpus: Any, selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_states = selection.get("states")
    if not isinstance(raw_states, list):
        raise PolicyCasebookError("selection states must be an array")
    selected = {str(row["state_id"]): dict(row) for row in raw_states}
    if len(selected) != len(raw_states):
        raise PolicyCasebookError("selection contains duplicate state IDs")
    for state_id, row in selected.items():
        if set(row) != {"state_id", "selection_reasons", "original_proposal_sha256"}:
            raise PolicyCasebookError("selected-state metadata has unexpected fields")
        if row["state_id"] != state_id or not isinstance(row["selection_reasons"], list):
            raise PolicyCasebookError("selected-state metadata is malformed")
    found: dict[str, dict[str, Any]] = {}
    for row in corpus.rows["counterfactuals"]:
        if row.get("operational_veto_chain_rank") != 1:
            continue
        state_id = _state_id(row)
        if state_id in selected:
            if state_id in found:
                raise PolicyCasebookError("selected state occurs more than once")
            if row.get("original_proposal_sha256") != selected[state_id].get("original_proposal_sha256"):
                raise PolicyCasebookError("selection/original proposal identity mismatch")
            found[state_id] = dict(row)
    if set(found) != set(selected):
        raise PolicyCasebookError("not every selected state exists in retained detector ledgers")
    return found


def _shot_detector_map(corpus: Any) -> dict[tuple[int, int], bytes]:
    result: dict[tuple[int, int], bytes] = {}
    for row in corpus.rows["shots"]:
        key = int(row["worker_id"]), int(row["global_shot_id"])
        if key in result:
            raise PolicyCasebookError("duplicate retained physical shot identity")
        # Deliberately extract only detector bytes.  Actual-observable and
        # correctness fields are neither copied nor passed across this boundary.
        result[key] = bytes.fromhex(str(row["packed_detectors_hex"]))
    return result


def _expansion_payloads(
    *, corpus: Any, selection: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rank_one = _selected_rank_one_rows(corpus, selection)
    shots = _shot_detector_map(corpus)
    prepared = prepare_cell(
        config["cell"], decoder_config=config["decoder"],
        dem_options=config["dem_options"], verify_hashes=bool(config.get("frozen")),
    )
    graph = prepared.compiled_pu.graph
    tolerance = OracleTolerance(**config["oracle"]["tolerance"])
    all_rows: list[dict[str, Any]] = []
    state_payloads: dict[str, bytes] = {}
    for state_id in sorted(rank_one):
        original = rank_one[state_id]
        key = int(original["worker_id"]), int(original["global_shot_id"])
        detector_bytes = shots.get(key)
        if detector_bytes is None:
            raise PolicyCasebookError("selected state has no retained detector shot")
        syndrome = np.unpackbits(
            np.frombuffer(detector_bytes, dtype=np.uint8), bitorder="little",
            count=graph.num_detectors,
        ).astype(np.uint8, copy=False)
        expanded = expand_policy_casebook_state(
            graph, syndrome,
            original_proposal_sha256=str(original["original_proposal_sha256"]),
            original_state_sha256=str(original["original_state_sha256"]),
            tolerance=tolerance,
        )
        _reject_ground_truth_keys(expanded)
        _verify_float_hex(expanded)
        for row in expanded:
            if (
                row.get("trajectory_origin") != "casebook-exhaustive"
                or row.get("original_proposal_sha256")
                != original["original_proposal_sha256"]
                or row.get("original_state_sha256") != original["original_state_sha256"]
            ):
                raise PolicyCasebookError("core expansion returned the wrong selected-state identity")
        identity = {
            name: original[name] for name in (
                "experiment_id", "cell_id", "worker_id", "worker_shot_index",
                "global_shot_id", "stim_seed", "physical_input_sha256",
            ) if name in original
        }
        rows = [
            _strip_nondeterministic_timing({
                "schema": EXPANSION_ROW_SCHEMA,
                **identity, "casebook_state_id": state_id, **row,
            })
            for row in expanded
        ]
        ranks = [row["operational_veto_chain_rank"] for row in rows]
        if ranks != list(range(1, len(rows) + 1)) or not rows:
            raise PolicyCasebookError("exhaustive ranks are not contiguous from one")
        if rows[0]["proposal_sha256"] != original["original_proposal_sha256"]:
            raise PolicyCasebookError("exhaustive rank one differs from selected original")
        if rows[-1].get("terminal_action") != "exhaustive-true-exhaustion" or rows[-1].get("exhaustion_kind") != "proposal":
            raise PolicyCasebookError("exhaustive slate did not prove true exhaustion")
        if any(row.get("casebook_state_id") != state_id for row in rows):
            raise PolicyCasebookError("casebook row identity mismatch")
        if any(
            row.get("schema") != EXPANSION_ROW_SCHEMA
            or not isinstance(row.get("graph_fingerprint"), str)
            or not isinstance(row.get("layout_fingerprint"), str)
            for row in rows
        ):
            raise PolicyCasebookError("casebook row schema or graph/layout identity is incomplete")
        support_evidence = [_support_evidence(row, graph=graph) for row in rows]
        _reject_ground_truth_keys(rows)
        _verify_float_hex(rows)
        first_safe = next(
            (row["operational_veto_chain_rank"] for row in rows if row["oracle_policy_accepts"]),
            None,
        )
        contexts = sorted({
            label for evidence in support_evidence
            for label in evidence["context_labels"]
        })
        remote_roles = sorted({
            label for evidence in support_evidence
            for label in evidence["remote_support_role_labels"]
        })
        visibility = sorted({
            label for row in rows for label in row.get("feature_visibility", {}).values()
        })
        relevant_memberships = sum(
            len(evidence["candidate_relevant_support_difference_edge_ids"])
            for evidence in support_evidence
        )
        disconnected_memberships = sum(
            len(evidence["disconnected_support_difference_edge_ids"])
            for evidence in support_evidence
        )
        cancellation_memberships = sum(
            len(evidence["support_cancellation_edge_ids"])
            for evidence in support_evidence
        )
        caption = (
            f"Original proposal is rank 1. First safe rank is "
            f"{first_safe if first_safe is not None else 'not present'}. "
            f"All {len(rows)} proposals were enumerated to true exhaustion. "
            f"Recorded candidate-relevant context: "
            f"{', '.join(contexts) if contexts else 'none'}. "
            f"Support evidence contains {relevant_memberships} candidate-relevant X, "
            f"{disconnected_memberships} disconnected X, and {cancellation_memberships} "
            "P intersection R cancellation rank-edge memberships. "
            f"Remote disconnected component roles: "
            f"{', '.join(remote_roles) if remote_roles else 'none'}; these are explanatory "
            "support evidence, not local candidate context. "
            f"Recorded visibility: {', '.join(visibility) if visibility else 'none'}."
        )
        support_snapshot = _support_graph_snapshot(
            graph, rows, graph_fingerprint=rows[0]["graph_fingerprint"],
            layout_fingerprint=rows[0]["layout_fingerprint"],
        )
        state_object = {
            "schema": EXPANSION_SCHEMA, "state_id": state_id,
            "selection_reasons": next(
                row["selection_reasons"] for row in selection["states"]
                if row["state_id"] == state_id
            ),
            "original_proposal_sha256": original["original_proposal_sha256"],
            "original_state_sha256": original["original_state_sha256"],
            "graph_fingerprint": rows[0]["graph_fingerprint"],
            "layout_fingerprint": rows[0]["layout_fingerprint"],
            "first_safe_rank": first_safe,
            "all_veto_timeline": [
                {
                    "rank": row["operational_veto_chain_rank"],
                    "stage": row["stage"],
                    "proposal_sha256": row["proposal_sha256"],
                    "oracle_policy_accepts": row["oracle_policy_accepts"],
                    "candidate_path_edge_ids": evidence["candidate_edge_ids"],
                    "support_difference_edge_ids": evidence["support_difference_edge_ids"],
                    "candidate_relevant_support_difference_edge_ids": evidence[
                        "candidate_relevant_support_difference_edge_ids"
                    ],
                    "disconnected_support_difference_edge_ids": evidence[
                        "disconnected_support_difference_edge_ids"
                    ],
                    "support_cancellation_edge_ids": evidence[
                        "support_cancellation_edge_ids"
                    ],
                    "support_difference_components": evidence["components"],
                    "disconnected_support_reconfiguration": bool(
                        evidence["disconnected_support_difference_edge_ids"]
                    ),
                    "context_labels": evidence["context_labels"],
                    "remote_support_role_labels": evidence["remote_support_role_labels"],
                }
                for row, evidence in zip(rows, support_evidence)
            ],
            "information_visibility": visibility,
            "context_labels": contexts,
            "remote_support_role_labels": remote_roles,
            "factual_caption": caption,
            "support_graph_snapshot": support_snapshot,
            "rows": rows,
        }
        _reject_ground_truth_keys(state_object)
        _verify_float_hex(state_object)
        state_payloads[f"states/{state_id}.json"] = canonical_json_bytes(state_object) + b"\n"
        all_rows.extend(rows)
    exhaustive = b"".join(canonical_json_bytes(row) + b"\n" for row in all_rows)
    return all_rows, {"exhaustive.jsonl.gz": _deterministic_gzip(exhaustive), **state_payloads}


def _verify_expansion_files(
    root: Path, *, expected_payloads: Mapping[str, bytes], experiment_id: str,
    analysis_sha256: str, selection_sha256: str,
) -> dict[str, Any]:
    expansion = root / "casebook" / "expansion"
    manifest = _read_json(expansion / "manifest.json")
    expected_fields = {
        "schema", "experiment_id", "analysis_sha256", "selection_sha256",
        "selected_states", "exhaustive_rows", "scientific_file_sha256",
        "diagram_images", "diagram_images_scientifically_digested",
        "timing_fields_removed_from_scientific_rows",
    }
    if set(manifest) != expected_fields or manifest.get("schema") != EXPANSION_MANIFEST_SCHEMA:
        raise PolicyCasebookError("expansion manifest has unexpected fields or schema")
    if (
        manifest.get("experiment_id") != experiment_id
        or manifest.get("analysis_sha256") != analysis_sha256
        or manifest.get("selection_sha256") != selection_sha256
        or manifest.get("timing_fields_removed_from_scientific_rows") is not True
    ):
        raise PolicyCasebookError("expansion manifest identity differs from authenticated inputs")
    expected_hashes = {name: _sha(data) for name, data in sorted(expected_payloads.items())}
    if manifest.get("scientific_file_sha256") != expected_hashes:
        raise PolicyCasebookError("expansion scientific hashes differ from recomputation")
    for relative, data in expected_payloads.items():
        path = expansion / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise PolicyCasebookError(f"expansion artifact differs: {relative}")
    diagrams = manifest.get("diagram_images")
    if not isinstance(diagrams, list) or manifest.get("diagram_images_scientifically_digested") is not False:
        raise PolicyCasebookError("expansion diagram manifest is invalid")
    for relative in diagrams:
        path = expansion / str(relative)
        if path.is_symlink() or not path.is_file() or path.suffix != ".png":
            raise PolicyCasebookError(f"missing or unsafe casebook diagram: {relative}")
    actual_files = {
        path.relative_to(expansion).as_posix()
        for path in expansion.rglob("*") if path.is_file()
    }
    if actual_files != set(expected_payloads) | {"manifest.json"} | set(diagrams):
        raise PolicyCasebookError("expansion contains unexpected or missing files")
    ready = _read_json(root / "EXPANSION_READY")
    expected_ready = {
        "schema": EXPANSION_READY_SCHEMA, "experiment_id": experiment_id,
        "manifest_sha256": _sha((expansion / "manifest.json").read_bytes()),
        "selected_states": manifest["selected_states"],
        "exhaustive_rows": manifest["exhaustive_rows"],
    }
    if ready != expected_ready:
        raise PolicyCasebookError("EXPANSION_READY does not authenticate expansion manifest")
    return manifest


def expand_authenticated_policy_casebook(
    root: Path, *, config: Mapping[str, Any], protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticates, exhaustively expands, and atomically installs casebook states."""
    root = Path(root).resolve()
    if (root / "COMPLETE").exists():
        raise PolicyCasebookError("cannot install expansion after COMPLETE")
    corpus, analysis, selection = _authenticate_inputs(
        root, config=config, protocol_path=protocol_path
    )
    rows, payloads = _expansion_payloads(corpus=corpus, selection=selection, config=config)
    final = root / "casebook" / "expansion"
    ready_path = root / "EXPANSION_READY"
    if final.exists() or ready_path.exists():
        if not final.is_dir() or not ready_path.is_file():
            raise PolicyCasebookError("partial existing casebook expansion")
        return _verify_expansion_files(
            root, expected_payloads=payloads, experiment_id=str(config["experiment_id"]),
            analysis_sha256=str(analysis["analysis_sha256"]),
            selection_sha256=_sha(canonical_json_bytes(selection) + b"\n"),
        )
    scratch = os.environ.get("TMPDIR")
    if not scratch:
        raise RuntimeError("TMPDIR must be set for casebook expansion")
    if os.stat(scratch).st_dev != os.stat(root).st_dev:
        raise RuntimeError("TMPDIR and corpus must share a filesystem")
    temporary = Path(tempfile.mkdtemp(prefix="promatch-casebook-expansion-", dir=scratch))
    installed = False
    try:
        for relative, data in payloads.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        diagrams = []
        by_state: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_state.setdefault(str(row["casebook_state_id"]), []).append(row)
        for state_id, state_rows in sorted(by_state.items()):
            relative = f"diagrams/{state_id}.png"
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            state_record = json.loads(payloads[f"states/{state_id}.json"])
            _render_state_diagram(state_record, path)
            diagrams.append(relative)
        manifest = {
            "schema": EXPANSION_MANIFEST_SCHEMA,
            "experiment_id": config["experiment_id"],
            "analysis_sha256": analysis["analysis_sha256"],
            "selection_sha256": _sha(canonical_json_bytes(selection) + b"\n"),
            "selected_states": len(by_state), "exhaustive_rows": len(rows),
            "scientific_file_sha256": {
                name: _sha(data) for name, data in sorted(payloads.items())
            },
            "diagram_images": diagrams,
            "diagram_images_scientifically_digested": False,
            "timing_fields_removed_from_scientific_rows": True,
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        installed = True
        ready = {
            "schema": EXPANSION_READY_SCHEMA,
            "experiment_id": config["experiment_id"],
            "manifest_sha256": _sha((final / "manifest.json").read_bytes()),
            "selected_states": len(by_state), "exhaustive_rows": len(rows),
        }
        fd, name = tempfile.mkstemp(prefix="promatch-expansion-ready-", dir=scratch)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_json_bytes(ready) + b"\n")
                stream.flush(); os.fsync(stream.fileno())
            os.replace(name, ready_path)
        finally:
            if os.path.exists(name): os.unlink(name)
        return _verify_expansion_files(
            root, expected_payloads=payloads, experiment_id=str(config["experiment_id"]),
            analysis_sha256=str(analysis["analysis_sha256"]),
            selection_sha256=_sha(canonical_json_bytes(selection) + b"\n"),
        )
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def verify_policy_casebook_expansion(
    root: Path, *, config: Mapping[str, Any], protocol_path: Path | None = None,
) -> dict[str, Any]:
    corpus, analysis, selection = _authenticate_inputs(
        Path(root).resolve(), config=config, protocol_path=protocol_path
    )
    _, payloads = _expansion_payloads(corpus=corpus, selection=selection, config=config)
    return _verify_expansion_files(
        Path(root).resolve(), expected_payloads=payloads,
        experiment_id=str(config["experiment_id"]),
        analysis_sha256=str(analysis["analysis_sha256"]),
        selection_sha256=_sha(canonical_json_bytes(selection) + b"\n"),
    )


def _fatal_gate_evidence_digest(gates: Any) -> str:
    if not isinstance(gates, list) or len(gates) != 18 or any(
        not isinstance(gate, Mapping) for gate in gates
    ):
        raise PolicyCasebookError("not every frozen fatal gate passed authentication")
    gate_numbers = [gate.get("gate") for gate in gates]
    if sorted(gate_numbers) != list(range(1, 19)) or len(set(gate_numbers)) != 18:
        raise PolicyCasebookError("fatal gate numbers are missing or duplicated")
    recognized = {"passed-ledger-recomputed", "collector-attested"}
    for gate in gates:
        evidence = gate.get("evidence")
        evidence_nonempty = (
            isinstance(evidence, str) and bool(evidence.strip())
        ) or (isinstance(evidence, Mapping) and bool(evidence)) or (
            isinstance(evidence, list) and bool(evidence)
        )
        if gate.get("status") not in recognized or not evidence_nonempty:
            raise PolicyCasebookError("fatal gate status/evidence is incomplete")
    return _sha(canonical_json_bytes(gates))


def finalize_policy_audit(
    root: Path, *, config: Mapping[str, Any], protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Re-verifies every layer and writes authenticated COMPLETE last."""
    root = Path(root).resolve()
    corpus, analysis, _ = _authenticate_inputs(root, config=config, protocol_path=protocol_path)
    expansion = verify_policy_casebook_expansion(
        root, config=config, protocol_path=protocol_path
    )
    gates = analysis.get("tables", {}).get("fatal_gates")
    gate_evidence_sha256 = _fatal_gate_evidence_digest(gates)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "experiment_id": config["experiment_id"],
        "collection_ready_sha256": _sha((root / "COLLECTION_READY").read_bytes()),
        "analysis_ready_sha256": _sha((root / "ANALYSIS_READY").read_bytes()),
        "expansion_ready_sha256": _sha((root / "EXPANSION_READY").read_bytes()),
        "collection_manifest_sha256": _sha((root / "manifest.json").read_bytes()),
        "analysis_manifest_sha256": _sha((root / "analysis" / "manifest.json").read_bytes()),
        "expansion_manifest_sha256": _sha((root / "casebook" / "expansion" / "manifest.json").read_bytes()),
        "analysis_sha256": analysis["analysis_sha256"],
        "expansion_selected_states": expansion["selected_states"],
        "fatal_gates_verified": 18,
        "fatal_gate_evidence_sha256": gate_evidence_sha256,
    }
    data = canonical_json_bytes(complete) + b"\n"
    target = root / "COMPLETE"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != data:
            raise PolicyCasebookError("existing COMPLETE differs from authenticated finalization")
        return complete
    scratch = os.environ.get("TMPDIR")
    if not scratch:
        raise RuntimeError("TMPDIR must be set for finalization")
    if os.stat(scratch).st_dev != os.stat(root).st_dev:
        raise RuntimeError("TMPDIR and corpus must share a filesystem")
    fd, name = tempfile.mkstemp(prefix="promatch-complete-", dir=scratch)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name): os.unlink(name)
    installed = _read_json(target)
    if installed != complete or _sha(canonical_json_bytes(gates)) != installed.get(
        "fatal_gate_evidence_sha256"
    ):
        raise PolicyCasebookError("written COMPLETE failed exact verification")
    return complete
