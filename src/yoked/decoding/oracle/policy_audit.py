"""Ground-truth-free, shot-level B1 policy audit.

The public entry point deliberately accepts only a compiled graph and detector
syndrome.  In particular, sampled observables cannot enter candidate selection
or oracle certification through this API.

The module follows one shot through four layers: reconstruct the frozen V3
shadow trajectory, score its proposals, enumerate unchanged-state
counterfactuals for unsafe choices, and summarize the three sequential oracle
arms. It returns plain records; collection and artifact I/O live in
:mod:`yoked.decoding.oracle.policy_experiment`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from yoked.decoding._promatch import (
    CommitProposal,
    DomainProposalStepper,
    PrematchedPath,
    apply_detector_boundary,
    predecode,
)
from yoked.decoding._promatch_graph import CompiledPromatchGraph
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1TerminalDetector,
    L1WindowDomain,
    YokeDetector,
)
from yoked.decoding.oracle.full_graph import (
    FullGraphOracle,
    OracleEvaluation,
    OracleTolerance,
)
from yoked.decoding.oracle.replay import run_oracle_trajectory


ARM_IDS = (
    "u0-joint-y2",
    "pu-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2-shadow",
    "pu-ocost-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2",
    "pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2",
    "pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2",
)
# The V3 shadow arm whose trajectory this module reconstructs and audits.
SHADOW_ARM_ID = ARM_IDS[1]
# Frozen V3 shadow predecoder configuration, mirroring the "decoder" contract
# of docs/PROMATCH_POLICY_AUDIT_20K_FROZEN_V2.json in predecode/stepper
# keyword form ("edge-zero" is the normalized alias of the protocol's
# "zero-frame" observable policy).  Every replay, stepper, and oracle
# trajectory in this module must use exactly this configuration.
FROZEN_V3_DECODER_KWARGS: Mapping[str, Any] = MappingProxyType(
    {
        "residual_hw_limit": 10,
        "boundary_policy": "disabled",
        "observable_policy": "edge-zero",
    }
)
# Sync note: yoked.decoding.oracle.policy_analysis.CONTEXT_PRIORITY is a
# deliberate twin.  That module is downstream-only and must not import this
# decoding-adjacent producer, so the tuple is duplicated; keep both copies
# and their most-specific-first ordering (first match wins for the exclusive
# context) identical.
CONTEXT_PRIORITY = (
    "yoke",
    "true-boundary",
    "terminal",
    "cross-window",
    "cross-patch-or-basis",
    "support-cancellation",
    "in-domain",
)
_SPECIFIC_CONTEXT = frozenset(CONTEXT_PRIORITY[:-1])
_STATIC_GRAPH_CACHE: dict[
    tuple[Any, ...],
    tuple[dict[Any, dict[int, float]], dict[tuple[Any, int, int], float | None]],
] = {}


@dataclasses.dataclass(frozen=True)
class _CounterfactualCandidate:
    """One scored entry in an unchanged-state counterfactual veto chain."""

    proposal: CommitProposal
    evaluation: OracleEvaluation
    proposal_id: str
    evaluation_metadata: dict[str, Any]
    candidate_enumeration_ns: int
    stage3_enumeration_ns: int


def _static_graph_metadata(
    graph: CompiledPromatchGraph,
) -> tuple[dict[Any, dict[int, float]], dict[tuple[Any, int, int], float | None]]:
    key = (
        graph.fingerprint,
        graph.num_detectors,
        len(graph.edges),
        tuple((edge.source, edge.target, edge.weight.hex()) for edge in graph.edges),
        tuple(
            (repr(domain), tuple(domain_graph.detector_ids))
            for domain, domain_graph in sorted(
                graph.domain_graphs.items(), key=lambda item: repr(item[0])
            )
        ),
    )
    cached = _STATIC_GRAPH_CACHE.get(key)
    if cached is None:
        cached = (
            {
                domain: _static_boundary_distances(graph, domain)
                for domain in graph.domain_graphs
            },
            {},
        )
        _STATIC_GRAPH_CACHE[key] = cached
    return cached


def _normalized_labels(labels: Sequence[str] | set[str]) -> list[str]:
    result = set(labels)
    if result.intersection(_SPECIFIC_CONTEXT):
        result.discard("in-domain")
    return sorted(result)


def _mask_bits(mask: bytes, count: int) -> np.ndarray:
    if not count:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(
        np.frombuffer(mask, dtype=np.uint8), bitorder="little", count=count
    )


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _domain_json(domain: Any) -> dict[str, Any]:
    result = {"patch_id": domain.patch_id, "basis": domain.check_basis}
    result["window_id"] = (
        domain.window_id if isinstance(domain, L1WindowDomain) else None
    )
    return result


def _proposal_signature(proposal: CommitProposal) -> list[Any]:
    return [proposal.stage, list(proposal.endpoints), list(proposal.edge_ids)]


def _path_signature(path: PrematchedPath) -> list[Any]:
    return [path.stage, list(path.endpoints), list(path.edge_ids)]


def _proposal_id(domain: Any, proposal: CommitProposal, state: str) -> str:
    return _sha([_domain_json(domain), state, _proposal_signature(proposal)])


def _state_fingerprint(
    graph: CompiledPromatchGraph, syndrome: np.ndarray, frame: bytes
) -> str:
    return hashlib.sha256(
        b"promatch-policy-complete-state-v1\0"
        + graph.fingerprint.encode()
        + np.packbits(syndrome, bitorder="little").tobytes()
        + frame
    ).hexdigest()


def _role_tags(
    graph: CompiledPromatchGraph, detector: int | None, domain: Any
) -> set[str]:
    if detector is None:
        return {"true-boundary"}
    role = graph.layout.role_of(detector)
    if isinstance(role, YokeDetector):
        return {"yoke"}
    if isinstance(role, L1TerminalDetector):
        tags = {"terminal"}
        if role.patch_id != domain.patch_id or role.check_basis != domain.check_basis:
            tags.add("cross-patch-or-basis")
        return tags
    assert isinstance(role, L1BodyDetector)
    tags: set[str] = set()
    if role.patch_id != domain.patch_id or role.check_basis != domain.check_basis:
        tags.add("cross-patch-or-basis")
    elif isinstance(domain, L1WindowDomain) and role.window_id != domain.window_id:
        tags.add("cross-window")
    else:
        tags.add("in-domain")
    return tags


def _support_component_labels(
    graph: CompiledPromatchGraph,
    edge_ids: Sequence[int],
    endpoints: Sequence[int | None],
    domain: Any,
) -> list[str]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    boundary_sources: set[int] = set()
    for edge_id in edge_ids:
        edge = graph.edges[int(edge_id)]
        if edge.target is None:
            boundary_sources.add(edge.source)
        else:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)
    seen: set[int] = set()
    present = set(adjacency) | boundary_sources
    queue = deque(int(v) for v in endpoints if v is not None and int(v) in present)
    labels: set[str] = set()
    while queue:
        vertex = queue.popleft()
        if vertex in seen:
            continue
        seen.add(vertex)
        labels.update(_role_tags(graph, vertex, domain))
        if vertex in boundary_sources:
            labels.add("true-boundary")
        queue.extend(adjacency.get(vertex, ()))
    return _normalized_labels(labels)


def _matched_partner_labels(
    graph: CompiledPromatchGraph,
    syndrome: np.ndarray,
    endpoints: Sequence[int | None],
    domain: Any,
    *,
    cache: dict[bytes, tuple[list[list[int | None]], dict[int, int | None]]]
    | None = None,
    telemetry: dict[str, int] | None = None,
) -> tuple[list[list[int | None]], list[str], dict[str, Any]]:
    def result(
        pairs: list[list[int | None]],
        partner: Mapping[int, int | None],
        *,
        cache_hit: bool,
        backend_ns: int,
    ) -> tuple[list[list[int | None]], list[str], dict[str, Any]]:
        labels: set[str] = set()
        for endpoint in endpoints:
            if endpoint is not None and endpoint in partner:
                labels.update(_role_tags(graph, partner[endpoint], domain))
        return (
            pairs,
            _normalized_labels(labels),
            {
                "matched_backend_cache_hit": cache_hit,
                "matched_backend_call_count_delta": 0 if cache_hit else 1,
                "matched_backend_wall_ns": backend_ns,
            },
        )

    key = np.packbits(syndrome, bitorder="little").tobytes()
    cached = None if cache is None else cache.get(key)
    backend_ns = 0
    if cached is None:
        start_ns = time.perf_counter_ns()
        raw = np.asarray(graph.matcher.decode_to_matched_dets_array(syndrome))
        backend_ns = time.perf_counter_ns() - start_ns
        if telemetry is not None:
            telemetry["backend_calls"] = telemetry.get("backend_calls", 0) + 1
            telemetry["backend_wall_ns"] = (
                telemetry.get("backend_wall_ns", 0) + backend_ns
            )
    else:
        pairs, partner = cached
        return result(pairs, partner, cache_hit=True, backend_ns=0)
    if raw.size == 0:
        raw = np.empty((0, 2), dtype=np.int64)
    pairs: list[list[int | None]] = []
    partner: dict[int, int | None] = {}
    active = set(int(v) for v in np.flatnonzero(syndrome))
    paired_active: set[int] = set()
    for a0, b0 in raw:
        a, b = int(a0), int(b0)
        if a < -1 or b < -1 or a >= graph.num_detectors or b >= graph.num_detectors:
            raise AssertionError("matched-active pair contains an invalid detector ID")
        if a == -1 and b == -1:
            raise AssertionError("matched-active pair cannot contain two boundaries")
        av, bv = (None if a == -1 else a), (None if b == -1 else b)
        for value in (av, bv):
            if value is not None:
                if value not in active:
                    raise AssertionError(
                        "matched-active pair contains an inactive detector"
                    )
                if value in paired_active:
                    raise AssertionError(
                        "active detector occurs in multiple matched pairs"
                    )
                paired_active.add(value)
        pair = [av, bv]
        if av is None or (bv is not None and bv < av):
            pair = [bv, av]
        pairs.append(pair)
        if av is not None:
            partner[av] = bv
        if bv is not None:
            partner[bv] = av
    if paired_active != active:
        raise AssertionError(
            "matched-active pairs do not cover the syndrome exactly once"
        )
    pairs.sort(
        key=lambda pair: (pair[0], graph.num_detectors if pair[1] is None else pair[1])
    )
    if cache is not None:
        cache[key] = pairs, partner
    return result(pairs, partner, cache_hit=False, backend_ns=backend_ns)


def _difference_components(
    graph: CompiledPromatchGraph, evaluation: OracleEvaluation, domain: Any
) -> tuple[list[dict[str, Any]], list[str], bool]:
    candidate = set(evaluation.candidate_edge_ids)
    residual = set(evaluation.residual_support_edge_ids)
    cancellations = candidate.intersection(residual)
    ids = set(evaluation.base_support_edge_ids)
    ids.symmetric_difference_update(evaluation.candidate_edge_ids)
    ids.symmetric_difference_update(evaluation.residual_support_edge_ids)
    if not ids and not evaluation.frame_compatible:
        raise AssertionError(
            "X-empty support decomposition cannot have a different observable frame"
        )
    by_vertex: dict[int, set[int]] = defaultdict(set)
    for edge_id in ids:
        edge = graph.edges[edge_id]
        by_vertex[edge.source].add(edge_id)
        if edge.target is not None:
            by_vertex[edge.target].add(edge_id)
    remaining = set(ids)
    components: list[dict[str, Any]] = []
    candidate_context_labels: set[str] = set()
    has_disconnected_reconfiguration = False
    candidate_boundary = set(evaluation.candidate_boundary_detector_ids)
    while remaining:
        seed = min(remaining)
        component: set[int] = set()
        queue = [seed]
        while queue:
            edge_id = queue.pop()
            if edge_id not in remaining:
                continue
            remaining.remove(edge_id)
            component.add(edge_id)
            edge = graph.edges[edge_id]
            for vertex in (edge.source, edge.target):
                if vertex is not None:
                    queue.extend(by_vertex[vertex])
        component_vertices: set[int] = set()
        for edge_id in component:
            edge = graph.edges[edge_id]
            component_vertices.add(edge.source)
            if edge.target is not None:
                component_vertices.add(edge.target)
        relevance_reasons: list[str] = []
        if component.intersection(candidate):
            relevance_reasons.append("candidate-support-edge")
        if component_vertices.intersection(candidate_boundary):
            relevance_reasons.append("candidate-boundary-detector")
        relevance_reasons.sort()
        candidate_relevant = bool(relevance_reasons)
        labels = set(_support_component_labels(graph, sorted(component), (), domain))
        # With no seed endpoints, explicitly classify every component vertex.
        for edge_id in component:
            edge = graph.edges[edge_id]
            labels.update(_role_tags(graph, edge.source, domain))
            labels.update(_role_tags(graph, edge.target, domain))
        normalized = _normalized_labels(labels)
        if candidate_relevant:
            candidate_context_labels.update(normalized)
        else:
            has_disconnected_reconfiguration = True
        components.append(
            {
                "certificate_kind": "real-x-component",
                "canonical_edge_ids": sorted(component),
                "support_cancellation_edge_ids": [],
                "component_detector_ids": sorted(component_vertices),
                "candidate_support_witness_edge_ids": sorted(
                    component.intersection(candidate)
                ),
                "candidate_boundary_witness_detector_ids": sorted(
                    component_vertices.intersection(candidate_boundary)
                ),
                "labels": normalized,
                "candidate_relevant": candidate_relevant,
                "candidate_relevance_reasons": relevance_reasons,
            }
        )
    components.sort(key=lambda row: row["canonical_edge_ids"])

    # P intersection R is an independent certificate: these edges disappear
    # from Q=P xor R but are paid twice by the forced composite.  It must not
    # be attached to a geometrically unrelated X component.
    if cancellations:
        cancellation_labels: set[str] = {"support-cancellation"}
        for edge_id in cancellations:
            edge = graph.edges[edge_id]
            cancellation_labels.update(_role_tags(graph, edge.source, domain))
            cancellation_labels.update(_role_tags(graph, edge.target, domain))
        normalized = _normalized_labels(cancellation_labels)
        candidate_context_labels.update(normalized)
        components.append(
            {
                "certificate_kind": "support-cancellation",
                "canonical_edge_ids": [],
                "support_cancellation_edge_ids": sorted(cancellations),
                "component_detector_ids": [],
                "candidate_support_witness_edge_ids": [],
                "candidate_boundary_witness_detector_ids": [],
                "labels": normalized,
                "candidate_relevant": True,
                "candidate_relevance_reasons": [
                    "candidate-residual-support-cancellation"
                ],
            }
        )
    unsafe = not evaluation.accepted
    if unsafe and not components:
        raise AssertionError(
            "unsafe proposal has neither an X support difference nor a support-cancellation certificate"
        )
    return (
        components,
        _normalized_labels(candidate_context_labels),
        has_disconnected_reconfiguration,
    )


def _evaluation_fields(
    graph: CompiledPromatchGraph,
    syndrome: np.ndarray,
    domain: Any,
    proposal: CommitProposal,
    evaluation: OracleEvaluation,
    oracle_meta: Mapping[str, Any] | None = None,
    matched_pair_cache: dict[
        bytes, tuple[list[list[int | None]], dict[int, int | None]]
    ]
    | None = None,
    matched_telemetry: dict[str, int] | None = None,
) -> dict[str, Any]:
    classification_start_ns = time.perf_counter_ns()
    pairs, matched, matched_meta = _matched_partner_labels(
        graph,
        syndrome,
        proposal.endpoints,
        domain,
        cache=matched_pair_cache,
        telemetry=matched_telemetry,
    )
    support_path = _support_component_labels(
        graph, evaluation.base_support_edge_ids, proposal.endpoints, domain
    )
    components, difference, disconnected_reconfiguration = _difference_components(
        graph, evaluation, domain
    )
    support_cancellation_ids = sorted(
        set(evaluation.candidate_edge_ids).intersection(
            evaluation.residual_support_edge_ids
        )
    )
    exclusive = next((tag for tag in CONTEXT_PRIORITY if tag in difference), None)
    omitted = _normalized_labels(set(matched) | set(support_path))
    same_pair = (
        len(proposal.endpoints) == 2
        and proposal.endpoints[0] is not None
        and any(set(pair) == set(proposal.endpoints) for pair in pairs)
    )
    diagnostics: set[str] = set()
    if same_pair and (
        set(evaluation.base_support_edge_ids) != set(evaluation.candidate_edge_ids)
        or evaluation.base_frame != evaluation.candidate_composite_frame
    ):
        diagnostics.add("same-pair-different-path-or-frame")
    if evaluation.cost_compatible and not evaluation.frame_compatible:
        diagnostics.add("equal-weight-logical-class")
    b_support = list(evaluation.base_support_edge_ids)
    p_support = list(evaluation.candidate_edge_ids)
    r_support = list(evaluation.residual_support_edge_ids)
    if any(
        len(values) != len(set(values)) for values in (b_support, p_support, r_support)
    ):
        raise AssertionError("oracle B/P/R supports must be square-free")
    q_support = sorted(set(p_support).symmetric_difference(r_support))
    x_support = sorted(set(b_support).symmetric_difference(q_support))
    component_union = sorted(
        edge_id
        for component in components
        if component["certificate_kind"] == "real-x-component"
        for edge_id in component["canonical_edge_ids"]
    )
    if component_union != x_support:
        raise AssertionError(
            "real support-difference components do not partition X exactly"
        )
    if disconnected_reconfiguration:
        diagnostics.add("disconnected-support-reconfiguration")
    if not evaluation.accepted and not diagnostics and not difference:
        diagnostics.add("unclassified")
    degeneracy = sorted(diagnostics)
    result = {
        "cost_compatible": evaluation.cost_compatible,
        "frame_compatible": evaluation.frame_compatible,
        "oracle_policy_accepts": evaluation.accepted,
        "cost_classification": evaluation.cost_classification.value,
        "cost_excess": evaluation.cost_excess,
        "cost_excess_hex": evaluation.cost_excess.hex(),
        "tau_k": evaluation.tau_k,
        "tau_k_hex": evaluation.tau_k.hex(),
        "composite_weight": evaluation.composite_weight,
        "composite_weight_hex": evaluation.composite_weight.hex(),
        "base_support_weight": evaluation.base_support_weight,
        "base_support_weight_hex": evaluation.base_support_weight.hex(),
        "residual_support_weight": evaluation.residual_support_weight,
        "residual_support_weight_hex": evaluation.residual_support_weight.hex(),
        "base_backend_weight": evaluation.base_backend_weight,
        "base_backend_weight_hex": evaluation.base_backend_weight.hex(),
        "residual_backend_weight": evaluation.residual_backend_weight,
        "residual_backend_weight_hex": evaluation.residual_backend_weight.hex(),
        "base_tau_weight": evaluation.base_tau_weight,
        "base_tau_weight_hex": evaluation.base_tau_weight.hex(),
        "residual_tau_weight": evaluation.residual_tau_weight,
        "residual_tau_weight_hex": evaluation.residual_tau_weight.hex(),
        "candidate_weight": evaluation.candidate_weight,
        "candidate_weight_hex": evaluation.candidate_weight.hex(),
        "base_support_edge_ids": list(evaluation.base_support_edge_ids),
        "residual_support_edge_ids": list(evaluation.residual_support_edge_ids),
        "candidate_support_edge_ids": list(evaluation.candidate_edge_ids),
        "B_base_support_edge_ids": b_support,
        "P_candidate_support_edge_ids": p_support,
        "R_residual_support_edge_ids": r_support,
        "Q_forced_parity_support_edge_ids": q_support,
        "X_support_difference_edge_ids": x_support,
        "P_intersection_R_edge_ids": support_cancellation_ids,
        "supports_square_free": True,
        "B_base_support_square_free": True,
        "P_candidate_support_square_free": True,
        "R_residual_support_square_free": True,
        "Q_forced_parity_support_square_free": True,
        "X_support_difference_square_free": True,
        "base_frame": evaluation.base_frame.hex(),
        "candidate_frame": evaluation.candidate_composite_frame.hex(),
        "base_matched_active_pairs": pairs,
        "matched_partner_labels": matched,
        "support_path_labels": support_path,
        "support_difference_components": components,
        "support_difference_component_labels": difference,
        "support_cancellation_edge_ids": support_cancellation_ids,
        "exclusive_support_component_context": exclusive,
        "omitted_context_labels": omitted,
        "degeneracy_diagnostics": degeneracy,
        "support_difference_representation_version": "promatch-support-difference-v2",
        "disconnected_support_reconfiguration": disconnected_reconfiguration,
        "same_pair_different_path_or_frame": "same-pair-different-path-or-frame"
        in diagnostics,
        "equal_weight_logical_class": "equal-weight-logical-class" in diagnostics,
        "degeneracy_unclassified": "unclassified" in diagnostics,
    }
    if oracle_meta is not None:
        result.update(oracle_meta)
    result.update(matched_meta)
    result["support_classification_wall_ns"] = (
        time.perf_counter_ns() - classification_start_ns
    )
    return result


def _oracle_evaluate(
    oracle: FullGraphOracle,
    *,
    syndrome: np.ndarray,
    frame: bytes,
    proposal: CommitProposal,
    call_ordinal: int,
) -> tuple[OracleEvaluation, dict[str, Any]]:
    before = oracle.cache_stats
    evaluation = oracle.evaluate(
        syndrome=syndrome,
        accumulated_frame=_mask_bits(frame, oracle.graph.num_observables),
        candidate_edge_ids=proposal.edge_ids,
        policy="frame",
    )
    after = oracle.cache_stats
    base_solution_id = _sha(
        [
            evaluation.base_support_edge_ids,
            evaluation.base_prediction.hex(),
            evaluation.base_support_weight.hex(),
            evaluation.base_backend_weight.hex(),
        ]
    )
    residual_solution_id = _sha(
        [
            evaluation.residual_support_edge_ids,
            evaluation.residual_prediction.hex(),
            evaluation.residual_support_weight.hex(),
            evaluation.residual_backend_weight.hex(),
        ]
    )
    return evaluation, {
        "oracle_call_ordinal": call_ordinal,
        "oracle_cache_hits_before": before.hits,
        "oracle_cache_hits_after": after.hits,
        "oracle_cache_misses_before": before.misses,
        "oracle_cache_misses_after": after.misses,
        "oracle_cache_hit_delta": after.hits - before.hits,
        "oracle_cache_miss_delta": after.misses - before.misses,
        "oracle_base_solution_id": base_solution_id,
        "oracle_residual_solution_id": residual_solution_id,
        "oracle_evaluation_id": _sha(
            [
                call_ordinal,
                base_solution_id,
                residual_solution_id,
                proposal.edge_ids,
                evaluation.cost_excess.hex(),
                evaluation.candidate_composite_frame.hex(),
            ]
        ),
    }


def _proposal_fields(proposal: CommitProposal) -> dict[str, Any]:
    return {
        "proposal_signature": _proposal_signature(proposal),
        "stage": proposal.stage,
        "within_state_stage_rank": proposal.within_state_stage_rank,
        "state_stage_candidate_count": proposal.state_stage_candidate_count,
        "state_total_candidate_count": proposal.state_total_candidate_count,
        "state_veto_count_before": proposal.state_veto_count_before,
        "ordered_endpoints": list(proposal.endpoints),
        "canonical_edge_ids": list(proposal.edge_ids),
        "canonical_edge_count": len(proposal.edge_ids),
        "detector_boundary_ids": list(proposal.detector_boundary),
        "observable_frame": proposal.observable_frame.hex(),
        "decision_weight": proposal.decision_weight,
        "decision_weight_hex": proposal.decision_weight.hex(),
        "canonical_path_weight": proposal.decision_weight,
        "canonical_path_weight_hex": proposal.decision_weight.hex(),
        "path_weight_agreement": True,
        "events_removed_if_committed": len(proposal.detector_boundary),
    }


def _static_boundary_distances(
    graph: CompiledPromatchGraph, domain: Any
) -> dict[int, float]:
    domain_ids = set(graph.domain_graphs[domain].detector_ids)
    adjacency: dict[int, list[tuple[int | None, float]]] = defaultdict(list)
    best: dict[int, float] = {}
    queue: list[tuple[float, int]] = []
    for edge in graph.edges:
        if edge.source not in domain_ids or (
            edge.target is not None and edge.target not in domain_ids
        ):
            continue
        if edge.target is None:
            if edge.weight < best.get(edge.source, math.inf):
                best[edge.source] = edge.weight
                heapq.heappush(queue, (edge.weight, edge.source))
        else:
            adjacency[edge.source].append((edge.target, edge.weight))
            adjacency[edge.target].append((edge.source, edge.weight))
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != best[vertex]:
            continue
        for other, weight in adjacency[vertex]:
            candidate = distance + weight
            if candidate < best.get(other, math.inf):
                best[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return best


def _static_distance_between(
    graph: CompiledPromatchGraph,
    source: int,
    target: int,
    *,
    allowed_edge_ids: set[int] | None = None,
) -> float | None:
    if source == target:
        return 0.0
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in graph.edges:
        if allowed_edge_ids is not None and edge.edge_id not in allowed_edge_ids:
            continue
        if edge.target is not None:
            adjacency[edge.source].append((edge.target, edge.weight))
            adjacency[edge.target].append((edge.source, edge.weight))
    queue = [(0.0, source)]
    best = {source: 0.0}
    while queue:
        distance, vertex = heapq.heappop(queue)
        if vertex == target:
            return distance
        if distance != best[vertex]:
            continue
        for other, weight in adjacency[vertex]:
            candidate = distance + weight
            if candidate < best.get(other, math.inf):
                best[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return None


def _local_policy_fields(
    graph: CompiledPromatchGraph,
    domain: Any,
    proposal: CommitProposal,
    active: Sequence[int],
    boundary_distances: Mapping[int, float],
    partner_distance_cache: dict[tuple[Any, int, int], float | None],
) -> dict[str, Any]:
    domain_graph = graph.domain_graphs[domain]
    active_set = set(active)
    endpoints = [v for v in proposal.endpoints if v is not None]
    endpoint_rows = []
    offsets: list[int] = []
    endpoint_boundary_distances: list[float] = []
    for detector in endpoints:
        role = graph.layout.role_of(detector)
        first = last = False
        offset = start_offset = end_offset = terminal_offset = None
        if isinstance(role, L1BodyDetector) and isinstance(domain, L1WindowDomain):
            within = role.time - domain.window_id * graph.layout.distance
            start_offset = within
            end_offset = graph.layout.distance - 1 - within
            offset = min(start_offset, end_offset)
            offsets.append(offset)
            first, last = within == 0, within == graph.layout.distance - 1
            terminal_offset = graph.layout.rounds - role.time
        boundary_distance = boundary_distances.get(detector)
        if boundary_distance is not None:
            endpoint_boundary_distances.append(boundary_distance)
        endpoint_rows.append(
            {
                "detector_id": detector,
                "local_degree": len(domain_graph.adjacency.get(detector, ())),
                "active_neighbor_count": sum(
                    v in active_set for v in domain_graph.neighbors.get(detector, ())
                ),
                "eligible_incident_candidate_count": sum(
                    v in active_set for v in domain_graph.neighbors.get(detector, ())
                ),
                "first_window_round": first,
                "last_window_round": last,
                "window_offset": offset,
                "window_start_offset": start_offset,
                "window_end_offset": end_offset,
                "circuit_terminal_offset": terminal_offset,
                "static_weighted_distance_to_true_boundary": boundary_distance,
                "static_weighted_distance_to_true_boundary_hex": (
                    None if boundary_distance is None else boundary_distance.hex()
                ),
            }
        )
    minimum = min(endpoint_boundary_distances) if endpoint_boundary_distances else None
    partner_distance = None
    if len(endpoints) == 2:
        partner_key = (domain, min(endpoints), max(endpoints))
        if partner_key not in partner_distance_cache:
            partner_distance_cache[partner_key] = _static_distance_between(
                graph,
                endpoints[0],
                endpoints[1],
                allowed_edge_ids={edge.edge_id for edge in domain_graph.edges},
            )
        partner_distance = partner_distance_cache[partner_key]
    return {
        "candidate_multiplicity": proposal.state_total_candidate_count,
        "window_offset": min(offsets) if offsets else None,
        "window_round_start": (
            domain.window_id * graph.layout.distance
            if isinstance(domain, L1WindowDomain)
            else None
        ),
        "window_round_stop": (
            (domain.window_id + 1) * graph.layout.distance
            if isinstance(domain, L1WindowDomain)
            else None
        ),
        "endpoint_local_features": endpoint_rows,
        "either_endpoint_first_window_round": any(
            row["first_window_round"] for row in endpoint_rows
        ),
        "either_endpoint_last_window_round": any(
            row["last_window_round"] for row in endpoint_rows
        ),
        "minimum_static_weighted_distance_to_true_boundary": minimum,
        "minimum_static_weighted_distance_to_true_boundary_hex": None
        if minimum is None
        else minimum.hex(),
        "static_boundary_competition": minimum is not None
        and minimum <= proposal.decision_weight,
        "static_candidate_partner_distance": partner_distance,
        "static_candidate_partner_distance_hex": (
            None if partner_distance is None else partner_distance.hex()
        ),
        "stage_isolation_predicate": proposal.stage == 1,
        "stage_safety_predicate": proposal.stage == 2,
        "stage_singleton_predicate": proposal.stage == 3,
        "feature_visibility": {
            "candidate_multiplicity": "L1-local-dynamic",
            "window_offset": "L1-local-dynamic",
            "endpoint_local_features": "L1-local-dynamic",
            "first_last_window_round": "L1-local-dynamic",
            "window_start_end_offsets": "L1-local-dynamic",
            "circuit_terminal_offset": "L1-static-boundary",
            "stage_isolation_safety_singleton_predicates": "L1-local-dynamic",
            "static_candidate_partner_distance": "L1-local-dynamic",
            "static_boundary_competition": "L1-static-boundary",
            "minimum_static_weighted_distance_to_true_boundary": "L1-static-boundary",
        },
    }


def _new_stepper(
    graph: CompiledPromatchGraph, domain: Any, active: Sequence[int]
) -> DomainProposalStepper:
    return DomainProposalStepper(
        graph.domain_graphs[domain],
        active,
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        veto_budget=None,
        **FROZEN_V3_DECODER_KWARGS,
    )


def _trajectory_summary(result: Any) -> dict[str, Any]:
    provisional = sum(
        len(record.outcome.initial_active) - len(record.outcome.provisional_active)
        for record in result.domains
    )
    durable = sum(
        len(record.outcome.initial_active) - len(record.outcome.durable_active)
        for record in result.domains
    )
    return {
        "initial_detector_hw": sum(result.initial_syndrome),
        "final_residual_detector_hw": sum(result.final_residual_syndrome),
        "proposals_attempted": len(result.proposals),
        "durable_commitments": sum(record.durable for record in result.proposals),
        "provisional_events_removed": provisional,
        "durable_events_removed": durable,
        "events_lost_to_rollback": provisional - durable,
    }


def _layout_fingerprint(graph: CompiledPromatchGraph) -> str:
    return getattr(graph.layout, "fingerprint", "synthetic-layout")


def _provenance_fields(
    graph: CompiledPromatchGraph, *, trajectory_origin: str, arm_id: str
) -> dict[str, Any]:
    """Shared trajectory/arm/graph/layout provenance quad for every row."""

    return {
        "trajectory_origin": trajectory_origin,
        "arm_id": arm_id,
        "graph_fingerprint": graph.fingerprint,
        "layout_fingerprint": _layout_fingerprint(graph),
    }


class _ShotAuditState:
    """Mutable bookkeeping threaded through one shot's shadow replay.

    Owns the oracle (with its call ordinal), the matched-pair and static-graph
    caches, the residual/frame trajectory state, and the three output ledgers.
    Splitting the replay into stage helpers keeps each loop readable without
    changing any evaluation order.
    """

    def __init__(
        self, graph: CompiledPromatchGraph, tolerance: OracleTolerance
    ) -> None:
        self.graph = graph
        self.oracle = FullGraphOracle(graph, tolerance=tolerance)
        self.oracle_call_ordinal = 0
        self.matched_pair_cache: dict[
            bytes, tuple[list[list[int | None]], dict[int, int | None]]
        ] = {}
        self.matched_telemetry: dict[str, int] = {
            "backend_calls": 0,
            "backend_wall_ns": 0,
        }
        boundary_distances, partner_distances = _static_graph_metadata(graph)
        self.boundary_distances_by_domain = boundary_distances
        self.partner_distance_cache = partner_distances
        self.residual: np.ndarray | None = None
        self.frame = b""
        self.proposal_rows: list[dict[str, Any]] = []
        self.counterfactual_rows: list[dict[str, Any]] = []
        self.domain_rows: list[dict[str, Any]] = []
        self.commit_index = 0

    def evaluate(self, proposal: CommitProposal) -> tuple[OracleEvaluation, dict]:
        """Scores one proposal against the current residual/frame state."""

        evaluation, metadata = _oracle_evaluate(
            self.oracle,
            syndrome=self.residual,
            frame=self.frame,
            proposal=proposal,
            call_ordinal=self.oracle_call_ordinal,
        )
        self.oracle_call_ordinal += 1
        return evaluation, metadata

    def scored_candidate_fields(
        self,
        domain: Any,
        proposal: CommitProposal,
        evaluation: OracleEvaluation,
        evaluation_meta: Mapping[str, Any],
        *,
        active: Sequence[int],
    ) -> dict[str, Any]:
        """One scored candidate's shared domain/proposal/local/oracle fields."""

        return {
            **_domain_json(domain),
            **_proposal_fields(proposal),
            **_local_policy_fields(
                self.graph,
                domain,
                proposal,
                active,
                self.boundary_distances_by_domain[domain],
                self.partner_distance_cache,
            ),
            **_evaluation_fields(
                self.graph,
                self.residual,
                domain,
                proposal,
                evaluation,
                evaluation_meta,
                self.matched_pair_cache,
                self.matched_telemetry,
            ),
        }


def _reconstruct_reference_arms(
    state: _ShotAuditState, original: np.ndarray
) -> tuple[Any, Any, bytes, bytes, dict[str, Any]]:
    """Decodes U0, replays frozen V3, and runs the three sequential arms."""

    graph = state.graph
    oracle = state.oracle
    u0 = oracle.decode_state(original)
    legacy = predecode(graph, original, **FROZEN_V3_DECODER_KWARGS)
    legacy_frame = bytes(np.packbits(legacy.observable_frame, bitorder="little"))
    legacy_prediction = _xor(
        legacy_frame, oracle.decode_state(legacy.residual_syndrome).prediction
    )
    trajectories = {
        "o-cost-tx": run_oracle_trajectory(
            graph,
            original,
            policy="cost",
            transaction_policy="tx",
            oracle=oracle,
            **FROZEN_V3_DECODER_KWARGS,
        ),
        "o-frame-tx": run_oracle_trajectory(
            graph,
            original,
            policy="frame",
            transaction_policy="tx",
            oracle=oracle,
            **FROZEN_V3_DECODER_KWARGS,
        ),
        "o-frame-partial": run_oracle_trajectory(
            graph,
            original,
            policy="frame",
            transaction_policy="partial",
            oracle=oracle,
            **FROZEN_V3_DECODER_KWARGS,
        ),
    }
    if (
        trajectories["o-frame-tx"].final_prediction != u0.prediction
        or trajectories["o-frame-partial"].final_prediction != u0.prediction
    ):
        raise AssertionError("O-frame arm did not preserve deterministic U0 frame")
    return u0, legacy, legacy_frame, legacy_prediction, trajectories


def _walk_counterfactual_chain(
    state: _ShotAuditState,
    domain: Any,
    chain_stepper: DomainProposalStepper,
    proposal: CommitProposal,
    evaluation: OracleEvaluation,
    evaluation_meta: Mapping[str, Any],
    pre_state: str,
) -> list[_CounterfactualCandidate]:
    """Chain-stepper loop: re-emit rank one, then veto and rescore candidates
    until the first frame-safe alternative or genuine proposal exhaustion."""

    chain_seen: set[tuple[Any, ...]] = set()
    chain: list[_CounterfactualCandidate] = []
    while (candidate := chain_stepper.next_proposal()) is not None:
        signature = (
            candidate.active_state_fingerprint,
            candidate.stage,
            candidate.endpoints,
            candidate.edge_ids,
        )
        if signature in chain_seen:
            raise AssertionError("counterfactual proposal cycle")
        chain_seen.add(signature)
        if not chain:
            if _proposal_signature(candidate) != _proposal_signature(proposal):
                raise AssertionError("counterfactual rank one differs from original")
            scored, scored_meta = evaluation, evaluation_meta
        else:
            scored, scored_meta = state.evaluate(candidate)
        chain.append(
            _CounterfactualCandidate(
                proposal=candidate,
                evaluation=scored,
                proposal_id=_proposal_id(domain, candidate, pre_state),
                evaluation_metadata=scored_meta,
                candidate_enumeration_ns=(
                    chain_stepper.last_candidate_enumeration_ns
                ),
                stage3_enumeration_ns=(chain_stepper.last_stage3_enumeration_ns),
            )
        )
        if scored.accepted:
            break
        before = chain_stepper.active
        chain_stepper.veto(candidate)
        if chain_stepper.active != before:
            raise AssertionError("counterfactual veto mutated local active state")
    return chain


def _counterfactual_chain_row(
    state: _ShotAuditState,
    domain: Any,
    chain_stepper: DomainProposalStepper,
    candidate_record: _CounterfactualCandidate,
    *,
    rank: int,
    chain_length: int,
    action: str,
    first_safe_id: str | None,
    first_safe_rank: int | None,
    proposal_id: str,
    pre_state: str,
    state_call_count: int,
    counterfactual_wall_ns: int,
) -> dict[str, Any]:
    """Assembles one unchanged-state counterfactual chain row."""

    candidate = candidate_record.proposal
    scored = candidate_record.evaluation
    return {
        **state.scored_candidate_fields(
            domain,
            candidate,
            scored,
            candidate_record.evaluation_metadata,
            active=sorted(chain_stepper.active),
        ),
        **_provenance_fields(
            state.graph,
            trajectory_origin="shadow-original-state-counterfactual",
            arm_id=SHADOW_ARM_ID,
        ),
        "proposal_sha256": candidate_record.proposal_id,
        "original_proposal_sha256": proposal_id,
        "original_state_sha256": pre_state,
        "complete_pre_state_fingerprint": pre_state,
        "local_active_state_fingerprint": list(candidate.active_state_fingerprint),
        "operational_veto_chain_rank": rank,
        "decision": "inspect-only" if scored.accepted else "veto",
        "terminal_action": action if rank == chain_length else None,
        "first_safe_alternative_proposal_sha256": first_safe_id,
        "first_safe_rank": first_safe_rank,
        "censored": False,
        "exhaustion_kind": (
            "proposal"
            if action == "abstain-true-exhaustion" and rank == chain_length
            else None
        ),
        "veto_budget": None,
        "state_veto_count_after": (
            candidate.state_veto_count_before + (0 if scored.accepted else 1)
        ),
        "state_oracle_call_count": state_call_count,
        "counterfactual_wall_ns": counterfactual_wall_ns,
        "candidate_enumeration_wall_ns": (
            candidate_record.candidate_enumeration_ns
        ),
        "stage3_enumeration_wall_ns": (candidate_record.stage3_enumeration_ns),
        "state_total_candidate_enumeration_wall_ns": (
            chain_stepper.total_candidate_enumeration_ns
        ),
        "state_total_stage3_enumeration_wall_ns": (
            chain_stepper.total_stage3_enumeration_ns
        ),
    }


def _explore_unsafe_counterfactuals(
    state: _ShotAuditState,
    domain: Any,
    stepper: DomainProposalStepper,
    proposal: CommitProposal,
    evaluation: OracleEvaluation,
    evaluation_meta: Mapping[str, Any],
    row: dict[str, Any],
    pre_state: str,
    proposal_id: str,
) -> None:
    """Unsafe-evaluation branch: walk the unchanged-state veto chain, resolve
    its terminal action, emit the chain rows, and aggregate state timing."""

    state_call_start = state.oracle_call_ordinal - 1
    counterfactual_start_ns = time.perf_counter_ns()
    counterfactual_row_start = len(state.counterfactual_rows)
    chain_stepper = _new_stepper(state.graph, domain, sorted(stepper.active))
    chain = _walk_counterfactual_chain(
        state, domain, chain_stepper, proposal, evaluation, evaluation_meta, pre_state
    )
    if chain and chain[-1].evaluation.accepted:
        first_safe = chain[-1]
        if first_safe.proposal.stage < proposal.stage:
            raise AssertionError(
                "unchanged-state safe alternative moved to an earlier stage"
            )
        action = (
            "same-stage-alternative"
            if first_safe.proposal.stage == proposal.stage
            else "later-stage-alternative"
        )
        first_safe_id = first_safe.proposal_id
        first_safe_rank = len(chain)
    else:
        exhausted = chain_stepper.outcome("tx")
        if (
            exhausted.exhaustion_kind != "proposal"
            or exhausted.veto_budget is not None
        ):
            raise AssertionError(
                "abstention was not genuine uncapped proposal exhaustion"
            )
        action = "abstain-true-exhaustion"
        first_safe_id = None
        first_safe_rank = None
    state_call_count = state.oracle_call_ordinal - state_call_start
    counterfactual_wall_ns = time.perf_counter_ns() - counterfactual_start_ns
    row["state_oracle_call_count"] = state_call_count
    row["counterfactual_wall_ns"] = counterfactual_wall_ns
    for rank, candidate_record in enumerate(chain, 1):
        state.counterfactual_rows.append(
            _counterfactual_chain_row(
                state,
                domain,
                chain_stepper,
                candidate_record,
                rank=rank,
                chain_length=len(chain),
                action=action,
                first_safe_id=first_safe_id,
                first_safe_rank=first_safe_rank,
                proposal_id=proposal_id,
                pre_state=pre_state,
                state_call_count=state_call_count,
                counterfactual_wall_ns=counterfactual_wall_ns,
            )
        )
    state_support_classification_ns = sum(
        int(item["support_classification_wall_ns"])
        for item in state.counterfactual_rows[counterfactual_row_start:]
    ) + int(row["support_classification_wall_ns"])
    for item in state.counterfactual_rows[counterfactual_row_start:]:
        item["state_support_classification_wall_ns"] = (
            state_support_classification_ns
        )
    row["state_support_classification_wall_ns"] = state_support_classification_ns


def _audit_shadow_commit(
    state: _ShotAuditState,
    domain: Any,
    stepper: DomainProposalStepper,
    path: PrematchedPath,
    *,
    initial_active: tuple[int, ...],
) -> None:
    """One durable V3 commit: verify the stepper emission, score it, probe the
    same-stage competitor, assemble the proposal row, explore unchanged-state
    counterfactuals when unsafe, then accept and advance the durable state."""

    graph = state.graph
    proposal = stepper.next_proposal()
    if proposal is None or _proposal_signature(proposal) != _path_signature(path):
        raise AssertionError(
            "DomainProposalStepper first emission differs from durable V3 path"
        )
    if (
        proposal.decision_weight != path.decision_weight
        or proposal.observable_frame != path.observable_mask
    ):
        raise AssertionError("DomainProposalStepper numerical path differs from V3")
    pre_state = _state_fingerprint(graph, state.residual, state.frame)
    evaluation, evaluation_meta = state.evaluate(proposal)
    proposal_id = _proposal_id(domain, proposal, pre_state)

    # Disposable same-state chain. Rank 1 must be the original.
    clone = _new_stepper(graph, domain, sorted(stepper.active))
    first = clone.next_proposal()
    if first is None or _proposal_signature(first) != _proposal_signature(proposal):
        raise AssertionError(
            "counterfactual clone did not reproduce original first emission"
        )
    clone.veto(first)
    competitor = clone.next_proposal()
    same_stage = competitor is not None and competitor.stage == proposal.stage
    margin = (
        None
        if not same_stage
        else competitor.decision_weight - proposal.decision_weight
    )
    row = {
        **state.scored_candidate_fields(
            domain,
            proposal,
            evaluation,
            evaluation_meta,
            active=sorted(stepper.active),
        ),
        **_provenance_fields(
            graph, trajectory_origin="shadow-original", arm_id=SHADOW_ARM_ID
        ),
        "proposal_sha256": proposal_id,
        "complete_pre_state_fingerprint": pre_state,
        "local_active_state_fingerprint": list(proposal.active_state_fingerprint),
        "trajectory_commit_index": state.commit_index,
        "decision": "shadow-commit",
        "durable": True,
        "provisional": True,
        "rolled_back": False,
        "domain_initial_hw": len(initial_active),
        "domain_current_hw": len(stepper.active),
        "global_detector_hw": int(state.residual.sum()),
        "residual_hw_target": 10,
        "accepted_prefix_length": len(stepper.accepted_paths),
        "state_oracle_call_count": 1,
        "candidate_enumeration_wall_ns": stepper.last_candidate_enumeration_ns,
        "stage3_enumeration_wall_ns": stepper.last_stage3_enumeration_ns,
        "same_stage_competitor_exists": same_stage,
        "same_stage_competitor_weight": None
        if not same_stage
        else competitor.decision_weight,
        "same_stage_competitor_weight_hex": None
        if not same_stage
        else competitor.decision_weight.hex(),
        "absolute_weight_margin": margin,
        "absolute_weight_margin_hex": None if margin is None else margin.hex(),
        "relative_weight_margin": None
        if margin is None or proposal.decision_weight == 0
        else margin / proposal.decision_weight,
        "relative_weight_margin_hex": None
        if margin is None or proposal.decision_weight == 0
        else (margin / proposal.decision_weight).hex(),
    }
    state.proposal_rows.append(row)

    if not evaluation.accepted:
        _explore_unsafe_counterfactuals(
            state,
            domain,
            stepper,
            proposal,
            evaluation,
            evaluation_meta,
            row,
            pre_state,
            proposal_id,
        )
    stepper.accept(proposal)
    state.residual = apply_detector_boundary(state.residual, proposal.detector_boundary)
    state.frame = _xor(state.frame, proposal.observable_frame)
    row["complete_post_decision_state_fingerprint"] = _state_fingerprint(
        graph, state.residual, state.frame
    )
    state.commit_index += 1


def _replay_shadow_domains(
    state: _ShotAuditState, original: np.ndarray, legacy: Any
) -> None:
    """Domain replay loop: re-emit every durable V3 path in domain order,
    audit each commit, and reconcile every domain's terminal outcome."""

    graph = state.graph
    paths_by_domain: dict[Any, list[PrematchedPath]] = defaultdict(list)
    for path in legacy.paths:
        paths_by_domain[path.domain].append(path)
    state.residual = original.copy()
    state.frame = bytes((graph.num_observables + 7) // 8)
    for domain in sorted(graph.domain_graphs):
        detector_ids = graph.domain_graphs[domain].detector_ids
        initial_active = tuple(v for v in detector_ids if original[v])
        expected_paths = paths_by_domain.get(domain, [])
        stepper = _new_stepper(graph, domain, initial_active)
        for path in expected_paths:
            _audit_shadow_commit(
                state, domain, stepper, path, initial_active=initial_active
            )
        outcome = stepper.outcome("tx") if stepper.is_finished else None
        stats = legacy.domain_stats[domain]
        if stats.status in {"success", "below-limit"}:
            if outcome is None or outcome.status != stats.status:
                raise AssertionError(
                    "successful V3 domain and stepper replay status differ"
                )
            if outcome.durable_active != frozenset(
                v for v in detector_ids if legacy.residual_syndrome[v]
            ):
                raise AssertionError(
                    "successful V3 domain and stepper durable active set differ"
                )
        state.domain_rows.append(
            {
                **_domain_json(domain),
                **_provenance_fields(
                    graph, trajectory_origin="shadow-original", arm_id=SHADOW_ARM_ID
                ),
                "domain_initial_hw": stats.initial_hw,
                "provisional_residual_hw": stats.attempted_residual_hw,
                "final_residual_detector_hw": stats.final_residual_hw,
                "residual_hw_target": 10,
                "domain_terminal_status": stats.status,
                "fallback_reason": None
                if stats.fallback_reason is None
                else stats.fallback_reason.value,
                "committed_matches": stats.committed_matches,
                "accepted_prefix_length": stats.attempted_matches,
                "provisional_events_removed": stats.initial_hw
                - stats.attempted_residual_hw,
                "durable_events_removed": stats.initial_hw - stats.final_residual_hw,
                "events_lost_to_rollback": stats.final_residual_hw
                - stats.attempted_residual_hw,
                "stepper_replay_status": None if outcome is None else outcome.status,
            }
        )


def _append_sequential_arm_domain_rows(
    state: _ShotAuditState, trajectories: Mapping[str, Any]
) -> None:
    """Row assembly for the three sequential oracle arms' domain outcomes."""

    for arm_label, trajectory in trajectories.items():
        arm_id = {
            "o-cost-tx": ARM_IDS[2],
            "o-frame-tx": ARM_IDS[3],
            "o-frame-partial": ARM_IDS[4],
        }[arm_label]
        origin = "sequential-" + arm_label
        for record in trajectory.domains:
            outcome = record.outcome
            state.domain_rows.append(
                {
                    **_domain_json(record.domain),
                    **_provenance_fields(
                        state.graph, trajectory_origin=origin, arm_id=arm_id
                    ),
                    "domain_initial_hw": len(outcome.initial_active),
                    "residual_hw_target": 10,
                    "provisional_residual_hw": len(outcome.provisional_active),
                    "final_residual_detector_hw": len(outcome.durable_active),
                    "domain_terminal_status": outcome.status,
                    "accepted_prefix_length": outcome.accepted_proposals,
                    "fallback_reason": None
                    if outcome.fallback_reason is None
                    else outcome.fallback_reason.value,
                    "exhaustion_kind": outcome.exhaustion_kind,
                    "veto_budget": outcome.veto_budget,
                    "provisional_events_removed": len(outcome.initial_active)
                    - len(outcome.provisional_active),
                    "durable_events_removed": len(outcome.initial_active)
                    - len(outcome.durable_active),
                    "events_lost_to_rollback": len(outcome.durable_active)
                    - len(outcome.provisional_active),
                }
            )


def _assemble_shot_summary(
    state: _ShotAuditState,
    original: np.ndarray,
    legacy: Any,
    trajectories: Mapping[str, Any],
) -> dict[str, Any]:
    """Row assembly for the per-shot summary: arm summaries, oracle cache
    counters, and the exact wall-time telemetry sums."""

    shadow_provisional = sum(
        stats.initial_hw - stats.attempted_residual_hw
        for stats in legacy.domain_stats.values()
    )
    shadow_durable = sum(
        stats.initial_hw - stats.final_residual_hw
        for stats in legacy.domain_stats.values()
    )
    return {
        "original_detector_hw": int(original.sum()),
        "graph_fingerprint": state.graph.fingerprint,
        "arm_summaries": {
            ARM_IDS[0]: {
                "final_residual_detector_hw": int(original.sum()),
                "proposals_attempted": 0,
                "durable_commitments": 0,
                "provisional_events_removed": 0,
                "durable_events_removed": 0,
                "events_lost_to_rollback": 0,
            },
            ARM_IDS[1]: {
                "final_residual_detector_hw": int(legacy.residual_syndrome.sum()),
                "proposals_attempted": len(legacy.paths),
                "durable_commitments": len(legacy.paths),
                "provisional_events_removed": shadow_provisional,
                "durable_events_removed": shadow_durable,
                "events_lost_to_rollback": shadow_provisional - shadow_durable,
            },
            ARM_IDS[2]: _trajectory_summary(trajectories["o-cost-tx"]),
            ARM_IDS[3]: _trajectory_summary(trajectories["o-frame-tx"]),
            ARM_IDS[4]: _trajectory_summary(trajectories["o-frame-partial"]),
        },
        "oracle_cache_hits": state.oracle.cache_stats.hits,
        "oracle_cache_misses": state.oracle.cache_stats.misses,
        "oracle_evaluation_call_count": state.oracle_call_ordinal,
        "shadow_audit_oracle_evaluation_call_count": state.oracle_call_ordinal,
        "total_full_mwpm_cache_miss_count_all_arms": state.oracle.cache_stats.misses,
        "matched_active_pair_backend_call_count": (
            state.matched_telemetry["backend_calls"]
        ),
        "matched_active_pair_backend_wall_ns": (
            state.matched_telemetry["backend_wall_ns"]
        ),
        "timing_telemetry": {
            "counterfactual_wall_ns": sum(
                int(row.get("counterfactual_wall_ns", 0))
                for row in state.proposal_rows
            ),
            "support_classification_wall_ns": sum(
                int(row.get("support_classification_wall_ns", 0))
                for row in (*state.proposal_rows, *state.counterfactual_rows)
            ),
            "stage3_specific_wall_ns": sum(
                int(row.get("stage3_enumeration_wall_ns", 0))
                for row in (*state.proposal_rows, *state.counterfactual_rows)
            ),
            "exact_wall_timing_available": True,
            "identity_or_order_input": False,
        },
    }


def audit_policy_shot(
    graph: CompiledPromatchGraph,
    syndrome: Sequence[int] | np.ndarray,
    *,
    tolerance: OracleTolerance,
) -> Mapping[str, Any]:
    """Audits one detector shot without accepting or returning ground truth."""

    if not isinstance(graph, CompiledPromatchGraph):
        raise TypeError("graph must be a CompiledPromatchGraph")
    values = np.asarray(syndrome)
    if values.shape != (graph.num_detectors,) or np.any((values != 0) & (values != 1)):
        raise ValueError("syndrome must be a binary vector of graph.num_detectors")
    original = np.asarray(values, dtype=np.uint8).copy()
    state = _ShotAuditState(graph, tolerance)
    u0, legacy, legacy_frame, legacy_prediction, trajectories = (
        _reconstruct_reference_arms(state, original)
    )
    _replay_shadow_domains(state, original, legacy)

    if not np.array_equal(state.residual, legacy.residual_syndrome):
        raise AssertionError("reconstructed durable shadow residual differs from V3")
    if state.frame != legacy_frame:
        raise AssertionError("reconstructed durable shadow frame differs from V3")

    _append_sequential_arm_domain_rows(state, trajectories)
    shot = _assemble_shot_summary(state, original, legacy, trajectories)
    return {
        "arm_predictions": {
            ARM_IDS[0]: u0.prediction,
            ARM_IDS[1]: legacy_prediction,
            ARM_IDS[2]: trajectories["o-cost-tx"].final_prediction,
            ARM_IDS[3]: trajectories["o-frame-tx"].final_prediction,
            ARM_IDS[4]: trajectories["o-frame-partial"].final_prediction,
        },
        "shot": shot,
        "proposals": state.proposal_rows,
        "counterfactuals": state.counterfactual_rows,
        "domains": state.domain_rows,
    }


def expand_policy_casebook_state(
    graph: CompiledPromatchGraph,
    syndrome: Sequence[int] | np.ndarray,
    *,
    original_proposal_sha256: str,
    original_state_sha256: str,
    tolerance: OracleTolerance,
) -> list[dict[str, Any]]:
    """Exhaustively scores one selected original state through true exhaustion.

    Unlike the all-shot audit, this deliberately continues after the first
    frame-safe proposal.  It accepts detector syndrome only and never mutates
    the original V3 trajectory with a counterfactual choice.
    """

    original = np.asarray(syndrome, dtype=np.uint8)
    if original.shape != (graph.num_detectors,) or np.any(
        (original != 0) & (original != 1)
    ):
        raise ValueError("syndrome must be a binary vector of graph.num_detectors")
    legacy = predecode(graph, original, **FROZEN_V3_DECODER_KWARGS)
    paths_by_domain: dict[Any, list[PrematchedPath]] = defaultdict(list)
    for path in legacy.paths:
        paths_by_domain[path.domain].append(path)
    state = _ShotAuditState(graph, tolerance)
    state.residual = original.copy()
    state.frame = bytes((graph.num_observables + 7) // 8)
    found: list[dict[str, Any]] | None = None
    for domain in sorted(graph.domain_graphs):
        initial_active = tuple(
            v for v in graph.domain_graphs[domain].detector_ids if original[v]
        )
        stepper = _new_stepper(graph, domain, initial_active)
        for path in paths_by_domain.get(domain, ()):
            proposal = stepper.next_proposal()
            if proposal is None or _proposal_signature(proposal) != _path_signature(
                path
            ):
                raise AssertionError("casebook replay differs from durable V3 path")
            found = _expand_selected_casebook_state(
                state,
                domain,
                stepper,
                proposal,
                original_proposal_sha256=original_proposal_sha256,
                original_state_sha256=original_state_sha256,
                found=found,
            )
            stepper.accept(proposal)
            state.residual = apply_detector_boundary(
                state.residual, proposal.detector_boundary
            )
            state.frame = _xor(state.frame, proposal.observable_frame)
    if found is None:
        raise ValueError(
            "selected original proposal/state was not found in detector replay"
        )
    return found


def _expand_selected_casebook_state(
    state: _ShotAuditState,
    domain: Any,
    stepper: DomainProposalStepper,
    proposal: CommitProposal,
    *,
    original_proposal_sha256: str,
    original_state_sha256: str,
    found: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Runs the exhaustive full-slate loop when this commit is the selection."""

    graph = state.graph
    pre_state = _state_fingerprint(graph, state.residual, state.frame)
    proposal_id = _proposal_id(domain, proposal, pre_state)
    if (
        proposal_id != original_proposal_sha256
        or pre_state != original_state_sha256
    ):
        return found
    if found is not None:
        raise AssertionError("selected casebook state is not unique in its shot")
    evaluation, evaluation_meta = state.evaluate(proposal)
    if evaluation.accepted:
        raise ValueError("casebook expansion requires an unsafe original proposal")
    clone = _new_stepper(graph, domain, sorted(stepper.active))
    expansion_start_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    first_safe_rank: int | None = None
    rank = 0
    while (candidate := clone.next_proposal()) is not None:
        rank += 1
        signature = (
            candidate.active_state_fingerprint,
            candidate.stage,
            candidate.endpoints,
            candidate.edge_ids,
        )
        if signature in seen:
            raise AssertionError("casebook exhaustive replay contains a cycle")
        seen.add(signature)
        if rank == 1:
            if _proposal_signature(candidate) != _proposal_signature(proposal):
                raise AssertionError("casebook rank one differs from original")
            scored, scored_meta = evaluation, evaluation_meta
        else:
            scored, scored_meta = state.evaluate(candidate)
        if scored.accepted and first_safe_rank is None:
            first_safe_rank = rank
        candidate_id = _proposal_id(domain, candidate, pre_state)
        rows.append(
            {
                **state.scored_candidate_fields(
                    domain,
                    candidate,
                    scored,
                    scored_meta,
                    active=sorted(clone.active),
                ),
                **_provenance_fields(
                    graph,
                    trajectory_origin="casebook-exhaustive",
                    arm_id=SHADOW_ARM_ID,
                ),
                "proposal_sha256": candidate_id,
                "original_proposal_sha256": original_proposal_sha256,
                "original_state_sha256": original_state_sha256,
                "complete_pre_state_fingerprint": pre_state,
                "local_active_state_fingerprint": list(
                    candidate.active_state_fingerprint
                ),
                "operational_veto_chain_rank": rank,
                "decision": "inspect-only" if scored.accepted else "veto",
                "is_first_safe_alternative": scored.accepted
                and first_safe_rank == rank,
                "first_safe_rank": first_safe_rank,
                "terminal_action": None,
                "censored": False,
                "veto_budget": None,
                "candidate_enumeration_wall_ns": clone.last_candidate_enumeration_ns,
                "stage3_enumeration_wall_ns": clone.last_stage3_enumeration_ns,
            }
        )
        before = clone.active
        clone.veto(candidate)
        if clone.active != before:
            raise AssertionError("casebook veto mutated the selected local state")
    outcome = clone.outcome("tx")
    if outcome.exhaustion_kind != "proposal" or outcome.veto_budget is not None:
        raise AssertionError("casebook slate did not reach genuine uncapped exhaustion")
    if not rows:
        raise AssertionError("selected unsafe state emitted no proposal")
    if rows[0]["proposal_sha256"] != original_proposal_sha256:
        raise AssertionError("casebook rank one differs from selected original")
    for row in rows:
        row["first_safe_rank"] = first_safe_rank
        row["state_oracle_call_count"] = len(rows)
        row["counterfactual_wall_ns"] = time.perf_counter_ns() - expansion_start_ns
        row["state_total_candidate_enumeration_wall_ns"] = (
            clone.total_candidate_enumeration_ns
        )
        row["state_total_stage3_enumeration_wall_ns"] = (
            clone.total_stage3_enumeration_ns
        )
    rows[-1]["terminal_action"] = "exhaustive-true-exhaustion"
    rows[-1]["exhaustion_kind"] = "proposal"
    return rows
