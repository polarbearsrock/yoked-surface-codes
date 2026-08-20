"""In-memory Phase-A replay trajectories for the full-graph ProMatch oracle.

This module owns no artifact I/O and never accepts actual observables.  It
only compares ProMatch proposals with complete-graph decoder certificates and
records the resulting deterministic state transitions.  Keeping sampled
observables outside this API makes it impossible for an oracle decision to
peek at the shot's success label.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np

from yoked.decoding._promatch import (
    BoundaryPolicy,
    CommitProposal,
    DomainProposalStepper,
    DomainStepOutcome,
    ObservablePolicy,
    PrematchedPath,
    TransactionPolicy,
    apply_detector_boundary,
)
from yoked.decoding._promatch_graph import CompiledPromatchGraph
from yoked.decoding._promatch_layout import L1DomainKey
from yoked.decoding.oracle.full_graph import (
    FullGraphOracle,
    OracleCacheStats,
    OracleEvaluation,
    OracleMatchingSolution,
    OraclePolicy,
    OracleTolerance,
    _bits_to_mask,
    _xor_masks,
)


ProposalDecision: TypeAlias = Literal["accepted", "vetoed"]


def _jsonable(value: Any) -> Any:
    """Converts the frozen replay records into a deterministic JSON tree."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            raw = getattr(value, field.name)
            result[field.name] = _jsonable(raw)
            # JSON's shortest-round-trip decimal is deterministic, but the
            # experiment contract also retains the exact IEEE-754 value used
            # by every weight/tolerance decision.
            if isinstance(raw, float):
                result[f"{field.name}_hex"] = raw.hex()
        return result
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, frozenset):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value)!r} is not JSON-serializable")


def _binary_syndrome(
    syndrome: Sequence[int] | np.ndarray, *, num_detectors: int
) -> np.ndarray:
    result = np.asarray(syndrome)
    if result.ndim != 1 or result.shape != (num_detectors,):
        raise ValueError(
            f"syndrome must have shape ({num_detectors},), got {result.shape}"
        )
    if not np.issubdtype(result.dtype, np.bool_) and not np.issubdtype(
        result.dtype, np.integer
    ):
        raise TypeError("syndrome must contain boolean or integer bits")
    if np.any((result != 0) & (result != 1)):
        raise ValueError("syndrome entries must be binary")
    return np.asarray(result, dtype=np.uint8).copy()


def _mask_to_bits(mask: bytes, *, count: int) -> np.ndarray:
    expected_bytes = (count + 7) // 8
    if len(mask) != expected_bytes:
        raise ValueError(
            f"observable mask has {len(mask)} bytes; expected {expected_bytes}"
        )
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(
        np.frombuffer(mask, dtype=np.uint8), bitorder="little", count=count
    ).astype(np.uint8, copy=False)


def _apply_frame(frame: np.ndarray, mask: bytes) -> np.ndarray:
    result = frame.copy()
    result ^= _mask_to_bits(mask, count=len(frame))
    return result


def _combined_prediction(
    frame: np.ndarray, residual_solution: OracleMatchingSolution
) -> bytes:
    return _xor_masks(_bits_to_mask(frame), residual_solution.prediction)


def _complete_state_fingerprint(
    graph: CompiledPromatchGraph, syndrome: np.ndarray, frame: np.ndarray
) -> str:
    """Hashes the graph identity plus packed complete syndrome and frame."""

    digest = hashlib.sha256()
    digest.update(b"promatch-oracle-complete-state-v1\0")
    digest.update(graph.fingerprint.encode("ascii"))
    digest.update(graph.num_detectors.to_bytes(8, "little"))
    digest.update(graph.num_observables.to_bytes(8, "little"))
    digest.update(_bits_to_mask(syndrome))
    digest.update(_bits_to_mask(frame))
    return digest.hexdigest()


def _assert_evaluation_matches_candidate(
    *,
    evaluation: OracleEvaluation,
    edge_ids: tuple[int, ...],
    detector_boundary: tuple[int, ...],
    observable_frame: bytes,
    decision_weight: float,
) -> None:
    if evaluation.candidate_edge_ids != tuple(sorted(edge_ids)):
        raise AssertionError("oracle normalized a different candidate edge support")
    if evaluation.candidate_boundary_detector_ids != detector_boundary:
        raise AssertionError("proposal and oracle disagree on detector boundary")
    if evaluation.candidate_observable_frame != observable_frame:
        raise AssertionError("proposal and oracle disagree on observable frame")
    if not math.isclose(
        evaluation.candidate_weight,
        decision_weight,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("proposal and oracle disagree on candidate weight")


def _path_detector_boundary(
    graph: CompiledPromatchGraph, path: PrematchedPath
) -> tuple[int, ...]:
    """Reconstructs a legacy path boundary without trusting the oracle."""

    edge_by_id = {edge.edge_id: edge for edge in graph.edges}
    if len(edge_by_id) != len(graph.edges):
        raise ValueError("compiled graph contains duplicate edge IDs")
    toggled: set[int] = set()
    seen: set[int] = set()
    for edge_id in path.edge_ids:
        if edge_id in seen:
            raise ValueError("legacy path support is not square-free")
        seen.add(edge_id)
        try:
            edge = edge_by_id[edge_id]
        except KeyError as ex:
            raise ValueError(f"legacy path references unknown edge {edge_id}") from ex
        toggled.symmetric_difference_update((edge.source,))
        if edge.target is not None:
            toggled.symmetric_difference_update((edge.target,))
    boundary = tuple(sorted(toggled))
    public_endpoints = tuple(sorted(v for v in path.endpoints if v is not None))
    if boundary != public_endpoints:
        raise AssertionError(
            "legacy path canonical boundary does not equal its recorded endpoints"
        )
    return boundary


@dataclasses.dataclass(frozen=True)
class ShadowProposalRecord:
    """Oracle score for a path that the archived predecoder already committed."""

    ordinal: int
    path: PrematchedPath
    evaluation: OracleEvaluation
    oracle_cost_accepts: bool
    oracle_frame_accepts: bool
    pre_state_fingerprint: str
    post_decision_state_fingerprint: str
    oracle_evaluation_ordinal: int


@dataclasses.dataclass(frozen=True)
class ShadowReplayResult:
    """The legacy path sequence followed unconditionally under oracle shadowing."""

    initial_syndrome: tuple[int, ...]
    initial_u0_prediction: bytes
    final_residual_syndrome: tuple[int, ...]
    final_observable_frame: bytes
    final_residual_solution: OracleMatchingSolution
    final_prediction: bytes
    proposals: tuple[ShadowProposalRecord, ...]

    def to_json(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclasses.dataclass(frozen=True)
class ProposalReplayRecord:
    """One proposal decision plus its eventual durability classification."""

    ordinal: int
    domain: L1DomainKey
    proposal: CommitProposal
    evaluation: OracleEvaluation
    decision: ProposalDecision
    accepted: bool
    vetoed: bool
    durable: bool
    rolled_back: bool
    pre_state_fingerprint: str
    post_decision_state_fingerprint: str
    oracle_evaluation_ordinal: int
    state_veto_count_after: int


@dataclasses.dataclass(frozen=True)
class DomainReplayRecord:
    """Terminal transaction result and the proposal slice belonging to it."""

    domain: L1DomainKey
    proposal_start: int
    proposal_stop: int
    outcome: DomainStepOutcome


@dataclasses.dataclass(frozen=True)
class OracleTrajectoryResult:
    """One sequential O-cost or O-frame trajectory over every L1 domain."""

    policy: OraclePolicy
    transaction_policy: TransactionPolicy
    initial_syndrome: tuple[int, ...]
    initial_u0_prediction: bytes
    final_residual_syndrome: tuple[int, ...]
    final_observable_frame: bytes
    final_residual_solution: OracleMatchingSolution
    final_prediction: bytes
    proposals: tuple[ProposalReplayRecord, ...]
    domains: tuple[DomainReplayRecord, ...]

    def to_json(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclasses.dataclass(frozen=True)
class PhaseAReplayResult:
    """The shadow audit and three frozen sequential Phase-A arms."""

    shadow: ShadowReplayResult
    cost_tx: OracleTrajectoryResult
    frame_tx: OracleTrajectoryResult
    frame_partial: OracleTrajectoryResult
    cache_stats: OracleCacheStats
    initial_uncached_repeatability_verified: bool
    initial_cached_equivalence_verified: bool

    def to_json(self) -> dict[str, Any]:
        return _jsonable(self)


def shadow_score_paths(
    graph: CompiledPromatchGraph,
    syndrome: Sequence[int] | np.ndarray,
    paths: Sequence[PrematchedPath],
    *,
    oracle: FullGraphOracle | None = None,
    tolerance: OracleTolerance = OracleTolerance(),
) -> ShadowReplayResult:
    """Scores archived paths against global context while following them anyway."""

    if oracle is None:
        oracle = FullGraphOracle(graph, tolerance=tolerance)
    elif oracle.graph is not graph:
        raise ValueError("oracle was compiled for a different graph object")

    residual = _binary_syndrome(syndrome, num_detectors=graph.num_detectors)
    frame = np.zeros(graph.num_observables, dtype=np.uint8)
    initial = residual.copy()
    initial_u0 = oracle.decode_state(initial).prediction
    records: list[ShadowProposalRecord] = []
    for ordinal, path in enumerate(paths):
        pre_state_fingerprint = _complete_state_fingerprint(graph, residual, frame)
        expected_boundary = _path_detector_boundary(graph, path)
        evaluation = oracle.evaluate(
            syndrome=residual,
            accumulated_frame=frame,
            candidate_edge_ids=path.edge_ids,
            policy="frame",
        )
        _assert_evaluation_matches_candidate(
            evaluation=evaluation,
            edge_ids=path.edge_ids,
            detector_boundary=expected_boundary,
            observable_frame=path.observable_mask,
            decision_weight=path.decision_weight,
        )
        next_residual = apply_detector_boundary(
            residual, evaluation.candidate_boundary_detector_ids
        )
        next_frame = _apply_frame(frame, path.observable_mask)
        records.append(
            ShadowProposalRecord(
                ordinal=ordinal,
                path=path,
                evaluation=evaluation,
                oracle_cost_accepts=evaluation.cost_compatible,
                oracle_frame_accepts=evaluation.accepted,
                pre_state_fingerprint=pre_state_fingerprint,
                post_decision_state_fingerprint=_complete_state_fingerprint(
                    graph, next_residual, next_frame
                ),
                oracle_evaluation_ordinal=ordinal,
            )
        )
        residual = next_residual
        frame = next_frame

    final_solution = oracle.decode_state(residual)
    return ShadowReplayResult(
        initial_syndrome=tuple(int(v) for v in initial),
        initial_u0_prediction=initial_u0,
        final_residual_syndrome=tuple(int(v) for v in residual),
        final_observable_frame=_bits_to_mask(frame),
        final_residual_solution=final_solution,
        final_prediction=_combined_prediction(frame, final_solution),
        proposals=tuple(records),
    )


def run_oracle_trajectory(
    graph: CompiledPromatchGraph,
    syndrome: Sequence[int] | np.ndarray,
    *,
    policy: OraclePolicy,
    transaction_policy: TransactionPolicy,
    residual_hw_limit: int,
    boundary_policy: BoundaryPolicy = "disabled",
    observable_policy: ObservablePolicy = "edge-zero",
    veto_budget: int | None = None,
    oracle: FullGraphOracle | None = None,
    tolerance: OracleTolerance = OracleTolerance(),
) -> OracleTrajectoryResult:
    """Runs one globally scored, domain-proposed Phase-A trajectory."""

    if policy not in ("cost", "frame"):
        raise ValueError(f"unsupported oracle policy {policy!r}")
    if transaction_policy not in ("tx", "partial"):
        raise ValueError(f"unsupported transaction policy {transaction_policy!r}")
    if oracle is None:
        oracle = FullGraphOracle(graph, tolerance=tolerance)
    elif oracle.graph is not graph:
        raise ValueError("oracle was compiled for a different graph object")

    residual = _binary_syndrome(syndrome, num_detectors=graph.num_detectors)
    frame = np.zeros(graph.num_observables, dtype=np.uint8)
    initial = residual.copy()
    initial_u0 = oracle.decode_state(initial).prediction
    proposals: list[ProposalReplayRecord] = []
    domains: list[DomainReplayRecord] = []
    assigned: set[int] = set()

    for domain in sorted(graph.domain_graphs):
        domain_graph = graph.domain_graphs[domain]
        detector_ids = frozenset(int(v) for v in domain_graph.detector_ids)
        overlap = assigned.intersection(detector_ids)
        if overlap:
            raise ValueError(
                f"detectors belong to multiple predecode domains: {sorted(overlap)}"
            )
        assigned.update(detector_ids)
        initial_active = tuple(v for v in sorted(detector_ids) if residual[v])
        stepper = DomainProposalStepper(
            domain_graph,
            initial_active,
            num_detectors=graph.num_detectors,
            num_observables=graph.num_observables,
            residual_hw_limit=residual_hw_limit,
            boundary_policy=boundary_policy,
            observable_policy=observable_policy,
            veto_budget=veto_budget,
        )
        before_residual = residual.copy()
        before_frame = frame.copy()
        proposal_start = len(proposals)

        while (proposal := stepper.next_proposal()) is not None:
            state_residual = residual.copy()
            state_frame = frame.copy()
            pre_state_fingerprint = _complete_state_fingerprint(
                graph, state_residual, state_frame
            )
            evaluation = oracle.evaluate(
                syndrome=residual,
                accumulated_frame=frame,
                candidate_edge_ids=proposal.edge_ids,
                policy=policy,
            )
            if not np.array_equal(residual, state_residual) or not np.array_equal(
                frame, state_frame
            ):
                raise AssertionError("oracle evaluation mutated trajectory state")
            _assert_evaluation_matches_candidate(
                evaluation=evaluation,
                edge_ids=proposal.edge_ids,
                detector_boundary=proposal.detector_boundary,
                observable_frame=proposal.observable_frame,
                decision_weight=proposal.decision_weight,
            )

            ordinal = len(proposals)
            if evaluation.accepted:
                stepper.accept(proposal)
                residual = apply_detector_boundary(residual, proposal.detector_boundary)
                frame = _apply_frame(frame, proposal.observable_frame)
                proposals.append(
                    ProposalReplayRecord(
                        ordinal=ordinal,
                        domain=domain,
                        proposal=proposal,
                        evaluation=evaluation,
                        decision="accepted",
                        accepted=True,
                        vetoed=False,
                        durable=False,
                        rolled_back=False,
                        pre_state_fingerprint=pre_state_fingerprint,
                        post_decision_state_fingerprint=_complete_state_fingerprint(
                            graph, residual, frame
                        ),
                        oracle_evaluation_ordinal=ordinal,
                        state_veto_count_after=0,
                    )
                )
            else:
                stepper.veto(proposal)
                if not np.array_equal(residual, state_residual) or not np.array_equal(
                    frame, state_frame
                ):
                    raise AssertionError("veto changed syndrome or observable frame")
                proposals.append(
                    ProposalReplayRecord(
                        ordinal=ordinal,
                        domain=domain,
                        proposal=proposal,
                        evaluation=evaluation,
                        decision="vetoed",
                        accepted=False,
                        vetoed=True,
                        durable=False,
                        rolled_back=False,
                        pre_state_fingerprint=pre_state_fingerprint,
                        post_decision_state_fingerprint=_complete_state_fingerprint(
                            graph, residual, frame
                        ),
                        oracle_evaluation_ordinal=ordinal,
                        state_veto_count_after=proposal.state_veto_count_before + 1,
                    )
                )

        outcome = stepper.outcome(transaction_policy)
        proposal_stop = len(proposals)
        provisional_active = frozenset(v for v in detector_ids if residual[v])
        if provisional_active != outcome.provisional_active:
            raise AssertionError(
                "trajectory syndrome disagrees with provisional stepper state"
            )
        if outcome.status == "rollback":
            residual = before_residual
            frame = before_frame
            for index in range(proposal_start, proposal_stop):
                record = proposals[index]
                if record.accepted:
                    proposals[index] = dataclasses.replace(
                        record, durable=False, rolled_back=True
                    )
        else:
            for index in range(proposal_start, proposal_stop):
                record = proposals[index]
                if record.accepted:
                    proposals[index] = dataclasses.replace(record, durable=True)

        durable_active = frozenset(v for v in detector_ids if residual[v])
        if durable_active != outcome.durable_active:
            raise AssertionError(
                "trajectory syndrome disagrees with durable stepper state"
            )

        domains.append(
            DomainReplayRecord(
                domain=domain,
                proposal_start=proposal_start,
                proposal_stop=proposal_stop,
                outcome=outcome,
            )
        )

    final_solution = oracle.decode_state(residual)
    final_prediction = _combined_prediction(frame, final_solution)
    if policy == "frame" and final_prediction != initial_u0:
        raise AssertionError(
            "sequential O-frame trajectory changed the initial U0 prediction"
        )
    return OracleTrajectoryResult(
        policy=policy,
        transaction_policy=transaction_policy,
        initial_syndrome=tuple(int(v) for v in initial),
        initial_u0_prediction=initial_u0,
        final_residual_syndrome=tuple(int(v) for v in residual),
        final_observable_frame=_bits_to_mask(frame),
        final_residual_solution=final_solution,
        final_prediction=final_prediction,
        proposals=tuple(proposals),
        domains=tuple(domains),
    )


def run_phase_a_replay(
    graph: CompiledPromatchGraph,
    syndrome: Sequence[int] | np.ndarray,
    legacy_paths: Sequence[PrematchedPath],
    *,
    residual_hw_limit: int,
    boundary_policy: BoundaryPolicy = "disabled",
    observable_policy: ObservablePolicy = "edge-zero",
    veto_budget: int | None = None,
    tolerance: OracleTolerance = OracleTolerance(),
) -> PhaseAReplayResult:
    """Runs the shadow audit and O-cost/tx, O-frame/tx and O-frame/partial."""

    oracle = FullGraphOracle(graph, tolerance=tolerance)
    # These are real-graph preflight gates, not merely unit tests.  Each
    # retained shot must reconstruct the same deterministic prediction,
    # support, fsum weight, backend weight, and tolerance certificate across
    # independent backend calls and through the cache.
    uncached_first = oracle.decode_state(syndrome, use_cache=False)
    uncached_second = oracle.decode_state(syndrome, use_cache=False)
    if uncached_first != uncached_second:
        raise AssertionError("repeated uncached complete solves are not deterministic")
    cached_initial = oracle.decode_state(syndrome)
    if cached_initial != uncached_first:
        raise AssertionError("cached and uncached complete solves disagree")
    common = dict(
        graph=graph,
        syndrome=syndrome,
        residual_hw_limit=residual_hw_limit,
        boundary_policy=boundary_policy,
        observable_policy=observable_policy,
        veto_budget=veto_budget,
        oracle=oracle,
        tolerance=tolerance,
    )
    shadow = shadow_score_paths(
        graph, syndrome, legacy_paths, oracle=oracle, tolerance=tolerance
    )
    cost_tx = run_oracle_trajectory(**common, policy="cost", transaction_policy="tx")
    frame_tx = run_oracle_trajectory(**common, policy="frame", transaction_policy="tx")
    frame_partial = run_oracle_trajectory(
        **common, policy="frame", transaction_policy="partial"
    )
    return PhaseAReplayResult(
        shadow=shadow,
        cost_tx=cost_tx,
        frame_tx=frame_tx,
        frame_partial=frame_partial,
        cache_stats=oracle.cache_stats,
        initial_uncached_repeatability_verified=True,
        initial_cached_equivalence_verified=True,
    )


__all__ = [
    "DomainReplayRecord",
    "OracleTrajectoryResult",
    "PhaseAReplayResult",
    "ProposalReplayRecord",
    "ShadowProposalRecord",
    "ShadowReplayResult",
    "run_oracle_trajectory",
    "run_phase_a_replay",
    "shadow_score_paths",
]
