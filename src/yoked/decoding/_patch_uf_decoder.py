"""Packed decoder adapters for confidence-gated patch-local weighted UF.

It provides parameterized, pickleable factories, the frozen V1 policy, and the
complete packed execution boundary used by the public decoder registry.
"""

from __future__ import annotations

import dataclasses
import enum
from fractions import Fraction
from typing import ClassVar, Literal

import numpy as np
import sinter
import stim

from yoked.decoding._patch_uf import (
    BudgetLimits,
    CompiledUFLane,
    LaneOutcome,
    UFCounters,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
    compile_lane,
)
from yoked.decoding._patch_uf_graph import (
    ExactDyadic,
    PatchUFLaneProjection,
    PatchUFProjection,
    compile_patch_uf_projection,
    replay_support,
)
from yoked.decoding._promatch_graph import CompiledPromatchGraph, compile_matching_graph
from yoked.decoding._promatch_layout import compile_layout


GLOBAL_MWPM_DECODER_NAME = "global-mwpm-u0-joint-y2"
ADAPTER_CONTROL_DECODER_NAME = "adapter-control-global-mwpm-v1"
UF_SHADOW_DECODER_NAME = "weighted-uf-shadow-global-mwpm-v1"
PATCH_UF_TREATMENT_DECODER_NAME = (
    "weighted-uf-fullhistory-patchlocal-zeroframe-residual-global-mwpm-v1"
)


_PATCH_UF_V1_SEMANTIC_LIMITS = BudgetLimits(
    growth_event_count=3_403,
    simultaneous_event_batch_count=3_403,
    union_attempt_count=3_171,
    successful_union_count=695,
    forest_edge_count=696,
    absorbed_vertex_count=696,
    peel_operation_count=696,
    heap_push_count=None,
    heap_pop_count=None,
    heap_operation_count=None,
    peak_heap_size=None,
    temporary_memory_units=None,
)
_PATCH_UF_V1_PRODUCTION_LIMITS = BudgetLimits(
    growth_event_count=None,
    simultaneous_event_batch_count=None,
    union_attempt_count=None,
    successful_union_count=None,
    forest_edge_count=None,
    absorbed_vertex_count=None,
    peel_operation_count=None,
    heap_push_count=4_194_304,
    heap_pop_count=4_194_304,
    heap_operation_count=8_388_608,
    peak_heap_size=262_144,
    temporary_memory_units=262_144,
)

# Literal, outcome-independent scientific V1 policy.  The semantic caps are
# authenticated per-lane topology bounds for the selected d=7 cell; the heap
# and temporary-workspace caps are deliberately generous safety ceilings.
PATCH_UF_V1_POLICY = UFPolicy(
    tau=Fraction(0),
    semantic_limits=_PATCH_UF_V1_SEMANTIC_LIMITS,
    production_limits=_PATCH_UF_V1_PRODUCTION_LIMITS,
)


class CaptureMode(enum.Enum):
    """Controls retained immutable UF results, never semantic decisions."""

    NONE = "none"
    METRICS = "metrics"
    TRACE = "trace"


AdapterMode = Literal["adapter-control", "uf-shadow", "treatment"]
PatchStatus = Literal["control", "durable", "aborted"]


@dataclasses.dataclass(frozen=True)
class CompiledPatchUFLane:
    """Compile-time bridge from one YSC lane to the generic UF core graph."""

    lane_id: int
    projection: PatchUFLaneProjection
    graph: UFLaneGraph
    engine: CompiledUFLane


@dataclasses.dataclass(frozen=True)
class PatchTransactionOutcome:
    """Immutable result of validating both basis lanes of one patch."""

    patch_id: int
    status: PatchStatus
    lane_outcomes: tuple[LaneOutcome, ...]
    durable_component_refs: tuple[tuple[int, int], ...]
    durable_support_edge_ids: tuple[int, ...]
    durable_detector_boundary: tuple[int, ...]
    durable_observable_frame: bytes
    durable_exact_weight: ExactDyadic
    abort_reason: str | None

    @property
    def detector_boundary(self) -> tuple[int, ...]:
        return self.durable_detector_boundary

    @property
    def observable_frame(self) -> bytes:
        return self.durable_observable_frame

    @property
    def durable_support_count(self) -> int:
        return len(self.durable_support_edge_ids)

    @property
    def durable_boundary_count(self) -> int:
        return len(self.durable_detector_boundary)

    @property
    def durable_frame_weight(self) -> int:
        return sum(value.bit_count() for value in self.durable_observable_frame)


@dataclasses.dataclass(frozen=True)
class ShotCorrection:
    """Validated patch-atomic correction plan for one detector shot."""

    projection_fingerprint: str
    num_detectors: int
    num_observables: int
    capture: CaptureMode
    lane_outcomes: tuple[LaneOutcome, ...]
    patch_outcomes: tuple[PatchTransactionOutcome, ...]
    durable_support_edge_ids: tuple[int, ...]
    durable_detector_boundary: tuple[int, ...]
    durable_observable_frame: bytes
    durable_exact_weight: ExactDyadic
    input_detector_count: int
    lane_owned_detector_count: int
    residual_detector_count: int
    lane_original_detector_counts: tuple[int, ...] | None
    lane_residual_detector_counts: tuple[int, ...] | None
    original_body_detector_count: int | None
    residual_body_detector_count: int | None
    original_terminal_detector_count: int | None
    residual_terminal_detector_count: int | None
    original_yoke_detector_count: int | None
    residual_yoke_detector_count: int | None

    @property
    def detector_boundary(self) -> tuple[int, ...]:
        return self.durable_detector_boundary

    @property
    def observable_frame(self) -> bytes:
        return self.durable_observable_frame

    @property
    def has_durable_correction(self) -> bool:
        return bool(self.durable_support_edge_ids)

    @property
    def durable_support_count(self) -> int:
        return len(self.durable_support_edge_ids)

    @property
    def durable_boundary_count(self) -> int:
        return len(self.durable_detector_boundary)

    @property
    def durable_frame_weight(self) -> int:
        return sum(value.bit_count() for value in self.durable_observable_frame)

    @property
    def residual_detection_event_count(self) -> int:
        return self.residual_detector_count

    @property
    def original_detector_count(self) -> int:
        return self.input_detector_count

    @property
    def committed_defect_count(self) -> int:
        total = 0
        for patch in self.patch_outcomes:
            if patch.status != "durable":
                continue
            for lane_id, component_index in patch.durable_component_refs:
                outcome = self.lane_outcomes[lane_id]
                matching = [
                    component
                    for component in outcome.completed_components
                    if component.component_index == component_index
                ]
                if len(matching) != 1:
                    raise ValueError("durable component reference is inconsistent")
                total += matching[0].cluster_defect_count
        return total

    @property
    def cluster_summary_complete(self) -> bool:
        return all(
            outcome.status in ("empty", "completed")
            for outcome in self.lane_outcomes
        )

    @property
    def completed_final_component_count(self) -> int:
        return sum(
            len(outcome.completed_components) for outcome in self.lane_outcomes
        )

    @property
    def completed_component_size_histogram(self) -> tuple[tuple[int, int], ...]:
        counts: dict[int, int] = {}
        for outcome in self.lane_outcomes:
            for component in outcome.completed_components:
                size = component.cluster_defect_count
                counts[size] = counts.get(size, 0) + 1
        return tuple(sorted(counts.items()))

    @property
    def maximum_final_component_defect_count(self) -> int | None:
        if not self.cluster_summary_complete:
            return None
        histogram = self.completed_component_size_histogram
        return 0 if not histogram else histogram[-1][0]

    def component_durable_decision(
        self, lane_id: int, component_index: int
    ) -> tuple[bool, str]:
        """Returns transaction-derived durability without altering core records."""

        if lane_id < 0 or lane_id >= len(self.lane_outcomes):
            raise ValueError("lane ID is out of range")
        patch_id = lane_id // 2
        if patch_id >= len(self.patch_outcomes):
            raise ValueError("lane has no patch transaction")
        patch = self.patch_outcomes[patch_id]
        if patch.patch_id != patch_id:
            raise ValueError("patch outcomes are not in dense patch order")
        outcome = self.lane_outcomes[lane_id]
        component = next(
            (
                value
                for value in outcome.completed_components
                if value.component_index == component_index
            ),
            None,
        )
        if component is None:
            raise ValueError("completed component index is out of range")
        durable = (lane_id, component_index) in patch.durable_component_refs
        if durable:
            return True, "committed"
        if patch.status == "aborted":
            return False, str(patch.abort_reason)
        return False, component.primary_gate_reason


def _exact_fraction(value: ExactDyadic) -> Fraction:
    if value.binary_exponent >= 0:
        return Fraction(value.integer << value.binary_exponent)
    return Fraction(value.integer, 1 << -value.binary_exponent)


def _validate_projection_edge(
    projection: PatchUFProjection,
    *,
    edge_id: int,
    owner_lane: int | None,
) -> None:
    if edge_id < 0 or edge_id >= len(projection.support_edges):
        raise ValueError(f"lane edge ID {edge_id} is out of range")
    edge = projection.support_edges[edge_id]
    if edge.edge_id != edge_id:
        raise ValueError("projection support edge IDs must be dense tuple indices")
    if owner_lane is not None and edge.owner_lane != owner_lane:
        raise ValueError(
            f"correction edge {edge_id} belongs to lane {edge.owner_lane}, "
            f"expected {owner_lane}"
        )


def compile_patch_uf_lane_graph(
    projection: PatchUFProjection,
    lane: PatchUFLaneProjection,
    policy: UFPolicy,
) -> CompiledPatchUFLane:
    """Copies one projected lane into the generic immutable UF graph."""

    if not isinstance(projection, PatchUFProjection):
        raise TypeError("projection must be a PatchUFProjection")
    if not isinstance(lane, PatchUFLaneProjection):
        raise TypeError("lane must be a PatchUFLaneProjection")
    _validate_policy(policy)
    if lane.lane_id < 0 or lane.lane_id >= len(projection.lanes):
        raise ValueError("lane ID is out of range")
    if projection.lanes[lane.lane_id] != lane:
        raise ValueError("lane does not belong to the supplied projection")

    edges: list[UFEdge] = []
    for correction in lane.internal_correction_edges:
        _validate_projection_edge(
            projection, edge_id=correction.edge_id, owner_lane=lane.lane_id
        )
        support = projection.support_edges[correction.edge_id]
        edges.append(
            UFEdge(
                edge_id=correction.edge_id,
                source=correction.local_source,
                target=correction.local_target,
                weight=_exact_fraction(
                    projection.exact_weights[correction.exact_weight_index]
                ),
                kind="correction",
                observable_mask=support.observable_mask,
            )
        )
    for boundary in lane.true_boundary_edges:
        _validate_projection_edge(
            projection, edge_id=boundary.edge_id, owner_lane=lane.lane_id
        )
        support = projection.support_edges[boundary.edge_id]
        edges.append(
            UFEdge(
                edge_id=boundary.edge_id,
                source=boundary.local_vertex,
                target=None,
                weight=_exact_fraction(
                    projection.exact_weights[boundary.exact_weight_index]
                ),
                kind="boundary",
                observable_mask=support.observable_mask,
            )
        )
    for port in lane.guard_ports:
        _validate_projection_edge(
            projection, edge_id=port.edge_id, owner_lane=None
        )
        support = projection.support_edges[port.edge_id]
        if support.owner_kind != "global-port" or support.owner_lane is not None:
            raise ValueError(f"guard port {port.edge_id} is locally owned")
        edges.append(
            UFEdge(
                edge_id=port.edge_id,
                source=port.local_vertex,
                target=None,
                weight=_exact_fraction(
                    projection.exact_weights[port.exact_weight_index]
                ),
                kind="port",
                observable_mask=port.observable_mask,
                port_kind=port.port_kind,
            )
        )
    edges.sort(key=lambda edge: (edge.edge_id, edge.kind, edge.source))
    graph = UFLaneGraph(num_vertices=len(lane.global_detector_ids), edges=tuple(edges))
    return CompiledPatchUFLane(lane.lane_id, lane, graph, compile_lane(graph, policy))


def compile_patch_uf_lane_graphs(
    projection: PatchUFProjection,
    policy: UFPolicy,
) -> tuple[CompiledPatchUFLane, ...]:
    """Compiles every lane once, retaining projection order."""

    if not isinstance(projection, PatchUFProjection):
        raise TypeError("projection must be a PatchUFProjection")
    _validate_policy(policy)
    result = tuple(
        compile_patch_uf_lane_graph(projection, lane, policy)
        for lane in projection.lanes
    )
    if tuple(value.lane_id for value in result) != tuple(range(len(result))):
        raise ValueError("projection lane IDs must be dense tuple indices")
    return result


def _validate_policy(policy: UFPolicy) -> None:
    if not isinstance(policy, UFPolicy):
        raise TypeError("policy must be an explicit UFPolicy")


def _validate_packed_batch(data: np.ndarray, *, num_detectors: int) -> np.ndarray:
    raw = np.asarray(data)
    expected_width = (num_detectors + 7) // 8
    if raw.dtype != np.uint8:
        raise ValueError(f"packed detector data must have dtype uint8, got {raw.dtype}")
    if raw.ndim != 2:
        raise ValueError(f"packed detector data must be 2-D, got shape {raw.shape}")
    if raw.shape[1] != expected_width:
        raise ValueError(
            "packed detector width mismatch: "
            f"expected {expected_width}, got {raw.shape[1]}"
        )
    if num_detectors % 8 and raw.shape[0] and raw.shape[1]:
        unused_mask = 0xFF ^ ((1 << (num_detectors % 8)) - 1)
        if np.any(np.bitwise_and(raw[:, -1], unused_mask)):
            raise ValueError("packed detector data has nonzero unused tail bits")
    return raw


def _validate_packed_row(data: np.ndarray, *, num_detectors: int) -> np.ndarray:
    raw = np.asarray(data)
    expected_width = (num_detectors + 7) // 8
    if raw.dtype != np.uint8:
        raise ValueError(f"packed detector data must have dtype uint8, got {raw.dtype}")
    if raw.ndim != 1:
        raise ValueError(f"packed detector shot must be 1-D, got shape {raw.shape}")
    if raw.shape[0] != expected_width:
        raise ValueError(
            "packed detector width mismatch: "
            f"expected {expected_width}, got {raw.shape[0]}"
        )
    _validate_packed_batch(raw.reshape(1, -1), num_detectors=num_detectors)
    return raw


def _normalize_predictions(
    predictions: object,
    *,
    shot_count: int,
    num_observables: int,
) -> np.ndarray:
    width = (num_observables + 7) // 8
    raw = np.asarray(predictions)
    if raw.dtype != np.uint8:
        raise ValueError(
            f"matcher predictions must have dtype uint8, got {raw.dtype}"
        )
    if raw.shape != (shot_count, width):
        raise ValueError(
            "matcher prediction shape mismatch: "
            f"expected {(shot_count, width)}, got {raw.shape}"
        )
    result = np.array(raw, dtype=np.uint8, order="C", copy=True)
    if num_observables % 8 and width:
        result[:, -1] &= (1 << (num_observables % 8)) - 1
    return result


def _decode_complete_batch(
    graph: CompiledPromatchGraph,
    packed: np.ndarray,
    *,
    num_detectors: int,
    num_observables: int,
) -> np.ndarray:
    validated = _validate_packed_batch(packed, num_detectors=num_detectors)
    if validated.shape[0] == 0:
        return np.zeros((0, (num_observables + 7) // 8), dtype=np.uint8)
    # Copy even contiguous input: the caller's storage is never shared with a
    # backend whose mutability contract is outside this adapter.
    backend_input = np.array(validated, dtype=np.uint8, order="C", copy=True)
    predictions = graph.matcher.decode_batch(
        backend_input,
        bit_packed_shots=True,
        bit_packed_predictions=True,
    )
    return _normalize_predictions(
        predictions,
        shot_count=len(backend_input),
        num_observables=num_observables,
    )


def _invoke_backend_prevalidated(
    graph: CompiledPromatchGraph,
    packed: np.ndarray,
) -> object:
    """Invokes only PyMatching; callers own validation, copying, and output checks."""

    return graph.matcher.decode_batch(
        packed,
        bit_packed_shots=True,
        bit_packed_predictions=True,
    )


def _validate_unpacked_shot(shot: np.ndarray, *, num_detectors: int) -> np.ndarray:
    raw = np.asarray(shot)
    if raw.ndim != 1 or raw.shape[0] != num_detectors:
        raise ValueError(
            f"unpacked detector shot must have shape {(num_detectors,)}, got {raw.shape}"
        )
    if raw.dtype != np.uint8:
        raise ValueError(f"unpacked detector shot must have dtype uint8, got {raw.dtype}")
    if np.any(raw > 1):
        raise ValueError("unpacked detector shot values must be bits")
    return raw


def _lane_defects(lane: PatchUFLaneProjection, shot: np.ndarray) -> tuple[int, ...]:
    return tuple(
        local_index
        for local_index, detector_id in enumerate(lane.global_detector_ids)
        if shot[detector_id]
    )


def _xor_boundary(destination: set[int], boundary: tuple[int, ...]) -> None:
    for detector_id in boundary:
        if detector_id in destination:
            destination.remove(detector_id)
        else:
            destination.add(detector_id)


def _xor_frame(destination: bytearray, frame: bytes) -> None:
    if len(destination) != len(frame):
        raise ValueError("observable frame width mismatch")
    for index, value in enumerate(frame):
        destination[index] ^= value


def _empty_patch_outcome(
    patch_id: int,
    *,
    num_observables: int,
    status: PatchStatus = "control",
) -> PatchTransactionOutcome:
    return PatchTransactionOutcome(
        patch_id=patch_id,
        status=status,
        lane_outcomes=(),
        durable_component_refs=(),
        durable_support_edge_ids=(),
        durable_detector_boundary=(),
        durable_observable_frame=bytes((num_observables + 7) // 8),
        durable_exact_weight=ExactDyadic(0, 0),
        abort_reason=None,
    )


def _empty_lane_outcome() -> LaneOutcome:
    return LaneOutcome(
        status="empty",
        completed_components=(),
        censored_components=(),
        counters=UFCounters(),
        censor_reason=None,
        budget_exceeded_set=(),
        primary_budget_cap=None,
        terminal_event_time=Fraction(0),
        last_complete_batch_id=None,
    )


def _plan_control_shot(
    projection: PatchUFProjection,
    shot: np.ndarray,
    *,
    capture: CaptureMode,
) -> ShotCorrection:
    # Exercise precisely the lane selection and patch traversal used by the
    # treatment adapter without running UF or creating a proposal.
    for lane in projection.lanes:
        _lane_defects(lane, shot)
    lane_outcomes = tuple(_empty_lane_outcome() for _ in projection.lanes)
    lane_counts, role_counts = (
        _workload_breakdown(projection, shot)
        if capture is not CaptureMode.NONE
        else (None, (None, None, None))
    )
    return ShotCorrection(
        projection_fingerprint=projection.fingerprint,
        num_detectors=projection.num_detectors,
        num_observables=projection.num_observables,
        capture=capture,
        lane_outcomes=lane_outcomes,
        patch_outcomes=tuple(
            _empty_patch_outcome(
                patch_id, num_observables=projection.num_observables
            )
            for patch_id in range(projection.num_patches)
        ),
        durable_support_edge_ids=(),
        durable_detector_boundary=(),
        durable_observable_frame=bytes((projection.num_observables + 7) // 8),
        durable_exact_weight=ExactDyadic(0, 0),
        input_detector_count=int(np.count_nonzero(shot)),
        lane_owned_detector_count=int(
            sum(
                int(shot[detector_id])
                for detector_id, lane_id in enumerate(projection.detector_lane_id)
                if lane_id is not None
            )
        ),
        residual_detector_count=int(np.count_nonzero(shot)),
        lane_original_detector_counts=lane_counts,
        lane_residual_detector_counts=lane_counts,
        original_body_detector_count=role_counts[0],
        residual_body_detector_count=role_counts[0],
        original_terminal_detector_count=role_counts[1],
        residual_terminal_detector_count=role_counts[1],
        original_yoke_detector_count=role_counts[2],
        residual_yoke_detector_count=role_counts[2],
    )


def _workload_breakdown(
    projection: PatchUFProjection, bits: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    """Returns exact lane and body/terminal/yoke detector-event counts."""

    if bits.ndim != 1 or len(bits) != projection.num_detectors:
        raise ValueError("workload breakdown requires one unpacked detector shot")
    lane_counts = [0] * len(projection.lanes)
    role_counts = {"body": 0, "terminal": 0, "yoke": 0}
    for detector_id, raw in enumerate(bits):
        active = int(raw)
        if active not in (0, 1):
            raise ValueError("unpacked detector data must be binary")
        if not active:
            continue
        lane_id = projection.detector_lane_id[detector_id]
        if lane_id is not None:
            lane_counts[lane_id] += 1
        role_counts[projection.detector_role_kind[detector_id]] += 1
    if sum(lane_counts) != role_counts["body"] + role_counts["terminal"]:
        raise ValueError("lane workload does not reconcile with detector roles")
    return tuple(lane_counts), (
        role_counts["body"],
        role_counts["terminal"],
        role_counts["yoke"],
    )


def _component_global_defects(
    lane: PatchUFLaneProjection,
    local_defects: tuple[int, ...],
) -> tuple[int, ...]:
    result: list[int] = []
    for local_vertex in local_defects:
        if local_vertex < 0 or local_vertex >= len(lane.global_detector_ids):
            raise ValueError("UF component defect vertex is out of range")
        result.append(lane.global_detector_ids[local_vertex])
    if tuple(result) != tuple(sorted(set(result))):
        raise ValueError("UF component defects must be sorted and unique")
    return tuple(result)


def _plan_patch(
    *,
    projection: PatchUFProjection,
    compiled_lanes: tuple[CompiledPatchUFLane, ...],
    patch_id: int,
    shot: np.ndarray,
    policy: UFPolicy,
) -> PatchTransactionOutcome:
    lane_ids = projection.patch_lane_ids[patch_id]
    outcomes: list[LaneOutcome] = []
    for lane_id in lane_ids:
        compiled = compiled_lanes[lane_id]
        if compiled.lane_id != lane_id:
            raise ValueError("compiled lane tuple is not indexed by lane ID")
        outcomes.append(
            compiled.engine.run(_lane_defects(compiled.projection, shot))
        )

    censored = [outcome for outcome in outcomes if outcome.status == "censored"]
    if censored:
        reasons = {outcome.censor_reason for outcome in censored}
        if "budget-exhaustion" in reasons:
            abort_reason = "budget-exhaustion-patch-abort"
        elif reasons == {"local-incomplete-neutralization"}:
            abort_reason = "local-incomplete-neutralization-patch-abort"
        else:
            raise ValueError(
                "unsupported UF lane censor reason set "
                f"{sorted(reasons, key=repr)!r}"
            )
        return dataclasses.replace(
            _empty_patch_outcome(
                patch_id,
                num_observables=projection.num_observables,
                status="aborted",
            ),
            lane_outcomes=tuple(outcomes),
            abort_reason=abort_reason,
        )

    support_ids: set[int] = set()
    component_refs: list[tuple[int, int]] = []
    expected_boundary: set[int] = set()
    for lane_id, outcome in zip(lane_ids, outcomes):
        if outcome.status not in ("empty", "completed"):
            raise ValueError(f"unsupported UF lane status {outcome.status!r}")
        lane = compiled_lanes[lane_id].projection
        for component in outcome.completed_components:
            support = tuple(component.peeled_support_edge_ids)
            if support != tuple(sorted(set(support))):
                raise ValueError("UF component support must be sorted and unique")
            replay = replay_support(
                projection, support, expected_owner_lane=lane_id
            )
            component_boundary = _component_global_defects(
                lane, component.original_defects
            )
            if replay.detector_boundary != component_boundary:
                raise ValueError("UF component support has an inconsistent boundary")
            if any(replay.observable_mask):
                raise ValueError("UF V1 component support has a nonzero frame")
            if component.gate_decision == "eligible":
                overlap = support_ids.intersection(support)
                if overlap:
                    raise ValueError(
                        f"eligible UF component supports overlap at {sorted(overlap)!r}"
                    )
                support_ids.update(support)
                component_refs.append((lane_id, component.component_index))
                _xor_boundary(expected_boundary, component_boundary)
            elif component.gate_decision != "deferred":
                raise ValueError(
                    f"unsupported UF gate decision {component.gate_decision!r}"
                )

    support = tuple(sorted(support_ids))
    replay = replay_support(projection, support)
    if replay.detector_boundary != tuple(sorted(expected_boundary)):
        raise ValueError("patch aggregate support has an inconsistent boundary")
    if any(replay.observable_mask):
        raise ValueError("UF V1 patch support has a nonzero observable frame")
    return PatchTransactionOutcome(
        patch_id=patch_id,
        status="durable",
        lane_outcomes=tuple(outcomes),
        durable_component_refs=tuple(component_refs),
        durable_support_edge_ids=support,
        durable_detector_boundary=replay.detector_boundary,
        durable_observable_frame=replay.observable_mask,
        durable_exact_weight=replay.exact_weight,
        abort_reason=None,
    )


def _plan_shot(
    *,
    projection: PatchUFProjection,
    compiled_lanes: tuple[CompiledPatchUFLane, ...],
    shot: np.ndarray,
    policy: UFPolicy,
    capture: CaptureMode,
) -> ShotCorrection:
    outcomes = tuple(
        _plan_patch(
            projection=projection,
            compiled_lanes=compiled_lanes,
            patch_id=patch_id,
            shot=shot,
            policy=policy,
        )
        for patch_id in range(projection.num_patches)
    )
    support_ids: set[int] = set()
    expected_boundary: set[int] = set()
    expected_frame = bytearray((projection.num_observables + 7) // 8)
    for outcome in outcomes:
        if outcome.status != "durable":
            continue
        overlap = support_ids.intersection(outcome.durable_support_edge_ids)
        if overlap:
            raise ValueError(
                f"durable patch supports overlap at {sorted(overlap)!r}"
            )
        support_ids.update(outcome.durable_support_edge_ids)
        _xor_boundary(expected_boundary, outcome.durable_detector_boundary)
        _xor_frame(expected_frame, outcome.durable_observable_frame)
    support = tuple(sorted(support_ids))
    replay = replay_support(projection, support)
    if replay.detector_boundary != tuple(sorted(expected_boundary)):
        raise ValueError("shot aggregate support has an inconsistent boundary")
    if replay.observable_mask != bytes(expected_frame):
        raise ValueError("shot aggregate support has an inconsistent frame")
    if any(replay.observable_mask):
        raise ValueError("UF V1 shot support has a nonzero observable frame")
    residual_bits = np.array(shot, dtype=np.uint8, copy=True)
    for detector_id in replay.detector_boundary:
        residual_bits[detector_id] ^= 1
    lane_outcomes_by_id: dict[int, LaneOutcome] = {}
    for patch_outcome in outcomes:
        for lane_id, lane_outcome in zip(
            projection.patch_lane_ids[patch_outcome.patch_id],
            patch_outcome.lane_outcomes,
        ):
            if lane_id in lane_outcomes_by_id:
                raise ValueError(f"duplicate lane outcome {lane_id}")
            lane_outcomes_by_id[lane_id] = lane_outcome
    if tuple(sorted(lane_outcomes_by_id)) != tuple(range(len(projection.lanes))):
        raise ValueError("shot plan did not produce one outcome per lane")
    if capture is CaptureMode.NONE:
        original_lane_counts = residual_lane_counts = None
        original_role_counts = residual_role_counts = (None, None, None)
    else:
        original_lane_counts, original_role_counts = _workload_breakdown(
            projection, shot
        )
        residual_lane_counts, residual_role_counts = _workload_breakdown(
            projection, residual_bits
        )
    return ShotCorrection(
        projection_fingerprint=projection.fingerprint,
        num_detectors=projection.num_detectors,
        num_observables=projection.num_observables,
        capture=capture,
        lane_outcomes=tuple(
            lane_outcomes_by_id[lane_id]
            for lane_id in range(len(projection.lanes))
        ),
        patch_outcomes=outcomes,
        durable_support_edge_ids=support,
        durable_detector_boundary=replay.detector_boundary,
        durable_observable_frame=replay.observable_mask,
        durable_exact_weight=replay.exact_weight,
        input_detector_count=int(np.count_nonzero(shot)),
        lane_owned_detector_count=int(
            sum(
                int(shot[detector_id])
                for detector_id, lane_id in enumerate(projection.detector_lane_id)
                if lane_id is not None
            )
        ),
        residual_detector_count=int(np.count_nonzero(residual_bits)),
        lane_original_detector_counts=original_lane_counts,
        lane_residual_detector_counts=residual_lane_counts,
        original_body_detector_count=original_role_counts[0],
        residual_body_detector_count=residual_role_counts[0],
        original_terminal_detector_count=original_role_counts[1],
        residual_terminal_detector_count=residual_role_counts[1],
        original_yoke_detector_count=original_role_counts[2],
        residual_yoke_detector_count=residual_role_counts[2],
    )


def apply_shot_correction(
    original_packed_shot: np.ndarray,
    correction: ShotCorrection,
    *,
    projection: PatchUFProjection | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies one already planned correction to a fresh packed-shot copy."""

    if not isinstance(correction, ShotCorrection):
        raise TypeError("correction must be a ShotCorrection")
    original = _validate_packed_row(
        original_packed_shot, num_detectors=correction.num_detectors
    )
    if projection is not None:
        if not isinstance(projection, PatchUFProjection):
            raise TypeError("projection must be a PatchUFProjection")
        if correction.projection_fingerprint != projection.fingerprint:
            raise ValueError("shot correction projection fingerprint mismatch")
        replay = replay_support(projection, correction.durable_support_edge_ids)
        if replay.detector_boundary != correction.durable_detector_boundary:
            raise ValueError("shot correction boundary fails independent replay")
        if replay.observable_mask != correction.durable_observable_frame:
            raise ValueError("shot correction frame fails independent replay")
        if replay.exact_weight != correction.durable_exact_weight:
            raise ValueError("shot correction weight fails independent replay")
    frame_width = (correction.num_observables + 7) // 8
    if len(correction.durable_observable_frame) != frame_width:
        raise ValueError("shot correction observable frame width mismatch")
    if correction.num_observables % 8 and frame_width:
        if correction.durable_observable_frame[-1] >> (
            correction.num_observables % 8
        ):
            raise ValueError("shot correction has nonzero observable tail bits")
    if tuple(correction.durable_detector_boundary) != tuple(
        sorted(set(correction.durable_detector_boundary))
    ):
        raise ValueError("shot correction boundary must be sorted and unique")

    residual = np.array(original, dtype=np.uint8, order="C", copy=True)
    for detector_id in correction.durable_detector_boundary:
        if detector_id < 0 or detector_id >= correction.num_detectors:
            raise ValueError(f"shot correction detector {detector_id} is out of range")
        residual[detector_id // 8] ^= 1 << (detector_id % 8)
    frame = np.frombuffer(correction.durable_observable_frame, dtype=np.uint8).copy()
    return residual, frame


@dataclasses.dataclass
class CompiledGlobalMWPMDecoder(sinter.CompiledDecoder):
    """Minimal strict packed wrapper around the complete Global MWPM graph."""

    graph: CompiledPromatchGraph
    num_detectors: int
    num_observables: int

    def decode_backend_original_bit_packed(self, packed: np.ndarray) -> np.ndarray:
        """Untimed backend-original helper used by latency workload setup."""

        return _decode_complete_batch(
            self.graph,
            packed,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )

    def decode_backend_residual_bit_packed(self, packed: np.ndarray) -> np.ndarray:
        """Untimed backend-residual helper; no preprocessing is performed."""

        return self.decode_backend_original_bit_packed(packed)

    decode_backend_original = decode_backend_original_bit_packed
    decode_backend_residual = decode_backend_residual_bit_packed

    def invoke_backend_prevalidated(self, packed: np.ndarray) -> object:
        """Calls only ``matcher.decode_batch`` for an already frozen workload."""

        return _invoke_backend_prevalidated(self.graph, packed)

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        return self.decode_backend_original_bit_packed(
            bit_packed_detection_event_data
        )

    def decode_shot_bit_packed(self, packed_detection_event_data: np.ndarray) -> np.ndarray:
        row = _validate_packed_row(
            packed_detection_event_data, num_detectors=self.num_detectors
        )
        return self.decode_shots_bit_packed(
            bit_packed_detection_event_data=row.reshape(1, -1)
        )[0]


@dataclasses.dataclass
class CompiledPatchUFAdapter(sinter.CompiledDecoder):
    """Shared compiled adapter for control, shadow, and treatment paths."""

    graph: CompiledPromatchGraph
    projection: PatchUFProjection
    compiled_lanes: tuple[CompiledPatchUFLane, ...]
    policy: UFPolicy
    num_detectors: int
    num_observables: int

    adapter_mode: ClassVar[AdapterMode]

    def __post_init__(self) -> None:
        _validate_policy(self.policy)
        if self.projection.num_detectors != self.num_detectors:
            raise ValueError("projection detector count mismatch")
        if self.projection.num_observables != self.num_observables:
            raise ValueError("projection observable count mismatch")
        if len(self.compiled_lanes) != len(self.projection.lanes):
            raise ValueError("compiled lane count mismatch")
        if any(value.engine.policy != self.policy for value in self.compiled_lanes):
            raise ValueError("compiled lane policy mismatch")

    def plan_shot(
        self,
        unpacked_detection_events: np.ndarray,
        *,
        capture: CaptureMode,
    ) -> ShotCorrection:
        if not isinstance(capture, CaptureMode):
            raise TypeError("capture must be a CaptureMode")
        shot = _validate_unpacked_shot(
            unpacked_detection_events, num_detectors=self.num_detectors
        )
        if self.adapter_mode == "adapter-control":
            return _plan_control_shot(self.projection, shot, capture=capture)
        return _plan_shot(
            projection=self.projection,
            compiled_lanes=self.compiled_lanes,
            shot=shot,
            policy=self.policy,
            capture=capture,
        )

    def apply_shot_correction(
        self,
        original_packed_shot: np.ndarray,
        correction: ShotCorrection,
    ) -> tuple[np.ndarray, np.ndarray]:
        return apply_shot_correction(
            original_packed_shot, correction, projection=self.projection
        )

    def precompute_residual_batch(
        self,
        bit_packed_detection_event_data: np.ndarray,
        *,
        capture: CaptureMode,
    ) -> tuple[np.ndarray, np.ndarray, tuple[ShotCorrection, ...]]:
        """Plans/applies a batch, retaining corrections only for telemetry."""

        if not isinstance(capture, CaptureMode):
            raise TypeError("capture must be a CaptureMode")
        packed = _validate_packed_batch(
            bit_packed_detection_event_data, num_detectors=self.num_detectors
        )
        if packed.shape[0] == 0:
            return (
                np.zeros(packed.shape, dtype=np.uint8),
                np.zeros((0, (self.num_observables + 7) // 8), dtype=np.uint8),
                (),
            )
        if capture is CaptureMode.NONE:
            residual = np.empty(packed.shape, dtype=np.uint8)
            frames = np.zeros(
                (len(packed), (self.num_observables + 7) // 8),
                dtype=np.uint8,
            )
            for shot_index, packed_shot in enumerate(packed):
                unpacked_shot = np.unpackbits(
                    packed_shot,
                    count=self.num_detectors,
                    bitorder="little",
                )
                correction = self.plan_shot(
                    unpacked_shot, capture=CaptureMode.NONE
                )
                corrected, frame = self.apply_shot_correction(
                    packed_shot, correction
                )
                residual[shot_index] = corrected
                frames[shot_index] = frame
                # NONE is explicitly non-retaining: release the complete
                # lane/patch/component tree before planning the next shot.
                del correction
            return residual, frames, ()
        corrections = self.plan_batch(packed, capture=capture)
        residual = np.empty(packed.shape, dtype=np.uint8)
        frames = np.zeros(
            (len(packed), (self.num_observables + 7) // 8), dtype=np.uint8
        )
        for shot_index, correction in enumerate(corrections):
            corrected, frame = self.apply_shot_correction(
                packed[shot_index], correction
            )
            residual[shot_index] = corrected
            frames[shot_index] = frame
        return residual, frames, corrections

    def _plan_and_discard_batch(self, packed: np.ndarray) -> None:
        """Streams NONE plans without retaining a correction batch."""

        for packed_shot in packed:
            unpacked_shot = np.unpackbits(
                packed_shot,
                count=self.num_detectors,
                bitorder="little",
            )
            correction = self.plan_shot(
                unpacked_shot, capture=CaptureMode.NONE
            )
            del correction

    def plan_batch(
        self,
        bit_packed_detection_event_data: np.ndarray,
        *,
        capture: CaptureMode,
    ) -> tuple[ShotCorrection, ...]:
        """Runs lane selection/UF/gating only; never constructs residual storage."""

        if not isinstance(capture, CaptureMode):
            raise TypeError("capture must be a CaptureMode")
        packed = _validate_packed_batch(
            bit_packed_detection_event_data, num_detectors=self.num_detectors
        )
        if packed.shape[0] == 0:
            return ()
        unpacked = np.unpackbits(
            packed, axis=1, count=self.num_detectors, bitorder="little"
        )
        return tuple(self.plan_shot(shot, capture=capture) for shot in unpacked)

    def decode_backend_original_bit_packed(self, packed: np.ndarray) -> np.ndarray:
        """Untimed complete-backend helper on a prevalidated original workload."""

        return _decode_complete_batch(
            self.graph,
            packed,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )

    def decode_backend_residual_bit_packed(self, packed: np.ndarray) -> np.ndarray:
        """Untimed complete-backend helper on a precomputed residual workload."""

        return _decode_complete_batch(
            self.graph,
            packed,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )

    decode_backend_original = decode_backend_original_bit_packed
    decode_backend_residual = decode_backend_residual_bit_packed

    def invoke_backend_prevalidated(self, packed: np.ndarray) -> object:
        """Calls only ``matcher.decode_batch`` for an already frozen workload."""

        return _invoke_backend_prevalidated(self.graph, packed)

    def decode_shots_bit_packed_with_capture(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
        capture: CaptureMode,
    ) -> tuple[np.ndarray, tuple[ShotCorrection, ...]]:
        if not isinstance(capture, CaptureMode):
            raise TypeError("capture must be a CaptureMode")
        packed = _validate_packed_batch(
            bit_packed_detection_event_data, num_detectors=self.num_detectors
        )
        if packed.shape[0] == 0:
            return (
                np.zeros(
                    (0, (self.num_observables + 7) // 8), dtype=np.uint8
                ),
                (),
            )

        if capture is CaptureMode.NONE:
            if self.adapter_mode in ("adapter-control", "uf-shadow"):
                self._plan_and_discard_batch(packed)
                predictions = self.decode_backend_original_bit_packed(packed)
            elif self.adapter_mode == "treatment":
                residual, frames, corrections = self.precompute_residual_batch(
                    packed, capture=CaptureMode.NONE
                )
                if corrections:
                    raise AssertionError("NONE residual planning retained telemetry")
                predictions = self.decode_backend_residual_bit_packed(residual)
                predictions ^= frames
                if self.num_observables % 8 and predictions.shape[1]:
                    predictions[:, -1] &= (1 << (self.num_observables % 8)) - 1
            else:  # pragma: no cover - subclasses freeze the mode set.
                raise AssertionError(f"unknown adapter mode {self.adapter_mode!r}")
            return predictions, ()

        if self.adapter_mode == "adapter-control":
            corrections = self.plan_batch(packed, capture=capture)
            predictions = self.decode_backend_original_bit_packed(packed)
            return predictions, corrections

        if self.adapter_mode == "uf-shadow":
            corrections = self.plan_batch(packed, capture=capture)
            predictions = self.decode_backend_original_bit_packed(packed)
        elif self.adapter_mode == "treatment":
            residual, frames, corrections = self.precompute_residual_batch(
                packed, capture=capture
            )
            predictions = self.decode_backend_residual_bit_packed(residual)
            predictions ^= frames
            if self.num_observables % 8 and predictions.shape[1]:
                predictions[:, -1] &= (1 << (self.num_observables % 8)) - 1
        else:  # pragma: no cover - subclasses freeze the closed mode set.
            raise AssertionError(f"unknown adapter mode {self.adapter_mode!r}")
        return predictions, corrections

    def decode_shots_bit_packed_with_telemetry(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> tuple[np.ndarray, tuple[ShotCorrection, ...]]:
        """Standard immutable metrics entry point used by the collector."""

        return self.decode_shots_bit_packed_with_capture(
            bit_packed_detection_event_data=bit_packed_detection_event_data,
            capture=CaptureMode.METRICS,
        )

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        predictions, _ = self.decode_shots_bit_packed_with_capture(
            bit_packed_detection_event_data=bit_packed_detection_event_data,
            capture=CaptureMode.NONE,
        )
        return predictions

    def decode_shot_bit_packed(self, packed_detection_event_data: np.ndarray) -> np.ndarray:
        row = _validate_packed_row(
            packed_detection_event_data, num_detectors=self.num_detectors
        )
        return self.decode_shots_bit_packed(
            bit_packed_detection_event_data=row.reshape(1, -1)
        )[0]


class CompiledAdapterControlDecoder(CompiledPatchUFAdapter):
    adapter_mode: ClassVar[AdapterMode] = "adapter-control"


class CompiledUFShadowDecoder(CompiledPatchUFAdapter):
    adapter_mode: ClassVar[AdapterMode] = "uf-shadow"


class CompiledPatchUFTreatmentDecoder(CompiledPatchUFAdapter):
    adapter_mode: ClassVar[AdapterMode] = "treatment"


def _compile_graph(dem: stim.DetectorErrorModel) -> CompiledPromatchGraph:
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"dem must be a stim.DetectorErrorModel, got {type(dem)!r}")
    layout = compile_layout(dem, mode="fullhistory")
    return compile_matching_graph(
        dem,
        layout,
        require_zero_frame=False,
        retain_cross_lane_edges=True,
    )


def _compile_patch_adapter(
    dem: stim.DetectorErrorModel,
    *,
    policy: UFPolicy,
    compiled_type: type[CompiledPatchUFAdapter],
) -> CompiledPatchUFAdapter:
    _validate_policy(policy)
    graph = _compile_graph(dem)
    projection = compile_patch_uf_projection(dem, graph)
    return compiled_type(
        graph=graph,
        projection=projection,
        compiled_lanes=compile_patch_uf_lane_graphs(projection, policy),
        policy=policy,
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
    )


@dataclasses.dataclass(frozen=True)
class GlobalMWPMDecoder(sinter.Decoder):
    """Pickleable factory for the strict direct Global MWPM control."""

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledGlobalMWPMDecoder:
        graph = _compile_graph(dem)
        return CompiledGlobalMWPMDecoder(
            graph=graph,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )


@dataclasses.dataclass(frozen=True)
class AdapterControlDecoder(sinter.Decoder):
    """Pickleable factory for adapter/lane-selection overhead control."""

    policy: UFPolicy

    def __post_init__(self) -> None:
        _validate_policy(self.policy)

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledAdapterControlDecoder:
        return _compile_patch_adapter(
            dem, policy=self.policy, compiled_type=CompiledAdapterControlDecoder
        )


@dataclasses.dataclass(frozen=True)
class UFShadowDecoder(sinter.Decoder):
    """Pickleable factory that plans UF but sends the original batch to MWPM."""

    policy: UFPolicy

    def __post_init__(self) -> None:
        _validate_policy(self.policy)

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledUFShadowDecoder:
        return _compile_patch_adapter(
            dem, policy=self.policy, compiled_type=CompiledUFShadowDecoder
        )


@dataclasses.dataclass(frozen=True)
class PatchUFTreatmentDecoder(sinter.Decoder):
    """Pickleable parameterized treatment factory; policy has no defaults."""

    policy: UFPolicy

    def __post_init__(self) -> None:
        _validate_policy(self.policy)

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledPatchUFTreatmentDecoder:
        return _compile_patch_adapter(
            dem, policy=self.policy, compiled_type=CompiledPatchUFTreatmentDecoder
        )


def plan_shot(
    adapter: CompiledPatchUFAdapter,
    unpacked_detection_events: np.ndarray,
    *,
    capture: CaptureMode,
) -> ShotCorrection:
    """Functional wrapper for the compiled adapter's pure planning stage."""

    if not isinstance(adapter, CompiledPatchUFAdapter):
        raise TypeError("adapter must be a CompiledPatchUFAdapter")
    return adapter.plan_shot(unpacked_detection_events, capture=capture)


__all__ = [
    "ADAPTER_CONTROL_DECODER_NAME",
    "GLOBAL_MWPM_DECODER_NAME",
    "PATCH_UF_TREATMENT_DECODER_NAME",
    "PATCH_UF_V1_POLICY",
    "UF_SHADOW_DECODER_NAME",
    "AdapterControlDecoder",
    "CaptureMode",
    "CompiledAdapterControlDecoder",
    "CompiledGlobalMWPMDecoder",
    "CompiledPatchUFAdapter",
    "CompiledPatchUFLane",
    "CompiledPatchUFTreatmentDecoder",
    "CompiledUFShadowDecoder",
    "GlobalMWPMDecoder",
    "PatchTransactionOutcome",
    "PatchUFTreatmentDecoder",
    "ShotCorrection",
    "UFShadowDecoder",
    "apply_shot_correction",
    "compile_patch_uf_lane_graph",
    "compile_patch_uf_lane_graphs",
    "plan_shot",
]
