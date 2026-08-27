"""Pure Pinball-style schedule compilation and transactional predecoding.

This is a graph-native adaptation of the nine-stage Pinball pipeline from
Knapen et al., arXiv:2512.09807v2 (MIT artifact commit
``8f16f24b621aacfaa4f456a2aeec8df088faf3a7``).  It is deliberately called
*Pinball-style*: the maintained yoked circuits contain several CZ-coupled
patches and both check bases, unlike the artifact's single-patch CNOT model.

The module has no Sinter-facing code.  Compilation is deterministic and fails
closed on unrecognised within-domain topology.  Execution is a whole-shot
transaction: tentative length-one corrections are committed only when every
inner syndrome is cleared.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Literal, TypeAlias

import numpy as np

from yoked.decoding._promatch_graph import CompiledPromatchGraph, Edge
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1TerminalDetector,
)


PinballBasis: TypeAlias = Literal["X", "Z"]
PinballStage: TypeAlias = Literal[
    "M", "B1", "B2", "B3", "B4", "ST1", "ST2", "H", "E"
]

PINBALL_STAGE_ORDER: tuple[PinballStage, ...] = (
    "M",
    "B1",
    "B2",
    "B3",
    "B4",
    "ST1",
    "ST2",
    "H",
    "E",
)

_STAGE_RANK = {stage: rank for rank, stage in enumerate(PINBALL_STAGE_ORDER)}
_InnerRole: TypeAlias = L1BodyDetector | L1TerminalDetector


@dataclasses.dataclass(frozen=True)
class PinballPrimitive:
    """One canonical graph edge assigned to a Pinball-style pipeline stage."""

    edge_id: int
    source: int
    target: int | None
    patch_id: int
    check_basis: PinballBasis
    sweep_time: int
    stage: PinballStage


@dataclasses.dataclass(frozen=True)
class PinballStageSchedule:
    """Conflict-free primitives executed in parallel at one streaming stage."""

    patch_id: int
    check_basis: PinballBasis
    sweep_time: int
    stage: PinballStage
    primitives: tuple[PinballPrimitive, ...]


@dataclasses.dataclass(frozen=True)
class CompiledPinballSchedule:
    """Deterministic semantic schedule bound to one compiled matching graph."""

    graph_fingerprint: str
    fingerprint: str
    num_detectors: int
    num_observables: int
    stages: tuple[PinballStageSchedule, ...]

    @property
    def primitives(self) -> tuple[PinballPrimitive, ...]:
        """Returns all primitives in exact execution order."""

        return tuple(
            primitive for stage in self.stages for primitive in stage.primitives
        )


@dataclasses.dataclass(frozen=True)
class PinballResult:
    """Committed and tentative outputs of one whole-shot transaction.

    ``stage_match_counts`` is aligned one-for-one with ``schedule.stages``.
    On a complex shot the durable syndrome is the original input, the durable
    frame/support are zero/empty, and all attempted work remains available in
    the ``tentative_*`` audit fields.
    """

    complex: bool
    residual_syndrome: np.ndarray
    observable_frame: np.ndarray
    edge_support: tuple[int, ...]
    stage_match_counts: tuple[int, ...]
    tentative_residual_syndrome: np.ndarray
    tentative_observable_frame: np.ndarray
    tentative_edge_support: tuple[int, ...]


def _inner_role(role: object) -> bool:
    return isinstance(role, (L1BodyDetector, L1TerminalDetector))


def _local_coordinates(
    graph: CompiledPromatchGraph,
    detector_id: int,
    role: _InnerRole,
) -> tuple[float, float, int]:
    x, y, *_ = graph.layout.coordinates[detector_id]
    local_x = x - role.patch_id * graph.layout.pitch
    if role.check_basis == "X":
        u, v = local_x, y
    else:
        u, v = y, local_x
    return u, v, role.time


def _close(value: float, expected: float) -> bool:
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-7)


def _classify_inner_edge(
    graph: CompiledPromatchGraph,
    edge: Edge,
) -> tuple[PinballStage, int]:
    """Returns ``(stage, sweep_time)`` for a supported local inner edge."""

    if edge.target is None:
        raise AssertionError("boundary edge passed to inner-edge classifier")
    source_role = edge.source_role
    target_role = edge.target_role
    if not isinstance(
        source_role, (L1BodyDetector, L1TerminalDetector)
    ) or not isinstance(target_role, (L1BodyDetector, L1TerminalDetector)):
        raise AssertionError("non-inner role passed to inner-edge classifier")

    a_u, a_v, a_t = _local_coordinates(graph, edge.source, source_role)
    b_u, b_v, b_t = _local_coordinates(graph, edge.target, target_role)
    dt = b_t - a_t
    du = b_u - a_u
    dv = b_v - a_v

    if abs(dt) == 1 and _close(du, 0) and _close(dv, 0):
        return "M", max(a_t, b_t)

    if dt == 0 and _close(abs(du), 1) and _close(abs(dv), 1):
        # The upstream implementation uses alternate rows as centers.  The
        # normalized u coordinate makes the X- and Z-check domains symmetric.
        source_center = round(a_u - 0.5) % 2 == 0
        target_center = round(b_u - 0.5) % 2 == 0
        if source_center == target_center:
            raise ValueError(
                f"Pinball-style B edge {edge.edge_id} has ambiguous center parity"
            )
        if source_center:
            center_to_neighbor = (round(du), round(dv))
        else:
            center_to_neighbor = (round(-du), round(-dv))
        stage_by_delta: dict[tuple[int, int], PinballStage] = {
            (+1, +1): "B1",
            (+1, -1): "B2",
            (-1, -1): "B3",
            (-1, +1): "B4",
        }
        try:
            stage = stage_by_delta[center_to_neighbor]
        except KeyError as ex:
            raise ValueError(
                f"Pinball-style B edge {edge.edge_id} has unsupported orientation "
                f"{center_to_neighbor}"
            ) from ex
        return stage, a_t

    if abs(dt) == 1 and _close(abs(du), 1) and _close(abs(dv), 1):
        later_minus_earlier_u = du if dt > 0 else -du
        if later_minus_earlier_u < 0:
            return "ST1", max(a_t, b_t)
        if later_minus_earlier_u > 0:
            return "ST2", max(a_t, b_t)
        raise AssertionError("diagonal spacetime edge has zero u displacement")

    if abs(dt) == 1 and _close(du, 0) and _close(abs(dv), 2):
        return "H", max(a_t, b_t)

    raise ValueError(
        "unsupported within-patch/basis Pinball-style topology: "
        f"edge={edge.edge_id}, dt={dt}, du={du}, dv={dv}, "
        f"source_role={source_role!r}, target_role={target_role!r}"
    )


def _primitive_sort_key(primitive: PinballPrimitive) -> tuple[int, int, int]:
    target = primitive.source if primitive.target is None else primitive.target
    return primitive.source, target, primitive.edge_id


def _validate_compiled_schedule(
    graph: CompiledPromatchGraph,
    stages: tuple[PinballStageSchedule, ...],
) -> None:
    expected_keys = sorted(
        (
            patch_id,
            basis,
            sweep_time,
            _STAGE_RANK[stage],
        )
        for patch_id in range(graph.layout.num_patches)
        for basis in ("X", "Z")
        for sweep_time in range(graph.layout.rounds + 1)
        for stage in PINBALL_STAGE_ORDER
    )
    # One extra E-only flush follows the terminal layer.
    expected_keys.extend(
        (patch_id, basis, graph.layout.rounds + 1, _STAGE_RANK["E"])
        for patch_id in range(graph.layout.num_patches)
        for basis in ("X", "Z")
    )
    expected_keys.sort()
    actual_keys = [
        (stage.patch_id, stage.check_basis, stage.sweep_time, _STAGE_RANK[stage.stage])
        for stage in stages
    ]
    if actual_keys != expected_keys:
        raise AssertionError("Pinball-style stage schedule is incomplete or misordered")

    seen_edge_ids: set[int] = set()
    for stage_schedule in stages:
        used_detectors: set[int] = set()
        if (
            tuple(sorted(stage_schedule.primitives, key=_primitive_sort_key))
            != stage_schedule.primitives
        ):
            raise AssertionError(
                "Pinball-style primitives are not deterministically sorted"
            )
        for primitive in stage_schedule.primitives:
            if (
                primitive.patch_id != stage_schedule.patch_id
                or primitive.check_basis != stage_schedule.check_basis
                or primitive.sweep_time != stage_schedule.sweep_time
                or primitive.stage != stage_schedule.stage
            ):
                raise AssertionError(
                    "primitive metadata does not match its stage schedule"
                )
            if primitive.edge_id in seen_edge_ids:
                raise ValueError(
                    f"canonical edge {primitive.edge_id} was assigned more than once"
                )
            seen_edge_ids.add(primitive.edge_id)
            if primitive.edge_id < 0 or primitive.edge_id >= len(graph.edges):
                raise ValueError(f"primitive has invalid edge ID {primitive.edge_id}")
            edge = graph.edges[primitive.edge_id]
            if edge.edge_id != primitive.edge_id or (
                edge.source,
                edge.target,
            ) != (primitive.source, primitive.target):
                raise ValueError(
                    f"primitive {primitive.edge_id} does not match the canonical graph"
                )
            endpoints = (primitive.source,) if primitive.target is None else (
                primitive.source,
                primitive.target,
            )
            if any(detector in used_detectors for detector in endpoints):
                raise ValueError(
                    "Pinball-style stage is not conflict-free: "
                    f"patch={stage_schedule.patch_id}, "
                    f"basis={stage_schedule.check_basis}, "
                    f"t={stage_schedule.sweep_time}, stage={stage_schedule.stage}"
                )
            used_detectors.update(endpoints)

    boundary_count: dict[int, int] = {}
    measurement_pairs: dict[tuple[int, str, float, float, int], int] = {}
    sites_by_domain_time: dict[tuple[int, str, int], set[tuple[float, float]]] = {}
    for detector_id, role in enumerate(graph.layout.roles):
        if not isinstance(role, (L1BodyDetector, L1TerminalDetector)):
            continue
        u, v, time = _local_coordinates(graph, detector_id, role)
        sites_by_domain_time.setdefault(
            (role.patch_id, role.check_basis, time), set()
        ).add((u, v))
    for primitive in (p for stage in stages for p in stage.primitives):
        if primitive.stage == "E":
            boundary_count[primitive.source] = (
                boundary_count.get(primitive.source, 0) + 1
            )
        elif primitive.stage == "M":
            edge = graph.edges[primitive.edge_id]
            roles = (edge.source_role, edge.target_role)
            ids = (edge.source, edge.target)
            earlier_index = (
                0 if roles[0].time < roles[1].time else 1  # type: ignore[union-attr]
            )
            earlier_role = roles[earlier_index]
            assert isinstance(earlier_role, (L1BodyDetector, L1TerminalDetector))
            u, v, time = _local_coordinates(graph, ids[earlier_index], earlier_role)
            key = (
                earlier_role.patch_id,
                earlier_role.check_basis,
                u,
                v,
                time,
            )
            measurement_pairs[key] = measurement_pairs.get(key, 0) + 1
    repeated_boundaries = sorted(k for k, count in boundary_count.items() if count != 1)
    if repeated_boundaries:
        raise ValueError(
            "inner detector has more than one Pinball-style E edge: "
            f"{repeated_boundaries[:8]}"
        )

    for patch_id in range(graph.layout.num_patches):
        for basis in ("X", "Z"):
            reference = sites_by_domain_time[(patch_id, basis, 0)]
            for time in range(graph.layout.rounds + 1):
                if sites_by_domain_time.get((patch_id, basis, time)) != reference:
                    raise ValueError(
                        "Pinball-style M validation found inconsistent spatial sites"
                    )
            for time in range(graph.layout.rounds):
                for u, v in reference:
                    key = (patch_id, basis, u, v, time)
                    if measurement_pairs.get(key, 0) != 1:
                        raise ValueError(
                            "Pinball-style M stage does not cover every site across "
                            f"consecutive layers: missing/duplicate {key!r}"
                        )


def _schedule_fingerprint(
    graph: CompiledPromatchGraph,
    stages: tuple[PinballStageSchedule, ...],
) -> str:
    payload = {
        "schema": "pinball-style-schedule-v1",
        "source": {
            "paper": "arXiv:2512.09807v2",
            "artifact_commit": "8f16f24b621aacfaa4f456a2aeec8df088faf3a7",
        },
        "graph_fingerprint": graph.fingerprint,
        "stage_order": PINBALL_STAGE_ORDER,
        "stages": [
            [
                stage.patch_id,
                stage.check_basis,
                stage.sweep_time,
                stage.stage,
                [
                    [primitive.edge_id, primitive.source, primitive.target]
                    for primitive in stage.primitives
                ],
            ]
            for stage in stages
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compile_pinball_schedule(
    graph: CompiledPromatchGraph,
) -> CompiledPinballSchedule:
    """Compiles the deterministic nine-stage Pinball-style streaming schedule."""

    if not isinstance(graph, CompiledPromatchGraph):
        raise TypeError(
            f"graph must be a CompiledPromatchGraph, got {type(graph)!r}"
        )
    if graph.layout.mode != "fullhistory":
        raise ValueError("Pinball-style compilation requires a fullhistory layout")
    if graph.require_zero_frame:
        raise ValueError(
            "Pinball-style compilation requires require_zero_frame=False"
        )
    if graph.layout.distance % 2 == 0:
        raise ValueError(
            "Pinball-style compilation supports only odd code distance, matching "
            "the upstream artifact"
        )
    if graph.num_detectors != graph.layout.num_detectors:
        raise ValueError("graph/layout detector counts disagree")
    if [edge.edge_id for edge in graph.edges] != list(range(len(graph.edges))):
        raise ValueError("graph edge IDs are not dense canonical tuple indices")

    grouped: dict[
        tuple[int, PinballBasis, int, PinballStage], list[PinballPrimitive]
    ] = {}
    for edge in graph.edges:
        if edge.source < 0 or edge.source >= graph.num_detectors:
            raise ValueError(f"edge {edge.edge_id} has an out-of-range source")
        if edge.source_role != graph.layout.role_of(edge.source):
            raise ValueError(f"edge {edge.edge_id} source role disagrees with layout")
        if edge.target is not None:
            if edge.target < 0 or edge.target >= graph.num_detectors:
                raise ValueError(f"edge {edge.edge_id} has an out-of-range target")
            if edge.target_role != graph.layout.role_of(edge.target):
                raise ValueError(
                    f"edge {edge.edge_id} target role disagrees with layout"
                )

        source_role = edge.source_role
        target_role = edge.target_role
        if edge.target is None:
            if not isinstance(source_role, (L1BodyDetector, L1TerminalDetector)):
                continue
            stage: PinballStage = "E"
            # E(t-1) is the last stage of sweep t.  Terminal E(r) is the
            # artifact's final post-loop flush and receives a sentinel sweep.
            sweep_time = source_role.time + 1
            patch_id = source_role.patch_id
            basis = source_role.check_basis
        else:
            if not (_inner_role(source_role) and _inner_role(target_role)):
                continue
            assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
            assert isinstance(target_role, (L1BodyDetector, L1TerminalDetector))
            if (
                source_role.patch_id != target_role.patch_id
                or source_role.check_basis != target_role.check_basis
            ):
                # Multi-patch, cross-basis, and yoke connectivity belongs only
                # to the residual matcher in this graph-native adaptation.
                continue
            stage, sweep_time = _classify_inner_edge(graph, edge)
            patch_id = source_role.patch_id
            basis = source_role.check_basis
            if stage == "M" and any(edge.observable_mask):
                raise ValueError(
                    "Pinball-style M edge must have an all-zero observable mask: "
                    f"edge={edge.edge_id}, mask={edge.observable_mask.hex()}"
                )

        primitive = PinballPrimitive(
            edge_id=edge.edge_id,
            source=edge.source,
            target=edge.target,
            patch_id=patch_id,
            check_basis=basis,
            sweep_time=sweep_time,
            stage=stage,
        )
        grouped.setdefault((patch_id, basis, sweep_time, stage), []).append(primitive)

    stages: list[PinballStageSchedule] = []
    for patch_id in range(graph.layout.num_patches):
        for basis in ("X", "Z"):
            for sweep_time in range(graph.layout.rounds + 1):
                for stage in PINBALL_STAGE_ORDER:
                    primitives = tuple(
                        sorted(
                            grouped.pop((patch_id, basis, sweep_time, stage), ()),
                            key=_primitive_sort_key,
                        )
                    )
                    stages.append(
                        PinballStageSchedule(
                            patch_id=patch_id,
                            check_basis=basis,
                            sweep_time=sweep_time,
                            stage=stage,
                            primitives=primitives,
                        )
                    )
            terminal_flush = tuple(
                sorted(
                    grouped.pop(
                        (patch_id, basis, graph.layout.rounds + 1, "E"), ()
                    ),
                    key=_primitive_sort_key,
                )
            )
            stages.append(
                PinballStageSchedule(
                    patch_id=patch_id,
                    check_basis=basis,
                    sweep_time=graph.layout.rounds + 1,
                    stage="E",
                    primitives=terminal_flush,
                )
            )
    if grouped:
        raise AssertionError(
            f"unemitted Pinball-style schedule groups: {sorted(grouped)[:8]!r}"
        )

    result_stages = tuple(stages)
    _validate_compiled_schedule(graph, result_stages)
    return CompiledPinballSchedule(
        graph_fingerprint=graph.fingerprint,
        fingerprint=_schedule_fingerprint(graph, result_stages),
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        stages=result_stages,
    )


def _immutable_uint8(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.uint8).copy()
    result.flags.writeable = False
    return result


def _mask_to_frame(mask: bytes, *, num_observables: int) -> np.ndarray:
    return np.unpackbits(
        np.frombuffer(mask, dtype=np.uint8),
        bitorder="little",
        count=num_observables,
    ).astype(np.uint8, copy=True)


def _xor_edge_mask(frame_bytes: bytearray, edge: Edge) -> None:
    if len(edge.observable_mask) != len(frame_bytes):
        raise AssertionError("canonical edge observable mask has the wrong width")
    for index, value in enumerate(edge.observable_mask):
        frame_bytes[index] ^= value


def _assert_tentative_algebra(
    graph: CompiledPromatchGraph,
    *,
    original: np.ndarray,
    residual: np.ndarray,
    frame: np.ndarray,
    edge_support: tuple[int, ...],
) -> None:
    boundary = np.zeros(graph.num_detectors, dtype=np.uint8)
    expected_mask = bytearray((graph.num_observables + 7) // 8)
    for edge_id in edge_support:
        if edge_id < 0 or edge_id >= len(graph.edges):
            raise AssertionError("tentative support contains an invalid edge ID")
        edge = graph.edges[edge_id]
        boundary[edge.source] ^= 1
        if edge.target is not None:
            boundary[edge.target] ^= 1
        _xor_edge_mask(expected_mask, edge)
    if not np.array_equal(original ^ residual, boundary):
        raise AssertionError("Pinball-style GF(2) detector-boundary invariant failed")
    expected_frame = _mask_to_frame(
        bytes(expected_mask), num_observables=graph.num_observables
    )
    if not np.array_equal(frame, expected_frame):
        raise AssertionError("Pinball-style observable-frame invariant failed")


def predecode_pinball(
    graph: CompiledPromatchGraph,
    schedule: CompiledPinballSchedule,
    detection_events: np.ndarray,
) -> PinballResult:
    """Runs one immutable-input, whole-shot Pinball-style transaction."""

    if not isinstance(graph, CompiledPromatchGraph):
        raise TypeError(
            f"graph must be a CompiledPromatchGraph, got {type(graph)!r}"
        )
    if not isinstance(schedule, CompiledPinballSchedule):
        raise TypeError(
            f"schedule must be a CompiledPinballSchedule, got {type(schedule)!r}"
        )
    if schedule.graph_fingerprint != graph.fingerprint:
        raise ValueError("Pinball-style schedule was compiled for a different graph")
    if (
        schedule.num_detectors != graph.num_detectors
        or schedule.num_observables != graph.num_observables
    ):
        raise ValueError("Pinball-style schedule dimensions disagree with graph")

    raw = np.asarray(detection_events)
    if raw.ndim != 1:
        raise ValueError(
            f"detection_events must be one-dimensional, got shape {raw.shape}"
        )
    if len(raw) != graph.num_detectors:
        raise ValueError(
            f"detection_events has length {len(raw)}, expected {graph.num_detectors}"
        )
    if not np.issubdtype(raw.dtype, np.bool_) and not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError("detection_events must contain boolean or integer bits")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("detection_events entries must be binary")

    original = np.asarray(raw, dtype=np.uint8).copy()
    working = original.copy()
    frame_bytes = bytearray((graph.num_observables + 7) // 8)
    edge_support: list[int] = []
    stage_match_counts: list[int] = []
    for stage_schedule in schedule.stages:
        matches = 0
        for primitive in stage_schedule.primitives:
            if primitive.edge_id < 0 or primitive.edge_id >= len(graph.edges):
                raise ValueError(
                    f"schedule primitive has invalid edge ID {primitive.edge_id}"
                )
            edge = graph.edges[primitive.edge_id]
            if (edge.source, edge.target) != (primitive.source, primitive.target):
                raise ValueError(
                    f"schedule primitive {primitive.edge_id} disagrees with graph"
                )
            active = bool(working[primitive.source])
            if primitive.target is not None:
                active = active and bool(working[primitive.target])
            if not active:
                continue
            working[primitive.source] ^= 1
            if primitive.target is not None:
                working[primitive.target] ^= 1
            _xor_edge_mask(frame_bytes, edge)
            edge_support.append(primitive.edge_id)
            matches += 1
        stage_match_counts.append(matches)

    tentative_support = tuple(edge_support)
    tentative_frame = _mask_to_frame(
        bytes(frame_bytes), num_observables=graph.num_observables
    )
    _assert_tentative_algebra(
        graph,
        original=original,
        residual=working,
        frame=tentative_frame,
        edge_support=tentative_support,
    )

    complex_shot = bool(
        np.any(working[np.asarray(graph.layout.body_detector_ids, dtype=np.int64)])
        or np.any(
            working[np.asarray(graph.layout.terminal_detector_ids, dtype=np.int64)]
        )
    )
    if complex_shot:
        durable_residual = original
        durable_frame = np.zeros(graph.num_observables, dtype=np.uint8)
        durable_support: tuple[int, ...] = ()
    else:
        durable_residual = working
        durable_frame = tentative_frame
        durable_support = tentative_support

    # Whole-shot transaction invariant, independent of the tentative algebra.
    if complex_shot:
        if not np.array_equal(durable_residual, original) or np.any(durable_frame):
            raise AssertionError(
                "complex Pinball-style shot did not roll back globally"
            )
    else:
        _assert_tentative_algebra(
            graph,
            original=original,
            residual=durable_residual,
            frame=durable_frame,
            edge_support=durable_support,
        )

    return PinballResult(
        complex=complex_shot,
        residual_syndrome=_immutable_uint8(durable_residual),
        observable_frame=_immutable_uint8(durable_frame),
        edge_support=durable_support,
        stage_match_counts=tuple(stage_match_counts),
        tentative_residual_syndrome=_immutable_uint8(working),
        tentative_observable_frame=_immutable_uint8(tentative_frame),
        tentative_edge_support=tentative_support,
    )


__all__ = [
    "PINBALL_STAGE_ORDER",
    "CompiledPinballSchedule",
    "PinballPrimitive",
    "PinballResult",
    "PinballStageSchedule",
    "compile_pinball_schedule",
    "predecode_pinball",
]
