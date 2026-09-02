"""Strict YSC-CZ Pinball V2 schedule and domain-local transactions.

Unlike the older graph-native Pinball-style adapter, this module recognizes
the signed detector geometry emitted by the maintained YSC-CZ circuits.  It
also includes the inner-to-yoke E mechanisms: those mechanisms are activated
by their inner endpoint alone, but their correction boundary contains both
the inner and yoke detectors.

Execution is transactional per full-history ``(patch, basis)`` domain.  Work
from a simple domain is committed even when another domain remains complex;
all detector-boundary and observable-frame arithmetic is nevertheless applied
on the complete canonical matching edge over GF(2).

Physical supports use the pinned upstream row-major formulas.  X-check domains
use the upstream grid's horizontal reflection.  Transposed Z-check domains use
the complementary checkerboard, the symmetry permutation
``B1<->B2, B3<->B4, ST1<->ST2``, and the corresponding reflected data column.
Targets are then stored in actual patch-local YSC coordinates with Z Paulis for
X-check domains and X Paulis for Z-check domains.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np

from yoked.decoding._promatch_graph import CompiledPromatchGraph, Edge
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1FullHistoryDomain,
    L1TerminalDetector,
    YokeDetector,
)


PinballV2Basis: TypeAlias = Literal["X", "Z"]
PinballV2Pauli: TypeAlias = Literal["X", "Z"]
PinballV2Stage: TypeAlias = Literal[
    "M", "B1", "B2", "B3", "B4", "ST1", "ST2", "H", "E"
]

PINBALL_V2_STAGE_ORDER: tuple[PinballV2Stage, ...] = (
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

PINBALL_V2_SOURCE_COMMIT = "8f16f24b621aacfaa4f456a2aeec8df088faf3a7"
PINBALL_V2_GEOMETRY_PROFILE = "ysc-cz-signed-two-yoke-physical-support-v2"
PINBALL_V2_BASIS_PROFILE = "x-reflected-z-conjugated-stage-order-v2"

_Z_REFERENCE_STAGE: dict[PinballV2Stage, PinballV2Stage] = {
    "B1": "B2",
    "B2": "B1",
    "B3": "B4",
    "B4": "B3",
    "ST1": "ST2",
    "ST2": "ST1",
}
_Z_STAGE_ORDER: tuple[PinballV2Stage, ...] = (
    "M",
    "B2",
    "B1",
    "B4",
    "B3",
    "ST2",
    "ST1",
    "H",
    "E",
)
_InnerRole: TypeAlias = L1BodyDetector | L1TerminalDetector


@dataclasses.dataclass(frozen=True, order=True)
class PinballV2PauliTarget:
    """One physical data-qubit Pauli in patch-local YSC coordinates."""

    patch_id: int
    local_x: int
    y: int
    pauli: PinballV2Pauli


@dataclasses.dataclass(frozen=True)
class PinballV2Primitive:
    """One canonical correction with separate activation and full boundary."""

    edge_id: int
    domain: L1FullHistoryDomain
    sweep_time: int
    stage: PinballV2Stage
    activation_detectors: tuple[int, ...]
    detector_boundary: tuple[int, ...]
    physical_support: tuple[PinballV2PauliTarget, ...]


@dataclasses.dataclass(frozen=True)
class PinballV2StageSchedule:
    """Conflict-free primitives for one domain, sweep, and pipeline stage."""

    domain: L1FullHistoryDomain
    sweep_time: int
    stage: PinballV2Stage
    primitives: tuple[PinballV2Primitive, ...]


@dataclasses.dataclass(frozen=True)
class CompiledPinballV2Schedule:
    """Deterministic strict schedule bound to one compiled graph."""

    graph_fingerprint: str
    fingerprint: str
    num_detectors: int
    num_observables: int
    domains: tuple[L1FullHistoryDomain, ...]
    stages: tuple[PinballV2StageSchedule, ...]

    @property
    def primitives(self) -> tuple[PinballV2Primitive, ...]:
        return tuple(p for stage in self.stages for p in stage.primitives)


@dataclasses.dataclass(frozen=True)
class PinballV2HardwareProxies:
    """Decision-neutral raw work and depth counters for one domain.

    These values describe the maintained nine-stage schedule; they are not a
    latency, frequency, resource, or power model.  In particular,
    ``ideal_stream_cycle_lower_bound`` assumes one full layer block can enter a
    fully spatial nine-stage pipeline per cycle, plus the final E-only flush.
    A concrete RTL implementation can require additional buffering, routing,
    arbitration, correction reduction, and I/O cycles.

    ``stage_primitive_evaluation_counts`` is aligned one-for-one with the
    domain's ``stage_match_counts``.  Every scheduled primitive is evaluated,
    so the tuple is also the conflict-free parallel width of each stage slot.
    Tentative writes include work later rolled back for a complex domain;
    committed writes count only the correction made durable by this policy.
    Physical-target toggle counts are pre-XOR raw work, not the final reduced
    Pauli correction weight.
    """

    stage_family_pipeline_depth: int
    scheduled_stage_slot_count: int
    nonempty_stage_slot_count: int
    stage_primitive_evaluation_counts: tuple[int, ...]
    maximum_parallel_primitive_width: int
    primitive_evaluation_count: int
    activation_bit_read_count: int
    fired_primitive_count: int
    tentative_detector_xor_write_count: int
    tentative_physical_target_toggle_count: int
    committed_primitive_count: int
    committed_detector_xor_write_count: int
    committed_physical_target_toggle_count: int
    residual_or_reduction_input_count: int
    residual_or_reduction_tree_depth: int
    streamed_full_block_count: int
    streamed_terminal_flush_block_count: int
    pipeline_drain_cycle_lower_bound: int
    ideal_stream_cycle_lower_bound: int


@dataclasses.dataclass(frozen=True)
class PinballV2DomainResult:
    """Audit record for one independently committed or rolled-back domain."""

    domain: L1FullHistoryDomain
    complex: bool
    initial_hw: int
    tentative_residual_hw: int
    final_residual_hw: int
    edge_support: tuple[int, ...]
    tentative_edge_support: tuple[int, ...]
    physical_correction: tuple[PinballV2PauliTarget, ...]
    tentative_physical_correction: tuple[PinballV2PauliTarget, ...]
    stage_match_counts: tuple[int, ...]
    hardware_proxies: PinballV2HardwareProxies | None


@dataclasses.dataclass(frozen=True)
class PinballV2Result:
    """Durable mixed-domain result plus the all-attempts audit result."""

    complex: bool
    residual_syndrome: np.ndarray
    observable_frame: np.ndarray
    edge_support: tuple[int, ...]
    stage_match_counts: tuple[int, ...]
    tentative_residual_syndrome: np.ndarray
    tentative_observable_frame: np.ndarray
    tentative_edge_support: tuple[int, ...]
    physical_correction: tuple[PinballV2PauliTarget, ...]
    tentative_physical_correction: tuple[PinballV2PauliTarget, ...]
    domain_results: Mapping[L1FullHistoryDomain, PinballV2DomainResult]

    @property
    def hardware_proxies_by_domain(
        self,
    ) -> Mapping[L1FullHistoryDomain, PinballV2HardwareProxies]:
        """Returns the immutable per-domain hardware-proxy view."""

        return MappingProxyType(
            {
                domain: result.hardware_proxies
                for domain, result in self.domain_results.items()
                if result.hardware_proxies is not None
            }
        )


def _is_inner(role: object) -> bool:
    return isinstance(role, (L1BodyDetector, L1TerminalDetector))


def _domain_of(role: _InnerRole) -> L1FullHistoryDomain:
    return L1FullHistoryDomain(role.patch_id, role.check_basis)


def _local_coordinates(
    graph: CompiledPromatchGraph,
    detector_id: int,
    role: _InnerRole,
) -> tuple[float, float, int]:
    x, y, *_ = graph.layout.coordinates[detector_id]
    local_x = x - role.patch_id * graph.layout.pitch
    if role.check_basis == "X":
        return local_x, y, role.time
    return y, local_x, role.time


def _close(value: float, expected: float) -> bool:
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-7)


def _exact_int(value: float, *, name: str) -> int:
    result = round(value)
    if not _close(value, result):
        raise ValueError(f"YSC-CZ {name} is not integral: {value!r}")
    return result


def _upstream_site(
    graph: CompiledPromatchGraph,
    detector_id: int,
) -> tuple[int, int, int]:
    """Maps one normalized YSC detector site into the pinned upstream grid."""

    role = graph.layout.role_of(detector_id)
    if not _is_inner(role):
        raise ValueError(f"detector {detector_id} is not an inner YSC-CZ site")
    assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
    u, v, time = _local_coordinates(graph, detector_id, role)
    row = _exact_int(v + 0.5, name="upstream detector row")
    if role.check_basis == "X":
        # The X-domain data orientation is the horizontal reflection of the
        # upstream row-major syndrome grid.
        col_value = (
            (graph.layout.distance - 3) / 2
            - (u - 0.5 - (row % 2)) / 2
        )
    else:
        # Z checks occupy the complementary checkerboard.  Transposition into
        # normalized (u, v) coordinates therefore flips the row-parity offset
        # and induces the stage-family permutation used below.
        col_value = (u - 0.5 - ((row + 1) % 2)) / 2
    col = _exact_int(col_value, name="upstream detector column")
    cols = (graph.layout.distance - 1) // 2
    if not (0 <= row <= graph.layout.distance and 0 <= col < cols):
        raise ValueError(
            "YSC-CZ detector site maps outside the pinned upstream grid: "
            f"detector={detector_id}, row={row}, col={col}"
        )
    return row, col, time


def _physical_target(
    graph: CompiledPromatchGraph,
    domain: L1FullHistoryDomain,
    *,
    data_row: int,
    data_col: int,
) -> PinballV2PauliTarget:
    d = graph.layout.distance
    if not (0 <= data_row < d and 0 <= data_col < d):
        raise ValueError(
            "YSC-CZ Pinball physical target lies outside the data patch: "
            f"domain={domain!r}, row={data_row}, col={data_col}"
        )
    v = data_row
    if domain.check_basis == "X":
        u = d - 1 - data_col
        local_x, y, pauli = u, v, "Z"
    else:
        # The Z-check checkerboard symmetry reflects the upstream correction
        # column together with the detector grid.  In actual normalized YSC
        # coordinates this is u=data_col (then transpose into local_x/y).
        u = data_col
        local_x, y, pauli = v, u, "X"
    return PinballV2PauliTarget(
        patch_id=domain.patch_id,
        local_x=local_x,
        y=y,
        pauli=pauli,
    )


def _physical_support_for_slot(
    graph: CompiledPromatchGraph,
    *,
    domain: L1FullHistoryDomain,
    stage: PinballV2Stage,
    activation_detectors: tuple[int, ...],
) -> tuple[PinballV2PauliTarget, ...]:
    """Returns the exact pinned-upstream physical support for one YSC slot."""

    sites = [
        (*_upstream_site(graph, detector_id), detector_id)
        for detector_id in activation_detectors
    ]
    for _, _, _, detector_id in sites:
        role = graph.layout.role_of(detector_id)
        assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
        if _domain_of(role) != domain:
            raise ValueError(
                f"primitive activation detector {detector_id} belongs to another domain"
            )

    reference_stage: PinballV2Stage = stage
    if domain.check_basis == "Z":
        reference_stage = _Z_REFERENCE_STAGE.get(stage, stage)

    if reference_stage == "M":
        if len(sites) != 2:
            raise ValueError("M primitive must have two activation detectors")
        return ()

    if reference_stage in ("B1", "B2", "B3", "B4"):
        if len(sites) != 2:
            raise ValueError(f"{stage} primitive must have two activation detectors")
        centers = [site for site in sites if site[0] % 2 == 1]
        if len(centers) != 1:
            raise ValueError(f"{stage} primitive has no unique upstream center")
        row, col, time, center_id = centers[0]
        other = next(site for site in sites if site[3] != center_id)
        expected_neighbor = {
            "B1": (row - 1, col),
            "B2": (row + 1, col),
            "B3": (row + 1, col - 1),
            "B4": (row - 1, col - 1),
        }[reference_stage]
        if other[:3] != (*expected_neighbor, time):
            raise ValueError(
                f"{stage} primitive does not match the pinned upstream slot"
            )
        data_row, data_col = {
            "B1": (row - 1, 2 * col + 1),
            "B2": (row, 2 * col + 1),
            "B3": (row, 2 * col),
            "B4": (row - 1, 2 * col),
        }[reference_stage]
        return (
            _physical_target(
                graph,
                domain,
                data_row=data_row,
                data_col=data_col,
            ),
        )

    if reference_stage in ("ST1", "ST2", "H"):
        if len(sites) != 2:
            raise ValueError(f"{stage} primitive must have two activation detectors")
        previous, current = sorted(sites, key=lambda site: site[2])
        row, col, time, _ = current
        if previous[2] != time - 1:
            raise ValueError(f"{stage} primitive does not join consecutive layers")
        if reference_stage == "ST1":
            expected_previous = (row - 1, col + 1 - row % 2, time - 1)
            data_col = 2 * (col + 1) - row % 2
        elif reference_stage == "ST2":
            expected_previous = (row - 1, col - row % 2, time - 1)
            data_col = 2 * (col + 1) - row % 2 - 1
        else:
            expected_previous = (row - 2, col, time - 1)
            data_col = 2 * (col + 1) - row % 2 - 1
        if previous[:3] != expected_previous:
            raise ValueError(
                f"{stage} primitive does not match the pinned upstream slot"
            )
        if reference_stage == "H":
            return (
                _physical_target(
                    graph,
                    domain,
                    data_row=row - 1,
                    data_col=data_col,
                ),
                _physical_target(
                    graph,
                    domain,
                    data_row=row - 2,
                    data_col=data_col,
                ),
            )
        return (
            _physical_target(
                graph,
                domain,
                data_row=row - 1,
                data_col=data_col,
            ),
        )

    if reference_stage == "E":
        if len(sites) != 1:
            raise ValueError("E primitive must have one activation detector")
        row, col, _, _ = sites[0]
        last_col = (graph.layout.distance - 3) // 2
        if row % 2 == 1 and col == 0:
            data_row, data_col = row - 1, 0
        elif row % 2 == 0 and col == last_col:
            data_row, data_col = row, graph.layout.distance - 1
        else:
            raise ValueError("E primitive is not an upstream spatial-boundary slot")
        return (
            _physical_target(
                graph,
                domain,
                data_row=data_row,
                data_col=data_col,
            ),
        )

    raise AssertionError(f"unsupported Pinball V2 stage {stage!r}")


def _physical_support_mask(
    graph: CompiledPromatchGraph,
    domain: L1FullHistoryDomain,
    support: tuple[PinballV2PauliTarget, ...],
) -> bytes:
    """Computes the symplectic logical parity of a physical Pauli support."""

    mask = bytearray((graph.num_observables + 7) // 8)
    d = graph.layout.distance
    for target in support:
        if target.patch_id != domain.patch_id:
            raise ValueError("physical target belongs to another patch")
        if not (0 <= target.local_x < d and 0 <= target.y < d):
            raise ValueError(f"physical target is outside the data patch: {target!r}")
        if domain.check_basis == "X":
            if target.pauli != "Z":
                raise ValueError("X-check domain correction must contain only Z Paulis")
            crosses_logical = target.local_x == 0
            observable_id = 2 * domain.patch_id
        else:
            if target.pauli != "X":
                raise ValueError("Z-check domain correction must contain only X Paulis")
            crosses_logical = target.y == 0
            observable_id = 2 * domain.patch_id + 1
        if crosses_logical:
            mask[observable_id // 8] ^= 1 << (observable_id % 8)
    return bytes(mask)


def _xor_physical_correction(
    schedule: CompiledPinballV2Schedule,
    edge_support: tuple[int, ...],
) -> tuple[PinballV2PauliTarget, ...]:
    """XOR-reduces primitive supports into one canonical Pauli correction."""

    support_by_edge = {
        primitive.edge_id: primitive.physical_support
        for primitive in schedule.primitives
    }
    correction: set[PinballV2PauliTarget] = set()
    for edge_id in edge_support:
        try:
            physical_support = support_by_edge[edge_id]
        except KeyError as ex:
            raise AssertionError(
                f"edge {edge_id} has no Pinball V2 physical support"
            ) from ex
        for target in physical_support:
            if target in correction:
                correction.remove(target)
            else:
                correction.add(target)
    return tuple(sorted(correction))


def _physical_correction_frame(
    graph: CompiledPromatchGraph,
    correction: tuple[PinballV2PauliTarget, ...],
) -> np.ndarray:
    mask = bytearray((graph.num_observables + 7) // 8)
    for target in correction:
        if target.pauli == "Z":
            crosses_logical = target.local_x == 0
            observable_id = 2 * target.patch_id
        elif target.pauli == "X":
            crosses_logical = target.y == 0
            observable_id = 2 * target.patch_id + 1
        else:  # pragma: no cover - the frozen target type and compiler fail closed.
            raise AssertionError(f"unsupported physical Pauli {target.pauli!r}")
        if crosses_logical:
            mask[observable_id // 8] ^= 1 << (observable_id % 8)
    return _mask_to_frame(bytes(mask), num_observables=graph.num_observables)


def _zero_mask(graph: CompiledPromatchGraph) -> bytes:
    return bytes((graph.num_observables + 7) // 8)


def _owned_e_mask(
    graph: CompiledPromatchGraph, domain: L1FullHistoryDomain
) -> bytes:
    observable_id = 2 * domain.patch_id + (0 if domain.check_basis == "X" else 1)
    result = bytearray((graph.num_observables + 7) // 8)
    result[observable_id // 8] = 1 << (observable_id % 8)
    return bytes(result)


def _classify_inner_edge(
    graph: CompiledPromatchGraph, edge: Edge
) -> tuple[PinballV2Stage, int]:
    """Classifies exactly one maintained signed YSC-CZ inner profile."""

    if edge.target is None or not _is_inner(edge.source_role) or not _is_inner(
        edge.target_role
    ):
        raise AssertionError("non-inner edge passed to inner classifier")
    source_role = edge.source_role
    target_role = edge.target_role
    assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
    assert isinstance(target_role, (L1BodyDetector, L1TerminalDetector))
    a_u, a_v, a_t = _local_coordinates(graph, edge.source, source_role)
    b_u, b_v, b_t = _local_coordinates(graph, edge.target, target_role)
    du, dv, dt = b_u - a_u, b_v - a_v, b_t - a_t

    if abs(dt) == 1:
        later_du = du if dt > 0 else -du
        later_dv = dv if dt > 0 else -dv
        later_time = max(a_t, b_t)
    else:
        later_du = later_dv = 0
        later_time = max(a_t, b_t)

    if abs(dt) == 1 and _close(later_du, 0) and _close(later_dv, 0):
        return "M", later_time

    if dt == 0 and _close(abs(du), 1) and _close(abs(dv), 1):
        source_center = round(a_u - 0.5) % 2 == 0
        target_center = round(b_u - 0.5) % 2 == 0
        if source_center == target_center:
            raise ValueError(f"YSC-CZ B edge {edge.edge_id} has ambiguous center")
        delta = (round(du), round(dv)) if source_center else (
            round(-du),
            round(-dv),
        )
        stage_by_delta: dict[tuple[int, int], PinballV2Stage] = {
            (+1, +1): "B1",
            (+1, -1): "B2",
            (-1, -1): "B3",
            (-1, +1): "B4",
        }
        try:
            return stage_by_delta[delta], a_t
        except KeyError as ex:
            raise ValueError(
                f"YSC-CZ B edge {edge.edge_id} has unsupported orientation {delta}"
            ) from ex

    # These are deliberately signed.  Reflected or time-reversed copies are
    # not mechanisms in the maintained CZ circuit and must fail closed.
    if abs(dt) == 1 and _close(later_dv, 1) and _close(later_du, +1):
        return "ST1", later_time
    if abs(dt) == 1 and _close(later_dv, 1) and _close(later_du, -1):
        return "ST2", later_time
    if abs(dt) == 1 and _close(later_du, 0) and _close(later_dv, +2):
        return "H", later_time

    raise ValueError(
        "unsupported signed YSC-CZ within-domain topology: "
        f"edge={edge.edge_id}, dt={dt}, du={du}, dv={dv}, "
        f"source_role={source_role!r}, target_role={target_role!r}"
    )


def _primitive_sort_key(
    primitive: PinballV2Primitive,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    return (
        primitive.activation_detectors,
        primitive.detector_boundary,
        primitive.edge_id,
    )


def _stage_keys(
    graph: CompiledPromatchGraph, domains: tuple[L1FullHistoryDomain, ...]
) -> list[tuple[L1FullHistoryDomain, int, PinballV2Stage]]:
    result: list[tuple[L1FullHistoryDomain, int, PinballV2Stage]] = []
    for domain in domains:
        stage_order = (
            PINBALL_V2_STAGE_ORDER if domain.check_basis == "X" else _Z_STAGE_ORDER
        )
        for sweep_time in range(graph.layout.rounds + 1):
            result.extend((domain, sweep_time, stage) for stage in stage_order)
        result.append((domain, graph.layout.rounds + 1, "E"))
    return result


def _validate_schedule(
    graph: CompiledPromatchGraph,
    domains: tuple[L1FullHistoryDomain, ...],
    stages: tuple[PinballV2StageSchedule, ...],
) -> None:
    if [(s.domain, s.sweep_time, s.stage) for s in stages] != _stage_keys(
        graph, domains
    ):
        raise ValueError("YSC-CZ Pinball V2 stages are incomplete or misordered")

    seen_edges: set[int] = set()
    measurement_pairs: dict[tuple[L1FullHistoryDomain, float, float, int], int] = {}
    sites: dict[tuple[L1FullHistoryDomain, int], set[tuple[float, float]]] = {}
    site_ids: dict[
        tuple[L1FullHistoryDomain, int, float, float], int
    ] = {}
    e_kind_counts: dict[tuple[L1FullHistoryDomain, int, str], int] = {}
    e_sources: dict[int, int] = {}
    actual_slots: dict[
        tuple[L1FullHistoryDomain, int, PinballV2Stage],
        set[tuple[int, ...]],
    ] = {}

    for detector_id, role in enumerate(graph.layout.roles):
        if not _is_inner(role):
            continue
        assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
        u, v, time = _local_coordinates(graph, detector_id, role)
        domain = _domain_of(role)
        sites.setdefault((domain, time), set()).add((u, v))
        site_key = (domain, time, u, v)
        if site_key in site_ids:
            raise ValueError(f"duplicate YSC-CZ detector site {site_key!r}")
        site_ids[site_key] = detector_id

    for stage_schedule in stages:
        if tuple(sorted(stage_schedule.primitives, key=_primitive_sort_key)) != (
            stage_schedule.primitives
        ):
            raise ValueError("YSC-CZ Pinball V2 primitives are not sorted")
        used_activation: set[int] = set()
        for primitive in stage_schedule.primitives:
            if (
                primitive.domain != stage_schedule.domain
                or primitive.sweep_time != stage_schedule.sweep_time
                or primitive.stage != stage_schedule.stage
            ):
                raise ValueError("primitive metadata disagrees with its stage")
            if primitive.edge_id in seen_edges:
                raise ValueError(f"edge {primitive.edge_id} was assigned twice")
            seen_edges.add(primitive.edge_id)
            actual_slots.setdefault(
                (
                    primitive.domain,
                    primitive.sweep_time,
                    primitive.stage,
                ),
                set(),
            ).add(tuple(sorted(primitive.activation_detectors)))
            if primitive.edge_id < 0 or primitive.edge_id >= len(graph.edges):
                raise ValueError(f"invalid primitive edge ID {primitive.edge_id}")
            edge = graph.edges[primitive.edge_id]
            expected_boundary = (edge.source,) if edge.target is None else (
                edge.source,
                edge.target,
            )
            if primitive.detector_boundary != expected_boundary:
                raise ValueError(
                    f"primitive {primitive.edge_id} lost its full detector boundary"
                )
            expected_physical_support = _physical_support_for_slot(
                graph,
                domain=primitive.domain,
                stage=primitive.stage,
                activation_detectors=primitive.activation_detectors,
            )
            if primitive.physical_support != expected_physical_support:
                raise ValueError(
                    f"primitive {primitive.edge_id} has the wrong physical support"
                )
            expected_mask = _physical_support_mask(
                graph,
                primitive.domain,
                primitive.physical_support,
            )
            if edge.observable_mask != expected_mask:
                raise ValueError(
                    "primitive physical support disagrees with its canonical "
                    f"observable frame: edge={primitive.edge_id}, "
                    f"actual={edge.observable_mask.hex()}, "
                    f"expected={expected_mask.hex()}"
                )
            if not primitive.activation_detectors or any(
                detector in used_activation
                for detector in primitive.activation_detectors
            ):
                raise ValueError(
                    "YSC-CZ Pinball V2 stage has ambiguous activation overlap: "
                    f"domain={stage_schedule.domain!r}, "
                    f"t={stage_schedule.sweep_time}, stage={stage_schedule.stage}"
                )
            used_activation.update(primitive.activation_detectors)

            if primitive.stage == "E":
                if len(primitive.activation_detectors) != 1:
                    raise ValueError("E primitive must have one activation detector")
                inner_id = primitive.activation_detectors[0]
                e_sources[inner_id] = e_sources.get(inner_id, 0) + 1
                role = graph.layout.role_of(inner_id)
                assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
                u, _, time = _local_coordinates(graph, inner_id, role)
                if edge.target is None:
                    if not _close(u, graph.layout.distance - 1.5) or (
                        edge.observable_mask != _zero_mask(graph)
                    ):
                        raise ValueError(
                            f"edge {edge.edge_id} is not an exact true-boundary E"
                        )
                    kind = "true"
                else:
                    if (
                        len(primitive.detector_boundary) != 2
                        or not any(
                            isinstance(graph.layout.role_of(k), YokeDetector)
                            for k in primitive.detector_boundary
                        )
                        or not _close(u, 0.5)
                        or edge.observable_mask
                        != _owned_e_mask(graph, primitive.domain)
                    ):
                        raise ValueError(
                            f"edge {edge.edge_id} is not an exact inner-yoke E"
                        )
                    kind = "yoke"
                e_kind_counts[(primitive.domain, time, kind)] = (
                    e_kind_counts.get((primitive.domain, time, kind), 0) + 1
                )
            elif primitive.stage == "M":
                edge_roles = (edge.source_role, edge.target_role)
                assert all(_is_inner(role) for role in edge_roles)
                source_role = edge.source_role
                target_role = edge.target_role
                assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
                assert isinstance(target_role, (L1BodyDetector, L1TerminalDetector))
                earlier_index = 0 if source_role.time < target_role.time else 1
                earlier_role = edge_roles[earlier_index]
                assert isinstance(earlier_role, (L1BodyDetector, L1TerminalDetector))
                earlier_id = expected_boundary[earlier_index]
                u, v, time = _local_coordinates(graph, earlier_id, earlier_role)
                key = (primitive.domain, u, v, time)
                measurement_pairs[key] = measurement_pairs.get(key, 0) + 1

    duplicated_e = sorted(k for k, count in e_sources.items() if count != 1)
    if duplicated_e:
        raise ValueError(
            f"inner detector has ambiguous E mechanisms: {duplicated_e[:8]}"
        )

    expected_e_count = (graph.layout.distance + 1) // 2
    bulk_delta: dict[PinballV2Stage, tuple[int, int]] = {
        "B1": (+1, +1),
        "B2": (+1, -1),
        "B3": (-1, -1),
        "B4": (-1, +1),
    }
    temporal_delta: dict[PinballV2Stage, tuple[int, int]] = {
        "M": (0, 0),
        "ST1": (+1, +1),
        "ST2": (-1, +1),
        "H": (0, +2),
    }
    for domain in domains:
        reference = sites.get((domain, 0))
        if reference is None:
            raise ValueError(f"domain {domain!r} has no time-zero detector sites")
        for time in range(graph.layout.rounds + 1):
            if sites.get((domain, time)) != reference:
                raise ValueError(
                    f"domain {domain!r} has inconsistent detector sites at t={time}"
                )
            for kind in ("yoke", "true"):
                actual = e_kind_counts.get((domain, time, kind), 0)
                if actual != expected_e_count:
                    raise ValueError(
                        "YSC-CZ E profile has the wrong per-layer count: "
                        f"domain={domain!r}, time={time}, kind={kind}, "
                        f"actual={actual}, expected={expected_e_count}"
                    )
            expected_e_sources = {
                site_ids[(domain, time, u, v)]
                for u, v in reference
                if _close(u, 0.5)
                or _close(u, graph.layout.distance - 1.5)
            }
            actual_e_sources = {
                slot[0]
                for slot in actual_slots.get((domain, time + 1, "E"), set())
            }
            if actual_e_sources != expected_e_sources:
                raise ValueError(
                    "YSC-CZ E profile does not match the exact boundary-source "
                    f"set: domain={domain!r}, time={time}, "
                    f"missing={sorted(expected_e_sources - actual_e_sources)[:8]}, "
                    f"extra={sorted(actual_e_sources - expected_e_sources)[:8]}"
                )

            for stage, (du, dv) in bulk_delta.items():
                expected: set[tuple[int, ...]] = set()
                for u, v in reference:
                    if round(u - 0.5) % 2 != 0:
                        continue
                    neighbor = (u + du, v + dv)
                    if neighbor not in reference:
                        continue
                    expected.add(
                        tuple(
                            sorted(
                                (
                                    site_ids[(domain, time, u, v)],
                                    site_ids[
                                        (domain, time, neighbor[0], neighbor[1])
                                    ],
                                )
                            )
                        )
                    )
                actual = actual_slots.get((domain, time, stage), set())
                if actual != expected:
                    raise ValueError(
                        "YSC-CZ bulk profile slot mismatch: "
                        f"domain={domain!r}, time={time}, stage={stage}, "
                        f"missing={sorted(expected - actual)[:8]}, "
                        f"extra={sorted(actual - expected)[:8]}"
                    )

        for time in range(graph.layout.rounds):
            for u, v in reference:
                key = (domain, u, v, time)
                if measurement_pairs.get(key, 0) != 1:
                    raise ValueError(
                        "YSC-CZ M profile does not cover each consecutive-layer site: "
                        f"{key!r}"
                    )
            later_time = time + 1
            for stage, (du, dv) in temporal_delta.items():
                expected = set()
                for u, v in reference:
                    later = (u + du, v + dv)
                    if later not in reference:
                        continue
                    expected.add(
                        tuple(
                            sorted(
                                (
                                    site_ids[(domain, time, u, v)],
                                    site_ids[
                                        (
                                            domain,
                                            later_time,
                                            later[0],
                                            later[1],
                                        )
                                    ],
                                )
                            )
                        )
                    )
                actual = actual_slots.get((domain, later_time, stage), set())
                if actual != expected:
                    raise ValueError(
                        "YSC-CZ temporal profile slot mismatch: "
                        f"domain={domain!r}, time={later_time}, stage={stage}, "
                        f"missing={sorted(expected - actual)[:8]}, "
                        f"extra={sorted(actual - expected)[:8]}"
                    )


def _schedule_fingerprint(
    graph: CompiledPromatchGraph,
    domains: tuple[L1FullHistoryDomain, ...],
    stages: tuple[PinballV2StageSchedule, ...],
) -> str:
    payload = {
        "schema": "ysc-cz-pinball-v2-schedule-v3",
        "source": {
            "artifact_commit": PINBALL_V2_SOURCE_COMMIT,
            "geometry_profile": PINBALL_V2_GEOMETRY_PROFILE,
            "basis_profile": PINBALL_V2_BASIS_PROFILE,
        },
        "graph_fingerprint": graph.fingerprint,
        "domains": [[d.patch_id, d.check_basis] for d in domains],
        "stage_order": {
            "X": PINBALL_V2_STAGE_ORDER,
            "Z": _Z_STAGE_ORDER,
        },
        "stages": [
            [
                s.domain.patch_id,
                s.domain.check_basis,
                s.sweep_time,
                s.stage,
                [
                    [
                        p.edge_id,
                        list(p.activation_detectors),
                        list(p.detector_boundary),
                        [
                            [target.patch_id, target.local_x, target.y, target.pauli]
                            for target in p.physical_support
                        ],
                    ]
                    for p in s.primitives
                ],
            ]
            for s in stages
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compile_pinball_v2_schedule(
    graph: CompiledPromatchGraph,
) -> CompiledPinballV2Schedule:
    """Compiles a strict signed YSC-CZ schedule, failing on ambiguity."""

    if not isinstance(graph, CompiledPromatchGraph):
        raise TypeError(f"graph must be CompiledPromatchGraph, got {type(graph)!r}")
    if graph.layout.mode != "fullhistory":
        raise ValueError("YSC-CZ Pinball V2 requires a fullhistory layout")
    if graph.require_zero_frame:
        raise ValueError("YSC-CZ Pinball V2 requires require_zero_frame=False")
    if graph.layout.distance % 2 == 0:
        raise ValueError("YSC-CZ Pinball V2 supports only odd code distance")
    if graph.num_detectors != graph.layout.num_detectors:
        raise ValueError("graph/layout detector counts disagree")
    if graph.num_observables != 2 * graph.layout.num_patches:
        raise ValueError("YSC-CZ observable count is not two per patch")
    if len(graph.layout.yoke_detector_ids) != 2:
        raise ValueError(
            "YSC-CZ Pinball V2 currently requires exactly two yoke detectors"
        )
    if [edge.edge_id for edge in graph.edges] != list(range(len(graph.edges))):
        raise ValueError("graph edge IDs are not dense canonical tuple indices")

    domains = tuple(
        L1FullHistoryDomain(patch_id, basis)
        for patch_id in range(graph.layout.num_patches)
        for basis in ("X", "Z")
    )
    grouped: dict[
        tuple[L1FullHistoryDomain, int, PinballV2Stage],
        list[PinballV2Primitive],
    ] = {}
    mask_width = (graph.num_observables + 7) // 8

    for edge in graph.edges:
        if edge.source < 0 or edge.source >= graph.num_detectors:
            raise ValueError(f"edge {edge.edge_id} has out-of-range source")
        if edge.source_role != graph.layout.role_of(edge.source):
            raise ValueError(f"edge {edge.edge_id} source role disagrees with layout")
        if len(edge.observable_mask) != mask_width:
            raise ValueError(f"edge {edge.edge_id} has the wrong mask width")
        if edge.target is not None:
            if edge.target <= edge.source or edge.target >= graph.num_detectors:
                raise ValueError(f"edge {edge.edge_id} is not canonically oriented")
            if edge.target_role != graph.layout.role_of(edge.target):
                raise ValueError(
                    f"edge {edge.edge_id} target role disagrees with layout"
                )

        source_role, target_role = edge.source_role, edge.target_role
        domain: L1FullHistoryDomain
        stage: PinballV2Stage
        sweep_time: int
        activation: tuple[int, ...]

        if edge.target is None:
            if not _is_inner(source_role):
                continue
            assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
            domain = _domain_of(source_role)
            stage = "E"
            sweep_time = source_role.time + 1
            activation = (edge.source,)
        elif _is_inner(source_role) and isinstance(target_role, YokeDetector):
            assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
            domain = _domain_of(source_role)
            stage = "E"
            sweep_time = source_role.time + 1
            activation = (edge.source,)
        elif _is_inner(source_role) and _is_inner(target_role):
            assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
            assert isinstance(target_role, (L1BodyDetector, L1TerminalDetector))
            if _domain_of(source_role) != _domain_of(target_role):
                continue
            domain = _domain_of(source_role)
            stage, sweep_time = _classify_inner_edge(graph, edge)
            activation = (edge.source, edge.target)
            if edge.observable_mask != _zero_mask(graph):
                raise ValueError(
                    f"within-domain edge {edge.edge_id} must have zero frame mask"
                )
        elif isinstance(source_role, YokeDetector) and _is_inner(target_role):
            # Canonical graphs place the lower-numbered inner detector first.
            # Treat a reversed representation as ambiguity, not as equivalent.
            raise ValueError(f"edge {edge.edge_id} reverses an inner-yoke E")
        else:
            continue

        boundary = (edge.source,) if edge.target is None else (
            edge.source,
            edge.target,
        )
        primitive = PinballV2Primitive(
            edge_id=edge.edge_id,
            domain=domain,
            sweep_time=sweep_time,
            stage=stage,
            activation_detectors=activation,
            detector_boundary=boundary,
            physical_support=_physical_support_for_slot(
                graph,
                domain=domain,
                stage=stage,
                activation_detectors=activation,
            ),
        )
        grouped.setdefault((domain, sweep_time, stage), []).append(primitive)

    stages = tuple(
        PinballV2StageSchedule(
            domain=domain,
            sweep_time=sweep_time,
            stage=stage,
            primitives=tuple(
                sorted(
                    grouped.pop((domain, sweep_time, stage), ()),
                    key=_primitive_sort_key,
                )
            ),
        )
        for domain, sweep_time, stage in _stage_keys(graph, domains)
    )
    if grouped:
        raise ValueError(f"unemitted YSC-CZ schedule groups: {sorted(grouped)!r}")
    _validate_schedule(graph, domains, stages)
    return CompiledPinballV2Schedule(
        graph_fingerprint=graph.fingerprint,
        fingerprint=_schedule_fingerprint(graph, domains, stages),
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        domains=domains,
        stages=stages,
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


def _apply_support(
    graph: CompiledPromatchGraph,
    original: np.ndarray,
    support: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    residual = original.copy()
    mask = bytearray((graph.num_observables + 7) // 8)
    for edge_id in support:
        if edge_id < 0 or edge_id >= len(graph.edges):
            raise AssertionError("support contains an invalid edge ID")
        edge = graph.edges[edge_id]
        residual[edge.source] ^= 1
        if edge.target is not None:
            residual[edge.target] ^= 1
        if len(edge.observable_mask) != len(mask):
            raise AssertionError("canonical edge mask has the wrong width")
        for index, value in enumerate(edge.observable_mask):
            mask[index] ^= value
    return residual, _mask_to_frame(bytes(mask), num_observables=graph.num_observables)


def _domain_detector_ids(
    graph: CompiledPromatchGraph,
) -> dict[L1FullHistoryDomain, tuple[int, ...]]:
    result: dict[L1FullHistoryDomain, list[int]] = {}
    for detector_id, role in enumerate(graph.layout.roles):
        if _is_inner(role):
            assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
            result.setdefault(_domain_of(role), []).append(detector_id)
    return {domain: tuple(ids) for domain, ids in result.items()}


def _binary_reduction_tree_depth(input_count: int) -> int:
    """Returns the balanced binary-tree depth needed to reduce ``input_count``."""

    if input_count < 0:
        raise ValueError("reduction input count must be nonnegative")
    return 0 if input_count <= 1 else (input_count - 1).bit_length()


def _domain_hardware_proxies(
    graph: CompiledPromatchGraph,
    schedule: CompiledPinballV2Schedule,
    *,
    stage_indices: tuple[int, ...],
    tentative_support: tuple[int, ...],
    residual_or_input_count: int,
    committed: bool,
) -> PinballV2HardwareProxies:
    """Derives hardware work counters after, and independently of, decisions."""

    stages = tuple(schedule.stages[index] for index in stage_indices)
    stage_widths = tuple(len(stage.primitives) for stage in stages)
    primitives = tuple(
        primitive for stage in stages for primitive in stage.primitives
    )
    primitive_by_edge = {primitive.edge_id: primitive for primitive in primitives}
    if len(primitive_by_edge) != len(primitives):
        raise AssertionError("domain hardware proxy saw a duplicate primitive edge")
    try:
        fired = tuple(primitive_by_edge[edge_id] for edge_id in tentative_support)
    except KeyError as ex:
        raise AssertionError(
            "domain hardware proxy support contains an out-of-domain edge"
        ) from ex

    detector_writes = sum(len(primitive.detector_boundary) for primitive in fired)
    physical_toggles = sum(len(primitive.physical_support) for primitive in fired)
    full_block_count = len(
        {
            stage.sweep_time
            for stage in stages
            if stage.sweep_time <= graph.layout.rounds
        }
    )
    terminal_flush_count = sum(
        stage.sweep_time == graph.layout.rounds + 1 and stage.stage == "E"
        for stage in stages
    )
    pipeline_depth = len(PINBALL_V2_STAGE_ORDER)
    pipeline_drain = max(0, pipeline_depth - 1)
    return PinballV2HardwareProxies(
        stage_family_pipeline_depth=pipeline_depth,
        scheduled_stage_slot_count=len(stages),
        nonempty_stage_slot_count=sum(bool(width) for width in stage_widths),
        stage_primitive_evaluation_counts=stage_widths,
        maximum_parallel_primitive_width=max(stage_widths, default=0),
        primitive_evaluation_count=len(primitives),
        activation_bit_read_count=sum(
            len(primitive.activation_detectors) for primitive in primitives
        ),
        fired_primitive_count=len(fired),
        tentative_detector_xor_write_count=detector_writes,
        tentative_physical_target_toggle_count=physical_toggles,
        committed_primitive_count=len(fired) if committed else 0,
        committed_detector_xor_write_count=detector_writes if committed else 0,
        committed_physical_target_toggle_count=(
            physical_toggles if committed else 0
        ),
        residual_or_reduction_input_count=residual_or_input_count,
        residual_or_reduction_tree_depth=_binary_reduction_tree_depth(
            residual_or_input_count
        ),
        streamed_full_block_count=full_block_count,
        streamed_terminal_flush_block_count=terminal_flush_count,
        pipeline_drain_cycle_lower_bound=pipeline_drain,
        ideal_stream_cycle_lower_bound=(
            full_block_count + terminal_flush_count + pipeline_drain
        ),
    )


def predecode_pinball_v2(
    graph: CompiledPromatchGraph,
    schedule: CompiledPinballV2Schedule,
    detection_events: np.ndarray,
    *,
    collect_hardware_proxies: bool = False,
    _schedule_is_validated: bool = False,
) -> PinballV2Result:
    """Runs independent full-history domain transactions on one syndrome.

    ``_schedule_is_validated`` is the compiled-adapter fast path.  External
    callers should leave it false so a manually constructed or replaced
    schedule is checked before use.

    Hardware proxies are deliberately opt-in.  The default production path
    stores ``None`` on every domain result and never invokes the proxy builder,
    keeping hardware characterization outside software-latency measurements.
    """

    if not isinstance(graph, CompiledPromatchGraph):
        raise TypeError(f"graph must be CompiledPromatchGraph, got {type(graph)!r}")
    if not isinstance(schedule, CompiledPinballV2Schedule):
        raise TypeError(
            f"schedule must be CompiledPinballV2Schedule, got {type(schedule)!r}"
        )
    if not isinstance(collect_hardware_proxies, bool):
        raise TypeError("collect_hardware_proxies must be bool")
    if schedule.graph_fingerprint != graph.fingerprint:
        raise ValueError("YSC-CZ Pinball V2 schedule belongs to another graph")
    if (schedule.num_detectors, schedule.num_observables) != (
        graph.num_detectors,
        graph.num_observables,
    ):
        raise ValueError("YSC-CZ Pinball V2 schedule dimensions disagree with graph")
    if not _schedule_is_validated:
        _validate_schedule(graph, schedule.domains, schedule.stages)
        if schedule.fingerprint != _schedule_fingerprint(
            graph, schedule.domains, schedule.stages
        ):
            raise ValueError("YSC-CZ Pinball V2 schedule fingerprint is invalid")

    raw = np.asarray(detection_events)
    if raw.ndim != 1:
        raise ValueError(f"detection_events must be one-dimensional, got {raw.shape}")
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
    ids_by_domain = _domain_detector_ids(graph)
    stage_counts = [0] * len(schedule.stages)
    tentative_by_domain: dict[L1FullHistoryDomain, tuple[int, ...]] = {}
    domain_counts: dict[L1FullHistoryDomain, tuple[int, ...]] = {}
    domain_complex: dict[L1FullHistoryDomain, bool] = {}
    domain_tentative_hw: dict[L1FullHistoryDomain, int] = {}

    stage_indices_by_domain: dict[L1FullHistoryDomain, list[int]] = {
        domain: [] for domain in schedule.domains
    }
    for index, stage in enumerate(schedule.stages):
        stage_indices_by_domain[stage.domain].append(index)

    for domain in schedule.domains:
        working = original.copy()
        support: list[int] = []
        local_counts: list[int] = []
        for stage_index in stage_indices_by_domain[domain]:
            stage = schedule.stages[stage_index]
            matches = 0
            for primitive in stage.primitives:
                if all(working[k] for k in primitive.activation_detectors):
                    edge = graph.edges[primitive.edge_id]
                    for detector_id in primitive.detector_boundary:
                        working[detector_id] ^= 1
                    support.append(edge.edge_id)
                    matches += 1
            stage_counts[stage_index] = matches
            local_counts.append(matches)

        domain_ids = np.asarray(ids_by_domain[domain], dtype=np.int64)
        replayed_working, _ = _apply_support(graph, original, tuple(support))
        if not np.array_equal(working, replayed_working):
            raise AssertionError(
                "YSC-CZ Pinball V2 domain mutation disagrees with edge support"
            )
        tentative_hw = int(np.count_nonzero(working[domain_ids]))
        tentative_by_domain[domain] = tuple(support)
        domain_counts[domain] = tuple(local_counts)
        domain_tentative_hw[domain] = tentative_hw
        domain_complex[domain] = tentative_hw != 0

    tentative_support = tuple(
        edge_id
        for domain in schedule.domains
        for edge_id in tentative_by_domain[domain]
    )
    durable_support = tuple(
        edge_id
        for domain in schedule.domains
        if not domain_complex[domain]
        for edge_id in tentative_by_domain[domain]
    )
    tentative_physical_correction = _xor_physical_correction(
        schedule, tentative_support
    )
    durable_physical_correction = _xor_physical_correction(schedule, durable_support)
    tentative_residual, tentative_frame = _apply_support(
        graph, original, tentative_support
    )
    durable_residual, durable_frame = _apply_support(
        graph, original, durable_support
    )
    if not np.array_equal(
        tentative_frame,
        _physical_correction_frame(graph, tentative_physical_correction),
    ):
        raise AssertionError(
            "tentative physical correction disagrees with its observable frame"
        )
    if not np.array_equal(
        durable_frame,
        _physical_correction_frame(graph, durable_physical_correction),
    ):
        raise AssertionError(
            "durable physical correction disagrees with its observable frame"
        )

    domain_results: dict[L1FullHistoryDomain, PinballV2DomainResult] = {}
    for domain in schedule.domains:
        domain_ids = np.asarray(ids_by_domain[domain], dtype=np.int64)
        is_complex = domain_complex[domain]
        tentative = tentative_by_domain[domain]
        domain_results[domain] = PinballV2DomainResult(
            domain=domain,
            complex=is_complex,
            initial_hw=int(np.count_nonzero(original[domain_ids])),
            tentative_residual_hw=domain_tentative_hw[domain],
            final_residual_hw=int(np.count_nonzero(durable_residual[domain_ids])),
            edge_support=() if is_complex else tentative,
            tentative_edge_support=tentative,
            physical_correction=(
                () if is_complex else _xor_physical_correction(schedule, tentative)
            ),
            tentative_physical_correction=_xor_physical_correction(
                schedule, tentative
            ),
            stage_match_counts=domain_counts[domain],
            hardware_proxies=(
                _domain_hardware_proxies(
                    graph,
                    schedule,
                    stage_indices=tuple(stage_indices_by_domain[domain]),
                    tentative_support=tentative,
                    residual_or_input_count=len(domain_ids),
                    committed=not is_complex,
                )
                if collect_hardware_proxies
                else None
            ),
        )

    return PinballV2Result(
        complex=any(domain_complex.values()),
        residual_syndrome=_immutable_uint8(durable_residual),
        observable_frame=_immutable_uint8(durable_frame),
        edge_support=durable_support,
        stage_match_counts=tuple(stage_counts),
        tentative_residual_syndrome=_immutable_uint8(tentative_residual),
        tentative_observable_frame=_immutable_uint8(tentative_frame),
        tentative_edge_support=tentative_support,
        physical_correction=durable_physical_correction,
        tentative_physical_correction=tentative_physical_correction,
        domain_results=MappingProxyType(domain_results),
    )


__all__ = [
    "PINBALL_V2_BASIS_PROFILE",
    "PINBALL_V2_GEOMETRY_PROFILE",
    "PINBALL_V2_SOURCE_COMMIT",
    "PINBALL_V2_STAGE_ORDER",
    "CompiledPinballV2Schedule",
    "PinballV2DomainResult",
    "PinballV2HardwareProxies",
    "PinballV2PauliTarget",
    "PinballV2Primitive",
    "PinballV2Result",
    "PinballV2StageSchedule",
    "compile_pinball_v2_schedule",
    "predecode_pinball_v2",
]
