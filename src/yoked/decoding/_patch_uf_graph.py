"""Terminal-inclusive patch-local graph projection for weighted UF decoding."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from typing import Literal

import stim

from yoked.decoding._dem_catalog import validate_uf_dem_catalog
from yoked.decoding._promatch_graph import CompiledPromatchGraph, Edge
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1TerminalDetector,
    YokeDetector,
)


LaneBasis = Literal["X", "Z"]
EdgeOwnerKind = Literal["local-correction", "global-port"]
IncidenceKind = Literal["correction", "true-boundary", "guard-port"]
PortKind = Literal["yoke", "cross-lane"]
DetectorRoleKind = Literal["body", "terminal", "yoke"]
_NONE_SENTINEL = -1


@dataclasses.dataclass(frozen=True, order=True)
class ExactDyadic:
    """Normalized exact value ``integer * 2**binary_exponent``."""

    integer: int
    binary_exponent: int

    def __post_init__(self) -> None:
        if isinstance(self.integer, bool) or not isinstance(self.integer, int):
            raise TypeError("dyadic integer must be an int")
        if isinstance(self.binary_exponent, bool) or not isinstance(
            self.binary_exponent, int
        ):
            raise TypeError("dyadic binary_exponent must be an int")
        if self.integer == 0:
            if self.binary_exponent != 0:
                raise ValueError("zero dyadic must use exponent zero")
        elif self.integer % 2 == 0:
            raise ValueError("nonzero dyadic integer must be odd")

    @classmethod
    def from_float(cls, value: float) -> "ExactDyadic":
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("cannot represent a nonfinite dyadic")
        numerator, denominator = value.as_integer_ratio()
        if numerator == 0:
            return cls(0, 0)
        exponent = -(denominator.bit_length() - 1)
        while numerator % 2 == 0:
            numerator //= 2
            exponent += 1
        return cls(numerator, exponent)

    @classmethod
    def _normalized(cls, integer: int, exponent: int) -> "ExactDyadic":
        if integer == 0:
            return cls(0, 0)
        while integer % 2 == 0:
            integer //= 2
            exponent += 1
        return cls(integer, exponent)

    def __add__(self, other: object) -> "ExactDyadic":
        if not isinstance(other, ExactDyadic):
            return NotImplemented
        exponent = min(self.binary_exponent, other.binary_exponent)
        integer = (self.integer << (self.binary_exponent - exponent)) + (
            other.integer << (other.binary_exponent - exponent)
        )
        return ExactDyadic._normalized(integer, exponent)


@dataclasses.dataclass(frozen=True, order=True)
class PatchUFLaneKey:
    patch_id: int
    check_basis: LaneBasis


@dataclasses.dataclass(frozen=True)
class PatchUFCorrectionEdge:
    edge_id: int
    local_source: int
    local_target: int
    exact_weight_index: int


@dataclasses.dataclass(frozen=True)
class TrueBoundaryIncidence:
    edge_id: int
    local_vertex: int
    exact_weight_index: int


@dataclasses.dataclass(frozen=True)
class GuardPortIncidence:
    edge_id: int
    lane_id: int
    local_vertex: int
    remote_detector_id: int
    remote_lane_id: int | None
    port_kind: PortKind
    exact_weight_index: int
    observable_mask: bytes


@dataclasses.dataclass(frozen=True)
class PatchUFIncidence:
    kind: IncidenceKind
    item_index: int
    edge_id: int
    local_vertex: int


@dataclasses.dataclass(frozen=True)
class PatchUFLaneProjection:
    lane_id: int
    key: PatchUFLaneKey
    global_detector_ids: tuple[int, ...]
    local_x2: tuple[int, ...]
    y2: tuple[int, ...]
    times: tuple[int, ...]
    internal_correction_edges: tuple[PatchUFCorrectionEdge, ...]
    true_boundary_edges: tuple[TrueBoundaryIncidence, ...]
    guard_ports: tuple[GuardPortIncidence, ...]
    incidences: tuple[PatchUFIncidence, ...]
    incidence_offsets: tuple[int, ...]
    incidence_indices: tuple[int, ...]

    def incident(self, local_vertex: int) -> tuple[PatchUFIncidence, ...]:
        if local_vertex < 0 or local_vertex >= len(self.global_detector_ids):
            raise ValueError(f"local vertex {local_vertex} is out of range")
        start = self.incidence_offsets[local_vertex]
        stop = self.incidence_offsets[local_vertex + 1]
        return tuple(self.incidences[self.incidence_indices[k]] for k in range(start, stop))


@dataclasses.dataclass(frozen=True)
class PatchUFSupportEdge:
    edge_id: int
    source: int
    target: int | None
    observable_mask: bytes
    exact_weight_index: int
    owner_kind: EdgeOwnerKind
    owner_lane: int | None


@dataclasses.dataclass(frozen=True)
class PatchUFProjection:
    canonical_graph_fingerprint: str
    validated_catalog_fingerprint: str
    num_detectors: int
    num_observables: int
    num_patches: int
    lanes: tuple[PatchUFLaneProjection, ...]
    patch_lane_ids: tuple[tuple[int, int], ...]
    detector_lane_id: tuple[int | None, ...]
    detector_local_index: tuple[int | None, ...]
    detector_role_kind: tuple[DetectorRoleKind, ...]
    detector_lane_id_array: tuple[int, ...]
    detector_local_index_array: tuple[int, ...]
    support_edges: tuple[PatchUFSupportEdge, ...]
    edge_owner_kind: tuple[EdgeOwnerKind, ...]
    edge_owner_lane: tuple[int | None, ...]
    edge_owner_lane_array: tuple[int, ...]
    exact_weights: tuple[ExactDyadic, ...]
    fingerprint: str

    def lane(self, key: PatchUFLaneKey) -> PatchUFLaneProjection:
        lane_id = self.patch_lane_ids[key.patch_id][0 if key.check_basis == "X" else 1]
        lane = self.lanes[lane_id]
        if lane.key != key:
            raise AssertionError("patch lane index is inconsistent")
        return lane


@dataclasses.dataclass(frozen=True)
class SupportReplay:
    detector_boundary: tuple[int, ...]
    observable_mask: bytes
    exact_weight: ExactDyadic


_InnerRole = L1BodyDetector | L1TerminalDetector


def _inner_lane_key(role: object) -> PatchUFLaneKey | None:
    if isinstance(role, (L1BodyDetector, L1TerminalDetector)):
        return PatchUFLaneKey(role.patch_id, role.check_basis)
    return None


def _exact_coordinate2(value: float, *, name: str) -> int:
    doubled = 2 * float(value)
    rounded = round(doubled)
    if not math.isclose(doubled, rounded, rel_tol=0, abs_tol=2e-7):
        raise ValueError(f"{name} must be half-integral, got {value!r}")
    return int(rounded)


def _validate_mask(mask: bytes, *, num_observables: int) -> bytes:
    result = bytes(mask)
    expected = (num_observables + 7) // 8
    if len(result) != expected:
        raise ValueError(
            f"observable mask has {len(result)} bytes; expected {expected}"
        )
    if num_observables % 8 and result and result[-1] >> (num_observables % 8):
        raise ValueError("observable mask has nonzero unused tail bits")
    return result


def _role_row(role: object) -> list[object]:
    if isinstance(role, L1BodyDetector):
        return ["body", role.patch_id, role.check_basis, role.time, role.window_id]
    if isinstance(role, L1TerminalDetector):
        return ["terminal", role.patch_id, role.check_basis, role.time]
    if isinstance(role, YokeDetector):
        return ["yoke"]
    raise ValueError(f"unsupported detector role {role!r}")


def _role_kind(role: object) -> DetectorRoleKind:
    if isinstance(role, L1BodyDetector):
        return "body"
    if isinstance(role, L1TerminalDetector):
        return "terminal"
    if isinstance(role, YokeDetector):
        return "yoke"
    raise ValueError(f"unsupported detector role {role!r}")


def _projection_fingerprint(
    *,
    graph: CompiledPromatchGraph,
    catalog_fingerprint: str,
    lanes: tuple[PatchUFLaneProjection, ...],
    support_edges: tuple[PatchUFSupportEdge, ...],
    exact_weights: tuple[ExactDyadic, ...],
) -> str:
    payload = {
        "schema": "patch-uf-projection-v1",
        "canonical_graph_fingerprint": graph.fingerprint,
        "validated_catalog_fingerprint": catalog_fingerprint,
        "policies": {
            "boundary": "true-none-is-local-v1",
            "cross_lane": "global-port-two-incidences-v1",
            "frame": "local-corrections-zero-v1",
            "topology": "ysc-inner-terminal-yoke-allowlist-v1",
            "weight": "finite-strict-positive-exact-binary64-dyadic-v1",
            "none_sentinel": _NONE_SENTINEL,
        },
        "roles": [_role_row(role) for role in graph.layout.roles],
        "coordinates": [list(row) for row in graph.layout.coordinates],
        "exact_weights": [
            [str(value.integer), value.binary_exponent] for value in exact_weights
        ],
        "lanes": [
            {
                "lane_id": lane.lane_id,
                "key": [lane.key.patch_id, lane.key.check_basis],
                "detectors": list(lane.global_detector_ids),
                "local_x2": list(lane.local_x2),
                "y2": list(lane.y2),
                "times": list(lane.times),
                "corrections": [dataclasses.astuple(edge) for edge in lane.internal_correction_edges],
                "boundaries": [dataclasses.astuple(edge) for edge in lane.true_boundary_edges],
                "ports": [
                    [
                        port.edge_id,
                        port.lane_id,
                        port.local_vertex,
                        port.remote_detector_id,
                        port.remote_lane_id,
                        port.port_kind,
                        port.exact_weight_index,
                        port.observable_mask.hex(),
                    ]
                    for port in lane.guard_ports
                ],
                "incidences": [dataclasses.astuple(value) for value in lane.incidences],
                "incidence_offsets": list(lane.incidence_offsets),
                "incidence_indices": list(lane.incidence_indices),
            }
            for lane in lanes
        ],
        "edge_ownership": [
            [
                edge.edge_id,
                edge.source,
                edge.target,
                edge.observable_mask.hex(),
                edge.exact_weight_index,
                edge.owner_kind,
                edge.owner_lane,
            ]
            for edge in support_edges
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compile_patch_uf_projection(
    dem: stim.DetectorErrorModel,
    graph: CompiledPromatchGraph,
) -> PatchUFProjection:
    """Builds the immutable terminal-inclusive projection used by patch UF."""

    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"dem must be a stim.DetectorErrorModel, got {type(dem)!r}")
    if not isinstance(graph, CompiledPromatchGraph):
        raise TypeError(
            f"graph must be a CompiledPromatchGraph, got {type(graph)!r}"
        )
    if graph.layout.mode != "fullhistory":
        raise ValueError("patch UF requires a fullhistory layout")
    if (dem.num_detectors, dem.num_observables) != (
        graph.num_detectors,
        graph.num_observables,
    ):
        raise ValueError("DEM and canonical graph dimensions disagree")
    catalog = validate_uf_dem_catalog(dem, graph)

    lane_keys = tuple(
        sorted(
            {
                key
                for role in graph.layout.roles
                if (key := _inner_lane_key(role)) is not None
            }
        )
    )
    expected_keys = tuple(
        PatchUFLaneKey(patch_id, basis)
        for patch_id in range(graph.layout.num_patches)
        for basis in ("X", "Z")
    )
    if lane_keys != expected_keys:
        raise ValueError("inner detector roles do not form two lanes per patch")
    lane_id_by_key = {key: lane_id for lane_id, key in enumerate(lane_keys)}

    detector_lists: list[list[int]] = [[] for _ in lane_keys]
    detector_lane_id: list[int | None] = [None] * graph.num_detectors
    for detector_id, role in enumerate(graph.layout.roles):
        key = _inner_lane_key(role)
        if key is None:
            if not isinstance(role, YokeDetector):
                raise ValueError(f"unsupported detector role {role!r}")
            continue
        lane_id = lane_id_by_key[key]
        detector_lane_id[detector_id] = lane_id
        detector_lists[lane_id].append(detector_id)
    detector_local_index: list[int | None] = [None] * graph.num_detectors
    for detector_ids in detector_lists:
        for local_index, detector_id in enumerate(detector_ids):
            detector_local_index[detector_id] = local_index
    detector_role_kind = tuple(_role_kind(role) for role in graph.layout.roles)

    exact_weights: list[ExactDyadic] = []
    exact_weight_index: dict[ExactDyadic, int] = {}
    edge_weight_indices: list[int] = []
    for edge in graph.edges:
        if not math.isfinite(edge.weight) or edge.weight <= 0:
            raise ValueError(
                f"patch UF edge {edge.edge_id} must have finite strictly positive "
                f"weight, got {edge.weight!r}"
            )
        value = ExactDyadic.from_float(edge.weight)
        index = exact_weight_index.get(value)
        if index is None:
            index = len(exact_weights)
            exact_weight_index[value] = index
            exact_weights.append(value)
        edge_weight_indices.append(index)

    corrections: list[list[PatchUFCorrectionEdge]] = [[] for _ in lane_keys]
    boundaries: list[list[TrueBoundaryIncidence]] = [[] for _ in lane_keys]
    ports: list[list[GuardPortIncidence]] = [[] for _ in lane_keys]
    owner_kind: list[EdgeOwnerKind] = []
    owner_lane: list[int | None] = []
    support_edges: list[PatchUFSupportEdge] = []

    for edge in graph.edges:
        if edge.edge_id != len(support_edges):
            raise ValueError("canonical edge IDs must be dense tuple indices")
        mask = _validate_mask(
            edge.observable_mask, num_observables=graph.num_observables
        )
        weight_index = edge_weight_indices[edge.edge_id]
        source_lane = detector_lane_id[edge.source]
        target_lane = None if edge.target is None else detector_lane_id[edge.target]

        if edge.target is None:
            if source_lane is None:
                raise ValueError(
                    f"unsupported yoke-to-boundary edge {edge.edge_id}"
                )
            if any(mask):
                raise ValueError(
                    f"local correction edge {edge.edge_id} has nonzero observable frame"
                )
            local = detector_local_index[edge.source]
            assert local is not None
            boundaries[source_lane].append(
                TrueBoundaryIncidence(edge.edge_id, local, weight_index)
            )
            kind: EdgeOwnerKind = "local-correction"
            lane_owner: int | None = source_lane
        elif source_lane is not None and target_lane is not None and source_lane == target_lane:
            if any(mask):
                raise ValueError(
                    f"local correction edge {edge.edge_id} has nonzero observable frame"
                )
            local_source = detector_local_index[edge.source]
            local_target = detector_local_index[edge.target]
            assert local_source is not None and local_target is not None
            corrections[source_lane].append(
                PatchUFCorrectionEdge(
                    edge.edge_id, local_source, local_target, weight_index
                )
            )
            kind = "local-correction"
            lane_owner = source_lane
        else:
            local_endpoints: list[tuple[int, int, int, int | None, PortKind]] = []
            if source_lane is not None:
                local = detector_local_index[edge.source]
                assert local is not None and edge.target is not None
                local_endpoints.append(
                    (
                        source_lane,
                        local,
                        edge.target,
                        target_lane,
                        "yoke" if target_lane is None else "cross-lane",
                    )
                )
            if target_lane is not None:
                local = detector_local_index[edge.target]
                assert local is not None
                local_endpoints.append(
                    (
                        target_lane,
                        local,
                        edge.source,
                        source_lane,
                        "yoke" if source_lane is None else "cross-lane",
                    )
                )
            if not local_endpoints:
                raise ValueError(f"unsupported yoke-to-yoke edge {edge.edge_id}")
            for lane_id, local, remote, remote_lane, port_kind in local_endpoints:
                ports[lane_id].append(
                    GuardPortIncidence(
                        edge_id=edge.edge_id,
                        lane_id=lane_id,
                        local_vertex=local,
                        remote_detector_id=remote,
                        remote_lane_id=remote_lane,
                        port_kind=port_kind,
                        exact_weight_index=weight_index,
                        observable_mask=mask,
                    )
                )
            kind = "global-port"
            lane_owner = None

        owner_kind.append(kind)
        owner_lane.append(lane_owner)
        support_edges.append(
            PatchUFSupportEdge(
                edge_id=edge.edge_id,
                source=edge.source,
                target=edge.target,
                observable_mask=mask,
                exact_weight_index=weight_index,
                owner_kind=kind,
                owner_lane=lane_owner,
            )
        )

    lanes: list[PatchUFLaneProjection] = []
    incidence_order = {"correction": 0, "true-boundary": 1, "guard-port": 2}
    for lane_id, key in enumerate(lane_keys):
        detector_ids = tuple(detector_lists[lane_id])
        local_x2: list[int] = []
        y2: list[int] = []
        times: list[int] = []
        for detector_id in detector_ids:
            role = graph.layout.role_of(detector_id)
            assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
            x, y, *_ = graph.layout.coordinates[detector_id]
            local_x = x - key.patch_id * graph.layout.pitch
            local_x2.append(
                _exact_coordinate2(local_x, name=f"detector {detector_id} local_x")
            )
            y2.append(_exact_coordinate2(y, name=f"detector {detector_id} y"))
            times.append(role.time)

        lane_corrections = tuple(sorted(corrections[lane_id], key=lambda e: e.edge_id))
        lane_boundaries = tuple(sorted(boundaries[lane_id], key=lambda e: e.edge_id))
        lane_ports = tuple(
            sorted(ports[lane_id], key=lambda p: (p.edge_id, p.local_vertex))
        )
        raw_incidences: list[PatchUFIncidence] = []
        for index, edge in enumerate(lane_corrections):
            raw_incidences.append(
                PatchUFIncidence("correction", index, edge.edge_id, edge.local_source)
            )
            raw_incidences.append(
                PatchUFIncidence("correction", index, edge.edge_id, edge.local_target)
            )
        for index, edge in enumerate(lane_boundaries):
            raw_incidences.append(
                PatchUFIncidence("true-boundary", index, edge.edge_id, edge.local_vertex)
            )
        for index, port in enumerate(lane_ports):
            raw_incidences.append(
                PatchUFIncidence("guard-port", index, port.edge_id, port.local_vertex)
            )
        incidences = tuple(
            sorted(
                raw_incidences,
                key=lambda value: (
                    value.local_vertex,
                    incidence_order[value.kind],
                    value.edge_id,
                    value.item_index,
                ),
            )
        )
        by_vertex: dict[int, list[int]] = defaultdict(list)
        for index, incidence in enumerate(incidences):
            by_vertex[incidence.local_vertex].append(index)
        offsets = [0]
        indices: list[int] = []
        for local_vertex in range(len(detector_ids)):
            indices.extend(by_vertex[local_vertex])
            offsets.append(len(indices))
        lanes.append(
            PatchUFLaneProjection(
                lane_id=lane_id,
                key=key,
                global_detector_ids=detector_ids,
                local_x2=tuple(local_x2),
                y2=tuple(y2),
                times=tuple(times),
                internal_correction_edges=lane_corrections,
                true_boundary_edges=lane_boundaries,
                guard_ports=lane_ports,
                incidences=incidences,
                incidence_offsets=tuple(offsets),
                incidence_indices=tuple(indices),
            )
        )

    concrete_lanes = tuple(lanes)
    concrete_support = tuple(support_edges)
    concrete_weights = tuple(exact_weights)
    patch_lane_ids = tuple(
        (
            lane_id_by_key[PatchUFLaneKey(patch_id, "X")],
            lane_id_by_key[PatchUFLaneKey(patch_id, "Z")],
        )
        for patch_id in range(graph.layout.num_patches)
    )
    fingerprint = _projection_fingerprint(
        graph=graph,
        catalog_fingerprint=catalog.fingerprint,
        lanes=concrete_lanes,
        support_edges=concrete_support,
        exact_weights=concrete_weights,
    )
    return PatchUFProjection(
        canonical_graph_fingerprint=graph.fingerprint,
        validated_catalog_fingerprint=catalog.fingerprint,
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        num_patches=graph.layout.num_patches,
        lanes=concrete_lanes,
        patch_lane_ids=patch_lane_ids,
        detector_lane_id=tuple(detector_lane_id),
        detector_local_index=tuple(detector_local_index),
        detector_role_kind=detector_role_kind,
        detector_lane_id_array=tuple(
            _NONE_SENTINEL if value is None else value for value in detector_lane_id
        ),
        detector_local_index_array=tuple(
            _NONE_SENTINEL if value is None else value
            for value in detector_local_index
        ),
        support_edges=concrete_support,
        edge_owner_kind=tuple(owner_kind),
        edge_owner_lane=tuple(owner_lane),
        edge_owner_lane_array=tuple(
            _NONE_SENTINEL if value is None else value for value in owner_lane
        ),
        exact_weights=concrete_weights,
        fingerprint=fingerprint,
    )


def replay_support(
    projection: PatchUFProjection,
    edge_ids: tuple[int, ...],
    *,
    expected_owner_lane: int | None = None,
) -> SupportReplay:
    """Reconstructs the exact GF(2) boundary, frame, and weight of a support."""

    if not isinstance(projection, PatchUFProjection):
        raise TypeError("projection must be a PatchUFProjection")
    normalized: list[int] = []
    for edge_id in edge_ids:
        if isinstance(edge_id, bool) or not isinstance(edge_id, int):
            raise TypeError("support edge IDs must be integers")
        normalized.append(edge_id)
    if tuple(normalized) != tuple(sorted(set(normalized))):
        raise ValueError("support edge IDs must be sorted and unique")
    if expected_owner_lane is not None and (
        isinstance(expected_owner_lane, bool)
        or not isinstance(expected_owner_lane, int)
        or expected_owner_lane < 0
        or expected_owner_lane >= len(projection.lanes)
    ):
        raise ValueError("expected_owner_lane is out of range")

    boundary: set[int] = set()
    mask = bytearray((projection.num_observables + 7) // 8)
    weight = ExactDyadic(0, 0)
    for edge_id in normalized:
        if edge_id < 0 or edge_id >= len(projection.support_edges):
            raise ValueError(f"support edge ID {edge_id} is out of range")
        edge = projection.support_edges[edge_id]
        if edge.owner_kind != "local-correction" or edge.owner_lane is None:
            raise ValueError(f"support edge {edge_id} is not correction-eligible")
        if expected_owner_lane is not None and edge.owner_lane != expected_owner_lane:
            raise ValueError(
                f"support edge {edge_id} belongs to lane {edge.owner_lane}, "
                f"expected {expected_owner_lane}"
            )
        if edge.source in boundary:
            boundary.remove(edge.source)
        else:
            boundary.add(edge.source)
        if edge.target is not None:
            if edge.target in boundary:
                boundary.remove(edge.target)
            else:
                boundary.add(edge.target)
        for index, value in enumerate(edge.observable_mask):
            mask[index] ^= value
        weight = weight + projection.exact_weights[edge.exact_weight_index]
    _validate_mask(bytes(mask), num_observables=projection.num_observables)
    return SupportReplay(tuple(sorted(boundary)), bytes(mask), weight)


__all__ = [
    "DetectorRoleKind",
    "EdgeOwnerKind",
    "ExactDyadic",
    "GuardPortIncidence",
    "IncidenceKind",
    "PatchUFCorrectionEdge",
    "PatchUFIncidence",
    "PatchUFLaneKey",
    "PatchUFLaneProjection",
    "PatchUFProjection",
    "PatchUFSupportEdge",
    "PortKind",
    "SupportReplay",
    "TrueBoundaryIncidence",
    "compile_patch_uf_projection",
    "replay_support",
]
