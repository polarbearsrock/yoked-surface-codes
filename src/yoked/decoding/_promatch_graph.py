"""Canonical PyMatching graph compilation for the L1 ProMatch experiment."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
from typing import Iterable

import pymatching
import stim

from yoked.decoding._promatch_layout import (
    DetectorRole,
    L1BodyDetector,
    L1DomainKey,
    L1TerminalDetector,
    PromatchLayout,
    YokeDetector,
)


@dataclasses.dataclass(frozen=True)
class Edge:
    """Canonical matching edge shared by the global and domain-local graphs.

    Edge IDs are dense tuple indices. ``target=None`` denotes the true matching
    boundary, and ``observable_mask`` is packed little-endian by observable ID.
    """

    edge_id: int
    source: int
    target: int | None
    weight: float
    observable_mask: bytes
    source_role: DetectorRole
    target_role: DetectorRole | None

    def other_endpoint(self, detector_id: int) -> int | None:
        """Returns the opposite detector, or ``None`` for a boundary edge."""

        if detector_id == self.source:
            return self.target
        if detector_id == self.target:
            return self.source
        raise ValueError(
            f"detector {detector_id} is not incident to edge {self.edge_id}"
        )


@dataclasses.dataclass(frozen=True)
class DomainGraph:
    """Deterministically ordered domain-local projection of the full graph.

    Only eligible local edges appear in ``edges``. The adjacency mappings are
    precomputed views over those same canonical :class:`Edge` instances.
    ``adjacency``, ``neighbors``, and ``boundary_adjacency`` are plain dicts
    for pickling; they are treated as immutable after construction.
    """

    domain: L1DomainKey
    detector_ids: tuple[int, ...]
    edges: tuple[Edge, ...]
    adjacency: dict[int, tuple[Edge, ...]]
    neighbors: dict[int, tuple[int, ...]]
    boundary_edges: tuple[Edge, ...]
    boundary_adjacency: dict[int, tuple[Edge, ...]]

    def incident_edges(self, detector_id: int) -> tuple[Edge, ...]:
        """Returns the domain-local edges incident to ``detector_id``."""

        try:
            return self.adjacency[detector_id]
        except KeyError as ex:
            raise ValueError(
                f"detector {detector_id} is not in domain {self.domain!r}"
            ) from ex

    @staticmethod
    def other_endpoint(edge: Edge, detector_id: int) -> int | None:
        """Delegates canonical endpoint handling to :class:`Edge`."""

        return edge.other_endpoint(detector_id)


@dataclasses.dataclass(frozen=True)
class CompiledPromatchGraph:
    """Full residual matcher plus deterministic L1 domain projections.

    ``edges[edge_id]`` is an invariant used by correction algebra, replay, and
    provenance hashing. ``matcher`` always represents the complete DEM graph;
    domain graphs restrict only the predecoder's candidate visibility.
    ``domain_graphs`` is a plain dict for pickling; it is treated as
    immutable after construction.
    """

    layout: PromatchLayout
    matcher: pymatching.Matching
    edges: tuple[Edge, ...]
    domain_graphs: dict[L1DomainKey, DomainGraph]
    fingerprint: str
    require_zero_frame: bool
    num_detectors: int
    num_observables: int

    @property
    def edge_by_id(self) -> tuple[Edge, ...]:
        """Returns the dense canonical edge table indexed by edge ID."""

        return self.edges

    def domain_graph(self, domain: L1DomainKey) -> DomainGraph:
        """Returns the compiled local graph for ``domain``."""

        return self.domain_graphs[domain]


@dataclasses.dataclass(frozen=True)
class _UnnumberedEdge:
    source: int
    target: int | None
    weight: float
    observable_mask: bytes


def _observable_mask(fault_ids: Iterable[int], *, num_observables: int) -> bytes:
    result = bytearray((num_observables + 7) // 8)
    normalized: set[int] = set()
    for raw_fault_id in fault_ids:
        if not isinstance(raw_fault_id, numbers.Integral):
            raise ValueError(f"non-integral observable ID {raw_fault_id!r}")
        fault_id = int(raw_fault_id)
        if fault_id < 0 or fault_id >= num_observables:
            raise ValueError(f"observable ID {fault_id} outside [0, {num_observables})")
        normalized.add(fault_id)
    for fault_id in sorted(normalized):
        result[fault_id // 8] |= 1 << (fault_id % 8)
    return bytes(result)


def _is_zero_mask(mask: bytes) -> bool:
    return not any(mask)


def _validate_observable_ownership(
    edge: Edge,
    *,
    patch_id: int,
    layout: PromatchLayout,
) -> None:
    for observable_id in range(layout.num_patches * 2):
        if edge.observable_mask[observable_id // 8] & (1 << (observable_id % 8)):
            owner = layout.observable_owner(observable_id)
            if owner != patch_id:
                raise ValueError(
                    f"domain-local edge {edge.edge_id} carries observable "
                    f"{observable_id} owned by patch {owner}, expected patch {patch_id}"
                )


def _is_domain_local_candidate(edge: Edge, layout: PromatchLayout) -> bool:
    if edge.target is None:
        return False
    source_role = edge.source_role
    target_role = edge.target_role
    if not isinstance(source_role, L1BodyDetector) or not isinstance(
        target_role, L1BodyDetector
    ):
        return False
    return layout.domain_of(edge.source) == layout.domain_of(edge.target)


def _edge_sort_key(edge: _UnnumberedEdge, *, num_detectors: int) -> tuple:
    # Boundaries sort after real detector IDs and are stable across runs.
    target_key = num_detectors if edge.target is None else edge.target
    return edge.source, target_key, edge.weight, edge.observable_mask


def _adjacency_edge_key(edge: Edge, detector_id: int) -> tuple:
    other = edge.other_endpoint(detector_id)
    other_key = math.inf if other is None else other
    return edge.weight, other_key, edge.edge_id


def _graph_fingerprint(
    *,
    layout: PromatchLayout,
    edges: tuple[Edge, ...],
    require_zero_frame: bool,
) -> str:
    payload = {
        "schema": "promatch-graph-v1",
        "layout_fingerprint": layout.fingerprint,
        "require_zero_frame": require_zero_frame,
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
        "edges": [
            [
                edge.edge_id,
                edge.source,
                edge.target,
                edge.weight.hex(),
                edge.observable_mask.hex(),
            ]
            for edge in edges
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compile_matching_graph(
    dem: stim.DetectorErrorModel,
    layout: PromatchLayout,
    *,
    require_zero_frame: bool = True,
    retain_cross_lane_edges: bool = False,
) -> CompiledPromatchGraph:
    """Builds the residual matcher and canonical domain-local predecoder graphs.

    By default, the maintained ProMatch topology validation rejects non-yoke
    edges crossing patches.  ``retain_cross_lane_edges=True`` is the additive UF
    compile path: these edges remain canonical/global so a later role-derived
    projection can expose them as guard-port incidences.  The option does not
    add them to any existing :class:`DomainGraph` and intentionally does not
    alter fingerprints when the canonical edge table itself is unchanged.
    """

    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"dem must be a stim.DetectorErrorModel, got {type(dem)!r}")
    if not isinstance(retain_cross_lane_edges, bool):
        raise TypeError("retain_cross_lane_edges must be a bool")
    # Stim's DEM count properties scan the model. Cache them before the edge
    # loop instead of accidentally rescanning a large DEM for every edge.
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables
    if layout.num_detectors != num_detectors:
        raise ValueError(
            f"layout has {layout.num_detectors} detectors but DEM has {num_detectors}"
        )
    detector_coordinates = dem.get_detector_coordinates()
    if (
        tuple(
            tuple(float(v) for v in detector_coordinates[k])
            for k in range(num_detectors)
        )
        != layout.coordinates
    ):
        raise ValueError("layout coordinates do not match the supplied DEM")

    matcher = pymatching.Matching.from_detector_error_model(dem)
    matcher.ensure_num_fault_ids(num_observables)

    raw_edges: list[_UnnumberedEdge] = []
    seen: set[tuple[int, int | None, float, bytes]] = set()
    for raw_source, raw_target, edge_data in matcher.edges():
        if not isinstance(raw_source, numbers.Integral):
            raise ValueError(f"non-integral matching detector ID {raw_source!r}")
        source = int(raw_source)
        if source < 0 or source >= num_detectors:
            raise ValueError(f"matching detector ID {source} is out of range")
        if raw_target is None:
            target = None
        else:
            if not isinstance(raw_target, numbers.Integral):
                raise ValueError(f"non-integral matching detector ID {raw_target!r}")
            target = int(raw_target)
            if target < 0 or target >= num_detectors:
                raise ValueError(f"matching detector ID {target} is out of range")
            if source == target:
                raise ValueError(
                    f"matching graph contains self-loop at detector {source}"
                )
            if target < source:
                source, target = target, source

        try:
            weight = float(edge_data["weight"])
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError(
                f"matching edge {(source, target)} has invalid weight"
            ) from ex
        if not math.isfinite(weight):
            raise ValueError(
                f"matching edge {(source, target)} has nonfinite weight {weight!r}"
            )
        mask = _observable_mask(
            edge_data.get("fault_ids", ()), num_observables=num_observables
        )
        identity = (source, target, weight, mask)
        if identity in seen:
            raise ValueError(f"duplicate normalized matching edge {identity!r}")
        seen.add(identity)
        raw_edges.append(_UnnumberedEdge(source, target, weight, mask))

    raw_edges.sort(key=lambda edge: _edge_sort_key(edge, num_detectors=num_detectors))
    edges = tuple(
        Edge(
            edge_id=edge_id,
            source=edge.source,
            target=edge.target,
            weight=edge.weight,
            observable_mask=edge.observable_mask,
            source_role=layout.role_of(edge.source),
            target_role=None if edge.target is None else layout.role_of(edge.target),
        )
        for edge_id, edge in enumerate(raw_edges)
    )

    # Validate the supported graph topology before frame-policy filtering.
    for edge in edges:
        if edge.target is None:
            if isinstance(edge.source_role, L1BodyDetector):
                _validate_observable_ownership(
                    edge,
                    patch_id=edge.source_role.patch_id,
                    layout=layout,
                )
            continue
        source_role = edge.source_role
        target_role = edge.target_role
        if isinstance(source_role, YokeDetector) or isinstance(
            target_role, YokeDetector
        ):
            continue
        source_inner = isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
        target_inner = isinstance(target_role, (L1BodyDetector, L1TerminalDetector))
        if not source_inner or not target_inner:
            raise ValueError(f"unsupported detector roles on edge {edge.edge_id}")
        if (
            source_role.patch_id != target_role.patch_id  # type: ignore[union-attr]
            and not retain_cross_lane_edges
        ):
            raise ValueError(
                "unsupported non-boundary edge crosses patches: "
                f"edge={edge.edge_id}, source_role={source_role!r}, "
                f"target_role={target_role!r}"
            )

        # One-yoke (YBerg) DEMs contain genuine within-patch X/Z edges, and a
        # future supported noise model may expose the same residual topology.
        # They remain canonical/global but cannot enter a patch-and-basis
        # DomainGraph below, independent of the number of yoke detectors.

        if _is_domain_local_candidate(edge, layout):
            _validate_observable_ownership(
                edge,
                patch_id=source_role.patch_id,  # type: ignore[union-attr]
                layout=layout,
            )
            if edge.weight < 0:
                raise ValueError(
                    f"eligible edge {edge.edge_id} has negative weight {edge.weight}"
                )
            if require_zero_frame and not _is_zero_mask(edge.observable_mask):
                raise ValueError(
                    f"eligible edge {edge.edge_id} has nonzero observable mask "
                    f"{edge.observable_mask.hex()} in zero-frame mode"
                )

    domain_graphs: dict[L1DomainKey, DomainGraph] = {}
    for domain in layout.domains:
        detector_ids = layout.domain_detector_ids[domain]
        detector_set = set(detector_ids)
        domain_edges: list[Edge] = []
        boundary_edges: list[Edge] = []
        adjacency_lists: dict[int, list[Edge]] = {k: [] for k in detector_ids}
        neighbor_sets: dict[int, set[int]] = {k: set() for k in detector_ids}
        boundary_adjacency_lists: dict[int, list[Edge]] = {k: [] for k in detector_ids}
        for edge in edges:
            if edge.target is None:
                if edge.source not in detector_set:
                    continue
                if require_zero_frame and not _is_zero_mask(edge.observable_mask):
                    continue
                if edge.weight < 0:
                    raise ValueError(
                        f"eligible boundary edge {edge.edge_id} has negative weight {edge.weight}"
                    )
                domain_edges.append(edge)
                boundary_edges.append(edge)
                adjacency_lists[edge.source].append(edge)
                boundary_adjacency_lists[edge.source].append(edge)
                continue
            if edge.source not in detector_set or edge.target not in detector_set:
                continue
            # Membership in the same detector set is the strongest possible
            # domain-local check, including mode-specific window assignment.
            if require_zero_frame and not _is_zero_mask(edge.observable_mask):
                raise AssertionError(
                    "nonzero domain edge escaped zero-frame validation"
                )
            domain_edges.append(edge)
            adjacency_lists[edge.source].append(edge)
            adjacency_lists[edge.target].append(edge)
            neighbor_sets[edge.source].add(edge.target)
            neighbor_sets[edge.target].add(edge.source)

        adjacency = {
            detector_id: tuple(
                sorted(
                    adjacency_lists[detector_id],
                    key=lambda edge, d=detector_id: _adjacency_edge_key(edge, d),
                )
            )
            for detector_id in detector_ids
        }
        boundary_adjacency = {
            detector_id: tuple(
                sorted(
                    boundary_adjacency_lists[detector_id],
                    key=lambda edge, d=detector_id: _adjacency_edge_key(edge, d),
                )
            )
            for detector_id in detector_ids
        }
        domain_graphs[domain] = DomainGraph(
            domain=domain,
            detector_ids=detector_ids,
            edges=tuple(sorted(domain_edges, key=lambda edge: edge.edge_id)),
            adjacency=adjacency,
            neighbors={k: tuple(sorted(v)) for k, v in neighbor_sets.items()},
            boundary_edges=tuple(sorted(boundary_edges, key=lambda edge: edge.edge_id)),
            boundary_adjacency=boundary_adjacency,
        )

    fingerprint = _graph_fingerprint(
        layout=layout, edges=edges, require_zero_frame=require_zero_frame
    )
    return CompiledPromatchGraph(
        layout=layout,
        matcher=matcher,
        edges=edges,
        domain_graphs=domain_graphs,
        fingerprint=fingerprint,
        require_zero_frame=require_zero_frame,
        num_detectors=num_detectors,
        num_observables=num_observables,
    )
