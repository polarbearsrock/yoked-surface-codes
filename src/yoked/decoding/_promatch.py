"""Deterministic, domain-local ProMatch-style predecoding core.

This module deliberately has no Sinter-facing code.  It consumes the canonical
graph compiled by :mod:`yoked.decoding._promatch_graph`, operates on one
unpacked shot, and returns enough edge-level information to check the complete
GF(2) correction algebra independently of the residual decoder.
"""

from __future__ import annotations

import dataclasses
import enum
import heapq
import math
from collections import defaultdict
from typing import Iterable, Literal, TypeAlias

import numpy as np

from yoked.decoding._promatch_graph import CompiledPromatchGraph, DomainGraph, Edge
from yoked.decoding._promatch_layout import L1DomainKey


ObservablePolicy: TypeAlias = Literal["edge-zero", "path-zero", "any", "zero-frame"]
BoundaryPolicy: TypeAlias = Literal["disabled", "odd-parity"]

# Detector IDs are nonnegative.  This sentinel exists only inside the domain
# algorithm and is never written into a syndrome or PrematchedPath.
_BOUNDARY = -1


@dataclasses.dataclass(frozen=True)
class PrematchedPath:
    domain: L1DomainKey
    stage: int
    endpoints: tuple[int, int | None]
    edge_ids: tuple[int, ...]
    decision_weight: float
    observable_mask: bytes


class FallbackReason(enum.Enum):
    NO_CANDIDATE = "no-candidate"
    DISCONNECTED = "disconnected"
    BOUNDARY_UNAVAILABLE = "boundary-unavailable"


@dataclasses.dataclass(frozen=True)
class DomainPrematchStats:
    initial_hw: int
    attempted_residual_hw: int
    final_residual_hw: int
    attempted_stage_counts: tuple[int, int, int, int]
    committed_stage_counts: tuple[int, int, int, int]
    attempted_matches: int
    committed_matches: int
    status: Literal["below-limit", "success", "rollback"]
    fallback_reason: FallbackReason | None
    boundary_was_added: bool = False
    boundary_was_used: bool = False
    boundary_discarded_unused: bool = False


@dataclasses.dataclass(frozen=True)
class PrematchResult:
    residual_syndrome: np.ndarray
    observable_frame: np.ndarray
    paths: tuple[PrematchedPath, ...]
    domain_stats: dict[L1DomainKey, DomainPrematchStats]
    decision_weight: float
    xor_support_weight: float


@dataclasses.dataclass(frozen=True)
class _Candidate:
    stage: int
    endpoints: tuple[int, int]
    edges: tuple[Edge, ...]
    key: tuple


@dataclasses.dataclass(frozen=True)
class _DomainAttempt:
    active: frozenset[int]
    paths: tuple[PrematchedPath, ...]
    stage_counts: tuple[int, int, int, int]
    fallback_reason: FallbackReason | None
    boundary_was_added: bool
    boundary_was_used: bool

    @property
    def success(self) -> bool:
        return self.fallback_reason is None


def _is_zero_mask(mask: bytes) -> bool:
    return not any(mask)


def _xor_mask(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError(
            f"observable masks have inconsistent lengths {len(left)} and {len(right)}"
        )
    return bytes(a ^ b for a, b in zip(left, right))


def _path_mask(edges: Iterable[Edge], *, mask_bytes: int) -> bytes:
    result = bytes(mask_bytes)
    for edge in edges:
        result = _xor_mask(result, bytes(edge.observable_mask))
    return result


def _validate_path_observable_ownership(
    path: PrematchedPath,
    *,
    compiled_graph: CompiledPromatchGraph,
) -> None:
    layout = getattr(compiled_graph, "layout", None)
    for observable_id in range(compiled_graph.num_observables):
        if not path.observable_mask[observable_id // 8] & (1 << (observable_id % 8)):
            continue
        owner = (
            layout.observable_owner(observable_id)
            if layout is not None
            else observable_id // 2
        )
        if owner != path.domain.patch_id:
            raise ValueError(
                f"prematch path in patch {path.domain.patch_id} carries observable "
                f"{observable_id} owned by patch {owner}"
            )


def _normalized_observable_policy(policy: ObservablePolicy) -> str:
    if policy == "zero-frame":
        return "edge-zero"
    if policy not in {"edge-zero", "path-zero", "any"}:
        raise ValueError(f"unsupported observable policy {policy!r}")
    return policy


def _endpoint_key(endpoint: int, *, num_detectors: int) -> int:
    return num_detectors if endpoint == _BOUNDARY else endpoint


def _public_endpoints(endpoints: tuple[int, int]) -> tuple[int, int | None]:
    a, b = endpoints
    if a == _BOUNDARY:
        return b, None
    if b == _BOUNDARY:
        return a, None
    return a, b


class _DomainEngine:
    def __init__(
        self,
        graph: DomainGraph,
        *,
        num_detectors: int,
        num_observables: int,
        hw_limit: int,
        boundary_policy: BoundaryPolicy,
        observable_policy: ObservablePolicy,
    ) -> None:
        self.graph = graph
        self.num_detectors = num_detectors
        self.mask_bytes = (num_observables + 7) // 8
        self.hw_limit = hw_limit
        self.boundary_policy = boundary_policy
        self.observable_policy = _normalized_observable_policy(observable_policy)
        self.vertices = frozenset(int(v) for v in graph.detector_ids)

        if boundary_policy not in {"disabled", "odd-parity"}:
            raise ValueError(f"unsupported boundary policy {boundary_policy!r}")

        self.edges = tuple(graph.edges)
        self.edge_by_id: dict[int, Edge] = {}
        self.detector_adjacency: dict[int, list[Edge]] = {
            v: [] for v in self.vertices
        }
        self.boundary_edges: list[Edge] = []
        for edge in self.edges:
            self._validate_edge(edge)
            if edge.edge_id in self.edge_by_id:
                raise ValueError(f"duplicate edge ID {edge.edge_id} in domain graph")
            self.edge_by_id[edge.edge_id] = edge
            self.detector_adjacency[edge.source].append(edge)
            if edge.target is None:
                self.boundary_edges.append(edge)
            else:
                self.detector_adjacency[edge.target].append(edge)

        for incident in self.detector_adjacency.values():
            incident.sort(key=lambda e: (e.weight, e.edge_id))
        self.boundary_edges.sort(key=lambda e: (e.weight, e.source, e.edge_id))

    def _validate_edge(self, edge: Edge) -> None:
        if edge.source not in self.vertices:
            raise ValueError(
                f"edge {edge.edge_id} source {edge.source} is outside its domain"
            )
        if edge.target is not None:
            if edge.target not in self.vertices:
                raise ValueError(
                    f"edge {edge.edge_id} target {edge.target} is outside its domain"
                )
            if edge.source == edge.target:
                raise ValueError(f"edge {edge.edge_id} is a self-loop")
        if not math.isfinite(edge.weight) or edge.weight < 0:
            raise ValueError(f"edge {edge.edge_id} has invalid weight {edge.weight!r}")
        if len(edge.observable_mask) != self.mask_bytes:
            raise ValueError(
                f"edge {edge.edge_id} has {len(edge.observable_mask)} observable-mask "
                f"bytes; expected {self.mask_bytes}"
            )

    def _edge_allowed_directly(self, edge: Edge) -> bool:
        if self.observable_policy in {"edge-zero", "path-zero"}:
            return _is_zero_mask(edge.observable_mask)
        return True

    def _edge_allowed_in_path(self, edge: Edge) -> bool:
        if self.observable_policy == "edge-zero":
            return _is_zero_mask(edge.observable_mask)
        return True

    def _active_neighbors(
        self, active: set[int], *, boundary_active: bool
    ) -> dict[int, frozenset[int]]:
        nodes = set(active)
        if boundary_active:
            nodes.add(_BOUNDARY)
        result: dict[int, set[int]] = {node: set() for node in nodes}
        for edge in self.edges:
            if not self._edge_allowed_directly(edge):
                continue
            a = edge.source
            b = _BOUNDARY if edge.target is None else edge.target
            if a in nodes and b in nodes:
                result[a].add(b)
                result[b].add(a)
        return {node: frozenset(neighbors) for node, neighbors in result.items()}

    @staticmethod
    def _creates_new_singleton(
        endpoints: tuple[int, int], neighbors: dict[int, frozenset[int]]
    ) -> bool:
        removed = frozenset(endpoints)
        for node, old_neighbors in neighbors.items():
            if node in removed or not old_neighbors:
                continue
            if not old_neighbors.difference(removed):
                return True
        return False

    def _direct_candidates(
        self, active: set[int], *, boundary_active: bool
    ) -> list[tuple[tuple[int, int], Edge]]:
        result: list[tuple[tuple[int, int], Edge]] = []
        for edge in self.edges:
            if not self._edge_allowed_directly(edge):
                continue
            a = edge.source
            b = _BOUNDARY if edge.target is None else edge.target
            if a not in active:
                continue
            if b == _BOUNDARY:
                if not boundary_active:
                    continue
            elif b not in active:
                continue
            result.append(((a, b), edge))
        return result

    def _direct_key(
        self,
        endpoints: tuple[int, int],
        edge: Edge,
        *,
        substage: int | None = None,
    ) -> tuple:
        a, b = endpoints
        endpoint_keys = sorted(
            (
                _endpoint_key(a, num_detectors=self.num_detectors),
                _endpoint_key(b, num_detectors=self.num_detectors),
            )
        )
        suffix = (edge.weight, endpoint_keys[0], endpoint_keys[1], edge.edge_id)
        return suffix if substage is None else (substage, *suffix)

    def _stage1(
        self,
        active: set[int],
        *,
        boundary_active: bool,
        neighbors: dict[int, frozenset[int]],
    ) -> _Candidate | None:
        candidates: list[_Candidate] = []
        for endpoints, edge in self._direct_candidates(
            active, boundary_active=boundary_active
        ):
            a, b = endpoints
            if neighbors[a] == frozenset((b,)) and neighbors[b] == frozenset((a,)):
                candidates.append(
                    _Candidate(1, endpoints, (edge,), self._direct_key(endpoints, edge))
                )
        return min(candidates, key=lambda c: c.key, default=None)

    def _adjacent_stage(
        self,
        active: set[int],
        *,
        boundary_active: bool,
        neighbors: dict[int, frozenset[int]],
        safe: bool,
    ) -> _Candidate | None:
        stage = 2 if safe else 4
        candidates: list[_Candidate] = []
        for endpoints, edge in self._direct_candidates(
            active, boundary_active=boundary_active
        ):
            creates = self._creates_new_singleton(endpoints, neighbors)
            if safe != (not creates):
                continue
            substage = 0 if min(len(neighbors[v]) for v in endpoints) == 1 else 1
            candidates.append(
                _Candidate(
                    stage,
                    endpoints,
                    (edge,),
                    self._direct_key(endpoints, edge, substage=substage),
                )
            )
        return min(candidates, key=lambda c: c.key, default=None)

    def _traversal_adjacency(self) -> dict[int, tuple[tuple[int, Edge], ...]]:
        adjacency: dict[int, list[tuple[int, Edge]]] = {
            v: [] for v in self.vertices
        }
        for edge in self.edges:
            if edge.target is None or not self._edge_allowed_in_path(edge):
                continue
            adjacency[edge.source].append((edge.target, edge))
            adjacency[edge.target].append((edge.source, edge))
        for items in adjacency.values():
            items.sort(key=lambda item: (item[1].weight, item[0], item[1].edge_id))
        return {v: tuple(items) for v, items in adjacency.items()}

    def _shortest_path(self, source: int, target: int) -> tuple[Edge, ...] | None:
        """Returns the deterministic shortest eligible simple path."""

        adjacency = self._traversal_adjacency()
        zero_mask = bytes(self.mask_bytes)

        # Whole-path XOR-zero is an explicitly slower ablation.  Keeping the
        # visited set in the label is necessary: pruning solely by (node, mask)
        # can discard the only prefix that can still complete a simple path.
        if self.observable_policy == "path-zero":
            heap: list[tuple[float, int, tuple[int, ...], int, bytes, tuple[int, ...]]] = [
                (0.0, 0, (), source, zero_mask, (source,))
            ]
            while heap:
                weight, count, edge_ids, node, mask, vertices = heapq.heappop(heap)
                if node == target and mask == zero_mask:
                    return tuple(self.edge_by_id[edge_id] for edge_id in edge_ids)
                visited = frozenset(vertices)
                for other, edge in adjacency[node]:
                    if other in visited:
                        continue
                    heapq.heappush(
                        heap,
                        (
                            weight + edge.weight,
                            count + 1,
                            (*edge_ids, edge.edge_id),
                            other,
                            _xor_mask(mask, edge.observable_mask),
                            (*vertices, other),
                        ),
                    )
            return None

        heap2: list[tuple[float, int, tuple[int, ...], int, tuple[int, ...]]] = [
            (0.0, 0, (), source, (source,))
        ]
        finalized: set[int] = set()
        while heap2:
            weight, count, edge_ids, node, vertices = heapq.heappop(heap2)
            if node in finalized:
                continue
            finalized.add(node)
            if node == target:
                return tuple(self.edge_by_id[edge_id] for edge_id in edge_ids)
            visited = frozenset(vertices)
            for other, edge in adjacency[node]:
                if other in finalized or other in visited:
                    continue
                heapq.heappush(
                    heap2,
                    (
                        weight + edge.weight,
                        count + 1,
                        (*edge_ids, edge.edge_id),
                        other,
                        (*vertices, other),
                    ),
                )
        return None

    def _stage3(
        self,
        active: set[int],
        *,
        neighbors: dict[int, frozenset[int]],
    ) -> tuple[_Candidate | None, bool]:
        # The parity boundary is deliberately excluded from stage 3.
        singletons = sorted(v for v in active if not neighbors[v])
        candidates: list[_Candidate] = []
        saw_unreachable = False
        for singleton in singletons:
            for other in sorted(active):
                if other == singleton:
                    continue
                if self._creates_new_singleton((singleton, other), neighbors):
                    continue
                path = self._shortest_path(singleton, other)
                if path is None:
                    saw_unreachable = True
                    continue
                mask = _path_mask(path, mask_bytes=self.mask_bytes)
                if self.observable_policy in {"edge-zero", "path-zero"} and not _is_zero_mask(mask):
                    continue
                edge_ids = tuple(edge.edge_id for edge in path)
                weight = math.fsum(edge.weight for edge in path)
                key = (weight, singleton, other, len(path), edge_ids)
                candidates.append(
                    _Candidate(3, (singleton, other), path, key)
                )
        return min(candidates, key=lambda c: c.key, default=None), saw_unreachable

    def _candidate(
        self, active: set[int], *, boundary_active: bool
    ) -> tuple[_Candidate | None, bool]:
        neighbors = self._active_neighbors(active, boundary_active=boundary_active)
        candidate = self._stage1(
            active, boundary_active=boundary_active, neighbors=neighbors
        )
        if candidate is not None:
            return candidate, False
        candidate = self._adjacent_stage(
            active,
            boundary_active=boundary_active,
            neighbors=neighbors,
            safe=True,
        )
        if candidate is not None:
            return candidate, False
        candidate, saw_unreachable = self._stage3(active, neighbors=neighbors)
        if candidate is not None:
            return candidate, saw_unreachable
        candidate = self._adjacent_stage(
            active,
            boundary_active=boundary_active,
            neighbors=neighbors,
            safe=False,
        )
        return candidate, saw_unreachable

    def run(self, initial_active: set[int]) -> _DomainAttempt:
        active = set(initial_active)
        boundary_was_added = (
            self.boundary_policy == "odd-parity" and len(active) % 2 == 1
        )
        boundary_active = boundary_was_added
        boundary_was_used = False
        paths: list[PrematchedPath] = []
        stage_counts = [0, 0, 0, 0]
        saw_unreachable = False

        while len(active) > self.hw_limit:
            candidate, unreachable = self._candidate(
                active, boundary_active=boundary_active
            )
            saw_unreachable |= unreachable
            if candidate is None:
                if boundary_active and not any(
                    edge.source in active and self._edge_allowed_directly(edge)
                    for edge in self.boundary_edges
                ):
                    reason = FallbackReason.BOUNDARY_UNAVAILABLE
                elif saw_unreachable:
                    reason = FallbackReason.DISCONNECTED
                else:
                    reason = FallbackReason.NO_CANDIDATE
                return _DomainAttempt(
                    active=frozenset(active),
                    paths=tuple(paths),
                    stage_counts=tuple(stage_counts),  # type: ignore[arg-type]
                    fallback_reason=reason,
                    boundary_was_added=boundary_was_added,
                    boundary_was_used=boundary_was_used,
                )

            path_mask = _path_mask(candidate.edges, mask_bytes=self.mask_bytes)
            public_endpoints = _public_endpoints(candidate.endpoints)
            paths.append(
                PrematchedPath(
                    domain=self.graph.domain,
                    stage=candidate.stage,
                    endpoints=public_endpoints,
                    edge_ids=tuple(edge.edge_id for edge in candidate.edges),
                    decision_weight=math.fsum(edge.weight for edge in candidate.edges),
                    observable_mask=path_mask,
                )
            )
            stage_counts[candidate.stage - 1] += 1
            for endpoint in candidate.endpoints:
                if endpoint == _BOUNDARY:
                    if not boundary_active:
                        raise AssertionError("inactive virtual boundary was selected")
                    boundary_active = False
                    boundary_was_used = True
                else:
                    if endpoint not in active:
                        raise AssertionError("inactive detector endpoint was selected")
                    active.remove(endpoint)

        return _DomainAttempt(
            active=frozenset(active),
            paths=tuple(paths),
            stage_counts=tuple(stage_counts),  # type: ignore[arg-type]
            fallback_reason=None,
            boundary_was_added=boundary_was_added,
            boundary_was_used=boundary_was_used,
        )


def _validate_syndrome(
    syndrome: np.ndarray, *, num_detectors: int
) -> np.ndarray:
    array = np.asarray(syndrome)
    if array.ndim != 1 or array.shape != (num_detectors,):
        raise ValueError(
            f"syndrome must have shape ({num_detectors},), got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.bool_) and not np.issubdtype(
        array.dtype, np.integer
    ):
        raise TypeError("syndrome must contain boolean or integer bits")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("syndrome entries must be binary")
    return np.asarray(array, dtype=np.uint8).copy()


def _assert_domain_algebra(
    *,
    original: np.ndarray,
    residual: np.ndarray,
    paths: Iterable[PrematchedPath],
    edge_by_id: dict[int, Edge],
    detector_ids: frozenset[int],
) -> None:
    boundary = np.zeros_like(original)
    for path in paths:
        for edge_id in path.edge_ids:
            edge = edge_by_id[edge_id]
            boundary[edge.source] ^= 1
            if edge.target is not None:
                boundary[edge.target] ^= 1
    changed = original ^ residual
    domain_indices = np.asarray(sorted(detector_ids), dtype=np.int64)
    if not np.array_equal(changed[domain_indices], boundary[domain_indices]):
        raise AssertionError("prematch paths violate the domain syndrome-boundary invariant")
    outside = np.ones(original.shape[0], dtype=np.bool_)
    outside[domain_indices] = False
    if np.any(boundary[outside]):
        raise AssertionError("a domain prematch path has a cross-domain boundary")


def predecode(
    compiled_graph: CompiledPromatchGraph,
    syndrome: np.ndarray,
    *,
    residual_hw_limit: int = 10,
    boundary_policy: BoundaryPolicy = "disabled",
    observable_policy: ObservablePolicy = "edge-zero",
) -> PrematchResult:
    """Predecodes one unpacked detector shot transactionally by L1 domain."""

    if isinstance(residual_hw_limit, bool) or not isinstance(
        residual_hw_limit, (int, np.integer)
    ):
        raise TypeError("residual_hw_limit must be an integer")
    residual_hw_limit = int(residual_hw_limit)
    if residual_hw_limit < 0:
        raise ValueError("residual_hw_limit must be nonnegative")

    policy = _normalized_observable_policy(observable_policy)
    original = _validate_syndrome(
        syndrome, num_detectors=compiled_graph.num_detectors
    )
    residual = original.copy()
    mask_bytes = (compiled_graph.num_observables + 7) // 8
    frame = bytes(mask_bytes)
    committed_paths: list[PrematchedPath] = []
    domain_stats: dict[L1DomainKey, DomainPrematchStats] = {}
    assigned: set[int] = set()

    edge_by_id = {edge.edge_id: edge for edge in compiled_graph.edges}
    if len(edge_by_id) != len(compiled_graph.edges):
        raise ValueError("compiled graph contains duplicate edge IDs")

    for domain in sorted(compiled_graph.domain_graphs):
        domain_graph = compiled_graph.domain_graphs[domain]
        detector_ids = frozenset(int(v) for v in domain_graph.detector_ids)
        overlap = assigned.intersection(detector_ids)
        if overlap:
            raise ValueError(f"detectors belong to multiple predecode domains: {sorted(overlap)}")
        assigned.update(detector_ids)
        initial_active = {v for v in detector_ids if original[v]}
        initial_hw = len(initial_active)

        if initial_hw <= residual_hw_limit:
            domain_stats[domain] = DomainPrematchStats(
                initial_hw=initial_hw,
                attempted_residual_hw=initial_hw,
                final_residual_hw=initial_hw,
                attempted_stage_counts=(0, 0, 0, 0),
                committed_stage_counts=(0, 0, 0, 0),
                attempted_matches=0,
                committed_matches=0,
                status="below-limit",
                fallback_reason=None,
            )
            continue

        attempt = _DomainEngine(
            domain_graph,
            num_detectors=compiled_graph.num_detectors,
            num_observables=compiled_graph.num_observables,
            hw_limit=residual_hw_limit,
            boundary_policy=boundary_policy,
            observable_policy=policy,  # type: ignore[arg-type]
        ).run(initial_active)
        attempted_matches = len(attempt.paths)
        attempted_hw = len(attempt.active)

        if not attempt.success:
            domain_stats[domain] = DomainPrematchStats(
                initial_hw=initial_hw,
                attempted_residual_hw=attempted_hw,
                final_residual_hw=initial_hw,
                attempted_stage_counts=attempt.stage_counts,
                committed_stage_counts=(0, 0, 0, 0),
                attempted_matches=attempted_matches,
                committed_matches=0,
                status="rollback",
                fallback_reason=attempt.fallback_reason,
                boundary_was_added=attempt.boundary_was_added,
                boundary_was_used=attempt.boundary_was_used,
                boundary_discarded_unused=False,
            )
            continue

        if policy == "any":
            for path in attempt.paths:
                _validate_path_observable_ownership(
                    path,
                    compiled_graph=compiled_graph,
                )
        before_commit = residual.copy()
        for detector_id in detector_ids:
            residual[detector_id] = int(detector_id in attempt.active)
        _assert_domain_algebra(
            original=before_commit,
            residual=residual,
            paths=attempt.paths,
            edge_by_id=edge_by_id,
            detector_ids=detector_ids,
        )
        for path in attempt.paths:
            frame = _xor_mask(frame, path.observable_mask)
        committed_paths.extend(attempt.paths)
        domain_stats[domain] = DomainPrematchStats(
            initial_hw=initial_hw,
            attempted_residual_hw=attempted_hw,
            final_residual_hw=attempted_hw,
            attempted_stage_counts=attempt.stage_counts,
            committed_stage_counts=attempt.stage_counts,
            attempted_matches=attempted_matches,
            committed_matches=attempted_matches,
            status="success",
            fallback_reason=None,
            boundary_was_added=attempt.boundary_was_added,
            boundary_was_used=attempt.boundary_was_used,
            boundary_discarded_unused=(
                attempt.boundary_was_added and not attempt.boundary_was_used
            ),
        )

    edge_parity: dict[int, int] = defaultdict(int)
    decision_weight = 0.0
    for path in committed_paths:
        decision_weight += path.decision_weight
        for edge_id in path.edge_ids:
            edge_parity[edge_id] ^= 1
    xor_support_weight = math.fsum(
        edge_by_id[edge_id].weight
        for edge_id, parity in edge_parity.items()
        if parity
    )

    observable_frame = np.unpackbits(
        np.frombuffer(frame, dtype=np.uint8),
        bitorder="little",
        count=compiled_graph.num_observables,
    ).astype(np.uint8, copy=False)
    if policy in {"edge-zero", "path-zero"} and np.any(observable_frame):
        raise AssertionError("zero-frame observable policy produced a nonzero frame")

    return PrematchResult(
        residual_syndrome=residual,
        observable_frame=observable_frame,
        paths=tuple(committed_paths),
        domain_stats=domain_stats,
        decision_weight=decision_weight,
        xor_support_weight=xor_support_weight,
    )


__all__ = [
    "BoundaryPolicy",
    "DomainPrematchStats",
    "FallbackReason",
    "ObservablePolicy",
    "PrematchResult",
    "PrematchedPath",
    "predecode",
]
