"""Exact semantic reference for confidence-gated patch-local weighted UF.

This module deliberately has no dependency on the YSC graph compiler.  The
small immutable graph below is the adapter boundary: a future projection only
needs to translate its lane-local correction, true-boundary, and guard-port
incidences into :class:`UFLaneGraph` once at compile time.

The reference uses :class:`fractions.Fraction` for every weight, charge, event
time, slack, and threshold comparison.  It processes a complete equal-time
event set atomically and retains at most one (the canonical-minimum) saturated
true-boundary incidence in each component forest.  Other saturated boundary
incidences remain zero-slack confidence competitors.

Guard ports are walls, not taints.  A component that saturates a port stops
growing exactly as one that reaches a true boundary does, the contact is
inherited through later unions, and the component is a deferred final
component with no peeled support: everything it contains is handed to the
global residual decoder, which owns every edge leaving the lane.
"""

from __future__ import annotations

import copy
import dataclasses
import heapq
import math
import numbers
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, Literal, Protocol, TypeAlias


UFEdgeKind: TypeAlias = Literal["correction", "boundary", "port"]
GateDecision: TypeAlias = Literal["eligible", "deferred"]
LaneStatus: TypeAlias = Literal["empty", "completed", "censored"]


def as_fraction(value: object, *, name: str = "exact value") -> Fraction:
    """Converts an integer, finite float, Fraction, or exact adapter value."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool")
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value)
    method = getattr(value, "as_fraction", None)
    if method is not None:
        result = method()
        if not isinstance(result, Fraction):
            raise TypeError(f"{name}.as_fraction() must return Fraction")
        return result
    raise TypeError(f"{name} must be an exact numeric value, got {type(value)!r}")


@dataclasses.dataclass(frozen=True)
class UFEdge:
    """One lane-local correction, true-boundary, or guard-port incidence."""

    edge_id: int
    source: int
    target: int | None
    weight: object
    kind: UFEdgeKind
    observable_mask: bytes = b""
    port_kind: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.edge_id, bool) or not isinstance(
            self.edge_id, numbers.Integral
        ):
            raise TypeError("edge_id must be an integer")
        if int(self.edge_id) < 0:
            raise ValueError("edge_id must be nonnegative")
        if isinstance(self.source, bool) or not isinstance(
            self.source, numbers.Integral
        ):
            raise TypeError("source must be an integer")
        if self.kind not in ("correction", "boundary", "port"):
            raise ValueError(f"unknown UF edge kind {self.kind!r}")
        if self.kind == "correction":
            if isinstance(self.target, bool) or not isinstance(
                self.target, numbers.Integral
            ):
                raise ValueError("a correction edge requires an integer target")
            if int(self.target) == int(self.source):
                raise ValueError("self-loop corrections are unsupported")
            if self.port_kind is not None:
                raise ValueError("a correction edge cannot have port_kind")
        elif self.target is not None:
            raise ValueError("boundary and port incidences must be one-ended")
        if self.kind == "port":
            if not isinstance(self.port_kind, str) or not self.port_kind:
                raise ValueError("a port requires a nonempty port_kind")
        elif self.port_kind is not None:
            raise ValueError("only ports may have port_kind")
        if not isinstance(self.observable_mask, bytes):
            raise TypeError("observable_mask must be bytes")
        exact_weight = as_fraction(self.weight, name="edge weight")
        if exact_weight <= 0:
            raise ValueError("UF edge weights must be strictly positive")


class LaneGraphProtocol(Protocol):
    """Minimal structural protocol accepted by the UF core."""

    @property
    def num_vertices(self) -> int: ...

    @property
    def edges(self) -> Sequence[UFEdge]: ...


@dataclasses.dataclass(frozen=True)
class UFLaneGraph:
    """Immutable generic lane graph, independent of projection implementation."""

    num_vertices: int
    edges: tuple[UFEdge, ...]

    def __post_init__(self) -> None:
        if isinstance(self.num_vertices, bool) or not isinstance(
            self.num_vertices, numbers.Integral
        ):
            raise TypeError("num_vertices must be an integer")
        if int(self.num_vertices) < 0:
            raise ValueError("num_vertices must be nonnegative")
        normalized = tuple(self.edges)
        if normalized != self.edges:
            object.__setattr__(self, "edges", normalized)
        ids: set[int] = set()
        for edge in self.edges:
            if not isinstance(edge, UFEdge):
                raise TypeError("edges must contain UFEdge values")
            if edge.edge_id in ids:
                raise ValueError(f"duplicate edge_id {edge.edge_id}")
            ids.add(edge.edge_id)
            if not 0 <= edge.source < self.num_vertices:
                raise ValueError(f"edge {edge.edge_id} source is out of range")
            if edge.target is not None and not 0 <= edge.target < self.num_vertices:
                raise ValueError(f"edge {edge.edge_id} target is out of range")

    @classmethod
    def from_protocol(cls, graph: LaneGraphProtocol) -> "UFLaneGraph":
        """Copies any object satisfying :class:`LaneGraphProtocol`."""

        if isinstance(graph, cls):
            return graph
        return cls(num_vertices=int(graph.num_vertices), edges=tuple(graph.edges))


@dataclasses.dataclass(frozen=True)
class BudgetLimits:
    """Literal caps; ``None`` means the counter belongs to the other cap class."""

    growth_event_count: int | None
    simultaneous_event_batch_count: int | None
    union_attempt_count: int | None
    successful_union_count: int | None
    forest_edge_count: int | None
    absorbed_vertex_count: int | None
    peel_operation_count: int | None
    heap_push_count: int | None
    heap_pop_count: int | None
    heap_operation_count: int | None
    peak_heap_size: int | None
    temporary_memory_units: int | None

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) < 0
            ):
                raise ValueError(f"budget {field.name} must be null or nonnegative")

    @classmethod
    def unbounded_for_testing(cls) -> "BudgetLimits":
        """Returns an explicitly test-only unbounded limit set."""

        return cls(**{field.name: None for field in dataclasses.fields(cls)})


@dataclasses.dataclass(frozen=True)
class UFPolicy:
    """Complete generic UF policy; all values are explicit constructor inputs."""

    tau: object
    semantic_limits: BudgetLimits
    production_limits: BudgetLimits

    def __post_init__(self) -> None:
        tau = as_fraction(self.tau, name="tau")
        if tau < 0:
            raise ValueError("tau must be nonnegative")
        if not isinstance(self.semantic_limits, BudgetLimits):
            raise TypeError("semantic_limits must be BudgetLimits")
        if not isinstance(self.production_limits, BudgetLimits):
            raise TypeError("production_limits must be BudgetLimits")


@dataclasses.dataclass(frozen=True, order=True)
class BudgetExceeded:
    cap_name: str
    limit: int
    rejected_next_value: int


@dataclasses.dataclass(frozen=True)
class UFCounters:
    growth_event_count: int = 0
    simultaneous_event_batch_count: int = 0
    union_attempt_count: int = 0
    successful_union_count: int = 0
    failed_union_count: int = 0
    forest_edge_count: int = 0
    peel_operation_count: int = 0
    heap_push_count: int = 0
    heap_pop_count: int = 0
    stale_heap_pop_count: int = 0
    heap_operation_count: int = 0
    peak_heap_size: int = 0
    peak_live_component_count: int = 0
    temporary_memory_units: int = 0


@dataclasses.dataclass(frozen=True)
class CompletedComponent:
    component_index: int
    absorbed_vertices: tuple[int, ...]
    original_defects: tuple[int, ...]
    forest_edge_ids: tuple[int, ...]
    peeled_support_edge_ids: tuple[int, ...]
    exact_margin: object | None
    gate_decision: GateDecision
    gate_reason_set: tuple[str, ...]
    primary_gate_reason: str
    boundary_reached: bool
    port_tainted: bool
    port_kind_set: tuple[str, ...]
    saturated_port_count: int
    merge_count: int
    simultaneous_event_batch_count: int
    event_batch_ids: tuple[int, ...]
    event_batch_times: tuple[object, ...]
    last_membership_event_time: object
    maximum_incident_half_edge_charge: object

    @property
    def cluster_defect_count(self) -> int:
        return len(self.original_defects)

    @property
    def absorbed_vertex_count(self) -> int:
        return len(self.absorbed_vertices)


@dataclasses.dataclass(frozen=True)
class CensoredComponent:
    component_index: int
    absorbed_vertices: tuple[int, ...]
    current_defects: tuple[int, ...]
    forest_edge_ids: tuple[int, ...]
    boundary_reached: bool
    port_tainted: bool
    port_kind_set: tuple[str, ...]
    saturated_port_count: int
    merge_count: int
    simultaneous_event_batch_count: int
    event_batch_ids: tuple[int, ...]
    event_batch_times: tuple[object, ...]
    last_membership_event_time: object
    maximum_incident_half_edge_charge: object

    @property
    def partial_cluster_defect_lower_bound(self) -> int:
        return len(self.current_defects)


@dataclasses.dataclass(frozen=True)
class LaneOutcome:
    status: LaneStatus
    completed_components: tuple[CompletedComponent, ...]
    censored_components: tuple[CensoredComponent, ...]
    counters: UFCounters
    censor_reason: str | None
    budget_exceeded_set: tuple[BudgetExceeded, ...]
    primary_budget_cap: str | None
    terminal_event_time: object
    last_complete_batch_id: int | None


class _ExactOps(Protocol):
    def convert(self, value: object) -> Any: ...

    def zero(self) -> Any: ...

    def divide_int(self, value: Any, divisor: int) -> Any: ...


class _FractionOps:
    def convert(self, value: object) -> Fraction:
        return as_fraction(value)

    def zero(self) -> Fraction:
        return Fraction(0)

    def divide_int(self, value: Fraction, divisor: int) -> Fraction:
        return value / divisor


@dataclasses.dataclass
class _ComponentState:
    vertices: set[int]
    defects: set[int]
    correction_forest: set[int]
    boundary_forest_edge: int | None
    saturated_boundary_edges: set[int]
    port_kinds: set[str]
    saturated_port_edges: set[int]
    merge_count: int
    batch_ids: set[int]
    batch_times: dict[int, object]
    last_membership_event_time: object

    @property
    def boundary_reached(self) -> bool:
        return self.boundary_forest_edge is not None

    @property
    def port_reached(self) -> bool:
        return bool(self.saturated_port_edges)

    @property
    def active(self) -> bool:
        return (
            bool(len(self.defects) & 1)
            and not self.boundary_reached
            and not self.port_reached
        )

    @property
    def forest_edges(self) -> set[int]:
        result = set(self.correction_forest)
        if self.boundary_forest_edge is not None:
            result.add(self.boundary_forest_edge)
        return result


class _DSU:
    def __init__(self, num_vertices: int, defects: set[int], zero: object) -> None:
        self.parent = list(range(num_vertices))
        self.size = [1] * num_vertices
        self.state = [
            _ComponentState(
                vertices={v},
                defects={v} if v in defects else set(),
                correction_forest=set(),
                boundary_forest_edge=None,
                saturated_boundary_edges=set(),
                port_kinds=set(),
                saturated_port_edges=set(),
                merge_count=0,
                batch_ids=set(),
                batch_times={},
                last_membership_event_time=zero,
            )
            for v in range(num_vertices)
        ]

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(
        self,
        a: int,
        b: int,
        edge_id: int,
        batch_id: int,
        event_time: object,
    ) -> tuple[int, bool]:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra, False
        if self.size[ra] < self.size[rb] or (
            self.size[ra] == self.size[rb] and rb < ra
        ):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        left = self.state[ra]
        right = self.state[rb]
        boundary_candidates = [
            value
            for value in (left.boundary_forest_edge, right.boundary_forest_edge)
            if value is not None
        ]
        left.vertices |= right.vertices
        left.defects ^= right.defects
        left.correction_forest |= right.correction_forest
        left.correction_forest.add(edge_id)
        left.saturated_boundary_edges |= right.saturated_boundary_edges
        left.boundary_forest_edge = (
            min(boundary_candidates) if boundary_candidates else None
        )
        left.port_kinds |= right.port_kinds
        left.saturated_port_edges |= right.saturated_port_edges
        left.merge_count += right.merge_count + 1
        left.batch_ids |= right.batch_ids
        left.batch_ids.add(batch_id)
        left.batch_times.update(right.batch_times)
        left.batch_times[batch_id] = event_time
        left.last_membership_event_time = event_time
        return ra, True

    def roots(self) -> list[int]:
        return [v for v in range(len(self.parent)) if self.find(v) == v]


@dataclasses.dataclass(frozen=True)
class _Scheduled:
    edge_index: int
    delay: object


@dataclasses.dataclass(frozen=True)
class _ScheduleCost:
    pushes: int = 0
    pops: int = 0
    stale_pops: int = 0
    peak_size: int = 0
    temporary_units: int = 0


def _validate_defects(graph: UFLaneGraph, defects: Iterable[int]) -> set[int]:
    result: set[int] = set()
    for raw in defects:
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
            raise TypeError("defect IDs must be integers")
        value = int(raw)
        if not 0 <= value < graph.num_vertices:
            raise ValueError(f"defect vertex {value} is out of range")
        if value in result:
            raise ValueError(f"duplicate defect vertex {value}")
        result.add(value)
    return result


def _edge_order(edge: UFEdge, weight: object) -> tuple[object, ...]:
    target = edge.source if edge.target is None else edge.target
    lo, hi = sorted((edge.source, target))
    return weight, lo, hi, edge.edge_id


def _candidate_delays(
    *,
    graph: UFLaneGraph,
    dsu: _DSU,
    weights: list[object],
    charge_source: list[object],
    charge_target: list[object],
    consumed: list[bool],
    adjacency: tuple[tuple[int, ...], ...],
    active_roots: set[int],
    ops: _ExactOps,
) -> list[_Scheduled]:
    result: list[_Scheduled] = []
    candidate_indices: set[int] = set()
    for root in active_roots:
        for vertex in dsu.state[root].vertices:
            candidate_indices.update(adjacency[vertex])
    for k in sorted(candidate_indices):
        edge = graph.edges[k]
        if consumed[k]:
            continue
        source_root = dsu.find(edge.source)
        source_active = source_root in active_roots
        if edge.kind == "correction":
            assert edge.target is not None
            target_root = dsu.find(edge.target)
            if source_root == target_root:
                continue
            rate = int(source_active) + int(target_root in active_roots)
            if rate == 0:
                continue
            slack = weights[k] - charge_source[k] - charge_target[k]
        else:
            if not source_active:
                continue
            rate = 1
            slack = weights[k] - charge_source[k]
        if slack < ops.zero():
            raise AssertionError(f"negative scheduling slack on edge {edge.edge_id}")
        result.append(_Scheduled(k, ops.divide_int(slack, rate)))
    return result


def _select_events(
    candidates: list[_Scheduled],
    *,
    graph: UFLaneGraph,
    weights: list[object],
    production: bool,
) -> tuple[object, list[int], _ScheduleCost]:
    if not candidates:
        raise ValueError("cannot select from an empty event set")
    if not production:
        minimum = min(item.delay for item in candidates)
        selected = [item.edge_index for item in candidates if item.delay == minimum]
        return minimum, selected, _ScheduleCost()

    heap: list[tuple[object, tuple[object, ...], int]] = []
    for item in candidates:
        edge = graph.edges[item.edge_index]
        heapq.heappush(
            heap,
            (item.delay, _edge_order(edge, weights[item.edge_index]), item.edge_index),
        )
    peak = len(heap)
    minimum = heap[0][0]
    selected: list[int] = []
    while heap and heap[0][0] == minimum:
        selected.append(heapq.heappop(heap)[2])
    return minimum, selected, _ScheduleCost(
        pushes=len(candidates),
        pops=len(selected),
        peak_size=peak,
        temporary_units=len(candidates),
    )


def _active_roots(dsu: _DSU) -> set[int]:
    return {root for root in dsu.roots() if dsu.state[root].active}


def _advance_charges(
    *,
    graph: UFLaneGraph,
    dsu: _DSU,
    delay: object,
    charge_source: list[object],
    charge_target: list[object],
    consumed: list[bool],
    adjacency: tuple[tuple[int, ...], ...],
    active_roots: set[int],
) -> None:
    candidate_indices: set[int] = set()
    for root in active_roots:
        for vertex in dsu.state[root].vertices:
            candidate_indices.update(adjacency[vertex])
    for k in candidate_indices:
        edge = graph.edges[k]
        if consumed[k]:
            continue
        source_root = dsu.find(edge.source)
        if edge.kind == "correction":
            assert edge.target is not None
            target_root = dsu.find(edge.target)
            if source_root == target_root:
                continue
            if source_root in active_roots:
                charge_source[k] = charge_source[k] + delay
            if target_root in active_roots:
                charge_target[k] = charge_target[k] + delay
        elif source_root in active_roots:
            charge_source[k] = charge_source[k] + delay


def _preview_batch(
    dsu: _DSU,
    correction_edges: list[UFEdge],
    boundary_edges: list[UFEdge],
    current_forest_count: int,
) -> tuple[int, int, int, int]:
    touched_roots = {
        dsu.find(vertex)
        for edge in correction_edges
        for vertex in (edge.source, int(edge.target))
    }
    touched_roots.update(dsu.find(edge.source) for edge in boundary_edges)
    parent = {root: root for root in touched_roots}
    sizes = {root: len(dsu.state[root].vertices) for root in touched_roots}
    boundary_forest = {
        root: dsu.state[root].boundary_forest_edge for root in touched_roots
    }

    def find(value: int) -> int:
        while parent.get(value, value) != value:
            value = parent[value]
        return value

    successes = 0
    largest = max(sizes.values(), default=0)
    for edge in correction_edges:
        assert edge.target is not None
        a = find(dsu.find(edge.source))
        b = find(dsu.find(edge.target))
        if a == b:
            continue
        if sizes[a] < sizes[b] or (sizes[a] == sizes[b] and b < a):
            a, b = b, a
        parent[b] = a
        sizes[a] += sizes[b]
        candidates = [
            value
            for value in (boundary_forest[a], boundary_forest[b])
            if value is not None
        ]
        boundary_forest[a] = min(candidates) if candidates else None
        largest = max(largest, sizes[a])
        successes += 1
    for edge in boundary_edges:
        root = find(dsu.find(edge.source))
        current = boundary_forest[root]
        boundary_forest[root] = edge.edge_id if current is None else min(
            current, edge.edge_id
        )
    roots = {find(root) for root in touched_roots}
    prior_boundary_count = sum(
        dsu.state[root].boundary_forest_edge is not None for root in touched_roots
    )
    next_boundary_count = sum(boundary_forest[root] is not None for root in roots)
    # Equivalent to current + successful correction forests, minus boundary
    # demotions caused by merges, plus newly selected boundary incidences.
    forest_count = (
        current_forest_count
        + successes
        + next_boundary_count
        - prior_boundary_count
    )
    return len(correction_edges), successes, largest, forest_count


def _counter_dict(counters: UFCounters) -> dict[str, int]:
    return {field.name: int(getattr(counters, field.name)) for field in dataclasses.fields(counters)}


def _exceeded(
    limits: BudgetLimits,
    proposed: dict[str, int],
) -> list[BudgetExceeded]:
    result: list[BudgetExceeded] = []
    for field in dataclasses.fields(limits):
        limit = getattr(limits, field.name)
        if limit is not None and proposed.get(field.name, 0) > limit:
            result.append(
                BudgetExceeded(field.name, int(limit), proposed[field.name])
            )
    return sorted(result)


def _maximum_incident_charge(
    state: _ComponentState,
    *,
    graph: UFLaneGraph,
    charge_source: list[object],
    charge_target: list[object],
    zero: object,
    adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> object:
    values: list[object] = []
    if adjacency is None:
        incident_indices: Iterable[int] = range(len(graph.edges))
    else:
        incident_indices = {
            k for vertex in state.vertices for k in adjacency[vertex]
        }
    for k in incident_indices:
        edge = graph.edges[k]
        if edge.source in state.vertices:
            values.append(charge_source[k])
        if edge.target is not None and edge.target in state.vertices:
            values.append(charge_target[k])
    return max(values, default=zero)


def _snapshot_censored(
    dsu: _DSU,
    *,
    graph: UFLaneGraph,
    charge_source: list[object],
    charge_target: list[object],
    zero: object,
    adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[CensoredComponent, ...]:
    states = [dsu.state[root] for root in dsu.roots() if dsu.state[root].defects]
    states.sort(key=lambda state: tuple(sorted(state.vertices)))
    return tuple(
        CensoredComponent(
            component_index=index,
            absorbed_vertices=tuple(sorted(state.vertices)),
            current_defects=tuple(sorted(state.defects)),
            forest_edge_ids=tuple(sorted(state.forest_edges)),
            boundary_reached=state.boundary_reached,
            port_tainted=bool(state.saturated_port_edges),
            port_kind_set=tuple(sorted(state.port_kinds)),
            saturated_port_count=len(state.saturated_port_edges),
            merge_count=state.merge_count,
            simultaneous_event_batch_count=len(state.batch_ids),
            event_batch_ids=tuple(sorted(state.batch_ids)),
            event_batch_times=tuple(
                state.batch_times[batch_id] for batch_id in sorted(state.batch_ids)
            ),
            last_membership_event_time=state.last_membership_event_time,
            maximum_incident_half_edge_charge=_maximum_incident_charge(
                state,
                graph=graph,
                charge_source=charge_source,
                charge_target=charge_target,
                zero=zero,
                adjacency=adjacency,
            ),
        )
        for index, state in enumerate(states)
    )


def _censored_outcome(
    *,
    dsu: _DSU,
    counters: UFCounters,
    reason: str,
    exceeded: list[BudgetExceeded],
    event_time: object,
    graph: UFLaneGraph,
    charge_source: list[object],
    charge_target: list[object],
    zero: object,
    adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> LaneOutcome:
    return LaneOutcome(
        status="censored",
        completed_components=(),
        censored_components=_snapshot_censored(
            dsu,
            graph=graph,
            charge_source=charge_source,
            charge_target=charge_target,
            zero=zero,
            adjacency=adjacency,
        ),
        counters=counters,
        censor_reason=reason,
        budget_exceeded_set=tuple(exceeded),
        primary_budget_cap=exceeded[0].cap_name if exceeded else None,
        terminal_event_time=event_time,
        last_complete_batch_id=(
            counters.simultaneous_event_batch_count
            if counters.simultaneous_event_batch_count
            else None
        ),
    )


def _peel_component(
    state: _ComponentState,
    *,
    graph: UFLaneGraph,
    edge_by_id: dict[int, UFEdge],
) -> tuple[tuple[int, ...], int]:
    forest_ids = sorted(state.forest_edges)
    adjacency: dict[int, list[tuple[int, int]]] = {
        vertex: [] for vertex in state.vertices
    }
    virtual_root: int | None = None
    for edge_id in forest_ids:
        edge = edge_by_id[edge_id]
        if edge.kind == "correction":
            assert edge.target is not None
            adjacency[edge.source].append((edge.target, edge_id))
            adjacency[edge.target].append((edge.source, edge_id))
        elif edge.kind == "boundary":
            virtual = graph.num_vertices + edge_id + 1
            if virtual in adjacency:
                raise AssertionError("virtual boundary node collision")
            adjacency[virtual] = [(edge.source, edge_id)]
            adjacency[edge.source].append((virtual, edge_id))
            virtual_root = virtual
        else:
            raise AssertionError("a port entered the component forest")
    root = virtual_root if virtual_root is not None else min(state.vertices)
    parent: dict[int, tuple[int, int] | None] = {root: None}
    order: list[int] = [root]
    for node in order:
        for neighbor, edge_id in sorted(adjacency[node], key=lambda item: (item[1], item[0])):
            if neighbor in parent:
                continue
            parent[neighbor] = (node, edge_id)
            order.append(neighbor)
    if len(parent) != len(adjacency):
        raise AssertionError("component forest is disconnected")
    parity = {node: node in state.defects for node in adjacency}
    support: set[int] = set()
    operations = 0
    for node in reversed(order[1:]):
        operations += 1
        relation = parent[node]
        assert relation is not None
        parent_node, edge_id = relation
        if parity[node]:
            if edge_id in support:
                raise AssertionError("peeling emitted a duplicate edge")
            support.add(edge_id)
            parity[parent_node] = not parity[parent_node]
    if virtual_root is None and parity[root]:
        raise AssertionError("boundary-free component has odd peeling parity")
    for edge_id in support:
        edge = edge_by_id[edge_id]
        if edge.kind == "port" or any(edge.observable_mask):
            raise AssertionError("peeled support is not correction-eligible zero-frame")
    boundary: set[int] = set()
    for edge_id in support:
        edge = edge_by_id[edge_id]
        boundary.symmetric_difference_update((edge.source,))
        if edge.target is not None:
            boundary.symmetric_difference_update((edge.target,))
    if boundary != state.defects:
        raise AssertionError("peeled support has the wrong detector boundary")
    return tuple(sorted(support)), operations


_PRIMARY_REASON_ORDER = (
    "port-contact",
    "port-yoke",
    "port-cross-lane",
    "below-threshold",
)


def _finish_components(
    *,
    graph: UFLaneGraph,
    dsu: _DSU,
    weights: list[object],
    charge_source: list[object],
    charge_target: list[object],
    tau: object,
    counters: UFCounters,
    policy: UFPolicy,
    production: bool,
    adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[tuple[CompletedComponent, ...] | None, UFCounters, list[BudgetExceeded]]:
    edge_by_id = {edge.edge_id: edge for edge in graph.edges}
    states = [dsu.state[root] for root in dsu.roots() if dsu.state[root].defects]
    states.sort(key=lambda state: tuple(sorted(state.vertices)))
    # Port-contact components are deferred without peeling; their forest edges
    # are retained for telemetry only and cost no peel operations.
    total_peel = sum(
        len(state.forest_edges) for state in states if not state.port_reached
    )
    proposed = _counter_dict(counters)
    proposed["peel_operation_count"] = counters.peel_operation_count + total_peel
    exceeded = _exceeded(policy.semantic_limits, proposed)
    if production:
        exceeded += _exceeded(policy.production_limits, proposed)
        exceeded = sorted(set(exceeded))
    if exceeded:
        return None, counters, exceeded

    completed: list[CompletedComponent] = []
    actual_peel = 0
    for index, state in enumerate(states):
        if state.port_reached:
            support, operations = (), 0
        else:
            support, operations = _peel_component(
                state, graph=graph, edge_by_id=edge_by_id
            )
        actual_peel += operations
        forest = state.forest_edges
        competitors: list[object] = []
        if adjacency is None:
            incident_indices: Iterable[int] = range(len(graph.edges))
        else:
            incident_indices = {
                k for vertex in state.vertices for k in adjacency[vertex]
            }
        for k in incident_indices:
            edge = graph.edges[k]
            if edge.source not in state.vertices and (
                edge.target is None or edge.target not in state.vertices
            ):
                continue
            if edge.kind == "correction":
                if edge.edge_id in forest:
                    continue
                slack = weights[k] - charge_source[k] - charge_target[k]
            elif edge.kind == "boundary":
                if edge.edge_id in forest:
                    continue
                slack = weights[k] - charge_source[k]
            else:
                slack = weights[k] - charge_source[k]
            if slack < 0:
                raise AssertionError(f"negative confidence slack on edge {edge.edge_id}")
            competitors.append(slack)
        margin = min(competitors) if competitors else None
        reasons: set[str] = set()
        if state.port_reached:
            reasons.add("port-contact")
            for kind in state.port_kinds:
                reasons.add("port-yoke" if kind == "yoke" else "port-cross-lane")
        if margin is not None and margin <= tau:
            reasons.add("below-threshold")
        primary = "eligible"
        for candidate in _PRIMARY_REASON_ORDER:
            if candidate in reasons:
                primary = candidate
                break
        decision: GateDecision = "eligible" if not reasons else "deferred"
        completed.append(
            CompletedComponent(
                component_index=index,
                absorbed_vertices=tuple(sorted(state.vertices)),
                original_defects=tuple(sorted(state.defects)),
                forest_edge_ids=tuple(sorted(forest)),
                peeled_support_edge_ids=support,
                exact_margin=margin,
                gate_decision=decision,
                gate_reason_set=tuple(sorted(reasons)),
                primary_gate_reason=primary,
                boundary_reached=state.boundary_reached,
                port_tainted=bool(state.saturated_port_edges),
                port_kind_set=tuple(sorted(state.port_kinds)),
                saturated_port_count=len(state.saturated_port_edges),
                merge_count=state.merge_count,
                simultaneous_event_batch_count=len(state.batch_ids),
                event_batch_ids=tuple(sorted(state.batch_ids)),
                event_batch_times=tuple(
                    state.batch_times[batch_id]
                    for batch_id in sorted(state.batch_ids)
                ),
                last_membership_event_time=state.last_membership_event_time,
                maximum_incident_half_edge_charge=_maximum_incident_charge(
                    state,
                    graph=graph,
                    charge_source=charge_source,
                    charge_target=charge_target,
                    zero=weights[0] - weights[0] if weights else tau - tau,
                    adjacency=adjacency,
                ),
            )
        )
    counters = dataclasses.replace(counters, peel_operation_count=actual_peel)
    return tuple(completed), counters, []


def _run_lane_exact(
    graph_like: LaneGraphProtocol,
    defects: Iterable[int],
    policy: UFPolicy,
    *,
    ops: _ExactOps,
    production: bool,
) -> LaneOutcome:
    graph = UFLaneGraph.from_protocol(graph_like)
    if not isinstance(policy, UFPolicy):
        raise TypeError("policy must be UFPolicy")
    defect_set = _validate_defects(graph, defects)
    weights = [ops.convert(edge.weight) for edge in graph.edges]
    edge_index_by_id = {edge.edge_id: k for k, edge in enumerate(graph.edges)}
    adjacency_lists: list[list[int]] = [[] for _ in range(graph.num_vertices)]
    for k, edge in enumerate(graph.edges):
        adjacency_lists[edge.source].append(k)
        if edge.target is not None:
            adjacency_lists[edge.target].append(k)
    adjacency = tuple(tuple(values) for values in adjacency_lists)
    tau = ops.convert(policy.tau)
    zero = ops.zero()
    charges_source = [zero for _ in graph.edges]
    charges_target = [zero for _ in graph.edges]
    consumed = [False for _ in graph.edges]
    dsu = _DSU(graph.num_vertices, defect_set, zero)
    initial_live = sum(bool(dsu.state[root].defects) for root in dsu.roots())
    counters = UFCounters(peak_live_component_count=initial_live)
    event_time = zero
    if not defect_set:
        return LaneOutcome(
            "empty", (), (), counters, None, (), None, event_time, None
        )

    initial_proposed = _counter_dict(counters)
    initial_proposed["absorbed_vertex_count"] = 1
    initial_exceeded = _exceeded(policy.semantic_limits, initial_proposed)
    if production:
        initial_exceeded += _exceeded(policy.production_limits, initial_proposed)
        initial_exceeded = sorted(set(initial_exceeded))
    if initial_exceeded:
        return _censored_outcome(
            dsu=dsu,
            counters=counters,
            reason="budget-exhaustion",
            exceeded=initial_exceeded,
            event_time=event_time,
            graph=graph,
            charge_source=charges_source,
            charge_target=charges_target,
            zero=zero,
        )

    batch_id = 0
    while (active_roots := _active_roots(dsu)):
        candidates = _candidate_delays(
            graph=graph,
            dsu=dsu,
            weights=weights,
            charge_source=charges_source,
            charge_target=charges_target,
            consumed=consumed,
            adjacency=adjacency,
            active_roots=active_roots,
            ops=ops,
        )
        if not candidates:
            return _censored_outcome(
                dsu=dsu,
                counters=counters,
                reason="local-incomplete-neutralization",
                exceeded=[],
                event_time=event_time,
                graph=graph,
                charge_source=charges_source,
                charge_target=charges_target,
                zero=zero,
            )
        delay, selected_indices, schedule_cost = _select_events(
            candidates, graph=graph, weights=weights, production=production
        )
        correction_edges = sorted(
            (graph.edges[k] for k in selected_indices if graph.edges[k].kind == "correction"),
            key=lambda edge: _edge_order(edge, weights[edge_index_by_id[edge.edge_id]]),
        )
        boundary_edges = [
            graph.edges[k]
            for k in selected_indices
            if graph.edges[k].kind == "boundary"
        ]
        union_attempts, successes, largest, proposed_forest_count = _preview_batch(
            dsu, correction_edges, boundary_edges, counters.forest_edge_count
        )
        proposed = _counter_dict(counters)
        proposed.update(
            growth_event_count=counters.growth_event_count + len(selected_indices),
            simultaneous_event_batch_count=counters.simultaneous_event_batch_count + 1,
            union_attempt_count=counters.union_attempt_count + union_attempts,
            successful_union_count=counters.successful_union_count + successes,
            forest_edge_count=proposed_forest_count,
            absorbed_vertex_count=largest,
            heap_push_count=counters.heap_push_count + schedule_cost.pushes,
            heap_pop_count=counters.heap_pop_count + schedule_cost.pops,
            heap_operation_count=(
                counters.heap_operation_count + schedule_cost.pushes + schedule_cost.pops
            ),
            peak_heap_size=max(counters.peak_heap_size, schedule_cost.peak_size),
            temporary_memory_units=max(
                counters.temporary_memory_units, schedule_cost.temporary_units
            ),
        )
        exceeded = _exceeded(policy.semantic_limits, proposed)
        if production:
            exceeded += _exceeded(policy.production_limits, proposed)
            exceeded = sorted(set(exceeded))
        if exceeded:
            return _censored_outcome(
                dsu=dsu,
                counters=counters,
                reason="budget-exhaustion",
                exceeded=exceeded,
                event_time=event_time,
                graph=graph,
                charge_source=charges_source,
                charge_target=charges_target,
                zero=zero,
            )

        _advance_charges(
            graph=graph,
            dsu=dsu,
            delay=delay,
            charge_source=charges_source,
            charge_target=charges_target,
            consumed=consumed,
            adjacency=adjacency,
            active_roots=active_roots,
        )
        event_time = event_time + delay
        batch_id += 1
        successful = 0
        failed = 0
        for edge in correction_edges:
            root, did_union = dsu.union(
                edge.source,
                int(edge.target),
                edge.edge_id,
                batch_id,
                event_time,
            )
            if not did_union:
                dsu.state[root].batch_ids.add(batch_id)
                dsu.state[root].batch_times[batch_id] = event_time
            successful += int(did_union)
            failed += int(not did_union)
        for k in sorted(selected_indices, key=lambda i: graph.edges[i].edge_id):
            edge = graph.edges[k]
            consumed[k] = True
            if edge.kind == "boundary":
                root = dsu.find(edge.source)
                state = dsu.state[root]
                state.saturated_boundary_edges.add(edge.edge_id)
                if state.boundary_forest_edge is None:
                    state.boundary_forest_edge = edge.edge_id
                else:
                    state.boundary_forest_edge = min(
                        state.boundary_forest_edge, edge.edge_id
                    )
                state.batch_ids.add(batch_id)
                state.batch_times[batch_id] = event_time
            elif edge.kind == "port":
                root = dsu.find(edge.source)
                state = dsu.state[root]
                state.port_kinds.add(str(edge.port_kind))
                state.saturated_port_edges.add(edge.edge_id)
                state.batch_ids.add(batch_id)
                state.batch_times[batch_id] = event_time

        counters = UFCounters(
            growth_event_count=proposed["growth_event_count"],
            simultaneous_event_batch_count=proposed["simultaneous_event_batch_count"],
            union_attempt_count=proposed["union_attempt_count"],
            successful_union_count=counters.successful_union_count + successful,
            failed_union_count=counters.failed_union_count + failed,
            forest_edge_count=proposed_forest_count,
            peel_operation_count=counters.peel_operation_count,
            heap_push_count=proposed["heap_push_count"],
            heap_pop_count=proposed["heap_pop_count"],
            stale_heap_pop_count=counters.stale_heap_pop_count,
            heap_operation_count=proposed["heap_operation_count"],
            peak_heap_size=proposed["peak_heap_size"],
            # Components only merge; a nonempty original-defect set is never
            # split or canceled because component memberships are disjoint.
            peak_live_component_count=counters.peak_live_component_count,
            temporary_memory_units=proposed["temporary_memory_units"],
        )

    completed, counters, exceeded = _finish_components(
        graph=graph,
        dsu=dsu,
        weights=weights,
        charge_source=charges_source,
        charge_target=charges_target,
        tau=tau,
        counters=counters,
        policy=policy,
        production=production,
        adjacency=adjacency,
    )
    if completed is None:
        return _censored_outcome(
            dsu=dsu,
            counters=counters,
            reason="budget-exhaustion",
            exceeded=exceeded,
            event_time=event_time,
            graph=graph,
            charge_source=charges_source,
            charge_target=charges_target,
            zero=zero,
        )
    return LaneOutcome(
        status="completed",
        completed_components=completed,
        censored_components=(),
        counters=counters,
        censor_reason=None,
        budget_exceeded_set=(),
        primary_budget_cap=None,
        terminal_event_time=event_time,
        last_complete_batch_id=(
            counters.simultaneous_event_batch_count
            if counters.simultaneous_event_batch_count
            else None
        ),
    )


class _PersistentQueue:
    """Lazy exact event queue used by the production lifecycle."""

    def __init__(
        self,
        *,
        graph: UFLaneGraph,
        dsu: _DSU,
        weights: list[object],
        ops: _ExactOps,
        adjacency: tuple[tuple[int, ...], ...],
        charge_source: list[object],
        charge_target: list[object],
    ) -> None:
        self.graph = graph
        self.dsu = dsu
        self.weights = weights
        self.ops = ops
        self.adjacency = adjacency
        self.charge_source = charge_source
        self.charge_target = charge_target
        zero = ops.zero()
        self.last_source = [zero for _ in graph.edges]
        self.last_target = [zero for _ in graph.edges]
        self.consumed = [False for _ in graph.edges]
        self.version = [0 for _ in graph.edges]
        self.heap: list[tuple[object, tuple[object, ...], int, int]] = []

    def incident_indices(self, roots: Iterable[int]) -> set[int]:
        result: set[int] = set()
        for root in roots:
            for vertex in self.dsu.state[root].vertices:
                result.update(self.adjacency[vertex])
        return result

    def touch(self, k: int, now: object) -> None:
        edge = self.graph.edges[k]
        source_root = self.dsu.find(edge.source)
        source_delta = now - self.last_source[k]
        if source_delta < self.ops.zero():
            raise AssertionError("event queue moved backwards in time")
        if self.consumed[k]:
            self.last_source[k] = now
            self.last_target[k] = now
            return
        if edge.kind == "correction":
            assert edge.target is not None
            target_root = self.dsu.find(edge.target)
            if source_root != target_root:
                if self.dsu.state[source_root].active:
                    self.charge_source[k] = self.charge_source[k] + source_delta
                target_delta = now - self.last_target[k]
                if target_delta < self.ops.zero():
                    raise AssertionError("event queue moved backwards in time")
                if self.dsu.state[target_root].active:
                    self.charge_target[k] = self.charge_target[k] + target_delta
            self.last_target[k] = now
        elif self.dsu.state[source_root].active:
            self.charge_source[k] = self.charge_source[k] + source_delta
        self.last_source[k] = now

    def reschedule(self, k: int, now: object) -> bool:
        self.touch(k, now)
        self.version[k] += 1
        if self.consumed[k]:
            return False
        edge = self.graph.edges[k]
        source_root = self.dsu.find(edge.source)
        source_active = self.dsu.state[source_root].active
        if edge.kind == "correction":
            assert edge.target is not None
            target_root = self.dsu.find(edge.target)
            if source_root == target_root:
                return False
            rate = int(source_active) + int(self.dsu.state[target_root].active)
            if rate == 0:
                return False
            slack = self.weights[k] - self.charge_source[k] - self.charge_target[k]
        else:
            if not source_active:
                return False
            rate = 1
            slack = self.weights[k] - self.charge_source[k]
        if slack < self.ops.zero():
            raise AssertionError(f"negative scheduling slack on edge {edge.edge_id}")
        event_time = now + self.ops.divide_int(slack, rate)
        heapq.heappush(
            self.heap,
            (event_time, _edge_order(edge, self.weights[k]), k, self.version[k]),
        )
        return True

    def reschedule_many(self, indices: Iterable[int], now: object) -> _ScheduleCost:
        before = len(self.heap)
        pushes = sum(self.reschedule(k, now) for k in sorted(set(indices)))
        return _ScheduleCost(
            pushes=pushes,
            peak_size=max(before, len(self.heap)),
            temporary_units=len(self.heap),
        )

    def next_batch(
        self,
    ) -> tuple[object | None, list[int], _ScheduleCost]:
        pops = 0
        stale = 0
        first: tuple[object, tuple[object, ...], int, int] | None = None
        while self.heap:
            candidate = heapq.heappop(self.heap)
            pops += 1
            _, _, k, version = candidate
            if self.consumed[k] or version != self.version[k]:
                stale += 1
                continue
            first = candidate
            break
        if first is None:
            return None, [], _ScheduleCost(pops=pops, stale_pops=stale)
        event_time = first[0]
        selected = [first[2]]
        while self.heap and self.heap[0][0] == event_time:
            candidate = heapq.heappop(self.heap)
            pops += 1
            _, _, k, version = candidate
            if self.consumed[k] or version != self.version[k]:
                stale += 1
                continue
            selected.append(k)
        return event_time, selected, _ScheduleCost(pops=pops, stale_pops=stale)


def _run_lane_persistent(
    graph_like: LaneGraphProtocol,
    defects: Iterable[int],
    policy: UFPolicy,
    *,
    ops: _ExactOps,
    prepared_graph: UFLaneGraph | None = None,
    prepared_weights: tuple[object, ...] | None = None,
    prepared_tau: object | None = None,
    prepared_adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> LaneOutcome:
    """Runs the exact persistent-heap production policy."""

    graph = (
        prepared_graph
        if prepared_graph is not None
        else UFLaneGraph.from_protocol(graph_like)
    )
    if not isinstance(policy, UFPolicy):
        raise TypeError("policy must be UFPolicy")
    defect_set = _validate_defects(graph, defects)
    weights = (
        list(prepared_weights)
        if prepared_weights is not None
        else [ops.convert(edge.weight) for edge in graph.edges]
    )
    edge_index_by_id = {edge.edge_id: k for k, edge in enumerate(graph.edges)}
    tau = prepared_tau if prepared_tau is not None else ops.convert(policy.tau)
    zero = ops.zero()
    charge_source = [zero for _ in graph.edges]
    charge_target = [zero for _ in graph.edges]
    if prepared_adjacency is None:
        adjacency_lists: list[list[int]] = [[] for _ in range(graph.num_vertices)]
        for k, edge in enumerate(graph.edges):
            adjacency_lists[edge.source].append(k)
            if edge.target is not None:
                adjacency_lists[edge.target].append(k)
        adjacency = tuple(tuple(values) for values in adjacency_lists)
    else:
        adjacency = prepared_adjacency
    dsu = _DSU(graph.num_vertices, defect_set, zero)
    initial_live = sum(bool(dsu.state[root].defects) for root in dsu.roots())
    counters = UFCounters(peak_live_component_count=initial_live)
    event_time = zero
    if not defect_set:
        return LaneOutcome("empty", (), (), counters, None, (), None, zero, None)

    initial_proposed = _counter_dict(counters)
    initial_proposed["absorbed_vertex_count"] = 1
    exceeded = _exceeded(policy.semantic_limits, initial_proposed)
    if exceeded:
        return _censored_outcome(
            dsu=dsu,
            counters=counters,
            reason="budget-exhaustion",
            exceeded=exceeded,
            event_time=zero,
            graph=graph,
            charge_source=charge_source,
            charge_target=charge_target,
            zero=zero,
        )

    queue = _PersistentQueue(
        graph=graph,
        dsu=dsu,
        weights=weights,
        ops=ops,
        adjacency=adjacency,
        charge_source=charge_source,
        charge_target=charge_target,
    )
    active_roots = set(_active_roots(dsu))
    initial_schedule = queue.reschedule_many(
        queue.incident_indices(active_roots), zero
    )
    proposed = _counter_dict(counters)
    proposed.update(
        heap_push_count=initial_schedule.pushes,
        heap_operation_count=initial_schedule.pushes,
        peak_heap_size=initial_schedule.peak_size,
        temporary_memory_units=initial_schedule.temporary_units,
    )
    exceeded = _exceeded(policy.production_limits, proposed)
    if exceeded:
        return _censored_outcome(
            dsu=dsu,
            counters=counters,
            reason="budget-exhaustion",
            exceeded=exceeded,
            event_time=zero,
            graph=graph,
            charge_source=charge_source,
            charge_target=charge_target,
            zero=zero,
        )
    counters = dataclasses.replace(
        counters,
        heap_push_count=initial_schedule.pushes,
        heap_operation_count=initial_schedule.pushes,
        peak_heap_size=initial_schedule.peak_size,
        temporary_memory_units=initial_schedule.temporary_units,
    )

    batch_id = 0
    while active_roots:
        prior_event_time = event_time
        heap_before = list(queue.heap)
        next_time, selected_indices, pop_cost = queue.next_batch()
        if next_time is None:
            proposed = _counter_dict(counters)
            proposed.update(
                heap_pop_count=counters.heap_pop_count + pop_cost.pops,
                stale_heap_pop_count=(
                    counters.stale_heap_pop_count + pop_cost.stale_pops
                ),
                heap_operation_count=counters.heap_operation_count + pop_cost.pops,
            )
            exceeded = _exceeded(policy.production_limits, proposed)
            if exceeded:
                queue.heap = heap_before
                return _censored_outcome(
                    dsu=dsu,
                    counters=counters,
                    reason="budget-exhaustion",
                    exceeded=exceeded,
                    event_time=event_time,
                    graph=graph,
                    charge_source=charge_source,
                    charge_target=charge_target,
                    zero=zero,
                    adjacency=adjacency,
                )
            counters = dataclasses.replace(
                counters,
                heap_pop_count=proposed["heap_pop_count"],
                stale_heap_pop_count=proposed["stale_heap_pop_count"],
                heap_operation_count=proposed["heap_operation_count"],
            )
            return _censored_outcome(
                dsu=dsu,
                counters=counters,
                reason="local-incomplete-neutralization",
                exceeded=[],
                event_time=event_time,
                graph=graph,
                charge_source=charge_source,
                charge_target=charge_target,
                zero=zero,
                adjacency=adjacency,
            )
        correction_edges = sorted(
            (
                graph.edges[k]
                for k in selected_indices
                if graph.edges[k].kind == "correction"
            ),
            key=lambda edge: _edge_order(
                edge, weights[edge_index_by_id[edge.edge_id]]
            ),
        )
        boundary_edges = [
            graph.edges[k]
            for k in selected_indices
            if graph.edges[k].kind == "boundary"
        ]
        affected_roots = {
            dsu.find(vertex)
            for k in selected_indices
            for vertex in (
                (graph.edges[k].source, graph.edges[k].target)
                if graph.edges[k].target is not None
                else (graph.edges[k].source,)
            )
        }
        affected_indices = queue.incident_indices(affected_roots)
        attempts, successes, largest, proposed_forest = _preview_batch(
            dsu,
            correction_edges,
            boundary_edges,
            counters.forest_edge_count,
        )
        proposed = _counter_dict(counters)
        proposed.update(
            growth_event_count=counters.growth_event_count + len(selected_indices),
            simultaneous_event_batch_count=counters.simultaneous_event_batch_count + 1,
            union_attempt_count=counters.union_attempt_count + attempts,
            successful_union_count=counters.successful_union_count + successes,
            forest_edge_count=proposed_forest,
            absorbed_vertex_count=largest,
            heap_pop_count=counters.heap_pop_count + pop_cost.pops,
            heap_operation_count=counters.heap_operation_count + pop_cost.pops,
        )
        exceeded = _exceeded(policy.semantic_limits, proposed)
        exceeded += _exceeded(policy.production_limits, proposed)
        exceeded = sorted(set(exceeded))
        if exceeded:
            queue.heap = heap_before
            return _censored_outcome(
                dsu=dsu,
                counters=counters,
                reason="budget-exhaustion",
                exceeded=exceeded,
                event_time=prior_event_time,
                graph=graph,
                charge_source=charge_source,
                charge_target=charge_target,
                zero=zero,
                adjacency=adjacency,
            )
        vertices_before = {
            root: tuple(dsu.state[root].vertices) for root in affected_roots
        }
        defect_counts_before = {
            root: len(dsu.state[root].defects) for root in affected_roots
        }
        active_before = {
            root: dsu.state[root].active for root in affected_roots
        }
        boundary_before = {
            root: dsu.state[root].boundary_reached for root in affected_roots
        }
        port_before = {
            root: dsu.state[root].port_reached for root in affected_roots
        }
        old_root_by_vertex = {
            vertex: root
            for root, vertices in vertices_before.items()
            for vertex in vertices
        }
        scratch_parent = {root: root for root in affected_roots}

        def scratch_find(root: int) -> int:
            while scratch_parent[root] != root:
                root = scratch_parent[root]
            return root

        for edge in correction_edges:
            source_root = scratch_find(dsu.find(edge.source))
            target_root = scratch_find(dsu.find(int(edge.target)))
            if source_root != target_root:
                scratch_parent[target_root] = source_root
        grouped_roots: dict[int, set[int]] = {}
        for root in affected_roots:
            grouped_roots.setdefault(scratch_find(root), set()).add(root)
        boundary_groups = {
            scratch_find(dsu.find(edge.source)) for edge in boundary_edges
        }
        port_groups = {
            scratch_find(dsu.find(graph.edges[k].source))
            for k in selected_indices
            if graph.edges[k].kind == "port"
        }
        touch_indices = set(selected_indices)
        for group_root, roots in grouped_roots.items():
            post_active = (
                bool(sum(defect_counts_before[root] for root in roots) & 1)
                and group_root not in boundary_groups
                and not any(boundary_before[root] for root in roots)
                and group_root not in port_groups
                and not any(port_before[root] for root in roots)
            )
            for root in roots:
                if active_before[root] != post_active:
                    for vertex in vertices_before[root]:
                        touch_indices.update(adjacency[vertex])
        for k in affected_indices:
            edge = graph.edges[k]
            if edge.kind != "correction":
                continue
            assert edge.target is not None
            old_source = old_root_by_vertex.get(edge.source)
            old_target = old_root_by_vertex.get(edge.target)
            if (
                old_source is not None
                and old_target is not None
                and old_source != old_target
                and scratch_find(old_source) == scratch_find(old_target)
            ):
                touch_indices.add(k)
        possible_post_indices = touch_indices.difference(selected_indices)
        upper_heap_size = len(queue.heap) + len(possible_post_indices)
        upper_proposed = dict(proposed)
        upper_proposed.update(
            heap_push_count=(
                counters.heap_push_count + len(possible_post_indices)
            ),
            heap_operation_count=(
                counters.heap_operation_count
                + pop_cost.pops
                + len(possible_post_indices)
            ),
            peak_heap_size=max(counters.peak_heap_size, upper_heap_size),
            temporary_memory_units=max(
                counters.temporary_memory_units, upper_heap_size
            ),
        )
        rollback_needed = bool(
            _exceeded(policy.production_limits, upper_proposed)
        )
        parent_before = list(dsu.parent) if rollback_needed else None
        size_before = list(dsu.size) if rollback_needed else None
        states_rollback = (
            {
                root: copy.deepcopy(dsu.state[root])
                for root in affected_roots
            }
            if rollback_needed
            else None
        )
        queue_values_before = (
            {
                k: (
                    charge_source[k],
                    charge_target[k],
                    queue.last_source[k],
                    queue.last_target[k],
                    queue.consumed[k],
                    queue.version[k],
                )
                for k in touch_indices
            }
            if rollback_needed
            else None
        )
        for k in touch_indices:
            queue.touch(k, next_time)

        event_time = next_time
        batch_id += 1
        successful = 0
        failed = 0
        for edge in correction_edges:
            root, did_union = dsu.union(
                edge.source,
                int(edge.target),
                edge.edge_id,
                batch_id,
                event_time,
            )
            if not did_union:
                dsu.state[root].batch_ids.add(batch_id)
                dsu.state[root].batch_times[batch_id] = event_time
            successful += int(did_union)
            failed += int(not did_union)
        for k in sorted(selected_indices, key=lambda i: graph.edges[i].edge_id):
            edge = graph.edges[k]
            queue.consumed[k] = True
            queue.version[k] += 1
            if edge.kind == "boundary":
                root = dsu.find(edge.source)
                state = dsu.state[root]
                state.saturated_boundary_edges.add(edge.edge_id)
                state.boundary_forest_edge = (
                    edge.edge_id
                    if state.boundary_forest_edge is None
                    else min(state.boundary_forest_edge, edge.edge_id)
                )
                state.batch_ids.add(batch_id)
                state.batch_times[batch_id] = event_time
            elif edge.kind == "port":
                root = dsu.find(edge.source)
                state = dsu.state[root]
                state.port_kinds.add(str(edge.port_kind))
                state.saturated_port_edges.add(edge.edge_id)
                state.batch_ids.add(batch_id)
                state.batch_times[batch_id] = event_time

        post_affected_roots = {
            dsu.find(vertex)
            for root in affected_roots
            for vertex in vertices_before[root]
        }
        post_indices: set[int] = set()
        for old_root, was_active in active_before.items():
            if was_active != dsu.state[dsu.find(old_root)].active:
                for vertex in vertices_before[old_root]:
                    post_indices.update(adjacency[vertex])
        # A correction between two just-merged pre-batch components becomes
        # internal even when the surviving component remains active.  Those
        # entries alone need invalidation; all other unchanged-rate absolute
        # event times remain valid in the persistent heap.
        for k in affected_indices:
            edge = graph.edges[k]
            if queue.consumed[k] or edge.kind != "correction":
                continue
            assert edge.target is not None
            old_source = old_root_by_vertex.get(edge.source)
            old_target = old_root_by_vertex.get(edge.target)
            if (
                old_source is not None
                and old_target is not None
                and old_source != old_target
                and dsu.find(edge.source) == dsu.find(edge.target)
            ):
                post_indices.add(k)
        push_cost = queue.reschedule_many(post_indices, event_time)
        proposed.update(
            successful_union_count=counters.successful_union_count + successful,
            heap_push_count=counters.heap_push_count + push_cost.pushes,
            heap_operation_count=(
                counters.heap_operation_count + pop_cost.pops + push_cost.pushes
            ),
            peak_heap_size=max(counters.peak_heap_size, push_cost.peak_size),
            temporary_memory_units=max(
                counters.temporary_memory_units, push_cost.temporary_units
            ),
        )
        post_exceeded = _exceeded(policy.production_limits, proposed)
        if post_exceeded:
            assert rollback_needed
            assert parent_before is not None
            assert size_before is not None
            assert states_rollback is not None
            assert queue_values_before is not None
            dsu.parent[:] = parent_before
            dsu.size[:] = size_before
            for root, state in states_rollback.items():
                dsu.state[root] = state
            queue.heap = heap_before
            for k, values in queue_values_before.items():
                (
                    charge_source[k],
                    charge_target[k],
                    queue.last_source[k],
                    queue.last_target[k],
                    queue.consumed[k],
                    queue.version[k],
                ) = values
            return _censored_outcome(
                dsu=dsu,
                counters=counters,
                reason="budget-exhaustion",
                exceeded=post_exceeded,
                event_time=prior_event_time,
                graph=graph,
                charge_source=charge_source,
                charge_target=charge_target,
                zero=zero,
                adjacency=adjacency,
            )
        active_roots.difference_update(affected_roots)
        active_roots.update(
            root for root in post_affected_roots if dsu.state[root].active
        )
        counters = UFCounters(
            growth_event_count=proposed["growth_event_count"],
            simultaneous_event_batch_count=proposed[
                "simultaneous_event_batch_count"
            ],
            union_attempt_count=proposed["union_attempt_count"],
            successful_union_count=counters.successful_union_count + successful,
            failed_union_count=counters.failed_union_count + failed,
            forest_edge_count=proposed_forest,
            peel_operation_count=counters.peel_operation_count,
            heap_push_count=proposed["heap_push_count"],
            heap_pop_count=proposed["heap_pop_count"],
            stale_heap_pop_count=counters.stale_heap_pop_count + pop_cost.stale_pops,
            heap_operation_count=proposed["heap_operation_count"],
            peak_heap_size=proposed["peak_heap_size"],
            peak_live_component_count=counters.peak_live_component_count,
            temporary_memory_units=proposed["temporary_memory_units"],
        )

    # Bring every final-component incidence charge to the terminal event time
    # before confidence and charge telemetry are computed.
    for k in range(len(graph.edges)):
        queue.touch(k, event_time)
    completed, counters, exceeded = _finish_components(
        graph=graph,
        dsu=dsu,
        weights=weights,
        charge_source=charge_source,
        charge_target=charge_target,
        tau=tau,
        counters=counters,
        policy=policy,
        production=True,
        adjacency=adjacency,
    )
    if completed is None:
        return _censored_outcome(
            dsu=dsu,
            counters=counters,
            reason="budget-exhaustion",
            exceeded=exceeded,
            event_time=event_time,
            graph=graph,
            charge_source=charge_source,
            charge_target=charge_target,
            zero=zero,
        )
    return LaneOutcome(
        status="completed",
        completed_components=completed,
        censored_components=(),
        counters=counters,
        censor_reason=None,
        budget_exceeded_set=(),
        primary_budget_cap=None,
        terminal_event_time=event_time,
        last_complete_batch_id=(
            counters.simultaneous_event_batch_count
            if counters.simultaneous_event_batch_count
            else None
        ),
    )


def run_reference_lane(
    graph: LaneGraphProtocol,
    defects: Iterable[int],
    policy: UFPolicy,
) -> LaneOutcome:
    """Runs the exact Fraction semantic reference on one lane."""

    return _run_lane_exact(
        graph, defects, policy, ops=_FractionOps(), production=False
    )


__all__ = [
    "BudgetExceeded",
    "BudgetLimits",
    "CensoredComponent",
    "CompletedComponent",
    "LaneGraphProtocol",
    "LaneOutcome",
    "UFEdge",
    "UFCounters",
    "UFLaneGraph",
    "UFPolicy",
    "as_fraction",
    "run_reference_lane",
]
