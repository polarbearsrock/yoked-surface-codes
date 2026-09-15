"""Union-Find reference decoding with the cluster-gap soft output.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 5.1.

For a patch graph, run the repository's weighted UF (growth then peeling) to
obtain the reference bits r and a validated correction. Then compute, per
observable k, the cluster gap phi_k of Meister, Pattison and Preskill
(arXiv:2405.07433, Definition 9) on the terminated growth state:

    cost(e) = 0                   if both endpoints lie in one cluster,
    cost(e) = max(0, w_e - g_e)   otherwise: the growth the edge still needs,

    phi_k = minimum cost of a walk from the boundary B back to B whose
            edges flip observable k an odd number of times.

Every boundary terminal is merged into one vertex B. The walk is found by
Dijkstra over states (vertex, parity) from (B, 0) to (B, 1); traversing an
edge toggles the parity when its mask has bit k. The search is restricted
to the connected component that carries observable k. The number of settled
states is recorded as the soft-output work proxy.

``_cluster_gap_test.py`` checks phi against a brute-force enumeration of odd
closed walks on small graphs, with and without partial growth, and checks
that the reference bits equal plain UF's on the distance-3 fixture.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._arrays import readonly_array


@dataclass(frozen=True)
class ClusterGapResult:
    """L1 output of one patch for one shot.

    Fields: ``prediction`` (num_observables,) bool, the reference bits r;
    ``cluster_gap`` (num_observables,) float64 in nats, ``inf`` if no odd walk
    exists; ``dijkstra_states`` (num_observables,) int64 settled states;
    ``selected_edges`` edge ids of the validated correction.
    """
    prediction: np.ndarray
    cluster_gap: np.ndarray
    dijkstra_states: np.ndarray
    selected_edges: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, dtype in (('prediction', bool), ('cluster_gap', np.float64), ('dijkstra_states', np.int64)):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=dtype))


class ClusterGapUnionFindDecoder:
    """Repository UF plus the cluster gap of every observable of the graph."""

    def __init__(self, graph: DecodingGraph):
        self.graph = graph
        self._uf = UnionFindDecoder(graph)
        # One extra vertex id stands for every boundary terminal at once.
        self._boundary = graph.num_detectors
        self._adjacency = tuple(self._sector_adjacency(k) for k in range(graph.num_observables))

    def _sector_adjacency(self, observable: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Adjacency of the component carrying ``observable``, terminals merged into B."""
        parent = list(range(self.graph.num_detectors))

        def find(v: int) -> int:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for u, v, _, _ in self.graph.edges:
            if v is not None:
                parent[find(u)] = find(v)
        labels = {find(u) for u, _, _, mask in self.graph.edges if (mask >> observable) & 1}
        if len(labels) != 1:
            raise ValueError(f'Observable {observable} must be flipped by edges of exactly one component')
        label, = labels
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.graph.num_detectors + 1)]
        for e, (u, v, _, _) in enumerate(self.graph.edges):
            if find(u) != label:
                continue
            other = self._boundary if v is None else v
            adjacency[u].append((e, other))
            adjacency[other].append((e, u))
        return tuple(tuple(neighbours) for neighbours in adjacency)

    def decode_with_gaps(self, syndrome: np.ndarray) -> ClusterGapResult:
        decoded = self._uf.decode_with_growth_costs(syndrome)
        selected, mask = decoded.selected_edges, decoded.observable_mask
        costs = decoded.remaining_costs
        gaps, states = [], []
        for observable in range(self.graph.num_observables):
            gap, settled = self._shortest_odd_walk(costs, observable)
            gaps.append(gap)
            states.append(settled)
        prediction = np.array([(mask >> k) & 1 for k in range(self.graph.num_observables)], dtype=bool)
        return ClusterGapResult(prediction, np.array(gaps, dtype=np.float64),
                                np.array(states, dtype=np.int64), selected)

    def _shortest_odd_walk(self, costs: np.ndarray, observable: int) -> tuple[float, int]:
        """Dijkstra over states 2 * vertex + parity from (B, 0) to (B, 1)."""
        adjacency = self._adjacency[observable]
        start, target = 2 * self._boundary, 2 * self._boundary + 1
        distance = {start: 0.0}
        heap = [(0.0, start)]
        settled = 0
        while heap:
            d, state = heapq.heappop(heap)
            if d > distance.get(state, math.inf):
                continue  # a stale entry superseded by a shorter one
            settled += 1
            if state == target:
                return d, settled
            vertex, parity = divmod(state, 2)
            for e, other in adjacency[vertex]:
                flips = (self.graph.edges[e][3] >> observable) & 1
                next_state = 2 * other + (parity ^ flips)
                candidate = d + costs[e]
                if candidate < distance.get(next_state, math.inf):
                    distance[next_state] = candidate
                    heapq.heappush(heap, (candidate, next_state))
        return math.inf, settled
