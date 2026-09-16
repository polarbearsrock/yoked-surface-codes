from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import chain

import numpy as np

from yoked.decoders._graph import DecodingGraph


class InvalidSyndromeError(ValueError):
    """The syndrome cannot be produced by any edge set in the supplied graph."""


def _validate_syndromes(syndromes: np.ndarray, num_detectors: int, ndim: int) -> np.ndarray:
    data = np.asarray(syndromes)
    if data.ndim != ndim or data.shape[-1] != num_detectors:
        raise ValueError(f'Expected {ndim} dimensions with {num_detectors} detector bits')
    if data.dtype.kind not in 'buif' or not np.all((data == 0) | (data == 1)):
        raise ValueError('Syndromes must contain only binary values')
    return data.astype(np.bool_, copy=False)


@dataclass(frozen=True)
class _Correction:
    forest_edges: tuple[int, ...]
    selected_edges: tuple[int, ...]
    observable_mask: int


@dataclass(frozen=True)
class GrowthDecodeResult:
    """UF correction and the settled costs needed by a soft-output consumer.

    selected_edges: correction edge ids in graph order.
    observable_mask: XOR of selected edge masks, an integer bit mask.
    remaining_costs: one cost per graph edge in nats, zero within a cluster.
    """
    selected_edges: tuple[int, ...]
    observable_mask: int
    remaining_costs: tuple[float, ...]


class UnionFindDecoder:
    """Default UF decoder: repository growth and peeling on a fixed graph.

    All vertices use the same growth rule. Each call owns its working state,
    so the graph can be reused after either successful or failed decoding.
    """

    def __init__(self, graph: DecodingGraph):
        self.graph = graph

    def _validate(self, syndromes: np.ndarray, ndim: int) -> np.ndarray:
        return _validate_syndromes(syndromes, self.graph.num_detectors, ndim)

    def _decode_state(self, syndrome: np.ndarray) -> tuple[_Correction, _Growth]:
        """Run growth and peeling once, returning the correction and the growth state.

        Shared by ``_decode`` and ``decode_with_growth_costs`` so both entry
        points make exactly one growth-and-peel pass per syndrome and agree
        on the resulting correction. The ensemble decoder also uses the final
        partition; ``_Growth`` remains private to the decoder implementation.
        """
        syndrome = self._validate(syndrome, 1)
        growth = _Growth(self.graph, syndrome)
        growth.run()
        selected, mask = _peel(self.graph, syndrome, growth.forest)
        return _Correction(tuple(growth.forest), selected, mask), growth

    def _decode(self, syndrome: np.ndarray) -> _Correction:
        correction, _ = self._decode_state(syndrome)
        return correction

    def decode_to_edge_ids(self, syndrome: np.ndarray) -> tuple[int, ...]:
        """Return correction edge ids, in graph order, using this decoder's passes.

        This exposes the correction without settling growth costs when a consumer
        only needs to sum its weights. Subclasses overriding ``_decode`` retain
        their decoding semantics here, just as they do in ``decode``.
        """
        return self._decode(syndrome).selected_edges

    def decode_with_growth_costs(self, syndrome: np.ndarray) -> GrowthDecodeResult:
        """Decode and additionally report each edge's remaining growth cost.

        An edge whose endpoints share a cluster root after growth costs
        nothing; every other edge costs the growth it still needs to reach
        its weight. Settling each edge here (rather than during growth)
        keeps the hot path in ``_Growth`` free of this extra bookkeeping,
        since only soft-output consumers need it.

        This reports the single-pass growth state terminated by
        ``_decode_state``; a subclass that overrides ``_decode`` with a
        different decoding procedure (for example a second pass over
        reweighted edges) must override this method too, or it will report
        costs from a growth state that does not correspond to its correction.
        """
        correction, growth = self._decode_state(syndrome)
        costs = []
        for edge_id, (u, v) in enumerate(self.graph.endpoints):
            growth._settle(edge_id)
            cost = (0.0 if growth.find(u) == growth.find(v)
                    else max(0.0, self.graph.edges[edge_id][2] - growth.grown[edge_id]))
            costs.append(cost)
        return GrowthDecodeResult(correction.selected_edges, correction.observable_mask, tuple(costs))

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Return a boolean vector of predicted observables for one syndrome."""
        mask = self._decode(syndrome).observable_mask
        return np.fromiter(
            ((mask >> k) & 1 for k in range(self.graph.num_observables)),
            dtype=np.bool_, count=self.graph.num_observables,
        )

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a binary (shots, num_detectors) array without changing it."""
        syndromes = self._validate(syndromes, 2)
        predictions = np.empty((len(syndromes), self.graph.num_observables), dtype=np.bool_)
        for k, syndrome in enumerate(syndromes):
            predictions[k] = self.decode(syndrome)
        return predictions


class _Growth:
    """Per-shot DSU, merge forest, and lazy heap of edge completions."""

    def __init__(self, graph: DecodingGraph, syndrome: np.ndarray):
        self.graph = graph
        n = len(graph.adjacency)
        m = len(graph.edges)
        self.parent = list(range(n))
        self.size = [1] * n
        self.parity = [bool(x) for x in syndrome] + [False] * (n - graph.num_detectors)
        self.terminal = [v >= graph.num_detectors for v in range(n)]
        self.frontier = [set(a) for a in graph.adjacency]
        self.num_active = sum(self.parity)
        self.forest: list[int] = []
        self.time = 0.0
        self.grown = [0.0] * m
        self.settled_at = [0.0] * m
        self.rate = [int(self.active(u)) + int(self.active(v)) for u, v in graph.endpoints]
        self.generation = [0] * m
        self.heap: list[tuple[float, int, int]] = []
        for e, (_, _, weight, _) in enumerate(graph.edges):
            if weight == 0:
                self.heap.append((0.0, e, 0))
            elif self.rate[e]:
                self.heap.append((weight / self.rate[e], e, 0))
        heapq.heapify(self.heap)

    def find(self, v: int) -> int:
        while v != self.parent[v]:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def active(self, root: int) -> bool:
        return self.parity[root] and not self.terminal[root]

    def _settle(self, e: int) -> None:
        self.grown[e] = min(
            self.graph.edges[e][2],
            self.grown[e] + self.rate[e] * (self.time - self.settled_at[e]),
        )
        self.settled_at[e] = self.time

    def _next_time(self) -> float:
        while self.heap and self.heap[0][2] != self.generation[self.heap[0][1]]:
            heapq.heappop(self.heap)
        return self.heap[0][0] if self.heap else float('inf')

    def _merge_group(self, edges: list[int]) -> None:
        # Snapshot the original parts before unions change their frontiers.
        # Only parts whose final activity differs need their rates revisited.
        parts = {self.find(v) for e in edges for v in self.graph.endpoints[e]}
        before = {r: (self.active(r), self.frontier[r].copy()) for r in parts}
        internal = set()
        for e in sorted(edges):
            u, v = self.graph.endpoints[e]
            a, b = self.find(u), self.find(v)
            if a == b:
                continue
            if (self.size[a], -a) < (self.size[b], -b):
                a, b = b, a
            self.num_active -= int(self.active(a)) + int(self.active(b))
            self.parent[b] = a
            self.size[a] += self.size[b]
            self.parity[a] ^= self.parity[b]
            self.terminal[a] |= self.terminal[b]
            self.num_active += int(self.active(a))
            internal.update(self.frontier[a] & self.frontier[b])
            self.frontier[a].symmetric_difference_update(self.frontier[b])
            self.frontier[b] = set()
            self.forest.append(e)

        completed = set(edges)
        for e in internal:
            self._settle(e)
            if e in completed:
                self.grown[e] = self.graph.edges[e][2]
            self.rate[e] = 0
            self.generation[e] += 1

        affected = set()
        for r, (was_active, frontier) in before.items():
            if was_active != self.active(self.find(r)):
                affected.update(frontier)
        affected.difference_update(internal)
        for e in affected:
            u, v = self.graph.endpoints[e]
            a, b = self.find(u), self.find(v)
            new_rate = int(self.active(a)) + int(self.active(b)) if a != b else 0
            if new_rate == self.rate[e]:
                continue
            self._settle(e)
            self.rate[e] = new_rate
            self.generation[e] += 1
            if new_rate:
                remaining = max(0.0, self.graph.edges[e][2] - self.grown[e])
                heapq.heappush(self.heap, (self.time + remaining / new_rate, e, self.generation[e]))

    def step(self) -> bool:
        """Process one anchored tie batch, including newly scheduled ties."""
        next_time = self._next_time()
        if not self.num_active and next_time != 0:
            return False
        if not self.heap:
            raise InvalidSyndromeError('Odd detector component has no boundary or remaining growth edge')
        self.time = next_time
        limit = self.time + 1e-12 + 1e-12 * abs(self.time)
        if not math.isfinite(limit):
            raise OverflowError('UF growth time overflowed; rescale the graph weights')
        while self._next_time() <= limit:
            edges = []
            while self._next_time() <= limit:
                _, e, _ = heapq.heappop(self.heap)
                edges.append(e)
            self._merge_group(edges)
        return True

    def run(self) -> None:
        while self.step():
            pass


def _peel(graph: DecodingGraph, syndrome: np.ndarray, forest: list[int]) -> tuple[tuple[int, ...], int]:
    tree: list[list[tuple[int, int]]] = [[] for _ in graph.adjacency]
    for e in forest:
        u, v = graph.endpoints[e]
        tree[u].append((v, e))
        tree[v].append((u, e))
    nd = graph.num_detectors
    parent = [-1] * len(tree)
    parent_edge = [-1] * len(tree)
    parity = [bool(x) for x in syndrome] + [False] * (len(tree) - nd)
    selected = []
    mask = 0
    # Visiting terminals first roots each tree at its smallest terminal;
    # a tree without one is rooted at its smallest detector.
    for root in chain(range(nd, len(tree)), range(nd)):
        if parent[root] != -1:
            continue
        parent[root] = root
        order = [root]
        for u in order:
            for v, e in tree[u]:
                if parent[v] == -1:
                    parent[v] = u
                    parent_edge[v] = e
                    order.append(v)
        for v in reversed(order[1:]):
            if v < nd and parity[v]:
                e = parent_edge[v]
                selected.append(e)
                parity[parent[v]] ^= True
                mask ^= graph.edges[e][3]
        if root < nd and parity[root]:
            raise InvalidSyndromeError('Odd residual parity at a detector root')
    return tuple(selected), mask
