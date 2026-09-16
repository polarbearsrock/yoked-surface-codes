"""Random-priority forest ensembles on fixed UF clusters.

This implements the forest exploration and minimum-size coset vote of
Liang et al., arXiv:2606.11076, adapted to weighted detector graphs with
open boundaries. It is a software experiment, not the paper's FPGA design.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np

from yoked.decoders._correlations import (
    CorrelationRule, apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.decoders._graph import DecodingGraph, _index
from yoked.decoders._union_find import InvalidSyndromeError, UnionFindDecoder, _validate_syndromes

if TYPE_CHECKING:
    import stim

Score = Literal['size', 'weight']


@dataclass(frozen=True)
class ForestCandidate:
    selected_edges: tuple[int, ...]
    observable_mask: int
    weight: float


@dataclass(frozen=True)
class EnsembleDecodeResult:
    """Candidates and a certificate of the cluster graph's logical freedom.

    ``logical_rank`` is dim L(ker H) on all edges internal to the final UF
    partitions. Rank zero proves that *every* syndrome-valid correction
    supported there has the baseline logical label, irrespective of sampling.
    The baseline is diagnostic only: it does not receive an extra vote.
    """
    baseline: ForestCandidate
    candidates: tuple[ForestCandidate, ...]
    logical_rank: int
    internal_edges: int

    def choose(self, score: Score = 'size', *, count: int | None = None) -> ForestCandidate:
        """Vote among minimum-score candidates; break tied cosets by first sample.

        Repeated candidates retain their votes. ``weight`` is an experimental
        variant using summed shot-specific edge weights instead of edge count.
        Its minimum-cost tie tolerance is 1e-12 absolute and relative.
        """
        if score not in ('size', 'weight'):
            raise ValueError('score must be size or weight')
        count = len(self.candidates) if count is None else _index(count, 'count')
        if not 1 <= count <= len(self.candidates):
            raise ValueError('count must select a nonempty candidate prefix')
        candidates = self.candidates[:count]
        costs = [len(c.selected_edges) if score == 'size' else c.weight for c in candidates]
        minimum = min(costs)
        eligible = [c for c, cost in zip(candidates, costs)
                    if (cost == minimum if score == 'size' else
                        math.isclose(cost, minimum, rel_tol=1e-12, abs_tol=1e-12))]
        votes = Counter(c.observable_mask for c in eligible)
        largest = max(votes.values())
        return next(c for c in eligible if votes[c.observable_mask] == largest)


class CosetEnsembleDecoder:
    """UF growth once, K priority BFS forests, reverse-order peeling, coset vote.

    Supplying correlation rules preserves ordinary UF's first correction as
    reweighting evidence and ensembles only the final pass. No matching solve
    is used. All graph edges internal to a final partition are eligible, even
    if they were not fully grown. No edge between partitions is introduced.

    Boundary terminals are identified *within each partition*, and its BFS
    starts at that free-parity vertex. This represents boundary-to-boundary
    alternatives without connecting separate UF clusters.

    Priorities are sampled once from PCG64 with a fixed seed. Reusing the same
    priorities on each syndrome makes results independent of batch ordering,
    worker count, and prior decode calls. K-prefixes use identical priorities.
    """

    def __init__(self, graph: DecodingGraph, *, candidates: int = 24, seed: int = 20260916,
                 score: Score = 'size', correlation_rules: Iterable[CorrelationRule] = ()):
        candidates = _index(candidates, 'candidates')
        seed = _index(seed, 'seed')
        if candidates < 1 or seed < 0:
            raise ValueError('candidates must be positive and seed nonnegative')
        if score not in ('size', 'weight'):
            raise ValueError('score must be size or weight')
        self.graph = graph
        self.score = score
        self.seed = seed
        self.num_candidates = candidates
        self._rules = index_rules_by_source(graph, correlation_rules)
        self._weights = tuple(w for _, _, w, _ in graph.edges)
        rng = np.random.Generator(np.random.PCG64(seed))
        # Interleave per-candidate draws so extending K preserves every prefix.
        self._priorities = tuple(
            (rng.random(graph.num_detectors), rng.random(len(graph.edges)))
            for _ in range(candidates)
        )

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel, *, correlated: bool = False,
                 candidates: int = 24, seed: int = 20260916, score: Score = 'size') -> CosetEnsembleDecoder:
        graph = DecodingGraph.from_dem(dem)
        rules = correlation_rules_from_dem(graph, dem) if correlated else ()
        return cls(graph, candidates=candidates, seed=seed, score=score, correlation_rules=rules)

    def decode_ensemble(self, syndrome: np.ndarray) -> EnsembleDecodeResult:
        syndrome = _validate_syndromes(syndrome, self.graph.num_detectors, 1)
        graph = self.graph
        correction, growth = UnionFindDecoder(graph)._decode_state(syndrome)
        weights = apply_correlation_rules(self._weights, self._rules, correction.selected_edges)
        if weights is not None:
            graph = DecodingGraph(graph.num_detectors, graph.num_observables,
                                  [(u, v, weights[e], mask)
                                   for e, (u, v, _, mask) in enumerate(graph.edges)])
            correction, growth = UnionFindDecoder(graph)._decode_state(syndrome)
        baseline = _candidate(graph, correction.selected_edges, correction.observable_mask)
        roots = [growth.find(v) for v in range(len(graph.adjacency))]
        adjacency, components, internal_edges = _cluster_graph(graph, roots)
        rank = _logical_rank(graph, adjacency, components)
        candidates = tuple(_explore(graph, syndrome, adjacency, components, vp, ep)
                           for vp, ep in self._priorities)
        return EnsembleDecodeResult(baseline, candidates, rank, internal_edges)

    def decode_to_edge_ids(self, syndrome: np.ndarray) -> tuple[int, ...]:
        return self.decode_ensemble(syndrome).choose(self.score).selected_edges

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        mask = self.decode_ensemble(syndrome).choose(self.score).observable_mask
        return np.fromiter(((mask >> k) & 1 for k in range(self.graph.num_observables)),
                           dtype=np.bool_, count=self.graph.num_observables)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = _validate_syndromes(syndromes, self.graph.num_detectors, 2)
        result = np.empty((len(syndromes), self.graph.num_observables), dtype=np.bool_)
        for row, syndrome in enumerate(syndromes):
            result[row] = self.decode(syndrome)
        return result


def _candidate(graph, edges, mask):
    edges = tuple(sorted(edges))
    return ForestCandidate(edges, mask, math.fsum(graph.edges[e][2] for e in edges))


def _cluster_graph(graph, roots):
    adjacency: dict[int, list[tuple[int, int]]] = {}
    components: dict[int, list[int]] = {}
    count = 0
    for e, (u, v) in enumerate(graph.endpoints):
        r = roots[u]
        if r != roots[v]:
            continue
        # Negative IDs distinguish each cluster's virtual boundary from detectors.
        if v >= graph.num_detectors:
            v = -r - 1
        for a, b in ((u, v), (v, u)):
            if a not in adjacency:
                adjacency[a] = []
                components.setdefault(r, []).append(a)
            adjacency[a].append((b, e))
        count += 1
    return adjacency, tuple(tuple(vs) for vs in components.values()), count


def _logical_rank(graph, adjacency, components):
    basis: dict[int, int] = {}
    for vertices in components:
        labels = {vertices[0]: 0}
        order = [vertices[0]]
        for u in order:
            for v, e in adjacency[u]:
                label = labels[u] ^ graph.edges[e][3]
                if v not in labels:
                    labels[v] = label
                    order.append(v)
                else:
                    delta = label ^ labels[v]
                    while delta:
                        pivot = delta.bit_length() - 1
                        if pivot not in basis:
                            basis[pivot] = delta
                            break
                        delta ^= basis[pivot]
    return len(basis)


def _explore(graph, syndrome, adjacency, components, vertex_priority, edge_priority):
    selected = []
    mask = 0
    for vertices in components:
        boundary = next((v for v in vertices if v < 0), None)
        root = boundary if boundary is not None else min(vertices, key=vertex_priority.__getitem__)
        parent = {root: (root, -1)}
        order = [root]
        for u in order:
            for v, e in sorted(adjacency[u], key=lambda item: (edge_priority[item[1]], item[1])):
                if v not in parent:
                    parent[v] = (u, e)
                    order.append(v)
        parity = {v: bool(syndrome[v]) if v >= 0 else False for v in order}
        for v in reversed(order[1:]):
            if parity[v]:
                u, e = parent[v]
                selected.append(e)
                parity[u] ^= True
                mask ^= graph.edges[e][3]
        if root >= 0 and parity[root]:
            raise InvalidSyndromeError('Odd residual parity in an ensemble tree')
    return _candidate(graph, selected, mask)
