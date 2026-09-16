"""UF-only confidence experiments with a fixed plain-UF reference.

The forced-class scores are *UF correction costs*, not minimum-weight costs
or likelihood ratios. Two check-graph decodes, with check bits (0, 0) and
(1, 1), supply both costs of each disconnected sector. Sector costs are
summed separately, so four whole-patch decodes are unnecessary.

Correlation weights use only the first UF correction and are frozen before
comparing classes. A second free UF pass supplies a separate, inexpensive
confidence candidate: its bounded cluster gap, signed relative to the
original reference. No matching solve runs in this module.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from yoked.decoders._correlations import apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source
from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._patch_graphs import PatchGraph

# Predeclared, not selected on evaluation shots. 20 dB in natural log units.
DEFAULT_GAP_CAP = math.log(100)
SCORE_NAMES = ('bounded_cluster', 'correlated_bounded_cluster', 'uf_gap', 'correlated_uf_gap')
TIME_NAMES = ('reference_uf', 'bounded_search', 'plain_forced', 'reweight',
              'correlated_uf', 'correlated_search', 'correlated_forced')


def quantize_gap(scores) -> np.ndarray:
    """Sixteen signed score bins, width 2 nats, with saturated tails.

    Calibrating these integer scores gives a 16-entry probability ROM per
    sector. This quantizes the confidence output, not UF edge arithmetic.
    """
    return np.clip(np.floor(np.asarray(scores) / 2), -8, 7)


def _reweighted(graph: DecodingGraph, weights) -> DecodingGraph:
    return DecodingGraph(graph.num_detectors, graph.num_observables,
                         [(u, v, weights[e], mask) for e, (u, v, _, mask) in enumerate(graph.edges)])


def _edge_sectors(patch: PatchGraph) -> np.ndarray:
    """Audit that the check vertices belong to disjoint components covering G."""
    graph = patch.check_graph
    labels = np.full(len(graph.adjacency), -1, dtype=np.int8)
    for sector, start in enumerate(patch.check_vertices):
        pending = [start]
        while pending:
            vertex = pending.pop()
            if labels[vertex] == sector:
                continue
            if labels[vertex] != -1:
                raise ValueError('Forced UF requires disconnected sector components')
            labels[vertex] = sector
            for edge in graph.adjacency[vertex]:
                u, v = graph.endpoints[edge]
                pending.append(v if u == vertex else u)
    sectors = np.array([labels[u] for u, _ in graph.endpoints], dtype=np.int8)
    if (sectors < 0).any():
        raise ValueError('A check-graph edge belongs to neither sector')
    return sectors


@dataclass(frozen=True)
class UFSoftResult:
    reference: np.ndarray
    correlated_prediction: np.ndarray
    scores: np.ndarray                 # (method, sector)
    forced_costs: np.ndarray           # (plain/correlated, sector, class)
    states: np.ndarray                 # (plain/correlated, sector)
    rules_fired: bool
    seconds: np.ndarray               # TIME_NAMES; measured CPU implementation work


class UFSoftDecoder:
    """Collect all UF confidence candidates while sharing their first pass."""

    def __init__(self, patch: PatchGraph, *, gap_cap: float = DEFAULT_GAP_CAP):
        if not math.isfinite(gap_cap) or gap_cap <= 0:
            raise ValueError('gap_cap must be positive and finite')
        self.patch = patch
        self.gap_cap = gap_cap
        self._uf = UnionFindDecoder(patch.graph)
        self._check = UnionFindDecoder(patch.check_graph)
        self._gap = ClusterGapUnionFindDecoder(patch.graph)
        self._sectors = _edge_sectors(patch)
        self._weights = tuple(w for _, _, w, _ in patch.graph.edges)
        self._rules = index_rules_by_source(patch.graph, correlation_rules_from_dem(patch.graph, patch.local_dem))

    def forced_costs(self, syndrome, decoder: UnionFindDecoder | None = None) -> np.ndarray:
        """Cost[sector, class] from two decodes of independent X/Z components."""
        decoder = self._check if decoder is None else decoder
        syndrome = np.asarray(syndrome)
        if syndrome.shape != (self.patch.num_detectors,):
            raise ValueError('Wrong number of local detector bits')
        if syndrome.dtype.kind not in 'buif' or not np.isin(syndrome, (0, 1)).all():
            raise ValueError('Expected binary detector bits')
        costs = np.zeros((2, 2))
        for logical_class in range(2):
            forced = np.concatenate([syndrome, [logical_class, logical_class]])
            selected = decoder.decode_to_edge_ids(forced)
            for sector in range(2):
                costs[sector, logical_class] = math.fsum(
                    decoder.graph.edges[e][2] for e in selected if self._sectors[e] == sector)
        return costs

    def decode(self, syndrome) -> UFSoftResult:
        seconds = np.zeros(len(TIME_NAMES))
        start = time.perf_counter()
        first = self._uf.decode_with_growth_costs(syndrome)
        reference = np.array([bool((first.observable_mask >> k) & 1) for k in range(2)])
        seconds[0] = time.perf_counter() - start
        start = time.perf_counter()
        gap, states = self._gap.gaps_from_costs(first.remaining_costs, max_gap=self.gap_cap)
        seconds[1] = time.perf_counter() - start
        start = time.perf_counter()
        plain_costs = self.forced_costs(syndrome)
        seconds[2] = time.perf_counter() - start
        start = time.perf_counter()
        adjusted = apply_correlation_rules(self._weights, self._rules, first.selected_edges)
        free = None if adjusted is None else UnionFindDecoder(_reweighted(self.patch.graph, adjusted))
        check = None if adjusted is None else UnionFindDecoder(_reweighted(self.patch.check_graph, adjusted))
        seconds[3] = time.perf_counter() - start
        if adjusted is None:
            prediction, corr_gap, corr_states, corr_costs = reference, gap, states, plain_costs
        else:
            start = time.perf_counter()
            second = free.decode_with_growth_costs(syndrome)
            prediction = np.array([bool((second.observable_mask >> k) & 1) for k in range(2)])
            seconds[4] = time.perf_counter() - start
            start = time.perf_counter()
            corr_gap, corr_states = self._gap.gaps_from_costs(second.remaining_costs, max_gap=self.gap_cap)
            seconds[5] = time.perf_counter() - start
            start = time.perf_counter()
            corr_costs = self.forced_costs(syndrome, check)
            seconds[6] = time.perf_counter() - start
        sign = 1 - 2 * reference.astype(int)
        scores = np.stack([gap, np.where(reference == prediction, corr_gap, -corr_gap),
                           sign * (plain_costs[:, 1] - plain_costs[:, 0]),
                           sign * (corr_costs[:, 1] - corr_costs[:, 0])])
        return UFSoftResult(reference, prediction, scores, np.stack([plain_costs, corr_costs]),
                            np.stack([states, corr_states]), adjusted is not None, seconds)
