from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Iterable

import numpy as np

from yoked.decoders._graph import DecodingGraph, _index
from yoked.decoders._union_find import UnionFindDecoder, _Correction

if TYPE_CHECKING:
    import stim


class CorrelatedUnionFindDecoder(UnionFindDecoder):
    """Two UF passes, with edge correlations informing the second pass.

    Each correlation rule is (source_edge, target_edge, implied_weight).
    Selecting the source in the first correction lowers the target's weight
    for the second pass. Edge IDs refer to positions in ``graph.edges``.

    Use ``from_dem`` to derive the rules from shared DEM error mechanisms,
    or supply them explicitly for a graph. Both passes use the original
    syndrome and the repository's weighted growth and peeling algorithm.
    """

    def __init__(self, graph: DecodingGraph, *, correlation_rules: Iterable[tuple[int, int, float]]):
        super().__init__(graph)
        rules: list[list[tuple[int, float]]] = [[] for _ in graph.edges]
        for source, target, weight in correlation_rules:
            source = _index(source, 'Correlation source edge')
            target = _index(target, 'Correlation target edge')
            weight = float(weight)
            if not 0 <= source < len(graph.edges) or not 0 <= target < len(graph.edges):
                raise ValueError('Correlation rule has an out-of-range edge ID')
            if source == target:
                raise ValueError('Correlation rules must connect different edges')
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('Implied weights must be finite and nonnegative')
            rules[source].append((target, weight))
        # Own an immutable copy, just as DecodingGraph owns its edge list.
        self._correlation_rules = tuple(tuple(targets) for targets in rules)

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> CorrelatedUnionFindDecoder:
        """Import a decomposed DEM and its pairwise correlation rules.

        Graph import performs the usual component/weight/label audit. A DEM
        error that repeats an edge across components is additionally rejected:
        the repeated detector incidences cancel instead of forming partners.
        PyMatching is used for graph export only; neither UF pass runs MWPM.
        """
        graph = DecodingGraph.from_dem(dem)
        return cls(graph, correlation_rules=_correlation_rules_from_dem(graph, dem))

    def _decode(self, syndrome: np.ndarray) -> _Correction:
        first = super()._decode(syndrome)
        weights = [weight for _, _, weight, _ in self.graph.edges]
        changed = False
        # Evidence comes from the selected correction, not every grown edge.
        # Several sources can support one target; retain the lowest weight.
        for source in first.selected_edges:
            for target, implied_weight in self._correlation_rules[source]:
                if implied_weight < weights[target]:
                    weights[target] = implied_weight
                    changed = True
        if not changed:
            return first

        # Keep edge IDs, endpoints, and observable labels unchanged. A fresh
        # graph keeps these shot-specific weights out of subsequent calls.
        adjusted = DecodingGraph(
            self.graph.num_detectors,
            self.graph.num_observables,
            [(u, v, weights[e], mask) for e, (u, v, _, mask) in enumerate(self.graph.edges)],
        )
        # Restart growth on the original syndrome. The first correction was
        # evidence only; the second correction supplies the complete answer.
        return UnionFindDecoder(adjusted)._decode(syndrome)


def _correlation_rules_from_dem(
    graph: DecodingGraph, dem: stim.DetectorErrorModel,
) -> list[tuple[int, int, float]]:
    """Compile PyMatching's pairwise implied-weight approximation once."""
    edge_ids = {
        (u, None) if v is None else tuple(sorted((u, v))): e
        for e, (u, v, _, _) in enumerate(graph.edges)
    }
    marginal = [0.0] * len(graph.edges)
    shared: dict[tuple[int, int], float] = {}
    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        probability, = instruction.args_copy()
        if probability == 0:
            continue

        # Separators split one error mechanism into its graph components.
        # The graph audit has already checked detector counts and labels.
        components: list[list[int]] = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                components.append([])
            elif target.is_relative_detector_id():
                components[-1].append(target.val)
        edges = [
            edge_ids[(ds[0], None) if len(ds) == 1 else tuple(sorted(ds))]
            for ds in components if ds
        ]
        if len(set(edges)) != len(edges):
            raise ValueError('A DEM error repeats a graph edge across components')
        for e in edges:
            marginal[e] = _xor_probability(marginal[e], probability)
        for a, b in itertools.combinations(sorted(edges), 2):
            shared[a, b] = _xor_probability(shared.get((a, b), 0.0), probability)

    rules = []
    for (a, b), probability in shared.items():
        if probability == 0:
            continue
        for source, target in ((a, b), (b, a)):
            # This is an implied probability from shared mechanisms, not a
            # full posterior over physical faults. Clipping keeps weights
            # nonnegative, including zero for strongly supported partners.
            implied_probability = min(0.5, probability / marginal[source])
            weight = math.log1p(-implied_probability) - math.log(implied_probability)
            if weight < graph.edges[target][2]:
                rules.append((source, target, weight))
    return rules


def _xor_probability(a: float, b: float) -> float:
    # Independent mechanisms toggle an edge: two occurrences cancel.
    return a * (1 - b) + b * (1 - a)
