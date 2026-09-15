"""Pairwise correlation rules derived from a decomposed DEM.

A DEM error mechanism that decomposes into several graph edges makes those
edges correlated: selecting one edge in a first decoding pass is evidence
that its partners occurred too. Following PyMatching's pairwise
approximation, every ordered pair (source, target) of edges that share at
least one mechanism yields a rule ``(source, target, implied_weight)``.
When ``source`` is selected, the second pass lowers ``target``'s weight to
``implied_weight = ln((1 - p) / p)`` with ``p = min(1/2, P(shared) / P(source))``.

Rules never raise a weight. When several selected sources support one
target, the lowest implied weight wins. ``_correlations_test.py`` checks the
compiled weights on hand-written DEMs and the application rule; the
correlated UF tests cover the decoder that uses them.
"""
from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Iterable, Sequence

from yoked.decoders._graph import DecodingGraph, _index

if TYPE_CHECKING:
    import stim

CorrelationRule = tuple[int, int, float]
"""``(source edge id, target edge id, implied weight in nats)``."""

RulesBySource = tuple[tuple[tuple[int, float], ...], ...]
"""``rules_by_source[source]`` lists ``(target, implied_weight)`` pairs in rule order."""


def correlation_rules_from_dem(
        graph: DecodingGraph, dem: stim.DetectorErrorModel,
) -> list[CorrelationRule]:
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


def index_rules_by_source(graph: DecodingGraph, rules: Iterable[CorrelationRule]) -> RulesBySource:
    """Validate rules against ``graph`` and group them by source edge, keeping order."""
    by_source: list[list[tuple[int, float]]] = [[] for _ in graph.edges]
    for source, target, weight in rules:
        source = _index(source, 'Correlation source edge')
        target = _index(target, 'Correlation target edge')
        weight = float(weight)
        if not 0 <= source < len(graph.edges) or not 0 <= target < len(graph.edges):
            raise ValueError('Correlation rule has an out-of-range edge ID')
        if source == target:
            raise ValueError('Correlation rules must connect different edges')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Implied weights must be finite and nonnegative')
        by_source[source].append((target, weight))
    return tuple(tuple(targets) for targets in by_source)


def apply_correlation_rules(
        weights: Sequence[float], rules_by_source: RulesBySource, selected_edges: Iterable[int],
) -> list[float] | None:
    """Lower target weights supported by the selected edges; ``None`` if nothing changes.

    Evidence comes from the selected correction only, never from every grown
    edge. Several sources can support one target; the lowest weight is kept.
    """
    adjusted = list(weights)
    changed = False
    for source in selected_edges:
        for target, implied_weight in rules_by_source[source]:
            if implied_weight < adjusted[target]:
                adjusted[target] = implied_weight
                changed = True
    return adjusted if changed else None


def _xor_probability(a: float, b: float) -> float:
    # Independent mechanisms toggle an edge: two occurrences cancel.
    return a * (1 - b) + b * (1 - a)
