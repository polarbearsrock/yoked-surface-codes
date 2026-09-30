from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from yoked.decoders._correlations import (
    CorrelationRule, apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import GrowthDecodeResult, UnionFindDecoder, _Correction

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
    The rule compilation and application live in ``_correlations`` so that
    the hierarchical matching gaps share them.
    """

    def __init__(self, graph: DecodingGraph, *, correlation_rules: Iterable[CorrelationRule]):
        super().__init__(graph)
        # Own an immutable copy, just as DecodingGraph owns its edge list.
        self._correlation_rules = index_rules_by_source(graph, correlation_rules)

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> CorrelatedUnionFindDecoder:
        """Import a decomposed DEM and its pairwise correlation rules.

        Graph import performs the usual component/weight/label audit. A DEM
        error that repeats an edge across components is additionally rejected:
        the repeated detector incidences cancel instead of forming partners.
        PyMatching is used for graph export only; neither UF pass runs MWPM.
        """
        graph = DecodingGraph.from_dem(dem)
        return cls(graph, correlation_rules=correlation_rules_from_dem(graph, dem))

    def _decode(self, syndrome) -> _Correction:
        first = super()._decode(syndrome)
        second = self._second_pass_decoder(first.selected_edges)
        return first if second is None else second._decode(syndrome)

    def _second_pass_decoder(self, selected_edges) -> UnionFindDecoder | None:
        """Freeze first-pass correlation evidence in a fresh graph for this shot."""
        weights = apply_correlation_rules(
            [weight for _, _, weight, _ in self.graph.edges],
            self._correlation_rules,
            selected_edges,
        )
        if weights is None:
            return None

        # Keep edge IDs, endpoints, and observable labels unchanged. A fresh
        # graph keeps these shot-specific weights out of subsequent calls.
        adjusted = DecodingGraph(
            self.graph.num_detectors,
            self.graph.num_observables,
            [(u, v, weights[e], mask) for e, (u, v, _, mask) in enumerate(self.graph.edges)],
        )
        # Restart growth on the original syndrome. The first correction was
        # evidence only; the second correction supplies the complete answer.
        return UnionFindDecoder(adjusted)

    def decode_with_growth_costs(self, syndrome) -> GrowthDecodeResult:
        """Return the final correction and growth costs on the same weighted graph.

        The first pass supplies correlation evidence only. When it lowers any
        weight, both the returned correction and the remaining edge lengths
        come from a fresh second pass. The original graph remains unchanged.
        Computing first-pass costs also handles shots with no reweighting
        without rerunning their UF decode.
        """
        first = super().decode_with_growth_costs(syndrome)
        second = self._second_pass_decoder(first.selected_edges)
        return first if second is None else second.decode_with_growth_costs(syndrome)
