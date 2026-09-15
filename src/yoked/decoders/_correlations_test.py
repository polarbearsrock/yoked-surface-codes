import math

import pytest
import stim

from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)


def test_apply_rules_keeps_the_lowest_weight_and_never_raises_one():
    graph = DecodingGraph(3, 0, [(0, 1, 1.0, 0), (1, 2, 2.0, 0), (2, None, 3.0, 0)])
    rules = index_rules_by_source(graph, [(0, 1, 0.5), (2, 1, 1.5), (0, 2, 4.0)])
    weights = [1.0, 2.0, 3.0]
    assert apply_correlation_rules(weights, rules, []) is None
    # Sources 0 and 2 both support edge 1; the lower implied weight wins.
    # Source 0 also names edge 2 with weight 4.0 > 3.0, which must not raise it.
    assert apply_correlation_rules(weights, rules, [0, 2]) == [1.0, 0.5, 3.0]
    assert apply_correlation_rules(weights, rules, [2]) == [1.0, 1.5, 3.0]
    assert weights == [1.0, 2.0, 3.0]


def test_index_rules_validates_and_preserves_order():
    graph = DecodingGraph(2, 0, [(0, 1, 1.0, 0), (1, None, 1.0, 0)])
    with pytest.raises(ValueError, match='out-of-range'):
        index_rules_by_source(graph, [(0, 5, 0.0)])
    with pytest.raises(ValueError, match='different edges'):
        index_rules_by_source(graph, [(0, 0, 0.0)])
    with pytest.raises(ValueError, match='nonnegative'):
        index_rules_by_source(graph, [(0, 1, -1.0)])
    assert index_rules_by_source(graph, [(1, 0, 0.25), (1, 0, 0.5)]) == ((), ((0, 0.25), (0, 0.5)))


def test_rules_from_dem_reproduce_the_documented_implied_weights():
    dem = stim.DetectorErrorModel('''
        error(0.1) D1 D0 ^ D2 ^ D3
        error(0.2) D0 D1
        error(0.05) D0 D1 ^ D2
        error(0.3) D3
    ''')
    graph = DecodingGraph.from_dem(dem)
    rules = {(source, target): weight for source, target, weight in correlation_rules_from_dem(graph, dem)}
    # Edge 0 = D0D1 (marginal .284), edge 1 = D2 (.14), edge 2 = D3 (.34).
    # Shared mechanism probabilities: 01 -> .14, 02 -> .1, 12 -> .1.
    assert rules[(0, 1)] == pytest.approx(math.log(0.144 / 0.14))
    assert rules[(0, 2)] == pytest.approx(math.log(0.184 / 0.1))
    assert rules[(1, 0)] == 0 and rules[(1, 2)] == 0
    assert rules[(2, 0)] == pytest.approx(math.log(0.24 / 0.1))
    assert rules[(2, 1)] == pytest.approx(math.log(0.24 / 0.1))


def test_rules_from_dem_reject_a_repeated_edge_across_components():
    dem = stim.DetectorErrorModel('error(0.1) D0 D1 ^ D1 D0')
    with pytest.raises(ValueError, match='repeats a graph edge'):
        correlation_rules_from_dem(DecodingGraph.from_dem(dem), dem)
