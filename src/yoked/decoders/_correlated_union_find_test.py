import math

import numpy as np
import pymatching
import pytest
import stim

from yoked.decoders import CorrelatedUnionFindDecoder, DecodingGraph, InvalidSyndromeError, UnionFindDecoder


def _coupled_dem():
    # One mechanism connects two otherwise separate graph components. In the
    # second component, the marginal weights initially favor two boundaries.
    return stim.DetectorErrorModel('''
        error(0.01) D0 D1 L0 ^ D2 D3 L1
        error(0.2) D2
        error(0.2) D3
    ''')


def _assert_valid(graph, syndrome, result):
    reconstructed = np.zeros(graph.num_detectors, dtype=np.uint8)
    mask = 0
    for e in result.selected_edges:
        u, v, _, label = graph.edges[e]
        reconstructed[u] ^= 1
        if v is not None:
            reconstructed[v] ^= 1
        mask ^= label
    np.testing.assert_array_equal(reconstructed, syndrome)
    assert result.observable_mask == mask
    assert len(result.selected_edges) == len(set(result.selected_edges))
    assert set(result.selected_edges) <= set(result.forest_edges)


def test_two_passes_rebuild_clusters_and_return_the_second_correction(monkeypatch):
    decoder = CorrelatedUnionFindDecoder.from_dem(_coupled_dem())
    syndrome = np.ones(4, dtype=np.bool_)
    first = UnionFindDecoder(decoder.graph)._decode(syndrome)
    assert set(first.selected_edges) == {0, 2, 3}
    assert first.observable_mask == 1

    def unexpected_matching(*args, **kwargs):
        pytest.fail('Correlated UF must use UF for both decoding passes')

    for method in ['decode', 'decode_batch', 'decode_to_edges_array']:
        monkeypatch.setattr(pymatching.Matching, method, unexpected_matching)

    # Selecting edge 0 implies edge 1 with zero weight. Restarted growth joins
    # D2 to D3 before either reaches its boundary, changing the logical answer.
    second = decoder._decode(syndrome)
    assert set(second.selected_edges) == {0, 1}
    assert set(second.forest_edges) == {0, 1}
    assert second.observable_mask == 3
    _assert_valid(decoder.graph, syndrome, second)
    np.testing.assert_array_equal(decoder.decode(syndrome), [True, True])


def test_only_selected_edges_supply_evidence():
    graph = DecodingGraph(4, 1, [
        (0, 1, 0, 0), (2, 3, 3, 1), (2, None, 1, 0), (3, None, 1, 0),
    ])
    decoder = CorrelatedUnionFindDecoder(graph, correlation_rules=[(0, 1, 0)])
    syndrome = [0, 0, 1, 1]
    result = decoder._decode(syndrome)
    assert 0 in result.forest_edges and 0 not in result.selected_edges
    # The zero-cost edge is grown but unused; it must not trigger a discount.
    assert set(result.selected_edges) == {2, 3}
    assert result.observable_mask == 0
    _assert_valid(graph, syndrome, result)


def test_multiple_sources_keep_the_strongest_discount_and_never_raise_weights():
    graph = DecodingGraph(6, 1, [
        (0, 1, 1, 0), (2, 3, 1, 0), (4, 5, 3, 1),
        (4, None, 1, 0), (5, None, 1, 0),
    ])
    decoder = CorrelatedUnionFindDecoder(graph, correlation_rules=[(0, 2, 0.5), (1, 2, 5)])
    np.testing.assert_array_equal(decoder.decode([1] * 6), [True])
    np.testing.assert_array_equal(decoder.decode([0, 0, 1, 1, 1, 1]), [False])


def test_weights_reset_for_each_shot_and_after_an_invalid_syndrome():
    decoder = CorrelatedUnionFindDecoder.from_dem(_coupled_dem())
    graph = decoder.graph
    original_edges = graph.edges
    original_rules = decoder._correlation_rules
    syndromes = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    original = syndromes.copy()
    syndromes.flags.writeable = False
    expected = np.array([[1, 1], [0, 0], [0, 0], [1, 1]], dtype=np.bool_)
    np.testing.assert_array_equal(decoder.decode_batch(syndromes), expected)
    with pytest.raises(InvalidSyndromeError):
        decoder.decode([1, 0, 0, 0])
    np.testing.assert_array_equal(decoder.decode_batch(syndromes), expected)
    np.testing.assert_array_equal(syndromes, original)
    assert decoder.graph is graph and graph.edges == original_edges
    assert decoder._correlation_rules == original_rules
    empty = decoder.decode_batch(np.empty((0, 4), dtype=np.bool_))
    assert empty.shape == (0, 2) and empty.dtype == np.bool_


def test_dem_compiles_directional_rules_with_parallel_probability_merges():
    decoder = CorrelatedUnionFindDecoder.from_dem(stim.DetectorErrorModel('''
        error(0.1) D1 D0 ^ D2 ^ D3
        error(0.2) D0 D1
        error(0.05) D0 D1 ^ D2
        error(0.3) D3
    '''))
    # Marginals are XOR probabilities: A=.284, B=.14, C=.34.
    # Shared mechanisms have probabilities AB=.14, AC=.1, BC=.1.
    # These expected weights are calculated independently from those values.
    a, b, c = map(dict, decoder._correlation_rules)
    assert a == pytest.approx({1: math.log(0.144 / 0.14), 2: math.log(0.184 / 0.1)})
    assert b == pytest.approx({0: 0, 2: 0})
    assert c == pytest.approx({0: math.log(0.24 / 0.1), 1: math.log(0.24 / 0.1)})


def test_dem_repeat_shifts_and_ignored_zero_probability_instruction():
    decoder = CorrelatedUnionFindDecoder.from_dem(stim.DetectorErrorModel('''
        repeat 2 {
            error(0.1) D0 D1 ^ D2
            shift_detectors 3
        }
        error(0) D0 D1 D2
    '''))
    assert decoder._correlation_rules == (((1, 0.0),), ((0, 0.0),), ((3, 0.0),), ((2, 0.0),))


def test_dem_repeated_edge_across_components_is_rejected():
    with pytest.raises(ValueError, match='repeats a graph edge'):
        CorrelatedUnionFindDecoder.from_dem(stim.DetectorErrorModel('error(0.1) D0 D1 ^ D1 D0'))


def test_uncorrelated_dem_and_empty_graph_match_plain_uf():
    dem = stim.DetectorErrorModel('''
        error(0.1) D0 D1 L0
        error(0.2) D1 D2
        error(0.3) D2
    ''')
    decoder = CorrelatedUnionFindDecoder.from_dem(dem)
    assert not any(decoder._correlation_rules)
    plain = UnionFindDecoder(decoder.graph)
    for syndrome in [[1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]]:
        assert decoder._decode(syndrome) == plain._decode(syndrome)
    empty = CorrelatedUnionFindDecoder.from_dem(stim.DetectorErrorModel('logical_observable L1'))
    np.testing.assert_array_equal(empty.decode([]), [False, False])


def test_direct_rules_are_owned_and_can_address_parallel_edges():
    graph = DecodingGraph(3, 1, [(0, None, 1, 0), (1, 2, 2, 0), (1, 2, 3, 1)])
    rules = [[np.int64(0), np.int64(2), 0.0]]
    decoder = CorrelatedUnionFindDecoder(graph, correlation_rules=rules)
    rules[0][2] = 10.0
    rules.clear()
    np.testing.assert_array_equal(decoder.decode([1, 1, 1]), [True])


@pytest.mark.parametrize('rule', [
    (-1, 1, 0), (0, 2, 0), (0, 0, 0), (0.5, 1, 0), (True, 1, 0),
    (0, 1, -1), (0, 1, float('nan')), (0, 1, float('inf')),
])
def test_invalid_correlation_rules(rule):
    graph = DecodingGraph(3, 0, [(0, 1, 1, 0), (1, 2, 1, 0)])
    with pytest.raises(ValueError):
        CorrelatedUnionFindDecoder(graph, correlation_rules=[rule])


def test_growth_costs_follow_the_second_pass_when_its_prediction_changes():
    # Reproduce a case where returning the inherited first-pass growth state
    # would pair confidence with the wrong logical prediction.
    graph = DecodingGraph(4, 3, [
        (0, 1, 5, 1), (1, 2, 5, 2), (2, 3, 2, 4), (0, None, 4, 1), (3, None, 5, 2), (0, 2, 4, 3),
    ])
    decoder = CorrelatedUnionFindDecoder(graph, correlation_rules=[(5, 4, 2.0)])
    syndrome = [1, 1, 0, 1]
    adjusted = DecodingGraph(4, 3, [
        (0, 1, 5, 1), (1, 2, 5, 2), (2, 3, 2, 4), (0, None, 4, 1), (3, None, 2, 2), (0, 2, 4, 3),
    ])
    first = UnionFindDecoder(graph).decode_with_growth_costs(syndrome)
    expected = UnionFindDecoder(adjusted).decode_with_growth_costs(syndrome)
    result = decoder.decode_with_growth_costs(syndrome)
    assert result == expected
    assert result.observable_mask != first.observable_mask
    np.testing.assert_array_equal(decoder.decode(syndrome), [True, True, False])
    assert graph.edges[4][2] == 5  # Original weights must survive the shot.


def test_growth_costs_reset_between_correlated_and_uncorrelated_shots():
    decoder = CorrelatedUnionFindDecoder.from_dem(_coupled_dem())
    shots = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
    for syndrome in shots:
        result = decoder.decode_with_growth_costs(syndrome)
        fresh = CorrelatedUnionFindDecoder.from_dem(_coupled_dem()).decode_with_growth_costs(syndrome)
        assert result == fresh
        assert result.selected_edges == decoder.decode_to_edge_ids(syndrome)
        bits = [(result.observable_mask >> k) & 1 for k in range(2)]
        np.testing.assert_array_equal(bits, decoder.decode(syndrome))
    with pytest.raises(InvalidSyndromeError):
        decoder.decode_with_growth_costs([1, 0, 0, 0])
    assert decoder.decode_with_growth_costs(shots[0]) == fresh
