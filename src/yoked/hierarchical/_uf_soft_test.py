"""Correctness and work boundaries of the UF-only confidence candidates."""
import itertools
import math

import numpy as np
import pymatching
import pytest

from yoked.decoders._correlated_union_find import CorrelatedUnionFindDecoder
from yoked.decoders._correlations import apply_correlation_rules
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._fixtures import yoked_fixture
from yoked.hierarchical._uf_soft import UFSoftDecoder, _reweighted, quantize_gap


def test_two_forced_decodes_equal_all_four_class_costs_and_satisfy_checks():
    fx = yoked_fixture(shots=12, p=0.01)
    patch = fx.patches[0]
    decoder = UFSoftDecoder(patch)
    for syndrome in patch.local_syndromes(fx.detectors):
        first = UnionFindDecoder(patch.graph).decode_to_edge_ids(syndrome)
        adjusted = apply_correlation_rules(decoder._weights, decoder._rules, first)
        for graph in [patch.check_graph, _reweighted(patch.check_graph, adjusted)
                      if adjusted is not None else patch.check_graph]:
            uf = UnionFindDecoder(graph)
            costs = decoder.forced_costs(syndrome, uf)
            for cx, cz in itertools.product(range(2), repeat=2):
                forced = np.concatenate([syndrome, [cx, cz]])
                edges = uf.decode_to_edge_ids(forced)
                reproduced = np.zeros(graph.num_detectors, dtype=bool)
                mask = 0
                for e in edges:
                    u, v, _, flips = graph.edges[e]
                    reproduced[u] ^= True
                    if v is not None:
                        reproduced[v] ^= True
                    mask ^= flips
                np.testing.assert_array_equal(reproduced, forced)
                assert mask == cx + 2 * cz
                weight = math.fsum(graph.edges[e][2] for e in edges)
                assert weight == pytest.approx(costs[0, cx] + costs[1, cz], abs=1e-9)


def test_fixed_reference_correlated_prediction_and_no_matching_solves(monkeypatch):
    fx = yoked_fixture(shots=20, p=0.01)
    patch = fx.patches[0]
    decoder = UFSoftDecoder(patch)
    plain = UnionFindDecoder(patch.graph)
    correlated = CorrelatedUnionFindDecoder.from_dem(patch.local_dem)

    def forbidden(*args, **kwargs):
        raise AssertionError('Matching solve used for UF confidence')

    for method in ('decode', 'decode_batch', 'decode_to_edges_array'):
        monkeypatch.setattr(pymatching.Matching, method, forbidden)
    fired = 0
    for syndrome in patch.local_syndromes(fx.detectors):
        result = decoder.decode(syndrome)
        np.testing.assert_array_equal(result.reference, plain.decode(syndrome))
        np.testing.assert_array_equal(result.correlated_prediction, correlated.decode(syndrome))
        costs = result.forced_costs
        sign = 1 - 2 * result.reference.astype(int)
        np.testing.assert_allclose(result.scores[2:], sign * (costs[:, :, 1] - costs[:, :, 0]))
        assert (np.abs(result.scores[:2]) <= decoder.gap_cap).all()
        assert (result.states >= 0).all()
        if not result.rules_fired:
            np.testing.assert_array_equal(result.scores[0], result.scores[1])
            np.testing.assert_array_equal(result.scores[2], result.scores[3])
        fired += result.rules_fired
    assert fired > 0


def test_bounded_search_is_censoring_with_actual_early_stopping():
    fx = yoked_fixture(shots=16)
    patch = fx.patches[0]
    decoder = ClusterGapUnionFindDecoder(patch.graph)
    uf = UnionFindDecoder(patch.graph)
    avoided = 0
    for syndrome in patch.local_syndromes(fx.detectors):
        costs = uf.decode_with_growth_costs(syndrome).remaining_costs
        full, full_states = decoder.gaps_from_costs(costs)
        for cap in (0, 0.1, math.log(100), 100):
            bounded, states = decoder.gaps_from_costs(costs, max_gap=cap)
            np.testing.assert_allclose(bounded, np.minimum(full, cap))
            assert (states <= full_states).all()
            avoided += int((states < full_states).sum())
    assert avoided > 0


@pytest.mark.parametrize('cap', [-1, float('nan'), float('inf'), 0])
def test_invalid_cap_refused(cap):
    with pytest.raises(ValueError, match='gap_cap'):
        UFSoftDecoder(yoked_fixture(shots=1).patches[0], gap_cap=cap)


def test_output_quantization_has_sixteen_levels_and_saturates():
    np.testing.assert_array_equal(quantize_gap([-100, -16, -0.01, 0, 1.99, 2, 14, 100]),
                                  [-8, -8, -1, 0, 0, 1, 7, 7])
    assert len(np.unique(quantize_gap(np.arange(-100, 100)))) == 16
