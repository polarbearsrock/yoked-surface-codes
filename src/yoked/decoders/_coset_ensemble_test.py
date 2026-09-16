import itertools

import numpy as np
import pytest
import stim

from yoked.decoders import CorrelatedUnionFindDecoder, CosetEnsembleDecoder, DecodingGraph
from yoked.decoders._coset_ensemble import EnsembleDecodeResult, ForestCandidate, _cluster_graph
from yoked.decoders._union_find import InvalidSyndromeError, UnionFindDecoder


def _response(graph, edges):
    parity = np.zeros(graph.num_detectors, dtype=bool)
    mask = 0
    for e in edges:
        u, v, _, label = graph.edges[e]
        parity[u] ^= True
        if v is not None:
            parity[v] ^= True
        mask ^= label
    return parity, mask


def _check(graph, syndrome, result):
    for candidate in (result.baseline, *result.candidates):
        assert len(set(candidate.selected_edges)) == len(candidate.selected_edges)
        parity, mask = _response(graph, candidate.selected_edges)
        np.testing.assert_array_equal(parity, syndrome)
        assert candidate.observable_mask == mask
        assert candidate.weight == pytest.approx(sum(graph.edges[e][2] for e in candidate.selected_edges))


@pytest.mark.parametrize('boundary', [False, True])
def test_exhaustive_small_graph_syndromes_and_kernel_certificate(boundary):
    edges = [(0, 1, 0, 1), (1, 2, 0, 2), (2, 0, 0, 0), (0, 1, 0, 4)]
    if boundary:
        edges += [(0, None, 0, 8), (2, None, 0, 0)]
    graph = DecodingGraph(3, 4, edges)
    decoder = CosetEnsembleDecoder(graph, candidates=32)
    kernel_masks = {_response(graph, [e for e in range(len(edges)) if bits >> e & 1])[1]
                    for bits in range(1 << len(edges))
                    if not _response(graph, [e for e in range(len(edges)) if bits >> e & 1])[0].any()}
    for bits in itertools.product((0, 1), repeat=3):
        syndrome = np.array(bits, dtype=bool)
        if not boundary and syndrome.sum() % 2:
            with pytest.raises(InvalidSyndromeError):
                decoder.decode_ensemble(syndrome)
            continue
        result = decoder.decode_ensemble(syndrome)
        _check(graph, syndrome, result)
        assert 2 ** result.logical_rank == len(kernel_masks)


def test_samples_different_logical_cosets_and_parallel_boundary_choices():
    graph = DecodingGraph(1, 70, [(0, None, 0, 0), (0, None, 0, 1 << 69)])
    result = CosetEnsembleDecoder(graph, candidates=32).decode_ensemble(np.array([1]))
    _check(graph, [1], result)
    assert result.logical_rank == 1
    assert {c.observable_mask for c in result.candidates} == {0, 1 << 69}
    assert {c.selected_edges for c in result.candidates} == {(0,), (1,)}


def test_cycles_generate_distinct_forests_and_corrections():
    graph = DecodingGraph(3, 1, [(0, 1, 0, 1), (1, 2, 0, 0), (2, 0, 0, 0)])
    result = CosetEnsembleDecoder(graph, candidates=32).decode_ensemble(np.array([1, 1, 0]))
    _check(graph, [1, 1, 0], result)
    assert {c.observable_mask for c in result.candidates} == {0, 1}
    assert result.choose('size').selected_edges == (0,)


def test_rank_zero_clustering_excludes_a_different_valid_logical_path():
    graph = DecodingGraph(3, 1, [(0, 1, 1, 1), (1, 2, 2, 0), (2, 0, 2, 0)])
    syndrome = np.array([1, 1, 0])
    result = CosetEnsembleDecoder(graph).decode_ensemble(syndrome)
    assert result.logical_rank == 0
    assert result.internal_edges == 1
    assert {c.observable_mask for c in result.candidates} == {1}
    parity, alternative_mask = _response(graph, (1, 2))
    np.testing.assert_array_equal(parity, syndrome)
    assert alternative_mask == 0


def test_boundary_contraction_never_connects_separate_clusters():
    graph = DecodingGraph(2, 1, [(0, None, 1, 0), (1, None, 1, 1), (0, 1, 10, 0)])
    syndrome = np.array([1, 1])
    _, growth = UnionFindDecoder(graph)._decode_state(syndrome)
    roots = [growth.find(v) for v in range(len(graph.adjacency))]
    adjacency, components, count = _cluster_graph(graph, roots)
    assert count == 2
    assert len(components) == 2
    assert len([v for v in adjacency if v < 0]) == 2
    result = CosetEnsembleDecoder(graph).decode_ensemble(syndrome)
    _check(graph, syndrome, result)
    assert all(c.selected_edges == (0, 1) for c in result.candidates)


def test_logical_rank_counts_independent_masks_across_components():
    graph = DecodingGraph(2, 1, [(0, None, 0, 0), (0, None, 0, 1),
                               (1, None, 0, 0), (1, None, 0, 1)])
    result = CosetEnsembleDecoder(graph).decode_ensemble(np.array([1, 1]))
    _check(graph, [1, 1], result)
    # Both independent graph cycles carry the same logical bit.
    assert result.logical_rank == 1


def test_small_circuit_with_correlations_reproduces_every_candidate_syndrome():
    circuit = stim.Circuit.generated('surface_code:rotated_memory_x', distance=3, rounds=4,
                                     after_clifford_depolarization=.01)
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes = circuit.compile_detector_sampler(seed=918).sample(40)
    ensemble = CosetEnsembleDecoder.from_dem(dem, correlated=True, candidates=8)
    ordinary = CorrelatedUnionFindDecoder.from_dem(dem)
    before = syndromes.copy()
    for syndrome in syndromes:
        result = ensemble.decode_ensemble(syndrome)
        assert result.baseline.observable_mask == ordinary._decode(syndrome).observable_mask
        for candidate in result.candidates:
            parity, mask = _response(ensemble.graph, candidate.selected_edges)
            np.testing.assert_array_equal(parity, syndrome)
            assert mask == candidate.observable_mask
    np.testing.assert_array_equal(before, syndromes)


def test_vote_uses_only_minimum_size_and_counts_repeated_samples():
    a = ForestCandidate((0,), 7, 5.)
    b = ForestCandidate((1,), 3, 1.)
    c = ForestCandidate((2, 3), 8, .5)
    result = EnsembleDecodeResult(a, (b, a, a, c, c, c), 1, 4)
    assert result.choose('size') == a
    assert result.choose('weight') == c
    assert result.choose('size', count=2) == b  # First sampled tied coset.
    with pytest.raises(ValueError):
        result.choose(count=0)
    with pytest.raises(ValueError):
        result.choose('unknown')


def test_weight_ties_and_prefixes():
    a = ForestCandidate((0,), 1, 2.)
    b = ForestCandidate((1,), 0, 2. + 1e-13)
    result = EnsembleDecodeResult(a, (a, b, b), 1, 2)
    assert result.choose('weight') == b
    assert result.choose('weight', count=1) == a


def test_reproducible_prefixes_batch_order_and_no_input_mutation():
    graph = DecodingGraph(3, 1, [(0, 1, 0, 1), (1, 2, 0, 0), (2, 0, 0, 0)])
    small = CosetEnsembleDecoder(graph, candidates=4)
    large = CosetEnsembleDecoder(graph, candidates=24)
    batch = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]])
    saved = batch.copy()
    assert small.decode_ensemble(batch[0]).candidates == large.decode_ensemble(batch[0]).candidates[:4]
    np.testing.assert_array_equal(large.decode_batch(batch), large.decode_batch(batch[::-1])[::-1])
    np.testing.assert_array_equal(batch, saved)
    assert not large.decode(batch[2]).any()
    assert large.decode_batch(np.empty((0, 3))).shape == (0, 1)


def test_correlated_baseline_is_unchanged_and_weights_do_not_leak():
    dem = stim.DetectorErrorModel('''
        error(0.1) D0 L0 ^ D1 L1
        error(0.05) D0 D1
        error(0.01) D0
        error(0.01) D1
    ''')
    decoder = CosetEnsembleDecoder.from_dem(dem, correlated=True)
    baseline = CorrelatedUnionFindDecoder.from_dem(dem)
    original_edges = decoder.graph.edges
    for bits in itertools.product((0, 1), repeat=2):
        syndrome = np.array(bits)
        result = decoder.decode_ensemble(syndrome)
        assert result.baseline.observable_mask == baseline._decode(syndrome).observable_mask
        assert set(result.baseline.selected_edges) == set(baseline.decode_to_edge_ids(syndrome))
        for candidate in result.candidates:
            parity, mask = _response(decoder.graph, candidate.selected_edges)
            np.testing.assert_array_equal(parity, syndrome)
            assert mask == candidate.observable_mask
    assert decoder.graph.edges == original_edges


@pytest.mark.parametrize('kwargs', [{'candidates': 0}, {'candidates': True}, {'candidates': 1.5},
                                   {'seed': -1}, {'score': 'unknown'}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        CosetEnsembleDecoder(DecodingGraph(0, 0, []), **kwargs)


@pytest.mark.parametrize('syndrome', [[2], [np.nan], [[1]], []])
def test_invalid_syndromes(syndrome):
    with pytest.raises(ValueError):
        CosetEnsembleDecoder(DecodingGraph(1, 0, [(0, None, 1, 0)])).decode(np.array(syndrome))
