"""Native prediction identity, independent score oracle, and state reuse checks.

Set YOKED_MPP_BUILD to a build made by tools/build_mpp. Without that optional
dependency these tests skip; the baseline's suite remains independently usable.
"""
import itertools
import math
import os

import numpy as np
import pymatching
import pytest

from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._mpp import MppHierarchicalDecoder, load_mpp_native
from yoked.hierarchical._paper_decoder import PaperHierarchicalDecoder
from yoked.hierarchical._patch_graphs import PatchGraphs
import stim

pytestmark = pytest.mark.skipif(not os.environ.get('YOKED_MPP_BUILD'), reason='optional native MPP build not selected')


def toy_dem(probability):
    # Three-edge paths between opposite boundaries, one per independent sector.
    return '\n'.join(f'error({probability}) {targets}' for targets in
                     ('D0 L0', 'D0 D1', 'D1', 'D2 L1', 'D2 D3', 'D3'))


@pytest.mark.parametrize('method', ['dijkstra', 'incremental'])
@pytest.mark.parametrize('mode', ['none', 'cross', 'all'])
@pytest.mark.parametrize('probability', [0.1, 0.5])
def test_analytic_score_all_syndromes_and_zero_weights(method, mode, probability):
    module, _ = load_mpp_native()
    text = toy_dem(probability)
    native = module.SoftDecoder(text, mode, method)
    shots = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    prediction, scores, weights = native.decode_batch(shots, verify=True)
    expected = 3 * math.log((1-probability) / probability)
    np.testing.assert_allclose(scores[0], [expected, expected], atol=1e-6, rtol=0)
    # Two decodes in one object must behave like two newly constructed objects.
    again = native.decode_batch(shots, verify=True)
    for actual, previous in zip(again, (prediction, scores, weights)):
        np.testing.assert_array_equal(actual, previous)
    hard, no_scores, hard_weights = native.decode_batch(shots, soft=False)
    assert no_scores.shape == (16, 0)
    np.testing.assert_array_equal(hard, prediction)
    np.testing.assert_array_equal(hard_weights, weights)
    assert native.radius_gather_mismatches(shots) == 0
    if mode != 'cross':
        matching = pymatching.Matching.from_detector_error_model(
            stim.DetectorErrorModel(text), enable_correlations=mode == 'all')
        expected, expected_weights = matching.decode_batch(shots, return_weights=True,
                                                           enable_correlations=mode == 'all')
        np.testing.assert_array_equal(prediction, expected)
        np.testing.assert_allclose(weights, expected_weights, atol=1e-10, rtol=0)


@pytest.fixture(scope='module', params=[3, 5, 7])
def circuit_sample(request):
    sample = SampleSet.sample(CircuitParameters(distance=request.param, rounds=12, p=0.003),
                              shots=48, seed=934 + request.param)
    detectors, _ = sample.rows(np.arange(sample.shots))
    patches = PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text), num_patches=6)
    return patches[0], detectors


@pytest.mark.parametrize('mode', ['none', 'cross', 'all'])
def test_circuit_sparse_and_dense_syndromes_against_oracle(circuit_sample, mode):
    patch, detectors = circuit_sample
    local = patch.local_syndromes(detectors).astype(np.uint8)
    # Random syndromes stress large clusters/blossoms beyond the sampled noise
    # distribution. Both sectors have boundaries, so these are decodable.
    dense = np.random.default_rng(19).integers(0, 2, size=(8, patch.num_detectors), dtype=np.uint8)
    local = np.concatenate((local, dense, np.zeros((1, patch.num_detectors), dtype=np.uint8)))
    module, _ = load_mpp_native()
    answers = []
    for method in ('dijkstra', 'incremental'):
        decoder = module.SoftDecoder(str(patch.local_dem), mode, method)
        answer = decoder.decode_batch(local, verify=True)
        assert decoder.radius_gather_mismatches(local) == 0
        # Reversed and non-contiguous batches exercise resets and numpy strides.
        repeated = decoder.decode_batch(local[::-1], verify=True)
        for forward, reverse in zip(answer, repeated):
            np.testing.assert_array_equal(forward, reverse[::-1])
        answers.append(answer)
    for first, second in zip(*answers):
        np.testing.assert_array_equal(first, second)
    if mode != 'cross':
        matching = pymatching.Matching.from_detector_error_model(patch.local_dem, enable_correlations=mode == 'all')
        expected, weights = matching.decode_batch(local, enable_correlations=mode == 'all', return_weights=True)
        np.testing.assert_array_equal(answers[0][0], expected)
        np.testing.assert_allclose(answers[0][2], weights, atol=1e-10, rtol=0)


def test_hierarchy_changes_only_confidence_and_excludes_yokes():
    sample = SampleSet.sample(CircuitParameters(distance=3, rounds=12, p=0.01), shots=24, seed=47)
    detectors, _ = sample.rows(np.arange(sample.shots))
    dem = stim.DetectorErrorModel(sample.dem_text)
    decoder = MppHierarchicalDecoder(dem, verify=True)
    result = decoder.decode_with_scores_batch(detectors)
    baseline = PaperHierarchicalDecoder(dem).decode_with_gaps_batch(detectors)
    np.testing.assert_array_equal(result.reference, baseline.reference)
    # Ensure this test actually exercises different confidence information.
    assert np.any(np.abs(result.cluster_score - baseline.complementary_gap) > 1e-5)
    changed = detectors.copy()
    changed[:, decoder.patches.yoke_detector_ids] ^= True
    flipped = decoder.decode_with_scores_batch(changed)
    np.testing.assert_array_equal(flipped.reference, result.reference)
    np.testing.assert_array_equal(flipped.cluster_score, result.cluster_score)
    np.testing.assert_array_equal(flipped.sigma, ~result.sigma)
    np.testing.assert_array_equal(result.prediction.reshape(24, 6, 2).sum(axis=1) % 2,
                                  detectors[:, decoder.patches.yoke_detector_ids])
    assert all(not array.flags.writeable for array in result.arrays().values())
    empty = decoder.decode_with_scores_batch(detectors[:0])
    assert empty.prediction.shape == (0, 12)
    with pytest.raises(ValueError, match='binary detectors'):
        decoder.decode_batch(np.full(detectors.shape, 0.5))


def test_native_input_guards_and_reuse_after_rejection():
    module, _ = load_mpp_native()
    decoder = module.SoftDecoder(toy_dem(0.1), 'all', 'incremental')
    valid = np.zeros((2, 4), dtype=np.uint8)
    expected = decoder.decode_batch(valid)
    for bad in (np.zeros(4, dtype=np.uint8), np.zeros((2, 5), dtype=np.uint8), valid + 2):
        with pytest.raises(ValueError):
            decoder.decode_batch(bad)
        with pytest.raises(ValueError):
            decoder.radius_gather_mismatches(bad)
    with pytest.raises(TypeError):
        decoder.decode_batch(valid.astype(float))
    for actual, previous in zip(decoder.decode_batch(valid, verify=True), expected):
        np.testing.assert_array_equal(actual, previous)
    with pytest.raises(ValueError, match='method'):
        module.SoftDecoder(toy_dem(0.1), 'all', 'unknown')
    with pytest.raises(ValueError, match='Correlation'):
        module.SoftDecoder(toy_dem(0.1), 'unknown', 'dijkstra')
    with pytest.raises(ValueError, match='nonnegative'):
        module.SoftDecoder(toy_dem(0.9), 'none', 'dijkstra')


def test_missing_build_has_actionable_error(monkeypatch):
    monkeypatch.delenv('YOKED_MPP_BUILD', raising=False)
    with pytest.raises(RuntimeError, match='tools/build_mpp'):
        load_mpp_native()
