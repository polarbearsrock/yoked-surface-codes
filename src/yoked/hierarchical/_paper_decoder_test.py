"""Check native L1 identity, complementary classes, and the complete parity flow.

The small hand-built model has analytically known class costs. Circuit fixtures
exercise real SI1000 correlations; independent native solves and exhaustive
six-bit outer searches provide checks separate from the production pipeline.
"""
import itertools
import math

import numpy as np
import pymatching
import pytest
import stim

from yoked.hierarchical import (
    CorrelatedMatchingGapDecoder, PaperHierarchicalDecoder, SinterPaperHierarchicalDecoder,
    mwpm_outer_log_odds_batch, mwpm_outer_map_batch,
)
from yoked.hierarchical._fixtures import yoked_fixture
from yoked.hierarchical._patch_graphs import PatchGraphs


def _toy_patch():
    # Two separate detector chains with opposite logical and plain boundaries.
    # One shared mechanism makes the two logical-boundary edges correlated.
    dem = stim.DetectorErrorModel('''
        error(0.45) D0 D4 L0 ^ D2 D5 L1
        error(0.10) D0 D1
        error(0.30) D1
        error(0.10) D2 D3
        error(0.30) D3
    ''')
    return PatchGraphs.from_yoked_dem(dem, num_patches=1)[0]


def test_gap_compares_classes_on_the_same_frozen_correlated_graph():
    decoder = CorrelatedMatchingGapDecoder(_toy_patch())
    result = decoder.decode([1, 0, 1, 0])
    # First-pass logical edges imply one another with conditional probability 1,
    # clipped to 1/2 by PyMatching. Both become zero weight. A complementary
    # logical class must instead take its two-edge route to the plain boundary.
    path = math.log(0.9 / 0.1) + math.log(0.7 / 0.3)
    np.testing.assert_array_equal(result.reference, [1, 1])
    np.testing.assert_allclose(result.complementary_gap, [path, path], atol=1e-6, rtol=0)
    np.testing.assert_allclose(result.forced_weights, [[2*path, path], [path, 0]], atol=1e-6, rtol=0)
    assert result.native_weight == 0
    for array in (result.reference, result.complementary_gap, result.forced_weights):
        assert not array.flags.writeable


def test_all_toy_syndromes_match_native_and_fresh_decoder_state():
    patch = _toy_patch()
    decoder = CorrelatedMatchingGapDecoder(patch)
    native = pymatching.Matching.from_detector_error_model(patch.local_dem, enable_correlations=True)
    for syndrome in itertools.product((0, 1), repeat=4):
        result = decoder.decode(syndrome)
        prediction, weight = native.decode(syndrome, enable_correlations=True, return_weight=True)
        np.testing.assert_array_equal(result.reference, prediction)
        assert result.forced_weights[tuple(prediction)] == pytest.approx(weight, abs=1e-10)
        fresh = CorrelatedMatchingGapDecoder(patch).decode(syndrome)
        np.testing.assert_array_equal(result.complementary_gap, fresh.complementary_gap)
        np.testing.assert_array_equal(result.reference, fresh.reference)


@pytest.mark.parametrize('probability', [0.1, 0.5])
def test_uncorrelated_model_and_exact_zero_gap_ties(probability):
    # Equal-weight edges in each sector: the logical boundary costs one edge,
    # while the other class costs two. At p=1/2 all weights and gaps are zero.
    dem = stim.DetectorErrorModel('\n'.join(
        f'error({probability}) {targets}' for targets in
        ('D0 D4 L0', 'D0 D1', 'D1', 'D2 D5 L1', 'D2 D3', 'D3')))
    patch = PatchGraphs.from_yoked_dem(dem, num_patches=1)[0]
    result = CorrelatedMatchingGapDecoder(patch).decode([1, 0, 1, 0])
    expected_gap = math.log((1 - probability) / probability)
    np.testing.assert_allclose(result.complementary_gap, [expected_gap, expected_gap], atol=1e-10, rtol=0)
    native = pymatching.Matching.from_detector_error_model(patch.local_dem, enable_correlations=True)
    np.testing.assert_array_equal(result.reference, native.decode([1, 0, 1, 0], enable_correlations=True))


@pytest.fixture(scope='module')
def circuit_case():
    fixture = yoked_fixture(shots=48, p=0.01)
    decoder = PaperHierarchicalDecoder(fixture.dem)
    return fixture, decoder, decoder.decode_with_gaps_batch(fixture.detectors)


def test_circuit_references_are_native_correlated_predictions(circuit_case):
    fixture, decoder, result = circuit_case
    different_from_plain = 0
    for i, patch in enumerate(decoder.patches):
        native = pymatching.Matching.from_detector_error_model(patch.local_dem, enable_correlations=True)
        local = patch.local_syndromes(fixture.detectors).astype(np.uint8)
        expected, weights = native.decode_batch(local, enable_correlations=True, return_weights=True)
        np.testing.assert_array_equal(result.reference[:, 2*i:2*i+2], expected)
        forced = result.forced_weights[:, i]
        np.testing.assert_allclose(forced[np.arange(len(local)), expected[:, 0], expected[:, 1]],
                                   weights, rtol=0, atol=1e-10)
        different_from_plain += np.count_nonzero(expected != native.decode_batch(local))
    # A fixture where correlations change the actual reference, not just the gap.
    assert different_from_plain > 0


def test_outer_matches_independent_six_bit_search(circuit_case):
    _, _, result = circuit_case
    patterns = np.array([[(value >> bit) & 1 for bit in range(6)] for value in range(64)], dtype=bool)
    for shot in range(len(result.prediction)):
        for sector in range(2):
            weights = result.complementary_gap[shot, sector::2]
            costs = patterns @ weights
            costs[patterns.sum(axis=1) % 2 != result.sigma[shot, sector]] = np.inf
            # Enumeration is only an oracle in this test. Production runs MWPM.
            best = np.flatnonzero(costs <= costs.min() + 1e-9)[0]
            np.testing.assert_array_equal(result.residual[shot, sector::2], patterns[best])
    np.testing.assert_array_equal(result.prediction, result.reference ^ result.residual)
    for array in result.arrays().values():
        assert not array.flags.writeable


def test_yoke_information_is_only_used_by_l2(circuit_case):
    fixture, decoder, result = circuit_case
    detectors = fixture.detectors[:3].copy()
    detectors[:, list(decoder.patches.yoke_detector_ids)] ^= True
    changed = decoder.decode_with_gaps_batch(detectors)
    np.testing.assert_array_equal(changed.reference, result.reference[:3])
    np.testing.assert_array_equal(changed.complementary_gap, result.complementary_gap[:3])
    np.testing.assert_array_equal(changed.sigma, ~result.sigma[:3])
    np.testing.assert_array_equal(changed.prediction.reshape(3, 6, 2).sum(axis=1) % 2,
                                  detectors[:, list(decoder.patches.yoke_detector_ids)])


def test_raw_gap_weights_avoid_probability_clipping_and_underflow():
    # Both large gaps would saturate an isotonic probability clip, erasing which
    # patch is less trustworthy; exp(-1000) also underflows in float64.
    result = mwpm_outer_log_odds_batch([[1001.0, 1000.0]], [1])
    assert result.patterns.tolist() == [[False, True]]
    assert result.tied.tolist() == [False]


def test_direct_weights_agree_with_probability_interface():
    weights = np.array([[0., 1., 2., 3.], [-2., 1., -1., 0.], [10., 5., 3., 1.]])
    parity = np.array([1, 0, 1])
    q = 1 / (1 + np.exp(weights))
    direct = mwpm_outer_log_odds_batch(weights, parity)
    probability = mwpm_outer_map_batch(q, parity)
    np.testing.assert_array_equal(direct.patterns, probability.patterns)
    np.testing.assert_array_equal(direct.tied, probability.tied)


def test_scaling_and_bit_packed_sinter_adapter(circuit_case):
    fixture, _, result = circuit_case
    scaled = PaperHierarchicalDecoder(fixture.dem, gap_scale=0.9).decode_with_gaps_batch(fixture.detectors[:4])
    np.testing.assert_array_equal(scaled.reference, result.reference[:4])
    np.testing.assert_allclose(scaled.outer_weights, result.complementary_gap[:4] * 0.9)
    np.testing.assert_array_equal(scaled.prediction, result.prediction[:4])
    compiled = SinterPaperHierarchicalDecoder().compile_decoder_for_dem(dem=fixture.dem)
    actual = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.packbits(fixture.detectors[:4], axis=1, bitorder='little'))
    np.testing.assert_array_equal(actual, np.packbits(result.prediction[:4], axis=1, bitorder='little'))
    empty = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.zeros((0, (fixture.dem.num_detectors + 7) // 8), dtype=np.uint8))
    assert empty.shape == (0, 2)


def test_rejects_unvalidated_native_version_and_weight_disagreement(monkeypatch):
    decoder = CorrelatedMatchingGapDecoder(_toy_patch())
    native_decode = decoder.matching.decode

    def wrong_weight(*args, **kwargs):
        prediction, weight = native_decode(*args, **kwargs)
        return prediction, weight + 1

    monkeypatch.setattr(decoder.matching, 'decode', wrong_weight)
    with pytest.raises(ValueError, match='disagree with native'):
        decoder.decode([1, 0, 1, 0])
    monkeypatch.setattr(pymatching, '__version__', '2.5.0')
    with pytest.raises(RuntimeError, match='2.4.x'):
        CorrelatedMatchingGapDecoder(_toy_patch())


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf'), True])
def test_invalid_scale(circuit_case, scale):
    fixture, _, _ = circuit_case
    with pytest.raises(ValueError, match='gap_scale'):
        PaperHierarchicalDecoder(fixture.dem, gap_scale=scale)


def test_invalid_syndromes(circuit_case):
    fixture, decoder, _ = circuit_case
    with pytest.raises(ValueError, match='binary detectors'):
        decoder.decode_batch(np.full_like(fixture.detectors[:1], 0.5, dtype=float))
    with pytest.raises(ValueError, match='local detector bits'):
        decoder.decoders[0].decode([0.5] * decoder.patches[0].num_detectors)
    with pytest.raises(ValueError, match='finite'):
        mwpm_outer_log_odds_batch([[np.inf]], [1])
