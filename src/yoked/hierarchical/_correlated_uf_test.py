"""Checks for final-pass confidence, reference framing, and plain MWPM at L2."""
import math

import numpy as np
import pymatching
import pytest
import stim

from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._correlated_uf import CorrelatedUFHierarchicalDecoder
from yoked.hierarchical._outer_decoder import exact_outer_map_batch, frame_adjusted_syndrome


@pytest.fixture(scope='module')
def sample():
    return SampleSet.sample(CircuitParameters(distance=3, rounds=12, p=0.01), shots=24, seed=47)


def test_final_reference_matches_correlated_uf_and_l2_matches_exhaustive_map(sample, monkeypatch):
    decoder = CorrelatedUFHierarchicalDecoder(stim.DetectorErrorModel(sample.dem_text))
    detectors, _ = sample.rows(np.arange(sample.shots))
    calls = []
    matching_decode = pymatching.Matching.decode

    def counted_decode(self, *args, **kwargs):
        calls.append(self.num_detectors)
        return matching_decode(self, *args, **kwargs)

    monkeypatch.setattr(pymatching.Matching, 'decode', counted_decode)
    result = decoder.decode_with_gaps_batch(detectors)
    # L1 never calls matching: exactly one outer solve per shot and sector.
    assert len(calls) == 2 * sample.shots
    plain = np.zeros_like(result.reference)
    for index, local_decoder in enumerate(decoder.decoders):
        local = decoder.patches[index].local_syndromes(detectors)
        np.testing.assert_array_equal(result.reference[:, 2*index:2*index+2], local_decoder.decode_batch(local))
        plain[:, 2*index:2*index+2] = UnionFindDecoder(local_decoder.graph).decode_batch(local)
    assert np.any(plain != result.reference), 'Fixture must exercise a changed second-pass reference'
    for sector in range(2):
        probabilities = 1 / (1 + np.exp(result.cluster_gap[:, sector::2]))
        exhaustive = exact_outer_map_batch(probabilities, result.sigma[:, sector])
        np.testing.assert_array_equal(result.residual[:, sector::2], exhaustive.patterns)
    np.testing.assert_array_equal(result.sigma, frame_adjusted_syndrome(
        detectors[:, decoder.patches.yoke_detector_ids], result.reference))


def test_yokes_do_not_enter_l1_and_scores_are_not_capped(sample):
    decoder = CorrelatedUFHierarchicalDecoder(stim.DetectorErrorModel(sample.dem_text))
    detectors, _ = sample.rows(np.arange(4))
    result = decoder.decode_with_gaps_batch(detectors)
    changed = detectors.copy()
    changed[:, decoder.patches.yoke_detector_ids] ^= True
    flipped = decoder.decode_with_gaps_batch(changed)
    for name in ('reference', 'cluster_gap', 'dijkstra_states'):
        np.testing.assert_array_equal(getattr(result, name), getattr(flipped, name))
    np.testing.assert_array_equal(flipped.sigma, ~result.sigma)
    quiet = decoder.decode_with_gaps_batch(np.zeros((1, detectors.shape[1]), dtype=bool))
    assert np.all(quiet.cluster_gap > math.log(100))
    assert all(not value.flags.writeable for value in result.arrays().values())
    empty = decoder.decode_with_gaps_batch(detectors[:0])
    assert empty.prediction.shape == (0, 12)
    with pytest.raises(ValueError, match='binary detectors'):
        decoder.decode_batch(np.full(detectors.shape, 0.5))


def test_batch_partition_and_order_do_not_change_predictions_or_scores(sample):
    decoder = CorrelatedUFHierarchicalDecoder(stim.DetectorErrorModel(sample.dem_text))
    detectors, _ = sample.rows(np.arange(6))
    batch = decoder.decode_with_gaps_batch(detectors)
    reverse = decoder.decode_with_gaps_batch(detectors[::-1])
    for name, value in batch.arrays().items():
        np.testing.assert_array_equal(value, reverse.arrays()[name][::-1])
