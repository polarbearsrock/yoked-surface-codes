"""Tests for forced matching weights and signed gaps.

Checks: signed_gaps indexes the forced-weight table by the reference class
(spec test 3); the forced plain optimum agrees with PyMatching's own DEM
decode and its argmin matches the unforced first pass, and sector weights
add exactly (spec test 4); the same additivity and self-consistency hold
for the correlated variant (spec test 5); a hand-built mechanism shows the
correlated reweighting lowering a partner and widening a gap; and the
syndrome-shape guard rejects the wrong number of detector bits.
"""
import math

import numpy as np
import pymatching
import pytest
import stim

from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._fixtures import yoked_fixture
from yoked.hierarchical._matching_gaps import MatchingGaps, signed_gaps
from yoked.hierarchical._patch_graphs import PatchGraphs

ADDITIVITY_TOLERANCE = 1e-9  # spec test 4: sector weights add exactly up to float rounding


def _gaps_for(patch):
    return MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))


def test_signed_gaps_index_the_forced_weights_by_reference_class():
    forced = np.array([[0.0, 1.0], [2.0, 3.5]])  # W[c_X, c_Z]
    np.testing.assert_allclose(signed_gaps(forced, [0, 0]), [2.0, 1.0])
    np.testing.assert_allclose(signed_gaps(forced, [1, 1]), [1.0 - 3.5, 2.0 - 3.5])
    batch = np.stack([forced, forced])
    np.testing.assert_allclose(signed_gaps(batch, [[0, 0], [1, 0]]), [[2.0, 1.0], [-2.0, 1.5]])


def test_forced_weights_agree_with_pymatching_on_the_fixture():
    fx = yoked_fixture(shots=48)
    patch = fx.patches[0]
    gaps = _gaps_for(patch)
    reference = pymatching.Matching.from_detector_error_model(patch.local_dem)
    local = patch.local_syndromes(fx.detectors)
    expected_prediction, expected_weight = reference.decode_batch(local.astype(np.uint8), return_weights=True)
    fired = 0
    for shot in range(len(local)):
        f = gaps.forced_weights(local[shot])
        # The unforced optimum is the cheapest class, and sectors add up (spec test 4).
        assert f.plain.min() == pytest.approx(expected_weight[shot], abs=ADDITIVITY_TOLERANCE)
        assert f.plain[0, 0] + f.plain[1, 1] == pytest.approx(f.plain[0, 1] + f.plain[1, 0], abs=ADDITIVITY_TOLERANCE)
        plain_gap = signed_gaps(f.plain, f.first_pass)
        assert (plain_gap >= -ADDITIVITY_TOLERANCE).all()
        for s in range(2):
            if plain_gap[s] > ADDITIVITY_TOLERANCE:
                assert f.first_pass[s] == bool(expected_prediction[shot, s])
        # Under one reweighted model the same two invariants hold (spec test 5).
        assert f.correlated[0, 0] + f.correlated[1, 1] == pytest.approx(
            f.correlated[0, 1] + f.correlated[1, 0], abs=ADDITIVITY_TOLERANCE)
        assert (signed_gaps(f.correlated, f.correlated_prediction) >= -ADDITIVITY_TOLERANCE).all()
        if not f.rules_fired:
            np.testing.assert_array_equal(f.correlated, f.plain)
            np.testing.assert_array_equal(f.correlated_prediction, f.first_pass)
        fired += f.rules_fired
    assert fired > 0  # SI1000 mechanisms decompose, so some shots must exercise the correlated path


def test_hand_built_correlation_lowers_the_partner_and_widens_the_gap():
    # One patch. X sector: D0 - D1 with a flipping boundary at D0 and a plain boundary at D1.
    # Z sector: D2 - D3 likewise. D4 and D5 are the yokes. Each detector has one boundary edge,
    # as in the real graphs, so the importer merges nothing.
    dem = stim.DetectorErrorModel('''
        error(0.45) D0 D4 L0 ^ D2 D5 L1
        error(0.10) D0 D1
        error(0.30) D1
        error(0.10) D2 D3
        error(0.30) D3
    ''')
    patch = PatchGraphs.from_yoked_dem(dem, num_patches=1)[0]
    gaps = _gaps_for(patch)
    flip = math.log(0.55 / 0.45)                          # the observable-flipping boundary edge
    path = math.log(0.90 / 0.10) + math.log(0.70 / 0.30)  # the detour through the plain boundary
    f = gaps.forced_weights(np.array([1, 0, 1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(f.first_pass, [True, True])
    np.testing.assert_allclose(f.plain, [[2 * path, path + flip], [path + flip, 2 * flip]])
    # Selecting D0's flipping edge implies D2's with probability min(1/2, .45/.45) = 1/2, weight 0,
    # and symmetrically, so under the reweighted model both flipping edges cost nothing.
    assert f.rules_fired
    np.testing.assert_allclose(f.correlated, [[2 * path, path], [path, 0.0]], atol=1e-12)
    np.testing.assert_array_equal(f.correlated_prediction, [True, True])
    np.testing.assert_allclose(signed_gaps(f.plain, f.first_pass), [path - flip, path - flip])
    np.testing.assert_allclose(signed_gaps(f.correlated, f.first_pass), [path, path])


def test_syndrome_shape_is_validated():
    fx = yoked_fixture(shots=1)
    with pytest.raises(ValueError, match='detector bits'):
        _gaps_for(fx.patches[0]).forced_weights(np.zeros(3, dtype=np.uint8))
