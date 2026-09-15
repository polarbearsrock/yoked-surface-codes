"""Tests for isotonic calibration of soft-output scores (_calibration.py).

Checks: the fit recovers a monotone step function against noisy synthetic
data in both directions; the exact PAV knot convention on small hand-picked
examples (block centres are count-weighted means, equal-valued adjacent
blocks stay separate knots, a strict violation pools into one block);
interpolation between knots and constant extrapolation beyond them; clipped
outputs keep the log-odds finite; invalid input to `fit`/`probability` (bad
direction, non-finite scores, outcomes outside {0, 1}, mismatched or empty
samples) is rejected; every knot-validity branch in the dataclass
constructor itself (`__post_init__`) is separately exercised by constructing
`IsotonicCalibrator` directly, since `fit` only ever builds valid knots; and
the calibrator round-trips through JSON.
"""
import json

import numpy as np
import pytest

from yoked.hierarchical._calibration import CLIP, IsotonicCalibrator


def test_recovers_a_decreasing_step_function():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 10, 20000)
    truth = np.where(scores < 5, 0.4, 0.05)
    outcomes = rng.random(20000) < truth
    calibrator = IsotonicCalibrator.fit(scores, outcomes, direction='decreasing')
    assert calibrator.num_samples == 20000
    assert calibrator.probability([1.0]) == pytest.approx(0.4, abs=0.03)
    assert calibrator.probability([9.0]) == pytest.approx(0.05, abs=0.02)
    probabilities = calibrator.probability(np.linspace(-1, 11, 500))
    assert (np.diff(probabilities) <= 1e-12).all()   # monotone, including constant extrapolation


def test_decreasing_fit_interpolates_between_block_centres():
    calibrator = IsotonicCalibrator.fit([1, 2, 3, 4], [1, 1, 0, 0], direction='decreasing')
    np.testing.assert_allclose(calibrator.centers, [1, 2, 3, 4])
    np.testing.assert_allclose(calibrator.probabilities, [1 - CLIP, 1 - CLIP, CLIP, CLIP])
    np.testing.assert_allclose(calibrator.probability([1.5, 2, 3, 3.5]), [1 - CLIP, 1 - CLIP, CLIP, CLIP])
    assert calibrator.probability([2.5]) == pytest.approx(0.5, abs=1e-6)
    assert calibrator.probability([0.0]) == pytest.approx(1 - CLIP)
    assert calibrator.probability([9.0]) == pytest.approx(CLIP)


def test_increasing_direction_pools_equal_scores():
    calibrator = IsotonicCalibrator.fit([1, 1, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 0, 1, 1, 1], direction='increasing')
    np.testing.assert_allclose(calibrator.probability([1, 2, 3]), [0.5, 0.5, 1 - CLIP])
    assert calibrator.probability([0]) == pytest.approx(0.5)


def test_violators_pool_into_one_block():
    calibrator = IsotonicCalibrator.fit([1, 2, 3, 4], [1, 1, 0, 0], direction='increasing')
    assert len(calibrator.centers) == 1
    np.testing.assert_allclose(calibrator.probability([0, 1, 2.5, 4, 9]), 0.5)


def test_clipping_keeps_log_odds_finite():
    calibrator = IsotonicCalibrator.fit([0.0, 1.0, 2.0], [0, 0, 0], direction='decreasing')
    p = calibrator.probability([1.0])
    assert p == pytest.approx(CLIP) and np.isfinite(np.log(p / (1 - p)))


def test_rejects_bad_input():
    with pytest.raises(ValueError, match='direction'):
        IsotonicCalibrator.fit([1.0], [0], direction='sideways')
    with pytest.raises(ValueError, match='finite'):
        IsotonicCalibrator.fit([np.inf, 1.0], [0, 1], direction='decreasing')
    with pytest.raises(ValueError, match='0 or 1'):
        IsotonicCalibrator.fit([1.0, 2.0], [0, 2], direction='decreasing')
    with pytest.raises(ValueError, match='length'):
        IsotonicCalibrator.fit([1.0, 2.0], [0], direction='decreasing')
    with pytest.raises(ValueError, match='empty'):
        IsotonicCalibrator.fit([], [], direction='decreasing')
    calibrator = IsotonicCalibrator.fit([1.0, 2.0], [0, 1], direction='increasing')
    with pytest.raises(ValueError, match='finite'):
        calibrator.probability([np.nan])


@pytest.mark.parametrize('direction, centers, probabilities, num_samples, message', [
    ('sideways', [1.0, 2.0], [0.4, 0.5], 10, 'direction'),               # not one of DIRECTIONS
    ('increasing', [], [], 10, 'nonempty'),                              # no knots at all
    ('increasing', [1.0, 2.0], [0.4], 10, 'equally sized'),              # centers/probabilities length mismatch
    ('increasing', [1.0, np.inf], [0.4, 0.5], 10, 'finite'),             # a non-finite center
    ('increasing', [2.0, 1.0], [0.4, 0.5], 10, 'strictly increasing'),   # centers out of order
    ('increasing', [1.0, 2.0], [0.0, 0.5], 10, 'clipping range'),        # probability below CLIP
    ('increasing', [1.0, 2.0], [0.5, 1.0], 10, 'clipping range'),        # probability above 1 - CLIP
    ('increasing', [1.0, 2.0], [0.9, 0.1], 10, 'monotone'),              # falling probabilities, increasing direction
    ('decreasing', [1.0, 2.0], [0.1, 0.9], 10, 'monotone'),              # rising probabilities, decreasing direction
    ('increasing', [1.0, 2.0], [0.4, 0.5], 0, 'positive integer'),       # num_samples zero
    ('increasing', [1.0, 2.0], [0.4, 0.5], -1, 'positive integer'),      # num_samples negative
    ('increasing', [1.0, 2.0], [0.4, 0.5], True, 'positive integer'),    # num_samples a bool, not a count
])
def test_constructor_rejects_every_invalid_knot_case(direction, centers, probabilities, num_samples, message):
    # fit() only ever hands the constructor knots that are already valid, so these branches of
    # __post_init__ are otherwise untested; construct directly to hit each one.
    with pytest.raises(ValueError, match=message):
        IsotonicCalibrator(direction, centers, probabilities, num_samples)


def test_constructor_accepts_valid_knots_and_interpolates():
    # A direct construction with hand-picked knots, exercising the happy path through the same
    # validation that the rejection cases above each trip on one branch of.
    calibrator = IsotonicCalibrator('increasing', [1.0, 3.0], [0.2, 0.8], 5)
    assert calibrator.num_samples == 5
    assert calibrator.probability([2.0]) == pytest.approx(0.5)   # linear interpolation, midpoint of the two knots
    assert calibrator.probability([0.0]) == pytest.approx(0.2)   # constant extrapolation below the first knot
    assert calibrator.probability([5.0]) == pytest.approx(0.8)   # constant extrapolation above the last knot


def test_json_round_trip():
    calibrator = IsotonicCalibrator.fit(np.arange(50) % 7, (np.arange(50) % 3) == 0, direction='decreasing')
    restored = IsotonicCalibrator.from_json(json.loads(json.dumps(calibrator.to_json())))
    grid = np.linspace(-2, 9, 200)
    np.testing.assert_array_equal(restored.probability(grid), calibrator.probability(grid))
    assert restored.direction == 'decreasing' and restored.num_samples == 50
