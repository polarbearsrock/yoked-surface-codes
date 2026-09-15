"""Tests for the exact L2 outer decoder (_outer_decoder.py).

Checks: brute-force enumeration agrees with an independent analytic
threshold-and-parity rule across random probabilities, parities, and
candidate masks, including the infeasible odd-parity-with-no-candidate
case; a single q = 0.5 bit does not spuriously tie across parities; the
tie tolerance is absolute in log weight and allows several small toggles
at once; the two-error rescue example from the spec runs with an unfired
yoke; a strictly increasing (rank-preserving) transform of q can change
the argmax pattern; ties break to the lowest binary value and are
reported, including an equal-cost single-clear-vs-single-set case;
candidate restriction fixes non-candidates to zero and can force a
candidate to flip under odd parity; the batched decoder agrees with the
single-shot decoder shot by shot and validates probabilities strictly
inside (0, 1); and frame_adjusted_syndrome computes the residual parity
against a fixture with known flips per sector.
"""
import numpy as np
import pytest

from yoked.hierarchical._outer_decoder import (
    TIE_TOLERANCE, exact_outer_map, exact_outer_map_batch, frame_adjusted_syndrome,
)


def _analytic_rule(q, parity, candidates=None):
    """Threshold/parity minima, with conditional minima resolving absolute-tolerance ties.

    No patterns are enumerated. Fixing some bits leaves the same analytic
    problem: take the preferred free bits and, if needed, the cheapest toggle.
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    candidates = np.ones(n, dtype=bool) if candidates is None else np.asarray(candidates, dtype=bool)
    b = (q > 0.5) & candidates
    cost = np.abs(np.log1p(-q) - np.log(q))

    def minimum_with(fixed):
        if np.any((fixed == 1) & ~candidates):
            return np.inf
        assigned = fixed >= 0
        preferred = b.copy()
        preferred[assigned] = fixed[assigned].astype(bool)
        result = cost[assigned & (preferred != b)].sum()
        free = ~assigned & candidates
        if preferred.sum() % 2 != parity:
            result += cost[free].min(initial=np.inf)
        return result

    fixed = np.full(n, -1, dtype=np.int8)
    best = minimum_with(fixed)
    if not np.isfinite(best):
        raise ValueError('no candidate')
    limit = best + TIE_TOLERANCE
    # Highest bit first: prefer zero whenever an admissible completion exists.
    for bit in reversed(range(n)):
        fixed[bit] = 0
        if minimum_with(fixed) > limit:
            fixed[bit] = 1
    pattern = fixed.astype(bool)
    # Another admissible pattern must differ at at least one candidate bit.
    tied = False
    for bit in np.flatnonzero(candidates):
        alternative = np.full(n, -1, dtype=np.int8)
        alternative[bit] = 1 - fixed[bit]
        tied |= minimum_with(alternative) <= limit
    return pattern, bool(tied)


GRID = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98])


@pytest.mark.parametrize('seed', range(20))
def test_enumeration_matches_the_analytic_rule(seed):
    rng = np.random.default_rng(seed)
    for n in (1, 2, 6, 8):
        for _ in range(25):
            q = rng.choice(GRID, size=n)
            parity = int(rng.integers(2))
            candidates = None if rng.random() < 0.5 else rng.random(n) < 0.6
            if candidates is not None and parity == 1 and not candidates.any():
                with pytest.raises(ValueError, match='no candidate'):
                    exact_outer_map(q, parity, candidates)
                continue
            expected, expected_tie = _analytic_rule(q, parity, candidates)
            decision = exact_outer_map(q, parity, candidates)
            np.testing.assert_array_equal(decision.pattern, expected)
            assert decision.tied == expected_tie


def test_one_half_bit_does_not_create_a_parity_preserving_tie():
    for parity in (0, 1):
        result = exact_outer_map([0.5], parity)
        assert result.pattern.tolist() == [bool(parity)] and not result.tied
        expected, tied = _analytic_rule([0.5], parity)
        np.testing.assert_array_equal(result.pattern, expected)
        assert result.tied == tied


def test_tie_tolerance_is_absolute_and_allows_multiple_small_toggles():
    q = 1 / (1 + np.exp(np.array([1.0, 1.0 + 5e-7])))
    result = exact_outer_map(q, 1)
    assert result.pattern.tolist() == [True, False] and not result.tied
    expected, tied = _analytic_rule(q, 1)
    np.testing.assert_array_equal(result.pattern, expected)
    assert result.tied == tied
    q = np.full(3, 1 / (1 + np.exp(-0.2 * TIE_TOLERANCE)))
    result = exact_outer_map(q, 0)
    assert result.pattern.tolist() == [False, False, False] and result.tied
    expected, tied = _analytic_rule(q, 0)
    np.testing.assert_array_equal(result.pattern, expected)
    assert result.tied == tied


def test_two_error_rescue_with_unfired_yoke():
    q = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    assert exact_outer_map(q, 0).pattern.tolist() == [1, 1, 0, 0, 0, 0]
    assert exact_outer_map(q, 1).pattern.tolist() == [1, 0, 0, 0, 0, 0]


def test_rank_preserving_transformation_can_change_the_map_pattern():
    q = np.array([0.45, 0.4, 0.1, 0.1, 0.1, 0.1])
    assert not exact_outer_map(q, 0).pattern.any()
    stretched = np.sqrt(q)   # strictly increasing, so rankings are unchanged
    assert exact_outer_map(stretched, 0).pattern.tolist() == [1, 1, 0, 0, 0, 0]


def test_ties_break_to_the_lowest_binary_value_and_are_reported():
    decision = exact_outer_map([0.5, 0.5, 0.1], 0)
    assert decision.pattern.tolist() == [0, 0, 0] and decision.tied
    decision = exact_outer_map([0.1, 0.1, 0.1], 1)
    assert decision.pattern.tolist() == [1, 0, 0] and decision.tied
    decision = exact_outer_map([0.7, 0.3, 0.1], 0)   # clearing bit 0 costs the same as setting bit 1
    assert decision.pattern.tolist() == [0, 0, 0] and decision.tied


def test_candidate_restriction_fixes_non_candidates_to_zero():
    q = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    decision = exact_outer_map(q, 1, candidates=[False, True, True, True, True, True])
    assert decision.pattern.tolist() == [0, 1, 0, 0, 0, 0] and decision.tied
    decision = exact_outer_map(q, 0, candidates=[False, True, True, True, True, True])
    assert not decision.pattern.any() and not decision.tied
    with pytest.raises(ValueError, match='no candidate'):
        exact_outer_map(q, 1, candidates=[False] * 6)


def test_batch_and_single_agree_and_validate_probabilities():
    rng = np.random.default_rng(3)
    q = rng.choice(GRID, size=(40, 6))
    parity = rng.integers(2, size=40)
    candidates = rng.random((40, 6)) < 0.7
    candidates[parity == 1, 0] = True
    batch = exact_outer_map_batch(q, parity, candidates)
    patterns, tied = batch.patterns, batch.tied
    for shot in range(40):
        single = exact_outer_map(q[shot], parity[shot], candidates[shot])
        np.testing.assert_array_equal(patterns[shot], single.pattern)
        assert tied[shot] == single.tied
    with pytest.raises(ValueError, match='inside'):
        exact_outer_map([0.0, 0.5], 0)
    with pytest.raises(ValueError, match='inside'):
        exact_outer_map([1.0, 0.5], 0)


def test_frame_adjusted_syndrome_is_the_residual_parity():
    yoke = np.array([[1, 0], [0, 1]], dtype=bool)
    reference = np.zeros((2, 12), dtype=bool)
    reference[0, [0, 2]] = True      # two X-sector reference flips: parity 0 -> sigma_X = 1
    reference[1, [1]] = True         # one Z-sector reference flip: parity 1 -> sigma_Z = 0
    np.testing.assert_array_equal(frame_adjusted_syndrome(yoke, reference), [[1, 0], [0, 0]])
