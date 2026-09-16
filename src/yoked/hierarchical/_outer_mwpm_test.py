"""Tests for the subdivided-star MWPM outer decoder."""
import numpy as np
import pytest

from yoked.hierarchical._outer_decoder import exact_outer_map
from yoked.hierarchical._outer_mwpm import (
    _matching_for_costs,
    mwpm_outer_map,
    mwpm_outer_map_batch,
)


@pytest.mark.parametrize('seed', range(10))
def test_matches_independent_brute_force_oracle(seed):
    rng = np.random.default_rng(seed)
    grid = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    for n in (1, 2, 6, 8):
        for _ in range(20):
            q = rng.choice(grid, n)
            parity = int(rng.integers(2))
            candidates = rng.random(n) < 0.65
            if parity and not candidates.any():
                with pytest.raises(ValueError, match='no candidate'):
                    mwpm_outer_map(q, parity, candidates)
                continue
            expected = exact_outer_map(q, parity, candidates)
            actual = mwpm_outer_map(q, parity, candidates)
            np.testing.assert_array_equal(actual.pattern, expected.pattern)
            assert actual.tied == expected.tied


def test_above_half_preflips_masks_and_multiple_tiny_toggles():
    q = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1])
    assert mwpm_outer_map(q, 0).pattern.tolist() == [1, 1, 0, 0, 0, 0]
    assert mwpm_outer_map(q, 1).pattern.tolist() == [1, 0, 0, 0, 0, 0]
    assert mwpm_outer_map(q, 1, [0, 1, 1, 1, 1, 1]).pattern.tolist() == [0, 1, 0, 0, 0, 0]
    tiny = np.full(4, 1 / (1 + np.exp(-2e-10)))
    result = mwpm_outer_map(tiny, 0)
    assert result.pattern.tolist() == [0, 0, 0, 0]
    assert result.tied


def test_half_probability_and_no_candidate_even_case():
    for parity in (0, 1):
        result = mwpm_outer_map([0.5], parity)
        assert result.pattern.tolist() == [bool(parity)]
        assert not result.tied
    result = mwpm_outer_map([0.9, 0.8], 0, [0, 0])
    assert result.pattern.tolist() == [0, 0] and not result.tied


def test_more_than_sixteen_patches_and_batch_read_only_repeatability():
    q = np.linspace(0.05, 0.95, 25)
    first = mwpm_outer_map(q, 1)
    second = mwpm_outer_map(q, 1)
    np.testing.assert_array_equal(first.pattern, second.pattern)
    batch = mwpm_outer_map_batch(np.stack([q, q]), np.array([0, 1]))
    assert batch.patterns.shape == (2, 25)
    with pytest.raises(ValueError):
        batch.patterns[0, 0] = True
    with pytest.raises(ValueError):
        batch.tied[0] = True


def test_validation():
    with pytest.raises(ValueError, match='inside'):
        mwpm_outer_map([0, 0.5], 0)
    with pytest.raises(ValueError, match='Parity'):
        mwpm_outer_map_batch(np.ones((1, 2)) * 0.5, np.array([2]))
    with pytest.raises(ValueError, match='same shape'):
        mwpm_outer_map([0.2, 0.3], 0, [1])
    with pytest.raises(ValueError, match='binary'):
        mwpm_outer_map([0.2, 0.3], 0, [1, 2])


def test_subdivision_preserves_distinct_patch_fault_ids():
    matching = _matching_for_costs(np.array([4.0, 9.0, 2.0]), np.array([0, 2]))
    graph = matching.to_networkx()
    weighted = [data for _, _, data in graph.edges(data=True) if data['weight']]
    assert len(weighted) == 2
    assert {frozenset(data['fault_ids']) for data in weighted} == {frozenset({0}), frozenset({2})}
    assert matching.decode(np.array([1, 0, 0], dtype=np.uint8)).tolist() == [0, 0, 1]


def test_precision_repair_when_matching_returns_wrong_unique_fault(monkeypatch):
    # Simulate a finite-precision reversal while retaining a unique float optimum.
    import pymatching
    original = pymatching.Matching.decode

    def wrong(self, syndrome, *args, **kwargs):
        result = np.asarray(original(self, syndrome, *args, **kwargs))
        if syndrome[0] and len(result) >= 2:
            result[:] = 0
            result[1] = 1
        return result

    monkeypatch.setattr(pymatching.Matching, 'decode', wrong)
    result = mwpm_outer_map([0.49, 0.1], 1)
    assert result.pattern.tolist() == [1, 0]
    assert not result.tied


def test_close_costs_remain_float_optimal_beside_a_large_cost():
    # PyMatching 2.4 rounds the first two arms to equal integer weights here,
    # although their original costs differ by much more than the tie tolerance.
    costs = np.array([1.0000001, 1.0, 100.0])
    q = 1 / (1 + np.exp(costs))
    result = mwpm_outer_map(q, 1)
    assert result.pattern.tolist() == [0, 1, 0]
    assert not result.tied


@pytest.mark.parametrize('restricted', [False, True])
def test_replay_calls_mwpm_in_both_sectors_without_enumeration(monkeypatch, restricted):
    import importlib
    implementation = importlib.import_module('yoked.hierarchical._replay')
    oracle = importlib.import_module('yoked.hierarchical._outer_decoder')
    calls = []

    def track(q, parity, candidates):
        calls.append(None if candidates is None else candidates.copy())
        return mwpm_outer_map_batch(q, parity, candidates)

    def refuse(*args, **kwargs):
        raise AssertionError('Production replay must not enumerate residual patterns')

    monkeypatch.setattr(implementation, 'mwpm_outer_map_batch', track)
    monkeypatch.setattr(oracle, 'exact_outer_map_batch', refuse)
    q = np.full((2, 2, 3), 0.1)
    sigma = np.array([[1, 0], [0, 1]], dtype=bool)
    requested = np.ones(q.shape, dtype=bool)
    patterns, _ = implementation._outer_patterns(q, sigma, requested, restricted)
    assert len(calls) == 2
    assert all((call is not None) == restricted for call in calls)
    np.testing.assert_array_equal(patterns.sum(axis=2) % 2, sigma)
