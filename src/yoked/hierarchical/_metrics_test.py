"""Tests for the pilot's accuracy metrics and paired bootstrap (_metrics.py).

Checks: a rate reports its denominator and is undefined rather than zero when
nothing is eligible; sector, block, and residual-failure counts on hand-built
bit patterns; misattribution over the exactly-one-failure stratum, pooled
across sectors; every stratum is reported with its denominator, including an
empty one; tie and above-half frequencies; the sinter normalized LER matches a
direct call and is undefined for an undefined rate; the paired bootstrap
reproduces a hand-computed point estimate, is reproducible from its seed,
gives the same answer however the replicates are blocked, keeps the two
sectors of a shot together so cross-sector dependence widens the interval,
counts zero-denominator replicates instead of dropping them silently, and
reports undefined bounds as None; one configuration's rate interval reports a
hand-computed rate, is reproducible from its seed, resamples exactly the shots
the paired bootstrap resamples, and reports an empty input as unavailable; the
paired block failure of two prediction arrays reproduces a hand-computed
difference, is antisymmetric in its arguments, and is the block-failure
comparison ``compare_endpoints`` reports; and the endpoint summary and
comparison report hand-worked counts as JSON-ready mappings.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import sinter

from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._metrics import (
    DEFAULT_REPLICATES, DEFAULT_SEED, OUTER_LOGICAL_VALUES, STRATA, PairedDifference, Rate,
    RateInterval, above_half_frequency, block_failures, bootstrap_rate, compare_endpoints,
    failure_by_stratum, misattribution, normalized_ler, paired_block_failure, paired_bootstrap,
    residual_failure_counts, sector_failures, summarize_result, tie_frequency,
)
from yoked.hierarchical._policies import NoRefinement, RefineAll
from yoked.hierarchical._record import L1Record
from yoked.hierarchical._replay import Estimator, ReplayConfig, replay


# --- rates ---------------------------------------------------------------------


def test_rate_reports_its_denominator():
    rate = Rate(3, 8)
    assert rate.value == pytest.approx(0.375)
    assert rate.to_json() == {'count': 3, 'total': 8, 'value': pytest.approx(0.375)}


def test_an_empty_rate_is_undefined_not_zero():
    rate = Rate(0, 0)
    assert np.isnan(rate.value)
    assert rate.to_json() == {'count': 0, 'total': 0, 'value': None}


@pytest.mark.parametrize('count, total', [(-1, 3), (4, 3), (1, -1), (True, 3), (1.5, 3)])
def test_rate_rejects_impossible_counts(count, total):
    with pytest.raises(ValueError):
        Rate(count, total)


# --- failures over predictions --------------------------------------------------

# Three patches, so a sector holds three observables; column 2i + s is patch i, sector s.
FINAL = np.array([
    [True, False, False, False, False, False],      # patch 0 sector X differs
    [False, False, False, True, False, True],       # patches 1 and 2 differ in sector Z
    [False, False, False, False, False, False],     # matches
])
ACTUAL = np.zeros((3, 6), dtype=bool)


def test_sector_and_block_failures():
    np.testing.assert_array_equal(sector_failures(FINAL, ACTUAL),
                                  [[True, False], [False, True], [False, False]])
    np.testing.assert_array_equal(block_failures(FINAL, ACTUAL), [True, True, False])


def test_residual_failure_counts_count_patches_per_sector():
    np.testing.assert_array_equal(residual_failure_counts(FINAL, ACTUAL), [[1, 0], [0, 2], [0, 0]])


def test_failure_arrays_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        sector_failures(FINAL, ACTUAL[:2])
    with pytest.raises(ValueError):
        block_failures(FINAL[:, :5], ACTUAL[:, :5])


# --- strata and misattribution ---------------------------------------------------

SECTOR_FAILURE = np.array([[True, False], [False, True], [False, False]])
RESIDUAL_COUNTS = np.array([[1, 0], [1, 2], [1, 1]])


def test_misattribution_uses_only_sectors_with_exactly_one_residual_failure():
    rates = misattribution(SECTOR_FAILURE, RESIDUAL_COUNTS)
    assert rates['X'].to_json() == {'count': 1, 'total': 3, 'value': pytest.approx(1 / 3)}
    assert rates['Z'].to_json() == {'count': 0, 'total': 1, 'value': 0.0}
    assert rates['pooled'].to_json() == {'count': 1, 'total': 4, 'value': pytest.approx(0.25)}


def test_misattribution_is_undefined_when_no_sector_is_eligible():
    rates = misattribution(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=int))
    for key in ('X', 'Z', 'pooled'):
        assert rates[key].to_json()['value'] is None
        assert rates[key].to_json()['total'] == 0


def test_every_stratum_is_reported_with_its_denominator():
    strata = failure_by_stratum(SECTOR_FAILURE, RESIDUAL_COUNTS)
    assert list(strata) == list(STRATA)
    # Sector Z holds one zero-failure sector (shot 0), one multiple (shot 1), one single (shot 2).
    assert strata['zero']['Z'].to_json() == {'count': 0, 'total': 1, 'value': 0.0}
    assert strata['one']['X'].to_json() == {'count': 1, 'total': 3, 'value': pytest.approx(1 / 3)}
    assert strata['multiple']['Z'].to_json() == {'count': 1, 'total': 1, 'value': 1.0}
    # Sector X never has zero or multiple residual failures here: both denominators are empty.
    assert strata['zero']['X'].to_json() == {'count': 0, 'total': 0, 'value': None}
    assert strata['multiple']['X'].to_json() == {'count': 0, 'total': 0, 'value': None}
    assert strata['one']['pooled'].to_json() == {'count': 1, 'total': 4, 'value': pytest.approx(0.25)}


def test_strata_denominators_cover_every_sector():
    strata = failure_by_stratum(SECTOR_FAILURE, RESIDUAL_COUNTS)
    assert sum(strata[name]['pooled'].total for name in STRATA) == SECTOR_FAILURE.size


def test_tie_and_above_half_frequencies():
    ties = np.array([[True, False], [False, False], [True, True]])
    assert tie_frequency(ties)['X'].to_json() == {'count': 2, 'total': 3, 'value': pytest.approx(2 / 3)}
    assert tie_frequency(ties)['pooled'].to_json() == {'count': 3, 'total': 6, 'value': 0.5}
    above = np.array([[[True, False, False], [False, False, False]]])
    assert above_half_frequency(above).to_json() == {'count': 1, 'total': 6, 'value': pytest.approx(1 / 6)}


# --- normalized LER ---------------------------------------------------------------


def test_normalized_ler_matches_the_four_decoder_conversion():
    assert OUTER_LOGICAL_VALUES == 8
    expected = sinter.shot_error_rate_to_piece_error_rate(0.25, pieces=216, values=8)
    assert normalized_ler(0.25, pieces=216) == pytest.approx(expected)


def test_normalized_ler_is_undefined_for_an_undefined_rate():
    assert normalized_ler(float('nan'), pieces=216) is None
    assert normalized_ler(Rate(0, 0).value, pieces=72) is None


@pytest.mark.parametrize('pieces', [0, -3])
def test_normalized_ler_requires_positive_pieces(pieces):
    with pytest.raises(ValueError):
        normalized_ler(0.1, pieces=pieces)


# --- paired bootstrap --------------------------------------------------------------


def test_paired_point_estimate_is_the_ratio_of_sums():
    """Hand-computed: a = 3/5 = 0.6, b = 1/5 = 0.2, difference = -0.4."""
    numerator_a = np.array([2.0, 1.0, 0.0])
    numerator_b = np.array([1.0, 0.0, 0.0])
    denominator = np.array([2.0, 2.0, 1.0])
    result = paired_bootstrap(numerator_a, numerator_b, denominator, denominator, replicates=200, seed=DEFAULT_SEED)
    assert result.estimate_a == pytest.approx(0.6)
    assert result.estimate_b == pytest.approx(0.2)
    assert result.difference == pytest.approx(-0.4)
    assert result.low <= result.difference <= result.high
    assert result.low_a <= result.estimate_a <= result.high_a
    assert result.replicates == 200 and result.seed == DEFAULT_SEED
    assert result.zero_denominator_replicates == 0


def test_the_bootstrap_defaults_are_the_four_decoder_convention():
    assert (DEFAULT_REPLICATES, DEFAULT_SEED) == (10000, 43)


def _fixture(seed: int = 1, shots: int = 40):
    rng = np.random.default_rng(seed)
    denominator = rng.integers(1, 3, size=shots).astype(float)
    return (rng.integers(0, 2, size=shots) * denominator, rng.integers(0, 2, size=shots) * denominator,
            denominator, denominator)


def test_the_bootstrap_is_reproducible_from_its_seed():
    arguments = _fixture()
    first = paired_bootstrap(*arguments, replicates=500, seed=DEFAULT_SEED)
    second = paired_bootstrap(*arguments, replicates=500, seed=DEFAULT_SEED)
    assert first == second
    other = paired_bootstrap(*arguments, replicates=500, seed=DEFAULT_SEED + 1)
    assert (other.low, other.high) != (first.low, first.high)


def test_blocking_the_replicates_does_not_change_the_answer(monkeypatch):
    import yoked.hierarchical._metrics as metrics

    arguments = _fixture()
    whole = paired_bootstrap(*arguments, replicates=64, seed=DEFAULT_SEED)
    monkeypatch.setattr(metrics, 'BOOTSTRAP_BLOCK_ENTRIES', 40)      # one replicate per block
    assert paired_bootstrap(*arguments, replicates=64, seed=DEFAULT_SEED) == whole


def test_an_empty_stratum_is_reported_as_unavailable():
    zeros = np.zeros(4)
    result = paired_bootstrap(zeros, zeros, zeros, zeros, replicates=100, seed=DEFAULT_SEED)
    assert np.isnan(result.estimate_a) and np.isnan(result.difference)
    assert result.zero_denominator_replicates == 100
    payload = result.to_json()
    assert payload['estimate_a'] is None and payload['low'] is None and payload['high'] is None
    assert payload['zero_denominator_replicates'] == 100
    json.dumps(payload)


def test_replicates_that_draw_only_empty_shots_are_counted_not_dropped():
    """Two shots, one of them eligible: some replicates resample the empty shot twice."""
    result = paired_bootstrap(np.array([1.0, 0.0]), np.array([0.0, 0.0]),
                              np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                              replicates=400, seed=DEFAULT_SEED)
    assert 0 < result.zero_denominator_replicates < 400
    assert result.estimate_a == pytest.approx(1.0) and result.estimate_b == pytest.approx(0.0)


def test_whole_shots_are_resampled_so_cross_sector_dependence_is_kept():
    """Both fixtures have the same pooled rate; only the dependent one can vary by shot."""
    shots = 20
    denominator = np.full(shots, 2.0)
    dependent = np.where(np.arange(shots) % 2 == 0, 2.0, 0.0)     # both sectors fail, or neither
    independent = np.ones(shots)                                 # exactly one sector fails in every shot
    assert dependent.sum() == independent.sum()
    spread = paired_bootstrap(dependent, dependent, denominator, denominator, replicates=300, seed=DEFAULT_SEED)
    flat = paired_bootstrap(independent, independent, denominator, denominator, replicates=300, seed=DEFAULT_SEED)
    assert spread.estimate_a == flat.estimate_a == pytest.approx(0.5)
    assert spread.high_a - spread.low_a > 0.1
    assert flat.high_a == flat.low_a == pytest.approx(0.5)


@pytest.mark.parametrize('kwargs', [dict(replicates=0), dict(replicates=-5), dict(seed=True)])
def test_paired_bootstrap_validates_its_settings(kwargs):
    arguments = (np.zeros(2), np.zeros(2), np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        paired_bootstrap(*arguments, **{'replicates': 10, 'seed': DEFAULT_SEED, **kwargs})


def test_paired_bootstrap_validates_its_samples():
    with pytest.raises(ValueError, match='length'):
        paired_bootstrap(np.zeros(2), np.zeros(3), np.ones(2), np.ones(3), replicates=10, seed=1)
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([-1.0]), np.zeros(1), np.ones(1), np.ones(1), replicates=10, seed=1)
    with pytest.raises(ValueError, match='numerator'):
        paired_bootstrap(np.array([2.0]), np.zeros(1), np.ones(1), np.ones(1), replicates=10, seed=1)
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([np.nan]), np.zeros(1), np.ones(1), np.ones(1), replicates=10, seed=1)


def test_paired_difference_json_is_serializable():
    payload = paired_bootstrap(*_fixture(), replicates=50, seed=DEFAULT_SEED).to_json()
    assert set(payload) == {field for field in PairedDifference.__dataclass_fields__}
    json.dumps(payload)


# --- one rate's interval -------------------------------------------------------------


def test_bootstrap_rate_reports_a_hand_computed_rate_with_its_interval():
    """Two failures in five shots."""
    result = bootstrap_rate(np.array([1, 0, 0, 1, 0]), replicates=200, seed=DEFAULT_SEED)
    assert (result.count, result.total) == (2, 5)
    assert result.estimate == pytest.approx(0.4)
    assert 0.0 <= result.low <= result.estimate <= result.high <= 1.0
    assert result.replicates == 200 and result.seed == DEFAULT_SEED
    payload = result.to_json()
    assert set(payload) == {field for field in RateInterval.__dataclass_fields__}
    assert (payload['count'], payload['total']) == (2, 5)
    assert payload['estimate'] == pytest.approx(0.4)
    assert (payload['replicates'], payload['seed']) == (200, DEFAULT_SEED)
    json.dumps(payload)


def test_bootstrap_rate_reads_boolean_and_integer_indicators_alike():
    flags = np.array([True, False, True, True])
    assert bootstrap_rate(flags, replicates=50, seed=1) == bootstrap_rate(flags.astype(int), replicates=50, seed=1)
    assert bootstrap_rate(flags, replicates=50, seed=1).count == 3


def test_bootstrap_rate_is_reproducible_from_its_seed():
    indicators = np.random.default_rng(3).integers(0, 2, size=40)
    first = bootstrap_rate(indicators, replicates=500, seed=DEFAULT_SEED)
    assert bootstrap_rate(indicators, replicates=500, seed=DEFAULT_SEED) == first
    other = bootstrap_rate(indicators, replicates=500, seed=DEFAULT_SEED + 1)
    assert other.estimate == first.estimate
    assert (other.low, other.high) != (first.low, first.high)


def test_bootstrap_rate_resamples_the_shots_the_paired_bootstrap_resamples():
    """Under one seed the two draw the same shots, so a rate's interval is exactly the
    interval that side of a paired comparison carries, and the report can quote either."""
    indicators = np.random.default_rng(4).integers(0, 2, size=40).astype(np.float64)
    ones = np.ones(len(indicators))
    rate = bootstrap_rate(indicators, replicates=64, seed=DEFAULT_SEED)
    paired = paired_bootstrap(indicators, ones - indicators, ones, ones, replicates=64, seed=DEFAULT_SEED)
    assert (rate.estimate, rate.low, rate.high) == (paired.estimate_a, paired.low_a, paired.high_a)


def test_blocking_the_rate_replicates_does_not_change_the_answer(monkeypatch):
    import yoked.hierarchical._metrics as metrics

    indicators = np.random.default_rng(4).integers(0, 2, size=40)
    whole = bootstrap_rate(indicators, replicates=64, seed=DEFAULT_SEED)
    monkeypatch.setattr(metrics, 'BOOTSTRAP_BLOCK_ENTRIES', 40)      # one replicate per block
    assert bootstrap_rate(indicators, replicates=64, seed=DEFAULT_SEED) == whole


def test_an_empty_input_is_reported_as_unavailable():
    result = bootstrap_rate(np.zeros(0, dtype=bool), replicates=100, seed=DEFAULT_SEED)
    assert (result.count, result.total) == (0, 0)
    assert np.isnan(result.estimate) and np.isnan(result.low) and np.isnan(result.high)
    assert (result.replicates, result.seed) == (100, DEFAULT_SEED)
    payload = result.to_json()
    assert payload['estimate'] is None and payload['low'] is None and payload['high'] is None
    assert (payload['count'], payload['total']) == (0, 0)
    json.dumps(payload)


@pytest.mark.parametrize('indicators', [np.array([[1, 0]]), np.array([2, 0]), np.array([0.5]),
                                        np.array([np.nan]), np.array([-1]), np.array(['1'])])
def test_bootstrap_rate_rejects_values_that_are_not_indicators(indicators):
    with pytest.raises(ValueError):
        bootstrap_rate(indicators, replicates=10, seed=DEFAULT_SEED)


@pytest.mark.parametrize('kwargs', [dict(replicates=0), dict(replicates=1.5), dict(seed=-1),
                                    dict(seed=True)])
def test_bootstrap_rate_validates_its_settings(kwargs):
    with pytest.raises(ValueError):
        bootstrap_rate(np.array([1, 0]), **{'replicates': 10, 'seed': DEFAULT_SEED, **kwargs})


@pytest.mark.parametrize('fields', [dict(count=3), dict(count=True), dict(replicates=0),
                                    dict(seed=-1), dict(low=None)])
def test_rate_interval_rejects_impossible_fields(fields):
    valid = dict(count=1, total=2, estimate=0.5, low=0.0, high=1.0, replicates=10, seed=0)
    RateInterval(**valid)
    with pytest.raises((TypeError, ValueError)):
        RateInterval(**{**valid, **fields})


# --- block failure of two prediction arrays -------------------------------------------


def test_paired_block_failure_reports_the_hand_computed_difference():
    """Three shots, two patches: a fails shot 0, b fails shots 0 and 2, so b - a = 1/3."""
    record = _record(shots=3, patches=2)
    a = np.zeros((3, 4), dtype=bool)
    a[0, 1] = True
    b = np.zeros((3, 4), dtype=bool)
    b[0, 0] = True
    b[2, 3] = True
    result = paired_block_failure(record, a, b, replicates=200, seed=DEFAULT_SEED)
    assert isinstance(result, PairedDifference)
    assert result.estimate_a == pytest.approx(1 / 3) and result.estimate_b == pytest.approx(2 / 3)
    assert result.difference == pytest.approx(1 / 3)
    assert result.low <= result.difference <= result.high
    assert result.zero_denominator_replicates == 0
    assert (result.replicates, result.seed) == (200, DEFAULT_SEED)
    reverse = paired_block_failure(record, b, a, replicates=200, seed=DEFAULT_SEED)
    assert reverse.difference == pytest.approx(-1 / 3)
    assert (reverse.low, reverse.high) == (pytest.approx(-result.high), pytest.approx(-result.low))


def test_paired_block_failure_scores_against_the_records_actual_flips():
    """A prediction equal to ``actual`` never fails, whatever the reference columns hold."""
    actual = np.array([[True, False, False, True], [False, False, True, False]])
    record = _record(shots=2, patches=2, actual=actual)
    result = paired_block_failure(record, actual, np.zeros((2, 4), dtype=bool), replicates=50, seed=1)
    assert result.estimate_a == 0.0 and result.estimate_b == 1.0 and result.difference == 1.0


def test_paired_block_failure_rejects_the_wrong_record_shapes_and_values():
    record = _record(shots=3, patches=2)
    good = np.zeros((3, 4), dtype=bool)
    with pytest.raises(TypeError):
        paired_block_failure(record.actual, good, good, replicates=10, seed=1)
    with pytest.raises(ValueError):
        paired_block_failure(record, good[:2], good, replicates=10, seed=1)
    with pytest.raises(ValueError):
        paired_block_failure(record, good, np.zeros((3, 6), dtype=bool), replicates=10, seed=1)
    with pytest.raises(ValueError):
        paired_block_failure(record, good, np.full((3, 4), 2), replicates=10, seed=1)
    with pytest.raises(ValueError):
        paired_block_failure(record, good, good, replicates=0, seed=1)


# --- summaries over a replayed record ----------------------------------------------


def _record(*, shots: int, patches: int, **overrides) -> L1Record:
    """A record whose fields are all zero except the ones a test names."""
    columns = 2 * patches
    fields = dict(
        actual=np.zeros((shots, columns), dtype=bool),
        yoke=np.zeros((shots, 2), dtype=bool),
        uf_reference=np.zeros((shots, columns), dtype=bool),
        mwpm_reference=np.zeros((shots, columns), dtype=bool),
        correlated_prediction=np.zeros((shots, columns), dtype=bool),
        joint_mwpm=np.zeros((shots, columns), dtype=bool),
        cluster_gap=np.zeros((shots, columns), dtype=np.float64),
        dijkstra_states=np.zeros((shots, columns), dtype=np.int64),
        forced_plain=np.zeros((shots, patches, 2, 2), dtype=np.float64),
        forced_correlated=np.zeros((shots, patches, 2, 2), dtype=np.float64),
        reweighted_patches=np.zeros((shots, patches), dtype=bool),
        rows=np.arange(shots, dtype=np.int64),
    )
    fields.update(overrides)
    return L1Record(**fields)


def _forced(gap_x: float, gap_z: float) -> np.ndarray:
    """A nonnegative W[c_X, c_Z] with the named signed gaps at the reference bits (0, 0)."""
    base = max(0.0, -gap_x, -gap_z, -(gap_x + gap_z))
    return np.array([[base, base + gap_z], [base + gap_x, base + gap_x + gap_z]])


def _lookup_calibrator(knots) -> IsotonicCalibrator:
    centers = np.array([score for score, _ in knots], dtype=np.float64)
    probabilities = np.array([value for _, value in knots], dtype=np.float64)
    return IsotonicCalibrator('decreasing', centers, probabilities, num_samples=len(knots))


INITIAL = Estimator('uf', 'cluster_gap')
REFINED = Estimator('uf', 'gap_plain')
CALIBRATORS = {
    INITIAL.name: (_lookup_calibrator(((0.0, 0.9), (1.0, 0.7), (2.0, 0.3), (3.0, 0.1))),) * 2,
    REFINED.name: (_lookup_calibrator(((-1.0, 0.95), (0.0, 0.8), (1.0, 0.2), (2.0, 0.05))),) * 2,
    # The control pair, used only to show that two references cannot be compared.
    'mwpm:gap_plain': (_lookup_calibrator(((0.0, 0.4), (1.0, 0.2))),) * 2,
    'mwpm:gap_correlated': (_lookup_calibrator(((0.0, 0.3), (1.0, 0.1))),) * 2,
}
PIECES = 2 * 12
"""Two patches times twelve rounds, the piece count of this two-patch fixture."""


def _endpoint_fixture():
    """Two shots whose hand-worked endpoints disagree, with the actual flips they are scored against.

    Shot 0 has two residual X failures and an unfired yoke; shot 1 has exactly
    one, so it is the only eligible sector of the primary metric.
    """
    record = _record(
        shots=2, patches=2,
        cluster_gap=np.array([[0.0, 3.0, 1.0, 3.0], [2.0, 1.0, 3.0, 0.0]]),
        forced_plain=np.array([[_forced(1.0, -1.0), _forced(2.0, 1.0)],
                               [_forced(0.0, 2.0), _forced(-1.0, 1.0)]]),
        yoke=np.array([[False, False], [True, False]]),
        actual=np.array([[True, False, True, False], [True, False, False, False]]))
    initial = replay(record, CALIBRATORS, ReplayConfig(INITIAL, REFINED, NoRefinement()))
    refined = replay(record, CALIBRATORS, ReplayConfig(INITIAL, REFINED, RefineAll()))
    return record, initial, refined


def test_the_endpoint_fixture_is_the_hand_worked_one():
    record, initial, refined = _endpoint_fixture()
    np.testing.assert_array_equal(initial.final, [[True, False, True, False], [True, True, False, True]])
    np.testing.assert_array_equal(refined.final, [[False, True, False, True], [False, False, True, False]])
    np.testing.assert_array_equal(residual_failure_counts(record.uf_reference, record.actual), [[2, 0], [1, 0]])


def test_summarize_result_reports_hand_worked_counts():
    record, initial, _ = _endpoint_fixture()
    summary = summarize_result(record, initial, pieces=PIECES)
    assert summary['config'] == initial.config and summary['reference'] == 'uf' and summary['shots'] == 2
    # Shot 0 matches; shot 1's Z sector predicts two flips that did not happen.
    assert summary['block_failure'] == {'count': 1, 'total': 2, 'value': 0.5}
    assert summary['sector_failure']['Z'] == {'count': 1, 'total': 2, 'value': 0.5}
    assert summary['sector_failure']['X'] == {'count': 0, 'total': 2, 'value': 0.0}
    assert summary['misattribution']['pooled'] == {'count': 0, 'total': 1, 'value': 0.0}
    # Sector Z has no eligible sector at all: an unavailable rate, not a zero one.
    assert summary['misattribution']['Z']['value'] is None
    assert summary['strata']['zero']['pooled'] == {'count': 1, 'total': 2, 'value': 0.5}
    assert summary['strata']['multiple']['X'] == {'count': 0, 'total': 1, 'value': 0.0}
    assert summary['strata']['one']['Z']['value'] is None
    assert summary['normalized_ler'] == pytest.approx(
        sinter.shot_error_rate_to_piece_error_rate(0.5, pieces=PIECES, values=OUTER_LOGICAL_VALUES))
    assert summary['work']['totals']['requested_patch_sectors'] == 0
    assert summary['ties']['pooled']['count'] == 0
    assert summary['above_half']['total'] == 2 * 2 * 2
    json.dumps(summary)


def test_compare_endpoints_pairs_the_two_configurations():
    record, initial, refined = _endpoint_fixture()
    comparison = compare_endpoints(record, initial, refined, pieces=PIECES, replicates=200, seed=DEFAULT_SEED)
    assert comparison['initial']['config'] == initial.config
    assert comparison['refined']['config'] == refined.config
    # The one eligible sector is right under initial_only and wrong under all_refined.
    primary = comparison['misattribution_pooled']
    assert primary['estimate_a'] == 0.0 and primary['estimate_b'] == 1.0 and primary['difference'] == 1.0
    # Block failure goes from one shot in two to two in two.
    block = comparison['block_failure']
    assert block['estimate_a'] == 0.5 and block['estimate_b'] == 1.0 and block['difference'] == 0.5
    assert block['replicates'] == 200 and block['seed'] == DEFAULT_SEED
    json.dumps(comparison)


def test_compare_endpoints_block_failure_is_the_paired_block_failure():
    record, initial, refined = _endpoint_fixture()
    comparison = compare_endpoints(record, initial, refined, pieces=PIECES, replicates=200, seed=DEFAULT_SEED)
    direct = paired_block_failure(record, initial.final, refined.final, replicates=200, seed=DEFAULT_SEED)
    assert comparison['block_failure'] == direct.to_json()
    # And the rate interval of either side is the one ``bootstrap_rate`` gives that side.
    side = bootstrap_rate(block_failures(refined.final, record.actual), replicates=200, seed=DEFAULT_SEED)
    assert (side.estimate, side.low, side.high) == (direct.estimate_b, direct.low_b, direct.high_b)


def test_compare_endpoints_rejects_results_that_are_not_paired():
    record, initial, refined = _endpoint_fixture()
    other = replay(_record(shots=2, patches=2), CALIBRATORS,
                   ReplayConfig(Estimator('mwpm', 'gap_plain'), Estimator('mwpm', 'gap_correlated'),
                                NoRefinement()))
    with pytest.raises(ValueError, match='reference'):
        compare_endpoints(record, initial, other, pieces=PIECES, replicates=10, seed=DEFAULT_SEED)
    with pytest.raises(ValueError, match='shots'):
        compare_endpoints(_record(shots=3, patches=2), initial, refined,
                          pieces=PIECES, replicates=10, seed=DEFAULT_SEED)
