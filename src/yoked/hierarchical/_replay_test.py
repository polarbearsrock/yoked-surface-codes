"""Tests for endpoint replay and its work accounting (_replay.py).

Checks: an estimator validates its reference/score pair and round-trips
through ``parse``; signed scores index the forced weights against the chosen
reference's bits, hand-computed; calibrators pool the patches of a sector, so
``num_samples`` is shots times patches, and the fitted mapping is immutable;
only the three specified estimator pairs make a replay configuration; the
endpoint final bits match hand-worked L2 answers, including a reversal that
flips patches under an unfired yoke and a deterministic tie; mixed and
candidate-restricted L2 differ where designed and the restricted rule raises
when odd parity has no requested candidate; replay leaves the record and the
calibrators unchanged, is deterministic, ignores ``q1`` where ``M`` is false,
and rejects a final whose sector parity is not the yoke bit; each estimator
pair's work counts match the specified table at no and at full refinement,
requests on both sectors of one patch cost one distinct patch while requests
on two patches cost two; and replay work is a separate object from the
``CollectionWork`` a real distance-3 collection reports.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

import numpy as np
import pytest

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._l1 import WORK_FIELDS, L1Context, collect_rows
from yoked.hierarchical._fixtures import NUM_PATCHES, yoked_fixture
from yoked.hierarchical._policies import NoRefinement, RefineAll
from yoked.hierarchical._record import L1Record, by_sector
from yoked.hierarchical._replay import (
    REFINEMENT_PAIRS, REPLAY_BATCH_SHOTS, Estimator, ReplayConfig, ReplayResult, WorkCounts,
    calibrated_probabilities, estimator_scores, fit_calibrators, residual_errors, replay,
)

# --- small hand-built records ------------------------------------------------


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


def _forced(gap_x: float, gap_z: float, reference=(0, 0)) -> np.ndarray:
    """A nonnegative W[c_X, c_Z] whose signed gaps at ``reference`` are the ones asked for."""
    r_x, r_z = reference
    base = max(0.0, -gap_x, -gap_z, -(gap_x + gap_z))
    weights = np.empty((2, 2), dtype=np.float64)
    weights[r_x, r_z] = base
    weights[1 - r_x, r_z] = base + gap_x
    weights[r_x, 1 - r_z] = base + gap_z
    weights[1 - r_x, 1 - r_z] = base + gap_x + gap_z
    return weights


def _lookup_calibrator(knots) -> IsotonicCalibrator:
    """A decreasing calibrator that maps each listed score exactly to its probability."""
    centers = np.array([score for score, _ in knots], dtype=np.float64)
    probabilities = np.array([value for _, value in knots], dtype=np.float64)
    return IsotonicCalibrator('decreasing', centers, probabilities, num_samples=len(knots))


# Scores are chosen so that every probability below is read off a knot exactly.
INITIAL_KNOTS = ((0.0, 0.9), (1.0, 0.7), (2.0, 0.3), (3.0, 0.1))
REFINED_KNOTS = ((-1.0, 0.95), (0.0, 0.8), (1.0, 0.2), (2.0, 0.05))
TIED_KNOTS = ((0.0, 0.5), (1.0, 0.5))

INITIAL = Estimator('uf', 'cluster_gap')
REFINED = Estimator('uf', 'gap_plain')


def _calibrators(initial=INITIAL_KNOTS, refined=REFINED_KNOTS,
                 initial_estimator=INITIAL, refined_estimator=REFINED) -> dict:
    """One (X, Z) calibrator pair per estimator; the two sectors share a map here."""
    return {
        initial_estimator.name: (_lookup_calibrator(initial), _lookup_calibrator(initial)),
        refined_estimator.name: (_lookup_calibrator(refined), _lookup_calibrator(refined)),
    }


def _columns(values) -> np.ndarray:
    """(shots, 2P) from a nested [shot][patch][sector] listing."""
    return np.array([[value for patch in shot for value in patch] for shot in values])


@dataclass(frozen=True)
class _FixedRequest:
    """A test-only policy that always requests the listed patch-sectors.

    Production policies are only the two endpoints, so a partial request mask
    has to come from somewhere; this stands in for the selective policies of a
    later milestone without pretending to be one.
    """

    mask: tuple
    name: str = 'fixed_request'

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        wanted = np.asarray(self.mask, dtype=bool)
        return readonly_array(np.broadcast_to(wanted, q0.shape), dtype=bool)


# --- estimators ---------------------------------------------------------------


def test_estimator_name_and_direction():
    assert Estimator('uf', 'cluster_gap').name == 'uf:cluster_gap'
    assert Estimator('mwpm', 'gap_correlated').name == 'mwpm:gap_correlated'
    # Every score decreases in the error probability: a larger gap is more confident.
    assert Estimator('mwpm', 'gap_plain').direction == 'decreasing'


@pytest.mark.parametrize('reference, score', [
    ('joint', 'gap_plain'),        # not a stored reference decoder
    ('uf', 'gap'),                 # not a stored score
    ('mwpm', 'cluster_gap'),       # the cluster gap exists only for the UF reference
])
def test_estimator_rejects_impossible_pairs(reference, score):
    with pytest.raises(ValueError):
        Estimator(reference, score)


@pytest.mark.parametrize('text', ['uf:cluster_gap', 'mwpm:gap_correlated'])
def test_estimator_parse_round_trips(text):
    assert Estimator.parse(text).name == text


@pytest.mark.parametrize('text', ['uf', 'uf:gap_plain:extra', '', 'mwpm:cluster_gap'])
def test_estimator_parse_names_the_text_it_rejects(text):
    with pytest.raises(ValueError, match=re.escape(repr(text))):
        Estimator.parse(text)


# --- scores and outcomes ------------------------------------------------------


def test_cluster_gap_scores_are_the_stored_cluster_gaps():
    gaps = np.array([[0.5, 1.5, 2.5, 3.5]])
    record = _record(shots=1, patches=2, cluster_gap=gaps)
    scores = estimator_scores(record, Estimator('uf', 'cluster_gap'))
    np.testing.assert_array_equal(scores, gaps)
    assert not scores.flags.writeable


def test_signed_scores_index_the_forced_weights_against_the_reference_bits():
    """Hand-computed: with r = (1, 0), delta_X = W[0, 0] - W[1, 0] and delta_Z = W[1, 1] - W[1, 0]."""
    weights = np.array([[[1.0, 5.0], [2.0, 8.0]]])          # (patches=1, 2, 2) indexed [c_X, c_Z]
    record = _record(shots=1, patches=1,
                     uf_reference=np.array([[True, False]]),
                     mwpm_reference=np.array([[False, False]]),
                     forced_plain=weights[None])
    uf = estimator_scores(record, Estimator('uf', 'gap_plain'))
    np.testing.assert_allclose(uf, [[1.0 - 2.0, 8.0 - 2.0]])
    # The same weights read against the other reference's bits r = (0, 0).
    mwpm = estimator_scores(record, Estimator('mwpm', 'gap_plain'))
    np.testing.assert_allclose(mwpm, [[2.0 - 1.0, 5.0 - 1.0]])


def test_correlated_scores_read_the_correlated_weights():
    plain = np.array([[[0.0, 3.0], [1.0, 4.0]]])
    correlated = np.array([[[0.0, 7.0], [2.0, 9.0]]])
    record = _record(shots=1, patches=1, forced_plain=plain[None], forced_correlated=correlated[None])
    np.testing.assert_allclose(estimator_scores(record, Estimator('uf', 'gap_plain')), [[1.0, 3.0]])
    np.testing.assert_allclose(estimator_scores(record, Estimator('uf', 'gap_correlated')), [[2.0, 7.0]])


def test_residual_errors_are_actual_xor_reference():
    record = _record(shots=2, patches=1,
                     actual=np.array([[True, False], [True, True]]),
                     uf_reference=np.array([[True, True], [False, False]]),
                     mwpm_reference=np.array([[False, False], [False, True]]))
    np.testing.assert_array_equal(residual_errors(record, 'uf'), [[False, True], [True, True]])
    np.testing.assert_array_equal(residual_errors(record, 'mwpm'), [[True, False], [True, False]])
    assert not residual_errors(record, 'uf').flags.writeable


# --- calibration over a record ------------------------------------------------


def _pooling_record(shots: int = 8, patches: int = 3) -> L1Record:
    """A record whose cluster gaps and residual errors differ patch by patch."""
    rng = np.random.default_rng(0)
    columns = 2 * patches
    return _record(shots=shots, patches=patches,
                   cluster_gap=rng.integers(0, 4, size=(shots, columns)).astype(np.float64),
                   actual=rng.integers(0, 2, size=(shots, columns)).astype(bool))


def test_fit_calibrators_pools_the_patches_of_each_sector():
    record = _pooling_record()
    calibrators = fit_calibrators(record, [Estimator('uf', 'cluster_gap')])
    assert set(calibrators) == {'uf:cluster_gap'}
    sector_x, sector_z = calibrators['uf:cluster_gap']
    for calibrator in (sector_x, sector_z):
        assert calibrator.direction == 'decreasing'
        assert calibrator.num_samples == record.shots * record.num_patches
    # The two sectors see different samples, so they are separately fitted objects.
    assert sector_x is not sector_z


def test_fitted_calibrators_are_immutable():
    calibrators = fit_calibrators(_pooling_record(), [Estimator('uf', 'cluster_gap')])
    with pytest.raises(TypeError):
        calibrators['uf:gap_plain'] = None


def test_fit_calibrators_rejects_an_empty_or_repeated_estimator_list():
    record = _pooling_record()
    with pytest.raises(ValueError):
        fit_calibrators(record, [])
    with pytest.raises(ValueError, match='uf:cluster_gap'):
        fit_calibrators(record, [Estimator('uf', 'cluster_gap'), Estimator('uf', 'cluster_gap')])


def test_calibrated_probabilities_apply_each_sector_calibrator():
    record = _record(shots=1, patches=2, cluster_gap=np.array([[0.0, 1.0, 2.0, 3.0]]))
    calibrators = {'uf:cluster_gap': (_lookup_calibrator(INITIAL_KNOTS), _lookup_calibrator(TIED_KNOTS))}
    q = calibrated_probabilities(record, INITIAL, calibrators)
    assert q.shape == (1, 2, 2)
    np.testing.assert_allclose(q[0, 0], [0.9, 0.3])     # sector X: patches 0 and 1, scores 0 and 2
    np.testing.assert_allclose(q[0, 1], [0.5, 0.5])     # sector Z under the flat calibrator
    assert not q.flags.writeable


def test_calibrated_probabilities_name_a_missing_estimator():
    record = _record(shots=1, patches=1)
    with pytest.raises(ValueError, match='uf:cluster_gap'):
        calibrated_probabilities(record, INITIAL, {})


# --- configurations -----------------------------------------------------------


def test_replay_config_name():
    config = ReplayConfig(INITIAL, REFINED, RefineAll())
    assert config.name == 'uf:cluster_gap->gap_plain:all_refined:mixed'
    assert config.reference == 'uf'


def test_replay_config_requires_one_reference():
    with pytest.raises(ValueError, match='reference'):
        ReplayConfig(INITIAL, Estimator('mwpm', 'gap_plain'), RefineAll())


@pytest.mark.parametrize('initial, refined', [
    (Estimator('uf', 'gap_plain'), Estimator('uf', 'gap_correlated')),
    (Estimator('uf', 'cluster_gap'), Estimator('uf', 'cluster_gap')),
    (Estimator('mwpm', 'gap_correlated'), Estimator('mwpm', 'gap_plain')),
])
def test_replay_config_rejects_pairs_outside_the_specified_table(initial, refined):
    with pytest.raises(ValueError):
        ReplayConfig(initial, refined, RefineAll())


def test_replay_config_rejects_an_unknown_outer_rule_and_a_non_policy():
    with pytest.raises(ValueError, match='outer'):
        ReplayConfig(INITIAL, REFINED, RefineAll(), outer='candidates')
    with pytest.raises(TypeError):
        ReplayConfig(INITIAL, REFINED, policy='all_refined')


def test_the_three_specified_pairs_are_the_whole_table():
    assert set(REFINEMENT_PAIRS) == {
        ('uf', 'cluster_gap', 'gap_plain'),
        ('uf', 'cluster_gap', 'gap_correlated'),
        ('mwpm', 'gap_plain', 'gap_correlated'),
    }


# --- endpoint replay ----------------------------------------------------------


def _endpoint_record() -> L1Record:
    """Two shots of two patches with hand-chosen initial and refined scores.

    Shot 0 has sigma = (0, 0); shot 1 has sigma = (1, 0). The reference is all
    zero, so the residual errors are the actual flips and sigma is the yoke.
    """
    cluster_gap = _columns([
        [(0.0, 3.0), (1.0, 3.0)],     # shot 0: q0 X = (0.9, 0.7), q0 Z = (0.1, 0.1)
        [(2.0, 1.0), (3.0, 0.0)],     # shot 1: q0 X = (0.3, 0.1), q0 Z = (0.7, 0.9)
    ])
    forced_plain = np.array([
        [_forced(1.0, -1.0), _forced(2.0, 1.0)],      # shot 0: q1 X = (0.2, 0.05), q1 Z = (0.95, 0.2)
        [_forced(0.0, 2.0), _forced(-1.0, 1.0)],      # shot 1: q1 X = (0.8, 0.95), q1 Z = (0.05, 0.2)
    ])
    return _record(shots=2, patches=2, cluster_gap=cluster_gap, forced_plain=forced_plain,
                   yoke=np.array([[False, False], [True, False]]))


def test_initial_only_final_bits_are_the_hand_worked_l2_answer():
    """Shot 0 X: q = (0.9, 0.7), sigma = 0 flips both. Shot 1 X: q = (0.3, 0.1), sigma = 1 flips patch 0."""
    result = replay(_endpoint_record(), _calibrators(), ReplayConfig(INITIAL, REFINED, NoRefinement()))
    np.testing.assert_array_equal(result.final, _columns([
        [(True, False), (True, False)],
        [(True, True), (False, True)],
    ]))
    assert not result.requested.any() and not result.refined_patches.any()
    assert not result.ties.any()
    assert result.reference == 'uf' and result.config.startswith('uf:cluster_gap->')


def test_all_refined_final_bits_are_the_hand_worked_l2_answer():
    """The refined scores reverse shot 0's X sector and move shot 1's X flip to patch 1."""
    result = replay(_endpoint_record(), _calibrators(), ReplayConfig(INITIAL, REFINED, RefineAll()))
    np.testing.assert_array_equal(result.final, _columns([
        [(False, True), (False, True)],
        [(False, False), (True, False)],
    ]))
    assert result.requested.all() and result.refined_patches.all()
    # q1 above one half: shot 0 Z patch 0 (0.95); shot 1 X patches 0 and 1 (0.8, 0.95).
    np.testing.assert_array_equal(result.above_half, [[[False, False], [True, False]],
                                                      [[True, True], [False, False]]])


def test_probabilities_above_one_half_flip_patches_under_an_unfired_yoke():
    """Section 7's rescue: sigma = 0 with two probabilities above one half flips both patches."""
    record = _record(shots=1, patches=2, cluster_gap=np.array([[0.0, 3.0, 1.0, 3.0]]))
    result = replay(record, _calibrators(), ReplayConfig(INITIAL, REFINED, NoRefinement()))
    np.testing.assert_array_equal(result.final, [[True, False, True, False]])
    assert not result.ties.any()


def test_ties_break_to_the_lowest_binary_value_and_are_reported():
    """With q = (0.5, 0.5) both sectors tie; X takes patch 0 under sigma = 1 and Z flips nothing."""
    record = _record(shots=1, patches=2, yoke=np.array([[True, False]]))
    result = replay(record, _calibrators(initial=TIED_KNOTS), ReplayConfig(INITIAL, REFINED, NoRefinement()))
    np.testing.assert_array_equal(result.ties, [[True, True]])
    np.testing.assert_array_equal(result.final, [[True, False, False, False]])


def _mixed_versus_restricted():
    """One shot of three patches, sigma = (1, 0), with only patch 1's X score requested."""
    record = _record(shots=1, patches=3,
                     cluster_gap=_columns([[(2.0, 3.0), (3.0, 3.0), (3.0, 3.0)]]),
                     forced_plain=np.array([[_forced(2.0, 2.0), _forced(1.0, 2.0), _forced(2.0, 2.0)]]),
                     yoke=np.array([[True, False]]))
    policy = _FixedRequest(mask=((False, True, False), (False, False, False)))
    return record, policy


def test_mixed_l2_may_flip_an_unrequested_patch_that_restricted_l2_cannot():
    record, policy = _mixed_versus_restricted()
    calibrators = _calibrators()
    mixed = replay(record, calibrators, ReplayConfig(INITIAL, REFINED, policy, outer='mixed'))
    restricted = replay(record, calibrators, ReplayConfig(INITIAL, REFINED, policy, outer='restricted'))
    # Mixed compares q0 = 0.3 at patch 0 with the refined 0.2 at patch 1 and flips patch 0.
    np.testing.assert_array_equal(mixed.final, _columns([[(True, False), (False, False), (False, False)]]))
    # Restricted may only flip a requested patch, so parity forces patch 1.
    np.testing.assert_array_equal(restricted.final, _columns([[(False, False), (True, False), (False, False)]]))


def test_restricted_l2_raises_when_odd_parity_has_no_requested_candidate():
    record, _ = _mixed_versus_restricted()
    config = ReplayConfig(INITIAL, REFINED, NoRefinement(), outer='restricted')
    with pytest.raises(ValueError, match='no candidate'):
        replay(record, _calibrators(), config)


def test_unrequested_refined_scores_do_not_change_the_answer():
    record, policy = _mixed_versus_restricted()
    changed = np.array(record.forced_plain, dtype=np.float64)
    changed[0, 0] = _forced(-1.0, -1.0)        # patch 0 is never requested
    changed[0, 2] = _forced(-1.0, -1.0)
    other = dataclasses.replace(record, forced_plain=changed)
    config = ReplayConfig(INITIAL, REFINED, policy)
    np.testing.assert_array_equal(replay(record, _calibrators(), config).final,
                                  replay(other, _calibrators(), config).final)


def test_replay_leaves_the_record_and_the_calibrators_unchanged():
    record = _endpoint_record()
    calibrators = _calibrators()
    before = {name: np.array(array) for name, array in record.arrays().items()}
    knots = {name: (np.array(pair[0].centers), np.array(pair[1].probabilities))
             for name, pair in calibrators.items()}
    replay(record, calibrators, ReplayConfig(INITIAL, REFINED, RefineAll()))
    for name, array in record.arrays().items():
        np.testing.assert_array_equal(array, before[name])
    for name, pair in calibrators.items():
        np.testing.assert_array_equal(pair[0].centers, knots[name][0])
        np.testing.assert_array_equal(pair[1].probabilities, knots[name][1])


def test_replay_is_deterministic():
    record, calibrators = _endpoint_record(), _calibrators()
    config = ReplayConfig(INITIAL, REFINED, RefineAll())
    first, second = replay(record, calibrators, config), replay(record, calibrators, config)
    assert first.config == second.config
    for field in ('final', 'requested', 'refined_patches', 'ties', 'above_half'):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    for field in (f.name for f in dataclasses.fields(WorkCounts)):
        np.testing.assert_array_equal(getattr(first.work, field), getattr(second.work, field))


def test_replay_results_are_read_only():
    result = replay(_endpoint_record(), _calibrators(), ReplayConfig(INITIAL, REFINED, RefineAll()))
    for field in ('final', 'requested', 'refined_patches', 'ties', 'above_half'):
        assert not getattr(result, field).flags.writeable
    assert not result.work.requested_patch_sectors.flags.writeable


def test_replay_rejects_a_final_whose_sector_parity_is_not_the_yoke(monkeypatch):
    """A guard: exact L2 cannot break parity, so the check only fires if L2 is wrong."""
    import yoked.hierarchical._replay as replay_module

    class _WrongParity:
        patterns = np.ones((2, 2), dtype=bool)
        tied = np.zeros((2,), dtype=bool)

    monkeypatch.setattr(replay_module, 'mwpm_outer_map_batch', lambda *args, **kwargs: _WrongParity())
    with pytest.raises(ValueError, match='parity'):
        replay(_endpoint_record(), _calibrators(), ReplayConfig(INITIAL, REFINED, NoRefinement()))


def test_replay_batches_shots_without_changing_the_answer(monkeypatch):
    import yoked.hierarchical._replay as replay_module

    record, calibrators = _endpoint_record(), _calibrators()
    config = ReplayConfig(INITIAL, REFINED, RefineAll())
    whole = replay(record, calibrators, config)
    monkeypatch.setattr(replay_module, 'REPLAY_BATCH_SHOTS', 1)
    np.testing.assert_array_equal(replay(record, calibrators, config).final, whole.final)
    assert REPLAY_BATCH_SHOTS > 1


# --- work accounting ----------------------------------------------------------

PER_PATCH_WORK = {
    ('uf', 'cluster_gap', 'gap_plain'): (
        dict(initial_uf_calls=1, initial_dijkstra_searches=2),
        dict(incremental_plain_forced_calls=4),
    ),
    ('uf', 'cluster_gap', 'gap_correlated'): (
        dict(initial_uf_calls=1, initial_dijkstra_searches=2),
        dict(incremental_unforced_plain_calls=1, incremental_reweight_passes=1,
             incremental_correlated_forced_calls=4),
    ),
    ('mwpm', 'gap_plain', 'gap_correlated'): (
        dict(initial_unforced_plain_calls=1, initial_plain_forced_calls=4),
        dict(incremental_reweight_passes=1, incremental_correlated_forced_calls=4),
    ),
}
"""The brief's table, restated here so the test does not read the implementation's copy."""


def _pair_config(pair, policy) -> ReplayConfig:
    reference, initial, refined = pair
    return ReplayConfig(Estimator(reference, initial), Estimator(reference, refined), policy)


def _six_patch_record(shots: int = 3) -> L1Record:
    rng = np.random.default_rng(5)
    return _record(shots=shots, patches=NUM_PATCHES,
                   cluster_gap=rng.integers(0, 4, size=(shots, 2 * NUM_PATCHES)).astype(np.float64),
                   dijkstra_states=rng.integers(1, 50, size=(shots, 2 * NUM_PATCHES)),
                   forced_plain=np.tile(_forced(1.0, 1.0), (shots, NUM_PATCHES, 1, 1)),
                   forced_correlated=np.tile(_forced(2.0, 0.0), (shots, NUM_PATCHES, 1, 1)))


def _pair_calibrators(pair) -> dict:
    reference, initial, refined = pair
    return _calibrators(initial_estimator=Estimator(reference, initial),
                        refined_estimator=Estimator(reference, refined))


@pytest.mark.parametrize('pair', sorted(PER_PATCH_WORK))
@pytest.mark.parametrize('policy, requested, distinct', [(NoRefinement(), 0, 0), (RefineAll(), 12, 6)])
def test_endpoint_work_counts_follow_the_specified_table(pair, policy, requested, distinct):
    record = _six_patch_record()
    result = replay(record, _pair_calibrators(pair), _pair_config(pair, policy))
    initial_work, increment = PER_PATCH_WORK[pair]
    counts = result.work
    np.testing.assert_array_equal(counts.requested_patch_sectors, [requested] * record.shots)
    np.testing.assert_array_equal(counts.distinct_patches, [distinct] * record.shots)
    for field in (f.name for f in dataclasses.fields(WorkCounts)):
        if field in ('requested_patch_sectors', 'distinct_patches', 'initial_dijkstra_states'):
            continue
        expected = (initial_work.get(field, 0) * NUM_PATCHES if field.startswith('initial_')
                    else increment.get(field, 0) * distinct)
        np.testing.assert_array_equal(getattr(counts, field), [expected] * record.shots,
                                      err_msg=f'{pair} {policy.name} {field}')


@pytest.mark.parametrize('pair', sorted(PER_PATCH_WORK))
def test_initial_dijkstra_states_are_charged_only_to_the_cluster_gap(pair):
    record = _six_patch_record()
    result = replay(record, _pair_calibrators(pair), _pair_config(pair, RefineAll()))
    expected = record.dijkstra_states.sum(axis=1) if pair[1] == 'cluster_gap' else np.zeros(record.shots)
    np.testing.assert_array_equal(result.work.initial_dijkstra_states, expected)


def test_both_sectors_of_one_patch_cost_one_refinement():
    record = _six_patch_record(shots=1)
    pair = ('uf', 'cluster_gap', 'gap_correlated')
    one_patch = _FixedRequest(mask=tuple(tuple(i == 0 for i in range(NUM_PATCHES)) for _ in range(2)))
    two_patches = _FixedRequest(mask=(tuple(i == 0 for i in range(NUM_PATCHES)),
                                      tuple(i == 1 for i in range(NUM_PATCHES))))
    calibrators = _pair_calibrators(pair)
    shared = replay(record, calibrators, _pair_config(pair, one_patch)).work
    split = replay(record, calibrators, _pair_config(pair, two_patches)).work
    assert int(shared.requested_patch_sectors[0]) == int(split.requested_patch_sectors[0]) == 2
    assert int(shared.distinct_patches[0]) == 1 and int(split.distinct_patches[0]) == 2
    assert int(shared.incremental_correlated_forced_calls[0]) == 4
    assert int(split.incremental_correlated_forced_calls[0]) == 8


def test_work_counts_serialize_with_totals_and_per_shot_means():
    record = _six_patch_record(shots=4)
    pair = ('uf', 'cluster_gap', 'gap_plain')
    counts = replay(record, _pair_calibrators(pair), _pair_config(pair, RefineAll())).work
    totals = counts.totals()
    assert totals['requested_patch_sectors'] == 12 * record.shots
    assert totals['incremental_plain_forced_calls'] == 24 * record.shots
    payload = counts.to_json()
    assert payload['shots'] == record.shots
    assert payload['totals'] == totals
    assert payload['per_shot_mean']['distinct_patches'] == pytest.approx(6.0)
    json.dumps(payload)     # the summary writes this straight out


# --- replay work is not collection work ---------------------------------------


@pytest.fixture(scope='module')
def distance3_collection():
    """A real distance-3 record of a few dozen shots, with the work that actually ran."""
    fixture = yoked_fixture(shots=32)
    context = L1Context(fixture.dem, NUM_PATCHES)
    return collect_rows(context, fixture.detectors, fixture.actual, np.arange(32))


def test_replay_of_a_collected_record_keeps_the_yoke_parity(distance3_collection):
    record = distance3_collection.record
    estimators = [INITIAL, Estimator('uf', 'gap_correlated')]
    calibrators = fit_calibrators(record, estimators)
    for policy in (NoRefinement(), RefineAll()):
        result = replay(record, calibrators, ReplayConfig(estimators[0], estimators[1], policy))
        parity = by_sector(result.final).sum(axis=2) % 2
        np.testing.assert_array_equal(parity.astype(bool), record.yoke)
        assert result.final.shape == record.actual.shape


def test_replay_work_is_a_separate_object_from_collection_work(distance3_collection):
    """The replay counts describe the specified procedure; they are not the calls that ran."""
    record, collected = distance3_collection.record, distance3_collection.work
    assert set(WORK_FIELDS).isdisjoint(f.name for f in dataclasses.fields(WorkCounts))
    pair = ('uf', 'cluster_gap', 'gap_correlated')
    counts = replay(record, fit_calibrators(record, [Estimator(*(pair[0], pair[1])),
                                                     Estimator(*(pair[0], pair[2]))]),
                    _pair_config(pair, NoRefinement())).work.totals()
    # Collection ran forced and validation matchings for every patch; initial-only replay charges none.
    assert collected.plain_forced_calls == 4 * NUM_PATCHES * record.shots
    assert collected.correlated_validation_calls > 0 and collected.joint_decodes == record.shots
    assert counts['initial_plain_forced_calls'] == 0
    assert counts['incremental_correlated_forced_calls'] == 0
    assert counts['initial_uf_calls'] == NUM_PATCHES * record.shots == collected.uf_decodes


def test_replay_result_rejects_inconsistent_shapes():
    shots, patches = 2, 3
    work = WorkCounts(**{f.name: np.zeros(shots, dtype=np.int64) for f in dataclasses.fields(WorkCounts)})
    good = dict(config='c', reference='uf', final=np.zeros((shots, 2 * patches), dtype=bool),
                requested=np.zeros((shots, 2 * patches), dtype=bool),
                refined_patches=np.zeros((shots, patches), dtype=bool),
                ties=np.zeros((shots, 2), dtype=bool),
                above_half=np.zeros((shots, 2, patches), dtype=bool), work=work)
    ReplayResult(**good)
    with pytest.raises(ValueError):
        ReplayResult(**{**good, 'ties': np.zeros((shots + 1, 2), dtype=bool)})
    with pytest.raises(ValueError):
        ReplayResult(**{**good, 'refined_patches': np.zeros((shots, patches + 1), dtype=bool)})
    with pytest.raises(ValueError, match='reference'):
        ReplayResult(**{**good, 'reference': 'joint'})
