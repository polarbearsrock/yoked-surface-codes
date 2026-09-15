"""Tests for the two endpoint refinement policies (_policies.py).

Checks: ``NoRefinement`` requests nothing and ``RefineAll`` requests every
patch-sector, whatever the frame-adjusted syndrome says; both return owned
read-only masks that are not shared between calls; ``select`` rejects a q0
that is not (shots, 2, P), a sigma whose shape disagrees with it, non-binary
sigma bits, and probabilities outside (0, 1); ``policy_from_name`` builds
exactly these two endpoints and names an unsupported policy in its error; and
both endpoints satisfy the ``DeterministicPolicy`` protocol.
"""
import numpy as np
import pytest

from yoked.hierarchical._policies import (
    ALL_REFINED, INITIAL_ONLY, DeterministicPolicy, NoRefinement, RefineAll, policy_from_name,
)

SHOTS, PATCHES = 3, 4


def _q0(shots: int = SHOTS, patches: int = PATCHES) -> np.ndarray:
    """Distinct probabilities strictly inside (0, 1), in sector-major layout."""
    values = np.linspace(0.05, 0.95, shots * 2 * patches)
    return values.reshape(shots, 2, patches)


def _sigma(bits=(0, 1)) -> np.ndarray:
    return np.tile(np.asarray(bits, dtype=bool), (SHOTS, 1))


def test_no_refinement_requests_nothing():
    mask = NoRefinement().select(_q0(), _sigma())
    assert mask.shape == (SHOTS, 2, PATCHES)
    assert mask.dtype == bool
    assert not mask.any()


def test_refine_all_requests_every_patch_sector():
    mask = RefineAll().select(_q0(), _sigma())
    assert mask.shape == (SHOTS, 2, PATCHES)
    assert mask.all()


def test_endpoints_ignore_the_frame_adjusted_syndrome():
    """all_refined refines both sectors even where sigma = 0, and initial_only neither."""
    for sigma in (_sigma((0, 0)), _sigma((1, 1))):
        assert RefineAll().select(_q0(), sigma).all()
        assert not NoRefinement().select(_q0(), sigma).any()


def test_masks_are_read_only_and_not_shared_between_calls():
    policy = RefineAll()
    first = policy.select(_q0(), _sigma())
    second = policy.select(_q0(), _sigma())
    assert not first.flags.writeable
    assert first is not second
    with pytest.raises(ValueError):
        first[0, 0, 0] = False


def test_policy_names():
    assert NoRefinement().name == INITIAL_ONLY == 'initial_only'
    assert RefineAll().name == ALL_REFINED == 'all_refined'


@pytest.mark.parametrize('policy', [NoRefinement(), RefineAll()])
def test_select_rejects_a_q0_that_is_not_sector_major(policy):
    with pytest.raises(ValueError, match='shots'):
        policy.select(np.full((SHOTS, 2 * PATCHES), 0.5), _sigma())
    with pytest.raises(ValueError, match='shots'):
        policy.select(np.full((SHOTS, 3, PATCHES), 0.5), _sigma())


@pytest.mark.parametrize('policy', [NoRefinement(), RefineAll()])
def test_select_rejects_a_sigma_that_does_not_match_q0(policy):
    with pytest.raises(ValueError, match='sigma'):
        policy.select(_q0(), np.zeros((SHOTS + 1, 2), dtype=bool))
    with pytest.raises(ValueError, match='sigma'):
        policy.select(_q0(), np.zeros((SHOTS,), dtype=bool))


@pytest.mark.parametrize('policy', [NoRefinement(), RefineAll()])
def test_select_rejects_non_binary_sigma(policy):
    sigma = np.full((SHOTS, 2), 2, dtype=np.int64)
    with pytest.raises(ValueError, match='binary'):
        policy.select(_q0(), sigma)


@pytest.mark.parametrize('policy', [NoRefinement(), RefineAll()])
@pytest.mark.parametrize('bad', [0.0, 1.0, -0.5, np.nan])
def test_select_rejects_probabilities_outside_the_open_unit_interval(policy, bad):
    q0 = _q0()
    q0[1, 0, 2] = bad
    with pytest.raises(ValueError, match='probabilit'):
        policy.select(q0, _sigma())


def test_policy_from_name_builds_the_two_endpoints():
    assert policy_from_name(INITIAL_ONLY) == NoRefinement()
    assert policy_from_name(ALL_REFINED) == RefineAll()


@pytest.mark.parametrize('name', ['top_2_given_yoke', 'random_1_unconditional', 'initial', ''])
def test_policy_from_name_names_the_unsupported_policy(name):
    """Selective and random policies belong to a later plan: no stub may stand in."""
    with pytest.raises(ValueError, match=repr(name)):
        policy_from_name(name)


def test_endpoints_satisfy_the_deterministic_policy_protocol():
    assert isinstance(NoRefinement(), DeterministicPolicy)
    assert isinstance(RefineAll(), DeterministicPolicy)
    assert not isinstance(object(), DeterministicPolicy)
