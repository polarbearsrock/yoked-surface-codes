"""Refinement policies: the request mask M a replay hands to the refined estimator.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 8.

A policy sees only the initial calibrated probabilities ``q0`` of shape
(shots, 2, P) and the frame-adjusted syndrome ``sigma`` of shape (shots, 2).
It never sees the refined scores, the actual flips, or the outcome. It returns
the request mask ``M`` of shape (shots, 2, P): ``M[shot, s, i]`` is true when
patch ``i``'s refined score replaces its initial score in sector ``s``.

This module holds the two endpoint policies of the pilot and nothing else.
``NoRefinement`` (``initial_only``) requests nothing and ``RefineAll``
(``all_refined``) requests every patch-sector regardless of ``sigma``; they
bracket the accuracy any selective policy can reach on the same reference.
The selective policies (``top_k_given_yoke``, ``top_k_uncertain``) and the
random controls of the spec belong to a later milestone, so
``policy_from_name`` raises on their names rather than offering a stub that
would silently behave like an endpoint.

``_policies_test.py`` checks that each endpoint requests what its name says
whatever ``sigma`` holds, that the returned masks are owned and read-only,
that ``select`` rejects layouts and values it cannot interpret, and that
``policy_from_name`` names an unsupported policy in its error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._patch_graphs import NUM_SECTORS

INITIAL_ONLY = 'initial_only'
"""The baseline endpoint of section 8: L2 runs on the initial probabilities alone."""

ALL_REFINED = 'all_refined'
"""The full-refinement comparator of section 8: every patch-sector is refined."""


@runtime_checkable
class DeterministicPolicy(Protocol):
    """A policy whose request mask is a function of ``q0`` and ``sigma`` alone.

    ``name`` identifies the policy in a configuration name and in a report;
    ``select`` returns the (shots, 2, P) boolean request mask M. Random
    controls are not deterministic policies and are not modelled here.
    """

    name: str

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        ...


def _checked_inputs(q0, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Validate a policy's inputs before it reads them, and return owned views.

    Values are checked before any cast: a probability of exactly 0 or 1 would
    give L2 an infinite log weight, and a sigma bit of 2 would silently become
    True. Both are caller errors, not values to round.
    """
    q0 = np.asarray(q0)
    sigma = np.asarray(sigma)
    if q0.ndim != 3 or q0.shape[0] < 1 or q0.shape[1] != NUM_SECTORS or q0.shape[2] < 1:
        raise ValueError(f'q0 must have shape (shots >= 1, {NUM_SECTORS}, patches >= 1), got {q0.shape}')
    if sigma.shape != (q0.shape[0], NUM_SECTORS):
        raise ValueError(f'sigma must have shape {(q0.shape[0], NUM_SECTORS)}, got {sigma.shape}')
    if q0.dtype.kind != 'f' or not ((q0 > 0) & (q0 < 1)).all():
        raise ValueError('q0 must hold probabilities strictly inside (0, 1)')
    if sigma.dtype.kind not in 'buif' or not np.isin(sigma, (0, 1)).all():
        raise ValueError('sigma must hold binary values')
    return q0, sigma.astype(bool)


@dataclass(frozen=True)
class NoRefinement:
    """The ``initial_only`` endpoint: no patch-sector is refined."""

    name: ClassVar[str] = INITIAL_ONLY

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """An all-false (shots, 2, P) request mask."""
        q0, _ = _checked_inputs(q0, sigma)
        return readonly_array(np.zeros(q0.shape, dtype=bool), dtype=bool)


@dataclass(frozen=True)
class RefineAll:
    """The ``all_refined`` endpoint: every patch-sector is refined, including sigma = 0."""

    name: ClassVar[str] = ALL_REFINED

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """An all-true (shots, 2, P) request mask."""
        q0, _ = _checked_inputs(q0, sigma)
        return readonly_array(np.ones(q0.shape, dtype=bool), dtype=bool)


POLICIES = {INITIAL_ONLY: NoRefinement, ALL_REFINED: RefineAll}
"""Every policy this pilot supports. Selective policies and random controls are
deliberately absent: a name that is not here raises instead of resolving to an
endpoint that would be reported under the wrong label."""


def policy_from_name(name: str) -> DeterministicPolicy:
    """The endpoint policy called ``name``, or a ``ValueError`` naming it."""
    if name not in POLICIES:
        raise ValueError(f'Unsupported policy {name!r}; this pilot implements {sorted(POLICIES)}')
    return POLICIES[name]()
