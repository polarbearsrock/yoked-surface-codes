"""Exact L2 decoding for the factorized patch model.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 7.

Per sector, n residual-error probabilities q_i and the frame-adjusted
syndrome sigma (the parity of the residual errors) define the weight

    w(x) = prod_i q_i^{x_i} (1 - q_i)^{1 - x_i}

over residual patterns x in {0,1}^n with parity(x) = sigma. L2 returns the
maximum-weight feasible pattern. With a candidate mask, non-candidates are
fixed to x_i = 0. Log weights within TIE_TOLERANCE of the maximum tie; ties
are broken toward the lowest binary value (bit i is patch i) and reported.
Multiple flips are allowed, including for sigma = 0.

The frame-adjusted syndrome is sigma[s] = y[s] XOR parity(r[:, s]).

``_outer_decoder_test.py`` checks the enumeration against an analytic
threshold-and-parity rule, the fixtures named in the spec, candidate
restrictions, and the infeasible case.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from yoked.hierarchical._arrays import readonly_array

TIE_TOLERANCE = 1e-9
"""Log-weight differences below this are ties: far below any calibrated probability's resolution."""

MAX_PATCHES = 16
"""Enumeration builds 2**n patterns per shot; 16 keeps a batch to 65,536 columns."""


@dataclass(frozen=True)
class OuterDecision:
    """L2's answer for one sector: ``pattern`` (n,) bool and whether it was ``tied``."""
    pattern: np.ndarray
    tied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, 'pattern', readonly_array(self.pattern, dtype=bool))


@dataclass(frozen=True)
class BatchOuterDecision:
    """Read-only L2 outputs: patterns (shots, n) bool and tied (shots,) bool."""
    patterns: np.ndarray
    tied: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'patterns', readonly_array(self.patterns, dtype=bool))
        object.__setattr__(self, 'tied', readonly_array(self.tied, dtype=bool))


def frame_adjusted_syndrome(yoke: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """sigma[s] = y[s] XOR parity of r over patches, for yoke (..., 2) and reference (..., 2 * patches)."""
    yoke, reference = np.asarray(yoke), np.asarray(reference)
    if (yoke.ndim < 1 or reference.ndim < 1 or yoke.shape[-1] != 2
            or reference.shape[-1] % 2 or yoke.shape[:-1] != reference.shape[:-1]):
        raise ValueError('Expected matching leading shapes, two yoke bits, and two bits per patch')
    for values in (yoke, reference):
        if values.dtype.kind not in 'buif' or not np.isin(values, (0, 1)).all():
            raise ValueError('Expected binary values')
    yoke, reference = yoke.astype(bool), reference.astype(bool)
    by_patch = reference.reshape(reference.shape[:-1] + (-1, 2))   # (..., patches, sectors)
    return yoke ^ (by_patch.sum(axis=-2) % 2).astype(bool)


def exact_outer_map(q: np.ndarray, parity: int, candidates: np.ndarray | None = None) -> OuterDecision:
    """Maximum-weight residual pattern of the required parity for one sector."""
    batch_candidates = None if candidates is None else np.asarray(candidates)[None]
    result = exact_outer_map_batch(np.asarray(q, dtype=np.float64)[None], np.asarray([parity]), batch_candidates)
    return OuterDecision(result.patterns[0], bool(result.tied[0]))


def exact_outer_map_batch(
        q: np.ndarray, parity: np.ndarray, candidates: np.ndarray | None = None,
) -> BatchOuterDecision:
    """Vectorized ``exact_outer_map``: q (shots, n), parity (shots,), candidates (shots, n) or None."""
    q = np.asarray(q, dtype=np.float64)
    parity = np.asarray(parity)
    if parity.dtype.kind not in 'buif' or not np.isin(parity, (0, 1)).all():
        raise ValueError('Parity must contain binary values')
    parity = parity.astype(bool, copy=False)
    if q.ndim != 2 or parity.shape != (q.shape[0],):
        raise ValueError('Expected q of shape (shots, patches) and parity of shape (shots,)')
    if q.shape[1] > MAX_PATCHES:
        raise ValueError(f'At most {MAX_PATCHES} patches are supported')
    if not ((q > 0) & (q < 1)).all():
        raise ValueError('Probabilities must lie strictly inside (0, 1)')
    patterns = _patterns(q.shape[1])                                             # (P, n)
    log_weight = np.log(q) @ patterns.T.astype(np.float64) + np.log1p(-q) @ (~patterns).T.astype(np.float64)
    feasible = ((patterns.sum(axis=1) % 2) == 1)[None, :] == parity[:, None]     # (shots, P)
    if candidates is not None:
        candidates = np.asarray(candidates)
        if candidates.shape != q.shape:
            raise ValueError('candidates must have the same shape as q')
        if candidates.dtype.kind not in 'buif' or not np.isin(candidates, (0, 1)).all():
            raise ValueError('Candidates must contain binary values')
        candidates = candidates.astype(bool, copy=False)
        feasible &= ~(patterns[None, :, :] & ~candidates[:, None, :]).any(axis=2)
    if not feasible.any(axis=1).all():
        raise ValueError('Odd parity with no candidate patch: no feasible pattern')
    log_weight = np.where(feasible, log_weight, -np.inf)
    best = log_weight.max(axis=1, keepdims=True)
    near = log_weight >= best - TIE_TOLERANCE
    chosen = near.argmax(axis=1)   # the first tie in pattern order has the lowest binary value
    return BatchOuterDecision(patterns[chosen], near.sum(axis=1) > 1)


@functools.lru_cache(maxsize=None)
def _patterns(n: int) -> np.ndarray:
    """All 2**n patterns as a (2**n, n) bool array whose row p has bit i equal to (p >> i) & 1.

    Read-only: the cache hands the same array object to every caller for a
    given n, so an in-place edit would otherwise corrupt it for the rest of
    the process.
    """
    raw = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    return readonly_array(raw, dtype=bool)
