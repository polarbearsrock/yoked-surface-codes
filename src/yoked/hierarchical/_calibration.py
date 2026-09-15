"""Isotonic calibration of soft-output scores to residual-error probabilities.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 6.

A calibrator maps a score to P(e = 1 | score) with a monotone map fit by
pool-adjacent-violators (PAV) on the calibration sample only. The direction
is fixed by the score's definition: 'decreasing' for gaps, since a larger
gap means a smaller error probability. Equal scores are pooled before the
fit. PAV merges strict violations only; equal-valued adjacent blocks stay as
separate knots. Between block centres the map is linear; beyond the fitted range it is
constant. Outputs are clipped to [CLIP, 1 - CLIP] so that log-odds stay
finite. Probabilities above one half are allowed: a refined signed gap can
favor reversing the fixed reference.

``_calibration_test.py`` checks that the fit recovers a monotone step
function, is monotone in both directions including flat blocks and
clipping, pools equal scores, rejects non-finite input, and round-trips
through JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yoked.hierarchical._arrays import readonly_array

CLIP = 1e-6
"""Keeps log-odds finite; far below any residual-error rate this experiment can resolve."""

DIRECTIONS = ('increasing', 'decreasing')
"""The only two monotone senses a calibrator can be fit in; anything else is a caller error."""

KNOT_CONVENTION = ('pav-blocks-of-distinct-scores; knot at each block weighted-mean score; '
                   'linear between knots; constant beyond; clipped to [1e-6, 1 - 1e-6]')
"""Exactly the rule ``fit`` and ``probability`` implement, written down so that a stage can
record it and a later stage can refuse knots fitted under a different one. Two calibrators
with the same knots but different rules between them are different maps, and nothing in the
knot arrays themselves would say so. It is part of the calibration identity, so changing the
string without changing the rule invalidates stored calibrations, and changing the rule
without changing the string would let an incompatible fit pass as this one."""


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A fitted monotone map from score to residual-error probability.

    Fields: ``direction`` in DIRECTIONS; ``centers`` (blocks,) increasing
    scores, each the weighted mean score of one PAV block; ``probabilities``
    (blocks,) the block means clipped to [CLIP, 1 - CLIP], monotone in
    ``direction``; ``num_samples`` used for the fit.
    """
    direction: str
    centers: np.ndarray
    probabilities: np.ndarray
    num_samples: int

    def __post_init__(self) -> None:
        # Validate direction and num_samples on their raw values before any cast could hide a
        # type problem (e.g. int(True) == 1 would silently pass a positive-integer check).
        if self.direction not in DIRECTIONS:
            raise ValueError('Invalid direction')
        if (isinstance(self.num_samples, (bool, np.bool_))
                or not isinstance(self.num_samples, (int, np.integer)) or self.num_samples < 1):
            raise ValueError('num_samples must be a positive integer')
        centers = readonly_array(self.centers, dtype=np.float64)
        probabilities = readonly_array(self.probabilities, dtype=np.float64)
        if centers.ndim != 1 or not len(centers) or probabilities.shape != centers.shape:
            raise ValueError('Expected nonempty, equally sized knot arrays')
        if not np.isfinite(centers).all() or not (np.diff(centers) > 0).all():
            raise ValueError('Knot centers must be finite and strictly increasing')
        if not ((probabilities >= CLIP) & (probabilities <= 1 - CLIP)).all():
            raise ValueError('Probabilities must be finite and inside the clipping range')
        differences = np.diff(probabilities)
        violates_order = (differences < 0).any() if self.direction == 'increasing' else (differences > 0).any()
        if violates_order:
            raise ValueError('Probabilities must be monotone')
        object.__setattr__(self, 'centers', centers)
        object.__setattr__(self, 'probabilities', probabilities)

    @classmethod
    def fit(cls, scores, outcomes, *, direction: str) -> IsotonicCalibrator:
        if direction not in DIRECTIONS:
            raise ValueError(f'direction must be one of {DIRECTIONS}')
        scores = np.asarray(scores, dtype=np.float64).ravel()
        outcomes = np.asarray(outcomes, dtype=np.float64).ravel()
        if len(scores) != len(outcomes):
            raise ValueError('scores and outcomes must have the same length')
        if len(scores) == 0:
            raise ValueError('Cannot fit a calibrator on an empty sample')
        if not np.isfinite(scores).all():
            raise ValueError('Scores must be finite')
        if not np.isin(outcomes, (0.0, 1.0)).all():
            raise ValueError('Outcomes must be 0 or 1')
        # Pool repeated scores into one weighted point per distinct value before PAV, so that
        # equal-valued inputs become a single candidate knot rather than several coincident ones.
        unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=outcomes) / counts
        order = slice(None) if direction == 'increasing' else slice(None, None, -1)
        # PAV fits a non-decreasing sequence; for a decreasing map, fit it on scores in reverse order.
        centers, values = _pool_adjacent_violators(unique[order], means[order], counts[order])
        return cls(direction, centers[order], np.clip(values[order], CLIP, 1 - CLIP), int(len(scores)))

    def probability(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('Scores must be finite')
        return np.interp(scores, self.centers, self.probabilities)

    def to_json(self) -> dict:
        return dict(direction=self.direction, centers=self.centers.tolist(),
                    probabilities=self.probabilities.tolist(), num_samples=self.num_samples)

    @classmethod
    def from_json(cls, data: dict) -> IsotonicCalibrator:
        return cls(data['direction'], np.asarray(data['centers'], dtype=np.float64),
                   np.asarray(data['probabilities'], dtype=np.float64), data['num_samples'])


def _pool_adjacent_violators(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted non-decreasing fit of y against the order of x.

    Returns the weighted mean x and the fitted value of each block. Adjacent
    blocks whose values violate the order are merged until none do.
    """
    centers: list[float] = []
    values: list[float] = []
    weights: list[float] = []
    for xi, yi, wi in zip(x, y, w):
        centers.append(float(xi))
        values.append(float(yi))
        weights.append(float(wi))
        while len(values) > 1 and values[-2] > values[-1]:
            total = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total
            centers[-2] = (centers[-2] * weights[-2] + centers[-1] * weights[-1]) / total
            weights[-2] = total
            del values[-1], centers[-1], weights[-1]
    return np.array(centers), np.array(values)
