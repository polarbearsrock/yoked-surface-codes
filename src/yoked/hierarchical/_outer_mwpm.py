"""PyMatching L2 decoder for the factorized single-yoke model.

Each row is decoded on a subdivided star.  Subdivision keeps one distinct
fault id per patch; direct parallel edges from the yoke detector to boundary
would be merged by matching graph implementations.  Probabilities above one
half are preflipped, leaving nonnegative absolute log-likelihood penalties.

PyMatching uses finite-precision integer weights internally.  We therefore
audit its answer in the original float penalties.  A quantized answer outside
the documented tolerance is repaired to the analytic strict optimum.  Rows
whose first two optima are within the tolerance use a polynomial conditional
minimum resolver to preserve the public lowest-binary tie rule.  Matching is
still run for every feasible row; neither repair enumerates patterns.
"""
from __future__ import annotations

import numpy as np
import pymatching

from yoked.hierarchical._outer_decoder import (
    BatchOuterDecision,
    OuterDecision,
    TIE_TOLERANCE,
)


def mwpm_outer_map(
        q: np.ndarray, parity: int, candidates: np.ndarray | None = None,
) -> OuterDecision:
    """Decode one sector using the subdivided-star MWPM representation."""
    cs = None if candidates is None else np.asarray(candidates)[None]
    result = mwpm_outer_map_batch(
        np.asarray(q, dtype=np.float64)[None], np.asarray([parity]), cs)
    return OuterDecision(result.patterns[0], bool(result.tied[0]))


def mwpm_outer_map_batch(
        q: np.ndarray, parity: np.ndarray, candidates: np.ndarray | None = None,
) -> BatchOuterDecision:
    """Decode rows of probabilities, preserving the exact decoder's tie rule."""
    q = np.asarray(q, dtype=np.float64)
    parity = np.asarray(parity)
    if parity.dtype.kind not in 'buif' or not np.isin(parity, (0, 1)).all():
        raise ValueError('Parity must contain binary values')
    parity = parity.astype(bool, copy=False)
    if q.ndim != 2 or parity.shape != (q.shape[0],):
        raise ValueError('Expected q of shape (shots, patches) and parity of shape (shots,)')
    if not ((q > 0) & (q < 1)).all():
        raise ValueError('Probabilities must lie strictly inside (0, 1)')
    if candidates is None:
        candidates = np.ones(q.shape, dtype=bool)
    else:
        candidates = np.asarray(candidates)
        if candidates.shape != q.shape:
            raise ValueError('candidates must have the same shape as q')
        if candidates.dtype.kind not in 'buif' or not np.isin(candidates, (0, 1)).all():
            raise ValueError('Candidates must contain binary values')
        candidates = candidates.astype(bool, copy=False)

    patterns = np.zeros(q.shape, dtype=bool)
    tied = np.zeros(q.shape[0], dtype=bool)
    for shot in range(q.shape[0]):
        allowed = candidates[shot]
        baseline = (q[shot] > 0.5) & allowed
        costs = np.abs(np.log1p(-q[shot]) - np.log(q[shot]))
        residual = bool(parity[shot]) ^ bool(np.count_nonzero(baseline) & 1)
        indices = np.flatnonzero(allowed)
        if residual and not len(indices):
            raise ValueError('Odd parity with no candidate patch: no feasible pattern')

        matching = _matching_for_costs(costs, indices)
        syndrome = np.zeros(1 + len(indices), dtype=np.uint8)
        syndrome[0] = residual
        toggles = np.asarray(matching.decode(syndrome), dtype=bool)
        if (toggles.shape != costs.shape or toggles[~allowed].any()
                or bool(np.count_nonzero(toggles) & 1) != residual):
            raise ValueError('MWPM returned an invalid patch correction')

        sorted_costs = np.sort(costs[indices])
        if residual:
            strict = sorted_costs[0]
            is_tied = len(sorted_costs) >= 2 and sorted_costs[1] - strict <= TIE_TOLERANCE
        else:
            strict = 0.0
            is_tied = len(sorted_costs) >= 2 and sorted_costs[0] + sorted_costs[1] <= TIE_TOLERANCE
        penalty = float(costs[toggles].sum())

        if is_tied:
            patterns[shot] = _canonical_pattern(
                baseline, costs, allowed, bool(parity[shot]), strict)
            tied[shot] = True
        elif penalty > strict + TIE_TOLERANCE:
            # Repair a choice reversed by PyMatching's integer quantization.
            toggles[:] = False
            if residual:
                toggles[indices[np.argmin(costs[indices])]] = True
            patterns[shot] = baseline ^ toggles
        else:
            patterns[shot] = baseline ^ toggles
    return BatchOuterDecision(patterns, tied)


def _matching_for_costs(costs: np.ndarray, indices: np.ndarray) -> pymatching.Matching:
    """Build a star whose weighted arms retain the original patch fault ids."""
    matching = pymatching.Matching()
    matching.ensure_num_fault_ids(len(costs))
    if not len(indices):
        # Materialize the central detector so the all-zero syndrome is decoded.
        matching.add_boundary_edge(0, weight=0.0, fault_ids=set())
        return matching
    for auxiliary, patch in enumerate(indices, start=1):
        matching.add_edge(0, auxiliary, weight=float(costs[patch]), fault_ids={int(patch)})
        matching.add_boundary_edge(auxiliary, weight=0.0, fault_ids=set())
    return matching


def _canonical_pattern(
        baseline: np.ndarray,
        costs: np.ndarray,
        candidates: np.ndarray,
        parity: bool,
        strict: float,
) -> np.ndarray:
    """Return the lowest binary pattern within tolerance of the strict optimum."""
    n = len(baseline)
    fixed = np.full(n, -1, dtype=np.int8)
    limit = strict + TIE_TOLERANCE

    def completion_cost() -> float:
        assigned = fixed >= 0
        if np.any((fixed == 1) & ~candidates):
            return np.inf
        result = float(costs[assigned & (fixed.astype(bool) != baseline)].sum())
        preferred_parity = int(np.count_nonzero(baseline[~assigned]) & 1)
        required = int(parity) ^ int(np.count_nonzero(fixed[assigned]) & 1)
        if preferred_parity != required:
            free = (~assigned) & candidates
            result += float(np.min(costs[free], initial=np.inf))
        return result

    for bit in range(n - 1, -1, -1):
        fixed[bit] = 0
        if completion_cost() > limit:
            fixed[bit] = 1
    return fixed.astype(bool)
