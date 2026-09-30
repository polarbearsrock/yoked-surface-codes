"""Native PyMatching correlated predictions with frozen-weight complementary gaps.

Section 4 of https://arxiv.org/abs/2312.04522 compares logical classes on one
reweighted graph. Calling correlated matching separately in the forced classes
would change the conditioning correction and would not compute that gap.

PyMatching 2.4 exposes correlated predictions, but not its temporary weights.
The small adapter below reconstructs its pairwise conditional weights and integer
discretization from the DEM. First-pass edges and the reference prediction both
come from the same native, correlation-enabled Matching object. All forced
matches use PyMatching too. Every decode checks the frozen graph's optimum and
reference-class cost against the native correlated solve before returning a gap.

The adapter follows PyMatching v2.4.0's driver/user_graph.cc (joint probabilities,
normalization), driver/user_graph.h (discretization), and search/search_graph.cc
(implied-weight conversion). Support is deliberately limited to the 2.4 series;
an upstream numerical change needs validation, not a silent fallback.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pymatching

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._matching_gaps import CHECK_PATTERNS, _CheckMatrixGraph, signed_gaps
from yoked.hierarchical._patch_graphs import PatchGraph


_MAX_HALF_WEIGHT = (1 << 24) - 1
"""PyMatching 2.4 uses uint32 weights with eight bits reserved for arithmetic."""


def _edge_key(u: int, v: int | None) -> tuple[int, int | None]:
    """Use one key for an undirected edge and PyMatching's two boundary spellings."""
    return (int(u), None) if v is None or v == -1 else tuple(sorted((int(u), int(v))))


def _xor_probability(a: float, b: float) -> float:
    # Independent mechanisms toggle an edge; two occurrences cancel.
    return a * (1 - b) + b * (1 - a)


class _FrozenNativeWeights:
    """Reconstruct PyMatching 2.4's nonnegative integer weights in patch edge order."""

    def __init__(self, patch: PatchGraph, matching: pymatching.Matching):
        self.edge_ids = {_edge_key(u, v): i for i, (u, v, _, _) in enumerate(patch.graph.edges)}
        native = {_edge_key(u, v): data for u, v, data in matching.edges()}
        if native.keys() != self.edge_ids.keys():
            raise ValueError('Native correlated graph and patch graph have different edges')
        weights = []
        for u, v, _, mask in patch.graph.edges:
            data = native[_edge_key(u, v)]
            if sum(1 << k for k in data['fault_ids']) != mask:
                raise ValueError('Native correlated graph and patch graph have different logical labels')
            weights.append(data['weight'])
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Complementary-gap decoding requires finite nonnegative edge weights')

        rules = self._conditional_rules(patch)
        # Include even non-discounting rules: native normalization includes them.
        all_weights = [*weights, *(weight for _, _, weight in rules)]
        maximum = max(all_weights, default=0.0)
        if maximum > _MAX_HALF_WEIGHT or not math.isfinite(maximum):
            raise ValueError('Edge or implied weight exceeds PyMatching 2.4 weight range')
        all_integral = all(float(weight).is_integer() for weight in all_weights)
        self.scale = 1.0 if all_integral else _MAX_HALF_WEIGHT / maximum
        # C++ round, not NumPy's ties-to-even rounding. Store half of the even
        # internal weights; supplying integer weights to another matcher is exact.
        self.weights = np.floor(weights * self.scale + 0.5).astype(np.int64)
        self.rules: list[list[tuple[int, int]]] = [[] for _ in weights]
        for source, target, weight in rules:
            self.rules[source].append((target, math.floor(weight * self.scale + 0.5)))

    def _conditional_rules(self, patch: PatchGraph) -> list[tuple[int, int, float]]:
        """Compile source → target weights from shared and marginal fault probabilities.

        Separators in a DEM instruction describe correlated graph components of
        one physical fault. Distinct instructions are independent mechanisms.
        """
        marginal = np.zeros(len(self.edge_ids))
        shared: dict[tuple[int, int], float] = {}
        for instruction in patch.local_dem.flattened():
            if instruction.type != 'error':
                continue
            probability, = instruction.args_copy()
            if probability == 0:
                continue
            edges = []
            for group in instruction.target_groups():
                detectors = [target.val for target in group if target.is_relative_detector_id()]
                if detectors:
                    other = None if len(detectors) == 1 else detectors[1]
                    edges.append(self.edge_ids[_edge_key(detectors[0], other)])
            if len(set(edges)) != len(edges):
                raise ValueError('A DEM error repeats an edge across its components')
            for edge in edges:
                marginal[edge] = _xor_probability(marginal[edge], probability)
            for pair in itertools.combinations(sorted(edges), 2):
                shared[pair] = _xor_probability(shared.get(pair, 0.0), probability)

        rules = []
        for (a, b), probability in shared.items():
            for source, target in ((a, b), (b, a)):
                implied_probability = min(0.5, probability / marginal[source])
                rules.append((source, target, math.log((1 - implied_probability) / implied_probability)))
        return rules

    def for_edges(self, selected: np.ndarray) -> np.ndarray:
        """Freeze discounts from the first-pass edges; multiple discounts take the minimum."""
        weights = self.weights.copy()
        for u, v in selected:
            source = self.edge_ids[_edge_key(u, v)]
            for target, implied in self.rules[source]:
                weights[target] = min(weights[target], implied)
        return weights


@dataclass(frozen=True)
class ComplementaryGapResult:
    """One patch: native reference (X, Z), gaps in nats, and W[c_X, c_Z]."""

    reference: np.ndarray
    complementary_gap: np.ndarray
    forced_weights: np.ndarray
    native_weight: float

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference)
        gap, weights = np.asarray(self.complementary_gap), np.asarray(self.forced_weights)
        if reference.shape != (2,) or not np.isin(reference, (0, 1)).all():
            raise ValueError('Expected two binary reference bits')
        if gap.shape != (2,) or not np.isfinite(gap).all() or (gap < 0).any():
            raise ValueError('Expected two finite nonnegative complementary gaps')
        if weights.shape != (2, 2) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Expected a finite nonnegative 2 by 2 class-weight table')
        if not math.isfinite(self.native_weight) or self.native_weight < 0:
            raise ValueError('Native solution weight must be finite and nonnegative')
        for name, dtype in (('reference', bool), ('complementary_gap', np.float64),
                            ('forced_weights', np.float64)):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=dtype))


class CorrelatedMatchingGapDecoder:
    """L1: native two-pass correlated MWPM and its complementary logical costs."""

    def __init__(self, patch: PatchGraph):
        if pymatching.__version__.split('.')[:2] != ['2', '4']:
            raise RuntimeError('Frozen complementary gaps require the validated PyMatching 2.4.x series')
        self.patch = patch
        self.matching = pymatching.Matching.from_detector_error_model(
            patch.local_dem, enable_correlations=True)
        self._weights = _FrozenNativeWeights(patch, self.matching)
        self._check = _CheckMatrixGraph(patch.check_graph)

    def decode(self, syndrome) -> ComplementaryGapResult:
        """Return the native prediction and the extra cost of reversing either bit."""
        syndrome = np.asarray(syndrome)
        if syndrome.shape != (self.patch.num_detectors,) or not np.isin(syndrome, (0, 1)).all():
            raise ValueError(f'Expected {self.patch.num_detectors} binary local detector bits')
        syndrome = syndrome.astype(np.uint8)
        # Use the native object's own first-pass topology and discretization,
        # including correlation metadata, so first-pass ties condition identically.
        selected = self.matching.decode_to_edges_array(syndrome, enable_correlations=False)
        reference, native_weight = self.matching.decode(
            syndrome, enable_correlations=True, return_weight=True)
        weights = self._weights.for_edges(selected)
        matcher = self._check.matcher(weights)
        forced_syndromes = np.concatenate(
            [np.broadcast_to(syndrome, (len(CHECK_PATTERNS), len(syndrome))), CHECK_PATTERNS], axis=1)
        _, costs = matcher.decode_batch(forced_syndromes, return_weights=True, enable_correlations=False)
        costs = np.asarray(costs).reshape(2, 2)
        base = costs[tuple(reference)]
        # Forced costs are integers. Recover the native integer cost after its
        # conversion to nats; below 2**50 the float round trip cannot lose a unit.
        # This checks compatibility with the native solver on every shot, without
        # accepting a different reference or taking abs() of an inconsistent gap.
        if (not np.isfinite(costs).all() or not math.isfinite(native_weight)
                or np.max(costs) >= 2**50 or not np.equal(costs, np.floor(costs)).all()
                or base != costs.min() or base != round(native_weight * self._weights.scale)):
            raise ValueError('Frozen complementary weights disagree with native correlated PyMatching')
        if costs[0, 0] + costs[1, 1] != costs[0, 1] + costs[1, 0]:
            raise ValueError('Patch logical sectors do not have additive frozen matching costs')
        gaps = signed_gaps(costs, reference) / self._weights.scale
        return ComplementaryGapResult(reference, gaps, costs / self._weights.scale, float(native_weight))
