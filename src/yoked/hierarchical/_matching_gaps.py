"""Minimum-weight matching in each logical class: forced weights and gaps.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 5.2.

For a patch, W(c_X, c_Z) is the minimum matching weight on the check graph
with the X check bit c_X and the Z check bit c_Z. Because the two sectors
are disconnected, W(c_X, c_Z) = W_X(c_X) + W_Z(c_Z). The signed gap of
sector X relative to reference bits (r_X, r_Z) is

    delta_X = W(1 - r_X, r_Z) - W(r_X, r_Z),

and symmetrically for Z; a negative gap means the matcher prefers the
complement of the reference.

Two variants share one code path. Plain: the original weights. Correlated:
an unforced first pass on the check-free graph selects edges; the DEM
correlation rules lower the weights of their partners; the four forced
decodes then run on the check graph with those weights, so both classes are
compared under one reweighted model. The unforced second pass on the
reweighted check-free graph is recorded for validation only.

Matchers are built from DecodingGraph edge lists through a check matrix so
that adjusted weights can be supplied per shot. ``_matching_gaps_test.py``
checks equivalence with PyMatching's DEM import, sector additivity,
consistency of the forced argmin with the unforced predictions, and a
hand-built mechanism whose correlation changes a gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pymatching
import scipy.sparse

from yoked.decoders._correlations import CorrelationRule, apply_correlation_rules, index_rules_by_source
from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph
from yoked.hierarchical._arrays import readonly_array

CHECK_PATTERNS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
"""Rows are (c_X, c_Z) in the order that reshapes to an array indexed [c_X, c_Z]."""


@dataclass(frozen=True)
class ForcedWeights:
    """Forced class weights of one patch for one shot.

    Fields: ``plain`` and ``correlated`` are (2, 2) float64 arrays in nats
    indexed [c_X, c_Z]; ``first_pass`` (2,) bool is the unforced plain
    prediction; ``correlated_prediction`` (2,) bool is the unforced second
    pass under the reweighted model; ``rules_fired`` records whether any
    correlation rule lowered a weight (when false the correlated arrays
    equal the plain ones).
    """
    plain: np.ndarray
    correlated: np.ndarray
    first_pass: np.ndarray
    correlated_prediction: np.ndarray
    rules_fired: bool

    def __post_init__(self) -> None:
        for name, dtype in (('plain', np.float64), ('correlated', np.float64),
                            ('first_pass', bool), ('correlated_prediction', bool)):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=dtype))


class _CheckMatrixGraph:
    """A DecodingGraph as PyMatching check and fault matrices, for per-shot matchers."""

    def __init__(self, graph: DecodingGraph):
        rows, cols = [], []
        for e, (u, v, _, _) in enumerate(graph.edges):
            rows.append(u)
            cols.append(e)
            if v is not None:
                rows.append(v)
                cols.append(e)
        # A column with a single entry is a boundary edge in PyMatching.
        self.check_matrix = scipy.sparse.csc_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(graph.num_detectors, len(graph.edges)))
        fault_rows, fault_cols = [], []
        for e, (_, _, _, mask) in enumerate(graph.edges):
            for k in range(graph.num_observables):
                if (mask >> k) & 1:
                    fault_rows.append(k)
                    fault_cols.append(e)
        self.faults_matrix = scipy.sparse.csc_matrix(
            (np.ones(len(fault_rows)), (fault_rows, fault_cols)), shape=(graph.num_observables, len(graph.edges)))
        self.weights = np.array([weight for _, _, weight, _ in graph.edges], dtype=np.float64)
        self.masks = [mask for _, _, _, mask in graph.edges]
        self.edge_ids = {
            (u, -1) if v is None else (min(u, v), max(u, v)): e for e, (u, v, _, _) in enumerate(graph.edges)
        }

    def matcher(self, weights: Sequence[float] | None = None) -> pymatching.Matching:
        weights = self.weights if weights is None else np.asarray(weights, dtype=np.float64)
        return pymatching.Matching.from_check_matrix(
            self.check_matrix, weights=weights, faults_matrix=self.faults_matrix, merge_strategy='disallow')

    def edge_ids_of(self, edges_array: np.ndarray) -> list[int]:
        """Map PyMatching's [u, v] rows (v = -1 for a boundary) to edge ids."""
        return [
            self.edge_ids[(int(u), -1) if v < 0 else (min(int(u), int(v)), max(int(u), int(v)))]
            for u, v in edges_array
        ]


class MatchingGaps:
    """Forced class weights of one patch, plain and correlated."""

    def __init__(self, patch: PatchGraph, correlation_rules: Sequence[CorrelationRule]):
        self.patch = patch
        self._rules = index_rules_by_source(patch.graph, correlation_rules)
        self._free = _CheckMatrixGraph(patch.graph)
        self._check = _CheckMatrixGraph(patch.check_graph)
        self._plain_free = self._free.matcher()
        self._plain_check = self._check.matcher()

    def forced_weights(self, local_syndrome: np.ndarray) -> ForcedWeights:
        syndrome = np.asarray(local_syndrome)
        if syndrome.shape != (self.patch.num_detectors,):
            raise ValueError(f'Expected {self.patch.num_detectors} detector bits, got shape {syndrome.shape}')
        if syndrome.dtype.kind not in 'buif' or not np.isin(syndrome, (0, 1)).all():
            raise ValueError('Expected binary detector bits')
        syndrome = syndrome.astype(np.uint8, copy=False)
        selected = self._free.edge_ids_of(self._plain_free.decode_to_edges_array(syndrome))
        first_pass = self._prediction_of(selected)
        plain = self._forced(self._plain_check, syndrome)
        adjusted = apply_correlation_rules(self._free.weights, self._rules, selected)
        if adjusted is None:
            return ForcedWeights(plain, plain.copy(), first_pass, first_pass.copy(), False)
        correlated_prediction = self._free.matcher(adjusted).decode(syndrome).astype(bool)
        correlated = self._forced(self._check.matcher(adjusted), syndrome)
        return ForcedWeights(plain, correlated, first_pass, correlated_prediction, True)

    def _prediction_of(self, selected: Sequence[int]) -> np.ndarray:
        mask = 0
        for e in selected:
            mask ^= self._free.masks[e]
        return np.array([(mask >> k) & 1 for k in range(NUM_SECTORS)], dtype=bool)

    def _forced(self, matcher: pymatching.Matching, syndrome: np.ndarray) -> np.ndarray:
        """Decode the four check patterns at once; returns W indexed [c_X, c_Z]."""
        rows = np.concatenate(
            [np.broadcast_to(syndrome, (len(CHECK_PATTERNS), len(syndrome))), CHECK_PATTERNS], axis=1)
        _, weights = matcher.decode_batch(rows, return_weights=True)
        return np.asarray(weights, dtype=np.float64).reshape(NUM_SECTORS, NUM_SECTORS)


def signed_gaps(forced: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Gaps relative to reference bits, sector by sector, for any leading shape.

    ``forced`` has shape (..., 2, 2) indexed [c_X, c_Z]; ``reference`` has
    shape (..., 2). The result (..., 2) holds delta_X and delta_Z.
    """
    forced = np.asarray(forced, dtype=np.float64)
    reference = np.asarray(reference).astype(np.intp)
    r_x, r_z = reference[..., 0], reference[..., 1]

    def pick(c_x: np.ndarray, c_z: np.ndarray) -> np.ndarray:
        by_x = np.take_along_axis(forced, c_x[..., None, None], axis=-2)[..., 0, :]
        return np.take_along_axis(by_x, c_z[..., None], axis=-1)[..., 0]

    base = pick(r_x, r_z)
    return np.stack([pick(1 - r_x, r_z) - base, pick(r_x, 1 - r_z) - base], axis=-1)
