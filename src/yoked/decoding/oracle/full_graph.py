"""Read-only full-graph oracle certificates for ProMatch experiments.

The oracle in this module is deliberately independent of the ProMatch state
machine.  It evaluates one already-proposed canonical edge support against the
complete, ordinary PyMatching graph.  In particular, its API has no place for
sampled observables or decoder-success labels.

All correction supports are reconstructed through ``decode_to_edges_array``
and the canonical :class:`~yoked.decoding._promatch_graph.Edge` table.  This is
stricter than trusting the weight returned by ``decode``: endpoint pairs must
be unique, the reconstructed fault mask must agree with ``decode``, and the
backend and canonical-support weights must reconcile within the frozen
tolerance.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import numbers
from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np

from yoked.decoding._promatch_graph import CompiledPromatchGraph, Edge


OraclePolicy: TypeAlias = Literal["cost", "frame"]
EndpointPair: TypeAlias = tuple[int, int | None]


class CostClassification(str, enum.Enum):
    """Numerical classification of a forced-composite cost excess."""

    COMPATIBLE = "numerically-cost-compatible"
    POSITIVE_EXCESS = "positive-cost-excess"
    ACCOUNTING_ANOMALY = "numeric-accounting-anomaly"


class OracleGraphError(ValueError):
    """The canonical graph cannot support the oracle contract."""


class OracleDecodeError(RuntimeError):
    """PyMatching output cannot be reconciled with the canonical graph."""


class OracleAccountingError(RuntimeError):
    """A forced-composite score violates minimum-weight accounting."""

    def __init__(self, *, cost_excess: float, tau_k: float) -> None:
        self.classification = CostClassification.ACCOUNTING_ANOMALY
        self.cost_excess = cost_excess
        self.tau_k = tau_k
        super().__init__(
            "forced-composite cost is below the complete optimum beyond "
            f"tolerance: cost_excess={cost_excess!r}, tau_k={tau_k!r}"
        )


@dataclasses.dataclass(frozen=True)
class OracleTolerance:
    """Frozen absolute and relative numerical tolerances."""

    absolute: float = 1e-9
    relative: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} tolerance must be finite and nonnegative")

    def tau_weight(self, *, support_weight: float, backend_weight: float) -> float:
        """Returns the backend-versus-support reconciliation tolerance."""

        return self.absolute + self.relative * max(
            1.0, abs(support_weight), abs(backend_weight)
        )

    def tau_k(self, *, base_weight: float, composite_weight: float) -> float:
        """Returns the forced-composite classification tolerance."""

        return self.absolute + self.relative * max(
            1.0, abs(base_weight), abs(composite_weight)
        )


@dataclasses.dataclass(frozen=True)
class OracleMatchingSolution:
    """One deterministic PyMatching solution reconstructed canonically."""

    prediction: bytes
    support_edge_ids: tuple[int, ...]
    support_weight: float
    backend_weight: float
    tau_weight: float


@dataclasses.dataclass(frozen=True)
class OracleCacheStats:
    """Deterministic counters for the per-shot complete-solve cache."""

    hits: int
    misses: int
    entries: int


@dataclasses.dataclass(frozen=True)
class OracleEvaluation:
    """Ledger-complete result for one candidate and one oracle policy."""

    policy: OraclePolicy
    accepted: bool
    cost_compatible: bool
    frame_compatible: bool
    cost_classification: CostClassification
    cost_excess: float
    tau_k: float
    composite_weight: float

    accumulated_frame: bytes
    base_prediction: bytes
    residual_prediction: bytes
    base_frame: bytes
    candidate_composite_frame: bytes

    base_support_edge_ids: tuple[int, ...]
    residual_support_edge_ids: tuple[int, ...]
    base_support_weight: float
    residual_support_weight: float
    base_backend_weight: float
    residual_backend_weight: float
    base_tau_weight: float
    residual_tau_weight: float

    candidate_edge_ids: tuple[int, ...]
    candidate_boundary_detector_ids: tuple[int, ...]
    candidate_observable_frame: bytes
    candidate_weight: float


def classify_cost_excess(*, cost_excess: float, tau_k: float) -> CostClassification:
    """Classifies an excess using the frozen symmetric numerical band."""

    if not math.isfinite(cost_excess):
        raise OracleAccountingError(cost_excess=cost_excess, tau_k=tau_k)
    if not math.isfinite(tau_k) or tau_k < 0:
        raise ValueError("tau_k must be finite and nonnegative")
    if cost_excess > tau_k:
        return CostClassification.POSITIVE_EXCESS
    if cost_excess < -tau_k:
        return CostClassification.ACCOUNTING_ANOMALY
    return CostClassification.COMPATIBLE


def _xor_masks(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError(
            f"observable masks have different lengths {len(left)} and {len(right)}"
        )
    return bytes(a ^ b for a, b in zip(left, right))


def _mask_from_fault_ids(
    fault_ids: object,
    *,
    num_observables: int,
) -> bytes:
    try:
        values = tuple(fault_ids)  # type: ignore[arg-type]
    except TypeError as ex:
        raise OracleGraphError("matching fault_ids must be iterable") from ex
    result = bytearray((num_observables + 7) // 8)
    seen: set[int] = set()
    for raw_fault_id in values:
        if not isinstance(raw_fault_id, numbers.Integral):
            raise OracleGraphError(
                f"non-integral matching observable ID {raw_fault_id!r}"
            )
        fault_id = int(raw_fault_id)
        if fault_id < 0 or fault_id >= num_observables:
            raise OracleGraphError(
                f"matching observable ID {fault_id} outside [0, {num_observables})"
            )
        if fault_id in seen:
            raise OracleGraphError(f"duplicate matching observable ID {fault_id}")
        seen.add(fault_id)
        result[fault_id // 8] ^= 1 << (fault_id % 8)
    return bytes(result)


def _normalize_endpoint_pair(
    raw_source: object,
    raw_target: object,
    *,
    num_detectors: int,
    boundary_is_minus_one: bool,
) -> EndpointPair:
    if not isinstance(raw_source, numbers.Integral) or (
        raw_target is not None and not isinstance(raw_target, numbers.Integral)
    ):
        raise OracleGraphError(
            f"matching endpoint pair {(raw_source, raw_target)!r} is not integral"
        )
    source = int(raw_source)
    if boundary_is_minus_one and source == -1:
        source_value: int | None = None
    else:
        source_value = source
    if raw_target is None:
        target_value: int | None = None
    else:
        target = int(raw_target)
        target_value = None if boundary_is_minus_one and target == -1 else target

    nonboundary = [v for v in (source_value, target_value) if v is not None]
    if len(nonboundary) == 0:
        raise OracleGraphError("matching edge cannot have two boundary endpoints")
    for detector_id in nonboundary:
        if detector_id < 0 or detector_id >= num_detectors:
            raise OracleGraphError(
                f"matching detector ID {detector_id} outside [0, {num_detectors})"
            )
    if len(nonboundary) == 1:
        return nonboundary[0], None
    if nonboundary[0] == nonboundary[1]:
        raise OracleGraphError(f"matching self-loop at detector {nonboundary[0]}")
    return min(nonboundary), max(nonboundary)


def _bits_to_mask(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    return bytes(np.packbits(bits, bitorder="little"))


def _binary_vector(
    values: Sequence[int] | np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.shape[0] != expected_size:
        raise ValueError(
            f"{name} must have shape ({expected_size},), got {result.shape}"
        )
    if not np.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must contain only binary values")
    return np.asarray(result, dtype=np.uint8)


class FullGraphOracle:
    """Evaluates local supports against a validated complete matching graph."""

    def __init__(
        self,
        graph: CompiledPromatchGraph,
        *,
        tolerance: OracleTolerance = OracleTolerance(),
    ) -> None:
        if not isinstance(graph, CompiledPromatchGraph):
            raise TypeError(
                f"graph must be a CompiledPromatchGraph; got {type(graph)!r}"
            )
        if not isinstance(tolerance, OracleTolerance):
            raise TypeError("tolerance must be an OracleTolerance")
        self.graph = graph
        self.tolerance = tolerance
        self._mask_bytes = (graph.num_observables + 7) // 8
        self._edge_by_id, self._edge_id_by_endpoints = self._validate_graph(graph)
        self._solution_cache: dict[bytes, OracleMatchingSolution] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self) -> OracleCacheStats:
        return OracleCacheStats(
            hits=self._cache_hits,
            misses=self._cache_misses,
            entries=len(self._solution_cache),
        )

    def clear_cache(self) -> None:
        """Drops all per-shot solutions and resets auditable counters."""

        self._solution_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _validate_graph(
        graph: CompiledPromatchGraph,
    ) -> tuple[tuple[Edge, ...], dict[EndpointPair, int]]:
        if graph.num_detectors < 0 or graph.num_observables < 0:
            raise OracleGraphError("graph dimensions must be nonnegative")
        matcher = graph.matcher
        if matcher.num_detectors != graph.num_detectors:
            raise OracleGraphError(
                "matcher/canonical detector-count mismatch: "
                f"{matcher.num_detectors} != {graph.num_detectors}"
            )
        if matcher.num_fault_ids != graph.num_observables:
            raise OracleGraphError(
                "matcher/canonical observable-count mismatch: "
                f"{matcher.num_fault_ids} != {graph.num_observables}"
            )

        edge_by_id: list[Edge | None] = [None] * len(graph.edges)
        canonical_by_pair: dict[EndpointPair, Edge] = {}
        mask_bytes = (graph.num_observables + 7) // 8
        for edge in graph.edges:
            if (
                not isinstance(edge.edge_id, numbers.Integral)
                or edge.edge_id < 0
                or edge.edge_id >= len(graph.edges)
            ):
                raise OracleGraphError(f"invalid canonical edge ID {edge.edge_id!r}")
            if edge_by_id[int(edge.edge_id)] is not None:
                raise OracleGraphError(f"duplicate canonical edge ID {edge.edge_id}")
            if not math.isfinite(edge.weight) or edge.weight <= 0:
                raise OracleGraphError(
                    f"canonical edge {edge.edge_id} must have a strictly positive "
                    f"finite weight; got {edge.weight!r}"
                )
            if len(edge.observable_mask) != mask_bytes:
                raise OracleGraphError(
                    f"canonical edge {edge.edge_id} has observable-mask length "
                    f"{len(edge.observable_mask)}, expected {mask_bytes}"
                )
            unused_bits = mask_bytes * 8 - graph.num_observables
            if (
                unused_bits
                and edge.observable_mask
                and (edge.observable_mask[-1] >> (8 - unused_bits))
            ):
                raise OracleGraphError(
                    f"canonical edge {edge.edge_id} sets unused observable bits"
                )
            pair = _normalize_endpoint_pair(
                edge.source,
                edge.target,
                num_detectors=graph.num_detectors,
                boundary_is_minus_one=False,
            )
            if pair in canonical_by_pair:
                raise OracleGraphError(
                    f"ambiguous canonical endpoint pair {pair!r} on edges "
                    f"{canonical_by_pair[pair].edge_id} and {edge.edge_id}"
                )
            canonical_by_pair[pair] = edge
            edge_by_id[int(edge.edge_id)] = edge
        if any(edge is None for edge in edge_by_id):
            raise OracleGraphError("canonical edge IDs must be dense from zero")

        matcher_by_pair: dict[EndpointPair, tuple[float, bytes]] = {}
        for raw_source, raw_target, data in matcher.edges():
            pair = _normalize_endpoint_pair(
                raw_source,
                raw_target,
                num_detectors=graph.num_detectors,
                boundary_is_minus_one=False,
            )
            if pair in matcher_by_pair:
                raise OracleGraphError(f"ambiguous matcher endpoint pair {pair!r}")
            try:
                weight = float(data["weight"])
            except (KeyError, TypeError, ValueError) as ex:
                raise OracleGraphError(
                    f"matcher edge {pair!r} has an invalid weight"
                ) from ex
            mask = _mask_from_fault_ids(
                data.get("fault_ids", ()),
                num_observables=graph.num_observables,
            )
            matcher_by_pair[pair] = weight, mask

        if set(matcher_by_pair) != set(canonical_by_pair):
            missing = sorted(set(canonical_by_pair) - set(matcher_by_pair))
            extra = sorted(set(matcher_by_pair) - set(canonical_by_pair))
            raise OracleGraphError(
                "matcher/canonical endpoint tables differ: "
                f"missing={missing[:8]!r}, extra={extra[:8]!r}"
            )
        for pair, edge in canonical_by_pair.items():
            matcher_weight, matcher_mask = matcher_by_pair[pair]
            if matcher_weight != edge.weight or matcher_mask != edge.observable_mask:
                raise OracleGraphError(
                    f"matcher/canonical data differ for endpoint pair {pair!r}"
                )
        return (
            tuple(edge for edge in edge_by_id if edge is not None),
            {pair: edge.edge_id for pair, edge in canonical_by_pair.items()},
        )

    def decode_state(
        self,
        syndrome: Sequence[int] | np.ndarray,
        *,
        use_cache: bool = True,
    ) -> OracleMatchingSolution:
        """Runs and canonically reconstructs deterministic ordinary MWPM."""

        syndrome_array = _binary_vector(
            syndrome,
            expected_size=self.graph.num_detectors,
            name="syndrome",
        )
        cache_key = syndrome_array.tobytes()
        if use_cache:
            cached = self._solution_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached
        self._cache_misses += 1
        prediction_raw, backend_weight_raw = self.graph.matcher.decode(
            syndrome_array, return_weight=True
        )
        prediction_bits = _binary_vector(
            np.asarray(prediction_raw),
            expected_size=self.graph.num_observables,
            name="matching prediction",
        )
        backend_weight = float(backend_weight_raw)
        if not math.isfinite(backend_weight):
            raise OracleDecodeError(
                f"PyMatching returned nonfinite weight {backend_weight!r}"
            )

        returned_pairs = np.asarray(
            self.graph.matcher.decode_to_edges_array(syndrome_array)
        )
        if returned_pairs.size == 0:
            returned_pairs = np.empty((0, 2), dtype=np.int64)
        if returned_pairs.ndim != 2 or returned_pairs.shape[1] != 2:
            raise OracleDecodeError(
                "decode_to_edges_array must return an array with shape (n, 2); "
                f"got {returned_pairs.shape}"
            )

        edge_ids: list[int] = []
        seen: set[int] = set()
        support_boundary: set[int] = set()
        support_mask = bytes(self._mask_bytes)
        for raw_source, raw_target in returned_pairs:
            try:
                pair = _normalize_endpoint_pair(
                    raw_source,
                    raw_target,
                    num_detectors=self.graph.num_detectors,
                    boundary_is_minus_one=True,
                )
            except OracleGraphError as ex:
                raise OracleDecodeError(
                    f"invalid returned endpoint pair {(raw_source, raw_target)!r}"
                ) from ex
            try:
                edge_id = self._edge_id_by_endpoints[pair]
            except KeyError as ex:
                raise OracleDecodeError(
                    f"returned endpoint pair {pair!r} has no canonical edge"
                ) from ex
            if edge_id in seen:
                raise OracleDecodeError(
                    f"decode_to_edges_array returned canonical edge {edge_id} twice"
                )
            seen.add(edge_id)
            edge_ids.append(edge_id)
            edge = self._edge_by_id[edge_id]
            support_boundary.symmetric_difference_update((edge.source,))
            if edge.target is not None:
                support_boundary.symmetric_difference_update((edge.target,))
            support_mask = _xor_masks(support_mask, edge.observable_mask)
        edge_ids.sort()
        expected_boundary = tuple(int(k) for k in np.flatnonzero(syndrome_array))
        actual_boundary = tuple(sorted(support_boundary))
        if actual_boundary != expected_boundary:
            raise OracleDecodeError(
                "boundary of reconstructed support does not match syndrome: "
                f"support={actual_boundary!r}, syndrome={expected_boundary!r}"
            )
        prediction = _bits_to_mask(prediction_bits)
        if support_mask != prediction:
            raise OracleDecodeError(
                "XOR of reconstructed support fault masks does not match decode: "
                f"support={support_mask.hex()}, prediction={prediction.hex()}"
            )
        support_weight = math.fsum(
            self._edge_by_id[edge_id].weight for edge_id in edge_ids
        )
        tau_weight = self.tolerance.tau_weight(
            support_weight=support_weight, backend_weight=backend_weight
        )
        if abs(support_weight - backend_weight) > tau_weight:
            raise OracleDecodeError(
                "backend and canonical-support weights disagree beyond tolerance: "
                f"support={support_weight!r}, backend={backend_weight!r}, "
                f"tau_weight={tau_weight!r}"
            )
        solution = OracleMatchingSolution(
            prediction=prediction,
            support_edge_ids=tuple(edge_ids),
            support_weight=support_weight,
            backend_weight=backend_weight,
            tau_weight=tau_weight,
        )
        if use_cache:
            self._solution_cache[cache_key] = solution
        return solution

    def evaluate(
        self,
        *,
        syndrome: Sequence[int] | np.ndarray,
        accumulated_frame: Sequence[int] | np.ndarray,
        candidate_edge_ids: Sequence[int],
        policy: OraclePolicy,
    ) -> OracleEvaluation:
        """Evaluates one square-free candidate without mutating caller state."""

        if policy not in ("cost", "frame"):
            raise ValueError(f"unsupported oracle policy {policy!r}")
        syndrome_array = _binary_vector(
            syndrome,
            expected_size=self.graph.num_detectors,
            name="syndrome",
        )
        frame_bits = _binary_vector(
            accumulated_frame,
            expected_size=self.graph.num_observables,
            name="accumulated_frame",
        )
        accumulated_mask = _bits_to_mask(frame_bits)

        normalized_ids: list[int] = []
        seen_ids: set[int] = set()
        for raw_edge_id in candidate_edge_ids:
            if not isinstance(raw_edge_id, numbers.Integral):
                raise ValueError(f"candidate edge ID {raw_edge_id!r} is not integral")
            edge_id = int(raw_edge_id)
            if edge_id < 0 or edge_id >= len(self._edge_by_id):
                raise ValueError(f"candidate edge ID {edge_id} is out of range")
            if edge_id in seen_ids:
                raise ValueError(
                    f"candidate support is not square-free: repeated edge {edge_id}"
                )
            seen_ids.add(edge_id)
            normalized_ids.append(edge_id)
        normalized_ids.sort()

        boundary: set[int] = set()
        candidate_observable_frame = bytes(self._mask_bytes)
        candidate_weights: list[float] = []
        for edge_id in normalized_ids:
            edge = self._edge_by_id[edge_id]
            boundary.symmetric_difference_update((edge.source,))
            if edge.target is not None:
                boundary.symmetric_difference_update((edge.target,))
            candidate_observable_frame = _xor_masks(
                candidate_observable_frame, edge.observable_mask
            )
            candidate_weights.append(edge.weight)

        residual_syndrome = syndrome_array.copy()
        boundary_ids = tuple(sorted(boundary))
        if boundary_ids:
            residual_syndrome[np.asarray(boundary_ids, dtype=np.int64)] ^= 1

        base = self.decode_state(syndrome_array)
        residual = self.decode_state(residual_syndrome)
        candidate_weight = math.fsum(candidate_weights)
        residual_weights = [
            self._edge_by_id[edge_id].weight for edge_id in residual.support_edge_ids
        ]
        composite_weight = math.fsum((*candidate_weights, *residual_weights))
        cost_excess = composite_weight - base.support_weight
        tau_k = self.tolerance.tau_k(
            base_weight=base.support_weight,
            composite_weight=composite_weight,
        )
        classification = classify_cost_excess(cost_excess=cost_excess, tau_k=tau_k)
        if classification is CostClassification.ACCOUNTING_ANOMALY:
            raise OracleAccountingError(cost_excess=cost_excess, tau_k=tau_k)

        base_frame = _xor_masks(accumulated_mask, base.prediction)
        candidate_accumulated = _xor_masks(accumulated_mask, candidate_observable_frame)
        candidate_composite_frame = _xor_masks(
            candidate_accumulated, residual.prediction
        )
        cost_compatible = classification is CostClassification.COMPATIBLE
        frame_compatible = candidate_composite_frame == base_frame
        accepted = cost_compatible and (policy == "cost" or frame_compatible)

        return OracleEvaluation(
            policy=policy,
            accepted=accepted,
            cost_compatible=cost_compatible,
            frame_compatible=frame_compatible,
            cost_classification=classification,
            cost_excess=cost_excess,
            tau_k=tau_k,
            composite_weight=composite_weight,
            accumulated_frame=accumulated_mask,
            base_prediction=base.prediction,
            residual_prediction=residual.prediction,
            base_frame=base_frame,
            candidate_composite_frame=candidate_composite_frame,
            base_support_edge_ids=base.support_edge_ids,
            residual_support_edge_ids=residual.support_edge_ids,
            base_support_weight=base.support_weight,
            residual_support_weight=residual.support_weight,
            base_backend_weight=base.backend_weight,
            residual_backend_weight=residual.backend_weight,
            base_tau_weight=base.tau_weight,
            residual_tau_weight=residual.tau_weight,
            candidate_edge_ids=tuple(normalized_ids),
            candidate_boundary_detector_ids=boundary_ids,
            candidate_observable_frame=candidate_observable_frame,
            candidate_weight=candidate_weight,
        )


__all__ = [
    "CostClassification",
    "FullGraphOracle",
    "OracleAccountingError",
    "OracleCacheStats",
    "OracleDecodeError",
    "OracleEvaluation",
    "OracleGraphError",
    "OracleMatchingSolution",
    "OraclePolicy",
    "OracleTolerance",
    "classify_cost_excess",
]
