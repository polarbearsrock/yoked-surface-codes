from __future__ import annotations

import dataclasses
import inspect
import math
from fractions import Fraction
from typing import cast

import numpy as np
import pymatching
import pytest

from yoked.decoding._promatch_graph import CompiledPromatchGraph, Edge
from yoked.decoding._promatch_layout import PromatchLayout, YokeDetector
from yoked.decoding._promatch_oracle import (
    CostClassification,
    FullGraphOracle,
    OracleAccountingError,
    OracleDecodeError,
    OracleGraphError,
    OracleTolerance,
    classify_cost_excess,
)


def _fault_mask(fault_ids: set[int], *, num_observables: int) -> bytes:
    result = bytearray((num_observables + 7) // 8)
    for fault_id in fault_ids:
        result[fault_id // 8] |= 1 << (fault_id % 8)
    return bytes(result)


def _compiled_graph(
    matcher: pymatching.Matching,
    *,
    num_observables: int,
) -> CompiledPromatchGraph:
    matcher.ensure_num_fault_ids(num_observables)
    edges = []
    role = YokeDetector()
    for edge_id, (raw_source, raw_target, data) in enumerate(matcher.edges()):
        source = int(raw_source)
        target = None if raw_target is None else int(raw_target)
        if target is not None and target < source:
            source, target = target, source
        edges.append(
            Edge(
                edge_id=edge_id,
                source=source,
                target=target,
                weight=float(data["weight"]),
                observable_mask=_fault_mask(
                    set(data.get("fault_ids", ())),
                    num_observables=num_observables,
                ),
                source_role=role,
                target_role=None if target is None else role,
            )
        )
    return CompiledPromatchGraph(
        layout=cast(PromatchLayout, None),
        matcher=matcher,
        edges=tuple(edges),
        domain_graphs={},
        fingerprint="tiny-oracle-test",
        require_zero_frame=False,
        num_detectors=matcher.num_detectors,
        num_observables=num_observables,
    )


def _three_way_graph(
    *,
    direct_weight: float,
    boundary_weights: tuple[float, float],
    direct_faults: set[int] = frozenset(),
    left_faults: set[int] = frozenset(),
    right_faults: set[int] = frozenset(),
    num_observables: int = 0,
) -> CompiledPromatchGraph:
    matcher = pymatching.Matching()
    matcher.add_edge(0, 1, weight=direct_weight, fault_ids=set(direct_faults))
    matcher.add_boundary_edge(
        0, weight=boundary_weights[0], fault_ids=set(left_faults)
    )
    matcher.add_boundary_edge(
        1, weight=boundary_weights[1], fault_ids=set(right_faults)
    )
    return _compiled_graph(matcher, num_observables=num_observables)


def _edge_id(graph: CompiledPromatchGraph, pair: tuple[int, int | None]) -> int:
    normalized = (min(pair), max(pair)) if pair[1] is not None else pair
    for edge in graph.edges:
        if (edge.source, edge.target) == normalized:
            return edge.edge_id
    raise AssertionError(f"missing test edge {pair!r}")


def test_cost_and_frame_oracles_on_equal_weight_different_logical_classes() -> None:
    graph = _three_way_graph(
        direct_weight=2,
        boundary_weights=(1, 1),
        direct_faults={0},
        left_faults=set(),
        right_faults={1},
        num_observables=2,
    )
    oracle = FullGraphOracle(graph, tolerance=OracleTolerance(absolute=0, relative=0))
    left_boundary = _edge_id(graph, (0, None))

    cost = oracle.evaluate(
        syndrome=np.array([1, 1], dtype=np.uint8),
        accumulated_frame=np.array([0, 0], dtype=np.uint8),
        candidate_edge_ids=(left_boundary,),
        policy="cost",
    )
    frame = oracle.evaluate(
        syndrome=np.array([1, 1], dtype=np.uint8),
        accumulated_frame=np.array([0, 0], dtype=np.uint8),
        candidate_edge_ids=(left_boundary,),
        policy="frame",
    )

    assert cost.cost_classification is CostClassification.COMPATIBLE
    assert cost.cost_compatible
    assert not cost.frame_compatible
    assert cost.accepted
    assert not frame.accepted
    assert cost.base_frame != cost.candidate_composite_frame
    assert cost.candidate_boundary_detector_ids == (0,)
    assert cost.candidate_weight == 1
    assert cost.composite_weight == 2
    assert cost.cost_excess == 0


def test_cost_compatible_same_frame_is_accepted_by_frame_oracle() -> None:
    graph = _three_way_graph(
        direct_weight=2,
        boundary_weights=(1, 1),
        direct_faults={0},
        left_faults=set(),
        right_faults={0},
        num_observables=1,
    )
    result = FullGraphOracle(
        graph, tolerance=OracleTolerance(absolute=0, relative=0)
    ).evaluate(
        syndrome=[1, 1],
        accumulated_frame=[1],
        candidate_edge_ids=(_edge_id(graph, (0, None)),),
        policy="frame",
    )
    assert result.accepted
    assert result.cost_compatible
    assert result.frame_compatible
    assert result.base_frame == result.candidate_composite_frame
    assert result.base_support_weight == result.base_backend_weight == 2
    assert result.residual_support_weight == result.residual_backend_weight == 1
    assert result.base_tau_weight == result.residual_tau_weight == 0


def test_positive_cost_excess_is_vetoed() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(1, 1))
    result = FullGraphOracle(
        graph, tolerance=OracleTolerance(absolute=0, relative=0)
    ).evaluate(
        syndrome=[1, 1],
        accumulated_frame=[],
        candidate_edge_ids=(_edge_id(graph, (0, None)),),
        policy="cost",
    )
    assert not result.accepted
    assert not result.cost_compatible
    assert result.frame_compatible
    assert result.cost_classification is CostClassification.POSITIVE_EXCESS
    assert result.cost_excess == 1


def test_candidate_support_must_be_square_free_and_inputs_are_read_only() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    oracle = FullGraphOracle(graph)
    syndrome = np.array([1, 1], dtype=np.uint8)
    frame = np.empty(0, dtype=np.uint8)
    edge_id = _edge_id(graph, (0, 1))
    with pytest.raises(ValueError, match="square-free"):
        oracle.evaluate(
            syndrome=syndrome,
            accumulated_frame=frame,
            candidate_edge_ids=(edge_id, edge_id),
            policy="cost",
        )
    assert syndrome.tolist() == [1, 1]
    assert frame.tolist() == []


@pytest.mark.parametrize("weight", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_every_canonical_edge_must_have_positive_finite_weight(weight: float) -> None:
    valid = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    changed = dataclasses.replace(valid.edges[-1], weight=weight)
    graph = dataclasses.replace(valid, edges=(*valid.edges[:-1], changed))
    with pytest.raises(OracleGraphError, match="strictly positive finite"):
        FullGraphOracle(graph)


def test_endpoint_pairs_must_be_unique_even_if_candidate_does_not_use_them() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    duplicate = dataclasses.replace(
        graph.edges[0], edge_id=len(graph.edges), weight=graph.edges[0].weight + 1
    )
    graph = dataclasses.replace(graph, edges=(*graph.edges, duplicate))
    with pytest.raises(OracleGraphError, match="ambiguous canonical endpoint pair"):
        FullGraphOracle(graph)


def test_matcher_and_canonical_edge_data_must_agree() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    changed = dataclasses.replace(graph.edges[0], weight=1.25)
    graph = dataclasses.replace(graph, edges=(changed, *graph.edges[1:]))
    with pytest.raises(OracleGraphError, match="data differ"):
        FullGraphOracle(graph)


class _ScriptedMatcher:
    def __init__(
        self,
        base: pymatching.Matching,
        *,
        prediction: list[int],
        backend_weight: float,
        returned_pairs: list[tuple[int, int]],
        responses: dict[
            tuple[int, ...], tuple[list[int], float, list[tuple[int, int]]]
        ]
        | None = None,
    ) -> None:
        self._base = base
        self._prediction = np.asarray(prediction, dtype=np.uint8)
        self._backend_weight = backend_weight
        self._returned_pairs = np.asarray(returned_pairs, dtype=np.int64).reshape((-1, 2))
        self._responses = responses
        self.num_detectors = base.num_detectors
        self.num_fault_ids = base.num_fault_ids

    def edges(self):
        return self._base.edges()

    def decode(self, syndrome, *, return_weight=False):
        assert return_weight
        if self._responses is not None:
            prediction, weight, _ = self._responses[tuple(map(int, syndrome))]
            return np.asarray(prediction, dtype=np.uint8), weight
        return self._prediction.copy(), self._backend_weight

    def decode_to_edges_array(self, syndrome):
        if self._responses is not None:
            _, _, pairs = self._responses[tuple(map(int, syndrome))]
            return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
        return self._returned_pairs.copy()


def _scripted_graph(
    graph: CompiledPromatchGraph,
    *,
    prediction: list[int],
    backend_weight: float,
    returned_pairs: list[tuple[int, int]],
) -> CompiledPromatchGraph:
    matcher = _ScriptedMatcher(
        graph.matcher,
        prediction=prediction,
        backend_weight=backend_weight,
        returned_pairs=returned_pairs,
    )
    return dataclasses.replace(graph, matcher=cast(pymatching.Matching, matcher))


@pytest.mark.parametrize(
    ("returned_pairs", "message"),
    [
        ([(0, 1), (1, 0)], "returned canonical edge"),
        ([(0, 7)], "invalid returned endpoint pair"),
    ],
)
def test_returned_support_rejects_duplicate_or_invalid_endpoint_pairs(
    returned_pairs: list[tuple[int, int]], message: str
) -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    graph = _scripted_graph(
        graph,
        prediction=[],
        backend_weight=2,
        returned_pairs=returned_pairs,
    )
    with pytest.raises(OracleDecodeError, match=message):
        FullGraphOracle(graph).decode_state([1, 1])


def test_reconstructed_fault_mask_must_match_decode_prediction() -> None:
    graph = _three_way_graph(
        direct_weight=1,
        boundary_weights=(2, 2),
        direct_faults={0},
        num_observables=1,
    )
    graph = _scripted_graph(
        graph,
        prediction=[0],
        backend_weight=1,
        returned_pairs=[(0, 1)],
    )
    with pytest.raises(OracleDecodeError, match="fault masks does not match"):
        FullGraphOracle(graph).decode_state([1, 1])


def test_reconstructed_support_boundary_must_match_syndrome() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    graph = _scripted_graph(
        graph,
        prediction=[],
        backend_weight=1,
        returned_pairs=[(0, 1)],
    )
    with pytest.raises(OracleDecodeError, match="boundary.*does not match"):
        FullGraphOracle(graph).decode_state([1, 0])


def test_backend_and_support_weights_must_reconcile() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    graph = _scripted_graph(
        graph,
        prediction=[],
        backend_weight=1.1,
        returned_pairs=[(0, 1)],
    )
    with pytest.raises(OracleDecodeError, match="weights disagree"):
        FullGraphOracle(
            graph, tolerance=OracleTolerance(absolute=0, relative=1e-6)
        ).decode_state([1, 1])


def test_cached_and_uncached_complete_solutions_are_identical_and_audited() -> None:
    graph = _three_way_graph(direct_weight=1, boundary_weights=(2, 2))
    oracle = FullGraphOracle(graph)

    first = oracle.decode_state([1, 1])
    assert oracle.cache_stats.hits == 0
    assert oracle.cache_stats.misses == 1
    assert oracle.cache_stats.entries == 1
    cached = oracle.decode_state([1, 1])
    uncached = oracle.decode_state([1, 1], use_cache=False)

    assert cached is first
    assert uncached == first
    assert oracle.cache_stats.hits == 1
    assert oracle.cache_stats.misses == 2
    assert oracle.cache_stats.entries == 1
    oracle.clear_cache()
    assert oracle.cache_stats.hits == 0
    assert oracle.cache_stats.misses == 0
    assert oracle.cache_stats.entries == 0


def test_negative_cost_excess_is_a_fatal_accounting_anomaly() -> None:
    graph = _three_way_graph(direct_weight=3, boundary_weights=(1, 1))
    # Script the base solve to claim the non-optimal direct edge.  Both decode
    # APIs and their weights agree, isolating the negative-excess guard.
    matcher = _ScriptedMatcher(
        graph.matcher,
        prediction=[],
        backend_weight=3,
        returned_pairs=[(0, 1)],
        responses={
            (1, 1): ([], 3, [(0, 1)]),
            (0, 1): ([], 1, [(1, -1)]),
        },
    )
    graph = dataclasses.replace(graph, matcher=cast(pymatching.Matching, matcher))
    oracle = FullGraphOracle(
        graph, tolerance=OracleTolerance(absolute=0, relative=0)
    )
    with pytest.raises(OracleAccountingError) as exc_info:
        oracle.evaluate(
            syndrome=[1, 1],
            accumulated_frame=[],
            candidate_edge_ids=(_edge_id(graph, (0, None)),),
            policy="cost",
        )
    assert exc_info.value.classification is CostClassification.ACCOUNTING_ANOMALY
    assert exc_info.value.cost_excess == -1


def test_tau_weight_and_tau_k_have_their_distinct_scales() -> None:
    tolerance = OracleTolerance(absolute=0.25, relative=0.1)
    assert tolerance.tau_weight(support_weight=3, backend_weight=20) == 2.25
    assert tolerance.tau_k(base_weight=3, composite_weight=7) == pytest.approx(0.95)
    assert classify_cost_excess(cost_excess=0.95, tau_k=0.95) is CostClassification.COMPATIBLE
    assert classify_cost_excess(cost_excess=0.951, tau_k=0.95) is CostClassification.POSITIVE_EXCESS
    assert classify_cost_excess(cost_excess=-0.951, tau_k=0.95) is CostClassification.ACCOUNTING_ANOMALY


def test_oracle_api_cannot_accept_ground_truth() -> None:
    parameters = inspect.signature(FullGraphOracle.evaluate).parameters
    assert set(parameters) == {
        "self",
        "syndrome",
        "accumulated_frame",
        "candidate_edge_ids",
        "policy",
    }
    assert not any("actual" in name or "truth" in name for name in parameters)


def _support_boundary(graph: CompiledPromatchGraph, support_mask: int) -> tuple[int, ...]:
    boundary: set[int] = set()
    for edge in graph.edges:
        if support_mask & (1 << edge.edge_id):
            boundary.symmetric_difference_update((edge.source,))
            if edge.target is not None:
                boundary.symmetric_difference_update((edge.target,))
    return tuple(sorted(boundary))


def _syndrome(graph: CompiledPromatchGraph, boundary: tuple[int, ...]) -> list[int]:
    result = [0] * graph.num_detectors
    for detector_id in boundary:
        result[detector_id] = 1
    return result


def _exact_weight(graph: CompiledPromatchGraph, support_mask: int) -> Fraction:
    return sum(
        (
            Fraction.from_float(edge.weight)
            for edge in graph.edges
            if support_mask & (1 << edge.edge_id)
        ),
        start=Fraction(),
    )


def _exhaustive_minima(
    graph: CompiledPromatchGraph,
) -> dict[tuple[int, ...], tuple[Fraction, tuple[int, ...]]]:
    by_boundary: dict[tuple[int, ...], list[tuple[Fraction, int]]] = {}
    for support in range(1 << len(graph.edges)):
        by_boundary.setdefault(_support_boundary(graph, support), []).append(
            (_exact_weight(graph, support), support)
        )
    result = {}
    for boundary, candidates in by_boundary.items():
        minimum = min(weight for weight, _ in candidates)
        result[boundary] = (
            minimum,
            tuple(support for weight, support in candidates if weight == minimum),
        )
    return result


def _small_positive_graphs() -> tuple[CompiledPromatchGraph, ...]:
    boundary_tie = _three_way_graph(direct_weight=2, boundary_weights=(1, 1))

    hub = pymatching.Matching()
    hub.add_edge(0, 2, weight=1)
    hub.add_edge(1, 2, weight=1)
    hub.add_edge(0, 1, weight=2)

    disconnected = pymatching.Matching()
    # Integral weights retain a wide enough dynamic range without conflating
    # this exact-arithmetic theorem test with backend weight quantization.
    disconnected.add_edge(0, 1, weight=1)
    disconnected.add_boundary_edge(0, weight=1_000_000)
    disconnected.add_boundary_edge(1, weight=1_000_000)
    disconnected.add_edge(2, 3, weight=3)

    return (
        boundary_tie,
        _compiled_graph(hub, num_observables=0),
        _compiled_graph(disconnected, num_observables=0),
    )


@pytest.mark.parametrize("graph", _small_positive_graphs())
def test_exact_small_graph_support_containment_theorem(
    graph: CompiledPromatchGraph,
) -> None:
    """Exhausts all syndromes and square-free candidate supports."""

    minima = _exhaustive_minima(graph)
    oracle = FullGraphOracle(
        graph, tolerance=OracleTolerance(absolute=0, relative=0)
    )
    for syndrome_boundary, (minimum_weight, minimum_supports) in minima.items():
        syndrome = _syndrome(graph, syndrome_boundary)
        decoded = oracle.decode_state(syndrome)
        assert Fraction.from_float(decoded.support_weight) == minimum_weight
        for candidate in range(1 << len(graph.edges)):
            candidate_ids = tuple(
                edge.edge_id
                for edge in graph.edges
                if candidate & (1 << edge.edge_id)
            )
            result = oracle.evaluate(
                syndrome=syndrome,
                accumulated_frame=[],
                candidate_edge_ids=candidate_ids,
                policy="cost",
            )
            contained_in_an_optimum = any(
                candidate & optimum == candidate for optimum in minimum_supports
            )
            assert result.cost_excess >= 0
            assert result.cost_compatible == contained_in_an_optimum
            assert result.accepted == contained_in_an_optimum


def test_sequential_frame_acceptance_preserves_initial_prediction() -> None:
    graph = _three_way_graph(
        direct_weight=2,
        boundary_weights=(1, 1),
        direct_faults={0},
        left_faults={1},
        right_faults={0, 1},
        num_observables=2,
    )
    oracle = FullGraphOracle(
        graph, tolerance=OracleTolerance(absolute=0, relative=0)
    )
    syndrome = np.array([1, 1], dtype=np.uint8)
    frame = np.array([0, 0], dtype=np.uint8)
    initial = oracle.decode_state(syndrome).prediction

    # Try every edge in deterministic ID order.  Applying only accepted
    # candidates is an explicit small induction over durable steps.
    for edge in graph.edges:
        result = oracle.evaluate(
            syndrome=syndrome,
            accumulated_frame=frame,
            candidate_edge_ids=(edge.edge_id,),
            policy="frame",
        )
        if not result.accepted:
            continue
        for detector_id in result.candidate_boundary_detector_ids:
            syndrome[detector_id] ^= 1
        frame_bits = np.unpackbits(
            np.frombuffer(result.candidate_observable_frame, dtype=np.uint8),
            bitorder="little",
            count=graph.num_observables,
        )
        frame ^= frame_bits
        current = oracle.decode_state(syndrome).prediction
        packed_frame = bytes(np.packbits(frame, bitorder="little"))
        assert bytes(a ^ b for a, b in zip(packed_frame, current)) == initial
