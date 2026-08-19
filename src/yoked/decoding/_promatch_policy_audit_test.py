from __future__ import annotations

import numpy as np
import pymatching
import pytest
import json
from types import SimpleNamespace

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._promatch_graph import CompiledPromatchGraph, DomainGraph, Edge, compile_matching_graph
from yoked.decoding._promatch_layout import L1BodyDetector, L1FullHistoryDomain, compile_layout
from yoked.decoding._promatch_oracle import (
    CostClassification,
    OracleEvaluation,
    OracleTolerance,
)
from yoked.decoding._promatch_policy_audit import (
    ARM_IDS,
    _matched_partner_labels,
    _difference_components,
    _normalized_labels,
    _support_component_labels,
    audit_policy_shot,
    expand_policy_casebook_state,
)
from yoked.decoding._promatch_policy_experiment import _audit_policy_shot


class _SyntheticLayout:
    def __init__(self, count: int) -> None:
        self.roles = tuple(L1BodyDetector(0, "X", k, 0) for k in range(count))

    def role_of(self, detector_id: int):
        return self.roles[detector_id]


def _unsafe_graph() -> CompiledPromatchGraph:
    matcher = pymatching.Matching()
    matcher.add_edge(0, 1, weight=10)
    for detector in range(12):
        matcher.add_boundary_edge(detector, weight=1)
    matcher.ensure_num_fault_ids(1)
    role = L1BodyDetector(0, "X", 0, 0)
    edges = []
    local = None
    for edge_id, (source, target, data) in enumerate(matcher.edges()):
        edge = Edge(edge_id, int(source), None if target is None else int(target),
                    float(data["weight"]), b"\0", role, None if target is None else role)
        edges.append(edge)
        if edge.target == 1 and edge.source == 0:
            local = edge
    assert local is not None
    domain = L1FullHistoryDomain(0, "X")
    adjacency = {k: () for k in range(12)}
    adjacency[0] = (local,)
    adjacency[1] = (local,)
    domain_graph = DomainGraph(domain, tuple(range(12)), (local,), adjacency,
                               {0: (1,), 1: (0,), **{k: () for k in range(2, 12)}}, (),
                               {k: () for k in range(12)})
    return CompiledPromatchGraph(_SyntheticLayout(12), matcher, tuple(edges),
                                 {domain: domain_graph}, "unsafe-synthetic-v1", True, 12, 1)


def _evaluation(
    *, base=(), candidate=(), residual=(), frame_compatible=True, accepted=False
) -> OracleEvaluation:
    return OracleEvaluation(
        policy="frame", accepted=accepted, cost_compatible=accepted,
        frame_compatible=frame_compatible,
        cost_classification=(CostClassification.COMPATIBLE if accepted
                             else CostClassification.POSITIVE_EXCESS),
        cost_excess=0.0 if accepted else 1.0, tau_k=1e-6,
        composite_weight=1.0, accumulated_frame=b"\0", base_prediction=b"\0",
        residual_prediction=b"\0", base_frame=b"\0",
        candidate_composite_frame=b"\0" if frame_compatible else b"\1",
        base_support_edge_ids=tuple(base), residual_support_edge_ids=tuple(residual),
        base_support_weight=0.0, residual_support_weight=0.0,
        base_backend_weight=0.0, residual_backend_weight=0.0,
        base_tau_weight=1e-6, residual_tau_weight=1e-6,
        candidate_edge_ids=tuple(candidate), candidate_boundary_detector_ids=(0, 1),
        candidate_observable_frame=b"\0", candidate_weight=1.0,
    )


def test_difference_components_retain_remote_x_and_independent_cancellation() -> None:
    graph = _unsafe_graph()
    domain = next(iter(graph.domain_graphs))
    local = next(edge.edge_id for edge in graph.edges if edge.target == 1 and edge.source == 0)
    remote = next(edge.edge_id for edge in graph.edges if edge.source == 2 and edge.target is None)
    components, labels, disconnected = _difference_components(
        graph,
        _evaluation(candidate=(local,), residual=(local, remote)),
        domain,
    )
    real = [row for row in components if row["certificate_kind"] == "real-x-component"]
    cancellation = [row for row in components if row["certificate_kind"] == "support-cancellation"]
    assert [row["canonical_edge_ids"] for row in real] == [[remote]]
    assert not real[0]["candidate_relevant"]
    assert real[0]["candidate_relevance_reasons"] == []
    assert len(cancellation) == 1
    assert cancellation[0]["support_cancellation_edge_ids"] == [local]
    assert labels == ["support-cancellation"]
    assert disconnected


def test_difference_components_accept_x_empty_cancellation_and_reject_unexplained() -> None:
    graph = _unsafe_graph()
    domain = next(iter(graph.domain_graphs))
    local = next(edge.edge_id for edge in graph.edges if edge.target == 1 and edge.source == 0)
    components, labels, disconnected = _difference_components(
        graph, _evaluation(candidate=(local,), residual=(local,)), domain
    )
    assert len(components) == 1
    assert components[0]["certificate_kind"] == "support-cancellation"
    assert labels == ["support-cancellation"]
    assert not disconnected
    with pytest.raises(AssertionError, match="neither an X support difference"):
        _difference_components(graph, _evaluation(), domain)
    with pytest.raises(AssertionError, match="X-empty support decomposition"):
        _difference_components(
            graph,
            _evaluation(candidate=(local,), residual=(local,), frame_compatible=False),
            domain,
        )


def test_difference_components_retain_remote_x_without_cancellation() -> None:
    graph = _unsafe_graph()
    domain = next(iter(graph.domain_graphs))
    local = next(edge.edge_id for edge in graph.edges if edge.target == 1 and edge.source == 0)
    remote = next(edge.edge_id for edge in graph.edges if edge.source == 2 and edge.target is None)
    components, labels, disconnected = _difference_components(
        graph,
        _evaluation(base=(local, remote), candidate=(local,)),
        domain,
    )
    assert len(components) == 1
    assert components[0]["certificate_kind"] == "real-x-component"
    assert components[0]["canonical_edge_ids"] == [remote]
    assert not components[0]["candidate_relevant"]
    assert labels == []
    assert disconnected


def _rollback_accounting_graph() -> CompiledPromatchGraph:
    matcher = pymatching.Matching()
    matcher.add_edge(0, 1, weight=1)
    matcher.add_edge(2, 3, weight=10)
    for detector in range(14):
        matcher.add_boundary_edge(detector, weight=1)
    matcher.ensure_num_fault_ids(1)
    role = L1BodyDetector(0, "X", 0, 0)
    edges, local = [], []
    for edge_id, (source, target, data) in enumerate(matcher.edges()):
        edge = Edge(edge_id, int(source), None if target is None else int(target),
                    float(data["weight"]), b"\0", role, None if target is None else role)
        edges.append(edge)
        if edge.target is not None:
            local.append(edge)
    domain = L1FullHistoryDomain(0, "X")
    adjacency = {k: [] for k in range(14)}
    neighbors = {k: [] for k in range(14)}
    for edge in local:
        adjacency[edge.source].append(edge); adjacency[edge.target].append(edge)
        neighbors[edge.source].append(edge.target); neighbors[edge.target].append(edge.source)
    domain_graph = DomainGraph(
        domain, tuple(range(14)), tuple(local),
        {k: tuple(v) for k, v in adjacency.items()},
        {k: tuple(v) for k, v in neighbors.items()}, (), {k: () for k in range(14)},
    )
    return CompiledPromatchGraph(_SyntheticLayout(14), matcher, tuple(edges),
                                 {domain: domain_graph}, "rollback-synthetic-v1", True, 14, 1)


def _unsafe_then_safe_graph() -> CompiledPromatchGraph:
    matcher = pymatching.Matching()
    matcher.add_edge(0, 1, weight=0.5)
    matcher.add_edge(2, 3, weight=1)
    for detector in range(14):
        matcher.add_boundary_edge(detector, weight=0.1 if detector < 2 else 10)
    matcher.ensure_num_fault_ids(1)
    role = L1BodyDetector(0, "X", 0, 0)
    edges, local = [], []
    for edge_id, (source, target, data) in enumerate(matcher.edges()):
        edge = Edge(edge_id, int(source), None if target is None else int(target),
                    float(data["weight"]), b"\0", role, None if target is None else role)
        edges.append(edge)
        if edge.target is not None:
            local.append(edge)
    domain = L1FullHistoryDomain(0, "X")
    adjacency = {k: [] for k in range(14)}
    neighbors = {k: [] for k in range(14)}
    for edge in local:
        adjacency[edge.source].append(edge); adjacency[edge.target].append(edge)
        neighbors[edge.source].append(edge.target); neighbors[edge.target].append(edge.source)
    domain_graph = DomainGraph(
        domain, tuple(range(14)), tuple(local),
        {k: tuple(v) for k, v in adjacency.items()},
        {k: tuple(v) for k, v in neighbors.items()}, (), {k: () for k in range(14)},
    )
    return CompiledPromatchGraph(_SyntheticLayout(14), matcher, tuple(edges),
                                 {domain: domain_graph}, "unsafe-safe-synthetic-v1", True, 14, 1)


@pytest.fixture(scope="module")
def graph():
    circuit = yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=6,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=2,
        num_patches=2,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    return compile_matching_graph(dem, compile_layout(dem))


def test_policy_audit_is_ground_truth_free_deterministic_and_adapter_compatible(graph) -> None:
    syndrome = np.zeros(graph.num_detectors, dtype=np.uint8)
    first = audit_policy_shot(graph, syndrome, tolerance=OracleTolerance())
    second = audit_policy_shot(graph, syndrome, tolerance=OracleTolerance())

    assert first == second
    assert set(first) == {"arm_predictions", "shot", "proposals", "counterfactuals", "domains"}
    assert set(first["arm_predictions"]) == set(ARM_IDS)
    assert first["arm_predictions"][ARM_IDS[3]] == first["arm_predictions"][ARM_IDS[0]]
    assert first["arm_predictions"][ARM_IDS[4]] == first["arm_predictions"][ARM_IDS[0]]
    assert "actual_observables" not in repr(first)

    normalized = _audit_policy_shot(graph, syndrome, tolerance=OracleTolerance())
    assert set(normalized.arm_predictions) == set(ARM_IDS)


def test_policy_audit_validates_detector_input(graph) -> None:
    with pytest.raises(ValueError, match="binary vector"):
        audit_policy_shot(
            graph,
            np.zeros(graph.num_detectors + 1, dtype=np.uint8),
            tolerance=OracleTolerance(),
        )
    bad = np.zeros(graph.num_detectors, dtype=np.int8)
    bad[0] = 2
    with pytest.raises(ValueError, match="binary vector"):
        audit_policy_shot(graph, bad, tolerance=OracleTolerance())


def test_context_views_are_sorted_and_separate_on_a_nontrivial_shot(graph) -> None:
    # Activate enough detectors in the first large domain to force at least one
    # durable V3 proposal while keeping the test deterministic and sampler-free.
    syndrome = np.zeros(graph.num_detectors, dtype=np.uint8)
    domain = next(d for d in sorted(graph.domain_graphs) if len(graph.domain_graphs[d].detector_ids) > 10)
    syndrome[list(graph.domain_graphs[domain].detector_ids)] = 1
    result = audit_policy_shot(graph, syndrome, tolerance=OracleTolerance())

    for row in result["proposals"] + result["counterfactuals"]:
        assert row["matched_partner_labels"] == sorted(set(row["matched_partner_labels"]))
        assert row["support_path_labels"] == sorted(set(row["support_path_labels"]))
        assert row["support_difference_component_labels"] == sorted(
            set(row["support_difference_component_labels"])
        )
        assert row["decision_weight_hex"] == float(row["decision_weight"]).hex()
        assert row["omitted_context_labels"] == sorted(set(row["omitted_context_labels"]))
        assert row["omitted_context_labels"] == _normalized_labels(
            set(row["matched_partner_labels"]) | set(row["support_path_labels"])
        )
        if any(label != "in-domain" for label in row["omitted_context_labels"]):
            assert "in-domain" not in row["omitted_context_labels"]
        assert row["degeneracy_diagnostics"] == sorted(set(row["degeneracy_diagnostics"]))
        assert set(row["degeneracy_diagnostics"]) <= {
            "same-pair-different-path-or-frame", "equal-weight-logical-class",
            "disconnected-support-reconfiguration", "unclassified"
        }
        real_union = set()
        cancellation_union = set()
        for component in row["support_difference_components"]:
            assert "support_cancellation_edge_ids" in component
            assert component["component_detector_ids"] == sorted(
                set(component["component_detector_ids"])
            )
            assert component["candidate_support_witness_edge_ids"] == sorted(
                set(component["candidate_support_witness_edge_ids"])
            )
            assert component["candidate_boundary_witness_detector_ids"] == sorted(
                set(component["candidate_boundary_witness_detector_ids"])
            )
            assert component["candidate_relevance_reasons"] == sorted(
                set(component["candidate_relevance_reasons"])
            )
            if component["certificate_kind"] == "real-x-component":
                real_union.update(component["canonical_edge_ids"])
                assert not component["support_cancellation_edge_ids"]
            else:
                assert component["certificate_kind"] == "support-cancellation"
                assert not component["canonical_edge_ids"]
                cancellation_union.update(component["support_cancellation_edge_ids"])
        assert row["support_cancellation_edge_ids"] == sorted(
            set(row["candidate_support_edge_ids"]).intersection(row["residual_support_edge_ids"])
        )
        assert row["feature_visibility"]["window_offset"] == "L1-local-dynamic"
        b = set(row["B_base_support_edge_ids"])
        p = set(row["P_candidate_support_edge_ids"])
        r = set(row["R_residual_support_edge_ids"])
        assert row["Q_forced_parity_support_edge_ids"] == sorted(p ^ r)
        assert row["X_support_difference_edge_ids"] == sorted(b ^ p ^ r)
        assert row["P_intersection_R_edge_ids"] == sorted(p & r)
        assert sorted(real_union) == row["X_support_difference_edge_ids"]
        assert sorted(cancellation_union) == row["P_intersection_R_edge_ids"]
        assert row["support_difference_representation_version"] == (
            "promatch-support-difference-v2"
        )
        assert row["supports_square_free"]
        for name in (
            "base_support_weight", "residual_support_weight", "base_backend_weight",
            "residual_backend_weight", "base_tau_weight", "residual_tau_weight",
            "candidate_weight", "composite_weight", "cost_excess", "tau_k",
        ):
            assert row[f"{name}_hex"] == float(row[name]).hex()
        assert isinstance(row["oracle_evaluation_id"], str)
        assert isinstance(row["oracle_base_solution_id"], str)
        assert isinstance(row["oracle_residual_solution_id"], str)
        assert row["state_oracle_call_count"] >= 1
        assert row["graph_fingerprint"] == graph.fingerprint
        assert row["layout_fingerprint"] == graph.layout.fingerprint
        assert "window_start_offset" in row["endpoint_local_features"][0]
        assert "window_end_offset" in row["endpoint_local_features"][0]
        assert "circuit_terminal_offset" in row["endpoint_local_features"][0]
        assert isinstance(row["stage_isolation_predicate"], bool)
        assert row["candidate_enumeration_wall_ns"] >= 0
        assert row["stage3_enumeration_wall_ns"] >= 0
        assert row["support_classification_wall_ns"] >= 0
    if result["counterfactuals"]:
        grouped = {}
        for row in result["counterfactuals"]:
            grouped.setdefault(row["original_proposal_sha256"], []).append(row)
        for rows in grouped.values():
            assert [r["operational_veto_chain_rank"] for r in rows] == list(
                range(1, len(rows) + 1)
            )
            assert sum(r["terminal_action"] is not None for r in rows) == 1


def test_unsafe_original_is_rank_one_and_runs_to_true_exhaustion() -> None:
    result = audit_policy_shot(
        _unsafe_graph(), np.ones(12, dtype=np.uint8), tolerance=OracleTolerance()
    )
    assert len(result["proposals"]) == 1
    original = result["proposals"][0]
    assert not original["cost_compatible"]
    chain = result["counterfactuals"]
    assert len(chain) >= 1
    assert chain[0]["proposal_sha256"] == original["proposal_sha256"]
    assert [row["operational_veto_chain_rank"] for row in chain] == list(
        range(1, len(chain) + 1)
    )
    assert chain[-1]["terminal_action"] == "abstain-true-exhaustion"
    assert chain[-1]["first_safe_rank"] is None
    assert chain[-1]["exhaustion_kind"] == "proposal"
    assert chain[-1]["veto_budget"] is None


def test_specific_context_suppresses_in_domain() -> None:
    assert _normalized_labels({"in-domain"}) == ["in-domain"]
    assert _normalized_labels({"in-domain", "yoke", "terminal"}) == ["terminal", "yoke"]


def test_selected_support_does_not_label_an_absent_proposal_endpoint() -> None:
    graph = _unsafe_graph()
    domain = next(iter(graph.domain_graphs))
    boundary_edge = next(edge for edge in graph.edges if edge.target is None and edge.source not in {0, 1})
    assert _support_component_labels(graph, [boundary_edge.edge_id], (0, 1), domain) == []


def test_transaction_and_partial_workload_accounting_diverge_after_rollback() -> None:
    result = audit_policy_shot(
        _rollback_accounting_graph(), np.ones(14, dtype=np.uint8),
        tolerance=OracleTolerance(),
    )
    summaries = result["shot"]["arm_summaries"]
    tx = summaries[ARM_IDS[3]]
    partial = summaries[ARM_IDS[4]]
    assert tx["provisional_events_removed"] > 0
    assert tx["durable_events_removed"] == 0
    assert tx["events_lost_to_rollback"] == tx["provisional_events_removed"]
    assert partial["durable_events_removed"] == partial["provisional_events_removed"] > 0
    assert partial["events_lost_to_rollback"] == 0
    origins = {row["trajectory_origin"] for row in result["domains"]}
    assert {"sequential-o-cost-tx", "sequential-o-frame-tx", "sequential-o-frame-partial"} <= origins


def test_casebook_expansion_continues_after_first_safe_to_true_exhaustion() -> None:
    graph = _unsafe_then_safe_graph()
    syndrome = np.ones(14, dtype=np.uint8)
    audit = audit_policy_shot(graph, syndrome, tolerance=OracleTolerance())
    original = audit["proposals"][0]
    assert not original["oracle_policy_accepts"]
    rows = expand_policy_casebook_state(
        graph, syndrome,
        original_proposal_sha256=original["proposal_sha256"],
        original_state_sha256=original["complete_pre_state_fingerprint"],
        tolerance=OracleTolerance(),
    )
    safe_indices = [k for k, row in enumerate(rows) if row["oracle_policy_accepts"]]
    assert safe_indices
    assert safe_indices[0] < len(rows) - 1
    assert rows[safe_indices[0]]["is_first_safe_alternative"]
    assert rows[-1]["terminal_action"] == "exhaustive-true-exhaustion"
    assert rows[-1]["exhaustion_kind"] == "proposal"
    assert [row["operational_veto_chain_rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert {row["state_oracle_call_count"] for row in rows} == {len(rows)}
    assert sum(row["matched_backend_call_count_delta"] for row in rows) == 1
    assert rows[0]["matched_backend_cache_hit"] is False
    assert all(row["matched_backend_cache_hit"] for row in rows[1:])
    json.dumps(rows, sort_keys=True, allow_nan=False)




@pytest.mark.parametrize(
    "pairs, message",
    [
        ([[0, 0]], "multiple matched pairs"),
        ([[0, 2]], "inactive detector"),
        ([[0, -1], [0, -1]], "multiple matched pairs"),
        ([[-1, -1]], "two boundaries"),
        ([[0, 3]], "invalid detector ID"),
    ],
)
def test_matched_active_pair_validation_rejects_malformed_backend(pairs, message) -> None:
    class Matcher:
        def decode_to_matched_dets_array(self, syndrome):
            return np.asarray(pairs, dtype=np.int64)

    fake = SimpleNamespace(
        matcher=Matcher(), num_detectors=3,
        layout=_SyntheticLayout(3),
    )
    domain = L1FullHistoryDomain(0, "X")
    with pytest.raises(AssertionError, match=message):
        _matched_partner_labels(fake, np.asarray([1, 0, 0], dtype=np.uint8), (0, 1), domain)
