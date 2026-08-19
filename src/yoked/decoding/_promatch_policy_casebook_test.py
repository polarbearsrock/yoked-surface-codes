from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding._promatch_policy_analysis import canonical_json_bytes
from yoked.decoding import _promatch_policy_casebook as casebook


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _fixture_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutate_rows=None,
):
    root = tmp_path / "corpus"
    root.mkdir()
    config = {
        "experiment_id": "e" * 64,
        "cell": {}, "decoder": {}, "dem_options": {},
        "oracle": {"tolerance": {}},
    }
    original = {
        "experiment_id": config["experiment_id"], "cell_id": "cell",
        "worker_id": 0, "worker_shot_index": 0, "global_shot_id": 0,
        "stim_seed": 7, "physical_input_sha256": "d" * 64,
        "patch_id": 0, "basis": "X", "window_id": 0,
        "original_state_sha256": "a" * 64,
        "complete_pre_state_fingerprint": "a" * 64,
        "original_proposal_sha256": "b" * 64,
        "operational_veto_chain_rank": 1,
    }
    state_id = casebook._state_id(original)
    selection = {
        "states": [{
            "state_id": state_id,
            "original_proposal_sha256": "b" * 64,
            "selection_reasons": ["test"],
        }]
    }
    analysis = {
        "analysis_sha256": "c" * 64,
        "casebook_selection": selection,
        "tables": {"fatal_gates": [
            {"gate": k, "status": "passed-ledger-recomputed", "evidence": f"gate {k}"}
            for k in range(1, 19)
        ]},
    }
    shot = {
        "worker_id": 0, "global_shot_id": 0,
        "packed_detectors_hex": "01",
        "packed_actual_observables_hex": "ff",
        "arm_failures": {"forbidden": True},
    }
    corpus = SimpleNamespace(
        root=root, config=config,
        rows={"shots": (shot,), "counterfactuals": (original,)},
    )
    _write_json(root / "casebook" / "selection.json", selection)
    _write_json(root / "ANALYSIS_READY", {
        "casebook_selection_sha256": casebook._sha(
            (root / "casebook" / "selection.json").read_bytes()
        )
    })
    for path in (
        root / "COLLECTION_READY", root / "manifest.json",
        root / "analysis" / "manifest.json",
    ):
        _write_json(path, {"test": True})

    monkeypatch.setattr(casebook, "load_policy_audit", lambda value: corpus)
    monkeypatch.setattr(casebook, "validate_policy_protocol", lambda *args, **kwargs: None)
    monkeypatch.setattr(casebook, "analyze_policy_audit", lambda value: analysis)
    monkeypatch.setattr(casebook, "verify_existing_policy_analysis", lambda *args: {})
    graph = SimpleNamespace(
        num_detectors=8,
        edges=(
            SimpleNamespace(edge_id=0, source=0, target=1),
            SimpleNamespace(edge_id=1, source=2, target=3),
        ),
        layout=SimpleNamespace(
            coordinates=((0.0, 0.0), (1.0, 0.0), *(() for _ in range(6))),
            roles=(SimpleNamespace(patch_id=0, check_basis="X", time=0, window_id=0),) * 8,
        ),
    )
    prepared = SimpleNamespace(compiled_pu=SimpleNamespace(graph=graph))
    monkeypatch.setattr(casebook, "prepare_cell", lambda *args, **kwargs: prepared)

    observed = {}
    def expand(graph_arg, syndrome, **kwargs):
        observed["syndrome"] = np.asarray(syndrome).copy()
        observed["kwargs"] = kwargs
        assert "actual_observables" not in kwargs
        rows = [
            {
                "trajectory_origin": "casebook-exhaustive",
                "graph_fingerprint": "g" * 64, "layout_fingerprint": "l" * 64,
                "original_proposal_sha256": "b" * 64,
                "original_state_sha256": "a" * 64,
                "operational_veto_chain_rank": 1, "proposal_sha256": "b" * 64,
                "cost_excess": 1.0, "cost_excess_hex": float(1).hex(),
                "oracle_policy_accepts": False, "stage": 1,
                "local_active_state_fingerprint": [0, 1],
                "detector_boundary_ids": [0, 1],
                "P_candidate_support_edge_ids": [0],
                "X_support_difference_edge_ids": [1],
                "P_intersection_R_edge_ids": [0],
                "support_cancellation_edge_ids": [0],
                "support_difference_components": [
                    {
                        "certificate_kind": "real-x-component",
                        "canonical_edge_ids": [1],
                        "support_cancellation_edge_ids": [],
                        "component_detector_ids": [2, 3],
                        "candidate_support_witness_edge_ids": [],
                        "candidate_boundary_witness_detector_ids": [],
                        "labels": ["yoke-hub"],
                        "candidate_relevant": False,
                        "candidate_relevance_reasons": [],
                    },
                    {
                        "certificate_kind": "support-cancellation",
                        "canonical_edge_ids": [],
                        "support_cancellation_edge_ids": [0],
                        "component_detector_ids": [],
                        "candidate_support_witness_edge_ids": [],
                        "candidate_boundary_witness_detector_ids": [],
                        "labels": ["support-cancellation"],
                        "candidate_relevant": True,
                        "candidate_relevance_reasons": [
                            "candidate-residual-support-cancellation"
                        ],
                    },
                ],
                "support_difference_component_labels": ["support-cancellation"],
                "support_difference_representation_version": (
                    "promatch-support-difference-v2"
                ),
                "disconnected_support_reconfiguration": True,
                "feature_visibility": {"candidate": "L1-local-dynamic"},
                "terminal_action": None, "exhaustion_kind": None,
                "support_classification_wall_ns": 999,
            },
            {
                "trajectory_origin": "casebook-exhaustive",
                "graph_fingerprint": "g" * 64, "layout_fingerprint": "l" * 64,
                "original_proposal_sha256": "b" * 64,
                "original_state_sha256": "a" * 64,
                "operational_veto_chain_rank": 2, "proposal_sha256": "f" * 64,
                "cost_excess": 0.0, "cost_excess_hex": float(0).hex(),
                "oracle_policy_accepts": True, "stage": 2,
                "local_active_state_fingerprint": [0, 1],
                "detector_boundary_ids": [0, 1],
                "P_candidate_support_edge_ids": [0],
                "X_support_difference_edge_ids": [0],
                "P_intersection_R_edge_ids": [],
                "support_cancellation_edge_ids": [],
                "support_difference_components": [{
                    "certificate_kind": "real-x-component",
                    "canonical_edge_ids": [0],
                    "support_cancellation_edge_ids": [],
                    "component_detector_ids": [0, 1],
                    "candidate_support_witness_edge_ids": [0],
                    "candidate_boundary_witness_detector_ids": [0, 1],
                    "labels": ["in-domain"],
                    "candidate_relevant": True,
                    "candidate_relevance_reasons": [
                        "candidate-boundary-detector", "candidate-support-edge",
                    ],
                }],
                "support_difference_component_labels": ["in-domain"],
                "support_difference_representation_version": (
                    "promatch-support-difference-v2"
                ),
                "disconnected_support_reconfiguration": False,
                "feature_visibility": {"candidate": "L1-local-dynamic"},
                "terminal_action": "exhaustive-true-exhaustion",
                "exhaustion_kind": "proposal", "counterfactual_wall_ns": 888,
            },
        ]
        if mutate_rows is not None:
            mutate_rows(rows)
        return rows
    monkeypatch.setattr(casebook, "expand_policy_casebook_state", expand)
    return root, config, state_id, observed


def test_authenticated_expansion_is_detector_only_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, state_id, observed = _fixture_root(tmp_path, monkeypatch)
    manifest = casebook.expand_authenticated_policy_casebook(root, config=config)
    assert manifest["selected_states"] == 1
    assert observed["syndrome"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert (root / "EXPANSION_READY").is_file()
    assert (root / "casebook" / "expansion" / "diagrams" / f"{state_id}.png").is_file()
    assert manifest["diagram_images_scientifically_digested"] is False
    state_object = json.loads(
        (root / "casebook" / "expansion" / "states" / f"{state_id}.json").read_text()
    )
    assert state_object["graph_fingerprint"] == "g" * 64
    assert state_object["layout_fingerprint"] == "l" * 64
    assert state_object["first_safe_rank"] == 2
    assert len(state_object["all_veto_timeline"]) == 2
    assert "true exhaustion" in state_object["factual_caption"]
    assert "1 candidate-relevant X" in state_object["factual_caption"]
    assert "1 disconnected X" in state_object["factual_caption"]
    assert "1 P intersection R cancellation" in state_object["factual_caption"]
    assert state_object["context_labels"] == ["in-domain", "support-cancellation"]
    assert state_object["remote_support_role_labels"] == ["yoke-hub"]
    assert "yoke-hub" not in state_object["context_labels"]
    assert state_object["all_veto_timeline"][0] == {
        "rank": 1,
        "stage": 1,
        "proposal_sha256": "b" * 64,
        "oracle_policy_accepts": False,
        "candidate_path_edge_ids": [0],
        "support_difference_edge_ids": [1],
        "candidate_relevant_support_difference_edge_ids": [],
        "disconnected_support_difference_edge_ids": [1],
        "support_cancellation_edge_ids": [0],
        "support_difference_components": [
            {
                "certificate_kind": "real-x-component",
                "canonical_edge_ids": [1],
                "support_cancellation_edge_ids": [],
                "component_detector_ids": [2, 3],
                "candidate_support_witness_edge_ids": [],
                "candidate_boundary_witness_detector_ids": [],
                "labels": ["yoke-hub"],
                "candidate_relevant": False,
                "candidate_relevance_reasons": [],
            },
            {
                "certificate_kind": "support-cancellation",
                "canonical_edge_ids": [],
                "support_cancellation_edge_ids": [0],
                "component_detector_ids": [],
                "candidate_support_witness_edge_ids": [],
                "candidate_boundary_witness_detector_ids": [],
                "labels": ["support-cancellation"],
                "candidate_relevant": True,
                "candidate_relevance_reasons": [
                    "candidate-residual-support-cancellation"
                ],
            },
        ],
        "disconnected_support_reconfiguration": True,
        "context_labels": ["support-cancellation"],
        "remote_support_role_labels": ["yoke-hub"],
    }
    snapshot = state_object["support_graph_snapshot"]
    assert snapshot["schema"] == "promatch-l1-policy-audit-support-graph-snapshot-v2"
    assert snapshot["graph_fingerprint"] == "g" * 64
    assert [row["detector_id"] for row in snapshot["detectors"]] == [0, 1, 2, 3]
    assert snapshot["edges"] == [
        {
            "candidate_ranks": [1, 2], "edge_id": 0,
            "membership": [
                "candidate-P", "support-difference-X", "candidate-relevant-X",
                "support-cancellation-P-intersection-R",
            ],
            "source": 0, "support_difference_ranks": [2], "target": 1,
            "candidate_relevant_ranks": [2], "disconnected_ranks": [],
            "support_cancellation_ranks": [1],
        },
        {
            "candidate_ranks": [], "edge_id": 1,
            "membership": ["support-difference-X", "disconnected-X"],
            "source": 2, "support_difference_ranks": [1], "target": 3,
            "candidate_relevant_ranks": [], "disconnected_ranks": [1],
            "support_cancellation_ranks": [],
        },
    ]
    with __import__("gzip").open(
        root / "casebook" / "expansion" / "exhaustive.jsonl.gz", "rt"
    ) as stream:
        rows = [json.loads(line) for line in stream]
    assert all(not any(key.endswith("_wall_ns") for key in row) for row in rows)
    assert casebook.expand_authenticated_policy_casebook(root, config=config) == manifest


def _remove_v2_version(rows) -> None:
    del rows[0]["support_difference_representation_version"]


def _add_component_field(rows) -> None:
    rows[0]["support_difference_components"][0]["unexpected"] = True


def _invalidate_component_kind(rows) -> None:
    rows[0]["support_difference_components"][0]["certificate_kind"] = "remote-context"


def _invalidate_component_relevance(rows) -> None:
    rows[0]["support_difference_components"][0]["candidate_relevant"] = True


def _misstate_candidate_support_relevance(rows) -> None:
    rows[1]["support_difference_components"][0]["candidate_relevance_reasons"] = [
        "candidate-boundary-detector"
    ]


def _fake_candidate_support_witness(rows) -> None:
    rows[0]["support_difference_components"][0][
        "candidate_support_witness_edge_ids"
    ] = [1]


def _fake_candidate_boundary_witness(rows) -> None:
    rows[0]["support_difference_components"][0][
        "candidate_boundary_witness_detector_ids"
    ] = [2]


def _fake_cancellation_witness(rows) -> None:
    rows[0]["support_difference_components"][1]["component_detector_ids"] = [0]


def _disconnect_component_partition(rows) -> None:
    rows[0]["X_support_difference_edge_ids"] = []


def _disconnect_cancellation_certificate(rows) -> None:
    rows[0]["P_intersection_R_edge_ids"] = []
    rows[0]["support_cancellation_edge_ids"] = []


def _mislabel_remote_component_as_context(rows) -> None:
    rows[0]["support_difference_component_labels"] = [
        "support-cancellation", "yoke-hub",
    ]


def test_casebook_normalizes_in_domain_across_candidate_relevant_components() -> None:
    graph = SimpleNamespace(edges=(
        SimpleNamespace(source=0, target=1),
        SimpleNamespace(source=2, target=3),
    ))
    row = {
        "support_difference_representation_version": "promatch-support-difference-v2",
        "P_candidate_support_edge_ids": [0, 1],
        "X_support_difference_edge_ids": [0, 1],
        "P_intersection_R_edge_ids": [],
        "support_cancellation_edge_ids": [],
        "detector_boundary_ids": [0, 2],
        "support_difference_component_labels": ["yoke"],
        "disconnected_support_reconfiguration": False,
        "support_difference_components": [
            {
                "certificate_kind": "real-x-component",
                "canonical_edge_ids": [0], "support_cancellation_edge_ids": [],
                "component_detector_ids": [0, 1],
                "candidate_support_witness_edge_ids": [0],
                "candidate_boundary_witness_detector_ids": [0],
                "labels": ["in-domain"], "candidate_relevant": True,
                "candidate_relevance_reasons": [
                    "candidate-boundary-detector", "candidate-support-edge"
                ],
            },
            {
                "certificate_kind": "real-x-component",
                "canonical_edge_ids": [1], "support_cancellation_edge_ids": [],
                "component_detector_ids": [2, 3],
                "candidate_support_witness_edge_ids": [1],
                "candidate_boundary_witness_detector_ids": [2],
                "labels": ["yoke"], "candidate_relevant": True,
                "candidate_relevance_reasons": [
                    "candidate-boundary-detector", "candidate-support-edge"
                ],
            },
        ],
    }
    evidence = casebook._support_evidence(row, graph=graph)
    assert evidence["context_labels"] == ["yoke"]
    row["support_difference_component_labels"] = ["in-domain", "yoke"]
    with pytest.raises(casebook.PolicyCasebookError, match="candidate-relevant component"):
        casebook._support_evidence(row, graph=graph)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_remove_v2_version, "support_difference_representation_version"),
        (_add_component_field, "exact v2 component fields"),
        (_invalidate_component_kind, "unsupported support component kind"),
        (_invalidate_component_relevance, "candidate relevance"),
        (_misstate_candidate_support_relevance, "relevance reasons disagree"),
        (_fake_candidate_support_witness, "candidate support witness disagrees"),
        (_fake_candidate_boundary_witness, "candidate boundary witness disagrees"),
        (_fake_cancellation_witness, "deterministic empty shape"),
        (_disconnect_component_partition, "exactly partition X support"),
        (_disconnect_cancellation_certificate, "exactly represent P intersection R"),
        (_mislabel_remote_component_as_context, "candidate-relevant component"),
    ],
)
def test_exhaustive_v2_support_evidence_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, message: str,
) -> None:
    root, config, _, _ = _fixture_root(
        tmp_path, monkeypatch, mutate_rows=mutation,
    )
    with pytest.raises(casebook.PolicyCasebookError, match=message):
        casebook.expand_authenticated_policy_casebook(root, config=config)


def test_expansion_tamper_and_complete_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, _, _ = _fixture_root(tmp_path, monkeypatch)
    casebook.expand_authenticated_policy_casebook(root, config=config)
    complete = casebook.finalize_policy_audit(root, config=config)
    assert complete["fatal_gates_verified"] == 18
    assert casebook.finalize_policy_audit(root, config=config) == complete
    target = root / "casebook" / "expansion" / "exhaustive.jsonl.gz"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(casebook.PolicyCasebookError, match="expansion artifact differs"):
        casebook.verify_policy_casebook_expansion(root, config=config)


def test_existing_mismatched_complete_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, _, _ = _fixture_root(tmp_path, monkeypatch)
    casebook.expand_authenticated_policy_casebook(root, config=config)
    (root / "COMPLETE").write_text("{}\n", encoding="utf-8")
    with pytest.raises(casebook.PolicyCasebookError, match="existing COMPLETE differs"):
        casebook.finalize_policy_audit(root, config=config)


def test_recursive_ground_truth_and_gate_validation_are_fail_closed() -> None:
    with pytest.raises(casebook.PolicyCasebookError, match="ground-truth-like"):
        casebook._reject_ground_truth_keys({"nested": [{"arm_failures": {}}]})
    gates = [
        {"gate": k, "status": "collector-attested", "evidence": {"source": "manifest"}}
        for k in range(1, 19)
    ]
    assert len(casebook._fatal_gate_evidence_digest(gates)) == 64
    duplicate = [dict(row) for row in gates]
    duplicate[-1]["gate"] = 17
    with pytest.raises(casebook.PolicyCasebookError, match="missing or duplicated"):
        casebook._fatal_gate_evidence_digest(duplicate)
    missing_evidence = [dict(row) for row in gates]
    missing_evidence[0]["evidence"] = ""
    with pytest.raises(casebook.PolicyCasebookError, match="status/evidence"):
        casebook._fatal_gate_evidence_digest(missing_evidence)


def test_zero_selected_states_has_deterministic_empty_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = SimpleNamespace(rows={"counterfactuals": (), "shots": ()})
    graph = SimpleNamespace(num_detectors=8)
    monkeypatch.setattr(
        casebook, "prepare_cell",
        lambda *args, **kwargs: SimpleNamespace(compiled_pu=SimpleNamespace(graph=graph)),
    )
    rows, payloads = casebook._expansion_payloads(
        corpus=corpus, selection={"states": []},
        config={"cell": {}, "decoder": {}, "dem_options": {}, "oracle": {"tolerance": {}}},
    )
    assert rows == []
    assert set(payloads) == {"exhaustive.jsonl.gz"}
    assert __import__("gzip").decompress(payloads["exhaustive.jsonl.gz"]) == b""
