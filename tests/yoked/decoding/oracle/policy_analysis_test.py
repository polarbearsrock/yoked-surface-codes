from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding.oracle.policy_analysis import (
    COLLECTOR_GATE_ATTESTATION_SCHEMA,
    COLLECTOR_GATE_CHECKS,
    PolicyAnalysisError,
    PolicyAuditCorpus,
    analyze_policy_audit,
    canonical_json_bytes,
    clustered_bootstrap_ratios,
    empirical_type7,
    exact_ecdf,
    load_policy_audit,
    policy_human_report_bytes,
    select_casebook,
    _context,
    _sum_optional,
    verify_existing_policy_analysis,
    write_policy_analysis,
)
from yoked.decoding.oracle.policy_experiment import ARM_IDS


def test_disconnected_only_support_has_no_candidate_context() -> None:
    row = _proposal(0, "a", stage=1, safe=False, context="yoke", commit_index=0)
    row["support_difference_components"][0]["candidate_relevant"] = False
    row["support_difference_components"][0]["candidate_relevance_reasons"] = []
    row["support_difference_components"][0]["component_detector_ids"] = []
    row["support_difference_components"][0]["candidate_support_witness_edge_ids"] = []
    row["support_difference_components"][0][
        "candidate_boundary_witness_detector_ids"
    ] = []
    row["base_support_edge_ids"] = row["X_support_difference_edge_ids"]
    row["B_base_support_edge_ids"] = row["X_support_difference_edge_ids"]
    row["candidate_support_edge_ids"] = []
    row["P_candidate_support_edge_ids"] = []
    row["Q_forced_parity_support_edge_ids"] = []
    row["support_difference_component_labels"] = []
    row["exclusive_support_component_context"] = None
    row["disconnected_support_reconfiguration"] = True
    row["degeneracy_diagnostics"] = [
        "disconnected-support-reconfiguration",
        "equal-weight-logical-class",
    ]
    context = _context(row)
    assert context["support_difference_component_labels"] == ()
    assert context["exclusive_support_component_context"] is None


def test_omitted_context_union_normalizes_in_domain_with_specific_path_labels() -> None:
    row = _proposal(0, "a", stage=1, safe=False, context="yoke", commit_index=0)
    row["base_matched_partner_labels"] = ["in-domain"]
    row["base_support_path_labels"] = ["cross-patch-or-basis", "cross-window", "yoke"]
    row["omitted_context_labels"] = ["cross-patch-or-basis", "cross-window", "yoke"]
    context = _context(row)
    assert context["omitted_context_labels"] == (
        "cross-patch-or-basis",
        "cross-window",
        "yoke",
    )
    row["omitted_context_labels"] = [
        "cross-patch-or-basis",
        "cross-window",
        "in-domain",
        "yoke",
    ]
    with pytest.raises(PolicyAnalysisError, match="in-domain is not exclusive"):
        _context(row)


def test_candidate_component_union_normalizes_in_domain_across_components() -> None:
    row = _proposal(0, "a", stage=1, safe=False, context="yoke", commit_index=0)
    row["detector_boundary_ids"] = [0, 1, 50]
    for field in (
        "candidate_support_edge_ids",
        "P_candidate_support_edge_ids",
        "Q_forced_parity_support_edge_ids",
        "X_support_difference_edge_ids",
    ):
        row[field] = [1, 2]
    row["support_difference_components"].append(
        {
            "certificate_kind": "real-x-component",
            "canonical_edge_ids": [2],
            "support_cancellation_edge_ids": [],
            "component_detector_ids": [50, 51],
            "candidate_support_witness_edge_ids": [2],
            "candidate_boundary_witness_detector_ids": [50],
            "labels": ["in-domain"],
            "candidate_relevant": True,
            "candidate_relevance_reasons": [
                "candidate-boundary-detector",
                "candidate-support-edge",
            ],
        }
    )
    context = _context(row)
    assert context["support_difference_component_labels"] == ("yoke",)
    row["support_difference_component_labels"] = ["in-domain", "yoke"]
    with pytest.raises(PolicyAnalysisError, match="component union"):
        _context(row)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(shot: int) -> dict:
    if shot < 16:
        worker = shot // 4
        worker_start = 4 * worker
    else:
        worker = 4 + (shot - 16) // 3
        worker_start = 16 + 3 * (worker - 4)
    return {
        "experiment_id": "b1-test",
        "cell_id": "cell-a",
        "worker_id": worker,
        "worker_shot_index": shot - worker_start,
        "global_shot_id": shot,
        "physical_input_sha256": f"{shot + 1:064x}",
    }


def _shot(
    shot: int, *, u0: str, shadow: str, u0_failed: bool, shadow_failed: bool
) -> dict:
    predictions = {
        ARM_IDS[0]: u0,
        ARM_IDS[1]: shadow,
        ARM_IDS[2]: u0,
        ARM_IDS[3]: u0,
        ARM_IDS[4]: u0,
    }
    failures = {
        ARM_IDS[0]: u0_failed,
        ARM_IDS[1]: shadow_failed,
        ARM_IDS[2]: u0_failed,
        ARM_IDS[3]: u0_failed,
        ARM_IDS[4]: u0_failed,
    }
    summaries = {
        arm: {
            "final_residual_detector_hw": 2 if arm == ARM_IDS[1] else 3,
            "provisional_events_removed": 3 if arm == ARM_IDS[1] else 2,
            "events_lost_to_rollback": 0,
        }
        for arm in ARM_IDS
    }
    return {
        "schema": "promatch-l1-policy-audit-shot-v1",
        **_identity(shot),
        "original_detector_hw": 5,
        "arm_predictions_hex": predictions,
        "arm_failures": failures,
        "arm_summaries": summaries,
    }


def _proposal(
    shot: int,
    digest_digit: str,
    *,
    stage: int,
    safe: bool,
    context: str,
    commit_index: int,
) -> dict:
    weight = float(stage)
    margin = 0.25 * stage
    endpoints = [shot, shot + 1]
    edge_ids = [10 * shot + stage]
    degeneracy = [] if safe else ["equal-weight-logical-class"]
    return {
        "schema": "promatch-l1-policy-audit-proposal-v1",
        **_identity(shot),
        "arm_id": ARM_IDS[1],
        "trajectory_origin": "shadow-original",
        "proposal_sha256": digest_digit * 64,
        "proposal_signature": [stage, endpoints, edge_ids],
        "stage": stage,
        "patch_id": 0,
        "basis": "X",
        "window_id": 0,
        "decision": "shadow-commit",
        "durable": True,
        "trajectory_commit_index": commit_index,
        "cost_classification": "cost-compatible",
        "frame_compatible": safe,
        "oracle_policy_accepts": safe,
        "cost_excess": 0.0,
        "cost_excess_hex": (0.0).hex(),
        "tau_k": 1e-9,
        "tau_k_hex": (1e-9).hex(),
        "decision_weight": weight,
        "decision_weight_hex": weight.hex(),
        "canonical_edge_count": stage,
        "ordered_endpoints": endpoints,
        "detector_boundary_ids": endpoints,
        "canonical_edge_ids": edge_ids,
        "base_matched_active_pairs": [[1000, 1001]],
        "base_support_edge_ids": [],
        "candidate_support_edge_ids": edge_ids,
        "residual_support_edge_ids": [],
        "B_base_support_edge_ids": [],
        "P_candidate_support_edge_ids": edge_ids,
        "R_residual_support_edge_ids": [],
        "Q_forced_parity_support_edge_ids": edge_ids,
        "X_support_difference_edge_ids": edge_ids,
        "P_intersection_R_edge_ids": [],
        "support_cancellation_edge_ids": [],
        "supports_square_free": True,
        "B_base_support_square_free": True,
        "P_candidate_support_square_free": True,
        "R_residual_support_square_free": True,
        "Q_forced_parity_support_square_free": True,
        "X_support_difference_square_free": True,
        "base_frame": "00",
        "candidate_frame": "00",
        "absolute_weight_margin": margin,
        "absolute_weight_margin_hex": margin.hex(),
        "same_stage_competitor_exists": True,
        "same_stage_competitor_weight": weight + margin,
        "same_stage_competitor_weight_hex": (weight + margin).hex(),
        "events_removed_if_committed": 2,
        "window_offset": 0,
        "static_boundary_competition": False,
        "domain_current_hw": 4,
        "state_total_candidate_count": 3,
        "base_matched_partner_labels": [context],
        "base_support_path_labels": [context],
        "support_difference_component_labels": [context],
        "support_difference_representation_version": "promatch-support-difference-v2",
        "support_difference_components": [
            {
                "certificate_kind": "real-x-component",
                "canonical_edge_ids": edge_ids,
                "support_cancellation_edge_ids": [],
                "component_detector_ids": endpoints,
                "candidate_support_witness_edge_ids": edge_ids,
                "candidate_boundary_witness_detector_ids": endpoints,
                "labels": [context],
                "candidate_relevant": True,
                "candidate_relevance_reasons": [
                    "candidate-boundary-detector",
                    "candidate-support-edge",
                ],
            }
        ],
        "exclusive_support_component_context": context,
        "omitted_context_labels": [context],
        "degeneracy_diagnostics": degeneracy,
        "same_pair_different_path_or_frame": False,
        "equal_weight_logical_class": not safe,
        "disconnected_support_reconfiguration": False,
        "degeneracy_unclassified": False,
    }


def _counterfactual_chain(
    original: dict, state_digit: str, *, safe_stage: int
) -> list[dict]:
    base = {
        key: value
        for key, value in original.items()
        if key not in {"schema", "decision", "durable"}
    }
    action = (
        "same-stage-alternative"
        if original["stage"] == safe_stage
        else "later-stage-alternative"
    )
    first = {
        "schema": "promatch-l1-policy-audit-counterfactual-v1",
        **base,
        "trajectory_origin": "shadow-original-state-counterfactual",
        "original_state_sha256": state_digit * 64,
        "complete_pre_state_fingerprint": state_digit * 64,
        "local_active_state_fingerprint": [1, 2],
        "original_proposal_sha256": original["proposal_sha256"],
        "operational_veto_chain_rank": 1,
        "decision": "veto",
        "terminal_action": None,
        "first_safe_alternative_proposal_sha256": f"{int(state_digit, 16) + 1:x}" * 64,
        "first_safe_rank": 2,
        "censored": False,
        "exhaustion_kind": None,
        "veto_budget": None,
        "state_oracle_call_count": 2,
        "state_total_stage3_enumeration_wall_ns": 7,
    }
    second = {
        **first,
        "proposal_sha256": f"{int(state_digit, 16) + 1:x}" * 64,
        "proposal_signature": [safe_stage, original["ordered_endpoints"], [99]],
        "canonical_edge_ids": [99],
        "stage": safe_stage,
        "operational_veto_chain_rank": 2,
        "frame_compatible": True,
        "oracle_policy_accepts": True,
        "decision": "inspect-only",
        "terminal_action": action,
        "degeneracy_diagnostics": [],
        "equal_weight_logical_class": False,
        "decision_weight": float(safe_stage) + 0.5,
        "decision_weight_hex": (float(safe_stage) + 0.5).hex(),
        "canonical_edge_count": 1,
    }
    return [first, second]


def _domain(shot: int) -> dict:
    return {
        "schema": "promatch-l1-policy-audit-domain-v1",
        **_identity(shot),
        "arm_id": ARM_IDS[1],
        "status": "success",
        "domain_initial_hw": 5,
        "residual_hw_target": 2,
        "durable_events_removed": 3,
        "provisional_events_removed": 3,
        "events_lost_to_rollback": 0,
    }


def _rows() -> dict[str, list[dict]]:
    unsafe_early = _proposal(
        2, "c", stage=2, safe=False, context="terminal", commit_index=1
    )
    unsafe_late = _proposal(2, "b", stage=4, safe=False, context="yoke", commit_index=2)
    unsafe_regression = _proposal(
        0, "a", stage=1, safe=False, context="yoke", commit_index=1
    )
    safe = _proposal(1, "d", stage=3, safe=True, context="in-domain", commit_index=1)
    counterfactuals = (
        _counterfactual_chain(unsafe_late, "e", safe_stage=4)
        + _counterfactual_chain(unsafe_early, "d", safe_stage=3)
        + _counterfactual_chain(unsafe_regression, "c", safe_stage=1)
    )
    return {
        "shots": [
            _shot(0, u0="00", shadow="01", u0_failed=False, shadow_failed=True),
            _shot(1, u0="00", shadow="00", u0_failed=False, shadow_failed=False),
            _shot(2, u0="01", shadow="00", u0_failed=True, shadow_failed=False),
        ]
        + [
            _shot(shot, u0="00", shadow="00", u0_failed=False, shadow_failed=False)
            for shot in range(3, 100)
        ],
        "proposals": [unsafe_late, safe, unsafe_regression, unsafe_early],
        "counterfactuals": counterfactuals,
        "domains": [_domain(shot) for shot in range(100)],
    }


def _write_corpus(root: Path) -> None:
    root.mkdir()
    for worker in range(32):
        (root / "shards" / f"worker-{worker:02d}").mkdir(parents=True)
    experiment = {"schema": "experiment", "experiment_id": "b1-test"}
    config = {
        "schema": "config",
        "experiment_id": "b1-test",
        "sampling": {"total_shots": 20_000},
        "bootstrap": {
            "replicates": 64,
            "seed_roots": {"proposal": "11" * 32, "workload": "22" * 32},
        },
    }
    (root / "experiment.json").write_bytes(canonical_json_bytes(experiment) + b"\n")
    (root / "config.json").write_bytes(canonical_json_bytes(config) + b"\n")
    artifacts = []
    manifest_shards = []
    all_rows = _rows()
    for worker in range(32):
        core_timing = {kind: [] for kind in all_rows}
        for kind, rows in all_rows.items():
            worker_rows = [row for row in rows if row["worker_id"] == worker]
            scientific_rows = []
            for row_index, row in enumerate(worker_rows):
                scientific = {}
                captured = {}
                for key, value in row.items():
                    if key.endswith("_wall_ns") or key == "timing_telemetry":
                        captured[key] = value
                    else:
                        scientific[key] = value
                scientific_rows.append(scientific)
                if captured:
                    core_timing[kind].append(
                        {
                            "row_index": row_index,
                            "worker_shot_index": row["worker_shot_index"],
                            "global_shot_id": row["global_shot_id"],
                            "timing": captured,
                        }
                    )
            raw = b"".join(canonical_json_bytes(row) + b"\n" for row in scientific_rows)
            compressed = gzip.compress(raw, compresslevel=9, mtime=0)
            relative = f"shards/worker-{worker:02d}/{kind}.jsonl.gz"
            (root / relative).write_bytes(compressed)
            artifacts.append(
                {
                    "path": relative,
                    "compressed_sha256": _digest(compressed),
                    "uncompressed_sha256": _digest(raw),
                    "row_count": len(worker_rows),
                }
            )
        timing = {
            "schema": "promatch-l1-policy-audit-timing-v2",
            "worker_id": worker,
            "core_timing_by_ledger": core_timing,
            "scientifically_deterministic": False,
            "excluded_from_bit_exact_ledger_contract": True,
        }
        timing_relative = f"shards/worker-{worker:02d}/timing.json"
        timing_bytes = canonical_json_bytes(timing) + b"\n"
        (root / timing_relative).write_bytes(timing_bytes)
        manifest_shards.append(
            {
                "worker": {"worker_id": worker},
                "timing_path": timing_relative,
                "timing_sha256": _digest(timing_bytes),
                "nondeterministic_telemetry_paths": [timing_relative],
            }
        )
    manifest = {
        "schema": "promatch-l1-policy-audit-manifest-v1",
        "experiment_id": "b1-test",
        "mode": "probe",
        "workers": 32,
        "shots": 100,
        "artifacts": artifacts,
        "shards": manifest_shards,
        "tail_censor_attestation": {
            "uncapped_counterfactuals": True,
            "censored_states": 0,
            "repeated_same_state_proposal_signatures": 0,
            "worker_timeouts": 0,
            "output_truncations": 0,
        },
        "fatal_gate_attestations": {
            str(gate): {
                "schema": COLLECTOR_GATE_ATTESTATION_SCHEMA,
                "gate": gate,
                "status": "passed",
                "scope": "frozen-protocol-required-scope",
                "checks": list(checks),
                "failures": 0,
            }
            for gate, checks in COLLECTOR_GATE_CHECKS.items()
        },
    }
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    ready = {
        "schema": "promatch-l1-policy-audit-collection-ready-v1",
        "experiment_id": "b1-test",
        "manifest_sha256": _digest((root / "manifest.json").read_bytes()),
    }
    (root / "COLLECTION_READY").write_bytes(canonical_json_bytes(ready) + b"\n")


def _replace_corpus(
    corpus: PolicyAuditCorpus,
    *,
    kind: str | None = None,
    rows: list[dict] | None = None,
    manifest: dict | None = None,
) -> PolicyAuditCorpus:
    replacement = corpus.rows
    if kind is not None:
        assert rows is not None
        replacement = {**corpus.rows, kind: tuple(rows)}
    return PolicyAuditCorpus(
        root=corpus.root,
        experiment=corpus.experiment,
        config=corpus.config,
        manifest=corpus.manifest if manifest is None else manifest,
        rows=replacement,
        source_hashes=corpus.source_hashes,
    )


def test_type7_and_ecdf_are_exact() -> None:
    assert empirical_type7([0.0, 10.0], 0.25) == 2.5
    assert exact_ecdf([2, 1, 2]) == [
        {
            "value": 1.0,
            "count": 1,
            "cumulative_count": 1,
            "cumulative_fraction": 1 / 3,
            "denominator": 3,
        },
        {
            "value": 2.0,
            "count": 2,
            "cumulative_count": 3,
            "cumulative_fraction": 1.0,
            "denominator": 3,
        },
    ]


def test_complete_shot_bootstrap_is_deterministic_and_handles_zero_denominator() -> (
    None
):
    contributions = np.array([[0, 1, 0, 0], [10, 10, 0, 0]], dtype=float)
    first = clustered_bootstrap_ratios(contributions, replicates=128, seed=123)
    second = clustered_bootstrap_ratios(contributions, replicates=128, seed=123)
    assert first == second
    assert first[0]["defined_replicates"] == 128
    assert first[1]["defined_replicates"] == 0
    assert first[1]["undefined_replicates"] == 128
    constant = clustered_bootstrap_ratios(
        np.array([[2, 4], [2, 4], [2, 4]], dtype=float),
        replicates=31,
        seed=7,
    )[0]
    assert constant["lower"] == constant["upper"] == 0.5


def test_missing_workload_telemetry_is_not_reported_as_zero() -> None:
    assert _sum_optional([0, 0], name="work") == 0
    assert _sum_optional([None, None], name="work") is None
    assert _sum_optional([0, None], name="work") is None


def test_loader_analysis_reconciliation_ratio_of_sums_and_first_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    analysis = analyze_policy_audit(corpus)
    assert analysis["tables"]["overview"]["u0_shadow_prediction_disagreements"] == 2
    assert analysis["tables"]["paired_outcomes"]["u0_vs_shadow"]["regressions"] == 1
    assert analysis["tables"]["paired_outcomes"]["u0_vs_shadow"]["recoveries"] == 1
    shadow = analysis["tables"]["event_and_transaction_summary"][0]
    assert shadow["arm"] == "shadow"
    assert shadow["R_event"] == 200 / 500
    first = analysis["tables"]["first_conflict_discordant"]
    assert {
        tuple((row["first_unsafe_stage"], row["first_unsafe_context"], row["count"]))
        for row in first
    } == {
        (1, "yoke", 1),
        (2, "terminal", 1),
    }
    # The context views and their two reconciliation ledgers remain distinct.
    assert set(analysis["tables"]["context_views"]) == {
        "matched_partner_labels",
        "support_path_labels",
        "support_difference_component_labels",
        "omitted_context_labels",
        "degeneracy_diagnostics",
        "exclusive_support_component_context",
    }
    assert all(
        row["stratum_status"] == "insufficient-for-rule-formulation"
        for row in analysis["tables"]["unsafe_fraction_by_stage"]
    )
    assert all(
        "bootstrap" in row for row in analysis["tables"]["certificate_by_domain"]
    )
    assert all("bootstrap" in row for row in analysis["tables"]["first_safe_rank"])
    assert analysis["tables"]["local_competitor_summary"]["availability_fractions"]
    assert "none" in analysis["tables"]["cost_excess_ecdf_by_context"]
    assert analysis["tables"]["unsafe_count_distribution"]
    assert all(row["evidence"] for row in analysis["tables"]["fatal_gates"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].__setitem__("censored", True), "censored"),
        (lambda rows: rows[0].__setitem__("veto_budget", 2), "uncapped"),
        (lambda rows: rows[0].__setitem__("decision", "inspect-only"), "decision"),
        (lambda rows: rows[0].__setitem__("first_safe_rank", 1), "first-safe"),
        (
            lambda rows: (
                rows[1].__setitem__(
                    "proposal_signature", rows[0]["proposal_signature"]
                ),
                rows[1].__setitem__("stage", rows[0]["stage"]),
                rows[1].__setitem__("ordered_endpoints", rows[0]["ordered_endpoints"]),
                rows[1].__setitem__(
                    "canonical_edge_ids", rows[0]["canonical_edge_ids"]
                ),
            ),
            "repeats a proposal signature",
        ),
        (
            lambda rows: rows[1].__setitem__("local_active_state_fingerprint", [99]),
            "changed",
        ),
        (lambda rows: rows[3].__setitem__("stage", 1), "earlier stage"),
        (
            lambda rows: rows[0].__setitem__(
                "terminal_action", rows[1]["terminal_action"]
            ),
            "only on the final",
        ),
        (
            lambda rows: rows[0].__setitem__("exhaustion_kind", "proposal"),
            "claims proposal exhaustion",
        ),
    ],
)
def test_counterfactual_gate_is_recomputed_from_rows(
    tmp_path: Path, mutation, message: str
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    rows = [dict(row) for row in corpus.rows["counterfactuals"]]
    mutation(rows)
    with pytest.raises(PolicyAnalysisError, match=message):
        analyze_policy_audit(_replace_corpus(corpus, kind="counterfactuals", rows=rows))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.__setitem__("matched_partner_labels", ["unknown"]),
            "unknown labels",
        ),
        (
            lambda row: row.__setitem__("omitted_context_labels", []),
            "matched/support-path union",
        ),
        (
            lambda row: row.__setitem__("degeneracy_diagnostics", []),
            "explicit diagnostic flags",
        ),
        (
            lambda row: row["support_difference_components"][0].__setitem__(
                "candidate_support_witness_edge_ids", []
            ),
            "relevance disagrees",
        ),
        (lambda row: row.pop("durable"), "missing semantic field"),
        (lambda row: row.pop("decision"), "missing semantic field"),
    ],
)
def test_proposal_context_and_commit_fields_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    rows = [dict(row) for row in corpus.rows["proposals"]]
    mutation(rows[0])
    with pytest.raises(PolicyAnalysisError, match=message):
        analyze_policy_audit(_replace_corpus(corpus, kind="proposals", rows=rows))


def test_prediction_discordant_both_wrong_is_not_agreement(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    rows = [dict(row) for row in corpus.rows["shots"]]
    target = next(row for row in rows if row["global_shot_id"] == 0)
    target["arm_failures"] = dict(target["arm_failures"])
    for arm_id in ARM_IDS:
        target["arm_failures"][arm_id] = True
    analysis = analyze_policy_audit(_replace_corpus(corpus, kind="shots", rows=rows))
    bin_one = next(
        row
        for row in analysis["tables"]["association_by_unsafe_count"]
        if row["unsafe_count_bin"] == "1"
    )
    assert bin_one["prediction_discordant_both_wrong"] == 1
    assert bin_one["prediction_agreement"] == 0


def test_required_tail_telemetry_is_probe_fatal_but_smoke_explicitly_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    rows = [dict(row) for row in corpus.rows["counterfactuals"]]
    for row in rows:
        row.pop("state_total_stage3_enumeration_wall_ns")
    missing = _replace_corpus(corpus, kind="counterfactuals", rows=rows)
    with pytest.raises(PolicyAnalysisError, match="probe corpus lacks required"):
        analyze_policy_audit(missing)
    smoke = _replace_corpus(missing, manifest={**missing.manifest, "mode": "smoke"})
    analysis = analyze_policy_audit(smoke)
    assert (
        analysis["tables"]["veto_chain_tails"]["stage3_enumeration_ns_per_state"][
            "telemetry_status"
        ]
        == "incomplete-smoke"
    )


def test_gate7_is_missing_without_exact_authenticated_collector_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    manifest = dict(corpus.manifest)
    manifest.pop("fatal_gate_attestations")
    unattested = _replace_corpus(corpus, manifest=manifest)
    analysis = analyze_policy_audit(unattested)
    gate7 = analysis["tables"]["fatal_gates"][6]
    assert gate7["gate"] == 7
    assert gate7["status"] == "missing-collector-attestation"
    assert gate7["evidence"]["kind"] == "missing"
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with pytest.raises(
        PolicyAnalysisError, match="fatal gate 3 status is not a passing status"
    ):
        write_policy_analysis(unattested, analysis, render_plots=False)


def test_analysis_is_input_order_invariant_and_uses_trajectory_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    corpus = load_policy_audit(root)
    baseline = analyze_policy_audit(corpus)
    reversed_corpus = PolicyAuditCorpus(
        root=corpus.root,
        experiment=corpus.experiment,
        config=corpus.config,
        manifest=corpus.manifest,
        rows={name: tuple(reversed(rows)) for name, rows in corpus.rows.items()},
        source_hashes=corpus.source_hashes,
    )
    reordered = analyze_policy_audit(reversed_corpus)
    assert reordered["analysis_sha256"] == baseline["analysis_sha256"]
    assert (
        reordered["tables"]["first_conflict_discordant"]
        == baseline["tables"]["first_conflict_discordant"]
    )

    exhaustive = dict(corpus.rows["counterfactuals"][0])
    exhaustive["trajectory_origin"] = "casebook-exhaustive"
    with_casebook = PolicyAuditCorpus(
        root=corpus.root,
        experiment=corpus.experiment,
        config=corpus.config,
        manifest=corpus.manifest,
        rows={
            **corpus.rows,
            "counterfactuals": (*corpus.rows["counterfactuals"], exhaustive),
        },
        source_hashes=corpus.source_hashes,
    )
    assert (
        analyze_policy_audit(with_casebook)["analysis_sha256"]
        == baseline["analysis_sha256"]
    )


def test_human_report_is_deterministic_separates_visibility_and_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    analysis = analyze_policy_audit(load_policy_audit(root))
    first = policy_human_report_bytes(analysis)
    assert policy_human_report_bytes(analysis) == first
    assert first.endswith(b"\n")
    text = first.decode("utf-8")
    assert "## Locally observable policy clues" in text
    assert "## Nonlocal and oracle-only explanations" in text
    assert "hypothesis-generating, not causal proof" in text
    assert "fewer than 100 unsafe states" in text
    assert "Tied-support diagnostic" in text
    assert "Logical-error denominators are physical shots" in text

    malformed = json.loads(json.dumps(analysis))
    malformed["tables"]["visibility_summary"] = malformed["tables"][
        "visibility_summary"
    ][:-1]
    with pytest.raises(PolicyAnalysisError, match="exact visibility taxonomy"):
        policy_human_report_bytes(malformed)


def test_casebook_selection_is_deterministic_outcome_blind_and_sparse_flagged() -> None:
    states = [
        {
            "state_id": f"{index:064x}",
            "exclusive_context": "yoke",
            "original_stage": 1,
            "original_cost_excess": float(index),
            "terminal_action": "same-stage-alternative",
            "veto_chain_length": index + 1,
            "original_proposal_sha256": f"{index:064x}",
            "actual_observables": "ignored",
            "logical_failure": bool(index % 2),
        }
        for index in range(20)
    ]
    first = select_casebook(states)
    changed = [
        dict(
            state,
            actual_observables="changed",
            logical_failure=not state["logical_failure"],
        )
        for state in states
    ]
    assert select_casebook(changed) == first
    assert first["states"][0]["state_id"] == f"{9:064x}"
    assert len(first["states"]) in {1, 2}
    sparse = select_casebook(states[:19])
    assert sparse["selection_audit"][0]["selected_state_id"] is None


def test_loader_rejects_noncanonical_or_digest_mismatched_shard(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    path = root / "shards/worker-00/shots.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"damage")
    with pytest.raises(PolicyAnalysisError, match="gzip ledger|digest mismatch"):
        load_policy_audit(root)


def test_loader_authenticates_collection_ready(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    ready = json.loads((root / "COLLECTION_READY").read_text())
    ready["manifest_sha256"] = "00" * 32
    (root / "COLLECTION_READY").write_bytes(canonical_json_bytes(ready) + b"\n")
    with pytest.raises(PolicyAnalysisError, match="COLLECTION_READY manifest digest"):
        load_policy_audit(root)


def test_atomic_writer_emits_tables_plot_data_and_analysis_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    corpus = load_policy_audit(root)
    analysis = analyze_policy_audit(corpus)
    manifest = write_policy_analysis(corpus, analysis, render_plots=True)
    assert (root / "ANALYSIS_READY").is_file()
    assert not (root / "COMPLETE").exists()
    assert (root / "analysis/summary.json").is_file()
    report_path = root / "analysis/report.md"
    assert report_path.is_file()
    report_bytes = policy_human_report_bytes(analysis)
    assert report_path.read_bytes() == report_bytes
    assert (root / "casebook/selection.json").is_file()
    assert len(manifest["plot_images"]) == 11
    assert all((root / "analysis" / path).is_file() for path in manifest["plot_images"])
    assert manifest["plot_images_scientifically_digested"] is False
    assert manifest["table_file_hashes"]
    assert manifest["plot_data_file_hashes"]
    assert manifest["report_file_sha256"] == _digest(report_bytes)
    ready = json.loads((root / "ANALYSIS_READY").read_text())
    assert ready["report_file_sha256"] == manifest["report_file_sha256"]
    # Re-analysis is supported and bit-for-bit verified without rewriting.
    loaded_again = load_policy_audit(root)
    recomputed = analyze_policy_audit(loaded_again)
    assert verify_existing_policy_analysis(loaded_again, recomputed) == manifest
    report_path.write_bytes(report_bytes + b"tamper")
    with pytest.raises(PolicyAnalysisError, match="human report bytes"):
        verify_existing_policy_analysis(loaded_again, recomputed)
    report_path.write_bytes(report_bytes)
    (root / "COMPLETE").write_text("{}\n", encoding="utf-8")
    # A later finalizer may add COMPLETE; ANALYSIS_READY remains the analysis
    # authority and COMPLETE is neither rejected nor trusted as a substitute.
    assert verify_existing_policy_analysis(loaded_again, recomputed) == manifest
    # A self-consistent rewrite of an installed table plus its manifest hashes
    # is still rejected because verification reconstructs the expected bytes.
    table_path = root / "analysis/tables/fatal_gates.json"
    table_path.write_bytes(b"{}\n")
    installed_manifest = json.loads((root / "analysis/manifest.json").read_text())
    installed_manifest["table_file_hashes"]["tables/fatal_gates.json"] = _digest(
        table_path.read_bytes()
    )
    (root / "analysis/manifest.json").write_bytes(
        canonical_json_bytes(installed_manifest) + b"\n"
    )
    ready = json.loads((root / "ANALYSIS_READY").read_text())
    ready["analysis_manifest_sha256"] = _digest(
        (root / "analysis/manifest.json").read_bytes()
    )
    (root / "ANALYSIS_READY").write_bytes(canonical_json_bytes(ready) + b"\n")
    with pytest.raises(PolicyAnalysisError, match="recomputed artifacts"):
        verify_existing_policy_analysis(loaded_again, recomputed)

    (root / "ANALYSIS_READY").unlink()
    with pytest.raises(
        PolicyAnalysisError, match="COMPLETE is not an analysis substitute"
    ):
        verify_existing_policy_analysis(loaded_again, recomputed)
    assert load_policy_audit(root).source_hashes == corpus.source_hashes
