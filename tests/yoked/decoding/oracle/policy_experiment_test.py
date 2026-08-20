from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import REPO_ROOT
import yoked.decoding.oracle.policy_experiment as policy_experiment
from yoked.decoding._promatch_experiment import PreparedCell
from yoked.decoding.oracle.policy_analysis import (
    COLLECTOR_GATE_ATTESTATION_SCHEMA,
    COLLECTOR_GATE_CHECKS,
)
from yoked.decoding.oracle.policy_experiment import (
    ANALYSIS_PLOT_NAMES,
    ANALYSIS_TABLE_NAMES,
    ARM_IDS,
    SCIENTIFIC_WORKERS,
    WorkerSpec,
    _audit_policy_shot,
    _validate_context_union_ledger,
    _validate_support_difference_ledger,
    _worker_task,
    _worker_tail_censor_attestation,
    attest_completed_policy_probe,
    collect_policy_worker_shard,
    default_policy_audit_draft,
    derive_policy_worker_seed,
    deterministic_gzip,
    install_worker_shard,
    policy_config_self_sha256,
    policy_experiment_id,
    policy_worker_schedule,
    run_policy_collection,
    validate_policy_protocol,
    verify_worker_shard,
)
from yoked.decoding.oracle.full_graph import OracleTolerance


def test_tail_censor_attestation_canonicalizes_structured_proposal_signatures():
    rows = [
        {
            "original_state_sha256": "state-a",
            "censored": False,
            "veto_budget": None,
            "proposal_signature": [1, [2, None], {"stage": 3}],
        },
        {
            "original_state_sha256": "state-a",
            "censored": False,
            "veto_budget": None,
            "proposal_signature": [1, [2, None], {"stage": 3}],
        },
        {
            "original_state_sha256": "state-b",
            "censored": False,
            "veto_budget": None,
            "proposal_signature": [1, [2, None], {"stage": 3}],
        },
    ]
    assert _worker_tail_censor_attestation(rows) == {
        "uncapped_counterfactuals": True,
        "censored_states": 0,
        "repeated_same_state_proposal_signatures": 1,
        "output_truncations": 0,
    }


class _FakeSampler:
    def __init__(self, *, observable_byte: int) -> None:
        self.observable_byte = observable_byte

    def sample(self, *, shots: int, separate_observables: bool, bit_packed: bool):
        assert separate_observables and bit_packed
        dets = np.arange(shots, dtype=np.uint8).reshape(shots, 1)
        obs = np.full((shots, 1), self.observable_byte, dtype=np.uint8)
        return dets, obs


class _FakeCircuit:
    def __init__(self, *, observable_byte: int = 0) -> None:
        self.observable_byte = observable_byte

    def compile_detector_sampler(self, *, seed: int):
        assert isinstance(seed, int)
        return _FakeSampler(observable_byte=self.observable_byte)


class _FakeMatcher:
    def decode_batch(
        self, dets, *, bit_packed_shots: bool, bit_packed_predictions: bool
    ):
        assert bit_packed_shots and bit_packed_predictions
        return np.zeros((len(dets), 1), dtype=np.uint8)


def _runtime_config() -> dict:
    result = default_policy_audit_draft()
    result["cell"].update(
        {
            "circuit_sha256": "01" * 32,
            "dem_sha256": "02" * 32,
            "layout_fingerprint": "03" * 32,
            "graph_fingerprint": "04" * 32,
            "num_detectors": 3,
            "num_observables": 2,
            "undecomposed_dem_sha256": "05" * 32,
            "matcher_edge_table_sha256": "06" * 32,
            "domain_graphs_sha256": "07" * 32,
        }
    )
    result["experiment_id"] = policy_experiment_id(result)
    result["config_self_sha256"] = policy_config_self_sha256(result)
    return result


def _prepared(*, observable_byte: int = 0) -> PreparedCell:
    graph = SimpleNamespace(
        num_detectors=3,
        num_observables=2,
        matcher=_FakeMatcher(),
        fingerprint="04" * 32,
    )
    return PreparedCell(
        cell=_runtime_config()["cell"],
        circuit=_FakeCircuit(observable_byte=observable_byte),
        dem=SimpleNamespace(num_detectors=3, num_observables=2),
        compiled_pu=SimpleNamespace(graph=graph),
        provenance={},
    )


def _audit(graph, syndrome, *, tolerance):
    assert syndrome.shape == (3,)
    assert isinstance(tolerance, OracleTolerance)
    return {
        "arm_predictions": {arm_id: b"\x00" for arm_id in ARM_IDS},
        "shot": {"original_detector_hw": int(np.count_nonzero(syndrome))},
        "proposals": [
            {
                "arm_id": ARM_IDS[1],
                "graph_fingerprint": graph.fingerprint,
                "observable_frame": "00",
                "decision_weight": 1.25,
                "decision_weight_hex": (1.25).hex(),
                "support_difference_representation_version": "promatch-support-difference-v2",
                "support_difference_components": [],
                "detector_boundary_ids": [],
                "support_difference_component_labels": [],
                "support_cancellation_edge_ids": [],
                "disconnected_support_reconfiguration": False,
                "exclusive_support_component_context": None,
                "matched_partner_labels": [],
                "support_path_labels": [],
                "omitted_context_labels": [],
                "degeneracy_diagnostics": [],
                "cost_compatible": True,
                "frame_compatible": True,
                "oracle_policy_accepts": True,
                "B_base_support_edge_ids": [],
                "P_candidate_support_edge_ids": [],
                "R_residual_support_edge_ids": [],
                "base_support_edge_ids": [],
                "candidate_support_edge_ids": [],
                "residual_support_edge_ids": [],
                "Q_forced_parity_support_edge_ids": [],
                "X_support_difference_edge_ids": [],
                "P_intersection_R_edge_ids": [],
                "supports_square_free": True,
                "B_base_support_square_free": True,
                "P_candidate_support_square_free": True,
                "R_residual_support_square_free": True,
                "Q_forced_parity_support_square_free": True,
                "X_support_difference_square_free": True,
            }
        ],
        "counterfactuals": [],
        "domains": [{"arm_id": ARM_IDS[1], "status": "success"}],
    }


def _shot_rows(payloads: dict[str, bytes]) -> list[dict]:
    raw = gzip.decompress(payloads["shots.jsonl.gz"])
    return [json.loads(line) for line in raw.splitlines()]


def _rewrite_first_ledger_row(payloads: dict[str, bytes], name: str, mutate) -> None:
    filename = f"{name}.jsonl.gz"
    rows = [
        json.loads(line) for line in gzip.decompress(payloads[filename]).splitlines()
    ]
    mutate(rows[0])
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    compressed = deterministic_gzip(raw)
    payloads[filename] = compressed
    shard = json.loads(payloads["shard.json"])
    metadata = shard["artifacts"][filename]
    metadata.update(
        {
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "compressed_bytes": len(compressed),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "uncompressed_bytes": len(raw),
            "rows": len(rows),
        }
    )
    payloads["shard.json"] = (
        json.dumps(shard, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fake_completed_probe(
    root: Path,
    config: dict,
    *,
    commit: str = "a" * 40,
    gates_passed: bool = True,
    observed_pids: int = 32,
    plots_rendered: bool = True,
) -> list[dict]:
    experiment_id = config["experiment_id"]
    experiment_hash = _write_json(
        root / "experiment.json",
        {
            "schema": "promatch-l1-policy-audit-experiment-v1",
            "experiment_id": experiment_id,
            "mode": "probe",
            "implementation_commit": config.get("implementation_commit"),
            "config_commit": commit,
        },
    )
    config_hash = _write_json(root / "config.json", config)
    gate_checks = {
        check for checks in COLLECTOR_GATE_CHECKS.values() for check in checks
    }
    shards = [
        {
            "worker_id": spec.worker_id,
            "collector_gate_evidence": {
                "real_graph": True,
                "checks": {
                    check: {"status": "passed", "observations": 1}
                    for check in gate_checks
                },
            },
        }
        for spec in policy_worker_schedule("probe")
    ]
    compressed_probe_bytes = 0 if gates_passed else (20 * 1024**3) // 300 + 1
    projected_artifact_bytes = 300 * compressed_probe_bytes
    zero = 0.0
    manifest = {
        "schema": "promatch-l1-policy-audit-manifest-v1",
        "experiment_id": experiment_id,
        "mode": "probe",
        "workers": 32,
        "shots": 100,
        "new_worker_processes_observed": observed_pids,
        "shards": shards,
        "probe_projection": {
            "parent_setup_seconds": zero,
            "parent_setup_seconds_hex": zero.hex(),
            "parallel_worker_compile_seconds": zero,
            "parallel_worker_compile_seconds_hex": zero.hex(),
            "fixed_setup_seconds": zero,
            "fixed_setup_seconds_hex": zero.hex(),
            "variable_100_shot_seconds": zero,
            "variable_100_shot_seconds_hex": zero.hex(),
            "compressed_probe_bytes": compressed_probe_bytes,
            "projected_wall_seconds": zero,
            "projected_wall_seconds_hex": zero.hex(),
            "projected_artifact_bytes": projected_artifact_bytes,
            "free_output_bytes": 100 * 1024**3,
            "wall_gate_passed": True,
            "artifact_gate_passed": gates_passed,
            "free_space_gate_passed": True,
            "all_launch_gates_passed": gates_passed,
        },
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
        "performance_telemetry": {
            "schema": "promatch-l1-policy-audit-campaign-performance-v1",
            "parent_setup_ns": 0,
            "worker_phase_ns": 0,
            "parent_peak_rss_bytes": 0,
            "parent_peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib",
            "new_worker_compile_ns": [],
            "scientifically_deterministic": False,
            "excluded_from_scientific_decisions": True,
        },
        "campaign_wall_ns": 1,
    }
    manifest_hash = _write_json(root / "manifest.json", manifest)
    collection_hash = _write_json(
        root / "COLLECTION_READY",
        {
            "schema": "promatch-l1-policy-audit-collection-ready-v1",
            "experiment_id": experiment_id,
            "mode": "probe",
            "manifest_sha256": manifest_hash,
            "verified_worker_shards": 32,
            "verified_shots": 100,
        },
    )
    ledger_hashes = {}
    for spec in policy_worker_schedule("probe"):
        worker = root / "shards" / f"worker-{spec.worker_id:02d}"
        worker.mkdir(parents=True)
        for name in ("shots", "proposals", "counterfactuals", "domains"):
            relative = f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz"
            (root / relative).write_bytes(b"fake-ledger")
            ledger_hashes[relative] = hashlib.sha256(b"fake-ledger").hexdigest()
        timing_relative = f"shards/worker-{spec.worker_id:02d}/timing.json"
        (root / timing_relative).write_bytes(b"fake-timing")
        ledger_hashes[timing_relative] = hashlib.sha256(b"fake-timing").hexdigest()
    selection_hash = _write_json(root / "casebook" / "selection.json", {})
    source_hashes = {
        "experiment.json": experiment_hash,
        "config.json": config_hash,
        "manifest.json": manifest_hash,
        "COLLECTION_READY": collection_hash,
        **ledger_hashes,
    }
    tables = {name: {} for name in ANALYSIS_TABLE_NAMES}
    tables["fatal_gates"] = [
        {"gate": gate, "status": "passed-ledger-recomputed"} for gate in range(1, 19)
    ]
    summary_without_digest = {
        "schema": "promatch-l1-policy-audit-analysis-v1",
        "experiment_id": experiment_id,
        "cell_id": config["cell"]["cell_id"],
        "analysis_contract": {
            "source": "immutable-canonical-gzip-jsonl-only",
            "sampling_or_decoding_reconstruction": False,
            "bootstrap_unit": "complete-physical-shot",
            "bootstrap_quantile": "empirical-type-7",
            "proposal_bootstrap_replicates": 10_000,
            "workload_bootstrap_replicates": 10_000,
            "casebook_outcome_blind": True,
            "casebook_exhaustive_rows_excluded": True,
            "support_context_views_kept_distinct": True,
            "required_tail_telemetry": "complete",
            "complete_written_by_analyzer": False,
            "complete_accepted_as_analysis_substitute": False,
            "next_required_stage": "casebook-expansion-and-finalization-external",
        },
        "source_hashes": source_hashes,
        "casebook_selection": {},
        "tables": tables,
    }
    analysis_sha = hashlib.sha256(
        json.dumps(
            summary_without_digest,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary = {**summary_without_digest, "analysis_sha256": analysis_sha}
    summary_hash = _write_json(root / "analysis" / "summary.json", summary)
    report_bytes = b"# Authenticated policy-audit report\n"
    (root / "analysis" / "report.md").write_bytes(report_bytes)
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    table_hashes = {
        f"tables/{name}.json": _write_json(
            root / "analysis" / "tables" / f"{name}.json", tables[name]
        )
        for name in ANALYSIS_TABLE_NAMES
    }
    plot_data_hashes = {
        f"plot-data/{name}.json": _write_json(
            root / "analysis" / "plot-data" / f"{name}.json", {}
        )
        for name in ANALYSIS_PLOT_NAMES
    }
    plot_images = []
    for name in sorted(ANALYSIS_PLOT_NAMES):
        relative = f"plots/{name}.png"
        plot_path = root / "analysis" / relative
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"fake-png")
        plot_images.append(relative)
    analysis_manifest_hash = _write_json(
        root / "analysis" / "manifest.json",
        {
            "schema": "promatch-l1-policy-audit-analysis-manifest-v1",
            "experiment_id": experiment_id,
            "analysis_sha256": analysis_sha,
            "summary_file_sha256": summary_hash,
            "report_file_sha256": report_hash,
            "source_hashes": source_hashes,
            "table_file_hashes": table_hashes,
            "plot_data_file_hashes": plot_data_hashes,
            "plot_images": plot_images,
            "plot_images_scientifically_digested": False,
        },
    )
    _write_json(
        root / "ANALYSIS_READY",
        {
            "schema": "promatch-l1-policy-audit-analysis-ready-v1",
            "experiment_id": experiment_id,
            "analysis_manifest_sha256": analysis_manifest_hash,
            "casebook_selection_sha256": selection_hash,
            "report_file_sha256": report_hash,
            "plots_rendered": plots_rendered,
            "casebook_exhaustive_expansion_required_before_complete": True,
        },
    )
    return shards


def _fake_probe_attestation(commit: str) -> dict:
    return {
        "schema": "promatch-l1-policy-audit-probe-attestation-v1",
        "probe_experiment_id": "1" * 64,
        "implementation_commit": commit,
        "probe_config_self_sha256": "2" * 64,
        "probe_experiment_sha256": "3" * 64,
        "probe_config_sha256": "4" * 64,
        "probe_manifest_sha256": "5" * 64,
        "collection_ready_sha256": "6" * 64,
        "analysis_ready_sha256": "7" * 64,
        "analysis_manifest_sha256": "8" * 64,
        "analysis_summary_sha256": "9" * 64,
        "casebook_selection_sha256": "a" * 64,
        "verified_workers": 32,
        "verified_shots": 100,
        "all_launch_gates_passed": True,
    }


def test_exact_worker_schedules() -> None:
    scientific = policy_worker_schedule("scientific")
    assert len(scientific) == SCIENTIFIC_WORKERS
    assert all(row.shots == 625 for row in scientific)
    assert scientific[0] == WorkerSpec(0, 0, 625)
    assert scientific[-1] == WorkerSpec(31, 19_375, 625)
    assert sum(row.shots for row in policy_worker_schedule("smoke")) == 32
    probe = policy_worker_schedule("probe")
    assert [row.shots for row in probe[:5]] == [4, 4, 4, 4, 3]
    assert sum(row.shots for row in probe) == 100


def test_seed_derivation_is_deterministic_and_worker_separated() -> None:
    config = _runtime_config()
    kwargs = {
        "seed_root": config["sampling"]["seed_roots"]["scientific"],
        "experiment_id": config["experiment_id"],
        "cell_id": config["cell"]["cell_id"],
    }
    assert derive_policy_worker_seed(
        **kwargs, worker_id=0
    ) == derive_policy_worker_seed(**kwargs, worker_id=0)
    assert derive_policy_worker_seed(
        **kwargs, worker_id=0
    ) != derive_policy_worker_seed(**kwargs, worker_id=1)


def test_draft_enforces_fixed_cell_20k_and_exact_arms() -> None:
    draft = default_policy_audit_draft()
    validate_policy_protocol(draft, scientific=False)
    changed = copy.deepcopy(draft)
    changed["sampling"]["total_shots"] = 19_999
    with pytest.raises(ValueError, match="20000=32x625"):
        validate_policy_protocol(changed, scientific=False)
    changed = copy.deepcopy(draft)
    changed["cell"]["p"] = 0.001
    with pytest.raises(ValueError, match="exact fixed"):
        validate_policy_protocol(changed, scientific=False)
    changed = copy.deepcopy(draft)
    changed["arms"].reverse()
    with pytest.raises(ValueError, match="five-arm"):
        validate_policy_protocol(changed, scientific=False)


def test_current_source_scope_includes_the_pytest_configuration() -> None:
    root = REPO_ROOT
    assert "pytest.ini" in policy_experiment.POLICY_SOURCE_PATHS
    assert all(
        (root / path).is_file() for path in policy_experiment.POLICY_SOURCE_PATHS
    )


def test_historical_frozen_protocol_remains_readable_after_source_move() -> None:
    root = REPO_ROOT
    config = json.loads(
        (root / "docs" / "PROMATCH_POLICY_AUDIT_20K_FROZEN_V2.json").read_text()
    )
    assert validate_policy_protocol(config, scientific=False) == config["experiment_id"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["arms"][2].update(policy="wrong"), "five-arm"),
        (lambda value: value["counterfactual"].update(stop="wrong"), "counterfactual"),
        (
            lambda value: value["context_taxonomy"].update(
                version="promatch-support-context-v1"
            ),
            "context taxonomy",
        ),
        (
            lambda value: value["report_contract"]["human_report"].update(
                format="wrong"
            ),
            "table/plot/binning contract",
        ),
        (lambda value: value.update(source_paths=["requirements.txt"]), "source_paths"),
        (
            lambda value: value.update(unvalidated_semantics=True),
            "fields must be exactly",
        ),
    ],
)
def test_protocol_rejects_unfrozen_or_unscoped_semantic_changes(
    mutation, message: str
) -> None:
    config = default_policy_audit_draft()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_policy_protocol(config, scientific=False)


def test_scientific_rejects_max_errors_before_repository_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config()
    config.update(
        {
            "status": "FROZEN",
            "frozen": True,
            "implementation_commit": "0" * 40,
            "config_commit": "verified-runtime-head",
            "protocol_relative_path": "docs/PROMATCH_POLICY_AUDIT_FROZEN_V1.json",
        }
    )
    config["probe_attestation"] = _fake_probe_attestation("0" * 40)
    config["software_versions"] = {}
    config["execution_environment"] = {}
    config["source_hashes"] = {
        relative: "b" * 64 for relative in config["source_paths"]
    }
    config["requirements_sha256"] = "b" * 64
    config["experiment_id"] = policy_experiment_id(config)
    config["config_self_sha256"] = policy_config_self_sha256(config)
    monkeypatch.setenv("MAX_ERRORS", "1")
    with pytest.raises(ValueError, match="MAX_ERRORS"):
        validate_policy_protocol(
            config,
            scientific=True,
            protocol_path=Path("docs/PROMATCH_POLICY_AUDIT_FROZEN_V1.json"),
        )


def test_adapter_allows_observable_frame_but_rejects_actual_observables() -> None:
    graph = _prepared().compiled_pu.graph
    result = _audit_policy_shot(
        graph,
        np.zeros(3, dtype=np.uint8),
        tolerance=OracleTolerance(),
        audit_fn=_audit,
    )
    assert result.proposals[0]["observable_frame"] == "00"

    def leaking(graph, syndrome, *, tolerance):
        value = _audit(graph, syndrome, tolerance=tolerance)
        value["shot"]["packed_actual_observables_hex"] = "00"
        return value

    with pytest.raises(ValueError, match="ground-truth-like"):
        _audit_policy_shot(
            graph,
            np.zeros(3, dtype=np.uint8),
            tolerance=OracleTolerance(),
            audit_fn=leaking,
        )


def test_adapter_requires_exact_hex_companion_for_float() -> None:
    graph = _prepared().compiled_pu.graph

    def missing_hex(graph, syndrome, *, tolerance):
        value = _audit(graph, syndrome, tolerance=tolerance)
        value["proposals"][0].pop("decision_weight_hex")
        return value

    with pytest.raises(ValueError, match=r"exact \*_hex"):
        _audit_policy_shot(
            graph,
            np.zeros(3, dtype=np.uint8),
            tolerance=OracleTolerance(),
            audit_fn=missing_hex,
        )


def test_core_cannot_override_collector_owned_identity() -> None:
    config = _runtime_config()

    def overriding(graph, syndrome, *, tolerance):
        value = _audit(graph, syndrome, tolerance=tolerance)
        value["proposals"][0]["global_shot_id"] = 999
        return value

    with pytest.raises(ValueError, match="collector-owned"):
        collect_policy_worker_shard(
            _prepared(),
            config=config,
            mode="smoke",
            spec=WorkerSpec(0, 0, 1),
            audit_fn=overriding,
        )


def test_deterministic_gzip_has_reproducible_bytes_and_zero_mtime() -> None:
    data = b'{"a":1}\n'
    first = deterministic_gzip(data)
    second = deterministic_gzip(data)
    assert first == second
    assert gzip.decompress(first) == data
    assert first[4:8] == b"\x00\x00\x00\x00"


def test_worker_samples_once_and_stores_replay_complete_packed_payloads() -> None:
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 2)
    _, payloads = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=_audit
    )
    rows = _shot_rows(payloads)
    assert [row["global_shot_id"] for row in rows] == [0, 1]
    assert rows[0]["packed_detectors_hex"] == "00"
    assert rows[1]["packed_detectors_hex"] == "01"
    assert rows[0]["packed_actual_observables_hex"] == "00"
    assert rows[0]["packed_detector_bits"] == 3
    assert rows[0]["packed_actual_observable_bits"] == 2
    assert len(rows[0]["packed_detectors_sha256"]) == 64
    assert len(rows[0]["packed_actual_observables_sha256"]) == 64


def test_detector_only_identity_does_not_change_with_actual_observables() -> None:
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 1)
    _, zero_payloads = collect_policy_worker_shard(
        _prepared(observable_byte=0),
        config=config,
        mode="smoke",
        spec=spec,
        audit_fn=_audit,
    )
    _, one_payloads = collect_policy_worker_shard(
        _prepared(observable_byte=1),
        config=config,
        mode="smoke",
        spec=spec,
        audit_fn=_audit,
    )
    zero = _shot_rows(zero_payloads)[0]
    one = _shot_rows(one_payloads)[0]
    assert zero["physical_input_sha256"] == one["physical_input_sha256"]
    assert zero["detector_input_sha256"] == one["detector_input_sha256"]
    assert (
        zero["packed_actual_observables_sha256"]
        != one["packed_actual_observables_sha256"]
    )


def test_atomic_shard_install_verify_and_corruption_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 1)
    shard, payloads = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=_audit
    )
    verification_paths: list[Path] = []
    real_verify = policy_experiment.verify_worker_shard

    def recording_verify(path: Path, *, config, mode, spec):
        result = real_verify(path, config=config, mode=mode, spec=spec)
        verification_paths.append(path.resolve())
        return result

    monkeypatch.setattr(policy_experiment, "verify_worker_shard", recording_verify)
    out = tmp_path / "out"
    installed = install_worker_shard(
        out,
        shard=shard,
        payloads=payloads,
        config=config,
        mode="smoke",
        spec=spec,
    )
    assert len(verification_paths) == 2
    assert verification_paths[0] != installed.resolve()
    assert verification_paths[1] == installed.resolve()
    assert (
        verify_worker_shard(installed, config=config, mode="smoke", spec=spec) == shard
    )
    with (installed / "shots.jsonl.gz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="invalid compressed|digest/count"):
        verify_worker_shard(installed, config=config, mode="smoke", spec=spec)


def test_fresh_shard_is_verified_before_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 1)
    shard, payloads = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=_audit
    )
    _rewrite_first_ledger_row(
        payloads, "shots", lambda row: row["arm_predictions_hex"].pop(ARM_IDS[-1])
    )
    with pytest.raises(ValueError, match="prediction arm set"):
        install_worker_shard(
            tmp_path / "invalid",
            shard=shard,
            payloads=payloads,
            config=config,
            mode="smoke",
            spec=spec,
        )
    assert not (tmp_path / "invalid" / "shards" / "worker-00").exists()


def test_fresh_shard_rejects_tampered_v2_support_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 1)
    shard, payloads = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=_audit
    )
    _rewrite_first_ledger_row(
        payloads,
        "proposals",
        lambda row: row.__setitem__("support_difference_representation_version", "v1"),
    )
    shard = json.loads(payloads["shard.json"])
    with pytest.raises(ValueError, match="exact v2 support-difference"):
        install_worker_shard(
            tmp_path / "invalid-support",
            shard=shard,
            payloads=payloads,
            config=config,
            mode="smoke",
            spec=spec,
        )
    assert not (tmp_path / "invalid-support" / "shards" / "worker-00").exists()


def test_fresh_core_rejects_component_detector_witness_tamper() -> None:
    graph = SimpleNamespace(
        fingerprint="04" * 32,
        edges=(SimpleNamespace(source=0, target=1),),
    )
    row = _audit(graph, np.zeros(3, dtype=np.uint8), tolerance=OracleTolerance())[
        "proposals"
    ][0]
    row.update(
        {
            "detector_boundary_ids": [0],
            "base_support_edge_ids": [],
            "candidate_support_edge_ids": [0],
            "residual_support_edge_ids": [],
            "B_base_support_edge_ids": [],
            "P_candidate_support_edge_ids": [0],
            "R_residual_support_edge_ids": [],
            "Q_forced_parity_support_edge_ids": [0],
            "X_support_difference_edge_ids": [0],
            "P_intersection_R_edge_ids": [],
            "support_difference_component_labels": ["in-domain"],
            "support_difference_components": [
                {
                    "certificate_kind": "real-x-component",
                    "canonical_edge_ids": [0],
                    "support_cancellation_edge_ids": [],
                    "component_detector_ids": [2],
                    "candidate_support_witness_edge_ids": [0],
                    "candidate_boundary_witness_detector_ids": [],
                    "labels": ["in-domain"],
                    "candidate_relevant": True,
                    "candidate_relevance_reasons": ["candidate-support-edge"],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="detector witness disagrees with graph"):
        _validate_support_difference_ledger(row, path="test", graph=graph)


def test_candidate_component_union_normalizes_in_domain_across_components() -> None:
    graph = SimpleNamespace(
        fingerprint="04" * 32,
        edges=(
            SimpleNamespace(source=0, target=1),
            SimpleNamespace(source=2, target=3),
        ),
    )
    row = _audit(graph, np.zeros(3, dtype=np.uint8), tolerance=OracleTolerance())[
        "proposals"
    ][0]
    row.update(
        {
            "detector_boundary_ids": [0, 2],
            "base_support_edge_ids": [],
            "candidate_support_edge_ids": [0, 1],
            "residual_support_edge_ids": [],
            "B_base_support_edge_ids": [],
            "P_candidate_support_edge_ids": [0, 1],
            "R_residual_support_edge_ids": [],
            "Q_forced_parity_support_edge_ids": [0, 1],
            "X_support_difference_edge_ids": [0, 1],
            "P_intersection_R_edge_ids": [],
            "support_difference_component_labels": ["yoke"],
            "exclusive_support_component_context": "yoke",
            "support_difference_components": [
                {
                    "certificate_kind": "real-x-component",
                    "canonical_edge_ids": [0],
                    "support_cancellation_edge_ids": [],
                    "component_detector_ids": [0, 1],
                    "candidate_support_witness_edge_ids": [0],
                    "candidate_boundary_witness_detector_ids": [0],
                    "labels": ["in-domain"],
                    "candidate_relevant": True,
                    "candidate_relevance_reasons": [
                        "candidate-boundary-detector",
                        "candidate-support-edge",
                    ],
                },
                {
                    "certificate_kind": "real-x-component",
                    "canonical_edge_ids": [1],
                    "support_cancellation_edge_ids": [],
                    "component_detector_ids": [2, 3],
                    "candidate_support_witness_edge_ids": [1],
                    "candidate_boundary_witness_detector_ids": [2],
                    "labels": ["yoke"],
                    "candidate_relevant": True,
                    "candidate_relevance_reasons": [
                        "candidate-boundary-detector",
                        "candidate-support-edge",
                    ],
                },
            ],
        }
    )
    _validate_support_difference_ledger(row, path="test", graph=graph)
    row["support_difference_component_labels"] = ["in-domain", "yoke"]
    with pytest.raises(ValueError, match="candidate-context labels"):
        _validate_support_difference_ledger(row, path="test", graph=graph)


def test_context_union_normalizes_in_domain_before_reconciliation() -> None:
    row = {
        "matched_partner_labels": ["in-domain"],
        "support_path_labels": ["cross-patch-or-basis", "cross-window", "yoke"],
        "omitted_context_labels": ["cross-patch-or-basis", "cross-window", "yoke"],
    }
    _validate_context_union_ledger(row, path="test")
    row["omitted_context_labels"] = [
        "cross-patch-or-basis",
        "cross-window",
        "in-domain",
        "yoke",
    ]
    with pytest.raises(ValueError, match="context label group|normalized"):
        _validate_context_union_ledger(row, path="test")


def test_fresh_shard_rejects_tampered_omitted_context_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 1)
    shard, payloads = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=_audit
    )
    _rewrite_first_ledger_row(
        payloads,
        "proposals",
        lambda row: row.__setitem__("omitted_context_labels", ["in-domain"]),
    )
    shard = json.loads(payloads["shard.json"])
    with pytest.raises(ValueError, match="omitted_context_labels"):
        install_worker_shard(
            tmp_path / "invalid-context",
            shard=shard,
            payloads=payloads,
            config=config,
            mode="smoke",
            spec=spec,
        )


def test_scientific_ledgers_are_bit_exact_while_timing_is_excluded() -> None:
    config = _runtime_config()
    spec = WorkerSpec(0, 0, 2)

    def timed_audit(wall_ns: int):
        def run(graph, syndrome, *, tolerance):
            value = _audit(graph, syndrome, tolerance=tolerance)
            value["shot"]["matched_active_pair_backend_wall_ns"] = wall_ns
            value["shot"]["timing_telemetry"] = {
                "counterfactual_wall_ns": wall_ns,
                "support_classification_wall_ns": wall_ns,
                "stage3_specific_wall_ns": wall_ns,
            }
            value["proposals"][0]["candidate_enumeration_wall_ns"] = wall_ns
            value["proposals"][0]["stage3_enumeration_wall_ns"] = wall_ns
            return value

        return run

    _, first = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=timed_audit(11)
    )
    _, second = collect_policy_worker_shard(
        _prepared(), config=config, mode="smoke", spec=spec, audit_fn=timed_audit(29)
    )
    for name in ("shots", "proposals", "counterfactuals", "domains"):
        assert first[f"{name}.jsonl.gz"] == second[f"{name}.jsonl.gz"]
        for row in gzip.decompress(first[f"{name}.jsonl.gz"]).splitlines():
            assert b"_wall_ns" not in row
            assert b"timing_telemetry" not in row
    timing = json.loads(first["timing.json"])
    second_timing = json.loads(second["timing.json"])
    assert timing["scientifically_deterministic"] is False
    assert timing["excluded_from_bit_exact_ledger_contract"] is True
    assert (
        timing["core_timing_by_ledger"]["proposals"][0]["timing"]
        != (second_timing["core_timing_by_ledger"]["proposals"][0]["timing"])
    )
    assert timing["per_shot"][0]["audit_wall_ns"] >= 0
    assert timing["peak_rss_bytes"] >= 0
    assert set(timing["serialization_by_artifact"]) == {
        "shots.jsonl.gz",
        "proposals.jsonl.gz",
        "counterfactuals.jsonl.gz",
        "domains.jsonl.gz",
    }


def test_collection_rejects_any_process_count_other_than_32(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 32"):
        run_policy_collection(
            default_policy_audit_draft(),
            mode="smoke",
            out=tmp_path / "out",
            processes=31,
            scientific=False,
        )


def test_worker_authenticates_and_installs_its_disjoint_shard_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared()
    config = _runtime_config()
    spec = WorkerSpec(7, 7, 1)
    shard = {"worker": spec.to_json()}
    payloads = {"sentinel": b"compressed"}
    installed: list[tuple[Path, dict, dict, dict, str, WorkerSpec]] = []
    monkeypatch.setattr(policy_experiment, "_WORKER_PREPARED", prepared)
    monkeypatch.setattr(
        policy_experiment,
        "collect_policy_worker_shard",
        lambda actual, *, config, mode, spec: (shard, payloads),
    )
    monkeypatch.setattr(
        policy_experiment,
        "install_worker_shard",
        lambda out, *, shard, payloads, config, mode, spec: installed.append(
            (out, shard, payloads, config, mode, spec)
        ),
    )
    result = _worker_task(
        {
            "config": config,
            "mode": "smoke",
            "spec": spec.to_json(),
            "scientific": False,
            "out": str(tmp_path / "campaign"),
        }
    )
    assert result == (spec.worker_id, shard, result[2], 0)
    assert result[2] > 0
    assert installed == [
        (tmp_path / "campaign", shard, payloads, config, "smoke", spec)
    ]


def test_collection_rejects_nonempty_unrelated_or_symlink_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("not a B1 artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected entries"):
        run_policy_collection(
            default_policy_audit_draft(),
            mode="smoke",
            out=unrelated,
            processes=32,
            scientific=False,
        )

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-output"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_policy_collection(
            default_policy_audit_draft(),
            mode="smoke",
            out=link,
            processes=32,
            scientific=False,
        )


def test_probe_attestation_requires_analysis_and_exact_materialized_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config()
    commit = "a" * 40
    shards = _fake_completed_probe(tmp_path / "probe", config, commit=commit)
    monkeypatch.setattr(
        "yoked.decoding.oracle.policy_experiment.verify_worker_shard",
        lambda path, *, config, mode, spec: shards[spec.worker_id],
    )
    wrong = copy.deepcopy(config)
    wrong["cell"]["circuit_sha256"] = "f0" * 32
    wrong["experiment_id"] = policy_experiment_id(wrong)
    wrong["config_self_sha256"] = policy_config_self_sha256(wrong)
    with pytest.raises(ValueError, match="materialized supplied draft"):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=wrong, implementation_commit=commit
        )
    (tmp_path / "probe" / "ANALYSIS_READY").unlink()
    with pytest.raises(ValueError, match="missing.*analysis_ready"):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=config, implementation_commit=commit
        )


def test_probe_attestation_blocks_failed_launch_gate_and_wrong_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config()
    commit = "a" * 40
    shards = _fake_completed_probe(
        tmp_path / "probe", config, commit=commit, gates_passed=False
    )
    monkeypatch.setattr(
        "yoked.decoding.oracle.policy_experiment.verify_worker_shard",
        lambda path, *, config, mode, spec: shards[spec.worker_id],
    )
    with pytest.raises(ValueError, match="every wall/storage/free-space"):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=config, implementation_commit=commit
        )
    with pytest.raises(ValueError, match="current implementation commit"):
        attest_completed_policy_probe(
            tmp_path / "probe",
            expected_config=config,
            implementation_commit="b" * 40,
        )


@pytest.mark.parametrize(
    "observed_pids, plots_rendered, message",
    [
        (31, True, "32 worker processes"),
        (32, False, "full analysis"),
    ],
)
def test_probe_attestation_requires_32_processes_and_rendered_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_pids: int,
    plots_rendered: bool,
    message: str,
) -> None:
    config = _runtime_config()
    commit = "a" * 40
    shards = _fake_completed_probe(
        tmp_path / "probe",
        config,
        commit=commit,
        observed_pids=observed_pids,
        plots_rendered=plots_rendered,
    )
    monkeypatch.setattr(
        "yoked.decoding.oracle.policy_experiment.verify_worker_shard",
        lambda path, *, config, mode, spec: shards[spec.worker_id],
    )
    with pytest.raises(ValueError, match=message):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=config, implementation_commit=commit
        )


def test_probe_attestation_authenticates_complete_100_shot_analysis_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config()
    commit = "a" * 40
    shards = _fake_completed_probe(tmp_path / "probe", config, commit=commit)
    monkeypatch.setattr(
        "yoked.decoding.oracle.policy_experiment.verify_worker_shard",
        lambda path, *, config, mode, spec: shards[spec.worker_id],
    )
    attestation = attest_completed_policy_probe(
        tmp_path / "probe", expected_config=config, implementation_commit=commit
    )
    assert attestation["verified_workers"] == 32
    assert attestation["verified_shots"] == 100
    assert attestation["all_launch_gates_passed"] is True
    assert attestation["implementation_commit"] == commit
    assert attestation["probe_config_self_sha256"] == config["config_self_sha256"]

    report_path = tmp_path / "probe" / "analysis" / "report.md"
    report_bytes = report_path.read_bytes()
    report_path.write_bytes(report_bytes + b"tamper")
    with pytest.raises(ValueError, match="ANALYSIS_READY"):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=config, implementation_commit=commit
        )
    report_path.write_bytes(report_bytes)

    (tmp_path / "probe" / "analysis" / "tables" / "overview.json").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=config, implementation_commit=commit
        )
    _write_json(tmp_path / "probe" / "analysis" / "tables" / "overview.json", {})

    with (tmp_path / "probe" / "analysis" / "summary.json").open("ab") as file:
        file.write(b" ")
    with pytest.raises(ValueError, match="summary authentication"):
        attest_completed_policy_probe(
            tmp_path / "probe", expected_config=config, implementation_commit=commit
        )


def test_frozen_protocol_validation_requires_probe_attestation() -> None:
    config = _runtime_config()
    config.update(
        {
            "status": "FROZEN",
            "frozen": True,
            "implementation_commit": "a" * 40,
            "config_commit": "verified-runtime-head",
            "protocol_relative_path": "docs/PROMATCH_POLICY_AUDIT_FROZEN_V1.json",
        }
    )
    config["experiment_id"] = policy_experiment_id(config)
    config["config_self_sha256"] = policy_config_self_sha256(config)
    with pytest.raises(ValueError, match="probe attestation"):
        validate_policy_protocol(config, scientific=False)
