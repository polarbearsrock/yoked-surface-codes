from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import REPO_ROOT
from yoked.decoding._promatch_experiment import default_smoke_protocol, prepare_cell
from yoked.decoding._promatch_graph import Edge
from yoked.decoding._promatch_layout import YokeDetector


def _load_tool():
    path = REPO_ROOT / "tools" / "diagnose_promatch_l1"
    loader = importlib.machinery.SourceFileLoader("promatch_diagnose_tool", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


TOOL = _load_tool()


def _source_hashes() -> dict[str, str]:
    paths = (
        "tools/diagnose_promatch_l1",
        "src/yoked/decoding/_artifact_io.py",
        "src/yoked/decoding/_promatch.py",
        "src/yoked/decoding/_promatch_experiment.py",
        "src/yoked/decoding/_promatch_graph.py",
        "src/yoked/decoding/_promatch_layout.py",
        "src/yoked/decoding/__init__.py",
        "src/yoked/decoding/oracle/__init__.py",
        "src/yoked/decoding/oracle/full_graph.py",
        "src/yoked/decoding/oracle/replay.py",
        "requirements.txt",
    )
    return {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in paths
    }


def _config(
    input_dir: Path, *, experiment_id: str, protocol_hash: str, summary_hash: str
):
    config = {
        "schema": TOOL.ORACLE_REPLAY_CONFIG_SCHEMA,
        "status": "FROZEN",
        "frozen": True,
        "experiment_id": "",
        "processes": 1,
        "native_threads_per_process": 1,
        "software_versions": TOOL.current_software_versions(),
        "input": {
            "path": str(input_dir),
            "experiment_id": experiment_id,
            "protocol_sha256": protocol_hash,
            "summary_sha256": summary_hash,
        },
        "selection": {
            "cell_ids": ["cell-a"],
            "categories": ["recovery", "regression"],
        },
        "oracle": {
            "absolute": 1e-9,
            "relative": 1e-6,
            "veto_budget": None,
            "sensitivity_grid": list(TOOL.ORACLE_TOLERANCE_SENSITIVITY),
            "decimal_precision": TOOL.ORACLE_DECIMAL_PRECISION,
        },
        "arms": list(TOOL.ORACLE_ARMS),
        "decoder": {
            "residual_hw_limit": 10,
            "domain_mode": "windowd",
            "boundary_policy": "disabled",
            "observable_policy": "zero-frame",
        },
        "implementation": {
            "repository_commit": "0" * 40,
            "source_hashes": _source_hashes(),
        },
    }
    semantic = dict(config)
    del semantic["experiment_id"]
    config["experiment_id"] = hashlib.sha256(
        TOOL._canonical_json_bytes(semantic)
    ).hexdigest()
    return config


def _write_json(path: Path, value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_current_source_scope_includes_parent_package_initializer() -> None:
    assert "src/yoked/decoding/__init__.py" in TOOL.ORACLE_SOURCE_PATHS
    assert frozenset(_source_hashes()) == TOOL.ORACLE_SOURCE_PATHS


def test_oracle_replay_parser_requires_frozen_config_and_output() -> None:
    args = TOOL._parser().parse_args(
        ["oracle-replay", "--config", "frozen.json", "--out", "result"]
    )
    assert args.command == "oracle-replay"
    assert args.config == Path("frozen.json")
    assert args.out == Path("result")
    with pytest.raises(SystemExit):
        TOOL._parser().parse_args(["oracle-replay", "--out", "result"])


def test_oracle_output_refuses_input_descendants_and_nonempty(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="nested"):
        TOOL._ensure_oracle_output_target(corpus, corpus / "oracle")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already contain"):
        TOOL._ensure_oracle_output_target(corpus, occupied)
    assert (
        TOOL._ensure_oracle_output_target(corpus, tmp_path / "new")
        == (tmp_path / "new").resolve()
    )


def test_diagnostic_json_output_is_no_clobber_and_outside_inputs(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="nested inside an input collection"):
        TOOL._validate_diagnostic_json_output(
            corpus / "summary.json", protected_roots=(corpus,)
        )
    with pytest.raises(ValueError, match="immutable round-one corpus"):
        TOOL._validate_diagnostic_json_output(
            tmp_path / "promatch_l1_round1_v3" / "diagnostic.json"
        )

    output = tmp_path / "diagnostic.json"
    TOOL._write_json(output, {"value": 1})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        TOOL._write_json(output, {"value": 2})
    assert output.read_bytes() == before


def test_oracle_bundle_install_is_atomic_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = {
        "shots.json": b"[]\n",
        "proposals.jsonl": b"",
        "summary.json": b"{}\n",
        "experiment.json": b"{}\n",
    }
    output = tmp_path / "bundle"
    output.mkdir()

    def fail_install(_source, _destination) -> None:
        raise OSError("injected directory install failure")

    monkeypatch.setattr(TOOL.os, "replace", fail_install)
    with pytest.raises(OSError, match="injected"):
        TOOL._install_oracle_replay_bundle(output, artifacts)
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_oracle_bundle_install_publishes_exact_complete_set(tmp_path: Path) -> None:
    artifacts = {
        "shots.json": b"[]\n",
        "proposals.jsonl": b"",
        "summary.json": b"{}\n",
        "experiment.json": b"{}\n",
    }
    for name, precreate in (("new-bundle", False), ("empty-bundle", True)):
        output = tmp_path / name
        if precreate:
            output.mkdir()
        TOOL._install_oracle_replay_bundle(output, artifacts)
        assert {path.name: path.read_bytes() for path in output.iterdir()} == artifacts


def test_packed_hex_validation_is_exact_and_rejects_padding_bits() -> None:
    assert TOOL._validate_packed_hex("0100", bit_count=9, label="x") == b"\x01\x00"
    with pytest.raises(ValueError, match="exactly 2 bytes"):
        TOOL._validate_packed_hex("01", bit_count=9, label="x")
    with pytest.raises(ValueError, match="unused high bits"):
        TOOL._validate_packed_hex("0102", bit_count=9, label="x")


def test_strict_loader_deduplicates_physical_shot_and_keeps_selection_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    experiment_id = "1" * 64
    decoder = {
        "residual_hw_limit": 10,
        "domain_mode": "windowd",
        "boundary_policy": "disabled",
        "observable_policy": "zero-frame",
    }
    protocol = {
        "schema": TOOL.PROTOCOL_SCHEMA,
        "kind": TOOL.PROTOCOL_KIND,
        "phase": "pilot",
        "experiment_id": experiment_id,
        "decoder": decoder,
        "dem_options": {"decompose_errors": True, "approximate_disjoint_errors": True},
        "software_versions": TOOL.current_software_versions(),
        "sampler_seed_roots": {"pilot": "a" * 64},
        "cell_batch_schedules": {
            "cell-a": [{"batch_id": 7, "shot_start": 70000, "shots": 10}]
        },
        "cells": [{"cell_id": "cell-a", "num_detectors": 9, "num_observables": 2}],
    }
    sample_base = {
        "batch_id": 7,
        "shot_offset": 3,
        "shot_index": 70003,
        "stim_seed": TOOL.derive_stim_batch_seed(seed_root="a" * 64, batch_id=7),
        "detection_events_hex": "0100",
        "observables_hex": "01",
        "u0_prediction_hex": "01",
        "pu_prediction_hex": "00",
    }
    samples = []
    for category in ("regression", "recovery"):
        samples.append(
            {
                **sample_base,
                "category": category,
                "selection_sha256": TOOL._replay_selection_sha256(
                    cell_id="cell-a", batch_id=7, shot_index=70003, category=category
                ),
            }
        )
    summary = {
        "schema": TOOL.SUMMARY_SCHEMA,
        "experiment_id": experiment_id,
        "phase": "pilot",
        "collection_scope": "paired-accuracy-and-workload",
        "cells": [
            {
                "cell_id": "cell-a",
                "batches": 1,
                "shots": 1,
                "paired_contingency": {},
                "telemetry": {},
                "replay_samples": samples,
            }
        ],
    }
    protocol_hash = _write_json(corpus / "protocol.json", protocol)
    summary_hash = _write_json(corpus / "summary.json", summary)
    config = _config(
        corpus,
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        summary_hash=summary_hash,
    )
    config_path = tmp_path / "oracle.json"
    _write_json(config_path, config)
    monkeypatch.setattr(
        TOOL,
        "validate_experiment_protocol",
        lambda manifest, **kwargs: manifest["experiment_id"],
    )
    monkeypatch.setattr(
        TOOL,
        "_git_blob",
        lambda *, commit, relative_path: (REPO_ROOT / relative_path).read_bytes(),
    )

    _, loaded_dir, _, _, shots = TOOL._load_oracle_replay_corpus(config_path)
    assert loaded_dir == corpus.resolve()
    assert len(shots) == 1
    assert [label["category"] for label in shots[0]["selection_labels"]] == [
        "recovery",
        "regression",
    ]


def test_frozen_config_rejects_source_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        experiment_id="2" * 64,
        protocol_hash="3" * 64,
        summary_hash="4" * 64,
    )
    config["implementation"]["source_hashes"]["tools/diagnose_promatch_l1"] = "0" * 64
    semantic = dict(config)
    del semantic["experiment_id"]
    config["experiment_id"] = hashlib.sha256(
        TOOL._canonical_json_bytes(semantic)
    ).hexdigest()
    monkeypatch.setattr(
        TOOL,
        "_git_blob",
        lambda *, commit, relative_path: (REPO_ROOT / relative_path).read_bytes(),
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        TOOL._validate_frozen_oracle_config(config)


def test_frozen_config_rejects_nonobject_source_hashes(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        experiment_id="2" * 64,
        protocol_hash="3" * 64,
        summary_hash="4" * 64,
    )
    config["implementation"]["source_hashes"] = sorted(_source_hashes())
    semantic = dict(config)
    del semantic["experiment_id"]
    config["experiment_id"] = hashlib.sha256(
        TOOL._canonical_json_bytes(semantic)
    ).hexdigest()

    with pytest.raises(ValueError, match="source_hashes must be an object"):
        TOOL._validate_frozen_oracle_config(config)


def test_historical_frozen_config_remains_readable_after_source_move() -> None:
    config = json.loads(
        (REPO_ROOT / "docs" / "PROMATCH_ORACLE_REPLAY_FROZEN_V1.json").read_text()
    )
    assert TOOL._validate_frozen_oracle_config(config) == config


def test_frozen_config_rejects_unreadable_implementation_commit(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        experiment_id="2" * 64,
        protocol_hash="3" * 64,
        summary_hash="4" * 64,
    )

    with pytest.raises(ValueError, match="cannot read frozen source"):
        TOOL._validate_frozen_oracle_config(config)


def test_frozen_config_rejects_nonprimary_operational_tolerance(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        experiment_id="2" * 64,
        protocol_hash="3" * 64,
        summary_hash="4" * 64,
    )
    config["oracle"]["relative"] = 1e-5
    semantic = dict(config)
    del semantic["experiment_id"]
    config["experiment_id"] = hashlib.sha256(
        TOOL._canonical_json_bytes(semantic)
    ).hexdigest()
    with pytest.raises(ValueError, match="operational tolerance"):
        TOOL._validate_frozen_oracle_config(config)


def test_clean_implementation_allows_only_the_post_freeze_config_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = "1" * 40
    head = "2" * 40
    config_path = REPO_ROOT / "docs" / "ORACLE_FIXTURE.json"

    def git_output(*args: str) -> str:
        if args[0] == "status":
            return ""
        if args[:2] == ("rev-parse", "HEAD"):
            return head
        if args[:2] == ("diff", "--name-only"):
            return "docs/ORACLE_FIXTURE.json"
        raise AssertionError(args)

    monkeypatch.setattr(TOOL, "_git_output", git_output)
    monkeypatch.setattr(
        TOOL.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    result = TOOL._validate_clean_implementation(
        {"implementation": {"repository_commit": implementation}},
        config_path=config_path,
    )
    assert result["post_implementation_changed_paths"] == ["docs/ORACLE_FIXTURE.json"]

    monkeypatch.setattr(
        TOOL,
        "_git_output",
        lambda *args: (
            ""
            if args[0] == "status"
            else head
            if args[0] == "rev-parse"
            else "docs/ORACLE_FIXTURE.json\nsrc/yoked/decoding/_promatch.py"
        ),
    )
    with pytest.raises(ValueError, match="exactly the frozen config"):
        TOOL._validate_clean_implementation(
            {"implementation": {"repository_commit": implementation}},
            config_path=config_path,
        )


def test_run_oracle_replay_shot_exercises_all_arms_without_observables_in_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_smoke_protocol(processes=1, shots=1)
    cell = protocol["cells"][0]
    prepared = prepare_cell(
        cell,
        decoder_config=protocol["decoder"],
        dem_options=protocol["dem_options"],
        verify_hashes=False,
    )
    cell.update(prepared.provenance)
    calls = []

    def fake_prepare(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["verify_hashes"] is True
        return prepared

    monkeypatch.setattr(TOOL, "prepare_cell", fake_prepare)
    no = prepared.dem.num_observables
    nd = prepared.dem.num_detectors
    syndrome = np.zeros(nd, dtype=np.uint8)
    first_domain = prepared.compiled_pu.graph.domain_graphs[
        sorted(prepared.compiled_pu.graph.domain_graphs)[0]
    ]
    active = sorted(first_domain.detector_ids)[:12]
    assert len(active) == 12
    syndrome[active] = 1
    diag = TOOL.Diagnoser(prepared, protocol["decoder"])
    u0 = diag.matcher.decode(syndrome).astype(np.uint8)
    _, pu = diag.run_pu(syndrome)
    config = {
        "experiment_id": "e" * 64,
        "selection": {"cell_ids": [cell["cell_id"]]},
        "decoder": protocol["decoder"],
        "oracle": {
            "absolute": 1e-9,
            "relative": 1e-6,
            "veto_budget": None,
            "sensitivity_grid": list(TOOL.ORACLE_TOLERANCE_SENSITIVITY),
            "decimal_precision": TOOL.ORACLE_DECIMAL_PRECISION,
        },
    }
    shots = [
        {
            "shot_id": "a" * 64,
            "cell_id": cell["cell_id"],
            "batch_id": 0,
            "shot_index": 0,
            "shot_offset": 0,
            "stim_seed": 0,
            "selection_labels": [],
            "detection_events_hex": bytes(
                np.packbits(syndrome, bitorder="little")
            ).hex(),
            "observables_hex": bytes((no + 7) // 8).hex(),
            "u0_prediction_hex": bytes(np.packbits(u0, bitorder="little")).hex(),
            "pu_prediction_hex": bytes(np.packbits(pu, bitorder="little")).hex(),
        }
    ]

    shot_rows, proposal_rows, summary = TOOL._run_oracle_replay_shots(
        config, protocol, shots
    )
    assert len(calls) == 1
    assert len(shot_rows) == 1
    assert proposal_rows
    assert all("omitted_context_labels" in row for row in proposal_rows)
    assert set(TOOL.ORACLE_ARMS).issubset(shot_rows[0]["trajectories"])
    assert all(
        summary["pooled_arms"][arm]["logical_failures"] == 0 for arm in TOOL.ORACLE_ARMS
    )
    assert set(summary["per_cell"]) == {cell["cell_id"]}
    numeric = summary["numeric_characterization"]
    assert numeric["status"] == "passed"
    assert numeric["checks"]["classification_changes_across_grid"] == 0
    assert numeric["pooled"]["proposal_evaluations"] == len(proposal_rows)


def test_retained_origin_batch_is_regenerated_and_verified() -> None:
    protocol = default_smoke_protocol(processes=1, shots=1)
    cell = protocol["cells"][0]
    prepared = prepare_cell(
        cell,
        decoder_config=protocol["decoder"],
        dem_options=protocol["dem_options"],
        verify_hashes=False,
    )
    batch = protocol["cell_batch_schedules"][cell["cell_id"]][0]
    seed = TOOL.derive_stim_batch_seed(
        seed_root=protocol["sampler_seed_roots"]["smoke"],
        batch_id=batch["batch_id"],
    )
    dets, obs = prepared.circuit.compile_detector_sampler(seed=seed).sample(
        shots=1, separate_observables=True, bit_packed=True
    )
    retained = {
        "cell_id": cell["cell_id"],
        "batch_id": batch["batch_id"],
        "shot_index": 0,
        "shot_offset": 0,
        "stim_seed": seed,
        "detection_events_hex": np.asarray(dets, dtype=np.uint8)[0].tobytes().hex(),
        "observables_hex": np.asarray(obs, dtype=np.uint8)[0].tobytes().hex(),
    }
    rows = TOOL._regenerate_retained_batches(
        prepared_by_cell={cell["cell_id"]: prepared},
        protocol=protocol,
        shots=[retained],
    )
    assert len(rows) == 1
    assert rows[0]["retained_rows_verified"] == 1
    assert rows[0]["detectors"]["shape"][0] == 1


def _numeric_characterization_fixture(*, residual: float, base_backend: float = 1.0):
    role = YokeDetector()
    edges = (
        Edge(0, 0, 1, 1.0, b"", role, role),
        Edge(1, 2, 3, 0.5, b"", role, role),
        Edge(2, 4, 5, residual, b"", role, role),
    )
    base = 1.0
    candidate = 0.5
    # Match the implementation's one-rounding math.fsum contract exactly.
    composite = math.fsum((candidate, residual))
    excess = composite - base
    tau = 1e-9 + 1e-6 * max(1.0, abs(base), abs(composite))
    values = {
        "base_support_weight": base,
        "residual_support_weight": residual,
        "candidate_weight": candidate,
        "composite_weight": composite,
        "cost_excess": excess,
        "tau_k": tau,
        "base_backend_weight": base_backend,
        "residual_backend_weight": residual,
        "base_tau_weight": 1e-9 + 1e-6 * max(1.0, abs(base), abs(base_backend)),
        "residual_tau_weight": 1e-9 + 1e-6 * max(1.0, abs(residual), abs(residual)),
    }
    evaluation = {
        "base_support_edge_ids": [0],
        "residual_support_edge_ids": [2],
        "candidate_edge_ids": [1],
        "cost_classification": "numerically-cost-compatible",
        **values,
        **{f"{name}_hex": value.hex() for name, value in values.items()},
    }
    config = {
        "oracle": {
            "relative": 1e-6,
            "absolute": 1e-9,
            "sensitivity_grid": list(TOOL.ORACLE_TOLERANCE_SENSITIVITY),
            "decimal_precision": TOOL.ORACLE_DECIMAL_PRECISION,
        }
    }
    graph = SimpleNamespace(edges=edges, fingerprint="numeric-test")
    prepared = SimpleNamespace(compiled_pu=SimpleNamespace(graph=graph))
    row = {
        "cell_id": "cell",
        "arm": "shadow",
        "graph_fingerprint": "numeric-test",
        "proposal_sha256": "6" * 64,
        "evaluation": evaluation,
    }
    return config, row, prepared


def test_numeric_characterization_rejects_sensitivity_class_change() -> None:
    config, row, prepared = _numeric_characterization_fixture(residual=0.5000005)
    with pytest.raises(ValueError, match="classification changes"):
        TOOL._characterize_numeric_proposals(
            config=config,
            proposal_rows=[row],
            prepared_by_cell={"cell": prepared},
        )


def test_numeric_characterization_rejects_strict_grid_weight_mismatch() -> None:
    config, row, prepared = _numeric_characterization_fixture(
        residual=0.5,
        base_backend=1.0000005,
    )
    with pytest.raises(ValueError, match="weight.*sensitivity"):
        TOOL._characterize_numeric_proposals(
            config=config,
            proposal_rows=[row],
            prepared_by_cell={"cell": prepared},
        )


def test_numeric_characterization_rejects_corrupt_backend_float_hex() -> None:
    config, row, prepared = _numeric_characterization_fixture(residual=0.5)
    row["evaluation"]["base_backend_weight_hex"] = (0.0).hex()
    with pytest.raises(ValueError, match="exact-float companion"):
        TOOL._characterize_numeric_proposals(
            config=config,
            proposal_rows=[row],
            prepared_by_cell={"cell": prepared},
        )


def test_numeric_characterization_rejects_nonfinite_backend_weight() -> None:
    config, row, prepared = _numeric_characterization_fixture(residual=0.5)
    row["evaluation"]["base_backend_weight"] = float("nan")
    row["evaluation"]["base_backend_weight_hex"] = "nan"
    with pytest.raises(ValueError, match="nonfinite"):
        TOOL._characterize_numeric_proposals(
            config=config,
            proposal_rows=[row],
            prepared_by_cell={"cell": prepared},
        )


def test_numeric_characterization_rejects_noninteger_support_id() -> None:
    config, row, prepared = _numeric_characterization_fixture(residual=0.5)
    row["evaluation"]["base_support_edge_ids"] = [True]
    with pytest.raises(ValueError, match="canonical integer IDs"):
        TOOL._characterize_numeric_proposals(
            config=config,
            proposal_rows=[row],
            prepared_by_cell={"cell": prepared},
        )


def test_stage3_context_uses_candidate_adjacent_internal_vertices() -> None:
    role = YokeDetector()
    edges = (
        Edge(0, 0, 1, 1.0, b"", role, role),
        Edge(1, 1, 2, 1.0, b"", role, role),
    )
    diag = SimpleNamespace(
        graph=SimpleNamespace(edges=edges, fingerprint="stage3-test"),
        partner_kind=lambda endpoint, partner: "in-domain",
    )
    proposal = {
        "ordinal": 0,
        "proposal": {
            "endpoints": [0, 2],
            "edge_ids": [0, 1],
            "state_total_candidate_count": 1,
            "state_veto_count_before": 0,
        },
        "evaluation": {
            "candidate_edge_ids": [0, 1],
            "base_support_edge_ids": [0, 1],
        },
        "pre_state_fingerprint": "1" * 64,
        "post_decision_state_fingerprint": "2" * 64,
        "oracle_evaluation_ordinal": 0,
        "state_veto_count_after": 0,
        "accepted": True,
        "vetoed": False,
        "durable": True,
        "rolled_back": False,
    }
    empty = {"proposals": [], "domains": []}
    phase = {
        "shadow": {"proposals": []},
        "o-cost-tx": {
            "proposals": [proposal],
            "domains": [
                {
                    "proposal_start": 0,
                    "proposal_stop": 1,
                    "outcome": {
                        "transaction_policy": "tx",
                        "status": "success",
                        "fallback_reason": None,
                        "exhaustion_kind": None,
                    },
                }
            ],
        },
        "o-frame-tx": dict(empty),
        "o-frame-partial": dict(empty),
    }
    retained = {
        "shot_id": "3" * 64,
        "cell_id": "cell",
        "batch_id": 1,
        "shot_index": 2,
        "shot_offset": 2,
        "stim_seed": 4,
        "selection_labels": [],
        "detection_events_hex": "01",
    }
    rows = TOOL._proposal_rows(
        experiment_id="5" * 64,
        retained=retained,
        phase_json=phase,
        diag=diag,
        legacy_result=SimpleNamespace(paths=(), domain_stats={}),
    )
    assert len(rows) == 1
    assert rows[0]["base_mwpm_contains_candidate_support"]
    assert rows[0]["omitted_context_labels"] == []
    assert [
        endpoint["candidate_adjacent_partner_id"]
        for endpoint in rows[0]["base_mwpm_endpoint_partners"]
    ] == [1, 1]
