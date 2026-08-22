"""Synthetic artifact tests for tools/plot_fig8_paired."""

from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import json
import os
import pathlib
import runpy
import subprocess
import sys
import types

import pytest
import sinter

from yoked.decoding._fig8_paired_sweep import DECODER, DEM_OPTIONS, GRID
from yoked.decoding._promatch_experiment import GENERATOR, SEED_DERIVATION
from yoked.decoding._promatch_stats import derive_stim_batch_seed


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "tools" / "plot_fig8_paired"


def _experiment_id(campaign: dict) -> str:
    semantic = dict(campaign)
    semantic.pop("experiment_id", None)
    encoded = json.dumps(
        semantic,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_campaign(tmp_path: pathlib.Path, *, p: float = 0.001) -> pathlib.Path:
    campaign_dir = tmp_path / "campaign"
    cells = []
    for index, (d, patches, multiplier) in enumerate(
        (d, patches, multiplier)
        for d in GRID["distances"]
        for patches in GRID["patches"]
        for multiplier in GRID["round_multipliers"]
    ):
        rounds = d * multiplier
        cells.append(
            {
                "cell_id": (f"fig8-d{d}-n{patches}-y2-r{rounds}-p{format(p, '.17g')}"),
                "generator": GENERATOR,
                "d": d,
                "r": rounds,
                "p": p,
                "patches": patches,
                "yokes": 2,
                "style": "cz",
                "noise": "si1000",
                "remove_x_yoke": False,
                "circuit_sha256": format(index + 1, "064x"),
                "dem_sha256": format(index + 101, "064x"),
                "layout_fingerprint": format(index + 201, "064x"),
                "graph_fingerprint": format(index + 301, "064x"),
                "num_detectors": 100 + index,
                "num_observables": 12,
            }
        )
    campaign = {
        "schema": "yoked.fig8-paired-sweep-campaign-v1",
        "kind": "yoked-fig8b-paired-fixed-shot-sweep",
        "status": "FROZEN",
        "frozen": True,
        "claim_bearing": False,
        "phase": "fig8-sweep",
        "repository_commit": "a" * 40,
        "clean_worktree": True,
        "created_utc": "2026-08-21T00:00:00Z",
        "software_versions": {"synthetic": "1"},
        "execution_environment": {"synthetic": "test"},
        "source_hashes": {
            "src/yoked/decoding/_fig8_paired_sweep.py": "b" * 64,
            "tools/benchmark_fig8_paired": "c" * 64,
            "tools/plot_fig8_paired": "d" * 64,
        },
        "processes": 32,
        "threads_per_process": 1,
        "sample_batch_size": 1000,
        "shots_per_cell": 10,
        "p": p,
        "seed_derivation": SEED_DERIVATION,
        "sampler_seed_root": "0" * 64,
        "dem_options": DEM_OPTIONS,
        "decoder": DECODER,
        "grid": GRID,
        "expected_shots_by_cell": {cell["cell_id"]: 10 for cell in cells},
        "cell_batch_schedules": {
            cell["cell_id"]: [{"batch_id": index, "shot_start": 0, "shots": 10}]
            for index, cell in enumerate(cells)
        },
        "cells": cells,
    }
    campaign["experiment_id"] = _experiment_id(campaign)
    campaign_dir.mkdir()
    (campaign_dir / "campaign.json").write_text(
        json.dumps(campaign, sort_keys=True, indent=2) + "\n"
    )

    for index, cell in enumerate(cells):
        table = (
            {
                "both_correct": 6,
                "recoveries": 1,
                "regressions": 2,
                "both_wrong": 1,
            }
            if index == 0
            else {
                "both_correct": 7,
                "recoveries": 0,
                "regressions": 1,
                "both_wrong": 2,
            }
        )
        batch_dir = campaign_dir / "collection" / "batches" / cell["cell_id"]
        batch_dir.mkdir(parents=True)
        shots = 10
        ledger = {
            "schema": "promatch-l1-paired-batch-v1",
            "experiment_id": campaign["experiment_id"],
            "phase": "fig8-sweep",
            "cell_id": cell["cell_id"],
            "batch": {"batch_id": index, "shot_start": 0, "shots": shots},
            "stim_seed": derive_stim_batch_seed(
                seed_root=campaign["sampler_seed_root"], batch_id=index
            ),
            "detectors": {
                "sha256": "c" * 64,
                "shape": [shots, (cell["num_detectors"] + 7) // 8],
                "dtype": "|u1",
            },
            "observables": {
                "sha256": "d" * 64,
                "shape": [shots, (cell["num_observables"] + 7) // 8],
                "dtype": "|u1",
            },
            "provenance": {
                key: cell[key]
                for key in (
                    "circuit_sha256",
                    "dem_sha256",
                    "layout_fingerprint",
                    "graph_fingerprint",
                    "num_detectors",
                    "num_observables",
                )
            },
            "paired_contingency": table,
            "telemetry": _synthetic_telemetry(shots),
            "replay_samples": [],
        }
        (batch_dir / f"batch-{index:08d}.json").write_text(
            json.dumps(ledger, sort_keys=True, indent=2) + "\n"
        )
    return campaign_dir


def _synthetic_telemetry(shots: int) -> dict:
    return {
        "shots": shots,
        "original_event_sum": 0,
        "residual_event_sum": 0,
        "original_hw_histogram": {"0": shots},
        "residual_hw_histogram": {"0": shots},
        "original_residual_hw_joint_histogram": {"0,0": shots},
        "domain_initial_hw_histogram": {},
        "domain_attempted_hw_histogram": {},
        "domain_final_hw_histogram": {},
        "domain_status_counts": {},
        "domain_identity_counts": {},
        "fallback_reason_counts": {},
        "decision_weight_histogram_float_hex": {"0x0.0p+0": shots},
        "xor_support_weight_histogram_float_hex": {"0x0.0p+0": shots},
        "committed_path_length_histogram": {},
        "terminal_withheld_event_sum": 0,
        "yoke_withheld_event_sum": 0,
        "attempted_stage_counts": [0, 0, 0, 0],
        "committed_stage_counts": [0, 0, 0, 0],
        "attempted_matches": 0,
        "committed_matches": 0,
        "boundary_added_domains": 0,
        "boundary_used_domains": 0,
        "boundary_discarded_domains": 0,
        "activated_shots": 0,
        "rollback_shots": 0,
        "success_shots": 0,
    }


def _run(
    campaign_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    *,
    runtime_validator=None,
    profile_resolver=None,
) -> subprocess.CompletedProcess:
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    namespace = runpy.run_path(str(SCRIPT))
    namespace["main"].__globals__["validate_analysis_runtime"] = runtime_validator or (
        lambda _: None
    )
    if profile_resolver is not None:
        namespace["main"].__globals__["_campaign_profile"] = profile_resolver
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT), "--campaign", str(campaign_dir)]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = namespace["main"]()
    finally:
        sys.argv = previous_argv
    return subprocess.CompletedProcess(
        args=sys.argv,
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _rewrite_campaign_identity(
    campaign_dir: pathlib.Path, *, schema: str, kind: str, processes: int
) -> dict:
    campaign_path = campaign_dir / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign.update(schema=schema, kind=kind, processes=processes)
    campaign["experiment_id"] = _experiment_id(campaign)
    campaign_path.write_text(json.dumps(campaign, sort_keys=True, indent=2) + "\n")
    for ledger_path in (campaign_dir / "collection" / "batches").glob("*/*.json"):
        ledger = json.loads(ledger_path.read_text())
        ledger["experiment_id"] = campaign["experiment_id"]
        ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    return campaign


def test_writes_plot_and_paired_ler_table(tmp_path: pathlib.Path) -> None:
    campaign_dir = _write_campaign(tmp_path)
    validated = []

    proc = _run(
        campaign_dir,
        tmp_path,
        runtime_validator=lambda campaign: validated.append(campaign["experiment_id"]),
    )

    assert proc.returncode == 0, proc.stderr
    assert len(validated) == 1
    png = campaign_dir / "plots" / "fig8b_paired_ler_p0.001.png"
    table = campaign_dir / "plots" / "fig8b_paired_results.csv"
    assert png.stat().st_size > 0
    assert table.is_file()

    with table.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16
    row = rows[0]
    assert row["cell_id"] == "fig8-d5-n6-y2-r20-p0.001"
    assert int(row["u0_failures"]) == 2
    assert int(row["pu_failures"]) == 3
    assert int(row["recoveries"]) == 1
    assert int(row["regressions"]) == 2

    cell = json.loads((campaign_dir / "campaign.json").read_text())["cells"][0]
    u0_shot_fit = sinter.fit_binomial(
        num_shots=10, num_hits=2, max_likelihood_factor=1000
    )
    expected = sinter.shot_error_rate_to_piece_error_rate(
        u0_shot_fit,
        pieces=cell["r"] * cell["patches"],
        values=(cell["patches"] - cell["yokes"]) * 2,
    )
    assert float(row["u0_ler_per_patch_round_low"]) == expected.low
    assert float(row["u0_ler_per_patch_round_best"]) == expected.best
    assert float(row["u0_ler_per_patch_round_high"]) == expected.high


def test_aws_profile_api_is_selected_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    validator = lambda _campaign: None
    loader = lambda _directory, require_complete: ({}, ())
    fake_module = types.SimpleNamespace(
        CAMPAIGN_SCHEMA="synthetic.aws-campaign-v1",
        CAMPAIGN_KIND="synthetic-aws-paired-sweep",
        REQUIRED_PROCESSES=192,
        validate_analysis_environment=validator,
        load_validated_collection=loader,
    )
    importlib_module = namespace["_aws_campaign_profile"].__globals__["importlib"]
    monkeypatch.setattr(
        importlib_module,
        "import_module",
        lambda name: fake_module
        if name == "yoked.decoding._fig8_paired_aws_sweep"
        else pytest.fail(f"unexpected import {name}"),
    )

    profile = namespace["_campaign_profile"](
        {"schema": "synthetic.aws-campaign-v1"}
    )

    assert profile.schema == "synthetic.aws-campaign-v1"
    assert profile.kind == "synthetic-aws-paired-sweep"
    assert profile.processes == 192
    assert profile.validate_analysis_runtime is validator
    assert profile.load_validated_collection is loader


def test_real_aws_campaign_module_exposes_plotting_profile() -> None:
    from yoked.decoding import _fig8_paired_aws_sweep as aws_sweep

    namespace = runpy.run_path(str(SCRIPT))
    profile = namespace["_campaign_profile"](
        {"schema": aws_sweep.CAMPAIGN_SCHEMA}
    )

    assert profile.schema == aws_sweep.CAMPAIGN_SCHEMA
    assert profile.kind == aws_sweep.CAMPAIGN_KIND
    assert profile.processes == aws_sweep.REQUIRED_PROCESSES == 192
    assert profile.validate_analysis_runtime is aws_sweep.validate_analysis_runtime
    assert profile.load_validated_collection is aws_sweep.load_validated_collection


def test_aws_profile_can_write_the_same_paired_plot(tmp_path: pathlib.Path) -> None:
    campaign_dir = _write_campaign(tmp_path)
    schema = "synthetic.aws-campaign-v1"
    kind = "synthetic-aws-paired-sweep"
    campaign = _rewrite_campaign_identity(
        campaign_dir, schema=schema, kind=kind, processes=192
    )
    ledgers = tuple(
        json.loads(path.read_text())
        for path in sorted(
            (campaign_dir / "collection" / "batches").glob("*/*.json")
        )
    )
    validated: list[str] = []
    namespace = runpy.run_path(str(SCRIPT))
    profile = namespace["CampaignProfile"](
        schema=schema,
        kind=kind,
        processes=192,
        validate_analysis_runtime=lambda value: validated.append(
            value["experiment_id"]
        ),
        load_validated_collection=lambda _directory, require_complete: (
            campaign,
            ledgers,
        ),
    )

    proc = _run(
        campaign_dir,
        tmp_path,
        profile_resolver=lambda _campaign: profile,
    )

    assert proc.returncode == 0, proc.stderr
    assert validated == [campaign["experiment_id"]]
    assert (campaign_dir / "plots" / "fig8b_paired_results.csv").is_file()
    assert (campaign_dir / "plots" / "fig8b_paired_ler_p0.001.png").is_file()


def test_non_paper_probability_omits_paper_fit_and_writes_plot(
    tmp_path: pathlib.Path,
) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    include_paper_fit = namespace["_include_paper_fit"]
    assert include_paper_fit(0.001)
    assert not include_paper_fit(0.002)

    campaign_dir = _write_campaign(tmp_path, p=0.002)
    proc = _run(campaign_dir, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert (campaign_dir / "plots" / "fig8b_paired_ler_p0.002.png").is_file()


def test_rejects_batch_experiment_identity_mismatch(tmp_path: pathlib.Path) -> None:
    campaign_dir = _write_campaign(tmp_path)
    ledger_path = next((campaign_dir / "collection" / "batches").glob("*/*.json"))
    ledger = json.loads(ledger_path.read_text())
    ledger["experiment_id"] = "f" * 64
    ledger_path.write_text(json.dumps(ledger) + "\n")

    proc = _run(campaign_dir, tmp_path)

    assert proc.returncode != 0
    assert "protocol identity mismatch" in proc.stderr
    assert not (campaign_dir / "plots").exists()


def test_runtime_provenance_failure_prevents_plot_writes(
    tmp_path: pathlib.Path,
) -> None:
    campaign_dir = _write_campaign(tmp_path)

    def reject_runtime(_campaign):
        raise ValueError("synthetic runtime drift")

    proc = _run(campaign_dir, tmp_path, runtime_validator=reject_runtime)

    assert proc.returncode != 0
    assert "synthetic runtime drift" in proc.stderr
    assert not (campaign_dir / "plots").exists()


@pytest.mark.parametrize("failure", ["missing_batch", "wrong_shots", "extra_batch"])
def test_rejects_incomplete_or_nonexact_collection(
    tmp_path: pathlib.Path, failure: str
) -> None:
    campaign_dir = _write_campaign(tmp_path)
    batch_paths = sorted((campaign_dir / "collection" / "batches").glob("*/*.json"))
    if failure == "missing_batch":
        batch_paths[0].unlink()
    elif failure == "wrong_shots":
        campaign = json.loads((campaign_dir / "campaign.json").read_text())
        campaign["expected_shots_by_cell"][campaign["cells"][0]["cell_id"]] = 11
        campaign["experiment_id"] = _experiment_id(campaign)
        (campaign_dir / "campaign.json").write_text(json.dumps(campaign) + "\n")
        for path in batch_paths:
            ledger = json.loads(path.read_text())
            ledger["experiment_id"] = campaign["experiment_id"]
            path.write_text(json.dumps(ledger) + "\n")
    else:
        extra = batch_paths[0].with_name("batch-99999999.json")
        extra.write_text(batch_paths[0].read_text())

    proc = _run(campaign_dir, tmp_path)

    assert proc.returncode != 0
    assert not (campaign_dir / "plots").exists()
