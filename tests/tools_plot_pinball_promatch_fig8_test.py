from __future__ import annotations

import contextlib
import csv
import io
import json
import pathlib
import runpy
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "plot_pinball_promatch_fig8"


def _payload(*, cell_id: str, scale: int = 1) -> dict:
    cube = {
        "000": 80 * scale,
        "001": 5 * scale,
        "010": 4 * scale,
        "011": 1 * scale,
        "100": 3 * scale,
        "101": 2 * scale,
        "110": 1 * scale,
        "111": 4 * scale,
    }
    return {
        "cell_id": cell_id,
        "batch": {"batch_id": 0, "shot_start": 0, "shots": 100 * scale},
        "correctness_cube": cube,
        "pairwise_contingencies": {
            "pinball_minus_promatch": {
                "both_correct": 83 * scale,
                "regressions": 7 * scale,
                "recoveries": 5 * scale,
                "both_wrong": 5 * scale,
            },
            "pinball_minus_u0": {
                "both_correct": 84 * scale,
                "regressions": 6 * scale,
                "recoveries": 4 * scale,
                "both_wrong": 6 * scale,
            },
            "promatch_minus_u0": {
                "both_correct": 85 * scale,
                "regressions": 5 * scale,
                "recoveries": 5 * scale,
                "both_wrong": 5 * scale,
            },
        },
        "prediction_agreement": {
            pair: {"agree": 90 * scale, "disagree": 10 * scale}
            for pair in (
                "pinball_minus_promatch",
                "pinball_minus_u0",
                "promatch_minus_u0",
            )
        },
        "telemetry": {
            "common": {
                "shots": 100 * scale,
                "original_event_sum": 1_000 * scale,
            },
            "promatch": {
                "shots": 100 * scale,
                "residual_event_sum": 700 * scale,
                "activated_shots": 50 * scale,
                "attempted_stage_counts": [scale] * 4,
            },
            "pinball": {
                "shots": 100 * scale,
                "residual_event_sum": 500 * scale,
                "complex_shots": 40 * scale,
                "stage_match_counts_by_family": {"M": 2 * scale},
            },
        },
    }


def _campaign(directory: pathlib.Path) -> tuple[dict, tuple[dict, ...]]:
    cells = [
        {
            "cell_id": "fig8-d5-n6-y2-r20-p0.002",
            "d": 5,
            "patches": 6,
            "r": 20,
            "yokes": 2,
            "p": 0.002,
        },
        {
            "cell_id": "fig8-d7-n10-y2-r56-p0.002",
            "d": 7,
            "patches": 10,
            "r": 56,
            "yokes": 2,
            "p": 0.002,
        },
    ]
    campaign = {
        "experiment_id": "a" * 64,
        "cells": cells,
        "expected_shots_by_cell": {cell["cell_id"]: 100 for cell in cells},
    }
    return campaign, tuple(_payload(cell_id=cell["cell_id"]) for cell in cells)


def _run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete: bool = True,
    overwrite: bool = False,
) -> tuple[int, str, str, list[bool], pathlib.Path]:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(exist_ok=True)
    namespace = runpy.run_path(str(SCRIPT))
    campaign, rows = _campaign(campaign_dir)
    required: list[bool] = []

    def load(_directory, *, require_complete):
        required.append(require_complete)
        if not complete:
            raise ValueError("collection is incomplete")
        return campaign, rows

    namespace["main"].__globals__["load_validated_collection"] = load
    namespace["main"].__globals__["validate_analysis_runtime"] = lambda _: None
    argv = [str(SCRIPT), "--campaign", str(campaign_dir)]
    if overwrite:
        argv.append("--overwrite")
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = namespace["main"]()
    finally:
        sys.argv = previous
    return code, stdout.getvalue(), stderr.getvalue(), required, campaign_dir


def test_complete_collection_writes_json_csv_and_clear_png(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, stdout, stderr, required, campaign_dir = _run(tmp_path, monkeypatch)
    assert code == 0, stderr
    assert required == [True]
    assert "analysis.png" in stdout
    output = campaign_dir / "analysis"
    assert {path.name for path in output.iterdir()} == {
        "analysis.json",
        "analysis.csv",
        "analysis.png",
    }
    report = json.loads((output / "analysis.json").read_text())
    assert report["campaign_id"] == "a" * 64
    assert len(report["cells"]) == 2
    assert (output / "analysis.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "analysis.png").stat().st_size > 10_000
    with (output / "analysis.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["d"] == "5"
    assert rows[0]["patches"] == "6"
    assert rows[0]["rounds"] == "20"
    assert rows[0]["round_multiplier"] == "4"
    assert float(rows[0]["pinball_ler"]) == 0.12
    assert float(rows[0]["pinball_minus_promatch_delta"]) == 0.02
    assert float(rows[0]["workload_pinball_residual_over_original"]) == 0.5


def test_incomplete_collection_fails_before_creating_analysis(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, _, stderr, required, campaign_dir = _run(
        tmp_path, monkeypatch, complete=False
    )
    assert code == 2
    assert required == [True]
    assert "incomplete" in stderr
    assert not (campaign_dir / "analysis").exists()


def test_dirty_runtime_fails_before_replacing_analysis(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    analysis = campaign_dir / "analysis"
    analysis.mkdir()
    marker = analysis / "keep.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    namespace = runpy.run_path(str(SCRIPT))
    campaign, rows = _campaign(campaign_dir)
    namespace["main"].__globals__["load_validated_collection"] = (
        lambda *_args, **_kwargs: (campaign, rows)
    )
    namespace["main"].__globals__["validate_analysis_runtime"] = (
        lambda *_: (_ for _ in ()).throw(ValueError("clean worktree required"))
    )
    previous = sys.argv
    stderr = io.StringIO()
    try:
        sys.argv = [str(SCRIPT), "--campaign", str(campaign_dir), "--overwrite"]
        with contextlib.redirect_stderr(stderr):
            code = namespace["main"]()
    finally:
        sys.argv = previous

    assert code == 2
    assert "clean worktree" in stderr.getvalue()
    assert marker.read_text() == "preserve\n"
    assert {path.name for path in analysis.iterdir()} == {"keep.txt"}


def test_inconsistent_ledger_fails_before_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    namespace = runpy.run_path(str(SCRIPT))
    campaign, rows = _campaign(campaign_dir)
    changed = [dict(row) for row in rows]
    changed[0] = dict(changed[0])
    changed[0]["correctness_cube"] = dict(changed[0]["correctness_cube"])
    changed[0]["correctness_cube"]["000"] -= 1
    namespace["main"].__globals__["load_validated_collection"] = (
        lambda *_args, **_kwargs: (campaign, tuple(changed))
    )
    namespace["main"].__globals__["validate_analysis_runtime"] = lambda _: None
    previous = sys.argv
    stderr = io.StringIO()
    try:
        sys.argv = [str(SCRIPT), "--campaign", str(campaign_dir)]
        with contextlib.redirect_stderr(stderr):
            code = namespace["main"]()
    finally:
        sys.argv = previous
    assert code == 2
    assert "reconcile" in stderr.getvalue()
    assert not (campaign_dir / "analysis").exists()


def test_refuses_existing_artifacts_without_overwrite(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(tmp_path, monkeypatch)
    assert first[0] == 0
    second = _run(tmp_path, monkeypatch)
    assert second[0] == 2
    assert "--overwrite" in second[2]
    third = _run(tmp_path, monkeypatch, overwrite=True)
    assert third[0] == 0, third[2]


def test_preserves_symlink_spelling_for_campaign_loader_rejection(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "real-campaign"
    target.mkdir()
    alias = tmp_path / "campaign-alias"
    alias.symlink_to(target, target_is_directory=True)
    namespace = runpy.run_path(str(SCRIPT))
    seen: list[pathlib.Path] = []

    def load(directory, *, require_complete):
        assert require_complete is True
        seen.append(directory)
        if directory.is_symlink():
            raise ValueError("campaign root may not be a symlink")
        pytest.fail("plotter resolved campaign symlink before validation")

    namespace["main"].__globals__["load_validated_collection"] = load
    previous = sys.argv
    stderr = io.StringIO()
    try:
        sys.argv = [str(SCRIPT), "--campaign", str(alias)]
        with contextlib.redirect_stderr(stderr):
            code = namespace["main"]()
    finally:
        sys.argv = previous
    assert code == 2
    assert seen == [alias.absolute()]
    assert "symlink" in stderr.getvalue()
    assert not (target / "analysis").exists()
