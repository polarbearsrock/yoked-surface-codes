from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path

from tests.conftest import REPO_ROOT


def _load_tool():
    path = REPO_ROOT / "tools" / "benchmark_fig8_paired"
    loader = importlib.machinery.SourceFileLoader(
        "benchmark_fig8_paired_tool", str(path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_import_forces_one_native_thread(monkeypatch) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "8")
    _load_tool()
    assert {
        os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    } == {"1"}


def test_create_cli_prints_compact_identity(monkeypatch, capsys) -> None:
    tool = _load_tool()
    seen = {}

    def create(out: Path, *, p: float, shots_per_cell: int):
        seen.update(out=out, p=p, shots=shots_per_cell)
        return {
            "experiment_id": "a" * 64,
            "cells": [{}] * 16,
            "shots_per_cell": shots_per_cell,
        }

    monkeypatch.setattr(tool, "create_campaign", create)
    assert (
        tool.main(
            [
                "create",
                "--out",
                "campaign",
                "--p",
                "0.001",
                "--shots-per-cell",
                "123",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["cells"] == 16
    assert payload["shots_per_cell"] == 123
    assert seen == {"out": Path("campaign"), "p": 0.001, "shots": 123}


def test_run_cli_rejects_non_32_process_count(capsys) -> None:
    tool = _load_tool()
    assert tool.main(["run", "--campaign", "campaign", "--processes", "31"]) == 2
    assert "exactly 32" in capsys.readouterr().err


def test_status_cli_emits_compact_json(monkeypatch, capsys) -> None:
    tool = _load_tool()
    expected = {
        "schema": "status",
        "complete": False,
        "cells": [],
        "totals": {"completed_batches": 0},
    }
    monkeypatch.setattr(tool, "campaign_status", lambda _: expected)
    assert tool.main(["status", "--campaign", "campaign"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == expected
    assert ": " not in output
