"""Hermetic tests for the AWS On-Demand continuation wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WRAPPER = REPO_ROOT / "aws/run_fig8_paired_ondemand"


def _mini_repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    log = tmp_path / "calls.log"
    (repo / "aws").mkdir(parents=True)
    (repo / "tools").mkdir()
    campaign = runtime / "runs/fig8-paired-aws192/run1"
    campaign.mkdir(parents=True)
    shutil.copy2(WRAPPER, repo / "aws/run_fig8_paired_ondemand")

    activation = repo / "aws/activate_environment"
    activation.write_text(
        r"""#!/usr/bin/env bash
export YSC_AWS_RUNTIME_ROOT="$FAKE_RUNTIME_ROOT"
export YSC_AWS_RUNS_ROOT="$FAKE_RUNTIME_ROOT/runs"
unset MAX_ERRORS
""",
        encoding="utf-8",
    )
    benchmark = repo / "tools/benchmark_fig8_paired_aws_ondemand"
    benchmark.write_text(
        r"""#!/usr/bin/env bash
{
    printf 'args=%s\n' "$*"
    printf 'threads=%s,%s,%s,%s,%s,%s\n' \
        "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" \
        "$NUMEXPR_NUM_THREADS" "$VECLIB_MAXIMUM_THREADS" "$BLIS_NUM_THREADS"
    printf 'layout=%s,%s,%s,%s\n' \
        "${PROCESSES-unset}" "${THREADS_PER_PROCESS-unset}" \
        "${YSC_AWS_NUMA_POOLS-unset}" "${YSC_AWS_PROCESSES_PER_POOL-unset}"
} >> "$FAKE_TOOL_LOG"
""",
        encoding="utf-8",
    )
    for path in (activation, benchmark, repo / "aws/run_fig8_paired_ondemand"):
        path.chmod(0o755)
    return repo, runtime, log


def _run(repo: Path, runtime: Path, log: Path, action: str):
    environment = dict(os.environ)
    environment.update(FAKE_RUNTIME_ROOT=str(runtime), FAKE_TOOL_LOG=str(log))
    return subprocess.run(
        [str(repo / "aws/run_fig8_paired_ondemand"), action, "--run-id", "run1"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_prepare_and_run_forward_fixed_campaign_and_layout(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = runtime / "runs/fig8-paired-aws192/run1"

    prepared = _run(repo, runtime, log, "prepare")
    assert prepared.returncode == 0, prepared.stderr
    first = log.read_text(encoding="utf-8")
    assert f"args=prepare --campaign {campaign}" in first
    assert "layout=192,1,2,96" in first

    log.write_text("", encoding="utf-8")
    run = _run(repo, runtime, log, "run")
    assert run.returncode == 0, run.stderr
    second = log.read_text(encoding="utf-8")
    assert f"args=run --campaign {campaign} --processes 192" in second
    assert "threads=1,1,1,1,1,1" in second
    assert "layout=192,1,2,96" in second


def test_status_is_read_only_and_does_not_create_lock_directory(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = runtime / "runs/fig8-paired-aws192/run1"
    result = _run(repo, runtime, log, "status")
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").startswith(
        f"args=status --campaign {campaign}\n"
    )
    assert not (runtime / "locks").exists()


def test_help_and_shell_syntax() -> None:
    help_result = subprocess.run(
        [str(WRAPPER), "--help"], capture_output=True, text=True
    )
    syntax = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True)
    assert help_result.returncode == 0
    assert "prepare" in help_result.stdout
    assert syntax.returncode == 0
    assert os.access(WRAPPER, os.X_OK)
