"""Hermetic tests for the dedicated AWS paired Figure-8 wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WRAPPER = REPO_ROOT / "aws" / "run_fig8_paired"


def _mini_repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "clone with spaces"
    runtime = tmp_path / "persistent runtime"
    log = tmp_path / "tool invocations.log"
    (repo / "aws").mkdir(parents=True)
    (repo / "tools").mkdir()
    (runtime / "runs").mkdir(parents=True)
    shutil.copy2(WRAPPER, repo / "aws" / "run_fig8_paired")

    activation = repo / "aws" / "activate_environment"
    activation.write_text(
        r"""#!/usr/bin/env bash
runtime=${FAKE_RUNTIME_ROOT:?}
export YSC_AWS_RUNTIME_ROOT="$runtime"
export YSC_AWS_RUNS_ROOT="$runtime/runs"
export TMPDIR="$runtime/tmp"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PROCESSES=192
export THREADS_PER_PROCESS=1
export YSC_AWS_NUMA_POOLS=2
export YSC_AWS_PROCESSES_PER_POOL=96
unset MAX_ERRORS
mkdir -p -- "$TMPDIR" "$YSC_AWS_RUNS_ROOT"
""",
        encoding="utf-8",
    )
    benchmark = repo / "tools" / "benchmark_fig8_paired_aws"
    benchmark.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
{
    printf 'BEGIN\n'
    printf 'argc=%s\n' "$#"
    index=0
    for argument in "$@"; do
        printf 'arg%s=%s\n' "$index" "$argument"
        index=$((index + 1))
    done
    printf 'threads=%s,%s,%s,%s,%s,%s\n' \
        "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" \
        "$NUMEXPR_NUM_THREADS" "$VECLIB_MAXIMUM_THREADS" "$BLIS_NUM_THREADS"
    printf 'controls=%s,%s,%s,%s,%s\n' \
        "${PROCESSES-unset}" "${THREADS_PER_PROCESS-unset}" \
        "${YSC_AWS_NUMA_POOLS-unset}" "${YSC_AWS_PROCESSES_PER_POOL-unset}" \
        "${MAX_ERRORS-unset}"
} >> "$FAKE_TOOL_LOG"
if [[ ${1:-} == create ]]; then
    shift
    while (($#)); do
        if [[ $1 == --out ]]; then
            mkdir -- "$2"
            printf '{}\n' > "$2/campaign.json"
            exit 0
        fi
        shift
    done
    exit 9
fi
""",
        encoding="utf-8",
    )
    plot = repo / "tools" / "plot_fig8_paired"
    plot.write_text(
        r"""#!/usr/bin/env bash
printf 'plot=%s\n' "$*" >> "$FAKE_TOOL_LOG"
""",
        encoding="utf-8",
    )
    for path in (activation, benchmark, plot, repo / "aws/run_fig8_paired"):
        path.chmod(0o755)
    return repo, runtime, log


def _run(
    repo: Path, runtime: Path, log: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(FAKE_RUNTIME_ROOT=str(runtime), FAKE_TOOL_LOG=str(log))
    return subprocess.run(
        [str(repo / "aws/run_fig8_paired"), *arguments],
        cwd=runtime,
        env=environment,
        capture_output=True,
        text=True,
    )


def _invocations(log: Path) -> list[dict[str, str]]:
    blocks = log.read_text().split("BEGIN\n")
    return [
        dict(line.split("=", 1) for line in block.splitlines())
        for block in blocks
        if block
    ]


def _create(repo: Path, runtime: Path, log: Path, run_id: str = "aws_001") -> Path:
    result = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        run_id,
        "--p",
        "0.001",
        "--shots-per-cell",
        "10000",
    )
    assert result.returncode == 0, result.stderr
    return runtime / "runs/fig8-paired-aws192" / run_id


def test_help_syntax_and_executable_mode() -> None:
    help_result = subprocess.run([str(WRAPPER), "--help"], capture_output=True, text=True)
    syntax = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True)
    assert help_result.returncode == 0
    assert "host-check" in help_result.stdout
    assert "2x96" not in help_result.stderr
    assert syntax.returncode == 0
    assert os.access(WRAPPER, os.X_OK)


def test_create_checks_host_then_forwards_exact_campaign_arguments(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)
    calls = _invocations(log)
    assert calls[0]["arg0"] == "host-check"
    assert [calls[1][f"arg{index}"] for index in range(7)] == [
        "create",
        "--out",
        str(campaign),
        "--p",
        "0.001",
        "--shots-per-cell",
        "10000",
    ]


def test_run_rechecks_host_and_forwards_only_exact_192_layout(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)
    log.write_text("", encoding="utf-8")

    result = _run(repo, runtime, log, "run", "--run-id", campaign.name)

    assert result.returncode == 0, result.stderr
    host_check, run = _invocations(log)
    assert host_check["arg0"] == "host-check"
    assert [run[f"arg{index}"] for index in range(5)] == [
        "run",
        "--campaign",
        str(campaign),
        "--processes",
        "192",
    ]
    assert run["threads"] == "1,1,1,1,1,1"
    assert run["controls"] == "192,1,2,96,unset"
    assert (runtime / "locks/collection-aws192.lock").is_file()


@pytest.mark.parametrize("run_id", ["../escape", "two/components", ".", "has.dot", "-dash"])
def test_run_id_is_one_safe_component(tmp_path: Path, run_id: str) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    result = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        run_id,
        "--p",
        "0.001",
        "--shots-per-cell",
        "1",
    )
    assert result.returncode != 0
    assert "run ID must match" in result.stderr
    assert not log.exists()


def test_status_and_plot_do_not_run_host_check_or_take_collection_lock(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)
    log.write_text("", encoding="utf-8")

    status = _run(repo, runtime, log, "status", "--run-id", campaign.name)
    assert status.returncode == 0, status.stderr
    calls = _invocations(log)
    assert [calls[0][f"arg{index}"] for index in range(3)] == [
        "status",
        "--campaign",
        str(campaign),
    ]
    assert not (runtime / "locks").exists()

    log.write_text("", encoding="utf-8")
    plot = _run(repo, runtime, log, "plot", "--run-id", campaign.name)
    assert plot.returncode == 0, plot.stderr
    assert log.read_text().strip() == f"plot=--campaign {campaign}"
