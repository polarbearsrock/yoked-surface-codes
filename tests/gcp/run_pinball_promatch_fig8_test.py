"""Hermetic tests for the GCP Pinball/ProMatch Figure-8 wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WRAPPER = REPO_ROOT / "gcp" / "run_pinball_promatch_fig8"


def _mini_repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "clone with spaces"
    runtime = tmp_path / "persistent runtime"
    log = tmp_path / "tool invocation.log"
    (repo / "gcp").mkdir(parents=True)
    (repo / "tools").mkdir()
    (runtime / "runs").mkdir(parents=True)
    shutil.copy2(WRAPPER, repo / "gcp" / "run_pinball_promatch_fig8")

    activation = repo / "gcp" / "activate_environment"
    activation.write_text(
        r"""#!/usr/bin/env bash
runtime=${FAKE_RUNTIME_ROOT:?}
export YSC_GCP_RUNTIME_ROOT="$runtime"
export YSC_GCP_RUNS_ROOT="$runtime/runs"
export TMPDIR="$runtime/tmp"
export OMP_NUM_THREADS=77
export OPENBLAS_NUM_THREADS=77
export MKL_NUM_THREADS=77
export NUMEXPR_NUM_THREADS=77
export VECLIB_MAXIMUM_THREADS=77
export BLIS_NUM_THREADS=77
export PROCESSES=77
export THREADS_PER_PROCESS=77
if [[ ${FAKE_LEAVE_MAX_ERRORS:-0} != 1 ]]; then
    unset MAX_ERRORS
fi
mkdir -p -- "$TMPDIR" "$YSC_GCP_RUNS_ROOT"
""",
        encoding="utf-8",
    )

    benchmark = repo / "tools" / "benchmark_pinball_promatch_fig8"
    benchmark.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
{
    printf 'tool=benchmark\n'
    printf 'argc=%s\n' "$#"
    index=0
    for argument in "$@"; do
        printf 'arg%s=%s\n' "$index" "$argument"
        index=$((index + 1))
    done
    printf 'threads=%s,%s,%s,%s,%s,%s\n' \
        "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" \
        "$NUMEXPR_NUM_THREADS" "$VECLIB_MAXIMUM_THREADS" "$BLIS_NUM_THREADS"
    printf 'controls=%s,%s,%s\n' \
        "${PROCESSES-unset}" "${THREADS_PER_PROCESS-unset}" "${MAX_ERRORS-unset}"
} > "$FAKE_TOOL_LOG"
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

    plot = repo / "tools" / "plot_pinball_promatch_fig8"
    plot.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
{
    printf 'tool=plot\n'
    printf 'argc=%s\n' "$#"
    index=0
    for argument in "$@"; do
        printf 'arg%s=%s\n' "$index" "$argument"
        index=$((index + 1))
    done
} > "$FAKE_TOOL_LOG"
""",
        encoding="utf-8",
    )
    for executable in (
        activation,
        benchmark,
        plot,
        repo / "gcp/run_pinball_promatch_fig8",
    ):
        executable.chmod(0o755)
    return repo, runtime, log


def _environment(runtime: Path, log: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["FAKE_RUNTIME_ROOT"] = str(runtime)
    env["FAKE_TOOL_LOG"] = str(log)
    return env


def _run(
    repo: Path,
    runtime: Path,
    log: Path,
    *arguments: str,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _environment(runtime, log)
    if extra_environment:
        env.update(extra_environment)
    return subprocess.run(
        [str(repo / "gcp/run_pinball_promatch_fig8"), *arguments],
        cwd=runtime,
        env=env,
        capture_output=True,
        text=True,
    )


def _logged_fields(log: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in log.read_text().splitlines())


def _create(
    repo: Path, runtime: Path, log: Path, run_id: str = "pbpm_001"
) -> Path:
    result = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        run_id,
        "--shots-per-cell",
        "100000",
    )
    assert result.returncode == 0, result.stderr
    return runtime / "runs" / "pinball-promatch-fig8-gcp32" / run_id


def test_help_syntax_and_executable_mode() -> None:
    help_result = subprocess.run(
        [str(WRAPPER), "--help"], capture_output=True, text=True
    )
    syntax = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True)

    assert help_result.returncode == 0
    assert "fixed at p=0.002" in help_result.stdout
    assert "create --run-id ID --shots-per-cell N" in help_result.stdout
    assert help_result.stderr == ""
    assert syntax.returncode == 0
    assert os.access(WRAPPER, os.X_OK)


def test_create_uses_fixed_p_campaign_path_and_exact_arguments(
    tmp_path: Path,
) -> None:
    repo, runtime, log = _mini_repository(tmp_path)

    result = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        "pbpm_001",
        "--shots-per-cell",
        "100000",
    )

    assert result.returncode == 0, result.stderr
    campaign = runtime / "runs/pinball-promatch-fig8-gcp32/pbpm_001"
    assert campaign.is_dir()
    fields = _logged_fields(log)
    assert [fields[f"arg{k}"] for k in range(5)] == [
        "create",
        "--out",
        str(campaign),
        "--shots-per-cell",
        "100000",
    ]


def test_run_reasserts_exact_limits_and_uses_shared_lock(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)

    result = _run(
        repo,
        runtime,
        log,
        "run",
        "--run-id",
        campaign.name,
        extra_environment={"MAX_ERRORS": "19"},
    )

    assert result.returncode == 0, result.stderr
    fields = _logged_fields(log)
    assert [fields[f"arg{k}"] for k in range(5)] == [
        "run",
        "--campaign",
        str(campaign),
        "--processes",
        "32",
    ]
    assert fields["threads"] == "1,1,1,1,1,1"
    assert fields["controls"] == "32,1,unset"
    assert (runtime / "locks/collection-32.lock").is_file()


def test_activation_that_leaves_max_errors_set_is_rejected(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)
    log.unlink()

    result = _run(
        repo,
        runtime,
        log,
        "run",
        "--run-id",
        campaign.name,
        extra_environment={"MAX_ERRORS": "19", "FAKE_LEAVE_MAX_ERRORS": "1"},
    )

    assert result.returncode != 0
    assert "MAX_ERRORS must be unset" in result.stderr
    assert not log.exists()


def test_run_fails_nonblocking_while_shared_collection_lock_is_held(
    tmp_path: Path,
) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)
    lock_directory = runtime / "locks"
    lock_directory.mkdir()
    lock_path = lock_directory / "collection-32.lock"
    holder = subprocess.Popen(
        [
            "flock",
            str(lock_path),
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            "echo ready; read -r ignored",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        log.unlink()

        result = _run(repo, runtime, log, "run", "--run-id", campaign.name)

        assert result.returncode != 0
        assert "another 32-worker collection is already running" in result.stderr
        assert not log.exists()
    finally:
        holder.terminate()
        holder.communicate(timeout=5)


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "two/components", ".", "..", "has.dot", "white space", "-dash"],
)
def test_run_id_must_be_one_safe_component(tmp_path: Path, run_id: str) -> None:
    repo, runtime, log = _mini_repository(tmp_path)

    result = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        run_id,
        "--shots-per-cell",
        "1",
    )

    assert result.returncode != 0
    assert "run ID must match" in result.stderr
    assert not (runtime / "escape").exists()
    assert not log.exists()


def test_symlinked_campaign_or_parent_is_rejected(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    parent = runtime / "runs/pinball-promatch-fig8-gcp32"
    parent.mkdir()
    (parent / "linked").symlink_to(external, target_is_directory=True)

    linked_campaign = _run(repo, runtime, log, "status", "--run-id", "linked")
    assert linked_campaign.returncode != 0
    assert "campaign may not be a symbolic link" in linked_campaign.stderr

    (parent / "linked").unlink()
    parent.rmdir()
    parent.symlink_to(external, target_is_directory=True)
    linked_parent = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        "new_run",
        "--shots-per-cell",
        "1",
    )
    assert linked_parent.returncode != 0
    assert "campaign parent may not be a symbolic link" in linked_parent.stderr


def test_create_is_fresh_only_but_run_is_resumable(tmp_path: Path) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)

    duplicate = _run(
        repo,
        runtime,
        log,
        "create",
        "--run-id",
        campaign.name,
        "--shots-per-cell",
        "100000",
    )
    first_resume = _run(repo, runtime, log, "run", "--run-id", campaign.name)
    second_resume = _run(repo, runtime, log, "run", "--run-id", campaign.name)

    assert duplicate.returncode != 0
    assert "fresh run ID" in duplicate.stderr
    assert first_resume.returncode == 0, first_resume.stderr
    assert second_resume.returncode == 0, second_resume.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((), "Usage:"),
        (("unknown",), "unknown action"),
        (("run",), "run requires --run-id"),
        (("create", "--run-id", "x"), "create requires --shots-per-cell"),
        (
            (
                "create",
                "--run-id",
                "x",
                "--shots-per-cell",
                "1",
                "--p",
                "0.002",
            ),
            "unknown argument '--p'",
        ),
        (("status", "--run-id", "x", "--shots-per-cell", "1"), "accepts only"),
        (("run", "--run-id", "x", "--overwrite"), "does not accept"),
        (("plot", "--bad"), "unknown argument"),
    ],
)
def test_action_specific_requirements(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    repo, runtime, log = _mini_repository(tmp_path)

    result = _run(repo, runtime, log, *arguments)

    assert result.returncode != 0
    assert message in result.stderr
    assert not log.exists()


def test_status_and_plot_forward_exact_campaign_without_collection_lock(
    tmp_path: Path,
) -> None:
    repo, runtime, log = _mini_repository(tmp_path)
    campaign = _create(repo, runtime, log)
    before = {path.relative_to(campaign) for path in campaign.rglob("*")}

    status = _run(repo, runtime, log, "status", "--run-id", campaign.name)
    assert status.returncode == 0, status.stderr
    fields = _logged_fields(log)
    assert [fields[f"arg{k}"] for k in range(3)] == [
        "status",
        "--campaign",
        str(campaign),
    ]
    assert not (runtime / "locks").exists()

    plot = _run(
        repo,
        runtime,
        log,
        "plot",
        "--run-id",
        campaign.name,
        "--overwrite",
    )
    assert plot.returncode == 0, plot.stderr
    fields = _logged_fields(log)
    assert fields == {
        "tool": "plot",
        "argc": "3",
        "arg0": "--campaign",
        "arg1": str(campaign),
        "arg2": "--overwrite",
    }
    after = {path.relative_to(campaign) for path in campaign.rglob("*")}
    assert after == before
