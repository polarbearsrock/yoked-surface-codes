"""Network-free integration tests for the GCP environment helpers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETUP = REPO_ROOT / "gcp" / "setup_environment"
ACTIVATE = REPO_ROOT / "gcp" / "activate_environment"


def _mini_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "clone with spaces"
    (repo / "gcp").mkdir(parents=True)
    (repo / "src").mkdir()
    shutil.copy2(SETUP, repo / "gcp" / "setup_environment")
    shutil.copy2(ACTIVATE, repo / "gcp" / "activate_environment")
    shutil.copy2(REPO_ROOT / ".python-version", repo / ".python-version")
    shutil.copy2(REPO_ROOT / ".gitignore", repo / ".gitignore")
    shutil.copy2(REPO_ROOT / "requirements.txt", repo / "requirements.txt")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=GCP Setup Test",
            "-c",
            "user.email=gcp-setup@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )
    return repo


def _install_fake_uv(runtime_root: Path, log_path: Path) -> None:
    uv = runtime_root / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ $1 == --version ]]; then
    echo 'uv 0.11.17 (x86_64-unknown-linux-gnu)'
    exit 0
fi
printf '%s|TMPDIR=%s|UV_CACHE_DIR=%s|UV_PYTHON_INSTALL_DIR=%s|UV_PYTHON_BIN_DIR=%s|UV_MANAGED_PYTHON=%s|MAX_ERRORS=%s\\n' \\
    "$*" "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \\
    "$UV_PYTHON_BIN_DIR" "$UV_MANAGED_PYTHON" "${MAX_ERRORS-unset}" >> "$FAKE_UV_LOG"
if [[ $1 == python && $2 == install ]]; then
    exit 0
fi
if [[ $1 == venv ]]; then
    target=${!#}
    mkdir -p "$target/bin"
    printf '%s\\n' \\
        'export VIRTUAL_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"' \\
        'export PATH="$VIRTUAL_ENV/bin:$PATH"' > "$target/bin/activate"
    cat > "$target/bin/python" <<'PYTHON'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n ${FAKE_PYTHON_LOG:-} ]]; then
    printf '%s\\n' "$*" >> "$FAKE_PYTHON_LOG"
fi
if [[ ${2:-} == *platform.python_version* ]]; then
    echo '3.14.5'
elif [[ ${2:-} == *sys.base_prefix* ]]; then
    echo "$UV_PYTHON_INSTALL_DIR/fake-cpython-3.14.5"
elif [[ ${1:-} == -m && ${2:-} == pytest && -n ${FAKE_PYTEST_CWD:-} ]]; then
    pwd > "$FAKE_PYTEST_CWD"
fi
PYTHON
    chmod +x "$target/bin/python"
    exit 0
fi
if [[ $1 == pip && $2 == install ]]; then
    exit 0
fi
echo "unexpected fake uv invocation: $*" >&2
exit 9
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    log_path.write_text("", encoding="utf-8")


def _run_setup(
    repo: Path,
    runtime_root: Path,
    log_path: Path,
    extra_environment: dict[str, str] | None = None,
    extra_arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["FAKE_UV_LOG"] = str(log_path)
    env["FAKE_PYTHON_LOG"] = str(log_path.with_suffix(".python.log"))
    env["MAX_ERRORS"] = "17"
    if extra_environment:
        env.update(extra_environment)
    return subprocess.run(
        [
            str(repo / "gcp" / "setup_environment"),
            "--runtime-root",
            str(runtime_root),
            *extra_arguments,
        ],
        cwd=runtime_root.parent,
        env=env,
        capture_output=True,
        text=True,
    )


def test_help_and_shell_syntax_have_no_setup_side_effects(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    help_result = subprocess.run(
        [str(SETUP), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    setup_syntax = subprocess.run(["bash", "-n", str(SETUP)], capture_output=True)
    activate_syntax = subprocess.run(["bash", "-n", str(ACTIVATE)], capture_output=True)

    assert help_result.returncode == 0
    assert "source gcp/activate_environment" in help_result.stdout
    assert help_result.stderr == ""
    assert set(tmp_path.iterdir()) == before
    assert setup_syntax.returncode == 0
    assert activate_syntax.returncode == 0
    assert os.access(SETUP, os.X_OK)
    assert os.access(ACTIVATE, os.X_OK)


def test_setup_is_portable_idempotent_and_keeps_runtime_outside_repo(
    tmp_path: Path,
) -> None:
    repo = _mini_repository(tmp_path)
    runtime_root = tmp_path / "persistent runtime"
    log_path = tmp_path / "uv.log"
    _install_fake_uv(runtime_root, log_path)

    first = _run_setup(repo, runtime_root, log_path)
    assert first.returncode == 0, first.stderr
    assert "GCP environment setup complete" in first.stdout
    repository_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert repository_commit in first.stdout
    assert (repo / ".venv" / "gcp-runtime-root").read_text().strip() == str(
        runtime_root
    )
    for relative in (
        "bin",
        "tmp",
        "uv-cache",
        "uv-python",
        "tmp/yoked-surface-codes-matplotlib",
        "runs",
    ):
        assert (runtime_root / relative).is_dir()

    first_log = log_path.read_text()
    assert "python install --no-bin 3.14.5" in first_log
    assert f"venv --python 3.14.5 {repo / '.venv'}" in first_log
    assert f"pip install --python {repo / '.venv/bin/python'}" in first_log
    assert f"TMPDIR={runtime_root / 'tmp'}" in first_log
    assert f"UV_CACHE_DIR={runtime_root / 'uv-cache'}" in first_log
    assert f"UV_PYTHON_BIN_DIR={runtime_root / 'bin'}" in first_log
    assert "UV_MANAGED_PYTHON=1" in first_log
    assert "MAX_ERRORS=unset" in first_log
    assert "import matplotlib, numpy" in log_path.with_suffix(".python.log").read_text()
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    second = _run_setup(repo, runtime_root, log_path)
    assert second.returncode == 0, second.stderr
    second_log = log_path.read_text()
    assert second_log.count("venv --python") == 1
    assert second_log.count("pip install --python") == 2


def test_setup_warns_about_a_dirty_worktree(tmp_path: Path) -> None:
    repo = _mini_repository(tmp_path)
    runtime_root = tmp_path / "persistent runtime"
    log_path = tmp_path / "uv.log"
    _install_fake_uv(runtime_root, log_path)
    (repo / "untracked.txt").write_text("operator note\n", encoding="utf-8")

    result = _run_setup(repo, runtime_root, log_path)

    assert result.returncode == 0, result.stderr
    assert "Git worktree is not clean" in result.stderr


def test_setup_runs_tests_from_repository_root(tmp_path: Path) -> None:
    repo = _mini_repository(tmp_path)
    runtime_root = tmp_path / "persistent runtime"
    log_path = tmp_path / "uv.log"
    pytest_cwd = tmp_path / "pytest-cwd.txt"
    _install_fake_uv(runtime_root, log_path)

    result = _run_setup(
        repo,
        runtime_root,
        log_path,
        {"FAKE_PYTEST_CWD": str(pytest_cwd)},
        ("--run-tests",),
    )

    assert result.returncode == 0, result.stderr
    assert pytest_cwd.read_text().strip() == str(repo)


def test_setup_bootstraps_pinned_uv_without_writing_home(tmp_path: Path) -> None:
    repo = _mini_repository(tmp_path)
    runtime_root = tmp_path / "persistent runtime"
    source_runtime = tmp_path / "fake uv source"
    log_path = tmp_path / "uv.log"
    _install_fake_uv(source_runtime, log_path)

    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    installer = tmp_path / "uv-installer.sh"
    installer.write_text(
        """#!/usr/bin/env sh
set -eu
cp "$FAKE_UV_SOURCE" "$UV_UNMANAGED_INSTALL/uv"
chmod 755 "$UV_UNMANAGED_INSTALL/uv"
""",
        encoding="utf-8",
    )
    curl_log = tmp_path / "curl.log"
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env sh
set -eu
printf '%s\\n' "$*" > "$FAKE_CURL_LOG"
cat "$FAKE_UV_INSTALLER"
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    empty_home = tmp_path / "empty home"
    empty_home.mkdir()

    result = _run_setup(
        repo,
        runtime_root,
        log_path,
        {
            "FAKE_CURL_LOG": str(curl_log),
            "FAKE_UV_INSTALLER": str(installer),
            "FAKE_UV_SOURCE": str(source_runtime / "bin/uv"),
            "HOME": str(empty_home),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Installing uv 0.11.17" in result.stdout
    assert (runtime_root / "bin/uv").is_file()
    assert "https://astral.sh/uv/0.11.17/install.sh" in curl_log.read_text()
    assert list(empty_home.iterdir()) == []


def test_setup_rejects_relative_or_worktree_runtime_roots(tmp_path: Path) -> None:
    repo = _mini_repository(tmp_path)

    relative = subprocess.run(
        [str(repo / "gcp" / "setup_environment"), "--runtime-root", "relative"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    inside = subprocess.run(
        [
            str(repo / "gcp" / "setup_environment"),
            "--runtime-root",
            str(repo / "runtime"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert relative.returncode != 0
    assert "absolute path" in relative.stderr
    assert inside.returncode != 0
    assert "outside the Git checkout" in inside.stderr
    assert not (repo / "runtime").exists()


def test_activation_exports_exact_controls_and_is_idempotent(tmp_path: Path) -> None:
    repo = _mini_repository(tmp_path)
    runtime_root = tmp_path / "persistent runtime"
    log_path = tmp_path / "uv.log"
    _install_fake_uv(runtime_root, log_path)
    setup_result = _run_setup(repo, runtime_root, log_path)
    assert setup_result.returncode == 0, setup_result.stderr

    env = dict(os.environ)
    env["MAX_ERRORS"] = "99"
    env["PYTHONPATH"] = "/existing"
    command = r"""
source "$1"
source "$1"
printf 'RESULT|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$YSC_GCP_RUNTIME_ROOT" "$YSC_GCP_RUNS_ROOT" "$TMPDIR" \
    "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" "$VIRTUAL_ENV" "$(command -v uv)" \
    "$OMP_NUM_THREADS,$OPENBLAS_NUM_THREADS,$MKL_NUM_THREADS,$NUMEXPR_NUM_THREADS,$VECLIB_MAXIMUM_THREADS,$BLIS_NUM_THREADS" \
    "$PROCESSES" "$THREADS_PER_PROCESS" "${MAX_ERRORS-unset}" "$PYTHONPATH"
"""
    activated = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
            "bash",
            str(repo / "gcp/activate_environment"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert activated.returncode == 0, activated.stderr
    result = next(
        line for line in activated.stdout.splitlines() if line.startswith("RESULT|")
    ).split("|")
    assert result[1] == str(runtime_root)
    assert result[2] == str(runtime_root / "runs")
    assert result[3] == str(runtime_root / "tmp")
    assert result[4] == str(runtime_root / "uv-python")
    assert result[5] == str(runtime_root / "bin")
    assert result[6] == str(repo / ".venv")
    assert result[7] == str(runtime_root / "bin/uv")
    assert result[8] == "1,1,1,1,1,1"
    assert result[9:12] == ["32", "1", "unset"]
    assert result[12].split(os.pathsep).count(str(repo / "src")) == 1
    assert activated.stdout.count("Activated yoked-surface-codes") == 2


def test_activate_must_be_sourced() -> None:
    direct = subprocess.run([str(ACTIVATE)], capture_output=True, text=True)

    assert direct.returncode == 2
    assert "source gcp/activate_environment" in direct.stderr
