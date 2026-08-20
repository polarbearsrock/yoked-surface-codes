"""Tests for the tools/sinter_plot_print command line tool."""

import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TOOL = REPO_ROOT / 'tools' / 'sinter_plot_print'


def _env():
    # Ensure the venv's `sinter` CLI is resolvable from the child process.
    env = dict(os.environ)
    env['PATH'] = str(pathlib.Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    return env


def test_propagates_child_exit_code():
    proc = subprocess.run(
        [sys.executable, str(TOOL), '--bogus_flag_that_does_not_exist'],
        env=_env(),
        capture_output=True,
        text=True,
    )
    # `sinter plot` rejects unknown flags with argparse's exit code 2, and the
    # wrapper must propagate the child's exit code instead of flattening it.
    assert proc.returncode == 2


def test_help_prints_description():
    proc = subprocess.run(
        [sys.executable, str(TOOL), '--help'],
        env=_env(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert 'sinter plot' in proc.stdout
