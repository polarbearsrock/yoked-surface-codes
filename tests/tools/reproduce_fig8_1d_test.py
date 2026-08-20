"""Integration tests for the reproduce_fig8_1d shell entry point."""

import os
import pathlib
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / 'reproduce_fig8_1d'


def _environment(tmp_path: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    env['TMPDIR'] = str(tmp_path)
    env['MPLCONFIGDIR'] = str(tmp_path / 'mplconfig')
    return env


def test_help_succeeds_without_runtime_setup(tmp_path):
    proc = subprocess.run(
        [str(SCRIPT), '--help'],
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert 'Usage:' in proc.stdout
    assert proc.stderr == ''


def test_rejects_more_than_one_native_thread(tmp_path):
    env = _environment(tmp_path)
    env['THREADS_PER_PROCESS'] = '2'
    proc = subprocess.run(
        [str(SCRIPT), 'plot-paper-data', str(tmp_path / 'out')],
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert 'THREADS_PER_PROCESS must be exactly 1' in proc.stderr


def test_unset_max_errors_uses_only_fixed_shot_limit(tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    captured_args = tmp_path / 'sinter-args.txt'
    fake_sinter = fake_bin / 'sinter'
    fake_sinter.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURED_ARGS"\n')
    fake_sinter.chmod(0o755)

    output = tmp_path / 'out'
    circuits = output / 'validation_grid' / 'circuits'
    circuits.mkdir(parents=True)
    (circuits / 'example.stim').write_text('')

    env = _environment(tmp_path)
    env.pop('MAX_ERRORS', None)
    env['CAPTURED_ARGS'] = str(captured_args)
    env['PATH'] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    proc = subprocess.run(
        [str(SCRIPT), 'collect-open-validation-grid', str(output)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    args = captured_args.read_text().splitlines()
    assert '--max_shots' in args
    assert '--max_errors' not in args
