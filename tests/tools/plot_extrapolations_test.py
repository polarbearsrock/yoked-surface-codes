"""Tests for the tools/plot_extrapolations command line tool."""

import os
import pathlib
import subprocess
import sys

import sinter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TOOL = REPO_ROOT / 'tools' / 'plot_extrapolations'


def _run(*args, tmp_path):
    env = dict(os.environ)
    env.setdefault('MPLBACKEND', 'Agg')
    env.setdefault('MPLCONFIGDIR', str(tmp_path / 'mplconfig'))
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        env=env,
        capture_output=True,
        text=True,
    )


def test_help_documents_decoder_option(tmp_path):
    proc = _run('--help', tmp_path=tmp_path)
    assert proc.returncode == 0
    assert '--decoder' in proc.stdout
    assert 'sparse_blossom_correlated' in proc.stdout


def test_decoder_option_admits_other_decoders(tmp_path):
    stat = sinter.TaskStats(
        strong_id='f' * 64,
        decoder='pymatching',
        json_metadata={'d': 3, 'r': 12, 'p': 0.001, 'patches': 1, 'yokes': 0},
        shots=1000,
        errors=10,
    )
    csv_path = tmp_path / 'stats.csv'
    csv_path.write_text(sinter.CSV_HEADER + '\n' + stat.to_csv_line() + '\n')
    out_path = tmp_path / 'plot.png'
    proc = _run(
        str(csv_path),
        '--decoder', 'pymatching',
        '--out', str(out_path),
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
