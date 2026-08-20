"""Tests for the tools/plot_gap_distribution command line tool."""

import importlib.machinery
import importlib.util
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_tool():
    path = REPO_ROOT / 'tools' / 'plot_gap_distribution'
    sys.path.insert(0, str(path.parent))
    loader = importlib.machinery.SourceFileLoader('plot_gap_distribution_tool', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_emulate_unyoked_accepts_error_only_bins():
    tool = _load_tool()

    c_hits, e_hits = tool.split_gap_counts(
        {'C1': 2, 'E5': 3},
        emulate_unyoked_decoding=True,
    )

    assert c_hits == {1.0: 2, 5.0: 3}
    assert e_hits == {5.0: 0}
