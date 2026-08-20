"""Tests for the tools/collect_gap command line tool."""

import importlib.machinery
import importlib.util
import os
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _load_tool():
    path = REPO_ROOT / "tools" / "collect_gap"
    loader = importlib.machinery.SourceFileLoader("collect_gap_tool", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_import_forces_exactly_one_native_thread(monkeypatch):
    for name in NATIVE_THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")

    _load_tool()

    assert {os.environ[name] for name in NATIVE_THREAD_VARIABLES} == {"1"}
