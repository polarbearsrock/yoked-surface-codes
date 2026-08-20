"""Tests for the tools/gen_memory_circuit command line tool."""

import importlib.machinery
import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_tool():
    path = REPO_ROOT / 'tools' / 'gen_memory_circuit'
    loader = importlib.machinery.SourceFileLoader('gen_memory_circuit_tool', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_debug_writes_distinct_ideal_circuit(tmp_path, monkeypatch):
    tool = _load_tool()
    monkeypatch.setattr(sys, 'argv', [
        'gen_memory_circuit',
        '--patch_diameter', '3',
        '--rounds', 'd',
        '--noise_strength', '0.001',
        '--patches', '1',
        '--yokes', '0',
        '--gateset', 'cz',
        '--out_dir', str(tmp_path),
        '--debug',
    ])
    tool.main()
    noisy = (tmp_path / 'debug_noisy.html').read_text()
    ideal = (tmp_path / 'debug_ideal.html').read_text()
    # The ideal viewer must show the noiseless circuit, not a copy of the
    # noisy one.
    assert noisy != ideal
