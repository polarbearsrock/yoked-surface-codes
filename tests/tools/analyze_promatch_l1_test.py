from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT


def _load_tool():
    path = REPO_ROOT / "tools" / "analyze_promatch_l1"
    loader = importlib.machinery.SourceFileLoader("promatch_analysis_tool", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


TOOL = _load_tool()


def test_atomic_write_refuses_overwrite_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    destination = tmp_path / "analysis" / "result.json"
    TOOL._atomic_write(destination, b"first\n", overwrite=False)
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        TOOL._atomic_write(destination, b"second\n", overwrite=False)
    assert destination.read_bytes() == b"first\n"


def test_atomic_write_reuses_cross_filesystem_safe_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    destination = tmp_path / "analysis" / "result.json"
    destination.parent.mkdir()
    destination.write_bytes(b"old\n")

    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def cross_device_first(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        calls.append((str(src), str(dst)))
        if len(calls) == 1:
            raise OSError(18, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", cross_device_first)
    TOOL._atomic_write(destination, b"new\n", overwrite=True)
    assert destination.read_bytes() == b"new\n"
    assert len(calls) == 2
    assert Path(calls[1][0]).parent == destination.parent
    assert list(scratch.iterdir()) == []


def test_cli_rejects_output_nested_beside_file_form_latency_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    collection = tmp_path / "latency"
    collection.mkdir()
    suite = collection / "suite.json"
    suite.write_text("{}", encoding="utf-8")
    output = collection / "analysis"
    manifest = {"phase": "smoke", "claim_bearing": False}
    monkeypatch.setattr(TOOL, "_load_json", lambda _: manifest)
    monkeypatch.setattr(TOOL, "normalize_protocol", lambda value: value)
    monkeypatch.setattr(
        TOOL.sys,
        "argv",
        [
            "analyze_promatch_l1",
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--latency-input",
            str(suite),
            "--latency-cell",
            "cell",
            "--out",
            str(output),
            "--allow-non-scientific",
        ],
    )

    assert TOOL.main() == 2
    assert "must be separate, non-nested directories" in capsys.readouterr().err
    assert not output.exists()
