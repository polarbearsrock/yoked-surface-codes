from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT
from yoked.decoding._artifact_io import (
    IMMUTABLE_OUTPUT_PATTERNS,
    THREAD_ENVIRONMENT,
    install_bytes_atomic,
    is_lowercase_hex,
    load_json_artifact,
    load_json_strict,
    reject_json_constant,
    repo_root,
    unique_json_object,
    validate_resumable_output_root,
)


def test_thread_environment_is_the_six_pinned_names() -> None:
    assert THREAD_ENVIRONMENT == (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )


def test_immutable_output_pattern_is_shared() -> None:
    assert IMMUTABLE_OUTPUT_PATTERNS == ("promatch_l1_round1",)


def test_unique_json_object_rejects_duplicate_keys() -> None:
    assert unique_json_object([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="duplicate JSON object key 'a'"):
        unique_json_object([("a", 1), ("a", 2)])


def test_reject_json_constant_raises_for_every_nonfinite_literal() -> None:
    for literal in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="nonfinite JSON constant"):
            reject_json_constant(literal)


def test_load_json_artifact_rejects_duplicates_and_nonfinite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_json_artifact(path)
    path.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON constant"):
        load_json_artifact(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON artifact"):
        load_json_artifact(path)
    path.write_text('[1, 2]', encoding="utf-8")
    assert load_json_artifact(path) == [1, 2]


def test_load_json_strict_requires_one_regular_object_file(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert load_json_strict(path, description="latency artifact") == {"a": 1}

    array_path = tmp_path / "array.json"
    array_path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain one JSON object"):
        load_json_strict(array_path, description="latency artifact")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        load_json_strict(tmp_path / "missing.json", description="latency artifact")

    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular non-symlink file"):
        load_json_strict(link, description="latency artifact")

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read strict JSON latency input"):
        load_json_strict(broken, description="latency input")


def test_is_lowercase_hex_checks_type_length_and_charset() -> None:
    assert is_lowercase_hex("0f" * 32, length=64)
    assert is_lowercase_hex("abc123")
    assert not is_lowercase_hex("ABC123")
    assert not is_lowercase_hex("0f" * 32, length=32)
    assert not is_lowercase_hex("zz")
    assert not is_lowercase_hex(64)
    assert not is_lowercase_hex(b"0f" * 32, length=64)


def test_repo_root_finds_the_git_marker() -> None:
    root = repo_root()
    assert (root / ".git").exists()
    assert root == REPO_ROOT
    assert repo_root(Path(__file__)) == root


def test_repo_root_fails_without_marker(tmp_path: Path) -> None:
    orphan = tmp_path / "a" / "b"
    orphan.mkdir(parents=True)
    resolved = orphan.resolve()
    if any(
        (candidate / ".git").exists() for candidate in (resolved, *resolved.parents)
    ):
        pytest.skip("temporary directory unexpectedly lives inside a git checkout")
    with pytest.raises(RuntimeError, match="no .git repository marker"):
        repo_root(orphan)


def test_validate_resumable_output_root_rejects_unsafe_or_unrecognized_state(
    tmp_path: Path,
) -> None:
    fresh, names = validate_resumable_output_root(
        tmp_path / "fresh",
        allowed_entries={"protocol.json"},
        description="test output",
    )
    assert fresh == (tmp_path / "fresh").resolve()
    assert names == frozenset()

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected entries"):
        validate_resumable_output_root(
            unrelated,
            allowed_entries={"protocol.json"},
            description="test output",
        )

    target = tmp_path / "target"
    target.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="root may not be a symlink"):
        validate_resumable_output_root(
            linked_root,
            allowed_entries=set(),
            description="test output",
        )

    recognized = tmp_path / "recognized"
    recognized.mkdir()
    artifact = recognized / "protocol.json"
    artifact.write_text("{}", encoding="utf-8")
    link = recognized / "linked.json"
    link.symlink_to(artifact)
    with pytest.raises(ValueError, match="entries may not be symlinks"):
        validate_resumable_output_root(
            recognized,
            allowed_entries={"protocol.json", "linked.json"},
            description="test output",
        )

    for component in (
        "promatch_l1_round1",
        "promatch_l1_round1_20260817_32p",
        "promatch_l1_round1_v3",
        "promatch_l1_round1_v3_20260817_32p",
    ):
        immutable = tmp_path / component / "nested-output"
        with pytest.raises(ValueError, match="immutable round-one corpus"):
            validate_resumable_output_root(
                immutable,
                allowed_entries=set(),
                description="test output",
            )

    for component in (
        "promatch_l1_round10",
        "promatch_l1_round1_notes",
        "prefix-promatch_l1_round1",
    ):
        permitted = tmp_path / component / "nested-output"
        validated, permitted_names = validate_resumable_output_root(
            permitted,
            allowed_entries=set(),
            description="test output",
        )
        assert validated == permitted.resolve()
        assert permitted_names == frozenset()


def test_install_bytes_atomic_stages_in_tmpdir_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    destination = tmp_path / "out" / "value.json"
    install_bytes_atomic(destination, b'{"value": 1}\n')
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 1}
    # The scratch directory is created for staging and left empty afterwards.
    assert list(scratch.iterdir()) == []


def test_install_bytes_atomic_works_without_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TMPDIR", raising=False)
    destination = tmp_path / "out" / "value.json"
    install_bytes_atomic(destination, b'{"value": 2}\n')
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 2}
    # Staging happened next to the destination and left no residue behind.
    assert {entry.name for entry in destination.parent.iterdir()} == {"value.json"}


def test_install_bytes_atomic_overwrites_and_survives_exdev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    destination = tmp_path / "out" / "value.json"
    install_bytes_atomic(destination, b"first\n")
    install_bytes_atomic(destination, b"second\n")
    assert destination.read_bytes() == b"second\n"

    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def cross_device_first(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        calls.append((str(src), str(dst)))
        if len(calls) == 1:
            raise OSError(18, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", cross_device_first)
    install_bytes_atomic(destination, b"third\n")
    assert destination.read_bytes() == b"third\n"
    assert len(calls) == 2
    # The fallback restaged next to the destination for a same-fs rename.
    assert Path(calls[1][0]).parent == destination.parent
    assert list(scratch.iterdir()) == []
    assert {entry.name for entry in destination.parent.iterdir()} == {"value.json"}


def test_install_bytes_atomic_no_clobber_is_atomic_and_survives_exdev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    destination = tmp_path / "out" / "value.json"
    install_bytes_atomic(destination, b"first\n", overwrite=False)
    with pytest.raises(FileExistsError):
        install_bytes_atomic(destination, b"second\n", overwrite=False)
    assert destination.read_bytes() == b"first\n"

    destination.unlink()
    real_link = os.link
    calls: list[tuple[str, str]] = []

    def cross_device_first(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        calls.append((str(src), str(dst)))
        if len(calls) == 1:
            raise OSError(18, "Invalid cross-device link")
        return real_link(src, dst)

    monkeypatch.setattr(os, "link", cross_device_first)
    install_bytes_atomic(destination, b"third\n", overwrite=False)
    assert destination.read_bytes() == b"third\n"
    assert len(calls) == 2
    assert Path(calls[1][0]).parent == destination.parent
    assert list(scratch.iterdir()) == []
    assert {entry.name for entry in destination.parent.iterdir()} == {"value.json"}
