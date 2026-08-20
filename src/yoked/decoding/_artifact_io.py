"""Shared artifact-I/O infrastructure for the ProMatch experiment modules.

This module is shared infrastructure with no scientific logic: it holds only
generic file/JSON plumbing (atomic file installs, strict JSON loading, the
pinned native-thread environment names, hex-digest validation, and repository
root discovery) used by the collection, analysis, and latency modules.
Nothing here decides what an experiment accepts, samples, or reports, and it
imports no decoding, statistics, or experiment code.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
from typing import Any


__all__ = [
    "IMMUTABLE_OUTPUT_PATTERNS",
    "THREAD_ENVIRONMENT",
    "install_bytes_atomic",
    "is_lowercase_hex",
    "load_json_artifact",
    "load_json_strict",
    "reject_json_constant",
    "reject_immutable_output_path",
    "repo_root",
    "unique_json_object",
    "validate_resumable_output_root",
]


# The six native-thread environment variables every simulation on this host
# pins to "1" (see AGENTS.md section 2).
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# Historical paired round-one corpora are immutable audit evidence.  Keep the
# path fence in shared I/O infrastructure so every writer applies the same
# rule instead of relying on CLI documentation or a caller remembering it.
# The value names an exact path-component family, not an arbitrary substring:
# nearby destinations such as ``promatch_l1_round10`` and
# ``promatch_l1_round1_notes`` are unrelated and remain writable.
IMMUTABLE_OUTPUT_PATTERNS = ("promatch_l1_round1",)

_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


def is_lowercase_hex(value: Any, *, length: int | None = None) -> bool:
    """Return whether ``value`` is a lowercase hexadecimal string.

    ``length`` additionally pins the exact character count (64 for SHA-256).
    This is a predicate, not a validator: callers keep their own error
    messages so each validation site's semantics stay local and exact.
    """

    return (
        isinstance(value, str)
        and (length is None or len(value) == length)
        and all(ch in _LOWERCASE_HEX_DIGITS for ch in value)
    )


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` rejecting duplicate keys instead of overwriting."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    """``parse_constant`` hook rejecting the NaN/Infinity/-Infinity literals."""

    raise ValueError(f"nonfinite JSON constant {value!r}")


def load_json_artifact(path: Path) -> Any:
    """Load a JSON artifact of any top-level shape with strict parsing.

    Duplicate object keys and nonfinite number literals are rejected;
    malformed JSON is reported as ``ValueError`` naming the artifact path.
    """

    try:
        with path.open(encoding="utf-8") as f:
            return json.load(
                f,
                object_pairs_hook=unique_json_object,
                parse_constant=reject_json_constant,
            )
    except json.JSONDecodeError as ex:
        raise ValueError(f"invalid JSON artifact {path}: {ex}") from ex


def load_json_strict(path: Path, *, description: str = "artifact") -> dict[str, Any]:
    """Load exactly one strict JSON object from a regular, non-symlink file.

    ``description`` names the artifact kind in every error message so callers
    keep their established diagnostics.  Duplicate object keys and nonfinite
    number literals are rejected during parsing.
    """

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(
                source,
                object_pairs_hook=unique_json_object,
                parse_constant=reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as ex:
        raise ValueError(f"cannot read strict JSON {description} {path}") from ex
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return value


def repo_root(start: Path | None = None) -> Path:
    """Locate the enclosing repository root by searching upward for ``.git``.

    ``.git`` may be a directory or (in linked worktrees) a file; either marks
    the root.  Searching upward instead of counting fixed ``parents[N]``
    levels keeps the resolution correct if a module moves within the tree.
    """

    origin = (Path(__file__) if start is None else Path(start)).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"no .git repository marker above {origin}")


def reject_immutable_output_path(
    path: str | os.PathLike[str], *, description: str
) -> Path:
    """Resolves ``path`` after rejecting historical round-one destinations."""

    candidate = Path(path).absolute()
    resolved = candidate.resolve()

    def is_immutable_component(component: str) -> bool:
        for family in IMMUTABLE_OUTPUT_PATTERNS:
            if component == family:
                return True
            if not component.startswith(family + "_"):
                continue
            first_suffix_component = component[len(family) + 1 :].split("_", 1)[0]
            if (
                first_suffix_component.startswith("v")
                and first_suffix_component[1:].isdigit()
            ) or (
                len(first_suffix_component) == 8
                and first_suffix_component.isdigit()
            ):
                return True
        return False

    if any(
        is_immutable_component(component)
        for spelling in (candidate, resolved)
        for component in spelling.parts
    ):
        raise ValueError(
            f"{description} may not write into an immutable round-one corpus"
        )
    return resolved


def validate_resumable_output_root(
    path: str | os.PathLike[str],
    *,
    allowed_entries: set[str] | frozenset[str],
    description: str,
) -> tuple[Path, frozenset[str]]:
    """Validates a fresh or recognized-partial experiment output directory.

    The directory itself and every existing top-level entry must be real
    non-symlinks.  Existing names are limited to ``allowed_entries`` and paths
    naming a historical round-one corpus are rejected even when the requested
    output is a not-yet-created descendant.  Callers retain responsibility for
    validating the semantic ordering and contents of their recognized partial
    artifacts.
    """

    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise ValueError(f"{description} root may not be a symlink")
    resolved = reject_immutable_output_path(candidate, description=description)
    if not candidate.exists():
        return resolved, frozenset()
    if not candidate.is_dir():
        raise ValueError(f"{description} root must be a directory")

    entries = tuple(candidate.iterdir())
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    if symlinks:
        raise ValueError(
            f"{description} entries may not be symlinks: {symlinks}"
        )
    special = sorted(
        entry.name for entry in entries if not entry.is_file() and not entry.is_dir()
    )
    if special:
        raise ValueError(
            f"{description} entries must be regular files or directories: {special}"
        )
    names = frozenset(entry.name for entry in entries)
    unexpected = sorted(names - frozenset(allowed_entries))
    if unexpected:
        raise ValueError(
            f"{description} root contains unexpected entries: {unexpected}"
        )
    return resolved, names


def install_bytes_atomic(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    prefix: str = "promatch-artifact-",
    suffix: str = "",
    overwrite: bool = True,
) -> None:
    """Atomically install ``payload`` at ``path`` (tmp + fsync + rename).

    The scratch file is staged in ``$TMPDIR`` when set (keeping staging off
    the output filesystem, as on this quota-limited host) and otherwise next
    to the destination, which guarantees a same-filesystem rename.  If the
    staged rename crosses filesystems (EXDEV), the completed bytes are
    restaged next to the destination so the final operation is still atomic.
    With ``overwrite=False``, a hard-link install gives atomic no-clobber
    semantics instead of a racy existence check.  The destination directory
    is fsynced best-effort afterwards.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = os.environ.get("TMPDIR")
    if scratch_root:
        stage_dir = Path(scratch_root)
        stage_dir.mkdir(parents=True, exist_ok=True)
    else:
        stage_dir = destination.parent

    def write_stage(directory: Path) -> str:
        fd, temporary_name = tempfile.mkstemp(
            prefix=prefix, suffix=suffix, dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            os.unlink(temporary_name)
            raise
        return temporary_name

    temporary_name = write_stage(stage_dir)

    def install_stage(stage_name: str) -> None:
        if overwrite:
            os.replace(stage_name, destination)
        else:
            os.link(stage_name, destination)

    try:
        try:
            install_stage(temporary_name)
        except OSError as ex:
            if ex.errno != errno.EXDEV:
                raise
            # Cross-device staging: restage the completed bytes next to the
            # destination so the final install is same-filesystem and atomic.
            fallback_name = write_stage(destination.parent)
            try:
                install_stage(fallback_name)
            finally:
                if os.path.exists(fallback_name):
                    os.unlink(fallback_name)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
