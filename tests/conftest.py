"""Shared pytest configuration for the whole test tree.

The single repo-root anchor lives here so individual test modules do not
each hard-code a ``Path(__file__).resolve().parents[N]`` hop count that
silently breaks when a file moves. Import the constant at module scope
(``from tests.conftest import REPO_ROOT``) or request the ``repo_root``
fixture inside a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/conftest.py lives one directory below the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root directory as an absolute path."""
    return REPO_ROOT
