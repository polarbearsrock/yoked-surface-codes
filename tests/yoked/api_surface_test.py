"""Locks down the public API surface of the ``yoked`` packages.

Checks that each package's ``__all__`` is a well-formed, fully resolvable
export list and that the ``yoked.decoding:custom_decoders`` sinter entry
point resolves. The exact decoder-name set returned by ``custom_decoders``
is already asserted by
``tests/yoked/decoding/_promatch_decoder_test.py``; this module does not
repeat that assertion.
"""

import importlib
import os
import subprocess
import sys

import pytest
import sinter

from tests.conftest import REPO_ROOT

PACKAGES_WITH_EXPORTS = [
    "yoked",
    "yoked.decoding",
    "yoked.decoding.oracle",
    "yoked.gap",
]


@pytest.mark.parametrize("package_name", PACKAGES_WITH_EXPORTS)
def test_package_imports_and_every_export_resolves(package_name: str) -> None:
    module = importlib.import_module(package_name)
    names = module.__all__
    assert names == sorted(names), f"{package_name}.__all__ is not sorted"
    assert len(names) == len(set(names)), f"{package_name}.__all__ has duplicates"
    for name in names:
        assert not name.startswith("_"), f"{package_name} exports private {name!r}"
        getattr(module, name)  # Raises AttributeError if the export is broken.


def test_star_import_of_yoked_decoding_matches_all() -> None:
    namespace: dict[str, object] = {}
    exec("from yoked.decoding import *", namespace)
    module = importlib.import_module("yoked.decoding")
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(module.__all__)


def test_custom_decoders_entry_point_resolves_to_sinter_decoders() -> None:
    # Resolve exactly the way `sinter collect --custom-decoders-module-function
    # yoked.decoding:custom_decoders` does: import the module, look up the
    # attribute, call it.
    module_name, _, function_name = "yoked.decoding:custom_decoders".partition(":")
    factories = getattr(importlib.import_module(module_name), function_name)()
    assert isinstance(factories, dict)
    assert factories, "custom_decoders() returned no decoders"
    for name, decoder in factories.items():
        assert isinstance(name, str)
        assert isinstance(decoder, sinter.Decoder)
    # Same mapping as the package-level export.
    import yoked.decoding

    assert set(factories) == set(yoked.decoding.custom_decoders())


def test_oracle_package_import_stays_lazy() -> None:
    """`import yoked.decoding.oracle` must not eagerly import its submodules.

    The package docstring promises this so that Matplotlib and the heavy
    oracle/collection modules stay out of callers that only need the parent
    decoding package. Checked in a fresh interpreter so previously imported
    test modules cannot mask an eager import.
    """
    code = (
        "import sys\n"
        "import yoked.decoding.oracle\n"
        "eager = sorted(m for m in sys.modules"
        " if m.startswith('yoked.decoding.oracle.'))\n"
        "assert not eager, f'eagerly imported: {eager}'\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
