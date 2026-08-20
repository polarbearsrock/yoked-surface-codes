# Test layout

The test tree mirrors the production packages under `src/`:

- `tests/gen/` covers circuit-generation utilities.
- `tests/yoked/` covers yoked circuits and gap collection.
- `tests/yoked/decoding/` covers the ProMatch core and experiment harness.
- `tests/yoked/decoding/oracle/` covers full-graph replay and the B1 policy
  audit.

Run the complete suite from the repository root with `python -m pytest -q`.
`pytest.ini` supplies the `src/` import path and uses importlib collection so
same-named test modules in different mirrored directories remain independent.

Tests may import private production helpers when they are checking a frozen
algorithmic invariant. Production modules must never import from this tree.

## Naming convention

New test files are named `<module>_test.py` after the production module they
cover (for example `full_graph_test.py` covers
`src/yoked/decoding/oracle/full_graph.py`). Do not introduce new
`test_<module>.py` names.

Shared helpers live in `conftest.py` files: `tests/conftest.py` provides the
repository-root anchor (`REPO_ROOT` constant and `repo_root` fixture) —
import it instead of hard-coding `Path(__file__).resolve().parents[N]` hop
counts.

## Provenance

`tests/gen/` mirrors the upstream "Yoked Surface Codes" paper release, and
its terse golden-output style is inherited from that codebase. The
`tests/yoked/decoding/` and `tests/yoked/decoding/oracle/` trees are
fork-authored for the ProMatch L1 predecoder experiment.
