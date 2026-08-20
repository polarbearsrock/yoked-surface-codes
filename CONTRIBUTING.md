# Contributing

Thanks for your interest in this project. This repository is a research
artifact: part of it reproduces published results under frozen protocols, so a
few rules here are stricter than in a typical library. Please read this page
before opening a pull request.

## Getting set up

Direct runtime and test dependencies are pinned in `requirements.txt` (Python
>= 3.12; validated with CPython 3.14.5). Their transitive dependencies are
resolved by [uv](https://docs.astral.sh/uv/) when you create a repo-local
virtual environment:

```bash
uv python install 3.14.5
uv venv --python 3.14.5 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The packages live under `src/` and import as `gen` and `yoked`. The explicit
`PYTHONPATH` makes them available to interactive Python sessions; `pytest.ini`
does the same for tests, and the scripts in `tools/` bootstrap `src/`
themselves. There is deliberately no `pyproject.toml`/wheel packaging yet.

## Running tests and lint

```bash
python -m pytest -q                    # full suite, ~30 s
uvx ruff==0.16.3 check .               # lint (configured by ruff.toml)
uvx ruff==0.16.3 format --check .      # prospective formatting baseline
```

All three must pass before a PR is ready. The formatter excludes the explicitly
listed vendored and source-hashed legacy files in `ruff.toml`; do not expand
that baseline when adding new code. Tests mirror `src/` under `tests/` and are
named `<module>_test.py`; see `tests/README.md` for the layout rules.

## Repository layout

- `src/gen/` — vendored circuit/geometry/flow framework from the upstream
  "Yoked Surface Codes" paper release. Fix real defects, but do not restyle or
  refactor it wholesale.
- `src/yoked/` — yoked-memory circuit constructions, the gap-collection
  harness, and this fork's ProMatch-L1 predecoder experiment
  (`src/yoked/decoding/`, with the oracle/policy-audit code in
  `src/yoked/decoding/oracle/`).
- `tools/` — command-line entry points (run with `--help`).
- `docs/` — the implementation plan and **frozen protocol manifests**
  (`docs/PROMATCH_*.json`).
- `experiments/` — experiment specs and status.

## Scientific-integrity rules (non-negotiable)

- **Never edit a frozen manifest.** Files matching `docs/PROMATCH_*FROZEN*.json`
  (and the other frozen protocol JSONs indexed in `docs/README.md`) are
  immutable records; their bytes authenticate completed experiments.
- **Do not bump dependency pins casually.** Frozen protocols record package
  versions; a different pin set is a different experiment. If you must change
  a pin, say so prominently in the PR description.
- **Protocol-governed collections** start from a clean worktree, keep
  `MAX_ERRORS` unset, and use the process count recorded by the protocol
  (currently exactly 32) with one native thread per worker. Non-claim smoke and
  debug runs may use fewer workers but never more than 32.
- **Never resume or overwrite a completed or retained results corpus.** Start a
  new campaign in a fresh directory. Resume an incomplete current campaign only
  through a tool's documented resume path, with the identical verified protocol
  and implementation.
- Code under `src/yoked/decoding/` participates in source-hash freezing
  (`*_SOURCE_PATHS` lists). If you add or move a module there, update those
  lists in the same PR.

## Pull requests

- Keep changes focused; separate mechanical cleanups from behavior changes.
- Behavior changes need a test that fails before the fix and passes after.
- Error messages in the decoding package are deliberately specific and
  fail-closed; keep that property.
