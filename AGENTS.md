# AGENTS.md — working in `yoked-surface-codes` on this workstation

Instructions for coding agents (Claude Code, Codex, etc.) and humans. Read
this before installing anything or launching a simulation. `CLAUDE.md`
imports this file; keep the two in sync by editing only this one.

## 1. What is here

- Code and data for the paper "Yoked Surface Codes" (`src/gen`, `src/yoked`,
  `assets/`, legacy `step*`/`gap_step*` scripts, `make_paper_figures`).
- This fork's Figure 8 1D-yoked reproduction workflow: `reproduce_fig8_1d`
  and `REPRODUCING_FIG8_1D.md`.
- The L1 ProMatch-style predecoder experiment: `src/yoked/decoding/`,
  `tools/benchmark_promatch_l1`, `tools/analyze_promatch_l1`,
  `docs/PROMATCH_IMPLEMENTATION_PLAN.md`, and the frozen protocols
  `docs/PROMATCH_*.json`. Sinter decoder names are registered by
  `yoked.decoding:custom_decoders`.
- Python packages live under `src/`; scripts either add `src` to
  `sys.path` themselves or (for `reproduce_fig8_1d`) export `PYTHONPATH`.
- Generated outputs go to `out/` (git-ignored). Existing pilot corpora under
  `out/promatch_l1_round1*` are immutable audit artifacts: never edit,
  resume, or "promote" them.

## 2. Hard rules on this machine

1. **Disk.** The home directory (`/homes/s2chitni`, a.k.a. `/home/s2chitni`)
   is at its quota. Every dependency, cache, interpreter, scratch file, and
   output must live under `/data2/s2chitni`:
   - repository: `/data2/s2chitni/yoked-surface-codes`
   - virtual environment: `/data2/s2chitni/yoked-surface-codes/.venv`
   - uv cache: `/data2/s2chitni/.cache/uv` (pinned by the repo `uv.toml`)
   - uv-managed Pythons: `/data2/s2chitni/.local/share/uv/python`
   - scratch: `TMPDIR=/data2/s2chitni/.tmp` (never `~` or `/tmp`)
   - Matplotlib cache: `MPLCONFIGDIR="$TMPDIR/yoked-surface-codes-matplotlib"`
   Never `pip install --user`, never install into the system Python, and never
   let uv fall back to `~/.cache/uv` (that fails with "Disk quota exceeded").
2. **Parallelism.** Simulations run with **at most 32 worker processes and
   exactly one native numerical thread per process**, even though the host
   exposes 128 logical CPUs. Concretely:
   - `reproduce_fig8_1d`: `PROCESSES=32 THREADS_PER_PROCESS=1` (values above
     32 are rejected by the script).
   - `tools/benchmark_promatch_l1 ...`: `--processes 32` (hard-capped; the
     harness also forces one native thread in the parent and every worker).
     Scientific ProMatch collection requires exactly 32.
   - `tools/collect_gap`: `--processes N` with `1 <= N <= 32`; `auto` now
     means `min(cpu_count, 32)`. Never pass more than 32.
   - Direct `sinter collect`: pass `--processes 32` and export the thread
     variables below first.
   - Do not run two collection campaigns at once if their process counts sum
     to more than 32.
   Always export, before importing NumPy/PyMatching or launching workers:
   ```bash
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
          NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
   ```
   `reproduce_fig8_1d` and the ProMatch harness set these themselves;
   exporting them in the shell is still the safe default.
3. **ProMatch scientific runs** keep `MAX_ERRORS` unset (fixed-`N` paired
   design), use `--processes 32`, and must start from a clean worktree at a
   commit whose only change over the implementation HEAD is the frozen
   protocol JSON. Untracked files (including new notes) make the worktree
   dirty; commit or stash them before a claim-bearing collection.

## 3. Dependency management: `uv` + pinned `requirements.txt`

There is no `pyproject.toml`, Poetry, or Conda environment. Dependencies are
the exact pins in `requirements.txt` (Python >= 3.12; validated and frozen
with CPython 3.14.5, `stim==1.16.0`, `pymatching==2.4.0`, `sinter==1.16.0`,
`numpy==2.5.1`, `scipy==1.18.0`, `matplotlib==3.11.1`, `pytest==9.1.1`,
`pygltflib==1.16.5`). Install them with **uv** (>= 0.11 is on `PATH`) into the
repo `.venv`. Repo files that make this reproducible:

- `uv.toml` — pins `cache-dir = "/data2/s2chitni/.cache/uv"`.
- `.python-version` — `3.14`, so `uv venv` selects CPython 3.14.
- `requirements.txt` — the pinned package set. Do not change pins casually:
  the frozen ProMatch protocols record these versions and a mismatch is a
  different experiment.

### One-time setup

```bash
export TMPDIR=/data2/s2chitni/.tmp
export UV_CACHE_DIR=/data2/s2chitni/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data2/s2chitni/.local/share/uv/python
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

cd /data2/s2chitni/yoked-surface-codes
uv python install 3.14                     # managed CPython under /data2 (idempotent)
uv venv .venv                              # honours .python-version + UV_PYTHON_INSTALL_DIR
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -c 'import stim, pymatching, sinter, numpy, scipy; \
print(stim.__version__, pymatching.__version__, sinter.__version__, numpy.__version__, scipy.__version__)'
# expected: 1.16.0 2.4.0 1.16.0 2.5.1 1.18.0
```

`cat .venv/pyvenv.cfg` must show `home = /data2/s2chitni/.local/share/uv/python/...`.
If it points into `/homes/s2chitni`, recreate the venv with
`UV_PYTHON_INSTALL_DIR` exported as above.

### Every session

```bash
cd /data2/s2chitni/yoked-surface-codes
source .venv/bin/activate          # tools use '#!/usr/bin/env python3' and 'sinter' from PATH
export TMPDIR=/data2/s2chitni/.tmp
export MPLCONFIGDIR="$TMPDIR/yoked-surface-codes-matplotlib"; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
unset MAX_ERRORS
```

Activation matters: `reproduce_fig8_1d` shells out to the `sinter` CLI and
`tools/*` start with `#!/usr/bin/env python3`, so both must resolve to
`.venv/bin`. When you cannot activate (subprocess, cron), call
`.venv/bin/python tools/<script>` explicitly and prefix
`PATH=/data2/s2chitni/yoked-surface-codes/.venv/bin:$PATH` for the shell
script.

### Adding or upgrading a dependency

Edit `requirements.txt` with an exact pin, rerun the `uv pip install` line,
run the test suite, and mention in the commit that frozen protocol version
hashes will no longer match. `threadpoolctl` is an optional extra the harness
uses if present; it is intentionally not pinned.

## 4. Verifying the environment

```bash
python -m pytest -q                                   # full suite (~30 s; 394 tests)
python -m pytest -q src/yoked/decoding                # ProMatch tests only
```

Integration smoke for the predecoder (non-claim-bearing; writes only under
`$TMPDIR`):

```bash
DECODER=promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching \
MAX_SHOTS=10000 PROCESSES=32 THREADS_PER_PROCESS=1 \
./reproduce_fig8_1d smoke "$TMPDIR/yoked-promatch-smoke"
```

## 5. Running simulations (cheat sheet)

| Task | Command (after "Every session" setup) |
| --- | --- |
| Replot paper panels, no sampling | `./reproduce_fig8_1d plot-paper-data` |
| Small full-circuit smoke | `PROCESSES=32 THREADS_PER_PROCESS=1 ./reproduce_fig8_1d smoke [OUT_DIR]` |
| Generate Fig. 8b grid | `./reproduce_fig8_1d generate-validation-grid [OUT_DIR]` |
| Collect grid (resumable) | `PROCESSES=32 THREADS_PER_PROCESS=1 MAX_SHOTS=1000000 ./reproduce_fig8_1d collect-open-validation-grid [OUT_DIR]` |
| Collect grid with predecoder | add `DECODER=promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching` |
| ProMatch paired smoke | `tools/benchmark_promatch_l1 smoke --out "$TMPDIR/promatch-smoke" --processes 32` |
| ProMatch pilot/confirm/target/latency | see `REPRODUCING_FIG8_1D.md` §"Paired ProMatch experiment workflow"; always `--processes 32`, `MAX_ERRORS` unset |
| Legacy gap collection | `tools/collect_gap ... --processes 32` (never `auto`; legacy `step*`/`gap_step*` scripts also need GNU `parallel`, which is not installed here) |
| Diagnose PU-vs-U0 disagreements | `tools/diagnose_promatch_l1 replay --input <collection dir> [--cell ID]` (bit-exact replay of retained regressions) and `tools/diagnose_promatch_l1 probe --protocol P.json --cell ID --shots N --seed S` or `probe --d D --patches N --rounds R --p P ...` (single process; read-only) |
| Phase-A global-context oracle replay | `tools/diagnose_promatch_l1 oracle-replay --config docs/PROMATCH_ORACLE_REPLAY_FROZEN_V1.json --out out/promatch_l1_global_context_oracle_v1/replay` (single process, one native thread, retained shots only; no fresh sampling) |

`OUT_DIR` defaults to `out/fig8_1d`; anything scientific should be written to
a fresh directory under `$TMPDIR` or a new `out/<name>` so existing corpora
stay untouched.

## 6. Where results and diagnostics live

- `out/promatch_l1_round1_v3_20260817_32p/pilot/summary.json` — V3 pilot
  paired tables, unconditional telemetry, and up to 100 replayable
  regression/recovery/rollback samples per cell (detection events, actual
  observables, U0/PU predictions, batch seed).
- `out/promatch_l1_round1_v3_20260817_32p/pilot_analysis/` — analyzer output.
- V1/V2 directories are superseded diagnostic corpora (see
  `docs/PROMATCH_IMPLEMENTATION_PLAN.md` §15.2); do not reuse them.
- Diagnosis of the V3 result (2026-08-17): PU-window loses to U0-direct because
  the L1 domain graph omits yoke-hub, true-boundary, cross-window and terminal
  edges, so "isolated/safe/singleton" are judged on a truncated graph; a wrong
  commit is then turned into a two-observable logical error by yoke re-routing
  in the residual matcher. `tools/diagnose_promatch_l1 replay` reproduces the
  evidence from `summary.json`.
