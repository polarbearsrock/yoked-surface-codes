# CLAUDE.md

Project instructions for Claude Code in `/data2/s2chitni/yoked-surface-codes`.
The full, authoritative guidance (dependency installation with `uv`, disk
rules, the 32-process / 1-thread simulation limit, and run commands) lives in
`AGENTS.md` and is imported here so both agents read one source:

@AGENTS.md

Non-negotiables, restated for quick reference:

- Install/run only from the repo `.venv` built with `uv` from
  `requirements.txt`; everything (venv, uv cache, managed Python, `TMPDIR`,
  outputs) stays under `/data2/s2chitni`. Home is at quota.
- Every simulation: at most **32 processes**, **1 native thread per process**
  (`PROCESSES=32 THREADS_PER_PROCESS=1`, `--processes 32`, and the six
  `*_NUM_THREADS=1` variables exported).
- Activate the venv (`source .venv/bin/activate`) before running
  `reproduce_fig8_1d` or `tools/*`; they resolve `python3`/`sinter` from
  `PATH`.
- Do not modify or reuse existing `out/promatch_l1_round1*` corpora; write new
  results to a fresh directory under `$TMPDIR` or `out/<new-name>`.
- Keep `MAX_ERRORS` unset for ProMatch scientific runs.
