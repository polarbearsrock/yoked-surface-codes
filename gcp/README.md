# Google Cloud setup

This directory contains the portable Google Cloud workflow. The current first
step creates and activates the pinned Python environment; a separate run
launcher will be added after the setup path has been reviewed.

## Fresh-clone quick start

Clone the pushed experiment branch onto the VM's persistent disk:

```bash
cd /mnt/ysc
git clone \
  --branch codex/fig8-1d-reproduction \
  --single-branch \
  https://github.com/polarbearsrock/yoked-surface-codes.git
cd yoked-surface-codes
```

Then set up and activate the environment:

```bash
./gcp/setup_environment --run-tests
source gcp/activate_environment
```

With this layout, the default runtime root is the persistent sibling directory
`/mnt/ysc/yoked-surface-codes-runtime`; no extra path argument is needed.

## VM and storage assumptions

- Bash on Linux `x86_64`.
- A persistent filesystem with enough space for scratch data and results.
- `curl` and standard Unix utilities. The setup script installs its own pinned
  `uv`; no system Python installation is required.
- Scientific ProMatch collection still uses exactly 32 worker processes and
  one native numerical thread per worker.

Place the clone on the persistent disk when possible. The default runtime root
is `../yoked-surface-codes-runtime`, adjacent to the clone. To use a particular
mounted disk, pass it explicitly.

## One-time setup

From the repository root:

```bash
./gcp/setup_environment
```

Or choose an explicit persistent runtime location:

```bash
./gcp/setup_environment --runtime-root /mnt/ysc/runtime
```

The script:

1. installs pinned `uv` under the runtime root if necessary;
2. installs the exact Python version from `.python-version`;
3. creates the ignored repository-local `.venv`;
4. installs the exact direct dependencies from `requirements.txt`;
5. checks the Git commit and warns if the worktree is dirty;
6. verifies exact package versions and the scientific Python imports; and
7. records the runtime root inside `.venv` for future activation and run
   scripts.

It is idempotent. Pass `--run-tests` to run the full test suite after setup.
It will never silently replace an existing `.venv` using a different Python
version.

## Every login or new shell

From the repository root:

```bash
source gcp/activate_environment
```

Activation configures:

- the repository `.venv` and `PYTHONPATH`;
- `TMPDIR`, uv caches, managed Python, and Matplotlib state under the recorded
  runtime root;
- `YSC_GCP_RUNS_ROOT` for future launch scripts;
- all six native numerical thread limits to `1`;
- `PROCESSES=32` and `THREADS_PER_PROCESS=1`; and
- fixed-shot collection by unsetting `MAX_ERRORS`.

The environment setup does not start simulations, modify frozen protocols, or
create Git commits. A dirty worktree produces a setup warning and will be a
hard error in the future scientific launcher. Existing workstation-frozen
scientific protocols remain machine-specific; a future GCP launcher must use a
GCP-frozen protocol.
