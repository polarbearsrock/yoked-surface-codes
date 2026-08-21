# Google Cloud setup

This directory contains the portable Google Cloud workflow. The setup scripts
create the pinned Python environment, and `run_fig8_paired` creates, resumes,
inspects, and plots the parameterized paired Figure-8b sweep.

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
- `YSC_GCP_RUNS_ROOT` for persistent campaign data;
- all six native numerical thread limits to `1`;
- `PROCESSES=32` and `THREADS_PER_PROCESS=1`; and
- fixed-shot collection by unsetting `MAX_ERRORS`.

The environment setup does not start simulations, modify frozen protocols, or
create Git commits. A dirty worktree produces a setup warning and is a hard
error when creating a paired sweep campaign.

## Parameterized paired Figure-8b sweep

The paired sweep evaluates two decoders on every identical sampled shot:

- `U0-direct`: ordinary, uncorrelated joint PyMatching on the full detector
  graph; and
- `PU-window`: the current `d`-round, `HW=10`, zero-frame ProMatch-style L1
  predecoder followed by the same full-graph residual PyMatching decoder.

The geometry grid is fixed to the 16 Figure-8b full-circuit cells
`d in {5,7,9,11}`, `patches in {6,10}`, and `r in {4d,8d}`. The two campaign
parameters are the SI1000 physical error rate and the exact number of paired
shots in *each* cell. For example, create a one-million-shot-per-cell campaign
at `p=0.001`:

```bash
./gcp/run_fig8_paired create \
  --run-id p001-1m-v1 \
  --p 0.001 \
  --shots-per-cell 1000000
```

Creation performs no sampling, but it does construct all 16 circuits and
decoder graphs once so it can freeze their fingerprints. It also freezes the
parameter, repository, environment, seed derivation, and complete 1,000-shot
batch schedule in the campaign manifest. A run ID is one safe path component
and may not be reused or overwritten.

Start or resume collection with exactly 32 processes and one native numerical
thread per process:

```bash
./gcp/run_fig8_paired run --run-id p001-1m-v1
```

The launcher holds a runtime-wide non-blocking lock, so a second 32-process
collection cannot accidentally run concurrently. Each completed batch is
written atomically below
`$YSC_GCP_RUNS_ROOT/fig8-paired/p001-1m-v1/collection/`. Reissuing the same
`run` command after an SSH disconnect, process failure, or spot-VM restart
authenticates the frozen campaign and skips valid completed batches. Do not
create a replacement campaign to resume an existing run.

Inspect progress at any time:

```bash
./gcp/run_fig8_paired status --run-id p001-1m-v1
```

After all 16 cells have their exact requested shot count, generate the paired
side-by-side logical-error-rate plot and results table:

```bash
./gcp/run_fig8_paired plot --run-id p001-1m-v1
```

Outputs are written under the campaign's `plots/` directory. The plot command
refuses incomplete or inconsistent collections instead of silently plotting a
partial sweep.

`p` and `shots-per-cell` are immutable once `create` succeeds. Use a new run ID
for a different value of either parameter. The SI1000 input must satisfy
`0 < p <= 0.2`; the upper bound follows from its measurement-error probability
`5p <= 1`. Creation also compiles every matching graph and rejects a value for
which this decoder cannot form a valid nonnegative-weight graph. The current
v1 launcher accepts at most 1,000,000 shots per cell; larger campaigns should
use a newly reviewed storage/batching protocol instead of creating millions of
small ledger files.

This workflow is a parameterized, non-claim-bearing characterization sweep. It
uses the released Figure-8b full-circuit generator, not the unreleased
long-history gap simulator behind Figure 8d. See
[`../experiments/PROMATCH_FIG8_PAIRED_GCP_SWEEP.md`](../experiments/PROMATCH_FIG8_PAIRED_GCP_SWEEP.md)
for the complete experiment contract and interpretation.
