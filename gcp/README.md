# Google Cloud setup

This directory contains the portable Google Cloud workflows. The setup scripts
create the pinned Python environment. `run_fig8_paired` owns the two-arm
U0/ProMatch sweep, while `run_pinball_promatch_fig8` owns the fixed-`p=0.002`
three-arm U0/ProMatch/Pinball sweep.

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

For the memory-intensive Pinball campaign, the reference AMD shape is
`n4d-custom-32-524288-ext`: 32 AMD EPYC Turin vCPUs and 512 GiB of memory.
Provision a 200 GiB Hyperdisk Balanced boot disk so comfortably more than
100 GiB remains after the OS, environment, formatting, and logs. N4D requires
gVNIC and Hyperdisk. Use an on-demand instance for the first production
characterization; changing to a replacement VM can alter the execution
identity frozen in the campaign manifest. Google documents the
[N4D family](https://cloud.google.com/compute/docs/general-purpose-machines)
and [extended-memory custom syntax](https://cloud.google.com/compute/docs/instances/creating-instance-with-custom-machine-type).

Place the clone on the persistent disk when possible. The default runtime root
is `../yoked-surface-codes-runtime`, adjacent to the clone. To use a particular
mounted disk, pass it explicitly.

### Reference N4D provisioning template

Authenticate the Cloud SDK and select a billed project before using this
template. Confirm N4D availability, regional CPU/Hyperdisk quota, the network,
and the estimated price in the Cloud console. The following command is a
template and is not run by any repository script:

```bash
GCP_PROJECT_ID=your-project-id
GCP_ZONE=us-central1-a
GCP_INSTANCE=ysc-pinball-promatch

gcloud compute instances create "$GCP_INSTANCE" \
  --project="$GCP_PROJECT_ID" \
  --zone="$GCP_ZONE" \
  --machine-type=n4d-custom-32-524288-ext \
  --provisioning-model=STANDARD \
  --boot-disk-size=200GB \
  --boot-disk-type=hyperdisk-balanced \
  --no-boot-disk-auto-delete \
  --image-project=ubuntu-os-cloud \
  --image-family=ubuntu-2404-lts-amd64 \
  --network-interface=network=default,nic-type=GVNIC
```

If the project has no default VPC, replace `network=default` with the approved
subnet. Keeping boot-disk auto-delete disabled makes experiment files
recoverable after accidental instance deletion, but the operator remains
responsible for snapshots, access controls, and storage charges. After SSH:

```bash
sudo mkdir -p /mnt/ysc
sudo chown "$USER:$USER" /mnt/ysc
```

Clone the exact pushed experiment commit below `/mnt/ysc`; do not create a
campaign from a working tree with local edits.

The VM shape, RAM, disk size, and provisioning model are operational
recommendations verified by the operator; the launcher does not query or
freeze those GCP resource fields. It does enforce the 32-worker/thread policy,
and the campaign separately freezes the observed CPU, kernel, microcode, and
affinity identity.

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
create Git commits. A dirty worktree produces a setup warning. Campaign
`create`, `run`, and `plot` reject a dirty checkout; read-only `status` remains
available for diagnosis.

## Native Pinball/ProMatch 32-worker sweep

This workflow samples each physical shot once and evaluates three arms:

- U0-direct complete-graph PyMatching;
- native ProMatch followed by complete-graph residual PyMatching; and
- native domain-atomic Pinball V2 followed by the same residual backend.

It fixes `p=0.002` and the 16-cell Figure-8 grid. The only creation parameter
is the exact shot count in every cell. Campaign data lives at
`$YSC_GCP_RUNS_ROOT/pinball-promatch-fig8-gcp32/<run-id>/`.

After committing and pushing the complete implementation, run a disposable
functional shakeout:

```bash
./gcp/run_pinball_promatch_fig8 create \
  --run-id p002-pb-pm-shakeout-1k-v1 \
  --shots-per-cell 1000
./gcp/run_pinball_promatch_fig8 run \
  --run-id p002-pb-pm-shakeout-1k-v1
./gcp/run_pinball_promatch_fig8 status \
  --run-id p002-pb-pm-shakeout-1k-v1
./gcp/run_pinball_promatch_fig8 plot \
  --run-id p002-pb-pm-shakeout-1k-v1
```

One 1,000-shot batch does not saturate 32 workers. Use a separate 32,000-shot-
per-cell campaign before production, interrupt and resume it once, and measure
PSS/private-dirty memory and swap activity on the largest
`d=11, patches=10, rounds=88` cell. Do not promote either shakeout directory.

After the saturated gate passes, create the planned 100,000-shot campaign:

```bash
./gcp/run_pinball_promatch_fig8 create \
  --run-id p002-pb-pm-100k-gcp32-v1 \
  --shots-per-cell 100000
./gcp/run_pinball_promatch_fig8 run \
  --run-id p002-pb-pm-100k-gcp32-v1
```

The run is resumable in 1,000-shot atomic ledgers and must use exactly 32
workers. It acquires the same runtime-wide `collection-32.lock` as the older
GCP collector, so the two workflows cannot overlap. `status` is read-only and
does not require the lock. A completed campaign can be analyzed with:

```bash
./gcp/run_pinball_promatch_fig8 plot \
  --run-id p002-pb-pm-100k-gcp32-v1
```

Add `--overwrite` only when intentionally replacing an already-generated
analysis after the completed campaign revalidates. See
[`../experiments/PINBALL_PROMATCH_FIG8_PAIRED_32.md`](../experiments/PINBALL_PROMATCH_FIG8_PAIRED_32.md)
for the scientific contract, telemetry, resume rules, and interpretation
boundary.

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
