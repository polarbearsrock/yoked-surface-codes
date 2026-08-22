# AWS c8a.48xlarge setup

This directory contains the dedicated Amazon EC2 workflow for the
parameterized paired Figure-8b sweep. It is intentionally narrow: the
scientific collector accepts a **Spot `c8a.48xlarge` in `us-east-1`**, with
exactly 192 physical CPUs arranged as two 96-CPU NUMA nodes and one native
numerical thread per worker. The ordinary workstation and Google Cloud paths
remain capped at 32 workers.

## Protect the data before starting

The campaign is batch-resumable only if its ledger files survive. `tmux`
protects a process from an SSH disconnect, but it does nothing when EC2
interrupts and terminates a Spot instance. The default EC2 root EBS volume is
commonly configured with `DeleteOnTermination=true`; if so, a Spot termination
deletes both a clone and a default sibling runtime directory.

Use a separately attached EBS data volume, or set `DeleteOnTermination=false`
for the volume containing the runtime root. EBS volumes are tied to an
availability zone, so record the instance's zone as well. If the AWS CLI is
installed and the instance role permits `DescribeInstances`, inspect the
actual mapping with:

```bash
IMDS_TOKEN="$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
INSTANCE_ID="$(curl -fsS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)"
REGION="$(curl -fsS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region)"
aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].BlockDeviceMappings[].{Device:DeviceName,Volume:Ebs.VolumeId,DeleteOnTermination:Ebs.DeleteOnTermination}'
```

Do not assume that a mounted path is durable merely because it is EBS. Verify
the volume ID and deletion flag in the EC2 console or with the command above.
Periodic copies of the campaign directory to an independently durable store
provide a second layer of protection; this repository does not choose or
configure an S3 bucket on your behalf.

## Fresh-clone quick start

Clone the pushed experiment branch onto the persistent volume when possible:

```bash
cd /mnt/ysc
git clone \
  --branch codex/fig8-1d-reproduction \
  --single-branch \
  https://github.com/polarbearsrock/yoked-surface-codes.git
cd yoked-surface-codes
```

Set up the pinned environment and put all mutable runtime data on the
persistent volume:

```bash
./aws/setup_environment \
  --runtime-root /mnt/ysc/yoked-surface-codes-aws-runtime \
  --run-tests
source aws/activate_environment
./aws/run_fig8_paired host-check
```

If the clone itself is on the persistent volume, omitting `--runtime-root`
uses the persistent sibling directory
`../yoked-surface-codes-aws-runtime`. If the clone is under the instance's
disposable root filesystem, pass an explicit path on the durable volume.

The setup script installs pinned `uv`, the exact Python version in
`.python-version`, and the exact direct dependencies from `requirements.txt`.
It records the runtime root inside the ignored `.venv`, configures all native
thread limits to one, and performs the authoritative AWS identity/topology
check. It is safe to rerun and performs no sampling.

## Create the paired sweep

The paired experiment samples each shot once and applies both decoders:

- `U0-direct`: uncorrelated joint PyMatching on the complete detector graph;
- `PU-window`: the current `d`-round, `HW=10`, zero-frame ProMatch-style L1
  predecoder followed by full-graph residual PyMatching.

It uses the fixed 16-cell Figure-8b grid: `d in {5,7,9,11}`, patch count in
`{6,10}`, and rounds in `{4d,8d}`. The SI1000 physical error rate and exact
paired shots per cell are parameters. For one million shots in each cell at
`p=0.001`:

```bash
./aws/run_fig8_paired create \
  --run-id p001-1m-aws-v1 \
  --p 0.001 \
  --shots-per-cell 1000000
```

Creation does not sample shots. It constructs and authenticates the circuits,
graphs, complete batch schedule, repository state, AWS identity, and worker
layout. A run ID is immutable and cannot be reused. Use a new run ID to change
`p` or the shot count.

## Run in tmux

Start a named session:

```bash
tmux new-session -s ysc-fig8-aws
```

Inside it, run:

```bash
cd ~/yoked-surface-codes
source aws/activate_environment
./aws/run_fig8_paired run --run-id p001-1m-aws-v1
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach-session -t ysc-fig8-aws
```

The launcher rechecks the AWS host immediately before collection and holds a
runtime-wide, non-blocking `collection-aws192.lock`. Do not launch any other
simulation outside this wrapper while the campaign is active; the lock cannot
detect unrelated commands.

The collector itself creates two process pools:

- pool 0: 96 workers pinned to NUMA node 0, CPUs 0-95;
- pool 1: 96 workers pinned to NUMA node 1, CPUs 96-191.

Each worker uses one native numerical thread. Worker affinity is applied
before decoder-graph compilation, allowing Linux first-touch placement to
keep most memory local to its node. Do **not** add an outer `numactl` command;
that would conflict with the collector's per-pool affinity policy.

Every completed 1,000-shot batch is written atomically under:

```text
$YSC_AWS_RUNS_ROOT/fig8-paired-aws192/p001-1m-aws-v1/collection/
```

If SSH disconnects, leave the tmux session alone. If the process is stopped or
the Spot instance is interrupted, attach the retained EBS volume to a
replacement `c8a.48xlarge` in the same availability zone, mount it at the same
canonical path, check out the exact campaign commit, rerun setup if necessary,
and issue the same `run` command. The collector authenticates existing ledgers
and schedules only missing batches. A replacement with a different frozen
environment is rejected instead of silently mixing results.

## Continue a Spot campaign on an On-Demand host

Do not edit `campaign.json`, retag ledger experiment IDs, or relax the original
Spot validator. Those operations would invalidate the seed derivation and make
the operational provenance ambiguous. The dedicated continuation path instead
keeps the original campaign and ledgers byte-for-byte unchanged and writes a
separate sibling record:

```text
$YSC_AWS_RUNS_ROOT/fig8-paired-aws192/.RUN_ID.ondemand-continuation-v1.json
```

The On-Demand host must be the same `c8a.48xlarge` shape, region, availability
zone, 2x96 CPU partition, software environment, kernel, CPU model, microcode,
and source state. Reported total/per-node usable memory may differ, but the
original 350/175 GiB safety minima remain mandatory. The record freezes the
new instance and AMI IDs, both host descriptions, the continuation commit, and
the hash of every ledger that existed at the transition.

After copying the repository and runtime to the same canonical paths, pull the
continuation commit and freeze the boundary once:

```bash
cd ~/yoked-surface-codes
source aws/activate_environment
./aws/run_fig8_paired_ondemand prepare --run-id p001-1m-aws-v1
./aws/run_fig8_paired_ondemand status --run-id p001-1m-aws-v1
```

`prepare` performs validation and writes provenance but samples no shots. Run
the continuation in `tmux`:

```bash
tmux new-session -s ysc-fig8-ondemand
cd ~/yoked-surface-codes
source aws/activate_environment
./aws/run_fig8_paired_ondemand run --run-id p001-1m-aws-v1
```

The wrapper holds the same machine-wide 192-worker lock as the Spot launcher.
The collector uses the original experiment ID, seed root, schedules, decoder,
and atomic ledger installer, and therefore schedules only missing original
batches. A changed baseline ledger, campaign manifest, host, continuation
checkout, or source hash fails closed.

## Progress, memory, and completion

The coordinator process can show low CPU while its 192 children are busy. Use
these read-only checks in another shell:

```bash
cd ~/yoked-surface-codes
source aws/activate_environment
./aws/run_fig8_paired status --run-id p001-1m-aws-v1
top
free -h
```

In `top`, press `1` for per-CPU utilization. A healthy saturated run should
show all 192 CPUs busy and a load average near 192. Watch available memory and
swap closely during the first distance-11 cell: each worker maintains compiled
decoder state, so 96 copies must fit within each NUMA node's roughly 190 GiB.
If the kernel starts swapping or invokes the OOM killer, stop the run and keep
the completed ledgers; do not weaken the frozen 2x96 campaign layout in place.

The status JSON reports completed and expected batches/shots for every cell.
After all cells are complete, generate the comparison plot and results table:

```bash
./aws/run_fig8_paired plot --run-id p001-1m-aws-v1
```

The plot command refuses incomplete or inconsistent collections. Outputs are
written under the campaign's `plots/` directory.

## Scope of the 192-worker exception

The AWS Spot launcher and its audited On-Demand continuation are the only
workflows authorized to exceed 32 processes. They are fixed to the exact
`c8a.48xlarge` host shape and exactly two 96-worker pools.
`gcp/run_fig8_paired`, `reproduce_fig8_1d`, the legacy gap collector, and the
general ProMatch tools keep their existing 32-process ceiling. These AWS paths
are a parameterized characterization sweep, not permission to run other
repository workloads with 192 workers.
