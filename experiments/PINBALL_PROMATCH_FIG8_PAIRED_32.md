# Paired Pinball/ProMatch Figure-8 experiment (32 workers)

## 1. Scientific question

At the fixed SI1000 physical error probability `p=0.002`, how do the current
native Pinball V2 and ProMatch integrations change logical accuracy and
residual-matching workload in the maintained 1D-yoked surface-code stack?

This is an end-to-end YSC integration comparison, not an algorithm-isolation
experiment. The two predecoders intentionally keep their native domains,
visibility, commit rules, and observable-frame policies.

Every physical shot is sampled once and decoded by all three arms:

1. `U0-direct`: uncorrelated PyMatching on the complete original DEM.
2. `ProMatch-native`: the registered `windowd`, `HW=10`, boundary-disabled,
   zero-frame ProMatch policy followed by complete-graph residual PyMatching.
3. `Pinball-v2-native`: full-history, nine-stage, `(patch,basis)`-domain-atomic,
   yoke-edge Pinball V2 followed by complete-graph residual PyMatching.

The direct Pinball-versus-ProMatch contrast is paired on the exact detector and
observable arrays. U0 anchors whether either integration helps or harms the
same complete-graph baseline.

The campaign is a frozen, non-claim-bearing characterization sweep. A formal
confirmatory claim requires a separately frozen pilot/holdout protocol; a
production corpus must not be relabeled as a holdout after its results are
examined.

## 2. Frozen circuit grid

The campaign fixes all circuit parameters:

```text
p                  0.002
d                  {5, 7, 9, 11}
patches            {6, 10}
rounds             {4d, 8d}
yokes              2 (one X and one Z)
style              CZ
noise              SI1000
remove_x_yoke      false
DEM                decompose_errors=true
                   approximate_disjoint_errors=true
```

This is 16 cells. One million shots per cell means 16 million sampled circuits
and three predictions for every sample.

## 3. Paired outcomes

Each batch ledger records the complete eight-cell correctness cube in arm
order `(U0, ProMatch, Pinball)`, using `0=correct` and `1=wrong`. It also
records three reconciled paired tables:

- `pinball_minus_promatch`;
- `pinball_minus_u0`;
- `promatch_minus_u0`.

For `pinball_minus_promatch`, define

```text
b = #(ProMatch correct, Pinball wrong)
c = #(ProMatch wrong, Pinball correct)
delta = (b - c) / N = LER_Pinball - LER_ProMatch
```

The paired imbalance, its confidence interval, and the raw `b,c` counts are
the primary accuracy description. Marginal LERs alone are insufficient.

The ledger additionally records exact prediction agreement. Two arms can both
be logically wrong while predicting different observable frames, so agreement
is not inferred from correctness alone.

## 4. Telemetry and replay

Common telemetry includes original event weight, per-arm residual weight,
joint before/after histograms, activation, frame activity, and workload
ratios. ProMatch retains its native domain/stage/rollback telemetry. Pinball
telemetry includes:

- simple and complex `(patch,basis)` domains;
- all-simple, mixed, and no-commit shots;
- per-domain initial, tentative, and final residual weight;
- tentative and durable graph-edge/physical-correction support;
- match counts aggregated over `M`, `B1`--`B4`, `ST1`--`ST2`, `H`, and `E`;
- terminal and yoke residual events; and
- observable-frame activity.

Per-shot decoder records are aggregated in bounded microbatches so a worker
does not retain 1,000 large full-history Pinball results at once.

Collection also bounds compiled-state memory one cell at a time. The
single-threaded parent compiles and authenticates one incomplete cell, then
starts a short-lived 32-worker `fork` pool whose children inherit that
read-mostly `PreparedCell` through copy-on-write. After all missing batches for
the cell are installed, the pool exits and the parent releases the preload
before advancing. Workers must hit the inherited cache and do not independently
compile 32 copies. A cell that is already complete is neither compiled nor
forked during resume.

Every ledger retains a bounded, deterministic lowest-hash set of replayable
shots for Pinball wins, ProMatch wins, prediction disagreements, ProMatch
rollbacks, and Pinball-complex cases. Replay rows carry the packed syndrome,
actual observables, and all three predictions. The frozen source, graph, and
Pinball-schedule fingerprints authenticate re-execution.

## 5. Immutable campaign and resume model

`create` performs no sampling. It requires a clean Git worktree and compiles
all 16 circuits and both treatment decoders before writing `campaign.json`.
This compilation is compute- and memory-intensive at the largest cell, so run
`create` from a durable `tmux` session as well. At the atomic publication
point, the final path changes from absent to complete. An interruption before
publication leaves the run ID retryable; an interruption after publication can
leave a complete campaign that must be validated and used instead of replaced.
The manifest freezes:

- the repository commit and source-file hashes;
- exact package and execution environment;
- the fixed grid, decoder policies, and DEM options;
- circuit and DEM hashes;
- U0, ProMatch, and Pinball graph/layout provenance;
- the Pinball schedule and upstream-source fingerprints;
- a random 256-bit sampler seed root; and
- every fixed 1,000-shot batch range.

A run ID is immutable. Changing the shot count requires a fresh run ID.
`MAX_ERRORS` must remain unset: collection never stops on errors, significance,
or a favorable trend.

Completed batches are installed atomically under:

```text
<campaign>/collection/batches/<cell-id>/batch-XXXXXXXX.json
```

Resume validates every existing ledger and schedules only missing declared
batches. Unknown, malformed, duplicated, or provenance-inconsistent artifacts
fail closed.

## 6. Remote resource profile and manual launch gate

Both launchers run exactly 32 worker processes with one native numerical
thread per worker. The GCP launcher uses the repository-wide
`collection-32.lock`; the AWS launcher holds both its campaign lock and the
legacy AWS192 lock. Do not start another simulation in the same runtime while
either collector is active.

The recommended GCP deployment is an on-demand AMD N4D custom VM with 32 vCPUs
and 512 GiB of memory (`n4d-custom-32-524288-ext`) plus a 200 GiB durable
Hyperdisk Balanced boot disk, leaving comfortably more than 100 GiB after the
OS, environment, formatting, and logs. The shape keeps the 32-vCPU ceiling
while providing memory headroom for the largest full-history Pinball graph.

This resource profile is a manual operational gate, not a manifest field. The
launcher enforces worker/thread controls, and the campaign freezes CPU/kernel/
affinity identity, but neither records nor enforces RAM, disk capacity, GCP
machine type, or provisioning model. The operator must verify those values and
must not use a smaller profile for production until a saturated largest-cell
shakeout demonstrates safe proportional-set-size and private-dirty headroom.

The AWS launcher remains available on the existing validated
`c8a.48xlarge` environment, but it deliberately overrides the separately
authorized two-pool 192-worker layout and launches only 32 workers.

For each incomplete cell, the single-threaded parent compiles and authenticates
one `PreparedCell`, then starts a fresh `fork` pool whose 32 workers inherit
that read-mostly state through copy-on-write. Workers are forbidden from
silently recompiling if the preload is absent. The pool is closed and the
parent copy is released before the next cell is prepared. This avoids 32
independent copies of the large Pinball graph and schedule, while keeping all
matcher scratch state process-private on write.

Keep the clone, runtime root, and campaign on storage that survives VM loss.
`tmux` protects against SSH loss but not deletion of a VM or an auto-deleted
disk. Campaign creation freezes the exact execution environment; a campaign
created on GCP cannot be resumed on AWS, and a replacement VM whose kernel,
CPU model, microcode, or affinity differs from the manifest is rejected.

## 7. Environment setup

On the reference GCP VM:

```bash
cd /mnt/ysc/yoked-surface-codes
./gcp/setup_environment \
  --runtime-root /mnt/ysc/yoked-surface-codes-runtime \
  --run-tests
source gcp/activate_environment
```

On the validated AWS host profile:

```bash
cd /mnt/ysc/yoked-surface-codes
./aws/setup_environment \
  --runtime-root /mnt/ysc/yoked-surface-codes-aws-runtime \
  --run-tests
./aws/run_pinball_promatch_fig8 host-check
```

Campaign `create`, `run`, and `plot` fail until the complete implementation is
committed and the checkout is clean; read-only `status` intentionally remains
available from a dirty checkout. Create and collect the campaign on the same
VM and runtime profile.

## 8. Mandatory shakeout campaigns

First use a disposable 1,000-shot-per-cell campaign to check compilation,
ledger validation, plotting, and basic resume behavior:

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

One 1,000-shot batch does not saturate 32 workers. Before production, create a
separate 32,000-shot-per-cell shakeout, interrupt it after several ledgers are
installed, and resume it with the identical command:

```bash
./gcp/run_pinball_promatch_fig8 create \
  --run-id p002-pb-pm-shakeout-32k-v1 \
  --shots-per-cell 32000
./gcp/run_pinball_promatch_fig8 run \
  --run-id p002-pb-pm-shakeout-32k-v1
```

Observe the largest `d=11, patches=10, rounds=88` cell while all 32 workers are
decoding. Measure summed proportional-set size and private-dirty memory, not
summed RSS, which double-counts copy-on-write pages. Require safe memory
headroom, no swap activity, no preload/fork/worker-recompile error, exact
ledger recovery after resume, and a calibrated production ETA. Never resume or
promote either shakeout directory as the production campaign.

## 9. 100,000-shot production characterization

Create a fresh immutable GCP campaign for the planned 100,000 shots per cell:

```bash
./gcp/run_pinball_promatch_fig8 create \
  --run-id p002-pb-pm-100k-gcp32-v1 \
  --shots-per-cell 100000
```

Start the collector in a detached `tmux` session, then attach to observe it:

```bash
tmux new-session -d -s ysc-pinball-promatch \
  -c /mnt/ysc/yoked-surface-codes \
  './gcp/run_pinball_promatch_fig8 run --run-id p002-pb-pm-100k-gcp32-v1'
tmux attach-session -t ysc-pinball-promatch
```

Detach with `Ctrl-b`, then `d`. Read-only progress is available from another
shell:

```bash
cd /mnt/ysc/yoked-surface-codes
./gcp/run_pinball_promatch_fig8 status \
  --run-id p002-pb-pm-100k-gcp32-v1
```

After an interruption, check out the exact recorded commit, restore the same
runtime and campaign paths, and issue the identical `run` command. Do not edit
`campaign.json` or any completed ledger.

After completion, generate the authenticated report, CSV, and comparison plot:

```bash
./gcp/run_pinball_promatch_fig8 plot \
  --run-id p002-pb-pm-100k-gcp32-v1
```

If a previous analysis directory is present and you intentionally want to
replace it after revalidating the completed collection, add `--overwrite`.

The outputs are written to:

```text
<campaign>/analysis/analysis.json
<campaign>/analysis/analysis.csv
<campaign>/analysis/analysis.png
```

The same scientific contract can be launched on the validated AWS runtime by
using `aws/run_pinball_promatch_fig8` and a fresh AWS-specific run ID. The
collector permits at most 1,000,000 shots per cell, but changing the shot count
always requires a new campaign.

## 10. Interpretation boundary

The experiment can support statements about the frozen YSC integrations, for
example:

> At `p=0.002`, native YSC Pinball V2 had a higher/lower paired logical-failure
> rate and residual-event workload than native YSC ProMatch on this grid.

It cannot establish intrinsic algorithm superiority, reproduce Pinball's
cryogenic hardware bandwidth/power/latency results, or describe a true
patch-logical outer QDPC decoder. Both treatment arms remain local
predecoders followed by flat complete-graph residual MWPM.
