# Parameterized paired ProMatch Figure-8b sweep on GCP

## 1. Question

For the released 1D-yoked full-circuit construction, how does the current L1
ProMatch-style predecoder change logical accuracy relative to decoding the
same syndrome directly with joint MWPM?

This is deliberately a *paired* experiment. Every sampled detector/observable
shot is sent to both arms. A difference therefore measures decoder behavior,
not Monte Carlo noise from two independently sampled corpora.

This first version has two create-time parameters:

- `p`: the SI1000 physical error probability; and
- `shots_per_cell`: the exact number of paired shots sampled for every
  geometry cell.

All other experiment choices are fixed so changing either parameter produces a
comparable campaign rather than a subtly different experiment.

The SI1000 input range is `0 < p <= 0.2`. SI1000 assigns measurement flips
probability `5p`, so values above `0.2` are not valid probabilities. Campaign
creation additionally compiles all 16 matching graphs and rejects a `p` for
which the current decoder produces an unsupported graph (for example, a
negative eligible-edge weight); no manifest or shots are produced in that
case.

## 2. Fixed circuit grid

The sweep uses 16 full-circuit Figure-8b cells:

```text
d                 in {5, 7, 9, 11}
patches           in {6, 10}
rounds r          in {4d, 8d}
yokes                 2 (one X yoke and one Z yoke)
style                 CZ
noise                 SI1000 at campaign p
remove_x_yoke         false
```

Thus `shots_per_cell=1_000_000` means 16,000,000 physical circuit samples and
32,000,000 decoder evaluations: both decoders process each one of the 16
million shots.

This is the fresh-sampling path available for Figure 8b. It is not a literal
regeneration of Figure 8d: the repository does not contain the paper's
multi-round complementary-gap simulator.

## 3. Decoder arms

### U0-direct

`U0-direct` runs ordinary, uncorrelated PyMatching on the complete yoked
detector error model. It is the control arm and the residual decoder used by
the treatment arm.

### PU-window

`PU-window` first partitions the detector sample into local L1 domains indexed
by surface-code patch, check basis, and non-overlapping `d`-round time window.
For domains above `HW=10`, the current ProMatch-style stages 1--4 may commit
zero-observable-frame local pairs. The committed local endpoints are removed,
and ordinary, uncorrelated PyMatching then decodes the residual *complete
yoked graph*.

This is therefore:

```text
local L1 ProMatch-style predecode + flat global residual MWPM
```

It is not a fully hierarchical L1/L2 decoder and it does not use an outer QDPC
message-passing decoder. The current policy is already known to regress in the
V3 pilot; this sweep measures where and by how much over a broader geometry
grid and configurable `p`.

The built-in correlated PyMatching decoder is not an accuracy arm. Adding it
would require a separately specified experiment because it changes the
postdecoder rather than isolating the predecoder intervention.

## 4. Paired outcome table

For each shot, let `U0` and `PU` mean that the decoder's predicted observable
frame equals the actual observable frame. The collector records exactly one of
four mutually exclusive outcomes:

| Outcome | U0-direct | PU-window | Meaning |
| --- | --- | --- | --- |
| `both_correct` | correct | correct | no logical failure in either arm |
| `both_wrong` | wrong | wrong | logical failure in both arms |
| `recovery` | wrong | correct | predecoder fixes a U0 failure |
| `regression` | correct | wrong | predecoder creates a failure |

For a cell with `N` shots:

```text
U0 failures = both_wrong + recoveries
PU failures = both_wrong + regressions
U0 LER      = U0 failures / N
PU LER      = PU failures / N
paired delta failures = regressions - recoveries
```

The primary policy signal is the paired imbalance between regressions and
recoveries. Marginal LER alone cannot reveal whether the same hard shots fail
in both arms.

## 5. Frozen campaign contract

Creation writes `campaign.json` before sampling. It constructs all 16 circuits
and decoder graphs once, then records:

- the clean repository commit and source hashes;
- exact Python/package and execution environment details;
- `p`, `shots_per_cell`, the fixed grid, and decoder policy;
- circuit, detector-error-model, layout, and matching-graph fingerprints for
  every cell;
- a random 256-bit sampler seed root and deterministic seed derivation; and
- the complete per-cell fixed-shot batch schedule.

The campaign is marked frozen and non-claim-bearing. `p` and
`shots_per_cell` cannot be changed in place; create a new run ID to compare a
different value. Run-time validation fails closed on commit, source,
environment, manifest, circuit, DEM, layout, graph, or batch-ledger drift.

This external campaign freeze is designed for robust remote characterization.
It does not replace the repository's two-commit frozen-protocol procedure for
a future claim-bearing confirmatory experiment.

## 6. Fixed-shot batching and resume

Each cell is divided into deterministic batches of at most 1,000 shots. The
smaller batch size bounds the largest unpacked detector array and reduces lost
work on a preemptible VM. Each batch is sampled once, decoded by both arms,
validated, and atomically committed to:

```text
<campaign>/collection/batches/<cell_id>/batch-XXXXXXXX.json
```

Re-running the campaign checks every existing ledger against the frozen
schedule and skips valid batches. Missing batches are deterministically
regenerated. Unknown, duplicate, malformed, or inconsistent artifacts are an
error, not silently ignored.

Collection always uses exactly 32 worker processes and one native numerical
thread per worker. `MAX_ERRORS` must be unset: adaptive stopping would give
different shot counts per cell and undermine the fixed paired design. The GCP
wrapper holds one runtime-wide collection lock to prevent CPU oversubscription.

## 7. Commands

After cloning the pushed branch and running `./gcp/setup_environment`, create a
campaign on persistent storage:

```bash
./gcp/run_fig8_paired create \
  --run-id p001-1m-v1 \
  --p 0.001 \
  --shots-per-cell 1000000
```

Creation generates and fingerprints the 16 cells but performs no sampling.
Start or resume the heavy work:

```bash
./gcp/run_fig8_paired run --run-id p001-1m-v1
```

Read-only progress and final plotting are separate:

```bash
./gcp/run_fig8_paired status --run-id p001-1m-v1
./gcp/run_fig8_paired plot   --run-id p001-1m-v1
```

The run directory is
`$YSC_GCP_RUNS_ROOT/fig8-paired/p001-1m-v1/`. Keep that runtime root on a
persistent disk. An SSH session manager can keep the process alive across a
terminal disconnect; after VM preemption, run the identical resume command.

## 8. Outputs and acceptance checks

`status` reports completed/expected shots and batches per cell and for the
whole campaign. A completed campaign must have:

- all 16 declared cell IDs and no undeclared artifacts;
- exactly `shots_per_cell` paired outcomes in every cell;
- paired counts summing to the cell shot count;
- no duplicate batch IDs or overlapping shot ranges; and
- hashes and experiment identity matching `campaign.json`.

`plot` accepts only such a complete collection. It writes a machine-readable
CSV and a side-by-side U0/PU logical-error-rate figure with shared axes under
`<campaign>/plots/`. Zero observed failures are displayed as upper bounds,
not as measured zero LER. The paper's analytic reference curve is shown only
when `p=0.001`, the physical error rate at which that curve was reported; a
campaign at another `p` is never overlaid with that fixed-`p` reference.

## 9. Scope and interpretation

This sweep answers whether the *current* local commit policy improves or harms
accuracy across the full-circuit geometry grid at the selected `p`. It also
shows which distance, patch count, and history length concentrate paired
regressions. It does not by itself prove latency improvement: Python wall time
is implementation telemetry, not a hardware latency model, and the residual
MWPM still sees the full graph.

The v1 interface caps `shots_per_cell` at 1,000,000. At that setting, a
campaign processes 16 million samples and can take roughly one to two days on
a 32-process VM based on current PU-window throughput; actual time depends on
the VM and syndrome density. Start with a smaller campaign to validate remote
storage and resume behavior before committing to the maximum.

For policy design, examine paired regressions versus recoveries rather than
expecting this sweep alone to identify the missing context. The retained V3
and 20,000-shot policy-audit experiments remain the detailed diagnostic source
for why a local candidate was unsafe.
