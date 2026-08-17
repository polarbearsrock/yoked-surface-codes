# Reproducing the 1D results in published Figure 8

The published Figure 8 has two panels involving 1D yoked surface codes:

- **Figure 8b** compares a full, one-outer-round circuit simulation against a
  complementary-gap simulation and the fitted scaling law.
- **Figure 8d** compares long, ten-outer-round phenomenological simulations of
  unyoked (0D) and 1D-yoked memories against their fitted scaling laws.

The release contains the data and plotting code for both panels. It contains a
path for generating and sampling the full circuits in Figure 8b, but it does not
contain the internal correlated-matching and gap-simulation tools needed to
regenerate all samples from scratch.

## Parameter conventions

For the main 1D construction:

- `yokes=2` means the outer code has both an X-type and a Z-type yoke. This is
  the 1D `[[n, n-2, 2]]` quantum parity-check code used in the main text.
- `patches=n` is the number of inner surface-code patches in one outer block.
- `d` is the inner surface-code patch diameter.
- `r` is the number of noisy inner-code rounds between the perfect initial and
  final time boundaries in a full-circuit experiment.
- The circuit uses the CZ gateset and SI1000 noise at `p=0.001`.

The Figure 8b validation grid is

- `d in {5, 7, 9, 11}`;
- `n in {6, 10}`; and
- `r in {4d, 8d}`.

The comparison uses the per-patch-round form of the 1D fit

```text
p_L / (r*n) ~= r*n*8^(-d)/500,
```

which comes from the one-outer-round cumulative fit

```text
p_L ~= r^2*n^2*8^(-d)/500.
```

## Environment

Use Python 3.14 with the upgraded dependencies. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency versions are pinned in `requirements.txt`. The plotting and gap
collection code use repository-owned compatibility helpers instead of private
Sinter modules, which were removed after Sinter 1.12.

GNU `parallel` is not needed by the focused workflow below.

## Stage 1: reproduce the panels from released data

```bash
./reproduce_fig8_1d plot-paper-data
```

This writes:

```text
out/fig8_1d/paper_data/fig8b_1d_full_vs_gap.png
out/fig8_1d/paper_data/fig8d_0d_vs_1d.png
```

This stage reproduces the data, axes, fits, and uncertainty bars in the
checked-in `assets/gap_vs_full_1D.png` and `assets/gap_0D_and_1D.png` figures
without running Monte Carlo. Raster pixels and text spacing can differ with
the installed Matplotlib and font versions.

## Stage 2: run a small end-to-end full-circuit smoke test

```bash
./reproduce_fig8_1d smoke
```

The smoke test uses `d=3`, `n=6`, `r=4d`, and PyMatching's public two-pass
correlated decoder. It checks that circuit generation, SI1000 noise insertion,
detector-error-model conversion, sampling, decoding, and resumable CSV output
all work.

Collection controls can be overridden. Simulation commands default to 32
worker processes and refuse values above 32. Native numerical libraries use
one thread per worker by default:

```bash
MAX_SHOTS=1000000 MAX_ERRORS=1000 PROCESSES=32 THREADS_PER_PROCESS=1 \
./reproduce_fig8_1d smoke
```

## Stage 3: generate the Figure 8b validation grid

Generating the circuits is inexpensive compared with sampling them:

```bash
./reproduce_fig8_1d generate-validation-grid
```

An open correlated-decoder baseline can then be collected with:

```bash
./reproduce_fig8_1d collect-open-validation-grid
```

The command defaults to Sinter's `pymatching-correlated` decoder and is
resumable through
`out/fig8_1d/validation_grid/pymatching-correlated_stats.csv`. To collect an
ordinary matching baseline instead, set `DECODER=pymatching`; this writes to
`pymatching_stats.csv`. Start with small shot and error limits. The
larger-distance paper points required tens or hundreds of millions of shots,
and some checked-in runs used one billion shots per circuit.

The runner defaults to 32 worker processes and one native numerical thread per
worker. `PROCESSES` must be in `1..32`; a larger value fails before collection.
All project simulation-collection commands use 32 processes. Smaller counts are
reserved for explicitly non-claim local smoke/debug work. A bounded resumable
debug collection that follows the project CPU setting is:

```bash
PROCESSES=32 THREADS_PER_PROCESS=1 \
MAX_SHOTS=1000000 MAX_ERRORS=100 \
./reproduce_fig8_1d collect-open-validation-grid
```

`PROCESSES` controls Sinter's worker-process count and is hard-capped at 32.
`THREADS_PER_PROCESS` caps
OpenMP, OpenBLAS, MKL, NumExpr, Accelerate, and BLIS threads inside each worker,
preventing nested thread oversubscription. Rerunning the command continues from
the existing CSV instead of discarding completed samples.

Plot the newly collected points and the paper's fitted scaling law with:

```bash
./reproduce_fig8_1d plot-open-validation-grid
```

This writes
`out/fig8_1d/validation_grid/pymatching-correlated_fig8b.png`. Each point is
annotated as `logical errors / shots`; hollow downward triangles indicate
zero-error upper bounds instead of measured nonzero rates.

## Decoder limitation

The paper data uses `sparse_blossom_correlated`, an internal correlated
minimum-weight perfect-matching decoder. Current Sinter and PyMatching releases
provide a public two-pass correlated decoder under the name
`pymatching-correlated`, and the focused workflow now uses it by default. It is
the closest supported open replacement, but it is not the same decoder binary
used for the published data. Fresh results must therefore be compared against
matched rows in `assets/stats_check.csv` before being described as an exact
reproduction.

Therefore the milestones are deliberately separated:

1. exact plot reproduction from released statistics;
2. end-to-end circuit and sampling validation with an available decoder;
3. validation of `pymatching-correlated` against matched rows in
   `assets/stats_check.csv`; and
4. only then, expensive production sampling.

Figure 8d additionally depends on the unreleased multi-round gap simulator.
Reimplementing that simulator is a separate task from reproducing Figure 8b's
full-circuit points.

## L1 ProMatch-style predecoder

The repository also registers four custom decoder paths for studying a local
L1 predecoder in front of the existing flat, joint PyMatching residual decode:

| Short role | Sinter decoder name | Scientific scope |
| --- | --- | --- |
| `PU-window` | `promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching` | Primary ProMatch-style treatment: independent `(patch, basis, d-round window)` attempts, `HW=10`, stages 1--4, no boundary matching, zero observable-frame local paths. |
| `PU-boundary` | `promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching` | Mandatory exploratory boundary-policy comparator, not the primary treatment. |
| `PU-full` | `promatch-l1-v1-fullhistory-hw10-stages1234-noboundary-zeroframe-pymatching` | Full-history diagnostic only; it is not claim-bearing because it changes the online-window model. |
| `U0-wrap` | `pymatching-u0-wrap-v1-windowd` | Identity adapter control used to measure packing/layout overhead. |

The built-in `pymatching` decoder is the authoritative `U0-direct` baseline.
The built-in `pymatching-correlated` decoder is retained as descriptive context
(`C0`), not as the baseline for the causal predecoder comparison. The first
experiment is therefore **L1 predecode plus a flat global residual decoder**;
it is not yet a hierarchical L1/L2 decoder and does not run an outer QDPC
message-passing decoder.

The custom decoders can be exercised through the Figure 8 integration runner,
for example:

```bash
DECODER=promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching \
PROCESSES=32 THREADS_PER_PROCESS=1 MAX_SHOTS=10000 MAX_ERRORS=20 \
./reproduce_fig8_1d smoke "$TMPDIR/yoked-promatch-sinter-smoke"
```

`MAX_ERRORS` is acceptable in this integration smoke only. It is forbidden in
the fixed-shot paired pilot, confirmatory, and target collections (even though
target-cell accuracy is descriptive rather than claim-bearing).

## Paired ProMatch experiment workflow

The paired harness samples each immutable Stim batch once and decodes the same
detector and observable arrays with the two accuracy arms: `U0-direct` and
`PU-window`. That pairing is necessary for the matched accuracy analysis and
gives exact batch-level resume. `U0-wrap` is not a third accuracy arm; it is the
identity-adapter control used only by the latency experiment. The paired path is
separate from Sinter's independently sampled per-decoder CSV collection.

The checked-in protocol JSON files are deliberately marked
`DRAFT_TEMPLATE_NOT_FROZEN`. They include literal, mutually distinct 256-bit
seed roots, but contain `null` commit, circuit, DEM, graph, and experiment
hashes. A sampling command must refuse a draft. Inspect the templates before
freezing:

```bash
tools/benchmark_promatch_l1 inspect \
    --protocol docs/PROMATCH_PILOT_PROTOCOL.json

tools/benchmark_promatch_l1 inspect \
    --protocol docs/PROMATCH_FIRST_ROUND_PROTOCOL.json
```

Use two commits for each scientific protocol. First freeze from a clean
implementation commit. Then commit the exact generated
`docs/*FROZEN*.json` file as the sole change after that implementation commit.
The frozen JSON records the implementation parent commit; the runner permits
exactly this one protocol-only child commit and rejects any intervening source
or documentation change.

The complete freeze/run sequence is:

The completed V1 pilot (`docs/PROMATCH_PILOT_FROZEN.json`) exposed an output-
contract mismatch between per-batch replay candidates and the per-cell summary
cap. Its artifacts are diagnostic-only and must not be resumed. V2 uses a fresh
pilot seed, an explicit two-stage replay policy, and a new output directory.

```bash
# 1. Commit the implementation and verify that the worktree is clean. Freeze
#    the pilot against that implementation HEAD.
git status --short
tools/benchmark_promatch_l1 freeze \
    --protocol docs/PROMATCH_PILOT_PROTOCOL.json \
    --out-protocol docs/PROMATCH_PILOT_FROZEN_V2.json

# 2. Make the exact frozen pilot JSON the only post-freeze change, commit it,
#    and return to a clean worktree before sampling.
git add docs/PROMATCH_PILOT_FROZEN_V2.json
git commit -m "Freeze ProMatch L1 pilot protocol V2"
git status --short

# 3. Optional functional paired smoke. It is explicitly non-claim-bearing even
#    though the project invocation uses 32 workers.
tools/benchmark_promatch_l1 smoke \
    --out "$TMPDIR/yoked-promatch-paired-smoke" \
    --processes 32 \
    --shots 10000

# 4. Run all five ordered, 200,000-shot discovery cells at exactly 32 workers.
env -u MAX_ERRORS tools/benchmark_promatch_l1 pilot \
    --protocol docs/PROMATCH_PILOT_FROZEN_V2.json \
    --out "$TMPDIR/yoked-promatch-pilot-v2" \
    --processes 32

# 5. Verify the pilot and apply the frozen unsigned selection rule. Scientific
#    analysis regenerates every pilot batch with the manifest's 32 processes.
env -u MAX_ERRORS tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_PILOT_FROZEN_V2.json \
    --input "$TMPDIR/yoked-promatch-pilot-v2"

# 6. From the resulting clean implementation/protocol HEAD, derive and freeze
#    the confirmatory protocol. Manual selection literals are not trusted.
tools/benchmark_promatch_l1 freeze \
    --protocol docs/PROMATCH_FIRST_ROUND_PROTOCOL.json \
    --pilot-protocol docs/PROMATCH_PILOT_FROZEN_V2.json \
    --pilot-input "$TMPDIR/yoked-promatch-pilot-v2" \
    --out-protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json

# 7. Again, commit only the exact generated frozen JSON and restore a clean
#    worktree before any holdout, target, or latency measurement.
git add docs/PROMATCH_FIRST_ROUND_FROZEN.json
git commit -m "Freeze ProMatch L1 first-round protocol"
git status --short

# 8. Collect the selected-cell fixed-N holdout. MAX_ERRORS is forbidden.
tools/benchmark_promatch_l1 confirm \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --out "$TMPDIR/yoked-promatch-confirm" \
    --processes 32

# 9. Collect the fixed one-million-shot target performance corpus. Target-cell
#    accuracy remains descriptive and cannot support an accuracy claim.
tools/benchmark_promatch_l1 target \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --out "$TMPDIR/yoked-promatch-target" \
    --processes 32

# 10. Analyze accuracy/workload. Each scientific analysis independently
#     regenerates every declared batch at 32 processes before trusting ledgers.
tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --input "$TMPDIR/yoked-promatch-confirm"

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --input "$TMPDIR/yoked-promatch-target"
```

Read `analysis_config.selection.selected_cell_id` from the frozen first-round
protocol and use it together with the fixed target ID for the two latency suites:

```bash
SELECTED_CELL_ID='<analysis_config.selection.selected_cell_id from the frozen protocol>'
TARGET_CELL_ID='target-d11-n6-y2-r44-p0.001'

tools/benchmark_promatch_l1 latency \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --cell-id "$SELECTED_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-latency-selected" \
    --processes 32

tools/benchmark_promatch_l1 latency \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --cell-id "$TARGET_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-latency-target" \
    --processes 32

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --input "$TMPDIR/yoked-promatch-confirm" \
    --latency-input "$TMPDIR/yoked-promatch-latency-selected" \
    --latency-cell "$SELECTED_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-analysis-selected"

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN.json \
    --input "$TMPDIR/yoked-promatch-target" \
    --latency-input "$TMPDIR/yoked-promatch-latency-target" \
    --latency-cell "$TARGET_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-analysis-target"
```

Latency uses the frozen 10,000-shot corpus per restart. The ten fresh timing
restarts are deliberately serialized (`timed_restart_concurrency=1`) so they do
not contend with one another; `--processes 32` remains the recorded global
experiment setting and immutable maximum, not the number of simultaneous timed
workers.

For an additional real-circuit latency plumbing check, use the distinct
non-claim command:

```bash
tools/benchmark_promatch_l1 latency-smoke \
    --out "$TMPDIR/yoked-promatch-latency-smoke" \
    --processes 32
```

Neither `smoke`, `latency-smoke`, nor the Sinter smoke with `MAX_ERRORS` can
support accuracy, workload, or latency claims.

Freezing is not merely a JSON format conversion. It fills every required
placeholder, verifies a clean implementation commit, records package and
machine metadata, hashes the circuit/DEM/compiled graph and relevant source
files, writes the exact schedules, computes the canonical experiment ID, and
changes the status to `FROZEN`. The confirmatory freeze also regenerates and
verifies the pilot and derives the selected cell, margin, fixed shot count, and
pilot provenance; it does not accept manually supplied adaptive values. If the
pilot and first-round implementation hashes differ, rerun the pilot.

Both paired templates set `processes=32`, `max_processes=32`,
`threads_per_process=1`, and `sample_batch_size=10000`. Scientific simulation
collection and regeneration use exactly 32 processes. Every path refuses more
than 32 regardless of command-line or environment overrides.

## Accuracy and rare-event limitations

Direct Monte Carlo cost scales with the inverse logical failure probability.
For example, even observing roughly 100 failures at a true rate of `1e-12`
requires on the order of `1e14` shots before accounting for confidence-interval
or paired-discordance requirements. Thirty-two processes improve throughput
but do not change that statistical scaling. A zero-error run supplies an upper
bound; it does not demonstrate equality or non-inferiority.

For this reason the preregistered discovery grid selects one measurable stress
cell for the paired accuracy question. The fixed target cell
`(d=11, patches=6, yokes=2, r=44, p=0.001)` uses one million paired shots for
activation/workload and the frozen timing corpus; its natural-noise accuracy
counts are descriptive only. Do not combine accuracy evidence from the stress
cell with latency evidence from the target cell to claim target-cell accuracy.
Similarly, residual detector-count reduction and residual-backend speedup are
not automatically end-to-end latency improvements: the full adapter time and
batch-1 tail ratio must pass their separate gates.
