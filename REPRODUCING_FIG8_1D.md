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

Use the pinned CPython 3.14.5 environment described in [`README.md`](README.md).
On the experiment workstation, follow [`AGENTS.md`](AGENTS.md) exactly: its
`uv` cache, managed interpreter, and temporary-directory settings prevent
writes to the quota-limited home directory. Direct dependency versions are
pinned in `requirements.txt`; uv resolves their transitive dependencies. The
plotting and gap collection code use repository-owned compatibility helpers
instead of private Sinter modules, which were removed after Sinter 1.12.

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
env -u MAX_ERRORS \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
PROCESSES=32 THREADS_PER_PROCESS=1 \
./reproduce_fig8_1d smoke
```

The smoke test uses `d=3`, `n=6`, `r=4d`, and PyMatching's public two-pass
correlated decoder. It checks that circuit generation, SI1000 noise insertion,
detector-error-model conversion, sampling, decoding, and resumable CSV output
all work.

Shot and process controls can be overridden. Simulation commands default to 32
worker processes and refuse values above 32. Native numerical libraries must
use exactly one thread per worker:

```bash
env -u MAX_ERRORS \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
MAX_SHOTS=1000000 PROCESSES=32 THREADS_PER_PROCESS=1 \
./reproduce_fig8_1d smoke
```

## Stage 3: generate the Figure 8b validation grid

Generating the circuits is inexpensive compared with sampling them:

```bash
./reproduce_fig8_1d generate-validation-grid
```

An open correlated-decoder baseline can then be collected with:

```bash
env -u MAX_ERRORS \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
PROCESSES=32 THREADS_PER_PROCESS=1 \
./reproduce_fig8_1d collect-open-validation-grid
```

The command defaults to Sinter's `pymatching-correlated` decoder and is
resumable through
`out/fig8_1d/validation_grid/pymatching-correlated_stats.csv`. To collect an
ordinary matching baseline instead, set `DECODER=pymatching`; this writes to
`pymatching_stats.csv`. Start with a small fixed shot limit. The
larger-distance paper points required tens or hundreds of millions of shots,
and some checked-in runs used one billion shots per circuit.

The runner defaults to 32 worker processes and requires exactly one native
numerical thread per worker. `PROCESSES` must be in `1..32`; a larger value
fails before collection.
All project simulation-collection commands use 32 processes. Smaller counts are
reserved for explicitly non-claim local smoke/debug work. A bounded resumable
debug collection that follows the project CPU setting is:

```bash
env -u MAX_ERRORS \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
PROCESSES=32 THREADS_PER_PROCESS=1 MAX_SHOTS=1000000 \
./reproduce_fig8_1d collect-open-validation-grid
```

`PROCESSES` controls Sinter's worker-process count and is hard-capped at 32.
`THREADS_PER_PROCESS` must equal `1`; it configures OpenMP, OpenBLAS, MKL,
NumExpr, Accelerate, and BLIS inside each worker, preventing nested thread
oversubscription. Rerunning the command continues from the existing CSV instead
of discarding completed samples.

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

The repository also registers four ProMatch-related custom decoder paths for
studying a local L1 predecoder in front of the existing flat, joint PyMatching
residual decode:

| Short role | Sinter decoder name | Scientific scope |
| --- | --- | --- |
| `PU-window` | `promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching` | Primary ProMatch-style treatment: independent `(patch, basis, d-round window)` attempts, `HW=10`, stages 1--4, no boundary matching, zero observable-frame local paths. |
| `PU-boundary` | `promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching` | Deferred exploratory boundary-policy prototype; V3 does not require or collect it, and it is not claim-bearing. |
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
env -u MAX_ERRORS \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
DECODER=promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching \
PROCESSES=32 THREADS_PER_PROCESS=1 MAX_SHOTS=10000 \
./reproduce_fig8_1d smoke "$TMPDIR/yoked-promatch-sinter-smoke"
```

All documented collections keep `MAX_ERRORS` unset and use a fixed shot count.
This keeps smoke and scientific command lines operationally consistent; the
smoke remains non-claim-bearing.

## Pinball-style predecoder

The frozen exploratory Pinball-style V1 path is registered as
`pinball-style-v1-fullhistory-nine-stage-wholeshotrollback-pymatching`. It
applies the fixed `M`, `B1`--`B4`, `ST1`--`ST2`, `H`, `E` schedule to the
compiled full-history detector graph. A simple shot commits the tentative
predecode and sends its residual syndrome to the joint PyMatching decoder; a
complex shot rolls back the whole tentative predecode and sends the original
syndrome to PyMatching. The implementation currently accepts odd code
distances only and includes the terminal detector layer.

This adapter is intentionally outside the frozen paired ProMatch workflow and
is not yet claim-bearing. Its source mapping, adaptation limits, and validation
gates are documented in [`docs/PINBALL_INTEGRATION_PLAN.md`](docs/PINBALL_INTEGRATION_PLAN.md).
A full Sinter smoke can be run with:

```bash
env -u MAX_ERRORS \
DECODER=pinball-style-v1-fullhistory-nine-stage-wholeshotrollback-pymatching \
PROCESSES=32 THREADS_PER_PROCESS=1 MAX_SHOTS=1000 \
./reproduce_fig8_1d smoke "$TMPDIR/yoked-pinball-sinter-smoke"
```

A stricter YSC V2 path is separately registered as
`pinball-ysc-v2-cz-fullhistory-nine-stage-domainatomic-yokeedge-pymatching`.
V2 uses an exact signed CZ temporal profile, validates the complete geometric
slot catalog, consumes both the true-boundary and inner-to-yoke `E` mechanisms,
and commits or rolls back each full-history `(patch, check_basis)` domain
independently. An inner-to-yoke `E` match is activated by its inner detector
but XORs the complete inner/yoke detector boundary and patch observable frame.
The residual, including yoke deltas and every complex domain, is decoded by the
same global PyMatching backend.

V2 currently supports odd distance, the maintained CZ circuit, exactly two
yoke detectors, both check bases, and arbitrary full-history round counts. It
records each primitive's patch-local physical Pauli support, XOR-reduces the
durable and tentative correction buffers, and validates their logical parity
against the corresponding DEM frame. The X-domain map is a reflection of the
pinned public correction buffer; the Z-domain map uses an explicitly
conjugated checkerboard symmetry and stage order. It remains a YSC-specific
software extension: raw circuit qubit IDs and fault-location provenance are
not reconstructed, and the public cryogenic artifact has no multi-patch/yoke
contract.

A V2 generic Sinter smoke uses:

```bash
env -u MAX_ERRORS \
DECODER=pinball-ysc-v2-cz-fullhistory-nine-stage-domainatomic-yokeedge-pymatching \
PROCESSES=32 THREADS_PER_PROCESS=1 MAX_SHOTS=1000 \
./reproduce_fig8_1d smoke "$TMPDIR/yoked-pinball-v2-sinter-smoke"
```

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

The completed V1 pilot (`docs/PROMATCH_PILOT_FROZEN.json`) exposed a mismatch
between per-batch replay retention and the per-cell summary cap. The completed
V2 pilot (`docs/PROMATCH_PILOT_FROZEN_V2.json`) fixed that cap, but a later audit
found that its unscoped output contract allowed analysis to create files inside
the collection directory. V2 also produced an unfavorable accuracy signal in
all five cells: regressions exceeded recoveries. V1 and V2 manifests and results
are immutable diagnostic artifacts. Do not resume, edit, or promote either
corpus.

V3 starts from fresh literal seed roots and separates collection, accuracy
analysis, and per-cell latency artifacts. A collection directory contains
exactly `experiment.json`, `protocol.json`, `summary.json`, and the frozen
`batches/<cell_id>/batch-<batch_id:08d>.json` schedule. Analysis always writes
to a distinct, initially absent directory. The analyzer must verify the exact
preexisting collection before deterministic regeneration and must not create or
modify any collection artifact.

The complete V3 freeze/run sequence is:

```bash
# Apply the required process/thread policy to the whole workflow.
unset MAX_ERRORS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# 1. Commit the implementation and verify that the worktree is clean. Freeze
#    the pilot against that implementation HEAD.
git status --short
tools/benchmark_promatch_l1 freeze \
    --protocol docs/PROMATCH_PILOT_PROTOCOL.json \
    --out-protocol docs/PROMATCH_PILOT_FROZEN_V3.json

# 2. Make the exact frozen pilot JSON the only post-freeze change, commit it,
#    and return to a clean worktree before sampling.
git add docs/PROMATCH_PILOT_FROZEN_V3.json
git commit -m "Freeze ProMatch L1 pilot protocol V3"
git status --short

# 3. Optional functional paired smoke. It is explicitly non-claim-bearing even
#    though the project invocation uses 32 workers.
tools/benchmark_promatch_l1 smoke \
    --out "$TMPDIR/yoked-promatch-paired-smoke" \
    --processes 32 \
    --shots 10000

# 4. Run all five ordered, 200,000-shot discovery cells at exactly 32 workers.
tools/benchmark_promatch_l1 pilot \
    --protocol docs/PROMATCH_PILOT_FROZEN_V3.json \
    --out "$TMPDIR/yoked-promatch-pilot-v3" \
    --processes 32

# 5. Verify the pilot and apply the frozen unsigned selection rule. Scientific
#    analysis regenerates every pilot batch with the manifest's 32 processes,
#    writes only to the separate analysis directory, and leaves collection
#    artifacts byte-for-byte unchanged.
tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_PILOT_FROZEN_V3.json \
    --input "$TMPDIR/yoked-promatch-pilot-v3" \
    --out "$TMPDIR/yoked-promatch-pilot-v3-analysis"

# This must succeed before any confirmatory protocol is frozen or any holdout
# shot is sampled. If it fails, stop and report the V3 pilot as non-viable.
jq -e '.status == "selected"' \
    "$TMPDIR/yoked-promatch-pilot-v3-analysis/pilot_selection.json"

# 6. From the resulting clean implementation/protocol HEAD, derive and freeze
#    the confirmatory protocol. Manual selection literals are not trusted.
tools/benchmark_promatch_l1 freeze \
    --protocol docs/PROMATCH_FIRST_ROUND_PROTOCOL.json \
    --pilot-protocol docs/PROMATCH_PILOT_FROZEN_V3.json \
    --pilot-input "$TMPDIR/yoked-promatch-pilot-v3" \
    --out-protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json

# 7. Again, commit only the exact generated frozen JSON and restore a clean
#    worktree before any holdout, target, or latency measurement.
git add docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json
git commit -m "Freeze ProMatch L1 first-round protocol V3"
git status --short

# 8. Only after the V3 viability check, collect the selected-cell fixed-N
#    holdout. MAX_ERRORS remains unset.
tools/benchmark_promatch_l1 confirm \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --out "$TMPDIR/yoked-promatch-confirm-v3" \
    --processes 32

# 9. Collect the fixed one-million-shot target performance corpus. Target-cell
#    accuracy remains descriptive and cannot support an accuracy claim.
tools/benchmark_promatch_l1 target \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --out "$TMPDIR/yoked-promatch-target-v3" \
    --processes 32

# 10. Analyze accuracy/workload into separate artifact-set directories. Each
#     scientific analysis independently regenerates every declared batch at 32
#     processes before trusting ledgers.
tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --input "$TMPDIR/yoked-promatch-confirm-v3" \
    --out "$TMPDIR/yoked-promatch-confirm-v3-analysis"

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --input "$TMPDIR/yoked-promatch-target-v3" \
    --out "$TMPDIR/yoked-promatch-target-v3-analysis"
```

If `pilot_selection.json` reports `confirmation-infeasible`, stop after the
pilot. Do not freeze the first-round protocol or collect holdout data. A new
cell grid or algorithm requires a versioned protocol change and another fresh
pilot; the unfavorable V2 direction must not be erased or relabeled. Here,
`selected` means that the frozen unsigned measurability and power gates passed;
it is not a claim that the pilot showed a favorable accuracy direction.

After a viable pilot, read `analysis_config.selection.selected_cell_id` from
the frozen first-round protocol and use it together with the fixed target ID
for the two separate per-cell latency suites and directories. Both suites use
the one frozen `timing_corpus` root; their cell identities and artifact
directories remain distinct:

```bash
SELECTED_CELL_ID=$(jq -r '.analysis_config.selection.selected_cell_id' \
    docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json)
TARGET_CELL_ID='target-d11-n6-y2-r44-p0.001'

tools/benchmark_promatch_l1 latency \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --cell-id "$SELECTED_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-latency-selected-v3" \
    --processes 32

tools/benchmark_promatch_l1 latency \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --cell-id "$TARGET_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-latency-target-v3" \
    --processes 32

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --latency-input "$TMPDIR/yoked-promatch-latency-selected-v3" \
    --latency-cell "$SELECTED_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-latency-selected-v3-analysis"

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --latency-input "$TMPDIR/yoked-promatch-latency-target-v3" \
    --latency-cell "$TARGET_CELL_ID" \
    --out "$TMPDIR/yoked-promatch-latency-target-v3-analysis"
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

Neither `smoke`, `latency-smoke`, nor the Sinter smoke can support accuracy,
workload, or latency claims.

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

### Diagnosing PU-versus-U0 disagreements

`tools/diagnose_promatch_l1` is a read-only diagnostic (single process, one
native thread, not claim-bearing). `replay` re-decodes the regression and
recovery samples retained in a paired collection's `summary.json`, checks that
they reproduce bit-for-bit, and classifies every committed prematch against
the flat MWPM solution for the same shot (which endpoint MWPM instead sent to
the yoke hub, the true boundary, the neighbouring window, or the terminal
layer, and how the residual matching re-routed yoke edges). `probe` samples
fresh shots for a protocol cell or an explicit `--d/--patches/--rounds/--p`
and reports activation, commits and disagreement rate per stage, and paired
U0/PU failures:

```bash
tools/diagnose_promatch_l1 replay \
    --input out/promatch_l1_round1_v3_20260817_32p/pilot \
    --cell pilot-01-d7-n6-y2-r28-p0.001 --show 5
tools/diagnose_promatch_l1 probe \
    --protocol out/promatch_l1_round1_v3_20260817_32p/pilot/protocol.json \
    --cell pilot-01-d7-n6-y2-r28-p0.001 --shots 20000 --seed 12345
tools/diagnose_promatch_l1 probe --d 11 --patches 10 --rounds 88 --p 0.001 \
    --shots 100 --seed 777 --out-json "$TMPDIR/probe-d11.json"
```

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
