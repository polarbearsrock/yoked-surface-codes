# Fixed BP pre-decoding of one d=7 patch, SI1000 p=0.003

Fixed before evaluation: one d=7 patch, 28 noisy CZ rounds, SI1000 p=0.003,
zero yokes, and the repository's ideal preparation/readout convention with
two logical observables. The empty detector emitted by the zero-yoke circuit
is retained and always zero. BP uses the joint X/Z DEM fault model, including
unfired parity checks and detector cancellation across a fault's components.
It receives no sampled logical outcomes or reference-decoder predictions.

The five variants are weighted UF, correlated UF, and fixed BP1, BP2, and
BP5 followed by weighted UF. Every BP variant always finishes with UF on the
original syndrome. There is no convergence test, direct BP return, adaptive
budget, graph pruning, preliminary hard correction, or best-answer selection.
The user explicitly narrowed the experiment to these five variants.

BP reuses the archived sum-product flooding arithmetic, check-message damping
0.5, LLR clipping at +/-30, and double precision. Map probabilities q to edge
weights using `-log(clip(sum(q_f for faults supporting e), 1e-15, 1))`.
This heuristic projection differs from the baseline's prior log-odds weights;
without a separate prior-projection control this comparison measures the
complete BP-plus-projection design, not inference alone. UF growth and
peeling reuse the verified native implementation. Correlated UF skips a
redundant second pass when no weights change, matching production behavior.

## Sampling and analysis

Use 32 OpenMP threads pinned to 32 distinct physical cores. Run 10,000 paired
pilot shots (Stim seed 2026092801), followed by 100,000 fresh paired
confirmation shots (seed 2026092802), without tuning between phases.
Generate and retain packed detector/truth arrays once per phase. Every
variant sees identical detector rows. Report the phases separately.

Primary outcome: probability that either logical observable is wrong per
28-round memory shot. Report X- and Z-observable errors separately. These
are single-patch memory-shot rates, not per-round or six-patch block rates.
The primary contrast is BP5 + weighted UF versus correlated UF. Report
Wilson 95% marginal intervals, paired repairs/regressions, paired rate
differences and 95% intervals, and exact McNemar tests. Other contrasts are
exploratory and unadjusted for multiple comparisons. One distance and one
noise strength cannot establish a threshold or distance-scaling law.

## Validation and timing

Before collection, reuse independent exact-posterior/tree, zero-message,
high-degree, and physical-UF checks from the archived pilot. Check all
five new native paths against production Python UF and independent Python BP
on synthetic and predetermined validation syndromes, including zero syndrome.
Verify physical corrections, logical reconstruction, fixed iteration counts,
and invariance between 1 and 32 OpenMP threads. Full physical-syndrome checks
remain enabled throughout collection. Stop on any failure.

OpenMP parallelizes independent shots; BP and UF within each shot are serial.
Reuse a common BP trajectory for budgets 1, 2, and 5, but charge each policy
its full cumulative BP initialization/inference, projection, and UF work.
Charge correlated UF for both necessary UF passes and reweighting. Collect
1,024 predetermined serial kernel timing samples from the pilot separately
from collection wall time. Report empirical p50/p95/p99 and mean timings;
these are software kernel measurements, not hardware latency or a measured
32-core speedup. Timing excludes compilation, Python orchestration, unpacking,
physical-correction audits, and some buffer destruction. Fixed variant order
and shared work may affect cache state. Tail quantiles have finite-sample
uncertainty. No native-versus-Python latency comparison is used.

## Artifacts and launch

Temporary builds, raw samples, logs, and resumable checkpoints live under
`$DANTE_SCRATCH/runs`. Retain per-shot results, source snapshots, configuration,
source/model/sample hashes, validation records, progress, and reports under
`results/`. Original production code and historical experiments are preserved.

Source `env/workspace.sh`, activate `repos/yoked-surface-codes/.venv`, and run
`python experiments/bp_uf_single_patch_d7/launch.py --threads 32`.
The launcher freezes sources and starts the resumable runner in a detached
process. Its output gives the PID, log, progress path, and result directory.
Validation must pass before the first collection phase starts.
Use `--wait` in a managed execution session to keep the process supervisor
alive until the run finishes.

Background reference: Higgott et al., *Improved decoding of circuit noise and
fragile boundaries of tailored surface codes*, Appendices A-C:
https://arxiv.org/html/2203.04948v5#A3
This experiment uses fixed BP evidence budgets and always ends with UF;
it does not reproduce the paper's stopping policy or entire implementation.

## Add correlated MWPM on the saved sample

Run `python experiments/bp_uf_single_patch_d7/add_correlated_mwpm.py --base RESULTS_DIRECTORY --workers 32`.
This adds the subsequently requested correlated-MWPM reference on both saved
sample sets without changing the five original decoder outputs. Correlations
are enabled at PyMatching graph import and decoding. Thirty-two independent
worker processes run within a 32-physical-core affinity mask. Serial timing
uses the original 1,024 timing rows and records both the bound native decoder
and public API; the difference in timing boundaries is stated in the report.
Combined results and provenance are retained under `correlated_mwpm/`, and
the main report is updated after validation. The previous report is preserved
as `report_five_decoders.md`. Use `--output` with a fresh directory to repeat
the addition. New reference comparisons are exploratory.

## Add fixed BP10 + UF on the saved sample

Run `python experiments/bp_uf_single_patch_d7/add_bp10.py --base RESULTS_DIRECTORY --threads 32`.
This adds the subsequently requested BP10 arm to both saved sample sets,
using the original frozen BP/UF sources and the same damping, clipping,
projection, circuit, and detector rows. Exactly ten BP iterations always
precede UF. Thirty-two native OpenMP threads decode independent shots on
32 distinct physical cores. This addition expects the correlated-MWPM
extension to have completed.

Before collection, check ten-round posteriors against independent Python BP,
selected corrections against production UF, physical syndrome reconstruction,
serial/parallel agreement, and reproduction of archived BP5 outputs with the
new wrapper at five rounds. Every collection correction is checked physically.
Benchmark BP10 serially on the same 1,024 timing rows, separately recording
initialization, ten iterations, edge-weight projection, and UF. Preserve the
original timing measurements; standalone BP10 and the earlier shared BP
trajectory have different cache conditions.

Combined seven-decoder predictions, paired statistics, sources, verification,
and timing data are retained under `bp10_uf/`. The updated main report preserves
the previous report as `report_six_decoders.md`. Use `--output` with a fresh
directory for a repeated extension. BP10 comparisons are exploratory because
this arm was requested after viewing the original results; the reused sample
named confirmation is not a fresh BP10 holdout.
