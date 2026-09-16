# Coset ensemble on the saved d=7/9 experiment

Date: 2026-09-16. Previous failure-analysis work was committed as `e94dfe1`
before this experiment. Decoder and runner source hashes are recorded in each
distance's request manifest.

**Result:** the final-pass coset ensemble gives **no LER improvement** at either
distance. Both scoring rules and all tested candidate counts reproduce the
baseline's failure indicator on every shot. The logical-space certificate
shows that none of these baseline failures is repairable while keeping the
final UF partitions fixed, even with unlimited candidates.

## Question and hypothesis

Can ensemble forest exploration improve our correlated UF decoder without
introducing a matching solve? The hypothesis is that UF's chosen spanning
forest sometimes gives a poor logical answer, while other forests inside the
same completed clusters offer a better logical class. Sampling those forests
and voting among their best candidates would then repair some UF failures.

The competing hypothesis, suggested by the preceding failure analysis, is that
UF's final partitions already exclude the correct logical class. Changing the
forest within those partitions would then alter physical corrections while
preserving the wrong logical answer. We measure the available logical space
as well as the sampled candidates to distinguish these explanations.

## Fixed experiment

The same saved 100,000 evaluation shots per distance are reused, with noise
seed 42, `noise=si1000`, `style=cz`, `p=0.003`, six patches, two ideal yokes,
and `rounds=4d` (28 and 36).
Sample payload, circuit, DEM, and saved-record hashes are checked before
collection. There is no resampling, calibration, or parameter fitting.

This is a **joint detector-graph decoder comparison** against correlated UF
and correlated MWPM. It does not add an L2 stage to the earlier hierarchical
confidence experiment.

The implementation applies the forest-ensemble idea in
[Liang et al., Algorithm 1–3](https://arxiv.org/html/2606.11076#S3), with these
explicit choices for our graph:

1. Run the existing UF first pass and apply its existing DEM correlation rules.
2. Run the unchanged second UF growth pass on those weights. Preserve its
   final partition and its ordinary correction for comparison.
3. Admit every graph edge whose endpoints share a final UF root. This includes
   internal edges that were not fully grown, giving the ensemble a larger
   search space than just the merge forest. No edge joins separate partitions.
4. Generate 24 random-priority BFS forests and peel in reverse discovery order.
   For open boundaries, identify terminals within each partition as one vertex
   with unconstrained parity and root that partition's BFS there. Detectors
   retain their original syndrome constraints.
5. For the primary rule, retain candidates with minimum edge count and vote on
   their logical masks. Repeated candidates retain their votes; a tied coset
   vote chooses the first eligible sample. The ordinary UF correction is a
   diagnostic and receives no extra vote.
6. Also evaluate a minimum-weight vote, using the same candidates and the sum
   of their correlation-adjusted weights. This is a separate weighted variant;
   its cost ties use absolute and relative tolerances of `1e-12`.

The PCG64 seed is `20260916`. Priorities are fixed per candidate across shots,
so results do not depend on worker scheduling or prior decoder calls. Prefixes
`K=1,4,8,16,24` share the same candidates. `K=1` means one random forest, not the
ordinary UF merge forest. The report presents all prefixes without selecting
one using evaluation labels.

This adapts the paper's exploration idea to our existing weighted clustering
and open boundaries. It does not reproduce the authors' periodic-code noise
experiment, graph compression, or hardware design. The first-pass correction
and resulting correlation weights remain unchanged. Ensembles that change
that first-pass evidence are outside this experiment.

## Results

| Decoder | d=7 failures | d=7 normalized LER | d=9 failures | d=9 normalized LER |
|---|---:|---:|---:|---:|
| Correlated MWPM | 13,785 | 0.000890785900 | 6,925 | 0.000333684421 |
| Correlated UF | 23,230 | 0.001598998922 | 14,247 | 0.000718249245 |
| Correlated UF + 24-candidate size vote | 23,230 | 0.001598998922 | 14,247 | 0.000718249245 |
| Correlated UF + 24-candidate weight vote | 23,230 | 0.001598998922 | 14,247 | 0.000718249245 |

All denominators are 100,000 shots. Both ensemble rules have zero repairs and
zero regressions at both distances. Every prefix has the same failure indicator
as ordinary correlated UF on every shot. At d=7, the 24-candidate size vote
changes one logical prediction, but both its old and new predictions are wrong.
The 24-candidate weight vote agrees with the baseline on every shot. At d=9,
all tested votes agree with the baseline on every shot.

The ensemble therefore retains correlated UF's LER gap to correlated MWPM:
approximately **1.80x at d=7** and **2.15x at d=9**. Full prefix results,
confidence intervals, and paired comparisons are in
[`tables.md`](coset_ensemble_d7_d9_p003_100k/tables.md).

Normalized LER uses the prior experiment's exact convention:
`sinter.shot_error_rate_to_piece_error_rate(failures / shots, pieces=6*4*d, values=8)`.
The raw block failure rate is reported separately in the machine-readable
summary. Confidence intervals use 10,000 paired whole-shot bootstrap replicates
with seed 43. Identical observed failure indicators give an empirical paired
difference of zero; this is not a claim about every future sample.

## Why physical diversity does not imply logical improvement

Let `A` contain all graph edges internal to the final partitions, `H_A` be their
detector incidence matrix, and `L_A` their logical labels. If `u` is the baseline
correction, the complete set of attainable logical answers is

`L(u) XOR L_A(ker H_A)`.

The decoder computes the dimension of this logical space using fundamental
cycles and binary elimination. Boundary-to-boundary paths are included. At
rank zero, every valid candidate has the same logical mask as the baseline,
regardless of priorities, ensemble size, weights, or voting rule.

| Diagnostic | d=7 | d=9 |
|---|---:|---:|
| Shots with 24 distinct physical corrections | 100,000 | 100,000 |
| Shots whose permitted logical space has rank zero | 99,999 | 100,000 |
| Shots with multiple sampled logical classes | 1 | 0 |
| Baseline failures with a correct sampled candidate | 0 | 0 |
| Baseline failures repairable anywhere inside the partitions | 0 | 0 |

All shots produce 24 distinct physical corrections, so the sampler is changing
paths. For d=7, **99,999 of 100,000 shots have logical rank zero**. On shot
26816, the remaining rank-one case, the only possible logical masks are 3489
and 3499; the actual mask is 4009. Both logical classes are sampled, and neither
can succeed. Correlated MWPM also predicts 3499 on this shot. An independent
DSU calculation confirms that case and 32 random rank-zero controls per distance.
For d=9, **all 100,000 shots have logical rank zero**.

Consequently **none of the 23,230 d=7 or 14,247 d=9 baseline failures can be
repaired inside these partitions**, even with unlimited forest candidates.
This includes every shot where correlated MWPM succeeds and correlated UF
fails: 12,915 shots at d=7 and 9,522 at d=9. Improving those cases requires
changing which corrections clustering admits, or the correlation evidence
that determines that clustering. Increasing K or changing the vote alone
cannot repair them on this sample.

This conclusion concerns the fixed second-pass partitions in this experiment.
It does not rule out ensembles that change the first-pass correction,
correlation weights, or growth process.

The physical corrections also tend to be longer in this adaptation. The
minimum edge count among 24 random forests exceeds the baseline by an average
of 42.15 edges at d=7 and 103.61 at d=9. No sampled candidate beats the baseline's
summed adjusted weight at either distance. The logical-space certificate is
independent of this sampler quality: even an exact optimizer inside these
partitions cannot repair the observed failures.

## Software cost

After collection finished, a sequential benchmark decoded 32 randomly selected
saved shots per distance, with one untimed warmup and rotating decoder order.
Construction and graph import are excluded. These are mean milliseconds per
complete shot:

| Implementation | d=7 | d=9 |
|---|---:|---:|
| Native PyMatching correlated MWPM | 2.740 | 7.151 |
| Repository Python correlated UF | 161.252 | 431.500 |
| Python 24-candidate ensemble, including rank diagnostic | 201.249 | 509.788 |

The ensemble adds approximately 25% and 18% to this Python UF implementation's
time. This compares Python UF with native PyMatching and **does not measure
hardware latency or establish a hardware advantage**. No FPGA design or
synthesis was performed. Row IDs, environment, source hashes, medians, and
95th percentiles are in
[`benchmark.json`](coset_ensemble_d7_d9_p003_100k/benchmark.json).

## Validation and reproduction

The implementation is in
[`_coset_ensemble.py`](../../src/yoked/decoders/_coset_ensemble.py), with tests in
[`_coset_ensemble_test.py`](../../src/yoked/decoders/_coset_ensemble_test.py).

The decoder suite passes **136 tests, with one skipped**. New tests cover
exhaustive small-graph syndromes and logical spaces, parallel boundary edges,
logical masks wider than 64 bits, disconnected boundary clusters, voting,
prefix reproducibility, and a noisy Stim circuit with correlation reweighting.

For every evaluation shot, collection generates all 24 candidates, checks each
candidate's detector parity and observable XOR, and compares fresh ordinary
correlated UF with the saved baseline. It uses no rank-zero shortcut. Reporting
recomputes votes independently from saved candidate costs and logical labels.
Candidate construction never reads actual observables or the MWPM predictions.
In total, **4.8 million candidates** pass these checks, and fresh baseline
predictions agree with all **200,000 saved correlated-UF predictions**.

The raw run and all resumable chunks are under
`$TMPDIR/coset-ensemble-d7-d9-100k-a8hHNP`. Repository artifacts are in
[`coset_ensemble_d7_d9_p003_100k/`](coset_ensemble_d7_d9_p003_100k/), including
compact per-shot predictions and diagnostics. Full candidate costs remain in
the raw run, with their hashes recorded in the copied manifests.

From the repository root:

```bash
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/uf-soft-mpl"
coset_run_dir=$(mktemp -d "$TMPDIR/coset-ensemble-d7-d9-100k-XXXXXX")
.venv/bin/python docs/results/coset_ensemble_d7_d9_p003_100k/run.py \
  --output "$coset_run_dir" --workers 96
.venv/bin/python docs/results/coset_ensemble_d7_d9_p003_100k/report.py \
  --root "$coset_run_dir"
.venv/bin/python docs/results/coset_ensemble_d7_d9_p003_100k/certify.py
.venv/bin/python docs/results/coset_ensemble_d7_d9_p003_100k/benchmark.py \
  --output docs/results/coset_ensemble_d7_d9_p003_100k/benchmark.json
```

The report defaults to requiring exactly 100,000 shots per distance. The
collection request binds the seed, source hashes, input hashes, and chunk size;
resuming accepts a different worker count but refuses a changed request or
corrupt completed chunk.
