# Soft evidence improves UF accuracy in a paired pilot

Syndrome-dependent evidence substantially improves the existing UF decoder
without changing its growth, stopping, or peeling rules. A five-iteration
belief-propagation pre-decoder reduces **whole-shot failures by 37.4–49.5%**
relative to correlated UF across d=7,9,11,13. Twenty iterations reduce
failures by **41.9–47.6%**. Both budgets were specified before evaluation;
there was no tuning or selection of the better answer per shot.

This supports testing richer evidence before UF commits to its clusters.
It does not establish a decoder that is faster than the current correlated
UF, or superior to correlated MWPM.

## What was held fixed

SI1000 p=0.003, six patches, two ideal yokes, CZ circuits, rounds=4d. Each
distance uses 5,000 uniformly selected rows from the original 100,000-shot
experiment. All variants decode identical shots. The full parent samples
were reproduced and their payload hashes checked; these are historical
evaluation shots, not a fresh independent confirmation set.

The pre-decoder runs damped sum-product BP on the joint DEM fault–detector
graph. A correlated fault remains one variable, including the cancellation
of detector incidences across its graph components. The resulting soft
estimates change the weights on the existing graph. UF always supplies the
final correction on the original syndrome. There is no graph pruning,
preliminary hard correction, new growth rule, bridge extension, or use of
truth or MWPM predictions as decoder inputs.

The nonnegative weight projection follows the approach in
[Higgott et al., Appendix C](https://arxiv.org/html/2203.04948v5#A3).
Our fixed budgets, damping, and unchanged repository UF make this a
controlled evidence experiment, rather than a reproduction of the paper's
complete belief-find decoder. The [frozen protocol](uf_evidence_predecoder_d7_d13/README.md)
specifies every setting and approximation.

## Failure counts

Every entry is failures out of **5,000 shots**. A failure means at least one
of the 12 logical-observable predictions is wrong.

| d | Plain UF | Current correlated UF | Prior projection + UF | BP5 + UF | BP20 + UF | Correlated MWPM |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 2,335 | 1,173 | 2,334 | 734 | 681 | 713 |
| 9 | 2,027 | 711 | 2,027 | 421 | 380 | 347 |
| 11 | 1,584 | 376 | 1,588 | 213 | 197 | 157 |
| 13 | 1,256 | 196 | 1,256 | 99 | 106 | 76 |

Prior projection uses exactly the BP probability-to-weight conversion,
applied to prior probabilities without syndrome evidence. Its results are
almost identical to plain UF. This control separates the benefit of
syndrome-dependent inference from the change in weight convention.

## Repairs and regressions

These compare each BP variant with current correlated UF on the same shots.
The intervals are paired 95% normal intervals for the reduction in raw
whole-shot failure probability, expressed in percentage points.

| d | BP5 repairs / regressions | BP5 reduction, pp (95% CI) | BP20 repairs / regressions | BP20 reduction, pp (95% CI) |
|---:|---:|---:|---:|---:|
| 7 | 632 / 193 | 8.78 [7.68, 9.88] | 710 / 218 | 9.84 [8.68, 11.00] |
| 9 | 427 / 137 | 5.80 [4.88, 6.72] | 505 / 174 | 6.62 [5.62, 7.62] |
| 11 | 250 / 87 | 3.26 [2.55, 3.97] | 288 / 109 | 3.58 [2.81, 4.35] |
| 13 | 150 / 53 | 1.94 [1.38, 2.50] | 156 / 66 | 1.80 [1.22, 2.38] |

Every comparison favors BP even after counting the new failures it creates.
The exact paired McNemar p-values for these eight comparisons are all below
1.4e-9. The analyses are exploratory; the displayed intervals are not
adjusted for multiple comparisons.

More iterations do not consistently lower the observed error count. At
d=13, BP5 has 99 failures and BP20 has 106; their paired difference is
uncertain (McNemar p=0.616). Conversely, BP20's 681 versus MWPM's 713 at d=7
does not establish an advantage over MWPM (paired p=0.212). Correlated MWPM
has fewer observed failures than both BP variants at the other distances.

Four finite distances and one noise strength do not establish a threshold
or an asymptotic suppression law. The detailed report gives normalized LERs
under the same convention as the earlier experiments.

## How the evidence changes UF's available logical corrections

Select 32 shots per distance on which original correlated UF fails and
correlated MWPM succeeds. Restore every original edge internal to each
final UF component, and test whether a syndrome-preserving change can reach
the true logical class. This diagnoses 128 conditional disagreements; it
is not a population error-rate estimate.

| d | Original UF partition admits truth | BP5 partition admits truth | BP20 partition admits truth |
|---:|---:|---:|---:|
| 7 | 0 / 32 | 24 / 32 | 30 / 32 |
| 9 | 0 / 32 | 25 / 32 | 29 / 32 |
| 11 | 0 / 32 | 24 / 32 | 25 / 32 |
| 13 | 0 / 32 | 25 / 32 | 26 / 32 |

BP5 makes the correct class reachable in **98/128** cases, and BP20 in
**110/128**, compared with **0/128** for the original clusters. In these
selected cases, accessibility on the merge forest, accessibility on the
induced partition, and actual logical success agree. All remaining sampled
BP failures still exclude the true class from their final partitions.
These diagnostics do not cover the new regressions caused by BP.

The weights therefore change which useful connections UF reaches under its
existing rules. Improving evidence helps expose the alternatives that the
earlier diagnosis found missing, while leaving a residual exploration
problem.

## Preprocessing cost

The accuracy gain is expensive in this implementation. Relative to current
correlated UF, BP5 takes **4.0–5.8 times** the measured serial kernel time;
BP20 takes **14.6–21.8 times**. These are means over 16 predetermined shots
at each distance, charging all evidence work to each variant and both UF
passes to the correlated baseline.

| d | Correlated UF total, ms | BP5 total, ms | BP20 total, ms |
|---:|---:|---:|---:|
| 7 | 16.77 | 97.57 | 366.28 |
| 9 | 44.03 | 226.99 | 855.45 |
| 11 | 103.95 | 464.30 | 1,726.98 |
| 13 | 200.24 | 796.55 | 2,915.47 |

These timings exclude model compilation, Python orchestration, and the
post-solve syndrome audit. They are software measurements, not FPGA latency,
area, or throughput results. The [timing environment](uf_evidence_predecoder_d7_d13/benchmark_environment.json)
records the machine, compiler, and measurement scope.

![Accuracy and software cost](uf_evidence_predecoder_d7_d13/comparison.png)

The next useful test is a cheaper approximation to this evidence layer,
such as a bounded min-sum implementation, with the current BP variants as
accuracy controls. The experiment does not yet measure confidence
calibration or a separate L1/L2 hierarchy.

## Reproduction and detailed evidence

The [experiment directory](uf_evidence_predecoder_d7_d13/README.md) contains
the runnable protocol and scripts. The [detailed report](uf_evidence_predecoder_d7_d13/report.md)
includes all controls, normalized LERs, conditional logical-accessibility
diagnostics, separate evidence/UF timings, and validation details. Each
distance has compact per-shot arrays, its selected row IDs, source and
sample identities, and verification records. Builds, raw parent samples,
and checkpoints remain under `$TMPDIR`.

The production decoder is unchanged. The experiment imports the previously
verified native UF kernel and checks its new weighted forests and physical
corrections against production Python UF.

All **100,000 pilot corrections** (five variants on 20,000 shots) pass the
full-syndrome checks, and all **20,000** correlated-UF predictions reproduce
their archived baseline. Independent checks cover exact BP posteriors on
acyclic models, zero messages, high-degree checks, correlated detector
cancellation, real-shot BP values and physical UF corrections, and thread
invariance. The full verification records are retained with the results.
