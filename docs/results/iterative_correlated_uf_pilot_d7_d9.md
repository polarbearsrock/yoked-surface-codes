# Iterative correlated UF: an eight-pass pilot

**Repeated correlation feedback can reach correct logical candidates that the
original two-pass decoder misses.** The simple iteration tested here also
frequently cycles, and later passes can lose an earlier correct answer. It is
a useful direction for further study, with no confirmed full-population LER
improvement yet.

This is a bounded diagnostic run, not another 100k-shot decoder benchmark.
It reuses the existing 1,024-shot stratified sample at each distance: 256 from
each of the four original correlated-UF/correlated-MWPM outcome groups. The
original conditions remain `p=0.003`, SI1000, six patches, two ideal yokes,
`d=7,9`, `rounds=4d`. No physical faults are removed for this probe.

## Iteration tested

Let `s` be the unchanged measured syndrome, `w0` the original graph weights,
and `R(w0, c)` the repository's existing pairwise correlation reweighting based
on the selected physical edges of correction `c`:

```
c1 = UF(s, w0)
w2 = R(w0, c1); c2 = UF(s, w2)   # existing correlated UF
w3 = R(w0, c2); c3 = UF(s, w3)
...
w8 = R(w0, c7); c8 = UF(s, w8)
```

Every pass creates a fresh growth state. Weights are rebuilt from the original
priors before applying the latest correction's evidence. Discounts from obsolete
corrections are not retained. This is simultaneous hard-decision feedback on the
joint graph, with no damping, no message passing, and no matching solve in any
decoding pass. The correlation rule uses a minimum over supported implied weights,
not a full physical-fault posterior.

Every fixed pass count is reported. There is no truth-based candidate selection,
no selection of a best pass count from the observed LERs, and no scoring each
candidate using its own favorable reweighted model to choose an output.

## Estimated LER

Stratum results are weighted by their original population frequencies.
Normalized LER uses the preceding convention with `pieces=6*rounds, values=8`.
The two-pass and MWPM baselines are known from the full saved 100k records;
all other entries are estimates from the diagnostic sample.

| Total UF passes | d=7 normalized LER | d=9 normalized LER |
|---|---:|---:|
| 1 | 0.00389098 | 0.00234920 |
| **2: existing correlated UF** | **0.00159900** | **0.000718249** |
| 3 | 0.00138656 | 0.000757571 |
| 4 | 0.00135690 | 0.000674912 |
| 5 | 0.00134345 | 0.000714880 |
| 6 | 0.00136338 | 0.000685077 |
| 7 | 0.00132365 | 0.000713409 |
| 8 | 0.00133043 | 0.000677049 |
| Correlated MWPM reference | 0.000890786 | 0.000333684 |

Eight passes have an estimated relative LER reduction of **16.8% at d=7 and
5.7% at d=9** versus two passes. More iterations are not monotonically better.

The estimated reduction in whole-shot failure probability at eight passes is
3.438 percentage points at d=7 and 0.753 points at d=9. The ordinary paired
stratified bootstrap intervals are `[2.263, 4.505]` and `[-1.213, 2.503]` points.
Because the both-correct stratum is large and some transitions are rare, the
archive audit also computes conservative finite-population intervals that allow
unseen outcomes: **`[-0.013, 5.674]` points at d=7 and `[-3.439, 3.560]` points
at d=9**. These include zero at both distances. The d=7 point estimate is
encouraging; a full paired evaluation is needed to establish net benefit.

These intervals describe subsampling the existing 100k records, not generating
a new noise experiment. They are marginal per fixed pass count, with no
simultaneous significance claim across the seven comparisons against pass two.

## The process usually cycles

Exact repeated states are detected by comparing the complete sets of selected
physical correction edges. For this undamped, memoryless update, repeating that
set determines all subsequent weights and corrections. Equality of logical
predictions alone is not a convergence test.

| State by pass 8, population-weighted estimate | d=7 | d=9 |
|---|---:|---:|
| Fixed physical correction, period 1 | 4.0% | 0.0% observed |
| Repeating period 2 | 85.1% | 82.3% |
| Repeating period 3 or 4 | 3.7% | 6.3% |
| No repeated correction observed yet | 7.2% | 11.3% |

Physical oscillation need not change the logical prediction. The estimated
population fraction whose detected cycle includes different logical answers
is **11.9% at d=7 and 8.3% at d=9**. This distinction matters when interpreting
the much larger physical-cycle rates.

A possible feedback mechanism is that correction A makes the partners used by
correction B cheap; selecting B then changes which discounts are applied, causing
UF to return to A. The measured cycles establish repeated corrections, not a
complete classification of which physical mechanisms cause each cycle.

Some shots have the same logical answer at passes two and three but change at a
later pass. Within the UF-only failure cohort this occurs for 17/256 d=7 cases
and 10/256 d=9 cases. Stopping on one repeated logical label would miss this
subsequent exploration.

## Candidate generation works; output selection remains a problem

Among the 256 originally UF-only failures per distance:

| Diagnostic | d=7 | d=9 |
|---|---:|---:|
| At least one correct answer in passes 3–8 | **133/256** | **161/256** |
| Correct answer appears after pass 2, with neither initial pass correct | **81/256** | **99/256** |
| Correct at pass 8 specifically | 70/256 | 84/256 |

The first row uses truth only for diagnosis. It is an oracle statement about
candidate availability, not an achievable decoder output policy.

Some later successes return to a correct answer already present at pass one.
The second row separates the additional correct answers that appear only after
both original passes have failed. Plain UF at pass one was already correct on
75/256 and 81/256 of these cases, respectively; its correctness did not always
survive the first correlation update.

Repeated reweighting can change the final cluster partition and therefore escape
the limitation of the preceding ensemble, which varied forests within fixed
clusters. The correct candidate is nevertheless often lost on a later pass.
Regressions also matter: pass eight spoils 2/256 and 8/256 originally both-correct
shots, respectively, and 39/256 and 28/256 originally MWPM-only failures. These
cohorts have different population sizes; their raw counts cannot be subtracted
from repairs to estimate overall benefit.

## Implications for a follow-up

Three separate hypotheses are worth testing under a fixed compute budget:

1. **Damped feedback:** blend the next weights with the current weights, for
   example with coefficient 1/2, while deriving the proposed update from the
   original priors. This may reduce oscillations; improved LER is untested.
   A damped decoder's state includes its weights, so repeating a correction alone
   would no longer certify a cycle or fixed point.
2. **A few alternative clusterings:** retain candidates from different reweighted
   UF passes and evaluate a fixed, shared scoring rule. This requires a useful
   correlation-aware score; selecting by each candidate's own conditioned cost
   compares different models and can reward its own assumptions.
3. **Soft physical-fault evidence:** a bounded message-passing stage can infer
   weights from the syndrome and the full detector error mechanisms instead of
   relying solely on a single previous correction. The published belief-find
   decoder combines belief propagation with weighted UF; it provides a relevant
   starting point, not a measured result for our yoked circuit. See
   [Higgott et al., Phys. Rev. X 13, 031007](https://doi.org/10.1103/PhysRevX.13.031007).

Related iterative reweighting has also been studied with MWPM. That algorithm
alternates updates between X and Z lattices and uses matching at each step;
its convergence result does not establish convergence of the simultaneous UF
feedback tested here. See
[Tian et al., Iterative Lattice Reweighting](https://arxiv.org/html/2509.06756v1).

A hardware design could reuse a UF engine and precomputed correlation tables,
but the passes are sequentially dependent. Four total passes require roughly
twice the UF solves of the current two-pass decoder; eight require four times
as many. Memory traffic, weight precision, area, and hardware latency have not
been measured. A fixed pass cap provides a bounded workload.

## Validation and reproduction

All 2,048 second-pass predictions exactly reproduce the saved correlated-UF
baseline. Every correction in every pass is checked against the complete
original syndrome, and its logical mask is recomputed from its selected edges.
Eight shots per distance additionally compare the second-pass physical edge set
against the production correlated decoder. Original sample and DEM hashes are
verified. No production decoder file was changed.

The [probe](iterative_correlated_uf_pilot_d7_d9/probe.py), archived requests,
per-shot results, and summaries are in the companion directory. Run the probe
with `PYTHONPATH=src`, the repository `.venv/bin/python`, `--distance 7` or `9`,
and a new `--output-dir` under `$TMPDIR`; `--workers 48` was used here. The
[audit](iterative_correlated_uf_pilot_d7_d9/audit.py) reproduces the weighted
estimates and conservative intervals using the finite-population helper from
the preceding physical-fault analysis.

Original run outputs are retained at
`$TMPDIR/iterative-correlated-uf-pilot-k12YiS`.
