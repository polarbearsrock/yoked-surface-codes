# Hierarchical L1/L2 decoding experiment design

**Status:** approved design, revised after review on 2026-09-14.
Implementation plan to follow.

**Goal.** Split decoding of the 1D yoked surface code into a patch-local
layer (L1) that emits a reference correction and a confidence score per
patch and sector, and an outer layer (L2) that uses those confidences and
the yoke syndrome to decide which patches to flip. Then measure whether L2
can obtain most of the benefit of better confidence information by asking
for refined confidences from only a few patches.

**Question.** With L1 reference corrections held fixed, how much of the
improvement from replacing every patch's initial confidence with a refined
one is recovered when only k of six patches per sector are selected for
refinement, and does an informed choice beat a random choice with the same
trigger and selection budget? Report both the patch-sector selections and
the distinct patches and matching calls needed to provide them.

This document fixes the definitions, data, code structure, metrics, and
acceptance tests. It does not add asynchronous message arrival, streaming
windows, a third layer, noisy lattice-surgery yokes, or 2D yokes.

## 1. Facts about the workload that shape the design

The workload is the repository's `yoked_magic_memory_circuit` at
`patch_diameter=9, rounds=36, noise=si1000(0.003), style='cz', yokes=2,
num_patches=6`, decoded from the DEM built with `decompose_errors=True,
approximate_disjoint_errors=True`. These facts were measured on that DEM
and on the saved seed-42 100,000-shot sample on 2026-09-14.

- The two yoke detectors are the last two detectors. Each is an ideal
  end-of-block parity check: on every sampled shot the X yoke bit equals
  the XOR of the six X observables (even observable indices), and the Z
  yoke bit equals the XOR of the six Z observables (odd indices).
- Each yoke detector has degree 1,110 in the imported decoding graph. The
  median detector degree is 11 and no non-yoke detector exceeds 12. These
  are degrees after parallel-edge merging by the audited importer. The
  counts 14,556, 70, and 82 are repeated detector-target occurrences in the
  flattened DEM, not graph degrees. The yoke remains a hub: UF growth from
  a fired hub can spread into every patch.
- With the two yoke detectors removed, the DEM has 12 connected
  components, one per patch and sector, each with 1,480 detectors and
  exactly one observable. Each decomposed error component has at most two
  detectors, retains a physical detector when the yoke is removed, and has
  yoke membership matching its observable's sector.
- Every observable-flipping component has exactly one physical detector.
  In the yoke-free view such components are boundary edges.
- Up to matching ties, joint MWPM on the hub graph equals patch-local
  MWPM plus "flip the patch with the smallest complementary gap when the
  frame-adjusted yoke fires". On the saved sample joint MWPM is wrong on
  0.03% of sectors with no patch-local failure and on 99.6% of sectors with
  two or more.
- Patch-local uncorrelated MWPM fails on 6.3% of patch-sectors. Per sector,
  27.2% of shots have exactly one failed patch and 5.1% have two or more.
  Joint MWPM misattributes 32.2% of the exactly-one cases.

Consequences: the check-free graphs have 12 disconnected sectors per
shot, with DEM correlations still relating X and Z edges within a patch.
The yokes supply the outer parity constraints. Single-failure
misattribution is the primary diagnostic, accompanied by overall block
failure and outcomes stratified by zero, one, and multiple reference
failures. Multiple failures are not an irreducible floor for the general
L2 in section 7. The MWPM-reference pipeline has a known answer up to
matching ties, which becomes the end-to-end validation test.

## 2. Notation

- Patches `i in {0,...,5}`, sectors `s in {X, Z}`. Observable index
  `o(i, X) = 2i`, `o(i, Z) = 2i + 1`. The yoke detector for sector `s` is
  `D[n_d - 2]` for X and `D[n_d - 1]` for Z, where `n_d` is the number of
  detectors.
- `a[i, s]`: sampled observable flip. `y[s]`: sampled yoke bit, equal to
  the XOR over `i` of `a[i, s]`.
- `r[i, s]`: L1 reference bit, the logical prediction of the reference
  decoder for that patch and sector.
- `e[i, s] = a[i, s] XOR r[i, s]`: residual error after the reference.
- Frame-adjusted syndrome `sigma[s] = y[s] XOR (XOR over i of r[i, s])`,
  which equals the parity of the residual errors.
- `q0[i, s]`: initial calibrated residual-error probability.
  `q1[i, s]`: refined calibrated residual-error probability.
- L2 output `x[i, s] in {0, 1}`; final prediction `f = r XOR x`.
- `M[i, s]`: the policy requests the refined score for this patch-sector;
  only these scores replace `q0`. `U[i] = M[i, X] OR M[i, Z]`: the patch
  requires refinement work. A touched patch computes both sector gaps,
  but an unrequested score is not substituted into L2.
- A shot succeeds when `f == a` on all 12 observables. This is the scoring
  used by the existing four-decoder comparison and is kept unchanged.
- All weights, gaps, and scores are in nats: edge weight `ln((1 - p) / p)`.

## 3. Data plan

Three sample sets, each one Stim sampling call on the same circuit, with
`separate_observables=True`, recorded with the SHA-256 of the packed
detector bytes followed by the packed observable bytes, as the existing
comparison docs do.

| Set | Stim seed | Shots | Use |
|---|---:|---:|---|
| Evaluation | 42 | 100,000 | The saved four-decoder sample. Exploratory comparisons and policy development. |
| Calibration | 142 | 50,000 | Fitting calibrators only. Never used for evaluation. |
| Confirmation | 242 | 100,000 | Sample, collect, and evaluate once, only after the analysis is frozen. Reported beside the evaluation numbers. |

The saved seed-42 arrays are reused from the recorded run directory after
verifying their hash. First run a small d=9 pilot on the first 2,000 shots
of this saved sample and the first 2,000 shots of the calibration sample.
Generate the calibration sample in its single full 50,000-shot call, but
initially decode only the pilot subset. Fit pilot calibrators only on the
calibration subset and assess endpoints only on the evaluation subset.
Pilot records retain their parent sample hash and row indices; do not
resample with a smaller call or treat these subsets as independent sets.

The pilot precedes full L1 collection and tests whether initial-only and
all-refined separate enough to justify the full experiment. It is
exploratory, and its shots remain part of their respective full sets. If
the pilot supports proceeding, reuse its L1 outputs when graph, decoder,
and collection provenance still match, and collect the remaining
calibration and evaluation rows. Otherwise invalidate and recollect the
affected pilot outputs. At full size, calibration has
50,000 x 6 patch outcomes per sector, with roughly 6% to 9% positives
(about 18,000 to 27,000 residual-error events per sector).

After policy development, freeze the calibrators, estimator definitions,
policy configurations, tie rules, work accounting, reliability bins,
metrics, and analysis code in a manifest with source and artifact hashes.
Only then sample and collect confirmation. Confirmation results cannot be
used to revise these choices while retaining the same set as confirmation.
Distance 7 is a replication after distance 9 is complete, using its saved
seed-42 sample and the same pilot, three-set, and freeze procedure.

## 4. Patch graphs: splitting the hub

The splitter derives per-patch inputs from the six-patch DEM used by the
joint decoders, not from regenerated single-patch circuits, so that the
per-patch graphs contain literally the edges and weights the joint decoders
saw. This makes the validation in section 10 exact.

**Local DEMs.** Flatten the six-patch DEM. Assign each physical detector
to a patch by connected components after removing the yoke detectors; the
component's observable gives the patch and sector. For each error
instruction, rewrite every component with local detector ids and local
observable ids (0 for the patch's X observable, 1 for Z) and drop yoke
targets, keeping the instruction's component structure and probability
intact. Components of one instruction always fall in one patch because
patches share no gates; the splitter raises if they do not. The result is six `stim.DetectorErrorModel`
objects with 2,960 detectors and 2 observables each. The splitter raises
if any component has no physical detector, more than two physical
detectors, or a yoke membership that disagrees with its observable sector.

**Check-free graph.** `DecodingGraph.from_dem(local_dem)`, which reuses
the existing audited importer. This graph has two connected components,
one per sector. It is the graph L1 decodes.

**Check graph.** The same edge list with two extra vertices, local ids
2,960 and 2,961, the X and Z check detectors. Every boundary edge whose
observable mask has the sector's bit is re-targeted from its own terminal
to the sector's check vertex. Edge ids are unchanged, so weights and
correlation rules apply to both graphs by edge id. Setting a check bit in
a syndrome forces the matching into the corresponding logical class,
exactly as the repository's gap circuits do with their check detector.

**Local syndromes.** A patch's local syndrome is the global syndrome
gathered at that patch's sorted global detector ids. The record keeps the
global ids so that any local result can be mapped back.

Public interface:

```python
patches = PatchGraphs.from_yoked_dem(dem, num_patches=6)
patch = patches[i]
patch.global_detector_ids   # sorted global ids of its 2,960 detectors
patch.observable_ids        # (o(i, X), o(i, Z)) as global observable indices
patch.local_dem             # stim.DetectorErrorModel, local ids
patch.graph                 # DecodingGraph, check-free
patch.check_graph           # DecodingGraph with check vertices
patch.check_vertices        # (2960, 2961)
patches.yoke_detector_ids   # (n_d - 2, n_d - 1)
local = patches.local_syndromes(global_syndromes)   # shape (shots, 6, 2960)
```

## 5. L1: reference decoders and soft outputs

L1 reference decoding runs once per patch on the check-free graph and
reports a reference bit per sector. The score calculations use only local
physical syndromes; they never receive a sampled yoke bit. Matching gaps
use synthetic check bits to force each class as defined in section 5.2.
Scores are defined so that a larger score means more confidence that the
reference bit is right, and calibration in section 6 maps them to
residual-error probabilities.

### 5.1 UF reference and the cluster gap

The reference decoder is the repository `UnionFindDecoder` on
`patch.graph`. Its correction is validated as in the existing tests:
`H c = s` over GF(2) and `L c = r`.

The initial score uses the cluster-gap construction of
[Meister, Pattison, and Preskill](https://arxiv.org/html/2405.07433)
(Definition 9), evaluated on the terminated growth state of the
repository's weighted UF with the following edge-length convention:

- Settle all edges at the final growth time. For edge `e` with weight
  `w_e` and accumulated growth `g_e`: if both endpoints are in one cluster
  the cost is 0, otherwise the cost is `max(0, w_e - g_e)`, the growth that
  remains before the edge would complete.
- Contract every boundary terminal into one boundary super-vertex `B`.
- The cluster gap for sector `s` is the minimum cost of a walk from `B`
  back to `B` whose edges flip observable `o(i, s)` an odd number of times.
  It is found by one Dijkstra search over states `(vertex, parity)` from
  `(B, 0)` to `(B, 1)`, where traversing an edge toggles the parity when
  the edge's mask has the sector's bit. Edges of the other sector's
  component never toggle this parity, so they lie on no shortest odd walk;
  the implementation restricts the search to the sector's component.

For this patch graph every odd-parity closed walk must use the boundary,
because observable-flipping edges are boundary edges. The parity search
therefore implements the shortest path between inequivalent boundaries
with cluster interiors free. Remaining edge growth is the explicit metric
used for this weighted UF state; with no partial growth it reduces to
zero cost inside clusters and full edge weight outside. The small-graph
tests verify this convention, including partial growth. Record the number
of settled Dijkstra states per patch-sector as the initial soft-output
work proxy.

This count describes the initial soft-output work, which is incurred on
all patches. A small refinement count alone does not establish a latency
or throughput improvement; cluster-gap extraction itself has overhead, as
discussed by [Kishi et al.](https://arxiv.org/html/2602.03336). Latency
measurement remains outside this experiment.

```python
decoder = ClusterGapUnionFindDecoder(patch.graph)
result = decoder.decode_with_gaps(local_syndrome)
result.prediction        # (2,) bool, reference bits for X and Z
result.cluster_gap       # (2,) float, nats
result.dijkstra_states   # (2,) int
result.selected_edges    # edge ids of the validated correction
```

### 5.2 Matching gaps: forced weights, plain and correlated

Refined scores come from minimum-weight matching in each logical class.
For a patch, `W(c_X, c_Z)` is the matching weight on the check graph with
the X check bit set to `c_X` and the Z check bit set to `c_Z`. All four
combinations are computed and stored, so every later quantity is
independent of which reference decoder is used. For sector X with the
other sector held at its reference bit, the signed gap relative to
reference `r` is

    delta_X = W(1 - r_X, r_Z) - W(r_X, r_Z)

and symmetrically for Z. A negative gap means the matcher prefers the
complement of the reference. The sign is kept; it is never folded into an
absolute value.

Two variants share one code path:

- **Plain.** Forced decodes on the check graph with the original weights.
  Because the sectors are disconnected, `W(c_X, c_Z) = W_X(c_X) + W_Z(c_Z)`
  exactly; the test suite checks this additivity.
- **Correlated.** One unforced decode on the check-free graph gives the
  first-pass edge set. The repository's DEM-derived correlation rules,
  today private in the correlated UF module and to be moved unchanged to a
  shared module, lower the weights of correlated target edges. The four
  forced decodes then run on the check graph with these adjusted weights.
  Both classes are therefore compared under one reweighted model, which is
  what makes the difference a gap. PyMatching's built-in correlated mode is
  not used here because it does not expose its reweighted model.

All matchers are constructed from `DecodingGraph` edge lists, not from
DEMs, so that adjusted weights can be supplied. The construction disallows
merging of parallel edges; the audited importer already guarantees there
are none.

Both sector gaps are produced by each four-class calculation. Refining
either sector therefore incurs the patch-level work; refining the other
sector of that same patch does not repeat it. The unforced correlated
second pass is collected for validation, not required to produce a gap.
The full offline collection computes both variants for replay; section 8
separately counts the calls each replay configuration would require.

```python
gaps = MatchingGaps(patch)                      # builds the matchers once
forced = gaps.forced_weights(local_syndrome)    # ForcedWeights
forced.plain        # (2, 2) float, indexed [c_X, c_Z]
forced.correlated   # (2, 2) float
forced.first_pass   # (2,) bool, the unforced plain prediction
forced.correlated_prediction   # (2,) bool, unforced second pass under the reweighted model
signed_gap(forced.correlated, reference=(r_X, r_Z))   # -> (2,) float
```

### 5.3 The stored L1 record

The collect stage evaluates L1 once per sample set and stores everything
downstream stages need. Nothing downstream re-runs a decoder.

| Array | Shape | Contents |
|---|---|---|
| `actual` | (shots, 12) | sampled observable flips |
| `yoke` | (shots, 2) | sampled yoke bits, X then Z |
| `uf_reference` | (shots, 12) | UF reference bits |
| `mwpm_reference` | (shots, 12) | plain patch-local MWPM prediction, the unforced first pass |
| `correlated_prediction` | (shots, 12) | unforced second pass under the reweighted model |
| `joint_mwpm` | (shots, 12) | joint PyMatching on the hub DEM, for validation test 6 |
| `joint_uf` | (shots, 12), optional | saved joint UF baseline on matching evaluation rows |
| `joint_correlated_uf` | (shots, 12), optional | saved joint correlated UF baseline on matching evaluation rows |
| `joint_correlated_mwpm` | (shots, 12), optional | saved built-in correlated PyMatching baseline on matching evaluation rows |
| `cluster_gap` | (shots, 12) | nats |
| `dijkstra_states` | (shots, 12) | cost proxy |
| `forced_plain` | (shots, 6, 2, 2) | `W(c_X, c_Z)` |
| `forced_correlated` | (shots, 6, 2, 2) | `W(c_X, c_Z)` under the reweighted model |

Column `2i + s` holds patch `i`, sector `s`. The record is saved as one
`.npz` beside a JSON manifest holding the circuit parameters, seed, shot
count, sample hash, package versions, source hashes, and timing.

Import the optional historical baselines only after verifying their sample
identity, prediction hashes, row mapping, and recorded implementation
provenance. They are absent on calibration and confirmation records;
historical evaluation results are never presented as confirmation
measurements. The built-in correlated MWPM baseline is distinct from the
fixed-reweighting gap estimator in section 5.2.

## 6. Calibration

An estimator is a pair (reference decoder, score). Each estimator gets a
calibrator for each sector, pooling the six patches, that maps its score
to `P(e[i, s] = 1)`, fit on the calibration set only. Fitting uses isotonic
regression with the pool-adjacent-violators algorithm implemented in NumPy,
with the monotone direction fixed by definition: decreasing in the cluster gap, decreasing
in a signed matching gap. Between adjacent PAV block centres the map is
linear and non-increasing, with constant extrapolation beyond the fitted
range. Strict monotonicity is not assumed; flat regions and clipping can
introduce ties. Outputs are clipped to `[1e-6, 1 - 1e-6]` so that log-odds
stay finite. Probabilities above one half are allowed: a refined signed
gap can favor reversing the fixed reference.

Calibration can affect both the initial-only and all-refined endpoints,
as well as selective refinement. Only in the restricted regime where
every probability is below one half does L2 make no flips for `sigma = 0`
and one flip at the largest probability for `sigma = 1`. A shared strictly
monotone transformation preserving that regime leaves this ranking
unchanged, apart from ties. In the general model, transformations can
change which probabilities exceed one half and which bit has the smallest
absolute log-odds, changing L2's answer even when rankings are unchanged.
The mixed cells also require the two calibrators to agree on a common
probability scale. Report the frequency of probabilities above one half
and calibration-induced ties for each estimator.

Reliability diagrams for every estimator are required on evaluation and,
after freezing, confirmation. In addition to overall reliability, report
refined-score reliability on the patch-sectors actually queried by each
policy, with sample counts, mean predicted probabilities, and observed
error frequencies. Break this out by initial-confidence bins where sample
counts permit. For random controls, use their exact selection weights.
Freeze the bin definitions before confirmation.

Interpret the queried-population diagrams in light of the policy trigger:
conditioning on `sigma` can change error prevalence even for exact local
probabilities. Alongside the raw-score diagnostic, compare observed errors
with the residual marginals of the L2 distribution after conditioning on
the observed parity. These marginals are obtained by summing the weights
of parity-compatible patterns with `x_i = 1` and normalizing by the total
compatible weight. They are diagnostic probabilities, not a change to
the MAP decision rule. Do not fit yoke-conditioned probabilities and then
feed them to L2 as independent local inputs, conditioning on the same
yoke twice.

Selection on `q0` supplies information that a marginal calibration of the
refined score alone may discard: `P(e = 1 | score1, selected)` need not
equal `P(e = 1 | score1)`. If the queried population shows systematic
miscalibration beyond the modeled parity conditioning, diagnose this
during exploration and consider a combined estimate using both initial
and refined scores. Any such extension must specify its model and fitting
procedure before confirmation, fit its
parameters on calibration data only, and retain the univariate estimator
as a baseline. It is not silently substituted for the specified estimator,
and satisfactory overall reliability alone is not evidence that the
selected population is calibrated.

```python
calibrator = IsotonicCalibrator.fit(scores, outcomes, direction='decreasing')
q = calibrator.probability(scores)
calibrator.to_json() / IsotonicCalibrator.from_json(...)
```

## 7. L2: MWPM outer decoder for the factorized model

Per sector, L2 receives six probabilities and the frame-adjusted syndrome
`sigma[s]`. It assigns to every residual pattern `x in {0,1}^6` the weight
`prod_i q_i^{x_i} (1 - q_i)^{1 - x_i}`, discards patterns whose parity is
not `sigma[s]`, and returns the pattern of maximum weight. Production replay
solves this objective with PyMatching MWPM. Enumeration over 64 patterns is
retained as an independent small-system validation oracle. Ties are broken toward the pattern
with the lowest binary value, bit `i` being patch `i`, and are counted. The final prediction is
`f = r XOR x`, and by construction its sector parity equals the yoke bit.

There is an independent analytic characterization. Let
`b_i = 1[q_i > 1/2]` and `lambda_i = ln((1 - q_i) / q_i)`. If `b` already
has parity `sigma`, it is an optimum. Otherwise toggle a bit with minimum
`abs(lambda_i)`. Starting from `b`, every changed bit costs
`abs(lambda_i)` in log weight, so the cheapest parity change suffices.
Handle equal costs and `q_i = 1/2` explicitly to return the lowest binary
optimum and count ties. This rule validates enumeration without repeating
the same brute-force algorithm.

For the matching graph, first take the preferred allowed bits
`b_i = 1[q_i > 1/2]`; restricted non-candidates stay zero. Decode the residual
parity `sigma XOR parity(b)` with nonnegative weights `abs(lambda_i)`, then
XOR the matching's selected patch bits into `b`. Each allowed patch has its
own two-edge path from the parity detector through an auxiliary detector to
the boundary. The first edge carries its log-odds cost and patch fault ID;
the second has zero weight. Auxiliary syndromes are zero. Subdivision keeps
distinct patch choices from being merged as parallel boundary edges.

PyMatching quantizes weights and has its own tie choices. Check its result
against the original floating-point costs. Resolve quantization discrepancies
and the existing absolute `1e-9` tie class using the analytic parity structure;
canonical tie resolution fixes bits from highest to lowest, choosing zero when
a completion remains within tolerance. Production replay must not enumerate
patterns. Collection correctness gates retain enumeration as an independent
small-block oracle. Source hashes and the PyMatching version belong to replay provenance;
existing L1 records and fitted calibrators remain reusable in new replay directories.

Multiple flips relative to the reference are allowed, even for an unfired
frame-adjusted yoke. For example, with
`q = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]` and `sigma = 0`, L2 returns
`x = [1, 1, 0, 0, 0, 0]`, repairing a reference with those two residual
errors. The experiment therefore has no presumed floor at two failures.

The maximum is over patterns, not over outer logical classes, because
success is scored per observable. The complement of a correct pattern
flips all six observables and is scored as six errors, so marginalizing
over classes would be inconsistent with the scoring.

A second rule, the candidate-restricted L2, is the same maximization with
unrefined patches fixed to `x_i = 0`. It never compares a refined
probability against an initial one and therefore isolates the selection
policy from cross-estimator calibration error.

It still depends on the refined probabilities and imposes a different
feasible set from mixed L2. In particular, with one candidate and
`sigma = 1`, parity forces that candidate to flip regardless of its refined
score. That cell measures selection alone, not the value of refinement.
With no candidates, `sigma = 0` returns the all-zero pattern; `sigma = 1`
is infeasible and raises explicitly. The listed selective policies always
provide at least one candidate when `sigma = 1`.

```python
decision = mwpm_outer_map(q, parity, candidates=None)  # candidates: bool mask or None
x, tied = decision.pattern, decision.tied
```

## 8. Refinement policies and offline replay

Replay reads the stored record and calibrators, builds `q0` from the
initial estimator and `q1` from the refined estimator, and for each shot
and sector lets a policy choose the set of patches whose `q1` replaces
`q0`, recorded as `M`. L2 then runs on the mixed vector. A policy sees only
`q0` and the frame-adjusted syndrome; never `q1`, the actual flips, or the
outcome. Deterministic policies break selection ties by increasing patch
index. Random controls specify a uniform distribution over eligible
subsets, integrated exactly during replay.

Configurations, all on the same shots and the same reference bits:

| Name | Refined set per sector | Purpose |
|---|---|---|
| `initial_only` | none | baseline hierarchy |
| `all_refined` | all six, regardless of `sigma[s]` | full-refinement comparator |
| `top_k_given_yoke` | the k largest `q0` when `sigma[s] = 1`, none otherwise; k in {1, 2, 3, 6} | primary selective policy |
| `top_k_uncertain` | the k smallest absolute log-odds of `q0`, regardless of `sigma[s]`; same k values | unconditioned selective policy |
| `random_k_given_yoke` | a uniform k-subset when `sigma[s] = 1`, none otherwise | control for `top_k_given_yoke` |
| `random_k_unconditional` | a uniform k-subset regardless of `sigma[s]` | control for `top_k_uncertain` |

`top_k_given_yoke(k=6)` equals `all_refined` on the exactly-one-failure
stratum, where `sigma = 1`, but need not equal it overall. It leaves
`sigma = 0` sectors unrefined and can miss multiple-error rescues there.
Keep this gated endpoint distinct in tables and plots.

**Exact random controls.** At each active sector, average over all
`binomial(6, k)` subsets with equal weights (at most 20); an inactive gated
sector has only the empty subset. The distribution is fixed without
access to refined scores or outcomes. Evaluating every branch computes
the policy's expectation, not an informed choice among the branches.
Subset choices are independent between X and Z. Store per-shot expected
sector and block failures, coverage, ties, and work. For example, if the
conditional sector failure probabilities on one shot are `h_X` and `h_Z`,
its expected block failure is `1 - (1 - h_X) * (1 - h_Z)`. These
expectations replace a single seed-dependent random realization in all
reported comparisons and are bootstrapped by whole shot.

The controls match eligibility and requested patch-sector counts. Their
distinct-patch costs can still differ from an informed policy because X/Z
selection overlap can differ; report these costs instead of assuming they
are equal.

Each selective configuration is run with both the mixed L2 and the
candidate-restricted L2. The primary estimator pair is UF reference,
cluster gap initial, correlated gap refined. The control pair is MWPM
reference, plain gap initial, correlated gap refined. The plain gap is also
run as a refined estimator for the UF reference as a secondary cell.

**Work accounting.** Keep separate counts for requested patch-sectors
`sum(M)` and distinct patches refined `sum(U)`. Refining a patch computes
both sector gaps once; L2 consumes only the scores marked in `M`. For
random controls, if the selection probabilities of a patch are `p_X` and
`p_Z`, its refinement probability is
`1 - (1 - p_X) * (1 - p_Z)`. This accounts for shared work exactly.

Count matching calls for the specified four-class procedure, separating
fixed initial work from incremental refinement work:

| Estimator configuration | Fixed initial work per patch | Incremental work per distinct patch refined |
|---|---|---|
| UF reference, cluster gap to plain gap | one UF decode and two cluster-gap searches | four plain forced matching calls |
| UF reference, cluster gap to correlated gap | one UF decode and two cluster-gap searches | one unforced plain matching call, one reweighting pass, four correlated forced matching calls |
| MWPM reference, plain gap to correlated gap | one unforced plain matching call and four plain forced matching calls | one reweighting pass using the retained first-pass edges, four correlated forced matching calls |

The four forced calls provide both sector gaps. Compiling graphs and
correlation rules is setup work. Offline collection additionally computes
validation predictions and all scores for every patch; record that actual
collection work separately from the work implied by each replay. These
counts characterize the specified procedure, not elapsed time or a claim
of minimal matching work. Any later optimization must update the frozen
accounting and avoid charging X and Z twice for shared work.

```python
config = ReplayConfig(reference='uf', initial='cluster_gap', refined='gap_correlated',
                      policy=TopKGivenYoke(k=2), outer='mixed')
result = replay(record, calibrators, config)
result.final            # (shots, 12) bool
result.refined          # (shots, 12) bool, M: scores requested and consumed
result.refined_patches  # (shots, 6) bool, U: patches requiring matching work
result.work             # per-shot call counts, with initial and incremental work separate
result.ties             # (shots, 2) bool
```

For an averaged random control, replay returns an `ExpectedReplayResult`
with per-shot expectations and weighted subset outcomes sufficient to
compute the metrics and transitions. It has no single `final` prediction
array. JSON summaries identify expected counts, which may be fractional;
the per-shot arrays retain the pairing needed for bootstrap comparisons.

## 9. Metrics and report

**Primary.** Misattribution rate: among sectors with exactly one residual
reference failure, the fraction where the final prediction is wrong. It is
reported per configuration with a paired 95% bootstrap interval,
resampling whole shots, 10,000 replicates, RNG seed 43. All configurations
within an estimator pair use the same reference and eligible sectors.
Different reference decoders define different eligible populations, so
their misattribution rates alone do not rank overall decoder quality.
For random controls, bootstrap their per-shot expected outcomes; do not
draw fresh subsets inside each bootstrap replicate.

Every headline primary result is accompanied by overall block failure
and its paired comparison to the relevant baseline, together with
refinement call counts. Improvement on the single-failure stratum alone is not
presented as an overall accuracy improvement.

**Recovery fraction.** For policy P at budget k,
`eta(k) = (m_initial - m_P(k)) / (m_initial - m_all)` on the primary
metric, with a paired bootstrap interval. Report the absolute differences
and denominator interval as well. If the initial-to-all-refined benefit
is nonpositive or its paired interval includes zero, label the recovery
fraction unresolved instead of quoting a stable percentage. Do not clip
values to `[0, 1]`: selective use can harm performance or outperform the
all-refined comparator. Identify any zero-denominator bootstrap samples
explicitly instead of silently dropping them.

**Reference-failure strata.** For zero, exactly one, and two or more
residual reference failures, report the number and fraction of eligible
sectors, final sector failures, and rescued/harmed outcomes relative to
initial-only, separately for each reference decoder. Report outcomes on
`sigma = 0` sectors as well, since yoke-gated policies skip refinement
there. These are measured strata, not an assumed multi-failure floor.

**Selection coverage.** Among sectors with exactly one residual reference
failure, report the probability that its failed patch is in the selected
set `M`, and the final success rate conditional on inclusion. This
separates selection failures from failures to exploit the refined score.
Uniform random k-subsets have coverage exactly `k / 6` on this stratum.
Candidate-restricted success cannot exceed coverage, and at `k = 1` it
equals coverage. Coverage is not an upper bound for mixed L2, which can
still flip an unselected patch using its initial probability.

**Secondary.** Sector and block failure counts; normalized LER per patch
per round via `sinter.shot_error_rate_to_piece_error_rate` with
`pieces = patches * rounds` (216 at d=9) and `values = 8`, so the numbers
sit beside the four-decoder table; tie counts and frequencies of calibrated
probabilities above one half. Expected random-control failure counts are
labeled as such; normalization is applied to their mean block failure
rate, not separately to each subset's normalized LER.

**Work.** Report requested patch-sector counts and fractions, distinct
patches refined per shot and their fraction of the six patches, and the
initial and incremental matching calls and reweighting passes from
section 8. Include the mean and 99th percentile of `dijkstra_states` for
the initial cluster-gap calculation. Plot primary and block failure
against distinct-patch and matching-call costs as well as nominal `k`.
Random-control work is the exact expectation under the specified subset
distribution. Keep offline collection work and timing in a separate
table; replay work counts do not imply measured speedups.

**Transitions.** For each pair of configurations, per sector, counts of
rescued, harmed, both-fail, and both-succeed shots. Rescues are split into
magnitude-only rescues, where every refined patch has a nonnegative signed
gap, and reversal rescues, where at least one refined patch's gap is
negative, for comparisons against initial-only. Also tabulate transitions
within the reference-failure strata. Transitions involving random controls
are exact weighted expectations; different random controls use independent
subset draws conditional on the same shot. Label their potentially
fractional counts explicitly.

**Validation numbers.** Agreement rate of the MWPM-reference pipeline with
joint PyMatching, additivity errors of the plain and correlated forced
weights, check parity agreement, and imported graph degrees, all reported.

**Connection to the UF experiments.** The evaluation report includes the
saved joint UF, joint correlated UF, joint MWPM, and built-in correlated
MWPM baselines alongside local UF plus initial-only L2, all-refined L2,
and selective L2. Report block failure and normalized LER on the same
evaluation shots. This distinguishes gains from treating the yokes in a
separate layer, gains from refined information, and the benefit retained
by selective use. Historical baselines appear only in the evaluation
column; newly collected joint MWPM and the frozen hierarchy configurations
also have confirmation measurements.

The report is a markdown file under `docs/results/`, in the style of the
existing four-decoder comparison: configuration table, results tables,
transitions, overall and queried-population reliability diagrams, coverage
and recovery curves, and accuracy-versus-work plots as PNG files beside
it, reproduction details, and links to the run directory. The
evaluation-set and confirmation-set numbers appear side by side, with
pilot and exploratory results clearly identified and the freeze manifest
linked. Missing or ineligible strata are labeled, not assigned a zero
failure rate.

## 10. Validation and acceptance tests

Unit tests live beside their modules. The end-to-end tests run on a
distance-3 circuit with a few hundred shots so that they finish in
seconds. Repeat graph and correction invariants on the pilot and on each
full distance-9 record during collection. Calibration and replay validate
their own invariants when those stages run. Each stage writes its check
results to its manifest; confirmation checks run only after the freeze.

1. **Hub split is lossless.** Merging the six check graphs' check vertices
   into the two yoke vertices reproduces `DecodingGraph.from_dem` of the
   six-patch DEM as a multiset of `(u, v, weight, mask)` up to `1e-9` in
   weight, with the local-to-global maps applied.
2. **Check parity.** On every sampled shot, the XOR over patches of the
   actual per-patch observable flips equals the yoke bit, for both sectors.
3. **Corrections are valid.** Every UF correction satisfies `H c = s` and
   `L c = r` on its local graph. Every final prediction has sector parity
   equal to the yoke bit.
4. **Plain forced weights are additive** across sectors to within `1e-9`,
   and `argmin` over each sector's forced weights equals the unforced plain
   prediction on every shot where the two weights differ by more than `1e-9`.
5. **Correlated gap sign is consistent.** Under the same frozen reweighted
   model, sector forced weights are additive to within `1e-9`, and their
   preferred class equals the unforced second-pass prediction whenever
   the sector weights differ by more than `1e-9`. Report ties separately;
   disagreement outside ties is a failure, not an allowed error fraction.
6. **The MWPM-reference pipeline reproduces joint MWPM.** With the plain
   gap mapped through the uncalibrated logistic `1 / (1 + exp(delta))` and
   the mixed L2, final predictions reproduce a joint MWPM optimum on the
   hub DEM. Compare with joint PyMatching recomputed by the collect stage.
   Predictions must agree when the optimum observable pattern is unique.
   In every disagreement, both predictions obey yoke parity and their
   total forced costs, `sum_i W_i(f[i, X], f[i, Z])`, agree to within
   `1e-6` nats. Report prediction agreement and tie-induced differences
   without treating a fixed agreement percentage as a correctness test.
7. **L2 is exact.** On random instances with up to 8 patches, enumeration
   agrees with the analytic threshold-and-parity rule in section 7,
   including equal-cost choices, probabilities equal to one half,
   candidate restrictions, and the lowest-binary tie rule. Explicit
   fixtures cover a two-error rescue with `sigma = 0` and a ranking-
   preserving probability transformation that changes the MAP pattern.
8. **Cluster gap is exact.** On small hand-built graphs, the Dijkstra
   value equals a brute-force minimum over all odd-parity closed walks
   through the boundary, both with and without partial growth.
9. **Calibration is monotone and held out.** The fitted map is
   non-increasing, including flat blocks and clipping, and the driver
   refuses to evaluate on its calibration sample or any subset of that
   parent sample. Queried-population reliability uses the exact policy
   selection mask or selection weights and reports its eligible counts.
   A small model with known probabilities checks the distinction between
   local calibration and residual marginals conditioned on yoke parity.
10. **Replay is deterministic.** Two replays with the same inputs and
    configuration produce identical outputs, including exact expectations
    for the random controls. Bootstrap reproducibility uses its recorded
    seed.
11. **Graph statistics describe imported edges.** On the recorded d=9
    DEM, independently counted adjacency degrees give yoke degrees 1,110,
    median detector degree 11, and maximum non-yoke degree 12. Raw DEM
    target multiplicities are kept separate. Other distances derive their
    statistics from their own graphs.
12. **Coverage has the stated meaning.** On single-failure fixtures,
    candidate-restricted success is bounded by coverage and equals it for
    one candidate. Changing the sole candidate's refined probability does
    not change that answer. No-candidate odd parity raises explicitly.
13. **Shared work is counted once.** Fixtures querying X and Z on the
    same patch incur one patch refinement, while queries on two patches
    incur two. Unrequested scores remain at `q0` even when computed.
    Call counts follow the estimator-specific table and exclude
    collection-only validation work.
14. **Random controls are exact and correctly gated.** Subset weights
    sum to one, single-failure coverage is `k / 6`, and gated controls
    request nothing for `sigma = 0`. Independent enumeration of X/Z
    subset pairs agrees with expected block failure and shared work.
    Verify that `top_k_given_yoke(k=6)` agrees with `all_refined` on
    single-failure sectors without imposing agreement on `sigma = 0`.
15. **Pilot, baselines, and confirmation preserve provenance.** Pilot
    records identify parent samples and rows; resuming collection neither
    drops nor duplicates them. Historical prediction imports reject hash
    or row mismatches. Confirmation sampling, collection, and replay
    require a freeze manifest and reject changed analysis artifacts or
    configurations.

## 11. Code layout, interfaces, and documentation standards

New package `src/yoked/hierarchical/`, one module per concept, tests
beside each module as in `src/yoked/decoders/`:

```text
src/yoked/hierarchical/
    __init__.py            public names only
    _patch_graphs.py       PatchGraphs, PatchGraph, hub split, local syndromes
    _cluster_gap.py        ClusterGapUnionFindDecoder, parity-augmented Dijkstra
    _matching_gaps.py      MatchingGaps, ForcedWeights, signed_gap, matcher construction
    _calibration.py        IsotonicCalibrator (PAV + interpolation)
    _outer_decoder.py      enumeration oracle, frame adjustment
    _outer_mwpm.py         production MWPM L2, canonical ties and precision checks
    _policies.py           deterministic selections and uniform subset distributions
    _replay.py             ReplayConfig, replay results, L1Record load/save, work counts
    _metrics.py            primary metric, eta, strata, coverage, reliability, bootstrap
src/yoked/decoders/
    _correlations.py       correlation_rules_from_dem, moved from the correlated UF module
tools/hierarchical_experiment   collect | calibrate | replay | report
docs/hierarchical_decoding.md   usage, mirroring docs/union_find_usage.md
docs/results/hierarchical_l1_l2_d9_p003.md   the report
```

Stages of the driver, with explicit input artifacts and provenance:
`collect` takes circuit parameters, a seed, and a shot count, or the path of
a saved sample, plus the dataset role and optional row selection, and writes
the L1 record and manifest of section 5.3. It supports pilot collection and
resuming the remaining rows, and imports verified historical baseline
predictions for evaluation when supplied;
`calibrate` takes a calibration record and writes one calibrator JSON per
estimator and sector, including the source sample and fitted row indices;
`replay` takes a record, the calibrators, and a list of configurations and
writes per-configuration results JSON plus final prediction arrays for
deterministic policies or per-shot expectations and weighted subset
outcomes for random controls;
`report` takes these outputs and writes the markdown report and figures,
supporting an exploratory report before confirmation exists. Frozen
analysis artifacts and configurations are recorded in the freeze manifest;
confirmation sampling, collection, replay, and reporting verify it.

Standards, applied to every module:

- The module docstring states the mathematical object the module computes,
  in the notation of section 2, and names the invariants its tests check.
- Records crossing module boundaries are frozen dataclasses with a docstring
  per field, including units and array shapes. No bare tuples.
- Functions are pure where possible; the only mutable state is inside a
  single decode call, as in the existing UF decoder.
- Every numeric constant has a name and a comment saying why it holds that
  value. Seeds are explicit parameters, never defaults hidden in code.
- The driver is a thin argparse layer over library functions and never
  contains logic that a test cannot reach without the command line.
- Each stage writes its outputs atomically, as the benchmark tool does, and
  records a manifest that later stages verify before reading.
- Names follow the notation of this document, so a reader can move between
  the spec, the docstrings, and the report without translation.

## 12. Milestones

**M1: graphs, validation, and a small d=9 pilot.** Implement the graph
split, L1, exact L2, shared correlation rules, collection, and the
calibration and endpoint replay needed for the pilot. Run applicable unit
and distance-3 checks, then the 2,000-shot calibration and evaluation
subsets from section 3. Report pilot initial-only versus all-refined for
both estimator pairs, including primary and block failure intervals,
reference-failure strata, and work counts. Check graph equivalence and
corrections on these records. Record whether the endpoint separation
justifies full collection; if no benefit is apparent, diagnose or stop
with a pilot report before committing to the full run. Pilot conclusions
are exploratory. Confirmation is not sampled or collected.

**M2: full calibration, evaluation, and endpoints.** If the pilot supports
proceeding, collect the remaining calibration and evaluation rows, refit
calibrators on the full calibration set, and replay `initial_only` and
`all_refined`. Report both estimator pairs with paired intervals, block
failures, strata, and the historical UF/MWPM baseline comparison. If
all-refined does not beat initial-only on the primary UF-reference metric,
do not start the primary selective experiment; report the absolute effects
and uncertainty. Do not use an unstable recovery fraction as evidence of
benefit. Confirmation remains untouched.

**M3: selective refinement and analysis freeze.** Implement all section 8
policies, exact random controls, both L2 rules, shared-work accounting,
coverage, transitions, and queried-population reliability. Develop and
assess policies on evaluation only; any calibration parameters continue
to be fit on calibration only. Complete the exploratory report, resolve
or document calibration limitations, then write the freeze manifest from
section 3. No policy or estimator is chosen using confirmation outcomes.

**M4: confirmation and final report.** After verifying the freeze manifest,
sample and collect the confirmation set once, apply the frozen analysis,
and report its results beside evaluation. Include paired effects, coverage,
strata, reliability, and accuracy-versus-work curves, and distinguish
historical evaluation baselines from confirmation measurements.

## 13. Decisions recorded

- Initial UF soft output is the cluster gap by Dijkstra, chosen over
  extra-cluster growth with a cap and over forced-complement UF.
- Success is scored per observable, so L2 maximizes over patterns and may
  correct multiple residual reference errors. No multi-failure floor is
  assumed, and calibration can change endpoint decisions.
- The correlated matching gap is the primary refined estimate; the plain
  gap is secondary.
- Nominal selection budgets are per sector; distinct-patch work and
  matching calls are counted with X/Z reuse. Random controls match each
  policy's trigger and are averaged exactly over uniform subsets.
- A small pilot precedes full L1 collection. Confirmation begins only
  after the estimator, policy, metric, and analysis artifacts are frozen.
- Historical joint UF and correlated UF results are included as evaluation
  baselines, alongside joint and built-in correlated MWPM.
- Correlated UF as a reference decoder, joint four-Pauli messages, latency
  measurement, and any change to the operating point are out of scope.
