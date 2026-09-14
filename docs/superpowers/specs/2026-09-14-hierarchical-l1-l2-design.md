# Hierarchical L1/L2 decoding experiment design

**Status:** approved design, 2026-09-14. Implementation plan to follow.

**Goal.** Split decoding of the 1D yoked surface code into a patch-local
layer (L1) that emits a reference correction and a confidence score per
patch and sector, and an outer layer (L2) that uses those confidences and
the yoke syndrome to decide which patches to flip. Then measure whether L2
can obtain most of the benefit of better confidence information by asking
for refined confidences from only a few patches.

**Question.** With L1 reference corrections held fixed, how much of the
improvement from replacing every patch's initial confidence with a refined
one is recovered when only k of six patches are refined, and does an
informed choice of those k patches beat a random choice?

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
- Each yoke detector has degree 14,556 in the decoding graph. The median
  detector has degree 70 and no other detector exceeds 82. The yoke is a
  hub, and UF growth from a fired hub reaches every patch at once.
- With the two yoke detectors removed, the DEM has 12 connected
  components, one per patch and sector, each with 1,480 detectors and
  exactly one observable. No component has more than two detectors, no
  component loses its last detector when the yoke is removed, and every
  component's yoke membership matches its observable's sector.
- Every observable-flipping component has exactly one physical detector.
  In the yoke-free view such components are boundary edges.
- Joint MWPM on the hub graph equals patch-local MWPM plus "flip the
  patch with the smallest complementary gap when the yoke fires". On the
  saved sample it is wrong on 0.03% of sectors with no patch-local failure
  and on 99.6% of sectors with two or more.
- Patch-local uncorrelated MWPM fails on 6.3% of patch-sectors. Per sector,
  27.2% of shots have exactly one failed patch and 5.1% have two or more.
  Joint MWPM misattributes 32.2% of the exactly-one cases.

Consequences: L1 operates on 12 independent graphs per shot; the yoke is
the only coupling; the multi-failure floor is large at this operating
point, so the primary metric must condition on exactly one residual
failure; and the MWPM-reference pipeline has a known exact answer, which
becomes the end-to-end validation test.

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
| Confirmation | 242 | 100,000 | Run once, after calibrators and policies are frozen. Reported beside the evaluation numbers. |

The saved seed-42 arrays are reused from the recorded run directory after
verifying their hash. Calibration has 50,000 x 6 patch outcomes per sector
with roughly 6% to 9% positives, so each calibrator sees more than 18,000
residual-error events. Distance 7 is a replication after distance 9 is
complete, using the saved d=7 seed-42 sample with the same three-set plan.

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

L1 runs once per patch on the check-free graph and reports, per sector, a
reference bit and one or more scores. Nothing in L1 sees a yoke or check
bit. Scores are defined so that a larger score means more confidence that
the reference bit is right, and calibration in section 6 maps them to
residual-error probabilities.

### 5.1 UF reference and the cluster gap

The reference decoder is the repository `UnionFindDecoder` on
`patch.graph`. Its correction is validated as in the existing tests:
`H c = s` over GF(2) and `L c = r`.

The initial score is the cluster gap of Meister, Pattison, and Preskill
(arXiv:2405.07433, Definition 9), evaluated on the terminated growth state
of the repository's weighted UF:

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

For a planar patch every odd-parity closed walk must use the boundary, so
this equals Meister's shortest path between inequivalent boundaries with
cluster interiors free. Charging partially grown edges only their remaining
growth is the one deviation from the unweighted original; it matches the
extra-cluster-growth view of Kishi et al. (arXiv:2602.03336) and reduces to
the original when no partial growth exists. The number of settled Dijkstra
states is recorded per patch-sector as the cost proxy.

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
| `cluster_gap` | (shots, 12) | nats |
| `dijkstra_states` | (shots, 12) | cost proxy |
| `forced_plain` | (shots, 6, 2, 2) | `W(c_X, c_Z)` |
| `forced_correlated` | (shots, 6, 2, 2) | `W(c_X, c_Z)` under the reweighted model |

Column `2i + s` holds patch `i`, sector `s`. The record is saved as one
`.npz` beside a JSON manifest holding the circuit parameters, seed, shot
count, sample hash, package versions, source hashes, and timing.

## 6. Calibration

An estimator is a pair (reference decoder, score). Each estimator gets its
own calibrator that maps its score to `P(e[i, s] = 1)`, fit on the
calibration set only. Fitting uses isotonic regression with the
pool-adjacent-violators algorithm implemented in NumPy, with the monotone
direction fixed by definition: decreasing in the cluster gap, decreasing
in a signed matching gap. Between adjacent PAV block centres the map is
linear, so it is strictly monotone inside the fitted range and flat
beyond it. Outputs are clipped to `[1e-6, 1 - 1e-6]` so that log-odds stay
finite.

Two properties matter for interpretation and are stated in the report.
First, within one estimator, L2's choice under a fired yoke is an argmax
over probabilities, which any strictly monotone map leaves unchanged; so
the initial-only and all-refined endpoints do not depend on calibration
except through ties. Second, the selective cells compare a refined
probability against initial probabilities of other patches, so they do
depend on the two calibrators agreeing on a common scale. Reliability
diagrams for every estimator are part of the report.

```python
calibrator = IsotonicCalibrator.fit(scores, outcomes, direction='decreasing')
q = calibrator.probability(scores)
calibrator.to_json() / IsotonicCalibrator.from_json(...)
```

## 7. L2: exact outer decoder for the factorized model

Per sector, L2 receives six probabilities and the frame-adjusted syndrome
`sigma[s]`. It assigns to every residual pattern `x in {0,1}^6` the weight
`prod_i q_i^{x_i} (1 - q_i)^{1 - x_i}`, discards patterns whose parity is
not `sigma[s]`, and returns the pattern of maximum weight. Enumeration over
64 patterns is exact for this model. Ties are broken toward the pattern
with the lowest binary value, bit `i` being patch `i`, and are counted. The final prediction is
`f = r XOR x`, and by construction its sector parity equals the yoke bit.

The maximum is over patterns, not over outer logical classes, because
success is scored per observable. The complement of a correct pattern
flips all six observables and is scored as six errors, so marginalizing
over classes would be inconsistent with the scoring. The two rules differ
only when both a pattern and its complement are plausible, which requires
several probabilities near one half.

A second rule, the candidate-restricted L2, is the same maximization with
unrefined patches fixed to `x_i = 0`. It never compares a refined
probability against an initial one and therefore isolates the selection
policy from cross-estimator calibration error.

```python
x, tied = exact_outer_map(q, parity, candidates=None)   # candidates: bool mask or None
```

## 8. Refinement policies and offline replay

Replay reads the stored record and calibrators, builds `q0` from the
initial estimator and `q1` from the refined estimator, and for each shot
and sector lets a policy choose the set of patches whose `q1` replaces
`q0`. L2 then runs on the mixed vector. A policy sees only `q0`, the
frame-adjusted syndrome, and its own random state; never `q1`, the actual
flips, or the outcome.

Configurations, all on the same shots and the same reference bits:

| Name | Refined set per sector | Purpose |
|---|---|---|
| `initial_only` | none | baseline hierarchy |
| `all_refined` | all six | available benefit |
| `top_k_given_yoke` | the k largest `q0` when `sigma[s] = 1`, none otherwise; k in {1, 2, 3, 6} | primary selective policy |
| `top_k_uncertain` | the k smallest absolute log-odds of `q0`, regardless of `sigma[s]` | the unconditioned policy from the proposal |
| `random_k` | k patches uniformly at random, policy seed 1234 recorded in the manifest | selection control |

Each selective configuration is run with both the mixed L2 and the
candidate-restricted L2. The primary estimator pair is UF reference,
cluster gap initial, correlated gap refined. The control pair is MWPM
reference, plain gap initial, correlated gap refined. The plain gap is also
run as a refined estimator for the UF reference as a secondary cell.

```python
config = ReplayConfig(reference='uf', initial='cluster_gap', refined='gap_correlated',
                      policy=TopKGivenYoke(k=2), outer='mixed')
result = replay(record, calibrators, config)
result.final          # (shots, 12) bool
result.refined        # (shots, 12) bool, which patch-sectors were refined
result.ties           # (shots, 2) bool
```

## 9. Metrics and report

**Primary.** Misattribution rate: among sectors with exactly one residual
reference failure, the fraction where the final prediction is wrong. It is
reported per configuration with a paired bootstrap interval, resampling
whole shots, 10,000 replicates, RNG seed 43.

**Recovery fraction.** For policy P at budget k,
`eta(k) = (m_initial - m_P(k)) / (m_initial - m_all)` on the primary
metric, with a paired bootstrap interval.

**Floor.** Fraction of sectors with two or more residual reference
failures, per reference decoder. Soft information cannot repair these.

**Secondary.** Sector and block failure counts; normalized LER per patch
per round via `sinter.shot_error_rate_to_piece_error_rate` with
`pieces = patches * rounds` (216 at d=9) and `values = 8`, so the numbers
sit beside the four-decoder table; fraction of patch-sectors refined, the cost proxy;
mean and 99th percentile of `dijkstra_states`; tie counts.

**Transitions.** For each pair of configurations, per sector, counts of
rescued, harmed, both-fail, and both-succeed shots. Rescues are split into
magnitude-only rescues, where every refined patch has a nonnegative signed
gap, and reversal rescues, where at least one refined patch's gap is
negative.

**Validation numbers.** Agreement rate of the MWPM-reference pipeline with
joint PyMatching, additivity error of the plain forced weights, and check
parity agreement, all reported.

The report is a markdown file under `docs/results/`, in the style of the
existing four-decoder comparison: configuration table, results tables,
transitions, reliability diagrams and the recovery curve as PNG files
beside it, reproduction details, and links to the run directory. The
evaluation-set and confirmation-set numbers appear side by side.

## 10. Validation and acceptance tests

Unit tests live beside their modules. The end-to-end tests run on a
distance-3 circuit with a few hundred shots so that they finish in
seconds; the same checks are repeated on the full distance-9 record by the
collect stage and their outcomes are written to the manifest.

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
5. **Correlated gap sign is consistent.** The class preferred by the
   correlated forced weights equals the prediction of a two-pass correlated
   matching with the same rules on at least 99.9% of patch-sectors.
6. **The MWPM-reference pipeline reproduces joint MWPM.** With the plain
   gap mapped through the uncalibrated logistic `1 / (1 + exp(delta))` and
   the mixed L2, final predictions equal joint PyMatching's predictions on
   the hub DEM, recomputed by the collect stage, on at least 99.9% of shots, and in every disagreement the two competing
   patches' gaps differ by less than `1e-6` nats.
7. **L2 is exact.** On random instances with up to 8 patches, enumeration
   agrees with an independent brute-force implementation, including ties.
8. **Cluster gap is exact.** On small hand-built graphs, the Dijkstra
   value equals a brute-force minimum over all odd-parity closed walks
   through the boundary, both with and without partial growth.
9. **Calibration is monotone and held out.** The fitted map is
   non-increasing, and the driver refuses to evaluate on the set it
   calibrated on.
10. **Replay is deterministic.** Two replays with the same inputs and
    seeds produce identical outputs.

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
    _outer_decoder.py      exact_outer_map, frame adjustment
    _policies.py           policy classes with one shared interface
    _replay.py             ReplayConfig, replay, L1Record load/save
    _metrics.py            primary metric, eta, floors, transitions, paired bootstrap
src/yoked/decoders/
    _correlations.py       correlation_rules_from_dem, moved from the correlated UF module
tools/hierarchical_experiment   collect | calibrate | replay | report
docs/hierarchical_decoding.md   usage, mirroring docs/union_find_usage.md
docs/results/hierarchical_l1_l2_d9_p003.md   the report
```

Stages of the driver, each reading only files the previous stage wrote:
`collect` takes circuit parameters, a seed, and a shot count, or the path of
a saved sample, and writes the L1 record and manifest of section 5.3;
`calibrate` takes a record and writes one calibrator JSON per estimator;
`replay` takes a record, the calibrators, and a list of configurations and
writes per-configuration results JSON plus the final prediction arrays;
`report` takes replay outputs for the evaluation and confirmation sets and
writes the markdown report and its figures.

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

**M1: graphs, L1, and the exact validation.** Sections 4, 5, and 7, the
correlation-rule move, and the collect stage. Done when tests 1 to 8 pass
on the distance-3 fixture and the collect stage has produced the d=9
records for all three sample sets with tests 1 to 6 passing on them.

**M2: calibration and the two endpoints.** Section 6, `replay` with
`initial_only` and `all_refined`, the primary metric, floors, and bootstrap.
Done when the report shows initial-only versus all-refined for both
estimator pairs on the evaluation set with intervals. This is the decision
point: if all-refined does not beat initial-only on the primary metric, M3
is not started and the report says why.

**M3: selective refinement.** Section 8 policies, both L2 rules,
transitions, the recovery curve, the confirmation run, and the final
report.

## 13. Decisions recorded

- Initial UF soft output is the cluster gap by Dijkstra, chosen over
  extra-cluster growth with a cap and over forced-complement UF.
- Success is scored per observable, so L2 maximizes over patterns.
- The correlated matching gap is the primary refined estimate; the plain
  gap is secondary.
- Correlated UF as a reference decoder, joint four-Pauli messages, latency
  measurement, and any change to the operating point are out of scope.
