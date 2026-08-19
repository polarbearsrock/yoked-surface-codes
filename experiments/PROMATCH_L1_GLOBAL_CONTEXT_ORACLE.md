# ProMatch L1 Global-Context Oracle Experiment

- **Status:** draft exploratory experiment specification
- **Date:** 2026-08-18
- **Claim-bearing:** no
- **Implementation status:** not yet implemented
- **Collection status:** no new shots have been sampled for this experiment

This document specifies the next diagnostic experiment following the V3
`PU-window` pilot. It is intentionally separate from the frozen first-round
protocol and from the immutable V3 output corpus.

The experiment has two purposes:

1. determine whether locally committed corrections lose accuracy because the
   local predecoder cannot see relevant parts of the complete yoked matching
   problem; and
2. measure whether a nontrivial amount of detector-event removal remains when
   every local commitment is required to preserve the deterministic U0
   prediction under a conservative two-stage certificate.

The oracle is a conservative diagnostic certificate, not a proposed production
decoder. It may call full-joint MWPM for every proposed local commitment.
Consequently, it cannot support a latency claim.

## Contents

1. [Decision summary](#1-decision-summary)
2. [Why this experiment is next](#2-why-this-experiment-is-next)
3. [Questions and falsifiable hypotheses](#3-questions-and-falsifiable-hypotheses)
4. [Terminology and non-goals](#4-terminology-and-non-goals)
5. [Critical workload distinction](#5-critical-workload-distinction)
6. [Oracle contract](#6-oracle-contract)
7. [Sequential candidate semantics](#7-sequential-candidate-semantics)
8. [Descriptive context classification](#8-descriptive-context-classification)
9. [Decoder arms](#9-decoder-arms)
10. [Phase A: retained V3 replay](#10-phase-a-retained-v3-replay)
11. [Phase B: fresh paired mechanism screen](#11-phase-b-fresh-paired-mechanism-screen)
12. [Endpoints and statistics](#12-endpoints-and-statistics)
13. [Architecture checkpoint before deployable guards](#13-phase-cd-architecture-checkpoint-before-deployable-guards)
14. [Latency after the checkpoint and accuracy](#14-latency-follows-the-architecture-checkpoint-and-guard-accuracy)
15. [Required implementation work](#15-required-implementation-work)
16. [Required tests](#16-required-tests)
17. [Artifact and provenance contract](#17-artifact-and-provenance-contract)
18. [Decision table](#18-decision-table)
19. [Relationship to hierarchical decoding](#19-relationship-to-later-hierarchical-decoding)
20. [References](#20-references)

---

## 1. Decision summary

The next experiment will proceed in five ordered phases:

1. **Shadow audit on retained V3 cases.** Evaluate every originally committed
   path against complete-graph cost and logical-frame information without
   changing the original trajectory.
2. **Sequential oracle replay.** Rerun the predecoder while a full-context
   oracle accepts or vetoes each proposal. A veto changes subsequent candidate
   selection, so this phase cannot be implemented by simply deleting paths from
   an existing trace.
3. **Fresh shot-paired screens.** At `p=0.002`, compare the joint, oracle,
   held-trace per-patch, and native per-patch arms. At `p=0.001`, measure
   certified headroom using partial/HW0 sensitivity arms instead of applying
   the stress-cell workload gate.
4. **Architecture checkpoint.** Compare joint versus per-patch hard outputs and
   determine whether a validated per-patch complementary-gap/soft-output path
   has enough signal to prioritize a true L1-to-L2 decoder.
5. **Conditional guard and latency work.** Only if the checkpoint says the flat
   hybrid remains worthwhile, fit boundary/temporal and explicitly nonlocal
   flat-yoke veto guards, freeze one, and test its accuracy and latency.

The primary joint/oracle comparison does **not** replace windowing, change `HW=10`, add
correlated matching, or implement a true L1-to-L2 hierarchy. Named per-patch,
partial, HW0, and odd-boundary sensitivity arms are kept separate so their
architectural questions do not contaminate the primary comparison.

---

## 2. Why this experiment is next

The V3 pilot compared:

```text
U0-direct:
    complete syndrome -> ordinary uncorrelated full-joint PyMatching

PU-window:
    patch/basis/window-local ProMatch-style predecode
    -> complete-width residual syndrome
    -> the same ordinary uncorrelated full-joint PyMatching
```

PU-window reduced the number of active detector events passed to the residual
matcher, but it had a higher logical failure rate in every sampled cell:

| Cell | U0 failure | PU failure | Regressions | Recoveries | Residual/original event ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d=7, p=0.001` | 0.0305% | 0.0700% | 83 | 4 | 0.964657 |
| `d=7, p=0.002` | 3.6280% | 11.0000% | 16,239 | 1,495 | 0.786366 |
| `d=7, p=0.003` | 33.7870% | 66.9490% | 74,234 | 7,910 | 0.602523 |
| `d=5, p=0.003` | 41.3560% | 43.3240% | 5,938 | 2,002 | 0.977175 |
| `d=5, p=0.005` | 92.9015% | 94.3465% | 6,515 | 3,625 | 0.898375 |

The retained `d=7, p=0.001` corpus is especially useful because all 83
regressions and all 4 recoveries fit below the per-category retention cap. The
existing replay diagnosis reproduced them bit-for-bit and assigned the dominant
omitted context among regressions as:

| Dominant retained classification | Regressions |
| --- | ---: |
| Yoke | 34 |
| True boundary | 33 |
| Cross-window | 7 |
| Terminal | 1 |
| In-domain/tie behavior | 8 |

Every retained regression changed exactly two observables in the same basis.
The current evidence therefore supports a concrete hypothesis: a path that is
locally attractive in one `(patch, basis, window)` domain can be incompatible
with the complete yoked correction problem.

The classifications above come from one deterministic optimum returned by
PyMatching. They are strong debugging evidence, but they are not yet unique or
causal attributions because another equal-weight optimum may pair events
differently. The oracle experiment addresses that limitation by checking
objective cost and logical class, followed by controlled context ablations.

Primary evidence:

- [`pilot/summary.json`](../out/promatch_l1_round1_v3_20260817_32p/pilot/summary.json)
- [`pilot_analysis/analysis.md`](../out/promatch_l1_round1_v3_20260817_32p/pilot_analysis/analysis.md)
- [`tools/diagnose_promatch_l1`](../tools/diagnose_promatch_l1)

---

## 3. Questions and falsifiable hypotheses

### 3.1 Primary questions

1. Which original output-changing local commitments conflict with the
   full-context two-stage certificate?
2. After certificate-rejected commitments are vetoed, how many local
   commitments and how much detector-event removal remain?
3. Which omitted context categories are necessary for a practical veto rule?
4. Can a cheap context guard approximate the oracle without consulting a full
   MWPM solution?
5. How much of the joint-backend harm persists when the same shot and held V3
   trace are decoded by a native independent-patch backend?
6. At `p=0.001`, how much certified removal is available only after partial
   commit, HW0, or boundary-aware sensitivity policies?

### 3.2 Induction theorem and implementation invariant

Assume deterministic scalar/batch U0 consistency and correct GF(2), frame, and
transaction ledgers. If every **durable** original PU commitment preserves:

```text
A_k XOR l(s_k)
```

where `l(s_k)` is the logical-prediction component of `D_G(s_k)`. Then the
final PU prediction equals the initial U0 prediction by induction.
Therefore every shot with `PU-window != U0-direct`—regression or recovery—must
contain at least one durable frame-conflicting commitment.

This is a theorem and an implementation invariant, not an empirical
hypothesis. The informative measurements are the first conflicting commitment,
whether it is also numerically cost-compatible, and its context labels. Rolled-back
provisional proposals are excluded from this theorem because they do not affect
the final PU result.

### 3.3 Empirical hypotheses

**H1: certified removable work.** Some original PU commitments satisfy the
cost-and-frame certificate, so the oracle can remove a nontrivial fraction of
active detector events while remaining exactly U0-equivalent.

**H2: architecture-consistent context.** Among conflicts that survive the
held-trace and native per-patch interventions, useful local veto information is
concentrated in true-boundary and temporal/terminal alternatives. Yoke-labeled
conflicts are analyzed separately as a nonlocal flat-backend interaction, not
as context available to an independent-patch L1 decoder.

**H3: performance remains an open question.** Detector-event removal may or may
not reduce PyMatching runtime. Even if residual matching becomes faster, the
predecoder can erase that benefit end to end.

**H4: backend interaction.** Holding the V3 predecode trace fixed while
replacing the joint backend with a native independent-patch backend materially
reduces the PU-versus-U0 harm, consistent with residual yoke rerouting and/or
the associated native-graph marginal reweighting, while leaving a nonzero
boundary/temporal/local remainder. Unique attribution to topology requires the
optional weight-preserving projected-`yokes=2` reference/treatment pair.

**H5: hierarchical information.** Native per-patch complementary gaps contain
discriminative information about hard-output correctness that a later outer
decoder could calibrate and use; this is tested only after a new
constrained-class gap implementation passes validation.

### 3.4 Implementation stop conditions versus scientific evidence

The following are implementation failures, not evidence for or against the
missing-context hypothesis:

- a sequential cost-and-frame oracle fails to reproduce U0 exactly;
- forced-composite accounting reports a cost below the full-joint optimum
  beyond frozen tolerance;
- residual, frame, rollback, determinism, scalar-versus-batch, or ledger
  invariants fail; or
- an original U0/PU-discordant shot follows an unchanged durable trajectory
  without a frame-conflicting commitment.

Stop and fix these conditions before interpreting any oracle output.

Scientific evidence against a simple omitted-boundary/yoke/time explanation
would instead include:

- first conflicts concentrated in in-domain ordering, path, or logical-class
  behavior instead of omitted context;
- controlled boundary/yoke/temporal guards failing to reduce paired
  regressions despite matching their oracle labels; or
- residual guard failures explained by correction algebra or candidate
  generation rather than the information visible to the guard.

`O-frame` cannot independently prove causality: it vetoes every proposal that
would change deterministic U0 under its certificate. Causal evidence comes
from the later controlled guard ablations on fresh shots.

---

## 4. Terminology and non-goals

### 4.1 Terminology

**U0-direct** is ordinary uncorrelated PyMatching on the complete original
syndrome and complete joint DEM graph.

**PU-window** is the frozen V3 local predecoder followed by ordinary
uncorrelated PyMatching on a complete-width residual syndrome.

**Candidate path** is one correction path proposed by an existing ProMatch
stage before its endpoints and edge support are committed.

**Global-context oracle** is a diagnostic policy that consults the complete U0
matching problem before accepting a candidate. It does not inspect the sampled
ground-truth observables.

**Deployable context guard** is a veto-only rule based on precomputed graph
distances and the current active syndrome. It does not run a complete MWPM to
approve each candidate.

**Per-patch `yokes=0` comparator** projects the inner and terminal syndrome from
the same sampled `yokes=2` shot into a separately generated native `yokes=0`
matching problem. It measures an L1 hard output and is not a complete yoked
decoder without an outer decision.

**Detector-event relief** is a reduction in active residual syndrome bits. It
is not automatically graph-size, operation-count, or latency relief.

### 4.2 Non-goals

No arm in this experiment:

- use actual sampled observables when accepting or rejecting a path;
- discover the physical error perfectly;
- claim that the oracle improves latency;
- reproduce the published ProMatch hardware architecture;
- add correlated PyMatching;
- implement an explicit outer yoke/QDPC decoder; or
- modify, resume, or promote the immutable V3 corpus.

The V3-compatible joint arms additionally keep `HW=10`, disabled boundary,
zero-frame local paths, stage ordering, window length/history, and the static
full-joint matcher fixed. Two explicitly named diagnostic exceptions do not
alter that controlled joint comparison:

- per-patch arms replace the final backend to study architecture transfer but
  do not produce a complete yoked-code decision; and
- the full HW/boundary/observable-policy sensitivity factorial deliberately
  varies handoff, boundary, and frame policies to measure aggressive certified
  headroom.

These sensitivity arms are never pooled with or relabeled as the primary
HW10 result. This experiment also does not tune a dynamic/deadline-dependent HW
schedule.

The oracle is intentionally relative to the deterministic U0 implementation,
version, graph, and tie behavior. Exact U0 equivalence is not proof that U0 is
physically correct or uniquely optimal.

---

## 5. Critical workload distinction

The current stack compiles one matcher from the complete detector error model:

```text
Without predecode:
    complete static graph + K active detector events -> one joint MWPM

With predecode:
    complete static graph + K' active detector events -> one joint MWPM
    where K' may be smaller than K
```

The matcher vertices, edges, memory footprint, packed syndrome width, and number
of residual matcher calls are unchanged. The current workload estimand is only:

```text
R_event = sum(residual active detector events)
          / sum(original active detector events)
```

This is implemented as a ratio of aggregate detector Hamming weights, not as a
graph-size or instruction-count measurement. Fewer active defects may reduce
search inside PyMatching, but that must be timed separately.

The oracle itself performs additional full matching calls and is therefore
expected to be slower. Its `R_event` is a conservative estimate of event
removal certified along this particular greedy trajectory. Because sequential
`O-frame` can reject a globally safe sequence whose intermediate frame changes,
the accepted removal is not an upper bound on all possible safe event removal.
It is also not an end-to-end workload or latency measurement.

Relevant implementation points:

- complete matcher construction:
  [`_promatch_graph.py`](../src/yoked/decoding/_promatch_graph.py#L182)
- complete-width residual matcher call:
  [`_promatch_decoder.py`](../src/yoked/decoding/_promatch_decoder.py#L200)
- paired U0 and residual calls:
  [`U0`](../src/yoked/decoding/_promatch_experiment.py#L1705) and
  [`residual`](../src/yoked/decoding/_promatch_experiment.py#L1720)
- event-ratio analysis:
  [`_promatch_analysis.py`](../src/yoked/decoding/_promatch_analysis.py#L294)

---

## 6. Oracle contract

### 6.1 State and notation

Let:

- `G` be the complete ordinary uncorrelated matching graph compiled from the
  exact frozen decomposed joint DEM artifact;
- `s_k` be the complete provisional syndrome immediately before proposal `k`;
- `A_k` be the accumulated local observable frame before proposal `k`;
- `P_k` be the proposed local correction's canonical square-free GF(2) edge
  support, with no repeated edge;
- `b(P_k)` be the detector boundary of that path as a complete-length bit
  vector;
- `o(P_k)` be the path's observable frame;
- `w(P_k)` be `math.fsum` of the candidate's canonical edge weights, asserted
  equal to its frozen matching-path decision weight; and
- `D_G(s) = (l(s), M(s), W(s), W_backend(s))` be deterministic ordinary
  PyMatching on `G`, returning its logical prediction, canonical returned edge
  support, `math.fsum` of that support's canonical edge weights, and the
  backend-reported weight.

`decode_to_edges_array` exposes endpoint pairs, not canonical edge IDs. Normalize
each detector pair and detector-to-boundary pair (`-1` is the boundary marker)
and require it to map to exactly one edge in the frozen `matcher.edges()` table.
If a returned pair is missing or ambiguous because parallel canonical edges
survive compilation, this oracle representation is unsupported and fails
closed. XORing the mapped edges' fault masks must reproduce `l(s)` exactly;
duplicate returned canonical edge IDs are also forbidden.

The V3 primary policy permits only zero-observable-frame local paths, so
`o(P_k)=0` in the first implementation. The general term remains in the
contract so that the invariant is explicit.

For the current state, calculate:

```text
(l_k, M_k, W_k, W_backend_k) = D_G(s_k)
F_base                         = A_k XOR l_k
```

For the candidate composite, calculate:

```text
s'_k                                      = s_k XOR b(P_k)
A'_k                                      = A_k XOR o(P_k)
(l'_k, M'_k, W'_k, W_backend'_k)          = D_G(s'_k)
F_candidate                               = A'_k XOR l'_k
C_candidate                               = math.fsum(
    canonical_weights(P_k) concatenated with canonical_weights(M'_k)
)
cost_excess_k                             = C_candidate - W_k
```

`C_candidate` is a **forced two-stage composite score**: commit the local path,
then optimally complete its residual syndrome. The concatenated sum deliberately
counts an edge twice if it occurs in both supports; in exact arithmetic it is
`w(P_k) + W'_k`. It must not be described as PyMatching literally forcing one
edge into one selected blossom solution.

For an exact T-join problem with the same strictly positive canonical edge
weights on both sides:

```text
cost_excess_k = 0
    iff
P_k is contained in the edge support of at least one minimum-weight
correction of s_k.
```

Proof sketch:

- If an optimum `M*` contains `P_k`, then `M* \ P_k` corrects `s'_k`, so
  `w(P_k) + W'_k <= W_k`.
- Conversely, `P_k XOR M'` is a valid correction of `s_k` for any optimum `M'`
  of `s'_k`, so minimality gives
  `W_k <= w(P_k XOR M') <= w(P_k) + W'_k`.
- Equality throughout and strictly positive weights force `P_k` and `M'` to be
  edge-disjoint; their union is an optimum containing `P_k`.

Thus `O-cost` is an exact objective/support-compatibility certificate in exact
arithmetic under these preconditions. The compiled experiment must fail closed
if a canonical matching edge is nonfinite or nonpositive. Zero-weight graphs
are unsupported by the membership theorem because zero-weight cancellation can
break the support-containment conclusion.

Current ProMatch candidates are deterministic simple paths, but the
square-free-support property remains an asserted invariant. Exhaustive
small-graph tests in Section 16 verify the equivalence before any yoked run.

### 6.2 Numerical compatibility

Define a protocol-frozen tolerance:

```text
tau_k = absolute_tolerance
        + relative_tolerance * max(1, abs(W_k), abs(C_candidate))

tau_weight(s) = absolute_tolerance
                + relative_tolerance
                  * max(1, abs(W(s)), abs(W_backend(s)))
```

Classify the numerical cost result as:

```text
numerically cost-compatible: abs(cost_excess_k) <= tau_k
positive cost excess:        cost_excess_k > tau_k
numeric/accounting anomaly:  cost_excess_k < -tau_k
```

A negative cost excess beyond tolerance is a fatal accounting,
weight-semantics, or numeric anomaly. It must not be silently accepted.

Use the same canonical weight source and one `math.fsum` operation on each side
of `cost_excess`:

- `W_k` and `W'_k` are `math.fsum` over the edge supports returned by
  `decode_to_edges_array`, looked up in the `matcher.edges()`-derived canonical
  edge table; and
- `C_candidate` is one `math.fsum` over the concatenation of the candidate and
  residual-support weight sequences, rather than the binary floating-point
  addition of two pre-summed values.

Record both PyMatching's `decode(..., return_weight=True)` value and the
support-`fsum` value, but use the support-`fsum` values in the operational
`cost_excess`. A non-claim-bearing 1,000-shot development check at the primary
geometry observed a maximum relative difference of `1.125e-8` and a maximum
absolute difference of `1.469e-5`. These are characterization observations,
not universal bounds; they characterize one source of solver/accounting
discrepancy but do not by themselves determine the operational tolerance. The
initial tolerance proposal is:

```text
relative_tolerance = 1e-6
absolute_tolerance = 1e-9
```

Before freezing, run a deterministic characterization corpus for every frozen
graph and require:

1. scalar and batch predictions agree bit-for-bit;
2. `decode` and `decode_to_edges_array` select solutions whose logical frames
   agree bit-for-bit;
3. backend-reported and support-`fsum` weights differ by no more than
   `tau_weight(s)`;
4. repeated support reconstruction and candidate/residual concatenation are
   deterministic;
5. replacing `math.fsum` by a `decimal.Decimal.from_float` reference sum causes
   zero cost-classification changes; and
6. no cost classifications change across the preregistered sensitivity set:

   ```text
   (relative_tolerance, absolute_tolerance) in
       {(1e-7, 1e-10), (1e-6, 1e-9), (1e-5, 1e-8)}
   ```

If these checks fail, stop instead of widening tolerance after observing
logical outcomes. In particular, any classification change makes the numeric
band ambiguous for that graph and blocks protocol freeze; it is not resolved by
choosing the most favorable tolerance. A numerically compatible proposal is
not described as an exact support-membership proof; exact membership language
is reserved for the ideal theorem and exhaustive arithmetic tests.

### 6.3 Two oracle variants

The experiment evaluates two named variants:

**Oracle-Cost (`O-cost`)** accepts a candidate iff it is numerically
cost-compatible. It operationalizes the exact objective/support theorem within
the frozen numeric tolerance.

**Oracle-Cost+Frame (`O-frame`)** accepts a candidate iff it is numerically
cost-compatible and:

```text
F_candidate == F_base
```

`O-frame` is the primary conservative certificate. Sequential acceptance
preserves the current deterministic U0 prediction by induction. Exact U0
agreement is therefore an implementation invariant, not evidence that the
oracle empirically improved accuracy or found every safe commitment.

Among numerically cost-compatible proposals, a difference between `O-cost` and
`O-frame` measures deterministic tie-dependent logical-class or path-frame
behavior. `O-cost` is the structural/objective compatibility label;
`O-frame` is the stricter deterministic-U0-equivalence label.

### 6.4 No ground-truth access

The oracle gate accepts only:

```text
graph + current syndrome + accumulated frame + candidate metadata
```

Its API must not accept actual observables, sampler error labels, or a Boolean
indicating whether U0 or PU was correct. Actual observables are joined only
after decoding for post hoc regression/recovery analysis.

### 6.5 Conservative interpretation

`O-frame` answers:

> How many local events could be removed while preserving this deterministic
> U0 decoder's prediction under this conservative two-stage certificate?

It does not answer:

> Which decoder is physically correct?

Because the certificate requires intermediate prediction equivalence, it may
reject a sequence whose intermediate logical class changes but whose later
commits would restore the original class. Its certified-removal estimate is
therefore conservative.

---

## 7. Sequential candidate semantics

Oracle decisions must occur at the point where a stage proposes a path, before
the path mutates active detector state or the accumulated frame.

For every domain:

1. Build ordered candidate lists using the unchanged V3 stage priority and
   within-stage ordering rules.
2. Evaluate the first eligible proposal against the current complete
   provisional syndrome and frame.
3. If accepted, commit it transactionally, update the complete provisional
   state, clear state-specific vetoes, and recompute the next proposal.
4. If vetoed, leave the syndrome and frame unchanged, blacklist that exact
   proposal for the current active-state fingerprint, and request the next
   deterministic eligible proposal.
5. If a stage has no unvetoed proposal, advance to the next stage exactly as
   V3 does when that stage has no candidate. If all four stages are exhausted
   while `HW` remains above the target, return the existing `NO_CANDIDATE`
   exhaustion status. Existing boundary-unavailable and disconnected reasons
   retain their V3 precedence. The named transaction policy below determines
   whether the accepted prefix is rolled back or retained.
6. Re-evaluate the oracle after every accepted commit; never reuse a decision
   made for an earlier syndrome state.

On an unchanged active state, eligibility is recomputed and the complete
candidate set for the current stage is sorted by the exact existing
`_Candidate.key`; blacklisted signatures are skipped in that order. Only after
that ordered stage list is exhausted may selection advance to the next stage.
An acceptance changes the active state and restarts from stage 1. A veto never
changes the active state. This is the only path-level oracle policy in this
experiment; immediate rollback after the first veto would be a differently
named ablation.

A proposal signature must include at least:

```text
domain + stage + ordered endpoints + canonical path edge IDs
```

The blacklist is scoped to an exact active-state fingerprint. It is discarded
when an accepted commitment changes that state. This prevents both infinite
reproposal and accidental suppression of a candidate in a different state.

After an acceptance, the next iteration restarts candidate selection at stage
1, matching the current V3 loop. These transition and exhaustion rules are part
of the versioned oracle contract and require direct unit tests.

### 7.1 Named transaction policies

The experiment includes both policies from the start:

**Transactional (`tx`).** If a domain cannot reach its residual-HW target, roll
back the whole domain exactly as V3 does. All provisional paths and frames are
discarded.

**Partial (`partial`).** If a domain exhausts its eligible proposals before
reaching the target, durably commit the longest sequential prefix already
accepted by `O-frame`, stop that domain, and leave all unresolved events for the
final backend. Record `status=partial-exhausted` and a separate frozen
exhaustion reason.

Every prefix step in `partial` preserves the complete deterministic U0
prediction, so the final partial result remains U0-equivalent by induction even
though the nominal HW target was missed. This makes partial semantics
scientifically important: it separates safe certified removal from V3's
all-or-nothing capacity/rollback policy.

The readable display aliases are explicit, for example:

```text
PU-O-frame-tx-HW10
PU-O-frame-partial-HW10
```

Outputs from these policies cannot be pooled.

### 7.2 Per-state veto budget

Stage 3 can expose `O(HW^2)` candidates. The primary semantic oracle remains
uncapped if the preflight shows it is feasible. A separately named budgeted
variant may freeze a positive integer `B`, the maximum number of rejected
proposals evaluated at one unchanged active-state fingerprint. The counter
starts at zero; after recording the `B`th veto, stop immediately without
evaluating candidate `B+1`. An accepted proposal changes the fingerprint and
starts a new counter.

```text
pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-tx-budget-<B>-joint-y2
pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-budget-<B>-joint-y2
```

`<B>` is a documentation template only. Every frozen `arm_id` contains the
literal decimal integer; unresolved template tokens are invalid.

On budget exhaustion, never accept an uncertified candidate:

- `tx` rolls back the domain; and
- `partial` commits the already certified prefix and stops the domain.

Record candidates enumerated, vetoes, oracle calls, stage, enumeration time,
budget hits, and accepted-before-stop counts per state. Budgeted and uncapped
results are different decoder arms and cannot be pooled.

Two replay products must remain distinct:

- **shadow audit:** score the original PU proposals without changing them;
- **sequential oracle:** actually veto proposals and allow the future candidate
  sequence to change.

The shadow audit locates the first conflict in an existing U0/PU-discordant
shot. It is not a decoder result.

---

## 8. Descriptive context classification

Every proposal is classified into unconditional multi-label aggregates
describing which complete-graph context is absent from its local domain:

- `true_boundary`
- `yoke`
- `cross_window`
- `terminal`
- `other_patch_or_basis`
- `in_domain_alternative`
- `same_pair_different_path_or_frame`
- `equal_weight_logical_class`
- `unclassified`

One proposal may have several labels. The complete multi-label set must be
retained. A deterministic dominant label may be frozen for compact tables, but
it cannot replace the multi-label record or be interpreted as causal by itself.

The two PyMatching diagnostic APIs have distinct meanings:

- `decode_to_matched_dets_array` reports the selected matched active-defect
  partner, or the boundary, but not the correction path; and
- `decode_to_edges_array` reports selected correction-support edges, whose
  endpoints may be internal inactive vertices rather than matched defects.

The classifier stores these as separate `matched_partner_labels` and
`support_path_labels`; it never calls an internal support neighbor a matched
destination. Membership in the selected optimum is descriptive only and must
not be the oracle acceptance rule because degenerate equal-weight optima may
contain different pairs or paths.

For each proposal endpoint, the descriptive classifier applies these frozen
predicates to the selected partner and to every vertex/edge in its selected
support component:

- a virtual-boundary endpoint or boundary edge adds `true_boundary`;
- any yoke detector or yoke-incident edge adds `yoke`;
- an L1 body detector in another window adds `cross_window`;
- an L1 terminal detector adds `terminal`;
- a detector owned by another patch or basis adds `other_patch_or_basis`;
- a selected matched partner different from the proposed partner but wholly
  inside the domain adds `in_domain_alternative`;
- the same matched endpoint pair with different canonical support or
  observable frame adds `same_pair_different_path_or_frame`; and
- a numerically cost-compatible but frame-incompatible proposal adds
  `equal_weight_logical_class`.

If a cost/frame conflict receives none of these labels, add `unclassified`.
The implementation must freeze the partner mapping, support-component
construction, detector-role mapping, tie behavior, and synthetic examples for
every label before fresh sampling. Detailed per-proposal rows are retained only
under the bounded deterministic policy in Section 17; every proposal still
contributes to the unconditional aggregates.

For every veto, separately record:

- positive cost excess;
- equal-cost logical-class change;
- both cost and class conflict;
- same endpoint pair but different path/frame;
- numeric anomaly; and
- the complete omitted-context label set.

Mechanism evidence comes later from the shot-paired backend interventions and
controlled guard ablations.

---

## 9. Decoder arms

All fresh arms within a cell receive the same sampled `yokes=2` detector and
actual-observable arrays. Per-patch arms use the frozen projection in Section
11.4; they do not resample a `yokes=0` circuit.

Tables use a readable display label and a filesystem-safe canonical `arm_id`.
Only `arm_id` enters protocols, artifact keys, directories, or CLIs; aliases
containing `/` are display text only. Policy tokens map literally to the current
configuration vocabulary:

```text
boundary-disabled   -> boundary_policy="disabled"
boundary-odd-parity -> boundary_policy="odd-parity"
observable-zero-frame -> observable_policy="zero-frame"
observable-any        -> observable_policy="any"
```

### 9.1 Joint and oracle arms

| Display label | Canonical `arm_id` | Candidate behavior | Final backend | Interpretation |
| --- | --- | --- | --- | --- |
| `U0-joint` | `u0-joint-y2` | No predecode | Complete `yokes=2` ordinary PyMatching | Existing U0 reference |
| `PU-window/joint` | `pu-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2` | Existing V3 commitments | Complete `yokes=2` ordinary PyMatching | Positive control for known harm |
| `PU-O-cost-tx-HW10` | `pu-ocost-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2` | Numerically cost-compatible proposals; transactional | Complete `yokes=2` ordinary PyMatching | Objective/support compatibility sensitivity |
| `PU-O-frame-tx-HW10` | `pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2` | Cost- and U0-frame-compatible proposals; transactional | Complete `yokes=2` ordinary PyMatching | V3 rollback comparator |
| `PU-O-frame-partial-HW10` | `pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2` | Same certificate; retain certified prefix on exhaustion | Complete `yokes=2` ordinary PyMatching | Durable certified-removal estimate |

`U0-joint` and `PU-window/joint` are the renamed experiment-table labels for
the existing `U0-direct` and `PU-window`; their implementations do not change.

### 9.2 Shot-paired per-patch arms

| Display label | Canonical `arm_id` | Predecode | Final backend | Interpretation |
| --- | --- | --- | --- | --- |
| `U0-perpatch-y0` | `u0-perpatch-y0` | None | Native matcher compiled from a directly generated `yokes=0` DEM | Independent-patch reference |
| `PU-heldV3/perpatch-y0` | `pu-held-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-perpatch-y0` | Hold V3 `yokes=2` candidate generation and commits fixed, then project the residual | Native `yokes=0` matcher | Final-backend intervention with held predecode |
| `PU-native/perpatch-y0` | `pu-native-window-hw10-boundary-disabled-observable-zero-frame-tx-perpatch-y0` | Project original syndrome, then compile and run the predecoder from the native `yokes=0` graph | Native `yokes=0` matcher | Architecture-transfer L1 arm |

These arms answer different questions. `PU-heldV3/perpatch-y0` changes only the
residual backend after the V3 predecode trace, whereas
`PU-native/perpatch-y0` changes both local weights/candidates and the backend to
the native independent-patch model. They cannot be pooled or called equivalent.

The native arm uses the same `d`-round nonoverlapping windows, stages 1--4,
`HW=10`, deterministic stage/candidate ordering, `boundary_policy="disabled"`,
`observable_policy="zero-frame"`, and whole-domain `tx` semantics as V3. Only its
domain graphs, candidate weights/paths, and residual backend come from the
directly generated native `yokes=0` matcher. The held arm uses the exact V3
`yokes=2` configuration and trace before projection.

An optional, separately named pair:

```text
u0-perpatch-projected-y2
pu-held-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-perpatch-projected-y2
```

may transform yoke-hub incidences into independent boundaries while preserving
frozen `yokes=2` weights. Both reference and treatment are required for the
paired treatment-effect contrast. This is a pure topology/constraint
intervention, not the native per-patch architecture, and it requires its own
transformation proof and graph hash. The pair becomes mandatory before making
a unique yoke-topology attribution.

### 9.3 Aggressive headroom arms

At `p=0.001`, run the complete `O-frame` partial-commit factorial:

```text
HW in {10, 0}
boundary_policy in {disabled, odd-parity}
observable_policy in {zero-frame, any}
```

The eight canonical IDs are:

```text
pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2
pu-oframe-window-hw10-boundary-odd-parity-observable-zero-frame-partial-joint-y2
pu-oframe-window-hw0-boundary-disabled-observable-zero-frame-partial-joint-y2
pu-oframe-window-hw0-boundary-odd-parity-observable-zero-frame-partial-joint-y2
pu-oframe-window-hw10-boundary-disabled-observable-any-partial-joint-y2
pu-oframe-window-hw10-boundary-odd-parity-observable-any-partial-joint-y2
pu-oframe-window-hw0-boundary-disabled-observable-any-partial-joint-y2
pu-oframe-window-hw0-boundary-odd-parity-observable-any-partial-joint-y2
```

`observable_policy="any"` admits locally generated frame-bearing paths and lets
`O-frame` decide whether each proposal preserves deterministic U0; it does not
use actual observables. The full factorial prevents HW, boundary, and observable
policy changes from being silently conflated.

These are greedy certified-headroom arms within the frozen candidate family,
ordering, domains, and deterministic U0 tie behavior. They are not universal
ceilings on safe local removal.

Every oracle arm still calls full MWPM repeatedly while certifying candidates
and once more on its final residual syndrome. Oracle runtime must not be
presented as a candidate speedup.

Deployable boundary/temporal and optional flat-yoke guards are separately named
Phase D arms and must never be reported as oracle results.

---

## 10. Phase A: retained V3 replay

### 10.1 Input

Read the immutable corpus:

```text
out/promatch_l1_round1_v3_20260817_32p/pilot/
```

Do not edit, resume, normalize, or copy new rows into this directory.

Start with:

```text
pilot-01-d7-n6-y2-r28-p0.001
```

All 83 regressions and 4 recoveries are retained for this cell. After the
low-noise cell passes replay invariants, process the deterministically retained
samples from the other four cells as secondary stress cases. Those other
samples are capped and must not be treated as representative of their complete
discordant populations.

### 10.2 Required replay steps

For every retained shot:

1. verify retained per-shot detector and actual-observable values, regenerate
   the originating deterministic batch from its seed and shot index when
   checking whole-batch digests, and leave the input corpus unchanged;
2. reproduce recorded U0 and PU predictions bit-for-bit;
3. deterministically reconstruct the original V3 PU path sequence and
   telemetry—the V3 retained rows did not store a per-proposal path ledger;
4. shadow-score every original path using `O-cost` and `O-frame`;
5. identify the first cost or frame conflict in the original path sequence;
6. run the true sequential `PU-O-cost-tx-HW10` trajectory;
7. run `PU-O-frame-tx-HW10` and `PU-O-frame-partial-HW10` as separate
   uncapped trajectories from the same original shot;
8. verify residual-syndrome and accumulated-frame algebra; and
9. emit deterministic per-shot and per-proposal ledgers.

### 10.3 Replay gates

Replay passes only if:

- recorded batch U0 and PU predictions reproduce with zero mismatches;
- scalar unpacked `decode(..., return_weight=True)` returns the same logical
  prediction as the exact bit-packed `decode_batch` U0 path on every replayed
  and fresh shot;
- repeated oracle replay is bit-for-bit deterministic;
- every tx and partial `PU-O-frame` prediction equals `U0-direct` for every
  replayed shot;
- no accepted step violates the cost or frame certificate;
- there are zero negative-`cost_excess` anomalies beyond tolerance;
- every retained original U0/PU-discordant shot, regression or recovery,
  contains at least one **durable original commitment** with
  `F_candidate != F_base` in the shadow audit, independent of its cost class;
- input, residual, path-boundary, and frame GF(2) invariants pass; and
- a test proves that changing actual observables while holding the syndrome
  fixed cannot change an oracle decision.

### 10.4 Replay limitations

The retained corpus is conditional and partly capped. It is suitable for:

- reproducing failures;
- testing implementation invariants;
- locating first conflicting proposals; and
- developing hypotheses about context categories.

It cannot estimate unconditional activation, veto, surviving-commit,
detector-event, or failure rates. It also cannot support a new accuracy claim.

---

## 11. Phase B: fresh paired mechanism screen

Fresh sampling occurs only after Phase A and all unit/property tests pass.

### 11.1 Scope: stress mechanism, not low-noise claim

The primary mechanism cell is explicitly:

```text
d=7
patches=6
yokes=2
rounds=28
p=0.002
style=cz
noise=SI1000
```

Reasons for this choice:

- the original PU effect is large enough to detect implementation mistakes
  quickly;
- V3 observed 16,239 regressions and 1,495 recoveries in 200,000 paired shots;
- V3 also observed material detector-event removal (`R_event=0.786366`); and
- the cell tests the desired accuracy/event-relief tradeoff without rare-event
  sampling.

This is a `p=0.002` stress/mechanism experiment. It does not establish the same
tradeoff at the `p=0.001` low-noise operating point.

### 11.2 Size and ordering

The proposed discovery sequence is:

1. a tiny deterministic integration smoke;
2. a 100-shot-per-arm oracle cost/call-count timing probe at both the primary
   `p=0.002` arm and the worst-case `p=0.001`, HW0, odd-parity,
   observable-any arm;
3. one fixed 20,000-shot `p=0.002` paired mechanism screen containing all frozen
   joint/oracle and per-patch arms whose smoke invariants pass;
4. one fixed 20,000-shot `p=0.001` paired operating-point screen containing the
   per-patch arms and complete eight-arm headroom factorial;
5. the architecture checkpoint in Section 13; and
6. only if that checkpoint selects a practical flat guard, a separately frozen
   guard-selection/power pilot followed by a disjoint fixed-`N` holdout.

The 20,000-shot screen and any later guard-selection pilot are exploratory.
Neither may be promoted into, pooled with, or reused as the holdout. The
holdout size is set by a protocol-frozen power calculation, not assumed in this
document.

Existing timing estimates make the 20,000-shot screen plausible, but they are
not protocol evidence. Stage-3 veto storms can multiply both candidate
enumeration and scalar-decode calls. The cost probe therefore remains and must
record oracle calls, candidate counts, vetoes per state, enumeration time, and
tail wall time. If the uncapped oracle is impractical, use only an explicitly
named frozen budgeted arm; do not silently substitute it under the uncapped
decoder name.

### 11.3 Execution controls

Any fresh collection must:

- use exactly 32 worker processes;
- use exactly one native numerical thread per worker;
- keep `MAX_ERRORS` unset;
- use a fixed shot count independent of results;
- use the same sampled arrays for every arm in a cell;
- for every fresh scientific screen, including the 20,000-shot screen, run
  from a clean worktree at a commit whose only change over the implementation
  HEAD is the frozen protocol JSON;
- write to a fresh output root; and
- never resume into any `promatch_l1_round1*` directory.

The tiny integration smoke and 100-shot cost probe are explicitly
non-scientific scratch runs under `$TMPDIR`; they do not relax the process or
native-thread caps. Do not run two campaigns whose process totals exceed 32.

### 11.4 Shot-paired per-patch projection contract

Generate both exact circuits/DEMs from the maintained generator with identical
`d`, patches, rounds, `p`, style, and noise settings, differing only in
`yokes=2` versus `yokes=0`. Sample only the `yokes=2` circuit.

Construct a frozen role/coordinate mapping:

```text
yokes=2 inner and terminal detector bits
    -> corresponding yokes=0 inner and terminal detector IDs

yokes=2 yoke detector bits
    -> discarded

yokes=0 isolated dummy-yoke detector
    -> forced to zero

observable k
    -> the same observable k
```

The current geometry places the dummy/final yoke detectors after all inner and
terminal detectors, but the implementation must map by validated roles and
coordinates instead of assuming numeric prefixes.

Before accepting the arm, verify and hash:

1. identical noisy body/circuit prefix and observable definitions;
2. bit-identical projected inner/terminal events and observables under fixed
   sampler seeds;
3. equality of the **undecomposed** physical mechanisms after dropping
   `yokes=2` yoke targets and applying the canonical parity-merge algorithm
   below;
4. a native `yokes=0` matcher with exactly `2 * num_patches` nontrivial
   patch/basis components plus one isolated dummy, with no cross-patch or
   cross-basis edge and with every component's fault IDs owned by its expected
   patch/basis observable; and
5. separate `yokes=0` and `yokes=2` circuit, DEM, graph, layout, and projection
   hashes.

Canonicalize the physical-mechanism catalog deterministically:

1. Generate undecomposed DEMs, flatten repeat/shift structure, and treat each
   `error(p)` instruction as one independent Bernoulli mechanism. A `^`
   separator does not split that mechanism.
2. For the `yokes=2` catalog, discard yoke-detector targets. Map every remaining
   detector and observable through the frozen projection, XOR duplicate targets
   modulo two, and sort the resulting detector/observable IDs into one
   signature. Apply the same normalization to native `yokes=0` instructions.
3. Record and omit identity signatures after projection; they have no effect on
   the compared inner/terminal/observable distribution.
4. Group mechanisms by identical signature. For probabilities `p_1, ..., p_m`,
   compute the probability of odd parity:

   ```text
   p_merged = (1 - product_i(1 - 2*p_i)) / 2
   ```

   Evaluate the sorted product using `decimal.Decimal.from_float` at precision
   80 so input-order differences cannot change the result.
5. Sort signatures lexicographically and require identical signature sets and:

   ```text
   abs(p_merged_y2_projected - p_merged_y0)
       <= 1e-15 + 1e-10 * max(abs(p_merged_y2_projected), abs(p_merged_y0))
   ```

Hash the projection, normalized catalogs, omitted-identity ledger, Decimal
precision, merge formula, and tolerance. A separator, target, or instruction
form that cannot be normalized by this algorithm fails closed instead of being
silently dropped.

Do **not** require the decomposed uncorrelated graph weights to be equal.
Development audits found material local-weight changes after yoke removal
(approximately 9% maximum relative change at the primary geometry), even when
the projected physical streams agree. The native `yokes=0` backend therefore
includes a marginal model/reweighting change in addition to removing yoke
constraints.

Evaluate two distinct paired interventions:

- `PU-heldV3/perpatch-y0`: generate the exact V3 `yokes=2` predecode result,
  project its residual, change only the final backend to native `yokes=0`, and
  apply the same accumulated V3 observable frame to that backend prediction;
- `PU-native/perpatch-y0`: project the original syndrome first, then use a
  predecoder and backend both compiled from the native `yokes=0` graph.

The first holds the predecode trace fixed and isolates the final-backend
intervention; reduced harm would support, but not uniquely prove, the yoke
rerouting mechanism. The second measures transfer to independent-patch L1
decoding. Neither is a complete yoked decoder because no L2 decision uses the
discarded yoke syndrome.

An optional weight-preserving projected-`yokes=2` reference/treatment pair may
separately isolate yoke topology/constraint removal. It requires a versioned
transformation that replaces each yoke-hub incidence with an independent
boundary while preserving canonical weights and fault IDs; it must not be
conflated with the native `yokes=0` arm.

### 11.5 Low-noise certified-headroom screen

Run a separately named operating-point screen at:

```text
d=7, patches=6, yokes=2, rounds=28, p=0.001
```

In V3, `PU-window` had `R_event=0.9646567707`, only 3.53% removal,
with zero rollback. Under fixed `HW=10`, disabled boundary, and the same
activated domains, an oracle that only vetoes or rolls back cannot reach the
`R_event < 0.90` stress-cell gate on those shots. Therefore:

- the `<0.90` flat-guard event gate is specific to `p=0.002`;
- no guard selected at `p=0.002` is assumed valid at `p=0.001`;
- the `p=0.001` screen estimates event relief and per-patch/soft-output
  headroom, not powered accuracy noninferiority; and
- the `HW=0` partial/odd-boundary family from Section 9.3 measures aggressive
  greedy certified headroom independent of the fixed HW10 handoff target.

The low-noise screen uses exactly 20,000 shots regardless of interim results.
It is an exploratory precision screen for event/gap estimands, not a powered
logical-accuracy claim. Its HW0 results are sensitivity results under a changed
operating policy, not evidence that the original HW10 configuration meets its
workload gate.

---

## 12. Endpoints and statistics

### 12.1 Oracle mechanism endpoints

Record unconditionally on fresh shots:

- proposals, provisional acceptances, vetoes, and durable post-transaction
  commitments by stage and domain;
- first durable frame incompatibility (`F_candidate != F_base`) for every
  U0/PU-discordant shot, plus whether that same proposal is numerically
  cost-compatible;
- `cost_excess`, backend-reported weight, support-`fsum` weight, and frozen
  tolerance;
- numerically cost-compatible fraction;
- frame-compatible fraction among numerically cost-compatible proposals;
- veto multi-label context;
- provisionally accepted event removal, event removal discarded by rollback,
  durable committed event removal, and vetoed event removal;
- retained fraction of the original PU event removal;
- initial and final global detector HW;
- initial and final per-domain detector HW;
- successful domains, transactional rollbacks, and accepted-before-rollback
  proposals;
- `partial-exhausted` domains, exhaustion reasons, and durable certified prefix
  length;
- number of full-MWPM oracle calls;
- candidates, vetoes, and oracle calls per unchanged state;
- stage-3 candidate-enumeration time and veto-count tails;
- veto-budget hits by stage and state; and
- observable prediction difference masks.

For the fresh unconditioned corpus:

```text
R_event = sum(final residual detector events)
          / sum(original detector events)
```

Do not calculate or report the mean of per-shot ratios as the primary event
estimand. Zero-original-event shots remain in all unconditional activation and
accuracy denominators.

Only commitments durable under the named policy affect the final residual-event
numerator: successful-domain commits for `tx`, and successful-domain commits
plus every certified prefix durably retained by `partial`, whether stopping was
caused by proposal exhaustion or a frozen veto-budget hit. Provisional removals
from a rolled-back domain are diagnostic telemetry and must never be counted as
event relief.

Define partial-policy coverage explicitly:

```text
partial_domain_rate = number of activated domains ending partial-exhausted
                      / number of activated domains

partial_shot_rate = number of shots with at least one partial-exhausted domain
                    / total shots
```

An activated domain is one whose initial domain HW exceeds its target. A zero
activated-domain denominator makes `partial_domain_rate` undefined; never
replace it by zero. Give both rates two-sided 95% percentile intervals from
10,000 empirical type-7 bootstrap replicates that resample complete shots; use
a seed derived from the frozen partial-rate bootstrap root and cell ID.

For per-patch arms, compute a separately named `R_event_L1` over the common
mapped inner/terminal detector support only. Discarded `yokes=2` bits and the
forced-zero `yokes=0` dummy are backend-interface changes, not predecoder event
removal, and must not make the per-patch workload ratio look better. Report
their counts separately.

Reuse and freeze the first-round workload method exactly:

```text
bootstrap unit:
    paired shot via the exact joint (original HW, residual HW) histogram
replicates:
    10,000
quantile method:
    empirical type 7
one-sided alpha:
    0.025
seed:
    derived from the protocol-frozen workload-bootstrap seed root and cell ID
```

If the aggregate original-event denominator is zero, `R_event` and its bound
are undefined and the flat-guard event gate fails.

Also report how much of the original PU relief survives:

```text
R_relief_retained = (K_original - K_oracle)
                    / (K_original - K_PU)
```

where each `K` is an aggregate event sum over the same shots. This quantity is
undefined when the denominator is nonpositive and is descriptive rather than
an advancement gate.

### 12.2 Accuracy endpoints

For arms not equivalent to U0 by construction, retain the paired contingency:

```text
both correct
U0 correct, treatment wrong   (regression)
U0 wrong, treatment correct   (recovery)
both wrong
```

The paired risk difference is:

```text
Delta = (regressions - recoveries) / N
```

Report the one-sided Tango confidence bound using the same implementation as
the first-round experiment.

For every `PU-O-frame` tx or partial arm, exact equality to U0 is an invariant.
Reporting a confidence interval around an equality enforced by the gate would
be misleading. `PU-O-cost` still needs the full paired accuracy table because
objective compatibility does not fix the logical class.

### 12.3 Per-patch and backend-interaction endpoints

For every shot, retain hard-output failure indicators by patch and basis for:

```text
U0-joint
PU-window/joint
U0-perpatch-y0
PU-heldV3/perpatch-y0
PU-native/perpatch-y0
```

The per-patch arms are L1 comparators, not complete yoked decoders. Label their
any-observable rate as an **inner hard-output rate**, never as the logical
failure rate of a replacement yoked decoder.

Use the full shot pairing to report the exploratory backend interaction:

```text
I_backend = mean[
    (Y_PU-window/joint - Y_U0-joint)
    -
    (Y_PU-heldV3/perpatch-y0 - Y_U0-perpatch-y0)
]
```

where `Y=1` iff any retained patch/basis prediction differs from that shot's
actual observable vector. Use a fixed-seed paired-shot bootstrap over this
four-arm contrast;
do not compare two unpaired confidence intervals. The bootstrap contract is:

```text
resampling unit: one complete shot, including every patch/basis output and arm
replicates: 10,000
interval: two-sided 95% percentile interval
quantile method: empirical type 7
seed: derived from the protocol-frozen interaction-bootstrap root and cell ID
```

Report the native per-patch effect
`PU-native/perpatch-y0 - U0-perpatch-y0` under the same contract because its
local candidate weights and ordering also change. Patch/basis outputs are
correlated within a shot and must never be resampled as independent rows.

This interaction is a controlled backend intervention, not unique causal
attribution to one yoke edge: the native `yokes=0` backend also changes the
uncorrelated marginal weights.

If the optional projected-`yokes=2` reference/treatment pair is run, compute its
paired PU-minus-U0 effect and its difference from the joint effect using the
same shot-cluster bootstrap. Only that weight-preserving contrast may support a
specific topology-removal attribution.

### 12.4 Complementary-gap/soft-output checkpoint endpoints

The architecture checkpoint needs per-patch, per-basis confidence in addition
to hard output. The existing
[`GapWorkHandler`](../src/yoked/gap/_gap_worker_handler.py) cannot be applied
directly to the `yokes=0` matcher because its last detector is the isolated
dummy. Before use, implement and validate a per-patch constrained-logical-class
gap, for example by a synthetic logical-check detector or a single-patch
augmented matching problem.

Compute the primary gap comparison on the same patch/basis outputs for both:

- `U0-perpatch-y0`, using the original projected syndrome; and
- `PU-native/perpatch-y0`, using its final residual syndrome and accumulated
  predecoder frame.

For the PU arm, orient the signed complementary classes back to the original
shot by XORing the accumulated frame. The forced predecode-prefix cost is common
to both complementary residual classes, so it cancels from the gap; record it
separately and verify this cancellation explicitly. The resulting PU gap is a
confidence conditional on that forced prefix, not an assertion that the prefix
is part of the native `yokes=0` optimum. `PU-heldV3/perpatch-y0` may receive the
same diagnostic, but it is not the architecture-transfer primary.

For each patch/basis output, report:

- best and complementary constrained correction weights;
- signed/absolute complementary gap;
- hard-output correctness;
- gap distributions conditional on correct versus incorrect output;
- error-discrimination AUROC and selective risk when low-confidence outputs are
  withheld; and
- deterministic agreement with exhaustive constrained decoding on small
  graphs.

Freeze the exploratory soft-output statistics as follows. Let `E=1` denote an
incorrect patch/basis hard output and `q=abs(gap)` denote confidence. Compute
midrank AUROC using `-q` as the score for `E=1`. For each coverage
`c in {0.50, 0.75, 0.90, 0.95, 1.00}`, let `M` be the number of eligible
patch/basis outputs in that estimand, retain the `ceil(c*M)` outputs with the
largest `q`, break ties by `(shot_id, patch_id, basis)`, and report selective
risk `sum(E retained) / ceil(c*M)`. Report pooled micro results and every
patch/basis result separately.

All uncertainty uses 10,000 paired bootstrap replicates, empirical type-7
quantiles, and two-sided 95% percentile intervals. Resample complete shots so
all patch/basis outputs and both arms remain clustered; derive the seed from a
protocol-frozen gap-bootstrap root and cell ID. AUROC is undefined if a
replicate lacks either outcome and such replicate counts are reported; the
soft-output routing flag then fails rather than imputing a value.

Set `native_gap_signal_present=true` only if, for `PU-native/perpatch-y0`:

1. the bootstrap 2.5th percentile of AUROC is greater than `0.5`; and
2. the bootstrap 97.5th percentile of the paired difference
   `selective_risk(0.90) - selective_risk(1.00)` is below zero.

Otherwise set `native_gap_signal_present=false`.

Also report paired PU-minus-U0 differences for AUROC and each selective-risk
point under the same shot-cluster bootstrap. This checkpoint tests whether a
gap is useful as a confidence ranking; fitting and validating calibrated outer
error probabilities remains a later L1-to-L2 protocol. No soft-output or
hierarchical claim is made until the new constrained-class implementation and
these statistics pass their specified tests.

### 12.5 Exploratory validity and routing gates

The `p=0.002` fresh screen is valid for the architecture checkpoint if:

- all replay and algebraic invariants pass;
- the positive-control `PU-window/joint` reproduces the direction of V3 harm;
- every tx and partial `PU-O-frame` arm remains bit-identical to U0;
- the held-trace and native per-patch projection/equivalence invariants pass;
  and
- the constrained per-patch gap implementation passes its exhaustive and
  integration validation tests.

Separately set `stress_flat_guard_gate_passed=true` only if:

- the one-sided 97.5% paired-bootstrap upper bound on `R_event` for the
  `pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2`
  arm is strictly below `0.90`; and
- zero first trajectory-changing vetoes remain `unclassified`.

Otherwise set `stress_flat_guard_gate_passed=false`.

An `unclassified` first veto pauses context-guard interpretation, not the
per-patch hard-output/gap checkpoint. Failure of the event gate likewise routes
the checkpoint toward explicit L1-to-L2 soft output instead of preventing the
checkpoint.

The `0.90` event threshold means this conservative certificate demonstrates at
least 10% detector-event relief. It is not a latency threshold or a maximum on
what another correct certificate might remove.

If the event ratio trends near 1, this certificate has not demonstrated useful
removal and `stress_flat_guard_gate_passed=false`. That result does not prove
that no better exact, batch-aware, or sequence-aware certificate exists.

At `p=0.001`, do not apply the `<0.90` gate or claim accuracy
noninferiority. Report all eight factorial event-ratio bounds, partial
domain/shot rates, per-patch hard output, and validated gap endpoints
descriptively.

---

## 13. Phase C/D: architecture checkpoint before deployable guards

### 13.1 Phase C architecture checkpoint

Do not automatically proceed from the oracle screen to flat-stack guard
optimization. First review together:

- certified partial/HW0 event headroom at `p=0.001`;
- the joint-versus-held-perpatch backend interaction;
- native per-patch hard-output accuracy by patch and basis;
- the fraction of conflicts that require nonlocal yoke visibility; and
- validated per-patch complementary-gap discrimination and risk-coverage.

`O-frame` can reproduce U0 but cannot improve on U0 by construction. The flat
hybrid's potential benefit is therefore cheaper U0-like decoding. A true
hierarchy has a different potential benefit: L1 confidence/soft output may let
an outer decoder outperform hard independent patch decisions.

Apply two independent, frozen routing decisions:

- Advance a separately frozen L1-to-L2 calibration/outer-decoder experiment if
  `native_gap_signal_present=true`.
- Permit Phase D flat-guard discovery only if
  `stress_flat_guard_gate_passed=true` and at least one fresh
  trajectory-changing veto has an architecture-consistent `true_boundary`,
  `cross_window`, or `terminal` label. Otherwise Phase D is not launched.

Thus a yoke-only result cannot justify a local guard, and a failed stress event
gate cannot be overridden by a favorable gap result. Conversely, useful gap
signal can advance the hierarchy path even when the flat guard gate fails.

The `p=0.001` headroom estimates inform priority but have no additional binary
routing threshold in this exploratory plan. Both routed paths may proceed if
both flags pass, but no flat-guard holdout or latency campaign is launched
before this checkpoint is recorded.

### 13.2 Phase D guard contract

Architecture-consistent practical guards keep all candidate generation and
correction paths inside the original `(patch, basis, window)` domain. Added
boundary or temporal context may only veto a proposal; it cannot become a
cross-domain local correction in this study.

Keep fixed for the primary guard comparison:

- `HW=10`;
- stages 1 through 4;
- stage and candidate ordering;
- `d`-round nonoverlapping windows;
- zero-observable-frame local commits;
- local correction paths; and
- the final complete ordinary PyMatching backend.

Use the architecture-consistent factorial first:

| Guard | Boundary context | Temporal/terminal context | Locality |
| --- | ---: | ---: | --- |
| `G-local` (exact `PU-window` positive control) | No | No | Patch/window local |
| `G-boundary` | Yes | No | Patch/window plus static boundary metadata |
| `G-temporal` | No | Yes | Patch plus neighboring-window/terminal metadata |
| `G-boundary+temporal` | Yes | Yes | Patch plus both static context classes |

Yoke context is a separately named flat-stack diagnostic:

| Guard | Permitted visibility | Interpretation |
| --- | --- | --- |
| `G-yoke-flat` | Current yoke bits and every active body/terminal detector in all patches and bases connected through the same yoke hub; complete-graph paths may traverse inactive vertices | Nonlocal global-syndrome veto for the flat hybrid |
| `G-all-flat` | `G-boundary+temporal` plus `G-yoke-flat` visibility | Fully context-aware flat diagnostic |

`G-yoke-flat` is not an independent-patch L1 decoder and must never be called a
local or architecture-transfer guard. A static rule that sees only a local
distance to a yoke hub but not the active defects beyond it is a different,
likely over-vetoing arm and requires a different name.

### 13.3 Veto metric

A one-hop excluded-edge test is insufficient: a competing shortest path may
traverse inactive detector vertices or a yoke hub.

For candidate endpoints `a` and `b`, current visible active alternatives `q`,
and the true boundary `partial`, a provisional conservative guard may require:

```text
d(a,b) < min(d(a,q), d(a,partial)) - margin
d(b,a) < min(d(b,q), d(b,partial)) - margin
```

for every alternative enabled by the guard arm. Distances are complete-graph
shortest-path distances under the same frozen edge weights. Both endpoints must
strictly and reciprocally prefer each other. Ties or distances within the
frozen margin are vetoed.

The exact visible-destination set, boundary treatment, distance caching,
singleton test, margin, and numeric tolerance must be specified and frozen
before fresh guard sampling. They cannot be tuned on the holdout.

This reciprocal strict-nearest rule is a conservative heuristic, not proof of
global MWPM compatibility. Evaluate it against two different labels:

- `O-cost` for objective/support compatibility with at least one optimum; and
- conditional `O-frame` for equivalence to PyMatching's selected deterministic
  U0 tie behavior.

Do not train a guard to reproduce arbitrary `O-frame` ties unless exact U0 bit
equivalence is the explicit target. Neither label replaces the fresh paired
accuracy outcome: `O-cost` can select another logical class, and `O-frame` can
reject a physically equivalent degenerate optimum.

Partial context graphs must not be constructed by casually deleting edges if
that creates parity or boundary inconsistencies. Each guard is an explicit veto
predicate over the complete graph metadata.

### 13.4 Practical-guard advancement gate

For the `d=7, p=0.002` reference cell, a later frozen guard protocol should use:

```text
accuracy noninferiority margin
    delta_NI = 0.05 * 0.03628 = 0.001814

required accuracy result
    one-sided 97.5% Tango upper bound on Delta < delta_NI

required detector-event result
    one-sided 97.5% paired-bootstrap upper bound on R_event < 0.90
```

These are guard criteria, not oracle criteria. A discovery screen only selects
and debugs one guard. The selected rule must then be frozen and evaluated on
new shots. The `<0.90` event criterion remains a `p=0.002` stress-cell gate;
the chosen guard receives a separate descriptive `p=0.001` operating-point
evaluation under the rules in Section 11.5.

---

## 14. Latency follows the architecture checkpoint and guard accuracy

The oracle never enters the latency comparison.

For a deployable guard, measure three distinct quantities on paired pregenerated
corpora:

### 14.1 Detector-event relief

```text
R_event = residual active events / original active events
```

### 14.2 Residual-backend latency

Time the same precompiled complete matcher on original and residual syndromes:

```text
T_backend_original = MWPM(original syndrome)
T_backend_residual = MWPM(residual syndrome)
```

This isolates whether fewer active events make PyMatching faster even though
the static graph is unchanged.

### 14.3 End-to-end latency

Compare:

```text
T_U0 = MWPM(original syndrome)

T_guarded_PU = context guard
               + predecode
               + conversion/packing
               + MWPM(residual syndrome)
               + frame application
```

A flat-stack latency benefit requires:

```text
T_guarded_PU < T_U0
```

Possible outcomes:

- backend and end-to-end latency improve: useful flat-stack speedup;
- backend improves but end-to-end does not: predecoder overhead erases the
  backend saving;
- backend does not improve: detector-event count is not a useful PyMatching
  workload proxy at this operating point; or
- software does not improve but a finite-capacity hardware backend remains a
  separate, unproven hypothesis.

A genuinely smaller outer decoding graph requires a true L1-to-L2 decoder or a
backend that extracts only the unresolved residual subgraph. This oracle study
does not provide that architecture.

---

## 15. Required implementation work

No command below exists merely because it appears in this document. Tooling is
implemented and tested before any proposed CLI is advertised as runnable.

### 15.1 Decoder changes

1. Add an immutable `CommitProposal` record containing domain, stage,
   endpoints, canonical edge IDs, detector boundary, observable frame,
   decision weight, and active-state fingerprint.
2. Refactor the domain engine into a deterministic proposal/accept/veto
   stepper without changing the V3-disabled behavior.
3. Add state-scoped candidate blacklisting and termination rules.
4. Assert canonical square-free candidate support and strictly positive finite
   canonical graph weights for theorem-bearing oracle arms.
5. Add a read-only complete-graph oracle evaluator using
   `decode(..., return_weight=True)`.
6. Reconstruct every returned support weight with `math.fsum` from the same
   canonical edge table and persist the weight-comparison calibration.
7. Cache the current state's U0 prediction and weight; invalidate the cache
   after every accepted proposal or rollback.
8. Implement both whole-domain transactional and durable-prefix partial
   semantics.
9. Add an optional frozen per-state veto budget that never accepts an
   uncertified proposal.
10. Add the shadow-audit and sequential-oracle modes as different APIs.
11. Add multi-label full-context classification.
12. Add bounded telemetry and deterministic retained proposal ledgers.
13. Add separately named configurations encoding oracle type, HW target,
   boundary/observable policy, tx/partial semantics, and veto budget.
14. Validate every canonical `arm_id` against
   `^[a-z0-9]+(?:-[a-z0-9]+)*$`; display aliases never become artifact paths.
15. Add a per-patch constrained-logical-class gap implementation; do not reuse
   the current last-detector gap trick on the isolated `yokes=0` dummy.

### 15.2 Harness changes

1. Run every joint, oracle, held-trace per-patch, and native per-patch arm on
   identical sampled `yokes=2` arrays.
2. Retain exact paired predictions and digest checks.
3. Build, verify, and hash the `yokes=2` to `yokes=0` detector/observable
   projection, both directly generated DEMs, and the canonical projected
   mechanism catalogs.
4. Aggregate proposal, veto, `cost_excess`, frame, transaction, veto-budget,
   enumeration-time, oracle-call, and event histograms.
5. Retain patch/basis hard outputs and compute the paired backend interaction.
6. Persist complementary-gap discrimination/risk-coverage statistics only
   after the constrained-class implementation passes validation.
7. Derive batch seeds only from the frozen seed root and batch ID.
8. Record complete circuit, DEM, graph, layout, projection, decoder, tolerance,
   runtime, and source
   hashes.
9. Keep input V3 paths read-only and write all new artifacts elsewhere.
10. Fail closed on missing proposals, malformed weights, invariant violations,
   crashes, or incomplete batches.

### 15.3 Planned diagnostic interface

A future interface may look like:

```bash
# Planned only; not implemented at the time of this specification.
tools/diagnose_promatch_l1 oracle-replay \
    --input out/promatch_l1_round1_v3_20260817_32p/pilot \
    --cell pilot-01-d7-n6-y2-r28-p0.001 \
    --out "$TMPDIR/promatch-oracle-replay-v1"
```

Fresh sampling requires a separate frozen protocol and benchmark-harness mode.
Do not overload the existing V3 decoder name or output schema.

---

## 16. Required tests

### 16.1 Oracle unit tests

- numerically cost-compatible proposal with the same frame is accepted;
- positive-cost-excess proposal is vetoed;
- equal-cost proposal with a different logical frame distinguishes `O-cost`
  from `O-frame`;
- negative `cost_excess` beyond tolerance fails closed;
- nonpositive or nonfinite theorem-bearing edge weight fails closed;
- a nonpositive or nonfinite canonical edge fails closed even when no candidate
  traverses it;
- repeated candidate edge fails the square-free-support invariant;
- backend-reported and canonical support-`fsum` weights reconcile under the
  frozen tolerance;
- every returned endpoint pair maps to exactly one canonical edge, duplicate
  returned edges fail, and XOR of reconstructed fault masks equals `decode`;
- `tau_weight` and `tau_k` use their separately defined scales, and the frozen
  tolerance-sensitivity grid produces zero classification changes;
- actual observables cannot enter the oracle API;
- cached and uncached results agree;
- disabled oracle reproduces V3 proposal order and output bit-for-bit;
- accepted `O-frame` steps preserve the complete prediction invariant;
- veto leaves syndrome, frame, path support, and telemetry state unchanged;
- state-scoped blacklisting terminates and does not leak across states;
- whole-domain rollback discards all provisional paths and frames;
- partial exhaustion durably keeps exactly the certified prefix and remains
  U0-equivalent;
- veto-budget exhaustion rolls back `tx`, stops-and-keeps `partial`, and never
  commits the rejected candidate; and
- repeated execution is deterministic.

### 16.2 Exact small-graph tests

Construct exhaustive graphs containing:

- one boundary alternative;
- one yoke-hub alternative;
- adjacent time-window alternatives;
- a terminal alternative;
- multiple equal-weight optima;
- equal-weight optima with different logical classes;
- overlapping paths and GF(2) cancellation;
- disconnected components with and without valid boundaries;
- very small and large strictly positive finite edge weights; and
- zero/nonpositive candidate edges as unsupported fail-closed cases.

Enumerate all correction supports on sufficiently small positive-weight graphs
and prove both directions:

```text
exact cost_excess = 0
    iff
candidate support is contained in at least one minimum-weight correction.
```

Also verify `cost_excess >= 0`, the O-frame induction invariant, and explicit
rejection of unsupported zero/nonpositive theorem-bearing cases.

### 16.3 Integration/property tests

- input syndrome equals residual syndrome XOR committed path boundary;
- accumulated observable frame equals XOR of committed path frames;
- inactive shots match U0 exactly;
- scalar `decode(..., return_weight=True)` prediction matches bit-packed
  `decode_batch` U0 prediction for every tested syndrome;
- V3 retained samples replay bit-for-bit before oracle intervention;
- every sequential `O-frame` final prediction matches U0;
- tx and partial arms begin from the same original shot and, whenever their
  active-state fingerprints agree, produce the same ordered proposals and
  oracle decisions; after a rollback-versus-prefix outcome changes the state,
  each trajectory proceeds independently with correct transaction telemetry;
- the `yokes=2`/`yokes=0` role projection is bijective on inner/terminal
  detectors and identity on observables;
- projected fixed-seed inner detector streams and observables agree;
- projected undecomposed mechanisms agree after parity merge within frozen
  tolerance;
- mechanism-catalog canonicalization is invariant to instruction order,
  reproduces the analytic odd-parity probability, preserves one mechanism
  across `^` separators, and records projected identities;
- native `yokes=0` matching is block diagonal with an isolated dummy and no
  cross-patch/cross-basis edges;
- decomposed `yokes=0` and `yokes=2` weights are recorded separately and are
  not required to agree;
- held-V3-trace and native-y0-predecode arms remain distinct in configuration
  and provenance;
- per-patch constrained-class gaps agree with exhaustive small-graph gaps and
  never flip the isolated dummy detector;
- proposal and batch ledgers reconcile to aggregate histograms;
- single-process and 32-process results match for fixed seeds; and
- interrupted fresh collection resumes without duplicate or missing batch IDs.

---

## 17. Artifact and provenance contract

### 17.1 Inputs

The V3 input directory remains immutable:

```text
out/promatch_l1_round1_v3_20260817_32p/pilot/
```

### 17.2 Future outputs

Use new versioned roots, for example:

```text
out/promatch_l1_global_context_oracle_v1/replay/
out/promatch_l1_global_context_oracle_v1/screen/
out/promatch_l1_global_context_oracle_v1/analysis/
```

The exact names are frozen in the future protocol. Scratch and cost probes go
under `$TMPDIR`, never the home directory.

### 17.3 Per-proposal record

Each retained proposal record includes at least:

```text
schema version
experiment/arm_id/cell/batch/shot identity
input syndrome digest
state fingerprint
domain, stage, proposal index
ordered endpoints and canonical path edge IDs
path detector boundary and observable frame
path decision weight
current U0 prediction and weight
candidate residual U0 prediction and weight
current/residual returned canonical edge supports
backend-reported and canonical support-fsum weights
cost_excess, tau_k, and tau_weight values
cost/frame classifications
oracle decision
multi-label omitted context
post-decision state fingerprint
transaction policy and domain outcome
partial-prefix/rollback status
veto budget, veto count, candidate count, and oracle-call count
```

Unbounded per-proposal data must not be embedded into one `summary.json`.
Persist aggregate histograms unconditionally and retain detailed records using
a deterministic, protocol-frozen policy. Every retained ledger receives a
cryptographic digest and reconciles to its batch summary.

### 17.4 Reproducibility

Record:

- Git commit and dirty-worktree status;
- exact protocol and experiment IDs;
- Python and dependency versions;
- circuit and DEM hashes;
- both yokes=2/yokes=0 and optional projected graph/layout fingerprints;
- detector/observable projection, canonical mechanism-catalog, omitted-identity
  ledger, and equivalence-report hashes;
- tolerance-calibration corpus, graph, result, and configuration hashes;
- canonical `arm_id`, display label, decoder semantics, HW,
  boundary/observable policy, tx/partial policy, veto budget, backend type, and
  source hashes;
- oracle-cache, call-count, veto-tail, and enumeration-time telemetry;
- seed roots and derived batch seeds;
- process and native-thread settings; and
- host metadata required by the existing harness.

This document itself makes the worktree dirty until committed. Every future
fresh scientific screen, not only a later claim-bearing holdout, must start
from the clean committed state required by the repository's scientific-run
rules.

---

## 18. Decision table

| Observation | Interpretation | Next action |
| --- | --- | --- |
| `O-frame` violates U0 equality | Oracle/stepper/frame implementation bug | Stop; fix invariants before sampling |
| Negative `cost_excess` occurs beyond tolerance | Weight source, support algebra, or tolerance is invalid | Stop; reconcile backend and support-`fsum` weights |
| A U0/PU-discordant shot has no durable original frame incompatibility | Shadow ledger or frame induction is wrong | Stop; fix reconstruction before interpretation |
| `PU-O-frame-partial-HW10` passes the `p=0.002` event gate and no first veto is `unclassified` | Certified durable relief exists in the stress regime and is interpretable for guard design | Set `stress_flat_guard_gate_passed=true`; use the architecture checkpoint before guard work |
| `PU-O-frame-partial-HW10` fails the `p=0.002` event gate | The tested flat certificate lacks stress-regime workload relief | Set `stress_flat_guard_gate_passed=false`; keep the checkpoint and prioritize the hierarchy path |
| Tx relief collapses but partial relief survives | All-or-nothing rollback, not commitment safety, dominates durable relief | Retain partial policy as the practical comparator |
| HW10 relief is weak at `p=0.001`, but HW0/odd-boundary relief is material | Headroom exists only after changing the operating policy | Treat as sensitivity; redesign handoff/boundary before any claim |
| All low-noise headroom arms have `R_event` near 1 | Tested greedy certificates find little removable work | Prioritize per-patch soft output or another decoder architecture |
| Held-V3 per-patch backend sharply reduces the joint-backend interaction | The final-backend intervention—yoke topology and/or native marginal reweighting—materially changes PU harm | Prioritize L1/L2 separation; require the projected-`yokes=2` pair before unique topology attribution |
| Held-V3 and native per-patch arms retain substantial PU harm | Boundary, temporal, local ordering, or frame mechanisms remain | Prioritize architecture-consistent boundary/temporal diagnosis |
| `native_gap_signal_present=true` | L1 gaps preserve useful confidence ranking after native predecode | Advance a separately frozen calibration and explicit hierarchical L1-to-L2 experiment |
| `O-cost` accepts many proposals that `O-frame` rejects | Deterministic tie-dependent logical class is important | Quantify it; do not fit arbitrary U0 ties unless bit-equivalence is the target |
| Boundary/temporal guards pass fresh accuracy and event gates | Architecture-consistent flat operating point exists | Freeze a disjoint holdout, then run backend/end-to-end timing |
| Only `G-yoke-flat` passes | Useful guard requires global all-patch syndrome | Classify it as flat/nonlocal and prioritize explicit L2 instead |
| Guard passes accuracy but not event relief | Correct but not useful for this backend target | Do not proceed to latency claims |

---

## 19. Relationship to later hierarchical decoding

This remains a flat hybrid:

```text
local L1 predecode -> complete-width residual syndrome -> full-joint MWPM
```

A true hierarchy instead requires:

```text
patch-level L1 decoder
    -> logical result plus calibrated confidence/soft information
    -> explicit outer yoke/QDPC syndrome
    -> small L2 decoder
```

The oracle and per-patch arms inform that architecture in different ways:

- oracle conflicts show where a hard local commitment needs abstention;
- held-trace per-patch decoding separates residual yoke rerouting from the
  original local trace;
- native per-patch decoding measures the actual independent-L1 hard output;
  and
- constrained complementary gaps test whether L1 can export useful ranked soft
  information instead of only a hard frame.

Phase D flat guards and their latency study are sequenced behind or alongside
this soft-output checkpoint. If low-noise certified removal is weak or only a
globally visible yoke guard works, the primary next architecture is explicit
L1-to-L2 decoding, not additional tuning of the flat hybrid.

This experiment still does not define the final outer likelihood model or
measure full hierarchical logical accuracy. Those require a separately frozen
L1-to-L2 experiment after the constrained per-patch gap implementation is
validated.

---

## 20. References

- [First-round implementation plan](../docs/PROMATCH_IMPLEMENTATION_PLAN.md)
- [Figure 8 1D reproduction and diagnostic notes](../REPRODUCING_FIG8_1D.md)
- [V3 frozen pilot protocol](../docs/PROMATCH_PILOT_FROZEN_V3.json)
- [Current ProMatch core](../src/yoked/decoding/_promatch.py)
- [Current compiled adapter](../src/yoked/decoding/_promatch_decoder.py)
- [Current graph compiler](../src/yoked/decoding/_promatch_graph.py)
- [Current paired experiment harness](../src/yoked/decoding/_promatch_experiment.py)
- [Current diagnostic tool](../tools/diagnose_promatch_l1)
