# Confidence-Gated Union-Find with Residual Global MWPM

- **Status:** V1 stopped at replay; V2 stopped at the characterization
  telemetry-cap gate; V3 recovery in progress
- **Date:** 2026-08-30
- **Scientific role:** non-claim-bearing, single-cell paired characterization
- **Physical cell:** `d=7`, SI1000 `p=0.003`, six patches, two yokes, 28 rounds
- **Accuracy sampling:** a fresh 1,000-shot engineering shakeout followed by a
  fresh 10,000-shot characterization
- **Latency sampling:** controlled timing on the immutable 10,000-shot
  characterization corpus; timing observations do not enter the accuracy
  denominator
- **Implementation status:** decoder, paired collector, analyzer, and latency
  paths are implemented. The first frozen 1,000-shot run was rejected because
  non-characterization verification did not expose its already-authenticated
  detector bytes to the replay request. V2 fixes that harness boundary without
  changing decoder semantics and restarted from disjoint roots. Its 1,000-shot
  gate passed, but the disjoint 10,000-shot run stopped before committing any
  range because the frozen 128 MiB normalized-telemetry ceiling was smaller
  than a 312/313-shot range. V3 retains the decoder and statistical design,
  raises only that ceiling to 512 MiB, adds an exact 313-shot capacity probe
  plus a probe-derived 2x safety gate, and restarts from five new roots.
- **Claim status:** no accuracy-preservation, speedup, workload-reduction, or
  hardware-latency claim is authorized

This document specifies the first experiment for a fast weighted Union-Find
(UF) frontend in the Yoked Surface Code (YSC) decoding stack. The frontend
works patch-locally, commits only components that pass a frozen confidence
gate, and hands the exact residual syndrome to ordinary Global MWPM on the
complete yoked graph. It does not delete matching-graph edges or replace the
global residual decoder.

The words *weighted*, *patch-local*, *confidence-gated*, and *UF–MWPM* are not
by themselves an executable decoder definition. The confidence threshold,
numeric rules, deterministic budgets, and seed roots listed as freeze blockers
in [Section 21](#21-open-decisions-and-freeze-blockers) must be made literal in
a frozen protocol before any shakeout or characterization shot is sampled.
The smaller smoke/probe inputs in Section 10 are non-scientific scratch inputs
and cannot enter either experiment corpus.

## Contents

1. [Decision summary](#1-decision-summary)
2. [Scientific questions and claim boundary](#2-scientific-questions-and-claim-boundary)
3. [Standard terminology and arm names](#3-standard-terminology-and-arm-names)
4. [Physical cell](#4-physical-cell)
5. [Canonical graph and patch projection](#5-canonical-graph-and-patch-projection)
6. [Decoder-arm contracts](#6-decoder-arm-contracts)
7. [Weighted UF and confidence-gate contract](#7-weighted-uf-and-confidence-gate-contract)
8. [Residual algebra, transactions, and fallback](#8-residual-algebra-transactions-and-fallback)
9. [Sampling and pairing](#9-sampling-and-pairing)
10. [Ordered launch gates](#10-ordered-launch-gates)
11. [Accuracy endpoints and statistics](#11-accuracy-endpoints-and-statistics)
12. [Workload and confidence diagnostics](#12-workload-and-confidence-diagnostics)
13. [Latency experiment](#13-latency-experiment)
14. [Telemetry and deterministic replay](#14-telemetry-and-deterministic-replay)
15. [Artifacts, integrity, and resume](#15-artifacts-integrity-and-resume)
16. [Protocol freeze and provenance](#16-protocol-freeze-and-provenance)
17. [Implementation plan](#17-implementation-plan)
18. [Required tests](#18-required-tests)
19. [Planned command workflow](#19-planned-command-workflow)
20. [Interpretation and follow-up routing](#20-interpretation-and-follow-up-routing)
21. [Open decisions and freeze blockers](#21-open-decisions-and-freeze-blockers)
22. [Local references](#22-local-references)

---

## 1. Decision summary

The initial experiment makes the following choices.

| Item | Decision |
| --- | --- |
| Geometry | `d=7`, `patches=6`, `yokes=2`, `rounds=28=4d` |
| Noise | SI1000 at `p=0.003` (0.3%) |
| Circuit | CZ style, `remove_x_yoke=false` |
| Accuracy reference | **Global MWPM** |
| Treatment | **Confidence-Gated UF–MWPM** |
| Latency controls | **Adapter-Control MWPM** and **UF-Shadow MWPM** |
| UF scope | Full time history within each `(patch, basis)` lane |
| UF ownership | Body, terminal, cross-window, and true-boundary mechanisms that remain inside one lane |
| Global ownership | Yoke detectors and every edge crossing a patch/basis ownership boundary |
| Commit unit | A final UF component is indivisible; accepted components from both basis lanes are applied as one checked patch transaction |
| Observable policy | Zero-observable-frame UF commits in V1; frame algebra is nevertheless computed and checked |
| Residual backend | One ordinary uncorrelated Global MWPM decode on the complete unchanged graph |
| Graph sparsification | Syndrome reduction only; no per-shot graph-edge deletion in V1 |
| Sampling | Fresh fixed `N=1,000`, then fresh fixed `N=10,000`; no pooling |
| Primary accuracy endpoint | Paired any-observable physical-shot failure difference |
| Primary workload endpoint | Total residual detector events divided by total original detector events |
| Primary frontend-coverage endpoint | Total detector defects in durable UF components divided by total original detector events |
| Primary cluster endpoints | Exact final-component defect-count histogram and the shot-weighted largest-final-component distribution |
| Primary latency endpoint | Batch-1 in-process treatment/Global-MWPM total-latency ratio |
| Scientific posture | Exploratory characterization, not a preservation or speedup test |

The 1,000-shot run is an engineering shakeout, not a pilot whose favorable
outcome earns promotion. The 10,000-shot run is a new disjoint paired corpus.
The implementation and frozen protocol must not change between an accepted
shakeout and characterization.

## 2. Scientific questions and claim boundary

### 2.1 Questions

The experiment asks:

1. How often can a conservative patch-local weighted UF frontend commit a
   correction before complete Global MWPM?
2. How does the resulting complete-decoder failure rate change relative to
   Global MWPM on the exact same physical shots?
3. How much syndrome workload reaches the residual Global MWPM backend?
4. Does reduced backend work compensate for UF, gate, packing, and residual
   construction overhead in measured software latency?
5. What are the component-weighted cluster-size distribution and the
   shot-weighted largest-cluster tail?
6. Which component, port, confidence, or budget conditions cause the frontend
   to commit or defer?

The motivating engineering hypothesis is that clearly isolated local syndrome
components can be handled cheaply, leaving a smaller residual problem for the
global matcher. The competing risk is that a locally plausible correction
changes the global pairing or yoke routing and creates a logical regression.
The confidence gate is intended to expose that tradeoff rather than assume it
away.

### 2.2 What this experiment may report

The fixed 10,000-shot corpus may report:

- the two marginal failure rates and their complete paired contingency table;
- paired regressions, recoveries, and risk difference;
- activation, acceptance, deferral, rollback, and fallback rates;
- original and residual syndrome workload;
- completed and censored cluster-size, geometry, support, and resource
  distributions;
- descriptive associations between confidence and downstream outcomes; and
- controlled in-process software latency on the recorded workstation.

### 2.3 What this experiment cannot establish

This experiment cannot establish:

- equivalence, noninferiority, or preservation of Global-MWPM accuracy;
- a general speedup over distances, physical error rates, or patch counts;
- a threshold, pseudothreshold, or distance-scaling result;
- hardware, FPGA, cryogenic, controller, or real-time latency;
- a benefit from hard graph-edge deletion;
- optimality or calibration of the confidence threshold; or
- performance of correlated matching or any non-MWPM residual backend.

There is no noninferiority margin and no result-dependent pass/fail threshold.
A nonsignificant paired test, zero observed discordance, or an interval that
includes zero is not evidence of equivalence.

## 3. Standard terminology and arm names

The following names are normative in code, tables, plots, and prose.

| Arm ID | Display name | Role |
| --- | --- | --- |
| `global-mwpm-u0-joint-y2` | **Global MWPM** | Accuracy reference and timing reference |
| `weighted-uf-fullhistory-patchlocal-zeroframe-residual-global-mwpm-v1` | **Confidence-Gated UF–MWPM** | Accuracy treatment and timing treatment |
| `adapter-control-global-mwpm-v1` | **Adapter-Control MWPM** | Latency-only interface/packing control |
| `weighted-uf-shadow-global-mwpm-v1` | **UF-Shadow MWPM** | Latency-only UF/gate-work control |

Only Global MWPM and Confidence-Gated UF–MWPM are accuracy arms.
Adapter-Control MWPM and UF-Shadow MWPM must return exactly the same packed
prediction as Global MWPM for every valid input. Their equality is a software
validity invariant, not an additional accuracy result.

The maintained codebase also calls the Global-MWPM path `U0-direct` and uses
historical IDs such as `u0-joint-y2`. Those are exact aliases recorded in the
protocol, not additional arms. Tables and prose use the clearer display name
**Global MWPM**.

Definitions used throughout the specification:

- **UF frontend:** weighted patch-local clustering, deterministic peeling,
  confidence evaluation, and proposal construction before the global matcher.
- **Residual syndrome:** the original detector vector XOR the boundary of all
  durable UF support.
- **Residual Global MWPM:** ordinary Global MWPM evaluated once on that exact
  residual syndrome and the complete original matching graph.
- **Lane:** the full-history detector subgraph belonging to one
  `(patch_id, check_basis)` pair.
- **Port:** an immutable reference to a canonical edge that leaves a lane.
  A port supplies confidence context but is not correction support in V1.
- **Final component:** a terminal UF connected component after the lane has
  processed every simultaneous event batch and reached termination.
- **Cluster size:** the number of original nonzero detector events in a
  completed final component's claimed syndrome boundary; graph vertices and
  correction edges are separate metrics.
- **Commit:** make the component's square-free support durable and remove its
  exact detector boundary from the residual input.
- **Defer:** leave the component's detector events for Global MWPM.
- **Rollback:** discard a tentative support and restore the exact pre-transaction
  syndrome/frame state.

The earlier phrase `frontend-bypass MWPM` is retired because it obscures which
work is bypassed. When it meant direct decoding without a frontend, the
standard name is **Global MWPM**. When it meant an identity wrapper used to
measure adapter overhead, the standard name is **Adapter-Control MWPM**.

## 4. Physical cell

The only physical cell in V1 is:

```text
cell_id                         cguf-01-d7-n6-y2-r28-p0.003
patch_diameter d                7
num_patches                     6
yokes                           2 (one X yoke and one Z yoke)
rounds                          28 (= 4d)
style                           cz
noise                           si1000
p                               0.003
remove_x_yoke                   false
DEM decompose_errors            true
DEM approximate_disjoint_errors true
```

The circuit has 8,354 detectors and 12 observables under the current generator.
Those values are authenticated protocol fields, not values to infer at run
time.

### 4.1 Historical planning anchor

The immutable V3 ProMatch pilot contains the same physical cell under the old
ID `pilot-03-d7-n6-y2-r28-p0.003`. Its frozen reference values are:

```text
circuit_sha256  8cfa9bb9eaf6db86dfc9ffcfefa4582eb29932d11dc1fd911239ef1425841ff9
DEM_sha256      9b06141668bef9b334df78e4700853f60505a71f3c4185746d0261e7e3790e0a
historical_windowd_layout_fingerprint
                5b0269e3827e55b2ac5d89da6f71b94c666951a2c9678dfd44f0fcf235f456e9
```

The old zero-frame graph fingerprint was
`f51536e5d3f7c0af9ef809513431648a79d1b8127a267acb2afe01c5bb16c556`.
The new implementation compiles an all-frame canonical edge table before its
explicit port classification, so it must generate and freeze a new graph and
projection fingerprint instead of reusing the old decoder fingerprint.

At this cell, historical Global MWPM failed on 67,574 of 200,000 shots
(`0.33787`). This suggests roughly 3,379 Global-MWPM failures in 10,000 new
shots if the new sample is similar. The old PU-window predecoder failed at
`0.66949`, with 74,234 regressions and 7,910 recoveries. These numbers justify
the cell as a measurable stress case, but they are neither new data nor a
prediction for weighted UF.

## 5. Canonical graph and patch projection

### 5.1 Compile the complete graph first

The implementation must compile one canonical graph from the decomposed DEM
before constructing any patch projection. Every canonical edge stores:

```text
edge_id
source detector ID
target detector ID or true boundary (None)
exact binary64 weight, serialized with float.hex()
little-endian observable mask
source and target detector roles
```

Dense `edge_id` is the only support identity used by UF, replay, frame
composition, and Global MWPM provenance. The compile path uses
`require_zero_frame=false`; zero-frame eligibility is checked explicitly only
after every edge has been classified. This prevents a nonzero-frame port from
disappearing before the confidence gate can see it.

The compiler must also validate the unmerged decomposed-DEM mechanism catalog.
If PyMatching graph merging has erased an ambiguity in parallel mechanisms or
observable masks that cannot be represented losslessly, protocol compilation
fails closed.

### 5.2 Full-history patch lanes

There are 12 lanes: two check bases for each of six patches. Unlike the
existing windowed ProMatch projection, each lane spans the entire 28-round
history and contains both body and terminal detectors belonging to its patch
and basis. Cross-window edges remain local and correction-eligible.

For V1, a canonical edge is correction-eligible exactly when:

1. both non-boundary endpoints belong to the same full-history lane, or it is
   a genuine `target=None` edge incident to a detector in that lane;
2. its observable mask is zero;
3. its weight is finite and strictly positive; and
4. its role pair is one of: same-lane body–body, same-lane body–terminal,
   same-lane terminal–terminal, or lane-detector–true-boundary.

Body/terminal–yoke edges are ports. Every other role pair is rejected at
compile time. The allowlist and its classification precedence are part of the
projection fingerprint.

Every edge with one lane-local endpoint and another endpoint outside that lane
is a guard port. In the selected two-yoke circuit, that includes body–yoke and
terminal–yoke edges. A yoke detector is never reinterpreted as a matching
boundary. Ports retain the remote detector ID, weight, observable mask, and
canonical edge ID, but they never enter durable support. The deployed gate
cannot read the remote/yoke syndrome bit. The experimental ledger may join
that bit only after the gate decision for diagnostic telemetry; it cannot
affect V1 growth, confidence, commit, or defer behavior.

True `target=None` boundary edges are local correction mechanisms, not ports.
This distinction is mandatory.

### 5.3 Current development taxonomy

An unauthenticated development observation at source commit
`5888b372f62fc616efcea20fb785b02f81a3d67b` compiled the selected circuit and
DEM, called `compile_layout(dem, mode="fullhistory")`, then called
`compile_matching_graph(dem, layout, require_zero_frame=False)`. It produced
40,836 canonical edges, full-history layout fingerprint
`0af0345e3e778552f0b2392b20e59efa2b4143b4847d3ed1808a84b939003a1a`, and
all-frame graph fingerprint
`5f4639248bec80d73cd1a4cf85006188e964cfa23498d9ba6af41a49afc6e0cb`.
The following table is a review aid, not authenticated future protocol data.
Implementation commit A must regenerate and authenticate it.

| Edge class | Count | Nonzero-frame count | V1 owner |
| --- | ---: | ---: | --- |
| Same-window body–body | 33,936 | 0 | UF lane |
| Cross-window body–body | 2,772 | 0 | UF lane |
| Body true-boundary | 1,344 | 0 | UF lane |
| Body–terminal | 924 | 0 | UF lane |
| Terminal–terminal | 420 | 0 | UF lane |
| Terminal true-boundary | 48 | 0 | UF lane |
| Body–yoke | 1,344 | 1,344 | Guard port/global |
| Terminal–yoke | 48 | 48 | Guard port/global |

The expected V1 projection therefore has 39,444 correction-eligible edges and
1,392 canonical guard-port edges. Each canonical port edge has one local
reference in its incident lane. If a future topology has a non-yoke cross-lane
edge, both incident lanes receive immutable references to the same canonical
edge ID while the edge itself remains globally owned.

A mismatch in detector roles, edge counts, weights, masks, ownership, or
fingerprints blocks protocol freeze. The runner never silently accepts a
nearby graph.

### 5.4 Projection fingerprint

The projection fingerprint covers at least:

- the full canonical graph fingerprint;
- ordered lane keys and ordered detector IDs;
- every correction-eligible canonical edge ID;
- every port record, kind, local/remote detector and lane, weight hex, and
  observable-mask hex;
- the zero-frame, weight, and topology policies; and
- a projection schema version.

No unordered set or map iteration may affect this representation.

## 6. Decoder-arm contracts

### 6.1 Global MWPM

Global MWPM runs ordinary uncorrelated PyMatching on the complete decomposed
two-yoke detector error model and returns one packed prediction for all 12
observables. It receives the original packed detector sample without UF work.

The experiment's baseline implementation must bit-match the maintained direct
PyMatching path on scalar, empty, noncontiguous, and batch inputs. It is both
the accuracy reference and the residual backend used by the treatment.

### 6.2 Confidence-Gated UF–MWPM

For each input shot, the treatment performs this exact high-level sequence:

1. Copy or immutably view the original packed detector vector.
2. Run weighted UF independently in every active full-history lane.
3. Construct final UF components and deterministic tentative peeling support.
4. Evaluate the frozen confidence and eligibility rules per final component.
5. Assemble accepted component support for both basis lanes of each patch.
6. Independently recompute and validate the aggregate patch boundary and
   observable frame, then make the patch transaction durable atomically.
7. XOR all durable patch boundaries into the original detector vector.
8. After every shot in the public batch has a residual vector, invoke one
   complete-graph Global-MWPM `decode_batch` call for that residual batch.
9. XOR the durable UF observable frame with the Global-MWPM frame.
10. Return one packed prediction for all 12 original observables.

UF never consumes yoke syndrome as if it were a local boundary. All yoke
detectors remain in the residual vector for the complete global matcher.

### 6.3 Adapter-Control MWPM

Adapter-Control MWPM executes the treatment's public adapter, validation,
packing, and projection-selection plumbing, but constructs no UF proposal and
makes no syndrome or frame change. It then calls Global MWPM on the original
syndrome.

Its purpose is to isolate representation and interface overhead. It is a
latency-only control and must bit-match Global MWPM on every input.

### 6.4 UF-Shadow MWPM

UF-Shadow MWPM executes the same weighted UF proposal, peeling, confidence
gate, and transaction validation as the treatment. It records no telemetry
inside timed intervals, discards every proposed support/frame change, and
calls Global MWPM on the original syndrome.

Its purpose is to isolate UF and gate overhead from the effect of changing the
backend workload. It is a latency-only control and must bit-match Global MWPM
on every input.

### 6.5 Decoder isolation

All arms receive the same immutable detector corpus. No decoder receives the
actual observable sample, another arm's prediction, or a post-hoc correctness
label. Actual observables are joined only after both accuracy predictions are
immutable.

A crash, malformed packed output, missing prediction, input mutation, or
invariant violation fails the run. It is never converted into an excluded
shot or an ordinary decoder failure.

## 7. Weighted UF and confidence-gate contract

This section fixes the semantic shape of V1. The remaining literal values are
listed in [Section 21](#21-open-decisions-and-freeze-blockers).

### 7.1 Weighted growth

This is a full-history, offline patch-local decoder; V1 does not claim a
streaming L1 schedule. For each lane, the semantic reference is:

1. Every local detector vertex begins as a singleton component. Its active-
   detector set contains that detector exactly when the original lane syndrome
   bit is one. Component parity is the cardinality of this set modulo two.
2. A component is neutral when its parity is even or it has reached a genuine
   true-boundary node. A non-neutral component is active and grows; a neutral
   component is inactive.
3. For every edge incidence `(e, v)`, maintain a nonnegative half-edge charge
   `g(e, v)`, initially zero. It increases at unit rate exactly while the
   component containing `v` is active, `e` leaves that component, and the
   incidence has not been consumed. An internal edge stops accumulating
   charge. Therefore an edge between two active components closes at rate two,
   one with one active side closes at rate one, and a true-boundary edge or
   unconsumed port has only its local-side charge.
4. At the next exact event time, discover every saturating correction,
   true-boundary, and port event from the same pre-event state and advance all
   charges atomically. Union the connected correction-edge endpoint
   components, combine their active-detector sets by symmetric difference,
   and OR their existing boundary/taint flags. Apply simultaneous boundary
   flags and port taints to those post-union components, then recompute
   parity/activity. No event-type ordering may change this result.
5. Within a simultaneous correction-edge batch, select a deterministic
   spanning forest by `(event_time, weight, normalized endpoints, edge_id)`.
   Ordering chooses representation only; every unselected simultaneous edge
   remains visible to confidence with zero slack.
6. When a true-boundary edge saturates, add its virtual boundary incidence to
   the forest, set the component's boundary flag, and make that component
   neutral. If a later growing odd component reaches it through a correction
   edge, the union inherits the boundary flag and is neutral after the batch.
7. When a guard port saturates, mark the current component tainted and record
   the exact event once. Permanently freeze its local charge at exactly the
   canonical edge weight, mark that port incidence consumed, and exclude it
   from later growth and event scheduling. Do not union through the port and
   do not stop an otherwise active component. For a mixed simultaneous event,
   attach the taint to the post-correction-union component from step 4. Taint
   is permanent and propagates through later unions. The remote syndrome bit
   is unavailable during this transition.
8. Continue until no active component remains. If an active component has no
   future finite correction, boundary, or port event, the lane terminates as
   locally incomplete and its patch transaction aborts without a durable
   commit.
9. Components at termination are the **final components** used by the gate.
   Their membership is therefore no longer circularly defined by an earlier
   neutralization event. Untouched syndrome-free singleton components have an
   empty active-detector set and are ignored; they are neither gate candidates
   nor counted as UF syndrome components.
10. For a final component `C`, its claimed syndrome boundary is the sorted set
    of original active detector IDs in `C`; virtual boundary nodes are
    excluded. Deterministically peel its selected forest in the frozen reverse
    order to realize exactly that boundary.
11. A component forest cannot emit the same canonical edge twice. A duplicate
    is fatal. Valid supports from distinct final components are required to be
    disjoint; only already-validated patch supports are combined by symmetric
    difference.

The exact reference parses every canonical binary64 weight as its exact dyadic
rational value. All event times, half-edge charges, and slacks are exact
rationals. The production engine must use a frozen representation that gives
identical event batches, final components, supports, margins, comparisons, and
defer reasons. Quantizing weights or changing arithmetic creates a new decoder
version.

### 7.2 Component atomicity

The gate acts on final components, not on individual peeled edges. Taint
propagates through every pre-termination union, so the implementation cannot
retain a favorable subtree that existed before a tainted merge. Independent
final components may make independent commit decisions.

Tentative and durable supports are separate immutable values. No tentative
edge becomes visible to the residual syndrome until the containing patch
transaction passes its independent algebra check.

### 7.3 Confidence statistic

V1 uses a weighted isolation margin evaluated after the lane terminates, when
final component membership and every half-edge charge are fixed. For any edge
incident to final component `C`, define:

```text
two local endpoints: slack(e) = weight(e) - g(e, u) - g(e, v)
true boundary/port:   slack(e) = weight(e) - g(e, local)
```

The enumerable competing set contains:

- every nonforest correction edge incident to a vertex in `C`, including an
  internal cycle edge or an edge to a different final component;
- every incident true-boundary edge not selected into `C`'s forest; and
- every guard port incident to a vertex in `C`.

The component confidence is:

```text
margin(C) = min(slack(e) for e in competing_set(C))
```

An empty competing set has `margin=+infinity`. A saturated port, an earlier
port contact, or a simultaneous nonforest alternative has margin zero and
cannot be made confident by a tie-break order. Negative exact slack is a fatal
invariant violation.

Finite margins are serialized canonically as signed integer `mantissa` and
integer binary `exponent` fields, representing `mantissa * 2**exponent`.
Positive infinity is the literal string `"infinity"` and occupies a separately
named histogram bin; JSON `Infinity`/`NaN` values are forbidden. The frozen
binary64-hex threshold is parsed as its exact dyadic rational before comparison.

The frozen protocol supplies one threshold `tau` as a binary64 hex literal.
The comparison is strict:

```text
commit candidate only if margin(C) > tau
```

Equality defers. V1 freezes `tau=0x0.0p+0`: every strictly positive margin may
commit, while a tie or zero margin defers. This deliberately permissive first
characterization value was selected before outcome sampling so the experiment
can measure the frontend's natural coverage/risk frontier instead of claiming
that zero is an optimized threshold.

### 7.4 Commit eligibility

A final component is eligible to commit only if all of these conditions hold:

- the weighted UF run completely neutralized its local active syndrome;
- the peeled support's independently recomputed boundary is exactly the
  sorted original active-detector set defined in Section 7.1;
- every durable edge is correction-eligible and belongs to the lane;
- the support is square-free and deterministic;
- the component has not saturated or tied a guard port;
- its observable frame is exactly zero in V1;
- `margin > tau` under the frozen arithmetic and comparison rule; and
- all deterministic operation and memory budgets were respected.

A port-tainted component or a component with `margin <= tau` is ordinarily
deferred. Local incompleteness or budget exhaustion aborts the patch as
specified below. Every other failed condition in the list—wrong boundary,
ineligible/duplicate support, nonzero V1 frame, reference disagreement,
algebra mismatch, input mutation, or malformed output—is a fatal invariant
violation, not an ordinary defer.

### 7.5 Deterministic budgets

The protocol must freeze maximum counts for growth events, heap operations,
unions, forest edges, component size, and temporary memory per lane/shot.
Here `component size` means nonvirtual `absorbed_vertex_count`, not defect
count; the temporary-memory counter's allocation classes and units must also
be literal protocol fields.
If either lane exhausts a deterministic budget before termination, the patch
transaction discards all of that patch's tentative proposals and records
`budget-exhaustion-patch-abort`. Because tentative state has not yet been
applied, this transactional rollback restores the exact original patch
syndrome/frame by construction. Budgets cannot depend on observed correctness
or wall-clock time.

The following counter semantics are normative:

```text
growth_event_count
    number of canonical correction-edge, true-boundary, or guard-port
    saturations processed, counting all members of a simultaneous batch
simultaneous_event_batch_count
    number of distinct exact event times whose complete atomic batch commits
union_attempt_count
    number of saturated correction edges presented to DSU in canonical order
successful_union_count
    union attempts whose endpoint roots differed
failed_union_count
    union attempts whose endpoint roots were already equal
heap_push_count
    every event-queue entry insertion, including replacement entries
heap_pop_count
    every event-queue removal, including stale removals
stale_heap_pop_count
    popped entries rejected because their roots, activity, charge, or version
    no longer match current state; this is a subset of heap_pop_count
heap_operation_count
    heap_push_count + heap_pop_count
peel_operation_count
    forest-edge visits made by deterministic reverse peeling
```

Consequently,
`union_attempt_count = successful_union_count + failed_union_count`.
On each lane record, `peak_heap_size` is the maximum queue length after
initialization and after every push or pop, including stale entries.
`peak_live_component_count` is the maximum number of DSU roots with a nonempty
original-defect set, measured after initialization and after each fully
completed atomic batch. A consumed
port incidence contributes at most one growth event. Per-component merge and
event-batch counts follow the final component's complete ancestry; heap
counters are lane/shot counters and are not apportioned to components.

Because heap operations expose optimized-engine policy rather than only UF
mathematics, the protocol freezes the exact queue-entry versioning,
insertion/replacement, invalidation, and stale-pop lifecycle. The slow
reference need not have the same heap counts; golden traces of the frozen
production lifecycle are authoritative for these resource metrics.

Budget enforcement is transactional at the simultaneous-batch boundary. The
engine checks a proposed operation before applying it. If completing the next
atomic batch would exceed any cap, it discards every mutation made while
preparing that batch and exposes the state after the last fully completed
batch. The cap-triggering operation is neither applied nor included in the
committed counters; preparatory operations rolled back with the batch remain
part of measured wall time but not scientific operation counts. The ledger
records the sorted set of every cap that the proposed batch would exceed, with
each limit and rejected next value; its diagnostic primary cap is the
ASCII-lexicographically first frozen cap name. No decoder decision depends on
that diagnostic precedence. A censored snapshot therefore cannot depend on
how far an optimized implementation happened to progress through a
simultaneous batch.

## 8. Residual algebra, transactions, and fallback

For a sorted square-free support `C`, define:

```text
boundary(C) = XOR of every non-boundary endpoint of every edge in C
frame(C)    = XOR of every edge.observable_mask in C

s_residual  = s_original XOR boundary(C_durable)
prediction  = frame(C_durable) XOR GlobalMWPM(s_residual)
```

All XOR operations are over GF(2). Observable masks use little-endian bit
packing, and unused tail bits are always clear.

The implementation independently reconstructs `boundary(C_durable)` and
`frame(C_durable)` from canonical edge IDs after all tentative decisions. It
does not trust an incrementally mutated syndrome or frame. It asserts:

```text
boundary(A symmetric_difference B) = boundary(A) XOR boundary(B)
frame(A symmetric_difference B)    = frame(A) XOR frame(B)
```

### 8.1 Patch transaction

Each patch has two internal basis lanes. The lanes propose independently from
the immutable original shot. Accepted final components from both lanes are
required to have disjoint supports, combined by symmetric difference, and
validated together. Only then is the patch support made durable as one
transaction.

Any disagreement in the independently recomputed aggregate boundary or frame
is fatal; it is never converted into rollback or fallback. The only ordinary
patch-wide transaction aborts in V1 are deterministic budget exhaustion and a
locally incomplete lane. Ordinary confidence/port deferral of one independent
final component does not erase another eligible independent component; a
component itself is never partially committed.

Component telemetry separates the local gate result from the patch
transaction result. A component in a normally terminated lane can pass its
gate and still be non-durable because the sibling lane later causes a patch
abort. Such a component retains `gate_decision=eligible`, while its
`durable_decision=deferred` and `durable_reason` is the patch-abort reason.
Patch-abort reasons take precedence only at this durable transaction layer;
they do not rewrite the component's local gate reason set. This distinction is
required for every routing count and cluster-size stratum.

Because V1 durable support has zero observable frame and never contains a yoke
port, it must not directly toggle a yoke observable. The complete Global MWPM
still receives every residual yoke detector and remains responsible for the
joint 12-observable answer.

### 8.2 Fallback semantics

After all shots in one public input batch have been predecoded, the treatment
calls complete-graph PyMatching `decode_batch` exactly once on the residual
batch. Scalar decoding is a one-shot batch. There is no separate emergency
matcher and no reduced graph with deleted edges. On every shot with no durable
component, the treatment must bit-match Global MWPM exactly.

The ordinary defer/rollback enum is frozen and includes at least:

```text
below-threshold
port-cross-lane
port-yoke
port-tie
local-incomplete-neutralization-patch-abort
budget-exhaustion-patch-abort
```

Compile-time nonpositive/nonfinite weights, ownership ambiguity, algebra
mismatch, impossible peeling, unsupported compiled topology, duplicate or
overlapping component support, nonzero candidate frame,
reference/optimized disagreement, or corrupted artifacts are fatal errors and
are not hidden inside this enum.

## 9. Sampling and pairing

### 9.1 Two fresh stages

| Stage | Fixed shots | Purpose | May enter final characterization? |
| --- | ---: | --- | --- |
| Engineering shakeout | 1,000 | Correctness, schema, replay, storage, and timing-plumbing validation | No |
| Characterization | 10,000 | Paired accuracy, workload, confidence, and timing workload | Yes |

The stages use disjoint literal 256-bit seed roots and distinct experiment run
IDs. Shakeout shots are never appended to, promoted into, copied into, or
pooled with characterization shots.

### 9.2 Paired physical shots

For shot `i`, the sampler produces `(D_i, O_i)` exactly once. Byte-identical
`D_i` is decoded by Global MWPM and Confidence-Gated UF–MWPM. Neither decoder
can access `O_i`. After both packed predictions are immutable, the analyzer
compares them with `O_i`.

The characterization denominator is exactly 10,000 complete physical shots,
not 60,000 patch observations, 120,000 basis-lane observations, or a count of
decoder calls.

### 9.3 Fixed-N execution

Accuracy collection uses exactly 32 worker processes and one native numerical
thread per process. Worker `w` owns the half-open range:

```text
start_w = floor(N * w / 32)
stop_w  = floor(N * (w + 1) / 32)
```

Thus the 1,000-shot stage has eight 32-shot ranges and twenty-four 31-shot
ranges. The 10,000-shot stage has sixteen 313-shot ranges and sixteen 312-shot
ranges. The formula, worker ordering, seed derivation, and batch subdivision
are frozen in the protocol.

`MAX_ERRORS` remains unset. There is no error-count, discordance-count,
confidence, elapsed-time, significance, or favorable-trend stopping rule. An
incomplete run remains incomplete until only its exact missing deterministic
ranges are resumed and validated.

No other simulation campaign may run concurrently if the configured worker
total would exceed 32. Before any NumPy or PyMatching import, the parent and
workers set:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
BLIS_NUM_THREADS=1
```

### 9.4 Seed separation

The frozen protocol contains unrelated literal roots for:

- the 1,000-shot sampler;
- the 10,000-shot sampler;
- latency pair ordering;
- analysis/bootstrap resampling; and
- deterministic replay selection.

Seeds are derived with a named cryptographic hash over the experiment ID,
stage, cell ID, worker/range identity, and purpose label. No seed is reused
from a ProMatch corpus or chosen after viewing an outcome.

## 10. Ordered launch gates

### 10.1 Before the engineering shakeout

The following must pass before the first 1,000 shots:

1. The confidence threshold, arithmetic, budgets, seed roots, and all fields
   in Section 21 are resolved in a draft protocol.
2. The graph, projection, UF reference, optimized engine, adapters, collector,
   analyzer, timing harness, and replay code have complete tests.
3. The selected physical cell regenerates its authenticated circuit, DEM,
   detector count, observable count, full graph, and projection.
4. An exhaustive small-graph differential suite and selected real-DEM fault
   tests pass.
5. A deterministic 32-shot integration smoke under `$TMPDIR` passes twice
   bit-for-bit.
6. A disjoint 100-shot storage/runtime probe under `$TMPDIR` establishes that
   all required artifacts fit and that no unbounded trace is retained.
7. A one-worker scratch capacity probe executes one exact 313-shot logical
   range through normalization and serialization. Its canonical normalized
   metric tree, and the largest authenticated single-shot tree from the
   100-shot probe scaled to 313 shots, must each fit below half the configured
   per-range ceiling.
8. The analyzer accepts a synthetic complete corpus and rejects missing,
   duplicated, overlapping, tampered, or inconsistent records.

The smoke and probe are engineering inputs only. They are not copied into the
shakeout or characterization.

### 10.2 Engineering-shakeout gate

Proceed to the 10,000-shot characterization only if the frozen 1,000-shot run
satisfies every operational condition below:

- exactly 1,000 nonoverlapping shot IDs and no missing ranges;
- one valid Global-MWPM and treatment prediction per shot;
- outside timed intervals, Adapter-Control MWPM and UF-Shadow MWPM bit-match
  Global MWPM on all 1,000 indexed shakeout corpus rows, with counts and
  digests persisted;
- the treatment's ordinary, telemetry, and timing callables agree on all 1,000
  indexed shakeout workload keys;
- the four paired-outcome cells sum to exactly 1,000;
- prediction-agreement, activation, gate, fallback, and telemetry totals
  reconcile exactly;
- detector and observable corpus digests are unchanged after decoding;
- every durable UF support passes independent boundary/frame replay;
- every shot with no durable component reproduces the original backend input;
- every retained example replays bit-exactly in a fresh process;
- all deterministic shard and manifest hashes verify;
- latency plumbing emits complete positive arrays with exact pair-order
  balance; and
- no fatal invariant, reference disagreement, or unexpected resource-cap
  violation occurred.

Accuracy direction, latency direction, detector reduction, activation rate,
failure count, and statistical significance are explicitly not launch gates.

Any source, threshold, budget, arithmetic, configuration, artifact-schema, or
protocol change after the shakeout requires a new decoder/protocol version,
new seed roots, and a fresh 1,000-shot shakeout. There is no patch-and-promote
path.

### 10.3 Post-characterization diagnostic-volume flags

After all 10,000 shots are complete, the analyzer emits three descriptive
tags:

```text
baseline_failures_lt_200
discordant_pairs_lt_100
durable_commit_shots_lt_500
```

These are diagnostic-volume annotations for tables and replay strata, not
tests of statistical adequacy. They do not stop, extend, exclude, or rerun
sampling. Sampling uncertainty is represented by the reported intervals; a
tag cannot be interpreted as zero, equivalence, or preservation.

## 11. Accuracy endpoints and statistics

### 11.1 Failure definition

A decoder fails a physical shot when any bit of its 12-observable packed
prediction differs from the sampled actual-observable frame. Per-observable
and observable-Hamming-weight summaries are secondary; the physical-shot
any-observable result is primary.

For each shot, record one cell of this table:

| Symbol | Global MWPM | Confidence-Gated UF–MWPM | Meaning |
| --- | --- | --- | --- |
| `a` | correct | correct | both correct |
| `b` | correct | wrong | regression |
| `c` | wrong | correct | recovery |
| `d` | wrong | wrong | both wrong |

For `N=10,000`:

```text
p_global    = (c + d) / N
p_treatment = (b + d) / N
delta       = p_treatment - p_global = (b - c) / N
discordance = (b + c) / N
agreement   = count(prediction_global == prediction_treatment) / N
```

Prediction agreement is reported separately from correctness. Two decoders
can disagree while both are wrong, or agree on the same wrong frame.

### 11.2 Required uncertainty summaries

The analyzer reports:

- raw `a`, `b`, `c`, `d`, all numerators, and all denominators;
- each marginal failure rate with a two-sided 95% exact Clopper–Pearson
  interval;
- paired `delta` with a two-sided 95% Tango efficient-score interval;
- discordance and exact packed-prediction agreement.

Both intervals use two-sided `alpha=0.05`. Each equal-tail Clopper–Pearson
endpoint calls the maintained one-sided routine with `alpha=0.025`. Each Tango
upper calculation uses `alpha=0.025`, root tolerance `1e-12`, and at most 200
iterations; the lower endpoint swaps `b` and `c` and negates the corresponding
upper endpoint. With zero discordance, the one-sided Tango boundary is
`z**2 / (N + z**2)`, where `z` is the standard-normal 97.5th percentile. The
protocol freezes the exact source hash of the reused routine, and independent
golden/zero-boundary tests prevent solver drift.

No hypothesis-test p-value is part of V1. No multiplicity claim is made for
secondary tables.

### 11.3 Required breakdowns

Secondary accuracy tables include:

- failure and paired outcomes by observable ID;
- outcomes conditional on any UF activation and on any durable commit;
- outcomes by number of committed patches/components;
- outcomes by cluster-summary completeness and predeclared shot-level
  largest-final-cluster, largest-committed-cluster, and committed-defect bins;
- outcomes by defer/rollback reason;
- outcomes by original and residual detector-Hamming-weight bins; and
- treatment disagreement by yoke-observable and patch-observable masks.

Every conditional table gives its complete-shot denominator. It cannot be
used to replace the unconditional primary result.

## 12. Workload and confidence diagnostics

### 12.1 Primary workload estimand

For shot `i`, let `H_i` be the original global detector-event count and `R_i`
the residual count presented to complete Global MWPM. The primary descriptive
workload ratio is:

```text
workload_ratio = sum_i R_i / sum_i H_i
workload_mean_difference = mean_i(R_i - H_i)
```

The report includes both totals, both per-shot means, the signed paired mean
difference above, the ratio, and the complete original/residual joint
histogram. If `sum_i H_i=0`, the ratio and its interval are serialized as
`null` with status `not-estimable`; other endpoints remain valid. Workload
reduction is not treated as latency reduction.

Uncertainty uses the exact complete-shot `(H_i, R_i, L_i, K_i)` joint
histogram, where `L_i` and `K_i` are defined in Section 12.2, and 10,000
fixed-seed multinomial bootstrap replicates, followed by empirical type-7 2.5%
and 97.5% quantiles.
Direct row resampling is not an alternative implementation because it would
not be bit-identical under the same seed. Patch or basis rows are not
independent bootstrap units. A bootstrap replicate with zero original-event
denominator contributes `null`; the report gives the number of estimable
replicates and emits no ratio interval if fewer than all 10,000 are estimable.
Canonical JSON never contains numeric `NaN` or infinity.

### 12.2 Frontend coverage and cluster-size endpoints

For every completed final component `C`, define its primary size as:

```text
cluster_defect_count(C) =
    number of original nonzero detector events in C's claimed syndrome boundary
```

This is the normative meaning of **cluster size** in V1. It measures how much
original syndrome the component owns. Absorbed graph vertices, forest edges,
and correction edges are reported separately because they measure memory,
growth, and correction complexity rather than syndrome size.

Let `K_i` be the total defect count of durable components on shot `i`. Let
`L_i` be the number of original active detector IDs owned by one of the 12
full-history patch/basis lanes. `L_i` excludes both yoke detectors and any
detector that the authenticated projection does not assign to exactly one
lane. The primary global and secondary lane-owned coverage estimands are:

```text
frontend_coverage   = sum_i K_i / sum_i H_i
lane_owned_coverage = sum_i K_i / sum_i L_i
```

Calling the second denominator *eligible defects* is forbidden: eligibility
is a property of a component after growth and gating, whereas `L_i` is an
input-ownership count. V1 durable boundaries contain only original active
detector IDs and cannot add a residual event. Every completed shot must satisfy:

```text
0 <= K_i <= L_i <= H_i
R_i = H_i - K_i
frontend_coverage = 1 - workload_ratio
```

A violation is fatal. Coverage is reported because it is the clearest routing
interpretation of the workload result, not because it is statistically
independent of that result. A zero denominator produces `null` with status
`not-estimable`. Routing rates are fixed as:

```text
shot_commit_rate = shots with K_i > 0 / all physical shots
component_commit_rate = durable committed components / completed components
all_patch_commit_rate = patch transactions with a durable component / (6N)
active_patch_commit_rate = patch transactions with a durable component
                           / patches with at least one original lane-owned defect
```

Every report includes the numerator and denominator. None of these rates is
substituted for the global coverage estimand. Both coverage intervals are
computed from the same frozen `(H_i, R_i, L_i, K_i)` multinomial bootstrap
replicates as the workload interval, preserving their complete-shot
correlation.

#### 12.2.1 Per-component record and metric formulas

Every component row first records its identity and state:

```text
stage_id, corpus_digest, global_shot_id, patch_id, check_basis
state_kind = completed | censored
deterministic component_id
sorted absorbed_detector_ids
absorbed_detector_membership_sha256
shot_cluster_summary_complete
```

`absorbed_detector_membership_sha256` is SHA-256 of the canonical sorted array
of nonvirtual absorbed detector IDs. `component_id` is SHA-256 over the
canonical schema tag and tuple `(stage_id, corpus_digest, global_shot_id,
patch_id, check_basis, state_kind, absorbed_detector_membership_sha256)`.
It is invariant to worker count, edge iteration, DSU root choice, and
serialization order. Replay independently regenerates and checks the
membership digest.

Every completed final component then records:

```text
sorted original_defect_detector_ids and cluster_defect_count
absorbed_vertex_count
sorted forest_edge_ids, forest_edge_count, and exact forest_weight
sorted peeled_support_edge_ids, peeled_support_edge_count,
    and exact peeled_support_weight
defect_minimum_round, defect_maximum_round, defect_time_span_rounds
absorbed_minimum_round, absorbed_maximum_round, absorbed_time_span_rounds
defect_minimum/maximum_local_x2, defect_minimum/maximum_y2,
    defect_span_local_x2, defect_span_y2
absorbed_minimum/maximum_local_x2, absorbed_minimum/maximum_y2,
    absorbed_span_local_x2, absorbed_span_y2
last_membership_event_time and maximum_incident_half_edge_charge
merge_count and simultaneous_event_batch_count
boundary_reached
port_tainted, sorted port_kind_set, and saturated_port_count
exact confidence_margin
gate_decision = eligible | deferred
sorted gate_reason_set and exclusive primary_gate_reason
durable_decision = committed | deferred
exclusive durable_reason
```

Absorbed vertices are all lane detector vertices in the final component,
including vertices whose original syndrome bit was zero and excluding virtual
boundary sentinels. Forest and support counts are over unique canonical edge
IDs; their weights are exact sums over those square-free sets. Temporal span
is `maximum_round - minimum_round`, so a one-round component has span zero.

Geometry comes only from the authenticated normalized layout. For `d=7`,
`local_x = x - patch_id * (d + 1)`. The layout must prove that `local_x` and
`y` are half-integral; rows store the unambiguous integers
`local_x2 = 2*local_x` and `y2 = 2*y`. Each spatial span is integer
`maximum_coordinate2 - minimum_coordinate2`, so a singleton has span zero and
the displayed lattice-coordinate span is the stored value divided by two.

`last_membership_event_time` is the latest exact simultaneous correction
batch in the component's ancestry that changed its nonvirtual membership; it
is exact zero for a never-merged singleton. `merge_count` is the number of
successful DSU root unions in that ancestry.
`simultaneous_event_batch_count` counts distinct committed atomic batch IDs in
the ancestry that contained a correction, boundary, or port event.
`saturated_port_count` counts unique consumed canonical port incidences; a
port incidence can contribute at most once.
`maximum_incident_half_edge_charge` is read at the lane's terminal snapshot
over every correction, true-boundary, and guard-port incidence whose local
endpoint is an absorbed nonvirtual member; both local incidences of an
internal edge participate, while a virtual boundary endpoint does not. The
censored version uses the last-complete-batch snapshot instead. Weights, event
times, charges, and confidence use the exact rational representation from
Section 7.

Gate reasons are multi-label because one component may encounter several port
kinds and also fail the threshold. The exclusive `primary_gate_reason` uses
this frozen precedence:

```text
port-tie > port-yoke > port-cross-lane > below-threshold > eligible
```

`eligible` means the reason set is empty. Port, boundary, and exact reason-set
attributes remain available for overlapping descriptive views, but only the
exclusive primary-reason field may be summed as a partition. At the durable
layer, `budget-exhaustion-patch-abort` or
`local-incomplete-neutralization-patch-abort` overrides the local primary
reason. Otherwise an eligible component has `durable_decision=committed` and
`durable_reason=committed`; a locally deferred component retains its exclusive
primary reason as `durable_reason`. A completed sibling-lane component in an
aborted patch is therefore completed but non-durable, regardless of its local
gate result.

A lane stopped by budget exhaustion or local incompleteness has not produced
the terminal components defined in Section 7. Its censored snapshot is the
state after the last fully completed atomic event batch; a budget-triggering
operation or partial batch is not applied. Record only current components
with a nonempty original-defect set, using:

```text
censored=true, censor_reason
budget_exceeded_set with limit/rejected-next-value pairs, or null
partial_cluster_defect_lower_bound
current absorbed-vertex, forest, geometry, ancestry, boundary, port,
    and charge metrics; lane_telemetry_id
peeled support, support weight, confidence margin, gate decision/reasons,
    and durable decision/reason = null
```

For `local-incomplete-neutralization-patch-abort`, `budget_exceeded_set=null`.
For `budget-exhaustion-patch-abort`, it is nonempty. Operation counters and
peaks are stored once in the referenced lane-telemetry record, not duplicated
on each partial-component row.

The current defect count is a lower bound on the size of a hypothetical final
component because later batches could merge it. Censored rows never enter a
completed-component histogram or quantile. Completed components from a sibling
lane remain valid completed-component rows, but the patch abort makes all of
that patch's components non-durable as specified above.

#### 12.2.2 Per-shot cluster summary

Every physical shot records:

```text
cluster_summary_complete
original_detector_count, lane_owned_detector_count,
    residual_detector_count, committed_defect_count
completed_final_component_count
gate_eligible_component_count
committed_component_count, durable_deferred_component_count
port_tainted_component_count, boundary_component_count
censored_lane_count, censored_partial_component_count
maximum_final_component_defect_count
maximum_committed_component_defect_count
maximum_absorbed_vertex_count
maximum_defect_time_span_rounds
maximum_absorbed_time_span_rounds
maximum_partial_component_defect_lower_bound
active_patch_count, committed_patch_count, aborted_patch_count
growth_event_count, simultaneous_event_batch_count
union_attempt_count, successful_union_count, failed_union_count
heap_push_count, heap_pop_count, stale_heap_pop_count,
    heap_operation_count, peel_operation_count
maximum_lane_peak_heap_size, sum_lane_peak_heap_size
maximum_lane_peak_live_component_count, sum_lane_peak_live_component_count
exact sparse histogram of completed component defect counts
```

`cluster_summary_complete=false` if any lane with a nonempty original-defect
set ends in either censor state. On an incomplete-summary shot, every maximum
over completed final components is `null`, including the committed and
geometry maxima; the separately named maximum partial lower bound remains
available. On a complete-summary shot with no completed syndrome component,
the final-component maxima are zero. If completed components exist but none
commits, `maximum_committed_component_defect_count=0`. Untouched syndrome-free
singletons remain excluded. If no censored partial component exists,
`maximum_partial_component_defect_lower_bound=0`; therefore the corresponding
batch maximum is also zero when every constituent shot has none. Counts and
additive totals that are still exactly known remain populated even when the
summary is incomplete.

All event, union, heap, and peel counters in the shot row are sums over the 12
lane-telemetry records. Shot peak fields do not pretend that lanes executed
concurrently: they report both the maximum lane peak and the sum of lane peaks
under the exact names above. A lane record is stored once even when it has
multiple censored partial components.

The per-shot shard contains exactly 12 lane-telemetry records keyed by
`(global_shot_id, patch_id, check_basis)`. Each records
`lane_status=empty|completed|censored`, all Section 7.5 counters and peaks, its
last-complete batch ID, censor reason/exceeded-budget set when applicable, and
the ordered component-ledger references it produced. This nested lane table is
the target of each component row's `lane_telemetry_id`.

#### 12.2.3 Required cluster-size views and uncertainty

The report contains both of these noninterchangeable views:

1. **Component-weighted:** the exact sparse histogram of
   `cluster_defect_count` over completed final components. Normally terminated
   components from a shot with a censored sibling lane remain in this view,
   and the report gives their count separately.
2. **Shot-weighted tail:** the distribution of
   `maximum_final_component_defect_count` conditional on
   `cluster_summary_complete=true`.

For both, report raw counts and denominators. Also report
`complete_cluster_summary_shots`, `censored_cluster_summary_shots`, and the
shot censor rate over all 10,000 physical shots. The shot-weighted view reports
median, p90, p95, p99, and maximum using empirical type-7 quantiles. A zero
denominator is `null/not-estimable`, never an observed zero cluster size.
The all-completed component-weighted view reports the same five summaries over
component rows; these do not replace its exact integer histogram.

The component artifact includes an exact sparse joint histogram of defect
count, `gate_decision`, exact gate-reason-set code, exclusive primary gate
reason, `durable_decision`, exclusive durable reason, boundary flag, and exact
port-kind-set code. The report renders at least all-completed, committed,
primary-below-threshold, patch-aborted, any-port-tainted, and true-boundary views,
pooled and split by patch/basis. Exact reason-set and exclusive reason views
form partitions; marginal boundary and individual port-kind views may overlap
and must never be summed. Censored partial components have a separate
lower-bound histogram by censor reason.

The characterization report must include these cluster-shape displays, each
with its raw denominator printed in the caption or adjacent table:

- an exact integer component-size count/PMF plot, with committed and durable-
  deferred overlays;
- component-size and shot-maximum complementary CDFs, without extrapolating
  beyond the observed maximum;
- exact sparse two-dimensional tables, plus frozen-bin heatmaps, of defect
  count versus absorbed-vertex count, defect time span, and each doubled-
  coordinate spatial span;
- durable commit fraction by every observed exact component size, with
  numerator/denominator shown and predeclared display bins used only for
  readability; and
- a separate censored lower-bound size display by censor reason.

Completed and censored curves cannot share an unlabeled denominator. Patch and
basis small multiples use identical axes and frozen bins so visual differences
cannot be created by panel-specific scaling.

Integer sizes are retained as exact sparse histograms. The frozen protocol may
also define display bins for readability, but those bins cannot replace the
exact artifact or be changed after sampling. The p50, p90, p95, and p99
shot-weighted and all-completed component-weighted statistics receive
two-sided 95% percentile intervals; each predeclared component-size display-
bin proportion receives a pointwise two-sided 95% percentile interval with no
simultaneous-coverage claim.

These intervals use 10,000 fixed-seed complete-shot bootstrap replicates. Each
replicate draws exactly `N` physical-shot indices with replacement using the
frozen RNG implementation and carries each selected shot's whole component
row group, sparse histogram, and completeness flag. A shot quantile uses only
selected copies with `cluster_summary_complete=true`. A component quantile and
component-bin proportion use all selected completed-component rows, including
completed rows from incomplete-summary shots; the bin proportion divides its
selected bin count by that completed-component total. A shot-quantile
replicate with no selected complete summary, or a component endpoint replicate
with no selected completed component, is `null`. For each endpoint, the report
records the estimable replicate count and emits no interval unless all 10,000
replicates are estimable. Direct component resampling and multinomial
component counts are not alternative implementations.

Accuracy may be stratified by complete-shot quantities such as largest
committed cluster, total committed defects, or committed-component count. A
complete-shot logical failure is not assigned as the correctness label of an
individual cluster. Per-cluster causal attribution would require a separately
specified counterfactual ablation experiment.

### 12.3 Unconditional routing telemetry

The characterization records, over all shots:

- original and residual detector counts globally and by patch/basis lane;
- original and residual terminal and yoke detector counts;
- shots and lanes with UF work, neutralization, eligible components, gate
  passes, durable commits, deferrals, patch rollbacks, and all-fallback;
- every exact component and per-shot cluster metric from Section 12.2;
- tentative and durable support size and total canonical weight;
- growth events, simultaneous batches, union attempts/successes/failures,
  heap pushes/pops/stale pops, forest edges, peel operations, and peak live
  components;
- port incidence, contact, tie, and kind, plus the remote-active bit joined
  post-decision for telemetry only;
- confidence margin, threshold equality, and distance from threshold;
- deterministic budget consumption, rejected next operation, and exhaustion;
  and
- residual backend calls and batch/problem sizes.

Activation, acceptance, rollback, fallback, censor, and completion rates
always include exact numerators and denominators. Conditional-on-activation
workload is secondary to the unconditional totals.

### 12.4 Confidence diagnostics

The frozen protocol defines exact rational confidence histogram edges before
sampling and retains the canonical rational margin for bounded replay cases.
For complete-shot tables, the shot-level confidence is the minimum margin
among its durable components; no-commit shots form a separate category. The
analyzer reports:

- component and shot acceptance by confidence bin;
- downstream regression/recovery rates by accepted-confidence bin;
- risk/coverage curves formed by progressively requiring larger margins;
- counts exactly equal to the threshold; and
- confidence distributions split by port kind, boundary use, component size,
  and original workload.

These are post-hoc descriptive diagnostics. Actual observables never influence
the deployed gate, and V1 cannot be used to select a new threshold and then
claim that threshold was validated on the same shots.

### 12.5 Cluster/workload association with latency

Cluster telemetry is never collected or serialized inside a timed interval.
Before timing, the untimed production path computes and authenticates the
Section 12.2 summary for every corpus row. Its immutable `workload_key` is the
tuple `(characterization_corpus_digest, global_shot_id)`. `corpus_index` is the
zero-based physical row position and maps bijectively to `global_shot_id` in
the authenticated corpus index. Byte-identical detector vectors at different
shot IDs remain distinct workloads.

Every timed call records or references, outside the timed interval:

```text
timing_call_id, restart, batch_size, pair, side, block, call_index
ordered list of (corpus_index, workload_key)
detector_batch_digest
precomputed_workload_summary_digest
```

The ordered list is stored in a canonical shared schedule table when reused;
the call row authenticates its exact range and digest. It includes every
cyclic wrap and permits exact reconstruction of the input order. The detector
digest distinguishes original and residual backend arrays. The precomputed
summary digest covers the ordered per-shot metric records. A join with a
missing precomputed-table key, duplicate precomputed-table row, unexpected
within-call duplicate/reordering relative to the frozen schedule, or digest
mismatch is fatal. Reuse of a valid workload key across different timed calls
is expected and retained.

The analysis unit for workload/latency association is one timed call. Repeated
calls of the same workload are retained, and every table is computed within a
single `(batch_size, pair, side)` context; pair contexts and sides are never
pooled. For batch 1, mandatory descriptive tables use these covariates
separately:

```text
cluster summary status = complete | censored
exact maximum_final_component_defect_count, complete shots only
committed_defect_count
growth_event_count
successful_union_count
heap_operation_count
peel_operation_count
residual_detector_count
```

The protocol freezes display bins for non-cluster count covariates. For each
status, exact cluster size, or predeclared count bin, report timed-call count,
unique workload-key count, and empirical type-7 median, p95, and p99
nanoseconds. Censored batch-1 calls form their own status group and never enter
an exact final-cluster-size group. These tables have no causal interpretation
and no inferential interval in V1.

For batch 64 and 1,024, persist these aligned batch covariates for descriptive
inspection without making them additional latency endpoints:

```text
batch_cluster_summary_complete  = all shot summaries are complete
batch_max_final_cluster_defects = max over shots, or null if any is censored
batch_max_partial_defect_lower_bound = max partial lower bound
batch_committed_defects         = sum over shots in the batch
batch_growth_events             = sum over shots in the batch
batch_successful_unions         = sum over shots in the batch
batch_heap_operations           = sum over shots in the batch
batch_peel_operations           = sum over shots in the batch
batch_residual_detector_events  = sum over shots in the batch
```

These are descriptive workload/latency associations, not causal latency
effects. Actual observables remain absent from timing workers.

## 13. Latency experiment

### 13.1 Purpose and deliberate scope

Latency is measured alongside accuracy, but not from 32-worker collection
wall-clock time. The initial goal is a quick software estimate, so the timing
suite reuses the immutable 10,000-shot characterization detector corpus as its
fixed natural-noise workload across process restarts. It does not sample an
additional 100,000 physical shots.

This is a deliberate non-claim-bearing departure from the repository's larger
claim-oriented ProMatch timing design, which uses a distinct natural-noise
corpus per restart. Repeated use of one workload means the timing analysis
characterizes software variation on that fixed corpus; it does not add
independent physical-noise evidence. No timing call enters the accuracy
denominator. Timing workers receive detector inputs and authenticated residual
inputs only; they do not load the actual-observable array.

### 13.2 Timed intervals

Two intervals are measured directly with `time.perf_counter_ns()`:

```text
total path:
    public decoder adapter entry -> packed prediction return

backend only:
    PyMatching invocation -> PyMatching return
```

The total interval includes validation performed by the production adapter,
packing/conversion, UF growth, confidence decisions, residual construction,
the residual matcher, and frame composition as applicable to the variant.
`T_total` is measured directly; it is not reconstructed by summing component
timers.

Circuit/DEM generation, graph compilation, corpus loading, input generation,
telemetry collection/serialization, logging, file I/O, provenance collection,
and analysis remain outside every timed interval. Garbage collection follows
one frozen policy, initially disabled during warmup and timing.

The backend-only pair uses aligned original and treatment-residual packed
syndromes generated once by the untimed production path. Their support,
boundary, frame, and corpus digests are validated before timing and remain
immutable throughout the suite.

Latency materialization first authenticates the characterization collection's
complete outside-timer control-equality ledgers. It then invokes the complete
matcher exactly once over the full 10,000-row original corpus and exactly once
over the full 10,000-row persisted-residual corpus. Those predictions must
bit-match the recorded Global-MWPM and treatment predictions respectively.
Canonical corpus, prediction, equality, and control-ledger digests are bound
into the detector/residual-only materialization provenance. This preflight
does not read, copy, hash, or name the actual-observable corpus.

### 13.3 Direct paired comparisons

| Pair | Numerator / denominator | Interpretation |
| --- | --- | --- |
| Net total | Treatment / Global MWPM | Observed end-to-end ratio |
| Adapter cost | Adapter-Control / Global MWPM | Observed adapter-overhead ratio |
| UF/gate cost | UF-Shadow / Adapter-Control | Observed UF/gate-overhead ratio |
| Residual application | Treatment / UF-Shadow | Observed treatment-versus-shadow ratio |
| Backend relief | Residual backend / original backend | Observed matcher-only ratio |

Each pair uses identical inputs and exactly balanced randomized `AB`/`BA`
blocks. The two sides of a pair run serially. Pair order and within-pair order
come from frozen timing seed roots and are recorded.

### 13.4 Counts and batch sizes

This initial, non-claim-bearing timing characterization uses a deliberately
bounded batch-specific schedule. Restarts remain serialized and each restart
uses balanced paired blocks:

| Batch size | Role | Restarts | Paired blocks/restart | Warmup calls/variant/restart | Timed calls/side/block |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Primary latency | 10 | 20 | 50 | 10 |
| 64 | Secondary throughput | 5 | 4 | 5 | 2 |
| 1,024 | Secondary throughput | 3 | 2 | 1 | 1 |

This schedule represents about 107,000 timed variant-shot equivalents before
warmup. It is sized to provide an actionable first latency estimate without
turning the initial experiment into a multi-week UF workload. A later
claim-bearing latency protocol may increase counts in a separately versioned
campaign; it must not silently extend or pool with this one.

The fixed 10,000-shot corpus is traversed in deterministic cyclic order.
Batches never straddle an undefined tail: the protocol freezes the wrap rule,
packed shape, and starting offset for every restart/pair/block.

Within each `(restart, batch_size)`, warm every named timed variant exactly
once in a frozen deterministic order before any timed pair, including the
separate `backend_original` and `backend_residual` workload variants even when
they share a matcher object. Then execute the five pairs in one frozen
deterministic pair order; there is no pair-specific re-warmup. Within a paired
block, numerator call `j` and denominator call `j` receive the same corpus
batch index. Starting offsets are derived by the frozen cryptographic seed
function from `(restart, batch_size, pair, block)` and are persisted in the
restart ledger.

Batch 1 is the only tail-latency endpoint. Batch 64 and 1,024 are throughput
diagnostics; their tails are per-batch, not single-shot tail latency.

### 13.5 Host controls

Every restart runs in a fresh spawned process with restart concurrency one,
one native numerical thread, a frozen CPU-affinity/NUMA policy, and no
concurrent simulation. Record:

- CPU model, topology, microcode, affinity, and NUMA placement;
- OS/kernel, Python, Stim, PyMatching, NumPy, and dependency versions;
- governor, turbo/frequency policy when observable, and host-load snapshot;
- all thread environment variables;
- graph, adapter, decoder, corpus, and schedule digests;
- GC policy and clock identity; and
- all raw call and block durations.

After rebuilding all six variants and authenticating the full-corpus
materialization attestation, each fresh restart performs one untimed equality
check on a single deterministic corpus row. The row is derived from the frozen
schedule seed, restart index, and batch size. All six variants run once on that
row: Adapter-Control, UF-Shadow, and original-backend predictions must equal
Global MWPM, while the residual-backend prediction must equal treatment. No
restart repeats a full-corpus UF/control decode.

Unexpected process overlap, affinity drift, corpus mutation, or host-policy
mismatch invalidates the affected restart. Its ledger is never installed. A
replacement uses the same frozen restart index, corpus digest, seed, pair
schedule, and call offsets in a new process; individual calls are never
cherry-picked or replaced.

### 13.6 Latency statistics

For every `(batch_size, pair, side)` separately, report raw median, p90, p95,
and p99 nanoseconds per call. Never pool a callable's durations across pair
contexts. Aggregate throughput for that same array is:

```text
throughput_shots_per_second =
    batch_size * number_of_calls * 1e9 / sum(call_duration_ns)
```

Do not average reciprocal call durations. For each pair, report:

```text
geometric_block_ratio = geometric mean of paired block-total ratios
p99_ratio             = pooled empirical type-7 p99(numerator calls)
                        / pooled empirical type-7 p99(denominator calls)
```

Use 10,000 fixed-seed hierarchical bootstrap replicates: resample process
restarts, then paired blocks within each selected restart, retaining all calls
inside a block as a cluster. Report two-sided 95% percentile intervals. Do not
bootstrap calls as if they were independent. Every interval is conditional on
the immutable characterization corpus, recorded host, and frozen execution
policy; the analysis records
`inference_scope=fixed_characterization_corpus_recorded_host`.

The primary latency result is the batch-1 Treatment/Global-MWPM geometric
block ratio, accompanied by both variants' raw distributions and the p99
ratio. A p99 may be displayed for batch 64 or 1,024 only as a per-batch
throughput diagnostic; only batch-1 p99 is called tail latency. Backend-only
relief is never labeled end-to-end speedup.

### 13.7 Latency validity gates

The timing suite is valid only if:

- every protocol-specified restart, pair, block, call, and shape is present;
- every block schedule is exactly balanced and every duration is positive;
- outside timed intervals, Adapter-Control and UF-Shadow bit-match Global MWPM
  on all 1,000 indexed shakeout rows and all 10,000 indexed characterization
  rows, with counts and digests persisted;
- outside timed intervals, the treatment timing callable matches the untimed
  production treatment on all 10,000 indexed characterization workload keys;
- corpus and graph digests are constant and inputs remain immutable;
- the timer excludes all forbidden work; and
- all host/runtime policies match the frozen protocol.

There is no numerical speedup gate. In particular, the existing ProMatch
thresholds `0.90`, `0.95`, and `1.05` belong to another frozen experiment and
must not be reused here.

## 14. Telemetry and deterministic replay

### 14.1 Per-shot ledger

The experimental collector retains a deterministic per-shot ledger containing
at least:

```text
stage, cell_id, worker_id, range, global_shot_id, derived_seed identity
packed detector and actual-observable digests
Global-MWPM and treatment packed predictions
paired correctness category and prediction agreement
original and residual detector counts
durable support IDs and durable frame
per-patch transaction status
per-shot cluster summary and exact sparse component-size histogram
exact 12-row lane-telemetry table and lane-to-component references
component-ledger range/count/digest
aggregated component, port, confidence, operation, and budget telemetry
result and trace digests
```

The standard decoder API used outside the experiment does not retain unbounded
per-shot traces. Detailed records are available only through an explicit
experimental telemetry path.

### 14.2 Bounded detailed cases

Retain at most 100 cases per category. For ordinary categories, select the
lowest SHA-256 of `replay_selection_root || canonical_selection_payload`:

```text
regression
recovery
prediction-disagreement
port-yoke-defer
port-cross-lane-defer
threshold-tie
local-incomplete-neutralization-patch-abort
budget-exhaustion-patch-abort
boundary-using-commit
largest-final-component
largest-committed-component
largest-censored-partial-lower-bound
highest-heap-operation-count
```

The `largest-final-component` and `largest-committed-component` categories
consider only complete-summary shots.
The `largest-censored-partial-lower-bound` category considers only censored
shots and sorts by `maximum_partial_component_defect_lower_bound`. These two,
`largest-committed-component`, and `highest-heap-operation-count` sort by
their named metric descending and break ties by that rooted SHA-256 ascending;
the last category's named metric is the per-shot `heap_operation_count`. The
metric definition, canonical payload, root, category cap, and tie rule are
frozen before sampling.

A detailed case stores the packed original syndrome and actual observables,
both predictions, residual syndrome, tentative and durable support IDs,
component forest/peel trace, confidence values, first port contact, patch
transaction result, every Section 12.2 component metric, and all relevant
digests. Actual observables are attached only after decoder/gate outputs are
immutable.

Every retained case must replay bit-exactly in a fresh process from either the
stored packed input or an authenticated deterministic regeneration path.

### 14.3 Reconciliation

Component records, per-shot sparse size histograms, global component
histograms, per-shot ledgers, shard summaries, replay indices, paired tables,
and global summaries must reconcile exactly. An unknown enum, count mismatch,
unbounded category, invalid tail bit, or digest mismatch is fatal.

For every shot and at every shard/run aggregation level, the analyzer checks
at least:

```text
completed_final_component_count
    = committed_component_count + durable_deferred_component_count
sum(exact completed-size histogram counts)
    = completed_final_component_count
sum(size * exact completed-size histogram count)
    = sum(cluster_defect_count over completed component rows)
committed_defect_count
    = sum(cluster_defect_count where durable_decision = committed)
union_attempt_count
    = successful_union_count + failed_union_count
heap_operation_count
    = heap_push_count + heap_pop_count
```

Within each lane, `successful_union_count` equals the sum of `merge_count`
over its completed terminal components, or over its current censored
components when the lane is censored. Per-shot additive operation counters,
maximum lane peaks, and summed lane peaks must reproduce the exact 12-row lane
table.

On complete-summary shots, every stored maximum must reproduce the relevant
component rows. On incomplete-summary shots, final-component maxima must be
`null` and the partial lower-bound maximum must reproduce the censored rows.
Censored rows contribute to no completed count, completed histogram, completed
quantile, gate-decision count, or durable-component total. `K_i` must equal the
committed-component defect sum, `L_i` must equal the original active detectors
owned by the authenticated 12 lanes, and `H_i`, `R_i`, `K_i`, and `L_i` must
satisfy every identity in Section 12.2.

## 15. Artifacts, integrity, and resume

The planned output root is:

```text
out/cguf_mwpm_d7_p003_v3/
├── shakeout_1k_collection/
│   ├── protocol.json
│   ├── manifest.json
│   ├── collection/shards/
│   ├── collection/component_metrics/
│   └── collection/summary.json
├── shakeout_1k_analysis/
│   ├── analysis.json
│   └── report.md
├── shakeout_1k_replay/
│   └── replay.json
├── characterization_10k_collection/
│   ├── protocol.json
│   ├── manifest.json
│   ├── corpus/
│   │   ├── detectors.bitpack
│   │   ├── observables.bitpack
│   │   └── index.json
│   ├── collection/shards/
│   ├── collection/component_metrics/
│   └── collection/summary.json
├── characterization_10k_analysis/
│   ├── analysis.json
│   └── report.md
├── characterization_10k_replay/
│   └── replay.json
├── latency_collection/
│   ├── protocol.json
│   ├── suite.json
│   └── batch-<size>.restart-<index>.json
├── latency_collection.workload/
│   ├── detectors.npy
│   ├── residuals.npy
│   ├── summaries.json
│   └── manifest.json
├── latency_collection.source.json
├── latency_analysis/
│   ├── latency_analysis.json
│   └── report.md
└── finalization/
    ├── finalization.json
    └── report.md
```

The latency analyzer consumes `latency_collection/` as an exact, flat generic
suite root. The authenticated detector/residual workload and its link back to
the characterization collection are sibling sidecars so they cannot appear as
unexpected entries in that exact-root check.

The exact file schemas and required-file lists are frozen before sampling.
Scientific artifacts are write-once. Collection and analysis have disjoint
required-file sets in sibling directories. Analysis writes only to a new,
initially absent analysis directory and never mutates collection shards,
corpus files, protocol files, or manifests.

Each range uses a two-file commit protocol. First write the component-metric
file to a same-filesystem temporary path, validate its exact shot range,
component count, schema, source/protocol identity, and content digest, then
atomically install it. Next write and validate the per-shot shard; it contains
the component file's relative path, digest, component count, shot range, and a
cross-digest over each shot's component-ledger range/count. The per-shot shard
is atomically installed last and is the range's commit marker.

Resume accepts a range only when both installed files exist and agree on the
protocol, exact range, counts, and cross-digests. An orphan component file is
not a completed range. Resume deterministically regenerates that range into a
temporary file; only if its bytes and digest equal the valid installed orphan
may it reuse the orphan and atomically install the missing per-shot commit
marker. A malformed orphan or regeneration mismatch fails the output root
closed and is never overwritten. A missing, partial, overlapping, foreign,
tampered, or mutually inconsistent pair is likewise rejected and never
repaired in place. Analysis repeats the same cross-file validation before
reading any scientific row and schedules only exact missing ranges.

Deterministic accuracy/workload ledgers are separated from nondeterministic
wall-clock timestamps and timing arrays so collection identities remain
reproducible. Existing `out/promatch_l1_round1*` corpora are immutable and are
not copied, resumed, or promoted into this experiment.

## 16. Protocol freeze and provenance

### 16.1 Two-commit freeze

The active V3 scientific collection follows the repository's two-commit
pattern:

1. **Implementation commit A3** contains all decoder, graph, collector,
   analyzer, timing, replay, plotting, CLI, and test code, plus the final form
   of this specification, its `experiments/README.md` dashboard entry, and the
   resolved `docs/PATCH_UF_MWPM_D7_P003_DRAFT.json`.
2. Generate the complete frozen protocol from a clean worktree at A3.
3. **Config-only commit B3** changes exactly one file relative to A3:
   `docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json`.
4. Run the shakeout and characterization from a clean worktree at B3.
5. The runner records A3 and B3 and verifies that
   `git diff --name-only A3..B3`
   contains only that frozen JSON path.

### 16.1.1 Superseded V1 launch record

V1 reached the frozen 1,000-shot engineering shakeout, passed collection and
read-only verification, and then failed the mandatory fresh-process replay
gate: its collection intentionally had no aggregate `corpus/` directory, but
the replay builder incorrectly required the characterization-only aggregate
detector file instead of consuming the detector-only bytes already validated
inside the 32 range shards. No characterization or latency run was started.

The V1 files under `out/cguf_mwpm_d7_p003_v1/` are immutable failed-gate audit
artifacts. They are never resumed, pooled, or promoted. V2 exposed an
authenticated detector-only aggregate from `verify_collection`, additionally
checks it against the installed detector corpus during characterization, and
passes only detector bytes into the replay workers. Per Section 10.2, this
source change requires a new freeze and fresh roots. The five V2 roots are the
SHA-256 digests of
`patch-uf-mwpm-d7-p003-v2-shakeout-replay-fix\\0<PURPOSE>`, with purposes
`engineering-shakeout`, `characterization`, `latency-schedule`, `bootstrap`,
and `replay-selection`; their literal values live in the draft/frozen JSON.

### 16.1.2 Superseded V2 launch record

V2 passed its fresh 1,000-shot collection, verification, analysis, and
fresh-process replay gates. Its disjoint 10,000-shot characterization sampled
the fixed worker ranges, but `_normalize_metrics` rejected the first completed
range before joining actual outcomes or installing either range artifact: the
uncompressed canonical normalized metric tree exceeded the frozen 128 MiB
per-range limit. The partial characterization root contains only its immutable
protocol copy and no committed scientific row. Nevertheless, the V2
characterization root is burned and is never resumed, deleted, pooled, or
promoted.

The V3 implementation adds a capacity gate without changing decoder
semantics. A scratch command executes one exact 313-shot logical range, and
freeze authenticates both that full-range measurement and a 2x projection of
the largest single-shot metric tree in the 100-shot probe. The V3 ceiling is
512 MiB. Its five roots are the SHA-256 digests of
`patch-uf-mwpm-d7-p003-v3-telemetry-cap-recovery\0<PURPOSE>`, with purposes
`engineering-shakeout`, `characterization`, `latency-schedule`, `bootstrap`,
and `replay-selection`, where `\0` denotes one NUL byte; the literal values
live in the draft/frozen JSON.

The eventual draft protocol may be developed at
`docs/PATCH_UF_MWPM_D7_P003_DRAFT.json`, but neither that path nor this
specification authorizes sampling.

### 16.2 Frozen fields

The protocol contains at least:

- schema, status, experiment ID, claim status, and canonical self-hash;
- implementation/config commits and complete relevant source hashes;
- requirements, Python/package versions, platform, and host identity;
- exact physical cell, circuit/DEM options, hashes, and dimensions;
- canonical graph, unmerged mechanism catalog, projection, weights, masks,
  port table, and fingerprints;
- lane/patch ordering and every UF growth, event, forest, peeling, arithmetic,
  tie, confidence, eligibility, budget, transaction, fallback, and frame rule;
- exact arm IDs and adapter/backend semantics;
- fixed stage shot counts, literal seed roots and derivation, worker/range
  schedule, batch shapes, and fixed-N stopping rule;
- paired accuracy and workload estimands/statistics;
- normative cluster-size definition, complete/censored component schemas,
  persisted memberships, deterministic component IDs, coordinate units,
  operation counters and 12-row lane tables, local gate versus durable
  transaction outcomes, per-shot summaries, censor/null rules, exact sparse
  histograms, quantiles, stratifications, and complete-shot resampling rules;
- latency pairs, intervals, corpus identity, call counts, order/bootstrap seeds,
  per-call ordered workload keys/digests, precomputed cluster/workload joins,
  process isolation, affinity, GC, and host policies;
- artifact schemas, required files, atomic-install/resume rules, replay policy,
  and fatal gates; and
- analyzer/plotter source hashes and permitted-result language.

Any `TBD`, environment-derived default, unordered iteration, outcome-selected
threshold, implicit tolerance, or omitted fallback path blocks `frozen=true`.

### 16.3 Run manifest

At launch, the run manifest records and verifies the frozen fields plus the
clean-worktree state, command line, thread environment, resolved CPU affinity,
available memory/storage, start/end timestamps, shard hashes, and final
artifact Merkle or ordered digest. Timestamps are provenance, not inputs to a
deterministic decision ledger.

## 17. Implementation plan

The planned code organization is:

```text
src/yoked/decoding/_patch_uf_graph.py       canonical full-history projection and ports
src/yoked/decoding/_patch_uf.py             weighted UF, reference semantics, peeling
src/yoked/decoding/_patch_uf_decoder.py     four decoder adapters and residual algebra
src/yoked/decoding/_patch_uf_experiment.py  fixed-shot paired collector and ledgers
src/yoked/decoding/_patch_uf_stats.py       paired/workload statistics
src/yoked/decoding/_patch_uf_latency.py     controlled timing collection
src/yoked/decoding/_patch_uf_analysis.py    validation, report, tables, and plots
tools/benchmark_patch_uf_mwpm               experiment CLI
tools/analyze_patch_uf_mwpm                 analysis CLI
```

These paths and commands are planned, not currently implemented.

### 17.1 Reuse boundaries

Reuse the maintained circuit generator, layout roles, canonical matching-edge
representation, direct Global-MWPM path, GF(2) support replay patterns, atomic
artifact I/O, paired contingency/statistics, and controlled latency machinery
where their contracts match this specification.

Do not reuse the current windowed `DomainGraph` as the UF projection: it omits
terminal and cross-window context required here. Do not use the full-graph
oracle as the deployed confidence gate; it is a testing/audit tool, not online
information available to the treatment.

### 17.2 Decoder API

The production decoder exposes ordinary batch prediction without telemetry.
The experiment path may explicitly request immutable bounded records. Both
paths call the same core proposal/gate implementation and must return identical
predictions.

Compilation produces a top-level pickle-friendly decoder object suitable for
spawned workers. The adapter supports bit-packed and unpacked inputs, scalar
and batch shapes, empty batches, noncontiguous views, and strict validation of
detector/observable widths and unused bits.

## 18. Required tests

No collection begins until the following test families exist and pass.

### 18.1 Graph and projection

- The selected real cell reproduces circuit/DEM/layout dimensions and hashes.
- Every canonical edge has exactly one ownership class.
- Full-history lanes include terminal and cross-window edges.
- Yoke/cross-lane ports preserve remote ID, weight, mask, and edge ID.
- The gate receives identical outputs when post-decision remote telemetry bits
  are changed.
- A true boundary is correction-eligible and never converted into a port.
- The V1 role-pair allowlist rejects every unlisted topology.
- All V1 correction edges are zero-frame, finite, and strictly positive.
- Parallel unmerged DEM mechanisms with ambiguous frames fail closed.
- Projection fingerprints are deterministic and parameter-sensitive.
- Permuting source iteration cannot change canonical ordering or ownership.
- Authenticated patch-local half-integer coordinates serialize exactly as
  doubled integers, including boundary and terminal detector cases.

### 18.2 Weighted UF and confidence

- Exhaustive small positive graphs agree with the exact slow reference.
- Two-sided internal and one-sided boundary/port growth times are correct.
- Equal-time events are processed simultaneously.
- A neutral component later reached by an odd growing component follows the
  frozen merge/parity/boundary transition.
- Edge iteration permutations do not change components, support, or margin.
- Deterministic forest selection and reverse peeling are golden-tested.
- Peeling support has the claimed exact boundary.
- Defect, absorbed-vertex, forest/support, temporal/spatial, growth, merge, and
  operation metrics match hand-computed small-graph values.
- Singleton spans and last-membership time are zero; half-step coordinate spans
  and `maximum - minimum` round spans match golden values.
- Component IDs and membership digests are invariant to edge order, DSU root
  choice, worker count, and serialization order.
- Persisted sorted absorbed memberships independently reproduce component IDs,
  membership digests, counts, and absorbed geometry.
- A component support is unique; duplicates or overlaps are fatal, while
  validated patch supports obey the stated GF(2) identities.
- Every saturated port taints its terminal component; a simultaneous port or
  nonforest correction event has zero confidence margin.
- The strict threshold rejects equality and accepts the next representable
  value above it when otherwise eligible.
- A merged tainted component cannot partially commit an earlier subtree.
- Every deterministic budget boundary and exhaustion reason is tested.
- Heap push/pop/stale-pop, union attempt/success/failure, event-batch, port,
  peel, and peak counters match exact golden traces and reconciliation
  identities.
- Multiple caps exceeded by one proposed batch record the complete exceeded
  set and deterministic diagnostic primary cap; frozen queue lifecycle and
  preparatory-operation rollback match golden traces.
- A cap reached while preparing a simultaneous batch restores the state after
  the prior complete batch; the rejected operation is not applied or counted.
- Budget/incomplete partial components carry only lower-bound sizes, null all
  terminal-only fields, and cannot enter a completed-component histogram.
- The deployed gate never calls an oracle and cannot receive actual observables.

### 18.3 Transactions and decoder

- Patch aggregation independently recomputes exact boundary and frame.
- Any aggregate algebra disagreement is fatal and cannot become fallback.
- A budget/incomplete-lane patch abort leaves the exact original
  syndrome/frame unchanged.
- A locally gate-eligible component in the sibling lane of an aborted patch
  retains its gate result but is durably deferred with the patch-abort reason.
- Multi-reason gate cases obey the frozen primary-reason precedence while
  retaining their full reason and port-kind sets.
- Independent accepted components survive an unrelated ordinary defer.
- Shots with no durable component bit-match Global MWPM.
- Every shot satisfies the detector-count identity in Section 12.2.
- The residual matcher is complete-graph Global MWPM and is called exactly
  once per treatment batch.
- Frame XOR, little-endian packing, and unused tail bits are correct.
- Inputs and compiled graph/projection objects are immutable.
- Scalar, batch, empty, noncontiguous, malformed, and wrong-width inputs are
  covered.
- Adapter-Control and UF-Shadow bit-match Global MWPM.
- Ordinary and telemetry-enabled paths return identical predictions.
- Decoder registration, pickling, and spawned-process use are tested.

### 18.4 Physical-fault and differential tests

- Exhaust all single canonical mechanisms on small maintained circuits.
- Test selected two-mechanism combinations including cross-window, terminal,
  true-boundary, yoke-port, and simultaneous-tie cases.
- On the selected real DEM, independently replay every durable support and
  assert the complete GF(2) formula.
- Compare candidate boundary/frame/weight calculations with the existing
  full-graph oracle in shadow tests without using oracle outcomes in the gate.

### 18.5 Collector, analysis, and replay

- The same seed produces a bit-exact paired ledger.
- Both accuracy arms share the exact corpus and cannot mutate it.
- Fixed-shot collection rejects a set `MAX_ERRORS`.
- Worker ranges cover exactly `N` with no overlaps or gaps.
- Component rows, per-shot sparse histograms, size/decision strata, workload
  totals, and run summaries reconcile exactly at shard and run levels.
- Complete versus censored shot maxima obey the zero/null policy; censored rows
  are excluded from every completed-component total and quantile.
- Lane operation rows are stored once; additive shot counters equal lane sums,
  and maximum/sum lane-peak fields reproduce those rows without component-row
  duplication.
- Component-size bootstrap replicates preserve shot groups, normalize by the
  selected completed-component denominator, and handle zero denominators
  exactly as specified.
- Atomic resume accepts only cross-authenticated component/per-shot file pairs;
  orphan component files never commit a range.
- A simulated crash between component-file and commit-marker installation
  reuses an orphan only after byte-identical deterministic regeneration;
  malformed or mismatched orphans fail closed without overwrite.
- Tampered syndrome, support, margin, port trace, prediction, or fingerprint is
  rejected.
- Bounded replay selection is deterministic and self-verifying.
- Every retained category replays in a fresh process.
- Cluster quantiles and complete-shot cluster resampling preserve within-shot
  component groups and match golden values.
- Clopper–Pearson, Tango, workload bootstrap, and zero-event cases match golden
  values.

### 18.6 Latency

- Fake-clock tests prove the exact timer scopes.
- Pair schedules have 50/50 `AB`/`BA` balance per restart.
- Warmup/timed counts and array shapes match every batch size.
- Input generation, compilation, telemetry, logging, and I/O remain outside
  the timer.
- Precomputed cluster/workload metrics join by immutable workload ID and are
  never recomputed or serialized inside timed intervals.
- Byte-identical detector vectors with different global shot IDs remain
  distinct workload keys, and cyclic batch wraps reconstruct the exact ordered
  key list and detector/precomputed-summary digests.
- The corpus index is a bijection to global shot ID; duplicate precomputed
  workload rows fail while valid workload reuse across calls succeeds.
- Workload/latency tables retain repeated calls, separate every pair/side
  context, and place censored batch-1 calls only in the censor-status group.
- Fresh-process restart isolation and concurrency one are enforced.
- Invalid/nonpositive timings and corpus/host-policy drift are rejected.
- Hierarchical bootstrap resamples restarts and blocks, never individual calls.
- Materialization checks the complete matcher on both full authenticated
  corpora against recorded Global/treatment predictions and binds the verified
  collection control-equality ledgers into canonical provenance.
- Every fresh restart authenticates that full-corpus attestation and checks
  exactly one deterministic row across all six rebuilt variants outside timed
  intervals; tests bound the preflight row count at one.

The full repository test suite and `tests/yoked/decoding` suite must pass in
the pinned environment before the implementation commit is made.

## 19. Planned command workflow

The following is the implemented CLI workflow. The commands and frozen JSON
must agree literally before use.

```bash
source .venv/bin/activate
export TMPDIR=/data2/s2chitni/.tmp
export MPLCONFIGDIR="$TMPDIR/yoked-surface-codes-matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
unset MAX_ERRORS

python -m pytest -q

tools/benchmark_patch_uf_mwpm smoke \
    --protocol docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --out "$TMPDIR/patch-uf-mwpm-smoke" \
    --processes 32

tools/benchmark_patch_uf_mwpm probe \
    --protocol docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --shots 100 \
    --out "$TMPDIR/patch-uf-mwpm-probe" \
    --processes 32

tools/benchmark_patch_uf_mwpm capacity-probe \
    --protocol docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --out "$TMPDIR/patch-uf-mwpm-capacity"
```

After those development gates pass, commit all implementation, tests, the
resolved draft protocol, this specification, and the dashboard as
implementation commit A. Verify a clean worktree at A, then rerun the final
smoke and probe from that exact commit into fresh authenticated scratch roots:

```bash
tools/benchmark_patch_uf_mwpm smoke \
    --protocol docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --out "$TMPDIR/patch-uf-mwpm-smoke-commit-a" \
    --processes 32

tools/benchmark_patch_uf_mwpm probe \
    --protocol docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --shots 100 \
    --out "$TMPDIR/patch-uf-mwpm-probe-commit-a" \
    --processes 32

tools/benchmark_patch_uf_mwpm capacity-probe \
    --protocol docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --out "$TMPDIR/patch-uf-mwpm-capacity-commit-a"
```

The only permitted repository write before config commit B is then:

```bash
tools/benchmark_patch_uf_mwpm freeze \
    --draft docs/PATCH_UF_MWPM_D7_P003_DRAFT.json \
    --smoke-root "$TMPDIR/patch-uf-mwpm-smoke-commit-a" \
    --probe-root "$TMPDIR/patch-uf-mwpm-probe-commit-a" \
    --capacity-root "$TMPDIR/patch-uf-mwpm-capacity-commit-a" \
    --out-protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json
```

Authenticate the generated protocol, confirm that the A-to-B diff contains
only `docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json`, commit that one file as B,
and verify a clean worktree at B. Then run:

```bash
tools/benchmark_patch_uf_mwpm verify-protocol \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json

tools/benchmark_patch_uf_mwpm collect \
    --stage shakeout-1k \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --out out/cguf_mwpm_d7_p003_v3/shakeout_1k_collection \
    --processes 32

tools/benchmark_patch_uf_mwpm verify-collection \
    --stage shakeout-1k \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --input out/cguf_mwpm_d7_p003_v3/shakeout_1k_collection

tools/analyze_patch_uf_mwpm stage \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --collection out/cguf_mwpm_d7_p003_v3/shakeout_1k_collection \
    --stage engineering-shakeout \
    --out out/cguf_mwpm_d7_p003_v3/shakeout_1k_analysis

tools/benchmark_patch_uf_mwpm replay \
    --stage shakeout-1k \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --collection out/cguf_mwpm_d7_p003_v3/shakeout_1k_collection \
    --analysis out/cguf_mwpm_d7_p003_v3/shakeout_1k_analysis/analysis.json \
    --processes 32 \
    --out out/cguf_mwpm_d7_p003_v3/shakeout_1k_replay

tools/benchmark_patch_uf_mwpm collect \
    --stage characterization-10k \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --out out/cguf_mwpm_d7_p003_v3/characterization_10k_collection \
    --processes 32

tools/benchmark_patch_uf_mwpm verify-collection \
    --stage characterization-10k \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --input out/cguf_mwpm_d7_p003_v3/characterization_10k_collection

tools/analyze_patch_uf_mwpm stage \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --collection out/cguf_mwpm_d7_p003_v3/characterization_10k_collection \
    --stage characterization \
    --out out/cguf_mwpm_d7_p003_v3/characterization_10k_analysis

tools/benchmark_patch_uf_mwpm replay \
    --stage characterization-10k \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --collection out/cguf_mwpm_d7_p003_v3/characterization_10k_collection \
    --analysis out/cguf_mwpm_d7_p003_v3/characterization_10k_analysis/analysis.json \
    --processes 32 \
    --out out/cguf_mwpm_d7_p003_v3/characterization_10k_replay

tools/benchmark_patch_uf_mwpm latency \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --collection out/cguf_mwpm_d7_p003_v3/characterization_10k_collection \
    --out out/cguf_mwpm_d7_p003_v3/latency_collection

tools/benchmark_patch_uf_mwpm verify-latency \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --input out/cguf_mwpm_d7_p003_v3/latency_collection

tools/analyze_patch_uf_mwpm latency \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --input out/cguf_mwpm_d7_p003_v3/latency_collection \
    --out out/cguf_mwpm_d7_p003_v3/latency_analysis \
    --bootstrap-replicates 10000

tools/analyze_patch_uf_mwpm finalize \
    --protocol docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json \
    --shakeout out/cguf_mwpm_d7_p003_v3/shakeout_1k_analysis \
    --characterization out/cguf_mwpm_d7_p003_v3/characterization_10k_analysis \
    --latency out/cguf_mwpm_d7_p003_v3/latency_analysis \
    --out out/cguf_mwpm_d7_p003_v3/finalization
```

Smoke/probe outputs live under `$TMPDIR`. Scientific output uses a fresh
`out/` directory and is never run over an old corpus. Accuracy collection is
exactly 32 processes; latency restarts are serialized fresh processes.
The `smoke`, `probe`, `capacity-probe`, `collect`, `latency`, and `replay`
commands share one exclusive campaign lock under `$TMPDIR`; a stale lock is
removed only after its recorded owner PID is proven absent.

## 20. Interpretation and follow-up routing

### 20.1 Permitted result language

Examples of permitted language are:

- “On 10,000 paired shots at `d=7`, `p=0.003`, the estimated failure-probability
  difference (treatment minus Global MWPM) was X with a two-sided 95% paired
  interval of Y to Z.”
- “The UF frontend durably committed on X/Y shots and changed the total
  detector workload delivered to Global MWPM by the reported descriptive
  ratio.”
- “Among X completed components, the observed defect-count distribution was
  ..., while the largest-component distribution used Y cluster-summary-
  complete shots; Z censored shots were reported separately.”
- “Conditional on the immutable characterization corpus, recorded host, and
  frozen execution policy, the batch-1 in-process geometric latency ratio was
  X with the reported hierarchical-bootstrap interval.”
- “The fixed run carried the predeclared low-volume diagnostic tag for
  discordance/activation; the interval reports its sampling uncertainty.”

Forbidden language includes “accuracy preserved,” “equivalent,” “safe,”
“proved faster,” “hardware speedup,” or “production-ready” based on V1.

### 20.2 Follow-up branches

After the immutable report is complete:

- If integrity fails, fix/version the implementation and restart from a new
  engineering shakeout; do not reinterpret partial data.
- If UF rarely commits, inspect port and margin telemetry before changing the
  gate. Any changed gate uses a new protocol and disjoint shots.
- If the final-cluster tail is heavy or censoring is nonzero, use the retained
  size/geometry/operation casebook to distinguish genuine large components
  from budget or implementation pressure before changing a cap.
- If workload falls but total latency rises, optimize the frontend/adapter
  before increasing Monte Carlo sample size.
- If total latency falls but paired accuracy regresses materially, use the
  replay casebook to revise confidence/ownership rather than hiding the loss
  behind backend speed.
- If both accuracy and latency estimates justify a claim-oriented follow-up,
  design a separate frozen multi-cell pilot/holdout study with an explicit
  noninferiority margin and power calculation.

## 21. Resolved freeze literals

All former freeze blockers are literal in
`docs/PATCH_UF_MWPM_D7_P003_DRAFT.json` and enforced by the protocol validator.
Scientific sampling still requires the clean two-commit freeze in Section 16.

| Item | Frozen V1 resolution |
| --- | --- |
| Confidence threshold | `tau=0x0.0p+0`, strict `margin > tau`; equality defers |
| Production arithmetic | Per-lane exact integer ticks with a graph-derived binary exponent; differential agreement with the `Fraction` reference |
| Simultaneous events | Complete next-time batches are discovered from one pre-event state and applied in canonical edge order |
| Semantic caps | growth/batches `3403`; union attempts `3171`; successful unions `695`; forest, absorbed vertices, and peel operations `696` |
| Production caps | heap pushes/pops `4,194,304`; heap operations `8,388,608`; peak heap and temporary-memory units `262,144` |
| Confidence bins | `[0, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, +inf]` |
| Cluster display bins | `1`, `2`, `3–4`, `5–8`, `9+` defects |
| Seed roots | Independent literal 256-bit roots for shakeout, characterization, timing, bootstrap, and replay selection |
| Affinity/NUMA | Logical CPU 31, NUMA node 0, with the exact host identity recorded in the draft |
| Schemas | Versioned protocol, collection, component, corpus, analysis, replay, and latency schemas |
| Source inventory | Sorted repository-relative source list in the draft; byte hashes and inventory digest generated only by the commit-A freeze |

The physical cell, arm names, fixed shot counts, paired design, full-history
projection boundary, zero-frame policy, residual algebra, latency comparisons,
and non-claim-bearing posture are already fixed by this document. Changing one
of those decisions creates a new experiment specification rather than an
in-place edit to a frozen protocol.

## 22. Local references

- [`README.md`](README.md) — mutable experiment-status dashboard and frozen-
  artifact policy.
- [`PROMATCH_FIG8_PAIRED_GCP_SWEEP.md`](PROMATCH_FIG8_PAIRED_GCP_SWEEP.md) —
  paired outcome and fixed-shot campaign precedent.
- [`PROMATCH_L1_POLICY_AUDIT_20K.md`](PROMATCH_L1_POLICY_AUDIT_20K.md) —
  two-commit freeze, deterministic artifacts, and discovery discipline.
- [`PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md`](PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md)
  — evidence motivating complete graph context and yoke-aware residual
  decoding.
- [`../docs/PROMATCH_IMPLEMENTATION_PLAN.md`](../docs/PROMATCH_IMPLEMENTATION_PLAN.md)
  — existing paired statistics, workload, latency, provenance, and readiness
  contracts reused where applicable.
- [`../docs/PROMATCH_PILOT_FROZEN_V3.json`](../docs/PROMATCH_PILOT_FROZEN_V3.json)
  — immutable historical physical-cell hashes and planning evidence.
- [`../AGENTS.md`](../AGENTS.md) — workstation process, thread, environment,
  clean-worktree, and immutable-corpus rules.
- [`../REPRODUCING_FIG8_1D.md`](../REPRODUCING_FIG8_1D.md) — maintained YSC
  experiment and timing workflow.
- [`../docs/CODEBASE_GUIDE.md`](../docs/CODEBASE_GUIDE.md) — source-layout and
  decoder integration guide.
