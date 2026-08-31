# Confidence-Gated Weighted Union-Find Decoder Implementation Plan

- **Status:** implemented; final freeze and scientific execution gates pending
- **Last updated:** 2026-08-30
- **Target experiment:** `cguf-01-d7-n6-y2-r28-p0.003`
- **Target decoder:** `weighted-uf-fullhistory-patchlocal-zeroframe-residual-global-mwpm-v1`
- **Scientific posture:** engineering and non-claim-bearing characterization
- **Normative experiment specification:**
  [`experiments/CONFIDENCE_GATED_UF_MWPM_D7_P003.md`](../experiments/CONFIDENCE_GATED_UF_MWPM_D7_P003.md)

This document records the ordered software implementation plan used for the
experiment. It defines module boundaries, data contracts, development
milestones, tests, and readiness gates. The user authorized implementation and
execution on 2026-08-30; protocol freezing and scientific sampling remain
subject to the explicit gates below. If this plan and the experiment
specification disagree, the experiment specification wins.

The production design is a **confidence-gated weighted Union-Find frontend with
residual Global MWPM**. It borrows the clustering, component-validity, spanning-
forest, and peeling structure of the Delfosse--Nickerson Union-Find decoder, but
it is not a paper-faithful complete UF decoder. It adds exact likelihood-weighted
growth, patch-local ownership, guard ports, confidence-gated partial commits,
atomic two-lane patch transactions, and one complete joint MWPM residual decode.

## 1. Scope and definition of done

### 1.1 In scope

The plan covers:

1. terminal-inclusive compilation of twelve full-history `(patch, basis)` UF
   lanes from the canonical all-frame matching graph;
2. an exact slow semantic reference for weighted growth, simultaneous events,
   forest construction, peeling, confidence, and semantic-budget censoring;
3. a production engine that is identical to the reference for mathematical
   semantics and reference-comparable caps, while production-only heap/memory
   cap outcomes follow separate frozen lifecycle golden traces;
4. final-component gating and atomic X/Z patch transactions;
5. packed decoder adapters for Global MWPM, Adapter-Control MWPM, UF-Shadow
   MWPM, and Confidence-Gated UF--MWPM;
6. residual-syndrome and observable-frame algebra followed by exactly one
   complete-graph PyMatching call per nonempty public batch;
7. bounded metrics and trace capture needed by the paired experiment;
8. collector, replay, analysis, and controlled latency integration required to
   execute the already-specified 1,000-shot and 10,000-shot stages; and
9. protocol generation, provenance, and launch-readiness checks.

### 1.2 Out of scope

V1 does not include:

- deletion or rebuilding of matching-graph edges per shot;
- a reduced or patch-local MWPM backend;
- a correlated residual matcher;
- streaming or online syndrome processing;
- local access to yoke or remote-port syndrome bits;
- nonzero-observable-frame UF commits;
- threshold, distance-scaling, hardware, FPGA, cryogenic, or real-time claims;
- adoption of the original paper's `O(n alpha(n))` complexity claim; or
- tuning `tau` or budgets using the characterization outcomes.

### 1.3 Decoder completion criteria

The decoder portion is implementation-complete only when all of the following
are true:

- [ ] Every canonical edge receives exactly one fail-closed ownership class.
- [ ] All body and terminal detectors belong to exactly one full-history lane;
      yoke detectors belong to none.
- [ ] The exact reference passes exhaustive small-graph golden tests.
- [ ] The production engine agrees with the reference on event batches, final
      memberships, forests, peeled supports, margins, decisions, reasons, and
      censoring caused by reference-comparable semantic caps.
- [ ] Production-only heap and representation-memory cap decisions match the
      frozen production-lifecycle golden traces.
- [ ] The lane API makes remote syndrome access impossible by construction.
- [ ] Every durable support independently replays to its claimed detector
      boundary and zero observable frame.
- [ ] Patch aborts leave the original syndrome and frame unchanged by
      construction, without mutating and then rolling back live state.
- [ ] A shot with no durable component bit-matches direct Global MWPM.
- [ ] Adapter-Control MWPM and UF-Shadow MWPM always bit-match Global MWPM.
- [ ] The Global-MWPM wrapper itself bit-matches the maintained direct
      PyMatching/Sinter baseline on every supported input shape.
- [ ] The treatment invokes the complete residual matcher exactly once per
      nonempty public batch.
- [ ] Ordinary, metrics, trace, and latency callables return identical packed
      treatment predictions.
- [ ] Scalar, batch, empty, noncontiguous, malformed, packed-tail, factory-
      pickling, fork-inheritance, and spawned-reconstruction cases pass.
- [ ] All single mechanisms and the selected two-mechanism physical cases pass
      independent GF(2) replay.
- [ ] `tests/yoked/decoding` and the full repository test suite pass in the
      pinned environment.

Experiment infrastructure is complete only after the additional collection,
analysis, replay, artifact, and latency gates in Sections 10--13 pass.

## 2. Fixed architectural contract

### 2.1 Compile path

The compile path is additive and fail-closed:

```text
decomposed Stim DEM
        |
        v
compile_layout(dem, mode="fullhistory")
        |
        v
compile_matching_graph(
    dem,
    layout,
    require_zero_frame=False,
    retain_cross_lane_edges=True,  # new UF opt-in; default remains False
)
        |
        +---- complete PyMatching backend retained unchanged
        |
        v
validate flattened, unmerged DEM mechanism catalog
        |
        v
compile terminal-inclusive PatchUFProjection
        |
        +---- 12 PatchUFLaneProjection objects
        +---- correction-edge ownership
        +---- true-boundary incidences
        +---- immutable guard-port incidences
        +---- exact dyadic weight table
        `---- authenticated projection fingerprint
```

Do not derive lanes from the existing `DomainGraph`. Even in `fullhistory`
mode, the maintained domain graph contains body detectors only and omits the
terminal layer. Lane membership must instead be derived from the detector roles
in [`_promatch_layout.py`](../src/yoked/decoding/_promatch_layout.py), following
the role-based full-history mapping already used by
[`_pinball_v2.py`](../src/yoked/decoding/_pinball_v2.py).

### 2.2 Per-shot treatment path

```text
immutable original packed detector shot
        |
        v
select local syndrome bits for each active lane
        |
        v
exact weighted UF growth -> final components -> deterministic peeling
        |
        v
component confidence/eligibility decisions
        |
        v
validate X/Z proposals as one atomic patch transaction
        |
        v
independently replay all durable canonical edge IDs
        |
        +---- durable boundary
        `---- durable observable frame (must be zero in V1)
        |
        v
residual = original XOR durable boundary
        |
        v
one complete Global-MWPM decode_batch(residual_batch)
        |
        v
prediction = backend prediction XOR durable frame
```

The original input remains immutable throughout planning and validation.
Tentative support is never applied to the live residual. The residual is built
once from independently validated durable transactions.

### 2.3 Experimental paths

| Arm ID | Implementation path | Accuracy arm? | Expected output |
| --- | --- | --- | --- |
| `global-mwpm-u0-joint-y2` | Minimal packed validation, then complete matcher on original input | Yes | Direct Global MWPM |
| `adapter-control-global-mwpm-v1` | Treatment adapter and lane-selection plumbing, no UF proposal, complete matcher on original input | No | Bit-identical to Global MWPM |
| `weighted-uf-shadow-global-mwpm-v1` | Full UF, peeling, gate, and transaction validation; discard correction; complete matcher on original input | No | Bit-identical to Global MWPM |
| `weighted-uf-fullhistory-patchlocal-zeroframe-residual-global-mwpm-v1` | Full UF and durable residual application, then complete matcher on residual input | Yes | Treatment prediction |

The Phase 11 policy literals are resolved, and only the treatment is registered
as a general Sinter custom decoder. The adapter-control and shadow callables
remain experiment-local. Their role is timing validity, not additional
scientific arms.

### 2.4 Non-negotiable invariants

1. **Canonical identity.** A support item is always a dense canonical
   `edge_id` from the complete graph, never a lane-local edge index.
2. **Remote blindness.** The lane engine accepts only lane-owned syndrome bits.
   A port record may retain a remote detector ID for provenance, but its remote
   bit is unavailable during growth, confidence, and gating.
3. **Boundary distinction.** A genuine `target=None` edge is local correction
   support. A yoke or cross-lane edge is a port. A yoke is never a matching
   boundary.
4. **No shared boundary root.** True-boundary edges are distinct virtual-leaf
   incidences or typed one-ended edges. They must not be represented as one
   shared DSU vertex that merges otherwise independent components.
5. **Exact decisions.** Binary64 weights are interpreted as exact dyadic
   rationals. Event grouping, slack, threshold comparison, and budget behavior
   use no epsilon and no binary64 arithmetic in the decision path.
6. **Atomic simultaneous events.** All events at the next exact time are found
   from the same pre-event state and applied as one transaction.
7. **Final-component atomicity.** A tainted or low-confidence final component
   cannot partially commit a favorable subtree that existed earlier.
8. **Patch atomicity.** Budget exhaustion or local incompleteness in either
   basis lane prevents every tentative proposal in that patch from becoming
   durable. An ordinary component defer does not erase an independent eligible
   component.
9. **Independent algebra.** Durable boundary and frame are reconstructed from
   canonical edges after decisions. Incrementally maintained values are never
   trusted as the only check.
10. **One backend.** There is no emergency matcher and no graph deletion. All
    shots reach the unchanged complete matcher exactly once per public batch.
11. **No oracle in production.** The full-graph oracle and actual observables
    may appear in tests or post-decision analysis only.
12. **Telemetry separation.** Capture changes what is copied out, never what
    is computed or decided. Mandatory in-core counters used by budgets run in
    every mode; retained metrics, joins, hashing, copying, serialization, and
    logging remain outside latency intervals.

### 2.5 Measurement outputs the implementation must support

The decoder and capture API must expose sufficient information for these fixed
experiment endpoints without recomputing decoder decisions in the analyzer:

| Category | Primary quantity | Required decoder-side inputs |
| --- | --- | --- |
| Accuracy | Paired any-observable physical-shot failure difference | Packed prediction from Global MWPM and treatment; actual observables are joined later |
| Backend workload | `sum(residual detector events) / sum(original detector events)` | Original and residual detector counts per shot |
| Frontend coverage | `sum(committed component defects) / sum(original detector events)` | Durable component boundaries and original detector count |
| Cluster size | Exact completed-component defect-count histogram | Original defect membership for every completed final component |
| Cluster tail | Shot-weighted largest completed final-component defect count | Per-shot completeness flag and component memberships |
| Routing | Commit, defer, abort, censor, boundary, port, confidence, and budget rates | Local gate and durable transaction outcomes plus lane counters |
| Latency | Batch-1 treatment/Global-MWPM total-path ratio | Telemetry-free callables and immutable original/residual workloads |

The component-weighted cluster distribution and shot-weighted cluster tail have
different denominators. A censored lane contributes only censored lower-bound
records; it never contributes a completed-component size.

The normative V1 size is:

```text
cluster_defect_count(C) =
    number of original nonzero detector events in C's claimed syndrome boundary
```

It is not absorbed-vertex count, forest-edge count, peeled-support size, or
correction weight. Those are separate component metrics.

## 3. Graph, projection, and data contracts

### 3.1 Reused maintained objects

Reuse these stable primitives without importing their old policy assumptions:

| Existing code | Reuse | Do not reuse |
| --- | --- | --- |
| [`_promatch_layout.py`](../src/yoked/decoding/_promatch_layout.py) | `compile_layout`, detector roles, coordinates, patch and basis labels | `domain_detector_ids` as terminal-inclusive lane membership |
| [`_promatch_graph.py`](../src/yoked/decoding/_promatch_graph.py) | `Edge`, dense IDs, masks, deterministic canonical order, complete matcher, graph fingerprint | `DomainGraph` as the UF graph; its permissive zero-weight and topology assumptions |
| [`_pinball_v2_decoder.py`](../src/yoked/decoding/_pinball_v2_decoder.py) | flattened unmerged-DEM validation pattern and one-residual-batch adapter pattern | Pinball schedule, primitive priorities, or domain disposition |
| [`_pinball_v2.py`](../src/yoked/decoding/_pinball_v2.py) | role-to-full-history-lane mapping and GF(2) replay patterns | Pinball ownership of yoke-coupled edges |
| [`_artifact_io.py`](../src/yoked/decoding/_artifact_io.py) | strict JSON, output fencing, no-clobber atomic installation | mutation of existing immutable output roots |
| [`_promatch_stats.py`](../src/yoked/decoding/_promatch_stats.py) | paired contingency, Clopper--Pearson, Tango, digest, seed, and process-count primitives where contracts match | frozen ProMatch thresholds, stopping rules, schemas, or source identities |

The unmerged mechanism validator should be extracted into a small shared
internal module, with characterization tests proving no behavior change for
Pinball V2. Do not import a private Pinball function as a permanent UF
dependency.

### 3.2 Planned graph types

`src/yoked/decoding/_patch_uf_graph.py` should define immutable plain-data
types equivalent to:

```python
PatchUFLaneKey(
    patch_id: int,
    check_basis: str,
)

GuardPortIncidence(
    edge_id: int,
    lane_id: int,
    local_vertex: int,
    remote_detector_id: int,
    remote_lane_id: int | None,
    port_kind: str,
    exact_weight_index: int,
    observable_mask: bytes,
)

PatchUFLaneProjection(
    lane_id: int,
    key: PatchUFLaneKey,
    global_detector_ids: tuple[int, ...],
    local_x2: tuple[int, ...],
    y2: tuple[int, ...],
    times: tuple[int, ...],
    internal_correction_edges: tuple[...],
    true_boundary_edges: tuple[...],
    guard_ports: tuple[GuardPortIncidence, ...],
    incidence_offsets: tuple[int, ...],
    incidence_indices: tuple[int, ...],
)

PatchUFProjection(
    canonical_graph_fingerprint: str,
    num_detectors: int,
    num_observables: int,
    lanes: tuple[PatchUFLaneProjection, ...],
    patch_lane_ids: tuple[tuple[int, int], ...],
    detector_lane_id: tuple[int | None, ...],
    detector_local_index: tuple[int | None, ...],
    edge_owner_kind: tuple[str, ...],
    edge_owner_lane: tuple[int | None, ...],
    exact_weights: tuple[...],
    fingerprint: str,
)
```

Arrays may replace tuples in the optimized representation, but they must be
read-only after compilation and preserve a canonical serialized form. An
optimized integer array may encode `None` with a frozen sentinel such as `-1`;
the sentinel is part of the schema and fingerprint.

The projection does not embed `pymatching.Matching`, which is not pickleable.
The compiled decoder owns the complete matcher separately. Plain projection
data may be serialized, inherited by forked workers, or reconstructed in a
spawned worker from authenticated compile inputs.

A port's stable local identity is `(edge_id, lane_id, local_vertex)`, not only
`edge_id`. A future cross-lane edge may be globally owned once while appearing
as one immutable incidence in each adjacent lane.

### 3.3 Classification precedence

Classify every canonical edge once in this order:

1. A one-ended `target=None` edge incident to an inner detector is a true-
   boundary correction in that detector's lane.
2. Two inner endpoints in the same `(patch, basis)` lane form a local
   correction edge. This includes body--body, body--terminal,
   terminal--terminal, and cross-window edges.
3. An inner--yoke edge is globally owned and produces one yoke-port incidence
   in its local lane.
4. Inner endpoints in different lanes are globally owned guard ports. Each
   incident lane receives an immutable incidence referencing the same canonical
   edge ID and the other endpoint/lane. The selected cell currently has none,
   but this ownership rule is already fixed.
5. Yoke--boundary, yoke--yoke, missing-role, and every other unlisted topology
   fail compilation.

Only correction edges may enter support. They require finite, strictly
positive weight and zero observable mask. Ports require finite, strictly
positive weight so saturation is defined, retain their possibly nonzero mask,
and never enter support.

The returned ownership partition is exactly
`local-correction(owner_lane)` or `global-port(owner_lane=None)`. Unsupported
topologies cause compilation to fail and therefore never appear in a returned
partition. One global-port edge may have multiple local incidence records;
those incidences are not additional ownership classes.

The current canonical compiler rejects some non-yoke cross-patch edges before a
UF projection can see them. Add an opt-in all-frame canonical compile policy for
UF that preserves the existing default behavior and fingerprints for ProMatch
and Pinball while retaining cross-lane edges as globally owned canonical edges.
Test both the unchanged default and the UF opt-in path.

### 3.4 Selected-cell compile assertions

The experiment factory authenticates this exact physical cell before compiling
the projection:

```text
distance                         7
patches                          6
yokes                            2 (X and Z)
rounds                           28 (= 4d)
circuit style                    cz
noise model                      si1000
physical error probability       0.003
remove_x_yoke                    false
DEM decompose_errors             true
DEM approximate_disjoint_errors true
```

The following are development expectations, not authenticated protocol values:

```text
detectors                         8,354
observables                          12
canonical edges                  40,836
canonical nonzero-frame edges     1,392 (all guard ports)
lanes                                12
detectors per lane                  696
internal corrections per lane     3,171
true-boundary corrections/lane       116
ports per lane                       116
global correction edges           39,444
global guard-port edges            1,392
```

Implementation commit A must regenerate these values and all circuit, DEM,
catalog, graph, and projection fingerprints. A mismatch blocks protocol freeze;
the implementation must not alter expectations merely to make a test pass.

### 3.5 Projection fingerprint

Hash a versioned canonical serialization containing at least:

- canonical graph fingerprint;
- complete ordered detector role table;
- ordered lane keys, detector IDs, coordinates, and local indices;
- every correction edge and its owner;
- every port incidence, including local/remote detector and lane IDs, kind,
  exact weight, mask, and canonical edge ID;
- exact weight-table representation;
- topology, positivity, zero-frame, boundary, and cross-lane policies; and
- projection schema version.

No set, dictionary iteration order, DSU root identity, or process-dependent
hash may influence the fingerprint.

## 4. UF engine contracts

### 4.1 Runtime inputs and outputs

The core entry point should have the semantic shape:

```python
run_lane(
    lane: PatchUFLaneProjection,
    local_syndrome: ReadOnlyLocalSyndrome,
    policy: PatchUFPolicy,
    *,
    capture: CaptureMode,
    workspace: LaneWorkspace,
) -> LaneOutcome
```

`local_syndrome` contains only lane-owned bits or sorted active local vertex
indices. It cannot expose the global detector vector.

Use an explicit outcome hierarchy:

```text
LaneOutcome
    status = empty | completed | censored
    completed_components
    censored_components
    lane_counters
    censor_reason

ComponentOutcome
    absorbed_membership
    original_defects
    forest_support
    peeled_support
    exact_margin
    gate_decision and all gate reasons
    structural metrics

PatchTransactionOutcome
    status
    local component references
    durable component references
    durable support, boundary, and frame
    abort reason

ShotCorrection
    patch outcomes
    durable support, boundary, and frame
    shot summary
```

Keep `gate_decision` separate from `durable_decision`. A locally eligible
component remains locally eligible in telemetry even if its sibling lane later
causes a patch-wide abort.

These are semantic shapes, not a requirement that `CaptureMode.NONE` return
public component records. The core retains the minimum internal membership,
support, margin, and decision state needed to construct and validate
`ShotCorrection`; `NONE` discards that state after the durable shot plan is
built instead of copying it into telemetry objects.

### 4.2 Exact arithmetic

The semantic reference uses `fractions.Fraction` created from each weight's
exact `float.as_integer_ratio()`. The production representation must be chosen
and frozen before it becomes the scientific decoder. The preferred candidate
is a normalized dyadic pair:

```text
value = signed_integer * 2**binary_exponent
```

Required operations are exact construction from binary64, normalization,
addition, subtraction, comparison, minimum, doubling/halving when needed, and
canonical numerator/denominator serialization. The production type must be
differentially tested against `Fraction`, including subnormal values, very
different exponents, equality after different operation sequences, and exact
threshold equality.

No production decision may call `float()`, compare with a tolerance, round a
weight, or depend on platform floating-point contraction.

### 4.3 Reference growth algorithm

Implement the slow reference first in
`src/yoked/decoding/_patch_uf_reference.py`:

1. Initialize every lane detector as a singleton. Only original nonzero
   syndrome vertices contribute defect parity.
2. Mark a component active exactly when its defect parity is odd and it has not
   reached a true boundary.
3. Grow every unconsumed outgoing incidence of an active component at unit
   rate. An internal correction edge therefore closes at rate two with two
   active sides, rate one with one active side, and rate zero with no active
   side. Boundary and port incidences have only one local side.
4. Compute the next exact event time and collect every correction, boundary,
   and port incidence saturated from the same pre-event state.
5. Preflight the entire simultaneous batch against reference-comparable
   semantic budgets. If any such cap would be exceeded, preserve the state
   after the previous complete batch and return a censored result. Heap and
   representation-memory caps are adjudicated only by the frozen production
   lifecycle described in Section 4.6.
6. Select and record forest edges atomically from the saturated correction
   batch in canonical order, union distinct roots, XOR defect memberships, and
   OR inherited boundary and taint flags.
7. Apply simultaneous boundary contacts and port taints to the post-union
   components. A port is consumed once, freezes at its canonical weight, does
   not union, and does not neutralize an active component.
8. Recompute component parity and activity only after the complete event batch.
9. Continue until no active component remains or an active component has no
   future finite event.
10. Define final components only at termination. Ignore untouched,
    syndrome-free singletons.
11. Extract each final component's forest from the edges recorded during its
    committed event batches and reverse-peel it to the exact claimed original-
    defect boundary.
12. Enumerate all confidence competitors, calculate exact slack, and apply the
    strict frozen gate.

The unresolved simultaneous-boundary representation and forest-selection
details belong to the experiment specification's simultaneous-event freeze
blocker. Resolve them in Phase 0; do not let container iteration choose them.

### 4.4 Deterministic forest and peeling

For every simultaneous correction batch, use a canonical Kruskal-style order:

```text
(event_time, exact_weight, normalized_global_endpoints, edge_id)
```

Ordering chooses a spanning representation only. A saturated edge omitted from
the forest remains a zero-slack confidence competitor. True-boundary virtual
leaves must have a separately frozen selection rule when several contacts occur
in one component at the same time.

Peeling uses a deterministic reverse order over the selected forest. It must:

- return a square-free set of canonical correction edge IDs;
- reproduce exactly the component's original-defect boundary, allowing the
  selected virtual boundary incidence where appropriate;
- never emit a port;
- never emit one canonical edge twice; and
- fail fatally on an impossible parity or malformed forest.

### 4.5 Confidence and eligibility

After lane termination, enumerate each final component's competing set from
compiled incidence tables, not from the selected forest alone. Include:

- every nonforest correction edge incident to the component, including
  internal cycles and edges to another final component;
- every unselected incident true-boundary edge; and
- every incident guard port.

Compute the exact minimum slack. An empty set is the explicit value
`infinity`. Negative slack is fatal. Zero slack captures simultaneous choices
and port contact. Commit eligibility uses strict `margin > tau`; equality
defers.

A completed component is eligible only when it is neutralized, its peeling
boundary is correct, its support is local/unique/zero-frame, it is untainted,
its margin passes, and all budgets were respected. `tau` has no code default;
it must be supplied by a validated policy object.

### 4.6 Budget and counter model

Freeze the production queue lifecycle before scientific use. The production
engine records the exact counters defined in the experiment specification:

- growth events and simultaneous batches;
- union attempts, successes, and failures;
- heap pushes, pops, stale pops, and total operations;
- forest-edge count and the forest-edge cap;
- peel operations;
- peak heap size and peak live-component count; and
- nonvirtual `absorbed_vertex_count` as the component-size budget; and
- configured temporary-memory allocation classes and units.

Budget enforcement occurs before committing a simultaneous batch. A rejected
batch applies and counts none of its state changes. Report every cap whose
rejected next value would be exceeded and choose the lexicographically first
cap only as the diagnostic primary reason.

Reference and production mathematical outputs and semantic-cap censoring must
agree. Heap and representation-memory counters/censoring are production-policy
behavior and compare against frozen production golden traces, not against the
reference's data-structure operations. The test harness must not demand a
reference heap count that the reference algorithm does not have.

### 4.7 Production engine

Implement `src/yoked/decoding/_patch_uf.py` only after the reference is stable.
The expected implementation uses:

- array-backed union-find with union by size/rank and deterministic semantic
  representatives independent of physical root choice;
- exact root growth tags and vertex-to-root offsets;
- a versioned event heap whose entries can be invalidated lazily;
- canonical batch collection for all events sharing the exact minimum time;
- preallocated or generation-stamped shot/lane workspaces; and
- immutable compact outcomes separated from optional trace materialization.

The optimization may change data structures but not event or decision
semantics. Every optimization lands with a differential test that would fail if
event grouping, component membership, forest support, margin, or routing
changes.

## 5. Transactions and decoder adapter

### 5.1 Pure support replay

Provide one pure helper over sorted square-free canonical edge IDs:

```text
replay_support(edge_ids) -> detector_boundary, observable_frame, exact_weight
```

It validates ownership, uniqueness, endpoint bounds, tail bits, and mask width.
Use it for component validation, patch validation, shot validation, replay, and
tests. The deployed gate may call this pure algebra helper; it may not call the
full-graph oracle.

### 5.2 Patch transaction

For each patch:

1. Run both basis lanes from the immutable original shot.
2. Finish every local final-component decision.
3. If either lane is censored or locally incomplete, create an aborted patch
   outcome and make no support durable.
4. Otherwise collect only eligible complete components.
5. Require component supports to be pairwise disjoint.
6. Combine supports by symmetric difference.
7. Independently replay aggregate boundary, frame, ownership, and weight.
8. Require the aggregate V1 frame to be zero.
9. Mark the patch transaction durable without mutating the live residual.

Only `budget-exhaustion-patch-abort` and
`local-incomplete-neutralization-patch-abort` are ordinary patch aborts.
Confidence and port decisions are component-level defers. Algebra, support,
topology, reference, packing, and frame violations are fatal exceptions.

### 5.3 Shot planning and application

Factor the adapter into two explicit stages:

```python
plan_shot(local_syndromes, *, capture) -> ShotCorrection
apply_shot_correction(original_packed_shot, correction) \
    -> residual_packed_shot, durable_frame
```

For a public batch, plan one shot at a time into a preallocated residual batch,
then call the matcher once for that complete batch. Scalar input is a one-shot
batch. Empty input returns a correctly shaped empty prediction without calling
the matcher.

UF-Shadow invokes the same `plan_shot` implementation as treatment and differs
only after the validated `ShotCorrection` exists: shadow discards it and sends
the original packed input to the backend, while treatment applies it to form
the residual.

### 5.4 Capture modes

Use a small explicit enum:

```text
NONE     production and latency hot path; no retained component records
METRICS  immutable compact counters and completed/censored summaries
TRACE    bounded replay detail, forest/event/port provenance
```

Every decision, algebra check, budget counter, and gate computation runs in all
modes. Capture only controls copying and retention. Experiment-level component
IDs that contain stage, corpus, and global shot identity are added by the
collector after the decoder returns structural results.

### 5.5 Packed adapter behavior

`src/yoked/decoding/_patch_uf_decoder.py` must:

- validate `uint8`, rank, width, unused detector tail bits, and immutability;
- accept scalar-through-wrapper, empty, contiguous, and noncontiguous batches;
- normalize input without mutating caller memory;
- compile one complete graph and one projection per DEM;
- make the top-level decoder factory pickleable for Sinter/spawn;
- either inherit the non-pickleable compiled matcher through the frozen fork
  collector path or reconstruct it deterministically inside a spawned worker;
- keep actual observables completely outside the API;
- return little-endian packed predictions with clear unused tail bits; and
- provide separate untimed helpers for backend-original and precomputed
  backend-residual latency workloads.

The treatment decoder name is fixed. A source-level constant should hold it,
but it must enter `custom_decoders()` only after the scientific policy literals
exist. At that point export it through
[`src/yoked/decoding/__init__.py`](../src/yoked/decoding/__init__.py) and update
exact registry-name, factory-pickling, fork, and spawned-reconstruction tests
without modifying old frozen protocol JSON files.

### 5.6 Required capture-record contract

The versioned schemas must implement the complete field lists and formulas in
Sections 12.2.1--12.2.2 of the experiment specification. Keep these four
levels separate:

1. **Completed component record.** Retain sorted absorbed detector membership,
   sorted original-defect membership, the normative defect count, forest and
   peeled-support IDs/counts/exact weights, defect and absorbed time/geometry
   extrema and spans, last membership event time, maximum incident charge,
   merge and event-batch ancestry, boundary/port state, exact margin, local
   gate reasons, and durable transaction outcome.
2. **Censored partial-component record.** Retain only the last-complete-batch
   membership, lower-bound defect count, current forest/geometry/ancestry,
   boundary/port/charge state, lane telemetry reference, censor reason, and
   exceeded-cap set. Peeling, support, margin, gate, and durable fields are
   explicitly null and the record never enters a completed histogram.
3. **Lane record.** Store exactly one row per `(shot, patch, basis)`, including
   `empty|completed|censored`, every production counter and peak, last complete
   batch ID, censor data, and ordered component references. Counters are not
   duplicated into each component row.
4. **Shot record.** Retain original, lane-owned, residual, and committed defect
   counts; completeness; completed/eligible/committed/deferred/tainted/boundary
   and censored counts; component maxima with the specified zero/null rules;
   patch activity/commit/abort counts; lane-counter sums and peak max/sums; and
   the exact sparse completed-component defect-count histogram.

All geometry comes from authenticated layout coordinates. Store half-integral
spatial coordinates as exact doubled integers `local_x2` and `y2`; spans are
integer maximum-minus-minimum values. Store exact weights, times, charges, and
margins canonically without JSON `NaN` or numeric infinity.

The decoder returns structural shot-local identities. The experiment layer
adds `stage_id`, corpus digest, global shot ID, deterministic component ID, and
membership digest. Component identity is derived from the canonical sorted
absorbed membership, not DSU root identity. Schema snapshot tests must fail if
any normative field is omitted, renamed, duplicated at the wrong level, or
given the wrong null-versus-zero rule.

## 6. Planned source layout

| Path | Responsibility | Phase |
| --- | --- | ---: |
| `src/yoked/decoding/_dem_catalog.py` | Shared flattened unmerged-DEM catalog validation extracted without changing Pinball behavior | 1 |
| `src/yoked/decoding/_promatch_graph.py` | Opt-in retention of cross-lane canonical/global edges; default behavior and old decoder fingerprints unchanged | 1--2 |
| `src/yoked/decoding/_patch_uf_graph.py` | Terminal-inclusive projection, ports, exact weights, ownership, fingerprint, support replay | 2 |
| `src/yoked/decoding/_patch_uf_reference.py` | Slow exact semantic reference and paper-compatibility diagnostic | 3 |
| `src/yoked/decoding/_patch_uf.py` | Production weighted UF, heap/DSU, forest, peeling, confidence, budgets, compact outcomes | 4--5 |
| `src/yoked/decoding/_patch_uf_decoder.py` | Four adapter paths, patch transactions, residual construction, packed I/O | 6 |
| `src/yoked/decoding/_patch_uf_experiment.py` | Fixed-shot paired collection, range workers, ledgers, corpus, replay candidates | 8 |
| `src/yoked/decoding/_patch_uf_stats.py` | Paired, workload, coverage, grouped cluster bootstrap statistics | 9 |
| `src/yoked/decoding/_patch_uf_latency.py` | Fixed-corpus timing workload, five pairs, cyclic schedules, fresh restarts | 10 |
| `src/yoked/decoding/_patch_uf_analysis.py` | Validation, reconciliation, reports, tables, plots, casebook selection | 9--10 |
| `tools/benchmark_patch_uf_mwpm` | Smoke, probe, freeze, collect, verify, latency commands | 8--11 |
| `tools/analyze_patch_uf_mwpm` | Accuracy/workload/cluster/latency analysis and finalization commands | 9--11 |
| `docs/PATCH_UF_MWPM_D7_P003_DRAFT.json` | Resolved draft policy and experiment protocol | 0--11 |
| `docs/PATCH_UF_MWPM_D7_P003_FROZEN_V1.json` | Config-only frozen protocol generated after implementation commit A | 12 |

Mirror each implementation module with a focused file under
`tests/yoked/decoding/`. Keep the new experiment isolated from frozen ProMatch
collectors, analyzers, schemas, and output roots.

## 7. Ordered decoder implementation phases

### Phase 0 -- Resolve semantic and schema blockers

**Goal:** turn every implementation-affecting ambiguity into a literal,
testable policy field before the optimized engine is treated as versioned.

Tasks:

- [ ] Define versioned `PatchUFPolicy`, projection, outcome, counter, and error
      schemas.
- [ ] Choose and document the exact production dyadic representation.
- [ ] Freeze simultaneous correction/boundary/port event encoding.
- [ ] Freeze behavior for multiple simultaneous true-boundary contacts.
- [ ] Freeze production heap entry, versioning, replacement, invalidation, and
      stale-pop lifecycle.
- [ ] Define all deterministic budget counters, units, proposed-operation
      preflight rules, and multi-cap reporting.
- [ ] Classify budgets as semantic/reference-comparable or
      production-lifecycle-specific; define a golden production adjudicator for
      heap and representation-memory caps.
- [ ] Define the opt-in canonical compile policy that retains every fixed
      cross-lane guard-port edge without changing existing compiler defaults.
- [ ] Decide whether the core remains topology-generic while the experiment
      factory fails closed outside the authenticated two-yoke cell.
- [ ] Define capture schemas and null-versus-zero rules, including distinct
      lane-, component-, and shot-level simultaneous-batch fields.
- [ ] Create a draft protocol skeleton that rejects missing `tau`, budgets,
      seed roots, bins, affinity, and source inventory instead of supplying
      defaults.

`tau`, scientific histogram/display bins, seed roots, and host affinity remain
experiment freeze blockers; they must not be selected from outcomes. Unit tests
may use clearly named fixture policies that cannot be mistaken for the future
scientific configuration.

The frozen V1 JSON uses the same exact-field `BudgetLimits` schema for both
budget classes. Fields that do not belong to a class are literal `null`, not a
default: semantic limits carry only graph/algorithm counters, while production
limits carry only heap-lifecycle and temporary-workspace counters. The engine
checks both classes before committing a production batch; the reference checks
only the semantic class.

**Exit gate:** every algorithmic decision needed for reference/production
agreement is literal and golden-testable; unresolved scientific values are
required fields rather than implicit defaults.

### Phase 1 -- Shared foundations with no decoder behavior change

**Goal:** isolate reusable validation safely before adding UF policy.

Tasks:

- [ ] Extract flattened unmerged-DEM catalog parsing/validation from the
      Pinball-private implementation into `_dem_catalog.py`.
- [ ] Preserve Pinball V2 errors, accepted catalogs, outputs, and existing
      graph/schedule/decoder fingerprints through characterization tests; the
      current validator itself has no fingerprint.
- [ ] Add a UF catalog schema that retains ordered mechanism multiplicity,
      probability hex, normalized detector boundary, and observable mask.
- [ ] Freeze the exact parallel-mechanism merge-equivalence rule, recompute the
      effective canonical probability/weight under that rule, and reject any
      boundary group whose multiplicity, probability, or mask cannot be
      represented losslessly by the canonical edge.
- [ ] Fingerprint the complete unmerged catalog and merge policy.
- [ ] Define strict little-endian mask/tail-bit helpers needed by UF without
      weakening existing adapters.
- [ ] Add a pure canonical support-replay helper or its patch-local precursor.
- [ ] Add unit tests for parallel mechanisms, conflicting frames,
      non-graphlike components, duplicate endpoints, and boundary mechanisms.

**Exit gate:** existing Pinball, ProMatch, and full repository tests pass with
no registered decoder output change.

### Phase 2 -- Canonical terminal-inclusive projection

**Goal:** compile the complete static runtime graph and authenticate ownership.

Tasks:

- [ ] Compile the layout and all-frame canonical matching graph once.
- [ ] Use the UF opt-in global-edge policy so same-patch cross-basis and future
      cross-patch lane edges survive canonical compilation as global edges.
- [ ] Build lane membership directly from body/terminal roles.
- [ ] Compile doubled integer geometry and exact time coordinates.
- [ ] Classify every canonical edge according to Section 3.3.
- [ ] Compile separate typed tables for internal correction edges,
      true-boundary incidences, and guard ports.
- [ ] Compile compact CSR incidence arrays and exact dyadic weights.
- [ ] Compile global detector-to-lane/local-index and edge-ownership tables.
- [ ] Generate the deterministic projection fingerprint.
- [ ] Reproduce and independently inspect selected-cell counts.

Tests:

- exact-one ownership and adjacency cardinality;
- terminal and cross-window inclusion;
- true-boundary versus port distinction;
- strict positive finite local/port weights;
- zero frame for every correction edge and retained masks for ports;
- remote-blind lane interface;
- source/endpoint orientation independence;
- ordering/fingerprint stability under input iteration permutation;
- fail-closed unknown topology and ambiguous unmerged mechanism cases; and
- unchanged legacy compiler behavior/fingerprints when the UF opt-in is absent;
- selected-cell role, coordinate, count, and fingerprint fixtures.

**Exit gate:** a read-only compiled projection contains exactly twelve lanes,
classifies every edge, and is suitable for pickling before any UF algorithm is
introduced.

### Phase 3 -- Exact semantic reference

**Goal:** establish correctness independently of optimized data structures.

Tasks:

- [ ] Implement exact incidence charges and active/inactive component growth
      with `Fraction`.
- [ ] Implement atomic equal-time event discovery and application.
- [ ] Implement correction unions, boundary inheritance, permanent port taint,
      and local-incomplete termination.
- [ ] Implement deterministic forest selection and reverse peeling.
- [ ] Implement final-component competing-set enumeration and exact margin.
- [ ] Implement gate results without patch transactions.
- [ ] Produce compact semantic traces suitable for golden comparison.
- [ ] Add a paper-compatibility diagnostic mode.

The compatibility diagnostic uses one ordinary surface-code graph, equal
weights, no ports, no confidence rejection, no budgets, commit-all valid
components, standard rough-boundary handling, and no residual MWPM. It is a
test oracle, not an experimental arm. It should verify that the UF core reduces
to paper-style cluster growth and peeling without claiming identical choices in
degenerate forest ties.

Tests:

- hand-built chains, stars, cycles, disconnected graphs, and boundary cases;
- rate-two, rate-one, and no-growth events;
- neutral component later reached by an odd component;
- correction/boundary/port mixed ties;
- multiple simultaneous unions and forest alternatives;
- taint propagation across later merges;
- strict threshold equality and infinity margin;
- claimed-boundary peeling and impossible-peel rejection;
- input and edge-order permutation invariance; and
- exhaustive positive small graphs with bounded rational weights.

**Exit gate:** the reference is the authoritative semantic oracle and has no
dependency on PyMatching output, actual observables, the full-graph oracle, or
production heap behavior.

### Phase 4 -- Production exact weighted UF

**Goal:** match the reference with data structures suitable for latency work.

Tasks:

- [ ] Implement and test the exact dyadic type.
- [ ] Implement array-backed DSU, semantic component metadata, and exact growth
      potentials.
- [ ] Implement the frozen versioned event heap and stale-entry handling.
- [ ] Implement exact simultaneous-batch collection and transactional preflight.
- [ ] Implement deterministic forest state and terminal component extraction.
- [ ] Add generation-stamped workspace reuse and empty/inactive-lane fast paths.
- [ ] Expose production counters independently from optional retained records.

Differential comparison must cover:

```text
event times and simultaneous batches
final absorbed memberships and original-defect sets
boundary and port-taint state
forest and peeled support edge IDs
exact margins and threshold comparisons
gate reason sets and primary reasons
semantic-cap censor snapshots and rejected batch semantics
```

Use exhaustive tiny graphs plus deterministic randomized positive graphs.
Persist the random seeds in tests. Exercise root-choice and edge-order
permutations so semantic IDs cannot depend on physical DSU roots.

**Exit gate:** zero reference/production mathematical or semantic-cap
differences across the full differential suite, plus exact production-only
heap/memory censor behavior against golden lifecycle traces. Performance
observations cannot waive a mismatch.

### Phase 5 -- Budgets, metrics, and component gate

**Goal:** complete production behavior and make telemetry reconcile without
changing decisions.

Tasks:

- [ ] Enforce every frozen cap at the proposed-operation and atomic-batch
      boundary.
- [ ] Golden-test heap, union, event, forest-edge, peel, component-size,
      temporary-memory, and peak counters and cap boundaries.
- [ ] Materialize completed and censored component records with correct
      terminal-only null fields.
- [ ] Calculate membership digests and geometry outside the `NONE` hot path.
- [ ] Implement multi-label gate reasons and exclusive primary precedence:
      `port-tie > port-yoke > port-cross-lane > below-threshold > eligible`.
- [ ] Preserve exact completed sibling-lane rows even when another lane censors.
- [ ] Reconcile `union_attempt = successful + failed` and
      `heap_operations = pushes + pops` at lane and shot levels.

**Exit gate:** `NONE`, `METRICS`, and `TRACE` produce identical semantic
decisions; all counter and censor identities reproduce from retained records.

### Phase 6 -- Patch transactions and packed decoder adapters

**Goal:** produce a correct complete YSC decoder and the three timing controls.

Tasks:

- [ ] Implement pure patch proposal validation and durable transaction creation.
- [ ] Implement one-shot planning without residual mutation.
- [ ] Implement packed batch residual construction and frame composition.
- [ ] Implement direct Global MWPM, Adapter-Control, UF-Shadow, and treatment
      callables through one shared adapter layer.
- [ ] Instrument a test-only matcher spy proving one backend call per nonempty
      treatment batch.
- [ ] Implement parameterized treatment construction and defer normative
      `custom_decoders()` registration until the Phase 11 literals are fixed.
- [ ] Add ordinary and telemetry APIs that share the same proposal/gate code.
- [ ] Make the top-level factory pickleable; test fork inheritance and the
      deterministic spawned-worker reconstruction path separately from the
      non-pickleable compiled matcher.
- [ ] Define and test explicit workspace/reentrancy behavior.

Tests:

- independent component commits survive unrelated ordinary defers;
- budget/incomplete patch aborts make no syndrome/frame change;
- an eligible sibling retains its local gate result after patch abort;
- duplicate/overlapping/ineligible support and algebra mismatch are fatal;
- no-durable shots bit-match Global MWPM;
- the Global-MWPM wrapper bit-matches the maintained built-in/direct
  `pymatching` decoder for scalar, batch, empty, and noncontiguous inputs;
- controls bit-match Global MWPM on all valid inputs;
- residual formula and frame XOR for synthetic nonzero test frames, even though
  V1 durable candidates are zero-frame;
- input immutability and unused packed-tail validation;
- scalar, empty, noncontiguous, wrong-width, wrong-dtype, and malformed cases;
- ordinary/metrics/trace prediction identity; and
- registry exact-set, factory pickling, fork, and spawned-process behavior.

**Exit gate:** the decoder is semantically complete and usable outside the
experiment harness; no accuracy, workload, or speed claim follows from this
gate.

### Phase 7 -- Physical and full-stack deterministic validation

**Goal:** challenge graph projection and algebra with real maintained DEMs.

Tasks:

- [ ] Exhaust every complete single flattened decomposed-DEM mechanism on small
      maintained circuits, including mechanisms that canonical graph merging
      would otherwise obscure.
- [ ] Select and freeze two-mechanism cases covering body, terminal,
      cross-window, true-boundary, yoke-port, cancellation, and simultaneous
      tie behavior.
- [ ] Independently replay every durable support and complete prediction
      formula.
- [ ] Compare boundary, frame, and weight calculations with the full-graph
      oracle in shadow tests only.
- [ ] Run two identical deterministic 32-shot smokes under `$TMPDIR` and
      compare bytes and digests.
- [ ] Run the entire pinned test suite with one native numerical thread.

**Exit gate:** all deterministic gates pass twice, with no fatal invariant,
input mutation, or reference disagreement.

## 8. Performance implementation strategy

Latency is an empirical endpoint, not an assumed property. Optimize only after
the reference and differential suite are stable.

### 8.1 Hot-path rules

- Compile graph, projection, exact weights, incidence arrays, and matcher once.
- Skip lanes with no local detector events.
- Avoid constructing Python objects for all 8,354 vertices or 40,836 edges per
  shot; use compact arrays and touched/generation lists.
- Keep `Fraction`, dictionaries, per-event dataclasses, strings, hashing, and
  serialization out of the production `NONE` path.
- Reuse scratch buffers per compiled worker only under an explicit
  non-reentrant contract, or allocate a caller-owned workspace; test the chosen
  lifecycle.
- Apply durable detector boundaries directly to a packed residual copy where
  practical instead of unpacking/repacking the full vector repeatedly.
- Preallocate the public residual batch and call `decode_batch` once.
- Keep component membership expansion, remote-port telemetry joins, digests,
  and replay traces outside ordinary and timed decode calls.

### 8.2 Development microbenchmarks

Before the scientific timing harness, use non-claim-bearing scratch
microbenchmarks to record:

- compile time and resident-size change;
- empty-shot, sparse-shot, and dense-shot frontend latency;
- time by lane growth, peeling/gate, patch validation, residual construction,
  and backend call;
- heap events, stale-pop ratio, touched vertices, and workspace peak;
- `NONE` versus `METRICS` versus `TRACE` overhead; and
- batch sizes 1, 64, and 1,024.

These diagnostics may guide implementation optimization but cannot select the
scientific confidence threshold or alter sampled shots. Because PyMatching
still receives a fixed complete graph, a sparser syndrome may not materially
reduce backend latency; the five-pair timing suite must measure this rather
than presuppose it.

## 9. Test organization

Planned focused test files:

```text
tests/yoked/decoding/_dem_catalog_test.py
tests/yoked/decoding/_patch_uf_graph_test.py
tests/yoked/decoding/_patch_uf_reference_test.py
tests/yoked/decoding/_patch_uf_reference_differential_test.py
tests/yoked/decoding/_patch_uf_core_test.py
tests/yoked/decoding/_patch_uf_decoder_test.py
tests/yoked/decoding/_patch_uf_fault_order_test.py
tests/yoked/decoding/_patch_uf_experiment_test.py
tests/yoked/decoding/_patch_uf_stats_test.py
tests/yoked/decoding/_patch_uf_latency_test.py
tests/yoked/decoding/_patch_uf_analysis_test.py
```

### 9.1 Test layers

| Layer | Purpose | Must avoid |
| --- | --- | --- |
| Pure unit | Exact arithmetic, ownership, DSU transitions, peeling, gate, GF(2) algebra | Real-time sleeps and random unseeded data |
| Exhaustive/property | Small-graph reference/production equivalence and permutation invariance | Float tolerances |
| Physical mechanism | Real DEM roles, terminal/boundary/yoke behavior, frame replay | Outcome-based gate tuning |
| Adapter | Packed shapes, tails, immutability, one matcher call, control equality | Actual-observable access |
| Process | Factory pickle, fork inheritance, spawned reconstruction, fixed thread settings, deterministic worker ranges | More than 32 workers |
| Artifact | Resume, no-clobber, tamper, cross-digest, reconciliation | Mutation of historical corpora |
| Latency | Fake clock, exact timer scope, order balance, cyclic workload joins | Telemetry or I/O inside timing |

### 9.2 Standard verification commands

Run from the repository root in the pinned environment:

```bash
source .venv/bin/activate
export TMPDIR=/data2/s2chitni/.tmp
export MPLCONFIGDIR="$TMPDIR/yoked-surface-codes-matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
unset MAX_ERRORS

python -m pytest -q tests/yoked/decoding/_patch_uf_graph_test.py
python -m pytest -q tests/yoked/decoding/_patch_uf_reference_test.py \
    tests/yoked/decoding/_patch_uf_reference_differential_test.py
python -m pytest -q tests/yoked/decoding/_patch_uf_core_test.py \
    tests/yoked/decoding/_patch_uf_decoder_test.py
python -m pytest -q tests/yoked/decoding/_patch_uf_fault_order_test.py
python -m pytest -q tests/yoked/decoding
python -m pytest -q
```

All scratch output belongs under `$TMPDIR`; scientific outputs use a new
versioned `out/` root only after protocol freeze. Never set `MAX_ERRORS` for
the paired experiment.

## 10. Experiment infrastructure phases

These phases begin only after the decoder gates in Section 7 pass. They are
included because the decoder's capture and adapter APIs must support the frozen
experiment without later semantic rewrites.

### Phase 8 -- Paired collector and durable artifacts

- [ ] Implement immutable 32-range sampling for fixed `N=1,000` and
      `N=10,000`, with one native thread per process.
- [ ] Use the exact worker partition
      `start=floor(N*w/32), stop=floor(N*(w+1)/32)`: the 1,000-shot stage has
      eight 32-shot and twenty-four 31-shot ranges; the 10,000-shot stage has
      sixteen 313-shot and sixteen 312-shot ranges.
- [ ] Derive seeds with the frozen named cryptographic hash over experiment,
      stage, cell, worker/range, and purpose; keep sampler, timing-order,
      bootstrap, and replay roots unrelated.
- [ ] Precompile the selected graph in the parent and use a copy-on-write fork
      model where supported; never compile the 40k-edge graph independently in
      every worker without measuring the cost.
- [ ] Sample each packed detector/observable pair once and decode the two
      accuracy arms on byte-identical detector input.
- [ ] Keep actual observables unavailable until both predictions are immutable.
- [ ] Freeze public microbatch size, ordering, subdivision, and tail behavior
      in the protocol; one treatment backend call applies per public
      microbatch, not per worker range.
- [ ] Store exactly twelve lane telemetry records per physical shot.
- [ ] Join remote/yoke port bits only after gate and transaction outcomes are
      immutable; changing those joined bits must not change any decoder record
      produced before the join.
- [ ] Store completed component records and censored lower-bound records with
      distinct schemas and denominators.
- [ ] Keep detailed traces bounded to selected replay candidates and regenerate
      final traces in fresh processes.
- [ ] Implement component-file-first, per-shot-shard-last atomic installation,
      with the shard acting as the range commit marker.
- [ ] Bind each per-shot shard to the component file's relative path, digest,
      component count, exact shot range, and per-shot component-ledger
      range/count cross-digest. Resume accepts a range only when both installed
      files exist and mutually authenticate every one of those fields.
- [ ] Reject `MAX_ERRORS`, gaps, overlaps, mutation, digest drift, foreign
      protocols, malformed orphans, and partial resume state.
- [ ] Outside timed intervals, run Adapter-Control and UF-Shadow over every
      indexed row of both accepted 1,000- and 10,000-shot corpora; persist
      counts/digests proving equality with Global MWPM.
- [ ] Likewise persist corpus-wide equality of ordinary, telemetry, and timing
      treatment callables.
- [ ] Keep collection and analysis in separate write-once sibling directories;
      analysis never modifies corpus, shards, protocol, or manifest files.
- [ ] Regenerate an orphan component file deterministically and reuse it only
      after byte-for-byte and digest equality; otherwise fail closed without
      overwrite or repair in place.

**Exit gate:** the same seed reproduces byte-identical shards; injected crashes
and tampering fail closed; a 32-shot smoke repeats bit-for-bit.

The planned scientific root is `out/cguf_mwpm_d7_p003_v1/`, with disjoint
`shakeout_1k_collection`, `shakeout_1k_analysis`,
`characterization_10k_collection`, `characterization_10k_analysis`,
`latency_collection`, and `latency_analysis` subroots. Freeze each subroot's
required-file set before sampling.

### Phase 9 -- Statistics, reconciliation, replay, and reporting

- [ ] Reuse maintained paired contingency and interval routines under new
      source identities and golden tests.
- [ ] Implement Global-MWPM/treatment `a,b,c,d`, marginal exact intervals, and
      paired Tango interval.
- [ ] Freeze and golden-test two-sided `alpha=0.05`: Clopper--Pearson endpoints
      call the maintained one-sided routine with `alpha=0.025`; Tango uses
      `alpha=0.025`, tolerance `1e-12`, at most 200 iterations, and the
      specified `z**2/(N+z**2)` zero-discordance boundary.
- [ ] Emit no hypothesis-test p-value and make no equivalence/noninferiority
      decision from the exploratory corpus.
- [ ] Produce the required accuracy breakdowns by observable, activation,
      commit counts, complete-shot cluster bins, routing reason, original and
      residual workload, and yoke/patch observable disagreement masks.
- [ ] Implement exact `(H,R,L,K)` workload/coverage reconciliation and the
      frozen complete-shot bootstrap.
- [ ] Implement exact sparse component-size histograms, shot-maximum
      distributions, censor separation, geometry, and routing views.
- [ ] Persist the exact sparse joint component histogram over defect count,
      gate decision/reason set/primary reason, durable decision/reason,
      boundary flag, and port-kind set, pooled and split by patch/basis; keep
      censored lower bounds in a separate histogram.
- [ ] Resample whole physical-shot component groups for every cluster interval;
      never resample components independently.
- [ ] Implement deterministic bounded replay selection and fresh-process replay.
- [ ] Emit the three descriptive, non-gating volume tags:
      `baseline_failures_lt_200`, `discordant_pairs_lt_100`, and
      `durable_commit_shots_lt_500`.
- [ ] Implement confidence-bin acceptance, regression/recovery association,
      risk/coverage curves, exact threshold-equality counts, and splits by port,
      boundary, component size, and workload; label all as descriptive.
- [ ] Validate component, lane, shot, shard, run, replay, and analysis digests
      before producing any table or plot.
- [ ] Print raw numerators and denominators beside every conditional rate and
      cluster display.

The workload and coverage implementation must use these exact complete-shot
quantities and identities:

```text
H_i = original global detector-event count
R_i = residual detector-event count
L_i = original detector events owned by the twelve lanes
K_i = defect count in durable components

workload_ratio          = sum(R_i) / sum(H_i)
workload_mean_difference = mean(R_i - H_i)
frontend_coverage       = sum(K_i) / sum(H_i)
lane_owned_coverage     = sum(K_i) / sum(L_i)

0 <= K_i <= L_i <= H_i
R_i = H_i - K_i
frontend_coverage = 1 - workload_ratio
```

Implement every routing-rate numerator and denominator from the specification.
Use 10,000 fixed-seed multinomial replicates of the exact `(H,R,L,K)` histogram
for workload/coverage intervals. A zero denominator is
`null/not-estimable`; do not serialize `NaN` or infinity. Use empirical type-7
2.5% and 97.5% quantiles, persist the estimable replicate count, and suppress a
ratio interval unless all 10,000 replicates are estimable.

Cluster reporting has two noninterchangeable populations:

- the exact component-weighted histogram over all completed final components,
  including completed siblings of a censored lane; and
- the shot-weighted distribution of the largest final-component defect count,
  conditional on `cluster_summary_complete=true`.

For both, report count/denominator, p50, p90, p95, p99, and observed maximum.
Implement exact-size PMF/count plots, CCDFs, committed/deferred overlays,
defect-versus-absorbed/time/spatial tables and frozen-bin heatmaps, exact-size
commit fractions, and a separately denominated censored lower-bound display.
Cluster intervals use 10,000 complete-shot bootstrap replicates carrying each
selected shot's entire component group. Never resample components independently;
if any endpoint has fewer than all 10,000 estimable replicates, report no
interval and persist the estimable count. Apply the same grouped bootstrap to
the predeclared display-bin proportions and label their 95% intervals as
pointwise, not simultaneous.

Replay retains at most 100 cases per category:

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

Ordinary categories select the lowest rooted SHA-256. Metric categories sort
their named metric descending and use the rooted digest as the tie-break. The
frozen payload includes original/actual packed data, both predictions,
residual, tentative/durable supports, event/forest/peel/port/gate/transaction
detail, and every relevant digest; actual observables are attached only after
decoder outputs are immutable.

The analyzer enforces these identities at shot, shard, and run levels:

```text
completed_final_component_count
    = committed_component_count + durable_deferred_component_count
sum(completed-size histogram counts)
    = completed_final_component_count
sum(size * completed-size histogram count)
    = sum(cluster_defect_count over completed rows)
committed_defect_count
    = sum(cluster_defect_count over durable committed rows)
union_attempt_count
    = successful_union_count + failed_union_count
heap_operation_count
    = heap_push_count + heap_pop_count
```

Within each lane, successful unions reproduce the sum of merge ancestry over
terminal completed components or current censored components. Shot additive
counters, maximum lane peaks, and summed lane peaks reproduce the exact twelve-
row lane table. Stored maxima reproduce component rows on complete summaries;
final-component maxima are null on incomplete summaries, while the censored
lower-bound maximum reproduces censored rows. Censored rows contribute to no
completed histogram, quantile, gate-decision count, or durable total. Finally,
`K`, `L`, `H`, and `R` satisfy every identity above and in the workload block.

**Exit gate:** synthetic complete corpora analyze successfully; missing,
duplicate, overlapping, tampered, inconsistent, and zero-denominator fixtures
produce the specified rejection or `null/not-estimable` behavior.

### Phase 10 -- Controlled latency harness

- [ ] Define these six timed variants as data:
      `global_mwpm`, `adapter_control`, `uf_shadow`, `treatment`,
      `backend_original`, and `backend_residual`.
- [ ] Load the authenticated 10,000-shot characterization detector corpus in
      every fresh restart; do not resample it.
- [ ] Timing workers load detector and authenticated residual inputs only; they
      never load the characterization actual-observable corpus.
- [ ] Materialize and authenticate the aligned treatment-residual corpus once
      outside all timed intervals.
- [ ] Implement generic `TimedVariant` and `TimedPair` data rather than
      ProMatch-specific conditionals.
- [ ] Implement the five fixed pairs and balanced `AB`/`BA` block schedules.
- [ ] Implement batch-specific warmup and call counts for 1, 64, and 1,024.
- [ ] Within each `(restart, batch_size)`, run the full scheduled warmup for
      every one of the six named variants exactly once in frozen order before
      any pair; do not perform pair-specific re-warmup.
- [ ] Build a read-only extended corpus for allocation-free cyclic wrapped
      slices.
- [ ] Define `workload_key=(characterization_corpus_digest, global_shot_id)` and
      prove `corpus_index` is a bijection to global shot ID. Byte-identical
      detector vectors at different shot IDs remain distinct workloads.
- [ ] Persist `timing_call_id`, restart, batch size, pair, side, block, call
      index, shared schedule-table range, ordered `(corpus_index, workload_key)`
      list, detector digest, and precomputed-summary digest for every call plan.
- [ ] Reject missing or duplicate precomputed rows, non-bijective corpus
      indices, unexpected duplicate/reordered call keys, and any detector or
      summary digest mismatch.
- [ ] Derive and persist cyclic starting offsets from the frozen hash of
      `(restart, batch_size, pair, block)`; paired sides use the identical
      ordered workload plan, including wraps.
- [ ] Run the protocol-specified batch-dependent serialized fresh-process
      restarts with one native thread, frozen affinity/NUMA and GC policy, and
      no concurrent simulation.
- [ ] Freeze the initial GC policy as disabled throughout warmup and timing;
      setup/teardown may manage GC only outside measured intervals.
- [ ] Record CPU model/topology/microcode, affinity/NUMA, OS/kernel, Python and
      package versions, governor/turbo state when observable, host-load
      snapshot, thread variables, GC policy, clock identity, and all raw
      durations.
- [ ] Reject a whole restart on overlap, affinity/host-policy drift, mutation,
      or invalid timing. Replace it under the identical frozen restart index,
      seeds, schedules, and offsets; never cherry-pick individual calls.
- [ ] Time the total interval directly from public decoder-adapter entry through
      packed-prediction return. It includes production validation,
      packing/conversion, UF growth, gate, residual construction, matcher, and
      frame composition as applicable.
- [ ] Time backend-only directly from immediately before the PyMatching
      invocation through its return. Do not reconstruct total time from
      component sub-timers.
- [ ] Keep circuit/DEM generation, graph compilation, corpus loading, input or
      residual generation, retained telemetry, provenance, logging, file I/O,
      and analysis outside both timer scopes.
- [ ] Keep prediction checks, telemetry, workload joins, hashing, logging, and
      I/O outside the timer.
- [ ] Implement restart-then-block hierarchical bootstrap and pair-context-
      specific distributions.

The five direct pairs are:

| Pair | Numerator | Denominator |
| --- | --- | --- |
| Net total | `treatment` | `global_mwpm` |
| Adapter cost | `adapter_control` | `global_mwpm` |
| UF/gate cost | `uf_shadow` | `adapter_control` |
| Residual application | `treatment` | `uf_shadow` |
| Backend relief | `backend_residual` | `backend_original` |

The initial non-claim-bearing schedule is intentionally bounded and varies by
batch size:

| Batch size | Restarts | Paired blocks/restart | Warmup calls per variant/restart | Timed calls per side/block |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 20 | 50 | 10 |
| 64 | 5 | 4 | 5 | 2 |
| 1,024 | 3 | 2 | 1 | 1 |

This is approximately 107,000 timed variant-shot equivalents before warmup.
It supports a quick first estimate while retaining the strongest replication
for the batch-1 endpoint. Any larger claim-bearing timing campaign must use a
new versioned protocol and must not pool silently with this initial suite.

Batch 1 is the primary latency and tail-latency endpoint. Batches 64 and 1,024
are throughput diagnostics whose tails remain per-batch quantities.

For every `(batch_size, pair, side)` context, report raw median, p90, p95, and
p99 nanoseconds and aggregate throughput as:

```text
batch_size * number_of_calls * 1e9 / sum(call_duration_ns)
```

For each pair, report the geometric mean of paired block-total ratios and the
ratio of pooled empirical type-7 side p99 values. Use 10,000 hierarchical
bootstrap replicates, resampling restarts and then paired blocks while retaining
all calls in a block. Report two-sided 95% percentile intervals and persist
`inference_scope=fixed_characterization_corpus_recorded_host`. Calls are never
bootstrapped as independent observations.

Precomputed workload/cluster joins remain outside timing. Within each separate
`(batch_size, pair, side)` context, retain call count, unique workload-key
count, and median/p95/p99 latency. Batch-1 tables separately stratify by:

```text
cluster-summary status
exact maximum final-component defect count (complete shots only)
committed defect count
growth-event count
successful-union count
heap-operation count
peel-operation count
residual detector count
```

Censored batch-1 calls form their own status group and never receive a final-
cluster-size label. Batch 64/1,024 persist aligned completeness, maximum final
cluster, maximum censored lower bound, committed-defect, growth, union, heap,
peel, and residual-detector batch summaries as descriptive covariates only.

**Exit gate:** fake-clock tests prove timer scope; all controls and treatment
callables pass untimed corpus-wide equality; every restart/pair/block/call is
present, positive, balanced, and digest-valid.

## 11. Protocol and launch phases

### Phase 11 -- Resolve the scientific freeze blockers

Before even the 1,000-shot shakeout, record literal values and rationale for:

- confidence threshold `tau` as binary64 hex with strict `>` comparison;
- production arithmetic and simultaneous-event serialization;
- every operation and temporary-memory budget;
- confidence histogram and cluster display bins;
- independent 256-bit roots for shakeout, characterization, timing order,
  bootstrap, and replay;
- timing CPU affinity and NUMA placement;
- every versioned artifact schema; and
- the complete hashed source inventory.

Implement a launch manifest that records and verifies the protocol and A/B
commits, clean-worktree state, command line, thread environment, resolved CPU
affinity, available memory/storage, start/end timestamps, every shard hash, and
the final ordered or Merkle digest. Timestamps are provenance only and never
enter deterministic seeds or decision identities.

After these literals are present and validated, bind the normative treatment
name in `custom_decoders()` and complete the registry, factory-pickling, fork,
and spawned-reconstruction tests. No fixture policy may be reachable under
that name.

No value may be inferred from environment defaults or selected from shakeout or
characterization accuracy direction. The draft protocol validator must reject
`TBD`, missing fields, unknown fields, noncanonical ordering, or a self-hash
mismatch.

### Phase 12 -- Implementation/config two-commit freeze

1. Commit all implementation, tests, tools, this plan, the final experiment
   specification, dashboard, and resolved draft protocol as implementation
   commit A.
2. From a clean worktree at A, rerun the complete test suite, twice-identical
   32-shot smoke, and independent 100-shot storage/runtime probe under
   `$TMPDIR`.
3. Generate the frozen protocol from those authenticated artifacts.
4. Commit exactly `docs/PATCH_UF_MWPM_D7_P003_FROZEN_V1.json` as config commit
   B.
5. Verify that `git diff --name-only A..B` contains only that path and that the
   worktree is clean.
6. Run the 1,000-shot shakeout from B. Proceed to the disjoint 10,000-shot
   characterization only if every operational gate in the experiment
   specification passes.

The user authorized this conditional workflow on 2026-08-30.
There is no numerical accuracy, activation, workload, or speedup launch gate.
Any implementation, threshold, budget, arithmetic, schema, or protocol change
after the shakeout requires a new version and fresh disjoint seeds.

## 12. Failure taxonomy

The implementation must keep three classes distinct:

| Class | Examples | Action |
| --- | --- | --- |
| Ordinary component defer | `below-threshold`, `port-yoke`, `port-cross-lane`, `port-tie` | Leave that component for residual Global MWPM; preserve independent eligible components |
| Ordinary patch abort | `budget-exhaustion-patch-abort`, `local-incomplete-neutralization-patch-abort` | Make no tentative component in that patch durable; preserve local gate telemetry |
| Fatal invariant | Invalid/nonpositive weight, unsupported topology, unmerged catalog ambiguity, impossible peel, duplicate/overlapping support, boundary/frame mismatch, nonzero V1 candidate frame, reference disagreement, malformed packed data/output, input mutation, corrupted artifacts | Fail the call or run; never relabel as decoder failure or fallback |

The collector may not drop a crashed shot, substitute Global MWPM after a
fatal error, or reduce the denominator. A failed range remains absent until the
same deterministic range is rerun successfully.

## 13. Risk register

| Risk | Consequence | Mitigation/gate |
| --- | --- | --- |
| Reusing `DomainGraph` | Terminal and cross-window context silently missing | Role-derived terminal-inclusive projection and selected-cell census |
| Treating all boundaries as one root | Independent boundary components merge incorrectly | Distinct virtual-leaf/type representation and boundary golden tests |
| Reading remote yoke bits | Gate gains forbidden global information | Lane-local input type; post-decision join only; mutation-independence test |
| PyMatching merges parallel mechanisms | Local frame/ownership ambiguity is hidden | Validate flattened unmerged DEM catalog before projection |
| Zero or nonfinite edge weight | Undefined or immediate event semantics | Projection requires finite strict positivity |
| Floating-point event comparison | Nondeterministic ties and changed decoder | Exact reference and exact production dyadic differential tests |
| Event-type ordering within a tie | Different unions, taint, or boundary state | One pre-state event set and atomic-batch golden traces |
| DSU root choice leaks into IDs | Nondeterministic artifacts | Membership-derived component identity and root/edge permutation tests |
| Mutate-then-rollback transaction | Partial correction leaks into residual | Immutable planning plus apply-once durable boundary |
| Telemetry changes decisions or latency | Invalid prediction/timing comparison | Capture-mode equality and fake-clock scope tests |
| Per-shot Python allocation dominates | Frontend slower than baseline | Compiled arrays, touched-state workspaces, microbenchmarks |
| Sparse syndrome does not speed fixed matcher | Workload drops without latency benefit | Measure backend-original/residual and full treatment/control pairs |
| Unbounded component traces | Memory/storage failure | Compact metrics for all shots; bounded trace replay only |
| Old frozen experiment mutation | Provenance loss | New modules/protocol/output root; never edit `PROMATCH_*` artifacts |
| Outcome-driven `tau` or budget tuning | Biased characterization | Pre-sampling literal freeze and new-version requirement for changes |

## 14. Recommended development commit sequence

These are development checkpoints, not the scientific A/B freeze:

1. **Foundations:** shared catalog validator and strict support algebra, with no
   decoder behavior change.
2. **Projection:** terminal-inclusive lanes, ports, exact weights, fingerprints,
   and selected-cell census.
3. **Reference:** exact growth, ties, forest, peeling, gate, and compatibility
   diagnostic.
4. **Production core:** dyadic arithmetic, DSU/heap, budgets, counters, and
   reference differential suite.
5. **Decoder vertical slice:** patch transactions, packed treatment, controls,
   registration, physical-fault tests, and a small smoke.
6. **Telemetry and collection:** capture records, artifacts, fixed ranges,
   resume, replay, and storage probe.
7. **Analysis and latency:** statistics, plots, fixed-corpus timing, and complete
   validation.
8. **Implementation commit A:** resolved draft protocol and all readiness gates.
9. **Config-only commit B:** frozen protocol only.

Keep each checkpoint reviewable and preserve unrelated user changes. Do not
commit generated scratch data, `__pycache__`, or outputs under `$TMPDIR`.

## 15. Immediate next actions

The first implementation pass should stop after a narrow, testable vertical
slice rather than beginning with the experiment harness:

1. Resolve the Phase 0 simultaneous-boundary, exact-arithmetic,
   queue-lifecycle, budget-category, and topology-scope decisions, and specify
   the opt-in canonical compiler path for the already-fixed cross-lane port
   policy.
2. Extract and lock the shared unmerged-DEM validator.
3. Implement `_patch_uf_graph.py` and authenticate the twelve-lane selected-cell
   projection.
4. Implement the `Fraction` reference on hand-built graphs.
5. Implement one end-to-end synthetic shot through peeling, gate, patch
   transaction, residual construction, and a matcher spy.
6. Only then optimize the UF engine and expand to real DEM fault tests.

This ordering creates an early semantic checkpoint: the graph ownership and
residual algebra can be reviewed before event-heap performance work makes the
implementation harder to inspect.

## 16. Local references

- [Delfosse and Nickerson, *Almost-linear time decoding algorithm for
  topological codes*](https://quantum-journal.org/papers/q-2021-12-02-595/)
  -- source for the Union-Find clustering and peeling core used by the
  compatibility diagnostic; it does not define this hybrid frontend.
- [`experiments/CONFIDENCE_GATED_UF_MWPM_D7_P003.md`](../experiments/CONFIDENCE_GATED_UF_MWPM_D7_P003.md)
  -- normative V1 experiment, metrics, latency, artifact, and freeze contract.
- [`docs/PROMATCH_IMPLEMENTATION_PLAN.md`](../docs/PROMATCH_IMPLEMENTATION_PLAN.md)
  -- prior paired statistics, provenance, and readiness precedent; its windowed
  policy and claim thresholds do not transfer to UF.
- [`docs/PINBALL_INTEGRATION_PLAN.md`](../docs/PINBALL_INTEGRATION_PLAN.md)
  -- residual-decoder, terminal-layer, physical-fault, and differential-testing
  precedent.
- [`src/yoked/decoding/_promatch_layout.py`](../src/yoked/decoding/_promatch_layout.py)
  -- maintained detector roles and layout compilation.
- [`src/yoked/decoding/_promatch_graph.py`](../src/yoked/decoding/_promatch_graph.py)
  -- canonical edge table and complete PyMatching backend.
- [`src/yoked/decoding/_pinball_v2_decoder.py`](../src/yoked/decoding/_pinball_v2_decoder.py)
  -- unmerged-catalog and one-residual-batch implementation precedent.
- [`src/yoked/decoding/_artifact_io.py`](../src/yoked/decoding/_artifact_io.py)
  -- strict artifact loading, output fencing, and atomic installation.
- [`AGENTS.md`](../AGENTS.md) -- environment, process/thread, scratch-space,
  clean-worktree, and immutable-corpus requirements.
