# L1 ProMatch-Style Decoder: Airtight First-Round Implementation and Experiment Plan

> **Status:** living design document for the ProMatch-L1 experiment series.
> Sections describing frozen protocols are retained verbatim for provenance and
> are not rewritten after a freeze. For the outcome and current status of each
> campaign, see `experiments/README.md` (status dashboard) and the frozen
> protocol manifests indexed in `docs/README.md`.
> **Last updated:** 2026-08-19.

## 1. Decision Summary

The first experiment will test a **local L1 predecoder followed by the existing
flat joint residual matcher**. It is not yet a fully hierarchical L1/L2 decoder:
the residual decoder still sees the complete yoked detector graph and returns
all logical observables in one call.

The claim-bearing comparison is:

```text
U0-direct: flat, uncorrelated PyMatching
PU-window: d-round, patch-and-basis-local ProMatch-style predecode
           + the same flat, uncorrelated PyMatching backend
```

The repository's current correlated decoder is retained as context:

```text
C0: flat, correlated PyMatching
```

`PU-window` versus `U0-direct` isolates the effect of adding the predecoder to
an otherwise identical uncorrelated decoder. It does **not** establish an
improvement over the repository's correlated stack. That later claim requires
`PC` versus `C0`, where `PC` is a correlation-aware predecode composition.

Four decisions make the first round auditable:

1. The production scientific variant operates on explicit `d`-round L1
   windows. A full-history variant is retained only for functional/offline
   characterization; its `HW=10` result is not interpreted as hardware
   capacity or real-time L1 coverage.
2. The primary variant may commit only zero-observable-frame, domain-local
   paths. Observable-bearing local decisions are a separately named ablation.
3. Accuracy, workload, residual-backend latency, and end-to-end software
   latency are separate endpoints with separate claim gates.
4. Pilot/discovery data select and power one measurable accuracy cell. A
   disjoint, fixed-size holdout is analyzed using a frozen protocol only if the
   V3 pilot passes every preregistered unsigned viability gate.

The experiment is falsifiable. Ordinary MWPM already finds a minimum-weight
correction under its graph model. A greedy prefix can improve empirical logical
accuracy only through model mismatch or changed degeneracy/tie behavior; it can
also make accuracy worse. The plan therefore tests for accuracy preservation
first and treats accuracy improvement as a stronger, separately gated result.

## 2. Claims and Primary Estimands

The work must never collapse several outcomes into a vague statement that
"ProMatch is better." Each supported statement has its own evidence:

| Claim | Primary estimand | Required interpretation |
| --- | --- | --- |
| Accuracy preservation | Paired difference in any-observable shot-failure probability, `Delta` | One-sided non-inferiority must pass. |
| Accuracy improvement | The same paired `Delta` | The upper one-sided confidence bound must be below zero. |
| Workload improvement | Ratio of unconditional mean detector events delivered to the residual backend | This is backend work relief, not automatically latency. |
| Residual-backend latency improvement | Ratio of residual-matcher wall time on residual versus original syndromes | This excludes the predecoder cost. |
| End-to-end software-latency improvement | Ratio of adapter-entry-to-return wall time, `PU-window` versus `U0-direct` | This includes all predecoder and conversion overhead. |
| Real-time/hardware improvement | Not estimated in round one | Requires a native/hardware implementation and streaming deadline model. |

The primary accuracy outcome is:

```text
Y = 1 iff any predicted observable differs from any actual observable in a shot.
```

Per-observable failures are secondary. They cannot replace the primary outcome
after results are seen.

The first round has two scopes:

- A **measurable stress/validation cell**, selected by the frozen pilot rule in
  Section 15, supports a combined accuracy/performance tradeoff statement.
- A fixed **target-geometry performance cell**
  `(d=11, patches=6, yokes=2, r=44, p=0.001)` measures activation, workload,
  and latency only. Direct Monte Carlo at this cell is not expected to power an
  accuracy-preservation claim.

Results from these two cells must not be spliced into a claim that accuracy was
preserved at the target cell.

## 3. Supported-Circuit and Error-Model Contract

### 3.1 Supported circuits

Version 1 supports only circuits produced by the maintained
`yoked_magic_memory_circuit` path with:

- 1D patch placement and `pitch = d + 1`;
- `style="cz"` for claim-bearing experiments;
- `remove_x_yoke=False`;
- two observables per inner patch;
- `yokes in {0, 1, 2}` for tests, and `yokes=2` for confirmatory experiments;
- integer `rounds >= 2`, with `rounds % d == 0` for the windowed decoder.

Squareberg circuits, arbitrary user circuits, `remove_x_yoke=True`, missing
coordinates, and layouts that merely look similar are unsupported in `v1`.
The benchmark harness must assert generator metadata before compilation. Since
a Sinter decoder receives a DEM rather than the original generator call, the
decoder can enforce only the structural checks below; it cannot infer
`yoked_magic_memory_circuit` provenance or `style="cz"` from coordinates. Sinter
is therefore an integration path unless its DEM hash is allowlisted by a frozen
protocol/sidecar. Claim-bearing runs use the paired harness, which has both the
circuit metadata and the exact DEM.

For the standard observable layout:

```text
num_observables = 2 * num_patches
observable_owner(k) = k // 2
```

Any future frame-bearing local variant must verify that every nonzero fault ID
belongs to the path's patch. The primary zero-frame variant never relies on
that mapping to make a correction.

### 3.2 Exact DEM construction

The paired harness is the authoritative scientific path. It must construct one
DEM per circuit using exactly:

```python
dem = circuit.detector_error_model(
    decompose_errors=True,
    approximate_disjoint_errors=True,
)
```

The identical serialized DEM is supplied to both accuracy decoders in the
paired run (`U0-direct` and `PU-window`).
The circuit text and DEM text are hashed with SHA-256 and recorded. DEM
construction failure is fatal; there is no fallback to different decomposition
flags.

The Sinter workflow remains an integration path. Before collecting scientific
results, fixed-shot tests must show:

1. the harness `U0-direct` bit-matches Sinter's built-in `pymatching` decoder;
2. the contextual `C0` implementation bit-matches Sinter's built-in
   `pymatching-correlated` decoder; and
3. every compared decoder receives the same DEM hash.

`C0` should be instantiated from the built-in decoder where possible. If it is
reproduced directly, compilation must use `enable_correlations=True` and decode
must use `enable_correlations=True`, followed by the bit-for-bit equivalence
test.

## 4. Detector Roles and L1 Windows

### 4.1 Compile-time role assignment

Every detector is assigned exactly one role:

```python
L1BodyDetector(
    patch_id: int,
    check_basis: Literal["X", "Z"],
    time: int,
    window_id: int,
)
L1TerminalDetector(
    patch_id: int,
    check_basis: Literal["X", "Z"],
    time: int,
)
YokeDetector()
```

The maintained circuit emits detector coordinates `(x, y, t, ...)`. The layout
compiler uses `dem.get_detector_coordinates()` and requires:

- exactly one coordinate record for every detector ID;
- at least three finite coordinate values per detector;
- inner-detector time coordinates integral within a documented tolerance;
- yoke/dummy-yoke detectors at the reserved spatial coordinate `y=-2`;
- inner coordinates at the expected half-integer lattice positions;
- contiguous patch IDs and an identical local spatial layout in every patch;
- complete and consistent time-layer multiplicity.

For the maintained geometry, the spatial mapping is equivalent to:

```python
inner_coords = [c for c in coords.values() if not is_close(c[1], -2)]
min_y = min(c[1] for c in inner_coords)
max_y = max(c[1] for c in inner_coords)
d = round(max_y - min_y)

assert is_close(min_y, -0.5)
assert is_close(max_y, d - 0.5)
assert d >= 2

if is_close(y, -2):
    role = YokeDetector()
else:
    pitch = d + 1
    patch_id = floor((x + 0.5) / pitch)
    local_x = x - patch_id * pitch
    assert -0.5 <= local_x <= d - 0.5
    assert is_close(2 * local_x, round(2 * local_x))
    assert is_close(2 * y, round(2 * y))
    assert is_close(local_x + y, round(local_x + y))
    check_basis = "X" if int(round(local_x + y)) & 1 == 0 else "Z"
```

The basis calculation deliberately uses `local_x`, not global `x`, so an
even-distance patch translated by the odd pitch `d+1` cannot silently swap
basis labels.

For the maintained magic-boundary circuit, inner detector times must be exactly
`{0, ..., r}` where `r=max(time)`. The noisy body is `0 <= time < r`; the magic
terminal-readout layer is `time=r`. With `r % d == 0`:

```python
window_id = time // d       # only for 0 <= time < r
num_windows = r // d
```

Thus every claim-bearing predecode unit is:

```text
(patch_id, check_basis, window_id)
```

and contains exactly `d` body detector layers. Terminal inner detectors and all
yoke detectors are left unchanged for the global residual decoder.

This is an offline emulation of parallel `d`-round L1 units, not yet a streaming
implementation with live arrival and deadlines. Cross-window graph edges remain
available to the global residual matcher but are never committed locally.

If any structural assumption fails, compilation raises a descriptive error. It
must never silently fall back to a global predecoder.

### 4.2 Why full-history is not claim-bearing

The full-history prototype groups `r=4d` or `8d` rounds into one local syndrome
and then applies a single `HW=10` budget. Its activation and forced reduction
grow with experiment duration, so the budget no longer models one bounded L1
decoder invocation. Full-history results may validate correction composition
and expose accuracy hazards, but they cannot establish ProMatch capacity,
real-time coverage, or per-window latency.

The full-history variant is therefore named explicitly and confined to Phase 0:

```text
promatch-l1-v1-fullhistory-hw10-stages1234-noboundary-zeroframe-pymatching
```

No Figure 8 or hardware-style latency conclusion may be based on this variant.

## 5. Decoder Variants and Controls

The decoder/control namespace distinguishes the following fixed variants; the
V3 claim-bearing comparison itself collects only `U0-direct` and `PU-window`:

| Label | Definition | Role |
| --- | --- | --- |
| `U0-direct` | Packed syndrome directly into ordinary uncorrelated PyMatching | Causal baseline and deployable software baseline |
| `U0-wrap` | Unpack, classify/traverse, repack unchanged syndrome, then the same PyMatching call | Measures interface and Python traversal overhead |
| `PU-window` | `d`-round L1-window ProMatch stages 1-4, `HW=10`, no boundary, zero frame, then the same PyMatching call | Primary treatment |
| `PU-boundary` | Parity-aware actual-boundary prototype described in Section 9 | Deferred exploratory design; V3 does not require or collect it, and it is not claim-bearing |
| `PU-full` | Full-history version of the primary algorithm | Phase-0 diagnostic only |
| `C0` | Sinter `pymatching-correlated` | Context only |

The primary registered treatment name is:

```text
promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching
```

The reserved exploratory boundary-comparator name and implemented no-op control
are:

```text
promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching
pymatching-u0-wrap-v1-windowd
```

Every behavior-changing parameter appears in the name or forces an algorithm
version bump: domain scope, window rule, HW limit, enabled stages, boundary
policy, frame policy, and backend.

The following comparisons answer different questions:

```text
PU-window vs U0-direct  -> deployable end-to-end effect
PU-window vs U0-wrap    -> predecode cost after controlling adapter overhead
residual vs original    -> downstream matcher relief on paired syndromes
PU-window vs C0         -> descriptive only; not a causal predecoder comparison
```

## 6. Matching-Graph Compilation

Build one ordinary matcher from the shared DEM:

```python
matcher = pymatching.Matching.from_detector_error_model(dem)
matcher.ensure_num_fault_ids(dem.num_observables)
```

The predecoder graph is compiled from `matcher.edges()` rather than independently
reconstructing probabilities from the DEM. This makes the predecoder use the
same uncorrelated edge merging, weights, and fault IDs as its residual backend.

Normalize each edge into an immutable record:

```python
@dataclass(frozen=True)
class Edge:
    edge_id: int
    source: int
    target: int | None
    weight: float
    observable_mask: bytes
    source_role: DetectorRole
    target_role: DetectorRole | None
```

`observable_mask` has exactly `ceil(num_observables/8)` little-endian bytes.
Fault IDs are sorted before packing, every ID must satisfy
`0 <= id < dem.num_observables`, and unused high bits in the last byte must be
zero.

Before assigning IDs, require the complete normalized edge records to be
unique, then sort by:

```text
(normalized_source, normalized_target_or_boundary, weight, observable_mask)
```

Exact duplicate normalized records are rejected instead of being ordered by
their input enumeration. This makes IDs independent of `matcher.edges()`
iteration order. The compiled-graph fingerprint includes the normalized table,
domain assignments, package versions, and configuration.

Before applying any frame policy, call an edge a `domain_local_candidate` when
both endpoints are nonterminal L1 body detectors with identical
`(patch_id, check_basis, window_id)`. Compilation rejects:

- detector IDs outside `[0, dem.num_detectors)`;
- observable IDs outside `[0, dem.num_observables)`;
- NaN, infinite, or negative eligible-edge weights;
- self-loops;
- inconsistent normalized endpoints or duplicate normalized records;
- any non-boundary edge that crosses patches without being incident to a yoke
  detector; within-patch cross-basis edges are permitted only as canonical,
  residual-only edges and never enter an L1 domain graph; and
- any `domain_local_candidate` edge with a nonzero observable mask in the
  primary-v1 configuration.

Zero-weight edges are allowed. Dijkstra uses a finalized-node set and a
deterministic lexicographic label, so zero-weight cycles cannot cause
nontermination. Unreachable pairs are omitted; they are never assigned a large
sentinel weight.

### 6.1 Eligible primary paths

An edge is eligible for `PU-window` only when:

1. both endpoints are nonterminal L1 body detectors;
2. both endpoints have identical `(patch_id, check_basis, window_id)`;
3. its observable mask is zero; and
4. it is not a matching-boundary, yoke, cross-patch, cross-basis,
   cross-window, or terminal edge.

A multi-edge path is eligible only if every vertex and every edge satisfies the
same domain restriction. All withheld edges remain in the flat residual
matcher.

The maintained one-yoke/YBerg DEMs contain genuine within-patch X-to-Z edges.
Graph compilation retains these edges in the canonical table and unrestricted
residual matcher, but excludes them from every `(patch, basis, window)` domain.
This exception is needed for structural support of `yokes=1`; it does not relax
the primary two-yoke experiment's locality rule or make a cross-basis edge
prematch-eligible.

The zero-frame rule is a strict locality guard for the primary experiment. It
prevents a greedy subgraph that intentionally omits yoke constraints from
committing a logical decision whose evidence belongs to the outer graph. On the
maintained two-yoke circuits this is expected to exclude no same-domain direct
edges, but that expectation is enforced, not assumed.

A later path-zero-frame ablation may admit nonzero edge masks only when the XOR
of the **entire path's** masks is zero; canceling masks must be handled
correctly. A separately named frame-bearing ablation may admit a nonzero path
mask only after checking observable ownership as specified in Section 3.1.

## 7. Exact ProMatch-Style State and Predicates

For one predecode domain `D`, let the immutable eligible detector graph be:

```text
G_D = (V_D, E_D)
```

For the current active set of fired detectors `A subseteq V_D`, define:

```text
N_A(v)   = {u in A | at least one direct eligible edge {u,v} exists}
deg_A(v) = |N_A(v)|
```

Degree counts unique active neighboring detectors, not parallel-edge
multiplicity. For a detector pair with multiple exposed edges, candidate
selection uses the lowest deterministic edge key.

Definitions:

```text
v is an existing singleton iff deg_A(v) = 0.

For candidate endpoints C={u,v}, the candidate creates a new singleton iff
there exists w in A\C such that:

    deg_A(w) > 0 and |N_A(w)\C| = 0.
```

This definition is the correctness oracle. A cached `dependent[v]` count may be
used as an optimization only if randomized tests prove it agrees with direct
recomputation.

Only selected endpoints are removed from `A`. Internal vertices of a selected
multi-edge path remain active even if they are fired: their two path incidences
cancel in the correction boundary.

## 8. Exact Candidate Stages

For a domain with `initial_hw > hw_limit`, repeat until `|A| <= hw_limit`:

1. **Isolated pair.** Select adjacent `u,v` with
   `N_A(u)={v}` and `N_A(v)={u}`.
2. **Safe adjacent pair.** If stage 1 has no candidate, select a direct adjacent
   pair that creates no new singleton. Prefer a pair with a degree-one endpoint,
   then any other safe adjacent pair.
3. **Existing singleton.** If stage 2 has no candidate and an existing singleton
   exists, match one singleton to a distinct active detector using the
   lowest-weight eligible path that creates no new singleton when its two
   endpoints are removed.
4. **Risky adjacent pair.** If stages 1-3 have no candidate, select a direct
   adjacent pair even if it creates new singletons. Retain the degree-one-first
   substage ordering from stage 2.

Every stage commits at most one pair before recomputing the active graph and
restarting at stage 1. The HW condition is checked immediately after every
commit so the algorithm never predecodes beyond the requested capacity.

Deterministic candidate keys are:

```text
stage 1:
    (edge_weight, min_endpoint, max_endpoint, edge_id)

stage 2 or 4:
    (substage, edge_weight, min_endpoint, max_endpoint, edge_id)

stage 3:
    (path_weight, singleton_id, other_endpoint,
     edge_count, edge_id_sequence)
```

Stage-3 Dijkstra runs on immutable `G_D`, not the active induced graph. Its two
endpoints must be distinct and active. Internal vertices may be active or
inactive, remain active after selection, and may not be boundary, yoke,
cross-domain, or repeated vertices. The selected path is simple. The
no-new-singleton predicate considers removal of the two endpoints only.

Dijkstra labels use:

```text
(total_weight, edge_count, edge_id_sequence)
```

so equal-weight routes are reproducible.

Every detector-to-detector commit removes exactly two active detectors. The
number of commits before success is therefore bounded by:

```text
ceil((initial_hw - hw_limit) / 2)
```

If no candidate exists before reaching the limit, the domain attempt rolls
back as specified in Section 10.

## 9. Deferred Boundary-Policy Exploration

The primary `PU-window` variant disables local boundary matches. This is an
experimental restriction, **not** an accuracy guarantee. Removing boundary
alternatives changes degrees, singleton classification, coverage, and candidate
confidence.

V3 does not include or require a `PU-boundary` comparator in its scientific
experiment. A future, separately versioned exploratory study may use the
following parity-aware design:

1. If a domain's initial active HW is odd, add one active virtual boundary
   vertex for that domain; otherwise add none.
2. Connect it only through actual `target=None` PyMatching edges incident to
   detectors in the domain. Never reinterpret a yoke or cross-window edge as a
   boundary.
3. A detector-to-boundary commit toggles only the real detector, deactivates the
   one virtual vertex, and includes the actual edge's observable mask.
4. The virtual vertex does not count toward reported detector HW.
5. The strict zero-frame boundary comparator admits only zero-mask boundary
   corrections. A frame-bearing boundary study uses another decoder name.

In that future comparator, the virtual vertex would participate in
active-neighbor sets and therefore change degree/singleton predicates. A direct
detector-to-virtual-boundary edge may be considered in stages 1, 2, and 4 with
the same deterministic keys. The virtual vertex is not a stage-3 endpoint or
internal vertex. If it has no actual eligible boundary edge, it cannot be
matched and the ordinary no-progress/rollback rule applies.

If such a domain reached the HW limit without using its virtual vertex, the
study would discard that unmatched bookkeeping vertex with no correction
contribution; all real residual detectors would still go to the global matcher.
That study would record the rule's frequency in comparator telemetry.

This proposed comparator would remain exploratory because a time-window
interface is not itself a physical matching boundary. Cross-window evidence
remains exclusively for the global residual decoder in V3.

## 10. Correction Algebra and Transactional Rollback

Represent selected corrections as edge incidence over GF(2). Let `P` be the
XOR, not concatenation, of all committed prematch edge-incidence vectors.
Selected paths may overlap; repeated edges cancel in `P`, including their
detector boundary and observable masks.

For every shot:

```text
residual_syndrome = input_syndrome XOR boundary(P)
final_frame       = frame(P) XOR residual_prediction
```

For the primary zero-frame variant:

```text
frame(P) = 0
```

Required invariants are:

```text
input = residual XOR boundary(P)
yoke input bits = yoke residual bits
terminal-inner input bits = terminal-inner residual bits
every selected path belongs to exactly one L1 window domain
unused packed observable bits = 0
```

`decision_weight` is the sum of selected path costs with multiplicity. It is
telemetry about greedy decisions, not the weight of the composite correction.
Optionally report:

```text
xor_support_weight = sum(weight[e] for edges with odd multiplicity in P)
```

Do not add `decision_weight` to the residual match weight and call the sum a
composite correction weight unless the residual edge support has also been
reconstructed and XORed.

Each domain is decoded into a private `DomainAttempt` containing tentative
syndrome changes, paths, frame, and telemetry. Nothing mutates global result
state until the attempt reaches its limit. On failure, discard the attempt;
do not try to reverse already committed global mutations.

Rollback is domain-local:

- the original domain syndrome is passed unchanged to software PyMatching;
- successful attempts in other domains remain committed;
- the shot remains decodable and receives a prediction; and
- the failed domain is counted as an L1 capacity overflow.

Telemetry distinguishes attempted from committed work so failed work is not
mistaken for useful coverage.

## 11. Core Data Model

The core is independent of Sinter:

```python
L1DomainKey = L1WindowDomain | L1FullHistoryDomain


@dataclass(frozen=True)
class PrematchedPath:
    domain: L1DomainKey
    stage: int
    endpoints: tuple[int, int | None]
    edge_ids: tuple[int, ...]
    decision_weight: float
    observable_mask: bytes


class FallbackReason(Enum):
    NO_CANDIDATE = "no-candidate"
    DISCONNECTED = "disconnected"
    BOUNDARY_UNAVAILABLE = "boundary-unavailable"


@dataclass(frozen=True)
class DomainPrematchStats:
    initial_hw: int
    attempted_residual_hw: int
    final_residual_hw: int
    attempted_stage_counts: tuple[int, int, int, int]
    committed_stage_counts: tuple[int, int, int, int]
    attempted_matches: int
    committed_matches: int
    status: Literal["below-limit", "success", "rollback"]
    fallback_reason: FallbackReason | None


@dataclass(frozen=True)
class PrematchResult:
    residual_syndrome: np.ndarray
    observable_frame: np.ndarray
    paths: tuple[PrematchedPath, ...]
    domain_stats: dict[L1DomainKey, DomainPrematchStats]
    decision_weight: float
    xor_support_weight: float
```

Retaining path and edge identities is required for invariant checking,
disagreement replay, boundary/frame ablations, and later correlation-aware
work.

An invariant violation is a fatal decoder/run error, never a rollback reason.
Rollback represents modeled lack of coverage; silently converting a software
bug into fallback would bias both accuracy and overflow measurements.

## 12. Sinter and Paired-Harness Adapters

Implement the pinned interfaces:

```python
class PromatchDecoder(sinter.Decoder):
    def compile_decoder_for_dem(self, *, dem):
        ...


class CompiledPromatchDecoder(sinter.CompiledDecoder):
    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data,
    ):
        ...
```

Decoder factories must be picklable for Sinter multiprocessing.

For every batch:

1. Leave the input array unmodified.
2. Unpack exactly `dem.num_detectors` with `bitorder="little"`.
3. Run independent domain attempts in deterministic domain order.
4. Verify yoke and terminal bits are unchanged.
5. Repack the complete residual syndrome.
6. Call the same residual matcher exactly once:

   ```python
   matcher.decode_batch(
       residual_packed,
       bit_packed_shots=True,
       bit_packed_predictions=True,
   )
   ```

7. XOR the packed prematch frame into the packed prediction.
8. Return `uint8` with shape
   `(shots, ceil(dem.num_observables / 8))` and zero unused high bits.

The `U0-wrap` adapter performs the identical unpack, layout traversal, and
repack path but commits no matches. It must return predictions bit-for-bit
identical to `U0-direct`.

The production Sinter adapter does not retain unbounded shot telemetry. The
paired harness records bounded aggregate counts and capped, replayable
regression, recovery, and rollback examples. The paired accuracy/workload
collector has exactly two arms: `U0-direct` and `PU-window`, decoded from the
same sampled arrays. `U0-wrap` is validated against `U0-direct` in tests and
used as an identity-adapter timing control; it is not collected as a third
accuracy arm.

`reproduce_fig8_1d` must place `src` on `PYTHONPATH` and add the pinned Sinter
registration option to collection:

```bash
--custom_decoders_module_function yoked.decoding:custom_decoders
```

The factory exposes `PU-window`, the deferred `PU-boundary` prototype,
`PU-full`, and `U0-wrap` under distinct names. The V3 scientific protocol
collects only `PU-window` versus `U0-direct`; merely exposing another adapter
does not make it a required comparator. Built-in `pymatching` and
`pymatching-correlated` remain the authoritative `U0-direct` and `C0`
implementations.

### 12.1 Paired batch sampling and resume

Stim's RNG stream is not treated as random-access by global shot index. Paired
sampling and exact resume instead use immutable, counter-seeded batches. Each
pilot/final protocol freezes:

```text
sampler_seed_root       # independent literal 256-bit value per data split
sample_batch_size       # fixed to 10,000 in round one
Stim version
batch IDs and exact shot counts
```

For batch ID `j`, derive:

```text
seed_j = first 8 digest bytes, interpreted as unsigned little-endian, of SHA256(
    sampler_seed_root || "stim-batch" || uint64_little_endian(j)
)
```

Create a fresh detector sampler with `seed_j`, sample exactly the frozen batch
size (or the predeclared final remainder) once with bit-packed detector and
observable outputs, and feed those same in-memory arrays to both paired arms,
`U0-direct` and `PU-window`.
Immediately record SHA-256 digests of both arrays plus shapes/dtypes in the
batch ledger. On resume, regenerate a whole missing batch from its seed and
verify its digest; never continue partway through an RNG stream. Pilot,
confirmatory, workload, and timing corpora use distinct seed roots.

All prospective split roots are literal fields in the pre-pilot protocol and
cannot be regenerated after inspecting pilot outcomes. The final protocol
selects the already committed holdout schedule and its required prefix of batch
IDs after `N_confirm` is determined.

Changing Stim version, batch size, seed derivation, batch boundaries, or a
recorded digest creates a different experiment ID. This gives paired decoders
identical shots and makes interrupted runs reproducible without assuming that a
single Stim stream is seekable.

## 13. Planned Repository Changes

| File | Change |
| --- | --- |
| `src/yoked/decoding/__init__.py` | Export custom decoder factories. |
| `src/yoked/decoding/_promatch_layout.py` | Validate the supported DEM geometry and compile detector/window roles. |
| `src/yoked/decoding/_promatch_graph.py` | Normalize/fingerprint PyMatching edges and build immutable domain graphs. |
| `src/yoked/decoding/_promatch.py` | Implement exact predicates, stages, path algebra, and transactional attempts. |
| `src/yoked/decoding/_promatch_decoder.py` | Implement `PU`, `U0-wrap`, and Sinter adapters. |
| `src/yoked/decoding/_promatch_stats.py` | Implement paired tables, confidence intervals, power checks, and schema validation. |
| `tests/yoked/decoding/_promatch_{layout,graph,core,decoder,experiment,stats,analysis,latency,latency_integration,latency_analysis}_test.py` | Unit, property, algebraic, packing, statistical, latency, and integration tests. |
| `tools/benchmark_promatch_l1` | Run paired accuracy/workload and controlled latency experiments. |
| `tools/analyze_promatch_l1` | Read frozen raw results and produce the preregistered analysis. |
| `docs/PROMATCH_PILOT_PROTOCOL.json` | Candidate grid, pilot seeds, gates, and selection rule frozen before pilot sampling. |
| `docs/PROMATCH_FIRST_ROUND_PROTOCOL.json` | Frozen machine-readable protocol created after the pilot and before holdout. |
| `reproduce_fig8_1d` | Register custom decoders and expose them via `DECODER`. |
| `REPRODUCING_FIG8_1D.md` | Document names, scope, commands, and limitations. |

No new runtime dependency is required unless the selected paired score-interval
implementation cannot be validated adequately with the pinned SciPy version.
Any dependency change must be committed before the holdout manifest is frozen.

## 14. Verification and Test Plan

### 14.1 Layout and support-contract tests

For representative `d`, patch counts, and `yokes in {0,1,2}`:

- every detector has exactly one role;
- inferred `d`, `r`, patch count, basis, time, and window are correct;
- each body window contains the expected local time layers;
- all yoke/dummy-yoke detectors are at `y=-2` and excluded;
- terminal detectors are excluded from predecoding;
- no eligible edge crosses patch, basis, window, terminal, or yoke boundaries;
- a same-domain nonzero-mask candidate causes primary-v1 compilation failure
  before eligibility filtering;
- observable ownership matches `k//2`;
- malformed, incomplete, nonfinite, ambiguous, Squareberg-like, and
  `remove_x_yoke=True` layouts fail compilation.

### 14.2 Graph and algorithm tests

- golden small-graph patterns derived from ProMatch Algorithm 1 and, where the
  behavior is unambiguous, the official reference implementation pinned to a
  recorded upstream commit;
- exact singleton predicate versus brute-force recomputation;
- degree counts unique neighbors rather than edge multiplicity;
- pass-through when `HW <= limit`;
- stage-1 ordering and immediate stop at the threshold;
- stage-2 degree-one priority and rejection of new-singleton candidates;
- stage-3 deterministic shortest path;
- stage-3 path through an active internal detector, which remains active;
- stage-3 unreachable endpoint;
- stage-4 risky fallback and subsequent restart at stage 1;
- equal and zero-weight tie-breaking;
- nonfinite/negative weights, self-loops, invalid detector IDs, and invalid
  fault IDs;
- overlapping paths and GF(2) edge/mask cancellation;
- an ineligible yoke or cross-domain edge can never enter a path;
- odd-HW behavior under no-boundary and boundary policies;
- a boundary-adjacent pattern whose stage classification changes when boundary
  handling changes;
- per-domain rollback leaks neither path nor frame;
- multiple successful domains whose future frame masks cancel;
- independence from the original `matcher.edges()` enumeration order.

Add randomized small-graph property tests. For every generated graph and active
set, compare optimized candidate selection with a brute-force oracle and assert:

```text
locality
determinism
termination
monotonic committed detector HW
input = residual XOR boundary(P)
transactional rollback
```

### 14.3 Packing and API tests

- detector and observable counts `0,1,7,8,9,16`;
- empty and non-contiguous batches;
- mixed pass-through/success/rollback shots;
- input arrays remain unchanged;
- exact dtype, shape, little-endian order, and unused-bit masking;
- one residual `decode_batch` call per batch;
- `U0-wrap == U0-direct == Sinter pymatching` bit-for-bit;
- contextual `C0` equals Sinter `pymatching-correlated` bit-for-bit;
- decoder factory pickles and works in a multiprocessing smoke run.

### 14.4 Circuit and correction-algebra tests

- small circuits with `yokes=0,1,2`;
- multiple patches and simultaneous activated domains;
- inner and yoke events in the same shot;
- cross-window and terminal events remain residual;
- all below-threshold shots match `U0-direct` exactly;
- a frame-bearing diagnostic path and canceling masks;
- diagnostic residual-correction reconstruction for small shots verifies that
  its boundary equals the residual syndrome;
- prematch and residual edge supports can be XORed to reproduce the complete
  correction boundary.

### 14.5 Fault-order tests

Treat one complete DEM `error(...)` instruction as one atomic fault mechanism.
Do not split components separated by `^`. XOR the instruction's complete
detector and observable signatures when composing faults.

Run two test modes:

1. **Production threshold:** enumerate low fault orders with `HW=10` and report
   exactly how many cases activate predecoding. A nonactivated enumeration is
   explicitly called vacuous for predecoder accuracy.
2. **Forced threshold:** repeat with small limits such as `0,2,4` so all stages,
   correction composition, and failure paths are exercised.

For tiny circuits, exhaust all combinations through the declared order and
record total/attempted combinations and the first failing fault order for both
`U0` and `PU`. A preserved-distance statement is allowed only when the relevant
fault order is exhaustive and there are no `PU`-only failures below the claimed
order. Sampled pairs/triples are labeled a malignant-set search, not a proof of
distance preservation.

### 14.6 Statistical-code tests

- fixed known paired tables;
- symmetry under swapping decoder labels;
- boundary cases with zero failures or zero discordance;
- confidence-bound monotonicity;
- simulated coverage of the selected paired risk-difference interval;
- power-calculation recovery on synthetic multinomial tables;
- family-wise correction tests where applicable;
- manifest/schema rejection on missing, duplicate, or mismatched shot ranges.

## 15. Discovery, Freeze, and Confirmatory Protocol

### 15.1 Phase 0: implementation gates

No natural-noise conclusion is reported until all Section 14 tests pass.
Phase 0 includes:

- synthetic stage activation with forced small HW limits;
- full-history correction-composition diagnostics;
- fixed-shot equivalence for `U0-direct`, `U0-wrap`, and Sinter built-ins;
- a small multiprocessing Sinter smoke collection; and
- exact DEM/circuit/graph fingerprint checks.

### 15.2 Phase 1: independent pilot/discovery

Run exactly `200,000` paired natural-noise shots at each candidate, using
`PU-window` with all four stages and `HW=10`:

| Priority | `d` | patches | yokes | `r` | SI1000 `p` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 6 | 2 | 28 | 0.001 |
| 2 | 7 | 6 | 2 | 28 | 0.002 |
| 3 | 7 | 6 | 2 | 28 | 0.003 |
| 4 | 5 | 6 | 2 | 20 | 0.003 |
| 5 | 5 | 6 | 2 | 20 | 0.005 |

The first cell in this fixed order that satisfies all gates becomes the single
confirmatory accuracy cell:

- at least 5% of shots activate at least one L1 domain;
- at least 200 `U0-direct` shot failures occur in the pilot;
- at least 100 correctness-discordant pairs `b+c` occur;
- a fixed-size confirmatory design with at least 90% power under equality fits
  within `N_max = 10,000,000` paired shots; and
- no correctness, overflow-accounting, or data-integrity gate fails.

Selection may use only baseline failure rate, activation, total disagreement
`b+c`, and resource estimates. The analyst selecting the cell must not inspect
the signed accuracy difference `b-c`. Pilot shots are never reused in the
confirmatory test.

Before the first pilot shot, freeze the draft
`docs/PROMATCH_PILOT_PROTOCOL.json` from a clean implementation HEAD. Commit
the exact generated `docs/PROMATCH_PILOT_FROZEN_V3.json` as the sole change after
that HEAD, then sample from the resulting clean worktree. The frozen protocol
contains the table, fixed `200,000` shots/cell, seed ranges, gates, priority
rule, software hashes, and output schema.

V1 and V2 are immutable diagnostic corpora, not inputs to V3. V1 exposed the
two-stage replay-cap mismatch. V2 fixed that cap but retained an unscoped
artifact contract under which analysis could write into the collection
directory; it also showed regressions greater than recoveries in all five
cells. V3 therefore uses fresh roots and must be frozen and collected from
scratch. Neither earlier corpus may be edited, resumed, or promoted.

After pilot collection, run the scientific analyzer before constructing the
confirmatory protocol. It regenerates every scheduled pilot batch with the
frozen 32-process collector, compares the complete deterministic payload, and
then applies the fixed unsigned selection rule. Confirmatory freeze requires
both `--pilot-protocol` and `--pilot-input`; the tool derives the selected cell,
`p_U0_design`, `delta_NI`, `N_confirm`, schedules, and provenance hashes from
the verified pilot. Manually populated adaptive literals are not trusted.

If no candidate qualifies, the first round reports accuracy confirmation as
infeasible. Adding another cell requires a versioned protocol amendment made
before generating its data; it is not silently called confirmatory. In this
case the workflow stops after the V3 pilot: do not freeze a first-round
confirmatory manifest and do not sample a holdout.

### 15.3 Frozen accuracy design

For paired outcomes define:

```text
b = #(U0 correct, PU wrong)   # regressions
c = #(U0 wrong, PU correct)   # recoveries

Delta = p_PU - p_U0 = (b - c) / N
```

The non-inferiority hypotheses are:

```text
H0: Delta >= delta_NI
H1: Delta <  delta_NI
```

Use one-sided `alpha=0.025`. The design margin is:

```text
delta_NI = 0.05 * p_U0_design
```

where `p_U0_design` is the selected cell's pilot baseline failure rate. The
protocol file stores both as literal numeric values before holdout sampling.
This permits at most an absolute regression equal to 5% of the independently
estimated baseline risk; it is not recomputed from holdout data.

Use a named matched-pair risk-difference method that supports a nonzero margin:
the primary analysis is the Tango score interval/test for paired binary data.
Report the one-sided upper 97.5% confidence bound for `Delta`. Exact McNemar is
reported only for the zero-difference equality/superiority question; it is not
used as the non-inferiority test with a nonzero margin.

Let `x=b+c` and `q=x/N_pilot`. Before pilot sampling, the pilot protocol freezes
this sample-size rule:

1. Compute `q_U`, the one-sided 95% Clopper-Pearson upper bound for `q`, using
   the beta-quantile definition and its explicit `x=0`/`x=N` boundary cases.
2. Under the design alternative `Delta=0`, use

   ```text
   z_alpha = Phi^-1(0.975) = 1.959963984540054
   z_power = Phi^-1(0.90)  = 1.2815515655446004

   N_raw = ceil(q_U * (z_alpha + z_power)^2 / delta_NI^2)
   ```

3. Round `N_raw` upward to the next full 10,000-shot counter-seeded batch. If
   the result exceeds `N_max=10,000,000`, the cell fails the resource gate.
4. Before accepting the rule, verify by deterministic multinomial simulation
   of `b,c,other` under `b=c=q_U/2` that the implemented Tango decision has at
   least 90% power at the rounded N. If the frozen lower confidence criterion
   for simulated power fails, increase N by one 10,000-shot batch and repeat
   until it passes or exceeds `N_max`. The pilot protocol freezes the simulation
   replicate count, RNG seed, power confidence method/level, acceptance
   tolerance, and statistical-code hash.

The primary holdout interval uses Tango's published efficient-score equations,
solved with a bracketed root finder whose bracket, tolerance, maximum
iterations, and boundary-table behavior are fixed in the pilot protocol. Test
vectors and the simulated-coverage gate in Section 14.6 must pass before any
pilot shot is sampled. Thus the pilot outcomes cannot determine a convenient
CI, solver, or rounding convention.

Confirmation passes accuracy non-inferiority only when:

```text
upper_97.5_percent_bound(Delta) < delta_NI
```

After that gate passes, accuracy improvement may be claimed only when the same
hierarchical bound is below zero. This is a closed, ordered analysis; the
improvement question is not searched across cells or ablations.

If the selected cell has zero holdout baseline failures, the relative risk is
undefined and is reported as such. The paired absolute analysis remains
primary. With zero observed discordance, the Tango efficient-score inversion
uses its profiled boundary solution
`z_(1-alpha)^2 / (N + z_(1-alpha)^2)`; the frozen margin and ordinary decision
gate still determine whether the result passes. A zero-discordance table is
therefore neither an automatic success nor forced to the vacuous bound 1.

### 15.4 Fixed sampling and stopping

Accuracy collection uses the same fixed `N_confirm` shots for both decoders.
It never stops on `MAX_ERRORS`, observed significance, a favorable trend, or a
decoder-specific failure count. Interruptions resume exact missing shot-index
ranges. A resource interruption before fixed `N` is complete is reported as
inconclusive unless a sequential design was separately preregistered; round one
does not use such a design.

There are no decoder-specific exclusions. A rollback still yields an ordinary
fallback prediction and remains in the paired table. A crash, invariant
violation, missing prediction, or malformed output fails the run; it is not
dropped from `N`.

Every documented V3 command keeps `MAX_ERRORS` unset, including smoke tests.
Pilot, confirmatory, and target collection additionally reject any
result-dependent stopping rule.

## 16. Workload and Coverage Analysis

Record, unconditionally over all shots:

- initial and residual HW per L1 window domain;
- initial and residual global HW;
- detector events passed to residual PyMatching;
- shot/domain activation rates;
- successful-predecode and rollback/overflow rates;
- attempted and committed matches per stage;
- decision-weight, XOR-support-weight, and path-length distributions;
- withheld cross-window/terminal/yoke event counts; and
- no-op, pass-through, success, and rollback shot counts.

Conditional-on-activation summaries are secondary. The primary workload
estimand remains unconditional, so an operating point cannot look favorable by
discarding easy shots.

Inactive shots must satisfy:

```text
PU prediction = U0-wrap prediction = U0-direct prediction
```

Therefore all paired accuracy differences must be attributable to activated
shots. Report the activated-shot paired table descriptively as a debugging aid,
while retaining the all-shot table as the inference target.

At each preregistered cell define:

```text
R_work = mean(residual detector events) / mean(original detector events)
```

A material workload-improvement claim requires the upper one-sided 97.5%
paired-bootstrap confidence bound to be below `0.90` (at least 10% reduction).
The exact practical threshold is frozen in the protocol; changing it creates a
new exploratory analysis.

Rollback is counted in two ways:

- **reference fallback:** the unchanged domain is decoded by unrestricted
  software PyMatching, so the shot receives a valid output;
- **capacity accounting:** the domain is an L1 overflow even though software
  fallback hides no logical result.

The overflow rate must never be omitted from a coverage claim.

## 17. Latency Protocol

### 17.1 Timing definitions

Measure direct wall-clock intervals with `time.perf_counter_ns()`:

```text
T_total:
    adapter entry -> returned packed predictions

Diagnostic interval:
    T_backend
```

`T_total` is measured directly, not defined as the sum of instrumented
components. V3 persists only total-adapter and residual-backend intervals,
because those are the intervals emitted and validated by the current latency
harness. It makes no promised breakdown into unpack, layout traversal,
prematch, or repack time. Sampling, circuit/DEM compilation, file I/O, logging,
telemetry serialization, and result analysis are outside `T_total`. Input
batches are pregenerated before timing.

Measure uninstrumented `T_total` separately from the backend diagnostic so the
backend timer does not perturb the primary endpoint. For `T_backend`, first
pregenerate paired original/residual packed corpora, then time only the matcher
call on each corpus; the predecoder is outside this diagnostic interval.

The primary latency mode is batch size 1. It estimates in-process online-style
software latency, not a hardware deadline. Batch sizes `64` and `1024` are
secondary throughput modes; their percentiles are per-batch, and amortized time
per shot is labeled as such. They are never described as per-shot tail latency.
Sinter worker startup, IPC, and collection throughput are reported separately
from this in-process adapter latency.

### 17.2 Controls and execution

Benchmark all three paths on identical inputs:

```text
U0-direct
U0-wrap
PU-window
```

Run each inferential pair in randomized, balanced `AB`/`BA` blocks:
`PU-window/U0-direct`, `PU-window/U0-wrap`, and residual/original backend.
Any three-way diagnostic uses a balanced Latin-square order. Use multiple fresh
process restarts and the same fixed GC policy for all variants. Before each
timed run, record:

- CPU model, microcode, core affinity, NUMA node, OS, and kernel;
- governor, frequency/turbo state, and host-load checks;
- Python, Stim, Sinter, PyMatching, NumPy, and SciPy versions;
- native thread environment variables, fixed to one thread;
- batch size, warm-up count, block schedule, and process restart count;
- circuit/DEM/graph/config hashes and exact input-corpus digest.

Minimum batch-1 timing uses exactly 10 process restarts, 100 paired blocks per
restart, and 100 calls per decoder per block: `100,000` calls/decoder. Each
restart performs 1,000 unrecorded warm-up calls/decoder first. The final
manifest may increase these fixed counts after an independent timing pilot but
may not change them after confirmatory timing begins.

Each restart pregenerates a distinct, deterministic 10,000-shot natural-noise
corpus before warm-up. Calls cycle through its complete batches; batch-1
therefore sees 10,000 distinct syndromes per restart instead of repeatedly
timing one syndrome. Timing restarts execute serially in fresh processes
(`restart_concurrency=1`) to keep mutual CPU/cache contention out of the
estimand. The experiment-wide process setting remains 32 and is a hard cap;
serialization here is part of the frozen timing design, not a request to run
32 latency trials concurrently.

For mean/geometric-ratio inference, each block contributes the ratio of its two
100-call total durations. Compute the geometric mean of those paired block
ratios. For tail inference, pool the raw per-call durations with equal fixed
counts from every restart and define:

```text
Q99_variant = empirical type-7 0.99 quantile of all batch-1 calls
R_p99       = Q99_PU / Q99_U0_direct
```

Use a fixed 10,000-replicate hierarchical percentile bootstrap. Resample
process restarts, then paired blocks within each selected restart; retain all
calls inside a selected block as a cluster. Recompute the geometric-mean ratio
and pooled type-7 quantiles in every replicate. The one-sided upper bound is the
97.5th bootstrap percentile. The timing seed and exact percentile/quantile
implementation are frozen in the manifest. Individual calls are not treated as
independent experimental replicates for confidence intervals. Report median,
p90, p99, all raw block/call timings, and bounds.

### 17.3 Latency claim gates

Define:

```text
R_backend = geometric_mean(T_backend_residual / T_backend_original)
R_total   = geometric_mean(T_total_PU / T_total_U0_direct)
R_wrap    = geometric_mean(T_total_PU / T_total_U0_wrap)
```

The preregistered practical gates are:

```text
residual-backend relief:
    upper one-sided 97.5% CI(R_backend) < 0.90

end-to-end software-latency improvement:
    upper one-sided 97.5% CI(R_total) < 0.95
    and upper one-sided 97.5% CI(p99_PU / p99_U0_direct) < 1.05
```

`R_wrap` explains how much time is attributable to actual predecode work after
interface overhead but is not a substitute for `R_total`.

If only `R_backend` passes, report **residual-backend latency relief**. If
`R_total` does not pass, do not report end-to-end latency improvement even when
residual HW is much lower. No Python result is labeled FPGA, real-time, or
hardware latency.

## 18. Exact First-Round Experiment Matrix

### Phase 0: functional/offline characterization

- Synthetic graphs with limits `0,2,4,6,8,10`.
- Small maintained circuits with `d in {3,5}`, patches in `{2,6}`, and
  `yokes in {0,1,2}`.
- `PU-full` only for algebra, activation, and duration-scaling diagnostics.
- Production decoder smoke with `PU-window`.

### Phase 1: discovery pilot

Use exactly the ordered table and `200,000` shots/cell in Section 15.2. All
V3 pilot output comes only from the fixed primary comparison. Any later tuning,
boundary comparison, or stage/HW exploration is a separately versioned
discovery study.

### Phase 2A: confirmatory measurable cell

Use the one selected cell, fixed `PU-window` configuration, frozen
`delta_NI`, fixed `N_confirm`, and disjoint holdout seeds. Report:

- primary accuracy non-inferiority;
- hierarchical accuracy superiority;
- primary unconditional workload ratio;
- overflow/coverage;
- residual-backend latency; and
- end-to-end batch-1 software latency.

The combined tradeoff claim is scoped only to this cell.

### Phase 2B: fixed target-geometry performance cell

Use:

```text
d=11, patches=6, yokes=2, r=44, SI1000 p=0.001
PU-window, HW=10, stages 1-4, no-boundary, zero-frame
1,000,000 fixed paired workload shots
the frozen latency protocol from Section 17
```

Report workload, activation, overflow, backend latency, and total latency. Any
natural-noise accuracy counts are descriptive bounds only.

### Phase 3: optional exploratory sensitivities and ablations

Run after the confirmatory manifest and results are frozen:

- stages 1 only, 1-2, 1-3, and 1-4;
- `HW in {6,8,10}`;
- a separately versioned `PU-window` versus `PU-boundary` study, if the
  boundary prototype and its artifact contract are completed and validated;
- zero-frame versus separately named frame-bearing paths;
- window widths `d/2`, `d`, and `2d` where integral and structurally valid;
- full-history duration-scaling diagnostic; and
- broader Figure 8 grid
  `d in {5,7,9,11}`, patches in `{6,10}`, `r in {4d,8d}`, `p=0.001`.

These produce a Pareto/sensitivity picture. They do not retroactively choose a
better primary variant.

## 19. Multiplicity and Result Language

There is one selected confirmatory accuracy cell and one primary accuracy
outcome, so no across-cell selection remains in the holdout. Accuracy
superiority is tested only after non-inferiority passes.

Workload, backend latency, and total latency support distinct named claims. If
an omnibus statement that "at least one performance metric improved" is made,
apply Holm correction to that family; otherwise report each prespecified claim
and confidence interval without merging them.

All cells and variants are reported, including negative and inconclusive
results. Per-observable, conditional-activation, boundary, stage, window-width,
and HW-sweep analyses are labeled secondary/exploratory unless a separate
versioned protocol adjusts their multiplicity.

Permitted conclusion templates are:

```text
Accuracy preservation:
    "At the preregistered measurable cell, PU-window passed the paired
    one-sided non-inferiority test for any-observable shot failure."

Accuracy improvement:
    "At that cell, the paired upper confidence bound for Delta was below zero."

Workload relief:
    "PU-window reduced unconditional residual detector workload by the
    preregistered practical amount while passing the accuracy gate."

Backend latency relief:
    "The residual matcher ran faster on predecoded syndromes; total adapter
    latency did/did not improve."

End-to-end software improvement:
    "PU-window passed both the mean/block-ratio and p99 batch-1 total-latency
    gates relative to U0-direct, while passing accuracy non-inferiority at the
    same measurable cell."
```

Do not use "accuracy preserved," "equivalent," or "no degradation" merely
because a test is nonsignificant or no failures were observed.

## 20. Rare-Event Limitation

Naive Monte Carlo cannot validate logical error rates near `10^-12`. With zero
observed failures, a rough one-sided 95% upper bound is about `3/N`; this is a
bound on the sampled configuration, not proof of decoder equivalence or
effective-distance preservation.

Round one therefore establishes:

1. exact correction algebra, locality, and deterministic implementation;
2. paired accuracy behavior at a measurable validation/stress cell;
3. exhaustive low-fault behavior only where enumeration is genuinely complete;
4. activation, workload, overflow, and latency on target geometries; and
5. explicit disagreement examples that can be replayed.

A target-regime accuracy claim requires a separately validated rare-event
method such as fault-count stratification, importance sampling, or a compatible
gap estimator. Before comparing ProMatch, that estimator must reproduce an
ordinary-decoder reference point and demonstrate calibration/coverage on cells
where direct Monte Carlo is feasible.

## 21. Reproducibility Manifests and Raw Data

Scientific protocols use a two-commit sequence that avoids a self-referential
commit hash. Commit A is a clean implementation HEAD. Freeze against A, then
make commit B whose only change is the exact generated
`docs/*FROZEN*.json`. Collection accepts B only when A is its ancestor and the
single A-to-B change is that exact protocol file; any source, test, or other
documentation change is rejected. Apply this sequence first to
`docs/PROMATCH_PILOT_FROZEN_V3.json` and again to
`docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json` before holdout, target, or latency
collection.

The decoder, sampler, and statistical-analysis content hashes in the derived
first-round protocol must be identical to the pilot versions; otherwise the
pilot is rerun. The final protocol references the verified pilot
protocol/result hashes and contains literal values for:

- protocol and statistical-analysis version/hash;
- repository commit and explicit clean-worktree assertion;
- circuit metadata, complete circuit SHA-256, and DEM SHA-256;
- compiled-graph fingerprint and decoder names/configuration;
- pinned ProMatch paper version and reference-implementation commit used for
  semantic/golden-test comparison;
- Python/package/OS/kernel/CPU/microcode versions;
- CPU affinity, NUMA, thread environment, frequency/turbo, and GC policy;
- selected pilot cell, selection-gate inputs, `p_U0_design`, `delta_NI`, alpha,
  power target, `N_confirm`, and `N_max`;
- literal seed roots, derivation rule, exact disjoint batch-ID/range schedules,
  and fixed 10,000-shot batch size;
- fixed stopping rule, timing warmups/restarts/blocks, and practical thresholds;
- output schema and analysis-script hash; and
- bounded replay-retention policy.

V3 replaces the old unscoped `required_files` list with six explicit artifact
sets that match the emitters:

```text
pilot_collection:
    experiment.json
    protocol.json
    batches/<cell_id>/batch-<batch_id:08d>.json
    summary.json
pilot_analysis:
    analysis.json
    analysis.md
    pilot_selection.json
confirm_collection and target_collection, each:
    experiment.json
    protocol.json
    batches/<cell_id>/batch-<batch_id:08d>.json
    summary.json
accuracy_analysis:
    analysis.json
    analysis.md
latency_collection_per_cell:
    protocol.json
    suite.json
    batch-<batch_size>.restart-<restart_index:02d>.json
latency_analysis_per_cell:
    latency_analysis.json
    latency_analysis.md
```

Each set has its own directory. Scientific analysis requires an exact,
preexisting collection, writes only to a distinct analysis directory, and must
leave collection bytes unchanged. Before regeneration it checks the exact
protocol identity, batch schedule, file set, duplicate-free JSON parse, and
summary reconciliation. It then regenerates scientific batches without writes
and requires deterministic payload equality. The V3 schema records exact
emitted ledger, telemetry, replay, summary, analysis-cell, and selection-row
keys; prose aliases are not accepted.

The canonical JSON hash is the experiment ID. Resume is allowed only when the
current manifest hash matches. Raw output retains:

- all aggregate paired contingency counts;
- batch-ID/shot-range ledger with no gaps or duplicates and detector/observable
  array digests for every completed batch;
- unconditional telemetry sums/histograms needed to recompute every metric;
- raw block timing rows;
- data/corpus digests; and
- at most 100 deterministic candidates per regression, recovery, or rollback
  category in each batch ledger, followed by the globally lowest 100 hashes per
  category in each cell summary, with enough information to replay them.

Invariant violations are fatal run errors and are not converted into replay
rows. The completed V1 and V2 pilots and their frozen manifests are retained as
diagnostic audit artifacts only. They must never be edited, resumed, promoted,
or relabeled as V3 results. In particular, preserve V2's unfavorable paired
accuracy direction when reporting why a fresh V3 pilot is a go/no-go gate.

Regression and recovery replay semantics are self-verifying from the retained
logical predictions. Rollback is internal predecoder control flow, so the row
validator checks its retained detector corpus and exact count structurally;
claim-bearing scientific analysis authenticates rollback status by regenerating
and comparing the complete deterministic batch. Non-scientific resume is never
accepted as claim-bearing verification.

The final analyzer refuses mismatched hashes, duplicate ranges, missing ranges,
nonreconciling counts, or a total different from fixed `N`. Scientific paired
analysis does not merely trust the ledger: it regenerates every declared batch
with the frozen 32-process collector and compares the full deterministic
payload before computing a result.

## 22. Acceptance and Go/No-Go Gates

### 22.1 Implementation acceptance

- All Section 14 tests pass.
- The paired harness fails closed on generator provenance/metadata; the Sinter
  adapter fails closed on structural compatibility and, for scientific use, an
  allowlisted DEM hash.
- Every detector and eligible edge is classified exactly once.
- No primary path crosses patch, basis, window, terminal, or yoke boundaries.
- Every primary committed path has zero observable frame.
- Global syndrome-boundary and frame-composition invariants hold.
- Rollback is transactional and domain-local.
- Inactive/below-limit shots bit-match ordinary PyMatching.
- `U0-wrap`, `U0-direct`, and Sinter built-in baseline equivalence is proven.
- The multiprocessing Sinter smoke succeeds.

### 22.2 Statistical readiness

- Pilot and holdout seeds/ranges are disjoint.
- A cell passes every fixed selection gate without reading signed `b-c`.
- The V3 pilot analysis is written to its separate artifact-set directory and
  leaves the exact pilot collection byte-for-byte unchanged.
- The numeric margin, fixed N, score method, thresholds, and analysis hash are
  committed before holdout.
- The paired interval/power implementation passes simulated-coverage tests.
- Confirmatory collection has no `MAX_ERRORS` or result-dependent stopping.
- Raw rows reconcile exactly with the manifest and contingency table.

Failure of any readiness gate stops confirmation. It does not become a
post-hoc exploratory success.

### 22.3 Claim acceptance

- **Accuracy preservation:** the Section 15.3 NI bound passes.
- **Accuracy improvement:** the hierarchical superiority bound passes.
- **Workload improvement:** the Section 16 practical bound passes at a named
  cell, and accuracy NI passes at that same cell if a tradeoff claim is made.
- **Backend latency improvement:** `R_backend` passes; label it backend-only if
  `R_total` fails.
- **End-to-end software latency:** both Section 17.3 total-latency gates pass.
- **Target geometry:** performance claims only, unless a separately powered
  target accuracy protocol exists.

An underpowered, zero-activation, zero-information, incomplete, or interrupted
run is **inconclusive**, not non-inferior.

## 23. Verification Commands

Focused and complete tests:

```bash
python -m pytest \
    tests/yoked/decoding/_promatch_layout_test.py \
    tests/yoked/decoding/_promatch_graph_test.py \
    tests/yoked/decoding/_promatch_core_test.py \
    tests/yoked/decoding/_promatch_decoder_test.py \
    tests/yoked/decoding/_promatch_experiment_test.py \
    tests/yoked/decoding/_promatch_stats_test.py \
    tests/yoked/decoding/_promatch_analysis_test.py \
    tests/yoked/decoding/_promatch_latency_test.py \
    tests/yoked/decoding/_promatch_latency_integration_test.py \
    tests/yoked/decoding/_promatch_latency_analysis_test.py
python -m pytest
```

Integration smoke under the required scratch directory:

```bash
env -u MAX_ERRORS \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
DECODER=promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching \
MAX_SHOTS=10000 PROCESSES=32 THREADS_PER_PROCESS=1 \
./reproduce_fig8_1d smoke "$TMPDIR/yoked-promatch-smoke"
```

All documented simulations keep `MAX_ERRORS` unset, use exactly 32 processes,
and pin every supported native numerical runtime to one thread. The paired
`smoke` and `latency-smoke` commands remain explicitly non-claim-bearing; they
cannot substitute for a frozen pilot, holdout, target, or latency suite.

The end-to-end scientific workflow is:

```bash
# Keep these settings in the shell for every command below.
unset MAX_ERRORS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

git status --short
tools/benchmark_promatch_l1 freeze \
    --protocol docs/PROMATCH_PILOT_PROTOCOL.json \
    --out-protocol docs/PROMATCH_PILOT_FROZEN_V3.json
git add docs/PROMATCH_PILOT_FROZEN_V3.json
git commit -m "Freeze ProMatch L1 pilot protocol V3"

tools/benchmark_promatch_l1 pilot \
    --protocol docs/PROMATCH_PILOT_FROZEN_V3.json \
    --out "$TMPDIR/yoked-promatch-pilot-v3" \
    --processes 32

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_PILOT_FROZEN_V3.json \
    --input "$TMPDIR/yoked-promatch-pilot-v3" \
    --out "$TMPDIR/yoked-promatch-pilot-v3-analysis"

# Stop here unless the exact frozen selection gate returns "selected".
jq -e '.status == "selected"' \
    "$TMPDIR/yoked-promatch-pilot-v3-analysis/pilot_selection.json"

tools/benchmark_promatch_l1 freeze \
    --protocol docs/PROMATCH_FIRST_ROUND_PROTOCOL.json \
    --pilot-protocol docs/PROMATCH_PILOT_FROZEN_V3.json \
    --pilot-input "$TMPDIR/yoked-promatch-pilot-v3" \
    --out-protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json
git add docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json
git commit -m "Freeze ProMatch L1 first-round protocol V3"

tools/benchmark_promatch_l1 confirm \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --out "$TMPDIR/yoked-promatch-confirm-v3" \
    --processes 32

tools/benchmark_promatch_l1 target \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --out "$TMPDIR/yoked-promatch-target-v3" \
    --processes 32

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --input "$TMPDIR/yoked-promatch-confirm-v3" \
    --out "$TMPDIR/yoked-promatch-confirm-v3-analysis"

tools/analyze_promatch_l1 \
    --protocol docs/PROMATCH_FIRST_ROUND_FROZEN_V3.json \
    --input "$TMPDIR/yoked-promatch-target-v3" \
    --out "$TMPDIR/yoked-promatch-target-v3-analysis"

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

If the V3 pilot reports `confirmation-infeasible`, do not execute the
first-round freeze or any holdout command. A changed algorithm or grid requires
another versioned draft and fresh pilot; the V2 negative signal remains part of
the record. A `selected` result means only that the frozen unsigned
measurability and power gates passed, not that pilot accuracy favored the
treatment.

All simulation collection and scientific regeneration commands use exactly 32
processes and reject values above 32. Latency records the same frozen global
setting, but its ten fresh timing restarts are intentionally serialized and
each restart cycles through a distinct deterministic 10,000-shot corpus.

## 24. Explicit Departures from Published ProMatch

| Dimension | Published/reference ProMatch | This first-round yoked experiment |
| --- | --- | --- |
| Syndrome history | Approximately `d` rounds for one surface-code decode | Nonoverlapping `d`-round body windows; terminal/cross-window evidence left global |
| Capacity policy | Timing-adaptive handoff to a finite-capacity real-time matcher | Fixed `HW=10` primary; limits 6/8/10 are later sensitivity points |
| Domains | One surface-code matching problem | Independent `(patch, basis, window)` L1 units |
| Boundary | Surface-code boundary behavior in the source model | No-boundary V3 primary; parity-aware comparator deferred to a separate exploratory study |
| Outer code | None | Yoke constraints retained only in flat residual matching |
| Backend | Astrea/real-time finite-capacity setting in the paper | Unrestricted software PyMatching fallback |
| Correlations | Independent-error decoding graph | Ordinary PyMatching's merged uncorrelated graph |
| Ordering | Paper/source implementation ordering | Fully specified deterministic Python ordering |
| Execution | FPGA-oriented real-time design | Offline Python scientific prototype |

The implementation is therefore called **ProMatch-style**, not a reproduction
of the published hardware system.

## 25. Deferred Work

### Correlated residual decoding

- Add `PC` and compare it directly with `C0` on identical shots.
- Preserve/recover correlation-group provenance from the DEM; ordinary
  `matcher.edges()` is insufficient for reconstructing PyMatching's correlated
  two-pass reweighting semantics.
- Validate residualization and any correlation seed/reweight rule separately.

### True streaming L1 decoding

- Introduce arrival order, overlapping temporal halos, commit horizons, and
  cross-window state.
- Define how an L1 logical frame becomes an outer-code input rather than only a
  final observable XOR.
- Measure deadline miss rate and worst/tail latency per arriving window.

### Native/hardware implementation

- Replace Python traversal with native fixed-layout structures.
- Use proven incremental degree/dependency updates.
- Quantize weights and bound path-search memory.
- Build a cycle-accurate or FPGA evaluation before any real-time claim.

### Hierarchical L1/L2 decoder

- Decode each surface patch into a logical outcome plus calibrated confidence.
- Construct the explicit outer yoke/QDPC syndrome and likelihood model.
- Run an outer decoder on those L1 messages.
- Compare true hierarchical `L1 -> L2` decoding against both flat correlated
  joint MWPM and the hybrid predecode-plus-flat-residual experiment here.

### Rare-event target accuracy

- Integrate composite-correction telemetry with a validated rare-event sampler.
- Reproduce known ordinary-decoder points first.
- Predeclare estimator diagnostics and uncertainty before target comparison.

## 26. References

- [ProMatch paper](https://arxiv.org/abs/2404.03136)
- [ProMatch reference implementation](https://github.com/nargesalavi/Promatch)
- [Tango paired-proportion score method](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C891::AID-SIM779%3E3.0.CO;2-I)
- Existing maintained circuit generator:
  `src/yoked/_yoked_memory_circuits.py`
- Existing DEM construction:
  `src/yoked/gap/_collection_worker_state.py`
- Existing direct PyMatching integration:
  `src/yoked/gap/_gap_worker_handler.py`
- Maintained Sinter workflow: `reproduce_fig8_1d`
