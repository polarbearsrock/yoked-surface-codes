# Pinball-style predecoder integration plan

Status: semantic and validation contract, a frozen exploratory V1, an exact
pinned functional reference kernel, and a stricter non-claim-bearing YSC V2.
No performance result or end-to-end reproduction claim is made here.

## 1. Source boundary and terminology

The public algorithm sources used for this work are Knapen et al.,
[arXiv:2512.09807v2](https://arxiv.org/html/2512.09807v2) (12 December 2025),
and the authors' [Pinball repository at commit
`8f16f24b621aacfaa4f456a2aeec8df088faf3a7`](https://github.com/aknapen/Pinball/tree/8f16f24b621aacfaa4f456a2aeec8df088faf3a7).
The pinned [`src/predecoders.py`](https://github.com/aknapen/Pinball/blob/8f16f24b621aacfaa4f456a2aeec8df088faf3a7/src/predecoders.py)
is the functional reference for stage order and state mutation, while the paper
defines the intended primitives and complex-block behavior.

The arXiv v2 and the later [official HPCA 2026 program
entry](https://2026.hpca-conf.org/details/hpca-2026-main-conference/90/Pinball-A-Cryogenic-Predecoder-for-Quantum-Error-Correction-Decoding-Under-Circuit-L)
do not have identical headline results. For example, arXiv v2 reports up to
`3780.72x` bandwidth reduction, `<0.56 mW` peak power, `67.4x` total energy
savings, `32.58x` lower logical error rate than the cited room-temperature
predecoder, and 2,668 logical qubits at `d=21`; the HPCA entry instead states
`5892.75x`, `<0.54 mW`, `72.1x`, `38.88x`, and 2,777, respectively. It also
changes “nearly six orders of magnitude” to “more than six orders of
magnitude.” Those are author results under their configurations, not results
of this repository. Any future claim must identify its source/version and must
not transfer either set of numbers to this integration.

This repository's decoder is therefore called **Pinball-style**. It adapts the
published greedy primitives to a materially different circuit and output
representation; it is not a reproduction of the authors' software, hardware,
coverage, accuracy, logical-error, bandwidth, power, energy, or latency results.

## 2. Public algorithm state to preserve

The public experiment uses one rotated surface-code X-memory circuit, decodes
Z errors, and accepts only odd `d`. It generates `d` noisy circuit rounds but
decodes `d+1` detector layers: the code explicitly sets
`num_detector_rounds = num_circuit_rounds + 1`. This distinction must remain
visible in tests and documentation.

For each new detector layer, Pinball greedily applies nine fixed-priority
stages in this exact order:

| Stage | Public primitive |
| --- | --- |
| `M` | Clear equal-position active detectors in adjacent layers (a time-like measurement error); no data correction. |
| `B1` | Clear a same-layer bulk pair in the top-right direction and record the intervening one-data-qubit correction. |
| `B2` | As above, bottom-right. |
| `B3` | As above, bottom-left. |
| `B4` | As above, top-left. |
| `ST1` | Clear a current-layer detector with its top-right neighbor in the preceding layer; record the shared-data-qubit correction. |
| `ST2` | As above, with the top-left preceding-layer neighbor. |
| `H` | Clear the published two-row, cross-layer hook pair and record its two-data-qubit correction chain. |
| `E` | Last, clear a spatial-boundary detector against an artificial active boundary and record the selected boundary correction. |

Primitives within one hardware stage are intended to be conflict-free. When a
primitive sees both active endpoints (or one active endpoint for `E`), it
toggles its correction and clears the consumed detector bits. Later stages see
that mutated state. The public functional order is `M`, `B1`-`B4`, `ST1`-`ST2`,
`H`, then `E` on the layer being retired; after the final pair it applies `E`
to the last layer. Priorities are fixed by stage and direction. Edge
probabilities or MWPM weights do not alter choices at runtime.

“Complex” is a block-level disposition, not a partial result. If any detector
remains after the block, tentative Pinball corrections are discarded and the
original, unmodified detector block is sent to MWPM. The public experiment's
L2 path decodes its separately retained original detector sample, not the
mutated syndrome array.

This integration excludes the public artifact's HDL, pipeline timing,
cryogenic CMOS implementation, voltage/frequency and body-bias control,
physical-design results, power/area/energy model, and cryogenic-to-room-
temperature transport model. Those hardware-specific items are neither needed
to define the software heuristic nor supported by this repository.

## 3. Adaptation boundary in `yoked-surface-codes`

The maintained target is `yoked_magic_memory_circuit`, not Stim's generated
single-patch memory circuit. It has multiple spatially separated patches,
CZ-style syndrome extraction, both X- and Z-check bases, arbitrary measurement
round count `r` (Figure-8 work commonly uses `4d` or `8d`), a terminal magic
measurement layer, per-patch X/Z observable frames, and one or two outer yoke
detectors. Its inner detector times are exactly `0, ..., r`: `r` noisy circuit
rounds therefore produce `r+1` inner detector layers. Yoke detectors and edges
incident to them encode global constraints and are never local Pinball
primitives.

The v1 software contract is:

1. Compile the maintained detector layout in full-history mode. For patch
   `p`, convert global `x` to `local_x = x - p*(d+1)`, then normalize local
   algorithm coordinates by basis:
   `X: (u,v) = (local_x,y)` and `Z: (u,v) = (y,local_x)`. This transpose makes
   the two check bases share one directional schedule. “Top” is `+v` and
   “right” is `+u`.
2. Stream through all `r+1` inner layers, including the terminal magic layer.
   For an edge displacement `(du,dv,dt)`, `M` has `|dt|=1` and `du=dv=0`;
   `ST1`/`ST2` have `|dt|=1` and `|du|=|dv|=1`, with
   `u_later-u_earlier < 0` selecting `ST1` and `u_later-u_earlier > 0`
   selecting `ST2`; and `H` has `|dt|=1`, `du=0`, and `|dv|=2`.
   Same-layer bulk stages use, in priority order,
   `B1=(+1,+1,0)`, `B2=(+1,-1,0)`, `B3=(-1,-1,0)`, and
   `B4=(-1,+1,0)`, with centers selected by the upstream checkerboard parity.
   `E` contains only compatible spatial-boundary edges. The initial older
   layer is synthetic zero; an old layer is retired with `E`, and the terminal
   layer receives the final `E` pass.
3. Build primitives from the canonical graph edges obtained from the
   decomposed DEM/PyMatching graph, not from handwritten qubit maps or runtime
   weights. Compilation classifies endpoints by patch, basis, time, normalized
   displacement, spatial-boundary role, and observable mask; rejects ambiguous
   or unsupported geometry; assigns every eligible edge to exactly one stage;
   and freezes a deterministic order. Cross-patch, cross-basis, yoke-incident,
   and otherwise global edges are not locally consumed.
4. Represent a committed correction by the canonical graph edge's observable
   frame. Consuming an internal edge XOR-clears its two endpoint bits;
   consuming a boundary edge clears its one detector bit; either operation
   XORs that edge's observable mask into the tentative per-shot frame. This is
   the DEM-graph equivalent of the public data-qubit correction buffer.
5. Run the nine stages independently for every `(patch, check_basis)` domain,
   but decide disposition globally. If **any inner detector** remains after all
   domains finish, roll back the whole shot: discard every tentative frame,
   restore the complete original detector vector, and run global MWPM. There
   is no per-patch or per-basis partial commit in v1.
6. If all inner detectors clear, commit the accumulated graph-edge frame and
   invoke global PyMatching only on the remaining untouched yoke/global
   residual. The final prediction is `MWPM(residual) XOR Pinball_frame`.
7. Use the same uncorrelated backend as the current baseline,
   `pymatching.Matching.from_detector_error_model(dem)`, without correlated
   matching. Thus a complex shot is semantically the baseline path, while a
   simple shot differs only by the locally committed frame and the sparsified
   residual. V1 also fails closed on even `d`, preserving the only distance
   parity covered by the public artifact; even-distance support requires a
   separately validated schedule revision.

The proposed Sinter name records these choices:

```text
pinball-style-v1-fullhistory-nine-stage-wholeshotrollback-pymatching
```

The implementation belongs in a pure schedule/core module and a generic
Sinter adapter registered by `yoked.decoding:custom_decoders`. It must not add
an arm, option, schema field, analyzer branch, or output to
`tools/benchmark_promatch_l1` or the ProMatch latency harness. Those harnesses
authenticate a different frozen experiment; mutating them would blur
provenance and invalidate comparisons. A future claim-bearing Pinball-style
campaign needs its own harness and protocol.

### 3.1 Pinned physical reference kernel

`_pinball_reference.py` is a literal functional port of the public artifact at
commit `8f16f24...`. It deliberately retains the narrow upstream data model:
one row-major `(d+1) * ((d-1)/2)` syndrome grid per layer, a physical `d^2`
data-qubit correction buffer, synthetic-zero initial layer, two-layer mutation,
the exact nine-stage loop order, final `E` flush, odd distance, and the
left-column logical-parity rule. It has no YSC, DEM, PyMatching, patch, basis,
terminal-magic, or yoke semantics. Immutable after-stage traces expose the
mutated previous/current layers and correction buffer for differential tests.

This reference is an oracle and semantic anchor, not a registered YSC decoder.
During development, 1,800 randomized cases across `d=3,5,7` and batch sizes
`1`, `2`, and `d+1` matched the pinned implementation exactly in correction
buffer, complex flag, and fully mutated syndrome block.

### 3.2 YSC V2 contract

V2 is registered independently as:

```text
pinball-ysc-v2-cz-fullhistory-nine-stage-domainatomic-yokeedge-pymatching
```

It leaves V1 behavior and its decoder name unchanged. Relative to V1, V2:

1. Uses one explicit signed YSC-CZ temporal profile. After orienting a graph
   edge from earlier to later, `ST1=(+1,+1,+1)`, `ST2=(-1,+1,+1)`, and
   `H=(0,+2,+1)` in normalized `(u,v,t)` coordinates. Reflections and
   time-reversed copies fail compilation instead of being accepted through
   absolute displacements. X domains execute the public stage priority
   directly. The complementary Z checkerboard conjugates the geometric labels,
   so its YSC order is `M,B2,B1,B4,B3,ST2,ST1,H,E`; after the declared
   coordinate transform this is the public `M,B1,B2,B3,B4,ST1,ST2,H,E`
   priority. The Z-basis symmetry remains a documented YSC extension because
   the public artifact supplies only its X-memory path.
2. Generates and validates the complete expected `M`, `B1`--`B4`,
   `ST1`--`ST2`, `H`, and `E` slot sets independently from the edges that
   happen to appear. Missing or extra slots, direction changes, stage
   conflicts, and incorrect boundary counts fail closed.
3. Restores both spatial `E` sides. Each layer and `(patch,basis)` domain has
   `(d+1)/2` true-boundary sources at normalized `u=d-1.5` and `(d+1)/2`
   yoke-coupled sources at `u=0.5`. A yoke-side action tests only the inner
   source but accumulates the full `(inner,yoke)` detector boundary and the
   patch-owned observable frame. Shared yoke deltas are never read by a local
   stage and combine by XOR at commit.
4. Uses full-history transactions independently for every `(patch,basis)`.
   A simple domain commits even if another domain is complex; a complex
   domain discards its tentative support and remains unchanged for global
   MWPM. `result.complex` therefore means that at least one domain was complex,
   not that all tentative work was rolled back.
5. Validates the flattened, decomposed DEM component catalog before schedule
   compilation. Parallel components with incompatible observable frames,
   non-graphlike components, or any disagreement between the unmerged catalog
   and the canonical matcher graph are rejected before PyMatching can define a
   local choice accidentally.
6. Supports only odd-distance maintained CZ circuits with exactly two yoke
   detectors and two observables per patch. Other yoke counts and circuit
   profiles fail closed.
7. Associates every primitive with patch-local physical Pauli support: `M`
   has none, `B*`, `ST*`, and `E` have one target, and `H` has two. X-domain
   Z corrections use the explicit horizontal reflection of the public data
   buffer; Z-domain X corrections use the checkerboard/transposition symmetry.
   Supports combine by XOR into durable and tentative correction telemetry,
   and compilation checks their symplectic logical parity against the
   canonical edge frame.

V2 is a hybrid residual decoder: after local domain commits, global PyMatching
still decodes the combined residual and its prediction is XORed with the
committed frame. Its simple-domain rate is therefore not the public Pinball
cryogenic-offload or bandwidth metric.

V2 now reproduces the public correction buffer in patch-local physical
coordinates for the mapped single-domain cases covered by the differential
oracle. The remaining physical-integration gap is narrower but important: the
telemetry does not carry raw circuit qubit IDs, fault locations, or a
time-resolved correction sidecar, and the multi-patch/yoke binding is inferred
from fail-closed DEM components rather than supplied by the public artifact.
This is physical-support equivalence under an explicit coordinate map, not a
claim that the public cryogenic implementation handled YSC circuits.

## 4. Threat model and open uncertainties

- **Narrow public artifact.** The pinned repository models only X-memory/Z
  errors and rejects even distance. V2's Z-check/X-error map is differentially
  checked after a conjugated checkerboard transform, but that symmetry is not
  an independently published golden result. Even distances and the
  multi-patch/yoked circuit remain outside the public artifact.
- **Missing reproducibility material.** The public metadata are pickles for
  selected odd distances; the pinned tree provides no mapping generator,
  golden-vector/unit-test suite, or fixed sampler seeds. We can use its
  functional model as a differential oracle only where its assumptions hold.
- **Directional sensitivity.** Stage order and the meaning of top/right are
  part of the greedy policy. An axis swap, reflection, time reversal, or a
  different parity choice may cover the same edges yet change corrections and
  logical outcomes.
- **Canonical graph merging.** PyMatching may merge parallel DEM mechanisms
  into one graph edge and retain only one effective weight/frame choice.
  Compilation must detect ambiguous geometry or observable frames rather than
  assuming that every physical fault remains a distinct primitive.
- **Long blocks.** The public block has `d+1` detector layers. Full-history
  `r+1` processing for `r=4d`, `8d`, or another value changes conflict
  opportunities and block-level coverage; upstream coverage cannot be
  extrapolated to these circuits.
- **Yoke/global interaction.** Locally plausible corrections can change an
  observable frame that participates in an outer constraint. V2 retains yoke
  deltas for global MWPM and rolls back each complex domain, but a mixed shot
  may still commit other simple domains. That composition is tested
  algebraically; it does not by itself prove statistical accuracy.
- **Silent boundary failure.** As the paper itself illustrates, early greedy
  matches can isolate detectors that `E` later clears, producing a wrong
  correction without a complex flag. Boundary ordering, conditional accuracy,
  and false-simple cases therefore require direct tests, not only residual-
  clearing checks.
- **Artifact separation.** Frozen `PROMATCH_*` schemas, source manifests,
  seeds, summaries, and corpora describe ProMatch and must not be resumed,
  edited, promoted, or reused for Pinball-style evidence.

## 5. Validation gates

Gates are ordered; later measurement does not compensate for an earlier
semantic failure.

1. **Pure primitives and golden vectors.** Test every stage on active,
   inactive, overlapping, boundary, first-layer, and terminal-layer examples;
   test X/Z coordinate normalization and deterministic ordering. Add
   hand-derived golden vectors. For supported odd-distance single-patch
   X-memory cases, run a stage-by-stage differential oracle against the pinned
   upstream functional model, comparing consumed syndromes and correction/
   frame parity after an explicit coordinate translation.
2. **Schedule cardinality and conflicts.** For representative supported odd
   distances, assert closed-form per-stage/per-layer edge counts; exact-one
   classification of every eligible canonical edge; no duplicate geometric
   slot; pairwise-disjoint detector endpoints within each parallel stage; and
   stable schedule fingerprints under repeat compilation. Assert a fail-closed
   rejection for even `d`. Reject unexpected coordinate, frame, terminal,
   boundary, or merged-edge shapes.
3. **GF(2) algebra.** For every shot, verify
   `original XOR residual = XOR(boundaries of committed edges)` and
   `frame = XOR(edge observable masks)`. Verify double toggles cancel, stages
   mutate only their endpoints, domains do not leak, and final packing computes
   `MWPM(residual) XOR frame` for any observable count. For V1, whole-shot
   rollback must restore the complete input and zero the frame. For V2, each
   complex domain must restore only its own tentative support while simple
   domains and shared-yoke XOR deltas remain committed.
4. **Fault-order correctness.** Exhaustively inject every order-one atomic DEM
   fault for small maintained circuits across both bases, boundaries,
   terminal time, patches, and yokes; compare predicted observables with the
   fault's actual observables. Exercise selected order-two pairs spanning
   disjoint primitives, shared endpoints, cancellations, cross-stage races,
   hooks, silent-boundary patterns, cross-patch/global edges, and graph-edge
   merges. Record whether each case is simple or rolled back.
5. **Paired baseline checks.** On identical fixed-seed detector/observable
   batches, compare the Pinball-style decoder to the direct uncorrelated
   PyMatching baseline. V1 must be bit-exact to baseline on every complex shot.
   For V2, assert per-domain rollback and baseline identity only when no domain
   commits; a globally `complex` mixed shot may legitimately differ because a
   different simple domain committed. Retain paired recovery/regression/
   false-simple examples with enough information for exact replay. Include
   zero-shot, all-zero, packing-tail, multi-observable, and batch-size edge
   cases.
6. **Statistical semantics.** Only after the deterministic gates pass, measure
   block coverage `P(simple)`, conditional simple-shot accuracy
   `P(correct | simple)`, false-simple rate, fallback rate, and overall paired
   logical outcomes at the same `(d, patches, r, p)` cells. Report numerator
   and denominator plus appropriate binomial confidence intervals; report a
   paired interval for any baseline difference. Never compare an `r=d` source
   coverage number directly with an `r=4d` or `8d` result.
7. **Latency last.** Freeze semantics before timing. Then separately measure
   schedule compilation, local per-shot processing, fallback fraction, MWPM,
   and end-to-end Sinter throughput under the workstation's process/thread
   limits. Software timing is not evidence for cryogenic hardware latency,
   area, power, energy, or bandwidth.
8. **Claim gate and provenance freeze.** Any claim-bearing run requires a new
   Pinball-style protocol and new output directory, fixed shot counts and
   seeds, clean committed implementation, exact environment, circuit/DEM/
   layout/schedule fingerprints, implementation source hashes, the arXiv v2
   identifier, and upstream commit `8f16f24...`. Freeze the protocol before
   collection and make raw paired data replayable. No performance claim is
   permitted until all preceding gates pass.

## 6. Initial implementation checkpoint (2026-08-25)

The first vertical slice is implemented in `_pinball.py` (pure schedule and
transactional policy) and `_pinball_decoder.py` (bit-packed Sinter adapter and
full-graph residual composition). It is registered as
`pinball-style-v1-fullhistory-nine-stage-wholeshotrollback-pymatching`. The
frozen ProMatch factories and paired experiment schemas remain behaviorally
unchanged.

The deterministic checks completed at this checkpoint are:

- all nine stages on maintained odd-distance `d=3` and `d=5` DEMs, including
  both check bases, terminal-layer handling, conflict-free stage compilation,
  stable schedule fingerprints, and fail-closed even-distance rejection;
- input immutability, whole-shot rollback, exact GF(2) detector boundaries,
  observable-frame composition, little-endian packed I/O, one residual matcher
  call per nonempty batch, and complex-shot equivalence with direct PyMatching;
- all 820 complete order-one DEM mechanisms from one `d=3`, two-patch,
  two-yoke, three-round SI1000 circuit, including 600 correlated instructions
  containing `^` separators, with no Pinball-only logical failure; and
- a 1,000-shot generic Sinter smoke on the maintained `d=3`, six-patch,
  two-yoke, `r=12`, `p=0.001` circuit using 32 one-thread workers.

The repository test result at this checkpoint is `790 passed, 1 skipped`.
These are implementation checks, not evidence of accuracy improvement,
coverage, threshold, latency, bandwidth, or hardware performance. Differential
comparison with the upstream functional model, selected order-two analysis,
paired statistical evaluation, replay tooling, and a Pinball-specific frozen
protocol remain open validation gates.

## 7. Closer-to-source V2 checkpoint (2026-08-26)

The second implementation slice adds `_pinball_reference.py`,
`_pinball_v2.py`, and `_pinball_v2_decoder.py` without changing the meaning of
the V1 name. The deterministic additions cover:

- exact public physical-kernel behavior and per-stage correction traces;
- signed temporal primitives and exact slot census at `d=3,5,7`;
- both true and yoke-coupled `E` boundaries, including yoke residual and
  observable-frame effects;
- independent commit/rollback for full-history `(patch,basis)` domains;
- exact patch-local one- and two-qubit Pauli supports, durable/tentative XOR
  correction telemetry, and symplectic frame validation;
- mapped randomized differentials against the canonical public kernel for X
  domains on two translated patches and for the conjugated Z-domain symmetry,
  comparing the fully mutated syndrome, complex disposition, and physical
  correction buffer;
- unmerged DEM-component/frame validation before canonical graph use; and
- packed Sinter composition, explicit telemetry, immutable inputs, and
  residual PyMatching integration.

The bounded `d=3`, two-patch, two-yoke order-one corpus contains 820 complete
physical DEM mechanisms. V2 decodes all 820 without a logical failure and all
four domains are simple for every mechanism. This is useful order-one evidence,
not an order-two, statistical-accuracy, long-history, or performance claim.
The repository suite at this checkpoint is `838 passed, 1 skipped`, including
shared-yoke XOR cancellation and a spawned-worker adapter smoke. Selected
order-two coverage, a frozen paired campaign, and false-simple replay remain
required before any claim-bearing comparison with V1, ProMatch, or the
published Pinball results.
