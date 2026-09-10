# Shot 75: Cancelled detector events accompany errors in both sectors

A data-Y fault produces a detector pair whose two contributions are cancelled by other
events. The pair's direct edge never grows in UF, yet correlated MWPM selects it. The
shot also includes an ancilla X fault whose direct temporal edge is in the forest but
unused by peeling.

This is row 75 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
439 sampled physical fault events and 835 fired detectors. The measured yoke bits are
X=1, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3522 | — |
| Repository UF | 258 | L6, L7, L10, L11 |
| Ordinary joint MWPM | 2514 | L4, L10 |
| Correlated MWPM | 3522 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Y_L,3 Y_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F283 | patch 5, round 18; tick 176; instruction 6193 | DEPOLARIZE2(0.003) | Y on data q267 at (2, 2); I on ancilla q559 at (2.5, 1.5) | D5147, D5436, D5443, D5444 | None |
| F357 | patch 3, round 23; tick 221; instruction 7809 | X_ERROR(0.006) | X on ancilla q469 at (3.5, 1.5) | D6505, D6793 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F283's component e26020, D5436–D5443. Its original weight is 4.132583800266,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D5436<br/>observed syndrome 0"]
    B["D5443<br/>observed syndrome 0"]
    A ---|"e26020: weight 4.132584"| B
    C["UF cluster 5436<br/>1 vertex, 0 fired"]
    D["UF cluster 5443<br/>1 vertex, 0 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D5436 | 0 | [41.5, 2.5, 18.0] | 1 vertex, 0 fired; even detector parity |
| D5443 | 0 | [42.5, 1.5, 18.0] | 1 vertex, 0 fired; even detector parity |

The endpoints finish in different inactive clusters. The connection has grown
0.000000000000 of its required 4.132583800266 and never enters the forest. One side
stops because of even detector parity; the other stops because of even detector parity.

| Endpoint | Incident edges selected by UF |
|---|---|
| D5436 | None |
| D5443 | None |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D5436 | F283, F292 | 2 mod 2 = 0 |
| D5443 | F283, F303 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e26020: D5436–D5443 |
| First-pass selected source | e26021: D5147–D5444 |
| Original target weight | 4.132583800266 |
| Reweighted target weight | 1.417837019967 |
| Source marginal probability | 0.00359224460718038 |
| Shared DEM mechanism probability | 0.000700490911966637 |
| Clipped implied probability | 0.195000894584533 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3522. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3522 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 43 / 43 / 1 | 815 / 159 / 1 | T8714, T8762 | 16, 10, 32, 39, 21, 40 |
| Z | 34 / 34 / 0 | 208 / 157 / 1 | T9108 | 17, 24, 26, 32, 24, 33 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 258 |
| All original edges within each final UF cluster | 0 | No | 258 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
3264, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3522 | 258 | 3522 | L6, L7, L10, L11 |
| F283 removed | 3522 | 1346 | 3522 | L7, L11 |
| F357 removed | 3522 | 258 | 3522 | L6, L7, L10, L11 |
| F283 and F357 removed | 3522 | 3522 | 3522 | None |
| F283 alone | 0 | 0 | 0 | None |
| F357 alone | 0 | 0 | 0 | None |
| F283 and F357 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing F283 repairs only part of UF's logical error; removing F357 alone leaves the
original discrepancy. Removing both together fixes UF. This conditional effect shows why
inspecting only one fault deletion can miss an interaction in the surrounding shot.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L6 | X3 | T8714 | e6753, e6791, e8267, e9754, e9788, e11267, e9832 |
| L7 | Z3 | T9382 | e34248, e34250, e34219, e32779, e31302, e29824, e31265, e32710, e32751, e31316, e31358, e29886 |
| L10 | X5 | T9208 | e25943, e27423, e26020, e26052, e27529, e26128, e26130, e24692 |
| L11 | Z5 | T9108 | e17372, e17374, e18782, e20259, e18826, e20235, e20237, e21640, e21639 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 258 to 3522. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 75 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
439 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_75`. [The collection guide](README.md) records
common hashes, the method, and the limits of the diagnostic interventions. These are
deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 23](shot_23.md) · [Next: shot 125](shot_125.md)
