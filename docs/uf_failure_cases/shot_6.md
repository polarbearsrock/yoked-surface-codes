# Shot 6: Two even clusters block a physical connection

Both yokes fire, and their final clusters are even and have no boundary terminals. A
connection produced by a YX fault remains incomplete between two distinct even clusters.
The final Z-yoke membership fixes the wrong logical prediction, even though ordinary
MWPM succeeds.

This is row 6 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
393 sampled physical fault events and 742 fired detectors. The measured yoke bits are
X=1, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3522 | — |
| Repository UF | 3402 | L3, L7 |
| Ordinary joint MWPM | 3522 | None |
| Correlated MWPM | 3522 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,1 X_L,3`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F240 | patch 1, round 18; tick 178; instruction 6200 | DEPOLARIZE2(0.003) | Y on data q62 at (1, 4); X on ancilla q355 at (0.5, 3.5) | D5239, D5240, D5245, D5246 | None |
| F210 | patch 3, round 15; tick 150; instruction 5517 | DEPOLARIZE1(0.006) | X on data q163 at (1, 5) | D4472, D4479 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F240's component e25037, D5240–D5245. Its original weight is 4.132583800266,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D5240<br/>observed syndrome 0"]
    B["D5245<br/>observed syndrome 1"]
    A ---|"e25037: weight 4.132584"| B
    C["UF cluster 5240<br/>5 vertices, 2 fired"]
    D["UF cluster 498<br/>184 vertices, 132 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D5240 | 0 | [8.5, 4.5, 18.0] | 5 vertices, 2 fired; even detector parity |
| D5245 | 1 | [9.5, 3.5, 18.0] | 184 vertices, 132 fired; even detector parity |

The endpoints finish in different inactive clusters. The connection has grown
2.861081228183 of its required 4.132583800266 and never enters the forest. One side
stops because of even detector parity; the other stops because of even detector parity.

| Endpoint | Incident edges selected by UF |
|---|---|
| D5240 | None |
| D5245 | e25066 |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D5240 | F234, F240 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e25037: D5240–D5245 |
| First-pass selected source | e26476: D5239–D5246 |
| Original target weight | 4.132583800266 |
| Reweighted target weight | 0.000000000000 |
| Source marginal probability | 0.0103267636825066 |
| Shared DEM mechanism probability | 0.00519032127620351 |
| Clipped implied probability | 0.5 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

The zero weight results from clipping the implied probability at 1/2. It does not mean
the error is certain or has probability one.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3522. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3522 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 39 / 39 / 1 | 108 / 102 / 0 | None | 22, 14, 16, 13, 13, 23 |
| Z | 29 / 29 / 1 | 184 / 132 / 0 | None | 15, 21, 20, 14, 36, 25 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 3402 |
| All original edges within each final UF cluster | 0 | No | 3402 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
136, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3522 | 3402 | 3522 | L3, L7 |
| F240 removed | 3522 | 3522 | 3522 | None |
| F210 removed | 3522 | 3402 | 3522 | L3, L7 |
| F240 and F210 removed | 3522 | 3522 | 3522 | None |
| F240 alone | 0 | 0 | 0 | None |
| F210 alone | 0 | 0 | 0 | None |
| F240 and F210 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing F240 fixes UF. Removing the other highlighted event F210 leaves the same wrong
logical pair, providing a useful negative intervention. A fault can lie on the
difference between two corrections without its removal fixing the logical outcome.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L3 | Z1 | T9077 | e26451, e26452, e26454, e26499, e25066, e25037, e25038, e23598, e22158, e20718, e20725, e20688 |
| L7 | Z3 | T9093 | e15411, e16818, e16825, e18265, e19705, e21147, e22630, e21197, e22643, e21168 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 3402 to 3522. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 6 consistently across detector, actual-observable, and prediction arrays. The
original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All 393
recovered events reproduce this shot, both in aggregate and by XORing their individual
responses. A forced-fault circuit also reproduces it for eight checks with the ordinary
sampler using seeds 1 and 123456. Graph corrections and prediction masks reproduce the
saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_6`. [The collection guide](README.md) records
common hashes, the method, and the limits of the diagnostic interventions. These are
deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Next: shot 13](shot_13.md)
