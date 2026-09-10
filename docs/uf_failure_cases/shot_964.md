# Shot 964: First-round faults produce only a partial logical repair

Two faults in the first noisy round of patch 5 contribute to a failure involving both
logical sectors. Their relevant connections remain between different UF clusters.
Removing either fault repairs the X-logical pair, while a Z-logical pair remains wrong.

This is row 964 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
422 sampled physical fault events and 829 fired detectors. The measured yoke bits are
X=1, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 1588 | — |
| Repository UF | 2804 | L6, L7, L10, L11 |
| Ordinary joint MWPM | 1588 | None |
| Correlated MWPM | 1588 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Y_L,3 Y_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F4 | patch 5, round 1; tick 8; instruction 641 | DEPOLARIZE2(0.003) | Y on data q275 at (3, 3); I on ancilla q560 at (2.5, 2.5) | D548, D549, D554, D555 | None |
| F11 | patch 5, round 1; tick 10; instruction 939 | DEPOLARIZE1(0.006) | Y on data q287 at (5, 1) | D560, D561, D566, D567 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F4's component e1566, D549–D554. Its original weight is 4.132583800266, and its
observable label is None. Correlated MWPM selects this edge; UF does not. The full
detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D549<br/>observed syndrome 0"]
    B["D554<br/>observed syndrome 1"]
    A ---|"e1566: weight 4.132584"| B
    C["UF cluster 821<br/>203 vertices, 139 fired"]
    D["UF cluster 554<br/>2 vertices, 2 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D549 | 0 | [42.5, 3.5, 1.0] | 203 vertices, 139 fired; boundary terminal |
| D554 | 1 | [43.5, 2.5, 1.0] | 2 vertices, 2 fired; even detector parity |

The endpoints finish in different inactive clusters. The connection has grown
2.499666942951 of its required 4.132583800266 and never enters the forest. One side
stops because of a boundary terminal; the other stops because of even detector parity.

| Endpoint | Incident edges selected by UF |
|---|---|
| D549 | None |
| D554 | e1601 |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D549 | F4, F14 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e1566: D549–D554 |
| First-pass selected source | e3026: D548–D555 |
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
saved observable mask 1588. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 1588 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 32 / 32 / 0 | 203 / 139 / 1 | T8570, T8618 | 4, 11, 35, 32, 22, 34 |
| Z | 9 / 8 / 0 | 314 / 119 / 1 | T9302, T9350, T9398 | 20, 20, 5, 15, 1, 58 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 2804 |
| All original edges within each final UF cluster | 0 | No | 2804 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
3264, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 1588 | 2804 | 1588 | L6, L7, L10, L11 |
| F4 removed | 1588 | 3764 | 1588 | L7, L11 |
| F11 removed | 1588 | 3764 | 1588 | L7, L11 |
| F4 and F11 removed | 1588 | 3764 | 1588 | L7, L11 |
| F4 alone | 0 | 0 | 0 | None |
| F11 alone | 0 | 0 | 0 | None |
| F4 and F11 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing both selected faults still leaves L7 and L11 wrong. This early-circuit case is
therefore not a complete two-fault explanation of the shot. It also includes an unfired
endpoint: an individual fault's detector footprint must be distinguished from the full
syndrome after cancellations.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L6 | X3 | T8570 | e6764, e6804, e8277, e6879, e8353, e6954, e6951, e5512 |
| L7 | Z3 | T8470 | e1026, e1023, e1036, e2580, e2582, e2553, e3998, e2526 |
| L10 | X5 | T8488 | e1447, e1492, e1526, e1566, e1601, e1638, e3090, e3092 |
| L11 | Z5 | T9302 | e21768, e21735, e20254, e21662, e23141, e23146, e23117, e24558, e25998, e27446, e28958, e27486 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 2804 to 1588. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 964 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
422 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_964`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 295](shot_295.md) · [Next: shot 1026](shot_1026.md)
