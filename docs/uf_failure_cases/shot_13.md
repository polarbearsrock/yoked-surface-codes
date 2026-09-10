# Shot 13: An even yoke cluster stops short of a boundary defect

A two-qubit fault produces a connection between a boundary-containing cluster and the
final even Z-yoke cluster. The connection remains incomplete when both clusters are
inactive. Ordinary and correlated MWPM both succeed, showing that correlation
reweighting is not required for a successful decoder on this particular syndrome.

This is row 13 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
392 sampled physical fault events and 793 fired detectors. The measured yoke bits are
X=0, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3166 | — |
| Repository UF | 1630 | L9, L11 |
| Ordinary joint MWPM | 3166 | None |
| Correlated MWPM | 3166 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,4 X_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F106 | patch 5, round 9; tick 86; instruction 3250 | DEPOLARIZE2(0.003) | X on data q284 at (4, 5); Y on ancilla q569 at (3.5, 5.5) | D2573, D2862, D2868, D2869 | None |
| F94 | patch 5, round 8; tick 71; instruction 2906 | DEPOLARIZE1(0.006) | Y on data q258 at (1, 0) | D2259, D2260, D2266, D8353 | L11 |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F106's component e13151, D2573–D2868. Its original weight is 6.436150368370,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D2573<br/>observed syndrome 1"]
    B["D2868<br/>observed syndrome 1"]
    A ---|"e13151: weight 6.436150"| B
    C["UF cluster 2573<br/>2 vertices, 1 fired"]
    D["UF cluster 560<br/>138 vertices, 126 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D2573 | 1 | [43.5, 5.5, 8.0] | 2 vertices, 1 fired; boundary terminal |
| D2868 | 1 | [44.5, 4.5, 9.0] | 138 vertices, 126 fired; even detector parity |

The endpoints finish in different inactive clusters. The connection has grown
5.686450013132 of its required 6.436150368370 and never enters the forest. One side
stops because of a boundary terminal; the other stops because of even detector parity.

| Endpoint | Incident edges selected by UF |
|---|---|
| D2573 | e11646 |
| D2868 | e13188 |

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e13151: D2573–D2868 |
| First-pass selected source | e13160: D2862–D2869 |
| Original target weight | 6.436150368370 |
| Reweighted target weight | 4.192356992696 |
| Source marginal probability | 0.0134545656584025 |
| Shared DEM mechanism probability | 0.000200280561291177 |
| Clipped implied probability | 0.0148856950403375 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3166. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3166 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 16 / 15 / 1 | 420 / 136 / 0 | None | 44, 23, 21, 19, 10, 19 |
| Z | 41 / 41 / 1 | 138 / 126 / 0 | None | 25, 9, 20, 10, 25, 36 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 1630 |
| All original edges within each final UF cluster | 0 | No | 1630 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
2560, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3166 | 1630 | 3166 | L9, L11 |
| F106 removed | 3166 | 3166 | 3166 | None |
| F94 removed | 1118 | 1118 | 1118 | None |
| F106 and F94 removed | 1118 | 1118 | 1118 | None |
| F106 alone | 0 | 0 | 0 | None |
| F94 alone | 2048 | 2048 | 2048 | None |
| F106 and F94 alone | 2048 | 2048 | 2048 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

The Z-yoke cluster has no terminal, so the per-patch parity of its member defects fixes
its logical prediction. Replacing peeling cannot change that prediction while preserving
the partition. The second highlighted fault itself flips L11, so its removal changes the
actual logical outcome.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L9 | Z4 | T9439 | e34488, e34487, e34527, e33094, e31656, e31620, e31661, e31688, e31689, e31642 |
| L11 | Z5 | T8774 | e10131, e11574, e10142, e10179, e8781, e10255, e10301, e10306, e11746, e13186, e13188, e13151, e11646 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 1630 to 3166. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 13 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
392 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_13`. [The collection guide](README.md) records
common hashes, the method, and the limits of the diagnostic interventions. These are
deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 6](shot_6.md) · [Next: shot 21](shot_21.md)
