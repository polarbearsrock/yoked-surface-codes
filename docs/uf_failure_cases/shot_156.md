# Shot 156: Four yoke terminals still exclude the required logical pair

The final Z-yoke tree has four terminals, yet the forest and the full graph internal to
the clusters both have zero logical-change rank. The terminals therefore provide no way
to change the required patch-level prediction. UF and ordinary MWPM share the wrong L1
and L5 answer.

This is row 156 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
416 sampled physical fault events and 782 fired detectors. The measured yoke bits are
X=1, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 530 | — |
| Repository UF | 560 | L1, L5 |
| Ordinary joint MWPM | 560 | L1, L5 |
| Correlated MWPM | 530 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,0 X_L,2`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (0, 1) | (0, 0) | Z |
| 1 | (0, 0) | (0, 0) | None |
| 2 | (1, 0) | (1, 1) | Z |
| 3 | (0, 0) | (0, 0) | None |
| 4 | (0, 1) | (0, 1) | None |
| 5 | (0, 0) | (0, 0) | None |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F103 | patch 0, round 7; tick 70; instruction 2901 | DEPOLARIZE1(0.006) | Y on data q24 at (3, 2) | D2035, D2036, D2041, D2042 | None |
| F126 | patch 2, round 9; tick 85; instruction 3246 | DEPOLARIZE2(0.003) | Y on data q139 at (5, 3); I on ancilla q431 at (4.5, 3.5) | D2434, D2440, D2723, D2729 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F103's component e9021, D2036–D2041. Its original weight is 4.132583800266, and
its observable label is None. Correlated MWPM selects this edge; UF does not. The full
detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D2036<br/>observed syndrome 1"]
    B["D2041<br/>observed syndrome 0"]
    A ---|"e9021: weight 4.132584"| B
    C["UF cluster 2036<br/>2 vertices, 2 fired"]
    D["UF cluster 2434<br/>300 vertices, 113 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D2036 | 1 | [2.5, 2.5, 7.0] | 2 vertices, 2 fired; even detector parity |
| D2041 | 0 | [3.5, 1.5, 7.0] | 300 vertices, 113 fired; boundary terminal |

The endpoints finish in different inactive clusters. The connection has grown
3.465078413718 of its required 4.132583800266 and never enters the forest. One side
stops because of even detector parity; the other stops because of a boundary terminal.

| Endpoint | Incident edges selected by UF |
|---|---|
| D2036 | e10424 |
| D2041 | None |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D2041 | F99, F103 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

| Decoder | Selects the highlighted target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

**Recorded growth transitions for the highlighted edge.**

The initial rate is 1, counting active detector endpoints; a virtual terminal never
initiates growth. The table records changes in this edge's rate or internal status.
Other unions and parity-preserving size changes are omitted. Times are algorithmic
growth coordinates, not circuit rounds or software latency.

| Growth time | Accumulated growth | First endpoint cluster | Second endpoint cluster | Rate after event | Edge event |
|---|---|---|---|---|---|
| 1.839151649 | 1.839151649 | 2 fired, even; inactive | 0 fired, even; inactive | 0 | Activity changed |
| 3.678303297 | 1.839151649 | 2 fired, even; inactive | 1 fired, odd; active | 1 | Activity changed |
| 5.304230062 | 3.465078414 | 2 fired, even; inactive | 10 fired, boundary; inactive | 0 | Activity changed |

Required growth is 4.132583800266; final accumulated growth is 3.465078413718. Final
edge status: incomplete between final clusters.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e9021: D2036–D2041 |
| First-pass selected source | e10460: D2035–D2042 |
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
saved observable mask 530. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 530 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F126, the selected diagnostic component is e12463, D2434–D2729. Its observable label
is None, and its weight is 5.625379320882. Its growth outcome is: frozen inside cluster
before completion.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D2434 | 1 | 300 vertices, 113 fired; boundary terminal |
| D2729 | 1 | 300 vertices, 113 fired; boundary terminal |

The selected first-pass source is e12426 (D2440–D2723); it lowers the target weight from
5.625379320882 to 1.097009937212. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 37 / 37 / 1 | 168 / 120 / 0 | None | 26, 26, 13, 12, 20, 22 |
| Z | 22 / 21 / 1 | 300 / 113 / 1 | T8493, T8541, T8589, T8637 | 39, 18, 13, 14, 21, 8 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| Z | T8493 | D879 | 0 | Yes |
| Z | T8541 | D1167 | 0 | No |
| Z | T8589 | D1455 | 0 | No |
| Z | T8637 | D1743 | 0 | No |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 560 |
| All original edges within each final UF cluster | 0 | No | 560 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
34, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 530 | 560 | 530 | L1, L5 |
| F103 removed | 530 | 690 | 530 | L5, L7 |
| F126 removed | 530 | 530 | 530 | None |
| F103 and F126 removed | 530 | 530 | 530 | None |
| F103 alone | 0 | 0 | 0 | None |
| F126 alone | 0 | 0 | 0 | None |
| F103 and F126 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing F103 changes the wrong pair to L5 and L7 instead of fixing the shot. Removing
F126 fixes UF, as does removing both. F103's edge stays between an even cluster and the
boundary-containing yoke cluster; F126's edge freezes internally before completion.
These different growth outcomes accompany the same restriction on final logical freedom.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L1 | Z0 | T8493 | e7608, e7610, e9019, e9021, e10424, e10426, e10397, e10398, e8958, e8963, e7527, e6087, e4647, e3168 |
| L5 | Z2 | T8750 | e10968, e12447, e11016, e12463, e12468, e12431, e10926 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 560 to 530. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 156 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
416 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_156`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 144](shot_144.md) · [Next: shot 188](shot_188.md)
