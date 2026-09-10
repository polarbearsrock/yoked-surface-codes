# Shot 991: Two faults cancel a detector and repair only one logical sector

The ancilla X fault F57 and data-Y fault F82 both flip D1334, so its observed syndrome
bit is zero. Their target connections remain outside the final UF forest. Removing
either fault repairs the X-logical pair, but a Z-logical pair remains wrong.

This is row 991 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
463 sampled physical fault events and 864 fired detectors. The measured yoke bits are
X=1, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 2131 | — |
| Repository UF | 1 | L1, L4, L6, L11 |
| Ordinary joint MWPM | 2131 | None |
| Correlated MWPM | 2131 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,0 Z_L,2 Z_L,3 X_L,5`, ignoring global phase.
The decoder predicts observable flips; this notation does not describe literal physical
correction gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (1, 1) | (1, 0) | Z |
| 1 | (0, 0) | (0, 0) | None |
| 2 | (1, 0) | (0, 0) | X |
| 3 | (1, 0) | (0, 0) | X |
| 4 | (0, 0) | (0, 0) | None |
| 5 | (0, 1) | (0, 0) | Z |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F57 | patch 3, round 4; tick 31; instruction 1596 | X_ERROR(0.006) | X on ancilla q482 at (5.5, 0.5) | D1046, D1334 | None |
| F82 | patch 3, round 5; tick 42; instruction 1928 | DEPOLARIZE1(0.0003) | Y on data q186 at (5, 0) | D1327, D1328, D1334, D8353 | L7 |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F57's component e5490, D1046–D1334. Its original weight is 3.743791874133, and
its observable label is None. Correlated MWPM selects this edge; UF does not. This
component accounts for the event's entire detector response.

```mermaid
flowchart LR
    A["D1046<br/>observed syndrome 1"]
    B["D1334<br/>observed syndrome 0"]
    A ---|"e5490: weight 3.743792"| B
    C["UF cluster 758<br/>2 vertices, 2 fired"]
    D["UF cluster 1334<br/>1 vertex, 0 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D1046 | 1 | [29.5, 0.5, 3.0] | 2 vertices, 2 fired; even detector parity |
| D1334 | 0 | [29.5, 0.5, 4.0] | 1 vertex, 0 fired; even detector parity |

The endpoints finish in different inactive clusters. The connection has grown
1.871895937066 of its required 3.743791874133 and never enters the forest. One side
stops because of even detector parity; the other stops because of even detector parity.

| Endpoint | Incident edges selected by UF |
|---|---|
| D1046 | e4050 |
| D1334 | None |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D1334 | F57, F82 | 2 mod 2 = 0 |

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
| 1.871895937 | 1.871895937 | 2 fired, even; inactive | 0 fired, even; inactive | 0 | Activity changed |

Required growth is 3.743791874133; final accumulated growth is 1.871895937066. Final
edge status: incomplete between final clusters.

**What correlation changes locally.**

This edge receives no correlation discount: its weight remains 3.743791874133.
Correlated MWPM still selects it as part of its global solution. Its success cannot be
attributed to lowering this particular edge's cost. Correlation updates elsewhere and
the choice of a complete correction must be considered separately.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 2131. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 2131 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F82, the selected diagnostic component is e6924, D1327–D1334. Its observable label
is None, and its weight is 4.562635883352. Its growth outcome is: incomplete between
final clusters.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D1327 | 1 | 23 vertices, 10 fired; even detector parity |
| D1334 | 0 | 1 vertex, 0 fired; even detector parity |

At D1334, the recovered contributions are F57, F82; 2 contributions have even parity,
leaving the observed bit zero.

The selected first-pass source is e5448 (D1328–D8353); it lowers the target weight from
4.562635883352 to 1.483109493224. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 39 / 39 / 1 | 175 / 122 / 0 | T9331 | 13, 26, 26, 22, 14, 20 |
| Z | 9 / 8 / 0 | 268 / 109 / 1 | T8581 | 32, 24, 8, 20, 6, 19 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T9331 | D5900 | 2 | No |
| Z | T8581 | D1407 | 5 | Yes |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 1 |
| All original edges within each final UF cluster | 0 | No | 1 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
2130, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 2131 | 1 | 2131 | L1, L4, L6, L11 |
| F57 removed | 2131 | 81 | 2131 | L1, L11 |
| F82 removed | 2259 | 209 | 2259 | L1, L11 |
| F57 and F82 removed | 2259 | 209 | 2259 | L1, L11 |
| F57 alone | 0 | 0 | 0 | None |
| F82 alone | 128 | 128 | 128 | None |
| F57 and F82 alone | 128 | 128 | 128 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing both also leaves L1 and L11 wrong. F82 itself changes the recorded L7
observable, so the altered syndrome must be compared to the altered actual logical
outcome. The temporal component from the ancilla fault receives no correlation discount;
the data-Y component does. Both matching decoders nevertheless succeed on the original
complete shot.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L1 | Z0 | T8781 | e8972, e10453, e11937, e11942, e11913, e11878, e11808 |
| L4 | X2 | T9331 | e29564, e31048, e29639, e28162, e28232, e28273, e28278, e29758, e28331 |
| L6 | X3 | T8472 | e854, e853, e893, e2495, e2532, e3974, e5414, e6854, e8296, e6925, e6924, e5490, e4050, e2612 |
| L11 | Z5 | T8581 | e5928, e5930, e7377, e7382, e7353, e7315, e5808 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 1 to 2131. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 991 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
463 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_991`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 609](shot_609.md) · [Next: shot 1494](shot_1494.md)
