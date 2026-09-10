# Shot 188: A late Z-fault connection stays outside the final UF partition

A late data-Z fault supplies an edge between the odd, boundary-containing X-yoke cluster
and an unfired singleton detector. The edge remains incomplete and outside the forest.
Restoring every edge internal to the final clusters still cannot recover the required
X-logical change.

This is row 188 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
396 sampled physical fault events and 765 fired detectors. The measured yoke bits are
X=0, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3473 | — |
| Repository UF | 2448 | L0, L10 |
| Ordinary joint MWPM | 3473 | None |
| Correlated MWPM | 3473 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Z_L,0 Z_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (1, 0) | (0, 0) | X |
| 1 | (0, 0) | (0, 0) | None |
| 2 | (1, 0) | (1, 0) | None |
| 3 | (0, 1) | (0, 1) | None |
| 4 | (1, 0) | (1, 0) | None |
| 5 | (1, 1) | (0, 1) | X |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F373 | patch 5, round 26; tick 260; instruction 9114 | DEPOLARIZE1(0.006) | Z on data q268 at (2, 3) | D7740, D7749 | None |
| F198 | patch 0, round 14; tick 138; instruction 4892 | DEPOLARIZE2(0.003) | Y on data q8 at (1, 0); X on ancilla q303 at (0.5, -0.5) | D4035, D4036, D4042, D8353 | L1 |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F373's component e38985, D7740–D7749. Its original weight is 4.562635883352,
and its observable label is None. Correlated MWPM selects this edge; UF does not. This
component accounts for the event's entire detector response.

```mermaid
flowchart LR
    A["D7740<br/>observed syndrome 1"]
    B["D7749<br/>observed syndrome 0"]
    A ---|"e38985: weight 4.562636"| B
    C["UF cluster 1450<br/>160 vertices, 77 fired"]
    D["UF cluster 7749<br/>1 vertex, 0 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D7740 | 1 | [41.5, 2.5, 26.0] | 160 vertices, 77 fired; boundary terminal |
| D7749 | 0 | [42.5, 3.5, 26.0] | 1 vertex, 0 fired; even detector parity |

The endpoints finish in different inactive clusters. The connection has grown
3.608756355109 of its required 4.562635883352 and never enters the forest. One side
stops because of a boundary terminal; the other stops because of even detector parity.

| Endpoint | Incident edges selected by UF |
|---|---|
| D7740 | e37508 |
| D7749 | None |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D7749 | F373, F380 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

| Decoder | Selects the highlighted target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | Yes |
| joint_mwpm_correlated | Yes |

**Recorded growth transitions for the highlighted edge.**

The initial rate is 1, counting active detector endpoints; a virtual terminal never
initiates growth. The table records changes in this edge's rate or internal status.
Other unions and parity-preserving size changes are omitted. Times are algorithmic
growth coordinates, not circuit rounds or software latency.

| Growth time | Accumulated growth | First endpoint cluster | Second endpoint cluster | Rate after event | Edge event |
|---|---|---|---|---|---|
| 2.066291900 | 2.066291900 | 4 fired, even; inactive | 0 fired, even; inactive | 0 | Activity changed |
| 2.496343983 | 2.066291900 | 5 fired, odd; active | 0 fired, even; inactive | 1 | Activity changed |
| 3.803326642 | 3.373274558 | 28 fired, even; inactive (yoke) | 0 fired, even; inactive | 0 | Activity changed |
| 4.108736017 | 3.373274558 | 39 fired, odd; active (yoke) | 0 fired, even; inactive | 1 | Activity changed |
| 4.335876269 | 3.600414810 | 68 fired, even; inactive (yoke) | 0 fired, even; inactive | 0 | Activity changed |
| 4.638929828 | 3.600414810 | 73 fired, odd; active (yoke) | 0 fired, even; inactive | 1 | Activity changed |
| 4.647271373 | 3.608756355 | 74 fired, boundary; inactive (yoke) | 0 fired, even; inactive | 0 | Activity changed |

Required growth is 4.562635883352; final accumulated growth is 3.608756355109. Final
edge status: incomplete between final clusters.

**What correlation changes locally.**

This edge receives no correlation discount: its weight remains 4.562635883352.
Correlated MWPM still selects it as part of its global solution. Its success cannot be
attributed to lowering this particular edge's cost. Correlation updates elsewhere and
the choice of a complete correction must be considered separately.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3473. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3473 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F198, the selected diagnostic component is e20449, D4035–D4042. Its observable label
is None, and its weight is 4.562635883352. Its growth outcome is: forest edge; unused by
peeling.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | Yes |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D4035 | 1 | 160 vertices, 77 fired; boundary terminal |
| D4042 | 1 | 160 vertices, 77 fired; boundary terminal |

The selected first-pass source is e19011 (D4036–D8353); it lowers the target weight from
4.562635883352 to 1.457969043983. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 15 / 14 / 0 | 160 / 77 / 1 | T9024 | 39, 0, 5, 12, 3, 18 |
| Z | 10 / 9 / 1 | 179 / 72 / 0 | None | 14, 16, 12, 13, 4, 13 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T9024 | D4070 | 0 | Yes |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 2448 |
| All original edges within each final UF cluster | 0 | No | 2448 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
1025, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3473 | 2448 | 3473 | L0, L10 |
| F373 removed | 3473 | 3473 | 3473 | None |
| F198 removed | 3475 | 3475 | 3475 | None |
| F373 and F198 removed | 3475 | 3475 | 3475 | None |
| F373 alone | 0 | 0 | 0 | None |
| F198 alone | 2 | 2 | 2 | None |
| F373 and F198 alone | 2 | 2 | 2 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

The other highlighted event, F198, is a YX fault whose full response includes L1,
although the selected target component has no observable label. Removing it changes the
actual logical outcome; the intervention uses that adjusted truth. Removing either F198
or F373 fixes UF. Both matching decoders select the highlighted target connections and
decode the original shot correctly.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L0 | X0 | T9024 | e20420, e20449, e20493, e19057, e17652, e19129, e19133, e19168, e19172 |
| L10 | X5 | T9642 | e36033, e36071, e37508, e38985, e39032, e37597, e37669, e37672 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 2448 to 3473. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 188 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
396 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_188`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 156](shot_156.md) · [Next: shot 239](shot_239.md)
