# Shot 609: Two early faults repair the X sector while the Z sector remains wrong

Two round-2 faults in patch 3 contribute to a full-shot failure in both logical sectors.
The highlighted connection from F23 remains between separate boundary-containing
clusters. Removing either event repairs the X-logical pair but leaves L7 and L11 wrong.

This is row 609 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
419 sampled physical fault events and 816 fired detectors. The measured yoke bits are
X=0, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3947 | — |
| Repository UF | 939 | L6, L7, L10, L11 |
| Ordinary joint MWPM | 3887 | L2, L6 |
| Correlated MWPM | 3947 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Y_L,3 Y_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (1, 1) | (1, 1) | None |
| 1 | (0, 1) | (0, 1) | None |
| 2 | (0, 1) | (0, 1) | None |
| 3 | (1, 0) | (0, 1) | X, Z |
| 4 | (1, 1) | (1, 1) | None |
| 5 | (1, 1) | (0, 0) | X, Z |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F23 | patch 3, round 2; tick 14; instruction 954 | DEPOLARIZE1(0.0003) | Y on data q168 at (2, 3) | D444, D445, D452, D741 | None |
| F28 | patch 3, round 2; tick 18; instruction 968 | DEPOLARIZE2(0.003) | Y on data q195 at (6, 2); I on ancilla q483 at (5.5, 1.5) | D759, D760, D766 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F23's component e2507, D444–D741. Its original weight is 5.625379320882, and
its observable label is None. Correlated MWPM selects this edge; UF does not. The full
detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D444<br/>observed syndrome 1"]
    B["D741<br/>observed syndrome 0"]
    A ---|"e2507: weight 5.625379"| B
    C["UF cluster 2837<br/>228 vertices, 79 fired"]
    D["UF cluster 170<br/>18 vertices, 4 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D444 | 1 | [25.5, 2.5, 1.0] | 228 vertices, 79 fired; boundary terminal |
| D741 | 0 | [26.5, 3.5, 2.0] | 18 vertices, 4 fired; boundary terminal |

The endpoints finish in different inactive clusters. The connection has grown
3.524120865877 of its required 5.625379320882 and never enters the forest. One side
stops because of a boundary terminal; the other stops because of a boundary terminal.

| Endpoint | Incident edges selected by UF |
|---|---|
| D444 | e2463 |
| D741 | None |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D741 | F23, F31 | 2 mod 2 = 0 |

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
| 2.281317942 | 2.281317942 | 2 fired, even; inactive | 0 fired, even; inactive | 0 | Activity changed |
| 4.132583800 | 2.281317942 | 2 fired, even; inactive | 3 fired, odd; active | 1 | Activity changed |
| 5.304230062 | 3.452964204 | 2 fired, even; inactive | 4 fired, boundary; inactive | 0 | Activity changed |
| 8.393581161 | 3.452964204 | 69 fired, odd; active (yoke) | 4 fired, boundary; inactive | 1 | Activity changed |
| 8.464737824 | 3.524120866 | 79 fired, boundary; inactive (yoke) | 4 fired, boundary; inactive | 0 | Activity changed |

Required growth is 5.625379320882; final accumulated growth is 3.524120865877. Final
edge status: incomplete between final clusters.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e2507: D444–D741 |
| First-pass selected source | e953: D445–D452 |
| Original target weight | 5.625379320882 |
| Reweighted target weight | 3.069848658846 |
| Source marginal probability | 0.0157881144111313 |
| Shared DEM mechanism probability | 0.000700490911966637 |
| Clipped implied probability | 0.0443682439666614 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3947. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3947 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F28, the selected diagnostic component is e2620, D760–boundary. Its observable label
is None, and its weight is 3.390989621338. Its growth outcome is: forest edge; unused by
peeling.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D760 | 1 | 18 vertices, 4 fired; boundary terminal |
| T8473 | Unconstrained | 18 vertices, 4 fired; boundary terminal |

The selected first-pass source is e4082 (D759–D766); it lowers the target weight from
3.390989621338 to 0.000000000000. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 8 / 7 / 1 | 228 / 79 / 1 | T9593 | 17, 44, 4, 2, 3, 9 |
| Z | 35 / 35 / 1 | 225 / 160 / 0 | T9526, T9574 | 27, 25, 25, 35, 15, 32 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T9593 | D7480 | 5 | Yes |
| Z | T9526 | D7085 | 3 | No |
| Z | T9574 | D7373 | 3 | No |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 939 |
| All original edges within each final UF cluster | 0 | No | 939 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
3264, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3947 | 939 | 3947 | L6, L7, L10, L11 |
| F23 removed | 3947 | 2027 | 3947 | L7, L11 |
| F28 removed | 3947 | 2027 | 3947 | L7, L11 |
| F23 and F28 removed | 3947 | 2027 | 3947 | L7, L11 |
| F23 alone | 0 | 0 | 0 | None |
| F28 alone | 0 | 0 | 0 | None |
| F23 and F28 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing both selected faults still leaves that Z-logical pair. F28's boundary edge is
in the forest but unused, while F23's direct connection never completes and includes an
unfired endpoint. Even though the Z-yoke tree has two terminals, the fixed partition has
zero logical-change rank: those boundary choices do not supply the missing logical
freedom.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L6 | X3 | T8473 | e856, e2463, e2507, e2547, e4021, e2622, e2620 |
| L7 | Z3 | T8613 | e1026, e1028, e2577, e2582, e2514, e2503, e3944, e5384, e6866, e5433, e6836, e6838, e6768 |
| L10 | X5 | T9593 | e33153, e33191, e33220, e33252, e34694, e36143, e36176, e37655, e36220 |
| L11 | Z5 | T8966 | e24648, e26127, e26162, e24736, e23287, e23261, e21782, e21751, e20311, e18873, e20318, e18882, e17406 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 939 to 3947. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 609 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
419 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_609`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 239](shot_239.md) · [Next: shot 991](shot_991.md)
