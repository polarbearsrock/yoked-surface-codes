# Shot 64: A cancelled data-fault endpoint stays outside a boundary cluster

The selected data-Y connection joins an unfired singleton to a detector already absorbed
by a boundary cluster. It remains incomplete. A readout edge in the X-yoke tree is
retained but unused. Both matching decoders select these target connections and predict
the original shot correctly.

This is row 64 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
401 sampled physical fault events and 792 fired detectors. The measured yoke bits are
X=1, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 1284 | — |
| Repository UF | 4 | L8, L10 |
| Ordinary joint MWPM | 1284 | None |
| Correlated MWPM | 1284 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Z_L,4 Z_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (0, 0) | (0, 0) | None |
| 1 | (1, 0) | (1, 0) | None |
| 2 | (0, 0) | (0, 0) | None |
| 3 | (0, 0) | (0, 0) | None |
| 4 | (1, 0) | (0, 0) | X |
| 5 | (1, 0) | (0, 0) | X |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F267 | patch 4, round 19; tick 190; instruction 6825 | DEPOLARIZE1(0.006) | Y on data q237 at (5, 1) | D5696, D5697, D5702, D5703 | None |
| F277 | patch 4, round 20; tick 200; instruction 6860 | M(0.015) | readout on ancilla q504 at (1.5, 2.5) | D5676, D5964 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F267's component e27328, D5697–D5702. Its original weight is 4.132583800266,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D5697<br/>observed syndrome 0"]
    B["D5702<br/>observed syndrome 1"]
    A ---|"e27328: weight 4.132584"| B
    C["UF cluster 5697<br/>1 vertex, 0 fired"]
    D["UF cluster 5702<br/>2 vertices, 1 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| D
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D5697 | 0 | [36.5, 1.5, 19.0] | 1 vertex, 0 fired; even detector parity |
| D5702 | 1 | [37.5, 0.5, 19.0] | 2 vertices, 1 fired; boundary terminal |

The endpoints finish in different inactive clusters. The connection has grown
3.490897703677 of its required 4.132583800266 and never enters the forest. One side
stops because of even detector parity; the other stops because of a boundary terminal.

| Endpoint | Incident edges selected by UF |
|---|---|
| D5697 | None |
| D5702 | e27332 |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D5697 | F266, F267 | 2 mod 2 = 0 |

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
| 3.490897704 | 3.490897704 | 0 fired, even; inactive | 1 fired, boundary; inactive | 0 | Activity changed |

Required growth is 4.132583800266; final accumulated growth is 3.490897703677. Final
edge status: incomplete between final clusters.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e27328: D5697–D5702 |
| First-pass selected source | e28767: D5696–D5703 |
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
saved observable mask 1284. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 1284 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F277, the selected diagnostic component is e28628, D5676–D5964. Its observable label
is None, and its weight is 3.678303297125. Its growth outcome is: forest edge; unused by
peeling.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | Yes |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D5676 | 1 | 146 vertices, 131 fired; boundary terminal |
| D5964 | 1 | 146 vertices, 131 fired; boundary terminal |

This target receives no correlation discount. Its use by correlated MWPM is part of the
complete matching correction.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 37 / 37 / 1 | 146 / 131 / 1 | T9690 | 24, 11, 12, 30, 22, 31 |
| Z | 7 / 6 / 0 | 200 / 98 / 0 | None | 26, 0, 14, 14, 26, 18 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T9690 | D8058 | 5 | Yes |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 4 |
| All original edges within each final UF cluster | 0 | No | 4 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
1280, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 1284 | 4 | 1284 | L8, L10 |
| F267 removed | 1284 | 1284 | 1284 | None |
| F277 removed | 1284 | 1284 | 1284 | None |
| F267 and F277 removed | 1284 | 1284 | 1284 | None |
| F267 alone | 0 | 0 | 0 | None |
| F277 alone | 0 | 0 | 0 | None |
| F267 and F277 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing F267, removing F277, or removing both fixes UF. The data-error connection
receives a correlation discount, while the temporal readout edge does not. The X-yoke
tree itself is odd and touches one terminal, but changing peeling within the original
partition cannot recover the missing L8 and L10 logical change.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L8 | X4 | T9296 | e30033, e30071, e30068, e28628, e27220, e27218, e27291, e27328, e27332 |
| L10 | X5 | T9640 | e37473, e37471, e37516, e37521, e37552, e37587, e37616, e37648, e37652 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 4 to 1284. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 64 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
401 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_64`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 45](shot_45.md) · [Next: shot 78](shot_78.md)
