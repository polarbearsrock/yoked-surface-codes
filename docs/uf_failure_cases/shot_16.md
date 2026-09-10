# Shot 16: A retained readout edge cannot repair the final Z-logical partition

A round-26 readout connection is already present in the UF forest but unused. A second,
earlier data-Y connection remains incomplete between a boundary-containing cluster and
an unfired singleton. The correct full logical answer is unavailable even after all
graph edges internal to the final clusters are restored.

This is row 16 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
379 sampled physical fault events and 717 fired detectors. The measured yoke bits are
X=1, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3826 | — |
| Repository UF | 1650 | L7, L11 |
| Ordinary joint MWPM | 3826 | None |
| Correlated MWPM | 3826 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,3 X_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (0, 1) | (0, 1) | None |
| 1 | (0, 0) | (0, 0) | None |
| 2 | (1, 1) | (1, 1) | None |
| 3 | (1, 1) | (1, 0) | Z |
| 4 | (0, 1) | (0, 1) | None |
| 5 | (1, 1) | (1, 0) | Z |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F349 | patch 5, round 26; tick 260; instruction 8822 | M(0.015) | readout on ancilla q567 at (3.5, 3.5) | D7467, D7755 | None |
| F283 | patch 3, round 22; tick 211; instruction 7484 | DEPOLARIZE1(0.006) | Y on data q196 at (6, 3) | D6232, D6233, D6238 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F349's component e37591, D7467–D7755. Its original weight is 3.678303297125,
and its observable label is None. Correlated MWPM selects this edge; UF does not. This
component accounts for the event's entire detector response.

```mermaid
flowchart LR
    A["D7467<br/>observed syndrome 1"]
    B["D7755<br/>observed syndrome 1"]
    A ---|"e37591: weight 3.678303"| B
    C["Same final UF cluster<br/>8 vertices, 7 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| C
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D7467 | 1 | [43.5, 3.5, 25.0] | 8 vertices, 7 fired; boundary terminal |
| D7755 | 1 | [43.5, 3.5, 26.0] | 8 vertices, 7 fired; boundary terminal |

The edge enters the forest at growth time 1.839151648562; peeling subsequently leaves it
unselected. Having the physical event's direct edge available is therefore insufficient
to identify the final correction. The UF edges incident to its endpoints are shown
below.

| Endpoint | Incident edges selected by UF |
|---|---|
| D7467 | e37627 |
| D7755 | e39031 |

| Decoder | Selects the highlighted target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | Yes |
| joint_mwpm_correlated | Yes |

**Recorded growth transitions for the highlighted edge.**

The initial rate is 2, counting active detector endpoints; a virtual terminal never
initiates growth. The table records changes in this edge's rate or internal status.
Other unions and parity-preserving size changes are omitted. Times are algorithmic
growth coordinates, not circuit rounds or software latency.

| Growth time | Accumulated growth | First endpoint cluster | Second endpoint cluster | Rate after event | Edge event |
|---|---|---|---|---|---|
| 1.839151649 | 3.678303297 | 3 fired, odd; active | 3 fired, odd; active | 0 | Entered forest |

Required growth is 3.678303297125; final accumulated growth is 3.678303297125. Final
edge status: forest edge; unused by peeling.

**Why a grown edge can be unused.**

Growing adds an edge to the available forest; peeling chooses a subset of that forest as
the correction. Cutting the highlighted tree edge splits its final tree into these two
components:

| Side containing | Vertices | Fired detectors | Boundary terminals | Yokes |
|---|---|---|---|---|
| D7467 | 6 | 5 | T9591 | None |
| D7755 | 2 | 2 | None | None |

The D7755 side has 2 fired detectors and no boundary terminal. Its even parity requires
zero selected correction edges across this one-edge cut. Thus peeling cannot simply
select the highlighted edge while leaving this forest and syndrome otherwise unchanged.

**What correlation changes locally.**

This edge receives no correlation discount: its weight remains 3.678303297125.
Correlated MWPM still selects it as part of its global solution. Its success cannot be
attributed to lowering this particular edge's cost. Correlation updates elsewhere and
the choice of a complete correction must be considered separately.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3826. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3826 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F283, the selected diagnostic component is e30005, D6233–D6238. Its observable label
is None, and its weight is 4.294890877898. Its growth outcome is: incomplete between
final clusters.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D6233 | 1 | 13 vertices, 7 fired; boundary terminal |
| D6238 | 0 | 1 vertex, 0 fired; even detector parity |

At D6238, the recovered contributions are F266, F283; 2 contributions have even parity,
leaving the observed bit zero.

The selected first-pass source is e29980 (D6232–boundary); it lowers the target weight
from 4.294890877898 to 1.533005626579. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 39 / 39 / 1 | 113 / 106 / 0 | T9258 | 12, 14, 21, 15, 20, 23 |
| Z | 43 / 43 / 1 | 116 / 102 / 0 | None | 11, 10, 17, 20, 23, 20 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T9258 | D5466 | 5 | No |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 1650 |
| All original edges within each final UF cluster | 0 | No | 1650 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
2176, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3826 | 1650 | 3826 | L7, L11 |
| F349 removed | 3826 | 3826 | 3826 | None |
| F283 removed | 3826 | 3826 | 3826 | None |
| F349 and F283 removed | 3826 | 3826 | 3826 | None |
| F349 alone | 0 | 0 | 0 | None |
| F283 alone | 0 | 0 | 0 | None |
| F349 and F283 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing either F349 or F283 fixes UF, as does removing both. The readout edge receives
no correlation discount, whereas the data-Y component does. Both matching decoders
succeed on the original shot. This case separates a retained local physical explanation
from the missing inter-cluster connectivity needed to repair L7 and L11.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L7 | Z3 | T9430 | e27135, e25681, e27122, e27136, e28576, e30016, e30005, e29988, e31392, e31394, e31326 |
| L11 | Z5 | T9591 | e39048, e39050, e39021, e40466, e39031, e37591, e37627, e37673, e36202 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 1650 to 3826. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 16 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
379 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_16`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Next: shot 45](shot_45.md)
