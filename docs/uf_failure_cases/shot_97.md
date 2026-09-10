# Shot 97: Removing a late XY fault repairs only the Z sector

UF fails in both sectors. The late XY fault F389 supplies a connection that enters the
forest, but peeling leaves it unused. Removing that physical event fixes the Z-logical
pair while L2 and L4 remain wrong. Ordinary MWPM also fails, with a different full
prediction.

This is row 97 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
423 sampled physical fault events and 764 fired detectors. The measured yoke bits are
X=1, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3162 | — |
| Repository UF | 3790 | L2, L4, L7, L9 |
| Ordinary joint MWPM | 3222 | L2, L3, L6, L7 |
| Correlated MWPM | 3162 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Z_L,1 Z_L,2 X_L,3 X_L,4`, ignoring global phase.
The decoder predicts observable flips; this notation does not describe literal physical
correction gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (0, 1) | (0, 1) | None |
| 1 | (0, 1) | (1, 1) | X |
| 2 | (1, 0) | (0, 0) | X |
| 3 | (1, 0) | (1, 1) | Z |
| 4 | (0, 0) | (0, 1) | Z |
| 5 | (1, 1) | (1, 1) | None |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F389 | patch 3, round 26; tick 256; instruction 8809 | DEPOLARIZE2(0.003) | X on data q166 at (2, 1); Y on ancilla q455 at (1.5, 1.5) | D7355, D7644, D7650, D7651 | None |
| F253 | patch 4, round 17; tick 161; instruction 5849 | DEPOLARIZE1(0.006) | Y on data q209 at (1, 1) | D4804, D4805, D4810, D4811 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F389's component e37050, D7355–D7650. Its original weight is 6.436150368370,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D7355<br/>observed syndrome 1"]
    B["D7650<br/>observed syndrome 1"]
    A ---|"e37050: weight 6.436150"| B
    C["Same final UF cluster<br/>196 vertices, 112 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| C
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D7355 | 1 | [25.5, 1.5, 25.0] | 196 vertices, 112 fired; even detector parity |
| D7650 | 1 | [26.5, 0.5, 26.0] | 196 vertices, 112 fired; even detector parity |

The edge enters the forest at growth time 3.770173154295; peeling subsequently leaves it
unselected. Having the physical event's direct edge available is therefore insufficient
to identify the final correction. The UF edges incident to its endpoints are shown
below.

| Endpoint | Incident edges selected by UF |
|---|---|
| D7355 | e35584 |
| D7650 | e37052 |

| Decoder | Selects the highlighted target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

**Recorded growth transitions for the highlighted edge.**

The initial rate is 2, counting active detector endpoints; a virtual terminal never
initiates growth. The table records changes in this edge's rate or internal status.
Other unions and parity-preserving size changes are omitted. Times are algorithmic
growth coordinates, not circuit rounds or software latency.

| Growth time | Accumulated growth | First endpoint cluster | Second endpoint cluster | Rate after event | Edge event |
|---|---|---|---|---|---|
| 1.772435451 | 3.544870903 | 1 fired, odd; active | 24 fired, even; inactive (yoke) | 1 | Activity changed |
| 1.905867846 | 3.678303297 | 1 fired, odd; active | 39 fired, odd; active (yoke) | 2 | Activity changed |
| 2.133008097 | 4.132583800 | 1 fired, odd; active | 42 fired, even; inactive (yoke) | 1 | Activity changed |
| 2.310088858 | 4.309664561 | 1 fired, odd; active | 45 fired, odd; active (yoke) | 2 | Activity changed |
| 2.365193159 | 4.419873162 | 1 fired, odd; active | 50 fired, even; inactive (yoke) | 1 | Activity changed |
| 2.890562684 | 4.945242688 | 1 fired, odd; active | 57 fired, odd; active (yoke) | 2 | Activity changed |
| 3.435143698 | 6.034404715 | 1 fired, odd; active | 84 fired, even; inactive (yoke) | 1 | Activity changed |
| 3.703456957 | 6.302717974 | 1 fired, odd; active | 85 fired, odd; active (yoke) | 2 | Activity changed |
| 3.770173154 | 6.436150368 | 86 fired, even; inactive (yoke) | 86 fired, even; inactive (yoke) | 0 | Entered forest |

Required growth is 6.436150368370; final accumulated growth is 6.436150368370. Final
edge status: forest edge; unused by peeling.

**Why a grown edge can be unused.**

Growing adds an edge to the available forest; peeling chooses a subset of that forest as
the correction. Cutting the highlighted tree edge splits its final tree into these two
components:

| Side containing | Vertices | Fired detectors | Boundary terminals | Yokes |
|---|---|---|---|---|
| D7355 | 9 | 2 | None | None |
| D7650 | 187 | 110 | None | D8353 |

The D7355 side has 2 fired detectors and no boundary terminal. Its even parity requires
zero selected correction edges across this one-edge cut. Thus peeling cannot simply
select the highlighted edge while leaving this forest and syndrome otherwise unchanged.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e37050: D7355–D7650 |
| First-pass selected source | e37060: D7644–D7651 |
| Original target weight | 6.436150368370 |
| Reweighted target weight | 4.354526817561 |
| Source marginal probability | 0.0157881144111313 |
| Shared DEM mechanism probability | 0.000200280561291177 |
| Clipped implied probability | 0.0126855276111991 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 3162. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3162 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F253, the selected diagnostic component is e24294, D4804–D4811. Its observable label
is None, and its weight is 4.562635883352. Its growth outcome is: incomplete between
final clusters.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D4804 | 0 | 1 vertex, 0 fired; even detector parity |
| D4811 | 1 | 196 vertices, 112 fired; even detector parity |

At D4804, the recovered contributions are F253, F262; 2 contributions have even parity,
leaving the observed bit zero.

The selected first-pass source is e22855 (D4805–D4810); it lowers the target weight from
4.562635883352 to 0.713850186433. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 40 / 40 / 0 | 789 / 160 / 0 | T8904 | 24, 29, 20, 37, 20, 29 |
| Z | 24 / 24 / 0 | 196 / 112 / 0 | None | 11, 23, 14, 17, 27, 19 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T8904 | D3350 | 3 | No |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 3790 |
| All original edges within each final UF cluster | 0 | No | 3790 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
660, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3162 | 3790 | 3162 | L2, L4, L7, L9 |
| F389 removed | 3162 | 3150 | 3162 | L2, L4 |
| F253 removed | 3162 | 3790 | 3162 | L2, L4, L7, L9 |
| F389 and F253 removed | 3162 | 3150 | 3162 | L2, L4 |
| F389 alone | 0 | 0 | 0 | None |
| F253 alone | 0 | 0 | 0 | None |
| F389 and F253 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing F253 alone leaves the original UF discrepancy. Removing both selected faults
has the same partial repair as removing F389: the X-logical pair survives. The data-Y
connection associated with F253 is incomplete and has an unfired endpoint, whereas the
XY connection is fully available. Neither local observation by itself explains the
remaining X-sector failure.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L2 | X1 | T9179 | e25004, e25044, e25041, e25077, e25123, e23720, e25198, e23771 |
| L4 | X2 | T9329 | e26663, e28146, e28180, e28212, e29656, e29726, e29728, e29693, e29735, e28300 |
| L7 | Z3 | T9573 | e37052, e37050, e35584, e36989, e36994, e37043, e35568 |
| L9 | Z4 | T9006 | e24291, e24292, e24294, e22862, e21421, e21430, e21471, e20036, e20078, e18606 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 3790 to 3162. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 97 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
423 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_97`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 78](shot_78.md) · [Next: shot 128](shot_128.md)
