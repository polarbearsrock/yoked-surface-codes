# Shot 239: Retained physical edges accompany different UF and ordinary-MWPM failures

UF makes a Z-logical mistake, while ordinary MWPM makes an X-logical mistake; correlated
MWPM succeeds. Both highlighted physical connections are already in the UF forest, but
neither is selected by peeling. Their availability does not give the fixed partition
access to the correct full logical answer.

This is row 239 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
450 sampled physical fault events and 842 fired detectors. The measured yoke bits are
X=0, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 2570 | — |
| Repository UF | 2186 | L7, L9 |
| Ordinary joint MWPM | 3598 | L2, L10 |
| Correlated MWPM | 2570 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,3 X_L,4`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Reading the logical result by patch.**

| Patch | Actual flips (X, Z) | UF predicts (X, Z) | Wrong observable sector |
|---|---|---|---|
| 0 | (0, 1) | (0, 1) | None |
| 1 | (0, 1) | (0, 1) | None |
| 2 | (0, 0) | (0, 0) | None |
| 3 | (0, 0) | (0, 1) | Z |
| 4 | (0, 1) | (0, 0) | Z |
| 5 | (0, 1) | (0, 1) | None |

These are flip indicators relative to the ideal logical measurements: 1 means a flip and
0 means no flip. They are not the absolute encoded-state bits. Correlated MWPM matches
the actual column on every patch. A correction can satisfy every detector while
predicting the wrong logical flips; that leaves a residual logical error when its
correction or Pauli frame is used.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F389 | patch 4, round 25; tick 248; instruction 8489 | DEPOLARIZE2(0.003) | Y on data q232 at (4, 3); Y on ancilla q518 at (3.5, 2.5) | D7130, D7419, D7426, D7427 | None |
| F386 | patch 4, round 25; tick 242; instruction 8468 | DEPOLARIZE1(0.0003) | Y on data q233 at (4, 4) | D7131, D7132, D7139, D7140 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F389's component e35942, D7419–D7426. Its original weight is 4.132583800266,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D7419<br/>observed syndrome 1"]
    B["D7426<br/>observed syndrome 1"]
    A ---|"e35942: weight 4.132584"| B
    C["Same final UF cluster<br/>281 vertices, 102 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| C
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D7419 | 1 | [35.5, 3.5, 25.0] | 281 vertices, 102 fired; boundary terminal |
| D7426 | 1 | [36.5, 2.5, 25.0] | 281 vertices, 102 fired; boundary terminal |

The edge enters the forest at growth time 2.293432151703; peeling subsequently leaves it
unselected. Having the physical event's direct edge available is therefore insufficient
to identify the final correction. The UF edges incident to its endpoints are shown
below.

| Endpoint | Incident edges selected by UF |
|---|---|
| D7419 | e35911 |
| D7426 | e35940 |

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
| 1.839151649 | 3.678303297 | 2 fired, even; inactive | 1 fired, odd; active | 1 | Activity changed |
| 2.293432152 | 4.132583800 | 3 fired, odd; active | 3 fired, odd; active | 0 | Entered forest |

Required growth is 4.132583800266; final accumulated growth is 4.132583800266. Final
edge status: forest edge; unused by peeling.

**Why a grown edge can be unused.**

Growing adds an edge to the available forest; peeling chooses a subset of that forest as
the correction. Cutting the highlighted tree edge splits its final tree into these two
components:

| Side containing | Vertices | Fired detectors | Boundary terminals | Yokes |
|---|---|---|---|---|
| D7419 | 10 | 2 | None | None |
| D7426 | 271 | 100 | T8518 | D8353 |

The D7419 side has 2 fired detectors and no boundary terminal. Its even parity requires
zero selected correction edges across this one-edge cut. Thus peeling cannot simply
select the highlighted edge while leaving this forest and syndrome otherwise unchanged.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e35942: D7419–D7426 |
| First-pass selected source | e35943: D7130–D7427 |
| Original target weight | 4.132583800266 |
| Reweighted target weight | 1.417837019967 |
| Source marginal probability | 0.00359224460718038 |
| Shared DEM mechanism probability | 0.000700490911966637 |
| Clipped implied probability | 0.195000894584533 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 2570. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 2570 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The other traced physical connection.**

For F386, the selected diagnostic component is e35947, D7131–D7140. Its observable label
is None, and its weight is 4.562635883352. Its growth outcome is: forest edge; unused by
peeling.

| Decoder | Selects this target edge? |
|---|---|
| repository_uf | No |
| joint_mwpm_uncorrelated | No |
| joint_mwpm_correlated | Yes |

| Endpoint | Observed bit | Final cluster |
|---|---|---|
| D7131 | 1 | 281 vertices, 102 fired; boundary terminal |
| D7140 | 0 | 281 vertices, 102 fired; boundary terminal |

At D7140, the recovered contributions are F370, F386; 2 contributions have even parity,
leaving the observed bit zero.

The selected first-pass source is e34508 (D7132–D7139); it lowers the target weight from
4.562635883352 to 0.713850186433. That source is another component of this same
recovered physical event.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 28 / 27 / 1 | 263 / 89 / 1 | T9424 | 16, 12, 25, 20, 2, 14 |
| Z | 16 / 15 / 1 | 281 / 102 / 0 | T8518 | 13, 5, 14, 31, 20, 19 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Yoke tree | Terminal | Attached detector | Patch | Boundary edge selected by UF? |
|---|---|---|---|---|
| X | T9424 | D6470 | 2 | Yes |
| Z | T8518 | D1037 | 3 | No |

Several terminals can attach to the same patch. Their count does not determine the
number of independent logical changes; the logical-space calculation below tracks their
observable labels.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 2186 |
| All original edges within each final UF cluster | 0 | No | 2186 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
640, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 2570 | 2186 | 2570 | L7, L9 |
| F389 removed | 2570 | 2186 | 2570 | L7, L9 |
| F386 removed | 2570 | 2570 | 2570 | None |
| F389 and F386 removed | 2570 | 2570 | 2570 | None |
| F389 alone | 0 | 0 | 0 | None |
| F386 alone | 0 | 0 | 0 | None |
| F389 and F386 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Removing the YY fault F389 leaves UF's original L7 and L9 discrepancy. Removing the
data-Y fault F386 fixes it, and removing both is also correct. F386 has an unfired
endpoint because physical responses cancel. These results distinguish the visible
retained YY edge from a fault whose deletion changes the full decoding outcome.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L7 | Z3 | T8518 | e4008, e4010, e3978, e3986, e2553, e4000, e3966 |
| L9 | Z4 | T9487 | e34488, e35967, e34536, e35940, e35942, e35911, e35947, e34512, e33072, e33115, e33082 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 2186 to 2570. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 239 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
450 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-next-6a0xaw4f/shot_239`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 188](shot_188.md) · [Next: shot 609](shot_609.md)
