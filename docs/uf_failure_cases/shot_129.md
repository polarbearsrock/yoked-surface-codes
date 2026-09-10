# Shot 129: A boundary fault links an X-logical flip to a Z-sector mistake

The highlighted YX fault lies at the boundary of patch 5 and itself flips the patch's
X-logical observable L10. UF's full-shot mistakes are instead Z-logical observables L9
and L11. The same physical event supplies evidence across sectors, including a
discounted connection that UF freezes internally.

This is row 129 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
382 sampled physical fault events and 709 fired detectors. The measured yoke bits are
X=1, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 118 | — |
| Repository UF | 2678 | L9, L11 |
| Ordinary joint MWPM | 2678 | L9, L11 |
| Correlated MWPM | 118 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `X_L,4 X_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F100 | patch 5, round 8; tick 78; instruction 2930 | DEPOLARIZE2(0.003) | Y on data q255 at (0, 4); X on ancilla q541 at (-0.5, 3.5) | D2545, D2551, D2552, D8352 | L10 |
| F89 | patch 5, round 7; tick 70; instruction 2901 | DEPOLARIZE1(0.006) | Y on data q261 at (1, 3) | D2262, D2263, D2268, D2269 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F100's component e12994, D2545–D2552. Its original weight is 4.562635883352,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D2545<br/>observed syndrome 1"]
    B["D2552<br/>observed syndrome 0"]
    A ---|"e12994: weight 4.562636"| B
    C["Same final UF cluster<br/>214 vertices, 97 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| C
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D2545 | 1 | [39.5, 3.5, 8.0] | 214 vertices, 97 fired; boundary terminal |
| D2552 | 0 | [40.5, 4.5, 8.0] | 214 vertices, 97 fired; boundary terminal |

At growth time 4.649426612617, other connections merge the endpoints into one cluster.
The edge has grown only 4.108355380211 of its required 4.562635883352; it freezes
internally and never enters the forest. Immediately before the merge, the endpoint
clusters contain 23 and 0 fired detectors.

| Endpoint | Incident edges selected by UF |
|---|---|
| D2545 | e12998 |
| D2552 | None |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D2552 | F100, F105 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e12994: D2545–D2552 |
| First-pass selected source | e11553: D2551–D8352 |
| Original target weight | 4.562635883352 |
| Reweighted target weight | 1.483109493224 |
| Source marginal probability | 0.0280621299846357 |
| Shared DEM mechanism probability | 0.00519032127620351 |
| Clipped implied probability | 0.184958208056383 |

The rule uses min(1/2, shared probability / source probability) to obtain the implied
probability, then lowers the target's log-odds weight if appropriate. The same recovered
physical fault supplies both graph components. The source is selected in the first pass;
it is inferred evidence, not knowledge of the physical-fault log.

Reconstructing all second-pass weights in floating point reproduces correlated MWPM's
saved observable mask 118. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 118 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 40 / 40 / 0 | 671 / 128 / 0 | None | 22, 21, 15, 21, 26, 22 |
| Z | 8 / 7 / 1 | 214 / 97 / 1 | T8916 | 7, 16, 25, 10, 15, 24 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 2678 |
| All original edges within each final UF cluster | 0 | No | 2678 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
2560, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 118 | 2678 | 118 | L9, L11 |
| F100 removed | 1142 | 1142 | 1142 | None |
| F89 removed | 118 | 118 | 118 | None |
| F100 and F89 removed | 1142 | 1142 | 1142 | None |
| F100 alone | 1024 | 1024 | 1024 | None |
| F89 alone | 0 | 0 | 0 | None |
| F100 and F89 alone | 1024 | 1024 | 1024 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

Deleting the boundary fault changes both the syndrome and the actual logical mask;
comparing to the original truth would give the wrong intervention result. With the truth
adjusted, removing either highlighted event makes UF correct. Correlated MWPM also
succeeds on both altered shots.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L9 | Z4 | T8957 | e17132, e17134, e18579, e17146, e18591, e17158, e17088 |
| L11 | Z5 | T8916 | e10172, e10174, e10144, e11590, e10157, e11598, e12994, e12998, e14475, e14477, e15880, e15879 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 2678 to 118. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 129 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
382 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_129`. [The collection guide](README.md)
records common hashes, the method, and the limits of the diagnostic interventions. These
are deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 125](shot_125.md) · [Next: shot 145](shot_145.md)
