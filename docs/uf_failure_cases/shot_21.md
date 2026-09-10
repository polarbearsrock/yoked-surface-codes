# Shot 21: An initially even star later freezes a data-error connection

The first fired X-yoke star has even parity and initially pauses. Later growth produces
a much larger cluster, and a direct connection produced by a data-Y fault becomes
internal before its full weight has grown. Ordinary MWPM already succeeds on this shot.

This is row 21 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
394 sampled physical fault events and 746 fired detectors. The measured yoke bits are
X=1, Z=0.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 3308 | — |
| Repository UF | 2540 | L8, L10 |
| Ordinary joint MWPM | 3308 | None |
| Correlated MWPM | 3308 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Z_L,4 Z_L,5`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F112 | patch 4, round 8; tick 80; instruction 3228 | DEPOLARIZE1(0.006) | Y on data q217 at (2, 2) | D2507, D2508, D2515, D2516 | None |
| F88 | patch 4, round 7; tick 68; instruction 2603 | DEPOLARIZE2(0.003) | Y on data q240 at (5, 4); Y on ancilla q527 at (4.5, 3.5) | D1955, D2244, D2249, D2250 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F112's component e11380, D2508–D2515. Its original weight is 4.132583800266,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D2508<br/>observed syndrome 0"]
    B["D2515<br/>observed syndrome 1"]
    A ---|"e11380: weight 4.132584"| B
    C["Same final UF cluster<br/>1017 vertices, 144 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| C
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D2508 | 0 | [33.5, 2.5, 8.0] | 1017 vertices, 144 fired; boundary terminal |
| D2515 | 1 | [34.5, 1.5, 8.0] | 1017 vertices, 144 fired; boundary terminal |

At growth time 3.697022246669, other connections merge the endpoints into one cluster.
The edge has grown only 3.715741196212 of its required 4.132583800266; it freezes
internally and never enters the forest. Immediately before the merge, the endpoint
clusters contain 5 and 1 fired detectors.

| Endpoint | Incident edges selected by UF |
|---|---|
| D2508 | None |
| D2515 | e12828 |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D2508 | F99, F112 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e11380: D2508–D2515 |
| First-pass selected source | e12819: D2507–D2516 |
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
saved observable mask 3308. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 3308 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 28 / 28 / 0 | 1017 / 144 / 0 | T9017 | 20, 17, 20, 21, 33, 32 |
| Z | 11 / 10 / 0 | 102 / 40 / 0 | T8605 | 4, 5, 11, 1, 4, 15 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 2540 |
| All original edges within each final UF cluster | 0 | No | 2540 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
1280, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 3308 | 2540 | 3308 | L8, L10 |
| F112 removed | 3308 | 3308 | 3308 | None |
| F88 removed | 3308 | 3308 | 3308 | None |
| F112 and F88 removed | 3308 | 3308 | 3308 | None |
| F112 alone | 0 | 0 | 0 | None |
| F88 alone | 0 | 0 | 0 | None |
| F112 and F88 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

An even first star is an intermediate state, not a certificate of logical success. The
final cluster partition excludes the correct logical answer. Both highlighted
physical-fault removals fix UF; neither fault alone is uncorrectable.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L8 | X4 | T8722 | e9873, e9911, e11348, e11380, e12828, e12829, e12872, e11435, e9984, e10021, e10069, e10072 |
| L10 | X5 | T9017 | e10113, e10151, e11627, e13114, e13148, e14585, e16025, e17465, e18901, e17456, e18937, e18940 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 2540 to 3308. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 21 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
394 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_21`. [The collection guide](README.md) records
common hashes, the method, and the limits of the diagnostic interventions. These are
deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 13](shot_13.md) · [Next: shot 23](shot_23.md)
