# Shot 23: UF and ordinary MWPM choose different wrong logical pairs

UF and ordinary MWPM both fail, but they predict different wrong X-logical pairs.
Correlated MWPM resolves the full shot. A data-Y connection used by correlated MWPM
becomes internal and freezes before completion in UF.

This is row 23 (zero-based) of the preserved seed-42, 100,000-shot experiment: 1D YSC,
d=7, SI1000 p=0.003, 28 noisy rounds, six physical patches, and two yokes. It contains
396 sampled physical fault events and 730 fired detectors. The measured yoke bits are
X=1, Z=1.

[Collection, notation, and replay instructions](README.md).

**The full-shot logical outcome.**

| Result | Observable mask | Wrong logical predictions |
|---|---|---|
| Actual sampled outcome | 2229 | — |
| Repository UF | 2484 | L0, L8 |
| Ordinary joint MWPM | 2224 | L0, L2 |
| Correlated MWPM | 2229 | None |

All three graph corrections satisfy every detector, including the yokes. UF's residual
has the logical commutation pattern `Z_L,0 Z_L,4`, ignoring global phase. The decoder
predicts observable flips; this notation does not describe literal physical correction
gates.

**Two actual physical events.**

| Event | Circuit location | Noise channel | Sampled error, local coordinates | Detector flips in isolation | Observable flips in isolation |
|---|---|---|---|---|---|
| F271 | patch 0, round 19; tick 190; instruction 6825 | DEPOLARIZE1(0.006) | Y on data q33 at (4, 4) | D5499, D5500, D5507, D5508 | None |
| F273 | patch 0, round 20; tick 196; instruction 6847 | DEPOLARIZE2(0.003) | Y on data q27 at (3, 5); Z on ancilla q328 at (3.5, 4.5) | D5494, D5500, D5783, D5789 | None |

Fault IDs are local to this shot. Qubit, patch, and detector IDs are zero-based; rounds
are numbered 1–28. Instruction indices expand repeat blocks and retain SHIFT_COORDS. An
I label means that target receives no Pauli error. These are recovered simulator draws,
not merely compatible DEM explanations. Individual detector flips can cancel when all
faults are combined.

**A local decoding decision.**

Focus on F271's component e26348, D5500–D5507. Its original weight is 4.132583800266,
and its observable label is None. Correlated MWPM selects this edge; UF does not. The
full detector footprint above also includes another graph component of the same event.

```mermaid
flowchart LR
    A["D5500<br/>observed syndrome 0"]
    B["D5507<br/>observed syndrome 1"]
    A ---|"e26348: weight 4.132584"| B
    C["Same final UF cluster<br/>243 vertices, 131 fired"]
    A -.->|"membership"| C
    B -.->|"membership"| C
```

The solid line is the target graph edge, and the dashed arrows describe final UF
membership. This is not the full correction or a diagram of physical gates. Detector
time and circuit round are different from algorithmic growth time.

| Endpoint | Full syndrome bit | Detector coordinates (global x, y, t) | Final cluster |
|---|---|---|---|
| D5500 | 0 | [3.5, 4.5, 19.0] | 243 vertices, 131 fired; boundary terminal |
| D5507 | 1 | [4.5, 3.5, 19.0] | 243 vertices, 131 fired; boundary terminal |

At growth time 6.436150368370, other connections merge the endpoints into one cluster.
The edge has grown only 2.303566568104 of its required 4.132583800266; it freezes
internally and never enters the forest. Immediately before the merge, the endpoint
clusters contain 0 and 7 fired detectors.

| Endpoint | Incident edges selected by UF |
|---|---|
| D5500 | None |
| D5507 | e26382 |

| Unfired endpoint | Actual fault contributions | Resulting detector bit |
|---|---|---|
| D5500 | F271, F273 | 2 mod 2 = 0 |

Correlated MWPM may pass through an unfired detector using an even number of incident
correction edges. It does not need to interpret that detector as a defect. The absence
of a fired bit is not evidence that the corresponding physical faults were absent.

**What correlation changes locally.**

| Quantity | Value |
|---|---|
| Target | e26348: D5500–D5507 |
| First-pass selected source | e27787: D5499–D5508 |
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
saved observable mask 2229. As a separate intervention, running the repository UF growth
and peeling on those first-pass-adjusted weights returns mask 2229 (correct). This
intervention uses an MWPM first pass and is not an independently benchmarked UF decoder.
No single-edge-only weight intervention is claimed.

**The full growth result and the freedom left to peeling.**

| Yoke | First cluster: vertices / fired / parity | Final cluster: vertices / fired / parity | Final terminals | Final ordinary member-defect counts, patches 0–5 |
|---|---|---|---|---|
| X | 39 / 39 / 1 | 243 / 131 / 1 | T9009 | 24, 23, 37, 10, 20, 16 |
| Z | 27 / 27 / 1 | 104 / 92 / 0 | None | 12, 10, 25, 15, 14, 15 |

The yoke belongs to one cluster and is counted once. Vertex count includes unfired
vertices and terminals; detector parity counts only fired detectors. A cluster can stop
with even detector parity or by touching an unconstrained terminal. For a yoke cluster
without terminals, its per-patch member-defect parities fix that sector of the UF
prediction.

| Allowed correction edges | Logical-change rank | Correct logical answer exists? | Restricted MWPM prediction |
|---|---|---|---|
| UF forest | 0 | No | 2484 |
| All original edges within each final UF cluster | 0 | No | 2484 |

An exhaustive GF(2) cycle-space certificate shows that the required logical change, mask
257, is unavailable even if every original edge internal to the final clusters is
restored. The same syndrome and partition therefore cannot be repaired by a different
peeling choice. A correct answer requires connections beyond that partition. This is
stronger than merely observing that restricted MWPM returns the wrong answer.

**Physical interventions.**

| Physical error pattern | Actual mask | UF mask | Correlated MWPM mask | UF wrong predictions |
|---|---|---|---|---|
| Original | 2229 | 2484 | 2229 | L0, L8 |
| F271 removed | 2229 | 2229 | 2229 | None |
| F273 removed | 2229 | 2229 | 2229 | None |
| F271 and F273 removed | 2229 | 2229 | 2229 | None |
| F271 alone | 0 | 0 | 0 | None |
| F273 alone | 0 | 0 | 0 | None |
| F271 and F273 alone | 0 | 0 | 0 | None |

Each deletion retains all other sampled events and the original decoder graph. The
syndrome and actual observables are changed by the removed events' independently
verified responses. Every resulting decoder correction still satisfies its input
syndrome. These counterfactuals distinguish a full repair, a partial repair, and a
change to a different wrong answer.

The separate ordinary-MWPM prediction matters here: the accuracy improvement is not
merely reproducing the uncorrelated matching result. Removing either selected physical
event fixes UF. The local weight discount and the full logical correction exchange are
recorded separately, so a local explanation is not mistaken for a complete correction.

**An explicit logical correction exchange.**

| Wrong observable | Patch observable | Boundary endpoint | Path from its yoke, edge IDs in order |
|---|---|---|---|
| L0 | X0 | T9265 | e21873, e23321, e23364, e24837, e24842, e26282, e27722, e27723, e26348, e26382, e26380 |
| L8 | X4 | T9009 | e21383, e21382, e19977, e20013, e18612, e20089, e20135, e18700 |

Every listed edge belongs to the symmetric difference of the complete UF and correlated
MWPM corrections. Toggle the paths in pairs within each sector; doing this for all
listed paths changes mask 2484 to 2229. Internal detector incidences change by an even
number, the paired paths cancel their yoke incidence, and only the virtual boundary
endpoints are unconstrained. The resulting correction was explicitly checked against
every detector and all 12 observables. A single yoke-to-boundary path is not by itself a
valid syndrome-preserving toggle.

The local edge example illustrates one decision, while these complete paths supply a
valid logical repair. Restoring just the highlighted edge, or deleting one selected
fault, is not asserted to reproduce the entire correlated MWPM correction.

**Replay and scope.**

The original arrays are in `out/union_find_correlated_comparison_d7_p003_100k_seed42`;
select row 23 consistently across detector, actual-observable, and prediction arrays.
The original 100,000-shot call was regenerated byte for byte using Stim 1.16.0 SSE2. All
396 recovered events reproduce this shot, both in aggregate and by XORing their
individual responses. A forced-fault circuit also reproduces it for eight checks with
the ordinary sampler using seeds 1 and 123456. Graph corrections and prediction masks
reproduce the saved results with PyMatching 2.4.0.

Detailed scripts, fault traces, and numerical records are local temporary data under
`$TMPDIR/uf-case-collection-9pbSDiuL/shot_23`. [The collection guide](README.md) records
common hashes, the method, and the limits of the diagnostic interventions. These are
deliberately selected case studies; they do not estimate mechanism frequencies or
establish a decoder change's accuracy. The repository decoder was not modified.

[All cases](README.md) · [Previous: shot 21](shot_21.md) · [Next: shot 75](shot_75.md)
