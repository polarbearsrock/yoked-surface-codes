# Shot 10: a physical YY fault and the benefit of correlation-aware MWPM

Repository UF fails on this saved shot, while correlation-aware PyMatching
predicts every logical observable correctly. One relevant physical event is a
two-qubit **Y⊗Y fault in patch 0, round 24**, labelled **F324** in the recovered
fault trace. Correlation-aware decoding uses evidence in the Z-check sector to
lower the weight of a related X-check edge from **5.625 to 3.070**.

The failure depends on the complete error pattern. Both decoders succeed when
F324 is simulated alone. Removing F324 from the original shot, while retaining
the other 386 fault events, also makes UF succeed.

The companion [shot-0 readout-fault case](uf_correlated_mwpm_shot0.md) examines
a simultaneous UF merge at the yoke where ordinary MWPM also succeeds and the
relevant measurement-error edge receives no correlation discount.

The [shot-3148 boundary-choice case](uf_correlated_mwpm_shot3148.md) examines
peeling: changing only the terminal chosen as root fixes the logical prediction
within the same UF forest.

Browse the [full failure case collection](uf_failure_cases/README.md) for more
physical faults, correction exchanges, and verified interventions.

**Experiment and indexing.**

| Parameter | Value |
|---|---|
| Circuit | 1D yoked surface-code memory; CZ gates; ideal preparation, final boundaries, and yoke measurements |
| Distance and physical noise parameter | `d=7`, `p=0.003` |
| Noise model | Repository SI1000 |
| Noisy measurement rounds | 28 |
| Physical patches and yokes | 6 patches; X and Z yokes |
| Original sampling | Seed 42; one call sampling 100,000 shots |
| Selected shot | Index 10, the eleventh row of the original batch |
| Simulation and matching versions | Stim 1.16.0, SSE2; PyMatching 2.4.0 |
| UF implementation | Repository `UnionFindDecoder`, weighted growth and peeling |
| Shot contents | 387 physical fault events; 753 fired detectors |
| Measured yoke bits | X yoke: 0; Z yoke: 1 |

Patch, qubit, detector, observable, shot, and fault identifiers are zero-based.
Circuit rounds in this note are numbered 1–28. Detector coordinates `t23` and
`t24` correspond to noisy measurement rounds 24 and 25 in this circuit.

**The physical event.**

After 238 circuit `TICK`s, in round 24, a CZ involving qubits 30 and 324 was
followed by `DEPOLARIZE2(0.003)` noise. The sampled Pauli error on that target
pair was:

$$
Y_{30}Y_{324}.
$$

| Qubit | Role | Local coordinates within patch 0 | Sampled error |
|---|---|---|---|
| 30 | Data qubit | (4, 1) | Y |
| 324 | Check ancilla | (3.5, 0.5) | Y |

F324 is the event identifier; qubit 324 is a separate identifier. Their equal
numbers are incidental. The event belongs to expanded circuit instruction
index 8162, counting repeat iterations and retaining `SHIFT_COORDS` operations.

This location was recovered from the original simulator draws and independently
validated. It is more specific than an example physical location returned by
Stim's error-explanation API: several circuit faults can have the same detector
footprint.

**Detector effects.**

A detector indicates a change in a check measurement. Propagating this physical
fault through the circuit produces four detector flips, arranged into two
graph components:

| Component | Graph edge | Detector endpoints | Detector times |
|---|---|---|---|
| A: X-check component | `e33531` | D6648–D6945 | t23–t24 |
| B: Z-check component | `e33530` | D6937–D6944 | t24–t24 |

The physical Y errors contain X and Z components, so one physical event can
produce evidence in both sectors. Some effects appear in the next measurement
round because of the circuit's gate ordering. These graph edges describe
detector effects; their endpoints are detector identifiers, not data-qubit
identifiers.

The decomposed detector error model contains this shared mechanism:

```text
error(0.0007004909119666365) D6648 D6945 ^ D6937 D6944
```

The separator `^` splits one mechanism into graph components. It preserves the
fact that those components belong to the same error mechanism. The probability
on this DEM instruction combines compatible circuit mechanisms; it is not the
probability of this single physical YY draw. F324 itself flips no logical
observable.

**Why the edge weight changes.**

An ordinary weighted graph uses an edge's marginal error probability `q` to
assign the log-odds cost

$$
w(q)=\ln\left(\frac{1-q}{q}\right).
$$

A larger probability produces a smaller weight, making the edge less costly
to include in a correction. This is the weighting convention described in the
[PyMatching documentation](https://pymatching.readthedocs.io/en/stable/index.html#loading-from-a-parity-check-matrix).

For this edge pair, our replay of the PyMatching 2.4 rules gives:

| Quantity from the DEM | Probability |
|---|---:|
| Marginal probability of A | 0.0035922446071803764 |
| Marginal probability of B | 0.01578811441113132 |
| Shared-mechanism probability used for A and B | 0.0007004909119666365 |

The first MWPM pass selects B, the D6937–D6944 connection. PyMatching then uses
the shared mechanism to compute an implied probability for A:

$$
\widetilde q_A
=\min\left(\frac12,\frac{q_{\mathrm{shared}}}{q_B}\right)
=\frac{0.0007004909119666365}{0.01578811441113132}
=0.04436824396666137.
$$

It lowers A's weight when the implied weight is smaller than its existing
weight:

$$
w'_A=\min\bigl(w_A,w(\widetilde q_A)\bigr)
=3.069848658846031.
$$

| Information used for A | Effective probability | Weight |
|---|---:|---:|
| Marginal probability alone | 0.359224% | 5.625379 |
| Implied by first-pass edge B | 4.436824% | 3.069849 |

Evidence for B makes A more plausible because a shared physical fault can
explain both. MWPM runs a second pass with the updated weights, and in this
shot selects A. The probabilities and weights above are reconstructed in
floating point from the DEM rules; the reconstructed second pass agrees with
PyMatching's saved logical prediction.

This is an approximate use of conditional evidence, not an exact posterior
over all physical fault histories. In particular, selecting B in the first
pass is an inference rather than direct knowledge that B occurred. The shared
quantity is accumulated from correlated DEM mechanisms; it is not the full
joint probability including every combination of independent faults. The
decoder uses the syndrome and noise model, without access to the recovered
physical-fault list. The rule is implemented in PyMatching 2.4's
[`populate_implied_edge_weights`](https://github.com/oscarhiggott/PyMatching/blob/v2.4.0/src/pymatching/sparse_blossom/driver/user_graph.cc#L519).

**How UF and correlated MWPM differ locally.**

Both select the Z-check edge B. Their relevant X-check connections differ:

| Decoder | Selected local X-check edges |
|---|---|
| Repository UF | `e33488`: D6641–D6648; `e33533`: D6657–D6945 |
| Correlated MWPM | `e33531`: D6648–D6945 |

These are fragments of complete corrections. Other edges account for the
remaining detector endpoints, so the table is not a comparison of two complete
solutions by itself.

UF uses fixed marginal weights during growth. It selects the connections to
other nearby defects instead of A. In the recorded growth trace, A accumulates
4.562636 units of growth against its weight of 5.625379. Its endpoints become
members of the same cluster through other edges, and A is frozen without
entering the merge forest. Peeling that forest cannot subsequently select A.

The growth and peeling behavior is implemented in
[`_union_find.py`](../src/yoked/decoders/_union_find.py). Correlated MWPM has a
second opportunity to reconsider connections after incorporating the
cross-sector evidence.

**Logical outcome of the full shot.**

The observable ordering is `(X0, Z0, X1, Z1, ..., X5, Z5)`. Thus `L0` records
the patch-0 X-logical flip and `L2` records the patch-1 X-logical flip. A bit mask
is the integer obtained by setting bit `k` when observable `Lk` is predicted
to flip.

| Result | Observable mask | Incorrect observable predictions |
|---|---:|---|
| Actual sampled outcome | 2456 | — |
| Repository UF | 2461 | L0, L2 |
| Joint MWPM without correlations | 2201 | L0, L8 |
| Correlated MWPM | 2456 | None |

Both UF and correlated MWPM return graph corrections satisfying every measured
detector, including the yokes. UF's remaining logical error has the commutation
pattern

$$
Z_{L,0}Z_{L,1}.
$$

The two logical Z factors commute with the global X yoke as a pair and with
the Z yoke. Consequently, a correction can satisfy both yokes while making
these two patch-level logical predictions incorrectly. The notation describes
the residual logical error; the decoder returns observable predictions rather
than applying a literal sequence of physical correction gates.

**Interventions and limits of the conclusion.**

| Error pattern decoded | UF | Correlated MWPM |
|---|---|---|
| Original shot with all 387 events | Fails | Succeeds |
| Original shot with only F324 removed | Succeeds | Succeeds |
| F324 alone | Succeeds | Succeeds |

Removing F324 toggles D6648, D6937, D6944, and D6945 while leaving the actual
logical observable mask at 2456. UF's prediction then changes from 2461 to
2456. This establishes the event's influence with every other physical fault
held fixed. It does not make F324 a standalone uncorrectable error or establish
that changing only this one edge's weight would fix the original shot. The
correlated second pass updates multiple edges and recomputes a complete
correction.

**Provenance and replay.**

The original local data bundle is
`out/union_find_correlated_comparison_d7_p003_100k_seed42`. It retains the
circuit, DEM, packed detector samples, actual observables, decoder predictions,
and a manifest with versions and hashes. Shot 10 is the same row across these
arrays. The bundle is local experiment data and is not a tracked repository
fixture.

The original sampling call was:

```python
detectors, actual_observables = circuit.compile_detector_sampler(seed=42).sample(
    shots=100000,
    separate_observables=True,
    bit_packed=False,
)
```

Use the saved arrays for exact shot replay. Regeneration requires the preserved
circuit, Stim version, SIMD implementation, and sampling-call layout; sampling
eleven shots with seed 42 is not equivalent to selecting row 10 of this batch.

The physical faults were recovered later using read-only observation of
Stim 1.16.0's SSE2 Pauli-frame simulation, preserving its original RNG calls.
Validation established that:

- The reconstructed batch matched all 100,000 saved detector and observable
  rows byte for byte.
- Replaying only the 387 recovered fault events reproduced shot 10 exactly.
- XORing the individually propagated fault responses reproduced the same
  detector syndrome and logical observables.
- A forced-fault circuit, checked with the ordinary Stim sampler using seeds
  1 and 123456, reproduced shot 10 for all eight validation samples.

Detailed diagnostic outputs remain outside the repository under
`$TMPDIR/yoked-analysis-archive-nwujbk8b/out/uf_shot_10_walkthrough_d7_p003_seed42`.
The companion archived investigation directory contains the correlation-rule
replay. This note records the case without requiring those temporary files for
its interpretation.
