# UF failure case collection: 1D yoked surface codes

This collection contains **53 documented failures**: the three original notes
and two expansions of 25 cases each. In every original shot, the repository UF
decoder fails and correlated MWPM succeeds. Each additional note records actual
sampled physical faults, the full-shot logical outcome, a traced graph decision,
an inline diagram, and controlled interventions.

The **second expansion adds 25 previously undocumented shot indices** from the
same preserved seed-42 experiment. Its notes also include patch-by-patch flip
tables, recorded growth transitions, both selected physical connections, and
terminal locations. These help distinguish a satisfied detector syndrome from
a correct logical result, and a grown forest edge from a selected correction.

These are deliberately selected examples of mechanisms and their limits. They
do not establish distinct algorithmic bugs, population frequencies, or the
accuracy of a proposed decoder change.

**Start here for the second expansion.**

- [Shot 93829](shot_93829.md): removing one fault repairs UF; removing a second
  fault as well brings back the original mistake.
- [Shot 991](shot_991.md): two actual faults cancel at one detector; their
  removal repairs only one of the two failed logical sectors.
- [Shot 156](shot_156.md): four terminals in the Z-yoke tree still provide no
  independent logical correction alternatives.
- [Shot 68871](shot_68871.md): an even yoke tree misses a cheaper correct
  forest correction that needs two boundary branches.
- [Shot 80183](shot_80183.md): a 1207-vertex tree contains a correct root choice,
  but its cheapest correct class is more expensive at the original weights.
- [Shot 128](shot_128.md): UF still fails after using all correlation-adjusted
  weights from an MWPM first pass, while correlated MWPM succeeds.

**A useful reading order from the first expansion.**

- [Shot 146](shot_146.md): physical faults cancel at two detectors; correlated
  MWPM uses their connecting edge even though neither detector is fired.
- [Shot 4365](shot_4365.md): an even cluster touches four terminals. Changing
  the root cannot make the existing peeling rule use the two needed boundary
  branches, although a cheaper correct correction exists in the forest.
- [Shot 6269](shot_6269.md): the correct forest correction is more expensive
  than the wrong one under the original weights.
- [Shot 15986](shot_15986.md): correct and incorrect forest logical classes
  have equal minimum weight at the reported precision.
- [Shot 72480](shot_72480.md): removing a readout fault and removing a data fault
  repair different logical sectors; removing both repairs the full UF result.
- [Shot 93574](shot_93574.md): a temporal connection freezes during the first
  yoke batch, and the final yoke cluster grows to 1652 vertices.

**The original three notes.**

| Shot | Verified mechanism |
|---|---|
| [0](../uf_correlated_mwpm_shot0.md) | A readout fault, tied growth at the yoke, and why an odd intermediate cluster is allowed |
| [10](../uf_correlated_mwpm_shot10.md) | An actual YY fault and cross-sector evidence that lowers a matching-edge weight |
| [3148](../uf_correlated_mwpm_shot3148.md) | Changing only the peeling root fixes a two-terminal boundary choice |

**The first 25 additional cases.**

Every row below has a successful correlated-MWPM prediction on the original
shot. "UF wrong" lists logical observables, not detector bits. The yoke column
is the measured `(X, Z)` syndrome.

| Shot | Case | Yokes (X, Z) | UF wrong |
|---|---|---|---|
| [6](shot_6.md) | Two even clusters block a physical connection | (1, 1) | L3, L7 |
| [13](shot_13.md) | An even yoke cluster stops short of a boundary defect | (0, 1) | L9, L11 |
| [21](shot_21.md) | An initially even star later freezes a data-error connection | (1, 0) | L8, L10 |
| [23](shot_23.md) | UF and ordinary MWPM choose different wrong logical pairs | (1, 1) | L0, L8 |
| [75](shot_75.md) | Cancelled detector events accompany errors in both sectors | (1, 1) | L6, L7, L10, L11 |
| [125](shot_125.md) | An ancilla X fault with an unfired X yoke | (0, 1) | L6, L8 |
| [129](shot_129.md) | A boundary fault links an X-logical flip to a Z-sector mistake | (1, 0) | L9, L11 |
| [145](shot_145.md) | A completed yoke edge is discarded in a tied batch | (0, 0) | L4, L6 |
| [146](shot_146.md) | Correlated MWPM traverses detectors whose physical flips cancel | (0, 0) | L1, L11 |
| [295](shot_295.md) | Removing a readout fault fixes only the X-sector mistake | (0, 1) | L1, L2, L3, L8 |
| [964](shot_964.md) | First-round faults produce only a partial logical repair | (1, 0) | L6, L7, L10, L11 |
| [1026](shot_1026.md) | Late faults repair one logical sector while another remains wrong | (0, 0) | L3, L6, L8, L9 |
| [3807](shot_3807.md) | Choosing the other boundary root fixes an X-logical pair | (1, 0) | L0, L8 |
| [4365](shot_4365.md) | An even cluster needs two boundary branches | (1, 1) | L0, L6 |
| [6269](shot_6269.md) | A correct forest correction costs more than the wrong one | (0, 0) | L5, L7 |
| [12273](shot_12273.md) | Even parity prevents root changes from using the needed boundaries | (1, 1) | L0, L6 |
| [15986](shot_15986.md) | Equal-cost forest corrections predict different X-logical pairs | (0, 1) | L6, L10 |
| [26285](shot_26285.md) | Cluster separation in the lowest-defect selected failure | (1, 1) | L6, L8 |
| [44346](shot_44346.md) | A Z-sector boundary choice survives removal of a readout fault | (0, 1) | L1, L7 |
| [58115](shot_58115.md) | A tied Z-sector correction remains ambiguous at fixed weights | (1, 0) | L3, L9 |
| [67251](shot_67251.md) | Five boundary terminals and a preference for the wrong logical class | (0, 1) | L1, L5 |
| [72480](shot_72480.md) | Readout and data faults affect different logical sectors | (0, 0) | L3, L6, L7, L10 |
| [73966](shot_73966.md) | A 65-vertex first yoke star does not resolve the error | (1, 0) | L0, L10 |
| [76890](shot_76890.md) | An ancilla Y fault is trapped by the first yoke star | (0, 1) | L3, L7 |
| [93574](shot_93574.md) | An ancilla fault is frozen inside a 1652-vertex yoke cluster | (1, 1) | L1, L11 |

**The second 25 additional cases.**

All these shots are new to the collection. Correlated MWPM passes every one;
the ordinary joint-MWPM outcome is listed separately. The previous notes retain
their existing contents and links.

| Shot | Case | Yokes (X, Z) | UF wrong | Ordinary MWPM |
|---|---|---|---|---|
| [16](shot_16.md) | A retained readout edge cannot repair the final Z-logical partition | (1, 1) | L7, L11 | Pass |
| [45](shot_45.md) | A cancelled detector lies on a retained YZ-fault connection | (1, 1) | L6, L10 | Fail |
| [64](shot_64.md) | A cancelled data-fault endpoint stays outside a boundary cluster | (1, 0) | L8, L10 | Pass |
| [78](shot_78.md) | An available boundary edge is unused by an even yoke tree | (0, 1) | L3, L5 | Pass |
| [97](shot_97.md) | Removing a late XY fault repairs only the Z sector | (1, 1) | L2, L4, L7, L9 | Fail |
| [128](shot_128.md) | Correlation-adjusted UF repeats the original wrong X pair | (0, 1) | L0, L8 | Fail |
| [144](shot_144.md) | A first-batch yoke merge freezes an ancilla-fault edge | (1, 1) | L2, L4 | Pass |
| [156](shot_156.md) | Four yoke terminals still exclude the required logical pair | (1, 0) | L1, L5 | Fail |
| [188](shot_188.md) | A late Z-fault connection stays outside the final UF partition | (0, 0) | L0, L10 | Pass |
| [239](shot_239.md) | Retained physical edges accompany different UF and ordinary-MWPM failures | (0, 0) | L7, L9 | Fail |
| [609](shot_609.md) | Two early faults repair the X sector while the Z sector remains wrong | (0, 1) | L6, L7, L10, L11 | Fail |
| [991](shot_991.md) | Two faults cancel a detector and repair only one logical sector | (1, 0) | L1, L4, L6, L11 | Pass |
| [1494](shot_1494.md) | Late fault deletions move the Z mistake without repairing the shot | (0, 0) | L0, L1, L2, L9 | Pass |
| [25073](shot_25073.md) | A cheaper Z-yoke boundary choice survives an internally frozen XX edge | (1, 1) | L1, L9 | Pass |
| [26074](shot_26074.md) | An odd X-yoke tree has a cheaper correct boundary-root choice | (0, 0) | L6, L8 | Pass |
| [26898](shot_26898.md) | Four terminal-root combinations reduce to two logical answers | (1, 1) | L1, L3 | Pass |
| [44875](shot_44875.md) | An unfired yoke and a missed logical edge across clusters | (0, 0) | L5, L11 | Pass |
| [61448](shot_61448.md) | A tied cycle edge is discarded while a correct boundary choice remains | (0, 0) | L1, L3 | Pass |
| [68871](shot_68871.md) | An even Z-yoke tree needs two boundary branches | (1, 1) | L3, L9 | Pass |
| [71143](shot_71143.md) | A cancelled readout event and a costlier correct forest class | (1, 0) | L6, L10 | Pass |
| [80183](shot_80183.md) | A 1207-vertex tree retains a costlier correct root choice | (1, 1) | L6, L10 | Pass |
| [86347](shot_86347.md) | Two retained physical connections accompany an even-tree peeling restriction | (0, 0) | L6, L8 | Pass |
| [88907](shot_88907.md) | A retained ZY edge and a cheaper two-boundary correction | (1, 0) | L2, L10 | Pass |
| [90957](shot_90957.md) | An unfired readout endpoint inside a 950-vertex Z-yoke tree | (0, 1) | L9, L11 | Pass |
| [93829](shot_93829.md) | Deleting a second fault restores the original UF mistake | (0, 1) | L5, L9 | Fail |

**How the cases were selected.**

The source is the already preserved seed-42, 100,000-shot experiment. No new
random noise sample was substituted. From the UF-fails/correlated-MWPM-passes
shots, the first expansion was selected as follows:

- Twelve examples cover all four yoke-syndrome combinations and failures in
  the X-logical sector, the Z-logical sector, or both. The ordinary-MWPM
  success/failure condition alternates across these strata; the first eligible
  shot index was selected after excluding the existing notes.
- Four examples cover the smallest and largest defect counts, largest first
  yoke cluster, and largest final yoke cluster in that failure population.
  For equal extrema, the last index in stable ascending order was selected for
  maxima; the first was selected for minima. These are shots 26285, 72480,
  73966, and 93574.
- Nine cases were chosen from previously identified exceptions where the fixed
  UF clusters still admit the correct logical answer. They include easier
  peeling choices, even-cluster boundary restrictions, and cost degeneracies.

Within the first expansion, the verified structural results are:

| Finding | Cases |
|---|---|
| Correct logical answer excluded by the final cluster partition | 16 cases: all first-expansion cases outside the nine exceptions below |
| Correct forest class has a lower minimum cost than the wrong class | 3807, 4365, 44346 |
| Correct forest class has a higher minimum cost than the wrong class | 6269, 12273, 67251 |
| Correct and wrong forest classes are tied within `1e-9` | 15986, 58115, 76890 |

Those counts describe this curated collection, not the prevalence of these
mechanisms in the full experiment. Two faults per shot were chosen from actual
events whose graph components lie on a logical correction difference, with
preference for differing noise channels and explicit correlation witnesses.
They were not required to fix the shot when removed: unsuccessful and partial
interventions remain in the notes.

**Selection and structural results for the second expansion.**

The existing 28 documented indices were excluded before any selection. The
second expansion contains:

- All 11 remaining previously identified UF-fails/correlated-MWPM-passes cases
  whose fixed cluster partition admits the correct logical answer: 25073,
  26074, 26898, 61448, 68871, 71143, 80183, 86347, 88907, 90957, and 93829.
- Twelve strata formed by all four measured yoke-bit combinations and failures
  in X, Z, or both observable sectors. With yokes numbered X then Z, visit
  (0,0), (0,1), (1,0), (1,1), and within each visit X, Z, both. Require ordinary
  MWPM to pass in the first stratum, fail in the second, and alternate
  thereafter; take the first unused eligible shot index in each stratum.
- Shot 144, which has the largest remaining UF-minus-ordinary-MWPM correction
  weight at the original graph weights after those selections: 183.627637.
- Shot 44875, which then has the largest remaining number of correlated-MWPM
  correction edges crossing the final UF clusters: 51. Equal metric values
  are resolved by taking the smallest unused shot index.

These weight and edge-count extrema characterize saved corrections. They are
not latency measurements or logical-error-rate ratios.

| Finding | Second-expansion cases |
|---|---|
| Correct logical answer excluded by the final partition | 14: 16, 45, 64, 78, 97, 128, 144, 156, 188, 239, 609, 991, 1494, 44875 |
| Correct forest class has a lower minimum cost than the wrong class | 8: 25073, 26074, 26898, 61448, 68871, 86347, 88907, 90957 |
| Correct forest class has a higher minimum cost than the wrong class | 3: 71143, 80183, 93829 |
| Correct and wrong forest classes tied within 1e-9 | None |

Across the 50 expansion notes, 30 partitions exclude the correct answer and
20 forests retain it. Those 20 include 11 cheaper correct classes, six more
expensive correct classes, and three tied cases. These are counts within the
curated notes, not estimates of mechanism frequencies.

**Common circuit, graph, and notation.**

| Item | Value |
|---|---|
| Circuit | 1D yoked surface-code memory; CZ gates; ideal preparation, final boundaries, and yoke measurements |
| Physical noise | Repository SI1000, `p=0.003` |
| Distance and duration | `d=7`; 28 noisy measurement rounds |
| Patches | Six physical patches, numbered 0–5 |
| Graph | 8354 detectors, 40836 weighted edges, 12 observables |
| Yokes | X: D8352; Z: D8353 |
| Original sampler | Stim 1.16.0, SSE2; seed 42; one call of 100000 shots |
| Matching | PyMatching 2.4.0; ordinary joint and correlation-aware modes |
| UF | Repository `UnionFindDecoder`, with its original growth and peeling |

`Dk` denotes a detector and `ek` an edge in the preserved graph's deterministic
ordering. `Tk`, with `k >= 8354`, denotes a virtual boundary terminal created by
the graph importer. Each boundary edge has its own terminal leaf. A terminal
has no measured detector constraint; a yoke does. The detector graph can have
long yoke connections without those being physical gates between patches.

The observable ordering is `(X0, Z0, X1, Z1, ..., X5, Z5)`. `Lk` denotes bit `k`
of this ordering. An integer mask sets bit `k` when that observable flips.
Prediction XOR actual outcome gives the wrong-observable mask. A wrong
X-logical prediction on patch `i` corresponds to a residual `Z_L,i`; a wrong
Z-logical prediction corresponds to `X_L,i`; both correspond to `Y_L,i`, up to
phase. This describes the residual logical commutation pattern, not a literal
sequence of correction gates.

All identifiers except circuit rounds are zero-based. Ordinary detector time
`t=r-1` corresponds to circuit measurement round `r`; `t=28` is the final ideal
boundary. Growth time is an algorithmic variable, not circuit time or latency.
An individual physical fault can toggle a detector that is zero in the full
syndrome because another fault cancels it.

**What was checked.**

For every additional case, all three baseline graph corrections were replayed
and checked against the saved prediction masks. Every correction satisfies
`H c = s`; its observable mask was recomputed as `L c`. This separates syndrome
validity from logical success.

The physical-fault recovery observes the exact Stim SSE2 simulation without
adding RNG calls, splitting gates, or altering the simulated circuit. Each
25-case expansion replayed one complete 100,000-shot batch to recover its
selected rows. In both replays, the packed detector and observable arrays match
the saved arrays byte for byte. For each of the 50 rows, the recovered events
were also checked by:

- Propagating all events together and XORing their independently propagated
  responses; both reproduce the saved syndrome and observables.
- Running a forced-fault circuit with the ordinary Stim sampler for eight
  validation samples using seeds 1 and 123456.
- Decoding the shot with each of two selected faults removed, with both
  removed, with each alone, and with the pair alone. Actual observables are
  adjusted along with the syndrome when a removed fault flips a logical bit.

All 100 selected single-fault patterns and all 50 isolated pairs are decoded correctly
by all three decoders. Some removals from the surrounding noisy shots leave
errors or change which observables are wrong; those outcomes are reported.

**How the structural certificates work.**

For a fixed allowed edge set, differences between valid corrections lie in
`ker(H)`. Their possible logical changes form `L(ker(H))`. The analysis spans
that space using fundamental cycles, with unconstrained boundary terminals
collapsed to a reference boundary. It then checks whether the required mask
`actual XOR UF` belongs to that span.

This is checked for the UF forest and for the larger set containing every
original graph edge whose endpoints belong to the same final UF cluster. A
negative certificate for the larger set proves that keeping that partition
cannot give a correct logical answer, regardless of how peeling is changed.
It is not an inference from one unsuccessful matching run.

For the forest, an independent tree dynamic program tracks the parent-edge
parity and logical mask while allowing every boundary terminal to absorb
parity. It finds the minimum original-float cost separately for every available
logical prediction. All recovered solutions satisfy `H c = s` and the claimed
`L c`; their costs agree with direct edge-weight sums. The global forest
minimum also agrees with restricted PyMatching to numerical tolerance. This
avoids treating a PyMatching quantization decision as proof that one float
logical class is strictly cheaper than another.

Trying another terminal root and optimizing over all valid forest corrections
are different interventions. In particular, single-root peeling leaves all
boundary edges of an even tree unused, whereas a general valid correction may
use two boundary endpoints. The correct class and successful diagnostic roots
are identified with the saved truth available to the analyst; they are not a
deployable rule that knows the right answer in advance.

**How correlation weights are interpreted.**

The reconstructed PyMatching rule uses a selected first-pass edge as evidence
for another component of a shared DEM mechanism:

```text
q_implied = min(1/2, q_shared / q_source)
w_target_new = min(w_target_old, log((1-q_implied) / q_implied))
```

The shared probability aggregates compatible DEM mechanisms; it is not the
probability of the single logged physical event or an exact joint posterior
including every combination of independent faults. The first-pass selection
is an inference. A zero updated weight results from clipping at `1/2`, not
certainty of a physical error. The reconstructed second pass agrees with the
saved correlated-MWPM logical prediction on all 50 expansion cases; exact edge-set
equality at ties is not required.

Each note also reports a diagnostic UF rerun using all weights inferred from
the MWPM first pass. That operation requires an MWPM solve and is not an
independent UF performance result. Cases where it fails remain visible, such
as [6269](shot_6269.md) and [76890](shot_76890.md). A local weight change is
documented as evidence, not as proof that changing only that one edge repairs
the whole shot.

**Replaying the saved samples.**

The local bundle is
`out/union_find_correlated_comparison_d7_p003_100k_seed42`. It is preserved
experiment data, not a tracked test fixture. Its original sampling call was:

```python
detectors, actual_observables = circuit.compile_detector_sampler(seed=42).sample(
    shots=100000,
    separate_observables=True,
    bit_packed=False,
)
```

Sampling only `shot_index + 1` shots is not equivalent. Exact regeneration
requires the preserved circuit, Stim version, SIMD implementation, and
sampling-call layout. Prefer loading the saved arrays when replaying a case.

From the repository root, use `.venv/bin/python` with `PYTHONPATH=src` and
temporary cache directories under `$TMPDIR`. The following reads the existing
bundle and prints the predictions without saving additional artifacts:

```python
from pathlib import Path
import json
import numpy as np
import pymatching
import stim
from yoked.decoders import DecodingGraph, UnionFindDecoder

shot = 146  # Replace with any documented zero-based index.
bundle = Path("out/union_find_correlated_comparison_d7_p003_100k_seed42")
graph = DecodingGraph(**json.loads((bundle / "graph.json").read_text()))
dem = stim.DetectorErrorModel.from_file(bundle / "model.dem")
packed = np.load(bundle / "detectors_packed.npy", mmap_mode="r")
packed_actual = np.load(bundle / "actual_observables_packed.npy", mmap_mode="r")
syndrome = np.unpackbits(packed[shot], count=8354, bitorder="little").astype(bool)
actual = np.unpackbits(packed_actual[shot], count=12, bitorder="little").astype(bool)

uf = UnionFindDecoder(graph)
ordinary = pymatching.Matching.from_detector_error_model(dem)
correlated = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
predictions = {
    "repository_uf": uf.decode(syndrome),
    "joint_mwpm_uncorrelated": ordinary.decode(syndrome),
    "joint_mwpm_correlated": correlated.decode(syndrome, enable_correlations=True),
}
with np.load(bundle / "predictions.npz") as saved:
    for name, prediction in predictions.items():
        np.testing.assert_array_equal(prediction, saved[name][shot])
        mask = sum(int(b) << k for k, b in enumerate(prediction))
        wrong = np.flatnonzero(prediction.astype(bool) ^ actual).tolist()
        print(name, "mask", mask, "wrong observables", wrong)
```

The preserved input fingerprints are:

| Input | SHA-256 |
|---|---|
| `circuit.stim`, file bytes | `8cfa9bb9eaf6db86dfc9ffcfefa4582eb29932d11dc1fd911239ef1425841ff9` |
| `model.dem`, file bytes | `9b06141668bef9b334df78e4700853f60505a71f3c4185746d0261e7e3790e0a` |
| `graph.json`, file bytes | `b61050a3d10719d4e161cbec6d08a0dd69fb1aeaa85bc196ea4bc5117f71ff55` |
| Packed detector array payload, excluding `.npy` header | `365e797c8b5685f8fae27660e7f85326ba569dd5aacc425346d5724b184076f8` |
| Packed observable array payload, excluding `.npy` header | `f297aca56ceefe33e3e5d8b1ac0dca5b478d23dc2179fbe9f5b60a1d8c3c5cf0` |
| Repository `_union_find.py`, file bytes | `8482635867c74233465d68152dd153c56adf3b7323d5573926400bad7e436b59` |
| Repository `_graph.py`, file bytes | `cd96b94202be5bf38d29644d81f060a93fe8358a0638f9953e00b71defd4d6b8` |

Growth, tied-batch processing, and peeling are in
[`_union_find.py`](../../src/yoked/decoders/_union_find.py); graph export and
terminal construction are in [`_graph.py`](../../src/yoked/decoders/_graph.py).

The first expansion's scripts and detailed records remain under
`$TMPDIR/uf-case-collection-9pbSDiuL`; the second expansion's are under
`$TMPDIR/uf-case-collection-next-6a0xaw4f`. Each has a `selection.json` that
records its selected indices and criteria, a `batch_validation.json` with the
complete batch comparison, and a `shot_<index>` directory for each case. The
per-shot records contain the physical replay, growth trace, interventions,
logical-space certificates, and forest-cost solutions. The second expansion
also retains the scripts used to generate and audit its Markdown notes.

Temporary paths can disappear. The Markdown notes therefore contain the
selected physical locations, inline diagrams, numerical conclusions, and
explicit correction paths needed to review each case. Only documentation is
added to the repository; these investigations do not change the decoder.
