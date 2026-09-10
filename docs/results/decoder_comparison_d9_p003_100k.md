# 1D yoked surface code: four-decoder comparison at d=9, p=0.003

On 100,000 shared shots, correlated UF achieved a normalized logical error
rate (LER) of **7.182e-4 per patch per round**, compared with **3.337e-4** for
correlated MWPM. Adding correlations reduced the repository UF decoder's LER
by **70.4%**. Increasing distance from 7 to 9 improved all four decoders.

**Configuration.**

| Parameter | Value |
|---|---|
| Circuit | 1D yoked magic-memory circuit, CZ style |
| Inner patch distance | 9 |
| Physical noise | SI1000, `p=0.003` |
| Rounds | 36 (`4d`) |
| Patches / yokes | 6 / 2 |
| Shots per decoder | 100,000 |
| Stim sampling seed | 42 |
| Decoding graph | 17,762 detectors, 90,732 edges, 12 observables |
| Run date | 2026-09-10 UTC (2026-09-09 Pacific) |

All four decoders received the same syndromes and were scored against the
same sampled observable flips. Joint MWPM used PyMatching without
correlations; correlated MWPM enabled correlations during both construction
and decoding. UF used the repository's weighted growth and peeling decoder.
Correlated UF used two repository UF passes, with DEM-derived reweighting
based on the first correction before decoding the original syndrome again.
See [decoder usage](../union_find_usage.md) for the APIs.

This experiment measured accuracy. It did not measure decoder latency.

**d=9 results.** A block failure means at least one predicted observable
differs from the sampled observable flips.

| Decoder | Failed shots / 100,000 | Block failure rate | LER per patch per round | 95% interval for normalized LER |
|---|---:|---:|---:|---:|
| Joint MWPM | 25,660 | 25.660% | 1.398e-3 | [1.381e-3, 1.416e-3] |
| Repository UF | 39,847 | 39.847% | 2.430e-3 | [2.405e-3, 2.455e-3] |
| Correlated MWPM | 6,925 | 6.925% | 3.337e-4 | [3.258e-4, 3.417e-4] |
| Correlated UF | 14,247 | 14.247% | 7.182e-4 | [7.064e-4, 7.303e-4] |

Normalization follows the repository's [plotting convention](../../step3_plot):

```python
normalized_ler = sinter.shot_error_rate_to_piece_error_rate(
    block_failures / 100_000,
    pieces=6 * 36,       # 216 patch-rounds per shot
    values=2 * (6 - 2),  # 8 logical values
)
```

This is Sinter's piece conversion, rather than simple division of the block
failure rate by 216. The individual LER intervals are exact 95% binomial
intervals for the block failure rate, transformed using the same conversion.

**Comparison with d=7.** The earlier experiment used the same four decoder
implementations, `p=0.003`, six patches, two yokes, 100,000 shots, and seed 42.
It used 28 rounds (`4d`), so its normalization used `pieces=168, values=8`.
Samples were shared among decoders within each distance; the two distances
used different circuits and sample sets.

| Decoder | d=7 failed shots / 100,000 | d=7 normalized LER | d=7 LER / d=9 LER |
|---|---:|---:|---:|
| Joint MWPM | 33,565 | 2.496e-3 | 1.785x |
| Repository UF | 46,684 | 3.895e-3 | 1.603x |
| Correlated MWPM | 13,785 | 8.908e-4 | 2.670x |
| Correlated UF | 23,230 | 1.599e-3 | 2.226x |

Correlated UF's normalized LER was **2.152x** correlated MWPM's at d=9
(paired-bootstrap 95% interval **[2.105x, 2.202x]**), compared with **1.795x**
at d=7. Both improved with distance, with correlated MWPM improving more.
At d=9, correlated UF improved over plain UF by **3.383x** and over ordinary
joint MWPM by **1.947x**. Ratios use the unrounded normalized rates.

For the two correlated decoders, both succeeded on 83,553 shots; MWPM
succeeded and UF failed on 9,522; UF succeeded and MWPM failed on 2,200;
both failed on 4,725. The ratio interval used 10,000 paired bootstrap
replicates of these joint outcomes, with bootstrap RNG seed 43.

**Validation.** Every final correction from UF and correlated UF passed
`Hc = s` and `Lc = prediction`, for all 100,000 shots per decoder. All four
decoders' predictions satisfied the yoke parity checks. The correction-based
evaluation agreed with the public UF APIs on the preflight checks, and the
full run reproduced the 50-shot preflight predictions. Replaying the complete
Stim sampling call reproduced the saved detector and observable arrays byte
for byte.

**Reproducing the d=9 inputs.** Run from the repository root with
`PYTHONPATH=src`. The recorded environment was Python 3.14.5, Stim 1.16.0
(`stim._stim_sse2`), NumPy 2.5.1, PyMatching 2.4.0, Sinter 1.16.0, and
SciPy 1.18.0. The exact replay used a single 100,000-shot sampling call:

```python
import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

circuit = yoked_magic_memory_circuit(
    patch_diameter=9,
    rounds=36,
    noise=gen.NoiseModel.si1000(0.003),
    style="cz",
    yokes=2,
    num_patches=6,
)
dem = circuit.detector_error_model(
    decompose_errors=True,
    approximate_disjoint_errors=True,
)
detectors_packed, actual_observables_packed = (
    circuit.compile_detector_sampler(seed=42).sample(
        shots=100_000,
        separate_observables=True,
        bit_packed=True,
    )
)
```

Preserve the versions, sampling call size, and Stim architecture when
replaying the seed. The SHA-256 of the raw packed detector bytes followed by
the raw packed observable bytes is:

```text
d55da8f4c9b8287fa0af64f8382de755a243a774030499e108c348b665e102f5
```

The run used base commit `10dc867b632edcf011f802f0f8a6e05ca769c659` plus the
working-tree correlated UF implementation. The manifest records SHA-256
hashes of the three decoder source modules, and their source snapshots are
saved with the run; the base commit alone does not identify the measured
implementation.

The circuit, DEM, graph, packed samples, per-decoder predictions, source
snapshots, and experiment driver remain under `$TMPDIR`. Full precision
results and provenance are available at these local paths:

- [d=9 results](/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha/results.json)
- [d=9 manifest](/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha/manifest.json)
- [d=9 experiment driver](/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha/run.py)
- [d=7 results](/data2/s2chitni/.tmp/ysc-four-decoders-d7-p003-100k-46deb37u/results.json)
