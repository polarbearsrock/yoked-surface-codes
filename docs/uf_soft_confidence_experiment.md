# UF confidence without L1 matching

The completed [d=7/d=9 results](results/uf_soft_confidence_d7_d9_p003_100k.md)
include all 100,000 evaluation shots per distance, paired intervals and software
cost measurements.

This experiment keeps the original per-patch UF reference fixed and replaces the
matching solves used to estimate its residual-error probability. L2 remains the
weighted MWPM backend used in the preceding comparison. The question is whether
UF-derived confidence can retain enough accuracy to justify a simpler L1 hardware
implementation. Hardware latency, area and power are not measured here.

The experiment uses SI1000 `p=0.003`, `d=7,9`, six patches, two ideal yokes, and
`rounds=4d`. Each distance reuses the existing independent 50,000 calibration shots
(seed 142) and 100,000 evaluation shots (seed 42). Selective refinement remains
deferred: every patch receives the chosen confidence estimator.

## Hypotheses and fixed candidates

1. **UF class-cost gap:** UF correction costs in the two logical classes may be
   accurate enough to rank patches by error probability, even though they are not
   minimum costs. The gap relative to the original UF bit `r` is
   `cost_UF(1-r) - cost_UF(r)`. A negative value favors reversing that bit.
2. **Correlated UF class-cost gap:** use the original UF correction to apply the
   existing DEM correlation discounts, freeze those weights, and compare the two
   classes with UF. This tests whether correlation information matters more than
   exact class-cost minimization. Both the conditioning correction and the forced
   corrections come from UF; the previous matching endpoint conditioned on MWPM.
3. **Bounded cluster gap, with and without a correlated UF pass:** stop the
   existing residual-growth shortest-path search at `ln(100)` nats (20 dB).
   For the correlated variant, run one free UF decode under the UF-conditioned
   weights; sign its gap positive if that prediction agrees with the original UF
   reference and negative otherwise. This tests a smaller addition than the forced
   class decodes, at the cost of retaining a shortest-path search.
4. **Sixteen-bin correlated UF gap:** map the signed class-cost gap to
   `clip(floor(gap / 2), -8, 7)` before calibration. The resulting confidence can
   be implemented by a 16-entry ROM per sector. The binning and saturation are
   fixed before evaluation; this does not quantize UF's edge arithmetic.

The unbounded original UF cluster gap and the previous correlated matching gap
are reused as hierarchical controls. The four saved joint-decoder baselines are
also scored on the identical evaluation rows.

Two forced whole-patch UF decodes suffice for each weight model: one sets both
check bits to zero, the other sets both to one. The graph has disconnected X and Z
components. Summing selected-edge costs separately in each component supplies
`cost_X(0), cost_X(1), cost_Z(0), cost_Z(1)`. Tests compare this optimization against
all four independently forced patterns and verify each correction's syndrome and
logical class. A UF class-cost gap is a heuristic score, not a likelihood ratio.

## Calibration and L2

For every candidate and distance, fit one decreasing isotonic map per sector,
pooling all six patches on calibration shots only. The target is always whether
the **original UF bit** was wrong. Allow probabilities above one half and clip to
`[1e-6, 1-1e-6]`, as in the preceding experiment. Maps are fitted and saved before
evaluation labels are used for metrics. The quantized candidate exports all 16
probabilities and log-odds for each sector in `confidence_rom.json`.

L2 receives the six calibrated probabilities and the yoke parity adjusted by the
original UF frame. Its correction is XORed with that frame. Every final result is
checked against both measured yoke parities. No L1 method receives the yoke bits.

Report failed shots, block failure rate, and the existing normalized LER:
`sinter.shot_error_rate_to_piece_error_rate(block_rate, pieces=6*rounds, values=8)`.
Whole-shot paired bootstrap intervals use 10,000 replicates and seed 43, conditional
on the fitted calibrators. Compressing shots to observed joint decoder-failure
patterns preserves the pairing exactly. Comparisons are exploratory and the
previously examined evaluation sample is not a new confirmation sample.

## Work and hardware interpretation

| L1 candidate | UF passes per patch | Other work |
|---|---:|---|
| Bounded cluster gap | 1 | Two bounded shortest-path searches |
| Correlated bounded gap | Up to 2 | Correlation reweighting and two bounded searches on the chosen second-pass graph |
| UF class-cost gap | 3 | Sum correction weights by sector |
| Correlated UF class-cost gap | 3 | Correlation reweighting; two forced passes under the resulting weights |
| Sixteen-bin correlated UF gap | 3 | Same work; confidence quantizer and ROM |

These are standalone method counts. The experiment collector shares work while
evaluating all candidates, and therefore performs more total work than any one
candidate. Its timers separate reference UF, forced decodes, reweighting/graph
construction, and searches. Wall time is a Python implementation measurement;
it cannot establish an FPGA advantage over optimized C++ MWPM.

The hardware motivation is concrete but limited: reuse weighted UF growth,
peeling, accumulators, and local correlation-rule tables instead of adding an L1
matching solver. Multiple passes add latency or require replicated engines.
Forced-class check vertices are high-degree boundary hubs, so their communication
and reduction costs need a hardware design. A compact output ROM alone does not
prove that floating-point growth and reweighting can be quantized without losing
accuracy.

[Helios](https://arxiv.org/abs/2406.08491) supplies evidence that UF decoding itself
has practical FPGA implementations, not a hardware benchmark for these confidence
estimators. [Bounded cluster-gap and extra-cluster-growth research](https://arxiv.org/abs/2602.03336)
motivates early stopping; extra-cluster growth is a separate future candidate and
is not implemented by clipping a shortest-path search. Another promising next
experiment is a small fixed number of min-sum BP updates feeding weighted UF,
following [belief-find](https://arxiv.org/abs/2203.04948); its Tanner-graph messages
and memory traffic must be measured before assuming a cost advantage.

## Implementation and reproduction

- `src/yoked/hierarchical/_uf_soft.py`: confidence algorithms, without matching solves.
- `_uf_soft_collect.py`: verified samples, UF-reference checks, hashed checkpoints.
- `_uf_soft_analysis.py`: independent calibration, L2 replay, paired statistics.
- `tools/uf_soft_experiment`: command-line entry point.

The feature artifact is separate from the existing `L1Record`. The collector never
runs its expensive matching-confidence path. It loads and verifies the record and
sample identities, checks all newly computed UF bits against the saved reference,
and verifies the bounded plain gap against the censored saved full gap. Source
hashes, model/sample identities, record hashes, row ids, package versions, and
feature hashes are retained. Changed requests or corrupt checkpoints are refused.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$TMPDIR/uf-soft-mpl" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python tools/uf_soft_experiment collect \
    --record "$CALIBRATION_RECORD" --out "$TMPDIR/uf-soft/calibration" --workers 32

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$TMPDIR/uf-soft-mpl" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python tools/uf_soft_experiment collect \
    --record "$EVALUATION_RECORD" --out "$TMPDIR/uf-soft/evaluation" --workers 32

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$TMPDIR/uf-soft-mpl" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python tools/uf_soft_experiment analyze \
    --calibration-record "$CALIBRATION_RECORD" --evaluation-record "$EVALUATION_RECORD" \
    --calibration-features "$TMPDIR/uf-soft/calibration" \
    --evaluation-features "$TMPDIR/uf-soft/evaluation" --out "$TMPDIR/uf-soft/analysis"
```

Use `--sample-dir` when a verified record was copied without its saved sample
subdirectory. `--limit` is for collection pilots; analysis requires all record rows.
