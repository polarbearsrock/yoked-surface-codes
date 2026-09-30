# BP5 and correlated UF: 100,000 paired shots per distance

This extends the frozen 5,000-shot evidence pilot to every row of its
100,000-shot parent sample at d=7,9,11,13. Settings remain SI1000 p=0.003,
six patches, two ideal yokes, CZ circuits, rounds=4d, and Stim seed 42.
Samples must reproduce the archived packed-payload and model identities.
The 5,000 pilot rows are included; the other 95,000 rows are also reported
separately. This is not a newly seeded independent sample.

## Decoders fixed before the expanded run

- **BP5 + UF:** exactly the pilot's five damped sum-product BP iterations
  on joint DEM fault variables, nonnegative capped-sum probability
  projection, and one unchanged weighted UF growth/peeling pass.
- **BP5 + correlated UF:** start with those same BP5 weights and first UF
  correction. Apply the existing prior-derived DEM correlation rules to
  the selected correction edges, lowering each supported target weight to
  the minimum of its BP5 weight and the rule's implied weight. Run the
  unchanged UF again on the original syndrome. The second correction is
  the complete answer. This is an explicitly defined additional heuristic;
  BP already uses correlations, so this extra pass can double-count evidence.
- **Correlated MWPM:** PyMatching with correlations enabled at graph
  construction and batch decoding. Recompute all 100,000 predictions and
  require exact agreement with the archived predictions on identical shots.
- **Current correlated UF:** the archived predictions on these same
  verified samples, with predetermined rows recomputed during validation.

The first two variants share computation, but their recorded times charge
the full work required by each decoder. BP5 is unchanged: flooding,
check-message damping 0.5, LLR clipping at +/-30, double precision,
`-log(clip(sum of component-support probabilities, 1e-15, 1))` weights.
No pruning, hard BP correction, parameter tuning, per-shot variant
selection, or truth/reference input to the native decoder is permitted.
BP20 and the prior-projection control are not rerun in this extension.

## Validation and reporting

Reuse the independent BP exact-posterior and native/Python checks from the
pilot. Independently check both new native paths against production Python
UF and CorrelatedUnionFindDecoder, including their physical forests and
corrections. Check thread invariance, every correction's complete syndrome,
and every prediction's ideal-yoke parity. Require all 5,000 overlapping
BP5 predictions and non-timing diagnostics to reproduce the pilot exactly.

Report whole-shot failure counts/rates, Wilson marginal 95% intervals,
the historical normalized LER convention, and paired differences against
correlated MWPM and current correlated UF. Report repairs/regressions and
exact McNemar tests. Show the additional correlation pass's paired effect.
Intervals are exploratory and not adjusted for multiple comparisons.
Finite distances at one physical error rate do not establish a threshold.

All build products, raw samples, and resumable checkpoints go under
`$TMPDIR`. Checkpoints bind source, model, sample, parameters, and library
hashes. Compact results and the final report are retained beside this file.
Production decoder code and the original pilot remain unchanged.

## Reproduction

From the repository root, with the original pilot's complete samples in
`$sample_root/d{7,9,11,13}/sample` (missing samples can be regenerated):

```bash
work_dir=$(mktemp -d "$TMPDIR/uf-evidence-100k-XXXXXX")
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/uf-evidence-100k-mpl"
.venv/bin/python docs/results/uf_evidence_predecoder_100k_d7_d13/run.py \
    --work-dir "$work_dir" --sample-root "$sample_root" \
    --output "$work_dir/results" --threads 64 --matching-workers 8
.venv/bin/python docs/results/uf_evidence_predecoder_100k_d7_d13/report.py \
    "$work_dir/results"
```
