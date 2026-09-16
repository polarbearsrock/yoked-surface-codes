# Physical-fault family ablation

`analyze.py` is an offline experiment using the original seed-42, 100,000-shot
records at each distance. It does not change either production decoder.

The six primary interventions remove all realized events in one physical family:
readout flips, reset errors, extra data depolarization while waiting for
measurement/reset, other single-qubit data depolarization, single-qubit ancilla
depolarization, or CZ gate depolarization. A two-qubit Pauli product is one event.
The response archive further separates Pauli types and data/ancilla support.

Select 256 shots uniformly without replacement in each of the four original
decoder-outcome strata, independently at d=7 and d=9. The selection is saved
before recovering any faults. The C++ observer replays the entire original
sampling batch, logs the chosen rows without adding random draws, and must
reproduce all 100,000 saved detector and observable records exactly.

For each selected shot, deterministic Stim propagation computes each fault
family's detector and observable response. XOR removal updates both the
syndrome and the true logical labels. Aggregate responses must reproduce the
original shot. Independent ordinary-Stim simulations validate each primary
intervention on one shot from each outcome stratum, with two sampler seeds.

Both correlated decoders use the original detector error model for every
intervention, recomputing their first-pass evidence and second-pass result.
The unmodified shots must reproduce all saved baseline predictions.

The estimand is the reduction in the joint-shot failure-rate gap
`P(UF fails) - P(MWPM fails)`. Stratum means are weighted by the full population
counts, not by the equal sample sizes. The uncertainty calculation uses 20,000
paired stratified bootstrap replicates. These are marginal 95% intervals for
subsampling the existing records, not simultaneous confidence bands.

`audit.py` independently checks the compact archived predictions and weighted
estimates. It also provides conservative finite-population confidence intervals
by inverting hypergeometric distributions, allowing rare outcomes absent from
the sample. This matters when the residual failure rate is close to zero and
ordinary bootstrap intervals can be too narrow. Run it with the repository's
Python environment after archiving the results.

Whole-family deletion effects overlap through fault interactions. They are
neither additive shares of the gap nor per-event harmfulness estimates. The
decoders are not retuned to the modified noise channels, so these results are
counterfactual sensitivities of the current decoders, not optimal LERs under
alternative calibrated noise models.

## Running the analysis

Use the repository virtual environment and `PYTHONPATH=src`. Set
`PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MPLCONFIGDIR` to a location under `$TMPDIR`.

`prepare` requires a new `--work-dir` under `$TMPDIR`, `--record-root`, and a
`--tracer` binary. The observer source is included as `trace_cases.cc`; exact
replay depends on the original Stim version, SIMD width, seed, and full batch
size. The recorded run uses the previously built Stim 1.16 SSE2 observer.

Run `trace`, `recover`, `decode`, then `summarize` separately for `--distance 7`
and `--distance 9`, always with the same `--work-dir`. `decode` accepts
`--workers`, optional `--variants`, and a `--label` for additional subfamily
experiments; pass the same label to `summarize`.

Selections, compact decoded outcomes, validation records, and summaries are
retained alongside this file. The report records the scratch path containing
the larger fault logs and response arrays.
