# Four paired yoked-code decoders, d=7

Fixed workload: 100,000 fresh paired shots, SI1000 p=0.003, six distance-7
patches, two ideal yokes, 28 CZ extraction rounds, ideal time boundaries.
Use the installed Dante Stim fork, verifying `env/stim-fork.json` before sampling.
All variants use the existing plain-MWPM L2 and its deterministic tie convention.

1. Correlated MWPM + frozen-weight complementary gap.
2. Correlated MWPM + the existing native MPP cluster gap (matching radii).
3. Two-pass correlated UF + full cluster gap from the final growth state.
4. Exactly five BP iterations, followed by two-pass correlated UF, followed by
   the full cluster gap from the final growth state.

Variant 4 uses the existing sum-product flooding BP (damping 0.5, LLR limit 30)
on DEM fault variables. Its existing projection is
`-log(clip(sum of supporting fault posteriors, 1e-15, 1))` per graph edge.
UF first runs on these weights. The existing DEM-prior conditional correlation
rules then lower weights selected by that first correction, using minimum
discounts, and UF reruns on the original syndrome if any weight changed.
This is an explicit BP-then-correlated-UF composition; the prior single-patch
BP5 experiments ended with plain weighted UF instead. Posterior correlation
rules are not re-estimated, and this composition is not exact Bayesian inference.
No tuning, calibration, early BP termination, or gap cap is used.

MWPM confidence computation uses up to 32 independent worker processes.
Native UF/BP collection uses 32 OpenMP threads across shots, with serial
inference/growth/search inside a shot. The phases run separately on the same
32 physical cores. Stim uses its vectorized native detector sampler.

Validation must compare native final corrections, remaining edge costs, and
cluster gaps against Python oracles, check all-zero syndromes, check 1-versus-32
thread invariance, and validate every native correction's physical syndrome.
MWPM variants must retain identical L1 predictions and native matching weights.
All four final outputs must satisfy the ideal yoke checks.

Report final block failures (any of twelve tracked patch-observable bits wrong),
exact binomial 95% intervals, all paired repairs/regressions and whole-shot
bootstrap differences, L1 failure counts, and elapsed collection times. Any
per-patch-round normalization is explicitly an effective normalization rather
than an independently measured physical-patch failure probability.

Scratch runs and builds belong under `$DANTE_SCRATCH/runs`; retained samples,
predictions, reports, source snapshots, and command/revision records go under
`results/`. Preserve the actual sample seed and binary hashes.
