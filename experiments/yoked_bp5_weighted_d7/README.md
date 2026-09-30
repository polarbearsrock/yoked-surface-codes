# BP5 + weighted UF correction to the d=7 comparison

Use the exact 100,000 saved shots from the preceding four-way experiment:
six distance-7 patches, two ideal yokes, 28 CZ extraction rounds, SI1000
p=0.003, ideal time boundaries, sampling seed 202609290707, and the verified
Dante Stim fork. Retain the first three configurations' predictions unchanged.

The corrected fourth configuration performs exactly five sum-product flooding
BP iterations (damping 0.5, LLR clipping at 30). The existing posterior-to-edge
projection is `-log(clip(sum of supporting fault posteriors, 1e-15, 1))`.
Run weighted UF once on these weights and the original syndrome. Compute the
full, uncapped cluster gap from this pass's final growth costs. There are no
conditional correlation discounts or second UF pass. Use the unchanged plain
MWPM L2, without tuning or confidence calibration.

Reuse the frozen prior BP, UF growth/peeling, and cluster-gap implementations.
Validate native corrections, masks, remaining costs and gaps against independent
Python BP + weighted UF oracles, including all 16 synthetic syndromes and eight
fresh d=7 cases. Compare the L1 corrections against the existing native BP5 +
weighted UF implementation. Verify one-versus-32-thread invariance on 16 fresh
shots for two patches. Every collected native correction must satisfy its input
syndrome; every final L2 prediction must satisfy the two ideal yoke checks.

Collect only the corrected variant, using 32 OpenMP threads across shots on the
same 32 physical cores. Checkpoint every 1,250 shots. Keep temporary outputs and
builds under $DANTE_SCRATCH/runs, and preserve the earlier completed experiment.

Report the corrected four-way block failure comparison, exact binomial 95%
intervals, all paired differences with whole-shot bootstrap intervals, and the
corrected variant versus the prior BP5 + correlated UF variant separately.
The block event and optional effective per-patch-round normalization retain the
preceding experiment's definitions. Retain samples, predictions, source hashes,
build commands and validation results in a new results directory.

```bash
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
python experiments/yoked_bp5_weighted_d7/run.py --baseline results/yoked_fourway_d7_p003_100k_20260929T201317Z
# Resume using the frozen recipe and its printed output directory:
python RESULTS_DIRECTORY/source_snapshot/experiments/yoked_bp5_weighted_d7/run.py --output RESULTS_DIRECTORY
# After collection:
python RESULTS_DIRECTORY/source_snapshot/experiments/yoked_bp5_weighted_d7/analyze.py --output RESULTS_DIRECTORY
```
