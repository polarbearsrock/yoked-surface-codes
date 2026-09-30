# Dante experiment recipes

These recipes checkpoint the Dante workspace experiments through 2026-09-30,
including the paired sum-product BP5, weighted-UF, and four-decoder sweeps.
They were originally run from `$DANTE_WORKSPACE/experiments` with
`$DANTE_REPO` pointing to this repository. To reproduce their original directory
layout, copy the relevant recipes into that workspace directory, source
`$DANTE_WORKSPACE/env/workspace.sh`, and activate this repository's `.venv`.

`docs/results/archive_index.json` records removed raw payloads and the retained
d=9 comparison sample. Historical manifests retain their original absolute paths;
other raw data must be regenerated from the recorded recipes and configurations.
Bulky data and build products belong under `$DANTE_SCRATCH`, not in Git. The
workspace `results/summaries` directory retains compact reports and figures.

New experiments added here should accept explicit sample/output paths and place
temporary output under `$DANTE_SCRATCH/runs`.
