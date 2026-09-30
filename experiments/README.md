# Dante experiment recipes

These recipes checkpoint the Dante workspace experiments through 2026-09-30,
including the paired sum-product BP5, weighted-UF, and four-decoder sweeps.
They were originally run from `$DANTE_WORKSPACE/experiments` with
`$DANTE_REPO` pointing to this repository. To reproduce their original directory
layout, copy the relevant recipes into that workspace directory, source
`$DANTE_WORKSPACE/env/workspace.sh`, and activate this repository's `.venv`.

Use the archived, frozen sources when reproducing an existing result exactly.
`docs/results/archive_index.json` maps the archived samples and source snapshots;
historical manifests retain their original absolute paths. Bulky data and build
products belong under `$DANTE_SCRATCH`, not in Git. The workspace `results/summaries`
directory retains compact reports and figures.

New experiments added here should accept explicit sample/output paths and place
temporary output under `$DANTE_SCRATCH/runs`.
