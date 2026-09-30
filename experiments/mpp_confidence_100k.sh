#!/usr/bin/env bash
# Re-run or resume the paired 100k-shot SI1000 p=0.001 comparison.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"
mpp_p001_run="$DANTE_SCRATCH/runs/mpp-confidence-p001-100k-20260923"
mpp_p001_recipe="$DANTE_WORKSPACE/experiments/mpp_confidence_100k.py"
# Resume with the recorded recipe even when the workspace recipe has evolved.
if [[ -f "$mpp_p001_run/recipe.py" ]]; then
    mpp_p001_recipe="$mpp_p001_run/recipe.py"
fi
python "$mpp_p001_recipe" \
    --out "$mpp_p001_run" \
    --workers 32 --shots 100000 --shard-size 5000 --distances 7 9
