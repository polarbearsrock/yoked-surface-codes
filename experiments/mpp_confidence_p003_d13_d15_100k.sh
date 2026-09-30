#!/usr/bin/env bash
# Extend the paired accuracy comparison; all settings except distance stay fixed.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"
mpp_large_run="$DANTE_SCRATCH/runs/mpp-confidence-p003-d13-d15-100k-20260923"
mpp_large_recipe="$DANTE_WORKSPACE/experiments/mpp_confidence_100k.py"
if [[ -f "$mpp_large_run/recipe.py" ]]; then
    mpp_large_recipe="$mpp_large_run/recipe.py"
fi
python "$mpp_large_recipe" \
    --out "$mpp_large_run" \
    --p 0.003 --workers 32 --shots 100000 --shard-size 2500 --distances 13 15
