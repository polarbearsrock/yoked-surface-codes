#!/usr/bin/env bash
# Corrected paired sweep: explicitly enforce four rounds per unit of distance.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"
mpp_4d_run="$DANTE_SCRATCH/runs/mpp-confidence-p003-4d-100k-20260923"
mpp_4d_recipe="$DANTE_WORKSPACE/experiments/mpp_confidence_100k.py"
if [[ -f "$mpp_4d_run/recipe.py" ]]; then
    mpp_4d_recipe="$mpp_4d_run/recipe.py"
fi
python "$mpp_4d_recipe" \
    --out "$mpp_4d_run" --p 0.003 --rounds-per-distance 4 \
    --workers 64 --shots 100000 --shard-size 2500 --distances 7 9 11 13 15
