#!/usr/bin/env bash
# Extend the paired SI1000 p=0.003 comparison to distance 11.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"
mpp_d11_run="$DANTE_SCRATCH/runs/mpp-confidence-p003-d11-100k-20260923"
mpp_d11_recipe="$DANTE_WORKSPACE/experiments/mpp_confidence_100k.py"
if [[ -f "$mpp_d11_run/recipe.py" ]]; then
    mpp_d11_recipe="$mpp_d11_run/recipe.py"
fi
python "$mpp_d11_recipe" \
    --out "$mpp_d11_run" \
    --p 0.003 --workers 32 --shots 100000 --shard-size 2500 --distances 11
