#!/usr/bin/env bash
# Paired 100k-shot comparison at SI1000 p=0.003 (0.3%).
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"
mpp_p003_run="$DANTE_SCRATCH/runs/mpp-confidence-p003-100k-20260923"
mpp_p003_recipe="$DANTE_WORKSPACE/experiments/mpp_confidence_100k.py"
# Resume with the exact recorded recipe; source/build identity checks still apply.
if [[ -f "$mpp_p003_run/recipe.py" ]]; then
    mpp_p003_recipe="$mpp_p003_run/recipe.py"
fi
python "$mpp_p003_recipe" \
    --out "$mpp_p003_run" \
    --p 0.003 --workers 32 --shots 100000 --shard-size 5000 --distances 7 9
