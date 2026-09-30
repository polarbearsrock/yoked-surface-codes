#!/usr/bin/env bash
# One million paired shots at each distance; exactly one 32-worker pool at a time.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"

exec python "$DANTE_WORKSPACE/experiments/hierarchical_accuracy_sweep.py" \
    --out "$DANTE_SCRATCH/runs/hierarchical-accuracy-p003-4d-1m-20260924" \
    --retained "$DANTE_WORKSPACE/results/hierarchical_accuracy_p003_4d_1m_20260924" \
    --p 0.003 --shots 1000000 --workers 32 --shard-size 1250 \
    --seed-base 2026092400 --distances 7 9 11 13 15
