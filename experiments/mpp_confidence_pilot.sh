#!/usr/bin/env bash
# Paired baseline/MPP pilot. Build instructions: repos/yoked-surface-codes/docs/mpp_confidence_experiment.md
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"

for distance in 7 9; do
    python "$DANTE_REPO/tools/mpp_experiment" \
        --distance "$distance" --rounds 12 --p 0.003 --patches 6 \
        --shots 2000 --seed "202609220${distance}" --method dijkstra \
        --out "$DANTE_SCRATCH/runs/mpp-pilot-d${distance}-20260922"
done
