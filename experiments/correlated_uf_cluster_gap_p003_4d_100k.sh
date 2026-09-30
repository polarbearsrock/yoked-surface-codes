#!/usr/bin/env bash
# Run from any directory. Set UF_WORKERS to select the parallelism explicitly.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"

python "$DANTE_WORKSPACE/experiments/correlated_uf_cluster_gap.py" \
    --baseline-root "$DANTE_WORKSPACE/results/mpp_confidence_p003_4d_100k_20260923" \
    --out "$DANTE_SCRATCH/runs/correlated-uf-cluster-gap-p003-4d-100k" \
    --p 0.003 --rounds-per-distance 4 \
    --distances 7 9 11 13 15 --shots 100000 --shard-size 2500 \
    --workers "${UF_WORKERS:-8}"
