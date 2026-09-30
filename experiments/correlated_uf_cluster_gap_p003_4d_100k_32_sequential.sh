#!/usr/bin/env bash
# Resume completed shards, finishing one distance before starting the next.
set -euo pipefail
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"

python "$DANTE_WORKSPACE/experiments/correlated_uf_sequential.py" \
    --run "$DANTE_SCRATCH/runs/correlated-uf-cluster-gap-p003-4d-100k" \
    --workers 32 \
    --control "$DANTE_SCRATCH/runs/correlated-uf-cluster-gap-p003-4d-100k-control" \
    --retained "$DANTE_WORKSPACE/results/correlated_uf_cluster_gap_p003_4d_100k_20260923"
