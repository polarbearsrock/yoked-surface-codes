#!/bin/bash
# Usage: run_cell_pipeline.sh CELL_DIR D ROUNDS P SEED SHOTS PAGE_OUT
set -euo pipefail
CELL=$1; D=$2; ROUNDS=$3; P=$4; SEED=$5; SHOTS=$6; PAGE=$7
S=/data2/s2chitni/yoked-surface-codes/out/cluster_size_study_d7; PY=/data2/s2chitni/yoked-surface-codes/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONPATH=/data2/s2chitni/yoked-surface-codes/src TMPDIR=/data2/s2chitni/.tmp
cd /data2/s2chitni/yoked-surface-codes
stage(){ echo "=== $(date '+%F %T') $1"; }
[ -f "$CELL/cell.npz" ]                  || { stage collect;        $PY $S/collect_cell.py --d "$D" --rounds "$ROUNDS" --p "$P" --seed "$SEED" --shots "$SHOTS" --out "$CELL" --processes 32; }
[ -f "$CELL/frontier_interior.json" ]    || { stage ladders;        $PY $S/frontier_families.py --cell "$CELL" --processes 32; }
[ -f "$CELL/l2_timing.json" ]            || { stage l2_timing;      $PY $S/l2_timing.py --cell "$CELL" --processes 32; }
[ -f "$CELL/microblossom_cycles.json" ]  || { stage microblossom;   $PY $S/microblossom_cycles.py "$CELL"; }
[ -f "$CELL/zerog_cycles.json" ]         || { stage zerog;          $PY $S/zerog_cycles.py "$CELL"; }
[ -f "$CELL/closing_time.npy" ]          || { stage replay_closing; $PY $S/replay_closing_times.py "$CELL" 32; }
[ -f "$CELL/helios_quantum.json" ]       || { stage quantum;        $PY $S/helios_quantum.py "$CELL"; }
[ -f "$CELL/cap_ladder.json" ]           || { stage cap_ladder;     $PY $S/cap_ladder.py --cell "$CELL" --processes 32; }
[ -f "$CELL/event_times.npz" ]           || { stage replay_events;  $PY $S/replay_event_times.py "$CELL" 32; }
[ -f "$CELL/helios_budget_cycles.json" ] || { stage helios_budget;  $PY $S/helios_budget_cycles.py "$CELL"; }
stage page; $PY $S/build_talk_page.py "$CELL" "$PAGE"
stage done
