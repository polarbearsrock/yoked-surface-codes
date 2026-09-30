#!/usr/bin/env bash
# Build the Dante Stim checkout and the preserved reference sampler in scratch.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)/env/workspace.sh"
source /opt/rh/gcc-toolset-14/enable

dante_jobs="${DANTE_BUILD_JOBS:-32}"
if [[ ! "$dante_jobs" =~ ^[1-9][0-9]*$ ]] || (( dante_jobs > 32 )); then
    echo "DANTE_BUILD_JOBS must be between 1 and 32." >&2
    exit 1
fi
dante_revision="$(git -C "$DANTE_STIM" rev-parse HEAD)"
dante_build="$DANTE_SCRATCH/builds/stim-dante/${dante_revision:0:12}"
dante_reference="$DANTE_WORKSPACE/results/dongwhee_noise_model_review_20260929/source_snapshot/01_Baseline"
mkdir -p "$dante_build" "$DANTE_SCRATCH/cache/ccache/tmp"
export CCACHE_DIR="$DANTE_SCRATCH/cache/ccache"
export CCACHE_TEMPDIR="$CCACHE_DIR/tmp"

printf '%s\n' "$dante_revision" > "$dante_build/stim_revision.txt"
git -C "$DANTE_STIM" status --porcelain > "$dante_build/stim_worktree_status.txt"
g++ --version > "$dante_build/compiler_version.txt"

cmake -S "$DANTE_STIM" -B "$dante_build/cmake" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)"
cmake --build "$dante_build/cmake" --target stim libstim --parallel "$dante_jobs"

# The supplied Makefile recompiles every Stim source in one compiler invocation.
# Reuse the library above while compiling the original sampler sources unchanged.
g++ -O3 -std=c++20 -march=native -fno-strict-aliasing -fopenmp \
    -I"$DANTE_STIM/src" \
    "$dante_reference/main.cpp" "$dante_reference/simulation.cpp" \
    "$dante_build/cmake/out/libstim.a" -lpthread \
    -o "$dante_build/surface_sim"

printf 'Stim CLI: %s\nReference sampler: %s\n' \
    "$dante_build/cmake/out/stim" "$dante_build/surface_sim"
