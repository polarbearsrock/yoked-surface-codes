# Dante Stim fork

- Local checkout: `/data2/s2chitni/projects/dante/repos/stim-dante`
- GitHub fork: <https://github.com/polarbearsrock/stim-dante>
- Upstream: <https://github.com/quantumlib/Stim>
- Development/default branch: `dante/main`
- Current upstream base: `main`, commit `131793efb34238481fadb1b4959f2395ec3f722a`, version `1.17.dev0`
- Dante merge commit: `db493f5987514ff4724d972bdbd2597038f9e9a0`; its source tree equals the upstream base
- Initial validated release: tag `dante-baseline-v1.16.0`, commit `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`

This is an independent repository alongside the decoder checkout. The decoder
Python environment now uses the fork-built `1.17.dev0` wheel. Both the source
commit and installed binary hashes are recorded in `env/stim-fork.json`.
The fork retains upstream history, the initial release baseline, and a separate
`main` branch. The local `dante/main` branch tracks `origin/dante/main`.

## Build

```bash
cd /data2/s2chitni/projects/dante
source env/workspace.sh
bash experiments/stim_dante_setup/build.sh
```

The recipe uses the server's GCC 14 toolset, C++20, and up to 32 parallel build
jobs. It builds the upstream CLI and static library, then links Dongwhee's
unchanged `main.cpp` and `simulation.cpp` against that library with OpenMP.
It does not run the original sweep launcher. Compiler and source-revision
records accompany the binaries in
`$DANTE_SCRATCH/builds/stim-dante/<12-character-commit>/`.

All build files and caches stay in scratch; `build.sh` does not install or replace
the Python package in the decoder environment. The C++ sampler sources remain
in the retained local reference snapshot, outside the public Stim fork.

## Python package

```bash
cd /data2/s2chitni/projects/dante
source env/workspace.sh
bash experiments/stim_dante_setup/build_python.sh
dante_revision="$(git -C "$DANTE_STIM" rev-parse --short=12 HEAD)"
UV_CACHE_DIR="$DANTE_SCRATCH/cache/uv" uv pip install \
    --python "$DANTE_REPO/.venv/bin/python" --no-deps --no-index \
    --reinstall-package stim \
    "$DANTE_SCRATCH/builds/stim-dante/$dante_revision/python-build/wheels/"*.whl
```

The source build passed with Python 3.14.5, GCC 14.2.1 and upstream's existing
`pybind11==2.11.1` dependency. No Python binding code or build-dependency changes
were needed. The recipe copies the committed source into scratch and compiles
up to 32 source files at a time. Installing the wheel leaves other dependencies
alone; the usual `import stim` loads the fork build. It is a fixed wheel, so
subsequent source changes require rebuilding and reinstalling it.

Before installation, `verify_experiments.py` checked unchanged circuit/DEM
generation, all seven decoders against saved rows at d=7/9/11/13, native physical
corrections on fresh samples, and imports/sampling in spawned processes.
Dongwhee's sampler also passed the smoke checks on current `main`.
Records are retained under `results/stim_dante_main_20260929/`.

The previous 1.16.0 package was copied to
`$DANTE_SCRATCH/builds/stim-dante/db493f598751/python-build/baseline-package`.
For a read-only replay with that package, prepend this directory to `PYTHONPATH`.
Keep old saved experiment configurations and source snapshots unchanged.

## Smoke checks

```bash
cd /data2/s2chitni/projects/dante
source env/workspace.sh
source /opt/rh/gcc-toolset-14/enable
source "$DANTE_REPO/.venv/bin/activate"
dante_smoke_parent="$(mktemp -d "$DANTE_SCRATCH/runs/stim-dante-smoke-XXXXXX")"
dante_revision="$(git -C "$DANTE_STIM" rev-parse --short=12 HEAD)"
python experiments/stim_dante_setup/smoke.py \
    --build "$DANTE_SCRATCH/builds/stim-dante/$dante_revision" \
    --output "$dante_smoke_parent/output"
```

Both the initial release build and the subsequent main build passed these checks on 2026-09-29:

- Willow and Berlin, X and Z memory, d=7 and r=7: 128 samples per configuration,
  identical measurement samples and physical-fault counts with 1 and 32 OpenMP
  threads; all four sample sets converted to detectors and decoded with correlated MWPM.
- Noiseless X and Z memory: 64 samples each, zero detectors and logical flips.
- Built Stim CLI: successfully sampled 16 detector shots.

These check the build, C++/Python interfaces, and thread-count reproducibility.
They do not establish statistical equivalence between the original C++ sampler
and Python circuit sampler or constitute a performance benchmark. Retained
records are in `results/stim_dante_setup_20260929/` for the initial release and
`results/stim_dante_main_20260929/` for the subsequent main build.

## Dongwhee's directory convention

His original Makefile expects `01_Baseline/` and `Stim/` to be siblings. That
name is only a relative include/source path. Our build recipe instead passes
`$DANTE_STIM/src` explicitly and links the static library, so the checkout can
keep its descriptive `stim-dante` name without a compatibility symlink.

If reproducing the exact original Makefile later, prepare a temporary parent
under `$DANTE_SCRATCH/builds`, copy the baseline sources into `01_Baseline/`, and
create a sibling `Stim` symlink pointing to `$DANTE_STIM`.

## Updating or extending Stim

```bash
source /data2/s2chitni/projects/dante/env/workspace.sh
cd "$DANTE_STIM"
git fetch upstream
git switch -c dante/<experiment-name> dante/main
```

Use `dante-baseline-v1.16.0` for the initial reference reproduction. Continue
development from `dante/main`, recording the actual source commit for each build.
