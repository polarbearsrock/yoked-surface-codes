#!/usr/bin/env bash
# Build a wheel from the committed fork without writing build files into it.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)/env/workspace.sh"
source /opt/rh/gcc-toolset-14/enable
export UV_CACHE_DIR="$DANTE_SCRATCH/cache/uv"
export CCACHE_DIR="$DANTE_SCRATCH/cache/ccache"
export CCACHE_TEMPDIR="$CCACHE_DIR/tmp"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export DANTE_BUILD_JOBS="${DANTE_BUILD_JOBS:-32}"
if [[ ! "$DANTE_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || (( DANTE_BUILD_JOBS > 32 )); then
    echo "DANTE_BUILD_JOBS must be between 1 and 32." >&2
    exit 1
fi
if [[ -n "$(git -C "$DANTE_STIM" status --porcelain)" ]]; then
    echo "Commit the Stim changes first; this recipe records and builds a Git revision." >&2
    exit 1
fi
dante_revision="$(git -C "$DANTE_STIM" rev-parse HEAD)"
export DANTE_PYTHON_BUILD="$DANTE_SCRATCH/builds/stim-dante/${dante_revision:0:12}/python-build"
mkdir -p "$DANTE_PYTHON_BUILD/wheels" "$CCACHE_TEMPDIR"
if [[ ! -x "$DANTE_PYTHON_BUILD/venv/bin/python" ]]; then
    uv venv --python "$DANTE_REPO/.venv/bin/python" --no-python-downloads "$DANTE_PYTHON_BUILD/venv"
fi
mapfile -t dante_requirements < <("$DANTE_REPO/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import tomllib
print(*tomllib.loads((Path(os.environ['DANTE_STIM'])/'pyproject.toml').read_text())['build-system']['requires'], sep='\n')
PY
)
uv pip install --python "$DANTE_PYTHON_BUILD/venv/bin/python" "${dante_requirements[@]}"
git -C "$DANTE_STIM" archive --format=tar -o "$DANTE_PYTHON_BUILD/source.tar" HEAD
"$DANTE_PYTHON_BUILD/venv/bin/python" - <<'PY' > "$DANTE_PYTHON_BUILD/build.log" 2>&1
import os
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
from pybind11.setup_helpers import ParallelCompile
root = Path(os.environ['DANTE_PYTHON_BUILD'])
source = Path(tempfile.mkdtemp(prefix='source-', dir=root))
with tarfile.open(root/'source.tar') as archive:
    archive.extractall(source, filter='data')
os.chdir(source)
ParallelCompile('DANTE_BUILD_JOBS', default=32).install()
sys.argv = ['setup.py', 'build', '--build-base', str(root/'objects'),
    'bdist_wheel', '--dist-dir', str(root/'wheels'), '--bdist-dir', str(root/'wheel-stage')]
runpy.run_path('setup.py', run_name='__main__')
PY
uv pip freeze --python "$DANTE_PYTHON_BUILD/venv/bin/python" > "$DANTE_PYTHON_BUILD/build-requirements.txt"
printf '%s\n' "$dante_revision" > "$DANTE_PYTHON_BUILD/stim_revision.txt"
printf 'Wheel directory: %s\n' "$DANTE_PYTHON_BUILD/wheels"
