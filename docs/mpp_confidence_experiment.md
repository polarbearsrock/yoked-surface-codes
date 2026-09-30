# MPP confidence versus complementary gaps

This experiment integrates the optimized public-MPP implementation supplied for
Dante. It compares two confidence estimators while holding the decoding hierarchy
fixed:

| Component | Baseline | MPP experiment |
|---|---|---|
| L1 prediction | PyMatching 2.4.0 correlated MWPM | Same correlated MWPM prediction |
| Confidence | Complementary gap on frozen correlated weights | MPP cluster score on the live correlated graph |
| L2 | Plain PyMatching MWPM across patches | Same solver and tie convention |
| Evaluation data | Saved detector shots and truth labels | Identical shots and labels |

The baseline is documented in [paper_hierarchical_decoding.md](paper_hierarchical_decoding.md).
Both configurations currently cover the repository's six-patch, two-ideal-yoke
model (or another supported patch count), not a full multi-round outer code.

## What the score means

For each logical sector, MPP computes the shortest path between the ordinary and
logical boundaries. An internal edge has length

```
length(u, v) = max(0, w(u, v) - r(u) - r(v))
```

Here `w` is the engine's integer edge weight after first-pass correlation
reweighting. `r` is the final local radius after the second matching pass and
before blossoms are shattered. A boundary edge has length `max(0, w - r(u))`.
The reported score divides the integer path length by the native normalization.
Larger scores indicate greater confidence.

This path distance is exact for that modified graph. **It is an approximation
to complementary-gap confidence**, not the exact difference between optimal
logical-class matching costs, and not a calibrated error probability.
The first experiment passes raw MPP scores directly to L2 as heuristic weights.
No calibration or parameter tuning uses evaluation truth. A calibrated experiment
would need independent calibration shots and a calibrated-gap control.

The supplied `dijkstra` mode uses a bucket queue. `incremental` starts from
precomputed static distances and propagates only distance decreases. Both should
return exactly the same integer score; choosing a method is a performance choice,
not a different accuracy configuration. The bucket width can exceed the lightest
positive edge to control bucket count; scanning each bucket until stable maintains
correctness. This corrects the more restrictive statement in the supplied header.

The hierarchy always selects `correlation="all"`, matching the baseline's native
`enable_correlations=True`. The low-level extension retains `none` and `cross`
for separate studies, but either can change the reference decoder. The paired
experiment rejects any mismatch in L1 predictions or chosen-class matching weights.

## Build the optional extension

The adapter requires C++20, CMake, Ninja, and the repository's Python environment.
It compiles its own pinned engine; it does not modify the installed PyMatching
wheel or depend on symbols exported by that wheel. Python import never triggers
a build or download. All checkouts and build artifacts below stay in scratch.

From `/data2/s2chitni/projects/dante`:

```bash
source /data2/s2chitni/projects/dante/env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
mkdir -p "$DANTE_SCRATCH/native"

# Run these downloads once. Reuse existing verified checkouts on later builds.
git clone --depth 1 --branch v2.4.0 https://github.com/oscarhiggott/PyMatching.git \
    "$DANTE_SCRATCH/native/PyMatching-2.4.0"
git -C "$DANTE_SCRATCH/native/PyMatching-2.4.0" submodule update --init pybind11
git clone https://github.com/quantumlib/Stim.git "$DANTE_SCRATCH/native/Stim"
git -C "$DANTE_SCRATCH/native/Stim" checkout 1320ad7eac7de34d2e9c70daa44fbc6d84174450

python "$DANTE_REPO/tools/build_mpp" \
    --pymatching-source "$DANTE_SCRATCH/native/PyMatching-2.4.0" \
    --stim-source "$DANTE_SCRATCH/native/Stim" \
    --out "$DANTE_SCRATCH/build/mpp-gcc14" \
    --cxx /opt/rh/gcc-toolset-14/root/usr/bin/g++ \
    --cc /opt/rh/gcc-toolset-14/root/usr/bin/gcc
export YOKED_MPP_BUILD="$DANTE_SCRATCH/build/mpp-gcc14"
```

Those compiler paths select the workspace's installed GCC 14; on another host,
provide available C++20/C compilers or omit the flags when defaults are suitable.
Use a fresh output directory when changing compilers. The build tool requires
clean dependency checkouts at these revisions:

| Dependency | Revision |
|---|---|
| PyMatching 2.4.0 | `6f63b2b9474ba0fa7e511fe52bffdce858a06984` |
| Stim, as pinned by PyMatching | `1320ad7eac7de34d2e9c70daa44fbc6d84174450` |
| pybind11, as pinned by PyMatching | `a2e59f0e7065404b44dfe92a28aca47ba1378dc4` |

`build.json` records these revisions, adapter/build-source hashes, compiler,
Python version, commands, and binary hash. The loader refuses an edited binary
or a build whose adapter/build sources have changed. Re-run `tools/build_mpp`
after editing native code. The optional extension was tested with Python 3.14.

## Run a paired accuracy experiment

The distance sweep uses **4d rounds**: 28, 36, 44, 52, and 60 rounds at
distances 7, 9, 11, 13, and 15. Set the rounds explicitly when calling the
single-run CLI. Earlier retained fixed-12-round experiments describe a
different circuit family and must not be mixed into this sweep or its d=21
(84-round) extrapolation.

```bash
python "$DANTE_REPO/tools/mpp_experiment" \
    --distance 7 --rounds 28 --p 0.003 --patches 6 \
    --shots 2000 --seed 2026092207 \
    --method dijkstra \
    --out "$DANTE_SCRATCH/runs/mpp-4d-pilot-d7"
```

Use a new output directory for each run. To compare the alternative search on the
*same saved shots*, rather than generating another sample:

```bash
python "$DANTE_REPO/tools/mpp_experiment" \
    --sample "$DANTE_SCRATCH/runs/mpp-4d-pilot-d7/sample" \
    --rows 0:200 --method incremental --verify-scores \
    --out "$DANTE_SCRATCH/runs/mpp-4d-pilot-d7-incremental-check"
```

The workspace recipe `experiments/mpp_confidence_p003_4d_100k.sh` runs 100,000
paired shots at each of the five distances, with verified, resumable shards.
Its underlying runner requires either `--rounds-per-distance 4` or an explicit
fixed `--rounds` value and records the actual count for every distance.

`--sample` loads a verified `SampleSet`; `--recorded-run` imports saved shots from
the earlier four-decoder runs. Generation flags cannot be combined with either.
`--rows START:STOP` selects original parent row IDs. Both methods process exactly
the selected rows, with identical L2 tie handling and no postselection.

The output contains:

- `predictions.npz`: reference bits, both confidence arrays, baseline forced-class
  costs, native MPP matching weights, both final predictions and residuals, yoke
  syndrome, tie flags, truth labels, and parent row IDs.
- `results.json`: block failures, paired MPP repairs/regressions, single-L1-error
  misattribution with denominators, and failure counts stratified by L1 errors.
  The paired difference is **MPP minus complementary gap**; positive is worse.
  Its 95% percentile interval resamples whole shots, preserving sector dependence.
- `manifest.json`: exact sample identity, code commit and source hashes, native
  build metadata, dependency versions, settings, and hashes of result artifacts.
  It is written only after all checks pass.
- `sample/`: the exact circuit, DEM, and packed data when the run generated shots;
  existing verified samples are referenced instead of copied.

A failed run has no completed manifest. This initial harness requires a fresh
directory and does not resume partial decoding. For a large study, use bounded
parent-row ranges and retain each range's artifacts. An observed zero difference
or a narrow empirical interval on a small pilot does not establish equivalence.

## Validation and Python use

```bash
python -m pytest "$DANTE_REPO/src/yoked/hierarchical/_mpp_test.py" \
    "$DANTE_REPO/src/yoked/hierarchical/_mpp_experiment_test.py" \
    -q -o cache_dir="$DANTE_SCRATCH/cache/pytest"
```

Tests cover analytic boundary paths, all syndromes of small models, zero weights,
native predictions and weights, both search modes, all correlation modes, sparse
and dense syndromes, state reuse, non-contiguous arrays, invalid inputs, and the
complete local-to-outer parity flow. The optional native tests skip when no build
is selected. `--verify-scores` enables a separate full-radius-scan, binary-heap
Dijkstra oracle for exact integer comparison on each shot; it is intentionally
excluded from normal evaluation and any performance measurements.

```python
from yoked.hierarchical import MppHierarchicalDecoder

decoder = MppHierarchicalDecoder(dem, method="dijkstra")
result = decoder.decode_with_scores_batch(detectors)
# All arrays use patch-major columns (X0, Z0, X1, Z1, ...).
print(result.cluster_score, result.reference, result.prediction)
```

The implementation is split between `native/mpp/mpp_fast.cc` (the supplied
algorithm plus validation hooks), `_mpp.py` (optional loading and hierarchy), and
`_mpp_experiment.py` (paired accuracy and provenance). The native adapter performs
the usual two correlated-matching passes, then two sector path searches; it does
not run the baseline's forced complementary-class matchings. No speedup is claimed
from the accuracy runs, and agreement with a separate public MPP source revision
has not been checked; the independent oracle checks the supplied score definition.
