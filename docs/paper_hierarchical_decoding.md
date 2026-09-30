# Correlated MWPM → complementary gap → plain MWPM

This configuration follows the decoding construction in section 4 of
[Yoked surface codes, arXiv:2312.04522](https://arxiv.org/pdf/2312.04522):

1. **L1:** independently decode each patch using PyMatching's native two-pass
   correlated MWPM (`enable_correlations=True`). Keep those X/Z predictions as
   the reference bits `r[i, s]`.
2. **Confidence:** compare the two logical classes on the same frozen,
   correlation-reweighted graph. Their matching-cost difference is the
   complementary gap, in natural-log units (nats).
3. **L2:** use these gaps as edge weights in plain MWPM across patches. Adjust
   the measured yoke syndrome by the reference parity first, then XOR L2's
   residual flips with the reference bits.

The implementation uses the repository's existing **1D block with two ideal
yoke checks**, with one independent outer solve per X/Z sector. It does not
implement the paper's 2D layout or multi-round phenomenological gap sampling.
PyMatching replaces the paper's internal correlated matcher; equivalent logical
error rates to that implementation have not been established.

## Run the configuration

From the canonical repository path:

```bash
source /data2/s2chitni/projects/dante/env/workspace.sh
cd "$DANTE_REPO"
source .venv/bin/activate
export PYTHONPATH="$DANTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# A small end-to-end run, with a saved sample and decoder provenance.
python tools/hierarchical_experiment paper \
    --distance 3 --rounds 12 --p 0.003 --patches 6 \
    --seed 42 --shots 100 --gap-scale 0.9 \
    --out "$DANTE_SCRATCH/runs/paper-correlated-d3"
```

To compare with previous experiments on exactly the same shots, reuse the
verified sample under an existing collection:

```bash
python tools/hierarchical_experiment paper \
    --sample "$DANTE_SCRATCH/runs/existing-evaluation/sample" \
    --gap-scale 0.9 \
    --out "$DANTE_SCRATCH/runs/paper-correlated-evaluation"
```

`--recorded-run /path/to/four-decoder-run` also accepts the original recorded-run
format. Choose one of `--sample`, `--recorded-run`, or the complete generation
arguments. `--rows START:STOP` selects parent rows without resampling;
`--batch-size` bounds the number of unpacked syndromes held at once. Outputs
require a new or empty directory. This command runs serially; Sinter can run
the same decoder in parallel workers, as shown below.

The `paper` command is a direct decoding configuration. It needs neither a
calibration dataset nor a `calibrators.json` file. Its default `--gap-scale 1.0`
passes raw gaps to L2. `--gap-scale 0.9` applies the paper's empirical confidence
rescaling. A uniform positive scale leaves a unique minimum-cost outer pattern
unchanged in this single-round parity model. The reported raw gap is always
unscaled; `outer_weights` records the value actually used by L2.

## Python and Sinter

```python
from yoked.hierarchical import PaperHierarchicalDecoder

# dem is the decomposed model of the existing 1D, two-yoke circuit.
# detectors has shape (shots, dem.num_detectors).
decoder = PaperHierarchicalDecoder(dem, gap_scale=0.9)
result = decoder.decode_with_gaps_batch(detectors)
predictions = result.prediction
reference = result.reference
gaps = result.complementary_gap

# If only final predictions are needed:
predictions = decoder.decode_batch(detectors)
```

All per-patch arrays use columns `2*i + s`, with `s=0` for X and `s=1` for Z.
The result also contains `forced_weights[shot, patch, c_X, c_Z]`,
`outer_weights`, `sigma`, `residual`, and `outer_tied`. Returned arrays are
read-only. `CorrelatedMatchingGapDecoder(patch)` exposes L1 alone.

To use existing Sinter collection workflows:

```python
import sinter
from yoked.hierarchical import SinterPaperHierarchicalDecoder

stats = sinter.collect(
    tasks=[sinter.Task(circuit=circuit)],
    decoders=['paper_correlated_mwpm'],
    custom_decoders={
        'paper_correlated_mwpm': SinterPaperHierarchicalDecoder(gap_scale=0.9),
    },
    num_workers=4,
    max_shots=1000,
)
```

Place multiprocessing collection inside `if __name__ == '__main__':` in scripts.

## How the complementary gap is computed

For a patch syndrome, let `W(c_X, c_Z)` be the forced matching cost under the
frozen weights. The two sectors are disconnected after conditioning. Relative
to the native correlated prediction `(r_X, r_Z)`, the returned gaps are

```text
gap_X = W(1-r_X, r_Z) - W(r_X, r_Z)
gap_Z = W(r_X, 1-r_Z) - W(r_X, r_Z)
```

A zero gap denotes a logical tie. A large positive gap means a complementary
logical correction is expensive. The decoder preserves PyMatching's native
reference, including its choice on ties. It does not substitute the old UF
reference or the first-pass plain-MWPM prediction.

PyMatching 2.4's public Python API returns correlated predictions and their
weights, but does not expose the temporary reweighted graph. The gap adapter:

1. Loads the full local DEM with correlations enabled, preserving its separator
   groups, probabilities, and logical labels.
2. Gets the first-pass selected edges from that native matcher with
   `enable_correlations=False`, and obtains the native correlated reference
   with `enable_correlations=True`.
3. Reconstructs PyMatching 2.4's conditional edge discounts from the shared DEM
   mechanisms. It reproduces native integer discretization, including implied
   weights that do not lower an edge but still affect the normalization scale.
4. Holds those integer weights fixed and runs four ordinary PyMatching solves
   with check bits `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.
5. Checks on every shot that the native reference class is a minimum, its cost
   agrees with the native correlated solve, and the forced costs are additive
   across sectors. Any disagreement raises before publishing a result.

The reconstruction is a compatibility adapter for **PyMatching 2.4.x**; the
dependency is pinned to that minor series. Matching itself always runs in
PyMatching. A new minor release requires reviewing its weight semantics and
validating the adapter. Relevant upstream sources are
[user_graph.cc](https://github.com/oscarhiggott/PyMatching/blob/v2.4.0/src/pymatching/sparse_blossom/driver/user_graph.cc),
[user_graph.h](https://github.com/oscarhiggott/PyMatching/blob/v2.4.0/src/pymatching/sparse_blossom/driver/user_graph.h), and
[search_graph.cc](https://github.com/oscarhiggott/PyMatching/blob/v2.4.0/src/pymatching/sparse_blossom/search/search_graph.cc).

Running a fresh correlated solve separately for each forced class is deliberately
avoided: it could use different conditioning corrections, making a subtraction
compare costs from different reweighted graphs.

L2 receives `gap_scale * gap` directly as a log-odds weight. This preserves large
gaps without converting through `q = 1/(1+exp(gap))`, which can underflow or lose
information if probabilities are clipped. Its syndrome and output are

```text
sigma[s] = yoke[s] XOR parity_i(reference[i,s])
final[i,s] = reference[i,s] XOR residual[i,s]
```

The existing outer backend preserves one edge identity per patch and checks
finite-precision matching choices against the original weights. L2 performs
no correlation reweighting. Its tie rule is the existing lowest-binary-pattern
rule within `1e-9` nats. Every final prediction is checked against the yokes.

## Retained outputs and verification

The run writes:

- `predictions.npz`: parent row IDs, native references, raw gaps, all four forced
  class costs, scaled outer weights, adjusted syndromes, L2 flips and ties,
  final predictions, and actual observable flips.
- `results.json`: the configuration, scale, shot count, block failures before
  and after L2, outer ties, and elapsed software time.
- `manifest.json`: sample identities and hashes, configuration, source hashes,
  dependency versions, Git revision, and output hashes. This is written last.
- `sample/`: the complete sampling call when the input was newly generated.
  Existing samples are referenced by provenance instead of duplicated.

This configuration has its own artifact format. Existing UF/matching-gap records
and fitted calibrators retain their previous meanings; the new command decodes
the saved syndromes afresh.

Tests cover analytical correlated gaps, every syndrome of a small model, exact
native-reference agreement on a circuit fixture, frozen-cost agreement, yoke
isolation from L1, an exhaustive L2 oracle, high-gap arithmetic, Sinter packing,
row-subset reproducibility, and the CLI:

```bash
python -m pytest -q -o cache_dir="$DANTE_SCRATCH/cache/pytest" \
    src/yoked/hierarchical/_paper_decoder_test.py \
    src/yoked/hierarchical/_paper_stage_test.py \
    src/yoked/hierarchical/_outer_mwpm_test.py
```
