# Using the weighted Union Find decoder

Run from the repository root with `PYTHONPATH=src`. The implementation uses
the existing project dependencies and requires no native build.

`UnionFindDecoder` and `SinterUnionFindDecoder` always use this repository's
weighted growth and peeling implementation. This is the default UF for our
experiments. Installing Fusion Blossom does not change that default.

**Decode a graph.** Each edge is `(u, v, weight, observable_mask)`. Use `v=None`
for a boundary edge. Weights must be finite and nonnegative; bit `k` of the
mask indicates that selecting the edge flips observable `k`.

```python
import numpy as np
from yoked.decoders import DecodingGraph, UnionFindDecoder

graph = DecodingGraph(
    num_detectors=2,
    num_observables=1,
    edges=[(0, 1, 1.0, 1), (1, None, 2.0, 0)],
)
decoder = UnionFindDecoder(graph)
prediction = decoder.decode(np.array([1, 1], dtype=np.bool_))
assert prediction.tolist() == [True]
predictions = decoder.decode_batch(np.array([[1, 1], [0, 0]], dtype=np.bool_))
assert predictions.tolist() == [[True], [False]]
```

Outputs are boolean arrays with one column per observable. The decoder keeps
the graph unchanged across calls and raises `InvalidSyndromeError` if no
correction can explain the supplied syndrome. It uses the graph's connectivity
directly; there are no geometry or yoke-specific decoder parameters.

**Decode a 1D yoked circuit.** Build a decomposed DEM, then import its graph:

```python
import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoders import DecodingGraph, UnionFindDecoder

circuit = yoked_magic_memory_circuit(
    patch_diameter=7,
    rounds=28,
    noise=gen.NoiseModel.si1000(0.003),
    style='cz',
    yokes=2,
    num_patches=6,
)
dem = circuit.detector_error_model(
    decompose_errors=True, approximate_disjoint_errors=True,
)
graph = DecodingGraph.from_dem(dem)
decoder = UnionFindDecoder(graph)
detectors = circuit.compile_detector_sampler(seed=42).sample(shots=1)
predictions = decoder.decode_batch(detectors)
assert predictions.shape == (1, circuit.num_observables)
```

DEM import uses PyMatching's graph export and validates its edges, weights,
and retained observable labels. UF decoding performs its own growth and
peeling. Unsupported DEM components fail explicitly during import.

**Use the Sinter adapter.** With `dem` and `detectors` from the previous example:

```python
import numpy as np
from yoked.decoders import SinterUnionFindDecoder

custom_decoders = {'uf_weighted': SinterUnionFindDecoder()}
compiled = custom_decoders['uf_weighted'].compile_decoder_for_dem(dem=dem)
packed_predictions = compiled.decode_shots_bit_packed(
    bit_packed_detection_event_data=np.packbits(detectors, axis=1, bitorder='little'),
)
np.testing.assert_array_equal(
    packed_predictions, np.packbits(predictions, axis=1, bitorder='little'),
)
```

The same `custom_decoders` mapping registers the adapter with Sinter under
the name `uf_weighted`.

**Optional correlated UF.** This variant uses two passes of the repository UF
decoder. The first correction supplies evidence about shared DEM error
mechanisms. That evidence lowers related edge weights, then a fresh UF pass
decodes the original syndrome and supplies the final prediction.

```python
from yoked.decoders import (
    CorrelatedUnionFindDecoder, SinterCorrelatedUnionFindDecoder, SinterUnionFindDecoder,
)

correlated_decoder = CorrelatedUnionFindDecoder.from_dem(dem)
correlated_predictions = correlated_decoder.decode_batch(detectors)

custom_decoders = {
    'uf_weighted': SinterUnionFindDecoder(),
    'uf_correlated': SinterCorrelatedUnionFindDecoder(),
}
```

The correlation rules are compiled once from the decomposed DEM, using the
pairwise implied-weight approximation used by PyMatching. Several mechanisms
can contribute to an edge or an edge pair; their probabilities combine by
parity. Implied probabilities are capped at 0.5 to keep weights nonnegative.
Both decoding passes use UF, with no MWPM solve or new dependency.

For a graph with externally supplied correlation information, construct
`CorrelatedUnionFindDecoder(graph, correlation_rules=rules)`. Each rule is
`(source_edge_id, target_edge_id, implied_weight)`, where IDs index
`graph.edges`. A source must be selected in the first correction to lower its
target's weight. Multiple sources use the lowest weight. Marginal graph weights
alone do not determine these relationships.

Each shot starts from the original graph weights, and the decoder leaves the
graph and syndrome unchanged. Rules may connect any graph edges; there are no
geometry or yoke-specific parameters. DEM errors that repeat an edge across
decomposed components are rejected. The plain `UnionFindDecoder` remains the
default, and correlated UF is an explicit decoder choice.

**Optional Fusion Blossom UF variant.** Install the optional package only if
you want to select this backend (tested with version 0.2.13):

```bash
UV_CACHE_DIR="$TMPDIR/uv-cache" uv pip install --python .venv/bin/python 'fusion-blossom==0.2.13'
```

Select the variant explicitly, using the same graph and syndrome arrays:

```python
from yoked.decoders import FusionBlossomUnionFindDecoder

optional_decoder = FusionBlossomUnionFindDecoder(graph)
optional_predictions = optional_decoder.decode_batch(detectors)
```

For Sinter, register the variant under a separate name:

```python
from yoked.decoders import SinterFusionBlossomUnionFindDecoder, SinterUnionFindDecoder

custom_decoders = {
    'uf_weighted': SinterUnionFindDecoder(),                 # Default: repository UF.
    'fusion_blossom_uf': SinterFusionBlossomUnionFindDecoder(),  # Explicit opt-in.
}
```

Fusion Blossom is always constructed with `max_tree_size=0` by these optional
classes. It uses blossom cluster contraction and shortest-path correction
reconstruction, with weights `2 * round(weight * weight_scale)` and
`weight_scale=1000` by default. It is a distinct UF variant; label its results
as `fusion_blossom_uf`. Sinter's built-in `fusion_blossom` decoder selects MWPM
and is a different decoder.

The optional adapter rejects parallel detector edges and weights that exceed
the native integer range. `DecodingGraph.from_dem` already merges parallel
DEM components. Impossible syndromes raise `InvalidSyndromeError` before
entering the native solver. Reuse an instance sequentially; create separate
instances for concurrent workers. Sinter constructs an instance in each worker.

**Run correctness tests.** Both d=3 and d=7 integration fixtures are included:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$TMPDIR/uf-pycache"
export MPLCONFIGDIR="$TMPDIR/uf-mpl"
.venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/decoders
```

The tests verify growth behavior, scan-oracle agreement, syndrome validity,
observable reconstruction, correlation reweighting, and the public interfaces.
Optional Fusion Blossom tests are skipped when its package is absent. The
default decoder is tested both with and without that optional dependency.
