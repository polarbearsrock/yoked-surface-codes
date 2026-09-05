# Weighted Union Find decoder design

Implement a complete weighted Union Find (UF) decoder: receive a graph and a
syndrome, grow and merge clusters, peel a correction, and return predicted
logical observables. The core operates on graph connectivity, weights, and
observable labels. The same algorithm applies to supported graphs from 1D or
2D code generators; geometry and patch metadata are not decoder inputs.

The first integration target is the repository's 1D yoked surface code. The
[implementation plan](union_find_implementation_plan.md) defines the delivery
steps and correctness checks. Implementation is pending. Benchmarking will be
planned after the decoder has been implemented and reviewed.

**Graph input.** `DecodingGraph` contains detector and observable counts, an
ordered collection of edges, and adjacency lists. An edge is a tuple
`(u, v, weight, observable_mask)`, with detector endpoints `u` and `v`, or
`v=None` for a boundary edge. Weights are finite and nonnegative. Edge IDs are
their positions in the collection; directly supplied parallel edges stay distinct.
Preserve isolated detectors and observable columns absent from the edges.
Use Python integers for observable masks and keep the graph immutable between
shots. Validate endpoint ranges, distinct endpoints, weights, and mask widths.

Callers can construct this graph directly. A `DecodingGraph.from_dem(dem)`
adapter supports the repository's Stim circuits. Build the DEM with
`decompose_errors=True, approximate_disjoint_errors=True`. Audit flattened
error components, reducing detector and observable targets modulo two and
splitting at `^`. Reject nonzero-probability components with more than two
detectors or with observables but no detectors. Preserve zero-cost edges;
reject negative or nonfinite weights explicitly.

The adapter exports
`pymatching.Matching.from_detector_error_model(dem).edges()` to obtain weights
and observable labels. This reuses graph construction without running a
matching solve. Preserve the importer's parallel-edge result, including its
retained observable labels; do not XOR labels when combining parallel edges.
Take detector and observable counts from the DEM. The UF core itself depends
only on `DecodingGraph` and uses the supplied weights as growth lengths.

Give each boundary edge its own unconstrained virtual terminal internally.
Every real detector, including a yoke, keeps its syndrome constraint. Local
checks and yoke checks are already present in the complete input graph. A
cluster acquires another detector only when a connecting edge completes
growth. Reaching a yoke therefore uses exactly the same merge operation as
reaching any other detector.

**Decoder interface.** Provide one Python implementation with this public API:

```python
graph = DecodingGraph(num_detectors, num_observables, edges)
decoder = UnionFindDecoder(graph)
prediction = decoder.decode(syndrome)
predictions = decoder.decode_batch(syndromes)
```

`decode` accepts a binary array of shape `(num_detectors,)` and returns a boolean
array of shape `(num_observables,)`. `decode_batch` accepts
`(shots, num_detectors)` and returns `(shots, num_observables)`; initially it
can loop over the single-shot implementation. Inputs are not modified. Reset
per-shot state before each decode, including after an error. Retain selected
correction edge IDs in an internal result so tests can validate the correction.

A thin `SinterUnionFindDecoder` adapter compiles a DEM into this decoder and
implements Sinter's bit-packed batch interface. It handles little-endian
packing and zeroes output padding bits. Keep collection scripts and experiment
management outside the decoder.

**Growth.** Start with one cluster per vertex and the supplied syndrome bits
as detector parities. A cluster grows exactly when its parity is odd and it
contains no terminal. Merge parity by XOR and terminal presence by OR. An even
cluster can become active again after merging with an odd cluster.

Use a disjoint-set union structure with iterative path compression and union
by size. Root state consists of size, parity, terminal presence, and frontier
edges. Record physical merge edges separately from DSU parents: these edges
form the forest used for correction.

For an edge crossing two clusters, let `w` be its weight, `g` its accumulated
growth, and `k` the number of active endpoint clusters. For `k > 0`, its
remaining completion time is `(w - g) / k`. Partial growth persists when a
cluster pauses or merges. Internal edges stop growing.

Use one `heapq` scheduler with lazy deletion and per-edge generation tokens.
Before changing an affected edge's rate, settle its accumulated growth at the
old rate and current time, then publish its new deadline. Repeated settlement
at the same time must add no growth. Publish earlier deadlines immediately
when a rate increases. Update affected frontiers; an unchanged rate does not
need a new deadline. A simple scan helper belongs only in the small-graph tests.

Process simultaneous completions as a batch:

1. Anchor the batch at the earliest valid completion time `t`, using tolerance
   `epsilon(t) = 1e-12 + 1e-12 * abs(t)`.
2. Collect and latch all valid completions through `t + epsilon(t)` before
   changing cluster activity. Treat these edges as completed at the batch time.
3. Process the collected edges in stable edge-ID order. Re-find both roots for
   each edge; skip an edge that has become internal, otherwise union its roots
   and record it in the forest. An intermediate even parity must not cancel
   another latched completion.
4. Update activity and affected deadlines. Collect and process newly published
   completions within the same anchored tolerance until none remain. Then
   advance time. Never widen the tolerance window transitively.

Process zero-cost closure before advancing to positive time. Finish growth
when no active cluster remains. If an odd cluster has no possible continuation
to another cluster or terminal, raise `InvalidSyndromeError`.

**Peeling and correctness.** Peel each forest tree iteratively, rooted at a
terminal when one exists, otherwise at a deterministic detector. Select parent
edges to cancel leaf detector parity and propagate the residual toward the
root. Terminal vertices impose no parity constraint. A detector root must
finish with zero residual parity. XOR the selected edges' observable masks to
produce the logical prediction.

For syndrome `s` and selected correction edges `c`, correctness requires
`H c = s` over GF(2), where `H` is detector-edge incidence. The returned
prediction must equal `L c`, where `L` contains the edge observable labels.
Verify both equations independently in tests. Logical predictions can differ
between valid corrections; equality with MWPM is not a correctness condition.

The essential tests cover weighted paths, paused clusters, high-degree tied
merges, cycle edges, boundaries, zero weights, invalid syndromes, iterative
traversal, and independent consecutive shots. The 1D integration fixture
exercises the complete circuit-to-DEM-to-graph-to-prediction path using this
same generic decoder.
