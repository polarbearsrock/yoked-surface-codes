# Weighted Union Find decoder design

Implement a complete weighted Union Find (UF) decoder: receive a graph and a
syndrome, grow and merge clusters, peel a correction, and return predicted
logical observables. The core operates on graph connectivity, weights, and
observable labels. The same algorithm applies to supported graphs from 1D or
2D code generators; geometry and patch metadata are not decoder inputs.

The first integration target is the repository's 1D yoked surface code. The
[implementation plan](union_find_implementation_plan.md) defines the delivery
steps and correctness checks. The [decoder package](../src/yoked/decoders/__init__.py)
is implemented, with [working examples](union_find_usage.md). Benchmarking
will be planned after implementation review.

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
error components, splitting at `^` and ignoring zero-probability instructions.
Reject components with repeated detector targets before export, with more than
two detectors, or with observables but no detectors. Reduce observable targets
modulo two when auditing labels. Preserve zero-cost edges; reject negative or
nonfinite weights explicitly.

The adapter exports
`pymatching.Matching.from_detector_error_model(dem).edges()` to obtain weights
and observable labels. This reuses graph construction without running a
matching solve. Preserve the importer's parallel-edge result, including its
retained observable labels; do not XOR labels when combining parallel edges.
Preserve export order, which follows each edge's first occurrence in the
flattened DEM in the supported importer. This order defines edge IDs and ties.
Reconcile exported endpoint keys, retained labels, and weights against the
audit. Combine parallel-component probabilities as `p*(1-q) + (1-p)*q` and
compare the resulting log-likelihood weights with numerical tolerance. Raise
on a mismatch; equal endpoint sets alone cannot detect a dropped parallel
contribution. The initial 1D fixture also checks that parallel labels agree.
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

A thin `SinterUnionFindDecoder` adapter is a stateless top-level class, with
graph construction and core compilation inside `compile_decoder_for_dem` so
the adapter can be passed to Sinter workers. The compiled decoder implements
the bit-packed batch interface, with little-endian packing and zero output
padding bits. Keep collection scripts and experiment management outside the
decoder.

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
Store each edge's current rate and last-settled time. Before changing its rate,
settle accumulated growth using that stored rate and the current time.
Repeated settlement at the same time must add no growth. After processing each
collected group of tied edges, publish changed deadlines before advancing time;
an increased rate must not wait for its obsolete later deadline to be popped.
A simple scan helper belongs only in the small-graph tests.

Process simultaneous completions as a batch:

1. Anchor the batch at the earliest valid completion time `t`, using tolerance
   `epsilon(t) = 1e-12 + 1e-12 * abs(t)`.
2. Collect and latch all valid completions through `t + epsilon(t)` before
   changing cluster activity. Treat these edges as completed at the batch time.
3. Process the collected edges in stable edge-ID order. Re-find both roots for
   each edge; skip an edge that has become internal, otherwise union its roots
   and record it in the forest. An intermediate even parity must not cancel
   another latched completion.
4. After all unions in this collected group, recompute activity. Visit each
   affected edge once, settle it from its stored rate, invalidate it if now
   internal, and publish a deadline if its rate changed. An unchanged rate
   needs no new deadline. Repeat collection and processing for newly published
   completions within the same anchored tolerance until none remain. These
   updates occur after each collected group, not after individual unions.
   Only then advance time; never widen the tolerance window transitively.

Process zero-cost closure before advancing to positive time. Finish growth
when no active cluster remains. After same-time closure, if an active root
remains but the heap has no valid entry after stale entries are discarded,
raise `InvalidSyndromeError`. A crossing edge incident to an active cluster
must have a pending completion, so no reachability search is needed.

**Peeling and correctness.** Peel each forest tree iteratively, rooted at a
terminal when one exists, choosing the smallest terminal ID; otherwise choose
the smallest detector ID. Select parent edges to cancel leaf detector parity
and propagate the residual toward the root. Each non-root terminal is a leaf
because it belongs to just one boundary edge; leave its edge unselected.
Terminal vertices impose no parity constraint. A detector root must finish
with zero residual parity. XOR the selected edges' observable masks to produce
the logical prediction.

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
