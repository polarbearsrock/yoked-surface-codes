# Weighted Union Find decoder implementation plan

**Goal:** implement a usable, graph-based weighted Union Find decoder, end to
end. Given a decoding graph and syndrome, it grows clusters, peels a valid
correction, and returns logical-observable predictions. Follow the
[decoder design](union_find_design.md).

The core takes graph data, so it has no 1D/2D switch or yoke-specific growth
policy. Start integration with the 1D yoked surface code at **d=7, p=0.003,
rounds=28, six patches, two yokes, CZ gates, SI1000 noise**. These parameters
belong to the circuit fixture, not the decoder implementation.

Status: implementation pending. Deliver the decoder and correctness tests for
review first. Benchmarking is a separate task after that review.

**Keep the implementation small.** Use one Python runtime implementation and
the repository's existing dependencies. Place tests beside their modules:

```text
src/yoked/decoders/
    __init__.py
    _graph.py
    _graph_test.py
    _union_find.py
    _union_find_test.py
    _sinter.py
    _sinter_test.py
    _yoked_decoding_test.py
```

The core needs graph connectivity, weights, observable labels, and per-shot UF
state. Keep a small scan scheduler inside the tests as an oracle for the heap.

**M1: graph input and DEM adapter.**

- [ ] Define `DecodingGraph` with explicit detector/observable counts, ordered
  edges, observable masks, and adjacency lists. Support direct construction
  from graph data; preserve isolated detectors and unused observable columns.
- [ ] Validate endpoints, weights, and masks. Internally assign a distinct
  unconstrained terminal to each boundary edge. Keep every detector constrained.
- [ ] Implement `DecodingGraph.from_dem` using the graph export described in
  the design. Reject nonzero-probability components with more than two
  detectors, observables but no detectors, or repeated detector targets within
  a component. Preserve export order for edge IDs and deterministic tie-breaking.
- [ ] Reconcile the audit with the export: compare endpoint keys, retained
  observable masks, and weights from independently combined parallel-component
  probabilities, allowing numerical tolerance for weights. Keep the first
  parallel component's label and raise on any mismatch. Endpoint sets alone
  cannot detect a dropped contribution to an existing parallel edge.
- [ ] Test direct graph construction, repeated/shifted DEM instructions,
  export order, parallel weights and labels, zero-cost and boundary edges.
  Include the three rejected component shapes above, including a repeated-target
  component whose normalized endpoints coincide with an otherwise valid edge.

Acceptance: both direct graph construction and DEM import produce the same
documented graph representation. UF requires no circuit geometry or yoke tags.

**M2: weighted growth and peeling.**

- [ ] Implement iterative DSU find/union, cluster parity and terminal state,
  frontier updates, and the merge forest separate from DSU parents.
- [ ] Implement continuous weighted growth using a heap with generation
  tokens. Store each edge's current rate and last-settled time. Preserve partial
  growth and make repeated settlement at one timestamp idempotent.
- [ ] Implement the design's complete tied-event batching and zero-cost
  closure. Re-check roots when processing each completed edge; record only
  actual merges. After all unions in each collected group, settle affected
  edges using their stored rates, invalidate internal edges, and publish
  deadlines for changed rates once per affected edge. Continue collecting
  newly published same-time events until closure is complete.
- [ ] Peel the forest iteratively, construct correction edge IDs, and XOR
  observable masks. Choose the smallest terminal ID as root when present,
  otherwise the smallest detector ID. Non-root terminals are leaves whose
  edges are not selected. After same-time closure, raise `InvalidSyndromeError`
  if active roots remain and the heap has no valid entry after stale entries
  are discarded.
- [ ] Add the focused fixtures below and small realizable random syndromes
  generated from edge sets. Independently verify `H c = s` and `prediction = L c`.
  For the random oracle cases, draw weights from `{1, 2, 3}` to exercise ties.
  Require identical forest edge IDs and predictions from heap and scan under
  the same edge order, tie rule, and peeling-root rule.

| Fixture | Required behavior |
| --- | --- |
| Competing weighted paths | Earlier weighted completions determine the merge; reaching a hub does not automatically absorb its neighbors. |
| Odd cluster meets an even cluster | The merged cluster stays odd and continues growing with its accumulated edge growth preserved. |
| Equal-weight hub with fired neighbors | With hub bit `s_hub` and `k` fired neighbors, all completed spokes join in one batch; parity is `s_hub XOR (k mod 2)`. Cover fired/unfired hubs and `k=32,33`, with a slower boundary continuation for odd cases. |
| Hub neighbors have fired partners | When partner edges also complete in the batch, include all `m` additional fired partners; parity is `s_hub XOR ((k+m) mod 2)`. Intermediate even parity cannot truncate the batch. |
| Neighbors pair before reaching an unfired hub | The even neighbor clusters pause and the hub can remain untouched. |
| Equal-weight triangle | Skip the third edge after it becomes internal. Three defects without a terminal are invalid; a slower terminal path makes the companion case valid. |
| Zero costs and tied terminal contacts | Complete zero-time closure and peel a valid correction, including trees with multiple terminals. |
| Isolated fired detector | Raise `InvalidSyndromeError`. |
| Both endpoint activities change | Repeated settlement at one timestamp adds growth only once; newly published same-time completions are processed before time advances. |
| Long chain | Find and peeling work beyond Python's recursion limit. |

Acceptance: the core passes the specified growth and pause behavior, agrees
with the scan oracle on the small integer-weight cases, and returns valid
corrections with matching observable reconstruction. High-degree hubs follow
the same rules as every other vertex.

**M3: usable interfaces and 1D integration.**

- [ ] Export `DecodingGraph`, `UnionFindDecoder`, and `InvalidSyndromeError`.
  Provide `decode(syndrome)` and `decode_batch(syndromes)` with the shapes
  specified in the design. Reuse the single-shot implementation for batches.
- [ ] Test input validation, input immutability, all-zero syndromes, empty
  batches, output dimensions, and state reset across successful and failed calls.
- [ ] Add and export a thin `SinterUnionFindDecoder` as a stateless top-level
  class. Build its graph and core decoder only in `compile_decoder_for_dem`.
  Check bit-packed batch decoding, little-endian packing, padding bits, and
  agreement with direct calls on the same inputs.
- [ ] Use [the existing memory generator](../src/yoked/_yoked_memory_circuits.py)
  with keyword arguments to build the initial 1D circuit and its decomposed DEM.
  Decode 16 shots with seed 42 at the d=7 configuration above. Verify every
  correction and logical-mask reconstruction, including the yoke detector
  constraints. Assert that this fixture has no parallel components with
  conflicting observable labels. This is a correctness check.
- [ ] Add a smaller d=3, rounds=12 version with the same noise strength, six
  patches, two yokes, and 16 shots for routine tests. Measure the d=7 test's
  runtime once; if it exceeds about 30 seconds, make only that test opt-in
  using `YOKED_UF_RUN_D7=1` and `pytest.skip`. The d=7 check must still run and
  pass before implementation is considered complete.
- [ ] Document a short working example of direct graph decoding, DEM import,
  and use of the Sinter adapter under the name `uf_weighted`.

Acceptance: a caller can decode a supplied graph, and the repository's 1D
yoked circuit runs through sampling, graph import, UF correction, and logical
prediction using the public interfaces. The same immutable graph supports
repeated calls, and the d=7 correctness check has passed. Deliver the
implementation for review at this point.

**Validation command after implementation.** Run from the repository root.
This includes the required d=7 check even if it is separated from routine tests:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$TMPDIR/uf-pycache"
export MPLCONFIGDIR="$TMPDIR/uf-mpl"
YOKED_UF_RUN_D7=1 .venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/decoders
```

Use `$TMPDIR` for temporary files and caches. Keep integration fixtures
in memory where possible.
