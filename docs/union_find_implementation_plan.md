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
  the design. Audit decomposed components and fail clearly on unsupported
  inputs. Preserve the exporter's weights and parallel-edge labels.
- [ ] Test direct graph construction, repeated/shifted DEM instructions,
  observable labels, zero-cost and boundary edges, and rejected inputs.

Acceptance: both direct graph construction and DEM import produce the same
documented graph representation. UF requires no circuit geometry or yoke tags.

**M2: weighted growth and peeling.**

- [ ] Implement iterative DSU find/union, cluster parity and terminal state,
  frontier updates, and the merge forest separate from DSU parents.
- [ ] Implement continuous weighted growth using a heap with generation
  tokens. Preserve partial growth, make settlement idempotent, and
  immediately publish deadlines when activity changes.
- [ ] Implement the design's complete tied-event batching and zero-cost
  closure. Re-check roots when processing each completed edge; record only
  actual merges. Continue same-time closure after publishing new events.
- [ ] Peel the forest iteratively, construct correction edge IDs, and XOR
  observable masks. Raise `InvalidSyndromeError` for unsatisfiable syndromes.
- [ ] Add the focused fixtures below and small realizable random syndromes
  generated from edge sets. Independently verify `H c = s` and `prediction = L c`.
  Cross-check the heap against the scan helper under the same tie rule.

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

Acceptance: the core returns syndrome-valid corrections and their observable
predictions on deterministic fixtures and realizable random inputs. High-degree
hub behavior follows the same rules as every other vertex.

**M3: usable interfaces and 1D integration.**

- [ ] Export `DecodingGraph`, `UnionFindDecoder`, and `InvalidSyndromeError`.
  Provide `decode(syndrome)` and `decode_batch(syndromes)` with the shapes
  specified in the design. Reuse the single-shot implementation for batches.
- [ ] Test input validation, input immutability, all-zero syndromes, empty
  batches, output dimensions, and state reset across successful and failed calls.
- [ ] Add and export a thin `SinterUnionFindDecoder` implementing compile-for-DEM and
  bit-packed batch decoding. Check little-endian packing, padding bits, and
  agreement with direct calls on the same inputs.
- [ ] Generate the initial 1D circuit using
  [the existing memory generator](../src/yoked/_yoked_memory_circuits.py), build
  its decomposed DEM, and decode a fixed-seed sample of 16 syndromes at the
  configuration above. Verify every correction and logical-mask reconstruction,
  including the yoke detector constraints. This is a correctness check.
- [ ] Document a short working example of direct graph decoding, DEM import,
  and use of the Sinter adapter under the name `uf_weighted`.

Acceptance: a caller can decode a supplied graph, and the repository's 1D
yoked circuit runs through sampling, graph import, UF correction, and logical
prediction using the public interfaces. The same immutable graph supports
repeated calls. Deliver the implementation for review at this point.

**Validation command after implementation.** Run from the repository root:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$TMPDIR/uf-pycache"
export MPLCONFIGDIR="$TMPDIR/uf-mpl"
.venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/decoders
```

Use `$TMPDIR` for temporary files and caches. Keep integration fixtures
in memory where possible.
