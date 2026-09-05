# Weighted Union Find comparison design for 1D yoked surface codes

Implement weighted Union Find (UF) on the complete joint detector graph used by
vanilla PyMatching. The initial operating point is **d=7, p=0.003, rounds=28,
CZ gates, SI1000 noise**. Compare the **1D yoked code with six patches and two
yokes** against an **unyoked control with four patches**. Both encode four
logical qubits. The two yokes check the logical X and Z parities of the same
patch group; `yokes=2` denotes this 1D code. The single-Y-yoke variant and 2D
Squareberg code are outside the current implementation and benchmark scope. The
[implementation plan](union_find_implementation_plan.md) defines the work order,
accuracy gate, and native benchmark. Decoder implementation remains pending.

**Baseline and scope.** [step2_collect](../step2_collect) selects `pymatching`
through Sinter. [The memory generator](../src/yoked/_yoked_memory_circuits.py)
incorporates local checks and yoke parity checks into the same detector record.
Joint decoding describes the use of this complete graph. Correlated decoding
additionally exploits relations between components of decomposed errors.

| Decoder | Graph | Weights | Decomposition correlations |
| --- | --- | --- | --- |
| `pymatching` | Complete joint graph | Log likelihood ratio | Ignored |
| `uf_weighted` | Same graph | Same exported log likelihood ratio | Ignored |
| `pymatching-correlated` | Complete joint graph | Reweighted using correlations | Used |

The primary comparison is `uf_weighted` versus `pymatching`. The installed
Stim/Sinter 1.16.0 and PyMatching 2.4.0 also provide the correlated baseline.
The [README](../README.md) identifies the paper's correlated implementation as
internal; historical `sparse_blossom_correlated` results are not assumed to be
identical to the current public implementation. Record actual versions.

**Measured initial graphs.** A fresh probe at the selected point used vanilla
PyMatching, 10,000 circuit samples per arm, seed 42, and the versions above.
Parameters and sample/DEM digests are in the
[baseline evidence](union_find_baseline_d7_p003.json). The table includes only
the two arms in scope; the manifest retains the original single-yoke probe row
as historical evidence and marks it excluded from the active comparison.

| Yokes | Patches | Detectors | Observables | Graph edges | Yoke degrees | Yoke fire fractions | MWPM block failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 | 5,569 | 8 | 27,224 | — | — | 4,794 / 10,000 |
| 2 | 6 | 8,354 | 12 | 40,836 | 696, 696 | 50.11%, 50.60% | 3,424 / 10,000 |

These are graph/baseline measurements, not UF results or decoder timing results.
The review's earlier d=3/d=5 measurements at p=0.001 similarly found frequent
fired yokes. Half-rate firing is an observation at these points, not a universal
property of yoke detectors. The new point has substantial block failure rates;
report absolute paired differences as well as ratios and avoid low-error scaling
claims. The paired comparison still requires enough discordant outcomes.

**The main hypothesis is growth from active yokes.** A fired yoke starts as an
odd singleton with a large frontier and can grow toward several patches at once.
The design freezes internal edges and peels a merge forest, so growth order and
tie decisions determine most correction choices. A performance assessment must
therefore examine whether this behavior introduces a large accuracy penalty,
before optimizing or running the full campaign.

A yoke is a stabilizer formed from patch-level logical operators, represented by
an additional constrained detector. Its graph edges connect relevant patch-boundary
detectors through decomposed fault mechanisms. Every local detector is already in
the full graph; only detectors reached through completed weighted edges join a
particular cluster. Reaching a yoke never absorbs all neighboring checks or
removes the cluster's parity constraint. This uses the joint graph construction
in the paper's [benchmarking discussion](https://www.nature.com/articles/s41467-025-59714-1).

**Compile one shared graph representation.** Construct the DEM explicitly with
`decompose_errors=True, approximate_disjoint_errors=True`, matching the runnable
baseline. Fail if this representation cannot be constructed. Before importing,
audit `dem.flattened()`, reduce detector/observable targets modulo two, and check
each component separated by `^`. The components remain correlated physical
faults even though this first UF decoder uses their marginal graph.

Export `pymatching.Matching.from_detector_error_model(dem).edges()` to reuse the
baseline's weights and observable labels. Preserve import order, independent
parallel-edge merging, and the first edge's observable mask. Audit conflicts;
never merge such masks by XOR. This imports the graph without using a matching
solve during UF decoding. Release the temporary Matching object afterward.
The [PyMatching import API](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.Matching.from_detector_error_model)
documents its decomposition and parallel-edge behavior.

Store immutable endpoints, CSR adjacency, float64 weights, and observable masks.
Use counts from the DEM, including the zero-yoke circuit's isolated final detector
and observable columns absent from all edges. Coordinates are diagnostic only.
Each one-detector edge gets a separate unconstrained virtual terminal. All real
detectors, including yokes, keep their syndrome constraints. The checked yokes
have detector-to-detector edges rather than virtual-boundary edges.

Initially accept finite nonnegative weights. Preserve zero-cost edges and their
labels during initial closure. Reject unsupported nonzero-probability hyperedges,
nontrivial observable-only components, and negative/nonfinite weights explicitly.
The Python representation can use arbitrary-width integers; the first native
kernel uses a `uint64_t` mask for the pilot's 8 or 12 observables, with an explicit
error above 64 until broader support is implemented.

**Growth and correction contract.** Given syndrome `s`, choose edges `c` with
`H*c=s` over GF(2), then return the logical prediction `L*c`. UF may choose a
different valid correction from MWPM. Independently verify both equations on
every Python pilot shot; syndrome-invalid output is a bug, not a logical failure.

Each DSU root stores size, syndrome parity, terminal presence, and frontier state.
A root grows when odd and without a virtual terminal. Merge parity by XOR and
terminal presence by OR. A paused even cluster can become active after merging
with an odd cluster. Record physical forest edges separately from path-compressed
DSU parents. Diagnostic `contains_yoke` flags propagate by OR and do not change
the decoding policy.

Use continuous, simultaneous weighted growth. Edge length is its exported
`w_e = log((1-p_e)/p_e)`. For a crossing edge with accumulated growth `g_e` and
`k_e` active endpoints in different clusters, its remaining completion time is
`(w_e-g_e)/k_e` when `k_e>0`. Advance to the earliest completion, process tied
completions, and recompute activity. Retain partial growth across merges.

At each completed edge, re-find both endpoint roots before processing. If already
in the same component, skip the edge without recording it; otherwise union and
record it. Settle all events at the current time before advancing. A closed odd
component with no continuation raises `InvalidSyndromeError`. No silent discard,
zero prediction, or fallback matching is permitted. This follows UF's usual
growth-and-peeling structure, with the specified weighted schedule.
[Original UF](https://arxiv.org/abs/1709.06218),
[edge-weighted UF](https://arxiv.org/abs/2004.04693).

The run implementation uses a heap of completion times with generation tokens
and lazy deletion. Settle affected edges using their old activity rates before
publishing new deadlines. Rate increases must immediately publish earlier
possible completions. A stale-key check only when popping is insufficient if it
can hide an earlier event. Reschedule changed frontiers; do not rescan a stable
active yoke frontier for unrelated events. Count rescheduling and stale work.
Keep an explicit scan implementation as a small-graph test oracle.

Peel each merge tree toward a virtual terminal when present, otherwise a
stable detector root. Select parent edges to cancel leaf detector syndrome and
XOR their observable masks. Non-root terminal leaves impose no constraint.
Only a detector root must finish with zero residual parity. Peeling selects from
the grown forest; cluster membership alone does not select a correction edge.

**Required tests and early gate.** The implementation plan specifies numeric
fixtures for an unfired yoke between defects, a fired yoke with competing defects
at unequal distances in different patches, and a fired yoke meeting an even
cluster. Include a three-defect equal-weight triangle: the cycle-closing edge
must be skipped, and the closed odd component must fail unless a valid terminal
continuation exists. Inject a fired isolated detector into the zero-yoke circuit.
Retain zero-cost and terminal-root tests.

After growth and peeling work, run 2,000 shared shots on the initial two-yoke
arm with the Python heap backend and verify every correction. A UF/MWPM failure
ratio above about 2 triggers investigation before expanding the implementation;
record uncertainty and mark borderline evidence inconclusive. A bounded extra
pilot or a minimal native port can resolve uncertainty without a long Python
campaign. Use fresh samples for the later main experiment.

**Native implementation and interfaces.** Select C++17, pybind11 3.0.4, and
CMake/Ninja now. Build a binding smoke test with the existing Python at the start,
then implement the native kernel immediately after the accuracy gate. Keep
compiled dependencies and artifacts under `$TMPDIR`. Use Release mode without
fast-math and record the compiler/build configuration.

The native kernel owns the packed batch loop, reusable workspace, heap, DSU,
and peeling. A top-level `_yoked_uf_native` extension loaded from an explicit
`$TMPDIR` path avoids source-package shadowing. A thin `UnionFindDecoder` Sinter
facade implements `compile_decoder_for_dem` and
`decode_shots_bit_packed`; register it as `uf_weighted`. Input and output are
little-endian packed `uint8` arrays with the exact detector/observable widths.
Keep input immutable, output padding zero, and decoder state isolated.
[Sinter interface](https://github.com/quantumlib/Stim/blob/main/glue/sample/src/sinter/_decoding/_decoding_decoder_class.py).

Backend validation requires valid corrections and agreement on tie-free cases.
Report prediction mismatch rates on tied/random cases and trace tolerance-induced
differences instead of requiring universal exact equality. Unexplained mismatches
remain defects. PyMatching also quantizes weights internally, so near-tie
prediction differences need not indicate a broken graph or decoder.

**Accuracy and latency measurements.** Use the 1D yoked arm and unyoked control
at the fixed initial point, with **100,000 shared shots per arm** and native UF
for the main comparison.
Sample each circuit batch once, including actual observables, and feed identical
inputs to both decoders. Store seed, batch partition, actual shot totals, hashes,
versions, source/backend identity, policy/tolerance settings, and raw counts in
a manifest and tables. A simple fixed-seed paired tool is sufficient initially.

Primary accuracy outputs are block failure probabilities, UF/MWPM ratios
`R_0` (unyoked control) and `R_2` (1D yoked code), absolute paired differences,
four paired outcome counts, discordant counts, and prediction disagreements,
with uncertainty and zero-count handling.
The fixed shot budget is not itself a guarantee of power. No baseline failures
means the ratio is undefined or unbounded, rather than evidence of zero penalty.

Every failed shot gets its yoke firing mask and UF cluster-yoke tags. Keep
per-component flags and a small set of detailed traces. Exclude untouched
zero-syndrome yoke singletons from the cluster-contact summary. Multiple
components can contribute to a shot, so these tags describe association rather
than a unique causal attribution. Stratify work and failures by yoke firing and
cluster contact to test the central hypothesis.

Count edge/frontier visits, heap operations, DSU operations, maximum sizes, and
yoke-related activity from the first growth implementation. Record instrumentation
settings and overhead. Python is limited to the gate and at most 256 additional
profiled shots per arm; it is not the backend for the full timing experiment.

For native latency, warm up precompiled decoders and time calls on pre-sampled
inputs. Include unpacking, resets, growth, peeling, and packing; exclude sampling,
compilation, syndrome verification, diagnostic serialization, and plotting.
Report batch-one median/p95/p99 and fixed-batch throughput separately. Measure
normal-width all-zero-syndrome calls as an interface floor including input/reset/
output work, report it beside real-shot times, and do not subtract it. Fix thread
counts, alternate decoder order, and record CPU/build details. These are software
latencies at d=7, not hardware timing estimates.

**Deferred extensions.** Large-sweep collection and Sinter resume/pickling tests,
manifest merge/compatibility frameworks, isolated peak-RSS measurement, larger
observable masks, and extra 1D operating points follow the first comparison.
Later noise sweeps can use `p in {0.001,0.002,0.003}`. Broader
collection must eventually version task metadata to prevent incompatible resume.

The existing [GapWorkHandler](../src/yoked/gap/_gap_worker_handler.py) subtracts
weights of two minimum matchings. UF does not provide those minima. Any future
UF-based confidence score requires separate definition and calibration before
it can replace complementary gaps or drive an inner-UF/outer-MWPM decoder.
