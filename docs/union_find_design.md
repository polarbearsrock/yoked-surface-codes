# Weighted Union Find comparison design for 1D yoked surface codes

Implement weighted Union Find (UF) on the complete joint detector graph used by
vanilla PyMatching. The initial operating point is **d=7, p=0.003, rounds=28,
CZ gates, SI1000 noise**. Compare **UF against joint MWPM on the 1D yoked code
with six patches and two yokes**, encoding four logical qubits. The two yokes
check the logical X and Z parities of the same patch group; `yokes=2` denotes
this 1D code. The single-Y-yoke variant and 2D
Squareberg code are outside the current implementation and benchmark scope. The
[implementation plan](union_find_implementation_plan.md) defines the work order,
paired pilot, and native benchmark. Decoder implementation remains pending.

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

**Measured initial graph.** The probe at the selected point used vanilla
PyMatching, 10,000 circuit samples, seed 42, and the versions above.
Parameters and sample/DEM digests are in the
[baseline evidence](union_find_baseline_d7_p003.json). The table includes only
the six-patch, two-yoke circuit. Other original probe rows remain historical
evidence in the manifest and are excluded from the active comparison.

| Yokes | Patches | Detectors | Observables | Graph edges | Yoke degrees | Yoke fire fractions | MWPM block failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 6 | 8,354 | 12 | 40,836 | 696, 696 | 50.11%, 50.60% | 3,424 / 10,000 |

These are graph/baseline measurements, not UF results. Half-rate yoke firing is
expected in a regime where raw patch logical parities are sufficiently noisy.
For independent patches with raw flip probabilities `q_i`, a yoke fires with
probability `(1 - product_i(1 - 2*q_i))/2`. The product becomes small in the
long/noisy memories considered here. This is a statement about raw parity,
distinct from the logical failure probability after decoding; it does not assert
half-rate firing at every possible operating point.

The latest review supplies these additional machine measurements. They motivate
the experiment; the UF event summaries must be measured again using the actual
implemented scheduler and its documented batch definition.

| Circuit | Mean defects/shot | MWPM batch-one median / p99 | All-zero floor |
| --- | --- | --- | --- |
| 1D yoked | 767 | 712 / 1,553 microseconds | 4.4 microseconds |

Each yoke has eight distinct edge weights, including 324 of its 696 edges at
the minimum weight `3.544870902667`. A local graph audit confirms these values.
The review reports about 33 fired minimum-weight neighbors per fired yoke;
the first tied merge has median 33, p90 41, and maximum 54 ordinary detectors,
touches all six patches in 95% of shots, and leaves the cluster odd about half
the time. Treat these as review-reported event statistics, not existing UF
decoder results. Record unfired-yoke events too, including an absent first
event on syndromes such as the all-zero input.

**The main hypothesis is growth through large tied yoke events.** A fired yoke
starts as an odd singleton. Many equal-weight edges can complete together and
join defects across patches in one event. Half of these large clusters may
remain active, and later activity changes reschedule a large frontier. Internal
edges are frozen and correction uses a merge forest, so growth and tie order
determine most correction choices. Compare UF's accuracy directly with joint
MWPM and use cluster diagnostics to investigate correction choices. Work
counters expose the cost of rescheduling the yoke cluster.

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
Use counts from the DEM, preserving isolated detectors and observable columns
absent from all edges. Exercise these cases with synthetic fixtures.
Coordinates are diagnostic only.
Each one-detector edge gets a separate unconstrained virtual terminal. All real
detectors, including yokes, keep their syndrome constraints. The checked yokes
have detector-to-detector edges rather than virtual-boundary edges.

For the 1D circuit, audit every edge `e` using observable sets
`X={0,2,...,10}` and `Z={1,3,...,11}`: the parity of `L_X(e)` must equal
its incidence to the X yoke, and likewise for Z. Also verify that each labelled
edge joins its matching yoke to the patch named by its sole observable; ordinary
detector-to-detector edges stay within one patch and check sector. Obtain patch
and check-sector metadata from the known generator and validate it against the
graph. All observable-carrying edges in this circuit are yoke edges.
The 1D graph still has 1,392 unlabelled boundary edges away from its yokes;
terminal contact must remain part of growth
and diagnostics.

Initially accept finite nonnegative weights. Preserve zero-cost edges and their
labels during initial closure. Reject unsupported nonzero-probability hyperedges,
nontrivial observable-only components, and negative/nonfinite weights explicitly.
The Python representation can use arbitrary-width integers; the first native
kernel uses a `uint64_t` mask for the pilot's 12 observables, with an explicit
error above 64 until broader support is implemented.

**Growth and correction contract.** Given syndrome `s`, choose edges `c` with
`H*c=s` over GF(2), then return the logical prediction `L*c`. UF may choose a
different valid correction from MWPM. Independently verify both equations on
every Python pilot shot; syndrome-invalid output is a bug, not a logical failure.

Each DSU root stores size, syndrome parity, terminal presence, and frontier state.
A root grows when odd and without a virtual terminal. Merge parity by XOR and
terminal presence by OR. A paused even cluster can become active after merging
with an odd cluster. Record physical forest edges separately from path-compressed
DSU parents. Use iterative find with path compression and iterative peeling;
large clusters must not depend on Python's recursion limit. Track yoke IDs and
per-patch, per-sector syndrome parities as diagnostic metadata.

Use continuous, simultaneous weighted growth. Edge length is its exported
`w_e = log((1-p_e)/p_e)`. For a crossing edge with accumulated growth `g_e` and
`k_e` active endpoints in different clusters, its remaining completion time is
`(w_e-g_e)/k_e` when `k_e>0`. Advance to the earliest completion, process tied
completions, and recompute activity. Retain partial growth across merges.

Define a batch at the earliest valid completion time `t`. Collect all completions
through `t + epsilon(t)`, with defaults `epsilon(t)=1e-12 + 1e-12*abs(t)`
recorded in the policy manifest, and settle their growth using the rates before
the batch. Latch these completed edges:
an intermediate parity change must not cancel another already-completed edge.
Process each closure wave in stable edge-ID order, re-finding both roots for every edge;
skip internal edges without recording them, otherwise union and record once.
Recompute activity after these unions, publish affected deadlines, and collect
again until no valid entry lies within the same batch tolerance. Do not extend
the tolerance window transitively. Only then advance time. A closed odd
component with no continuation raises `InvalidSyndromeError`. No silent discard,
zero prediction, or fallback matching is permitted. This follows UF's usual
growth-and-peeling structure, with the specified weighted schedule.
[Original UF](https://arxiv.org/abs/1709.06218),
[edge-weighted UF](https://arxiv.org/abs/2004.04693).

The run implementation uses a heap of completion times with generation tokens
and lazy deletion. Settling an edge at time `t` is idempotent: repeated visits
at that time add no growth, including when both endpoints change activity.
Settle affected edges using their old activity rates before publishing new
deadlines. Rate increases must immediately publish earlier possible completions.
A stale-key check only when popping is insufficient if it
can hide an earlier event. Reschedule changed frontiers; do not rescan a stable
active yoke frontier for unrelated events. Coalesce updates within a batch where
rates have not changed between visits, but account for every real activity
change. Count rescheduled edges, rescheduling passes, and stale work separately,
including the part attributable to clusters containing each yoke.
Keep an explicit scan implementation as a small-graph test oracle.

Peel each merge tree toward a virtual terminal when present, otherwise a
stable detector root. Select parent edges to cancel leaf detector syndrome and
XOR their observable masks. Non-root terminal leaves impose no constraint.
Only a detector root must finish with zero residual parity. Peeling selects from
the grown forest; cluster membership alone does not select a correction edge.

**Required tests and yoke diagnostics.** Retain the unequal-distance, even-cluster,
triangle, isolated-detector, zero-cost, and terminal fixtures in the plan. Add
equal-weight yoke stars spanning multiple patches, with odd and even numbers
of fired neighbors, unfired yokes, and simultaneous neighbor-partner completions.
For `k` fired neighbors and no extra fired members, post-batch parity is
`s_yoke XOR (k mod 2)`. If `m` additional fired partners join, it is
`s_yoke XOR ((k+m) mod 2)`. Count all unique members; union order cannot change
membership or parity in these fixtures, though cycles can change the forest.

Replace a boolean yoke-contact failure tag with one record per yoke per shot:
input bit; first union batch time and post-closure size (ordinary detectors and
fired ordinary members separately); parity and patch bitset at that time; final
ordinary-member and fired-member counts, patch bitset, terminal presence, and
per-patch member-defect parity by X/Z sector.
Record an absent first event explicitly. Save compact records on successful and
failed shots for denominators, plus failed-observable masks and selected traces.
Keep stable yoke IDs if roots change or yoke-containing components ever merge.

For the final component `C_b` containing yoke `b`, let `D_i,b` be the XOR of
input defects among its ordinary members in patch `i`, sector `b`. Let `B_i,b`
be the parity of selected unlabelled boundary edges from those members.
The structural audit and `H*c=s` imply
`prediction_i,b = D_i,b XOR B_i,b`, since selected internal edges cancel.
When the component has no terminal, `B_i,b=0`: member-defect parity alone
predicts that patch's observable. Always cross-check the direct parity of
selected yoke edges, `L*c`, and this identity with its terminal term. Also check
`XOR_i prediction_i,X = s_X_yoke` and its Z counterpart. Cluster tags describe
association, not unique causation of a shot failure.

**Native implementation and interfaces.** Select C++17, pybind11 3.0.4, and
CMake/Ninja, using GCC toolset 14 at
`/opt/rh/gcc-toolset-14/root/usr/bin/{gcc,g++}` (locally verified as 14.2.1).
The Python 3.14.5 venv has headers and the GIL enabled, but no pip. Install the
pinned optional requirements with `uv pip install --python .venv/bin/python -r
requirements-uf-native.txt`; use `$TMPDIR` for uv and compiler caches. Build and
import a binding smoke test at M1, then port the decoder after the correctness
checks and paired pilot. Use Release mode without fast-math and record
compiler/build settings.
[pybind11 compatibility](https://pybind11.readthedocs.io/en/stable/changelog.html),
[uv installation](https://docs.astral.sh/uv/pip/packages/).

Keep intermediates and the built extension under `$TMPDIR`, following the
workspace instruction that build artifacts belong there. Use an explicit,
stable extension path on `PYTHONPATH` and record its digest. A repo-local build
directory is unnecessary for this plan.

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
Use `pytest.importorskip` for optional native tests when the extension is absent;
the explicit native validation run must first require a successful import, so
missing builds cannot turn required native checks into a passing skipped suite.

**Accuracy and latency measurements.** Use the six-patch, two-yoke 1D circuit
at the fixed initial point, with **100,000 shared shots** and native UF for the
main comparison with joint MWPM.
Sample each circuit batch once, including actual observables, and feed identical
inputs to both decoders. Store seed, batch partition, actual shot totals, hashes,
versions, source/backend identity, policy/tolerance settings, and raw counts in
a manifest and tables. A simple fixed-seed paired tool is sufficient initially.

For decoder `D`, retain the block failure probability `P_D` and
report an effective per-patch-round rate using the exact [Sinter conversion](https://github.com/quantumlib/Stim/blob/main/glue/sample/src/sinter/_probability_util.py):

```python
q_D = sinter.shot_error_rate_to_piece_error_rate(
    P_D, pieces=patches * rounds, values=2 * (patches - yokes))
```

This matches [step3_plot](../step3_plot). `values` counts twice the encoded
patches, not all tracked observable columns. Record the numeric arguments:

| Circuit | Pieces | Values | MWPM block probability | MWPM effective rate / patch-round |
| --- | --- | --- | --- | --- |
| 1D yoked, six patches | 168 | 8 | 0.3424 | 0.002560290509 |

This rate is derived from the committed 10,000-shot probe. The review's
earlier 100,000-shot value `2.48e-3` is a separate reported estimate in the same
unit; do not substitute it for this manifest's counts. This is an equivalent
independent-piece normalization, not a direct measurement of independent patch
failures. Preserve raw block rates and paired differences. If an observed rate
or its bootstrap interval reaches the independent-value saturation
`P=1-2**(-values)`, flag the converted ratio as unresolved. Report the raw rates
and the number of saturated replicates instead of forcing finite penalty bounds.

Report the direct UF/MWPM ratios `Q = q_UF / q_MWPM` in per-patch-round units
and `R = P_UF / P_MWPM` in block units, alongside absolute rates and differences
in both units. Also report all four paired outcome counts, discordance counts,
and prediction disagreements. The objective is to measure UF's accuracy and
speed relative to joint MWPM on this circuit.

Use 10,000 paired bootstrap replicates with a recorded seed and 95% percentile
intervals. Resample whole `(UF_failure, MWPM_failure)` shot pairs; equivalently
draw multinomial replicates of their four joint counts.
With count indices `(UF_failure, MWPM_failure)`, the block-rate difference is
`(n_10-n_01)/N`, and the individual rates are `(n_10+n_11)/N` and
`(n_01+n_11)/N`.
Recompute the piece conversions, ratios, and differences in each replicate.
Preserve the pairing between decoders. These are approximate bootstrap
intervals. Retain and report zero-denominator or saturated replicates; do not drop them or add
pseudocounts to force finite bounds. Zero observed failures/discordances can
make the empirical bootstrap degenerate: report count-based one-sided bounds
and mark ratio evidence unresolved where needed. Shot count alone does
not guarantee power.

**Paired pilot after M3.** First profile 50 fully checked, instrumented Python
shots on the same 1D circuit. Use elapsed time only to choose a pilot size:
`N=min(2000, floor(300 / (1.25 * t)))`, where `t` is mean seconds per checked
paired calibration shot. Record the 300-second budget and 25% timing margin.
Calibration samples are separate from pilot accuracy samples. Freeze N before
decoding them, and record actual completed shots if the runtime cap is reached.
Keep a short or imprecise pilot labelled preliminary and proceed to native work
once correction-validity and behavioral checks pass; do not extend Python runs
just to obtain a narrow accuracy interval. Reserve fresh seeds for M6.

The pilot checks correctness, estimates runtime, and provides an early direct
UF/MWPM accuracy comparison. There is no numerical accuracy threshold for
continuing: worse UF accuracy is still a result of this experiment. Fix invalid
corrections and unexplained implementation mismatches before the main run.
Report rescheduled edges/passes per shot, yoke-cluster activity toggles, and heap
work so the native port is informed by the dominant cost.

Count edge/frontier visits, heap operations, DSU operations, maximum sizes, and
yoke-related activity from the first growth implementation. Record instrumentation
settings and overhead. Python is limited to the pilot and at most 256 additional
profiled shots, including the 50-shot calibration; it is not the backend
for the full timing experiment. Validate every Python correction. Verify the
100,000-shot native accuracy run in native code or by sparse-matrix operations
per batch, without a Python loop over shots. Time verification separately.

For native latency, warm up precompiled decoders and time calls on pre-sampled
inputs. Include unpacking, resets, growth, peeling, and packing; exclude sampling,
compilation, syndrome verification, diagnostic serialization, and plotting.
Report batch-one median/p95/p99 and fixed-batch throughput separately. Measure
normal-width all-zero-syndrome calls as an interface floor including input/reset/
output work, report it beside real-shot times, and do not subtract it. Fix thread
counts, alternate decoder order, and record CPU/build details. These are software
latencies at d=7, not hardware timing estimates.

Add a timing-only point at `p=0.001` for the same six-patch, two-yoke circuit,
retaining d=7 and rounds=28. Regenerate circuits, samples, DEMs, and weights at
that noise strength. A local
audit found the same endpoint/observable topology at the two noise strengths,
but weights differ; do not decode the lower-noise samples with p=0.003 weights.
No extra accuracy campaign is required. Keep both noise points separate in plots
and manifests. Commit the results note and small tables under
`docs/decoder_comparison/`; bulky samples and traces may remain in ignored `out/`.

**Preregistered follow-up.** After the direct comparison, an investigation of
growth-policy changes may consider `uf_weighted_passive_yoke_v1` as a separate
policy. This is outside the initial implementation and benchmark. It uses the
same graph, weights, constrained yokes, tie rule, and peeling; only ordinary detector
endpoints contribute growth. For a crossing edge, replace the endpoint rate by
`active(root(u))*ordinary(u) + active(root(v))*ordinary(v)`, where `ordinary`
excludes yokes and virtual terminals. A yoke endpoint contributes zero even after
joining another cluster; this is stronger than suppressing only a singleton.
A stalled odd cluster is a policy failure (`GrowthStalledError`), not proof of
an invalid syndrome. A local audit found a seven-edge path from each yoke to a
boundary whose syndrome contains only that yoke and whose observable mask has
one bit. This realizable syndrome has no ordinary growth source, so the strict
passive rule stalls. Retain this as a required counterexample: the candidate
needs a separately versioned completion rule before it can be a valid decoder.
No MWPM fallback or silent discard is allowed. Establish validity before any
performance comparison. Keep the original
active-yoke results and separate policy IDs/digests, and use fresh confirmatory
samples after diagnosis.

**Deferred extensions.** Large-sweep collection and Sinter resume/pickling tests,
manifest merge/compatibility frameworks, isolated peak-RSS measurement, larger
observable masks, and extra 1D accuracy operating points follow the first
comparison. The p=0.001 timing point above is included now. Later accuracy
sweeps can use `p in {0.001,0.002,0.003}`. Broader
collection must eventually version task metadata to prevent incompatible resume.

The existing [GapWorkHandler](../src/yoked/gap/_gap_worker_handler.py) subtracts
weights of two minimum matchings. UF does not provide those minima. Any future
UF-based confidence score requires separate definition and calibration before
it can replace complementary gaps or drive an inner-UF/outer-MWPM decoder.
