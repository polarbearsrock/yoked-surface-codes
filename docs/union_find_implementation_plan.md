# Weighted Union Find implementation plan for 1D yoked surface codes

Implement the [weighted UF design](union_find_design.md) on the complete joint
detector graph. The initial operating point is **d=7, p=0.003, rounds=28,
CZ gates, SI1000 noise**. Compare **UF against joint MWPM on the 1D yoked code
with six patches and two yokes**, encoding four logical qubits. The two yokes
are the logical X- and Z-parity checks on the same patch group.
The single-Y-yoke variant and 2D Squareberg code
are outside this implementation plan. The primary baseline is vanilla
`pymatching`; correlated matching is a later, separately labelled comparison.

Status: implementation pending. Measure UF's accuracy and latency relative to
joint MWPM, with diagnostics for large tied yoke merges. A small paired pilot
precedes the main native port. The native stack is **C++17, GCC toolset 14,
pybind11 3.0.4, and CMake/Ninja**. Broader collection infrastructure is deferred. Add a separate
timing-only point at p=0.001 with the same d=7, rounds=28, and six-patch circuit.

**Evidence and interpretation.** The review measured roughly 42–51% yoke firing
on its d=3/d=5 circuits at p=0.001. A fresh vanilla-MWPM probe at the requested
point produced these results with 10,000 shots per circuit, seed 42,
Stim/Sinter 1.16.0, and PyMatching 2.4.0:

| Yokes | Patches | Yoke degrees | Yoke firing fractions | MWPM block failures |
| --- | --- | --- | --- | --- |
| 2 | 6 | 696, 696 | 50.11%, 50.60% | 3,424 / 10,000 |

The [probe manifest](union_find_baseline_d7_p003.json) records parameters,
versions, and sample/DEM hashes. The table includes only the six-patch,
two-yoke circuit. Other original probe rows remain historical evidence in the
manifest and are excluded from the active comparison. These are baseline
measurements, not UF results.
The block failure ratio is capped at about 2.921 at this baseline rate. Follow
[step3_plot](../step3_plot) and Sinter's piece conversion with
`(pieces, values)=(168,8)`. The committed counts give a MWPM rate of
`2.560290509e-3` per patch-round. The review's earlier `2.48e-3` estimate comes
from a separate 100,000-shot run. Preserve raw counts and label the converted
unit as an effective normalization; it does not imply independent patch errors.

The design records the review's timing and tied-event summaries: about 33
minimum-weight fired neighbors join a fired yoke at its first batch, usually
spanning all six patches. Half of these clusters remain odd. Frequent yoke
firing is expected when raw logical parities are sufficiently noisy. Test these
events directly and measure them under the implemented policy. A yoke retains
its syndrome constraint, and all edges completed in a batch must be processed
even if early unions temporarily make its cluster even.

**Delivery sequence.** Each stage has an acceptance condition.

| Stage | Deliverable | Depends on |
| --- | --- | --- |
| M1 | Audited graph, initial fixtures, native build smoke check | Design |
| M2 | Scan oracle, heap growth, work counters | M1 |
| M3 | Peeling, tied-star tests, paired pilot sized from timing | M2 |
| M4 | Native batch decoder and backend validation | M3 correctness checks and bounded pilot |
| M5 | Minimal paired/timing harness and Sinter facade | M4 |
| M6 | 100,000-shot native comparison and plots | M5 |

M6 supplies the accuracy/latency comparison. Python runs are for correctness,
the early pilot, and a few hundred profiled shots. Accuracy is a measured
outcome; a worse UF result does not prevent completing the comparison.

**M1: graph and build foundations.** Add `src/yoked/decoders/_graph.py`, adjacent
tests, `__init__.py`, and a native build skeleton under `native/uf/`.

- [ ] Construct an explicit DEM with `decompose_errors=True` and
  `approximate_disjoint_errors=True`. Audit flattened components before exporting
  `Matching.from_detector_error_model(dem).edges()`. Preserve import order and
  the independent parallel-edge merge / first-observable-mask behavior. Reject
  unsupported nonzero-probability hyperedges, observable-only components, and
  negative/nonfinite weights. Retain zero-cost edges.
- [ ] Define immutable endpoints, weights, observable masks, and CSR adjacency.
  Preserve full DEM detector/observable counts. Give each one-detector edge its
  own virtual terminal. Every yoke remains constrained. Record graph digests
  and degree statistics.
- [ ] Generate the initial d=7, p=0.003 circuit with the 1D memory generator:
  `yokes=2, patches=6`. Limit benchmark circuit selection to this 1D yoked
  configuration; keep synthetic graphs for fast tests. Check isolated detectors,
  repeat/shift handling, boundary edges,
  labels, and baseline graph agreement. Supply yoke IDs as diagnostic metadata
  from the known generator (last IDs, X before Z).
  These tags must not change the generic parser or growth policy.
- [ ] Audit each edge's X-observable-mask parity against incidence to the X
  yoke, and likewise for Z. Check that every labelled edge in the 1D circuit is a
  yoke edge with one observable matching its ordinary endpoint's patch/sector;
  ordinary edges stay within a patch/sector. Audit the circuit's unlabelled
  boundary edges. Verify generator patch/sector metadata, yoke weight
  multiplicities, and absent-observable widths.
  Repeat the topology/label audit when generating p=0.001 timing graphs; weights
  and DEM hashes must come from their own noise point.
- [ ] Pin pybind11 3.0.4 in an optional native-build requirements file. Add
  `native/uf/CMakeLists.txt` and `tools/build_uf_native`; build/import a minimal
  binding with the existing Python. Install `requirements-uf-native.txt` using
  `uv pip install --python .venv/bin/python -r requirements-uf-native.txt`.
  Select `/opt/rh/gcc-toolset-14/root/usr/bin/gcc` and `g++` explicitly. Use
  C++17, Release mode, Ninja, and no fast-math. Put downloads, build files,
  uv/compiler caches, and the extension under `$TMPDIR`, as required by the
  workspace temporary-file instructions. Record paths, versions, flags, and
  the extension digest; retain the explicit `PYTHONPATH` import arrangement.

The local check found GCC toolset 14.2.1, Python 3.14.5 with the GIL enabled and
headers present, and uv available. The venv has neither pip nor pybind11 yet.
CMake 3.26.5, Ninja 1.8.2, and ccache are available. pybind11 3 supports Python
3.14; the M1 binding build/import check remains necessary for this combination.
[Compatibility](https://pybind11.readthedocs.io/en/stable/changelog.html),
[CMake integration](https://pybind11.readthedocs.io/en/stable/compiling.html).

Acceptance: graph semantics match vanilla PyMatching on the initial 1D circuit,
unsupported models fail explicitly, and the pinned binding imports from `$TMPDIR`.

**M2: heap growth and a scan oracle.** Add `_union_find.py`, `_growth_scan.py`,
and adjacent behavioral tests.

- [ ] Implement workspace reset, iterative path-compressed DSU find/union by
  size, parity XOR, terminal-flag OR, and forest edges separate from DSU parents.
  Initialize fired yokes as active singletons and close zero-cost edges. Roots grow iff
  odd and without a terminal; even roots can reactivate after merging.
- [ ] Implement a simple scan scheduler as the oracle for the accepted continuous
  event rule. Use it on tiny graphs and random realizable syndromes rather than
  for the operating-point campaign.
- [ ] Implement the Python run scheduler with `heapq` and lazy deletion. Entries
  contain absolute completion time, edge ID, and generation token. Maintain
  residual growth and the time/rate at which an edge was last settled. Make
  settlement at the same timestamp idempotent, including when both endpoints
  change activity. Settle affected edges at old rates before changing rates,
  then publish new deadlines. A rate increase must immediately publish its
  earlier event; waiting to pop an obsolete later deadline is incorrect.
- [ ] Reschedule frontiers whose activity changed, including an even yoke or
  cluster becoming active. Avoid rescanning an unchanged active yoke frontier
  for unrelated events. Root renaming alone need not reschedule an edge. Track
  invalidation/rebuild work; a heap alone does not bound frontier-update cost.
- [ ] Collect and latch tied completions before updating activity. Union the
  entire collected batch even when intermediate parity becomes even. Re-find
  roots per edge and skip internal edges without recording them. Use stable
  edge-ID order per closure wave. Publish changed deadlines, then collect again
  until no valid entry lies within tolerance of the original batch time. Use
  `epsilon(t)=1e-12 + 1e-12*abs(t)` initially and record both constants. Never
  widen the window transitively. Test newly published same-time events and
  edges affected at both endpoints. The scan oracle uses the same batch rule.
- [ ] Record edge visits, rescheduled edges and rescheduling passes, heap
  pushes/pops/stale pops, maximum heap/frontier size, find/union counts,
  completed edges, and growth
  events from this stage onward. Attribute work at yoke endpoints and record
  yoke-cluster activity toggles, first-batch size/parity/patch coverage, and final
  component diagnostics. Coalesce redundant same-time frontier updates; count
  the work actually performed. Full traces remain optional.

Cross-check scan/heap validity, event ordering away from ties, and predictions
on small cases. Trace mismatches to a numerical tie or a bug. Work counts are
algorithmic-work measures, not hardware latency. Label counter settings and
measure instrumentation overhead separately when timing the native backend.

Acceptance: no stale deadline can conceal an earlier event. High-degree tests
exercise activation, deactivation, and merging without a yoke shortcut or MWPM
fallback. Successful growth leaves valid components.

**M3: peeling, required fixtures, and the paired pilot.** Extend the decoder,
add `_yoked_decoding_test.py`, and introduce a small paired runner that will
become `tools/benchmark_decoders`. Add `_comparison_stats.py` and adjacent
tests here so piece conversion and paired bootstrap are available at the pilot.

- [ ] Peel trees iteratively toward a terminal when available, otherwise a
  deterministic detector. Cancel residual detector parity and XOR selected
  observable masks.
  Raise `InvalidSyndromeError` on impossible syndromes, and reset correctly after
  errors. Keep correction edges available for independent validation.
- [ ] Verify `H*c=s` on **every Python pilot shot**, and verify the reported mask
  equals `L*c`. Check the separate X- and Z-yoke observable identities of the
  1D circuit. Use synthetic fixtures with one yoke vertex to exercise either
  check locally.
  Exhaustively test tiny realizable syndromes and random heap/scan cases.
  MWPM fixture comparisons use well-separated, unambiguous weights because its
  internal quantization can change near-tie outcomes.
- [ ] Record diagnostics for each yoke on every shot: input bit, first union
  batch time, post-closure ordinary-member/fired-member counts, parity, and
  patch bitset; final counts, patch bitset, terminal presence, and per-patch
  member-defect parity by sector. An untouched yoke has an absent first event.
  Retain stable yoke IDs across root changes, failed-observable masks, and
  selected traces. Successful-shot records provide comparison denominators.
- [ ] Verify `prediction_i,b = D_i,b XOR B_i,b` from the design, where `B_i,b`
  is selected unlabelled boundary-edge parity. Without a terminal this reduces
  to member-defect parity; with a terminal it generally does not. Check direct
  selected-yoke-edge parity and the global X/Z yoke constraints too. Metadata
  and diagnostics must not select a different correction.
- [ ] Implement every fixture below. Profile **50 checked, instrumented shots**
  on the 1D circuit before selecting the Python pilot size. Use elapsed time,
  never failure counts, to select `N=min(2000, floor(300/(1.25*t)))` fresh paired
  shots. Here `t` is mean seconds per checked paired calibration shot; this
  budgets 300 seconds with a 25% margin. Record N before decoding accuracy
  samples and actual completed shots if the runtime cap is reached. A short
  or imprecise pilot remains preliminary and does not delay native work once
  correctness checks pass. Additional Python profiling, including calibration,
  is capped at 256 shots.

| Required fixture | Expected behavior |
| --- | --- |
| Unfired yoke between defects | Edges `A--Y=1`, `B--Y=4`, `C--Y=100`; only A/B fired. A reaches Y at time 1, B joins at 2.5, Y has even correction incidence, and C stays outside. |
| Fired yoke, competing patches | Y/A/B/C fired; edges `Y--A=1`, `Y--B=4`, `B--C=1`. Y joins nearer A while B/C become even. Reaching Y does not automatically absorb B/C. |
| Fired yoke meets an even cluster | All Y/E1/E2/D fired; edges `E1--E2=1`, `Y--E1=3`, `E2--D=6`. E1/E2 become even first; absorbing that cluster leaves the Y cluster odd and growth continues to D. |
| Fired tied star across six patches | Y and `k` ordinary neighbors fired; all spoke weights 1, with `k=32` and `k=33` assigned across six patches. At time 0.5 all spokes union in one batch despite intermediate parity changes. Size excluding Y is k and parity is `1 XOR (k mod 2)`. Add a slower terminal path to finish the odd case. |
| Tied star with fired partners | Add `m` distinct fired partners to selected star neighbors, partner-edge weight 1. At time 0.5 spokes and partner edges all complete. Every reachable partner joins; size is `k+m`, parity is `1 XOR ((k+m) mod 2)`. Vary m and edge order; parity must include partners. |
| Unfired tied star | Y unfired, k fired neighbors, spoke weights 1 and no earlier partner edges. All neighbors reach Y at time 1; parity is `k mod 2`. Test odd/even k and give the odd case a slower terminal continuation. |
| Unfired yoke with earlier partner pairing | Give each fired neighbor a distinct fired partner at weight 1 and its Y spoke weight 2. Pairs become even at time 0.5; the unfired Y can remain untouched. Its first-event diagnostic is absent. |
| Equal-weight triangle, three defects | After two unions the third completed edge is internal and skipped. With no terminal, the remaining odd component raises an error. Add a slower terminal edge for a valid companion case. |
| Fired isolated detector | Use a synthetic graph with an isolated detector, fire it, and require `InvalidSyndromeError`. |
| Zero costs / multiple terminals | Preserve labels through zero-time closure and satisfy detector constraints when tied growth joins terminals. |
| Boundary diagnostic identity | A yoke component reaching a terminal satisfies member-defect parity XOR selected-boundary parity. The same fixture without a terminal satisfies the simpler member-parity identity. |
| Long tree / repeated settlement | Grow and peel a chain longer than Python's recursion limit. Advancing the same edge twice at a timestamp, with both endpoints changing activity, adds growth only once. |

**Pilot reporting and readiness for M4.** Compute the effective per-patch-round
rate using `sinter.shot_error_rate_to_piece_error_rate(P, pieces=168, values=8)`.
Report the direct ratios `Q=q_UF/q_MWPM` and `R=P_UF/P_MWPM`.
Use the design's paired bootstrap: 10,000 replicates, recorded seed, 95%
percentile intervals; preserve the pairing of UF and MWPM outcomes on each shot.
Test conversion against Sinter, joint-count reconstruction,
identical paired outcomes, and zero/saturated cases. Do not discard problematic
replicates or manufacture finite ratios with pseudocounts.

Proceed to M4 after correction-validity and behavioral checks pass. There is no
numerical accuracy threshold: the goal is to measure the UF/MWPM tradeoff.
Record early accuracy estimates and uncertainty, fix invalid corrections or
unexplained implementation mismatches, and leave statistical precision to M6.
Save block rates/ratios, differences in both units, four paired counts, and
per-shot rescheduled-edge/pass counts, yoke activity toggles, heap work, and
cluster diagnostics. Keep Python and native estimates separate and reserve
fresh samples for M6.

The optional follow-up after the initial comparison is
`uf_weighted_passive_yoke_v1`, with endpoint rates and the `GrowthStalledError`
condition specified in the design. The design's
seven-edge yoke-to-boundary counterexample shows why this strict candidate needs
a completion rule before any performance comparison. Preserve it as a test.
Retain active-yoke results; any necessary completion rule is a separately
versioned policy. This follow-up is not a prerequisite for M4 through M6.

**M4: native decoding before large experiments.** Add `native/uf/uf_core.h`,
`uf_core.cc`, `bindings.cc`, and `_native_test.py`.

- [ ] Port heap scheduling, union, peeling, reset, and the full batch loop to
  C++17. Use contiguous CSR arrays, `std::priority_queue` with generation tokens,
  float64 growth/time, and reusable workspace. Preserve tested activity-change
  and simultaneous-event rules, iterative traversal, and idempotent settlement;
  use touched-state resets where profiling helps. Prioritize frontier-update
  costs using the M3 rescheduling measurements.
- [ ] Bind a top-level `_yoked_uf_native` module loaded from the explicit `$TMPDIR`
  output directory on `PYTHONPATH`, avoiding source-package shadowing. Keep
  buffers alive and release the GIL around the native batch kernel. Return
  packed observables without per-shot Python callbacks.
- [ ] Use `uint64_t` masks for the pilot's 12 observables; explicitly reject
  models above 64 until broader support is added. Test pilot byte padding,
  immutable inputs, empty batches, and error recovery. Retain both Python engines.
- [ ] Require validity and observable reconstruction for both backends, and exact
  agreement on tie-free fixtures. On sampled/tied cases, report prediction
  mismatch rate and trace tolerance-induced differences rather than requiring
  universal equality. Unexplained or tie-free mismatches remain defects. Provide
  native correction verification, including boundary/yoke identities, for the
  full accuracy run; batched sparse verification is an acceptable alternative.
  No Python per-shot verification loop belongs in the 100,000-shot path.
- [ ] Guard optional native tests with `pytest.importorskip`. The required
  native validation command must first import the extension explicitly and then
  execute its tests; a missing build must not count as successful validation.
- [ ] Measure normal-width all-zero-syndrome calls for both decoders as an
  interface floor. This includes input/reset/output work. Report it separately
  without subtracting it from real-shot latency. Record compiler flags, binding
  version, counters, and profiling overhead.

Acceptance: the native kernel passes operating-point validity checks and runs
the packed batch loop natively. Tied differences are explained and quantified.
There is no prerequisite long Python timing campaign.

**M5: minimal paired/timing harness.** Finish `_benchmark.py`, harness checks,
`tools/benchmark_decoders`, and a thin `_sinter.py` facade exposing `uf_weighted`.

- [ ] Provide `pilot`, `paired`, `profile`, and `timing` modes. Build graphs once
  and stream shared circuit samples. Save seed/batching, circuit/DEM/graph/sample hashes,
  actual shot totals, backend/source digest, versions, tolerances, and flags in
  a simple run manifest. Also record piece/value arguments, bootstrap seed and
  repetitions, calibration/pilot budgets, verification backend, and
  the policy version. Native diagnostic output is returned per batch, without
  Python callbacks per shot.
- [ ] Reuse M3's piece conversion and paired bootstrap for the direct ratio
  `Q=q_UF/q_MWPM`, while retaining block probabilities, `R=P_UF/P_MWPM`,
  differences, discordant-pair counts, and prediction disagreements. Stratify failures and
  work by first-event size/parity/patch coverage, final member parities, terminal
  contact, and yoke firing. Retain compact successful-shot diagnostics too.
- [ ] Report approximate bootstrap intervals and zero/saturation handling as
  specified in the design. For degenerate zero-count cases add one-sided count
  bounds and flag unresolved accuracy estimates. Do not claim exact coverage
  for the bootstrap or replace joint resampling with marginal-quotient intervals.
- [ ] Time warmed, precompiled decoders on pre-sampled inputs: batch size one for
  median/p95/p99, a fixed larger batch for throughput. Include decoder-owned
  unpack/reset/grow/peel/pack work; exclude sampling, compilation, syndrome
  verification, serialization, and plotting. Label instrumentation consistently.
- [ ] Use fresh output directories and a manifest. Defer resume, multiprocessing,
  and general merge infrastructure. The Sinter facade needs only compile-for-DEM,
  packed decoding, and a bounded interface smoke check. The paired path must be
  usable before this facade is complete.

Acceptance: counts reconstruct both decoder error rates, decoders receive the
same samples without discards, and timing scope is explicit. Validate count
arithmetic and the harness with constructed outcomes and a tiny 1D real circuit.

**M6: native comparison and reporting.** Add `tools/plot_decoder_comparison` and
usage documentation after the commands exist.

- [ ] Run **100,000 shared shots** on the six-patch, two-yoke circuit at d=7,
  p=0.003, with seeds distinct from the pilot. Validate corrections natively or
  with sparse operations per batch on the accuracy path. If discordances remain insufficient, report
  uncertainty rather than claiming the shot budget
  guarantees power. Timing runs disable syndrome verification.
- [ ] Separately measure native single-shot latency, batch-1024 throughput, and
  all-zero-syndrome latency on every graph. Fix pools/warm-up/repetitions/thread
  counts, alternate decoder order, and record CPU and build details. Python
  profiling remains capped at a few hundred shots. Repeat timing only
  at p=0.001 on regenerated circuits/DEMs with the appropriate weights and
  samples. Keep d=7, rounds=28, six patches, and two yokes; no p=0.001 accuracy run.
- [ ] Plot per-patch-round rates and the direct UF/MWPM ratio `Q` with
  paired-bootstrap intervals as primary outputs. Include raw block rates/ratios/differences for
  interpretation at this noisy point. Plot latency and throughput separately by
  noise strength. Summarize rescheduled edges/passes, activity toggles, first
  and final yoke-cluster sizes, patch coverage, and parity diagnostics.
- [ ] Commit `docs/decoder_comparison/1d_d7_p003_results.md`, small CSV/JSON
  tables, and compact manifests under `docs/decoder_comparison/`; ignored `out/`
  is for bulky samples/traces and intermediate plots. Include the pilot findings,
  backend mismatch rate, timing point, and limitations. Identify the effective
  normalization, noisy accuracy point, and ideal terminal yoke measurements.
  Do not infer low-error scaling or hardware latency from software timing.

Acceptance: native UF and vanilla MWPM are compared on identical samples with
uncertainty and interface-floor measurements. UF need not outperform MWPM for
the experiment to yield a complete result.

**Deferred scope.** Large-sweep `collect` mode, Sinter resume/pickling tests,
general manifest compatibility/merge guards, isolated-process peak-RSS,
more-than-64-observable support/tests, and complementary gaps. Python may retain
arbitrary-width masks without expanding the initial test matrix. Later 1D
accuracy sweeps may use `p in {0.001,0.002,0.003}`; the initial
accuracy point remains p=0.003. The p=0.001 timing-only point is included now.

**Commands for implementation sessions.** New tools below are specified
interfaces, not implemented commands. Run from the repository root.

```bash
set -euo pipefail

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$TMPDIR/uf-pycache"
export MPLCONFIGDIR="$TMPDIR/uf-mpl"
export CCACHE_DIR="$TMPDIR/uf-ccache"
export UV_CACHE_DIR="$TMPDIR/uf-uv-cache"
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++

# M1: install the pinned optional binding dependency and smoke-test the skeleton.
uv pip install --python .venv/bin/python -r requirements-uf-native.txt
.venv/bin/python tools/build_uf_native \
    --build_dir "$TMPDIR/yoked-uf-build" --out_dir "$TMPDIR/yoked-uf-native"
export PYTHONPATH="$TMPDIR/yoked-uf-native:$PYTHONPATH"
.venv/bin/python -c 'import _yoked_uf_native'

.venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/decoders

.venv/bin/python tools/gen_memory_circuit \
    --patch_diameter 7 --rounds 28 --noise_strength 0.003 \
    --patches 6 --yokes 2 --gateset cz \
    --out_dir out/decoder_comparison/1d_d7_p003/circuits

# M3: calibrate 50 shots, freeze a pilot size, then verify every correction.
# Defaults record a 25% timing margin and 10k bootstraps; no accuracy threshold.
.venv/bin/python tools/benchmark_decoders \
    --mode pilot --backend python \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*patches=6,yokes=2,*.stim \
    --decoders pymatching uf_weighted --max_shots 2000 --batch_size 32 \
    --calibration_shots 50 --seconds_budget 300 \
    --seed 42 --calibration_seed 41 --bootstrap_seed 4201 \
    --verify_syndrome --out_dir out/decoder_comparison/1d_d7_p003/pilot

# M4: rebuild after the port; the import must succeed before native validation.
.venv/bin/python tools/build_uf_native \
    --build_dir "$TMPDIR/yoked-uf-build" --out_dir "$TMPDIR/yoked-uf-native"
.venv/bin/python -c 'import _yoked_uf_native'
.venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/decoders

# M6: native accuracy comparison on fresh samples.
.venv/bin/python tools/benchmark_decoders \
    --mode paired --backend native \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*patches=6,yokes=2,*.stim \
    --decoders pymatching uf_weighted --shots 100000 --batch_size 256 --seed 4200 \
    --bootstrap_seed 4202 --verify_syndrome --verification_backend native \
    --out_dir out/decoder_comparison/1d_d7_p003/native-accuracy

# Same geometry, new noise-dependent weights and samples; timing only at p=0.001.
.venv/bin/python tools/gen_memory_circuit \
    --patch_diameter 7 --rounds 28 --noise_strength 0.001 \
    --patches 6 --yokes 2 --gateset cz \
    --out_dir out/decoder_comparison/1d_d7_p001/circuits

for uf_noise_tag in 003 001; do
    .venv/bin/python tools/benchmark_decoders \
        --mode timing --backend native \
        --circuits out/decoder_comparison/1d_d7_p${uf_noise_tag}/circuits/*patches=6,yokes=2,*.stim \
        --decoders pymatching uf_weighted --shots 10000 --batch_size 1 --seed 43 \
        --empty_syndrome_floor \
        --out_dir out/decoder_comparison/1d_d7_p${uf_noise_tag}/native-latency

    .venv/bin/python tools/benchmark_decoders \
        --mode timing --backend native \
        --circuits out/decoder_comparison/1d_d7_p${uf_noise_tag}/circuits/*patches=6,yokes=2,*.stim \
        --decoders pymatching uf_weighted --shots 10240 --batch_size 1024 --seed 43 \
        --out_dir out/decoder_comparison/1d_d7_p${uf_noise_tag}/native-throughput
done
```

Keep tests beside their modules in `src/yoked/decoders/`. Use `$TMPDIR` for
pytest scratch, non-result traces, downloaded build dependencies, and caches.
Record actual commands and result paths as tasks complete. Commit the design,
implementation plan, and small baseline evidence together.
