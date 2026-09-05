# Weighted Union Find implementation plan for 1D yoked surface codes

Implement the [weighted UF design](union_find_design.md) on the complete joint
detector graph. The initial operating point is **d=7, p=0.003, rounds=28,
CZ gates, SI1000 noise**. Compare the **1D yoked code with six patches and two
yokes** against an **unyoked control with four patches**, preserving the
four-encoded-qubit comparison. The two yokes are the logical X- and Z-parity
checks on the same patch group. The single-Y-yoke variant and 2D Squareberg code
are outside this implementation plan. The primary baseline is vanilla
`pymatching`; correlated matching is a later, separately labelled arm.

Status: implementation pending. This revision makes active-yoke behavior the
main hypothesis, adds a heap scheduler and an accuracy gate after M3, and puts
native decoding before the large experiment. The native stack is **C++17,
pybind11 3.0.4, and CMake/Ninja**. Broader collection infrastructure is deferred.

**Evidence and interpretation.** The review measured roughly 42–51% yoke firing
on its d=3/d=5 circuits at p=0.001. A fresh vanilla-MWPM probe at the requested
point produced these results with 10,000 shots per circuit, seed 42,
Stim/Sinter 1.16.0, and PyMatching 2.4.0:

| Yokes | Patches | Yoke degrees | Yoke firing fractions | MWPM block failures |
| --- | --- | --- | --- | --- |
| 0 | 4 | — | — | 4,794 / 10,000 |
| 2 | 6 | 696, 696 | 50.11%, 50.60% | 3,424 / 10,000 |

The [probe manifest](union_find_baseline_d7_p003.json) records parameters,
versions, and sample/DEM hashes. The table includes only the two arms in scope;
the manifest retains the original single-yoke probe row as historical evidence
and marks it excluded from the active comparison. These are baseline
measurements, not UF results.
The point provides many baseline failures, but paired power still depends on
discordant outcomes. Block error rates are high: report absolute differences
alongside failure ratios, whose maximum is bounded by `1 / P_fail(MWPM)`.
This experiment does not establish low-error scaling.

The hypothesis is that a fired yoke starts a large active frontier, and its
growth through multiple patches can degrade UF's correction. The decoder peels
a merge forest, so growth and tie order determine most correction choices.
Instrument and test this before large experiments. A yoke retains its syndrome
constraint; adjacent local detectors join only after weighted edges complete.

**Delivery sequence.** Each stage has an acceptance condition.

| Stage | Deliverable | Depends on |
| --- | --- | --- |
| M1 | Audited graph, initial fixtures, native build smoke check | Design |
| M2 | Scan oracle, heap growth, work counters | M1 |
| M3 | Peeling, yoke tests, 2,000-shot accuracy gate | M2 |
| M4 | Native batch decoder and backend validation | M3 gate |
| M5 | Minimal paired/timing harness and Sinter facade | M4 |
| M6 | 100,000-shot native comparison and plots | M5 |

M6 supplies the accuracy/latency comparison. Python runs are for correctness,
the early gate, and a few hundred profiled shots. A negative accuracy result at
M3 is a useful finding and changes the next work accordingly.

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
- [ ] Generate the two initial d=7, p=0.003 arms with the 1D memory generator:
  `yokes=2, patches=6` and the control `yokes=0, patches=4`. Limit benchmark
  circuit selection to 1D memory and its unyoked control; keep synthetic graphs
  for fast tests. Check isolated detectors, repeat/shift handling, boundary edges,
  labels, and baseline graph agreement. Supply yoke IDs as diagnostic metadata
  from the known generator (last IDs, X before Z); exclude the zero-yoke dummy.
  These tags must not change the generic parser or growth policy.
- [ ] Pin pybind11 3.0.4 in an optional native-build requirements file. Add
  `native/uf/CMakeLists.txt` and `tools/build_uf_native`; build/import a minimal
  binding with the existing Python. Use C++17, Release mode, Ninja, and no
  fast-math. Put downloads, build files, compiler caches, and the extension under
  `$TMPDIR`.

The inspected environment has GCC 8.5.0, CMake 3.26.5, and Ninja 1.8.2;
pybind11 is not installed yet. pybind11 3 supports Python 3.14 and provides
CMake module helpers. The build smoke check verifies this particular combination
before decoder work depends on it.
[Compatibility](https://pybind11.readthedocs.io/en/stable/changelog.html),
[CMake integration](https://pybind11.readthedocs.io/en/stable/compiling.html).

Acceptance: graph semantics match vanilla PyMatching on both initial arms,
unsupported models fail explicitly, and the pinned binding imports from `$TMPDIR`.

**M2: heap growth and a scan oracle.** Add `_union_find.py`, `_growth_scan.py`,
and adjacent behavioral tests.

- [ ] Implement workspace reset, path-compressed DSU find/union by size, parity
  XOR, terminal-flag OR, and forest edges separate from DSU parents. Initialize
  fired yokes as active singletons and close zero-cost edges. Roots grow iff
  odd and without a terminal; even roots can reactivate after merging.
- [ ] Implement a simple scan scheduler as the oracle for the accepted continuous
  event rule. Use it on tiny graphs and random realizable syndromes rather than
  for the operating-point campaign.
- [ ] Implement the Python run scheduler with `heapq` and lazy deletion. Entries
  contain absolute completion time, edge ID, and generation token. Maintain
  residual growth and the time/rate at which an edge was last settled. Settle
  affected edges at the old rates before an activity change, then publish new
  deadlines. A rate increase must immediately publish its earlier event;
  waiting to pop an obsolete later deadline is incorrect.
- [ ] Reschedule frontiers whose activity changed, including an even yoke or
  cluster becoming active. Avoid rescanning an unchanged active yoke frontier
  for unrelated events. Root renaming alone need not reschedule an edge. Track
  invalidation/rebuild work; a heap alone does not bound frontier-update cost.
- [ ] Collect tied completions before updating activity. For each edge, re-find
  both endpoint roots at processing time. Skip internal edges without recording
  them; otherwise union and record the edge once. Process zero-time closure
  before advancing, with documented tolerances and stable edge order.
- [ ] Record edge visits, frontier reschedules, heap pushes/pops/stale pops,
  maximum heap/frontier size, find/union counts, completed edges, and growth
  events from this stage onward. Attribute work at yoke endpoints and record
  active-yoke cluster activity. Full traces remain optional.

Cross-check scan/heap validity, event ordering away from ties, and predictions
on small cases. Trace mismatches to a numerical tie or a bug. Work counts are
algorithmic-work measures, not hardware latency. Label counter settings and
measure instrumentation overhead separately when timing the native backend.

Acceptance: no stale deadline can conceal an earlier event. High-degree tests
exercise activation, deactivation, and merging without a yoke shortcut or MWPM
fallback. Successful growth leaves valid components.

**M3: peeling, required fixtures, and the accuracy gate.** Extend the decoder,
add `_yoked_decoding_test.py`, and introduce a small paired runner that will
become `tools/benchmark_decoders`.

- [ ] Peel trees toward a terminal when available, otherwise a deterministic
  detector. Cancel residual detector parity and XOR selected observable masks.
  Raise `InvalidSyndromeError` on impossible syndromes, and reset correctly after
  errors. Keep correction edges available for independent validation.
- [ ] Verify `H*c=s` on **every Python pilot shot**, and verify the reported mask
  equals `L*c`. Check the separate X- and Z-yoke observable identities of the
  1D circuit. Use synthetic fixtures with one yoke vertex to exercise either
  check locally.
  Exhaustively test tiny realizable syndromes and random heap/scan cases.
  MWPM fixture comparisons use well-separated, unambiguous weights because its
  internal quantization can change near-tie outcomes.
- [ ] Maintain a `contains_yoke` flag per cluster by OR on union. For every failed
  shot save the input yoke firing mask, failed-observable mask, and whether any
  syndrome-bearing/correction cluster contained a yoke. Retain per-component
  flags for investigation; an untouched zero-syndrome yoke singleton does not
  count. These tags are associations, not a unique causal attribution.
- [ ] Implement every fixture below. Then run **2,000 shared shots on d=7,
  p=0.003, rounds=28, six patches, two yokes**, using heap UF and vanilla MWPM.
  Save seed/batching, paired counts, work statistics, failure tags, and gate
  status. Cap additional Python profiling at 256 shots per arm.

| Required fixture | Expected behavior |
| --- | --- |
| Unfired yoke between defects | Edges `A--Y=1`, `B--Y=4`, `C--Y=100`; only A/B fired. A reaches Y at time 1, B joins at 2.5, Y has even correction incidence, and C stays outside. |
| Fired yoke, competing patches | Y/A/B/C fired; edges `Y--A=1`, `Y--B=4`, `B--C=1`. Y joins nearer A while B/C become even. Reaching Y does not automatically absorb B/C. |
| Fired yoke meets an even cluster | All Y/E1/E2/D fired; edges `E1--E2=1`, `Y--E1=3`, `E2--D=6`. E1/E2 become even first; absorbing that cluster leaves the Y cluster odd and growth continues to D. |
| Equal-weight triangle, three defects | After two unions the third completed edge is internal and skipped. With no terminal, the remaining odd component raises an error. Add a slower terminal edge for a valid companion case. |
| Fired isolated detector | Flip the zero-yoke circuit's isolated final detector and require `InvalidSyndromeError`. |
| Zero costs / multiple terminals | Preserve labels through zero-time closure and satisfy detector constraints when tied growth joins terminals. |

**Gate after M3.** Report `R = UF failures / MWPM failures`, absolute paired
difference, four paired outcome counts, and uncertainty. A ratio above about
2 triggers investigation before native/benchmark expansion. If the ratio
interval lies above 2, diagnose yoke traces and growth order before continuing
with the current accuracy tradeoff. If uncertainty straddles 2, mark the result
inconclusive and use a bounded additional gate batch or a minimal native port
to resolve it. Do not silently change scheduling to pass the gate. A Python
runtime problem does not require a new native-stack decision. Reserve different
samples for the later 100,000-shot experiment.

**M4: native decoding before large experiments.** Add `native/uf/uf_core.h`,
`uf_core.cc`, `bindings.cc`, and `_native_test.py`.

- [ ] Port heap scheduling, union, peeling, reset, and the full batch loop to
  C++17. Use contiguous CSR arrays, `std::priority_queue` with generation tokens,
  float64 growth/time, and reusable workspace. Preserve tested activity-change
  and simultaneous-event rules; use touched-state resets where profiling helps.
- [ ] Bind a top-level `_yoked_uf_native` module loaded from the explicit `$TMPDIR`
  output directory on `PYTHONPATH`, avoiding source-package shadowing. Keep
  buffers alive and release the GIL around the native batch kernel. Return
  packed observables without per-shot Python callbacks.
- [ ] Use `uint64_t` masks for the pilot's 8 or 12 observables; explicitly reject
  models above 64 until broader support is added. Test pilot byte padding,
  immutable inputs, empty batches, and error recovery. Retain both Python engines.
- [ ] Require validity and observable reconstruction for both backends, and exact
  agreement on tie-free fixtures. On sampled/tied cases, report prediction
  mismatch rate and trace tolerance-induced differences rather than requiring
  universal equality. Unexplained or tie-free mismatches remain defects.
- [ ] Measure normal-width all-zero-syndrome calls for both decoders as an
  interface floor. This includes input/reset/output work. Report it separately
  without subtracting it from real-shot latency. Record compiler flags, binding
  version, counters, and profiling overhead.

Acceptance: the native kernel passes operating-point validity checks and runs
the packed batch loop natively. Tied differences are explained and quantified.
There is no prerequisite long Python timing campaign.

**M5: minimal paired/timing harness.** Finish `_benchmark.py`, count tests,
`tools/benchmark_decoders`, and a thin `_sinter.py` facade exposing `uf_weighted`.

- [ ] Provide `paired`, `profile`, and `timing` modes. Build graphs once and stream
  shared circuit samples. Save seed/batching, circuit/DEM/graph/sample hashes,
  actual shot totals, backend/source digest, versions, tolerances, and flags in
  a simple run manifest.
- [ ] Report block failure probabilities and primary UF/MWPM ratios `R_0`
  (unyoked control) and `R_2` (1D yoked code), absolute paired differences,
  discordant-pair counts, and prediction disagreements. Break out failures by
  firing and cluster-yoke tags. Save every failed shot's compact tags, with full
  traces for a small subset.
- [ ] Add binomial uncertainty and zero-count handling. From 97.5% exact marginal
  intervals, use `[UF_low/MWPM_high, UF_high/MWPM_low]` as a conservative ratio
  interval. For the difference, subtract 97.5% intervals for UF-only and MWPM-only
  event probabilities. The union bound gives at least 95% coverage without
  independence. Report undefined/unbounded ratios if the baseline has no failures.
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
arithmetic and the harness with constructed outcomes and a tiny real circuit.

**M6: native comparison and reporting.** Add `tools/plot_decoder_comparison` and
usage documentation after the commands exist.

- [ ] Run **100,000 shared shots per arm** at d=7, p=0.003, with seeds distinct
  from the gate. Validate corrections on the accuracy path. If discordances
  remain insufficient, report uncertainty rather than claiming the shot budget
  guarantees power. Timing runs disable syndrome verification.
- [ ] Separately measure native single-shot latency, batch-1024 throughput, and
  all-zero-syndrome latency on every graph. Fix pools/warm-up/repetitions/thread
  counts, alternate decoder order, and record CPU and build details. Python
  profiling remains capped at a few hundred shots per arm.
- [ ] Plot `R_0` and `R_2` with intervals as a primary output, alongside absolute
  block error rates/differences. Plot latency and throughput separately. Summarize
  yoke work, frontier/heap size, and failure tags to test the main hypothesis.
- [ ] Write a results note with manifests, tables, plots, gate outcome, backend
  mismatch rate, and limitations. Identify the noisy d=7 operating point and
  ideal terminal yoke measurements. Do not infer low-error scaling or hardware
  latency from these software measurements.

Acceptance: native UF and vanilla MWPM are compared on identical samples with
uncertainty and interface-floor measurements. UF need not outperform MWPM for
the experiment to yield a complete result.

**Deferred scope.** Large-sweep `collect` mode, Sinter resume/pickling tests,
general manifest compatibility/merge guards, isolated-process peak-RSS,
more-than-64-observable support/tests, and complementary gaps. Python may retain
arbitrary-width masks without expanding the initial test
matrix. Later 1D noise sweeps may use `p in {0.001,0.002,0.003}`; the initial point
remains p=0.003 throughout.

**Commands for implementation sessions.** New tools below are specified
interfaces, not implemented commands. Run from the repository root.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$TMPDIR/uf-pycache"
export MPLCONFIGDIR="$TMPDIR/uf-mpl"
export CCACHE_DIR="$TMPDIR/uf-ccache"

.venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/decoders

.venv/bin/python tools/gen_memory_circuit \
    --patch_diameter 7 --rounds 28 --noise_strength 0.003 \
    --patches '4+yokes' --yokes 0 2 --gateset cz \
    --out_dir out/decoder_comparison/1d_d7_p003/circuits

# M3: verify every Python correction in the early accuracy gate.
.venv/bin/python tools/benchmark_decoders \
    --mode paired --backend python \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*yokes=2,*.stim \
    --decoders pymatching uf_weighted --shots 2000 --batch_size 32 --seed 42 \
    --verify_syndrome --out_dir out/decoder_comparison/1d_d7_p003/gate

# Diagnostic profiling, capped at 256 Python shots per arm.
.venv/bin/python tools/benchmark_decoders \
    --mode profile --backend python \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*.stim \
    --decoders pymatching uf_weighted --shots 256 --batch_size 1 --seed 41 \
    --verify_syndrome --out_dir out/decoder_comparison/1d_d7_p003/python-profile

# Native dependencies, intermediates, and extension stay under TMPDIR.
.venv/bin/python tools/build_uf_native \
    --build_dir "$TMPDIR/yoked-uf-build" --out_dir "$TMPDIR/yoked-uf-native"
export PYTHONPATH="$TMPDIR/yoked-uf-native:$PYTHONPATH"

# M6: native accuracy comparison on fresh samples.
.venv/bin/python tools/benchmark_decoders \
    --mode paired --backend native \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*.stim \
    --decoders pymatching uf_weighted --shots 100000 --batch_size 256 --seed 4200 \
    --verify_syndrome --out_dir out/decoder_comparison/1d_d7_p003/native-accuracy

.venv/bin/python tools/benchmark_decoders \
    --mode timing --backend native \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*.stim \
    --decoders pymatching uf_weighted --shots 10000 --batch_size 1 --seed 43 \
    --empty_syndrome_floor --out_dir out/decoder_comparison/1d_d7_p003/native-latency

.venv/bin/python tools/benchmark_decoders \
    --mode timing --backend native \
    --circuits out/decoder_comparison/1d_d7_p003/circuits/*.stim \
    --decoders pymatching uf_weighted --shots 10240 --batch_size 1024 --seed 43 \
    --out_dir out/decoder_comparison/1d_d7_p003/native-throughput
```

Keep tests beside their modules in `src/yoked/decoders/`. Use `$TMPDIR` for
pytest scratch, non-result traces, downloaded build dependencies, and caches.
Record actual commands and result paths as tasks complete. Commit the design,
implementation plan, and small baseline evidence together.
