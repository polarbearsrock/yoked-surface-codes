# Matched frontend comparison at `d=7`, `p=0.003`

Status: detailed experiment specification. The implementation, command-line
interface, and frozen JSON protocol named below are not created by this file.

Claim status: **non-claim-bearing descriptive characterization**. This study
reuses a Union-Find corpus whose Union-Find and Global-MWPM outcomes have
already been inspected. It is therefore not a fresh holdout, even though the
ProMatch and Pinball predictions will be newly evaluated after this design is
frozen.

## 1. Question and scope

On one fixed six-patch yoked surface-code circuit, compare the maintained YSC
integrations of three syndrome frontends followed by global minimum-weight
perfect matching (MWPM):

1. logical accuracy on exactly the same physical shots;
2. reduction in the detector-event workload presented to residual MWPM; and
3. measured software latency, separated into complete decoder latency and
   residual-MWPM backend latency.

The fixed cell is `d=7`, physical error probability `p=0.003`, six patches,
two yokes, and 28 rounds. The experiment compares complete integrations, not
abstract algorithms. Each frontend retains its maintained domain, commit,
rollback, and observable-frame semantics.

No result from this study may be described as a threshold, scaling law,
hardware latency, cryogenic bandwidth reduction, or intrinsic ranking of
Union-Find, ProMatch, and Pinball.

## 2. Standard arm terminology

Reports, tables, plots, and prose must use the following names. Internal
schema identifiers may be shorter, but abbreviations such as `U0`, `PU`,
`PB`, and `UF treatment` must not appear as user-facing arm names.

| Canonical schema identifier | Required user-facing name | Frozen behavior |
| --- | --- | --- |
| `global_mwpm` | **Global MWPM baseline** | Uncorrelated PyMatching on the complete original DEM. |
| `promatch_assisted_mwpm` | **ProMatch-assisted MWPM** | Native L1 ProMatch, then one complete-graph residual MWPM call. |
| `pinball_assisted_mwpm` | **Pinball-assisted MWPM** | Native YSC Pinball V2, then one complete-graph residual MWPM call. |
| `union_find_assisted_mwpm` | **Union-Find-assisted MWPM** | Frozen Patch-UF V3 result, then one complete-graph residual MWPM call. |

The exact frontend policies are:

- ProMatch decoder name:
  `promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching`.
  It uses `windowd` domains, residual Hamming-weight limit 10, disabled local
  boundary matching, and a zero observable frame.
- Pinball decoder name:
  `pinball-ysc-v2-cz-fullhistory-nine-stage-domainatomic-yokeedge-pymatching`.
  It uses the native full-history, nine-stage, `(patch,basis)`-domain-atomic
  YSC V2 policy, including its yoke-edge and observable-frame behavior.
- Union-Find decoder name:
  `weighted-uf-fullhistory-patchlocal-zeroframe-residual-global-mwpm-v1`.
  It uses the frozen V3 exact-weight policy, strict `margin > tau` comparison,
  and `tau=0x0.0p+0`.

Pinball may emit a nonzero observable frame. The frozen ProMatch and
Union-Find policies are zero-frame policies. This difference is part of the
native integrations and must not be normalized away in this experiment.

## 3. Fixed physical cell

The circuit and detector error model are fixed exactly as follows:

```text
cell_id                    cguf-01-d7-n6-y2-r28-p0.003
generator                  yoked._yoked_memory_circuits:yoked_magic_memory_circuit
distance                   7
physical error probability 0.003
patches                    6
yokes                      2
rounds                     28
style                      cz
noise                      si1000
remove_x_yoke              false
DEM decompose_errors       true
DEM approximate_disjoint_errors true
```

The authenticated identities are:

```text
circuit SHA-256  8cfa9bb9eaf6db86dfc9ffcfefa4582eb29932d11dc1fd911239ef1425841ff9
DEM SHA-256      9b06141668bef9b334df78e4700853f60505a71f3c4185746d0261e7e3790e0a
detectors        8354
observables      12
UF graph edges   40836
UF lanes         12
```

Every preparation path must regenerate the circuit and DEM and require these
identities before compiling an arm. A mismatch is a hard failure, not a reason
to regenerate a new corpus.

## 4. Authenticated donor corpus

### 4.1 Why the donor is reused

The existing Union-Find V3 characterization retained all 10,000 packed
detector and observable rows, plus per-shot Global-MWPM and Union-Find
predictions and workload metrics. Reusing these rows makes every accuracy and
workload comparison physically paired. Independent resampling would be less
precise and would not permit direct ProMatch-versus-Pinball-versus-Union-Find
contingency tables.

No new physical shots are sampled in this experiment. The imported detector
and observable arrays are immutable inputs. The existing V3 directory must
never be edited, resumed, or promoted into the new output directory.

### 4.2 Donor provenance

The frozen donor is:

```text
UF experiment ID
  c90f2c578638670eccc73007d6a061f4c8d1b75a16699961937a5ca584432604
UF implementation commit
  51dc959c1310ab09676d7f1ac2e6f981abb4cecb
UF config commit
  872f62f2201d1533e7f333b9d662de1f6b27fd9c
UF protocol self SHA-256
  1ca511db862da9c79dbc06dd519b0f5213f3337d6d71ea31cb5f8392d09586b0
UF source inventory SHA-256
  e28d8ff9a3cecbff5fdcb13e58154dc10755000740af563c77e3a17c564f1678
UF finalization payload SHA-256
  8117c028c05fd16af791bfcf30238950b59e65718cd3eba8e073fec1d32734c9
UF characterization analysis payload SHA-256
  041e9a57152b4d570c977afe366e9f590b72377e8f84bb3c49dfb52d8e768bbd
```

The source files are:

- `docs/PATCH_UF_MWPM_D7_P003_FROZEN_V3.json`;
- `out/cguf_mwpm_d7_p003_v3/characterization_10k_collection/`;
- `out/cguf_mwpm_d7_p003_v3/latency_collection.workload/`;
- `out/cguf_mwpm_d7_p003_v3/characterization_10k_analysis/analysis.json`;
- `out/cguf_mwpm_d7_p003_v3/characterization_10k_replay/replay.json`; and
- `out/cguf_mwpm_d7_p003_v3/finalization/finalization.json`.

The donor byte identities required at import are:

| Object | Canonical/payload identity | File SHA-256 |
| --- | --- | --- |
| Frozen V3 protocol | `1ca511...86b0` self hash | `2a74776f6465d4ae54dc4a993fc41be945c8ccdfe9a42c9f74b7eed64fc3e7db` |
| Characterization collection summary | `6f4f694f475cd10d580e98ac4e81f4a2eb5e5e444bbecc4804f1ad17f1246f61` | `abc0181217686670bca2f883a7c96a380a03910d9b58ab756ac07e4cad4cc123` |
| Corpus index | `88666fc642cbe06c3db179ba3782f91c35f538b5168465e00d19ab9d8ae927ad` | `c98dd67809e4e755dd4ba84e9cc96482fc67bfb953f059dc916c852e88125246` |
| Packed detectors | `614850856efb9ad65de36d1b8990471bf96c8abf9ad306202dbdf6b788cc0622` | same raw-array SHA-256 |
| Packed observables | `dda8541867b879fa9ab78f3f08f0eb4ba2b3ffbbd4478404c2d3e970a35c7ece` | same raw-array SHA-256 |
| UF latency-workload manifest | `2b637e8bd129f5127049113d4658352b4e42167339d503a5a97c84f39238e0f0` self hash | `45b9e3f7b065bd836d08f292c753d54318249992b0c0f1b41ee166c3820c5f61` |
| UF packed residual array | `569f5aaebbb52f5b2483c4367146852b68631fc5909c8c01cd1db328fe5de742` array digest | `03c2e57911fd2bd08d098a5be9bcad6ff3cbd9b8034dba56b0b41ca01c33f82e` |
| Characterization analysis | `041e9a...68bbd` payload | `1b2c3c5370bb7fa01c65aa8e40835bc40b42aeee6ee3430e6ab494a516b83edf` |
| Characterization replay | `f8eb38b5f9ed29d65019f7d425c33bfca4e517af797cd8d9d6ec96c4a121b36d` payload | `474bae22388df7c41aa503443a8d9f6984e12d5a3c96ef3a6789d9b06caf0a74` |
| Finalization | `8117c028...34c9` payload | `8c93c5dd7a87d7bf70ea42d846815732ab6e9498d208e496121a86f394eb5e03` |

The shortened payloads in the display table are labels only. The frozen JSON
protocol must carry the complete 64-character values shown elsewhere in this
section.

The detector array is packed little-endian `uint8`, shape `(10000, 1045)`,
and 10,450,000 bytes. The observable array is packed little-endian `uint8`,
shape `(10000, 2)`, and 20,000 bytes. Unused tail bits must be zero.

### 4.3 Import validation

Before any new decoding, the importer must:

1. validate the frozen donor protocol's canonical self hash;
2. validate the donor finalization and its characterization source identity;
3. validate the collection summary and corpus index self/payload hashes;
4. validate the packed array sizes, dtypes, shapes, tail bits, and SHA-256
   digests;
5. validate all 32 range descriptors and their detector, observable, shot,
   shard, component-file, cross, and payload digests;
6. validate a bijection of `global_shot_id=0,...,9999` with no duplicate,
   gap, or reorder;
7. validate the UF latency-workload manifest and complete packed residual
   array, including its file hash, array digest, shape `(10000, 1045)`, shot-ID
   order, and full-corpus prediction attestation;
8. validate each imported actual-observable, Global-MWPM prediction,
   Union-Find prediction, failure flag, and Union-Find workload record against
   the corresponding packed detector, observable, and UF residual rows; and
9. construct a new read-only import manifest that records every accepted
   input file hash.

The new runtime is allowed to be a later clean commit, so this is byte-level
authentication of the completed donor artifact, not a claim that the old
source tree has been recreated at `HEAD`.

## 5. Fixed 10,000-shot range schedule

Collection uses the donor's exact 32 half-open ranges. No range may be merged,
split, reordered, resampled, or assigned a new shot identifier.

```text
range  0: [   0,  312)  312 shots
range  1: [ 312,  625)  313 shots
range  2: [ 625,  937)  312 shots
range  3: [ 937, 1250)  313 shots
range  4: [1250, 1562)  312 shots
range  5: [1562, 1875)  313 shots
range  6: [1875, 2187)  312 shots
range  7: [2187, 2500)  313 shots
range  8: [2500, 2812)  312 shots
range  9: [2812, 3125)  313 shots
range 10: [3125, 3437)  312 shots
range 11: [3437, 3750)  313 shots
range 12: [3750, 4062)  312 shots
range 13: [4062, 4375)  313 shots
range 14: [4375, 4687)  312 shots
range 15: [4687, 5000)  313 shots
range 16: [5000, 5312)  312 shots
range 17: [5312, 5625)  313 shots
range 18: [5625, 5937)  312 shots
range 19: [5937, 6250)  313 shots
range 20: [6250, 6562)  312 shots
range 21: [6562, 6875)  313 shots
range 22: [6875, 7187)  312 shots
range 23: [7187, 7500)  313 shots
range 24: [7500, 7812)  312 shots
range 25: [7812, 8125)  313 shots
range 26: [8125, 8437)  312 shots
range 27: [8437, 8750)  313 shots
range 28: [8750, 9062)  312 shots
range 29: [9062, 9375)  313 shots
range 30: [9375, 9687)  312 shots
range 31: [9687,10000)  313 shots
```

Scientific collection uses exactly 32 worker processes and one native
numerical thread per process. `MAX_ERRORS` must remain unset. The experiment
is fixed-`N`; no error count, confidence interval, observed ranking, elapsed
time, or favorable trend may stop it early.

## 6. Four-arm same-shot execution

For every range, the immutable packed detector and observable slices are read
once. The same rows are decoded by all four arms. Decoder order may not affect
input data; before/after array digests must detect mutation.

The Global-MWPM baseline and both newly evaluated frontends run in the new
process. The Union-Find final predictions and residual-count telemetry are
imported from the authenticated V3 donor. A range ledger is publishable only
after all four arm rows reconcile.

### 6.1 Mandatory Global-MWPM equality gate

The new baseline must be constructed from the regenerated common DEM with
`pymatching.Matching.from_detector_error_model(dem)` and sufficient fault IDs
for all 12 observables. Its packed prediction must equal the imported donor
Global-MWPM prediction for every one of the 10,000 rows, byte for byte.

Required final attestation:

```text
rows checked       10000
equal              10000
mismatches         0
prediction digest  2e7e2694ce9db1759529aeed77dba09c79f4bb098a0434731115626f0ee13faf
```

Any mismatch invalidates the comparison. The tool must report the first
global shot ID and both packed predictions, but it must not continue, replace
the donor prediction, or silently define a new baseline.

The ProMatch and Pinball compiled graph/layout/schedule fingerprints, decoder
configuration hashes, and complete-source hashes must be frozen during the
config-only protocol step. Different domain layouts are permitted; a
different circuit or DEM is not.

### 6.2 Required compact per-shot data

The new collection must retain, indexed by `global_shot_id`:

- packed actual observables;
- packed final prediction for all four arms;
- failure bit for all four arms;
- original detector-event count `H`;
- residual detector-event count for each assisted arm;
- body, terminal, and yoke residual-event counts for each assisted arm; and
- enough compact native status fields to reconcile the aggregate telemetry in
  Section 9.

The packed donor detector corpus is already retained. Full per-shot Python
decoder objects are not required. New retained arrays and range ledgers must
be self-hashing, immutable, and resumable by missing range only.

## 7. Accuracy estimands

For shot `i` and arm `a`, define failure as at least one incorrect observable:

```text
F[i,a] = any(prediction[i,a] XOR actual_observables[i])
```

The report must retain the complete 16-bin correctness cube in canonical arm
order:

```text
(Global MWPM baseline,
 ProMatch-assisted MWPM,
 Pinball-assisted MWPM,
 Union-Find-assisted MWPM)
```

`0` means correct and `1` means wrong. All marginal failure counts and all
pairwise tables must be independently reconstructed from this cube.

### 7.1 Marginal estimates

For every arm, report:

- logical failures and shots;
- logical failure rate; and
- two-sided 95% Clopper-Pearson interval.

### 7.2 Paired estimates

Report all six unordered arm pairs. For a named comparison `A minus B`, use
`B` as the comparator and `A` as the candidate:

```text
a = B correct, A correct
b = B correct, A wrong     (regression)
c = B wrong,   A correct   (recovery)
d = B wrong,   A wrong
```

The absolute difference in logical failure rate is:

```text
delta(A minus B) = (b - c) / 10000
                 = failure_rate(A) - failure_rate(B)
```

Reports must spell out **percentage points** when multiplying this quantity
by 100. They must not abbreviate it as `pp`, and they must not call it a
relative percentage change.

The three primary descriptive comparisons are:

1. ProMatch-assisted MWPM minus Global MWPM baseline;
2. Pinball-assisted MWPM minus Global MWPM baseline; and
3. Union-Find-assisted MWPM minus Global MWPM baseline.

The three secondary direct comparisons are ProMatch versus Pinball, ProMatch
versus Union-Find, and Pinball versus Union-Find. Canonical direction must be
frozen in the JSON protocol and used consistently in tables and plots.

For every pair, report:

- `a`, `b`, `c`, and `d`;
- discordant count `b+c` and discordance rate;
- absolute difference in logical failure rate;
- two-sided 95% Tango efficient-score interval for the paired difference;
- exact two-sided McNemar/binomial p-value from `b` and `c`; and
- packed-prediction agreement and disagreement counts.

Raw p-values are descriptive. The analyzer may also report a Holm adjustment
over the six frozen pairwise tests, but neither raw nor adjusted p-values
create a confirmatory claim in this already-inspected corpus.

Relative change and failure-rate ratio may be shown as explicitly secondary
descriptions when the comparator rate is nonzero. The absolute paired
difference remains the primary accuracy effect.

## 8. Common workload estimands

Let `H_i` be the number of nonzero detectors in the original syndrome for
shot `i`, and `R_i,a` the number presented to residual global MWPM by assisted
arm `a`.

The primary common workload ratio is a ratio of totals:

```text
workload_ratio[a] = sum_i R[i,a] / sum_i H[i]
```

The primary workload reduction is:

```text
workload_reduction[a] = 1 - workload_ratio[a]
```

This is not the unweighted mean of per-shot ratios. Shots with `H_i=0` remain
in the experiment and create no division by zero in the ratio-of-totals
definition.

For each assisted arm, report:

- original and residual detector-event totals and means;
- workload ratio and workload reduction;
- mean residual-minus-original detector count;
- body, terminal, and yoke original/residual totals; and
- joint original/residual Hamming-weight histogram.

For each assisted-arm pair, report the paired difference in residual detector
count per shot and the difference in workload ratio using the common original
denominator. Positive values must always mean that the first named arm leaves
more residual work.

### 8.1 Workload intervals

The range ledgers must retain the complete-shot joint histogram

```text
(H, R_promatch, R_pinball, R_union_find, count).
```

Use one shared multinomial bootstrap over these complete-shot joint cells,
with 10,000 replicates and a fresh 256-bit seed root frozen in the new JSON
protocol. A single replicate must preserve all four values from each selected
shot. Use two-sided percentile 95% intervals for every workload ratio,
reduction, and paired workload difference. A replicate with a zero denominator
is non-estimable for that endpoint; an endpoint receives no interval unless
all 10,000 replicates are estimable.

Residual-event workload reduction is a software graph-sparsification metric.
It is not cryogenic link bandwidth reduction, memory traffic, number of MWPM
edges explored, or guaranteed latency reduction.

## 9. Common and algorithm-specific telemetry

### 9.1 Directly comparable telemetry

The following fields have identical definitions across arms and may appear in
comparison tables:

- shots and activation/no-change shots;
- original and residual detector-event counts;
- original/residual joint histogram;
- body, terminal, and yoke event counts;
- final observable-frame activity and Hamming weight;
- whether any local correction was durably committed;
- complete final prediction and failure bit; and
- residual-MWPM backend latency on the same timing corpus.

“Activation” means that the frontend attempted policy work beyond a no-op. A
separate `durable_commit` flag is required because activation may end in
rollback or deferral.

### 9.2 Supplemental native telemetry

These fields may be reported within an arm but must not be placed in a column
that implies numerical equivalence across algorithms:

- ProMatch domain status, attempted/committed stages, committed paths,
  residual-Hamming-weight limit, fallback reason, and rollback counts;
- Pinball simple/complex domains, all-simple/mixed/no-commit shots, tentative
  and committed primitive/physical support, and `M`, `B1`--`B4`, `ST1`--`ST2`,
  `H`, and `E` match counts; and
- Union-Find cluster size, confidence margin, committed/deferred component,
  growth/union/heap/peel counters, port taint, and patch transaction outcome.

A ProMatch match, Pinball primitive, and Union-Find component are different
units. Their counts must never be used to claim that one frontend performed
more or less “work” than another. Cluster-size plots remain Union-Find-only.

## 10. Latency experiment

### 10.1 Why every arm is retimed

The completed UF latency study remains authenticated, but its wall-clock
measurements must not simply be joined to new ProMatch/Pinball measurements.
All four complete decoder paths are retimed on the same host, corpus, restart
schedule, and session so host drift and run-to-run conditions are shared.

The existing UF latency workload may be used as an authenticated input. Its
relevant identities are:

```text
corpus digest
  614850856efb9ad65de36d1b8990471bf96c8abf9ad306202dbdf6b788cc0622
latency corpus manifest self SHA-256
  2b637e8bd129f5127049113d4658352b4e42167339d503a5a97c84f39238e0f0
latency corpus manifest file SHA-256
  45b9e3f7b065bd836d08f292c753d54318249992b0c0f1b41ee166c3820c5f61
UF residual packed-array digest
  569f5aaebbb52f5b2483c4367146852b68631fc5909c8c01cd1db328fe5de742
full-corpus prediction attestation
  0444258bbbb3807c760138540046b97b251f82d6eac580f75f189bde2890edc5
```

ProMatch and Pinball residual corpora are materialized from the same 10,000
detector rows before timing and authenticated by packed-array and prediction
digests. Materialization, verification, and serialization are outside every
measured interval.

### 10.2 Timed paths

For each assisted arm, measure these balanced pairs:

1. complete assisted decoder divided by complete Global MWPM baseline; and
2. MWPM on that arm's precomputed residual corpus divided by MWPM on the
   original corpus.

The complete decoder timer begins at the public production adapter entry with
an already-selected packed detector batch and ends when the packed prediction
is returned. It includes input validation, bit unpacking, frontend work,
residual packing, the residual matcher call, observable-frame application, and
output masking that occur in the production path.

The backend timer contains only a prevalidated matcher `decode_batch`
invocation and its return. The original and residual arrays are already
packed, selected, and validated. Backend results must equal the corresponding
complete-path predictions in untimed checks.

The following are always outside the timer:

- circuit, DEM, layout, graph, schedule, and decoder compilation;
- process startup and CPU/NUMA pinning;
- donor import and corpus loading;
- row scheduling and batch-slice construction;
- residual-corpus generation;
- actual-observable loading or correctness scoring;
- telemetry capture;
- digest calculation, equality checks, serialization, and file I/O; and
- warmup.

The timing worker must have no actual-observable array or loader path. Garbage
collection is disabled during warmup and timing and restored afterward. The
clock is `time.perf_counter_ns`.

### 10.3 Exact UF-matched schedule

Use the prior UF V3 schedule seed:

```text
5cbee24595d86127816900494c44ac84a5ca3893b53d03e9279d355bf1093abe
```

| Batch size | Fresh restarts | Balanced blocks per restart | Timed calls per side per block | Warmup calls per variant |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 20 | 10 | 50 |
| 64 | 5 | 4 | 2 | 5 |
| 1024 | 3 | 2 | 1 | 1 |

Every pair uses deterministically derived AB/BA-balanced block order and the
same scheduled corpus keys on both sides. Pair execution order and warmup
variant order are independently derived from the frozen schedule seed. A
restart ledger records every call, block total, row key, order, and runtime
identity.

Batch size 1 is the primary latency result. Batch sizes 64 and 1024 are
descriptive batching sensitivity only.

### 10.4 Host and isolation

Each restart runs in a fresh spawned process. Timed restart concurrency is
exactly one. No accuracy collection or other simulation may run concurrently.

The required host policy is:

```text
OS          Linux
machine     x86_64
kernel      4.18.0-553.50.1.el8_10.x86_64
CPU         AMD EPYC 9374F 32-Core Processor
microcode   0xa101148
affinity    CPU 31 only
NUMA nodes  [0]
```

The runtime must set all six native numerical thread variables to `1` before
importing NumPy or PyMatching:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
```

Runtime identity is checked both before and after every restart. A host,
affinity, NUMA, thread, source, package, corpus, or prediction mismatch
invalidates that restart rather than producing a partial latency report.

### 10.5 Latency estimands

For each batch size and timed pair, report:

- geometric mean of paired block-total ratios;
- two-sided 95% interval from a 10,000-replicate hierarchical percentile
  bootstrap that samples restarts first and paired blocks within restarts;
- numerator and denominator p50 and p95 call latency;
- pooled p99 call latency and ratio with its frozen bootstrap interval; and
- exact restart, block, and call counts.

The complete-decoder ratio to Global MWPM is the primary performance
estimand. Residual-backend relief explains whether sparsification helps MWPM;
it does not make the frontend overhead additive in a strict arithmetic sense.
Cross-frontend timing ratios may be derived from common-baseline,
restart-aligned estimates, but no independent host or historical run may be
substituted into them.

These are latencies of the current Python/native software integrations. They
must not be used as estimates of FPGA, ASIC, or cryogenic frontend latency.

## 11. Compilation, memory, and worker model

Pinball V2 full-history graph and schedule compilation is materially heavier
than the Union-Find lane compilation. Accuracy collection therefore follows
the existing Pinball/ProMatch campaign model:

1. the single-threaded parent authenticates the donor and compiles the common
   cell and all required decoders once;
2. a short-lived 32-worker `fork` pool inherits this read-mostly state through
   copy-on-write;
3. every worker must hit the inherited preload and is forbidden from silently
   recompiling;
4. workers process bounded microbatches, recommended size 32, and return
   compact additive telemetry rather than full per-shot decoder objects; and
5. the pool exits after the one cell is complete.

Before the 10,000-shot run, execute a maximum-range capacity probe on a
313-shot range. Record wall time, parent and worker proportional-set size,
private-dirty memory, output bytes, and peak temporary telemetry. Require no
swap activity and sufficient headroom for 32 workers. Summed RSS is not a
valid copy-on-write memory estimate.

Latency restarts deliberately do not use the 32-worker collection pool. They
are fresh, serialized processes; compilation remains outside the timer even
when it dominates wall-clock setup.

## 12. Immutable lifecycle and gates

### 12.1 Two-commit freeze

Use an additive two-commit sequence:

1. **Implementation commit A** adds the dedicated matched-corpus importer,
   collector, analyzer, replay, latency, finalizer, command-line tools, tests,
   and this specification. It should not modify the frozen V3 output or change
   existing decoder behavior.
2. At clean commit A, run the full test suite, exact donor audit, cell compile
   probe, maximum-range capacity probe, and non-scientific smoke.
3. **Config commit B** adds exactly one file,
   `docs/MATCHED_FRONTENDS_D7_P003_FROZEN_V1.json`. It records commit A, every
   relevant source hash, complete donor identities, exact arm policies and
   fingerprints, range schedule, bootstrap seeds, latency schedule, host
   policy, output limits, and gate attestations.
4. Scientific collection and analysis require a clean checkout at commit B.

No source, test, specification, or tool may change in config commit B. If a
gate requires code changes, make a new implementation commit and freeze a new
protocol version.

### 12.2 Required gates

Before full collection:

- clean worktree and exact commit/source inventory;
- pinned software versions: CPython 3.14.5, Stim 1.16.0, PyMatching 2.4.0,
  Sinter 1.16.0, NumPy 2.5.1, SciPy 1.18.0;
- all repository tests pass;
- donor import validates every byte and range;
- regenerated circuit/DEM hashes match Section 3;
- all arms compile and frozen fingerprints match;
- a small smoke under `$TMPDIR` reconciles all arm predictions and telemetry;
- 313-shot capacity and memory probe passes;
- a disposable 1,000-row engineering shakeout using global shot IDs
  `0,...,999` completes, verifies, analyzes, and replays;
- interrupted-run/resume test installs only missing immutable range ledgers;
- all 1,000 shakeout Global predictions equal their donor values; and
- output roots for the 10,000-shot characterization and latency suite do not
  exist.

The shakeout is a repeated subset of the already-inspected donor corpus. It
tests engineering only and cannot be promoted or counted in addition to the
10,000 characterization rows.

### 12.3 Collection validity gates

The 10,000-shot collection is valid only if:

- all 32 declared ranges exist exactly once and no unknown file exists;
- every range and compact array passes schema and digest validation;
- exactly 10,000 unique global shot IDs reconcile in order;
- every four-arm correctness cube, marginal, pairwise table, agreement table,
  and workload histogram reconciles independently;
- Global-MWPM equality is `10000/10000` with zero mismatches;
- imported Union-Find marginal and baseline contingency reproduce the donor
  counts `a=5664`, `b=923`, `c=347`, `d=3066` and 8,022 packed-prediction
  agreements;
- ProMatch and Pinball ordinary production predictions equal their explicitly
  instrumented telemetry paths on all rows;
- no input mutation, censoring, unrecorded fallback, or over-budget truncation
  is present; native ProMatch rollback and Pinball complex-domain dispositions
  remain valid when explicitly recorded and reconciled; and
- fresh-process replay validates every retained case.

Failure of a validity gate stops analysis. Scientific outcome values are not
validity gates: an arm may be more accurate, less accurate, faster, slower,
more sparse, or less sparse without invalidating the run.

## 13. Output tree and finalization

Use a new immutable output root, for example:

```text
out/matched_frontends_d7_p003_v1/
  donor_import/
    manifest.json
  shakeout_1k/
  characterization_10k/
    collection/
      ranges/
      predictions/
      workload/
      summary.json
    verification/
    replay/
    analysis/
      analysis.json
      report.md
      comparison.csv
  latency/
    workload/
    restarts/
    suite.json
    verification/
    analysis/
      latency_analysis.json
      report.md
  finalization/
    finalization.json
    report.md
```

Every publication is atomic and fail-if-exists. Resume validates all existing
artifacts and schedules only missing declared ranges or restarts. Unknown,
malformed, duplicate, partially written, or provenance-inconsistent files
fail closed.

Finalization binds the frozen protocol, donor-import manifest, 1,000-row
shakeout, complete 10,000-shot collection, independent verification, replay,
accuracy/workload analysis, latency suite, and latency analysis into one
self-hashing identity. Finalization authenticates what was run; it does not
turn the descriptive study into a holdout.

## 14. Command workflow placeholders

The following commands specify the intended interface for the future
`tools/benchmark_matched_frontends` implementation. They are placeholders
until that tool and the frozen JSON protocol exist.

Session setup:

```bash
cd /data2/s2chitni/yoked-surface-codes
source .venv/bin/activate
export TMPDIR=/data2/s2chitni/.tmp
export MPLCONFIGDIR="$TMPDIR/yoked-surface-codes-matplotlib"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
unset MAX_ERRORS
```

Pre-freeze engineering gates at implementation commit A:

```bash
python -m pytest -q

tools/benchmark_matched_frontends donor-audit \
  --donor out/cguf_mwpm_d7_p003_v3 \
  --out "$TMPDIR/matched-frontends-donor-audit"

tools/benchmark_matched_frontends compile-probe \
  --donor out/cguf_mwpm_d7_p003_v3 \
  --out "$TMPDIR/matched-frontends-compile-probe"

tools/benchmark_matched_frontends smoke \
  --donor out/cguf_mwpm_d7_p003_v3 \
  --out "$TMPDIR/matched-frontends-smoke"

tools/benchmark_matched_frontends capacity-probe \
  --donor out/cguf_mwpm_d7_p003_v3 \
  --range-id 1 \
  --processes 32 \
  --out "$TMPDIR/matched-frontends-capacity-probe"
```

After adding and checking out the config-only frozen protocol commit B:

```bash
PROTOCOL=docs/MATCHED_FRONTENDS_D7_P003_FROZEN_V1.json
ROOT=out/matched_frontends_d7_p003_v1

tools/benchmark_matched_frontends import-donor \
  --protocol "$PROTOCOL" \
  --donor out/cguf_mwpm_d7_p003_v3 \
  --out "$ROOT/donor_import"

tools/benchmark_matched_frontends collect \
  --protocol "$PROTOCOL" \
  --donor-import "$ROOT/donor_import" \
  --shot-start 0 --shots 1000 \
  --processes 32 \
  --out "$ROOT/shakeout_1k"

tools/benchmark_matched_frontends verify \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/shakeout_1k"

tools/benchmark_matched_frontends replay \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/shakeout_1k" \
  --out "$ROOT/shakeout_1k/replay"

tools/benchmark_matched_frontends analyze \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/shakeout_1k" \
  --out "$ROOT/shakeout_1k/analysis"

tools/benchmark_matched_frontends collect \
  --protocol "$PROTOCOL" \
  --donor-import "$ROOT/donor_import" \
  --shots 10000 \
  --processes 32 \
  --out "$ROOT/characterization_10k/collection"

tools/benchmark_matched_frontends verify \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/characterization_10k/collection" \
  --out "$ROOT/characterization_10k/verification"

tools/benchmark_matched_frontends replay \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/characterization_10k/collection" \
  --out "$ROOT/characterization_10k/replay"

tools/benchmark_matched_frontends analyze \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/characterization_10k/collection" \
  --out "$ROOT/characterization_10k/analysis"

tools/benchmark_matched_frontends materialize-latency \
  --protocol "$PROTOCOL" \
  --collection "$ROOT/characterization_10k/collection" \
  --out "$ROOT/latency/workload"

tools/benchmark_matched_frontends latency \
  --protocol "$PROTOCOL" \
  --workload "$ROOT/latency/workload" \
  --out "$ROOT/latency/restarts"

tools/benchmark_matched_frontends verify-latency \
  --protocol "$PROTOCOL" \
  --workload "$ROOT/latency/workload" \
  --latency "$ROOT/latency/restarts" \
  --out "$ROOT/latency/verification"

tools/benchmark_matched_frontends analyze-latency \
  --protocol "$PROTOCOL" \
  --latency "$ROOT/latency/restarts" \
  --out "$ROOT/latency/analysis"

tools/benchmark_matched_frontends finalize \
  --protocol "$PROTOCOL" \
  --root "$ROOT" \
  --out "$ROOT/finalization"
```

The eventual CLI may refine argument spelling, but it must preserve the
separate audit, collection, verification, replay, analysis, latency, and
finalization phases and their immutable input/output boundaries.

## 15. Interpretation limits

This experiment can support statements of the form:

> On the authenticated 10,000-shot `d=7`, `p=0.003`, six-patch, two-yoke,
> 28-round YSC corpus, the maintained ProMatch-, Pinball-, and Union-Find-
> assisted MWPM integrations had the following paired differences in logical
> failure rate, residual detector-event workload, and measured software
> latency relative to Global MWPM.

It cannot support:

- a fresh-holdout or confirmatory claim;
- a general ordering over distances, physical error rates, patch counts,
  yoke counts, or round counts;
- an intrinsic algorithmic comparison after controlling domain visibility,
  frame policy, or implementation language;
- a reproduction of the public Pinball software/hardware results;
- FPGA, ASIC, cryogenic, power, area, energy, or bandwidth claims;
- a statement that detector-count reduction equals matcher-runtime reduction;
  or
- an accuracy claim for a future hardware implementation unless it is shown
  bit-exact to the frozen software policy.

The study's strongest feature is exact same-shot pairing. Its main limitation
is equally explicit: it describes one already-inspected corpus and three
different maintained frontend contracts.
