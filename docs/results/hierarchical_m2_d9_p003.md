# Hierarchical L1/L2 full set at d=9, p=0.003: M2 endpoints and decision

**Every result on this page is exploratory.** It is measured on the
**evaluation set** — all 100,000 shots of the recorded four-decoder run — with
calibrators fitted on all 50,000 shots of the generated calibration call. The
analysis is not frozen, no confirmation-role collection exists, and the tooling
still refuses to collect one at this milestone; nothing here is a confirmation
measurement. The page decides whether milestone M3 (selective refinement)
starts.

On 100,000 held-out shots, the primary endpoint pair — UF cluster gap as the
initial estimator, correlated matching gap as the refined one, mixed outer rule
— moved the pooled misattribution rate from **0.361840 [22179 / 61295]** under
`initial_only` to **0.056481 [3462 / 61295]** under `all_refined`. The paired
all-refined minus initial-only difference is **-0.305359**, 95% percentile
interval **(-0.309384, -0.301485)**, which is negative and excludes zero. Block
failure moved from 0.32809 to 0.06820, a paired difference of **-0.259890**,
95% interval **(-0.262810, -0.256990)**. Against the recorded joint MWPM on the
same shots (block failure 0.25660), the `all_refined` cell's paired
block-failure difference is **-0.188400**, 95% interval
**(-0.191100, -0.185690)**. **Decision: proceed to M3** (section 8).

---

## 1. Configuration and provenance

| Parameter | Value |
|---|---|
| Circuit | 1D yoked magic-memory circuit, CZ style |
| Inner patch distance | 9 |
| Rounds | 36 (`4d`) |
| Patches / yokes | 6 / 2 (yokes ideal, excluded from the L1 patch graphs) |
| Physical noise | SI1000, `p=0.003` |
| Decoding graph | 17,762 detectors, 90,732 edges, 12 observables |
| Outer rule | `mixed` (the only rule replayed at M2) |
| Policies | the two endpoints `initial_only` and `all_refined` |
| Normalization | `pieces = 6 x 36 = 216`, `values = 2 x (6 - 2) = 8` |
| Bootstrap | 10,000 replicates resampling whole shots, seed 43 |
| Run date | 2026-09-15 UTC |

**Parent samples and rows.** Each full record decodes every row of one Stim
call: rows **0 to 99,999** of the saved 100,000-shot evaluation call and rows
**0 to 49,999** of the 50,000-shot calibration call. The calibration sample
directory is the pilot's saved sample, copied byte for byte (every file hashes
to the value its `sample.json` declares), so the calibration shots are the
pilot's shots, not a resample; the evaluation sample is imported again from the
recorded run and verified against that run's declared hashes.

| | Calibration | Evaluation |
|---|---|---|
| Source | generated (pilot's saved sample) | imported from the recorded four-decoder run |
| Seed | 142 | 42 |
| Shots in the full call | 50,000 | 100,000 |
| Rows decoded | 0:50000 | 0:100000 |
| Row-id sha256 | `33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41` | `baa5f49fbad78af4964d9ec7eaf2d6327b2d2ca1f4dcf54e2394dfff2e36d58e` |
| Packed payload sha256 | `05f2cdf043a4974a2a972213ee3042f05fe16e254b408b59b1b585b2f8507f2f` | `d55da8f4c9b8287fa0af64f8382de755a243a774030499e108c348b665e102f5` |
| Model identity | `73751a5aa99ac806a4ad5007b310b9a22d3f39c6a4f609d466bf83295268cd9d` | `73751a5aa99ac806a4ad5007b310b9a22d3f39c6a4f609d466bf83295268cd9d` |
| Parent sample identity | `e9ace633bc9ff34b2dc228d63393f4daf58c4f4acbde3c678669b8b9645d8398` | `d1bc5fcf04da0c265f67b23637ee518f730cebc86700d04db09fec097a5130f1` |
| Sampling family identity | `060a8f93813c288d32cf5b8801cfd74c052df1c08092c1e07df152c4d5498933` | `51918c6cd9505f780bd9ef6a6b029a52bb3f68ae81a6ea195f4dbe185668bf69` |
| Decoder identity | `79fafa60cf62051e5132dc38d881e515b969bd703616ed3ddaac5a137ac2fa67` | `79fafa60cf62051e5132dc38d881e515b969bd703616ed3ddaac5a137ac2fa67` |
| Check identity | `728ed4b50714bbf3b9364b9323eed9a82395a0d545739cf4150c552ff2709a45` | `728ed4b50714bbf3b9364b9323eed9a82395a0d545739cf4150c552ff2709a45` |
| Collection identity | `3839df3b5248343d6dd1b0344e060eea6390c2a4baf71bb3164af3dd8bfe3255` | `2d0984a04c42896dc8c5f70f7100f9e191cbb35a841aaabad9d7cf8adb636543` |
| `record.npz` sha256 as collected | `ec042a53bf7a226ab3f8fef315878e4ac0c0e8f1dc0d8b4a56e0ce714ce335bf` | `b6c7d413be632360d97abd367ec07c6da3ad4b51396f1baadb41c27547ec423a` |
| `manifest.json` sha256 as collected | `ac6410275c5a170980ffd97f41304688c05e7c13363624b1b825a294c9bae8dc` | `2d70ded310678a1a37b8b27fc3fd562f7b726283189ada36b5a5e9b07917afd5` |
| `record.npz` sha256 after `import-baselines` | (calibration records carry no baselines) | `5f061fa6e81988b25e1f2cf926d45f21c68ce2e31ed6ad0fab8b9f4ea7e4a5c8` |
| `manifest.json` sha256 after `import-baselines` | | `f84a4231209af8f6ffb61d48e9ba10187efb339b8de8db14a5775865efa756be` |

The two records share one model, one decoder and one check implementation, and
differ in both parent sample and sampling family, which is what the replay stage
requires before it will evaluate a calibration on held-out shots. Importing the
baselines changed the evaluation `record.npz` (65,203,462 to 70,004,596 bytes)
and therefore its artifact hash and manifest hash; its collection, decoder,
model, parent-sample and sampling-family identities are unchanged, and every
manifest field other than `artifacts.record.npz` and the new `baselines` block
is byte for byte the collected one. The calibrators, the replay and the summary
all name the post-import hashes.

**Identities relative to the pilot.** The model, parent-sample and
sampling-family identities are the pilot's. The decoder identity moved from the
pilot's `bd305c71…` to `79fafa60…` and the check identity from `c30067b0…` to
`728ed4b5…`, because M2 changed source files those identities hash:
`src/yoked/hierarchical/_record.py` (the record gained its baselines field; it
is a decoder source) and, among the check sources, `_collect.py` plus the new
`_baselines.py` and `_reproduction.py`. Section 3 shows that the decoded
numbers did not move: the two 2,000-row subset records collected under the new
identities hash byte for byte to the pilot's records.

**Original circuit and DEM.** The evaluation sample's `circuit.stim` and
`model.dem` are the bytes of the recorded run
`/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha`, re-verified by
the `import-baselines` gate of section 7 against that run's own manifest:

```text
circuit.stim  d6f1f852b5d12ca2816c2411fe0dcfb0d2948ea50d6eb12655a9cfd775976d2b
model.dem     858edcae41ba5f5ead2862d28dd517d5f22734b1ebd17c0dc12139ecd6a6278a
```

The saved calibration sample carries the same two hashes, which is why the two
model identities agree; the collect stage rebuilt the circuit and DEM from
today's generator before reusing the copied sample and required the rebuilt
model identity to equal the saved one.

**Calibration artifact.** `calibrators_full.json`, identity
`ccdc1f68a1fa457a9dea4132bac0dae44eaa5340957358ba7400d057853b6d2e`, sha256
`382eaff951e5150cccdf09e2a6b17bbaf7d36e699bc9494f967f2927b4002ff3`,
`payload_sha256`
`fb3cfbb03716a2af403fb7b7d38c305892aae377582063f2df9f185f408efbf7`, fitted on
the 50,000-row calibration record (`record.npz` `ec042a53…`, manifest
`ac641027…`). One isotonic fit per estimator and sector, the six patches of a
sector pooled. Knot convention, as recorded in the artifact:

```text
pav-blocks-of-distinct-scores; knot at each block weighted-mean score;
linear between knots; constant beyond; clipped to [1e-6, 1 - 1e-6]
```

All five estimators are fitted `decreasing` in the residual-error probability.
Knots per estimator (X, Z): `uf:cluster_gap` (6849, 9777), `uf:gap_plain`
(11482, 11909), `uf:gap_correlated` (35245, 35570), `mwpm:gap_plain`
(11439, 11812), `mwpm:gap_correlated` (35595, 34337). The pilot's fits on
2,000 rows had (2751, 1359), (2139, 2009), (3386, 6909), (2118, 1993),
(3307, 6825).

**Code and packages.** Every collection, the two subset verifications, the
baseline import, the calibrators, the replay and the summary were produced at
one commit, `44c3bce9af4cc44e7e004f02a378a946529b7774`, on Python 3.14.5.
Decoder packages `{numpy 2.5.1, pymatching 2.4.0, scipy 1.18.0, stim 1.16.0}`,
check packages `{numpy 2.5.1, pymatching 2.4.0}`, sampling `{stim 1.16.0}`,
calibration `{numpy 2.5.1}`, replay and summary `{numpy 2.5.1, sinter 1.16.0}`.
Replay identity
`8ed8fddc0d1c50fe4144c94d712d33f5781654201d96304d74f8e24b9f72f68d`, replay
manifest sha256
`40cb996a9bbbb8f0e6c12b3856dc09f87d01c3fea53d9242ea94eb3fd5836863`; replay tie
rule "log-weight ties within 1e-09 break to the lowest binary pattern".

## 2. Graph and record checks

The graph gate runs before any row is decoded and is identical on all four
collections (the two full records and the two 2,000-row subsets):

| Quantity | Value |
|---|---:|
| Joint model edges | 90,732 |
| Rebuilt per-patch edges | 90,732 |
| Graph equivalence | equivalent |
| Maximum weight discrepancy | 0.0 |
| Weight mismatches / multiplicity mismatches | 0 / 0 |
| Missing groups / extra groups | 0 / 0 |
| Yoke detector degrees | 1,110 and 1,110 |
| Median detector degree (all detectors, the two yokes included) | 11.0 |
| Maximum non-yoke detector degree | 12 |

Record checks, run before publication. `Joint agreement` compares the final
prediction reconstructed from the per-patch forced costs with the collector's
own joint MWPM decode of the same row; a disagreement is tie-explained when the
two total forced costs agree within the cost tolerance.

| Quantity | Calibration (50,000) | Evaluation (100,000) |
|---|---:|---:|
| Value violations | 0 | 0 |
| Check parity violations (`Hc = s`) | 0 | 0 |
| Final parity violations | 0 | 0 |
| Plain additivity max error | 2.274e-13 | 2.842e-13 |
| Correlated additivity max error | 3.411e-13 | 3.411e-13 |
| Plain preferred-sign disagreements | 0 | 0 |
| Correlated preferred-sign disagreements | 0 | 0 |
| Joint agreement | 0.99996 | 0.99989 |
| Joint disagreements | 2 | 11 |
| Tie-explained disagreements | 2 | 11 |
| **Unexplained disagreements** | **0** | **0** |
| Max disagreement cost difference | 0.0 | 4.547e-13 |

Every disagreement on both records is a cost tie: the largest total-cost
difference over the 11 evaluation disagreements is 4.547e-13 nats, against a
tolerance of 1e-6. No collection recorded any failure or diagnostic entry, and
`checks.passed` is true on all four manifests.

## 3. Subset reproduction and hash equality with the pilot

Two 2,000-row records were collected after the full ones, from the same
samples, with `--rows 0:2000`, and compared with the full records by
`verify-subset`, which locates every subset row in the full record by parent
row id and requires every stored array equal there exactly, floats bit for
bit, baselines excluded.

| | Calibration | Evaluation |
|---|---|---|
| Subset directory | `calibration_pilot_rows` | `evaluation_pilot_rows` |
| Subset collection identity | `68491e1f4eb6b68b0ac915f70fd2762d6b8c305527433e0ba837bd228a2990f9` | `6bc068e35eb8ca7a645fee4ec3499b585c514560c5caf298d7794739872732c3` |
| Subset row-id sha256 | `55f385cf2332d9056aaed6f496e7bebd2df52c6a9547ce2144b309432d4b0290` | `55f385cf2332d9056aaed6f496e7bebd2df52c6a9547ce2144b309432d4b0290` |
| Subset `record.npz` sha256 | `9612c5f2b79b6834163cf3b3df259ae30356112d5236c8aee405326032668f05` | `8cbc6a3ae9bd818dc74b3d075093aa257bf4f72148a5a9931c16986b0b2dea1b` |
| Pilot `record.npz` sha256 (published in the pilot report) | `9612c5f2b79b6834163cf3b3df259ae30356112d5236c8aee405326032668f05` | `8cbc6a3ae9bd818dc74b3d075093aa257bf4f72148a5a9931c16986b0b2dea1b` |
| **Equal to the pilot's record** | **yes** | **yes** |
| Subset `manifest.json` sha256 | `6317dd15f283fefdb863aab0525241972907fb14945ab861f2056af001c3e9ad` | `ae5ee1f444140e3e21fae0474a336cf93a768a3d8c9ed526c547707a3dc8f53c` |
| Full record compared (`record.npz`) | `ec042a53bf7a226ab3f8fef315878e4ac0c0e8f1dc0d8b4a56e0ce714ce335bf` | `b6c7d413be632360d97abd367ec07c6da3ad4b51396f1baadb41c27547ec423a` |
| Subset rows / matched rows | 2,000 / 2,000 | 2,000 / 2,000 |
| Arrays compared | 12 | 12 |
| Arrays equal | 12 of 12 | 12 of 12 |
| Mismatched rows, every array | 0 | 0 |
| `passed` | true | true |
| Verification JSON | `verify_calibration.json`, sha256 `da6f93bb7bf575c4c04641125f735ede1a420bea68f7101a2a63b21e8cb59162` | `verify_evaluation.json`, sha256 `85e988a9891ce4cef71e8412a70c138415ec217ffd6b52159d3f8acbe123c5ae` |

The twelve arrays are `actual`, `yoke`, `uf_reference`, `mwpm_reference`,
`correlated_prediction`, `joint_mwpm`, `cluster_gap`, `dijkstra_states`,
`forced_plain`, `forced_correlated`, `reweighted_patches` and `rows`. Both
checks were made by checker identity `728ed4b5…` at commit `44c3bce9…`. The
evaluation comparison ran against the full record as collected, before the
import changed its hash; the verification JSON names that pre-import hash.

Two facts follow. First, each subset `record.npz` is the pilot's `record.npz`,
byte for byte (`cmp` reports no difference), even though the pilot was
collected at commit `610762d9…` under decoder identity `bd305c71…`: the
identity change of section 1 is a source-hash matter and moved no stored
value. Second, the full records reproduce those 2,000 rows exactly, so the
pilot's rows are a strict subset of this run's rows under the current
provenance contract, and the pilot numbers can be re-derived from the full
records rather than merely compared with them. The subset manifests differ
from the pilot's manifests (`45a3fbfa…`, `b65ab2a4…`) in commit, identities,
source hashes and timestamps, as expected.

## 4. Endpoint tables

Rates carry their denominators; intervals are 95% percentile intervals from
10,000 whole-shot bootstrap replicates at seed 43. Misattribution is measured
on patch-sectors where the fixed reference made exactly one mistake; block
failure is measured on shots. No statistic in this run was `unavailable`, and
no comparison had any zero-denominator replicate.

### 4.1 Primary: `uf:cluster_gap -> gap_correlated`, mixed

| Quantity | initial_only:mixed | all_refined:mixed |
|---|---|---|
| Block failure | 0.32809 [32809 / 100000] (0.32522, 0.33096) | 0.06820 [6820 / 100000] (0.06663, 0.06975) |
| Normalized LER (per patch per round) | 1.887031e-3 | 3.284195e-4 |
| Misattribution (pooled) | 0.361840 [22179 / 61295] (0.358132, 0.365639) | 0.056481 [3462 / 61295] (0.054658, 0.058294) |
| Misattribution (X) | 0.361781 [11094 / 30665] | 0.056318 [1727 / 30665] |
| Misattribution (Z) | 0.361900 [11085 / 30630] | 0.056644 [1735 / 30630] |
| Sector failure (X / Z) | 0.18113 [18113 / 100000] / 0.18147 [18147 / 100000] | 0.03485 [3485 / 100000] / 0.03487 [3487 / 100000] |
| Ties (pooled) | 0.000070 [14 / 200000] | 0 [0 / 200000] |
| Probabilities above one half | 0.001218 [1462 / 1200000] | 0.067633 [81160 / 1200000] |

| Metric | Paired difference | 95% interval | Eligible | Zero-denominator replicates |
|---|---:|---|---:|---:|
| Misattribution (pooled) | **-0.305359** | **(-0.309384, -0.301485)** | 61,295 | 0 |
| Block failure | **-0.259890** | **(-0.262810, -0.256990)** | 100,000 | 0 |

Final sector failures by residual reference-failure stratum (200,000
shot-sectors):

| Stratum | Eligible sectors | initial_only:mixed | all_refined:mixed |
|---|---:|---|---|
| zero | 124,620 | 0.000152 [19 / 124620] | 0.005465 [681 / 124620] |
| one | 61,295 | 0.361840 [22179 / 61295] | 0.056481 [3462 / 61295] |
| multiple | 14,085 | 0.998367 [14062 / 14085] | 0.200852 [2829 / 14085] |

### 4.2 Secondary: `uf:cluster_gap -> gap_plain`, mixed

| Quantity | initial_only:mixed | all_refined:mixed |
|---|---|---|
| Block failure | 0.32809 [32809 / 100000] (0.32522, 0.33096) | 0.25612 [25612 / 100000] (0.25346, 0.25878) |
| Normalized LER (per patch per round) | 1.887031e-3 | 1.395005e-3 |
| Misattribution (pooled) | 0.361840 [22179 / 61295] (0.358132, 0.365639) | 0.255127 [15638 / 61295] (0.251606, 0.258550) |
| Misattribution (X) | 0.361781 [11094 / 30665] | 0.256090 [7853 / 30665] |
| Misattribution (Z) | 0.361900 [11085 / 30630] | 0.254163 [7785 / 30630] |
| Sector failure (X / Z) | 0.18113 [18113 / 100000] / 0.18147 [18147 / 100000] | 0.13846 [13846 / 100000] / 0.13775 [13775 / 100000] |
| Ties (pooled) | 0.000070 [14 / 200000] | 0.000045 [9 / 200000] |
| Probabilities above one half | 0.001218 [1462 / 1200000] | 0.032405 [38886 / 1200000] |

| Metric | Paired difference | 95% interval | Eligible | Zero-denominator replicates |
|---|---:|---|---:|---:|
| Misattribution (pooled) | -0.106713 | (-0.110483, -0.103007) | 61,295 | 0 |
| Block failure | -0.071970 | (-0.074410, -0.069580) | 100,000 | 0 |

| Stratum | Eligible sectors | initial_only:mixed | all_refined:mixed |
|---|---:|---|---|
| zero | 124,620 | 0.000152 [19 / 124620] | 0.009798 [1221 / 124620] |
| one | 61,295 | 0.361840 [22179 / 61295] | 0.255127 [15638 / 61295] |
| multiple | 14,085 | 0.998367 [14062 / 14085] | 0.764075 [10762 / 14085] |

### 4.3 Control: `mwpm:gap_plain -> gap_correlated`, mixed

| Quantity | initial_only:mixed | all_refined:mixed |
|---|---|---|
| Block failure | 0.25660 [25660 / 100000] (0.25396, 0.25926) | 0.06782 [6782 / 100000] (0.06626, 0.06935) |
| Normalized LER (per patch per round) | 1.398106e-3 | 3.265157e-4 |
| Misattribution (pooled) | 0.325433 [17726 / 54469] (0.321493, 0.329260) | 0.067396 [3671 / 54469] (0.065291, 0.069496) |
| Misattribution (X) | 0.326476 [8840 / 27077] | 0.067068 [1816 / 27077] |
| Misattribution (Z) | 0.324401 [8886 / 27392] | 0.067721 [1855 / 27392] |
| Sector failure (X / Z) | 0.13853 [13853 / 100000] / 0.13836 [13836 / 100000] | 0.03455 [3455 / 100000] / 0.03478 [3478 / 100000] |
| Ties (pooled) | 0.000065 [13 / 200000] | 0.000005 [1 / 200000] |
| Probabilities above one half | 0.001583 [1899 / 1200000] | 0.053799 [64559 / 1200000] |

| Metric | Paired difference | 95% interval | Eligible | Zero-denominator replicates |
|---|---:|---|---:|---:|
| Misattribution (pooled) | -0.258037 | (-0.262195, -0.253911) | 54,469 | 0 |
| Block failure | -0.188780 | (-0.191480, -0.186090) | 100,000 | 0 |

| Stratum | Eligible sectors | initial_only:mixed | all_refined:mixed |
|---|---:|---|---|
| zero | 135,569 | 0.000133 [18 / 135569] | 0.004713 [639 / 135569] |
| one | 54,469 | 0.325433 [17726 / 54469] | 0.067396 [3671 / 54469] |
| multiple | 9,962 | 0.998294 [9945 / 9962] | 0.263301 [2623 / 9962] |

The two references define different eligible populations — 61,295 shot-sectors
for the UF reference, 54,469 for the MWPM one — so the three pairs are compared
only against their own `initial_only` cell, never against each other. The
`initial_only` cells of 4.1 and 4.2 are one and the same replay (same arrays,
sha256 `45756c33…`), as they were in the pilot.

**Relative to the pilot.** The pilot's 2,000-row primary difference was
-0.322231 with interval (-0.351237, -0.293117); the full-set value -0.305359
lies inside that interval, and the full-set interval is about 7.4 times
narrower (width 0.007899 against 0.058120). The pilot's block-failure
difference was -0.268000
(-0.288500, -0.247500); the full-set -0.259890 lies inside it. The secondary
pair (pilot -0.115737, interval (-0.143472, -0.087212); full -0.106713) and the
control (pilot -0.263063, interval (-0.291932, -0.235184); full -0.258037)
behave the same way. These are the same 2,000 rows inside the 100,000, not an
independent replication.

## 5. Replay work

These are the calls the specified replay procedure implies: the estimator
pair's fixed initial work charged once per patch, plus incremental work charged
once per distinct refined patch. They are neither the collection work of
section 6 nor elapsed time, and no latency or speedup claim follows from them.
Totals are over 100,000 shots; per-shot means are the totals divided by
100,000.

| Counter | `uf:cluster_gap -> gap_correlated` initial_only | per shot | all_refined | per shot |
|---|---:|---:|---:|---:|
| requested_patch_sectors | 0 | 0 | 1,200,000 | 12 |
| distinct_patches | 0 | 0 | 600,000 | 6 |
| initial_uf_calls | 600,000 | 6 | 600,000 | 6 |
| initial_dijkstra_searches | 1,200,000 | 12 | 1,200,000 | 12 |
| initial_dijkstra_states | 941,470,103 | 9,414.70 | 941,470,103 | 9,414.70 |
| initial_unforced_plain_calls | 0 | 0 | 0 | 0 |
| initial_plain_forced_calls | 0 | 0 | 0 | 0 |
| incremental_unforced_plain_calls | 0 | 0 | 600,000 | 6 |
| incremental_plain_forced_calls | 0 | 0 | 0 | 0 |
| incremental_reweight_passes | 0 | 0 | 600,000 | 6 |
| incremental_correlated_forced_calls | 0 | 0 | 2,400,000 | 24 |

| Counter | `uf:cluster_gap -> gap_plain` initial_only | per shot | all_refined | per shot |
|---|---:|---:|---:|---:|
| requested_patch_sectors | 0 | 0 | 1,200,000 | 12 |
| distinct_patches | 0 | 0 | 600,000 | 6 |
| initial_uf_calls | 600,000 | 6 | 600,000 | 6 |
| initial_dijkstra_searches | 1,200,000 | 12 | 1,200,000 | 12 |
| initial_dijkstra_states | 941,470,103 | 9,414.70 | 941,470,103 | 9,414.70 |
| incremental_plain_forced_calls | 0 | 0 | 2,400,000 | 24 |
| incremental_reweight_passes | 0 | 0 | 0 | 0 |
| incremental_correlated_forced_calls | 0 | 0 | 0 | 0 |

| Counter | `mwpm:gap_plain -> gap_correlated` initial_only | per shot | all_refined | per shot |
|---|---:|---:|---:|---:|
| requested_patch_sectors | 0 | 0 | 1,200,000 | 12 |
| distinct_patches | 0 | 0 | 600,000 | 6 |
| initial_uf_calls | 0 | 0 | 0 | 0 |
| initial_dijkstra_searches | 0 | 0 | 0 | 0 |
| initial_dijkstra_states | 0 | 0 | 0 | 0 |
| initial_unforced_plain_calls | 600,000 | 6 | 600,000 | 6 |
| initial_plain_forced_calls | 2,400,000 | 24 | 2,400,000 | 24 |
| incremental_reweight_passes | 0 | 0 | 600,000 | 6 |
| incremental_correlated_forced_calls | 0 | 0 | 2,400,000 | 24 |

Counters omitted from the second and third tables are zero in both of that
pair's cells. The per-shot means are exact integers except the Dijkstra states,
because every endpoint policy requests every patch-sector or none.

## 6. Actual collection work

What really ran while the records were produced, from each collection's
`manifest.json`. Collection counts every score for every patch, because
collection computes them all once so that any configuration can be replayed
offline; they are therefore larger than any single configuration's replay
counts. Neither number is elapsed time.

| Counter | Calibration (50,000 rows) | Evaluation (100,000 rows) | Calibration subset (2,000) | Evaluation subset (2,000) |
|---|---:|---:|---:|---:|
| Rows | 50,000 | 100,000 | 2,000 | 2,000 |
| Joint decodes | 50,000 | 100,000 | 2,000 | 2,000 |
| UF decodes | 300,000 | 600,000 | 12,000 | 12,000 |
| Dijkstra searches | 600,000 | 1,200,000 | 24,000 | 24,000 |
| Dijkstra states | 470,500,971 | 941,470,103 | 18,816,400 | 18,731,135 |
| Unforced plain matchings | 300,000 | 600,000 | 12,000 | 12,000 |
| Plain forced matchings | 1,200,000 | 2,400,000 | 48,000 | 48,000 |
| Correlated forced matchings | 1,200,000 | 2,400,000 | 48,000 | 48,000 |
| Reweight attempts | 300,000 | 600,000 | 12,000 | 12,000 |
| Correlated validation matchings | 300,000 | 600,000 | 12,000 | 12,000 |
| Resumptions | 0 | 0 | 0 | 0 |

The two subsets' Dijkstra-state counts are the pilot's (18,816,400 and
18,731,135), as the byte-equal records of section 3 require.

| Timing | Calibration | Evaluation | Calibration subset | Evaluation subset |
|---|---:|---:|---:|---:|
| Workers / chunk size | 32 / 50 | 32 / 50 | 32 / 50 | 32 / 50 |
| Collection seconds (`timing.seconds_this_run`) | 410.681 | 804.687 | 36.388 | 36.714 |
| Wall clock of the command | 425.52 s | 818.23 s | 50.79 s | 49.63 s |
| Setup, graph gate and record gate (wall minus collection) | 14.8 s | 13.5 s | 14.4 s | 12.9 s |
| Measured throughput | 121.749 rows/s (3.805 per worker) | 124.272 rows/s (3.883 per worker) | 54.963 rows/s (1.718 per worker) | 54.475 rows/s (1.702 per worker) |

Each throughput is that collection's own rows divided by its own manifest's
`timing.seconds_this_run`, then divided by its worker count:
50000 / 410.68080 = 121.749 rows/s (121.749 / 32 = 3.805 per worker);
100000 / 804.68700 = 124.272 rows/s (124.272 / 32 = 3.883 per worker);
2000 / 36.38785 = 54.963 rows/s (54.963 / 32 = 1.718 per worker);
2000 / 36.71428 = 54.475 rows/s (54.475 / 32 = 1.702 per worker).
Each setup row is that command's wall clock minus the same
`timing.seconds_this_run`: 425.52 - 410.681 = 14.8 s; 818.23 - 804.687 =
13.5 s; 50.79 - 36.388 = 14.4 s; 49.63 - 36.714 = 12.9 s.

The two full collections ran one after the other, never concurrently, each
with 32 workers on a 128-core machine that other users were also loading (load
average about 8 to 10 before the run). The pilot's projection of section 7 of
its report was 37.9 min for the evaluation rows and 19.3 min for the calibration
rows at the pilot's 16-worker per-row cost; the runs took 13.4 min (804.687 s)
and 6.8 min (410.681 s) with 32 workers, a per-worker throughput of 3.8 to 3.9
rows/s against the pilot's 2.7 to 2.8. The 2,000-row subsets are slower per
worker (1.7 rows/s): 2,000 rows are 40 chunks of 50 over 32 workers, so once
the first 32 chunks are handed out at most 8 workers have anything left to do;
their wall time is dominated by setup and that tail, and no per-row cost should
be read from them.

No collection was interrupted, none was resumed, and no `checkpoint.npz` or
`failed_checks.json` remains: `resumptions` is 0 everywhere and no attempted-work
telemetry was lost. The setup row is not decoder work; it is process start, the
sample load, the per-worker rebuild of the d=9 decoders from the
19,665,395-byte DEM, the graph gate and the record gate.

Sizes in MiB (1,048,576 bytes): the evaluation `record.npz` is 65,203,462
bytes (62.18 MiB) as collected, in line with the pilot's projection of about
62 MiB per 100,000 rows, and 70,004,596 bytes (66.76 MiB) with the four
baseline columns; the calibration `record.npz` is 32,603,462 bytes
(31.09 MiB); the saved samples keep their packed detector arrays, 222,100,128
bytes (211.81 MiB) and 111,050,128 bytes (105.91 MiB); `calibrators_full.json`
is 9,584,073 bytes (9.14 MiB).

## 7. Historical baselines on the same 100,000 shots

`import-baselines` attached the recorded run's four saved decoders to the
evaluation record on 2026-09-15T22:06:00Z, before calibration and replay, after
the gates of the usage doc's "Historical baselines" section all passed. The
manifest's `baselines` block is committed beside this report as
`hierarchical_m2_d9_p003/evaluation_baselines.json`. What it records:

| Gate | Result |
|---|---|
| Run directory | `/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha` |
| Run `manifest.json` sha256 | `77d44c3579dcb7baed6fd5352bc4ff4c43b4f282091f8f650082af06810872d5` |
| Run `results.json` sha256 | `a5a780305a4b3561c517f32b953c812b5aba5c085858decb7f1af3007c39af23` |
| Run `circuit.stim` / `model.dem` / packed payload sha256 | `d6f1f852…` / `858edcae…` / `d55da8f4…`, equal to the record's `sample_identity_inputs` and `parent_payload_sha256` |
| Run provenance | commit `10dc867b632edcf011f802f0f8a6e05ca769c659`, versions `{numpy 2.5.1, pymatching 2.4.0, scipy 1.18.0, sinter 1.16.0, stim 1.16.0}` |
| Row mapping | the record's parent row ids index the run's prediction rows (rows 0 to 99,999, all of them) |
| Yoke parity | 100,000 rows checked, 0 violations, all four baselines |
| Recorded joint MWPM versus collected `joint_mwpm` | agreement 1.0, 0 disagreements, 0 tie-explained, max cost difference 0.0 (tolerance 1e-06) |
| Importer | check identity `728ed4b5…`, commit `44c3bce9…`, `{numpy 2.5.1, pymatching 2.4.0}` |

Every prediction file was re-hashed as little-endian bit-packed rows and
required to equal the `prediction_packed_sha256` the run's `results.json`
declares:

| Baseline | File | Verified `prediction_packed_sha256` |
|---|---|---|
| `joint_mwpm_recorded` | `mwpm_predictions.npy` | `f081813d6ec90e9bfc0431bb544e1e193e3d0d1ed39f96bde2c4b896a227e666` |
| `joint_uf` | `uf_predictions.npy` | `48390885502a6e0d5c5081af3c1e2cb2ff49d95c0f5697045af366e7a7e5c078` |
| `joint_correlated_mwpm` | `correlated_mwpm_predictions.npy` | `deb795e0f7087716282de87dd1b3b75cb0af77a4b70fd40a4e3235e8bc7f7d20` |
| `joint_correlated_uf` | `correlated_uf_predictions.npy` | `3e26339a37a839a490a2a70adb73a928aaab81941b8a932b3038e30ebf033877` |

These are the four hashes the pilot's audit script verified; the pilot's
`baselines_pilot.json` read the same files.

The summary's "Baselines on the same shots" table, from `summary_full.json`:
the recorded run's decoders (`historical`), the collector's recomputed
`joint_mwpm` (`collected`), and every replayed cell (`hierarchical`), each
scored on the same 100,000 shots, with 95% bootstrap intervals of the rate over
the same resampled shots (10,000 replicates, seed 43) and the normalized LER
per patch per round (`pieces=216, values=8`).

| Decoder | Kind | Failed shots / 100,000 | Block failure | 95% interval | Normalized LER |
|---|---|---:|---:|---|---:|
| `joint_mwpm_recorded` | historical | 25,660 | 0.25660 | (0.25395, 0.25928) | 1.398106e-3 |
| `joint_uf` | historical | 39,847 | 0.39847 | (0.39543, 0.40150) | 2.429672e-3 |
| `joint_correlated_mwpm` | historical | 6,925 | 0.06925 | (0.06767, 0.07078) | 3.336844e-4 |
| `joint_correlated_uf` | historical | 14,247 | 0.14247 | (0.14031, 0.14462) | 7.182492e-4 |
| `joint_mwpm` | collected | 25,660 | 0.25660 | (0.25395, 0.25928) | 1.398106e-3 |
| `mwpm:gap_plain->gap_correlated:initial_only:mixed` | hierarchical | 25,660 | 0.25660 | (0.25396, 0.25926) | 1.398106e-3 |
| `mwpm:gap_plain->gap_correlated:all_refined:mixed` | hierarchical | 6,782 | 0.06782 | (0.06626, 0.06935) | 3.265157e-4 |
| `uf:cluster_gap->gap_correlated:initial_only:mixed` | hierarchical | 32,809 | 0.32809 | (0.32522, 0.33096) | 1.887031e-3 |
| `uf:cluster_gap->gap_correlated:all_refined:mixed` | hierarchical | 6,820 | 0.06820 | (0.06663, 0.06975) | 3.284195e-4 |
| `uf:cluster_gap->gap_plain:initial_only:mixed` | hierarchical | 32,809 | 0.32809 | (0.32522, 0.33096) | 1.887031e-3 |
| `uf:cluster_gap->gap_plain:all_refined:mixed` | hierarchical | 25,612 | 0.25612 | (0.25346, 0.25878) | 1.395005e-3 |

The four historical rates are exactly the four-decoder comparison's 100,000-shot
rates (25.660%, 39.847%, 6.925%, 14.247%,
[`decoder_comparison_d9_p003_100k.md`](decoder_comparison_d9_p003_100k.md)),
as they must be: same arrays, same shots. The recorded and the recomputed joint
MWPM agree on every row, so their two table rows are identical.

Paired block-failure differences of each hierarchical cell against
`joint_mwpm_recorded`, resampling whole shots (cell minus recorded joint MWPM):

| Cell | Difference | 95% interval |
|---|---:|---|
| `mwpm:gap_plain->gap_correlated:initial_only:mixed` | 0.000000 | (-0.000120, 0.000130) |
| `mwpm:gap_plain->gap_correlated:all_refined:mixed` | -0.188780 | (-0.191470, -0.186090) |
| `uf:cluster_gap->gap_correlated:initial_only:mixed` | 0.071490 | (0.069030, 0.073980) |
| `uf:cluster_gap->gap_correlated:all_refined:mixed` | -0.188400 | (-0.191100, -0.185690) |
| `uf:cluster_gap->gap_plain:initial_only:mixed` | 0.071490 | (0.069030, 0.073980) |
| `uf:cluster_gap->gap_plain:all_refined:mixed` | -0.000480 | (-0.001150, 0.000190) |

Read plainly: the two `all_refined` cells that refine with the correlated gap
fail on 6,782 and 6,820 shots against the recorded joint MWPM's 25,660, with
paired differences whose intervals exclude zero; the UF-referenced
`initial_only` cell is worse than joint MWPM by 0.071490 with an interval
excluding zero; and the plain-gap refinement of the UF reference lands at the
joint MWPM's rate (difference -0.000480, interval including zero), as does the
MWPM-referenced `initial_only` cell, which reproduces the joint MWPM's failed
shot count exactly with a paired interval spanning zero. No paired difference
against `joint_correlated_mwpm` is computed by the summary, so no claim about
the hierarchy relative to the correlated joint decoder is made here beyond the
rates and rate intervals in the table.

## 8. The M3 gate, and the decision

**Decision: proceed to M3.** The plan's criterion is a negative paired
all-refined minus initial-only difference on the pooled misattribution rate of
the primary pair (`uf:cluster_gap -> gap_correlated`, mixed) with a 95%
interval excluding zero. The measured difference on the 100,000-shot evaluation
set is **-0.305359** with 95% interval **(-0.309384, -0.301485)** over
**61,295 eligible shot-sectors** and no zero-denominator replicate, so the
criterion is met. The block-failure effect for the same pair is **-0.259890**,
95% interval **(-0.262810, -0.256990)**, over 100,000 shots, and the same
cell's paired block-failure difference against the recorded joint MWPM is
**-0.188400**, 95% interval **(-0.191100, -0.185690)** (section 7). The replay
work that buys this is, per shot, 6 incremental unforced plain matchings, 6
reweight passes and 24 correlated forced matchings on top of the shared initial
6 UF decodes and 12 Dijkstra searches (section 5). The secondary pair and the
MWPM control move in the same direction with intervals excluding zero.

This decision is exploratory. It is measured on the evaluation set with
calibrators fitted on a disjoint sampling family, the analysis is not frozen,
and no confirmation set exists; M3 adds selective refinement policies and
random controls on these same records, and none of the endpoints above is a
confirmed result until a confirmation-role collection is made after the
analysis freeze.

## 9. Reproduction

Run from the repository root with `PYTHONPATH=src`, at commit
`44c3bce9af4cc44e7e004f02a378a946529b7774`, with
`OUT2="$TMPDIR/hier-d9-p003-m2"`, the pilot's run directory
`OUT="$TMPDIR/hier-d9-p003"` (for the saved calibration sample), and
`RUN=/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha`. See
[the usage doc](../hierarchical_decoding.md) for what each stage writes. The
copies of the calibration sample come first, so that the calibration records
decode the pilot's saved shots rather than a resample; the collect stage
verifies the copied sample against the request before decoding.

```bash
mkdir -p "$OUT2/calibration" "$OUT2/calibration_pilot_rows"
cp -r "$OUT/calibration/sample" "$OUT2/calibration/sample"
cp -r "$OUT/calibration/sample" "$OUT2/calibration_pilot_rows/sample"

.venv/bin/python tools/hierarchical_experiment collect --out "$OUT2/evaluation" --role evaluation \
    --recorded-run "$RUN" --workers 32 --chunk-size 50
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT2/calibration" --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --workers 32 --chunk-size 50
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT2/evaluation_pilot_rows" --role evaluation \
    --recorded-run "$RUN" --rows 0:2000 --workers 32 --chunk-size 50
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT2/calibration_pilot_rows" --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --rows 0:2000 --workers 32 --chunk-size 50

.venv/bin/python tools/hierarchical_experiment verify-subset --full "$OUT2/evaluation" \
    --subset "$OUT2/evaluation_pilot_rows" --out "$OUT2/verify_evaluation.json"
.venv/bin/python tools/hierarchical_experiment verify-subset --full "$OUT2/calibration" \
    --subset "$OUT2/calibration_pilot_rows" --out "$OUT2/verify_calibration.json"

.venv/bin/python tools/hierarchical_experiment import-baselines --record "$OUT2/evaluation" \
    --recorded-run "$RUN"
.venv/bin/python tools/hierarchical_experiment calibrate --record "$OUT2/calibration" \
    --out "$OUT2/calibrators_full.json" \
    --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated
.venv/bin/python tools/hierarchical_experiment replay --record "$OUT2/evaluation" \
    --calibrators "$OUT2/calibrators_full.json" --out "$OUT2/replay_full" \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=all_refined,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=all_refined,outer=mixed
.venv/bin/python tools/hierarchical_experiment summarize --replays "$OUT2/replay_full" \
    --out "$OUT2/summary_full.md" --replicates 10000 --seed 43
```

The order matters: `verify-subset` for the evaluation role ran against the
full record as collected, and `import-baselines` ran after it and before
`calibrate` and `replay`, so the calibrators, the replay manifest and the
summary name the post-import record hash `5f061fa6…`. Bootstrap: 10,000
replicates, seed 43, resampling whole shots; both are recorded in
`summary_full.manifest.json` under `bootstrap`.

**Resumptions.** None. No collection was interrupted or resumed, and every
directory was collected once, from empty (the two calibration directories held
only the copied `sample/` before collection started).

**Artifacts.** Samples, records, per-configuration replay arrays and the
calibration artifact stay under `$TMPDIR`; only this report and the four
machine-readable files it cites are committed.

- `/data2/s2chitni/.tmp/hier-d9-p003-m2/evaluation/manifest.json` (post-import,
  sha256 `f84a4231…`; `record.npz` `5f061fa6…`)
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/calibration/manifest.json` (sha256
  `ac641027…`; `record.npz` `ec042a53…`)
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/evaluation_pilot_rows/manifest.json`
  (`record.npz` `8cbc6a3a…`, the pilot's)
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/calibration_pilot_rows/manifest.json`
  (`record.npz` `9612c5f2…`, the pilot's)
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/verify_evaluation.json`,
  `verify_calibration.json`
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/calibrators_full.json`
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/replay_full/replay_manifest.json`
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/summary_full.md`, `summary_full.json`,
  `summary_full.manifest.json` (sha256 `201c605a…`, `24883efa…`, `9868e1f3…`)
- `/data2/s2chitni/.tmp/hier-d9-p003-m2/*.log` (one per command, each with the
  command line, start and end times in UTC, wall clock and exit status)

Committed beside this report:

- [`hierarchical_m2_d9_p003/summary_full.json`](hierarchical_m2_d9_p003/summary_full.json)
  — every number the section 4, 5 and 7 tables print, sha256
  `24883efa614983ab3f055a566880c2a4f0ed28329c3dde2c88e13a406def5437`
  (byte-identical to the run's copy)
- [`hierarchical_m2_d9_p003/verify_evaluation.json`](hierarchical_m2_d9_p003/verify_evaluation.json)
  — the section 3 evaluation check, sha256
  `85e988a9891ce4cef71e8412a70c138415ec217ffd6b52159d3f8acbe123c5ae`
- [`hierarchical_m2_d9_p003/verify_calibration.json`](hierarchical_m2_d9_p003/verify_calibration.json)
  — the section 3 calibration check, sha256
  `da6f93bb7bf575c4c04641125f735ede1a420bea68f7101a2a63b21e8cb59162`
- [`hierarchical_m2_d9_p003/evaluation_baselines.json`](hierarchical_m2_d9_p003/evaluation_baselines.json)
  — the `baselines` block of the post-import evaluation manifest, copied
  verbatim, with the manifest's `artifacts`, `identities`, `role`, `rows`,
  `shots`, `parent_shots` and `parent_payload_sha256` copied beside it and the
  source manifest's path and sha256 recorded, sha256
  `057446a5081628f711a7aaaa819b8feb86a90a70af8b82216399372757b44091`

The pilot report, [`hierarchical_pilot_d9_p003.md`](hierarchical_pilot_d9_p003.md),
remains the record of the 2,000-row run and of the decision that started M2;
its two `record.npz` hashes are the ones section 3 reproduces.
