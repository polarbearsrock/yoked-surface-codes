# Hierarchical L1/L2 pilot at d=9, p=0.003: M1 endpoints and decision

**Every result on this page is exploratory.** It is a 2,000-shot pilot whose
purpose is to decide whether milestone M2 (full collection) starts; it is not a
confirmation run, the analysis is not frozen, and confirmation-role collection
is refused by the tooling at this milestone.

On 2,000 held-out shots, the primary endpoint pair — UF cluster gap as the
initial estimator, correlated matching gap as the refined one, mixed outer rule
— moved the pooled misattribution rate from **0.374688 [450 / 1201]** under
`initial_only` to **0.052456 [63 / 1201]** under `all_refined`. The paired
all-refined minus initial-only difference is **-0.322231**, 95% percentile
interval **(-0.351237, -0.293117)**, which is negative and excludes zero. Block
failure moved from 0.3335 to 0.0655, a paired difference of **-0.268**, 95%
interval **(-0.288500, -0.247500)**. **Decision: proceed to M2** (section 7).

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
| Outer rule | `mixed` (the only rule replayed at M1) |
| Policies | the two endpoints `initial_only` and `all_refined` |
| Normalization | `pieces = 6 x 36 = 216`, `values = 2 x (6 - 2) = 8` |
| Bootstrap | 10,000 replicates resampling whole shots, seed 43 |
| Run date | 2026-09-15 UTC |

**Parent samples and rows.** Both records decode parent rows **0 to 1,999**
(row-id sha256 `55f385cf2332d9056aaed6f496e7bebd2df52c6a9547ce2144b309432d4b0290`
on both sides) of one full Stim call each. Rows are not an identity input, so
each record keeps its parent's identity and can be extended in M2.

| | Calibration | Evaluation |
|---|---|---|
| Source | generated | imported from the recorded four-decoder run |
| Seed | 142 | 42 |
| Shots in the full call | 50,000 | 100,000 |
| Rows decoded | 0:2000 | 0:2000 |
| Packed payload sha256 | `05f2cdf043a4974a2a972213ee3042f05fe16e254b408b59b1b585b2f8507f2f` | `d55da8f4c9b8287fa0af64f8382de755a243a774030499e108c348b665e102f5` |
| Model identity | `73751a5aa99ac806a4ad5007b310b9a22d3f39c6a4f609d466bf83295268cd9d` | `73751a5aa99ac806a4ad5007b310b9a22d3f39c6a4f609d466bf83295268cd9d` |
| Parent sample identity | `e9ace633bc9ff34b2dc228d63393f4daf58c4f4acbde3c678669b8b9645d8398` | `d1bc5fcf04da0c265f67b23637ee518f730cebc86700d04db09fec097a5130f1` |
| Sampling family identity | `060a8f93813c288d32cf5b8801cfd74c052df1c08092c1e07df152c4d5498933` | `51918c6cd9505f780bd9ef6a6b029a52bb3f68ae81a6ea195f4dbe185668bf69` |
| Decoder identity | `af532393733d652a58ea69e69363ec61150dbbd8193b7d560c23ce69eb4a49a0` | `af532393733d652a58ea69e69363ec61150dbbd8193b7d560c23ce69eb4a49a0` |
| Collection identity | `91fddd2b6a1046a5d85f671b28fb82992711d5c2962338aad40178c07a51d7e5` | `6db1561bbcc8a3d2dfb94608076bce787877bc298396010b59e7d8d84a41a498` |
| `record.npz` sha256 | `9612c5f2b79b6834163cf3b3df259ae30356112d5236c8aee405326032668f05` | `8cbc6a3ae9bd818dc74b3d075093aa257bf4f72148a5a9931c16986b0b2dea1b` |

The two records share one model and one decoder, and differ in both parent
sample and sampling family, which is what the replay stage requires before it
will evaluate a calibration on held-out shots.

**Original circuit and DEM.** The evaluation sample's `circuit.stim` and
`model.dem` are the bytes of the recorded run
`/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha`, verified equal
to that run's declared hashes by the audit of section 6:

```text
circuit.stim  d6f1f852b5d12ca2816c2411fe0dcfb0d2948ea50d6eb12655a9cfd775976d2b
model.dem     858edcae41ba5f5ead2862d28dd517d5f22734b1ebd17c0dc12139ecd6a6278a
```

The generated calibration sample rebuilds that same circuit and DEM from
today's generator and hashes to the same values, which is why the two model
identities agree.

**Calibration artifact.** `calibrators_pilot.json`, identity
`19b0390daa341b30dfe7c2f81b05bdfccd706aeba82f3060c44afd93b6eb2c52`, sha256
`273bb3d7e0f2e31d1a7ecfdbcf90ae09e7336c3ed8001f95a308d7644cca57da`. One
isotonic fit per estimator and sector, the six patches of a sector pooled.
Knot convention, as recorded in the artifact:

```text
pav-blocks-of-distinct-scores; knot at each block weighted-mean score;
linear between knots; constant beyond; clipped to [1e-6, 1 - 1e-6]
```

All five estimators are fitted `decreasing` in the residual-error probability.
Knots per estimator (X, Z): `uf:cluster_gap` (2751, 1359), `uf:gap_plain`
(2139, 2009), `uf:gap_correlated` (3386, 6909), `mwpm:gap_plain` (2118, 1993),
`mwpm:gap_correlated` (3307, 6825).

**Code and packages.** The calibration record, the calibrators, the replays and
the summary were produced at commit
`0078e0589bc16a97fd29d9128a7748c654f289fb`; the evaluation and preflight
records were decoded at its parent `f2b5d19744d757c68d88486e656de574a2127852`
and revalidated and republished at `0078e05` (section 5). Python 3.14.5;
decoder packages `{numpy 2.5.1, pymatching 2.4.0, scipy 1.18.0, stim 1.16.0}`,
check packages `{numpy 2.5.1, pymatching 2.4.0}`, sampling `{stim 1.16.0}`,
summary `{numpy 2.5.1, sinter 1.16.0}`. Replay identity
`b331a585dc30ceba43c2d474567f99df2fe09cd68bef7c7d0670d469bc15600a`; replay tie
rule "log-weight ties within 1e-09 break to the lowest binary pattern".

## 2. Graph and record checks

The graph gate runs before any row is decoded and is identical on all three
collections (the 50-row preflight, calibration, evaluation):

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

Record checks, run before publication:

| Quantity | Preflight (50) | Calibration (2,000) | Evaluation (2,000) |
|---|---:|---:|---:|
| Value violations | 0 | 0 | 0 |
| Check parity violations (`Hc = s`) | 0 | 0 | 0 |
| Final parity violations | 0 | 0 | 0 |
| Plain additivity max error | 2.274e-13 | 2.274e-13 | 2.274e-13 |
| Correlated additivity max error | 2.274e-13 | 2.274e-13 | 2.274e-13 |
| Plain preferred-sign disagreements | 0 | 0 | 0 |
| Correlated preferred-sign disagreements | 0 | 0 | 0 |
| Joint agreement | 1.0 | 1.0 | 0.9995 |
| Joint disagreements | 0 | 0 | 1 |
| Tie-explained disagreements | 0 | 0 | 1 |
| **Unexplained disagreements** | **0** | **0** | **0** |
| Max disagreement cost difference | 0.0 | 0.0 | 0.0 |

The single joint disagreement in the 2,000 evaluation rows is a tie: its cost
difference is 0.0 and it is fully tie-explained. No collection recorded any
failure or diagnostic entry.

## 3. Endpoint tables

Rates carry their denominators; intervals are 95% percentile intervals from
10,000 whole-shot bootstrap replicates at seed 43. Misattribution is measured
on patch-sectors where the fixed reference made exactly one mistake; block
failure is measured on shots. No statistic in this pilot was `unavailable`, and
no comparison had any zero-denominator replicate.

### 3.1 Primary: `uf:cluster_gap -> gap_correlated`, mixed

| Quantity | initial_only:mixed | all_refined:mixed |
|---|---|---|
| Block failure | 0.3335 [667 / 2000] (0.312500, 0.354012) | 0.0655 [131 / 2000] (0.055000, 0.076500) |
| Normalized LER (per patch per round) | 1.926406e-3 | 3.149115e-4 |
| Misattribution (pooled) | 0.374688 [450 / 1201] (0.346154, 0.403376) | 0.052456 [63 / 1201] (0.040134, 0.065310) |
| Misattribution (X) | 0.387043 [233 / 602] | 0.053156 [32 / 602] |
| Misattribution (Z) | 0.362270 [217 / 599] | 0.051753 [31 / 599] |
| Ties (pooled) | 0.00025 [1 / 4000] | 0.00025 [1 / 4000] |
| Probabilities above one half | 0.008875 [213 / 24000] | 0.066458 [1595 / 24000] |

| Metric | Paired difference | 95% interval | Eligible | Zero-denominator replicates |
|---|---:|---|---:|---:|
| Misattribution (pooled) | **-0.322231** | **(-0.351237, -0.293117)** | 1,201 | 0 |
| Block failure | **-0.268000** | **(-0.288500, -0.247500)** | 2,000 | 0 |

Final sector failures by residual reference-failure stratum (4,000 shot-sectors):

| Stratum | Eligible sectors | initial_only:mixed | all_refined:mixed |
|---|---:|---|---|
| zero | 2,508 | 0.011563 [29 / 2508] | 0.005981 [15 / 2508] |
| one | 1,201 | 0.374688 [450 / 1201] | 0.052456 [63 / 1201] |
| multiple | 291 | 0.951890 [277 / 291] | 0.189003 [55 / 291] |

### 3.2 Secondary: `uf:cluster_gap -> gap_plain`, mixed

| Quantity | initial_only:mixed | all_refined:mixed |
|---|---|---|
| Block failure | 0.3335 [667 / 2000] (0.312500, 0.354012) | 0.2585 [517 / 2000] (0.239000, 0.278000) |
| Normalized LER (per patch per round) | 1.926406e-3 | 1.410401e-3 |
| Misattribution (pooled) | 0.374688 [450 / 1201] (0.346154, 0.403376) | 0.258951 [311 / 1201] (0.234309, 0.284450) |
| Misattribution (X) | 0.387043 [233 / 602] | 0.254153 [153 / 602] |
| Misattribution (Z) | 0.362270 [217 / 599] | 0.263773 [158 / 599] |
| Ties (pooled) | 0.00025 [1 / 4000] | 0.00025 [1 / 4000] |
| Probabilities above one half | 0.008875 [213 / 24000] | 0.036292 [871 / 24000] |

| Metric | Paired difference | 95% interval | Eligible | Zero-denominator replicates |
|---|---:|---|---:|---:|
| Misattribution (pooled) | -0.115737 | (-0.143472, -0.087212) | 1,201 | 0 |
| Block failure | -0.075000 | (-0.092000, -0.057500) | 2,000 | 0 |

| Stratum | Eligible sectors | initial_only:mixed | all_refined:mixed |
|---|---:|---|---|
| zero | 2,508 | 0.011563 [29 / 2508] | 0.009569 [24 / 2508] |
| one | 1,201 | 0.374688 [450 / 1201] | 0.258951 [311 / 1201] |
| multiple | 291 | 0.951890 [277 / 291] | 0.783505 [228 / 291] |

### 3.3 Control: `mwpm:gap_plain -> gap_correlated`, mixed

| Quantity | initial_only:mixed | all_refined:mixed |
|---|---|---|
| Block failure | 0.2565 [513 / 2000] (0.237000, 0.275500) | 0.0650 [130 / 2000] (0.054500, 0.076000) |
| Normalized LER (per patch per round) | 1.397459e-3 | 3.124148e-4 |
| Misattribution (pooled) | 0.322523 [358 / 1110] (0.295036, 0.350988) | 0.059459 [66 / 1110] (0.045966, 0.073530) |
| Misattribution (X) | 0.314545 [173 / 550] | 0.069091 [38 / 550] |
| Misattribution (Z) | 0.330357 [185 / 560] | 0.050000 [28 / 560] |
| Ties (pooled) | 0.00075 [3 / 4000] | 0 [0 / 4000] |
| Probabilities above one half | 0.002583 [62 / 24000] | 0.057583 [1382 / 24000] |

| Metric | Paired difference | 95% interval | Eligible | Zero-denominator replicates |
|---|---:|---|---:|---:|
| Misattribution (pooled) | -0.263063 | (-0.291932, -0.235184) | 1,110 | 0 |
| Block failure | -0.191500 | (-0.210000, -0.172500) | 2,000 | 0 |

| Stratum | Eligible sectors | initial_only:mixed | all_refined:mixed |
|---|---:|---|---|
| zero | 2,685 | 0 [0 / 2685] | 0.006704 [18 / 2685] |
| one | 1,110 | 0.322523 [358 / 1110] | 0.059459 [66 / 1110] |
| multiple | 205 | 0.985366 [202 / 205] | 0.234146 [48 / 205] |

The two references define different eligible populations — 1,201 shot-sectors
for the UF reference, 1,110 for the MWPM one — so the three pairs are compared
only against their own `initial_only` cell, never against each other.

## 4. Replay work

These are the calls the specified replay procedure implies: the estimator
pair's fixed initial work charged once per patch, plus incremental work charged
once per distinct refined patch. They are neither the collection work of
section 5 nor elapsed time, and no latency or speedup claim follows from them.
Totals are over 2,000 shots; per-shot means are the totals divided by 2,000.

| Counter | `uf:cluster_gap -> gap_correlated` initial_only | per shot | all_refined | per shot |
|---|---:|---:|---:|---:|
| requested_patch_sectors | 0 | 0 | 24,000 | 12 |
| distinct_patches | 0 | 0 | 12,000 | 6 |
| initial_uf_calls | 12,000 | 6 | 12,000 | 6 |
| initial_dijkstra_searches | 24,000 | 12 | 24,000 | 12 |
| initial_dijkstra_states | 18,731,135 | 9,365.57 | 18,731,135 | 9,365.57 |
| initial_unforced_plain_calls | 0 | 0 | 0 | 0 |
| initial_plain_forced_calls | 0 | 0 | 0 | 0 |
| incremental_unforced_plain_calls | 0 | 0 | 12,000 | 6 |
| incremental_plain_forced_calls | 0 | 0 | 0 | 0 |
| incremental_reweight_passes | 0 | 0 | 12,000 | 6 |
| incremental_correlated_forced_calls | 0 | 0 | 48,000 | 24 |

| Counter | `uf:cluster_gap -> gap_plain` initial_only | per shot | all_refined | per shot |
|---|---:|---:|---:|---:|
| requested_patch_sectors | 0 | 0 | 24,000 | 12 |
| distinct_patches | 0 | 0 | 12,000 | 6 |
| initial_uf_calls | 12,000 | 6 | 12,000 | 6 |
| initial_dijkstra_searches | 24,000 | 12 | 24,000 | 12 |
| initial_dijkstra_states | 18,731,135 | 9,365.57 | 18,731,135 | 9,365.57 |
| incremental_plain_forced_calls | 0 | 0 | 48,000 | 24 |
| incremental_reweight_passes | 0 | 0 | 0 | 0 |
| incremental_correlated_forced_calls | 0 | 0 | 0 | 0 |

| Counter | `mwpm:gap_plain -> gap_correlated` initial_only | per shot | all_refined | per shot |
|---|---:|---:|---:|---:|
| requested_patch_sectors | 0 | 0 | 24,000 | 12 |
| distinct_patches | 0 | 0 | 12,000 | 6 |
| initial_uf_calls | 0 | 0 | 0 | 0 |
| initial_dijkstra_searches | 0 | 0 | 0 | 0 |
| initial_dijkstra_states | 0 | 0 | 0 | 0 |
| initial_unforced_plain_calls | 12,000 | 6 | 12,000 | 6 |
| initial_plain_forced_calls | 48,000 | 24 | 48,000 | 24 |
| incremental_reweight_passes | 0 | 0 | 12,000 | 6 |
| incremental_correlated_forced_calls | 0 | 0 | 48,000 | 24 |

Counters omitted from the second and third tables are zero in both of that
pair's cells.

## 5. Actual collection work

What really ran while the records were produced, from each collection's
`manifest.json`. Collection counts every score for every patch, because
collection computes them all once so that any configuration can be replayed
offline; they are therefore larger than any single configuration's replay
counts. Neither number is elapsed time.

| Counter | Preflight (50 rows) | Calibration (2,000 rows) | Evaluation (2,000 rows) |
|---|---:|---:|---:|
| Rows | 50 | 2,000 | 2,000 |
| Joint decodes | 50 | 2,000 | 2,000 |
| UF decodes | 300 | 12,000 | 12,000 |
| Dijkstra searches | 600 | 24,000 | 24,000 |
| Dijkstra states | 482,524 | 18,816,400 | 18,731,135 |
| Unforced plain matchings | 300 | 12,000 | 12,000 |
| Plain forced matchings | 1,200 | 48,000 | 48,000 |
| Correlated forced matchings | 1,200 | 48,000 | 48,000 |
| Reweight attempts | 300 | 12,000 | 12,000 |
| Correlated validation matchings | 300 | 12,000 | 12,000 |
| Resumptions | 0 | 0 | 0 |

| Timing | Preflight | Calibration | Evaluation |
|---|---:|---:|---:|
| Workers / chunk size | 2 / 25 | 16 / 25 | 16 / 25 |
| Collection seconds (`timing.seconds_this_run`) | 17.863 | 45.326 | 45.276 |
| Wall clock of the command | 30.71 s | 63.65 s | 58.17 s |
| Setup, graph gate and record gate (wall minus collection) | 12.8 s | 18.3 s | 12.9 s |
| Measured throughput | 2.799 rows/s (1.400 per worker) | 44.125 rows/s (2.758 per worker) | 44.174 rows/s (2.761 per worker) |

Each throughput is that collection's own rows divided by its own manifest's
`timing.seconds_this_run`, then divided by its worker count:
50 / 17.863 = 2.799 rows/s (2.799 / 2 = 1.400 per worker);
2000 / 45.326 = 44.125 rows/s (44.125 / 16 = 2.758 per worker);
2000 / 45.276 = 44.174 rows/s (44.174 / 16 = 2.761 per worker).
The calibration figures come from the current `$OUT/calibration/manifest.json`
(`timing.seconds_this_run` 45.32559), not from the superseded pre-fix
collection described below, whose manifest records 45.56216 s.

No collection was interrupted, none was resumed, and no `checkpoint.npz` or
`failed_checks.json` remains: `resumptions` is 0 everywhere and no attempted-work
telemetry was lost. The setup row is not decoder work; it is process start, the
sample load or generation, the per-worker rebuild of the d=9 decoders from the
19,665,395-byte DEM, the graph gate and the record gate.

Two operational notes belong with these numbers. First, the calibration record
was collected twice. The first collection (2026-09-15T06:50:28Z, commit
`f2b5d19`) was made before the sample-text newline fix
(`0078e05`, "Terminate generated sample text with Stim's file newline"); its
generated `circuit.stim` and `model.dem` were one byte short of Stim's
`to_file` convention, so its model identity did not match the imported
evaluation record's and `replay` correctly refused to pair them. That directory
was moved aside to `calibration_stale_no_newline` and the identical collect
command was rerun into a fresh `calibration` directory at commit `0078e05`. The
decoded values are unaffected: the recollected `record.npz` hashes to
`9612c5f2…`, the same value as the stale collection, and the parent payload
hash is unchanged. Second, the evaluation and preflight collections were rerun
after the fix as well; because they import the recorded run's model bytes
verbatim their model identity never changed, so both took the
revalidate-and-republish path — 12.93 s and 12.95 s, no rows recollected,
`record.npz` hash, `timing`, `collection_work` and `created_utc` all unchanged,
only the check identity and the check source hashes updated to the new
`_collect.py`. Every timing above is that of the run whose rows the published
record holds: for calibration that is the post-fix recollection (45.326 s), not
the superseded one (45.562 s), and for evaluation and preflight it is their
original collection, which the later revalidation left untouched. The 12.93 s
and 12.95 s revalidation passes decoded nothing and are excluded from the
throughput figures.

## 6. Historical baselines on the same rows

From `hierarchical_pilot_d9_p003/baselines_pilot.json`, which re-verifies the
recorded run's saved `circuit.stim` and `model.dem` hashes against the run
manifest **and** against the evaluation sample's copies, recomputes the packed
payload hash from the recorded arrays, checks it against both the run manifest
and the collection's `parent_payload_sha256`, and re-derives each decoder's
packed prediction hash from the `.npy` file rather than trusting it. All
assertions passed. Rates are on the pilot's 2,000 rows, normalized with
`pieces=216, values=8`.

| Decoder | Failed shots / 2,000 | Block failure | Normalized LER (per patch per round) | Verified `prediction_packed_sha256` |
|---|---:|---:|---:|---|
| Joint MWPM | 517 | 0.25850 | 1.410401e-3 | `f081813d6ec90e9bfc0431bb544e1e193e3d0d1ed39f96bde2c4b896a227e666` |
| Repository UF | 804 | 0.40200 | 2.458765e-3 | `48390885502a6e0d5c5081af3c1e2cb2ff49d95c0f5697045af366e7a7e5c078` |
| Correlated MWPM | 134 | 0.06700 | 3.224105e-4 | `deb795e0f7087716282de87dd1b3b75cb0af77a4b70fd40a4e3235e8bc7f7d20` |
| Correlated UF | 286 | 0.14300 | 7.211659e-4 | `3e26339a37a839a490a2a70adb73a928aaab81941b8a932b3038e30ebf033877` |

The recorded run's own provenance, carried into the audit: code commit
`10dc867b632edcf011f802f0f8a6e05ca769c659`, `results.json` sha256
`a5a780305a4b3561c517f32b953c812b5aba5c085858decb7f1af3007c39af23`, versions
`{stim 1.16.0, numpy 2.5.1, pymatching 2.4.0, sinter 1.16.0, scipy 1.18.0}`.
On all 100,000 shots of that run the same four decoders measured 25.660%,
39.847%, 6.925% and 14.247% block failure
([the four-decoder comparison](decoder_comparison_d9_p003_100k.md)); the
2,000-row rates above are a subset of those shots, not an independent
measurement.

## 7. Projected full collection cost, and the decision

Projections use the measured 16-worker pilot throughput of section 5, not an
assumed scaling. Per-worker throughput was 1.400 rows/s on 2 workers and 2.758
to 2.761 rows/s on 16, so the numbers below assume the 16-worker per-row cost
holds over the whole sample; setup adds 13 to 19 s per invocation and does not
scale with rows. The evaluation rows are projected at 44.174 rows/s and the
calibration rows at 44.125 rows/s, each from its own current manifest.

| Work | Rows | Projected collection time |
|---|---:|---:|
| Full evaluation set | 100,000 | 37.7 min (100000 / 44.174 = 2,264 s) |
| Evaluation rows remaining after the pilot | 98,000 | 37.0 min (98000 / 44.174 = 2,219 s) |
| Full calibration set | 50,000 | 18.9 min (50000 / 44.125 = 1,133 s) |
| Calibration rows remaining after the pilot | 48,000 | 18.1 min (48000 / 44.125 = 1,088 s) |

Sizes below are MiB (1,048,576 bytes). A record grows to about 62 MiB per
100,000 rows (`record.npz` is 1,307,462 bytes, 1.25 MiB, for 2,000), and each
saved sample keeps its packed detector array: 222,100,128 bytes (211.8 MiB) for
the 100,000-shot evaluation call and 111,050,128 bytes (105.9 MiB) for the
50,000-shot calibration call.

**Decision: proceed to M2.** The plan's criterion is a negative paired
all-refined minus initial-only difference on the pooled misattribution rate of
the primary pair (`uf:cluster_gap -> gap_correlated`, mixed) with a 95%
interval excluding zero. The measured difference is **-0.322231** with 95%
interval **(-0.351237, -0.293117)** over **1,201 eligible shot-sectors** and no
zero-denominator replicate, so the criterion is met. The block-failure effect
for the same pair is **-0.268000**, 95% interval **(-0.288500, -0.247500)**,
over 2,000 shots. The replay work that buys this is, per shot, 6 incremental
unforced plain matchings, 6 reweight passes and 24 correlated forced matchings
on top of the shared initial 6 UF decodes and 12 Dijkstra searches (section 4).
The secondary pair and the MWPM control move in the same direction with
intervals excluding zero. This decision is exploratory and rests on 2,000
shots; M2 collects the remaining rows and refits before any of these endpoints
is quoted as a result.

## 8. Reproduction

Run from the repository root with `PYTHONPATH=src`, at commit
`0078e0589bc16a97fd29d9128a7748c654f289fb`, with
`OUT="$TMPDIR/hier-d9-p003"` and
`RUN=/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha`. See
[the usage doc](../hierarchical_decoding.md) for what each stage writes.

```bash
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/preflight_evaluation" --role evaluation \
    --recorded-run "$RUN" --rows 0:50 --workers 2 --chunk-size 25
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/calibration" --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --rows 0:2000 --workers 16 --chunk-size 25
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/evaluation" --role evaluation \
    --recorded-run "$RUN" --rows 0:2000 --workers 16 --chunk-size 25
.venv/bin/python tools/hierarchical_experiment calibrate --record "$OUT/calibration" \
    --out "$OUT/calibrators_pilot.json" \
    --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated
.venv/bin/python tools/hierarchical_experiment replay --record "$OUT/evaluation" \
    --calibrators "$OUT/calibrators_pilot.json" --out "$OUT/replay_pilot" \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=all_refined,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=all_refined,outer=mixed
.venv/bin/python tools/hierarchical_experiment summarize --replays "$OUT/replay_pilot" \
    --out "$OUT/summary_pilot.md" --replicates 10000 --seed 43
.venv/bin/python "$OUT/baselines_pilot.py"
```

Bootstrap: 10,000 replicates, seed 43, resampling whole shots; both are
recorded in `summary_pilot.manifest.json` under `bootstrap`.

**Resumptions.** None. No collection was interrupted or resumed. The
calibration collection was *repeated* once for the provenance reason described
in section 5, which is a second collection into a new directory rather than a
resumption; the evaluation and preflight directories were revalidated and
republished in place without recollecting rows.

**Artifacts.** Samples, records, checkpoints, per-configuration replay arrays
and the calibration artifact stay under `$TMPDIR`; only this report and the two
machine-readable files it cites are committed.

- `/data2/s2chitni/.tmp/hier-d9-p003/preflight_evaluation/manifest.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/calibration/manifest.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/evaluation/manifest.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/calibrators_pilot.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/replay_pilot/replay_manifest.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/summary_pilot.md`, `summary_pilot.json`,
  `summary_pilot.manifest.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/baselines_pilot.py`, `baselines_pilot.json`
- `/data2/s2chitni/.tmp/hier-d9-p003/calibration_stale_no_newline/` (the
  superseded pre-fix calibration collection, kept for provenance)

Committed beside this report:

- [`hierarchical_pilot_d9_p003/summary_pilot.json`](hierarchical_pilot_d9_p003/summary_pilot.json)
  — every number the section 3 and 4 tables print, sha256
  `c739cab211809d0ec7fb173f0503ed31cc31b48a6a3ea7669a800b4870958b44`
- [`hierarchical_pilot_d9_p003/baselines_pilot.json`](hierarchical_pilot_d9_p003/baselines_pilot.json)
  — the section 6 audit, with the parent payload hash, the 2,000 row ids and
  the verified prediction hashes
