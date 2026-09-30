# Four-way distance sweep: normalized LER and latency

Six rotated surface-code patches; two ideal yokes; SI1000 p=0.003 (0.3%); CZ extraction; r=4d rounds; ideal time boundaries. There are 100,000 paired shots per distance. All variants use the same plain PyMatching outer MWPM.

The four configurations are correlated MWPM + complementary gap, correlated MWPM + MPP cluster gap, correlated UF + cluster gap, and BP5 + single-pass weighted UF + cluster gap. BP5 uses five sum-product flooding iterations, damping 0.5 and LLR clip 30, with the existing posterior-to-edge projection; all confidence scores are uncapped and uncalibrated.

## Normalized logical error rates

| d | r | cMWPM + complementary | cMWPM + cluster | cUF + cluster | BP5 + weighted UF + cluster |
|---:|---:|---:|---:|---:|---:|
| 7 | 28 | 0.000895638 | 0.000920882 | 0.00115289 | 0.000884183 |
| 9 | 36 | 0.000332531 | 0.000349872 | 0.000493766 | 0.000336596 |
| 11 | 44 | 0.000120845 | 0.000128469 | 0.00020752 | 0.000126424 |
| 13 | 52 | 4.21684e-05 | 4.5423e-05 | 8.47033e-05 | 4.58463e-05 |
| 15 | 60 | 1.45985e-05 | 1.56885e-05 | 3.47425e-05 | 1.66111e-05 |

Values are effective LER per physical patch per round, computed as `sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=6*r, values=8)`. The measured failure event is any tracked patch observable being wrong after L2 for the whole six-patch block. This is the established effective normalization, not a directly measured independent patch hazard. Exact binomial confidence limits are transformed through the same function. Zero failures, if present, retain nonzero upper limits.

| d | Configuration | Failures / 100,000 | Normalized LER | Exact normalized 95% CI |
|---:|---|---:|---:|---:|
| 7 | cMWPM + complementary gap | 13,854 | 0.000895638 | 0.000880634–0.000910832 |
| 7 | cMWPM + cluster gap | 14,212 | 0.000920882 | 0.000905646–0.000936309 |
| 7 | cUF + cluster gap | 17,425 | 0.00115289 | 0.0011356–0.00117037 |
| 7 | BP5 + weighted UF + cluster gap | 13,691 | 0.000884183 | 0.000869286–0.00089927 |
| 9 | cMWPM + complementary gap | 6,902 | 0.000332531 | 0.000324695–0.000340507 |
| 9 | cMWPM + cluster gap | 7,247 | 0.000349872 | 0.000341823–0.000358062 |
| 9 | cUF + cluster gap | 10,055 | 0.000493766 | 0.000484094–0.000503582 |
| 9 | BP5 + weighted UF + cluster gap | 6,983 | 0.000336596 | 0.00032871–0.000344622 |
| 11 | cMWPM + complementary gap | 3,134 | 0.000120845 | 0.000116642–0.00012516 |
| 11 | cMWPM + cluster gap | 3,328 | 0.000128469 | 0.000124132–0.000132919 |
| 11 | cUF + cluster gap | 5,314 | 0.00020752 | 0.000201958–0.000213196 |
| 11 | BP5 + weighted UF + cluster gap | 3,276 | 0.000126424 | 0.000122122–0.000130838 |
| 13 | cMWPM + complementary gap | 1,306 | 4.21684e-05 | 3.99102e-05–4.4521e-05 |
| 13 | cMWPM + cluster gap | 1,406 | 4.5423e-05 | 4.30773e-05–4.78631e-05 |
| 13 | cUF + cluster gap | 2,604 | 8.47033e-05 | 8.14754e-05–8.80262e-05 |
| 13 | BP5 + weighted UF + cluster gap | 1,419 | 4.58463e-05 | 4.34895e-05–4.82976e-05 |
| 15 | cMWPM + complementary gap | 524 | 1.45985e-05 | 1.33747e-05–1.59041e-05 |
| 15 | cMWPM + cluster gap | 563 | 1.56885e-05 | 1.44187e-05–1.70401e-05 |
| 15 | cUF + cluster gap | 1,242 | 3.47425e-05 | 3.28354e-05–3.67314e-05 |
| 15 | BP5 + weighted UF + cluster gap | 596 | 1.66111e-05 | 1.53036e-05–1.80005e-05 |

All six paired comparisons at every distance are in results.json, with 10,000 whole-shot bootstrap replicates for block and normalized differences. McNemar p-values are exploratory and unadjusted. The d=7 accuracy result is reused exactly from the preceding corrected experiment; larger distances use new saved samples.

## CPU latency

Each cell is **median / p99 milliseconds per full six-patch block**.

| d | cMWPM + complementary | cMWPM + cluster | cUF + cluster | BP5 + weighted UF + cluster |
|---:|---:|---:|---:|---:|
| 7 | 42.966 / 47.456 | 3.699 / 4.065 | 27.801 / 31.955 | 143.458 / 243.065 |
| 9 | 101.738 / 121.527 | 7.341 / 8.364 | 62.598 / 68.782 | 329.986 / 349.876 |
| 11 | 166.935 / 354.287 | 9.715 / 22.895 | 93.951 / 201.582 | 415.532 / 853.404 |
| 13 | 412.827 / 862.121 | 16.613 / 32.600 | 193.678 / 432.815 | 733.409 / 1504.575 |
| 15 | 1046.054 / 1692.387 | 25.811 / 40.363 | 309.301 / 519.571 | 1184.755 / 1865.685 |

Latency uses 1,000 uniformly selected saved shots per distance, the same rows for every variant, 32 warm-up rows, batch size one and one CPU thread pinned to one physical core. Variant order is balanced and interleaved. All six patches are decoded serially within a block. The timer includes local input preparation, L1, confidence extraction and L2. Simulation, file I/O, construction, compilation and external post-decode correctness checks are excluded. The collection jobs are finished before latency measurement starts.

These are measurements of the current software implementation, including Python adapters and built-in validation. The complementary-gap adapter reconstructs frozen weighted matching graphs; that work is included. They are not pure PyMatching-engine benchmarks, hardware latency estimates or online streaming deadlines. Every timed prediction is compared with the corresponding accuracy-run prediction after the timer. Raw wall/process times, row IDs and variant order are retained in d*/latency.npz.

latency.csv additionally reports mean, p95, empirical maximum, process time and amortized mean microseconds per patch-round. Dividing elapsed block time by 6*r is only an amortized work measure. The p99 estimates are empirical tails from 1,000 samples, not worst-case bounds.

## Implementation and reproducibility

PyMatching **2.4.0** supplies the native correlated MWPM and all forced complementary matches. The MPP extension compiles the same pinned PyMatching 2.4.0 engine, commit `6f63b2b9474ba0fa7e511fe52bffdce858a06984`, to expose matching-radius confidence. MWPM variants are required to agree on each local reference prediction and chosen-class matching cost during collection.

Stim **1.17.dev0** is the Dante fork at `db493f5987514ff4724d972bdbd2597038f9e9a0`. Its installed native binary hash is verified. New sampling seeds are 202609290700 + d. The unchanged d=7 seed is 202609290707.

Accuracy collection uses 32 MWPM processes followed by 32 OpenMP threads for UF/BP, on the same 32 distinct physical cores. Collection wall time is recorded separately and is not used to infer single-shot latency. UF adapters are checked against independent Python oracles, prior native implementations, one-versus-32-thread execution and the separate dispatch used for latency. All corrections and final yoke parities are checked.

launch.json, uf_build.json, source_manifest.json and source_snapshot/ record dependencies, revisions, original local changes, commands and binary hashes. Each d*/ directory retains the exact sample, all four paired predictions and confidence values, detailed uncertainty and validation records. See protocol.md and the retained experiment recipes.
