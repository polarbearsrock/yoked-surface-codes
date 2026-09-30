# BP5 and correlated UF versus correlated MWPM: 100,000 paired shots

Each distance uses all 100,000 rows of the same archived SI1000 p=0.003 sample: six patches, two ideal yokes, CZ circuits, rounds=4d, seed 42. All decoders see identical syndromes. A whole-shot failure means at least one of the 12 logical-observable predictions differs from truth.

**BP5 + UF** reproduces the original pilot exactly. **BP5 + correlated UF** adds the repository's prior-derived correlation discounts, based on the BP-weighted first correction, followed by a second unchanged UF pass. Both passes use the original syndrome. BP already includes correlated faults, so the additional hard-evidence pass can double-count information. The variants and parameters were fixed before this expanded run.

Correlated MWPM was recomputed on every row and verified against the archived predictions. Current correlated-UF predictions are reused from the identical verified samples and checked on predetermined rows. The BP pilot's 5,000 rows are included; the remaining 95,000 rows are reported separately below. These are historical evaluation samples, not newly seeded independent confirmation samples.

## Whole-shot failures out of 100,000

| d | Correlated UF | BP5 + UF | BP5 + correlated UF | Correlated MWPM |
|---:|---:|---:|---:|---:|
| 7 | 23,230 | 14,652 | 14,492 | 13,785 |
| 9 | 14,247 | 8,059 | 7,773 | 6,925 |
| 11 | 7,754 | 4,016 | 3,822 | 3,047 |
| 13 | 4,083 | 1,876 | 1,723 | 1,304 |

## Failure rates and marginal uncertainty

| d | Decoder | Whole-shot failure % (95% CI) | Ratio to MWPM, raw failure rate |
|---:|---|---:|---:|
| 7 | Correlated UF | 23.230 [22.969, 23.493] | 1.6852 |
| 7 | BP5 + UF | 14.652 [14.434, 14.873] | 1.0629 |
| 7 | BP5 + correlated UF | 14.492 [14.275, 14.712] | 1.0513 |
| 7 | Correlated MWPM | 13.785 [13.573, 14.000] | 1.0000 |
| 9 | Correlated UF | 14.247 [14.032, 14.465] | 2.0573 |
| 9 | BP5 + UF | 8.059 [7.892, 8.229] | 1.1638 |
| 9 | BP5 + correlated UF | 7.773 [7.609, 7.941] | 1.1225 |
| 9 | Correlated MWPM | 6.925 [6.769, 7.084] | 1.0000 |
| 11 | Correlated UF | 7.754 [7.590, 7.921] | 2.5448 |
| 11 | BP5 + UF | 4.016 [3.896, 4.139] | 1.3180 |
| 11 | BP5 + correlated UF | 3.822 [3.705, 3.943] | 1.2543 |
| 11 | Correlated MWPM | 3.047 [2.942, 3.155] | 1.0000 |
| 13 | Correlated UF | 4.083 [3.962, 4.207] | 3.1311 |
| 13 | BP5 + UF | 1.876 [1.794, 1.962] | 1.4387 |
| 13 | BP5 + correlated UF | 1.723 [1.644, 1.806] | 1.3213 |
| 13 | Correlated MWPM | 1.304 [1.236, 1.376] | 1.0000 |

## Paired comparison against correlated MWPM

A repair is an MWPM failure corrected by the BP variant. A regression is an MWPM success that the BP variant fails. Positive excess means more failures than MWPM. Differences use pairing, not overlapping marginal confidence intervals.

| d | Decoder | Repairs | Regressions | Excess failure rate, pp (95% CI) | McNemar p |
|---:|---|---:|---:|---:|---:|
| 7 | BP5 + UF | 5,634 | 6,501 | 0.867 [0.651, 1.083] | 3.701e-15 |
| 7 | BP5 + correlated UF | 5,672 | 6,379 | 0.707 [0.492, 0.922] | 1.251e-10 |
| 9 | BP5 + UF | 3,411 | 4,545 | 1.134 [0.959, 1.309] | 4.361e-37 |
| 9 | BP5 + correlated UF | 3,443 | 4,291 | 0.848 [0.676, 1.020] | 5.378e-22 |
| 11 | BP5 + UF | 1,711 | 2,680 | 0.969 [0.839, 1.099] | 1.033e-48 |
| 11 | BP5 + correlated UF | 1,734 | 2,509 | 0.775 [0.647, 0.903] | 9.825e-33 |
| 13 | BP5 + UF | 811 | 1,383 | 0.572 [0.480, 0.664] | 1.477e-34 |
| 13 | BP5 + correlated UF | 830 | 1,249 | 0.419 [0.330, 0.508] | 3.632e-20 |

## Effect of the additional correlation pass

Here repairs and regressions compare BP5 + correlated UF with BP5 + UF. Positive excess means that the extra pass hurts accuracy.

| d | Repairs | Regressions | Excess failure rate, pp (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 7 | 2,039 | 1,879 | -0.160 [-0.283, -0.037] | 0.01107 |
| 9 | 1,508 | 1,222 | -0.286 [-0.388, -0.184] | 4.777e-08 |
| 11 | 955 | 761 | -0.194 [-0.275, -0.113] | 3.105e-06 |
| 13 | 553 | 400 | -0.153 [-0.213, -0.093] | 8.062e-07 |

## Improvement over current correlated UF

| d | Decoder | Failure-count reduction | Repairs | Regressions |
|---:|---|---:|---:|---:|
| 7 | BP5 + UF | 36.93% | 12,558 | 3,980 |
| 7 | BP5 + correlated UF | 37.62% | 12,580 | 3,842 |
| 9 | BP5 + UF | 43.43% | 8,969 | 2,781 |
| 9 | BP5 + correlated UF | 45.44% | 9,062 | 2,588 |
| 11 | BP5 + UF | 48.21% | 5,419 | 1,681 |
| 11 | BP5 + correlated UF | 50.71% | 5,460 | 1,528 |
| 13 | BP5 + UF | 54.05% | 3,111 | 904 |
| 13 | BP5 + correlated UF | 57.80% | 3,150 | 790 |

## Historical normalized LER

Use `sinter.shot_error_rate_to_piece_error_rate(failures/shots, pieces=24*d, values=8)`, matching earlier reports. This is a reporting convention, not an independently measured per-round failure rate. The transformation is nonlinear, so its ratios differ from raw whole-shot failure-rate ratios.

| d | Correlated UF | BP5 + UF | BP5 + correlated UF | Correlated MWPM | BP5/MWPM | BP5+cUF/MWPM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.001598999 | 0.0009520716 | 0.0009407092 | 0.0008907859 | 1.0688 | 1.0560 |
| 9 | 0.0007182492 | 0.0003909744 | 0.0003764511 | 0.0003336844 | 1.1717 | 1.1282 |
| 11 | 0.0003072293 | 0.0001556463 | 0.0001479609 | 0.0001174314 | 1.3254 | 1.2600 |
| 13 | 0.0001339519 | 6.07693e-05 | 5.576455e-05 | 4.210334e-05 | 1.4433 | 1.3245 |

## The 95,000 rows outside the original BP pilot

This subset excludes all original pilot rows. It is reported without changing decoder parameters or selecting the better decoder per shot.

| d | Correlated UF | BP5 + UF | BP5 + correlated UF | Correlated MWPM |
|---:|---:|---:|---:|---:|
| 7 | 22,057 | 13,918 | 13,776 | 13,072 |
| 9 | 13,536 | 7,638 | 7,377 | 6,578 |
| 11 | 7,378 | 3,803 | 3,626 | 2,890 |
| 13 | 3,887 | 1,777 | 1,640 | 1,228 |

All cohort-level confidence intervals and paired comparisons are available in the per-distance `summary.json` files.

## Software cost

Means over 16 predetermined serial shots per distance, measured inside the native kernel after parallel decoding and matching have finished. Each variant is charged all of its shared evidence work. The second variant's evidence time includes the first UF solve and correlation discounts. These are software kernel timings, not FPGA or ASIC estimates. See `environment.json` for scope and hardware.

| d | Decoder | Evidence ms | Final UF ms | Total ms |
|---:|---|---:|---:|---:|
| 7 | BP5 + UF | 85.798 | 8.270 | 94.068 |
| 7 | BP5 + correlated UF | 94.135 | 7.587 | 101.722 |
| 9 | BP5 + UF | 201.506 | 24.118 | 225.624 |
| 9 | BP5 + correlated UF | 225.749 | 22.629 | 248.378 |
| 11 | BP5 + UF | 403.129 | 45.514 | 448.642 |
| 11 | BP5 + correlated UF | 449.072 | 43.977 | 493.050 |
| 13 | BP5 + UF | 687.098 | 155.613 | 842.712 |
| 13 | BP5 + correlated UF | 844.150 | 161.702 | 1005.852 |

## Validation and limitations

All 800,000 new UF corrections reproduce the full syndrome and satisfy the ideal-yoke prediction parities. All 400,000 freshly recomputed correlated-MWPM predictions match their archived references. All 20,000 overlapping BP5 predictions and every non-timing diagnostic reproduce the frozen pilot exactly. Independent Python checks validate BP posteriors, physical correction edges, merge forests, the extra correlation pass, and thread invariance. Exact synthetic BP checks cover trees, zero messages, high-degree constraints and correlated detector cancellations.

BP and the projection into graph weights are approximate. Adding prior-derived hard correlation discounts after BP is also a heuristic. This is joint decoding with ideal yokes; it does not separately test the L1/L2 hierarchy. One physical noise rate and four finite distances do not establish an asymptotic threshold or suppression law. Intervals and paired tests are exploratory, without multiplicity adjustment.

![Accuracy and paired difference from correlated MWPM](comparison.png)

See the [fixed protocol](README.md), per-distance request/sample identities, verification records, and per-shot `results.npz` arrays. Source identities are in each request; durable artifact hashes are in `artifact_manifest.json`.
