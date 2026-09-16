# Correlated and hierarchical decoder comparison

Configuration: SI1000 `p=0.003`, six patches, two ideal yokes, and `rounds=4d`. The hierarchical row is `uf:cluster_gap -> gap_correlated`, `all_refined`, with the mixed exact outer rule.

Evaluation uses all 100,000 saved seed-42 shots at each distance; calibration uses a separate 50,000-shot seed-142 call at that distance. All comparisons use the same evaluation shots within a distance. The two distances are separate sampling calls and are resampled independently; their row numbers are not paired. Intervals use 10,000 empirical whole-shot bootstrap replicates at seed 43; the eight joint decoder-failure states preserve pairing. Calibration is held fixed, so these intervals quantify evaluation-shot uncertainty conditional on the fitted calibrators. Block failure means any of the 12 observables is wrong; normalized LER is Sinter's aggregate conversion of that block rate, not a directly counted per-patch error rate.

## d=7

| Decoder | Failed shots | Block rate (95% CI) | Normalized LER (95% CI) |
|---|---:|---:|---:|
| Correlated MWPM | 13,785 / 100,000 | 0.137850 (0.135700, 0.139980) | 0.0008907859 (0.00087570, 0.00090578) |
| Correlated UF | 23,230 / 100,000 | 0.232300 (0.229690, 0.234940) | 0.0015989989 (0.00157814, 0.00162018) |
| Hierarchical UF + correlated gap | 13,537 / 100,000 | 0.135370 (0.133210, 0.137470) | 0.00087338275 (0.00085827, 0.00088812) |

Paired hierarchical comparisons:

| Baseline | Block-rate difference (95% CI) | Normalized-LER ratio (95% CI) |
|---|---:|---:|
| Correlated MWPM | -0.002480 (-0.003430, -0.001530) | 0.980463 (0.973148, 0.987888) |
| Correlated UF | -0.096930 (-0.099350, -0.094550) | 0.546206 (0.537747, 0.554659) |

Discordant and concordant shot counts:

| Baseline | Baseline succeeds / hierarchy fails | Baseline fails / hierarchy succeeds | Both succeed | Both fail |
|---|---:|---:|---:|---:|
| Correlated MWPM | 1,050 | 1,298 | 85,165 | 12,487 |
| Correlated UF | 3,271 | 12,964 | 73,499 | 10,266 |

## d=9

| Decoder | Failed shots | Block rate (95% CI) | Normalized LER (95% CI) |
|---|---:|---:|---:|
| Correlated MWPM | 6,925 / 100,000 | 0.069250 (0.067670, 0.070850) | 0.00033368442 (0.00032576, 0.00034172) |
| Correlated UF | 14,247 / 100,000 | 0.142470 (0.140270, 0.144660) | 0.00071824925 (0.00070616, 0.00073031) |
| Hierarchical UF + correlated gap | 6,820 / 100,000 | 0.068200 (0.066660, 0.069810) | 0.00032841945 (0.00032071, 0.00033650) |

Paired hierarchical comparisons:

| Baseline | Block-rate difference (95% CI) | Normalized-LER ratio (95% CI) |
|---|---:|---:|
| Correlated MWPM | -0.001050 (-0.001770, -0.000330) | 0.984222 (0.973527, 0.995062) |
| Correlated UF | -0.074270 (-0.076310, -0.072220) | 0.457250 (0.447196, 0.467667) |

Discordant and concordant shot counts:

| Baseline | Baseline succeeds / hierarchy fails | Baseline fails / hierarchy succeeds | Both succeed | Both fail |
|---|---:|---:|---:|---:|
| Correlated MWPM | 627 | 732 | 92,448 | 6,193 |
| Correlated UF | 2,089 | 9,516 | 83,664 | 4,731 |

## Distance scaling

Ratios are d=7 normalized LER divided by d=9 normalized LER, with the two distances bootstrapped independently.

| Decoder | d7 / d9 normalized-LER ratio (95% CI) |
|---|---:|
| Correlated MWPM | 2.669546 (2.593837, 2.749476) |
| Correlated UF | 2.226245 (2.178289, 2.275088) |
| Hierarchical UF + correlated gap | 2.659351 (2.581677, 2.738966) |

## Cost scope

`all_refined` is a cost-unconstrained endpoint, not a selective-refinement policy. The stored work counters count decoder operations rather than elapsed time, and no comparable end-to-end latency benchmark was run, so these results support no latency or speedup claim.
