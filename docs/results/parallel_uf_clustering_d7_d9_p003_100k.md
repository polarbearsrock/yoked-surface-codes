# Parallel UF clustering: paired 100k-shot d=7/d=9 experiment

Frontier-weighted growth plus bounded bridges reduces normalized LER by **32.3% at d=7 and 37.1% at d=9** relative to correlated UF. It remains 21.5% and 35.3% above correlated MWPM, respectively. Changing growth supplies most of the benefit; optimizing the existing forests alone produces no LER improvement.

SI1000 p=0.003, six patches, two ideal yokes, CZ circuits, rounds=4d. Both distances reuse their original complete 100,000-shot samples. The first-pass UF correction and resulting second-pass correlation weights are identical across variants. No truth or MWPM output is used in decoding.

## Logical error rates

The normalized LER uses the existing convention `sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=6*4*d, values=8)`. Whole-shot failure counts are included to make the comparison unambiguous.

| Decoder | d=7 failures | d=7 normalized LER | d=9 failures | d=9 normalized LER |
|---|---:|---:|---:|---:|
| Correlated MWPM (saved reference) | 13,785 | 0.000890786 | 6,925 | 0.000333684 |
| Correlated UF | 23,230 | 0.001598999 | 14,247 | 0.000718249 |
| UF + forest-cost control | 23,230 | 0.001598999 | 14,247 | 0.000718249 |
| UF + bounded bridges | 21,065 | 0.001428312 | 12,776 | 0.000638098 |
| Frontier-weighted UF | 17,263 | 0.001140948 | 9,820 | 0.000481532 |
| Frontier-weighted UF + forest-cost control | 17,263 | 0.001140948 | 9,820 | 0.000481532 |
| Frontier-weighted UF + bounded bridges | 16,462 | 0.001082300 | 9,242 | 0.000451591 |

## Paired changes from correlated UF

Positive reduction means improvement. Confidence intervals refer to the reduction in whole-shot failure probability, in percentage points; they are paired normal 95% intervals. Comparisons are exploratory and are not adjusted for multiplicity. Earlier diagnoses used this same shot pool, so this is not an untouched holdout; the intervals do not account for adaptive algorithm-development choices.

| d | Variant | Failures repaired | Successes spoiled | Net repairs | Failure reduction, percentage points (95% CI) |
|---:|---|---:|---:|---:|---:|
| 7 | UF + forest-cost control | 0 | 0 | +0 | +0.000 [+0.000, +0.000] |
| 7 | UF + bounded bridges | 3,310 | 1,145 | +2,165 | +2.165 [+2.035, +2.295] |
| 7 | Frontier-weighted UF | 8,601 | 2,634 | +5,967 | +5.967 [+5.763, +6.171] |
| 7 | Frontier-weighted UF + forest-cost control | 8,601 | 2,634 | +5,967 | +5.967 [+5.763, +6.171] |
| 7 | Frontier-weighted UF + bounded bridges | 9,271 | 2,503 | +6,768 | +6.768 [+6.560, +6.976] |
| 9 | UF + forest-cost control | 0 | 0 | +0 | +0.000 [+0.000, +0.000] |
| 9 | UF + bounded bridges | 2,215 | 744 | +1,471 | +1.471 [+1.365, +1.577] |
| 9 | Frontier-weighted UF | 6,138 | 1,711 | +4,427 | +4.427 [+4.256, +4.598] |
| 9 | Frontier-weighted UF + forest-cost control | 6,138 | 1,711 | +4,427 | +4.427 [+4.256, +4.598] |
| 9 | Frontier-weighted UF + bounded bridges | 6,615 | 1,610 | +5,005 | +5.005 [+4.830, +5.180] |

The bridge stage also improves on frontier-weighted growth alone:

| d | Additional repairs | Additional spoiled successes | Net additional repairs | Failure reduction, percentage points (95% CI) |
|---:|---:|---:|---:|---:|
| 7 | 1,697 | 896 | +801 | +0.801 [+0.701, +0.901] |
| 9 | 1,113 | 535 | +578 | +0.578 [+0.499, +0.657] |

## Growth work and dependency proxies

These cover only the modified second UF pass. They are software-measured algorithmic quantities, not FPGA cycles, latency, or resource estimates. An epoch is a global next-event batch. Edge evaluations count initial growth-rate calculations and later reevaluations.

| d | Growth policy | Mean event epochs | Mean edge evaluations | Mean frontier visits | Mean largest cluster | p99 largest cluster | Mean tree depth | p99 tree depth |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | Existing UF | 218.7 | 76,288 | 140,122 | 234.9 | 1076 | 14.5 | 25 |
| 7 | Frontier-weighted | 265.5 | 74,932 | 80,891 | 129.1 | 784 | 13.4 | 25 |
| 9 | Existing UF | 365.2 | 173,342 | 422,657 | 473.6 | 1977 | 18.1 | 32 |
| 9 | Frontier-weighted | 454.9 | 169,894 | 204,551 | 223.2 | 1508 | 16.7 | 32 |

## Bounded refinement work

Two refinement rounds are permitted. Disjoint cluster pairs can merge in parallel. Tree messages count directed min-sum messages plus final traceback traversals. They exclude growth communication, proposal aggregation/broadcast, candidate-edge exchanges, and arbitration; this is a partial communication count, not total bytes.

| d | Starting growth | Mean eligible edges | Mean accepted bridges | Mean peak parallel pairs | Mean tree messages | p99 extended tree depth |
|---:|---|---:|---:|---:|---:|---:|
| 7 | Existing UF | 264.3 | 0.8 | 0.8 | 4,209 | 25 |
| 7 | Frontier-weighted | 213.9 | 0.7 | 0.7 | 3,677 | 25 |
| 9 | Existing UF | 553.8 | 1.4 | 1.4 | 9,993 | 33 |
| 9 | Frontier-weighted | 440.2 | 1.2 | 1.2 | 8,588 | 32 |

## Interpretation limits and validation

Frontier weighting changes growth order using powers-of-two rates. Refinement admits only individually cost-improving forest bridges with at most 0.5 nats of growth remaining. It cannot introduce cycles or jointly accept a sequence of individually unprofitable connections. A graph-cost improvement does not guarantee a logical correction.

All 200,000 baseline predictions agree with the saved experiment; all 1,200,000 variant corrections satisfy their full detector syndromes. Independent checks cover random graphs with exhaustive correction enumeration, full-edge scans for the new growth, bridge proposals evaluated by independent forest solves, production physical corrections, and native thread determinism. Seventy production UF/correlation tests passed.

The native kernel is a software event simulator. Hardware implementation still requires choosing fixed-point precision, designing reductions and merge synchronization, and accounting for high-degree yoke routing. No hardware speedup or MWPM resource comparison is claimed.

[Protocol, counters, and reproduction instructions](parallel_uf_clustering_d7_d9_p003_100k/README.md). [Independent validation](parallel_uf_clustering_d7_d9_p003_100k/verification.json).
