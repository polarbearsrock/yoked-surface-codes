# Four decoders across d=7,9,11,13,15

Fixed protocol: six rotated surface-code patches, two ideal yokes, SI1000
p=0.003 (0.3%), CZ extraction with r=4d rounds and ideal time boundaries.
100,000 paired shots at each distance. All four configurations use the same
sample and the same plain PyMatching outer MWPM with its existing tie convention.

1. Correlated PyMatching v2 MWPM + frozen-weight complementary gap.
2. Correlated PyMatching v2 MWPM + existing MPP matching-radius cluster gap.
3. Two-pass correlated UF + full uncapped cluster gap.
4. BP5 + single-pass weighted UF + full uncapped cluster gap.

BP5 uses the existing sum-product flooding implementation: five fixed iterations,
damping 0.5, LLR clip 30, and edge projection
`-log(clip(sum of supporting fault posteriors, 1e-15, 1))`. Variant 4 does not
apply conditional correlation discounts after BP. No fitting or calibration.

Use the frozen source snapshot from the corrected d=7 experiment and the same
verified Dante Stim fork. Reuse its d=7 accuracy samples and predictions exactly.
Generate d=9,11,13,15 with sampling seed 202609290700 + distance. The installed
wheel must be PyMatching 2.4.0; the MPP adapter must use its verified pinned
PyMatching 2.4.0 source engine. Record both versions, revisions and binary hashes.

Accuracy collection uses 32 independent MWPM processes, followed by 32 OpenMP
threads for UF/BP, on the same 32 physical cores. Checkpoint all work. Validate
corrections, confidence scores, thread invariance, row pairing and yoke parity.
The new combined UF adapter must match the prior two native implementations and
independent Python oracles before collection at each distance.

Primary accuracy metric: effective logical error rate per physical patch per
round, using Sinter `shot_error_rate_to_piece_error_rate(block_rate,
pieces=6*4*d, values=8)`. Retain block counts, exact binomial intervals transformed
through this normalization, and paired whole-shot comparisons. This is the
existing effective normalization, not an independently measured patch hazard.

Latency protocol: warm decoder objects, batch size one, one CPU thread pinned
to one physical core, with no sweep collection running concurrently. Use 1,000
uniformly selected distinct saved shots per distance, the same rows for all four
configurations, and 32 separate warm-up rows. Balance/interleave variant order.
Time a full six-patch decode from an in-memory detector row, including local
input preparation, L1 decoding, confidence extraction, and the unchanged L2.
Exclude simulation, file I/O, construction/compilation, and post-timing correctness
comparisons. Retain raw wall/process times, predictions and selected row IDs.
Report mean, median, p95, p99 and empirical maximum. Amortized time per patch-round
is explicitly labeled; it is not streaming response latency. Report the current
software implementation, including Python adapters, rather than hardware latency.

No run may infer latency by dividing 32-core collection time by shots. Preserve
collection wall times as separate throughput measurements. Preserve earlier
experiments and all original repository changes. Build and checkpoint files go
under $DANTE_SCRATCH/runs; retained results and provenance go under results/.
