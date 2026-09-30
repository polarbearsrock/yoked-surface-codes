# Correlated UF with full cluster-gap confidence

This experiment runs **L1 correlated UF + full cluster gaps → L2 plain MWPM**.
It compares final block logical error rates with the two existing configurations:
correlated MWPM with complementary gaps, and correlated MWPM with MPP scores.
All three use identical saved shots and the same outer MWPM implementation.

## Decoder and confidence

For each patch, independently of the yoke measurements:

1. Run weighted Union-Find growth and peeling on the original decoding graph.
2. Use its selected correction edges as evidence for the existing DEM-derived
   pairwise correlation rules. Supported partner edges receive lower weights.
3. Restart UF on the original syndrome with those weights. This final pass
   supplies both the hard prediction and the terminated cluster geometry. If
   no weight changes, the first pass already supplies this result.
4. Compute one full cluster gap per logical sector from that final growth state.

The cluster gap follows the contracted-cluster shortest-path construction in
[Meister, Pattison and Preskill, *Efficient soft-output decoders for surface code*,
Definition 9](https://arxiv.org/html/2405.07433v2#S3). Applying it to this
repository's two-pass correlated, weighted UF decoder is our experimental
adaptation; this configuration is not a reproduction of a published correlated-UF
benchmark.

An edge inside one final cluster has cost zero. Other edges have cost
`max(0, final_weight - accumulated_growth)`, in natural-log weight units. For
each sector, Dijkstra finds the cheapest boundary-to-boundary walk that flips
that logical observable. The implementation tracks logical parity in the
search state so it handles observable labels directly.

Larger gaps mean that more remaining edge cost separates the logical boundaries.
These are confidence heuristics, not exact complementary gaps or calibrated
error probabilities. This experiment passes the raw gaps to L2 as edge weights,
with **no calibration, score cap, or confidence quantization**. The existing
PyMatching outer solver retains its own numerical precision and tie convention.
Full shortest-path searches still cost work; this Python implementation is an
accuracy reference, with no claim of a measured latency advantage.

For a sector, the outer syndrome is the measured yoke parity XOR the parity of
the **final correlated-UF predictions**. L2 finds a minimum-weight residual
correction, and the final prediction is the UF prediction XOR that correction.
Neither UF pass runs MWPM; PyMatching imports the local graph and solves L2.

The earlier `uf_soft` experiment keeps a plain-UF reference and signs some
second-pass scores relative to it. This experiment has its own entry point
because the reference here is the correlated-UF prediction itself.

## Paired accuracy experiment

The workspace recipe fixes:

| Setting | Value |
| --- | --- |
| Noise | SI1000, `p=0.003` (0.3%) |
| Distances | 7, 9, 11, 13, 15 |
| Syndrome rounds | **4d**: 28, 36, 44, 52, 60 |
| Block | Six patches, two ideal yokes, CZ circuits |
| Shots | 100,000 per distance, paired across configurations |
| Baseline | `results/mpp_confidence_p003_4d_100k_20260923` |

The primary metric is the probability of **any wrong logical observable in the
six-patch block over the full circuit**, after L2. This is not a per-round LER.
Reports include candidate/baseline LER ratios, relative increases, and paired
repairs and regressions. They also save each decoder's own L1 block failure
count: unlike the earlier confidence-only comparison, L1 predictions can differ.

Confidence intervals are stored as analysis metadata: exact binomial intervals
for rates and whole-shot paired-bootstrap intervals for differences and ratios.
Replicates with no baseline failures are counted and omitted from ratio
intervals. Very small pilot samples are validation data, not accuracy estimates.

From the workspace, launch the full sweep with:

```bash
bash experiments/correlated_uf_cluster_gap_p003_4d_100k.sh
```

The wrapper sources `env/workspace.sh` and the decoder virtual environment. Set
`UF_WORKERS` to change the number of worker processes; the default is eight.
New outputs go to `$DANTE_SCRATCH/runs/correlated-uf-cluster-gap-p003-4d-100k`.
Only the UF candidate is decoded; both saved MWPM outputs are verified and reused.
No native MPP extension build is needed for this comparison.

The production sweep was subsequently changed to **32 workers, one distance at
a time**, reusing completed shards. To resume that existing run, use:

```bash
bash experiments/correlated_uf_cluster_gap_p003_4d_100k_32_sequential.sh
```

This scheduler finishes and publishes d=7 before starting d=9, then d=11,13,15.
It keeps the original decoder and sample identities and records the scheduling
change separately. A terminated shard restarts; completed shards are verified
and reused.

For a small pilot, choose a different output directory and fewer shots:

```bash
source env/workspace.sh
source "$DANTE_REPO/.venv/bin/activate"
python experiments/correlated_uf_cluster_gap.py \
    --baseline-root "$DANTE_WORKSPACE/results/mpp_confidence_p003_4d_100k_20260923" \
    --out "$DANTE_SCRATCH/runs/correlated-uf-cluster-gap-pilot" \
    --p 0.003 --rounds-per-distance 4 \
    --distances 7 9 11 13 15 --shots 16 --shard-size 8 --workers 4
```

Completed shards are reusable when the command is repeated with the same inputs,
sources, package versions, shot count and shard size. Worker count may change.
The recipe validates every requested distance's noise and round count before
decoding. It saves source snapshots and hashes, sample identities, row indices,
baseline provenance, and artifact hashes. A failed worker does not publish a
completed shard; source or setting changes require a new output directory.

For each distance, `predictions.npz` contains the candidate's reference, raw
gaps, search-state counts, outer syndrome, residual and final prediction, along
with truth, row IDs and both saved baseline predictions. `results.json` contains
the statistics; root `summary.json` combines distances. `completion.json` is
written after aggregation and verification succeed.

## Entry points and checks

- `yoked.hierarchical.CorrelatedUFHierarchicalDecoder`: in-memory decoding.
- `tools/correlated_uf_experiment`: one saved sample or baseline distance,
  optionally a `--rows START:STOP` subset.
- Workspace `experiments/correlated_uf_cluster_gap.py`: parallel shards,
  resume checks and paired aggregation across distances.
- Workspace `experiments/correlated_uf_audit.py RUN_DIRECTORY`: verify saved
  artifacts and statistics, and redecode a small subset through both the soft
  and existing hard correlated-UF APIs.

Tests check final-pass growth costs against an explicitly reweighted graph,
hard predictions against the existing correlated UF API, L2 against exhaustive
outer MAP, independence of L1 from yokes, uncapped gaps, decoder reuse, paired
statistics and artifact validation. The shared cluster-gap tests independently
enumerate logical walks on small graphs.
