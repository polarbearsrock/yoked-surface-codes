# Parallel UF clustering experiment

This is an isolated experiment, not a change to the production decoder. It
reuses all 100,000 saved shots at each distance: SI1000, p=0.003, d=7/9,
six patches, two ideal yokes, CZ circuits, and 4d rounds.

## Frozen comparison

Every variant receives exactly the same second-pass correlation weights.
The first pass remains production UF; only its selected physical correction
edges trigger the existing correlation rules. Neither truth nor MWPM output
is passed into the native decoder.

The six outputs are:

1. Existing correlated UF.
2. The same final forest with minimum-cost tree correction (control).
3. Existing growth plus two bounded bridge-refinement rounds.
4. Frontier-weighted growth followed by existing peeling.
5. Frontier-weighted growth with minimum-cost tree correction (control).
6. Frontier-weighted growth plus the same bounded refinement.

For an active cluster with b frontier edges, the new growth rate at each
frontier edge is `1 / 2**ceil(log2(max(1,b)))`. The rates at the two endpoints
add. All rates are positive for active clusters with a nonempty frontier;
yoke vertices receive the same rule as other vertices. The original graph,
double-precision edge weights, and anchored tie-batch rule are preserved.
This is a power-of-two approximation to frontier normalization, not the
published sequential smallest-boundary-first UF algorithm.

## Bounded refinement

After growth ends, freeze a candidate list of intercluster edges whose
remaining growth is at most **0.5 nats**. This cap is motivated by the prior
growth diagnosis, not fitted to outcomes of this experiment. Two rounds are
allowed. There is no tuning sweep or truth-based choice among outputs.

For every tree component, compute min-sum messages in both directions. These
give its optimal cost C and the cost C(u) if the syndrome at vertex u were
toggled. At a boundary vertex the constraint is free and C(u)=C. For an edge
e=(u,v) between distinct components A and B, its exact possible cost change is

`delta(e) = weight(e) + C_A(u)-C_A + C_B(v)-C_B`.

Each component proposes its most negative eligible edge; ties use edge ID.
Only mutual proposals with delta < -1e-9 are accepted. Each component thus
participates in at most one merge in a round. Accepted pairs are disjoint,
the union stays a forest, and the minimum forest cost cannot increase.
Recompute messages for the next round and extract the final minimum-cost
correction. Detector constraints remain enforced; each boundary terminal
may independently absorb parity.

A lower graph cost need not imply the correct logical class. Also, a single
bridge between two even components without boundaries cannot support a
different correction. This search cannot jointly accept two unprofitable
edges, create cycles, or discover arbitrary multiedge paths through isolated
vertices. A negative result does not rule out those larger searches.

## Hardware interpretation

The native event heap is a fast **software simulator**, not the proposed
hardware implementation. Simultaneous edge growth is parallelizable; an
event-driven realization needs a global next-event reduction, distributed
cluster updates, and synchronization. A clock-stepped realization would
require choosing and testing weight/time quantization separately.

Bridge scoring can use parallel tree messages and independent edge units.
Cluster proposal reduction/broadcast and mutual acceptance are separate
synchronized phases. Disjoint accepted pairs can update in parallel. High
degree yoke vertices and large cluster diameters remain communication costs.

Recorded fields are deliberately algorithmic proxies:

- `epochs`: anchored growth event batches, **not clock cycles**.
- `edge_evaluations`: initial edge-rate evaluations plus rate reevaluations.
- `frontier_visits`: entries examined when collecting affected frontiers.
- `max_cluster`, `max_frontier`: maximum intermediate component vertex and
  frontier-edge counts, including the initial graph state.
- `tree_depth`: maximum root-to-leaf depth, using the smallest boundary
  vertex as root when present. Extension values include its intermediate
  trees. Parallel reductions at high-degree vertices add work beyond depth.
- `bridges`, `eligible_bridges`, `improving_proposals`, `max_parallel_pairs`:
  accepted edges, initial candidate count, improving candidates counted
  across attempted rounds, and maximum simultaneously accepted pairs.
- `tree_messages`: two directed min-sum messages per tree edge per attempted
  refinement round, plus upward solve/downward traceback for the final tree.
  This **excludes** growth communication, cluster proposal reductions,
  candidate-edge exchanges, headers, and arbitration; it is not total bytes
  or a measured FPGA communication volume. Each min-sum message contains two
  costs; traceback messages contain parity decisions.

The first pass is common and excluded from growth comparison fields. No
latency, resource usage, or advantage over hardware MWPM is inferred from
host wall time. The experiment preserves floating-point weights; fixed-point
accuracy and an RTL implementation are future validation steps.

## Validation and reproduction

`verify.py` compares first-pass selected edges and baseline final forests and
corrections against production Python UF. It independently scans all graph
edges to check the normalized growth. On small random graphs it evaluates
every proposed bridge by separately solving the enlarged forest, and checks
tree costs by exhaustive enumeration. Real-shot checks also compare tree
costs to the prior independent Python dynamic program. Results must be
identical with one and four native threads.

The full runner verifies the original sample, circuit, and DEM hashes;
checks every baseline prediction against the saved 100k-shot record; and
validates all six corrections against the full syndrome. Chunk checksums
allow resumable runs under an unchanged request identity. Per-shot truth
and saved MWPM labels are attached only after decoding, for reporting.

All builds, checkpoints, and other intermediate files go under TMPDIR:

```bash
work_dir=$(mktemp -d "$TMPDIR/parallel-uf-clustering-XXXXXX")
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/uf-soft-mpl"
.venv/bin/python docs/results/parallel_uf_clustering_d7_d9_p003_100k/verify.py --work-dir "$work_dir"
.venv/bin/python docs/results/parallel_uf_clustering_d7_d9_p003_100k/experiment.py \
    --output "$work_dir/measurement" --build-dir "$work_dir/build" --threads 48
```

The durable `d7/` and `d9/` artifacts contain requests, summaries, and compact
per-shot results. `report.py` recomputes summaries and generates the adjacent
Markdown report. `verification.json` records independent algorithm checks;
`artifact_manifest.json` hashes the final scripts and saved artifacts.
