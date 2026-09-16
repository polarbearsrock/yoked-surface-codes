# Distance extension of the frozen UF clustering experiment

This experiment extends the d=7/9 comparison to **d=11 and d=13**, with
100,000 shared shots at each distance. Parameters remain SI1000 p=0.003,
six patches, two ideal yokes, CZ circuits, and rounds=4d (44 and 52).
The samples are new at these distances, generated using seed 42 and one
full bit-packed Stim sampling call per distance.

The existing [kernel and protocol](../parallel_uf_clustering_d7_d9_p003_100k/README.md)
are reused without modification. `run.py` pins their hashes and imports
their native wrapper, build command, result layout and summary function.
There are six UF outputs: correlated UF, its forest-cost control, its
bridge extension, frontier-weighted UF, its forest-cost control, and its
bridge extension. Correlated PyMatching MWPM provides the reference.
All variants receive the same sampled detector rows. Truth and MWPM
predictions are used only in scoring, never by the UF decoder.

The first UF pass and correlation rules are unchanged. Only second-pass
growth/refinement differ. The frontier rate is
`2**(-ceil(log2(max(1,frontier_edge_count))))` per active endpoint;
bridge candidates have at most 0.5 nats of growth remaining. Two rounds
accept disjoint, mutually best, individually cost-improving bridges.
There is no tuning sweep or selection using the new outcomes.

## Run and reproduce

From the repository root, with the recorded dependencies installed:

```bash
work_dir=$(mktemp -d "$TMPDIR/parallel-uf-distance-scaling-XXXXXX")
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/uf-soft-mpl"
.venv/bin/python docs/results/parallel_uf_clustering_d11_d13_p003_100k/run.py \
    --work-dir "$work_dir" --output "$work_dir/results" \
    --threads 64 --matching-workers 16
```

The archived run used the default output directory, this directory. Builds,
raw samples and resumable chunks remain under TMPDIR; compact per-shot
arrays, input manifests, requests, verification records and summaries are
retained here. For a separate reproduction, use the output override above
and compare `d11/results.npz` and `d13/results.npz` arrays with this archive.
NPZ byte hashes may differ because of archive metadata; array equality is
the relevant check across separately created files. Exact Stim reproduction
also requires the recorded Stim version, instruction-set module and full
sampling call shape. The stored payload hashes identify the original shots.

The runner first confirms that the current circuit generator reproduces the
old d=9 circuit and DEM hashes. It verifies every saved sample on reload.
MWPM workers own separate matching state; they consume packed inputs.
The OpenMP UF kernel runs after the process pool exits. UF checkpoints are
bound to a request identity and file checksum. Changed requests or sources
are rejected rather than silently mixed.

At each new distance, 32 predetermined random rows compare native first-pass
physical edges and final baseline forests/corrections with production
Python, compare four tree costs with an independent forest solver, compare
one-thread and 64-thread output, and compare packed parallel MWPM with its
unpacked serial output. Two rows additionally check frontier growth with a
full-edge scan. All six native corrections on every shot must reproduce
the complete syndrome. All seven predictions must satisfy ideal-yoke
parity. The full new native baseline is not separately rerun in Python.

## Reporting

```bash
.venv/bin/python docs/results/parallel_uf_clustering_d11_d13_p003_100k/report.py
```

`report.py` audits the archived arrays and recomputes the old and new
summaries, then creates the adjacent Markdown report, `scaling.json`, and
PNG/SVG figures. Report normalization is unchanged:
`sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=24*d, values=8)`.
Whole-shot counts are also reported. Ratio confidence intervals use 20,000
multinomial resamples of the four joint success/failure outcomes, preserving
the paired comparison. Marginal LER intervals use Wilson intervals.

Operation counts describe the software algorithm; they are not hardware
latency, area or energy measurements. The original kernel still uses
floating-point weights and a global event heap. Fixed-point implementation
and routing costs require separate hardware evaluation.
