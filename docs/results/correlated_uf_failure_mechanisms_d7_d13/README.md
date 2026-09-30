# Correlated-UF failure mechanism diagnostics

See the [report](../correlated_uf_failure_mechanisms_d7_d13.md) for interpretation.
These scripts inspect the existing p=0.003, 100k-shot datasets at d=7/9/11/13.
They do not resample noise or change production decoders.

- `diagnose.py`: join all saved outcomes; replay 128 UF-only failures per
  distance under fixed UF correlation weights; certify accessible logical
  classes on the original, frontier, and extended frontier forests/partitions.
- `audit.py`: verify recorded source hashes and summaries, and independently
  check 192 logical spaces using DFS and explicit logical spans.
- `reachable_cases.py`: examine the seven residual failures whose extended
  forests admit truth; compute both class costs using tree DP and independently
  constrained matching.
- `trace_76890.py`, `full_shot_76890.json`: trace the complete original d=7
  shot 76890; restore a blocked boundary edge and omitted internal routes
  under fixed UF weights, with independent forest and logical-class checks.
- `protocol.json`, `summary.json`: configuration and aggregate replay results.
- `d*/population.json`: exact outcome transitions and logical-error counts over
  all 100k saved predictions at that distance.
- `d*/selection.json`, `d*/rows.json`, `d*/summary.json`: chosen diagnostic shots,
  individual interventions/certificates, and their summaries.
- `d*/verification.json`, `audit.json`, `reachable_cases.json`: provenance and
  independent checks.
- `artifact_manifest.json`: SHA-256 hashes of all artifacts, including the report.

The d=7/9 selections exactly reuse the prior failure investigation. The d=11/13
selections use seed `20260915+d` and sample uniformly without replacement from
the original UF-only failure population. This is a conditional diagnostic,
not a new 100k-shot decoder evaluation. In particular, the residual frontier
sample excludes regressions on originally UF-correct shots.

Run from the repository root. Raw detector payloads are read from the sample
directories recorded in the frozen experiment requests; those directories
currently reside under `$TMPDIR`. The committed records identify their payload,
circuit, and DEM hashes. The scripts require those existing inputs.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export MPLCONFIGDIR="$TMPDIR/uf-soft-mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
diagnostic_work=$(mktemp -d "$TMPDIR/uf-failure-mechanisms-XXXXXX")
diagnostic_scripts=docs/results/correlated_uf_failure_mechanisms_d7_d13
.venv/bin/python "$diagnostic_scripts/diagnose.py" \
  --work-dir "$diagnostic_work" --output "$diagnostic_work/results"
.venv/bin/python "$diagnostic_scripts/audit.py" \
  --root "$diagnostic_work/results" --library "$diagnostic_work/build/clustering.so"
.venv/bin/python "$diagnostic_scripts/reachable_cases.py" \
  --root "$diagnostic_work/results" --library "$diagnostic_work/build/clustering.so"
.venv/bin/python "$diagnostic_scripts/trace_76890.py" \
  --output "$diagnostic_work/full_shot_76890.json"
```

The native decoder is built from the previously frozen `kernel.cc` and never
receives truth or MWPM output. Truth is used outside decoding to select cohorts,
evaluate answers, and compute diagnostic class constraints.
