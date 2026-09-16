# Frozen UF clustering comparison at p=0.001

This experiment changes physical noise probability from p=0.003 to p=0.001.
It retains d=7,9,11,13, six patches, two ideal yokes, CZ circuits, SI1000
noise and rounds=4d. There are exactly 100,000 new shots per distance,
shared by all seven decoder outputs. The sampling seed is 42, with one
full bit-packed Stim sampling call per distance.

The native kernel and [algorithm settings](../parallel_uf_clustering_d7_d9_p003_100k/README.md)
are reused without modification. All six UF variants retain the original
UF first pass and its correlation evidence. Frontier growth uses the same
power-of-two rate normalization. Bridge eligibility remains 0.5 nats of
remaining growth, with two rounds of mutually best improving proposals.
The graph weights and correlation rules are compiled from the DEM at the
new physical noise probability. No truth or MWPM output enters UF inference.

`collect.py` imports the existing matching worker, native wrapper, summary
function and validation routines from the frozen earlier experiments. It
adds a configurable noise parameter and distance list. Earlier experiment
sources and artifacts remain unchanged. The six native outputs also retain
their operation counters; host runtime is not a hardware cost estimate.

## Reproduce

From the repository root:

```bash
work_dir=$(mktemp -d "$TMPDIR/parallel-uf-p001-XXXXXX")
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/uf-soft-mpl"
.venv/bin/python docs/results/parallel_uf_clustering_p001_100k/collect.py \
    --work-dir "$work_dir" --output "$work_dir/results" --p 0.001 \
    --distances 7 9 11 13 --shots 100000 --threads 64 --matching-workers 16
```

The archived run used the default output location, this directory. Builds,
raw samples and resumable chunks remain under TMPDIR. The archive keeps
per-shot arrays, sample manifests, requests, summaries and verification.
Compare arrays from a separate reproduction with the archived arrays;
NPZ byte hashes can differ because archive metadata changes. Exact sample
reproduction requires the recorded Stim version, instruction-set module
and full sampling-call shape. Input payload hashes identify the exact shots.

The `--prepare-only` flag generates/verifies samples without compiling or
decoding. During the archived run, d=11/13 samples were prepared in a
separate process while the lower distances decoded. Each distance still
uses exactly one full sampling call. All decoding requests record source,
sample and library hashes; resumed chunks must match their request identity.

## Validate and report the archived run

```bash
audit_dir=$(mktemp -d "$TMPDIR/parallel-uf-p001-audit-XXXXXX")
.venv/bin/python docs/results/parallel_uf_clustering_p001_100k/audit.py \
    --work-dir "$audit_dir"
.venv/bin/python docs/results/parallel_uf_clustering_p001_100k/report.py
```

The collector performs the same 32 predetermined random-case comparisons
at each distance as before: Python physical corrections, independent tree
costs, two independent full-edge scans of frontier growth, native thread
determinism, and serial/unpacked versus parallel/packed MWPM. Every UF
correction must satisfy its full syndrome; all outputs must satisfy yoke
parity. The audit then independently rechecks every row where any decoder
failed, including all six native outputs and independent frontier scans.
That additional validation cannot change predictions or decoder settings.

`report.py` recomputes summaries and audits the saved hashes, then produces
the adjacent report, `analysis.json` and PNG/SVG figures. Because failure
counts can be small, it uses exact binomial intervals and one-sided 95%
upper bounds when no failures occur. An observed relative gap is undefined
when MWPM has zero failures. Conservative ratio intervals combine two exact
97.5% marginal intervals using Bonferroni, giving at least 95% joint coverage
without assuming the decoders' paired outcomes are independent. Exact
McNemar tests also retain the pairing. Comparisons are exploratory and are
not adjusted across distances or decoders.

The old summary format is retained for reproducibility; its normal
intervals on paired changes are not used for the small-count conclusions.
The normalized LER convention remains
`sinter.shot_error_rate_to_piece_error_rate(failures/100000, pieces=24*d, values=8)`.
A decreasing zero-count normalized upper bound with distance reflects the
normalization and does not measure error suppression.
