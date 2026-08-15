# Reproducing the 1D results in published Figure 8

The published Figure 8 has two panels involving 1D yoked surface codes:

- **Figure 8b** compares a full, one-outer-round circuit simulation against a
  complementary-gap simulation and the fitted scaling law.
- **Figure 8d** compares long, ten-outer-round phenomenological simulations of
  unyoked (0D) and 1D-yoked memories against their fitted scaling laws.

The release contains the data and plotting code for both panels. It contains a
path for generating and sampling the full circuits in Figure 8b, but it does not
contain the internal correlated-matching and gap-simulation tools needed to
regenerate all samples from scratch.

## Parameter conventions

For the main 1D construction:

- `yokes=2` means the outer code has both an X-type and a Z-type yoke. This is
  the 1D `[[n, n-2, 2]]` quantum parity-check code used in the main text.
- `patches=n` is the number of inner surface-code patches in one outer block.
- `d` is the inner surface-code patch diameter.
- `r` is the number of noisy inner-code rounds between the perfect initial and
  final time boundaries in a full-circuit experiment.
- The circuit uses the CZ gateset and SI1000 noise at `p=0.001`.

The Figure 8b validation grid is

- `d in {5, 7, 9, 11}`;
- `n in {6, 10}`; and
- `r in {4d, 8d}`.

The comparison uses the per-patch-round form of the 1D fit

```text
p_L / (r*n) ~= r*n*8^(-d)/500,
```

which comes from the one-outer-round cumulative fit

```text
p_L ~= r^2*n^2*8^(-d)/500.
```

## Environment

Use Python 3.14 with the upgraded dependencies. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency versions are pinned in `requirements.txt`. The plotting and gap
collection code use repository-owned compatibility helpers instead of private
Sinter modules, which were removed after Sinter 1.12.

GNU `parallel` is not needed by the focused workflow below.

## Stage 1: reproduce the panels from released data

```bash
./reproduce_fig8_1d plot-paper-data
```

This writes:

```text
out/fig8_1d/paper_data/fig8b_1d_full_vs_gap.png
out/fig8_1d/paper_data/fig8d_0d_vs_1d.png
```

This stage reproduces the data, axes, fits, and uncertainty bars in the
checked-in `assets/gap_vs_full_1D.png` and `assets/gap_0D_and_1D.png` figures
without running Monte Carlo. Raster pixels and text spacing can differ with
the installed Matplotlib and font versions.

## Stage 2: run a small end-to-end full-circuit smoke test

```bash
./reproduce_fig8_1d smoke
```

The smoke test uses `d=3`, `n=6`, `r=4d`, and PyMatching's public two-pass
correlated decoder. It checks that circuit generation, SI1000 noise insertion,
detector-error-model conversion, sampling, decoding, and resumable CSV output
all work.

Collection controls can be overridden, for example:

```bash
MAX_SHOTS=1000000 MAX_ERRORS=1000 PROCESSES=16 ./reproduce_fig8_1d smoke
```

## Stage 3: generate the Figure 8b validation grid

Generating the circuits is inexpensive compared with sampling them:

```bash
./reproduce_fig8_1d generate-validation-grid
```

An open correlated-decoder baseline can then be collected with:

```bash
./reproduce_fig8_1d collect-open-validation-grid
```

The command defaults to Sinter's `pymatching-correlated` decoder and is
resumable through
`out/fig8_1d/validation_grid/pymatching-correlated_stats.csv`. To collect an
ordinary matching baseline instead, set `DECODER=pymatching`; this writes to
`pymatching_stats.csv`. Start with small shot and error limits. The
larger-distance paper points required tens or hundreds of millions of shots,
and some checked-in runs used one billion shots per circuit.

The runner defaults to four worker processes and one native numerical thread
per worker. For an especially gentle resumable run, use:

```bash
PROCESSES=2 THREADS_PER_PROCESS=1 \
MAX_SHOTS=1000000 MAX_ERRORS=100 \
./reproduce_fig8_1d collect-open-validation-grid
```

`PROCESSES` controls Sinter's worker-process count. `THREADS_PER_PROCESS` caps
OpenMP, OpenBLAS, MKL, NumExpr, Accelerate, and BLIS threads inside each worker,
preventing nested thread oversubscription. Rerunning the command continues from
the existing CSV instead of discarding completed samples.

Plot the newly collected points and the paper's fitted scaling law with:

```bash
./reproduce_fig8_1d plot-open-validation-grid
```

This writes
`out/fig8_1d/validation_grid/pymatching-correlated_fig8b.png`. Each point is
annotated as `logical errors / shots`; hollow downward triangles indicate
zero-error upper bounds instead of measured nonzero rates.

## Decoder limitation

The paper data uses `sparse_blossom_correlated`, an internal correlated
minimum-weight perfect-matching decoder. Current Sinter and PyMatching releases
provide a public two-pass correlated decoder under the name
`pymatching-correlated`, and the focused workflow now uses it by default. It is
the closest supported open replacement, but it is not the same decoder binary
used for the published data. Fresh results must therefore be compared against
matched rows in `assets/stats_check.csv` before being described as an exact
reproduction.

Therefore the milestones are deliberately separated:

1. exact plot reproduction from released statistics;
2. end-to-end circuit and sampling validation with an available decoder;
3. validation of `pymatching-correlated` against matched rows in
   `assets/stats_check.csv`; and
4. only then, expensive production sampling.

Figure 8d additionally depends on the unreleased multi-round gap simulator.
Reimplementing that simulator is a separate task from reproducing Figure 8b's
full-circuit points.
