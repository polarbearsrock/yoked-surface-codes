# Code repository for "Yoked Surface Codes"

This repository contains code to generate circuits, the data for, and code to plot the figures
appearing in the paper "Yoked Surface Codes".

## Reproducing figures

Sitting next to this README, at the root of the repository,
are several folders. `assets` contains the paper figures, along
with the data used to produce them as `.csv` files. Run the `make_paper_figures`
script to regenerate these plots. The stats have been partitioned based on which
figure they are used in - one can also inspect the script to see what each `.csv`
data file corresponds to and how that data is used.

There are also a series of scripts for producing gap distributions,
as well as performing circuit simulations of yoking as described in the paper.
However, these do not include the gap simulations, nor can they produce the gap distribution
for a correlated minimum-weight perfect matching decoder, as these use internal tools.
A warning - this repo was 're-focused' halfway through the project. Consequently, there are
commands that allow for constructions not discussed in the paper. One example
is the functionality surrounding single Y-type yokes. The code is brittle, and
we emphasize caution when using these 'additional' functionalities - they have not been updated.

To install the requirements:
```bash
python -m venv .venv
source .venv/bin/activate
sudo apt install parallel
pip install -r requirements.txt
```

## Union Find decoder

The `yoked.decoders` package uses the repository's weighted growth-and-peeling
implementation as the default UF decoder (`UnionFindDecoder` and
`SinterUnionFindDecoder`). An optional `CorrelatedUnionFindDecoder` uses two UF
passes with correlation information from the DEM. Fusion Blossom's UF variant
is available separately as an optional dependency and explicit decoder choice. See
[usage and correctness tests](docs/union_find_usage.md).

## Hierarchical L1/L2 decoding

The paper-style configuration uses **native PyMatching correlated MWPM at L1,
frozen complementary gaps as confidence, and plain MWPM at L2**. Run it with
`tools/hierarchical_experiment paper`; see the [configuration, examples, and
PyMatching compatibility notes](docs/paper_hierarchical_decoding.md).

The optional [MPP confidence experiment](docs/mpp_confidence_experiment.md) replaces
the complementary gap with a native cluster score while keeping correlated L1
and plain-MWPM L2 fixed. `tools/mpp_experiment` compares both on identical shots.

The [correlated-UF experiment](docs/correlated_uf_cluster_gap_experiment.md) uses
two-pass correlated UF and full cluster gaps at L1, followed by the same plain
MWPM at L2. It compares final block accuracy against both saved MWPM baselines.

The `yoked.hierarchical` package decodes a 1D yoked block in two layers: L1
decodes each patch on its own and stores reference bits with soft outputs, and
a weighted-MWPM outer decoder (L2) uses the calibrated residual-error probabilities and
the yoke parities to decide which patches to correct. `tools/hierarchical_experiment`
drives it in four verified stages -- collect, calibrate, replay, summarize -- each
publishing a manifest that the next one checks before reading. See
[usage](docs/hierarchical_decoding.md) and the
[design spec](docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md).

## Directory structure

- `.`: top level of repository, with this README and the generation scripts
- `./assets`: contains plots from the paper, the data used to generate them, and a .skp file rendering the topological diagrams in SketchUp
- `./out`: scripts are configured to create this directory and write their output to various locations within it
- `./src`: source root of the code; the directory to include in `PYTHONPATH`
- `./src/yoked`: code for generating and debugging yoked surface code circuits
- `./src/gen`: utility code for generating and debugging circuits
- `./tools`: tools used by the top-level scripts to perform tasks such as generating circuits and plots
