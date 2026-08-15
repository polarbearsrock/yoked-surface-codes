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

## Setup

Use Python 3.12 or newer. Python 3.14 is the version used to validate this
fork. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

The virtual environment and generated `out/` directory are ignored by Git.
GNU Parallel is only required by some legacy top-level scripts; install it
separately when using those workflows (for example, `sudo apt install
parallel` on Debian or Ubuntu).

For the focused Figure 8 1D-yoked workflow, including resource controls and
the distinction between released and regenerated data, see
[`REPRODUCING_FIG8_1D.md`](REPRODUCING_FIG8_1D.md).

## Directory structure

- `.`: top level of repository, with this README and the generation scripts
- `./assets`: contains plots from the paper, the data used to generate them, and a .skp file rendering the topological diagrams in SketchUp
- `./out`: scripts are configured to create this directory and write their output to various locations within it
- `./src`: source root of the code; the directory to include in `PYTHONPATH`
- `./src/yoked`: code for generating and debugging yoked surface code circuits
- `./src/gen`: utility code for generating and debugging circuits
- `./tools`: tools used by the top-level scripts to perform tasks such as generating circuits and plots
