# Yoked Surface Codes

This repository contains the circuit generators, simulation tools, and data
used by the paper
[**“Yoked surface codes”**](https://doi.org/10.1038/s41467-025-59714-1). It
extends the
[original paper repository](https://github.com/Strilanc/yoked-surface-codes)
with a reproducible Figure 8 workflow and a sequence of L1 ProMatch-style
predecoder experiments.

The experiment code is deliberately conservative: sampled data, decoder
decisions, analysis, and provenance are kept separate, and claim-bearing runs
fail closed when their environment or authenticated source does not match the
frozen protocol.

## Start here

| Workflow | Current role | Entry point |
| --- | --- | --- |
| Published Figure 8 | Replot released data or run fresh 1D-yoked validation samples. | [`REPRODUCING_FIG8_1D.md`](REPRODUCING_FIG8_1D.md) |
| L1 ProMatch V3 pilot | Completed diagnostic comparison. The unsigned selector chose `d=7, p=0.002`, but the unblinded window-local predecoder was less accurate in every pilot cell, so no confirmatory run followed. | [`docs/PROMATCH_IMPLEMENTATION_PLAN.md`](docs/PROMATCH_IMPLEMENTATION_PLAN.md) |
| Global-context oracle replay | Completed Phase-A diagnosis over retained V3 shots; it preserved the input corpus and performed no new sampling. | [`experiments/PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md`](experiments/PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md) |
| B1 policy audit | The non-claim-bearing 20,000-shot V2 corpus was collected, analyzed, expanded, and finalized on the experiment workstation. | [`experiments/PROMATCH_L1_POLICY_AUDIT_20K.md`](experiments/PROMATCH_L1_POLICY_AUDIT_20K.md) |

[`experiments/README.md`](experiments/README.md) is the short status dashboard.
Frozen configurations and their lifecycle are indexed in
[`docs/README.md`](docs/README.md).

The released paper CSV files and figures are checked in under `assets/`.
Completed ProMatch corpora and their generated analyses are workstation-local
audit artifacts and are not distributed by a fresh clone; their status pages
document what was run without implying that those large artifacts are a public
archive.

## Setup

Python 3.12 or newer is required; this branch is validated with CPython 3.14.5.
The direct runtime and test dependencies are pinned in `requirements.txt`;
`uv` resolves their transitive dependencies when it creates the repository-local
environment.

```bash
uv python install 3.14.5
uv venv --python 3.14.5 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
```

For a fresh Google Cloud VM, [`gcp/README.md`](gcp/README.md) provides a
one-command environment bootstrap and a sourceable activation helper. It keeps
cloud scratch data and experiment outputs outside the Git checkout.

The `PYTHONPATH` export makes the non-packaged `gen` and `yoked` source trees
importable in an interactive Python session. Tests add `src/` through
`pytest.ini`, and the commands under `tools/` bootstrap it themselves.

On the experiment workstation, read [`AGENTS.md`](AGENTS.md) before running any
setup or sampling command. It defines the required `/data2` cache and temporary
paths, the 32-process ceiling, and the one-native-thread-per-process policy.

## Repository tour

- `src/gen/` — reusable circuit, geometry, flow, and visualization utilities.
- `src/yoked/` — yoked-memory circuit construction and gap-collection support.
- `src/yoked/decoding/` — ProMatch layout, graph, predecoder, adapters, paired
  experiment harness, statistics, and latency analysis.
- `src/yoked/decoding/oracle/` — full-graph oracle, retained-shot replay, and
  the B1 policy-audit pipeline.
- `tests/` — the test tree, mirroring the source packages.
- `tools/` — command-line entry points for collection, diagnosis, analysis,
  and plotting.
- `docs/` — implementation plans and immutable protocol manifests.
- `experiments/` — experiment specifications and the mutable status index.
- `assets/` — released paper data and figures.
- `out/` — generated artifacts; ignored by Git.

For a reader-oriented map of the main modules and data flows, see
[`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md).

## Reproducing paper figures

Run `./make_paper_figures` to regenerate the plots backed by released CSV data.
The focused Figure 8 workflow supports plotting, a small full-circuit smoke
test, validation-grid generation, and resumable sampling:

```bash
./reproduce_fig8_1d plot-paper-data
PROCESSES=32 THREADS_PER_PROCESS=1 ./reproduce_fig8_1d smoke
```

The release lacks the internal correlated-matching and gap-simulation tools
needed to regenerate all samples. `REPRODUCING_FIG8_1D.md` distinguishes exact
released-data reproduction from fresh runs using the available decoder stack.

## Tests

```bash
python -m pytest -q
python -m pytest -q tests/yoked/decoding
```

The suite is intentionally outside `src/`, so browsing the implementation does
not interleave production modules and tests. Test modules mirror the package
layout to keep the corresponding coverage easy to find.

## Citation

If this repository contributes to published work, cite the version of record:

> Craig Gidney, Michael Newman, Peter Brooks, and Cody Jones. “Yoked surface
> codes.” *Nature Communications* **16**, 4498 (2025).
> https://doi.org/10.1038/s41467-025-59714-1

Machine-readable metadata is available in [`CITATION.cff`](CITATION.cff); the
corresponding preprint is [arXiv:2312.04522](https://arxiv.org/abs/2312.04522).

## Frozen experiments

Files named `*FROZEN*.json` are immutable provenance records, not templates.
They authenticate a particular implementation commit, source layout, runtime,
and sampling design. After changing implementation code, create a new
versioned protocol before any new protocol-governed scientific collection;
never rewrite an older frozen manifest or reuse its output directory.
