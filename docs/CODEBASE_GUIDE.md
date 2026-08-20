# Codebase guide

This guide is the shortest path from an experiment question to the code that
answers it. Private module names describe implementation boundaries; the
command-line tools are the supported workflow entry points.

## Core decoding path

```text
yoked circuit / detector error model
        |
        v
_promatch_layout.py   detector roles and L1 domains
        |
        v
_promatch_graph.py    canonical matching edges and domain-local views
        |
        v
_promatch.py          staged proposals, transactions, and predecode result
        |
        v
_promatch_decoder.py  PyMatching/Sinter adapters and residual decoding
```

The core is intentionally independent of experiment artifact I/O. Start with
the module docstring in `_promatch.py`, then read layout and graph compilation
before following a staged proposal.

## Paired ProMatch experiment

| Concern | Module or command |
| --- | --- |
| Fixed-shot schedules, collection, provenance, and resume | `src/yoked/decoding/_promatch_experiment.py` |
| Paired tables, intervals, and power calculations | `src/yoked/decoding/_promatch_stats.py` |
| Fail-closed accuracy/workload analysis | `src/yoked/decoding/_promatch_analysis.py` |
| Controlled timing and hierarchical bootstrap | `src/yoked/decoding/_promatch_latency*.py` |
| User-facing collection and analysis | `tools/benchmark_promatch_l1`, `tools/analyze_promatch_l1` |

Detector and observable arrays are sampled once per declared batch and passed
to both decoder arms. Ledgers are immutable batch records; resume computes the
set difference between declared and completed batch IDs.

## Oracle and policy-audit path

The oracle package is isolated under `src/yoked/decoding/oracle/` because it is
a diagnostic layer, not part of the production predecoder:

| Module | Responsibility |
| --- | --- |
| `full_graph.py` | Reconstruct and certify deterministic complete-graph PyMatching solutions without accepting sampled observables. |
| `replay.py` | Apply oracle decisions to in-memory ProMatch trajectories and retained-shot replay. |
| `policy_audit.py` | Describe every B1 proposal, its local context, oracle certificate, and counterfactual alternatives. |
| `policy_experiment/` | Freeze/validate the B1 protocol, collect deterministic worker shards, and verify artifact integrity. |
| `policy_analysis/` | Load authenticated shards and build downstream tables, reports, and plots. |
| `policy_casebook.py` | Expand selected detector-only states and authenticate final casebook artifacts. |

The critical visibility boundary is simple: oracle and candidate-selection APIs
receive a compiled graph and detector syndrome, never the shot's actual logical
observables. Ground truth is joined only after predictions are fixed.

## Tests

`tests/` mirrors `src/`:

```text
tests/gen/...
tests/yoked/...
tests/yoked/decoding/oracle/...
```

Use the mirrored path to find coverage for a module. `pytest.ini` adds `src/`
to the import path and uses importlib collection so identically named test
modules in different subdirectories do not collide.

## Where to make changes

- Circuit geometry or schedules: `src/gen/` and `src/yoked/`.
- ProMatch eligibility, stage ordering, or transactions: `_promatch.py`.
- Detector/domain assumptions: `_promatch_layout.py`.
- Canonical edge support: `_promatch_graph.py`.
- Sampling, resume, manifests, or artifact integrity: the relevant experiment
  module or package (`_promatch_experiment.py` or `oracle/policy_experiment/`).
- Tables or inference only: the corresponding analysis module/package or
  `_promatch_stats.py`.
- CLI wording or argument routing: `tools/`; keep scientific logic in `src/`.

Changing any authenticated implementation file creates a new experiment
implementation. Preserve old frozen protocols and outputs, then freeze a new
version before any protocol-governed scientific sampling.
