# Experiment status

This page is the mutable dashboard for the experiment specifications in this
directory. The long specifications describe contracts and rationale; this
index records what has actually been implemented or run.

Status as of **2026-08-30**:

| Experiment | Status | Scientific role | Primary artifacts |
| --- | --- | --- | --- |
| Figure 8 1D-yoked reproduction | Implemented and exercised. Released panels can be replotted; fresh validation uses the open decoder path described in the guide. | Reproduction and validation workflow. | `out/fig8_1d/` |
| Parameterized paired Figure-8b GCP sweep | Infrastructure implemented; no production campaign has been sampled by this change. `p` and exact shots per cell are frozen at campaign creation. | Non-claim-bearing paired characterization of U0-direct versus PU-window on the 16-cell full-circuit grid. | Persistent GCP runtime under `$YSC_GCP_RUNS_ROOT/fig8-paired/<run-id>/`. |
| Native Pinball/ProMatch paired cloud32 sweep | AWS and GCP 32-worker infrastructure implemented and locally validated; no production cloud campaign has been sampled. The protocol fixes `p=0.002`, uses the exact same shots for U0, native ProMatch, and native Pinball V2, and permits at most 1,000,000 shots per cell. | Non-claim-bearing end-to-end YSC integration comparison on the 16-cell full-circuit grid. | Persistent GCP or AWS runtime under the launcher-specific `pinball-promatch-fig8-*32/<run-id>/` directory. |
| L1 ProMatch paired V3 pilot | Completed. The unsigned selector chose `pilot-02`, but unblinded PU-window accuracy was worse than U0-direct in all five cells. | Diagnostic pilot; no confirmatory accuracy run followed. | `out/promatch_l1_round1_v3_20260817_32p/` |
| Phase-A global-context oracle replay | Completed over retained V3 shots. | Detector-only oracle decisions followed by downstream outcome comparison; exploratory, input-preserving, and without new sampling. | `out/promatch_l1_global_context_oracle_v1/replay/` |
| B1 20,000-shot policy audit | Completed and finalized on 2026-08-19 using the frozen V2 protocol: 20,000 shots across 32 workers, followed by authenticated analysis and casebook expansion. | Exploratory policy-discovery corpus; explicitly non-claim-bearing. | Checked-in [human report](results/PROMATCH_L1_POLICY_AUDIT_20K_REPORT.md); full corpus workstation-local at `$TMPDIR/promatch-l1-policy-audit-20k-v2/`. |
| Confidence-Gated UF–MWPM at `d=7`, `p=0.003` | V1 stopped at the mandatory 1k fresh-process replay gate because the shakeout's authenticated detector bytes were not exposed to replay. V2 repairs that harness boundary and restarts with disjoint roots; no V1 characterization or latency run was launched. | Non-claim-bearing single-cell paired accuracy, workload, confidence-routing, and software-latency characterization. | [`CONFIDENCE_GATED_UF_MWPM_D7_P003.md`](CONFIDENCE_GATED_UF_MWPM_D7_P003.md). |

The released paper CSV files and figures under `assets/` are delivered by a
fresh clone. Every `out/` or `$TMPDIR` path in this table is instead a
workstation-local generated artifact ignored by Git; the table records where
the completed campaign was retained, not a public download location.
The checked-in B1 report is a byte-for-byte copy of the finalized generated
report (SHA-256
`acf3e2475752c243fbd032315028002b84e6c4bb986c02b3b936de3b1a151227`),
providing a small reviewable result while the complete authenticated corpus
remains outside Git.

## Documents and commands

- Figure 8 and paired ProMatch workflow:
  [`../REPRODUCING_FIG8_1D.md`](../REPRODUCING_FIG8_1D.md),
  `reproduce_fig8_1d`, `tools/benchmark_promatch_l1`, and
  `tools/analyze_promatch_l1`.
- Parameterized paired GCP sweep:
  [`PROMATCH_FIG8_PAIRED_GCP_SWEEP.md`](PROMATCH_FIG8_PAIRED_GCP_SWEEP.md),
  `gcp/run_fig8_paired`, `tools/benchmark_fig8_paired`, and
  `tools/plot_fig8_paired`.
- Native Pinball/ProMatch paired cloud32 sweep:
  [`PINBALL_PROMATCH_FIG8_PAIRED_32.md`](PINBALL_PROMATCH_FIG8_PAIRED_32.md),
  `gcp/run_pinball_promatch_fig8`, `aws/run_pinball_promatch_fig8`,
  `tools/benchmark_pinball_promatch_fig8`, and
  `tools/plot_pinball_promatch_fig8`.
- V3 diagnosis and global-context oracle design:
  [`PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md`](PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md)
  and `tools/diagnose_promatch_l1`.
- B1 policy audit:
  [`PROMATCH_L1_POLICY_AUDIT_20K.md`](PROMATCH_L1_POLICY_AUDIT_20K.md),
  `tools/benchmark_promatch_policy_audit`, and
  `tools/analyze_promatch_policy_audit`.
- Confidence-Gated UF–MWPM initial experiment:
  [`CONFIDENCE_GATED_UF_MWPM_D7_P003.md`](CONFIDENCE_GATED_UF_MWPM_D7_P003.md),
  `tools/benchmark_patch_uf_mwpm`, and `tools/analyze_patch_uf_mwpm`.

## Provenance rule

Frozen JSON manifests are historical records. They name and hash the exact
source layout at their recorded implementation commit. A later cleanup may
move or document that source without changing the historical experiment; use
Git to inspect or check out the recorded commit when reproducing the old
protocol exactly.

Any new scientific collection requires a clean implementation commit, a fresh
probe where the protocol requires one, and a newly versioned frozen manifest.
Do not edit an existing frozen JSON file to make it match newer code.
