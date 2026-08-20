# Protocol and design index

The files in this directory serve two different purposes:

- Markdown documents explain implementation and experimental design.
- JSON documents are machine-validated protocols. Files containing `FROZEN`
  are immutable provenance snapshots tied to their recorded source hashes.

## Reader guides

- [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md) — module map and execution flows.
- [`PROMATCH_IMPLEMENTATION_PLAN.md`](PROMATCH_IMPLEMENTATION_PLAN.md) — L1
  ProMatch algorithm, experiment design, acceptance gates, and verification.
- [`../REPRODUCING_FIG8_1D.md`](../REPRODUCING_FIG8_1D.md) — commands for Figure
  8 and the paired ProMatch workflow.
- [`../experiments/README.md`](../experiments/README.md) — current experiment
  status.

## Protocol families

| Family | Files | Notes |
| --- | --- | --- |
| First-round paired study | `PROMATCH_FIRST_ROUND_PROTOCOL.json`, `PROMATCH_PILOT_PROTOCOL.json`, `PROMATCH_PILOT_FROZEN*.json` | V1/V2 are superseded diagnostics; V3 is the retained pilot used by later diagnosis. |
| Phase-A oracle replay | `PROMATCH_ORACLE_REPLAY_FROZEN_V1.json` | Input-preserving replay over retained V3 shots; no new sampling. |
| B1 policy audit | `PROMATCH_POLICY_AUDIT_20K_DRAFT.json`, `PROMATCH_POLICY_AUDIT_20K_FROZEN_V1.json`, `PROMATCH_POLICY_AUDIT_20K_FROZEN_V2.json` | V1 was superseded. The V2 protocol produced a finalized, non-claim-bearing 20,000-shot corpus retained locally under `$TMPDIR`; the corpus itself is not checked into Git. |

Never reformat or update a frozen manifest in place. Its canonical bytes,
source paths, hashes, and experiment identifier are part of the audit trail.
