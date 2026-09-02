# Protocol and design index

The files in this directory serve two different purposes:

- Markdown documents explain implementation and experimental design.
- JSON documents are machine-validated protocols. Files containing `FROZEN`
  are immutable provenance snapshots tied to their recorded source hashes.

## Reader guides

- [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md) — module map and execution flows.
- [`PROMATCH_IMPLEMENTATION_PLAN.md`](PROMATCH_IMPLEMENTATION_PLAN.md) — L1
  ProMatch algorithm, experiment design, acceptance gates, and verification.
- [`PINBALL_INTEGRATION_PLAN.md`](PINBALL_INTEGRATION_PLAN.md) — pinned Pinball
  reference kernel, frozen V1, stricter domain-atomic YSC V2, adaptation
  semantics, threats, and validation gates.
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
| Confidence-Gated Patch-UF–MWPM | `PATCH_UF_MWPM_D7_P003_DRAFT.json`, `PATCH_UF_MWPM_D7_P003_FROZEN_V1.json`, `PATCH_UF_MWPM_D7_P003_FROZEN_V2.json`, `PATCH_UF_MWPM_D7_P003_FROZEN_V3.json` | V1 stopped at shakeout replay. V2 passed shakeout but stopped at the 10k telemetry-cap gate. V3 adds exact 313-shot capacity evidence and fresh roots; all versions are non-claim-bearing. |
| Port-wall Patch-UF–MWPM | `PATCH_UF_MWPM_D7_P003_PORTWALL_DRAFT.json` | Same cell, policy literals, and gates as the family above, but the treatment arm is the v2 port-wall decoder (`weighted-uf-fullhistory-patchlocal-zeroframe-portwall-residual-global-mwpm-v2`, commit `4d6e686`): guard-port contact stops a lane component and defers it whole. Five fresh seed roots. Not yet frozen; non-claim-bearing. |

Never reformat or update a frozen manifest in place. Its canonical bytes,
source paths, hashes, and experiment identifier are part of the audit trail.
