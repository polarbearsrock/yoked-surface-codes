<!-- format: promatch-l1-policy-audit-human-report-v1 -->
# ProMatch L1 B1 policy-audit report

Experiment `f0c097d461dd039c0f4da2c3a0f9c98fdf2185e84c06a180a67c22f88f4f92b4`; cell `b1-d7-n6-y2-r28-p0.002`.

This deterministic report is generated only from the authenticated downstream analysis. It does not reconstruct sampling or decoding. All associations and explanations below are hypothesis-generating, not causal proof.

## Population and denominators

- Physical shots: 20,000 across 32 workers.
- Nonzero-event shots: 20,000 / 20,000 (100.000%); zero-event shots: 0 / 20,000 (0.000%).
- Predecoder-activated shots: 20,000 / 20,000 (100.000%).
- Shots with at least one unsafe durable original commitment: 15,235 / 20,000 (76.175%).
- Unsafe durable original commitments (the denominator for context, visibility, and counterfactual summaries): 34,148.
- U0/shadow prediction-discordant shots: 1,906 / 20,000 (9.530%).

## Arm errors and detector-event workload

Logical-error denominators are physical shots. Workload is a ratio of sums over the same physical shots; its denominator is the summed original detector-event count.

| Arm | Logical errors | Final/original detector events | Durable / provisional / rollback-lost events |
| --- | ---: | ---: | ---: |
| U0 | 693 / 20,000 (3.465%) | reference; not separately tabulated | not applicable |
| shadow | 2,245 / 20,000 (11.225%) | 8,315,659 / 10,570,353 = 0.786696 | 2,254,694 / 2,254,694 / 0 |
| O-cost transactional | 693 / 20,000 (3.465%) | 8,315,675 / 10,570,353 = 0.786698 | 2,254,678 / 2,254,690 / 12 |
| O-frame transactional | 693 / 20,000 (3.465%) | 8,315,675 / 10,570,353 = 0.786698 | 2,254,678 / 2,254,690 / 12 |
| O-frame partial | 693 / 20,000 (3.465%) | 8,315,663 / 10,570,353 = 0.786697 | 2,254,690 / 2,254,690 / 0 |

For U0 versus shadow, regressions were 1,668 / 20,000 (8.340%) and recoveries were 116 / 20,000 (0.580%).
O-frame transactional and partial predictions were authenticated as exactly equal to U0 predictions.

## Locally observable policy clues

These are candidate clues available at the L1 decision surface, not labels showing whether a commitment was truly safe.

- A same-stage local competitor was recorded for 868,355 / 1,127,347 (77.026%) shadow commitments; 258,992 / 1,127,347 (22.974%) had none and 0 / 1,127,347 (0.000%) were unrecorded.
- `L1-local-dynamic` fields appeared on 34,148 / 34,148 (100.000%) unsafe states (239,036 recorded field occurrences).
- `L1-static-boundary` fields appeared on 34,148 / 34,148 (100.000%) unsafe states (102,444 recorded field occurrences).

No threshold or decision rule is licensed by these descriptive local associations.

## Nonlocal and oracle-only explanations

Certificates, matched-partner paths, support paths, and support-difference components are oracle-only explanations. Their context labels must not be treated as locally observable policy inputs, even when the label is `in-domain`.

- Exclusive support context `yoke`: 11,521 / 34,148 (33.738%) unsafe commitments.
- Exclusive support context `true-boundary`: 8,049 / 34,148 (23.571%) unsafe commitments.
- Exclusive support context `terminal`: 1,094 / 34,148 (3.204%) unsafe commitments.
- Exclusive support context `cross-window`: 7,256 / 34,148 (21.249%) unsafe commitments.
- Exclusive support context `cross-patch-or-basis`: 0 / 34,148 (0.000%) unsafe commitments.
- Exclusive support context `support-cancellation`: 1,021 / 34,148 (2.990%) unsafe commitments.
- Exclusive support context `in-domain`: 5,207 / 34,148 (15.248%) unsafe commitments.
- No exclusive support context: 0 / 34,148 (0.000%) unsafe commitments.
- Visibility `temporal-neighbor-dynamic`: 0 / 34,148 (0.000%) unsafe states (0 recorded field occurrences).
- Visibility `nonlocal-yoke-dynamic`: 0 / 34,148 (0.000%) unsafe states (0 recorded field occurrences).
- Visibility `oracle-only`: 34,148 / 34,148 (100.000%) unsafe states (136,592 recorded field occurrences).
- Visibility `posthoc-ground-truth`: 0 / 34,148 (0.000%) unsafe states (0 recorded field occurrences).

The exclusive context is only a display-priority partition; the distinct multi-label support views remain authoritative.

## Counterfactual outcomes and limitations

- `same-stage-alternative`: 25,873 / 34,148 (75.767%) unsafe commitments.
- `later-stage-alternative`: 8,275 / 34,148 (24.233%) unsafe commitments.
- `abstain-true-exhaustion`: 0 / 34,148 (0.000%) unsafe commitments.
- `censored-invalid`: 0 / 34,148 (0.000%) unsafe commitments.

Sparse-stratum rule: fewer than 100 unsafe states is insufficient for rule formulation. 1 / 4 stage strata are marked insufficient; all displayed strata remain descriptive only.
Tied-support diagnostic `equal-weight-logical-class`: 0 / 34,148 (0.000%) unsafe commitments.
Other support diagnostics are `same-pair-different-path-or-frame` 25 / 34,148 (0.073%), `disconnected-support-reconfiguration` 721 / 34,148 (2.111%), and `unclassified` 0 / 34,148 (0.000%). Disconnected component graph roles are retained as structural evidence but excluded from policy-visible candidate context. Diagnostics can overlap and therefore do not form a partition.

Sparse or tied support can make apparent context and margin patterns unstable. This audit can prioritize follow-up hypotheses; it cannot identify a causal policy rule.
