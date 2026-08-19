# ProMatch L1 20,000-Shot Human-Interpretable Policy Audit

- **Status:** infrastructure implemented; smoke, probe, freeze, and collection
  not yet started
- **Date:** 2026-08-19
- **Claim-bearing:** no; exploratory policy-discovery corpus
- **Scientific scope:** one fresh `d=7, p=0.002` joint-yoked cell
- **Sampling:** exactly 20,000 fixed, shot-paired samples
- **Execution:** exactly 32 worker processes, 625 shots per worker, and one
  native numerical thread per worker

This document specifies a narrow follow-up to the completed retained-shot
global-context oracle replay. Its purpose is to let a human understand why the
current local ProMatch-style policy makes harmful choices, what deterministic
alternative was available in the same local state, and which information would
have been required to make that alternative choice.

This is not a machine-learning experiment. It does not fit a classifier,
optimize a black-box score, or automatically select a production policy. The
full-graph oracle is used as a diagnostic microscope. The outputs are explicit
decision ledgers, conditional tables, distributions, and representative graph
diagrams from which a person can formulate simple veto, reranking, or
abstention rules.

The experiment is called **B1** below. It is deliberately narrower than the
larger architecture program in
[`PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md`](PROMATCH_L1_GLOBAL_CONTEXT_ORACLE.md):
it does not include native per-patch projection, complementary gaps, the
`p=0.001` headroom factorial, a learned or manually tuned guard, a holdout, or
latency claims.

The checked-in implementation now covers the decoder audit, immutable
collection, offline analysis, deterministic casebook selection, detector-only
casebook replay, and authenticated finalization described below. This document
also remains the preregistration for work that has **not** happened: no B1
shots, probe measurements, frozen protocol, or scientific conclusions exist
yet.

## Contents

1. [Decision summary](#1-decision-summary)
2. [Questions and interpretations](#2-questions-and-interpretations)
3. [Frozen physical cell and sampling contract](#3-frozen-physical-cell-and-sampling-contract)
4. [Decoder arms](#4-decoder-arms)
5. [Unit of analysis and terminology](#5-unit-of-analysis-and-terminology)
6. [Original-state counterfactual audit](#6-original-state-counterfactual-audit)
7. [Candidate and state ledger](#7-candidate-and-state-ledger)
8. [Support-component context taxonomy](#8-support-component-context-taxonomy)
9. [Information-visibility taxonomy](#9-information-visibility-taxonomy)
10. [Endpoints and statistics](#10-endpoints-and-statistics)
11. [Required tables and plots](#11-required-tables-and-plots)
12. [Deterministic casebook](#12-deterministic-casebook)
13. [Timing, storage, and launch gates](#13-timing-storage-and-launch-gates)
14. [Fatal correctness gates](#14-fatal-correctness-gates)
15. [Discovery and holdout discipline](#15-discovery-and-holdout-discipline)
16. [Artifact and provenance contract](#16-artifact-and-provenance-contract)
17. [Implementation checklist](#17-implementation-checklist)
18. [Required tests](#18-required-tests)
19. [Interpretation and routing](#19-interpretation-and-routing)
20. [Non-goals](#20-non-goals)

---

## 1. Decision summary

B1 asks one operational question at every decision made by the frozen V3 local
predecoder:

> If the original candidate does not preserve the deterministic full-joint U0
> answer, what is the first candidate that would have preserved U0 under the
> unchanged local state and unchanged ProMatch ordering?

For every unsafe original commitment, the audit will produce exactly one of
three answers:

1. **next candidate in the same stage**;
2. **candidate from a later stage**, after all earlier eligible candidates are
   vetoed; or
3. **abstain**, because the exact local candidate generator reaches true
   exhaustion without an O-frame-safe candidate.

The term “better” in this document means only **U0-preserving under the frozen
O-frame certificate**. It does not mean physically correct, ground-truth
optimal, or lower latency.

The main run uses exactly 20,000 fresh shots at `d=7, p=0.002`. All decoder
arms see the same sampled detector and observable arrays. Actual observables
are unavailable to all oracle and candidate-selection APIs and are joined only
after decoding.

The B1 run contains the frozen V3 joint baseline and three sequential oracle
arms. In addition, it performs a non-mutating, original-state counterfactual
veto chain for every O-frame-unsafe shadow commitment. The counterfactual chain
is uncapped: it ends only at the first safe candidate or at the existing
stepper's genuine terminal exhaustion.

## 2. Questions and interpretations

### 2.1 Primary questions

1. How often does the current V3 policy durably commit a path that is globally
   cost-incompatible or deterministic-frame-incompatible?
2. Which ProMatch stage, local state, window position, and omitted-context
   component accompany those unsafe commitments?
3. When an unsafe candidate is vetoed in the exact unchanged state, how often
   is the first safe action:
   - the next candidate in the same stage;
   - a candidate from a later stage; or
   - abstention?
4. Which **human-readable, locally visible** differences separate the original
   candidate from its first safe alternative?
5. How much durable detector-event removal remains under sequential O-frame
   transactional and partial policies?
6. Are failures of selection dominant, or are states with no safe candidate
   common enough to implicate candidate generation itself?

### 2.2 Diagnostic conclusions that B1 may support

B1 may support statements such as:

- unsafe decisions are concentrated in a named stage or time-window region;
- a large fraction of unsafe first choices have a safe second choice under the
  same local generator;
- boundary or temporal metadata is associated with unsafe decisions;
- yoke-dependent decisions cannot be resolved by a strictly patch-local L1
  policy and should be deferred;
- abstention is preferable in a specific, interpretable class of ambiguous
  local states; or
- the candidate family is insufficient because true exhaustion frequently
  follows an unsafe first choice.

Every such statement is exploratory and conditional on this `d=7, p=0.002`
cell.

### 2.3 Claims B1 cannot support

B1 cannot establish:

- a production policy's accuracy or latency;
- low-noise behavior at `p=0.001` or logical rates near `10^-12`;
- that U0 is physically correct;
- a true L1/L2 hierarchical decoder;
- that a context label is uniquely causal under matching degeneracy; or
- generalization of a rule formulated after inspecting this corpus.

### 2.4 Expected diagnostic yield

The earlier V3 `d=7, p=0.002` cell observed 16,239 regressions and 1,495
recoveries in 200,000 shots. If the new corpus behaves similarly, 20,000 shots
would contain roughly 1,624 regressions and 150 recoveries. These are planning
expectations, not acceptance criteria and not reasons to stop early.

## 3. Frozen physical cell and sampling contract

### 3.1 Cell

The sole scientific cell is:

```text
distance:             7
patches:              6
yokes:                2
rounds:               28
physical_error_rate:  0.002
circuit_style:        cz
noise_model:          SI1000
```

The frozen protocol must include the complete generator arguments rather than
relying only on these display fields. It must hash the generated circuit,
undecomposed and decomposed detector error models, complete matcher edge table,
L1 layout, and domain graphs.

### 3.2 Fixed-N sampling

The scientific sample size is exactly 20,000 shots, independent of interim or
final outcomes. `MAX_ERRORS` must be unset.

Work is partitioned into exactly 32 deterministic worker shards:

```text
worker_count:       32
shots_per_worker:   625
total_shots:        20,000
```

Worker `w` owns global shot IDs:

```text
625*w through 625*w + 624, inclusive
```

The protocol freezes one sampling seed root and a versioned, domain-separated
seed-derivation function. Each worker's Stim seed is derived only from:

```text
experiment_id + cell_id + worker_id + sampling_seed_root
```

No seed may depend on results, scheduling order, process ID, wall time, or a
Python hash.

### 3.3 Shot pairing

Each detector/observable sample is generated once and passed unchanged to all
arms. A physical shot identity contains:

```text
experiment_id, cell_id, worker_id, worker_shot_index, global_shot_id,
stim_seed, circuit_sha256
```

The worker stores the packed detector sample and packed actual-observable
sample once in the shot ledger, plus their digests, before decoding. Every arm
row references the same physical identity and input digest. The detector sample
is retained for bit-exact replay and casebook expansion. The actual-observable
sample lives in a separately named post-hoc field and is not passed into the
audit, oracle, context, or casebook-selection APIs.

### 3.4 Execution controls

The scientific run must:

- be launched with exactly 32 configured worker processes; a fresh smoke,
  probe, or scientific run must observe 32 distinct worker PIDs, while a
  whole-shard resume may have fewer missing worker tasks and therefore fewer
  newly observed PIDs;
- force `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `BLIS_NUM_THREADS` to
  `1` before importing NumPy or PyMatching in parent or workers;
- use a clean worktree at a config-only commit over the implementation commit;
- write to a new output root;
- never write transient files outside `$TMPDIR`;
- never run concurrently with another campaign if the combined worker count
  would exceed 32; and
- never write into or resume an immutable `promatch_l1_round1*` corpus.

## 4. Decoder arms

### 4.1 Frozen arms

| Display label | Canonical arm ID | Candidate behavior | Final backend |
| --- | --- | --- | --- |
| `U0-joint` | `u0-joint-y2` | No predecode | Complete `yokes=2` ordinary uncorrelated PyMatching |
| `PU-V3-shadow` | `pu-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2-shadow` | Exact V3 commits, globally scored without intervention | Same complete joint matcher |
| `PU-O-cost-tx` | `pu-ocost-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2` | Accept only numerically cost-compatible proposals | Same complete joint matcher |
| `PU-O-frame-tx` | `pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2` | Accept only cost- and frame-compatible proposals; whole-domain rollback | Same complete joint matcher |
| `PU-O-frame-partial` | `pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2` | Same certificate; retain certified prefix on exhaustion | Same complete joint matcher |

The V3 settings remain fixed:

```text
window length:       d rounds
window overlap:      none
residual HW target:  10 per patch/basis/window domain
stages:              1, 2, 3, 4 in existing order
boundary policy:     disabled
observable policy:   zero-frame
final matcher:       complete joint ordinary PyMatching
```

There is no correlated-matching arm.

### 4.2 Shared computation

Implementations may share immutable graph objects and per-shot oracle solution
caches. Shared computation must not change arm semantics. Each arm must remain
replayable from its ledger, and cached versus uncached results must be verified
on a frozen deterministic subset.

### 4.3 Invariants versus estimands

- `PU-V3-shadow` must reproduce the frozen V3 PU algorithm exactly.
- Both O-frame arms must be bit-identical to U0 on every shot by construction.
- O-cost requires a full paired accuracy table because cost compatibility does
  not force deterministic logical-frame equality.
- Oracle runtime is diagnostic overhead and must never be presented as a
  decoder speedup.

## 5. Unit of analysis and terminology

### 5.1 Hierarchy

```text
experiment
└── one physical cell
    └── worker shard
        └── physical shot
            └── trajectory arm
                └── L1 domain: patch × basis × time window
                    └── active-state decision
                        └── candidate proposal
```

Rows from different proposals in one shot are correlated. Statistical
resampling must keep the complete shot together.

### 5.2 Original shadow state

An **original shadow state** is the complete syndrome, accumulated observable
frame, local domain state, and stepper history immediately before one path that
the unmodified V3 predecoder commits.

It includes any earlier V3 commitments in that shot. It does not include later
commits and is not replaced by the state encountered by an oracle trajectory.

### 5.3 Unsafe commitment

An original commitment is **O-frame unsafe** when the frozen full-graph oracle
rejects it because:

1. its forced candidate-plus-optimal-residual cost has positive cost excess
   beyond tolerance; or
2. it is numerically cost-compatible but changes the deterministic U0 frame.

The ledger keeps these two classes separate:

```text
positive-cost-excess
cost-compatible-frame-conflict
O-frame-safe
```

### 5.4 First safe alternative

The **first safe alternative** is the first proposal emitted after vetoing the
unsafe original proposal and any subsequent unsafe proposals in the exact
existing stepper order, while holding the original complete syndrome, frame,
and local active state unchanged.

It is not the globally best alternative, the lowest-weight alternative among
all stages, or the physically correct correction.

### 5.5 Abstention

**Abstention** means the cloned stepper reaches its genuine existing terminal
exhaustion after every emitted alternative is vetoed. It does not include a
budget, timeout, storage limit, crash, or truncated ledger. Any such truncation
is **censored** and invalidates an uncapped result.

## 6. Original-state counterfactual audit

This audit is separate from all sequential oracle arms. It answers what the
local generator could have done differently at the exact state where the
original policy made its choice.

### 6.1 Required algorithm

For each original shadow commitment:

1. Capture the complete predecision syndrome and frame fingerprints and the
   exact local `DomainProposalStepper` state before the commitment.
2. Construct an independent clone of that stepper and state.
3. Call `next_proposal()` on the clone.
4. Require the emitted proposal signature to equal the original V3 commitment
   in domain, stage, ordered endpoints, canonical edge IDs, decision weight,
   detector boundary, and observable frame.
5. Record the original candidate's local competitor metadata and O-cost and
   O-frame evaluation.
6. If the original is O-frame-safe, stop; no counterfactual oracle chain is
   required.
7. If the original is unsafe, call `veto()` on the clone. Assert that the
   complete syndrome, accumulated frame, and local active detector set are
   unchanged.
8. Repeatedly request the next deterministic proposal and evaluate it against
   the **same unchanged complete syndrome and frame**.
9. Stop at the first O-frame-safe proposal. Record it as the first safe
   alternative without accepting it into the original shadow trajectory.
10. If the clone terminalizes first, record genuine abstention and its exact
    V3 exhaustion/fallback reason.

The chain is uncapped. `veto_budget=None` is part of the frozen scientific
protocol. A repeated proposal signature at the same active-state fingerprint,
a cycle, or any nonterminal exit is fatal.

### 6.2 Ordering semantics

The operational ordering is defined by the existing stepper, not by a new
post-hoc sort:

1. stage priority `1, 2, 3, 4`;
2. the exact current `_Candidate.key` within a stage;
3. state-scoped blacklisting after a veto; and
4. advancement to the next stage only after all unvetoed proposals in the
   current stage are exhausted.

A veto does not change the active state. An acceptance would restart at Stage
1, but the one-state counterfactual stops at the first safe proposal and never
accepts it. Sequential consequences of accepting alternatives are measured by
the named O-frame arms, not by this counterfactual record.

The operational alternative rank is one-based and includes the original:

```text
rank 1: original unsafe candidate
rank 2+: candidates encountered after successive vetoes
```

Thus the first safe alternative rank is always at least two.

### 6.3 Action classification

For an unsafe original candidate, assign exactly one terminal action:

- `same-stage-alternative`: the first safe candidate has the original stage;
- `later-stage-alternative`: it has a larger stage number;
- `abstain-true-exhaustion`: no safe proposal exists in the emitted chain; or
- `censored-invalid`: the chain ended for any other reason.

An earlier-stage alternative cannot occur because vetoing an unchanged state
does not restart stage selection. If observed, stop.

### 6.4 Trajectory separation

Every proposal row has one of these origins:

```text
shadow-original
shadow-original-state-counterfactual
sequential-o-cost-tx
sequential-o-frame-tx
sequential-o-frame-partial
casebook-exhaustive
```

Original-state counterfactual and sequential-oracle state distributions must
never be pooled. Oracle acceptances change future states; shadow commitments
follow the original trajectory even after an unsafe decision.

### 6.5 Cheap local competitor metadata for safe decisions

To compare safe and unsafe first choices, record the first same-stage
competitor for every original shadow state without globally scoring it when the
original is safe. This uses a disposable clone: emit and verify the original,
veto it, and inspect the next emission only if it remains in the same stage.
It must not mutate the shadow trajectory or create an oracle call.

Record the absolute and relative decision-weight margin. A missing same-stage
competitor is represented by `null`, never infinity or zero.

## 7. Candidate and state ledger

### 7.1 Identity and provenance fields

Every proposal or counterfactual row contains:

```text
schema, experiment_id, cell_id, worker_id, global_shot_id,
worker_shot_index, physical_input_sha256, arm_id, trajectory_origin,
graph_fingerprint, proposal_sha256
```

The shot row, rather than every proposal row, owns
`packed_detectors_hex`, `packed_actual_observables_hex`, their bit counts, and
their independent SHA-256 digests. Proposal and domain rows reference that
shot identity. Storing only an input digest is insufficient because selected
states must be replayable without resampling.

### 7.2 State fields

```text
complete_pre_state_fingerprint
complete_post_decision_state_fingerprint
local_active_state_fingerprint
patch_id, basis, window_id
window_round_start, window_round_stop
global_detector_hw
domain_initial_hw, domain_current_hw, residual_hw_target
accepted_prefix_length
state_veto_count_before, state_veto_count_after
state_stage_candidate_count
state_total_candidate_count, if exactly available
```

Counts unavailable from the exact stepper are `null`; they must not be
estimated from a truncated chain.

### 7.3 Candidate fields

```text
proposal_signature
stage
within_state_stage_rank
operational_veto_chain_rank
ordered_endpoints
canonical_edge_ids
canonical_edge_count
detector_boundary_ids
observable_frame
decision_weight and decision_weight_hex
canonical_path_weight and canonical_path_weight_hex
path_weight_agreement
```

### 7.4 Locally understandable fields

For both endpoints where applicable:

```text
local_degree
eligible_incident_candidate_count
stage isolation/safety/singleton predicates
round offset from window start
round offset from window end
round offset from circuit terminal
static distance to true boundary
static distance to candidate partner
```

For the candidate/state:

```text
same_stage_competitor_exists
same_stage_competitor_weight
absolute_weight_margin
relative_weight_margin
candidate multiplicity
whether either endpoint is on the first/last round of its window
```

Every field also carries or references the visibility class from Section 9.

### 7.5 Oracle diagnostic fields

```text
base_support_edge_ids
residual_support_edge_ids
candidate_support_edge_ids
base_matched_active_pairs
base_matched_partner_labels
base_support_path_labels
base_backend_weight, base_support_fsum
residual_backend_weight, residual_support_fsum
forced_composite_fsum
cost_excess, cost_excess_hex
tau_k, tau_k_hex
cost_classification
base_frame, candidate_frame
frame_compatible
oracle_policy_accepts
oracle cache hit/miss and solution IDs
```

The numerical contract and tolerance sensitivity are inherited exactly from
the frozen Phase-A oracle unless a new pre-freeze characterization requires a
stop. Tolerances must never be widened after observing outcomes.

### 7.6 Decision and transaction fields

```text
decision: shadow-commit | accept | veto | inspect-only
provisional, durable, rolled_back
domain_terminal_status
fallback_reason
exhaustion_kind
first_safe_alternative_proposal_sha256
first_safe_rank
terminal_action
events_removed_if_committed
```

## 8. Support-component context taxonomy

Immediate support adjacency is not the same thing as the active-defect pairing
chosen by MWPM, and neither one alone is sufficient to describe why a
candidate conflicts with the complete correction. B1 therefore records three
separate views and never aliases them:

1. **matched active partners**, obtained from the frozen PyMatching
   `decode_to_matched_dets_array` API for the unchanged complete syndrome;
2. **selected-support paths**, obtained by traversing the complete connected
   component of the selected base correction support from each candidate
   endpoint, including inactive intermediate detectors; and
3. **support-difference components**, defined below from the base and forced
   corrections.

The first view answers which active defect or true boundary MWPM paired with a
candidate endpoint. The second answers where the selected correction actually
travels. The third explains how forcing the local candidate changes the
selected correction. These can disagree under path and matching degeneracy,
so all three are retained.

### 8.1 Support-difference construction

For one evaluated candidate let:

- `B` be the square-free canonical support returned by full MWPM for the
  current syndrome;
- `P` be the candidate's asserted square-free support; and
- `R` be the square-free canonical support returned by full MWPM for the
  residual syndrome after the candidate boundary.

Construct the parity support of the forced correction:

```text
Q = P XOR R
```

and the support difference from the base optimum:

```text
X = B XOR Q
```

The cost calculation remains the forced concatenated sum specified by the
oracle; `X` is used only for structural explanation. Record `P ∩ R`
separately because XOR removes repeated edges that still contribute twice to
the forced composite cost.

Build an undirected graph from canonical edges in `X`. Each detector is a
vertex. Each detector-to-boundary edge receives its own unique virtual boundary
leaf keyed by canonical edge ID. Never connect all boundary edges through one
shared virtual vertex; doing so would spuriously merge unrelated components.

The **candidate-relevant components** are components that contain:

- an uncancelled candidate edge; or
- any detector endpoint of `P`.

If `X` is empty for a frame-conflicting or positive-cost proposal, or if no
candidate-relevant component can be identified, fail closed unless the entire
difference is explicitly accounted for by `P ∩ R` and a versioned
`support-cancellation` diagnostic. A correction/frame reconciliation failure
is fatal, not `unclassified`.

### 8.2 Multi-label component tags

Assign every candidate-relevant component all applicable tags:

- `yoke`: contains a yoke detector, yoke-hub incidence, or edge whose frozen
  layout role is yoke-mediated;
- `true-boundary`: contains a unique virtual true-boundary leaf;
- `terminal`: contains a detector or edge with a terminal-layer role;
- `cross-window`: contains detector support outside the candidate's frozen
  `(patch, basis, window)` time interval or spans more than one window ID;
- `cross-patch-or-basis`: contains physical detector roles from more than one
  patch or basis;
- `in-domain`: every physical detector and edge lies in the exact candidate
  patch, basis, and window and none of the preceding tags applies; or
- `support-cancellation`: candidate support cancels with residual support in a
  way material to the explanation and is recorded explicitly.

`in-domain` is exclusive of the omitted-context tags. The other tags are
multi-label and may overlap. The ledger stores tags per component and their
union per proposal.

Apply the same role vocabulary independently to matched active partners and to
complete selected-support paths. Store sorted unique
`matched_partner_labels`, sorted unique `support_path_labels`, and their sorted
union as `omitted_context_labels`. A support-path traversal must continue
through inactive intermediate detectors until the entire selected-support
component has been visited; inspecting only edges immediately incident to a
proposal endpoint is invalid.

For matching-pair extraction, every active detector appears exactly once in
the normalized pair ledger (paired either with another active detector or with
its own boundary sentinel). Missing, duplicated, inactive, or out-of-range
partners are fatal. Boundary sentinels in this pairing ledger remain distinct
from the unique per-edge virtual leaves used for support components.

Also record the following nonexclusive degeneracy diagnostics:

- `same-pair-different-path-or-frame`: the active pairing agrees with the
  candidate pairing but its selected support path or observable frame differs;
- `equal-weight-logical-class`: the candidate is cost-compatible but
  frame-incompatible; and
- `unclassified`: permitted only for a structurally reconciled conflict with
  none of the preceding context or degeneracy labels.

`unclassified` is a visible residual category, never a substitute for an
unknown graph role, missing support, or failed reconciliation.

### 8.3 Exclusive display label

Some plots need one category. Use this frozen priority only for display:

```text
yoke
true-boundary
terminal
cross-window
cross-patch-or-basis
support-cancellation
in-domain
```

The first present tag is the exclusive display label. All scientific tables
must also show the multi-label counts. Changing this priority after seeing the
plot is forbidden.

### 8.4 Tie and causality limitation

Support components are based on deterministic solutions returned by the frozen
PyMatching version. Equal-weight solutions may yield a different structural
decomposition. Therefore:

- positive cost excess is strong evidence that the candidate cannot be part
  of any minimum-weight correction under the oracle theorem;
- cost-compatible frame conflict identifies deterministic logical-class/tie
  behavior; and
- component labels are descriptive explanations of the selected supports, not
  unique causal attributions.

The report must repeat this limitation next to context plots.

## 9. Information-visibility taxonomy

Every explanatory field is assigned one visibility class. This separates
information a strict L1 policy could use from information available only to a
flat global decoder or oracle.

### 9.1 `L1-local-dynamic`

Available inside the current `(patch, basis, window)` domain without consulting
another domain:

- current local active detectors and domain HW;
- local graph topology and weights;
- current ProMatch stage and candidate order;
- local endpoint degree, isolation, safety, and singleton predicates;
- local candidate count, path weight, path length, and same-stage weight
  margin; and
- position within the current window.

### 9.2 `L1-static-boundary`

Precomputable from the circuit/graph without reading out-of-domain syndrome:

- shortest distance to a true boundary;
- static terminal distance; and
- static presence of a boundary or terminal route.

This is plausibly deployable at L1 but is not used by frozen V3 because its
boundary policy is disabled.

### 9.3 `temporal-neighbor-dynamic`

Requires active syndrome outside the current nonoverlapping window but within
the same patch and basis, such as the immediately previous or next window.
This is not strict current-window L1 information. Any later policy using it
must specify buffering, lookahead, and streaming latency.

### 9.4 `nonlocal-yoke-dynamic`

Requires yoke detector state or active detectors from another patch/basis
connected through the outer yoked problem. A policy using it is a flat
joint-stack guard, not an independent-patch L1 predecoder.

### 9.5 `oracle-only`

Requires a complete matching solve or its selected support:

- base/residual MWPM supports and frames;
- cost excess and O-frame label;
- support-component context derived from those solutions; and
- actual first-safe certification.

These fields explain and evaluate decisions. They cannot appear in a later
deployable rule.

### 9.6 Ground truth

Actual observable samples are a separate `posthoc-ground-truth` class. They
must not be present in the oracle, proposal generator, context classifier, or
casebook selector APIs.

## 10. Endpoints and statistics

### 10.1 Unconditional shot endpoints

Over all 20,000 shots report:

- zero-event and nonzero-event shot counts;
- predecoder activation rate;
- U0 and PU failure counts;
- paired `both-correct`, `regression`, `recovery`, and `both-wrong` counts;
- U0/PU disagreement rate;
- O-cost paired accuracy table;
- exact O-frame/U0 prediction equality;
- rollback, partial-exhaustion, and successful-domain rates;
- number of shots with at least one unsafe durable original commitment; and
- number of unsafe durable original commitments per shot.

The paired risk difference for PU and O-cost is:

```text
(regressions - recoveries) / 20,000
```

Use the repository's frozen one-sided Tango implementation where an accuracy
bound is reported. B1 remains exploratory even when this interval excludes
zero.

### 10.2 Proposal and state endpoints

Report raw numerator and denominator for:

- cost-incompatible, cost-compatible/frame-conflicting, and O-frame-safe
  shadow commitments;
- the same classes by stage and domain;
- original unsafe states ending in same-stage alternative, later-stage
  alternative, true abstention, or invalid censoring;
- first-safe operational rank;
- veto-chain length;
- no-safe rate;
- local competitor availability and weight margins;
- context component tags and exclusive display labels; and
- counts by visibility class needed to explain the conflict.

Candidate rows are not independent. Two-sided 95% intervals for proposal/state
fractions use 10,000 complete-shot bootstrap replicates, empirical type-7
quantiles, and a seed derived from a frozen bootstrap root and cell ID. Each
bootstrap draw brings every state and proposal belonging to the selected shot.

The implementation obtains each marginal interval from the exact empirical
histogram of complete-shot `(numerator, denominator)` contributions. This is
distributionally identical to resampling complete shots for that marginal and
avoids materializing a `10,000 × 20,000` index array. Metrics use independent
deterministic streams because B1 emits only marginal intervals; these draws
must not be used to infer covariance or joint contrasts.

Do not report proposal-IID standard errors.

### 10.3 Continuous distributions

For cost excess, local weight margin, veto-chain length, candidate count, and
events removed, report:

- raw empirical CDF;
- median;
- 10th and 90th percentiles; and
- 99th percentile where the denominator is at least 1,000.

Use empirical type-7 quantiles. State the denominator beside every statistic.
Do not replace missing values with zero.

### 10.4 Detector-event relief

For each PU/oracle arm calculate:

```text
R_event = sum(final residual detector HW)
          / sum(original detector HW)
```

Only durable commits affect the numerator. Provisional work discarded by a
transactional rollback is reported separately.

Use 10,000 paired complete-shot bootstrap replicates and the frozen first-round
workload method for confidence bounds. Also report:

```text
durable events removed
provisional events removed
events lost to rollback
R_relief_retained relative to original PU
```

Fewer active events are not called graph-size, operation-count, or latency
reduction.

### 10.5 Association with final disagreement

Tabulate U0/PU agreement, regression, and recovery by:

- zero, one, two, three, and four-or-more unsafe durable commitments;
- first unsafe stage;
- first unsafe exclusive context label; and
- terminal counterfactual action.

These are associations along the original trajectory, not causal effects of
one commitment. The O-frame induction invariant requires every U0/PU-discordant
shot to contain a durable frame conflict; its verification is an implementation
gate, not an empirical finding.

### 10.6 Sparse strata

Any displayed policy stratum with fewer than 100 unsafe original states is
marked `insufficient-for-rule-formulation`. Its raw examples and counts remain
visible, but no threshold or policy recommendation may be based on that
stratum alone.

## 11. Required tables and plots

All plot inputs must be regenerable from frozen ledgers. Each plot has an
adjacent machine-readable CSV or JSON table with exact counts and denominators.

### 11.1 Overview tables

1. Physical cell, hashes, shots, workers, and arm definitions.
2. Shot-paired U0/PU and U0/O-cost outcome tables.
3. Per-arm detector-event and transaction summary.
4. Certificate taxonomy by ProMatch stage:

   ```text
   positive-cost-excess
   cost-compatible-frame-conflict
   O-frame-safe
   ```

5. Counterfactual terminal action and first-safe-rank table.
6. Multi-label and exclusive support-component context tables.
7. Fatal-gate and nonfatal interpretation-checkpoint table.

### 11.2 Required plots

1. **Certificate flow.** Counts from shadow commitments to cost class, frame
   class, and counterfactual terminal action. This may be a compact flow chart
   or aligned bars; exact counts must remain readable.
2. **Unsafe fraction by stage.** O-frame-unsafe fraction with complete-shot
   bootstrap intervals and raw denominators.
3. **First conflict by stage and context.** Stacked bars over U0/PU-discordant
   shots, accompanied by the multi-label table.
4. **Cost-excess distribution.** ECDF and symmetric-log display, split by
   stage, context, and certificate class. The numeric zero/tolerance band must
   be shown.
5. **First safe action and rank.** Histogram of same-stage, later-stage, and
   abstain outcomes plus operational rank distribution.
6. **Stage transition matrix.** Original unsafe stage versus first-safe stage;
   abstention is a separate terminal column.
7. **Original versus alternative.** Paired plots for decision weight, path
   length, local weight margin, and immediate event removal. Lines connect
   candidates from the same unchanged state.
8. **Human-readable risk heatmaps.** At minimum:
   - stage × window offset;
   - stage × static-boundary competition;
   - domain HW × candidate multiplicity; and
   - stage × same-stage weight-margin bin.
9. **Disagreement association.** U0/PU disagreement, regression, and recovery
   proportions versus `0, 1, 2, 3, 4+` unsafe durable commitments.
10. **Event relief.** Per-shot residual-HW distributions and aggregate
    `R_event` for PU, O-frame-tx, and O-frame-partial. Zero-original-event shots
    remain in unconditional counts and are omitted only from per-shot ratios.
11. **Veto-chain tails.** ECDF and p90/p99/max for proposals, vetoes, oracle
    calls, and Stage-3 enumeration time per state and shot.

No regression model, decision tree, feature importance, trained score, or ML
metric belongs in B1.

### 11.3 Binning rules

Window offsets and integer HW/candidate-count bins are frozen in the protocol.
Continuous margin plots may show corpus deciles for visualization, but deciles
are explicitly descriptive and cannot become a policy threshold without a new
frozen rule specification and holdout. The exact unbinned ECDF is always
reported.

## 12. Deterministic casebook

Aggregate distributions are accompanied by a small, reproducible set of
decision diagrams. Examples are selected without actual observables or final
correctness.

### 12.1 Selection strata

From all unsafe original shadow states, form populated strata by:

```text
exclusive support-component context × original ProMatch stage
```

For every stratum with at least 20 states:

1. compute the empirical median cost excess;
2. choose the state with minimum absolute distance to that median; and
3. break ties by the lexicographically smallest SHA-256 of the state identity.

Add one state for each populated terminal action
`same-stage-alternative`, `later-stage-alternative`, and
`abstain-true-exhaustion`, selected by the same median-and-hash rule using
veto-chain length. Deduplicate identical states.

This selection is exploratory but deterministic. It does not select on
regression, recovery, or visual appeal.

### 12.2 Exhaustive slate only for casebook states

For each selected casebook state, clone the exact stepper and veto every
proposal to genuine exhaustion, globally scoring each proposal against the
same unchanged complete syndrome/frame. This is the only B1 analysis that
scores candidates after the first safe alternative.

The all-shot audit stops each unsafe chain at the first safe alternative. The
casebook exhaustive rows have `trajectory_origin=casebook-exhaustive` and must
not be pooled into all-shot rates.

This is an explicit two-stage operation. First, the outcome-blind analyzer
selects state identities using only immutable collection ledgers and writes a
selection manifest. Second, a casebook expander replays only those selected
states from the retained packed detector samples and frozen graph/code,
verifies the original proposal and state fingerprints, and writes the
exhaustive sidecar. The analyzer may then render diagrams from that sidecar.
Selection and expansion must be separate APIs so actual observables cannot
enter selection through a decoding callback.

### 12.3 Diagram contents

Each case contains:

- the local graph and active events visible to V3;
- the relevant complete-graph support-difference components;
- original chosen path;
- every veto before the first safe alternative;
- first safe alternative or abstention;
- stage, rank, local weight, cost excess, and context/visibility labels;
- domain HW before and immediate HW after each hypothetical candidate; and
- a short generated factual caption with no causal language.

The diagram distinguishes local/static/temporal/nonlocal/oracle-only
information visually and includes graph/layout hashes.

## 13. Timing, storage, and launch gates

### 13.1 Ordered preflight

Before the 20,000-shot collection:

1. run unit/property tests;
2. run a deterministic 32-shot integration smoke, one shot per worker;
3. run a disjoint deterministic 100-shot timing/storage probe with the exact
   scientific arms and counterfactual audit, using all 32 workers; workers 0--3
   receive four shots and workers 4--31 receive three shots;
4. generate the full analysis from the probe to validate schemas and plots;
5. evaluate the frozen launch gates below; and
6. only then create and commit the scientific frozen protocol.

Smoke and probe artifacts live under `$TMPDIR`, are explicitly
non-claim-bearing, and are never copied or pooled into the 20,000-shot corpus.

### 13.2 Required probe telemetry

Record separately:

- parent and worker graph compilation time;
- sampling time;
- complete five-arm shot-audit time (B1 does not attribute wall time to
  individual arms and makes no per-arm latency claim);
- counterfactual audit time;
- support-component and serialization time;
- full-MWPM calls and cache hits/misses;
- proposals and vetoes per state and shot;
- Stage-3 enumeration time;
- per-shot and per-worker p50, p90, p99, and maximum wall time;
- peak resident set size in parent and each worker; and
- compressed and uncompressed bytes per artifact row and physical shot.

### 13.3 Wall-time projection gate

Let `T_setup` be observed fixed setup/compilation wall time and `T_variable` the
remaining 100-shot probe wall time at 32 processes. Define:

```text
projected_wall_seconds = T_setup + 1.5 * (20,000 / 100) * T_variable
```

Proceed only if:

```text
projected_wall_seconds <= 7,200
```

The factor `1.5` is frozen headroom for denser tails and shard imbalance. If
the gate fails, do not launch and do not silently add a veto cap. Optimize the
implementation or write a separately named budgeted protocol.

### 13.4 Storage gate

Let `B_probe` be the total compressed scientific-format bytes produced by the
100-shot probe, excluding plots and temporary compilation caches. Define:

```text
projected_artifact_bytes = 1.5 * (20,000 / 100) * B_probe
```

Proceed only if:

```text
projected_artifact_bytes <= 20 GiB
free bytes at output filesystem >= max(40 GiB, 2*projected_artifact_bytes)
```

If this gate fails, reduce redundant serialization using graph references or
deterministic compression and repeat the probe. Do not omit required decisions,
unsafe chains, or hashes after observing their content.

### 13.5 Tail and censoring gate

The scientific arms and counterfactual chain have no proposal, veto, or oracle
call budget. The probe must complete with:

- zero censored states;
- zero repeated same-state proposal signatures;
- zero worker timeouts; and
- zero output truncations.

An operating-system failure during the main run produces an incomplete
campaign, not a censored scientific observation. It may be resumed only under
the verified shard contract in Section 16.

## 14. Fatal correctness gates

Stop and fix the implementation before interpreting results if any of the
following occurs:

1. not exactly 20,000 unique physical shot IDs and 625 verified rows per worker
   shard;
2. any arm receives a different input detector or observable sample;
3. scalar and batch U0 predictions differ;
4. shadow output differs from the frozen V3 PU implementation;
5. either O-frame arm differs from U0 on any shot;
6. negative cost excess beyond frozen tolerance;
7. backend-reported/support-`fsum`, Decimal, repeatability, or tolerance-grid
   checks fail;
8. an oracle decision changes when only actual observables are changed;
9. a veto mutates detector state, frame, accepted prefix, or active-state
   fingerprint;
10. the cloned first proposal differs from the original shadow commitment;
11. a counterfactual proposal repeats under one unchanged state, an earlier
    stage reappears, or an uncapped chain ends without a safe candidate or true
    stepper exhaustion;
12. detector-boundary, observable-frame, rollback, durability, or GF(2)
    accounting fails;
13. a U0/PU-discordant shot has no durable original frame conflict;
14. a canonical support pair is missing/ambiguous or a correction support does
    not reproduce its syndrome/frame;
15. a support component contains an unknown detector/edge role or cannot be
    classified by the frozen taxonomy;
16. cached and uncached oracle results differ on the frozen repeatability
    subset;
17. artifact row counts, canonical hashes, gzip determinism, or manifest
    digests fail; or
18. numerical thread count, process count, source/version hash, worktree, or
    config-only-commit requirements fail.

These are implementation failures, not evidence that local predecoding is
ineffective.

The campaign manifest's fatal-gate attestations are authenticated indexes into
the frozen worker evidence and the distributed runtime assertions that enforce
these gates.  They are not standalone proofs: interpretation requires the
hashed source, immutable shards, recomputed analyzer rows, and the referenced
assertions together.

The following are **not** fatal and must not alter the fixed shot count:

- fewer disagreements than expected;
- little certified event relief;
- many abstentions;
- most conflicts carrying nonlocal yoke context;
- no visible local signature separating choices; or
- PU not reproducing the earlier effect magnitude.

They are scientific outcomes to report.

## 15. Discovery and holdout discipline

### 15.1 The complete B1 corpus is discovery data

All 20,000 shots, all proposal rows, and all casebook examples are exploratory.
A person may inspect them to formulate simple rules, but no rule designed or
tuned after inspection is validated on B1.

Do not split B1 after observing outcomes and call one piece a holdout. Do not
promote unused workers, low shot IDs, or unplotted fields into validation data.

### 15.2 Permitted rule formulation after B1

Candidate rules must be written explicitly in human-readable form, for
example:

- allow only named stages;
- veto within a fixed window-edge guard band;
- require a fixed local weight margin;
- require reciprocal preference over a static true-boundary route;
- retain safe-looking early commits but abstain after a named ambiguity; or
- defer every yoke-visible ambiguity to L2.

Each rule specification must freeze:

- permitted information-visibility classes;
- exact predicates and numeric thresholds;
- tie behavior;
- action after veto;
- abstention/handoff behavior;
- transaction policy; and
- computational implementation.

Oracle labels, support components, actual observables, and final correctness
may not be inputs to a deployable rule.

### 15.3 Separate holdout

After rules are proposed, select at most a small, named set based on the
scientific rationale—not on repeated trial against new outcomes. Commit their
implementations, freeze a new disjoint seed root and protocol, and set holdout
sample size by a prospective power/precision calculation.

No tuning, threshold changes, or rule replacement occurs after holdout
decoding starts. The 100-shot timing probe and B1 shots are not pooled into the
holdout.

## 16. Artifact and provenance contract

### 16.1 Two-commit freeze

The scientific run requires:

1. **Implementation commit A:** all collector, oracle, ledger, context,
   analysis, plot, and test code.
2. **Config-only commit B:** exactly one frozen B1 JSON protocol file changed
   relative to A.

The worktree is clean at B. The protocol records both commits and verifies that
`git diff --name-only A..B` contains only its own path.

### 16.2 Frozen protocol fields

At minimum:

```text
schema, status, frozen, experiment_id
implementation_commit, config_commit, config_self_sha256
source_hashes and requirements_sha256
software_versions and execution_environment
complete cell/generator parameters
circuit, DEM, matcher, layout, domain-graph hashes
shot count, worker count, shots per worker
sampling seed root and derivation version
worker/global shot-ID schedule
arm IDs and complete decoder settings
oracle tolerances, Decimal precision, sensitivity grid
counterfactual ordering and uncapped veto semantics
context-component and display-priority definitions
visibility definitions
casebook selection algorithm
bootstrap seed roots, replicates, and quantile method
plot/table/binning specification
timing and storage gates
fatal gates
```

The configuration is canonicalized with sorted compact JSON and hashes its
semantic content with `experiment_id`/self-hash treatment defined by a tested
versioned function.

### 16.3 Output layout

Use a fresh root such as:

```text
out/promatch_l1_policy_audit_20k_v1/
├── experiment.json
├── manifest.json
├── config.json
├── shards/
│   ├── worker-00/
│   │   ├── shots.jsonl.gz
│   │   ├── proposals.jsonl.gz
│   │   ├── counterfactuals.jsonl.gz
│   │   ├── domains.jsonl.gz
│   │   └── timing.json
│   └── ... worker-31/
├── casebook/
│   ├── selection.json
│   └── expansion/
│       ├── exhaustive.jsonl.gz
│       ├── states/
│       ├── diagrams/
│       └── manifest.json
├── analysis/
│   ├── summary.json
│   ├── tables/
│   ├── plot-data/
│   └── plots/
├── COLLECTION_READY
├── ANALYSIS_READY
├── EXPANSION_READY
└── COMPLETE
```

The collector writes `COLLECTION_READY` only after every worker shard and the
campaign manifest reconcile. This marker permits the first offline analysis.
The analyzer writes `ANALYSIS_READY` after the all-shot tables, plot data,
figures, and outcome-blind casebook selection reconcile. The separate casebook
expander then writes its exhaustive selected-state sidecar. `COMPLETE` is
written last by finalization only after every manifest, count, invariant,
analysis, selection, and casebook-expansion gate passes. The analyzer must not
require `COMPLETE` before it can perform the first validation.

### 16.4 Deterministic storage

Use canonical newline-delimited JSON with sorted keys and exact float-hex
companions. Compress with deterministic gzip (`mtime=0`, frozen compression
level, no source filename). Each shard records compressed and uncompressed
SHA-256, byte count, row count, first/last shot ID, and schema version.

Large immutable graph arrays are stored once by content hash. Proposal rows
reference canonical edge IDs and graph hashes rather than duplicating edge
coordinates and weights.

Wall-clock values are never mixed into the bit-exact decision ledgers. The
collector strips them recursively into each worker's authenticated
`timing.json`. Timing bytes are explicitly nondeterministic and excluded from
the cross-run bit-exact contract; deterministic decision fields, proposal
ordering, predictions, and support identities remain in the gzip ledgers.

All temporary shard files are created under `$TMPDIR`. A completed shard is
atomically moved into its final worker directory only after validation.

### 16.5 Resume contract

Resume is permitted only after interruption and only for whole worker shards.
Before skipping a shard, verify:

- protocol/experiment/config/implementation identity;
- worker shot range and seed derivation;
- every compressed and uncompressed digest;
- row counts and input/arm pairing; and
- absence of a partial temporary file in the final directory.

A missing shard is regenerated from scratch into a new temporary path. An
invalid installed shard is a fail-closed error: the collector never deletes or
overwrites it automatically. Preserve it for diagnosis and resume into a
fresh, operator-chosen output root (or remove it only through an explicit,
separately reviewed recovery action). Never append to a shard, merge
configurations, or keep a partially decoded worker range.

### 16.6 Reproducible analysis

The analyzer reads only the frozen config, manifest, and immutable ledgers. It
must not import sampling or decoding code to reconstruct missing facts. Running
the analyzer twice produces identical tables, summaries, casebook selection,
and plot-data hashes. Rendered image metadata that cannot be deterministic is
excluded from scientific digests and documented.

## 17. Implementation checklist

### 17.1 Core decoding and counterfactuals

- [x] Add a fresh-shot B1 orchestration API around the existing graph, oracle,
      and replay primitives.
- [x] Capture or clone exact domain stepper state before each V3 shadow commit.
- [x] Assert first-emission identity with the V3 path.
- [x] Implement uncapped original-state veto-chain audit to first O-frame-safe
      candidate or true exhaustion.
- [x] Record cheap same-stage competitor metadata for every shadow decision.
- [x] Keep counterfactual inspection separate from original and sequential
      trajectory mutation.
- [x] Share oracle caches only under verified semantics.

### 17.2 Explanation data

- [x] Implement square-free `B`, `P`, `R`, `Q`, and `X` support accounting.
- [x] Build relevant support components with unique per-edge virtual boundary
      leaves.
- [x] Attach frozen detector/edge roles and multi-label context tags.
- [x] Implement exclusive display priority without discarding multi-label data.
- [x] Extract local, static-boundary, temporal, nonlocal-yoke, and oracle-only
      features with explicit visibility tags.
- [x] Reject unknown roles and reconciliation failures.

### 17.3 Parallel collector

- [x] Add deterministic 32-worker orchestration with 625 shots per worker.
- [x] Enforce one native numerical thread in parent and children before imports.
- [x] Derive worker seeds and shot IDs from the frozen versioned schedule.
- [x] Sample once per worker and feed identical arrays to all arms.
- [x] Add shot-audit, oracle-call, cache, enumeration, RSS, serialization, and
      byte telemetry; per-arm wall attribution is explicitly unavailable.
- [x] Write atomic deterministic shard artifacts and manifest digests.
- [x] Implement verified whole-shard resume.

### 17.4 Analysis

- [x] Implement complete-shot clustered bootstrap utilities.
- [x] Produce all frozen counts, denominators, intervals, ECDFs, tables, and
      plot-data artifacts.
- [x] Implement deterministic casebook selection without observables/outcomes.
- [x] Implement a separate replay-verified casebook expansion command that
      exhaustively scores only selected states after selection.
- [x] Render local-versus-complete support diagrams and timelines.
- [x] Mark sparse strata and tied-support causal limitations.
- [x] Generate a concise human-readable report that distinguishes local policy
      clues from nonlocal/oracle-only explanations.

### 17.5 Freeze and launch

- [x] Run the full test suite (522 tests passed on 2026-08-18).
- [ ] Commit implementation A and push it.
- [ ] Run 32-shot integration smoke under `$TMPDIR`.
- [ ] Run disjoint 100-shot, 32-process timing/storage probe.
- [ ] Verify the two-hour/20-GiB/free-space gates.
- [ ] Create and validate config-only commit B and push it.
- [ ] Reconfirm clean worktree, 32×625 schedule, one-thread environment, and
      `MAX_ERRORS` unset.
- [ ] Start the 20,000-shot campaign in a fresh output root.

### 17.6 Executable workflow (not yet run)

Run from the repository root with the pinned `.venv`. Every collection command
below rejects any process count other than 32. Before running the smoke or
probe, commit and push all implementation, analysis, documentation, and test
files as implementation commit A and require a clean worktree. The probe
records the current HEAD, so running it before A would make its attestation
invalid at freeze time.

```bash
source .venv/bin/activate
export TMPDIR=/data2/s2chitni/.tmp
export MPLCONFIGDIR="$TMPDIR/yoked-surface-codes-matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
unset MAX_ERRORS
mkdir -p "$MPLCONFIGDIR"

POLICY_DRAFT=docs/PROMATCH_POLICY_AUDIT_20K_DRAFT.json
SMOKE_ROOT="$TMPDIR/promatch-policy-audit-b1-smoke"
PROBE_ROOT="$TMPDIR/promatch-policy-audit-b1-probe"

tools/benchmark_promatch_policy_audit smoke \
  --protocol "$POLICY_DRAFT" --out "$SMOKE_ROOT" --processes 32
tools/benchmark_promatch_policy_audit verify \
  --protocol "$POLICY_DRAFT" --out "$SMOKE_ROOT" --mode smoke
tools/analyze_promatch_policy_audit "$SMOKE_ROOT"

tools/benchmark_promatch_policy_audit probe \
  --protocol "$POLICY_DRAFT" --out "$PROBE_ROOT" --processes 32
tools/benchmark_promatch_policy_audit verify \
  --protocol "$POLICY_DRAFT" --out "$PROBE_ROOT" --mode probe
tools/analyze_promatch_policy_audit "$PROBE_ROOT"
tools/analyze_promatch_policy_audit "$PROBE_ROOT" --verify
```

With the analyzed probe completed at the unchanged, clean implementation
commit A, freeze a new protocol; the resulting protocol file is the only
change allowed in commit B:

```bash
FROZEN_PROTOCOL=docs/PROMATCH_POLICY_AUDIT_20K_FROZEN_V1.json
tools/benchmark_promatch_policy_audit freeze \
  --protocol "$POLICY_DRAFT" \
  --probe-root "$PROBE_ROOT" \
  --out-protocol "$FROZEN_PROTOCOL"
```

Only after commit B, its push, and a final clean-worktree check may the fixed
20,000-shot corpus start:

```bash
SCIENTIFIC_ROOT=out/promatch_l1_policy_audit_20k_v1
tools/benchmark_promatch_policy_audit collect \
  --protocol "$FROZEN_PROTOCOL" \
  --out "$SCIENTIFIC_ROOT" --processes 32
tools/benchmark_promatch_policy_audit verify \
  --protocol "$FROZEN_PROTOCOL" \
  --out "$SCIENTIFIC_ROOT" --mode scientific
tools/analyze_promatch_policy_audit "$SCIENTIFIC_ROOT"
tools/analyze_promatch_policy_audit "$SCIENTIFIC_ROOT" --verify
tools/benchmark_promatch_policy_audit expand-casebook \
  --protocol "$FROZEN_PROTOCOL" --collection "$SCIENTIFIC_ROOT"
tools/benchmark_promatch_policy_audit verify-casebook \
  --protocol "$FROZEN_PROTOCOL" --collection "$SCIENTIFIC_ROOT"
tools/benchmark_promatch_policy_audit finalize \
  --protocol "$FROZEN_PROTOCOL" --collection "$SCIENTIFIC_ROOT"
```

## 18. Required tests

### 18.1 Counterfactual semantics

- first cloned emission exactly matches the original V3 proposal;
- veto does not mutate syndrome, frame, active detectors, or accepted prefix;
- candidates follow exact stage and `_Candidate.key` order;
- state-scoped blacklist prevents repetition but does not leak across states;
- first-safe rank and same/later-stage action are correct;
- true exhaustion is distinguished from censoring;
- an unsafe chain has no hidden proposal/call cap; and
- original trajectory output is unchanged by counterfactual inspection.

### 18.2 Oracle and numerical tests

Reuse all Phase-A tests and add fresh/batch coverage for:

- scalar versus batch U0;
- decode versus decoded-edge frame reconciliation;
- backend versus support weight tolerance;
- `math.fsum` versus 4096-digit Decimal reference;
- frozen tolerance-grid classification stability;
- cached versus uncached equivalence;
- repeated-run determinism; and
- no oracle API dependence on actual observables.

### 18.3 Support-component taxonomy

Synthetic graph fixtures must cover:

- in-domain reroute;
- unique true-boundary leaf;
- two unrelated boundary edges that must remain separate components;
- yoke-mediated support;
- cross-window support;
- terminal support;
- cross-patch/basis support;
- multi-label components;
- candidate/residual edge cancellation;
- empty/unknown/ambiguous support failure; and
- deterministic exclusive display priority.

They must additionally distinguish actual matched-active partners from
support adjacency, traverse selected support through inactive intermediate
detectors, normalize detector-to-boundary pairs without merging their support
components, and cover `same-pair-different-path-or-frame`,
`equal-weight-logical-class`, and the valid residual `unclassified` case.

### 18.4 Visibility fields

Tests verify that:

- L1-local features depend only on the current domain graph/state;
- changing out-of-domain syndrome cannot change an L1-local field;
- static-boundary fields do not depend on active syndrome;
- temporal and yoke fields are explicitly nonlocal; and
- oracle/ground-truth fields cannot enter a deployable-policy record.

### 18.5 Parallel collection and artifacts

- worker count above or below 32 is rejected for the scientific protocol;
- each scientific worker receives exactly 625 unique shot IDs;
- parent and worker thread environments equal one;
- all arms receive identical packed input arrays;
- deterministic seed derivation and shard regeneration are bit-exact;
- canonical JSON/gzip bytes and digests are reproducible;
- corrupt or partial shards are never resumed;
- valid complete shards can be skipped safely;
- output cannot be written into immutable pilot roots; and
- `COMPLETE` cannot appear before every gate passes.

### 18.6 Statistics, plots, and casebook

- shot-cluster bootstrap keeps all proposal rows from one shot together;
- quantiles use empirical type 7 and frozen seeds;
- zero/missing/undefined denominators remain explicit;
- R-event is a ratio of aggregate counts, not a mean of per-shot ratios;
- sparse strata are marked at the frozen threshold;
- every plot table reproduces plotted values;
- casebook selection ignores actual observables and correctness;
- median-and-hash selection is deterministic; and
- exhaustive casebook rows never enter all-shot rates.

## 19. Interpretation and routing

After a valid B1 run, use this transparent decision table:

| Observed pattern | Human interpretation | Candidate next step |
| --- | --- | --- |
| Most unsafe choices have rank-2 same-stage safe alternatives with a visible local margin/stage signature | Selection ordering is the likely bottleneck | Write one simple local reranking/veto rule and freeze a holdout |
| Unsafe choices cluster at window edges and alternatives are locally available | Strict nonoverlapping history is hiding relevant temporal context | Specify a conservative window-edge abstention or buffered temporal guard |
| Unsafe choices cluster near true boundaries using static metadata | Disabled-boundary policy is too aggressive | Specify a reciprocal boundary-preference veto |
| Most explanations require yoke/other-patch dynamic state | The decision is not genuinely L1-local | Abstain at L1 or pass soft information to L2; do not disguise a flat guard as L1 |
| True exhaustion is common after an unsafe first choice | Candidate family, not only ranking, is inadequate | Extend candidate generation or hand off earlier |
| Cost-compatible/frame-conflicting choices are material | Tie/logical-class behavior matters | Add explicit frame/tie policy; cost-only guards are insufficient |
| O-frame preserves little detector-event relief | This conservative local certificate does not reduce enough active work | Prioritize hierarchy/soft-output work over a flat guarded predecoder |
| No simple visible condition separates safe and unsafe choices | The desired decision requires more context than a transparent L1 rule provides | Prefer abstention or an L1-to-L2 interface |

These routes are judgments informed by B1, not automatically fitted policies.
Record the human decision and supporting tables before implementing the next
experiment.

## 20. Non-goals

B1 does not:

- train or evaluate an ML model;
- select policy thresholds automatically;
- test a policy on an independent holdout;
- run the `p=0.001` headroom factorial;
- build native `yokes=0` per-patch decoders or complementary gaps;
- implement L2/QDPC decoding;
- use correlated PyMatching;
- alter `HW=10`, windows, stages, boundary policy, or observable policy;
- claim the oracle improves latency;
- infer graph-size reduction from detector-event reduction; or
- modify, resume, or promote any prior immutable corpus.

The sole outcome is a reproducible, human-readable map from the current local
policy's decisions to globally certified alternatives and the information
needed to distinguish them.
