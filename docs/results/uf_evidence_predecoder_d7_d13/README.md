# Evidence-only UF pre-decoding experiment

This experiment asks whether soft evidence can improve the repository's UF
decoder without changing its growth schedule, stopping rule, or peeling.
Production decoder code and earlier experiments are unchanged.

## Protocol fixed before evaluation

Use SI1000 p=0.003, six patches, two ideal yokes, CZ circuits, rounds=4d,
and d=7,9,11,13. Regenerate the original full 100,000-shot Stim calls with
seed 42 and verify their packed payloads against the archived experiments.
Text identities must also match; a separately verified trailing-newline
difference is allowed and explicitly recorded. Never call a smaller sample
the original sample. Select **5,000 rows per distance**, uniformly without
replacement with seed `2026092000+d`, before looking at any outcomes. This is
an exploratory paired pilot, with no tuning or best-variant selection.

Five variants run on every selected syndrome:

1. `plain_uf`: original graph weights and one unchanged UF solve.
2. `correlated_uf`: original UF correction, existing correlation rules, and
   the unchanged second UF solve.
3. `prior_projection_uf`: the probability-to-weight projection below applied
   to fault priors, without syndrome evidence. This controls for changing
   the weight convention.
4. `bp5_uf`: five fixed damped sum-product BP iterations, then unchanged UF.
5. `bp20_uf`: twenty fixed iterations, then unchanged UF. This is a
   prespecified larger evidence budget, not an outcome-selected fallback.

The saved correlated-MWPM predictions are a reference, never an input.
Both BP budgets are reported. Comparisons against prior projection isolate
syndrome-dependent evidence; comparisons against correlated UF assess
whether it improves on the existing hard-correction evidence layer.

## Soft evidence

Construct a binary Tanner graph from the **joint DEM error mechanisms**.
Keep a correlated mechanism as one variable even if Stim decomposes it into
several graph components. Detector targets are XORed across components;
unfired detectors also impose parity constraints. Identical mechanisms
(same graph components and detector incidence) are combined by XOR
probability. Different decompositions remain distinct DEM mechanisms.
Verify the detector and logical incidence of every mechanism against its
graph components, and verify that its prior edge marginals reconstruct the
original graph. This retains the DEM's approximation to physical noise; it
does not reconstruct exact circuit-fault posteriors.

Use flooding sum-product BP with log-likelihood ratios, check-message
damping 0.5, and message/posterior LLR clipping at +/-30. Run exactly the
specified number of iterations. BP never supplies a hard correction or an
early-exit answer. No detectors, edges, or mechanisms are pruned.

For a mechanism posterior estimate q_f and its component edges E_f, use

`w_e = -log(min(1, max(1e-15, sum(q_f for f with e in E_f))))`.

This nonnegative projection follows Appendix C of Higgott et al.,
[Improved decoding of circuit noise and fragile boundaries of tailored
surface codes](https://arxiv.org/html/2203.04948v5#A3). The prior-projection
control uses the identical formula. These weights are a decoding heuristic,
not independent calibrated edge posteriors. We use our existing UF, fixed
BP budgets, damping, and always finish with UF; this is not a reproduction
of that paper's complete belief-find decoder.

## Measurements and safeguards

Report whole-shot failures, the historical normalized LER, paired repairs
and regressions, paired uncertainty, and distance dependence. Truth is
attached only after decoding. No result-conditioned row selection enters
the LER estimates. A separately labelled conditional diagnostic checks
logical-class accessibility on baseline disagreements.

Validate BP against exact enumeration on acyclic factor graphs and an
independent Python implementation, including zero messages and high-degree
checks. Check native UF forests/corrections against production Python UF;
check the two-pass baseline against saved predictions on every selected
shot. Every extracted correction must reproduce the full syndrome, and
every final prediction must satisfy the ideal-yoke parity. Verify that
thread count does not change any decoded value.

Record preprocessing and final-UF times separately, including both passes
in the correlated baseline. Batch wall times and isolated serial timings
are software measurements, not FPGA latency or a hardware speedup claim.
Record model sizes and message-update counts to expose preprocessing cost.

Builds, samples, caches, and checkpoints live under `$TMPDIR`. Compact
results, verification records, the protocol, and a report are retained here.
The runner binds checkpoints to source, configuration, and input hashes.

## Reproduce

From the repository root:

```bash
work_dir=$(mktemp -d "$TMPDIR/uf-evidence-XXXXXX")
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TMPDIR/uf-evidence-mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python docs/results/uf_evidence_predecoder_d7_d13/verify.py --work-dir "$work_dir"
.venv/bin/python docs/results/uf_evidence_predecoder_d7_d13/run.py \
    --work-dir "$work_dir" --output "$work_dir/results" --threads 32
.venv/bin/python docs/results/uf_evidence_predecoder_d7_d13/report.py "$work_dir/results"
```
