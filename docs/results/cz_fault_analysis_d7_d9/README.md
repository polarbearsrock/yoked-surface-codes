# Detailed CZ-fault analysis

See the [report](../cz_fault_analysis_d7_d9.md). This analysis reuses the saved
SI1000 p=0.003, d=7/9, six-patch, two-ideal-yoke, rounds=4d experiment. Production
decoders and the original 100,000 shots at each distance are unchanged.

## Measurements

- `analyze.py`: delete each of the 15 CZ Pauli families independently, using
  the previous 1,024-shot stratified selections per distance. Both complete
  correlated decoders recompute their evidence with the original DEM. Baseline
  and all-CZ controls must reproduce the previous experiment. Estimates restore
  the population proportions of the four original decoder-outcome strata.
- `single_events.py`: uniformly preselect 32 of the saved UF-only failures per
  distance and delete every recorded CZ event individually. Recover each event's
  detector and observable response, verify their XOR against the previous
  response archive, and decode each event both in isolation and removed from
  the original shot. Every extracted correction is syndrome-checked. Record
  whether the event's direct graph components were available and selected in
  the original final UF pass and selected by correlated MWPM.
- `trace_cases.py`: at F90 of complete shot 76890 and F247 of complete shot
  44875, replace the one recorded event with each of the 16 Pauli products,
  preserving all other faults and updating truth. Also inspect isolated ancilla
  faults at the four CZ slots. Ordinary Stim checks each response using two
  seeds and two samples per seed.
- `summarize.py`: independently verify the direct graph components' detector
  and observable labels, summarize the single-event interventions, and plot
  the Pauli-family removal estimates. Exploratory single-event associations
  resample whole shots to preserve dependence among event deletions.

Pauli products are always written **data first, ancilla second**, independently
of circuit target order. Fault IDs in single-event records are zero-based rows
of the full physical-event log. Detector IDs are classical parity checks, not
physical qubit IDs.

## Artifacts

- `protocol.json`: Pauli-family experiment definition and source hashes.
- `d*/selection.json`: frozen four-stratum samples and raw input locations/hashes.
- `d*/decoded_pauli.npz`, `d*/summary_pauli.json`: all predictions/truth labels,
  population-weighted estimates, paired bootstrap comparisons, and conservative
  finite-population bounds.
- `d*/input_validation.json`, `d*/decode_validation_pauli.json`,
  `d*/runner_validation.json`: input, baseline, implementation, and audit checks.
- `d*/single_event_selection.json`: the 32 preselected UF-only shots.
- `d*/single_events.json.gz`: every individual event, footprint, original graph
  component status, and both decoders' predictions after removal and in isolation.
  This is gzip-compressed JSON, with no omitted event rows.
- `d*/single_event_responses.npz`, `d*/single_event_validation.json`: packed
  physical responses, independent simulation checks, and hashes of source logs.
- `d9/case_6249_F210.json`: one illustrative non-Y example extracted from the
  complete event archive after analysis.
- `single_event_summary.json`: counts by Pauli, CZ slot, footprint size,
  detector cancellation, and original component selection.
- `trace_cases.json`: the two complete-shot substitution experiments and
  circuit-frame propagation records.
- `pauli_gap.png`, `pauli_gap.svg`: marginal 95% paired-bootstrap intervals for
  the 15 family-removal effects. Conservative bounds are in the JSON summaries.
- `artifact_manifest.json`: hashes of all saved artifacts and the report.

Family deletion effects overlap and are not additive shares of the LER gap.
The 15-Pauli ranking is exploratory; marginal intervals do not establish a
unique winner. Single-event results are conditional on original UF-only
failures, with multiple dependent interventions per shot. A correcting
deletion does not make that event the sole cause of failure. The two explicitly
traced locations are selected illustrations, not a prevalence estimate.

## Reproduce

Run from the repository root. The saved raw logs and atomic response arrays
are required under the source work directory. All scratch output stays under
`$TMPDIR`.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export MPLCONFIGDIR="$TMPDIR/uf-soft-mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cz_work=$(mktemp -d "$TMPDIR/cz-fault-analysis-XXXXXX")
cz_source="$TMPDIR/physical-fault-ablation-d7-d9-3r3wpmuq/analysis"
cz_scripts=docs/results/cz_fault_analysis_d7_d9
.venv/bin/python "$cz_scripts/analyze.py" \
  --source-work-dir "$cz_source" --work-dir "$cz_work" --workers 48
.venv/bin/python "$cz_scripts/trace_cases.py" --output "$cz_work/trace_cases.json"
.venv/bin/python "$cz_scripts/single_events.py" \
  --source-work-dir "$cz_source" --work-dir "$cz_work" --workers 48
.venv/bin/python "$cz_scripts/summarize.py" --work-dir "$cz_work"
```

The recorded run used `$TMPDIR/cz-fault-analysis-YncblJ`. No new 100k-shot
evaluation or decoder implementation was performed for this investigation.
