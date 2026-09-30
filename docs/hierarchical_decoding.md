# Hierarchical L1/L2 decoding of the 1D yoked surface code

Run from the repository root with `PYTHONPATH=src`. The `yoked.hierarchical`
package uses the repository's existing dependencies (Stim, PyMatching, NumPy,
SciPy, sinter) and requires no native build. Every run directory below lives
under `$TMPDIR`; only reports are committed.

The design this implements is
[the hierarchical L1/L2 spec](superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md).

The [UF-only confidence experiment](uf_soft_confidence_experiment.md) replaces
L1 matching-based confidence with UF class-cost gaps and bounded cluster gaps,
while retaining the fixed UF reference and the MWPM L2 backend. Its separate
`tools/uf_soft_experiment` driver reuses verified calibration/evaluation records.

## The fixed-reference model

A shot is decoded in two layers. **L1** decodes each of the six surface-code
patches on its own, ignoring the two yoke stabilizers, and produces for every
patch-sector:

- a **reference bit** `r[i, s]` from the patch's own decoder, either the
  repository's weighted Union Find or plain patch-local MWPM, and
- one or more **soft outputs**: the UF cluster gap, and the signed matching gap
  under the plain weights and under the correlated ones.

**L2** never re-decodes. It receives the calibrated residual-error probability
`q[i, s] = P(r[i, s] is wrong)` of each patch and the frame-adjusted syndrome
`sigma[s] = y[s] XOR parity(r[:, s])`, and uses weighted MWPM to return the maximum-weight residual
pattern `x` of the required parity. The final prediction is `f = r XOR x`.

The reference is *fixed*: it is decided once by L1 and stored, and every
configuration, policy, and report is measured against the same reference bits.
That is what makes the primary metric well posed. Among sectors where the
reference made exactly one mistake, the misattribution rate is the fraction
where the final prediction is still wrong, and two configurations may only be
compared when they share a reference, because different references define
different eligible populations.

## The six stages

`tools/hierarchical_experiment` is a thin argparse layer: it parses arguments,
calls one library function, and prints a path. Every check lives in
`yoked.hierarchical`, so any failure below can be reproduced from Python.

**Collect.** Sample or import one full Stim call, then decode the requested
parent rows into a record. The calibration set is generated; the evaluation set
is imported from the recorded four-decoder run so that it decodes the original
circuit and model rather than a regenerated one.

```bash
cd /data2/s2chitni/projects/dante/repos/yoked-surface-codes
export PYTHONPATH=src
export OUT="$TMPDIR/hier-d9-p003"
export RUN=/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha

.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/calibration" --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --rows 0:2000 --workers 16 --chunk-size 25
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/evaluation" --role evaluation \
    --recorded-run "$RUN" --rows 0:2000 --workers 16 --chunk-size 25
```

A request names exactly one sample source: `--recorded-run`, or the complete
`--distance --rounds --p --seed --shots` triple (with `--patches`, default six).
`--rows 0:2000` is half-open, like a slice: it names rows 0 to 1999. Omitting
`--rows` decodes every row of the call. `--role` accepts only `calibration` and
`evaluation`; confirmation is refused, because section 3 allows it only after
the analysis freeze and freeze verification is a later milestone.

**Verify a subset.** Compare a completed subset record with the completed full
record of the same sampling call, row for row. This is how M2 shows that a full
collection made under the current provenance contract reproduces the pilot's
subset number for number, rather than merely sharing its identities.

```bash
.venv/bin/python tools/hierarchical_experiment verify-subset --full "$OUT/evaluation" \
    --subset "$OUT/evaluation_pilot_rows" --out "$OUT/verify_evaluation.json"
```

`--full` names a collection of every row of the call and `--subset` a `--rows`
collection of the same call. Both are read through `load_record` and must share
their model, parent-sample, sampling-family, and decoder identities and their
role; the first that differs is named and nothing is written. Every subset row
is then located in the full record by its parent row id, a row the full record
does not hold being refused by name, and every stored array is compared there
exactly: no tolerance, float values bit for bit, and baselines excluded, because
a subset need not carry them and they are verified by `import-baselines` rather
than reproduced by collection. The JSON at `--out` records both records'
directories, identities, `record.npz` and manifest hashes, row summaries, and
roles, the checker's identity, and the check: per-array equality, the number of
differing rows per array, and the first eight parent rows where each differs. It
is written whether or not the check passed; a failed check then raises naming
the arrays and rows, so the numbers behind a refusal are on disk.

**Import baselines.** Attach the recorded run's four saved decoders to the
evaluation record as historical baselines, after verifying them (spec section 5.3).

```bash
.venv/bin/python tools/hierarchical_experiment import-baselines --record "$OUT/evaluation" \
    --recorded-run "$RUN"
```

The four baselines are `joint_mwpm_recorded`, `joint_uf`, `joint_correlated_mwpm`,
and `joint_correlated_uf`, read from the run's `<stem>_predictions.npy` files;
`--names` selects a subset. Import runs **before** calibration and replay on that
record: it rewrites `record.npz` with the baseline columns, so the record's artifact
hash changes while its collection identity does not, and a replay made before the
import is refused by the summary stage. Repeating an identical import returns without
writing; a run whose baselines differ, or a different selection of names, is refused
naming the first difference, because a record carries one set of baselines. Only
evaluation records accept baselines; calibration records never carry any. An
import interrupted while publishing is put back by the next `import-baselines`
call. The verification the import performs is described under [Historical
baselines](#historical-baselines).

The attachment has its own integrity digest covering the import provenance and
gate results, with a packed hash for each baseline's mapped prediction array.
`load_record` verifies this binding every time it opens an imported record.
Older M2 attachments must first be upgraded by repeating `import-baselines`
with their original recorded run. This re-verifies the saved predictions and
gates, then updates only the manifest; the record bytes and collection identity
stay unchanged. Replay outputs that name the old manifest need a fresh output
directory.

**Calibrate.** Fit one isotonic calibrator per estimator and sector on the
calibration record, pooling the six patches of a sector into one fit.

```bash
.venv/bin/python tools/hierarchical_experiment calibrate --record "$OUT/calibration" \
    --out "$OUT/calibrators_pilot.json" \
    --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated
```

An estimator is named `reference:score`, with `reference` in `uf`, `mwpm` and
`score` in `cluster_gap`, `gap_plain`, `gap_correlated`. The cluster gap is a UF
quantity and has no meaning for the MWPM reference.

Repeating calibration with identical inputs verifies and reuses the existing
artifact byte for byte, including its creation timestamp. Changed records,
estimators, fitting code, or dependency versions require a new output path.
This keeps an existing replay reusable when the calibration command is repeated.

**Replay.** Replay configurations over the held-out evaluation record.

```bash
.venv/bin/python tools/hierarchical_experiment replay --record "$OUT/evaluation" \
    --calibrators "$OUT/calibrators_pilot.json" --out "$OUT/replay_pilot" \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=all_refined,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=all_refined,outer=mixed
```

A configuration is `initial=<reference:score>,refined=<score>,policy=<name>`
with an optional `outer=<rule>` that defaults to `mixed`. `refined` names a
score rather than an estimator because both estimators of a configuration share
the initial reference. The supported pairs are `uf:cluster_gap` to `gap_plain`,
`uf:cluster_gap` to `gap_correlated`, and `mwpm:gap_plain` to `gap_correlated`;
the policies are the two endpoints `initial_only` and `all_refined`. Selective
policies and random controls belong to a later milestone and are refused by
name rather than silently resolved to an endpoint.

**Summarize.** Report only verified replay outputs.

```bash
.venv/bin/python tools/hierarchical_experiment summarize --replays "$OUT/replay_pilot" \
    --out "$OUT/summary_pilot.md" --replicates 10000 --seed 43
```

`--replicates` and `--seed` are written out in every pilot command rather than
left to a default, so a quoted number can always be reproduced. The report pairs
each estimator pair's cells against that pair's `initial_only` `mixed` cell and
prints, for both pooled misattribution and block failure, the rate with its
denominator, the 95% percentile interval of each side, and the paired
difference with its interval. A statistic no eligible case supports is printed
as `unavailable` beside its counts, never as a zero.

When the record carries imported baselines, its section ends with "Baselines on
the same shots" (spec section 9): the recorded run's imported decoders (`historical`),
the collector's recomputed `joint_mwpm` (`collected`), and every replayed cell
(`hierarchical`), each with its block failure count, the 95% bootstrap interval
of that rate over the same resampled shots, and the normalized LER per patch per
round, followed by each hierarchical cell's paired block-failure difference
against `joint_mwpm_recorded`. The rate interval is drawn under the same seed as
the paired comparisons, so a cell's interval here is the interval its endpoint
table prints. A record whose import left out `joint_mwpm_recorded` gets the
table with every paired difference `unavailable`, never a difference against
another column; a record without baselines gets no table and no `baselines` key
in the JSON. The summary manifest's inputs are the same either way: the record
hash they name already covers the baseline columns.

## What each stage writes

```text
<collect out>/sample/       circuit.stim, model.dem, packed arrays, sample.json
<collect out>/collection.json
<collect out>/checkpoint.npz        only while incomplete
<collect out>/record.npz            plus baseline_<name> arrays after import-baselines
<collect out>/record.npz.before-import   the previous record, only while
                                    import-baselines is publishing
<collect out>/manifest.json         completion marker, hashes, graph/record checks,
                                    and a baselines block after import-baselines
<verify out>.json                   both compared records' identities and hashes,
                                    and the row-for-row check
<calibrators>.json                  knots, verified parents and compatibility
<replay out>/<config>/              prediction/mask/tie/work arrays, results.json
<replay out>/replay_manifest.json    completion marker and all artifact hashes
<summary>.md and <summary>.manifest.json
```

`<summary>.json` is written beside the report with every number the tables
print, so a figure or a follow-up analysis never has to parse markdown.

Every artifact is written through a temporary sibling and one atomic rename, and
each completion manifest is published **last**, over the artifacts it declares
hashes for. That ordering is what gives the manifests their meaning: a
directory holding `record.npz` but no `manifest.json` is an interrupted
publication rather than a result, and a resumed run rebuilds and replaces it.

## Full-call sampling versus subset collection

A sample is one Stim call, and its identity is the call: the model, the seed,
the *full* shot count, and the hash of the packed payload. The pilot decodes the
first 2,000 rows of a 50,000-shot calibration call and of the saved 100,000-shot
evaluation call, and the records it publishes keep the parent identity and the
exact parent row ids of the whole call. Resampling a smaller call instead would
produce a different parent sample and different shots, and treating the subsets
as independent sets would lose the ability to extend them later.

So `--shots` is the size of the sampling call and `--rows` is what gets decoded.
Two records of the same call over different rows can be concatenated later; two
records of different calls cannot.

Two sampling calls that differ only in shot count are different parent samples
but one *sampling family*, because their shot streams may overlap. Replay
refuses calibrators whose record shares the evaluation record's parent sample
**or** its sampling family, so a fit can never quietly be evaluated on the shots
it was fitted on.

Imported and republished samples retain their original model and sampling
package versions. A newer runtime does not change the identity of historical
shots; its decoder versions are recorded separately when collecting them.

## Checkpoints and completion

Collection writes one `checkpoint.npz` holding every buffer and the completion
mask, replaced atomically about once a minute and whenever scheduling stops.
There is deliberately no second progress file: two files can disagree about
which rows are done.

`--max-chunks` schedules at most that many chunks and then returns, which is how
a long collection is exercised in pieces. Running the same command again
continues from the checkpoint; the checkpoint is removed only after the
completion manifest lands.

```bash
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/evaluation" --role evaluation \
    --recorded-run "$RUN" --rows 0:2000 --workers 16 --chunk-size 25 --max-chunks 4
```

Worker and chunk counts change no decoded value and enter no identity, so they
may differ between resumptions. The rows, the role, the model, the parent
sample, and the decoder may not: a request that changes any of them is refused
by name rather than merged into the directory. Use a new directory for a
different experiment; do not relabel an existing sample.

The decoder identity covers the per-row L1 path -- the patch split, the two
decoders, the correlation rules, the record layout, and the `_l1.py` evaluation
that fills each column -- and nothing else. Sampling, the two gates, and the
coordinator in `_collect.py` carry a separate *check* identity, which is why
changed validation code rechecks stored arrays in place and republishes the
manifest instead of recollecting the rows. A change to the L1 path is the case
that cannot be absorbed: the existing directory holds numbers a different
decoder produced, so it is refused and the collection starts fresh.

A failed gate publishes nothing. The graph gate runs before any row is decoded
and stops the collection outright; the record gate runs before publication,
leaves `failed_checks.json` and the checkpoint in place, and publishes no
manifest, so the failure can be investigated with the collected rows in hand.

## The verified saved model

The model is the saved `circuit.stim` and `model.dem` bytes, not the generator
that produced them. `SampleSet.load` re-verifies all four sample artifacts and
the packed payload on every call, and recomputes the declared identities from
that content rather than trusting them.

A second `collect` into an existing directory therefore checks the *request*
against the verified sample before anything is collected. For a generated
request the parameters, seed, and full shot count must match, and today's
generator must still build a model with the same identity: the circuit and DEM
are rebuilt in memory, hashed, and compared, never saved. For an imported
request the recorded run's circuit, model, and payload hashes must equal the
saved sample's. A changed generator cannot quietly hand old shots a new model,
and a path is never an identity.

## Records are read-only

`L1Record` validates every value before storing it as an owned, read-only array:
bits are 0 or 1 rather than anything that casts to `True`, gaps and weights are
finite and nonnegative, and row ids are whole, unique, and increasing. Neither
the caller's input arrays nor a later alias can change a stored record.

`load_record` is the only way to read a completed collection, and it verifies
before it returns: the manifest's schema and `status`, that the recorded checks
passed, that every declared artifact still hashes to its published value, that
the stored row ids match the manifest's summary exactly, and that the recorded
collection identity follows from the manifest's own fields.

The manifest includes the canonical sample inputs used to recompute its model,
parent-sample, and sampling-family identities. The loader also checks the
decoder identity against the recorded source hashes and versions, and checks
that the experiment parameters agree with the sample provenance.

Calibration and replay verify further. `load_calibrators` rebuilds every knot
array through `IsotonicCalibrator.from_json`, which rejects centers that are not
strictly increasing, probabilities outside `[1e-6, 1 - 1e-6]`, and probabilities
that are not monotone in the declared direction; it checks the declared knot
convention and clipping constant against the ones this code implements, and
recomputes the calibration identity from the artifact's own fields. A separate
`payload_sha256` covers the fitted values and complete source-record declaration,
including the parent and sampling-family identities used for holdout checks.
The loader reopens that source record and checks its role, rows, identities, and
artifact and manifest hashes, so the calibration record must remain accessible
at the path recorded in the calibrator artifact. Replay fingerprints the current
calibration application code as well as L2, policies, replay, and metrics.

The summary stage re-verifies the replay manifest, its identity, every array and JSON hash,
and that the record it names is still the record that was replayed: the same
rows, identities, role, record hash, and manifest hash. A record replaced after
replay is rejected rather than quietly summarized.

Records written before canonical sample provenance was required, and
calibrators without `payload_sha256`, are rejected explicitly. Generate new
collection, calibration, and replay outputs in a fresh run directory using the
saved full-call samples; do not add checksums or identity inputs to old outputs
by hand. The saved sample format remains supported.

## Historical baselines

The evaluation set is the recorded four-decoder run's sample, so that run's saved
predictions can be compared with the hierarchy on exactly the same shots (spec
section 9). They enter the record only through `import-baselines`, and only after
the run and the record have been verified against each other:

- The run's `circuit.stim`, `model.dem`, and packed payload are re-hashed against
  its own manifest, exactly as `collect --recorded-run` imports them, and the
  manifest must carry the run's recorded implementation provenance (`versions`,
  `source_sha256`, `code_commit`), which the record manifest keeps unchanged.
- Each prediction file is loaded without pickles, must hold binary values of shape
  `(shots, 12)`, and must reproduce the `prediction_packed_sha256` that the run's
  `results.json` declares: the SHA-256 of the little-endian bit-packed rows.
- The record must carry the evaluation role and the same payload, circuit, and
  model hashes as the run. Baseline rows are mapped by the record's parent row ids.
- Every baseline's per-sector parity must equal the sampled yoke bit on every row,
  the rule `check_record` applies to the collector's own joint decode.
- The recorded joint MWPM is compared with the record's recomputed `joint_mwpm`
  column under the collector's tie rule: every disagreeing row must have equal
  total forced cost within `COST_TOLERANCE`, so two decodes of one model differ
  only where two optimal matchings exist. The agreement fraction, the disagreement
  count, and the tie-explained count are recorded in the manifest.

Every failure raises naming the file, field, baseline, or parent rows, and nothing
is written. On success the republished manifest is assembled in full first, the new
`record.npz` is written and hashed beside the record, and only then does the record
change, by renames alone: the previous `record.npz` steps aside as
`record.npz.before-import`, the new one takes its place, and `manifest.json` is
republished last over it, after which the kept record is dropped. In the manifest
`artifacts.record.npz` takes the new hash, a `baselines` block records the run's
directory, manifest and results hashes, per-baseline prediction hashes, the run's
provenance, the importer's check identity, and the gate results, and every other
field is unchanged byte for byte. An import interrupted between the renames leaves a
record the manifest does not describe, which `load_record` (and so `calibrate` and
`replay`) refuses; run `import-baselines` again: it puts the kept record back, so the
directory reads as it did before the interrupted import, and then imports normally.
An import interrupted after the manifest leaves only the kept record behind, which the
next call drops. Do not edit either file by hand; a kept record the manifest describes
neither of is refused.

```python
from yoked.hierarchical import load_record, load_recorded_baselines, attach_baselines

recorded = load_recorded_baselines(os.environ['RUN'])          # verifies the run
loaded = load_record(f'{out}/evaluation')
attached, provenance = attach_baselines(loaded, recorded)     # gates, no writing
print(sorted(attached.baselines), provenance['joint_mwpm']['agreement'])
```

Once imported, the baselines are reported only by `summarize`, in the
"Baselines on the same shots" table described under that stage, and only for the
evaluation record they were attached to: historical results are never presented
as confirmation measurements.

## Two kinds of work count

The reports carry two work tables, and they measure different things.

**Actual collection work** (`manifest.json`, `collection_work`) is what really
ran while the record was produced: UF decodes, parity-augmented searches,
unforced and forced matchings, reweighting passes, and wall time, with the
number of rows, workers, and resumptions. It counts *every* score for *every*
patch, because collection computes them all once so that any configuration can
be replayed offline. Work attempted on rows lost to an interruption before the
next checkpoint is redone on the restart and is not counted.

**Replay work** (`results.json`, `work`) is what the specified replay procedure
*would* need: the fixed initial work of the estimator pair charged once per
patch, plus the incremental work charged once per distinct refined patch, so
that requesting both sectors of one patch costs one refinement rather than two.

Neither is elapsed time, and no latency or speedup claim follows from either.
The replay counts are a procedure's call counts; the collection counts describe
an offline data-production run that deliberately computes more than any single
configuration needs.

## The serial debug path

`--workers 1` runs every chunk in the calling process with no pool at all, so a
failing chunk raises where it happened and a debugger or a `pytest --pdb` sees
the real stack. It is the same code path a parallel run uses -- the workers call
the same `collect_rows` -- so a bug reproduced serially is the bug.

```bash
.venv/bin/python tools/hierarchical_experiment collect --out "$TMPDIR/debug" --role evaluation \
    --recorded-run "$RUN" --rows 0:4 --workers 1 --chunk-size 2
```

## Library examples

**Split the six-patch model.** The splitter derives the per-patch graphs from
the same DEM the joint decoders use, so the local graphs hold literally the
edges and weights the joint decoder would have seen.

```python
import gen
import numpy as np
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.hierarchical import PatchGraphs

circuit = yoked_magic_memory_circuit(
    patch_diameter=3, rounds=12, noise=gen.NoiseModel.si1000(0.003),
    style='cz', yokes=2, num_patches=6,
)
dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
patches = PatchGraphs.from_yoked_dem(dem, num_patches=6)
assert len(patches) == 6 and patches.num_observables == 12

detectors, actual = circuit.compile_detector_sampler(seed=42).sample(
    shots=4, separate_observables=True,
)
local = patches.local_syndromes(detectors)              # (4, 6, local detectors)
```

**Reference bits and the cluster gap.** `decode_with_gaps` returns a named
`ClusterGapResult`, not a tuple.

```python
from yoked.hierarchical import ClusterGapUnionFindDecoder

uf = ClusterGapUnionFindDecoder(patches[0].graph)
cluster = uf.decode_with_gaps(local[0, 0])
assert cluster.prediction.shape == (2,)                 # r[0, X] and r[0, Z]
assert cluster.cluster_gap.shape == (2,) and (cluster.cluster_gap >= 0).all()
assert cluster.dijkstra_states.sum() > 0                # the soft-output work proxy
```

**Forced class weights and signed gaps.** `forced_weights` returns a
`ForcedWeights` whose `plain` and `correlated` arrays are indexed `[c_X, c_Z]`
in nats; `signed_gaps` reads them against a chosen reference's bits.

```python
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical import MatchingGaps, signed_gaps

rules = correlation_rules_from_dem(patches[0].graph, patches[0].local_dem)
matching = MatchingGaps(patches[0], rules)
forced = matching.forced_weights(local[0, 0])
assert forced.plain.shape == (2, 2)
delta = signed_gaps(forced.plain, cluster.prediction)   # (2,) nats: delta_X, delta_Z
```

A negative gap means the matcher prefers the complement of the reference bit.

**MWPM L2 for one sector.** `mwpm_outer_map_batch` returns a
`BatchOuterDecision` with `patterns` of shape `(shots, patches)` and `tied` of
shape `(shots,)`; both are owned, read-only arrays.

```python
from yoked.hierarchical import BatchOuterDecision, mwpm_outer_map_batch, frame_adjusted_syndrome

yoke = np.array([[True, False]])                        # y_X fired, y_Z did not
reference = np.zeros((1, 12), dtype=bool)
sigma = frame_adjusted_syndrome(yoke, reference)        # (1, 2)
assert sigma.tolist() == [[True, False]]

q = np.array([[0.30, 0.02, 0.02, 0.02, 0.02, 0.02]])
decision = mwpm_outer_map_batch(q, sigma[:, 0])
assert isinstance(decision, BatchOuterDecision)
assert decision.patterns.tolist() == [[True, False, False, False, False, False]]
assert decision.tied.tolist() == [False]
assert not decision.patterns.flags.writeable
```

Odd parity with all probabilities below one half flips exactly the most likely
patch. With `sigma = 0` the answer is usually no flip, but multiple flips are
allowed and do occur once a probability exceeds one half. Log-weight ties within
`TIE_TOLERANCE` break to the lowest binary pattern and are reported in `tied`.

The matcher uses a separate two-edge path for each patch, weighted by its
absolute log odds after preflipping probabilities above one half. This keeps
patch labels distinct and supports multiple residual flips. Original-weight
checks repair numerical quantization, and polynomial tie resolution preserves
the documented choice among tied patterns. The exhaustive `exact_outer_map`
and `exact_outer_map_batch` functions remain available as validation oracles;
production replay calls MWPM and does not enumerate patterns.
Fresh collection still uses enumeration in its independent small-block
correctness gate; that validation does not produce the replay predictions.

This backend change needs only a new replay output directory. Reuse the saved
L1 record and calibrator artifact directly; no new shots or calibration fit
are required. Replay manifests include the MWPM source hash, PyMatching
version, and tie rule. Existing operation counters describe L1 work and remain
unchanged; they are not a count of L2 matcher calls or a latency measurement.

**Calibrate a score.** The direction is fixed by definition: every score is
decreasing in the residual-error probability.

```python
from yoked.hierarchical import IsotonicCalibrator

scores = np.array([0.1, 0.4, 0.4, 1.2, 2.0])
outcomes = np.array([1, 1, 0, 0, 0])
calibrator = IsotonicCalibrator.fit(scores, outcomes, direction='decreasing')
probabilities = calibrator.probability(scores)
assert (np.diff(probabilities) <= 0).all()
restored = IsotonicCalibrator.from_json(calibrator.to_json())   # revalidates the knots
np.testing.assert_allclose(restored.probabilities, calibrator.probabilities)
```

**Replay a published record from Python.** The stage functions are the same
objects the CLI calls, so a run can be reproduced or extended in a notebook.

```python
import os
from yoked.hierarchical import load_calibrators, load_record, parse_config, replay, summarize_result

out = os.environ['OUT']
loaded = load_record(f'{out}/evaluation')               # verifies before returning
calibrators, artifact = load_calibrators(f'{out}/calibrators_pilot.json')
config = parse_config('initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined')
result = replay(loaded.record, calibrators, config)
summary = summarize_result(loaded.record, result, pieces=6 * 36)
print(summary['misattribution']['pooled'], result.work.totals())
```

## Run the tests

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$TMPDIR/hier-pycache"
.venv/bin/python -m pytest -q -p no:cacheprovider src/yoked/hierarchical
```

The suite runs on a distance-3 six-patch circuit with a few hundred shots, so it
finishes in well under a minute. It covers the hub split and its graph equivalence
against the joint model, the cluster gap against brute-force odd walks, the
forced weights and their additivity, the per-row L1 evaluation with every stored
column pinned to an independent recomputation from the patch's own graph, the
PAV fit, the exact outer decoder, the
endpoint policies, replay and its work accounting, the metrics, their paired
bootstrap, one rate's interval, and the paired block failure of two prediction arrays,
the baseline import against a fake recorded run with each way a run or a record can
fail its gates, the subset check on synthetic records and on a full and a subset
collection of one sampling call, the baselines table of a summary over imported fake
baselines with every rate pinned to a direct computation from the arrays, and the
stage boundaries: a full pipeline through the functions and once through the command
line, valid resume, and each way a changed request, altered artifact, replaced record,
or interrupted publication is refused.
