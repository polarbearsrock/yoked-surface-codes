# Hierarchical L1/L2 decoding of the 1D yoked surface code

Run from the repository root with `PYTHONPATH=src`. The `yoked.hierarchical`
package uses the repository's existing dependencies (Stim, PyMatching, NumPy,
SciPy, sinter) and requires no native build. Every run directory below lives
under `$TMPDIR`; only reports are committed.

The design this implements is
[the hierarchical L1/L2 spec](superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md).

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
`sigma[s] = y[s] XOR parity(r[:, s])`, and returns the maximum-weight residual
pattern `x` of the required parity. The final prediction is `f = r XOR x`.

The reference is *fixed*: it is decided once by L1 and stored, and every
configuration, policy, and report is measured against the same reference bits.
That is what makes the primary metric well posed. Among sectors where the
reference made exactly one mistake, the misattribution rate is the fraction
where the final prediction is still wrong, and two configurations may only be
compared when they share a reference, because different references define
different eligible populations.

## The four stages

`tools/hierarchical_experiment` is a thin argparse layer: it parses arguments,
calls one library function, and prints a path. Every check lives in
`yoked.hierarchical`, so any failure below can be reproduced from Python.

**Collect.** Sample or import one full Stim call, then decode the requested
parent rows into a record. The calibration set is generated; the evaluation set
is imported from the recorded four-decoder run so that it decodes the original
circuit and model rather than a regenerated one.

```bash
cd /data2/s2chitni/yoked-surface-codes
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

## What each stage writes

```text
<collect out>/sample/       circuit.stim, model.dem, packed arrays, sample.json
<collect out>/collection.json
<collect out>/checkpoint.npz        only while incomplete
<collect out>/record.npz
<collect out>/manifest.json         completion marker, hashes, graph/record checks
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

Calibration and replay verify further. `load_calibrators` rebuilds every knot
array through `IsotonicCalibrator.from_json`, which rejects centers that are not
strictly increasing, probabilities outside `[1e-6, 1 - 1e-6]`, and probabilities
that are not monotone in the declared direction; it checks the declared knot
convention and clipping constant against the ones this code implements, and
recomputes the calibration identity from the artifact's own fields. The summary
stage re-verifies the replay manifest, its identity, every array and JSON hash,
and that the record it names is still the record that was replayed: the same
rows, identities, role, record hash, and manifest hash. A record replaced after
replay is rejected rather than quietly summarized.

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

**Exact L2 for one sector.** `exact_outer_map_batch` returns a
`BatchOuterDecision` with `patterns` of shape `(shots, patches)` and `tied` of
shape `(shots,)`; both are owned, read-only arrays.

```python
from yoked.hierarchical import BatchOuterDecision, exact_outer_map_batch, frame_adjusted_syndrome

yoke = np.array([[True, False]])                        # y_X fired, y_Z did not
reference = np.zeros((1, 12), dtype=bool)
sigma = frame_adjusted_syndrome(yoke, reference)        # (1, 2)
assert sigma.tolist() == [[True, False]]

q = np.array([[0.30, 0.02, 0.02, 0.02, 0.02, 0.02]])
decision = exact_outer_map_batch(q, sigma[:, 0])
assert isinstance(decision, BatchOuterDecision)
assert decision.patterns.tolist() == [[True, False, False, False, False, False]]
assert decision.tied.tolist() == [False]
assert not decision.patterns.flags.writeable
```

Odd parity with all probabilities below one half flips exactly the most likely
patch. With `sigma = 0` the answer is usually no flip, but multiple flips are
allowed and do occur once a probability exceeds one half. Log-weight ties within
`TIE_TOLERANCE` break to the lowest binary pattern and are reported in `tied`.

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
forced weights and their additivity, the PAV fit, the exact outer decoder, the
endpoint policies, replay and its work accounting, the metrics and their paired
bootstrap, and the stage boundaries: a full pipeline through the functions and
once through the command line, valid resume, and each way a changed request,
altered artifact, replaced record, or interrupted publication is refused.
