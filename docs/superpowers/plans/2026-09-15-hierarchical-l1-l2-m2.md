# Hierarchical L1/L2 Decoding, M2: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the full-set M2 result: collect all 100,000 evaluation rows and all 50,000 calibration rows under the current provenance contract, attach the recorded run's four historical decoders to the evaluation record as verified baselines, refit calibrators, replay the two endpoints for every estimator pair, and publish a report that decides whether M3 (selective refinement) starts.

**Architecture:** Three small additions to the existing package, then one run. A new `_baselines.py` imports historical predictions into a completed record and republishes its manifest with the run's provenance. A new `_reproduction.py` compares a full record with a subset record row for row. The summary stage gains a baselines table. Everything else reuses the M1 stages unchanged. Collections go into a fresh run directory from the saved full-call samples, as `docs/hierarchical_decoding.md` now requires for outputs written before commit `2c9d81f`.

**Tech Stack:** as M1 (Python 3.14 in `.venv`, NumPy, Stim, PyMatching, sinter, pytest; `PYTHONPATH=src` from the repository root).

**Spec:** `docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md`, sections 3 (data plan), 5.3 (baselines in the record and their verification), 9 ("Connection to the UF experiments"), and 12 (milestone M2). The M1 plan is `docs/superpowers/plans/2026-09-14-hierarchical-l1-l2-m1-pilot.md`; its Global Constraints apply unchanged.

## Global Constraints

- All M1 Global Constraints apply: notation, nats, read-only owned arrays, immutable mappings, validate before casting, named constants with reason comments, small functions and concrete records, module docstrings naming the object and the invariants tested, tests beside modules, `$TMPDIR` for run data, commit after each task with a description of the actual change.
- Provenance contract at HEAD (`2c9d81f`): `load_record` is the only completed-record loader and verifies `sample_identity_inputs`, decoder and collection identities, and artifact hashes; calibrator artifacts carry `payload_sha256`; records or calibrators written before that contract are rejected and must be regenerated in a fresh run directory from the saved samples, never patched by hand.
- Baselines never change a collection identity (the identity covers parent sample, decoder, role, and rows). Importing baselines changes `record.npz` and therefore the record artifact hash, so import runs before calibration and replay on that record; a replay manifest made earlier is correctly rejected by the summary stage.
- New verification modules (`_baselines.py`, `_reproduction.py`) join `CHECK_SOURCES`, not `DECODER_SOURCES`: they gate data and never produce L1 output.
- Confirmation stays untouched; no selective policies; no `eta`.
- Commit messages: one-line subject describing the change, a blank line, then `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM`.

## File Structure

| File | Responsibility |
|---|---|
| `src/yoked/hierarchical/_baselines.py` (+ test) | Load and verify a recorded run's saved decoder predictions; attach them to a completed evaluation record; provenance block for the manifest. |
| `src/yoked/hierarchical/_reproduction.py` (+ test) | Compare a full record against a subset record over the subset's rows, array by array. |
| `src/yoked/hierarchical/_metrics.py` (+ test) | `bootstrap_rate` for one configuration's block-failure interval; `paired_block_failure` for two prediction arrays. |
| `src/yoked/hierarchical/_stages.py` (+ test) | `stage_import_baselines`, `stage_verify_subset`, and the baselines table in `stage_summarize`. |
| `src/yoked/hierarchical/_provenance.py` | The two new modules added to `CHECK_SOURCES`. |
| `tools/hierarchical_experiment` | Subcommands `import-baselines` and `verify-subset`. |
| `docs/hierarchical_decoding.md` | The two new stages, the ordering rule, the baselines table. |
| `docs/results/hierarchical_m2_d9_p003.md` (+ JSON copies) | The M2 report and its machine-readable inputs. |

---

### Task 1: Historical baselines on the evaluation record

**Files:** create `src/yoked/hierarchical/_baselines.py` and `_baselines_test.py`; modify `_stages.py`, `_stages_test.py`, `_provenance.py`, `__init__.py`, `tools/hierarchical_experiment`, `docs/hierarchical_decoding.md`.

**Interfaces.**

- `BASELINE_DECODERS`: an immutable mapping from baseline name to the recorded decoder's file stem, exactly `joint_mwpm_recorded -> mwpm`, `joint_uf -> uf`, `joint_correlated_mwpm -> correlated_mwpm`, `joint_correlated_uf -> correlated_uf` (prediction files `<stem>_predictions.npy`, `results.json` keys `decoders.<stem>.prediction_packed_sha256`).
- `RecordedBaselines` frozen dataclass: `directory`, `manifest_sha256`, `results_sha256`, `circuit_sha256`, `dem_sha256`, `payload_sha256`, `parent_shots`, `versions`, `source_sha256`, `code_commit` (the run's own manifest values, immutable mappings), `predictions` (name -> owned read-only `(parent_shots, 2P)` bool array), `prediction_sha256` (name -> packed hash).
- `load_recorded_baselines(run_dir, names=BASELINE_DECODERS.keys()) -> RecordedBaselines`: reads the run's `manifest.json` and `results.json`; for each name loads the prediction file with `allow_pickle=False`, requires shape `(parameters.shots, 2 * parameters.patches)` and binary values before casting, recomputes the little-endian packed SHA-256 and requires equality with `results.json`; hashes `circuit.stim`, `model.dem`, and the packed payload; every failure is a `ValueError` naming the file or field.
- `attach_baselines(loaded: LoadedRecord, recorded: RecordedBaselines) -> tuple[L1Record, dict]`: requires role `evaluation`; requires `loaded.manifest['parent_payload_sha256'] == recorded.payload_sha256` and the manifest's `sample_identity_inputs` circuit and DEM hashes to equal the run's; maps rows as `recorded.predictions[name][loaded.record.rows]`; requires every baseline's sector parity to equal `record.yoke` on every row; gates the recorded joint MWPM against `record.joint_mwpm` with the collector's tie rule (every disagreeing row must have equal total forced cost within `COST_TOLERANCE`; reuse or promote `_collect._total_forced_cost`); returns `dataclasses.replace(record, baselines=...)` and a provenance block with the run directory, manifest and results hashes, per-name prediction hashes, the run's source hashes, versions, and commit, the joint-MWPM agreement fraction, disagreement count, and tie-explained count.
- `stage_import_baselines(record_dir, recorded_run, *, names=None) -> LoadedRecord` in `_stages.py`: `load_record`; if the record already carries baselines, require the same names with identical arrays and return (idempotent) or raise naming the first differing baseline; otherwise attach, write `record.npz` atomically through the package's private array container helpers with baselines under the `baseline_` prefix, and republish `manifest.json` last with the updated `artifacts` hash for `record.npz`, a `baselines` block (`imported_utc`, the provenance block above), and every other field byte-for-byte unchanged; return `load_record(record_dir)`.
- CLI: `import-baselines --record DIR --recorded-run DIR [--names NAME ...]`.

**Tests** (distance-3, a few dozen shots, `workers=1`): extend the fake recorded run helper in `_collect_test.py` (or a sibling helper in `_baselines_test.py`) to write the four prediction files and a `results.json` with their packed hashes; derive `mwpm` from the collector's own joint matcher on the fixture, and the other three from it by flipping both sectors' bits of one patch on a few rows (parity preserved). Cover: round trip (`load_record` after import exposes the four baselines with the mapped rows), idempotent second import, a tampered `results.json` hash rejected, a prediction file with the wrong shape or non-binary values rejected, a payload mismatch (another seed) rejected, a circuit or DEM hash mismatch rejected, a parity-violating baseline rejected, a joint-MWPM disagreement that is not a cost tie rejected, import onto a calibration-role record rejected, a differing second import rejected, and calibrators fit before the import still loading and replaying after it (calibration records carry no baselines). The stage test runs `stage_import_baselines` and the CLI once.

---

### Task 2: Subset reproduction check

**Files:** create `src/yoked/hierarchical/_reproduction.py` and `_reproduction_test.py`; modify `_stages.py`, `_stages_test.py`, `_provenance.py`, `__init__.py`, `tools/hierarchical_experiment`, `docs/hierarchical_decoding.md`.

**Interfaces.**

- `SubsetCheck` frozen dataclass: `subset_rows` (count), `matched_rows`, per-array equality (name -> bool), `mismatched_rows` (name -> bounded tuple of parent row ids), `passed`, `to_json()`, `raise_if_failed()`.
- `subset_reproduction(full: L1Record, subset: L1Record) -> SubsetCheck`: requires every subset row to exist in the full record (positions by `searchsorted` with an equality check), then compares every `ARRAY_FIELDS` array at those positions with exact equality (bit-for-bit for floats); baselines are not compared (the subset need not carry them).
- `stage_verify_subset(full_dir, subset_dir, out_path) -> SubsetCheck` in `_stages.py`: `load_record` both; require equal `model`, `parent_sample`, `sampling_family`, `decoder` identities and role, naming the first that differs; run the check; write `out_path` JSON atomically with both records' directories, identities, `record.npz` hashes, row summaries, and the check; raise if failed.
- CLI: `verify-subset --full DIR --subset DIR --out PATH`.

**Tests:** synthetic records (a full record and a subset built with `subset(positions)`) pass; one changed value in the subset is reported under its array name with the parent row id; a subset row absent from the full record raises; on the distance-3 fixture, a 48-row collection and a 10-row collection of the same sample pass through `stage_verify_subset`; a subset from another seed is rejected on identity; the CLI once.

---

### Task 3: Baselines in the summary

**Files:** modify `_metrics.py`, `_metrics_test.py`, `_stages.py`, `_stages_test.py`, `docs/hierarchical_decoding.md`.

**Interfaces.**

- `bootstrap_rate(indicators, *, replicates, seed) -> RateInterval` (frozen: `count`, `total`, `estimate`, `low`, `high`, `replicates`, `seed`, `to_json()`), resampling whole shots with the same block-wise resampler `paired_bootstrap` uses; undefined values as None.
- `paired_block_failure(record, predictions_a, predictions_b, *, replicates, seed) -> PairedDifference`: block-failure paired difference `b - a` for any two `(shots, 2P)` prediction arrays over the record's `actual`.
- `stage_summarize`: when a record carries baselines, the section gains `baselines`: for each baseline name and for the collector's own `joint_mwpm`, block failures (count, total), `bootstrap_rate` of block failure, and normalized LER; for each hierarchical cell in the replay, the same rate plus `paired_block_failure` against `joint_mwpm_recorded`; rendered as a "Baselines on the same shots" table after the endpoint tables and stored in the summary JSON. Records without baselines render no table. The summary manifest's inputs are unchanged (the record hash already covers baselines).

**Tests:** hand-built indicator fixtures for `bootstrap_rate` (a known rate, reproducibility with the seed, an empty input reported as unavailable) and for `paired_block_failure` (hand-computed difference); the distance-3 pipeline with imported fake baselines renders the table with the four names plus `joint_mwpm`, and every rate in it equals a direct computation from the arrays; a pipeline without baselines renders no table.

---

### Task 4: The M2 run and report

**Files:** create `docs/results/hierarchical_m2_d9_p003.md` and, beside it under `docs/results/hierarchical_m2_d9_p003/`, the summary JSON, the two subset-verification JSONs, and the baseline audit block copied from the evaluation manifest.

**Run** (all under `OUT2="$TMPDIR/hier-d9-p003-m2"`, `RUN=/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha`, commands as `PYTHONPATH=src .venv/bin/python tools/hierarchical_experiment ...`, long steps under `nohup` with bounded polling of `manifest.json`):

1. Copy the pilot's saved calibration sample directory (`$TMPDIR/hier-d9-p003/calibration/sample`) to `$OUT2/calibration/sample` and to `$OUT2/calibration_pilot_rows/sample`. Collect `$OUT2/evaluation` (role evaluation, `--recorded-run $RUN`, all rows, `--workers 32 --chunk-size 50`) and `$OUT2/calibration` (role calibration, `--distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000`, all rows, same workers). Then collect the two 2,000-row subsets `$OUT2/evaluation_pilot_rows` (`--rows 0:2000` from the recorded run) and `$OUT2/calibration_pilot_rows` (`--rows 0:2000`).
2. `verify-subset` full versus subset for both roles; record the JSONs. Confirm that each subset's `record.npz` SHA-256 equals the hash the pilot report published for that role (the pilot's `record.npz` hashes are listed in `docs/results/hierarchical_pilot_d9_p003.md`); this is the byte-exact reproduction of the pilot under the current provenance contract.
3. `import-baselines --record $OUT2/evaluation --recorded-run $RUN`.
4. `calibrate --record $OUT2/calibration --out $OUT2/calibrators_full.json --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated`.
5. `replay --record $OUT2/evaluation --calibrators $OUT2/calibrators_full.json --out $OUT2/replay_full` with the six pilot configurations.
6. `summarize --replays $OUT2/replay_full --out $OUT2/summary_full.md --replicates 10000 --seed 43`.

**Report** (style of the pilot report; every number copied from the summary JSON, the manifests, and the verification JSONs; labeled as the evaluation set, exploratory, confirmation untouched): configuration and provenance (identities, hashes, commits, sample inputs); graph and record checks for both full records; the subset reproduction results and the hash equality with the pilot; endpoint tables for all three estimator pairs with rate intervals and paired differences for pooled misattribution and block failure; strata; replay work; actual collection work, wall times, and throughput with the arithmetic shown; the baselines table on the same 100,000 shots (four recorded decoders, the recomputed joint MWPM, and every hierarchical cell) with the paired block-failure differences against the recorded joint MWPM; the M3 gate: proceed if the primary pair's all-refined minus initial-only pooled-misattribution difference is negative with a 95% interval excluding zero, else report the effect and stop; reproduction commands and artifact paths.

**Commit:** the report and the JSON copies only; run data stays under `$TMPDIR`.

---

## Plan consistency checklist

- [ ] Task 1 satisfies spec 5.3's baseline verification (sample identity, prediction hashes, row mapping, provenance) and spec 9's baseline comparison inputs.
- [ ] Task 2 gives M2 the reproduction guarantee the fresh-collection decision relies on.
- [ ] Task 3 reports baselines only in the evaluation column, as spec 9 requires.
- [ ] Task 4 follows the current provenance contract: fresh run directory, saved samples, no hand-patched outputs, import before calibrate and replay.
