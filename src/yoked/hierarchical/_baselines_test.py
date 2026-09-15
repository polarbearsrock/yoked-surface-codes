"""Tests for the historical baselines of a recorded run (_baselines.py) and the import stage.

Checks: the four baseline names map to exactly the recorded decoder stems and the
module is a check source rather than a decoder source; the packed prediction hash
reproduces the recorded runs' little-endian convention on a hand-packed example; a fake
recorded run built from the distance-3 fixture loads with every prediction hash
recomputed and compared, while a tampered ``results.json`` hash, a prediction file with
the wrong shape or non-binary values, a missing file, an unknown or repeated name, and
a run without recorded implementation provenance are each rejected naming the file or
field; the import round trip, where ``load_record`` exposes the four baselines with the
parent rows mapped, the record hash changes, and every other manifest field is unchanged
byte for byte; a second import returning without writing, and a differing second import
or one naming other baselines refused naming the first difference; a payload mismatch
from another seed, a circuit or DEM hash mismatch, a parity-violating baseline, a
joint-MWPM disagreement that is not a cost tie, and an import onto a calibration-role
record refused, while a disagreement explained by a cost tie is accepted and counted;
calibrators fitted before the import loading and replaying after it, with the
calibration record carrying no baselines; a replay made before the import rejected by
the summary; and an interrupted import leaving a directory ``load_record`` refuses
rather than a half-imported record.

The fake run's ``mwpm`` predictions come from the collector's own joint matcher, so the
tie gate sees zero disagreements unless a test introduces one; the other three decoders
flip both sectors of two patches on a few rows, which keeps every sector parity.
"""
from __future__ import annotations

import dataclasses
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from yoked.hierarchical._baselines import (
    BASELINE_DECODERS, BASELINES_FIELD, RECORDED_JOINT_MWPM, RECORDED_RESULTS,
    RecordedBaselines, attach_baselines, load_recorded_baselines, packed_prediction_sha256,
    prediction_file,
)
from yoked.hierarchical._collect import (
    CIRCUIT_FILE, COST_TOLERANCE, DEM_FILE, RECORDED_MANIFEST, SAMPLE_MANIFEST,
    CircuitParameters, SampleSet,
)
from yoked.hierarchical._l1 import L1Context
from yoked.hierarchical._metrics import DEFAULT_SEED
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import (
    CHECK_SOURCES, DECODER_SOURCES, MODEL_PACKAGES, REPOSITORY_ROOT, SAMPLING_PACKAGES,
    canonical_json, package_versions, read_json, sha256_file, write_json_atomic,
)
from yoked.hierarchical._record import RECORD_FILE, RECORD_MANIFEST, LoadedRecord, load_record
from yoked.hierarchical._stages import (
    CollectRequest, _ImportHooks, load_calibrators, stage_calibrate, stage_collect,
    stage_import_baselines, stage_replay, stage_summarize,
)

DISTANCE, ROUNDS, P = 3, 12, 0.005
"""The stage tests' distance-3 circuit: a few dozen shots hold residual reference
failures and decode in about a second."""

SHOTS = 48
"""Rows in each full sampling call; every one of them is decoded and imported."""

CHUNK = 16
"""Rows per scheduled chunk."""

CALIBRATION_SEED, EVALUATION_SEED, OTHER_SEED = 1, 2, 3
"""The calibration and evaluation calls, and a third call whose payload no record has."""

PARAMETERS = CircuitParameters(distance=DISTANCE, rounds=ROUNDS, p=P)

ESTIMATORS = ('uf:cluster_gap', 'uf:gap_correlated')
CONFIGS = ('initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed',
           'initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined')
"""One estimator pair and its two endpoints, enough to replay before and after an import."""

REPLICATES = 50
"""Bootstrap replicates for the one summary a test runs."""

FLIPS = MappingProxyType({
    'uf': (np.array([1, 5, 9]), (0, 3)),
    'correlated_mwpm': (np.array([2, 6]), (1, 4)),
    'correlated_uf': (np.array([3, 7, 11]), (2, 5)),
})
"""How each non-MWPM fake decoder differs from the joint matcher: the parent rows and
the two patches whose both sectors are flipped there. A sector's parity is the XOR over
patches, so flipping one patch would break it and flipping two keeps it."""

FAKE_SOURCE = 'src/yoked/decoders/_union_find.py'
"""One real file the fake run hashes as its recorded implementation provenance."""

FAKE_COMMIT = 'a' * 40
"""A commit hash that is visibly the run's own value, not the importing tree's."""


# --- the fake recorded run ---------------------------------------------------

def recorded_hash(prediction: np.ndarray) -> str:
    """The recorded runs' ``prediction_packed_sha256``, written out independently here:
    SHA-256 of the little-endian bit-packed rows of the boolean prediction array."""
    packed = np.packbits(np.asarray(prediction, dtype=bool), axis=1, bitorder='little')
    return hashlib.sha256(packed.tobytes()).hexdigest()


def flipped(prediction: np.ndarray, rows, patches) -> np.ndarray:
    """A copy with both sectors of every named patch flipped on the named rows."""
    changed = np.array(prediction, dtype=bool)
    for patch in patches:
        columns = [NUM_SECTORS * patch + sector for sector in range(NUM_SECTORS)]
        changed[np.ix_(np.asarray(rows), columns)] ^= True
    return changed


def fake_predictions(sample: SampleSet, context: L1Context) -> dict[str, np.ndarray]:
    """The four decoders' predictions for every row of ``sample``, by file stem."""
    detectors, _ = sample.rows(np.arange(sample.shots))
    mwpm = np.asarray(context.joint.decode_batch(detectors.astype(np.uint8))).astype(bool)
    predictions = {'mwpm': mwpm}
    for stem, (rows, patches) in FLIPS.items():
        predictions[stem] = flipped(mwpm, rows, patches)
    return predictions


def write_predictions(directory: Path, predictions) -> None:
    """Write ``<stem>_predictions.npy`` for every decoder and the ``results.json`` naming
    their packed hashes, in the recorded runs' layout."""
    for stem, prediction in predictions.items():
        np.save(directory / prediction_file(stem), np.asarray(prediction, dtype=bool))
    write_json_atomic(directory / RECORDED_RESULTS, {
        'decoders': {stem: {'shots': int(len(prediction)),
                            'prediction_packed_sha256': recorded_hash(prediction)}
                     for stem, prediction in predictions.items()},
    })


def write_recorded_run_with_predictions(sample: SampleSet, directory, predictions) -> Path:
    """The recorded four-decoder run layout with its saved predictions, from a sample."""
    saved = sample.save(directory)
    (saved / SAMPLE_MANIFEST).unlink()
    write_json_atomic(saved / RECORDED_MANIFEST, {
        'parameters': {**sample.parameters.to_json(), 'shots': sample.shots, 'seed': sample.seed},
        'input_sha256': {
            CIRCUIT_FILE: sha256_file(saved / CIRCUIT_FILE),
            DEM_FILE: sha256_file(saved / DEM_FILE),
            'packed_detectors_then_observables_payload': sample.payload_sha256,
        },
        'versions': package_versions(sorted(set(MODEL_PACKAGES) | set(SAMPLING_PACKAGES))),
        'source_sha256': {FAKE_SOURCE: sha256_file(REPOSITORY_ROOT / FAKE_SOURCE)},
        'code_commit': FAKE_COMMIT,
        'created_utc': '2026-09-10T05:22:38Z',
    })
    write_predictions(saved, predictions)
    return saved


def copied(source, destination) -> Path:
    """A private copy of a directory, so a tamper test leaves the shared one alone."""
    shutil.copytree(source, destination)
    return destination


def replaced_prediction(run_dir: Path, stem: str, prediction: np.ndarray) -> None:
    """Overwrite one decoder's file in a run copy and refresh its hash in ``results.json``."""
    np.save(run_dir / prediction_file(stem), np.asarray(prediction, dtype=bool))
    results = read_json(run_dir / RECORDED_RESULTS)
    results['decoders'][stem]['prediction_packed_sha256'] = recorded_hash(prediction)
    write_json_atomic(run_dir / RECORDED_RESULTS, results)


@dataclass(frozen=True)
class FakeRun:
    """A fake recorded run and what it was built from."""
    directory: Path
    sample: SampleSet
    predictions: dict


@dataclass(frozen=True)
class Pipeline:
    """The records, calibrators, and replay a module's tests share.

    ``before`` is the evaluation record as collected, ``before_dir`` a copy of its
    directory taken before the import, ``imported`` the same directory after it,
    ``manifest_before`` the pre-import manifest, and ``replay_before`` a replay published
    before the import, which the summary must refuse afterwards.
    """
    root: Path
    run: FakeRun
    calibration: LoadedRecord
    before: LoadedRecord
    before_dir: Path
    imported: LoadedRecord
    manifest_before: dict
    calibrators: Path
    replay_before: Path


@pytest.fixture(scope='module')
def run(tmp_path_factory) -> FakeRun:
    sample = SampleSet.sample(PARAMETERS, seed=EVALUATION_SEED, shots=SHOTS)
    context = L1Context.from_dem_text(sample.dem_text, sample.parameters.patches)
    predictions = fake_predictions(sample, context)
    directory = write_recorded_run_with_predictions(
        sample, tmp_path_factory.mktemp('recorded') / 'run', predictions)
    return FakeRun(directory=directory, sample=sample, predictions=predictions)


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory, run) -> Pipeline:
    root = tmp_path_factory.mktemp('pipeline')
    calibration = stage_collect(CollectRequest(
        out_dir=root / 'calibration', role='calibration', parameters=PARAMETERS,
        seed=CALIBRATION_SEED, shots=SHOTS, chunk_size=CHUNK))
    before = stage_collect(CollectRequest(out_dir=root / 'evaluation', role='evaluation',
                                          recorded_run=run.directory, chunk_size=CHUNK))
    assert calibration is not None and before is not None
    calibrators = root / 'calibrators.json'
    stage_calibrate(calibration.directory, calibrators, ESTIMATORS)
    replay_before = root / 'replay_before'
    stage_replay(before.directory, calibrators, replay_before, CONFIGS)
    before_dir = copied(before.directory, root / 'evaluation_before')
    manifest_before = read_json(before.directory / RECORD_MANIFEST)
    imported = stage_import_baselines(before.directory, run.directory)
    return Pipeline(root=root, run=run, calibration=calibration, before=before,
                    before_dir=before_dir, imported=imported, manifest_before=manifest_before,
                    calibrators=calibrators, replay_before=replay_before)


def mapped(run: FakeRun, name: str, rows) -> np.ndarray:
    """What a baseline must hold on a record's parent rows."""
    return run.predictions[BASELINE_DECODERS[name]][np.asarray(rows)]


# --- names and conventions ---------------------------------------------------

def test_the_baseline_names_map_to_exactly_the_recorded_decoder_stems():
    assert dict(BASELINE_DECODERS) == {
        'joint_mwpm_recorded': 'mwpm', 'joint_uf': 'uf',
        'joint_correlated_mwpm': 'correlated_mwpm', 'joint_correlated_uf': 'correlated_uf'}
    assert RECORDED_JOINT_MWPM in BASELINE_DECODERS
    with pytest.raises(TypeError):
        BASELINE_DECODERS['joint_other'] = 'other'
    assert prediction_file('mwpm') == 'mwpm_predictions.npy'


def test_the_baselines_module_is_a_check_source_and_not_a_decoder_source():
    assert 'src/yoked/hierarchical/_baselines.py' in CHECK_SOURCES
    assert 'src/yoked/hierarchical/_baselines.py' not in DECODER_SOURCES


def test_the_packed_prediction_hash_follows_the_recorded_convention():
    prediction = np.zeros((2, 12), dtype=bool)
    prediction[0, [0, 3, 9]] = True          # bits 0 and 3 of byte 0, bit 1 of byte 1
    prediction[1, 11] = True                 # bit 3 of byte 1
    packed = bytes([0b00001001, 0b00000010, 0b00000000, 0b00001000])
    assert packed_prediction_sha256(prediction) == hashlib.sha256(packed).hexdigest()
    assert packed_prediction_sha256(prediction) == recorded_hash(prediction)
    assert packed_prediction_sha256(prediction.astype(np.uint8)) == recorded_hash(prediction)
    with pytest.raises(ValueError, match='only 0 and 1'):
        packed_prediction_sha256(prediction.astype(np.float64) + 0.5)


# --- loading a recorded run --------------------------------------------------

def test_loading_a_recorded_run_verifies_its_predictions_and_hashes(run):
    recorded = load_recorded_baselines(run.directory)
    sample = run.sample
    assert isinstance(recorded, RecordedBaselines)
    assert recorded.directory == run.directory
    assert recorded.parent_shots == SHOTS
    assert (recorded.circuit_sha256, recorded.dem_sha256, recorded.payload_sha256) == (
        sample.circuit_sha256, sample.dem_sha256, sample.payload_sha256)
    assert recorded.manifest_sha256 == sha256_file(run.directory / RECORDED_MANIFEST)
    assert recorded.results_sha256 == sha256_file(run.directory / RECORDED_RESULTS)
    assert tuple(recorded.predictions) == tuple(BASELINE_DECODERS)
    results = read_json(run.directory / RECORDED_RESULTS)
    for name, stem in BASELINE_DECODERS.items():
        prediction = recorded.predictions[name]
        assert prediction.shape == (SHOTS, NUM_SECTORS * PARAMETERS.patches)
        assert prediction.dtype == bool and not prediction.flags.writeable
        np.testing.assert_array_equal(prediction, run.predictions[stem])
        assert recorded.prediction_sha256[name] == \
            results['decoders'][stem]['prediction_packed_sha256']
    assert dict(recorded.source_sha256) == {FAKE_SOURCE: sha256_file(REPOSITORY_ROOT / FAKE_SOURCE)}
    assert recorded.code_commit == FAKE_COMMIT
    assert recorded.versions['stim'] == package_versions(MODEL_PACKAGES)['stim']
    for mapping in (recorded.predictions, recorded.prediction_sha256, recorded.versions,
                    recorded.source_sha256):
        with pytest.raises(TypeError):
            mapping['changed'] = None


def test_loading_a_subset_of_names_keeps_the_canonical_order(run):
    recorded = load_recorded_baselines(run.directory, names=('joint_uf', 'joint_mwpm_recorded'))
    assert tuple(recorded.predictions) == ('joint_mwpm_recorded', 'joint_uf')
    assert tuple(recorded.prediction_sha256) == ('joint_mwpm_recorded', 'joint_uf')


@pytest.mark.parametrize('names, message', [
    (('joint_uf', 'joint_other'), 'joint_other'),
    (('joint_uf', 'joint_uf'), 'distinct'),
    ((), 'at least one'),
])
def test_loading_rejects_unknown_repeated_or_no_names(run, names, message):
    with pytest.raises(ValueError, match=message):
        load_recorded_baselines(run.directory, names=names)


def test_a_tampered_results_hash_is_rejected(run, tmp_path):
    directory = copied(run.directory, tmp_path / 'run')
    results = read_json(directory / RECORDED_RESULTS)
    results['decoders']['mwpm']['prediction_packed_sha256'] = 'ff' * 32
    write_json_atomic(directory / RECORDED_RESULTS, results)
    with pytest.raises(ValueError, match=f'mwpm_predictions.npy.*{RECORDED_RESULTS}'):
        load_recorded_baselines(directory)


@pytest.mark.parametrize('shape', [(SHOTS, 13), (SHOTS - 1, 12), (SHOTS * 12,)])
def test_a_prediction_file_with_the_wrong_shape_is_rejected(run, tmp_path, shape):
    directory = copied(run.directory, tmp_path / 'run')
    np.save(directory / prediction_file('uf'), np.zeros(shape, dtype=bool))
    with pytest.raises(ValueError, match='uf_predictions.npy.*shape'):
        load_recorded_baselines(directory)


def test_a_nonbinary_prediction_file_is_rejected(run, tmp_path):
    directory = copied(run.directory, tmp_path / 'run')
    prediction = run.predictions['correlated_uf'].astype(np.float64)
    prediction[0, 0] = 0.5
    np.save(directory / prediction_file('correlated_uf'), prediction)
    with pytest.raises(ValueError, match='correlated_uf_predictions.npy.*only 0 and 1'):
        load_recorded_baselines(directory)


def test_a_missing_prediction_file_is_rejected(run, tmp_path):
    directory = copied(run.directory, tmp_path / 'run')
    (directory / prediction_file('correlated_mwpm')).unlink()
    with pytest.raises(ValueError, match='correlated_mwpm_predictions.npy'):
        load_recorded_baselines(directory)


@pytest.mark.parametrize('field', ['source_sha256', 'code_commit', 'versions'])
def test_a_run_without_recorded_implementation_provenance_is_rejected(run, tmp_path, field):
    directory = copied(run.directory, tmp_path / 'run')
    manifest = read_json(directory / RECORDED_MANIFEST)
    del manifest[field]
    write_json_atomic(directory / RECORDED_MANIFEST, manifest)
    with pytest.raises(ValueError, match=field):
        load_recorded_baselines(directory)


def test_a_results_file_without_a_decoder_entry_is_rejected(run, tmp_path):
    directory = copied(run.directory, tmp_path / 'run')
    results = read_json(directory / RECORDED_RESULTS)
    del results['decoders']['uf']
    write_json_atomic(directory / RECORDED_RESULTS, results)
    with pytest.raises(ValueError, match='uf'):
        load_recorded_baselines(directory)


def test_recorded_baselines_validate_their_own_fields(run):
    recorded = load_recorded_baselines(run.directory)
    with pytest.raises(ValueError, match='circuit_sha256'):
        dataclasses.replace(recorded, circuit_sha256='not a digest')
    with pytest.raises(ValueError, match='parent_shots'):
        dataclasses.replace(recorded, parent_shots=SHOTS - 1)
    with pytest.raises(ValueError, match='joint_uf'):
        dataclasses.replace(recorded, prediction_sha256={**recorded.prediction_sha256,
                                                         'joint_uf': 'ff' * 32})
    with pytest.raises(ValueError, match='joint_uf'):
        dataclasses.replace(recorded, predictions={**recorded.predictions,
                                                   'joint_uf': recorded.predictions['joint_mwpm_recorded']})
    with pytest.raises(ValueError, match='at least one'):
        dataclasses.replace(recorded, predictions={}, prediction_sha256={})


# --- the import --------------------------------------------------------------

def test_the_import_round_trips_through_load_record(pipeline):
    loaded = load_record(pipeline.imported.directory)
    rows = pipeline.before.record.rows
    assert tuple(loaded.record.baselines) == tuple(BASELINE_DECODERS)
    for name in BASELINE_DECODERS:
        baseline = loaded.record.baselines[name]
        assert baseline.shape == (SHOTS, NUM_SECTORS * PARAMETERS.patches)
        assert not baseline.flags.writeable
        np.testing.assert_array_equal(baseline, mapped(pipeline.run, name, rows))
    np.testing.assert_array_equal(loaded.record.baselines[RECORDED_JOINT_MWPM],
                                  loaded.record.joint_mwpm)
    assert loaded.identities == pipeline.before.identities
    assert dict(pipeline.before.record.baselines) == {}
    assert loaded.manifest['artifacts'][RECORD_FILE] == \
        sha256_file(loaded.directory / RECORD_FILE)
    assert loaded.manifest['artifacts'][RECORD_FILE] != \
        pipeline.before.manifest['artifacts'][RECORD_FILE]


def test_the_manifest_records_the_runs_provenance_and_the_gates(pipeline):
    block = read_json(pipeline.imported.directory / RECORD_MANIFEST)[BASELINES_FIELD]
    run = pipeline.run
    assert block['imported_utc'].endswith('Z')
    assert block['names'] == list(BASELINE_DECODERS)
    recorded = block['run']
    assert recorded['directory'] == str(run.directory.resolve())
    assert recorded['manifest_sha256'] == sha256_file(run.directory / RECORDED_MANIFEST)
    assert recorded['results_sha256'] == sha256_file(run.directory / RECORDED_RESULTS)
    assert recorded['payload_sha256'] == run.sample.payload_sha256
    assert recorded['circuit_sha256'] == run.sample.circuit_sha256
    assert recorded['dem_sha256'] == run.sample.dem_sha256
    assert recorded['parent_shots'] == SHOTS
    assert recorded['source_sha256'] == {FAKE_SOURCE: sha256_file(REPOSITORY_ROOT / FAKE_SOURCE)}
    assert recorded['code_commit'] == FAKE_COMMIT
    assert recorded['versions'] == read_json(run.directory / RECORDED_MANIFEST)['versions']
    results = read_json(run.directory / RECORDED_RESULTS)
    for name, stem in BASELINE_DECODERS.items():
        entry = block['decoders'][name]
        assert entry['file'] == prediction_file(stem)
        assert entry['prediction_packed_sha256'] == \
            results['decoders'][stem]['prediction_packed_sha256']
    joint = block['joint_mwpm']
    assert joint['baseline'] == RECORDED_JOINT_MWPM and joint['compared_with'] == 'joint_mwpm'
    assert joint['agreement'] == 1.0 and joint['disagreements'] == 0
    assert joint['tie_explained'] == 0 and joint['cost_tolerance'] == COST_TOLERANCE
    assert block['yoke_parity'] == {'rows': SHOTS, 'violations': 0}
    assert block['importer']['check_identity'] == pipeline.imported.manifest['checks']['identity']


def test_the_republished_manifest_changes_only_the_record_hash_and_the_baselines_block(pipeline):
    before = pipeline.manifest_before
    after = read_json(pipeline.imported.directory / RECORD_MANIFEST)
    assert set(after) == set(before) | {BASELINES_FIELD}
    for name in before:
        if name != 'artifacts':
            assert canonical_json(after[name]) == canonical_json(before[name]), name
    assert set(after['artifacts']) == set(before['artifacts']) == {RECORD_FILE}
    assert after['artifacts'][RECORD_FILE] != before['artifacts'][RECORD_FILE]


def test_a_second_import_returns_without_writing(pipeline, monkeypatch):
    directory = pipeline.imported.directory
    record_before = (directory / RECORD_FILE).read_bytes()
    manifest_before = (directory / RECORD_MANIFEST).read_bytes()

    def refuse(*arguments, **keywords):
        raise AssertionError('an identical import must not rewrite the record or its manifest')

    monkeypatch.setattr('yoked.hierarchical._stages._save_arrays', refuse)
    monkeypatch.setattr('yoked.hierarchical._stages.write_json_atomic', refuse)
    again = stage_import_baselines(directory, pipeline.run.directory)
    assert tuple(again.record.baselines) == tuple(BASELINE_DECODERS)
    assert (directory / RECORD_FILE).read_bytes() == record_before
    assert (directory / RECORD_MANIFEST).read_bytes() == manifest_before


def test_a_differing_second_import_names_the_first_differing_baseline(pipeline, tmp_path):
    directory = copied(pipeline.imported.directory, tmp_path / 'evaluation')
    run_dir = copied(pipeline.run.directory, tmp_path / 'run')
    replaced_prediction(run_dir, 'uf', flipped(pipeline.run.predictions['uf'], [20], (1, 2)))
    before = (directory / RECORD_MANIFEST).read_bytes()
    with pytest.raises(ValueError, match="already carries.*'joint_uf'"):
        stage_import_baselines(directory, run_dir)
    assert (directory / RECORD_MANIFEST).read_bytes() == before


def test_a_second_import_naming_other_baselines_is_refused(pipeline):
    with pytest.raises(ValueError, match='already carries'):
        stage_import_baselines(pipeline.imported.directory, pipeline.run.directory,
                               names=('joint_uf',))


def test_a_payload_mismatch_from_another_seed_is_rejected(pipeline, tmp_path):
    other = SampleSet.sample(PARAMETERS, seed=OTHER_SEED, shots=SHOTS)
    columns = NUM_SECTORS * PARAMETERS.patches
    run_dir = write_recorded_run_with_predictions(
        other, tmp_path / 'other-run',
        {stem: np.zeros((SHOTS, columns), dtype=bool) for stem in BASELINE_DECODERS.values()})
    assert other.payload_sha256 != pipeline.run.sample.payload_sha256
    with pytest.raises(ValueError, match='payload'):
        stage_import_baselines(pipeline.imported.directory, run_dir)


@pytest.mark.parametrize('field, name', [('circuit_sha256', CIRCUIT_FILE), ('dem_sha256', DEM_FILE)])
def test_a_circuit_or_dem_hash_mismatch_is_rejected(pipeline, field, name):
    recorded = dataclasses.replace(load_recorded_baselines(pipeline.run.directory),
                                   **{field: 'ff' * 32})
    with pytest.raises(ValueError, match=name):
        attach_baselines(pipeline.before, recorded)


def test_a_parity_violating_baseline_is_rejected(pipeline, tmp_path):
    run_dir = copied(pipeline.run.directory, tmp_path / 'run')
    prediction = np.array(pipeline.run.predictions['correlated_uf'])
    prediction[4, 0] ^= True                 # one sector of one patch: parity breaks
    replaced_prediction(run_dir, 'correlated_uf', prediction)
    with pytest.raises(ValueError, match="'joint_correlated_uf'.*parity.*4"):
        stage_import_baselines(pipeline.imported.directory, run_dir)


def most_costly_flip(record) -> tuple[int, tuple[int, int]]:
    """The parent row and two patches where flipping both sectors changes the total
    forced cost the most, so the flipped prediction is unmistakably not a cost tie."""
    shots, patches = record.shots, record.num_patches
    bits = record.joint_mwpm.reshape(shots, patches, NUM_SECTORS).astype(np.intp)
    shot, patch = np.arange(shots)[:, None], np.arange(patches)[None, :]
    current = record.forced_plain[shot, patch, bits[:, :, 0], bits[:, :, 1]]
    complement = record.forced_plain[shot, patch, 1 - bits[:, :, 0], 1 - bits[:, :, 1]]
    delta = complement - current                                   # (shots, patches)
    best, choice = 0.0, None
    for position in range(shots):
        for first in range(patches):
            for second in range(first + 1, patches):
                change = abs(delta[position, first] + delta[position, second])
                if change > best:
                    best, choice = change, (int(record.rows[position]), (first, second))
    assert best > 100 * COST_TOLERANCE
    return choice


def test_a_joint_mwpm_disagreement_that_is_not_a_cost_tie_is_rejected(pipeline, tmp_path):
    row, patches = most_costly_flip(pipeline.before.record)
    run_dir = copied(pipeline.run.directory, tmp_path / 'run')
    replaced_prediction(run_dir, 'mwpm', flipped(pipeline.run.predictions['mwpm'], [row], patches))
    with pytest.raises(ValueError, match=f"'{RECORDED_JOINT_MWPM}'.*cost.*{row}"):
        stage_import_baselines(pipeline.imported.directory, run_dir)


def test_a_joint_mwpm_disagreement_explained_by_a_cost_tie_is_accepted_and_counted(pipeline):
    before = pipeline.before
    position, patches = 4, (0, 3)
    row = int(before.record.rows[position])
    forced = np.array(before.record.forced_plain)
    for patch in patches:
        forced[position, patch] = 1.0        # every class costs the same: flipping is free
    loaded = LoadedRecord(record=dataclasses.replace(before.record, forced_plain=forced),
                          directory=before.directory, identities=before.identities,
                          manifest=before.manifest)
    recorded = load_recorded_baselines(pipeline.run.directory)
    mwpm = flipped(pipeline.run.predictions['mwpm'], [row], patches)
    recorded = dataclasses.replace(
        recorded, predictions={**recorded.predictions, RECORDED_JOINT_MWPM: mwpm},
        prediction_sha256={**recorded.prediction_sha256, RECORDED_JOINT_MWPM: recorded_hash(mwpm)})
    attached, block = attach_baselines(loaded, recorded)
    np.testing.assert_array_equal(attached.baselines[RECORDED_JOINT_MWPM], mwpm[before.record.rows])
    assert (attached.baselines[RECORDED_JOINT_MWPM] != attached.joint_mwpm).any()
    joint = block['joint_mwpm']
    assert joint['disagreements'] == 1 and joint['tie_explained'] == 1
    assert joint['agreement'] == (SHOTS - 1) / SHOTS
    assert joint['max_disagreement_cost_difference'] <= COST_TOLERANCE


def test_import_onto_a_calibration_record_is_refused(pipeline):
    with pytest.raises(ValueError, match="'calibration'.*'evaluation'"):
        stage_import_baselines(pipeline.calibration.directory, pipeline.run.directory)
    assert BASELINES_FIELD not in read_json(pipeline.calibration.directory / RECORD_MANIFEST)


def test_calibrators_fitted_before_the_import_still_load_and_replay_after_it(pipeline, tmp_path):
    calibrators, artifact = load_calibrators(pipeline.calibrators)
    assert sorted(calibrators) == sorted(ESTIMATORS)
    assert dict(load_record(artifact['record']['directory']).record.baselines) == {}
    assert BASELINES_FIELD not in pipeline.calibration.manifest
    results = stage_replay(pipeline.imported.directory, pipeline.calibrators,
                           tmp_path / 'replay_after', CONFIGS)
    assert len(results) == len(CONFIGS)
    assert 'Block failure' in stage_summarize([tmp_path / 'replay_after'], tmp_path / 'summary.md',
                                              replicates=REPLICATES, seed=DEFAULT_SEED)


def test_a_replay_made_before_the_import_is_rejected_by_the_summary(pipeline, tmp_path):
    with pytest.raises(ValueError, match='hashes'):
        stage_summarize([pipeline.replay_before], tmp_path / 'summary.md',
                        replicates=REPLICATES, seed=DEFAULT_SEED)


def test_an_interrupted_import_leaves_a_directory_the_loader_refuses(pipeline, tmp_path):
    directory = copied(pipeline.before_dir, tmp_path / 'interrupted')

    def stop() -> None:
        raise RuntimeError('interrupted before the manifest')

    with pytest.raises(RuntimeError, match='interrupted'):
        stage_import_baselines(directory, pipeline.run.directory,
                               hooks=_ImportHooks(before_manifest=stop))
    assert read_json(directory / RECORD_MANIFEST) == pipeline.manifest_before
    with pytest.raises(ValueError, match=f'{RECORD_FILE} hashes to'):
        load_record(directory)


def test_attaching_leaves_the_loaded_record_untouched(pipeline):
    recorded = load_recorded_baselines(pipeline.run.directory)
    attached, _ = attach_baselines(pipeline.before, recorded)
    assert dict(pipeline.before.record.baselines) == {}
    assert tuple(attached.baselines) == tuple(BASELINE_DECODERS)
    for name in ('actual', 'yoke', 'joint_mwpm', 'rows'):
        np.testing.assert_array_equal(getattr(attached, name), getattr(pipeline.before.record, name))
