"""Tests for the verified experiment stages and the thin CLI over them (_stages.py).

Checks: a distance-3 calibration/evaluation pipeline run both through the stage
functions and once through ``tools/hierarchical_experiment`` as a subprocess; a
stopped collection resuming into the same record; a second collect request with a
changed seed, shot count, parameters, recorded run, raw sample bytes, decoder
version, or generated model rejected by name even though the output directory
already exists; calibration refused on an evaluation record and replay refused on a
calibration record, on a record sharing the calibration parent sample or sampling
family, and on calibrators fitted under another model; a record whose collection
checks did not pass refused downstream; invalid knots, an altered calibrator
artifact, an altered prediction container, a replaced record, and a missing
completion manifest all refused; an interrupted replay publication leaving no
manifest and rerunning cleanly; a reused replay directory returning its stored
results without replaying again; a confirmation request refused; a change to the
replay sources leaving the collection identity and its reuse untouched; the import
stage attaching a fake recorded run's four baselines to an evaluation record once
through ``stage_import_baselines`` and once through the command line, the latter on a
record that starts a few parent rows into the run; the subset check passing a 48-row
collection against a 10-row collection of the same sampling call, once through
``stage_verify_subset`` and once through the command line, with its JSON naming both
records, refusing a subset of another seed on the parent-sample identity and a
calibration-role collection of the same sample on the role, writing nothing either
time, and reporting a subset whose stored values were altered under the array and
parent row, with the JSON written before the refusal; the summary scoring a record's
imported baselines, the collector's own joint MWPM, and every replayed cell on the same
shots, every rate equal to a direct computation from the arrays, rendered as a baselines
table after the endpoint groups with the paired difference of each cell against the
recorded joint MWPM, that difference unavailable by name when the record does not carry
the recorded joint MWPM, the summary manifest's inputs unchanged by baselines, and no
table and no JSON key for a record without them; and the report renderers printing
``unavailable`` with the denominator or eligible count beside it whenever a number, an
interval bound, or a rate is undefined.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import stim

from yoked.hierarchical._baselines import BASELINE_DECODERS, BASELINES_FIELD, RECORDED_JOINT_MWPM
from yoked.hierarchical._baselines_test import fake_predictions, write_recorded_run_with_predictions
from yoked.hierarchical._calibration import CLIP, KNOT_CONVENTION
from yoked.hierarchical._collect import (
    DETECTORS_FILE, SAMPLE_MANIFEST, CircuitParameters, SampleSet, CIRCUIT_FILE, DEM_FILE,
    RECORDED_MANIFEST,
)
from yoked.hierarchical._l1 import L1Context
from yoked.hierarchical._metrics import (
    DEFAULT_SEED, block_failures, bootstrap_rate, normalized_ler, paired_block_failure,
)
from yoked.hierarchical._provenance import (
    CALIBRATION_PACKAGES, DECODER_PACKAGES, MODEL_PACKAGES, REPLAY_SOURCES, REPOSITORY_ROOT,
    SAMPLING_PACKAGES, package_versions, read_json, sha256_file, source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import (
    RECORD_FILE, RECORD_MANIFEST, RECORD_SCHEMA, LoadedRecord, _save_arrays, load_record,
)
from yoked.hierarchical._replay import ReplayResult
from yoked.hierarchical._stages import (
    BASELINES_HEADING, COLLECTED_JOINT_MWPM, COLLECTED_KIND, HIERARCHICAL_KIND, HISTORICAL_KIND,
    REPLAY_ARRAYS_FILE, REPLAY_MANIFEST, REPLAY_RESULTS_FILE, SAMPLE_DIRECTORY, UNAVAILABLE,
    VERIFY_SUBSET_STAGE, CollectRequest, _baseline_paired_rows, _baseline_rows,
    _calibrator_payload_sha256, _interval, _number, _paired_rows, _rate, _ReplayHooks,
    config_directory_name, load_calibrators, parse_config, stage_calibrate, stage_collect,
    stage_import_baselines, stage_replay, stage_summarize, stage_verify_subset,
)

DISTANCE, ROUNDS, P = 3, 12, 0.005
"""A distance-3 circuit with 4d rounds and a slightly loud noise setting, so that a few
dozen shots still contain residual reference failures and finish in about a second."""

SHOTS = 64
"""Rows in each full sampling call of the pipeline; every one of them is decoded."""

CHUNK = 16
"""Rows per scheduled chunk, small enough that ``max_chunks=1`` leaves real work pending."""

CALIBRATION_SEED, EVALUATION_SEED = 1, 2
"""Two different sampling calls, so the evaluation record is held out from the fit."""

PARAMETERS = CircuitParameters(distance=DISTANCE, rounds=ROUNDS, p=P)

ESTIMATORS = ('uf:cluster_gap', 'uf:gap_correlated')
"""The one estimator pair the stage tests replay; the mathematics of the others is
already covered beside ``_replay.py``."""

INITIAL_ONLY_CONFIG = 'initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed'
ALL_REFINED_CONFIG = 'initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined'
CONFIGS = (INITIAL_ONLY_CONFIG, ALL_REFINED_CONFIG)
"""Both endpoints of one pair. The second omits ``outer``, which must default to mixed."""

REPLICATES = 50
"""Bootstrap replicates in the tests: enough to exercise the paired path, few enough to
keep the suite fast. The pilot quotes the 10,000 of ``DEFAULT_REPLICATES``."""

CLI_SHOTS = 16
"""The command-line pipeline decodes fewer shots still: it checks the wiring, not the
statistics, and pays five interpreter startups for it."""

CLI_FIRST_ROW = 4
"""The command-line import test collects its record from this parent row on, so that its
baselines are the library record's from the same row on rather than a prefix of the run:
a baseline mapped by position instead of by parent row id would not match."""

FULL_SHOTS = 48
"""Rows in the sampling call the subset check is exercised on; the full record decodes
every one of them."""

SUBSET_ROWS = np.array([0, 3, 7, 11, 19, 23, 29, 31, 40, 47])
"""The ten parent rows of the subset collection: not a prefix, and not contiguous, so
that a row's position in the subset never equals its parent row id."""

FULL_SEED, OTHER_SEED = 5, 6
"""The subset check's sampling call, and another call of the same size that shares its
model but not its shots."""

BASELINE_SEED = 3
"""The sampling call the baselines summary imports as a recorded run: not the calibration
call, so the pipeline's calibrators are held out from it, and not the evaluation call
either, so no two module fixtures share a parent sample."""


@dataclass(frozen=True)
class Reproduction:
    """A full collection and a subset collection of one sampling call.

    Fields: ``full`` the 48-row evaluation record; ``subset`` the 10-row evaluation
    record over ``SUBSET_ROWS`` of the same call; ``other`` a 10-row evaluation record
    over the same rows of another seed's call; ``calibration`` a 10-row calibration-role
    record of the full record's call.
    """

    full: LoadedRecord
    subset: LoadedRecord
    other: LoadedRecord
    calibration: LoadedRecord


@pytest.fixture(scope='module')
def reproduction(tmp_path_factory) -> Reproduction:
    root = tmp_path_factory.mktemp('reproduction')
    records = {}
    for name, role, seed, rows in (('full', 'evaluation', FULL_SEED, None),
                                   ('subset', 'evaluation', FULL_SEED, SUBSET_ROWS),
                                   ('other', 'evaluation', OTHER_SEED, SUBSET_ROWS),
                                   ('calibration', 'calibration', FULL_SEED, SUBSET_ROWS)):
        record = stage_collect(CollectRequest(out_dir=root / name, role=role, parameters=PARAMETERS,
                                              seed=seed, shots=FULL_SHOTS, rows=rows,
                                              chunk_size=CHUNK))
        assert record is not None
        records[name] = record
    return Reproduction(**records)


@dataclass(frozen=True)
class Pipeline:
    """One completed run of every stage, shared by the tests that only read it.

    Fields: ``root`` the directory holding everything; ``calibration`` and
    ``evaluation`` the two verified records; ``calibrators`` the fitted artifact's
    path; ``replay_dir`` the published replay directory.
    """

    root: Path
    calibration: LoadedRecord
    evaluation: LoadedRecord
    calibrators: Path
    replay_dir: Path


def generated_request(out_dir, role: str, seed: int, **overrides) -> CollectRequest:
    """A collect request for one full sampling call of the fixture circuit."""
    fields = dict(out_dir=out_dir, role=role, parameters=PARAMETERS, seed=seed, shots=SHOTS,
                  chunk_size=CHUNK)
    fields.update(overrides)
    return CollectRequest(**fields)


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory) -> Pipeline:
    root = tmp_path_factory.mktemp('pipeline')
    calibration = stage_collect(generated_request(root / 'calibration', 'calibration',
                                                  CALIBRATION_SEED))
    evaluation = stage_collect(generated_request(root / 'evaluation', 'evaluation',
                                                 EVALUATION_SEED))
    assert calibration is not None and evaluation is not None
    calibrators = root / 'calibrators.json'
    stage_calibrate(calibration.directory, calibrators, ESTIMATORS)
    replay_dir = root / 'replay'
    stage_replay(evaluation.directory, calibrators, replay_dir, CONFIGS)
    return Pipeline(root=root, calibration=calibration, evaluation=evaluation,
                    calibrators=calibrators, replay_dir=replay_dir)


def copied(source, destination) -> Path:
    """A private copy of a published directory, so a tamper test leaves the shared one alone."""
    shutil.copytree(source, destination)
    return destination


@dataclass(frozen=True)
class Baselined:
    """One summary over an evaluation record that carries imported baselines.

    Fields: ``root`` the directory holding everything; ``run`` the fake recorded run;
    ``before_dir`` a copy of the evaluation record taken before the import; ``record``
    the imported record; ``results`` the replayed configurations by name; ``markdown``
    the report; ``section`` the summary JSON's one record section.
    """

    root: Path
    run: Path
    before_dir: Path
    record: LoadedRecord
    results: Mapping[str, ReplayResult]
    markdown: str
    section: dict


@pytest.fixture(scope='module')
def baselined(tmp_path_factory, pipeline) -> Baselined:
    root = tmp_path_factory.mktemp('baselined')
    sample = SampleSet.sample(PARAMETERS, seed=BASELINE_SEED, shots=SHOTS)
    context = L1Context.from_dem_text(sample.dem_text, sample.parameters.patches)
    run = write_recorded_run_with_predictions(sample, root / 'run',
                                              fake_predictions(sample, context))
    collected = stage_collect(CollectRequest(out_dir=root / 'evaluation', role='evaluation',
                                             recorded_run=run, chunk_size=CHUNK))
    assert collected is not None
    before_dir = copied(collected.directory, root / 'evaluation_before')
    record = stage_import_baselines(collected.directory, run)
    results = stage_replay(record.directory, pipeline.calibrators, root / 'replay', CONFIGS)
    markdown = stage_summarize([root / 'replay'], root / 'summary.md', replicates=REPLICATES,
                               seed=DEFAULT_SEED)
    section, = read_json(root / 'summary.json')['records']
    return Baselined(root=root, run=run, before_dir=before_dir, record=record, results=results,
                     markdown=markdown, section=section)


def write_recorded_run(sample: SampleSet, directory: Path) -> Path:
    """The recorded four-decoder run layout, rebuilt from a generated sample."""
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
        'created_utc': '2026-09-10T05:22:38Z',
    })
    return saved


def config_directories(replay_dir: Path) -> list[str]:
    return sorted(child.name for child in replay_dir.iterdir() if child.is_dir())


# --- the pipeline ------------------------------------------------------------

def test_the_pipeline_publishes_records_calibrators_a_replay_and_a_summary(pipeline, tmp_path):
    assert pipeline.calibration.manifest['role'] == 'calibration'
    assert pipeline.evaluation.manifest['role'] == 'evaluation'
    artifact = read_json(pipeline.calibrators)
    assert artifact['stage'] == 'calibrate' and artifact['clip'] == CLIP
    assert artifact['knot_convention'] == KNOT_CONVENTION
    assert sorted(artifact['estimators']) == sorted(ESTIMATORS)

    manifest = read_json(pipeline.replay_dir / REPLAY_MANIFEST)
    assert manifest['stage'] == 'replay' and manifest['status'] == 'complete'
    assert len(manifest['configurations']) == len(CONFIGS)
    for name in config_directories(pipeline.replay_dir):
        for artifact_name in (REPLAY_ARRAYS_FILE, REPLAY_RESULTS_FILE):
            relative = f'{name}/{artifact_name}'
            assert manifest['artifacts'][relative] == sha256_file(pipeline.replay_dir / relative)

    out = tmp_path / 'summary.md'
    markdown = stage_summarize([pipeline.replay_dir], out, replicates=REPLICATES, seed=DEFAULT_SEED)
    assert out.read_text(encoding='utf-8') == markdown
    assert 'Block failure' in markdown and 'uf:cluster_gap' in markdown
    for config in ('initial_only', 'all_refined'):
        assert config in markdown
    summary = read_json(tmp_path / 'summary.json')
    assert summary['bootstrap'] == {'replicates': REPLICATES, 'seed': DEFAULT_SEED}
    published = read_json(tmp_path / 'summary.manifest.json')
    assert published['status'] == 'complete'
    assert published['artifacts']['summary.md'] == sha256_file(out)


def test_the_directory_of_a_configuration_is_deterministic_and_plain():
    config = parse_config(ALL_REFINED_CONFIG)
    assert config_directory_name(config) == 'uf_cluster_gap_to_gap_correlated_all_refined_mixed'
    assert config_directory_name(parse_config(ALL_REFINED_CONFIG)) == config_directory_name(config)


def test_a_configuration_defaults_to_the_mixed_outer_rule():
    config = parse_config(ALL_REFINED_CONFIG)
    assert config.outer == 'mixed'
    assert config.initial.name == 'uf:cluster_gap' and config.refined.name == 'uf:gap_correlated'
    assert config.policy.name == 'all_refined'


@pytest.mark.parametrize('text, message', [
    ('initial=uf:cluster_gap,refined=gap_correlated', 'policy'),
    ('initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,budget=3', 'budget'),
    ('initial=uf:cluster_gap,refined=uf:gap_correlated,policy=all_refined', 'score'),
    ('initial=uf:cluster_gap,refined=gap_correlated,policy=top_k_given_yoke', 'top_k_given_yoke'),
    ('initial=uf,refined=gap_correlated,policy=all_refined', 'estimator'),
])
def test_an_unusable_configuration_names_what_is_wrong(text, message):
    with pytest.raises(ValueError, match=message):
        parse_config(text)


# --- the command line --------------------------------------------------------

def run_cli(*arguments, check: bool = True) -> subprocess.CompletedProcess:
    """Run the driver exactly as the pilot does: from the repository root with src on the path."""
    environment = {**os.environ, 'PYTHONPATH': 'src'}
    return subprocess.run([sys.executable, 'tools/hierarchical_experiment', *arguments],
                          cwd=REPOSITORY_ROOT, env=environment, capture_output=True, text=True,
                          check=check)


def test_the_command_line_runs_every_stage(tmp_path):
    root = tmp_path
    common = ['--distance', str(DISTANCE), '--rounds', str(ROUNDS), '--p', str(P),
              '--rows', f'0:{CLI_SHOTS}', '--workers', '1', '--chunk-size', '8']
    run_cli('collect', '--out', str(root / 'calibration'), '--role', 'calibration',
            '--seed', str(CALIBRATION_SEED), '--shots', str(CLI_SHOTS), *common)
    run_cli('collect', '--out', str(root / 'evaluation'), '--role', 'evaluation',
            '--seed', str(EVALUATION_SEED), '--shots', str(CLI_SHOTS), *common)
    run_cli('calibrate', '--record', str(root / 'calibration'),
            '--out', str(root / 'calibrators.json'), '--estimators', *ESTIMATORS)
    run_cli('replay', '--record', str(root / 'evaluation'),
            '--calibrators', str(root / 'calibrators.json'), '--out', str(root / 'replay'),
            '--config', INITIAL_ONLY_CONFIG, '--config', ALL_REFINED_CONFIG)
    run_cli('summarize', '--replays', str(root / 'replay'), '--out', str(root / 'summary.md'),
            '--replicates', str(REPLICATES), '--seed', str(DEFAULT_SEED))

    assert read_json(root / 'evaluation' / RECORD_MANIFEST)['status'] == 'complete'
    assert read_json(root / 'replay' / REPLAY_MANIFEST)['status'] == 'complete'
    assert read_json(root / 'summary.manifest.json')['status'] == 'complete'
    assert 'Block failure' in (root / 'summary.md').read_text(encoding='utf-8')


def test_the_stage_and_the_command_line_import_baselines_once_each(tmp_path):
    sample = SampleSet.sample(PARAMETERS, seed=7, shots=CLI_SHOTS)
    context = L1Context.from_dem_text(sample.dem_text, sample.parameters.patches)
    run = write_recorded_run_with_predictions(sample, tmp_path / 'run',
                                              fake_predictions(sample, context))
    library = stage_collect(CollectRequest(out_dir=tmp_path / 'library', role='evaluation',
                                           recorded_run=run, chunk_size=CHUNK))
    assert library is not None and dict(library.record.baselines) == {}
    imported = stage_import_baselines(library.directory, run)
    assert tuple(imported.record.baselines) == tuple(BASELINE_DECODERS)
    assert imported.manifest[BASELINES_FIELD]['names'] == tuple(BASELINE_DECODERS)

    run_cli('collect', '--out', str(tmp_path / 'cli'), '--role', 'evaluation',
            '--recorded-run', str(run), '--rows', f'{CLI_FIRST_ROW}:{CLI_SHOTS}',
            '--workers', '1', '--chunk-size', '8')
    finished = run_cli('import-baselines', '--record', str(tmp_path / 'cli'),
                       '--recorded-run', str(run))
    assert f'{len(BASELINE_DECODERS)} baselines' in finished.stdout
    loaded = load_record(tmp_path / 'cli')
    assert tuple(loaded.record.baselines) == tuple(BASELINE_DECODERS)
    np.testing.assert_array_equal(loaded.record.rows, np.arange(CLI_FIRST_ROW, CLI_SHOTS))
    for name in BASELINE_DECODERS:
        np.testing.assert_array_equal(loaded.record.baselines[name],
                                      imported.record.baselines[name][CLI_FIRST_ROW:])
    assert read_json(tmp_path / 'cli' / RECORD_MANIFEST)[BASELINES_FIELD]['run']['directory'] == \
        str(run.resolve())


def test_the_command_line_rejects_an_unusable_row_range(tmp_path):
    finished = run_cli('collect', '--out', str(tmp_path / 'out'), '--role', 'evaluation',
                       '--distance', str(DISTANCE), '--rounds', str(ROUNDS), '--p', str(P),
                       '--seed', '1', '--shots', '8', '--rows', '8:4', check=False)
    assert finished.returncode != 0 and 'rows' in finished.stderr


def test_the_command_line_rejects_a_confirmation_role(tmp_path):
    finished = run_cli('collect', '--out', str(tmp_path / 'out'), '--role', 'confirmation',
                       '--distance', str(DISTANCE), '--rounds', str(ROUNDS), '--p', str(P),
                       '--seed', '1', '--shots', '8', check=False)
    assert finished.returncode != 0


# --- collect requests --------------------------------------------------------

def test_a_confirmation_request_is_refused_until_freeze_verification_exists(tmp_path):
    with pytest.raises(ValueError, match='freeze'):
        generated_request(tmp_path / 'out', 'confirmation', 1)


def test_a_request_names_exactly_one_sample_source(tmp_path):
    with pytest.raises(ValueError, match='recorded_run'):
        CollectRequest(out_dir=tmp_path / 'out', role='evaluation', seed=1)
    with pytest.raises(ValueError, match='both'):
        generated_request(tmp_path / 'out', 'evaluation', 1, recorded_run=tmp_path / 'run')


def test_a_stopped_collection_resumes_into_the_same_record(tmp_path):
    out_dir = tmp_path / 'resumed'
    assert stage_collect(generated_request(out_dir, 'evaluation', EVALUATION_SEED,
                                           max_chunks=1)) is None
    assert not (out_dir / RECORD_MANIFEST).exists()
    finished = stage_collect(generated_request(out_dir, 'evaluation', EVALUATION_SEED))
    assert finished is not None and finished.record.shots == SHOTS
    assert finished.manifest['collection_work']['resumptions'] == 1


@pytest.mark.parametrize('field, overrides, message', [
    ('seed', dict(seed=CALIBRATION_SEED + 10), 'seed'),
    ('shots', dict(shots=SHOTS + 8), 'shots'),
    ('parameters', dict(parameters=CircuitParameters(distance=DISTANCE, rounds=ROUNDS, p=2 * P)),
     'parameters'),
])
def test_a_changed_request_cannot_reuse_an_existing_sample(pipeline, tmp_path, field, overrides,
                                                           message):
    out_dir = copied(pipeline.calibration.directory, tmp_path / f'changed-{field}')
    changed = {'seed': CALIBRATION_SEED, **overrides}
    with pytest.raises(ValueError, match=message):
        stage_collect(generated_request(out_dir, 'calibration', **changed))


def test_a_changed_generated_model_cannot_reuse_an_existing_sample(pipeline, tmp_path, monkeypatch):
    out_dir = copied(pipeline.calibration.directory, tmp_path / 'changed-model')
    original = CircuitParameters.dem

    def louder(self, circuit):
        """A generator whose model gains one extra mechanism under unchanged parameters."""
        dem = original(self, circuit)
        return dem + stim.DetectorErrorModel('error(0.000000001) D0')

    monkeypatch.setattr(CircuitParameters, 'dem', louder)
    with pytest.raises(ValueError, match='model'):
        stage_collect(generated_request(out_dir, 'calibration', CALIBRATION_SEED))


def test_changed_sample_bytes_under_an_unchanged_manifest_are_rejected(pipeline, tmp_path):
    out_dir = copied(pipeline.calibration.directory, tmp_path / 'tampered-sample')
    packed = np.load(out_dir / SAMPLE_DIRECTORY / DETECTORS_FILE)
    packed[0, 0] ^= 1
    np.save(out_dir / SAMPLE_DIRECTORY / DETECTORS_FILE, packed)
    with pytest.raises(ValueError, match=DETECTORS_FILE):
        stage_collect(generated_request(out_dir, 'calibration', CALIBRATION_SEED))


def test_a_changed_decoder_version_cannot_join_an_existing_collection(pipeline, tmp_path,
                                                                     monkeypatch):
    out_dir = copied(pipeline.calibration.directory, tmp_path / 'other-decoder')

    def bumped(names):
        installed = package_versions(names)
        if names is DECODER_PACKAGES:
            installed['numpy'] = installed['numpy'] + '.1'
        return installed

    monkeypatch.setattr('yoked.hierarchical._collect.package_versions', bumped)
    with pytest.raises(ValueError, match='decoder'):
        stage_collect(generated_request(out_dir, 'calibration', CALIBRATION_SEED))


def test_an_imported_request_reuses_its_saved_sample_and_rejects_another_run(tmp_path):
    sample = SampleSet.sample(PARAMETERS, seed=7, shots=CLI_SHOTS)
    run = write_recorded_run(sample, tmp_path / 'run')
    out_dir = tmp_path / 'imported'
    record = stage_collect(CollectRequest(out_dir=out_dir, role='evaluation', recorded_run=run,
                                          rows=np.arange(CLI_SHOTS // 2), chunk_size=CHUNK))
    assert record is not None and record.record.shots == CLI_SHOTS // 2
    again = stage_collect(CollectRequest(out_dir=out_dir, role='evaluation', recorded_run=run,
                                         rows=np.arange(CLI_SHOTS // 2), chunk_size=CHUNK))
    assert again.identities == record.identities

    other = write_recorded_run(SampleSet.sample(PARAMETERS, seed=8, shots=CLI_SHOTS),
                               tmp_path / 'other-run')
    with pytest.raises(ValueError, match='payload'):
        stage_collect(CollectRequest(out_dir=out_dir, role='evaluation', recorded_run=other,
                                     rows=np.arange(CLI_SHOTS // 2), chunk_size=CHUNK))


# --- verify subset -----------------------------------------------------------

def test_a_subset_collection_reproduces_the_full_one_and_the_json_names_both(reproduction,
                                                                             tmp_path):
    out = tmp_path / 'subset_check.json'
    check = stage_verify_subset(reproduction.full.directory, reproduction.subset.directory, out)
    assert check.passed
    assert check.subset_rows == len(SUBSET_ROWS) and check.matched_rows == len(SUBSET_ROWS)
    written = read_json(out)
    assert written['stage'] == VERIFY_SUBSET_STAGE and written['passed'] is True
    assert written['check'] == check.to_json()
    for name, loaded in (('full', reproduction.full), ('subset', reproduction.subset)):
        block = written[name]
        assert block['directory'] == str(loaded.directory.resolve())
        assert block['record_sha256'] == loaded.manifest['artifacts'][RECORD_FILE]
        assert block['manifest_sha256'] == sha256_file(loaded.directory / RECORD_MANIFEST)
        assert block['identities'] == dict(loaded.identities)
        assert block['rows'] == dict(loaded.manifest['rows'])
        assert block['role'] == 'evaluation' and block['shots'] == loaded.record.shots
    assert written['full']['rows']['count'] == FULL_SHOTS
    assert written['subset']['rows']['count'] == len(SUBSET_ROWS)
    assert written['full']['identities']['parent_sample'] == \
        written['subset']['identities']['parent_sample']
    assert written['full']['identities']['collection'] != \
        written['subset']['identities']['collection']
    assert written['checker']['check_identity']


def test_a_subset_of_another_sampling_call_is_refused_on_its_identity(reproduction, tmp_path):
    out = tmp_path / 'other_seed.json'
    with pytest.raises(ValueError, match='parent_sample'):
        stage_verify_subset(reproduction.full.directory, reproduction.other.directory, out)
    assert not out.exists()


def test_a_subset_of_another_role_is_refused_on_the_role(reproduction, tmp_path):
    # Same model, sampling call, and decoder, so every compared identity agrees and only
    # the role can name the difference.
    assert reproduction.calibration.identities['parent_sample'] == \
        reproduction.full.identities['parent_sample']
    out = tmp_path / 'other_role.json'
    with pytest.raises(ValueError, match='role'):
        stage_verify_subset(reproduction.full.directory, reproduction.calibration.directory, out)
    assert not out.exists()


def test_a_subset_whose_stored_values_differ_is_reported_and_refused(reproduction, tmp_path):
    subset_dir = copied(reproduction.subset.directory, tmp_path / 'altered-subset')
    arrays = reproduction.subset.record.arrays()
    gaps = np.array(arrays['cluster_gap'])
    gaps[0, 0] += 1.0
    arrays['cluster_gap'] = gaps
    # Republished consistently, so that load_record accepts the altered record and the
    # difference is found by the comparison rather than by an artifact hash.
    _save_arrays(subset_dir / RECORD_FILE, arrays, schema=RECORD_SCHEMA)
    manifest = read_json(subset_dir / RECORD_MANIFEST)
    manifest['artifacts'][RECORD_FILE] = sha256_file(subset_dir / RECORD_FILE)
    write_json_atomic(subset_dir / RECORD_MANIFEST, manifest)

    out = tmp_path / 'altered_check.json'
    with pytest.raises(ValueError, match=rf'cluster_gap.*\b{int(SUBSET_ROWS[0])}\b'):
        stage_verify_subset(reproduction.full.directory, subset_dir, out)
    written = read_json(out)
    assert written['passed'] is False
    assert written['check']['equal']['cluster_gap'] is False
    assert written['check']['mismatch_counts']['cluster_gap'] == 1
    assert written['check']['mismatched_rows']['cluster_gap'] == [int(SUBSET_ROWS[0])]
    assert written['check']['matched_rows'] == len(SUBSET_ROWS) - 1
    assert all(written['check']['equal'][name] for name in written['check']['equal']
               if name != 'cluster_gap')


def test_the_command_line_verifies_a_subset_once(reproduction, tmp_path):
    out = tmp_path / 'cli_check.json'
    finished = run_cli('verify-subset', '--full', str(reproduction.full.directory),
                       '--subset', str(reproduction.subset.directory), '--out', str(out))
    assert f'{len(SUBSET_ROWS)} of {len(SUBSET_ROWS)}' in finished.stdout
    written = read_json(out)
    assert written['passed'] is True and written['stage'] == VERIFY_SUBSET_STAGE
    assert written['check']['subset_rows'] == len(SUBSET_ROWS)


# --- calibration -------------------------------------------------------------

def test_calibration_requires_a_calibration_record(pipeline, tmp_path):
    with pytest.raises(ValueError, match='calibration'):
        stage_calibrate(pipeline.evaluation.directory, tmp_path / 'calibrators.json', ESTIMATORS)


def test_a_record_whose_checks_did_not_pass_is_refused_downstream(pipeline, tmp_path):
    out_dir = copied(pipeline.calibration.directory, tmp_path / 'failed-checks')
    manifest = read_json(out_dir / RECORD_MANIFEST)
    manifest['checks']['passed'] = False
    write_json_atomic(out_dir / RECORD_MANIFEST, manifest)
    with pytest.raises(ValueError, match='checks'):
        stage_calibrate(out_dir, tmp_path / 'calibrators.json', ESTIMATORS)


def test_a_missing_completion_manifest_is_refused_downstream(pipeline, tmp_path):
    out_dir = copied(pipeline.calibration.directory, tmp_path / 'incomplete')
    (out_dir / RECORD_MANIFEST).unlink()
    with pytest.raises(ValueError, match='missing'):
        stage_calibrate(out_dir, tmp_path / 'calibrators.json', ESTIMATORS)


def test_the_fitted_calibrators_round_trip_through_the_artifact(pipeline):
    calibrators, artifact = load_calibrators(pipeline.calibrators)
    assert sorted(calibrators) == sorted(ESTIMATORS)
    for pair in calibrators.values():
        assert all(item.direction == 'decreasing' for item in pair)
    assert artifact['record']['identities']['collection'] == \
        pipeline.calibration.identities['collection']


def test_nonmonotone_knots_are_rejected(pipeline, tmp_path):
    path = tmp_path / 'nonmonotone.json'
    artifact = read_json(pipeline.calibrators)
    entry = artifact['estimators'][ESTIMATORS[0]]['X']
    entry['probabilities'] = sorted(entry['probabilities'])
    entry['probabilities'][0] = CLIP
    entry['probabilities'][-1] = 1 - CLIP
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match='monotone'):
        load_calibrators(path)


@pytest.mark.parametrize('field, value, message', [
    ('knot_convention', 'weighted means of every score', 'knot_convention'),
    ('clip', 1e-3, 'clip'),
    ('stage', 'replay', 'stage'),
    ('schema_version', 'hierarchical-l1-l2/0', 'schema'),
])
def test_an_altered_calibrator_artifact_is_rejected(pipeline, tmp_path, field, value, message):
    path = tmp_path / 'altered.json'
    artifact = read_json(pipeline.calibrators)
    artifact[field] = value
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match=message):
        load_calibrators(path)


def test_an_altered_record_hash_breaks_the_calibration_identity(pipeline, tmp_path):
    path = tmp_path / 'relabelled.json'
    artifact = read_json(pipeline.calibrators)
    artifact['record']['record_sha256'] = 'ff' * 32
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match='identity'):
        load_calibrators(path)


@pytest.mark.parametrize('field', ['probabilities', 'centers', 'num_samples'])
def test_valid_edits_to_fitted_values_fail_the_payload_checksum(pipeline, tmp_path, field):
    path = tmp_path / 'edited-fit.json'
    artifact = read_json(pipeline.calibrators)
    entry = artifact['estimators'][ESTIMATORS[0]]['X']
    if field == 'probabilities':
        entry[field] = [0.25] * len(entry[field])
    elif field == 'centers':
        entry[field] = [center + 0.125 for center in entry[field]]
    else:
        entry[field] += 1
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match='payload checksum'):
        load_calibrators(path)
    with pytest.raises(ValueError, match='payload checksum'):
        stage_replay(pipeline.evaluation.directory, path, tmp_path / 'replay', CONFIGS)
    assert not (tmp_path / 'replay').exists()


@pytest.mark.parametrize('field', ['role', 'manifest_sha256', 'parent_sample', 'sampling_family',
                                  'collection'])
@pytest.mark.parametrize('refresh_checksum', [False, True])
def test_calibrator_provenance_must_match_the_verified_source_record(
        pipeline, tmp_path, field, refresh_checksum):
    path = tmp_path / 'edited-parent.json'
    artifact = read_json(pipeline.calibrators)
    record = artifact['record']
    if field == 'role':
        record[field] = 'evaluation'
    elif field == 'manifest_sha256':
        record[field] = 'ff' * 32
    else:
        record['identities'][field] = 'ff' * 32
    if refresh_checksum:
        artifact['payload_sha256'] = _calibrator_payload_sha256(record, artifact['estimators'])
    write_json_atomic(path, artifact)
    message = 'payload checksum'
    if refresh_checksum:
        message = {'manifest_sha256': RECORD_MANIFEST, 'role': 'evaluation'}.get(field, field)
    with pytest.raises(ValueError, match=message):
        load_calibrators(path)


def test_calibrator_loading_requires_the_source_record_to_have_the_calibration_role(pipeline,
                                                                                  tmp_path):
    path = tmp_path / 'evaluation-parent.json'
    artifact = read_json(pipeline.calibrators)
    artifact['record']['directory'] = str(pipeline.evaluation.directory)
    artifact['payload_sha256'] = _calibrator_payload_sha256(artifact['record'], artifact['estimators'])
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match='requires the.*calibration'):
        load_calibrators(path)


@pytest.mark.parametrize('damage', ['missing', 'replaced'])
def test_calibrator_loading_reverifies_its_source_record(pipeline, tmp_path, damage):
    record_dir = copied(pipeline.calibration.directory, tmp_path / 'calibration')
    path = tmp_path / 'calibrators.json'
    stage_calibrate(record_dir, path, ESTIMATORS)
    if damage == 'missing':
        (record_dir / RECORD_MANIFEST).unlink()
    else:
        shutil.copyfile(pipeline.evaluation.directory / RECORD_FILE, record_dir / RECORD_FILE)
    with pytest.raises(ValueError, match='missing' if damage == 'missing' else 'hashes'):
        load_calibrators(path)


def test_repeated_calibration_preserves_the_artifact_and_reuses_replay(pipeline, tmp_path,
                                                                     monkeypatch):
    path = tmp_path / 'calibrators.json'
    shutil.copyfile(pipeline.calibrators, path)
    before = path.read_bytes()

    def refuse(*arguments, **keywords):
        raise AssertionError('identical calibration and replay must reuse their outputs')

    monkeypatch.setattr('yoked.hierarchical._stages.fit_calibrators', refuse)
    monkeypatch.setattr('yoked.hierarchical._stages.replay', refuse)
    monkeypatch.setattr('yoked.hierarchical._stages.utc_now', lambda: '2099-01-01T00:00:00Z')
    fitted = stage_calibrate(pipeline.calibration.directory, path, reversed(ESTIMATORS))
    assert sorted(fitted) == sorted(ESTIMATORS)
    assert path.read_bytes() == before
    results = stage_replay(pipeline.evaluation.directory, path, pipeline.replay_dir, CONFIGS)
    assert sorted(results) == sorted(read_json(pipeline.replay_dir / REPLAY_MANIFEST)['configurations'])


@pytest.mark.parametrize('changed', ['estimators', 'sources', 'versions', 'record'])
def test_calibration_refuses_to_overwrite_different_inputs(pipeline, tmp_path, monkeypatch, changed):
    path = tmp_path / 'calibrators.json'
    shutil.copyfile(pipeline.calibrators, path)
    before = path.read_bytes()
    estimators = ESTIMATORS
    record_dir = pipeline.calibration.directory
    if changed == 'estimators':
        estimators = ESTIMATORS[:1]
    elif changed == 'sources':
        def moved(group):
            return {name: 'ff' * 32 for name in group}
        monkeypatch.setattr('yoked.hierarchical._stages.source_hashes', moved)
    elif changed == 'versions':
        def upgraded(group):
            versions = package_versions(group)
            if group is CALIBRATION_PACKAGES:
                versions['numpy'] = 'other-version'
            return versions
        monkeypatch.setattr('yoked.hierarchical._stages.package_versions', upgraded)
    else:
        record_dir = tmp_path / 'other-record'
        stage_collect(generated_request(record_dir, 'calibration', CALIBRATION_SEED,
                                        rows=np.arange(SHOTS // 2)))
    with pytest.raises(ValueError, match='new output path'):
        stage_calibrate(record_dir, path, estimators)
    assert path.read_bytes() == before


def test_repeating_calibration_does_not_overwrite_a_corrupt_artifact(pipeline, tmp_path):
    path = tmp_path / 'calibrators.json'
    artifact = read_json(pipeline.calibrators)
    entry = artifact['estimators'][ESTIMATORS[0]]['X']
    entry['probabilities'] = [0.25] * len(entry['probabilities'])
    write_json_atomic(path, artifact)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='payload checksum'):
        stage_calibrate(pipeline.calibration.directory, path, ESTIMATORS)
    assert path.read_bytes() == before


def test_a_legacy_calibrator_without_payload_integrity_is_rejected(pipeline, tmp_path):
    path = tmp_path / 'legacy.json'
    artifact = read_json(pipeline.calibrators)
    del artifact['payload_sha256']
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match='payload_sha256'):
        load_calibrators(path)


# --- replay ------------------------------------------------------------------

def test_replay_requires_an_evaluation_record(pipeline, tmp_path):
    with pytest.raises(ValueError, match='evaluation'):
        stage_replay(pipeline.calibration.directory, pipeline.calibrators, tmp_path / 'replay',
                     CONFIGS)


def test_replay_refuses_the_calibration_parent_sample(pipeline, tmp_path):
    out_dir = tmp_path / 'same-parent'
    record = stage_collect(generated_request(out_dir, 'evaluation', CALIBRATION_SEED))
    assert record is not None
    with pytest.raises(ValueError, match='parent_sample'):
        stage_replay(out_dir, pipeline.calibrators, tmp_path / 'replay', CONFIGS)


def test_replay_refuses_the_calibration_sampling_family(pipeline, tmp_path):
    out_dir = tmp_path / 'same-family'
    # The same circuit and seed with a shorter call: a different parent sample whose shot
    # stream may still overlap the calibration one.
    record = stage_collect(CollectRequest(out_dir=out_dir, role='evaluation', parameters=PARAMETERS,
                                          seed=CALIBRATION_SEED, shots=SHOTS // 2,
                                          chunk_size=CHUNK))
    assert record is not None
    with pytest.raises(ValueError, match='sampling_family'):
        stage_replay(out_dir, pipeline.calibrators, tmp_path / 'replay', CONFIGS)


def test_replay_refuses_calibrators_fitted_on_another_model(pipeline, tmp_path):
    other = tmp_path / 'distance-5'
    parameters = CircuitParameters(distance=5, rounds=4 * 5, p=P)
    record = stage_collect(CollectRequest(out_dir=other, role='calibration', parameters=parameters,
                                          seed=CALIBRATION_SEED, shots=8, chunk_size=8))
    assert record is not None
    calibrators = tmp_path / 'distance-5.json'
    stage_calibrate(other, calibrators, ESTIMATORS)
    with pytest.raises(ValueError, match='model'):
        stage_replay(pipeline.evaluation.directory, calibrators, tmp_path / 'replay', CONFIGS)


def test_replay_refuses_a_configuration_whose_estimator_is_absent(pipeline, tmp_path):
    absent = 'initial=uf:cluster_gap,refined=gap_plain,policy=all_refined,outer=mixed'
    with pytest.raises(ValueError, match='uf:gap_plain'):
        stage_replay(pipeline.evaluation.directory, pipeline.calibrators, tmp_path / 'replay',
                     [absent])


def test_an_interrupted_replay_publication_leaves_no_manifest_and_reruns_cleanly(pipeline,
                                                                                tmp_path):
    out_dir = tmp_path / 'interrupted'

    def stop() -> None:
        raise RuntimeError('interrupted before the replay manifest')

    with pytest.raises(RuntimeError, match='interrupted'):
        stage_replay(pipeline.evaluation.directory, pipeline.calibrators, out_dir, CONFIGS,
                     hooks=_ReplayHooks(before_manifest=stop))
    assert not (out_dir / REPLAY_MANIFEST).exists()
    assert config_directories(out_dir), 'the interruption should leave its partial artifacts'
    results = stage_replay(pipeline.evaluation.directory, pipeline.calibrators, out_dir, CONFIGS)
    assert sorted(results) == sorted(read_json(out_dir / REPLAY_MANIFEST)['configurations'])


def test_reusing_a_replay_directory_returns_the_stored_results(pipeline, monkeypatch):
    def refuse(*arguments, **keywords):
        raise AssertionError('a completed replay must not decode anything again')

    monkeypatch.setattr('yoked.hierarchical._stages.replay', refuse)
    results = stage_replay(pipeline.evaluation.directory, pipeline.calibrators,
                           pipeline.replay_dir, CONFIGS)
    assert sorted(results) == sorted(read_json(pipeline.replay_dir / REPLAY_MANIFEST)
                                     ['configurations'])


def test_a_replay_directory_of_other_inputs_requires_a_new_directory(pipeline, tmp_path):
    out_dir = copied(pipeline.replay_dir, tmp_path / 'other-inputs')
    with pytest.raises(ValueError, match='new'):
        stage_replay(pipeline.evaluation.directory, pipeline.calibrators, out_dir,
                     [INITIAL_ONLY_CONFIG])


def test_an_altered_calibrator_cannot_reuse_a_replay_directory(pipeline, tmp_path):
    out_dir = copied(pipeline.replay_dir, tmp_path / 'altered-calibrators')
    calibrators = tmp_path / 'calibrators.json'
    artifact = read_json(pipeline.calibrators)
    artifact['created_utc'] = '2026-01-01T00:00:00Z'
    write_json_atomic(calibrators, artifact)
    with pytest.raises(ValueError, match='new'):
        stage_replay(pipeline.evaluation.directory, calibrators, out_dir, CONFIGS)


# --- the summary -------------------------------------------------------------

def test_an_altered_prediction_container_is_rejected_by_the_summary(pipeline, tmp_path):
    out_dir = copied(pipeline.replay_dir, tmp_path / 'altered-arrays')
    target = out_dir / config_directories(out_dir)[0] / REPLAY_ARRAYS_FILE
    target.write_bytes(target.read_bytes() + b'\0')
    with pytest.raises(ValueError, match=REPLAY_ARRAYS_FILE):
        stage_summarize([out_dir], tmp_path / 'summary.md', replicates=REPLICATES, seed=DEFAULT_SEED)


def test_a_missing_replay_manifest_is_rejected_by_the_summary(pipeline, tmp_path):
    out_dir = copied(pipeline.replay_dir, tmp_path / 'no-manifest')
    (out_dir / REPLAY_MANIFEST).unlink()
    with pytest.raises(ValueError, match='missing'):
        stage_summarize([out_dir], tmp_path / 'summary.md', replicates=REPLICATES, seed=DEFAULT_SEED)


def test_a_record_replaced_after_replay_is_rejected_by_the_summary(pipeline, tmp_path):
    record_dir = copied(pipeline.evaluation.directory, tmp_path / 'evaluation')
    replay_dir = tmp_path / 'replay'
    stage_replay(record_dir, pipeline.calibrators, replay_dir, CONFIGS)
    fewer = tmp_path / 'fewer-rows'
    assert stage_collect(generated_request(fewer, 'evaluation', EVALUATION_SEED,
                                           rows=np.arange(SHOTS // 2))) is not None
    shutil.rmtree(record_dir)
    copied(fewer, record_dir)
    with pytest.raises(ValueError, match='rows'):
        stage_summarize([replay_dir], tmp_path / 'summary.md', replicates=REPLICATES,
                        seed=DEFAULT_SEED)


# --- baselines in the summary ------------------------------------------------

def predictions_of(baselined: Baselined, name: str) -> np.ndarray:
    """The (shots, 2P) prediction the baselines table scores under ``name``."""
    record = baselined.record.record
    if name in record.baselines:
        return record.baselines[name]
    if name == COLLECTED_JOINT_MWPM:
        return record.joint_mwpm
    return baselined.results[name].final


def test_the_summary_scores_the_baselines_and_every_cell_on_the_same_shots(baselined):
    section, record = baselined.section, baselined.record.record
    baselines = section['baselines']
    cells = [parse_config(text).name for text in CONFIGS]
    assert baselines['comparator'] == RECORDED_JOINT_MWPM
    assert baselines['order'] == list(BASELINE_DECODERS) + [COLLECTED_JOINT_MWPM] + cells
    assert sorted(baselines['decoders']) == sorted(baselines['order'])   # the JSON sorts its keys
    kinds = {name: HISTORICAL_KIND for name in BASELINE_DECODERS}
    kinds[COLLECTED_JOINT_MWPM] = COLLECTED_KIND
    kinds.update({name: HIERARCHICAL_KIND for name in cells})
    comparator = record.baselines[RECORDED_JOINT_MWPM]
    for name, entry in baselines['decoders'].items():
        predictions = predictions_of(baselined, name)
        failed = block_failures(predictions, record.actual)
        expected = bootstrap_rate(failed, replicates=REPLICATES, seed=DEFAULT_SEED)
        assert entry['kind'] == kinds[name]
        assert entry['block_failure'] == expected.to_json()
        assert entry['block_failure']['count'] == int(failed.sum())
        assert entry['block_failure']['total'] == record.shots == section['record']['shots']
        assert entry['normalized_ler'] == pytest.approx(
            normalized_ler(expected.estimate, pieces=section['pieces']))
        if kinds[name] == HIERARCHICAL_KIND:
            paired = paired_block_failure(record, comparator, predictions,
                                          replicates=REPLICATES, seed=DEFAULT_SEED)
            assert entry['paired_block_failure'] == paired.to_json()
            assert entry['paired_block_failure']['difference'] == pytest.approx(
                entry['block_failure']['estimate']
                - baselines['decoders'][RECORDED_JOINT_MWPM]['block_failure']['estimate'])
        else:
            assert 'paired_block_failure' not in entry
    # A cell's block failure and interval here are the ones its endpoint table prints:
    # the same shots are drawn under the same seed on both sides of every comparison.
    for group in section['groups']:
        for name in group['order']:
            cell, block = group['cells'][name]['block_failure'], baselines['decoders'][name]['block_failure']
            assert (cell['count'], cell['total']) == (block['count'], block['total'])
            assert group['intervals'][name]['block_failure'] == [block['low'], block['high']]


def test_the_report_renders_the_baselines_table_after_the_endpoint_groups(baselined):
    baselines = baselined.section['baselines']
    lines = baselined.markdown.splitlines()
    heading = lines.index(f'### {BASELINES_HEADING}')
    assert heading > max(index for index, line in enumerate(lines) if line.startswith('### uf:'))
    after = lines[heading:]
    for name, entry in baselines['decoders'].items():
        block = entry['block_failure']
        assert (f'| `{name}` | {entry["kind"]} | {_number(block["estimate"])} '
                f'[{block["count"]} / {block["total"]}] | {_interval([block["low"], block["high"]])} '
                f'| {_number(entry["normalized_ler"])} |') in after
        if 'paired_block_failure' in entry:
            paired = entry['paired_block_failure']
            assert (f'| `{name}` | {_number(paired["difference"])} | '
                    f'{_interval([paired["low"], paired["high"]])} |') in after
    assert f'against `{RECORDED_JOINT_MWPM}`' in baselined.markdown


def test_baselines_leave_the_summary_manifest_inputs_unchanged(baselined, pipeline, tmp_path):
    stage_summarize([pipeline.replay_dir], tmp_path / 'summary.md', replicates=REPLICATES,
                    seed=DEFAULT_SEED)
    plain = read_json(tmp_path / 'summary.manifest.json')['inputs']
    with_baselines = read_json(baselined.root / 'summary.manifest.json')['inputs']
    assert [sorted(entry) for entry in with_baselines] == [sorted(entry) for entry in plain]
    # The record hash the manifest names is the imported record's, whose container holds
    # the baseline columns, so the inputs already cover them.
    assert with_baselines[0]['record_sha256'] == baselined.record.manifest['artifacts'][RECORD_FILE]


def test_a_record_without_baselines_renders_no_baselines_table(pipeline, tmp_path):
    markdown = stage_summarize([pipeline.replay_dir], tmp_path / 'summary.md',
                               replicates=REPLICATES, seed=DEFAULT_SEED)
    assert BASELINES_HEADING not in markdown
    section, = read_json(tmp_path / 'summary.json')['records']
    assert 'baselines' not in section


def test_cells_without_the_recorded_joint_mwpm_have_no_paired_difference(baselined, pipeline,
                                                                          tmp_path):
    record_dir = copied(baselined.before_dir, tmp_path / 'evaluation')
    stage_import_baselines(record_dir, baselined.run, names=('joint_uf',))
    stage_replay(record_dir, pipeline.calibrators, tmp_path / 'replay', CONFIGS)
    markdown = stage_summarize([tmp_path / 'replay'], tmp_path / 'summary.md',
                               replicates=REPLICATES, seed=DEFAULT_SEED)
    section, = read_json(tmp_path / 'summary.json')['records']
    baselines = section['baselines']
    assert baselines['comparator'] is None
    cells = [parse_config(text).name for text in CONFIGS]
    assert baselines['order'] == ['joint_uf', COLLECTED_JOINT_MWPM] + cells
    for name in cells:
        assert baselines['decoders'][name]['paired_block_failure'] is None
        assert f'| `{name}` | {UNAVAILABLE} | {UNAVAILABLE} |' in markdown
    assert baselines['decoders']['joint_uf']['block_failure'] == \
        baselined.section['baselines']['decoders']['joint_uf']['block_failure']


# --- how an undefined statistic is rendered ----------------------------------
#
# A pilot whose bootstrap forms every interval prints none of these, so the renderers
# are exercised directly on the inputs a degenerate population produces rather than
# through a summary that may or may not contain one.

@pytest.mark.parametrize('value', [None, float('nan')])
def test_an_undefined_number_is_rendered_as_the_unavailable_word(value):
    assert _number(value) == UNAVAILABLE


def test_a_defined_number_is_rendered_with_the_reports_digits():
    assert _number(0.3746877602) == '0.374688'
    assert _number(0) == '0'


@pytest.mark.parametrize('bounds', [None, (None, 0.5), (0.5, None), (float('nan'), 0.5),
                                    (0.5, float('nan'))])
def test_an_interval_missing_either_bound_is_rendered_as_the_unavailable_word(bounds):
    assert _interval(bounds) == UNAVAILABLE


def test_a_defined_interval_is_rendered_as_its_two_bounds():
    assert _interval((-0.351237, -0.293117)) == '(-0.351237, -0.293117)'


@pytest.mark.parametrize('value', [None, float('nan')])
def test_an_undefined_rate_still_prints_its_denominator(value):
    # An empty eligible population is the usual reason a rate is undefined, and the
    # count and total are what say so; dropping them would hide the reason.
    assert _rate({'value': value, 'count': 0, 'total': 0}) == f'{UNAVAILABLE} [0 / 0]'
    assert _rate({'value': value, 'count': 7, 'total': 12}) == f'{UNAVAILABLE} [7 / 12]'


def test_a_defined_rate_prints_its_value_and_denominator():
    assert _rate({'value': 0.0655, 'count': 131, 'total': 2000}) == '0.0655 [131 / 2000]'


def test_a_paired_row_whose_bootstrap_formed_no_interval_shows_unavailable_and_its_counts():
    """A comparison with ``low is None`` still occupies a row, with its difference, its
    eligible count and its zero-denominator replicate count beside the missing bound."""
    group = {
        'order': ['refined'],
        'labels': {'refined': 'all_refined:mixed'},
        'cells': {'refined': {'misattribution': {'pooled': {'value': None, 'count': 0,
                                                            'total': 0}}}},
        'comparisons': {'refined': {
            'misattribution_pooled': {'difference': None, 'low': None, 'high': None,
                                      'zero_denominator_replicates': REPLICATES},
            'block_failure': {'difference': -0.268, 'low': -0.2885, 'high': -0.2475,
                              'zero_denominator_replicates': 0},
        }},
    }
    rows = _paired_rows(group, shots=2000)
    assert rows == [
        ['all_refined:mixed', 'misattribution_pooled', UNAVAILABLE, UNAVAILABLE, '0',
         str(REPLICATES)],
        ['all_refined:mixed', 'block_failure', '-0.268', '(-0.2885, -0.2475)', '2000', '0'],
    ]


def test_a_baseline_row_whose_statistics_are_undefined_shows_unavailable_and_its_counts():
    undefined = {'estimate': None, 'count': 0, 'total': 0, 'low': None, 'high': None,
                 'replicates': REPLICATES, 'seed': DEFAULT_SEED}
    defined = {'estimate': 0.125, 'count': 8, 'total': 64, 'low': 0.046875, 'high': 0.21875,
               'replicates': REPLICATES, 'seed': DEFAULT_SEED}
    baselines = {
        'comparator': None,
        'order': ['joint_uf', 'cell'],
        'decoders': {
            'joint_uf': {'kind': HISTORICAL_KIND, 'block_failure': undefined, 'normalized_ler': None},
            'cell': {'kind': HIERARCHICAL_KIND, 'block_failure': defined, 'normalized_ler': 0.00123,
                     'paired_block_failure': None},
        },
    }
    assert _baseline_rows(baselines) == [
        ['`joint_uf`', HISTORICAL_KIND, f'{UNAVAILABLE} [0 / 0]', UNAVAILABLE, UNAVAILABLE],
        ['`cell`', HIERARCHICAL_KIND, '0.125 [8 / 64]', '(0.046875, 0.21875)', '0.00123'],
    ]
    assert _baseline_paired_rows(baselines) == [['`cell`', UNAVAILABLE, UNAVAILABLE]]


# --- what a report change may not touch --------------------------------------

def test_a_replay_source_change_leaves_the_collection_identity_alone(pipeline, tmp_path,
                                                                     monkeypatch):
    def moved(requested):
        digests = source_hashes(requested)
        if requested is REPLAY_SOURCES:
            digests[next(iter(digests))] = 'ff' * 32
        return digests

    before = read_json(pipeline.replay_dir / REPLAY_MANIFEST)['identity']
    monkeypatch.setattr('yoked.hierarchical._collect.source_hashes', moved)
    monkeypatch.setattr('yoked.hierarchical._stages.source_hashes', moved)
    again = stage_collect(generated_request(pipeline.calibration.directory, 'calibration',
                                            CALIBRATION_SEED))
    assert again.identities == pipeline.calibration.identities
    assert sha256_file(pipeline.calibration.directory / RECORD_FILE) == \
        pipeline.calibration.manifest['artifacts'][RECORD_FILE]
    replayed = stage_replay(pipeline.evaluation.directory, pipeline.calibrators,
                            tmp_path / 'after-change', CONFIGS)
    assert sorted(replayed) == sorted(
        read_json(tmp_path / 'after-change' / REPLAY_MANIFEST)['configurations'])
    assert read_json(tmp_path / 'after-change' / REPLAY_MANIFEST)['identity'] != before


def test_a_calibration_application_change_invalidates_replay_but_keeps_l1(pipeline, tmp_path,
                                                                        monkeypatch):
    calibration_source = 'src/yoked/hierarchical/_calibration.py'

    def moved(group):
        digests = source_hashes(group)
        if calibration_source in digests:
            digests[calibration_source] = 'ff' * 32
        return digests

    monkeypatch.setattr('yoked.hierarchical._collect.source_hashes', moved)
    monkeypatch.setattr('yoked.hierarchical._stages.source_hashes', moved)
    monkeypatch.setattr('yoked.hierarchical._calibration.IsotonicCalibrator.probability',
                        lambda self, scores: np.full(np.shape(scores), 0.25))
    again = stage_collect(generated_request(pipeline.calibration.directory, 'calibration',
                                            CALIBRATION_SEED))
    assert again.identities == pipeline.calibration.identities
    with pytest.raises(ValueError, match='new output directory'):
        stage_replay(pipeline.evaluation.directory, pipeline.calibrators, pipeline.replay_dir, CONFIGS)
    fresh = tmp_path / 'changed-application'
    results = stage_replay(pipeline.evaluation.directory, pipeline.calibrators, fresh, CONFIGS)
    assert read_json(fresh / REPLAY_MANIFEST)['identity'] != \
        read_json(pipeline.replay_dir / REPLAY_MANIFEST)['identity']
    # Compare actual predictions, not just the list of files included in a fingerprint.
    initial = parse_config(INITIAL_ONLY_CONFIG)
    stored_path = pipeline.replay_dir / config_directory_name(initial) / REPLAY_ARRAYS_FILE
    with np.load(stored_path, allow_pickle=False) as stored:
        assert np.any(results[initial.name].final != stored['final'])
