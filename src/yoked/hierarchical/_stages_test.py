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
results without replaying again; a confirmation request refused; and a change to
the replay sources leaving the collection identity and its reuse untouched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import stim

from yoked.hierarchical._calibration import CLIP, KNOT_CONVENTION
from yoked.hierarchical._collect import (
    DETECTORS_FILE, SAMPLE_MANIFEST, CircuitParameters, SampleSet, CIRCUIT_FILE, DEM_FILE,
    RECORDED_MANIFEST,
)
from yoked.hierarchical._metrics import DEFAULT_SEED
from yoked.hierarchical._provenance import (
    DECODER_PACKAGES, MODEL_PACKAGES, REPLAY_SOURCES, REPOSITORY_ROOT, SAMPLING_PACKAGES,
    package_versions, read_json, sha256_file, source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import RECORD_FILE, RECORD_MANIFEST, LoadedRecord
from yoked.hierarchical._stages import (
    REPLAY_ARRAYS_FILE, REPLAY_MANIFEST, REPLAY_RESULTS_FILE, SAMPLE_DIRECTORY, CollectRequest,
    _ReplayHooks, config_directory_name, load_calibrators, parse_config, stage_calibrate,
    stage_collect, stage_replay, stage_summarize,
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


def test_an_undefined_interval_is_reported_as_unavailable(pipeline, tmp_path):
    """Every interval a summary cannot compute is printed, not omitted."""
    markdown = stage_summarize([pipeline.replay_dir], tmp_path / 'summary.md',
                               replicates=REPLICATES, seed=DEFAULT_SEED)
    summary = read_json(tmp_path / 'summary.json')
    group = summary['records'][0]['groups'][0]
    comparison = next(iter(group['comparisons'].values()))
    pooled = comparison['misattribution_pooled']
    if pooled['low'] is None:
        assert 'unavailable' in markdown
    assert group['cells'][group['baseline']]['misattribution']['pooled']['total'] is not None


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
