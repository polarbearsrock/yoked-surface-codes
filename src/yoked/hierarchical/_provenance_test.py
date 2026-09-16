"""Tests for artifact identities and the small I/O helpers: the historical packed
payload hash convention, canonical JSON, atomic JSON writes, required source files,
and which inputs each identity is sensitive and insensitive to."""
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import stat
import subprocess

import numpy as np
import pytest

from yoked.hierarchical import _provenance
from yoked.hierarchical._provenance import (
    AUDIT_SOURCES, CALIBRATION_PACKAGES, CALIBRATION_SOURCES, CHECK_SOURCES,
    DECODER_PACKAGES, DECODER_SOURCES, RECORD_CONVENTIONS, REPLAY_PACKAGES, REPLAY_SOURCES,
    REPOSITORY_ROOT, SAMPLE_CONVENTIONS, atomic_replacement, calibration_identity,
    canonical_json, check_identity, collection_identity, decoder_identity, git_commit,
    json_ready, model_identity, package_versions, packed_sample_hash, parent_sample_identity,
    read_json, replay_identity, row_ids_sha256, sampling_family_identity, sha256_bytes,
    sha256_file, source_hashes, utc_now, write_json_atomic,
)

MODEL_INPUTS = dict(
    parameters={'distance': 9, 'rounds': 36, 'p': 0.003, 'patches': 6},
    circuit_sha256='aa' * 32, dem_sha256='bb' * 32,
    num_detectors=17762, num_observables=12,
    conventions=dict(SAMPLE_CONVENTIONS), versions={'stim': '1.16.0'},
)
DECODER_INPUTS = dict(
    sources={'src/yoked/decoders/_graph.py': 'cc' * 32},
    conventions=dict(RECORD_CONVENTIONS),
    versions={'stim': '1.16.0', 'numpy': '2.5.1', 'pymatching': '2.4.0', 'scipy': '1.18.0'},
)


def replaced(inputs: dict, **changes) -> dict:
    """A copy of one identity's inputs with named fields changed."""
    return {**inputs, **changes}


# --- payload and file hashing ------------------------------------------------

def hand_packed_example():
    """Two shots of nine detector bits and twelve observable bits, packed little-endian."""
    detectors = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1],
                          [0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    actual = np.array([[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    return (np.packbits(detectors, axis=1, bitorder='little'),
            np.packbits(actual, axis=1, bitorder='little'))


def test_packed_sample_hash_reproduces_the_recorded_convention():
    detectors, actual = hand_packed_example()
    # The recorded runs hash the packed detector bytes, then the packed observable
    # bytes, with no .npy headers between them.
    expected = hashlib.sha256(detectors.tobytes() + actual.tobytes()).hexdigest()
    assert packed_sample_hash(detectors, actual) == expected


def test_packed_sample_hash_differs_from_hashing_the_npy_files(tmp_path):
    detectors, actual = hand_packed_example()
    np.save(tmp_path / 'detectors.npy', detectors)
    assert packed_sample_hash(detectors, actual) != sha256_file(tmp_path / 'detectors.npy')


def test_packed_sample_hash_depends_on_the_order_of_the_two_payloads():
    detectors, actual = hand_packed_example()
    assert packed_sample_hash(detectors, actual) != packed_sample_hash(actual, detectors)


def test_packed_sample_hash_ignores_memory_layout():
    detectors, actual = hand_packed_example()
    assert packed_sample_hash(np.asfortranarray(detectors), actual) == packed_sample_hash(detectors, actual)


def test_packed_sample_hash_streams_in_chunks_without_changing_the_result(monkeypatch):
    detectors = np.arange(64, dtype=np.uint8).reshape(16, 4)
    actual = np.arange(32, dtype=np.uint8).reshape(16, 2)
    whole = packed_sample_hash(detectors, actual)
    monkeypatch.setattr(_provenance, '_HASH_CHUNK_ROWS', 3)
    assert packed_sample_hash(detectors, actual) == whole


def test_packed_sample_hash_rejects_unpacked_bits():
    detectors, actual = hand_packed_example()
    with pytest.raises(ValueError):
        packed_sample_hash(detectors.astype(np.int64), actual)


def test_sha256_bytes_and_file_agree(tmp_path):
    path = tmp_path / 'payload.bin'
    path.write_bytes(b'hierarchical')
    assert sha256_file(path) == sha256_bytes(b'hierarchical')


def test_row_ids_hash_depends_on_the_exact_ordered_ids():
    assert row_ids_sha256(np.array([0, 1, 5])) == row_ids_sha256([0, 1, 5])
    assert row_ids_sha256([0, 1, 5]) != row_ids_sha256([0, 1, 6])
    assert row_ids_sha256([0, 1, 5]) != row_ids_sha256([5, 1, 0])


def test_row_ids_hash_rejects_non_integer_ids():
    with pytest.raises(ValueError):
        row_ids_sha256(np.array([0.0, 1.0]))


# --- JSON --------------------------------------------------------------------

def test_canonical_json_sorts_keys_and_omits_whitespace():
    assert canonical_json({'b': 1, 'a': [1, 2]}) == '{"a":[1,2],"b":1}'


def test_canonical_json_rejects_undefined_numbers():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            canonical_json({'rate': value})


def test_json_ready_replaces_undefined_numbers_with_null_recursively():
    ready = json_ready({'rate': math.nan, 'interval': [math.inf, 0.5], 'count': np.int64(3)})
    assert ready == {'rate': None, 'interval': [None, 0.5], 'count': 3}
    assert canonical_json(ready) == '{"count":3,"interval":[null,0.5],"rate":null}'


def test_json_ready_converts_numpy_scalars():
    ready = json_ready({'flag': np.True_, 'weight': np.float64(1.5), 'shots': np.int64(7)})
    assert ready == {'flag': True, 'weight': 1.5, 'shots': 7}
    assert [type(value) for value in ready.values()] == [bool, float, int]


def test_write_json_atomic_round_trips_and_leaves_no_temporary_files(tmp_path):
    path = tmp_path / 'manifest.json'
    write_json_atomic(path, {'status': 'complete', 'rate': math.nan})
    assert read_json(path) == {'status': 'complete', 'rate': None}
    assert [entry.name for entry in tmp_path.iterdir()] == ['manifest.json']


def test_write_json_atomic_replaces_an_existing_file_whole(tmp_path):
    path = tmp_path / 'manifest.json'
    write_json_atomic(path, {'status': 'partial'})
    write_json_atomic(path, {'status': 'complete'})
    assert read_json(path) == {'status': 'complete'}


def test_read_json_rejects_nonstandard_nan_literals(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text('{"rate": NaN}')
    with pytest.raises(ValueError):
        read_json(path)


def default_file_mode(directory) -> int:
    """The permission bits a plain ``open(path, 'w')`` produces under the current umask."""
    reference = directory / 'reference-mode'
    with open(reference, 'w'):
        pass
    mode = stat.S_IMODE(reference.stat().st_mode)
    reference.unlink()
    return mode


def test_a_written_artifact_has_the_permissions_an_ordinary_write_would_give_it(tmp_path):
    path = tmp_path / 'manifest.json'
    write_json_atomic(path, {'status': 'complete'})
    assert stat.S_IMODE(path.stat().st_mode) == default_file_mode(tmp_path)


def test_rewriting_an_artifact_does_not_reduce_its_permissions(tmp_path):
    path = tmp_path / 'manifest.json'
    write_json_atomic(path, {'status': 'partial'})
    os.chmod(path, 0o664)
    write_json_atomic(path, {'status': 'complete'})
    expected = default_file_mode(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) & expected == expected


def test_atomic_replacement_removes_its_temporary_file_when_the_caller_fails(tmp_path):
    path = tmp_path / 'record.npz'
    with pytest.raises(RuntimeError):
        with atomic_replacement(path) as temporary:
            temporary.write_bytes(b'partial')
            raise RuntimeError('interrupted')
    assert list(tmp_path.iterdir()) == []


# --- environment and sources -------------------------------------------------

def test_package_versions_reports_installed_versions():
    versions = package_versions(('numpy', 'stim'))
    assert versions['numpy'] == np.__version__
    assert set(versions) == {'numpy', 'stim'}


def test_package_versions_raises_for_an_unknown_package():
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        package_versions(('not-an-installed-package',))


def test_git_commit_matches_the_repository_head():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert git_commit() == head


def test_utc_now_is_a_zulu_timestamp():
    stamp = utc_now()
    assert len(stamp) == 20 and stamp.endswith('Z') and stamp[10] == 'T'


def test_source_hashes_match_the_file_contents():
    hashes = source_hashes(DECODER_SOURCES)
    assert set(hashes) == set(DECODER_SOURCES)
    for relative, digest in hashes.items():
        assert digest == sha256_file(REPOSITORY_ROOT / relative)


def test_source_hashes_raise_instead_of_omitting_a_missing_required_file():
    fake_group = ('src/yoked/hierarchical/_arrays.py', 'src/yoked/hierarchical/_absent.py')
    with pytest.raises(FileNotFoundError):
        source_hashes(fake_group)


def test_decoder_group_names_the_per_row_l1_path_and_excludes_the_rest():
    # _l1.py decides what every stored number is; _collect.py only samples, gates, and
    # schedules, so it is validation, not decoding, and lives in the check group.
    assert 'src/yoked/hierarchical/_l1.py' in DECODER_SOURCES
    assert 'src/yoked/hierarchical/_collect.py' not in DECODER_SOURCES
    excluded = ('_policies.py', '_metrics.py', '_replay.py', '_calibration.py', '_outer_decoder.py')
    assert not [name for name in DECODER_SOURCES if name.endswith(excluded)]


def test_check_group_names_the_validation_sources():
    assert 'src/yoked/hierarchical/_outer_decoder.py' in CHECK_SOURCES
    assert 'src/yoked/hierarchical/_collect.py' in CHECK_SOURCES
    assert 'src/yoked/hierarchical/_l1.py' not in CHECK_SOURCES


def test_calibration_and_replay_groups_name_their_own_sources():
    assert 'src/yoked/hierarchical/_calibration.py' in CALIBRATION_SOURCES
    assert 'src/yoked/hierarchical/_calibration.py' in REPLAY_SOURCES
    assert 'src/yoked/hierarchical/_policies.py' in REPLAY_SOURCES
    assert 'src/yoked/hierarchical/_metrics.py' in REPLAY_SOURCES


def test_the_mwpm_outer_backend_belongs_only_to_replay_provenance():
    backend = 'src/yoked/hierarchical/_outer_mwpm.py'
    assert backend in REPLAY_SOURCES
    assert backend not in DECODER_SOURCES
    assert backend not in CHECK_SOURCES
    assert backend not in CALIBRATION_SOURCES
    assert 'pymatching' in REPLAY_PACKAGES
    assert 'pymatching' in DECODER_PACKAGES
    assert 'pymatching' not in CALIBRATION_PACKAGES


def test_audit_group_covers_the_cli_and_every_circuit_generator_source():
    assert 'tools/hierarchical_experiment' in AUDIT_SOURCES
    assert 'src/yoked/_yoked_memory_circuits.py' in AUDIT_SOURCES
    generated = sorted(str(path.relative_to(REPOSITORY_ROOT)) for path in (REPOSITORY_ROOT / 'src/gen').glob('*.py'))
    assert generated and set(generated) <= set(AUDIT_SOURCES)


# --- identity sensitivity ----------------------------------------------------

def test_identities_are_hex_sha256_digests():
    identity = model_identity(**MODEL_INPUTS)
    assert len(identity) == 64 and set(identity) <= set('0123456789abcdef')


def test_model_identity_is_insensitive_to_input_ordering():
    reordered = dict(reversed(list(MODEL_INPUTS['parameters'].items())))
    assert model_identity(**replaced(MODEL_INPUTS, parameters=reordered)) == model_identity(**MODEL_INPUTS)


def test_a_changed_dem_with_unchanged_parameters_changes_the_model_identity():
    changed = model_identity(**replaced(MODEL_INPUTS, dem_sha256='cc' * 32))
    assert changed != model_identity(**MODEL_INPUTS)


def test_changed_sample_conventions_change_the_model_identity():
    conventions = {**SAMPLE_CONVENTIONS, 'bit_order': 'big'}
    assert model_identity(**replaced(MODEL_INPUTS, conventions=conventions)) != model_identity(**MODEL_INPUTS)


def test_parent_sample_identity_does_not_depend_on_selected_rows():
    parameters = set(inspect.signature(parent_sample_identity).parameters)
    assert parameters == {'model', 'seed', 'parent_shots', 'payload_sha256', 'versions'}


def test_two_sampling_calls_with_one_seed_share_a_family_but_not_a_parent_identity():
    model = model_identity(**MODEL_INPUTS)
    versions = {'stim': '1.16.0'}
    family = dict(circuit_sha256=MODEL_INPUTS['circuit_sha256'], seed=42, versions=versions)
    small = parent_sample_identity(model=model, seed=42, parent_shots=2000,
                                   payload_sha256='dd' * 32, versions=versions)
    large = parent_sample_identity(model=model, seed=42, parent_shots=100000,
                                   payload_sha256='ee' * 32, versions=versions)
    assert sampling_family_identity(**family) == sampling_family_identity(**family)
    assert small != large


def test_a_changed_seed_changes_the_sampling_family():
    versions = {'stim': '1.16.0'}
    first = sampling_family_identity(circuit_sha256='aa' * 32, seed=42, versions=versions)
    second = sampling_family_identity(circuit_sha256='aa' * 32, seed=142, versions=versions)
    assert first != second


def test_a_changed_decoder_source_changes_the_decoder_identity():
    changed = decoder_identity(**replaced(DECODER_INPUTS, sources={'src/yoked/decoders/_graph.py': 'ff' * 32}))
    assert changed != decoder_identity(**DECODER_INPUTS)


def test_a_changed_package_version_changes_the_decoder_identity():
    versions = {**DECODER_INPUTS['versions'], 'pymatching': '2.5.0'}
    assert decoder_identity(**replaced(DECODER_INPUTS, versions=versions)) != decoder_identity(**DECODER_INPUTS)


def collection(**changes):
    inputs = dict(parent_sample='11' * 32, decoder=decoder_identity(**DECODER_INPUTS),
                  role='evaluation', shots=3, rows_sha256=row_ids_sha256([0, 1, 2]))
    return collection_identity(**{**inputs, **changes})


def test_a_policy_or_report_change_leaves_the_collection_identity_unchanged():
    # A policy edit changes the replay sources only; neither the decoder identity nor the
    # collection identity reads them, so reusable L1 outputs stay valid.
    baseline_replay = dict(
        record_sha256='22' * 32, calibrators_sha256='33' * 32,
        configurations=[{'policy': 'all_refined'}], tie_rule={'tolerance': 1e-9},
        work_convention='endpoint counts', sources={'src/yoked/hierarchical/_policies.py': 'aa' * 32},
        versions={'numpy': '2.5.1', 'sinter': '1.16.0'})
    edited = {**baseline_replay, 'sources': {'src/yoked/hierarchical/_policies.py': 'bb' * 32}}
    assert replay_identity(**edited) != replay_identity(**baseline_replay)
    # The collection identity reads none of those sources, and does read the decoder's,
    # so it moves only when the decoder does.
    moved = decoder_identity(**replaced(DECODER_INPUTS,
                                        sources={'src/yoked/hierarchical/_l1.py': 'ff' * 32}))
    assert collection(decoder=moved) != collection()


def test_a_changed_decoder_changes_the_collection_identity():
    other = decoder_identity(**replaced(DECODER_INPUTS, sources={'src/yoked/decoders/_graph.py': 'ff' * 32}))
    assert collection(decoder=other) != collection()


def test_changed_rows_or_role_change_the_collection_identity():
    assert collection(rows_sha256=row_ids_sha256([0, 1, 3])) != collection()
    assert collection(role='calibration') != collection()
    assert collection(parent_sample='99' * 32) != collection()


def test_collection_identity_does_not_depend_on_a_filesystem_location():
    parameters = set(inspect.signature(collection_identity).parameters)
    assert parameters == {'parent_sample', 'decoder', 'role', 'shots', 'rows_sha256'}


def test_identity_kinds_do_not_collide():
    sources = {'src/yoked/hierarchical/_outer_decoder.py': 'aa' * 32}
    versions = {'numpy': '2.5.1'}
    assert check_identity(sources=sources, versions=versions) != _provenance._identity(
        'decoder', {'sources': sources, 'versions': versions})


def test_every_identity_includes_the_schema_version(monkeypatch):
    before = [model_identity(**MODEL_INPUTS), decoder_identity(**DECODER_INPUTS), collection()]
    monkeypatch.setattr(_provenance, 'SCHEMA_VERSION', 'hierarchical-l1-l2/changed')
    after = [model_identity(**MODEL_INPUTS), decoder_identity(**DECODER_INPUTS), collection()]
    assert all(old != new for old, new in zip(before, after))


def calibration(**changes):
    inputs = dict(record_sha256='44' * 32, rows_sha256=row_ids_sha256([0, 1, 2]),
                  model=model_identity(**MODEL_INPUTS), decoder=decoder_identity(**DECODER_INPUTS),
                  estimators=[{'reference': 'uf', 'score': 'cluster_gap', 'direction': 'decreasing'}],
                  knot_convention='pav block centers, linear between, constant outside',
                  sources={'src/yoked/hierarchical/_calibration.py': 'aa' * 32},
                  versions={'numpy': '2.5.1'})
    return calibration_identity(**{**inputs, **changes})


def test_calibration_identity_tracks_its_record_estimators_and_knot_convention():
    assert calibration(record_sha256='55' * 32) != calibration()
    assert calibration(estimators=[{'reference': 'mwpm', 'score': 'gap_plain'}]) != calibration()
    assert calibration(knot_convention='step function') != calibration()
    assert calibration() == calibration()


def replay(**changes):
    inputs = dict(record_sha256='22' * 32, calibrators_sha256='33' * 32,
                  configurations=[{'policy': 'initial_only'}], tie_rule={'tolerance': 1e-9},
                  work_convention='endpoint counts',
                  sources={'src/yoked/hierarchical/_replay.py': 'aa' * 32},
                  versions={'numpy': '2.5.1', 'sinter': '1.16.0'})
    return replay_identity(**{**inputs, **changes})


def test_replay_identity_tracks_its_artifacts_configurations_and_tie_rule():
    assert replay(calibrators_sha256='66' * 32) != replay()
    assert replay(configurations=[{'policy': 'all_refined'}]) != replay()
    assert replay(tie_rule={'tolerance': 1e-6}) != replay()
    assert replay(work_convention='other') != replay()
    assert replay() == replay()


def test_identity_inputs_reject_undefined_numbers():
    with pytest.raises(ValueError):
        model_identity(**replaced(MODEL_INPUTS, parameters={'p': math.nan}))


def test_identity_accepts_immutable_mapping_inputs():
    assert model_identity(**replaced(MODEL_INPUTS, conventions=SAMPLE_CONVENTIONS)) == model_identity(**MODEL_INPUTS)


def test_json_helpers_and_identities_agree_on_the_canonical_form():
    payload = {'b': 1, 'a': 2}
    assert json.loads(canonical_json(payload)) == payload
