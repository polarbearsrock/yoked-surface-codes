"""Tests for the stored L1 record: ownership of every array and baseline, the value
checks that run before any dtype cast, the sector/column conversions, subsets, the
plain-array boundary crossing, the schema-checked array container, and the verified
loader that is the only way downstream stages see a completed collection.

The loader tests build a completed directory by hand rather than collecting one: what
``load_record`` promises is about the manifest, the artifact hashes, the row summary,
and the recomputed collection identity, none of which needs a decoder.
"""
import dataclasses
import stat
import types
from pathlib import Path

import numpy as np
import pytest

from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import (
    SCHEMA_VERSION, collection_identity, read_json, row_ids_sha256, sha256_file, write_json_atomic,
)
from yoked.hierarchical._record import (
    ARRAY_FIELDS, BASELINE_PREFIX, COMPLETE_STATUS, IDENTITY_NAMES, L1Record, LoadedRecord,
    RECORD_FILE, RECORD_MANIFEST, RECORD_SCHEMA, ROW_SUMMARY_FIELDS, SCHEMA_KEY, _load_arrays,
    _save_arrays, by_sector, load_record, row_summary, to_columns,
)

SHOTS, PATCHES = 3, 2
COLUMNS = NUM_SECTORS * PATCHES


def make_arrays(**overrides) -> dict:
    """Valid raw inputs for a three-shot, two-patch record."""
    arrays = dict(
        actual=np.array([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
        yoke=np.array([[1, 0], [0, 0], [0, 1]], dtype=bool),
        uf_reference=np.array([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
        mwpm_reference=np.array([[1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=bool),
        correlated_prediction=np.array([[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
        joint_mwpm=np.array([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
        cluster_gap=np.array([[0.5, 1.25, 2.0, 0.0], [3.0, 0.25, 1.0, 2.5],
                              [0.75, 0.5, 1.5, 2.25]], dtype=np.float64),
        dijkstra_states=np.array([[10, 12, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]], dtype=np.int64),
        forced_plain=np.arange(SHOTS * PATCHES * 4, dtype=np.float64).reshape(SHOTS, PATCHES, 2, 2) / 4,
        forced_correlated=np.arange(SHOTS * PATCHES * 4, dtype=np.float64).reshape(SHOTS, PATCHES, 2, 2) / 8,
        reweighted_patches=np.array([[1, 0], [0, 0], [1, 1]], dtype=bool),
        rows=np.array([0, 4, 9], dtype=np.int64),
    )
    arrays.update(overrides)
    return arrays


def make_record(**overrides) -> L1Record:
    baselines = overrides.pop('baselines', {})
    return L1Record(**make_arrays(**overrides), baselines=baselines)


def historical_baseline() -> np.ndarray:
    return np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)


# --- ownership ---------------------------------------------------------------

def test_the_record_lists_every_stored_array_field():
    assert ARRAY_FIELDS == (
        'actual', 'yoke', 'uf_reference', 'mwpm_reference', 'correlated_prediction', 'joint_mwpm',
        'cluster_gap', 'dijkstra_states', 'forced_plain', 'forced_correlated',
        'reweighted_patches', 'rows')


def test_mutating_an_input_array_after_construction_does_not_change_the_record():
    arrays = make_arrays()
    record = L1Record(**arrays, baselines={'joint_uf': historical_baseline()})
    expected = {name: np.array(array) for name, array in record.arrays().items()}
    for array in arrays.values():
        array[...] = 0
    for name, array in record.arrays().items():
        np.testing.assert_array_equal(array, expected[name])


def test_mutating_the_baseline_source_array_does_not_change_the_record():
    baseline = historical_baseline()
    record = make_record(baselines={'joint_uf': baseline})
    baseline[...] = 0
    assert record.baselines['joint_uf'].any()


def test_writing_through_any_stored_array_raises():
    record = make_record(baselines={'joint_uf': historical_baseline()})
    for name, array in record.arrays().items():
        with pytest.raises(ValueError, match='read-only'):
            array[(0,) * array.ndim] = 0


def test_the_record_is_frozen_and_its_baselines_mapping_is_immutable():
    record = make_record(baselines={'joint_uf': historical_baseline()})
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.rows = np.array([1, 2, 3])
    assert isinstance(record.baselines, types.MappingProxyType)
    with pytest.raises(TypeError):
        record.baselines['joint_uf'] = historical_baseline()


def test_mutating_the_baselines_mapping_passed_in_does_not_change_the_record():
    baselines = {'joint_uf': historical_baseline()}
    record = make_record(baselines=baselines)
    baselines['joint_correlated_uf'] = historical_baseline()
    assert set(record.baselines) == {'joint_uf'}


def test_baselines_are_attached_with_replace_rather_than_by_mutation():
    record = make_record()
    assert dict(record.baselines) == {}
    attached = dataclasses.replace(record, baselines={'joint_uf': historical_baseline()})
    assert set(attached.baselines) == {'joint_uf'}
    assert dict(record.baselines) == {}
    np.testing.assert_array_equal(attached.rows, record.rows)


# --- shapes and value checks -------------------------------------------------

def test_shots_and_num_patches_come_from_the_record_layout():
    record = make_record()
    assert record.shots == SHOTS
    assert record.num_patches == PATCHES


def test_stored_dtypes_are_the_documented_ones():
    record = make_record()
    assert record.actual.dtype == bool and record.reweighted_patches.dtype == bool
    assert record.cluster_gap.dtype == np.float64 and record.forced_plain.dtype == np.float64
    assert record.dijkstra_states.dtype == np.int64 and record.rows.dtype == np.int64


@pytest.mark.parametrize('name', ['actual', 'uf_reference', 'mwpm_reference',
                                  'correlated_prediction', 'joint_mwpm', 'yoke', 'reweighted_patches'])
def test_fractional_values_are_rejected_instead_of_cast_to_true(name):
    array = make_arrays()[name].astype(np.float64)
    array[0, 0] = 0.5
    with pytest.raises(ValueError, match='0 and 1'):
        make_record(**{name: array})


@pytest.mark.parametrize('name', ['actual', 'uf_reference', 'joint_mwpm'])
def test_a_wrong_column_count_is_rejected(name):
    with pytest.raises(ValueError):
        make_record(**{name: np.zeros((SHOTS, COLUMNS + 1), dtype=bool)})


def test_an_odd_number_of_columns_is_rejected():
    with pytest.raises(ValueError):
        make_record(actual=np.zeros((SHOTS, COLUMNS + 1), dtype=bool))


def test_a_one_dimensional_actual_array_is_rejected():
    with pytest.raises(ValueError):
        make_record(actual=np.zeros(COLUMNS, dtype=bool))


def test_a_record_without_shots_is_rejected():
    with pytest.raises(ValueError):
        make_record(actual=np.zeros((0, COLUMNS), dtype=bool))


def test_a_mismatched_shot_count_between_fields_is_rejected():
    with pytest.raises(ValueError):
        make_record(yoke=np.zeros((SHOTS + 1, NUM_SECTORS), dtype=bool))


def test_a_wrong_yoke_width_is_rejected():
    with pytest.raises(ValueError):
        make_record(yoke=np.zeros((SHOTS, 3), dtype=bool))


@pytest.mark.parametrize('bad', [-1e-9, np.nan, np.inf])
def test_a_cluster_gap_that_is_negative_or_undefined_is_rejected(bad):
    gaps = make_arrays()['cluster_gap'].copy()
    gaps[1, 2] = bad
    with pytest.raises(ValueError):
        make_record(cluster_gap=gaps)


@pytest.mark.parametrize('name', ['forced_plain', 'forced_correlated'])
@pytest.mark.parametrize('bad', [-1.0, np.nan, -np.inf])
def test_a_forced_weight_that_is_negative_or_undefined_is_rejected(name, bad):
    weights = make_arrays()[name].copy()
    weights[2, 1, 0, 1] = bad
    with pytest.raises(ValueError):
        make_record(**{name: weights})


@pytest.mark.parametrize('name', ['forced_plain', 'forced_correlated'])
def test_a_forced_weight_array_in_column_layout_is_rejected(name):
    with pytest.raises(ValueError):
        make_record(**{name: np.zeros((SHOTS, COLUMNS), dtype=np.float64)})


def test_negative_state_counts_are_rejected():
    states = make_arrays()['dijkstra_states'].copy()
    states[0, 0] = -1
    with pytest.raises(ValueError):
        make_record(dijkstra_states=states)


def test_fractional_state_counts_are_rejected_instead_of_truncated():
    states = make_arrays()['dijkstra_states'].astype(np.float64)
    states[0, 0] = 2.5
    with pytest.raises(ValueError):
        make_record(dijkstra_states=states)


def test_boolean_state_counts_are_rejected():
    with pytest.raises(ValueError):
        make_record(dijkstra_states=np.ones((SHOTS, COLUMNS), dtype=bool))


@pytest.mark.parametrize('rows', [
    [0, 4, 4],          # duplicate parent rows
    [9, 4, 0],          # stored out of order
    [0, -1, 9],         # negative index
    [0.0, 4.5, 9.0],    # fractional index
])
def test_invalid_row_ids_are_rejected(rows):
    with pytest.raises(ValueError):
        make_record(rows=np.array(rows))


def test_boolean_row_ids_are_rejected():
    with pytest.raises(ValueError):
        make_record(rows=np.array([True, False, True]))


def test_two_dimensional_row_ids_are_rejected():
    with pytest.raises(ValueError):
        make_record(rows=np.zeros((SHOTS, 1), dtype=np.int64))


def test_row_ids_keep_the_order_they_were_given():
    record = make_record(rows=np.array([2, 3, 11], dtype=np.int64))
    np.testing.assert_array_equal(record.rows, [2, 3, 11])


def test_a_baseline_with_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        make_record(baselines={'joint_uf': np.zeros((SHOTS, COLUMNS + 2), dtype=bool)})


def test_a_nonbinary_baseline_is_rejected():
    baseline = historical_baseline().astype(np.float64)
    baseline[0, 0] = 0.5
    with pytest.raises(ValueError):
        make_record(baselines={'joint_uf': baseline})


def test_an_unnamed_baseline_is_rejected():
    with pytest.raises(ValueError):
        make_record(baselines={'': historical_baseline()})


# --- accessors ---------------------------------------------------------------

def test_reference_selects_the_named_reference_decoder():
    record = make_record()
    np.testing.assert_array_equal(record.reference('uf'), record.uf_reference)
    np.testing.assert_array_equal(record.reference('mwpm'), record.mwpm_reference)


def test_reference_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        make_record().reference('joint_mwpm')


def test_by_sector_maps_column_two_i_plus_s_to_sector_s_patch_i():
    columns = np.array([[10, 11, 20, 21]])
    np.testing.assert_array_equal(by_sector(columns), [[[10, 20], [11, 21]]])


def test_to_columns_inverts_by_sector_for_any_leading_shape():
    record = make_record()
    np.testing.assert_array_equal(to_columns(by_sector(record.cluster_gap)), record.cluster_gap)
    stacked = np.stack([record.cluster_gap, record.cluster_gap])
    assert by_sector(stacked).shape == (2, SHOTS, NUM_SECTORS, PATCHES)
    np.testing.assert_array_equal(to_columns(by_sector(stacked)), stacked)


def test_the_sector_views_are_owned_arrays_that_do_not_alias_the_record():
    record = make_record()
    sectors = by_sector(record.cluster_gap)
    sectors[0, 0, 0] = 99.0
    assert record.cluster_gap[0, 0] != 99.0


def test_by_sector_rejects_an_odd_column_count():
    with pytest.raises(ValueError):
        by_sector(np.zeros((SHOTS, 5)))


def test_to_columns_rejects_a_wrong_sector_count():
    with pytest.raises(ValueError):
        to_columns(np.zeros((SHOTS, 3, PATCHES)))


# --- subsets -----------------------------------------------------------------

def test_subset_keeps_the_selected_rows_and_their_parent_ids():
    record = make_record(baselines={'joint_uf': historical_baseline()})
    part = record.subset([0, 2])
    assert part.shots == 2 and part.num_patches == PATCHES
    np.testing.assert_array_equal(part.rows, [0, 9])
    np.testing.assert_array_equal(part.actual, record.actual[[0, 2]])
    np.testing.assert_array_equal(part.forced_plain, record.forced_plain[[0, 2]])
    np.testing.assert_array_equal(part.baselines['joint_uf'], record.baselines['joint_uf'][[0, 2]])


def test_subset_arrays_are_owned_and_read_only():
    part = make_record(baselines={'joint_uf': historical_baseline()}).subset([1, 2])
    for array in part.arrays().values():
        assert not array.flags.writeable and array.flags.owndata


@pytest.mark.parametrize('positions', [[2, 0], [0, 0], [0, SHOTS], [-1], [], [0.0, 1.0]])
def test_invalid_subset_positions_are_rejected(positions):
    with pytest.raises(ValueError):
        make_record().subset(np.array(positions))


def test_subset_rejects_a_boolean_mask():
    with pytest.raises(ValueError):
        make_record().subset(np.array([True, False, True]))


# --- crossing a process boundary ---------------------------------------------

def test_arrays_round_trip_through_plain_arrays_including_baselines():
    record = make_record(baselines={'joint_uf': historical_baseline()})
    payload = {name: np.array(array) for name, array in record.arrays().items()}
    assert BASELINE_PREFIX + 'joint_uf' in payload
    restored = L1Record.from_arrays(payload)
    for name, array in record.arrays().items():
        np.testing.assert_array_equal(restored.arrays()[name], array)
    assert set(restored.baselines) == {'joint_uf'}


def test_from_arrays_revalidates_on_the_receiving_side():
    payload = {name: np.array(array) for name, array in make_record().arrays().items()}
    payload['cluster_gap'] = payload['cluster_gap'].copy()
    payload['cluster_gap'][0, 0] = -1.0
    with pytest.raises(ValueError):
        L1Record.from_arrays(payload)


def test_from_arrays_rejects_missing_and_unknown_arrays():
    payload = {name: np.array(array) for name, array in make_record().arrays().items()}
    with pytest.raises(ValueError, match='Missing'):
        L1Record.from_arrays({k: v for k, v in payload.items() if k != 'yoke'})
    with pytest.raises(ValueError, match='Unknown'):
        L1Record.from_arrays({**payload, 'surprise': np.zeros(SHOTS)})


# --- the array container -----------------------------------------------------

def test_saved_arrays_round_trip_through_the_schema_checked_container(tmp_path):
    path = tmp_path / 'record.npz'
    record = make_record(baselines={'joint_uf': historical_baseline()})
    _save_arrays(path, record.arrays(), schema=RECORD_SCHEMA)
    loaded = _load_arrays(path, schema=RECORD_SCHEMA)
    assert set(loaded) == set(record.arrays())
    restored = L1Record.from_arrays(loaded)
    np.testing.assert_array_equal(restored.rows, record.rows)
    np.testing.assert_array_equal(restored.forced_correlated, record.forced_correlated)


def test_saving_replaces_atomically_and_leaves_no_temporary_files(tmp_path):
    path = tmp_path / 'record.npz'
    _save_arrays(path, make_record().arrays(), schema=RECORD_SCHEMA)
    _save_arrays(path, make_record(rows=np.array([1, 2, 3])).arrays(), schema=RECORD_SCHEMA)
    assert [entry.name for entry in tmp_path.iterdir()] == ['record.npz']
    np.testing.assert_array_equal(_load_arrays(path, schema=RECORD_SCHEMA)['rows'], [1, 2, 3])


def test_a_saved_container_has_the_permissions_an_ordinary_write_would_give_it(tmp_path):
    path = tmp_path / 'record.npz'
    _save_arrays(path, make_record().arrays(), schema=RECORD_SCHEMA)
    reference = tmp_path / 'reference-mode'
    with open(reference, 'w'):
        pass
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


def test_a_container_without_a_schema_entry_is_rejected(tmp_path):
    path = tmp_path / 'record.npz'
    np.savez(path, **make_record().arrays())
    with pytest.raises(ValueError, match='schema'):
        _load_arrays(path, schema=RECORD_SCHEMA)


def test_a_container_with_another_schema_is_rejected(tmp_path):
    path = tmp_path / 'record.npz'
    _save_arrays(path, make_record().arrays(), schema='L1Record/other')
    with pytest.raises(ValueError, match='schema'):
        _load_arrays(path, schema=RECORD_SCHEMA)


def test_a_truncated_container_is_rejected(tmp_path):
    path = tmp_path / 'record.npz'
    _save_arrays(path, make_record().arrays(), schema=RECORD_SCHEMA)
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])
    with pytest.raises(ValueError):
        _load_arrays(path, schema=RECORD_SCHEMA)


def test_a_missing_container_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_arrays(tmp_path / 'absent.npz', schema=RECORD_SCHEMA)


def test_saving_rejects_an_array_named_like_the_schema_entry(tmp_path):
    arrays = {**make_record().arrays(), SCHEMA_KEY: np.zeros(1)}
    with pytest.raises(ValueError):
        _save_arrays(tmp_path / 'record.npz', arrays, schema=RECORD_SCHEMA)


def test_loading_rejects_a_pickled_object_array(tmp_path):
    path = tmp_path / 'record.npz'
    np.savez(path, **{SCHEMA_KEY: np.array(RECORD_SCHEMA)},
             payload=np.array([{'not': 'an array'}], dtype=object))
    with pytest.raises(ValueError):
        _load_arrays(path, schema=RECORD_SCHEMA)


# --- the loaded record -------------------------------------------------------

def identities() -> dict:
    return {name: f'{index:02x}' * 32 for index, name in enumerate(IDENTITY_NAMES)}


def test_loaded_record_carries_its_directory_identities_and_manifest(tmp_path):
    loaded = LoadedRecord(record=make_record(), directory=str(tmp_path),
                          identities=identities(), manifest={'status': 'complete'})
    assert loaded.directory == Path(tmp_path)
    assert loaded.identities['model'] == identities()['model']
    assert loaded.manifest['status'] == 'complete'
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.record = make_record()


def test_loaded_record_identities_and_manifest_are_immutable(tmp_path):
    manifest = {'status': 'complete', 'checks': {'passed': True}, 'artifacts': ['record.npz']}
    loaded = LoadedRecord(record=make_record(), directory=tmp_path,
                          identities=identities(), manifest=manifest)
    with pytest.raises(TypeError):
        loaded.identities['model'] = 'ff' * 32
    with pytest.raises(TypeError):
        loaded.manifest['status'] = 'partial'
    with pytest.raises(TypeError):
        loaded.manifest['checks']['passed'] = False
    manifest['status'] = 'partial'
    assert loaded.manifest['status'] == 'complete'
    assert loaded.manifest['artifacts'] == ('record.npz',)


def test_loaded_record_requires_every_named_identity(tmp_path):
    incomplete = {name: value for name, value in identities().items() if name != 'decoder'}
    with pytest.raises(ValueError, match='decoder'):
        LoadedRecord(record=make_record(), directory=tmp_path,
                     identities=incomplete, manifest={'status': 'complete'})


def test_loaded_record_rejects_a_nonstring_identity(tmp_path):
    with pytest.raises(ValueError):
        LoadedRecord(record=make_record(), directory=tmp_path,
                     identities={**identities(), 'model': 0}, manifest={})


def test_loaded_record_rejects_something_that_is_not_a_record(tmp_path):
    with pytest.raises(TypeError):
        LoadedRecord(record=make_arrays(), directory=tmp_path,
                     identities=identities(), manifest={})


# --- the verified loader -----------------------------------------------------

def published(directory, record=None, **overrides) -> Path:
    """A hand-built completed collection directory, holding what ``load_record`` verifies.

    Nothing here decodes: the loader's contract is about the manifest, the artifact
    hashes, and the stored arrays, so the tests build those directly.
    """
    record = make_record() if record is None else record
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _save_arrays(directory / RECORD_FILE, record.arrays(), schema=RECORD_SCHEMA)
    summary = row_summary(record.rows)
    ids = {'model': 'a1' * 32, 'parent_sample': 'b2' * 32, 'sampling_family': 'c3' * 32,
           'decoder': 'd4' * 32}
    ids['collection'] = collection_identity(
        parent_sample=ids['parent_sample'], decoder=ids['decoder'], role='evaluation',
        shots=record.shots, rows_sha256=summary['sha256'])
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'status': COMPLETE_STATUS,
        'role': 'evaluation',
        'shots': record.shots,
        'rows': summary,
        'identities': ids,
        'artifacts': {RECORD_FILE: sha256_file(directory / RECORD_FILE)},
        'checks': {'passed': True, 'identity': 'e5' * 32, 'graph': {'passed': True},
                   'record': {'passed': True}},
    }
    write_json_atomic(directory / RECORD_MANIFEST, {**manifest, **overrides})
    return directory


def republish(directory, change) -> Path:
    """Rewrite the manifest of an already published directory through ``change``."""
    manifest = read_json(Path(directory) / RECORD_MANIFEST)
    change(manifest)
    write_json_atomic(Path(directory) / RECORD_MANIFEST, manifest)
    return directory


def test_row_summary_reports_the_count_bounds_and_exact_ids():
    summary = row_summary(np.array([3, 5, 11], dtype=np.int64))
    assert summary['count'] == 3 and summary['start'] == 3 and summary['stop'] == 12
    assert summary['sha256'] == row_ids_sha256(np.array([3, 5, 11], dtype=np.int64))
    assert set(summary) == set(ROW_SUMMARY_FIELDS)


def test_row_summary_rejects_ids_that_are_not_a_nonempty_integer_array():
    with pytest.raises(ValueError):
        row_summary(np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        row_summary(np.array([1.0, 2.0]))


def test_load_record_returns_the_stored_record_and_its_identities(tmp_path):
    record = make_record()
    loaded = load_record(published(tmp_path / 'collection', record))
    assert isinstance(loaded, LoadedRecord)
    np.testing.assert_array_equal(loaded.record.rows, record.rows)
    np.testing.assert_array_equal(loaded.record.forced_plain, record.forced_plain)
    assert set(loaded.identities) == set(IDENTITY_NAMES)
    assert loaded.manifest['status'] == COMPLETE_STATUS
    assert loaded.directory == tmp_path / 'collection'


def test_load_record_rejects_a_directory_with_no_manifest(tmp_path):
    with pytest.raises(ValueError, match=RECORD_MANIFEST):
        load_record(tmp_path)


def test_load_record_rejects_an_orphan_record_without_a_manifest(tmp_path):
    directory = published(tmp_path / 'collection')
    (directory / RECORD_MANIFEST).unlink()
    assert (directory / RECORD_FILE).is_file()
    with pytest.raises(ValueError, match=RECORD_MANIFEST):
        load_record(directory)


def test_load_record_rejects_a_status_other_than_complete(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest.update(status='partial'))
    with pytest.raises(ValueError, match='status'):
        load_record(directory)


def test_load_record_rejects_checks_that_did_not_pass(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['checks'].update(passed=False))
    with pytest.raises(ValueError, match='checks'):
        load_record(directory)


def test_load_record_rejects_another_schema_version(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest.update(schema_version='hierarchical-l1-l2/0'))
    with pytest.raises(ValueError, match='schema'):
        load_record(directory)


def test_load_record_rejects_a_manifest_missing_a_required_field(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest.pop('identities'))
    with pytest.raises(ValueError, match='identities'):
        load_record(directory)


def test_load_record_rejects_a_record_whose_bytes_changed(tmp_path):
    directory = published(tmp_path / 'collection')
    (directory / RECORD_FILE).write_bytes((directory / RECORD_FILE).read_bytes() + b'\0')
    with pytest.raises(ValueError, match=RECORD_FILE):
        load_record(directory)


def test_load_record_rejects_a_corrupt_record_container(tmp_path):
    directory = published(tmp_path / 'collection')
    data = (directory / RECORD_FILE).read_bytes()[: 64]
    (directory / RECORD_FILE).write_bytes(data)
    republish(directory, lambda manifest: manifest['artifacts'].update(
        {RECORD_FILE: sha256_file(directory / RECORD_FILE)}))
    with pytest.raises(ValueError):
        load_record(directory)


def test_load_record_rejects_a_declared_artifact_that_is_absent(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['artifacts'].update({'extra.npy': 'ff' * 32}))
    with pytest.raises(ValueError, match='extra.npy'):
        load_record(directory)


def test_load_record_rejects_an_artifact_name_that_leaves_the_directory(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['artifacts'].update({'../escape': 'ff' * 32}))
    with pytest.raises(ValueError, match='escape'):
        load_record(directory)


def test_load_record_requires_the_record_itself_among_the_artifacts(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['artifacts'].pop(RECORD_FILE))
    with pytest.raises(ValueError, match=RECORD_FILE):
        load_record(directory)


def test_load_record_rejects_a_row_summary_the_record_does_not_have(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['rows'].update(count=99))
    with pytest.raises(ValueError, match='count'):
        load_record(directory)


def test_load_record_rejects_a_row_hash_the_record_does_not_have(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['rows'].update(sha256='ff' * 32))
    with pytest.raises(ValueError, match='sha256'):
        load_record(directory)


def test_load_record_rejects_a_shot_count_the_record_does_not_have(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest.update(shots=99))
    with pytest.raises(ValueError, match='shots'):
        load_record(directory)


def test_load_record_rejects_a_collection_identity_that_does_not_follow(tmp_path):
    def relabel(manifest):
        """The role is one of the collection identity's inputs, so relabelling a stored
        record cannot leave the recorded identity standing."""
        manifest['role'] = 'calibration'

    with pytest.raises(ValueError, match='collection'):
        load_record(republish(published(tmp_path / 'collection'), relabel))


def test_load_record_rejects_a_missing_identity(tmp_path):
    directory = republish(published(tmp_path / 'collection'),
                          lambda manifest: manifest['identities'].pop('decoder'))
    with pytest.raises(ValueError, match='decoder'):
        load_record(directory)
