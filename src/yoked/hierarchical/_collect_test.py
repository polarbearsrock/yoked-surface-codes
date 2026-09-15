"""Tests for verified sample sets, single-process L1 collection, and the graph and
record checks.

Checks: a generated sample's save/load round trip and the file, payload, and
dimension verification that load performs; the imported recorded-run format and its
rejection of a tampered file; graph equivalence on the distance-3 fixture together
with deliberately changed edge multiplicity, mask, and weight (a difference below
1e-9 passing and a larger one failing); collection of a few hundred fixture shots
through both correlation branches, with valid corrections, instrumented work counts,
and equality between one call and a partitioned serial collection; every record
invariant failing on its own deliberately corrupted field; and the checkpointed
coordinator, where an uninterrupted serial run, a one-chunk run, a two-worker run, a
run stopped by ``max_chunks``, and a run interrupted at each of the four hook points
all publish the same arrays, row ids, and retained-row work counts, while a changed
decoder identity, role, row set, sample payload, checkpoint, or record is rejected
instead of being collected into the same directory.
"""
import dataclasses
import shutil

import numpy as np
import pytest

from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._collect import (
    ACTUAL_FILE, CHECKPOINT_FILE, CIRCUIT_FILE, COLLECTION_FILE, DEM_FILE, DETECTORS_FILE,
    EDGE_WEIGHT_TOLERANCE, FAILED_CHECKS_FILE, FORCED_CALLS_PER_DECODE, RECORDED_MANIFEST, ROLES,
    SAMPLE_MANIFEST, WEIGHT_TOLERANCE, WORK_FIELDS, CircuitParameters, CollectionSettings,
    CollectionWork, L1Context, SampleSet, _Buffers, _CollectionHooks, _run_chunks, check_graphs,
    check_record, collect_rows, collect_sample,
)
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import (
    CHECK_SOURCES, DECODER_PACKAGES, DECODER_SOURCES, MODEL_PACKAGES, RECORD_CONVENTIONS,
    SAMPLING_PACKAGES, decoder_identity, package_versions, read_json, row_ids_sha256,
    sha256_file, source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import (
    ARRAY_FIELDS, IDENTITY_NAMES, LoadedRecord, RECORD_FILE, RECORD_MANIFEST,
)

SHOTS = 200
"""Enough distance-3 shots to exercise both correlation branches in a few seconds."""

PATCHES = 6
"""The experiment's six patches, as the fixture circuit builds them."""


@pytest.fixture(scope='module')
def parameters() -> CircuitParameters:
    return CircuitParameters(distance=3, rounds=12, p=0.003)


@pytest.fixture(scope='module')
def sample(parameters) -> SampleSet:
    return SampleSet.sample(parameters, seed=11, shots=SHOTS)


@pytest.fixture(scope='module')
def saved(sample, tmp_path_factory):
    """One saved copy of the sample; tamper tests copy it rather than editing it."""
    directory = tmp_path_factory.mktemp('sample') / 'evaluation'
    sample.save(directory)
    return directory


@pytest.fixture(scope='module')
def context(sample) -> L1Context:
    return L1Context.from_dem_text(sample.dem_text, num_patches=PATCHES)


@pytest.fixture(scope='module')
def collected(sample, context):
    rows = np.arange(SHOTS)
    detectors, actual = sample.rows(rows)
    return collect_rows(context, detectors, actual, rows)


def copied(source, destination):
    """A private copy of a saved directory, so a tamper test cannot disturb the shared one."""
    shutil.copytree(source, destination)
    return destination


def write_recorded_run(saved, directory, sample):
    """The recorded four-decoder run layout, rebuilt from a saved sample."""
    copied(saved, directory)
    (directory / SAMPLE_MANIFEST).unlink()
    write_json_atomic(directory / RECORDED_MANIFEST, {
        'parameters': {**sample.parameters.to_json(), 'shots': sample.shots, 'seed': sample.seed},
        'input_sha256': {
            CIRCUIT_FILE: sha256_file(directory / CIRCUIT_FILE),
            DEM_FILE: sha256_file(directory / DEM_FILE),
            'packed_detectors_then_observables_payload': sample.payload_sha256,
        },
        'versions': package_versions(sorted(set(MODEL_PACKAGES) | set(SAMPLING_PACKAGES))),
        'created_utc': '2026-09-10T05:22:38Z',
    })
    return directory


# --- circuit parameters ------------------------------------------------------

def test_circuit_parameters_round_trip_through_json(parameters):
    assert CircuitParameters.from_json(parameters.to_json()) == parameters


def test_circuit_parameters_reject_an_unsupported_noise_model():
    with pytest.raises(ValueError, match='si1000'):
        CircuitParameters(distance=3, rounds=12, p=0.003, noise='uniform')


def test_circuit_parameters_reject_an_out_of_range_error_rate():
    with pytest.raises(ValueError, match='p'):
        CircuitParameters(distance=3, rounds=12, p=0.0)


def test_the_circuit_carries_two_observables_per_patch(parameters):
    circuit = parameters.circuit()
    assert circuit.num_observables == NUM_SECTORS * parameters.patches
    assert parameters.dem(circuit).num_observables == NUM_SECTORS * parameters.patches


# --- a generated sample ------------------------------------------------------

def test_a_generated_sample_reports_its_dimensions_and_identities(sample, parameters):
    assert (sample.shots, sample.seed, sample.parameters) == (SHOTS, 11, parameters)
    assert sample.num_observables == NUM_SECTORS * PATCHES
    assert sample.detectors_packed.shape == (SHOTS, -(-sample.num_detectors // 8))
    assert sample.actual_packed.shape == (SHOTS, -(-sample.num_observables // 8))
    assert not sample.detectors_packed.flags.writeable and not sample.actual_packed.flags.writeable
    assert set(sample.identities) == {'model', 'parent_sample', 'sampling_family'}
    assert sample.source['kind'] == 'generated'


def test_rows_unpack_only_the_requested_rows(sample):
    detectors, actual = sample.rows(np.array([0, 5, 7]))
    everything, every_actual = sample.rows(np.arange(SHOTS))
    assert detectors.dtype == bool and actual.dtype == bool
    np.testing.assert_array_equal(detectors, everything[[0, 5, 7]])
    np.testing.assert_array_equal(actual, every_actual[[0, 5, 7]])


def test_rows_reject_positions_that_are_out_of_range_or_out_of_order(sample):
    with pytest.raises(ValueError, match='range'):
        sample.rows(np.array([SHOTS]))
    with pytest.raises(ValueError, match='increasing'):
        sample.rows(np.array([3, 1]))


def test_a_saved_sample_loads_back_identically(sample, saved):
    loaded = SampleSet.load(saved)
    assert loaded.identities == sample.identities
    assert (loaded.parameters, loaded.seed, loaded.shots) == (sample.parameters, sample.seed, sample.shots)
    assert (loaded.circuit_sha256, loaded.dem_sha256, loaded.payload_sha256) == (
        sample.circuit_sha256, sample.dem_sha256, sample.payload_sha256)
    assert loaded.dem_text == sample.dem_text
    np.testing.assert_array_equal(loaded.detectors_packed, sample.detectors_packed)
    np.testing.assert_array_equal(loaded.actual_packed, sample.actual_packed)
    assert loaded.source['directory'] == str(saved)


def test_a_saved_sample_writes_every_artifact(saved):
    for name in (CIRCUIT_FILE, DEM_FILE, DETECTORS_FILE, ACTUAL_FILE, SAMPLE_MANIFEST):
        assert (saved / name).is_file()
    manifest = read_json(saved / SAMPLE_MANIFEST)
    assert set(manifest['files']) == {CIRCUIT_FILE, DEM_FILE, DETECTORS_FILE, ACTUAL_FILE}
    assert manifest['versions'] == package_versions(SAMPLING_PACKAGES)


def test_loading_rejects_a_tampered_payload(saved, tmp_path):
    directory = copied(saved, tmp_path / 'tampered')
    packed = np.load(directory / DETECTORS_FILE)
    packed[0, 0] ^= 1
    np.save(directory / DETECTORS_FILE, packed)
    with pytest.raises(ValueError, match=DETECTORS_FILE):
        SampleSet.load(directory)


def test_loading_rejects_a_tampered_manifest_hash(saved, tmp_path):
    directory = copied(saved, tmp_path / 'tampered')
    manifest = read_json(directory / SAMPLE_MANIFEST)
    manifest['payload_sha256'] = 'ff' * 32
    write_json_atomic(directory / SAMPLE_MANIFEST, manifest)
    with pytest.raises(ValueError, match='payload'):
        SampleSet.load(directory)


def test_loading_rejects_a_declared_shot_count_the_arrays_do_not_have(saved, tmp_path):
    directory = copied(saved, tmp_path / 'tampered')
    manifest = read_json(directory / SAMPLE_MANIFEST)
    manifest['shots'] = SHOTS + 1
    write_json_atomic(directory / SAMPLE_MANIFEST, manifest)
    with pytest.raises(ValueError, match='shots'):
        SampleSet.load(directory)


# --- an imported recorded run ------------------------------------------------

def test_a_recorded_run_imports_with_the_same_identities(sample, saved, tmp_path):
    imported = SampleSet.load_recorded_run(write_recorded_run(saved, tmp_path / 'recorded', sample))
    assert imported.identities == sample.identities
    assert imported.source['kind'] == 'imported'
    assert imported.source['manifest_sha256'] == sha256_file(tmp_path / 'recorded' / RECORDED_MANIFEST)
    assert imported.dem_text == sample.dem_text
    np.testing.assert_array_equal(imported.detectors_packed, sample.detectors_packed)
    np.testing.assert_array_equal(imported.actual_packed, sample.actual_packed)


def test_importing_rejects_a_tampered_model(sample, saved, tmp_path):
    directory = write_recorded_run(saved, tmp_path / 'recorded', sample)
    (directory / DEM_FILE).write_bytes((directory / DEM_FILE).read_bytes() + b'\nerror(0.1) D0\n')
    with pytest.raises(ValueError, match=DEM_FILE):
        SampleSet.load_recorded_run(directory)


def test_importing_rejects_a_tampered_payload(sample, saved, tmp_path):
    directory = write_recorded_run(saved, tmp_path / 'recorded', sample)
    packed = np.load(directory / ACTUAL_FILE)
    packed[1, 0] ^= 1
    np.save(directory / ACTUAL_FILE, packed)
    with pytest.raises(ValueError, match='payload'):
        SampleSet.load_recorded_run(directory)


# --- graph checks ------------------------------------------------------------

def corrupted(patches, transform, *, index=0):
    """The same split with one patch's check graph replaced by ``transform``'s edges."""
    patch = patches[index]
    graph = patch.check_graph
    replaced = dataclasses.replace(
        patch, check_graph=DecodingGraph(graph.num_detectors, graph.num_observables,
                                         transform(list(graph.edges))))
    return dataclasses.replace(
        patches, patches=patches.patches[:index] + (replaced,) + patches.patches[index + 1:])


def with_weight_change(delta):
    def transform(edges):
        u, v, weight, mask = edges[0]
        return [(u, v, weight + delta, mask)] + edges[1:]
    return transform


def test_check_graphs_accepts_the_fixture_split(context):
    checks = check_graphs(context.dem, context.patches)
    assert checks.passed and checks.equivalent
    assert checks.joint_edges == checks.rebuilt_edges > 0
    assert checks.max_weight_difference <= EDGE_WEIGHT_TOLERANCE
    assert checks.yoke_degrees[0] > checks.max_non_yoke_degree > 0
    assert checks.median_detector_degree > 0
    assert checks.to_json()['equivalent'] is True


def test_check_graphs_rejects_a_missing_edge(context):
    checks = check_graphs(context.dem, corrupted(context.patches, lambda edges: edges[1:]))
    assert not checks.passed and checks.rebuilt_edges == checks.joint_edges - 1
    assert checks.to_json()['multiplicity_mismatches'] + checks.to_json()['missing_groups'] > 0


def test_check_graphs_rejects_a_changed_multiplicity(context):
    checks = check_graphs(context.dem, corrupted(context.patches, lambda edges: edges + [edges[0]]))
    assert not checks.passed and checks.rebuilt_edges == checks.joint_edges + 1


def test_check_graphs_rejects_a_changed_mask(context):
    def flip_mask(edges):
        u, v, weight, mask = edges[0]
        return [(u, v, weight, mask ^ 1)] + edges[1:]

    checks = check_graphs(context.dem, corrupted(context.patches, flip_mask))
    assert not checks.passed
    assert checks.to_json()['missing_groups'] > 0 or checks.to_json()['extra_groups'] > 0


def test_check_graphs_accepts_a_weight_difference_below_the_tolerance(context):
    checks = check_graphs(context.dem, corrupted(context.patches, with_weight_change(1e-10)))
    assert checks.passed and 0 < checks.max_weight_difference <= EDGE_WEIGHT_TOLERANCE


def test_check_graphs_rejects_a_weight_difference_above_the_tolerance(context):
    checks = check_graphs(context.dem, corrupted(context.patches, with_weight_change(1e-6)))
    assert not checks.passed and checks.max_weight_difference > EDGE_WEIGHT_TOLERANCE
    assert checks.to_json()['weight_mismatches'] > 0


# --- collection --------------------------------------------------------------

def test_a_collected_record_holds_every_row_and_both_correlation_branches(collected, sample, context):
    record = collected.record
    assert record.shots == SHOTS and record.num_patches == PATCHES
    np.testing.assert_array_equal(record.rows, np.arange(SHOTS))
    detectors, actual = sample.rows(np.arange(SHOTS))
    np.testing.assert_array_equal(record.actual, actual)
    np.testing.assert_array_equal(record.yoke, detectors[:, list(context.patches.yoke_detector_ids)])
    assert record.reweighted_patches.any() and not record.reweighted_patches.all()
    assert (record.forced_correlated[~record.reweighted_patches]
            == record.forced_plain[~record.reweighted_patches]).all()


def test_collection_reproduces_one_call_when_partitioned(sample, context):
    rows = np.arange(SHOTS)
    single = collect_rows(context, *sample.rows(rows), rows)
    parts = [collect_rows(context, *sample.rows(chunk), chunk) for chunk in (rows[:73], rows[73:])]
    assert parts[0].work + parts[1].work == single.work
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(
            np.concatenate([getattr(part.record, name) for part in parts]), getattr(single.record, name))


def test_collection_work_counts_the_calls_that_actually_ran(sample):
    context = L1Context.from_dem_text(sample.dem_text, num_patches=PATCHES)
    decoders = tuple(CountingDecoder(decoder) for decoder in context.decoders)
    matchers = tuple(CountingMatcher(matcher) for matcher in context.matchers)
    context.decoders, context.matchers = decoders, matchers
    rows = np.arange(12)
    work = collect_rows(context, *sample.rows(rows), rows).work
    decodes = len(rows) * PATCHES
    fired = sum(matcher.fired for matcher in matchers)
    assert sum(decoder.calls for decoder in decoders) == decodes
    assert sum(matcher.calls for matcher in matchers) == decodes
    assert 0 < fired <= decodes
    assert work == CollectionWork(
        rows=len(rows), uf_decodes=decodes, dijkstra_searches=NUM_SECTORS * decodes,
        dijkstra_states=sum(decoder.states for decoder in decoders), unforced_plain_calls=decodes,
        plain_forced_calls=FORCED_CALLS_PER_DECODE * decodes, reweight_attempts=decodes,
        correlated_forced_calls=FORCED_CALLS_PER_DECODE * fired, correlated_validation_calls=fired,
        joint_decodes=len(rows))


class CountingDecoder:
    """A cluster-gap decoder that counts decoded syndromes and settled Dijkstra states."""

    def __init__(self, inner):
        self.inner, self.calls, self.states = inner, 0, 0

    def decode_with_gaps(self, syndrome):
        result = self.inner.decode_with_gaps(syndrome)
        self.calls += 1
        self.states += int(np.asarray(result.dijkstra_states).sum())
        return result


class CountingMatcher:
    """A forced-weight matcher that counts calls and how often a correlation rule fired."""

    def __init__(self, inner):
        self.inner, self.calls, self.fired = inner, 0, 0

    def forced_weights(self, syndrome):
        result = self.inner.forced_weights(syndrome)
        self.calls += 1
        self.fired += int(result.rules_fired)
        return result


class BrokenDecoder:
    """A decoder whose reported correction no longer reproduces its syndrome."""

    def __init__(self, inner):
        self.inner = inner

    def decode_with_gaps(self, syndrome):
        result = self.inner.decode_with_gaps(syndrome)
        selected = result.selected_edges[:-1] if result.selected_edges else (0,)
        return dataclasses.replace(result, selected_edges=selected)


def test_an_invalid_correction_names_the_parent_row_and_patch(sample):
    context = L1Context.from_dem_text(sample.dem_text, num_patches=PATCHES)
    context.decoders = context.decoders[:2] + (BrokenDecoder(context.decoders[2]),) + context.decoders[3:]
    rows = np.arange(5, 15)
    with pytest.raises(ValueError, match=r'row 5.*patch 2'):
        collect_rows(context, *sample.rows(rows), rows)


def test_collection_rejects_a_detector_array_of_the_wrong_width(sample, context):
    rows = np.arange(4)
    detectors, actual = sample.rows(rows)
    with pytest.raises(ValueError, match='detector'):
        collect_rows(context, detectors[:, :-1], actual, rows)


def test_collection_rejects_row_ids_that_are_not_increasing(sample, context):
    rows = np.arange(4)
    detectors, actual = sample.rows(rows)
    with pytest.raises(ValueError, match='increasing'):
        collect_rows(context, detectors, actual, np.array([3, 2, 1, 0]))


# --- record checks -----------------------------------------------------------

def most_decisive_entry(record):
    """The (shot, patch, sector) whose plain gap most strongly prefers the reference."""
    gaps = signed_gaps(record.forced_plain,
                       record.mwpm_reference.reshape(record.shots, record.num_patches, NUM_SECTORS))
    return np.unravel_index(int(np.argmax(gaps)), gaps.shape)


def test_a_collected_record_passes_every_invariant(collected):
    checks = check_record(collected.record)
    assert checks.passed
    assert checks.rows == SHOTS
    assert checks.check_parity_violations == 0 and checks.final_parity_violations == 0
    assert checks.plain_additivity_max_error <= WEIGHT_TOLERANCE
    assert checks.correlated_additivity_max_error <= WEIGHT_TOLERANCE
    assert checks.plain_preferred_disagreements == 0 and checks.correlated_preferred_disagreements == 0
    assert checks.unexplained_disagreements == 0 and checks.value_violations == 0
    assert checks.joint_disagreements == checks.tie_explained_disagreements
    assert 0.0 < checks.joint_agreement <= 1.0
    checks.raise_if_failed()
    assert checks.to_json()['passed'] is True


def test_a_flipped_yoke_bit_fails_the_check_parity_gate(collected):
    yoke = np.array(collected.record.yoke)
    yoke[3, 0] ^= True
    checks = check_record(dataclasses.replace(collected.record, yoke=yoke))
    assert not checks.passed and checks.check_parity_violations == 1
    assert checks.diagnostics['check_parity'] == (int(collected.record.rows[3]),)
    with pytest.raises(ValueError, match='check_parity'):
        checks.raise_if_failed()


def test_a_corrupted_reference_fails_the_preferred_class_gate(collected):
    shot, patch, sector = most_decisive_entry(collected.record)
    reference = np.array(collected.record.mwpm_reference)
    reference[shot, NUM_SECTORS * patch + sector] ^= True
    checks = check_record(dataclasses.replace(collected.record, mwpm_reference=reference))
    assert not checks.passed and checks.plain_preferred_disagreements >= 1
    assert int(collected.record.rows[shot]) in checks.diagnostics['plain_preferred']


def test_a_corrupted_forced_cost_fails_the_additivity_gate(collected):
    for name in ('forced_plain', 'forced_correlated'):
        forced = np.array(getattr(collected.record, name))
        forced[0, 0, 0, 0] += 0.5
        checks = check_record(dataclasses.replace(collected.record, **{name: forced}))
        assert not checks.passed
        assert getattr(checks, f'{name[len("forced_"):]}_additivity_max_error') > WEIGHT_TOLERANCE


def test_a_corrupted_joint_prediction_fails_the_parity_and_joint_gates(collected):
    shot, patch, sector = most_decisive_entry(collected.record)
    joint = np.array(collected.record.joint_mwpm)
    joint[shot, NUM_SECTORS * patch + sector] ^= True
    checks = check_record(dataclasses.replace(collected.record, joint_mwpm=joint))
    assert not checks.passed
    assert checks.final_parity_violations == 1
    assert checks.unexplained_disagreements >= 1
    assert int(collected.record.rows[shot]) in checks.diagnostics['unexplained_joint']


def test_a_non_finite_collected_value_fails_the_value_gate(collected):
    record = dataclasses.replace(collected.record)
    gaps = np.array(record.cluster_gap)
    gaps[2, 1] = np.inf
    # The L1Record constructor already rejects this, so the check is given a record that
    # bypassed it: check_record is the gate a stored record passes on its own terms.
    object.__setattr__(record, 'cluster_gap', gaps)
    checks = check_record(record)
    assert not checks.passed and checks.value_violations == 1
    assert checks.diagnostics['values'] == (int(record.rows[2]),)


# --- collection settings -----------------------------------------------------

def test_collection_settings_normalize_rows_and_summarize_them():
    settings = CollectionSettings(role='calibration', rows=np.array([9, 2, 5]))
    np.testing.assert_array_equal(settings.rows, [2, 5, 9])
    assert not settings.rows.flags.writeable and settings.rows.dtype == np.int64
    assert settings.shots == 3
    assert settings.rows_summary() == {'count': 3, 'start': 2, 'stop': 10,
                                       'sha256': row_ids_sha256(np.array([2, 5, 9]))}


def test_collection_settings_reject_duplicate_rows():
    with pytest.raises(ValueError, match='unique'):
        CollectionSettings(role='evaluation', rows=np.array([1, 1, 2]))


def test_collection_settings_reject_negative_float_and_boolean_rows():
    with pytest.raises(ValueError, match='nonnegative'):
        CollectionSettings(role='evaluation', rows=np.array([-1, 2]))
    with pytest.raises(ValueError, match='integer'):
        CollectionSettings(role='evaluation', rows=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match='mask'):
        CollectionSettings(role='evaluation', rows=np.array([True, False, True]))


def test_collection_settings_reject_an_empty_or_multidimensional_row_array():
    with pytest.raises(ValueError, match='nonempty'):
        CollectionSettings(role='evaluation', rows=np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match='one-dimensional'):
        CollectionSettings(role='evaluation', rows=np.arange(4).reshape(2, 2))


def test_collection_settings_reject_a_role_outside_the_two_m1_roles():
    with pytest.raises(ValueError, match='confirmation'):
        CollectionSettings(role='confirmation', rows=np.arange(4))
    assert ROLES == ('calibration', 'evaluation')


def test_collection_settings_reject_non_positive_worker_chunk_and_cap_counts():
    for field in ('workers', 'chunk_size', 'max_chunks'):
        with pytest.raises(ValueError, match=field):
            CollectionSettings(role='evaluation', rows=np.arange(4), **{field: 0})
    assert CollectionSettings(role='evaluation', rows=np.arange(4)).max_chunks is None


# --- the collection coordinator ----------------------------------------------

COLLECTED = np.arange(24)
"""Few enough distance-3 rows that a dozen collections stay fast, enough to split into
three chunks of eight."""

CHUNK = 8
"""Rows per scheduled chunk in the tests, so ``COLLECTED`` is three chunks."""


def evaluation_settings(**overrides) -> CollectionSettings:
    return CollectionSettings(**{'role': 'evaluation', 'rows': COLLECTED, 'chunk_size': CHUNK,
                                 **overrides})


@pytest.fixture(scope='module')
def reference(saved, tmp_path_factory):
    """One uninterrupted serial collection that every other run is compared against."""
    out_dir = tmp_path_factory.mktemp('reference') / 'evaluation'
    return collect_sample(saved, out_dir, evaluation_settings())


def retained_work(loaded) -> dict:
    """The manifest's retained-row work counts, without the resumption bookkeeping."""
    return {name: loaded.manifest['collection_work'][name] for name in WORK_FIELDS}


def assert_same_collection(left, right) -> None:
    """Two runs agree when every array, the row ids, the identities, and the retained-row
    work counts agree; how many chunks, workers, or restarts produced them cannot show."""
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(left.record, name), getattr(right.record, name))
    assert dict(left.identities) == dict(right.identities)
    assert retained_work(left) == retained_work(right)


def raise_once(message: str):
    """A hook that interrupts the first run and lets every later one through."""
    state = {'fired': False}

    def hook() -> None:
        if not state['fired']:
            state['fired'] = True
            raise RuntimeError(message)

    return hook


def test_a_completed_collection_publishes_a_verified_record(reference, saved, sample):
    out_dir = reference.directory
    assert isinstance(reference, LoadedRecord)
    for name in (RECORD_FILE, RECORD_MANIFEST, COLLECTION_FILE):
        assert (out_dir / name).is_file()
    assert not (out_dir / CHECKPOINT_FILE).exists()
    np.testing.assert_array_equal(reference.record.rows, COLLECTED)
    manifest = reference.manifest
    assert manifest['status'] == 'complete' and manifest['role'] == 'evaluation'
    assert manifest['shots'] == len(COLLECTED) and manifest['parent_shots'] == sample.shots
    assert manifest['seed'] == sample.seed
    assert manifest['parent_payload_sha256'] == sample.payload_sha256
    assert dict(manifest['parameters']) == sample.parameters.to_json()
    assert manifest['artifacts'][RECORD_FILE] == sha256_file(out_dir / RECORD_FILE)
    assert manifest['checks']['passed'] is True
    assert manifest['checks']['graph']['equivalent'] is True
    assert manifest['checks']['record']['passed'] is True
    assert manifest['collection_work']['rows'] == len(COLLECTED)
    assert manifest['collection_work']['resumptions'] == 0
    assert 'interruption' in manifest['collection_work']['telemetry']
    assert manifest['timing']['seconds_total'] >= manifest['timing']['seconds_this_run'] > 0
    assert manifest['code_commit'] is None or len(manifest['code_commit']) == 40


def test_the_published_identities_name_the_sample_the_decoder_and_the_collection(reference, sample):
    assert set(reference.identities) == set(IDENTITY_NAMES)
    for name in ('model', 'parent_sample', 'sampling_family'):
        assert reference.identities[name] == sample.identities[name]
    assert reference.identities['decoder'] == decoder_identity(
        sources=source_hashes(DECODER_SOURCES), conventions=RECORD_CONVENTIONS,
        versions=package_versions(DECODER_PACKAGES))


def test_the_manifest_lists_the_audit_sources_that_do_not_exist_yet(reference):
    audit = reference.manifest['source_sha256']['audit']
    assert 'src/yoked/hierarchical/_stages.py' in audit['missing']
    assert 'src/yoked/_yoked_memory_circuits.py' in audit
    assert set(reference.manifest['source_sha256']) == {'decoder', 'check', 'audit'}
    assert set(reference.manifest['versions']) == {'decoder', 'check'}


def test_a_collected_record_matches_a_direct_collect_rows_call(reference, sample, context):
    direct = collect_rows(context, *sample.rows(COLLECTED), COLLECTED)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(reference.record, name), getattr(direct.record, name))
    assert retained_work(reference) == direct.work.to_json()


def test_one_chunk_and_many_chunks_collect_the_same_record(reference, saved, tmp_path):
    whole = collect_sample(saved, tmp_path / 'whole', evaluation_settings(chunk_size=len(COLLECTED)))
    assert_same_collection(whole, reference)


def test_a_parallel_collection_matches_the_serial_one(reference, saved, tmp_path):
    parallel = collect_sample(saved, tmp_path / 'parallel', evaluation_settings(workers=2))
    assert_same_collection(parallel, reference)


def test_max_chunks_stops_scheduling_and_a_rerun_finishes_the_same_record(reference, saved, tmp_path):
    out_dir = tmp_path / 'resumed'
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    assert (out_dir / CHECKPOINT_FILE).is_file() and not (out_dir / RECORD_MANIFEST).exists()
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    resumed = collect_sample(saved, out_dir, evaluation_settings())
    assert_same_collection(resumed, reference)
    assert resumed.manifest['collection_work']['resumptions'] == 2
    assert not (out_dir / CHECKPOINT_FILE).exists()


@pytest.mark.parametrize('point', ['before_checkpoint', 'after_checkpoint', 'after_record',
                                   'before_manifest'])
def test_an_interruption_at_each_hook_point_restarts_to_the_same_record(point, reference, saved,
                                                                       tmp_path):
    out_dir = tmp_path / point
    hooks = _CollectionHooks(**{point: raise_once(point)})
    with pytest.raises(RuntimeError, match=point):
        collect_sample(saved, out_dir, evaluation_settings(), hooks=hooks)
    assert not (out_dir / RECORD_MANIFEST).exists()
    restarted = collect_sample(saved, out_dir, evaluation_settings(), hooks=hooks)
    assert_same_collection(restarted, reference)


def test_a_completed_collection_is_reloaded_without_decoding_again(reference, saved, monkeypatch):
    def refuse(*arguments, **keywords):
        raise AssertionError('a completed collection must not decode anything again')

    monkeypatch.setattr('yoked.hierarchical._collect.collect_rows', refuse)
    again = collect_sample(saved, reference.directory, evaluation_settings())
    assert_same_collection(again, reference)
    assert again.manifest['checks']['identity'] == reference.manifest['checks']['identity']


def test_a_changed_check_identity_rechecks_the_stored_arrays_and_republishes(reference, saved,
                                                                            tmp_path, monkeypatch):
    out_dir = copied(reference.directory, tmp_path / 'rechecked')
    before = sha256_file(out_dir / RECORD_FILE)
    monkeypatch.setattr('yoked.hierarchical._collect.source_hashes',
                        changed_sources(CHECK_SOURCES))
    rechecked = collect_sample(saved, out_dir, evaluation_settings())
    assert rechecked.manifest['checks']['identity'] != reference.manifest['checks']['identity']
    assert rechecked.manifest['checks']['passed'] is True
    assert sha256_file(out_dir / RECORD_FILE) == before
    assert_same_collection(rechecked, reference)


def test_a_changed_decoder_identity_cannot_join_a_completed_collection(reference, saved, tmp_path,
                                                                      monkeypatch):
    out_dir = copied(reference.directory, tmp_path / 'other-decoder')
    monkeypatch.setattr('yoked.hierarchical._collect.source_hashes',
                        changed_sources(DECODER_SOURCES))
    with pytest.raises(ValueError, match='decoder'):
        collect_sample(saved, out_dir, evaluation_settings())


def test_a_changed_decoder_identity_cannot_join_a_partial_collection(saved, tmp_path, monkeypatch):
    out_dir = tmp_path / 'partial-decoder'
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    monkeypatch.setattr('yoked.hierarchical._collect.package_versions', bumped_versions())
    with pytest.raises(ValueError, match='decoder'):
        collect_sample(saved, out_dir, evaluation_settings())


def test_changed_rows_are_rejected_rather_than_collected_into_the_same_directory(saved, tmp_path):
    out_dir = tmp_path / 'other-rows'
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    with pytest.raises(ValueError, match='rows'):
        collect_sample(saved, out_dir,
                       CollectionSettings(role='evaluation', rows=np.arange(25), chunk_size=CHUNK))


def test_a_changed_role_is_rejected_rather_than_collected_into_the_same_directory(saved, tmp_path):
    out_dir = tmp_path / 'other-role'
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    with pytest.raises(ValueError, match='role'):
        collect_sample(saved, out_dir,
                       CollectionSettings(role='calibration', rows=COLLECTED, chunk_size=CHUNK))


def test_failing_record_checks_retain_the_checkpoint_and_publish_no_manifest(saved, tmp_path,
                                                                            monkeypatch):
    out_dir = tmp_path / 'failed-checks'
    monkeypatch.setattr('yoked.hierarchical._collect.check_record', failing_record_checks)
    with pytest.raises(ValueError, match='check_parity'):
        collect_sample(saved, out_dir, evaluation_settings())
    assert (out_dir / CHECKPOINT_FILE).is_file()
    assert not (out_dir / RECORD_MANIFEST).exists()
    assert read_json(out_dir / FAILED_CHECKS_FILE)['record']['failures'] == ['check_parity']


def test_failing_graph_checks_stop_the_collection_before_any_row_is_decoded(saved, tmp_path,
                                                                           monkeypatch):
    out_dir = tmp_path / 'failed-graphs'
    monkeypatch.setattr('yoked.hierarchical._collect.check_graphs', failing_graph_checks)
    monkeypatch.setattr('yoked.hierarchical._collect.collect_rows', refusing_collect_rows)
    with pytest.raises(ValueError, match='[Gg]raph'):
        collect_sample(saved, out_dir, evaluation_settings())
    assert not (out_dir / CHECKPOINT_FILE).exists() and not (out_dir / RECORD_MANIFEST).exists()


def test_a_corrupt_checkpoint_is_rejected_rather_than_silently_replaced(saved, tmp_path):
    out_dir = tmp_path / 'corrupt-checkpoint'
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    data = (out_dir / CHECKPOINT_FILE).read_bytes()
    (out_dir / CHECKPOINT_FILE).write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError):
        collect_sample(saved, out_dir, evaluation_settings())
    assert (out_dir / CHECKPOINT_FILE).is_file()


def test_a_checkpoint_from_another_collection_is_rejected(saved, tmp_path):
    evaluation, calibration = tmp_path / 'evaluation', tmp_path / 'calibration'
    other = CollectionSettings(role='calibration', rows=COLLECTED, chunk_size=CHUNK, max_chunks=1)
    assert collect_sample(saved, evaluation, evaluation_settings(max_chunks=1)) is None
    assert collect_sample(saved, calibration, other) is None
    shutil.copy(evaluation / CHECKPOINT_FILE, calibration / CHECKPOINT_FILE)
    with pytest.raises(ValueError, match='collection'):
        collect_sample(saved, calibration, other)


def test_a_corrupt_record_is_rejected_on_a_completed_collection(reference, saved, tmp_path):
    out_dir = copied(reference.directory, tmp_path / 'corrupt-record')
    (out_dir / RECORD_FILE).write_bytes((out_dir / RECORD_FILE).read_bytes() + b'\0')
    with pytest.raises(ValueError, match=RECORD_FILE):
        collect_sample(saved, out_dir, evaluation_settings())


def test_a_missing_manifest_leaves_an_orphan_record_that_is_republished(reference, saved, tmp_path):
    out_dir = copied(reference.directory, tmp_path / 'orphan')
    (out_dir / RECORD_MANIFEST).unlink()
    republished = collect_sample(saved, out_dir, evaluation_settings())
    assert_same_collection(republished, reference)
    assert republished.manifest['collection_work']['rows'] == len(COLLECTED)


def test_changed_sample_bytes_under_an_unchanged_manifest_are_rejected(saved, tmp_path):
    sample_dir = copied(saved, tmp_path / 'tampered-sample')
    packed = np.load(sample_dir / DETECTORS_FILE)
    packed[0, 0] ^= 1
    np.save(sample_dir / DETECTORS_FILE, packed)
    with pytest.raises(ValueError, match=DETECTORS_FILE):
        collect_sample(sample_dir, tmp_path / 'out', evaluation_settings())


def test_rows_outside_the_parent_sample_are_rejected(saved, tmp_path):
    with pytest.raises(ValueError, match='range'):
        collect_sample(saved, tmp_path / 'out',
                       CollectionSettings(role='evaluation', rows=np.array([0, SHOTS])))


def test_collecting_into_the_sample_directory_is_refused(saved, tmp_path):
    sample_dir = copied(saved, tmp_path / 'in-place')
    with pytest.raises(ValueError, match='sample'):
        collect_sample(sample_dir, sample_dir, evaluation_settings())


def changed_sources(group):
    """A ``source_hashes`` stand-in that moves exactly one group's hashes."""
    def hashes(requested):
        digests = source_hashes(requested)
        if requested is group:
            digests[next(iter(digests))] = 'ff' * 32
        return digests
    return hashes


def bumped_versions():
    """A ``package_versions`` stand-in that reports a newer decoder dependency."""
    def versions(names):
        installed = package_versions(names)
        if names is DECODER_PACKAGES:
            installed['numpy'] = installed['numpy'] + '.1'
        return installed
    return versions


def failing_record_checks(record):
    """Record checks that fail the parity gate, as a damaged record's would."""
    checks = check_record(record)
    return dataclasses.replace(checks, check_parity_violations=1,
                               diagnostics={'check_parity': (int(record.rows[0]),)})


def failing_graph_checks(dem, patches):
    """Graph checks that report the split and the joint graph disagreeing."""
    checks = check_graphs(dem, patches)
    return dataclasses.replace(checks, equivalent=False, missing_groups=1,
                               examples=('missing (0, boundary, mask 1)',))


def refusing_collect_rows(*arguments, **keywords):
    raise AssertionError('no row may be decoded before the graph gate passes')


def test_checkpoints_written_while_chunks_arrive_do_not_change_the_result(reference, saved,
                                                                         tmp_path, monkeypatch):
    # Zero seconds makes every installed chunk trigger the interval write that a real run
    # reaches once a minute, so the mid-run checkpoint path is actually exercised.
    monkeypatch.setattr('yoked.hierarchical._collect.CHECKPOINT_INTERVAL_SECONDS', 0.0)
    writes = []
    hooks = _CollectionHooks(after_checkpoint=lambda: writes.append(None))
    flushed = collect_sample(saved, tmp_path / 'flushed', evaluation_settings(), hooks=hooks)
    assert len(writes) > len(COLLECTED) // CHUNK
    assert_same_collection(flushed, reference)


def test_a_sample_of_another_model_cannot_join_an_existing_collection(saved, tmp_path, parameters):
    out_dir = tmp_path / 'other-model'
    assert collect_sample(saved, out_dir, evaluation_settings(max_chunks=1)) is None
    louder = dataclasses.replace(parameters, p=0.006)
    other = SampleSet.sample(louder, seed=11, shots=SHOTS).save(tmp_path / 'louder-sample')
    with pytest.raises(ValueError, match='model'):
        collect_sample(other, out_dir, evaluation_settings())


def test_the_buffers_reject_a_chunk_that_does_not_belong_where_it_claims(reference, sample, context):
    buffers = _Buffers.allocate(rows=np.arange(4), patches=PATCHES)
    collected = collect_rows(context, *sample.rows(np.arange(2)), np.arange(2))
    buffers.install(np.array([0, 1]), collected.record, collected.work)
    with pytest.raises(ValueError, match='already completed'):
        buffers.install(np.array([0, 1]), collected.record, collected.work)
    with pytest.raises(ValueError, match=r'\[0, 4\)'):
        buffers.install(np.array([4, 5]), collected.record, collected.work)
    with pytest.raises(ValueError, match='different parent row ids'):
        buffers.install(np.array([2, 3]), collected.record, collected.work)
    assert not buffers.complete and list(buffers.pending()) == [2, 3]


def test_a_chunk_that_was_never_scheduled_is_rejected(sample, context):
    buffers = _Buffers.allocate(rows=np.arange(4), patches=PATCHES)
    collected = collect_rows(context, *sample.rows(np.arange(2)), np.arange(2))
    arrived = [(np.array([2, 3]), collected.record, collected.work)]
    with pytest.raises(ValueError, match='not\n?\\s*scheduled, or arrived twice'):
        _run_chunks(buffers, [np.array([0, 1])], arrived, checkpoint=lambda: None)
