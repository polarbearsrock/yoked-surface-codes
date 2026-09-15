"""Tests for verified sample sets, single-process L1 collection, and the graph and
record checks.

Checks: a generated sample's save/load round trip and the file, payload, and
dimension verification that load performs; the imported recorded-run format and its
rejection of a tampered file; graph equivalence on the distance-3 fixture together
with deliberately changed edge multiplicity, mask, and weight (a difference below
1e-9 passing and a larger one failing); collection of a few hundred fixture shots
through both correlation branches, with valid corrections, instrumented work counts,
and equality between one call and a partitioned serial collection; and every record
invariant failing on its own deliberately corrupted field.
"""
import dataclasses
import shutil

import numpy as np
import pytest

from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._collect import (
    ACTUAL_FILE, CIRCUIT_FILE, DEM_FILE, DETECTORS_FILE, EDGE_WEIGHT_TOLERANCE,
    FORCED_CALLS_PER_DECODE, RECORDED_MANIFEST, SAMPLE_MANIFEST, WEIGHT_TOLERANCE,
    CircuitParameters, CollectionWork, L1Context, SampleSet, check_graphs, check_record,
    collect_rows,
)
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import (
    MODEL_PACKAGES, SAMPLING_PACKAGES, package_versions, read_json, sha256_file, write_json_atomic,
)
from yoked.hierarchical._record import ARRAY_FIELDS

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
