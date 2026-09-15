"""Tests for the full-versus-subset record reproduction check (_reproduction.py).

Checks: the module is a check source rather than a decoder source; a subset built with
``L1Record.subset`` from a synthetic full record passes, with every array reported equal,
every row matched, and the JSON form plain and round-trippable; the full record compared
with itself passes; a subset need not carry the full record's baselines, and a baseline
it does carry is not compared; one changed value in any stored array is reported under
that array's name with the parent row id of the changed row, the other arrays stay
equal, the matched-row count drops by one, and ``raise_if_failed`` names the array and
the row; a float changed by one ulp is a mismatch, because the comparison is exact; a
row changed on several arrays is counted once in the matched-row count; mismatches on
more rows than the listing bound keep their full count while the list is truncated to
the first parent rows; a subset row absent from the full record, whether between two
full rows or past the last one, raises naming the row; records with different patch
counts and inputs that are not records are refused; and the ``SubsetCheck`` record
itself refuses inconsistent counts, lists, and flags.
"""
from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType

import numpy as np
import pytest

from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import CHECK_SOURCES, DECODER_SOURCES, canonical_json
from yoked.hierarchical._record import ARRAY_FIELDS, L1Record
from yoked.hierarchical._reproduction import (
    COMPARISON, MAX_LISTED_ROWS, SubsetCheck, subset_reproduction,
)

SHOTS, PATCHES = 12, 2
"""A dozen synthetic shots of a two-patch record: enough rows that a subset can skip
some, list more mismatches than the bound, and still be read by eye."""

ROWS = np.array([0, 2, 3, 7, 8, 11, 12, 15, 19, 20, 23, 24], dtype=np.int64)
"""Parent row ids of the synthetic full record, with gaps, so that a position in the
record and a parent row id are visibly different numbers."""

POSITIONS = np.array([1, 3, 4, 8, 10])
"""Positions of the synthetic subset inside the full record: parent rows 2, 7, 8, 19, 23."""

CHANGED_POSITION = 1
"""Which subset row a single-value test changes: position 1 of the subset is parent row 7."""

CHANGED_ROW = int(ROWS[POSITIONS[CHANGED_POSITION]])


def synthetic_record(*, shots: int = SHOTS, patches: int = PATCHES, rows=ROWS,
                     seed: int = 0) -> L1Record:
    """A valid record of random values; nothing about the check depends on decoding."""
    generator = np.random.default_rng(seed)
    columns = NUM_SECTORS * patches
    bits = lambda *shape: generator.integers(0, 2, size=shape, dtype=np.int64).astype(bool)
    return L1Record(
        actual=bits(shots, columns), yoke=bits(shots, NUM_SECTORS),
        uf_reference=bits(shots, columns), mwpm_reference=bits(shots, columns),
        correlated_prediction=bits(shots, columns), joint_mwpm=bits(shots, columns),
        cluster_gap=generator.uniform(0.0, 5.0, size=(shots, columns)),
        dijkstra_states=generator.integers(0, 100, size=(shots, columns), dtype=np.int64),
        forced_plain=generator.uniform(0.0, 20.0, size=(shots, patches, NUM_SECTORS, NUM_SECTORS)),
        forced_correlated=generator.uniform(0.0, 20.0,
                                            size=(shots, patches, NUM_SECTORS, NUM_SECTORS)),
        reweighted_patches=bits(shots, patches), rows=np.asarray(rows, dtype=np.int64),
    )


def changed(array: np.ndarray, index: tuple) -> np.ndarray:
    """A writable copy of a record array with one entry changed to another valid value."""
    copy = np.array(array)
    if copy.dtype == bool:
        copy[index] = not copy[index]
    elif copy.dtype.kind == 'f':
        copy[index] = np.nextafter(copy[index], np.inf)     # one ulp: still finite, nonnegative
    else:
        copy[index] = copy[index] + 1
    return copy


def first_entry(record: L1Record, name: str, position: int) -> tuple:
    """The index of the first entry of ``name`` on the row at ``position``."""
    return (position,) + (0,) * (getattr(record, name).ndim - 1)


@pytest.fixture(scope='module')
def full() -> L1Record:
    return synthetic_record()


@pytest.fixture(scope='module')
def subset(full) -> L1Record:
    return full.subset(POSITIONS)


VALUE_FIELDS = tuple(name for name in ARRAY_FIELDS if name != 'rows')
"""Every stored array but the row ids, which a changed value cannot be tested on: a
changed id is a different row, and a subset row absent from the full record raises."""


# --- provenance --------------------------------------------------------------

def test_the_module_is_a_check_source_and_not_a_decoder_source():
    assert 'src/yoked/hierarchical/_reproduction.py' in CHECK_SOURCES
    assert 'src/yoked/hierarchical/_reproduction.py' not in DECODER_SOURCES


# --- a subset that reproduces ------------------------------------------------

def test_a_subset_of_the_full_record_passes_on_every_array(full, subset):
    check = subset_reproduction(full, subset)
    assert isinstance(check, SubsetCheck)
    assert check.passed
    assert check.subset_rows == len(POSITIONS) and check.matched_rows == len(POSITIONS)
    assert tuple(check.equal) == ARRAY_FIELDS
    assert all(check.equal.values())
    assert all(count == 0 for count in check.mismatch_counts.values())
    assert all(rows == () for rows in check.mismatched_rows.values())
    check.raise_if_failed()                                 # a passing check is silent


def test_the_full_record_reproduces_itself(full):
    check = subset_reproduction(full, full)
    assert check.passed and check.matched_rows == full.shots == check.subset_rows


def test_the_check_is_frozen_and_its_mappings_are_immutable(full, subset):
    check = subset_reproduction(full, subset)
    with pytest.raises(dataclasses.FrozenInstanceError):
        check.subset_rows = 0
    for mapping in (check.equal, check.mismatch_counts, check.mismatched_rows):
        assert isinstance(mapping, MappingProxyType)
        with pytest.raises(TypeError):
            mapping['actual'] = None


def test_the_json_form_is_plain_and_says_what_was_compared(full, subset):
    check = subset_reproduction(full, subset)
    payload = check.to_json()
    assert json.loads(json.dumps(payload)) == payload      # plain JSON types only
    assert canonical_json(payload)
    assert payload['comparison'] == COMPARISON
    assert payload['arrays'] == list(ARRAY_FIELDS)
    assert payload['subset_rows'] == len(POSITIONS) and payload['matched_rows'] == len(POSITIONS)
    assert payload['passed'] is True
    assert payload['equal'] == {name: True for name in ARRAY_FIELDS}
    assert payload['mismatch_counts'] == {name: 0 for name in ARRAY_FIELDS}
    assert payload['mismatched_rows'] == {name: [] for name in ARRAY_FIELDS}
    assert payload['max_listed_rows'] == MAX_LISTED_ROWS


def test_baselines_are_not_compared_and_need_not_be_carried(full, subset):
    baseline = np.zeros((full.shots, NUM_SECTORS * PATCHES), dtype=bool)
    with_baseline = dataclasses.replace(full, baselines={'joint_uf': baseline})
    assert subset_reproduction(with_baseline, subset).passed
    other = np.ones((subset.shots, NUM_SECTORS * PATCHES), dtype=bool)
    differing = dataclasses.replace(subset, baselines={'joint_uf': other})
    check = subset_reproduction(with_baseline, differing)
    assert check.passed and 'joint_uf' not in check.equal


# --- a subset that does not --------------------------------------------------

@pytest.mark.parametrize('name', VALUE_FIELDS)
def test_one_changed_value_is_reported_under_its_array_with_the_parent_row(full, subset, name):
    index = first_entry(subset, name, CHANGED_POSITION)
    damaged = dataclasses.replace(subset, **{name: changed(getattr(subset, name), index)})
    check = subset_reproduction(full, damaged)
    assert not check.passed
    assert check.equal[name] is False
    assert all(check.equal[other] for other in ARRAY_FIELDS if other != name)
    assert check.mismatch_counts[name] == 1
    assert check.mismatched_rows[name] == (CHANGED_ROW,)
    assert check.matched_rows == len(POSITIONS) - 1
    with pytest.raises(ValueError, match=f'{name}.*{CHANGED_ROW}'):
        check.raise_if_failed()


def test_a_float_changed_by_one_ulp_is_a_mismatch_because_the_comparison_is_exact(full, subset):
    index = first_entry(subset, 'forced_plain', CHANGED_POSITION)
    nudged = np.array(subset.forced_plain)
    nudged[index] = np.nextafter(nudged[index], np.inf)
    assert np.allclose(nudged, subset.forced_plain)         # a tolerance would miss it
    check = subset_reproduction(full, dataclasses.replace(subset, forced_plain=nudged))
    assert check.equal['forced_plain'] is False
    assert check.mismatched_rows['forced_plain'] == (CHANGED_ROW,)


def test_a_row_changed_on_several_arrays_is_one_unmatched_row(full, subset):
    damaged = dataclasses.replace(
        subset,
        actual=changed(subset.actual, first_entry(subset, 'actual', CHANGED_POSITION)),
        cluster_gap=changed(subset.cluster_gap, first_entry(subset, 'cluster_gap', CHANGED_POSITION)),
        dijkstra_states=changed(subset.dijkstra_states,
                                first_entry(subset, 'dijkstra_states', CHANGED_POSITION)))
    check = subset_reproduction(full, damaged)
    assert [name for name in ARRAY_FIELDS if not check.equal[name]] == \
        ['actual', 'cluster_gap', 'dijkstra_states']
    assert check.matched_rows == len(POSITIONS) - 1
    assert json.loads(json.dumps(check.to_json()))['passed'] is False


def test_mismatches_beyond_the_listing_bound_keep_their_count_and_list_the_first_rows(full):
    assert full.shots > MAX_LISTED_ROWS
    inverted = dataclasses.replace(full, yoke=~full.yoke)   # every row differs
    check = subset_reproduction(full, inverted)
    assert check.mismatch_counts['yoke'] == full.shots
    assert check.mismatched_rows['yoke'] == tuple(int(row) for row in ROWS[:MAX_LISTED_ROWS])
    assert check.matched_rows == 0
    assert check.to_json()['mismatched_rows']['yoke'] == [int(row) for row in ROWS[:MAX_LISTED_ROWS]]


def test_a_subset_row_between_two_full_rows_raises_naming_it(full, subset):
    rows = np.array(subset.rows)
    rows[2] = 9                                             # 8 and 11 are rows; 9 is not
    with pytest.raises(ValueError, match=r'not in the full record.*\b9\b'):
        subset_reproduction(full, dataclasses.replace(subset, rows=rows))


def test_a_subset_row_past_the_last_full_row_raises_naming_it(full, subset):
    rows = np.array(subset.rows)
    rows[-1] = int(ROWS[-1]) + 5
    with pytest.raises(ValueError, match=rf'not in the full record.*\b{int(ROWS[-1]) + 5}\b'):
        subset_reproduction(full, dataclasses.replace(subset, rows=rows))


def test_records_with_different_patch_counts_are_refused(full):
    other = synthetic_record(patches=PATCHES + 1, rows=ROWS)
    with pytest.raises(ValueError, match='patches'):
        subset_reproduction(full, other)


def test_inputs_that_are_not_records_are_refused(full):
    with pytest.raises(TypeError, match='L1Record'):
        subset_reproduction(full, full.arrays())
    with pytest.raises(TypeError, match='L1Record'):
        subset_reproduction(full.arrays(), full)


# --- the check record itself -------------------------------------------------

def consistent_check(**overrides) -> dict:
    """Field values of a failing check over three rows: ``actual`` differs on row 7."""
    fields = dict(
        subset_rows=3, matched_rows=2,
        equal={name: name != 'actual' for name in ARRAY_FIELDS},
        mismatch_counts={name: int(name == 'actual') for name in ARRAY_FIELDS},
        mismatched_rows={name: (7,) if name == 'actual' else () for name in ARRAY_FIELDS},
    )
    fields.update(overrides)
    return fields


def test_a_hand_built_check_reports_its_failure():
    check = SubsetCheck(**consistent_check())
    assert not check.passed and check.matched_rows == 2
    with pytest.raises(ValueError, match=r'actual.*\b7\b'):
        check.raise_if_failed()


@pytest.mark.parametrize('overrides, message', [
    (dict(subset_rows=0), 'subset_rows'),
    (dict(matched_rows=4), 'matched_rows'),
    (dict(matched_rows=3), 'matched_rows'),                 # a mismatch means an unmatched row
    (dict(equal={name: True for name in ARRAY_FIELDS}), 'actual'),
    (dict(mismatch_counts={name: 0 for name in ARRAY_FIELDS}), 'actual'),
    (dict(mismatched_rows={name: () for name in ARRAY_FIELDS}), 'actual'),
    (dict(mismatched_rows={**{name: () for name in ARRAY_FIELDS}, 'actual': (7, 7)}), 'increasing'),
    (dict(mismatched_rows={**{name: () for name in ARRAY_FIELDS}, 'actual': (-1,)}), 'nonnegative'),
    (dict(equal={name: 1 for name in ARRAY_FIELDS}), 'bool'),
    (dict(mismatch_counts={name: 0 for name in ('actual',)}), 'arrays'),
])
def test_an_inconsistent_check_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        SubsetCheck(**consistent_check(**overrides))
