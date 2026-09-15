"""Tests for the per-row L1 evaluation of one chunk of parent rows (_l1.py).

Checks: the record a few dozen distance-3 fixture rows produce, with both correlation
branches exercised; every stored column pinned to an independent recomputation from the
patch's own graph -- the UF reference to ``UnionFindDecoder.decode_batch``, the cluster
gaps and settled states to ``ClusterGapUnionFindDecoder.decode_with_gaps``, and the
matching reference, forced weights and reweighted flag to ``MatchingGaps.forced_weights``
-- together with the demonstration that no transposition of sectors or of patches would
survive those pins; the work formulas under instrumented decoders and matchers; equality
between one call and a partitioned one, for both arrays and work; and the rejection of a
correction that does not reproduce its syndrome, of a detector array of the wrong width,
and of row ids that are not increasing.
"""
import dataclasses
import itertools

import numpy as np
import pytest

from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._fixtures import NUM_PATCHES, yoked_fixture
from yoked.hierarchical._l1 import (
    FORCED_CALLS_PER_DECODE, CollectionWork, L1Context, collect_rows,
)
from yoked.hierarchical._matching_gaps import MatchingGaps
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._record import ARRAY_FIELDS

SHOTS = 48
"""A few dozen distance-3 rows: enough that some patch fires a correlation rule and some
does not, and that every pinned column is compared over many syndromes, while the six
per-patch recomputations below still take a few seconds."""

PINNED_ROWS = 5
"""Rows the per-row decoders are rerun on directly. The batch pins are over every row;
these pins cost one decode each and are the handful the gaps and forced weights use."""


@pytest.fixture(scope='module')
def fixture():
    return yoked_fixture(shots=SHOTS)


@pytest.fixture(scope='module')
def context(fixture) -> L1Context:
    return L1Context(fixture.dem, NUM_PATCHES)


@pytest.fixture(scope='module')
def collected(context, fixture):
    return collect_rows(context, fixture.detectors, fixture.actual, np.arange(SHOTS))


def sectors_of(columns: np.ndarray, patch: int) -> np.ndarray:
    """Columns ``2i`` and ``2i + 1``: sector X then sector Z of patch ``i``."""
    return columns[:, NUM_SECTORS * patch:NUM_SECTORS * (patch + 1)]


# --- what one chunk holds ----------------------------------------------------

def test_a_collected_record_holds_every_row_and_both_correlation_branches(collected, fixture):
    record = collected.record
    assert record.shots == SHOTS and record.num_patches == NUM_PATCHES
    np.testing.assert_array_equal(record.rows, np.arange(SHOTS))
    np.testing.assert_array_equal(record.actual, fixture.actual)
    np.testing.assert_array_equal(record.yoke,
                                  fixture.detectors[:, list(fixture.patches.yoke_detector_ids)])
    assert record.reweighted_patches.any() and not record.reweighted_patches.all()
    assert (record.forced_correlated[~record.reweighted_patches]
            == record.forced_plain[~record.reweighted_patches]).all()


def test_collection_reproduces_one_call_when_partitioned(context, fixture):
    rows = np.arange(SHOTS)
    single = collect_rows(context, fixture.detectors, fixture.actual, rows)
    split = SHOTS // 3
    parts = [collect_rows(context, fixture.detectors[piece], fixture.actual[piece], rows[piece])
             for piece in (slice(None, split), slice(split, None))]
    assert parts[0].work + parts[1].work == single.work
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(
            np.concatenate([getattr(part.record, name) for part in parts]),
            getattr(single.record, name))


# --- the columns each patch owns ---------------------------------------------

def test_the_uf_reference_columns_hold_each_patchs_own_union_find_prediction(collected, fixture):
    """Column ``2i + s`` of ``uf_reference`` is sector ``s`` of patch ``i``, recomputed
    here by the repository's plain UF on the patch's own check-free graph."""
    record = collected.record
    for index, patch in enumerate(fixture.patches):
        local = patch.local_syndromes(fixture.detectors)
        expected = UnionFindDecoder(patch.graph).decode_batch(local)
        np.testing.assert_array_equal(sectors_of(record.uf_reference, index), expected)


def test_the_cluster_gap_columns_hold_each_patchs_own_search(collected, fixture):
    """The gaps and settled-state counts of columns ``2i`` and ``2i + 1``, recomputed by a
    fresh cluster-gap decoder on the patch's own graph."""
    record = collected.record
    for index, patch in enumerate(fixture.patches):
        decoder = ClusterGapUnionFindDecoder(patch.graph)
        local = patch.local_syndromes(fixture.detectors)
        for position in range(PINNED_ROWS):
            result = decoder.decode_with_gaps(local[position])
            np.testing.assert_array_equal(sectors_of(record.cluster_gap, index)[position],
                                          result.cluster_gap)
            np.testing.assert_array_equal(sectors_of(record.dijkstra_states, index)[position],
                                          result.dijkstra_states)


def test_the_matching_columns_hold_each_patchs_own_forced_weights(collected, fixture):
    """The matching reference, the two (2, 2) forced-weight tables, and the reweighted
    flag of patch ``i``, recomputed by a fresh matcher built from the same patch and the
    same correlation rules."""
    record = collected.record
    for index, patch in enumerate(fixture.patches):
        matcher = MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))
        local = patch.local_syndromes(fixture.detectors)
        for position in range(PINNED_ROWS):
            forced = matcher.forced_weights(local[position])
            np.testing.assert_array_equal(sectors_of(record.mwpm_reference, index)[position],
                                          forced.first_pass)
            np.testing.assert_array_equal(sectors_of(record.correlated_prediction, index)[position],
                                          forced.correlated_prediction)
            np.testing.assert_array_equal(record.forced_plain[position, index], forced.plain)
            np.testing.assert_array_equal(record.forced_correlated[position, index],
                                          forced.correlated)
            assert bool(record.reweighted_patches[position, index]) is bool(forced.rules_fired)


def test_no_transposition_of_sectors_or_patches_could_pass_those_pins(collected):
    """The pins above have power only if the stored blocks actually differ.

    Swapping the two sectors of a patch, or the blocks of any two patches, changes the
    cluster-gap columns, so either mistake in ``collect_rows`` fails the pins rather than
    landing on an indistinguishable layout.
    """
    record = collected.record
    gaps = record.cluster_gap.reshape(record.shots, record.num_patches, NUM_SECTORS)
    assert (gaps[:, :, 0] != gaps[:, :, 1]).any()
    for left, right in itertools.combinations(range(record.num_patches), 2):
        assert (gaps[:, left] != gaps[:, right]).any()


# --- the work the chunk cost -------------------------------------------------

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


def test_collection_work_counts_the_calls_that_actually_ran(fixture):
    context = L1Context(fixture.dem, NUM_PATCHES)
    decoders = tuple(CountingDecoder(decoder) for decoder in context.decoders)
    matchers = tuple(CountingMatcher(matcher) for matcher in context.matchers)
    context.decoders, context.matchers = decoders, matchers
    rows = np.arange(12)
    work = collect_rows(context, fixture.detectors[rows], fixture.actual[rows], rows).work
    decodes = len(rows) * NUM_PATCHES
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


# --- what a chunk refuses ----------------------------------------------------

class BrokenDecoder:
    """A decoder whose reported correction no longer reproduces its syndrome."""

    def __init__(self, inner):
        self.inner = inner

    def decode_with_gaps(self, syndrome):
        result = self.inner.decode_with_gaps(syndrome)
        selected = result.selected_edges[:-1] if result.selected_edges else (0,)
        return dataclasses.replace(result, selected_edges=selected)


def test_an_invalid_correction_names_the_parent_row_and_patch(fixture):
    context = L1Context(fixture.dem, NUM_PATCHES)
    context.decoders = context.decoders[:2] + (BrokenDecoder(context.decoders[2]),) \
        + context.decoders[3:]
    rows = np.arange(5, 15)
    with pytest.raises(ValueError, match=r'row 5.*patch 2'):
        collect_rows(context, fixture.detectors[rows], fixture.actual[rows], rows)


def test_collection_rejects_a_detector_array_of_the_wrong_width(context, fixture):
    rows = np.arange(4)
    with pytest.raises(ValueError, match='detector'):
        collect_rows(context, fixture.detectors[rows, :-1], fixture.actual[rows], rows)


def test_collection_rejects_row_ids_that_are_not_increasing(context, fixture):
    rows = np.arange(4)
    with pytest.raises(ValueError, match='increasing'):
        collect_rows(context, fixture.detectors[rows], fixture.actual[rows],
                     np.array([3, 2, 1, 0]))
