"""A full record compared with a subset record of the same sample, row for row.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 3 and 12.

The object this module computes is a ``SubsetCheck``: for a full ``L1Record`` and a
subset ``L1Record`` of the same sampling call, whether every stored array of the subset
equals the full record's array on the same parent rows, exactly. Section 3 says a pilot
subset's shots remain part of their full set and that the full collection reuses the
pilot's L1 outputs only when its provenance still matches; section 12's M2 collects the
remaining rows. This check is what lets M2 say that a full collection made under the
current provenance contract reproduces the pilot's subset, number for number, rather
than merely sharing its identities.

``subset_reproduction`` finds each subset row in the full record by its parent row id
(``np.searchsorted`` followed by an equality check, so that a row absent from the full
record raises rather than being matched to its neighbour) and then compares every array
in ``ARRAY_FIELDS`` at those positions. The comparison is exact: no tolerance, and float
values are compared as values, which is bit for bit on the finite, nonnegative numbers a
record stores. Baselines are deliberately not compared: a subset need not carry them,
and they are gated by ``_baselines.py`` on import rather than reproduced by collection.
Every array's result is reported by name, with the count of differing rows and the
first few parent row ids where it differs, so a failure says what changed and where.

Nothing here decodes: the check reads two records and compares them, so ``_provenance.py``
lists this file in ``CHECK_SOURCES`` rather than ``DECODER_SOURCES``. A change to it
changes what was verified, never what was collected.

``_reproduction_test.py`` checks the module's place among the check sources; a subset
built with ``L1Record.subset`` passing with every array equal and every row matched; the
JSON form; baselines being neither required nor compared; one changed value in any
array reported under that array with its parent row, a one-ulp float change among them;
a row changed on several arrays counted once; the listing bound keeping the full count;
an absent subset row raising by name; mismatched patch counts and non-records refused;
and the check record refusing inconsistent fields.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

# One bound for every diagnostic row list in the package: a failure names enough rows
# to look at and never a hundred thousand of them.
from yoked.hierarchical._collect import MAX_DIAGNOSTIC_ROWS
from yoked.hierarchical._l1 import _whole
from yoked.hierarchical._record import ARRAY_FIELDS, L1Record

MAX_LISTED_ROWS = MAX_DIAGNOSTIC_ROWS
"""Parent row ids listed per differing array. The count of differing rows is always
reported in full; only the listing is bounded."""

COMPARISON = ('exact equality of every stored record array on the subset rows, located in '
              'the full record by parent row id; no tolerance; baselines not compared')
"""What the check compares and how, written into its JSON so that a recorded result
cannot be mistaken for a comparison under a tolerance or one covering the baselines."""


def _whole_count(value, name: str, *, maximum: int | None = None) -> int:
    """A whole, nonnegative count, at most ``maximum`` when one is given."""
    count = _whole(value, name, minimum=0)
    if maximum is not None and count > maximum:
        raise ValueError(f'{name} must be at most {maximum}, got {count}')
    return count


def _by_array(value, name: str) -> dict:
    """The entries of a per-array mapping, in field order, after checking its keys."""
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping by array name, got {type(value).__name__}')
    if set(value) != set(ARRAY_FIELDS):
        raise ValueError(f'{name} must cover exactly the record arrays {list(ARRAY_FIELDS)}, '
                         f'got {sorted(value)}')
    return {array: value[array] for array in ARRAY_FIELDS}


def _flags(value, name: str) -> MappingProxyType:
    """An immutable copy of a mapping from every array name to a bool, in field order."""
    flags = {}
    for array, flag in _by_array(value, name).items():
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f'{name}[{array!r}] must be a bool, got {flag!r}')
        flags[array] = bool(flag)
    return MappingProxyType(flags)


def _row_list(value, name: str) -> tuple[int, ...]:
    """A tuple of whole, nonnegative, strictly increasing parent row ids."""
    rows = []
    for row in value:
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise ValueError(f'{name} must list integer parent rows, got {row!r}')
        if row < 0:
            raise ValueError(f'{name} must list nonnegative parent rows, got {row}')
        rows.append(int(row))
    if any(later <= earlier for earlier, later in zip(rows, rows[1:])):
        raise ValueError(f'{name} must list parent rows in strictly increasing order, got {rows}')
    return tuple(rows)


@dataclass(frozen=True)
class SubsetCheck:
    """What comparing a subset record with its full record found.

    Fields:

    - ``subset_rows``: rows in the subset record (shots), every one of them located in
      the full record; the check raises before this record exists otherwise.
    - ``matched_rows``: subset rows on which every compared array equals the full
      record's, so ``matched_rows == subset_rows`` exactly when the check passed.
    - ``equal``: immutable mapping from each name in ``ARRAY_FIELDS`` to whether that
      array is equal on every subset row.
    - ``mismatch_counts``: the same names to the number of subset rows on which that
      array differs; zero exactly where ``equal`` holds.
    - ``mismatched_rows``: the same names to the first ``MAX_LISTED_ROWS`` parent row ids
      on which that array differs, in increasing order; the count says how many there
      are in all.

    ``passed`` is derived from ``equal`` rather than stored, so it can never disagree
    with the per-array results.
    """
    subset_rows: int
    matched_rows: int
    equal: Mapping[str, bool]
    mismatch_counts: Mapping[str, int]
    mismatched_rows: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        set_field = object.__setattr__
        subset_rows = _whole(self.subset_rows, 'subset_rows', minimum=1)
        matched_rows = _whole_count(self.matched_rows, 'matched_rows', maximum=subset_rows)
        equal = _flags(self.equal, 'equal')
        counts, listed = {}, {}
        for name, count in _by_array(self.mismatch_counts, 'mismatch_counts').items():
            counts[name] = _whole_count(count, f'mismatch_counts[{name!r}]', maximum=subset_rows)
            if equal[name] != (counts[name] == 0):
                raise ValueError(f'{name} is reported {"equal" if equal[name] else "unequal"} '
                                 f'but has {counts[name]} mismatched rows')
        for name, rows in _by_array(self.mismatched_rows, 'mismatched_rows').items():
            listed[name] = _row_list(rows, f'mismatched_rows[{name!r}]')
            expected = min(counts[name], MAX_LISTED_ROWS)
            if len(listed[name]) != expected:
                raise ValueError(f'mismatched_rows[{name!r}] must list the first {expected} of '
                                 f'{counts[name]} mismatched rows, got {len(listed[name])}')
        # A row that differs on any array is unmatched, and a row that differs on several
        # is unmatched once, which bounds the matched count on both sides.
        largest, total = max(counts.values()), sum(counts.values())
        if not subset_rows - total <= matched_rows <= subset_rows - largest:
            raise ValueError(f'matched_rows {matched_rows} is inconsistent with {subset_rows} '
                             f'subset rows and per-array mismatch counts {dict(counts)}')
        set_field(self, 'subset_rows', subset_rows)
        set_field(self, 'matched_rows', matched_rows)
        set_field(self, 'equal', equal)
        set_field(self, 'mismatch_counts', MappingProxyType(counts))
        set_field(self, 'mismatched_rows', MappingProxyType(listed))

    @property
    def passed(self) -> bool:
        """Whether every array is equal on every subset row."""
        return all(self.equal.values())

    def to_json(self) -> dict:
        """The check as plain JSON types, with what was compared written beside the result."""
        return {
            'comparison': COMPARISON,
            'arrays': list(ARRAY_FIELDS),
            'subset_rows': self.subset_rows,
            'matched_rows': self.matched_rows,
            'passed': self.passed,
            'equal': dict(self.equal),
            'mismatch_counts': dict(self.mismatch_counts),
            'mismatched_rows': {name: list(rows) for name, rows in self.mismatched_rows.items()},
            'max_listed_rows': MAX_LISTED_ROWS,
        }

    def raise_if_failed(self, *, recorded_in=None) -> None:
        """Raise a ``ValueError`` naming every differing array, its count, and its first
        parent rows; silent when the check passed. ``recorded_in`` names where the
        check's JSON was written, when it was, so the message says where to look."""
        if self.passed:
            return
        details = [f'{name} differs on {self.mismatch_counts[name]} of {self.subset_rows} rows '
                   f'(parent rows {list(self.mismatched_rows[name])})'
                   for name in ARRAY_FIELDS if not self.equal[name]]
        where = f'; the check is recorded in {recorded_in}' if recorded_in is not None else ''
        raise ValueError(f'the subset does not reproduce the full record: {self.matched_rows} of '
                         f'{self.subset_rows} rows match on every array; ' + '; '.join(details)
                         + where)


# --- the comparison ----------------------------------------------------------

def _require_record(value, name: str) -> L1Record:
    if not isinstance(value, L1Record):
        raise TypeError(f'{name} must be an L1Record, got {type(value).__name__}')
    return value


def _positions_in_full(full: L1Record, subset: L1Record) -> np.ndarray:
    """Where each subset row sits in the full record, by parent row id.

    Both row arrays are increasing and unique, so ``searchsorted`` gives each subset id
    a candidate position; the id there must be the same id, and a position past the end
    means the id is beyond every full row. Any subset row the full record does not hold
    is named, up to the listing bound, and nothing is compared.
    """
    positions = np.searchsorted(full.rows, subset.rows)
    inside = positions < full.shots
    found = np.zeros(subset.shots, dtype=bool)
    found[inside] = full.rows[positions[inside]] == subset.rows[inside]
    if not found.all():
        missing = subset.rows[~found]
        raise ValueError(f'{int(len(missing))} subset rows are not in the full record; parent rows '
                         f'{[int(row) for row in missing[:MAX_LISTED_ROWS]]}')
    return positions


def _differing_rows(full_array: np.ndarray, subset_array: np.ndarray) -> np.ndarray:
    """Which rows differ anywhere across the trailing axes; a ``(rows,)`` bool array.

    Bits and counts compare as values. Floats compare by representation, so distinct
    encodings such as positive and negative zero cannot pass a check described as
    bit-for-bit reproduction. Record construction canonicalizes every float field to
    float64 and rejects NaN, but deliberately accepts either zero sign as nonnegative.
    """
    if full_array.dtype.kind == 'f':
        differs = full_array.view(np.uint8) != subset_array.view(np.uint8)
    else:
        differs = full_array != subset_array
    return differs.reshape(len(differs), -1).any(axis=1)


def subset_reproduction(full: L1Record, subset: L1Record) -> SubsetCheck:
    """Compare a subset record with the full record of the same sample, array by array.

    Every subset row must exist in the full record by parent row id, or the call raises
    naming the rows that do not; both records must have the same number of patches.
    Then every ``ARRAY_FIELDS`` array of the subset is compared with the full record's
    at the located positions, exactly. Baselines are not compared. Returns the
    ``SubsetCheck``; it is the caller's choice whether a failure raises, through
    ``raise_if_failed``, so that the result can be recorded first.
    """
    full = _require_record(full, 'full')
    subset = _require_record(subset, 'subset')
    if full.num_patches != subset.num_patches:
        raise ValueError(f'the full record has {full.num_patches} patches, '
                         f'the subset {subset.num_patches}')
    positions = _positions_in_full(full, subset)
    unmatched = np.zeros(subset.shots, dtype=bool)
    equal, counts, listed = {}, {}, {}
    for name in ARRAY_FIELDS:
        differs = _differing_rows(getattr(full, name)[positions], getattr(subset, name))
        unmatched |= differs
        equal[name] = not differs.any()
        counts[name] = int(differs.sum())
        listed[name] = tuple(int(row) for row in subset.rows[np.flatnonzero(differs)[:MAX_LISTED_ROWS]])
    return SubsetCheck(subset_rows=subset.shots, matched_rows=int((~unmatched).sum()),
                       equal=equal, mismatch_counts=counts, mismatched_rows=listed)
