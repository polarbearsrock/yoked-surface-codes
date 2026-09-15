"""The stored L1 record: every L1 output for a set of parent rows, owned and checked.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 3 and 5.3.

``L1Record`` holds the arrays of section 5.3 for ``shots`` rows of one parent sample,
plus the parent row ids and the per-patch flag saying whether a correlation rule fired
during collection. Column ``2i + s`` holds patch ``i``, sector ``s``; forced class
weights use the ``(shots, P, 2, 2)`` layout indexed ``[c_X, c_Z]`` in nats.
``by_sector`` and ``to_columns`` move between the ``(..., 2P)`` column layout and the
``(..., 2, P)`` sector-major layout the calibrators and L2 work in.

Nothing downstream re-runs a decoder, so every value is checked before it is stored:
bits are 0 or 1 rather than anything that casts to True, state counts are whole and
nonnegative, gaps and weights are finite and nonnegative, and row ids are whole,
nonnegative, unique, and stored in increasing order. Arrays are owned read-only
copies and the baseline mapping is immutable, so neither the caller's input arrays nor
a later alias can change a record; baselines are attached at construction or with
``dataclasses.replace``. ``arrays`` and ``from_arrays`` cross a process boundary as
plain arrays, with the receiving side re-running every check.

Concatenating the records of a full set is not implemented here; it belongs to M2,
where it consumes verified ``LoadedRecord`` values, requires equal parent sample,
model, decoder, and role identities together with identical baseline columns, rejects
overlapping rows, and keeps the parent row ids sorted.

``_record_test.py`` checks ownership of every field and baseline, each invalid-value
case, the column/sector conversions, subsets, the plain-array round trip, and the
schema-checked array container.
"""
from __future__ import annotations

import dataclasses
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import SCHEMA_VERSION, atomic_replacement

RECORD_SCHEMA = f'L1Record/{SCHEMA_VERSION}'
"""Stored inside ``record.npz``. One string covers the field names, dtypes, and column
layout, so a container written under an older layout is rejected rather than
reinterpreted."""

SCHEMA_KEY = '__schema__'
"""Entry name carrying the schema string. The dunder form cannot collide with a record
field name or a baseline column."""

BASELINE_PREFIX = 'baseline_'
"""Prefix for optional historical baselines in the flat array container, so that a
record needs no nested structure and therefore no pickle."""

REFERENCE_NAMES = ('uf', 'mwpm')
"""The two reference decoders an estimator can be defined against (spec section 6)."""

IDENTITY_NAMES = ('model', 'parent_sample', 'sampling_family', 'decoder', 'collection')
"""The identities a completed record carries; a loader that cannot name all five has
not verified what it is about to hand downstream."""

_NUMERIC_KINDS = 'buif'
"""NumPy dtype kinds a caller may offer for a bit array: bool, unsigned, signed, float.
Anything else (object, string, datetime) is a caller error, not something to cast."""


# --- value checks, all of which run before any dtype cast --------------------

def _as_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f'{name} must be numeric, got dtype {array.dtype}')
    return array


def _require_shape(array: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {array.shape}')


def _bits(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """An owned read-only bool array. A value of 0.5 is rejected, never cast to True."""
    array = _as_array(value, name)
    _require_shape(array, name, shape)
    # A bool array is already binary; checking one costs a full pass over the sample.
    if array.dtype.kind != 'b' and not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain only 0 and 1')
    return readonly_array(array, dtype=bool)


def _nonnegative_values(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """An owned read-only float64 array of finite, nonnegative values (nats)."""
    array = _as_array(value, name)
    _require_shape(array, name, shape)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    if (array < 0).any():
        raise ValueError(f'{name} must be nonnegative')
    return readonly_array(array, dtype=np.float64)


def _counts(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """An owned read-only int64 array of whole, nonnegative counts."""
    array = np.asarray(value)
    # A count is not a flag: bool would pass every check below and mean something else.
    if array.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be an integer or float count array, got dtype {array.dtype}')
    _require_shape(array, name, shape)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    if (array < 0).any():
        raise ValueError(f'{name} must be nonnegative')
    if array.dtype.kind == 'f' and not (array == np.floor(array)).all():
        raise ValueError(f'{name} must contain whole numbers')
    return readonly_array(array, dtype=np.int64)


def _row_ids(value, shots: int) -> np.ndarray:
    """An owned read-only int64 array of the parent row ids these shots came from."""
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise ValueError(f'rows must be an integer array of parent row ids, got dtype {array.dtype}')
    _require_shape(array, 'rows', (shots,))
    if not np.isfinite(array).all():
        raise ValueError('rows must be finite')
    if array.dtype.kind == 'f' and not (array == np.floor(array)).all():
        raise ValueError('rows must be whole numbers')
    if (array < 0).any():
        raise ValueError('rows must be nonnegative')
    steps = np.diff(array)
    if (steps == 0).any():
        raise ValueError('rows must be unique')
    if (steps < 0).any():
        raise ValueError('rows must be stored in increasing order')
    return readonly_array(array, dtype=np.int64)


def _baselines(value, shape: tuple[int, ...]) -> MappingProxyType:
    """An immutable mapping over a private dict of owned read-only bool arrays."""
    if not isinstance(value, Mapping):
        raise TypeError(f'baselines must be a mapping of name to prediction array, got {type(value).__name__}')
    stored: dict[str, np.ndarray] = {}
    for name, array in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError('baseline names must be nonempty strings')
        stored[name] = _bits(array, f'baseline {name!r}', shape)
    return MappingProxyType(stored)


def _positions(value, shots: int) -> np.ndarray:
    """Validated positions into a record's own rows, used by ``subset``."""
    array = np.asarray(value)
    if array.dtype.kind == 'b':
        raise ValueError('subset takes positions, not a boolean mask; use np.flatnonzero(mask)')
    if array.dtype.kind not in 'iu' or array.ndim != 1:
        raise ValueError(f'positions must be a one-dimensional integer array, got {array.dtype} {array.shape}')
    if len(array) == 0:
        raise ValueError('a subset needs at least one position')
    if (array < 0).any() or (array >= shots).any():
        raise ValueError(f'positions must lie in [0, {shots})')
    if not (np.diff(array) > 0).all():
        raise ValueError('positions must be unique and increasing, so that rows stay increasing')
    return array


# --- the record --------------------------------------------------------------

@dataclass(frozen=True)
class L1Record:
    """Every L1 output for ``shots`` rows of one parent sample, with P patches.

    Fields, all owned and read-only:

    - ``actual`` (shots, 2P) bool: sampled observable flips.
    - ``yoke`` (shots, 2) bool: sampled yoke bits, X then Z.
    - ``uf_reference`` (shots, 2P) bool: the UF reference prediction.
    - ``mwpm_reference`` (shots, 2P) bool: the plain patch-local MWPM prediction, the
      unforced first pass.
    - ``correlated_prediction`` (shots, 2P) bool: the unforced second pass under the
      reweighted model.
    - ``joint_mwpm`` (shots, 2P) bool: joint PyMatching on the hub DEM, for validation.
    - ``cluster_gap`` (shots, 2P) float64: the UF cluster gap in nats, finite and
      nonnegative.
    - ``dijkstra_states`` (shots, 2P) int64: settled states of the parity-augmented
      search, a nonnegative cost proxy.
    - ``forced_plain`` (shots, P, 2, 2) float64: W(c_X, c_Z) in nats, indexed
      [c_X, c_Z], finite and nonnegative.
    - ``forced_correlated`` (shots, P, 2, 2) float64: the same under the reweighted model.
    - ``reweighted_patches`` (shots, P) bool: whether a correlation rule fired, which is
      what the correlated collection work is counted against.
    - ``rows`` (shots,) int64: the parent sample row of each shot, unique and increasing.
    - ``baselines``: optional historical predictions, each an owned read-only
      (shots, 2P) bool array in an immutable mapping; empty by default and attached
      with ``dataclasses.replace``, never by mutation.
    """
    actual: np.ndarray
    yoke: np.ndarray
    uf_reference: np.ndarray
    mwpm_reference: np.ndarray
    correlated_prediction: np.ndarray
    joint_mwpm: np.ndarray
    cluster_gap: np.ndarray
    dijkstra_states: np.ndarray
    forced_plain: np.ndarray
    forced_correlated: np.ndarray
    reweighted_patches: np.ndarray
    rows: np.ndarray
    baselines: Mapping[str, np.ndarray] = MappingProxyType({})

    def __post_init__(self) -> None:
        actual = _as_array(self.actual, 'actual')
        if actual.ndim != 2 or actual.shape[0] < 1 or actual.shape[1] < NUM_SECTORS \
                or actual.shape[1] % NUM_SECTORS:
            raise ValueError(f'actual must have shape (shots >= 1, {NUM_SECTORS} * patches), got {actual.shape}')
        shots, columns = actual.shape
        patches = columns // NUM_SECTORS
        forced_shape = (shots, patches, NUM_SECTORS, NUM_SECTORS)
        set_field = object.__setattr__
        for name in ('actual', 'uf_reference', 'mwpm_reference', 'correlated_prediction', 'joint_mwpm'):
            set_field(self, name, _bits(getattr(self, name), name, (shots, columns)))
        set_field(self, 'yoke', _bits(self.yoke, 'yoke', (shots, NUM_SECTORS)))
        set_field(self, 'reweighted_patches', _bits(self.reweighted_patches, 'reweighted_patches', (shots, patches)))
        set_field(self, 'cluster_gap', _nonnegative_values(self.cluster_gap, 'cluster_gap', (shots, columns)))
        set_field(self, 'dijkstra_states', _counts(self.dijkstra_states, 'dijkstra_states', (shots, columns)))
        for name in ('forced_plain', 'forced_correlated'):
            set_field(self, name, _nonnegative_values(getattr(self, name), name, forced_shape))
        set_field(self, 'rows', _row_ids(self.rows, shots))
        set_field(self, 'baselines', _baselines(self.baselines, (shots, columns)))

    @property
    def shots(self) -> int:
        """Number of stored rows."""
        return int(self.actual.shape[0])

    @property
    def num_patches(self) -> int:
        """Number of patches P; the record has 2P columns."""
        return int(self.actual.shape[1] // NUM_SECTORS)

    def reference(self, name: str) -> np.ndarray:
        """The named reference decoder's (shots, 2P) prediction."""
        if name not in REFERENCE_NAMES:
            raise ValueError(f'reference must be one of {REFERENCE_NAMES}, got {name!r}')
        return self.uf_reference if name == 'uf' else self.mwpm_reference

    def subset(self, positions) -> L1Record:
        """A new record holding the rows at ``positions``, with their parent row ids.

        ``positions`` index this record, not the parent sample, and must be unique and
        increasing so that the subset's parent row ids stay increasing too.
        """
        positions = _positions(positions, self.shots)
        fields = {name: getattr(self, name)[positions] for name in ARRAY_FIELDS}
        baselines = {name: array[positions] for name, array in self.baselines.items()}
        return L1Record(**fields, baselines=baselines)

    def arrays(self) -> dict[str, np.ndarray]:
        """Every stored array by name, baselines under ``BASELINE_PREFIX``.

        This is the plain payload that crosses a process or file boundary; the checked
        record is rebuilt with ``from_arrays``.
        """
        payload = {name: getattr(self, name) for name in ARRAY_FIELDS}
        for name, array in self.baselines.items():
            payload[BASELINE_PREFIX + name] = array
        return payload

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> L1Record:
        """Rebuild a record from an ``arrays`` payload, re-running every check."""
        fields: dict[str, np.ndarray] = {}
        baselines: dict[str, np.ndarray] = {}
        for key, array in arrays.items():
            if key in ARRAY_FIELDS:
                fields[key] = array
            elif key.startswith(BASELINE_PREFIX):
                baselines[key[len(BASELINE_PREFIX):]] = array
            else:
                raise ValueError(f'Unknown record array {key!r}')
        missing = [name for name in ARRAY_FIELDS if name not in fields]
        if missing:
            raise ValueError(f'Missing record arrays: {missing}')
        return cls(**fields, baselines=baselines)


ARRAY_FIELDS = tuple(field.name for field in dataclasses.fields(L1Record) if field.name != 'baselines')
"""The stored array fields in declaration order; ``baselines`` is a mapping, not one."""


# --- column and sector layouts -----------------------------------------------

def by_sector(columns) -> np.ndarray:
    """Convert (..., 2P) column layout to (..., 2, P) sector-major layout.

    Column ``2i + s`` becomes ``[..., s, i]``. The result is a new owned array, so
    writing to it never reaches the record it came from.
    """
    array = np.asarray(columns)
    if array.ndim < 1 or array.shape[-1] % NUM_SECTORS:
        raise ValueError(f'Expected a trailing axis of {NUM_SECTORS} * patches, got shape {array.shape}')
    patches = array.shape[-1] // NUM_SECTORS
    reshaped = array.reshape(*array.shape[:-1], patches, NUM_SECTORS)
    return np.ascontiguousarray(reshaped.swapaxes(-1, -2))


def to_columns(sectors) -> np.ndarray:
    """Convert (..., 2, P) sector-major layout back to (..., 2P) column layout."""
    array = np.asarray(sectors)
    if array.ndim < 2 or array.shape[-2] != NUM_SECTORS:
        raise ValueError(f'Expected axes (..., {NUM_SECTORS}, patches), got shape {array.shape}')
    patches = array.shape[-1]
    swapped = np.ascontiguousarray(array.swapaxes(-1, -2))
    return swapped.reshape(*array.shape[:-2], NUM_SECTORS * patches)


# --- the array container -----------------------------------------------------

def _save_arrays(path, arrays: Mapping[str, np.ndarray], *, schema: str) -> None:
    """Write an ``.npz`` of named arrays plus its schema, atomically.

    The schema travels as a NumPy string entry inside the container, so a file can
    always say what layout it holds without a sidecar.
    """
    if SCHEMA_KEY in arrays:
        raise ValueError(f'{SCHEMA_KEY} is reserved for the schema string')
    payload = dict(arrays)
    payload[SCHEMA_KEY] = np.array(schema)
    with atomic_replacement(path) as temporary:
        with open(temporary, 'wb') as handle:
            np.savez(handle, **payload)


def _load_arrays(path, *, schema: str) -> dict[str, np.ndarray]:
    """Read an ``.npz`` written by ``_save_arrays``, requiring the expected schema.

    Pickled objects are refused, and a container that is unreadable, truncated, or
    written under another schema raises instead of yielding partial arrays.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as container:
            entries = {name: container[name] for name in container.files}
    except FileNotFoundError:
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f'{path} is not a readable array container') from exc
    stored = entries.pop(SCHEMA_KEY, None)
    if stored is None:
        raise ValueError(f'{path} has no {SCHEMA_KEY} entry, so its schema is unknown')
    if str(stored) != schema:
        raise ValueError(f'{path} holds schema {str(stored)!r}, expected {schema!r}')
    return entries


# --- a record as it was loaded -----------------------------------------------

def _frozen(value):
    """A recursively immutable view of decoded JSON: mappings become mapping proxies
    and sequences become tuples, so a manifest cannot be edited after verification."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class LoadedRecord:
    """A verified record, where it was read from, and what it is.

    Fields: ``record`` the checked ``L1Record``; ``directory`` the record directory it
    was read from, for diagnostics only, since no identity depends on a path;
    ``identities`` an immutable mapping of named identity hashes, carrying at least
    ``IDENTITY_NAMES``; ``manifest`` an immutable view of the completion manifest, in
    which nested mappings are mapping proxies and JSON arrays are tuples.

    Task 9's ``load_record`` is the only thing that constructs one, after verifying the
    manifest, the artifact hashes, and the recorded checks.
    """
    record: L1Record
    directory: Path
    identities: Mapping[str, str]
    manifest: Mapping

    def __post_init__(self) -> None:
        if not isinstance(self.record, L1Record):
            raise TypeError(f'record must be an L1Record, got {type(self.record).__name__}')
        if not isinstance(self.manifest, Mapping):
            raise TypeError(f'manifest must be a mapping, got {type(self.manifest).__name__}')
        identities = dict(self.identities)
        missing = [name for name in IDENTITY_NAMES if name not in identities]
        if missing:
            raise ValueError(f'Missing identities: {missing}')
        for name, value in identities.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f'identity {name!r} must be a nonempty string')
        object.__setattr__(self, 'directory', Path(self.directory))
        object.__setattr__(self, 'identities', MappingProxyType(identities))
        object.__setattr__(self, 'manifest', _frozen(self.manifest))
