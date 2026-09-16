"""The stored L1 record, and the verified loader every downstream stage reads it through.

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

``load_record`` is the only public way to read a completed collection. A record
directory means nothing until its completion manifest exists: the loader requires the
manifest's schema and ``status``, that its recorded checks passed, that every declared
artifact still hashes to what was published, that the stored row ids match the
manifest's row summary exactly, and that the recorded collection identity follows from
the manifest's own fields. It also derives the model, parent-sample, and sampling-family
identities from canonical inputs embedded by collection, rather than accepting those
labels on trust. Every failure is a ``ValueError`` naming what failed, and a missing
manifest is one of them: an orphan ``record.npz`` is an interrupted publication, not a
result.

Concatenating the records of a full set is not implemented here; it belongs to M2,
where it consumes verified ``LoadedRecord`` values, requires equal parent sample,
model, decoder, and role identities together with identical baseline columns, rejects
overlapping rows, and keeps the parent row ids sorted.

``_record_test.py`` checks ownership of every field and baseline, each invalid-value
case, the column/sector conversions, subsets, the plain-array round trip, the
schema-checked array container, and every way a published directory can fail to load:
a missing or incomplete manifest, failed checks, another schema, a changed or corrupt
record file, a declared artifact that is absent or outside the directory, a row summary
the record does not have, and a collection identity that does not follow.
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
from yoked.hierarchical._provenance import (
    RECORD_CONVENTIONS, SCHEMA_VERSION, atomic_replacement, canonical_json,
    collection_identity, decoder_identity, read_json, row_ids_sha256, sample_identities,
    sha256_bytes, sha256_file,
)

RECORD_FILE = 'record.npz'
"""The stored arrays of a completed collection, inside its record directory."""

RECORD_MANIFEST = 'manifest.json'
"""The completion manifest, published after every other artifact. A directory without
it holds no completed collection, whatever else it contains."""

COMPLETE_STATUS = 'complete'
"""The only status a loader accepts. A stage writes it once, last, and never writes any
other value: an unfinished collection has no manifest at all."""

ROW_SUMMARY_FIELDS = ('count', 'start', 'stop', 'sha256')
"""What a manifest says about the rows a record holds; ``sha256`` is the only one of the
four that identifies them exactly."""

MANIFEST_FIELDS = (
    'schema_version', 'status', 'role', 'parameters', 'seed', 'parent_shots', 'shots',
    'rows', 'parent_payload_sha256', 'sample_identity_inputs', 'identities', 'artifacts',
    'checks', 'versions', 'source_sha256',
)
"""The fields ``load_record`` verifies. Work, timing, and audit metadata are carried
through to the caller without being interpreted here."""

CHECK_FIELDS = ('passed', 'identity', 'graph', 'record')
"""The check block a completed manifest carries: whether both gates passed, the identity
of the validation code that ran them, and each gate's serialized record."""

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

BASELINE_AUDIT_SCHEMA = f'BaselineAttachment/{SCHEMA_VERSION}'
"""Schema of the manifest block binding imported predictions to their audit trail."""

BASELINE_AUDIT_FIELD = 'baselines'
BASELINE_AUDIT_HASH = 'attachment_sha256'
"""The optional manifest block and its canonical integrity digest."""

REFERENCE_NAMES = ('uf', 'mwpm')
"""The two reference decoders an estimator can be defined against (spec section 6)."""

IDENTITY_NAMES = ('model', 'parent_sample', 'sampling_family', 'decoder', 'collection')
"""The identities a completed record carries; a loader that cannot name all five has
not verified what it is about to hand downstream."""

_NUMERIC_KINDS = 'buif'
"""NumPy dtype kinds a caller may offer for a bit array: bool, unsigned, signed, float.
Anything else (object, string, datetime) is a caller error, not something to cast."""

SAMPLE_IDENTITY_INPUT_FIELDS = (
    'parameters', 'circuit_sha256', 'dem_sha256', 'num_detectors', 'num_observables',
    'seed', 'parent_shots', 'payload_sha256', 'model_versions', 'sampling_versions',
)
"""Canonical saved inputs needed to recompute a record's three sample identities."""


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


def row_summary(rows) -> dict:
    """What a manifest records about a set of parent row ids.

    ``count`` is how many there are, ``start`` and ``stop`` bound them as a half-open
    range, and ``sha256`` is ``row_ids_sha256`` of the exact ordered ids. The ids need
    not be contiguous, so only the hash identifies them: ``start`` and ``stop`` are
    there to read, and are meaningful because a record's rows are always increasing.
    """
    array = np.asarray(rows)
    if array.dtype.kind not in 'iu' or array.ndim != 1 or len(array) == 0:
        raise ValueError(f'rows must be a nonempty one-dimensional integer array, '
                         f'got {array.dtype} {array.shape}')
    return {'count': int(len(array)), 'start': int(array[0]), 'stop': int(array[-1]) + 1,
            'sha256': row_ids_sha256(array)}


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


# --- the verified loader -----------------------------------------------------

def _require_fields(mapping, fields, where: str) -> None:
    """Every field a manifest must declare; a missing one is malformed, not a KeyError."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f'{where} must hold an object, got {type(mapping).__name__}')
    missing = [name for name in fields if name not in mapping]
    if missing:
        raise ValueError(f'{where} declares no {missing}')


def _artifact_path(directory: Path, name) -> Path:
    """The path of one declared artifact, which must be a plain name inside the directory.

    A manifest is data, and data that can name ``../something`` would let a loader hash
    and open a file the collection never published.
    """
    if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError(f'artifact name {name!r} must be a plain filename inside {directory}')
    return directory / name


def _verify_artifacts(directory: Path, artifacts) -> None:
    """Every published file must still hash to what the manifest declared."""
    _require_fields(artifacts, (RECORD_FILE,), f'{RECORD_MANIFEST} artifacts')
    for name, declared in artifacts.items():
        path = _artifact_path(directory, name)
        if not path.is_file():
            raise ValueError(f'{RECORD_MANIFEST} declares the artifact {name!r}, which is missing')
        digest = sha256_file(path)
        if digest != declared:
            raise ValueError(f'{name} hashes to {digest}, {RECORD_MANIFEST} declares {declared!r}')


def _verify_rows(record: L1Record, manifest: Mapping) -> dict:
    """The stored row ids must be exactly the ones the manifest summarizes."""
    declared = manifest['rows']
    _require_fields(declared, ROW_SUMMARY_FIELDS, f'{RECORD_MANIFEST} rows')
    summary = row_summary(record.rows)
    for name in ROW_SUMMARY_FIELDS:
        if declared[name] != summary[name]:
            raise ValueError(f'{RECORD_FILE} holds rows with {name} {summary[name]!r}, '
                             f'{RECORD_MANIFEST} declares {declared[name]!r}')
    if manifest['shots'] != record.shots:
        raise ValueError(f'{RECORD_FILE} holds {record.shots} shots, '
                         f'{RECORD_MANIFEST} declares {manifest["shots"]!r}')
    return summary


def baseline_prediction_sha256(prediction) -> str:
    """Hash prediction bits using the recorded runs' little-endian packed convention."""
    array = np.asarray(prediction, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f'a baseline prediction must be two-dimensional, got {array.shape}')
    return sha256_bytes(np.ascontiguousarray(np.packbits(array, axis=1, bitorder='little')))


def baseline_attachment_sha256(block: Mapping) -> str:
    """Canonical digest of a complete baseline audit block, excluding the digest itself."""
    payload = dict(block)
    payload.pop(BASELINE_AUDIT_HASH, None)
    return sha256_bytes(canonical_json(payload).encode('utf-8'))


def _verify_baseline_audit(record: L1Record, manifest: Mapping, *, allow_legacy: bool) -> bool:
    """Bind every stored baseline to the import provenance that admitted it.

    Returns true only for the old M2 block that an explicit, verified re-import may
    upgrade. Public loading never permits that block.
    """
    if not record.baselines:
        if BASELINE_AUDIT_FIELD in manifest:
            raise ValueError(f'{RECORD_MANIFEST} declares baselines but {RECORD_FILE} holds none')
        return False
    if BASELINE_AUDIT_FIELD not in manifest:
        raise ValueError(f'{RECORD_FILE} holds baselines but {RECORD_MANIFEST} declares no '
                         f'{BASELINE_AUDIT_FIELD!r} audit block')
    block = manifest[BASELINE_AUDIT_FIELD]
    _require_fields(block, ('names', 'decoders'), f'{RECORD_MANIFEST} baselines')
    legacy = 'schema_version' not in block and BASELINE_AUDIT_HASH not in block
    if legacy:
        if allow_legacy:
            return True
        raise ValueError(f'{RECORD_MANIFEST} holds a legacy baseline audit block; re-run '
                         'import-baselines with the original recorded run to verify and upgrade it')
    _require_fields(block, ('schema_version', BASELINE_AUDIT_HASH, 'imported_utc', 'importer',
                            'row_mapping', 'run', 'yoke_parity', 'joint_mwpm'),
                    f'{RECORD_MANIFEST} baselines')
    if block['schema_version'] != BASELINE_AUDIT_SCHEMA:
        raise ValueError(f'{RECORD_MANIFEST} baselines holds schema {block["schema_version"]!r}, '
                         f'expected {BASELINE_AUDIT_SCHEMA!r}')
    names = block['names']
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names) \
            or len(set(names)) != len(names):
        raise ValueError(f'{RECORD_MANIFEST} baselines.names must be distinct nonempty strings')
    if tuple(names) != tuple(record.baselines):
        raise ValueError(f'{RECORD_MANIFEST} baselines names {names!r} do not match '
                         f'{RECORD_FILE} baselines {list(record.baselines)!r}')
    decoders = block['decoders']
    if not isinstance(decoders, Mapping) or set(decoders) != set(names):
        raise ValueError(f'{RECORD_MANIFEST} baselines.decoders must name exactly {names!r}')
    for name in names:
        entry = decoders[name]
        _require_fields(entry, ('mapped_prediction_sha256',),
                        f'{RECORD_MANIFEST} baselines.decoders.{name}')
        expected = baseline_prediction_sha256(record.baselines[name])
        if entry['mapped_prediction_sha256'] != expected:
            raise ValueError(f'{RECORD_MANIFEST} baseline {name!r} declares mapped prediction hash '
                             f'{entry["mapped_prediction_sha256"]!r}, {RECORD_FILE} holds {expected}')
    expected = baseline_attachment_sha256(block)
    if block[BASELINE_AUDIT_HASH] != expected:
        raise ValueError(f'{RECORD_MANIFEST} baseline attachment hashes to {expected}, '
                         f'declares {block[BASELINE_AUDIT_HASH]!r}')
    return False


def _load_record(record_dir, *, allow_legacy_baselines: bool = False) -> LoadedRecord:
    """Open a completed collection, verifying everything before exposing its arrays.

    This is the only public way to read a record: the manifest must exist, declare this
    schema and ``status='complete'``, and report checks that passed; every declared
    artifact must still hash to its published value; the stored arrays must pass every
    ``L1Record`` check; the row ids must match the manifest's summary exactly; and the
    three sample identities must follow from the manifest's canonical sample inputs.
    The experiment parameters must agree with those inputs, the decoder identity must
    follow from its recorded sources and versions, and the collection identity must
    follow from its parent sample, decoder, role, shot count, and row hash. Anything
    else raises a ``ValueError`` naming what failed, including a missing manifest,
    which means the collection is incomplete rather than damaged.
    """
    directory = Path(record_dir)
    manifest_path = directory / RECORD_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f'{manifest_path} is missing, so {directory} holds no completed collection')
    manifest = read_json(manifest_path)
    _require_fields(manifest, MANIFEST_FIELDS, str(manifest_path))
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'{manifest_path} holds schema {manifest["schema_version"]!r}, '
                         f'expected {SCHEMA_VERSION!r}')
    if manifest['status'] != COMPLETE_STATUS:
        raise ValueError(f'{manifest_path} declares status {manifest["status"]!r}, '
                         f'expected {COMPLETE_STATUS!r}')
    _require_fields(manifest['checks'], CHECK_FIELDS, f'{RECORD_MANIFEST} checks')
    # `is not True` rather than a truth test: a manifest is decoded JSON, and a non-empty
    # string or list must not be able to stand in for a gate that passed.
    if manifest['checks']['passed'] is not True:
        raise ValueError(f'{manifest_path} records checks that did not pass')
    _verify_artifacts(directory, manifest['artifacts'])

    record = L1Record.from_arrays(_load_arrays(directory / RECORD_FILE, schema=RECORD_SCHEMA))
    summary = _verify_rows(record, manifest)
    identities = manifest['identities']
    _require_fields(identities, IDENTITY_NAMES, f'{RECORD_MANIFEST} identities')
    sample_inputs = manifest['sample_identity_inputs']
    _require_fields(sample_inputs, SAMPLE_IDENTITY_INPUT_FIELDS,
                    f'{RECORD_MANIFEST} sample_identity_inputs')
    expected_sample = sample_identities(**{
        name: sample_inputs[name] for name in SAMPLE_IDENTITY_INPUT_FIELDS
    })
    for name, expected_identity in expected_sample.items():
        if identities[name] != expected_identity:
            raise ValueError(f'The {name} identity {identities[name]!r} in {manifest_path} '
                             'does not follow from its saved sample identity inputs')
    for flat_name, input_name in (
            ('parameters', 'parameters'), ('seed', 'seed'),
            ('parent_shots', 'parent_shots'),
            ('parent_payload_sha256', 'payload_sha256')):
        if manifest[flat_name] != sample_inputs[input_name]:
            raise ValueError(f'{flat_name} in {manifest_path} does not agree with '
                             f'sample_identity_inputs.{input_name}')
    _require_fields(manifest['versions'], ('decoder',), f'{RECORD_MANIFEST} versions')
    _require_fields(manifest['source_sha256'], ('decoder',),
                    f'{RECORD_MANIFEST} source_sha256')
    expected_decoder = decoder_identity(
        sources=manifest['source_sha256']['decoder'], conventions=RECORD_CONVENTIONS,
        versions=manifest['versions']['decoder'])
    if identities['decoder'] != expected_decoder:
        raise ValueError(f'The decoder identity {identities["decoder"]!r} in {manifest_path} '
                         'does not follow from its saved decoder sources and versions')
    # Recomputed rather than trusted: a record whose declared collection identity does
    # not follow from its own role, rows, parent, and decoder is not that collection.
    expected = collection_identity(
        parent_sample=identities['parent_sample'], decoder=identities['decoder'],
        role=manifest['role'], shots=record.shots, rows_sha256=summary['sha256'])
    if expected != identities['collection']:
        raise ValueError(f'The collection identity {identities["collection"]!r} in {manifest_path} '
                         f'does not follow from its role, rows, parent sample, and decoder')
    _verify_baseline_audit(record, manifest, allow_legacy=allow_legacy_baselines)
    return LoadedRecord(record=record, directory=directory,
                        identities={name: identities[name] for name in IDENTITY_NAMES},
                        manifest=manifest)


def load_record(record_dir) -> LoadedRecord:
    """Open and fully verify a completed collection, including imported baselines."""
    return _load_record(record_dir)
