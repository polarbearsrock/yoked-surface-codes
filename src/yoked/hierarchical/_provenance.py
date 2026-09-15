"""Content identities of the experiment's artifacts, and the small I/O they need.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 3 and 11.

An identity is the SHA-256 of the canonical JSON of everything that determines an
artifact: parameters, file and payload hashes, source hashes, and package versions,
tagged with a schema version and the kind of identity. A path is not an identity, so
none of these functions reads a file or looks at a directory: a subset saved somewhere
else keeps its parent sample identity, and two same-seed sampling calls of different
lengths share a sampling family but not a parent sample. Sources are grouped by what
they determine, so that editing a policy or a plot cannot invalidate collected L1
outputs, while editing an L1 algorithm does.

Sampled payloads keep the hash convention of the recorded four-decoder runs: the
packed detector bytes followed by the packed observable bytes, little bit order, with
no ``.npy`` headers between them.

``_provenance_test.py`` checks that convention against a hand-packed example, the
canonical JSON and undefined-number rules, atomic JSON writes, the refusal to omit a
missing required source file, and which inputs each identity is sensitive to.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

SCHEMA_VERSION = 'hierarchical-l1-l2/1'
"""One version covers the stored record layout and the identity input schema: a change
to either makes every stored identity incomparable, and a single string makes that
impossible to overlook."""

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
"""src/yoked/hierarchical/_provenance.py sits three directories below the repository root."""

_FILE_CHUNK_BYTES = 1 << 20
"""File hashing reads a megabyte at a time: large enough to keep syscalls rare, small
enough that hashing a 200 MB sample never holds a second copy of it."""

_HASH_CHUNK_ROWS = 4096
"""Rows of a packed sample hashed per update. At d=9 a row is about 2.2 kB, so a chunk
is under 10 MB and a 100,000-shot payload is never copied whole."""

_GIT_TIMEOUT_SECONDS = 10
"""``git rev-parse`` is a local lookup; anything slower means git is unusable here and
the commit is recorded as unknown rather than stalling a stage."""


# --- hashing -----------------------------------------------------------------

def sha256_bytes(data) -> str:
    """Hex SHA-256 of any bytes-like object, including a C-contiguous NumPy array."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    """Hex SHA-256 of a file's exact bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(_FILE_CHUNK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def packed_sample_hash(detectors_packed, actual_packed) -> str:
    """Hex SHA-256 of the packed detector bytes followed by the packed observable bytes.

    Both arguments are two-dimensional bit-packed ``uint8`` arrays as Stim returns them
    with ``bit_packed=True``. Only the payload bytes are hashed, in row-major order and
    without the ``.npy`` headers that surround them on disk, so the digest identifies
    the sample itself rather than a particular file.
    """
    digest = hashlib.sha256()
    for array in (detectors_packed, actual_packed):
        _update_with_packed(digest, array)
    return digest.hexdigest()


def _update_with_packed(digest, array) -> None:
    """Stream one packed payload into ``digest`` a bounded number of rows at a time."""
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim != 2:
        raise ValueError(f'Expected a two-dimensional bit-packed uint8 array, got {array.dtype} {array.shape}')
    for start in range(0, len(array), _HASH_CHUNK_ROWS):
        # ascontiguousarray returns the slice itself when it is already C-ordered, so a
        # memory-mapped sample is read once and never copied whole.
        digest.update(np.ascontiguousarray(array[start:start + _HASH_CHUNK_ROWS]))


def row_ids_sha256(rows) -> str:
    """Hex SHA-256 of the exact ordered parent row ids, as little-endian int64 bytes.

    The byte order is explicit so that the digest identifies the ids themselves rather
    than the architecture that wrote them.
    """
    array = np.asarray(rows)
    if array.ndim != 1 or array.dtype.kind not in 'iu':
        raise ValueError(f'Expected a one-dimensional integer array of row ids, got {array.dtype} {array.shape}')
    return sha256_bytes(np.ascontiguousarray(array, dtype='<i8'))


# --- JSON --------------------------------------------------------------------

def canonical_json(value) -> str:
    """The one serialization identities are computed from: sorted keys, no optional
    whitespace, and no nonstandard ``NaN``/``Infinity`` literals.

    Undefined numbers raise here rather than being written: inside an identity they
    would make distinct inputs indistinguishable.
    """
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


def json_ready(value):
    """Convert a value to plain JSON types, with undefined numbers becoming ``None``.

    An undefined statistic (an empty stratum's rate, an interval that needs one) is
    written as ``null`` beside its counts, never as a nonstandard ``NaN`` literal.
    NumPy scalars become their Python equivalents.
    """
    return _plain_json(value, replace_undefined=True)


def _plain_json(value, *, replace_undefined: bool):
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item, replace_undefined=replace_undefined) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, replace_undefined=replace_undefined) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if replace_undefined and not math.isfinite(number) else number
    return value


def _reject_constant(name: str):
    raise ValueError(f'Refusing to read the nonstandard JSON literal {name}')


def read_json(path):
    """Read a JSON artifact, rejecting nonstandard ``NaN``/``Infinity`` literals."""
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_constant)


def write_json_atomic(path, value) -> None:
    """Write a JSON artifact through a temporary sibling and one atomic replace.

    Keys are sorted and the text is indented, so the file is both readable and
    reproducible: the same value always writes the same bytes and therefore the same
    artifact hash.
    """
    text = json.dumps(json_ready(value), sort_keys=True, indent=2, allow_nan=False) + '\n'
    with atomic_replacement(path) as temporary:
        temporary.write_bytes(text.encode('utf-8'))


@contextlib.contextmanager
def atomic_replacement(path) -> Iterator[Path]:
    """Yield a temporary sibling of ``path``; on a clean exit it replaces ``path``.

    The sibling lives in the destination directory so that the replacement is a rename
    within one filesystem, and is removed if the caller raises, leaving either the old
    complete file or the new one but never a partial artifact.
    """
    path = Path(path)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.partial')
    os.close(handle)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# --- environment -------------------------------------------------------------

def package_versions(names: Iterable[str]) -> dict[str, str]:
    """Installed versions of the named distributions; a missing one raises."""
    return {name: importlib.metadata.version(name) for name in names}


def git_commit() -> str | None:
    """The repository HEAD commit, or None when git cannot answer.

    Recorded for auditability only. No identity depends on it, because the source
    hashes already determine the code that ran.
    """
    try:
        finished = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY_ROOT, capture_output=True,
                                  text=True, check=True, timeout=_GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.stdout.strip()


def utc_now() -> str:
    """The current UTC time, in the ``2026-09-14T05:22:38Z`` form the recorded runs use."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# --- what determines what ----------------------------------------------------

MODEL_PACKAGES = ('stim',)
"""Stim builds the circuit and derives the detector error model, so it alone decides
which model a set of parameters means."""

SAMPLING_PACKAGES = ('stim',)
"""Stim's sampler decides the shots a seed produces; a seed alone is not reproducible
across versions."""

DECODER_PACKAGES = ('stim', 'numpy', 'pymatching', 'scipy')
"""L1 reads the model through stim, runs UF in NumPy, matches with PyMatching, and
builds its check matrices with SciPy sparse; any of them can change an L1 output."""

CHECK_PACKAGES = ('numpy', 'pymatching')
"""Validation enumerates the exact outer decoder in NumPy and re-decodes the joint
model with PyMatching."""

CALIBRATION_PACKAGES = ('numpy',)
"""Pool-adjacent-violators and the interpolation between knots are pure NumPy."""

REPLAY_PACKAGES = ('numpy', 'sinter')
"""Replay is NumPy, and sinter performs the normalized error-rate conversion."""

SAMPLE_CONVENTIONS = MappingProxyType({
    'bit_order': 'little',
    'separate_observables': True,
    'payload_hash': 'packed detectors then packed observables',
})
"""Bit and layout conventions of a saved sample, recorded inside the model identity so
that shots packed under another convention cannot claim the same model."""

RECORD_CONVENTIONS = MappingProxyType({
    'record_columns': '2i + s',
    'forced_weight_layout': '(shots, patches, 2, 2) indexed [c_X, c_Z]',
    'weight_units': 'nats',
})
"""Layout conventions of the stored L1 record, recorded inside the decoder identity:
the same numbers under a different column order are different outputs."""

DECODER_SOURCES = (
    'src/yoked/decoders/_graph.py',
    'src/yoked/decoders/_union_find.py',
    'src/yoked/decoders/_correlations.py',
    'src/yoked/hierarchical/_arrays.py',
    'src/yoked/hierarchical/_patch_graphs.py',
    'src/yoked/hierarchical/_cluster_gap.py',
    'src/yoked/hierarchical/_matching_gaps.py',
    'src/yoked/hierarchical/_record.py',
)
"""Graph import, UF, the correlation compiler, the hub splitter, the cluster gap, the
matching gaps, and the array/record conventions: everything whose change makes a
stored L1 number mean something different. Policies, metrics, plotting, and
calibration are deliberately absent."""

CHECK_SOURCES = (
    'src/yoked/hierarchical/_outer_decoder.py',
    'src/yoked/hierarchical/_collect.py',
)
"""Validation sources, recorded as their own identity so that changed validation can
recheck stored arrays without rerunning L1."""

CALIBRATION_SOURCES = (
    'src/yoked/hierarchical/_calibration.py',
    'src/yoked/hierarchical/_replay.py',
)
"""The isotonic fit and the scores it is fitted on."""

REPLAY_SOURCES = (
    'src/yoked/hierarchical/_outer_decoder.py',
    'src/yoked/hierarchical/_policies.py',
    'src/yoked/hierarchical/_replay.py',
    'src/yoked/hierarchical/_metrics.py',
)
"""L2, the policies, the replay loop, and the metrics: what a replayed result depends
on and an L1 record does not."""


def _generator_sources() -> tuple[str, ...]:
    """Every ``.py`` under ``src/gen``, sorted, listed at import so that the audit group
    is a plain tuple of paths like the others."""
    generated = (REPOSITORY_ROOT / 'src/gen').rglob('*.py')
    return tuple(sorted(str(path.relative_to(REPOSITORY_ROOT)) for path in generated))


AUDIT_SOURCES = (
    'src/yoked/hierarchical/_stages.py',
    'tools/hierarchical_experiment',
    'src/yoked/_yoked_memory_circuits.py',
) + _generator_sources()
"""Stage boundaries, the CLI, and circuit generation: recorded for auditability and
part of no identity, because the saved circuit and DEM bytes already determine the
model that was actually used."""


def source_hashes(group: Iterable[str]) -> dict[str, str]:
    """Hash every file of a source group, keyed by its repository-relative path.

    A missing file raises instead of being omitted: a hash over fewer sources than the
    group names would silently equal the identity of a different code base.
    """
    hashes = {}
    for relative in group:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'Required source file is missing: {relative}')
        hashes[relative] = sha256_file(path)
    return hashes


# --- identities --------------------------------------------------------------

def _identity(kind: str, inputs: Mapping[str, object]) -> str:
    """Hex SHA-256 of the canonical JSON of one identity's schema, kind, and inputs.

    The kind is part of the digest, so two identities with coincidentally equal inputs
    are still different identities.
    """
    payload = {
        'schema_version': SCHEMA_VERSION,
        'identity': kind,
        'inputs': _plain_json(inputs, replace_undefined=False),
    }
    return sha256_bytes(canonical_json(payload).encode('utf-8'))


def model_identity(*, parameters: Mapping, circuit_sha256: str, dem_sha256: str, num_detectors: int,
                   num_observables: int, conventions: Mapping, versions: Mapping) -> str:
    """Identity of the decoding model actually used.

    The saved ``circuit.stim`` and ``model.dem`` bytes decide it, not the generator
    that produced them, so a changed generator cannot quietly hand old shots a new
    model and a changed DEM under unchanged parameters is a different model.
    """
    return _identity('model', {
        'parameters': parameters,
        'circuit_sha256': circuit_sha256,
        'dem_sha256': dem_sha256,
        'num_detectors': num_detectors,
        'num_observables': num_observables,
        'conventions': conventions,
        'versions': versions,
    })


def parent_sample_identity(*, model: str, seed: int, parent_shots: int, payload_sha256: str,
                           versions: Mapping) -> str:
    """Identity of one full sampling call: the model, seed, full shot count, sampling
    versions, and the complete packed payload hash.

    Rows are not an input, so every subset of a sample keeps its parent's identity.
    """
    return _identity('parent_sample', {
        'model': model,
        'seed': seed,
        'parent_shots': parent_shots,
        'payload_sha256': payload_sha256,
        'versions': versions,
    })


def sampling_family_identity(*, circuit_sha256: str, seed: int, versions: Mapping) -> str:
    """Identity of a circuit/seed/sampling-version combination.

    Two sampling calls that differ only in shot count are different parent samples but
    one family, and held-out checks reject a shared family: their shot streams may
    overlap even though neither payload hash matches.
    """
    return _identity('sampling_family', {
        'circuit_sha256': circuit_sha256,
        'seed': seed,
        'versions': versions,
    })


def decoder_identity(*, sources: Mapping[str, str], conventions: Mapping, versions: Mapping) -> str:
    """Identity of the L1 implementation: DECODER_SOURCES hashes, the record layout
    conventions, and DECODER_PACKAGES versions."""
    return _identity('decoder', {'sources': sources, 'conventions': conventions, 'versions': versions})


def check_identity(*, sources: Mapping[str, str], versions: Mapping) -> str:
    """Identity of the validation implementation: CHECK_SOURCES hashes and CHECK_PACKAGES
    versions. Recorded beside a record so that changed validation can recheck stored
    arrays without rerunning L1."""
    return _identity('check', {'sources': sources, 'versions': versions})


def collection_identity(*, parent_sample: str, decoder: str, role: str, shots: int,
                        rows_sha256: str) -> str:
    """Identity of a collected record: its parent sample, the decoder that produced it,
    the dataset role, and the exact ordered row ids.

    No path is an input: moving a record cannot change what it is, and changing rows,
    role, parent, or decoder requires a separate collection.
    """
    return _identity('collection', {
        'parent_sample': parent_sample,
        'decoder': decoder,
        'role': role,
        'shots': shots,
        'rows_sha256': rows_sha256,
    })


def calibration_identity(*, record_sha256: str, rows_sha256: str, model: str, decoder: str,
                         estimators, knot_convention: str, sources: Mapping[str, str],
                         versions: Mapping) -> str:
    """Identity of a fitted calibration: the verified record artifact and fitted rows,
    the model and decoder identities, the estimator definitions, the knot convention,
    and the calibration sources and versions.

    The parent sample id is not an input: a calibrator is compatible with any record of
    the same model and decoder, which is exactly what fitting and evaluating on
    different samples requires.
    """
    return _identity('calibration', {
        'record_sha256': record_sha256,
        'rows_sha256': rows_sha256,
        'model': model,
        'decoder': decoder,
        'estimators': estimators,
        'knot_convention': knot_convention,
        'sources': sources,
        'versions': versions,
    })


def replay_identity(*, record_sha256: str, calibrators_sha256: str, configurations, tie_rule,
                    work_convention: str, sources: Mapping[str, str], versions: Mapping) -> str:
    """Identity of a replay: the verified record and calibrator artifacts, the
    configurations, the tie rule, the work-count convention, and the replay and metric
    sources and versions."""
    return _identity('replay', {
        'record_sha256': record_sha256,
        'calibrators_sha256': calibrators_sha256,
        'configurations': configurations,
        'tie_rule': tie_rule,
        'work_convention': work_convention,
        'sources': sources,
        'versions': versions,
    })
