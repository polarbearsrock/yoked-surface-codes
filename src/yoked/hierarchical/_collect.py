"""Verified sample sets, the two collection gates, and the checkpointed coordinator.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 3, 5.3, 10.

Four objects, in the order a run uses them:

  * ``SampleSet`` is one Stim sampling call kept verifiable. It owns the exact circuit
    and model text its shots were drawn from, the packed detector and observable
    payloads, their hashes, and the model, parent-sample, and sampling-family
    identities. ``sample`` generates one; ``save``/``load`` publish and re-verify one;
    ``load_recorded_run`` imports an existing four-decoder run's **original** circuit
    and model rather than regenerating them from today's generator. A loaded set keeps
    its payloads memory-mapped and read-only, so a 100,000-shot sample is never
    unpacked whole to decode two thousand of its rows.
  * ``check_graphs`` and ``check_record`` are the two gates. The first is a collection
    prerequisite: the six check graphs, with check vertices mapped back onto the two
    yokes, must reproduce the imported joint graph edge for edge. The second is the
    publication gate: check parity, additivity, preferred-class consistency, yoke
    parity, and zero unexplained disagreements between the reconstructed
    MWPM-reference pipeline and joint MWPM. Neither encodes an agreement percentage;
    every invariant is exact, and a failure names the offending parent rows.
  * ``CollectionSettings`` is one collection request: the role, the exact parent rows,
    and the worker, chunk, and cap counts that do not enter any identity.
  * ``collect_sample`` is the coordinator: given a saved sample, an output directory,
    and a ``CollectionSettings``, it collects the requested rows in chunks, keeps one
    ``checkpoint.npz`` holding every buffer and the completion mask, and publishes a
    validated record. It returns the verified ``LoadedRecord`` when the collection is
    finished and ``None`` while rows remain, so calling it again continues where the
    last call stopped.

The per-row L1 evaluation itself is ``_l1.py``: ``L1Context``, ``collect_rows``, and
``CollectionWork``, imported here and scheduled over chunks. That split is the
identity boundary. ``_l1.py`` belongs to ``DECODER_SOURCES``, because it determines
what a stored record's numbers are; this module belongs to ``CHECK_SOURCES``, because
it determines only how they were sampled, validated, and published, so changed
validation can recheck stored arrays in place instead of rerunning L1.

``collect_rows`` validates corrections but deliberately does not call ``check_record``:
the record checks are defined on a whole record and are what ``collect_sample`` runs,
and must pass, before it publishes one.

The publication order is the whole point of the coordinator. It verifies the sample's
bytes, the saved model, the request, and any existing collection identity; it refuses a
request whose rows, role, model, parent, or decoder differ from what the directory is
already collecting; it runs the graph gate once per model before any row is decoded;
and it writes ``manifest.json`` last, after ``check_record`` has passed and
``record.npz`` is on disk. The manifest is therefore the commit point: an orphan
``record.npz`` without one is an interrupted publication, and a resumed run rebuilds
and replaces it. A failed gate keeps the checkpoint and the diagnostics and publishes
nothing. Worker and chunk counts do not enter any identity, so they may change between
resumptions; a changed decoder source or dependency version may not.

``_collect_test.py`` checks the sample round trip and each hash it verifies, the
imported recorded-run format with a tampered file rejected, graph equivalence against
deliberately changed edge multiplicity, mask, and weight (a difference below ``1e-9``
passing and a larger one failing) together with the fixture's exact degree statistics,
every record invariant failing on its own corrupted field, and, for the coordinator,
that an uninterrupted serial run, a one-chunk run, a two-worker run, a run stopped by
``max_chunks``, and a run interrupted at each of the four hook points all publish the
same arrays, row ids, and retained-row work counts, that a completed directory drops
any stale checkpoint or diagnostics left by a crash between publication and cleanup,
and that a changed decoder identity, role, row set, sample payload, checkpoint, or
record is rejected rather than collected into the same directory. What
``collect_rows`` itself produces is checked in ``_l1_test.py``.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import multiprocessing
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._arrays import readonly_array
# The per-row L1 path: this module schedules it and gates what it produced, and
# ``_whole`` is the package's whole-number check, which lives beside the work counters
# it validates. Nothing here may be imported the other way round; ``_l1.py`` is part of
# the decoder identity and must not depend on sampling, gating, or the coordinator.
from yoked.hierarchical._l1 import WORK_FIELDS, CollectionWork, L1Context, collect_rows, _whole
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._outer_decoder import exact_outer_map_batch, frame_adjusted_syndrome
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraphs
from yoked.hierarchical._provenance import (
    AUDIT_SOURCES, CHECK_PACKAGES, CHECK_SOURCES, DECODER_PACKAGES, DECODER_SOURCES,
    MODEL_PACKAGES, RECORD_CONVENTIONS, REPOSITORY_ROOT, SAMPLING_PACKAGES,
    SCHEMA_VERSION, atomic_replacement, check_identity, collection_identity, decoder_identity,
    git_commit, package_versions, packed_sample_hash, read_json, sample_identities, sha256_bytes,
    sha256_file, source_hashes, utc_now,
    write_json_atomic,
)
# The array container is private to this package: collection writes record.npz and
# checkpoint.npz through it, and every stage downstream reads records only through
# load_record, which verifies the manifest that stands over them.
from yoked.hierarchical._record import (
    ARRAY_FIELDS, COMPLETE_STATUS, L1Record, LoadedRecord, RECORD_FILE, RECORD_MANIFEST,
    RECORD_SCHEMA, _load_arrays, _save_arrays, by_sector, load_record, row_summary, to_columns,
)

CIRCUIT_FILE = 'circuit.stim'
DEM_FILE = 'model.dem'
DETECTORS_FILE = 'detectors_packed.npy'
ACTUAL_FILE = 'actual_observables_packed.npy'
"""The four artifact names, shared with the recorded four-decoder runs so that an
imported run and a generated one have the same layout."""

SAMPLE_FILES = (CIRCUIT_FILE, DEM_FILE, DETECTORS_FILE, ACTUAL_FILE)
"""Every file a sample manifest carries a hash for."""

SAMPLE_MANIFEST = 'sample.json'
"""Written last, so a directory that has it has all four verified artifacts."""

RECORDED_MANIFEST = 'manifest.json'
"""The manifest name of the recorded four-decoder runs this module imports. It is the
same name a completed collection publishes (``RECORD_MANIFEST``), which is one reason
``collect_sample`` refuses to write its outputs into a sample directory."""

RECORDED_PAYLOAD_KEY = 'packed_detectors_then_observables_payload'
"""The recorded runs' name for the hash ``packed_sample_hash`` reproduces."""

SAMPLE_SCHEMA = f'SampleSet/{SCHEMA_VERSION}'
"""Stored in ``sample.json``; a directory written under another layout is rejected
rather than reinterpreted."""

SAMPLE_IDENTITY_NAMES = ('model', 'parent_sample', 'sampling_family')
"""The identities a sample determines on its own. Decoder, check, and collection
identities belong to the records collected from it, not to the shots."""

SUPPORTED_NOISE = 'si1000'
"""The only noise model the experiment uses; ``gen.NoiseModel.si1000(p)`` is what the
saved circuits were built with, so accepting another name would silently mean a
different model under the same parameters."""

SUPPORTED_STYLES = ('cz', 'css')
"""The two circuit styles ``yoked_magic_memory_circuit`` accepts."""

EDGE_WEIGHT_TOLERANCE = 1e-9
"""Absolute nats, ``rtol=0``, for comparing a rebuilt edge weight with the joint graph's
(spec section 10, test 1). Both sides are the same ``log1p(-p) - log(p)`` evaluated by
the same import, so anything above this is a real difference, not rounding."""

WEIGHT_TOLERANCE = 1e-9
"""Absolute nats for matching-weight additivity and preferred-class consistency (spec
section 10, tests 4 and 5). A forced weight sums a few hundred double log-ratios, whose
accumulated rounding is orders of magnitude below this, while any meaningful weight
difference is orders of magnitude above it."""

COST_TOLERANCE = 1e-6
"""Absolute nats for calling a joint-MWPM disagreement a tie (spec section 10, test 6).
Two optimal matchings of the same syndrome have equal total cost in exact arithmetic;
this admits the two solvers' accumulated rounding and nothing else."""

LOGISTIC_FLOOR = float(np.nextafter(0.0, 1.0))
"""The exact outer decoder needs ``0 < q < 1``, and ``1 / (1 + exp(delta))`` saturates to
exactly 0 or 1 once ``|delta|`` passes about 745 nats. Clipping to the two doubles
nearest 0 and 1 keeps ``log q`` and ``log(1 - q)`` finite while binding only where a
double cannot represent the difference anyway."""

LOGISTIC_CEILING = float(np.nextafter(1.0, 0.0))
"""The upper end of the same clip; see ``LOGISTIC_FLOOR``."""

MAX_REPORTED = 5
"""Differing edge groups named in a graph report. Diagnosing a failed comparison needs a
few concrete edges, not a listing of every difference."""

MAX_DIAGNOSTIC_ROWS = 8
"""Offending parent row ids kept per failing record check, for the same reason."""

COLLECTED_VALUE_FIELDS = ('cluster_gap', 'dijkstra_states', 'forced_plain', 'forced_correlated')
"""The record's numeric outputs, which must all be finite and nonnegative."""


# --- shared value checks -----------------------------------------------------

def _require_keys(mapping, keys, where: str) -> None:
    """Every entry a manifest must declare; a missing one is malformed, not a KeyError."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f'{where} must hold an object, got {type(mapping).__name__}')
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f'{where} declares no {missing}')


def _hex_digest(value, name: str) -> str:
    """A 64-character lowercase SHA-256 hex digest."""
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a SHA-256 hex digest, got {value!r}')
    return value


def _positions(value, name: str, *, limit: int) -> np.ndarray:
    """Unique, increasing, in-range row positions, validated before any dtype narrowing."""
    array = np.asarray(value)
    if array.dtype.kind == 'b':
        raise ValueError(f'{name} must be integer positions, not a boolean mask; use np.flatnonzero(mask)')
    if array.dtype.kind not in 'iu' or array.ndim != 1 or len(array) == 0:
        raise ValueError(f'{name} must be a nonempty one-dimensional integer array, '
                         f'got {array.dtype} {array.shape}')
    if (array < 0).any() or (array >= limit).any():
        raise ValueError(f'{name} must lie in the range [0, {limit})')
    if len(array) > 1 and not (np.diff(array) > 0).all():
        raise ValueError(f'{name} must be unique and increasing')
    return array.astype(np.intp)


# --- circuit parameters ------------------------------------------------------

PARAMETER_FIELDS = ('distance', 'rounds', 'p', 'patches', 'yokes', 'style', 'noise')
"""The JSON field names, matching the recorded runs' ``manifest['parameters']``."""


@dataclass(frozen=True)
class CircuitParameters:
    """What determines one experiment circuit.

    Fields: ``distance`` the patch diameter in data qubits; ``rounds`` of stabilizer
    measurement; ``p`` the SI1000 noise strength, strictly inside (0, 1); ``patches``
    surface-code patches; ``yokes`` yoke stabilizers, one per sector; ``style`` the
    two-qubit gate layout; ``noise`` the noise model name.
    """
    distance: int
    rounds: int
    p: float
    patches: int = 6
    yokes: int = NUM_SECTORS
    style: str = 'cz'
    noise: str = SUPPORTED_NOISE

    def __post_init__(self) -> None:
        for name in ('distance', 'rounds', 'patches'):
            object.__setattr__(self, name, _whole(getattr(self, name), name, minimum=1))
        # One yoke per sector: the split reads the two yoke detectors as the last two of
        # the model, and the outer decoder solves one parity constraint per sector.
        object.__setattr__(self, 'yokes', _whole(self.yokes, 'yokes', minimum=1))
        if self.yokes != NUM_SECTORS:
            raise ValueError(f'Only the {NUM_SECTORS}-yoke circuit (one per sector) is supported')
        if isinstance(self.p, (bool, np.bool_)) or not isinstance(self.p, (int, float, np.floating)):
            raise ValueError(f'p must be a number, got {self.p!r}')
        if not 0 < float(self.p) < 1:
            raise ValueError(f'p must lie strictly inside (0, 1), got {self.p}')
        object.__setattr__(self, 'p', float(self.p))
        if self.style not in SUPPORTED_STYLES:
            raise ValueError(f'style must be one of {SUPPORTED_STYLES}, got {self.style!r}')
        if self.noise != SUPPORTED_NOISE:
            raise ValueError(f'Only the {SUPPORTED_NOISE} noise model is supported, got {self.noise!r}')

    def circuit(self) -> stim.Circuit:
        """Build the circuit these parameters name."""
        return yoked_magic_memory_circuit(
            patch_diameter=self.distance, rounds=self.rounds, noise=gen.NoiseModel.si1000(self.p),
            style=self.style, yokes=self.yokes, num_patches=self.patches)

    def dem(self, circuit: stim.Circuit) -> stim.DetectorErrorModel:
        """The decoding model of an already built circuit, with the design's options.

        The circuit is passed in rather than regenerated so that a sample's model always
        belongs to the exact circuit its shots were drawn from.
        """
        return circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)

    def to_json(self) -> dict:
        return {name: getattr(self, name) for name in PARAMETER_FIELDS}

    @classmethod
    def from_json(cls, data: Mapping) -> CircuitParameters:
        missing = [name for name in PARAMETER_FIELDS if name not in data]
        if missing:
            raise ValueError(f'Missing circuit parameters: {missing}')
        # A recorded run declares shots and seed in the same object; they belong to the
        # sampling call rather than to the circuit, so only the named fields are read.
        return cls(**{name: data[name] for name in PARAMETER_FIELDS})


# --- a verified sample -------------------------------------------------------

def _packed(value, name: str, *, rows: int, bits: int) -> np.ndarray:
    """A read-only two-dimensional packed ``uint8`` payload of the declared shape.

    Unlike the record's arrays this is not copied: a 100,000-shot payload is hundreds of
    megabytes and arrives either from Stim (owned by ``sample``) or as a read-only
    memory map (owned by the file). A writeable input is given a read-only view instead,
    so the stored field cannot be written through.
    """
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional bit-packed uint8 array, '
                         f'got {array.dtype} {array.shape}')
    expected = (rows, -(-bits // 8))
    if array.shape != expected:
        raise ValueError(f'{name} must have shape {expected}, got {array.shape}')
    if array.flags.writeable:
        array = array.view()
        array.setflags(write=False)
    return array


def _unpack_rows(packed: np.ndarray, positions: np.ndarray, count: int) -> np.ndarray:
    """The named rows of a packed payload, as a (k, count) boolean array.

    Fancy indexing a memory-mapped payload reads only the requested rows, which is what
    lets a worker hold a 100,000-shot sample open and unpack a chunk of sixty-four.
    ``SampleSet.rows`` and the pool workers both go through here, so both sides of a
    process boundary read the same bits in the same bit order.
    """
    return np.unpackbits(packed[positions], axis=1, count=count, bitorder='little').astype(bool)


def _sample_identities(*, parameters: CircuitParameters, circuit_sha256: str, dem_sha256: str,
                       num_detectors: int, num_observables: int, seed: int, shots: int,
                       payload_sha256: str, model_versions: Mapping, sampling_versions: Mapping) -> dict:
    """The three identities a sample determines, all from content rather than paths."""
    return sample_identities(
        parameters=parameters.to_json(), circuit_sha256=circuit_sha256, dem_sha256=dem_sha256,
        num_detectors=num_detectors, num_observables=num_observables, seed=seed,
        parent_shots=shots, payload_sha256=payload_sha256, model_versions=model_versions,
        sampling_versions=sampling_versions)


def _subset_versions(stored: Mapping, group) -> dict:
    """The recorded versions of one package group; a missing one raises.

    Falling back to today's installed version would let a sample recorded under another
    Stim claim an identity it never had.
    """
    missing = [name for name in group if name not in stored]
    if missing:
        raise ValueError(f'The manifest records no version for {missing}')
    return {name: stored[name] for name in group}


def _check_dimensions(parameters: CircuitParameters, num_detectors: int, num_observables: int) -> None:
    """The declared parameters and the model's own dimensions must agree."""
    expected = NUM_SECTORS * parameters.patches
    if num_observables != expected:
        raise ValueError(f'{parameters.patches} patches imply {expected} observables, '
                         f'the model declares {num_observables}')
    if num_detectors <= NUM_SECTORS:
        raise ValueError(f'The model must have the {NUM_SECTORS} yoke detectors and at least one '
                         f'patch detector, it declares {num_detectors}')


def _model_dimensions(parameters: CircuitParameters, circuit_text: str, dem_text: str) -> tuple[int, int]:
    """Detector and observable counts of the saved model, cross-checked with the circuit."""
    circuit = stim.Circuit(circuit_text)
    dem = stim.DetectorErrorModel(dem_text)
    if (circuit.num_detectors, circuit.num_observables) != (dem.num_detectors, dem.num_observables):
        raise ValueError(
            f'The saved circuit has {circuit.num_detectors} detectors and {circuit.num_observables} '
            f'observables, the saved model {dem.num_detectors} and {dem.num_observables}')
    _check_dimensions(parameters, dem.num_detectors, dem.num_observables)
    return dem.num_detectors, dem.num_observables


def _verify_file(path: Path, expected: str) -> str:
    """Hash a saved artifact and compare it with the manifest's declaration."""
    digest = sha256_file(path)
    if digest != _hex_digest(expected, f'the declared hash of {path.name}'):
        raise ValueError(f'{path.name} hashes to {digest}, the manifest declares {expected}')
    return digest


def _verify_payload(detectors: np.ndarray, actual: np.ndarray, expected: str) -> str:
    """Compare the packed payload hash with the manifest's declaration."""
    digest = packed_sample_hash(detectors, actual)
    if digest != _hex_digest(expected, 'the declared payload hash'):
        raise ValueError(f'The packed sample payload hashes to {digest}, '
                         f'the manifest declares {expected}')
    return digest


def _verify_shapes(detectors: np.ndarray, actual: np.ndarray, *, shots: int,
                   num_detectors: int, num_observables: int) -> None:
    """The packed arrays must hold exactly the declared shots and bit widths."""
    for name, array, bits in ((DETECTORS_FILE, detectors, num_detectors),
                              (ACTUAL_FILE, actual, num_observables)):
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError(f'{name} must hold a two-dimensional uint8 array, '
                             f'got {array.dtype} {array.shape}')
        if array.shape[0] != shots:
            raise ValueError(f'{name} holds {array.shape[0]} rows, the manifest declares {shots} shots')
        if array.shape[1] != -(-bits // 8):
            raise ValueError(f'{name} holds {array.shape[1]} bytes per row, '
                             f'{bits} bits need {-(-bits // 8)}')


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Publish a file through a temporary sibling, as the JSON writer does."""
    with atomic_replacement(path) as temporary:
        temporary.write_bytes(data)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    with atomic_replacement(path) as temporary:
        with open(temporary, 'wb') as handle:
            np.save(handle, array)


@dataclass(frozen=True)
class SampleSet:
    """One Stim sampling call, with everything needed to verify what it is.

    Fields:

    - ``parameters``: the circuit parameters the shots were drawn under.
    - ``seed``: the Stim sampler seed of the single sampling call.
    - ``shots``: the full number of rows in that call; a subset keeps this value,
      because it is part of the parent sample identity.
    - ``num_detectors`` / ``num_observables``: the model's dimensions.
    - ``circuit_sha256`` / ``dem_sha256``: hashes of the exact saved circuit and model
      text, in the encoding ``save`` writes.
    - ``payload_sha256``: the packed detector bytes followed by the packed observable
      bytes, the recorded runs' convention.
    - ``detectors_packed`` (shots, ceil(n_d / 8)) and ``actual_packed``
      (shots, ceil(n_o / 8)): read-only little-endian bit-packed ``uint8`` payloads,
      memory-mapped when they came from disk.
    - ``circuit_text`` / ``dem_text``: the exact saved text of both, so that ``save``
      republishes the same bytes and a worker rebuilds its decoders from the sampled
      model instead of regenerating one.
    - ``model_versions`` / ``sampling_versions``: immutable copies of the package
      versions that formed the identities, retained when a historical sample is
      republished under a newer runtime.
    - ``identities``: an immutable mapping with ``SAMPLE_IDENTITY_NAMES``.
    - ``source``: an immutable mapping with ``kind`` ('generated' or 'imported'), the
      ``directory`` it was read from, and that directory's ``manifest_sha256``; both are
      None for a set that has not been saved or loaded yet.
    """
    parameters: CircuitParameters
    seed: int
    shots: int
    num_detectors: int
    num_observables: int
    circuit_sha256: str
    dem_sha256: str
    payload_sha256: str
    detectors_packed: np.ndarray
    actual_packed: np.ndarray
    circuit_text: str
    dem_text: str
    model_versions: Mapping[str, str]
    sampling_versions: Mapping[str, str]
    identities: Mapping[str, str]
    source: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, CircuitParameters):
            raise TypeError(f'parameters must be CircuitParameters, got {type(self.parameters).__name__}')
        set_field = object.__setattr__
        set_field(self, 'seed', _whole(self.seed, 'seed', minimum=0))
        for name, minimum in (('shots', 1), ('num_detectors', 1), ('num_observables', 1)):
            set_field(self, name, _whole(getattr(self, name), name, minimum=minimum))
        for name in ('circuit_sha256', 'dem_sha256', 'payload_sha256'):
            set_field(self, name, _hex_digest(getattr(self, name), name))
        for name, text in (('circuit_text', self.circuit_text), ('dem_text', self.dem_text)):
            if not isinstance(text, str) or not text:
                raise ValueError(f'{name} must be the nonempty saved text')
        for name in ('model_versions', 'sampling_versions'):
            versions = dict(getattr(self, name))
            required = MODEL_PACKAGES if name == 'model_versions' else SAMPLING_PACKAGES
            versions = _subset_versions(versions, required)
            if not all(isinstance(value, str) and value for value in versions.values()):
                raise ValueError(f'{name} must map package names to nonempty version strings')
            set_field(self, name, MappingProxyType(versions))
        for name in self.model_versions.keys() & self.sampling_versions.keys():
            if self.model_versions[name] != self.sampling_versions[name]:
                raise ValueError(f'model and sampling versions disagree for {name!r}; '
                                 'the saved sample format records one version per package')
        set_field(self, 'detectors_packed', _packed(self.detectors_packed, DETECTORS_FILE,
                                                    rows=self.shots, bits=self.num_detectors))
        set_field(self, 'actual_packed', _packed(self.actual_packed, ACTUAL_FILE,
                                                 rows=self.shots, bits=self.num_observables))
        identities = dict(self.identities)
        if set(identities) != set(SAMPLE_IDENTITY_NAMES):
            raise ValueError(f'identities must name exactly {SAMPLE_IDENTITY_NAMES}, got {sorted(identities)}')
        for name, value in identities.items():
            _hex_digest(value, f'identity {name!r}')
        source = dict(self.source)
        if source.get('kind') not in ('generated', 'imported'):
            raise ValueError(f"source must declare kind 'generated' or 'imported', got {source.get('kind')!r}")
        if set(source) != {'kind', 'directory', 'manifest_sha256'}:
            raise ValueError(f'source must name kind, directory, and manifest_sha256, got {sorted(source)}')
        set_field(self, 'identities', MappingProxyType(identities))
        set_field(self, 'source', MappingProxyType(source))

    # --- reading rows --------------------------------------------------------

    def rows(self, indices) -> tuple[np.ndarray, np.ndarray]:
        """Unpack the requested rows only: (detectors (k, n_d), actual (k, n_o)), both bool.

        ``indices`` are positions in this sample, unique and increasing so that the
        record they feed keeps its parent row ids in order.
        """
        positions = _positions(indices, 'row positions', limit=self.shots)
        return (_unpack_rows(self.detectors_packed, positions, self.num_detectors),
                _unpack_rows(self.actual_packed, positions, self.num_observables))

    def identity_inputs(self) -> dict:
        """Canonical inputs from which a record loader can verify sample identities."""
        return {
            'parameters': self.parameters.to_json(),
            'circuit_sha256': self.circuit_sha256,
            'dem_sha256': self.dem_sha256,
            'num_detectors': self.num_detectors,
            'num_observables': self.num_observables,
            'seed': self.seed,
            'parent_shots': self.shots,
            'payload_sha256': self.payload_sha256,
            'model_versions': dict(self.model_versions),
            'sampling_versions': dict(self.sampling_versions),
        }

    # --- creating, saving, loading -------------------------------------------

    @classmethod
    def sample(cls, parameters: CircuitParameters, *, seed: int, shots: int) -> SampleSet:
        """Generate the circuit once, derive its model, and draw one full packed sample."""
        seed, shots = _whole(seed, 'seed', minimum=0), _whole(shots, 'shots', minimum=1)
        circuit = parameters.circuit()
        dem = parameters.dem(circuit)
        _check_dimensions(parameters, dem.num_detectors, dem.num_observables)
        detectors, actual = circuit.compile_detector_sampler(seed=seed).sample(
            shots=shots, separate_observables=True, bit_packed=True)
        # One trailing newline, matching Stim's own to_file: the model identity hashes
        # these bytes, and an imported recorded run's files carry that newline too.
        circuit_text, dem_text = str(circuit) + '\n', str(dem) + '\n'
        circuit_sha256 = sha256_bytes(circuit_text.encode('utf-8'))
        dem_sha256 = sha256_bytes(dem_text.encode('utf-8'))
        payload_sha256 = packed_sample_hash(detectors, actual)
        model_versions = package_versions(MODEL_PACKAGES)
        sampling_versions = package_versions(SAMPLING_PACKAGES)
        identities = _sample_identities(
            parameters=parameters, circuit_sha256=circuit_sha256, dem_sha256=dem_sha256,
            num_detectors=dem.num_detectors, num_observables=dem.num_observables, seed=seed,
            shots=shots, payload_sha256=payload_sha256,
            model_versions=model_versions, sampling_versions=sampling_versions)
        return cls(parameters=parameters, seed=seed, shots=shots, num_detectors=dem.num_detectors,
                   num_observables=dem.num_observables, circuit_sha256=circuit_sha256,
                   dem_sha256=dem_sha256, payload_sha256=payload_sha256, detectors_packed=detectors,
                   actual_packed=actual, circuit_text=circuit_text, dem_text=dem_text,
                   model_versions=model_versions, sampling_versions=sampling_versions,
                   identities=identities,
                   source={'kind': 'generated', 'directory': None, 'manifest_sha256': None})

    def save(self, directory) -> Path:
        """Write the artifacts and publish their original identity inputs last."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(directory / CIRCUIT_FILE, self.circuit_text.encode('utf-8'))
        _write_bytes_atomic(directory / DEM_FILE, self.dem_text.encode('utf-8'))
        _write_npy_atomic(directory / DETECTORS_FILE, self.detectors_packed)
        _write_npy_atomic(directory / ACTUAL_FILE, self.actual_packed)
        write_json_atomic(directory / SAMPLE_MANIFEST, {
            'schema_version': SAMPLE_SCHEMA,
            'parameters': self.parameters.to_json(),
            'seed': self.seed,
            'shots': self.shots,
            'num_detectors': self.num_detectors,
            'num_observables': self.num_observables,
            'files': {name: sha256_file(directory / name) for name in SAMPLE_FILES},
            'payload_sha256': self.payload_sha256,
            'versions': {**dict(self.model_versions), **dict(self.sampling_versions)},
            'identities': dict(self.identities),
            'source': dict(self.source),
            'created_utc': utc_now(),
        })
        return directory

    @classmethod
    def load(cls, directory) -> SampleSet:
        """Open a saved sample, verifying every hash, shape, and declared dimension."""
        directory = Path(directory)
        manifest = read_json(directory / SAMPLE_MANIFEST)
        _require_keys(manifest, ('schema_version', 'parameters', 'seed', 'shots', 'num_detectors',
                                 'num_observables', 'files', 'payload_sha256', 'versions',
                                 'identities', 'source'), SAMPLE_MANIFEST)
        if manifest['schema_version'] != SAMPLE_SCHEMA:
            raise ValueError(f'{directory / SAMPLE_MANIFEST} holds schema '
                             f'{manifest["schema_version"]!r}, expected {SAMPLE_SCHEMA!r}')
        _require_keys(manifest['files'], SAMPLE_FILES, f'{SAMPLE_MANIFEST} files')
        _require_keys(manifest['source'], ('kind',), f'{SAMPLE_MANIFEST} source')
        hashes = {name: _verify_file(directory / name, manifest['files'][name]) for name in SAMPLE_FILES}
        parameters = CircuitParameters.from_json(manifest['parameters'])
        seed = _whole(manifest['seed'], 'seed', minimum=0)
        shots = _whole(manifest['shots'], 'shots', minimum=1)
        circuit_text = (directory / CIRCUIT_FILE).read_text(encoding='utf-8')
        dem_text = (directory / DEM_FILE).read_text(encoding='utf-8')
        num_detectors, num_observables = _model_dimensions(parameters, circuit_text, dem_text)
        if (manifest['num_detectors'], manifest['num_observables']) != (num_detectors, num_observables):
            raise ValueError(f'The manifest declares {manifest["num_detectors"]} detectors and '
                             f'{manifest["num_observables"]} observables, the saved model has '
                             f'{num_detectors} and {num_observables}')
        detectors = np.load(directory / DETECTORS_FILE, mmap_mode='r')
        actual = np.load(directory / ACTUAL_FILE, mmap_mode='r')
        _verify_payload(detectors, actual, manifest['payload_sha256'])
        _verify_shapes(detectors, actual, shots=shots, num_detectors=num_detectors,
                       num_observables=num_observables)
        versions = manifest['versions']
        model_versions = _subset_versions(versions, MODEL_PACKAGES)
        sampling_versions = _subset_versions(versions, SAMPLING_PACKAGES)
        identities = _sample_identities(
            parameters=parameters, circuit_sha256=hashes[CIRCUIT_FILE], dem_sha256=hashes[DEM_FILE],
            num_detectors=num_detectors, num_observables=num_observables, seed=seed, shots=shots,
            payload_sha256=manifest['payload_sha256'],
            model_versions=model_versions, sampling_versions=sampling_versions)
        # Recomputed from the verified content rather than trusted: a manifest whose
        # declared identity does not follow from its own artifacts is not a sample.
        if dict(manifest['identities']) != identities:
            raise ValueError('The declared sample identities do not follow from the saved artifacts')
        return cls(parameters=parameters, seed=seed, shots=shots, num_detectors=num_detectors,
                   num_observables=num_observables, circuit_sha256=hashes[CIRCUIT_FILE],
                   dem_sha256=hashes[DEM_FILE], payload_sha256=manifest['payload_sha256'],
                   detectors_packed=detectors, actual_packed=actual, circuit_text=circuit_text,
                   dem_text=dem_text, model_versions=model_versions,
                   sampling_versions=sampling_versions, identities=identities,
                   source={'kind': manifest['source']['kind'], 'directory': str(directory),
                           'manifest_sha256': sha256_file(directory / SAMPLE_MANIFEST)})

    @classmethod
    def load_recorded_run(cls, directory) -> SampleSet:
        """Import a recorded four-decoder run's original circuit, model, and shots.

        The model is the one the run actually decoded against, verified against the
        run's own ``input_sha256``. Nothing here regenerates a circuit or a model from
        today's generator, which could silently hand old shots a new model.
        """
        directory = Path(directory)
        manifest = read_json(directory / RECORDED_MANIFEST)
        _require_keys(manifest, ('parameters', 'input_sha256', 'versions'), RECORDED_MANIFEST)
        declared = manifest['parameters']
        _require_keys(declared, ('seed', 'shots'), f'{RECORDED_MANIFEST} parameters')
        parameters = CircuitParameters.from_json(declared)
        seed = _whole(declared['seed'], 'seed', minimum=0)
        shots = _whole(declared['shots'], 'shots', minimum=1)
        hashes = manifest['input_sha256']
        _require_keys(hashes, (CIRCUIT_FILE, DEM_FILE, RECORDED_PAYLOAD_KEY),
                      f'{RECORDED_MANIFEST} input_sha256')
        circuit_sha256 = _verify_file(directory / CIRCUIT_FILE, hashes[CIRCUIT_FILE])
        dem_sha256 = _verify_file(directory / DEM_FILE, hashes[DEM_FILE])
        circuit_text = (directory / CIRCUIT_FILE).read_text(encoding='utf-8')
        dem_text = (directory / DEM_FILE).read_text(encoding='utf-8')
        num_detectors, num_observables = _model_dimensions(parameters, circuit_text, dem_text)
        detectors = np.load(directory / DETECTORS_FILE, mmap_mode='r')
        actual = np.load(directory / ACTUAL_FILE, mmap_mode='r')
        payload_sha256 = _verify_payload(detectors, actual, hashes[RECORDED_PAYLOAD_KEY])
        _verify_shapes(detectors, actual, shots=shots, num_detectors=num_detectors,
                       num_observables=num_observables)
        versions = manifest['versions']
        model_versions = _subset_versions(versions, MODEL_PACKAGES)
        sampling_versions = _subset_versions(versions, SAMPLING_PACKAGES)
        identities = _sample_identities(
            parameters=parameters, circuit_sha256=circuit_sha256, dem_sha256=dem_sha256,
            num_detectors=num_detectors, num_observables=num_observables, seed=seed, shots=shots,
            payload_sha256=payload_sha256,
            model_versions=model_versions, sampling_versions=sampling_versions)
        return cls(parameters=parameters, seed=seed, shots=shots, num_detectors=num_detectors,
                   num_observables=num_observables, circuit_sha256=circuit_sha256,
                   dem_sha256=dem_sha256, payload_sha256=payload_sha256, detectors_packed=detectors,
                   actual_packed=actual, circuit_text=circuit_text, dem_text=dem_text,
                   model_versions=model_versions, sampling_versions=sampling_versions,
                   identities=identities,
                   source={'kind': 'imported', 'directory': str(directory),
                           'manifest_sha256': sha256_file(directory / RECORDED_MANIFEST)})


# --- graph checks ------------------------------------------------------------

def _edge_key(u: int, v, mask: int):
    """An edge's identity for the comparison: unordered endpoints and observable mask."""
    return ((u, None) if v is None else (min(u, v), max(u, v))), mask


def _joint_groups(graph: DecodingGraph) -> dict:
    groups = collections.defaultdict(list)
    for u, v, weight, mask in graph.edges:
        groups[_edge_key(u, v, mask)].append(weight)
    return groups


def _rebuilt_groups(patches: PatchGraphs) -> dict:
    """The six check graphs with check vertices, detector ids, and masks made global."""
    groups = collections.defaultdict(list)
    for patch in patches:
        to_global = dict(enumerate(patch.global_detector_ids))
        to_global.update(zip(patch.check_vertices, patches.yoke_detector_ids))
        for u, v, weight, mask in patch.check_graph.edges:
            global_mask = sum(1 << patch.observable_ids[s] for s in range(NUM_SECTORS) if (mask >> s) & 1)
            key = _edge_key(to_global[u], None if v is None else to_global[v], global_mask)
            groups[key].append(weight)
    return groups


def _describe(key) -> str:
    """One differing edge group, as a short line for the report."""
    (u, v), mask = key
    return f'({u}, {"boundary" if v is None else v}, mask {mask})'


@dataclass(frozen=True)
class GraphChecks:
    """Whether the patch split reproduces the imported joint graph, and what it looks like.

    Fields: ``joint_edges`` and ``rebuilt_edges`` are edge counts of the imported joint
    graph and of the six check graphs remapped onto it; ``max_weight_difference`` is the
    largest absolute nats difference inside a group present on both sides with the same
    multiplicity; ``equivalent`` says the two edge multisets agree; ``missing_groups``,
    ``extra_groups``, ``multiplicity_mismatches``, and ``weight_mismatches`` count the
    endpoint/mask groups that differ, and ``examples`` names up to ``MAX_REPORTED`` of
    each kind. ``yoke_degrees`` (X, Z), ``median_detector_degree``, and
    ``max_non_yoke_degree`` are read from the imported joint graph's adjacency, after
    the importer's parallel-edge merge, never from raw DEM target multiplicities.
    """
    joint_edges: int
    rebuilt_edges: int
    max_weight_difference: float
    equivalent: bool
    missing_groups: int
    extra_groups: int
    multiplicity_mismatches: int
    weight_mismatches: int
    examples: tuple[str, ...]
    yoke_degrees: tuple[int, int]
    median_detector_degree: float
    max_non_yoke_degree: int

    @property
    def passed(self) -> bool:
        """Graph equivalence is the whole gate; the degrees are description."""
        return self.equivalent

    def raise_if_failed(self) -> None:
        """Raise naming the differing edge groups; nothing may be decoded until this holds.

        The split is a prerequisite rather than a diagnostic: if the six check graphs do
        not reproduce the joint graph, every gap and forced weight collected from them
        would describe a model nobody wrote down.
        """
        if not self.passed:
            raise ValueError(
                f'Graph checks failed: {self.missing_groups} missing, {self.extra_groups} extra, '
                f'{self.multiplicity_mismatches} multiplicity, and {self.weight_mismatches} weight '
                f'mismatches between the split and the joint graph; examples {list(self.examples)}')

    def to_json(self) -> dict:
        return {
            'passed': self.passed,
            'equivalent': self.equivalent,
            'joint_edges': self.joint_edges,
            'rebuilt_edges': self.rebuilt_edges,
            'max_weight_difference': self.max_weight_difference,
            'missing_groups': self.missing_groups,
            'extra_groups': self.extra_groups,
            'multiplicity_mismatches': self.multiplicity_mismatches,
            'weight_mismatches': self.weight_mismatches,
            'examples': list(self.examples),
            'yoke_degrees': list(self.yoke_degrees),
            'median_detector_degree': self.median_detector_degree,
            'max_non_yoke_degree': self.max_non_yoke_degree,
        }


def check_graphs(dem: stim.DetectorErrorModel, patches: PatchGraphs) -> GraphChecks:
    """Compare the six remapped check graphs with the imported joint graph, edge for edge.

    Weights are compared group by group after sorting, with absolute tolerance
    ``EDGE_WEIGHT_TOLERANCE`` and no relative term; rounding weights into dictionary keys
    would not be a tolerance comparison.
    """
    joint = DecodingGraph.from_dem(dem)
    expected, rebuilt = _joint_groups(joint), _rebuilt_groups(patches)
    missing = sorted(expected.keys() - rebuilt.keys())
    extra = sorted(rebuilt.keys() - expected.keys())
    multiplicity, weight_mismatches, examples = [], [], []
    max_difference = 0.0
    for key in expected.keys() & rebuilt.keys():
        left, right = sorted(expected[key]), sorted(rebuilt[key])
        if len(left) != len(right):
            multiplicity.append(key)
            continue
        difference = float(np.max(np.abs(np.array(left) - np.array(right)))) if left else 0.0
        max_difference = max(max_difference, difference)
        if difference > EDGE_WEIGHT_TOLERANCE:
            weight_mismatches.append(key)
    for label, keys in (('missing', missing), ('extra', extra),
                        ('multiplicity', multiplicity), ('weight', weight_mismatches)):
        examples.extend(f'{label} {_describe(key)}' for key in keys[:MAX_REPORTED])
    degrees = np.array([len(joint.adjacency[d]) for d in range(joint.num_detectors)], dtype=np.int64)
    yoke = patches.yoke_detector_ids
    return GraphChecks(
        joint_edges=len(joint.edges),
        rebuilt_edges=sum(len(weights) for weights in rebuilt.values()),
        max_weight_difference=max_difference,
        equivalent=not missing and not extra and not multiplicity and not weight_mismatches,
        missing_groups=len(missing), extra_groups=len(extra),
        multiplicity_mismatches=len(multiplicity), weight_mismatches=len(weight_mismatches),
        examples=tuple(examples),
        yoke_degrees=(int(degrees[yoke[0]]), int(degrees[yoke[1]])),
        median_detector_degree=float(np.median(degrees)),
        max_non_yoke_degree=int(np.delete(degrees, list(yoke)).max()))


# --- record checks -----------------------------------------------------------

def _sector_parity(columns: np.ndarray) -> np.ndarray:
    """Parity over patches of a (shots, 2P) column layout, as (shots, 2)."""
    return (by_sector(columns).sum(axis=2) % 2).astype(bool)


def _additivity_error(forced: np.ndarray) -> np.ndarray:
    """|W(0,0) + W(1,1) - W(0,1) - W(1,0)| per shot and patch, zero when the sectors add."""
    return np.abs(forced[:, :, 0, 0] + forced[:, :, 1, 1] - forced[:, :, 0, 1] - forced[:, :, 1, 0])


def _total_forced_cost(forced: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """sum_i W_i(f[i, X], f[i, Z]) in nats, for a (shots, 2P) prediction."""
    bits = prediction.reshape(len(prediction), -1, NUM_SECTORS).astype(np.intp)
    shot = np.arange(len(prediction))[:, None]
    patch = np.arange(bits.shape[1])[None, :]
    return forced[shot, patch, bits[:, :, 0], bits[:, :, 1]].sum(axis=1)


def _reconstructed_final(record: L1Record) -> np.ndarray:
    """The MWPM-reference pipeline's final prediction: plain gaps, logistic, exact L2.

    Patch-major gaps become sector-major once, with a single ``swapaxes``, rather than
    through a round trip via the column layout.
    """
    patch_major = record.mwpm_reference.reshape(record.shots, record.num_patches, NUM_SECTORS)
    gaps = signed_gaps(record.forced_plain, patch_major).swapaxes(1, 2)      # (shots, 2, patches)
    with np.errstate(over='ignore'):
        # A gap beyond ~709 nats overflows to inf, which the clip turns into the floor.
        q = np.clip(1.0 / (1.0 + np.exp(gaps)), LOGISTIC_FLOOR, LOGISTIC_CEILING)
    sigma = frame_adjusted_syndrome(record.yoke, record.mwpm_reference)      # (shots, 2)
    patterns = np.stack([exact_outer_map_batch(q[:, s, :], sigma[:, s]).patterns
                         for s in range(NUM_SECTORS)], axis=1)               # (shots, 2, patches)
    return record.mwpm_reference ^ to_columns(patterns)


@dataclass(frozen=True)
class RecordChecks:
    """Every applicable invariant of one stored record, with the rows that failed.

    Fields:

    - ``rows``: rows checked.
    - ``check_parity_violations``: rows whose sampled per-sector observable parity
      differs from the sampled yoke bit.
    - ``plain_additivity_max_error`` / ``correlated_additivity_max_error``: the largest
      absolute nats by which a patch's forced weights fail to add across sectors.
    - ``plain_preferred_disagreements`` / ``correlated_preferred_disagreements``:
      (row, patch, sector) entries whose forced weights prefer the complement of the
      recorded prediction by more than ``WEIGHT_TOLERANCE``.
    - ``joint_agreement``: fraction of rows where the reconstructed final prediction
      equals the recorded joint MWPM prediction in every column.
    - ``joint_disagreements``: rows where they differ.
    - ``tie_explained_disagreements``: of those, rows whose two total forced costs agree
      within ``COST_TOLERANCE``, so both predictions are joint optima.
    - ``unexplained_disagreements``: the rest, which must be zero.
    - ``max_disagreement_cost_difference``: the largest total forced cost difference in
      nats over the disagreeing rows; zero when there are none.
    - ``final_parity_violations``: rows where the reconstructed final prediction or the
      recorded joint MWPM prediction breaks yoke parity in either sector.
    - ``value_violations``: rows holding a non-finite or negative collected value.
    - ``diagnostics``: an immutable mapping from failing check name to the first
      ``MAX_DIAGNOSTIC_ROWS`` offending parent row ids; its keys are the names
      ``failures`` reports.
    """
    rows: int
    check_parity_violations: int
    plain_additivity_max_error: float
    correlated_additivity_max_error: float
    plain_preferred_disagreements: int
    correlated_preferred_disagreements: int
    joint_agreement: float
    joint_disagreements: int
    tie_explained_disagreements: int
    unexplained_disagreements: int
    max_disagreement_cost_difference: float
    final_parity_violations: int
    value_violations: int
    diagnostics: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'diagnostics', MappingProxyType(
            {str(name): tuple(int(row) for row in rows) for name, rows in self.diagnostics.items()}))

    @property
    def failures(self) -> tuple[str, ...]:
        """The names of the checks that did not hold; empty means the record may be published."""
        failed = []
        if self.check_parity_violations:
            failed.append('check_parity')
        if self.plain_additivity_max_error > WEIGHT_TOLERANCE:
            failed.append('plain_additivity')
        if self.correlated_additivity_max_error > WEIGHT_TOLERANCE:
            failed.append('correlated_additivity')
        if self.plain_preferred_disagreements:
            failed.append('plain_preferred')
        if self.correlated_preferred_disagreements:
            failed.append('correlated_preferred')
        if self.final_parity_violations:
            failed.append('final_parity')
        if self.unexplained_disagreements:
            failed.append('unexplained_joint')
        if self.value_violations:
            failed.append('values')
        return tuple(failed)

    @property
    def passed(self) -> bool:
        """Every invariant is exact; no agreement percentage is part of this."""
        return not self.failures

    def raise_if_failed(self) -> None:
        """Raise naming the failing checks and their offending parent rows."""
        failures = self.failures
        if failures:
            offending = {name: list(self.diagnostics.get(name, ())) for name in failures}
            raise ValueError(f'Record checks failed: {list(failures)}; offending rows {offending}')

    def to_json(self) -> dict:
        fields = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)
                  if field.name != 'diagnostics'}
        return {**fields, 'passed': self.passed, 'failures': list(self.failures),
                'diagnostics': {name: list(rows) for name, rows in self.diagnostics.items()}}


def check_record(record: L1Record) -> RecordChecks:
    """Re-derive every invariant of a stored record from the record alone.

    Nothing here re-runs a decoder: the checks are algebraic identities of the stored
    arrays plus one re-enumeration of the exact outer decoder, so a stored record can be
    rechecked without the sample it came from.
    """
    parent = np.asarray(record.rows)
    diagnostics: dict[str, tuple[int, ...]] = {}

    def note(name: str, offending_rows: np.ndarray) -> int:
        """Keep the first few offending parent row ids; return how many rows offended."""
        positions = np.flatnonzero(offending_rows)
        if len(positions):
            diagnostics[name] = tuple(int(row) for row in parent[positions[:MAX_DIAGNOSTIC_ROWS]])
        return int(len(positions))

    # The collected values are re-checked here rather than assumed from the constructor:
    # check_record is the gate a stored record passes on its own terms.
    value_rows = np.zeros(record.shots, dtype=bool)
    for name in COLLECTED_VALUE_FIELDS:
        array = np.asarray(getattr(record, name), dtype=np.float64)
        bad = ~np.isfinite(array) | (array < 0)
        if bad.any():
            value_rows |= bad.reshape(record.shots, -1).any(axis=1)
    value_violations = note('values', value_rows)

    check_parity = note('check_parity', (_sector_parity(record.actual) != record.yoke).any(axis=1))

    plain_error, correlated_error = (_additivity_error(record.forced_plain),
                                     _additivity_error(record.forced_correlated))
    note('plain_additivity', (plain_error > WEIGHT_TOLERANCE).any(axis=1))
    note('correlated_additivity', (correlated_error > WEIGHT_TOLERANCE).any(axis=1))

    preferred = {}
    for name, forced, prediction in (('plain', record.forced_plain, record.mwpm_reference),
                                     ('correlated', record.forced_correlated, record.correlated_prediction)):
        patch_major = prediction.reshape(record.shots, record.num_patches, NUM_SECTORS)
        # A negative signed gap means the forced weights prefer the complement of the
        # class the recorded unforced prediction chose.
        bad = signed_gaps(forced, patch_major) < -WEIGHT_TOLERANCE
        note(f'{name}_preferred', bad.any(axis=(1, 2)))
        preferred[name] = int(bad.sum())

    final = _reconstructed_final(record)
    parity_bad = ((_sector_parity(final) != record.yoke).any(axis=1)
                  | (_sector_parity(record.joint_mwpm) != record.yoke).any(axis=1))
    final_parity = note('final_parity', parity_bad)

    agree = (final == record.joint_mwpm).all(axis=1)
    difference = np.abs(_total_forced_cost(record.forced_plain, final)
                        - _total_forced_cost(record.forced_plain, record.joint_mwpm))
    tied = ~agree & (difference <= COST_TOLERANCE)
    unexplained = note('unexplained_joint', ~agree & ~tied)
    return RecordChecks(
        rows=record.shots,
        check_parity_violations=check_parity,
        plain_additivity_max_error=float(plain_error.max()),
        correlated_additivity_max_error=float(correlated_error.max()),
        plain_preferred_disagreements=preferred['plain'],
        correlated_preferred_disagreements=preferred['correlated'],
        joint_agreement=float(agree.mean()),
        joint_disagreements=int((~agree).sum()),
        tie_explained_disagreements=int(tied.sum()),
        unexplained_disagreements=unexplained,
        max_disagreement_cost_difference=float(difference[~agree].max()) if (~agree).any() else 0.0,
        final_parity_violations=final_parity,
        value_violations=value_violations,
        diagnostics=diagnostics)


# --- what to collect ---------------------------------------------------------

ROLES = ('calibration', 'evaluation')
"""The two dataset roles M1 collects. Confirmation exists in the data plan (spec
section 3) but may only be sampled and collected after the analysis is frozen, so
naming it here would be the one way to collect it by accident."""

COLLECTION_FILE = 'collection.json'
"""Written on the first run and compared on every later one: what this directory is
collecting. It is not a completion marker, which is why it is a separate file from the
manifest that appears only when the collection is finished."""

CHECKPOINT_FILE = 'checkpoint.npz'
"""Every buffer and the completion mask of an unfinished collection, in one container.
There is deliberately no second progress file: two files can disagree about which rows
are done, and a checkpoint that disagrees with itself is worse than none."""

CHECKPOINT_SCHEMA = f'L1Checkpoint/{SCHEMA_VERSION}'
"""Stored inside the checkpoint. A container written under another layout is rejected
rather than reinterpreted, exactly as a record's is."""

CHECKPOINT_ENTRIES = ('completed', 'collection', 'work', 'seconds_total', 'resumptions')
"""What a checkpoint holds besides the record buffers: the completion mask, the
collection identity it belongs to, the retained-row work counters in ``WORK_FIELDS``
order, the cumulative collection seconds, and how often it has been resumed."""

CHECKPOINT_INTERVAL_SECONDS = 60.0
"""How often the checkpoint is rewritten while chunks are arriving. A rewrite copies
every buffer, which at 100,000 d=9 rows is tens of megabytes, so writing after every
chunk would spend more time on I/O than on decoding; a minute keeps that overhead
negligible while bounding the work an interruption can lose to a minute of decoding.
The checkpoint is also written whenever scheduling stops, regardless of this interval."""

FAILED_CHECKS_FILE = 'failed_checks.json'
"""Where a failed publication gate leaves its diagnostics. The checkpoint stays too, so
a failure can be investigated with the collected rows in hand rather than recollected."""

WORKER_START_METHOD = 'forkserver'
"""Start method for the worker pool. Forking a coordinator that already holds Stim
models, PyMatching matchers, and a memory-mapped sample would duplicate that state into
every child; a forkserver child starts clean and builds exactly the one ``L1Context``
it needs."""


def _requested_rows(value) -> np.ndarray:
    """The exact parent rows to collect: whole, nonnegative, unique, owned, increasing.

    Input order does not matter and is normalized away, because a record stores its rows
    in increasing order. Duplicates are not: they would silently shrink the collection
    and change what the row hash identifies.
    """
    array = np.asarray(value)
    if array.dtype.kind == 'b':
        raise ValueError('rows must be parent row ids, not a boolean mask; use np.flatnonzero(mask)')
    if array.dtype.kind not in 'iu':
        raise ValueError(f'rows must be an integer array of parent row ids, got dtype {array.dtype}')
    if array.ndim != 1:
        raise ValueError(f'rows must be one-dimensional, got shape {array.shape}')
    if len(array) == 0:
        raise ValueError('rows must be nonempty; a collection of no rows is not a collection')
    if (array < 0).any():
        raise ValueError('rows must be nonnegative')
    ordered = np.unique(array)
    if len(ordered) != len(array):
        raise ValueError('rows must be unique')
    return readonly_array(ordered, dtype=np.int64)


@dataclass(frozen=True)
class CollectionSettings:
    """What to collect, and how much machinery to collect it with.

    Fields:

    - ``role``: one of ``ROLES``. It is part of the collection identity, so calibration
      and evaluation rows can never end up in one record.
    - ``rows``: the parent sample rows to collect, stored as an owned read-only int64
      array in increasing order. Duplicates, negatives, floats, and boolean masks are
      rejected rather than normalized; the range check against the parent sample happens
      in ``collect_sample``, which is where the sample's shot count is known.
    - ``workers``: processes that decode chunks. One means in this process, with no pool
      at all, so a failing chunk raises where it happened.
    - ``chunk_size``: parent rows per scheduled chunk (rows).
    - ``max_chunks``: schedule at most this many chunks in one call, then checkpoint and
      return; None schedules every pending chunk.

    ``workers``, ``chunk_size``, and ``max_chunks`` cannot change a decoded value, so
    they are deliberately absent from the collection identity and may differ between
    resumptions of the same collection.
    """
    role: str
    rows: np.ndarray
    workers: int = 1
    chunk_size: int = 64
    max_chunks: int | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f'role must be one of {ROLES}, got {self.role!r}')
        set_field = object.__setattr__
        set_field(self, 'rows', _requested_rows(self.rows))
        for name in ('workers', 'chunk_size'):
            set_field(self, name, _whole(getattr(self, name), name, minimum=1))
        if self.max_chunks is not None:
            set_field(self, 'max_chunks', _whole(self.max_chunks, 'max_chunks', minimum=1))

    @property
    def shots(self) -> int:
        """Rows this collection covers when it is complete."""
        return int(len(self.rows))

    def rows_summary(self) -> dict:
        """Count, bounds, and the hash of the exact ordered ids, as a manifest records them."""
        return row_summary(self.rows)


# --- test seams --------------------------------------------------------------

def _no_op() -> None:
    """The default at every hook point: collection does nothing observable there."""


@dataclass(frozen=True)
class _CollectionHooks:
    """Where a test may interrupt a collection, by raising from one of these callables.

    The four points bracket the two places where a collection commits something to disk:
    ``before_checkpoint`` and ``after_checkpoint`` surround replacing ``checkpoint.npz``,
    and ``after_record`` and ``before_manifest`` surround the window in which
    ``record.npz`` exists but no completion manifest does. They exist so that resume
    behavior is tested by actually interrupting a real run rather than by monkeypatching
    internals, and they do nothing in production: ``collect_sample`` defaults them to
    no-ops and never exposes them through the CLI.
    """
    before_checkpoint: Callable[[], None] = _no_op
    after_checkpoint: Callable[[], None] = _no_op
    after_record: Callable[[], None] = _no_op
    before_manifest: Callable[[], None] = _no_op


# --- the coordinator's private buffers ---------------------------------------

def _buffer_specs(shots: int, patches: int) -> dict[str, tuple[tuple[int, ...], type]]:
    """Shape and dtype of every record buffer, for a collection of ``shots`` rows.

    Allocation, checkpoint validation, and chunk installation all read the layout here,
    so there is one place where a record field's stored shape is written down.
    """
    columns = NUM_SECTORS * patches
    bits = (shots, columns)
    forced = (shots, patches, NUM_SECTORS, NUM_SECTORS)
    specs = {
        'actual': (bits, np.bool_),
        'yoke': ((shots, NUM_SECTORS), np.bool_),
        'uf_reference': (bits, np.bool_),
        'mwpm_reference': (bits, np.bool_),
        'correlated_prediction': (bits, np.bool_),
        'joint_mwpm': (bits, np.bool_),
        'cluster_gap': (bits, np.float64),
        'dijkstra_states': (bits, np.int64),
        'forced_plain': (forced, np.float64),
        'forced_correlated': (forced, np.float64),
        'reweighted_patches': ((shots, patches), np.bool_),
        'rows': ((shots,), np.int64),
    }
    if tuple(specs) != ARRAY_FIELDS:
        raise ValueError(f'The buffers cover {tuple(specs)}, a record stores {ARRAY_FIELDS}')
    return specs


def _checkpoint_array(entries: Mapping, name: str, *, dtype, shape) -> np.ndarray:
    """One checkpoint entry, with its declared dtype and shape required."""
    array = entries[name]
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f'{CHECKPOINT_FILE} holds {name} as {array.dtype} {array.shape}, '
                         f'expected {np.dtype(dtype)} {shape}')
    return array


class _Buffers:
    """The coordinator's private, mutable state for one in-progress collection.

    The arrays are writable, partly filled, and meaningless for any row the completion
    mask does not name, which is exactly why they never leave this module: ``record()``
    is the only way out and re-runs every ``L1Record`` check on the way. A row becomes
    completed only after its arrays and its chunk's work counters are both installed, so
    the work totals always describe precisely the retained rows.

    Attributes: ``arrays`` the record buffers by name; ``completed`` (shots,) bool;
    ``work`` the retained-row ``CollectionWork``; ``seconds_total`` cumulative collection
    seconds across resumptions; ``resumptions`` how often a checkpoint has been reopened.
    """

    def __init__(self, arrays: dict, completed: np.ndarray, work: CollectionWork,
                 seconds_total: float, resumptions: int):
        self.arrays = arrays
        self.completed = completed
        self.work = work
        self.seconds_total = seconds_total
        self.resumptions = resumptions

    @classmethod
    def allocate(cls, *, rows: np.ndarray, patches: int) -> _Buffers:
        """Empty buffers for a fresh collection, with the requested row ids already in place."""
        specs = _buffer_specs(len(rows), patches)
        arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in specs.items()}
        arrays['rows'][:] = rows
        return cls(arrays, np.zeros(len(rows), dtype=bool), CollectionWork(**dict.fromkeys(WORK_FIELDS, 0)),
                   seconds_total=0.0, resumptions=0)

    @classmethod
    def load(cls, path, *, rows: np.ndarray, patches: int, identity: str) -> _Buffers:
        """Reopen a checkpoint, validating everything about it before trusting a row.

        Reopening one is what a resumption is, so the count is incremented here.
        """
        entries = _load_arrays(path, schema=CHECKPOINT_SCHEMA)
        specs = _buffer_specs(len(rows), patches)
        expected = set(specs) | set(CHECKPOINT_ENTRIES)
        if set(entries) != expected:
            raise ValueError(f'{CHECKPOINT_FILE} holds {sorted(entries)}, expected {sorted(expected)}')
        stored = str(entries['collection'])
        if stored != identity:
            raise ValueError(f'{CHECKPOINT_FILE} belongs to collection {stored}, '
                             f'this request is collection {identity}')
        arrays = {name: _checkpoint_array(entries, name, dtype=dtype, shape=shape)
                  for name, (shape, dtype) in specs.items()}
        if not np.array_equal(arrays['rows'], rows):
            raise ValueError(f'{CHECKPOINT_FILE} holds different parent row ids than this request')
        completed = _checkpoint_array(entries, 'completed', dtype=np.bool_, shape=(len(rows),))
        counters = _checkpoint_array(entries, 'work', dtype=np.int64, shape=(len(WORK_FIELDS),))
        seconds = float(_checkpoint_array(entries, 'seconds_total', dtype=np.float64, shape=()))
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError(f'{CHECKPOINT_FILE} holds {seconds} collection seconds')
        resumptions = int(_checkpoint_array(entries, 'resumptions', dtype=np.int64, shape=()))
        return cls(arrays, completed,
                   CollectionWork(**dict(zip(WORK_FIELDS, (int(value) for value in counters)))),
                   seconds_total=seconds,
                   resumptions=_whole(resumptions, 'resumptions', minimum=0) + 1)

    @property
    def shots(self) -> int:
        return int(len(self.completed))

    @property
    def complete(self) -> bool:
        """Whether every requested row has been collected."""
        return bool(self.completed.all())

    def pending(self) -> np.ndarray:
        """Positions of the rows still to collect, in increasing order."""
        return np.flatnonzero(~self.completed)

    def install(self, positions: np.ndarray, record: L1Record, work: CollectionWork) -> None:
        """Copy one finished chunk into the buffers and mark its rows completed.

        ``record`` has already passed every ``L1Record`` check, whether it was built in
        this process or rebuilt from a worker's plain arrays. What is checked here is
        that it belongs where it claims: the positions must be in range, not already
        completed, and hold exactly the parent row ids the record carries.
        """
        if positions.dtype.kind not in 'iu' or positions.ndim != 1 or len(positions) == 0:
            raise ValueError(f'chunk positions must be a nonempty integer array, got '
                             f'{positions.dtype} {positions.shape}')
        if (positions < 0).any() or (positions >= self.shots).any():
            raise ValueError(f'chunk positions must lie in [0, {self.shots})')
        if self.completed[positions].any():
            raise ValueError('a chunk was delivered for rows that are already completed')
        if not np.array_equal(record.rows, self.arrays['rows'][positions]):
            raise ValueError('a chunk holds different parent row ids than the positions it claims')
        for name in ARRAY_FIELDS:
            self.arrays[name][positions] = getattr(record, name)
        # Last, and only now: a row counts as collected once all of its arrays are in.
        self.completed[positions] = True
        self.work = self.work + work

    def save(self, path, *, identity: str) -> None:
        """Replace the checkpoint atomically, through one temporary sibling."""
        _save_arrays(path, {
            **self.arrays,
            'completed': self.completed,
            'collection': np.array(identity),
            'work': np.array([getattr(self.work, name) for name in WORK_FIELDS], dtype=np.int64),
            'seconds_total': np.array(self.seconds_total, dtype=np.float64),
            'resumptions': np.array(self.resumptions, dtype=np.int64),
        }, schema=CHECKPOINT_SCHEMA)

    def record(self) -> L1Record:
        """The immutable record these buffers hold, once every row is in.

        ``L1Record`` copies what it is given, so the published record never aliases a
        buffer the coordinator could still write to.
        """
        if not self.complete:
            raise ValueError(f'{int((~self.completed).sum())} of {self.shots} rows are still pending')
        return L1Record.from_arrays(self.arrays)


# --- the workers -------------------------------------------------------------

@dataclass(frozen=True)
class _WorkerState:
    """One worker process's decoders and its read-only view of the packed sample."""
    context: L1Context
    detectors_packed: np.ndarray
    actual_packed: np.ndarray
    num_detectors: int
    num_observables: int


_WORKER: _WorkerState | None = None
"""Set once per worker process by the pool initializer; read only by ``_collect_chunk``."""


def _start_worker(sample_dir: str, dem_text: str, patches: int, num_detectors: int,
                  num_observables: int) -> None:
    """Pool initializer: build this process's decoders once and map the packed sample.

    An ``L1Context`` costs a model parse, the split, the correlation-rule compilation,
    and a joint matcher, so it is built per process rather than per chunk. The payload
    is mapped rather than read, and is deliberately not re-hashed: the coordinator
    verified every byte of this sample before the pool existed, and hashing hundreds of
    megabytes once per worker would cost more than the rows the workers decode.
    """
    global _WORKER
    directory = Path(sample_dir)
    _WORKER = _WorkerState(
        context=L1Context.from_dem_text(dem_text, patches),
        detectors_packed=np.load(directory / DETECTORS_FILE, mmap_mode='r'),
        actual_packed=np.load(directory / ACTUAL_FILE, mmap_mode='r'),
        num_detectors=num_detectors, num_observables=num_observables)


def _collect_chunk(task) -> tuple:
    """Decode one chunk in a worker and reply with plain data.

    ``task`` is ``(positions, rows)``: where the chunk belongs in the coordinator's
    buffers, and which parent rows it covers. The reply carries the record's arrays and
    the work counts as a plain dict of arrays and a plain dict of ints, because the
    immutable mapping wrappers a checked record and its work carry need not be
    pickleable; the coordinator rebuilds both at the receiving boundary.
    """
    positions, rows = task
    state = _WORKER
    if state is None:
        raise RuntimeError('a collection worker was started without its initializer')
    detectors = _unpack_rows(state.detectors_packed, rows, state.num_detectors)
    actual = _unpack_rows(state.actual_packed, rows, state.num_observables)
    collected = collect_rows(state.context, detectors, actual, rows)
    return positions, dict(collected.record.arrays()), collected.work.to_json()


# --- scheduling --------------------------------------------------------------

def _chunks(pending: np.ndarray, chunk_size: int, max_chunks: int | None) -> list[np.ndarray]:
    """Split the pending positions into the chunks this call will schedule."""
    chunks = [pending[start:start + chunk_size] for start in range(0, len(pending), chunk_size)]
    return chunks if max_chunks is None else chunks[:max_chunks]


def _chunk_results(chunks: list[np.ndarray], *, sample: SampleSet, sample_dir: Path,
                   settings: CollectionSettings, context: L1Context | None):
    """Yield ``(positions, record, work)`` for each scheduled chunk, in any order.

    ``workers=1`` decodes in this process with no pool at all, so a failure raises where
    it happened and can be stepped through. More workers fan the same ``collect_rows``
    call out over a forkserver pool; results come back unordered, which the coordinator
    is free to accept because each one names the positions it belongs to.
    """
    if not chunks:
        return
    rows = settings.rows
    if settings.workers == 1:
        for positions in chunks:
            chunk_rows = rows[positions]
            collected = collect_rows(context, *sample.rows(chunk_rows), chunk_rows)
            yield positions, collected.record, collected.work
        return
    pool_context = multiprocessing.get_context(WORKER_START_METHOD)
    tasks = [(positions, rows[positions]) for positions in chunks]
    with pool_context.Pool(settings.workers, initializer=_start_worker,
                           initargs=(str(sample_dir), sample.dem_text, sample.parameters.patches,
                                     sample.num_detectors, sample.num_observables)) as pool:
        for positions, arrays, work in pool.imap_unordered(_collect_chunk, tasks):
            _require_keys(work, WORK_FIELDS, 'the work counts a worker returned')
            yield (np.asarray(positions), L1Record.from_arrays(arrays),
                   CollectionWork(**{name: work[name] for name in WORK_FIELDS}))


def _run_chunks(buffers: _Buffers, chunks: list[np.ndarray], results, *, checkpoint) -> None:
    """Install finished chunks as they arrive, checkpointing on a bounded interval.

    Every chunk must be one that was scheduled and must arrive once: an unexpected or
    repeated delivery means the coordinator and its workers disagree about what is being
    collected, which is not something to reconcile silently. A chunk that never arrives
    at all simply leaves its rows pending, because the buffers only ever gain rows.
    """
    outstanding = {_chunk_key(positions): positions for positions in chunks}
    last_written = time.monotonic()
    for positions, record, work in results:
        if outstanding.pop(_chunk_key(positions), None) is None:
            raise ValueError(f'A chunk covering positions {positions[0]}..{positions[-1]} was not '
                             f'scheduled, or arrived twice')
        buffers.install(positions, record, work)
        if time.monotonic() - last_written >= CHECKPOINT_INTERVAL_SECONDS:
            checkpoint()
            last_written = time.monotonic()


def _chunk_key(positions: np.ndarray) -> bytes:
    """A scheduled chunk's identity, stable across a pickle round trip."""
    return np.ascontiguousarray(positions, dtype=np.int64).tobytes()


# --- collection identity -----------------------------------------------------

def _collection_identities(sample: SampleSet, settings: CollectionSettings) -> dict:
    """The five identities a collected record carries, all computed from content.

    The decoder identity covers the L1 sources, the stored record conventions, and the
    decoder package versions, so a changed algorithm or dependency produces a different
    collection and cannot join an existing one.
    """
    decoder = decoder_identity(sources=source_hashes(DECODER_SOURCES), conventions=RECORD_CONVENTIONS,
                               versions=package_versions(DECODER_PACKAGES))
    return {
        **{name: sample.identities[name] for name in SAMPLE_IDENTITY_NAMES},
        'decoder': decoder,
        'collection': collection_identity(
            parent_sample=sample.identities['parent_sample'], decoder=decoder, role=settings.role,
            shots=settings.shots, rows_sha256=settings.rows_summary()['sha256']),
    }


def _require_same_collection(recorded: Mapping, *, settings: CollectionSettings, identities: Mapping,
                             where: str) -> None:
    """Compare a request with what a directory is already collecting, naming any difference.

    Rows, role, model, parent sample, sampling family, decoder, and the resulting
    collection identity must all be unchanged: a different one of any of them is a
    different collection and belongs in a different directory. Worker and chunk counts
    are deliberately not compared, because they cannot change a decoded value.
    """
    _require_keys(recorded, ('schema_version', 'role', 'rows', 'identities'), where)
    if recorded['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'{where} holds schema {recorded["schema_version"]!r}, '
                         f'expected {SCHEMA_VERSION!r}')
    if recorded['role'] != settings.role:
        raise ValueError(f'{where} is collecting the {recorded["role"]!r} role, '
                         f'this request asks for {settings.role!r}')
    summary = settings.rows_summary()
    _require_keys(recorded['rows'], tuple(summary), f'{where} rows')
    for name, value in summary.items():
        if recorded['rows'][name] != value:
            raise ValueError(f'{where} is collecting rows with {name} {recorded["rows"][name]!r}, '
                             f'this request asks for {value!r}')
    _require_keys(recorded['identities'], tuple(identities), f'{where} identities')
    for name, value in identities.items():
        if recorded['identities'][name] != value:
            raise ValueError(f'{where} records the {name} identity '
                             f'{recorded["identities"][name]!r}, this request has {value!r}')


def _audit_source_hashes() -> dict:
    """Hashes of the audit sources that exist, plus the names of those that do not.

    This is the one place a missing source file is recorded instead of raising. The
    audit group names files that later stages create, and no identity is computed from
    it: it is there so a published record says which stage boundaries, CLI, and circuit
    generator were in the tree when it was collected.
    """
    present, missing = [], []
    for name in AUDIT_SOURCES:
        (present if (REPOSITORY_ROOT / name).is_file() else missing).append(name)
    return {**source_hashes(present), 'missing': missing}


def _graph_prerequisite(sample: SampleSet, *, workers: int, recorded, check: str
                        ) -> tuple[L1Context | None, dict]:
    """Run the graph gate, or reuse the result a previous run of this collection recorded.

    The gate must hold before any row is decoded: if the six check graphs do not
    reproduce the joint graph, every value collected through them would describe a
    different model. A resumption reuses the recorded result when the validation code is
    unchanged, so restarting a long collection does not re-import the joint graph; the
    model identity needs no comparison here because the caller has already refused a
    request whose model differs.

    A serial run builds its ``L1Context`` here and lends the gate its split; a parallel
    run builds only the split, because every decoder it needs lives in a worker.
    """
    if recorded is not None and recorded.get('check_identity') == check and 'graph' in recorded:
        graph = dict(recorded['graph'])
        if graph.get('passed') is not True:
            raise ValueError(f'{COLLECTION_FILE} records graph checks that did not pass')
        return None, graph
    dem = stim.DetectorErrorModel(sample.dem_text)
    patches = sample.parameters.patches
    context = L1Context(dem, patches) if workers == 1 else None
    split = context.patches if context is not None else PatchGraphs.from_yoked_dem(dem, num_patches=patches)
    checks = check_graphs(dem, split)
    checks.raise_if_failed()
    return context, checks.to_json()


def _write_collection(out_dir: Path, *, settings: CollectionSettings, identities: Mapping,
                      graph: dict, check: str, recorded) -> None:
    """Publish what this directory is collecting, or refresh its recorded graph gate.

    Everything an identity depends on is written once, on the first run. Only the
    validation code's identity and the graph result it produced are ever rewritten, and
    only when that code has changed; the creation time of the collection is kept.
    """
    write_json_atomic(out_dir / COLLECTION_FILE, {
        'schema_version': SCHEMA_VERSION,
        'role': settings.role,
        'rows': settings.rows_summary(),
        'identities': dict(identities),
        'check_identity': check,
        'graph': graph,
        'created_utc': (recorded or {}).get('created_utc') or utc_now(),
    })


# --- publication -------------------------------------------------------------

def _completion_manifest(*, record: L1Record, sample: SampleSet, settings: CollectionSettings,
                         identities: Mapping, artifacts: Mapping, graph: dict, checks: RecordChecks,
                         check: str, work: CollectionWork, resumptions: int,
                         seconds_this_run: float, seconds_total: float) -> dict:
    """The completion marker, holding everything a later stage must verify or audit.

    ``write_json_atomic`` serializes this through ``json_ready``, so an undefined
    statistic inside a check record is written as ``null`` rather than a nonstandard
    ``NaN`` literal.
    """
    return {
        'schema_version': SCHEMA_VERSION,
        'status': COMPLETE_STATUS,
        'role': settings.role,
        'parameters': sample.parameters.to_json(),
        'seed': sample.seed,
        'parent_shots': sample.shots,
        'shots': record.shots,
        'rows': row_summary(record.rows),
        'parent_payload_sha256': sample.payload_sha256,
        'sample_identity_inputs': sample.identity_inputs(),
        'identities': dict(identities),
        'artifacts': dict(artifacts),
        'checks': {'passed': graph['passed'] and checks.passed, 'identity': check,
                   'graph': graph, 'record': checks.to_json()},
        'versions': {'decoder': package_versions(DECODER_PACKAGES),
                     'check': package_versions(CHECK_PACKAGES)},
        'source_sha256': {'decoder': source_hashes(DECODER_SOURCES),
                          'check': source_hashes(CHECK_SOURCES),
                          'audit': _audit_source_hashes()},
        'code_commit': git_commit(),
        'collection_work': {
            **work.to_json(),
            'resumptions': resumptions,
            'telemetry': 'retained rows only: work attempted on rows lost to an interruption '
                         'before the next checkpoint is redone on the restart and is not counted '
                         'here, and setup, graph checks, and record checks are not decoder work',
        },
        'timing': {'seconds_this_run': seconds_this_run, 'seconds_total': seconds_total},
        'created_utc': utc_now(),
    }


def _publish(out_dir: Path, *, buffers: _Buffers, sample: SampleSet, settings: CollectionSettings,
             identities: Mapping, graph: dict, check: str, seconds_this_run: float,
             hooks: _CollectionHooks) -> LoadedRecord:
    """Gate the collected rows, write the record, and publish the manifest over it.

    Nothing about this order is negotiable. The record checks run on the immutable
    record, not on the buffers; a failure leaves the checkpoint and its diagnostics
    standing and publishes no marker, so the next run resumes rather than starting over.
    The manifest is written last and the checkpoint removed only after it lands, which is
    what makes an orphan ``record.npz`` mean an interrupted publication rather than a
    result.
    """
    if graph['passed'] is not True:
        raise ValueError('the graph gate did not pass, so no record may be published')
    record = buffers.record()
    checks = check_record(record)
    if not checks.passed:
        write_json_atomic(out_dir / FAILED_CHECKS_FILE, {
            'schema_version': SCHEMA_VERSION,
            'collection': identities['collection'],
            'check_identity': check,
            'graph': graph,
            'record': checks.to_json(),
            'created_utc': utc_now(),
        })
        checks.raise_if_failed()
    _save_arrays(out_dir / RECORD_FILE, record.arrays(), schema=RECORD_SCHEMA)
    hooks.after_record()
    manifest = _completion_manifest(
        record=record, sample=sample, settings=settings, identities=identities,
        artifacts={RECORD_FILE: sha256_file(out_dir / RECORD_FILE)}, graph=graph, checks=checks,
        check=check, work=buffers.work, resumptions=buffers.resumptions,
        seconds_this_run=seconds_this_run, seconds_total=buffers.seconds_total)
    hooks.before_manifest()
    write_json_atomic(out_dir / RECORD_MANIFEST, manifest)
    (out_dir / CHECKPOINT_FILE).unlink(missing_ok=True)
    # A failure recorded by an earlier attempt no longer describes this directory.
    (out_dir / FAILED_CHECKS_FILE).unlink(missing_ok=True)
    return load_record(out_dir)


def _recheck(out_dir: Path, *, loaded: LoadedRecord, sample: SampleSet, check: str) -> LoadedRecord:
    """Revalidate a published record under changed validation code and republish it.

    Nothing is decoded again: both gates are functions of the saved model and the stored
    arrays, which is exactly why the check identity is recorded separately from the
    decoder identity. A changed decoder cannot reach this path, because the identity
    comparison that runs first rejects it.
    """
    dem = stim.DetectorErrorModel(sample.dem_text)
    graph = check_graphs(dem, PatchGraphs.from_yoked_dem(dem, num_patches=sample.parameters.patches))
    graph.raise_if_failed()
    checks = check_record(loaded.record)
    checks.raise_if_failed()
    manifest = dict(loaded.manifest)
    manifest['checks'] = {'passed': True, 'identity': check, 'graph': graph.to_json(),
                          'record': checks.to_json()}
    manifest['versions'] = {**dict(loaded.manifest['versions']),
                            'check': package_versions(CHECK_PACKAGES)}
    manifest['source_sha256'] = {**dict(loaded.manifest['source_sha256']),
                                 'check': source_hashes(CHECK_SOURCES)}
    manifest['rechecked_utc'] = utc_now()
    write_json_atomic(out_dir / RECORD_MANIFEST, manifest)
    return load_record(out_dir)


# --- the coordinator ---------------------------------------------------------

def _rows_in_sample(rows: np.ndarray, parent_shots: int) -> None:
    """Every requested row must name a shot the parent sample actually drew."""
    if (rows >= parent_shots).any():
        raise ValueError(f'rows must lie in the range [0, {parent_shots}) of the parent sample, '
                         f'the request reaches {int(rows.max())}')


def collect_sample(sample_dir, out_dir, settings: CollectionSettings, *,
                   hooks: _CollectionHooks | None = None) -> LoadedRecord | None:
    """Collect ``settings.rows`` of a saved sample into ``out_dir``, resumably.

    Returns the verified ``LoadedRecord`` once every requested row is collected and
    published, and ``None`` while rows remain, which happens when ``max_chunks`` stops
    scheduling; calling it again with the same request continues from the checkpoint.
    Calling it again on a finished directory decodes nothing and returns the record it
    already holds.

    Every failure raises. The sample's bytes are re-verified on every call; a request
    whose rows, role, or identities differ from what the directory is already collecting
    is refused by name rather than merged; a corrupt or foreign checkpoint is refused
    rather than replaced; and a failed publication gate keeps the checkpoint and
    publishes nothing.

    ``hooks`` is a test seam documented on ``_CollectionHooks`` and does nothing by
    default.
    """
    if not isinstance(settings, CollectionSettings):
        raise TypeError(f'settings must be a CollectionSettings, got {type(settings).__name__}')
    if hooks is None:
        hooks = _CollectionHooks()
    elif not isinstance(hooks, _CollectionHooks):
        raise TypeError(f'hooks must be a _CollectionHooks, got {type(hooks).__name__}')
    sample_dir, out_dir = Path(sample_dir), Path(out_dir)
    if out_dir.resolve() == sample_dir.resolve():
        # A collection publishes manifest.json, which is also the name a recorded run's
        # sample manifest carries: writing outputs beside the shots could destroy them.
        raise ValueError('a collection must not write into its own sample directory')

    sample = SampleSet.load(sample_dir)          # re-verifies every file and payload hash
    _rows_in_sample(settings.rows, sample.shots)
    identities = _collection_identities(sample, settings)
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded = (read_json(out_dir / COLLECTION_FILE) if (out_dir / COLLECTION_FILE).is_file()
                else None)
    if recorded is not None:
        _require_same_collection(recorded, settings=settings, identities=identities,
                                 where=str(out_dir / COLLECTION_FILE))
    check = check_identity(sources=source_hashes(CHECK_SOURCES),
                           versions=package_versions(CHECK_PACKAGES))
    if (out_dir / RECORD_MANIFEST).is_file():
        return _completed(out_dir, sample=sample, settings=settings, identities=identities,
                          check=check)
    return _collect_pending(out_dir, sample_dir=sample_dir, sample=sample, settings=settings,
                            identities=identities, recorded=recorded, check=check, hooks=hooks)


def _completed(out_dir: Path, *, sample: SampleSet, settings: CollectionSettings,
               identities: Mapping, check: str) -> LoadedRecord:
    """Return what a finished collection already holds, re-verified rather than trusted.

    The manifest is the commit point, so this path never decodes anything. It does
    re-check: the request's identities against the manifest's, then, through
    ``load_record``, the schema, the status, the recorded checks, every artifact hash,
    the stored row ids, and the recomputed collection identity. When only the validation
    code has moved, the stored arrays are rechecked in place and the manifest
    republished with the new check identity and results.

    Once the record has loaded, any checkpoint or failure diagnostics beside it are
    removed. ``_publish`` unlinks both after writing the manifest, but a crash in that
    window leaves files describing a collection that is now finished; a later run would
    otherwise keep reporting a published directory as one holding pending work.
    """
    manifest_path = out_dir / RECORD_MANIFEST
    _require_same_collection(read_json(manifest_path), settings=settings, identities=identities,
                             where=str(manifest_path))
    loaded = load_record(out_dir)
    (out_dir / CHECKPOINT_FILE).unlink(missing_ok=True)
    (out_dir / FAILED_CHECKS_FILE).unlink(missing_ok=True)
    if loaded.manifest['checks']['identity'] == check:
        return loaded
    return _recheck(out_dir, loaded=loaded, sample=sample, check=check)


def _collect_pending(out_dir: Path, *, sample_dir: Path, sample: SampleSet,
                     settings: CollectionSettings, identities: Mapping, recorded, check: str,
                     hooks: _CollectionHooks) -> LoadedRecord | None:
    """Collect whatever rows are still outstanding, then checkpoint and maybe publish."""
    patches = sample.parameters.patches
    context, graph = _graph_prerequisite(sample, workers=settings.workers, recorded=recorded,
                                         check=check)
    if recorded is None or recorded.get('check_identity') != check:
        _write_collection(out_dir, settings=settings, identities=identities, graph=graph,
                          check=check, recorded=recorded)

    checkpoint_path = out_dir / CHECKPOINT_FILE
    buffers = (_Buffers.load(checkpoint_path, rows=settings.rows, patches=patches,
                             identity=identities['collection'])
               if checkpoint_path.is_file()
               else _Buffers.allocate(rows=settings.rows, patches=patches))
    chunks = _chunks(buffers.pending(), settings.chunk_size, settings.max_chunks)
    if chunks and context is None and settings.workers == 1:
        context = L1Context.from_dem_text(sample.dem_text, patches)

    base_seconds, started = buffers.seconds_total, time.monotonic()

    def checkpoint() -> None:
        """Publish the buffers, charging this run's elapsed seconds to the running total."""
        buffers.seconds_total = base_seconds + (time.monotonic() - started)
        hooks.before_checkpoint()
        buffers.save(checkpoint_path, identity=identities['collection'])
        hooks.after_checkpoint()

    results = _chunk_results(chunks, sample=sample, sample_dir=sample_dir, settings=settings,
                             context=context)
    with contextlib.closing(results) as arriving:
        _run_chunks(buffers, chunks, arriving, checkpoint=checkpoint)
    checkpoint()                                  # always, whether max_chunks stopped or work ran out
    if not buffers.complete:
        return None
    return _publish(out_dir, buffers=buffers, sample=sample, settings=settings,
                    identities=identities, graph=graph, check=check,
                    seconds_this_run=buffers.seconds_total - base_seconds, hooks=hooks)
