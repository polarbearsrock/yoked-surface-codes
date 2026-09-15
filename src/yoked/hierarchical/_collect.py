"""Verified sample sets, single-process L1 collection, and the graph and record checks.

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
  * ``L1Context`` owns the per-process decoders: the patch split, one cluster-gap UF
    decoder and one forced-weight matcher per patch, and the joint PyMatching matcher
    used only for validation.
  * ``collect_rows`` fills one ``L1Record`` for a chunk of parent rows through a single
    implementation path, validating every UF correction against ``H c = s`` and
    ``L c = r`` as it goes, and reports the decoder work that actually ran as a
    ``CollectionWork``. Task 9's chunked parallel collection calls this same function,
    which is why a ``workers=1`` run needs no process pool to reproduce a failure.
  * ``check_graphs`` and ``check_record`` are the two gates. The first is a collection
    prerequisite: the six check graphs, with check vertices mapped back onto the two
    yokes, must reproduce the imported joint graph edge for edge. The second is the
    publication gate: check parity, additivity, preferred-class consistency, yoke
    parity, and zero unexplained disagreements between the reconstructed
    MWPM-reference pipeline and joint MWPM. Neither encodes an agreement percentage;
    every invariant is exact, and a failure names the offending parent rows.

``collect_rows`` validates corrections but deliberately does not call ``check_record``:
the record checks are defined on a whole record and are what Task 9 runs, and must
pass, before it publishes one.

``_collect_test.py`` checks the sample round trip and each hash it verifies, the
imported recorded-run format with a tampered file rejected, graph equivalence against
deliberately changed edge multiplicity, mask, and weight (a difference below ``1e-9``
passing and a larger one failing), collection of a few hundred distance-3 shots through
both correlation branches, equality of one call with a partitioned serial collection,
the work counts under instrumented decoders, and every record invariant failing on its
own corrupted field.
"""
from __future__ import annotations

import collections
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._matching_gaps import CHECK_PATTERNS, MatchingGaps, signed_gaps
from yoked.hierarchical._outer_decoder import exact_outer_map_batch, frame_adjusted_syndrome
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph, PatchGraphs
from yoked.hierarchical._provenance import (
    MODEL_PACKAGES, SAMPLE_CONVENTIONS, SAMPLING_PACKAGES, SCHEMA_VERSION, atomic_replacement,
    model_identity, package_versions, packed_sample_hash, parent_sample_identity, read_json,
    sampling_family_identity, sha256_bytes, sha256_file, utc_now, write_json_atomic,
)
from yoked.hierarchical._record import L1Record, by_sector, to_columns

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
"""The manifest name of the recorded four-decoder runs this module imports."""

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

FORCED_CALLS_PER_DECODE = len(CHECK_PATTERNS)
"""Forced decodes per patch and variant. A matching count means one decoded syndrome,
so the four-row ``decode_batch`` behind ``forced_weights`` contributes four."""

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

def _whole(value, name: str, *, minimum: int) -> int:
    """A Python int from a whole, in-range value; ``True`` is a flag, not a count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}, got {value}')
    return int(value)


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


def _binary(value, name: str, *, columns: int) -> np.ndarray:
    """An (rows, columns) boolean array; 0.5 is rejected rather than cast to True."""
    array = np.asarray(value)
    if array.dtype.kind not in 'buif':
        raise ValueError(f'{name} must be numeric, got dtype {array.dtype}')
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f'{name} must have shape (shots, {columns}), got {array.shape}')
    if array.dtype.kind != 'b' and not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain only 0 and 1')
    return array.astype(bool, copy=False)


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


def _parent_rows(value, shots: int) -> np.ndarray:
    """The parent sample row of each collected shot: whole, nonnegative, and increasing."""
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise ValueError(f'rows must be an integer array of parent row ids, got dtype {array.dtype}')
    if array.ndim != 1 or len(array) != shots:
        raise ValueError(f'rows must have shape ({shots},), got {array.shape}')
    if not np.isfinite(array).all() or (array.dtype.kind == 'f' and not (array == np.floor(array)).all()):
        raise ValueError('rows must be whole numbers')
    if (array < 0).any():
        raise ValueError('rows must be nonnegative')
    if shots > 1 and not (np.diff(array) > 0).all():
        raise ValueError('rows must be unique and increasing')
    return array.astype(np.int64)


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


def _sample_identities(*, parameters: CircuitParameters, circuit_sha256: str, dem_sha256: str,
                       num_detectors: int, num_observables: int, seed: int, shots: int,
                       payload_sha256: str, model_versions: Mapping, sampling_versions: Mapping) -> dict:
    """The three identities a sample determines, all from content rather than paths."""
    model = model_identity(
        parameters=parameters.to_json(), circuit_sha256=circuit_sha256, dem_sha256=dem_sha256,
        num_detectors=num_detectors, num_observables=num_observables,
        conventions=SAMPLE_CONVENTIONS, versions=model_versions)
    return {
        'model': model,
        'parent_sample': parent_sample_identity(model=model, seed=seed, parent_shots=shots,
                                                payload_sha256=payload_sha256, versions=sampling_versions),
        'sampling_family': sampling_family_identity(circuit_sha256=circuit_sha256, seed=seed,
                                                    versions=sampling_versions),
    }


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
        detectors = np.unpackbits(self.detectors_packed[positions], axis=1,
                                  count=self.num_detectors, bitorder='little')
        actual = np.unpackbits(self.actual_packed[positions], axis=1,
                               count=self.num_observables, bitorder='little')
        return detectors.astype(bool), actual.astype(bool)

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
        circuit_text, dem_text = str(circuit), str(dem)
        circuit_sha256 = sha256_bytes(circuit_text.encode('utf-8'))
        dem_sha256 = sha256_bytes(dem_text.encode('utf-8'))
        payload_sha256 = packed_sample_hash(detectors, actual)
        identities = _sample_identities(
            parameters=parameters, circuit_sha256=circuit_sha256, dem_sha256=dem_sha256,
            num_detectors=dem.num_detectors, num_observables=dem.num_observables, seed=seed,
            shots=shots, payload_sha256=payload_sha256,
            model_versions=package_versions(MODEL_PACKAGES),
            sampling_versions=package_versions(SAMPLING_PACKAGES))
        return cls(parameters=parameters, seed=seed, shots=shots, num_detectors=dem.num_detectors,
                   num_observables=dem.num_observables, circuit_sha256=circuit_sha256,
                   dem_sha256=dem_sha256, payload_sha256=payload_sha256, detectors_packed=detectors,
                   actual_packed=actual, circuit_text=circuit_text, dem_text=dem_text,
                   identities=identities,
                   source={'kind': 'generated', 'directory': None, 'manifest_sha256': None})

    def save(self, directory) -> Path:
        """Write the four artifacts and then publish ``sample.json`` over them."""
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
            'versions': package_versions(SAMPLING_PACKAGES),
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
        identities = _sample_identities(
            parameters=parameters, circuit_sha256=hashes[CIRCUIT_FILE], dem_sha256=hashes[DEM_FILE],
            num_detectors=num_detectors, num_observables=num_observables, seed=seed, shots=shots,
            payload_sha256=manifest['payload_sha256'],
            model_versions=_subset_versions(versions, MODEL_PACKAGES),
            sampling_versions=_subset_versions(versions, SAMPLING_PACKAGES))
        # Recomputed from the verified content rather than trusted: a manifest whose
        # declared identity does not follow from its own artifacts is not a sample.
        if dict(manifest['identities']) != identities:
            raise ValueError('The declared sample identities do not follow from the saved artifacts')
        return cls(parameters=parameters, seed=seed, shots=shots, num_detectors=num_detectors,
                   num_observables=num_observables, circuit_sha256=hashes[CIRCUIT_FILE],
                   dem_sha256=hashes[DEM_FILE], payload_sha256=manifest['payload_sha256'],
                   detectors_packed=detectors, actual_packed=actual, circuit_text=circuit_text,
                   dem_text=dem_text, identities=identities,
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
        identities = _sample_identities(
            parameters=parameters, circuit_sha256=circuit_sha256, dem_sha256=dem_sha256,
            num_detectors=num_detectors, num_observables=num_observables, seed=seed, shots=shots,
            payload_sha256=payload_sha256,
            model_versions=_subset_versions(versions, MODEL_PACKAGES),
            sampling_versions=_subset_versions(versions, SAMPLING_PACKAGES))
        return cls(parameters=parameters, seed=seed, shots=shots, num_detectors=num_detectors,
                   num_observables=num_observables, circuit_sha256=circuit_sha256,
                   dem_sha256=dem_sha256, payload_sha256=payload_sha256, detectors_packed=detectors,
                   actual_packed=actual, circuit_text=circuit_text, dem_text=dem_text,
                   identities=identities,
                   source={'kind': 'imported', 'directory': str(directory),
                           'manifest_sha256': sha256_file(directory / RECORDED_MANIFEST)})


# --- the per-process decoders ------------------------------------------------

class L1Context:
    """The decoders one process reuses across every chunk it collects.

    Attributes: ``dem`` the joint model; ``patches`` its ``PatchGraphs`` split;
    ``decoders`` one ``ClusterGapUnionFindDecoder`` per patch; ``matchers`` one
    ``MatchingGaps`` per patch; ``joint`` the joint PyMatching matcher, used only for
    the validation decode that ``check_record`` compares against.

    Building one costs a model parse, the split, the correlation-rule compilation, and
    the joint matcher, which is why a worker builds it once per process rather than once
    per chunk. It deliberately does not run ``check_graphs``: that is a prerequisite the
    coordinator runs once per distinct model before any worker starts.
    """

    def __init__(self, dem: stim.DetectorErrorModel, num_patches: int):
        self.dem = dem
        self.patches = PatchGraphs.from_yoked_dem(dem, num_patches=num_patches)
        self.decoders = tuple(ClusterGapUnionFindDecoder(patch.graph) for patch in self.patches)
        self.matchers = tuple(
            MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))
            for patch in self.patches)
        self.joint = pymatching.Matching.from_detector_error_model(dem)

    @classmethod
    def from_dem_text(cls, text: str, num_patches: int) -> L1Context:
        """Build a context from the sample's saved model text."""
        return cls(stim.DetectorErrorModel(text), num_patches)

    @property
    def num_patches(self) -> int:
        return len(self.patches)


# --- collection --------------------------------------------------------------

@dataclass(frozen=True)
class CollectionWork:
    """Decoder work that actually ran while collecting a set of rows.

    Every field is a whole nonnegative count over the rows collected. Setup (building an
    ``L1Context``) and retries are recorded by the collection stage, not here.

    - ``rows``: parent rows collected.
    - ``uf_decodes``: syndromes decoded by the cluster-gap UF decoder (calls).
    - ``dijkstra_searches``: shortest-odd-walk searches run (calls), two per UF decode.
    - ``dijkstra_states``: states settled by those searches (states), the soft-output
      cost proxy.
    - ``unforced_plain_calls``: unforced first-pass matchings on the check-free graph (calls).
    - ``plain_forced_calls``: forced matchings under the plain weights (calls); a
      four-row ``decode_batch`` counts as four decoded syndromes.
    - ``reweight_attempts``: attempts to apply the correlation rules (calls), one per
      patch and row whether or not a rule fired.
    - ``correlated_forced_calls``: forced matchings under reweighted weights (calls),
      run only for the patches where a rule fired.
    - ``correlated_validation_calls``: unforced second-pass matchings under reweighted
      weights (calls), same condition.
    - ``joint_decodes``: joint PyMatching validation decodes on the full model (calls).
    """
    rows: int
    uf_decodes: int
    dijkstra_searches: int
    dijkstra_states: int
    unforced_plain_calls: int
    plain_forced_calls: int
    reweight_attempts: int
    correlated_forced_calls: int
    correlated_validation_calls: int
    joint_decodes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name, _whole(getattr(self, field.name), field.name, minimum=0))

    def __add__(self, other) -> CollectionWork:
        """Totals of two chunks, so a parallel collection sums what its workers reported."""
        if not isinstance(other, CollectionWork):
            return NotImplemented
        return CollectionWork(**{name: getattr(self, name) + getattr(other, name) for name in WORK_FIELDS})

    def to_json(self) -> dict:
        return {name: getattr(self, name) for name in WORK_FIELDS}


WORK_FIELDS = tuple(field.name for field in dataclasses.fields(CollectionWork))
"""The work counters in declaration order."""


@dataclass(frozen=True)
class CollectedRows:
    """What one ``collect_rows`` call produced: the checked ``record`` and its ``work``."""
    record: L1Record
    work: CollectionWork

    def __post_init__(self) -> None:
        if not isinstance(self.record, L1Record):
            raise TypeError(f'record must be an L1Record, got {type(self.record).__name__}')
        if not isinstance(self.work, CollectionWork):
            raise TypeError(f'work must be a CollectionWork, got {type(self.work).__name__}')


def _validate_correction(patch: PatchGraph, result, syndrome: np.ndarray, *, row: int, index: int) -> None:
    """Check ``H c = s`` and ``L c = r`` for one patch's UF correction.

    The graph's endpoints give every boundary edge its own terminal beyond the real
    detectors, so flipping both endpoints of each selected edge and then reading back
    only the detector entries is exactly ``H c``.
    """
    graph = patch.graph
    parity = np.zeros(len(graph.adjacency), dtype=bool)
    mask = 0
    for edge_id in result.selected_edges:
        u, v = graph.endpoints[edge_id]
        parity[u] ^= True
        parity[v] ^= True
        mask ^= graph.edges[edge_id][3]
    if not np.array_equal(parity[:graph.num_detectors], syndrome):
        raise ValueError(f'row {row}, patch {index}: the UF correction does not reproduce its syndrome')
    predicted = np.array([(mask >> k) & 1 for k in range(graph.num_observables)], dtype=bool)
    if not np.array_equal(predicted, result.prediction):
        raise ValueError(f'row {row}, patch {index}: the UF correction flips {predicted.tolist()}, '
                         f'its reference bits are {result.prediction.tolist()}')


def collect_rows(context: L1Context, detectors, actual, rows) -> CollectedRows:
    """Run L1 on a chunk of parent rows and return its record and the work that ran.

    ``detectors`` (k, n_d) and ``actual`` (k, 2P) are the unpacked bits of the parent
    rows named by ``rows``, which must be whole, nonnegative, and increasing. The patch
    decoders never see the yoke bits: they enter the record as the ``yoke`` column and
    are read again only by ``check_record``.
    """
    patches = context.patches
    detectors = _binary(detectors, 'detectors', columns=patches.num_detectors)
    actual = _binary(actual, 'actual', columns=patches.num_observables)
    if len(actual) != len(detectors):
        raise ValueError(f'detectors holds {len(detectors)} rows and actual {len(actual)}')
    rows = _parent_rows(rows, len(detectors))
    shots, num_patches = len(rows), len(patches)
    columns = NUM_SECTORS * num_patches

    uf_reference = np.zeros((shots, columns), dtype=bool)
    mwpm_reference = np.zeros((shots, columns), dtype=bool)
    correlated_prediction = np.zeros((shots, columns), dtype=bool)
    cluster_gap = np.zeros((shots, columns), dtype=np.float64)
    settled_states = np.zeros((shots, columns), dtype=np.int64)
    forced_plain = np.zeros((shots, num_patches, NUM_SECTORS, NUM_SECTORS), dtype=np.float64)
    forced_correlated = np.zeros_like(forced_plain)
    reweighted_patches = np.zeros((shots, num_patches), dtype=bool)

    # Gathered once per patch so the per-row loop only indexes: each patch reads its own
    # detector ids out of the chunk.
    local = [patch.local_syndromes(detectors) for patch in patches]
    joint_mwpm = np.asarray(context.joint.decode_batch(detectors.astype(np.uint8))).astype(bool)

    counts = dict.fromkeys(WORK_FIELDS, 0)
    counts['rows'] = shots
    counts['joint_decodes'] = shots   # decode_batch decodes one syndrome per row
    for position in range(shots):
        row = int(rows[position])
        for index, patch in enumerate(patches):
            syndrome = local[index][position]
            result = context.decoders[index].decode_with_gaps(syndrome)
            _validate_correction(patch, result, syndrome, row=row, index=index)
            forced = context.matchers[index].forced_weights(syndrome)
            sectors = slice(NUM_SECTORS * index, NUM_SECTORS * (index + 1))
            uf_reference[position, sectors] = result.prediction
            cluster_gap[position, sectors] = result.cluster_gap
            settled_states[position, sectors] = result.dijkstra_states
            mwpm_reference[position, sectors] = forced.first_pass
            correlated_prediction[position, sectors] = forced.correlated_prediction
            forced_plain[position, index] = forced.plain
            forced_correlated[position, index] = forced.correlated
            reweighted_patches[position, index] = forced.rules_fired
            counts['uf_decodes'] += 1
            counts['dijkstra_searches'] += len(result.dijkstra_states)
            counts['dijkstra_states'] += int(result.dijkstra_states.sum())
            counts['unforced_plain_calls'] += 1
            counts['plain_forced_calls'] += FORCED_CALLS_PER_DECODE
            counts['reweight_attempts'] += 1
            if forced.rules_fired:
                counts['correlated_forced_calls'] += FORCED_CALLS_PER_DECODE
                counts['correlated_validation_calls'] += 1

    record = L1Record(
        actual=actual, yoke=detectors[:, list(patches.yoke_detector_ids)], uf_reference=uf_reference,
        mwpm_reference=mwpm_reference, correlated_prediction=correlated_prediction,
        joint_mwpm=joint_mwpm, cluster_gap=cluster_gap, dijkstra_states=settled_states,
        forced_plain=forced_plain, forced_correlated=forced_correlated,
        reweighted_patches=reweighted_patches, rows=rows)
    return CollectedRows(record, CollectionWork(**counts))


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
