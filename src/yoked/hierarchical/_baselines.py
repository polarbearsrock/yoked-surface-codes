"""Historical decoder predictions of a recorded run, verified and attached to a record.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 3, 5.3, 9.

The object this module computes is the ``baselines`` mapping of an evaluation
``L1Record``: for each name in ``BASELINE_DECODERS``, the recorded four-decoder run's
saved ``(parent_shots, 2P)`` prediction restricted to the record's parent rows, so that
the joint UF, joint correlated UF, recorded joint MWPM, and built-in correlated MWPM
baselines of spec section 9 can be compared with the hierarchy on exactly the same
shots. Column ``2i + s`` holds patch ``i``, sector ``s``, as everywhere in the record.

Two steps, each a function over verified inputs:

  * ``load_recorded_baselines`` reads a run directory the way collection imports one:
    the circuit, model, and packed payload are re-hashed against the run's own manifest
    through ``SampleSet.load_recorded_run``, and every requested prediction file is
    loaded without pickles, required to hold binary values of the declared shape, and
    re-hashed under the run's little-endian packed convention against ``results.json``.
    The run's recorded implementation provenance (source hashes, versions, commit) is
    carried along unchanged, so that an imported baseline always says which code
    produced it.
  * ``attach_baselines`` gates a loaded record against a loaded run and returns the
    record with the baselines attached plus the provenance block its manifest records.
    It requires the evaluation role, because spec section 5.3 keeps calibration and
    confirmation records free of historical results; the same parent payload, circuit,
    and model; every baseline's sector parity equal to the sampled yoke on every row;
    and the recorded joint MWPM to agree with the collector's own joint MWPM up to the
    collector's tie rule, so that two decodes of one model can differ only where two
    optimal matchings exist.

Nothing here decodes: a baseline is read, verified, and mapped, never recomputed, and
the parity and tie gates are the algebraic checks of ``_collect.check_record`` applied
to the imported columns. ``_provenance.py`` therefore lists this file in
``CHECK_SOURCES``: it gates data and produces no L1 output, so a change to it changes
what was verified, never what was decoded.

``_baselines_test.py`` checks the name-to-stem mapping, the packed hash convention on a
hand-packed example, the loader against a fake run built from the distance-3 fixture
with every tamper (results hash, shape, values, missing file, missing provenance)
rejected by name, the row mapping and manifest block of an import, and each gate: the
role, the payload, circuit, and model hashes, the parity rule, and a joint-MWPM
disagreement that is not a cost tie, with one that is a tie accepted and counted.
"""
from __future__ import annotations

import dataclasses
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from yoked.hierarchical._arrays import readonly_array
# The run is verified the way collection imports it, and the two gates reuse the
# collector's own parity and total-cost helpers, so an imported baseline is held to the
# rules the collector applies to its own joint decode; a second implementation of either
# could drift from the one ``check_record`` enforces.
from yoked.hierarchical._collect import (
    CIRCUIT_FILE, COST_TOLERANCE, DEM_FILE, MAX_DIAGNOSTIC_ROWS, RECORDED_MANIFEST, ROLES,
    SampleSet, _hex_digest, _require_keys, _sector_parity, _total_forced_cost,
)
from yoked.hierarchical._l1 import _whole
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._provenance import read_json, sha256_file
from yoked.hierarchical._record import L1Record, LoadedRecord, baseline_prediction_sha256

BASELINE_DECODERS = MappingProxyType({
    'joint_mwpm_recorded': 'mwpm',
    'joint_uf': 'uf',
    'joint_correlated_mwpm': 'correlated_mwpm',
    'joint_correlated_uf': 'correlated_uf',
})
"""Baseline name, as the record and the report know it, to the recorded run's file stem:
predictions live in ``<stem>_predictions.npy`` and their hashes under
``decoders.<stem>.prediction_packed_sha256`` in ``results.json``. The recorded joint
MWPM is named apart from the record's own ``joint_mwpm`` column, which the collector
recomputes, because the two are compared rather than assumed equal."""

RECORDED_JOINT_MWPM = 'joint_mwpm_recorded'
"""The one baseline that decodes the same model with the same algorithm as the record's
``joint_mwpm`` column, and is therefore gated against it."""

RECORDED_RESULTS = 'results.json'
"""The recorded runs' results file, which declares each decoder's prediction hash."""

PREDICTION_SUFFIX = '_predictions.npy'
"""What follows a decoder's stem in its prediction file name."""

RECORDED_PROVENANCE_FIELDS = ('versions', 'source_sha256', 'code_commit')
"""The recorded implementation provenance a run must declare before its predictions can
be imported (spec section 5.3): the package versions, the source hashes, and the commit
its script recorded. They are carried into the record manifest unchanged."""

BASELINES_FIELD = 'baselines'
"""The record manifest block an import publishes, beside the fields ``load_record``
verifies. It carries the run's provenance and the gate results; the arrays themselves
travel inside ``record.npz`` under the record's ``baseline_`` prefix."""

ROW_MAPPING = "the record's parent row ids index the run's prediction rows"
"""How a baseline reaches the record, written into the manifest so the mapping is never
implicit: row ``r`` of a baseline is row ``rows[r]`` of the run's saved prediction."""

CALIBRATION_ROLE, EVALUATION_ROLE = ROLES
"""Baselines attach to the evaluation role only."""

_NUMERIC_KINDS = 'buif'
"""NumPy dtype kinds a saved prediction may use: bool, unsigned, signed, float. Anything
else cannot hold bits and is a file error rather than something to cast."""


def prediction_file(stem: str) -> str:
    """The recorded run's prediction file for one decoder stem."""
    return f'{stem}{PREDICTION_SUFFIX}'


def packed_prediction_sha256(prediction) -> str:
    """The recorded runs' ``prediction_packed_sha256`` of a ``(shots, 2P)`` bit array.

    Each row is bit-packed little-endian into ``ceil(2P / 8)`` bytes and the rows are
    hashed in order, with no ``.npy`` header: the digest identifies the predictions
    rather than a file. This is ``sha256(np.packbits(p, axis=1, bitorder='little'))``,
    which is what the recorded scripts wrote.
    """
    array = np.asarray(prediction)
    if array.ndim != 2:
        raise ValueError(f'a prediction array must be two-dimensional, got shape {array.shape}')
    if array.dtype.kind not in _NUMERIC_KINDS or (array.dtype.kind != 'b'
                                                  and not np.isin(array, (0, 1)).all()):
        raise ValueError('a prediction array must contain only 0 and 1')
    return baseline_prediction_sha256(array)


def _baseline_names(names: Iterable[str]) -> tuple[str, ...]:
    """The requested baseline names, checked and put in ``BASELINE_DECODERS`` order.

    The canonical order makes the record layout and the manifest block independent of
    the order the names were typed in.
    """
    requested = tuple(names)
    if not requested:
        raise ValueError('an import needs at least one baseline name')
    unknown = [name for name in requested if name not in BASELINE_DECODERS]
    if unknown:
        raise ValueError(f'unknown baseline names {unknown}; the baselines are {list(BASELINE_DECODERS)}')
    if len(set(requested)) != len(requested):
        raise ValueError(f'baseline names must be distinct, got {list(requested)}')
    return tuple(name for name in BASELINE_DECODERS if name in requested)


def _string_mapping(value, name: str) -> MappingProxyType:
    """An immutable copy of a manifest mapping of strings to nonempty strings."""
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping, got {type(value).__name__}')
    copy = dict(value)
    for key, item in copy.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item:
            raise ValueError(f'{name} must map names to nonempty strings, got {key!r}: {item!r}')
    return MappingProxyType(copy)


@dataclass(frozen=True)
class RecordedBaselines:
    """A recorded run's saved predictions, verified, with the provenance it declared.

    Fields:

    - ``directory``: the run directory, for the manifest's audit block; no identity
      depends on it.
    - ``manifest_sha256`` / ``results_sha256``: hashes of the run's ``manifest.json``
      and ``results.json`` as read.
    - ``circuit_sha256`` / ``dem_sha256`` / ``payload_sha256``: the run's verified
      circuit, model, and packed-payload hashes, which a record must share before a
      baseline may be attached.
    - ``parent_shots``: rows in the run's single sampling call (shots).
    - ``versions`` / ``source_sha256``: immutable copies of the run's own recorded
      package versions and source hashes; ``code_commit``: the commit its script
      recorded, or None when it could not.
    - ``predictions``: immutable mapping from baseline name to an owned read-only
      ``(parent_shots, 2P)`` bool array, in ``BASELINE_DECODERS`` order.
    - ``prediction_sha256``: immutable mapping from the same names to the packed hash
      of each array, which must reproduce it.
    """
    directory: Path
    manifest_sha256: str
    results_sha256: str
    circuit_sha256: str
    dem_sha256: str
    payload_sha256: str
    parent_shots: int
    versions: Mapping[str, str]
    source_sha256: Mapping[str, str]
    code_commit: str | None
    predictions: Mapping[str, np.ndarray]
    prediction_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        set_field = object.__setattr__
        set_field(self, 'directory', Path(self.directory))
        for name in ('manifest_sha256', 'results_sha256', 'circuit_sha256', 'dem_sha256',
                     'payload_sha256'):
            set_field(self, name, _hex_digest(getattr(self, name), name))
        set_field(self, 'parent_shots', _whole(self.parent_shots, 'parent_shots', minimum=1))
        set_field(self, 'versions', _string_mapping(self.versions, 'versions'))
        set_field(self, 'source_sha256', _string_mapping(self.source_sha256, 'source_sha256'))
        if self.code_commit is not None and not isinstance(self.code_commit, str):
            raise ValueError(f'code_commit must be a string or None, got {self.code_commit!r}')
        for name in ('predictions', 'prediction_sha256'):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f'{name} must be a mapping by baseline name, '
                                f'got {type(getattr(self, name)).__name__}')
        names = _baseline_names(self.predictions.keys())
        if set(self.prediction_sha256) != set(names):
            raise ValueError(f'prediction_sha256 names {sorted(self.prediction_sha256)}, '
                             f'the predictions are {list(names)}')
        predictions, hashes = {}, {}
        columns = None
        for name in names:
            array = _prediction_bits(self.predictions[name], f'baseline {name!r}',
                                     shots=self.parent_shots, columns=columns)
            columns = array.shape[1]
            digest = _hex_digest(self.prediction_sha256[name], f'prediction_sha256 of {name!r}')
            if packed_prediction_sha256(array) != digest:
                raise ValueError(f'the declared hash of baseline {name!r} does not reproduce its predictions')
            predictions[name], hashes[name] = array, digest
        set_field(self, 'predictions', MappingProxyType(predictions))
        set_field(self, 'prediction_sha256', MappingProxyType(hashes))

    @property
    def columns(self) -> int:
        """Columns of every prediction, 2P."""
        return int(next(iter(self.predictions.values())).shape[1])


def _prediction_bits(value, name: str, *, shots: int, columns: int | None) -> np.ndarray:
    """An owned read-only bool array of shape ``(shots, columns)``; values checked first.

    ``columns`` is None for the first array of a run, whose width every later one must
    match; it must be a whole number of sector pairs either way.
    """
    array = np.asarray(value)
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f'{name} must be numeric, got dtype {array.dtype}')
    expected = f'(parent_shots = {shots}, {NUM_SECTORS} * patches)' if columns is None \
        else f'(parent_shots = {shots}, {columns})'
    well_shaped = (array.ndim == 2 and array.shape[0] == shots
                   and array.shape[1] >= NUM_SECTORS and array.shape[1] % NUM_SECTORS == 0
                   and (columns is None or array.shape[1] == columns))
    if not well_shaped:
        raise ValueError(f'{name} must hold a {expected} prediction array, got shape {array.shape}')
    if array.dtype.kind != 'b' and not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain only 0 and 1')
    return readonly_array(array, dtype=bool)


def _load_prediction(path: Path, *, shots: int, columns: int) -> np.ndarray:
    """One decoder's saved prediction, refused unless it is a plain array of the declared shape."""
    if not path.is_file():
        raise ValueError(f'{path.name} is missing from {path.parent}')
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f'{path.name} is not a readable array file') from exc
    return _prediction_bits(array, path.name, shots=shots, columns=columns)


def load_recorded_baselines(run_dir, names: Iterable[str] = BASELINE_DECODERS.keys()
                            ) -> RecordedBaselines:
    """Read and verify the named decoders' predictions of a recorded four-decoder run.

    The run's circuit, model, and packed payload are re-hashed against its manifest
    exactly as collection imports them; the manifest must also carry
    ``RECORDED_PROVENANCE_FIELDS``. Each prediction file is loaded with
    ``allow_pickle=False``, must hold binary values of shape ``(shots, 2 * patches)``
    with ``shots`` and ``patches`` taken from the manifest's parameters, and must hash
    under the packed convention to the value ``results.json`` declares for its stem.
    Every failure is a ``ValueError`` naming the file or field.
    """
    names = _baseline_names(names)
    run_dir = Path(run_dir)
    sample = SampleSet.load_recorded_run(run_dir)            # re-verifies circuit, model, payload
    manifest = read_json(run_dir / RECORDED_MANIFEST)
    _require_keys(manifest, RECORDED_PROVENANCE_FIELDS, f'{run_dir / RECORDED_MANIFEST}')
    results_path = run_dir / RECORDED_RESULTS
    if not results_path.is_file():
        raise ValueError(f'{results_path} is missing, so the run declares no prediction hashes')
    results = read_json(results_path)
    _require_keys(results, ('decoders',), str(results_path))
    predictions, hashes = {}, {}
    for name in names:
        stem = BASELINE_DECODERS[name]
        _require_keys(results['decoders'], (stem,), f'{RECORDED_RESULTS} decoders')
        _require_keys(results['decoders'][stem], ('prediction_packed_sha256',),
                      f'{RECORDED_RESULTS} decoders.{stem}')
        declared = _hex_digest(results['decoders'][stem]['prediction_packed_sha256'],
                               f'{RECORDED_RESULTS} decoders.{stem}.prediction_packed_sha256')
        path = run_dir / prediction_file(stem)
        array = _load_prediction(path, shots=sample.shots, columns=sample.num_observables)
        digest = packed_prediction_sha256(array)
        if digest != declared:
            raise ValueError(f'{path.name} hashes to {digest}, {RECORDED_RESULTS} declares '
                             f'{declared} for decoders.{stem}.prediction_packed_sha256')
        predictions[name], hashes[name] = array, digest
    return RecordedBaselines(
        directory=run_dir, manifest_sha256=sha256_file(run_dir / RECORDED_MANIFEST),
        results_sha256=sha256_file(results_path), circuit_sha256=sample.circuit_sha256,
        dem_sha256=sample.dem_sha256, payload_sha256=sample.payload_sha256,
        parent_shots=sample.shots, versions=manifest['versions'],
        source_sha256=manifest['source_sha256'], code_commit=manifest['code_commit'],
        predictions=predictions, prediction_sha256=hashes)


# --- attaching ---------------------------------------------------------------

def _require_same_sample(loaded: LoadedRecord, recorded: RecordedBaselines) -> None:
    """A baseline may only join a record collected from the very same sampling call.

    The payload hash is the sample; the circuit and model hashes say the run decoded
    the model the record was collected against. All three are compared, and the shot
    count too, so that a mismatch is named by what differs rather than by a symptom.
    """
    manifest = loaded.manifest
    inputs = manifest['sample_identity_inputs']
    for name, stored, offered in (
            ('packed payload', manifest['parent_payload_sha256'], recorded.payload_sha256),
            (CIRCUIT_FILE, inputs['circuit_sha256'], recorded.circuit_sha256),
            (DEM_FILE, inputs['dem_sha256'], recorded.dem_sha256)):
        if stored != offered:
            raise ValueError(f'{loaded.directory} was collected from a sample whose {name} hashes '
                             f'to {stored}; {recorded.directory} hashes to {offered}')
    if manifest['parent_shots'] != recorded.parent_shots:
        raise ValueError(f'{loaded.directory} was collected from a call of '
                         f'{manifest["parent_shots"]} shots; {recorded.directory} holds '
                         f'{recorded.parent_shots}')
    columns = NUM_SECTORS * loaded.record.num_patches
    if recorded.columns != columns:
        raise ValueError(f'{loaded.directory} holds {columns} prediction columns, '
                         f'{recorded.directory} holds {recorded.columns}')


def _offending(record: L1Record, mask: np.ndarray) -> list[int]:
    """The first few parent row ids where ``mask`` holds, for an error message."""
    positions = np.flatnonzero(mask)[:MAX_DIAGNOSTIC_ROWS]
    return [int(row) for row in record.rows[positions]]


def _require_yoke_parity(record: L1Record, name: str, baseline: np.ndarray) -> None:
    """Every baseline's per-sector parity must equal the sampled yoke bit on every row,
    the same rule ``check_record`` applies to the collector's own joint decode."""
    violations = (_sector_parity(baseline) != record.yoke).any(axis=1)
    if violations.any():
        raise ValueError(f'baseline {name!r} breaks yoke parity on {int(violations.sum())} rows; '
                         f'parent rows {_offending(record, violations)}')


def _joint_mwpm_agreement(record: L1Record, baseline: np.ndarray) -> dict:
    """Gate the recorded joint MWPM against the collector's, under the collector's tie rule.

    Both decoded the same model with PyMatching, so they may differ only on rows where
    two optimal matchings exist: every disagreeing row must have equal total forced cost
    within ``COST_TOLERANCE``, exactly as ``check_record`` holds its reconstructed final
    prediction to the joint decode. The returned block records the agreement fraction,
    the disagreement count, and how many of those the tie rule explains.
    """
    agree = (baseline == record.joint_mwpm).all(axis=1)
    difference = np.abs(_total_forced_cost(record.forced_plain, baseline)
                        - _total_forced_cost(record.forced_plain, record.joint_mwpm))
    tied = ~agree & (difference <= COST_TOLERANCE)
    unexplained = ~agree & ~tied
    if unexplained.any():
        raise ValueError(f'baseline {RECORDED_JOINT_MWPM!r} disagrees with the collected joint MWPM '
                         f'on {int(unexplained.sum())} rows that are not cost ties (largest total '
                         f'forced cost difference {float(difference[unexplained].max()):.6g} nats, '
                         f'tolerance {COST_TOLERANCE}); parent rows {_offending(record, unexplained)}')
    return {
        'baseline': RECORDED_JOINT_MWPM,
        'compared_with': 'joint_mwpm',
        'agreement': float(agree.mean()),
        'disagreements': int((~agree).sum()),
        'tie_explained': int(tied.sum()),
        'max_disagreement_cost_difference': float(difference[~agree].max()) if (~agree).any() else 0.0,
        'cost_tolerance': COST_TOLERANCE,
    }


def attach_baselines(loaded: LoadedRecord, recorded: RecordedBaselines) -> tuple[L1Record, dict]:
    """Gate a run's baselines against a verified record and attach them to its rows.

    Requires the evaluation role, the same packed payload, circuit, and model as the
    record's sample, every baseline's sector parity to equal the record's yoke on every
    row, and the recorded joint MWPM (when imported) to agree with the collector's up to
    the tie rule. Returns the record with ``baselines`` replaced by the mapped arrays,
    and the provenance block the manifest records: the run's directory, manifest and
    results hashes, per-baseline prediction hashes, the run's own source hashes,
    versions, and commit, the row-mapping convention, and the two gates' results.
    """
    if not isinstance(loaded, LoadedRecord):
        raise TypeError(f'loaded must be a LoadedRecord, got {type(loaded).__name__}')
    if not isinstance(recorded, RecordedBaselines):
        raise TypeError(f'recorded must be a RecordedBaselines, got {type(recorded).__name__}')
    role = loaded.manifest['role']
    if role != EVALUATION_ROLE:
        raise ValueError(f'{loaded.directory} holds a {role!r} record; historical baselines attach '
                         f'only to the {EVALUATION_ROLE!r} role')
    _require_same_sample(loaded, recorded)
    record = loaded.record
    rows = record.rows
    if int(rows.max()) >= recorded.parent_shots:
        raise ValueError(f'{loaded.directory} holds parent row {int(rows.max())}, beyond the '
                         f'{recorded.parent_shots} rows of {recorded.directory}')
    baselines: dict[str, np.ndarray] = {}
    for name, prediction in recorded.predictions.items():
        baseline = prediction[rows]                        # fancy indexing copies the rows
        _require_yoke_parity(record, name, baseline)
        baselines[name] = baseline
    joint = (_joint_mwpm_agreement(record, baselines[RECORDED_JOINT_MWPM])
             if RECORDED_JOINT_MWPM in baselines else None)
    block = {
        'names': list(baselines),
        'row_mapping': ROW_MAPPING,
        'run': {
            'directory': str(recorded.directory.resolve()),
            'manifest_sha256': recorded.manifest_sha256,
            'results_sha256': recorded.results_sha256,
            'circuit_sha256': recorded.circuit_sha256,
            'dem_sha256': recorded.dem_sha256,
            'payload_sha256': recorded.payload_sha256,
            'parent_shots': recorded.parent_shots,
            'versions': dict(recorded.versions),
            'source_sha256': dict(recorded.source_sha256),
            'code_commit': recorded.code_commit,
        },
        'decoders': {name: {'stem': BASELINE_DECODERS[name],
                            'file': prediction_file(BASELINE_DECODERS[name]),
                            'prediction_packed_sha256': recorded.prediction_sha256[name],
                            'mapped_prediction_sha256': baseline_prediction_sha256(baseline)}
                     for name, baseline in baselines.items()},
        'yoke_parity': {'rows': record.shots, 'violations': 0},
        'joint_mwpm': joint,
    }
    return dataclasses.replace(record, baselines=baselines), block
