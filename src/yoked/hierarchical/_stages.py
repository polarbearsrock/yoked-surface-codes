"""The stage boundaries of the experiment: what each stage verifies, and what it publishes.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 3, 6, 9, and 11.

Four stages, each a function over already-verified inputs:

  * ``stage_collect`` turns a ``CollectRequest`` into a collected record. The request
    names exactly one sample source -- a recorded four-decoder run, or the complete
    parameters/seed/full-shot-count triple -- and is checked against the saved sample
    before anything is collected, so an existing ``sample.json`` can never make a new
    argument be ignored. A generated request additionally rebuilds today's circuit and
    model in memory and compares their model identity with the saved one, because a
    changed generator must not quietly hand old shots a new model.
  * ``stage_calibrate`` fits the calibrators of section 6 on a completed calibration
    record and publishes them with the exact record and manifest hashes, the five
    record identities, the fitted rows, the estimator definitions, the knot convention,
    the clipping constant, and the calibration sources and versions.
    ``load_calibrators`` reads that artifact back, rebuilds every knot array through
    ``IsotonicCalibrator.from_json``, checks the declared conventions, and recomputes
    the calibration identity from the artifact's own fields.
  * ``stage_replay`` replays configurations over a completed evaluation record. It
    refuses a calibration-role record, a record sharing the calibration record's parent
    sample or sampling family, a calibration fitted under another model or decoder, and
    a configuration naming an estimator the artifact does not carry, all before a single
    shot is replayed. Each configuration's arrays and results are written under its own
    directory and ``replay_manifest.json`` is published last, over them.
  * ``stage_summarize`` reads only verified replay outputs: the completion manifest and
    its recomputed identity, every declared artifact hash, and the referenced record's
    manifest hash, identities, and exact row summary. It pairs configurations only
    within one record and one estimator pair, and writes the markdown report, the
    machine-readable results, and the summary's own completion manifest last.

Publication order is the whole point of the module. A completion manifest appears only
after the artifacts it stands over, so a directory without one holds an interrupted
publication rather than a result, and a directory with one is reused only when the
recomputed identity and every artifact hash still match. No decoder algorithm lives
here: this module verifies, calls the library, and writes.

``_stages_test.py`` checks a distance-3 pipeline through the functions and once through
the command line; a stopped collection resuming; a changed seed, shot count, parameter
set, recorded run, raw sample byte, decoder version, or generated model rejected by name
even when the output directory already exists; calibration refused on an evaluation
record and replay refused on a calibration record, a shared parent sample, a shared
sampling family, and another model; invalid knots, an altered calibrator, an altered
prediction container, a replaced record, and a missing completion manifest refused; an
interrupted replay publication rerunning cleanly; a reused replay directory returning
its stored results; a confirmation request refused; and a replay-source change leaving
the collection identity untouched.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from yoked.hierarchical._calibration import CLIP, KNOT_CONVENTION, IsotonicCalibrator
from yoked.hierarchical._collect import (
    CIRCUIT_FILE, DEM_FILE, ROLES, SAMPLE_MANIFEST, CircuitParameters, CollectionSettings,
    SampleSet, collect_sample,
)
# The row normalization and the array container are private to this package: a request's
# rows must be normalized exactly as a collection's are, and a replay container is written
# and read the same way a record is.
from yoked.hierarchical._collect import _requested_rows
from yoked.hierarchical._metrics import (
    DEFAULT_REPLICATES, DEFAULT_SEED, SECTOR_NAMES, STRATA, compare_endpoints, summarize_result,
)
from yoked.hierarchical._outer_decoder import TIE_TOLERANCE
from yoked.hierarchical._policies import INITIAL_ONLY, policy_from_name
from yoked.hierarchical._provenance import (
    CALIBRATION_PACKAGES, CALIBRATION_SOURCES, MODEL_PACKAGES, REPLAY_PACKAGES, REPLAY_SOURCES,
    SAMPLE_CONVENTIONS, SCHEMA_VERSION, atomic_replacement, calibration_identity, git_commit,
    model_identity, package_versions, read_json, replay_identity, sha256_bytes, sha256_file,
    source_hashes, utc_now, write_json_atomic,
)
from yoked.hierarchical._record import (
    COMPLETE_STATUS, IDENTITY_NAMES, RECORD_FILE, RECORD_MANIFEST, ROW_SUMMARY_FIELDS,
    LoadedRecord, load_record,
)
from yoked.hierarchical._record import _frozen, _load_arrays, _require_fields, _save_arrays
from yoked.hierarchical._replay import (
    CONFIG_SEPARATOR, ESTIMATOR_SEPARATOR, Calibrators, Estimator, ReplayConfig, ReplayResult,
    WorkCounts, fit_calibrators, replay,
)

SAMPLE_DIRECTORY = 'sample'
"""Where a collect stage keeps the shots it decodes, inside its own output directory. A
collection refuses to write into its sample directory, because both publish a file called
``manifest.json``, so the sample lives one level down rather than beside the record."""

CONFIRMATION_ROLE = 'confirmation'
"""Named here only so that a request for it can be refused by name. Section 3's third set
may be sampled and collected only after the analysis freeze, and freeze verification is a
later milestone, so there is no way to honor such a request correctly today."""

CALIBRATION_ROLE, EVALUATION_ROLE = ROLES
"""The role each stage requires of the record it is handed: fitting reads only the
calibration set, and replay only the held-out evaluation set."""

CALIBRATE_STAGE = 'calibrate'
REPLAY_STAGE = 'replay'
SUMMARIZE_STAGE = 'summarize'
"""The ``stage`` field of each published artifact, so that a file says which stage wrote it
and a calibrator artifact cannot be read as a replay manifest."""

RECORD_BLOCK_FIELDS = ('directory', 'record_sha256', 'manifest_sha256', 'identities', 'rows',
                       'role')
"""What every stage records about the verified record it consumed."""

CALIBRATOR_FIELDS = ('schema_version', 'stage', 'identity', 'record', 'estimators',
                     'knot_convention', 'clip', 'versions', 'source_sha256')
"""The fields ``load_calibrators`` verifies. ``code_commit`` and ``created_utc`` are written
for auditability and are deliberately not among them: neither determines a fitted map."""

ESTIMATOR_FIELDS = ('reference', 'score', 'direction') + SECTOR_NAMES
"""One estimator's entry in a calibrator artifact: its definition and one knot set per sector."""

REPLAY_MANIFEST = 'replay_manifest.json'
"""The completion marker of a replay, published after every configuration's artifacts."""

REPLAY_ARRAYS_FILE = 'arrays.npz'
REPLAY_RESULTS_FILE = 'results.json'
"""One configuration's outputs: the per-shot arrays, and the metrics computed from them."""

REPLAY_ARRAY_SCHEMA = f'ReplayResult/{SCHEMA_VERSION}'
"""Stored inside a replay container, so one written under another layout is rejected rather
than reinterpreted."""

REPLAY_MANIFEST_FIELDS = ('schema_version', 'stage', 'status', 'identity', 'record',
                          'calibrators', 'configurations', 'artifacts', 'tie_rule',
                          'work_convention', 'versions', 'source_sha256')
"""The fields a replay manifest must declare for the summary stage to trust it."""

REPLAY_RESULT_ARRAYS = ('final', 'requested', 'refined_patches', 'ties', 'above_half')
"""The per-shot decision arrays of a ``ReplayResult``; the work counters follow them."""

REPLAY_WORK_FIELDS = tuple(field.name for field in dataclasses.fields(WorkCounts))
"""The eleven per-shot work counters, stored beside the decisions in the same container."""

RESULT_FIELDS = ('config', 'reference', 'pieces', 'initial_estimator', 'refined_estimator',
                 'refined_score', 'policy', 'outer')
"""What a configuration's ``results.json`` declares about itself, beyond the metrics that
``summarize_result`` computes. The summary groups cells by these fields."""

TIE_RULE = f'log-weight ties within {TIE_TOLERANCE} break to the lowest binary pattern'
"""The tie rule L2 actually applies, recorded inside the replay identity: the same
probabilities under another tie rule are a different decoder, and nothing in the stored
predictions would say so."""

WORK_CONVENTION = ('replay-procedure calls: initial work per patch and incremental work per '
                   'distinct refined patch, in the eleven WorkCounts fields; neither the '
                   'collection work that produced the record nor elapsed time')
"""What the eleven stored counters mean, recorded inside the replay identity so that counts
taken under another convention cannot be compared with these."""

CONFIG_FIELDS = ('initial', 'refined', 'policy', 'outer')
REQUIRED_CONFIG_FIELDS = ('initial', 'refined', 'policy')
"""A configuration string's keys; ``outer`` is the only optional one."""

DEFAULT_OUTER = 'mixed'
"""The L2 rule of the pilot's cells: unrequested patches keep their initial probability."""

CONFIG_ARROW = '_to_'
"""What the ``->`` of a configuration name becomes in a directory name. A word rather than a
punctuation mark, so the directory stays a plain identifier on every filesystem."""

DIRECTORY_NAME_PATTERN = re.compile(r'\A[A-Za-z0-9_]+\Z')
"""A configuration directory is letters, digits, and underscores only: no separator a shell
or a path parser could read as structure."""

MARKDOWN_SUFFIX = '.md'
"""The summary's report suffix. ``--out summary.md`` writes ``summary.md``, ``summary.json``,
and ``summary.manifest.json``."""

UNAVAILABLE = 'unavailable'
"""How an undefined rate, interval, or difference is printed. Section 9 requires missing
statistics to be labeled beside their counts, never rendered as a zero."""

PAIRED_METRICS = ('misattribution_pooled', 'block_failure')
"""The two metrics section 9 requires a paired difference for on every headline result."""

REPORT_DIGITS = 6
"""Significant digits in the report's numbers: enough to read a rate of a few times 1e-4
without suggesting precision the shot counts do not support."""


def _whole(value, name: str, *, minimum: int) -> int:
    """A whole number at least ``minimum``. A bool is a flag, not a count, and is refused."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}, got {value}')
    return int(value)


def _no_op() -> None:
    """The default at every hook point: a stage does nothing observable there."""


# --- collect -----------------------------------------------------------------


@dataclass(frozen=True)
class CollectRequest:
    """One collect invocation: where the shots come from, and which of them to decode.

    Fields:

    - ``out_dir``: the collection directory. The verified sample is saved into
      ``out_dir / SAMPLE_DIRECTORY`` and the checkpoint, record, and completion manifest
      are published beside it.
    - ``role``: one of ``ROLES``. It is part of the collection identity, so calibration
      and evaluation rows can never share a record.
    - ``rows``: the parent sample rows to decode, normalized to an owned read-only int64
      array in increasing order, or None for every row of the sample.
    - ``workers``: processes that decode chunks; one means in this process, with no pool.
    - ``chunk_size``: parent rows per scheduled chunk (rows).
    - ``max_chunks``: schedule at most this many chunks per call, then checkpoint and
      return; None schedules every pending chunk.
    - ``recorded_run``: the directory of a recorded four-decoder run to import, whose
      original circuit, model, and shots are used, or None.
    - ``parameters`` / ``seed`` / ``shots``: the generated-sample triple, or all None.
      ``shots`` is the full sampling call, not the number of rows decoded: a pilot subset
      keeps the parent sample identity of the whole call.

    A request names exactly one sample source, and ``workers``, ``chunk_size``, and
    ``max_chunks`` enter no identity, so they may differ between resumptions.
    """

    out_dir: Path
    role: str
    rows: np.ndarray | None = None
    workers: int = 1
    chunk_size: int = 64
    max_chunks: int | None = None
    recorded_run: Path | None = None
    parameters: CircuitParameters | None = None
    seed: int | None = None
    shots: int | None = None

    def __post_init__(self) -> None:
        set_field = object.__setattr__
        set_field(self, 'out_dir', Path(self.out_dir))
        if self.role == CONFIRMATION_ROLE:
            raise ValueError(f'the {CONFIRMATION_ROLE!r} role is unavailable: section 3 allows it '
                             f'only after the analysis freeze, and freeze verification is not '
                             f'implemented, so this milestone collects {ROLES} only')
        if self.role not in ROLES:
            raise ValueError(f'role must be one of {ROLES}, got {self.role!r}')
        generated = (('parameters', self.parameters), ('seed', self.seed), ('shots', self.shots))
        if self.recorded_run is not None:
            supplied = [name for name, value in generated if value is not None]
            if supplied:
                raise ValueError(f'a request names a recorded run or the generated triple, not '
                                 f'both; recorded_run came with {supplied}')
            set_field(self, 'recorded_run', Path(self.recorded_run))
        else:
            missing = [name for name, value in generated if value is None]
            if missing:
                raise ValueError(f'a request must name a recorded_run, or the complete generated '
                                 f'triple of parameters, seed, and shots; missing {missing}')
            if not isinstance(self.parameters, CircuitParameters):
                raise TypeError(f'parameters must be CircuitParameters, '
                                f'got {type(self.parameters).__name__}')
            set_field(self, 'seed', _whole(self.seed, 'seed', minimum=0))
            set_field(self, 'shots', _whole(self.shots, 'shots', minimum=1))
        if self.rows is not None:
            set_field(self, 'rows', _requested_rows(self.rows))
        for name in ('workers', 'chunk_size'):
            set_field(self, name, _whole(getattr(self, name), name, minimum=1))
        if self.max_chunks is not None:
            set_field(self, 'max_chunks', _whole(self.max_chunks, 'max_chunks', minimum=1))

    @property
    def imported(self) -> bool:
        """Whether the shots come from a recorded run rather than from a fresh sampling call."""
        return self.recorded_run is not None

    @property
    def sample_dir(self) -> Path:
        """Where this request's verified sample lives."""
        return self.out_dir / SAMPLE_DIRECTORY


def _generated_model_identity(parameters: CircuitParameters) -> str:
    """The model identity a fresh build of these parameters would have right now.

    The circuit and model are built in memory and hashed, never saved: the question is
    whether today's generator and its dependencies still mean the same model as the one
    the saved shots were drawn under. A changed generator must not quietly hand old shots
    a new model, and comparing paths or parameters alone could not notice.
    """
    circuit = parameters.circuit()
    dem = parameters.dem(circuit)
    # Same trailing newline as SampleSet.sample and Stim's own to_file: the model
    # identity hashes file bytes, so this independent rebuild must hash the same bytes.
    return model_identity(
        parameters=parameters.to_json(),
        circuit_sha256=sha256_bytes((str(circuit) + '\n').encode('utf-8')),
        dem_sha256=sha256_bytes((str(dem) + '\n').encode('utf-8')),
        num_detectors=dem.num_detectors, num_observables=dem.num_observables,
        conventions=SAMPLE_CONVENTIONS, versions=package_versions(MODEL_PACKAGES))


def _require_requested_sample(request: CollectRequest, sample: SampleSet) -> None:
    """Compare a request with the verified sample a directory already holds.

    Every difference raises and names the field, because a saved sample is what later
    stages identify their shots by: reusing a directory must never be a way to relabel
    one sampling call as another.
    """
    where = request.sample_dir
    expected = 'imported' if request.imported else 'generated'
    if sample.source['kind'] != expected:
        raise ValueError(f'{where} holds a {sample.source["kind"]!r} sample, this request asks '
                         f'for an {expected!r} one')
    if request.imported:
        # Loading the run re-verifies its own manifest, so the three hashes compared here
        # are the run's declared and checked content, not whatever is sitting at the path.
        imported = SampleSet.load_recorded_run(request.recorded_run)
        for name, requested, stored in ((CIRCUIT_FILE, imported.circuit_sha256, sample.circuit_sha256),
                                        (DEM_FILE, imported.dem_sha256, sample.dem_sha256),
                                        ('packed payload', imported.payload_sha256,
                                         sample.payload_sha256)):
            if requested != stored:
                raise ValueError(f'{where} was collected from a run whose {name} hashes to '
                                 f'{stored}; {request.recorded_run} hashes to {requested}')
        return
    if request.parameters != sample.parameters:
        raise ValueError(f'{where} holds a sample with parameters {sample.parameters.to_json()}, '
                         f'this request asks for {request.parameters.to_json()}')
    for name, requested, stored in (('seed', request.seed, sample.seed),
                                    ('shots', request.shots, sample.shots)):
        if requested != stored:
            raise ValueError(f'{where} holds a sample with {name} {stored}, this request asks '
                             f'for {requested}')
    model = _generated_model_identity(request.parameters)
    if model != sample.identities['model']:
        raise ValueError(f'{where} holds shots drawn under the model {sample.identities["model"]}; '
                         f'these parameters now build the model {model}, so the generator or one '
                         f'of its dependencies has changed')


def stage_collect(request: CollectRequest) -> LoadedRecord | None:
    """Verify a collect request against its sample, then collect the requested rows.

    Returns the verified record once every requested row is collected and published, and
    None while rows remain, which happens when ``max_chunks`` stops scheduling; calling
    again with the same request continues from the checkpoint.

    A directory that already holds a sample is checked against the request before
    anything is collected: the sample's own bytes are re-verified by ``SampleSet.load``,
    and a generated request must match the saved parameters, seed, and full shot count
    and rebuild the same model, while an imported one must name a run with the same
    circuit, model, and payload. Only then does collection start, and it applies its own
    rules about rows, role, and decoder identity on top.
    """
    if not isinstance(request, CollectRequest):
        raise TypeError(f'request must be a CollectRequest, got {type(request).__name__}')
    sample_dir = request.sample_dir
    if (sample_dir / SAMPLE_MANIFEST).is_file():
        sample = SampleSet.load(sample_dir)                 # re-verifies every saved byte
        _require_requested_sample(request, sample)
    else:
        sample = (SampleSet.load_recorded_run(request.recorded_run) if request.imported
                  else SampleSet.sample(request.parameters, seed=request.seed, shots=request.shots))
        sample.save(sample_dir)
    rows = request.rows if request.rows is not None else np.arange(sample.shots, dtype=np.int64)
    settings = CollectionSettings(role=request.role, rows=rows, workers=request.workers,
                                  chunk_size=request.chunk_size, max_chunks=request.max_chunks)
    return collect_sample(sample_dir, request.out_dir, settings)


# --- calibrate ---------------------------------------------------------------


def _require_role(loaded: LoadedRecord, role: str) -> None:
    """A stage reads only the dataset role it is defined on."""
    if loaded.manifest['role'] != role:
        raise ValueError(f'{loaded.directory} holds a {loaded.manifest["role"]!r} record; this '
                         f'stage requires the {role!r} role')


def _record_block(loaded: LoadedRecord) -> dict:
    """What a stage records about the verified record it consumed.

    The record artifact's hash comes from the manifest ``load_record`` has already
    compared with the file on disk, and the manifest's own hash is taken here, so a later
    stage can tell whether the record it is handed is the one that was actually read. The
    directory is written to find the inputs again and is never an identity input.
    """
    manifest = loaded.manifest
    return {
        'directory': str(loaded.directory.resolve()),
        'record_sha256': manifest['artifacts'][RECORD_FILE],
        'manifest_sha256': sha256_file(loaded.directory / RECORD_MANIFEST),
        'identities': {name: loaded.identities[name] for name in IDENTITY_NAMES},
        'rows': {name: manifest['rows'][name] for name in ROW_SUMMARY_FIELDS},
        'role': manifest['role'],
    }


def _estimators(values: Iterable) -> tuple[Estimator, ...]:
    """The estimators to fit, parsed from names where needed, distinct, in name order."""
    parsed = tuple(value if isinstance(value, Estimator) else Estimator.parse(value)
                   for value in values)
    if not parsed:
        raise ValueError('a calibration needs at least one estimator')
    names = [estimator.name for estimator in parsed]
    if len(set(names)) != len(names):
        raise ValueError(f'estimators must be distinct, got {names}')
    return tuple(sorted(parsed, key=lambda estimator: estimator.name))


def _definitions(estimators: Iterable[Estimator]) -> dict:
    """Each estimator's definition, which is what the calibration identity is taken over.

    The fitted knots are not an input: they follow from the record, the rows, the
    definitions, and the convention, all of which are.
    """
    return {estimator.name: {'reference': estimator.reference, 'score': estimator.score,
                            'direction': estimator.direction} for estimator in estimators}


def stage_calibrate(record_dir, out_path, estimators: Iterable) -> Calibrators:
    """Fit the calibrators of section 6 on a completed calibration record and publish them.

    The record must carry the calibration role and pass ``load_record``, which verifies
    its completion manifest, recorded checks, artifact hashes, rows, and identity. The
    artifact records everything a later stage needs to decide whether these knots may be
    used on another record: the exact record and manifest hashes, the five record
    identities, the fitted rows, the estimator definitions, the knot convention, the
    clipping constant, and the calibration sources and versions.
    """
    loaded = load_record(record_dir)
    _require_role(loaded, CALIBRATION_ROLE)
    estimators = _estimators(estimators)
    calibrators = fit_calibrators(loaded.record, estimators)
    definitions = _definitions(estimators)
    record = _record_block(loaded)
    sources, versions = source_hashes(CALIBRATION_SOURCES), package_versions(CALIBRATION_PACKAGES)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, {
        'schema_version': SCHEMA_VERSION,
        'stage': CALIBRATE_STAGE,
        'identity': calibration_identity(
            record_sha256=record['record_sha256'], rows_sha256=record['rows']['sha256'],
            model=record['identities']['model'], decoder=record['identities']['decoder'],
            estimators=definitions, knot_convention=KNOT_CONVENTION, sources=sources,
            versions=versions),
        'record': record,
        'estimators': {name: {**definition,
                              **{sector: calibrators[name][index].to_json()
                                 for index, sector in enumerate(SECTOR_NAMES)}}
                       for name, definition in definitions.items()},
        'knot_convention': KNOT_CONVENTION,
        'clip': CLIP,
        'versions': versions,
        'source_sha256': sources,
        'code_commit': git_commit(),
        'created_utc': utc_now(),
    })
    return calibrators


def load_calibrators(path) -> tuple[Calibrators, Mapping]:
    """Read a calibrator artifact, rebuilding and re-checking everything it declares.

    Every knot array goes back through ``IsotonicCalibrator.from_json``, which rejects
    centers that are not strictly increasing, probabilities outside the clipping range,
    and probabilities that are not monotone in the declared direction. The declared
    conventions must be the ones this code implements, and the calibration identity must
    follow from the artifact's own record hashes, estimator definitions, convention,
    sources, and versions. Returns the fitted calibrators and an immutable view of the
    artifact.
    """
    path = Path(path)
    artifact = read_json(path)
    _require_fields(artifact, CALIBRATOR_FIELDS, str(path))
    if artifact['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'{path} holds schema {artifact["schema_version"]!r}, '
                         f'expected {SCHEMA_VERSION!r}')
    if artifact['stage'] != CALIBRATE_STAGE:
        raise ValueError(f'{path} was written by the {artifact["stage"]!r} stage, '
                         f'expected {CALIBRATE_STAGE!r}')
    if artifact['knot_convention'] != KNOT_CONVENTION:
        raise ValueError(f'{path} declares the knot_convention {artifact["knot_convention"]!r}, '
                         f'this code implements {KNOT_CONVENTION!r}')
    if artifact['clip'] != CLIP:
        raise ValueError(f'{path} declares clip {artifact["clip"]!r}, this code uses {CLIP!r}')
    _require_fields(artifact['record'], RECORD_BLOCK_FIELDS, f'{path} record')
    _require_fields(artifact['record']['identities'], IDENTITY_NAMES, f'{path} record identities')
    _require_fields(artifact['record']['rows'], ROW_SUMMARY_FIELDS, f'{path} record rows')
    if not isinstance(artifact['estimators'], Mapping) or not artifact['estimators']:
        raise ValueError(f'{path} carries no estimators to calibrate with')
    fitted: dict[str, tuple[IsotonicCalibrator, IsotonicCalibrator]] = {}
    definitions: dict[str, dict] = {}
    for name, entry in artifact['estimators'].items():
        _require_fields(entry, ESTIMATOR_FIELDS, f'{path} estimator {name!r}')
        estimator = Estimator(entry['reference'], entry['score'])
        if estimator.name != name:
            raise ValueError(f'{path} stores the estimator {name!r} under the definition '
                             f'{estimator.name!r}')
        if entry['direction'] != estimator.direction:
            raise ValueError(f'{path} declares direction {entry["direction"]!r} for {name!r}, '
                             f'which is fitted {estimator.direction!r} by definition')
        pair = []
        for sector in SECTOR_NAMES:
            calibrator = IsotonicCalibrator.from_json(entry[sector])
            if calibrator.direction != estimator.direction:
                raise ValueError(f'the {sector} calibrator of {name!r} in {path} was fitted '
                                 f'{calibrator.direction!r}, expected {estimator.direction!r}')
            pair.append(calibrator)
        fitted[name] = tuple(pair)
        definitions[name] = {field: entry[field] for field in ('reference', 'score', 'direction')}
    # Recomputed rather than trusted, exactly as a record's collection identity is: an
    # artifact whose declared identity does not follow from its own inputs is not that fit.
    expected = calibration_identity(
        record_sha256=artifact['record']['record_sha256'],
        rows_sha256=artifact['record']['rows']['sha256'],
        model=artifact['record']['identities']['model'],
        decoder=artifact['record']['identities']['decoder'],
        estimators=definitions, knot_convention=artifact['knot_convention'],
        sources=artifact['source_sha256'], versions=artifact['versions'])
    if expected != artifact['identity']:
        raise ValueError(f'the calibration identity {artifact["identity"]!r} in {path} does not '
                         f'follow from its own record, estimators, convention, sources, and '
                         f'versions')
    return MappingProxyType(fitted), _frozen(artifact)


# --- configurations ----------------------------------------------------------


def parse_config(text: str) -> ReplayConfig:
    """Read one ``initial=ref:score,refined=score,policy=name[,outer=rule]`` configuration.

    ``refined`` names a score rather than an estimator, because both estimators of a
    configuration are defined against the same reference decoder. An unknown key, a
    repeated key, a missing required key, an unsupported estimator pair, and an
    unsupported policy each raise naming what is wrong.
    """
    if not isinstance(text, str) or not text:
        raise ValueError(f'a configuration must be a nonempty string, got {text!r}')
    settings: dict[str, str] = {}
    for part in text.split(','):
        key, separator, value = part.partition('=')
        if not separator or not key or not value:
            raise ValueError(f'expected comma-separated key=value pairs, got {part!r} in {text!r}')
        if key not in CONFIG_FIELDS:
            raise ValueError(f'unknown configuration key {key!r} in {text!r}; the keys are '
                             f'{CONFIG_FIELDS}')
        if key in settings:
            raise ValueError(f'configuration key {key!r} appears twice in {text!r}')
        settings[key] = value
    missing = [name for name in REQUIRED_CONFIG_FIELDS if name not in settings]
    if missing:
        raise ValueError(f'configuration {text!r} declares no {missing}')
    initial = Estimator.parse(settings['initial'])
    if ESTIMATOR_SEPARATOR in settings['refined']:
        raise ValueError(f'refined names a score, not a reference{ESTIMATOR_SEPARATOR}score pair, '
                         f'because both estimators share the initial reference; got '
                         f'{settings["refined"]!r} in {text!r}')
    try:
        refined = Estimator(initial.reference, settings['refined'])
    except ValueError as exc:
        raise ValueError(f'{exc} in configuration {text!r}') from exc
    return ReplayConfig(initial=initial, refined=refined,
                        policy=policy_from_name(settings['policy']),
                        outer=settings.get('outer', DEFAULT_OUTER))


def config_directory_name(config: ReplayConfig) -> str:
    """The directory one configuration's outputs go in: its name, made filesystem-safe.

    The mapping is a fixed textual substitution, so the same configuration always writes
    to the same directory and a reader can recognize the configuration in the path.
    """
    if not isinstance(config, ReplayConfig):
        raise TypeError(f'config must be a ReplayConfig, got {type(config).__name__}')
    name = config.name.replace(CONFIG_SEPARATOR, CONFIG_ARROW).replace(ESTIMATOR_SEPARATOR, '_')
    if not DIRECTORY_NAME_PATTERN.match(name):
        raise ValueError(f'configuration {config.name!r} does not map to a plain directory name, '
                         f'got {name!r}')
    return name


def _configs(values: Iterable) -> tuple[ReplayConfig, ...]:
    """The configurations to replay, parsed where needed, distinct, in one fixed order.

    Sorting by name makes the replay identity independent of the order the configurations
    were typed in, which is what a set of cells should mean.
    """
    parsed = [value if isinstance(value, ReplayConfig) else parse_config(value)
              for value in values]
    if not parsed:
        raise ValueError('a replay needs at least one configuration')
    ordered = tuple(sorted(parsed, key=lambda config: config.name))
    for kind, seen in (('configurations', [config.name for config in ordered]),
                       ('configuration directories', [config_directory_name(config)
                                                      for config in ordered])):
        if len(set(seen)) != len(seen):
            raise ValueError(f'{kind} must be distinct, got {seen}')
    return ordered


# --- replay ------------------------------------------------------------------


@dataclass(frozen=True)
class _ReplayHooks:
    """Where a test may interrupt a replay publication, by raising from one of these.

    ``after_config`` runs once a configuration's arrays and results are on disk and
    ``before_manifest`` just before the completion manifest is published, so a test can
    leave exactly the partial directory an interruption leaves. They do nothing in
    production: ``stage_replay`` defaults them to no-ops and the CLI never exposes them.
    """

    after_config: Callable[[], None] = _no_op
    before_manifest: Callable[[], None] = _no_op


def _artifact_path(directory: Path, name) -> Path:
    """The path of one declared artifact, which must be a relative path inside ``directory``.

    A manifest is data, and data that could name ``../elsewhere`` would let a loader hash
    and read a file the stage never published.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f'artifact name {name!r} must be a relative path inside {directory}')
    parts = name.split('/')
    if any(part in ('', '.', '..') or '\\' in part for part in parts):
        raise ValueError(f'artifact name {name!r} must be a relative path inside {directory}')
    return directory.joinpath(*parts)


def _verify_artifacts(directory: Path, artifacts: Mapping) -> None:
    """Every published file must still hash to what its manifest declared."""
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f'{directory} declares no artifacts to verify')
    for name, declared in artifacts.items():
        path = _artifact_path(directory, name)
        if not path.is_file():
            raise ValueError(f'the manifest in {directory} declares the artifact {name!r}, '
                             f'which is missing')
        digest = sha256_file(path)
        if digest != declared:
            raise ValueError(f'{name} hashes to {digest}, the manifest in {directory} declares '
                             f'{declared!r}')


def _pieces(manifest: Mapping) -> int:
    """Patches times rounds, the normalization of section 9, from a record's own parameters."""
    parameters = manifest['parameters']
    _require_fields(parameters, ('patches', 'rounds'), f'{RECORD_MANIFEST} parameters')
    return _whole(parameters['patches'], 'patches', minimum=1) * \
        _whole(parameters['rounds'], 'rounds', minimum=1)


def _require_compatible(loaded: LoadedRecord, artifact: Mapping, configs, *, path) -> None:
    """Refuse a replay whose record and calibration do not belong together.

    A calibration fitted on these shots would make the evaluation circular, so an equal
    parent sample is refused and so is an equal sampling family, whose two calls may
    overlap even when neither payload matches. A calibration fitted under another model
    or another decoder describes different scores entirely. A configuration naming an
    estimator the artifact does not carry has no knots to replay with.
    """
    stored = artifact['record']['identities']
    for name in ('parent_sample', 'sampling_family'):
        if loaded.identities[name] == stored[name]:
            raise ValueError(f'{path} was fitted on a record with this record\'s {name} identity '
                             f'{stored[name]}, so the evaluation would not be held out')
    for name in ('model', 'decoder'):
        if loaded.identities[name] != stored[name]:
            raise ValueError(f'{path} was fitted on a record with {name} identity {stored[name]}, '
                             f'{loaded.directory} has {loaded.identities[name]}')
    available = set(artifact['estimators'])
    for config in configs:
        for estimator in (config.initial, config.refined):
            if estimator.name not in available:
                raise ValueError(f'configuration {config.name} names the estimator '
                                 f'{estimator.name!r}, which {path} does not carry; it holds '
                                 f'{sorted(available)}')


def _save_result(directory: Path, result: ReplayResult, *, record, config: ReplayConfig,
                 pieces: int) -> None:
    """Write one configuration's per-shot arrays and the metrics computed from them."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(result, name) for name in REPLAY_RESULT_ARRAYS}
    arrays.update({name: getattr(result.work, name) for name in REPLAY_WORK_FIELDS})
    _save_arrays(directory / REPLAY_ARRAYS_FILE, arrays, schema=REPLAY_ARRAY_SCHEMA)
    write_json_atomic(directory / REPLAY_RESULTS_FILE, {
        **summarize_result(record, result, pieces=pieces),
        'initial_estimator': config.initial.name,
        'refined_estimator': config.refined.name,
        'refined_score': config.refined.score,
        'policy': config.policy.name,
        'outer': config.outer,
    })


def _load_result(directory: Path) -> tuple[ReplayResult, Mapping]:
    """Rebuild one configuration's result from its container, re-running every check."""
    results = read_json(directory / REPLAY_RESULTS_FILE)
    _require_fields(results, RESULT_FIELDS, str(directory / REPLAY_RESULTS_FILE))
    arrays = _load_arrays(directory / REPLAY_ARRAYS_FILE, schema=REPLAY_ARRAY_SCHEMA)
    expected = REPLAY_RESULT_ARRAYS + REPLAY_WORK_FIELDS
    unexpected = sorted(set(arrays) - set(expected))
    missing = [name for name in expected if name not in arrays]
    if missing or unexpected:
        raise ValueError(f'{directory / REPLAY_ARRAYS_FILE} is missing {missing} and holds '
                         f'unknown arrays {unexpected}')
    result = ReplayResult(config=results['config'], reference=results['reference'],
                          work=WorkCounts(**{name: arrays[name] for name in REPLAY_WORK_FIELDS}),
                          **{name: arrays[name] for name in REPLAY_RESULT_ARRAYS})
    return result, results


def _verified_replay_manifest(directory: Path) -> Mapping:
    """Read a published replay manifest and check everything it says about itself.

    A directory without the manifest holds an interrupted publication, not a result. The
    recorded identity must follow from the manifest's own record and calibrator hashes,
    configurations, conventions, sources, and versions, and every artifact it declares
    must still hash to its published value.
    """
    path = directory / REPLAY_MANIFEST
    if not path.is_file():
        raise ValueError(f'{path} is missing, so {directory} holds no completed replay')
    manifest = read_json(path)
    _require_fields(manifest, REPLAY_MANIFEST_FIELDS, str(path))
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'{path} holds schema {manifest["schema_version"]!r}, '
                         f'expected {SCHEMA_VERSION!r}')
    if manifest['stage'] != REPLAY_STAGE:
        raise ValueError(f'{path} was written by the {manifest["stage"]!r} stage, '
                         f'expected {REPLAY_STAGE!r}')
    if manifest['status'] != COMPLETE_STATUS:
        raise ValueError(f'{path} declares status {manifest["status"]!r}, '
                         f'expected {COMPLETE_STATUS!r}')
    _require_fields(manifest['record'], RECORD_BLOCK_FIELDS, f'{path} record')
    _require_fields(manifest['record']['identities'], IDENTITY_NAMES, f'{path} record identities')
    _require_fields(manifest['record']['rows'], ROW_SUMMARY_FIELDS, f'{path} record rows')
    _require_fields(manifest['calibrators'], ('path', 'sha256', 'identity'), f'{path} calibrators')
    expected = replay_identity(
        record_sha256=manifest['record']['record_sha256'],
        calibrators_sha256=manifest['calibrators']['sha256'],
        configurations=list(manifest['configurations']), tie_rule=manifest['tie_rule'],
        work_convention=manifest['work_convention'], sources=manifest['source_sha256'],
        versions=manifest['versions'])
    if expected != manifest['identity']:
        raise ValueError(f'the replay identity {manifest["identity"]!r} in {path} does not follow '
                         f'from its own record, calibrators, configurations, conventions, '
                         f'sources, and versions')
    _verify_artifacts(directory, manifest['artifacts'])
    return manifest


def stage_replay(record_dir, calibrators_path, out_dir, configs: Iterable, *,
                 hooks: _ReplayHooks | None = None) -> Mapping[str, ReplayResult]:
    """Replay the configurations over a completed evaluation record, and publish them.

    Everything is verified before a shot is replayed: the record through
    ``load_record``, its role, the calibrator artifact through ``load_calibrators``, the
    held-out and compatibility rules of section 6, and that every configuration's
    estimators are carried by the artifact. Each configuration writes its arrays and
    results under its own directory, and ``replay_manifest.json`` is published last, over
    them.

    Reusing an output directory that already holds a completion manifest requires the
    recomputed identity and every artifact hash to match, in which case the stored
    results are returned without replaying anything; otherwise it raises and asks for a
    new directory. A directory with partial artifacts and no manifest is replayed again,
    because replay is a deterministic function of the record, the calibrators, and the
    configuration, and a partial output is never a result.

    ``hooks`` is the test seam documented on ``_ReplayHooks`` and does nothing by default.
    """
    if hooks is None:
        hooks = _ReplayHooks()
    elif not isinstance(hooks, _ReplayHooks):
        raise TypeError(f'hooks must be a _ReplayHooks, got {type(hooks).__name__}')
    calibrators_path, out_dir = Path(calibrators_path), Path(out_dir)
    loaded = load_record(record_dir)                    # verifies every record artifact hash
    _require_role(loaded, EVALUATION_ROLE)
    calibrators, artifact = load_calibrators(calibrators_path)
    calibrators_sha256 = sha256_file(calibrators_path)
    configs = _configs(configs)
    _require_compatible(loaded, artifact, configs, path=calibrators_path)

    record = _record_block(loaded)
    sources, versions = source_hashes(REPLAY_SOURCES), package_versions(REPLAY_PACKAGES)
    identity = replay_identity(
        record_sha256=record['record_sha256'], calibrators_sha256=calibrators_sha256,
        configurations=[config.name for config in configs], tie_rule=TIE_RULE,
        work_convention=WORK_CONVENTION, sources=sources, versions=versions)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / REPLAY_MANIFEST).is_file():
        return _reuse_replay(out_dir, identity=identity, configs=configs)

    pieces = _pieces(loaded.manifest)
    results: dict[str, ReplayResult] = {}
    artifacts: dict[str, str] = {}
    for config in configs:
        name = config_directory_name(config)
        result = replay(loaded.record, calibrators, config)
        _save_result(out_dir / name, result, record=loaded.record, config=config, pieces=pieces)
        results[config.name] = result
        for published in (REPLAY_ARRAYS_FILE, REPLAY_RESULTS_FILE):
            artifacts[f'{name}/{published}'] = sha256_file(out_dir / name / published)
        hooks.after_config()
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'stage': REPLAY_STAGE,
        'status': COMPLETE_STATUS,
        'identity': identity,
        'record': record,
        'calibrators': {'path': str(calibrators_path.resolve()), 'sha256': calibrators_sha256,
                        'identity': artifact['identity']},
        'configurations': [config.name for config in configs],
        'artifacts': artifacts,
        'tie_rule': TIE_RULE,
        'work_convention': WORK_CONVENTION,
        'versions': versions,
        'source_sha256': sources,
        'code_commit': git_commit(),
        'created_utc': utc_now(),
    }
    hooks.before_manifest()
    write_json_atomic(out_dir / REPLAY_MANIFEST, manifest)
    return MappingProxyType(results)


def _reuse_replay(out_dir: Path, *, identity: str, configs) -> Mapping[str, ReplayResult]:
    """Return what a completed replay directory already holds, re-verified rather than trusted."""
    manifest = _verified_replay_manifest(out_dir)
    if manifest['identity'] != identity:
        raise ValueError(f'{out_dir} already holds a completed replay of other inputs or '
                         f'configurations; use a new output directory rather than mixing two '
                         f'replays in one')
    results = {}
    for config in configs:
        result, _ = _load_result(out_dir / config_directory_name(config))
        results[config.name] = result
    return MappingProxyType(results)


# --- summarize ---------------------------------------------------------------


def _configuration_directories(manifest: Mapping) -> list[str]:
    """The configuration directories a replay manifest declares artifacts in."""
    names = sorted({name.split('/')[0] for name in manifest['artifacts']})
    if not names:
        raise ValueError('a replay manifest declares no configuration directories')
    return names


def _require_same_record(loaded: LoadedRecord, block: Mapping, *, where) -> None:
    """The record a replay consumed must be the record that is there now.

    Rows are compared first because replacing a record with one over other rows is the
    difference a reader is most likely to create by accident, and the hashes below would
    only say that something changed.
    """
    for name in ROW_SUMMARY_FIELDS:
        if loaded.manifest['rows'][name] != block['rows'][name]:
            raise ValueError(f'{where} was replayed over rows with {name} {block["rows"][name]!r}, '
                             f'{loaded.directory} now holds {loaded.manifest["rows"][name]!r}')
    for name in IDENTITY_NAMES:
        if loaded.identities[name] != block['identities'][name]:
            raise ValueError(f'{where} was replayed over a record with {name} identity '
                             f'{block["identities"][name]}, {loaded.directory} now has '
                             f'{loaded.identities[name]}')
    if loaded.manifest['role'] != block['role']:
        raise ValueError(f'{where} was replayed over a {block["role"]!r} record, '
                         f'{loaded.directory} now holds a {loaded.manifest["role"]!r} one')
    for name, stored, declared in (
            (RECORD_FILE, loaded.manifest['artifacts'][RECORD_FILE], block['record_sha256']),
            (RECORD_MANIFEST, sha256_file(loaded.directory / RECORD_MANIFEST),
             block['manifest_sha256'])):
        if stored != declared:
            raise ValueError(f'{where} was replayed over a record whose {name} hashes to '
                             f'{declared}, {loaded.directory} now hashes to {stored}')


def _group_json(loaded: LoadedRecord, cells: Mapping, *, pieces: int, replicates: int,
                seed: int) -> dict:
    """One estimator pair's cells, their paired comparisons, and each side's interval.

    The comparison is defined against the pair's ``initial_only`` mixed cell, the
    baseline of section 8. Every comparison resamples the same shots under the same seed,
    so the baseline's own interval is the same in each of them.
    """
    names = sorted(cells)
    baselines = [name for name, (_, results) in cells.items()
                 if results['policy'] == INITIAL_ONLY and results['outer'] == DEFAULT_OUTER]
    if len(baselines) > 1:
        raise ValueError(f'an estimator pair has more than one initial_only mixed cell: {baselines}')
    baseline = baselines[0] if baselines else None
    summaries = {name: summarize_result(loaded.record, cells[name][0], pieces=pieces)
                 for name in names}
    comparisons, intervals = {}, {name: {} for name in names}
    if baseline is not None:
        for name in names:
            if name == baseline:
                continue
            comparison = compare_endpoints(loaded.record, cells[baseline][0], cells[name][0],
                                           pieces=pieces, replicates=replicates, seed=seed)
            comparisons[name] = comparison
            for metric in PAIRED_METRICS:
                paired = comparison[metric]
                intervals[baseline][metric] = [paired['low_a'], paired['high_a']]
                intervals[name][metric] = [paired['low_b'], paired['high_b']]
    first = cells[names[0]][1]
    return {
        'initial_estimator': first['initial_estimator'],
        'refined_score': first['refined_score'],
        'baseline': baseline,
        'order': ([baseline] if baseline else []) + [n for n in names if n != baseline],
        'policies': {name: {'policy': cells[name][1]['policy'], 'outer': cells[name][1]['outer']}
                     for name in names},
        # Within one estimator pair a cell differs only in its policy and outer rule, so the
        # tables name it by those two and the first table maps them to the full name.
        'labels': {name: f'{cells[name][1]["policy"]}{ESTIMATOR_SEPARATOR}{cells[name][1]["outer"]}'
                   for name in names},
        'cells': summaries,
        'comparisons': comparisons,
        'intervals': intervals,
    }


def _summarize_replay(directory: Path, *, replicates: int, seed: int) -> dict:
    """Verify one replay directory end to end and compute every number the report prints."""
    manifest = _verified_replay_manifest(directory)
    loaded = load_record(manifest['record']['directory'])
    _require_same_record(loaded, manifest['record'], where=directory / REPLAY_MANIFEST)
    pieces = _pieces(loaded.manifest)
    cells: dict[str, tuple[ReplayResult, Mapping]] = {}
    for name in _configuration_directories(manifest):
        result, results = _load_result(directory / name)
        if results['pieces'] != pieces:
            raise ValueError(f'{directory / name} normalizes with pieces {results["pieces"]!r}, '
                             f'the record implies {pieces}')
        cells[results['config']] = (result, results)
    if sorted(cells) != sorted(manifest['configurations']):
        raise ValueError(f'{directory} holds the configurations {sorted(cells)}, its manifest '
                         f'declares {sorted(manifest["configurations"])}')
    groups: dict[tuple[str, str], dict] = {}
    for name, cell in cells.items():
        key = (cell[1]['initial_estimator'], cell[1]['refined_score'])
        groups.setdefault(key, {})[name] = cell
    return {
        'replay_directory': str(directory.resolve()),
        'replay_identity': manifest['identity'],
        'replay_manifest_sha256': sha256_file(directory / REPLAY_MANIFEST),
        'artifacts': dict(manifest['artifacts']),
        'calibrators': dict(manifest['calibrators']),
        'record': {**{name: manifest['record'][name] for name in RECORD_BLOCK_FIELDS},
                   'shots': loaded.record.shots, 'patches': loaded.record.num_patches},
        'pieces': pieces,
        'groups': [_group_json(loaded, members, pieces=pieces, replicates=replicates, seed=seed)
                   for _, members in sorted(groups.items())],
    }


# --- the report ---------------------------------------------------------------


def _number(value) -> str:
    """One number, or ``unavailable`` when the statistic is not defined."""
    if value is None or not np.isfinite(value):
        return UNAVAILABLE
    return f'{float(value):.{REPORT_DIGITS}g}'


def _rate(rate: Mapping) -> str:
    """A rate with the denominator it was taken over, so an empty population is visible."""
    return f'{_number(rate["value"])} [{rate["count"]} / {rate["total"]}]'


def _interval(bounds) -> str:
    """A 95% percentile interval, or ``unavailable`` when the bootstrap could not form one."""
    if bounds is None or bounds[0] is None or bounds[1] is None \
            or not np.isfinite(bounds[0]) or not np.isfinite(bounds[1]):
        return UNAVAILABLE
    return f'({_number(bounds[0])}, {_number(bounds[1])})'


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """One markdown table, with a trailing blank line."""
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    lines.append('')
    return lines


def _endpoint_rows(group: Mapping) -> list[list[str]]:
    """Block failure, normalized LER, misattribution, ties, and above-half, cell by cell."""
    order, cells, intervals = group['order'], group['cells'], group['intervals']
    rows = [['Block failure'] + [f'{_rate(cells[name]["block_failure"])} '
                                 f'{_interval(intervals[name].get("block_failure"))}'
                                 for name in order],
            ['Normalized LER (per patch per round)'] + [_number(cells[name]['normalized_ler'])
                                                        for name in order],
            ['Misattribution (pooled)'] + [
                f'{_rate(cells[name]["misattribution"]["pooled"])} '
                f'{_interval(intervals[name].get("misattribution_pooled"))}' for name in order]]
    for sector in SECTOR_NAMES:
        rows.append([f'Misattribution ({sector})']
                    + [_rate(cells[name]['misattribution'][sector]) for name in order])
    rows.append(['Ties (pooled)'] + [_rate(cells[name]['ties']['pooled']) for name in order])
    rows.append(['Probabilities above one half']
                + [_rate(cells[name]['above_half']) for name in order])
    return rows


def _paired_rows(group: Mapping, shots: int) -> list[list[str]]:
    """Every comparison against the pair's baseline, for both required metrics."""
    rows = []
    for name in group['order']:
        comparison = group['comparisons'].get(name)
        if comparison is None:
            continue
        eligible = {'misattribution_pooled': group['cells'][name]['misattribution']['pooled']['total'],
                    'block_failure': shots}
        for metric in PAIRED_METRICS:
            paired = comparison[metric]
            rows.append([group['labels'][name], metric, _number(paired['difference']),
                         _interval([paired['low'], paired['high']]), str(eligible[metric]),
                         str(paired['zero_denominator_replicates'])])
    return rows


def _render_group(group: Mapping, shots: int) -> list[str]:
    """One estimator pair's endpoint, paired-difference, strata, and work tables."""
    order, cells, labels = group['order'], group['cells'], group['labels']
    columns = [labels[name] for name in order]
    lines = [f'### {group["initial_estimator"]} {CONFIG_SEPARATOR} {group["refined_score"]}', '']
    lines += _table(['Cell', 'Configuration'],
                    [[labels[name], f'`{name}`'] for name in order])
    lines += _table(['Quantity'] + columns, _endpoint_rows(group))
    if group['baseline'] is None:
        lines += [f'No `{INITIAL_ONLY}` `{DEFAULT_OUTER}` cell was replayed for this pair, so no '
                  f'paired difference is available.', '']
    else:
        lines += [f'Paired differences against `{labels[group["baseline"]]}`, resampling whole '
                  f'shots:', '']
        lines += _table(['Cell', 'Metric', 'Difference', '95% interval', 'Eligible',
                         'Zero-denominator replicates'], _paired_rows(group, shots))
    lines += ['Final sector failures by residual reference-failure stratum:', '']
    lines += _table(['Stratum', 'Eligible sectors'] + columns,
                    [[stratum, str(cells[order[0]]['strata'][stratum]['pooled']['total'])]
                     + [_rate(cells[name]['strata'][stratum]['pooled']) for name in order]
                     for stratum in STRATA])
    work_header = ['Counter']
    for label in columns:
        work_header += [f'{label} total', f'{label} per shot']
    work_rows = []
    for counter in REPLAY_WORK_FIELDS:
        row = [counter]
        for name in order:
            work = cells[name]['work']
            row += [str(work['totals'][counter]), _number(work['per_shot_mean'][counter])]
        work_rows.append(row)
    lines += ['Replay work. These are the calls the specified replay procedure implies, not the '
              'calls that ran during collection and not elapsed time:', '']
    lines += _table(work_header, work_rows)
    return lines


def _render(sections: Sequence[Mapping], *, replicates: int, seed: int) -> str:
    """The whole report: one section per replay directory, one subsection per estimator pair."""
    lines = ['# Hierarchical L1/L2 endpoint summary', '',
             f'Paired bootstrap: {replicates} replicates resampling whole shots, seed {seed}. '
             f'Every rate is printed with its denominator, and a statistic that no eligible '
             f'case supports is printed as `{UNAVAILABLE}` rather than as a zero.', '']
    for section in sections:
        record = section['record']
        lines += [f'## Replay {Path(section["replay_directory"]).name}', '']
        lines += _table(['Input', 'Value'], [
            ['Replay directory', f'`{section["replay_directory"]}`'],
            ['Replay identity', f'`{section["replay_identity"]}`'],
            ['Record directory', f'`{record["directory"]}`'],
            ['Record role', record['role']],
            ['Shots', str(record['shots'])],
            ['Rows', f'{record["rows"]["count"]} in [{record["rows"]["start"]}, '
                     f'{record["rows"]["stop"]}), sha256 `{record["rows"]["sha256"]}`'],
            ['Parent sample', f'`{record["identities"]["parent_sample"]}`'],
            ['Model', f'`{record["identities"]["model"]}`'],
            ['Decoder', f'`{record["identities"]["decoder"]}`'],
            ['Calibrators', f'`{section["calibrators"]["path"]}`'],
            ['Calibration identity', f'`{section["calibrators"]["identity"]}`'],
            ['Pieces (patches x rounds)', str(section['pieces'])],
        ])
        for group in section['groups']:
            lines += _render_group(group, record['shots'])
    return '\n'.join(lines).rstrip('\n') + '\n'


def _summary_paths(out_path) -> tuple[Path, Path, Path]:
    """The report, its machine-readable results, and its completion manifest."""
    text = str(Path(out_path))
    stem = text[:-len(MARKDOWN_SUFFIX)] if text.endswith(MARKDOWN_SUFFIX) else text
    return Path(stem + MARKDOWN_SUFFIX), Path(stem + '.json'), Path(stem + '.manifest.json')


def _write_text_atomic(path: Path, text: str) -> None:
    """Publish a text artifact through a temporary sibling and one atomic replace."""
    with atomic_replacement(path) as temporary:
        temporary.write_bytes(text.encode('utf-8'))


def stage_summarize(replay_dirs: Iterable, out_path, *, replicates: int = DEFAULT_REPLICATES,
                    seed: int = DEFAULT_SEED) -> str:
    """Summarize verified replay outputs into a report, its results, and its manifest.

    Each replay directory must hold a completion manifest whose identity follows from its
    own fields and whose declared artifacts still hash to their published values, and the
    record it names must still be the record it was replayed over: the same rows,
    identities, role, record hash, and manifest hash. Configurations are paired only
    within one record and one estimator pair, against that pair's ``initial_only`` mixed
    cell, so both sides share a reference decoder and therefore one eligible population.

    Writes ``<out>.md``, ``<out>.json``, and then ``<out>.manifest.json`` last, and
    returns the markdown text.
    """
    directories = tuple(Path(directory) for directory in replay_dirs)
    if not directories:
        raise ValueError('a summary needs at least one replay directory')
    replicates = _whole(replicates, 'replicates', minimum=1)
    seed = _whole(seed, 'seed', minimum=0)
    sections = [_summarize_replay(directory, replicates=replicates, seed=seed)
                for directory in directories]
    markdown = _render(sections, replicates=replicates, seed=seed)

    report_path, results_path, manifest_path = _summary_paths(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap = {'replicates': replicates, 'seed': seed}
    _write_text_atomic(report_path, markdown)
    write_json_atomic(results_path, {
        'schema_version': SCHEMA_VERSION,
        'stage': SUMMARIZE_STAGE,
        'bootstrap': bootstrap,
        'records': sections,
        'created_utc': utc_now(),
    })
    # Published last, over the two artifacts it declares, exactly as a record's manifest is.
    write_json_atomic(manifest_path, {
        'schema_version': SCHEMA_VERSION,
        'stage': SUMMARIZE_STAGE,
        'status': COMPLETE_STATUS,
        'bootstrap': bootstrap,
        'inputs': [{'replay_directory': section['replay_directory'],
                    'replay_identity': section['replay_identity'],
                    'replay_manifest_sha256': section['replay_manifest_sha256'],
                    'record_directory': section['record']['directory'],
                    'record_sha256': section['record']['record_sha256'],
                    'record_manifest_sha256': section['record']['manifest_sha256'],
                    'artifacts': section['artifacts']} for section in sections],
        'artifacts': {path.name: sha256_file(path) for path in (report_path, results_path)},
        'versions': package_versions(REPLAY_PACKAGES),
        'source_sha256': source_hashes(REPLAY_SOURCES),
        'code_commit': git_commit(),
        'created_utc': utc_now(),
    })
    return markdown
