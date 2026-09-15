"""Offline replay of a stored L1 record: final predictions and the work they imply.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 6, 7, and 8.

An estimator is a pair (reference decoder, score). Replay reads the stored
record, maps the initial estimator's scores to q0 and the refined estimator's
to q1 with the calibrators fitted on the calibration record, asks the policy
for the request mask M, and runs the exact L2 of section 7 per sector on
``q = where(M, q1, q0)`` with the frame-adjusted syndrome
``sigma = y XOR parity(r)``. The final prediction is ``f = r XOR x``, whose
sector parity is the yoke bit by construction; replay checks that on every
row. Nothing here re-runs a decoder: every score already sits in the record.

``WorkCounts`` records, per shot, the matching and search calls the specified
replay procedure would need: the fixed initial work of the estimator pair and
the increment per distinct refined patch, from the table of section 8. Both
sector requests on one patch share one refinement, so ``sum(M)`` (requested
patch-sectors) and ``sum(U)`` (distinct patches) are counted separately. These
counts describe that procedure only. They are not the calls that actually ran
during collection -- those are ``_l1.CollectionWork``, which additionally
computes validation predictions and every score for every patch -- and they
are not elapsed time, so no speedup follows from them.

``_replay_test.py`` checks the estimator pairs, hand-computed signed score
indexing, per-sector calibrator pooling, hand-worked endpoint final bits,
mixed versus candidate-restricted L2, deterministic ties, unchanged inputs,
repeatability, that unrequested refined scores are ignored, the work table at
no and full refinement, shared work across the two sectors of one patch, and
that replay work is a separate object from collection work.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._outer_decoder import exact_outer_map_batch, frame_adjusted_syndrome
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._policies import DeterministicPolicy
from yoked.hierarchical._record import REFERENCE_NAMES, L1Record, by_sector, to_columns

SCORE_NAMES = ('cluster_gap', 'gap_plain', 'gap_correlated')
"""The three soft outputs section 5 stores: the UF cluster gap and the signed forced-weight
difference under the plain and the correlated weights."""

SCORE_DIRECTION = 'decreasing'
"""Every score decreases in the residual-error probability by definition (section 6): a
larger cluster gap or signed matching gap means a more confident reference bit."""

ESTIMATOR_SEPARATOR = ':'
"""Separates the reference from the score in an estimator name, as the CLI writes it."""

CONFIG_SEPARATOR = '->'
"""Separates the initial estimator from the refined score in a configuration name."""

OUTER_RULES = ('mixed', 'restricted')
"""The two L2 rules of section 7: 'mixed' compares refined and initial probabilities,
'restricted' fixes every unrequested patch to x_i = 0."""

FORCED_CALLS_PER_DECODE = 4
"""One forced matching per (c_X, c_Z) class; the four calls yield both sector gaps."""

SEARCHES_PER_UF_DECODE = 2
"""The cluster gap runs one parity-augmented search per sector."""

REPLAY_BATCH_SHOTS = 4096
"""Shots per L2 enumeration call. The decoder builds a (batch, 2**P, P) candidate
intermediate, which is about 1.6 MB at this batch size with the experiment's six
patches, so a 100,000-shot record never materializes a large temporary."""


# --- estimators ---------------------------------------------------------------


@dataclass(frozen=True)
class Estimator:
    """A reference decoder and the score whose calibration estimates its residual errors.

    Fields: ``reference`` in ``REFERENCE_NAMES``; ``score`` in ``SCORE_NAMES``.
    The cluster gap is a UF quantity and has no meaning for the MWPM reference.
    """

    reference: str
    score: str

    def __post_init__(self) -> None:
        if self.reference not in REFERENCE_NAMES:
            raise ValueError(f'reference must be one of {REFERENCE_NAMES}, got {self.reference!r}')
        if self.score not in SCORE_NAMES:
            raise ValueError(f'score must be one of {SCORE_NAMES}, got {self.score!r}')
        if self.score == 'cluster_gap' and self.reference != 'uf':
            raise ValueError(f'the cluster gap is only defined for the uf reference, got {self.reference!r}')

    @property
    def name(self) -> str:
        """The estimator's name in calibrator files, configurations, and reports."""
        return f'{self.reference}{ESTIMATOR_SEPARATOR}{self.score}'

    @property
    def direction(self) -> str:
        """The monotone sense its calibrator is fitted in; always ``SCORE_DIRECTION``."""
        return SCORE_DIRECTION

    @classmethod
    def parse(cls, text: str) -> Estimator:
        """Read an estimator from its ``reference:score`` name."""
        parts = text.split(ESTIMATOR_SEPARATOR) if isinstance(text, str) else []
        if len(parts) != 2:
            raise ValueError(f'Expected an estimator named reference{ESTIMATOR_SEPARATOR}score, got {text!r}')
        try:
            return cls(parts[0], parts[1])
        except ValueError as exc:
            raise ValueError(f'{exc} in estimator {text!r}') from exc


def _checked_record(record) -> L1Record:
    if not isinstance(record, L1Record):
        raise TypeError(f'record must be an L1Record, got {type(record).__name__}')
    return record


def estimator_scores(record: L1Record, estimator: Estimator) -> np.ndarray:
    """The estimator's (shots, 2P) scores over a record, in column layout.

    The cluster gap is stored directly. A signed gap is the forced-weight
    difference of section 5.2 read against the estimator's own reference bits,
    so the same weights give different scores for the two references.
    """
    record = _checked_record(record)
    if not isinstance(estimator, Estimator):
        raise TypeError(f'estimator must be an Estimator, got {type(estimator).__name__}')
    if estimator.score == 'cluster_gap':
        return record.cluster_gap
    forced = record.forced_plain if estimator.score == 'gap_plain' else record.forced_correlated
    reference = record.reference(estimator.reference)
    patch_major = reference.reshape(record.shots, record.num_patches, NUM_SECTORS)
    gaps = signed_gaps(forced, patch_major)                     # (shots, P, 2), delta_X then delta_Z
    return readonly_array(gaps.reshape(record.shots, NUM_SECTORS * record.num_patches), dtype=np.float64)


def residual_errors(record: L1Record, reference: str) -> np.ndarray:
    """The (shots, 2P) residual errors ``e = actual XOR r`` of a reference decoder."""
    record = _checked_record(record)
    return readonly_array(record.actual ^ record.reference(reference), dtype=bool)


# --- calibrators over a record -------------------------------------------------

Calibrators = Mapping[str, tuple[IsotonicCalibrator, IsotonicCalibrator]]
"""Fitted calibrators by estimator name, X sector first then Z, as section 6 fits them."""


def fit_calibrators(record: L1Record, estimators: Iterable[Estimator]) -> Calibrators:
    """Fit one calibrator per estimator and sector on a calibration record.

    The P patches of a sector are pooled into one sample, so each fit sees
    ``shots * P`` score/outcome pairs. The outcome is the residual error of
    the estimator's own reference. The returned mapping is immutable.
    """
    record = _checked_record(record)
    estimators = tuple(estimators)
    if not estimators:
        raise ValueError('Fitting needs at least one estimator')
    fitted: dict[str, tuple[IsotonicCalibrator, IsotonicCalibrator]] = {}
    for estimator in estimators:
        if estimator.name in fitted:
            raise ValueError(f'Estimator {estimator.name!r} was listed twice')
        scores = by_sector(estimator_scores(record, estimator))                      # (shots, 2, P)
        outcomes = by_sector(residual_errors(record, estimator.reference))
        fitted[estimator.name] = tuple(
            IsotonicCalibrator.fit(scores[:, sector, :].ravel(), outcomes[:, sector, :].ravel(),
                                   direction=estimator.direction)
            for sector in range(NUM_SECTORS))
    return MappingProxyType(fitted)


def calibrated_probabilities(record: L1Record, estimator: Estimator, calibrators: Calibrators) -> np.ndarray:
    """The estimator's (shots, 2, P) residual-error probabilities over a record."""
    record = _checked_record(record)
    pair = calibrators.get(estimator.name) if isinstance(calibrators, Mapping) else None
    if pair is None:
        raise ValueError(f'No calibrator for estimator {estimator.name!r}')
    if len(pair) != NUM_SECTORS or not all(isinstance(item, IsotonicCalibrator) for item in pair):
        raise ValueError(f'Estimator {estimator.name!r} needs one calibrator per sector')
    scores = by_sector(estimator_scores(record, estimator))
    probabilities = np.empty(scores.shape, dtype=np.float64)
    for sector, calibrator in enumerate(pair):
        probabilities[:, sector, :] = calibrator.probability(scores[:, sector, :])
    return readonly_array(probabilities, dtype=np.float64)


# --- configurations and their work table ---------------------------------------


@dataclass(frozen=True)
class _PairWork:
    """Per-patch call counts of one estimator pair, from the table of section 8.

    ``initial_*`` is charged once per patch whatever the policy requests;
    ``incremental_*`` is charged once per distinct patch the policy refines.
    Units are calls, except the reweight passes, which are passes.
    """

    initial_uf_calls: int = 0
    initial_dijkstra_searches: int = 0
    initial_unforced_plain_calls: int = 0
    initial_plain_forced_calls: int = 0
    incremental_unforced_plain_calls: int = 0
    incremental_plain_forced_calls: int = 0
    incremental_reweight_passes: int = 0
    incremental_correlated_forced_calls: int = 0


REFINEMENT_PAIRS: Mapping[tuple[str, str, str], _PairWork] = MappingProxyType({
    # UF cluster gap to plain gap: the UF decode and its two searches are already done, so
    # refining a patch is the four forced plain matchings that give both sector gaps.
    ('uf', 'cluster_gap', 'gap_plain'): _PairWork(
        initial_uf_calls=1, initial_dijkstra_searches=SEARCHES_PER_UF_DECODE,
        incremental_plain_forced_calls=FORCED_CALLS_PER_DECODE),
    # UF cluster gap to correlated gap: the correlated weights need a first pass to select
    # edges and a reweighting pass before the four forced matchings.
    ('uf', 'cluster_gap', 'gap_correlated'): _PairWork(
        initial_uf_calls=1, initial_dijkstra_searches=SEARCHES_PER_UF_DECODE,
        incremental_unforced_plain_calls=1, incremental_reweight_passes=1,
        incremental_correlated_forced_calls=FORCED_CALLS_PER_DECODE),
    # MWPM plain gap to correlated gap: the reference itself is the unforced first pass and
    # the initial score already costs four forced matchings, whose edges the reweighting reuses.
    ('mwpm', 'gap_plain', 'gap_correlated'): _PairWork(
        initial_unforced_plain_calls=1, initial_plain_forced_calls=FORCED_CALLS_PER_DECODE,
        incremental_reweight_passes=1,
        incremental_correlated_forced_calls=FORCED_CALLS_PER_DECODE),
})
"""The only initial-to-refined estimator pairs the pilot replays, each with its per-patch
work. One table drives both the configuration check and the work accounting, so a pair
can never be replayed without a stated cost."""


@dataclass(frozen=True)
class ReplayConfig:
    """One replay: an estimator pair, a policy, and the L2 rule.

    Fields: ``initial`` and ``refined`` estimators sharing one reference and
    forming a pair in ``REFINEMENT_PAIRS``; ``policy``, a deterministic policy
    supplying the request mask; ``outer`` in ``OUTER_RULES``.
    """

    initial: Estimator
    refined: Estimator
    policy: DeterministicPolicy
    outer: str = 'mixed'

    def __post_init__(self) -> None:
        for name in ('initial', 'refined'):
            if not isinstance(getattr(self, name), Estimator):
                raise TypeError(f'{name} must be an Estimator, got {type(getattr(self, name)).__name__}')
        if not isinstance(self.policy, DeterministicPolicy):
            raise TypeError('policy must implement select(q0, sigma) and carry a name')
        if self.initial.reference != self.refined.reference:
            raise ValueError(f'both estimators must share one reference, got '
                             f'{self.initial.reference!r} and {self.refined.reference!r}')
        if self.pair not in REFINEMENT_PAIRS:
            raise ValueError(f'{self.initial.name} -> {self.refined.score} is not one of the '
                             f'supported estimator pairs {sorted(REFINEMENT_PAIRS)}')
        if self.outer not in OUTER_RULES:
            raise ValueError(f'outer must be one of {OUTER_RULES}, got {self.outer!r}')

    @property
    def pair(self) -> tuple[str, str, str]:
        """The (reference, initial score, refined score) key into ``REFINEMENT_PAIRS``."""
        return (self.initial.reference, self.initial.score, self.refined.score)

    @property
    def reference(self) -> str:
        """The one reference decoder both estimators are defined against."""
        return self.initial.reference

    @property
    def name(self) -> str:
        """The configuration's name in a directory, a results file, and a report."""
        return (f'{self.initial.name}{CONFIG_SEPARATOR}{self.refined.score}'
                f'{ESTIMATOR_SEPARATOR}{self.policy.name}{ESTIMATOR_SEPARATOR}{self.outer}')


# --- work counts ---------------------------------------------------------------


def _shot_counts(value, name: str, shots: int) -> np.ndarray:
    """An owned read-only (shots,) int64 array of whole, nonnegative counts."""
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be an integer count array, got dtype {array.dtype}')
    if array.shape != (shots,):
        raise ValueError(f'{name} must have shape {(shots,)}, got {array.shape}')
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f'{name} must be finite and nonnegative')
    if array.dtype.kind == 'f' and not (array == np.floor(array)).all():
        raise ValueError(f'{name} must contain whole numbers')
    return readonly_array(array, dtype=np.int64)


@dataclass(frozen=True)
class WorkCounts:
    """Per-shot work the specified replay procedure implies, all read-only (shots,) int64.

    - ``requested_patch_sectors``: ``sum(M)`` (patch-sectors) whose refined score L2 consumed.
    - ``distinct_patches``: ``sum(U)`` (patches) refined, counting a patch once even when
      both of its sectors were requested.
    - ``initial_uf_calls``: UF decodes of the initial estimator (calls).
    - ``initial_dijkstra_searches``: parity-augmented searches behind the cluster gap (calls).
    - ``initial_dijkstra_states``: states settled by those searches (states), read from the
      record; zero when the initial score is not the cluster gap.
    - ``initial_unforced_plain_calls``: unforced first-pass matchings of the initial estimator (calls).
    - ``initial_plain_forced_calls``: forced plain matchings of the initial estimator (calls).
    - ``incremental_unforced_plain_calls``: unforced matchings refinement adds (calls).
    - ``incremental_plain_forced_calls``: forced plain matchings refinement adds (calls).
    - ``incremental_reweight_passes``: correlation-rule reweightings refinement adds (passes).
    - ``incremental_correlated_forced_calls``: forced matchings under reweighted weights (calls).

    This is the cost of the replay procedure, not the collection that produced
    the record (``_l1.CollectionWork``) and not elapsed time.
    """

    requested_patch_sectors: np.ndarray
    distinct_patches: np.ndarray
    initial_uf_calls: np.ndarray
    initial_dijkstra_searches: np.ndarray
    initial_dijkstra_states: np.ndarray
    initial_unforced_plain_calls: np.ndarray
    initial_plain_forced_calls: np.ndarray
    incremental_unforced_plain_calls: np.ndarray
    incremental_plain_forced_calls: np.ndarray
    incremental_reweight_passes: np.ndarray
    incremental_correlated_forced_calls: np.ndarray

    def __post_init__(self) -> None:
        # The first counter fixes the number of shots every other one is checked against.
        first = np.asarray(self.requested_patch_sectors)
        if first.ndim != 1 or len(first) < 1:
            raise ValueError(f'requested_patch_sectors must be a nonempty per-shot array, '
                             f'got shape {first.shape}')
        shots = len(first)
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name,
                               _shot_counts(getattr(self, field.name), field.name, shots))

    @property
    def shots(self) -> int:
        """Number of shots these counts cover."""
        return int(len(self.requested_patch_sectors))

    def totals(self) -> dict[str, int]:
        """The sum of each counter over the shots."""
        return {field.name: int(getattr(self, field.name).sum()) for field in dataclasses.fields(self)}

    def to_json(self) -> dict:
        """Totals and per-shot means, as the report tables of section 9 print them."""
        return {'shots': self.shots, 'totals': self.totals(),
                'per_shot_mean': {field.name: float(getattr(self, field.name).mean())
                                  for field in dataclasses.fields(self)}}


def _replay_work(record: L1Record, config: ReplayConfig, requested: np.ndarray,
                 refined_patches: np.ndarray) -> WorkCounts:
    """The work counts implied by one replay's request mask.

    ``requested`` is M in (shots, 2, P) layout and ``refined_patches`` is U in
    (shots, P). Initial work is charged once per patch; incremental work once
    per distinct refined patch, so the two sector requests of one patch share it.
    """
    per_patch = REFINEMENT_PAIRS[config.pair]
    patches = record.num_patches
    distinct = refined_patches.sum(axis=1).astype(np.int64)
    counts = {
        'requested_patch_sectors': requested.sum(axis=(1, 2)).astype(np.int64),
        'distinct_patches': distinct,
        # The settled states of the searches that produced the stored cluster gaps.
        'initial_dijkstra_states': (record.dijkstra_states.sum(axis=1) if config.initial.score == 'cluster_gap'
                                    else np.zeros(record.shots, dtype=np.int64)),
    }
    for field in dataclasses.fields(per_patch):
        per_call = getattr(per_patch, field.name)
        scale = patches if field.name.startswith('initial_') else distinct
        counts[field.name] = np.full(record.shots, per_call, dtype=np.int64) * scale
    return WorkCounts(**counts)


# --- replay --------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """One configuration's replay over a record, every array owned and read-only.

    Fields: ``config`` the configuration name; ``reference`` the reference
    decoder both its estimators were defined against, in ``REFERENCE_NAMES``,
    which fixes the eligible population every metric is measured over;
    ``final`` (shots, 2P) bool the predicted observable flips; ``requested``
    (shots, 2P) bool the request mask M in column layout; ``refined_patches``
    (shots, P) bool the distinct-patch mask U; ``ties`` (shots, 2) bool whether
    L2's maximum was tied in that sector; ``above_half`` (shots, 2, P) bool
    whether the probability L2 consumed exceeded one half; ``work`` the
    per-shot ``WorkCounts``.
    """

    config: str
    reference: str
    final: np.ndarray
    requested: np.ndarray
    refined_patches: np.ndarray
    ties: np.ndarray
    above_half: np.ndarray
    work: WorkCounts

    def __post_init__(self) -> None:
        if not isinstance(self.config, str) or not self.config:
            raise ValueError('config must be a nonempty configuration name')
        if self.reference not in REFERENCE_NAMES:
            raise ValueError(f'reference must be one of {REFERENCE_NAMES}, got {self.reference!r}')
        if not isinstance(self.work, WorkCounts):
            raise TypeError(f'work must be a WorkCounts, got {type(self.work).__name__}')
        final = np.asarray(self.final)
        if final.ndim != 2 or final.shape[0] < 1 or final.shape[1] % NUM_SECTORS:
            raise ValueError(f'final must have shape (shots >= 1, {NUM_SECTORS} * patches), got {final.shape}')
        shots, columns = final.shape
        patches = columns // NUM_SECTORS
        shapes = {'final': (shots, columns), 'requested': (shots, columns),
                  'refined_patches': (shots, patches), 'ties': (shots, NUM_SECTORS),
                  'above_half': (shots, NUM_SECTORS, patches)}
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f'{name} must have shape {shape}, got {array.shape}')
            if array.dtype.kind not in 'bui' or not np.isin(array, (0, 1)).all():
                raise ValueError(f'{name} must contain only 0 and 1')
            object.__setattr__(self, name, readonly_array(array, dtype=bool))
        if self.work.shots != shots:
            raise ValueError(f'work covers {self.work.shots} shots, the arrays cover {shots}')

    @property
    def shots(self) -> int:
        """Number of replayed shots."""
        return int(self.final.shape[0])

    @property
    def num_patches(self) -> int:
        """Number of patches P."""
        return int(self.refined_patches.shape[1])


def _outer_patterns(q: np.ndarray, sigma: np.ndarray, requested: np.ndarray,
                    restricted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Run the exact L2 of section 7 per sector, in bounded batches of shots."""
    shots = q.shape[0]
    patterns = np.zeros(q.shape, dtype=bool)
    ties = np.zeros((shots, NUM_SECTORS), dtype=bool)
    for sector in range(NUM_SECTORS):
        for start in range(0, shots, REPLAY_BATCH_SHOTS):
            rows = slice(start, min(start + REPLAY_BATCH_SHOTS, shots))
            candidates = requested[rows, sector, :] if restricted else None
            decision = exact_outer_map_batch(q[rows, sector, :], sigma[rows, sector], candidates)
            patterns[rows, sector, :] = decision.patterns
            ties[rows, sector] = decision.tied
    return patterns, ties


def replay(record: L1Record, calibrators: Calibrators, config: ReplayConfig) -> ReplayResult:
    """Replay one configuration over a stored record.

    Neither the record nor the calibrators are modified; the result is a
    deterministic function of the three inputs.
    """
    record = _checked_record(record)
    if not isinstance(config, ReplayConfig):
        raise TypeError(f'config must be a ReplayConfig, got {type(config).__name__}')
    reference = record.reference(config.reference)                            # (shots, 2P)
    q0 = calibrated_probabilities(record, config.initial, calibrators)        # (shots, 2, P)
    # Both calibrators are required even for initial_only, which requests none of q1: a
    # configuration names an estimator pair, and a missing refined calibrator is a caller
    # error rather than something for the policy to hide.
    q1 = calibrated_probabilities(record, config.refined, calibrators)
    sigma = frame_adjusted_syndrome(record.yoke, reference)                   # (shots, 2)

    requested = np.asarray(config.policy.select(q0, sigma), dtype=bool)       # M
    if requested.shape != q0.shape:
        raise ValueError(f'policy {config.policy.name!r} returned a mask of shape {requested.shape}, '
                         f'expected {q0.shape}')
    refined_patches = requested.any(axis=1)                                   # U, (shots, P)
    q = np.where(requested, q1, q0)

    patterns, ties = _outer_patterns(q, sigma, requested, config.outer == 'restricted')
    final = reference ^ to_columns(patterns)
    # Exact L2 returns a pattern of the required parity, so this can only fail if the
    # decoder or the layout conversions are wrong. It is checked on every row anyway.
    parity = (by_sector(final).sum(axis=2) % 2).astype(bool)
    if not np.array_equal(parity, record.yoke):
        rows = np.flatnonzero((parity != record.yoke).any(axis=1))
        raise ValueError(f'final sector parity does not equal the yoke bit on rows {rows[:10].tolist()}')

    return ReplayResult(config=config.name, reference=config.reference, final=final,
                        requested=to_columns(requested),
                        refined_patches=refined_patches, ties=ties, above_half=q > 0.5,
                        work=_replay_work(record, config, requested, refined_patches))
