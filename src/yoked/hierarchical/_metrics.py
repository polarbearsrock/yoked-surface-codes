"""The pilot's accuracy metrics over a replayed record, with paired bootstrap intervals.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 9.

The primary metric is the misattribution rate: among sectors with exactly one
residual reference failure, the fraction whose final prediction is wrong. It is
reported per sector and pooled, beside the overall block failure rate, the
normalized LER per patch per round of the four-decoder comparison, the
zero/one/multiple residual-failure strata with their denominators, the tie and
above-half frequencies of section 6, and the replay work of section 8.

Every rate carries its denominator. A rate with no eligible sector is
undefined and is reported as ``None`` with its zero count, never as a zero
failure rate, and no agreement fraction is required anywhere.

Comparisons resample whole shots with replacement, 10,000 replicates at seed
43, the convention of the existing four-decoder comparison. A replicate's
statistic is ``sum(numerator) / sum(denominator)`` over the resampled shots, so
a shot contributes its eligible-sector count and its misattributed count
together and the dependence between the X and Z sectors of a shot is retained.
Replicates whose denominator is empty are counted and excluded, not replaced
by zero. Both endpoints are resampled under the same draw, so the reported
difference ``all_refined - initial_only`` is paired.

``_metrics_test.py`` checks the rates and strata on hand-built bit patterns,
the undefined cases, the sinter conversion, a hand-computed paired point
estimate, reproducibility from the seed, invariance to replicate blocking,
retained cross-sector dependence, the zero-denominator count, and the
hand-worked endpoint summary and comparison.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import sinter

from yoked.hierarchical._record import L1Record, by_sector
from yoked.hierarchical._replay import ReplayResult

SECTOR_NAMES = ('X', 'Z')
"""Sector 0 is X and sector 1 is Z, the order of the record's column layout."""

POOLED = 'pooled'
"""Key for a rate taken over both sectors at once."""

STRATA = ('zero', 'one', 'multiple')
"""The residual-failure strata of section 9: no reference failure in the sector, exactly
one (the eligible population of the primary metric), or two or more."""

OUTER_LOGICAL_VALUES = 8
"""Logical values per shot. The six-patch block with two yokes is a [[6, 4, 2]] outer
code, so 2 * (6 - 2) = 8 observables are scored; the four-decoder comparison normalizes
with the same ``values``."""

DEFAULT_REPLICATES = 10000
"""Bootstrap replicates, as in the existing four-decoder comparison."""

DEFAULT_SEED = 43
"""The bootstrap seed this experiment quotes; stated explicitly so a report can be redone."""

INTERVAL_PERCENTILES = (2.5, 97.5)
"""The two-sided 95% percentile interval."""

BOOTSTRAP_BLOCK_ENTRIES = 1 << 22
"""Resampled shot indices drawn per block, about 32 MB of int64. Blocking keeps the
100,000-shot by 10,000-replicate case vectorized without allocating the whole
(replicates, shots) index matrix."""


# --- rates ---------------------------------------------------------------------


def _whole(value, name: str) -> int:
    """A nonnegative whole count. A bool is a flag, not a count, and is refused."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if value < 0:
        raise ValueError(f'{name} must be nonnegative, got {value}')
    return int(value)


@dataclass(frozen=True)
class Rate:
    """``count`` events out of ``total`` eligible cases.

    ``value`` is the fraction, or nan when nothing was eligible; the
    denominator travels with it so an undefined rate can be reported as
    unavailable rather than as zero.
    """

    count: int
    total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'count', _whole(self.count, 'count'))
        object.__setattr__(self, 'total', _whole(self.total, 'total'))
        if self.count > self.total:
            raise ValueError(f'count {self.count} exceeds total {self.total}')

    @property
    def value(self) -> float:
        """The fraction, or nan when ``total`` is zero."""
        return float('nan') if self.total == 0 else self.count / self.total

    def to_json(self) -> dict:
        """The rate with its denominator; an undefined value serializes as None."""
        value = self.value
        return {'count': self.count, 'total': self.total,
                'value': None if not np.isfinite(value) else float(value)}


def _rate_of(flags: np.ndarray, eligible: np.ndarray) -> Rate:
    """The rate of true flags among the eligible entries of the same array."""
    return Rate(int((flags & eligible).sum()), int(eligible.sum()))


def _by_sector_rates(flags: np.ndarray, eligible: np.ndarray) -> dict[str, Rate]:
    """Per-sector rates and their pooled total, for (shots, 2) flag and eligibility arrays."""
    rates = {name: _rate_of(flags[:, sector], eligible[:, sector])
             for sector, name in enumerate(SECTOR_NAMES)}
    rates[POOLED] = _rate_of(flags, eligible)
    return rates


# --- failures over predictions --------------------------------------------------


def _bit_pair(first, second, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Two (shots, 2P) bit arrays of the same shape."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        raise ValueError(f'{name} arrays must have the same shape, got {first.shape} and {second.shape}')
    if first.ndim != 2 or first.shape[0] < 1 or first.shape[1] % len(SECTOR_NAMES):
        raise ValueError(f'{name} arrays must have shape (shots >= 1, 2 * patches), got {first.shape}')
    for array in (first, second):
        if array.dtype.kind not in 'buif' or not np.isin(array, (0, 1)).all():
            raise ValueError(f'{name} arrays must contain only 0 and 1')
    return first.astype(bool), second.astype(bool)


def sector_failures(final: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(shots, 2) bool: whether any observable of that sector was predicted wrong."""
    final, actual = _bit_pair(final, actual, 'prediction')
    return by_sector(final ^ actual).any(axis=2)


def block_failures(final: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(shots,) bool: whether any of the shot's observables was predicted wrong."""
    final, actual = _bit_pair(final, actual, 'prediction')
    return (final ^ actual).any(axis=1)


def residual_failure_counts(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(shots, 2) int: how many patches of that sector the reference decoder got wrong."""
    reference, actual = _bit_pair(reference, actual, 'reference')
    return by_sector(reference ^ actual).sum(axis=2).astype(np.int64)


def _checked_sector_inputs(sector_failure, counts) -> tuple[np.ndarray, np.ndarray]:
    sector_failure, counts = np.asarray(sector_failure), np.asarray(counts)
    if sector_failure.shape != counts.shape or sector_failure.ndim != 2 \
            or sector_failure.shape[1] != len(SECTOR_NAMES):
        raise ValueError(f'expected two (shots, {len(SECTOR_NAMES)}) arrays, '
                         f'got {sector_failure.shape} and {counts.shape}')
    if sector_failure.dtype.kind not in 'buif' or not np.isin(sector_failure, (0, 1)).all():
        raise ValueError('sector failures must contain only 0 and 1')
    if counts.dtype.kind not in 'iu' or (counts < 0).any():
        raise ValueError('residual failure counts must be nonnegative integers')
    return sector_failure.astype(bool), counts


def misattribution(sector_failure: np.ndarray, counts: np.ndarray) -> dict[str, Rate]:
    """The primary metric: failures among sectors with exactly one residual reference failure."""
    sector_failure, counts = _checked_sector_inputs(sector_failure, counts)
    return _by_sector_rates(sector_failure, counts == 1)


def failure_by_stratum(sector_failure: np.ndarray, counts: np.ndarray) -> dict[str, dict[str, Rate]]:
    """Final sector failures within each residual-failure stratum, with denominators.

    Every stratum is reported even when no sector falls in it, so an empty
    stratum is visible as an undefined rate rather than absent.
    """
    sector_failure, counts = _checked_sector_inputs(sector_failure, counts)
    eligibility = {'zero': counts == 0, 'one': counts == 1, 'multiple': counts >= 2}
    return {name: _by_sector_rates(sector_failure, eligibility[name]) for name in STRATA}


def tie_frequency(ties: np.ndarray) -> dict[str, Rate]:
    """How often L2's maximum was tied, per sector and pooled over the shots."""
    ties = np.asarray(ties)
    if ties.ndim != 2 or ties.shape[1] != len(SECTOR_NAMES):
        raise ValueError(f'ties must have shape (shots, {len(SECTOR_NAMES)}), got {ties.shape}')
    return _by_sector_rates(ties.astype(bool), np.ones(ties.shape, dtype=bool))


def above_half_frequency(above_half: np.ndarray) -> Rate:
    """How often a calibrated probability L2 consumed exceeded one half, over all patch-sectors."""
    above_half = np.asarray(above_half)
    if above_half.ndim != 3 or above_half.shape[1] != len(SECTOR_NAMES):
        raise ValueError(f'above_half must have shape (shots, {len(SECTOR_NAMES)}, patches), '
                         f'got {above_half.shape}')
    return Rate(int(above_half.astype(bool).sum()), int(above_half.size))


def normalized_ler(rate, *, pieces, values: int = OUTER_LOGICAL_VALUES) -> float | None:
    """The block failure rate as an error rate per patch per round, or None if undefined.

    ``pieces`` is patches times rounds and ``values`` the number of scored
    observables, so the number sits beside the four-decoder comparison's table.
    """
    if not isinstance(pieces, (int, np.integer)) or isinstance(pieces, (bool, np.bool_)) or pieces <= 0:
        raise ValueError(f'pieces must be a positive integer, got {pieces!r}')
    if not isinstance(values, (int, np.integer)) or isinstance(values, (bool, np.bool_)) or values <= 0:
        raise ValueError(f'values must be a positive integer, got {values!r}')
    if rate is None:
        return None
    rate = float(rate)
    if not np.isfinite(rate):
        return None
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f'a block failure rate must lie in [0, 1], got {rate}')
    return float(sinter.shot_error_rate_to_piece_error_rate(rate, pieces=int(pieces), values=int(values)))


# --- paired bootstrap -------------------------------------------------------------


SETTING_FIELDS = ('replicates', 'seed', 'zero_denominator_replicates')
"""The whole-number fields of a ``PairedDifference``; every other field is a rate or a
bound and may be nan when the comparison is undefined."""


@dataclass(frozen=True)
class PairedDifference:
    """One paired comparison of two configurations on the same shots.

    ``estimate_a`` and ``estimate_b`` are the observed rates of the two sides
    and ``difference`` is ``estimate_b - estimate_a``. ``low_a``/``high_a`` and
    ``low_b``/``high_b`` are the 95% percentile intervals of each side and
    ``low``/``high`` that of the difference, all from the same resampled
    shots, so the sides stay paired. ``zero_denominator_replicates`` counts the
    replicates whose eligible population was empty; they are excluded from the
    percentiles and reported rather than replaced. Undefined values are nan
    here and None in ``to_json``.
    """

    estimate_a: float
    estimate_b: float
    low_a: float
    high_a: float
    low_b: float
    high_b: float
    difference: float
    low: float
    high: float
    replicates: int
    seed: int
    zero_denominator_replicates: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in SETTING_FIELDS:
                object.__setattr__(self, field.name, _whole(value, field.name))
            else:
                object.__setattr__(self, field.name, float(value))
        if self.replicates < 1:
            raise ValueError('replicates must be positive')
        if self.zero_denominator_replicates > self.replicates:
            raise ValueError('more empty replicates than replicates')

    def to_json(self) -> dict:
        """Every field, with undefined estimates and bounds serialized as None."""
        payload = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            payload[field.name] = (None if isinstance(value, float) and not np.isfinite(value)
                                   else value)
        return payload



def _ratio(numerator: float, denominator: float) -> float:
    return float('nan') if denominator == 0 else float(numerator) / float(denominator)


def _per_shot(value, name: str, shots: int | None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1:
        raise ValueError(f'{name} must be a nonempty per-shot array, got shape {array.shape}')
    if shots is not None and len(array) != shots:
        raise ValueError(f'{name} has length {len(array)}, expected {shots}')
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f'{name} must be finite and nonnegative')
    return array


def _block_size(shots: int, replicates: int) -> int:
    """Replicates resampled in one vectorized draw."""
    return max(1, min(replicates, BOOTSTRAP_BLOCK_ENTRIES // shots))


def _resampled_sums(columns: np.ndarray, *, replicates: int, seed: int) -> np.ndarray:
    """(replicates, 4) sums of the four per-shot columns over resampled whole shots.

    One replicate draws ``shots`` shot indices with replacement; ``np.bincount``
    turns them into per-shot multiplicities, and one matrix product sums every
    column under those weights. Blocks of replicates are drawn together, which
    changes nothing but the memory the draw needs.
    """
    shots = columns.shape[0]
    generator = np.random.default_rng(seed)
    sums = np.empty((replicates, columns.shape[1]), dtype=np.float64)
    block = _block_size(shots, replicates)
    drawn = 0
    while drawn < replicates:
        size = min(block, replicates - drawn)
        draws = generator.integers(0, shots, size=(size, shots))
        # One bincount over the whole block: row r occupies bins [r * shots, (r + 1) * shots).
        offsets = shots * np.arange(size, dtype=np.int64)[:, None]
        weights = np.bincount((draws + offsets).ravel(), minlength=size * shots).reshape(size, shots)
        sums[drawn:drawn + size] = weights.astype(np.float64) @ columns
        drawn += size
    return sums


def paired_bootstrap(numerator_a, numerator_b, denominator_a, denominator_b, *,
                     replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED) -> PairedDifference:
    """Compare two configurations by resampling whole shots with replacement.

    The four arguments are per-shot numerators and denominators, one pair per
    side. A replicate's statistic is the ratio of the resampled sums, so a
    shot's sectors move together and its whole contribution is resampled at
    once. The reported difference is side b minus side a.
    """
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) \
            or replicates < 1:
        raise ValueError(f'replicates must be a positive integer, got {replicates!r}')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f'seed must be a nonnegative integer, got {seed!r}')
    numerator_a = _per_shot(numerator_a, 'numerator_a', None)
    shots = len(numerator_a)
    numerator_b = _per_shot(numerator_b, 'numerator_b', shots)
    denominator_a = _per_shot(denominator_a, 'denominator_a', shots)
    denominator_b = _per_shot(denominator_b, 'denominator_b', shots)
    for numerator, denominator, name in ((numerator_a, denominator_a, 'a'), (numerator_b, denominator_b, 'b')):
        if (numerator > denominator).any():
            raise ValueError(f'numerator_{name} exceeds denominator_{name} on some shot')

    columns = np.stack([numerator_a, numerator_b, denominator_a, denominator_b], axis=1)
    estimate_a = _ratio(numerator_a.sum(), denominator_a.sum())
    estimate_b = _ratio(numerator_b.sum(), denominator_b.sum())

    sums = _resampled_sums(columns, replicates=int(replicates), seed=int(seed))
    usable = (sums[:, 2] > 0) & (sums[:, 3] > 0)
    empty = int(replicates - usable.sum())
    if usable.any():
        side_a = sums[usable, 0] / sums[usable, 2]
        side_b = sums[usable, 1] / sums[usable, 3]
        low_a, high_a = np.percentile(side_a, INTERVAL_PERCENTILES)
        low_b, high_b = np.percentile(side_b, INTERVAL_PERCENTILES)
        low, high = np.percentile(side_b - side_a, INTERVAL_PERCENTILES)
    else:
        low_a = high_a = low_b = high_b = low = high = float('nan')
    return PairedDifference(
        estimate_a=estimate_a, estimate_b=estimate_b, low_a=low_a, high_a=high_a,
        low_b=low_b, high_b=high_b, difference=estimate_b - estimate_a, low=low, high=high,
        replicates=int(replicates), seed=int(seed), zero_denominator_replicates=empty)


# --- summaries ---------------------------------------------------------------------


def _checked_pair(record: L1Record, result: ReplayResult) -> str:
    """The reference decoder a result was replayed against, checked against the record."""
    if not isinstance(record, L1Record):
        raise TypeError(f'record must be an L1Record, got {type(record).__name__}')
    if not isinstance(result, ReplayResult):
        raise TypeError(f'result must be a ReplayResult, got {type(result).__name__}')
    if result.shots != record.shots:
        raise ValueError(f'{result.config} covers {result.shots} shots, the record has {record.shots}')
    if result.num_patches != record.num_patches:
        raise ValueError(f'{result.config} covers {result.num_patches} patches, '
                         f'the record has {record.num_patches}')
    return result.reference


def summarize_result(record: L1Record, result: ReplayResult, *, pieces) -> dict:
    """One configuration's accuracy, strata, diagnostics, and replay work, JSON-ready."""
    reference = _checked_pair(record, result)
    failures = sector_failures(result.final, record.actual)
    counts = residual_failure_counts(record.reference(reference), record.actual)
    block = Rate(int(block_failures(result.final, record.actual).sum()), record.shots)
    sector_rate = {name: Rate(int(failures[:, sector].sum()), record.shots).to_json()
                   for sector, name in enumerate(SECTOR_NAMES)}
    return {
        'config': result.config,
        'reference': reference,
        'shots': record.shots,
        'patches': record.num_patches,
        'pieces': int(pieces),
        'values': OUTER_LOGICAL_VALUES,
        'block_failure': block.to_json(),
        'normalized_ler': normalized_ler(block.value, pieces=pieces),
        'sector_failure': sector_rate,
        'misattribution': {key: rate.to_json() for key, rate in misattribution(failures, counts).items()},
        'strata': {name: {key: rate.to_json() for key, rate in rates.items()}
                   for name, rates in failure_by_stratum(failures, counts).items()},
        'ties': {key: rate.to_json() for key, rate in tie_frequency(result.ties).items()},
        'above_half': above_half_frequency(result.above_half).to_json(),
        'work': result.work.to_json(),
    }


def compare_endpoints(record: L1Record, initial: ReplayResult, refined: ReplayResult, *,
                      pieces, replicates: int, seed: int) -> dict:
    """The paired ``all_refined - initial_only`` comparison on one record.

    Both configurations must have been replayed against the same reference on
    the same shots, so they share the eligible population of the primary
    metric. Pooled misattribution passes each shot's eligible-sector count as
    the denominator and its misattributed count as the numerator; block failure
    passes per-shot indicators with denominator one.
    """
    reference = _checked_pair(record, initial)
    if reference != _checked_pair(record, refined):
        raise ValueError(f'{initial.config} and {refined.config} use different reference decoders, '
                         f'so their eligible sectors are not the same population')
    counts = residual_failure_counts(record.reference(reference), record.actual)
    eligible = counts == 1
    eligible_per_shot = eligible.sum(axis=1).astype(np.float64)
    misattributed = [(sector_failures(result.final, record.actual) & eligible).sum(axis=1).astype(np.float64)
                     for result in (initial, refined)]
    failed = [block_failures(result.final, record.actual).astype(np.float64) for result in (initial, refined)]
    ones = np.ones(record.shots, dtype=np.float64)
    return {
        'shots': record.shots,
        'reference': reference,
        'pieces': int(pieces),
        'initial': summarize_result(record, initial, pieces=pieces),
        'refined': summarize_result(record, refined, pieces=pieces),
        'misattribution_pooled': paired_bootstrap(
            misattributed[0], misattributed[1], eligible_per_shot, eligible_per_shot,
            replicates=replicates, seed=seed).to_json(),
        'block_failure': paired_bootstrap(
            failed[0], failed[1], ones, ones, replicates=replicates, seed=seed).to_json(),
    }
