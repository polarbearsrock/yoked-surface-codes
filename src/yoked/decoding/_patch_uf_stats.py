"""Statistics for the confidence-gated patch-UF experiment.

This module is deliberately independent of collection and artifact schemas.  It
accepts already authenticated integer counts, validates the experiment's exact
reconciliation identities, and returns immutable summaries.  Randomized
operations use an explicit NumPy ``Generator`` seed and never use global RNG
state.

The maintained ProMatch implementations remain the source of truth for exact
Clopper--Pearson endpoints, the one-sided Tango efficient-score bound, and the
explicit empirical type-7 quantile.  This module only supplies the experiment-
specific two-sided wrappers and grouped bootstrap designs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from yoked.decoding._promatch_stats import (
    PairedContingency,
    clopper_pearson_lower,
    clopper_pearson_upper,
    empirical_type7_quantile,
    tango_paired_risk_difference_upper,
)


__all__ = [
    "BootstrapEndpoint",
    "CLUSTER_QUANTILE_PROBABILITIES",
    "ClusterBootstrap",
    "ClusterSizeBin",
    "ClusterSizeSummary",
    "HRLKCell",
    "NullableEstimate",
    "PairedAccuracySummary",
    "ShotClusterRecord",
    "SparseDistributionSummary",
    "TwoSidedInterval",
    "WorkloadCoverageBootstrap",
    "WorkloadCoverageSummary",
    "bootstrap_cluster_sizes",
    "bootstrap_workload_coverage",
    "canonical_sparse_histogram",
    "clopper_pearson_interval",
    "merge_sparse_histograms",
    "sparse_histogram_total",
    "sparse_type7_quantile",
    "summarize_cluster_sizes",
    "summarize_paired_accuracy",
    "summarize_workload_coverage",
    "tango_paired_risk_difference_interval",
    "validate_hrlk_histogram",
]


CLUSTER_QUANTILE_PROBABILITIES = (0.5, 0.9, 0.95, 0.99)
_ESTIMABLE = "estimable"
_NOT_ESTIMABLE = "not-estimable"
_INCOMPLETE_BOOTSTRAP = "interval-not-estimable"


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: object, *, name: str, open_interval: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if open_interval:
        if not 0 < result < 1:
            raise ValueError(f"{name} must lie strictly between 0 and 1")
    elif not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class TwoSidedInterval:
    """A finite equal-tail or percentile interval with total error ``alpha``."""

    lower: float
    upper: float
    alpha: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        alpha = _probability(self.alpha, name="alpha", open_interval=True)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("interval endpoints must be finite")
        if lower > upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "alpha", alpha)


@dataclass(frozen=True)
class NullableEstimate:
    """A JSON-safe point estimate with an explicit estimability status."""

    value: float | None
    status: Literal["estimable", "not-estimable"]

    def __post_init__(self) -> None:
        if self.status not in {_ESTIMABLE, _NOT_ESTIMABLE}:
            raise ValueError("unknown estimate status")
        if self.status == _ESTIMABLE:
            if self.value is None or not math.isfinite(float(self.value)):
                raise ValueError("an estimable endpoint needs one finite value")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError("a non-estimable endpoint must have value=None")


@dataclass(frozen=True)
class BootstrapEndpoint:
    """Point estimate plus an all-replicates-estimable percentile interval."""

    estimate: NullableEstimate
    interval: TwoSidedInterval | None
    estimable_replicates: int
    replicates: int
    status: Literal[
        "estimable", "not-estimable", "interval-not-estimable"
    ]

    def __post_init__(self) -> None:
        estimable = _nonnegative_int(
            self.estimable_replicates, name="estimable_replicates"
        )
        replicates = _positive_int(self.replicates, name="replicates")
        if estimable > replicates:
            raise ValueError("estimable_replicates cannot exceed replicates")
        expected_status: str
        if self.estimate.value is None:
            expected_status = _NOT_ESTIMABLE
        elif estimable != replicates:
            expected_status = _INCOMPLETE_BOOTSTRAP
        else:
            expected_status = _ESTIMABLE
        if self.status != expected_status:
            raise ValueError(
                f"bootstrap status must be {expected_status!r} for these values"
            )
        if (self.interval is not None) != (expected_status == _ESTIMABLE):
            raise ValueError(
                "a bootstrap interval exists exactly when all replicates are estimable"
            )
        object.__setattr__(self, "estimable_replicates", estimable)
        object.__setattr__(self, "replicates", replicates)


def _bootstrap_endpoint(
    *,
    estimate: float | None,
    values: Sequence[float],
    replicates: int,
    alpha: float,
) -> BootstrapEndpoint:
    point = NullableEstimate(
        value=None if estimate is None else float(estimate),
        status=_NOT_ESTIMABLE if estimate is None else _ESTIMABLE,
    )
    estimable = len(values)
    if estimate is None:
        status = _NOT_ESTIMABLE
        interval = None
    elif estimable != replicates:
        status = _INCOMPLETE_BOOTSTRAP
        interval = None
    else:
        status = _ESTIMABLE
        interval = TwoSidedInterval(
            lower=empirical_type7_quantile(values, alpha / 2),
            upper=empirical_type7_quantile(values, 1 - alpha / 2),
            alpha=alpha,
        )
    return BootstrapEndpoint(
        estimate=point,
        interval=interval,
        estimable_replicates=estimable,
        replicates=replicates,
        status=status,  # type: ignore[arg-type]
    )


# Paired accuracy.


def clopper_pearson_interval(
    *, successes: int, trials: int, alpha: float = 0.05
) -> TwoSidedInterval:
    """Two-sided equal-tail exact Clopper--Pearson binomial interval."""

    alpha = _probability(alpha, name="alpha", open_interval=True)
    successes = _nonnegative_int(successes, name="successes")
    trials = _positive_int(trials, name="trials")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    return TwoSidedInterval(
        lower=clopper_pearson_lower(
            successes=successes, trials=trials, alpha=alpha / 2
        ),
        upper=clopper_pearson_upper(
            successes=successes, trials=trials, alpha=alpha / 2
        ),
        alpha=alpha,
    )


def tango_paired_risk_difference_interval(
    table: PairedContingency,
    *,
    alpha: float = 0.05,
    xtol: float = 1e-12,
    max_iterations: int = 200,
) -> TwoSidedInterval:
    """Two-sided Tango interval for treatment-minus-baseline failure risk."""

    if not isinstance(table, PairedContingency):
        raise TypeError("table must be a PairedContingency")
    if table.shots == 0:
        raise ValueError("Tango interval is undefined for an empty table")
    alpha = _probability(alpha, name="alpha", open_interval=True)
    if not math.isfinite(xtol) or xtol <= 0:
        raise ValueError("xtol must be finite and positive")
    max_iterations = _positive_int(max_iterations, name="max_iterations")
    upper = tango_paired_risk_difference_upper(
        table,
        alpha=alpha / 2,
        xtol=xtol,
        max_iterations=max_iterations,
    )
    swapped = PairedContingency(
        both_correct=table.both_correct,
        regressions=table.recoveries,
        recoveries=table.regressions,
        both_wrong=table.both_wrong,
    )
    lower = -tango_paired_risk_difference_upper(
        swapped,
        alpha=alpha / 2,
        xtol=xtol,
        max_iterations=max_iterations,
    )
    return TwoSidedInterval(lower=lower, upper=upper, alpha=alpha)


@dataclass(frozen=True)
class PairedAccuracySummary:
    """Validated primary ``a/b/c/d`` accuracy summary."""

    a: int
    b: int
    c: int
    d: int
    shots: int
    global_failures: int
    treatment_failures: int
    global_failure_rate: float
    treatment_failure_rate: float
    risk_difference: float
    discordant: int
    discordance_rate: float
    prediction_agreements: int | None
    prediction_agreement_rate: float | None
    global_failure_interval: TwoSidedInterval
    treatment_failure_interval: TwoSidedInterval
    risk_difference_interval: TwoSidedInterval


def summarize_paired_accuracy(
    *,
    a: int,
    b: int,
    c: int,
    d: int,
    prediction_agreements: int | None = None,
    alpha: float = 0.05,
) -> PairedAccuracySummary:
    """Validate and summarize the standard paired correctness table.

    ``a`` is both correct, ``b`` is baseline correct/treatment wrong, ``c`` is
    baseline wrong/treatment correct, and ``d`` is both wrong.  Packed-
    prediction agreement is separate from this correctness table and therefore
    must be supplied independently when it is available.
    """

    table = PairedContingency(
        both_correct=a,
        regressions=b,
        recoveries=c,
        both_wrong=d,
    )
    if table.shots == 0:
        raise ValueError("paired accuracy summary requires at least one shot")
    alpha = _probability(alpha, name="alpha", open_interval=True)
    agreement_count: int | None
    agreement_rate: float | None
    if prediction_agreements is None:
        agreement_count = None
        agreement_rate = None
    else:
        agreement_count = _nonnegative_int(
            prediction_agreements, name="prediction_agreements"
        )
        if agreement_count > table.shots:
            raise ValueError("prediction_agreements cannot exceed shots")
        agreement_rate = agreement_count / table.shots
    return PairedAccuracySummary(
        a=table.both_correct,
        b=table.regressions,
        c=table.recoveries,
        d=table.both_wrong,
        shots=table.shots,
        global_failures=table.baseline_failures,
        treatment_failures=table.treatment_failures,
        global_failure_rate=table.baseline_failures / table.shots,
        treatment_failure_rate=table.treatment_failures / table.shots,
        risk_difference=table.delta,
        discordant=table.discordant,
        discordance_rate=table.discordant / table.shots,
        prediction_agreements=agreement_count,
        prediction_agreement_rate=agreement_rate,
        global_failure_interval=clopper_pearson_interval(
            successes=table.baseline_failures,
            trials=table.shots,
            alpha=alpha,
        ),
        treatment_failure_interval=clopper_pearson_interval(
            successes=table.treatment_failures,
            trials=table.shots,
            alpha=alpha,
        ),
        risk_difference_interval=tango_paired_risk_difference_interval(
            table,
            alpha=alpha,
        ),
    )


# Exact sparse integer distributions and grouped cluster bootstrap.


SparseHistogramInput = Mapping[int, int] | Sequence[tuple[int, int]]


def canonical_sparse_histogram(
    histogram: SparseHistogramInput,
    *,
    minimum_key: int = 0,
    allow_empty: bool = True,
) -> tuple[tuple[int, int], ...]:
    """Return a sorted immutable sparse histogram after strict validation."""

    minimum_key = _nonnegative_int(minimum_key, name="minimum_key")
    if isinstance(histogram, Mapping):
        raw_items = tuple(histogram.items())
    elif isinstance(histogram, Sequence) and not isinstance(
        histogram, (str, bytes, bytearray)
    ):
        raw_items = tuple(histogram)
    else:
        raise TypeError("histogram must be a mapping or a sequence of pairs")
    normalized: dict[int, int] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, Sequence) or len(raw_item) != 2:
            raise TypeError("histogram entries must be (key, count) pairs")
        key = _nonnegative_int(raw_item[0], name="histogram key")
        count = _positive_int(raw_item[1], name="histogram count")
        if key < minimum_key:
            raise ValueError(f"histogram key must be at least {minimum_key}")
        if key in normalized:
            raise ValueError(f"histogram contains duplicate key {key}")
        normalized[key] = count
    if not normalized and not allow_empty:
        raise ValueError("histogram must not be empty")
    return tuple(sorted(normalized.items()))


def sparse_histogram_total(histogram: SparseHistogramInput) -> int:
    """Return the exact denominator of a validated sparse histogram."""

    return sum(count for _, count in canonical_sparse_histogram(histogram))


def merge_sparse_histograms(
    histograms: Iterable[SparseHistogramInput],
    *,
    minimum_key: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Add sparse histograms exactly and return canonical sorted cells."""

    result: dict[int, int] = {}
    for histogram in histograms:
        for key, count in canonical_sparse_histogram(
            histogram, minimum_key=minimum_key
        ):
            result[key] = result.get(key, 0) + count
    return tuple(sorted(result.items()))


def _histogram_rank(
    histogram: tuple[tuple[int, int], ...], rank: int
) -> int:
    cursor = 0
    for value, count in histogram:
        cursor += count
        if rank < cursor:
            return value
    raise AssertionError("histogram rank exceeds its denominator")


def sparse_type7_quantile(
    histogram: SparseHistogramInput, probability: float
) -> float:
    """Type-7 quantile computed from exact sparse integer multiplicities."""

    probability = _probability(
        probability, name="probability", open_interval=False
    )
    normalized = canonical_sparse_histogram(histogram, allow_empty=False)
    denominator = sum(count for _, count in normalized)
    h = (denominator - 1) * probability
    lower_rank = math.floor(h)
    fraction = h - lower_rank
    lower = _histogram_rank(normalized, lower_rank)
    if fraction == 0 or lower_rank + 1 == denominator:
        return float(lower)
    upper = _histogram_rank(normalized, lower_rank + 1)
    return float(lower + fraction * (upper - lower))


@dataclass(frozen=True)
class SparseDistributionSummary:
    """Exact sparse counts and the experiment's required type-7 summaries."""

    histogram: tuple[tuple[int, int], ...]
    denominator: int
    status: Literal["estimable", "not-estimable"]
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    maximum: int | None


def _summarize_sparse_distribution(
    histogram: SparseHistogramInput, *, minimum_key: int
) -> SparseDistributionSummary:
    normalized = canonical_sparse_histogram(
        histogram, minimum_key=minimum_key
    )
    denominator = sum(count for _, count in normalized)
    if denominator == 0:
        return SparseDistributionSummary(
            histogram=normalized,
            denominator=0,
            status=_NOT_ESTIMABLE,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
            maximum=None,
        )
    values = tuple(
        sparse_type7_quantile(normalized, p)
        for p in CLUSTER_QUANTILE_PROBABILITIES
    )
    return SparseDistributionSummary(
        histogram=normalized,
        denominator=denominator,
        status=_ESTIMABLE,
        p50=values[0],
        p90=values[1],
        p95=values[2],
        p99=values[3],
        maximum=normalized[-1][0],
    )


@dataclass(frozen=True)
class ShotClusterRecord:
    """One complete-shot bootstrap group and its completed components."""

    global_shot_id: int
    cluster_summary_complete: bool
    completed_component_size_histogram: SparseHistogramInput
    maximum_final_component_defect_count: int | None

    def __post_init__(self) -> None:
        shot_id = _nonnegative_int(self.global_shot_id, name="global_shot_id")
        if not isinstance(self.cluster_summary_complete, bool):
            raise TypeError("cluster_summary_complete must be bool")
        histogram = canonical_sparse_histogram(
            self.completed_component_size_histogram,
            minimum_key=1,
        )
        if self.cluster_summary_complete:
            maximum = _nonnegative_int(
                self.maximum_final_component_defect_count,
                name="maximum_final_component_defect_count",
            )
            expected = 0 if not histogram else histogram[-1][0]
            if maximum != expected:
                raise ValueError(
                    "complete-shot maximum does not match its component histogram"
                )
        else:
            if self.maximum_final_component_defect_count is not None:
                raise ValueError("an incomplete-shot final-component maximum must be null")
            maximum = None
        object.__setattr__(self, "global_shot_id", shot_id)
        object.__setattr__(self, "completed_component_size_histogram", histogram)
        object.__setattr__(self, "maximum_final_component_defect_count", maximum)


@dataclass(frozen=True)
class ClusterSizeSummary:
    """Component-weighted and complete-shot-weighted cluster distributions."""

    shots: int
    complete_cluster_summary_shots: int
    censored_cluster_summary_shots: int
    censor_rate: float
    completed_components: SparseDistributionSummary
    complete_shot_maxima: SparseDistributionSummary


def _validated_cluster_records(
    records: Iterable[ShotClusterRecord],
) -> tuple[ShotClusterRecord, ...]:
    normalized = tuple(records)
    if not normalized:
        raise ValueError("cluster records must not be empty")
    if any(not isinstance(record, ShotClusterRecord) for record in normalized):
        raise TypeError("cluster records must contain ShotClusterRecord values")
    shot_ids = [record.global_shot_id for record in normalized]
    if len(set(shot_ids)) != len(shot_ids):
        raise ValueError("cluster records contain duplicate global_shot_id values")
    return tuple(sorted(normalized, key=lambda record: record.global_shot_id))


def summarize_cluster_sizes(
    records: Iterable[ShotClusterRecord],
) -> ClusterSizeSummary:
    """Build the two noninterchangeable exact cluster-size views."""

    normalized = _validated_cluster_records(records)
    component_histogram = merge_sparse_histograms(
        (record.completed_component_size_histogram for record in normalized),
        minimum_key=1,
    )
    maximum_histogram: dict[int, int] = {}
    complete = 0
    for record in normalized:
        if not record.cluster_summary_complete:
            continue
        complete += 1
        maximum = record.maximum_final_component_defect_count
        if maximum is None:  # Defensive; ShotClusterRecord already rejects this.
            raise AssertionError("complete cluster record has no maximum")
        maximum_histogram[maximum] = maximum_histogram.get(maximum, 0) + 1
    shots = len(normalized)
    return ClusterSizeSummary(
        shots=shots,
        complete_cluster_summary_shots=complete,
        censored_cluster_summary_shots=shots - complete,
        censor_rate=(shots - complete) / shots,
        completed_components=_summarize_sparse_distribution(
            component_histogram, minimum_key=1
        ),
        complete_shot_maxima=_summarize_sparse_distribution(
            maximum_histogram, minimum_key=0
        ),
    )


@dataclass(frozen=True)
class ClusterSizeBin:
    """One predeclared half-open component-size display bin."""

    label: str
    minimum: int
    maximum_exclusive: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("cluster-size bin label must be a nonempty string")
        minimum = _positive_int(self.minimum, name="bin minimum")
        if self.maximum_exclusive is None:
            maximum = None
        else:
            maximum = _positive_int(
                self.maximum_exclusive, name="bin maximum_exclusive"
            )
            if maximum <= minimum:
                raise ValueError("bin maximum_exclusive must exceed its minimum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum_exclusive", maximum)

    def contains(self, value: int) -> bool:
        return value >= self.minimum and (
            self.maximum_exclusive is None or value < self.maximum_exclusive
        )


def _validated_cluster_bins(
    bins: Iterable[ClusterSizeBin],
) -> tuple[ClusterSizeBin, ...]:
    normalized = tuple(bins)
    if any(not isinstance(item, ClusterSizeBin) for item in normalized):
        raise TypeError("component bins must contain ClusterSizeBin values")
    if len({item.label for item in normalized}) != len(normalized):
        raise ValueError("component-bin labels must be unique")
    previous_stop = 1
    for index, item in enumerate(normalized):
        if item.minimum < previous_stop:
            raise ValueError("component bins must be ordered and nonoverlapping")
        if index + 1 < len(normalized) and item.maximum_exclusive is None:
            raise ValueError("only the final component bin may be open-ended")
        previous_stop = (
            item.minimum
            if item.maximum_exclusive is None
            else item.maximum_exclusive
        )
    return normalized


@dataclass(frozen=True)
class ClusterBootstrap:
    """Complete-shot grouped percentile intervals for cluster endpoints."""

    replicates: int
    alpha: float
    shot_quantiles: tuple[tuple[float, BootstrapEndpoint], ...]
    component_quantiles: tuple[tuple[float, BootstrapEndpoint], ...]
    component_bin_proportions: tuple[tuple[ClusterSizeBin, BootstrapEndpoint], ...]


def _histogram_bin_count(
    histogram: tuple[tuple[int, int], ...], size_bin: ClusterSizeBin
) -> int:
    return sum(count for value, count in histogram if size_bin.contains(value))


def _scaled_histogram_add(
    target: dict[int, int],
    source: tuple[tuple[int, int], ...],
    multiplicity: int,
) -> None:
    if multiplicity == 0:
        return
    for value, count in source:
        target[value] = target.get(value, 0) + multiplicity * count


def bootstrap_cluster_sizes(
    *,
    records: Iterable[ShotClusterRecord],
    component_bins: Iterable[ClusterSizeBin] = (),
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> ClusterBootstrap:
    """Bootstrap cluster endpoints by resampling whole physical shots.

    Every draw carries the selected shot's completeness flag, maximum, and
    complete component histogram together.  Component rows are never sampled
    independently.  An endpoint gets no interval unless all replicates have a
    nonzero denominator for that endpoint.
    """

    normalized = _validated_cluster_records(records)
    bins = _validated_cluster_bins(component_bins)
    replicates = _positive_int(replicates, name="replicates")
    seed = _nonnegative_int(seed, name="seed")
    alpha = _probability(alpha, name="alpha", open_interval=True)
    point = summarize_cluster_sizes(normalized)
    shot_point = {
        probability: (
            None
            if point.complete_shot_maxima.denominator == 0
            else sparse_type7_quantile(
                point.complete_shot_maxima.histogram, probability
            )
        )
        for probability in CLUSTER_QUANTILE_PROBABILITIES
    }
    component_point = {
        probability: (
            None
            if point.completed_components.denominator == 0
            else sparse_type7_quantile(
                point.completed_components.histogram, probability
            )
        )
        for probability in CLUSTER_QUANTILE_PROBABILITIES
    }
    bin_point = {
        item.label: (
            None
            if point.completed_components.denominator == 0
            else _histogram_bin_count(point.completed_components.histogram, item)
            / point.completed_components.denominator
        )
        for item in bins
    }

    shot_values = {p: [] for p in CLUSTER_QUANTILE_PROBABILITIES}
    component_values = {p: [] for p in CLUSTER_QUANTILE_PROBABILITIES}
    bin_values = {item.label: [] for item in bins}
    rng = np.random.default_rng(seed)
    shots = len(normalized)
    for _ in range(replicates):
        selected = rng.integers(0, shots, size=shots)
        multiplicities = np.bincount(selected, minlength=shots)
        maximum_histogram: dict[int, int] = {}
        component_histogram: dict[int, int] = {}
        for index, raw_multiplicity in enumerate(multiplicities):
            multiplicity = int(raw_multiplicity)
            if multiplicity == 0:
                continue
            record = normalized[index]
            if record.cluster_summary_complete:
                maximum = record.maximum_final_component_defect_count
                if maximum is None:
                    raise AssertionError("complete cluster record has no maximum")
                maximum_histogram[maximum] = (
                    maximum_histogram.get(maximum, 0) + multiplicity
                )
            _scaled_histogram_add(
                component_histogram,
                record.completed_component_size_histogram,  # type: ignore[arg-type]
                multiplicity,
            )
        if maximum_histogram:
            for probability in CLUSTER_QUANTILE_PROBABILITIES:
                shot_values[probability].append(
                    sparse_type7_quantile(maximum_histogram, probability)
                )
        if component_histogram:
            component_total = sum(component_histogram.values())
            for probability in CLUSTER_QUANTILE_PROBABILITIES:
                component_values[probability].append(
                    sparse_type7_quantile(component_histogram, probability)
                )
            canonical_components = tuple(sorted(component_histogram.items()))
            for item in bins:
                bin_values[item.label].append(
                    _histogram_bin_count(canonical_components, item) / component_total
                )

    return ClusterBootstrap(
        replicates=replicates,
        alpha=alpha,
        shot_quantiles=tuple(
            (
                probability,
                _bootstrap_endpoint(
                    estimate=shot_point[probability],
                    values=shot_values[probability],
                    replicates=replicates,
                    alpha=alpha,
                ),
            )
            for probability in CLUSTER_QUANTILE_PROBABILITIES
        ),
        component_quantiles=tuple(
            (
                probability,
                _bootstrap_endpoint(
                    estimate=component_point[probability],
                    values=component_values[probability],
                    replicates=replicates,
                    alpha=alpha,
                ),
            )
            for probability in CLUSTER_QUANTILE_PROBABILITIES
        ),
        component_bin_proportions=tuple(
            (
                item,
                _bootstrap_endpoint(
                    estimate=bin_point[item.label],
                    values=bin_values[item.label],
                    replicates=replicates,
                    alpha=alpha,
                ),
            )
            for item in bins
        ),
    )


# H/R/L/K reconciliation and multinomial workload/coverage bootstrap.


@dataclass(frozen=True, order=True)
class HRLKCell:
    """One exact joint-histogram cell and its positive shot multiplicity."""

    original: int
    residual: int
    lane_owned: int
    committed: int
    count: int

    def __post_init__(self) -> None:
        original = _nonnegative_int(self.original, name="H original")
        residual = _nonnegative_int(self.residual, name="R residual")
        lane_owned = _nonnegative_int(self.lane_owned, name="L lane_owned")
        committed = _nonnegative_int(self.committed, name="K committed")
        count = _positive_int(self.count, name="joint count")
        if not 0 <= committed <= lane_owned <= original:
            raise ValueError("H/R/L/K must satisfy 0 <= K <= L <= H")
        if residual != original - committed:
            raise ValueError("H/R/L/K must satisfy R = H - K")
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "lane_owned", lane_owned)
        object.__setattr__(self, "committed", committed)
        object.__setattr__(self, "count", count)


def validate_hrlk_histogram(
    joint_counts: Mapping[tuple[int, int, int, int], int],
) -> tuple[HRLKCell, ...]:
    """Validate and canonicalize the exact complete-shot H/R/L/K histogram."""

    if not isinstance(joint_counts, Mapping) or not joint_counts:
        raise ValueError("H/R/L/K joint histogram must be a nonempty mapping")
    cells: list[HRLKCell] = []
    for raw_key, raw_count in joint_counts.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise TypeError("H/R/L/K histogram keys must be four-tuples")
        cells.append(HRLKCell(*raw_key, raw_count))
    cells.sort()
    return tuple(cells)


@dataclass(frozen=True)
class WorkloadCoverageSummary:
    """Unconditional workload and frontend-coverage point estimands."""

    shots: int
    original_total: int
    residual_total: int
    lane_owned_total: int
    committed_total: int
    original_mean: float
    residual_mean: float
    lane_owned_mean: float
    committed_mean: float
    workload_mean_difference: float
    workload_ratio: NullableEstimate
    frontend_coverage: NullableEstimate
    lane_owned_coverage: NullableEstimate


def _nullable_ratio(numerator: int, denominator: int) -> NullableEstimate:
    if denominator == 0:
        return NullableEstimate(value=None, status=_NOT_ESTIMABLE)
    return NullableEstimate(value=numerator / denominator, status=_ESTIMABLE)


def _workload_totals(cells: Sequence[HRLKCell]) -> tuple[int, int, int, int, int]:
    shots = sum(cell.count for cell in cells)
    original = sum(cell.count * cell.original for cell in cells)
    residual = sum(cell.count * cell.residual for cell in cells)
    lane_owned = sum(cell.count * cell.lane_owned for cell in cells)
    committed = sum(cell.count * cell.committed for cell in cells)
    return shots, original, residual, lane_owned, committed


def _summary_from_cells(cells: Sequence[HRLKCell]) -> WorkloadCoverageSummary:
    shots, original, residual, lane_owned, committed = _workload_totals(cells)
    if shots == 0:
        raise AssertionError("validated H/R/L/K cells have no shots")
    if residual != original - committed:
        raise AssertionError("aggregate H/R/L/K reconciliation failed")
    workload_ratio = _nullable_ratio(residual, original)
    frontend_coverage = _nullable_ratio(committed, original)
    lane_coverage = _nullable_ratio(committed, lane_owned)
    if workload_ratio.value is not None and frontend_coverage.value is not None:
        if not math.isclose(
            frontend_coverage.value,
            1 - workload_ratio.value,
            rel_tol=0,
            abs_tol=2 * np.finfo(np.float64).eps,
        ):
            raise AssertionError("coverage/workload identity failed")
    return WorkloadCoverageSummary(
        shots=shots,
        original_total=original,
        residual_total=residual,
        lane_owned_total=lane_owned,
        committed_total=committed,
        original_mean=original / shots,
        residual_mean=residual / shots,
        lane_owned_mean=lane_owned / shots,
        committed_mean=committed / shots,
        workload_mean_difference=(residual - original) / shots,
        workload_ratio=workload_ratio,
        frontend_coverage=frontend_coverage,
        lane_owned_coverage=lane_coverage,
    )


def summarize_workload_coverage(
    joint_counts: Mapping[tuple[int, int, int, int], int],
) -> WorkloadCoverageSummary:
    """Validate H/R/L/K and compute all workload/coverage point estimands."""

    return _summary_from_cells(validate_hrlk_histogram(joint_counts))


@dataclass(frozen=True)
class WorkloadCoverageBootstrap:
    """Shared multinomial-bootstrap intervals for H/R/L/K endpoints."""

    replicates: int
    alpha: float
    workload_ratio: BootstrapEndpoint
    workload_mean_difference: BootstrapEndpoint
    frontend_coverage: BootstrapEndpoint
    lane_owned_coverage: BootstrapEndpoint


def bootstrap_workload_coverage(
    *,
    joint_counts: Mapping[tuple[int, int, int, int], int],
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> WorkloadCoverageBootstrap:
    """Exact complete-shot multinomial bootstrap over the H/R/L/K histogram.

    A replicate with a zero denominator is omitted only for that endpoint and
    is counted as non-estimable.  In accordance with the frozen experiment,
    the endpoint then receives no interval unless every replicate is
    estimable.  Numeric infinity and NaN are never produced.
    """

    cells = validate_hrlk_histogram(joint_counts)
    point = _summary_from_cells(cells)
    replicates = _positive_int(replicates, name="replicates")
    seed = _nonnegative_int(seed, name="seed")
    alpha = _probability(alpha, name="alpha", open_interval=True)
    shots = point.shots
    probabilities = np.asarray([cell.count / shots for cell in cells])
    rng = np.random.default_rng(seed)
    ratio_values: list[float] = []
    difference_values: list[float] = []
    frontend_values: list[float] = []
    lane_values: list[float] = []
    for _ in range(replicates):
        sampled = rng.multinomial(shots, probabilities)
        original = residual = lane_owned = committed = 0
        for cell, raw_count in zip(cells, sampled):
            count = int(raw_count)
            original += count * cell.original
            residual += count * cell.residual
            lane_owned += count * cell.lane_owned
            committed += count * cell.committed
        if residual != original - committed:
            raise AssertionError("bootstrap H/R/L/K reconciliation failed")
        difference_values.append((residual - original) / shots)
        if original != 0:
            ratio_values.append(residual / original)
            frontend_values.append(committed / original)
        if lane_owned != 0:
            lane_values.append(committed / lane_owned)
    return WorkloadCoverageBootstrap(
        replicates=replicates,
        alpha=alpha,
        workload_ratio=_bootstrap_endpoint(
            estimate=point.workload_ratio.value,
            values=ratio_values,
            replicates=replicates,
            alpha=alpha,
        ),
        workload_mean_difference=_bootstrap_endpoint(
            estimate=point.workload_mean_difference,
            values=difference_values,
            replicates=replicates,
            alpha=alpha,
        ),
        frontend_coverage=_bootstrap_endpoint(
            estimate=point.frontend_coverage.value,
            values=frontend_values,
            replicates=replicates,
            alpha=alpha,
        ),
        lane_owned_coverage=_bootstrap_endpoint(
            estimate=point.lane_owned_coverage.value,
            values=lane_values,
            replicates=replicates,
            alpha=alpha,
        ),
    )
