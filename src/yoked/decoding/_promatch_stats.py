"""Statistical and protocol utilities for the ProMatch first-round study.

The functions in this module deliberately avoid hidden global RNG state.  Every
randomized operation takes an explicit seed, and protocol serialization rejects
values (for example NaN) that do not have a portable JSON representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, norm


__all__ = [
    "ArrayDigest",
    "BatchSpec",
    "BootstrapRatio",
    "HierarchicalTimingBootstrap",
    "PairedContingency",
    "ROUND_ONE_BATCH_SIZE",
    "ROUND_ONE_MAX_PROCESSES",
    "SampleSizeDesign",
    "SimulatedPower",
    "canonical_json_bytes",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "confirmatory_sample_size",
    "derive_stim_batch_seed",
    "digest_array",
    "empirical_type7_quantile",
    "hierarchical_timing_bootstrap",
    "manifest_experiment_id",
    "paired_geometric_mean_ratio",
    "paired_workload_bootstrap",
    "paired_workload_histogram_bootstrap",
    "paired_workload_ratio",
    "simulate_tango_noninferiority_power",
    "tango_paired_risk_difference_upper",
    "validate_batch_schedule",
    "validate_process_count",
    "validate_protocol_manifest",
]


ROUND_ONE_BATCH_SIZE = 10_000
ROUND_ONE_MAX_PROCESSES = 32
Z_ONE_SIDED_97_5 = 1.959963984540054
Z_POWER_90 = 1.2815515655446004


def _require_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class PairedContingency:
    """A complete paired correctness table.

    ``regressions`` is b = #(U0 correct, PU wrong), and ``recoveries`` is
    c = #(U0 wrong, PU correct).  Consequently ``delta`` is the PU-minus-U0
    any-observable failure risk difference.
    """

    both_correct: int
    regressions: int
    recoveries: int
    both_wrong: int

    def __post_init__(self) -> None:
        for name in ("both_correct", "regressions", "recoveries", "both_wrong"):
            object.__setattr__(
                self,
                name,
                _require_nonnegative_int(getattr(self, name), name=name),
            )

    @property
    def shots(self) -> int:
        return self.both_correct + self.regressions + self.recoveries + self.both_wrong

    @property
    def discordant(self) -> int:
        return self.regressions + self.recoveries

    @property
    def baseline_failures(self) -> int:
        return self.recoveries + self.both_wrong

    @property
    def treatment_failures(self) -> int:
        return self.regressions + self.both_wrong

    @property
    def delta(self) -> float:
        if self.shots == 0:
            raise ValueError("delta is undefined for an empty paired table")
        return (self.regressions - self.recoveries) / self.shots

    @classmethod
    def from_failures(
        cls,
        *,
        baseline_failed: Sequence[bool] | np.ndarray,
        treatment_failed: Sequence[bool] | np.ndarray,
    ) -> "PairedContingency":
        baseline = np.asarray(baseline_failed, dtype=np.bool_)
        treatment = np.asarray(treatment_failed, dtype=np.bool_)
        if baseline.ndim != 1 or treatment.ndim != 1:
            raise ValueError("paired failure inputs must be one-dimensional")
        if baseline.shape != treatment.shape:
            raise ValueError("paired failure inputs must have identical shapes")
        return cls(
            both_correct=int(np.count_nonzero(~baseline & ~treatment)),
            regressions=int(np.count_nonzero(~baseline & treatment)),
            recoveries=int(np.count_nonzero(baseline & ~treatment)),
            both_wrong=int(np.count_nonzero(baseline & treatment)),
        )


def clopper_pearson_upper(*, successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided ``1-alpha`` Clopper-Pearson upper binomial bound."""

    successes = _require_nonnegative_int(successes, name="successes")
    trials = _require_nonnegative_int(trials, name="trials")
    if trials == 0:
        raise ValueError("trials must be positive")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if successes == trials:
        return 1.0
    if successes == 0:
        return float(1 - alpha ** (1 / trials))
    return float(beta.ppf(1 - alpha, successes + 1, trials - successes))


def clopper_pearson_lower(*, successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided ``1-alpha`` Clopper-Pearson lower binomial bound."""

    successes = _require_nonnegative_int(successes, name="successes")
    trials = _require_nonnegative_int(trials, name="trials")
    if trials == 0:
        raise ValueError("trials must be positive")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if successes == 0:
        return 0.0
    if successes == trials:
        return float(alpha ** (1 / trials))
    return float(beta.ppf(alpha, successes, trials - successes + 1))


@dataclass(frozen=True)
class SampleSizeDesign:
    discordance_upper: float
    raw_shots: int
    rounded_shots: int
    fits_resource_cap: bool


def confirmatory_sample_size(
    *,
    pilot_discordant: int,
    pilot_shots: int,
    delta_noninferiority: float,
    batch_size: int = ROUND_ONE_BATCH_SIZE,
    max_shots: int = 10_000_000,
    discordance_alpha: float = 0.05,
    z_alpha: float = Z_ONE_SIDED_97_5,
    z_power: float = Z_POWER_90,
) -> SampleSizeDesign:
    """Apply the frozen normal-approximation sample-size rule from Section 15."""

    batch_size = _require_nonnegative_int(batch_size, name="batch_size")
    max_shots = _require_nonnegative_int(max_shots, name="max_shots")
    if batch_size == 0 or max_shots == 0:
        raise ValueError("batch_size and max_shots must be positive")
    if not math.isfinite(delta_noninferiority) or delta_noninferiority <= 0:
        raise ValueError("delta_noninferiority must be finite and positive")
    if not math.isfinite(z_alpha) or not math.isfinite(z_power):
        raise ValueError("z scores must be finite")
    q_upper = clopper_pearson_upper(
        successes=pilot_discordant,
        trials=pilot_shots,
        alpha=discordance_alpha,
    )
    raw = math.ceil(q_upper * (z_alpha + z_power) ** 2 / delta_noninferiority**2)
    rounded = ((raw + batch_size - 1) // batch_size) * batch_size
    return SampleSizeDesign(
        discordance_upper=q_upper,
        raw_shots=raw,
        rounded_shots=rounded,
        fits_resource_cap=rounded <= max_shots,
    )


def _profile_discordance(*, table: PairedContingency, delta0: float) -> float:
    """Profile-MLE of p10+p01 under p10-p01=delta0."""

    n = table.shots
    x = table.discordant
    k = (table.recoveries - table.regressions) * delta0
    discriminant = (x - k) ** 2 + 4 * n * (k + (n - x) * delta0**2)
    # Tiny negative round-off is possible at a constrained boundary.
    discriminant = max(0.0, discriminant)
    q = ((x - k) + math.sqrt(discriminant)) / (2 * n)
    return min(1.0, max(abs(delta0), q))


def _tango_efficient_score(table: PairedContingency, delta0: float) -> float:
    """Signed efficient-score statistic for a hypothesized paired risk difference."""

    n = table.shots
    q = _profile_discordance(table=table, delta0=delta0)
    # The efficient-score statistic simplifies to the form below after using
    # the profile likelihood equation.  Unlike the unsimplified 0/0 form, this
    # remains well-defined when exactly one discordant cell is empty and the
    # nuisance estimate is on q=|delta0|.
    information_denominator = q - delta0 * delta0
    difference = table.delta - delta0
    if information_denominator <= 0:
        if difference == 0:
            return 0.0
        return math.copysign(math.inf, difference)
    return difference * math.sqrt(n / information_denominator)


def tango_paired_risk_difference_upper(
    table: PairedContingency,
    *,
    alpha: float = 0.025,
    xtol: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """One-sided Tango efficient-score upper bound for PU-minus-U0 risk.

    The nuisance discordance probability is profiled under each hypothesized
    difference. Inverting the signed efficient-score test gives the upper
    bound. For zero observed discordance, the profiled boundary solution is
    finite: ``z**2 / (N + z**2)``.
    """

    if table.shots == 0:
        raise ValueError("Tango interval is undefined for an empty table")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between 0 and 0.5")
    if not math.isfinite(xtol) or xtol <= 0:
        raise ValueError("xtol must be finite and positive")
    max_iterations = _require_nonnegative_int(max_iterations, name="max_iterations")
    if max_iterations == 0:
        raise ValueError("max_iterations must be positive")
    estimate = table.delta
    if estimate >= 1:
        return 1.0
    z = float(norm.ppf(1 - alpha))

    def objective(candidate: float) -> float:
        return _tango_efficient_score(table, candidate) + z

    lower = estimate
    upper = np.nextafter(1.0, 0.0)
    if objective(upper) >= 0:
        return 1.0
    return float(brentq(objective, lower, upper, xtol=xtol, maxiter=max_iterations))


@dataclass(frozen=True)
class SimulatedPower:
    trials: int
    passes: int
    estimate: float
    lower_bound: float


def simulate_tango_noninferiority_power(
    *,
    shots: int,
    discordance_probability: float,
    delta_noninferiority: float,
    replicates: int,
    seed: int,
    decision_alpha: float = 0.025,
    power_bound_alpha: float = 0.05,
) -> SimulatedPower:
    """Deterministically simulate the frozen equality-design Tango power gate."""

    shots = _require_nonnegative_int(shots, name="shots")
    replicates = _require_nonnegative_int(replicates, name="replicates")
    if shots == 0 or replicates == 0:
        raise ValueError("shots and replicates must be positive")
    if not 0 <= discordance_probability <= 1 or not math.isfinite(discordance_probability):
        raise ValueError("discordance_probability must be finite and in [0, 1]")
    if not math.isfinite(delta_noninferiority):
        raise ValueError("delta_noninferiority must be finite")
    seed = _require_nonnegative_int(seed, name="seed")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        shots,
        [discordance_probability / 2, discordance_probability / 2, 1 - discordance_probability],
        size=replicates,
    )
    pairs, multiplicities = np.unique(draws[:, :2], axis=0, return_counts=True)
    passes = 0
    for (b, c), multiplicity in zip(pairs, multiplicities):
        table = PairedContingency(
            both_correct=shots - int(b) - int(c),
            regressions=int(b),
            recoveries=int(c),
            both_wrong=0,
        )
        if tango_paired_risk_difference_upper(table, alpha=decision_alpha) < delta_noninferiority:
            passes += int(multiplicity)
    return SimulatedPower(
        trials=replicates,
        passes=passes,
        estimate=passes / replicates,
        lower_bound=clopper_pearson_lower(
            successes=passes,
            trials=replicates,
            alpha=power_bound_alpha,
        ),
    )


def validate_process_count(processes: int, *, cap: int = ROUND_ONE_MAX_PROCESSES) -> int:
    """Reject accidental oversubscription; round one permits at most 32 processes."""

    processes = _require_nonnegative_int(processes, name="processes")
    cap = _require_nonnegative_int(cap, name="cap")
    if processes == 0 or cap == 0:
        raise ValueError("processes and cap must be positive")
    if cap > ROUND_ONE_MAX_PROCESSES:
        raise ValueError(
            f"cap={cap} exceeds the immutable round-one ceiling of "
            f"{ROUND_ONE_MAX_PROCESSES}"
        )
    if processes > cap:
        raise ValueError(f"processes={processes} exceeds the frozen cap of {cap}")
    return processes


def _seed_root_bytes(seed_root: str | bytes) -> bytes:
    if isinstance(seed_root, bytes):
        root = seed_root
    elif isinstance(seed_root, str):
        if len(seed_root) != 64:
            raise ValueError("seed_root must be exactly 64 hexadecimal characters")
        try:
            root = bytes.fromhex(seed_root)
        except ValueError as ex:
            raise ValueError("seed_root must be hexadecimal") from ex
    else:
        raise TypeError("seed_root must be bytes or a hexadecimal string")
    if len(root) != 32:
        raise ValueError("seed_root must contain exactly 256 bits")
    return root


def derive_stim_batch_seed(*, seed_root: str | bytes, batch_id: int) -> int:
    """Derive the Section 12.1 uint64 Stim seed for one immutable batch."""

    batch_id = _require_nonnegative_int(batch_id, name="batch_id")
    if batch_id >= 2**64:
        raise ValueError("batch_id does not fit in uint64")
    digest = hashlib.sha256(
        _seed_root_bytes(seed_root) + b"stim-batch" + struct.pack("<Q", batch_id)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@dataclass(frozen=True)
class ArrayDigest:
    sha256: str
    shape: tuple[int, ...]
    dtype: str


def digest_array(array: np.ndarray) -> ArrayDigest:
    """Digest logical C-order array bytes while recording shape and dtype."""

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays cannot be used in a replay ledger")
    contiguous = np.ascontiguousarray(value)
    return ArrayDigest(
        sha256=hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        shape=tuple(int(e) for e in contiguous.shape),
        dtype=contiguous.dtype.str,
    )


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    shot_start: int
    shots: int

    def __post_init__(self) -> None:
        for name in ("batch_id", "shot_start", "shots"):
            object.__setattr__(self, name, _require_nonnegative_int(getattr(self, name), name=name))
        if self.shots == 0:
            raise ValueError("a batch must contain at least one shot")

    @property
    def shot_stop(self) -> int:
        return self.shot_start + self.shots

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "BatchSpec":
        if not isinstance(value, Mapping):
            raise TypeError("batch entries must be mappings")
        expected = {"batch_id", "shot_start", "shots"}
        if set(value) != expected:
            raise ValueError(f"batch entry fields must be exactly {sorted(expected)}")
        return cls(**value)


def validate_batch_schedule(
    batches: Iterable[BatchSpec | Mapping[str, Any]],
    *,
    expected_shots: int | None = None,
    batch_size: int = ROUND_ONE_BATCH_SIZE,
) -> tuple[BatchSpec, ...]:
    """Validate unique IDs and an exact, gap-free shot-range partition."""

    batch_size = _require_nonnegative_int(batch_size, name="batch_size")
    if batch_size == 0:
        raise ValueError("batch_size must be positive")
    parsed = tuple(e if isinstance(e, BatchSpec) else BatchSpec.from_json(e) for e in batches)
    if not parsed:
        raise ValueError("batch schedule must not be empty")
    if len({e.batch_id for e in parsed}) != len(parsed):
        raise ValueError("batch schedule contains duplicate batch IDs")
    ordered = tuple(sorted(parsed, key=lambda e: e.shot_start))
    cursor = 0
    for index, batch in enumerate(ordered):
        if batch.shot_start != cursor:
            kind = "overlap" if batch.shot_start < cursor else "gap"
            raise ValueError(f"batch schedule contains a shot-range {kind} at {cursor}")
        if batch.shots > batch_size:
            raise ValueError("a batch exceeds the frozen batch size")
        if index + 1 < len(ordered) and batch.shots != batch_size:
            raise ValueError("only the final batch may be shorter than batch_size")
        cursor = batch.shot_stop
    if expected_shots is not None:
        expected_shots = _require_nonnegative_int(expected_shots, name="expected_shots")
        if cursor != expected_shots:
            raise ValueError(f"batch schedule covers {cursor} shots, expected {expected_shots}")
    return ordered


def _validate_json_tree(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite float at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string object key at {path}")
            _validate_json_tree(item, path=f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value {type(value).__name__} at {path}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize the experiment's documented canonical JSON representation."""

    if not isinstance(value, Mapping):
        raise TypeError("manifest must be a mapping")
    plain = dict(value)
    _validate_json_tree(plain)
    return json.dumps(
        plain,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_experiment_id(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest, excluding its optional self-referential experiment_id."""

    content = dict(manifest)
    content.pop("experiment_id", None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def validate_protocol_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_experiment_id: str | None = None,
    required_fields: Iterable[str] = (),
) -> str:
    """Validate reproducibility-critical generic protocol fields and return its ID."""

    canonical_json_bytes(manifest)
    missing = set(required_fields) - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing required fields: {sorted(missing)}")
    if "clean_worktree" in manifest and manifest["clean_worktree"] is not True:
        raise ValueError("scientific protocols require clean_worktree=true")
    if "sample_batch_size" in manifest and manifest["sample_batch_size"] != ROUND_ONE_BATCH_SIZE:
        raise ValueError(f"round one sample_batch_size must be {ROUND_ONE_BATCH_SIZE}")
    if "processes" in manifest:
        validate_process_count(manifest["processes"])
    roots = manifest.get("sampler_seed_roots")
    if roots is not None:
        if not isinstance(roots, Mapping) or not roots:
            raise ValueError("sampler_seed_roots must be a nonempty mapping")
        root_values = []
        for name, root in roots.items():
            if not isinstance(name, str):
                raise TypeError("sampler_seed_roots keys must be strings")
            root_values.append(_seed_root_bytes(root))
        if len(set(root_values)) != len(root_values):
            raise ValueError("sampler seed roots for distinct corpora must be unique")
    schedules = manifest.get("batch_schedules")
    if schedules is not None:
        if not isinstance(schedules, Mapping) or not schedules:
            raise ValueError("batch_schedules must be a nonempty mapping")
        expected_by_split = manifest.get("expected_shots", {})
        if expected_by_split is not None and not isinstance(expected_by_split, Mapping):
            raise TypeError("expected_shots must be a mapping")
        for split, entries in schedules.items():
            validate_batch_schedule(
                entries,
                expected_shots=expected_by_split.get(split) if expected_by_split else None,
                batch_size=manifest.get("sample_batch_size", ROUND_ONE_BATCH_SIZE),
            )
    actual_id = manifest_experiment_id(manifest)
    embedded_id = manifest.get("experiment_id")
    if embedded_id is not None and embedded_id != actual_id:
        raise ValueError("embedded experiment_id does not match canonical manifest hash")
    if expected_experiment_id is not None and expected_experiment_id != actual_id:
        raise ValueError("manifest hash does not match the expected experiment ID")
    return actual_id


def paired_workload_ratio(
    *, original_events: Sequence[int] | np.ndarray, residual_events: Sequence[int] | np.ndarray
) -> float:
    """Ratio of unconditional paired mean residual/original detector events."""

    original, residual = _paired_nonnegative_arrays(original_events, residual_events)
    denominator = float(np.sum(original, dtype=np.float64))
    if denominator == 0:
        raise ValueError("workload ratio is undefined when original mean workload is zero")
    return float(np.sum(residual, dtype=np.float64) / denominator)


def _paired_nonnegative_arrays(
    first: Sequence[int] | np.ndarray, second: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(first)
    b = np.asarray(second)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("paired arrays must be nonempty, one-dimensional, and equally sized")
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number):
        raise TypeError("paired arrays must be numeric")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("paired arrays must be finite")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("paired arrays must be nonnegative")
    return a, b


def empirical_type7_quantile(values: Sequence[float] | np.ndarray, probability: float) -> float:
    """R/NumPy type-7 empirical quantile, made explicit for frozen protocols."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or np.any(np.isnan(data)):
        raise ValueError("quantile input must be a nonempty one-dimensional array without NaN")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = np.sort(data)
    h = (len(ordered) - 1) * probability
    lower_index = math.floor(h)
    fraction = h - lower_index
    if fraction == 0 or lower_index + 1 == len(ordered):
        return float(ordered[lower_index])
    lower = float(ordered[lower_index])
    upper = float(ordered[lower_index + 1])
    if math.isinf(upper):
        return upper
    return lower + fraction * (upper - lower)


@dataclass(frozen=True)
class BootstrapRatio:
    estimate: float
    upper_bound: float
    replicates: int


def paired_workload_bootstrap(
    *,
    original_events: Sequence[int] | np.ndarray,
    residual_events: Sequence[int] | np.ndarray,
    replicates: int,
    seed: int,
    alpha: float = 0.025,
) -> BootstrapRatio:
    """Paired shot bootstrap for the unconditional workload ratio."""

    original, residual = _paired_nonnegative_arrays(original_events, residual_events)
    estimate = paired_workload_ratio(original_events=original, residual_events=residual)
    replicates = _require_nonnegative_int(replicates, name="replicates")
    seed = _require_nonnegative_int(seed, name="seed")
    if replicates == 0:
        raise ValueError("replicates must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=np.float64)
    for k in range(replicates):
        indices = rng.integers(0, len(original), size=len(original))
        denominator = float(np.sum(original[indices], dtype=np.float64))
        ratios[k] = (
            math.inf
            if denominator == 0
            else float(np.sum(residual[indices], dtype=np.float64) / denominator)
        )
    return BootstrapRatio(
        estimate=estimate,
        upper_bound=empirical_type7_quantile(ratios, 1 - alpha),
        replicates=replicates,
    )


def paired_workload_histogram_bootstrap(
    *,
    joint_counts: Mapping[tuple[int, int], int],
    replicates: int,
    seed: int,
    alpha: float = 0.025,
    chunk_size: int = 256,
) -> BootstrapRatio:
    """Exact paired-shot bootstrap from a joint workload histogram.

    A direct index bootstrap allocates work proportional to
    ``shots * replicates`` and is impractical for the million-shot target
    corpus.  Counts drawn from a multinomial over the unique
    ``(original_events, residual_events)`` pairs have exactly the same
    bootstrap distribution, while the cost depends on the number of occupied
    histogram cells.  Replicates are generated in bounded chunks.
    """

    if not isinstance(joint_counts, Mapping) or not joint_counts:
        raise ValueError("joint_counts must be a nonempty mapping")
    replicates = _require_nonnegative_int(replicates, name="replicates")
    seed = _require_nonnegative_int(seed, name="seed")
    chunk_size = _require_nonnegative_int(chunk_size, name="chunk_size")
    if replicates == 0 or chunk_size == 0:
        raise ValueError("replicates and chunk_size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    normalized: list[tuple[int, int, int]] = []
    for pair, raw_count in joint_counts.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("joint workload keys must be (original, residual) tuples")
        original = _require_nonnegative_int(pair[0], name="original_events")
        residual = _require_nonnegative_int(pair[1], name="residual_events")
        count = _require_nonnegative_int(raw_count, name="joint_count")
        if count == 0:
            raise ValueError("joint workload histogram counts must be positive")
        normalized.append((original, residual, count))
    normalized.sort()

    original_values = np.asarray([e[0] for e in normalized], dtype=np.float64)
    residual_values = np.asarray([e[1] for e in normalized], dtype=np.float64)
    counts = np.asarray([e[2] for e in normalized], dtype=np.int64)
    shots = int(np.sum(counts, dtype=np.int64))
    original_total = float(counts @ original_values)
    if original_total == 0:
        raise ValueError("workload ratio is undefined when original mean workload is zero")
    estimate = float((counts @ residual_values) / original_total)

    probabilities = counts.astype(np.float64) / shots
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=np.float64)
    offset = 0
    while offset < replicates:
        size = min(chunk_size, replicates - offset)
        sampled_counts = rng.multinomial(shots, probabilities, size=size)
        denominators = sampled_counts @ original_values
        numerators = sampled_counts @ residual_values
        ratios[offset : offset + size] = np.divide(
            numerators,
            denominators,
            out=np.full(size, math.inf, dtype=np.float64),
            where=denominators != 0,
        )
        offset += size
    return BootstrapRatio(
        estimate=estimate,
        upper_bound=empirical_type7_quantile(ratios, 1 - alpha),
        replicates=replicates,
    )


def paired_geometric_mean_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Geometric mean of positive paired block-duration ratios."""

    a, b = _paired_nonnegative_arrays(numerator, denominator)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("timing durations must be strictly positive")
    return float(math.exp(np.mean(np.log(a) - np.log(b))))


@dataclass(frozen=True)
class HierarchicalTimingBootstrap:
    geometric_ratio: float
    geometric_ratio_upper: float
    p99_ratio: float
    p99_ratio_upper: float
    replicates: int


def hierarchical_timing_bootstrap(
    *,
    numerator_calls: np.ndarray,
    denominator_calls: np.ndarray,
    replicates: int,
    seed: int,
    alpha: float = 0.025,
) -> HierarchicalTimingBootstrap:
    """Restart/block cluster bootstrap from Section 17.

    Inputs have shape ``(restarts, blocks, calls_per_block)``.  Restart indices
    and then paired block indices are resampled; all calls in a block stay
    together.
    """

    a = np.asarray(numerator_calls, dtype=np.float64)
    b = np.asarray(denominator_calls, dtype=np.float64)
    if a.ndim != 3 or a.shape != b.shape or 0 in a.shape:
        raise ValueError("timing arrays must have identical nonempty 3D shapes")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("timing calls must be finite and strictly positive")
    replicates = _require_nonnegative_int(replicates, name="replicates")
    seed = _require_nonnegative_int(seed, name="seed")
    if replicates == 0:
        raise ValueError("replicates must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    block_a = np.sum(a, axis=2)
    block_b = np.sum(b, axis=2)
    estimate_g = paired_geometric_mean_ratio(block_a.ravel(), block_b.ravel())
    estimate_p99 = empirical_type7_quantile(a.ravel(), 0.99) / empirical_type7_quantile(
        b.ravel(), 0.99
    )
    rng = np.random.default_rng(seed)
    g_values = np.empty(replicates)
    p99_values = np.empty(replicates)
    restarts, blocks, _ = a.shape
    for k in range(replicates):
        selected_a = []
        selected_b = []
        for source_restart in rng.integers(0, restarts, size=restarts):
            block_ids = rng.integers(0, blocks, size=blocks)
            selected_a.append(a[source_restart, block_ids, :])
            selected_b.append(b[source_restart, block_ids, :])
        sample_a = np.stack(selected_a)
        sample_b = np.stack(selected_b)
        g_values[k] = paired_geometric_mean_ratio(
            np.sum(sample_a, axis=2).ravel(), np.sum(sample_b, axis=2).ravel()
        )
        p99_values[k] = empirical_type7_quantile(
            sample_a.ravel(), 0.99
        ) / empirical_type7_quantile(sample_b.ravel(), 0.99)
    return HierarchicalTimingBootstrap(
        geometric_ratio=estimate_g,
        geometric_ratio_upper=empirical_type7_quantile(g_values, 1 - alpha),
        p99_ratio=estimate_p99,
        p99_ratio_upper=empirical_type7_quantile(p99_values, 1 - alpha),
        replicates=replicates,
    )
