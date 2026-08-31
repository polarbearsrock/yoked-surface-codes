"""Additive analysis for the matched four-arm decoder experiment.

The analyzer consumes only a validated matched-accuracy aggregate.  It never
replays detector samples or invokes a decoder.  Accuracy inference is paired
on the common authenticated shot corpus, while each workload bootstrap keeps
the per-shot pairing between original and residual detector-event counts.

The aggregate currently does not retain the *cross-arm* joint distribution of
residual workloads.  Exact cross-arm workload point differences are therefore
reported, but paired confidence intervals for those differences are explicitly
marked unavailable instead of manufacturing independence between arms.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from typing import Any, Mapping

import numpy as np
from scipy.stats import binomtest

from yoked.decoding._pinball_promatch_matched_accuracy import (
    ARM_ORDER,
    PAIR_DEFINITIONS,
    MatchedCorpus,
    validate_matched_aggregate,
)
from yoked.decoding._promatch_stats import (
    PairedContingency,
    canonical_json_bytes,
    clopper_pearson_lower,
    clopper_pearson_upper,
    empirical_type7_quantile,
    tango_paired_risk_difference_upper,
)


ANALYSIS_SCHEMA = "yoked.pinball-promatch-uf-matched-analysis-v1"
CONFIDENCE_LEVEL = 0.95
_TWO_SIDED_ALPHA = 1.0 - CONFIDENCE_LEVEL

ARM_STANDARD_NAMES = {
    "global": "Global MWPM baseline",
    "promatch": "ProMatch-assisted MWPM",
    "pinball": "Pinball-assisted MWPM",
    "union_find": "Union-Find-assisted MWPM",
}


@dataclasses.dataclass(frozen=True)
class MatchedAnalysisConfig:
    """Frozen randomized inputs for the paired workload bootstrap."""

    workload_bootstrap_replicates: int
    workload_bootstrap_seed: int
    workload_bootstrap_chunk_size: int = 256

    def __post_init__(self) -> None:
        for name in (
            "workload_bootstrap_replicates",
            "workload_bootstrap_seed",
            "workload_bootstrap_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.workload_bootstrap_replicates == 0:
            raise ValueError("workload_bootstrap_replicates must be positive")
        if self.workload_bootstrap_chunk_size == 0:
            raise ValueError("workload_bootstrap_chunk_size must be positive")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _derived_bootstrap_seed(master_seed: int, *, arm: str) -> int:
    material = canonical_json_bytes(
        {
            "arm": arm,
            "master_seed": master_seed,
            "purpose": "matched-original-residual-workload-bootstrap-v1",
        }
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _parse_histogram(value: Any, *, name: str) -> dict[int, int]:
    raw = _mapping(value, name=name)
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    result: dict[int, int] = {}
    for encoded, count_value in raw.items():
        try:
            key = int(encoded)
        except ValueError as ex:
            raise ValueError(f"{name} has a non-integer key {encoded!r}") from ex
        if key < 0 or str(key) != encoded:
            raise ValueError(f"{name} key {encoded!r} is not canonical nonnegative decimal")
        count = _nonnegative_int(count_value, name=f"{name}[{encoded!r}]")
        if count == 0:
            raise ValueError(f"{name} counts must be positive")
        result[key] = count
    return dict(sorted(result.items()))


def _parse_joint_histogram(
    value: Any, *, name: str
) -> dict[tuple[int, int], int]:
    raw = _mapping(value, name=name)
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    result: dict[tuple[int, int], int] = {}
    for encoded, count_value in raw.items():
        pieces = encoded.split(",")
        if len(pieces) != 2:
            raise ValueError(
                f"{name} key {encoded!r} must be 'original,residual'"
            )
        try:
            original, residual = (int(piece) for piece in pieces)
        except ValueError as ex:
            raise ValueError(f"{name} key {encoded!r} is not integral") from ex
        if (
            original < 0
            or residual < 0
            or pieces != [str(original), str(residual)]
        ):
            raise ValueError(
                f"{name} key {encoded!r} is not canonical nonnegative decimal"
            )
        count = _nonnegative_int(count_value, name=f"{name}[{encoded!r}]")
        if count == 0:
            raise ValueError(f"{name} counts must be positive")
        pair = (original, residual)
        if pair in result:
            raise ValueError(f"{name} contains a duplicate normalized key")
        result[pair] = count
    return dict(sorted(result.items()))


def _histogram_json(histogram: Mapping[int, int]) -> dict[str, int]:
    return {str(key): int(histogram[key]) for key in sorted(histogram)}


def _joint_histogram_json(
    histogram: Mapping[tuple[int, int], int]
) -> dict[str, int]:
    return {
        f"{original},{residual}": int(histogram[(original, residual)])
        for original, residual in sorted(histogram)
    }


def _marginals_from_joint(
    histogram: Mapping[tuple[int, int], int]
) -> tuple[dict[int, int], dict[int, int]]:
    original: dict[int, int] = {}
    residual: dict[int, int] = {}
    for (before, after), count in histogram.items():
        original[before] = original.get(before, 0) + count
        residual[after] = residual.get(after, 0) + count
    return dict(sorted(original.items())), dict(sorted(residual.items()))


def _bootstrap_workload_ratio(
    histogram: Mapping[tuple[int, int], int],
    *,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Two-sided percentile interval from an exact histogram shot bootstrap."""

    rows = [(before, after, count) for (before, after), count in sorted(histogram.items())]
    before_values = np.asarray([row[0] for row in rows], dtype=np.float64)
    after_values = np.asarray([row[1] for row in rows], dtype=np.float64)
    counts = np.asarray([row[2] for row in rows], dtype=np.int64)
    shots = int(np.sum(counts, dtype=np.int64))
    # Keep the observed totals exact; the float arrays below are used only for
    # the bootstrap's bounded matrix operations.
    original_total = sum(before * count for before, _, count in rows)
    residual_total = sum(after * count for _, after, count in rows)
    if original_total == 0:
        raise ValueError(
            "workload ratio is undefined because the original detector-event total is zero"
        )
    estimate = residual_total / original_total
    probabilities = counts.astype(np.float64) / shots
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    offset = 0
    while offset < replicates:
        size = min(chunk_size, replicates - offset)
        sampled_counts = rng.multinomial(shots, probabilities, size=size)
        denominators = sampled_counts @ before_values
        numerators = sampled_counts @ after_values
        samples[offset : offset + size] = np.divide(
            numerators,
            denominators,
            out=np.full(size, math.inf, dtype=np.float64),
            where=denominators != 0,
        )
        offset += size
    lower = empirical_type7_quantile(samples, _TWO_SIDED_ALPHA / 2)
    upper = empirical_type7_quantile(samples, 1 - _TWO_SIDED_ALPHA / 2)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(
            "paired workload bootstrap interval is non-finite because too many "
            "resamples contain no original detector events"
        )
    return {
        "method": (
            "paired-shot multinomial bootstrap from joint histogram; "
            "type-7 percentile interval"
        ),
        "confidence_level": CONFIDENCE_LEVEL,
        "replicates": replicates,
        "seed": seed,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
    }


def _clopper_pearson_interval(*, failures: int, shots: int) -> dict[str, Any]:
    lower = clopper_pearson_lower(
        successes=failures, trials=shots, alpha=_TWO_SIDED_ALPHA / 2
    )
    upper = clopper_pearson_upper(
        successes=failures, trials=shots, alpha=_TWO_SIDED_ALPHA / 2
    )
    return {
        "method": "two-sided exact Clopper-Pearson",
        "confidence_level": CONFIDENCE_LEVEL,
        "lower": lower,
        "upper": upper,
        "percentage": {"lower": 100 * lower, "upper": 100 * upper},
    }


def _tango_interval(table: PairedContingency) -> dict[str, Any]:
    swapped = PairedContingency(
        both_correct=table.both_correct,
        regressions=table.recoveries,
        recoveries=table.regressions,
        both_wrong=table.both_wrong,
    )
    lower = -tango_paired_risk_difference_upper(
        swapped, alpha=_TWO_SIDED_ALPHA / 2
    )
    upper = tango_paired_risk_difference_upper(
        table, alpha=_TWO_SIDED_ALPHA / 2
    )
    return {
        "method": "two-sided Tango efficient-score",
        "confidence_level": CONFIDENCE_LEVEL,
        "lower": lower,
        "upper": upper,
        "percentage_points": {"lower": 100 * lower, "upper": 100 * upper},
    }


def _exact_two_sided_mcnemar(table: PairedContingency) -> float:
    if table.discordant == 0:
        return 1.0
    return float(
        binomtest(
            table.regressions,
            table.discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def _workload_arm(
    *,
    arm: str,
    branch: Mapping[str, Any],
    common_original_histogram: Mapping[int, int],
    common_original_total: int,
    shots: int,
    config: MatchedAnalysisConfig,
) -> dict[str, Any]:
    key = "original_residual_hw_joint_histogram"
    if key not in branch:
        raise ValueError(
            f"telemetry.{arm}.{key} is required for paired workload analysis"
        )
    joint = _parse_joint_histogram(
        branch[key], name=f"telemetry.{arm}.{key}"
    )
    if sum(joint.values()) != shots:
        raise ValueError(
            f"telemetry.{arm}.{key} must contain exactly one entry per common shot"
        )
    original_histogram, residual_histogram = _marginals_from_joint(joint)
    if original_histogram != dict(common_original_histogram):
        raise ValueError(
            f"telemetry.{arm}.{key} original marginal differs from the common shot corpus"
        )
    original_total = sum(value * count for value, count in original_histogram.items())
    residual_total = sum(value * count for value, count in residual_histogram.items())
    if original_total != common_original_total:
        raise ValueError(f"telemetry.{arm} original detector-event total differs")
    recorded_residual_total = _nonnegative_int(
        branch.get("residual_event_sum"),
        name=f"telemetry.{arm}.residual_event_sum",
    )
    if residual_total != recorded_residual_total:
        raise ValueError(
            f"telemetry.{arm}.{key} residual total differs from residual_event_sum"
        )
    if "residual_hw_histogram" not in branch:
        raise ValueError(f"telemetry.{arm}.residual_hw_histogram is required")
    recorded_residual_histogram = _parse_histogram(
        branch["residual_hw_histogram"],
        name=f"telemetry.{arm}.residual_hw_histogram",
    )
    if residual_histogram != recorded_residual_histogram:
        raise ValueError(
            f"telemetry.{arm}.{key} residual marginal differs from residual_hw_histogram"
        )
    if arm == "global" and any(before != after for before, after in joint):
        raise ValueError(
            "Global MWPM baseline workload must leave the original detector-event count unchanged"
        )
    derived_seed = _derived_bootstrap_seed(
        config.workload_bootstrap_seed, arm=arm
    )
    interval = _bootstrap_workload_ratio(
        joint,
        replicates=config.workload_bootstrap_replicates,
        seed=derived_seed,
        chunk_size=config.workload_bootstrap_chunk_size,
    )
    ratio = residual_total / original_total
    reduction = 1 - ratio
    return {
        "arm": arm,
        "standard_name": ARM_STANDARD_NAMES[arm],
        "shots": shots,
        "original_detector_events": original_total,
        "residual_detector_events": residual_total,
        "residual_over_original_ratio": ratio,
        "detector_event_reduction_fraction": reduction,
        "detector_event_reduction_percentage": 100 * reduction,
        "residual_over_original_ratio_confidence_interval": interval,
        "detector_event_reduction_confidence_interval": {
            "method": interval["method"],
            "confidence_level": CONFIDENCE_LEVEL,
            "lower": 1 - interval["upper"],
            "upper": 1 - interval["lower"],
            "percentage": {
                "lower": 100 * (1 - interval["upper"]),
                "upper": 100 * (1 - interval["lower"]),
            },
        },
        "original_hw_histogram": _histogram_json(original_histogram),
        "residual_hw_histogram": _histogram_json(residual_histogram),
        "original_residual_hw_joint_histogram": _joint_histogram_json(joint),
    }


def analyze_matched_aggregate(
    aggregate: Mapping[str, Any],
    *,
    corpus: MatchedCorpus,
    config: MatchedAnalysisConfig,
    expected_prepared_provenance: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Analyze one authenticated, additive, four-arm matched aggregate."""

    if not isinstance(config, MatchedAnalysisConfig):
        raise TypeError("config must be a MatchedAnalysisConfig")
    validate_matched_aggregate(
        aggregate,
        corpus=corpus,
        expected_prepared_provenance=expected_prepared_provenance,
        require_complete=require_complete,
    )
    raw = _mapping(aggregate, name="matched aggregate")
    shots = _nonnegative_int(raw.get("shots"), name="shots")
    if shots == 0:
        raise ValueError("shots must be positive")
    if raw.get("arm_order") != list(ARM_ORDER):
        raise ValueError("matched aggregate arm order differs")

    cube = _mapping(raw.get("correctness_cube"), name="correctness_cube")
    marginals: dict[str, Any] = {}
    for index, arm in enumerate(ARM_ORDER):
        failures = sum(
            _nonnegative_int(count, name=f"correctness_cube.{bits}")
            for bits, count in cube.items()
            if bits[index] == "1"
        )
        rate = failures / shots
        marginals[arm] = {
            "arm": arm,
            "standard_name": ARM_STANDARD_NAMES[arm],
            "failures": failures,
            "shots": shots,
            "any_observable_logical_error_rate": rate,
            "any_observable_logical_error_percentage": 100 * rate,
            "confidence_interval": _clopper_pearson_interval(
                failures=failures, shots=shots
            ),
        }

    tables = _mapping(
        raw.get("pairwise_contingencies"), name="pairwise_contingencies"
    )
    agreements = _mapping(
        raw.get("prediction_agreement"), name="prediction_agreement"
    )
    pairwise: dict[str, Any] = {}
    for pair, (baseline, treatment) in PAIR_DEFINITIONS.items():
        table_raw = _mapping(tables.get(pair), name=f"pairwise_contingencies.{pair}")
        table = PairedContingency(
            both_correct=table_raw.get("both_correct"),
            regressions=table_raw.get("regressions"),
            recoveries=table_raw.get("recoveries"),
            both_wrong=table_raw.get("both_wrong"),
        )
        if table.shots != shots:
            raise ValueError(f"pairwise_contingencies.{pair} does not reconcile")
        agreement_raw = _mapping(
            agreements.get(pair), name=f"prediction_agreement.{pair}"
        )
        agree = _nonnegative_int(
            agreement_raw.get("agree"), name=f"prediction_agreement.{pair}.agree"
        )
        disagree = _nonnegative_int(
            agreement_raw.get("disagree"),
            name=f"prediction_agreement.{pair}.disagree",
        )
        if agree + disagree != shots:
            raise ValueError(f"prediction_agreement.{pair} does not reconcile")
        numerator = table.regressions - table.recoveries
        pairwise[pair] = {
            "baseline": baseline,
            "baseline_standard_name": ARM_STANDARD_NAMES[baseline],
            "treatment": treatment,
            "treatment_standard_name": ARM_STANDARD_NAMES[treatment],
            "contingency": dataclasses.asdict(table),
            "discordant_shots": table.discordant,
            "failure_risk_difference": {
                "orientation": "treatment minus baseline",
                "fraction": {
                    "numerator": numerator,
                    "denominator": shots,
                    "text": f"{numerator}/{shots}",
                },
                "value": table.delta,
                "percentage_points": 100 * table.delta,
            },
            "tango_confidence_interval": _tango_interval(table),
            "exact_two_sided_mcnemar_p_value": _exact_two_sided_mcnemar(table),
            "prediction_agreement": {
                "agree": agree,
                "disagree": disagree,
                "fraction": agree / shots,
                "percentage": 100 * agree / shots,
            },
        }

    telemetry = _mapping(raw.get("telemetry"), name="telemetry")
    common = _mapping(telemetry.get("common"), name="telemetry.common")
    original_total = _nonnegative_int(
        common.get("original_event_sum"),
        name="telemetry.common.original_event_sum",
    )
    if original_total == 0:
        raise ValueError(
            "common original detector-event total is zero; workload ratios are undefined"
        )
    original_histogram = _parse_histogram(
        common.get("original_hw_histogram"),
        name="telemetry.common.original_hw_histogram",
    )
    if sum(original_histogram.values()) != shots:
        raise ValueError("common original workload histogram does not reconcile to shots")
    if sum(value * count for value, count in original_histogram.items()) != original_total:
        raise ValueError("common original workload histogram does not reconcile to total")

    workload_arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        branch = _mapping(telemetry.get(arm), name=f"telemetry.{arm}")
        workload_arms[arm] = _workload_arm(
            arm=arm,
            branch=branch,
            common_original_histogram=original_histogram,
            common_original_total=original_total,
            shots=shots,
            config=config,
        )

    workload_differences: dict[str, Any] = {}
    for pair, (baseline, treatment) in PAIR_DEFINITIONS.items():
        baseline_residual = workload_arms[baseline]["residual_detector_events"]
        treatment_residual = workload_arms[treatment]["residual_detector_events"]
        numerator = treatment_residual - baseline_residual
        estimate = numerator / original_total
        workload_differences[pair] = {
            "baseline": baseline,
            "baseline_standard_name": ARM_STANDARD_NAMES[baseline],
            "treatment": treatment,
            "treatment_standard_name": ARM_STANDARD_NAMES[treatment],
            "residual_over_original_ratio_difference": {
                "orientation": "treatment minus baseline",
                "fraction": {
                    "numerator": numerator,
                    "denominator": original_total,
                    "text": f"{numerator}/{original_total}",
                },
                "value": estimate,
                "percentage_points": 100 * estimate,
            },
            "paired_confidence_interval": None,
            "inference_status": "point estimate only",
        }

    result: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "source_aggregate_payload_sha256": raw.get("payload_sha256"),
        "source_identity": copy.deepcopy(
            dict(_mapping(raw.get("source_identity"), name="source_identity"))
        ),
        "prepared_provenance": copy.deepcopy(
            dict(_mapping(raw.get("prepared_provenance"), name="prepared_provenance"))
        ),
        "shots": shots,
        "arm_order": list(ARM_ORDER),
        "arm_standard_names": dict(ARM_STANDARD_NAMES),
        "confidence_level": CONFIDENCE_LEVEL,
        "analysis_config": dataclasses.asdict(config),
        "accuracy": {
            "correctness_cube": copy.deepcopy(dict(cube)),
            "marginals": marginals,
            "pairwise": pairwise,
        },
        "workload": {
            "definition": "detector events presented to the residual MWPM stage",
            "common_original_detector_events": original_total,
            "common_original_hw_histogram": _histogram_json(original_histogram),
            "arms": workload_arms,
            "pairwise_ratio_differences": workload_differences,
            "cross_arm_paired_inference": {
                "available": False,
                "reason": (
                    "the matched aggregate retains each arm's original/residual "
                    "joint histogram but not a common shot-level joint histogram "
                    "of residual workloads across different arms"
                ),
            },
        },
        "raw_inputs": {
            "pairwise_contingencies": copy.deepcopy(dict(tables)),
            "prediction_agreement": copy.deepcopy(dict(agreements)),
            "telemetry": copy.deepcopy(dict(telemetry)),
        },
        "caveats": [
            "Accuracy comparisons use the same authenticated shots and are paired.",
            (
                "Each workload interval preserves original-versus-residual pairing "
                "within an arm; cross-arm paired workload intervals are unavailable "
                "without a joint residual-workload histogram across arms."
            ),
            (
                "Detector-event workload is the input size presented to residual "
                "MWPM; it is not a latency measurement."
            ),
            "Native frontend telemetry is retained verbatim under raw_inputs.telemetry.",
        ],
    }
    result["payload_sha256"] = _digest_json(result)
    return result


def _validated_analysis(value: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _mapping(value, name="matched analysis")
    if raw.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("matched analysis schema differs")
    payload = dict(raw)
    digest = payload.pop("payload_sha256", None)
    if digest != _digest_json(payload):
        raise ValueError("matched analysis payload digest mismatch")
    return raw


def _format_interval(lower: float, upper: float, *, scale: float = 1.0) -> str:
    return f"[{scale * lower:.4g}, {scale * upper:.4g}]"


def render_matched_analysis_markdown(value: Mapping[str, Any]) -> str:
    """Render a compact human-readable report with standard arm names."""

    analysis = _validated_analysis(value)
    accuracy = _mapping(analysis.get("accuracy"), name="accuracy")
    marginals = _mapping(accuracy.get("marginals"), name="accuracy.marginals")
    pairwise = _mapping(accuracy.get("pairwise"), name="accuracy.pairwise")
    workload = _mapping(analysis.get("workload"), name="workload")
    workload_arms = _mapping(workload.get("arms"), name="workload.arms")
    shots = analysis["shots"]
    lines = [
        "# Matched four-arm decoder analysis",
        "",
        f"Common authenticated corpus: **{shots:,} shots**.",
        "",
        "## Accuracy",
        "",
        "| Decoder arm | Logical failures | Logical error rate | Exact 95% confidence interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARM_ORDER:
        item = marginals[arm]
        interval = item["confidence_interval"]
        lines.append(
            f"| {ARM_STANDARD_NAMES[arm]} | {item['failures']:,}/{shots:,} | "
            f"{item['any_observable_logical_error_percentage']:.4g}% | "
            f"{_format_interval(interval['lower'], interval['upper'], scale=100)}% |"
        )
    lines.extend(
        [
            "",
            "## Paired accuracy differences",
            "",
            (
                "Differences are treatment minus baseline. Positive values mean the "
                "treatment produced more logical failures."
            ),
            "",
            (
                "| Treatment minus baseline | Exact difference fraction | "
                "Difference (percentage points) | Tango 95% interval "
                "(percentage points) | Exact two-sided McNemar p-value | "
                "Prediction agreement |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in PAIR_DEFINITIONS:
        item = pairwise[pair]
        difference = item["failure_risk_difference"]
        interval = item["tango_confidence_interval"]
        agreement = item["prediction_agreement"]
        lines.append(
            f"| {item['treatment_standard_name']} minus {item['baseline_standard_name']} | "
            f"{difference['fraction']['text']} | {difference['percentage_points']:.4g} | "
            f"{_format_interval(interval['lower'], interval['upper'], scale=100)} | "
            f"{item['exact_two_sided_mcnemar_p_value']:.4g} | "
            f"{agreement['agree']:,}/{shots:,} ({agreement['percentage']:.4g}%) |"
        )
    lines.extend(
        [
            "",
            "## Residual matching workload",
            "",
            (
                "| Decoder arm | Original detector events | Residual detector "
                "events | Residual/original ratio | Reduction | "
                "Paired-bootstrap 95% ratio interval |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARM_ORDER:
        item = workload_arms[arm]
        interval = item["residual_over_original_ratio_confidence_interval"]
        lines.append(
            f"| {ARM_STANDARD_NAMES[arm]} | {item['original_detector_events']:,} | "
            f"{item['residual_detector_events']:,} | "
            f"{item['residual_over_original_ratio']:.4g} | "
            f"{item['detector_event_reduction_percentage']:.4g}% | "
            f"{_format_interval(interval['lower'], interval['upper'])} |"
        )
    lines.extend(["", "## Interpretation caveats", ""])
    for caveat in analysis["caveats"]:
        lines.append(f"- {caveat}")
        lines.append("")
    lines.append(f"Analysis payload SHA-256: `{analysis['payload_sha256']}`")
    return "\n".join(lines) + "\n"


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARM_STANDARD_NAMES",
    "CONFIDENCE_LEVEL",
    "MatchedAnalysisConfig",
    "analyze_matched_aggregate",
    "render_matched_analysis_markdown",
]
