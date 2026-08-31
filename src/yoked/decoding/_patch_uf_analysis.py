"""Deterministic characterization analysis for patch-UF paired collections.

The collection verifier remains the authentication boundary.  This module
then independently reconciles every shot/lane/component identity before
calling the maintained statistics routines.  Its output is deliberately
plot-free: canonical ``analysis.json`` data and a deterministic Markdown
report are the stable first-run artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from yoked.decoding._artifact_io import install_bytes_atomic
from yoked.decoding._patch_uf_experiment import (
    RANGE_COUNT,
    VerifiedCollection,
    verify_collection,
)
from yoked.decoding._patch_uf_stats import (
    ClusterSizeBin,
    bootstrap_cluster_sizes,
    bootstrap_workload_coverage,
    summarize_cluster_sizes,
    summarize_paired_accuracy,
    summarize_workload_coverage,
    validate_hrlk_histogram,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


ANALYSIS_SCHEMA = "patch-uf-analysis-v1"
ANALYSIS_SCHEMA_VERSION = 1
CASEBOOK_CATEGORIES = (
    "regression",
    "recovery",
    "prediction-disagreement",
    "port-yoke-defer",
    "port-cross-lane-defer",
    "threshold-tie",
    "local-incomplete-neutralization-patch-abort",
    "budget-exhaustion-patch-abort",
    "boundary-using-commit",
    "largest-final-component",
    "largest-committed-component",
    "largest-censored-partial-lower-bound",
    "highest-heap-operation-count",
)
_METRIC_CATEGORIES = {
    "largest-final-component",
    "largest-committed-component",
    "largest-censored-partial-lower-bound",
    "highest-heap-operation-count",
}

_DEFAULT_CONFIDENCE_BIN_EDGES: tuple[int | float | str, ...] = (
    0,
    0.0625,
    0.125,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    16,
    "+inf",
)


__all__ = [
    "ANALYSIS_SCHEMA",
    "ANALYSIS_SCHEMA_VERSION",
    "AnalysisArtifacts",
    "AnalysisConfig",
    "CASEBOOK_CATEGORIES",
    "analyze_collection",
    "analyze_verified_collection",
    "write_analysis_bundle",
]


def _default_bins() -> tuple[ClusterSizeBin, ...]:
    return (
        ClusterSizeBin("1", 1, 2),
        ClusterSizeBin("2", 2, 3),
        ClusterSizeBin("3-4", 3, 5),
        ClusterSizeBin("5-8", 5, 9),
        ClusterSizeBin("9+", 9, None),
    )


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """Frozen randomized-analysis and bounded-selection literals."""

    alpha: float = 0.05
    workload_bootstrap_replicates: int = 10_000
    cluster_bootstrap_replicates: int = 10_000
    workload_bootstrap_seed: int = 0
    cluster_bootstrap_seed: int = 1
    casebook_seed_root: str = "00" * 32
    maximum_cases_per_category: int = 100
    cluster_bins: tuple[ClusterSizeBin, ...] = dataclasses.field(
        default_factory=_default_bins
    )
    confidence_bin_edges: tuple[int | float | str, ...] = (
        _DEFAULT_CONFIDENCE_BIN_EDGES
    )

    def __post_init__(self) -> None:
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool):
            raise TypeError("alpha must be numeric")
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        for name in (
            "workload_bootstrap_replicates",
            "cluster_bootstrap_replicates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workload_bootstrap_seed", "cluster_bootstrap_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            not isinstance(self.maximum_cases_per_category, int)
            or isinstance(self.maximum_cases_per_category, bool)
            or not 0 <= self.maximum_cases_per_category <= 100
        ):
            raise ValueError("maximum_cases_per_category must be in [0, 100]")
        if (
            not isinstance(self.casebook_seed_root, str)
            or len(self.casebook_seed_root) != 64
        ):
            raise ValueError("casebook_seed_root must be 64 lowercase hex characters")
        try:
            bytes.fromhex(self.casebook_seed_root)
        except ValueError as ex:
            raise ValueError("casebook_seed_root must be hexadecimal") from ex
        if self.casebook_seed_root != self.casebook_seed_root.lower():
            raise ValueError("casebook_seed_root must be lowercase")
        bins = tuple(self.cluster_bins)
        if any(not isinstance(item, ClusterSizeBin) for item in bins):
            raise TypeError("cluster_bins must contain ClusterSizeBin values")
        confidence_edges = tuple(self.confidence_bin_edges)
        if len(confidence_edges) < 2 or confidence_edges[-1] != "+inf":
            raise ValueError("confidence_bin_edges must end in +inf")
        finite_edges = tuple(
            _protocol_edge_fraction(value, name="confidence bin edge")
            for value in confidence_edges[:-1]
        )
        if finite_edges[0] != 0 or any(
            left >= right for left, right in zip(finite_edges, finite_edges[1:])
        ):
            raise ValueError(
                "confidence_bin_edges must start at zero and be strictly increasing"
            )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "cluster_bins", bins)
        object.__setattr__(self, "confidence_bin_edges", confidence_edges)


@dataclasses.dataclass(frozen=True)
class AnalysisArtifacts:
    """Canonical JSON data and deterministic Markdown report."""

    analysis: Mapping[str, Any]
    report_markdown: str
    analysis_bytes: bytes
    report_bytes: bytes


def _protocol_edge_fraction(value: Any, *, name: str) -> Fraction:
    """Converts a frozen JSON number to its exact binary64 rational value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer or finite binary64 number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value)
    return Fraction(value)


def _exact_fraction(value: Any, *, name: str) -> Fraction:
    """Decodes only exact rational telemetry representations."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be exact rational telemetry")
    if isinstance(value, int):
        return Fraction(value)
    raw = _mapping(value, name=name)
    if set(raw) == {"numerator", "denominator"}:
        numerator = raw["numerator"]
        denominator = raw["denominator"]
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0
        ):
            raise ValueError(f"{name} fraction fields are malformed")
        return Fraction(numerator, denominator)
    if set(raw) in ({"mantissa", "exponent"}, {"integer", "binary_exponent"}):
        if "mantissa" in raw:
            integer = raw["mantissa"]
            exponent = raw["exponent"]
        else:
            integer = raw["integer"]
            exponent = raw["binary_exponent"]
        if (
            isinstance(integer, bool)
            or not isinstance(integer, int)
            or isinstance(exponent, bool)
            or not isinstance(exponent, int)
        ):
            raise ValueError(f"{name} dyadic fields are malformed")
        return (
            Fraction(integer << exponent)
            if exponent >= 0
            else Fraction(integer, 1 << -exponent)
        )
    raise ValueError(f"{name} has an unsupported exact representation")


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    numerator = _integer(numerator, name="rate numerator")
    denominator = _integer(denominator, name="rate denominator")
    if numerator > denominator:
        raise ValueError("rate numerator exceeds denominator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
        "status": "not-estimable" if denominator == 0 else "estimated",
    }


def _paired_outcome(
    *, global_failed: bool, treatment_failed: bool
) -> str:
    return (
        "d"
        if global_failed and treatment_failed
        else "c"
        if global_failed
        else "b"
        if treatment_failed
        else "a"
    )


def _outcome_summary(counter: Mapping[str, int]) -> dict[str, Any]:
    values = {
        name: _integer(counter.get(name, 0), name=f"outcome {name}")
        for name in "abcd"
    }
    denominator = sum(values.values())
    return {
        **values,
        "denominator": denominator,
        "global_errors": values["c"] + values["d"],
        "treatment_errors": values["b"] + values["d"],
        "regressions": values["b"],
        "recoveries": values["c"],
    }


def _decode_packed_hex(value: Any, *, bits: int, name: str) -> bytes:
    width = (bits + 7) // 8
    if (
        not isinstance(value, str)
        or value.lower() != value
        or len(value) != 2 * width
    ):
        raise ValueError(f"{name} must be canonical packed lowercase hex")
    try:
        result = bytes.fromhex(value)
    except ValueError as ex:
        raise ValueError(f"{name} is not hexadecimal") from ex
    if bits % 8 and result and result[-1] >> (bits % 8):
        raise ValueError(f"{name} has nonzero unused tail bits")
    return result


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("packed observable widths differ")
    return bytes(a ^ b for a, b in zip(left, right))


def _bit(value: bytes, index: int) -> bool:
    return bool(value[index // 8] & (1 << (index % 8)))


def _confidence_bin_index(
    value: Fraction | str, config: AnalysisConfig
) -> int | str:
    finite = tuple(
        _protocol_edge_fraction(edge, name="confidence bin edge")
        for edge in config.confidence_bin_edges[:-1]
    )
    if value == "infinity":
        return "infinity"
    if not isinstance(value, Fraction):
        raise TypeError("confidence value must be an exact fraction or +inf")
    if value < finite[0]:
        return -1
    for index, (lower, upper) in enumerate(zip(finite, finite[1:])):
        if lower <= value < upper:
            return index
    return len(finite) - 1


def _confidence_bin_rows(config: AnalysisConfig) -> list[dict[str, Any]]:
    finite = tuple(
        _protocol_edge_fraction(edge, name="confidence bin edge")
        for edge in config.confidence_bin_edges[:-1]
    )
    rows: list[dict[str, Any]] = [
        {
            "bin_index": -1,
            "lower": "-inf",
            "upper": _jsonable(config.confidence_bin_edges[0]),
            "lower_inclusive": False,
            "upper_inclusive": False,
        }
    ]
    for index, lower in enumerate(finite):
        upper: Any = config.confidence_bin_edges[index + 1]
        rows.append(
            {
                "bin_index": index,
                "lower": _jsonable(config.confidence_bin_edges[index]),
                "upper": _jsonable(upper),
                "lower_exact": _fraction_json(lower),
                "upper_exact": (
                    "+inf"
                    if upper == "+inf"
                    else _fraction_json(
                        _protocol_edge_fraction(
                            upper, name="confidence bin edge"
                        )
                    )
                ),
                "lower_inclusive": True,
                "upper_inclusive": False,
            }
        )
    rows.append(
        {
            "bin_index": "infinity",
            "lower": "infinity",
            "upper": "infinity",
            "lower_exact": "infinity",
            "upper_exact": "infinity",
            "lower_inclusive": True,
            "upper_inclusive": True,
            "category": "exact-positive-infinity-empty-competitor-set",
        }
    )
    return rows


def _minimum_confidence(
    values: Sequence[Fraction | str],
) -> Fraction | str | None:
    if not values:
        return None
    finite = [value for value in values if isinstance(value, Fraction)]
    return min(finite) if finite else "infinity"


def _confidence_strictly_greater(
    value: Fraction | str, threshold: Fraction
) -> bool:
    if value == "infinity":
        return True
    if not isinstance(value, Fraction):
        raise TypeError("confidence value must be an exact fraction or +inf")
    return value > threshold


def _confidence_equal(value: Fraction | str, threshold: Fraction) -> bool:
    return isinstance(value, Fraction) and value == threshold


def _count_bin_label(value: int, config: AnalysisConfig) -> str:
    value = _integer(value, name="binned count")
    if value == 0:
        return "0"
    for item in config.cluster_bins:
        if value >= item.minimum and (
            item.maximum_exclusive is None or value < item.maximum_exclusive
        ):
            return item.label
    raise ValueError(f"count {value} is outside the frozen display bins")


def _grouped_outcome_rows(
    grouped: Mapping[tuple[Any, ...], Mapping[str, int]],
    *,
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, counts in grouped.items():
        if len(key) != len(dimensions):
            raise ValueError("outcome grouping key has the wrong arity")
        rows.append(
            {
                **{
                    name: _jsonable(value)
                    for name, value in zip(dimensions, key)
                },
                **_outcome_summary(counts),
            }
        )
    rows.sort(
        key=lambda row: canonical_json_bytes(
            {name: row[name] for name in dimensions}
        )
    )
    return rows


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("analysis cannot serialize nonfinite values")
        return value
    raise TypeError(f"analysis cannot serialize {type(value).__name__}")


def _histogram_json(counter: Mapping[Any, int]) -> list[list[Any]]:
    rows = []
    for key, count in counter.items():
        if count <= 0:
            raise ValueError("histogram counts must be positive")
        normalized_key = list(key) if isinstance(key, tuple) else [key]
        rows.append([*_jsonable(normalized_key), int(count)])
    rows.sort(
        key=lambda row: json.dumps(
            row[:-1], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    )
    return rows


def _fraction_is_zero(value: Any) -> bool:
    if value == 0:
        return True
    if isinstance(value, Mapping):
        if set(value) == {"numerator", "denominator"}:
            return value["numerator"] == 0 and value["denominator"] != 0
        if set(value) == {"mantissa", "exponent"}:
            return value["mantissa"] == 0
    return False


def _decision(value: Any) -> tuple[bool, str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
        or not isinstance(value[0], bool)
        or not isinstance(value[1], str)
        or not value[1]
    ):
        raise ValueError("completed component durable_decision is malformed")
    return bool(value[0]), str(value[1])


def _counter_fields(counters: Mapping[str, Any]) -> dict[str, int]:
    result = {
        str(name): _integer(value, name=f"counter {name}")
        for name, value in counters.items()
    }
    required = {
        "union_attempt_count",
        "successful_union_count",
        "failed_union_count",
        "heap_push_count",
        "heap_pop_count",
        "heap_operation_count",
    }
    if not required <= set(result):
        raise ValueError(f"lane counters omit {sorted(required - set(result))}")
    if result["union_attempt_count"] != (
        result["successful_union_count"] + result["failed_union_count"]
    ):
        raise ValueError("union counter reconciliation failed")
    if result["heap_operation_count"] != (
        result["heap_push_count"] + result["heap_pop_count"]
    ):
        raise ValueError("heap counter reconciliation failed")
    return result


def _summary_contingency(summary: Mapping[str, Any]) -> dict[str, int]:
    raw = _mapping(summary.get("paired_contingency"), name="paired_contingency")
    if set(raw) != {"a", "b", "c", "d"}:
        raise ValueError("paired_contingency must contain exactly a/b/c/d")
    return {name: _integer(raw[name], name=f"paired {name}") for name in raw}


def _summary_hrlk(summary: Mapping[str, Any]) -> dict[tuple[int, int, int, int], int]:
    raw = summary.get("hrlk_joint_histogram")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError("summary H/R/L/K histogram must be an array")
    result: dict[tuple[int, int, int, int], int] = {}
    for row in raw:
        if not isinstance(row, Sequence) or len(row) != 5:
            raise ValueError("summary H/R/L/K rows must have five entries")
        key = tuple(_integer(row[k], name="H/R/L/K value") for k in range(4))
        if key in result:
            raise ValueError("summary H/R/L/K histogram has duplicate cells")
        result[key] = _integer(row[4], name="H/R/L/K count", minimum=1)
    validate_hrlk_histogram(result)
    return result


def _control_reconciliation(
    control: Mapping[str, Any], *, shots: int
) -> dict[str, Any]:
    expected = {
        "ordinary_treatment_vs_telemetry",
        "global_vs_adapter_control",
        "global_vs_uf_shadow",
    }
    if set(control) != expected:
        raise ValueError("control equality has an unexpected arm set")
    result: dict[str, Any] = {}
    for name in sorted(expected):
        row = _mapping(control[name], name=f"control {name}")
        row_shots = _integer(row.get("shots"), name=f"{name} shots")
        equal = _integer(row.get("equal"), name=f"{name} equal")
        mismatches = _integer(row.get("mismatches"), name=f"{name} mismatches")
        if row_shots != shots or equal + mismatches != shots or mismatches != 0:
            raise ValueError(f"control equality failed for {name}")
        result[name] = {
            "shots": row_shots,
            "equal": equal,
            "mismatches": mismatches,
            "status": "reconciled",
        }
    return result


def _characterization_tables(
    *,
    summary: Mapping[str, Any],
    shots: Sequence[Mapping[str, Any]],
    lanes_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
    components_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Builds required raw-count characterization views from authenticated rows."""

    provenance = _mapping(summary.get("provenance"), name="summary provenance")
    num_observables = _integer(
        provenance.get("num_observables"),
        name="provenance num_observables",
        minimum=1,
    )
    shot_count = len(shots)
    lane_counts = {len(lanes_by_shot[shot_id]) for shot_id in range(shot_count)}
    if len(lane_counts) != 1:
        raise ValueError("shots do not share one dense lane count")
    lanes_per_shot = next(iter(lane_counts))
    if lanes_per_shot <= 0 or lanes_per_shot % 2:
        raise ValueError("lane count must be positive and even")
    patch_count = lanes_per_shot // 2

    observable = [Counter() for _ in range(num_observables)]
    global_masks: Counter[str] = Counter()
    treatment_masks: Counter[str] = Counter()
    disagreement_masks: Counter[str] = Counter()
    global_weights: Counter[int] = Counter()
    treatment_weights: Counter[int] = Counter()
    disagreement_weights: Counter[int] = Counter()

    conditional_activation: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    conditional_commit: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    conditional_joint: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    exact_committed_patches: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    exact_committed_components: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    exact_h: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    exact_r: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    count_bins: dict[str, defaultdict[tuple[Any, ...], Counter[str]]] = {
        name: defaultdict(Counter)
        for name in (
            "original_detector_count",
            "residual_detector_count",
            "largest_final_component",
            "largest_committed_component",
            "committed_defect_count",
        )
    }
    completeness_outcomes: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    durable_reason_outcomes: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)

    lane_stats: dict[tuple[int, str], dict[str, Any]] = {}
    basis_stats: dict[str, dict[str, Any]] = {}

    def new_lane_stats() -> dict[str, Any]:
        return {
            "lanes": 0,
            "activated": 0,
            "original": 0,
            "residual": 0,
            "completed_components": 0,
            "eligible_components": 0,
            "committed_components": 0,
            "censored_components": 0,
            "statuses": Counter(),
            "activated_outcomes": Counter(),
        }

    component_confidence: defaultdict[int | str, Counter[str]] = defaultdict(Counter)
    component_confidence_by_lane: defaultdict[
        tuple[int, str, int | str], Counter[str]
    ] = defaultdict(Counter)
    shot_confidence: defaultdict[int | str, Counter[str]] = defaultdict(Counter)
    no_commit_outcomes: Counter[str] = Counter()
    committed_component_margins: list[tuple[Fraction | str, int]] = []
    shot_facts: list[dict[str, Any]] = []

    activated_shots = committed_shots = fallback_shots = 0
    activated_lanes = 0
    completed_components = eligible_components = committed_components = 0
    censored_component_rows = 0
    active_patches = committed_patches_total = aborted_patches = fallback_patches = 0
    total_h = total_r = total_l = total_k = 0
    unowned_histogram: Counter[int] = Counter()
    role_totals: Counter[str] = Counter()
    role_joint_histogram: Counter[
        tuple[int, int, int, int, int, int]
    ] = Counter()

    for shot in shots:
        shot_id = _integer(shot.get("global_shot_id"), name="global_shot_id")
        if shot_id >= shot_count:
            raise ValueError("shot ID exceeds characterization range")
        actual = _decode_packed_hex(
            shot.get("actual_observables_hex"),
            bits=num_observables,
            name="actual observables",
        )
        global_prediction = _decode_packed_hex(
            shot.get("global_prediction_hex"),
            bits=num_observables,
            name="global prediction",
        )
        treatment_prediction = _decode_packed_hex(
            shot.get("treatment_prediction_hex"),
            bits=num_observables,
            name="treatment prediction",
        )
        global_error = _xor_bytes(global_prediction, actual)
        treatment_error = _xor_bytes(treatment_prediction, actual)
        disagreement = _xor_bytes(global_prediction, treatment_prediction)
        global_failed = any(global_error)
        treatment_failed = any(treatment_error)
        if shot.get("global_failed") is not global_failed:
            raise ValueError("per-observable global mask disagrees with shot flag")
        if shot.get("treatment_failed") is not treatment_failed:
            raise ValueError("per-observable treatment mask disagrees with shot flag")
        if shot.get("prediction_agreement") is not (not any(disagreement)):
            raise ValueError("per-observable disagreement mask disagrees with shot flag")
        outcome = _paired_outcome(
            global_failed=global_failed, treatment_failed=treatment_failed
        )
        global_masks[global_error.hex()] += 1
        treatment_masks[treatment_error.hex()] += 1
        disagreement_masks[disagreement.hex()] += 1
        global_weights[sum(value.bit_count() for value in global_error)] += 1
        treatment_weights[sum(value.bit_count() for value in treatment_error)] += 1
        disagreement_weights[sum(value.bit_count() for value in disagreement)] += 1
        for observable_id in range(num_observables):
            global_bit = _bit(global_error, observable_id)
            treatment_bit = _bit(treatment_error, observable_id)
            observable[observable_id][
                _paired_outcome(
                    global_failed=global_bit,
                    treatment_failed=treatment_bit,
                )
            ] += 1
            observable[observable_id]["disagreement"] += int(
                _bit(disagreement, observable_id)
            )

        metrics = _mapping(shot.get("adapter_metrics"), name="adapter_metrics")
        h = _integer(metrics.get("original_detector_count"), name="H")
        r = _integer(metrics.get("residual_detector_count"), name="R")
        l = _integer(metrics.get("lane_owned_detector_count"), name="L")
        k = _integer(metrics.get("committed_defect_count"), name="K")
        complete = _bool(
            metrics.get("cluster_summary_complete"),
            name="cluster_summary_complete",
        )
        raw_lane_original = metrics.get("lane_original_detector_counts")
        raw_lane_residual = metrics.get("lane_residual_detector_counts")
        if (
            not isinstance(raw_lane_original, list)
            or not isinstance(raw_lane_residual, list)
            or len(raw_lane_original) != lanes_per_shot
            or len(raw_lane_residual) != lanes_per_shot
        ):
            raise ValueError(
                "shot telemetry must retain exact original/residual lane counts"
            )
        recorded_lane_original = [
            _integer(value, name="lane original detector count")
            for value in raw_lane_original
        ]
        recorded_lane_residual = [
            _integer(value, name="lane residual detector count")
            for value in raw_lane_residual
        ]
        role_names = ("body", "terminal", "yoke")
        original_roles = tuple(
            _integer(
                metrics.get(f"original_{role}_detector_count"),
                name=f"original {role} detector count",
            )
            for role in role_names
        )
        residual_roles = tuple(
            _integer(
                metrics.get(f"residual_{role}_detector_count"),
                name=f"residual {role} detector count",
            )
            for role in role_names
        )
        if (
            sum(recorded_lane_original) != l
            or sum(recorded_lane_residual) != l - k
            or original_roles[0] + original_roles[1] != l
            or residual_roles[0] + residual_roles[1] != l - k
            or sum(original_roles) != h
            or sum(residual_roles) != r
            or original_roles[2] != residual_roles[2]
        ):
            raise ValueError("shot lane/role workload telemetry does not reconcile")
        for role, original_count, residual_count in zip(
            role_names, original_roles, residual_roles
        ):
            role_totals[f"original_{role}"] += original_count
            role_totals[f"residual_{role}"] += residual_count
        role_joint_histogram[(*original_roles, *residual_roles)] += 1
        total_h += h
        total_r += r
        total_l += l
        total_k += k
        unowned_histogram[h - l] += 1

        rows_by_lane: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        completed_sizes: list[int] = []
        committed_sizes: list[int] = []
        committed_margins: list[Fraction] = []
        durable_reasons_on_shot: set[str] = set()
        lane_original = [0] * lanes_per_shot
        lane_committed = [0] * lanes_per_shot
        lane_completed = [0] * lanes_per_shot
        lane_eligible = [0] * lanes_per_shot
        lane_committed_components = [0] * lanes_per_shot
        lane_censored = [0] * lanes_per_shot
        seen_component_indices: dict[int, set[int]] = defaultdict(set)
        for component in components_by_shot[shot_id]:
            lane_offset = _integer(
                component.get("lane_offset"), name="component lane_offset"
            )
            if lane_offset >= lanes_per_shot:
                raise ValueError("component references an unknown lane")
            rows_by_lane[lane_offset].append(component)
            adapter = _mapping(component.get("adapter"), name="component adapter")
            component_index = _integer(
                adapter.get("component_index"), name="component_index"
            )
            if component_index in seen_component_indices[lane_offset]:
                raise ValueError("component indices must be unique within a lane")
            seen_component_indices[lane_offset].add(component_index)
            patch_id = lane_offset // 2
            basis = "X" if lane_offset % 2 == 0 else "Z"
            state = component.get("state_collection")
            if state == "completed_components":
                size = _integer(
                    adapter.get("cluster_defect_count"),
                    name="cluster_defect_count",
                    minimum=1,
                )
                raw_margin = adapter.get("exact_margin")
                margin: Fraction | str = (
                    "infinity"
                    if raw_margin == "infinity"
                    else _exact_fraction(
                        raw_margin, name="component exact_margin"
                    )
                )
                decision = adapter.get("gate_decision")
                if decision not in {"eligible", "deferred"}:
                    raise ValueError("component gate_decision is malformed")
                committed, durable_reason = _decision(
                    component.get("durable_decision")
                )
                if committed and (
                    decision != "eligible" or durable_reason != "committed"
                ):
                    raise ValueError("durable component was not gate eligible")
                confidence_bin = _confidence_bin_index(margin, config)
                confidence_counts = component_confidence[confidence_bin]
                confidence_counts["components"] += 1
                confidence_counts["eligible"] += int(decision == "eligible")
                confidence_counts["committed"] += int(committed)
                by_lane = component_confidence_by_lane[
                    (patch_id, basis, confidence_bin)
                ]
                by_lane["components"] += 1
                by_lane["eligible"] += int(decision == "eligible")
                by_lane["committed"] += int(committed)
                lane_original[lane_offset] += size
                lane_completed[lane_offset] += 1
                lane_eligible[lane_offset] += int(decision == "eligible")
                lane_committed_components[lane_offset] += int(committed)
                completed_sizes.append(size)
                completed_components += 1
                eligible_components += int(decision == "eligible")
                committed_components += int(committed)
                if committed:
                    lane_committed[lane_offset] += size
                    committed_sizes.append(size)
                    committed_margins.append(margin)
                    committed_component_margins.append((margin, size))
                else:
                    durable_reasons_on_shot.add(durable_reason)
            elif state == "censored_components":
                size = _integer(
                    adapter.get("partial_cluster_defect_lower_bound"),
                    name="partial_cluster_defect_lower_bound",
                    minimum=1,
                )
                if component.get("durable_decision") is not None:
                    raise ValueError("censored component has a durable decision")
                lane_original[lane_offset] += size
                lane_censored[lane_offset] += 1
                censored_component_rows += 1
            else:
                raise ValueError("component state_collection is malformed")

        lanes = {
            _integer(row.get("lane_offset"), name="lane_offset"): row
            for row in lanes_by_shot[shot_id]
        }
        if set(lanes) != set(range(lanes_per_shot)):
            raise ValueError("lane offsets are not one dense patch/basis partition")
        lane_active_flags: list[bool] = []
        lane_statuses: list[str] = []
        for lane_offset in range(lanes_per_shot):
            lane = _mapping(lanes[lane_offset].get("adapter"), name="lane adapter")
            status = lane.get("status")
            if status not in {"empty", "completed", "censored"}:
                raise ValueError("lane status is malformed")
            lane_statuses.append(str(status))
            if status == "empty" and (
                recorded_lane_original[lane_offset] != 0
                or rows_by_lane[lane_offset]
            ):
                raise ValueError("empty lane contains detector/component telemetry")
            if status == "completed" and (
                recorded_lane_original[lane_offset] == 0
                or lane_censored[lane_offset] != 0
                or lane_original[lane_offset]
                != recorded_lane_original[lane_offset]
            ):
                raise ValueError("completed lane component partition is malformed")
            if status == "censored" and (
                recorded_lane_original[lane_offset] == 0
                or lane_censored[lane_offset] == 0
            ):
                raise ValueError("censored lane component partition is malformed")
            active = status != "empty"
            if active is not (recorded_lane_original[lane_offset] > 0):
                raise ValueError("lane activation disagrees with component defects")
            lane_active_flags.append(active)
            activated_lanes += int(active)
            patch_id = lane_offset // 2
            basis = "X" if lane_offset % 2 == 0 else "Z"
            residual = recorded_lane_residual[lane_offset]
            if residual < 0 or residual > recorded_lane_original[lane_offset]:
                raise ValueError("lane committed defects exceed original defects")
            for key, target in (
                ((patch_id, basis), lane_stats),
                (basis, basis_stats),
            ):
                stats = target.setdefault(key, new_lane_stats())
                stats["lanes"] += 1
                stats["activated"] += int(active)
                stats["original"] += recorded_lane_original[lane_offset]
                stats["residual"] += residual
                stats["completed_components"] += lane_completed[lane_offset]
                stats["eligible_components"] += lane_eligible[lane_offset]
                stats["committed_components"] += lane_committed_components[
                    lane_offset
                ]
                stats["censored_components"] += lane_censored[lane_offset]
                stats["statuses"][str(status)] += 1
                if active:
                    stats["activated_outcomes"][outcome] += 1

        if sum(lane_committed) != k:
            raise ValueError("lane durable components do not reconcile with K")
        expected_lane_residual = [
            original - removed
            for original, removed in zip(
                recorded_lane_original, lane_committed
            )
        ]
        if expected_lane_residual != recorded_lane_residual:
            raise ValueError(
                "durable component partition differs from recorded lane residual counts"
            )
        if r != h - k:
            raise ValueError("shot residual does not reconcile with H-K")
        if not 0 <= k <= l <= h:
            raise ValueError("shot violates 0 <= K <= L <= H")

        activated = any(lane_active_flags)
        committed = bool(committed_sizes)
        if committed is not (k > 0):
            raise ValueError("shot durable-commit flag disagrees with K")
        activated_shots += int(activated)
        committed_shots += int(committed)
        fallback = activated and not committed
        fallback_shots += int(fallback)
        conditional_activation[(activated,)][outcome] += 1
        conditional_commit[(committed,)][outcome] += 1
        conditional_joint[(activated, committed,)][outcome] += 1

        committed_patch_count = 0
        for patch_id in range(patch_count):
            lane_ids = (2 * patch_id, 2 * patch_id + 1)
            patch_active = any(lane_active_flags[index] for index in lane_ids)
            patch_aborted = any(
                lane_statuses[index] == "censored" for index in lane_ids
            )
            patch_committed = any(
                lane_committed_components[index] > 0 for index in lane_ids
            )
            if patch_committed and patch_aborted:
                raise ValueError("aborted patch retained a durable component")
            active_patches += int(patch_active)
            aborted_patches += int(patch_aborted)
            committed_patches_total += int(patch_committed)
            fallback_patches += int(patch_active and not patch_committed)
            committed_patch_count += int(patch_committed)
        committed_component_count = len(committed_sizes)
        exact_committed_patches[(committed_patch_count,)][outcome] += 1
        exact_committed_components[(committed_component_count,)][outcome] += 1
        exact_h[(h,)][outcome] += 1
        exact_r[(r,)][outcome] += 1
        completeness_outcomes[(complete,)][outcome] += 1
        count_bins["original_detector_count"][
            (_count_bin_label(h, config),)
        ][outcome] += 1
        count_bins["residual_detector_count"][
            (_count_bin_label(r, config),)
        ][outcome] += 1
        count_bins["committed_defect_count"][
            (_count_bin_label(k, config),)
        ][outcome] += 1
        if complete:
            largest_final = max(completed_sizes, default=0)
            largest_committed = max(committed_sizes, default=0)
            recorded_largest = metrics.get("maximum_final_component_defect_count")
            if recorded_largest is not None and recorded_largest != largest_final:
                raise ValueError("recorded largest final component does not reconcile")
            count_bins["largest_final_component"][
                (_count_bin_label(largest_final, config),)
            ][outcome] += 1
            count_bins["largest_committed_component"][
                (_count_bin_label(largest_committed, config),)
            ][outcome] += 1
        else:
            count_bins["largest_final_component"][("incomplete",)][outcome] += 1
            count_bins["largest_committed_component"][("incomplete",)][outcome] += 1
        if durable_reasons_on_shot:
            for reason in sorted(durable_reasons_on_shot):
                durable_reason_outcomes[(reason,)][outcome] += 1
        else:
            durable_reason_outcomes[("none",)][outcome] += 1

        minimum_committed_margin = _minimum_confidence(committed_margins)
        if minimum_committed_margin is None:
            no_commit_outcomes[outcome] += 1
        else:
            shot_bin = _confidence_bin_index(minimum_committed_margin, config)
            shot_confidence[shot_bin][outcome] += 1
        shot_facts.append(
            {
                "outcome": outcome,
                "h": h,
                "k": k,
                "minimum_committed_margin": minimum_committed_margin,
                "committed_components": tuple(
                    (margin, size)
                    for margin, size in zip(committed_margins, committed_sizes)
                ),
            }
        )

    if total_r != total_h - total_k or total_l < total_k:
        raise ValueError("global lane/workload totals do not reconcile")
    if sum(global_masks.values()) != shot_count or sum(
        disagreement_masks.values()
    ) != shot_count:
        raise AssertionError("observable mask totals do not reconcile")

    observable_rows: list[dict[str, Any]] = []
    for observable_id, counts in enumerate(observable):
        paired = _outcome_summary(counts)
        if paired["denominator"] != shot_count:
            raise ValueError("per-observable denominator differs from shot count")
        observable_rows.append(
            {
                "observable_id": observable_id,
                "paired_outcomes": paired,
                "global_error": _rate(paired["global_errors"], shot_count),
                "treatment_error": _rate(
                    paired["treatment_errors"], shot_count
                ),
                "prediction_disagreement": _rate(
                    counts["disagreement"], shot_count
                ),
            }
        )

    def lane_table(
        source: Mapping[Any, Mapping[str, Any]], *, pooled: bool
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for key, stats in source.items():
            identity = (
                {"check_basis": key}
                if pooled
                else {"patch_id": key[0], "check_basis": key[1]}
            )
            result.append(
                {
                    **identity,
                    "lane_records": stats["lanes"],
                    "activation": _rate(stats["activated"], stats["lanes"]),
                    "original_detector_events": stats["original"],
                    "residual_detector_events": stats["residual"],
                    "removed_detector_events": stats["original"]
                    - stats["residual"],
                    "completed_components": stats["completed_components"],
                    "eligible_components": stats["eligible_components"],
                    "committed_components": stats["committed_components"],
                    "censored_partial_components": stats["censored_components"],
                    "statuses": {
                        name: int(stats["statuses"].get(name, 0))
                        for name in ("empty", "completed", "censored")
                    },
                    "shot_outcomes_conditional_on_lane_activation": _outcome_summary(
                        stats["activated_outcomes"]
                    ),
                }
            )
        result.sort(
            key=lambda row: (
                row.get("patch_id", -1), row["check_basis"]
            )
        )
        return result

    confidence_geometry = _confidence_bin_rows(config)
    component_bin_rows: list[dict[str, Any]] = []
    shot_bin_rows: list[dict[str, Any]] = []
    for geometry in confidence_geometry:
        index = geometry["bin_index"]
        component_counts = component_confidence[index]
        components_in_bin = int(component_counts.get("components", 0))
        eligible_in_bin = int(component_counts.get("eligible", 0))
        committed_in_bin = int(component_counts.get("committed", 0))
        component_bin_rows.append(
            {
                **geometry,
                "components": components_in_bin,
                "gate_acceptance": _rate(eligible_in_bin, components_in_bin),
                "durable_commit": _rate(committed_in_bin, components_in_bin),
            }
        )
        outcomes = _outcome_summary(shot_confidence[index])
        shot_bin_rows.append(
            {
                **geometry,
                "accepted_shots": _rate(outcomes["denominator"], shot_count),
                "accepted_shot_outcomes": outcomes,
                "regression_rate": _rate(
                    outcomes["regressions"], outcomes["denominator"]
                ),
                "recovery_rate": _rate(
                    outcomes["recoveries"], outcomes["denominator"]
                ),
            }
        )

    confidence_lane_rows: list[dict[str, Any]] = []
    for patch_id in range(patch_count):
        for basis in ("X", "Z"):
            for geometry in confidence_geometry:
                index = geometry["bin_index"]
                counts = component_confidence_by_lane[(patch_id, basis, index)]
                denominator = int(counts.get("components", 0))
                confidence_lane_rows.append(
                    {
                        "patch_id": patch_id,
                        "check_basis": basis,
                        **geometry,
                        "components": denominator,
                        "gate_acceptance": _rate(
                            int(counts.get("eligible", 0)), denominator
                        ),
                        "durable_commit": _rate(
                            int(counts.get("committed", 0)), denominator
                        ),
                    }
                )

    finite_thresholds = tuple(config.confidence_bin_edges[:-1])
    risk_coverage_rows: list[dict[str, Any]] = []
    for threshold_literal in finite_thresholds:
        threshold = _protocol_edge_fraction(
            threshold_literal, name="risk/coverage threshold"
        )
        outcomes: Counter[str] = Counter()
        qualifying_shots = 0
        retained_components = retained_defects = 0
        equal_components = equal_shots = 0
        for margin, size in committed_component_margins:
            retained = _confidence_strictly_greater(margin, threshold)
            retained_components += int(retained)
            retained_defects += size * int(retained)
            equal_components += int(_confidence_equal(margin, threshold))
        for fact in shot_facts:
            margin = fact["minimum_committed_margin"]
            if margin is not None and _confidence_equal(margin, threshold):
                equal_shots += 1
            if margin is not None and _confidence_strictly_greater(
                margin, threshold
            ):
                qualifying_shots += 1
                outcomes[str(fact["outcome"])] += 1
        outcome_summary = _outcome_summary(outcomes)
        risk_coverage_rows.append(
            {
                "threshold": _jsonable(threshold_literal),
                "threshold_exact": _fraction_json(threshold),
                "comparison": "strict-greater-than",
                "accepted_shots": _rate(qualifying_shots, shot_count),
                "accepted_shot_outcomes": outcome_summary,
                "regression_risk_given_accepted": _rate(
                    outcome_summary["regressions"], qualifying_shots
                ),
                "recovery_rate_given_accepted": _rate(
                    outcome_summary["recoveries"], qualifying_shots
                ),
                "frontend_defect_coverage": _rate(retained_defects, total_h),
                "committed_component_retention": _rate(
                    retained_components, committed_components
                ),
                "committed_defect_retention": _rate(retained_defects, total_k),
                "component_margin_equal_count": equal_components,
                "shot_minimum_margin_equal_count": equal_shots,
            }
        )
    if risk_coverage_rows and risk_coverage_rows[0][
        "frontend_defect_coverage"
    ]["numerator"] != total_k:
        raise ValueError(
            "zero-threshold strict confidence coverage differs from durable K"
        )

    total_patch_transactions = shot_count * patch_count
    routing_rates = {
        "shot_activation": _rate(activated_shots, shot_count),
        "shot_durable_commit": _rate(committed_shots, shot_count),
        "shot_all_fallback_unconditional": _rate(fallback_shots, shot_count),
        "shot_all_fallback_given_activation": _rate(
            fallback_shots, activated_shots
        ),
        "lane_activation": _rate(activated_lanes, shot_count * lanes_per_shot),
        "component_gate_acceptance": _rate(
            eligible_components, completed_components
        ),
        "component_durable_commit": _rate(
            committed_components, completed_components
        ),
        "patch_activation": _rate(active_patches, total_patch_transactions),
        "all_patch_commit": _rate(
            committed_patches_total, total_patch_transactions
        ),
        "active_patch_commit": _rate(committed_patches_total, active_patches),
        "patch_abort_unconditional": _rate(
            aborted_patches, total_patch_transactions
        ),
        "patch_abort_given_activation": _rate(aborted_patches, active_patches),
        "patch_fallback_unconditional": _rate(
            fallback_patches, total_patch_transactions
        ),
        "patch_fallback_given_activation": _rate(
            fallback_patches, active_patches
        ),
        "censored_partial_component_share": _rate(
            censored_component_rows,
            completed_components + censored_component_rows,
        ),
    }

    return {
        "observable_accuracy": {
            "num_observables": num_observables,
            "per_observable": observable_rows,
            "error_and_disagreement_masks": {
                "global_error_mask_histogram": _histogram_json(global_masks),
                "treatment_error_mask_histogram": _histogram_json(
                    treatment_masks
                ),
                "prediction_disagreement_mask_histogram": _histogram_json(
                    disagreement_masks
                ),
                "global_error_hamming_weight_histogram": _histogram_json(
                    global_weights
                ),
                "treatment_error_hamming_weight_histogram": _histogram_json(
                    treatment_weights
                ),
                "prediction_disagreement_hamming_weight_histogram": _histogram_json(
                    disagreement_weights
                ),
                "denominator_shots": shot_count,
            },
            "observable_role_masks": {
                "status": "not-recorded-v1",
                "reason": (
                    "the authenticated V1 shot ledger retains observable IDs and "
                    "packed masks but no yoke-versus-patch observable role map"
                ),
            },
        },
        "conditional_accuracy": {
            "by_any_uf_activation": _grouped_outcome_rows(
                conditional_activation, dimensions=("activated",)
            ),
            "by_any_durable_commit": _grouped_outcome_rows(
                conditional_commit, dimensions=("durable_commit",)
            ),
            "by_activation_and_durable_commit": _grouped_outcome_rows(
                conditional_joint,
                dimensions=("activated", "durable_commit"),
            ),
            "by_durable_defer_or_abort_reason": _grouped_outcome_rows(
                durable_reason_outcomes, dimensions=("reason",)
            ),
        },
        "shot_strata": {
            "exact_committed_patch_count": _grouped_outcome_rows(
                exact_committed_patches, dimensions=("committed_patches",)
            ),
            "exact_committed_component_count": _grouped_outcome_rows(
                exact_committed_components,
                dimensions=("committed_components",),
            ),
            "exact_original_detector_count": _grouped_outcome_rows(
                exact_h, dimensions=("original_detector_count",)
            ),
            "exact_residual_detector_count": _grouped_outcome_rows(
                exact_r, dimensions=("residual_detector_count",)
            ),
            "cluster_summary_complete": _grouped_outcome_rows(
                completeness_outcomes,
                dimensions=("cluster_summary_complete",),
            ),
            "display_bins": {
                "bin_source": "zero-plus-config.cluster_bins",
                "positive_bins": _jsonable(config.cluster_bins),
                **{
                    name: _grouped_outcome_rows(
                        grouped, dimensions=("bin",)
                    )
                    for name, grouped in count_bins.items()
                },
            },
        },
        "routing_rates": routing_rates,
        "lane_breakdown": {
            "lane_identity": (
                "dense lane_offset maps to patch_id=lane_offset//2 and "
                "check_basis=X for even offsets, Z for odd offsets"
            ),
            "by_patch_basis": lane_table(lane_stats, pooled=False),
            "by_basis": lane_table(basis_stats, pooled=True),
            "totals": {
                "original_lane_owned_detector_events": total_l,
                "residual_lane_owned_detector_events": total_l - total_k,
                "removed_lane_owned_detector_events": total_k,
            },
        },
        "detector_role_workload": {
            "global_unowned": {
                "status": "derived-exactly-as-H-minus-L",
                "original_detector_events": total_h - total_l,
                "residual_detector_events": total_h - total_l,
                "per_shot_histogram": _histogram_json(unowned_histogram),
                "shots": shot_count,
            },
            "terminal": {
                "status": "recorded-v1",
                "original_detector_events": role_totals["original_terminal"],
                "residual_detector_events": role_totals["residual_terminal"],
                "removed_detector_events": role_totals["original_terminal"]
                - role_totals["residual_terminal"],
                "shots": shot_count,
            },
            "yoke": {
                "status": "recorded-v1",
                "original_detector_events": role_totals["original_yoke"],
                "residual_detector_events": role_totals["residual_yoke"],
                "removed_detector_events": role_totals["original_yoke"]
                - role_totals["residual_yoke"],
                "shots": shot_count,
            },
            "body": {
                "status": "recorded-v1",
                "original_detector_events": role_totals["original_body"],
                "residual_detector_events": role_totals["residual_body"],
                "removed_detector_events": role_totals["original_body"]
                - role_totals["residual_body"],
                "shots": shot_count,
            },
            "joint_original_residual_role_histogram": _histogram_json(
                role_joint_histogram
            ),
        },
        "confidence": {
            "arithmetic": "exact-rational-telemetry-vs-exact-binary64-dyadic-edges",
            "bin_edges": _jsonable(config.confidence_bin_edges),
            "component_acceptance_by_bin": component_bin_rows,
            "component_acceptance_by_patch_basis_bin": confidence_lane_rows,
            "shot_acceptance_by_minimum_durable_margin_bin": shot_bin_rows,
            "no_durable_commit_shot_outcomes": _outcome_summary(
                no_commit_outcomes
            ),
            "downstream_regression_recovery_by_accepted_confidence_bin": shot_bin_rows,
        },
        "risk_coverage": {
            "status": "descriptive-no-counterfactual-redecode",
            "threshold_source": "config.confidence_bin_edges-finite-values",
            "rows": risk_coverage_rows,
        },
    }


def _selection_digest(root: bytes, category: str, shot_id: int) -> str:
    return hashlib.sha256(
        root
        + b"patch-uf-casebook-v1\0"
        + category.encode("utf-8")
        + b"\0"
        + str(shot_id).encode("ascii")
    ).hexdigest()


def _casebook(
    *,
    config: AnalysisConfig,
    shots: Sequence[Mapping[str, Any]],
    lanes_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
    components_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    root = bytes.fromhex(config.casebook_seed_root)
    candidates: dict[str, dict[int, int | None]] = {
        category: {} for category in CASEBOOK_CATEGORIES
    }
    for shot in shots:
        shot_id = int(shot["global_shot_id"])
        global_failed = bool(shot["global_failed"])
        treatment_failed = bool(shot["treatment_failed"])
        if not global_failed and treatment_failed:
            candidates["regression"][shot_id] = None
        if global_failed and not treatment_failed:
            candidates["recovery"][shot_id] = None
        if not bool(shot["prediction_agreement"]):
            candidates["prediction-disagreement"][shot_id] = None
        completed_sizes: list[int] = []
        committed_sizes: list[int] = []
        censored_sizes: list[int] = []
        for row in components_by_shot[shot_id]:
            adapter = row["adapter"]
            if row["state_collection"] == "completed_components":
                size = int(adapter["cluster_defect_count"])
                completed_sizes.append(size)
                committed, durable_reason = _decision(row["durable_decision"])
                reasons = set(adapter.get("gate_reason_set", ()))
                if not committed and "port-yoke" in reasons:
                    candidates["port-yoke-defer"][shot_id] = None
                if not committed and "port-cross-lane" in reasons:
                    candidates["port-cross-lane-defer"][shot_id] = None
                if (
                    "below-threshold" in reasons
                    and _fraction_is_zero(adapter.get("exact_margin"))
                ):
                    candidates["threshold-tie"][shot_id] = None
                if committed:
                    committed_sizes.append(size)
                    if bool(adapter.get("boundary_reached")):
                        candidates["boundary-using-commit"][shot_id] = None
                if durable_reason in (
                    "local-incomplete-neutralization-patch-abort",
                    "budget-exhaustion-patch-abort",
                ):
                    candidates[durable_reason][shot_id] = None
            else:
                censored_sizes.append(
                    int(adapter.get("partial_cluster_defect_lower_bound", 0))
                )
        for lane in lanes_by_shot[shot_id]:
            reason = lane["adapter"].get("censor_reason")
            if reason == "local-incomplete-neutralization":
                candidates[
                    "local-incomplete-neutralization-patch-abort"
                ][shot_id] = None
            elif reason == "budget-exhaustion":
                candidates["budget-exhaustion-patch-abort"][shot_id] = None
        if completed_sizes:
            candidates["largest-final-component"][shot_id] = max(completed_sizes)
        if committed_sizes:
            candidates["largest-committed-component"][shot_id] = max(committed_sizes)
        if censored_sizes:
            candidates["largest-censored-partial-lower-bound"][shot_id] = max(
                censored_sizes
            )
        candidates["highest-heap-operation-count"][shot_id] = sum(
            int(lane["adapter"]["counters"]["heap_operation_count"])
            for lane in lanes_by_shot[shot_id]
        )

    shot_by_id = {int(row["global_shot_id"]): row for row in shots}
    result: dict[str, Any] = {}
    for category in CASEBOOK_CATEGORIES:
        ordered = []
        for shot_id, metric in candidates[category].items():
            digest = _selection_digest(root, category, shot_id)
            sort_key = (
                (-int(metric), digest, shot_id)
                if category in _METRIC_CATEGORIES and metric is not None
                else (digest, shot_id)
            )
            ordered.append((sort_key, shot_id, metric, digest))
        ordered.sort(key=lambda item: item[0])
        selected = ordered[: config.maximum_cases_per_category]
        rows = [
            {
                "global_shot_id": shot_id,
                "metric": metric,
                "selection_sha256": digest,
                "shot": shot_by_id[shot_id],
                "lanes": list(lanes_by_shot[shot_id]),
                "components": list(components_by_shot[shot_id]),
            }
            for _, shot_id, metric, digest in selected
        ]
        result[category] = {
            "candidate_shots": len(ordered),
            "retained_shots": len(rows),
            "maximum_retained": config.maximum_cases_per_category,
            "selection": "metric-descending-then-rooted-sha256"
            if category in _METRIC_CATEGORIES
            else "rooted-sha256-ascending",
            "rows": rows,
        }
    return result


def _report(data: Mapping[str, Any]) -> str:
    accuracy = data["paired_accuracy"]
    workload = data["workload_coverage"]["summary"]
    cluster = data["cluster_sizes"]["summary"]
    routing = data["routing"]
    routing_rates = data["routing_rates"]
    roles = data["detector_role_workload"]
    confidence = data["confidence"]
    lines = [
        "# Confidence-gated patch-UF characterization",
        "",
        f"Analysis digest: `{data['payload_sha256']}`.",
        "",
        "## Paired accuracy",
        "",
        (
            f"Shots: {accuracy['shots']}. Paired table a/b/c/d: "
            f"{accuracy['a']}/{accuracy['b']}/{accuracy['c']}/{accuracy['d']}. "
            f"Global failures: {accuracy['global_failures']}/{accuracy['shots']}; "
            f"treatment failures: {accuracy['treatment_failures']}/{accuracy['shots']}; "
            f"discordant pairs: {accuracy['discordant']}/{accuracy['shots']}."
        ),
        "",
        "## Workload and coverage",
        "",
        (
            f"Exact totals H/R/L/K: {workload['original_total']}/"
            f"{workload['residual_total']}/{workload['lane_owned_total']}/"
            f"{workload['committed_total']} across {workload['shots']} shots."
        ),
        (
            "Workload ratio: "
            f"{workload['workload_ratio']['value']} "
            f"({workload['workload_ratio']['status']}); frontend coverage: "
            f"{workload['frontend_coverage']['value']} "
            f"({workload['frontend_coverage']['status']})."
        ),
        "",
        "## Cluster sizes",
        "",
        (
            f"Completed-component denominator: "
            f"{cluster['completed_components']['denominator']}; complete-shot maxima: "
            f"{cluster['complete_shot_maxima']['denominator']}/{cluster['shots']} shots; "
            f"censored cluster summaries: {cluster['censored_cluster_summary_shots']}/"
            f"{cluster['shots']}."
        ),
        "",
        "## Routing and reconciliation",
        "",
        (
            f"Completed components: {routing['completed_components']}; committed: "
            f"{routing['committed_components']}; durable-deferred: "
            f"{routing['durable_deferred_components']}; censored component rows: "
            f"{routing['censored_components']}."
        ),
        (
            "Shot activation/commit: "
            f"{routing_rates['shot_activation']['numerator']}/"
            f"{routing_rates['shot_activation']['denominator']} and "
            f"{routing_rates['shot_durable_commit']['numerator']}/"
            f"{routing_rates['shot_durable_commit']['denominator']}."
        ),
        "",
        "## Observable, role, and confidence tables",
        "",
        (
            f"Per-observable rows: {data['observable_accuracy']['num_observables']}; "
            f"terminal workload H/R: {roles['terminal']['original_detector_events']}/"
            f"{roles['terminal']['residual_detector_events']}; yoke workload H/R: "
            f"{roles['yoke']['original_detector_events']}/"
            f"{roles['yoke']['residual_detector_events']}."
        ),
        (
            "Confidence arithmetic: "
            f"{confidence['arithmetic']}; finite risk/coverage thresholds: "
            f"{len(data['risk_coverage']['rows'])}."
        ),
        "",
        "All displayed conditional quantities retain raw numerators and denominators "
        "in `analysis.json`. Zero-denominator ratios are null/not-estimable.",
        "",
    ]
    return "\n".join(lines)


def analyze_verified_collection(
    verified: VerifiedCollection,
    *,
    config: AnalysisConfig,
) -> AnalysisArtifacts:
    """Reconcile authenticated rows and build canonical analysis artifacts."""

    if not isinstance(verified, VerifiedCollection):
        raise TypeError("verified must be a VerifiedCollection")
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be AnalysisConfig")
    summary = _mapping(verified.summary, name="verified summary")
    shots = _integer(summary.get("shots"), name="summary shots", minimum=1)
    if len(verified.shot_rows) != shots or len(verified.cluster_records) != shots:
        raise ValueError("verified shot/cluster row count does not match summary")
    shot_rows = tuple(verified.shot_rows)
    shot_ids = [_integer(row.get("global_shot_id"), name="global_shot_id") for row in shot_rows]
    if shot_ids != list(range(shots)):
        raise ValueError("shot rows must be in complete canonical global-shot order")
    if [record.global_shot_id for record in verified.cluster_records] != shot_ids:
        raise ValueError("cluster records do not align with shot rows")

    lanes_by_shot: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in verified.lane_rows:
        item = _mapping(row, name="lane row")
        shot_id = _integer(item.get("global_shot_id"), name="lane global_shot_id")
        if shot_id >= shots:
            raise ValueError("lane row references an unknown shot")
        lanes_by_shot[shot_id].append(item)
    components_by_shot: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in verified.component_rows:
        item = _mapping(row, name="component row")
        shot_id = _integer(item.get("global_shot_id"), name="component global_shot_id")
        if shot_id >= shots:
            raise ValueError("component row references an unknown shot")
        components_by_shot[shot_id].append(item)
    if len(verified.lane_rows) != _integer(
        summary.get("lane_records"), name="summary lane_records"
    ) or len(verified.component_rows) != _integer(
        summary.get("component_records"), name="summary component_records"
    ):
        raise ValueError("summary lane/component counts do not reconcile")

    observed_contingency: Counter[str] = Counter({name: 0 for name in "abcd"})
    observed_hrlk: Counter[tuple[int, int, int, int]] = Counter()
    agreements = 0
    committed_shots = 0
    for row in shot_rows:
        shot_id = int(row["global_shot_id"])
        global_failed = _bool(row.get("global_failed"), name="global_failed")
        treatment_failed = _bool(
            row.get("treatment_failed"), name="treatment_failed"
        )
        agreement = _bool(
            row.get("prediction_agreement"), name="prediction_agreement"
        )
        agreements += int(agreement)
        observed_contingency[
            "d" if global_failed and treatment_failed else
            "c" if global_failed else
            "b" if treatment_failed else "a"
        ] += 1
        metrics = _mapping(row.get("adapter_metrics"), name="shot adapter_metrics")
        h = _integer(metrics.get("original_detector_count"), name="H")
        r = _integer(metrics.get("residual_detector_count"), name="R")
        l = _integer(metrics.get("lane_owned_detector_count"), name="L")
        k = _integer(metrics.get("committed_defect_count"), name="K")
        if not 0 <= k <= l <= h or r != h - k:
            raise ValueError("shot H/R/L/K reconciliation failed")
        observed_hrlk[(h, r, l, k)] += 1
        committed_shots += int(k > 0)
        expected_lane_count = _integer(row.get("lane_count"), name="shot lane_count")
        expected_component_count = _integer(
            row.get("component_count"), name="shot component_count"
        )
        if len(lanes_by_shot[shot_id]) != expected_lane_count:
            raise ValueError("shot lane cross-index does not reconcile")
        if len(components_by_shot[shot_id]) != expected_component_count:
            raise ValueError("shot component cross-index does not reconcile")
        lane_offsets = sorted(
            _integer(item.get("lane_offset"), name="lane_offset")
            for item in lanes_by_shot[shot_id]
        )
        if lane_offsets != list(range(expected_lane_count)):
            raise ValueError("lane offsets must be unique and dense within each shot")

    expected_contingency = _summary_contingency(summary)
    if dict(observed_contingency) != expected_contingency:
        raise ValueError("shot paired contingency does not reconcile with summary")
    expected_agreements = _integer(
        summary.get("prediction_agreements"), name="prediction_agreements"
    )
    if agreements != expected_agreements:
        raise ValueError("prediction agreement count does not reconcile")
    expected_hrlk = _summary_hrlk(summary)
    if dict(observed_hrlk) != expected_hrlk:
        raise ValueError("shot H/R/L/K histogram does not reconcile with summary")

    counter_sums: Counter[str] = Counter()
    counter_maxima: dict[str, int] = {}
    lane_status = Counter()
    lane_censor = Counter()
    components_by_lane: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for shot_id, rows in components_by_shot.items():
        for row in rows:
            lane_offset = _integer(row.get("lane_offset"), name="component lane_offset")
            if lane_offset >= len(lanes_by_shot[shot_id]):
                raise ValueError("component row references an unknown lane")
            components_by_lane[(shot_id, lane_offset)].append(row)
    for shot_id, rows in lanes_by_shot.items():
        for row in rows:
            lane_offset = int(row["lane_offset"])
            adapter = _mapping(row.get("adapter"), name="lane adapter")
            status = adapter.get("status")
            if status not in {"empty", "completed", "censored"}:
                raise ValueError("lane status is malformed")
            lane_status[str(status)] += 1
            reason = adapter.get("censor_reason")
            if status == "censored":
                if reason not in {"budget-exhaustion", "local-incomplete-neutralization"}:
                    raise ValueError("censored lane has an unsupported reason")
                lane_censor[str(reason)] += 1
            elif reason is not None:
                raise ValueError("non-censored lane has a censor reason")
            counters = _counter_fields(
                _mapping(adapter.get("counters"), name="lane counters")
            )
            counter_sums.update(counters)
            for name, value in counters.items():
                counter_maxima[name] = max(counter_maxima.get(name, 0), value)
            rows_for_lane = components_by_lane[(shot_id, lane_offset)]
            ancestry = sum(
                _integer(
                    _mapping(item.get("adapter"), name="component adapter").get(
                        "merge_count"
                    ),
                    name="merge_count",
                )
                for item in rows_for_lane
            )
            if ancestry != counters["successful_union_count"]:
                raise ValueError("lane merge ancestry does not reconcile")
            completed_count = sum(
                item.get("state_collection") == "completed_components"
                for item in rows_for_lane
            )
            censored_count = sum(
                item.get("state_collection") == "censored_components"
                for item in rows_for_lane
            )
            if "completed_components" in adapter and len(adapter["completed_components"]) != completed_count:
                raise ValueError("lane completed component rows do not reconcile")
            if "censored_components" in adapter and len(adapter["censored_components"]) != censored_count:
                raise ValueError("lane censored component rows do not reconcile")

    gate_decisions = Counter()
    primary_reasons = Counter()
    gate_reason_sets = Counter()
    durable_reasons = Counter()
    port_kind_sets = Counter()
    completed_size_histogram = Counter()
    censored_size_histogram = Counter()
    joint_components = Counter()
    completed_components = committed_components = censored_components = 0
    committed_defects = 0
    for row in verified.component_rows:
        state = row.get("state_collection")
        adapter = _mapping(row.get("adapter"), name="component adapter")
        if state == "completed_components":
            completed_components += 1
            size = _integer(
                adapter.get("cluster_defect_count"),
                name="cluster_defect_count",
                minimum=1,
            )
            completed_size_histogram[size] += 1
            decision = adapter.get("gate_decision")
            primary = adapter.get("primary_gate_reason")
            reasons = tuple(sorted(adapter.get("gate_reason_set", ())))
            ports = tuple(sorted(adapter.get("port_kind_set", ())))
            if decision not in {"eligible", "deferred"} or not isinstance(primary, str):
                raise ValueError("completed component gate fields are malformed")
            if not all(isinstance(item, str) for item in (*reasons, *ports)):
                raise ValueError("component reason/port sets are malformed")
            committed, durable_reason = _decision(row.get("durable_decision"))
            committed_components += int(committed)
            committed_defects += size * int(committed)
            gate_decisions[str(decision)] += 1
            primary_reasons[str(primary)] += 1
            gate_reason_sets[reasons] += 1
            durable_reasons[(committed, durable_reason)] += 1
            port_kind_sets[ports] += 1
            joint_components[
                (
                    size,
                    str(decision),
                    reasons,
                    str(primary),
                    committed,
                    durable_reason,
                    bool(adapter.get("boundary_reached")),
                    ports,
                )
            ] += 1
        elif state == "censored_components":
            censored_components += 1
            if row.get("durable_decision") is not None:
                raise ValueError("censored components cannot have durable decisions")
            lower = _integer(
                adapter.get("partial_cluster_defect_lower_bound"),
                name="partial_cluster_defect_lower_bound",
                minimum=1,
            )
            censored_size_histogram[lower] += 1
        else:
            raise ValueError("component state_collection is malformed")

    durable_deferred = completed_components - committed_components
    if completed_components != committed_components + durable_deferred:
        raise AssertionError("completed component routing identity failed")
    if sum(completed_size_histogram.values()) != completed_components:
        raise ValueError("completed size histogram count does not reconcile")
    if sum(size * count for size, count in completed_size_histogram.items()) != sum(
        _integer(row["adapter"].get("cluster_defect_count"), name="cluster size", minimum=1)
        for row in verified.component_rows
        if row.get("state_collection") == "completed_components"
    ):
        raise ValueError("completed size histogram weighted sum does not reconcile")
    if committed_defects != sum(key[3] * count for key, count in observed_hrlk.items()):
        raise ValueError("committed component defects do not reconcile with K")

    accuracy = summarize_paired_accuracy(
        **expected_contingency,
        prediction_agreements=agreements,
        alpha=config.alpha,
    )
    workload = summarize_workload_coverage(expected_hrlk)
    workload_bootstrap = bootstrap_workload_coverage(
        joint_counts=expected_hrlk,
        replicates=config.workload_bootstrap_replicates,
        seed=config.workload_bootstrap_seed,
        alpha=config.alpha,
    )
    cluster = summarize_cluster_sizes(verified.cluster_records)
    if dict(cluster.completed_components.histogram) != dict(completed_size_histogram):
        raise ValueError("cluster records do not reconcile with completed components")
    cluster_bootstrap = bootstrap_cluster_sizes(
        records=verified.cluster_records,
        component_bins=config.cluster_bins,
        replicates=config.cluster_bootstrap_replicates,
        seed=config.cluster_bootstrap_seed,
        alpha=config.alpha,
    )
    controls = _control_reconciliation(verified.control_equality, shots=shots)
    casebook = _casebook(
        config=config,
        shots=shot_rows,
        lanes_by_shot=lanes_by_shot,
        components_by_shot=components_by_shot,
    )
    characterization = _characterization_tables(
        summary=summary,
        shots=shot_rows,
        lanes_by_shot=lanes_by_shot,
        components_by_shot=components_by_shot,
        config=config,
    )

    payload: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source": {
            "experiment_id": summary.get("experiment_id"),
            "protocol_self_sha256": summary.get("protocol_self_sha256"),
            "collection_payload_sha256": summary.get("payload_sha256"),
            "stage": summary.get("stage"),
            "cell_id": summary.get("cell_id"),
            "corpus_identity": verified.corpus_identity,
        },
        "config": _jsonable(config),
        "paired_accuracy": _jsonable(accuracy),
        "hrlk_joint_histogram": [
            [*key, count] for key, count in sorted(expected_hrlk.items())
        ],
        "workload_coverage": {
            "summary": _jsonable(workload),
            "bootstrap": _jsonable(workload_bootstrap),
        },
        "cluster_sizes": {
            "summary": _jsonable(cluster),
            "bootstrap": _jsonable(cluster_bootstrap),
            "completed_size_histogram": _histogram_json(completed_size_histogram),
            "censored_partial_lower_bound_histogram": _histogram_json(
                censored_size_histogram
            ),
        },
        "routing": {
            "completed_components": completed_components,
            "committed_components": committed_components,
            "durable_deferred_components": durable_deferred,
            "censored_components": censored_components,
            "gate_decisions": _histogram_json(gate_decisions),
            "primary_gate_reasons": _histogram_json(primary_reasons),
            "gate_reason_sets": _histogram_json(gate_reason_sets),
            "durable_reasons": _histogram_json(durable_reasons),
            "port_kind_sets": _histogram_json(port_kind_sets),
            "joint_completed_component_histogram": _histogram_json(
                joint_components
            ),
            "lane_statuses": _histogram_json(lane_status),
            "lane_censor_reasons": _histogram_json(lane_censor),
        },
        "counters": {
            "lane_sum": dict(sorted(counter_sums.items())),
            "lane_maximum": dict(sorted(counter_maxima.items())),
        },
        "controls": controls,
        **characterization,
        "reconciliation": {
            "status": "reconciled",
            "shots": shots,
            "lane_records": len(verified.lane_rows),
            "component_records": len(verified.component_rows),
            "completed_final_component_count": completed_components,
            "committed_component_count": committed_components,
            "durable_deferred_component_count": durable_deferred,
            "completed_histogram_count": sum(completed_size_histogram.values()),
            "committed_defect_count": committed_defects,
        },
        "volume_tags": {
            "baseline_failures_lt_200": accuracy.global_failures < 200,
            "discordant_pairs_lt_100": accuracy.discordant < 100,
            "durable_commit_shots_lt_500": committed_shots < 500,
            "durable_commit_shots": committed_shots,
        },
        "casebook": casebook,
    }
    unsigned = _jsonable(payload)
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    normalized = _jsonable(payload)
    analysis_bytes = canonical_json_bytes(normalized)
    report_markdown = _report(normalized)
    report_bytes = report_markdown.encode("utf-8")
    return AnalysisArtifacts(normalized, report_markdown, analysis_bytes, report_bytes)


def analyze_collection(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    collection_out: Path,
    config: AnalysisConfig,
    processes: int = RANGE_COUNT,
    scientific: bool = True,
) -> AnalysisArtifacts:
    """Authenticate a collection and analyze it without mutating its tree."""

    verified = verify_collection(
        protocol,
        stage=stage,
        out=Path(collection_out),
        processes=processes,
        scientific=scientific,
    )
    return analyze_verified_collection(verified, config=config)


def write_analysis_bundle(out: Path, artifacts: AnalysisArtifacts) -> None:
    """Install ``analysis.json`` and ``report.md`` once, or verify exact bytes."""

    if not isinstance(artifacts, AnalysisArtifacts):
        raise TypeError("artifacts must be AnalysisArtifacts")
    out = Path(out)
    if out.exists() and (out.is_symlink() or not out.is_dir()):
        raise ValueError("analysis output must be a regular directory")
    out.mkdir(parents=True, exist_ok=True)
    allowed = {"analysis.json", "report.md"}
    if {item.name for item in out.iterdir()} - allowed:
        raise ValueError("analysis output contains unexpected entries")
    for path, data, prefix in (
        (out / "analysis.json", artifacts.analysis_bytes, "patch-uf-analysis-"),
        (out / "report.md", artifacts.report_bytes, "patch-uf-report-"),
    ):
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"existing analysis artifact differs: {path}")
        else:
            install_bytes_atomic(path, data, prefix=prefix, overwrite=False)
