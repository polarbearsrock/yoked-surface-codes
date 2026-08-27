"""Strict analysis of paired U0/ProMatch/Pinball campaign aggregates.

The analyzer deliberately consumes a small, additive projection of authenticated
batch ledgers.  Authentication, duplicate-batch detection, and campaign
completeness belong to the collector; this module then reconciles every
accuracy count against the three-arm correctness cube before emitting a
statistic.  Cube keys are three failure bits in ``(u0, promatch, pinball)``
order, where ``1`` means at least one observable was predicted incorrectly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from yoked.decoding._promatch_stats import (
    PairedContingency,
    canonical_json_bytes,
    clopper_pearson_lower,
    clopper_pearson_upper,
    tango_paired_risk_difference_upper,
)


ANALYSIS_SCHEMA = "ysc-pinball-promatch-analysis-v1"
ARM_ORDER = ("u0", "promatch", "pinball")
PAIR_DEFINITIONS = {
    "pinball_minus_promatch": ("pinball", "promatch"),
    "pinball_minus_u0": ("pinball", "u0"),
    "promatch_minus_u0": ("promatch", "u0"),
}
CELL_FIELDS = {
    "cell_id",
    "shots",
    "correctness_cube",
    "pairwise_contingencies",
    "prediction_agreement",
    "telemetry",
}
CONTINGENCY_FIELDS = {
    "both_correct",
    "regressions",
    "recoveries",
    "both_wrong",
}
AGREEMENT_FIELDS = {"agree", "disagree"}
TELEMETRY_ARMS = {"common", "promatch", "pinball"}


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    value = _nonnegative_int(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be numeric")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return alpha


def _cube(value: Any, *, shots: int) -> dict[str, int]:
    raw = _mapping(value, name="correctness_cube")
    expected = {f"{u}{p}{b}" for u in "01" for p in "01" for b in "01"}
    if set(raw) != expected:
        raise ValueError("correctness_cube must contain exactly 000 through 111")
    cube = {
        key: _nonnegative_int(raw[key], name=f"correctness_cube[{key!r}]")
        for key in sorted(expected)
    }
    if sum(cube.values()) != shots:
        raise ValueError("correctness_cube does not reconcile to shots")
    return cube


def _table_from_cube(
    cube: Mapping[str, int], *, treatment: str, baseline: str
) -> PairedContingency:
    treatment_index = ARM_ORDER.index(treatment)
    baseline_index = ARM_ORDER.index(baseline)
    cells = {
        "both_correct": 0,
        "regressions": 0,
        "recoveries": 0,
        "both_wrong": 0,
    }
    for bits, count in cube.items():
        treatment_failed = bits[treatment_index] == "1"
        baseline_failed = bits[baseline_index] == "1"
        if not baseline_failed and not treatment_failed:
            cells["both_correct"] += count
        elif not baseline_failed and treatment_failed:
            cells["regressions"] += count
        elif baseline_failed and not treatment_failed:
            cells["recoveries"] += count
        else:
            cells["both_wrong"] += count
    return PairedContingency(**cells)


def _validate_pairwise(
    value: Any, *, cube: Mapping[str, int], shots: int
) -> dict[str, PairedContingency]:
    raw = _mapping(value, name="pairwise_contingencies")
    if set(raw) != set(PAIR_DEFINITIONS):
        raise ValueError("pairwise_contingencies has incorrect pair fields")
    result: dict[str, PairedContingency] = {}
    for pair, (treatment, baseline) in PAIR_DEFINITIONS.items():
        item = _mapping(raw[pair], name=f"pairwise_contingencies.{pair}")
        if set(item) != CONTINGENCY_FIELDS:
            raise ValueError(f"pairwise_contingencies.{pair} has incorrect fields")
        table = PairedContingency(
            **{
                field: _nonnegative_int(
                    item[field], name=f"pairwise_contingencies.{pair}.{field}"
                )
                for field in CONTINGENCY_FIELDS
            }
        )
        if table.shots != shots:
            raise ValueError(f"pairwise_contingencies.{pair} does not reconcile")
        expected = _table_from_cube(
            cube, treatment=treatment, baseline=baseline
        )
        if table != expected:
            raise ValueError(
                f"pairwise_contingencies.{pair} disagrees with correctness_cube"
            )
        result[pair] = table
    return result


def _validate_agreement(value: Any, *, shots: int) -> dict[str, dict[str, int]]:
    raw = _mapping(value, name="prediction_agreement")
    if set(raw) != set(PAIR_DEFINITIONS):
        raise ValueError("prediction_agreement has incorrect pair fields")
    result: dict[str, dict[str, int]] = {}
    for pair in PAIR_DEFINITIONS:
        item = _mapping(raw[pair], name=f"prediction_agreement.{pair}")
        if set(item) != AGREEMENT_FIELDS:
            raise ValueError(f"prediction_agreement.{pair} has incorrect fields")
        counts = {
            field: _nonnegative_int(
                item[field], name=f"prediction_agreement.{pair}.{field}"
            )
            for field in AGREEMENT_FIELDS
        }
        if sum(counts.values()) != shots:
            raise ValueError(f"prediction_agreement.{pair} does not reconcile")
        result[pair] = counts
    return result


def _telemetry_leaf(value: Any, *, name: str) -> int | list[int] | dict[str, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} has an invalid boolean count")
    if isinstance(value, int):
        return _nonnegative_int(value, name=name)
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} count vector must be nonempty")
        return [
            _nonnegative_int(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        raw = _mapping(value, name=name)
        return {
            key: _nonnegative_int(item, name=f"{name}[{key!r}]")
            for key, item in sorted(raw.items())
        }
    raise ValueError(f"{name} must be an additive count, vector, or histogram")


def _validate_telemetry(
    value: Any, *, shots: int
) -> dict[str, dict[str, int | list[int] | dict[str, int]]]:
    raw = _mapping(value, name="telemetry")
    if set(raw) != TELEMETRY_ARMS:
        raise ValueError("telemetry must contain exactly common, promatch, and pinball")
    result: dict[str, dict[str, int | list[int] | dict[str, int]]] = {}
    for arm in ("common", "promatch", "pinball"):
        branch = _mapping(raw[arm], name=f"telemetry.{arm}")
        if not branch:
            raise ValueError(f"telemetry.{arm} must not be empty")
        validated = {
            key: _telemetry_leaf(item, name=f"telemetry.{arm}.{key}")
            for key, item in sorted(branch.items())
        }
        if validated.get("shots") != shots:
            raise ValueError(f"telemetry.{arm}.shots does not reconcile")
        result[arm] = validated
    common = result["common"]
    promatch = result["promatch"]
    pinball = result["pinball"]
    for branch, key in (
        (common, "original_event_sum"),
        (promatch, "residual_event_sum"),
        (pinball, "residual_event_sum"),
    ):
        if not isinstance(branch.get(key), int):
            raise ValueError(f"required scalar telemetry field {key!r} is missing")
    return result


def validate_cell_aggregate(value: Any) -> dict[str, Any]:
    """Validate and normalize one completed, additive cell aggregate."""

    raw = _mapping(value, name="cell aggregate")
    if set(raw) != CELL_FIELDS:
        raise ValueError("cell aggregate has incorrect fields")
    cell_id = raw["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a nonempty string")
    shots = _positive_int(raw["shots"], name="shots")
    cube = _cube(raw["correctness_cube"], shots=shots)
    tables = _validate_pairwise(
        raw["pairwise_contingencies"], cube=cube, shots=shots
    )
    agreement = _validate_agreement(raw["prediction_agreement"], shots=shots)
    telemetry = _validate_telemetry(raw["telemetry"], shots=shots)
    return {
        "cell_id": cell_id,
        "shots": shots,
        "correctness_cube": cube,
        "pairwise_contingencies": {
            pair: dataclasses.asdict(tables[pair]) for pair in PAIR_DEFINITIONS
        },
        "prediction_agreement": agreement,
        "telemetry": telemetry,
    }


def _sum_leaf(
    accumulator: int | list[int] | dict[str, int],
    value: int | list[int] | dict[str, int],
    *,
    name: str,
) -> int | list[int] | dict[str, int]:
    if isinstance(accumulator, int) and isinstance(value, int):
        return accumulator + value
    if isinstance(accumulator, list) and isinstance(value, list):
        if len(accumulator) != len(value):
            raise ValueError(f"{name} vectors have different lengths")
        return [a + b for a, b in zip(accumulator, value)]
    if isinstance(accumulator, dict) and isinstance(value, dict):
        result = dict(accumulator)
        for key, count in value.items():
            result[key] = result.get(key, 0) + count
        return dict(sorted(result.items()))
    raise ValueError(f"{name} changes additive telemetry type between ledgers")


def aggregate_cell_ledgers(
    ledgers: Iterable[Mapping[str, Any]],
    *,
    expected_cell_id: str | None = None,
    expected_shots: int | None = None,
) -> dict[str, Any]:
    """Sum validated per-batch projections for one completed campaign cell.

    Each iterable item uses :data:`CELL_FIELDS`.  Callers must authenticate
    rows and reject duplicate batch IDs before projecting them into this
    function.  ``expected_shots`` is required for a scientific completeness
    check; omitting it is useful only for smoke analysis.
    """

    rows = [validate_cell_aggregate(row) for row in ledgers]
    if not rows:
        raise ValueError("at least one ledger aggregate is required")
    cell_id = rows[0]["cell_id"]
    if any(row["cell_id"] != cell_id for row in rows):
        raise ValueError("ledger aggregates contain multiple cell IDs")
    if expected_cell_id is not None and cell_id != expected_cell_id:
        raise ValueError("ledger cell_id disagrees with expected_cell_id")

    result: dict[str, Any] = {
        "cell_id": cell_id,
        "shots": sum(row["shots"] for row in rows),
        "correctness_cube": {key: 0 for key in rows[0]["correctness_cube"]},
        "pairwise_contingencies": {
            pair: {field: 0 for field in CONTINGENCY_FIELDS}
            for pair in PAIR_DEFINITIONS
        },
        "prediction_agreement": {
            pair: {field: 0 for field in AGREEMENT_FIELDS}
            for pair in PAIR_DEFINITIONS
        },
        "telemetry": {
            arm: {
                key: (
                    0
                    if isinstance(value, int)
                    else [0] * len(value)
                    if isinstance(value, list)
                    else {}
                )
                for key, value in rows[0]["telemetry"][arm].items()
            }
            for arm in ("common", "promatch", "pinball")
        },
    }
    for row in rows:
        for key, count in row["correctness_cube"].items():
            result["correctness_cube"][key] += count
        for pair in PAIR_DEFINITIONS:
            for field in CONTINGENCY_FIELDS:
                result["pairwise_contingencies"][pair][field] += row[
                    "pairwise_contingencies"
                ][pair][field]
            for field in AGREEMENT_FIELDS:
                result["prediction_agreement"][pair][field] += row[
                    "prediction_agreement"
                ][pair][field]
        for arm in ("common", "promatch", "pinball"):
            if set(row["telemetry"][arm]) != set(result["telemetry"][arm]):
                raise ValueError(f"telemetry.{arm} fields change between ledgers")
            for key, value in row["telemetry"][arm].items():
                result["telemetry"][arm][key] = _sum_leaf(
                    result["telemetry"][arm][key],
                    value,
                    name=f"telemetry.{arm}.{key}",
                )
    if expected_shots is not None:
        expected_shots = _positive_int(expected_shots, name="expected_shots")
        if result["shots"] != expected_shots:
            raise ValueError(
                f"cell has {result['shots']} shots, expected {expected_shots}"
            )
    return validate_cell_aggregate(result)


def _marginal_interval(*, failures: int, shots: int, alpha: float) -> dict[str, Any]:
    return {
        "method": "clopper_pearson_exact",
        "confidence_level": 1 - alpha,
        "lower": clopper_pearson_lower(
            successes=failures, trials=shots, alpha=alpha / 2
        ),
        "upper": clopper_pearson_upper(
            successes=failures, trials=shots, alpha=alpha / 2
        ),
    }


def _paired_interval(table: PairedContingency, *, alpha: float) -> dict[str, Any]:
    swapped = PairedContingency(
        both_correct=table.both_correct,
        regressions=table.recoveries,
        recoveries=table.regressions,
        both_wrong=table.both_wrong,
    )
    return {
        "method": "tango_efficient_score",
        "confidence_level": 1 - alpha,
        "lower": -tango_paired_risk_difference_upper(swapped, alpha=alpha / 2),
        "upper": tango_paired_risk_difference_upper(table, alpha=alpha / 2),
    }


def _telemetry_summary(
    telemetry: Mapping[str, Mapping[str, Any]], *, shots: int
) -> dict[str, Any]:
    original = int(telemetry["common"]["original_event_sum"])
    residuals = {
        arm: int(telemetry[arm]["residual_event_sum"])
        for arm in ("promatch", "pinball")
    }
    workload_ratios = {
        f"{arm}_residual_over_original": (
            None if original == 0 else residuals[arm] / original
        )
        for arm in ("promatch", "pinball")
    }
    workload_ratios["pinball_residual_over_promatch"] = (
        None
        if residuals["promatch"] == 0
        else residuals["pinball"] / residuals["promatch"]
    )
    per_shot: dict[str, dict[str, float]] = {}
    for arm in ("promatch", "pinball"):
        per_shot[arm] = {
            key: value / shots
            for key, value in telemetry[arm].items()
            if isinstance(value, int) and key != "shots"
        }
    return {
        "workload_ratios": workload_ratios,
        "per_shot_scalar_telemetry": per_shot,
        "counts": telemetry,
    }


def analyze_cell(value: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """Emit reconciled marginal, paired, agreement, and workload statistics."""

    alpha = _validate_alpha(alpha)
    cell = validate_cell_aggregate(value)
    shots = cell["shots"]
    cube = cell["correctness_cube"]
    marginals: dict[str, Any] = {}
    for index, arm in enumerate(ARM_ORDER):
        failures = sum(count for bits, count in cube.items() if bits[index] == "1")
        marginals[arm] = {
            "failures": failures,
            "shots": shots,
            "any_observable_logical_error_rate": failures / shots,
            "confidence_interval": _marginal_interval(
                failures=failures, shots=shots, alpha=alpha
            ),
        }

    pairwise: dict[str, Any] = {}
    for pair, (treatment, baseline) in PAIR_DEFINITIONS.items():
        table = PairedContingency(**cell["pairwise_contingencies"][pair])
        agreement = cell["prediction_agreement"][pair]
        pairwise[pair] = {
            "treatment": treatment,
            "baseline": baseline,
            "contingency": dataclasses.asdict(table),
            "discordant": table.discordant,
            "paired_risk_difference": table.delta,
            "confidence_interval": _paired_interval(table, alpha=alpha),
            "prediction_agreement": {
                **agreement,
                "fraction": agreement["agree"] / shots,
            },
        }
    return {
        "cell_id": cell["cell_id"],
        "shots": shots,
        "correctness_cube": cube,
        "marginals": marginals,
        "pairwise": pairwise,
        "telemetry_summary": _telemetry_summary(cell["telemetry"], shots=shots),
    }


def build_json_report(
    cells: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str,
    alpha: float = 0.05,
    expected_cell_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a canonical, self-hashing JSON-ready campaign report."""

    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a nonempty string")
    alpha = _validate_alpha(alpha)
    analyzed = [analyze_cell(cell, alpha=alpha) for cell in cells]
    if not analyzed:
        raise ValueError("report must contain at least one cell")
    identifiers = [cell["cell_id"] for cell in analyzed]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("report contains duplicate cell IDs")
    if expected_cell_ids is not None:
        expected = list(expected_cell_ids)
        if len(set(expected)) != len(expected):
            raise ValueError("expected_cell_ids contains duplicates")
        if set(identifiers) != set(expected):
            raise ValueError("report cells do not match expected_cell_ids")
    analyzed.sort(key=lambda cell: cell["cell_id"])
    report: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": campaign_id,
        "confidence_level": 1 - alpha,
        "cells": analyzed,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def report_json_bytes(report: Mapping[str, Any]) -> bytes:
    """Serialize a report after validating its top-level identity and digest."""

    raw = _mapping(report, name="report")
    expected = {
        "schema",
        "campaign_id",
        "confidence_level",
        "cells",
        "report_sha256",
    }
    if set(raw) != expected or raw.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("report has incorrect top-level fields or schema")
    without_hash = dict(raw)
    recorded = without_hash.pop("report_sha256")
    expected_hash = hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest()
    if recorded != expected_hash:
        raise ValueError("report_sha256 does not reconcile")
    return canonical_json_bytes(raw)


def cell_csv_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one analyzed cell; telemetry counts remain canonical JSON text."""

    raw = _mapping(cell, name="analyzed cell")
    required = {
        "cell_id",
        "shots",
        "correctness_cube",
        "marginals",
        "pairwise",
        "telemetry_summary",
    }
    if set(raw) != required:
        raise ValueError("analyzed cell has incorrect fields")
    row: dict[str, Any] = {"cell_id": raw["cell_id"], "shots": raw["shots"]}
    marginals = _mapping(raw["marginals"], name="marginals")
    if set(marginals) != set(ARM_ORDER):
        raise ValueError("marginals has incorrect arm fields")
    for arm in ARM_ORDER:
        item = _mapping(marginals[arm], name=f"marginals.{arm}")
        interval = _mapping(item.get("confidence_interval"), name="confidence_interval")
        row[f"{arm}_failures"] = item.get("failures")
        row[f"{arm}_ler"] = item.get("any_observable_logical_error_rate")
        row[f"{arm}_ler_ci_lower"] = interval.get("lower")
        row[f"{arm}_ler_ci_upper"] = interval.get("upper")
    pairwise = _mapping(raw["pairwise"], name="pairwise")
    if set(pairwise) != set(PAIR_DEFINITIONS):
        raise ValueError("pairwise has incorrect pair fields")
    for pair in PAIR_DEFINITIONS:
        item = _mapping(pairwise[pair], name=f"pairwise.{pair}")
        interval = _mapping(item.get("confidence_interval"), name="confidence_interval")
        agreement = _mapping(
            item.get("prediction_agreement"), name="prediction_agreement"
        )
        row[f"{pair}_delta"] = item.get("paired_risk_difference")
        row[f"{pair}_ci_lower"] = interval.get("lower")
        row[f"{pair}_ci_upper"] = interval.get("upper")
        row[f"{pair}_prediction_agreement"] = agreement.get("fraction")
    summary = _mapping(raw["telemetry_summary"], name="telemetry_summary")
    ratios = _mapping(summary.get("workload_ratios"), name="workload_ratios")
    for key, value in sorted(ratios.items()):
        row[f"workload_{key}"] = value
    counts = _mapping(summary.get("counts"), name="telemetry counts")
    for arm in ("common", "promatch", "pinball"):
        row[f"{arm}_telemetry_json"] = json.dumps(
            counts[arm], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    return row


def report_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return stable CSV-ready rows from a validated JSON report."""

    report_json_bytes(report)
    cells = report.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("report cells must be a nonempty array")
    rows = [cell_csv_row(cell) for cell in cells]
    if any(set(row) != set(rows[0]) for row in rows[1:]):
        raise ValueError("CSV rows have inconsistent fields")
    return rows


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARM_ORDER",
    "PAIR_DEFINITIONS",
    "aggregate_cell_ledgers",
    "analyze_cell",
    "build_json_report",
    "cell_csv_row",
    "report_csv_rows",
    "report_json_bytes",
    "validate_cell_aggregate",
]
