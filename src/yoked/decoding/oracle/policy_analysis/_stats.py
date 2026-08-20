"""Deterministic statistics: quantiles, ECDFs, and shot-clustered bootstrap.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` implements the
protocol's exact type-7 quantiles, exact ECDF points, domain-separated
bootstrap seeding, the complete-shot clustered bootstrap, and the paired
contingency table.  It inherits the package's downstream-only contract: it
never imports circuit generation, sampling, matching, or decoding code (the
paired-statistics import from :mod:`yoked.decoding._promatch_stats` predates
the split and carries its own sync note).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from yoked.decoding._promatch_stats import (
    PairedContingency,
    tango_paired_risk_difference_upper,
)

from ._contract import SPARSE_UNSAFE_STATES, PolicyAnalysisError
from ._fields import _as_nonnegative_int, _at, _one_deep


def empirical_type7(values: Sequence[float], probability: float) -> float | None:
    """Exact public wrapper for the protocol's empirical type-7 quantile."""

    if not 0 <= probability <= 1 or not math.isfinite(probability):
        raise ValueError("probability must be finite and in [0, 1]")
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("quantile values must be a finite one-dimensional sequence")
    return float(np.quantile(array, probability, method="linear"))


def exact_ecdf(values: Sequence[float]) -> list[dict[str, Any]]:
    """Returns every unique finite value and its exact empirical CDF point."""

    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("ECDF values must be finite and one-dimensional")
    unique, counts = np.unique(array, return_counts=True)
    cumulative = np.cumsum(counts)
    denominator = len(array)
    return [
        {
            "value": float(value),
            "count": int(count),
            "cumulative_count": int(cum),
            "cumulative_fraction": float(cum / denominator),
            "denominator": denominator,
        }
        for value, count, cum in zip(unique, counts, cumulative)
    ]


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    """Summarizes a finite sample using the protocol's fixed quantiles and ECDF."""

    data = [float(v) for v in values]
    return {
        "denominator": len(data),
        "median": empirical_type7(data, 0.5),
        "p10": empirical_type7(data, 0.1),
        "p90": empirical_type7(data, 0.9),
        "p99": empirical_type7(data, 0.99) if len(data) >= 1000 else None,
        "maximum": max(data) if data else None,
        "ecdf": exact_ecdf(data),
    }


def derive_bootstrap_seed(seed_root: str, *, cell_id: str, estimand: str) -> int:
    """Derives a domain-separated 128-bit seed for one cell and estimand."""

    if not isinstance(seed_root, str) or not seed_root:
        raise PolicyAnalysisError("bootstrap seed root must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(b"promatch-b1-shot-bootstrap-v1\0")
    for value in (seed_root, cell_id, estimand):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:16], "little")


def clustered_bootstrap_ratios(
    contributions: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Bootstraps ratio-of-sums columns while preserving complete shots.

    ``contributions`` has columns ``num0, den0, num1, den1, ...``.  One row is
    one physical shot and therefore carries every proposal/domain contribution.
    """

    array = np.asarray(contributions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] % 2:
        raise ValueError("contributions must have an even number of columns")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("contributions must contain finite rows")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ValueError("replicates must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    metrics = array.shape[1] // 2
    draws: list[list[float]] = [[] for _ in range(metrics)]
    undefined = [0] * metrics
    # For one ratio, an ordinary complete-shot bootstrap depends only on the
    # empirical histogram of that shot's (numerator, denominator) pair.  A
    # multinomial draw from this histogram is exactly the same marginal
    # bootstrap distribution as gathering N sampled shot rows, while avoiding
    # O(replicates * shots * metrics) index and gather work.  Metrics are drawn
    # independently because the public product contains only marginal
    # intervals; no joint/covariance claim is made from these replicates.
    for metric in range(metrics):
        pairs = array[:, 2 * metric : 2 * metric + 2]
        unique_pairs, frequencies = np.unique(pairs, axis=0, return_counts=True)
        probabilities = frequencies.astype(np.float64) / array.shape[0]
        remaining = replicates
        while remaining:
            max_chunk = max(1, 4_000_000 // max(1, len(unique_pairs)))
            chunk = min(256, max_chunk, remaining)
            multiplicities = rng.multinomial(array.shape[0], probabilities, size=chunk)
            totals = multiplicities @ unique_pairs
            denominator = totals[:, 1]
            defined = denominator != 0
            undefined[metric] += int((~defined).sum())
            draws[metric].extend(
                (totals[defined, 0] / denominator[defined]).astype(float).tolist()
            )
            remaining -= chunk
    result = []
    for metric, samples in enumerate(draws):
        result.append(
            {
                "replicates": replicates,
                "defined_replicates": len(samples),
                "undefined_replicates": undefined[metric],
                "lower": empirical_type7(samples, alpha / 2),
                "upper": empirical_type7(samples, 1 - alpha / 2),
                # Historical name: this key currently carries the same
                # two-sided (1 - alpha/2) upper bound as "upper", not a
                # (1 - alpha) one-sided bound.  Do not change the emitted
                # value; artifacts and reports pin both keys.
                "upper_one_sided": empirical_type7(samples, 1 - alpha / 2),
                "quantile_method": "empirical-type-7",
            }
        )
    return result


def _paired_table(
    baseline: Sequence[bool], treatment: Sequence[bool]
) -> dict[str, Any]:
    if len(baseline) != len(treatment):
        raise PolicyAnalysisError("paired outcome vectors differ in length")
    both_correct = regressions = recoveries = both_wrong = 0
    for base_failed, treatment_failed in zip(baseline, treatment):
        if not base_failed and not treatment_failed:
            both_correct += 1
        elif not base_failed and treatment_failed:
            regressions += 1
        elif base_failed and not treatment_failed:
            recoveries += 1
        else:
            both_wrong += 1
    table = PairedContingency(both_correct, regressions, recoveries, both_wrong)
    return {
        "shots": table.shots,
        "both_correct": both_correct,
        "regressions": regressions,
        "recoveries": recoveries,
        "both_wrong": both_wrong,
        "paired_risk_difference": table.delta,
        "tango_upper_one_sided_97_5": tango_paired_risk_difference_upper(
            table, alpha=0.025
        ),
    }


def _bootstrap_config(
    config: Mapping[str, Any], *, family: Literal["proposal", "workload"]
) -> tuple[int, str]:
    replicates = _one_deep(config, ("bootstrap_replicates", "replicates"))
    if replicates is None:
        replicates = 10_000
    replicates = _as_nonnegative_int(replicates, name="bootstrap replicates")
    if replicates == 0:
        raise PolicyAnalysisError("bootstrap replicates must be positive")
    seed_root = _at(config, f"bootstrap.seed_roots.{family}")
    if seed_root is None:
        names = (
            f"{family}_bootstrap_seed_root",
            "policy_bootstrap_seed_root",
            "bootstrap_seed_root",
        )
        seed_root = _one_deep(config, names)
    if not isinstance(seed_root, str) or not seed_root:
        raise PolicyAnalysisError("config lacks a bootstrap seed root")
    return replicates, seed_root


def _bootstrap_fraction_table(
    *,
    shot_keys: Sequence[tuple[str, int]],
    categories: Sequence[str],
    observations: Sequence[tuple[tuple[str, int], str, bool]],
    replicates: int,
    seed_root: str,
    cell_id: str,
    estimand: str,
) -> list[dict[str, Any]]:
    index = {key: offset for offset, key in enumerate(shot_keys)}
    category_index = {value: offset for offset, value in enumerate(categories)}
    matrix = np.zeros((len(shot_keys), 2 * len(categories)), dtype=np.float64)
    for shot_key, category, numerator in observations:
        if shot_key not in index or category not in category_index:
            raise PolicyAnalysisError("bootstrap observation references an unknown key")
        row = index[shot_key]
        column = category_index[category]
        matrix[row, 2 * column + 1] += 1
        matrix[row, 2 * column] += int(numerator)
    intervals = clustered_bootstrap_ratios(
        matrix,
        replicates=replicates,
        seed=derive_bootstrap_seed(seed_root, cell_id=cell_id, estimand=estimand),
    )
    result = []
    for position, category in enumerate(categories):
        numerator = int(matrix[:, 2 * position].sum())
        denominator = int(matrix[:, 2 * position + 1].sum())
        result.append(
            {
                "category": category,
                "numerator": numerator,
                "denominator": denominator,
                "fraction": None if denominator == 0 else numerator / denominator,
                "bootstrap": intervals[position],
                "stratum_status": (
                    "insufficient-for-rule-formulation"
                    if numerator < SPARSE_UNSAFE_STATES
                    else "descriptive-only"
                ),
            }
        )
    return result
