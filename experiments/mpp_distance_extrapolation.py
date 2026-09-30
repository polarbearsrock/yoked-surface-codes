#!/usr/bin/env python3
"""Fit distance scaling to paired block-failure counts and project to d=21.

This analyzes completed accuracy experiments; it does not run decoders. The
models combine an exponential distance component with optional round-count
prefactors. For a distance-scaled round schedule, the primary model uses the
paper's 1D-yoked quadratic-round heuristic. Binomial likelihood weights the
observed counts correctly, including zero-count rows.
Paired whole-shot resampling preserves correlation between the two decoders.
Intervals describe sampling uncertainty under the fitted scaling assumption;
changing the fit window separately probes sensitivity to that assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import chi2


METHODS = ("complementary_gap", "mpp")
OUTCOMES = ("both_succeed", "mpp_repairs", "mpp_regressions", "both_fail")


def round_schedule(results, target):
    """Infer one consistent round policy and reject mixed experiment families."""
    counts = {int(d): row["parameters"]["rounds"] for d, row in results.items()}
    if len(set(counts.values())) == 1:
        fixed = next(iter(counts.values()))
        return {"fixed": fixed, "target_rounds": fixed}
    multipliers = {rounds / distance for distance, rounds in counts.items()}
    if len(multipliers) == 1:
        multiplier = next(iter(multipliers))
        if multiplier.is_integer():
            return {"per_distance": int(multiplier), "target_rounds": int(multiplier) * target}
    raise ValueError("Round counts mix fixed and distance-scaled schedules")


def fit_binomial(distances, shots, counts, initial=None, log_offset=None):
    """Fit many two-parameter log-link binomial models with vectorized Newton steps.

    Each row of counts is a separate fit. The observed information is positive
    definite for these data, so solving its 2x2 system avoids thousands of
    independent optimizer calls during bootstrap. Backtracking preserves p < 1
    and requires a decrease in negative log likelihood.
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    shots = np.asarray(shots, dtype=float)
    x = (np.asarray(distances, dtype=float) - 11) / 2
    offset = np.zeros_like(x) if log_offset is None else np.asarray(log_offset)
    if initial is None:
        # This initialization is only a starting guess, not the fitted result.
        design = np.column_stack((np.ones_like(x), x))
        initial = np.linalg.lstsq(design, np.log((counts[0] + 0.5) / (shots + 1)) - offset, rcond=None)[0]
    beta = np.broadcast_to(initial, (len(counts), 2)).copy()

    def objective(parameters):
        log_p = parameters[:, :1] + parameters[:, 1:] * x + offset
        valid = np.all(log_p < 0, axis=1)
        safe_log_p = np.minimum(log_p, -1e-14)
        nll = -(counts * safe_log_p + (shots - counts) * np.log(-np.expm1(safe_log_p))).sum(axis=1)
        return np.where(valid, nll, np.inf)

    for _ in range(100):
        p = np.exp(beta[:, :1] + beta[:, 1:] * x + offset)
        score = (counts - shots * p) / (1 - p)
        information = p * (shots - counts) / (1 - p) ** 2
        h00 = information.sum(axis=1)
        h01 = (information * x).sum(axis=1)
        h11 = (information * x * x).sum(axis=1)
        u0, u1 = score.sum(axis=1), (score * x).sum(axis=1)
        determinant = h00 * h11 - h01 * h01
        if np.any(determinant <= 0):
            raise ValueError("Unidentifiable distance fit")
        step = np.column_stack(((h11 * u0 - h01 * u1) / determinant,
                                (h00 * u1 - h01 * u0) / determinant))
        if np.max(np.abs(step)) < 1e-9:
            return beta
        old = objective(beta)
        for _ in range(40):
            candidate = beta + step
            worse = objective(candidate) > old + 1e-9
            if not worse.any():
                break
            step[worse] *= 0.5
        else:
            raise RuntimeError("Distance-fit line search failed")
        beta = candidate
    raise RuntimeError("Distance fit did not converge")


def check_fit_against_scipy(distances, shots, counts, beta, log_offset=None):
    """Independently check the numerical fit using a general-purpose optimizer."""
    x = (np.asarray(distances) - 11) / 2
    offset = np.zeros_like(x) if log_offset is None else np.asarray(log_offset)

    def objective(parameters):
        log_p = parameters[0] + parameters[1] * x + offset
        if np.any(log_p >= 0):
            return np.inf
        return -np.sum(counts * log_p + (shots - counts) * np.log(-np.expm1(log_p)))

    reference = minimize(objective, beta + [0.1, -0.05], method="Nelder-Mead",
                         options={"xatol": 1e-10, "fatol": 1e-9, "maxiter": 2000})
    if not reference.success:
        raise RuntimeError(reference.message)
    np.testing.assert_allclose(beta, reference.x, atol=2e-6, rtol=0)


def fit_window(results, distances, target, replicates, seed, round_power=0):
    """Resample within each distance, retaining paired decoder outcomes."""
    shots = np.array([results[str(d)]["shots"] for d in distances])
    counts = np.array([[results[str(d)][name]["block_failures"] for d in distances]
                       for name in METHODS])
    # Optional powers probe model uncertainty when rounds grow with distance.
    # They are explicit assumptions, not inferred physical scaling laws.
    reference_rounds = results["11"]["parameters"]["rounds"]
    rounds = np.array([results[str(d)]["parameters"]["rounds"] for d in distances])
    log_offset = round_power * np.log(rounds / reference_rounds)
    target_rounds = round_schedule(results, target)["target_rounds"]
    target_offset = round_power * np.log(target_rounds / reference_rounds)
    generator = np.random.default_rng(seed)
    sampled_counts = np.empty((2, replicates, len(distances)))
    for column, distance in enumerate(distances):
        paired = results[str(distance)]["paired"]
        cells = np.array([paired[name] for name in OUTCOMES])
        assert cells.sum() == shots[column]
        draws = generator.multinomial(shots[column], cells / shots[column], size=replicates)
        sampled_counts[0, :, column] = draws[:, 1] + draws[:, 3]
        sampled_counts[1, :, column] = draws[:, 2] + draws[:, 3]

    report = {"distances": list(map(int, distances)), "target_distance": target,
              "round_prefactor_power": round_power, "reference_rounds": reference_rounds}
    point_rates, bootstrap_rates = [], []
    for index, name in enumerate(METHODS):
        beta = fit_binomial(distances, shots, counts[index], log_offset=log_offset)[0]
        check_fit_against_scipy(distances, shots, counts[index], beta, log_offset=log_offset)
        bootstrap_beta = fit_binomial(distances, shots, sampled_counts[index], initial=beta, log_offset=log_offset)
        target_x = (target - 11) / 2
        target_rate = float(np.exp(beta @ [1, target_x] + target_offset))
        draws = np.exp(bootstrap_beta @ [1, target_x] + target_offset)
        point_rates.append(target_rate)
        bootstrap_rates.append(draws)
        fitted = np.exp(beta[0] + beta[1] * (np.asarray(distances) - 11) / 2 + log_offset)
        expected = shots * fitted
        # Binomial deviance compares the distance model with one free rate per
        # distance. Its chi-squared reference is an approximate fit diagnostic.
        observed = counts[index]
        deviance = 2 * np.sum(xlogy(observed, observed / expected)
                             + xlogy(shots - observed, (shots - observed) / (shots - expected)))
        report[name] = {
            "intercept_at_d11": float(beta[0]), "slope_per_two_distance_units": float(beta[1]),
            "suppression_per_two_distance_units": float(np.exp(-beta[1])),
            "predicted_block_ler": target_rate,
            "predicted_block_ler_ci95_sampling": np.percentile(draws, [2.5, 97.5]).tolist(),
            "fitted_observed_distance_rates": fitted.tolist(),
            "binomial_deviance": float(deviance), "deviance_degrees_of_freedom": len(distances) - 2,
            "deviance_approximate_p": float(chi2.sf(deviance, len(distances) - 2)),
            "scipy_optimizer_agreement": True,
        }
    ratio = point_rates[1] / point_rates[0]
    ratio_draws = bootstrap_rates[1] / bootstrap_rates[0]
    report["paired_projection"] = {
        "mpp_to_gap_ratio": ratio,
        "ratio_ci95_sampling": np.percentile(ratio_draws, [2.5, 97.5]).tolist(),
        "relative_increase_percent": 100 * (ratio - 1),
        "relative_increase_ci95_sampling_percent": (100 * (np.percentile(ratio_draws, [2.5, 97.5]) - 1)).tolist(),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Completed run directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=21)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--expect-rounds-per-distance", type=int,
                        help="require this round multiplier, e.g. 4 for the corrected sweep")
    args = parser.parse_args()
    results, sources = {}, []
    provenance = None
    for directory in args.input:
        run = json.loads((directory / "run.json").read_text())
        comparable = {key: run["provenance"][key] for key in ("source_sha256", "native_build", "versions")}
        if provenance is not None and comparable != provenance:
            raise ValueError("Input runs used different decoder sources, builds, or dependencies")
        provenance = comparable
        payload = (directory / "summary.json").read_bytes()
        summary = json.loads(payload)
        if results.keys() & summary.keys():
            raise ValueError("Duplicate distance in input runs")
        results.update(summary)
        sources.append({"directory": str(directory.resolve()),
                        "summary_sha256": hashlib.sha256(payload).hexdigest()})
    settings = [{k: v for k, v in row["parameters"].items() if k not in ("distance", "rounds")}
                for row in results.values()]
    if any(setting != settings[0] for setting in settings):
        raise ValueError("Circuit settings differ beyond distance and the round schedule")
    schedule = round_schedule(results, args.target)
    if args.expect_rounds_per_distance is not None and schedule.get("per_distance") != args.expect_rounds_per_distance:
        raise ValueError("Measured rounds do not match the requested distance multiplier")
    distances = sorted(map(int, results))
    if distances != [7, 9, 11, 13, 15]:
        raise ValueError("This report expects the d=7,9,11,13,15 experiment")
    report = {
        "model": "log(block LER) = a + b*(distance-11)/2 + power*log(rounds/rounds_at_d11); binomial likelihood",
        "parameters_held_fixed": settings[0], "target_distance": args.target,
        "round_schedule": schedule,
        "primary_round_prefactor_power": 2 if "per_distance" in schedule else 0,
        "bootstrap": {"replicates": args.replicates, "seed": 20260923,
                      "unit": "whole paired shot, independently within each distance"},
        "interval_scope": "Sampling uncertainty conditional on log-linear distance scaling; excludes model bias",
        "windows": [fit_window(results, distances[start:], args.target, args.replicates, 20260923)
                    for start in range(3)],
    }
    if "per_distance" in schedule:
        report["round_prefactor_sensitivity"] = {
            "model": "log(block LER) = a + b*(d-11)/2 + power*log(rounds/rounds_at_d11)",
            "scope": "Power 2 follows the paper's 1D-yoked path-counting heuristic; alternatives probe model uncertainty. Applicability to these data, especially MPP, remains an assumption.",
            "source": {"url": "https://arxiv.org/pdf/2312.04522", "section": "5.1", "model": "block error scales as r^2 * n^2 * lambda^(-d); n is fixed here"},
            "fits": [fit_window(results, distances[start:], args.target, args.replicates, 20260923, round_power=power)
                     for power in (1, 2) for start in range(3)],
        }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (("comparison_summary.json", results), ("comparison_sources.json", sources),
                        ("extrapolation.json", report)):
        (args.out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
