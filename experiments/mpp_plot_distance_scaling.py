#!/usr/bin/env python3
"""Plot measured block LERs and the conditional d=21 distance projection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--min-distance", type=int, choices=(7, 9, 11), default=7)
    parser.add_argument("--round-power", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    directory = args.result
    results = json.loads((directory / "comparison_summary.json").read_text())
    projection = json.loads((directory / "extrapolation.json").read_text())
    power = (projection.get("primary_round_prefactor_power", 0)
             if args.round_power is None else args.round_power)
    candidates = (projection["windows"] if power == 0
                  else projection["round_prefactor_sensitivity"]["fits"])
    fit = next(row for row in candidates
               if row.get("round_prefactor_power", 0) == power and row["distances"][0] == args.min_distance)
    target = projection["target_distance"]
    distances = np.array(sorted(map(int, results)))
    settings = projection["parameters_held_fixed"]
    schedule = (projection["round_schedule"] if "round_schedule" in projection
                else {"fixed": settings["rounds"]})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1.3, 1])

    for name, label, color, marker_offset, marker in (
        ("complementary_gap", "Complementary gap", "#1f77b4", -0.08, "o"),
        ("mpp", "MPP cluster score", "#d55e00", 0.08, "s"),
    ):
        rates = np.array([results[str(d)][name]["block_failure_rate"] for d in distances])
        intervals = np.array([results[str(d)][name]["block_failure_ci95_exact"] for d in distances])
        axes[0].errorbar(
            distances + marker_offset, rates,
            yerr=[rates - intervals[:, 0], intervals[:, 1] - rates],
            fmt=marker, color=color, capsize=3, label=label, zorder=3,
        )
        # Solid curves cover measured distances; dashed curves show the part
        # supported by the extrapolation assumption alone.
        for x, style in ((np.linspace(args.min_distance, distances.max(), 100), "-"),
                         (np.linspace(distances.max(), target, 100), "--")):
            rounds = schedule["per_distance"] * x if "per_distance" in schedule else schedule["fixed"]
            round_offset = power * np.log(rounds / fit.get("reference_rounds", rounds))
            y = np.exp(fit[name]["intercept_at_d11"]
                       + fit[name]["slope_per_two_distance_units"] * (x - 11) / 2 + round_offset)
            axes[0].plot(x, y, style, color=color, linewidth=1.2, alpha=0.7)
        rate = fit[name]["predicted_block_ler"]
        low, high = fit[name]["predicted_block_ler_ci95_sampling"]
        axes[0].errorbar(
            target + marker_offset, rate, yerr=[[rate - low], [high - rate]],
            fmt="D", markerfacecolor="white", color=color, capsize=3, zorder=3,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Block logical failure probability")
    axes[0].set_title("Observed LER and distance fits" + (f" ($r^{power}$ prefactor)" if power else ""))
    axes[0].legend(frameon=False, fontsize=9)

    changes = np.array([results[str(d)]["paired"]["relative_degradation_percent"] for d in distances])
    intervals = 100 * (np.array([
        results[str(d)]["paired"]["rate_ratio_ci95_paired_bootstrap"] for d in distances
    ]) - 1)
    # Empirical resampling cannot invent an unobserved repair or regression.
    # If either is absent, show only the estimate here and refer to the exact
    # paired test instead of drawing a misleadingly one-sided interval.
    enough_discordant = np.array([
        min(results[str(d)]["paired"]["mpp_repairs"],
            results[str(d)]["paired"]["mpp_regressions"]) > 0 for d in distances
    ])
    axes[1].errorbar(
        distances[enough_discordant], changes[enough_discordant],
        yerr=[(changes - intervals[:, 0])[enough_discordant],
              (intervals[:, 1] - changes)[enough_discordant]],
        fmt="s", color="#d55e00", capsize=3, label="Measured",
    )
    if (~enough_discordant).any():
        axes[1].plot(distances[~enough_discordant], changes[~enough_discordant],
                     "*", color="#d55e00", markersize=10, label="Sparse paired outcomes")
    relative = fit["paired_projection"]
    change = relative["relative_increase_percent"]
    low, high = relative["relative_increase_ci95_sampling_percent"]
    axes[1].errorbar(
        target, change, yerr=[[change - low], [high - change]],
        fmt="D", markerfacecolor="white", color="#d55e00", capsize=3,
        label="d=21 projection",
    )
    axes[1].axhline(0, color="0.45", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Relative LER increase with MPP (%)")
    axes[1].set_title("Paired 95% bootstrap intervals")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.axvspan(distances.max() + 0.5, target + 0.5, color="0.95", zorder=0)
        axis.set_xticks([*distances, target])
        axis.set_xlabel("Code distance")
        axis.set_xlim(distances.min() - 0.6, target + 0.6)
        axis.grid(axis="y", alpha=0.2)
    rounds_label = (f"{schedule['per_distance']}d rounds" if "per_distance" in schedule
                    else f"{schedule['fixed']} rounds")
    shots = results[str(distances[0])]["shots"]
    figure.suptitle(f"SI1000 p = {settings['p']} · {settings['patches']} patches · "
                   f"{rounds_label} · {shots:,} paired shots per measured distance")
    figure.text(0.5, 0.04,
                f"Open diamonds: d={target} extrapolation from d={args.min_distance}–15. Intervals exclude uncertainty in the scaling model.",
                ha="center", fontsize=9)
    if (~enough_discordant).any():
        sparse = ", ".join(f"d={d}" for d in distances[~enough_discordant])
        figure.text(0.5, 0.005, f"Paired intervals omitted for sparse outcomes at {sparse}; see exact paired tests in the report.",
                    ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(directory / "distance_scaling.png", dpi=180)
    figure.savefig(directory / "distance_scaling.pdf")


if __name__ == "__main__":
    main()
