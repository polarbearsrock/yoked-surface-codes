#!/usr/bin/env python3
"""Plot the measured MPP accuracy penalty and its saved distance projection.

This only visualizes the completed experiment. It uses the existing binomial
fits and paired bootstrap intervals; it does not refit or simulate any shots.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def relative_increase(fit, distances):
    """Evaluate 100 * (MPP LER / gap LER - 1) from the two saved fits.

    Both fits have the same round prefactor, which cancels in their ratio.
    Subtracting log rates also avoids unnecessary exponentiation of small LERs.
    """
    gap, mpp = fit["complementary_gap"], fit["mpp"]
    log_ratio = (
        mpp["intercept_at_d11"] - gap["intercept_at_d11"]
        + (mpp["slope_per_two_distance_units"] - gap["slope_per_two_distance_units"])
        * (np.asarray(distances) - 11) / 2
    )
    return 100 * np.expm1(log_ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-confidence-intervals", action="store_true",
                        help="show point estimates without confidence-interval bars")
    args = parser.parse_args()
    show_intervals = not args.no_confidence_intervals
    summary_path = args.result / "comparison_summary.json"
    projection_path = args.result / "extrapolation.json"
    results = json.loads(summary_path.read_text())
    projection = json.loads(projection_path.read_text())
    power = projection["primary_round_prefactor_power"]
    candidates = (projection["windows"] if power == 0
                  else projection["round_prefactor_sensitivity"]["fits"])
    distances = np.array(sorted(map(int, results)))
    fits = [row for row in candidates if row["round_prefactor_power"] == power]
    fit = next(row for row in fits if row["distances"] == distances.tolist())
    target = projection["target_distance"]
    target_rounds = projection["round_schedule"]["target_rounds"]
    paired = fit["paired_projection"]
    estimate = paired["relative_increase_percent"]
    low, high = paired["relative_increase_ci95_sampling_percent"]
    # Ensure the drawn curve reaches the reported paired projection exactly.
    np.testing.assert_allclose(relative_increase(fit, target), estimate, atol=1e-10, rtol=0)

    measured = np.array([results[str(d)]["paired"]["relative_degradation_percent"]
                         for d in distances])
    intervals = 100 * (np.array([
        results[str(d)]["paired"]["rate_ratio_ci95_paired_bootstrap"]
        for d in distances
    ]) - 1)
    # Size and typography are set at the final single-column publication size.
    # Experimental details and uncertainty qualifications belong in the caption.
    measured_color = "#17365d"
    projected_color = "#278f83"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.linewidth": 0.65,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, axis = plt.subplots(figsize=(3.55, 2.45))
    figure.subplots_adjust(left=0.16, right=0.975, bottom=0.19, top=0.965)
    axis.errorbar(
        distances, measured,
        yerr=([measured - intervals[:, 0], intervals[:, 1] - measured]
              if show_intervals else None),
        fmt="o", color=measured_color, markersize=4.2, markeredgewidth=0.7,
        capsize=2.2, elinewidth=0.8, zorder=4,
    )
    # A solid line marks the fitted measured range. The dashed continuation
    # beyond the largest simulated distance is explicitly an extrapolation.
    for start, stop, style, color in (
        (distances[0], distances[-1], "-", measured_color),
        (distances[-1], target, "--", projected_color),
    ):
        x = np.linspace(start, stop, 150)
        axis.plot(x, relative_increase(fit, x), style, color=color,
                  linewidth=1.2, zorder=2)
    axis.errorbar(
        target, estimate,
        yerr=[[estimate - low], [high - estimate]] if show_intervals else None,
        fmt="D", color=projected_color, markerfacecolor="white", markeredgewidth=1,
        markersize=4.8, capsize=2.5, elinewidth=0.9, zorder=5,
    )
    axis.annotate(
        f"+{estimate:.1f}%", xy=(target, estimate), xytext=(-7, 7),
        textcoords="offset points", ha="right", va="bottom", fontsize=8,
        color=projected_color, fontweight="bold", zorder=6,
    )
    axis.set(xlim=(distances[0] - 0.7, target + 0.8), ylim=(0, 25),
             xlabel="Code distance, $d$", ylabel="Relative LER increase (%)")
    axis.set_xticks(np.arange(distances[0], target + 1, 2))
    axis.set_yticks(np.arange(0, 26, 5))
    axis.grid(color="#d9d9d9", linewidth=0.4)
    axis.set_axisbelow(True)
    # Explicit handles combine the dashed curve and hollow prediction marker
    # into one legend entry, leaving the error bars to the caption.
    handles = [
        Line2D([], [], color=measured_color, marker="o", linestyle="none",
               markersize=4.2, label="Measured"),
        Line2D([], [], color=measured_color, linewidth=1.2, label="Fit"),
        Line2D([], [], color=projected_color, linestyle="--", linewidth=1.2,
               marker="D", markerfacecolor="white", markersize=4, label="Extrapolated"),
    ]
    legend = axis.legend(handles=handles, loc="upper left", fontsize=7,
                         frameon=True, fancybox=False, facecolor="white", framealpha=1,
                         edgecolor="#cccccc", borderpad=0.4, labelspacing=0.3,
                         handlelength=2, handletextpad=0.6)
    legend.get_frame().set_linewidth(0.5)

    settings = projection["parameters_held_fixed"]
    multiplier = projection["round_schedule"]["per_distance"]
    shots = results[str(distances[0])]["shots"]
    alternatives = [row["paired_projection"]["relative_increase_percent"] for row in fits]
    interval_caption = (
        f"Error bars show paired 95% bootstrap intervals; the d={target} interval is "
        f"{low:.1f}%–{high:.1f}%, conditional on the scaling model. These intervals exclude "
        f"model uncertainty. "
    ) if show_intervals else ""
    caption = (
        f"Relative block LER increase with MPP confidence over complementary gaps, "
        f"100 × (LER_MPP / LER_gap − 1), for SI1000 p={100 * settings['p']:g}%, "
        f"{settings['patches']} patches, and {multiplier}d rounds. Each measured distance uses "
        f"{shots:,} paired shots. The solid curve is the ratio of the saved r^{power} × "
        f"exponential distance fits over d={distances[0]}–{distances[-1]}; the dashed curve "
        f"extrapolates to d={target} ({target_rounds} rounds), giving a {estimate:.1f}% increase. "
        f"{interval_caption}Alternative fit windows give central projections of "
        f"{min(alternatives):.1f}%–{max(alternatives):.1f}%.\n"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.out / f"relative_increase_d{target}.{extension}", dpi=450)
    (args.out / "caption.txt").write_text(caption)
    shutil.copy2(__file__, args.out / "plot_recipe.py")
    provenance = {
        "source_directory": str(args.result.resolve()),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (summary_path, projection_path)},
        "fit_distances": fit["distances"], "round_prefactor_power": power,
        "target_distance": target, "target_rounds": target_rounds,
        "relative_increase_percent": estimate,
        "paired_ci95_sampling_percent": [low, high],
        "fit_window_sensitivity_percent": [min(alternatives), max(alternatives)],
        "interval_scope": "Sampling uncertainty conditional on the saved distance model",
        "confidence_intervals_shown": show_intervals,
        "figure_inches": [3.55, 2.45], "raster_dpi": 450,
        "style_references": [
            {"paper": "Astrea", "figure": 12,
             "url": "https://fast.cc.gatech.edu/papers/ISCA_2023_1.pdf#page=10"},
            {"paper": "Promatch", "figure": 14,
             "url": "https://www.poulamidas.com/assets/pdf/PROMATCH_ASPLOS_24.pdf#page=12"},
        ],
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
