#!/usr/bin/env python3
"""Compare measured LER and relative accuracy penalties for completed distances.

The percentage differences use the same effective per-round, per-patch LER as
the absolute plot. Ratios of raw block failure rates would give different
numbers, particularly at the smaller distances. No decoding is performed here.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys

from plot_hierarchical_ler import (
    SERIES, draw_slide_figure, read_results, sha256, shot_label,
)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter
import numpy as np
import sinter


BASELINE, MPP, UF = [series[0] for series in SERIES]
FIGURE_NAME = "decoder_comparison"
ABSOLUTE_FIGURE_NAME = "hierarchical_ler_per_patch_round"
RELATIVE_FIGURE_NAME = "relative_ler_increase"
LABELS = (
    "Correlated MWPM\nComplementary gap",
    "Correlated MWPM\nMPP cluster score",
    "Correlated UF\nCluster gap",
)


def compare_rates(rows):
    """Keep all pairwise comparisons, with the denominator named explicitly."""
    comparisons = []
    for distance in sorted({row["distance"] for row in rows}):
        rates = {
            row["configuration"]: row["ler_per_patch_round"]
            for row in rows if row["distance"] == distance
        }
        if min(rates.values()) <= 0:
            raise ValueError("This relative-comparison plot requires positive measured LERs")
        for numerator, denominator in ((MPP, BASELINE), (UF, BASELINE), (UF, MPP)):
            ratio = rates[numerator] / rates[denominator]
            comparisons.append(dict(
                distance=distance, numerator=numerator, denominator=denominator,
                ler_ratio=ratio, ler_increase_percent=100 * (ratio - 1),
                absolute_ler_difference=rates[numerator] - rates[denominator],
            ))
    return comparisons


def draw_figure(rows, comparisons, out):
    """Show absolute accuracy beside its percentage penalty on a slide canvas."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 14, "axes.linewidth": 0.8,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, (absolute, relative) = plt.subplots(1, 2, figsize=(12.8, 7.2))
    figure.subplots_adjust(left=0.09, right=0.975, bottom=0.18, top=0.72, wspace=0.30)
    distances = sorted({row["distance"] for row in rows})

    for series, label in zip(SERIES, LABELS):
        name, _, _, color, marker, line, size = series
        points = sorted(
            (row for row in rows if row["configuration"] == name),
            key=lambda row: row["distance"],
        )
        style = dict(
            color=color, marker=marker, linestyle=line, linewidth=2,
            markersize=size * 1.55, markerfacecolor="white", markeredgewidth=1.6,
        )
        absolute.plot(
            [row["distance"] for row in points],
            [row["ler_per_patch_round"] for row in points],
            label=label, zorder=3, **style,
        )
        if name == BASELINE:
            continue
        penalties = [
            row for row in comparisons
            if row["numerator"] == name and row["denominator"] == BASELINE
        ]
        relative.plot(
            [row["distance"] for row in penalties],
            [row["ler_increase_percent"] for row in penalties],
            zorder=3, **style,
        )
        for row in penalties:
            relative.annotate(
                f"+{row['ler_increase_percent']:.1f}%",
                (row["distance"], row["ler_increase_percent"]),
                xytext=(0, 11), textcoords="offset points",
                ha="center", va="bottom", fontsize=11.5, color=color,
            )

    rates = [row["ler_per_patch_round"] for row in rows]
    absolute.set(
        yscale="log", ylabel="LER per round per patch",
        ylim=(min(rates) / 1.25, max(rates) * 1.20),
    )
    absolute.set_title("(a) Decoder accuracy", loc="left", fontsize=14, pad=15)
    absolute.yaxis.set_major_locator(LogLocator(base=10))
    absolute.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7)))
    absolute.yaxis.set_minor_formatter(NullFormatter())
    absolute.tick_params(axis="y", which="minor", length=0)
    absolute.grid(axis="y", which="minor", color="#edf0f3", linewidth=0.45)

    maximum_penalty = max(
        row["ler_increase_percent"] for row in comparisons if row["denominator"] == BASELINE
    )
    relative.set(ylabel="LER increase (%)", ylim=(-8, maximum_penalty + 18))
    relative.set_title("(b) Increase over complementary gap", loc="left", fontsize=14, pad=15)
    relative.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    relative.set_yticks(np.arange(0, maximum_penalty + 1, 25))
    relative.axhline(0, color=SERIES[0][3], linewidth=1, linestyle=":", zorder=2)

    for axis in (absolute, relative):
        axis.set(xlabel="Code distance, $d$", xlim=(distances[0] - 0.5, distances[-1] + 0.5))
        axis.set_xticks(distances)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", which="major", pad=7)
        axis.grid(axis="y", which="major", color="#dce1e7", linewidth=0.65)
        axis.set_axisbelow(True)

    figure.legend(
        *absolute.get_legend_handles_labels(), loc="upper center",
        bbox_to_anchor=(0.54, 0.88), ncol=3, fontsize=12.5,
        frameon=False, handlelength=2.5, columnspacing=3.0,
    )
    figure.text(
        0.54, 0.96,
        rf"SI1000 $p=0.3\%$  ·  $4d$ rounds  ·  {shot_label(rows[0]['shots'])} shots per distance",
        ha="center", va="top", fontsize=15,
    )
    figure.text(
        0.54, 0.065, "L2: plain MWPM  ·  6 patches  ·  2 ideal yokes",
        ha="center", va="bottom", fontsize=11, color="#4b5563",
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(out / f"{FIGURE_NAME}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def draw_relative_figure(rows, comparisons, out):
    """Export the percentage penalty separately, with its baseline at zero."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 15, "axes.linewidth": 0.8,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.18, top=0.75)
    axis.axhline(
        0, color=SERIES[0][3], linewidth=1.2, linestyle=":",
        label="Correlated MWPM\nComplementary gap (baseline)", zorder=2,
    )
    for series, label in zip(SERIES[1:], LABELS[1:]):
        name, _, _, color, marker, line, size = series
        points = [
            row for row in comparisons
            if row["numerator"] == name and row["denominator"] == BASELINE
        ]
        axis.plot(
            [row["distance"] for row in points],
            [row["ler_increase_percent"] for row in points],
            label=label, color=color, marker=marker, linestyle=line,
            markersize=size * 1.7, markerfacecolor="white",
            markeredgewidth=1.6, linewidth=2, zorder=3,
        )
        for row in points:
            axis.annotate(
                f"+{row['ler_increase_percent']:.1f}%",
                (row["distance"], row["ler_increase_percent"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=11.5, color=color,
            )
    distances = sorted({row["distance"] for row in rows})
    maximum = max(
        row["ler_increase_percent"] for row in comparisons
        if row["denominator"] == BASELINE
    )
    axis.set(
        xlabel="Code distance, $d$", ylabel="LER increase over baseline (%)",
        xlim=(distances[0] - 0.4, distances[-1] + 0.55), ylim=(-12, maximum + 28),
    )
    axis.set_xticks(distances)
    axis.set_yticks(np.arange(0, 25 * np.ceil(maximum / 25) + 1, 25))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", which="major", pad=7)
    axis.grid(axis="y", which="major", color="#dce1e7", linewidth=0.65)
    axis.set_axisbelow(True)
    figure.legend(
        *axis.get_legend_handles_labels(), loc="upper center",
        bbox_to_anchor=(0.56, 0.88), ncol=3, fontsize=10.5,
        frameon=False, handlelength=2.4, columnspacing=2.1,
    )
    figure.text(
        0.56, 0.95,
        rf"SI1000 $p=0.3\%$  ·  $4d$ rounds  ·  {shot_label(rows[0]['shots'])} shots per distance",
        ha="center", va="top", fontsize=14,
    )
    figure.text(
        0.56, 0.035, "L2: plain MWPM  ·  6 patches  ·  2 ideal yokes",
        ha="center", va="bottom", fontsize=10.5, color="#4b5563",
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(out / f"{RELATIVE_FIGURE_NAME}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=[7, 9, 11, 13])
    parser.add_argument("--shots", type=int, default=1_000_000)
    parser.add_argument("--layout", choices=("combined", "separate"), default="combined",
                        help="One two-panel figure or two independent slide figures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive")
    distances = sorted(set(args.distances))
    rows, sources = read_results(args.result.resolve(), distances, expected_shots=args.shots)
    comparisons = compare_rates(rows)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.layout == "separate":
        draw_slide_figure(rows, out)
        draw_relative_figure(rows, comparisons, out)
        figure_names = [ABSOLUTE_FIGURE_NAME, RELATIVE_FIGURE_NAME]
    else:
        draw_figure(rows, comparisons, out)
        figure_names = [FIGURE_NAME]
    write_csv(out / "plotted_values.csv", rows)
    write_csv(out / "pairwise_comparisons.csv", comparisons)

    # Save the small shared reader beside the recipe so the retained version
    # can also be run directly, without depending on the experiments directory.
    recipe = out / "plot_recipe.py"
    if Path(__file__).resolve() != recipe:
        shutil.copy2(__file__, recipe)
    shared_reader = Path(__file__).with_name("plot_hierarchical_ler.py")
    if shared_reader.resolve() != out / shared_reader.name:
        shutil.copy2(shared_reader, out / shared_reader.name)
    for source in sources:
        snapshot = out / "source_results" / f"d{source['distance']}"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("results.json", "manifest.json", "mwpm_results.json"):
            path = Path(source["directory"]) / name
            if path.exists():
                shutil.copy2(path, snapshot / name)

    caption = (
        f"Completed {args.shots:,}-shot hierarchical decoder results at d="
        + ", ".join(map(str, distances))
        + ". SI1000 p=0.3%, 4d rounds, six patches, and two ideal yokes. "
        "All configurations use plain MWPM at L2. The absolute plot shows effective "
        "LER per round per physical surface-code patch. The relative plot shows "
        "100*(LER_method/LER_baseline-1), "
        "where the baseline is correlated MWPM with complementary gaps. Ratios "
        "are formed after applying the repository's per-patch-round normalization. "
        "No confidence intervals, fitted curves, or extrapolations are shown; "
        "lines connect measured points. The MPP comparison preserves the L1 "
        "hard decoder; the UF comparison changes both L1 and its confidence score.\n"
    )
    (out / "caption.txt").write_text(caption)
    provenance = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, *sys.argv]),
        distances=distances, shots_per_distance=args.shots, sources=sources,
        normalization=dict(
            function="sinter.shot_error_rate_to_piece_error_rate",
            pieces="6 * (4 * distance)", values=8,
            interpretation="Effective repository normalization of block LER",
        ),
        relative_penalty="100 * (normalized_ler_method / normalized_ler_baseline - 1)",
        confidence_intervals_shown=False, extrapolation=False,
        layout=args.layout, figure_names=figure_names,
        figure_inches=[9.6, 5.4] if args.layout == "separate" else [12.8, 7.2],
        raster_dpi=300,
        versions=dict(sinter=sinter.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
        recipe_sha256=sha256(recipe), shared_reader_sha256=sha256(out / shared_reader.name),
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    validation = dict(
        input_artifact_hashes_verified=True,
        complete_paired_row_ids_verified=True,
        block_failure_counts_recounted_from_predictions=True,
        unique_paired_block_shots_checked=len(distances) * args.shots,
        configurations=3, plotted_absolute_points=len(rows),
        outputs_sha256={name: sha256(out / name) for name in (
            [f"{stem}.{extension}" for stem in figure_names for extension in ("png", "pdf", "svg")]
            + ["plotted_values.csv", "pairwise_comparisons.csv"]
        )},
    )
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    lines = [
        "# Hierarchical decoder accuracy and relative penalties", "",
    ]
    for name in figure_names:
        lines += [
            f"![{name.replace('_', ' ')}]({name}.png)", "",
            f"[PNG]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg)", "",
        ]
    lines += [caption.strip(), "",
        "## Relative LER increases", "",
        "| Distance | MPP vs complementary gap | UF vs complementary gap | UF vs MPP |",
        "|---:|---:|---:|---:|",
    ]
    for distance in distances:
        values = [row["ler_increase_percent"] for row in comparisons if row["distance"] == distance]
        lines.append(f"| {distance} | " + " | ".join(f"+{value:.2f}%" for value in values) + " |")
    lines += [
        "", "All percentages use normalized per-round, per-patch LER. The absolute differences are included in [pairwise_comparisons.csv](pairwise_comparisons.csv).", "",
        "## Normalization and provenance", "",
        "Normalization follows `sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=6*4*d, values=8)`. This is an effective normalization of block failures, not a direct observation of each patch in each round. It approaches `block_ler/(6*4*d)` at small block failure rates.", "",
        "[Plotted values](plotted_values.csv), [source and command provenance](provenance.json), [validation](validation.json), and the [reproduction recipe](plot_recipe.py) are retained together. The recipe's shared reader is included. Input hashes and paired-shot coverage were checked, and all failure counts were recounted from saved predictions. No decoder was rerun.", "",
        "From `/data2/s2chitni/projects/dante`, source `env/workspace.sh` and activate `repos/yoked-surface-codes/.venv`, then run:", "",
        "```bash",
        "python experiments/plot_hierarchical_accuracy_separation.py \\",
        f"    --result {args.result} \\",
        "    --distances " + " ".join(map(str, distances)) + f" --shots {args.shots} \\",
        f"    --layout {args.layout} \\",
        '    --out "$DANTE_SCRATCH/runs/hierarchical-accuracy-separation"',
        "```", "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    print(f"Saved verified comparison to {out}")
    for row in comparisons:
        print(f"d={row['distance']}: {row['numerator']} / {row['denominator']}: +{row['ler_increase_percent']:.4f}%")


if __name__ == "__main__":
    main()
