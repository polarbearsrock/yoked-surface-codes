#!/usr/bin/env python3
"""Plot completed hierarchical-decoder runs using the repository's LER units.

Only saved results are read. This script does not run a decoder or modify the
ongoing sweep. PNG, PDF, SVG, a numeric table, and provenance are saved together.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
import numpy as np
import sinter


# Distinct shapes and line styles keep the two close MWPM curves identifiable.
SERIES = (
    ("correlated_mwpm_complementary_gap", "gap_prediction",
     "Corr. MWPM + complementary gap", "#17365d", "o", "-", 5.2),
    ("correlated_mwpm_mpp", "mpp_prediction",
     "Corr. MWPM + MPP", "#278f83", "D", "--", 3.7),
    ("correlated_uf_cluster_gap", "prediction",
     "Corr. UF + cluster gap", "#d55e00", "^", "-", 4.8),
)
FIGURE_NAME = "hierarchical_ler_per_patch_round"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patch_round_rate(block_rate, *, patches, rounds, yokes):
    """Use the convention in yoked-surface-codes/tools/plot_extrapolations.

    `pieces` is the number of physical patch-rounds. `values` is twice the
    encoded logical-qubit count, following the repository's plotting recipe.
    The returned rate already includes these values; do not divide by them.

    This is an effective normalization of the final block failure probability,
    based on Sinter's independent, equal-rate XOR/OR model. It is not a direct
    measurement of an individual patch's failure probability each round.
    For small block rates it approaches block_rate / (patches * rounds).
    """
    if yokes != 2:
        raise ValueError("This figure is for the experiment with two ideal yokes")
    return sinter.shot_error_rate_to_piece_error_rate(
        block_rate, pieces=patches * rounds, values=2 * (patches - yokes),
    )


def read_results(directory, distances, *, expected_shots=100_000):
    """Check each completed artifact and recover the plotted numeric values."""
    rows, sources = [], []
    for distance in distances:
        folder = directory / "distances" / f"d{distance}"
        manifest = json.loads((folder / "manifest.json").read_text())
        for name in ("results.json", "predictions.npz"):
            if sha256(folder / name) != manifest["artifacts"][name]:
                raise ValueError(f"Artifact hash mismatch: {folder / name}")
        report = json.loads((folder / "results.json").read_text())
        parameters = report["parameters"]
        expected = dict(distance=distance, rounds=4 * distance, patches=6,
                        yokes=2, noise="si1000", p=0.003, style="cz")
        if parameters != expected or report["shots"] != expected_shots:
            raise ValueError(f"Unexpected experiment configuration at d={distance}")

        # Recount final failures from the saved paired predictions. This checks
        # the figure's numerator without repeating any decoding work.
        with np.load(folder / "predictions.npz") as arrays:
            actual = arrays["actual"]
            if actual.shape != (report["shots"], 2 * parameters["patches"]):
                raise ValueError(f"Unexpected truth-array shape at d={distance}")
            np.testing.assert_array_equal(arrays["row_ids"], np.arange(report["shots"]))
            for name, key, *_ in SERIES:
                prediction = arrays[key]
                if prediction.shape != actual.shape:
                    raise ValueError(f"Unexpected prediction shape: {name}, d={distance}")
                failures = int(np.any(prediction != actual, axis=1).sum())
                saved = report["configurations"][name]
                if failures != saved["block_failures"]:
                    raise ValueError(f"Failure count mismatch: {name}, d={distance}")
                block_rate = failures / report["shots"]
                if not np.isclose(block_rate, saved["block_failure_rate"], atol=1e-15, rtol=0):
                    raise ValueError(f"Failure rate mismatch: {name}, d={distance}")
                normalized = patch_round_rate(
                    block_rate, patches=parameters["patches"],
                    rounds=parameters["rounds"], yokes=parameters["yokes"],
                )
                rows.append(dict(
                    distance=distance, rounds=parameters["rounds"],
                    patches=parameters["patches"], shots=report["shots"],
                    configuration=name, block_failures=failures,
                    block_ler=block_rate, ler_per_patch_round=normalized,
                    linear_block_ler_per_patch_round=block_rate / (
                        parameters["patches"] * parameters["rounds"]),
                ))
        sources.append(dict(
            distance=distance, directory=str(folder),
            manifest_sha256=sha256(folder / "manifest.json"),
            artifacts=manifest["artifacts"],
            decoder_provenance=manifest["provenance"],
            baseline_provenance=manifest["baseline"]["provenance"],
        ))
    return rows, sources


def shot_label(shots):
    """Keep figure annotations short without rounding the actual shot count."""
    if shots % 1_000_000 == 0:
        return f"{shots // 1_000_000}M"
    if shots % 1_000 == 0:
        return f"{shots // 1_000}k"
    return f"{shots:,}"


def draw_slide_figure(rows, out):
    """Use a widescreen canvas, larger type, and a legend outside the data."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 15, "axes.linewidth": 0.8,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.18, top=0.75)
    labels = (
        "Correlated MWPM\nComplementary gap",
        "Correlated MWPM\nMPP cluster score",
        "Correlated UF\nCluster gap",
    )
    for series, label in zip(SERIES, labels):
        name, _, _, color, marker, line, size = series
        points = [row for row in rows if row["configuration"] == name]
        axis.plot(
            [row["distance"] for row in points],
            [row["ler_per_patch_round"] for row in points],
            label=label, color=color, marker=marker, linestyle=line,
            markersize=size * 1.7, markerfacecolor="white",
            markeredgewidth=1.6, linewidth=2, zorder=3,
        )
    distances = sorted({row["distance"] for row in rows})
    rates = [row["ler_per_patch_round"] for row in rows]
    axis.set(
        xlabel="Code distance, $d$", ylabel="LER per round per patch",
        yscale="log", xlim=(distances[0] - 0.28, distances[-1] + 0.28),
        ylim=(min(rates) / 1.5, max(rates) * 1.18),
    )
    axis.set_xticks(distances)
    # Two distances span less than a decade. Intermediate labeled ticks make
    # that small log range readable without leaving just a single y-axis label.
    narrow_range = max(rates) / min(rates) < 10
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 3, 5, 7) if narrow_range else (1,)))
    axis.yaxis.set_major_formatter(LogFormatterSciNotation(minor_thresholds=(float("inf"), float("inf"))))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="y", which="minor", length=0)
    axis.tick_params(axis="both", which="major", pad=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", which="major", color="#dce1e7", linewidth=0.65)
    axis.set_axisbelow(True)
    figure.legend(
        *axis.get_legend_handles_labels(), loc="upper center",
        bbox_to_anchor=(0.56, 0.88), ncol=3, fontsize=11.5,
        frameon=False, handlelength=2.5, columnspacing=2.3,
    )
    shots = shot_label(rows[0]["shots"])
    figure.text(
        0.56, 0.95,
        rf"SI1000 $p=0.3\%$  ·  $4d$ rounds  ·  {shots} shots per distance",
        ha="center", va="top", fontsize=14,
    )
    figure.text(
        0.56, 0.035, "L2: plain MWPM  ·  6 patches  ·  2 ideal yokes",
        ha="center", va="bottom", fontsize=10.5, color="#4b5563",
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(out / f"{FIGURE_NAME}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def draw_figure(rows, out, *, style="paper"):
    """Draw a compact log-scale figure at a journal-column width."""
    if style == "slides":
        draw_slide_figure(rows, out)
        return
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.linewidth": 0.65,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "ytick.minor.width": 0.45, "ytick.minor.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, axis = plt.subplots(figsize=(3.55, 2.65))
    figure.subplots_adjust(left=0.185, right=0.975, bottom=0.17, top=0.90)
    for name, _, label, color, marker, line, size in SERIES:
        points = [row for row in rows if row["configuration"] == name]
        axis.plot(
            [row["distance"] for row in points],
            [row["ler_per_patch_round"] for row in points],
            label=label, color=color, marker=marker, linestyle=line,
            markersize=size, markerfacecolor="white", markeredgewidth=0.9,
            linewidth=1.15, zorder=3,
        )
    distances = sorted({row["distance"] for row in rows})
    rates = [row["ler_per_patch_round"] for row in rows]
    axis.set(
        xlabel="Code distance, $d$", ylabel="LER per round per patch",
        yscale="log", xlim=(distances[0] - 0.45, distances[-1] + 0.45),
        ylim=(min(rates) / 1.4, max(rates) * 1.4),
    )
    axis.set_xticks(distances)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis="y", which="major", color="#cccccc", linewidth=0.45)
    axis.set_axisbelow(True)
    shots = shot_label(rows[0]["shots"])
    axis.set_title(rf"SI1000 $p=0.3\%$  ·  $4d$ rounds  ·  {shots} shots", fontsize=8, pad=7)
    legend = axis.legend(
        loc="lower left", fontsize=6.5, frameon=True, fancybox=False,
        facecolor="white", framealpha=1, edgecolor="#cccccc",
        borderpad=0.5, labelspacing=0.45, handlelength=2.1,
    )
    legend.get_frame().set_linewidth(0.5)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(out / f"{FIGURE_NAME}.{extension}", dpi=450)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Retained sweep directory containing distances/d*/results.json")
    parser.add_argument("--distances", type=int, nargs="+", default=[7, 9, 11, 13])
    parser.add_argument("--shots", type=int, default=100_000,
                        help="Expected paired block shots at every plotted distance")
    parser.add_argument("--style", choices=("paper", "slides"), default="paper")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive")
    distances = sorted(set(args.distances))
    rows, sources = read_results(args.result.resolve(), distances, expected_shots=args.shots)
    args.out.mkdir(parents=True, exist_ok=True)
    draw_figure(rows, args.out, style=args.style)
    with (args.out / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    caption = (
        "Final hierarchical logical error rate per round per physical surface-code "
        "patch versus code distance. SI1000 p=0.3%, six patches, two ideal yokes, "
        f"4d stabilizer rounds, and {args.shots:,} paired block shots at each distance. "
        "The three configurations use correlated MWPM with complementary gaps, "
        "correlated MWPM with MPP scores, or correlated UF with cluster gaps at L1; "
        "all use plain MWPM at L2. Points are completed measurements; connecting "
        "lines guide the eye. Confidence intervals are omitted. The effective "
        "normalization follows the yoked-surface-codes plotting convention: "
        "sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=6*(4*d), "
        "values=2*(6-2)). It approaches block_ler/(6*4*d) at small block LER. "
        "It is not a measurement of the instantaneous failure probability of "
        "an individual patch, nor a rate per encoded logical qubit.\n"
    )
    (args.out / "caption.txt").write_text(caption)
    recipe = args.out / "plot_recipe.py"
    if Path(__file__).resolve() != recipe.resolve():
        shutil.copy2(__file__, recipe)
    for source in sources:
        snapshot = args.out / "source_results" / f"d{source['distance']}"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("results.json", "manifest.json"):
            shutil.copy2(Path(source["directory"]) / name, snapshot / name)
    provenance = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, *sys.argv]),
        distances=distances, sources=sources,
        normalization=dict(
            function="sinter.shot_error_rate_to_piece_error_rate",
            pieces="patches * rounds = 6 * 4 * d", values=8,
            reference="repos/yoked-surface-codes/tools/plot_extrapolations:113",
            interpretation="Effective repository normalization of block LER",
        ),
        confidence_intervals_shown=False, extrapolation=False,
        style=args.style, shots_per_distance=args.shots,
        figure_inches=[9.6, 5.4] if args.style == "slides" else [3.55, 2.65],
        raster_dpi=300 if args.style == "slides" else 450,
        versions=dict(sinter=sinter.__version__, numpy=np.__version__,
                      matplotlib=matplotlib.__version__),
        plot_recipe_sha256=sha256(recipe),
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified {len(distances)} completed distances; saved figure to {args.out.resolve()}")
    for row in rows:
        print(f"d={row['distance']:2d} {row['configuration']}: {row['ler_per_patch_round']:.8g}")


if __name__ == "__main__":
    main()
