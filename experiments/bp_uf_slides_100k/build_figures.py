"""Reproduce the 100k-shot BP figures from retained experiment summaries."""

from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/bp_uf_single_patch_d7_p003_20260929T054325Z"
SWEEP = ROOT / "results/bp_uf_single_patch_d9_d11_d13_p003_20260929T062831Z"
DISTANCES = [7, 9, 11, 13]
VARIANTS = ["weighted_uf", "correlated_uf", "correlated_mwpm", "bp1_uf", "bp2_uf", "bp5_uf", "bp10_uf"]
LABELS = ["Weighted UF", "Correlated UF", "Corr. MWPM\n(baseline)", "BP1 + UF", "BP2 + UF", "BP5 + UF", "BP10 + UF"]
COLORS = ["#737373", "#b17b14", "#202020", "#63a5d5", "#2176ac", "#a17fc5", "#4b2e83"]
MARKERS = ["s", "D", "x", "v", "^", "o", "o"]
STYLES = ["--", "--", ":", "-", "-", "-", "-"]


def main(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    stats = {7: read(OLD / "bp10_uf/confirmation_summary.json")}
    timing = {7: read(OLD / "serial_timing.json")["metrics"]}
    timing[7].update(read(OLD / "bp10_uf/serial_timing.json")["metrics"])
    timing[7]["correlated_mwpm"] = read(OLD / "correlated_mwpm/serial_timing.json")["metrics"]["bound_native_decode"]
    for d in DISTANCES[1:]:
        stats[d] = read(SWEEP / f"d{d}/confirmation/summary.json")
        timing[d] = read(SWEEP / f"d{d}/serial_timing.json")["metrics"]
    config = read(SWEEP / "configuration.json")
    for d in DISTANCES:
        for key in VARIANTS:
            m = stats[d]["metrics"][key]
            assert m["shots"] == 100000
            assert abs(m["failure_rate"] - m["errors"] / m["shots"]) < 1e-12
        for key in ["bp1_uf", "bp2_uf", "bp5_uf", "bp10_uf"]:
            m = timing[d][key]
            if "mean_bp_rounds_ms" in m:
                components = sum(m[k] for k in ["mean_bp_initialization_ms", "mean_bp_rounds_ms", "mean_projection_ms", "mean_uf_ms"])
                assert abs(components - m["mean_total_ms"]) < 1e-9

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 18.5,
        "axes.labelsize": 19.5, "xtick.labelsize": 18.5, "ytick.labelsize": 18.5,
        "legend.fontsize": 17.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "savefig.facecolor": "white",
    })

    def canvas(footer):
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = fig.add_axes([0.155, 0.225, 0.805, 0.56])
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.set_xticks(DISTANCES)
        ax.set_xlim(6.6, 13.4)
        ax.set_xlabel("Code distance, d", labelpad=7)
        fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=15, color="#555555")
        return fig, ax

    def finish(fig, ax, stem, columns=4):
        handles, labels = ax.get_legend_handles_labels()
        legend_top = 1.015 if stem == "bp_uf_breakdown" else 0.995
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, legend_top),
                   ncol=columns, frameon=False, columnspacing=0.95, handlelength=1.35,
                   handletextpad=0.45, labelspacing=0.4)
        fig.savefig(output / f"{stem}.png", dpi=220)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)

    fig, ax = canvas("100k paired shots / d  ·  95% Wilson intervals")
    for key, label, color, marker, style in zip(VARIANTS, LABELS, COLORS, MARKERS, STYLES):
        y = np.array([100 * stats[d]["metrics"][key]["failure_rate"] for d in DISTANCES])
        ci = np.array([stats[d]["metrics"][key]["wilson_95"] for d in DISTANCES]).T * 100
        ax.errorbar(DISTANCES, y, yerr=[y-ci[0], ci[1]-y], label=label,
                    color=color, marker=marker, linestyle=style, linewidth=2.0,
                    markersize=6.5, capsize=3, elinewidth=1.25)
    ax.set_ylabel("Logical error rate, LER (%)", labelpad=8)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 5, 10, 15])
    finish(fig, ax, "accuracy")

    # Match slide 13: protocol header, named zero baseline, percentage ticks,
    # and annotated endpoint gaps. The absolute plot retains weighted UF.
    fig, ax = canvas("1 patch  ·  0 yokes  ·  LER per full memory shot")
    ax.set_position([0.155, 0.19, 0.805, 0.55])
    fig.text(0.54, 0.965, "SI1000 p = 0.3%  ·  4d rounds  ·  100k shots per distance",
             ha="center", va="top", fontsize=16)
    ax.axhline(0, color=COLORS[2], linestyle=":", linewidth=1.5,
               label="Corr. MWPM (baseline)")
    relative_rows = []
    for i in [1, 3, 4, 5, 6]:
        key = VARIANTS[i]
        y = [100 * (stats[d]["metrics"][key]["failure_rate"] / stats[d]["metrics"]["correlated_mwpm"]["failure_rate"] - 1) for d in DISTANCES]
        relative_rows.extend({"distance": d, "decoder": key, "baseline": "correlated_mwpm", "relative_ler_percent": value} for d, value in zip(DISTANCES, y))
        ax.plot(DISTANCES, y, label=LABELS[i], color=COLORS[i], marker=MARKERS[i],
                linestyle=STYLES[i], linewidth=2.2, markersize=7)
        for j in ([0, 3] if key == "bp10_uf" else [3]):
            offset = -7 if key == "bp10_uf" else 7
            ax.annotate(f"{y[j]:+.1f}%", (DISTANCES[j], y[j]),
                        xytext=(0, offset), textcoords="offset points",
                        ha="left" if j == 0 else "right",
                        va="top" if offset < 0 else "bottom",
                        fontsize=15, color=COLORS[i])
    ax.set_ylabel("LER increase over baseline (%)", labelpad=8)
    ax.set_ylim(-45, 210)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.89),
               ncol=3, frameon=False, fontsize=15, columnspacing=1.2,
               handlelength=1.5, handletextpad=0.45, labelspacing=0.4)
    fig.savefig(output / "relative_accuracy.png", dpi=220)
    fig.savefig(output / "relative_accuracy.pdf")
    plt.close(fig)
    with (output / "relative_ler_vs_correlated_mwpm.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=relative_rows[0])
        writer.writeheader()
        writer.writerows(relative_rows)

    fig, ax = canvas("Serial mean over 1,024 pilot shots  ·  log scale")
    for key, label, color, marker, style in zip(VARIANTS, LABELS, COLORS, MARKERS, STYLES):
        y = [timing[d][key]["mean_total_ms"] for d in DISTANCES]
        ax.plot(DISTANCES, y, label=label, color=color, marker=marker, linestyle=style,
                linewidth=2.0, markersize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(0.06, 330)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Mean serial decode time (ms)", labelpad=8)
    finish(fig, ax, "serial_latency")

    fig, ax = canvas("d = 13  ·  1,024 pilot shots  ·  serial kernel timing")
    ax.set_xticks(range(4), ["BP1", "BP2", "BP5", "BP10"])
    ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("Fixed BP budget + weighted UF", labelpad=7)
    ax.set_ylim(0, 270)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_ylabel("Mean serial decode time (ms)", labelpad=8)
    keys = ["bp1_uf", "bp2_uf", "bp5_uf", "bp10_uf"]
    bottom = np.zeros(4)
    components = [
        ("mean_bp_initialization_ms", "BP init", "#b6b6b6"),
        ("mean_bp_rounds_ms", "BP rounds", "#4b2e83"),
        ("mean_projection_ms", "Projection", "#c79936"),
        ("mean_uf_ms", "UF", "#398a8c"),
    ]
    for field, label, color in components:
        values = np.array([timing[13][key][field] for key in keys])
        ax.bar(range(4), values, width=0.62, bottom=bottom, color=color, label=label)
        bottom += values
    for i, total in enumerate(bottom):
        ax.text(i, total + 5, f"{total:.1f}", ha="center", va="bottom", fontsize=18.5)
    finish(fig, ax, "bp_uf_breakdown", columns=4)

    rows = []
    for d in DISTANCES:
        for key, label in zip(VARIANTS, LABELS):
            m = stats[d]["metrics"][key]
            rows.append({"distance": d, "decoder": key, "failures": m["errors"], "shots": m["shots"],
                         "failure_percent": 100*m["failure_rate"], "mean_serial_ms": timing[d][key]["mean_total_ms"],
                         "wilson_low_percent": 100*m["wilson_95"][0], "wilson_high_percent": 100*m["wilson_95"][1]})
    with (output / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (output / "plot_data.json").write_text(json.dumps({"statistics": stats, "timing": timing, "configuration": config}, indent=2) + "\n")
    (output / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps({"output": str(output), "figures": 4, "rows": len(rows),
                      "d13_bp10_bp_round_fraction": timing[13]["bp10_uf"]["mean_bp_rounds_ms"] / timing[13]["bp10_uf"]["mean_total_ms"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
