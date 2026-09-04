"""Presentation set: one visual family for the coverage ladder, the three L2 cost views, and a cross-model summary.

Conventions shared by every figure: paper unit on a log y-axis with explicit sub-decade ticks, multiplier axis on the
right, size ladder blue circles, margin ladder orange triangles, Global MWPM alone as a black line with a black point
for the untouched syndrome, the goal as a green star, the free region (within 1% of MWPM alone) as a faint band, two
direct labels at most per panel, cell parameters in a footer rather than the title.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
import sinter

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/figures/presentation")
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.titleweight": "medium"})
BLUE, ORANGE, GREEN, INK = "#2a78d6", "#eb6834", "#16a34a", "#111111"
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json"))
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
BASE = conv(fr["global_failures"] / N)
F = {f"{r['family']}:{r['value']}": r for r in fr["rows"]}
FOOT = f"d = {c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, SI1000 p = {c['p']}, {N:,} paired shots · vertical axis: logical error per patch round (paper unit), log scale · star: MWPM's accuracy at the best x"
Y_LO, Y_HI = 2.42e-3, 3.0e-3
def ler(k): return conv(F[k]["failures"] / N)
def keys(fam, source): return sorted([k for k in source if k.startswith(fam + ":") and k in F], key=lambda k: F[k]["coverage_pct"])
def paper_y(ax, label=True):
    ax.set_yscale("log"); ax.set_ylim(Y_LO, Y_HI)
    t = np.arange(2.5e-3, 3.01e-3, 1e-4); ax.yaxis.set_major_locator(FixedLocator(t)); ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks(t)
    ax.set_yticklabels([f"{v*1e3:.1f}×10$^{{-3}}$" for v in t])
    if not label: ax.tick_params(axis="y", labelleft=False)
    ax.axhspan(BASE, BASE * 1.01, color=GREEN, alpha=0.10, lw=0); ax.axhline(BASE, color=INK, lw=1.1)
    ax.grid(True, which="major", alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def mult_axis(ax):
    sec = ax.secondary_yaxis("right", functions=(lambda v: v / BASE, lambda r: r * BASE)); rt = np.array([1.0, 1.02, 1.05, 1.1, 1.15, 1.2])
    sec.set_yticks(rt); sec.set_yticklabels([f"×{r:.2f}" for r in rt]); sec.yaxis.set_minor_formatter(NullFormatter()); sec.set_ylabel("multiple of Global MWPM alone"); ax.spines["right"].set_visible(True)
def ladder(ax, xs, ys, fam):
    st = dict(size=(BLUE, "o", "size cap"), margin=(ORANGE, "o", "margin threshold"))[fam]
    ax.plot(xs, ys, color=st[0], marker=st[1], ms=6.5, lw=1.5, label=st[2] + ", interior clusters")
def label(ax, x, y, text, dx, dy, ha="left", color=INK):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=11.5, ha=ha, va="center", color=color)
def finish(fig, name, title, handles=None, labels=None, xlabel=None, left=0.105, foot=None):
    fig.subplots_adjust(left=left, right=0.915, top=0.86, bottom=0.27, wspace=0.07)
    fig.suptitle(title, fontsize=15, y=0.975, x=0.02, ha="left", fontweight="medium")
    if xlabel: fig.supxlabel(xlabel, fontsize=13, y=0.155)
    if handles: fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035), columnspacing=2.0, handletextpad=0.6, fontsize=12)
    fig.text(0.02, 0.008, foot or FOOT, fontsize=9.5, color="0.4", ha="left")
    for ext in ("png", "pdf", "svg"): fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig); print("wrote", name)
def two_panel(name, title, source, xkey, xlabel, panel_titles, xscale="linear", xlim=None, xticks=None, xfmt=None, goal_from_data=True):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, gridspec_kw=dict(wspace=0.07))
    for ax, stat, ptitle in zip(axes, ("mean", "p99"), panel_titles):
        best = np.inf
        for fam in ("size", "margin"):
            ks = keys(fam, source); xs = np.array([xkey(source[k], stat) for k in ks]); ys = np.array([ler(k) for k in ks]); ladder(ax, xs, ys, fam); best = min(best, xs.min())
        ax.plot([xkey(source["none:None"], stat)], [BASE], "o", color=INK, ms=7, label="untouched syndrome")
        ax.plot([best], [BASE], marker="*", color=GREEN, ms=17, ls="none", label="goal", zorder=5)
        paper_y(ax, label=ax is axes[0]); ax.set_title(ptitle); ax.set_xscale(xscale)
        if xlim: ax.set_xlim(*xlim)
        if xticks: ax.set_xticks(xticks)
        if xfmt: ax.xaxis.set_major_formatter(FuncFormatter(xfmt)); ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("logical error rate per patch round"); mult_axis(axes[1])
    h, l = axes[0].get_legend_handles_labels(); h.append(plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.10)); l.append("within 1% of MWPM alone (free region)")
    finish(fig, name, title, h, l, xlabel)

# 1. coverage ladder -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 5.6))
for fam in ("size", "margin"):
    ks = [k for k in keys(fam, F) if F[k]["coverage_pct"] > 0 and (fam != "size" or F[k]["value"] is None or F[k]["value"] % 2 == 0)]
    ladder(ax, [F[k]["coverage_pct"] for k in ks], [ler(k) for k in ks], fam)
label(ax, F["size:2"]["coverage_pct"], ler("size:2"), "size ≤ 2", -10, 8, ha="right", color=BLUE)
z = np.load(cell / "cell.npz"); ceiling = 100 * float(z["component_size"][z["component_committed"] & ~z["component_boundary"]].sum()) / float(z["component_size"].sum())
ax.plot([0], [BASE], "o", color=INK, ms=7, label="untouched syndrome"); ax.plot([ceiling], [BASE], marker="*", color=GREEN, ms=17, ls="none", label="goal", zorder=5)
ax.axvline(ceiling, color="0.6", lw=1, ls=(0, (4, 4))); label(ax, ceiling, 2.98e-3, f"interior-only ceiling {ceiling:.0f}%", 6, 0, color="0.35")
paper_y(ax); mult_axis(ax); ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 10)); ax.set_xticklabels([f"{t}%" for t in range(0, 101, 10)])
ax.set_ylabel("logical error rate per patch round"); ax.set_xlabel("coverage: share of detector events removed by the L1 frontend before Global MWPM", labelpad=8)
h, l = ax.get_legend_handles_labels(); h.append(plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.10)); l.append("within 1% of MWPM alone (free region)")
finish(fig, "1_coverage_vs_ler", "Accuracy cost of the L1 frontend against coverage", h, l, left=0.105)

# 2. software time ---------------------------------------------------------------------------------
T = {f"{r['family']}:{r['value']}": r for r in json.load(open(cell / "l2_timing.json"))["rows"]}
two_panel("2_software_time", "Software sparse blossom (PyMatching): decode time of the residual", T, lambda r, st: 100 * r[f"rel_{st}"],
          "Global MWPM decode time on the residual, % of the untouched syndrome", ("mean over 100,000 shots", "99th-percentile shot"),
          xlim=(35, 105), xticks=range(40, 101, 10), xfmt=lambda v, _: f"{v:g}%")
# 3. Micro Blossom ---------------------------------------------------------------------------------
MB = {f"{r['family']}:{r['value']}": r for r in json.load(open(cell / "microblossom_cycles.json"))["rows"]}
two_panel("3_microblossom_cycles", "Micro Blossom-style accelerator: coarse cycles per shot (isolated pairs resolved in hardware)", MB, lambda r, st: r[f"cyc_shot_{st}"],
          "estimated accelerator cycles per 28-round shot, 62 MHz", ("mean over 100,000 shots", "99th-percentile shot"),
          xlim=(1500, 8000), xticks=[2000, 3000, 4000, 5000, 6000, 7000, 8000], xfmt=lambda v, _: f"{v/1000:g}k")
# 4. Zero-G ----------------------------------------------------------------------------------------
ZG = json.load(open(cell / "zerog_cycles.json")); Z = {f"{r['family']}:{r['value']}": r for r in ZG["rows"]}
two_panel("4_zerog_cycles", "Zero-G-style decoder: coarse cycles per decoding window (cost counts active detectors)", Z, lambda r, st: r[f"cyc_{st}"],
          f"estimated cycles per {ZG['window_rounds']}-round window of the whole block, 250 MHz, log scale", ("mean over 400,000 windows", "99th-percentile window"),
          xscale="log", xlim=(1e3, 1.2e5), xticks=[1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5], xfmt=lambda v, _: f"{v/1000:g}k")

# 5. summary: what the free region buys, by L2 model ----------------------------------------------
models = [("software\nsparse blossom\n(measured time)", T, lambda r, st: r[f"rel_{st}"]), ("Micro Blossom\nstyle\n(coarse cycles)", MB, lambda r, st: r[f"rel_{st}"]), ("Zero-G\nstyle\n(coarse cycles)", Z, lambda r, st: r[f"rel_{st}"])]
rules = [("margin > 1.5  (×1.001)", "margin:1.5"), ("margin > 1.0  (×1.009)", "margin:1.0")]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (rtitle, rk) in zip(axes, rules):
    xs = np.arange(len(models)); w = 0.36
    mean = [100 * f(src[rk], "mean") for _, src, f in models]; mx = [100 * f(src[rk], "p99") for _, src, f in models]
    b1 = ax.bar(xs - w / 2, mean, w, color="#9ed9cf", label="mean"); b2 = ax.bar(xs + w / 2, mx, w, color="#0f766e", label="99th percentile")
    for bars in (b1, b2):
        for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%", ha="center", fontsize=11)
    ax.axhline(100, color=INK, lw=1.1); ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in models], fontsize=11.5); ax.set_title(rtitle); ax.set_ylim(0, 128)
    ax.grid(True, axis="y", alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("L2 cost with frontend, % of untouched"); axes[0].legend(loc="upper left", frameon=False, ncol=2)
axes[1].text(2.62, 101.5, "100% = no saving", fontsize=10.5, color="0.35", ha="right", va="bottom")
fig.suptitle("L2 work reduction within an exploratory 1% accuracy budget", fontsize=15, y=0.975, x=0.02, ha="left", fontweight="medium")
fig.text(0.02, 0.008, f"d = {c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, SI1000 p = {c['p']}, {N:,} paired shots · lower is better · Micro Blossom and Zero-G columns are coarse models with borrowed constants", fontsize=9.5, color="0.4")
fig.subplots_adjust(left=0.085, right=0.97, top=0.86, bottom=0.2, wspace=0.08)
for ext in ("png", "pdf", "svg"): fig.savefig(out / f"5_free_region_by_l2.{ext}", dpi=200 if ext == "png" else None)
plt.close(fig); print("wrote 5_free_region_by_l2")

# 6. Helios growth budget: x = truncated Helios proxy cycles at cap K, mean and slowest shot ---------------
if (cell / "cap_ladder.json").exists() and (cell / "helios_budget_cycles.json").exists():
    CL = json.load(open(cell / "cap_ladder.json")); HB = json.load(open(cell / "helios_budget_cycles.json"))
    cyc = {r["cap"]: r for r in HB["rows"] if r["hop"] == 1}
    cols = dict(cap=("#7c3aed", "cap alone"), cap_m05=("#f59e0b", "cap with margin > 0.5"), cap_m10=(ORANGE, "cap with margin > 1.0"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, gridspec_kw=dict(wspace=0.07))
    for ax, stat, ptitle in zip(axes, ("mean", "p99"), ("mean over 100,000 shots", "99th-percentile shot")):
        for sid in ("cap", "cap_m05", "cap_m10"):
            rr = sorted([r for r in CL["rows"] if r["series"] == sid], key=lambda r: 1e9 if r["cap"] is None else r["cap"])
            ax.plot([cyc[r["cap"]][stat] for r in rr], [conv(r["failures"] / N) for r in rr], color=cols[sid][0], marker="o", ms=6.5, lw=1.5, label=cols[sid][1])
        paper_y(ax, label=ax is axes[0]); ax.set_title(ptitle); ax.set_xscale("log", base=2); ax.set_xlim(3, 1000)
        for k, lab, yy in ((5, "K = 5", 0.995), (6, "K = 6", 0.975), (None, "no cap", 0.995)):
            gx = cyc[k][stat]; ax.axvline(gx, color="0.6", lw=1, ls=(0, (3, 4))); ax.text(gx, Y_HI * yy, lab, ha="center", va="top", fontsize=11, color="0.3", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ticks = [4, 8, 16, 32, 64, 128, 256, 512]; ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("logical error rate per patch round"); mult_axis(axes[1])
    h, l = axes[0].get_legend_handles_labels(); h.append(plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.10)); l.append("within 1% of MWPM alone (free region)")
    finish(fig, "6_helios_budget", "Accuracy versus capped Helios-style UF growth-depth proxy", h, l,
           "growth-depth proxy cycles at budget K, per 28-round shot, slowest lane (4 fixed cycles per iteration + merge flooding)",
           foot=FOOT.replace(" · star: MWPM's accuracy at the best x", " · dashed lines: budgets of 5 and 6 iterations, and no cap"))
