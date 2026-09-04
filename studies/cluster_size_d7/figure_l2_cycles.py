"""The colleague's plot with a cycle count on the x-axis: coarse Micro Blossom cycles per shot, mean and maximum in two panels."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
import sinter

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/figures/fig_l2_cycles_vs_ler.png")
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json")); cy = json.load(open(cell / "microblossom_cycles.json"))
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
BASE = conv(fr["global_failures"] / N)
F = {f"{r['family']}:{r['value']}": r for r in fr["rows"]}; C = {f"{r['family']}:{r['value']}": r for r in cy["rows"]}
def series(fam):
    keys = sorted([k for k in C if k.startswith(fam + ":")], key=lambda k: F[k]["coverage_pct"])
    return keys, np.array([conv(F[k]["failures"] / N) for k in keys])
S = {fam: series(fam) for fam in ("size", "margin")}
STYLE = dict(size=dict(color="#2a78d6", marker="o", label="size cap, interior clusters"), margin=dict(color="#eb6834", marker="v", label="margin threshold, interior clusters"))
orig = C["none:None"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, stat, title in zip(axes, ("mean", "max"), ("mean cycles per shot", "maximum cycles per shot")):
    allx = [orig[f"cyc_shot_{stat}"]]
    for fam in ("size", "margin"):
        keys, y = S[fam]; x = np.array([C[k][f"cyc_shot_{stat}"] for k in keys]); allx += x.tolist()
        ax.plot(x, y, color=STYLE[fam]["color"], marker=STYLE[fam]["marker"], ms=5.5, lw=1.3, label=STYLE[fam]["label"])
    ax.axhline(BASE, color="k", lw=1); ax.plot([orig[f"cyc_shot_{stat}"]], [BASE], "ko", ms=6, label="original syndrome")
    ax.plot([min(C[k][f"cyc_shot_{stat}"] for fam in S for k in S[fam][0])], [BASE], marker="*", color="#16a34a", ms=15, ls="none", label="goal: MWPM accuracy at the best x")
    lo, hi = np.floor(min(allx) * 0.9 / 500) * 500, np.ceil(max(allx) * 1.04 / 500) * 500
    ax.set_xlim(lo, hi); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))
    ax.set_title(title, fontsize=11); ax.grid(True, which="major", alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
ax = axes[0]
ax.set_yscale("log"); ax.set_ylim(2.42e-3, 3.0e-3)
ticks = np.arange(2.5e-3, 3.01e-3, 1e-4); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(ticks); ax.set_yticklabels([f"{t*1e3:.1f}×10$^{{-3}}$" for t in ticks]); ax.set_ylabel("Logical Error Rate (per patch round)")
ax.text(0.01, BASE * 0.998, " Global MWPM alone", transform=ax.get_yaxis_transform(), fontsize=9, va="top")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=9, frameon=False)
sec = axes[1].secondary_yaxis("right", functions=(lambda v: v / BASE, lambda r: r * BASE))
rt = np.array([1.0, 1.02, 1.05, 1.1, 1.15, 1.2]); sec.set_yticks(rt); sec.set_yticklabels([f"×{r:.2f}" for r in rt]); sec.yaxis.set_minor_formatter(NullFormatter()); sec.set_ylabel("multiple of Global MWPM alone")
axes[1].spines["right"].set_visible(True)
m = cy["model"]
fig.supxlabel(f"coarse Micro Blossom accelerator cycles per shot  ({m['accelerator_mhz']:.0f} MHz; {m['cycles_per_cpu_defect']:.1f} cycles per CPU-handled defect, isolated pairs free)", fontsize=11, y=0.02)
fig.suptitle(f"Logical Error Rate vs L2 Cycle Count   ·   d={c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, SI1000 p={c['p']}, {N:,} paired shots", fontsize=11.5, y=0.985)
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig(out, dpi=180); fig.savefig(out.with_suffix(".pdf")); print("wrote", out)
