"""The colleague's plot, clean: LER per patch round (log, paper unit) against L2 decode time, mean and maximum in two panels."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
import sinter

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/figures/fig_l2_time_vs_ler.png")
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json")); tim = json.load(open(cell / "l2_timing.json"))
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
BASE = conv(fr["global_failures"] / N)
F = {f"{r['family']}:{r['value']}": r for r in fr["rows"]}; T = {f"{r['family']}:{r['value']}": r for r in tim["rows"]}
def series(fam):
    keys = [k for k in T if k.startswith(fam + ":")]
    keys.sort(key=lambda k: F[k]["coverage_pct"])
    return keys, np.array([conv(F[k]["failures"] / N) for k in keys])
S = {fam: series(fam) for fam in ("size", "margin")}
STYLE = dict(size=dict(color="#2a78d6", marker="o", label="size cap, interior clusters"), margin=dict(color="#eb6834", marker="v", label="margin threshold, interior clusters"))

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, stat, title in zip(axes, ("mean", "max"), ("mean L2 decode time", "maximum L2 decode time")):
    for fam in ("size", "margin"):
        keys, y = S[fam]; x = np.array([100 * T[k][f"rel_{stat}"] for k in keys])
        ax.plot(x, y, color=STYLE[fam]["color"], marker=STYLE[fam]["marker"], ms=5.5, lw=1.3, label=STYLE[fam]["label"])
    ax.axhline(BASE, color="k", lw=1); ax.plot([100], [BASE], "ko", ms=6, label="nothing committed")
    ax.plot([min(100 * T[k][f"rel_{stat}"] for fam in S for k in S[fam][0])], [BASE], marker="*", color="#16a34a", ms=15, ls="none", label="goal: MWPM accuracy at the best x")
    ax.set_title(title, fontsize=11); ax.set_xlim(35, 105); ax.set_xticks(range(40, 101, 10)); ax.set_xticklabels([f"{t}%" for t in range(40, 101, 10)])
    ax.grid(True, which="major", alpha=0.5); ax.grid(True, which="minor", alpha=0.2); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
ax = axes[0]
ax.set_yscale("log"); ax.set_ylim(2.42e-3, 3.0e-3)
ticks = np.arange(2.5e-3, 3.01e-3, 1e-4); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(ticks); ax.set_yticklabels([f"{t*1e3:.1f}×10$^{{-3}}$" for t in ticks])
ax.set_ylabel("Logical Error Rate (per patch round)")
ax.text(36, BASE * 0.998, "Global MWPM alone", fontsize=9, va="top")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=9, frameon=False)
sec = axes[1].secondary_yaxis("right", functions=(lambda v: v / BASE, lambda r: r * BASE))
rt = np.array([1.0, 1.02, 1.05, 1.1, 1.15, 1.2]); sec.set_yticks(rt); sec.set_yticklabels([f"×{r:.2f}" for r in rt]); sec.yaxis.set_minor_formatter(NullFormatter()); sec.set_ylabel("multiple of Global MWPM alone")
axes[1].spines["right"].set_visible(True)
fig.supxlabel("Global MWPM decode time on the residual, % of the untouched syndrome", fontsize=11, y=0.02)
fig.suptitle(f"Logical Error Rate vs L2 Decode Time   ·   d={c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, SI1000 p={c['p']}, {N:,} paired shots", fontsize=11.5, y=0.985)
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig(out, dpi=180); fig.savefig(out.with_suffix(".pdf")); print("wrote", out)
