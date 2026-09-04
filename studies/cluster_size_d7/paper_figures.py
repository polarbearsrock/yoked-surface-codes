"""Paper-style figures (Yoked Surface Codes Fig. 8 conventions) for the commit-rule study in the paper's unit.

y-axis everywhere: logical error rate per patch round on a log scale, as in the paper's figures; explicit tick
labels because the data spans less than one decade; right-hand axis gives the same values as a multiple of
Global MWPM alone.  Reads frontier_interior.json, l2_timing.json and provenance.json of one cell.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
import sinter

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/figures")
out.mkdir(parents=True, exist_ok=True)
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json")); tim = json.load(open(cell / "l2_timing.json"))
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
gbase = fr["global_failures"] / N; BASE = conv(gbase)
ALL = conv(prov["summary"]["treatment_failures"] / N)  # everything committed, walls included (the frozen v2 arm)
rows = fr["rows"]
def ladder(fam, keep=lambda r: True):
    rs = [r for r in rows if r["family"] == fam and r["coverage_pct"] > 0 and keep(r)]
    rs.sort(key=lambda r: r["coverage_pct"])
    return dict(cov=np.array([r["coverage_pct"] for r in rs]), y=np.array([conv(r["failures"] / N) for r in rs]),
                lo=np.array([conv(gbase + r["ci_pp"][0] / 100) for r in rs]), hi=np.array([conv(gbase + r["ci_pp"][1] / 100) for r in rs]),
                val=[r["value"] for r in rs], key=[f"{fam}:{r['value']}" for r in rs])
L = dict(size=ladder("size", lambda r: r["value"] is None or r["value"] % 2 == 0), margin=ladder("margin"), diameter=ladder("diameter"))
T = {f"{r['family']}:{r['value']}": r for r in tim["rows"]}
STYLE = dict(size=dict(color="C0", marker="o", label="size cap, interior clusters"), margin=dict(color="C1", marker="v", label="margin threshold, interior clusters"), diameter=dict(color="C2", marker="*", label="diameter cap, interior clusters"))
subtitle = f"d={c['d']}, patches={c['patches']}+{c['yokes']} yokes, rounds={c['rounds']}, gateset={c['style']}, noise={c['noise']}, p={c['p']}, {N:,} paired shots"
def paper_axis(ax, lo=2.3e-3, hi=3.9e-3, base_label="right"):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ticks = np.arange(2.4e-3, hi, 2e-4); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t*1e3:.1f}×10$^{{-3}}$" for t in ticks])
    ax.set_ylabel("Logical Error Rate (per patch round)")
    ax.grid(True, which="major", alpha=0.6); ax.grid(True, which="minor", alpha=0.25)
    ax.axhline(BASE, color="k", lw=1)
    if base_label == "right": ax.text(0.99, BASE, f"Global MWPM alone  {BASE*1e3:.2f}×10⁻³ ", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=9)
    else: ax.text(0.01, BASE * 0.998, f" Global MWPM alone  {BASE*1e3:.2f}×10⁻³", transform=ax.get_yaxis_transform(), va="top", ha="left", fontsize=9)
    sec = ax.secondary_yaxis("right", functions=(lambda v: v / BASE, lambda r: r * BASE))
    rt = np.array([1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5]); sec.set_yticks(rt); sec.set_yticklabels([f"×{r:.2f}" for r in rt]); sec.yaxis.set_minor_formatter(NullFormatter())
    sec.set_ylabel("multiple of Global MWPM alone")
    return sec

# ---- Figure 1: coverage vs LER per patch round -----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.5))
for fam in ("size", "margin", "diameter"):
    d = L[fam]; s = STYLE[fam]
    ax.errorbar(d["cov"], d["y"], yerr=[d["y"] - d["lo"], d["hi"] - d["y"]], color=s["color"], marker=s["marker"], ms=6, lw=1.5, capsize=2, label=s["label"])
ax.plot([100 * prov["summary"].get("coverage_pct", 84.2) if False else 84.2], [ALL], marker="s", mfc="none", color="C3", ms=8, ls="none", label="everything committed, walls included")
for fam, picks in (("size", [2, 4, None]), ("margin", [0.5, 1.0, 1.5])):
    d = L[fam]
    for v, cov, y in zip(d["val"], d["cov"], d["y"]):
        if v in picks:
            lab = ("any size" if v is None else f"size ≤ {v}") if fam == "size" else f"margin > {v:g}"
            ax.annotate(f"{lab}  ×{y/BASE:.3f}", (cov, y), xytext=(-8 if fam == "size" else 8, 6 if fam == "size" else -13), textcoords="offset points", fontsize=8, ha="right" if fam == "size" else "left")
ax.annotate(f"everything committed  ×{ALL/BASE:.3f}", (84.2, ALL), xytext=(-10, 6), textcoords="offset points", fontsize=8, ha="right")
paper_axis(ax); ax.set_xlim(0, 100); ax.set_xlabel("coverage: share of detector events removed by the L1 frontend before Global MWPM (%)")
ax.set_title("Logical Error Rate vs L1 Coverage\n" + subtitle, fontsize=11); ax.legend(loc="upper left", fontsize=9)
ax.text(0.99, 0.02, f"paper's 1D fit at p=0.001 for this block: {6*28*8**-7/500:.1e} per patch round (off scale)", transform=ax.transAxes, ha="right", fontsize=8, color="0.35")
fig.tight_layout(); fig.savefig(out / "fig_coverage_vs_ler_paper_unit.png", dpi=180); fig.savefig(out / "fig_coverage_vs_ler_paper_unit.pdf"); plt.close(fig)

# ---- Figure 2: the colleague's plot: x = L2 decode time, mean and max as separate series -----
fig, ax = plt.subplots(figsize=(9, 6.5))
for fam in ("size", "margin"):
    d = L[fam]; s = STYLE[fam]
    timed = [i for i, k in enumerate(d["key"]) if k in T]  # the timing run covered a subset of the ladder steps
    for stat, ls, mfc, tag in (("mean", "-", s["color"], "mean"), ("max", "--", "none", "maximum")):
        xs = np.array([100 * T[d["key"][i]][f"rel_{stat}"] for i in timed])
        ax.plot(xs, d["y"][timed], ls=ls, color=s["color"], marker=s["marker"], mfc=mfc, ms=6, lw=1.4, label=f"{fam} ladder, {tag} L2 time")
ax.plot([100], [BASE], "ko", ms=6, label="nothing committed (100%)")
ax.axhspan(BASE, BASE * 1.01, color="0.5", alpha=0.15, lw=0); ax.text(31, BASE * 1.011, "within 1% of Global MWPM alone", fontsize=8, color="0.3", va="bottom")
for v in (1.5, 1.0, 0.5):
    i = L["margin"]["val"].index(v); k = L["margin"]["key"][i]; y = L["margin"]["y"][i]
    ax.annotate(f"margin > {v:g}", (100 * T[k]["rel_max"], y), xytext=(6, 8) if v == 0.5 else (-4, -13), textcoords="offset points", fontsize=8, ha="left" if v == 0.5 else "right", color="C1")
    ax.annotate("", (100 * T[k]["rel_max"], y), (100 * T[k]["rel_mean"], y), arrowprops=dict(arrowstyle="-", color="C1", alpha=0.35, lw=0.8))
for v in (2, None):
    i = L["size"]["val"].index(v); k = L["size"]["key"][i]; y = L["size"]["y"][i]
    ax.annotate("size ≤ 2" if v == 2 else "any size", (100 * T[k]["rel_max"], y), xytext=(6, -13) if v == 2 else (-6, 6), textcoords="offset points", fontsize=8, ha="left" if v == 2 else "right", color="C0")
    ax.annotate("", (100 * T[k]["rel_max"], y), (100 * T[k]["rel_mean"], y), arrowprops=dict(arrowstyle="-", color="C0", alpha=0.35, lw=0.8))
paper_axis(ax, hi=3.1e-3, base_label="left"); ax.set_xlim(30, 105); ax.set_xlabel(f"Global MWPM decode time on the residual, % of the untouched syndrome  (baseline mean {T['none:None']['mean_ms']:.2f} ms, max {T['none:None']['max_ms']:.2f} ms)")
ax.set_title("Logical Error Rate vs L2 Decode Time, mean and maximum as separate series\n" + subtitle, fontsize=11); ax.legend(loc="upper right", fontsize=9)
ax.text(0.99, 0.02, "PyMatching per-shot decode of the residual, batch of one, min over reps; thin lines join a rule's mean to its maximum", transform=ax.transAxes, ha="right", fontsize=8, color="0.35")
fig.tight_layout(); fig.savefig(out / "fig_l2_time_vs_ler_paper_unit.png", dpi=180); fig.savefig(out / "fig_l2_time_vs_ler_paper_unit.pdf"); plt.close(fig)

# ---- Figure 3: the arms table as a dot plot ---------------------------------------------------
arms = [("Global MWPM alone", 0.0, BASE, BASE, BASE)]
for fam, v, lab in (("margin", 1.5, "margin > 1.5, interior"), ("margin", 1.0, "margin > 1.0, interior"), ("size", 2, "size ≤ 2, interior"), ("size", None, "interior, any size")):
    d = L[fam]; i = d["val"].index(v); arms.append((lab, d["cov"][i], d["y"][i], d["lo"][i], d["hi"][i]))
arms.append(("everything committed, walls included", 84.2, ALL, ALL, ALL))
fig, ax = plt.subplots(figsize=(9, 4.6))
ys = np.arange(len(arms))[::-1]
for (lab, cov, y, lo, hi), yy in zip(arms, ys):
    ax.errorbar([y], [yy], xerr=[[y - lo], [hi - y]], fmt="o", color="C0" if lab != "Global MWPM alone" else "k", capsize=3)
    ax.text(y, yy + 0.22, f"{y*1e3:.2f}×10⁻³   ×{y/BASE:.3f}   {cov:.0f}% coverage", fontsize=8, ha="center", va="bottom")
ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in arms]); ax.set_ylim(-0.6, len(arms) - 0.2); ax.set_xscale("log"); ax.set_xlim(2.3e-3, 4.15e-3)
xt = np.arange(2.4e-3, 4.15e-3, 4e-4); ax.set_xticks(xt); ax.set_xticklabels([f"{t*1e3:.1f}×10$^{{-3}}$" for t in xt]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.axvline(BASE, color="k", lw=1); ax.grid(True, axis="x", which="major", alpha=0.6); ax.grid(True, axis="x", which="minor", alpha=0.25)
ax.set_xlabel("Logical Error Rate (per patch round), paired 95% CI"); ax.set_title(f"Decoder Arms in the Paper's Unit\nd={c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, p={c['p']}, {N:,} paired shots", fontsize=11)
fig.tight_layout(); fig.savefig(out / "fig_arms_paper_unit.png", dpi=180); fig.savefig(out / "fig_arms_paper_unit.pdf"); plt.close(fig)
print("wrote", sorted(p.name for p in out.iterdir()))
