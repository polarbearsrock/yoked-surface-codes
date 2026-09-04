"""Pool the per-cell analyses into one cross-p summary (markdown + JSON)."""
import json, sys
from pathlib import Path
root = Path("out/cluster_size_study_d7"); cells = sorted(root.glob("d7_p*/analysis.json"))
A = [json.load(open(c)) for c in cells]; ps = [a["cell"]["p"] for a in A]
def row(rule, key="risk_difference", scale=100, fmt="{:.2f}"):
    return [fmt.format(scale * next(r for r in a["rules"] if r["rule"] == rule)[key]) for a in A]
L = ["# d=7 cluster-size study, p sweep (10k shots per cell, non-claim-bearing)", "", "| | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
L.append("| Global MWPM per-shot failure % | " + " | ".join(f"{100*a['global_failures']/a['shots']:.1f}" for a in A) + " |")
L.append("| Global MWPM per-observable failure % | " + " | ".join(f"{100*a['global_observable_failures']/(a['shots']*12):.1f}" for a in A) + " |")
L.append("| Event density % | " + " | ".join(f"{100*a['lane_owned_detector_events']/(a['shots']*8354):.1f}" for a in A) + " |")
L += ["", "## Regression vs Global MWPM, per shot, pp (positive = treatment worse)", "", "| commit rule | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
for rule in ("all committed", "interior, any size", "size<=4, interior", "size<=2, interior", "size<=2, interior, diameter<=1", "size<=1", "margin>0.5", "margin>1.0", "margin>1.5", "size<=2, interior, margin>0.5"):
    L.append(f"| {rule} | " + " | ".join(row(rule)) + " |")
L += ["", "## Same, per observable, pp", "", "| commit rule | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
for rule in ("all committed", "interior, any size", "size<=2, interior", "size<=1", "margin>0.5", "margin>1.0"):
    L.append(f"| {rule} | " + " | ".join(row(rule, "observable_risk_difference")) + " |")
L += ["", "## Coverage %", "", "| commit rule | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
for rule in ("all committed", "interior, any size", "size<=2, interior", "margin>0.5", "margin>1.0"):
    L.append(f"| {rule} | " + " | ".join(row(rule, "coverage", 100, "{:.1f}")) + " |")
L += ["", "## Marginal risk, interior clusters by size: net regressions per 1,000 added committed defects (per shot / per observable)", "", "| size step | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
steps = ["<=2", "3-4", "5-6", "7-8", "9+"]
for i, s in enumerate(steps):
    L.append(f"| {s} | " + " | ".join(f"{a['ladders']['interior_by_size'][i]['per_1000']:.2f} / {a['ladders']['interior_by_size'][i]['observable_per_1000']:.2f}" for a in A) + " |")
L.append("| + wall clusters | " + " | ".join(f"{a['ladders']['add_wall_clusters'][1]['per_1000']:.2f} / {a['ladders']['add_wall_clusters'][1]['observable_per_1000']:.2f}" for a in A) + " |")
L.append("| wall singletons alone | " + " | ".join(f"{a['ladders']['wall_singletons'][0]['per_1000']:.2f} / {a['ladders']['wall_singletons'][0]['observable_per_1000']:.2f}" for a in A) + " |")
L += ["", "## Culprit rate: share of committed components in regressed shots whose restoration alone fixes the shot", "", "| size band | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
for i in range(7):
    band = A[0]["attribution"]["by_size"][i]["band"]
    L.append(f"| {band} | " + " | ".join((f"{100*a['attribution']['by_size'][i]['rate']:.2f}% (n={a['attribution']['by_size'][i]['components']})" if a['attribution']['by_size'][i]['rate'] is not None else "-") for a in A) + " |")
L += ["", "| interior vs wall | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
for i, label in enumerate(("interior", "wall")):
    L.append(f"| {label} | " + " | ".join(f"{100*a['attribution']['by_wall'][i]['rate']:.2f}%" for a in A) + " |")
L += ["", "| margin band | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---:|" * len(ps)]
for i in range(5):
    band = A[0]["attribution"]["by_margin"][i]["band"]
    L.append(f"| {band} | " + " | ".join((f"{100*a['attribution']['by_margin'][i]['rate']:.2f}%" if a['attribution']['by_margin'][i]['rate'] is not None else "-") for a in A) + " |")
L.append(""); L.append("regressed shots per cell: " + ", ".join(f"p={p}: {a['attribution']['regressed_shots']} ({a['attribution']['shots_without_single_culprit']} without a single-component culprit)" for p, a in zip(ps, A)))
(root / "summary.md").write_text("\n".join(L) + "\n"); json.dump(dict(cells=[str(c) for c in cells]), open(root / "summary.json", "w"))
print("\n".join(L))
