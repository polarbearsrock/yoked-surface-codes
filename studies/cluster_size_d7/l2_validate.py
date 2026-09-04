"""Validate the structural proxies against measured per-shot sparse-blossom time and summarise the L2 tail."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
T = np.load(cell / "l2_timing.npz"); S = np.load(cell / "l2_structure.npz")
t_rules = [str(r) for r in T["rules"]]; s_rules = [str(r) for r in S["rules"]]; fields = [str(f) for f in S["fields"]]
assert t_rules == s_rules, (t_rules[:3], s_rules[:3])
times = T["times_ns"].astype(np.float64) / 1e6  # ms
prox = S["proxies"].astype(np.float64)
N = times.shape[1]

def corr_block(ri):
    t = times[ri]; out = {}
    for fi, f in enumerate(fields):
        v = prox[ri, :, fi]
        out[f] = dict(spearman=float(spearmanr(v, t).correlation), pearson=float(pearsonr(v, t)[0]))
    return out

report = dict(shots=N, rules=t_rules, fields=fields)
# 1. per-shot correlation between proxies and measured time, for three reference rules
report["correlations"] = {r: corr_block(t_rules.index(r)) for r in ("none:None", "margin:1.5", "size:2", "all:None") if r in t_rules}
# 2. pooled across all rules (each rule is a different residual distribution): does the proxy track the time across rules?
tm = times.mean(axis=1); tp99 = np.percentile(times, 99, axis=1); tmax = times.max(axis=1)
pm = prox.mean(axis=1); pp99 = np.percentile(prox, 99, axis=1); pmax = prox.max(axis=1)
report["across_rules"] = {f: dict(mean_spearman=float(spearmanr(pm[:, fi], tm).correlation), p99_spearman=float(spearmanr(pp99[:, fi], tp99).correlation), max_spearman=float(spearmanr(pmax[:, fi], tmax).correlation)) for fi, f in enumerate(fields)}
# 3. what do the slowest 1% of baseline shots look like versus the rest?
b = t_rules.index("none:None"); slow = times[b] >= np.percentile(times[b], 99)
report["baseline_slowest_1pct"] = {f: dict(slow_mean=float(prox[b, slow, fi].mean()), rest_mean=float(prox[b, ~slow, fi].mean())) for fi, f in enumerate(fields)}
# 4. the tail question: for each rule, relative max/p99/mean time and the share of baseline-slowest shots that stay slow
tail = []
for ri, r in enumerate(t_rules):
    still_slow = float((times[ri][slow] >= np.percentile(times[b], 99)).mean())
    tail.append(dict(rule=r, rel_mean=float(tm[ri] / tm[b]), rel_p99=float(tp99[ri] / tp99[b]), rel_max=float(tmax[ri] / tmax[b]), baseline_slowest_still_above_baseline_p99=still_slow,
                     max_shot=int(times[ri].argmax()), max_q_max=float(prox[ri, :, fields.index("q_max")].max()), max_path_w=float(prox[ri, :, fields.index("path_w_max")].max())))
report["tail"] = tail
json.dump(report, open(cell / "l2_validation.json", "w"), indent=1)

print(f"per-shot Spearman correlation of proxy with measured time (baseline residual, {N} shots):")
for f, c in report["correlations"]["none:None"].items(): print(f"  {f:12s} rho={c['spearman']:+.3f}  pearson={c['pearson']:+.3f}")
print("\nacross the 39 rules, Spearman of proxy statistic vs time statistic:")
for f, c in report["across_rules"].items(): print(f"  {f:12s} mean {c['mean_spearman']:+.3f}   p99 {c['p99_spearman']:+.3f}   max {c['max_spearman']:+.3f}")
print("\nbaseline slowest 1% of shots vs the rest (mean proxy):")
for f, c in report["baseline_slowest_1pct"].items(): print(f"  {f:12s} slow {c['slow_mean']:10.2f}   rest {c['rest_mean']:10.2f}   ratio {c['slow_mean']/c['rest_mean'] if c['rest_mean'] else float('nan'):5.2f}")
print(f"\n{'rule':16s} {'rel mean':>9s} {'rel p99':>8s} {'rel max':>8s} {'slow stay slow':>15s} {'max q_max':>10s} {'max path w':>11s}")
for d in tail: print(f"{d['rule']:16s} {d['rel_mean']:9.3f} {d['rel_p99']:8.3f} {d['rel_max']:8.3f} {d['baseline_slowest_still_above_baseline_p99']:15.2f} {d['max_q_max']:10.0f} {d['max_path_w']:11.1f}")
print("saved", cell / "l2_validation.json")
