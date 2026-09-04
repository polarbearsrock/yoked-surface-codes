"""Report every arm in the yoked-surface-codes paper's unit: logical error per patch per round.

The paper's yoked-memory figures (step3_plot) convert the per-shot failure with sinter's
shot_error_rate_to_piece_error_rate(pieces = patches * rounds, values = observables).  This script
applies the same conversion to the study cells and writes paper_ler.json per cell plus paper_ler.md.
"""
import json, sys
from pathlib import Path
import sinter

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7")
def conv(p, pieces, values): return sinter.shot_error_rate_to_piece_error_rate(p, pieces=pieces, values=values)
def binom_ci(k, n):
    f = sinter.fit_binomial(num_shots=n, num_hits=k, max_likelihood_factor=1e3)  # ~99.9% likelihood interval, sinter's plotting default
    return f.low, f.high
lines = ["| cell | p | shots | Global MWPM per shot | Global MWPM per patch-round | everything committed per patch-round | multiplier |", "|---|---:|---:|---:|---:|---:|---:|"]
landmark = []
for cell in sorted(root.glob("d7_p*")):
    prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; N = prov["shots"]; O = prov["num_observables"]; pieces = c["patches"] * c["rounds"]
    g, t = prov["summary"]["global_failures"], prov["summary"]["treatment_failures"]
    glo, ghi = binom_ci(g, N); tlo, thi = binom_ci(t, N)
    rec = dict(cell=c, shots=N, pieces=pieces, values=O, conversion="sinter.shot_error_rate_to_piece_error_rate(pieces=patches*rounds, values=observables)",
               global_mwpm=dict(failures=g, per_shot=g / N, per_patch_round=conv(g / N, pieces, O), per_patch_round_ci=[conv(glo, pieces, O), conv(ghi, pieces, O)]),
               everything_committed=dict(failures=t, per_shot=t / N, per_patch_round=conv(t / N, pieces, O), per_patch_round_ci=[conv(tlo, pieces, O), conv(thi, pieces, O)]))
    rec["everything_committed"]["multiplier"] = rec["everything_committed"]["per_patch_round"] / rec["global_mwpm"]["per_patch_round"]
    fi = cell / "frontier_interior.json"
    if fi.exists():
        fr = json.load(open(fi)); base = g / N; rules = []
        for r in fr["rows"]:
            pr = conv(r["failures"] / N, pieces, O); lo = conv(base + r["ci_pp"][0] / 100, pieces, O); hi = conv(base + r["ci_pp"][1] / 100, pieces, O)
            rules.append(dict(family=r["family"], value=r["value"], coverage_pct=r["coverage_pct"], per_shot=r["failures"] / N, per_patch_round=pr, paired_ci=[lo, hi], multiplier=pr / rec["global_mwpm"]["per_patch_round"]))
        rec["rules"] = rules
        for r in rules:
            if (r["family"], r["value"]) in {("size", 2), ("size", 4), ("size", None), ("margin", 0.5), ("margin", 1.0), ("margin", 1.5), ("diameter", 1), ("diameter", 4)}:
                landmark.append((cell.name, r))
    json.dump(rec, open(cell / "paper_ler.json", "w"), indent=1)
    gm, em = rec["global_mwpm"], rec["everything_committed"]
    lines.append(f"| {cell.name} | {c['p']} | {N:,} | {100*gm['per_shot']:.2f}% | {gm['per_patch_round']:.3e} [{gm['per_patch_round_ci'][0]:.2e}, {gm['per_patch_round_ci'][1]:.2e}] | {em['per_patch_round']:.3e} | ×{em['multiplier']:.3f} |")
if landmark:
    lines += ["", "Interior-only commit rules on the 100k cell, paired against Global MWPM alone (paired CI converted endpoint-wise):", "",
              "| rule | coverage | per patch-round | paired 95% CI | multiplier |", "|---|---:|---:|---:|---:|"]
    for name, r in landmark:
        label = f"{r['family']} {'any' if r['value'] is None else r['value']}"
        lines.append(f"| {label} | {r['coverage_pct']:.1f}% | {r['per_patch_round']:.3e} | [{r['paired_ci'][0]:.3e}, {r['paired_ci'][1]:.3e}] | ×{r['multiplier']:.3f} |")
lines += ["", "For scale, the paper's 1D fit p_L/(r n) ≈ r n 8^(-d)/500 at its operating point p = 0.001 gives "
          f"{6*28*8**-7/500:.2e} per patch-round for d = 7, n = 6, r = 28."]
(root / "paper_ler.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
