"""Coarse Micro Blossom cycle estimate: original syndrome versus the residual after each UF commit rule.

Model (Wu, Liyanage, Zhong, ASPLOS'25, Sec. 5-6): isolated conflicts (a size-2 cluster, or a singleton reaching the
real boundary, whose growth met nothing else, i.e. margin > 0) are resolved in the accelerator in O(1); every other
defect needs one CPU interaction (~200 ns) plus one grow round (~4 accelerator cycles).  Accelerator clock 62 MHz
(their d=13 prototype); I/O floor 0.37 us per decode.  Everything else is ignored.  Coarse by construction.
"""
import json, sys
from pathlib import Path
import numpy as np, gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
F_MHZ, T_INT_NS, ROUND_CYCLES, IO_US = 62.0, 200.0, 4.0, 0.37
CYC_PER_DEFECT = T_INT_NS * 1e-3 * F_MHZ + ROUND_CYCLES; IO_CYC = IO_US * F_MHZ
prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; N, D = prov["shots"], prov["num_detectors"]; R = c["rounds"]
circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=R, noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
coords = circuit.get_detector_coordinates(); det_round = np.minimum(np.array([coords[i][2] for i in range(D)]).astype(int), R - 1)
z = np.load(cell / "cell.npz")
cs, size, wall, port, committed, margin = z["component_shot"], z["component_size"], z["component_boundary"], z["component_port"], z["component_committed"], z["component_margin"]
ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep = np.diff(ids_off)
interior = ~wall & ~port
isolated = (interior & (size == 2) & (margin > 0)) | (wall & ~port & (size == 1) & (margin > 0))   # hardware O(1) in Micro Blossom
print(f"components {len(size):,}; isolated (hardware-resolved) {isolated.sum():,} holding {size[isolated].sum():,} of {size.sum():,} defects ({100*size[isolated].sum()/size.sum():.1f}%)")
def per_shot_and_round(mask):
    """Non-isolated defects per shot, and the busiest measurement round per shot, for the clusters left to the CPU."""
    sel = np.flatnonzero(mask); n_shot = np.bincount(cs[sel], weights=size[sel], minlength=N)
    det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); shots = np.repeat(cs[sel], rep[sel])
    per_round = np.zeros((N, R), np.int32); np.add.at(per_round, (shots, det_round[det]), 1)
    return n_shot, per_round.max(axis=1)
diam = z["component_diameter"]; lane_owned = int(size.sum())
def name(fam, v):
    if fam == "none": return "original syndrome (Micro Blossom alone)"
    if fam == "all": return "everything committed, walls included"
    if fam == "size": return "interior, any size" if v is None else f"interior, size ≤ {v}"
    if fam == "diameter": return "interior, any diameter" if v is None else f"interior, diameter ≤ {v}"
    return "interior, any margin" if v == 0 else f"interior, margin > {v:g}"
rules = [("none", None, np.zeros(len(size), bool))]
rules += [("size", k, interior & ((size <= k) if k is not None else True)) for k in list(range(2, 17, 2)) + [20, 24, None]]
rules += [("margin", round(0.1 * i, 2), interior & ((margin > 0.1 * i) if i > 0 else True)) for i in range(0, 19)]
rules += [("diameter", h, interior & ((diam <= h) if h is not None else True)) for h in list(range(1, 11)) + [None]]
rules += [("all", None, np.ones(len(size), bool))]
rows = []
for fam, v, m in rules:
    left = ~isolated & ~(m & committed)   # clusters the CPU still handles: not hardware-isolated and not pruned by the rule
    n_shot, n_round = per_shot_and_round(left)
    cyc_shot = CYC_PER_DEFECT * n_shot + IO_CYC; cyc_round = CYC_PER_DEFECT * n_round + IO_CYC
    rows.append(dict(family=fam, value=v, rule=name(fam, v), coverage_pct=100 * float(size[m & committed].sum()) / lane_owned,
                     noniso_mean=float(n_shot.mean()), noniso_p99=float(np.percentile(n_shot, 99)), noniso_max=int(n_shot.max()),
                     round_mean=float(n_round.mean()), round_p99=float(np.percentile(n_round, 99)), round_max=int(n_round.max()),
                     cyc_shot_mean=float(cyc_shot.mean()), cyc_shot_p99=float(np.percentile(cyc_shot, 99)), cyc_shot_max=float(cyc_shot.max()),
                     cyc_round_mean=float(cyc_round.mean()), cyc_round_p99=float(np.percentile(cyc_round, 99)), cyc_round_max=float(cyc_round.max()),
                     us_round_mean=float(cyc_round.mean() / F_MHZ), us_round_p99=float(np.percentile(cyc_round, 99) / F_MHZ), us_round_max=float(cyc_round.max() / F_MHZ)))
b = rows[0]
for r in rows:
    for k in ("mean", "p99", "max"): r[f"rel_{k}"] = r[f"cyc_shot_{k}"] / b[f"cyc_shot_{k}"]
print(f"\n{'rule':40s} {'cov':>5s} {'CPU defects/shot':>17s} {'max':>5s} | {'cycles/shot mean':>16s} {'p99':>6s} {'max':>6s} | {'us/round mean':>13s} {'max':>6s} | {'rel mean':>8s} {'rel p99':>7s} {'rel max':>7s}")
for r in rows:
    if r["family"] in ("none", "all") or (r["family"] == "size" and r["value"] in (2, 4, None)) or (r["family"] == "margin" and r["value"] in (0.5, 1.0, 1.5)) or (r["family"] == "diameter" and r["value"] in (1, 4)):
        print(f"{r['rule']:40s} {r['coverage_pct']:4.0f}% {r['noniso_mean']:17.1f} {r['noniso_max']:5d} | {r['cyc_shot_mean']:16.0f} {r['cyc_shot_p99']:6.0f} {r['cyc_shot_max']:6.0f} | {r['us_round_mean']:13.2f} {r['us_round_max']:6.2f} | {100*r['rel_mean']:7.0f}% {100*r['rel_p99']:6.0f}% {100*r['rel_max']:6.0f}%")
json.dump(dict(model=dict(accelerator_mhz=F_MHZ, interaction_ns=T_INT_NS, round_cycles=ROUND_CYCLES, io_floor_us=IO_US, cycles_per_cpu_defect=CYC_PER_DEFECT,
                          isolated_defect_share=float(size[isolated].sum() / size.sum()), source="Micro Blossom, Wu/Liyanage/Zhong ASPLOS 2025, Sec. 5-8; constants borrowed, blossoms and the yoke hub ignored"),
               cell=c, shots=N, rounds=R, rows=rows), open(cell / "microblossom_cycles.json", "w"), indent=1)
print("saved", cell / "microblossom_cycles.json")
