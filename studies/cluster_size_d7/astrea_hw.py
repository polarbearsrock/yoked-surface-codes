"""Residual Hamming weight per Astrea decoding unit (one patch, one basis, one d-round window) for each commit rule.

Astrea (ISCA'23) brute-forces all perfect matchings of a syndrome with Hamming weight <= 10 (945 matchings at HW 10)
and cannot decode higher weights in real time; Astrea-G extends the reach greedily.  So for an Astrea L2 the cost
axis is the residual HW per decoding unit, and the deadline is set by the largest unit in a shot.
"""
import json, sys
from pathlib import Path
import numpy as np, gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; N, D, O = prov["shots"], prov["num_detectors"], prov["num_observables"]; d = c["d"]
circuit = yoked_magic_memory_circuit(patch_diameter=d, rounds=c["rounds"], noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=not c["remove_x_yoke"] is False)
coords = circuit.get_detector_coordinates(); xyz = np.array([coords[i] for i in range(D)], dtype=float)
x, y, t = xyz[:, 0], xyz[:, 1], xyz[:, 2]
yoke = (y < -1.0)  # the yoke detectors sit below the patches
patch = np.where(yoke, -1, np.floor((x + 0.5) / (d + 1)).astype(int))
basis = np.where(yoke, -1, (np.round(x + y) % 2).astype(int))  # checkerboard: the two stabiliser types alternate
window = np.minimum(np.floor(t / d).astype(int), c["rounds"] // d - 1)  # d-round windows; the final layer joins the last
print("yoke detectors:", int(yoke.sum()), "at", sorted(set(map(tuple, xyz[yoke].tolist()))))
lanes = sorted(set(zip(patch[~yoke], basis[~yoke]))); print("lanes:", len(lanes), "detectors per lane:", sorted({int(((patch == p) & (basis == b)).sum()) for p, b in lanes}))
W = c["rounds"] // d; units = [(p, b, w) for (p, b) in lanes for w in range(W)]
idx = [np.flatnonzero((patch == p) & (basis == b) & (window == w)) for p, b, w in units]; print("units per shot:", len(units), "detectors per unit:", sorted({len(i) for i in idx})[:3], "...", sorted({len(i) for i in idx})[-2:])

z = np.load(cell / "cell.npz"); unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little")
cs, size, bdry, committed, margin = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"]
ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep_counts = np.diff(ids_off); interior = ~bdry
def residual_for(mask):
    r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
    if len(sel):
        det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep_counts[sel]), det), 1)
    return r
rules = [("nothing committed", np.zeros(len(size), bool)), ("margin > 1.5, interior", interior & (margin > 1.5)), ("margin > 1.0, interior", interior & (margin > 1.0)), ("margin > 0.5, interior", interior & (margin > 0.5)),
         ("size <= 2, interior", interior & (size <= 2)), ("interior, any size", interior), ("everything committed", np.ones(len(size), bool))]
rows = []
for name, m in rules:
    r = residual_for(m); hw = np.stack([r[:, i].sum(axis=1) for i in idx], axis=1).astype(np.int32)  # shots x units
    per_shot_max = hw.max(axis=1); flat = hw.ravel()
    rows.append(dict(rule=name, residual_per_shot=float(r.sum() / N), unit_mean=float(flat.mean()), unit_p99=float(np.percentile(flat, 99)), unit_max=int(flat.max()),
                     units_over_10_pct=float(100 * (flat > 10).mean()), units_over_6_pct=float(100 * (flat > 6).mean()),
                     shot_max_mean=float(per_shot_max.mean()), shot_max_p99=float(np.percentile(per_shot_max, 99)), shot_max_max=int(per_shot_max.max()),
                     shots_with_unit_over_10_pct=float(100 * (per_shot_max > 10).mean()), shots_all_units_le_10_pct=float(100 * (per_shot_max <= 10).mean())))
    print(f"{name:26s} res/shot {rows[-1]['residual_per_shot']:6.1f} | unit HW mean {rows[-1]['unit_mean']:5.2f} p99 {rows[-1]['unit_p99']:4.0f} max {rows[-1]['unit_max']:3d} | >10: {rows[-1]['units_over_10_pct']:5.2f}% of units, {rows[-1]['shots_with_unit_over_10_pct']:5.2f}% of shots | per-shot worst unit mean {rows[-1]['shot_max_mean']:5.2f} p99 {rows[-1]['shot_max_p99']:4.0f} max {rows[-1]['shot_max_max']}", flush=True)
json.dump(dict(cell=c, shots=N, unit="one patch, one basis, one d-round window", windows_per_lane=W, lanes=len(lanes), rows=rows), open(cell / "astrea_hw.json", "w"), indent=1); print("saved", cell / "astrea_hw.json")
