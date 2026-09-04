"""Residual active detectors per d-round decoding window for the whole yoked block, per rule: Zero-G's cost axis and capacity check.

Zero-G (Wegmann et al. 2026): runtime scales with the number of active detectors in the residual; fast path for <= 10,
generic path up to 128 per instance (their configured capacity); ~quadratic in the count above the predecoder's regime.
"""
import json, sys
from pathlib import Path
import numpy as np, gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; N, D = prov["shots"], prov["num_detectors"]; d, R = c["d"], c["rounds"]; W = R // d
circuit = yoked_magic_memory_circuit(patch_diameter=d, rounds=R, noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
coords = circuit.get_detector_coordinates(); t = np.array([coords[i][2] for i in range(D)]); window = np.minimum(np.floor(t / d).astype(int), W - 1)
z = np.load(cell / "cell.npz"); unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little")
cs, size, bdry, committed, margin = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"]
ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep = np.diff(ids_off); interior = ~bdry
def residual_for(mask):
    r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
    if len(sel):
        det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep[sel]), det), 1)
    return r
rules = [("original syndrome", np.zeros(len(size), bool)), ("margin > 1.5, interior", interior & (margin > 1.5)), ("margin > 1.0, interior", interior & (margin > 1.0)), ("margin > 0.5, interior", interior & (margin > 0.5)),
         ("size <= 2, interior", interior & (size <= 2)), ("size <= 4, interior", interior & (size <= 4)), ("interior, any size", interior), ("everything committed", np.ones(len(size), bool))]
idx = [np.flatnonzero(window == w) for w in range(W)]; rows = []
print(f"block-wide {d}-round windows: {W} per shot ({c['patches']} patches x 2 bases, 2 yoke detectors)\n")
print(f"{'rule':24s} {'active/window mean':>18s} {'p99':>5s} {'max':>5s} | {'<=10 fast path':>14s} {'<=128 capacity':>14s} | {'rel n':>6s} {'rel n^2':>7s} | {'worst window/shot mean':>22s} {'max':>4s}")
for name, m in rules:
    r = residual_for(m); n = np.stack([r[:, i].sum(axis=1) for i in idx], axis=1).astype(np.int64); flat = n.ravel(); worst = n.max(axis=1)
    rows.append(dict(rule=name, mean=float(flat.mean()), p99=float(np.percentile(flat, 99)), max=int(flat.max()), fast_path_pct=float(100 * (flat <= 10).mean()), within_capacity_pct=float(100 * (flat <= 128).mean()),
                     worst_mean=float(worst.mean()), worst_max=int(worst.max()), mean_sq=float((flat.astype(float) ** 2).mean())))
    b = rows[0]; x = rows[-1]
    print(f"{name:24s} {x['mean']:18.1f} {x['p99']:5.0f} {x['max']:5d} | {x['fast_path_pct']:13.1f}% {x['within_capacity_pct']:13.1f}% | {x['mean']/b['mean']:6.2f} {x['mean_sq']/b['mean_sq']:7.2f} | {x['worst_mean']:22.1f} {x['worst_max']:4d}")
json.dump(dict(cell=c, shots=N, window_rounds=d, windows_per_shot=W, rows=rows), open(cell / "zerog_capacity.json", "w"), indent=1); print("saved", cell / "zerog_capacity.json")
