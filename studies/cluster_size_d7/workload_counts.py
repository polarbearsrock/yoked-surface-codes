"""Per-rule residual workload counts: active detectors per d-round block window (Zero-G-style workload and capacity)
and per patch-basis window (Astrea compatibility screen, yoke detectors excluded).  Counts only; no latency."""
import json, sys
from pathlib import Path
import numpy as np, gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
cell = Path(sys.argv[1]); prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; N, D = prov["shots"], prov["num_detectors"]; d, R = c["d"], c["rounds"]; W = R // d
circuit = yoked_magic_memory_circuit(patch_diameter=d, rounds=R, noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
coords = circuit.get_detector_coordinates(); xyz = np.array([coords[i] for i in range(D)], dtype=float); x, y, t = xyz[:, 0], xyz[:, 1], xyz[:, 2]
yoke = y < -1.0; patch = np.where(yoke, -1, np.floor((x + 0.5) / (d + 1)).astype(int)); basis = np.where(yoke, -1, (np.round(x + y) % 2).astype(int)); window = np.minimum(np.floor(t / d).astype(int), W - 1)
lanes = sorted(set(zip(patch[~yoke], basis[~yoke]))); units = [np.flatnonzero((patch == p) & (basis == b) & (window == w)) for p, b in lanes for w in range(W)]; wins = [np.flatnonzero(window == w) for w in range(W)]
z = np.load(cell / "cell.npz"); unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little")
cs, size, bdry, committed, margin, diam = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"], z["component_diameter"]
ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep = np.diff(ids_off); interior = ~bdry; lane_owned = int(size.sum())
def residual_for(mask):
    r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
    if len(sel):
        det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep[sel]), det), 1)
    return r
rules = [("none", None, np.zeros(len(size), bool))]
rules += [("size", k, interior & ((size <= k) if k is not None else True)) for k in list(range(2, 17, 2)) + [20, 24, None]]
rules += [("margin", round(0.1 * i, 2), interior & ((margin > 0.1 * i) if i > 0 else True)) for i in range(0, 19)]
rules += [("diameter", h, interior & ((diam <= h) if h is not None else True)) for h in list(range(1, 11)) + [None]]
rules += [("all", None, np.ones(len(size), bool))]
rows = []
for fam, v, m in rules:
    r = residual_for(m); n = np.stack([r[:, i].sum(axis=1) for i in wins], axis=1).astype(np.int64); nf = n.ravel()
    u = np.stack([r[:, i].sum(axis=1) for i in units], axis=1).astype(np.int64); uf = u.ravel(); umax = u.max(axis=1)
    rows.append(dict(family=fam, value=v, coverage_pct=100 * float(size[m & committed].sum()) / lane_owned, residual_per_shot=float(r.sum() / N),
                     block=dict(mean=float(nf.mean()), p99=float(np.percentile(nf, 99)), max=int(nf.max()), edges_mean=float((nf * (nf - 1) / 2).mean()), within10_pct=float(100 * (nf <= 10).mean()), within128_pct=float(100 * (nf <= 128).mean()),
                                shot_max_mean=float(n.max(axis=1).mean()), shot_max_p99=float(np.percentile(n.max(axis=1), 99))),
                     unit=dict(mean=float(uf.mean()), p99=float(np.percentile(uf, 99)), max=int(uf.max()), le10_pct=float(100 * (uf <= 10).mean()), shot_max_mean=float(umax.mean()), shot_max_p99=float(np.percentile(umax, 99)), shots_all_le10_pct=float(100 * (umax <= 10).mean()))))
    print(f"{fam}:{v}  block n mean {rows[-1]['block']['mean']:.1f} p99 {rows[-1]['block']['p99']:.0f}  within128 {rows[-1]['block']['within128_pct']:.1f}%  unit<=10 {rows[-1]['unit']['le10_pct']:.1f}%  shots all<=10 {rows[-1]['unit']['shots_all_le10_pct']:.2f}%", flush=True)
json.dump(dict(cell=c, shots=N, window_rounds=d, windows_per_shot=W, lanes=len(lanes), units_per_shot=len(units), capacity=128, fast_path=10, astrea_cap=10, rows=rows), open(cell / "workload_counts.json", "w"), indent=1); print("saved")
