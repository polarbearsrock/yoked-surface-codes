"""Coarse Zero-G cycle estimate per decoding window (whole yoked block, d rounds) for every commit rule.

Model, from the Zero-G pipeline description (Wegmann et al. 2026, Sec. V-VI; HLS pipeline on a Versal V80 at 250 MHz):
  n = active detectors in the residual window, E = n(n-1)/2 candidate edges
  graph construction : one candidate edge per cycle                       -> E
  greedy matching    : n/2 selections through a comparator tree           -> (n/2) * ceil(log2 E)
  augmentation       : l rounds of 2-augmenting checks over matched pairs -> l * (n/2)^2
  observable fold    : one matched pair per cycle                         -> n/2
  pipeline floor                                                          -> 8 cycles
Calibration check: n = 5 gives ~30 cycles (120 ns) and n = 10 ~125 cycles (500 ns), the order of their reported
~100 ns average and ~200 ns p99 for sparse residuals.  Above their regime the quadratic terms dominate.  Coarse.
"""
import json, sys
from pathlib import Path
import numpy as np, gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
F_MHZ, L_ROUNDS, FLOOR, CAPACITY = 250.0, 2, 8, 128
def cycles(n):
    n = n.astype(np.float64); E = n * (n - 1) / 2; logE = np.ceil(np.log2(np.maximum(E, 1)))
    return FLOOR + E + (n / 2) * logE + L_ROUNDS * (n / 2) ** 2 + n / 2
prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; N, D = prov["shots"], prov["num_detectors"]; d, R = c["d"], c["rounds"]; W = R // d
circuit = yoked_magic_memory_circuit(patch_diameter=d, rounds=R, noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
coords = circuit.get_detector_coordinates(); t = np.array([coords[i][2] for i in range(D)]); window = np.minimum(np.floor(t / d).astype(int), W - 1)
z = np.load(cell / "cell.npz"); unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little")
cs, size, bdry, committed, margin, diam = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"], z["component_diameter"]
ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep = np.diff(ids_off); interior = ~bdry; lane_owned = int(size.sum())
def residual_for(mask):
    r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
    if len(sel):
        det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep[sel]), det), 1)
    return r
def name(fam, v):
    if fam == "none": return "original syndrome (Zero-G alone)"
    if fam == "all": return "everything committed, walls included"
    if fam == "size": return "interior, any size" if v is None else f"interior, size ≤ {v}"
    if fam == "diameter": return "interior, any diameter" if v is None else f"interior, diameter ≤ {v}"
    return "interior, any margin" if v == 0 else f"interior, margin > {v:g}"
rules = [("none", None, np.zeros(len(size), bool))]
rules += [("size", k, interior & ((size <= k) if k is not None else True)) for k in list(range(2, 17, 2)) + [20, 24, None]]
rules += [("margin", round(0.1 * i, 2), interior & ((margin > 0.1 * i) if i > 0 else True)) for i in range(0, 19)]
rules += [("diameter", h, interior & ((diam <= h) if h is not None else True)) for h in list(range(1, 11)) + [None]]
rules += [("all", None, np.ones(len(size), bool))]
idx = [np.flatnonzero(window == w) for w in range(W)]; rows = []
for fam, v, m in rules:
    r = residual_for(m); n = np.stack([r[:, i].sum(axis=1) for i in idx], axis=1); flat = n.ravel(); cyc = cycles(flat); worst = cycles(n).max(axis=1)
    rows.append(dict(family=fam, value=v, rule=name(fam, v), coverage_pct=100 * float(size[m & committed].sum()) / lane_owned,
                     n_mean=float(flat.mean()), n_p99=float(np.percentile(flat, 99)), n_max=int(flat.max()), fast_path_pct=float(100 * (flat <= 10).mean()), within_capacity_pct=float(100 * (flat <= CAPACITY).mean()),
                     cyc_mean=float(cyc.mean()), cyc_p99=float(np.percentile(cyc, 99)), cyc_max=float(cyc.max()), us_mean=float(cyc.mean() / F_MHZ), us_p99=float(np.percentile(cyc, 99) / F_MHZ), us_max=float(cyc.max() / F_MHZ),
                     shot_worst_mean=float(worst.mean()), shot_worst_max=float(worst.max())))
    print(f"{rows[-1]['rule']:40s} cov {rows[-1]['coverage_pct']:4.0f}%  n/window {rows[-1]['n_mean']:6.1f} max {rows[-1]['n_max']:3d}  cycles mean {rows[-1]['cyc_mean']:8.0f} p99 {rows[-1]['cyc_p99']:8.0f} max {rows[-1]['cyc_max']:8.0f}  us mean {rows[-1]['us_mean']:6.1f} max {rows[-1]['us_max']:6.1f}  capacity {rows[-1]['within_capacity_pct']:5.1f}%", flush=True)
b = rows[0]
for r in rows:
    for k in ("mean", "p99", "max"): r[f"rel_{k}"] = r[f"cyc_{k}"] / b[f"cyc_{k}"]
json.dump(dict(model=dict(fpga_mhz=F_MHZ, augmentation_rounds=L_ROUNDS, floor_cycles=FLOOR, capacity=CAPACITY, formula="8 + n(n-1)/2 + (n/2)ceil(log2 E) + 2(n/2)^2 + n/2",
                          source="Zero-G, Wegmann et al. 2026 (arXiv 2608.02030), pipeline of Sec. V-VI; one candidate edge, one comparator selection, one 2-augmenting check per cycle; coarse"),
               cell=c, shots=N, window_rounds=d, windows_per_shot=W, rows=rows), open(cell / "zerog_cycles.json", "w"), indent=1)
print("saved", cell / "zerog_cycles.json")
