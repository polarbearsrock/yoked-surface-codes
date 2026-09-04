"""Structural proxies of sparse-blossom work on each rule's residual, from the matching output alone.

Following Higgott & Gidney (sparse blossom, sec. 5 and app. B): the cost is a sum over "cluster regions"
(connected components of the subgraph the algorithm touches), polynomial in each region's size; a region of
radius r sweeps ~ r^3 edges; the tail comes from large alternating trees and blossoms.  Per shot and rule:
  q            residual detection events
  W            total matching weight (sum of correction-edge weights)
  path_w_max   heaviest connected component of the correction subgraph (longest match, ~ largest region radius x2)
  volume       sum over correction components of (w_c/2)^3 (swept-volume proxy in weight units)
  q_max        residual events in the largest cluster region (correction subgraph dilated by one hop)
  sum_q2       sum over cluster regions of q_c^2 (tree/blossom work proxy)
  n_touched    nodes on or adjacent to the correction (flooder work proxy)
"""
import argparse, json, multiprocessing, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import gen, stim
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import compile_layout

_G = {}
def _init(dem_text, residual_by_rule):
    dem = stim.DetectorErrorModel(dem_text); layout = compile_layout(dem, mode="fullhistory")
    graph = compile_matching_graph(dem, layout, require_zero_frame=False, retain_cross_lane_edges=True)
    D = dem.num_detectors; adj = [[] for _ in range(D)]; w = {}
    for e in graph.edges:
        if e.target is None: w[(e.source, -1)] = e.weight
        else: w[(min(e.source, e.target), max(e.source, e.target))] = e.weight; adj[e.source].append(e.target); adj[e.target].append(e.source)
    _G.update(matcher=graph.matcher, adj=adj, w=w, D=D, residual=residual_by_rule)

def _find(parent, a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
def _union(parent, a, b):
    a, b = _find(parent, a), _find(parent, b)
    if a != b: parent[b] = a

def _shot_proxies(row):
    m, adj, w, D = _G["matcher"], _G["adj"], _G["w"], _G["D"]
    events = np.flatnonzero(row); q = len(events)
    if q == 0: return (0, 0.0, 0.0, 0.0, 0, 0, 0)
    edges = m.decode_to_edges_array(row)
    # correction components (boundary edges get a private virtual node so they do not merge everything)
    parent = {}; cw = {}
    def node(x):
        if x not in parent: parent[x] = x
        return x
    W = 0.0; virt = D
    for u, v in edges:
        u = int(u); v = int(v)
        if v < 0 or u < 0:
            s = u if v < 0 else v; key = (s, -1); wt = w[key]; node(s); node(virt); _union(parent, s, virt); virt += 1
        else:
            key = (min(u, v), max(u, v)); wt = w[key]; node(u); node(v); _union(parent, u, v)
        W += wt; cw[key] = wt
    comp_w = {}
    for (a, b), wt in cw.items():
        r = _find(parent, a); comp_w[r] = comp_w.get(r, 0.0) + wt
    path_w_max = max(comp_w.values(), default=0.0); volume = sum((x / 2) ** 3 for x in comp_w.values())
    # cluster regions: correction nodes + residual events, dilated by one graph hop
    parent2 = {}
    seeds = set(int(x) for x in events); seeds.update(int(u) for u, v in edges if u >= 0); seeds.update(int(v) for u, v in edges if v >= 0)
    for s in seeds: parent2.setdefault(s, s)
    touched = set(seeds)
    for s in list(seeds):
        for nb in adj[s]:
            touched.add(nb); parent2.setdefault(nb, nb); _union(parent2, s, nb)
    qc = {}; nc = {}
    for x in touched:
        r = _find(parent2, x); nc[r] = nc.get(r, 0) + 1
    for e in events:
        r = _find(parent2, int(e)); qc[r] = qc.get(r, 0) + 1
    q_max = max(qc.values()); sum_q2 = sum(v * v for v in qc.values()); n_touched = len(touched)
    return (q, W, path_w_max, volume, q_max, sum_q2, n_touched)

def _range(job):
    ri, start, stop = job; res = _G["residual"][ri]
    out = np.zeros((stop - start, 7)); 
    for i in range(stop - start): out[i] = _shot_proxies(res[start + i])
    return ri, start, out

FIELDS = ("q", "W", "path_w_max", "volume", "q_max", "sum_q2", "n_touched")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", type=Path, required=True); ap.add_argument("--processes", type=int, default=16); ap.add_argument("--shots", type=int, default=None); ap.add_argument("--out", type=str, default="l2_structure"); a = ap.parse_args()
    prov = json.load(open(a.cell / "provenance.json")); z = np.load(a.cell / "cell.npz"); p = prov["cell"]["p"]; D = prov["num_detectors"]; N = a.shots or prov["shots"]
    circuit = yoked_magic_memory_circuit(patch_diameter=7, rounds=28, noise=gen.NoiseModel.si1000(p), style="cz", yokes=2, num_patches=6, remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    unpacked = np.unpackbits(z["detectors"][:N], axis=1, count=D, bitorder="little")
    cs, size, bdry, committed, margin, diam = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"], z["component_diameter"]
    ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep_counts = np.diff(ids_off); interior = ~bdry
    inN = cs < N
    def residual_for(mask):
        r = unpacked.copy(); sel = np.flatnonzero(mask & committed & inN)
        if len(sel):
            det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep_counts[sel]), det), 1)
        return r
    rules = [dict(family="none", value=None, mask=np.zeros(len(size), bool))]
    for k in [2, 4, 6, 8, 10, 12, None]: rules.append(dict(family="size", value=k, mask=interior & ((size <= k) if k is not None else True)))
    for tau in [round(0.1 * i, 1) for i in range(0, 19)]: rules.append(dict(family="margin", value=tau, mask=interior & ((margin > tau) if tau > 0 else True)))
    for h in list(range(1, 11)) + [None]: rules.append(dict(family="diameter", value=h, mask=interior & ((diam <= h) if h is not None else True)))
    rules.append(dict(family="all", value=None, mask=np.ones(len(size), bool)))
    if a.shots: rules = [rules[0], rules[1], rules[8], rules[-1]]
    residual_by_rule = [residual_for(r["mask"]) for r in rules]
    print(f"{len(rules)} rules over {N} shots with {a.processes} workers", flush=True)
    chunk = 1000; jobs = [(ri, s, min(N, s + chunk)) for ri in range(len(rules)) for s in range(0, N, chunk)]
    prox = np.zeros((len(rules), N, 7)); t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.processes, mp_context=multiprocessing.get_context("fork"), initializer=_init, initargs=(str(dem), residual_by_rule)) as pool:
        for ri, start, out in pool.map(_range, jobs, chunksize=1): prox[ri, start:start + len(out)] = out
    print(f"computed in {time.perf_counter() - t0:.0f}s", flush=True)
    rows = []
    for ri, r in enumerate(rules):
        d = {"family": r["family"], "value": r["value"]}
        for fi, f in enumerate(FIELDS):
            v = prox[ri, :, fi]; b = prox[0, :, fi]
            d[f] = dict(mean=float(v.mean()), p99=float(np.percentile(v, 99)), max=float(v.max()), rel_mean=float(v.mean() / b.mean()) if b.mean() else None, rel_p99=float(np.percentile(v, 99) / np.percentile(b, 99)) if np.percentile(b, 99) else None, rel_max=float(v.max() / b.max()) if b.max() else None)
        rows.append(d)
    np.savez_compressed(a.cell / f"{a.out}.npz", proxies=prox.astype(np.float32), rules=np.array([f"{r['family']}:{r['value']}" for r in rules]), fields=np.array(FIELDS))
    json.dump(dict(cell=prov["cell"], shots=N, fields=FIELDS, rows=rows), open(a.cell / f"{a.out}.json", "w"), indent=1)
    print(f"{'rule':16s} " + " ".join(f"{f:>11s}" for f in FIELDS) + "   (mean, relative to nothing committed)")
    for d in rows: print(f"{d['family']+':'+str(d['value']):16s} " + " ".join(f"{(d[f]['rel_mean'] if d[f]['rel_mean'] is not None else float('nan')):11.3f}" for f in FIELDS))
    print("saved", a.cell / f"{a.out}.json", flush=True)
