"""Truncated Helios-style proxy: accelerator cycles consumed by a frontend run capped at K growth iterations.

Mirrors the repo proxy (_patch_uf_hw_proxy): per iteration a fixed 4 cycles (1 growing + 2 controller + 1 merge
settle) plus merge flooding of merge_cycles_per_hop x the largest forest diameter among clusters that have an event in
that iteration (final diameter, an upper bound).  Lanes run in parallel, so the shot's critical path is the slowest
lane.  Reports p50 / p99 / max cycles at each cap K, for hop costs of 1 and 2.
"""
import json, sys
from pathlib import Path
import numpy as np
cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
z = np.load(cell / "cell.npz"); ev = np.load(cell / "event_times.npz"); q = json.load(open(cell / "helios_quantum.json"))["quantum"]
cs, lane, diam, size, absorbed = z["component_shot"], z["component_lane"], z["component_diameter"], z["component_size"], z["component_absorbed"]
times, off = ev["times"], ev["offsets"]; n_comp = len(size); N = int(cs.max()) + 1; L = int(lane.max()) + 1
FIXED = 4
# per event: (shot, lane, iteration, diameter of its component)
ncount = np.diff(off); comp_of_event = np.repeat(np.arange(n_comp), ncount)
it = np.maximum(1, np.ceil(times / q - 1e-12)).astype(np.int64)
d_ev = np.maximum(diam[comp_of_event], 0).astype(np.int64); d_ev = np.where(diam[comp_of_event] < 0, np.maximum(absorbed[comp_of_event] - 1, 0), d_ev)  # censored -> chain bound
key_sl = cs[comp_of_event].astype(np.int64) * L + lane[comp_of_event].astype(np.int64)
KMAX = int(it.max()); print(f"events {len(it):,}; iterations up to {KMAX}; lanes {L}; shots {N}", flush=True)
# max diameter per (shot-lane, iteration) via sorting
order = np.lexsort((-d_ev, it, key_sl)); k_sorted, it_sorted, d_sorted = key_sl[order], it[order], d_ev[order]
first = np.ones(len(order), bool); first[1:] = (k_sorted[1:] != k_sorted[:-1]) | (it_sorted[1:] != it_sorted[:-1])
sl_u, it_u, dmax_u = k_sorted[first], it_sorted[first], d_sorted[first]  # one row per (shot-lane, iteration) with its max diameter
caps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, None]
rows = []
for hop in (1, 2):
    for K in caps:
        k = KMAX if K is None else K; sel = it_u <= k
        merge = np.bincount(sl_u[sel], weights=hop * dmax_u[sel], minlength=N * L)  # merge flooding cycles per lane, iterations <= K
        # iterations actually run per lane: min(K, last event iteration of that lane) -- a lane with no events beyond K stops early
        last_it = np.zeros(N * L, np.int64); np.maximum.at(last_it, sl_u, it_u); ran = np.minimum(last_it, k)
        lane_cycles = (FIXED * ran + merge).reshape(N, L); shot_cycles = lane_cycles.max(axis=1)
        patch_cycles = lane_cycles.reshape(N, L // 2, 2).sum(axis=2).max(axis=1)   # one engine per patch running its X and Z lanes serially
        shared_cycles = lane_cycles.sum(axis=1)                                     # one engine for the whole block; also the aggregate work
        st = lambda a: dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)), max=float(a.max()))
        rows.append(dict(hop=hop, cap=K, **st(shot_cycles), growth_only=None if K is None else FIXED * K, engines12=st(shot_cycles), engines6=st(patch_cycles), engine1=st(shared_cycles)))
        if hop == 1: print(f"K={'∞' if K is None else K:>3}: shot critical path cycles p50 {rows[-1]['p50']:6.0f} p90 {rows[-1]['p90']:6.0f} p99 {rows[-1]['p99']:6.0f} max {rows[-1]['max']:6.0f}   (fixed only: {rows[-1]['growth_only']})", flush=True)
json.dump(dict(model=dict(fixed_cycles_per_iteration=FIXED, quantum=q, w_max=16, schedules="engines12: max over lanes (critical path); engines6: max over patches of the X+Z lane sum; engine1: sum over lanes (= aggregate work)", merge="merge_cycles_per_hop x max final forest diameter among clusters with an event in the iteration (upper bound); censored clusters use absorbed-1"), rows=rows), open(cell / "helios_budget_cycles.json", "w"), indent=1)
print("saved", cell / "helios_budget_cycles.json")
