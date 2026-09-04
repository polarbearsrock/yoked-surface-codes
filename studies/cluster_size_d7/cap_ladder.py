"""Growth-time cap ladder: commit only interior clusters that closed within K Helios growth iterations (w_max = 16).

Series: cap alone; cap with margin > 1.0; cap with margin > 0.5.  Each rule's residual is rebuilt from the retained
components and re-decoded with Global MWPM over all shots, exactly as the size and margin ladders were.
Iteration of a cluster = ceil(closing time / quantum), quantum = max lane edge weight / 16 (Helios-style proxy);
the cycle estimate is 3 accelerator cycles per iteration (1 grow + 2 controller), merge flooding excluded.
"""
import argparse, json, math, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, gen, stim
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import GlobalMWPMDecoder
_G = {}
def _init(dem_text): _G["dec"] = GlobalMWPMDecoder().compile_decoder_for_dem(dem=stim.DetectorErrorModel(dem_text))
def _decode(job):
    i, packed = job; return i, _G["dec"].decode_shots_bit_packed(bit_packed_detection_event_data=packed)
def wald(b, c, n):
    rd = (b - c) / n; se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n; return rd, rd - 1.96 * se, rd + 1.96 * se
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", type=Path, required=True); ap.add_argument("--processes", type=int, default=32); a = ap.parse_args(); assert 1 <= a.processes <= 32
    prov = json.load(open(a.cell / "provenance.json")); c = prov["cell"]; z = np.load(a.cell / "cell.npz"); q = json.load(open(a.cell / "helios_quantum.json"))["quantum"]
    D, O, N = prov["num_detectors"], prov["num_observables"], prov["shots"]
    circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=c["rounds"], noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little"); obs = np.unpackbits(z["observables"], axis=1, count=O, bitorder="little").astype(bool)
    gfail = np.any(np.unpackbits(z["global_prediction"], axis=1, count=O, bitorder="little").astype(bool) != obs, axis=1)
    cs, size, bdry, committed, margin = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"]
    ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep = np.diff(ids_off); lane_owned = int(unpacked.sum())
    closing = np.load(a.cell / "closing_time.npy"); it = np.maximum(1, np.ceil(closing / q - 1e-12)).astype(np.int32)
    interior = ~bdry; ic = interior & committed; tot = int(size[ic].sum())
    print(f"iterations: median {np.median(it[ic])}, p90 {np.percentile(it[ic], 90):.0f}, p99 {np.percentile(it[ic], 99):.0f}, max {it[ic].max()}", flush=True)
    def residual_for(mask):
        r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
        if len(sel):
            det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep[sel]), det), 1)
        return r
    caps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, None]
    series = [("cap", "cap alone", np.ones(len(size), bool)), ("cap_m10", "cap with margin > 1.0", margin > 1.0), ("cap_m05", "cap with margin > 0.5", margin > 0.5)]
    rules, jobs, stats = [], [], []
    for sid, sname, extra in series:
        for K in caps:
            m = interior & extra & ((it <= K) if K is not None else True); rules.append((sid, sname, K, m))
            r = residual_for(m); mm = m & committed
            stats.append(dict(committed_defects=int(size[mm].sum()), components=int(mm.sum()), residual_events=int(r.sum()), closed_share=float(size[mm].sum() / tot)))
            jobs.append((len(jobs), np.packbits(r, axis=1, bitorder="little")))
    print(f"{len(jobs)} rules over {N:,} shots", flush=True)
    with ProcessPoolExecutor(max_workers=a.processes, mp_context=multiprocessing.get_context("fork"), initializer=_init, initargs=(str(dem),)) as pool:
        preds = dict(pool.map(_decode, jobs, chunksize=1))
    rows = []
    for i, ((sid, sname, K, _), st) in enumerate(zip(rules, stats)):
        tfail = np.any(np.unpackbits(preds[i], axis=1, count=O, bitorder="little").astype(bool) != obs, axis=1)
        b = int((~gfail & tfail).sum()); cc = int((gfail & ~tfail).sum()); rd, lo, hi = wald(b, cc, N)
        rows.append(dict(series=sid, series_name=sname, cap=K, cycles_est=None if K is None else 3 * K, failures=int(tfail.sum()), regressions=b, recoveries=cc, risk_difference_pp=100 * rd, ci_pp=[100 * lo, 100 * hi],
                         coverage_pct=100 * st["committed_defects"] / lane_owned, residual_events_per_shot=st["residual_events"] / N, closed_share_pct=100 * st["closed_share"], **{k: st[k] for k in ("committed_defects", "components")}))
        print(f"{sname:24s} K={'∞' if K is None else K:>3}  cov {rows[-1]['coverage_pct']:5.1f}%  rd {100*rd:+.2f} pp  residual/shot {rows[-1]['residual_events_per_shot']:6.1f}", flush=True)
    json.dump(dict(cell=c, shots=N, quantum=q, w_max=16, cycles_per_iteration=3, global_failures=int(gfail.sum()), lane_owned_detector_events=lane_owned, rows=rows), open(a.cell / "cap_ladder.json", "w"), indent=1)
    print("saved", a.cell / "cap_ladder.json", flush=True)
