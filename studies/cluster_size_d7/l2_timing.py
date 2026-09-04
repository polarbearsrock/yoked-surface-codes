"""Per-shot Global MWPM (sparse blossom) decode time on each commit rule's residual, as the L2 latency proxy.

Each shot is decoded on its own (batch of one) so the per-shot time is observable; three repeats, the
per-shot minimum kept, to suppress scheduler noise.  Mean, p99, and max over shots per rule.
"""
import argparse, json, multiprocessing, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import gen, stim
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import GlobalMWPMDecoder
_G = {}
def _init(dem_text, packed_by_rule):
    _G["dec"] = GlobalMWPMDecoder().compile_decoder_for_dem(dem=stim.DetectorErrorModel(dem_text)); _G["packed"] = packed_by_rule
    warm = packed_by_rule[0][:64]
    for _ in range(3): _G["dec"].decode_shots_bit_packed(bit_packed_detection_event_data=warm)
def _time_range(job):
    rule_index, start, stop, reps = job
    dec = _G["dec"]; packed = _G["packed"][rule_index]; n = stop - start
    best = np.full(n, np.inf); pc = time.perf_counter_ns
    for _ in range(reps):
        for i in range(n):
            row = packed[start + i : start + i + 1]
            t0 = pc(); dec.decode_shots_bit_packed(bit_packed_detection_event_data=row); t1 = pc()
            if t1 - t0 < best[i]: best[i] = t1 - t0
    return rule_index, start, best
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", type=Path, required=True); ap.add_argument("--processes", type=int, default=16); ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
    prov = json.load(open(a.cell / "provenance.json")); z = np.load(a.cell / "cell.npz"); p = prov["cell"]["p"]; D, N = prov["num_detectors"], prov["shots"]
    c = prov["cell"]; circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=c["rounds"], noise=gen.NoiseModel.si1000(p), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little")
    cs, size, bdry, committed, margin, diam = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"], z["component_diameter"]
    ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; rep_counts = np.diff(ids_off); interior = ~bdry
    def residual_for(mask):
        r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
        if len(sel):
            det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep_counts[sel]), det), 1)
        return r
    rules = [dict(family="none", value=None, mask=np.zeros(len(size), bool))]
    for k in [2, 4, 6, 8, 10, 12, None]: rules.append(dict(family="size", value=k, mask=interior & ((size <= k) if k is not None else True)))
    for tau in [round(0.1 * i, 1) for i in range(0, 19)]: rules.append(dict(family="margin", value=tau, mask=interior & ((margin > tau) if tau > 0 else True)))
    for h in list(range(1, 11)) + [None]: rules.append(dict(family="diameter", value=h, mask=interior & ((diam <= h) if h is not None else True)))
    rules.append(dict(family="all", value=None, mask=np.ones(len(size), bool)))
    packed_by_rule, events = [], []
    for r in rules:
        res = residual_for(r["mask"]); events.append(res.sum(axis=1).astype(np.int32)); packed_by_rule.append(np.packbits(res, axis=1, bitorder="little"))
    print(f"{len(rules)} rules; timing {N} shots x {a.reps} reps each with {a.processes} workers", flush=True)
    chunk = 2500; jobs = [(ri, s, min(N, s + chunk), a.reps) for ri in range(len(rules)) for s in range(0, N, chunk)]
    times = np.zeros((len(rules), N), dtype=np.float64)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.processes, mp_context=multiprocessing.get_context("fork"), initializer=_init, initargs=(str(dem), packed_by_rule)) as pool:
        for ri, start, best in pool.map(_time_range, jobs, chunksize=1):
            times[ri, start:start + len(best)] = best
    print(f"timed in {time.perf_counter() - t0:.0f}s", flush=True)
    base = times[0]
    out_rows = []
    for ri, r in enumerate(rules):
        t = times[ri] / 1e6  # ms
        out_rows.append(dict(family=r["family"], value=r["value"], mean_ms=float(t.mean()), p50_ms=float(np.percentile(t, 50)), p99_ms=float(np.percentile(t, 99)), max_ms=float(t.max()),
                             mean_events=float(events[ri].mean()), max_events=int(events[ri].max()),
                             rel_mean=float(t.mean() / (base.mean() / 1e6)), rel_p99=float(np.percentile(t, 99) / (np.percentile(base, 99) / 1e6)), rel_max=float(t.max() / (base.max() / 1e6))))
    np.savez_compressed(a.cell / "l2_timing.npz", times_ns=times.astype(np.float32), events=np.stack(events), rules=np.array([f"{r['family']}:{r['value']}" for r in rules]))
    json.dump(dict(cell=prov["cell"], shots=N, reps=a.reps, processes=a.processes, host_note="software sparse blossom (PyMatching) per-shot decode of the residual, batch of one, min over reps", rows=out_rows), open(a.cell / "l2_timing.json", "w"), indent=1)
    print(f"{'rule':16s} {'mean ms':>8s} {'p99 ms':>8s} {'max ms':>8s} {'rel mean':>9s} {'rel p99':>8s} {'rel max':>8s} {'mean ev':>8s}")
    for r in out_rows: print(f"{r['family']+':'+str(r['value']):16s} {r['mean_ms']:8.3f} {r['p99_ms']:8.3f} {r['max_ms']:8.3f} {r['rel_mean']:9.3f} {r['rel_p99']:8.3f} {r['rel_max']:8.3f} {r['mean_events']:8.0f}")
    print("saved", a.cell / "l2_timing.json", flush=True)
