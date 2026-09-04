"""Replay the union-find pass and record every cluster's event times (weight units) for the truncated Helios proxy.

Output: event_times.npz with flat float64 `times`, int64 `offsets` (one slice per component, same order as cell.npz;
sizes are checked), and `lane` per component.
"""
import json, multiprocessing, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import CaptureMode, PATCH_UF_V1_POLICY, PatchUFTreatmentDecoder
_STATE = {}
def _worker(task):
    start, stop = task; treat, dets, D = _STATE["treat"], _STATE["dets"], _STATE["D"]
    unpacked = np.unpackbits(dets[start:stop], axis=1, count=D, bitorder="little")
    flat, counts, sizes = [], [], []
    for i in range(stop - start):
        corr = treat.plan_shot(unpacked[i], capture=CaptureMode.METRICS)
        for out in corr.lane_outcomes:
            for comp in out.completed_components:
                ts = [float(t) for t in comp.event_batch_times]; flat.extend(ts); counts.append(len(ts)); sizes.append(comp.cluster_defect_count)
    return start, np.asarray(flat, np.float64), np.asarray(counts, np.int64), np.asarray(sizes, np.int32)
if __name__ == "__main__":
    cell = Path(sys.argv[1]); processes = int(sys.argv[2]) if len(sys.argv) > 2 else 32; assert 1 <= processes <= 32
    t0 = time.perf_counter(); prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; z = np.load(cell / "cell.npz"); dets = z["detectors"]; N = dets.shape[0]
    circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=c["rounds"], noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    treat = PatchUFTreatmentDecoder(policy=PATCH_UF_V1_POLICY).compile_decoder_for_dem(dem=dem); assert treat.projection.fingerprint == prov["projection_fingerprint"]
    _STATE.update(treat=treat, dets=dets, D=dem.num_detectors)
    edges = np.linspace(0, N, processes + 1).astype(int); tasks = list(zip(edges[:-1], edges[1:]))
    with ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context("fork")) as pool:
        parts = sorted(pool.map(_worker, tasks), key=lambda r: r[0])
    flat = np.concatenate([p[1] for p in parts]); counts = np.concatenate([p[2] for p in parts]); sizes = np.concatenate([p[3] for p in parts])
    assert np.array_equal(sizes, z["component_size"]), "component order differs"
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez(cell / "event_times.npz", times=flat, offsets=offsets)
    print(f"replayed {N:,} shots, {len(sizes):,} components, {len(flat):,} events in {time.perf_counter()-t0:.0f} s; events per component mean {counts.mean():.2f} max {counts.max()}", flush=True)
