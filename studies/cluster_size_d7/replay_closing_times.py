"""Replay the union-find pass over the stored syndromes and record each cluster's closing time in weight units.

Deterministic: same circuit, DEM, policy and shots as collect_cell.py, so components come out in the same order;
sizes and margins are checked against cell.npz before anything is written.  Output: closing_time.npy (float64,
one entry per component, the engine's last_membership_event_time).
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
    times, sizes, margins = [], [], []
    for i in range(stop - start):
        corr = treat.plan_shot(unpacked[i], capture=CaptureMode.METRICS)
        for out in corr.lane_outcomes:
            for comp in out.completed_components:
                times.append(float(comp.last_membership_event_time)); sizes.append(comp.cluster_defect_count)
                m = comp.exact_margin; margins.append(float("inf") if m is None else float(m))
    return start, np.asarray(times, np.float64), np.asarray(sizes, np.int32), np.asarray(margins, np.float64)
if __name__ == "__main__":
    cell = Path(sys.argv[1]); processes = int(sys.argv[2]) if len(sys.argv) > 2 else 32; assert 1 <= processes <= 32
    t0 = time.perf_counter(); prov = json.load(open(cell / "provenance.json")); c = prov["cell"]; z = np.load(cell / "cell.npz"); dets = z["detectors"]; N = dets.shape[0]
    circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=c["rounds"], noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    treat = PatchUFTreatmentDecoder(policy=PATCH_UF_V1_POLICY).compile_decoder_for_dem(dem=dem)
    assert treat.projection.fingerprint == prov["projection_fingerprint"], "projection differs from the collection"
    _STATE.update(treat=treat, dets=dets, D=dem.num_detectors)
    edges = np.linspace(0, N, processes + 1).astype(int); tasks = list(zip(edges[:-1], edges[1:]))
    with ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context("fork")) as pool:
        parts = sorted(pool.map(_worker, tasks), key=lambda r: r[0])
    times = np.concatenate([p[1] for p in parts]); sizes = np.concatenate([p[2] for p in parts]); margins = np.concatenate([p[3] for p in parts])
    assert len(times) == len(z["component_size"]), (len(times), len(z["component_size"]))
    assert np.array_equal(sizes, z["component_size"]), "component order differs: sizes do not match"
    assert np.allclose(margins, z["component_margin"], equal_nan=True), "component order differs: margins do not match"
    np.save(cell / "closing_time.npy", times)
    print(f"replayed {N:,} shots, {len(times):,} components in {time.perf_counter()-t0:.0f} s; closing time min {times.min():.3f} median {np.median(times):.3f} p99 {np.percentile(times, 99):.3f} max {times.max():.3f} weight units", flush=True)
