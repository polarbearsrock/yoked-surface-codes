"""Lightweight, non-claim-bearing collection of one d=7 cell for the cluster-size study.

Samples `shots` shots with a recorded seed, decodes with Global MWPM and with the
port-wall Patch-UF treatment in telemetry mode across 32 forked workers, and keeps
only compact per-component records plus packed shots, predictions, and provenance.
"""
import argparse, hashlib, json, multiprocessing, os, platform, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import gen, stim
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import (CaptureMode, GlobalMWPMDecoder, PATCH_UF_TREATMENT_DECODER_NAME,
                                              PATCH_UF_V1_POLICY, PatchUFTreatmentDecoder)

_STATE = {}

def _worker(task):
    start, stop = task
    treat, glob, dets, D = _STATE["treat"], _STATE["glob"], _STATE["dets"], _STATE["D"]
    proj = treat.projection
    unpacked = np.unpackbits(dets[start:stop], axis=1, count=D, bitorder="little")
    residual = np.empty_like(dets[start:stop])
    rec = {k: [] for k in ("shot", "lane", "size", "absorbed", "boundary", "port", "committed", "margin", "diameter", "batches")}
    ids_flat, ids_off = [], [0]
    for i in range(stop - start):
        corr = treat.plan_shot(unpacked[i], capture=CaptureMode.METRICS)
        packed_residual, _ = treat.apply_shot_correction(dets[start + i], corr)
        residual[i] = packed_residual
        durable = {(l, c) for p in corr.patch_outcomes if p.status == "durable" for l, c in p.durable_component_refs}
        for lane_id, out in enumerate(corr.lane_outcomes):
            lane = proj.lanes[lane_id]
            for comp in out.completed_components:
                rec["shot"].append(start + i); rec["lane"].append(lane_id)
                rec["size"].append(comp.cluster_defect_count); rec["absorbed"].append(comp.absorbed_vertex_count)
                rec["boundary"].append(comp.boundary_reached); rec["port"].append(comp.port_tainted)
                rec["committed"].append((lane_id, comp.component_index) in durable)
                m = comp.exact_margin; rec["margin"].append(float("inf") if m is None else float(m))
                rec["diameter"].append(comp.forest_diameter_hops); rec["batches"].append(comp.simultaneous_event_batch_count)
                ids_flat.extend(lane.global_detector_ids[v] for v in comp.original_defects); ids_off.append(len(ids_flat))
    tpred = glob.decode_shots_bit_packed(bit_packed_detection_event_data=residual)
    dtypes = dict(shot=np.int32, lane=np.int8, size=np.int32, absorbed=np.int32, boundary=bool, port=bool, committed=bool, margin=np.float64, diameter=np.int32, batches=np.int32)
    rec = {k: np.asarray(v, dtype=dtypes[k]) for k, v in rec.items()}
    return start, stop, residual, tpred, rec, np.asarray(ids_flat, dtype=np.int32), np.asarray(ids_off, dtype=np.int64)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--p", type=float, required=True); ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--d", type=int, default=7); ap.add_argument("--rounds", type=int, default=None); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--processes", type=int, default=32)
    a = ap.parse_args(); assert 1 <= a.processes <= 32
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    rounds = a.rounds or 4 * a.d
    circuit = yoked_magic_memory_circuit(patch_diameter=a.d, rounds=rounds, noise=gen.NoiseModel.si1000(a.p), style="cz", yokes=2, num_patches=6, remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    treat = PatchUFTreatmentDecoder(policy=PATCH_UF_V1_POLICY).compile_decoder_for_dem(dem=dem)
    glob = GlobalMWPMDecoder().compile_decoder_for_dem(dem=dem)
    D, O = dem.num_detectors, dem.num_observables
    dets, obs = circuit.compile_detector_sampler(seed=a.seed).sample(shots=a.shots, separate_observables=True, bit_packed=True)
    gpred = glob.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    _STATE.update(treat=treat, glob=glob, dets=dets, D=D)
    edges = np.linspace(0, a.shots, a.processes + 1).astype(int); tasks = list(zip(edges[:-1], edges[1:]))
    residual = np.empty_like(dets); tpred = np.empty_like(gpred)
    recs = {k: [] for k in ("shot", "lane", "size", "absorbed", "boundary", "port", "committed", "margin", "diameter", "batches")}
    ids_parts, off_parts, off_base = [], [], 0
    with ProcessPoolExecutor(max_workers=a.processes, mp_context=multiprocessing.get_context("fork")) as pool:
        for start, stop, res, tp, rec, ids, off in sorted(pool.map(_worker, tasks), key=lambda r: r[0]):
            residual[start:stop] = res; tpred[start:stop] = tp
            for k in recs: recs[k].append(rec[k])
            ids_parts.append(ids); off_parts.append(off[1:] + off_base); off_base += len(ids)
    ids_flat = np.concatenate(ids_parts); ids_off = np.concatenate([[0]] + off_parts)
    recs = {k: np.concatenate(v) for k, v in recs.items()}
    arrays = dict(detectors=dets, observables=obs, residual=residual, global_prediction=gpred, treatment_prediction=tpred,
                  component_shot=recs["shot"], component_lane=recs["lane"], component_size=recs["size"], component_absorbed=recs["absorbed"],
                  component_boundary=recs["boundary"], component_port=recs["port"], component_committed=recs["committed"],
                  component_margin=recs["margin"], component_diameter=recs["diameter"], component_batches=recs["batches"],
                  component_defect_ids=ids_flat, component_defect_offsets=ids_off)
    np.savez_compressed(a.out / "cell.npz", **arrays)
    gfail = np.any(gpred != obs, axis=1); tfail = np.any(tpred != obs, axis=1)
    prov = dict(claim_status="non-claim-bearing-lightweight-study", cell=dict(d=a.d, rounds=rounds, patches=6, yokes=2, style="cz", noise="si1000", p=a.p, remove_x_yoke=False),
                shots=a.shots, stim_seed=a.seed, processes=a.processes, treatment=PATCH_UF_TREATMENT_DECODER_NAME, global_arm="global-mwpm-u0-joint-y2",
                circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(), dem_sha256=hashlib.sha256(str(dem).encode()).hexdigest(),
                num_detectors=D, num_observables=O, projection_fingerprint=treat.projection.fingerprint,
                repository_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                worktree_dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
                python=platform.python_version(), stim=stim.__version__, wall_seconds=round(time.perf_counter() - t0, 1),
                summary=dict(global_failures=int(gfail.sum()), treatment_failures=int(tfail.sum()),
                             paired=dict(a=int((~gfail & ~tfail).sum()), b=int((~gfail & tfail).sum()), c=int((gfail & ~tfail).sum()), d=int((gfail & tfail).sum())),
                             components=int(len(recs["size"])), committed_components=int(recs["committed"].sum())))
    json.dump(prov, open(a.out / "provenance.json", "w"), indent=2)
    print(json.dumps({k: prov[k] for k in ("cell", "shots", "wall_seconds", "summary")}), flush=True)
