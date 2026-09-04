"""Interior-only commit ladders on one collected cell: size caps (no margin) and margin thresholds (no size cap)."""
import argparse, json, math, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import gen, stim
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import GlobalMWPMDecoder
_G = {}
def _init(dem_text): _G["dec"] = GlobalMWPMDecoder().compile_decoder_for_dem(dem=stim.DetectorErrorModel(dem_text))
def _decode(job):
    name, packed = job; return name, _G["dec"].decode_shots_bit_packed(bit_packed_detection_event_data=packed)
def wald(b, c, n):
    rd = (b - c) / n; se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n; return rd, rd - 1.96 * se, rd + 1.96 * se
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", type=Path, required=True); ap.add_argument("--processes", type=int, default=16); a = ap.parse_args()
    prov = json.load(open(a.cell / "provenance.json")); z = np.load(a.cell / "cell.npz"); p = prov["cell"]["p"]; D, O, N = prov["num_detectors"], prov["num_observables"], prov["shots"]
    c = prov["cell"]; circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=c["rounds"], noise=gen.NoiseModel.si1000(p), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    unpacked = np.unpackbits(z["detectors"], axis=1, count=D, bitorder="little"); obs_bits = np.unpackbits(z["observables"], axis=1, count=O, bitorder="little").astype(bool)
    gfail_obs = np.unpackbits(z["global_prediction"], axis=1, count=O, bitorder="little").astype(bool) != obs_bits; gfail = np.any(gfail_obs, axis=1)
    cs, size, bdry, committed, margin, diam = z["component_shot"], z["component_size"], z["component_boundary"], z["component_committed"], z["component_margin"], z["component_diameter"]
    ids_flat, ids_off = z["component_defect_ids"], z["component_defect_offsets"]; lane_owned = int(unpacked.sum()); rep_counts = np.diff(ids_off)
    def residual_for(mask):
        r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
        if len(sel):
            det = np.concatenate([ids_flat[ids_off[i]:ids_off[i+1]] for i in sel]); np.bitwise_xor.at(r, (np.repeat(cs[sel], rep_counts[sel]), det), 1)
        return r
    interior = ~bdry
    rules = [(dict(family="none", value=None), np.zeros(len(size), bool))]
    for k in list(range(2, 17, 2)) + [20, 24, None]:
        rules.append((dict(family="size", value=k), interior & ((size <= k) if k is not None else True)))
    for tau in [round(0.05 * i, 2) for i in range(0, 37)]:
        rules.append((dict(family="margin", value=tau), interior & ((margin > tau) if tau > 0 else True)))
    for h in list(range(1, 31)) + [None]:
        rules.append((dict(family="diameter", value=h), interior & ((diam <= h) if h is not None else True)))
    jobs, stats = [], []
    for spec, m in rules:
        r = residual_for(m); mm = m & committed
        stats.append(dict(committed_defects=int(size[mm].sum()), components=int(mm.sum()), residual_events=int(r.sum())))
        jobs.append((len(jobs), np.packbits(r, axis=1, bitorder="little")))
    print(f"{len(jobs)} rules over {N} shots", flush=True)
    with ProcessPoolExecutor(max_workers=a.processes, mp_context=multiprocessing.get_context("fork"), initializer=_init, initargs=(str(dem),)) as pool:
        preds = dict(pool.map(_decode, jobs, chunksize=1))
    rows = []
    for i, ((spec, _), st) in enumerate(zip(rules, stats)):
        tb = np.unpackbits(preds[i], axis=1, count=O, bitorder="little").astype(bool); tfail_obs = tb != obs_bits; tfail = np.any(tfail_obs, axis=1)
        b = int((~gfail & tfail).sum()); c = int((gfail & ~tfail).sum()); rd, lo, hi = wald(b, c, N)
        bo = int((~gfail_obs & tfail_obs).sum()); co = int((gfail_obs & ~tfail_obs).sum()); rdo, loo, hio = wald(bo, co, N * O)
        rows.append(dict(**spec, failures=int(tfail.sum()), regressions=b, recoveries=c, risk_difference_pp=100 * rd, ci_pp=[100 * lo, 100 * hi],
                         observable_failures=int(tfail_obs.sum()), observable_regressions=bo, observable_recoveries=co, observable_risk_difference_pp=100 * rdo, observable_ci_pp=[100 * loo, 100 * hio],
                         coverage_pct=100 * st["committed_defects"] / lane_owned, residual_events_per_shot=st["residual_events"] / N, **st))
    json.dump(dict(cell=prov["cell"], shots=N, num_observables=O, global_failures=int(gfail.sum()), global_observable_failures=int(gfail_obs.sum()), lane_owned_detector_events=lane_owned,
                   interior_component_share=float(interior[committed].mean()),
                   interior_diameter_histogram={int(k): int(v) for k, v in zip(*np.unique(diam[committed & interior], return_counts=True))},
                   interior_size_histogram={int(k): int(v) for k, v in zip(*np.unique(size[committed & interior], return_counts=True))}, rows=rows), open(a.cell / "frontier_interior.json", "w"), indent=1)
    print("saved", a.cell / "frontier_interior.json", flush=True)
