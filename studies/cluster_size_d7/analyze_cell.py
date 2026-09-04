"""Exact post-hoc analysis of one collected cell: commit-rule sweep (per shot and per observable),
marginal risk ladders, and leave-one-out culprit attribution per committed component."""
import argparse, json, math, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import gen, stim
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import GlobalMWPMDecoder

_G = {}
def _init(dem_text):
    _G["dec"] = GlobalMWPMDecoder().compile_decoder_for_dem(dem=stim.DetectorErrorModel(dem_text))
def _decode(job):
    name, packed = job
    return name, _G["dec"].decode_shots_bit_packed(bit_packed_detection_event_data=packed)
def _culprits(job):
    """For one regressed shot: restore each committed component alone; culprit if the shot becomes correct."""
    shot, base_unpacked, comp_ids_list, obs_row = job
    n = len(comp_ids_list)
    variants = np.repeat(base_unpacked[None, :], n, axis=0)
    for k, ids in enumerate(comp_ids_list):
        variants[k, ids] ^= 1
    pred = _G["dec"].decode_shots_bit_packed(bit_packed_detection_event_data=np.packbits(variants, axis=1, bitorder="little"))
    pred_bits = np.unpackbits(pred, axis=1, count=obs_row.shape[0], bitorder="little").astype(bool)
    return shot, ~np.any(pred_bits != obs_row[None, :], axis=1)

def wald(b, c, n):
    rd = (b - c) / n; se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    return rd, rd - 1.96 * se, rd + 1.96 * se

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", type=Path, required=True); ap.add_argument("--processes", type=int, default=16)
    a = ap.parse_args()
    prov = json.load(open(a.cell / "provenance.json")); z = np.load(a.cell / "cell.npz")
    p = prov["cell"]["p"]; D, O, N = prov["num_detectors"], prov["num_observables"], prov["shots"]
    circuit = yoked_magic_memory_circuit(patch_diameter=7, rounds=28, noise=gen.NoiseModel.si1000(p), style="cz", yokes=2, num_patches=6, remove_x_yoke=False)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    import hashlib; assert hashlib.sha256(str(dem).encode()).hexdigest() == prov["dem_sha256"]

    dets, obs = z["detectors"], z["observables"]; gpred, tpred = z["global_prediction"], z["treatment_prediction"]
    unpacked = np.unpackbits(dets, axis=1, count=D, bitorder="little")
    obs_bits = np.unpackbits(obs, axis=1, count=O, bitorder="little").astype(bool)
    gbits = np.unpackbits(gpred, axis=1, count=O, bitorder="little").astype(bool)
    lane_owned = int(unpacked.sum())  # yoke events are a negligible share; matches L to within yoke counts
    cs, size, bdry, port, committed = z["component_shot"], z["component_size"], z["component_boundary"], z["component_port"], z["component_committed"]
    margin, diam, ids_flat, ids_off = z["component_margin"], z["component_diameter"], z["component_defect_ids"], z["component_defect_offsets"]
    comp_ids = [ids_flat[ids_off[i]:ids_off[i + 1]] for i in range(len(size))]
    assert not np.any(committed & port)

    def residual_for(mask):
        r = unpacked.copy(); sel = np.flatnonzero(mask & committed)
        if len(sel):
            np.bitwise_xor.at(r, (np.repeat(cs[sel], [len(comp_ids[i]) for i in sel]), np.concatenate([comp_ids[i] for i in sel])), 1)
        return r
    # validation: all committed reproduces the stored residual exactly
    assert np.array_equal(np.packbits(residual_for(np.ones(len(size), bool)), axis=1, bitorder="little"), z["residual"])

    rules = [("nothing committed", np.zeros(len(size), bool)), ("all committed", np.ones(len(size), bool))]
    for k in (1, 2, 3, 4, 6, 8):
        rules.append((f"size<={k}", size <= k)); rules.append((f"size<={k}, interior", (size <= k) & ~bdry))
    rules.append(("interior, any size", ~bdry))
    for tau in (0.25, 0.5, 1.0, 1.5):
        rules.append((f"margin>{tau}", margin > tau)); rules.append((f"size<=2, interior, margin>{tau}", (size <= 2) & ~bdry & (margin > tau)))
    rules.append(("size<=2, interior, diameter<=1", (size <= 2) & ~bdry & (diam <= 1)))
    jobs, stats = [], {}
    for name, mask in rules:
        r = residual_for(mask); m = mask & committed
        stats[name] = dict(components=int(m.sum()), committed_defects=int(size[m].sum()), residual_events=int(r.sum()))
        jobs.append((name, np.packbits(r, axis=1, bitorder="little")))
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(a.processes, 8), mp_context=ctx, initializer=_init, initargs=(str(dem),)) as pool:
        preds = dict(pool.map(_decode, jobs))
    assert np.array_equal(preds["all committed"], tpred) and np.array_equal(preds["nothing committed"], gpred)

    gfail = np.any(gbits != obs_bits, axis=1); gfail_obs = gbits != obs_bits
    rows = []
    for name, _ in rules:
        tb = np.unpackbits(preds[name], axis=1, count=O, bitorder="little").astype(bool)
        tfail = np.any(tb != obs_bits, axis=1); tfail_obs = tb != obs_bits
        b = int((~gfail & tfail).sum()); c = int((gfail & ~tfail).sum()); rd, lo, hi = wald(b, c, N)
        bo = int((~gfail_obs & tfail_obs).sum()); co = int((gfail_obs & ~tfail_obs).sum()); rdo, loo, hio = wald(bo, co, N * O)
        st = stats[name]
        rows.append(dict(rule=name, failures=int(tfail.sum()), regressions=b, recoveries=c, risk_difference=rd, ci=[lo, hi],
                         observable_failures=int(tfail_obs.sum()), observable_regressions=bo, observable_recoveries=co,
                         observable_risk_difference=rdo, observable_ci=[loo, hio], coverage=st["committed_defects"] / lane_owned, **st))
    by = {r["rule"]: r for r in rows}
    def ladder(names):
        out, prev = [], by["nothing committed"]
        for n in names:
            r = by[n]; dd = r["committed_defects"] - prev["committed_defects"]
            dn = (r["regressions"] - r["recoveries"]) - (prev["regressions"] - prev["recoveries"])
            dno = (r["observable_regressions"] - r["observable_recoveries"]) - (prev["observable_regressions"] - prev["observable_recoveries"])
            out.append(dict(step=f"{prev['rule']} -> {n}", added_defects=int(dd), net_regressions=int(dn), per_1000=1000 * dn / dd if dd else None,
                            observable_net_regressions=int(dno), observable_per_1000=1000 * dno / dd if dd else None)); prev = r
        return out
    ladders = dict(interior_by_size=ladder(["size<=2, interior", "size<=4, interior", "size<=6, interior", "size<=8, interior", "interior, any size"]),
                   add_wall_clusters=ladder(["interior, any size", "all committed"]), wall_singletons=ladder(["size<=1"]),
                   margin=ladder(["margin>1.5", "margin>1.0", "margin>0.5", "margin>0.25", "all committed"]))

    # culprit attribution on regressed shots under full commit
    tfail_all = np.any(np.unpackbits(tpred, axis=1, count=O, bitorder="little").astype(bool) != obs_bits, axis=1)
    regressed = np.flatnonzero(~gfail & tfail_all)
    comps_by_shot = {}
    for i in np.flatnonzero(committed): comps_by_shot.setdefault(int(cs[i]), []).append(i)
    base_res = residual_for(np.ones(len(size), bool))
    cjobs = [(int(s), base_res[s], [comp_ids[i] for i in comps_by_shot.get(int(s), [])], obs_bits[s]) for s in regressed if comps_by_shot.get(int(s))]
    culprit = np.zeros(len(size), bool); in_regressed = np.zeros(len(size), bool); shots_without_single_culprit = 0
    with ProcessPoolExecutor(max_workers=min(a.processes, 16), mp_context=ctx, initializer=_init, initargs=(str(dem),)) as pool:
        for s, fixed in pool.map(_culprits, cjobs, chunksize=4):
            idx = comps_by_shot[s]; in_regressed[idx] = True; culprit[idx] = fixed
            if not fixed.any(): shots_without_single_culprit += 1
    def rate(mask):
        n = int((mask & in_regressed).sum()); k = int((mask & culprit).sum())
        return dict(components=n, culprits=k, rate=(k / n) if n else None)
    bands = [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 10**9)]
    attribution = dict(regressed_shots=int(len(regressed)), attributed_shots=len(cjobs), shots_without_single_culprit=shots_without_single_culprit,
                       overall=rate(np.ones(len(size), bool)),
                       by_size=[dict(band=f"{lo}-{hi if hi < 10**9 else '+'}", **rate((size >= lo) & (size <= hi))) for lo, hi in bands],
                       by_size_interior=[dict(band=f"{lo}-{hi if hi < 10**9 else '+'}", **rate((size >= lo) & (size <= hi) & ~bdry)) for lo, hi in bands],
                       by_size_wall=[dict(band=f"{lo}-{hi if hi < 10**9 else '+'}", **rate((size >= lo) & (size <= hi) & bdry)) for lo, hi in bands],
                       by_wall=[dict(wall=bool(w), **rate(bdry == w)) for w in (False, True)],
                       by_margin=[dict(band=f"{lo}-{hi}", **rate((margin > lo) & (margin <= hi))) for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 1e9))],
                       by_diameter=[dict(band=f"{lo}-{hi}", **rate((diam >= lo) & (diam <= hi))) for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 6), (7, 10**6))])
    analysis = dict(cell=prov["cell"], shots=N, seed=prov["stim_seed"], lane_owned_detector_events=lane_owned,
                    global_failures=int(gfail.sum()), global_observable_failures=int(gfail_obs.sum()), rules=rows, ladders=ladders, attribution=attribution)
    json.dump(analysis, open(a.cell / "analysis.json", "w"), indent=2)
    lines = [f"# d=7 p={p} cell, {N} shots (non-claim-bearing)", "", f"Global MWPM failures {int(gfail.sum())}/{N} per shot, {int(gfail_obs.sum())}/{N*O} per observable.", "",
             "| rule | fail | regr | recov | RD pp | obs RD pp | coverage | resid/shot |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['rule']} | {r['failures']} | {r['regressions']} | {r['recoveries']} | {100*r['risk_difference']:.2f} [{100*r['ci'][0]:.2f},{100*r['ci'][1]:.2f}] | {100*r['observable_risk_difference']:.2f} | {100*r['coverage']:.1f}% | {r['residual_events']/N:.0f} |")
    lines += ["", "## Culprit attribution (share of committed components in regressed shots whose restoration alone fixes the shot)", "",
              f"regressed shots {attribution['regressed_shots']}, without a single-component culprit {shots_without_single_culprit}", "",
              "| size band | all | interior | wall |", "|---|---:|---:|---:|"]
    for s_all, s_int, s_wall in zip(attribution["by_size"], attribution["by_size_interior"], attribution["by_size_wall"]):
        f = lambda r: f"{100*r['rate']:.2f}% (n={r['components']})" if r["rate"] is not None else "-"
        lines.append(f"| {s_all['band']} | {f(s_all)} | {f(s_int)} | {f(s_wall)} |")
    lines += ["", "| margin band | rate |", "|---|---:|"] + [f"| {r['band']} | {100*r['rate']:.2f}% (n={r['components']})" if r["rate"] is not None else f"| {r['band']} | -" for r in attribution["by_margin"]]
    (a.cell / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4] + lines[6:12]), flush=True); print("saved", a.cell / "analysis.json")
