"""Resumable seven-decoder distance sweep, limited to 32 physical cores."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parents[1]
sys.path.insert(0, str(SNAPSHOT/'experiments/bp_uf_single_patch_d7'))
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import common
from common import evidence, np, sha256, write_json
import check as legacy_checks
from add_bp10 import statistics
import add_correlated_mwpm as mwpm
import pymatching
import scipy
import stim
from yoked.decoders import DecodingGraph, UnionFindDecoder

os.environ.update(OMP_NUM_THREADS='32', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE')
os.environ.pop('OMP_PLACES', None)
os.environ.pop('OMP_PROC_BIND', None)
NATIVE_VARIANTS = [*common.VARIANTS, 'bp10_uf']
VARIANTS = [*NATIVE_VARIANTS, 'correlated_mwpm']
FIELDS = [*common.FIELDS, 'bp_initialization_seconds', 'bp_rounds_seconds', 'projection_seconds']
DETERMINISTIC = [*range(9), 11]
DISTANCES = [9, 11, 13]
SAMPLE_SIZES = dict(pilot=10000, confirmation=100000)


def seed(distance, phase):
    return 202609290000 + 100*distance + phase


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint(path, **arrays):
    # Checkpoints are always in scratch, including this temporary file.
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


class Native(evidence.Native):
    def __init__(self, library, graph, rules, model):
        super().__init__(library, graph, rules, model)
        self.lib.distance_sweep_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.distance_sweep_decode.restype = ctypes.c_int

    def decode(self, packed, threads=1, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if not 1 <= threads <= 32 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid thread count or detector packing')
        out = np.zeros((len(packed), len(NATIVE_VARIANTS), len(FIELDS)), dtype=np.float64)
        flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint32) if audit else None
        team = ctypes.c_int()
        errors = self.lib.distance_sweep_decode(self.handle, len(packed), packed.shape[1], evidence.ptr(packed),
            evidence.ptr(out), threads, evidence.ptr(flags) if audit else None, ctypes.byref(team))
        if errors or team.value != threads or not np.isfinite(out).all():
            raise RuntimeError(f'Native errors={errors}; requested/actual OpenMP team={threads}/{team.value}')
        np.testing.assert_array_equal(out[:, :, 11], np.broadcast_to([0, 0, 1, 2, 5, 10], out.shape[:2]))
        np.testing.assert_allclose(out[:, 2:, 9], out[:, 2:, 12:15].sum(axis=2), atol=0, rtol=0)
        assert np.all(out[:, :, 0] == out[:, :, 0].astype(np.uint8)) and np.all(out[:, :, 0] < 4)
        return out, flags


def model(distance, library):
    circuit = common.yoked_magic_memory_circuit(patch_diameter=distance, rounds=4*distance,
        noise=common.gen.NoiseModel.si1000(0.003), style='cz', yokes=0, num_patches=1)
    assert circuit.num_observables == 2
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    graph = evidence.DecodingGraph.from_dem(dem)
    faults = evidence.FaultModel.from_dem(dem, graph)
    native = Native(library, graph, evidence.correlation_rules_from_dem(graph, dem), faults)
    return circuit, dem, native


def verify_reference(reference, library):
    circuit, dem, native = model(7, library)
    configuration = json.loads((reference/'configuration.json').read_text())
    assert hashlib.sha256(str(circuit).encode()).hexdigest() == configuration['circuit_sha256']
    assert hashlib.sha256(str(dem).encode()).hexdigest() == configuration['dem_sha256']
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    checks = {}
    try:
        for phase in SAMPLE_SIZES:
            sample = json.loads((reference/phase/'sample.json').read_text())
            assert sha256(sample['sample_path']) == sample['raw_npz_sha256']
            with np.load(sample['sample_path']) as raw, np.load(reference/phase/'results.npz') as old, \
                    np.load(reference/'bp10_uf'/f'{phase}_results.npz') as all_saved:
                rows = np.unique(np.linspace(0, len(raw['detectors'])-1, 32, dtype=int))
                packed = raw['detectors'][rows]
                out, _ = native.decode(packed, 32)
                np.testing.assert_array_equal(out[:, :5][:, :, DETERMINISTIC], old['fields'][rows][:, :, DETERMINISTIC])
                np.testing.assert_array_equal(out[:, 5][:, DETERMINISTIC], all_saved['bp10_fields'][rows][:, DETERMINISTIC])
                prediction = matching.decode_batch(packed, bit_packed_shots=True,
                    bit_packed_predictions=True, enable_correlations=True)[:, 0]
                index = all_saved['variants'].tolist().index('correlated_mwpm')
                np.testing.assert_array_equal(prediction, all_saved['predictions'][rows, index])
                checks[phase] = rows.tolist()
    finally:
        native.close()
    return dict(status='passed', reference=str(reference), row_ids=checks,
                native_deterministic_fields_match=True, all_seven_predictions_match=True)


def verify_distance(distance, circuit, dem, native, matching):
    packed, _ = circuit.compile_detector_sampler(seed=seed(distance, 3)).sample(
        shots=32, separate_observables=True, bit_packed=True)
    packed[0] = 0
    parallel, flags = native.decode(packed, 32, audit=True)
    serial, serial_flags = native.decode(packed, 1, audit=True)
    np.testing.assert_array_equal(parallel[:, :, DETERMINISTIC], serial[:, :, DETERMINISTIC])
    np.testing.assert_array_equal(flags, serial_flags)
    syndromes = np.unpackbits(packed, axis=1, count=circuit.num_detectors, bitorder='little')
    checks = []
    for row in range(4):
        syndrome = syndromes[row]
        old = legacy_checks.audit_case(native, dem, syndrome, parallel[row, :5], flags[row])
        posterior, weights = native.weights(syndrome, 10)
        expected_q, expected_w = evidence.python_bp(native.model, syndrome, 10)
        np.testing.assert_allclose(posterior, expected_q, atol=2e-10, rtol=2e-8)
        np.testing.assert_allclose(weights, expected_w, atol=2e-9, rtol=2e-8)
        graph = DecodingGraph(native.graph.num_detectors, native.graph.num_observables,
            [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(native.graph.edges)])
        expected = UnionFindDecoder(graph)._decode(syndrome)
        np.testing.assert_array_equal(np.flatnonzero(flags[row] & (1 << 5)), sorted(expected.selected_edges))
        assert int(parallel[row, 5, 0]) == expected.observable_mask
        checks.append(dict(row=row, first_five=old,
            bp10_posterior_max_difference=float(np.max(np.abs(posterior-expected_q)))))
        print(f'd={distance}: independent BP/UF validation row {row} passed.', flush=True)
    for syndrome, values, selected in zip(syndromes, parallel, flags):
        for k in range(len(NATIVE_VARIANTS)):
            residual, mask = syndrome.copy(), 0
            for edge in np.flatnonzero(selected & (1 << k)):
                u, v, _, label = native.graph.edges[edge]
                residual[u] ^= 1
                if v is not None:
                    residual[v] ^= 1
                mask ^= label
            assert not residual.any() and mask == int(values[k, 0])
    mwpm_prediction = matching.decode_batch(packed, bit_packed_shots=True,
        bit_packed_predictions=True, enable_correlations=True)[:, 0]
    audit = mwpm.verify_predictions(matching, packed, mwpm_prediction, circuit.num_detectors, seed(distance, 3))
    return dict(status='passed', distance=distance, verified_openmp_threads=[1, 32],
        serial_parallel_agreement_shots=32, native_physical_corrections_reconstructed=32*len(NATIVE_VARIANTS),
        independent_python_checks=checks, mwpm_validation=audit, actuals_passed_to_decoder=False)


def collect(phase, distance, circuit, native, matching, pool, directory, work, config_id, progress):
    shots = SAMPLE_SIZES[phase]
    phase_seed = seed(distance, 1 if phase == 'pilot' else 2)
    retained, scratch = directory/phase, work/phase
    retained.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    sample_path = scratch/'sample.npz'
    if sample_path.exists():
        with np.load(sample_path) as saved:
            packed, truth = saved['detectors'], saved['actual']
    else:
        packed, truth = circuit.compile_detector_sampler(seed=phase_seed).sample(
            shots=shots, separate_observables=True, bit_packed=True)
        checkpoint(sample_path, detectors=packed, actual=truth)
    assert packed.shape == (shots, (circuit.num_detectors+7)//8) and truth.shape == (shots, 1)
    actual = truth[:, 0] & 3
    sample = dict(shots=shots, seed=phase_seed, sample_path=str(sample_path), raw_npz_sha256=sha256(sample_path),
        packed_sample_sha256=mwpm.payload_hash(packed, truth), detectors=circuit.num_detectors, observables=2)
    if (retained/'sample.json').exists():
        assert json.loads((retained/'sample.json').read_text()) == sample
    write_json(retained/'sample.json', sample)
    phase_id = identity(dict(configuration=config_id, sample=sample))
    parts, wall = [], 0.
    errors = np.zeros(len(NATIVE_VARIANTS), dtype=np.int64)
    records = []
    for start in range(0, shots, 512):
        stop = min(start+512, shots)
        path = scratch/f'native_{start:06d}_{stop:06d}.npz'
        if path.exists():
            with np.load(path) as saved:
                assert str(saved['identity']) == phase_id
                out, elapsed = saved['fields'], float(saved['wall_seconds'])
        else:
            began = time.monotonic()
            out, _ = native.decode(packed[start:stop], 32)
            elapsed = time.monotonic()-began
            checkpoint(path, identity=phase_id, fields=out, wall_seconds=elapsed)
        assert out.shape == (stop-start, len(NATIVE_VARIANTS), len(FIELDS)) and np.isfinite(out).all()
        np.testing.assert_array_equal(out[:, :, 11], np.broadcast_to([0, 0, 1, 2, 5, 10], out.shape[:2]))
        errors += np.count_nonzero(out[:, :, 0] != actual[start:stop, None], axis=0)
        parts.append(out)
        wall += elapsed
        records.append(dict(kind='native', start=start, stop=stop, path=str(path), sha256=sha256(path)))
        progress('collecting_native', distance=distance, phase=phase, completed_shots=stop, target_shots=shots,
            phase_kernel_wall_seconds=wall, failures=dict(zip(NATIVE_VARIANTS, map(int, errors))),
            estimated_remaining_native_seconds=wall/stop*(shots-stop))
        if stop == shots or stop % 10240 == 0:
            print(f'd={distance} {phase}: {stop:,}/{shots:,} native shots; failures {errors.tolist()}', flush=True)
    fields = np.concatenate(parts)
    mapped = scratch/'detectors.npy'
    np.save(mapped, packed)
    predicted = np.full(shots, 255, dtype=np.uint8)
    completed = 0
    futures = {}
    began = time.monotonic()
    for start in range(0, shots, 512):
        stop = min(start+512, shots)
        path = scratch/f'mwpm_{start:06d}_{stop:06d}.npz'
        if path.exists():
            with np.load(path) as saved:
                assert str(saved['identity']) == phase_id
                predicted[start:stop] = saved['prediction']
            completed += stop-start
            records.append(dict(kind='mwpm', start=start, stop=stop, path=str(path), sha256=sha256(path)))
        else:
            futures[pool.submit(mwpm.decode_chunk, str(mapped), start, stop)] = (start, stop, path)
    worker_pids = set()
    for future in as_completed(futures):
        start, stop, path = futures[future]
        first, prediction, pid = future.result()
        assert first == start and prediction.shape == (stop-start,) and np.all(prediction < 4)
        predicted[start:stop] = prediction
        checkpoint(path, identity=phase_id, prediction=prediction)
        records.append(dict(kind='mwpm', start=start, stop=stop, path=str(path), sha256=sha256(path)))
        worker_pids.add(pid)
        completed += stop-start
        progress('collecting_mwpm', distance=distance, phase=phase, completed_shots=completed, target_shots=shots)
    matching_wall = time.monotonic()-began
    assert completed == shots and np.all(predicted < 4)
    audit = mwpm.verify_predictions(matching, packed, predicted, circuit.num_detectors,
                                   seed(distance, 5 if phase == 'pilot' else 6))
    predictions = np.column_stack([fields[:, :, 0].astype(np.uint8), predicted])
    summary = statistics(predictions, actual, VARIANTS)
    summary.pop('added_comparisons')
    summary.update(distance=distance, rounds=4*distance,
        rate_unit=f'one patch over {4*distance} rounds; either logical observable wrong',
        other_comparisons='exploratory; no multiplicity adjustment',
        collection_native_wall_seconds=wall, collection_mwpm_wall_seconds=matching_wall,
        physical_native_syndrome_checks=shots*len(NATIVE_VARIANTS), seed=phase_seed,
        identity=phase_id, mwpm_audit=audit, mwpm_worker_pids=sorted(worker_pids))
    np.savez_compressed(retained/'results.npz', predictions=predictions, actual=actual, fields=fields,
        variants=np.array(VARIANTS), native_variants=np.array(NATIVE_VARIANTS), field_names=np.array(FIELDS))
    summary['results_sha256'] = sha256(retained/'results.npz')
    write_json(retained/'summary.json', summary)
    write_json(retained/'checkpoints.json', sorted(records, key=lambda r: (r['kind'], r['start'])))
    print(f'd={distance} {phase}: all seven complete; MWPM failures {summary["metrics"]["correlated_mwpm"]["errors"]:,}.', flush=True)
    return packed, fields, predicted


def benchmark(distance, native, matching, pilot, directory, progress):
    packed, collected, matching_predictions = pilot
    rows = np.sort(np.random.default_rng(seed(distance, 4)).choice(len(packed), 1024, replace=False))
    native.decode(packed[rows[:16]], 1)
    parts = []
    for start in range(0, len(rows), 64):
        selected = rows[start:start+64]
        out, _ = native.decode(packed[selected], 1)
        np.testing.assert_array_equal(out[:, :, DETERMINISTIC], collected[selected][:, :, DETERMINISTIC])
        parts.append(out)
        progress('serial_benchmark', distance=distance, completed_shots=min(start+64, len(rows)), target_shots=len(rows))
    fields = np.concatenate(parts)
    np.savez_compressed(directory/'serial_timing.npz', rows=rows, fields=fields)
    metrics = {}
    for k, name in enumerate(NATIVE_VARIANTS):
        total = (fields[:, k, 9]+fields[:, k, 10])*1000
        q = np.quantile(total, [.5, .95, .99])
        metrics[name] = dict(mean_total_ms=float(total.mean()), p50_ms=float(q[0]),
            p95_ms=float(q[1]), p99_ms=float(q[2]),
            **{key: float(fields[:, k, column].mean()*1000) for key, column in
               [('mean_evidence_ms', 9), ('mean_uf_ms', 10), ('mean_bp_initialization_ms', 12),
                ('mean_bp_rounds_ms', 13), ('mean_projection_ms', 14)]})
    target = directory/'mwpm_timing'
    target.mkdir(exist_ok=True)
    matching_timing = mwpm.benchmark(matching, packed, matching_predictions, directory, target, native.graph.num_detectors)
    metrics['correlated_mwpm'] = matching_timing['metrics']['bound_native_decode']
    result = dict(shots=len(rows), metrics=metrics,
        correlated_mwpm_public_api=matching_timing['metrics']['public_decode_api'],
        native_scope='Serial BP initialization, cumulative BP iterations, per-budget weight projection, '
            'and UF growth/tree/peeling. Correlated UF charges both necessary passes plus reweighting. '
            'Excludes setup, Python, unpacking, physical audits and some destruction. '
            'Fixed arm order and shared trajectory affect cache state.',
        matching_scope=matching_timing['scope'])
    write_json(directory/'serial_timing.json', result)
    return result


def write_reports(output, reference):
    combined = ['# Fixed BP + UF: d=7, 9, 11, 13', '',
        'SI1000 p=0.003; one patch; 4d CZ rounds; ideal time boundaries; zero yokes. '
        'Weighted UF, correlated UF, fixed BP1/2/5/10 + UF, and correlated MWPM. '
        'Rates are per complete memory shot, either observable wrong. '
        'Distances 9, 11 and 13 use new paired samples with frozen parameters and 32 cores total.', '',
        '| Distance | Decoder | Confirmation failures / shots | Failure rate | Mean serial ms |',
        '|---:|---|---:|---:|---:|']
    for distance in [7, *DISTANCES]:
        directory = reference if distance == 7 else output/f'd{distance}'
        summary_path = directory/'bp10_uf/confirmation_summary.json' if distance == 7 else directory/'confirmation/summary.json'
        timing = {}
        if distance == 7:
            timing.update(json.loads((reference/'serial_timing.json').read_text())['metrics'])
            timing.update(json.loads((reference/'bp10_uf/serial_timing.json').read_text())['metrics'])
            timing['correlated_mwpm'] = json.loads((reference/'correlated_mwpm/serial_timing.json').read_text())['metrics']['bound_native_decode']
        elif (directory/'serial_timing.json').exists():
            timing = json.loads((directory/'serial_timing.json').read_text())['metrics']
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for name in VARIANTS:
                row = summary['metrics'][name]
                ms = f'{timing[name]["mean_total_ms"]:.3f}' if name in timing else 'pending'
                combined.append(f'| {distance} | {name} | {row["errors"]:,} / {row["shots"]:,} | '
                                f'{100*row["failure_rate"]:.3f}% | {ms} |')
        else:
            combined.append(f'| {distance} | All seven | pending | pending | pending |')
        if distance == 7 or not directory.exists():
            continue
        lines = [f'# Fixed BP + UF: d={distance}', '',
            f'SI1000 p=0.003; one patch; {4*distance} CZ rounds; zero yokes; ideal time boundaries. '
            'All variants decode identical shots. BP always runs its fixed budget and finishes with UF. '
            'Correlated MWPM enables correlations at import and decoding.', '']
        for phase in SAMPLE_SIZES:
            path = directory/phase/'summary.json'
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            lines += [f'## {phase.capitalize()}: {s["shots"]:,} shots', '',
                '| Decoder | Failures | Failure rate (95% Wilson interval) | X errors | Z errors |',
                '|---|---:|---:|---:|---:|']
            for name in VARIANTS:
                r = s['metrics'][name]
                lo, hi = r['wilson_95']
                lines.append(f'| {name} | {r["errors"]:,} | {100*r["failure_rate"]:.3f}% '
                    f'[{100*lo:.3f}, {100*hi:.3f}] | {r["x_observable_errors"]:,} | {r["z_observable_errors"]:,} |')
            lines += ['', '| Baseline | Target | Repairs | Regressions | Difference, pp (95% paired CI) | McNemar p |',
                '|---|---|---:|---:|---:|---:|']
            for baseline, targets in [('correlated_uf', ['bp1_uf', 'bp2_uf', 'bp5_uf', 'bp10_uf']),
                                      ('bp5_uf', ['bp10_uf']), ('correlated_mwpm', ['bp5_uf', 'bp10_uf'])]:
                for target in targets:
                    r = s['paired'][baseline][target]
                    lo, hi = r['paired_normal_95']
                    lines.append(f'| {baseline} | {target} | {r["repairs"]:,} | {r["regressions"]:,} | '
                        f'{100*r["difference_target_minus_baseline"]:.3f} [{100*lo:.3f}, {100*hi:.3f}] | '
                        f'{r["exact_mcnemar_p"]:.3g} |')
            lines += ['']
        if timing:
            lines += ['## Serial timing', '', '1,024 predetermined pilot rows; means in milliseconds.', '',
                '| Decoder | BP init | BP rounds | Weight projection | UF | Total | p95 total |',
                '|---|---:|---:|---:|---:|---:|---:|']
            for name in VARIANTS:
                r = timing[name]
                parts = ([f'{r[k]:.4f}' for k in ('mean_bp_initialization_ms', 'mean_bp_rounds_ms', 'mean_projection_ms')]
                         if name.startswith('bp') else ['—']*3)
                uf = f'{r["mean_uf_ms"]:.4f}' if name not in ('correlated_uf', 'correlated_mwpm') else '—'
                lines.append(f'| {name} | {" | ".join(parts)} | {uf} | {r["mean_total_ms"]:.4f} | {r["p95_ms"]:.4f} |')
            lines += ['', 'Correlated UF total includes both necessary UF passes and reweighting. '
                'Native timers exclude model setup, Python, unpacking, physical audits and some destruction; '
                'correlated MWPM bound decode includes Python binding overhead. '
                'Fixed variant order and shared BP trajectories affect cache state.', '']
        lines += ['Original primary comparison: BP5 versus correlated UF. Other comparisons are exploratory '
            'and not adjusted for multiple testing. The distance extension was selected after the d=7 results. '
            'BP comparisons include the projection change relative to baseline UF.', '']
        (directory/'report.md').write_text('\n'.join(lines))
    combined += ['', 'Detailed per-distance reports: [d=9](d9/report.md), [d=11](d11/report.md), [d=13](d13/report.md).', '',
        'The d=7 baseline is retained from the earlier study; its BP10 timing was standalone, while new '
        'distances share one BP trajectory across budgets. MWPM and UF/BP timer boundaries differ slightly. '
        'See the frozen protocol and per-distance timing metadata for interpretation.', '']
    (output/'report.md').write_text('\n'.join(combined))


def audit_distance(directory):
    checks = {}
    for phase, shots in SAMPLE_SIZES.items():
        s = json.loads((directory/phase/'summary.json').read_text())
        assert sha256(directory/phase/'results.npz') == s['results_sha256']
        with np.load(directory/phase/'results.npz') as data:
            assert data['predictions'].shape == (shots, len(VARIANTS))
            assert data['variants'].tolist() == VARIANTS
            failed = data['predictions'] != data['actual'][:, None]
            for k, name in enumerate(VARIANTS):
                assert int(failed[:, k].sum()) == s['metrics'][name]['errors']
            for baseline, contrasts in s['paired'].items():
                ref = failed[:, VARIANTS.index(baseline)]
                for target, row in contrasts.items():
                    test = failed[:, VARIANTS.index(target)]
                    assert int(np.count_nonzero(ref & ~test)) == row['repairs']
                    assert int(np.count_nonzero(~ref & test)) == row['regressions']
            np.testing.assert_array_equal(data['fields'][:, :, 11],
                np.broadcast_to([0, 0, 1, 2, 5, 10], (shots, len(NATIVE_VARIANTS))))
        checks[phase] = dict(shots=shots, marginal_and_paired_counts_verified=True, fixed_budgets_verified=True)
    with np.load(directory/'serial_timing.npz') as timing, np.load(directory/'mwpm_timing/serial_timing.npz') as matching:
        np.testing.assert_array_equal(timing['rows'], matching['rows'])
    return dict(status='passed', phases=checks, serial_matching_rows_identical=True,
                physical_native_syndrome_checks=sum(SAMPLE_SIZES.values())*len(NATIVE_VARIANTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    args = parser.parse_args()
    output, work, reference = args.output.resolve(), args.work_dir.resolve(), args.reference.resolve()
    lock = (work/'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = common.physical_cpus(32)
    os.sched_setaffinity(0, cpus)
    states = {str(d): 'queued' for d in DISTANCES}
    started = datetime.now(timezone.utc).isoformat()

    def progress(state, **details):
        d = details.get('distance')
        if d is not None:
            states[str(d)] = state
        value = dict(state=state, started_utc=started, updated_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), threads=32, cpu_affinity=cpus, distances=states, scratch=str(work), **details)
        write_json(output/'progress.json', value)
        if d is not None:
            write_json(output/f'd{d}'/'progress.json', value)

    try:
        progress('verifying_sources')
        for relative, expected in json.loads((output/'source_manifest.json').read_text()).items():
            assert sha256(SNAPSHOT/relative) == expected, relative
        library = work/'distance_sweep.so'
        command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp', '-ffp-contract=off',
                   '-I', str(SNAPSHOT), str(HERE/'kernel.cc'), '-o', str(library)]
        progress('building')
        subprocess.run(command, check=True)
        configuration = dict(distances=DISTANCES, total_physical_cores=32, cpu_affinity=cpus,
            variants=VARIANTS, native_variants=NATIVE_VARIANTS, fields=FIELDS, p=0.003, noise='si1000',
            rounds='4*d', patches=1, yokes=0, style='cz', ideal_time_boundaries=True,
            bp='sum_product_flooding', damping=0.5, llr_clip=30., fixed_budgets=[1, 2, 5, 10],
            always_finish_with_uf=True, projection=common.CONFIG['projection'],
            sample_sizes=SAMPLE_SIZES, serial_timing_shots=1024,
            seed_rule='202609290000 + 100*d + phase; phase=1 pilot,2 confirmation,3 validation,4 timing,5/6 MWPM audits',
            source_manifest_sha256=sha256(output/'source_manifest.json'), native_library_sha256=sha256(library),
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                          stim=stim.__version__, pymatching=pymatching.__version__),
            reference=str(reference), compile_command=command, command=[sys.executable, *sys.argv],
            primary_comparison='bp5_uf versus correlated_uf', parameter_tuning=False)
        if (output/'configuration.json').exists():
            assert json.loads((output/'configuration.json').read_text()) == configuration
        write_json(output/'configuration.json', configuration)
        write_reports(output, reference)
        progress('verifying_d7_regression')
        verified = verify_reference(reference, library)
        write_json(output/'d7_regression.json', verified)
        print('Generalized wrapper reproduces all seven d=7 predictions and native deterministic fields.', flush=True)
        for distance in DISTANCES:
            directory, scratch = output/f'd{distance}', work/f'd{distance}'
            directory.mkdir(exist_ok=True)
            scratch.mkdir(exist_ok=True)
            progress('building_model', distance=distance)
            circuit, dem, native = model(distance, library)
            try:
                circuit_path, dem_path = directory/'circuit.stim', directory/'model.dem'
                circuit_path.write_text(str(circuit))
                dem_path.write_text(str(dem))
                request = dict(distance=distance, rounds=4*distance,
                    global_configuration_sha256=sha256(output/'configuration.json'),
                    circuit_sha256=sha256(circuit_path), dem_sha256=sha256(dem_path),
                    model=native.model.describe(), seeds={name: seed(distance, phase) for name, phase in
                        [('pilot', 1), ('confirmation', 2), ('validation', 3), ('timing', 4), ('mwpm_pilot', 5), ('mwpm_confirmation', 6)]})
                if (directory/'configuration.json').exists():
                    assert json.loads((directory/'configuration.json').read_text()) == request
                write_json(directory/'configuration.json', request)
                print(f'd={distance}: {4*distance} rounds, model {native.model.describe()}', flush=True)
                matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
                progress('validating', distance=distance)
                validation = verify_distance(distance, circuit, dem, native, matching)
                write_json(directory/'verification.json', validation)
                with ProcessPoolExecutor(max_workers=32, mp_context=multiprocessing.get_context('spawn'),
                    initializer=mwpm.start_worker, initargs=(str(dem_path),)) as pool:
                    pilot = collect('pilot', distance, circuit, native, matching, pool, directory, scratch, identity(request), progress)
                    write_reports(output, reference)
                    collect('confirmation', distance, circuit, native, matching, pool, directory, scratch, identity(request), progress)
                    write_reports(output, reference)
                benchmark(distance, native, matching, pilot, directory, progress)
                audit = audit_distance(directory)
                write_json(directory/'completion_audit.json', audit)
                write_reports(output, reference)
                paths = sorted(p for p in directory.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json', 'progress.json'))
                write_json(directory/'artifact_manifest.json', {str(p.relative_to(directory)): sha256(p) for p in paths})
                progress('complete', distance=distance, report=str(directory/'report.md'))
                print(f'd={distance} complete: {directory / "report.md"}', flush=True)
            finally:
                native.close()
        progress('complete', report=str(output/'report.md'))
        print(f'All distances complete: {output / "report.md"}', flush=True)
    except BaseException as error:
        progress('failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
