"""Expand fixed BP5 to 100,000 paired shots and recompute correlated MWPM."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pymatching
import stim

from common import (FIELDS, HERE, PILOT, VARIANTS, build, pilot, pilot_run,
                    setup, sha256, source_hashes, write_json)
import check

_matching = None
_packed = None


def start_matching_worker(sample_dir):
    global _matching, _packed
    path = Path(sample_dir)
    dem = stim.DetectorErrorModel.from_file(path/'model.dem')
    _matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    _packed = np.load(path/'detectors_packed.npy', mmap_mode='r')


def matching_chunk(bounds):
    first, last = bounds
    start = time.monotonic()
    bits = _matching.decode_batch(_packed[first:last], bit_packed_shots=True, enable_correlations=True)
    return first, pilot_run.mask_bits(bits), time.monotonic()-start


def environment(args):
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(), cpu_model=cpu,
        affinity_cpus=len(os.sched_getaffinity(0)), platform=platform.platform(),
        compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
        native_threads=args.threads, matching_workers=args.matching_workers,
        openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
        omp_proc_bind=os.environ.get('OMP_PROC_BIND'), omp_places=os.environ.get('OMP_PLACES'),
        timing_scope='Serial native evidence initialization, inference and projection plus UF growth/tree/peeling. '
            'Excludes model setup, Python, packing, full-syndrome audit and some temporary destruction. '
            'Shared work is charged fully to each variant; fixed decoder order. '
            'Parallel per-shot durations are contended durations, not decoder latency.',
        serial_benchmark_shots_per_distance=16)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_checkpoint(path, request_id):
    metadata = path.with_suffix('.json')
    if not path.exists() or not metadata.exists():
        return None
    record = json.loads(metadata.read_text())
    if record['identity'] != request_id or record['sha256'] != sha256(path):
        raise ValueError(f'Checkpoint identity/checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    return arrays, record['wall_seconds']


def save_checkpoint(path, request_id, elapsed, **arrays):
    partial = path.with_name(path.stem+'.partial.npz')
    np.savez_compressed(partial, **arrays)
    partial.replace(path)
    write_json(path.with_suffix('.json'), dict(identity=request_id, sha256=sha256(path), wall_seconds=elapsed))


def run_distance(args, library, command, sources, distance):
    started = time.monotonic()
    output = args.output/f'd{distance}'
    output.mkdir(parents=True, exist_ok=True)
    sample, previous, provenance = pilot_run.prepare_sample(distance, args.sample_root)
    rows = np.arange(sample.shots, dtype=np.int64)
    with np.load(PILOT/f'd{distance}'/'results.npz', allow_pickle=False) as saved:
        pilot_arrays = dict(saved)
    pilot_positions = np.full(sample.shots, -1, dtype=np.int32)
    pilot_positions[pilot_arrays['rows']] = np.arange(len(pilot_arrays['rows']))
    dem = stim.DetectorErrorModel(sample.dem_text)
    print(f'd={distance}: construct frozen joint fault model', flush=True)
    graph, model, native = setup(dem, library)
    request = dict(distance=distance, shots=sample.shots, variants=list(VARIANTS), fields=list(FIELDS),
        bp_iterations=5, damping=pilot.DAMPING, llr_limit=pilot.LLR_LIMIT,
        projection='negative_log_capped_sum', correlation_rules='original prior-derived DEM rules; min with BP weights',
        selection='all parent sample rows in original order', inputs=provenance,
        pilot_results_sha256=sha256(PILOT/f'd{distance}'/'results.npz'),
        sources=sources, library_sha256=sha256(library), compile_command=command,
        model=model.describe(), threads=args.threads, chunk_size=args.chunk_size,
        matching_workers=args.matching_workers, matching_chunk_size=2500,
        versions=dict(numpy=np.__version__, stim=stim.__version__, pymatching=pymatching.__version__,
                      python=platform.python_version()),
        baselines=dict(correlated_mwpm='Recomputed on all rows; verified against archive',
                       correlated_uf='Archived identical-sample predictions; checked on predetermined real rows'))
    request_id = identity(request)
    request['identity'] = request_id
    request_path = output/'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Request changed; use a new output directory')
    write_json(request_path, request)
    print(f'd={distance}: independent physical corrections, BP values and thread checks', flush=True)
    validation = check.real(native, dem, sample, previous, library, args.threads, distance)
    write_json(output/'preflight.json', validation)
    chunks = args.work_dir/f'd{distance}'/'chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    matching_path = args.work_dir/f'd{distance}'/'matching.npz'
    matching_saved = load_checkpoint(matching_path, request_id)
    pool, futures = None, []
    if matching_saved is None:
        pool = ProcessPoolExecutor(max_workers=args.matching_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=start_matching_worker, initargs=(str(args.sample_root/f'd{distance}'/'sample'),))
        matching_start = time.monotonic()
        futures = [pool.submit(matching_chunk, (first, min(first+2500, sample.shots)))
                   for first in range(0, sample.shots, 2500)]
        print(f'd={distance}: correlated MWPM recomputation started ({args.matching_workers} workers)', flush=True)
    parts, elapsed_parts = [], []
    decode_start = time.monotonic()
    overlap_count = 0
    try:
        for first in range(0, sample.shots, args.chunk_size):
            last = min(first+args.chunk_size, sample.shots)
            path = chunks/f'{first:06d}.npz'
            saved = load_checkpoint(path, request_id)
            if saved is None:
                tick = time.monotonic()
                out, _ = native.decode(sample.detectors_packed[first:last], args.threads)
                elapsed = time.monotonic()-tick
                save_checkpoint(path, request_id, elapsed, out=out, rows=rows[first:last])
            else:
                arrays, elapsed = saved
                np.testing.assert_array_equal(arrays['rows'], rows[first:last])
                out = arrays['out']
            if out.shape != (last-first, len(VARIANTS), len(FIELDS)) or not np.isfinite(out).all():
                raise ValueError('Unexpected checkpoint output')
            pilot_run.check_yokes(out[:, :, 0].astype(np.uint16), sample.detectors_packed[first:last], graph.num_detectors)
            present = np.flatnonzero(pilot_positions[first:last] >= 0)
            pp = pilot_positions[first+present]
            for field, column in [('predictions', 0), ('costs', 1), ('selected_edges', 2),
                                  ('forest_edges', 3), ('max_cluster', 4), ('epochs', 5),
                                  ('edge_evaluations', 6), ('frontier_visits', 7), ('zero_weight_edges', 8)]:
                np.testing.assert_array_equal(out[present, 0, column], pilot_arrays[field][pp, 3])
            overlap_count += len(present)
            parts.append(out)
            elapsed_parts.append(elapsed)
            if first == 0 or (last//args.chunk_size) % 4 == 0 or last == sample.shots:
                seconds = time.monotonic()-decode_start
                status = dict(distance=distance, decoded=last, shots=sample.shots,
                    elapsed_seconds=seconds, rate_shots_per_second=last/max(seconds, 1e-9),
                    overlap_verified=overlap_count, matching_chunks_completed=sum(f.done() for f in futures))
                write_json(output/'progress.json', status)
                print(json.dumps(status), flush=True)
        out = np.concatenate(parts)
        assert overlap_count == len(pilot_arrays['rows']) == 5000
        if matching_saved is None:
            matching = np.empty(sample.shots, dtype=np.uint16)
            matching_times = []
            for future in futures:
                first, values, elapsed = future.result()
                matching[first:first+len(values)] = values
                matching_times.append(elapsed)
            pool.shutdown()
            pool = None
            matching_elapsed = time.monotonic()-matching_start
            np.testing.assert_array_equal(matching, previous['matching'])
            save_checkpoint(matching_path, request_id, matching_elapsed, matching=matching,
                            chunk_compute_seconds=np.array(matching_times))
        else:
            matching = matching_saved[0]['matching']
            matching_elapsed = matching_saved[1]
        np.testing.assert_array_equal(matching, previous['matching'])
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    benchmark_rows = np.sort(np.random.default_rng(2026092110+distance).choice(sample.shots, 16, replace=False))
    print(f'd={distance}: all 100,000 MWPM outputs verified; serial timing sample', flush=True)
    serial, _ = native.decode(sample.detectors_packed[benchmark_rows], 1)
    np.testing.assert_array_equal(serial[:, :, :9], out[benchmark_rows, :, :9])
    arrays = dict(rows=rows, predictions=out[:, :, 0].astype(np.uint16), actual=previous['actual'],
        matching=matching, correlated_uf=previous['correlated_uf'], pilot_rows=pilot_arrays['rows'],
        benchmark_rows=benchmark_rows, serial_evidence_seconds=serial[:, :, 9], serial_uf_seconds=serial[:, :, 10])
    for column, name in enumerate(FIELDS[1:], 1):
        arrays[name] = out[:, :, column]
    np.savez_compressed(output/'results.npz', **arrays)
    validation.update(full_syndrome_checks=sample.shots*len(VARIANTS),
        ideal_yoke_prediction_checks=sample.shots*len(VARIANTS),
        pilot_bp5_predictions_and_all_nontiming_diagnostics_reproduced=overlap_count,
        recomputed_correlated_mwpm_matches=sample.shots, truth_passed_to_native=False,
        batch_decode_wall_seconds=sum(elapsed_parts), matching_wall_seconds_until_collection=matching_elapsed,
        total_elapsed_seconds=time.monotonic()-started, result_sha256=sha256(output/'results.npz'),
        note='Matching ran concurrently; collection wall time is not isolated matching latency.')
    native.close()
    write_json(output/'verification.json', validation)
    shutil.copyfile(args.sample_root/f'd{distance}'/'sample'/'sample.json', output/'sample.json')
    if source_hashes() != sources:
        raise ValueError('Sources changed during the experiment')
    counts = {name:int(np.count_nonzero(arrays['predictions'][:, k] != arrays['actual']))
              for k, name in enumerate(VARIANTS)}
    counts.update(correlated_mwpm=int(np.count_nonzero(matching != arrays['actual'])),
                  correlated_uf=int(np.count_nonzero(previous['correlated_uf'] != arrays['actual'])))
    print(f'd={distance}: COMPLETE {json.dumps(counts)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distances', nargs='+', type=int, choices=(7,9,11,13), default=[7,9,11,13])
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--matching-workers', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=512)
    args = parser.parse_args()
    if min(args.threads, args.matching_workers, args.chunk_size) < 1:
        parser.error('Need positive thread, worker and chunk counts')
    for path in (args.work_dir, args.sample_root, args.output):
        if not path.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve()):
            parser.error('Scratch paths must be under TMPDIR')
    args.output.mkdir(parents=True, exist_ok=True)
    library, command = build(args.work_dir/'build')
    sources = source_hashes()
    write_json(args.output/'environment.json', environment(args))
    checks = check.synthetic(library)
    write_json(args.output/'verification.json', dict(status='passed', synthetic_checks=checks,
        sources=sources, compiler_command=command, library_sha256=sha256(library)))
    print('Synthetic exact-posterior, physical-correction and thread checks passed', flush=True)
    for distance in args.distances:
        run_distance(args, library, command, sources, distance)


if __name__ == '__main__':
    main()
