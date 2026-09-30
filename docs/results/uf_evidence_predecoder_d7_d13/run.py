"""Run a paired pilot of soft evidence followed by unchanged UF."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pymatching
import scipy.sparse
from scipy.sparse.csgraph import connected_components
import stim

from evidence import (DAMPING, FIELDS, HERE, ITERATIONS, LLR_LIMIT, REPO, VARIANTS,
                      build, load_module, setup, sha256, source_hashes, write_json)
from verify import check_physical, synthetic
from yoked.hierarchical._collect import CircuitParameters, SampleSet

CERTIFICATE_SOURCE = HERE.parent/'correlated_uf_mwpm_failure_analysis_d7_d9'/'analyze.py'
certificate = load_module('uf_evidence_certificate', CERTIFICATE_SOURCE)


def archive(distance):
    name = ('parallel_uf_clustering_d7_d9_p003_100k' if distance < 10 else
            'parallel_uf_clustering_d11_d13_p003_100k')
    return HERE.parent/name/f'd{distance}'


def mask_bits(bits):
    return np.asarray(bits, dtype=np.uint16) @ (1 << np.arange(bits.shape[1], dtype=np.uint16))


def prepare_sample(distance, work):
    directory = work/f'd{distance}'/'sample'
    parameters = CircuitParameters(distance=distance, rounds=4*distance, p=.003,
                                   patches=6, yokes=2, style='cz', noise='si1000')
    if not (directory/'sample.json').exists():
        print(f'd={distance}: reproduce the complete 100,000-shot parent sample', flush=True)
        SampleSet.sample(parameters, seed=42, shots=100000).save(directory)
    sample = SampleSet.load(directory)
    if sample.parameters != parameters or sample.seed != 42 or sample.shots != 100000:
        raise ValueError('Unexpected sample parameters')
    old = json.loads((archive(distance)/'request.json').read_text())
    old_input = old['input_run'] if distance < 10 else old['sample']
    if sample.payload_sha256 != old_input['payload_sha256']:
        raise ValueError('Parent payload differs from archive; do not reuse archived predictions')
    text_checks = {}
    for name, text, key in [('circuit', sample.circuit_text, 'circuit_sha256'),
                            ('dem', sample.dem_text, 'dem_sha256')]:
        current = hashlib.sha256(text.encode()).hexdigest()
        if current == old_input[key]:
            convention = 'exact_bytes'
        elif text.endswith('\n') and hashlib.sha256(text[:-1].encode()).hexdigest() == old_input[key]:
            convention = 'exact_after_removing_one_trailing_newline_from_current_text'
        else:
            raise ValueError(f'{name} content differs from archived model')
        text_checks[name] = dict(current_sha256=current, archived_sha256=old_input[key], match=convention)
    with np.load(archive(distance)/'results.npz', allow_pickle=False) as saved:
        previous = dict(actual=saved['actual'], matching=saved['matching'],
                        correlated_uf=saved['predictions'][:, 0])
    actual = mask_bits(np.unpackbits(sample.actual_packed, axis=1, count=12, bitorder='little'))
    np.testing.assert_array_equal(previous['actual'], actual)
    provenance = dict(parent_payload_reproduced_exactly=True, text_checks=text_checks,
                      archived_request_sha256=sha256(archive(distance)/'request.json'),
                      archived_results_sha256=sha256(archive(distance)/'results.npz'),
                      sample_identity=sample.identity_inputs(), sample_directory=str(directory))
    return sample, previous, provenance


def check_yokes(predictions, packed, nd):
    for sector in (0, 1):
        parity = np.zeros_like(predictions, dtype=np.uint8)
        for bit in range(sector, 12, 2):
            parity ^= ((predictions >> bit) & 1).astype(np.uint8)
        detector = nd-2+sector
        expected = ((packed[:, detector//8] >> (detector % 8)) & 1)
        if predictions.ndim == 2:
            expected = np.broadcast_to(expected[:, None], predictions.shape)
        np.testing.assert_array_equal(parity, expected)


def validate_real(native, dem, sample, rows, previous, threads):
    chosen = rows[:4]
    packed = sample.detectors_packed[chosen]
    serial, sf = native.decode(packed, 1, audit=True)
    parallel, pf = native.decode(packed, threads, audit=True)
    np.testing.assert_array_equal(serial[:, :, :9], parallel[:, :, :9])
    np.testing.assert_array_equal(sf, pf)
    np.testing.assert_array_equal(serial[:, 1, 0].astype(np.uint16), previous['correlated_uf'][chosen])
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    predictions = mask_bits(matching.decode_batch(packed, bit_packed_shots=True, enable_correlations=True))
    np.testing.assert_array_equal(predictions, previous['matching'][chosen])
    checks = []
    for row, syndrome in zip(chosen, np.unpackbits(packed, axis=1, count=sample.num_detectors, bitorder='little')):
        check = check_physical(native, dem, syndrome)
        checks.append(dict(row=int(row), **check))
    return dict(rows=chosen.tolist(), physical_checks=checks, native_thread_counts=[1, threads],
                thread_results_identical_except_timing=True, saved_mwpm_predictions_reproduced=len(chosen))


def diagnose(native, sample, arrays, distance):
    target = np.flatnonzero((arrays['predictions'][:, 1] != arrays['actual']) &
                            (arrays['matching'] == arrays['actual']))
    rng = np.random.default_rng(2026092070+distance)
    selected = np.sort(rng.choice(target, min(32, len(target)), replace=False))
    graph = native.graph
    ends = np.asarray(graph.endpoints)
    records = []
    # Bound the audit flags' memory independently of graph size.
    for position in selected:
        row = int(arrays['rows'][position])
        out, flags = native.decode(sample.detectors_packed[row:row+1], audit=True)
        np.testing.assert_array_equal(out[0, :, 0], arrays['predictions'][position])
        record = dict(row=row, actual=int(arrays['actual'][position]), variants={})
        for k in (1, 3, 4):
            forest = np.flatnonzero(flags[0] & (1 << (2*k)))
            u, v = ends[forest].T
            adjacency = scipy.sparse.coo_matrix((np.ones(len(u)), (u, v)),
                                                 shape=(len(graph.adjacency),)*2).tocsr()
            _, owner = connected_components(adjacency, directed=False)
            allowed = np.flatnonzero(owner[ends[:, 0]] == owner[ends[:, 1]])
            difference = int(arrays['predictions'][position, k]) ^ int(arrays['actual'][position])
            forest_ok = certificate.contains(certificate.logical_basis(graph, forest), difference)
            partition_ok = certificate.contains(certificate.logical_basis(graph, allowed), difference)
            if forest_ok and not partition_ok:
                raise AssertionError('Induced partition excludes a forest alternative')
            record['variants'][VARIANTS[k]] = dict(correct=difference == 0,
                forest_allows_truth=forest_ok, partition_allows_truth=partition_ok,
                forest_edges=len(forest), partition_edges=len(allowed))
        records.append(record)
    return dict(selection='Uniform subset of correlated-UF-fails / correlated-MWPM-succeeds '
                           'shots within the pilot; diagnostic only, not an LER estimate.',
                eligible=len(target), shots=len(records), rows=records)


def run_distance(args, library, compile_command, distance, sources):
    start = time.monotonic()
    output = args.output/f'd{distance}'
    output.mkdir(parents=True, exist_ok=True)
    sample, previous, provenance = prepare_sample(distance, args.work_dir)
    # Select by index before accessing any outcome for the new methods.
    rows = np.sort(np.random.default_rng(2026092000+distance).choice(sample.shots, args.shots, replace=False))
    dem = stim.DetectorErrorModel(sample.dem_text)
    graph, model, native = setup(dem, library)
    print(f'd={distance}: {model.nf:,} fault variables; {len(model.detectors):,} incidences', flush=True)
    request = dict(distance=distance, shots=args.shots, parent_shots=sample.shots,
        selection_seed=2026092000+distance, row_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        variants=list(VARIANTS), fields=list(FIELDS), bp_iterations=list(ITERATIONS),
        damping=DAMPING, llr_limit=LLR_LIMIT, projection='negative_log_capped_sum',
        inputs=provenance, sources=sources, library_sha256=sha256(library), compile_command=compile_command,
        model=model.describe(), versions=dict(numpy=np.__version__, stim=stim.__version__,
        pymatching=pymatching.__version__, python=platform.python_version()), threads=args.threads,
        chunk_size=args.chunk_size, certificate_source_sha256=sha256(CERTIFICATE_SOURCE))
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    request['identity'] = identity
    request_path = output/'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Request changed; use a new output directory')
    write_json(request_path, request)
    print(f'd={distance}: verify BP weights and UF physical corrections independently', flush=True)
    validation = validate_real(native, dem, sample, rows, previous, args.threads)
    work = args.work_dir/f'd{distance}'/'chunks'
    work.mkdir(parents=True, exist_ok=True)
    parts, chunk_seconds = [], []
    decode_start = time.monotonic()
    for first in range(0, len(rows), args.chunk_size):
        last = min(first+args.chunk_size, len(rows))
        path = work/f'{first:06d}.npz'
        metadata = path.with_suffix('.json')
        if path.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved['identity'] != identity or saved['sha256'] != sha256(path):
                raise ValueError('Checkpoint identity or checksum differs')
            with np.load(path, allow_pickle=False) as saved_data:
                out = saved_data['out']
                np.testing.assert_array_equal(saved_data['rows'], rows[first:last])
            elapsed = saved['wall_seconds']
        else:
            tick = time.monotonic()
            out, _ = native.decode(sample.detectors_packed[rows[first:last]], args.threads)
            elapsed = time.monotonic()-tick
            np.testing.assert_array_equal(out[:, 1, 0].astype(np.uint16), previous['correlated_uf'][rows[first:last]])
            check_yokes(out[:, :, 0].astype(np.uint16), sample.detectors_packed[rows[first:last]], graph.num_detectors)
            temporary = path.with_name(path.stem+'.partial.npz')
            np.savez_compressed(temporary, out=out, rows=rows[first:last])
            temporary.replace(path)
            write_json(metadata, dict(identity=identity, sha256=sha256(path), wall_seconds=elapsed))
        parts.append(out)
        chunk_seconds.append(elapsed)
        print(f'd={distance}: {last}/{len(rows)} decoded, {time.monotonic()-decode_start:.1f}s; baseline verified', flush=True)
    out = np.concatenate(parts)
    np.testing.assert_array_equal(out[:, 1, 0].astype(np.uint16), previous['correlated_uf'][rows])
    check_yokes(out[:, :, 0].astype(np.uint16), sample.detectors_packed[rows], graph.num_detectors)
    arrays = dict(rows=rows, predictions=out[:, :, 0].astype(np.uint16),
                   actual=previous['actual'][rows], matching=previous['matching'][rows],
                   costs=out[:, :, 1], selected_edges=out[:, :, 2].astype(np.uint32),
                   forest_edges=out[:, :, 3].astype(np.uint32), max_cluster=out[:, :, 4].astype(np.uint32),
                   epochs=out[:, :, 5].astype(np.uint32), edge_evaluations=out[:, :, 6].astype(np.uint32),
                   frontier_visits=out[:, :, 7].astype(np.uint32), zero_weight_edges=out[:, :, 8].astype(np.uint32),
                   parallel_evidence_seconds=out[:, :, 9], parallel_uf_seconds=out[:, :, 10])
    # Isolated serial kernel measurements avoid presenting contended per-shot
    # parallel durations as latency. Sixteen samples are not a tail-latency study.
    benchmark_indices = np.sort(np.random.default_rng(2026092050+distance).choice(len(rows), min(16, len(rows)), replace=False))
    serial, _ = native.decode(sample.detectors_packed[rows[benchmark_indices]], 1)
    np.testing.assert_array_equal(serial[:, :, 0].astype(np.uint16), arrays['predictions'][benchmark_indices])
    arrays.update(benchmark_rows=rows[benchmark_indices], serial_evidence_seconds=serial[:, :, 9],
                   serial_uf_seconds=serial[:, :, 10])
    np.savez_compressed(output/'results.npz', **arrays)
    print(f'd={distance}: audit logical accessibility on 32 conditional disagreements', flush=True)
    diagnosis = diagnose(native, sample, arrays, distance)
    native.close()
    write_json(output/'diagnostic.json', diagnosis)
    validation.update(full_syndrome_checks=len(rows)*len(VARIANTS), saved_baseline_matches=len(rows),
        ideal_yoke_prediction_checks=len(rows)*len(VARIANTS), truth_passed_to_native=False,
        batch_decode_wall_seconds=sum(chunk_seconds), total_elapsed_seconds=time.monotonic()-start,
        result_sha256=sha256(output/'results.npz'), diagnostic_sha256=sha256(output/'diagnostic.json'))
    write_json(output/'verification.json', validation)
    shutil.copyfile(args.work_dir/f'd{distance}'/'sample'/'sample.json', output/'sample.json')
    if source_hashes() != sources:
        raise ValueError('Sources changed during the experiment')
    print(f'd={distance}: complete and verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distances', nargs='+', type=int, choices=(7,9,11,13), default=[7,9,11,13])
    parser.add_argument('--shots', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=128)
    args = parser.parse_args()
    if not 4 <= args.shots <= 100000 or args.threads < 1 or args.chunk_size < 1:
        parser.error('Need at least four shots, positive threads/chunks, and at most the full parent sample')
    if not args.work_dir.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('Scratch work must be under TMPDIR')
    args.output.mkdir(parents=True, exist_ok=True)
    library, command = build(args.work_dir/'build')
    sources = source_hashes()
    checks = synthetic(library)
    write_json(args.output/'verification.json', dict(status='passed', synthetic_checks=checks,
        sources=sources, compiler_command=command, library_sha256=sha256(library)))
    for distance in args.distances:
        run_distance(args, library, command, distance, sources)


if __name__ == '__main__':
    main()
