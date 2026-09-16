"""Run a final-pass UF forest ensemble on verified, previously saved shots.

Run from the repository root with PYTHONPATH=src and BLAS/OMP threads set to 1.
All intermediate files belong in the supplied --output directory under TMPDIR.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import stim

from yoked.decoders import CosetEnsembleDecoder
from yoked.hierarchical._provenance import (
    canonical_json, package_versions, packed_sample_hash, sha256_bytes, sha256_file,
    source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import load_record
from yoked.hierarchical._uf_soft_collect import _write_arrays

PREFIXES = (1, 4, 8, 16, 24)
SCORES = ('size', 'weight')
SOURCES = (
    'src/yoked/decoders/_coset_ensemble.py',
    'src/yoked/decoders/_union_find.py',
    'src/yoked/decoders/_graph.py',
    'src/yoked/decoders/_correlations.py',
    'docs/results/coset_ensemble_d7_d9_p003_100k/run.py',
)
_WORKER = None


def masks(bits):
    return np.asarray(bits, dtype=np.uint16) @ (1 << np.arange(bits.shape[1], dtype=np.uint16))


def collect_chunk(rows):
    decoder, packed, expected, endpoints, labels = _WORKER
    n, k = len(rows), decoder.num_candidates
    arrays = dict(
        rows=rows, baseline=np.empty(n, dtype=np.uint16),
        predictions=np.empty((n, len(SCORES), len(PREFIXES)), dtype=np.uint16),
        candidate_masks=np.empty((n, k), dtype=np.uint16),
        candidate_sizes=np.empty((n, k), dtype=np.uint32),
        candidate_weights=np.empty((n, k), dtype=np.float64),
        baseline_sizes=np.empty(n, dtype=np.uint32),
        baseline_weights=np.empty(n),
        logical_rank=np.empty(n, dtype=np.uint8),
        internal_edges=np.empty(n, dtype=np.uint32),
        physical_diversity=np.empty(n, dtype=np.uint8),
        decode_seconds=np.empty(n), validation_seconds=np.empty(n),
    )
    for i, row in enumerate(rows):
        syndrome = np.unpackbits(packed[row], count=decoder.graph.num_detectors, bitorder='little')
        start = time.perf_counter()
        result = decoder.decode_ensemble(syndrome)
        arrays['decode_seconds'][i] = time.perf_counter() - start
        if result.baseline.observable_mask != expected[row]:
            raise ValueError(f'Fresh correlated UF differs from saved baseline at row {row}')
        start = time.perf_counter()
        for candidate in (result.baseline, *result.candidates):
            ids = np.asarray(candidate.selected_edges, dtype=np.int64)
            parity = np.bincount(endpoints[ids].ravel(), minlength=len(decoder.graph.adjacency))
            if not np.array_equal(parity[:decoder.graph.num_detectors] % 2, syndrome):
                raise ValueError(f'Candidate violates syndrome at row {row}')
            if int(np.bitwise_xor.reduce(labels[ids], initial=0)) != candidate.observable_mask:
                raise ValueError(f'Candidate logical mask mismatch at row {row}')
        if result.logical_rank == 0 and any(
                c.observable_mask != result.baseline.observable_mask for c in result.candidates):
            raise ValueError(f'Candidate violates logical-rank certificate at row {row}')
        arrays['validation_seconds'][i] = time.perf_counter() - start
        arrays['baseline'][i] = result.baseline.observable_mask
        arrays['baseline_sizes'][i] = len(result.baseline.selected_edges)
        arrays['baseline_weights'][i] = result.baseline.weight
        arrays['candidate_masks'][i] = [c.observable_mask for c in result.candidates]
        arrays['candidate_sizes'][i] = [len(c.selected_edges) for c in result.candidates]
        arrays['candidate_weights'][i] = [c.weight for c in result.candidates]
        arrays['logical_rank'][i] = result.logical_rank
        arrays['internal_edges'][i] = result.internal_edges
        arrays['physical_diversity'][i] = len({c.selected_edges for c in result.candidates})
        for si, score in enumerate(SCORES):
            for ki, count in enumerate(PREFIXES):
                arrays['predictions'][i, si, ki] = result.choose(score, count=count).observable_mask
    return arrays


def run_distance(record_dir, output, *, workers, chunk_size, shots, seed):
    global _WORKER
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_record(record_dir)
    record = loaded.record
    if not 1 <= shots <= record.shots:
        raise ValueError('Requested shots outside saved sample')
    if not np.array_equal(record.rows, np.arange(record.shots)):
        raise ValueError('This experiment expects the complete saved sample in original order')
    run = loaded.manifest['baselines']['run']
    directory = Path(run['directory'])
    packed = np.load(directory / 'detectors_packed.npy', mmap_mode='r')
    actual = np.load(directory / 'actual_observables_packed.npy', mmap_mode='r')
    if packed_sample_hash(packed, actual) != run['payload_sha256']:
        raise ValueError('Saved sample hash mismatch')
    np.testing.assert_array_equal(np.unpackbits(actual, axis=1, count=12, bitorder='little'), record.actual)
    for filename, key in [('model.dem', 'dem_sha256'), ('circuit.stim', 'circuit_sha256')]:
        if sha256_file(directory / filename) != run[key]:
            raise ValueError(f'Saved {filename} hash mismatch')
    decoder = CosetEnsembleDecoder.from_dem(
        stim.DetectorErrorModel.from_file(directory / 'model.dem'), correlated=True,
        candidates=max(PREFIXES), seed=seed)
    if decoder.graph.num_observables != 12:
        raise ValueError('Expected six patches and twelve physical observable columns')
    request = dict(
        schema='coset-ensemble/1', record_dir=str(record_dir.resolve()),
        record_manifest_sha256=sha256_file(record_dir / 'manifest.json'),
        input_run={key: run[key] for key in ('directory', 'payload_sha256', 'dem_sha256', 'circuit_sha256')},
        parameters=dict(loaded.manifest['parameters']),
        shots=shots, seed=seed, prefixes=PREFIXES, scores=SCORES,
        source_sha256=source_hashes(SOURCES),
        versions=package_versions(('numpy', 'stim', 'pymatching', 'sinter')),
        chunk_size=chunk_size, protocol='correlated UF with final-pass ensemble; all internal edges',
        boundaries='one free-parity virtual vertex per final cluster',
        priority_rng='PCG64; fixed per candidate across shots',
        fast_paths='none; all 24 candidates generated for every shot',
    )
    identity = sha256_bytes(canonical_json(request).encode())
    request_file = output / 'request.json'
    if request_file.exists():
        saved = json.loads(request_file.read_text())
        if saved['identity'] != identity or canonical_json(saved['request']) != canonical_json(request):
            raise ValueError('Experiment request changed; use a new output directory')
    else:
        write_json_atomic(request_file, dict(identity=identity, request=request))
    # Fork shares these immutable arrays and priorities without rebuilding each
    # DEM per worker. The experiment is Linux-only, with numerical threads = 1.
    _WORKER = (decoder, packed, masks(record.baselines['joint_correlated_uf']),
               np.asarray(decoder.graph.endpoints, dtype=np.int64),
               np.asarray([e[3] for e in decoder.graph.edges], dtype=np.uint16))
    chunk_dir = output / 'chunks'
    chunk_dir.mkdir(exist_ok=True)
    chunks = [np.arange(start, min(start + chunk_size, shots)) for start in range(0, shots, chunk_size)]

    def path_for(rows):
        return chunk_dir / f'{int(rows[0]):06d}.npz'

    pending = []
    for rows in chunks:
        path = path_for(rows)
        if not path.exists() or not path.with_suffix('.json').exists():
            pending.append(rows)
        elif json.loads(path.with_suffix('.json').read_text()) != dict(identity=identity, sha256=sha256_file(path)):
            raise ValueError(f'Corrupt checkpoint: {path}')
    start = last_update = time.perf_counter()
    completed = shots - sum(map(len, pending))
    print(f'{output.name}: starting {len(pending)} chunks, {workers} workers', flush=True)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(collect_chunk, rows) for rows in pending]
        for future in concurrent.futures.as_completed(futures):
            arrays = future.result()
            path = path_for(arrays['rows'])
            _write_arrays(path, arrays)
            write_json_atomic(path.with_suffix('.json'), dict(identity=identity, sha256=sha256_file(path)))
            completed += len(arrays['rows'])
            if time.perf_counter() - last_update >= 20 or completed == shots:
                print(f'{output.name}: {completed}/{shots}, {time.perf_counter() - start:.1f}s', flush=True)
                last_update = time.perf_counter()
    parts = []
    for rows in chunks:
        with np.load(path_for(rows)) as saved:
            part = dict(saved)
        np.testing.assert_array_equal(part['rows'], rows)
        parts.append(part)
    arrays = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    if source_hashes(SOURCES) != request['source_sha256']:
        raise ValueError('Decoder or runner sources changed during collection')
    _write_arrays(output / 'candidates.npz', arrays)
    write_json_atomic(output / 'manifest.json', dict(
        identity=identity, request_sha256=sha256_file(request_file),
        candidates_sha256=sha256_file(output / 'candidates.npz'), shots=shots,
        candidates_checked=shots * max(PREFIXES), baseline_matches=shots,
        workers=workers, wall_seconds_this_invocation=time.perf_counter() - start,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-root', type=Path, default=Path(os.environ['TMPDIR']) /
                        'hier-three-decoders-d7-d9-p003-100k-3MxG5S')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distances', nargs='+', type=int, choices=(7, 9), default=[7, 9])
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260916)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        parser.error('workers and chunk size must be positive')
    for d in args.distances:
        run_distance(args.record_root / f'd{d}' / 'evaluation', args.output / f'd{d}',
                     workers=args.workers, chunk_size=args.chunk_size, shots=args.shots, seed=args.seed)


if __name__ == '__main__':
    main()
