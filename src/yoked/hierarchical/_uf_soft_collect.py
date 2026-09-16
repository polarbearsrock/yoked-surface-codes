"""Checkpointed collection of UF-only confidence on saved shots.

This is a separate experiment artifact, not an extension of L1Record: collecting
these candidates must never invoke its matching-based confidence path. Every
fresh reference is checked against the existing record. Calibration and evaluation
must have separate sampling families and matching model identities.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import time
from pathlib import Path

import numpy as np
import stim

from yoked.hierarchical._collect import SampleSet, DETECTORS_FILE
from yoked.hierarchical._patch_graphs import PatchGraphs
from yoked.hierarchical._provenance import (
    DECODER_SOURCES, atomic_replacement,
    canonical_json, package_versions, read_json, sha256_bytes, sha256_file,
    source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import load_record
from yoked.hierarchical._uf_soft import DEFAULT_GAP_CAP, SCORE_NAMES, TIME_NAMES, UFSoftDecoder

FEATURE_SOURCES = tuple(sorted(set(DECODER_SOURCES) | {
    'src/yoked/hierarchical/_uf_soft.py',
    'src/yoked/hierarchical/_uf_soft_collect.py',
    'src/yoked/hierarchical/_collect.py',
    'src/yoked/hierarchical/_provenance.py',
}))
VERSIONS = ('numpy', 'stim', 'pymatching', 'sinter')
_WORKER = None


def _start_worker(sample_dir: str, dem: str, patches: int, num_detectors: int, cap: float):
    global _WORKER
    graphs = PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(dem), num_patches=patches)
    _WORKER = (np.load(Path(sample_dir) / DETECTORS_FILE, mmap_mode='r'), num_detectors,
               graphs, tuple(UFSoftDecoder(p, gap_cap=cap) for p in graphs))


def _collect_chunk(rows):
    packed, width, graphs, decoders = _WORKER
    detectors = np.unpackbits(packed[rows], axis=1, count=width, bitorder='little').astype(bool)
    n, p = len(rows), len(graphs)
    arrays = dict(rows=rows, reference=np.empty((n, 2 * p), dtype=bool),
                  correlated_prediction=np.empty((n, 2 * p), dtype=bool),
                  scores=np.empty((n, len(SCORE_NAMES), 2 * p)),
                  forced_costs=np.empty((n, p, 2, 2, 2)),
                  states=np.empty((n, p, 2, 2), dtype=np.int64),
                  rules_fired=np.empty((n, p), dtype=bool),
                  seconds=np.zeros(len(TIME_NAMES)))
    local = graphs.local_syndromes(detectors)
    for shot in range(n):
        for patch, decoder in enumerate(decoders):
            result = decoder.decode(local[shot, patch])
            columns = slice(2 * patch, 2 * patch + 2)
            arrays['reference'][shot, columns] = result.reference
            arrays['correlated_prediction'][shot, columns] = result.correlated_prediction
            arrays['scores'][shot, :, columns] = result.scores
            arrays['forced_costs'][shot, patch] = result.forced_costs
            arrays['states'][shot, patch] = result.states
            arrays['rules_fired'][shot, patch] = result.rules_fired
            arrays['seconds'] += result.seconds
    return arrays


def _write_arrays(path, arrays):
    with atomic_replacement(path) as tmp:
        with tmp.open('wb') as handle:
            np.savez(handle, **arrays)


def collect_features(record_dir, output_dir, *, workers=1, chunk_size=64,
                     gap_cap=DEFAULT_GAP_CAP, limit=None, sample_dir=None):
    """Collect verified, checkpointed features on existing parent rows only."""
    if workers < 1 or chunk_size < 1:
        raise ValueError('workers and chunk_size must be positive')
    loaded = load_record(record_dir)
    sample_dir = Path(record_dir) / 'sample' if sample_dir is None else Path(sample_dir)
    sample = SampleSet.load(sample_dir)
    for key in ('model', 'parent_sample', 'sampling_family'):
        if sample.identities[key] != loaded.identities[key]:
            raise ValueError(f'Sample/record {key} mismatch')
    record = loaded.record
    if limit is not None:
        if not 1 <= limit <= record.shots:
            raise ValueError('limit must be within the record')
        record = record.subset(np.arange(limit))
    _, actual = sample.rows(record.rows)
    if not np.array_equal(actual, record.actual):
        raise ValueError('Sample/record actual bits mismatch')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    request = dict(schema='UFSoftFeatures/1', record_dir=str(Path(record_dir).resolve()),
                   record_manifest_sha256=sha256_file(Path(record_dir) / 'manifest.json'),
                   sample_dir=str(sample_dir.resolve()),
                   source_sha256=source_hashes(FEATURE_SOURCES), versions=package_versions(VERSIONS),
                   gap_cap=gap_cap, score_names=SCORE_NAMES, time_names=TIME_NAMES,
                   rows=record.rows.tolist(), chunk_size=chunk_size)
    identity = sha256_bytes(canonical_json(request).encode())
    request_path = output / 'request.json'
    if request_path.exists():
        if read_json(request_path)['identity'] != identity:
            raise ValueError('Feature collection request changed; use a new output directory')
    else:
        write_json_atomic(request_path, dict(identity=identity, request=request))
    chunks = [record.rows[k:k + chunk_size] for k in range(0, record.shots, chunk_size)]
    chunk_dir = output / 'chunks'
    chunk_dir.mkdir(exist_ok=True)
    def path_for(rows):
        return chunk_dir / f'{int(rows[0]):09d}.npz'
    pending = []
    for rows in chunks:
        path = path_for(rows)
        if not path.exists() or not path.with_suffix('.json').exists():
            pending.append(rows)
        elif read_json(path.with_suffix('.json')) != dict(identity=identity, sha256=sha256_file(path)):
            raise ValueError(f'Corrupt feature checkpoint {path}')
    start = time.perf_counter()
    initargs = (str(sample_dir), sample.dem_text, sample.parameters.patches, sample.num_detectors, gap_cap)
    if workers == 1:
        _start_worker(*initargs)
        iterator = map(_collect_chunk, pending)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_start_worker, initargs=initargs)
        iterator = pool.map(_collect_chunk, pending)
    try:
        completed = record.shots - sum(map(len, pending))
        last_update = time.perf_counter()
        for arrays in iterator:
            positions = np.searchsorted(record.rows, arrays['rows'])
            if not np.array_equal(arrays['reference'], record.uf_reference[positions]):
                raise ValueError('Fresh UF predictions differ from the saved reference')
            if not np.allclose(arrays['scores'][:, 0], np.minimum(record.cluster_gap[positions], gap_cap),
                               rtol=0, atol=1e-10):
                raise ValueError('Bounded UF gap differs from censored saved cluster gap')
            path = path_for(arrays['rows'])
            _write_arrays(path, arrays)
            write_json_atomic(path.with_suffix('.json'), dict(identity=identity, sha256=sha256_file(path)))
            completed += len(arrays['rows'])
            if time.perf_counter() - last_update > 20 or completed == record.shots:
                print(f'{output}: {completed}/{record.shots} rows, {time.perf_counter() - start:.1f}s', flush=True)
                last_update = time.perf_counter()
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    parts = []
    for rows in chunks:
        with np.load(path_for(rows)) as saved:
            part = dict(saved)
        if not np.array_equal(part['rows'], rows):
            raise ValueError('Checkpoint parent rows changed')
        parts.append(part)
    arrays = {name: (np.sum([p[name] for p in parts], axis=0) if name == 'seconds'
                      else np.concatenate([p[name] for p in parts])) for name in parts[0]}
    if source_hashes(FEATURE_SOURCES) != request['source_sha256']:
        raise ValueError('Source changed during collection')
    _write_arrays(output / 'features.npz', arrays)
    write_json_atomic(output / 'manifest.json', dict(
        identity=identity, request_sha256=sha256_file(request_path),
        features_sha256=sha256_file(output / 'features.npz'), shots=record.shots,
        reference_matches=True, bounded_gap_matches=True,
        workers=workers, seconds_this_invocation=time.perf_counter() - start))
    return output


def load_features(directory, loaded):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    request = read_json(directory / 'request.json')
    identity = sha256_bytes(canonical_json(request['request']).encode())
    if (identity != request['identity'] or identity != manifest['identity']
            or manifest['request_sha256'] != sha256_file(directory / 'request.json')
            or manifest['features_sha256'] != sha256_file(directory / 'features.npz')):
        raise ValueError('Feature artifact identity or checksum mismatch')
    if request['request']['record_manifest_sha256'] != sha256_file(loaded.directory / 'manifest.json'):
        raise ValueError('Features belong to another record')
    with np.load(directory / 'features.npz') as saved:
        arrays = dict(saved)
    if not np.array_equal(arrays['rows'], loaded.record.rows):
        raise ValueError('Features must cover every record row for analysis')
    if not np.array_equal(arrays['reference'], loaded.record.uf_reference):
        raise ValueError('Feature reference does not match the record')
    if not np.isfinite(arrays['scores']).all():
        raise ValueError('Nonfinite UF confidence score')
    return arrays, request['request']

