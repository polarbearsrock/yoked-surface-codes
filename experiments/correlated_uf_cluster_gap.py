#!/usr/bin/env python3
"""Paired correlated-UF experiment using the retained 4d-round MWPM samples.

Each subprocess owns its decoder and a disjoint range of sample rows. Completed
shards are verified before reuse. Incomplete work stays in temporary directories,
so a failed subprocess cannot publish a partial shard as a completed result.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
REPO = Path(os.environ['DANTE_REPO'])
sys.path.insert(0, str(REPO / 'src'))

import numpy as np

from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._correlated_uf import CORRELATED_UF_CONFIGURATION
from yoked.hierarchical._correlated_uf_experiment import SOURCES, load_baseline, summarize_accuracy
from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
from yoked.hierarchical._provenance import (
    atomic_replacement, git_commit, package_versions, sha256_file, source_hashes,
    utc_now, write_json_atomic,
)


def decode_shard(root, baseline, distance, start, stop):
    """Publish only a successfully completed child process's output directory."""
    shards = root / f'd{distance}' / 'shards'
    destination = shards / f'{start:06d}-{stop:06d}'
    if (destination / 'manifest.json').exists():
        return
    if destination.exists():
        raise ValueError(f'Unrecognized partial shard: {destination}')
    with tempfile.TemporaryDirectory(prefix='.working-', dir=shards) as temporary:
        output = Path(temporary) / 'output'
        command = [sys.executable, str(REPO / 'tools/correlated_uf_experiment'),
                   '--baseline-run', str(baseline / f'd{distance}'),
                   '--rows', f'{start}:{stop}', '--out', str(output),
                   '--batch-size', '32', '--bootstrap-replicates', '1000']
        log = root / 'logs' / f'd{distance}-{start:06d}-{stop:06d}.log'
        with log.open('w') as handle:
            handle.write('Command arguments: ' + json.dumps(command) + '\n')
            handle.flush()
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        output.rename(destination)


def verified_shard(directory, sample, rows, configuration, distance):
    """Check identity, hashes, row coverage, truth, and the candidate's frame."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != 'correlated-uf-cluster-gap/1':
        raise ValueError('Unexpected shard format')
    if manifest['configuration'] != CORRELATED_UF_CONFIGURATION:
        raise ValueError('Unexpected decoder configuration')
    for key in ('source_sha256', 'versions'):
        if manifest[key] != configuration['provenance'][key]:
            raise ValueError(f'Shard {key} differs from this run')
    if manifest['sample']['identities'] != dict(sample.identities):
        raise ValueError('Shard sample identity mismatch')
    if manifest['baseline']['manifest_sha256'] != configuration['baseline_manifests'][str(distance)]:
        raise ValueError('Shard was compared with a different MWPM baseline')
    for name, expected in manifest['artifacts'].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f'Shard artifact mismatch: {directory / name}')
    with np.load(directory / 'predictions.npz') as stored:
        arrays = {name: stored[name] for name in stored.files}
    np.testing.assert_array_equal(arrays['row_ids'], rows)
    detectors, actual = sample.rows(rows)
    np.testing.assert_array_equal(arrays['actual'], actual)
    yokes = detectors[:, -2:]
    np.testing.assert_array_equal(arrays['sigma'], frame_adjusted_syndrome(yokes, arrays['reference']))
    np.testing.assert_array_equal(arrays['prediction'], arrays['reference'] ^ arrays['residual'])
    parity = arrays['prediction'].reshape(len(rows), sample.parameters.patches, 2).sum(axis=1) % 2
    np.testing.assert_array_equal(parity, yokes)
    return arrays


def aggregate(root, baseline, distance, sample, configuration):
    directory = root / f'd{distance}'
    shots, size = configuration['shots_per_distance'], configuration['shard_size']
    chunks, manifests = [], {}
    for start in range(0, shots, size):
        stop = min(start + size, shots)
        shard = directory / 'shards' / f'{start:06d}-{stop:06d}'
        chunks.append(verified_shard(shard, sample, np.arange(start, stop), configuration, distance))
        manifests[str(shard.relative_to(directory))] = sha256_file(shard / 'manifest.json')
    arrays = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    np.testing.assert_array_equal(arrays['row_ids'], np.arange(shots))
    cached, metadata = load_baseline(sample, baseline / f'd{distance}')
    for target, source in (('mwpm_reference', 'reference'), ('gap_prediction', 'gap_prediction'),
                           ('mpp_prediction', 'mpp_prediction')):
        np.testing.assert_array_equal(arrays[target], cached[source][:shots])
    result = summarize_accuracy(arrays)
    result['parameters'] = sample.parameters.to_json()
    with atomic_replacement(directory / 'predictions.npz') as temporary:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    write_json_atomic(directory / 'results.json', result)
    write_json_atomic(directory / 'manifest.json', {
        'schema': 'correlated-uf-sharded/1', 'configuration': CORRELATED_UF_CONFIGURATION,
        'sample_identities': dict(sample.identities), 'baseline': metadata,
        'provenance': configuration['provenance'], 'shard_manifests': manifests,
        'created_utc': utc_now(),
        'artifacts': {name: sha256_file(directory / name) for name in ('predictions.npz', 'results.json')},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--distances', type=int, nargs='+', default=[7, 9, 11, 13, 15])
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=2500)
    parser.add_argument('--p', type=float, required=True)
    parser.add_argument('--rounds-per-distance', type=int, required=True)
    args = parser.parse_args()
    if min(args.shots, args.workers, args.shard_size, args.rounds_per_distance) < 1:
        parser.error('counts and the round multiplier must be positive')
    if len(set(args.distances)) != len(args.distances) or min(args.distances) < 1 or not 0 < args.p < 1:
        parser.error('distances must be distinct and positive; p must lie in (0, 1)')
    root, baseline = args.out.resolve(), args.baseline_root.resolve()
    if not root.is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('--out must be under $TMPDIR')
    samples, baseline_manifests = {}, {}
    # Check every distance before launching work. In particular, do not infer
    # rounds from a historical run that might have used the old fixed 12 rounds.
    for distance in args.distances:
        path = baseline / f'd{distance}'
        sample = SampleSet.load(path / 'sample')
        parameters = sample.parameters
        if (parameters.distance != distance or parameters.rounds != args.rounds_per_distance * distance
                or parameters.p != args.p or parameters.noise != 'si1000'
                or parameters.patches != 6 or parameters.yokes != 2 or parameters.style != 'cz'):
            raise ValueError(f'Baseline settings do not match the requested experiment: {path}')
        if sample.shots < args.shots:
            raise ValueError(f'Not enough saved shots in {path}')
        samples[distance] = sample
        baseline_manifests[str(distance)] = sha256_file(path / 'manifest.json')
    configuration = {
        'configuration': CORRELATED_UF_CONFIGURATION, 'distances': args.distances,
        'shots_per_distance': args.shots, 'shard_size': args.shard_size,
        'p': args.p, 'rounds_per_distance': args.rounds_per_distance,
        'parameters': {str(d): sample.parameters.to_json() for d, sample in samples.items()},
        'baseline_root': str(baseline), 'baseline_manifests': baseline_manifests,
        'provenance': {'code_commit': git_commit(), 'source_sha256': source_hashes(SOURCES),
                       'versions': package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')),
                       'recipe_sha256': sha256_file(__file__)},
    }
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'run.json').exists():
        if json.loads((root / 'run.json').read_text()) != configuration:
            raise ValueError('Existing run settings or sources differ; choose a new output directory')
    else:
        if any(root.iterdir()):
            raise ValueError('Output must be empty or an existing compatible run')
        write_json_atomic(root / 'run.json', configuration)
        shutil.copy2(__file__, root / 'recipe.py')
        for source in SOURCES:
            destination = root / 'source_snapshot' / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / source, destination)
    (root / 'logs').mkdir(exist_ok=True)
    tasks = []
    for distance in args.distances:
        (root / f'd{distance}' / 'shards').mkdir(parents=True, exist_ok=True)
        print(f'd={distance}, rounds={samples[distance].parameters.rounds}: {args.shots} paired shots', flush=True)
        for start in range(0, args.shots, args.shard_size):
            stop = min(start + args.shard_size, args.shots)
            shard = root / f'd{distance}' / 'shards' / f'{start:06d}-{stop:06d}'
            if (shard / 'manifest.json').exists():
                verified_shard(shard, samples[distance], np.arange(start, stop), configuration, distance)
            else:
                tasks.append((distance, start, stop))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(decode_shard, root, baseline, d, a, b): (d, a, b)
                   for d, a, b in sorted(tasks, key=lambda row: (row[1], -row[0]))}
        for future in as_completed(futures):
            future.result()
            d, a, b = futures[future]
            print(f'd={d}: rows {a}:{b} ready', flush=True)
    summary = {str(d): aggregate(root, baseline, d, samples[d], configuration) for d in args.distances}
    write_json_atomic(root / 'summary.json', summary)
    write_json_atomic(root / 'completion.json', {
        'created_utc': utc_now(), 'elapsed_seconds': time.perf_counter() - started,
        'workers': args.workers, 'run_sha256': sha256_file(root / 'run.json'),
        'summary_sha256': sha256_file(root / 'summary.json'),
        'distance_manifests': {str(d): sha256_file(root / f'd{d}/manifest.json') for d in args.distances},
    })
    print(f'Completed and verified: {root}', flush=True)


if __name__ == '__main__':
    main()
