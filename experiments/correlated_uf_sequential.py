#!/usr/bin/env python3
"""Resume a verified UF sweep with parallel shards and sequential distances.

Only scheduling changes. Decoder sources, sample rows, shard boundaries, and
the saved recipe's verification/aggregation functions remain the same.
Completed shards are reused; an interrupted shard is decoded again in full.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(os.environ['DANTE_REPO']) / 'src'))

import numpy as np

from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._correlated_uf_experiment import SOURCES
from yoked.hierarchical._provenance import (
    package_versions, sha256_file, source_hashes, utc_now, write_json_atomic,
)


def run_in_order(tasks, workers, decode, started, completed):
    """Yield each distance only after all its shards succeed.

    Yielding before opening the next pool lets the caller aggregate and save a
    complete distance before any work on the next distance is submitted.
    """
    for distance, rows in tasks.items():
        started(distance, len(rows))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(decode, distance, start, stop): (start, stop)
                       for start, stop in rows}
            for future in as_completed(futures):
                future.result()
                completed(distance, *futures[future])
        yield distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--retained', type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root, control, retained = args.run.resolve(), args.control.resolve(), args.retained.resolve()
    if not root.is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('--run must be under $TMPDIR')
    control.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)

    def status(state, **details):
        value = {'state': state, 'workers': args.workers, 'updated_utc': utc_now(), **details}
        for directory in (control, retained):
            write_json_atomic(directory / 'status.json', value)

    try:
        status('validating_restart')
        configuration = json.loads((root / 'run.json').read_text())
        expected = configuration['provenance']
        if expected['source_sha256'] != source_hashes(SOURCES):
            raise ValueError('Decoder sources changed since the original run')
        if expected['versions'] != package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')):
            raise ValueError('Package versions changed since the original run')
        if expected['recipe_sha256'] != sha256_file(root / 'recipe.py'):
            raise ValueError('Original saved recipe hash mismatch')
        spec = importlib.util.spec_from_file_location('original_uf_recipe', root / 'recipe.py')
        recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipe)

        # Keep the original scientific identity intact. Record the replacement
        # scheduler separately, so old and new shards share the same provenance.
        execution = {'workers': args.workers, 'distance_order': sorted(configuration['distances']),
                     'run_sha256': sha256_file(root / 'run.json'),
                     'scheduler_sha256': sha256_file(__file__)}
        record = root / 'sequential_execution.json'
        if record.exists() and json.loads(record.read_text()) != execution:
            raise ValueError('An incompatible sequential execution record already exists')
        write_json_atomic(record, execution)
        if Path(__file__).resolve() != root / 'sequential_recipe.py':
            shutil.copy2(__file__, root / 'sequential_recipe.py')

        baseline = Path(configuration['baseline_root'])
        shots, size = configuration['shots_per_distance'], configuration['shard_size']
        samples, tasks = {}, {}
        for distance in execution['distance_order']:
            sample = SampleSet.load(baseline / f'd{distance}' / 'sample')
            if sample.parameters.to_json() != configuration['parameters'][str(distance)]:
                raise ValueError('Circuit settings changed since the original run')
            if sha256_file(baseline / f'd{distance}/manifest.json') != configuration['baseline_manifests'][str(distance)]:
                raise ValueError('Saved MWPM baseline changed since the original run')
            samples[distance], tasks[distance] = sample, []
            for start in range(0, shots, size):
                stop = min(start + size, shots)
                shard = root / f'd{distance}/shards/{start:06d}-{stop:06d}'
                if (shard / 'manifest.json').exists():
                    recipe.verified_shard(shard, sample, np.arange(start, stop), configuration, distance)
                else:
                    tasks[distance].append((start, stop))
            print(f'd={distance}: reusing {shots - sum(b-a for a,b in tasks[distance])} saved shots', flush=True)

        started_at = time.perf_counter()
        summary = {}

        def started(distance, count):
            print(f'Starting d={distance}: {count} remaining shards, up to {args.workers} workers', flush=True)
            status('running', active_distance=distance, distance_order=execution['distance_order'],
                   completed_distances=list(summary))

        def completed(distance, start, stop):
            print(f'd={distance}: rows {start}:{stop} ready', flush=True)

        decode = lambda d, a, b: recipe.decode_shard(root, baseline, d, a, b)
        for distance in run_in_order(tasks, args.workers, decode, started, completed):
            summary[str(distance)] = recipe.aggregate(root, baseline, distance, samples[distance], configuration)
            write_json_atomic(root / 'summary.json', summary)
            # Publish each distance immediately. The next distance starts only
            # after this copy finishes, giving usable results during the sweep.
            destination = retained / 'distances' / f'd{distance}'
            destination.mkdir(parents=True, exist_ok=True)
            for name in ('predictions.npz', 'results.json', 'manifest.json'):
                shutil.copy2(root / f'd{distance}' / name, destination / name)
            print(f'Completed d={distance}: {shots} paired shots verified and retained', flush=True)

        write_json_atomic(root / 'completion.json', {
            'created_utc': utc_now(), 'elapsed_seconds_this_schedule': time.perf_counter() - started_at,
            'workers': args.workers, 'distance_order': execution['distance_order'],
            'run_sha256': sha256_file(root / 'run.json'), 'summary_sha256': sha256_file(root / 'summary.json'),
            'sequential_execution_sha256': sha256_file(record),
            'distance_manifests': {str(d): sha256_file(root / f'd{d}/manifest.json')
                                   for d in execution['distance_order']},
        })
        status('retaining_results', completed_distances=list(summary))
        shutil.copytree(root, retained / 'run', dirs_exist_ok=True)
        status('complete', completed_distances=list(summary), results=str(retained / 'run'))
        print(f'Completed and retained: {retained / "run"}', flush=True)
    except Exception as error:
        traceback.print_exc()
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
