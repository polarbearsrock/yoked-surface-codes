#!/usr/bin/env python3
"""Paired SI1000 experiment using the existing, unchanged decoders.

Source env/workspace.sh and activate the decoder venv before running. Decoding
uses independent subprocesses over disjoint rows of one saved sample per distance.
Completed shards can be reused on restart, but are checked before aggregation.
The round schedule must be explicit: --rounds-per-distance 4 implements 4d
rounds, while --rounds 12 reproduces the earlier fixed-round experiments.
Pass --p 0.003 for the 0.3% physical-noise comparison.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

# Avoid creating a BLAS thread pool in every decoder subprocess.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
REPO = Path(os.environ['DANTE_REPO'])
sys.path.insert(0, str(REPO / 'src'))

import numpy as np
from scipy.stats import binomtest
import sinter

from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._mpp import load_mpp_native
from yoked.hierarchical._mpp_experiment import SOURCES, compare_accuracy
from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
from yoked.hierarchical._provenance import (
    atomic_replacement, git_commit, package_versions, sha256_file, source_hashes, utc_now, write_json_atomic,
)


def decode_shard(root, distance, start, stop):
    """Publish a shard only after its decoder subprocess succeeds.

    Working directories stay separate from completed shards, allowing a long
    sweep to resume after an interruption without mistaking partial output for
    a finished result. Each child owns its decoder state.
    """
    directory = root / f'd{distance}'
    shard = directory / 'shards' / f'{start:06d}-{stop:06d}'
    if (shard / 'manifest.json').exists():
        return distance, start, stop, 'reused'
    if shard.exists():
        raise ValueError(f'Unrecognized partial shard: {shard}')
    log = root / 'logs' / f'd{distance}-{start:06d}-{stop:06d}.log'
    with tempfile.TemporaryDirectory(prefix='.working-', dir=shard.parent) as temporary:
        output = Path(temporary) / 'output'
        command = [sys.executable, str(REPO / 'tools/mpp_experiment'),
                   '--sample', str(directory / 'sample'), '--rows', f'{start}:{stop}',
                   '--out', str(output), '--method', 'dijkstra', '--batch-size', '128']
        with log.open('w') as handle:
            handle.write('Command arguments: ' + json.dumps(command) + '\n')
            handle.flush()
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'd={distance}, rows {start}:{stop} failed; inspect {log}')
        output.rename(shard)
    return distance, start, stop, 'decoded'


def load_verified_shard(directory, sample, start, stop, provenance):
    """Check artifact hashes, source identity, row coverage, truth, and parity."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['source_sha256'] != provenance['source_sha256']:
        raise ValueError(f'Source changed during decoding: {directory}')
    if manifest['native_build'] != provenance['native_build']:
        raise ValueError(f'Native build changed: {directory}')
    if manifest['sample']['identities'] != dict(sample.identities):
        raise ValueError(f'Wrong sample in {directory}')
    if manifest['method'] != 'dijkstra':
        raise ValueError(f'Wrong search method in {directory}')
    for name, digest in manifest['artifacts'].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f'Artifact hash mismatch: {directory / name}')
    with np.load(directory / 'predictions.npz') as stored:
        arrays = {key: stored[key] for key in stored.files}
    rows = np.arange(start, stop)
    np.testing.assert_array_equal(arrays['row_ids'], rows)
    detectors, actual = sample.rows(rows)
    np.testing.assert_array_equal(arrays['actual'], actual)
    yokes = detectors[:, -2:]
    np.testing.assert_array_equal(arrays['sigma'], frame_adjusted_syndrome(yokes, arrays['reference']))
    for name in ('gap', 'mpp'):
        np.testing.assert_array_equal(arrays[f'{name}_prediction'], arrays['reference'] ^ arrays[f'{name}_residual'])
        parity = arrays[f'{name}_prediction'].reshape(len(rows), sample.parameters.patches, 2).sum(axis=1) % 2
        np.testing.assert_array_equal(parity, yokes)
    return arrays


def additional_statistics(results, rounds):
    """Exact marginal intervals complement the paired empirical bootstrap.

    Especially at this low physical error rate, a zero observed failure count
    must carry a nonzero upper confidence limit rather than imply perfect decoding.
    The optional normalized LER follows the repository's existing convention.
    """
    for name in ('complementary_gap', 'mpp'):
        row = results[name]
        interval = binomtest(row['block_failures'], results['shots']).proportion_ci(method='exact')
        row['block_failure_ci95_exact'] = [float(interval.low), float(interval.high)]
        row['normalized_ler'] = sinter.shot_error_rate_to_piece_error_rate(
            row['block_failure_rate'], pieces=6 * rounds, values=8)
    paired = results['paired']
    base = results['complementary_gap']['block_failure_rate']
    mpp = results['mpp']['block_failure_rate']
    paired['mpp_to_gap_rate_ratio'] = mpp / base if base else None
    paired['relative_degradation_percent'] = 100 * (mpp / base - 1) if base else None
    discordant = paired['mpp_repairs'] + paired['mpp_regressions']
    paired['mcnemar_exact_two_sided_p'] = (
        float(binomtest(paired['mpp_regressions'], discordant, 0.5).pvalue) if discordant else 1.0)
    # Preserve pairing when assessing relative degradation as well as difference.
    counts = np.array([paired[key] for key in ('both_succeed', 'mpp_repairs', 'mpp_regressions', 'both_fail')])
    draws = np.random.default_rng(43).multinomial(results['shots'], counts / results['shots'], size=10000)
    gap_counts = draws[:, 1] + draws[:, 3]
    mpp_counts = draws[:, 2] + draws[:, 3]
    usable = gap_counts > 0
    paired['ratio_zero_baseline_bootstrap_replicates'] = int((~usable).sum())
    paired['rate_ratio_ci95_paired_bootstrap'] = (
        np.percentile(mpp_counts[usable] / gap_counts[usable], [2.5, 97.5]).tolist() if usable.any() else None)


def aggregate(root, distance, shots, shard_size, provenance):
    directory = root / f'd{distance}'
    sample = SampleSet.load(directory / 'sample')
    chunks = []
    manifests = {}
    for start in range(0, shots, shard_size):
        stop = min(start + shard_size, shots)
        shard = directory / 'shards' / f'{start:06d}-{stop:06d}'
        chunks.append(load_verified_shard(shard, sample, start, stop, provenance))
        manifests[str(shard.relative_to(directory))] = sha256_file(shard / 'manifest.json')
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    np.testing.assert_array_equal(arrays['row_ids'], np.arange(shots))
    results = compare_accuracy(arrays['reference'], arrays['actual'], arrays['gap_prediction'], arrays['mpp_prediction'])
    additional_statistics(results, sample.parameters.rounds)
    results.update(parameters=sample.parameters.to_json(), sampling_seed=sample.seed,
                   reference_disagreements=0, native_weight_disagreements=0,
                   method='dijkstra', calibration='none; raw gaps and raw MPP scores')
    for name in ('gap', 'mpp'):
        results['complementary_gap' if name == 'gap' else name]['outer_tied_sectors'] = int(arrays[f'{name}_outer_tied'].sum())
    with atomic_replacement(directory / 'predictions.npz') as temporary:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    write_json_atomic(directory / 'results.json', results)
    write_json_atomic(directory / 'manifest.json', {
        'schema': 'mpp-sharded-comparison/1', 'distance': distance, 'shots': shots,
        'sample_identities': dict(sample.identities), 'shard_manifests': manifests,
        'provenance': provenance, 'created_utc': utc_now(),
        'artifacts': {name: sha256_file(directory / name) for name in ('predictions.npz', 'results.json')},
    })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--shard-size', type=int, default=5000)
    parser.add_argument('--seed-base', type=int, default=2026092300,
                        help='sampling seed is this base plus code distance')
    parser.add_argument('--distances', type=int, nargs='+', default=[7, 9])
    parser.add_argument('--p', type=float, default=0.001, help='SI1000 noise strength, e.g. 0.003 for 0.3%%')
    schedule = parser.add_mutually_exclusive_group(required=True)
    schedule.add_argument('--rounds', type=int, help='fixed rounds at every distance')
    schedule.add_argument('--rounds-per-distance', type=int, help='rounds = this multiplier * distance')
    args = parser.parse_args()
    if min(args.workers, args.shots, args.shard_size) < 1 or len(set(args.distances)) != len(args.distances):
        parser.error('counts must be positive and distances must be distinct')
    if not 0 < args.p < 1:
        parser.error('--p must lie strictly between zero and one')
    if args.seed_base < 0 or args.seed_base + max(args.distances) >= 2**64:
        parser.error('sampling seeds must fit in an unsigned 64-bit integer')
    if (args.rounds if args.rounds is not None else args.rounds_per_distance) < 1:
        parser.error('round count or multiplier must be positive')
    parameters = {
        distance: CircuitParameters(distance=distance,
                                    rounds=args.rounds if args.rounds is not None else args.rounds_per_distance * distance,
                                    p=args.p)
        for distance in args.distances
    }
    root = args.out.resolve()
    if not root.is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('--out must be under $TMPDIR')
    _, native_build = load_mpp_native()
    provenance = dict(source_sha256=source_hashes(SOURCES), native_build=native_build,
                      code_commit=git_commit(), recipe_sha256=sha256_file(__file__),
                      versions=package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')))
    configuration = dict(distances=args.distances, shots_per_distance=args.shots, shard_size=args.shard_size,
                         noise='si1000', p=args.p, patches=6, seed_base=args.seed_base,
                         round_schedule={'fixed': args.rounds} if args.rounds is not None else {'per_distance': args.rounds_per_distance},
                         rounds_by_distance={str(d): item.rounds for d, item in parameters.items()},
                         provenance=provenance)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'run.json').exists():
        if json.loads((root / 'run.json').read_text()) != configuration:
            raise ValueError('Existing run has different settings or sources; choose a new output path')
    else:
        if any(root.iterdir()):
            raise ValueError('Output directory is not an existing run or an empty directory')
        write_json_atomic(root / 'run.json', configuration)
        shutil.copyfile(__file__, root / 'recipe.py')
        for source in SOURCES:
            destination = root / 'source_snapshot' / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / source, destination)
    (root / 'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    tasks = []
    for distance in args.distances:
        expected_parameters = parameters[distance]
        directory = root / f'd{distance}'
        (directory / 'shards').mkdir(parents=True, exist_ok=True)
        sample_path = directory / 'sample'
        if not (sample_path / 'sample.json').exists():
            sample = SampleSet.sample(expected_parameters,
                                      seed=args.seed_base + distance, shots=args.shots)
            sample.save(sample_path)
        sample = SampleSet.load(sample_path)
        if (sample.shots != args.shots or sample.parameters != expected_parameters
                or sample.seed != args.seed_base + distance):
            raise ValueError(f'Unexpected sample settings: {sample_path}')
        print(f'd={distance}, rounds={sample.parameters.rounds}: saved and verified {sample.shots} shots', flush=True)
        for start in range(0, args.shots, args.shard_size):
            tasks.append((distance, start, min(start + args.shard_size, args.shots)))
    completed = {distance: 0 for distance in args.distances}
    # Alternate distances so progress covers both rather than finishing one first.
    # Start the more expensive distance first within each range to limit the
    # final wait on large-distance shards, while interleaving all distances.
    tasks.sort(key=lambda task: (task[1], -task[0]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(decode_shard, root, *task) for task in tasks]
        for future in as_completed(futures):
            distance, start, stop, status = future.result()
            completed[distance] += stop - start
            print(f'd={distance}: {completed[distance]}/{args.shots} shots {status}; '
                  f'elapsed {time.perf_counter() - started:.1f}s', flush=True)
    summaries = {str(distance): aggregate(root, distance, args.shots, args.shard_size, provenance)
                 for distance in args.distances}
    write_json_atomic(root / 'summary.json', summaries)
    write_json_atomic(root / 'completion.json', dict(
        created_utc=utc_now(), elapsed_seconds=time.perf_counter() - started, workers=args.workers,
        run_sha256=sha256_file(root / 'run.json'), summary_sha256=sha256_file(root / 'summary.json'),
        distance_manifests={str(d): sha256_file(root / f'd{d}/manifest.json') for d in args.distances}))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
