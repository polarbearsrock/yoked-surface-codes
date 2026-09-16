"""Repeat the frozen clustering comparison at a configurable physical error rate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / 'parallel_uf_clustering_d11_d13_p003_100k'
sys.path.insert(0, str(PREVIOUS))
import run as previous
from experiment import FIELDS, PROTOCOL, VARIANTS, Native, build, masks, summary, unpack_results, write_json
from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._provenance import sha256_file


def sources():
    return {**previous.sources(), str(Path(__file__).relative_to(REPO)): sha256_file(Path(__file__))}


def prepare_sample(work, parameters, shots):
    directory = work / f'd{parameters.distance}' / 'sample'
    if not (directory / 'sample.json').exists():
        print(f'p={parameters.p}, d={parameters.distance}: sampling {shots} shots in one call', flush=True)
        SampleSet.sample(parameters, seed=42, shots=shots).save(directory)
    sample = SampleSet.load(directory)
    assert sample.parameters == parameters and sample.shots == shots and sample.seed == 42
    assert sample.num_observables == 12
    return directory, sample


def collect(args, library, command, distance):
    parameters = CircuitParameters(distance=distance, rounds=4*distance, p=args.p,
                                   patches=6, yokes=2, style='cz', noise='si1000')
    sample_dir, sample = prepare_sample(args.work_dir, parameters, args.shots)
    work = args.work_dir / f'd{distance}'
    output = args.output / f'd{distance}'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_dir/'sample.json', output/'sample.json')
    source_hashes = sources()
    request = dict(distance=distance, shots=args.shots, parameters=parameters.to_json(),
                   protocol=PROTOCOL, variants=list(VARIANTS), fields=list(FIELDS),
                   sample=dict(directory=str(sample_dir), manifest_sha256=sha256_file(sample_dir/'sample.json'),
                               payload_sha256=sample.payload_sha256, circuit_sha256=sample.circuit_sha256,
                               dem_sha256=sample.dem_sha256),
                   sources=source_hashes, library_sha256=sha256_file(library), compile_command=command,
                   threads=args.threads, matching_workers=args.matching_workers, chunk_size=args.chunk_size,
                   versions={name: importlib.metadata.version(name)
                             for name in ('numpy', 'stim', 'pymatching', 'sinter', 'scipy')},
                   python=platform.python_version(), stim_module=stim.Circuit.__module__,
                   sampling='Seed 42; one full bit-packed detector-sampler call per distance.',
                   validation_policy='32 fixed random rows per distance; first two also use independent full-edge frontier scans.')
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    request['identity'] = identity
    previous.require_same(work/'request.json', request)
    write_json(output/'request.json', request)
    matching = previous.matching_predictions(sample_dir, sample, work, args.matching_workers,
                                             args.chunk_size, identity)
    previous.check_yoke_parity(matching, sample.detectors_packed, sample.num_detectors)
    dem = stim.DetectorErrorModel(sample.dem_text)
    graph = DecodingGraph.from_dem(dem)
    rules = correlation_rules_from_dem(graph, dem)
    print(f'p={args.p}, d={distance}: {graph.num_detectors} detectors, {len(graph.edges)} edges', flush=True)
    native = Native(library, graph, rules)
    try:
        verified = previous.validate(library, graph, rules, sample, matching, native, args.threads)
        write_json(output/'verification.json', verified)
        chunks = work/'chunks'
        chunks.mkdir(exist_ok=True)
        parts = []
        start = last_print = time.monotonic()
        for first in range(0, sample.shots, args.chunk_size):
            last = min(first+args.chunk_size, sample.shots)
            path = chunks/f'{first:06d}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                assert json.loads(path.with_suffix('.json').read_text()) == {
                    'identity': identity, 'sha256': sha256_file(path)}
                with np.load(path) as saved:
                    part = dict(saved)
            else:
                out, _ = native.decode(sample.detectors_packed[first:last], args.threads)
                part = unpack_results(out)
                previous.check_yoke_parity(part['predictions'], sample.detectors_packed[first:last], sample.num_detectors)
                np.savez_compressed(path, **part)
                write_json(path.with_suffix('.json'), dict(identity=identity, sha256=sha256_file(path)))
            parts.append(part)
            if time.monotonic()-last_print >= 20 or last == sample.shots:
                print(f'p={args.p}, d={distance}: all six UF variants {last}/{sample.shots}, '
                      f'{time.monotonic()-start:.1f}s', flush=True)
                last_print = time.monotonic()
        arrays = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        arrays.update(actual=masks(np.unpackbits(sample.actual_packed, axis=1, count=12, bitorder='little')),
                      matching=matching)
        np.savez_compressed(output/'results.npz', **arrays)
        report = summary(arrays, distance)
        report.update(parameters=parameters.to_json(),
                      graph=dict(detectors=graph.num_detectors, edges=len(graph.edges), correlation_rules=len(rules)),
                      validation=dict(**verified, syndrome_checks=6*sample.shots, yoke_parity_checks=7*sample.shots,
                                      truth_passed_to_native=False, results_sha256=sha256_file(output/'results.npz'),
                                      native_elapsed_seconds=time.monotonic()-start))
        assert sources() == source_hashes, 'Sources changed during the experiment'
        write_json(output/'summary.json', report)
        print(json.dumps(dict(p=args.p, distance=distance, correlated_mwpm=report['correlated_mwpm'],
                              rows=report['rows'])), flush=True)
    finally:
        native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--p', type=float, default=.001)
    parser.add_argument('--distances', nargs='+', type=int, choices=(7, 9, 11, 13), default=[7, 9, 11, 13])
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--matching-workers', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--prepare-only', action='store_true', help='Create and verify samples without building or decoding.')
    args = parser.parse_args()
    if min(args.shots, args.threads, args.matching_workers, args.chunk_size) < 1 or not 0 < args.p < 1:
        parser.error('Counts must be positive, and p must lie in (0,1)')
    if not args.work_dir.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('The work directory must be under TMPDIR')
    for name, expected in previous.FROZEN_HASHES.items():
        assert sha256_file(previous.FROZEN/name) == expected, f'Frozen source changed: {name}'
    if args.prepare_only:
        for distance in args.distances:
            prepare_sample(args.work_dir, CircuitParameters(distance=distance, rounds=4*distance, p=args.p), args.shots)
        return
    library, command = build(args.work_dir/'build')
    for distance in args.distances:
        collect(args, library, command, distance)


if __name__ == '__main__':
    main()
