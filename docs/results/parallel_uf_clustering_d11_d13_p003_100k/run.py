"""Extend the frozen d=7/9 clustering experiment to fresh, paired d=11/13 shots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pymatching
import stim

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN = HERE.parent / 'parallel_uf_clustering_d7_d9_p003_100k'
sys.path.insert(0, str(FROZEN))
from experiment import FIELDS, PROTOCOL, VARIANTS, Native, build, masks, summary, unpack_results, write_json
from verify import REFERENCE, check_case
from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._provenance import sha256_file

FROZEN_HASHES = {
    'kernel.cc': '1a83ec7adc4b78fb23582bc8e5f4c7e857798b9b125a95ffd65a846d3c0678de',
    'experiment.py': '5e7a6decf3eaf34bc17fd549f5278a785f7723252651b28d58e86205590d6546',
    'verify.py': 'd5268ccd48a64177dd5cbbddb738e7669014282476361dc3d5fb576822ecf7c5',
}


def sources():
    paths = [Path(__file__), REFERENCE, *(FROZEN / name for name in FROZEN_HASHES)]
    for package in ('gen', 'yoked'):
        paths.extend(p for p in (REPO / 'src' / package).rglob('*.py') if not p.name.endswith('_test.py'))
    return {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(paths)}


def require_same(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Request changed: use a new work directory ({path})')
    else:
        write_json(path, value)


def matching_worker_init(sample_dir):
    # Each process owns its matching state. OpenMP is used only in the parent,
    # after this pool exits; no decoder object is shared across workers.
    global _matcher, _packed
    dem = stim.DetectorErrorModel.from_file(sample_dir / 'model.dem')
    _matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    _packed = np.load(sample_dir / 'detectors_packed.npy', mmap_mode='r')


def matching_chunk(bounds):
    first, last = bounds
    prediction = _matcher.decode_batch(_packed[first:last], bit_packed_shots=True,
                                       enable_correlations=True)
    return first, masks(prediction)


def matching_predictions(sample_dir, sample, work, workers, chunk_size, identity):
    path = work / 'matching.npz'
    if path.exists() and path.with_suffix('.json').exists():
        assert json.loads(path.with_suffix('.json').read_text()) == {
            'identity': identity, 'sha256': sha256_file(path)}
        with np.load(path) as saved:
            return saved['predictions']
    prediction = np.zeros(sample.shots, dtype=np.uint16)
    done = np.zeros(sample.shots, dtype=bool)
    jobs = [(first, min(first + chunk_size, sample.shots))
            for first in range(0, sample.shots, chunk_size)]
    start = last_print = time.monotonic()
    context = multiprocessing.get_context('spawn')
    with context.Pool(workers, initializer=matching_worker_init, initargs=(sample_dir,)) as pool:
        for first, part in pool.imap_unordered(matching_chunk, jobs):
            last = first + len(part)
            assert not done[first:last].any()
            prediction[first:last] = part
            done[first:last] = True
            if time.monotonic() - last_print >= 20 or done.all():
                print(f'd={sample.parameters.distance}: correlated MWPM {done.sum()}/{sample.shots}, '
                      f'{time.monotonic()-start:.1f}s', flush=True)
                last_print = time.monotonic()
    assert done.all()
    np.savez_compressed(path, predictions=prediction)
    write_json(path.with_suffix('.json'), dict(identity=identity, sha256=sha256_file(path)))
    return prediction


def validate(library, graph, rules, sample, matching, native, threads):
    # Chosen from row IDs before inspecting any outcomes, with a fixed seed.
    rng = np.random.default_rng(2026091600 + sample.parameters.distance)
    rows = np.sort(rng.choice(sample.shots, min(32, sample.shots), replace=False))
    packed = sample.detectors_packed[rows]
    serial, _ = native.decode(packed, 1)
    parallel, _ = native.decode(packed, threads)
    np.testing.assert_array_equal(serial, parallel)
    decoder = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(sample.dem_text), enable_correlations=True)
    unpacked = np.unpackbits(packed, axis=1, count=graph.num_detectors, bitorder='little')
    expected = masks(decoder.decode_batch(unpacked, enable_correlations=True))
    np.testing.assert_array_equal(expected, matching[rows])
    start = time.monotonic()
    for k, (row, syndrome) in enumerate(zip(rows, unpacked)):
        checked = check_case(library, graph, rules, syndrome, scan_normalized=k < 2)
        np.testing.assert_array_equal(checked, serial[k])
        if (k + 1) % 8 == 0:
            print(f'd={sample.parameters.distance}: independent physical-correction checks '
                  f'{k+1}/{len(rows)}, {time.monotonic()-start:.1f}s', flush=True)
    return dict(rows=rows.tolist(), production_physical_corrections=len(rows),
                independent_tree_cost_checks=4*len(rows), independent_frontier_scans=min(2, len(rows)),
                thread_counts=[1, threads], native_outputs_bit_identical=True,
                matching_serial_unpacked_equals_parallel_packed=True,
                elapsed_seconds=time.monotonic()-start)


def check_yoke_parity(predictions, packed, nd):
    predictions = np.asarray(predictions)
    for basis in (0, 1):
        parity = np.zeros_like(predictions, dtype=np.uint8)
        for bit in range(basis, 12, 2):
            parity ^= ((predictions >> bit) & 1).astype(np.uint8)
        detector = nd - 2 + basis
        expected = ((packed[:, detector // 8] >> (detector % 8)) & 1)
        if predictions.ndim == 2:
            expected = np.broadcast_to(expected[:, None], predictions.shape)
        np.testing.assert_array_equal(parity, expected)


def run(args, library, command, distance):
    work = args.work_dir / f'd{distance}'
    work.mkdir(parents=True, exist_ok=True)
    output = args.output / f'd{distance}'
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = work / 'sample'
    parameters = CircuitParameters(distance=distance, rounds=4*distance, p=.003,
                                   patches=6, yokes=2, style='cz', noise='si1000')
    if not (sample_dir / 'sample.json').exists():
        print(f'd={distance}: generating {args.shots} shots in one Stim call', flush=True)
        SampleSet.sample(parameters, seed=42, shots=args.shots).save(sample_dir)
    sample = SampleSet.load(sample_dir)
    assert sample.parameters == parameters and sample.shots == args.shots and sample.seed == 42
    assert sample.num_observables == 12
    shutil.copy2(sample_dir / 'sample.json', output / 'sample.json')
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
    require_same(work/'request.json', request)
    write_json(output/'request.json', request)
    matching = matching_predictions(sample_dir, sample, work, args.matching_workers, args.chunk_size, identity)
    check_yoke_parity(matching, sample.detectors_packed, sample.num_detectors)
    dem = stim.DetectorErrorModel(sample.dem_text)
    graph = DecodingGraph.from_dem(dem)
    rules = correlation_rules_from_dem(graph, dem)
    print(f'd={distance}: graph has {graph.num_detectors} detectors, {len(graph.edges)} edges, '
          f'{len(rules)} correlation rules', flush=True)
    native = Native(library, graph, rules)
    try:
        verified = validate(library, graph, rules, sample, matching, native, args.threads)
        write_json(output/'verification.json', verified)
        chunks = work / 'chunks'
        chunks.mkdir(exist_ok=True)
        parts = []
        start = last_print = time.monotonic()
        for first in range(0, sample.shots, args.chunk_size):
            last = min(first + args.chunk_size, sample.shots)
            path = chunks / f'{first:06d}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                assert json.loads(path.with_suffix('.json').read_text()) == {
                    'identity': identity, 'sha256': sha256_file(path)}
                with np.load(path) as saved:
                    part = dict(saved)
            else:
                out, _ = native.decode(sample.detectors_packed[first:last], args.threads)
                part = unpack_results(out)
                check_yoke_parity(part['predictions'], sample.detectors_packed[first:last], sample.num_detectors)
                np.savez_compressed(path, **part)
                write_json(path.with_suffix('.json'), dict(identity=identity, sha256=sha256_file(path)))
            parts.append(part)
            if time.monotonic() - last_print >= 20 or last == sample.shots:
                print(f'd={distance}: all six UF variants {last}/{sample.shots}, '
                      f'{time.monotonic()-start:.1f}s', flush=True)
                last_print = time.monotonic()
        arrays = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
        actual = masks(np.unpackbits(sample.actual_packed, axis=1, count=12, bitorder='little'))
        arrays.update(actual=actual, matching=matching)
        np.savez_compressed(output/'results.npz', **arrays)
        report = summary(arrays, distance)
        report['validation'] = dict(**verified, syndrome_checks=6*sample.shots,
                                    yoke_parity_checks=7*sample.shots, truth_passed_to_native=False,
                                    results_sha256=sha256_file(output/'results.npz'),
                                    native_elapsed_seconds=time.monotonic()-start)
        report['graph'] = dict(detectors=graph.num_detectors, edges=len(graph.edges), correlation_rules=len(rules))
        assert sources() == source_hashes, 'Sources changed while the experiment was running'
        write_json(output/'summary.json', report)
        print(json.dumps(dict(distance=distance, correlated_mwpm=report['correlated_mwpm'],
                              rows=report['rows'])), flush=True)
    finally:
        native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--distances', nargs='+', type=int, choices=(11, 13), default=[11, 13])
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--matching-workers', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=1024)
    args = parser.parse_args()
    if min(args.shots, args.threads, args.matching_workers, args.chunk_size) < 1:
        parser.error('Shot, thread, worker, and chunk counts must be positive')
    scratch_root = Path(os.environ['TMPDIR']).resolve()
    if not args.work_dir.resolve().is_relative_to(scratch_root):
        parser.error('The work directory must be under TMPDIR')
    for name, expected in FROZEN_HASHES.items():
        assert sha256_file(FROZEN/name) == expected, f'Frozen source changed: {name}'
    args.output.mkdir(parents=True, exist_ok=True)
    # Confirm this generator reproduces the original d=9 circuit and model.
    prior = json.loads((FROZEN/'d9'/'request.json').read_text())['input_run']
    params = CircuitParameters(distance=9, rounds=36, p=.003, patches=6, yokes=2, style='cz', noise='si1000')
    circuit = params.circuit()
    for key, value in [('circuit_sha256', circuit), ('dem_sha256', params.dem(circuit))]:
        assert hashlib.sha256((str(value)+'\n').encode()).hexdigest() == prior[key], key
    write_json(args.output/'reference_configuration.json', dict(distance=9,
               circuit_sha256=prior['circuit_sha256'], dem_sha256=prior['dem_sha256'], reproduced=True))
    library, command = build(args.work_dir/'build')
    for distance in args.distances:
        run(args, library, command, distance)


if __name__ == '__main__':
    main()
