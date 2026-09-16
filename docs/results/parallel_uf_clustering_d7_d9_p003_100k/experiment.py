"""Paired clustering experiment; native code receives no truth or MWPM outputs."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import sinter
import stim
from scipy.stats import binomtest

from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._provenance import packed_sample_hash, sha256_file
from yoked.hierarchical._record import load_record

HERE = Path(__file__).resolve().parent
VARIANTS = ('correlated_uf', 'uf_forest_cost', 'uf_bridges',
            'frontier_uf', 'frontier_forest_cost', 'frontier_bridges')
FIELDS = ('prediction', 'cost', 'size', 'epochs', 'edge_evaluations', 'frontier_visits',
          'max_cluster', 'max_frontier', 'tree_depth', 'growth_time', 'bridges',
          'eligible_bridges', 'improving_proposals', 'tree_messages', 'max_parallel_pairs')
PROTOCOL = dict(
    first_pass='Unchanged production UF growth and peeling; its physical correction supplies all correlation weights.',
    growth='Every active cluster has per-frontier-edge rate 2**(-ceil(log2(max(1,frontier_edge_count)))).',
    weight_precision='Original double-precision log-odds weights; no weight quantization.',
    extension_rounds=2, remaining_growth_cap_nats=0.5, improvement_epsilon=1e-9,
    bridge_policy='Score each eligible intercluster edge by its exact forest-cost change; accept only mutually best negative-cost proposals. Each round pairs disjoint clusters.',
    correction='Forest-cost controls and bridge variants use minimum-cost tree correction with free parity at each boundary vertex.',
    ties='Growth: existing anchored 1e-12 tie batches and ascending edge IDs. Bridges: (cost change,edge ID). Tree DP: parity zero on exact ties.',
    truth_access='No truth or MWPM predictions enter native decoding, proposal selection, or stopping.',
    hardware='Operation counts and dependency proxies only; event heap is a software simulator, not an FPGA implementation.',
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'clustering.so'
    command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp', '-ffp-contract=off',
               str(HERE / 'kernel.cc'), '-o', str(target)]
    subprocess.run(command, check=True)
    return target, command


def ptr(array):
    return ctypes.c_void_p(array.ctypes.data)


class Native:
    def __init__(self, library, graph, rules):
        self.graph = graph
        self.lib = ctypes.CDLL(str(library))
        self.lib.uf_create.restype = ctypes.c_void_p
        self.lib.uf_create.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p] * 2
        self.lib.uf_destroy.argtypes = [ctypes.c_void_p]
        self.lib.uf_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.uf_decode.restype = ctypes.c_int
        ends = np.asarray(graph.endpoints, dtype=np.int32)
        weights = np.asarray([e[2] for e in graph.edges], dtype=np.float64)
        masks = np.asarray([e[3] for e in graph.edges], dtype=np.uint32)
        rule_ends = np.asarray([(s, t) for s, t, _ in rules], dtype=np.int32)
        rule_weights = np.asarray([w for _, _, w in rules], dtype=np.float64)
        self.handle = self.lib.uf_create(graph.num_detectors, len(graph.adjacency), len(graph.edges),
                                         ptr(ends), ptr(weights), ptr(masks), len(rules), ptr(rule_ends), ptr(rule_weights))

    def close(self):
        if self.handle:
            self.lib.uf_destroy(self.handle)
            self.handle = None

    def decode(self, packed, threads=1, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        assert packed.ndim == 2 and packed.shape[1] == (self.graph.num_detectors + 7) // 8
        out = np.empty((len(packed), len(VARIANTS), len(FIELDS)), dtype=np.float64)
        flags = np.empty((len(packed), len(self.graph.edges)), dtype=np.uint16) if audit else None
        failures = self.lib.uf_decode(self.handle, len(packed), packed.shape[1], ptr(packed), ptr(out),
                                      threads, ptr(flags) if audit else None)
        if failures:
            raise RuntimeError(f'{failures} native shots failed syndrome, forest, or cost validation')
        return out, flags


def masks(bits):
    return np.asarray(bits, dtype=np.uint16) @ (1 << np.arange(bits.shape[1], dtype=np.uint16))


def unpack_results(out):
    arrays = dict(predictions=out[:, :, 0].astype(np.uint16), costs=out[:, :, 1],
                  sizes=out[:, :, 2].astype(np.uint32), tree_depth=out[:, :, 8].astype(np.uint32))
    for col in (3, 4, 5, 6, 7):
        arrays[FIELDS[col]] = out[:, [0, 3], col].astype(np.uint32)
    arrays['growth_time'] = out[:, [0, 3], 9]
    for col in range(10, 15):
        arrays[FIELDS[col]] = out[:, [2, 5], col].astype(np.uint32)
    return arrays


def summary(arrays, distance):
    n = len(arrays['actual'])
    failures = arrays['predictions'] != arrays['actual'][:, None]
    base = failures[:, 0]
    convert = lambda p: float(sinter.shot_error_rate_to_piece_error_rate(p, pieces=6*4*distance, values=8))
    rows = []
    for k, name in enumerate(VARIANTS):
        fail = failures[:, k]
        count = int(fail.sum())
        repaired = int(np.count_nonzero(base & ~fail))
        regressed = int(np.count_nonzero(~base & fail))
        delta = (repaired - regressed) / n
        se = np.sqrt(max(0., (repaired + regressed) / n - delta**2) / n)
        ci = binomtest(count, n).proportion_ci(method='wilson')
        rows.append(dict(variant=name, failures=count, shot_failure_rate=count/n, normalized_ler=convert(count/n),
                         normalized_ler_ci95=[convert(ci.low), convert(ci.high)], repaired=repaired, regressed=regressed,
                         failure_rate_reduction=delta, paired_reduction_ci95=[delta-1.96*se, delta+1.96*se],
                         mcnemar_p=binomtest(repaired, repaired+regressed).pvalue if repaired+regressed else 1.,
                         logical_changes=int(np.count_nonzero(arrays['predictions'][:, k] != arrays['predictions'][:, 0]))))
    matching_count = int(np.count_nonzero(arrays['matching'] != arrays['actual']))
    hardware = {}
    for key in ('epochs', 'edge_evaluations', 'frontier_visits', 'max_cluster', 'max_frontier',
                'tree_depth', 'bridges', 'eligible_bridges', 'improving_proposals', 'tree_messages', 'max_parallel_pairs'):
        values = arrays[key]
        hardware[key] = dict(mean=values.mean(axis=0).tolist(), p95=np.quantile(values, .95, axis=0).tolist(),
                             p99=np.quantile(values, .99, axis=0).tolist(), maximum=values.max(axis=0).tolist())
    return dict(distance=distance, shots=n, variants=list(VARIANTS), rows=rows,
                correlated_mwpm=dict(failures=matching_count, normalized_ler=convert(matching_count/n)),
                hardware_proxies=hardware,
                intervals='Wilson marginal LER intervals; paired normal intervals on raw failure-rate reduction. Exploratory comparisons, no multiplicity adjustment.')


def run(args, library, command, distance):
    output = args.output / f'd{distance}'
    output.mkdir(parents=True, exist_ok=True)
    record_dir = args.record_root / f'd{distance}' / 'evaluation'
    loaded = load_record(record_dir)
    record = loaded.record
    run = loaded.manifest['baselines']['run']
    source = Path(run['directory'])
    packed = np.load(source / 'detectors_packed.npy', mmap_mode='r')
    actual_packed = np.load(source / 'actual_observables_packed.npy', mmap_mode='r')
    assert packed_sample_hash(packed, actual_packed) == run['payload_sha256']
    for file, key in [('model.dem', 'dem_sha256'), ('circuit.stim', 'circuit_sha256')]:
        assert sha256_file(source / file) == run[key]
    np.testing.assert_array_equal(record.rows, np.arange(record.shots))
    np.testing.assert_array_equal(np.unpackbits(actual_packed, axis=1, count=12, bitorder='little'), record.actual)
    assert 0 < args.shots <= record.shots
    baseline = masks(record.baselines['joint_correlated_uf'])[:args.shots]
    actual = masks(record.actual)[:args.shots]
    matching = masks(record.baselines['joint_correlated_mwpm'])[:args.shots]
    dem = stim.DetectorErrorModel.from_file(source / 'model.dem')
    graph = DecodingGraph.from_dem(dem)
    native = Native(library, graph, correlation_rules_from_dem(graph, dem))
    request = dict(protocol=PROTOCOL, variants=list(VARIANTS), fields=list(FIELDS), distance=distance, shots=args.shots,
                   parameters=dict(loaded.manifest['parameters']),
                   input_run={key: run[key] for key in ('directory', 'payload_sha256', 'dem_sha256', 'circuit_sha256')},
                   record_manifest_sha256=sha256_file(record_dir / 'manifest.json'),
                   sources={p.name: sha256_file(p) for p in (HERE/'kernel.cc', HERE/'experiment.py')},
                   library_sha256=sha256_file(library), compile_command=command, threads=args.threads,
                   chunk_size=args.chunk_size, versions=dict(numpy=np.__version__, stim=stim.__version__))
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    request['identity'] = identity
    request_path = output / 'request.json'
    if request_path.exists():
        assert json.loads(request_path.read_text()) == request, 'Changed request: use a new output directory'
    else:
        write_json(request_path, request)
    chunks = output / 'chunks'
    chunks.mkdir(exist_ok=True)
    parts = []
    start = time.monotonic()
    for first in range(0, args.shots, args.chunk_size):
        last = min(args.shots, first+args.chunk_size)
        path = chunks / f'{first:06d}.npz'
        if path.exists():
            saved = json.loads(path.with_suffix('.json').read_text())
            assert saved == dict(identity=identity, sha256=sha256_file(path))
            with np.load(path) as data:
                part = dict(data)
        else:
            out, _ = native.decode(packed[first:last], args.threads)
            part = unpack_results(out)
            np.testing.assert_array_equal(part['predictions'][:, 0], baseline[first:last])
            np.savez_compressed(path, **part)
            write_json(path.with_suffix('.json'), dict(identity=identity, sha256=sha256_file(path)))
        parts.append(part)
        print(f'd={distance}: {last}/{args.shots}, {time.monotonic()-start:.1f}s; baseline verified', flush=True)
    native.close()
    arrays = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    arrays.update(actual=actual, matching=matching)
    np.testing.assert_array_equal(arrays['predictions'][:, 0], baseline)
    np.savez_compressed(output / 'results.npz', **arrays)
    report = summary(arrays, distance)
    report['validation'] = dict(baseline_matches=args.shots, syndrome_checks=6*args.shots,
                                 truth_passed_to_native=False, results_sha256=sha256_file(output/'results.npz'),
                                 elapsed_seconds=time.monotonic()-start)
    assert request['sources'] == {p.name: sha256_file(p) for p in (HERE/'kernel.cc', HERE/'experiment.py')}
    write_json(output / 'summary.json', report)
    print(json.dumps(report['rows']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--record-root', type=Path, default=Path(os.environ['TMPDIR'])/'hier-three-decoders-d7-d9-p003-100k-3MxG5S')
    parser.add_argument('--distances', nargs='+', type=int, choices=(7, 9), default=[7, 9])
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=48)
    parser.add_argument('--chunk-size', type=int, default=1024)
    args = parser.parse_args()
    if args.threads < 1 or args.chunk_size < 1:
        parser.error('Threads and chunk size must be positive')
    library, command = build(args.build_dir)
    for distance in args.distances:
        run(args, library, command, distance)


if __name__ == '__main__':
    main()
