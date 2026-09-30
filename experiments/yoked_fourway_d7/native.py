"""Native paired correlated-UF decoders and independent verification."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import stim

REPO = Path(os.environ['DANTE_REPO'])
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO/'docs/results/uf_evidence_predecoder_d7_d13'))
import evidence
from yoked.decoders import CorrelatedUnionFindDecoder, DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder

VARIANTS = ('correlated_uf_cluster_gap', 'bp5_correlated_uf_cluster_gap')
FIELDS = ('mask', 'gap_x', 'gap_z', 'states_x', 'states_z', 'bp_iterations',
          'uf_passes', 'discounted_edges', 'zero_edges', 'max_cluster',
          'evidence_seconds', 'uf_seconds', 'gap_seconds', 'correction_edges')
DETERMINISTIC = [*range(10), 13]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='dante-fourway-', dir=os.environ['TMPDIR'])
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        try:
            temporary.replace(path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Native(evidence.Native):
    def __init__(self, library, graph, rules, model):
        super().__init__(library, graph, rules, model)
        self.rules = rules
        self.lib.fourway_create.argtypes = [ctypes.c_void_p]
        self.lib.fourway_create.restype = ctypes.c_void_p
        self.lib.fourway_destroy.argtypes = [ctypes.c_void_p]
        self.lib.fourway_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.fourway_decode.restype = ctypes.c_int
        self.gap_handle = self.lib.fourway_create(self.handle)
        if not self.gap_handle:
            super().close()
            raise RuntimeError('Could not build cluster-gap topology')

    def close(self):
        if getattr(self, 'gap_handle', None):
            self.lib.fourway_destroy(self.gap_handle)
            self.gap_handle = None
        super().close()

    def decode(self, packed, threads=32, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if not 1 <= threads <= 32 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid packed array or thread count')
        out = np.zeros((len(packed), 2, len(FIELDS)), dtype=np.float64)
        flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint8) if audit else None
        costs = np.zeros((len(packed), 2, len(self.graph.edges)), dtype=np.float64) if audit else None
        team = ctypes.c_int()
        failures = self.lib.fourway_decode(self.gap_handle, len(packed), packed.shape[1],
            evidence.ptr(packed), evidence.ptr(out), threads,
            evidence.ptr(flags) if audit else None, evidence.ptr(costs) if audit else None,
            ctypes.byref(team))
        if failures or not np.isfinite(out).all() or team.value != threads:
            raise RuntimeError(f'Native failures={failures}; requested/actual OpenMP team={threads}/{team.value}')
        np.testing.assert_array_equal(out[:, :, 5], np.broadcast_to([0, 5], (len(out), 2)))
        return out, flags, costs


def make_native(library, dem):
    graph = DecodingGraph.from_dem(dem)
    model = evidence.FaultModel.from_dem(dem, graph)
    rules = correlation_rules_from_dem(graph, dem)
    return Native(library, graph, rules, model)


def audit_case(native, syndrome, out, flags, costs):
    graph = native.graph
    posterior, weights = native.weights(syndrome, 5)
    python_posterior, python_weights = evidence.python_bp(native.model, syndrome, 5)
    np.testing.assert_allclose(posterior, python_posterior, atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(weights, python_weights, atol=2e-9, rtol=2e-8)
    adjusted = DecodingGraph(graph.num_detectors, graph.num_observables,
        [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
    search = ClusterGapUnionFindDecoder(graph)
    differences = []
    for k, current in enumerate((graph, adjusted)):
        oracle = CorrelatedUnionFindDecoder(current, correlation_rules=native.rules)
        result = oracle.decode_with_growth_costs(syndrome)
        assert int(out[k, 0]) == result.observable_mask
        np.testing.assert_array_equal(np.flatnonzero(flags & (1 << k)), sorted(result.selected_edges))
        np.testing.assert_allclose(costs[k], result.remaining_costs, atol=2e-9, rtol=2e-10)
        gap, _ = search.gaps_from_costs(result.remaining_costs)
        np.testing.assert_allclose(out[k, 1:3], gap, atol=1e-8, rtol=2e-10)
        exact_gap, exact_states = search.gaps_from_costs(costs[k])
        np.testing.assert_array_equal(out[k, 1:3], exact_gap)
        np.testing.assert_array_equal(out[k, 3:5], exact_states)
        residual = np.array(syndrome, dtype=np.uint8)
        mask = 0
        for edge in np.flatnonzero(flags & (1 << k)):
            u, v, _, label = graph.edges[edge]
            residual[u] ^= 1
            if v is not None:
                residual[v] ^= 1
            mask ^= label
        assert not residual.any() and mask == result.observable_mask
        differences.append(float(np.max(np.abs(costs[k]-result.remaining_costs), initial=0)))
    return dict(maximum_bp_posterior_difference=float(np.max(np.abs(posterior-python_posterior))),
                maximum_remaining_cost_differences=differences, physical_corrections=2)


def verify(library, patches, fresh_detectors, threads=32):
    synthetic_dem = stim.DetectorErrorModel('''
        error(0.03) D0 D1 ^ D2 D3
        error(0.01) D0 D1
        error(0.02) D2 D3
        error(0.04) D0 L0
        error(0.04) D1
        error(0.04) D2 L1
        error(0.04) D3
    ''')
    synthetic = make_native(library, synthetic_dem)
    cases = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    packed = np.packbits(cases, axis=1, bitorder='little')
    out, flags, costs = synthetic.decode(packed, threads, True)
    checks = [audit_case(synthetic, s, o, f, c) for s, o, f, c in zip(cases, out, flags, costs)]
    synthetic.close()
    real = []
    for index in (0, len(patches)-1):
        native = make_native(library, patches[index].local_dem)
        local = patches[index].local_syndromes(fresh_detectors).astype(np.uint8)
        local[0] = 0
        packed = np.packbits(local, axis=1, bitorder='little')
        parallel = native.decode(packed, threads, True)
        serial = native.decode(packed, 1, True)
        np.testing.assert_array_equal(parallel[0][:, :, DETERMINISTIC], serial[0][:, :, DETERMINISTIC])
        np.testing.assert_array_equal(parallel[1], serial[1])
        np.testing.assert_array_equal(parallel[2], serial[2])
        for row in (0, 1, 2, 3):
            real.append(dict(patch=index, row=row,
                **audit_case(native, local[row], parallel[0][row], parallel[1][row], parallel[2][row])))
        native.close()
    return dict(status='passed', synthetic_syndromes=len(checks), synthetic_checks=checks,
        real_checks=real, thread_invariance_shots_per_patch=len(fresh_detectors),
        verified_openmp_threads=sorted(set([1, threads])),
        zero_syndrome_verified=True, cluster_gaps_unbounded=True,
        final_corrections_and_growth_costs_match_python=True,
        physical_syndrome_check_every_native_decode=True, truth_passed_to_decoder=False)


def build(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'fourway.so'
    compiler = '/opt/rh/gcc-toolset-14/root/usr/bin/g++'
    command = [compiler, '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp',
               '-ffp-contract=off', str(source), '-o', str(target)]
    subprocess.run(command, check=True)
    return target, command
