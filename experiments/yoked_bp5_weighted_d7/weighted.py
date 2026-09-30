"""Single-pass BP5-weighted UF and independent checks of its soft output."""
from __future__ import annotations

import ctypes
import itertools
from pathlib import Path
import sys

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'yoked_fourway_d7'))
import native as shared
from yoked.decoders import UnionFindDecoder

FIELDS = shared.FIELDS
DETERMINISTIC = shared.DETERMINISTIC
sha, write_json, build = shared.sha, shared.write_json, shared.build
VARIANT = 'bp5_weighted_uf_cluster_gap'


class Native(shared.Native):
    def __init__(self, library, graph, rules, model):
        super().__init__(library, graph, rules, model)
        self.lib.bp5_weighted_decode.argtypes = self.lib.fourway_decode.argtypes
        self.lib.bp5_weighted_decode.restype = ctypes.c_int

    def decode(self, packed, threads=32, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if not 1 <= threads <= 32 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid packed array or thread count')
        out = np.zeros((len(packed), len(FIELDS)), dtype=np.float64)
        flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint8) if audit else None
        costs = np.zeros((len(packed), len(self.graph.edges)), dtype=np.float64) if audit else None
        team = ctypes.c_int()
        ptr = shared.evidence.ptr
        failures = self.lib.bp5_weighted_decode(self.gap_handle, len(packed), packed.shape[1],
            ptr(packed), ptr(out), threads, ptr(flags) if audit else None,
            ptr(costs) if audit else None, ctypes.byref(team))
        if failures or not np.isfinite(out).all() or team.value != threads:
            raise RuntimeError(f'Native failures={failures}; OpenMP requested/actual={threads}/{team.value}')
        np.testing.assert_array_equal(out[:, 5:8], np.broadcast_to([5, 1, 0], (len(out), 3)))
        return out, flags, costs


def make_native(library, dem):
    graph = shared.DecodingGraph.from_dem(dem)
    model = shared.evidence.FaultModel.from_dem(dem, graph)
    return Native(library, graph, shared.correlation_rules_from_dem(graph, dem), model)


def audit_case(native, syndrome, out, flags, costs):
    graph = native.graph
    posterior, weights = native.weights(syndrome, 5)
    py_posterior, py_weights = shared.evidence.python_bp(native.model, syndrome, 5)
    np.testing.assert_allclose(posterior, py_posterior, atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(weights, py_weights, atol=2e-9, rtol=2e-8)
    adjusted = shared.DecodingGraph(graph.num_detectors, graph.num_observables,
        [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
    result = UnionFindDecoder(adjusted).decode_with_growth_costs(syndrome)
    assert int(out[0]) == result.observable_mask
    np.testing.assert_array_equal(np.flatnonzero(flags), sorted(result.selected_edges))
    np.testing.assert_allclose(costs, result.remaining_costs, atol=2e-9, rtol=2e-10)
    search = shared.ClusterGapUnionFindDecoder(graph)
    gap, _ = search.gaps_from_costs(result.remaining_costs)
    np.testing.assert_allclose(out[1:3], gap, atol=1e-8, rtol=2e-10)
    exact_gap, exact_states = search.gaps_from_costs(costs)
    np.testing.assert_array_equal(out[1:3], exact_gap)
    np.testing.assert_array_equal(out[3:5], exact_states)
    residual = np.array(syndrome, dtype=np.uint8)
    for edge in np.flatnonzero(flags):
        u, v, _, _ = graph.edges[edge]
        residual[u] ^= 1
        if v is not None:
            residual[v] ^= 1
    assert not residual.any()
    return dict(maximum_bp_posterior_difference=float(np.max(np.abs(posterior-py_posterior))),
        maximum_remaining_cost_difference=float(np.max(np.abs(costs-result.remaining_costs), initial=0)))


def check_existing_bp5(native, packed, out, flags):
    previous, previous_flags = shared.evidence.Native.decode(native, packed, threads=1, audit=True)
    bp5 = shared.evidence.VARIANTS.index('bp5_uf')
    np.testing.assert_array_equal(out[:, 0], previous[:, bp5, 0])
    # The old audit packs two bits per variant: forest, then selected correction.
    np.testing.assert_array_equal(flags.astype(bool), (previous_flags & (1 << (2*bp5+1))).astype(bool))


def verify(library, patches, fresh_detectors):
    dem = stim.DetectorErrorModel('''
        error(0.03) D0 D1 ^ D2 D3
        error(0.01) D0 D1
        error(0.02) D2 D3
        error(0.04) D0 L0
        error(0.04) D1
        error(0.04) D2 L1
        error(0.04) D3
    ''')
    synthetic = make_native(library, dem)
    cases = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    packed = np.packbits(cases, axis=1, bitorder='little')
    out, flags, costs = synthetic.decode(packed, 32, True)
    checks = [audit_case(synthetic, s, o, f, c) for s, o, f, c in zip(cases, out, flags, costs)]
    check_existing_bp5(synthetic, packed, out, flags)
    synthetic.close()
    real = []
    for index in (0, len(patches)-1):
        decoder = make_native(library, patches[index].local_dem)
        local = patches[index].local_syndromes(fresh_detectors).astype(np.uint8)
        local[0] = 0
        packed = np.packbits(local, axis=1, bitorder='little')
        parallel = decoder.decode(packed, 32, True)
        serial = decoder.decode(packed, 1, True)
        np.testing.assert_array_equal(parallel[0][:, DETERMINISTIC], serial[0][:, DETERMINISTIC])
        np.testing.assert_array_equal(parallel[1], serial[1])
        np.testing.assert_array_equal(parallel[2], serial[2])
        check_existing_bp5(decoder, packed, parallel[0], parallel[1])
        for row in range(4):
            real.append(dict(patch=index, row=row,
                **audit_case(decoder, local[row], parallel[0][row], parallel[1][row], parallel[2][row])))
        decoder.close()
    return dict(status='passed', synthetic_syndromes=len(cases), synthetic_checks=checks,
        real_checks=real, thread_invariance_shots_per_patch=len(fresh_detectors),
        verified_openmp_threads=[1, 32], zero_syndrome_verified=True,
        existing_bp5_weighted_uf_corrections_match=True,
        final_corrections_and_growth_costs_match_python=True,
        bp_iterations=5, uf_passes=1, conditional_discounted_edges=0,
        cluster_gaps_unbounded=True, physical_syndrome_check_every_native_decode=True,
        truth_passed_to_decoder=False)
