"""Shared configuration, native adapters, and numerical checks for the sweep."""
from __future__ import annotations
import ctypes
import itertools
import json
import os
from pathlib import Path
import sys
import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'yoked_bp5_weighted_d7'))
import weighted
shared = weighted.shared
sha, write_json, build = weighted.sha, weighted.write_json, weighted.build
FIELDS, DETERMINISTIC = weighted.FIELDS, weighted.DETERMINISTIC
NAMES = ('correlated_mwpm_complementary_gap', 'correlated_mwpm_cluster_gap',
         'correlated_uf_cluster_gap', 'bp5_weighted_uf_cluster_gap')
LABELS = ('cMWPM + complementary gap', 'cMWPM + cluster gap',
          'cUF + cluster gap', 'BP5 + weighted UF + cluster gap')


class Native(weighted.Native):
    def __init__(self, library, graph, rules, model):
        super().__init__(library, graph, rules, model)
        self.lib.sweep_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.sweep_decode.restype = ctypes.c_int

    def decode(self, packed, threads=32, audit=False, variant=-1, checked=True):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if variant not in (-1, 0, 1) or not 1 <= threads <= 32 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid native decode parameters')
        nv = 2 if variant == -1 else 1
        out = np.zeros((len(packed), nv, len(FIELDS)))
        flags = np.zeros((len(packed), nv, len(self.graph.edges)), dtype=np.uint8) if audit else None
        costs = np.zeros((len(packed), nv, len(self.graph.edges))) if audit else None
        team = ctypes.c_int()
        ptr = shared.evidence.ptr
        failures = self.lib.sweep_decode(self.gap_handle, len(packed), packed.shape[1], ptr(packed),
            ptr(out), threads, variant, ptr(flags) if audit else None,
            ptr(costs) if audit else None, ctypes.byref(team))
        if failures or team.value != threads:
            raise RuntimeError(f'Native failures={failures}; OpenMP requested/actual={threads}/{team.value}')
        if checked:
            if not np.isfinite(out).all():
                raise ValueError('Non-finite native output')
            for k, which in enumerate((0,1) if variant == -1 else (variant,)):
                np.testing.assert_array_equal(out[:,k,5], 5 if which else 0)
                if which:
                    np.testing.assert_array_equal(out[:,k,6:8], np.broadcast_to([1,0], (len(out),2)))
        return out, flags, costs


def make_native(library, dem):
    graph = shared.DecodingGraph.from_dem(dem)
    model = shared.evidence.FaultModel.from_dem(dem, graph)
    return Native(library, graph, shared.correlation_rules_from_dem(graph, dem), model)


def audit_correlated(decoder, syndrome, out, flags, costs):
    oracle = shared.CorrelatedUnionFindDecoder(decoder.graph, correlation_rules=decoder.rules)
    expected = oracle.decode_with_growth_costs(syndrome)
    assert int(out[0]) == expected.observable_mask
    np.testing.assert_array_equal(np.flatnonzero(flags), sorted(expected.selected_edges))
    np.testing.assert_allclose(costs, expected.remaining_costs, atol=2e-9, rtol=2e-10)
    search = shared.ClusterGapUnionFindDecoder(decoder.graph)
    gap, _ = search.gaps_from_costs(expected.remaining_costs)
    np.testing.assert_allclose(out[1:3], gap, atol=1e-8, rtol=2e-10)
    gap, states = search.gaps_from_costs(costs)
    np.testing.assert_array_equal(out[1:3], gap)
    np.testing.assert_array_equal(out[3:5], states)
    return dict(maximum_remaining_cost_difference=float(np.max(np.abs(costs-expected.remaining_costs), initial=0)))


def check_cases(decoder, cases, python_rows):
    packed = np.packbits(cases, axis=1, bitorder='little')
    parallel = decoder.decode(packed, 32, True)
    serial = decoder.decode(packed, 1, True)
    np.testing.assert_array_equal(parallel[0][:,:,DETERMINISTIC], serial[0][:,:,DETERMINISTIC])
    np.testing.assert_array_equal(parallel[1], serial[1])
    np.testing.assert_array_equal(parallel[2], serial[2])
    # Compare both new dispatch paths to their exact prior native implementations.
    old_cuf = shared.Native.decode(decoder, packed, 1, True)
    old_bp = weighted.Native.decode(decoder, packed, 1, True)
    np.testing.assert_array_equal(parallel[0][:,0,DETERMINISTIC], old_cuf[0][:,0,DETERMINISTIC])
    np.testing.assert_array_equal(parallel[1][:,0].astype(bool), (old_cuf[1]&1).astype(bool))
    np.testing.assert_array_equal(parallel[2][:,0], old_cuf[2][:,0])
    np.testing.assert_array_equal(parallel[0][:,1,DETERMINISTIC], old_bp[0][:,DETERMINISTIC])
    np.testing.assert_array_equal(parallel[1][:,1], old_bp[1])
    np.testing.assert_array_equal(parallel[2][:,1], old_bp[2])
    for variant in range(2):
        solo = decoder.decode(packed, 1, True, variant=variant)
        np.testing.assert_array_equal(solo[0][:,0,DETERMINISTIC], parallel[0][:,variant,DETERMINISTIC])
        np.testing.assert_array_equal(solo[1][:,0], parallel[1][:,variant])
        np.testing.assert_array_equal(solo[2][:,0], parallel[2][:,variant])
    checks = []
    for row in python_rows:
        checks.append(dict(row=int(row),
            correlated=audit_correlated(decoder, cases[row], parallel[0][row,0], parallel[1][row,0], parallel[2][row,0]),
            bp_weighted=weighted.audit_case(decoder, cases[row], parallel[0][row,1], parallel[1][row,1], parallel[2][row,1])))
    return checks


def verify(library, patches, fresh):
    synthetic_dem = stim.DetectorErrorModel('''
        error(0.03) D0 D1 ^ D2 D3
        error(0.01) D0 D1
        error(0.02) D2 D3
        error(0.04) D0 L0
        error(0.04) D1
        error(0.04) D2 L1
        error(0.04) D3
    ''')
    decoder = make_native(library, synthetic_dem)
    cases = np.array(list(itertools.product((0,1), repeat=4)), dtype=np.uint8)
    synthetic = check_cases(decoder, cases, range(16))
    decoder.close()
    real = []
    for index in (0,len(patches)-1):
        decoder = make_native(library, patches[index].local_dem)
        cases = patches[index].local_syndromes(fresh).astype(np.uint8)
        cases[0] = 0
        real.append(dict(patch=index, checks=check_cases(decoder,cases,range(4))))
        decoder.close()
    return dict(status='passed', synthetic=synthetic, real=real,
        verified_openmp_threads=[1,32], thread_invariance_shots_per_patch=len(fresh),
        matches_previous_native_implementations=True, individual_latency_dispatch_matches=True,
        bp_iterations=5, bp_weighted_uf_passes=1, gaps_uncapped=True,
        physical_corrections_checked_every_decode=True)


def finish_outer(patches, detectors, reference, gaps):
    from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
    from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
    yoke = detectors[:, patches.yoke_detector_ids]
    sigma = frame_adjusted_syndrome(yoke, reference)
    residual = np.zeros_like(reference)
    tied = np.zeros((len(reference),2), dtype=bool)
    for sector in range(2):
        decision = mwpm_outer_log_odds_batch(gaps[:,sector::2], sigma[:,sector])
        residual[:,sector::2] = decision.patterns
        tied[:,sector] = decision.tied
    return reference ^ residual, residual, sigma, tied


class UfHierarchicalDecoder:
    """One input row or batch through the same L1, gap and L2 as collection."""
    def __init__(self, library, patches, variant):
        self.patches, self.variant = patches, variant
        self.decoders = [make_native(library,p.local_dem) for p in patches]

    def decode(self, detectors):
        detectors = np.asarray(detectors)[None]
        reference = np.zeros((1,12), dtype=bool)
        gaps = np.zeros((1,12))
        for k,(patch,decoder) in enumerate(zip(self.patches,self.decoders)):
            packed = np.packbits(patch.local_syndromes(detectors),axis=1,bitorder='little')
            output,_,_ = decoder.decode(packed,1,variant=self.variant,checked=False)
            mask = int(output[0,0,0])
            reference[0,2*k:2*k+2] = [mask&1,(mask>>1)&1]
            gaps[0,2*k:2*k+2] = output[0,0,1:3]
        return finish_outer(self.patches,detectors,reference,gaps)[0][0]

    def close(self):
        for decoder in self.decoders:
            decoder.close()
