"""Independent checks of BP5 followed by one or two production UF passes."""
from __future__ import annotations

import itertools
import numpy as np
import stim

from common import Native, pilot, pilot_run, pilot_verify, setup
from yoked.decoders import CorrelatedUnionFindDecoder, DecodingGraph, UnionFindDecoder
from yoked.decoders._correlations import correlation_rules_from_dem


def check_physical(native, dem, syndrome):
    q, weights = native.weights(syndrome, 5)
    pq, pw = pilot.python_bp(native.model, syndrome, 5)
    np.testing.assert_allclose(q, pq, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(weights, pw, rtol=2e-8, atol=2e-9)
    graph = native.graph
    weighted = DecodingGraph(graph.num_detectors, graph.num_observables,
        [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
    decoders = [UnionFindDecoder(weighted), CorrelatedUnionFindDecoder(weighted,
        correlation_rules=correlation_rules_from_dem(graph, dem))]
    packed = np.packbits(syndrome.astype(np.uint8), bitorder='little')[None]
    out, flags = native.decode(packed, audit=True)
    for k, decoder in enumerate(decoders):
        correction = decoder._decode(syndrome)
        np.testing.assert_array_equal(sorted(correction.forest_edges),
            np.flatnonzero(flags[0] & (1 << (2*k))))
        np.testing.assert_array_equal(sorted(correction.selected_edges),
            np.flatnonzero(flags[0] & (1 << (2*k+1))))
        assert correction.observable_mask == int(out[0, k, 0])
    return dict(physical_forests_checked=2, physical_corrections_checked=2,
        posterior_max_absolute_difference=float(np.max(np.abs(q-pq))))


def synthetic(library):
    result = dict(frozen_pilot_checks=pilot_verify.synthetic(library))
    dem = stim.DetectorErrorModel('''
        error(0.03) D0 D1 ^ D1 D2
        error(0.01) D0 D1
        error(0.02) D1 D2
        error(0.04) D0 L0
        error(0.04) D2 L0
    ''')
    graph, model, native = setup(dem, library)
    syndromes = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.uint8)
    checks = [check_physical(native, dem, s) for s in syndromes]
    packed = np.packbits(syndromes, axis=1, bitorder='little')
    a, af = native.decode(packed, 1, audit=True)
    b, bf = native.decode(packed, 4, audit=True)
    np.testing.assert_array_equal(a[:, :, :9], b[:, :, :9])
    np.testing.assert_array_equal(af, bf)
    native.close()
    result.update(composition_physical_checks=checks, thread_invariance_shots=len(syndromes))
    return result


def real(native, dem, sample, previous, library, threads, distance):
    rows = np.sort(np.random.default_rng(2026092090+distance).choice(sample.shots, 4, replace=False))
    packed = sample.detectors_packed[rows]
    a, af = native.decode(packed, 1, audit=True)
    b, bf = native.decode(packed, threads, audit=True)
    np.testing.assert_array_equal(a[:, :, :9], b[:, :, :9])
    np.testing.assert_array_equal(af, bf)
    reference = pilot.Native(library, native.graph,
        correlation_rules_from_dem(native.graph, dem), native.model)
    old, _ = reference.decode(packed)
    np.testing.assert_array_equal(a[:, 0, :9], old[:, 3, :9])
    np.testing.assert_array_equal(old[:, 1, 0].astype(np.uint16), previous['correlated_uf'][rows])
    reference.close()
    checks = []
    for row, s in zip(rows, np.unpackbits(packed, axis=1, count=sample.num_detectors, bitorder='little')):
        checks.append(dict(row=int(row), **check_physical(native, dem, s)))
    pilot_run.check_yokes(a[:, :, 0].astype(np.uint16), packed, sample.num_detectors)
    return dict(rows=rows.tolist(), physical_checks=checks, native_thread_counts=[1, threads],
        thread_results_identical_except_timing=True, saved_correlated_uf_predictions_reproduced=len(rows),
        frozen_bp5_predictions_and_diagnostics_reproduced=len(rows))
