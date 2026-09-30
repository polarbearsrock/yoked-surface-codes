"""Independent fixed-budget BP and physical-correction verification."""
from __future__ import annotations

import itertools
import numpy as np
import stim

from common import CONFIG, PILOT, Native, VARIANTS, evidence
from yoked.decoders import CorrelatedUnionFindDecoder, DecodingGraph, UnionFindDecoder


def audit_case(native, dem, syndrome, out, flags):
    graph, model = native.graph, native.model
    expected = [UnionFindDecoder(graph)._decode(syndrome),
                CorrelatedUnionFindDecoder.from_dem(dem)._decode(syndrome)]
    maximum_posterior_difference = 0.
    for iteration in (1, 2, 5):
        posterior, weights = native.weights(syndrome, iteration)
        expected_q, expected_weights = evidence.python_bp(model, syndrome, iteration)
        np.testing.assert_allclose(posterior, expected_q, atol=2e-10, rtol=2e-8)
        np.testing.assert_allclose(weights, expected_weights, atol=2e-9, rtol=2e-8)
        maximum_posterior_difference = max(maximum_posterior_difference,
                                          float(np.max(np.abs(posterior-expected_q))))
        adjusted = DecodingGraph(graph.num_detectors, graph.num_observables,
            [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
        expected.append(UnionFindDecoder(adjusted)._decode(syndrome))
    for k, correction in enumerate(expected):
        assert int(out[k, 0]) == correction.observable_mask, VARIANTS[k]
        assert out[k, 11] == [0, 0, 1, 2, 5][k]
        np.testing.assert_array_equal(np.flatnonzero(flags & (1 << k)), sorted(correction.selected_edges))
        parity = np.zeros(graph.num_detectors, dtype=np.uint8)
        mask = 0
        for e in correction.selected_edges:
            u, v, _, label = graph.edges[e]
            parity[u] ^= 1
            if v is not None:
                parity[v] ^= 1
            mask ^= label
        np.testing.assert_array_equal(parity, syndrome)
        assert mask == correction.observable_mask
    return dict(physical_corrections=len(VARIANTS), posterior_budgets=[1, 2, 5],
                maximum_posterior_difference=maximum_posterior_difference)


def verify(library, circuit, dem, native, threads):
    previous = evidence.load_module('bp_single_patch_previous_verify', PILOT / 'verify.py')
    inherited = previous.synthetic(library)
    print('Archived BP/UF exact and physical checks passed.', flush=True)
    synthetic_dem = stim.DetectorErrorModel('''
        error(0.03) D0 D1 ^ D1 D2
        error(0.01) D0 D1
        error(0.02) D1 D2
        error(0.04) D0 L0
        error(0.04) D2 L0
    ''')
    graph = DecodingGraph.from_dem(synthetic_dem)
    model = evidence.FaultModel.from_dem(synthetic_dem, graph)
    synthetic = Native(library, graph, evidence.correlation_rules_from_dem(graph, synthetic_dem), model)
    cases = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.uint8)
    packed = np.packbits(cases, axis=1, bitorder='little')
    outputs, flags = synthetic.decode(packed, threads, audit=True)
    synthetic_checks = [audit_case(synthetic, synthetic_dem, s, o, f)
                        for s, o, f in zip(cases, outputs, flags)]
    synthetic.close()
    print('Synthetic fixed-budget BP and UF corrections passed.', flush=True)

    packed, _ = circuit.compile_detector_sampler(seed=CONFIG['validation_seed']).sample(
        shots=CONFIG['validation_shots'], separate_observables=True, bit_packed=True)
    packed[0] = 0
    outputs, flags = native.decode(packed, threads, audit=True)
    serial, serial_f = native.decode(packed, 1, audit=True)
    deterministic_columns = [*range(9), 11]
    np.testing.assert_array_equal(outputs[:, :, deterministic_columns], serial[:, :, deterministic_columns])
    np.testing.assert_array_equal(flags, serial_f)
    rows = range(4)
    cases = np.unpackbits(packed, axis=1, count=graph_num_detectors(native), bitorder='little')
    real_checks = []
    for row in sorted(rows):
        result = audit_case(native, dem, cases[row], outputs[row], flags[row])
        real_checks.append(dict(row=row, **result))
        print(f'Independent single-patch check passed: validation row {row}.', flush=True)
    return dict(status='passed', inherited_checks=inherited, synthetic_checks=synthetic_checks,
                real_checks=real_checks, thread_invariance_shots=len(packed),
                verified_openmp_threads=[1, threads], actuals_passed_to_decoder=False)


def graph_num_detectors(native):
    return native.graph.num_detectors
