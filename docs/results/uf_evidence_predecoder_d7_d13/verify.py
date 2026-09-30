"""Independent BP and unchanged-UF checks for the evidence experiment."""
from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import os
from pathlib import Path
import time

import numpy as np
import stim

from evidence import (DAMPING, FIELDS, VARIANTS, FaultModel, Native, build,
                      python_bp, setup, write_json)
from yoked.decoders import DecodingGraph, UnionFindDecoder
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)


def exact_marginals(model, syndrome):
    weights, states = [], []
    for state in itertools.product((0, 1), repeat=model.nf):
        state = np.asarray(state)
        parity = np.bincount(model.detectors, weights=state[model.incidence_faults],
                             minlength=model.nd).astype(int) % 2
        if np.array_equal(parity, syndrome):
            weights.append(np.prod(np.where(state, model.probabilities, 1-model.probabilities)))
            states.append(state)
    if not weights:
        raise ValueError('Impossible syndrome')
    return np.asarray(weights) @ np.asarray(states) / sum(weights)


def check_physical(native, dem, syndrome):
    graph, model = native.graph, native.model
    packed = np.packbits(syndrome.astype(np.uint8), bitorder='little')[None]
    out, flags = native.decode(packed, audit=True)
    out, flags = out[0], flags[0]
    first = UnionFindDecoder(graph)._decode(syndrome)
    original = np.array([e[2] for e in graph.edges])
    adjusted = apply_correlation_rules(original,
        index_rules_by_source(graph, correlation_rules_from_dem(graph, dem)), first.selected_edges)
    weight_sets = [original, original if adjusted is None else adjusted,
                   model.project(model.probabilities)]
    max_error = 0.
    for iterations in (5, 20):
        q, weights = native.weights(syndrome, iterations)
        expected_q, expected_weights = python_bp(model, syndrome, iterations)
        np.testing.assert_allclose(q, expected_q, rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-8, atol=2e-9)
        max_error = max(max_error, float(np.max(np.abs(q-expected_q))))
        weight_sets.append(weights)
    for k, weights in enumerate(weight_sets):
        adjusted_graph = DecodingGraph(graph.num_detectors, graph.num_observables,
            [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
        correction = UnionFindDecoder(adjusted_graph)._decode(syndrome)
        np.testing.assert_array_equal(sorted(correction.forest_edges),
                                       np.flatnonzero(flags & (1 << (2*k))))
        np.testing.assert_array_equal(sorted(correction.selected_edges),
                                       np.flatnonzero(flags & (1 << (2*k+1))))
        assert correction.observable_mask == int(out[k, 0])
    return dict(physical_forests_checked=len(VARIANTS), physical_corrections_checked=len(VARIANTS),
                posterior_max_absolute_difference=max_error)


def synthetic(library):
    dem = stim.DetectorErrorModel('''
        error(0.12) D0 L0
        error(0.08) D0 D1
        error(0.15) D1 D2
        error(0.07) D2 L0
    ''')
    graph, model, native = setup(dem, library)
    counts = dict(exact_tree_posteriors=0, python_bp_comparisons=0, physical_corrections=0,
                  high_degree_checks=0)
    for bits in itertools.product((0, 1), repeat=3):
        syndrome = np.array(bits, dtype=np.uint8)
        posterior, _ = native.weights(syndrome, 12, damping=1.)
        np.testing.assert_allclose(posterior, exact_marginals(model, syndrome), atol=1e-12, rtol=1e-12)
        counts['exact_tree_posteriors'] += 1
        for steps in (0, 1, 5, 20):
            q, w = native.weights(syndrome, steps)
            expected_q, expected_w = python_bp(model, syndrome, steps)
            np.testing.assert_allclose(q, expected_q, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(w, expected_w, atol=1e-12, rtol=1e-12)
            counts['python_bp_comparisons'] += 1
        check_physical(native, dem, syndrome)
        counts['physical_corrections'] += len(VARIANTS)
    native.close()
    # Symmetric priors generate exactly zero messages. Do not divide by them.
    symmetric = replace(model, probabilities=np.array([.5, .5, .2, .3]))
    native = Native(library, graph, [], symmetric)
    for bits in itertools.product((0, 1), repeat=3):
        syndrome = np.array(bits, dtype=np.uint8)
        q, _ = native.weights(syndrome, 12, damping=1.)
        np.testing.assert_allclose(q, exact_marginals(symmetric, syndrome), atol=1e-12)
        counts['exact_tree_posteriors'] += 1
    native.close()
    # A large parity check exercises the yoke-like high-degree update and
    # underflow. Parallel boundary edges are allowed by DecodingGraph.
    n = 1024
    graph = DecodingGraph(1, 1, [(0, None, 2., 0)]*n)
    model = FaultModel(np.full(n, .2), np.arange(n+1, dtype=np.int32),
                       np.zeros(n, dtype=np.int32), np.arange(n+1, dtype=np.int32),
                       np.arange(n, dtype=np.int32), 1, n, n)
    native = Native(library, graph, [], model)
    for bit in (0, 1):
        q, w = native.weights(np.array([bit]), 5)
        eq, ew = python_bp(model, np.array([bit]), 5)
        np.testing.assert_allclose(q, eq, atol=1e-12)
        np.testing.assert_allclose(w, ew, atol=1e-12)
        counts['high_degree_checks'] += 1
    native.close()
    # Correlated faults must remain a single variable, including cancellation.
    correlated = stim.DetectorErrorModel('''
        error(0.03) D0 D1 ^ D1 D2
        error(0.01) D0 D1
        error(0.02) D1 D2
        error(0.04) D0
        error(0.04) D2
    ''')
    graph, model, native = setup(correlated, library)
    assert any(np.array_equal(model.detectors[model.fault_offsets[f]:model.fault_offsets[f+1]], [0, 2])
               and model.edge_offsets[f+1]-model.edge_offsets[f] == 2 for f in range(model.nf))
    for bits in itertools.product((0, 1), repeat=3):
        check_physical(native, correlated, np.array(bits, dtype=np.uint8))
        counts['physical_corrections'] += len(VARIANTS)
    native.close()
    packed = np.packbits(np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.uint8),
                         axis=1, bitorder='little')
    graph, model, native = setup(correlated, library)
    a, af = native.decode(packed, 1, audit=True)
    b, bf = native.decode(packed, 4, audit=True)
    np.testing.assert_array_equal(a[:, :, :9], b[:, :, :9])
    np.testing.assert_array_equal(af, bf)
    native.close()
    counts['thread_invariance_shots'] = len(packed)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    library, command = build(args.work_dir/'build')
    counts = synthetic(library)
    result = dict(status='passed', checks=counts, compile_command=command,
                  elapsed_seconds=time.monotonic()-start)
    write_json(args.work_dir/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
