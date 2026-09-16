"""Independent checks for the native experiment, including brute-force forests."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import stim

from experiment import HERE, Native, build, write_json
from yoked.decoders import DecodingGraph, UnionFindDecoder
from yoked.decoders._correlations import apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source
from yoked.hierarchical._provenance import sha256_file

REFERENCE = HERE.parent/'correlated_uf_growth_diagnosis_d7_d9'/'algorithms.py'
spec = importlib.util.spec_from_file_location('independent_forest_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
minimum_cost_forest = reference.minimum_cost_forest


def scan(graph, syndrome, normalized):
    """Full-edge scan and explicit cluster labels; no heap or disjoint-set tree."""
    nd, n = graph.num_detectors, len(graph.adjacency)
    owner = np.arange(n)
    ends = np.asarray(graph.endpoints)
    u, v = ends.T
    w = np.asarray([e[2] for e in graph.edges])
    grown = np.zeros(len(w))
    forest, now = [], 0.
    while True:
        parity = np.bincount(owner[:nd], weights=syndrome, minlength=n).astype(int) % 2
        boundary = np.bincount(owner[nd:], minlength=n) > 0
        active = parity.astype(bool) & ~boundary
        crossing = owner[u] != owner[v]
        speeds = active.astype(float)
        if normalized:
            degree = np.bincount(np.concatenate([owner[u[crossing]], owner[v[crossing]]]), minlength=n)
            speeds /= np.exp2(np.ceil(np.log2(np.maximum(1, degree))))
        rate = (speeds[owner[u]] + speeds[owner[v]]) * crossing
        deadline = np.full(len(w), np.inf)
        moving = rate > 0
        deadline[moving] = now + np.maximum(0., w[moving]-grown[moving]) / rate[moving]
        deadline[crossing & (grown >= w)] = now
        when = float(deadline.min())
        if not active.any() and when != 0:
            break
        assert math.isfinite(when)
        grown = np.minimum(w, grown+rate*(when-now))
        now = when
        complete = np.flatnonzero(deadline <= now+1e-12+1e-12*abs(now))
        grown[complete] = w[complete]
        for e in complete:
            a, b = int(owner[u[e]]), int(owner[v[e]])
            if a == b:
                continue
            # Canonical labels suffice; growth depends only on frontier and parity.
            owner[owner == b] = a
            forest.append(int(e))
    return forest, owner, grown


def naive_extension(graph, syndrome, forest, owner, grown):
    """Evaluate each proposed bridge by solving the entire enlarged forest."""
    forest = list(forest)
    owner = owner.copy()
    eligible = [e for e, (u, v) in enumerate(graph.endpoints)
                if owner[u] != owner[v] and max(0., graph.edges[e][2]-grown[e]) <= .5]
    for _ in range(2):
        _, old_cost = minimum_cost_forest(graph, syndrome, forest)
        best = {}
        for e in eligible:
            u, v = graph.endpoints[e]
            a, b = owner[u], owner[v]
            if a == b:
                continue
            _, cost = minimum_cost_forest(graph, syndrome, forest+[e])
            delta = cost-old_cost
            if delta < -1e-9:
                for root in (a, b):
                    best[root] = min(best.get(root, (math.inf, -1)), (delta, e))
        accepted = [e for e in eligible if owner[graph.endpoints[e][0]] != owner[graph.endpoints[e][1]]
                    and best.get(owner[graph.endpoints[e][0]], (0, -1))[1] == e
                    and best.get(owner[graph.endpoints[e][1]], (0, -1))[1] == e]
        if not accepted:
            break
        for e in accepted:
            u, v = graph.endpoints[e]
            a, b = owner[u], owner[v]
            owner[owner == b] = a
            forest.append(e)
    return forest


def chosen(flags, bit):
    return np.flatnonzero(flags & (1 << bit)).tolist()


def check_case(library, graph, rules, syndrome, *, brute=False, scan_normalized=True):
    native = Native(library, graph, rules)
    packed = np.packbits(np.asarray(syndrome, dtype=np.uint8), bitorder='little')[None, :]
    out, flags = native.decode(packed, audit=True)
    native.close()
    out, flags = out[0], flags[0]
    first = UnionFindDecoder(graph)._decode(syndrome)
    assert sorted(first.selected_edges) == chosen(flags, 0)
    rules_by_source = index_rules_by_source(graph, rules)
    w0 = [e[2] for e in graph.edges]
    w = apply_correlation_rules(w0, rules_by_source, first.selected_edges)
    w = w0 if w is None else w
    adjusted = DecodingGraph(graph.num_detectors, graph.num_observables,
                            [(u, v, w[e], mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
    second, state = UnionFindDecoder(adjusted)._decode_state(syndrome)
    assert sorted(second.forest_edges) == chosen(flags, 1)
    assert sorted(second.selected_edges) == chosen(flags, 3)
    assert second.observable_mask == int(out[0, 0])
    if scan_normalized:
        forest, owner, grown = scan(adjusted, syndrome, True)
        assert sorted(forest) == chosen(flags, 2)
        if brute:
            extended = naive_extension(adjusted, syndrome, forest, owner, grown)
            assert sorted(extended) == chosen(flags, 10)
    if brute:
        owner = np.array([state.find(v) for v in range(len(graph.adjacency))])
        for e in range(len(graph.edges)):
            state._settle(e)
        extended = naive_extension(adjusted, syndrome, second.forest_edges, owner, state.grown)
        assert sorted(extended) == chosen(flags, 9)
    for policy in range(2):
        for extra in range(2):
            forest = chosen(flags, 9+policy if extra else 1+policy)
            variant = policy*3+1+extra
            _, cost = minimum_cost_forest(adjusted, syndrome, forest)
            np.testing.assert_allclose(out[variant, 1], cost, rtol=1e-11, atol=1e-8)
            if brute and len(forest) <= 16:
                best = math.inf
                # Exhaustively solve the parity constraints without tree recursion.
                for bits in itertools.product((0, 1), repeat=len(forest)):
                    residual = np.array(syndrome, dtype=np.uint8)
                    total = 0.
                    for e, selected in zip(forest, bits):
                        if selected:
                            u, v = adjusted.endpoints[e]
                            residual[u] ^= 1
                            if v < adjusted.num_detectors:
                                residual[v] ^= 1
                            total += w[e]
                    if not residual.any():
                        best = min(best, total)
                np.testing.assert_allclose(cost, best, atol=1e-10)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--random-cases', type=int, default=100)
    args = parser.parse_args()
    library, command = build(args.work_dir/'build')
    rng = np.random.default_rng(2026091617)
    start = time.monotonic()
    counts = dict(random_graphs=0, actual_shots=0, independent_normalized_scans=0)
    for case in range(args.random_cases):
        nd = int(rng.integers(2, 7))
        edges = [(u, u+1, float(rng.integers(0, 17))/4, int(rng.integers(0, 4))) for u in range(nd-1)]
        edges += [(u, None, float(rng.integers(0, 17))/4, int(rng.integers(0, 4))) for u in range(nd)]
        for u in range(nd):
            for v in range(u+2, nd):
                if rng.random() < .2:
                    edges.append((u, v, float(rng.integers(0, 17))/4, int(rng.integers(0, 4))))
        graph = DecodingGraph(nd, 2, edges)
        rules = [(e, (e+1)%len(edges), float(rng.integers(0, 4))/4) for e in range(len(edges)) if rng.random()<.25]
        syndrome = rng.integers(0, 2, size=nd, dtype=np.uint8)
        check_case(library, graph, rules, syndrome, brute=True)
        counts['random_graphs'] += 1
    print(f'Random graph and exhaustive correction checks passed: {counts}, {time.monotonic()-start:.1f}s', flush=True)
    for d in (7, 9):
        record = json.loads((HERE.parent/'coset_ensemble_d7_d9_p003_100k'/f'd{d}'/'request.json').read_text())['request']
        source = Path(record['input_run']['directory'])
        packed = np.load(source/'detectors_packed.npy', mmap_mode='r')
        graph = DecodingGraph.from_dem(stim.DetectorErrorModel.from_file(source/'model.dem'))
        rules = correlation_rules_from_dem(graph, stim.DetectorErrorModel.from_file(source/'model.dem'))
        rows = sorted(set([0, 1, 16, 6463, 34624, 44875, 76890, 99999] + rng.choice(len(packed), 16, replace=False).tolist()))
        native = Native(library, graph, rules)
        serial, _ = native.decode(packed[rows], 1)
        parallel, _ = native.decode(packed[rows], 4)
        np.testing.assert_array_equal(serial, parallel)
        native.close()
        for i, row in enumerate(rows):
            syndrome = np.unpackbits(packed[row], count=graph.num_detectors, bitorder='little')
            check_case(library, graph, rules, syndrome, scan_normalized=i<2)
            counts['actual_shots'] += 1
            counts['independent_normalized_scans'] += i<2
        print(f'd={d}: production edges, tree costs, and thread determinism passed', flush=True)
    result = dict(checks=counts, threads_bit_identical=True, compile_command=command,
                  elapsed_seconds=time.monotonic()-start,
                  sources={str(p): sha256_file(p) for p in (HERE/'kernel.cc', HERE/'experiment.py', Path(__file__), REFERENCE)})
    write_json(args.work_dir/'verification.json', result)


if __name__ == '__main__':
    main()
