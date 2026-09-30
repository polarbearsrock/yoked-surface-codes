"""Join saved outcomes and locate missing logical alternatives at fixed weights.

Analysis only: truth selects failure cohorts and evaluates answers. Neither
truth nor MWPM output enters UF growth, reweighting, or bridge selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pymatching
import scipy.sparse
from scipy.sparse.csgraph import connected_components
import stim

from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.hierarchical._matching_gaps import _CheckMatrixGraph
from yoked.hierarchical._provenance import packed_sample_hash, sha256_file

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
PRIOR = RESULTS / 'correlated_uf_mwpm_failure_analysis_d7_d9'
FROZEN = RESULTS / 'parallel_uf_clustering_d7_d9_p003_100k'
EXTENDED = RESULTS / 'parallel_uf_clustering_d11_d13_p003_100k'


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = helper('previous_failure_analysis', PRIOR / 'analyze.py')
experiment = helper('frozen_clustering_experiment', FROZEN / 'experiment.py')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def population(arrays):
    actual, predictions, matching = arrays['actual'], arrays['predictions'], arrays['matching']
    uf_bad = predictions[:, 0] != actual
    mwpm_bad = matching != actual
    groups = {
        'both_correct': ~uf_bad & ~mwpm_bad,
        'UF_only_failure': uf_bad & ~mwpm_bad,
        'MWPM_only_failure': ~uf_bad & mwpm_bad,
        'both_fail': uf_bad & mwpm_bad,
    }
    result = {'shots': len(actual), 'cohorts': {}, 'logical_errors': {}}
    for name, group in groups.items():
        result['cohorts'][name] = dict(
            shots=int(group.sum()),
            failures={variant: int(np.count_nonzero(group & (predictions[:, k] != actual)))
                      for k, variant in enumerate(experiment.VARIANTS)},
        )
    for name, pred in [('correlated_uf', predictions[:, 0]),
                       ('frontier_bridges', predictions[:, 5])]:
        error = pred ^ actual
        target = (error != 0) & ~mwpm_bad
        counts = Counter(int(x).bit_count() for x in error[target])
        pair_counts = Counter()
        sectors = Counter()
        parity_violations = 0
        for value in error[target]:
            bits = [k for k in range(12) if (int(value) >> k) & 1]
            sector = {k % 2 for k in bits}
            sectors['XZ'[next(iter(sector))] if len(sector) == 1 else 'both'] += 1
            parity_violations += any(sum(k % 2 == b for k in bits) % 2 for b in (0, 1))
            if len(bits) == 2:
                pair_counts[f'{"XZ"[bits[0] % 2]}:{bits[0]//2},{bits[1]//2}'] += 1
        assert parity_violations == 0
        result['logical_errors'][name] = dict(
            decoder_only_failures=int(target.sum()), wrong_bit_counts=dict(counts),
            sectors=dict(sectors), patch_pairs=dict(pair_counts),
            yoke_parity_violations=parity_violations,
        )
    return result, np.flatnonzero(groups['UF_only_failure'])


def correction(graph, matrix, syndrome, edges):
    response = np.asarray(matrix.check_matrix[:, edges].sum(axis=1)).ravel().astype(np.int64) % 2
    np.testing.assert_array_equal(response, syndrome)
    mask = 0
    for e in edges:
        mask ^= graph.edges[e][3]
    return mask


def certificate(graph, endpoints, forest, wanted):
    """Restore every edge internal to a forest component, then test L(ker H)."""
    n = len(graph.adjacency)
    u, v = endpoints[forest].T
    adjacency = scipy.sparse.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()
    _, owner = connected_components(adjacency, directed=False)
    allowed = np.flatnonzero(owner[endpoints[:, 0]] == owner[endpoints[:, 1]])
    forest_basis = previous.logical_basis(graph, forest)
    partition_basis = previous.logical_basis(graph, allowed)
    forest_ok = previous.contains(forest_basis, wanted)
    partition_ok = previous.contains(partition_basis, wanted)
    assert not forest_ok or partition_ok
    return dict(forest_allows_truth=forest_ok, partition_allows_truth=partition_ok,
                forest_logical_rank=len(forest_basis), partition_logical_rank=len(partition_basis),
                forest_edges=len(forest), partition_edges=len(allowed))


def summarize(rows):
    variants = {}
    for name in ('correlated_uf', 'frontier_uf', 'frontier_bridges'):
        failed = [r for r in rows if not r['variants'][name]['correct']]
        variants[name] = dict(
            correct=len(rows)-len(failed), failed=len(failed),
            failures_excluded_by_partition=sum(not r['variants'][name]['partition_allows_truth'] for r in failed),
            failures_excluded_by_forest=sum(not r['variants'][name]['forest_allows_truth'] for r in failed),
            failures_with_same_weight_MWPM_correct=sum(r['same_weight_MWPM_correct'] for r in failed),
        )
    return dict(shots=len(rows), variants=variants,
                same_weight_MWPM_correct=sum(r['same_weight_MWPM_correct'] for r in rows),
                same_weight_MWPM_strictly_cheaper=sum(r['UF_minus_MWPM_cost'] > 1e-6 for r in rows),
                median_UF_minus_MWPM_cost=float(np.median([r['UF_minus_MWPM_cost'] for r in rows])),
                old_rows_cross_checked=sum(r['old_row_cross_checked'] for r in rows),
                independent_forest_cost_checks=sum(r['independent_forest_cost_checks'] for r in rows))


def run_distance(distance, output, library, threads):
    start = time.monotonic()
    directory = (FROZEN if distance < 10 else EXTENDED) / f'd{distance}'
    out = output / f'd{distance}'
    out.mkdir(parents=True, exist_ok=True)
    request = json.loads((directory / 'request.json').read_text())
    source = request['input_run'] if distance < 10 else request['sample']
    sample_dir = Path(source['directory'])
    with np.load(directory / 'results.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    counts, targets = population(arrays)
    write(out / 'population.json', counts)
    seed = 20260915 + distance
    old = {}
    if distance < 10:
        ids = json.loads((PRIOR / f'd{distance}/selection.json').read_text())['sample']
        old = {r['shot']: r for r in json.loads((PRIOR / f'd{distance}/rows.json').read_text())}
    else:
        ids = sorted(np.random.default_rng(seed).choice(targets, size=128, replace=False).tolist())
    assert len(ids) == 128 and len(set(ids)) == 128 and set(ids) <= set(targets)
    write(out / 'selection.json', dict(distance=distance, seed=seed, shots=ids,
          parent_cohort='correlated UF fails; correlated MWPM succeeds', parent_size=len(targets),
          reused_prior_selection=distance < 10))

    packed = np.load(sample_dir / 'detectors_packed.npy', mmap_mode='r')
    actual_packed = np.load(sample_dir / 'actual_observables_packed.npy', mmap_mode='r')
    assert packed_sample_hash(packed, actual_packed) == source['payload_sha256']
    for filename, key in [('model.dem', 'dem_sha256'), ('circuit.stim', 'circuit_sha256')]:
        assert sha256_file(sample_dir / filename) == source[key]
    actual = experiment.masks(np.unpackbits(actual_packed, axis=1, count=12, bitorder='little'))
    np.testing.assert_array_equal(actual, arrays['actual'])
    dem = stim.DetectorErrorModel.from_file(sample_dir / 'model.dem')
    graph = DecodingGraph.from_dem(dem)
    matrix = _CheckMatrixGraph(graph)
    endpoints = np.asarray(graph.endpoints)
    flat_rules = correlation_rules_from_dem(graph, dem)
    rules = index_rules_by_source(graph, flat_rules)
    native = experiment.Native(library, graph, flat_rules)
    try:
        decoded, flags = native.decode(packed[ids], threads=threads, audit=True)
    finally:
        native.close()
    np.testing.assert_array_equal(decoded[:, :, 0].astype(np.uint16), arrays['predictions'][ids])
    np.testing.assert_allclose(decoded[:, :, 1], arrays['costs'][ids], rtol=1e-12, atol=1e-9)
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    full_basis = previous.logical_basis(graph, range(len(graph.edges)))
    assert len(full_basis) == 10
    rows = []
    for k, shot in enumerate(ids):
        syndrome = np.unpackbits(packed[shot], count=graph.num_detectors, bitorder='little')
        native_m = matrix.edge_ids_of(matching.decode_to_edges_array(syndrome, enable_correlations=True))
        assert correction(graph, matrix, syndrome, native_m) == int(arrays['matching'][shot]) == int(actual[shot])
        first = np.flatnonzero(flags[k] & 1).tolist()
        correction(graph, matrix, syndrome, first)
        adjusted = apply_correlation_rules(matrix.weights, rules, first)
        weights = matrix.weights if adjusted is None else np.asarray(adjusted)
        same_m = matrix.edge_ids_of(matrix.matcher(weights).decode_to_edges_array(syndrome))
        same_mask = correction(graph, matrix, syndrome, same_m)
        same_cost = math.fsum(weights[e] for e in same_m)
        row = dict(shot=shot, actual=int(actual[shot]), same_weight_MWPM_prediction=same_mask,
                   same_weight_MWPM_correct=same_mask == int(actual[shot]),
                   UF_minus_MWPM_cost=float(decoded[k, 0, 1]-same_cost),
                   variants={}, old_row_cross_checked=shot in old,
                   independent_forest_cost_checks=0)
        assert row['UF_minus_MWPM_cost'] >= -1e-6
        for name, col, forest_bit, selected_bit, optimum_col in (
            ('correlated_uf', 0, 1, 3, 1),
            ('frontier_uf', 3, 2, 6, 4),
            ('frontier_bridges', 5, 10, 8, 5),
        ):
            forest = np.flatnonzero(flags[k] & (1 << forest_bit)).tolist()
            selected = np.flatnonzero(flags[k] & (1 << selected_bit)).tolist()
            assert set(selected) <= set(forest)
            prediction = correction(graph, matrix, syndrome, selected)
            assert prediction == int(decoded[k, col, 0])
            assert math.isclose(math.fsum(weights[e] for e in selected), decoded[k, col, 1],
                                rel_tol=1e-12, abs_tol=1e-9)
            wanted = prediction ^ int(actual[shot])
            assert previous.contains(full_basis, wanted)
            row['variants'][name] = dict(prediction=prediction, correct=wanted == 0,
                                        **certificate(graph, endpoints, forest, wanted))
            if k < 4:
                restricted = pymatching.Matching.from_check_matrix(
                    matrix.check_matrix[:, forest], weights=weights[forest],
                    faults_matrix=matrix.faults_matrix[:, forest], merge_strategy='disallow')
                edges = matrix.edge_ids_of(restricted.decode_to_edges_array(syndrome))
                correction(graph, matrix, syndrome, edges)
                assert math.isclose(math.fsum(weights[e] for e in edges), decoded[k, optimum_col, 1],
                                    rel_tol=1e-10, abs_tol=1e-7)
                row['independent_forest_cost_checks'] += 1
        if shot in old:
            before = old[shot]
            assert same_mask == before['predictions']['UM']
            assert row['variants']['correlated_uf']['prediction'] == before['predictions']['UU']
            assert row['variants']['correlated_uf']['partition_allows_truth'] == before['partition_allows_truth']
            assert math.isclose(same_cost, before['same_U_weights_cost_MWPM'], rel_tol=1e-12, abs_tol=1e-9)
            row['MWPM_evidence_UF_correct'] = not before['wrong']['MU']
        rows.append(row)
        if len(rows) % 16 == 0:
            write(out / 'rows.json', rows)
            print(f'd={distance}: {len(rows)}/128, {time.monotonic()-start:.1f}s', flush=True)
    result = summarize(rows)
    write(out / 'summary.json', result)
    sources = [Path(__file__).resolve(), PRIOR / 'analyze.py', FROZEN / 'experiment.py',
               FROZEN / 'kernel.cc', directory / 'request.json', directory / 'results.npz']
    sources += [HERE.parents[2] / 'src/yoked' / relative for relative in (
        'decoders/_graph.py', 'decoders/_correlations.py', 'hierarchical/_matching_gaps.py')]
    write(out / 'verification.json', dict(
        source=source, hashes={str(p): sha256_file(p) for p in sources},
        library_sha256=sha256_file(library), versions=dict(numpy=np.__version__, stim=stim.__version__,
                                                         pymatching=pymatching.__version__),
        native_variants_reproduced=6*len(ids), native_correlated_MWPM_reproduced=len(ids),
        full_graph_logical_rank=len(full_basis), input_payload_verified=True,
        every_extracted_correction_syndrome_verified=True, elapsed_seconds=time.monotonic()-start))
    return distance, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library, command = experiment.build(args.work_dir / 'build')
    write(args.output / 'protocol.json', dict(
        distances=[7, 9, 11, 13], p=0.003, patches=6, yokes=2, rounds='4d',
        population_shots_per_distance=100000, selected_UF_only_failures_per_distance=128,
        selection='Uniform without replacement. Reuse d7/d9 selections; seed 20260915+d for d11/d13.',
        interventions='Hold original UF first-pass weights fixed; run MWPM or frozen UF growth variants.',
        certificate='Test true logical label on each forest and on all original edges internal to its components.',
        limitations='Conditional cohorts do not estimate new decoder LER. Logical alternatives do not uniquely identify a physical history.',
        compile_command=command))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(run_distance, d, args.output, library, args.threads) for d in (7, 9, 11, 13)]
        summaries = dict(job.result() for job in jobs)
    write(args.output / 'summary.json', summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
