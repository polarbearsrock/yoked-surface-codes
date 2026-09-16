"""Eight-pass correlation-feedback UF on the existing stratified diagnostic sample.

This offline probe changes no production decoder. Every pass restarts UF on the
original syndrome. Correlation weights are rebuilt from the original priors and
the immediately preceding physical correction, without accumulated discounts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import sinter
import stim

from yoked.decoders import CorrelatedUnionFindDecoder, DecodingGraph, UnionFindDecoder
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.hierarchical._collect import packed_sample_hash


STATE = None
PASSES = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bitmask(bits):
    return sum(int(b) << k for k, b in enumerate(bits))


def canonical_partition(growth):
    roots = np.fromiter((growth.find(v) for v in range(len(growth.parent))), dtype=np.int32)
    smallest = np.full(len(roots), len(roots), dtype=np.int32)
    np.minimum.at(smallest, roots, np.arange(len(roots), dtype=np.int32))
    return smallest[roots]


def decode_one(index):
    graph, rules, packed, baseline, native_correlated = STATE
    syndrome = np.unpackbits(packed[index], count=graph.num_detectors, bitorder='little').astype(bool)
    base_weights = [e[2] for e in graph.edges]
    endpoints = np.array(graph.endpoints)
    decoder = UnionFindDecoder(graph)
    seen = {}
    predictions, costs, edge_changes, partition_changes, changes_from_second = [], [], [], [], []
    seen_at, repeated_at, period = 0, 0, 0
    previous_edges = None
    previous_partition = first_partition = second_partition = None
    for step in range(1, PASSES + 1):
        correction, growth = decoder._decode_state(syndrome)
        selected = tuple(sorted(correction.selected_edges))
        # Validate the entire detector response, not just its logical prediction.
        response = np.zeros(len(graph.adjacency), dtype=np.bool_)
        chosen = np.array(selected, dtype=np.int64)
        np.logical_xor.at(response, endpoints[chosen, 0], True)
        np.logical_xor.at(response, endpoints[chosen, 1], True)
        np.testing.assert_array_equal(response[:graph.num_detectors], syndrome)
        label = 0
        for edge in selected:
            label ^= graph.edges[edge][3]
        assert label == correction.observable_mask
        predictions.append(label)
        costs.append(math.fsum(base_weights[e] for e in selected))
        edge_changes.append(0 if previous_edges is None else len(set(previous_edges) ^ set(selected)))
        partition = canonical_partition(growth)
        partition_changes.append(False if previous_partition is None else not np.array_equal(partition, previous_partition))
        if step == 1:
            first_partition = partition
            changes_from_second.append(False)  # set after pass two
        elif step == 2:
            second_partition = partition
            changes_from_second[0] = not np.array_equal(first_partition, second_partition)
            changes_from_second.append(False)
            assert label == baseline[index]
            if index % 128 == 0:
                reference = native_correlated._decode(syndrome)
                assert selected == tuple(sorted(reference.selected_edges))
        else:
            changes_from_second.append(not np.array_equal(partition, second_partition))
        if selected in seen and period == 0:
            seen_at, repeated_at = seen[selected], step
            period = step - seen_at
        seen.setdefault(selected, step)
        previous_edges, previous_partition = selected, partition
        if step < PASSES:
            adjusted = apply_correlation_rules(base_weights, rules, selected)
            weights = base_weights if adjusted is None else adjusted
            decoder = UnionFindDecoder(DecodingGraph(
                graph.num_detectors, graph.num_observables,
                [(u, v, weights[e], mask) for e, (u, v, _, mask) in enumerate(graph.edges)],
            ))
    if period:
        # Repeating a correction fixes every subsequent update of this memoryless map.
        for step in range(repeated_at, PASSES):
            assert predictions[step] == predictions[step - period]
    return index, dict(predictions=predictions, base_costs=costs, edge_changes=edge_changes,
                       partition_changes=partition_changes, changes_from_second=changes_from_second,
                       first_seen_pass=seen_at, first_repeat_pass=repeated_at, cycle_period=period)


def summarize(selection, actual, baseline, arrays):
    strata = np.array(selection['strata'])
    populations = np.array(selection['population_counts'])
    weights = populations / populations.sum()
    failures = arrays['predictions'] != actual[:, None]
    np.testing.assert_array_equal(failures[:, 1], baseline != actual)
    difference = failures[:, 1:2].astype(float) - failures
    estimates = np.zeros(PASSES)
    differences = np.zeros(PASSES)
    interval_draws = np.zeros((20000, PASSES))
    partition_rates = np.zeros(PASSES)
    rng = np.random.default_rng(2026091813)
    weighted_periods = {str(k): 0. for k in np.unique(arrays['cycle_period'])}
    for code, weight in enumerate(weights):
        selected = np.flatnonzero(strata == code)
        estimates += weight * failures[selected].mean(axis=0)
        differences += weight * difference[selected].mean(axis=0)
        partition_rates += weight * arrays['changes_from_second'][selected].mean(axis=0)
        draws = rng.multinomial(len(selected), np.full(len(selected), 1 / len(selected)), size=20000)
        interval_draws += weight * draws @ difference[selected] / len(selected)
        for period in weighted_periods:
            weighted_periods[period] += float(weight * np.mean(arrays['cycle_period'][selected] == int(period)))
    expected = (populations[1] + populations[3]) / populations.sum()
    np.testing.assert_allclose(estimates[1], expected)
    uf_only = strata == 1
    wrong = baseline != actual
    right = ~wrong
    rows = []
    for j in range(PASSES):
        rows.append(dict(
            pass_number=j + 1, estimated_shot_failure_probability=float(estimates[j]),
            estimated_normalized_ler=float(sinter.shot_error_rate_to_piece_error_rate(
                estimates[j], pieces=6 * 4 * selection['distance'], values=8)),
            shot_failure_reduction_vs_pass2=float(differences[j]),
            reduction_ci95=np.quantile(interval_draws[:, j], [.025, .975]).tolist(),
            estimated_partition_change_fraction_vs_pass2=float(partition_rates[j]),
            repairs_by_stratum=[int(np.count_nonzero((strata == code) & wrong & ~failures[:, j])) for code in range(4)],
            regressions_by_stratum=[int(np.count_nonzero((strata == code) & right & failures[:, j])) for code in range(4)],
            logical_changes_by_stratum=[int(np.count_nonzero((strata == code) & (arrays['predictions'][:, j] != baseline))) for code in range(4)],
        ))
    return dict(
        distance=selection['distance'], sample_shots=len(strata), strata=selection['stratum_names'],
        populations=selection['population_counts'], rows=rows,
        estimated_cycle_period_frequencies=weighted_periods,
        cycle_period_counts=dict(Counter(map(str, arrays['cycle_period']))),
        first_repeat_pass_counts=dict(Counter(map(str, arrays['first_repeat_pass']))),
        uf_only_failure_diagnostics=dict(
            shots=int(uf_only.sum()),
            any_correct_prediction_in_passes_3_to_8=int(np.count_nonzero(uf_only & np.any(~failures[:, 2:], axis=1))),
            stable_wrong_by_pass8=int(np.count_nonzero(uf_only & (arrays['cycle_period'] == 1) & failures[:, -1])),
            nontrivial_cycle_by_pass8=int(np.count_nonzero(uf_only & (arrays['cycle_period'] > 1))),
            no_repeated_correction_by_pass8=int(np.count_nonzero(uf_only & (arrays['cycle_period'] == 0))),
        ),
        bootstrap='20,000 paired stratified resamples; intervals describe the fixed saved population and may be too narrow for rare transitions',
        native_matching_failures_in_full_population=int(populations[2] + populations[3]),
    )


def main():
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=int, choices=[7, 9], required=True)
    parser.add_argument('--cohort-root', type=Path, default=Path('docs/results/physical_fault_ablation_d7_d9'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=48)
    args = parser.parse_args()
    cohort = args.cohort_root / f'd{args.distance}'
    selection = json.loads((cohort / 'selection.json').read_text())
    source = Path(selection['source'])
    packed = np.load(source / 'detectors_packed.npy', mmap_mode='r')
    actual_packed = np.load(source / 'actual_observables_packed.npy', mmap_mode='r')
    assert packed_sample_hash(packed, actual_packed) == selection['hashes']['payload_sha256']
    assert sha(source / 'model.dem') == selection['hashes']['dem_sha256']
    labels = dict(np.load(cohort / 'baseline_labels.npz'))
    indices = selection['selected_rows']
    np.testing.assert_array_equal(np.unpackbits(actual_packed[indices], axis=1, count=12, bitorder='little'), labels['actual'])
    actual = np.array([bitmask(row) for row in labels['actual']], dtype=np.uint16)
    baseline = np.array([bitmask(row) for row in labels['uf']], dtype=np.uint16)
    matching = np.array([bitmask(row) for row in labels['mwpm']], dtype=np.uint16)
    np.testing.assert_array_equal((baseline != actual).astype(int) + 2 * (matching != actual), selection['strata'])
    dem = stim.DetectorErrorModel.from_file(source / 'model.dem')
    graph = DecodingGraph.from_dem(dem)
    rules = index_rules_by_source(graph, correlation_rules_from_dem(graph, dem))
    STATE = graph, rules, packed[indices], baseline, CorrelatedUnionFindDecoder.from_dem(dem)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        protocol='Eight fresh UF passes on the unchanged syndrome; reset weights to original priors before conditioning on the previous correction.',
        sampling='Existing 256 shots in each of four baseline outcome strata; all selection rules frozen before this probe.',
        passes=PASSES, damping=None, accumulated_discounts=False, first_two_passes='exact existing correlated UF',
        candidate_choice='Report each fixed pass count; no truth-based selection or minimum across LERs.',
        original_selection=selection, analysis_sha256=sha(__file__), workers=args.workers,
        source_hashes={str(p): sha(p) for p in Path('src/yoked/decoders').glob('*.py') if not p.name.endswith('_test.py')},
        versions=dict(stim=stim.__version__, numpy=np.__version__),
    )
    (args.output_dir / 'request.json').write_text(json.dumps(metadata, indent=2) + '\n')
    results = [None] * len(indices)
    start = time.monotonic()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for done, (index, result) in enumerate(pool.imap_unordered(decode_one, range(len(indices)), chunksize=1), 1):
            results[index] = result
            if done % 128 == 0:
                print(f'd={args.distance}: {done}/{len(indices)}, {time.monotonic() - start:.1f}s', flush=True)
    arrays = {key: np.array([r[key] for r in results]) for key in results[0]}
    np.savez_compressed(args.output_dir / 'results.npz', actual=actual, baseline=baseline, matching=matching, **arrays)
    report = summarize(selection, actual, baseline, arrays)
    report['validation'] = dict(all_pass2_predictions_match_saved=True, every_correction_satisfies_syndrome=True,
                                selected_masks_recomputed=True, production_correlated_edge_checks=8,
                                exact_repeat_detection_on_full_physical_edge_sets=True,
                                results_sha256=sha(args.output_dir / 'results.npz'),
                                elapsed_seconds=time.monotonic() - start)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in report['rows']:
        print(f'pass {row["pass_number"]}: p_fail={row["estimated_shot_failure_probability"]:.6f}, '
              f'LER={row["estimated_normalized_ler"]:.9g}, change={row["shot_failure_reduction_vs_pass2"]:.6f}, '
              f'CI={row["reduction_ci95"]}', flush=True)
    print(json.dumps(report['uf_only_failure_diagnostics']), flush=True)
    print('weighted cycle periods', report['estimated_cycle_period_frequencies'], flush=True)


if __name__ == '__main__':
    main()
