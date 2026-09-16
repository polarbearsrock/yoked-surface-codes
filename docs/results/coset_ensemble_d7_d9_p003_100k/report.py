"""Verify ensemble checkpoints, recompute votes, and export paired results."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from yoked.hierarchical._provenance import (
    canonical_json, sha256_bytes, sha256_file, source_hashes, write_json_atomic,
)
from yoked.hierarchical._record import load_record
from yoked.hierarchical._uf_soft_analysis import compare_predictions

from run import PREFIXES, SCORES, masks


def histogram(values):
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def independently_vote(labels, scores, weighted):
    """Recompute votes from persisted scores, without calling the decoder's vote."""
    output = labels[:, 0].copy()
    for row in np.flatnonzero(np.any(labels != labels[:, :1], axis=1)):
        costs = scores[row]
        minimum = costs.min()
        if weighted:
            eligible = abs(costs - minimum) <= np.maximum(1e-12, 1e-12 * np.maximum(abs(costs), abs(minimum)))
        else:
            eligible = costs == minimum
        selected = labels[row, eligible]
        votes = Counter(map(int, selected))
        top = max(votes.values())
        output[row] = next(label for label in selected if votes[int(label)] == top)
    return output


def verify_and_export(root, out, distance, expected_shots):
    directory = root / f'd{distance}'
    request_document = json.loads((directory / 'request.json').read_text())
    request = request_document['request']
    manifest = json.loads((directory / 'manifest.json').read_text())
    identity = sha256_bytes(canonical_json(request).encode())
    assert identity == request_document['identity'] == manifest['identity']
    assert manifest['request_sha256'] == sha256_file(directory / 'request.json')
    assert manifest['candidates_sha256'] == sha256_file(directory / 'candidates.npz')
    assert source_hashes(request['source_sha256']) == request['source_sha256']
    assert request['shots'] == manifest['shots'] == expected_shots
    assert manifest['candidates_checked'] == 24 * expected_shots
    assert manifest['baseline_matches'] == expected_shots
    record_dir = Path(request['record_dir'])
    assert sha256_file(record_dir / 'manifest.json') == request['record_manifest_sha256']
    loaded = load_record(record_dir)
    params = dict(loaded.manifest['parameters'])
    assert params == request['parameters']
    assert params['distance'] == distance and params['rounds'] == 4 * distance
    assert params['p'] == .003 and params['patches'] == 6
    assert loaded.manifest['seed'] == 42
    with np.load(directory / 'candidates.npz') as saved:
        arrays = dict(saved)
    rows = arrays['rows']
    np.testing.assert_array_equal(rows, np.arange(expected_shots))
    actual = loaded.record.actual[rows]
    actual_masks = masks(actual)
    predictions = {name: loaded.record.baselines[name][rows]
                   for name in ('joint_correlated_mwpm', 'joint_correlated_uf')}
    uf_masks = masks(predictions['joint_correlated_uf'])
    np.testing.assert_array_equal(arrays['baseline'], uf_masks)
    assert arrays['candidate_masks'].shape == (expected_shots, 24)
    assert np.all(np.isfinite(arrays['candidate_weights']))
    assert np.all(arrays['candidate_weights'] >= 0)
    logical_diversity = 1 + np.sum(np.diff(np.sort(arrays['candidate_masks'], axis=1), axis=1) != 0, axis=1)
    certified = arrays['logical_rank'] == 0
    assert np.all(arrays['candidate_masks'][certified] == uf_masks[certified, None])
    for si, score in enumerate(SCORES):
        scores = arrays['candidate_sizes' if score == 'size' else 'candidate_weights']
        for ki, count in enumerate(PREFIXES):
            predicted = independently_vote(arrays['candidate_masks'][:, :count], scores[:, :count], score == 'weight')
            np.testing.assert_array_equal(predicted, arrays['predictions'][:, si, ki])
            predictions[f'ensemble_{score}_k{count}'] = (predicted[:, None] >> np.arange(12)) & 1
    methods = compare_predictions(predictions, actual, pieces=6 * 4 * distance, seed=43, replicates=10000)
    uf_fail = uf_masks != actual_masks
    paired = {}
    for name, prediction in predictions.items():
        candidate_masks = masks(prediction)
        fail = candidate_masks != actual_masks
        paired[name] = dict(
            repairs=int(np.sum(uf_fail & ~fail)), regressions=int(np.sum(~uf_fail & fail)),
            both_fail=int(np.sum(uf_fail & fail)), both_correct=int(np.sum(~uf_fail & ~fail)),
            different_logical_predictions=int(np.sum(candidate_masks != uf_masks)),
        )
    any_correct = np.any(arrays['candidate_masks'] == actual_masks[:, None], axis=1)
    diagnostics = dict(
        logical_rank_histogram=histogram(arrays['logical_rank']),
        logical_diversity_histogram=histogram(logical_diversity),
        physical_diversity_histogram=histogram(arrays['physical_diversity']),
        all_candidates_equal_baseline_shots=int(np.sum(np.all(arrays['candidate_masks'] == uf_masks[:, None], axis=1))),
        baseline_failures_with_correct_candidate=int(np.sum(uf_fail & any_correct)),
        baseline_successes_without_correct_candidate=int(np.sum(~uf_fail & ~any_correct)),
        oracle_failures_within_sampled_candidates=int(np.sum(~any_correct)),
        candidate_size_improved_shots=int(np.sum(arrays['candidate_sizes'].min(axis=1) < arrays['baseline_sizes'])),
        candidate_weight_improved_shots=int(np.sum(arrays['candidate_weights'].min(axis=1) < arrays['baseline_weights'] - 1e-10)),
        mean_minimum_size_change=float(np.mean(arrays['candidate_sizes'].min(axis=1).astype(float) - arrays['baseline_sizes'])),
        mean_minimum_weight_change=float(np.mean(arrays['candidate_weights'].min(axis=1) - arrays['baseline_weights'])),
        rank_positive_rows=rows[~certified].tolist(),
        logical_diversity_rows=rows[logical_diversity > 1].tolist(),
    )
    runtime = {name: dict(mean=float(values.mean()), median=float(np.median(values)),
                         p95=float(np.percentile(values, 95)), sum=float(values.sum()))
               for name in ('decode_seconds', 'validation_seconds') for values in (arrays[name],)}
    destination = out / f'd{distance}'
    destination.mkdir(parents=True, exist_ok=True)
    # Persist per-shot predictions and diversity in the repository. Full candidate
    # costs and chunk checkpoints remain at the hashed raw-artifact location.
    with (destination / 'predictions_and_diagnostics.npz').open('wb') as handle:
        np.savez_compressed(handle, rows=rows, actual_masks=actual_masks,
                            method_names=np.array(list(predictions)),
                            predictions=np.stack([masks(p) for p in predictions.values()], axis=1),
                            candidate_masks=arrays['candidate_masks'],
                            logical_rank=arrays['logical_rank'], logical_diversity=logical_diversity,
                            physical_diversity=arrays['physical_diversity'], internal_edges=arrays['internal_edges'])
    summary = dict(
        parameters=params, shots=expected_shots, ensemble_seed=request['seed'], methods=methods,
        paired_to_correlated_uf=paired, diagnostics=diagnostics,
        normalization=dict(pieces=6 * 4 * distance, values=8),
        bootstrap=dict(unit='whole shot', replicates=10000, seed=43),
        runtime_under_parallel_load=runtime, collection=manifest, raw_directory=str(directory.resolve()),
        input_sample_sha256=request['input_run']['payload_sha256'],
        analysis_source_sha256=source_hashes((
            'docs/results/coset_ensemble_d7_d9_p003_100k/report.py',
            'src/yoked/hierarchical/_uf_soft_analysis.py')),
        verification=dict(input_hashes_checked=True, all_candidates_syndrome_checked=True,
                          saved_baseline_bit_identical=True, votes_independently_recomputed=True,
                          source_hashes_checked=True),
    )
    write_json_atomic(destination / 'summary.json', summary)
    for name in ('request.json', 'manifest.json'):
        shutil.copyfile(directory / name, destination / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--distances', nargs='+', type=int, default=[7, 9])
    parser.add_argument('--expected-shots', type=int, default=100000)
    args = parser.parse_args()
    results = {str(d): verify_and_export(args.root, args.out, d, args.expected_shots) for d in args.distances}
    write_json_atomic(args.out / 'comparison.json', results)
    header = '| Decoder |' + ''.join(f' d={d} failures | d={d} LER (95% CI) |' for d in args.distances)
    lines = [header, '|---|' + '---:|---:|' * len(args.distances)]
    names = next(iter(results.values()))['methods']
    for name in names:
        line = f'| {name} |'
        for result in results.values():
            row = result['methods'][name]
            low, high = row['ler_ci95']
            line += f" {row['failures']:,} | {row['normalized_ler']:.9g} [{low:.9g}, {high:.9g}] |"
        lines.append(line)
    lines += ['', f'Each distance uses {args.expected_shots:,} saved shots. Normalization: '
              '`pieces=6*4*d`, `values=8`. Intervals use paired whole-shot bootstrap.', '',
              '| Decoder |' + ''.join(f' d={d} repairs | d={d} regressions | d={d} changed predictions |'
                                      for d in args.distances),
              '|---|' + '---:|---:|---:|' * len(args.distances)]
    for name in names:
        line = f'| {name} |'
        for result in results.values():
            row = result['paired_to_correlated_uf'][name]
            line += f" {row['repairs']:,} | {row['regressions']:,} | {row['different_logical_predictions']:,} |"
        lines.append(line)
    (args.out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({d: {name: value['failures'] for name, value in result['methods'].items()}
                      for d, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
