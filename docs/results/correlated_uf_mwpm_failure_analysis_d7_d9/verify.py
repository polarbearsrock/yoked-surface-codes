"""Independent checks of diagnostic certificates, archived labels, and reductions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import stim

from analyze import Analysis, bitmask, sha, write, summarize, logical_basis, contains
from yoked.decoders import DecodingGraph, UnionFindDecoder, CorrelatedUnionFindDecoder


def all_cycle_labels_zero(graph, edges):
    """Independent DFS potential check, without the DSU cycle-basis algorithm."""
    adjacency = defaultdict(list)
    boundary = graph.num_detectors
    for e in edges:
        u, v, _, label = graph.edges[e]
        v = boundary if v is None else v
        adjacency[u].append((v, label))
        adjacency[v].append((u, label))
    potentials = {}
    for start in adjacency:
        if start in potentials:
            continue
        potentials[start] = 0
        pending = [start]
        while pending:
            u = pending.pop()
            for v, label in adjacency[u]:
                value = potentials[u] ^ label
                if v in potentials:
                    if potentials[v] != value:
                        return False
                else:
                    potentials[v] = value
                    pending.append(v)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    toy = DecodingGraph(2, 2, [(0, None, 1., 1), (0, 1, 1., 0), (1, None, 1., 2)])
    basis = logical_basis(toy, range(3))
    assert contains(basis, 3) and not contains(basis, 1)
    assert not all_cycle_labels_zero(toy, range(3))
    assert not logical_basis(toy, range(2)) and all_cycle_labels_zero(toy, range(2))
    checked, independent = 0, 0
    for distance in (7, 9):
        analysis = Analysis(distance)
        report = json.loads((root / f'd{distance}/summary.json').read_text())
        rows = json.loads((root / f'd{distance}/rows.json').read_text())
        selection = json.loads((root / f'd{distance}/selection.json').read_text())
        assert analysis.population() == report['population']
        assert report['sample_summary'] == summarize([r for r in rows if r['in_random_sample']])
        for name, digest in report['source_hashes'].items():
            assert sha(name) == digest
        assert sha(root/'analyze.py') == report['analysis_script_sha256']
        expected = sorted(np.random.default_rng(selection['sample_seed']).choice(
            analysis.targets, size=128, replace=False).tolist())
        assert selection['sample'] == expected
        for row in rows:
            shot = row['shot']
            actual = bitmask(analysis.record.actual[shot])
            assert actual == row['actual_mask'] == row['predictions']['MM_native']
            assert row['predictions']['UU'] == bitmask(analysis.record.baselines['joint_correlated_uf'][shot])
            assert actual == row['predictions']['MM_rebuilt']
            checked += 1
        # Independently certify the first eight randomly selected failures per distance.
        for shot in expected[:8]:
            syndrome = analysis.syndrome(shot)
            first = analysis.uf._decode(syndrome)
            weights = analysis.weights(first.selected_edges)
            second, growth = UnionFindDecoder(analysis.regraph(weights))._decode_state(syndrome)
            roots = np.array([growth.find(v) for v in range(len(analysis.graph.adjacency))])
            allowed = np.flatnonzero(roots[analysis.endpoints[:, 0]] == roots[analysis.endpoints[:, 1]])
            assert all_cycle_labels_zero(analysis.graph, allowed)
            assert analysis.mask(second.selected_edges, syndrome) != bitmask(analysis.record.actual[shot])
            independent += 1
        if distance == 7:
            decoder = CorrelatedUnionFindDecoder(analysis.graph, correlation_rules=[
                (source, target, weight) for source, rules in enumerate(analysis.rules)
                for target, weight in rules])
            for shot in (44875, 76890):
                reduced = json.loads((root / f'reduced_{shot}.json').read_text())
                circuit = stim.Circuit.from_file(root / f'reduced_{shot}.stim')
                ds, actuals = circuit.compile_detector_sampler(seed=987).sample(2, separate_observables=True)
                for syndrome, actual in zip(ds, actuals):
                    assert np.flatnonzero(syndrome).tolist() == reduced['defect_ids']
                    assert bitmask(actual) == reduced['reduced']['actual_mask']
                    assert bitmask(decoder.decode(syndrome)) == reduced['reduced']['CUF_mask']
                    assert bitmask(analysis.native.decode(syndrome, enable_correlations=True)) == bitmask(actual)
                assert all(not r['CUF_wrong'] and not r['CMWPM_wrong']
                           for r in reduced['single_deletions'].values())
                assert all(not r['CUF_wrong'] and not r['CMWPM_wrong']
                           for r in reduced['individual_faults'].values())
    result = dict(saved_and_rebuilt_predictions_checked=checked,
                  independent_DFS_zero_cycle_certificates=independent,
                  population_counts_and_uniform_selections_verified=True,
                  two_reduced_circuits_replayed_with_both_decoders=True,
                  source_hashes_verified=True)
    write(root/'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
