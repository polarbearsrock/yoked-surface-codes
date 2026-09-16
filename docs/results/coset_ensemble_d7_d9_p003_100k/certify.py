"""Independently enumerate admissible logical classes for nonzero-rank shots.

The existing failure-analysis DSU certificate is independent of the ensemble's
DFS logical-rank calculation. Truth is used only here, after decoding, to ask
whether any valid correction inside the final partitions could succeed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import stim

from yoked.decoders import DecodingGraph, UnionFindDecoder
from yoked.decoders._correlations import apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source
from yoked.hierarchical._provenance import sha256_file, write_json_atomic
from yoked.hierarchical._record import load_record

from run import masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--distances', nargs='+', type=int, default=[7, 9])
    args = parser.parse_args()
    helper = Path(__file__).resolve().parent.parent / 'correlated_uf_mwpm_failure_analysis_d7_d9' / 'analyze.py'
    spec = importlib.util.spec_from_file_location('failure_analysis_certificate', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = {}
    for distance in args.distances:
        directory = args.results / f'd{distance}'
        request = json.loads((directory / 'request.json').read_text())['request']
        summary = json.loads((directory / 'summary.json').read_text())
        record = load_record(request['record_dir']).record
        raw = Path(request['input_run']['directory'])
        assert sha256_file(raw / 'model.dem') == request['input_run']['dem_sha256']
        packed = np.load(raw / 'detectors_packed.npy', mmap_mode='r')
        graph = DecodingGraph.from_dem(stim.DetectorErrorModel.from_file(raw / 'model.dem'))
        rules = index_rules_by_source(graph, correlation_rules_from_dem(
            graph, stim.DetectorErrorModel.from_file(raw / 'model.dem')))
        weights = [w for _, _, w, _ in graph.edges]
        endpoints = np.asarray(graph.endpoints)
        cases = []
        with np.load(directory / 'predictions_and_diagnostics.npz') as saved:
            ranks = saved['logical_rank']
        positive_rows = summary['diagnostics']['rank_positive_rows']
        np.testing.assert_array_equal(positive_rows, np.flatnonzero(ranks))
        control_rows = np.sort(np.random.default_rng(2026091600 + distance).choice(
            np.flatnonzero(ranks == 0), 32, replace=False)).tolist()
        for row in positive_rows + control_rows:
            syndrome = np.unpackbits(packed[row], count=graph.num_detectors, bitorder='little')
            first = UnionFindDecoder(graph)._decode(syndrome)
            adjusted = apply_correlation_rules(weights, rules, first.selected_edges)
            final_graph = graph if adjusted is None else DecodingGraph(
                graph.num_detectors, graph.num_observables,
                [(u, v, adjusted[e], mask) for e, (u, v, _, mask) in enumerate(graph.edges)])
            correction, growth = UnionFindDecoder(final_graph)._decode_state(syndrome)
            roots = np.array([growth.find(v) for v in range(len(graph.adjacency))])
            internal = np.flatnonzero(roots[endpoints[:, 0]] == roots[endpoints[:, 1]])
            basis = module.logical_basis(final_graph, internal)
            assert len(basis) == ranks[row]
            truth = int(masks(record.actual[row:row + 1])[0])
            uf = int(masks(record.baselines['joint_correlated_uf'][row:row + 1])[0])
            assert correction.observable_mask == uf
            admissible = {uf}
            for vector in basis.values():
                admissible |= {mask ^ vector for mask in admissible}
            if row in control_rows:
                continue
            cases.append(dict(
                shot=row, baseline=uf, actual=truth,
                correlated_mwpm=int(masks(record.baselines['joint_correlated_mwpm'][row:row + 1])[0]),
                logical_rank=len(basis), basis=list(basis.values()),
                admissible_logical_masks=sorted(admissible),
                truth_reachable=module.contains(basis, uf ^ truth),
            ))
        # At rank zero the baseline is the sole admissible logical label.
        # Every possible improvement must therefore come from a positive-rank
        # baseline failure whose actual label lies in the affine logical space.
        repairable = sum(c['baseline'] != c['actual'] and c['truth_reachable'] for c in cases)
        results[str(distance)] = dict(nonzero_rank_cases=cases, independently_checked_rank_zero_rows=control_rows,
                                     failures_repairable_within_partitions=repairable)
    write_json_atomic(args.results / 'logical_space_certificate.json', dict(
        independent_helper=str(helper), helper_sha256=sha256_file(helper), distances=results))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
