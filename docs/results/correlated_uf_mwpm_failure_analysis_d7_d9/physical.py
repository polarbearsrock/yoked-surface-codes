"""Recheck recovered physical faults using correlated UF, including deletions."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import Analysis, CASE_ROOTS, bitmask, sha, write, wrong_bits
from yoked.decoders import CorrelatedUnionFindDecoder, UnionFindDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analysis = Analysis(7)
    analysis.population()
    decoder = CorrelatedUnionFindDecoder(analysis.graph,
        correlation_rules=[(source, target, weight) for source, rules in enumerate(analysis.rules)
                           for target, weight in rules])
    labels = [-1] * len(analysis.graph.adjacency)
    for sector in range(2):
        pending = [analysis.graph.num_detectors - 2 + sector]
        while pending:
            vertex = pending.pop()
            if labels[vertex] == sector:
                continue
            assert labels[vertex] == -1
            labels[vertex] = sector
            for e in analysis.graph.adjacency[vertex]:
                u, v = analysis.graph.endpoints[e]
                pending.append(v if u == vertex else u)
    assert -1 not in labels

    def decode(syndrome, actual):
        correction = decoder._decode(syndrome)
        umask = analysis.mask(correction.selected_edges, syndrome)
        assert correction.observable_mask == umask
        edges = analysis.matrix.edge_ids_of(analysis.native.decode_to_edges_array(
            syndrome, enable_correlations=True))
        mmask = analysis.mask(edges, syndrome)
        return dict(actual_mask=actual, CUF_mask=umask, CMWPM_mask=mmask,
                    CUF_wrong=wrong_bits(umask, actual), CMWPM_wrong=wrong_bits(mmask, actual))

    rows = []
    for root in CASE_ROOTS:
        for directory in sorted(root.glob('shot_*')):
            shot = int(directory.name.removeprefix('shot_'))
            if shot not in analysis.targets:
                continue
            s = analysis.syndrome(shot)
            actual_bits = analysis.record.actual[shot]
            actual = bitmask(actual_bits)
            faults = json.loads((directory / 'physical_faults.json').read_text())
            evidence = json.loads((directory / 'evidence.json').read_text())
            focus = [entry['fault']['id'] for entry in evidence['focus']]
            with np.load(directory / 'fault_responses.npz') as data:
                detectors, observables = data['detectors'], data['observables']
            np.testing.assert_array_equal(np.logical_xor.reduce(detectors, axis=0), s)
            np.testing.assert_array_equal(np.logical_xor.reduce(observables, axis=0), actual_bits)
            for fault, dets, obs in zip(faults, detectors, observables, strict=True):
                assert fault['detectors'] == np.flatnonzero(dets).tolist()
                assert fault['observables'] == np.flatnonzero(obs).tolist()
            first = analysis.uf._decode(s)
            wu = analysis.weights(first.selected_edges)
            second, growth = UnionFindDecoder(analysis.regraph(wu))._decode_state(s)
            mfirst = analysis.matrix.edge_ids_of(analysis.native.decode_to_edges_array(s))
            wm = analysis.weights(mfirst)
            native = set(analysis.matrix.edge_ids_of(analysis.native.decode_to_edges_array(s, enable_correlations=True)))
            baseline = decode(s, actual)
            assert baseline['CUF_mask'] == bitmask(analysis.record.baselines['joint_correlated_uf'][shot])
            assert baseline['CMWPM_mask'] == actual
            summaries = []
            for fault_id in focus:
                fault = faults[fault_id]
                groups = [[d for d in fault['detectors'] if labels[d] == sector] for sector in range(2)]
                edges = []
                for group in groups:
                    if not group:
                        continue
                    assert len(group) in (1, 2)
                    key = (group[0], -1) if len(group) == 1 else tuple(sorted(group))
                    edges.append(analysis.matrix.edge_ids[key])
                assert analysis.mask(edges, detectors[fault_id]) == bitmask(observables[fault_id])
                components = []
                for e in edges:
                    u, v = analysis.graph.endpoints[e]
                    sources = [(source, weight) for source, rules in enumerate(analysis.rules)
                               for target, weight in rules if target == e]
                    components.append(dict(edge=e, endpoints=list(analysis.graph.edges[e][:2]),
                        fired=[bool(s[v]) if v < len(s) else None for v in (u, v)],
                        original_weight=float(analysis.matrix.weights[e]), UF_conditioned_weight=float(wu[e]),
                        MWPM_conditioned_weight=float(wm[e]),
                        first_U_selects=e in first.selected_edges, first_M_selects=e in mfirst,
                        second_U_selects=e in second.selected_edges, native_M_selects=e in native,
                        second_U_forest=e in growth.forest,
                        second_U_same_cluster=growth.find(u) == growth.find(v),
                        UF_support=[source for source, _ in sources if source in first.selected_edges],
                        MWPM_support=[source for source, _ in sources if source in mfirst]))
                summaries.append(dict(fault=fault, components=components,
                    removed=decode(s ^ detectors[fault_id], actual ^ bitmask(observables[fault_id])),
                    alone=decode(detectors[fault_id], bitmask(observables[fault_id]))))
            pair_d = np.logical_xor.reduce(detectors[focus], axis=0)
            pair_o = bitmask(np.logical_xor.reduce(observables[focus], axis=0))
            row = dict(shot=shot, physical_events=len(faults), baseline=baseline,
                       focus=summaries, both_removed=decode(s ^ pair_d, actual ^ pair_o),
                       pair_alone=decode(pair_d, pair_o),
                       physical_source=str(directory),
                       source_hashes={name: sha(directory/name) for name in
                                      ('physical_faults.json', 'fault_responses.npz', 'validation.json', 'evidence.json')},
                       original_validation=json.loads((directory/'validation.json').read_text()))
            rows.append(row)
            write(args.out / 'physical_cases.json', rows)
            print(json.dumps(dict(shot=shot, original=baseline['CUF_wrong'],
                                  removed={str(r['fault']['id']): r['removed']['CUF_wrong'] for r in summaries},
                                  both_removed=row['both_removed']['CUF_wrong'])), flush=True)


if __name__ == '__main__':
    main()
