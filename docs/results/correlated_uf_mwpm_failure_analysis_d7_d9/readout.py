"""Test actual readout faults on logical correction differences in traced shots.

Selection is diagnostic and deliberately biased towards decoding differences;
these counts must not be interpreted as noise-channel failure frequencies.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import Analysis, CASE_ROOTS, bitmask, sha, write, wrong_bits
from yoked.decoders import CorrelatedUnionFindDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    a = Analysis(7)
    decoder = CorrelatedUnionFindDecoder(a.graph, correlation_rules=[
        (source, target, weight) for source, rules in enumerate(a.rules) for target, weight in rules])
    rows = json.loads((args.root / 'd7/rows.json').read_text())
    results = []
    for row in rows:
        if 'detail' not in row:
            continue
        shot = row['shot']
        directory = next(root / f'shot_{shot}' for root in CASE_ROOTS if (root / f'shot_{shot}').is_dir())
        faults = json.loads((directory / 'physical_faults.json').read_text())
        with np.load(directory / 'fault_responses.npz') as data:
            detectors, observables = data['detectors'], data['observables']
        s = a.syndrome(shot)
        actual = bitmask(a.record.actual[shot])
        np.testing.assert_array_equal(np.logical_xor.reduce(detectors, axis=0), s)
        assert bitmask(np.logical_xor.reduce(observables, axis=0)) == actual
        diff = set(row['detail']['second_U_edges']) ^ set(row['detail']['second_M_edges'])
        adjacency = {}
        for e in diff:
            u, v = a.graph.endpoints[e]
            adjacency.setdefault(u, []).append((v, e))
            adjacency.setdefault(v, []).append((u, e))
        seen, logical_edges = set(), set()
        for vertex in adjacency:
            if vertex in seen:
                continue
            pending, component = [vertex], set()
            while pending:
                v = pending.pop()
                if v in seen:
                    continue
                seen.add(v)
                for w, e in adjacency[v]:
                    component.add(e)
                    pending.append(w)
            if a.mask(component):
                logical_edges.update(component)
        for f in faults:
            if f['gate'] != 'M' or len(f['detectors']) != 2:
                continue
            edge = a.matrix.edge_ids.get(tuple(sorted(f['detectors'])))
            if edge not in logical_edges:
                continue
            fid = f['id']
            changed_s = s ^ detectors[fid]
            changed_actual = actual ^ bitmask(observables[fid])
            uf = decoder._decode(changed_s)
            u = a.mask(uf.selected_edges, changed_s)
            m = bitmask(a.native.decode(changed_s, enable_correlations=True))
            alone_u = decoder._decode(detectors[fid])
            au = a.mask(alone_u.selected_edges, detectors[fid])
            am = bitmask(a.native.decode(detectors[fid], enable_correlations=True))
            result = dict(shot=shot, fault=f, edge=edge, original_U_wrong=row['wrong']['UU'],
                          after_removal_U_wrong=wrong_bits(u, changed_actual),
                          after_removal_M_wrong=wrong_bits(m, changed_actual),
                          alone_U_wrong=wrong_bits(au, bitmask(observables[fid])),
                          alone_M_wrong=wrong_bits(am, bitmask(observables[fid])),
                          syndrome_at_fault_detectors=s[f['detectors']].astype(int).tolist(),
                          physical_source=str(directory),
                          source_sha256=sha(directory/'physical_faults.json'))
            results.append(result)
            write(args.root / 'readout_cases.json', results)
            print(json.dumps(dict(shot=shot, fault=fid, removed=result['after_removal_U_wrong'],
                                  matching=result['after_removal_M_wrong'])), flush=True)


if __name__ == '__main__':
    main()
