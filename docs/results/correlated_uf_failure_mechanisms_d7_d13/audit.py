"""Independently check logical-space certificates using DFS and explicit spans."""
import argparse
import json
from pathlib import Path

import numpy as np
import stim

import diagnose as analysis
from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem


def owners_by_dfs(graph, forest):
    adjacency = {}
    for e in forest:
        u, v = graph.endpoints[e]
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    owner = np.arange(len(graph.adjacency))
    seen = set()
    for root in adjacency:
        if root in seen:
            continue
        pending = [root]
        while pending:
            u = pending.pop()
            if u in seen:
                continue
            seen.add(u)
            owner[u] = root
            pending.extend(adjacency[u])
    return owner


def logical_span_by_dfs(graph, edges):
    adjacency = {}
    for e in edges:
        u, v, _, mask = graph.edges[e]
        if v is None:
            v = graph.num_detectors
        adjacency.setdefault(u, []).append((v, mask))
        adjacency.setdefault(v, []).append((u, mask))
    labels, cycles = {}, set()
    for root in adjacency:
        if root in labels:
            continue
        labels[root] = 0
        pending = [root]
        while pending:
            u = pending.pop()
            for v, mask in adjacency[u]:
                candidate = labels[u] ^ mask
                if v in labels:
                    cycles.add(candidate ^ labels[v])
                else:
                    labels[v] = candidate
                    pending.append(v)
    span = {0}
    for mask in cycles:
        span |= {value ^ mask for value in span}
    return span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=analysis.HERE)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for distance in (7, 9, 11, 13):
        directory = args.root / f'd{distance}'
        rows = json.loads((directory / 'rows.json').read_text())
        assert analysis.summarize(rows) == json.loads((directory / 'summary.json').read_text())
        verified = json.loads((directory / 'verification.json').read_text())
        for path, expected in verified['hashes'].items():
            assert analysis.sha256_file(path) == expected
        sample = Path(verified['source']['directory'])
        graph = DecodingGraph.from_dem(stim.DetectorErrorModel.from_file(sample / 'model.dem'))
        dem = stim.DetectorErrorModel.from_file(sample / 'model.dem')
        packed = np.load(sample / 'detectors_packed.npy', mmap_mode='r')
        chosen = rows[::16]
        native = analysis.experiment.Native(args.library, graph, correlation_rules_from_dem(graph, dem))
        try:
            out, flags = native.decode(packed[[r['shot'] for r in chosen]], threads=1, audit=True)
        finally:
            native.close()
        endpoints = np.asarray(graph.endpoints)
        checked = 0
        for k, row in enumerate(chosen):
            for name, col, bit in [('correlated_uf', 0, 1), ('frontier_uf', 3, 2), ('frontier_bridges', 5, 10)]:
                saved = row['variants'][name]
                assert int(out[k, col, 0]) == saved['prediction']
                forest = np.flatnonzero(flags[k] & (1 << bit)).tolist()
                owner = owners_by_dfs(graph, forest)
                allowed = np.flatnonzero(owner[endpoints[:, 0]] == owner[endpoints[:, 1]])
                wanted = saved['prediction'] ^ row['actual']
                for prefix, edges in [('forest', forest), ('partition', allowed)]:
                    span = logical_span_by_dfs(graph, edges)
                    assert len(span) == 1 << saved[f'{prefix}_logical_rank']
                    assert (wanted in span) == saved[f'{prefix}_allows_truth']
                    checked += 1
        result[str(distance)] = dict(shots=[r['shot'] for r in chosen],
                                    independent_DFS_span_checks=checked, source_hashes_verified=True)
        print(f'd={distance}: {checked} independent logical-span checks passed', flush=True)
    result['audit_script_sha256'] = analysis.sha256_file(__file__)
    analysis.write(args.root / 'audit.json', result)


if __name__ == '__main__':
    main()
