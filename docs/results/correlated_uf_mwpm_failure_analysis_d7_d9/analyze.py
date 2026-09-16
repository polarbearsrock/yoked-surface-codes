"""Diagnose saved correlated-UF versus correlated-MWPM failures without resampling.

This is analysis code, not a proposed decoder. Truth labels are used for cohort
selection and logical-space certificates, never to set decoding weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pymatching
import stim

from yoked.decoders import DecodingGraph, UnionFindDecoder
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.hierarchical._collect import packed_sample_hash
from yoked.hierarchical._matching_gaps import _CheckMatrixGraph
from yoked.hierarchical._record import load_record

RECORD_ROOT = Path('/data2/s2chitni/.tmp/hier-three-decoders-d7-d9-p003-100k-3MxG5S')
CASE_ROOTS = [Path('/data2/s2chitni/.tmp') / name for name in (
    'uf-case-collection-9pbSDiuL', 'uf-case-collection-next-6a0xaw4f')]
SOURCES = [
    'src/yoked/decoders/_union_find.py', 'src/yoked/decoders/_graph.py',
    'src/yoked/decoders/_correlated_union_find.py', 'src/yoked/decoders/_correlations.py',
    'src/yoked/hierarchical/_matching_gaps.py',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=dict) + '\n')


def bitmask(bits):
    return sum(int(b) << k for k, b in enumerate(bits))


def wrong_bits(prediction, actual):
    return [k for k in range(12) if ((prediction ^ actual) >> k) & 1]


def logical_basis(graph, edge_ids):
    """Find L(ker H) by mapping fundamental cycles into logical masks.

    Boundary terminals are unconstrained; collapsing them to one vertex turns
    allowed boundary-to-boundary paths into cycles without adding constraints.
    Union-find potentials track the observable XOR along each spanning path.
    """
    n = graph.num_detectors + 1
    parent, size, potential, basis = list(range(n)), [1] * n, [0] * n, {}

    def find(v):
        label = 0
        while parent[v] != v:
            label ^= potential[v]
            v = parent[v]
        return v, label

    for e in edge_ids:
        u, v, _, label = graph.edges[e]
        a, x = find(u)
        b, y = find(graph.num_detectors if v is None else v)
        delta = x ^ y ^ label
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b], potential[b] = a, delta
            size[a] += size[b]
        else:
            while delta:
                pivot = delta.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = delta
                    break
                delta ^= basis[pivot]
    return basis


def contains(basis, mask):
    while mask:
        pivot = mask.bit_length() - 1
        if pivot not in basis:
            return False
        mask ^= basis[pivot]
    return True


class Analysis:
    def __init__(self, distance):
        self.distance = distance
        self.loaded = load_record(RECORD_ROOT / f'd{distance}/evaluation')
        self.record = self.loaded.record
        self.run = self.loaded.manifest['baselines']['run']
        self.directory = Path(self.run['directory'])
        self.packed = np.load(self.directory / 'detectors_packed.npy', mmap_mode='r')
        actual = np.load(self.directory / 'actual_observables_packed.npy', mmap_mode='r')
        assert packed_sample_hash(self.packed, actual) == self.run['payload_sha256']
        np.testing.assert_array_equal(
            np.unpackbits(actual, axis=1, count=12, bitorder='little'), self.record.actual)
        assert sha(self.directory / 'model.dem') == self.run['dem_sha256']
        assert sha(self.directory / 'circuit.stim') == self.run['circuit_sha256']
        dem = stim.DetectorErrorModel.from_file(self.directory / 'model.dem')
        self.graph = DecodingGraph.from_dem(dem)
        self.matrix = _CheckMatrixGraph(self.graph)
        self.endpoints = np.array(self.graph.endpoints)
        self.rules = index_rules_by_source(self.graph, correlation_rules_from_dem(self.graph, dem))
        self.native = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        self.uf = UnionFindDecoder(self.graph)
        self.full_basis = logical_basis(self.graph, range(len(self.graph.edges)))
        assert len(self.full_basis) == 10

    def syndrome(self, shot):
        return np.unpackbits(self.packed[shot], count=self.graph.num_detectors,
                             bitorder='little').astype(bool)

    def mask(self, edges, syndrome=None):
        label = 0
        response = np.zeros(self.graph.num_detectors, dtype=bool)
        for e in edges:
            u, v, _, obs = self.graph.edges[e]
            label ^= obs
            response[u] ^= True
            if v is not None:
                response[v] ^= True
        if syndrome is not None:
            np.testing.assert_array_equal(response, syndrome)
        return label

    def weights(self, edges):
        adjusted = apply_correlation_rules(self.matrix.weights, self.rules, edges)
        return self.matrix.weights if adjusted is None else np.asarray(adjusted)

    def regraph(self, weights):
        return DecodingGraph(self.graph.num_detectors, 12,
                             [(u, v, weights[e], label)
                              for e, (u, v, _, label) in enumerate(self.graph.edges)])

    def matching_edges(self, syndrome, weights):
        return self.matrix.edge_ids_of(self.matrix.matcher(weights).decode_to_edges_array(syndrome))

    def population(self):
        u = self.record.baselines['joint_correlated_uf'] ^ self.record.actual
        m = self.record.baselines['joint_correlated_mwpm'] ^ self.record.actual
        uf, mf = u.any(axis=1), m.any(axis=1)
        target = uf & ~mf
        self.targets = np.flatnonzero(target)
        sectors = u.reshape(-1, 6, 2).any(axis=1)
        return dict(shots=len(uf), both_pass=int((~uf & ~mf).sum()),
                    uf_only_fails=int(target.sum()), mwpm_only_fails=int((~uf & mf).sum()),
                    both_fail=int((uf & mf).sum()),
                    wrong_bit_counts={str(k): int(v) for k, v in
                                      zip(*np.unique(u[target].sum(axis=1), return_counts=True))},
                    X_only=int((target & sectors[:, 0] & ~sectors[:, 1]).sum()),
                    Z_only=int((target & ~sectors[:, 0] & sectors[:, 1]).sum()),
                    both_sectors=int((target & sectors[:, 0] & sectors[:, 1]).sum()))

    def analyze(self, shot, detail=False):
        s = self.syndrome(shot)
        actual = bitmask(self.record.actual[shot])
        first = self.uf._decode(s)
        wu = self.weights(first.selected_edges)
        second, growth = UnionFindDecoder(self.regraph(wu))._decode_state(s)
        assert second.observable_mask == bitmask(self.record.baselines['joint_correlated_uf'][shot])
        self.mask(second.selected_edges, s)
        native_edges = self.matrix.edge_ids_of(self.native.decode_to_edges_array(s, enable_correlations=True))
        assert self.mask(native_edges, s) == bitmask(self.record.baselines['joint_correlated_mwpm'][shot])
        assert second.observable_mask != actual == self.mask(native_edges)
        # The native correlation-enabled object's uncorrelated pass supplies its evidence.
        mfirst = self.matrix.edge_ids_of(self.native.decode_to_edges_array(s))
        wm = self.weights(mfirst)
        um = self.matching_edges(s, wu)
        mu = UnionFindDecoder(self.regraph(wm))._decode(s).selected_edges
        mm = self.matching_edges(s, wm)
        masks = dict(UU=second.observable_mask, UM=self.mask(um, s),
                     MU=self.mask(mu, s), MM_rebuilt=self.mask(mm, s), MM_native=self.mask(native_edges, s))
        roots = np.array([growth.find(v) for v in range(len(self.graph.adjacency))])
        allowed = np.flatnonzero(roots[self.endpoints[:, 0]] == roots[self.endpoints[:, 1]])
        wanted = second.observable_mask ^ actual
        assert contains(self.full_basis, wanted)
        forest_basis = logical_basis(self.graph, growth.forest)
        partition_basis = logical_basis(self.graph, allowed)
        forest_ok, partition_ok = contains(forest_basis, wanted), contains(partition_basis, wanted)
        assert not forest_ok or partition_ok
        cost = lambda weights, edges: math.fsum(weights[e] for e in edges)
        result = dict(shot=shot, actual_mask=actual, predictions=masks,
                      wrong={k: wrong_bits(v, actual) for k, v in masks.items()},
                      forest_allows_truth=forest_ok, partition_allows_truth=partition_ok,
                      forest_logical_rank=len(forest_basis), partition_logical_rank=len(partition_basis),
                      same_U_weights_cost_UF=cost(wu, second.selected_edges),
                      same_U_weights_cost_MWPM=cost(wu, um),
                      original_defects=int(s.sum()), yokes=s[-2:].astype(int).tolist(),
                      differing_weights=int(np.count_nonzero(wu != wm)))
        if detail:
            result['detail'] = dict(first_U_edges=list(first.selected_edges), first_M_edges=mfirst,
                                    second_U_edges=list(second.selected_edges), second_M_edges=native_edges,
                                    forest=list(growth.forest))
        return result


def summarize(rows):
    n = len(rows)
    return dict(shots=n, UM_correct=sum(not r['wrong']['UM'] for r in rows),
                MU_correct=sum(not r['wrong']['MU'] for r in rows),
                MM_rebuilt_correct=sum(not r['wrong']['MM_rebuilt'] for r in rows),
                both_swaps_correct=sum(not r['wrong']['UM'] and not r['wrong']['MU'] for r in rows),
                neither_swap_correct=sum(bool(r['wrong']['UM']) and bool(r['wrong']['MU']) for r in rows),
                partition_excludes_truth=sum(not r['partition_allows_truth'] for r in rows),
                forest_excludes_truth=sum(not r['forest_allows_truth'] for r in rows),
                weight_deficit_median=float(np.median([
                    r['same_U_weights_cost_UF'] - r['same_U_weights_cost_MWPM'] for r in rows])),
                weight_deficit_positive=sum(r['same_U_weights_cost_UF'] >
                                           r['same_U_weights_cost_MWPM'] + 1e-6 for r in rows))


def build_report(analysis, rows, seed, script):
    return dict(distance=analysis.distance, population=analysis.population(), sample_seed=seed,
                sample_summary=summarize([r for r in rows if r['in_random_sample']]),
                inputs=dict(analysis.run), source_hashes={p: sha(p) for p in SOURCES},
                analysis_script_sha256=sha(script),
                versions=dict(stim=stim.__version__, pymatching=pymatching.__version__, numpy=np.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    analysis = Analysis(args.distance)
    population = analysis.population()
    seed = 20260915 + args.distance
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(analysis.targets, size=args.count, replace=False).tolist())
    known = sorted({int(p.name.removeprefix('shot_')) for root in CASE_ROOTS
                    for p in root.glob('shot_*') if p.is_dir()}) if args.distance == 7 else []
    known = [i for i in known if i in analysis.targets]
    ids = known + [i for i in sample if i not in known]
    write(args.out / 'selection.json', dict(distance=args.distance, population=population,
                                           sample_seed=seed, sample=sample, physical_cases=known))
    rows = []
    for i, shot in enumerate(ids):
        row = analysis.analyze(shot, detail=shot in known)
        row['in_random_sample'] = shot in sample
        rows.append(row)
        write(args.out / 'rows.json', rows)
        if i % 8 == 0 or shot in known or i == len(ids) - 1:
            print(json.dumps(dict(completed=i+1, total=len(ids), shot=shot,
                                  wrong=row['wrong'], partition_allows_truth=row['partition_allows_truth'],
                                  seconds=round(time.perf_counter()-start, 2))), flush=True)
    report = build_report(analysis, rows, seed, __file__)
    write(args.out / 'summary.json', report)
    print(json.dumps(report['sample_summary']), flush=True)


if __name__ == '__main__':
    main()
