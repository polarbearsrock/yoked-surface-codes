"""Per-patch CPU time for the minimal incremental confidence paths.

All methods share a separately timed UF reference. Matching confidence runs
only the unforced MWPM conditioning pass and four forced class solves; it does
not pay for the collector's extra plain gaps or validation decode.
"""
from pathlib import Path
import argparse
import math
import os
import time

import numpy as np
import stim

from yoked.decoders._correlations import apply_correlation_rules, correlation_rules_from_dem
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._matching_gaps import MatchingGaps
from yoked.hierarchical._patch_graphs import PatchGraphs
from yoked.hierarchical._provenance import write_json_atomic
from yoked.hierarchical._record import load_record
from yoked.hierarchical._uf_soft import UFSoftDecoder, _reweighted


def benchmark(record_dir, shots):
    loaded = load_record(record_dir)
    sample = SampleSet.load(Path(record_dir) / 'sample')
    rows = loaded.record.rows[:shots]
    det, _ = sample.rows(rows)
    patches = PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text), num_patches=6)
    times = {name: [] for name in ('reference_uf', 'uf_gap', 'correlated_uf_gap',
                                 'correlated_bounded_cluster', 'correlated_matching_gap')}
    topology = []
    # Patch setup is excluded equally for all methods.
    for index, patch in enumerate(patches):
        decoder = UFSoftDecoder(patch)
        matching = MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))
        topology.append(dict(detectors=patch.num_detectors, edges=len(patch.graph.edges),
                             check_degrees=[len(patch.check_graph.adjacency[v]) for v in patch.check_vertices]))
        for shot, syndrome in enumerate(patch.local_syndromes(det)):
            start = time.process_time()
            first = decoder._uf.decode_to_edge_ids(syndrome)
            times['reference_uf'].append(time.process_time() - start)
            start = time.process_time()
            decoder.forced_costs(syndrome)
            times['uf_gap'].append(time.process_time() - start)
            start = time.process_time()
            weights = apply_correlation_rules(decoder._weights, decoder._rules, first)
            check = decoder._check if weights is None else UnionFindDecoder(_reweighted(patch.check_graph, weights))
            decoder.forced_costs(syndrome, check)
            times['correlated_uf_gap'].append(time.process_time() - start)
            start = time.process_time()
            weights = apply_correlation_rules(decoder._weights, decoder._rules, first)
            free = decoder._uf if weights is None else UnionFindDecoder(_reweighted(patch.graph, weights))
            second = free.decode_with_growth_costs(syndrome)
            decoder._gap.gaps_from_costs(second.remaining_costs, max_gap=decoder.gap_cap)
            times['correlated_bounded_cluster'].append(time.process_time() - start)
            start = time.process_time()
            selected = matching._free.edge_ids_of(matching._plain_free.decode_to_edges_array(syndrome))
            weights = apply_correlation_rules(matching._free.weights, matching._rules, selected)
            matcher = matching._plain_check if weights is None else matching._check.matcher(weights)
            costs = matching._forced(matcher, syndrome)
            times['correlated_matching_gap'].append(time.process_time() - start)
            np.testing.assert_allclose(costs, loaded.record.forced_correlated[shot, index], rtol=0, atol=1e-8)
    return dict(parent_rows=rows.tolist(), patch_samples=shots * 6, topology=topology,
                timer='time.process_time; graph setup excluded; calibration sample; parallel experiment active',
                seconds={name: dict(mean=float(np.mean(values)), p50=float(np.median(values)),
                                   p95=float(np.percentile(values, 95)), total=float(sum(values)))
                         for name, values in times.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, default=32)
    args = parser.parse_args()
    root = Path(os.environ['TMPDIR'])
    paths = {7: root / 'hier-three-decoders-d7-d9-p003-100k-3MxG5S/d7/calibration',
             9: root / 'hier-d9-p003-m2/calibration'}
    result = {}
    for distance, record in paths.items():
        result[str(distance)] = benchmark(record, args.shots)
        print(distance, result[str(distance)]['seconds'], flush=True)
    write_json_atomic(Path(__file__).with_name('benchmark.json'), result)


if __name__ == '__main__':
    main()
