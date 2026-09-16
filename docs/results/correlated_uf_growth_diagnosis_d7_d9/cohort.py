"""Classify missing connections and probe boundary completion at fixed weights.

Thresholds are exploratory after inspection of reduced cases. Conditional
failure and success cohorts cannot establish a population LER improvement.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
import pymatching

from trace import Analysis, PREVIOUS, difference_components, write
from algorithms import minimum_cost_forest
from yoked.decoders import UnionFindDecoder
from yoked.decoders._union_find import _peel

CAPS = (0., .1, .25, .5, 1., math.inf)


def cap_name(cap):
    return 'all' if math.isinf(cap) else str(cap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a = Analysis(args.distance)
    selection = json.loads((PREVIOUS/f'd{args.distance}/selection.json').read_text())
    target = selection['sample']
    uf_ok = np.all(a.record.baselines['joint_correlated_uf'] == a.record.actual, axis=1)
    mwpm_ok = np.all(a.record.baselines['joint_correlated_mwpm'] == a.record.actual, axis=1)
    control_seed = 2026091800 + args.distance
    controls = sorted(np.random.default_rng(control_seed).choice(
        np.flatnonzero(uf_ok & mwpm_ok), size=128, replace=False).tolist())
    write(args.out/'selection.json', dict(failures=target, both_correct_controls=controls,
                                        control_seed=control_seed, caps=[cap_name(x) for x in CAPS]))
    boundary = [e for e, (_, v, _, _) in enumerate(a.graph.edges) if v is None]
    rows = []
    for cohort, ids in [('UF_only_failure',target), ('both_correct',controls)]:
        for shot in ids:
            syndrome = a.syndrome(shot)
            actual = sum(int(b) << k for k,b in enumerate(a.record.actual[shot]))
            first = a.uf._decode(syndrome)
            weights = a.weights(first.selected_edges)
            graph = a.regraph(weights)
            second, growth = UnionFindDecoder(graph)._decode_state(syndrome)
            assert second.observable_mask == sum(int(b)<<k for k,b in
                                                 enumerate(a.record.baselines['joint_correlated_uf'][shot]))
            forest = list(second.forest_edges)
            baseline_dp, baseline_cost = minimum_cost_forest(graph, syndrome, forest)
            baseline_dp_mask = a.mask(baseline_dp, syndrome)
            states = {}
            # Boundary terminals have degree one, so every omitted boundary edge
            # adds a new leaf and preserves the forest property.
            for e in boundary:
                u,v = graph.endpoints[e]
                if growth.find(u) == growth.find(v):
                    continue
                spent = min(weights[e], growth.grown[e] + growth.rate[e]*(growth.time-growth.settled_at[e]))
                states[e] = max(0., weights[e]-spent)
            variants = {}
            for cap in CAPS:
                extra = [e for e, remaining in states.items() if remaining <= cap]
                candidate = forest+extra
                selected,cost = minimum_cost_forest(graph,syndrome,candidate)
                mask = a.mask(selected,syndrome)
                peel, peelmask = _peel(graph,syndrome,candidate)
                assert a.mask(peel,syndrome)==peelmask
                variants[cap_name(cap)] = dict(mask=mask,correct=mask==actual,cost=cost,
                                               added_edges=len(extra),ordinary_peel_correct=peelmask==actual)
            row = dict(shot=shot,cohort=cohort,actual=actual,baseline_UF=second.observable_mask,
                       tree_DP_original_mask=baseline_dp_mask,tree_DP_original_cost=baseline_cost,
                       variants=variants)
            if cohort == 'UF_only_failure':
                matching = a.matching_edges(syndrome,weights)
                row['same_weight_MWPM_correct'] = a.mask(matching,syndrome)==actual
                components = [c for c in difference_components(graph,second.selected_edges,matching) if c['mask']]
                mset = set(matching)
                logical_edges = {e for c in components for e in c['edges'] if e in mset}
                missing = [e for e in logical_edges if growth.find(graph.endpoints[e][0]) !=
                                                        growth.find(graph.endpoints[e][1])]
                kinds = Counter('boundary' if graph.edges[e][1] is None else
                                'yoke' if any(v >= graph.num_detectors-2 and v < graph.num_detectors
                                              for v in graph.endpoints[e]) else 'ordinary' for e in missing)
                row['missing_edges_by_kind']=dict(kinds)
                row['missing_edges']=missing
                row['missed_boundary_remaining']=[states[e] for e in missing if e in states]
                row['logical_components']=[dict(mask=c['mask'],edges=c['edges']) for c in components]
            # Compare the tree DP's cost with an independent exact matching solve
            # on both its original and completed trees for 4 shots per cohort.
            if ids.index(shot) < 4:
                for cap in (0., .25, math.inf):
                    edges=forest+[e for e,rem in states.items() if rem<=cap]
                    matcher=pymatching.Matching.from_check_matrix(
                        a.matrix.check_matrix[:,edges],weights=weights[edges],
                        faults_matrix=a.matrix.faults_matrix[:,edges],merge_strategy='disallow')
                    selected=a.matrix.edge_ids_of(matcher.decode_to_edges_array(syndrome))
                    assert a.mask(selected,syndrome) >= 0
                    exact_cost=math.fsum(weights[e] for e in selected)
                    assert math.isclose(exact_cost,variants[cap_name(cap)]['cost'],rel_tol=1e-9,abs_tol=1e-5)
                row['independent_tree_cost_checks']=3
            rows.append(row)
            if len(rows)%16==0:
                write(args.out/'rows.json',rows)
                print(json.dumps(dict(distance=args.distance,completed=len(rows),cohort=cohort)),flush=True)
    write(args.out/'rows.json',rows)
    fail=[r for r in rows if r['cohort']=='UF_only_failure']
    same=[r for r in fail if r['same_weight_MWPM_correct']]
    categories = Counter('boundary_only' if set(r['missing_edges_by_kind'])=={'boundary'} else
                         'ordinary_or_yoke' for r in same)
    summary=dict(distance=args.distance,failures=len(fail),controls=len(controls),
        same_weight_MWPM_repairs=len(same),missing_connection_categories=dict(categories),
        single_missing_boundary=sum(r['missing_edges_by_kind']=={'boundary':1} for r in same),
        variants={cap_name(cap):dict(
            failure_repairs=sum(r['variants'][cap_name(cap)]['correct'] for r in fail),
            control_regressions=sum(not r['variants'][cap_name(cap)]['correct'] for r in rows if r['cohort']=='both_correct'),
            ordinary_peel_failure_repairs=sum(r['variants'][cap_name(cap)]['ordinary_peel_correct'] for r in fail),
            mean_added_edges=float(np.mean([r['variants'][cap_name(cap)]['added_edges'] for r in rows]))) for cap in CAPS},
        original_tree_DP_failure_repairs=sum(r['tree_DP_original_mask']==r['actual'] for r in fail),
        original_tree_DP_control_regressions=sum(r['tree_DP_original_mask']!=r['actual'] for r in rows if r['cohort']=='both_correct'),
        independent_tree_cost_checks=sum(r.get('independent_tree_cost_checks',0) for r in rows))
    write(args.out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
