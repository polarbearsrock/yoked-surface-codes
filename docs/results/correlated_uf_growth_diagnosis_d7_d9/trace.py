"""Observe the second UF pass on reduced failures, holding its weights fixed."""
from pathlib import Path
import json
import math
import sys

import numpy as np

PREVIOUS = Path('docs/results/correlated_uf_mwpm_failure_analysis_d7_d9').resolve()
sys.path.insert(0, str(PREVIOUS))
from analyze import Analysis, bitmask, write, logical_basis, contains
from yoked.decoders._union_find import _Growth, _peel


def difference_components(graph, first, second):
    adjacency = {}
    diff = set(first) ^ set(second)
    for e in diff:
        u, v = graph.endpoints[e]
        adjacency.setdefault(u, []).append((v, e))
        adjacency.setdefault(v, []).append((u, e))
    seen, components = set(), []
    for start in adjacency:
        if start in seen:
            continue
        pending, edges, vertices = [start], set(), []
        while pending:
            u = pending.pop()
            if u in seen:
                continue
            seen.add(u)
            vertices.append(u)
            for v, e in adjacency[u]:
                edges.add(e)
                pending.append(v)
        mask = 0
        for e in edges:
            mask ^= graph.edges[e][3]
        components.append(dict(edges=sorted(edges), vertices=sorted(vertices), mask=mask))
    return components


class TracedGrowth(_Growth):
    def __init__(self, graph, syndrome, tracked):
        super().__init__(graph, syndrome)
        self.syndrome = syndrome
        self.tracked = sorted(tracked)
        self.fired = np.flatnonzero(syndrome).tolist()
        self.history = []

    def cluster(self, vertex):
        root = self.find(vertex)
        return dict(root=root, size=self.size[root], parity=int(self.parity[root]),
                    boundary=bool(self.terminal[root]), active=bool(self.active(root)),
                    fired=[d for d in self.fired if self.find(d) == root],
                    yokes=[d for d in (self.graph.num_detectors-2, self.graph.num_detectors-1)
                           if self.find(d) == root])

    def edge(self, e):
        u, v = self.graph.endpoints[e]
        weight = self.graph.edges[e][2]
        grown = min(weight, self.grown[e] + self.rate[e] * (self.time-self.settled_at[e]))
        return dict(edge=e, endpoints=[u,v], weight=float(weight), grown=float(grown),
                    remaining=float(max(0.,weight-grown)), rate=self.rate[e],
                    same_cluster=self.find(u)==self.find(v),
                    u=self.cluster(u), v=self.cluster(v))

    def _merge_group(self, edges):
        before = {e:self.edge(e) for e in self.tracked}
        old_count = len(self.forest)
        old_active = self.num_active
        super()._merge_group(edges)
        changes = []
        for e, old in before.items():
            new = self.edge(e)
            if any(old[k] != new[k] for k in ('rate','same_cluster')) or any(
                    old[p][k] != new[p][k] for p in ('u','v') for k in ('active','parity','boundary')):
                changes.append(dict(edge=e, before=old, after=new,
                                    added_to_forest=e in self.forest[old_count:]))
        self.history.append(dict(time=self.time, completed=edges,
                                 added_forest=self.forest[old_count:], active_before=old_active,
                                 active_after=self.num_active, changes=changes))


def run_case(analysis, shot, out):
    reduced=json.loads((PREVIOUS/f'reduced_{shot}.json').read_text())
    syndrome=np.zeros(analysis.graph.num_detectors,dtype=bool)
    syndrome[reduced['defect_ids']]=True
    actual=reduced['reduced']['actual_mask']
    return trace_syndrome(analysis, syndrome, actual, shot, out)


def trace_syndrome(analysis, syndrome, actual, shot, out, prefix='trace_'):
    first=analysis.uf._decode(syndrome)
    weights=analysis.weights(first.selected_edges)
    graph=analysis.regraph(weights)
    from yoked.decoders import UnionFindDecoder
    baseline=UnionFindDecoder(graph)._decode(syndrome)
    matching=analysis.matching_edges(syndrome,weights)
    assert analysis.mask(matching,syndrome)==actual!=baseline.observable_mask
    components=difference_components(graph,baseline.selected_edges,matching)
    logical=[c for c in components if c['mask']]
    tracked={e for c in logical for e in c['edges']}
    growth=TracedGrowth(graph,syndrome,tracked)
    growth.run()
    selected,mask=_peel(graph,syndrome,growth.forest)
    assert tuple(growth.forest)==baseline.forest_edges
    assert set(selected)==set(baseline.selected_edges) and mask==baseline.observable_mask
    rows=[]
    ufset,mset=set(selected),set(matching)
    for e in sorted(tracked):
        state=growth.edge(e)
        state.update(in_UF=e in ufset,in_MWPM=e in mset,in_forest=e in growth.forest)
        rows.append(state)
    cost=lambda es:math.fsum(weights[e] for e in es)
    components=[dict(c,UF_edges=sorted(set(c['edges'])&ufset),
                     MWPM_edges=sorted(set(c['edges'])&mset),
                     UF_cost=cost(set(c['edges'])&ufset),MWPM_cost=cost(set(c['edges'])&mset))
                for c in components]
    frozen=[r for r in rows if r['in_MWPM'] and not r['in_UF'] and not r['same_cluster']]
    summary=dict(shot=shot,actual=actual,UF=mask,MWPM=actual,UF_cost=cost(selected),
                 MWPM_cost=cost(matching),final_time=growth.time,batches=len(growth.history),
                 logical_components=[c for c in components if c['mask']],
                 all_difference_components=components,blocked_MWPM_edges=frozen,
                 final_yokes=[growth.cluster(d) for d in (graph.num_detectors-2,graph.num_detectors-1)])
    write(out/f'{prefix}{shot}.json',dict(summary=summary,edges=rows,history=growth.history,
                                      UF_selected=list(selected),MWPM_selected=matching,
                                      UF_forest=growth.forest))
    print(json.dumps(dict(shot=shot,UF=mask,actual=actual,UF_cost=cost(selected),
                         MWPM_cost=cost(matching),final_time=growth.time,batches=len(growth.history),
                         logical_components=[dict(mask=c['mask'],UF_edges=len(c['UF_edges']),
                                                  MWPM_edges=len(c['MWPM_edges']),UF_cost=c['UF_cost'],
                                                  MWPM_cost=c['MWPM_cost']) for c in components if c['mask']],
                         blocked=[dict(edge=r['edge'],endpoints=r['endpoints'],remaining=r['remaining'],
                                       clusters=[r['u']['fired'],r['v']['fired']],
                                       boundaries=[r['u']['boundary'],r['v']['boundary']]) for r in frozen])),flush=True)
    return graph, growth, syndrome, summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    analysis=Analysis(7)
    for shot in (44875,76890):run_case(analysis,shot,args.out)
