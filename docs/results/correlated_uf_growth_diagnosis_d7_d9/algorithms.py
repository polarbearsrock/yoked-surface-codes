"""Independent, analysis-only growth and exact minimum-cost forest correction."""
import math

import numpy as np


def scan_growth(graph, syndrome, mode='standard'):
    """Eager full-edge scan, independently implemented without UF's heap.

    Passive-yoke and frontier-normalized growth are diagnostic schedules.
    Weights, graph topology, and syndrome remain fixed.
    """
    nd = graph.num_detectors
    n = len(graph.adjacency)
    owner = np.arange(n)
    members = {v: {v} for v in range(n)}
    endpoints = np.array(graph.endpoints)
    u, v = endpoints.T
    weights = np.array([e[2] for e in graph.edges])
    grown = np.zeros(len(weights))
    now, forest, batches = 0., [], 0

    def events():
        parity = np.bincount(owner[:nd], weights=syndrome, minlength=n).astype(int) % 2
        boundary = np.bincount(owner[nd:], minlength=n) > 0
        active = parity.astype(bool) & ~boundary
        crossing = owner[u] != owner[v]
        speed = active.astype(float)
        if mode == 'frontier_normalized':
            degree = np.bincount(np.concatenate([owner[u[crossing]], owner[v[crossing]]]), minlength=n)
            speed /= np.maximum(1, degree)
        ru, rv = speed[owner[u]], speed[owner[v]]
        if mode == 'passive_yoke':
            ru = ru * ((u != nd-2) & (u != nd-1))
            rv = rv * ((v != nd-2) & (v != nd-1))
        rate = (ru + rv) * crossing
        deadlines = np.full(len(weights), np.inf)
        moving = rate > 0
        deadlines[moving] = now + np.maximum(0., weights[moving]-grown[moving]) / rate[moving]
        deadlines[crossing & (grown >= weights)] = now
        return rate, deadlines, bool(active.any())

    rate, deadlines, active = events()
    while active or float(deadlines.min()) == 0.:
        next_time = float(deadlines.min())
        if not math.isfinite(next_time):
            return dict(valid=False, reason='Unresolved odd cluster with no growing frontier')
        grown = np.minimum(weights, grown + rate * (next_time-now))
        now = next_time
        limit = now + 1e-12 + 1e-12*abs(now)
        batches += 1
        while True:
            complete = np.flatnonzero(deadlines <= limit)
            if not len(complete):
                break
            grown[complete] = weights[complete]
            for e in complete.tolist():
                a, b = int(owner[u[e]]), int(owner[v[e]])
                if a == b:
                    continue
                if (len(members[a]), -a) < (len(members[b]), -b):
                    a, b = b, a
                moved = members.pop(b)
                owner[list(moved)] = a
                members[a].update(moved)
                forest.append(e)
            rate, deadlines, active = events()
    return dict(valid=True, forest=forest, owner=owner, grown=grown, time=now, batches=batches)


def minimum_cost_forest(graph, syndrome, forest):
    """Two-state tree DP: each boundary vertex may independently absorb parity.

    DP[v, p] minimizes subtree cost given parent-edge parity p. Detector vertices
    impose their syndrome parity; boundary vertices impose no constraint.
    No truth, matching, or logical-class optimization is used.
    """
    adjacency = {}
    for e in forest:
        u, v = graph.endpoints[e]
        adjacency.setdefault(u, []).append((v, e))
        adjacency.setdefault(v, []).append((u, e))
    parent, order, roots, children = {}, [], [], {}
    for root in sorted(adjacency):
        if root in parent:
            continue
        roots.append(root)
        parent[root] = None
        pending = [root]
        while pending:
            u = pending.pop()
            order.append(u)
            children[u] = []
            for v, e in adjacency[u]:
                if v == parent[u]:
                    continue
                if v in parent:
                    raise ValueError('Allowed edges contain a cycle')
                parent[v] = u
                children[u].append((v, e))
                pending.append(v)
    assert all(int(v) in parent for v in np.flatnonzero(syndrome))
    dp, history, child_parity = {}, {}, {}
    for u in reversed(order):
        costs, records = [0., math.inf], []
        for v, e in children[u]:
            options = [dp[v][0], dp[v][1] + graph.edges[e][2]]
            next_costs, previous = [], []
            for total in range(2):
                candidates = [costs[q] + options[total ^ q] for q in range(2)]
                q = min(range(2), key=lambda q: (candidates[q], q))
                next_costs.append(candidates[q])
                previous.append(q)
            records.append((v, e, previous))
            costs = next_costs
        if u < graph.num_detectors:
            choice = [int(syndrome[u]) ^ p for p in range(2)]
        else:
            best = min(range(2), key=lambda p: (costs[p], p))
            choice = [best, best]
        dp[u] = [costs[p] for p in choice]
        child_parity[u], history[u] = choice, records
    selected, pending = [], [(r, 0) for r in roots]
    assert all(math.isfinite(dp[r][0]) for r in roots)
    while pending:
        u, incoming = pending.pop()
        parity = child_parity[u][incoming]
        for v, e, previous in reversed(history[u]):
            q = previous[parity]
            value = parity ^ q
            if value:
                selected.append(e)
            pending.append((v, value))
            parity = q
        assert parity == 0
    cost = math.fsum(graph.edges[e][2] for e in selected)
    assert math.isclose(cost, math.fsum(dp[r][0] for r in roots), rel_tol=1e-12, abs_tol=1e-9)
    return selected, cost
