"""Shared graph-construction helpers for the oracle test modules.

These were previously duplicated (byte-identically for ``fault_mask``, and
as near-identical loops for the pymatching-matcher-to-``Edge`` conversion)
across the full-graph, replay, and policy-audit test files.
"""

from __future__ import annotations

import pymatching

from yoked.decoding._promatch_graph import Edge
from yoked.decoding._promatch_layout import DetectorRole


def fault_mask(fault_ids: set[int], *, num_observables: int) -> bytes:
    """Pack observable fault ids into the little-endian mask ``Edge`` expects."""
    result = bytearray((num_observables + 7) // 8)
    for fault_id in fault_ids:
        result[fault_id // 8] |= 1 << (fault_id % 8)
    return bytes(result)


def edges_from_matcher(
    matcher: pymatching.Matching,
    *,
    num_observables: int,
    role: DetectorRole,
) -> tuple[Edge, ...]:
    """Convert ``matcher.edges()`` into canonical ``Edge`` tuples.

    Endpoints are normalized so ``source <= target``; ``target=None`` marks a
    true boundary edge. Every detector endpoint is assigned ``role``.
    """
    matcher.ensure_num_fault_ids(num_observables)
    edges: list[Edge] = []
    for edge_id, (raw_source, raw_target, data) in enumerate(matcher.edges()):
        source = int(raw_source)
        target = None if raw_target is None else int(raw_target)
        if target is not None and target < source:
            source, target = target, source
        edges.append(
            Edge(
                edge_id=edge_id,
                source=source,
                target=target,
                weight=float(data["weight"]),
                observable_mask=fault_mask(
                    set(data.get("fault_ids", ())),
                    num_observables=num_observables,
                ),
                source_role=role,
                target_role=None if target is None else role,
            )
        )
    return tuple(edges)
