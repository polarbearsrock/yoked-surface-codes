from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import stim


Edge = tuple[int, int | None, float, int]


def _index(value: int, name: str) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError()
        return operator.index(value)
    except TypeError as ex:
        raise ValueError(f'{name} must be an integer') from ex


@dataclass(frozen=True)
class DecodingGraph:
    """Immutable weighted detector graph.

    Edges are (u, v, weight, observable_mask), with v=None for a boundary.
    Directly supplied parallel edges remain distinct. Each boundary edge gets
    its own internal terminal, numbered after the real detectors.
    """

    num_detectors: int
    num_observables: int
    edges: tuple[Edge, ...] | Iterable[Edge]
    endpoints: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    adjacency: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nd = _index(self.num_detectors, 'num_detectors')
        no = _index(self.num_observables, 'num_observables')
        if nd < 0 or no < 0:
            raise ValueError('Detector and observable counts must be nonnegative')
        edges = []
        endpoints = []
        adjacency: list[list[int]] = [[] for _ in range(nd)]
        for edge_id, edge in enumerate(self.edges):
            if len(edge) != 4:
                raise ValueError('Each edge must be (u, v, weight, observable_mask)')
            u, v, weight, mask = edge
            u = _index(u, 'Detector endpoint')
            v = None if v is None else _index(v, 'Detector endpoint')
            mask = _index(mask, 'Observable mask')
            weight = float(weight)
            if not 0 <= u < nd or (v is not None and not 0 <= v < nd):
                raise ValueError(f'Edge {edge_id} has an out-of-range detector endpoint')
            if u == v:
                raise ValueError(f'Edge {edge_id} is a self-loop')
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('Edge weights must be finite and nonnegative')
            if mask < 0 or mask.bit_length() > no:
                raise ValueError(f'Edge {edge_id} has an out-of-range observable mask')
            edges.append((u, v, weight, mask))
            if v is None:
                target = len(adjacency)
                adjacency.append([])
            else:
                target = v
            endpoints.append((u, target))
            adjacency[u].append(edge_id)
            adjacency[target].append(edge_id)
        object.__setattr__(self, 'num_detectors', nd)
        object.__setattr__(self, 'num_observables', no)
        object.__setattr__(self, 'edges', tuple(edges))
        object.__setattr__(self, 'endpoints', tuple(endpoints))
        object.__setattr__(self, 'adjacency', tuple(tuple(a) for a in adjacency))

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> DecodingGraph:
        """Import a decomposed DEM, checking for silently dropped components.

        PyMatching is used only for graph export, never for a matching solve.
        Parallel DEM components use its independent merge and first-label rule.
        """
        import pymatching

        audited = _audit_dem(dem)
        exported = pymatching.Matching.from_detector_error_model(dem).edges()
        edges = []
        keys = set()
        for u, v, data in exported:
            key = (u, v) if v is None else (min(u, v), max(u, v))
            if key not in audited or key in keys:
                raise ValueError('DEM audit and graph export disagree on edge keys')
            keys.add(key)
            probability, expected_mask = audited[key]
            if not 0 < probability < 1:
                raise ValueError('DEM produces a nonfinite edge weight')
            expected_weight = math.log1p(-probability) - math.log(probability)
            mask = sum(1 << k for k in data['fault_ids'])
            if mask != expected_mask or not math.isclose(
                data['weight'], expected_weight, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError('DEM audit and graph export disagree on labels or weights')
            edges.append((u, v, data['weight'], mask))
        if keys != audited.keys():
            raise ValueError('DEM audit and graph export disagree on edge keys')
        return cls(dem.num_detectors, dem.num_observables, edges)


def _audit_dem(dem: stim.DetectorErrorModel) -> dict[tuple[int, int | None], tuple[float, int]]:
    audited: dict[tuple[int, int | None], tuple[float, int]] = {}

    def add_component(detectors: list[int], mask: int, probability: float) -> None:
        if len(set(detectors)) != len(detectors):
            raise ValueError('Repeated detector targets in a DEM component are unsupported')
        if len(detectors) > 2:
            raise ValueError('DEM components must have at most two detectors; decompose errors first')
        if not detectors:
            if mask:
                raise ValueError('Observable-only DEM components are unsupported')
            return
        key = (detectors[0], None) if len(detectors) == 1 else tuple(sorted(detectors))
        if key in audited:
            previous, first_mask = audited[key]
            probability = previous * (1 - probability) + (1 - previous) * probability
            mask = first_mask
        audited[key] = probability, mask

    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        probability = instruction.args_copy()[0]
        if probability == 0:
            continue
        detectors = []
        mask = 0
        for target in instruction.targets_copy():
            if target.is_separator():
                add_component(detectors, mask, probability)
                detectors = []
                mask = 0
            elif target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                mask ^= 1 << target.val
        add_component(detectors, mask, probability)
    return audited
