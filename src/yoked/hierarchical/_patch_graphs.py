"""Split the six-patch yoked decoding problem into per-patch graphs.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 4.

The six-patch DEM has two yoke detectors, the last two, each equal to the
parity of one sector's six observables (X: even observable indices, Z: odd).
Every error component that flips observable o(i, s) also contains the yoke
detector of sector s, so the yoke is a hub joining all patches. Removing the
yoke detectors leaves one connected component per (patch, sector), each
carrying exactly one observable.

For each patch i this module builds
  * ``local_dem``:   the patch's error mechanisms with local detector ids and
                     local observable ids (0 = X, 1 = Z), yoke targets dropped,
                     component structure and probabilities kept so that
                     correlation rules can be derived from it;
  * ``graph``:       DecodingGraph.from_dem(local_dem), the check-free graph
                     that L1 decodes; it has one component per sector;
  * ``check_graph``: the same edge ids and weights, with every
                     observable-flipping boundary edge re-targeted to the
                     sector's check vertex (local ids n and n + 1). Setting a
                     check bit forces a matching into that logical class, as
                     the repository's gap circuits do with their check detector.

Invariants checked by ``_patch_graphs_test.py``: the split is lossless
(merging check vertices back into the yoke vertices reproduces the six-patch
graph), every component keeps one or two physical detectors, an
observable-flipping component keeps exactly one physical detector (checked on
the raw DEM component in ``_validate_component``, before the DEM is ever
imported into a graph, per spec section 4), a component's yoke membership
equals its observable's sector, an observable-flipping edge is a boundary
edge, and one mechanism never spans two patches.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import stim

from yoked.decoders._graph import DecodingGraph

NUM_SECTORS = 2
"""Sector 0 is X (observable 2i), sector 1 is Z (observable 2i + 1)."""


@dataclass(frozen=True)
class _Component:
    """One graphlike piece of a DEM error mechanism, global ids, yokes included."""
    detectors: tuple[int, ...]
    observables: tuple[int, ...]   # flipped an odd number of times


@dataclass(frozen=True)
class _Instruction:
    """One DEM error mechanism after flattening."""
    probability: float
    components: tuple[_Component, ...]


@dataclass(frozen=True)
class PatchGraph:
    """Everything L1 needs for one surface-code patch.

    Fields:
      patch_index: i in 0..num_patches - 1.
      global_detector_ids: sorted global ids of the patch's physical detectors;
        local id k is global id ``global_detector_ids[k]``.
      observable_ids: (o(i, X), o(i, Z)) as global observable indices.
      local_dem: the patch's mechanisms with local detector and observable ids.
      graph: check-free DecodingGraph with two connected components (X, Z).
      check_graph: DecodingGraph whose vertices n and n + 1 (n = detector count)
        are the X and Z check vertices. Edge ids match ``graph`` one to one.
    """
    patch_index: int
    global_detector_ids: tuple[int, ...]
    observable_ids: tuple[int, int]
    local_dem: stim.DetectorErrorModel
    graph: DecodingGraph
    check_graph: DecodingGraph

    @property
    def num_detectors(self) -> int:
        return len(self.global_detector_ids)

    @property
    def check_vertices(self) -> tuple[int, int]:
        """Local ids of the X and Z check vertices in ``check_graph``."""
        return self.num_detectors, self.num_detectors + 1

    def local_syndromes(self, global_syndromes: np.ndarray) -> np.ndarray:
        """Gather this patch's detector bits from (n_d,) or (shots, n_d) input."""
        return np.asarray(global_syndromes)[..., list(self.global_detector_ids)]


@dataclass(frozen=True)
class PatchGraphs:
    """The per-patch view of one six-patch yoked DEM.

    Fields: ``patches`` in patch order; ``yoke_detector_ids`` = (X yoke, Z yoke)
    global ids; ``num_detectors`` and ``num_observables`` of the six-patch DEM.
    """
    patches: tuple[PatchGraph, ...]
    yoke_detector_ids: tuple[int, int]
    num_detectors: int
    num_observables: int

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> PatchGraph:
        return self.patches[index]

    def __iter__(self) -> Iterator[PatchGraph]:
        return iter(self.patches)

    def local_syndromes(self, global_syndromes: np.ndarray) -> np.ndarray:
        """Stack every patch's local syndrome: shape (..., patches, n_local)."""
        return np.stack([patch.local_syndromes(global_syndromes) for patch in self.patches], axis=-2)

    @classmethod
    def from_yoked_dem(cls, dem: stim.DetectorErrorModel, *, num_patches: int) -> PatchGraphs:
        num_detectors, num_observables = dem.num_detectors, dem.num_observables
        if num_observables != NUM_SECTORS * num_patches:
            raise ValueError(
                f'Expected {NUM_SECTORS * num_patches} observables for {num_patches} patches, found {num_observables}')
        # The circuit appends the X yoke detector first, then the Z yoke detector.
        yoke = (num_detectors - 2, num_detectors - 1)
        instructions = _parse_errors(dem)
        labels = _component_labels(instructions, num_physical=num_detectors - 2, yoke=yoke)
        observable_of_label = _observable_of_label(instructions, labels, yoke, num_observables)
        label_of_observable = {observable: label for label, observable in observable_of_label.items()}
        patches = []
        for i in range(num_patches):
            sector_labels = tuple(label_of_observable[NUM_SECTORS * i + s] for s in range(NUM_SECTORS))
            detector_ids = tuple(d for d in range(num_detectors - 2) if labels[d] in sector_labels)
            patches.append(_build_patch(i, detector_ids, instructions, labels, sector_labels, yoke))
        return cls(tuple(patches), yoke, num_detectors, num_observables)


def _parse_errors(dem: stim.DetectorErrorModel) -> list[_Instruction]:
    instructions = []
    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        probability, = instruction.args_copy()
        if probability == 0:
            continue
        components = []
        for group in instruction.target_groups():
            detectors = tuple(t.val for t in group if t.is_relative_detector_id())
            if len(set(detectors)) != len(detectors):
                raise ValueError('Repeated detector targets in a DEM component are unsupported')
            counts = collections.Counter(t.val for t in group if t.is_logical_observable_id())
            observables = tuple(sorted(o for o, n in counts.items() if n % 2))
            components.append(_Component(detectors, observables))
        instructions.append(_Instruction(probability, tuple(components)))
    return instructions


def _physical(component: _Component, yoke: tuple[int, int]) -> tuple[int, ...]:
    return tuple(d for d in component.detectors if d not in yoke)


def _validate_component(component: _Component, yoke: tuple[int, int]) -> tuple[int, ...]:
    """Return the physical detectors of a component whose shape the split supports."""
    physical = _physical(component, yoke)
    if not 1 <= len(physical) <= 2:
        raise ValueError(f'Component {component} must keep one or two physical detectors after removing the yoke')
    if len(component.observables) > 1:
        raise ValueError(f'Component {component} flips more than one observable')
    # This is the real gate for observable-flipping shape (spec section 4): an
    # observable-flipping component must keep exactly one physical detector,
    # so its boundary edge can be re-targeted to a single check vertex. Catch
    # it here, on the raw component, rather than relying on the graph built
    # from the imported DEM: DecodingGraph.from_dem's audit merges parallel
    # components that share a detector-pair key using a first-label rule, so
    # a malformed two-physical-detector component can silently merge into an
    # existing plain edge and never surface as its own graph edge.
    if component.observables and len(physical) != 1:
        raise ValueError(
            f'Observable-flipping component {component} must keep exactly one physical detector, found {len(physical)}')
    present = {d for d in component.detectors if d in yoke}
    implied = {yoke[o % NUM_SECTORS] for o in component.observables}
    if present != implied:
        raise ValueError(f'Component {component} has yoke membership {present} but its observables imply {implied}')
    return physical


def _component_labels(instructions: list[_Instruction], *, num_physical: int, yoke: tuple[int, int]) -> list[int]:
    """Label physical detectors by connected component of the yoke-free graph."""
    parent = list(range(num_physical))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for instruction in instructions:
        for component in instruction.components:
            physical = _validate_component(component, yoke)
            if len(physical) == 2:
                parent[find(physical[0])] = find(physical[1])
    return [find(v) for v in range(num_physical)]


def _observable_of_label(
        instructions: list[_Instruction], labels: list[int], yoke: tuple[int, int], num_observables: int,
) -> dict[int, int]:
    """Map each component label to the single observable it carries."""
    observable_of_label: dict[int, int] = {}
    for instruction in instructions:
        for component in instruction.components:
            if not component.observables:
                continue
            observable, = component.observables
            label = labels[_physical(component, yoke)[0]]
            if observable_of_label.setdefault(label, observable) != observable:
                raise ValueError(f'One component carries observables {observable_of_label[label]} and {observable}')
    if len(set(labels)) != num_observables:
        raise ValueError(f'Expected {num_observables} components after removing the yoke, found {len(set(labels))}')
    if set(observable_of_label.values()) != set(range(num_observables)) or len(observable_of_label) != num_observables:
        raise ValueError('Each observable must belong to exactly one component')
    return observable_of_label


def _build_patch(
        patch_index: int, detector_ids: tuple[int, ...], instructions: list[_Instruction],
        labels: list[int], sector_labels: tuple[int, ...], yoke: tuple[int, int],
) -> PatchGraph:
    local_id = {d: k for k, d in enumerate(detector_ids)}
    local_dem = stim.DetectorErrorModel()
    for instruction in instructions:
        owned = {labels[d] in sector_labels for component in instruction.components
                 for d in _physical(component, yoke)}
        if owned == {False}:
            continue
        if owned != {True}:
            raise ValueError('A DEM error mechanism spans more than one patch')
        targets = []
        for k, component in enumerate(instruction.components):
            if k:
                targets.append(stim.DemTarget.separator())
            targets.extend(stim.DemTarget.relative_detector_id(local_id[d]) for d in _physical(component, yoke))
            targets.extend(stim.DemTarget.logical_observable_id(o % NUM_SECTORS) for o in component.observables)
        local_dem.append('error', instruction.probability, targets)
    # Declare the counts so that a detector or observable absent from every
    # mechanism still exists in the local DEM.
    local_dem.append('detector', [], [stim.DemTarget.relative_detector_id(len(detector_ids) - 1)])
    local_dem.append('logical_observable', [], [stim.DemTarget.logical_observable_id(NUM_SECTORS - 1)])
    graph = DecodingGraph.from_dem(local_dem)
    observable_ids = (NUM_SECTORS * patch_index, NUM_SECTORS * patch_index + 1)
    return PatchGraph(patch_index, detector_ids, observable_ids, local_dem, graph, _with_check_vertices(graph))


def _with_check_vertices(graph: DecodingGraph) -> DecodingGraph:
    """Re-target observable-flipping boundary edges to the sector's check vertex.

    The real gate for observable-flipping shape is ``_validate_component``,
    run on the raw DEM components before import. The two raises below are
    cheap defensive guards against the imported graph disagreeing with that
    raw-component shape; they are not expected to trigger given a DEM that
    already passed ``_validate_component``.
    """
    check = (graph.num_detectors, graph.num_detectors + 1)
    edges = []
    for u, v, weight, mask in graph.edges:
        if mask == 0:
            edges.append((u, v, weight, mask))
            continue
        if v is not None:
            raise ValueError('An observable-flipping edge must be a boundary edge')  # defensive
        if mask not in (1, 2):
            raise ValueError('An edge may flip only one of the two patch observables')  # defensive
        edges.append((u, check[mask.bit_length() - 1], weight, mask))
    return DecodingGraph(graph.num_detectors + NUM_SECTORS, graph.num_observables, edges)
