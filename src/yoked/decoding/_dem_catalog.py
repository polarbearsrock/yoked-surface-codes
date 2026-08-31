"""Lossless graphlike-component catalogs for decomposed Stim DEMs.

PyMatching merges parallel graphlike mechanisms when constructing a matching
graph.  Decoder frontends that reason about individual canonical edges must
therefore validate the still-unmerged Stim representation before trusting the
merged edge's observable frame or weight.  This module keeps that validation
independent of any particular frontend policy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable

import stim


@dataclasses.dataclass(frozen=True, order=True)
class DemCatalogComponent:
    """One ``^``-separated graphlike component of a flattened DEM error."""

    instruction_index: int
    component_index: int
    probability_hex: str
    detector_boundary: tuple[int, int | None]
    observable_mask: bytes

    @property
    def probability(self) -> float:
        return float.fromhex(self.probability_hex)


@dataclasses.dataclass(frozen=True)
class DemMechanismCatalog:
    """Canonical ordered unmerged component catalog."""

    num_detectors: int
    num_observables: int
    components: tuple[DemCatalogComponent, ...]
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class DemCatalogMergeRecord:
    """Reconciliation of one parallel boundary group with one canonical edge."""

    detector_boundary: tuple[int, int | None]
    component_indices: tuple[int, ...]
    observable_mask: bytes
    effective_probability_hex: str
    recomputed_weight_hex: str
    canonical_edge_id: int
    canonical_weight_hex: str


@dataclasses.dataclass(frozen=True)
class ValidatedDemCatalog:
    """A catalog whose independent-merge interpretation matches the graph."""

    catalog: DemMechanismCatalog
    merge_records: tuple[DemCatalogMergeRecord, ...]
    merge_policy: str
    fingerprint: str


def _xor_item(items: set[int], item: int) -> None:
    if item in items:
        items.remove(item)
    else:
        items.add(item)


def _observable_mask(observable_ids: Iterable[int], *, count: int) -> bytes:
    result = bytearray((count + 7) // 8)
    for observable_id in sorted(observable_ids):
        if observable_id < 0 or observable_id >= count:
            raise ValueError(
                f"DEM component observable {observable_id} is out of range"
            )
        result[observable_id // 8] |= 1 << (observable_id % 8)
    return bytes(result)


def _catalog_fingerprint(
    *,
    num_detectors: int,
    num_observables: int,
    components: tuple[DemCatalogComponent, ...],
) -> str:
    payload = {
        "schema": "patch-uf-unmerged-dem-catalog-v1",
        "num_detectors": num_detectors,
        "num_observables": num_observables,
        "components": [
            [
                component.instruction_index,
                component.component_index,
                component.probability_hex,
                component.detector_boundary[0],
                component.detector_boundary[1],
                component.observable_mask.hex(),
            ]
            for component in components
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_dem_mechanism_catalog(dem: stim.DetectorErrorModel) -> DemMechanismCatalog:
    """Parses every flattened graphlike DEM component without merging parallels."""

    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"dem must be a stim.DetectorErrorModel, got {type(dem)!r}")
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables
    components: list[DemCatalogComponent] = []

    for instruction_index, instruction in enumerate(dem.flattened()):
        if instruction.type != "error":
            continue
        args = instruction.args_copy()
        if len(args) != 1:
            raise ValueError(
                f"DEM error instruction {instruction_index} must have one probability"
            )
        probability = float(args[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                f"DEM error instruction {instruction_index} has invalid probability"
            )

        detector_ids: set[int] = set()
        observable_ids: set[int] = set()
        component_index = 0

        def retain_component() -> None:
            nonlocal component_index
            if not detector_ids:
                if observable_ids:
                    raise ValueError(
                        "decomposed DEM contains a detector-free logical component"
                    )
                component_index += 1
                return
            if len(detector_ids) > 2:
                raise ValueError(
                    "decomposed DEM requires graphlike components with at most two "
                    f"detectors; got {sorted(detector_ids)!r}"
                )
            ordered = sorted(detector_ids)
            if ordered[0] < 0 or ordered[-1] >= num_detectors:
                raise ValueError("DEM component detector ID is out of range")
            boundary = (
                (ordered[0], None)
                if len(ordered) == 1
                else (ordered[0], ordered[1])
            )
            components.append(
                DemCatalogComponent(
                    instruction_index=instruction_index,
                    component_index=component_index,
                    probability_hex=probability.hex(),
                    detector_boundary=boundary,
                    observable_mask=_observable_mask(
                        observable_ids, count=num_observables
                    ),
                )
            )
            component_index += 1

        for target in instruction.targets_copy():
            if target.is_separator():
                retain_component()
                detector_ids = set()
                observable_ids = set()
            elif target.is_relative_detector_id():
                _xor_item(detector_ids, target.val)
            elif target.is_logical_observable_id():
                _xor_item(observable_ids, target.val)
            else:  # pragma: no cover - fail closed if Stim adds a target kind.
                raise ValueError(f"unsupported DEM component target {target!r}")
        retain_component()

    concrete = tuple(components)
    return DemMechanismCatalog(
        num_detectors=num_detectors,
        num_observables=num_observables,
        components=concrete,
        fingerprint=_catalog_fingerprint(
            num_detectors=num_detectors,
            num_observables=num_observables,
            components=concrete,
        ),
    )


def _graph_boundary_masks(graph: Any) -> dict[tuple[int, int | None], set[bytes]]:
    result: dict[tuple[int, int | None], set[bytes]] = defaultdict(set)
    for edge in graph.edges:
        result[(int(edge.source), None if edge.target is None else int(edge.target))].add(
            bytes(edge.observable_mask)
        )
    return dict(result)


def _boundary_sort_key(boundary: tuple[int, int | None]) -> tuple[int, int]:
    source, target = boundary
    return source, (2**63 - 1 if target is None else target)


def validate_unmerged_dem_catalog(dem: stim.DetectorErrorModel, graph: Any) -> None:
    """Applies the maintained Pinball-compatible frame-ambiguity validation.

    This intentionally preserves the behavior of the validator originally
    embedded in ``_pinball_v2_decoder.py``.  UF's stronger probability and
    multiplicity reconciliation is implemented by :func:`validate_uf_dem_catalog`.
    """

    catalog = parse_dem_mechanism_catalog(dem)
    if catalog.num_detectors != graph.num_detectors:
        raise ValueError("DEM/catalog detector counts disagree")
    if catalog.num_observables != graph.num_observables:
        raise ValueError("DEM/catalog observable counts disagree")
    component_masks: dict[tuple[int, int | None], set[bytes]] = defaultdict(set)
    for component in catalog.components:
        component_masks[component.detector_boundary].add(component.observable_mask)
    ambiguous = {
        boundary: sorted(mask.hex() for mask in masks)
        for boundary, masks in component_masks.items()
        if len(masks) != 1
    }
    if ambiguous:
        first_boundary = sorted(ambiguous, key=_boundary_sort_key)[0]
        raise ValueError(
            "decomposed DEM has ambiguous observable frames for boundary "
            f"{first_boundary!r}: {ambiguous[first_boundary]!r}"
        )
    graph_masks = _graph_boundary_masks(graph)
    if graph_masks != dict(component_masks):
        missing = sorted(
            set(component_masks) - set(graph_masks), key=_boundary_sort_key
        )
        extra = sorted(set(graph_masks) - set(component_masks), key=_boundary_sort_key)
        mismatched = sorted(
            (
                boundary
                for boundary in set(component_masks) & set(graph_masks)
                if component_masks[boundary] != graph_masks[boundary]
            ),
            key=_boundary_sort_key,
        )
        raise ValueError(
            "canonical matcher graph does not preserve the decomposed DEM "
            "component catalog: "
            f"missing={missing[:8]!r}, extra={extra[:8]!r}, "
            f"frame_mismatch={mismatched[:8]!r}"
        )


def _independent_odd_probability(probabilities: Iterable[float]) -> float:
    # This product form is canonical for the catalog.  Exact UF decisions use
    # the canonical graph's already-rounded binary64 weight, not this diagnostic.
    parity_factor = math.prod(1 - 2 * probability for probability in probabilities)
    return (1 - parity_factor) / 2


def _merge_fingerprint(
    catalog: DemMechanismCatalog,
    records: tuple[DemCatalogMergeRecord, ...],
    *,
    merge_policy: str,
) -> str:
    payload = {
        "schema": "patch-uf-validated-dem-catalog-v1",
        "catalog_fingerprint": catalog.fingerprint,
        "merge_policy": merge_policy,
        "records": [
            [
                record.detector_boundary[0],
                record.detector_boundary[1],
                list(record.component_indices),
                record.observable_mask.hex(),
                record.effective_probability_hex,
                record.recomputed_weight_hex,
                record.canonical_edge_id,
                record.canonical_weight_hex,
            ]
            for record in records
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_uf_dem_catalog(
    dem: stim.DetectorErrorModel,
    graph: Any,
) -> ValidatedDemCatalog:
    """Validates parallel multiplicity, frames, probabilities, and merged weights."""

    validate_unmerged_dem_catalog(dem, graph)
    catalog = parse_dem_mechanism_catalog(dem)
    groups: dict[tuple[int, int | None], list[int]] = defaultdict(list)
    for index, component in enumerate(catalog.components):
        groups[component.detector_boundary].append(index)
    graph_edges = {
        (int(edge.source), None if edge.target is None else int(edge.target)): edge
        for edge in graph.edges
    }
    if len(graph_edges) != len(graph.edges):
        raise ValueError("canonical graph contains parallel endpoint records")

    merge_policy = "independent-parity-pymatching-v2.4-binary64-v1"
    records: list[DemCatalogMergeRecord] = []
    for boundary in sorted(groups, key=_boundary_sort_key):
        indices = tuple(groups[boundary])
        components = tuple(catalog.components[index] for index in indices)
        masks = {component.observable_mask for component in components}
        if len(masks) != 1:
            raise AssertionError("frame ambiguity escaped compatibility validation")
        probability = _independent_odd_probability(
            component.probability for component in components
        )
        if not 0 < probability < 1:
            raise ValueError(
                f"parallel DEM boundary {boundary!r} has non-decodable effective probability"
            )
        recomputed_weight = math.log((1 - probability) / probability)
        edge = graph_edges[boundary]
        canonical_weight = float(edge.weight)
        if not math.isfinite(canonical_weight) or not math.isclose(
            recomputed_weight,
            canonical_weight,
            rel_tol=2e-14,
            abs_tol=2e-14,
        ):
            raise ValueError(
                "canonical matcher edge weight disagrees with independent parallel "
                f"mechanisms at boundary {boundary!r}: "
                f"recomputed={recomputed_weight.hex()}, canonical={canonical_weight.hex()}"
            )
        records.append(
            DemCatalogMergeRecord(
                detector_boundary=boundary,
                component_indices=indices,
                observable_mask=next(iter(masks)),
                effective_probability_hex=probability.hex(),
                recomputed_weight_hex=recomputed_weight.hex(),
                canonical_edge_id=int(edge.edge_id),
                canonical_weight_hex=canonical_weight.hex(),
            )
        )
    concrete = tuple(records)
    return ValidatedDemCatalog(
        catalog=catalog,
        merge_records=concrete,
        merge_policy=merge_policy,
        fingerprint=_merge_fingerprint(catalog, concrete, merge_policy=merge_policy),
    )


__all__ = [
    "DemCatalogComponent",
    "DemCatalogMergeRecord",
    "DemMechanismCatalog",
    "ValidatedDemCatalog",
    "parse_dem_mechanism_catalog",
    "validate_uf_dem_catalog",
    "validate_unmerged_dem_catalog",
]
