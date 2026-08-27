"""Sinter adapter for the versioned YSC Pinball V2 policy.

The V2 core owns schedule and transaction semantics.  This module owns only
the packed Sinter boundary and composition with the complete global
PyMatching graph.  Explicit telemetry methods retain the immutable core
results; the production Sinter path deliberately discards them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sinter
import stim

from yoked.decoding._pinball_v2 import (
    CompiledPinballV2Schedule,
    PinballV2Result,
    compile_pinball_v2_schedule,
    predecode_pinball_v2,
)
from yoked.decoding._promatch_graph import CompiledPromatchGraph, compile_matching_graph
from yoked.decoding._promatch_layout import compile_layout


PINBALL_V2_DECODER_NAME = (
    "pinball-ysc-v2-cz-fullhistory-nine-stage-domainatomic-yokeedge-pymatching"
)


def _xor_set_item(items: set[int], item: int) -> None:
    if item in items:
        items.remove(item)
    else:
        items.add(item)


def _component_mask(observable_ids: set[int], *, num_observables: int) -> bytes:
    result = bytearray((num_observables + 7) // 8)
    for observable_id in sorted(observable_ids):
        if observable_id < 0 or observable_id >= num_observables:
            raise ValueError(
                f"DEM component observable {observable_id} is out of range"
            )
        result[observable_id // 8] |= 1 << (observable_id % 8)
    return bytes(result)


def _validate_unmerged_dem_catalog(
    dem: stim.DetectorErrorModel,
    graph: CompiledPromatchGraph,
) -> None:
    """Rejects correction-frame information lost during matcher construction.

    Pinball choices must not depend on whichever parallel mechanism PyMatching
    retained.  A flattened, decomposed DEM still preserves the ``^``-separated
    graphlike components of each complete physical mechanism.  V2 requires
    every component boundary to have one unambiguous observable frame and the
    resulting catalog to agree exactly with the canonical residual graph.
    """

    if dem.num_detectors != graph.num_detectors:
        raise ValueError("DEM/catalog detector counts disagree")
    if dem.num_observables != graph.num_observables:
        raise ValueError("DEM/catalog observable counts disagree")

    component_masks: dict[tuple[int, int | None], set[bytes]] = {}

    def retain_component(
        detector_ids: set[int], observable_ids: set[int]
    ) -> None:
        if not detector_ids:
            if observable_ids:
                raise ValueError(
                    "decomposed DEM contains a detector-free logical component"
                )
            return
        if len(detector_ids) > 2:
            raise ValueError(
                "Pinball V2 requires graphlike DEM components with at most two "
                f"detectors; got {sorted(detector_ids)!r}"
            )
        ordered = sorted(detector_ids)
        boundary = (
            (ordered[0], None)
            if len(ordered) == 1
            else (ordered[0], ordered[1])
        )
        mask = _component_mask(
            observable_ids,
            num_observables=dem.num_observables,
        )
        component_masks.setdefault(boundary, set()).add(mask)

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detector_ids: set[int] = set()
        observable_ids: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_separator():
                retain_component(detector_ids, observable_ids)
                detector_ids = set()
                observable_ids = set()
            elif target.is_relative_detector_id():
                _xor_set_item(detector_ids, target.val)
            elif target.is_logical_observable_id():
                _xor_set_item(observable_ids, target.val)
            else:  # pragma: no cover - fail closed if Stim adds a target kind.
                raise ValueError(f"unsupported DEM component target {target!r}")
        retain_component(detector_ids, observable_ids)

    ambiguous = {
        boundary: sorted(mask.hex() for mask in masks)
        for boundary, masks in component_masks.items()
        if len(masks) != 1
    }
    if ambiguous:
        first_boundary = sorted(ambiguous)[0]
        raise ValueError(
            "decomposed DEM has ambiguous observable frames for boundary "
            f"{first_boundary!r}: {ambiguous[first_boundary]!r}"
        )

    graph_masks: dict[tuple[int, int | None], set[bytes]] = {}
    for edge in graph.edges:
        boundary = (edge.source, edge.target)
        graph_masks.setdefault(boundary, set()).add(edge.observable_mask)
    if graph_masks != component_masks:
        missing = sorted(set(component_masks) - set(graph_masks))
        extra = sorted(set(graph_masks) - set(component_masks))
        mismatched = sorted(
            boundary
            for boundary in set(component_masks) & set(graph_masks)
            if component_masks[boundary] != graph_masks[boundary]
        )
        raise ValueError(
            "canonical matcher graph does not preserve the decomposed DEM "
            "component catalog: "
            f"missing={missing[:8]!r}, extra={extra[:8]!r}, "
            f"frame_mismatch={mismatched[:8]!r}"
        )


def _validate_packed_input(data: np.ndarray, *, num_detectors: int) -> np.ndarray:
    result = np.asarray(data)
    expected_width = (num_detectors + 7) // 8
    if result.dtype != np.uint8:
        raise ValueError(
            f"packed detector data must have dtype uint8, got {result.dtype}"
        )
    if result.ndim != 2:
        raise ValueError(f"packed detector data must be 2-D, got shape {result.shape}")
    if result.shape[1] != expected_width:
        raise ValueError(
            "packed detector width mismatch: "
            f"expected {expected_width}, got {result.shape[1]}"
        )
    return result


def _pack_observable_frames(
    frames: np.ndarray,
    *,
    num_observables: int,
) -> np.ndarray:
    width = (num_observables + 7) // 8
    if width == 0:
        return np.zeros((frames.shape[0], 0), dtype=np.uint8)
    result = np.packbits(frames, axis=1, bitorder="little")
    if num_observables % 8:
        result[:, -1] &= (1 << (num_observables % 8)) - 1
    return result


@dataclass(frozen=True)
class PinballV2Decoder(sinter.Decoder):
    """Parameter-free factory for the versioned YSC Pinball V2 policy."""

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> "CompiledPinballV2Decoder":
        layout = compile_layout(dem, mode="fullhistory")
        graph = compile_matching_graph(dem, layout, require_zero_frame=False)
        _validate_unmerged_dem_catalog(dem, graph)
        schedule = compile_pinball_v2_schedule(graph)
        return CompiledPinballV2Decoder(
            graph=graph,
            schedule=schedule,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )


@dataclass
class CompiledPinballV2Decoder(sinter.CompiledDecoder):
    """Pinball V2 preprocessing followed by one global residual MWPM batch."""

    graph: CompiledPromatchGraph
    schedule: CompiledPinballV2Schedule
    num_detectors: int
    num_observables: int

    def predecode_shots(
        self,
        unpacked_detection_events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[PinballV2Result, ...]]:
        """Predecodes unpacked shots and retains immutable per-shot telemetry."""

        return self._predecode_shots(
            unpacked_detection_events,
            retain_results=True,
        )

    def _predecode_shots(
        self,
        unpacked_detection_events: np.ndarray,
        *,
        retain_results: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[PinballV2Result, ...]]:
        raw = np.asarray(unpacked_detection_events)
        if raw.ndim != 2:
            raise ValueError("unpacked_detection_events must be 2-D")
        if raw.shape[1] != self.num_detectors:
            raise ValueError("unpacked detector width mismatch")

        residual = np.empty(raw.shape, dtype=np.uint8)
        frames = np.zeros((raw.shape[0], self.num_observables), dtype=np.uint8)
        results: list[PinballV2Result] | None = [] if retain_results else None
        for shot_index, shot in enumerate(raw):
            result = predecode_pinball_v2(
                self.graph,
                self.schedule,
                shot,
                _schedule_is_validated=True,
            )
            if result.residual_syndrome.shape != (self.num_detectors,):
                raise ValueError("Pinball V2 core returned a residual width mismatch")
            if result.observable_frame.shape != (self.num_observables,):
                raise ValueError("Pinball V2 core returned an observable width mismatch")
            residual[shot_index] = result.residual_syndrome
            frames[shot_index] = result.observable_frame
            if results is not None:
                results.append(result)
        return residual, frames, () if results is None else tuple(results)

    def decode_shots_bit_packed_with_telemetry(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> tuple[np.ndarray, tuple[PinballV2Result, ...]]:
        """Decodes one packed batch and explicitly retains V2 core telemetry."""

        return self._decode_shots_bit_packed(
            bit_packed_detection_event_data=bit_packed_detection_event_data,
            retain_results=True,
        )

    def _decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
        retain_results: bool,
    ) -> tuple[np.ndarray, tuple[PinballV2Result, ...]]:
        packed_input = _validate_packed_input(
            bit_packed_detection_event_data,
            num_detectors=self.num_detectors,
        )
        if packed_input.shape[0] == 0:
            return (
                np.zeros(
                    (0, (self.num_observables + 7) // 8),
                    dtype=np.uint8,
                ),
                (),
            )

        unpacked = np.unpackbits(
            packed_input,
            axis=1,
            count=self.num_detectors,
            bitorder="little",
        )
        residual, frames, results = self._predecode_shots(
            unpacked,
            retain_results=retain_results,
        )
        residual_packed = np.packbits(residual, axis=1, bitorder="little")
        predictions = self.graph.matcher.decode_batch(
            residual_packed,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        predictions = np.asarray(predictions, dtype=np.uint8)
        predictions ^= _pack_observable_frames(
            frames,
            num_observables=self.num_observables,
        )
        if self.num_observables % 8 and predictions.shape[1]:
            predictions[:, -1] &= (1 << (self.num_observables % 8)) - 1
        return predictions, results

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        predictions, _ = self._decode_shots_bit_packed(
            bit_packed_detection_event_data=bit_packed_detection_event_data,
            retain_results=False,
        )
        return predictions


__all__ = [
    "PINBALL_V2_DECODER_NAME",
    "CompiledPinballV2Decoder",
    "PinballV2Decoder",
]
