"""Sinter adapter for whole-shot Pinball-style preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sinter
import stim

from yoked.decoding._pinball import (
    CompiledPinballSchedule,
    PinballResult,
    compile_pinball_schedule,
    predecode_pinball,
)
from yoked.decoding._promatch_decoder import (
    _pack_observable_frames,
    _validate_packed_input,
)
from yoked.decoding._promatch_graph import CompiledPromatchGraph, compile_matching_graph
from yoked.decoding._promatch_layout import compile_layout


PINBALL_DECODER_NAME = (
    "pinball-style-v1-fullhistory-nine-stage-wholeshotrollback-pymatching"
)


@dataclass(frozen=True)
class PinballDecoder(sinter.Decoder):
    """Parameter-free Sinter factory for the versioned Pinball-style policy."""

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> "CompiledPinballDecoder":
        layout = compile_layout(dem, mode="fullhistory")
        graph = compile_matching_graph(dem, layout, require_zero_frame=False)
        schedule = compile_pinball_schedule(graph)
        return CompiledPinballDecoder(
            graph=graph,
            schedule=schedule,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )


@dataclass
class CompiledPinballDecoder(sinter.CompiledDecoder):
    """Compiled Pinball preprocessing followed by one residual MWPM batch."""

    graph: CompiledPromatchGraph
    schedule: CompiledPinballSchedule
    num_detectors: int
    num_observables: int

    def predecode_shots(
        self,
        unpacked_detection_events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[PinballResult, ...]]:
        """Predecodes unpacked shots and retains one result record per shot."""

        return self._predecode_shots(
            unpacked_detection_events,
            retain_results=True,
        )

    def _predecode_shots(
        self,
        unpacked_detection_events: np.ndarray,
        *,
        retain_results: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[PinballResult, ...]]:
        if unpacked_detection_events.ndim != 2:
            raise ValueError("unpacked_detection_events must be 2-D")
        if unpacked_detection_events.shape[1] != self.num_detectors:
            raise ValueError("unpacked detector width mismatch")

        residual = np.empty_like(unpacked_detection_events, dtype=np.uint8)
        frames = np.zeros(
            (unpacked_detection_events.shape[0], self.num_observables),
            dtype=np.uint8,
        )
        results: list[PinballResult] | None = [] if retain_results else None
        for shot_index, shot in enumerate(unpacked_detection_events):
            result = predecode_pinball(self.graph, self.schedule, shot)
            residual[shot_index] = result.residual_syndrome
            frames[shot_index] = result.observable_frame
            if results is not None:
                results.append(result)
        return residual, frames, () if results is None else tuple(results)

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        packed_input = _validate_packed_input(
            bit_packed_detection_event_data,
            num_detectors=self.num_detectors,
        )
        if packed_input.shape[0] == 0:
            return np.zeros(
                (0, (self.num_observables + 7) // 8),
                dtype=np.uint8,
            )

        unpacked = np.unpackbits(
            packed_input,
            axis=1,
            count=self.num_detectors,
            bitorder="little",
        )
        residual, frames, _ = self._predecode_shots(
            unpacked,
            retain_results=False,
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
        return predictions
