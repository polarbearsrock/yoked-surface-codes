from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import sinter
import stim

from yoked.decoding._promatch import PrematchResult, predecode
from yoked.decoding._promatch_graph import CompiledPromatchGraph, compile_matching_graph
from yoked.decoding._promatch_layout import compile_layout


BoundaryPolicy = Literal["disabled", "odd-parity"]
ObservablePolicy = Literal["zero-frame", "path-zero", "any"]
DomainMode = Literal["windowd", "fullhistory"]


@dataclass(frozen=True)
class PromatchDecoder(sinter.Decoder):
    """Sinter adapter for an L1-local ProMatch-style predecoder."""

    residual_hw_limit: int = 10
    domain_mode: DomainMode = "windowd"
    boundary_policy: BoundaryPolicy = "disabled"
    observable_policy: ObservablePolicy = "zero-frame"

    def __post_init__(self) -> None:
        if isinstance(self.residual_hw_limit, bool) or not isinstance(
            self.residual_hw_limit, (int, np.integer)
        ):
            raise TypeError("residual_hw_limit must be an integer")
        if self.residual_hw_limit < 0:
            raise ValueError(
                f"residual_hw_limit must be non-negative, got {self.residual_hw_limit}"
            )
        if self.domain_mode not in {"windowd", "fullhistory"}:
            raise ValueError(f"unsupported domain_mode={self.domain_mode!r}")
        if self.boundary_policy not in {"disabled", "odd-parity"}:
            raise ValueError(f"unsupported boundary_policy={self.boundary_policy!r}")
        if self.observable_policy not in {"zero-frame", "path-zero", "any"}:
            raise ValueError(f"unsupported observable_policy={self.observable_policy!r}")

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> "CompiledPromatchDecoder":
        layout = compile_layout(dem, mode=self.domain_mode)
        graph = compile_matching_graph(
            dem,
            layout,
            require_zero_frame=self.observable_policy == "zero-frame",
        )
        return CompiledPromatchDecoder(
            graph=graph,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
            residual_hw_limit=self.residual_hw_limit,
            boundary_policy=self.boundary_policy,
            observable_policy=self.observable_policy,
        )


@dataclass(frozen=True)
class IdentityWrappedPyMatchingDecoder(sinter.Decoder):
    """No-op adapter controlling for unpack/classify/repack overhead."""

    domain_mode: DomainMode = "windowd"

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> "CompiledIdentityWrappedPyMatchingDecoder":
        layout = compile_layout(dem, mode=self.domain_mode)
        graph = compile_matching_graph(
            dem,
            layout,
            require_zero_frame=False,
        )
        return CompiledIdentityWrappedPyMatchingDecoder(
            graph=graph,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )


def _validate_packed_input(
    data: np.ndarray,
    *,
    num_detectors: int,
) -> np.ndarray:
    result = np.asarray(data)
    expected_width = (num_detectors + 7) // 8
    if result.dtype != np.uint8:
        raise ValueError(f"packed detector data must have dtype uint8, got {result.dtype}")
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
    packed = np.packbits(frames, axis=1, bitorder="little")
    if num_observables % 8:
        packed[:, -1] &= (1 << (num_observables % 8)) - 1
    return packed


@dataclass
class CompiledPromatchDecoder(sinter.CompiledDecoder):
    graph: CompiledPromatchGraph
    num_detectors: int
    num_observables: int
    residual_hw_limit: int
    boundary_policy: BoundaryPolicy
    observable_policy: ObservablePolicy

    def predecode_shots(
        self,
        unpacked_detection_events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[PrematchResult, ...]]:
        return self._predecode_shots(
            unpacked_detection_events,
            retain_results=True,
        )

    def _predecode_shots(
        self,
        unpacked_detection_events: np.ndarray,
        *,
        retain_results: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[PrematchResult, ...]]:
        if unpacked_detection_events.ndim != 2:
            raise ValueError("unpacked_detection_events must be 2-D")
        if unpacked_detection_events.shape[1] != self.num_detectors:
            raise ValueError("unpacked detector width mismatch")

        residual = np.empty_like(unpacked_detection_events, dtype=np.uint8)
        frames = np.zeros(
            (unpacked_detection_events.shape[0], self.num_observables),
            dtype=np.uint8,
        )
        results: list[PrematchResult] | None = [] if retain_results else None
        for shot_index, shot in enumerate(unpacked_detection_events):
            result = predecode(
                self.graph,
                shot,
                residual_hw_limit=self.residual_hw_limit,
                boundary_policy=self.boundary_policy,
                observable_policy=self.observable_policy,
            )
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
        # The production Sinter path deliberately does not retain unbounded
        # per-shot telemetry. The paired scientific harness calls
        # ``predecode_shots`` explicitly when it needs replayable details.
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


@dataclass
class CompiledIdentityWrappedPyMatchingDecoder(sinter.CompiledDecoder):
    graph: CompiledPromatchGraph
    num_detectors: int
    num_observables: int

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
        # Deliberately exercise the same conversion and domain traversal path as
        # the treatment while committing no correction.
        unpacked = np.unpackbits(
            packed_input,
            axis=1,
            count=self.num_detectors,
            bitorder="little",
        )
        # Match PU's shot-first deterministic domain traversal while committing
        # no correction and retaining no telemetry.
        for shot in unpacked:
            for domain in sorted(self.graph.domain_graphs):
                detector_ids = self.graph.domain_graphs[domain].detector_ids
                sum(int(shot[detector_id]) for detector_id in detector_ids)
        repacked = np.packbits(unpacked, axis=1, bitorder="little")
        predictions = self.graph.matcher.decode_batch(
            repacked,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        predictions = np.asarray(predictions, dtype=np.uint8)
        if self.num_observables % 8 and predictions.shape[1]:
            predictions[:, -1] &= (1 << (self.num_observables % 8)) - 1
        return predictions
