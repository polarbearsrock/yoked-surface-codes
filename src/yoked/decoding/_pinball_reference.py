"""Literal functional reference for the pinned upstream Pinball kernel.

This module ports the row-major software model from ``aknapen/Pinball`` at
commit ``8f16f24b621aacfaa4f456a2aeec8df088faf3a7``.  It intentionally keeps
the upstream representation: one syndrome grid per detector layer, a flat
``d**2`` data-qubit correction vector, in-place syndrome mutation, streaming
over consecutive layers, and a final boundary pass.

It is a differential-test oracle, not the graph-native yoked-circuit decoder.
In particular, it has no notion of check basis, patches, yokes, detector error
model edges, observable masks, or residual matching.

The upstream MIT notice is retained in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal, TypeAlias

import numpy as np


PINBALL_REFERENCE_COMMIT = "8f16f24b621aacfaa4f456a2aeec8df088faf3a7"

PinballReferenceStage: TypeAlias = Literal[
    "M", "B1", "B2", "B3", "B4", "ST1", "ST2", "H", "E"
]
PINBALL_REFERENCE_STAGE_ORDER: tuple[PinballReferenceStage, ...] = (
    "M",
    "B1",
    "B2",
    "B3",
    "B4",
    "ST1",
    "ST2",
    "H",
    "E",
)


def _immutable_uint8(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8).copy()
    result.flags.writeable = False
    return result


@dataclasses.dataclass(frozen=True)
class PinballReferenceStageTrace:
    """Immutable after-stage snapshot from one streaming sweep.

    ``previous_layer`` is ``None`` for the initial synthetic zero layer.
    ``current_layer`` is ``None`` only for the final post-loop ``E`` pass.
    The syndrome snapshots contain the state *after* ``stage``.  Correction
    fields are batch-level values, which makes traces directly comparable even
    when the same physical correction cancels across different sweeps.
    """

    sweep_index: int
    previous_layer: int | None
    current_layer: int | None
    stage: PinballReferenceStage
    previous_syndrome: np.ndarray
    current_syndrome: np.ndarray | None
    correction_delta: np.ndarray
    accumulated_corrections: np.ndarray


PinballReferenceTraceHook: TypeAlias = Callable[[PinballReferenceStageTrace], None]


class PinballReference:
    """Pinned upstream Pinball functional kernel for one odd-distance patch."""

    def __init__(self, distance: int, batch_size: int):
        if isinstance(distance, bool) or not isinstance(distance, (int, np.integer)):
            raise TypeError("distance must be an integer")
        if isinstance(batch_size, bool) or not isinstance(
            batch_size, (int, np.integer)
        ):
            raise TypeError("batch_size must be an integer")
        distance = int(distance)
        batch_size = int(batch_size)
        if distance < 3 or distance % 2 == 0:
            raise ValueError("the pinned Pinball reference requires odd distance >= 3")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.distance = distance
        self.batch_size = batch_size
        self.num_syndrome_rows = distance + 1
        self.num_syndrome_cols = (distance - 1) // 2
        self.num_syndromes = self.num_syndrome_rows * self.num_syndrome_cols
        self.num_data_qubits = distance**2

    def _validate_syndrome(self, value: np.ndarray, *, name: str) -> np.ndarray:
        result = np.asarray(value)
        if result.ndim != 1 or result.shape[0] != self.num_syndromes:
            raise ValueError(
                f"{name} must have shape ({self.num_syndromes},), got "
                f"{result.shape}"
            )
        if not np.issubdtype(result.dtype, np.bool_) and not np.issubdtype(
            result.dtype, np.integer
        ):
            raise TypeError(f"{name} must contain boolean or integer bits")
        if np.any((result != 0) & (result != 1)):
            raise ValueError(f"{name} entries must be binary")
        if not result.flags.writeable:
            raise ValueError(f"{name} must be writeable because upstream mutates it")
        return result

    def _validate_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        result = np.asarray(syndrome_batch)
        expected = (self.batch_size, self.num_syndromes)
        if result.ndim != 2 or result.shape != expected:
            raise ValueError(
                f"syndrome_batch must have shape {expected}, got {result.shape}"
            )
        if not np.issubdtype(result.dtype, np.bool_) and not np.issubdtype(
            result.dtype, np.integer
        ):
            raise TypeError("syndrome_batch must contain boolean or integer bits")
        if np.any((result != 0) & (result != 1)):
            raise ValueError("syndrome_batch entries must be binary")
        if not result.flags.writeable:
            raise ValueError(
                "syndrome_batch must be writeable because upstream mutates it"
            )
        return result

    def _emit_trace(
        self,
        hook: PinballReferenceTraceHook | None,
        *,
        sweep_index: int,
        previous_layer: int | None,
        current_layer: int | None,
        stage: PinballReferenceStage,
        previous_syndrome: np.ndarray,
        current_syndrome: np.ndarray | None,
        corrections_before: np.ndarray,
        corrections: np.ndarray,
    ) -> None:
        if hook is None:
            return
        hook(
            PinballReferenceStageTrace(
                sweep_index=sweep_index,
                previous_layer=previous_layer,
                current_layer=current_layer,
                stage=stage,
                previous_syndrome=_immutable_uint8(previous_syndrome),
                current_syndrome=(
                    None
                    if current_syndrome is None
                    else _immutable_uint8(current_syndrome)
                ),
                correction_delta=_immutable_uint8(
                    np.asarray(corrections_before) ^ np.asarray(corrections)
                ),
                accumulated_corrections=_immutable_uint8(corrections),
            )
        )

    def _clear_measurement_errors(
        self, previous_syndrome: np.ndarray, current_syndrome: np.ndarray
    ) -> None:
        for index in range(self.num_syndromes):
            active = current_syndrome[index] & previous_syndrome[index]
            current_syndrome[index] ^= active
            previous_syndrome[index] ^= active

    def _clear_bulk_data_errors(
        self,
        syndrome: np.ndarray,
        corrections: np.ndarray,
        *,
        stage: Literal["B1", "B2", "B3", "B4"],
    ) -> None:
        trial = ("B1", "B2", "B3", "B4").index(stage)
        d = self.distance
        cols = self.num_syndrome_cols
        for i in range(self.num_syndrome_rows):
            if i % 2 == 0:
                continue
            for j in range(cols):
                if trial == 0:
                    parity_row = i - 1
                    parity_col = j + 1 - i % 2
                    data_row = i - 1
                    data_col = 2 * (j + 1) - i % 2
                elif trial == 1:
                    parity_row = i + 1
                    parity_col = j + 1 - i % 2
                    data_row = i
                    data_col = 2 * (j + 1) - i % 2
                elif trial == 2:
                    parity_row = i + 1
                    parity_col = j - i % 2
                    data_row = i
                    data_col = 2 * (j + 1) - i % 2 - 1
                else:
                    parity_row = i - 1
                    parity_col = j - i % 2
                    data_row = i - 1
                    data_col = 2 * (j + 1) - i % 2 - 1

                if not (0 <= parity_row < self.num_syndrome_rows):
                    continue
                if not (0 <= parity_col < cols):
                    continue
                if not (0 <= data_row < d and 0 <= data_col < d):
                    continue

                center = i * cols + j
                neighbor = parity_row * cols + parity_col
                active = syndrome[center] & syndrome[neighbor]
                if active:
                    corrections[data_row * d + data_col] ^= active
                    syndrome[center] ^= active
                    syndrome[neighbor] ^= active

    def _clear_spacetime_errors(
        self,
        previous_syndrome: np.ndarray,
        current_syndrome: np.ndarray,
        corrections: np.ndarray,
        *,
        stage: Literal["ST1", "ST2"],
    ) -> None:
        trial = ("ST1", "ST2").index(stage)
        d = self.distance
        cols = self.num_syndrome_cols
        for i in range(self.num_syndrome_rows):
            for j in range(cols):
                parity_row = i - 1
                if trial == 0:
                    parity_col = j + 1 - i % 2
                    data_col = 2 * (j + 1) - i % 2
                else:
                    parity_col = j - i % 2
                    data_col = 2 * (j + 1) - i % 2 - 1
                data_row = i - 1

                if not (0 <= parity_row < self.num_syndrome_rows):
                    continue
                if not (0 <= parity_col < cols):
                    continue
                if not (0 <= data_row < d and 0 <= data_col < d):
                    continue

                center = i * cols + j
                neighbor = parity_row * cols + parity_col
                active = current_syndrome[center] & previous_syndrome[neighbor]
                if active:
                    corrections[data_row * d + data_col] ^= active
                    current_syndrome[center] ^= active
                    previous_syndrome[neighbor] ^= active

    def _clear_hook_errors(
        self,
        previous_syndrome: np.ndarray,
        current_syndrome: np.ndarray,
        corrections: np.ndarray,
    ) -> None:
        d = self.distance
        cols = self.num_syndrome_cols
        for i in range(2, self.num_syndrome_rows):
            for j in range(cols):
                current_index = i * cols + j
                previous_index = (i - 2) * cols + j
                active = current_syndrome[current_index] & previous_syndrome[
                    previous_index
                ]
                current_syndrome[current_index] ^= active
                previous_syndrome[previous_index] ^= active

                data_col = 2 * (j + 1) - i % 2 - 1
                corrections[(i - 1) * d + data_col] ^= active
                corrections[(i - 2) * d + data_col] ^= active

    def _clear_edge_data_errors(
        self, syndrome: np.ndarray, corrections: np.ndarray
    ) -> None:
        d = self.distance
        cols = self.num_syndrome_cols

        for i in range(self.num_syndrome_rows):
            if i % 2 == 0:
                continue
            center = i * cols
            if syndrome[center]:
                corrections[(i - 1) * d] ^= 1
                syndrome[center] ^= 1

        last_col = cols - 1
        for i in range(self.num_syndrome_rows):
            if i % 2 == 1:
                continue
            center = i * cols + last_col
            if syndrome[center]:
                corrections[i * d + (d - 1)] ^= 1
                syndrome[center] ^= 1

    def _decode_pair_into(
        self,
        previous_syndrome: np.ndarray,
        current_syndrome: np.ndarray,
        corrections: np.ndarray,
        *,
        sweep_index: int,
        previous_layer: int | None,
        current_layer: int,
        trace_hook: PinballReferenceTraceHook | None,
    ) -> None:
        before = corrections.copy()
        self._clear_measurement_errors(previous_syndrome, current_syndrome)
        self._emit_trace(
            trace_hook,
            sweep_index=sweep_index,
            previous_layer=previous_layer,
            current_layer=current_layer,
            stage="M",
            previous_syndrome=previous_syndrome,
            current_syndrome=current_syndrome,
            corrections_before=before,
            corrections=corrections,
        )

        for stage in ("B1", "B2", "B3", "B4"):
            before = corrections.copy()
            self._clear_bulk_data_errors(
                current_syndrome, corrections, stage=stage
            )
            self._emit_trace(
                trace_hook,
                sweep_index=sweep_index,
                previous_layer=previous_layer,
                current_layer=current_layer,
                stage=stage,
                previous_syndrome=previous_syndrome,
                current_syndrome=current_syndrome,
                corrections_before=before,
                corrections=corrections,
            )

        for stage in ("ST1", "ST2"):
            before = corrections.copy()
            self._clear_spacetime_errors(
                previous_syndrome,
                current_syndrome,
                corrections,
                stage=stage,
            )
            self._emit_trace(
                trace_hook,
                sweep_index=sweep_index,
                previous_layer=previous_layer,
                current_layer=current_layer,
                stage=stage,
                previous_syndrome=previous_syndrome,
                current_syndrome=current_syndrome,
                corrections_before=before,
                corrections=corrections,
            )

        before = corrections.copy()
        self._clear_hook_errors(
            previous_syndrome, current_syndrome, corrections
        )
        self._emit_trace(
            trace_hook,
            sweep_index=sweep_index,
            previous_layer=previous_layer,
            current_layer=current_layer,
            stage="H",
            previous_syndrome=previous_syndrome,
            current_syndrome=current_syndrome,
            corrections_before=before,
            corrections=corrections,
        )

        before = corrections.copy()
        self._clear_edge_data_errors(previous_syndrome, corrections)
        self._emit_trace(
            trace_hook,
            sweep_index=sweep_index,
            previous_layer=previous_layer,
            current_layer=current_layer,
            stage="E",
            previous_syndrome=previous_syndrome,
            current_syndrome=current_syndrome,
            corrections_before=before,
            corrections=corrections,
        )

    def decode(
        self,
        previous_syndrome: np.ndarray,
        current_syndrome: np.ndarray,
        *,
        trace_hook: PinballReferenceTraceHook | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Ports upstream ``decode`` for one consecutive-layer pair.

        Both supplied syndromes are mutated in place.  As upstream notes,
        round-local complexity is not meaningful, so the second return value
        is always ``False``.
        """

        previous = self._validate_syndrome(
            previous_syndrome, name="previous_syndrome"
        )
        current = self._validate_syndrome(current_syndrome, name="current_syndrome")
        corrections = np.zeros(self.num_data_qubits, dtype=np.uint8)
        self._decode_pair_into(
            previous,
            current,
            corrections,
            sweep_index=0,
            previous_layer=None,
            current_layer=0,
            trace_hook=trace_hook,
        )
        return corrections, False

    def decode_batch(
        self,
        syndrome_batch: np.ndarray,
        *,
        trace_hook: PinballReferenceTraceHook | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Streams over a batch, mutating it exactly like the upstream model."""

        batch = self._validate_batch(syndrome_batch)
        corrections = np.zeros(self.num_data_qubits, dtype=np.uint8)
        previous = np.zeros(self.num_syndromes, dtype=np.uint8)
        previous_layer: int | None = None

        for current_layer in range(self.batch_size):
            current = batch[current_layer]
            self._decode_pair_into(
                previous,
                current,
                corrections,
                sweep_index=current_layer,
                previous_layer=previous_layer,
                current_layer=current_layer,
                trace_hook=trace_hook,
            )
            previous = current
            previous_layer = current_layer

        before = corrections.copy()
        self._clear_edge_data_errors(batch[-1], corrections)
        self._emit_trace(
            trace_hook,
            sweep_index=self.batch_size,
            previous_layer=self.batch_size - 1,
            current_layer=None,
            stage="E",
            previous_syndrome=batch[-1],
            current_syndrome=None,
            corrections_before=before,
            corrections=corrections,
        )

        return corrections, bool(np.any(batch))

    def decode_batch_traced(
        self, syndrome_batch: np.ndarray
    ) -> tuple[
        np.ndarray, bool, tuple[PinballReferenceStageTrace, ...]
    ]:
        """Convenience wrapper returning the complete immutable stage trace."""

        traces: list[PinballReferenceStageTrace] = []
        corrections, complex_batch = self.decode_batch(
            syndrome_batch, trace_hook=traces.append
        )
        return corrections, complex_batch, tuple(traces)

    def is_logical_error(
        self,
        errors: object,
        corrections: np.ndarray,
        observable_flip: bool | int | np.integer,
    ) -> bool:
        """Ports upstream's one-observable left-column parity check.

        ``errors`` is accepted for signature compatibility and intentionally
        unused, matching the pinned functional implementation.
        """

        del errors
        correction_array = np.asarray(corrections)
        if correction_array.shape != (self.num_data_qubits,):
            raise ValueError(
                "corrections must have shape "
                f"({self.num_data_qubits},), got {correction_array.shape}"
            )
        if np.any((correction_array != 0) & (correction_array != 1)):
            raise ValueError("corrections entries must be binary")
        prediction = np.bitwise_xor.reduce(
            correction_array[
                np.asarray(
                    [row * self.distance for row in range(self.distance)],
                    dtype=np.int64,
                )
            ]
        )
        return bool(prediction) != bool(observable_flip)


__all__ = [
    "PINBALL_REFERENCE_COMMIT",
    "PINBALL_REFERENCE_STAGE_ORDER",
    "PinballReference",
    "PinballReferenceStageTrace",
    "PinballReferenceTraceHook",
]
