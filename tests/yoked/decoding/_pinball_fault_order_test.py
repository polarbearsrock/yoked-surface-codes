"""Bounded order-one safety checks for the Pinball-style adapter.

Each complete ``error(...)`` instruction is one physical-order mechanism.
Separator-delimited ``^`` components stay together instead of being split into
fake independent faults.  This test makes no order-two or distance claim.
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._pinball_decoder import PinballDecoder


def _atomic_order_one_corpus(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    detector_rows: list[np.ndarray] = []
    observable_rows: list[np.ndarray] = []
    instructions: list[str] = []
    separator_instruction_count = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors = np.zeros(dem.num_detectors, dtype=np.uint8)
        observables = np.zeros(dem.num_observables, dtype=np.uint8)
        saw_separator = False
        for target in instruction.targets_copy():
            if target.is_separator():
                saw_separator = True
            elif target.is_relative_detector_id():
                detectors[target.val] ^= 1
            elif target.is_logical_observable_id():
                observables[target.val] ^= 1
            else:  # pragma: no cover - fail closed if Stim adds a target kind.
                raise AssertionError(
                    f"unsupported target {target!r} in atomic DEM mechanism"
                )
        detector_rows.append(detectors)
        observable_rows.append(observables)
        instructions.append(str(instruction))
        separator_instruction_count += int(saw_separator)
    if not detector_rows:
        raise AssertionError("maintained DEM unexpectedly has no error mechanisms")
    return (
        np.stack(detector_rows),
        np.stack(observable_rows),
        tuple(instructions),
        separator_instruction_count,
    )


def _packed(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits, axis=1, bitorder="little")


def _unpacked_predictions(packed: np.ndarray, *, count: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1, count=count, bitorder="little")


def test_exhaustive_atomic_dem_order_one_faults_have_no_pinball_only_failure() -> None:
    circuit = yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=3,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=2,
        num_patches=2,
        remove_x_yoke=False,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    detectors, observables, instructions, separator_count = _atomic_order_one_corpus(
        dem
    )

    # Frozen counts prove that complete correlated mechanisms were retained.
    assert len(instructions) == 820
    assert separator_count == 600

    packed_detectors = _packed(detectors)
    direct = pymatching.Matching.from_detector_error_model(dem)
    direct.ensure_num_fault_ids(dem.num_observables)
    direct_predictions = _unpacked_predictions(
        np.asarray(
            direct.decode_batch(
                packed_detectors,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        ),
        count=dem.num_observables,
    )
    direct_failed = np.any(direct_predictions ^ observables, axis=1)
    assert not np.any(direct_failed)

    compiled = PinballDecoder().compile_decoder_for_dem(dem=dem)
    residual, frames, results = compiled.predecode_shots(detectors)
    pinball_predictions = _unpacked_predictions(
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed_detectors
        ),
        count=dem.num_observables,
    )
    pinball_failed = np.any(pinball_predictions ^ observables, axis=1)
    pinball_only_failures = np.flatnonzero(~direct_failed & pinball_failed)
    assert not pinball_only_failures.size, (
        "Pinball-style has order-one failures absent from direct PyMatching at "
        f"mechanisms {pinball_only_failures[:8].tolist()}: "
        f"{[instructions[k] for k in pinball_only_failures[:8]]}"
    )

    complex_indices = np.asarray(
        [index for index, result in enumerate(results) if result.complex],
        dtype=np.int64,
    )
    simple_indices = np.asarray(
        [index for index, result in enumerate(results) if not result.complex],
        dtype=np.int64,
    )
    assert complex_indices.size
    assert simple_indices.size
    np.testing.assert_array_equal(residual[complex_indices], detectors[complex_indices])
    np.testing.assert_array_equal(frames[complex_indices], 0)
    np.testing.assert_array_equal(
        pinball_predictions[complex_indices], direct_predictions[complex_indices]
    )
