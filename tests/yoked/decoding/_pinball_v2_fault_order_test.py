"""Bounded complete-mechanism order-one check for YSC Pinball V2."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._pinball_v2_decoder import PinballV2Decoder


def _complete_mechanisms(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, int]:
    detector_rows: list[np.ndarray] = []
    observable_rows: list[np.ndarray] = []
    separator_count = 0
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
                raise AssertionError(f"unsupported DEM target {target!r}")
        detector_rows.append(detectors)
        observable_rows.append(observables)
        separator_count += int(saw_separator)
    return np.stack(detector_rows), np.stack(observable_rows), separator_count


def test_complete_order_one_mechanisms_are_simple_and_correct_in_every_domain() -> None:
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
    detectors, observables, separator_count = _complete_mechanisms(dem)
    assert len(detectors) == 820
    assert separator_count == 600

    packed = np.packbits(detectors, axis=1, bitorder="little")
    baseline = pymatching.Matching.from_detector_error_model(dem)
    baseline.ensure_num_fault_ids(dem.num_observables)
    baseline_predictions = np.unpackbits(
        np.asarray(
            baseline.decode_batch(
                packed,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        ),
        axis=1,
        count=dem.num_observables,
        bitorder="little",
    )
    assert not np.any(baseline_predictions ^ observables)

    compiled = PinballV2Decoder().compile_decoder_for_dem(dem=dem)
    predictions = np.unpackbits(
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed,
        ),
        axis=1,
        count=dem.num_observables,
        bitorder="little",
    )
    _, _, results = compiled.predecode_shots(detectors)

    assert not np.any(predictions ^ observables)
    assert len(results) == 820
    assert all(not result.complex for result in results)
    assert all(
        not domain_result.complex
        for result in results
        for domain_result in result.domain_results.values()
    )
