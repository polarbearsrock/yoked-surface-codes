"""Mapped differential checks between the physical reference and YSC V2."""

from __future__ import annotations

import numpy as np
import pytest

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._pinball_reference import PinballReference
from yoked.decoding._pinball_v2 import (
    PinballV2PauliTarget,
    compile_pinball_v2_schedule,
    predecode_pinball_v2,
)
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1FullHistoryDomain,
    L1TerminalDetector,
    compile_layout,
)


def _mapped_problem(*, distance: int, rounds: int, patch_id: int):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=distance,
        rounds=rounds,
        noise=gen.NoiseModel.si1000(0.002),
        style="cz",
        yokes=2,
        num_patches=2,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    layout = compile_layout(dem, mode="fullhistory")
    graph = compile_matching_graph(dem, layout, require_zero_frame=False)
    schedule = compile_pinball_v2_schedule(graph)

    site_to_detector: dict[tuple[int, int, int], int] = {}
    for detector_id, role in enumerate(layout.roles):
        if not isinstance(role, (L1BodyDetector, L1TerminalDetector)):
            continue
        if role.patch_id != patch_id or role.check_basis != "X":
            continue
        x, y, *_ = layout.coordinates[detector_id]
        local_x = x - role.patch_id * layout.pitch
        row = round(y + 0.5)
        column = round(
            (distance - 3) / 2
            - (local_x - 0.5 - (row % 2)) / 2
        )
        site_to_detector[(role.time, row, column)] = detector_id
    return graph, schedule, site_to_detector


def _run_mapped_case(
    *,
    distance: int,
    reference_input: np.ndarray,
    graph,
    schedule,
    site_to_detector: dict[tuple[int, int, int], int],
    patch_id: int,
) -> None:
    reference_working = reference_input.copy()
    reference = PinballReference(distance, len(reference_working))
    corrections, reference_complex = reference.decode_batch(reference_working)

    shot = np.zeros(graph.num_detectors, dtype=np.uint8)
    columns = (distance - 1) // 2
    for (time, row, column), detector_id in site_to_detector.items():
        shot[detector_id] = reference_input[time, row * columns + column]

    result = predecode_pinball_v2(graph, schedule, shot)
    domain_result = result.domain_results[L1FullHistoryDomain(patch_id, "X")]
    mapped_tentative = np.zeros_like(reference_working)
    for (time, row, column), detector_id in site_to_detector.items():
        mapped_tentative[time, row * columns + column] = (
            result.tentative_residual_syndrome[detector_id]
        )

    np.testing.assert_array_equal(mapped_tentative, reference_working)
    assert domain_result.complex == reference_complex

    # The YSC transform reflects the upstream data columns.  Consequently the
    # maintained X observable at local_x=0 is the reference rightmost-column
    # correction parity, not the reference circuit's left-column convention.
    reflected_logical_parity = np.bitwise_xor.reduce(
        corrections[
            np.arange(distance, dtype=np.int64) * distance + distance - 1
        ]
    )
    expected_frame = np.zeros(graph.num_observables, dtype=np.uint8)
    expected_frame[2 * patch_id] = reflected_logical_parity
    np.testing.assert_array_equal(result.tentative_observable_frame, expected_frame)

    expected_targets = tuple(
        sorted(
            PinballV2PauliTarget(
                patch_id=patch_id,
                local_x=distance - 1 - data_index % distance,
                y=data_index // distance,
                pauli="Z",
            )
            for data_index in np.flatnonzero(corrections)
        )
    )
    assert domain_result.tentative_physical_correction == expected_targets
    assert domain_result.physical_correction == (
        () if reference_complex else expected_targets
    )

    ordered_sites = tuple(site_to_detector.items())
    domain_detector_ids = np.asarray(
        [detector_id for _, detector_id in ordered_sites],
        dtype=np.int64,
    )
    expected_tentative = np.asarray(
        [
            reference_working[time, row * columns + column]
            for (time, row, column), _ in ordered_sites
        ],
        dtype=np.uint8,
    )
    if reference_complex:
        np.testing.assert_array_equal(
            result.residual_syndrome[domain_detector_ids],
            shot[domain_detector_ids],
        )
        assert not np.any(result.observable_frame)
    else:
        np.testing.assert_array_equal(
            result.residual_syndrome[domain_detector_ids],
            expected_tentative,
        )
        np.testing.assert_array_equal(
            result.observable_frame,
            result.tentative_observable_frame,
        )


@pytest.mark.parametrize("patch_id", [0, 1])
def test_mapped_reference_matches_adversarial_and_random_sparse_blocks(
    patch_id,
) -> None:
    distance = 5
    rounds = 2
    width = (distance + 1) * ((distance - 1) // 2)
    graph, schedule, site_to_detector = _mapped_problem(
        distance=distance,
        rounds=rounds,
        patch_id=patch_id,
    )

    adversarial = np.zeros((rounds + 1, width), dtype=np.uint8)
    adversarial.reshape(-1)[[2, 8, 10, 27]] = 1
    cases = [adversarial]
    rng = np.random.default_rng(0x8F16F24)
    cases.extend(
        (rng.random((rounds + 1, width)) < 0.12).astype(np.uint8)
        for _ in range(128)
    )

    for reference_input in cases:
        _run_mapped_case(
            distance=distance,
            reference_input=reference_input,
            graph=graph,
            schedule=schedule,
            site_to_detector=site_to_detector,
            patch_id=patch_id,
        )
