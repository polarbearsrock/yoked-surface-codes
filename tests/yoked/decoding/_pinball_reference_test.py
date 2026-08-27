import numpy as np
import pytest

from yoked.decoding._pinball_reference import (
    PINBALL_REFERENCE_COMMIT,
    PINBALL_REFERENCE_STAGE_ORDER,
    PinballReference,
)


def _syndrome_index(decoder: PinballReference, row: int, col: int) -> int:
    return row * decoder.num_syndrome_cols + col


def _active_pair(
    decoder: PinballReference,
    *positions: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(decoder.num_syndromes, dtype=np.uint8)
    for row, col in positions:
        result[_syndrome_index(decoder, row, col)] = 1
    return result


def test_reference_identity_and_constructor_contract() -> None:
    decoder = PinballReference(distance=5, batch_size=6)
    assert PINBALL_REFERENCE_COMMIT == (
        "8f16f24b621aacfaa4f456a2aeec8df088faf3a7"
    )
    assert decoder.num_syndrome_rows == 6
    assert decoder.num_syndrome_cols == 2
    assert decoder.num_syndromes == 12
    assert decoder.num_data_qubits == 25

    with pytest.raises(ValueError, match="odd distance"):
        PinballReference(distance=4, batch_size=5)
    with pytest.raises(ValueError, match="positive"):
        PinballReference(distance=3, batch_size=0)


def test_measurement_stage_clears_equal_consecutive_sites_without_correction() -> None:
    decoder = PinballReference(distance=3, batch_size=2)
    batch = np.zeros((2, decoder.num_syndromes), dtype=np.uint8)
    batch[:, _syndrome_index(decoder, 1, 0)] = 1

    corrections, complex_batch, trace = decoder.decode_batch_traced(batch)

    assert not complex_batch
    assert not np.any(batch)
    assert not np.any(corrections)
    second_measurement = trace[len(PINBALL_REFERENCE_STAGE_ORDER)]
    assert second_measurement.stage == "M"
    assert not np.any(second_measurement.previous_syndrome)
    assert not np.any(second_measurement.current_syndrome)


@pytest.mark.parametrize(
    "stage,positions,data_index",
    [
        ("B1", ((3, 0), (2, 0)), 2 * 5 + 1),
        ("B2", ((3, 0), (4, 0)), 3 * 5 + 1),
        ("B3", ((3, 1), (4, 0)), 3 * 5 + 2),
        ("B4", ((3, 1), (2, 0)), 2 * 5 + 2),
    ],
)
def test_each_bulk_stage_matches_upstream_direction_and_data_qubit(
    stage: str,
    positions: tuple[tuple[int, int], tuple[int, int]],
    data_index: int,
) -> None:
    decoder = PinballReference(distance=5, batch_size=1)
    batch = _active_pair(decoder, *positions).reshape(1, -1)

    corrections, complex_batch, trace = decoder.decode_batch_traced(batch)

    assert not complex_batch
    assert not np.any(batch)
    assert np.flatnonzero(corrections).tolist() == [data_index]
    selected = next(item for item in trace if item.stage == stage)
    assert np.flatnonzero(selected.correction_delta).tolist() == [data_index]


@pytest.mark.parametrize(
    "stage,current_position,previous_position,data_index",
    [
        ("ST1", (3, 0), (2, 0), 2 * 5 + 1),
        ("ST2", (3, 1), (2, 0), 2 * 5 + 2),
    ],
)
def test_each_spacetime_stage_uses_the_preceding_top_neighbor(
    stage: str,
    current_position: tuple[int, int],
    previous_position: tuple[int, int],
    data_index: int,
) -> None:
    decoder = PinballReference(distance=5, batch_size=2)
    batch = np.zeros((2, decoder.num_syndromes), dtype=np.uint8)
    batch[0] = _active_pair(decoder, previous_position)
    batch[1] = _active_pair(decoder, current_position)

    corrections, complex_batch, trace = decoder.decode_batch_traced(batch)

    assert not complex_batch
    assert not np.any(batch)
    assert np.flatnonzero(corrections).tolist() == [data_index]
    selected = next(
        item for item in trace if item.sweep_index == 1 and item.stage == stage
    )
    assert np.flatnonzero(selected.correction_delta).tolist() == [data_index]


def test_hook_stage_emits_the_upstream_two_data_qubit_chain() -> None:
    decoder = PinballReference(distance=5, batch_size=2)
    batch = np.zeros((2, decoder.num_syndromes), dtype=np.uint8)
    batch[0] = _active_pair(decoder, (1, 0))
    batch[1] = _active_pair(decoder, (3, 0))

    corrections, complex_batch, trace = decoder.decode_batch_traced(batch)

    assert not complex_batch
    assert not np.any(batch)
    assert np.flatnonzero(corrections).tolist() == [5, 10]
    hook = next(
        item for item in trace if item.sweep_index == 1 and item.stage == "H"
    )
    assert np.flatnonzero(hook.correction_delta).tolist() == [5, 10]


def test_final_edge_flush_clears_both_upstream_boundaries() -> None:
    decoder = PinballReference(distance=5, batch_size=1)
    batch = _active_pair(decoder, (3, 0), (2, 1)).reshape(1, -1)

    corrections, complex_batch, trace = decoder.decode_batch_traced(batch)

    assert not complex_batch
    assert not np.any(batch)
    assert np.flatnonzero(corrections).tolist() == [10, 14]
    final = trace[-1]
    assert final.stage == "E"
    assert final.sweep_index == 1
    assert final.previous_layer == 0
    assert final.current_layer is None
    assert np.flatnonzero(final.correction_delta).tolist() == [10, 14]


def test_trace_has_nine_stages_per_sweep_plus_final_e_and_owned_snapshots() -> None:
    decoder = PinballReference(distance=3, batch_size=3)
    batch = np.zeros((3, decoder.num_syndromes), dtype=np.uint8)

    corrections, complex_batch, trace = decoder.decode_batch_traced(batch)

    assert not complex_batch
    assert not np.any(corrections)
    assert len(trace) == 3 * len(PINBALL_REFERENCE_STAGE_ORDER) + 1
    for sweep in range(3):
        stages = tuple(
            item.stage for item in trace if item.sweep_index == sweep
        )
        assert stages == PINBALL_REFERENCE_STAGE_ORDER
    assert trace[-1].stage == "E"
    assert trace[-1].sweep_index == 3
    for item in trace:
        assert not item.previous_syndrome.flags.writeable
        assert not item.correction_delta.flags.writeable
        assert not item.accumulated_corrections.flags.writeable
        if item.current_syndrome is not None:
            assert not item.current_syndrome.flags.writeable


def test_complex_batch_retains_upstream_tentative_mutation_and_correction() -> None:
    decoder = PinballReference(distance=5, batch_size=1)
    # Interior singleton remains complex.  The boundary singleton is still
    # consumed during the final E pass; upstream does not roll it back.
    batch = _active_pair(decoder, (3, 1), (3, 0)).reshape(1, -1)

    corrections, complex_batch = decoder.decode_batch(batch)

    assert complex_batch
    assert batch[0, _syndrome_index(decoder, 3, 1)] == 1
    assert batch[0, _syndrome_index(decoder, 3, 0)] == 0
    assert np.flatnonzero(corrections).tolist() == [10]


def test_nonadjacent_boundary_corrections_cancel_across_batch() -> None:
    decoder = PinballReference(distance=5, batch_size=3)
    batch = np.zeros((3, decoder.num_syndromes), dtype=np.uint8)
    boundary = _syndrome_index(decoder, 3, 0)
    batch[0, boundary] = 1
    batch[2, boundary] = 1

    corrections, complex_batch = decoder.decode_batch(batch)

    assert not complex_batch
    assert not np.any(batch)
    assert not np.any(corrections)


def test_left_column_parity_matches_upstream_logical_prediction() -> None:
    decoder = PinballReference(distance=5, batch_size=1)
    corrections = np.zeros(decoder.num_data_qubits, dtype=np.uint8)
    corrections[0] = 1
    assert not decoder.is_logical_error(None, corrections, True)
    assert decoder.is_logical_error(None, corrections, False)
    corrections[5] = 1
    assert not decoder.is_logical_error(None, corrections, False)


def test_input_validation_rejects_nonbinary_or_read_only_batches() -> None:
    decoder = PinballReference(distance=3, batch_size=1)
    bad = np.zeros((1, decoder.num_syndromes), dtype=np.uint8)
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        decoder.decode_batch(bad)

    read_only = np.zeros((1, decoder.num_syndromes), dtype=np.uint8)
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        decoder.decode_batch(read_only)
