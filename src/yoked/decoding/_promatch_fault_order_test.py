"""Bounded exhaustive fault-order acceptance tests for L1 ProMatch.

This module exhausts every *order-one* mechanism in one tiny maintained DEM.
Each complete ``error(...)`` instruction is one mechanism: separator-delimited
``^`` components are XORed together instead of being split into fake faults.
No order-two-or-higher distance-preservation claim is made here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._promatch_decoder import PromatchDecoder


def _tiny_maintained_circuit() -> stim.Circuit:
    return yoked_magic_memory_circuit(
        patch_diameter=2,
        rounds=4,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=2,
        num_patches=2,
        remove_x_yoke=False,
    )


def _atomic_order_one_corpus(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    """Returns one detector/observable signature per complete DEM error."""

    detector_rows: list[np.ndarray] = []
    observable_rows: list[np.ndarray] = []
    instructions: list[str] = []
    separator_instruction_count = 0
    # flattened() expands repeat blocks and resolves detector shifts while
    # preserving each complete error instruction and its separators.
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
        separator_instruction_count += int(saw_separator)
        detector_rows.append(detectors)
        observable_rows.append(observables)
        instructions.append(str(instruction))
    if not detector_rows:
        raise AssertionError("tiny maintained DEM unexpectedly has no errors")
    return (
        np.stack(detector_rows),
        np.stack(observable_rows),
        tuple(instructions),
        separator_instruction_count,
    )


def _packed(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits, axis=1, bitorder="little")


def _shot_failed(predictions: np.ndarray, observables: np.ndarray) -> np.ndarray:
    return np.any(np.bitwise_xor(predictions, observables), axis=1)


def _prematch_boundary(
    *,
    paths: Sequence[object],
    edge_by_id: dict[int, object],
    num_detectors: int,
) -> np.ndarray:
    boundary = np.zeros(num_detectors, dtype=np.uint8)
    for path in paths:
        for edge_id in path.edge_ids:  # type: ignore[attr-defined]
            edge = edge_by_id[edge_id]
            boundary[edge.source] ^= 1  # type: ignore[attr-defined]
            target = edge.target  # type: ignore[attr-defined]
            if target is not None:
                boundary[target] ^= 1
    return boundary


def _matching_boundary(
    matcher: pymatching.Matching,
    syndrome: np.ndarray,
) -> np.ndarray:
    boundary = np.zeros(syndrome.shape, dtype=np.uint8)
    for source, target in matcher.decode_to_edges_array(syndrome):
        if source >= 0:
            boundary[int(source)] ^= 1
        if target >= 0:
            boundary[int(target)] ^= 1
    return boundary


def test_exhaustive_atomic_dem_order_one_faults_and_complete_boundary() -> None:
    """Exhausts order one; HW=10/4/2 are vacuous and HW=0 is active."""

    circuit = _tiny_maintained_circuit()
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    detectors, observable_bits, instructions, separator_count = (
        _atomic_order_one_corpus(dem)
    )

    # These frozen counts prove that complete mechanisms were enumerated. In
    # particular, 280 correlated mechanisms contain one or more ``^``
    # separators and are still represented by exactly one corpus row each.
    assert len(instructions) == 372
    assert separator_count == 280

    packed_detectors = _packed(detectors)
    packed_observables = _packed(observable_bits)
    u0_matcher = pymatching.Matching.from_detector_error_model(dem)
    u0_matcher.ensure_num_fault_ids(dem.num_observables)
    u0_predictions = np.asarray(
        u0_matcher.decode_batch(
            packed_detectors,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        ),
        dtype=np.uint8,
    )
    u0_failed = _shot_failed(u0_predictions, packed_observables)
    assert np.count_nonzero(u0_failed) == 0

    # Counts are (activated shots, shots with a successful domain, shots with a
    # rolled-back domain, committed paths). An order-one SI1000 mechanism has
    # at most two events in any L1 domain, so limits 10, 4, and 2 are explicitly
    # recorded as vacuous. Limit 0 is the non-vacuous forced-threshold check.
    expected_counts = {
        10: (0, 0, 0, 0),
        4: (0, 0, 0, 0),
        2: (0, 0, 0, 0),
        0: (350, 96, 310, 104),
    }

    for hw_limit, expected in expected_counts.items():
        compiled = PromatchDecoder(
            residual_hw_limit=hw_limit,
        ).compile_decoder_for_dem(dem=dem)
        residual, frames, results = compiled.predecode_shots(detectors)
        activated = sum(
            any(stat.status != "below-limit" for stat in result.domain_stats.values())
            for result in results
        )
        successful = sum(
            any(stat.status == "success" for stat in result.domain_stats.values())
            for result in results
        )
        rolled_back = sum(
            any(stat.status == "rollback" for stat in result.domain_stats.values())
            for result in results
        )
        committed_paths = sum(len(result.paths) for result in results)
        assert (activated, successful, rolled_back, committed_paths) == expected

        residual_predictions = np.asarray(
            compiled.graph.matcher.decode_batch(
                _packed(residual),
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
        pu_predictions = np.bitwise_xor(residual_predictions, _packed(frames))
        pu_failed = _shot_failed(pu_predictions, packed_observables)
        pu_only_failures = np.flatnonzero(~u0_failed & pu_failed)
        assert not pu_only_failures.size, (
            f"HW={hw_limit} has PU-only order-one failures at mechanisms "
            f"{pu_only_failures[:8].tolist()}: "
            f"{[instructions[k] for k in pu_only_failures[:8]]}"
        )

        if hw_limit == 0:
            edge_by_id = {edge.edge_id: edge for edge in compiled.graph.edges}
            for original, residual_shot, result in zip(detectors, residual, results):
                prematch = _prematch_boundary(
                    paths=result.paths,
                    edge_by_id=edge_by_id,
                    num_detectors=dem.num_detectors,
                )
                residual_match = _matching_boundary(
                    compiled.graph.matcher,
                    residual_shot,
                )
                np.testing.assert_array_equal(residual_match, residual_shot)
                # Edge incidence is composed over GF(2), including overlaps.
                np.testing.assert_array_equal(
                    np.bitwise_xor(prematch, residual_match),
                    original,
                )
