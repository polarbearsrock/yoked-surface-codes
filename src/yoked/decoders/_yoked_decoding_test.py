import numpy as np
import pytest
import scipy.sparse

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoders import (
    CorrelatedUnionFindDecoder, DecodingGraph, SinterCorrelatedUnionFindDecoder,
    SinterUnionFindDecoder, UnionFindDecoder,
)


def _check_parallel_labels(dem):
    labels = {}
    for instruction in dem.flattened():
        if instruction.type != 'error' or instruction.args_copy()[0] == 0:
            continue
        components = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                components.append([])
            else:
                components[-1].append(target)
        for component in components:
            key = frozenset(t.val for t in component if t.is_relative_detector_id())
            mask = 0
            for target in component:
                if target.is_logical_observable_id():
                    mask ^= 1 << target.val
            if key:
                assert labels.setdefault(key, mask) == mask


@pytest.mark.parametrize('distance', [3, 7])
@pytest.mark.parametrize('correlated', [False, True])
def test_yoked_circuit_corrections_and_public_interfaces(distance, correlated):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=distance,
        rounds=4 * distance,
        noise=gen.NoiseModel.si1000(0.003),
        style='cz',
        yokes=2,
        num_patches=6,
    )
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    _check_parallel_labels(dem)
    decoder = (CorrelatedUnionFindDecoder.from_dem(dem) if correlated else
               UnionFindDecoder(DecodingGraph.from_dem(dem)))
    graph = decoder.graph
    assert (graph.num_detectors, graph.num_observables) == (circuit.num_detectors, circuit.num_observables)
    syndromes = circuit.compile_detector_sampler(seed=42).sample(shots=16)
    assert syndromes[:, -2:].any(axis=0).all()  # Both yokes are exercised.

    # Independently construct H and L from the public graph, including every
    # real detector row. The final two rows are the yoke constraints.
    rows, columns = [], []
    for e, (u, v, _, _) in enumerate(graph.edges):
        rows.append(u)
        columns.append(e)
        if v is not None:
            rows.append(v)
            columns.append(e)
    incidence = scipy.sparse.csc_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, columns)),
        shape=(graph.num_detectors, len(graph.edges)),
    )
    logical = np.array([
        [(mask >> k) & 1 for _, _, _, mask in graph.edges]
        for k in range(graph.num_observables)
    ], dtype=np.uint8)
    predictions = []
    for syndrome in syndromes:
        result = decoder._decode(syndrome)
        assert len(result.selected_edges) == len(set(result.selected_edges))
        correction = np.zeros(len(graph.edges), dtype=np.uint8)
        correction[list(result.selected_edges)] = 1
        np.testing.assert_array_equal((incidence @ correction) & 1, syndrome)
        prediction = np.array([(result.observable_mask >> k) & 1 for k in range(graph.num_observables)], dtype=np.bool_)
        np.testing.assert_array_equal((logical @ correction) & 1, prediction)
        predictions.append(prediction)

    # Check the public batch and Sinter paths on the same first two shots.
    np.testing.assert_array_equal(decoder.decode_batch(syndromes[:2]), predictions[:2])
    adapter = SinterCorrelatedUnionFindDecoder() if correlated else SinterUnionFindDecoder()
    compiled = adapter.compile_decoder_for_dem(dem=dem)
    packed = np.packbits(syndromes[:2], axis=1, bitorder='little')
    actual = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    expected = np.packbits(np.array(predictions[:2]), axis=1, bitorder='little')
    np.testing.assert_array_equal(actual, expected)
