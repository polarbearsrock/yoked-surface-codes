import numpy as np
import pytest
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoders import (
    DecodingGraph, FusionBlossomUnionFindDecoder, InvalidSyndromeError,
    SinterFusionBlossomUnionFindDecoder, SinterUnionFindDecoder, UnionFindDecoder,
)

pytest.importorskip('fusion_blossom')


def test_fusion_uf_mode_on_graph_with_suboptimal_correction():
    # Distinct, exactly representable weights. Full MWPM instead selects
    # edges 4, 7, 9 with weight 176; UF selects a valid correction of weight 190.
    weighted_edges = [
        (0, 1, 194), (0, 2, 178), (0, 3, 196), (0, 4, 176), (0, 5, 24),
        (1, 2, 138), (1, 3, 80), (1, 4, 112), (1, 5, 110), (2, 3, 40),
        (2, 4, 164), (2, 5, 30), (3, 4, 46), (3, 5, 130), (4, 5, 156),
    ]
    graph = DecodingGraph(6, 15, [(u, v, w, 1 << e) for e, (u, v, w) in enumerate(weighted_edges)])
    decoder = FusionBlossomUnionFindDecoder(graph)
    syndrome = np.ones(6, dtype=np.bool_)
    selected = np.flatnonzero(decoder.decode(syndrome)).tolist()
    assert selected == [4, 6, 9, 12]
    assert sum(graph.edges[e][2] for e in selected) == 190
    incidence = np.bincount(np.array(graph.endpoints)[selected].ravel(), minlength=6) % 2
    np.testing.assert_array_equal(incidence, syndrome)


def test_shapes_immutable_inputs_and_invalid_syndrome_recovery():
    graph = DecodingGraph(3, 2, [(0, 1, 1, 2)])
    decoder = FusionBlossomUnionFindDecoder(graph)
    batch = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=np.uint8)
    original = batch.copy()
    batch.flags.writeable = False
    expected = [[False, True], [False, False], [False, True]]
    np.testing.assert_array_equal(decoder.decode_batch(batch), expected)
    np.testing.assert_array_equal(batch, original)
    # Even total parity is insufficient when disconnected components are odd.
    for invalid in [[1, 0, 1], [0, 0, 1], [1, 0, 0]]:
        with pytest.raises(InvalidSyndromeError):
            decoder.decode(invalid)
    np.testing.assert_array_equal(decoder.decode_batch(batch), expected)
    empty = decoder.decode_batch(np.empty((0, 3), dtype=np.bool_))
    assert empty.dtype == np.bool_ and empty.shape == (0, 2)


@pytest.mark.parametrize('syndrome', [[1], [[1, 1]], [2, 0], [-1, 0], [float('nan'), 0], ['1', '0']])
def test_invalid_input(syndrome):
    decoder = FusionBlossomUnionFindDecoder(DecodingGraph(2, 0, [(0, 1, 1, 0)]))
    with pytest.raises(ValueError):
        decoder.decode(syndrome)


def test_empty_graph_zero_weights_and_boundary_edges():
    empty = FusionBlossomUnionFindDecoder(DecodingGraph(0, 2, []))
    np.testing.assert_array_equal(empty.decode_batch(np.empty((3, 0))), np.zeros((3, 2)))
    zero = FusionBlossomUnionFindDecoder(DecodingGraph(2, 1, [(0, 1, 0, 1)]))
    np.testing.assert_array_equal(zero.decode([1, 1]), [True])
    boundary = FusionBlossomUnionFindDecoder(DecodingGraph(1, 2, [(0, None, 1, 1), (0, None, 4, 2)]))
    np.testing.assert_array_equal(boundary.decode([1]), [True, False])


@pytest.mark.parametrize('scale', [0, -1, True, 0.5, float('inf')])
def test_invalid_weight_scale(scale):
    with pytest.raises(ValueError, match='positive integer'):
        FusionBlossomUnionFindDecoder(DecodingGraph(0, 0, []), weight_scale=scale)


def test_unsupported_native_graph_inputs():
    with pytest.raises(ValueError, match='parallel detector edges'):
        FusionBlossomUnionFindDecoder(DecodingGraph(2, 1, [(0, 1, 1, 0), (1, 0, 2, 1)]))
    for weight in [2**30, 1e308]:
        with pytest.raises(ValueError, match='integer range'):
            FusionBlossomUnionFindDecoder(DecodingGraph(1, 0, [(0, None, weight, 0)]))


def test_explicit_sinter_variant_and_default_with_native_library_installed():
    dem = stim.DetectorErrorModel('\n'.join(f'error(0.1) D{k} L{k}' for k in range(9)))
    adapter = SinterFusionBlossomUnionFindDecoder()
    compiled = adapter.compile_decoder_for_dem(dem=dem)
    assert not vars(adapter)
    assert type(compiled.decoder) is FusionBlossomUnionFindDecoder
    assert type(SinterUnionFindDecoder().compile_decoder_for_dem(dem=dem).decoder) is UnionFindDecoder
    syndromes = np.random.default_rng(42).integers(0, 2, size=(7, 9), dtype=np.uint8)
    packed = np.packbits(syndromes, axis=1, bitorder='little')
    expected = packed.copy()
    packed[:, -1] |= 0xfe
    result = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('distance', [3, 7])
def test_yoked_graph_correction_and_observable_reconstruction(distance):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=distance, rounds=4 * distance,
        noise=gen.NoiseModel.si1000(0.003), style='cz', yokes=2, num_patches=6,
    )
    graph = DecodingGraph.from_dem(circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True,
    ))
    decoder = FusionBlossomUnionFindDecoder(graph)
    endpoints = np.array(graph.endpoints)
    syndromes = circuit.compile_detector_sampler(seed=42).sample(shots=16)
    assert syndromes[:, -2:].any(axis=0).all()
    for syndrome in syndromes:
        prediction = decoder.decode(syndrome)
        selected = decoder._solver.subgraph()
        rebuilt = np.bincount(endpoints[selected].ravel(), minlength=len(graph.adjacency))[:graph.num_detectors] % 2
        np.testing.assert_array_equal(rebuilt, syndrome)
        mask = 0
        for edge in selected:
            mask ^= graph.edges[edge][3]
        np.testing.assert_array_equal(prediction, [(mask >> k) & 1 for k in range(graph.num_observables)])
