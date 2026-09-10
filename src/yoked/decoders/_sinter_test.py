import sys

import numpy as np
import pytest
import stim

from yoked.decoders import (
    CorrelatedUnionFindDecoder, DecodingGraph, FusionBlossomUnionFindDecoder, InvalidSyndromeError,
    SinterCorrelatedUnionFindDecoder, SinterUnionFindDecoder, UnionFindDecoder,
)


@pytest.mark.parametrize('adapter_type,decoder_type', [
    (SinterUnionFindDecoder, UnionFindDecoder),
    (SinterCorrelatedUnionFindDecoder, CorrelatedUnionFindDecoder),
])
def test_packed_batches_padding_and_direct_agreement(adapter_type, decoder_type):
    dem = stim.DetectorErrorModel('\n'.join(
        [f'error(0.1) D{k} L{k}' for k in range(9)] + ['logical_observable L9']
    ))
    adapter = adapter_type()
    compiled = adapter.compile_decoder_for_dem(dem=dem)
    assert type(compiled.decoder) is decoder_type
    assert not vars(adapter)
    syndromes = np.random.default_rng(42).integers(0, 2, size=(7, 9), dtype=np.uint8)
    syndromes[0] = 0
    syndromes[1] = 1
    packed = np.packbits(syndromes, axis=1, bitorder='little')
    packed[:, 1] |= 0xfe  # Detector padding is ignored, not interpreted as defects.
    original = packed.copy()
    packed.flags.writeable = False
    result = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    direct = UnionFindDecoder(DecodingGraph.from_dem(dem)).decode_batch(syndromes)
    np.testing.assert_array_equal(result, np.packbits(direct, axis=1, bitorder='little'))
    np.testing.assert_array_equal(direct[:, :9], syndromes)
    assert not direct[:, 9].any()
    assert result.dtype == np.uint8 and result.shape == (7, 2)
    assert not (result[:, 1] & 0xfe).any()
    np.testing.assert_array_equal(packed, original)
    empty = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=np.empty((0, 2), dtype=np.uint8))
    assert empty.shape == (0, 2)


def test_default_uf_works_without_optional_fusion_blossom(monkeypatch):
    monkeypatch.setitem(sys.modules, 'fusion_blossom', None)
    dem = stim.DetectorErrorModel('error(0.1) D0 L0')
    graph = DecodingGraph.from_dem(dem)
    np.testing.assert_array_equal(UnionFindDecoder(graph).decode([1]), [True])
    compiled = SinterUnionFindDecoder().compile_decoder_for_dem(dem=dem)
    assert type(compiled.decoder) is UnionFindDecoder
    np.testing.assert_array_equal(
        compiled.decode_shots_bit_packed(bit_packed_detection_event_data=np.array([[1]], dtype=np.uint8)),
        [[1]],
    )
    with pytest.raises(ImportError, match='Fusion Blossom UF is optional'):
        FusionBlossomUnionFindDecoder(graph)


@pytest.mark.parametrize('data', [
    np.zeros((1, 1), dtype=np.int64),
    np.zeros(1, dtype=np.uint8),
    np.zeros((1, 2), dtype=np.uint8),
])
@pytest.mark.parametrize('adapter_type', [SinterUnionFindDecoder, SinterCorrelatedUnionFindDecoder])
def test_packed_input_validation(data, adapter_type):
    compiled = adapter_type().compile_decoder_for_dem(dem=stim.DetectorErrorModel('error(0.1) D0'))
    with pytest.raises(ValueError, match='uint8 array of shape'):
        compiled.decode_shots_bit_packed(bit_packed_detection_event_data=data)


@pytest.mark.parametrize('adapter_type', [SinterUnionFindDecoder, SinterCorrelatedUnionFindDecoder])
def test_empty_model_and_independent_compilations(adapter_type):
    adapter = adapter_type()
    empty = adapter.compile_decoder_for_dem(dem=stim.DetectorErrorModel())
    isolated = adapter.compile_decoder_for_dem(dem=stim.DetectorErrorModel('detector D0\nlogical_observable L2'))
    assert empty.decode_shots_bit_packed(bit_packed_detection_event_data=np.empty((3, 0), dtype=np.uint8)).shape == (3, 0)
    with pytest.raises(InvalidSyndromeError):
        isolated.decode_shots_bit_packed(bit_packed_detection_event_data=np.array([[1]], dtype=np.uint8))
    result = isolated.decode_shots_bit_packed(bit_packed_detection_event_data=np.zeros((2, 1), dtype=np.uint8))
    np.testing.assert_array_equal(result, np.zeros((2, 1), dtype=np.uint8))
    assert not vars(adapter)
