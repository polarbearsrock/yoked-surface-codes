import numpy as np
import sinter
import stim

from yoked.decoders._correlated_union_find import CorrelatedUnionFindDecoder
from yoked.decoders._fusion_blossom import FusionBlossomUnionFindDecoder
from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import UnionFindDecoder


class SinterUnionFindDecoder(sinter.Decoder):
    """Default UF adapter using the repository implementation in each worker."""

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledUnionFindDecoder(UnionFindDecoder(DecodingGraph.from_dem(dem)))


class SinterCorrelatedUnionFindDecoder(sinter.Decoder):
    """Optional two-pass UF; compile the graph and correlations per worker."""

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledUnionFindDecoder(CorrelatedUnionFindDecoder.from_dem(dem))


class SinterFusionBlossomUnionFindDecoder(sinter.Decoder):
    """Explicit opt-in to Fusion Blossom UF; native state is created per worker."""

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledUnionFindDecoder(FusionBlossomUnionFindDecoder(DecodingGraph.from_dem(dem)))


class _CompiledUnionFindDecoder(sinter.CompiledDecoder):
    def __init__(self, decoder: UnionFindDecoder | FusionBlossomUnionFindDecoder):
        self.decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        data = np.asarray(bit_packed_detection_event_data)
        nd = self.decoder.graph.num_detectors
        if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] != (nd + 7) // 8:
            raise ValueError(f'Expected a uint8 array of shape (shots, {(nd + 7) // 8})')
        syndromes = np.unpackbits(data, axis=1, count=nd, bitorder='little')
        predictions = self.decoder.decode_batch(syndromes)
        return np.packbits(predictions, axis=1, bitorder='little')
