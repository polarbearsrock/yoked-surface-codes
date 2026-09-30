"""Correlated MWPM + complementary gap -> plain MWPM, following paper section 4.

This configuration uses direct gap weights, without an isotonic calibration or
a UF/plain-MWPM reference. It covers the existing 1D, two-ideal-yoke patch split,
not the paper's multi-round phenomenological outer-code simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import sinter
import stim

from yoked.hierarchical._arrays import readonly_array
from yoked.hierarchical._correlated_matching_gap import CorrelatedMatchingGapDecoder
from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
from yoked.hierarchical._patch_graphs import PatchGraphs


PAPER_CONFIGURATION = 'correlated_mwpm_complementary_gap__plain_mwpm'


@dataclass(frozen=True)
class PaperDecodeResult:
    """Batch results in patch-major columns 2*i+s; class weights are (shots,P,2,2)."""

    reference: np.ndarray
    complementary_gap: np.ndarray
    outer_weights: np.ndarray
    forced_weights: np.ndarray
    residual: np.ndarray
    prediction: np.ndarray
    sigma: np.ndarray
    outer_tied: np.ndarray

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference)
        if reference.ndim != 2 or reference.shape[1] < 2 or reference.shape[1] % 2:
            raise ValueError('Expected reference shape (shots, 2 * patches)')
        shots, columns = reference.shape
        special_shapes = {
            'forced_weights': (shots, columns // 2, 2, 2),
            'sigma': (shots, 2),
            'outer_tied': (shots, 2),
        }
        for field in fields(self):
            value = np.asarray(getattr(self, field.name))
            shape = special_shapes.get(field.name, reference.shape)
            if value.shape != shape:
                raise ValueError(f'{field.name} must have shape {shape}')
            floating = field.name in ('complementary_gap', 'outer_weights', 'forced_weights')
            if floating:
                if not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f'{field.name} must be finite and nonnegative')
            elif not np.isin(value, (0, 1)).all():
                raise ValueError(f'{field.name} must be binary')
            object.__setattr__(self, field.name, readonly_array(value, dtype=float if floating else bool))
        if not np.array_equal(self.prediction, self.reference ^ self.residual):
            raise ValueError('Final prediction must equal reference XOR residual')
        if not np.array_equal(self.residual.reshape(shots, columns // 2, 2).sum(axis=1) % 2, self.sigma):
            raise ValueError('Residual parity must equal the frame-adjusted yoke syndrome')

    def arrays(self) -> dict[str, np.ndarray]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class PaperHierarchicalDecoder:
    """Native correlated L1; frozen complementary gaps; uncorrelated MWPM L2.

    ``gap_scale=1`` uses raw gaps as outer weights. ``gap_scale=0.9`` applies
    the paper's empirical confidence rescaling. A uniform positive scaling does
    not change a unique optimum in this single-round outer parity model.
    """

    def __init__(self, dem: stim.DetectorErrorModel, *, num_patches: int | None = None,
                 gap_scale: float = 1.0):
        if isinstance(gap_scale, (bool, np.bool_)) or not math.isfinite(gap_scale) or gap_scale <= 0:
            raise ValueError('gap_scale must be finite and positive')
        self.gap_scale = float(gap_scale)
        self.patches = PatchGraphs.from_yoked_dem(
            dem, num_patches=dem.num_observables // 2 if num_patches is None else num_patches)
        self.decoders = tuple(CorrelatedMatchingGapDecoder(patch) for patch in self.patches)

    def decode_with_gaps_batch(self, detectors) -> PaperDecodeResult:
        """Decode a batch, retaining the L1 reference, gap, and L2 decision for inspection."""
        detectors = np.asarray(detectors)
        if (detectors.ndim != 2 or detectors.shape[1] != self.patches.num_detectors
                or not np.isin(detectors, (0, 1)).all()):
            raise ValueError(f'Expected binary detectors of shape (shots, {self.patches.num_detectors})')
        shots, patches = len(detectors), len(self.patches)
        reference = np.zeros((shots, 2 * patches), dtype=bool)
        gaps = np.zeros(reference.shape)
        forced = np.zeros((shots, patches, 2, 2))
        # L1 sees only local physical detectors. The yoke bits enter below at L2.
        for patch_index, decoder in enumerate(self.decoders):
            local = self.patches[patch_index].local_syndromes(detectors)
            columns = slice(2 * patch_index, 2 * patch_index + 2)
            for shot, syndrome in enumerate(local):
                result = decoder.decode(syndrome)
                reference[shot, columns] = result.reference
                gaps[shot, columns] = result.complementary_gap
                forced[shot, patch_index] = result.forced_weights

        # L2 asks which local logical predictions to reverse. Direct log-odds
        # input preserves the complete gap without fitting or clipping a probability.
        weights = self.gap_scale * gaps
        yoke = detectors[:, self.patches.yoke_detector_ids]
        sigma = frame_adjusted_syndrome(yoke, reference)
        residual = np.zeros_like(reference)
        tied = np.zeros((shots, 2), dtype=bool)
        for sector in range(2):
            result = mwpm_outer_log_odds_batch(weights[:, sector::2], sigma[:, sector])
            residual[:, sector::2] = result.patterns
            tied[:, sector] = result.tied
        prediction = reference ^ residual
        if not np.array_equal(prediction.reshape(shots, patches, 2).sum(axis=1) % 2, yoke):
            raise ValueError('Final predictions violate the measured yoke checks')
        return PaperDecodeResult(
            reference=reference, complementary_gap=gaps, outer_weights=weights,
            forced_weights=forced, residual=residual, prediction=prediction, sigma=sigma, outer_tied=tied)

    def decode_batch(self, detectors) -> np.ndarray:
        return self.decode_with_gaps_batch(detectors).prediction

    def decode(self, detectors) -> np.ndarray:
        return self.decode_batch(np.asarray(detectors)[None])[0]


class SinterPaperHierarchicalDecoder(sinter.Decoder):
    """Sinter adapter for the same L1/gap/L2 configuration, compiled per worker."""

    def __init__(self, *, gap_scale: float = 1.0):
        self.gap_scale = gap_scale

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledPaperDecoder(PaperHierarchicalDecoder(dem, gap_scale=self.gap_scale))


class _CompiledPaperDecoder(sinter.CompiledDecoder):
    """Translate Sinter's packed detector/observable layout at the API boundary."""
    def __init__(self, decoder: PaperHierarchicalDecoder):
        self.decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        data = np.asarray(bit_packed_detection_event_data)
        nd = self.decoder.patches.num_detectors
        if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] != (nd + 7) // 8:
            raise ValueError(f'Expected a uint8 array of shape (shots, {(nd + 7) // 8})')
        detectors = np.unpackbits(data, axis=1, count=nd, bitorder='little')
        return np.packbits(self.decoder.decode_batch(detectors), axis=1, bitorder='little')
