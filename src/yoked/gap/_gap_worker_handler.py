import collections
import math
import time
from typing import Optional

import numpy as np
import pymatching
import sinter
import stim

from yoked.gap._collection_work_handler import CollectionWorkHandler


class GapWorkHandler(CollectionWorkHandler):
    """Samples shots and tallies them by logical error and complementary gap.

    Each shot is decoded twice: once on the syndrome as sampled, and once
    with the comparison detector (by convention the last detector in the
    circuit) flipped. The difference in matching weight, converted to
    decibels, is the shot's complementary gap (see `yoked.gap`). Shots are
    counted under custom_counts keys 'C<gap>' or 'E<gap>' depending on
    whether the unflipped decode predicted the actual observables.
    """

    def __init__(self):
        self.loaded_key = None
        self.matcher: Optional[pymatching.Matching] = None
        self.sampler: Optional[stim.CompiledDetectorSampler] = None
        self.decibels_per_w: float = 1
        self.check_mask_for_last_byte: int = 0
        self.num_safe_work = 1

    def do_some_work(self, task: sinter.Task, max_shots: int) -> sinter.AnonTaskStats:
        self._load_task(task)
        t0 = time.monotonic()
        result = self._sample_loaded_task(num_shots=min(max_shots, self.num_safe_work))
        dt = time.monotonic() - t0
        if self.num_safe_work > 1:
            if dt > 10:
                self.num_safe_work //= 2
        if self.num_safe_work < 1024:
            if dt < 1:
                self.num_safe_work *= 2
            if dt < 0.5:
                self.num_safe_work *= 2
            if dt < 0.25:
                self.num_safe_work *= 2
            if dt < 0.125:
                self.num_safe_work *= 2
        return result

    def _load_task(self, task: sinter.Task) -> None:
        key = task.strong_id()
        if self.loaded_key == key:
            return
        self.loaded_key = key
        self.matcher = pymatching.Matching.from_detector_error_model(task.detector_error_model)
        self.sampler = task.circuit.compile_detector_sampler()
        self.num_safe_work = 1

        # Matching weights are proportional to log likelihood ratios, so any
        # single edge's (weight, probability) pair calibrates the conversion
        # from pymatching weight units into decibels.
        edge = next(iter(self.matcher.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        self.decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w

        # The comparison detector is by convention the LAST detector in the
        # circuit. In bit-packed detection event data it lands in the final
        # byte of each shot's row, at this bit position.
        self.check_mask_for_last_byte = 1 << ((task.circuit.num_detectors - 1) % 8)

    def _sample_loaded_task(self, *, num_shots: int) -> sinter.AnonTaskStats:
        assert self.loaded_key is not None

        t0 = time.monotonic()
        dets, actual_obs = self.sampler.sample(
            shots=num_shots,
            bit_packed=True,
            separate_observables=True,
        )

        predicted_obs, weights = self.matcher.decode_batch(
            dets,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )

        # Flip the comparison (last) detector in every shot and decode again;
        # the weight difference between the two decodes is the shot's
        # complementary gap.
        dets[:, -1] ^= self.check_mask_for_last_byte
        predicted_obs_with_inverted_check, weights_with_inverted_check = self.matcher.decode_batch(
            dets,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )

        errors = np.any(predicted_obs != actual_obs, axis=1)
        num_errors = np.count_nonzero(errors)

        # Classify all shots by their error + gap.
        custom_counts = collections.Counter()
        gaps = (weights_with_inverted_check - weights) * self.decibels_per_w
        gaps = np.round(gaps).astype(dtype=np.int64)
        for k in range(len(gaps)):
            g = gaps[k]
            key = f'E{g}' if errors[k] else f'C{g}'
            custom_counts[key] += 1
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=num_shots,
            errors=num_errors,
            seconds=t1 - t0,
            custom_counts=custom_counts,
        )
