"""Yoked-surface-code constructions, gap collection, and decoder experiments.

This package bundles the yoked-memory circuit constructions from the "Yoked
Surface Codes" paper (multi-patch memories under 1D/2D yokes, squareberg
grids, and patch rotation), the complementary-gap collection machinery in
:mod:`yoked.gap`, the histogram/curve helpers the gap-plotting tools use, and
the ProMatch-L1 predecoder experiment in :mod:`yoked.decoding` (whose sinter
decoders register via the ``yoked.decoding:custom_decoders`` entry point).
See ``docs/CODEBASE_GUIDE.md`` for a tour of the repository layout and the
experiment workflows.
"""

from yoked._histogram_conversion import (
    curve_rescaled_to_target_area,
    histogram_cosine_convolve,
    histogram_cumulative_meet_in_the_middle,
    with_unsigned_gap,
)
from yoked._patch_rotation import (
    MagicableCircuit,
    patch_rotation_circuit,
)
from yoked._squareberg_circuits import (
    squareberg_magic_memory_circuit,
    squareberg_phenomenological_circuit,
)
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.gap import collect_gap_stats

__all__ = [
    "MagicableCircuit",
    "collect_gap_stats",
    "curve_rescaled_to_target_area",
    "histogram_cosine_convolve",
    "histogram_cumulative_meet_in_the_middle",
    "patch_rotation_circuit",
    "squareberg_magic_memory_circuit",
    "squareberg_phenomenological_circuit",
    "with_unsigned_gap",
    "yoked_magic_memory_circuit",
]
