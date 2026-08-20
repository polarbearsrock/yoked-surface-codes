"""Sinter-style shot collection that also records each shot's complementary gap.

The complementary gap of a shot is the difference between the weight of the
minimum weight matching of the syndrome as observed and the weight of the
matching obtained after flipping the shot's *comparison detector* (the
detector comparing the yoke check's value at the two time boundaries),
converted into decibels of likelihood ratio. Its magnitude measures how
confident the decoder is about the comparison detector's value; its sign
encodes which decode won: a negative gap means the decoder prefers the decode
in which the comparison detector is flipped, i.e. it predicts the yoke check
failed, so the sign doubles as the decoder's predicted success/failure of the
comparison. By convention the comparison detector is the LAST detector in the
circuit - `GapWorkHandler` flips the highest-index detector bit - so circuits
collected here must append their comparison detector last. That convention is
load-bearing: flipping any other detector measures a different quantity.

Shots are tallied into sinter `custom_counts` under keys like 'C<gap>'
(decoded correctly) and 'E<gap>' (logical error), with the gap rounded to an
integer number of decibels. See `yoked.gap._gap_worker_handler.GapWorkHandler`
for the implementation and `yoked._histogram_conversion.with_unsigned_gap`
for how the sign conventions are re-encoded during analysis.

`collect_gap_stats` is the collection entry point and `collect_circuit_paths`
wraps it for a list of circuit files (used by `tools/collect_gap`). The rest
of the surface is the collection machinery it is built from - a fork of
sinter's collection loop (`CollectionManager`, `CollectionWorkerState`,
`collection_worker_loop`, `ThrottledProgressPrinter`) driving the abstract
`CollectionWorkHandler`, of which `GapWorkHandler` is the gap-recording
implementation.
"""

from yoked.gap._collection_manager import CollectionManager
from yoked.gap._collection_work_handler import CollectionWorkHandler
from yoked.gap._collection_worker_loop import collection_worker_loop
from yoked.gap._collection_worker_state import CollectionWorkerState
from yoked.gap._gap_collect import collect_gap_stats
from yoked.gap._gap_collect_paths import collect_circuit_paths
from yoked.gap._gap_worker_handler import GapWorkHandler
from yoked.gap._progress_printer import ThrottledProgressPrinter

__all__ = [
    "CollectionManager",
    "CollectionWorkHandler",
    "CollectionWorkerState",
    "GapWorkHandler",
    "ThrottledProgressPrinter",
    "collect_circuit_paths",
    "collect_gap_stats",
    "collection_worker_loop",
]
