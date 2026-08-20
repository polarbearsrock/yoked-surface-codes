"""Shared multiprocessing-queue helpers for the gap collection tests.

These were previously duplicated with subtly divergent variants in
``_collection_manager_test.py`` and ``_collection_worker_test.py``. The
reconciled versions keep the strictest semantics of each: the drain helper
reports the unexpected extra item on failure, and the put helper accepts
both ``Queue`` and ``SimpleQueue``.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, List, Union


def assert_drain_queue(q: multiprocessing.Queue, expected_contents: List[Any]) -> None:
    for v in expected_contents:
        assert q.get(timeout=0.1) == v
    if not q.empty():
        raise AssertionError(f"queue had another item: {q.get()=}")


def put_wait_not_empty(
    q: Union[multiprocessing.Queue, multiprocessing.SimpleQueue], item: Any
) -> None:
    q.put(item)
    while q.empty():
        time.sleep(0.0001)
