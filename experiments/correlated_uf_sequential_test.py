"""Check distance barriers, bounded concurrency, and failure handling."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import threading
import time

import pytest

spec = spec_from_file_location('sequential', Path(__file__).with_name('correlated_uf_sequential.py'))
runner = module_from_spec(spec)
spec.loader.exec_module(runner)


def test_parallel_shards_finish_and_aggregate_before_next_distance():
    lock = threading.Lock()
    running, maximum = 0, 0
    aggregated, decoded = [], []

    def decode(distance, start, stop):
        nonlocal running, maximum
        with lock:
            if distance == 9:
                assert aggregated == [7]
            running += 1
            maximum = max(maximum, running)
        time.sleep(0.02)
        with lock:
            decoded.append((distance, start, stop))
            running -= 1

    tasks = {7: [(0, 2), (2, 4), (4, 6)], 9: [(0, 2), (2, 4)]}
    for distance in runner.run_in_order(tasks, 2, decode, lambda *_: None, lambda *_: None):
        assert running == 0
        assert sum(d == distance for d, _, _ in decoded) == len(tasks[distance])
        aggregated.append(distance)
    assert maximum == 2
    assert aggregated == [7, 9]


def test_completed_distance_needs_no_workers_and_failure_blocks_next_distance():
    decoded = []

    def fail(distance, start, stop):
        decoded.append(distance)
        raise RuntimeError('worker failed')

    iterator = runner.run_in_order({7: [], 9: [(0, 2)], 11: [(0, 2)]}, 32,
                                   fail, lambda *_: None, lambda *_: None)
    assert next(iterator) == 7
    with pytest.raises(RuntimeError, match='worker failed'):
        next(iterator)
    assert decoded == [9]
