import io

import pytest
import sinter
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.gap import _gap_collect
from yoked.gap._collection_manager import CollectionManager
from yoked.gap._gap_collect import collect_gap_stats


def test_collect():
    task = sinter.Task(
        circuit=yoked_magic_memory_circuit(
            patch_diameter=5,
            rounds=25,
            noise=gen.NoiseModel.uniform_depolarizing(1e-2),
            yokes=True,
            style='cz',
            num_patches=1,
        ),
        decoder='pymatching',
        json_metadata={'d': 5, 'r': 25, 'p': 1e-3},
    )
    out = io.StringIO()
    collect_gap_stats(
        num_workers=2,
        tasks=[task],
        num_shots=1000,
        out=out,
        print_progress=False,
        print_header=True,
        existing_data={},
        worker_flush_period=5,
    )

    out.seek(0)
    data = out.read()
    assert data.startswith(sinter.CSV_HEADER)
    assert ''',""C''' in data and ''',""E''' in data


def test_collect_stops_workers_when_distributing_work_fails(monkeypatch):
    started_processes = []

    class RecordingManager(CollectionManager):
        def start_workers(self, **kwargs):
            super().start_workers(**kwargs)
            started_processes.extend(ws.process for ws in self.worker_states)

    monkeypatch.setattr(_gap_collect, 'CollectionManager', RecordingManager)

    task = sinter.Task(
        circuit=stim.Circuit.generated(
            'repetition_code:memory',
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        ),
        decoder='pymatching',
        json_metadata={'d': 3},
    )

    try:
        # Giving the same task twice makes distributing work fail after the
        # worker processes have already been started.
        with pytest.raises(ValueError, match='Same task given twice'):
            collect_gap_stats(
                num_workers=2,
                tasks=[task, task],
                num_shots=1000,
                out=io.StringIO(),
                print_progress=False,
                print_header=False,
                existing_data={},
                worker_flush_period=5,
            )
        assert len(started_processes) == 2
        assert not any(p.is_alive() for p in started_processes)
    finally:
        # Don't leak workers (and hang pytest at exit) if the assert fails.
        for p in started_processes:
            if p.is_alive():
                p.kill()
                p.join()
