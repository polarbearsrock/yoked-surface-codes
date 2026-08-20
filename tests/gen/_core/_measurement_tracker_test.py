import pytest

from gen._core._measurement_tracker import AtLayer, MeasurementTracker


def test_record_obstacle_then_copy_preserves_obstacle():
    tracker = MeasurementTracker()
    tracker.record_measurement(AtLayer('a', 0))
    tracker.record_obstacle(AtLayer('b', 0))

    copied = tracker.copy()
    assert copied.recorded == {
        AtLayer('a', 0): [0],
        AtLayer('b', 0): None,
    }
    assert copied.next_measurement_index == 1

    # The copy must be independent of the original.
    copied.recorded[AtLayer('a', 0)].append(5)
    assert tracker.recorded[AtLayer('a', 0)] == [0]


def test_measurement_indices_raises_on_obstacle():
    tracker = MeasurementTracker()
    tracker.record_measurement(AtLayer('a', 0))
    tracker.record_obstacle(AtLayer('b', 0))

    assert tracker.measurement_indices([AtLayer('a', 0)]) == [0]
    with pytest.raises(ValueError, match='Obstacle'):
        tracker.measurement_indices([AtLayer('b', 0)])
    with pytest.raises(ValueError, match='No such measurement'):
        tracker.measurement_indices([AtLayer('c', 0)])
