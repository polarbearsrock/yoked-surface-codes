"""Tests for the ``readonly_array`` invariants: copy-on-construction,
write-protection, and dtype/shape fidelity."""
import numpy as np
import pytest

from yoked.hierarchical._arrays import readonly_array


def test_mutating_the_input_after_construction_does_not_change_the_result():
    source = np.array([1, 2, 3], dtype=np.int64)
    result = readonly_array(source, dtype=np.int64)
    source[0] = 99
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_assignment_through_the_result_raises():
    result = readonly_array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        result[0] = 5


def test_requested_dtype_and_shape_are_preserved():
    result = readonly_array([[1, 2], [3, 4]], dtype=np.float32)
    assert result.dtype == np.float32
    assert result.shape == (2, 2)
