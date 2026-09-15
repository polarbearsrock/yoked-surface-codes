"""Owned array storage for immutable mathematical results.

Callers validate shapes and values before converting to the documented dtype.
A frozen dataclass alone does not prevent changes through a NumPy alias.
"""
import numpy as np


def readonly_array(value, *, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order='C', copy=True)
    result.setflags(write=False)
    return result
