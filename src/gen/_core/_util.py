from typing import Dict, List, Callable, Iterable, TypeVar, Any, Tuple, Optional

TItem = TypeVar('TItem')

# Maps a desired measurement basis to the qubit orientation that turns a CZ
# interaction into a controlled-Pauli of that basis. Lives in gen._core so
# both gen._core._patch and gen._interaction_planner can use it without an
# upward import.
DESIRED_Z_TO_ORIENTATION: Dict[str, str] = {
    'X': 'ZX',
    'Y': 'ZY',
    'Z': 'XZ',
}


def complex_key(c: complex) -> Any:
    """Returns the canonical ordering key used for complex grid coordinates."""
    return c.real != int(c.real), c.real, c.imag


def sorted_complex(
        values: Iterable[TItem],
        *,
        key: Callable[[TItem], Any] = lambda e: e) -> List[TItem]:
    """Sorts values by complex coordinates selected by ``key``."""
    return sorted(values, key=lambda e: complex_key(key(e)))


def min_max_complex(coords: Iterable[complex], *, default: Optional[complex] = None) -> Tuple[complex, complex]:
    """Computes the bounding box of a collection of complex numbers.

    Args:
        coords: The complex numbers to place a bounding box around.
        default: If the collection of complex numbers is empty, the bounding
            box will cover just this single value. If this argument isn't set
            (or is set to None), an exception is raised instead when given an
            empty collection.

    Returns:
        A pair of complex values (c_min, c_max) where c_min is the minimum
        corner of the bounding box and c_max is the maximum corner of the
        bounding box.
    """
    coords = list(coords)
    if not coords and default is not None:
        return default, default
    min_r = min([c.real for c in coords])
    min_i = min([c.imag for c in coords])
    max_r = max([c.real for c in coords])
    max_i = max([c.imag for c in coords])
    return min_r + min_i*1j, max_r + max_i*1j
