from typing import Iterator, Tuple

from gen._surf._path_outline import PathOutline


class CssObservableBoundaryPair:
    """Pairs the X- and Z-boundary paths defining one CSS logical qubit."""

    def __init__(self, *, x_obs: PathOutline, z_obs: PathOutline):
        self.x_obs = x_obs
        self.z_obs = z_obs

    def __iter__(self) -> Iterator[Tuple[str, PathOutline]]:
        yield 'X', self.x_obs
        yield 'Z', self.z_obs
