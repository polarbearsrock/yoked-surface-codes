"""Tools for defining surface code patches by their boundary outlines.

Coordinate convention: positions are complex numbers whose real part is the
x coordinate and whose imaginary part is the y coordinate. Data qubits sit
at integer coordinates (both parts integers) while measurement ancillae sit
at half-integer coordinates (both parts offset by 0.5, at plaquette
centers); `checkerboard_basis` classifies each plaquette center as X or Z
type. The line segments making up boundary curves must be horizontal,
vertical, or 45-degree diagonal (`int_points_on_line` rejects anything
else).
"""

from gen._surf._closed_curve import (
    ClosedCurve,
)
from gen._surf._css_observable_boundary_pair import (
    CssObservableBoundaryPair,
)
from gen._surf._geo import (
    int_points_on_line,
    int_points_inside_polygon,
)
from gen._surf._order import (
    checkerboard_basis,
    Order_Z,
    Order_ᴎ,
    Order_N,
    Order_S,
)
from gen._surf._patch_outline import (
    PatchOutline,
)
from gen._surf._step_sequence_outline import (
    StepSequenceOutline,
    StepOutline,
)
from gen._surf._patch_transition_outline import (
    PatchTransitionOutline,
)
from gen._surf._path_outline import (
    PathOutline,
)
from gen._surf._surface_code import (
    layer_begin,
    layer_loop,
    layer_transition,
    layer_end,
    layer_single_shot,
    surface_code_patch,
)
