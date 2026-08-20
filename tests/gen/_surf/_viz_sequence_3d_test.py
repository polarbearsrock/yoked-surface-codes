from gen._surf._closed_curve import ClosedCurve
from gen._surf._patch_outline import PatchOutline
from gen._surf._viz_sequence_3d import _patch_boundary_to_walls


def test_patch_boundary_to_walls_one_quad_per_segment():
    boundary = PatchOutline([
        ClosedCurve.from_cycle([0, 'X', 2, 'Z', 2 + 2j, 'X', 2j, 'Z', 0]),
    ])
    triangles = []
    lines = []
    _patch_boundary_to_walls(
        boundary=boundary,
        t=0,
        dt=1,
        out_triangles=triangles,
        out_lines=lines,
    )
    # One wall quad per boundary segment (materials are double sided, so no
    # duplicate back-face quad is needed).
    assert len(triangles) == 4
