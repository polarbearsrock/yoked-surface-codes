import re

from gen._surf._closed_curve import ClosedCurve
from gen._surf._patch_outline import PatchOutline
from gen._surf._patch_transition_outline import PatchTransitionOutline
from gen._surf._viz_patch_outline_svg import patch_outline_svg_viewer


def _path_ds(svg: str):
    return re.findall(r'<path d="([^"]*)"', svg)


def test_patch_outline_paths_are_single_subpaths():
    c1 = ClosedCurve.from_cycle([0, 'X', 2, 'Z', 2 + 2j, 'X', 2j, 'Z', 0])
    c2 = c1.offset_by(4)
    svg = patch_outline_svg_viewer([PatchOutline([c1, c2])])

    ds = _path_ds(svg)
    assert len(ds) == 2
    for d in ds:
        assert d.count('M') == 1
        assert d.count('Z') == 1
    # The second curve's path must not re-embed the first curve's subpath.
    first_start = ds[0].split()[1]
    assert first_start not in ds[1]


def test_patch_transition_outline_paths_are_single_subpaths():
    c1 = ClosedCurve(points=[0, 2, 2 + 2j, 2j], bases=['X'] * 4)
    c2 = c1.offset_by(4)
    outline = PatchTransitionOutline(
        observable_deltas={},
        data_boundary_planes=[c1, c2],
    )
    svg = patch_outline_svg_viewer([outline])

    ds = _path_ds(svg)
    assert len(ds) == 2
    for d in ds:
        assert d.count('M') == 1
        assert d.count('Z') == 1
    first_start = ds[0].split()[1]
    assert first_start not in ds[1]
