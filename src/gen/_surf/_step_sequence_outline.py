import pathlib
from typing import Union, Callable, Iterable

import gen
from gen._surf._order import Order_Z
from gen._surf._step_outline import StepOutline
from gen._util import write_file


class StepSequenceOutline:
    """An ordered surface-code deformation sequence with export helpers."""

    def __init__(self, steps: Iterable[StepOutline]):
        self.steps = list(steps)

    def write_outlines_svg(self, path: Union[str, pathlib.Path]) -> None:
        from gen._surf._viz_patch_outline_svg import patch_outline_svg_viewer
        viewer = patch_outline_svg_viewer([
            piece
            for segment in self.steps
            for piece in [segment.start, segment.body, segment.end]
            if any(piece.iter_control_points())
        ])
        write_file(path, viewer)

    def write_patches_svg(self, path: Union[str, pathlib.Path], *, rel_order_func: Callable[[complex], Iterable[complex]] = None) -> None:
        from gen._viz_patch_svg import patch_svg_viewer
        show_order = True
        if rel_order_func is None:
            rel_order_func = lambda _: Order_Z
            show_order = False
        patches = []
        for k in range(len(self.steps)):
            segment = self.steps[k]
            patch = segment.body.to_patch(rel_order_func=rel_order_func)
            if segment.start.data_set:
                b2 = segment.start.data_basis_map
                tiles = list(segment.start.patch.tiles)
                for t in patch.tiles:
                    if all(b2.get(q, b) == b for q, b in t.to_data_pauli_string().qubits.items()):
                        tiles.append(t)
                patches.append(gen.Patch(tiles))
            patches.append(patch)
            if segment.end.data_set:
                b2 = segment.end.data_basis_map
                tiles = list(segment.end.patch.tiles)
                for t in patch.tiles:
                    if all(b2.get(q, b) == b for q, b in t.to_data_pauli_string().qubits.items()):
                        tiles.append(t)
                patches.append(gen.Patch(tiles))
        viewer = patch_svg_viewer(patches, show_order=show_order, show_measure_qubits=False)
        write_file(path, viewer)

    def write_gltf(self, path: Union[str, pathlib.Path]) -> None:
        from gen._surf._viz_sequence_3d import patch_sequence_to_model
        patch_sequence_to_model(self).save_json(str(path))
        print(f'wrote file://{pathlib.Path(path).absolute()}')

    def write_3d_viewer_html(self, path: Union[str, pathlib.Path]) -> None:
        from gen._surf._viz_gltf_3d import viz_3d_gltf_model_html
        from gen._surf._viz_sequence_3d import patch_sequence_to_model
        write_file(path, viz_3d_gltf_model_html(patch_sequence_to_model(self)))
