"""Circuit-generation primitives used by the yoked surface-code experiments.

The package gathers the commonly used geometry, flow, noise, transpilation,
and visualization helpers into one convenience import surface. The names in
``__all__`` are the intentionally re-exported API; implementation modules
remain private.
"""

from gen._circuit_util import (
    make_phenomenological_circuit_for_stabilizer_code,
    make_code_capacity_circuit_for_stabilizer_code,
    gates_used_by_circuit,
    gate_counts_for_circuit,
)
from gen._core import (
    AtLayer,
    Builder,
    complex_key,
    MeasurementTracker,
    min_max_complex,
    NoiseModel,
    NoiseRule,
    occurs_in_classical_control_system,
    Patch,
    sorted_complex,
    Tile,
)
from gen._flows import (
    Chunk,
    ChunkLoop,
    Flow,
    PauliString,
    compile_chunks_into_circuit,
    magic_measure_for_flows,
    FlowStabilizerVerifier,
)
from gen._layers import (
    transpile_to_z_basis_interaction_circuit,
)
from gen._main_util import (
    main_generate_circuits,
    generate_noisy_circuit_from_chunks,
    CircuitBuildParams,
)
from gen._plaq_problem import (
    PlaqProblem,
)
from gen._stabilizer_code import (
    StabilizerCode,
)
from gen._surf import (
    ClosedCurve,
    CssObservableBoundaryPair,
    StepSequenceOutline,
    int_points_on_line,
    int_points_inside_polygon,
    checkerboard_basis,
    Order_Z,
    Order_ᴎ,
    Order_N,
    Order_S,
    PatchOutline,
    layer_begin,
    layer_loop,
    layer_transition,
    layer_end,
    layer_single_shot,
    surface_code_patch,
    PathOutline,
    PatchTransitionOutline,
    StepOutline,
)
from gen._util import (
    stim_circuit_with_transformed_coords,
    estimate_qubit_count_during_postselection,
    write_file,
)
from gen._viz_circuit_html import (
    stim_circuit_html_viewer,
)
from gen._viz_patch_svg import (
    patch_svg_viewer,
)

__all__ = [
    'AtLayer',
    'Builder',
    'Chunk',
    'ChunkLoop',
    'CircuitBuildParams',
    'ClosedCurve',
    'CssObservableBoundaryPair',
    'Flow',
    'FlowStabilizerVerifier',
    'MeasurementTracker',
    'NoiseModel',
    'NoiseRule',
    'Order_N',
    'Order_S',
    'Order_Z',
    'Order_ᴎ',
    'Patch',
    'PatchOutline',
    'PatchTransitionOutline',
    'PathOutline',
    'PauliString',
    'PlaqProblem',
    'StabilizerCode',
    'StepOutline',
    'StepSequenceOutline',
    'Tile',
    'checkerboard_basis',
    'compile_chunks_into_circuit',
    'complex_key',
    'estimate_qubit_count_during_postselection',
    'gate_counts_for_circuit',
    'gates_used_by_circuit',
    'generate_noisy_circuit_from_chunks',
    'int_points_inside_polygon',
    'int_points_on_line',
    'layer_begin',
    'layer_end',
    'layer_loop',
    'layer_single_shot',
    'layer_transition',
    'magic_measure_for_flows',
    'main_generate_circuits',
    'make_code_capacity_circuit_for_stabilizer_code',
    'make_phenomenological_circuit_for_stabilizer_code',
    'min_max_complex',
    'occurs_in_classical_control_system',
    'patch_svg_viewer',
    'sorted_complex',
    'stim_circuit_html_viewer',
    'stim_circuit_with_transformed_coords',
    'surface_code_patch',
    'transpile_to_z_basis_interaction_circuit',
    'write_file',
]
