import stim

from gen._viz_circuit_html import stim_circuit_html_viewer


def test_mixed_basis_mpp_html_is_deterministic():
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        MPP X0*Z1
    """)

    first = stim_circuit_html_viewer(circuit, known_error=[])
    second = stim_circuit_html_viewer(circuit, known_error=[])

    assert first == second
