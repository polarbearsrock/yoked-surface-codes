"""Data qubit interaction orders for surface code plaquette measurements.

Each order lists the four half-step offsets from a plaquette's central
measurement ancilla to its data qubits (UL/UR/DL/DR corners), in the order
the ancilla interacts with them. The names describe the letter traced out by
visiting the corners in order: Z, ᴎ (mirrored N), N, and S. The choice of
order per plaquette basis controls the orientation of hook errors.
"""

UL, UR, DL, DR = [e * 0.5 for e in [-1 - 1j, +1 - 1j, -1 + 1j, +1 + 1j]]
Order_Z = [UL, UR, DL, DR]
Order_ᴎ = [UL, DL, UR, DR]
Order_N = [DL, UL, DR, UR]
Order_S = [DL, DR, UL, UR]


def checkerboard_basis(q: complex) -> str:
    """Classifies a coordinate as X type or Z type according to a checkerboard.
    """
    is_x = int(q.real + q.imag) & 1 == 0
    return 'X' if is_x else 'Z'
