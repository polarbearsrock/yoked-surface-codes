import pytest

import gen
from gen._circuit_util import make_code_capacity_circuit_for_stabilizer_code, \
    make_phenomenological_circuit_for_stabilizer_code
from gen._stabilizer_code import StabilizerCode


def _make_code() -> StabilizerCode:
    patch = gen.Patch([
        gen.Tile(
            bases='Z',
            ordered_data_qubits=[0, 1, 1j, 1 + 1j],
            measurement_qubit=0.5 + 0.5j,
        ),
        gen.Tile(
            bases='X',
            ordered_data_qubits=[0, 1],
            measurement_qubit=0.5,
        ),
        gen.Tile(
            bases='X',
            ordered_data_qubits=[0 + 1j, 1 + 1j],
            measurement_qubit=0.5 + 1j,
        ),
    ])
    return StabilizerCode(
        patch=patch,
        obs_x=gen.PauliString({0: 'X', 1j: 'X'}),
        obs_z=gen.PauliString({0: 'Z', 1: 'Z'}),
    )


def test_make_code_capacity_circuit():
    code = _make_code()
    noise = gen.NoiseRule(after={'DEPOLARIZE1': 0.125})

    assert code.make_code_capacity_circuit(
        noise=noise,
        basis='X',
    ) == make_code_capacity_circuit_for_stabilizer_code(
        patch=code.patch,
        noise=noise,
        observables_x=[code.obs_x],
    )

    assert code.make_code_capacity_circuit(
        noise=noise,
        basis='Z',
    ) == make_code_capacity_circuit_for_stabilizer_code(
        patch=code.patch,
        noise=noise,
        observables_z=[code.obs_z],
    )

    with pytest.raises(ValueError):
        code.make_code_capacity_circuit(noise=noise, basis='M')


def test_make_phenomenological_circuit():
    code = _make_code()
    noise = gen.NoiseRule(flip_result=0.125, after={'DEPOLARIZE1': 0.25})

    assert code.make_phenomenological_circuit(
        noise=noise,
        rounds=5,
        basis='X',
    ) == make_phenomenological_circuit_for_stabilizer_code(
        patch=code.patch,
        noise=noise,
        rounds=5,
        observables_x=[code.obs_x],
    )

    assert code.make_phenomenological_circuit(
        noise=noise,
        rounds=5,
        basis='Z',
    ) == make_phenomenological_circuit_for_stabilizer_code(
        patch=code.patch,
        noise=noise,
        rounds=5,
        observables_z=[code.obs_z],
    )

    with pytest.raises(ValueError):
        code.make_phenomenological_circuit(noise=noise, rounds=5, basis='M')
