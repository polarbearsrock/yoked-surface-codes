import collections
from typing import Tuple, List, FrozenSet, Set, Counter, Optional

import stim

from gen._core._builder import Builder
from gen._flows._flow import PauliString
from gen._core._noise import NoiseModel, NoiseRule, OP_TYPES, CLIFFORD_2Q, ANNOTATION
from gen._core._patch import Patch
from gen._core._util import sorted_complex


def _commune_with_the_observables(
    *,
    patch: Patch,
    obs_x: Optional[List[PauliString]],
    obs_z: Optional[List[PauliString]],
    suggested_ancilla_qubits: Optional[List[complex]],
) -> Tuple[List[PauliString], FrozenSet[complex]]:
    """Validates the given observables and decides which ones the circuit will track.

    Validation checks that paired X/Z observables anticommute, and that all
    other pairs of observables commute.

    If only one of obs_x/obs_z is given, that list is returned unchanged with
    no extra qubits. If both are given, the observables can't all be tracked
    directly (paired ones anticommute), so this method silently switches to an
    EPR-style construction: each pair is combined with an out-of-patch ancilla
    qubit (one X-paired and one Z-paired combined observable per pair), and
    the ancillas are returned so callers can keep them noise-free.

    Returns:
        A (observables, immune_qubits) tuple, where observables are the
        observables the circuit should track and immune_qubits are the EPR
        ancilla qubits that must be excluded from noise.
    """
    if obs_z is None and obs_x is None:
        raise ValueError('No observables specified')
    if obs_x is not None and obs_z is not None:
        assert len(obs_x) == len(obs_z), 'obs_x and obs_z must pair up one-to-one'
    n = len(obs_x) if obs_x is not None else len(obs_z)
    for i in range(n):
        if obs_x is not None:
            assert isinstance(obs_x[i], PauliString)
        if obs_z is not None:
            assert isinstance(obs_z[i], PauliString)
        if obs_x is not None and obs_z is not None:
            assert not obs_x[i].commutes(obs_z[i]), f'paired observables {i} must anticommute'
        for j in range(i + 1, n):
            if obs_x is not None:
                assert obs_x[i].commutes(obs_x[j]), f'X observables {i} and {j} must commute'
            if obs_z is not None:
                assert obs_z[i].commutes(obs_z[j]), f'Z observables {i} and {j} must commute'
            if obs_x is not None and obs_z is not None:
                assert obs_x[i].commutes(obs_z[j]), f'X observable {i} must commute with Z observable {j}'
                assert obs_z[i].commutes(obs_x[j]), f'Z observable {i} must commute with X observable {j}'

    if obs_z is None:
        return obs_x, frozenset()
    if obs_x is None:
        return obs_z, frozenset()

    ancilla_qubits = []
    epr_observables = []
    a = min(q.real for q in patch.data_set) + min(q.imag for q in patch.data_set)*1j - 1j
    for k in range(n):
        if suggested_ancilla_qubits is not None:
            assert len(suggested_ancilla_qubits) == n
            a = suggested_ancilla_qubits[k]
        ancilla_qubits.append(a)
        epr_observables.append(obs_x[k] * PauliString({a: 'X'}))
        epr_observables.append(obs_z[k] * PauliString({a: 'Z'}))
        a += 1
    return epr_observables, frozenset(ancilla_qubits)


def make_phenomenological_circuit_for_stabilizer_code(
        *,
        patch: Patch,
        noise: NoiseRule,
        observables_x: Optional[List[PauliString]] = None,
        observables_z: Optional[List[PauliString]] = None,
        suggested_ancilla_qubits: Optional[List[complex]] = None,
        rounds: int,
) -> stim.Circuit:
    """Builds a phenomenological-noise memory circuit for a stabilizer patch.

    Each round measures the patch stabilizers and applies the supplied noise
    rule between rounds. Logical observables are measured at both temporal
    boundaries and linked with ``OBSERVABLE_INCLUDE`` instructions.
    """
    observables, immune = _commune_with_the_observables(
        patch=patch,
        obs_x=observables_x,
        obs_z=observables_z,
        suggested_ancilla_qubits=suggested_ancilla_qubits,
    )
    builder = Builder.for_qubits(patch.data_set | immune)

    for k, obs in enumerate(observables):
        builder.measure_pauli_product(q2b=obs.qubits, key=f'OBS_START{k}')
        builder.obs_include([f'OBS_START{k}'], obs_index=k)
    builder.measure_patch(patch, save_layer='init')
    builder.tick()

    loop = builder.fork()
    loop.measure_patch(patch, save_layer='loop', cmp_layer='init')
    loop.shift_coords(dt=1)
    loop.tick()
    noise_model = NoiseModel(
        tick_noise=NoiseRule(after=noise.after),
        any_measurement_rule=NoiseRule(flip_result=noise.flip_result, after={}),
        any_clifford_1q_rule=NoiseRule(after={}),
        any_clifford_2q_rule=NoiseRule(after={}),
        allow_multiple_uses_of_a_qubit_in_one_tick=True,
    )
    noisy_loop = noise_model.noisy_circuit(
        loop.circuit,
        immune_qubits={builder.q2i[q] for q in immune},
    )
    if len(noisy_loop) == 0 or (isinstance(noisy_loop[-1], stim.CircuitInstruction) and noisy_loop[-1].name != 'TICK'):
        noisy_loop.append('TICK')
    builder.circuit += noisy_loop * rounds

    builder.measure_patch(patch, save_layer='end', cmp_layer='loop')
    for k, obs in enumerate(observables):
        builder.measure_pauli_product(q2b=obs.qubits, key=f'OBS_END{k}')
        builder.obs_include([f'OBS_END{k}'], obs_index=k)

    return builder.circuit


def make_code_capacity_circuit_for_stabilizer_code(
        *,
        patch: Patch,
        noise: NoiseRule,
        observables_x: Optional[List[PauliString]] = None,
        observables_z: Optional[List[PauliString]] = None,
        suggested_ancilla_qubits: Optional[List[complex]] = None,
) -> stim.Circuit:
    """Builds a code-capacity circuit with one noisy data-qubit interval.

    The patch stabilizers and requested logical observables are measured
    before and after the supplied Pauli noise channels. Measurement-result
    flips are not supported by this circuit model.
    """
    assert noise.flip_result == 0
    observables, immune = _commune_with_the_observables(
        patch=patch,
        obs_x=observables_x,
        obs_z=observables_z,
        suggested_ancilla_qubits=suggested_ancilla_qubits,
    )
    builder = Builder.for_qubits(patch.data_set | immune)

    for k, obs in enumerate(observables):
        builder.measure_pauli_product(q2b=obs.qubits, key=f'OBS_START{k}')
        builder.obs_include([f'OBS_START{k}'], obs_index=k)
    builder.measure_patch(patch, save_layer='init')
    builder.tick()

    for k, p in noise.after.items():
        builder.circuit.append(k, [builder.q2i[q] for q in sorted_complex(patch.data_set - immune)], p)
    builder.tick()

    builder.measure_patch(patch, save_layer='end', cmp_layer='init')
    for k, obs in enumerate(observables):
        builder.measure_pauli_product(q2b=obs.qubits, key=f'OBS_END{k}')
        builder.obs_include([f'OBS_END{k}'], obs_index=k)

    return builder.circuit


def _disambiguated_gate_names(instruction: stim.CircuitInstruction) -> List[str]:
    """Expands an MPP or controlled-Pauli instruction into disambiguated gate names.

    CX/CY/CZ/XCZ/YCZ target pairs classify as 'feedback' (measurement record
    control), 'sweep' (sweep bit control), or the plain gate name; one entry is
    returned per pair. MPP instructions expand into one entry per measured
    product, named by the measured bases (e.g. "MXX" for X1*X2), with all-Z
    products named "M".
    """
    out: List[str] = []
    if instruction.name in ['CX', 'CY', 'CZ', 'XCZ', 'YCZ']:
        targets = instruction.targets_copy()
        for k in range(0, len(targets), 2):
            if targets[k].is_measurement_record_target or targets[k + 1].is_measurement_record_target:
                out.append('feedback')
            elif targets[k].is_sweep_bit_target or targets[k + 1].is_sweep_bit_target:
                out.append('sweep')
            else:
                out.append(instruction.name)
    elif instruction.name == 'MPP':
        op = 'M'
        targets = instruction.targets_copy()
        is_continuing = True
        for t in targets:
            if t.is_combiner:
                is_continuing = True
                continue
            p = 'X' if t.is_x_target else 'Y' if t.is_y_target else 'Z' if t.is_z_target else '?'
            if is_continuing:
                op += p
                is_continuing = False
            else:
                if op == 'MZ':
                    op = 'M'
                out.append(op)
                op = 'M' + p
        if op:
            if op == 'MZ':
                op = 'M'
            out.append(op)
    else:
        raise NotImplementedError(f'{instruction.name=}')
    return out


def gate_counts_for_circuit(circuit: stim.Circuit) -> Counter[str]:
    """Determines gates used by a circuit, disambiguating MPP/feedback cases.

    MPP instructions are expanded into what they actually measure, such as
    "MXX" for MPP X1*X2 and "MXYZ" for MPP X4*Y5*Z7.

    Feedback instructions like `CX rec[-1] 0` become the gate "feedback".

    Sweep instructions like `CX sweep[2] 0` become the gate "sweep".
    """
    out = collections.Counter()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            for k, v in gate_counts_for_circuit(instruction.body_copy()).items():
                out[k] += v * instruction.repeat_count

        elif instruction.name in ['CX', 'CY', 'CZ', 'XCZ', 'YCZ', 'MPP']:
            for name in _disambiguated_gate_names(instruction):
                out[name] += 1

        elif OP_TYPES[instruction.name] == CLIFFORD_2Q or instruction.name in ['PAULI_CHANNEL_2', 'DEPOLARIZE2']:
            out[instruction.name] += len(instruction.targets_copy()) // 2
        elif OP_TYPES[instruction.name] == ANNOTATION or instruction.name in ['E', 'ELSE_CORRELATED_ERROR']:
            out[instruction.name] += 1
        else:
            out[instruction.name] += len(instruction.targets_copy())

    return out


def gates_used_by_circuit(circuit: stim.Circuit) -> Set[str]:
    """Determines gates used by a circuit, disambiguating MPP/feedback cases.

    MPP instructions are expanded into what they actually measure, such as
    "MXX" for MPP X1*X2 and "MXYZ" for MPP X4*Y5*Z7.

    Feedback instructions like `CX rec[-1] 0` become the gate "feedback".

    Sweep instructions like `CX sweep[2] 0` become the gate "sweep".
    """
    out = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            out |= gates_used_by_circuit(instruction.body_copy())

        elif instruction.name in ['CX', 'CY', 'CZ', 'XCZ', 'YCZ', 'MPP']:
            out.update(_disambiguated_gate_names(instruction))

        else:
            out.add(instruction.name)

    return out
