"""Resolve two saved CZ faults into propagation and context interventions.

All 16 Pauli products are inserted at the same gate location, replacing the
one original event and preserving every other physical event. Independently
simulate each inserted product in isolation. Also propagate ancilla X/Y/Z
from each of that ancilla's four CZ slots to the next measurement.

These are selected illustrative cases, not a frequency estimate. Propagated
Pauli support is a circuit-frame representative, not a minimum weight modulo
stabilizers or a diagnosis of circuit distance.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import stim

from analyze import HERE, PREVIOUS, base, helper
from yoked.decoders import CorrelatedUnionFindDecoder


OLD = HERE.parent / 'correlated_uf_mwpm_failure_analysis_d7_d9'
previous = helper('cz_previous_decoder_analysis', OLD / 'analyze.py')
PRODUCTS = tuple(a + b for a, b in itertools.product('IXYZ', repeat=2))


def isolated_response(ops, location, data, ancilla, product):
    circuit = stim.Circuit()
    for index, op in enumerate(ops):
        if op.name not in base.CHANNELS:
            if op.name == 'M':
                circuit.append('M', op.targets_copy())
            else:
                circuit.append(op)
        if index == location:
            for qubit, pauli in zip((data, ancilla), product, strict=True):
                if pauli != 'I':
                    circuit.append(pauli + '_ERROR', [qubit], 1)
    first = None
    for seed in (1, 123456):
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(2, separate_observables=True)
        np.testing.assert_array_equal(dets[0], dets[1])
        np.testing.assert_array_equal(obs[0], obs[1])
        if first is None:
            first = dets[0], obs[0]
        else:
            np.testing.assert_array_equal(first[0], dets[0])
            np.testing.assert_array_equal(first[1], obs[0])
    return first


def propagate_to_measurement(ops, location, data, ancilla, product, circuit, roles):
    frame = stim.PauliString(circuit.num_qubits)
    frame[data], frame[ancilla] = product
    for index in range(location + 1, len(ops)):
        op = ops[index]
        qubits = [t.value for t in op.targets_copy() if t.is_qubit_target]
        if op.name == 'M' and ancilla in qubits:
            break
        if stim.gate_data(op.name).is_unitary:
            frame = frame.after(op)
        elif stim.gate_data(op.name).is_reset or stim.gate_data(op.name).produces_measurements:
            assert not any(frame[q] for q in qubits), 'Unexpected nonunitary on propagated support'
    else:
        raise AssertionError('Missing next ancilla measurement')
    support = [dict(qubit=q, pauli='IXYZ'[frame[q]], role=roles[q])
               for q in range(len(frame)) if frame[q]]
    return dict(before_measurement_instruction=index, support=support,
                data_support=[entry for entry in support if entry['role'] == 'data'],
                flipped_measurements=[q for q in qubits if frame[q] in (1, 2)])


def round_slots(ops, location, ancilla, roles):
    def acts_on(op):
        return any(t.is_qubit_target and t.value == ancilla for t in op.targets_copy())
    start = next(i for i in range(location, -1, -1) if ops[i].name == 'R' and acts_on(ops[i]))
    stop = next(i for i in range(location + 1, len(ops)) if ops[i].name == 'M' and acts_on(ops[i]))
    slots = []
    for index in range(start + 1, stop):
        if ops[index].name != 'DEPOLARIZE2' or not acts_on(ops[index]):
            continue
        targets = ops[index].targets_copy()
        pair = next([targets[k].value, targets[k+1].value] for k in range(0, len(targets), 2)
                    if ancilla in (targets[k].value, targets[k+1].value))
        data = next(q for q in pair if q != ancilla)
        assert roles[data] == 'data'
        slots.append((index, data))
    assert len(slots) == 4
    return slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a = previous.Analysis(7)
    circuit = stim.Circuit.from_file(a.directory / 'circuit.stim')
    ops, roles, _, _ = base.inventory(circuit)
    uf = CorrelatedUnionFindDecoder(a.graph, correlation_rules=[
        (source, target, weight) for source, rules in enumerate(a.rules) for target, weight in rules])
    archived = json.loads((OLD / 'physical_cases.json').read_text())

    def decode(syndrome, actual):
        correction = uf._decode(syndrome)
        u = a.mask(correction.selected_edges, syndrome)
        m = a.mask(a.matrix.edge_ids_of(a.native.decode_to_edges_array(
            syndrome, enable_correlations=True)), syndrome)
        assert u == correction.observable_mask
        return dict(actual=actual, UF=u, MWPM=m,
                    UF_correct=u == actual, MWPM_correct=m == actual,
                    UF_wrong_bits=previous.wrong_bits(u, actual),
                    MWPM_wrong_bits=previous.wrong_bits(m, actual))

    cases = []
    for shot, fault_id in ((76890, 90), (44875, 247)):
        archived_case = next(row for row in archived if row['shot'] == shot)
        source = Path(archived_case['physical_source'])
        for filename, expected in archived_case['source_hashes'].items():
            assert base.sha(source / filename) == expected
        faults = json.loads((source / 'physical_faults.json').read_text())
        fault = faults[fault_id]
        location = fault['instruction_index']
        data, ancilla = (fault['q0'], fault['q1']) if roles[fault['q0']] == 'data' else (fault['q1'], fault['q0'])
        original_product = (fault['pauli0'] + fault['pauli1'] if roles[fault['q0']] == 'data'
                            else fault['pauli1'] + fault['pauli0'])
        assert ops[location].name == 'DEPOLARIZE2'
        assert [t.value for t in ops[location].targets_copy()[fault['target_index']:fault['target_index']+2]] == [fault['q0'], fault['q1']]
        with np.load(source / 'fault_responses.npz') as responses:
            original_d, original_o = responses['detectors'], responses['observables']
        syndrome = a.syndrome(shot)
        actual_bits = a.record.actual[shot]
        np.testing.assert_array_equal(np.logical_xor.reduce(original_d), syndrome)
        np.testing.assert_array_equal(np.logical_xor.reduce(original_o), actual_bits)
        background_d = syndrome ^ original_d[fault_id]
        background_o = actual_bits ^ original_o[fault_id]
        variants = []
        for product in PRODUCTS:
            dets, obs = isolated_response(ops, location, data, ancilla, product)
            if product == original_product:
                np.testing.assert_array_equal(dets, original_d[fault_id])
                np.testing.assert_array_equal(obs, original_o[fault_id])
            alone = decode(dets, previous.bitmask(obs))
            context = decode(background_d ^ dets, previous.bitmask(background_o ^ obs))
            if product == original_product:
                assert context['UF'] == previous.bitmask(a.record.baselines['joint_correlated_uf'][shot])
                assert context['MWPM'] == previous.bitmask(a.record.baselines['joint_correlated_mwpm'][shot])
            variants.append(dict(pauli=product, detectors=np.flatnonzero(dets).tolist(),
                observables=np.flatnonzero(obs).tolist(),
                unfired_in_original_shot=[int(d) for d in np.flatnonzero(dets) if not syndrome[d]],
                alone=alone, in_original_background=context,
                propagation=propagate_to_measurement(ops, location, data, ancilla, product, circuit, roles)))
        timing = []
        slots = round_slots(ops, location, ancilla, roles)
        for slot, (index, partner) in enumerate(slots, 1):
            for product in ('IX', 'IY', 'IZ'):
                dets, obs = isolated_response(ops, index, partner, ancilla, product)
                timing.append(dict(slot=slot, instruction_index=index, data_qubit=partner,
                    ancilla_qubit=ancilla, pauli=product,
                    detectors=np.flatnonzero(dets).tolist(), observables=np.flatnonzero(obs).tolist(),
                    propagation=propagate_to_measurement(ops, index, partner, ancilla, product, circuit, roles)))
        case = dict(shot=shot, fault_id=fault_id, original_fault=fault,
            original_CZ_slot=next(k for k, (index, _) in enumerate(slots, 1) if index == location),
            variants=variants, ancilla_error_timing=timing,
            source_hashes=archived_case['source_hashes'], physical_source=str(source))
        cases.append(case)
        print(json.dumps(dict(shot=shot, original_product=original_product,
            all_products_correct_alone=all(v['alone']['UF_correct'] and v['alone']['MWPM_correct'] for v in variants),
            context_UF_failing=[v['pauli'] for v in variants if not v['in_original_background']['UF_correct']],
            context_MWPM_failing=[v['pauli'] for v in variants if not v['in_original_background']['MWPM_correct']]), indent=2), flush=True)
    result = dict(distance=7, physical_error_probability=.003, cases=cases,
        single_fault_response_checks=dict(locations_times_products=56, seeds=[1, 123456], samples_per_seed=2),
        all_original_physical_responses_and_predictions_reproduced=True,
        every_decoded_correction_syndrome_checked=True,
        interpretation='Selected complete-shot contexts; replacement is one event, with updated truth. '
            'Timing probes are isolated insertions, not relocated faults in the noisy context. '
            'Pauli frame support is not minimized modulo stabilizers.',
        source_hashes={str(p): base.sha(p) for p in (Path(__file__).resolve(), HERE/'analyze.py',
            PREVIOUS/'analyze.py', OLD/'analyze.py', a.directory/'circuit.stim', a.directory/'model.dem')})
    base.write_json(args.output, result)


if __name__ == '__main__':
    main()
