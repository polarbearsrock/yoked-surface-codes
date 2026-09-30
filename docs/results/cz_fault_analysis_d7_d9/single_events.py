"""Delete each actual CZ event in a preselected sample of UF-only failures.

Selection is uniform within the frozen UF-only diagnostic stratum. All events
in each selected shot are tested, avoiding selection of only promising faults.
Counts describe this conditional sample, not additive shares of the LER gap.
"""
import argparse
from collections import defaultdict
import csv
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import stim

from analyze import HERE, PREVIOUS, base, helper
from yoked.decoders import CorrelatedUnionFindDecoder, UnionFindDecoder


OLD = HERE.parent / 'correlated_uf_mwpm_failure_analysis_d7_d9'
previous = helper('cz_event_previous_analysis', OLD / 'analyze.py')
STATE = None


def decode_event(index):
    a, uf, original_d, original_o, event_d, event_o, events = STATE
    event = events[index]
    row = event['cohort_index']
    results = []
    for syndrome, actual in ((original_d[row] ^ event_d[index], original_o[row] ^ event_o[index]),
                             (event_d[index], event_o[index])):
        correction = uf._decode(syndrome)
        u = a.mask(correction.selected_edges, syndrome)
        m = a.mask(a.matrix.edge_ids_of(a.native.decode_to_edges_array(
            syndrome, enable_correlations=True)), syndrome)
        truth = previous.bitmask(actual)
        results.append((truth, u, m))
    return index, results


def run_distance(args, distance):
    source = args.source_work_dir / f'd{distance}'
    output = args.work_dir / f'd{distance}'
    output.mkdir(exist_ok=True)
    selection = json.loads((source / 'selection.json').read_text())
    assert selection == json.loads((PREVIOUS / f'd{distance}/selection.json').read_text())
    candidates = np.flatnonzero(np.asarray(selection['strata']) == 1)
    rng = np.random.default_rng(args.seed + distance)
    chosen = np.sort(rng.choice(candidates, size=args.shots, replace=False))
    shots = [selection['selected_rows'][k] for k in chosen]
    selection_path = output / 'single_event_selection.json'
    assert not selection_path.exists(), 'Use a fresh single-event output directory'
    base.write_json(selection_path, dict(distance=distance, sample_indices=chosen.tolist(),
        shots=shots, seed=args.seed + distance, selected_before_inspecting_events=True,
        source_selection_sha256=base.sha(source / 'selection.json'),
        sampling='Uniform subset of frozen UF-only stratum; test every realized CZ event'))
    a = previous.Analysis(distance)
    assert a.directory == Path(selection['source'])
    uf = CorrelatedUnionFindDecoder(a.graph, correlation_rules=[
        (s, t, w) for s, rules in enumerate(a.rules) for t, w in rules])
    circuit = stim.Circuit.from_file(a.directory / 'circuit.stim')
    ops, roles, classes, _ = base.inventory(circuit)
    cz_instructions = [i for i, op in enumerate(ops) if op.name == 'DEPOLARIZE2']
    assert len(cz_instructions) == 16 * distance
    cz_slot = {index: (k // 4 + 1, k % 4 + 1) for k, index in enumerate(cz_instructions)}
    events, by_instruction, log_hashes = [], defaultdict(list), {}
    for cohort_index, shot in enumerate(shots):
        path = source / f'replay_shot_{shot}_faults.csv'
        log_hashes[path.name] = base.sha(path)
        with path.open() as stream:
            for fault_id, raw in enumerate(csv.DictReader(stream)):
                if raw['gate'] != 'DEPOLARIZE2':
                    continue
                index, target = int(raw['instruction_index']), int(raw['target_index'])
                q0, q1 = int(raw['q0']), int(raw['q1'])
                p0, p1 = raw['pauli0'], raw['pauli1']
                assert classes[index, target] == 'cz'
                assert [t.value for t in ops[index].targets_copy()[target:target+2]] == [q0, q1]
                pauli = p0 + p1 if roles[q0] == 'data' else p1 + p0
                assert pauli != 'II'
                round_index, slot = cz_slot[index]
                row = dict(shot=shot, cohort_index=cohort_index, fault_id=fault_id,
                    instruction_index=index, target_index=target, tick=int(raw['tick']),
                    round=round_index, CZ_slot=slot, pauli=pauli,
                    data_qubit=q0 if roles[q0] == 'data' else q1,
                    ancilla_qubit=q1 if roles[q0] == 'data' else q0)
                by_instruction[index].append((len(events), q0, p0, q1, p1))
                events.append(row)
    sim = stim.FlipSimulator(batch_size=len(events) + 1, num_qubits=circuit.num_qubits,
                             disable_stabilizer_randomization=True, seed=43)
    for index, op in enumerate(ops):
        if op.name in base.CHANNELS:
            entries = by_instruction.get(index, ())
            if not entries:
                continue
            for pauli in 'XYZ':
                mask = np.zeros((circuit.num_qubits, len(events) + 1), dtype=bool)
                for column, q0, p0, q1, p1 in entries:
                    for q, p in ((q0, p0), (q1, p1)):
                        if p == pauli:
                            mask[q, column] = True
                if mask.any():
                    sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
        elif op.name == 'M':
            sim.do(stim.CircuitInstruction('M', op.targets_copy(), []))
        else:
            sim.do(op)
    all_d, all_o = sim.get_detector_flips().T, sim.get_observable_flips().T
    assert not all_d[-1].any() and not all_o[-1].any()
    event_d, event_o = all_d[:-1], all_o[:-1]
    original_d = np.stack([a.syndrome(shot) for shot in shots])
    original_o = np.asarray(a.record.actual)[shots]
    response_validation = json.loads((source / 'response_validation.json').read_text())
    assert base.sha(source / 'responses.npz') == response_validation['responses_sha256']
    with np.load(source / 'responses.npz') as responses:
        atoms = [base.ATOM_INDEX[p] for p in base.ATOMS if p.startswith('cz_')]
        for k, old_index in enumerate(chosen):
            indices = [i for i, e in enumerate(events) if e['cohort_index'] == k]
            assert len(indices) == int(responses['event_counts'][old_index, atoms].sum())
            np.testing.assert_array_equal(np.packbits(np.logical_xor.reduce(event_d[indices]), bitorder='little'),
                np.bitwise_xor.reduce(responses['detectors'][old_index, atoms], axis=0))
            np.testing.assert_array_equal(np.logical_xor.reduce(event_o[indices]),
                np.logical_xor.reduce(responses['observables'][old_index, atoms], axis=0))
    # Independent ordinary Stim checks cover all 15 products at each distance.
    case_helpers = helper('cz_single_event_trace_helpers', HERE / 'trace_cases.py')
    independent = []
    for product in sorted({e['pauli'] for e in events}):
        index = next(i for i, e in enumerate(events) if e['pauli'] == product)
        event = events[index]
        dets, obs = case_helpers.isolated_response(ops, event['instruction_index'],
            event['data_qubit'], event['ancilla_qubit'], product)
        np.testing.assert_array_equal(dets, event_d[index])
        np.testing.assert_array_equal(obs, event_o[index])
        independent.append(dict(event_index=index, pauli=product))
    # Record whether each actual fault's direct graph components are available
    # to the original final UF pass. Choosing a different equivalent component
    # is not by itself a decoding error.
    sector = np.full(len(a.graph.adjacency), -1, dtype=np.int8)
    for label in (0, 1):
        pending = [a.graph.num_detectors - 2 + label]
        while pending:
            v = pending.pop()
            if sector[v] != -1:
                assert sector[v] == label
                continue
            sector[v] = label
            for e in a.graph.adjacency[v]:
                u, w = a.graph.endpoints[e]
                pending.append(w if u == v else u)
    assert np.all(sector >= 0)
    baseline = []
    for k, shot in enumerate(shots):
        first = a.uf._decode(original_d[k])
        weights = a.weights(first.selected_edges)
        second, growth = UnionFindDecoder(a.regraph(weights))._decode_state(original_d[k])
        matching = a.matrix.edge_ids_of(a.native.decode_to_edges_array(original_d[k], enable_correlations=True))
        truth = previous.bitmask(original_o[k])
        assert a.mask(second.selected_edges, original_d[k]) == previous.bitmask(a.record.baselines['joint_correlated_uf'][shot]) != truth
        assert a.mask(matching, original_d[k]) == previous.bitmask(a.record.baselines['joint_correlated_mwpm'][shot]) == truth
        baseline.append(dict(shot=shot, actual=truth, UF=second.observable_mask, MWPM=truth))
        uset, mset = set(second.selected_edges), set(matching)
        for i, event in enumerate(events):
            if event['cohort_index'] != k:
                continue
            ids = np.flatnonzero(event_d[i])
            event['detectors'] = ids.tolist()
            event['observables'] = np.flatnonzero(event_o[i]).tolist()
            event['unfired_footprint_detectors'] = ids[~original_d[k, ids]].tolist()
            components = []
            mapped = True
            for label in (0, 1):
                group = ids[sector[ids] == label].tolist()
                if not group:
                    continue
                if len(group) > 2:
                    mapped = False
                    break
                key = (group[0], -1) if len(group) == 1 else tuple(sorted(group))
                if key not in a.matrix.edge_ids:
                    mapped = False
                    break
                e = a.matrix.edge_ids[key]
                u, v = a.graph.endpoints[e]
                components.append(dict(edge=e, endpoints=list(a.graph.edges[e][:2]),
                    UF_same_cluster=growth.find(u) == growth.find(v),
                    UF_selected=e in uset, MWPM_selected=e in mset,
                    UF_discounted=bool(weights[e] < a.matrix.weights[e] - 1e-10)))
            event['direct_graph_components'] = components if mapped else None
            if mapped:
                a.mask([c['edge'] for c in components], event_d[i])
    global STATE
    STATE = a, uf, original_d, original_o, event_d, event_o, events
    predictions = np.empty((len(events), 2, 3), dtype=np.int64)
    start = time.monotonic()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for done, (index, result) in enumerate(pool.imap_unordered(decode_event, range(len(events)), chunksize=8), 1):
            predictions[index] = result
            if done % 512 == 0 or done == len(events):
                print(f'd={distance}: {done}/{len(events)} single CZ events, {time.monotonic()-start:.1f}s', flush=True)
    for event, result in zip(events, predictions, strict=True):
        event['removed'] = dict(zip(('actual', 'UF', 'MWPM'), result[0].tolist(), strict=True))
        event['alone'] = dict(zip(('actual', 'UF', 'MWPM'), result[1].tolist(), strict=True))
    base.write_json(output / 'single_events.json', dict(distance=distance, baseline=baseline, events=events))
    np.savez_compressed(output / 'single_event_responses.npz',
        detectors=np.packbits(event_d, axis=1, bitorder='little'), observables=event_o)
    base.write_json(output / 'single_event_validation.json', dict(
        events=len(events), selected_shots=len(shots), all_baselines_match_saved_predictions=True,
        family_response_XORs_match_previous_archive=True, ideal_control_zero=True,
        ordinary_Stim_checks=independent, forced_seeds=[1, 123456], forced_samples_per_seed=2,
        every_correction_syndrome_checked=True, elapsed_decoding_seconds=time.monotonic()-start,
        responses_sha256=base.sha(output / 'single_event_responses.npz'),
        source_log_hashes=log_hashes, source_hashes={str(p): base.sha(p) for p in
            (Path(__file__).resolve(), HERE/'analyze.py', HERE/'trace_cases.py', PREVIOUS/'analyze.py', OLD/'analyze.py')},
        circuit_sha256=base.sha(a.directory/'circuit.stim'), dem_sha256=base.sha(a.directory/'model.dem'),
        inputs=selection['hashes'], source_work_directory=str(source.resolve())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-work-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--shots', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026091719)
    parser.add_argument('--workers', type=int, default=48)
    args = parser.parse_args()
    for distance in (7, 9):
        run_distance(args, distance)


if __name__ == '__main__':
    main()
