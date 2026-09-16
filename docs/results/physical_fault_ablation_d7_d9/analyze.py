"""Paired physical-fault ablations on a stratified sample of saved 100k shots.

This is offline analysis, not a decoder. Selection uses the four saved decoder
outcomes; population estimates restore their original proportions. All variants
decode with the original DEM and recompute their own correlation evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
import pymatching
import sinter
import stim

from yoked.decoders import CorrelatedUnionFindDecoder
from yoked.hierarchical._collect import packed_sample_hash
from yoked.hierarchical._record import load_record


STRATA = ('both_correct', 'uf_only_fails', 'mwpm_only_fails', 'both_fail')
SINGLE_FAMILIES = ('data_wait', 'data_1q', 'ancilla_1q', 'ancilla_post_measurement')
ATOMS = ('readout', 'reset') + tuple(
    f'{family}_{pauli}' for family in SINGLE_FAMILIES for pauli in 'XYZ'
) + tuple(f'cz_{a}{b}' for a, b in itertools.product('IXYZ', repeat=2) if a + b != 'II')
ATOM_INDEX = {name: k for k, name in enumerate(ATOMS)}
VARIANTS = {
    'readout': ['readout'],
    'reset': ['reset'],
    'data_wait': [a for a in ATOMS if a.startswith('data_wait_')],
    'data_1q': [a for a in ATOMS if a.startswith('data_1q_')],
    'ancilla_1q': [a for a in ATOMS if a.startswith('ancilla_')],
    'cz': [a for a in ATOMS if a.startswith('cz_')],
    'cz_data_only': [f'cz_{p}I' for p in 'XYZ'],
    'cz_ancilla_only': [f'cz_I{p}' for p in 'XYZ'],
    'cz_both': [f'cz_{a}{b}' for a, b in itertools.product('XYZ', repeat=2)],
    'cz_with_Y': [a for a in ATOMS if a.startswith('cz_') and 'Y' in a],
    'cz_without_Y': [a for a in ATOMS if a.startswith('cz_') and 'Y' not in a],
    **{f'data_wait_{p}': [f'data_wait_{p}'] for p in 'XYZ'},
}
PRIMARY = ('readout', 'reset', 'data_wait', 'data_1q', 'ancilla_1q', 'cz')
CHANNELS = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR'}
DECODING_STATE = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def expand(circuit):
    # Keep SHIFT_COORDS: the C++ observer indexes for_each_operation this way.
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            for _ in range(op.repeat_count):
                yield from expand(op.body_copy())
        else:
            yield op


def prepare(args):
    args.work_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.work_dir / 'protocol.json', dict(
        per_stratum=args.per_stratum, distances=[7, 9], selection_seed=args.seed,
        atoms=ATOMS, primary_variants=PRIMARY, variants=VARIANTS,
        sampling='uniform without replacement within each of four baseline outcome strata',
        intervention='remove every realized physical event in the named family; keep all others',
        decoding='original DEM, full correlated UF and native correlated PyMatching',
        estimand='population-weighted reduction in P(UF failure) - P(MWPM failure)',
        tracer=str(args.tracer.resolve()), tracer_sha256=sha(args.tracer),
    ))
    for distance in (7, 9):
        loaded = load_record(args.record_root / f'd{distance}/evaluation')
        record = loaded.record
        run = loaded.manifest['baselines']['run']
        source = Path(run['directory'])
        dets = np.load(source / 'detectors_packed.npy', mmap_mode='r')
        obs = np.load(source / 'actual_observables_packed.npy', mmap_mode='r')
        assert packed_sample_hash(dets, obs) == run['payload_sha256']
        for name, key in [('circuit.stim', 'circuit_sha256'), ('model.dem', 'dem_sha256')]:
            assert sha(source / name) == run[key]
        np.testing.assert_array_equal(
            np.unpackbits(obs, axis=1, count=12, bitorder='little'), record.actual)
        uf = np.any(record.baselines['joint_correlated_uf'] != record.actual, axis=1)
        mwpm = np.any(record.baselines['joint_correlated_mwpm'] != record.actual, axis=1)
        codes = uf.astype(np.int8) + 2 * mwpm.astype(np.int8)
        rng = np.random.default_rng(args.seed + distance)
        selected, strata, populations = [], [], []
        for code, name in enumerate(STRATA):
            candidates = np.flatnonzero(codes == code)
            populations.append(len(candidates))
            rows = np.sort(rng.choice(candidates, size=args.per_stratum, replace=False))
            selected.extend(rows.tolist())
            strata.extend([code] * len(rows))
        directory = args.work_dir / f'd{distance}'
        directory.mkdir()
        write_json(directory / 'selection.json', dict(
            distance=distance, source=str(source), record=str(args.record_root / f'd{distance}/evaluation'),
            hashes={key: run[key] for key in ('payload_sha256', 'circuit_sha256', 'dem_sha256')},
            selected_rows=selected, strata=strata, stratum_names=STRATA,
            population_counts=populations, population_shots=len(codes),
            sampling_seed=42, full_sampling_batch=len(codes), selection_seed=args.seed + distance,
        ))
        np.savez_compressed(directory / 'original.npz',
                            detectors=dets[selected], observables=record.actual[selected],
                            uf=record.baselines['joint_correlated_uf'][selected],
                            mwpm=record.baselines['joint_correlated_mwpm'][selected])
        print(f'd={distance}: populations={populations}, selected={len(selected)}', flush=True)


def trace(args, directory, selection):
    protocol = json.loads((args.work_dir / 'protocol.json').read_text())
    command = [protocol['tracer'], str(Path(selection['source']) / 'circuit.stim'),
               str(directory / 'replay'), str(selection['sampling_seed']),
               str(selection['full_sampling_batch']),
               ','.join(map(str, selection['selected_rows']))]
    with (directory / 'trace.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    validation = {}
    for kind, saved in [('detectors', 'detectors_packed.npy'),
                        ('observables', 'actual_observables_packed.npy')]:
        original = np.load(Path(selection['source']) / saved, mmap_mode='r')
        replayed = np.fromfile(directory / f'replay_{kind}.b8', dtype=np.uint8).reshape(original.shape)
        np.testing.assert_array_equal(replayed, original)
        validation[kind] = dict(all_original_rows_identical=True, shape=list(original.shape),
                                replay_sha256=sha(directory / f'replay_{kind}.b8'))
    write_json(directory / 'replay_validation.json', validation)
    print(f'd={args.distance}: all {selection["population_shots"]} original shots replay exactly', flush=True)


def inventory(circuit):
    coords = circuit.get_final_qubit_coordinates()
    roles = {}
    for q, (x, y, *_) in coords.items():
        roles[q] = ('data' if y >= 0 else 'reference') if x == int(x) and y == int(y) else 'ancilla'
    last, classes, counts = {}, {}, Counter()
    ops = list(expand(circuit))
    for index, op in enumerate(ops):
        targets = op.targets_copy()
        if op.name in CHANNELS or (op.name == 'M' and op.gate_args_copy()):
            probability = op.gate_args_copy()[0]
            stride = 2 if op.name == 'DEPOLARIZE2' else 1
            for k in range(0, len(targets), stride):
                q = targets[k].value
                assert roles[q] != 'reference', 'Unexpected noise on a reference qubit'
                if op.name == 'M':
                    assert roles[q] == 'ancilla' and np.isclose(probability, .015)
                    family = 'readout'
                elif op.name == 'X_ERROR':
                    assert roles[q] == 'ancilla' and last[q] == 'R' and np.isclose(probability, .006)
                    family = 'reset'
                elif op.name == 'DEPOLARIZE2':
                    other = targets[k + 1].value
                    assert {roles[q], roles[other]} == {'data', 'ancilla'}
                    assert last[q] == last[other] == 'CZ' and np.isclose(probability, .003)
                    family = 'cz'
                elif op.name == 'DEPOLARIZE1':
                    if np.isclose(probability, .006):
                        assert roles[q] == 'data'
                        family = 'data_wait'
                    elif np.isclose(probability, .003):
                        assert roles[q] == 'ancilla' and last[q] == 'M'
                        family = 'ancilla_post_measurement'
                    else:
                        assert np.isclose(probability, .0003)
                        family = roles[q] + '_1q'
                else:
                    raise ValueError(f'Unexpected noise channel {op}')
                classes[index, k] = family
                counts[family, probability] += 1
        elif stim.gate_data(op.name).is_noisy_gate:
            assert not op.gate_args_copy() or not op.gate_args_copy()[0], op
        if (stim.gate_data(op.name).is_unitary or stim.gate_data(op.name).is_reset
                or stim.gate_data(op.name).produces_measurements):
            for target in targets:
                if target.is_qubit_target:
                    last[target.value] = op.name
    return ops, roles, classes, [dict(family=f, probability=p, locations=n)
                                for (f, p), n in sorted(counts.items())]


def read_faults(directory, selection, ops, roles, classes):
    by_instruction = defaultdict(list)
    counts = np.zeros((len(selection['selected_rows']), len(ATOMS)), dtype=np.int32)
    targets = {index: ops[index].targets_copy() for index in {i for i, _ in classes}}
    for shot_index, row in enumerate(selection['selected_rows']):
        with (directory / f'replay_shot_{row}_faults.csv').open() as stream:
            for raw in csv.DictReader(stream):
                index, target_index = int(raw['instruction_index']), int(raw['target_index'])
                q0, q1, p0, p1 = int(raw['q0']), int(raw['q1']), raw['pauli0'], raw['pauli1']
                op = ops[index]
                assert op.name == raw['gate'] and targets[index][target_index].value == q0
                family = classes[index, target_index]
                if family == 'cz':
                    assert targets[index][target_index + 1].value == q1
                    pair = p0 + p1 if roles[q0] == 'data' else p1 + p0
                    atom = ATOM_INDEX['cz_' + pair]
                elif family in ('readout', 'reset'):
                    assert q1 == -1 and p0 == ('readout' if family == 'readout' else 'X')
                    atom = ATOM_INDEX[family]
                else:
                    assert q1 == -1
                    atom = ATOM_INDEX[f'{family}_{p0}']
                by_instruction[index].append((shot_index, atom, q0, p0, q1, p1))
                counts[shot_index, atom] += 1
    return by_instruction, counts


def forced_circuit(ops, by_instruction, shot_index, removed):
    forced = stim.Circuit()
    for index, op in enumerate(ops):
        events = [f for f in by_instruction.get(index, ()) if f[0] == shot_index and f[1] not in removed]
        if op.name in CHANNELS:
            for _, _, q0, p0, q1, p1 in events:
                for q, p in ((q0, p0), (q1, p1)):
                    if p in 'XYZ':
                        forced.append(p + '_ERROR', [q], 1)
        elif op.name == 'M':
            flipped = {f[2] for f in events}
            for target in op.targets_copy():
                forced.append('M', [target], int(target.value in flipped))
        else:
            forced.append(op)
    return forced


def recover(args, directory, selection):
    circuit = stim.Circuit.from_file(Path(selection['source']) / 'circuit.stim')
    ops, roles, classes, location_counts = inventory(circuit)
    by_instruction, counts = read_faults(directory, selection, ops, roles, classes)
    shots, atoms = counts.shape
    width = atoms + 1  # one response per atomic family, plus all actual faults
    batch_size = shots * width + 1  # final column is an ideal control
    sim = stim.FlipSimulator(batch_size=batch_size, num_qubits=circuit.num_qubits,
                             disable_stabilizer_randomization=True, seed=43)
    for index, op in enumerate(ops):
        events = by_instruction.get(index, ())
        if op.name in CHANNELS or op.name == 'M':
            for pauli in ('X',) if op.name == 'M' else 'XYZ':
                mask = np.zeros((circuit.num_qubits, batch_size), dtype=np.bool_)
                for shot_index, atom, q0, p0, q1, p1 in events:
                    for q, p in ((q0, p0), (q1, p1)):
                        if p == pauli or (op.name == 'M' and p == 'readout'):
                            mask[q, shot_index * width + atom] ^= True
                            mask[q, shot_index * width + atoms] ^= True
                if np.any(mask):
                    sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
                if op.name == 'M':
                    sim.do(stim.CircuitInstruction('M', op.targets_copy(), []))
                    if np.any(mask):
                        sim.broadcast_pauli_errors(pauli='X', mask=mask)
        else:
            sim.do(op)
    dets = sim.get_detector_flips().T
    obs = sim.get_observable_flips().T
    assert not np.any(dets[-1]) and not np.any(obs[-1])
    dets = dets[:-1].reshape(shots, width, circuit.num_detectors)
    obs = obs[:-1].reshape(shots, width, circuit.num_observables)
    original = np.load(directory / 'original.npz')
    original_dets = np.unpackbits(original['detectors'], axis=1, count=circuit.num_detectors,
                                  bitorder='little').astype(bool)
    for responses, saved in ((dets, original_dets), (obs, original['observables'])):
        np.testing.assert_array_equal(responses[:, -1], saved)
        np.testing.assert_array_equal(np.logical_xor.reduce(responses[:, :-1], axis=1), saved)
    post_m = [ATOM_INDEX[f'ancilla_post_measurement_{p}'] for p in 'XYZ']
    assert not np.any(dets[:, post_m]) and not np.any(obs[:, post_m])
    # Check independently using an ordinary (randomized-stabilizer) sampler, one
    # shot from each stratum, both unchanged and with each primary family removed.
    checked = []
    for code in range(4):
        shot_index = selection['strata'].index(code)
        for name in ('baseline', *PRIMARY):
            removed = [] if name == 'baseline' else [ATOM_INDEX[a] for a in VARIANTS[name]]
            forced = forced_circuit(ops, by_instruction, shot_index, set(removed))
            expected_d = original_dets[shot_index].copy()
            expected_o = original['observables'][shot_index].astype(bool).copy()
            for atom in removed:
                expected_d ^= dets[shot_index, atom]
                expected_o ^= obs[shot_index, atom]
            for seed in (1, 123456):
                d, o = forced.compile_detector_sampler(seed=seed).sample(2, separate_observables=True)
                np.testing.assert_array_equal(d, np.broadcast_to(expected_d, d.shape))
                np.testing.assert_array_equal(o, np.broadcast_to(expected_o, o.shape))
            checked.append(dict(row=selection['selected_rows'][shot_index], variant=name))
    np.savez_compressed(directory / 'responses.npz',
                        detectors=np.packbits(dets[:, :-1], axis=2, bitorder='little'),
                        observables=obs[:, :-1], event_counts=counts)
    write_json(directory / 'response_validation.json', dict(
        selected_shots=shots, atomic_families=ATOMS, aggregate_matches_saved_shots=True,
        xor_of_family_responses_matches_saved_shots=True, ideal_control_zero=True,
        post_measurement_ancilla_noise_has_zero_response=True,
        independent_forced_circuit_checks=checked, forced_seeds=[1, 123456],
        forced_samples_per_seed=2, physical_location_inventory=location_counts,
        total_logged_events=int(counts.sum()), responses_sha256=sha(directory / 'responses.npz'),
    ))
    print(f'd={args.distance}: recovered {counts.sum()} events on {shots} shots; all response checks pass', flush=True)


def decode_one(shot_index):
    original, responses, variants, uf, mwpm = DECODING_STATE
    predictions = np.empty((len(variants), 2, uf.graph.num_observables), dtype=np.bool_)
    actual = np.empty((len(variants), uf.graph.num_observables), dtype=np.bool_)
    for j, name in enumerate(variants):
        d = original['detectors'][shot_index].copy()
        o = original['observables'][shot_index].astype(bool).copy()
        if name != 'baseline':
            for atom in VARIANTS[name]:
                a = ATOM_INDEX[atom]
                d ^= responses['detectors'][shot_index, a]
                o ^= responses['observables'][shot_index, a]
        syndrome = np.unpackbits(d, count=uf.graph.num_detectors, bitorder='little').astype(bool)
        predictions[j, 0] = uf.decode(syndrome)
        predictions[j, 1] = mwpm.decode(syndrome, enable_correlations=True)
        actual[j] = o
    np.testing.assert_array_equal(predictions[0, 0], original['uf'][shot_index])
    np.testing.assert_array_equal(predictions[0, 1], original['mwpm'][shot_index])
    return shot_index, predictions, actual


def decode(args, directory, selection):
    global DECODING_STATE
    variants = ('baseline', *(args.variants or PRIMARY))
    assert len(set(variants)) == len(variants)
    original = dict(np.load(directory / 'original.npz'))
    responses = dict(np.load(directory / 'responses.npz'))
    dem = stim.DetectorErrorModel.from_file(Path(selection['source']) / 'model.dem')
    uf = CorrelatedUnionFindDecoder.from_dem(dem)
    mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    DECODING_STATE = original, responses, variants, uf, mwpm
    n = len(selection['selected_rows'])
    predictions = np.empty((n, len(variants), 2, dem.num_observables), dtype=np.bool_)
    actual = np.empty((n, len(variants), dem.num_observables), dtype=np.bool_)
    start = time.monotonic()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for done, (shot, p, o) in enumerate(pool.imap_unordered(decode_one, range(n), chunksize=1), 1):
            predictions[shot], actual[shot] = p, o
            if done % 64 == 0 or done == n:
                print(f'd={args.distance}: {done}/{n} shots, {len(variants)} variants, '
                      f'{time.monotonic() - start:.1f}s', flush=True)
    path = directory / f'decoded_{args.label}.npz'
    np.savez_compressed(path, predictions=predictions, actual=actual, variants=variants)
    source_paths = ['src/yoked/decoders/_union_find.py', 'src/yoked/decoders/_graph.py',
                    'src/yoked/decoders/_correlated_union_find.py', 'src/yoked/decoders/_correlations.py']
    write_json(directory / f'decode_validation_{args.label}.json', dict(
        variants=variants, selected_shots=n, all_baselines_match_saved_predictions=True,
        original_dem_used_for_all_variants=True, recompute_correlation_evidence=True,
        workers=args.workers, elapsed_seconds=time.monotonic() - start,
        result_sha256=sha(path), source_hashes={p: sha(p) for p in source_paths},
        analysis_sha256=sha(__file__), stim_version=stim.__version__,
        pymatching_version=pymatching.__version__,
    ))


def summarize(args, directory, selection):
    data = dict(np.load(directory / f'decoded_{args.label}.npz'))
    failures = np.any(data['predictions'] != data['actual'][:, :, None, :], axis=3)
    strata = np.array(selection['strata'])
    population = np.array(selection['population_counts'])
    weights = population / population.sum()
    gaps = failures[:, :, 0].astype(float) - failures[:, :, 1]
    delta = gaps[:, :1] - gaps
    rng = np.random.default_rng(args.bootstrap_seed)
    reps = args.bootstrap_replicates
    rate_estimate = np.zeros(failures.shape[1:])
    rate_bootstrap = np.zeros((reps, *failures.shape[1:]))
    delta_estimate = np.zeros(failures.shape[1])
    delta_bootstrap = np.zeros((reps, failures.shape[1]))
    mean_events = np.zeros(len(ATOMS))
    events = np.load(directory / 'responses.npz')['event_counts']
    for code, weight in enumerate(weights):
        chosen = np.flatnonzero(strata == code)
        rate_estimate += weight * failures[chosen].mean(axis=0)
        delta_estimate += weight * delta[chosen].mean(axis=0)
        mean_events += weight * events[chosen].mean(axis=0)
        # Multinomial bootstrap weights, paired across variants and decoders.
        draws = rng.multinomial(len(chosen), np.full(len(chosen), 1 / len(chosen)), size=reps)
        rate_bootstrap += weight * np.einsum('bs,svc->bvc', draws, failures[chosen]) / len(chosen)
        delta_bootstrap += weight * draws @ delta[chosen] / len(chosen)
    baseline_gap = (population[1] - population[2]) / population.sum()
    np.testing.assert_allclose(rate_estimate[0],
                               [(population[1] + population[3]) / population.sum(),
                                (population[2] + population[3]) / population.sum()])
    rows = []
    for j, name in enumerate(data['variants']):
        name = str(name)
        removed_events = 0 if name == 'baseline' else sum(mean_events[ATOM_INDEX[a]] for a in VARIANTS[name])
        gap_after = rate_estimate[j, 0] - rate_estimate[j, 1]
        row = dict(variant=name, mean_events_removed=float(removed_events),
                   uf_failure_probability=float(rate_estimate[j, 0]),
                   mwpm_failure_probability=float(rate_estimate[j, 1]),
                   gap_after=float(gap_after), gap_reduction=float(delta_estimate[j]),
                   gap_reduction_ci95=np.quantile(delta_bootstrap[:, j], [.025, .975]).tolist(),
                   fraction_of_gap_removed=float(delta_estimate[j] / baseline_gap),
                   fraction_of_gap_removed_ci95=np.quantile(delta_bootstrap[:, j] / baseline_gap, [.025, .975]).tolist(),
                   outcomes_by_stratum=[])
        for code in range(4):
            sample = failures[strata == code, j]
            outcome = sample[:, 0].astype(int) + 2 * sample[:, 1]
            row['outcomes_by_stratum'].append(np.bincount(outcome, minlength=4).tolist())
        for k, decoder in enumerate(('uf', 'mwpm')):
            conversion = lambda p: sinter.shot_error_rate_to_piece_error_rate(
                p, pieces=6 * 4 * args.distance, values=8)
            row[decoder + '_normalized_ler'] = float(conversion(rate_estimate[j, k]))
            row[decoder + '_failure_probability_ci95'] = np.quantile(rate_bootstrap[:, j, k], [.025, .975]).tolist()
        rows.append(row)
    comparisons = []
    for a, b in itertools.combinations(range(1, len(rows)), 2):
        comparisons.append(dict(first=rows[a]['variant'], second=rows[b]['variant'],
            difference_in_gap_reduction=float(delta_estimate[a] - delta_estimate[b]),
            ci95=np.quantile(delta_bootstrap[:, a] - delta_bootstrap[:, b], [.025, .975]).tolist()))
    result = dict(distance=args.distance, selected_shots=len(strata), population_counts=selection['population_counts'],
                  original_population_gap=baseline_gap, rows=rows, paired_comparisons=comparisons,
                  bootstrap_seed=args.bootstrap_seed, bootstrap_replicates=reps,
                  ci_method='percentile stratified paired bootstrap; finite-population correction omitted conservatively',
                  atom_mean_events={name: float(mean_events[k]) for k, name in enumerate(ATOMS)})
    write_json(directory / f'summary_{args.label}.json', result)
    for row in rows:
        print(f'{row["variant"]:20s} gap={row["gap_after"]:.6f} '
              f'reduction={row["gap_reduction"]:.6f} '
              f'fraction={100 * row["fraction_of_gap_removed"]:.1f}% '
              f'CI={row["fraction_of_gap_removed_ci95"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'trace', 'recover', 'decode', 'summarize'])
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--record-root', type=Path)
    parser.add_argument('--tracer', type=Path)
    parser.add_argument('--distance', type=int, choices=[7, 9])
    parser.add_argument('--per-stratum', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026091700)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--variants', choices=list(VARIANTS), nargs='+')
    parser.add_argument('--label', default='primary')
    parser.add_argument('--bootstrap-seed', type=int, default=2026091711)
    parser.add_argument('--bootstrap-replicates', type=int, default=20000)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.record_root is None or args.tracer is None:
            parser.error('prepare requires --record-root and --tracer')
        prepare(args)
    else:
        if args.distance is None:
            parser.error('this stage requires --distance')
        directory = args.work_dir / f'd{args.distance}'
        selection = json.loads((directory / 'selection.json').read_text())
        globals()[args.stage](args, directory, selection)


if __name__ == '__main__':
    main()
