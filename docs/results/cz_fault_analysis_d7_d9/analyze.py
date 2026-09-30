"""Resolve the saved CZ-family ablation into its 15 data/ancilla Paulis.

Reuse the original stratified selections and recovered physical responses.
No shots are regenerated, and every intervention reruns both complete
correlated decoders with the original noise model. These are whole-subfamily
deletions, not per-event causal attributions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'physical_fault_ablation_d7_d9'


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = helper('cz_previous_ablation', PREVIOUS / 'analyze.py')
audit = helper('cz_previous_audit', PREVIOUS / 'audit.py')
PAULIS = tuple(atom for atom in base.ATOMS if atom.startswith('cz_'))
base.VARIANTS.update({atom: [atom] for atom in PAULIS})


def prepare(source, destination, distance):
    old = source / f'd{distance}'
    target = destination / f'd{distance}'
    target.mkdir()
    selection = json.loads((old / 'selection.json').read_text())
    assert selection == json.loads((PREVIOUS / f'd{distance}/selection.json').read_text())
    validation = json.loads((old / 'response_validation.json').read_text())
    assert base.sha(old / 'responses.npz') == validation['responses_sha256']
    raw = Path(selection['source'])
    for filename, key in [('circuit.stim', 'circuit_sha256'), ('model.dem', 'dem_sha256')]:
        assert base.sha(raw / filename) == selection['hashes'][key]
    dets = np.load(raw / 'detectors_packed.npy', mmap_mode='r')
    obs = np.load(raw / 'actual_observables_packed.npy', mmap_mode='r')
    assert base.packed_sample_hash(dets, obs) == selection['hashes']['payload_sha256']
    with np.load(old / 'original.npz') as original:
        rows = selection['selected_rows']
        np.testing.assert_array_equal(original['detectors'], dets[rows])
        np.testing.assert_array_equal(original['observables'],
            np.unpackbits(obs[rows], axis=1, count=12, bitorder='little'))
    for filename in ('selection.json', 'original.npz', 'responses.npz'):
        shutil.copy2(old / filename, target / filename)
    base.write_json(target / 'input_validation.json', dict(
        source_work_directory=str(old.resolve()), selection_unchanged=True,
        raw_payload_and_model_hashes_verified=True,
        selected_original_responses_verified=True,
        inputs={filename: base.sha(old / filename) for filename in
                ('selection.json', 'original.npz', 'responses.npz', 'response_validation.json')},
    ))
    return target, selection


def verify_and_bound(directory, selection):
    data = dict(np.load(directory / 'decoded_pauli.npz'))
    summary_path = directory / 'summary_pauli.json'
    summary = json.loads(summary_path.read_text())
    population = np.asarray(selection['population_counts'])
    strata = np.asarray(selection['strata'])
    total = int(population.sum())
    original_gap = (population[1] - population[2]) / total
    old = dict(np.load(PREVIOUS / f'd{selection["distance"]}/decoded_primary.npz'))
    for name in ('baseline', 'cz'):
        j = list(data['variants']).index(name)
        k = list(old['variants']).index(name)
        np.testing.assert_array_equal(data['predictions'][:, j], old['predictions'][:, k])
        np.testing.assert_array_equal(data['actual'][:, j], old['actual'][:, k])
    for j, row in enumerate(summary['rows']):
        failed = (data['predictions'][:, j] != data['actual'][:, j, None]).any(axis=2)
        outcome = failed[:, 0].astype(int) + 2 * failed[:, 1]
        gap, lower, upper = 0., 0, 0
        for code, size in enumerate(population):
            sample = outcome[strata == code]
            counts = np.bincount(sample, minlength=4)
            assert counts.tolist() == row['outcomes_by_stratum'][code]
            gap += size / total * (counts[1] - counts[2]) / len(sample)
            plus = audit.count_interval(int(counts[1]), len(sample), int(size), .05 / 16)
            minus = audit.count_interval(int(counts[2]), len(sample), int(size), .05 / 16)
            lower += plus[0] - minus[1]
            upper += plus[1] - minus[0]
        np.testing.assert_allclose(gap, row['gap_after'], atol=1e-12)
        if j:
            row['fraction_of_gap_removed_finite_population_ci95'] = [
                (original_gap - upper / total) / original_gap,
                (original_gap - lower / total) / original_gap]
    summary['validation'] = dict(
        independent_weighted_outcome_count_check=True,
        baseline_and_all_CZ_deletion_match_previous_run=True,
        finite_population_intervals='Marginal 95% conservative hypergeometric bounds; '
            'simultaneous over eight stratum/outcome probabilities per variant, '
            'not over the 15 Pauli comparisons.',
    )
    base.write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-work-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--distances', type=int, nargs='+', default=[7, 9], choices=[7, 9])
    args = parser.parse_args()
    args.work_dir.mkdir(exist_ok=True)
    assert not (args.work_dir / 'protocol.json').exists(), 'Use a fresh output directory'
    base.write_json(args.work_dir / 'protocol.json', dict(
        distances=args.distances, source_work_dir=str(args.source_work_dir.resolve()),
        ordered_variants=['baseline', *PAULIS, 'cz'], pauli_order='data, then ancilla',
        selected_shots_per_distance=1024, per_original_outcome_stratum=256,
        sampling='Reuse frozen selections from the previous physical-fault ablation',
        intervention='Delete all realized CZ events of one Pauli product; preserve other faults',
        estimand='Population-weighted change in P(UF failure) - P(MWPM failure)',
        decoding='Original DEM, full correlated UF and native correlated MWPM; recompute both passes',
        interpretation='Overlapping family deletion effects, not additive attribution or per-event harm',
        bootstrap_replicates=20000, bootstrap_seed=2026091711,
        comparison_scope='Exploratory 15-Pauli comparison; marginal intervals do not establish a unique winner',
        source_hashes={str(p): base.sha(p) for p in
                       (Path(__file__).resolve(), PREVIOUS / 'analyze.py', PREVIOUS / 'audit.py')},
    ))
    for distance in args.distances:
        directory, selection = prepare(args.source_work_dir, args.work_dir, distance)
        options = argparse.Namespace(distance=distance, variants=[*PAULIS, 'cz'],
            workers=args.workers, label='pauli', bootstrap_seed=2026091711,
            bootstrap_replicates=20000)
        base.decode(options, directory, selection)
        base.summarize(options, directory, selection)
        verify_and_bound(directory, selection)
        base.write_json(directory / 'runner_validation.json', dict(
            script_sha256=base.sha(__file__), inherited_decoder='Python production correlated UF',
            independent_population_audit_passed=True,
        ))


if __name__ == '__main__':
    main()
