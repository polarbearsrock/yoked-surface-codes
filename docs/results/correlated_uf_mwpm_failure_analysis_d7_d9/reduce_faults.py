"""Find single-deletion-irreducible subsets of actual logged physical faults.

The predicate is correlated UF fails and correlated MWPM succeeds. This is a
counterfactual diagnostic at fixed original graph weights, not new sampling or
a proof of globally minimum fault weight. Neither decoder receives truth.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import stim

from analyze import Analysis, CASE_ROOTS, bitmask, sha, write, wrong_bits, logical_basis, contains
from yoked.decoders import CorrelatedUnionFindDecoder, UnionFindDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    a = Analysis(7)
    directory = next(root / f'shot_{args.shot}' for root in CASE_ROOTS
                     if (root / f'shot_{args.shot}').is_dir())
    faults = json.loads((directory / 'physical_faults.json').read_text())
    with np.load(directory / 'fault_responses.npz') as data:
        detectors, observables = data['detectors'], data['observables']
    np.testing.assert_array_equal(np.logical_xor.reduce(detectors, axis=0), a.syndrome(args.shot))
    np.testing.assert_array_equal(np.logical_xor.reduce(observables, axis=0), a.record.actual[args.shot])
    decoder = CorrelatedUnionFindDecoder(a.graph, correlation_rules=[
        (source, target, weight) for source, rules in enumerate(a.rules) for target, weight in rules])
    cache = {}

    def outcome(ids):
        key = tuple(ids)
        if key not in cache:
            s = np.logical_xor.reduce(detectors[ids], axis=0)
            actual = bitmask(np.logical_xor.reduce(observables[ids], axis=0))
            u = decoder._decode(s)
            umask = a.mask(u.selected_edges, s)
            m = a.matrix.edge_ids_of(a.native.decode_to_edges_array(s, enable_correlations=True))
            mmask = a.mask(m, s)
            cache[key] = dict(actual_mask=actual, CUF_mask=umask, CMWPM_mask=mmask,
                              CUF_wrong=wrong_bits(umask, actual), CMWPM_wrong=wrong_bits(mmask, actual),
                              qualifies=umask != actual == mmask)
        return cache[key]

    ids = list(range(len(faults)))
    original = outcome(ids)
    assert original['qualifies']
    start = time.perf_counter()
    granularity = 2
    while len(ids) >= 2:
        changed = False
        for chunk in np.array_split(np.arange(len(ids)), granularity):
            remove = set(chunk.tolist())
            candidate = [fid for i, fid in enumerate(ids) if i not in remove]
            if outcome(candidate)['qualifies']:
                ids = candidate
                granularity = max(2, granularity - 1)
                changed = True
                print(f'{args.shot}: retained {len(ids)} faults after {len(cache)} decodes', flush=True)
                break
        if not changed:
            if granularity >= len(ids):
                break
            granularity = min(len(ids), 2 * granularity)
    single_deletions = {str(fid): outcome([other for other in ids if other != fid]) for fid in ids}
    assert not any(o['qualifies'] for o in single_deletions.values())
    s = np.logical_xor.reduce(detectors[ids], axis=0)
    actual_bits = np.logical_xor.reduce(observables[ids], axis=0)
    actual = bitmask(actual_bits)
    first = a.uf._decode(s)
    wu = a.weights(first.selected_edges)
    second, growth = UnionFindDecoder(a.regraph(wu))._decode_state(s)
    mfirst = a.matrix.edge_ids_of(a.native.decode_to_edges_array(s))
    wm = a.weights(mfirst)
    swap_masks = dict(UU=second.observable_mask, UM=a.mask(a.matching_edges(s, wu), s),
                     MU=a.mask(UnionFindDecoder(a.regraph(wm))._decode(s).selected_edges, s),
                     MM_rebuilt=a.mask(a.matching_edges(s, wm), s))
    roots = np.array([growth.find(v) for v in range(len(a.graph.adjacency))])
    allowed = np.flatnonzero(roots[a.endpoints[:, 0]] == roots[a.endpoints[:, 1]])
    partition_ok = contains(logical_basis(a.graph, allowed), second.observable_mask ^ actual)
    # Independently propagate the reduced physical circuit with ordinary Stim.
    circuit = stim.Circuit.from_file(a.directory / 'circuit.stim')
    def expand(c):
        for op in c:
            if isinstance(op, stim.CircuitRepeatBlock):
                for _ in range(op.repeat_count):
                    yield from expand(op.body_copy())
            else:
                yield op
    by_instruction = {}
    for fid in ids:
        by_instruction.setdefault(faults[fid]['instruction_index'], []).append(faults[fid])
    forced = stim.Circuit()
    channels = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR'}
    for index, op in enumerate(expand(circuit)):
        group = by_instruction.get(index, [])
        if op.name in channels:
            for f in group:
                assert f['gate'] == op.name
                for loc in f['locations']:
                    if loc['pauli'] != 'I':
                        forced.append(loc['pauli'] + '_ERROR', [loc['qubit']], 1)
        elif op.name == 'M':
            flipped = {f['q0'] for f in group}
            for target in op.targets_copy():
                forced.append('M', [target], int(target.value in flipped))
        else:
            assert not (stim.gate_data(op.name).is_noisy_gate and op.gate_args_copy() and op.gate_args_copy()[0])
            forced.append(op)
    for seed in (1, 123456):
        d, o = forced.compile_detector_sampler(seed=seed).sample(4, separate_observables=True)
        np.testing.assert_array_equal(d, np.broadcast_to(s, d.shape))
        np.testing.assert_array_equal(o, np.broadcast_to(actual_bits, o.shape))
    result = dict(source_shot=args.shot, original_events=len(faults), original=original,
                  retained_events=len(ids), fault_ids=ids, faults=[faults[fid] for fid in ids],
                  reduced=outcome(ids), single_deletions=single_deletions,
                  individual_faults={str(fid): outcome([fid]) for fid in ids},
                  defect_ids=np.flatnonzero(s).tolist(), actual_observables=np.flatnonzero(actual_bits).tolist(),
                  swap_wrong={k: wrong_bits(v, actual) for k, v in swap_masks.items()},
                  second_U_partition_allows_truth=partition_ok,
                  forced_stim_samples_verified=8, source=str(directory),
                  physical_source_sha256=sha(directory/'physical_faults.json'),
                  response_source_sha256=sha(directory/'fault_responses.npz'),
                  script_sha256=sha(__file__), predicate_evaluations=len(cache),
                  seconds=time.perf_counter()-start)
    write(args.out/f'reduced_{args.shot}.json', result)
    forced.to_file(args.out/f'reduced_{args.shot}.stim')
    print(json.dumps(dict(shot=args.shot, ids=ids, result=result['reduced'],
                          swap_wrong=result['swap_wrong'], partition_allows_truth=partition_ok)), flush=True)


if __name__ == '__main__':
    main()
