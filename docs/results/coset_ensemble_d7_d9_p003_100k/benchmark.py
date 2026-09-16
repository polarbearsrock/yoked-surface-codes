"""Small sequential software benchmark; run after the parallel experiment ends.

This compares Python UF implementations with native PyMatching. It measures
prototype software costs, including the ensemble's logical-rank diagnostic,
and must not be interpreted as a hardware latency comparison.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import time

import numpy as np
import pymatching
import stim

from yoked.decoders import CorrelatedUnionFindDecoder, CosetEnsembleDecoder
from yoked.hierarchical._provenance import package_versions, source_hashes, write_json_atomic
from yoked.hierarchical._record import load_record


def benchmark(record_dir, *, shots, seed):
    loaded = load_record(record_dir)
    raw = Path(loaded.manifest['baselines']['run']['directory'])
    dem = stim.DetectorErrorModel.from_file(raw / 'model.dem')
    packed = np.load(raw / 'detectors_packed.npy', mmap_mode='r')
    rows = np.sort(np.random.default_rng(seed).choice(loaded.record.shots, shots, replace=False))
    syndromes = np.unpackbits(packed[rows], axis=1, count=dem.num_detectors, bitorder='little')
    uf = CorrelatedUnionFindDecoder.from_dem(dem)
    ensemble = CosetEnsembleDecoder.from_dem(dem, correlated=True, candidates=24, seed=20260916)
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    calls = dict(correlated_uf=uf.decode, coset_ensemble_k24=ensemble.decode,
                 correlated_mwpm=lambda s: matching.decode(s, enable_correlations=True))
    times = {name: [] for name in calls}
    for call in calls.values():
        call(syndromes[0])
    for i, (row, syndrome) in enumerate(zip(rows, syndromes)):
        names = list(calls)
        names = names[i % len(names):] + names[:i % len(names)]
        for name in names:
            start = time.perf_counter_ns()
            prediction = calls[name](syndrome)
            times[name].append((time.perf_counter_ns() - start) * 1e-9)
            if name != 'coset_ensemble_k24':
                np.testing.assert_array_equal(prediction, loaded.record.baselines[f'joint_{name}'][row])
    return dict(rows=rows.tolist(), row_seed=seed, seconds={
        name: dict(mean=float(np.mean(t)), median=float(np.median(t)),
                   p95=float(np.percentile(t, 95)), total=float(sum(t))) for name, t in times.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-root', type=Path, default=Path(os.environ['TMPDIR']) /
                        'hier-three-decoders-d7-d9-p003-100k-3MxG5S')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shots', type=int, default=32)
    args = parser.parse_args()
    result = dict(
        warning='Python UF versus native PyMatching; software only, no hardware latency inference',
        warmup=1, order='rotating decoder order per shot', constructor_timing=False,
        versions=package_versions(('numpy', 'stim', 'pymatching', 'sinter')),
        python=platform.python_version(), platform=platform.platform(),
        numerical_threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        affinity_cpus=len(os.sched_getaffinity(0)),
        source_sha256=source_hashes((
            'src/yoked/decoders/_coset_ensemble.py', 'src/yoked/decoders/_correlated_union_find.py',
            'src/yoked/decoders/_union_find.py', 'src/yoked/decoders/_correlations.py',
            'src/yoked/decoders/_graph.py', 'docs/results/coset_ensemble_d7_d9_p003_100k/benchmark.py')),
        distances={str(d): benchmark(args.record_root / f'd{d}' / 'evaluation',
                                     shots=args.shots, seed=2026091600 + d) for d in (7, 9)},
    )
    write_json_atomic(args.output, result)
    print(result['distances'], flush=True)


if __name__ == '__main__':
    main()
