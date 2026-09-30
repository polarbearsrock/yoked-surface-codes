"""Resumable, paired collection of the five fixed BP/UF decoder variants."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from common import (CONFIG, FIELDS, VARIANTS, build, circuit_model, np,
                    physical_cpus, sha256, write_json)
from check import verify
import scipy
from scipy.stats import binomtest
import stim
import pymatching


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def payload_hash(*arrays):
    result = hashlib.sha256()
    for a in arrays:
        result.update(np.ascontiguousarray(a).tobytes())
    return result.hexdigest()


def save_checkpoint(path, **arrays):
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def wilson(errors, n):
    z = 1.959963984540054
    p = errors/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    width = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., center-width), min(1., center+width)]


def summarize(out, actual):
    predictions = out[:, :, 0].astype(np.uint8)
    error_mask = predictions ^ actual[:, None]
    failures = error_mask != 0
    n = len(actual)
    metrics = {}
    for k, name in enumerate(VARIANTS):
        errors = int(failures[:, k].sum())
        x_errors = int(np.count_nonzero(error_mask[:, k] & 1))
        z_errors = int(np.count_nonzero(error_mask[:, k] & 2))
        metrics[name] = dict(errors=errors, shots=n, failure_rate=errors/n,
            wilson_95=wilson(errors, n), x_observable_errors=x_errors,
            x_observable_failure_rate=x_errors/n, z_observable_errors=z_errors,
            z_observable_failure_rate=z_errors/n,
            x_observable_wilson_95=wilson(x_errors, n), z_observable_wilson_95=wilson(z_errors, n))
    paired = {}
    for baseline in ('weighted_uf', 'correlated_uf'):
        ref = failures[:, VARIANTS.index(baseline)]
        comparisons = {}
        for k, name in enumerate(VARIANTS):
            if name == baseline:
                continue
            repairs = int(np.count_nonzero(ref & ~failures[:, k]))
            regressions = int(np.count_nonzero(~ref & failures[:, k]))
            delta = (regressions-repairs)/n
            variance = max(0., ((regressions+repairs)/n-delta*delta)/(n-1))
            radius = 1.959963984540054*math.sqrt(variance)
            comparisons[name] = dict(repairs=repairs, regressions=regressions,
                difference_target_minus_baseline=delta,
                paired_normal_95=[max(-1., delta-radius), min(1., delta+radius)],
                exact_mcnemar_p=float(binomtest(regressions, repairs+regressions).pvalue)
                    if repairs+regressions else 1.,
                relative_failure_reduction=1-metrics[name]['errors']/int(ref.sum())
                    if ref.any() else None)
        paired[baseline] = comparisons
    return dict(shots=n, metrics=metrics, paired=paired,
                rate_unit='one patch over 28 rounds; either logical observable wrong',
                primary_comparison=CONFIG['primary_comparison'],
                other_comparisons='exploratory, no multiplicity adjustment')


def write_report(output):
    lines = ['# Fixed BP + weighted UF on one d=7 patch', '',
        'SI1000 p=0.003, 28 CZ rounds, zero yokes, ideal time boundaries. '
        'All five variants decode identical shots within each phase. '
        'BP always runs its fixed budget and finishes with weighted UF.', '',
        'The five variants are weighted UF, correlated UF, BP1 + UF, BP2 + UF, and BP5 + UF. '
        'The two samples have independent seeds; no parameters change between them.', '']
    for phase in ('pilot', 'confirmation'):
        path = output/phase/'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        lines += [f'## {phase.capitalize()}: {summary["shots"]:,} paired shots', '',
            '| Decoder | Failures | Either-observable failure rate (95% Wilson CI) | X-observable errors | Z-observable errors |',
            '|---|---:|---:|---:|---:|']
        for name in VARIANTS:
            row = summary['metrics'][name]
            lo, hi = row['wilson_95']
            lines.append(f'| {name} | {row["errors"]:,} | {100*row["failure_rate"]:.3f}% '
                f'[{100*lo:.3f}, {100*hi:.3f}] | {row["x_observable_errors"]:,} | {row["z_observable_errors"]:,} |')
        lines += ['', 'Paired comparisons against correlated UF; negative differences favor the target.', '',
            '| Target | Repairs | Regressions | Difference, percentage points (95% paired CI) | Exact McNemar p |',
            '|---|---:|---:|---:|---:|']
        for name in ('bp1_uf', 'bp2_uf', 'bp5_uf'):
            row = summary['paired']['correlated_uf'][name]
            lo, hi = row['paired_normal_95']
            lines.append(f'| {name} | {row["repairs"]:,} | {row["regressions"]:,} | '
                f'{100*row["difference_target_minus_baseline"]:.3f} [{100*lo:.3f}, {100*hi:.3f}] | '
                f'{row["exact_mcnemar_p"]:.3g} |')
        lines += ['', f'Collection kernel wall time: {summary["collection_wall_seconds"]:.2f} seconds. '
            'This includes shared work across variants and is not a per-decoder latency measurement.', '']
    timing_path = output/'serial_timing.json'
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        lines += ['## Serial native kernel timing', '',
            f'{timing["shots"]:,} predetermined pilot rows; empirical quantiles. '
            'These are software kernel measurements with shared state and fixed variant order.', '',
            '| Decoder | Mean BP/reweighting ms | Mean UF ms | Mean total ms | p50 ms | p95 ms | p99 ms |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in timing['metrics'].items():
            lines.append(f'| {name} | {row["mean_evidence_ms"]:.3f} | {row["mean_uf_ms"]:.3f} | '
                f'{row["mean_total_ms"]:.3f} | {row["p50_ms"]:.3f} | {row["p95_ms"]:.3f} | {row["p99_ms"]:.3f} |')
        lines += ['', timing['scope'], '']
    lines += ['## Interpretation limits', '',
        'Rates are per single-patch 28-round memory shot, not per round or six-patch block. '
        'BP5 versus correlated UF is the primary contrast; other comparisons are exploratory. '
        'BP also changes the probability-to-weight projection, so this measures the combined design. '
        'There is no prior-projection control in the user-approved five-decoder scope. '
        'A single distance and noise strength cannot establish a threshold.', '',
        'See `configuration.json`, `verification.json`, phase `sample.json` files, '
        '`source_manifest.json`, and `source_snapshot/` for provenance and reproduction.', '']
    (output/'report.md').write_text('\n'.join(lines))


def collect_phase(name, shots, seed, circuit, native, args, identity, update):
    scratch = args.work_dir/name
    retained = args.output/name
    scratch.mkdir(exist_ok=True)
    retained.mkdir(exist_ok=True)
    sample_path = scratch/'sample.npz'
    if sample_path.exists():
        with np.load(sample_path) as data:
            packed, actual_packed = data['detectors'], data['actual']
    else:
        packed, actual_packed = circuit.compile_detector_sampler(seed=seed).sample(
            shots=shots, separate_observables=True, bit_packed=True)
        save_checkpoint(sample_path, detectors=packed, actual=actual_packed)
    assert packed.shape == (shots, (circuit.num_detectors+7)//8)
    assert actual_packed.shape == (shots, 1)
    actual = actual_packed[:, 0] & 3
    sample = dict(shots=shots, seed=seed, detectors=circuit.num_detectors,
        observables=circuit.num_observables, packed_sample_sha256=payload_hash(packed, actual_packed),
        sample_path=str(sample_path), raw_npz_sha256=sha256(sample_path))
    phase_identity = digest(dict(run=identity, sample=sample))
    sample['identity'] = phase_identity
    existing = retained/'sample.json'
    if existing.exists() and json.loads(existing.read_text()) != sample:
        raise ValueError('Saved sample identity differs from the resumed sample')
    write_json(existing, sample)
    pieces = []
    total_wall = 0.
    errors = np.zeros(len(VARIANTS), dtype=np.int64)
    checkpoint_records = []
    for start in range(0, shots, args.chunk_size):
        stop = min(shots, start+args.chunk_size)
        path = scratch/f'{start:08d}-{stop:08d}.npz'
        if path.exists():
            with np.load(path) as saved:
                assert str(saved['identity']) == phase_identity
                assert int(saved['start']) == start and int(saved['stop']) == stop
                out, elapsed = saved['out'], float(saved['wall_seconds'])
        else:
            began = time.monotonic()
            out, _ = native.decode(packed[start:stop], args.threads)
            elapsed = time.monotonic()-began
            save_checkpoint(path, identity=phase_identity, start=start, stop=stop,
                            out=out, wall_seconds=elapsed)
        assert out.shape == (stop-start, len(VARIANTS), len(FIELDS))
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out[:, :, 11], np.broadcast_to([0, 0, 1, 2, 5], out.shape[:2]))
        predictions = out[:, :, 0].astype(np.uint8)
        assert np.array_equal(out[:, :, 0], predictions) and np.all(predictions < 4)
        errors += np.count_nonzero(predictions != actual[start:stop, None], axis=0)
        total_wall += elapsed
        pieces.append(out)
        checkpoint_records.append(dict(start=start, stop=stop, sha256=sha256(path)))
        update('collecting', phase=name, completed_shots=stop, target_shots=shots,
               phase_kernel_wall_seconds=total_wall,
               cumulative_failures=dict(zip(VARIANTS, map(int, errors))),
               estimated_remaining_kernel_seconds=total_wall/stop*(shots-stop))
        print(f'{name}: {stop:,}/{shots:,} paired shots, failures {errors.tolist()}', flush=True)
    out = np.concatenate(pieces)
    result_path = retained/'results.npz'
    np.savez_compressed(result_path, predictions=out[:, :, 0].astype(np.uint8),
                        actual=actual, fields=out, variants=np.array(VARIANTS), field_names=np.array(FIELDS))
    summary = summarize(out, actual)
    summary.update(seed=seed, identity=phase_identity, collection_wall_seconds=total_wall,
                   physical_syndrome_checks=shots*len(VARIANTS), results_sha256=sha256(result_path))
    write_json(retained/'summary.json', summary)
    write_json(retained/'checkpoints.json', checkpoint_records)
    write_report(args.output)
    return packed, out


def benchmark(native, packed, pilot_output, args, update):
    count = min(CONFIG['serial_timing_shots'], len(packed))
    rows = np.sort(np.random.default_rng(CONFIG['serial_timing_selection_seed']).choice(len(packed), count, replace=False))
    pieces = []
    for start in range(0, count, 64):
        subset = rows[start:start+64]
        out, _ = native.decode(packed[subset], 1)
        columns = [*range(9), 11]
        np.testing.assert_array_equal(out[:, :, columns], pilot_output[subset][:, :, columns])
        pieces.append(out)
        update('serial_benchmark', completed_shots=min(count, start+64), target_shots=count)
    out = np.concatenate(pieces)
    np.savez_compressed(args.output/'serial_timing.npz', rows=rows, fields=out)
    metrics = {}
    for k, name in enumerate(VARIANTS):
        evidence_ms = out[:, k, 9]*1000
        uf_ms = out[:, k, 10]*1000
        total = evidence_ms+uf_ms
        q = np.quantile(total, [.5, .95, .99])
        metrics[name] = dict(mean_evidence_ms=float(evidence_ms.mean()), mean_uf_ms=float(uf_ms.mean()),
            mean_total_ms=float(total.mean()), p50_ms=float(q[0]), p95_ms=float(q[1]), p99_ms=float(q[2]))
    write_json(args.output/'serial_timing.json', dict(shots=count, metrics=metrics,
        scope='Serial native BP initialization/inference/projection plus UF growth/tree/peeling; '
              'correlated UF includes both necessary passes. Excludes compilation, Python, unpacking, '
              'physical auditing, and some buffer destruction. Parallel shot times are not latency. '
              'Fixed order and shared state affect cache conditions; quantiles have sampling uncertainty.'))
    write_report(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=32, choices=[32])
    parser.add_argument('--chunk-size', type=int, default=512)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('chunk size must be positive')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.work_dir/'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = physical_cpus(args.threads)
    os.sched_setaffinity(0, cpus)
    started = utc()

    def update(state, **details):
        write_json(args.output/'progress.json', dict(state=state, pid=os.getpid(), started_utc=started,
            updated_utc=utc(), threads=args.threads, cpu_affinity=cpus, **details))

    native = None
    try:
        update('building')
        manifest_path = args.output/'source_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        snapshot = args.output/'source_snapshot'
        for relative, expected in manifest.items():
            if sha256(snapshot/relative) != expected:
                raise ValueError(f'Frozen source changed: {relative}')
        library, command = build(args.work_dir/'build')
        circuit, dem, graph, model, native = circuit_model(library)
        circuit_path, dem_path = args.output/'circuit.stim', args.output/'model.dem'
        circuit_path.write_text(str(circuit))
        dem_path.write_text(str(dem))
        identity = dict(config=CONFIG, variants=VARIANTS, fields=FIELDS,
            source_manifest_sha256=sha256(manifest_path), circuit_sha256=sha256(circuit_path),
            dem_sha256=sha256(dem_path), native_library_sha256=sha256(library),
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                          stim=stim.__version__, pymatching=pymatching.__version__))
        config_path = args.output/'configuration.json'
        if config_path.exists() and json.loads(config_path.read_text())['identity'] != digest(identity):
            raise ValueError('Resume identity differs from the frozen experiment')
        write_json(config_path, dict(**identity, identity=digest(identity), model=model.describe(),
            compile_command=command, threads=args.threads, cpu_affinity=cpus,
            work_dir=str(args.work_dir), command=[sys.executable, *sys.argv],
            environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
                'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES', 'OPENBLAS_NUM_THREADS', 'TMPDIR')}))
        update('verifying')
        validation = verify(library, circuit, dem, native, args.threads)
        write_json(args.output/'verification.json', validation)
        pilot_packed, pilot_out = collect_phase('pilot', CONFIG['development_shots'], CONFIG['development_seed'],
            circuit, native, args, digest(identity), update)
        collect_phase('confirmation', CONFIG['confirmation_shots'], CONFIG['confirmation_seed'],
            circuit, native, args, digest(identity), update)
        benchmark(native, pilot_packed, pilot_out, args, update)
        update('complete', pilot_shots=CONFIG['development_shots'],
               confirmation_shots=CONFIG['confirmation_shots'], report=str(args.output/'report.md'))
        print(f'Completed experiment: {args.output / "report.md"}', flush=True)
    except BaseException as error:
        update('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if native is not None:
            native.close()


if __name__ == '__main__':
    main()
