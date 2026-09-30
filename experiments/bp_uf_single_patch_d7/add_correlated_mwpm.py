"""Add correlated MWPM to a completed fixed-BP experiment on its saved shots."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import pymatching
import pymatching._cpp_pymatching
from scipy.stats import binomtest
import stim

_matching = None
_samples = {}
REFERENCE = 'correlated_mwpm'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_hash(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def write_json(path, value):
    # Write final retained files directly: scratch is a different filesystem.
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def physical_cpus(count):
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (int((root/'physical_package_id').read_text()), int((root/'core_id').read_text()))
        cores.setdefault(key, cpu)
    if len(cores) < count:
        raise RuntimeError('Insufficient distinct physical cores')
    return [cores[k] for k in sorted(cores)[:count]]


def start_worker(dem_path):
    global _matching
    _matching = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel.from_file(dem_path), enable_correlations=True)


def decode_chunk(path, first, last):
    if path not in _samples:
        _samples[path] = np.load(path, mmap_mode='r')
    bits = _matching.decode_batch(_samples[path][first:last], bit_packed_shots=True,
                                 bit_packed_predictions=True, enable_correlations=True)
    assert bits.shape == (last-first, 1) and np.all(bits < 4)
    return first, bits[:, 0].copy(), os.getpid()


def wilson(errors, shots):
    z = 1.959963984540054
    p = errors/shots
    denominator = 1+z*z/shots
    center = (p+z*z/(2*shots))/denominator
    width = z*math.sqrt(p*(1-p)/shots+z*z/(4*shots*shots))/denominator
    return [max(0., center-width), min(1., center+width)]


def analyze(predictions, actual, variants):
    masks = predictions ^ actual[:, None]
    failed = masks != 0
    n = len(actual)
    metrics = {}
    for k, name in enumerate(variants):
        errors = int(failed[:, k].sum())
        x, z = (int(np.count_nonzero(masks[:, k] & bit)) for bit in (1, 2))
        metrics[name] = dict(errors=errors, shots=n, failure_rate=errors/n, wilson_95=wilson(errors, n),
            x_observable_errors=x, x_observable_failure_rate=x/n, x_observable_wilson_95=wilson(x, n),
            z_observable_errors=z, z_observable_failure_rate=z/n, z_observable_wilson_95=wilson(z, n))
    paired = {}
    for baseline in ('weighted_uf', 'correlated_uf', 'bp5_uf', REFERENCE):
        ref = failed[:, variants.index(baseline)]
        paired[baseline] = {}
        for k, name in enumerate(variants):
            if name == baseline:
                continue
            repairs = int(np.count_nonzero(ref & ~failed[:, k]))
            regressions = int(np.count_nonzero(~ref & failed[:, k]))
            delta = (regressions-repairs)/n
            radius = 1.959963984540054*math.sqrt(max(0., ((repairs+regressions)/n-delta*delta)/(n-1)))
            paired[baseline][name] = dict(repairs=repairs, regressions=regressions,
                difference_target_minus_baseline=delta, paired_normal_95=[max(-1., delta-radius), min(1., delta+radius)],
                exact_mcnemar_p=float(binomtest(regressions, repairs+regressions).pvalue) if repairs+regressions else 1.,
                relative_failure_reduction=1-metrics[name]['errors']/int(ref.sum()) if ref.any() else None)
    return dict(shots=n, variants=variants, metrics=metrics, paired=paired,
        rate_unit='one patch over 28 rounds; either logical observable wrong',
        primary_comparison='bp5_uf versus correlated_uf',
        added_reference_comparisons='exploratory; added after the original five-decoder experiment; no multiplicity adjustment')


def verify_predictions(matching, packed, predictions, num_detectors, seed):
    rows = np.sort(np.random.default_rng(seed).choice(len(packed), min(64, len(packed)), replace=False))
    syndromes = np.unpackbits(packed[rows], axis=1, count=num_detectors, bitorder='little')
    # Different input/output packing and a fresh serial object verify the worker interface.
    unpacked = matching.decode_batch(syndromes, enable_correlations=True)
    np.testing.assert_array_equal(unpacked[:, 0]+2*unpacked[:, 1], predictions[rows])
    for row, syndrome in zip(rows, syndromes):
        bits = matching.decode(syndrome, enable_correlations=True)
        assert int(bits[0]+2*bits[1]) == int(predictions[row])
        physical = np.zeros(num_detectors, dtype=np.uint8)
        mask = 0
        for u, v in matching.decode_to_edges_array(syndrome, enable_correlations=True):
            physical[u] ^= 1
            if v >= 0:
                physical[v] ^= 1
                data = matching.get_edge_data(int(u), int(v))
            else:
                data = matching.get_boundary_edge_data(int(u))
            for observable in data['fault_ids']:
                mask ^= 1 << observable
        np.testing.assert_array_equal(physical, syndrome)
        assert mask == int(predictions[row])
    zero = matching.decode(np.zeros(num_detectors, dtype=np.uint8), enable_correlations=True)
    assert not np.any(zero)
    return dict(physical_corrections_verified=len(rows), serial_single_vs_parallel_batch_verified=len(rows),
        packed_vs_unpacked_batch_verified=len(rows), zero_syndrome_verified=True, row_ids=rows.tolist())


def benchmark(matching, packed, predictions, base, output, num_detectors):
    with np.load(base/'serial_timing.npz') as original:
        rows = original['rows']
    syndromes = np.unpackbits(packed[rows], axis=1, count=num_detectors, bitorder='little')
    for syndrome in syndromes[:16]:
        matching.decode(syndrome, enable_correlations=True)
    core_seconds = np.zeros(len(rows))
    api_seconds = np.zeros(len(rows))
    for i, (row, syndrome) in enumerate(zip(rows, syndromes)):
        events = matching._syndrome_array_to_detection_events(syndrome)
        for kind in (('core', 'api') if i % 2 == 0 else ('api', 'core')):
            start = time.perf_counter_ns()
            if kind == 'core':
                bits, _ = matching._matching_graph.decode(events, enable_correlations=True)
            else:
                bits = matching.decode(syndrome, enable_correlations=True)
            elapsed = (time.perf_counter_ns()-start)*1e-9
            assert int(bits[0]+2*bits[1]) == int(predictions[row])
            (core_seconds if kind == 'core' else api_seconds)[i] = elapsed
    metrics = {}
    for name, seconds in [('bound_native_decode', core_seconds), ('public_decode_api', api_seconds)]:
        ms = seconds*1000
        q = np.quantile(ms, [.5, .95, .99])
        metrics[name] = dict(mean_total_ms=float(ms.mean()), p50_ms=float(q[0]),
                             p95_ms=float(q[1]), p99_ms=float(q[2]))
    np.savez_compressed(output/'serial_timing.npz', rows=rows, core_seconds=core_seconds, public_api_seconds=api_seconds)
    result = dict(shots=len(rows), metrics=metrics, seed='same archived timing row IDs',
        scope='Bound native decode includes both correlated matching passes, allocation/output and Python binding overhead; '
              'dense-to-sparse syndrome conversion and model construction are excluded. Public decode additionally includes '
              'the Python wrapper and syndrome conversion. Original BP/UF measurements use internal C++ timers. '
              'Boundaries differ slightly; software measurements do not establish hardware speed or algorithmic complexity.')
    write_json(output/'serial_timing.json', result)
    return result


def write_report(base, output, summaries, timing, variants):
    original_timing = json.loads((base/'serial_timing.json').read_text())
    lines = ['# Fixed BP + UF and correlated MWPM on one d=7 patch', '',
        'SI1000 p=0.003, 28 CZ rounds, zero yokes, ideal time boundaries. '
        'Correlated MWPM was added on the exact saved detector/truth arrays from the completed experiment. '
        'The original five predictions and samples are unchanged. Every BP variant always ends with UF.', '',
        'PyMatching 2.4.0 has correlations enabled at both DEM import and decoding. '
        'Collection uses 32 worker processes restricted to the same 32-physical-core budget. '
        'Original UF/BP collection used 32 OpenMP threads. New reference comparisons are exploratory; '
        'the original primary contrast remains BP5 versus correlated UF.', '']
    for phase in ('pilot', 'confirmation'):
        summary = summaries[phase]
        lines += [f'## {phase.capitalize()}: {summary["shots"]:,} paired shots', '',
            '| Decoder | Failures | Failure rate (95% Wilson interval) | X-observable errors | Z-observable errors |',
            '|---|---:|---:|---:|---:|']
        for name in variants:
            row = summary['metrics'][name]
            lo, hi = row['wilson_95']
            lines.append(f'| {name} | {row["errors"]:,} | {100*row["failure_rate"]:.3f}% '
                f'[{100*lo:.3f}, {100*hi:.3f}] | {row["x_observable_errors"]:,} | {row["z_observable_errors"]:,} |')
        lines += ['', 'Paired differences against correlated MWPM; positive values mean more failures than MWPM.', '',
            '| Target | MWPM failures repaired | New failures | Difference, percentage points (95% paired CI) | McNemar p |',
            '|---|---:|---:|---:|---:|']
        for name in variants[:-1]:
            row = summary['paired'][REFERENCE][name]
            lo, hi = row['paired_normal_95']
            lines.append(f'| {name} | {row["repairs"]:,} | {row["regressions"]:,} | '
                f'{100*row["difference_target_minus_baseline"]:.3f} [{100*lo:.3f}, {100*hi:.3f}] | '
                f'{row["exact_mcnemar_p"]:.3g} |')
        primary = summary['paired']['correlated_uf']['bp5_uf']
        lines += ['', f'Original BP5-versus-correlated-UF contrast: {primary["repairs"]:,} repairs, '
            f'{primary["regressions"]:,} regressions, rate difference '
            f'{100*primary["difference_target_minus_baseline"]:.3f} percentage points.', '']
    lines += ['## Serial software timing', '',
        'Same 1,024 predetermined pilot rows. Original BP/UF values are retained. '
        'MWPM values use the bound C++ decoder; timings include both necessary correlation passes.', '',
        '| Decoder | Mean ms | p50 ms | p95 ms | p99 ms |', '|---|---:|---:|---:|---:|']
    for name in variants:
        row = (timing['metrics']['bound_native_decode'] if name == REFERENCE else original_timing['metrics'][name])
        lines.append(f'| {name} | {row["mean_total_ms"]:.4f} | {row["p50_ms"]:.4f} | '
                     f'{row["p95_ms"]:.4f} | {row["p99_ms"]:.4f} |')
    lines += ['', timing['scope'], '',
        f'MWPM public `decode` API mean: {timing["metrics"]["public_decode_api"]["mean_total_ms"]:.4f} ms.', '',
        '## Verification and provenance', '',
        'Input DEM, raw samples, original predictions and source hashes were checked. '
        'Serial and parallel decoding agree on predetermined checks; 64 physical corrections per phase '
        'reproduce the complete syndrome and logical predictions. All 1,024 timing-row outputs also '
        'match collection. No new Monte Carlo samples were generated.', '',
        f'The `{output.name}/` directory contains configuration, the extension source, verification, '
        'combined six-decoder per-shot arrays, summaries and timing data. Original five-decoder '
        'artifacts remain under the original phase directories. The previous report is retained as '
        '`report_five_decoders.md`.', '',
        'Rates refer to either logical observable being wrong over one 28-round single-patch memory shot. '
        'One distance/noise strength does not establish a threshold. BP comparisons include the change '
        'in weight projection. New MWPM comparisons are not adjusted for multiple testing.', '']
    report = '\n'.join(lines)
    (output/'report.md').write_text(report)
    previous = base/'report_five_decoders.md'
    if not previous.exists():
        shutil.copyfile(base/'report.md', previous)
    (base/'report.md').write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32, choices=[32])
    parser.add_argument('--output', type=Path, help='Optional fresh directory for a repeated extension')
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve() if args.output else base/'correlated_mwpm'
    output.mkdir(exist_ok=False)
    scratch_root = Path(os.environ['DANTE_SCRATCH'])/'runs'
    work = Path(tempfile.mkdtemp(prefix='single-patch-correlated-mwpm-', dir=scratch_root))
    cpus = physical_cpus(args.workers)
    os.sched_setaffinity(0, cpus)

    def progress(state, **details):
        write_json(output/'progress.json', dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
            workers=args.workers, cpu_affinity=cpus, scratch=str(work), **details))

    try:
        progress('verifying_inputs')
        configuration = json.loads((base/'configuration.json').read_text())
        assert pymatching.__version__ == configuration['versions']['pymatching']
        assert stim.__version__ == configuration['versions']['stim']
        assert sha256(base/'model.dem') == configuration['dem_sha256']
        assert sha256(base/'source_manifest.json') == configuration['source_manifest_sha256']
        manifest = json.loads((base/'source_manifest.json').read_text())
        for relative, expected in manifest.items():
            assert sha256(base/'source_snapshot'/relative) == expected
        variants = [*configuration['variants'], REFERENCE]
        assert variants == ['weighted_uf', 'correlated_uf', 'bp1_uf', 'bp2_uf', 'bp5_uf', REFERENCE]
        source = Path(__file__).resolve()
        shutil.copyfile(source, output/source.name)
        request = dict(base=str(base), workers=args.workers, cpu_affinity=cpus,
            original_configuration_sha256=sha256(base/'configuration.json'), dem_sha256=sha256(base/'model.dem'),
            source_sha256=sha256(source), variants=variants, parameter_tuning=False, new_samples=0,
            correlations_enabled_at_import=True, correlations_enabled_at_decode=True,
            python=platform.python_version(), numpy=np.__version__, pymatching=pymatching.__version__,
            stim=stim.__version__, native_matching_binary_sha256=sha256(pymatching._cpp_pymatching.__file__),
            command=[sys.executable, *sys.argv], scratch=str(work))
        write_json(output/'configuration.json', request)
        (output/'README.md').write_text('Correlated MWPM addition to the completed five-decoder experiment.\n\n'
            'Decode the exact saved 10,000 pilot and 100,000 confirmation shots with PyMatching 2.4.0; '
            'enable correlations at both graph import and decoding. Use 32 worker processes on 32 physical cores. '
            'Keep original decoder outputs and models. Validate input hashes, selected physical corrections and '
            'serial/batch/worker agreement. Benchmark on the same archived 1,024 timing rows. '
            'Save combined arrays and paired statistics, preserving the original five-decoder artifacts.\n\n'
            'Reproduce with the command in configuration.json and --output pointing to a fresh directory.\n')
        dem = stim.DetectorErrorModel.from_file(base/'model.dem')
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        summaries, validation, phase_inputs, worker_pids = {}, {}, {}, set()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=start_worker, initargs=(str(base/'model.dem'),)) as pool:
            for phase in ('pilot', 'confirmation'):
                sample = json.loads((base/phase/'sample.json').read_text())
                original_summary = json.loads((base/phase/'summary.json').read_text())
                assert sha256(sample['sample_path']) == sample['raw_npz_sha256']
                assert sha256(base/phase/'results.npz') == original_summary['results_sha256']
                with np.load(sample['sample_path']) as raw, np.load(base/phase/'results.npz') as original:
                    packed, actual_packed = raw['detectors'], raw['actual']
                    actual, original_predictions = original['actual'], original['predictions']
                    assert original['variants'].tolist() == variants[:-1]
                assert payload_hash(packed, actual_packed) == sample['packed_sample_sha256']
                np.testing.assert_array_equal(actual, actual_packed[:, 0] & 3)
                path = work/f'{phase}_detectors.npy'
                np.save(path, packed)
                count = len(packed)
                predictions = np.full(count, 255, dtype=np.uint8)
                covered = np.zeros(count, dtype=np.uint8)
                started = time.monotonic()
                chunk_size = 256 if phase == 'pilot' else 2048
                futures = [pool.submit(decode_chunk, str(path), first, min(count, first+chunk_size))
                           for first in range(0, count, chunk_size)]
                completed = 0
                for future in as_completed(futures):
                    first, predicted, pid = future.result()
                    last = first+len(predicted)
                    assert not covered[first:last].any()
                    predictions[first:last] = predicted
                    covered[first:last] = 1
                    worker_pids.add(pid)
                    completed += len(predicted)
                    progress('collecting', phase=phase, completed_shots=completed, target_shots=count)
                elapsed = time.monotonic()-started
                assert covered.all() and np.all(predictions < 4)
                validation[phase] = verify_predictions(matching, packed, predictions, dem.num_detectors,
                                                       2026092900+(phase == 'confirmation'))
                combined = np.column_stack([original_predictions, predictions])
                summary = analyze(combined, actual, variants)
                for name in variants[:-1]:
                    assert summary['metrics'][name] == original_summary['metrics'][name]
                for name, contrasts in original_summary['paired'].items():
                    for target, row in contrasts.items():
                        assert summary['paired'][name][target] == row
                np.savez_compressed(output/f'{phase}_results.npz', predictions=combined, actual=actual,
                                    variants=np.array(variants))
                summary.update(reference_collection_wall_seconds=elapsed,
                    original_sample_sha256=sample['packed_sample_sha256'],
                    original_results_sha256=original_summary['results_sha256'],
                    results_sha256=sha256(output/f'{phase}_results.npz'))
                write_json(output/f'{phase}_summary.json', summary)
                summaries[phase] = summary
                phase_inputs[phase] = (packed, predictions)
                print(f'{phase}: {count:,} paired shots; correlated MWPM failures '
                      f'{summary["metrics"][REFERENCE]["errors"]:,}', flush=True)
        progress('serial_benchmark')
        timing = benchmark(matching, *phase_inputs['pilot'], base, output, dem.num_detectors)
        write_json(output/'verification.json', dict(status='passed', phases=validation,
            worker_pids=sorted(worker_pids), original_five_decoder_counts_and_pairs_unchanged=True,
            verified_source_files=len(manifest), timing_predictions_verified=timing['shots']))
        write_report(base, output, summaries, timing, variants)
        artifact_hashes = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'progress.json'}
        write_json(output/'artifact_manifest.json', artifact_hashes)
        progress('complete', pilot_shots=10000, confirmation_shots=100000, report=str(base/'report.md'))
        print(f'Updated report: {base / "report.md"}', flush=True)
    except BaseException as error:
        progress('failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
