"""Add fixed BP10 + weighted UF to the completed experiment's saved shots."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(OMP_NUM_THREADS='32', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE')

import numpy as np
from scipy.stats import binomtest
import stim

TARGET = 'bp10_uf'
DETERMINISTIC = [*range(9), 11]


def statistics(predictions, actual, variants):
    masks = predictions ^ actual[:, None]
    failed = masks != 0
    n = len(actual)
    metrics = {}
    for k, name in enumerate(variants):
        errors = int(failed[:, k].sum())
        x, z = (int(np.count_nonzero(masks[:, k] & bit)) for bit in (1, 2))
        metrics[name] = dict(errors=errors, shots=n, failure_rate=errors/n,
            wilson_95=wilson(errors, n), x_observable_errors=x, x_observable_failure_rate=x/n,
            x_observable_wilson_95=wilson(x, n), z_observable_errors=z,
            z_observable_failure_rate=z/n, z_observable_wilson_95=wilson(z, n))
    paired = {}
    for baseline in ('weighted_uf', 'correlated_uf', 'bp5_uf', 'correlated_mwpm'):
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
                difference_target_minus_baseline=delta,
                paired_normal_95=[max(-1., delta-radius), min(1., delta+radius)],
                exact_mcnemar_p=float(binomtest(regressions, repairs+regressions).pvalue)
                    if repairs+regressions else 1.,
                relative_failure_reduction=1-metrics[name]['errors']/int(ref.sum()) if ref.any() else None)
    return dict(shots=n, variants=variants, metrics=metrics, paired=paired,
        rate_unit='one patch over 28 rounds; either logical observable wrong',
        primary_comparison='bp5_uf versus correlated_uf',
        added_comparisons='BP10 was requested after viewing the existing results; exploratory, no multiplicity adjustment')


def wilson(errors, n):
    z = 1.959963984540054
    p = errors/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    width = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., center-width), min(1., center+width)]


def native_class(evidence):
    class FixedBP(evidence.Native):
        def __init__(self, library, graph, rules, model):
            super().__init__(library, graph, rules, model)
            self.lib.fixed_bp_decode.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self.lib.fixed_bp_decode.restype = ctypes.c_int

        def decode(self, packed, threads=1, audit=False, budget=10):
            packed = np.ascontiguousarray(packed, dtype=np.uint8)
            if not 1 <= threads <= 32 or budget not in (5, 10) or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
                raise ValueError('Invalid thread count, BP budget, or syndrome shape')
            out = np.zeros((len(packed), 15), dtype=np.float64)
            flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint32) if audit else None
            team = ctypes.c_int()
            errors = self.lib.fixed_bp_decode(self.handle, len(packed), packed.shape[1],
                evidence.ptr(packed), evidence.ptr(out), threads, budget,
                evidence.ptr(flags) if audit else None, ctypes.byref(team))
            if errors or team.value != threads or not np.isfinite(out).all():
                raise RuntimeError(f'Native errors={errors}; requested/actual threads={threads}/{team.value}')
            np.testing.assert_array_equal(out[:, 11], budget)
            np.testing.assert_allclose(out[:, 9], out[:, 12:15].sum(axis=1), atol=0, rtol=0)
            assert np.all(out[:, 0] == out[:, 0].astype(np.uint8)) and np.all(out[:, 0] < 4)
            return out, flags
    return FixedBP


def validate(common, library, native, circuit, threads, inputs):
    from yoked.decoders import DecodingGraph, UnionFindDecoder
    packed, _ = circuit.compile_detector_sampler(seed=common.CONFIG['validation_seed']).sample(
        shots=32, separate_observables=True, bit_packed=True)
    packed[0] = 0
    parallel, flags = native.decode(packed, threads, audit=True)
    serial, serial_flags = native.decode(packed, 1, audit=True)
    np.testing.assert_array_equal(parallel[:, DETERMINISTIC], serial[:, DETERMINISTIC])
    np.testing.assert_array_equal(flags, serial_flags)
    syndromes = np.unpackbits(packed, axis=1, count=native.graph.num_detectors, bitorder='little')
    differences = []
    for row in range(4):
        syndrome = syndromes[row]
        posterior, weights = native.weights(syndrome, 10)
        expected_q, expected_weights = common.evidence.python_bp(native.model, syndrome, 10)
        np.testing.assert_allclose(posterior, expected_q, atol=2e-10, rtol=2e-8)
        np.testing.assert_allclose(weights, expected_weights, atol=2e-9, rtol=2e-8)
        differences.append(float(np.max(np.abs(posterior-expected_q))))
        graph = DecodingGraph(native.graph.num_detectors, native.graph.num_observables,
            [(u, v, float(weights[e]), mask) for e, (u, v, _, mask) in enumerate(native.graph.edges)])
        expected = UnionFindDecoder(graph)._decode(syndrome)
        np.testing.assert_array_equal(np.flatnonzero(flags[row]), sorted(expected.selected_edges))
        assert int(parallel[row, 0]) == expected.observable_mask
    # Explicitly reconstruct every audited physical correction and logical mask.
    for syndrome, output, selected in zip(syndromes, parallel, flags):
        residual, mask = syndrome.copy(), 0
        for edge in np.flatnonzero(selected):
            u, v, _, label = native.graph.edges[edge]
            residual[u] ^= 1
            if v is not None:
                residual[v] ^= 1
            mask ^= label
        assert not residual.any() and mask == int(output[0])
    checks = {}
    for phase, data in inputs.items():
        rows = np.unique(np.linspace(0, len(data['packed'])-1, 32, dtype=int))
        fresh, _ = native.decode(data['packed'][rows], threads, budget=5)
        np.testing.assert_array_equal(fresh[:, DETERMINISTIC], data['original_fields'][rows, 4][:, DETERMINISTIC])
        checks[phase] = rows.tolist()
    return dict(status='passed', independent_python_bp_and_uf_rows=4,
        maximum_posterior_difference=max(differences), physical_corrections_reconstructed=32,
        zero_syndrome_verified=True, serial_vs_parallel_shots=32, verified_openmp_threads=[1, threads],
        bp5_regression_row_ids=checks, physical_syndrome_check_every_collection_shot=True,
        actuals_passed_to_decoder=False)


def benchmark(native, pilot, base, output, progress, write_json):
    with np.load(base/'serial_timing.npz') as saved:
        rows = saved['rows']
    native.decode(pilot['packed'][rows[:16]], 1)
    parts = []
    for start in range(0, len(rows), 64):
        selected = rows[start:start+64]
        out, _ = native.decode(pilot['packed'][selected], 1)
        np.testing.assert_array_equal(out[:, DETERMINISTIC], pilot['new_fields'][selected][:, DETERMINISTIC])
        parts.append(out)
        progress('serial_benchmark', completed_shots=min(start+64, len(rows)), target_shots=len(rows))
    out = np.concatenate(parts)
    np.savez_compressed(output/'serial_timing.npz', rows=rows, fields=out)
    total = (out[:, 9]+out[:, 10])*1000
    q = np.quantile(total, [.5, .95, .99])
    metrics = dict(mean_evidence_ms=float(out[:, 9].mean()*1000), mean_uf_ms=float(out[:, 10].mean()*1000),
        mean_total_ms=float(total.mean()), p50_ms=float(q[0]), p95_ms=float(q[1]), p99_ms=float(q[2]),
        mean_bp_initialization_ms=float(out[:, 12].mean()*1000),
        mean_bp_rounds_ms=float(out[:, 13].mean()*1000), mean_projection_ms=float(out[:, 14].mean()*1000))
    result = dict(shots=len(rows), metrics={TARGET: metrics},
        scope='Serial internal C++ timers: BP initialization, exactly ten BP iterations, edge-weight projection, '
        'and UF growth/tree/peeling. Excludes model construction, Python, unpacking, physical audit and some '
        'destruction. Standalone BP10 follows a warmup; original timings share work in fixed variant order. '
        'CPU state and cache differences limit precise ratios between runs.')
    write_json(output/'serial_timing.json', result)
    return result


def report(base, output, summaries, timing):
    old = json.loads((base/'serial_timing.json').read_text())['metrics']
    mwpm = json.loads((base/'correlated_mwpm/serial_timing.json').read_text())
    variants = summaries['pilot']['variants']
    lines = ['# Fixed BP + UF and correlated MWPM on one d=7 patch', '',
        'SI1000 p=0.003, 28 CZ rounds, zero yokes, ideal time boundaries. '
        'BP10 was added on the exact saved shots after the original results were viewed. '
        'Every BP variant runs its fixed budget and always ends with weighted UF. '
        'Damping, clipping, graph, and posterior-to-weight projection are unchanged.', '',
        'BP10 collection uses 32 OpenMP threads on 32 distinct physical cores. '
        'The original six decoder predictions are retained unchanged. '
        'BP10 comparisons are exploratory; the original primary contrast remains BP5 versus correlated UF.', '']
    for phase, summary in summaries.items():
        lines += [f'## {phase.capitalize()}: {summary["shots"]:,} paired shots', '',
            '| Decoder | Failures | Failure rate (95% Wilson interval) | X errors | Z errors |',
            '|---|---:|---:|---:|---:|']
        for name in variants:
            row = summary['metrics'][name]
            lo, hi = row['wilson_95']
            lines.append(f'| {name} | {row["errors"]:,} | {100*row["failure_rate"]:.3f}% '
                f'[{100*lo:.3f}, {100*hi:.3f}] | {row["x_observable_errors"]:,} | {row["z_observable_errors"]:,} |')
        lines += ['', 'BP10 paired comparisons; negative differences favor BP10.', '',
            '| Baseline | Failures repaired | New failures | Difference, percentage points (95% paired CI) | McNemar p |',
            '|---|---:|---:|---:|---:|']
        for name in ('weighted_uf', 'correlated_uf', 'bp5_uf', 'correlated_mwpm'):
            row = summary['paired'][name][TARGET]
            lo, hi = row['paired_normal_95']
            lines.append(f'| {name} | {row["repairs"]:,} | {row["regressions"]:,} | '
                f'{100*row["difference_target_minus_baseline"]:.3f} [{100*lo:.3f}, {100*hi:.3f}] | '
                f'{row["exact_mcnemar_p"]:.3g} |')
        lines += ['']
    lines += ['## Serial software timing', '',
        'Same 1,024 predetermined pilot rows. Original timing values are retained.', '',
        '| Decoder | BP/weight prep ms | UF ms | Total ms | p50 ms | p95 ms | p99 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in variants:
        row = (timing['metrics'][TARGET] if name == TARGET else
               mwpm['metrics']['bound_native_decode'] if name == 'correlated_mwpm' else old[name])
        # Correlated UF's field 9 includes its first UF pass, not BP.
        evidence = f'{row["mean_evidence_ms"]:.4f}' if name.startswith('bp') else '—'
        uf = f'{row["mean_uf_ms"]:.4f}' if name != 'correlated_mwpm' and name != 'correlated_uf' else '—'
        lines.append(f'| {name} | {evidence} | {uf} | {row["mean_total_ms"]:.4f} | '
            f'{row["p50_ms"]:.4f} | {row["p95_ms"]:.4f} | {row["p99_ms"]:.4f} |')
    t = timing['metrics'][TARGET]
    lines += ['', f'BP10 breakdown: initialization {t["mean_bp_initialization_ms"]:.4f} ms, '
        f'ten BP iterations {t["mean_bp_rounds_ms"]:.4f} ms, '
        f'weight projection {t["mean_projection_ms"]:.4f} ms, UF {t["mean_uf_ms"]:.4f} ms.', '',
        timing['scope'], '',
        'Correlated UF totals include both necessary UF passes and correlation reweighting. '
        'Correlated MWPM uses bound C++ decode with correlations enabled at import and decode; '
        'its timer includes Python binding overhead, while BP/UF use internal C++ timers. '
        'BP iteration stages were not separately recorded for the original BP1/2/5 runs.', '',
        '## Verification and provenance', '',
        'Verified saved source, DEM, circuit, raw sample and prediction hashes. '
        'Independent Python BP and production UF match the native BP10 implementation. '
        'Serial and 32-thread results agree; all collection corrections pass physical syndrome checks. '
        'The new kernel at five rounds reproduces archived BP5 outputs on predetermined checks. '
        'All 1,024 serial timing outputs match collection.', '',
        f'`{output.name}/` contains the extension sources, configuration, validation, combined seven-decoder '
        'predictions, paired summaries, and serial timing data. Original artifacts are preserved. '
        'The previous report is retained as `report_six_decoders.md`.', '',
        'Rates refer to either logical observable being wrong over one 28-round memory shot, not per round. '
        'The sample named confirmation was independent of the pilot when originally generated; '
        'it is reused for this exploratory extension, not a fresh holdout for BP10. '
        'No new Monte Carlo samples were generated. New comparisons are unadjusted for multiple testing. '
        'BP comparisons include the posterior-to-weight projection change relative to baseline UF.', '']
    text = '\n'.join(lines)
    (output/'report.md').write_text(text)
    previous = base/'report_six_decoders.md'
    if not previous.exists():
        shutil.copyfile(base/'report.md', previous)
    (base/'report.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--threads', type=int, choices=[32], default=32)
    parser.add_argument('--output', type=Path, help='Fresh output directory for a repeated extension')
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve() if args.output else base/'bp10_uf'
    output.mkdir(exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='single-patch-bp10-', dir=Path(os.environ['DANTE_SCRATCH'])/'runs'))
    snapshot = base/'source_snapshot'
    sys.path.insert(0, str(snapshot/'experiments/bp_uf_single_patch_d7'))
    import common
    from common import sha256, write_json
    cpus = common.physical_cpus(args.threads)
    os.sched_setaffinity(0, cpus)
    native = None

    def progress(state, **details):
        write_json(output/'progress.json', dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
            threads=args.threads, cpu_affinity=cpus, scratch=str(work), **details))

    try:
        progress('verifying_inputs')
        config = json.loads((base/'configuration.json').read_text())
        assert stim.__version__ == config['versions']['stim']
        assert np.__version__ == config['versions']['numpy']
        for name in ('circuit', 'dem'):
            filename = 'circuit.stim' if name == 'circuit' else 'model.dem'
            assert sha256(base/filename) == config[name+'_sha256']
        assert sha256(base/'source_manifest.json') == config['source_manifest_sha256']
        manifest = json.loads((base/'source_manifest.json').read_text())
        for relative, expected in manifest.items():
            assert sha256(snapshot/relative) == expected
        reference = base/'correlated_mwpm'
        for relative, expected in json.loads((reference/'artifact_manifest.json').read_text()).items():
            assert sha256(reference/relative) == expected
        variants = [*config['variants'], 'correlated_mwpm', TARGET]
        fields = [*common.FIELDS, 'bp_initialization_seconds', 'bp_rounds_seconds', 'projection_seconds']
        source = Path(__file__).resolve()
        for path in (source, source.with_name('kernel_bp10.cc')):
            shutil.copyfile(path, output/path.name)
        inputs = {}
        for phase in ('pilot', 'confirmation'):
            sample = json.loads((base/phase/'sample.json').read_text())
            summary = json.loads((base/phase/'summary.json').read_text())
            assert sha256(sample['sample_path']) == sample['raw_npz_sha256']
            assert sha256(base/phase/'results.npz') == summary['results_sha256']
            with np.load(sample['sample_path']) as raw, np.load(base/phase/'results.npz') as old, \
                    np.load(reference/f'{phase}_results.npz') as combined:
                np.testing.assert_array_equal(raw['actual'][:, 0] & 3, combined['actual'])
                np.testing.assert_array_equal(old['actual'], combined['actual'])
                np.testing.assert_array_equal(old['predictions'], combined['predictions'][:, :5])
                assert combined['variants'].tolist() == variants[:-1]
                inputs[phase] = dict(packed=raw['detectors'], actual=combined['actual'],
                    predictions=combined['predictions'], original_fields=old['fields'], sample=sample)
        progress('building')
        library = work/'bp10.so'
        command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp', '-ffp-contract=off',
                   '-I', str(snapshot), str(output/'kernel_bp10.cc'), '-o', str(library)]
        subprocess.run(command, check=True)
        dem = stim.DetectorErrorModel.from_file(base/'model.dem')
        graph = common.evidence.DecodingGraph.from_dem(dem)
        model = common.evidence.FaultModel.from_dem(dem, graph)
        native = native_class(common.evidence)(library, graph, common.evidence.correlation_rules_from_dem(graph, dem), model)
        assert model.describe() == config['model']
        write_json(output/'configuration.json', dict(base=str(base), threads=args.threads, cpu_affinity=cpus,
            variants=variants, fields=fields, bp_iterations=10, always_finish_with_uf=True,
            original_config=config['config'], original_configuration_sha256=sha256(base/'configuration.json'),
            original_source_manifest_sha256=sha256(base/'source_manifest.json'),
            dem_sha256=sha256(base/'model.dem'), circuit_sha256=sha256(base/'circuit.stim'),
            mwpm_artifact_manifest_sha256=sha256(reference/'artifact_manifest.json'),
            extension_sources={p.name: sha256(p) for p in (output/source.name, output/'kernel_bp10.cc')},
            native_library_sha256=sha256(library), compile_command=command, command=[sys.executable, *sys.argv],
            python=platform.python_version(), numpy=np.__version__, stim=stim.__version__, scratch=str(work),
            new_samples=0, interpretation='exploratory extension requested after viewing prior results'))
        progress('validating')
        circuit = stim.Circuit.from_file(base/'circuit.stim')
        verification = validate(common, library, native, circuit, args.threads, inputs)
        write_json(output/'verification.json', verification)
        print('BP10 verification passed; native OpenMP team verified at 32 threads.', flush=True)
        summaries = {}
        for phase, data in inputs.items():
            parts, wall, errors = [], 0., 0
            for start in range(0, len(data['packed']), 1024):
                stop = min(start+1024, len(data['packed']))
                began = time.monotonic()
                out, _ = native.decode(data['packed'][start:stop], args.threads)
                wall += time.monotonic()-began
                errors += int(np.count_nonzero(out[:, 0] != data['actual'][start:stop]))
                parts.append(out)
                np.savez_compressed(work/f'{phase}_{start:06d}_{stop:06d}.npz', fields=out, start=start, stop=stop)
                progress('collecting', phase=phase, completed_shots=stop, target_shots=len(data['packed']),
                    cumulative_failures=errors, collection_wall_seconds=wall)
                if stop == len(data['packed']) or stop % 10240 == 0:
                    print(f'{phase}: {stop:,}/{len(data["packed"]):,} shots; {errors:,} BP10 failures', flush=True)
            data['new_fields'] = np.concatenate(parts)
            combined = np.column_stack([data['predictions'], data['new_fields'][:, 0].astype(np.uint8)])
            summary = statistics(combined, data['actual'], variants)
            original = json.loads((reference/f'{phase}_summary.json').read_text())
            for name in variants[:-1]:
                assert summary['metrics'][name] == original['metrics'][name]
            for name, contrasts in original['paired'].items():
                for target, row in contrasts.items():
                    assert summary['paired'][name][target] == row
            np.savez_compressed(output/f'{phase}_results.npz', predictions=combined, actual=data['actual'],
                variants=np.array(variants), bp10_fields=data['new_fields'], field_names=np.array(fields))
            summary.update(collection_wall_seconds=wall, physical_syndrome_checks=len(combined),
                sample=data['sample'], results_sha256=sha256(output/f'{phase}_results.npz'))
            write_json(output/f'{phase}_summary.json', summary)
            summaries[phase] = summary
        progress('serial_benchmark')
        timing = benchmark(native, inputs['pilot'], base, output, progress, write_json)
        verification.update(original_six_decoder_counts_and_pairs_unchanged=True,
            timing_predictions_verified=timing['shots'], collection_physical_checks=110000,
            fixed_iterations_all_shots=10)
        write_json(output/'verification.json', verification)
        report(base, output, summaries, timing)
        write_json(output/'artifact_manifest.json', {p.name: sha256(p) for p in output.iterdir()
            if p.is_file() and p.name != 'progress.json'})
        progress('complete', pilot_shots=10000, confirmation_shots=100000, report=str(base/'report.md'))
        print(json.dumps(dict(bp10=summaries['confirmation']['metrics'][TARGET],
            versus_bp5=summaries['confirmation']['paired']['bp5_uf'][TARGET],
            timing=timing['metrics'][TARGET]), indent=2), flush=True)
    except BaseException as error:
        progress('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if native is not None:
            native.close()


if __name__ == '__main__':
    main()
