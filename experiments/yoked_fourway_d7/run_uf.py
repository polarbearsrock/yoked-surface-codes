"""Validate and collect two UF variants on the saved four-way experiment sample."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--threads', type=int, choices=[1, 32], default=32)
    parser.add_argument('--batch-size', type=int, default=1250)
    args = parser.parse_args()
    output = args.output.resolve()
    launch = json.loads((output/'launch.json').read_text())
    work = Path(launch['work'])
    repository = Path(launch['snapshot_repository'])
    os.environ['DANTE_REPO'] = str(repository)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), OMP_THREAD_LIMIT='32',
        OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', YOKED_MPP_BUILD=launch['mpp_build'])
    os.sched_setaffinity(0, launch['cpu_affinity'])
    sys.path.insert(0, str(repository/'src'))
    import numpy as np
    import stim
    from native import Native, FIELDS, VARIANTS, build, make_native, sha, verify, write_json
    from yoked.hierarchical._collect import SampleSet
    from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
    from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
    from yoked.hierarchical._patch_graphs import PatchGraphs

    expected_fork = launch['stim_fork']
    if stim.__version__ != expected_fork['version'] or sha(expected_fork['native_module']) != expected_fork['native_sha256']:
        raise RuntimeError('Stim package changed after sample generation')
    snapshot_recipe = output/'source_snapshot/experiments/yoked_fourway_d7'
    manifest = json.loads((output/'source_manifest.json').read_text())
    new_sources = {}
    for name in ('kernel.cc', 'native.py', 'run_uf.py', 'README.md'):
        source = Path(__file__).resolve().parent/name
        target = snapshot_recipe/name
        if source != target:
            if (work/'uf/configuration.json').exists() and target.exists() and sha(source) != sha(target):
                raise RuntimeError('Refusing to change sources of a collected run')
            shutil.copyfile(source, target)
        relative = str(target.relative_to(output/'source_snapshot'))
        manifest[relative] = sha(target)
        new_sources[relative] = sha(target)
    write_json(output/'source_manifest.json', manifest)
    library, command = build(snapshot_recipe/'kernel.cc', work/'uf-build')
    sample = SampleSet.load(work/'mwpm/d7/sample')
    assert sample.shots == launch['shots'] and sample.seed == launch['sample_seed']
    dem = stim.DetectorErrorModel(sample.dem_text)
    patches = PatchGraphs.from_yoked_dem(dem, num_patches=6)
    validation_detectors, _ = stim.Circuit(sample.circuit_text).compile_detector_sampler(
        seed=202609290708).sample(shots=16, separate_observables=True)
    started = time.perf_counter()
    validation = verify(library, patches, validation_detectors, args.threads)
    validation.update(elapsed_seconds=time.perf_counter()-started,
        source_sha256=new_sources, library_sha256=sha(library), command=command,
        validation_seed=202609290708)
    write_json(output/f'verification_{args.threads}threads.json', validation)
    print(f'UF/BP corrections, final growth costs, gaps, and {args.threads}-thread execution verified.', flush=True)
    if args.validate_only:
        return
    if args.threads != 32:
        raise ValueError('Full collection uses the predeclared 32 OpenMP threads')
    if not (work/'mwpm/completion.json').exists():
        raise RuntimeError('MWPM phase must finish before UF collection to preserve the 32-core cap')
    root = work/'uf'
    (root/'shards').mkdir(parents=True, exist_ok=True)
    configuration = dict(variants=VARIANTS, fields=FIELDS, source_sha256=new_sources,
        sample=dict(sample.identities), sample_seed=sample.seed, shots=sample.shots,
        parameters=sample.parameters.to_json(), threads=args.threads,
        batch_size=args.batch_size, l2='plain MWPM', confidence_calibration='none',
        bp_iterations=5, bp_damping=0.5, bp_llr_clip=30.,
        bp_projection='-log(clip(sum of supporting fault posteriors, 1e-15, 1))',
        bp_correlation_rules='DEM-prior conditional discounts after first UF on BP-projected weights',
        cpu_affinity=launch['cpu_affinity'], library_sha256=sha(library), build_command=command)
    config_path = root/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(configuration)):
        raise ValueError('Saved UF configuration differs from requested run')
    write_json(config_path, configuration)
    decoders = [make_native(library, patch.local_dem) for patch in patches]
    started = time.perf_counter()
    seconds_native = 0.
    seconds_outer = 0.
    for start in range(0, sample.shots, args.batch_size):
        stop = min(start+args.batch_size, sample.shots)
        name = f'{start:06d}-{stop:06d}'
        shard = root/'shards'/f'{name}.npz'
        checkpoint = root/'shards'/f'{name}.json'
        if checkpoint.exists():
            metadata = json.loads(checkpoint.read_text())
            if metadata['sha256'] != sha(shard) or metadata['source_sha256'] != new_sources:
                raise ValueError(f'Invalid saved shard {name}')
            continue
        rows = np.arange(start, stop)
        detectors, actual = sample.rows(rows)
        count = stop-start
        reference = np.zeros((count, 2, 12), dtype=bool)
        gaps = np.zeros((count, 2, 12))
        statistics = np.zeros((count, 2, 6, len(FIELDS)))
        native_start = time.perf_counter()
        for index, (patch, decoder) in enumerate(zip(patches, decoders)):
            local = patch.local_syndromes(detectors)
            packed = np.packbits(local, axis=1, bitorder='little')
            decoded, _, _ = decoder.decode(packed, args.threads)
            statistics[:, :, index] = decoded
            masks = decoded[:, :, 0].astype(np.uint8)
            reference[:, :, 2*index] = masks & 1
            reference[:, :, 2*index+1] = (masks >> 1) & 1
            gaps[:, :, 2*index:2*index+2] = decoded[:, :, 1:3]
        native_seconds = time.perf_counter()-native_start
        outer_start = time.perf_counter()
        yoke = detectors[:, patches.yoke_detector_ids]
        residual = np.zeros_like(reference)
        sigma = np.zeros((count, 2, 2), dtype=bool)
        tied = np.zeros_like(sigma)
        for variant in range(2):
            sigma[:, variant] = frame_adjusted_syndrome(yoke, reference[:, variant])
            for sector in range(2):
                decision = mwpm_outer_log_odds_batch(gaps[:, variant, sector::2], sigma[:, variant, sector])
                residual[:, variant, sector::2] = decision.patterns
                tied[:, variant, sector] = decision.tied
        prediction = reference ^ residual
        for variant in range(2):
            parity = prediction[:, variant].reshape(count, 6, 2).sum(axis=1) % 2
            np.testing.assert_array_equal(parity, yoke)
        outer_seconds = time.perf_counter()-outer_start
        # All temporary output lives in the scratch run.
        temporary = shard.with_suffix('.partial')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, row_ids=rows, actual=actual, reference=reference,
                gaps=gaps, prediction=prediction, residual=residual, sigma=sigma,
                outer_tied=tied, statistics=statistics)
        temporary.replace(shard)
        write_json(checkpoint, dict(start=start, stop=stop, sha256=sha(shard),
            source_sha256=new_sources, native_seconds=native_seconds, outer_seconds=outer_seconds))
        seconds_native += native_seconds
        seconds_outer += outer_seconds
        progress = dict(state='uf_running', completed_shots=stop, target_shots=sample.shots,
            threads=args.threads, elapsed_seconds=time.perf_counter()-started,
            updated_utc=datetime.now(timezone.utc).isoformat(), output=str(output), work=str(work))
        write_json(output/'progress.json', progress)
        print(f'UF/BP: {stop}/{sample.shots} paired shots; elapsed {progress["elapsed_seconds"]:.1f}s', flush=True)
    for decoder in decoders:
        decoder.close()
    paths = sorted((root/'shards').glob('*.npz'))
    chunks = {}
    for path in paths:
        with np.load(path) as values:
            for name in values.files:
                chunks.setdefault(name, []).append(values[name])
    arrays = {name:np.concatenate(values) for name, values in chunks.items()}
    np.testing.assert_array_equal(arrays['row_ids'], np.arange(sample.shots))
    np.savez_compressed(root/'predictions.npz', **arrays)
    write_json(root/'completion.json', dict(state='complete', shots=sample.shots,
        configuration_sha256=sha(config_path), predictions_sha256=sha(root/'predictions.npz'),
        elapsed_seconds_this_collection=time.perf_counter()-started,
        native_seconds_this_collection=seconds_native, outer_seconds_this_collection=seconds_outer,
        validation=validation))
    write_json(output/'progress.json', dict(state='decoding_complete', shots=sample.shots,
        output=str(output), work=str(work), updated_utc=datetime.now(timezone.utc).isoformat()))
    print('Both UF configurations complete. Ready to aggregate all four variants.', flush=True)


if __name__ == '__main__':
    main()
