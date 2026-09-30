"""Freeze and collect the corrected fourth variant on the prior paired sample."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

RECIPE = Path('experiments/yoked_bp5_weighted_d7')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(baseline, output):
    baseline = baseline.resolve()
    old = json.loads((baseline/'launch.json').read_text())
    if json.loads((baseline/'progress.json').read_text())['state'] != 'complete':
        raise ValueError('Baseline experiment is incomplete')
    manifest = json.loads((baseline/'artifact_manifest.json').read_text())
    for name, expected in manifest.items():
        if digest(baseline/name) != expected:
            raise ValueError(f'Baseline artifact checksum mismatch: {name}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if output is None:
        output = Path(os.environ['DANTE_WORKSPACE'])/'results'/f'yoked_bp5_weighted_d7_p003_100k_{stamp}'
    output = output.resolve()
    output.mkdir()
    scratch = Path(os.environ['DANTE_SCRATCH'])/'runs'
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'yoked_bp5_weighted_d7_{stamp}_', dir=scratch))
    snapshot = output/'source_snapshot'
    shutil.copytree(baseline/'source_snapshot', snapshot)
    sources = json.loads((baseline/'source_manifest.json').read_text())
    (snapshot/RECIPE).mkdir(parents=True)
    for source in Path(__file__).resolve().parent.iterdir():
        if source.suffix in ('.py', '.cc', '.md'):
            relative = RECIPE/source.name
            shutil.copyfile(source, snapshot/relative)
            sources[str(relative)] = digest(snapshot/relative)
    (output/'source_manifest.json').write_text(json.dumps(sources, indent=2, sort_keys=True)+'\n')
    (output/'baseline').mkdir()
    for name in ('launch.json', 'results.json', 'artifact_manifest.json', 'source_manifest.json',
                 'repository.diff', 'execution_environment.json', 'verification_32threads.json'):
        shutil.copyfile(baseline/name, output/'baseline'/name)
    (output/'data').mkdir()
    shutil.copytree(baseline/'data/sample', output/'data/sample')
    shutil.copyfile(baseline/'data/paired.npz', output/'data/baseline_paired.npz')
    shutil.copyfile(Path(__file__).parent/'README.md', output/'protocol.md')
    keys = ('shots', 'distance', 'rounds', 'p', 'patches', 'yokes', 'noise', 'sample_seed',
            'ideal_time_boundaries', 'l2', 'source_commit', 'stim_fork', 'cpu_affinity')
    launch = {key:old[key] for key in keys}
    launch.update(output=str(output), work=str(work), baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline/'artifact_manifest.json'),
        source_manifest_sha256=digest(output/'source_manifest.json'),
        snapshot_repository=str(snapshot/'repos/yoked-surface-codes'),
        variant='bp5_weighted_uf_cluster_gap', threads=32,
        command=[sys.executable, str(snapshot/RECIPE/'run.py'), '--output', str(output)])
    (output/'launch.json').write_text(json.dumps(launch, indent=2, sort_keys=True)+'\n')
    print(f'Results: {output}\nScratch: {work}', flush=True)
    return output


def collect(output):
    launch = json.loads((output/'launch.json').read_text())
    work = Path(launch['work'])
    repository = Path(launch['snapshot_repository'])
    os.environ['DANTE_REPO'] = str(repository)
    os.environ.update(OMP_NUM_THREADS='32', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    os.sched_setaffinity(0, launch['cpu_affinity'])
    sys.path.insert(0, str(repository/'src'))
    import numpy as np
    import stim
    from weighted import FIELDS, VARIANT, build, make_native, sha, verify, write_json
    from yoked.hierarchical._collect import SampleSet
    from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome
    from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
    from yoked.hierarchical._patch_graphs import PatchGraphs

    if sha(output/'source_manifest.json') != launch['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    sources = json.loads((output/'source_manifest.json').read_text())
    for name, expected in sources.items():
        if sha(output/'source_snapshot'/name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    fork = launch['stim_fork']
    if stim.__version__ != fork['version'] or sha(fork['native_module']) != fork['native_sha256']:
        raise ValueError('Installed Stim differs from the sample generation build')
    baseline_manifest = json.loads((output/'baseline/artifact_manifest.json').read_text())
    if sha(output/'data/baseline_paired.npz') != baseline_manifest['data/paired.npz']:
        raise ValueError('Baseline prediction checksum mismatch')
    for name, expected in baseline_manifest.items():
        if name.startswith('data/sample/') and sha(output/name) != expected:
            raise ValueError(f'Sample checksum mismatch: {name}')
    library, command = build(output/'source_snapshot'/RECIPE/'kernel.cc', work/'build')
    sample = SampleSet.load(output/'data/sample')
    assert sample.shots == launch['shots'] == 100000 and sample.seed == launch['sample_seed']
    patches = PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text), num_patches=6)
    fresh, _ = stim.Circuit(sample.circuit_text).compile_detector_sampler(seed=202609290708).sample(
        shots=16, separate_observables=True)
    started = time.perf_counter()
    validation = verify(library, patches, fresh)
    validation.update(elapsed_seconds=time.perf_counter()-started, validation_seed=202609290708,
        source_manifest_sha256=sha(output/'source_manifest.json'),
        library_sha256=sha(library), build_command=command)
    write_json(output/'verification.json', validation)
    print('Verified BP5 + weighted UF against existing native and Python implementations; '
          '32-thread execution, corrections, remaining costs and gaps passed.', flush=True)
    root = work/'weighted'
    (root/'shards').mkdir(parents=True, exist_ok=True)
    configuration = dict(variant=VARIANT, fields=FIELDS, shots=sample.shots, sample_seed=sample.seed,
        sample=dict(sample.identities), parameters=sample.parameters.to_json(),
        source_manifest_sha256=sha(output/'source_manifest.json'), library_sha256=sha(library),
        build_command=command, threads=32, cpu_affinity=launch['cpu_affinity'], batch_size=1250,
        bp_iterations=5, bp_damping=0.5, bp_llr_clip=30., uf_passes=1,
        bp_projection='-log(clip(sum of supporting fault posteriors, 1e-15, 1))',
        conditional_correlation_discounts=False, cluster_gap_cap=None,
        confidence_calibration='none', l2='plain MWPM')
    config_path = root/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(configuration)):
        raise ValueError('Saved configuration differs from requested run')
    write_json(config_path, configuration)
    decoders = [make_native(library, p.local_dem) for p in patches]
    started = time.perf_counter()
    for start in range(0, sample.shots, 1250):
        stop = min(start+1250, sample.shots)
        name = f'{start:06d}-{stop:06d}'
        shard, checkpoint = root/'shards'/f'{name}.npz', root/'shards'/f'{name}.json'
        if checkpoint.exists():
            metadata = json.loads(checkpoint.read_text())
            if metadata['sha256'] != sha(shard) or metadata['configuration_sha256'] != sha(config_path):
                raise ValueError(f'Invalid checkpoint {name}')
            continue
        rows = np.arange(start, stop)
        detectors, actual = sample.rows(rows)
        count = len(rows)
        reference = np.zeros((count, 12), dtype=bool)
        gaps = np.zeros((count, 12))
        statistics = np.zeros((count, 6, len(FIELDS)))
        native_start = time.perf_counter()
        for index, (patch, decoder) in enumerate(zip(patches, decoders)):
            packed = np.packbits(patch.local_syndromes(detectors), axis=1, bitorder='little')
            decoded, _, _ = decoder.decode(packed, 32)
            statistics[:, index] = decoded
            masks = decoded[:, 0].astype(np.uint8)
            reference[:, 2*index] = masks & 1
            reference[:, 2*index+1] = (masks >> 1) & 1
            gaps[:, 2*index:2*index+2] = decoded[:, 1:3]
        native_seconds = time.perf_counter()-native_start
        outer_start = time.perf_counter()
        yoke = detectors[:, patches.yoke_detector_ids]
        sigma = frame_adjusted_syndrome(yoke, reference)
        residual = np.zeros_like(reference)
        tied = np.zeros((count, 2), dtype=bool)
        for sector in range(2):
            decision = mwpm_outer_log_odds_batch(gaps[:, sector::2], sigma[:, sector])
            residual[:, sector::2] = decision.patterns
            tied[:, sector] = decision.tied
        prediction = reference ^ residual
        np.testing.assert_array_equal(prediction.reshape(count, 6, 2).sum(axis=1)%2, yoke)
        outer_seconds = time.perf_counter()-outer_start
        temporary = shard.with_suffix('.partial')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, row_ids=rows, actual=actual, reference=reference,
                gaps=gaps, prediction=prediction, residual=residual, sigma=sigma,
                outer_tied=tied, statistics=statistics)
        temporary.replace(shard)
        write_json(checkpoint, dict(start=start, stop=stop, sha256=sha(shard),
            configuration_sha256=sha(config_path), native_seconds=native_seconds,
            outer_seconds=outer_seconds))
        elapsed = time.perf_counter()-started
        write_json(output/'progress.json', dict(state='weighted_running', completed_shots=stop,
            target_shots=sample.shots, threads=32, elapsed_seconds=elapsed,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print(f'BP5 + weighted UF: {stop}/{sample.shots} paired shots; elapsed {elapsed:.1f}s', flush=True)
    for decoder in decoders:
        decoder.close()
    chunks, metadata = {}, []
    for path in sorted((root/'shards').glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != meta['sha256'] or meta['configuration_sha256'] != sha(config_path):
            raise ValueError(f'Invalid shard at final aggregation: {path}')
        metadata.append(meta)
        with np.load(path) as saved:
            for name in saved.files:
                chunks.setdefault(name, []).append(saved[name])
    arrays = {name:np.concatenate(values) for name, values in chunks.items()}
    np.testing.assert_array_equal(arrays['row_ids'], np.arange(sample.shots))
    np.savez_compressed(root/'predictions.npz', **arrays)
    write_json(root/'completion.json', dict(state='complete', shots=sample.shots,
        predictions_sha256=sha(root/'predictions.npz'), configuration_sha256=sha(config_path),
        elapsed_seconds_this_collection=time.perf_counter()-started,
        native_and_packing_seconds=sum(x['native_seconds'] for x in metadata),
        outer_seconds=sum(x['outer_seconds'] for x in metadata),
        validation_sha256=sha(output/'verification.json')))
    retained = output/'data/weighted'
    retained.mkdir(exist_ok=True)
    for name in ('predictions.npz', 'configuration.json', 'completion.json'):
        shutil.copyfile(root/name, retained/name)
    write_json(output/'progress.json', dict(state='decoding_complete', shots=sample.shots,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    print(f'Corrected decoder complete. Analyze with: python {output / "source_snapshot" / RECIPE / "analyze.py"} --output {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = prepare(args.baseline, args.output) if args.baseline else args.output
    if output is None:
        parser.error('Use --baseline to start or --output to resume')
    output = output.resolve()
    frozen = output/'source_snapshot'/RECIPE/'run.py'
    if Path(__file__).resolve() != frozen:
        os.execv(sys.executable, [sys.executable, str(frozen), '--output', str(output)])
    collect(output)


if __name__ == '__main__':
    main()
