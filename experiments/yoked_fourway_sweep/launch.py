"""Freeze the prior four-way experiment and launch the four new MWPM distances."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

WORKSPACE = Path(os.environ['DANTE_WORKSPACE'])
HERE = Path(__file__).resolve().parent


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=WORKSPACE/'results/yoked_bp5_weighted_d7_p003_100k_20260929T204943Z')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    prior = json.loads((baseline/'launch.json').read_text())
    artifacts = json.loads((baseline/'artifact_manifest.json').read_text())
    assert json.loads((baseline/'progress.json').read_text())['state'] == 'complete'
    for name, expected in artifacts.items():
        if sha(baseline/name) != expected:
            raise ValueError(f'Baseline checksum mismatch: {name}')
    import pymatching, stim
    assert pymatching.__version__ == '2.4.0'
    assert stim.__version__ == prior['stim_fork']['version']
    assert sha(prior['stim_fork']['native_module']) == prior['stim_fork']['native_sha256']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = WORKSPACE/'results'/f'yoked_fourway_sweep_p003_4d_100k_{stamp}'
    output.mkdir()
    scratch = Path(os.environ['DANTE_SCRATCH'])/'runs'
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'yoked_fourway_sweep_{stamp}_', dir=scratch))
    snapshot = output/'source_snapshot'
    shutil.copytree(baseline/'source_snapshot', snapshot)
    manifest = json.loads((baseline/'source_manifest.json').read_text())
    for source in HERE.iterdir():
        if source.suffix not in ('.py', '.cc', '.md'):
            continue
        relative = Path('experiments/yoked_fourway_sweep')/source.name
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[str(relative)] = sha(target)
    (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    shutil.copyfile(HERE/'README.md', output/'protocol.md')
    shutil.copytree(baseline/'data/sample', output/'d7/sample')
    for src, dst in [('data/paired.npz', 'd7/paired.npz'), ('results.json', 'd7/results.json'),
                     ('artifact_manifest.json', 'd7/baseline_artifact_manifest.json'),
                     ('launch.json', 'd7/baseline_launch.json'),
                     ('baseline/repository.diff', 'repository.diff'),
                     ('baseline/execution_environment.json', 'baseline_execution_environment.json')]:
        shutil.copyfile(baseline/src, output/dst)
    cores = prior['cpu_affinity']
    assert len(cores) == 32 and set(cores) <= os.sched_getaffinity(0)
    physical = []
    for cpu in cores:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append(((topology/'physical_package_id').read_text().strip(),
                         (topology/'core_id').read_text().strip()))
    assert len(set(physical)) == 32
    os.sched_setaffinity(0, cores)
    build = Path(os.environ['DANTE_SCRATCH'])/'build/mpp-gcc14'
    build_meta = json.loads((build/'build.json').read_text())
    assert build_meta['dependencies']['pymatching'] == '6f63b2b9474ba0fa7e511fe52bffdce858a06984'
    assert sha(build/build_meta['module']) == build_meta['module_sha256']
    environment = os.environ.copy()
    environment.update(DANTE_REPO=str(snapshot/'repos/yoked-surface-codes'), YOKED_MPP_BUILD=str(build),
        OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    environment.pop('OMP_PLACES', None)
    environment.pop('OMP_PROC_BIND', None)
    command = [sys.executable, str(snapshot/'experiments/mpp_confidence_100k.py'),
        '--out', str(work/'mwpm'), '--distances', '9', '11', '13', '15', '--shots', '100000',
        '--p', '0.003', '--rounds-per-distance', '4', '--workers', '32', '--shard-size', '1250',
        '--seed-base', '202609290700']
    log = work/'mwpm.log'
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=WORKSPACE, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(output=str(output), work=str(work), baseline=str(baseline),
        baseline_manifest_sha256=sha(baseline/'artifact_manifest.json'),
        snapshot_repository=environment['DANTE_REPO'], source_commit=prior['source_commit'],
        distances=[7,9,11,13,15], rounds={str(d):4*d for d in [7,9,11,13,15]},
        p=.003, noise='si1000', patches=6, yokes=2, ideal_time_boundaries=True,
        shots_per_distance=100000, sampling_seed_base=202609290700,
        cpu_affinity=cores, physical_cores=physical, workers=32, threads=32,
        l2='plain PyMatching v2 MWPM', stim_fork=prior['stim_fork'],
        pymatching_version=pymatching.__version__, pymatching_path=pymatching.__file__,
        mpp_build=str(build), mpp_build_metadata=build_meta,
        mwpm_command=command, mwpm_log=str(log), mwpm_pid=process.pid,
        reused_accuracy_distances=[7], latency_shots_per_distance=1000,
        latency_warmup_shots=32, latency_selection_seed_base=202609292900)
    (output/'launch.json').write_text(json.dumps(launch, indent=2, sort_keys=True)+'\n')
    (output/'progress.json').write_text(json.dumps(dict(state='mwpm_running', output=str(output),
        work=str(work), log=str(log)), indent=2)+'\n')
    print(json.dumps(launch, indent=2), flush=True)
    if args.wait:
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
