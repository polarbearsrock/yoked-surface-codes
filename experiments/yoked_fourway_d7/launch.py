"""Freeze sources and launch the unchanged paired MWPM confidence experiment."""
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
REPO = Path(os.environ['DANTE_REPO'])
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_cores(count=32):
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((root/'physical_package_id').read_text().strip(),
               (root/'core_id').read_text().strip())
        cores.setdefault(key, cpu)
    if len(cores) < count:
        raise RuntimeError(f'Need {count} physical cores; found {len(cores)}')
    return list(cores.values())[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    scratch = Path(os.environ['DANTE_SCRATCH'])/'runs'
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'yoked_fourway_d7_{stamp}_', dir=scratch))
    output = WORKSPACE/'results'/f'yoked_fourway_d7_p003_100k_{stamp}'
    output.mkdir()
    fork = json.loads((WORKSPACE/'env/stim-fork.json').read_text())
    import stim
    if stim.__version__ != fork['version'] or sha(fork['native_module']) != fork['native_sha256']:
        raise RuntimeError('Installed Stim differs from the recorded fork build')
    snapshot = output/'source_snapshot'
    paths = list((REPO/'src').rglob('*.py'))
    paths += [p for p in HERE.iterdir() if p.suffix in ('.py', '.cc', '.md')]
    paths += [WORKSPACE/'experiments/mpp_confidence_100k.py', WORKSPACE/'env/stim-fork.json']
    paths += [REPO/p for p in ('native/mpp/mpp_fast.cc', 'native/mpp/CMakeLists.txt',
                              'tools/build_mpp', 'tools/mpp_experiment', 'tools/hierarchical_experiment')]
    for name, files in {
        'uf_evidence_predecoder_d7_d13': ('evidence.py', 'kernel.cc'),
        'parallel_uf_clustering_d7_d9_p003_100k': ('experiment.py', 'kernel.cc'),
    }.items():
        paths += [REPO/'docs/results'/name/f for f in files]
    hashes = {}
    for source in sorted(set(paths)):
        relative = source.relative_to(WORKSPACE)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = sha(target)
        if hashes[str(relative)] != sha(source):
            raise RuntimeError(f'Source changed during snapshot: {relative}')
    (output/'source_manifest.json').write_text(json.dumps(hashes, indent=2, sort_keys=True)+'\n')
    shutil.copyfile(HERE/'README.md', output/'protocol.md')
    revision = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    diff = subprocess.check_output(['git','-C',str(REPO),'diff','--binary'], text=True)
    (output/'repository.diff').write_text(diff)
    cores = physical_cores()
    os.sched_setaffinity(0, cores)
    environment = os.environ.copy()
    environment.update(DANTE_REPO=str(snapshot/'repos/yoked-surface-codes'),
        YOKED_MPP_BUILD=str(Path(os.environ['DANTE_SCRATCH'])/'build/mpp-gcc14'),
        OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    environment.pop('OMP_PLACES', None)
    environment.pop('OMP_PROC_BIND', None)
    command = [sys.executable, str(snapshot/'experiments/mpp_confidence_100k.py'),
        '--out', str(work/'mwpm'), '--p','0.003','--rounds-per-distance','4',
        '--workers','32','--shots','100000','--shard-size','1250','--distances','7',
        '--seed-base','202609290700']
    log = work/'mwpm.log'
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=WORKSPACE, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(output=str(output), work=str(work), mwpm_pid=process.pid,
        mwpm_log=str(log), mwpm_command=command, cpu_affinity=cores, workers=32,
        shots=100000, distance=7, rounds=28, p=0.003, patches=6, yokes=2,
        noise='si1000', sample_seed=202609290707, ideal_time_boundaries=True,
        l2='plain MWPM', source_commit=revision, stim_fork=fork,
        snapshot_repository=environment['DANTE_REPO'], mpp_build=environment['YOKED_MPP_BUILD'])
    (output/'launch.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    (output/'progress.json').write_text(json.dumps(dict(state='mwpm_running', **record), indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    if args.wait:
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
