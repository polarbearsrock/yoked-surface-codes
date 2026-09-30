#!/usr/bin/env python3
"""Run the three existing hierarchies on paired shots, one distance at a time.

For each distance, first collect the two correlated-MWPM baselines, then run
correlated UF on the same saved sample. There is only one worker pool at a
time. Completed shards and distances can be reused after a restart.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
REPO = Path(os.environ['DANTE_REPO'])
WORKSPACE = Path(os.environ['DANTE_WORKSPACE'])
sys.path.insert(0, str(REPO / 'src'))

from yoked.hierarchical._collect import CircuitParameters
from yoked.hierarchical._correlated_uf_experiment import SOURCES as UF_SOURCES
from yoked.hierarchical._mpp import load_mpp_native
from yoked.hierarchical._mpp_experiment import SOURCES as MPP_SOURCES
from yoked.hierarchical._provenance import (
    git_commit, package_versions, sha256_file, source_hashes, utc_now, write_json_atomic,
)

SOURCES = tuple(sorted(set(UF_SOURCES) | set(MPP_SOURCES)))
RECIPES = {
    'mwpm': 'mpp_confidence_100k.py',
    'uf': 'correlated_uf_cluster_gap.py',
}


def stage_commands(args, distance):
    """Plan both stages without submitting work for another distance."""
    directory = args.out / f'd{distance}'
    shared = ['--distances', str(distance), '--shots', str(args.shots),
              '--workers', str(args.workers), '--shard-size', str(args.shard_size),
              '--p', str(args.p), '--rounds-per-distance', '4']
    for stage in ('mwpm', 'uf'):
        command = [sys.executable, str(args.out / 'recipes' / RECIPES[stage]),
                   '--out', str(directory / stage), *shared]
        if stage == 'mwpm':
            command += ['--seed-base', str(args.seed_base)]
        else:
            command += ['--baseline-root', str(directory / 'mwpm')]
        yield stage, directory / stage, command


def stage_progress(root, distance):
    """Count saved shards separately from work reported by live worker logs."""
    completed = {}
    for path in (root / f'd{distance}' / 'shards').glob('*/manifest.json'):
        if path.parent.name.startswith('.'):
            continue
        start, stop = map(int, path.parent.name.split('-'))
        completed[path.parent.name] = stop - start
    decoded = sum(completed.values())
    active_logs = 0
    for path in (root / 'logs').glob(f'd{distance}-*.log'):
        key = path.stem.removeprefix(f'd{distance}-')
        if key in completed:
            continue
        active_logs += 1
        matches = re.findall(r'(\d+)/(\d+) (?:paired )?shots decoded', path.read_text())
        if matches:
            decoded += int(matches[-1][0])
    return dict(saved_shots=sum(completed.values()), decoded_shots=decoded,
                completed_shards=len(completed), unfinished_worker_logs=active_logs)


def run_stage(command, log, progress):
    """Run one bounded pool and publish progress until it has exited.

    The child stage has its own process group so an interrupted supervisor can
    stop exactly that stage and its decoder workers, leaving other jobs alone.
    """
    with log.open('a') as stream:
        stream.write('Command arguments: ' + json.dumps(command) + '\n')
        stream.flush()
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while True:
                progress(child.pid)
                try:
                    returncode = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    pass
            progress(child.pid)
            if returncode:
                raise RuntimeError(f'Stage exited with code {returncode}; inspect {log}')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


def verify_completed_stage(root, distance, parameters, shots):
    """Validate the completion chain before reusing or publishing a distance."""
    completion = json.loads((root / 'completion.json').read_text())
    for name, key in (('run.json', 'run_sha256'), ('summary.json', 'summary_sha256')):
        if sha256_file(root / name) != completion[key]:
            raise ValueError(f'Completion hash mismatch: {root / name}')
    directory = root / f'd{distance}'
    manifest_path = directory / 'manifest.json'
    if sha256_file(manifest_path) != completion['distance_manifests'][str(distance)]:
        raise ValueError(f'Distance manifest hash mismatch: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['artifacts'].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f'Artifact hash mismatch: {directory / name}')
    report = json.loads((directory / 'results.json').read_text())
    if report['parameters'] != parameters or report['shots'] != shots:
        raise ValueError('Completed stage has different circuit settings or shot count')
    return report


def prepare_run(args):
    """Freeze the experiment recipe and record the exact decoder implementation."""
    _, native = load_mpp_native()
    configuration = dict(
        schema='hierarchical-sequential-sweep/1',
        distances=args.distances, shots_per_distance=args.shots,
        workers=args.workers, shard_size=args.shard_size, seed_base=args.seed_base,
        parameters={str(d): CircuitParameters(distance=d, rounds=4*d, p=args.p).to_json()
                    for d in args.distances},
        stage_order=['mwpm', 'uf'],
        configurations=['correlated_mwpm_complementary_gap', 'correlated_mwpm_mpp',
                        'correlated_uf_cluster_gap'],
        l2='plain MWPM', confidence_calibration='none',
        sampling='fresh sample per distance, shared by all three configurations',
        provenance=dict(
            code_commit=git_commit(), source_sha256=source_hashes(SOURCES),
            native_build=native,
            versions=package_versions(('numpy', 'scipy', 'stim', 'pymatching', 'sinter')),
        ),
        recipes={stage: sha256_file(WORKSPACE / 'experiments' / filename)
                 for stage, filename in RECIPES.items()},
        scheduler_sha256=sha256_file(Path(__file__)),
    )
    path = args.out / 'run.json'
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError('Existing run settings or sources differ; use a new run directory')
        for stage, filename in RECIPES.items():
            if sha256_file(args.out / 'recipes' / filename) != configuration['recipes'][stage]:
                raise ValueError('Saved recipe was modified')
    else:
        # The lock file belongs to this supervisor, not to an earlier run.
        if any(p.name != '.runner.lock' for p in args.out.iterdir()):
            raise ValueError('Run directory is neither empty nor a compatible saved run')
        (args.out / 'recipes').mkdir()
        for filename in RECIPES.values():
            shutil.copy2(WORKSPACE / 'experiments' / filename, args.out / 'recipes' / filename)
        shutil.copy2(__file__, args.out / 'recipe.py')
        for source in SOURCES:
            target = args.out / 'source_snapshot' / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / source, target)
        write_json_atomic(path, configuration)
    (args.out / 'logs').mkdir(exist_ok=True)
    return configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--retained', type=Path, required=True)
    parser.add_argument('--distances', type=int, nargs='+', default=[7, 9, 11, 13, 15])
    parser.add_argument('--shots', type=int, default=1_000_000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--shard-size', type=int, default=1250)
    parser.add_argument('--seed-base', type=int, default=2026092400)
    parser.add_argument('--p', type=float, default=0.003)
    args = parser.parse_args()
    if min(args.shots, args.workers, args.shard_size, *args.distances) < 1:
        parser.error('counts and distances must be positive')
    if len(set(args.distances)) != len(args.distances):
        parser.error('distances must be distinct')
    if not 0 < args.p < 1 or not 0 <= args.seed_base < 2**64-max(args.distances):
        parser.error('invalid probability or sampling seed')
    args.distances.sort()
    args.out, args.retained = args.out.resolve(), args.retained.resolve()
    if not args.out.is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('--out must be under $TMPDIR')
    if args.retained.is_relative_to(args.out) or args.out.is_relative_to(args.retained):
        parser.error('scratch and retained directories must be separate')
    args.out.mkdir(parents=True, exist_ok=True)
    args.retained.mkdir(parents=True, exist_ok=True)

    # A duplicate launch must not compete for shards or overwrite live status.
    lock = (args.out / '.runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another supervisor is already running this sweep') from None

    def status(state, **details):
        value = dict(state=state, updated_utc=utc_now(), supervisor_pid=os.getpid(),
                     workers=args.workers, distance_order=args.distances,
                     target_shots_per_distance=args.shots, **details)
        for directory in (args.out, args.retained):
            write_json_atomic(directory / 'status.json', value)

    def request_stop(signum, _frame):
        raise KeyboardInterrupt(f'Supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, request_stop)
    try:
        configuration = prepare_run(args)
        status('validated')
        write_json_atomic(args.retained / 'run.json', configuration)
        started, summaries = time.perf_counter(), {}
        for distance in args.distances:
            parameters = configuration['parameters'][str(distance)]
            for stage, root, command in stage_commands(args, distance):
                if source_hashes(SOURCES) != configuration['provenance']['source_sha256']:
                    raise ValueError('Decoder sources changed during the queued sweep')
                details = dict(active_distance=distance, active_stage=stage,
                               completed_distances=list(summaries))
                if not (root / 'completion.json').exists():
                    print(f'Starting d={distance}, {stage}: {args.shots} shots, '
                          f'at most {args.workers} workers', flush=True)
                    log = args.out / 'logs' / f'd{distance}-{stage}.log'
                    run_stage(command, log, lambda pid: status(
                        'running', **details, stage_pid=pid, log=str(log),
                        **stage_progress(root, distance)))
                report = verify_completed_stage(root, distance, parameters, args.shots)
                print(f'Verified d={distance}, {stage}: {args.shots} shots', flush=True)

            # The distance barrier includes aggregation and publication. The
            # next distance gets no CPU workers until this one is retained.
            status('retaining_distance', active_distance=distance,
                   completed_distances=list(summaries))
            source = args.out / f'd{distance}'
            destination = args.retained / 'run' / f'd{distance}'
            shutil.copytree(source, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('.working-*'))
            public = args.retained / 'distances' / f'd{distance}'
            public.mkdir(parents=True, exist_ok=True)
            for name in ('predictions.npz', 'results.json', 'manifest.json'):
                shutil.copy2(source / 'uf' / f'd{distance}' / name, public / name)
            shutil.copy2(source / 'mwpm' / f'd{distance}' / 'results.json',
                         public / 'mwpm_results.json')
            summaries[str(distance)] = report
            for directory in (args.out, args.retained):
                write_json_atomic(directory / 'summary.json', summaries)
            print(f'Completed and retained d={distance}: {args.shots} paired shots', flush=True)

        completion = dict(
            created_utc=utc_now(), elapsed_seconds=time.perf_counter()-started,
            run_sha256=sha256_file(args.out / 'run.json'),
            summary_sha256=sha256_file(args.out / 'summary.json'),
            completed_distances=args.distances,
            total_paired_shots=args.shots * len(args.distances),
        )
        for directory in (args.out, args.retained):
            write_json_atomic(directory / 'completion.json', completion)
        for name in ('run.json', 'recipe.py', 'summary.json', 'completion.json'):
            shutil.copy2(args.out / name, args.retained / 'run' / name)
        for name in ('recipes', 'source_snapshot', 'logs'):
            shutil.copytree(args.out / name, args.retained / 'run' / name, dirs_exist_ok=True)
        status('complete', completed_distances=args.distances,
               results=str(args.retained / 'run'))
        print(f'Completed entire sweep: {args.retained}', flush=True)
    except BaseException as error:
        traceback.print_exc()
        status('failed', error=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
