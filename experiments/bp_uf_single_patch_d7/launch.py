"""Freeze the experiment and launch its 32-core runner in a detached process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common import HERE, REPO, WORKSPACE, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[32], default=32)
    parser.add_argument('--wait', action='store_true',
                        help='Keep the launcher alive for managed execution sessions')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    scratch_root = Path(os.environ['DANTE_SCRATCH'])/'runs'
    scratch_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'bp_uf_single_patch_d7_{stamp}_', dir=scratch_root))
    output = WORKSPACE/'results'/f'bp_uf_single_patch_d7_p003_{stamp}'
    output.mkdir()
    snapshot = output/'source_snapshot'
    paths = [*HERE.glob('*.py'), HERE/'kernel.cc', HERE/'README.md']
    paths.extend((REPO/'src').rglob('*.py'))
    for folder, names in {
        'uf_evidence_predecoder_d7_d13': ('evidence.py', 'kernel.cc', 'verify.py'),
        'parallel_uf_clustering_d7_d9_p003_100k': ('experiment.py', 'kernel.cc'),
    }.items():
        paths.extend(REPO/'docs/results'/folder/name for name in names)
    manifest = {}
    for source in sorted(set(paths)):
        relative = source.relative_to(WORKSPACE)
        destination = snapshot/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = sha256(destination)
        if manifest[str(relative)] != sha256(source):
            raise RuntimeError(f'Source changed during snapshot: {source}')
    write_json(output/'source_manifest.json', manifest)
    shutil.copyfile(HERE/'README.md', output/'protocol.md')
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True)
    diff = subprocess.check_output(['git', '-C', str(REPO), 'diff', '--binary'], text=True)
    (output/'repository.diff').write_text(diff)
    write_json(output/'repository.json', dict(path=str(REPO), head=revision, status=status,
        note='Frozen source contents and hashes are authoritative for modified and untracked files.'))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='32', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    # The runner binds to physical cores before libgomp loads the native kernel.
    # Do not impose an inherited OMP placement mask that could narrow that set.
    env.pop('OMP_PLACES', None)
    env.pop('OMP_PROC_BIND', None)
    env['PYTHONPATH'] = str(snapshot/'repos/yoked-surface-codes/src')
    runner = snapshot/'experiments/bp_uf_single_patch_d7/run.py'
    command = [sys.executable, str(runner), '--threads', str(args.threads),
               '--work-dir', str(work), '--output', str(output)]
    log_path = work/'run.log'
    with log_path.open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=WORKSPACE, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    launch = dict(pid=process.pid, threads=32, scratch=str(work), output=str(output),
        log=str(log_path), progress=str(output/'progress.json'), command=command,
        environment_overrides={k: env[k] for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
            'OMP_DYNAMIC', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONPATH')})
    write_json(output/'launch.json', launch)
    (output/'README.md').write_text('# Fixed BP + UF single-patch experiment\n\n'
        'One d=7 patch; SI1000 p=0.003; 28 rounds; five fixed decoder variants; 32 physical cores.\n\n'
        '- [Frozen protocol](protocol.md)\n- [Progress](progress.json)\n'
        '- [Launch details](launch.json)\n- [Report](report.md) (created after the pilot)\n\n'
        f'Live log: `{log_path}`.\n\n'
        'The runner validates first, collects 10,000 pilot shots and 100,000 independent '
        'confirmation shots, and measures serial kernel timings. The launch record '
        'contains the exact command for resuming after an interruption.\n')
    print(json.dumps(launch, indent=2), flush=True)
    if args.wait:
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
