"""Freeze and launch the sequential d=9,11,13 comparison within 32 cores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--wait', action='store_true')
    mode.add_argument('--tmux', action='store_true', help='Run in a detached server-side tmux session')
    args = parser.parse_args()
    reference = args.reference.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = WORKSPACE/'results'/f'bp_uf_single_patch_d9_d11_d13_p003_{stamp}'
    output.mkdir()
    scratch = Path(os.environ['DANTE_SCRATCH'])/'runs'
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'bp_uf_d9_d11_d13_{stamp}_', dir=scratch))
    snapshot = output/'source_snapshot'
    manifest = {}

    def copy(source, relative, expected=None):
        if expected is not None and sha256(source) != expected:
            raise ValueError(f'Source identity mismatch: {source}')
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[str(relative)] = sha256(target)
        assert manifest[str(relative)] == sha256(source)

    for relative, expected in json.loads((reference/'source_manifest.json').read_text()).items():
        copy(reference/'source_snapshot'/relative, Path(relative), expected)
    for folder, name in [('bp10_uf', 'add_bp10.py'), ('correlated_mwpm', 'add_correlated_mwpm.py')]:
        expected = json.loads((reference/folder/'artifact_manifest.json').read_text())[name]
        copy(reference/folder/name, Path('experiments/bp_uf_single_patch_d7')/name, expected)
    for path in sorted(HERE.iterdir()):
        if path.suffix in ('.py', '.cc', '.md'):
            copy(path, path.relative_to(WORKSPACE))
    (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    shutil.copyfile(HERE/'README.md', output/'protocol.md')
    for name in ('repository.json', 'repository.diff'):
        shutil.copyfile(reference/name, output/name)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='32', OMP_THREAD_LIMIT='32', OMP_DYNAMIC='FALSE',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
        PYTHONPATH=str(snapshot/'repos/yoked-surface-codes/src'))
    env.pop('OMP_PLACES', None)
    env.pop('OMP_PROC_BIND', None)
    runner = snapshot/'experiments/bp_uf_single_patch_sweep/run.py'
    command = [sys.executable, str(runner), '--output', str(output), '--work-dir', str(work),
               '--reference', str(reference)]
    log_path = work/'run.log'
    detached = {}
    if args.tmux:
        tmux = shutil.which('tmux')
        if tmux is None:
            raise RuntimeError('tmux is not installed')
        socket = work/'control.sock'
        session = 'decoder_sweep'
        wrapper = work/'run.sh'
        exports = ['OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_DYNAMIC', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'PYTHONPATH',
            'TMPDIR', 'DANTE_SCRATCH', 'PYTHONPYCACHEPREFIX', 'MPLCONFIGDIR', 'RUFF_CACHE_DIR']
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            + '\n'.join(f'export {k}={shlex.quote(env[k])}' for k in exports if k in env)
            + '\nunset OMP_PLACES OMP_PROC_BIND\n'
            + f'cd {shlex.quote(str(WORKSPACE))}\n'
            + f'exec {shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1\n')
        subprocess.run([tmux, '-S', str(socket), 'new-session', '-d', '-s', session,
            '-c', str(WORKSPACE), shlex.join(['/usr/bin/bash', str(wrapper)])], env=env, check=True)
        subprocess.run([tmux, '-S', str(socket), 'set-window-option', '-t', session,
            'remain-on-exit', 'on'], env=env, check=True)
        process_pid = int(subprocess.check_output([tmux, '-S', str(socket), 'display-message',
            '-p', '-t', session, '#{pane_pid}'], env=env, text=True).strip())
        detached = dict(mode='detached tmux', tmux=tmux, socket=str(socket), session=session,
            status_command=[tmux, '-S', str(socket), 'display-message', '-p', '-t', session,
                            '#{pane_pid} #{pane_dead} #{pane_dead_status}'])
    else:
        with log_path.open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process_pid = process.pid
    record = dict(pid=process_pid, distances=[9, 11, 13], total_physical_cores=32,
        ordering='sequential distances; native and MWPM stages do not overlap',
        output=str(output), scratch=str(work), log=str(log_path), command=command,
        progress=str(output/'progress.json'), reference=str(reference), **detached)
    (output/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    (output/'README.md').write_text('# Fixed BP + UF distance extension\n\n'
        'd=9, 11, 13; one patch each; SI1000 p=0.003; 4d rounds; seven decoders; 32 cores total.\n\n'
        '- [Protocol](protocol.md)\n- [Live progress](progress.json)\n- [Launch and resume command](launch.json)\n'
        '- [Combined report](report.md)\n\n'
        f'Live log: `{log_path}`.\n\n'
        'Distances run sequentially, each with 10,000 pilot shots, 100,000 independent confirmation shots, '
        'and 1,024 serial timing rows. Reports update automatically as phases complete.\n')
    print(json.dumps(record, indent=2), flush=True)
    if args.wait:
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
