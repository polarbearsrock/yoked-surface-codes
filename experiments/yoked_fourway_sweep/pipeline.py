"""Continue the authorized sweep through accuracy, latency, and final publication."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    launch=json.loads((output/'launch.json').read_text())
    work=Path(launch['work'])
    target=output/'source_snapshot/experiments/yoked_fourway_sweep'
    manifest=json.loads((output/'source_manifest.json').read_text())
    for source in Path(__file__).resolve().parent.iterdir():
        if source.suffix not in ('.py','.cc','.md'):
            continue
        destination=target/source.name
        if source!=destination:
            if any(work.glob('uf/d*/configuration.json')) and source.name in ('README.md','core.py','kernel.cc','run_uf.py') and destination.read_bytes()!=source.read_bytes():
                raise ValueError('Collected decoder sources cannot change')
            shutil.copyfile(source,destination)
        manifest[str(destination.relative_to(output/'source_snapshot'))]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    if Path(__file__).resolve()!=target/'pipeline.py':
        os.execv(sys.executable,[sys.executable,str(target/'pipeline.py'),*sys.argv[1:]])
    print('Pipeline waiting for the active MWPM phase to complete.',flush=True)
    while not (work/'mwpm/completion.json').exists():
        if (output/'mwpm_failed.json').exists():
            raise RuntimeError('MWPM failure recorded; refusing to continue')
        time.sleep(10)
    commands=[
        [sys.executable,str(target/'run_uf.py'),'--output',str(output)],
        [sys.executable,str(target/'analyze.py'),'--output',str(output),'--accuracy-only'],
        [sys.executable,str(target/'latency.py'),'--output',str(output)],
        [sys.executable,str(target/'analyze.py'),'--output',str(output)],
    ]
    (output/'pipeline_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    for command in commands:
        print(datetime.now(timezone.utc).isoformat()+' RUN '+json.dumps(command),flush=True)
        result=subprocess.run(command,check=False)
        if result.returncode:
            (output/'pipeline_failure.json').write_text(json.dumps(dict(command=command,returncode=result.returncode),indent=2)+'\n')
            raise SystemExit(result.returncode)
    print('The full accuracy and latency sweep is complete.',flush=True)


if __name__=='__main__':
    main()
