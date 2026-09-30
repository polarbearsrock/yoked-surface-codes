"""Finish an existing sweep after its interactive terminal disappears.

Run on the host, in a detached session. This supervisor leaves an active
pipeline alone and uses the frozen, checkpoint-aware recipes only after all
writers for this experiment have exited. It never recollects accuracy data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def active_writers(output):
    """Find only Python jobs explicitly writing this run's output directory."""
    found = []
    names = {"pipeline.py", "run_uf.py", "latency.py", "analyze.py"}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().decode().split("\0")
            if not args or not Path(args[0]).name.startswith("python"):
                continue
            if len(args) < 2 or Path(args[1]).name not in names:
                continue
            if "--output" not in args:
                continue
            candidate = Path(args[args.index("--output") + 1])
            if not candidate.is_absolute():
                candidate = (entry / "cwd").resolve() / candidate
            if candidate.resolve() != output:
                continue
            stat = (entry / "stat").read_text().split(") ", 1)[1].split()
            if stat[0] != "Z":
                found.append({"pid": int(entry.name), "recipe": Path(args[1]).name})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    launch = read_json(output / "launch.json")
    work = Path(launch["work"])
    state_path = work / "detached_status.json"
    target = output / "source_snapshot/experiments/yoked_fourway_sweep"
    lock = (work / "detached_continuation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def record(state, **details):
        value = dict(state=state, pid=os.getpid(), parent_pid=os.getppid(),
                     session_id=os.getsid(0), host=os.uname().nodename,
                     updated_utc=datetime.now(timezone.utc).isoformat(), **details)
        write_json(state_path, value)

    for distance in launch["distances"]:
        assert (output / f"d{distance}/summary.json").exists(), "Accuracy incomplete"
        assert (output / f"d{distance}/paired.npz").exists(), "Missing paired results"

    print("Detached continuation active; waiting for the existing pipeline.", flush=True)
    try:
        while writers := active_writers(output):
            record("waiting_for_existing_pipeline", writers=writers)
            time.sleep(20)

        if read_json(output / "progress.json").get("state") == "complete":
            record("complete", resumed=False, report=str(output / "report.md"))
            print("Existing pipeline completed normally; no restart needed.", flush=True)
            return

        manifest = read_json(output / "source_manifest.json")
        for relative, expected in manifest.items():
            actual = hashlib.sha256((output / "source_snapshot" / relative).read_bytes()).hexdigest()
            assert actual == expected, f"Frozen source mismatch: {relative}"

        print("Original writers exited; resuming saved latency checkpoints.", flush=True)
        for name in ("latency.py", "analyze.py"):
            command = [sys.executable, str(target / name), "--output", str(output)]
            record("resuming", recipe=name, command=command)
            subprocess.run(command, check=True)
        assert read_json(output / "progress.json")["state"] == "complete"
        record("complete", resumed=True, report=str(output / "report.md"))
        print("Accuracy, latency, and final report are complete.", flush=True)
    except BaseException as error:
        record("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
