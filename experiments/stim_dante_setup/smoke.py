"""Check the built Stim CLI and reference C++ sampler; not a statistical benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pymatching
import stim

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "experiments/dongwhee_noise_model"))
from probe import build, extract_builder


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = WORKSPACE / "results/dongwhee_noise_model_review_20260929/source_snapshot/01_Baseline"
    devices, namespace, builder, _ = extract_builder(reference / "run.py")
    executable = args.build / "surface_sim"
    records = []
    for device in ("google_willow", "ibm_berlin"):
        for basis in ("x", "z"):
            scope = build(device, 7, 7, f"rotated_memory_{basis}", devices, namespace, builder)
            circuit = scope["circuit"]
            prefix = f"{device}_d7_r7_{basis}"
            circuit_path = args.output / f"{prefix}.stim"
            circuit_path.write_text(str(circuit))
            matching = pymatching.Matching.from_detector_error_model(
                scope["dem"], enable_correlations=True)
            measurements_by_threads = {}
            for threads in (1, 32):
                sample_path = args.output / f"{prefix}_omp{threads}.01"
                report_path = args.output / f"{prefix}_omp{threads}.txt"
                command = [str(executable), "7", "7", "128",
                    str(scope["p1"]), str(scope["p2"]), str(scope["pm"]), str(scope["pr"]),
                    str(scope["T1_NS"]), str(scope["T2_NS"]),
                    str(scope["T_DATA_NS"]), str(scope["T_ANCILLA_NS"]),
                    str(scope["n_dd_pulses"]), f"rotated_memory_{basis}",
                    str(report_path), str(sample_path), "20260929"]
                environment = dict(os.environ, OMP_NUM_THREADS=str(threads),
                    OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="32", OMP_DISPLAY_ENV="VERBOSE")
                result = subprocess.run(command, env=environment, text=True,
                    capture_output=True, check=True, timeout=60)
                (args.output / f"{prefix}_omp{threads}.log").write_text(result.stdout + result.stderr)
                measurements = stim.read_shot_data_file(path=str(sample_path), format="01",
                    num_measurements=circuit.num_measurements)
                assert measurements.shape == (128, circuit.num_measurements)
                measurements_by_threads[threads] = measurements
            assert np.array_equal(measurements_by_threads[1], measurements_by_threads[32])
            reports = [args.output / f"{prefix}_omp{threads}.txt" for threads in (1, 32)]
            assert reports[0].read_text().replace(str(reports[0]), "<report>") == (
                reports[1].read_text().replace(str(reports[1]), "<report>"))
            detectors, observable = circuit.compile_m2d_converter().convert(
                measurements=measurements_by_threads[32], separate_observables=True)
            predictions = matching.decode_batch(detectors, enable_correlations=True)
            assert predictions.shape == observable.shape == (128, 1)
            records.append(dict(device=device, basis=basis, distance=7, rounds=7, shots=128,
                dd_pulses=scope["n_dd_pulses"], detector_shape=list(detectors.shape),
                identical_samples_and_fault_counts_at_threads=[1, 32], correlated_mwpm_decoded=True))

    # Noiseless preparation, measurement and propagation must produce no detections
    # or logical flips in either memory basis.
    for basis in ("x", "z"):
        scope = build("google_willow", 7, 7, f"rotated_memory_{basis}", devices, namespace, builder)
        circuit = scope["circuit"].without_noise()
        sample_path = args.output / f"noiseless_{basis}.01"
        command = [str(executable), "7", "7", "64", "0", "0", "0", "0",
            "68000", "89000", "0", "0", "0", f"rotated_memory_{basis}",
            str(args.output / f"noiseless_{basis}.txt"), str(sample_path), "20260929"]
        subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="32", OMP_DYNAMIC="FALSE",
            OMP_THREAD_LIMIT="32"), capture_output=True, text=True, check=True, timeout=60)
        measurements = stim.read_shot_data_file(path=str(sample_path), format="01",
            num_measurements=circuit.num_measurements)
        detectors, observable = circuit.compile_m2d_converter().convert(
            measurements=measurements, separate_observables=True)
        assert measurements.shape[0] == 64 and not detectors.any() and not observable.any()

    result = subprocess.run([str(args.build / "cmake/out/stim"), "detect",
        f"--in={circuit_path}", "--shots=16", "--out_format=01"],
        capture_output=True, text=True, check=True, timeout=60)
    lines = result.stdout.splitlines()
    cli_circuit = stim.Circuit.from_file(circuit_path)
    assert len(lines) == 16 and all(len(line) == cli_circuit.num_detectors
        and set(line) <= {"0", "1"} for line in lines)
    (args.output / "stim_cli_samples.01").write_text(result.stdout)
    summary = dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
        stim_commit=(args.build / "stim_revision.txt").read_text().strip(),
        compiler=(args.build / "compiler_version.txt").read_text().splitlines()[0],
        sampler_sha256=digest(executable), python_stim_version=stim.__version__,
        python_stim_path=stim.__file__, cases=records,
        noiseless_checks=dict(bases=["x", "z"], shots_per_basis=64, detectors_and_observables_zero=True),
        cli_detection_samples=16, command=[sys.executable, *sys.argv],
        source_sha256={name: digest(reference / name) for name in
            ("main.cpp", "simulation.cpp", "simulation.hpp", "run.py")},
        limitations=["Smoke checks only; no statistical equivalence or performance claim.",
            "The original reference circuit and sampler logic are unchanged."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="passed", cases=len(records), output=str(args.output))))


if __name__ == "__main__":
    main()
