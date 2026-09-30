"""Fixed definitions for the paired single-patch BP pre-decoder study."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
REPO = WORKSPACE / 'repos/yoked-surface-codes'
sys.path.insert(0, str(REPO / 'src'))
PILOT = REPO / 'docs/results/uf_evidence_predecoder_d7_d13'
sys.path.insert(0, str(PILOT))
import evidence
import numpy as np
import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit

VARIANTS = ('weighted_uf', 'correlated_uf', 'bp1_uf', 'bp2_uf', 'bp5_uf')
FIELDS = (*evidence.FIELDS, 'bp_iterations')
CONFIG = dict(distance=7, p=0.003, noise='si1000', rounds=28, patches=1,
              yokes=0, style='cz', ideal_time_boundaries=True,
              bp='sum_product_flooding', damping=0.5, llr_clip=30.,
              fixed_budgets=[1, 2, 5], always_finish_with_uf=True,
              projection='-log(clip(sum of supporting fault posteriors, 1e-15, 1))',
              primary_comparison='bp5_uf versus correlated_uf',
              development_shots=10000, development_seed=2026092801,
              confirmation_shots=100000, confirmation_seed=2026092802,
              validation_seed=2026092803, validation_shots=32,
              serial_timing_shots=1024, serial_timing_selection_seed=2026092804,
              parameter_tuning=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='dante-bp-json-', dir=os.environ['TMPDIR'])
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        try:
            temporary.replace(path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # Scratch and retained results can be different filesystems. Keep
            # temporary data in TMPDIR and copy to the final retained pathname.
            shutil.copyfile(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / 'single_patch.so'
    command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp',
               '-ffp-contract=off', str(HERE / 'kernel.cc'), '-o', str(library)]
    subprocess.run(command, check=True)
    return library, command


def circuit_model(library):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=7, rounds=28, noise=gen.NoiseModel.si1000(0.003),
        style='cz', yokes=0, num_patches=1)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    graph = evidence.DecodingGraph.from_dem(dem)
    model = evidence.FaultModel.from_dem(dem, graph)
    native = Native(library, graph, evidence.correlation_rules_from_dem(graph, dem), model)
    assert circuit.num_observables == 2
    return circuit, dem, graph, model, native


class Native(evidence.Native):
    def __init__(self, library, graph, rules, model):
        super().__init__(library, graph, rules, model)
        self.lib.single_patch_decode.argtypes = [ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.single_patch_decode.restype = ctypes.c_int

    def decode(self, packed, threads=1, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if not 1 <= threads <= 32 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid thread count or syndrome layout')
        out = np.zeros((len(packed), len(VARIANTS), len(FIELDS)), dtype=np.float64)
        flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint32) if audit else None
        team = ctypes.c_int()
        failures = self.lib.single_patch_decode(self.handle, len(packed), packed.shape[1],
            evidence.ptr(packed), evidence.ptr(out), threads,
            evidence.ptr(flags) if audit else None, ctypes.byref(team))
        if failures or not np.isfinite(out).all():
            raise RuntimeError(f'{failures} native decoding/physical-syndrome checks failed')
        if team.value != threads:
            raise RuntimeError(f'Requested {threads} threads; OpenMP used {team.value}')
        np.testing.assert_array_equal(out[:, :, 11], np.broadcast_to([0, 0, 1, 2, 5], out.shape[:2]))
        return out, flags


def physical_cpus(count):
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (int((root/'physical_package_id').read_text()), int((root/'core_id').read_text()))
        cores.setdefault(key, cpu)
    if len(cores) < count:
        raise RuntimeError(f'Need {count} distinct physical cores, only {len(cores)} available')
    return [cores[key] for key in sorted(cores)[:count]]
