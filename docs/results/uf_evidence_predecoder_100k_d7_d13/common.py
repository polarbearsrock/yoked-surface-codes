"""Reuse the frozen pilot's model, BP, graph, and sample provenance."""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent/'uf_evidence_predecoder_d7_d13'
sys.path.insert(0, str(PILOT))
import evidence as pilot
import verify as pilot_verify

pilot_run = pilot.load_module('uf_evidence_pilot_run', PILOT/'run.py')
VARIANTS = ('bp5_uf', 'bp5_correlated_uf')
FIELDS = pilot.FIELDS
sha256, write_json = pilot.sha256, pilot.write_json


def source_hashes():
    hashes = pilot.source_hashes()
    for name in ('common.py', 'kernel.cc', 'run.py', 'check.py', 'README.md'):
        p = HERE/name
        hashes[str(p.relative_to(pilot.REPO))] = sha256(p)
    hashes[str(pilot_run.CERTIFICATE_SOURCE.relative_to(pilot.REPO))] = sha256(pilot_run.CERTIFICATE_SOURCE)
    return hashes


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    library = directory/'full_evidence.so'
    command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp',
               '-ffp-contract=off', str(HERE/'kernel.cc'), '-o', str(library)]
    subprocess.run(command, check=True)
    return library, command


class Native(pilot.Native):
    def __init__(self, library, graph, rules, model):
        super().__init__(library, graph, rules, model)
        self.lib.full_evidence_decode.argtypes = self.lib.evidence_decode.argtypes
        self.lib.full_evidence_decode.restype = ctypes.c_int

    def decode(self, packed, threads=1, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if threads < 1 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid packed syndromes or thread count')
        out = np.empty((len(packed), len(VARIANTS), len(FIELDS)), dtype=np.float64)
        flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint16) if audit else None
        failures = self.lib.full_evidence_decode(self.handle, len(packed), packed.shape[1],
            pilot.ptr(packed), pilot.ptr(out), threads, pilot.ptr(flags) if audit else None)
        if failures or not np.isfinite(out).all():
            raise RuntimeError(f'{failures} native shots failed validation')
        return out, flags


def setup(dem, library):
    graph = pilot.DecodingGraph.from_dem(dem)
    model = pilot.FaultModel.from_dem(dem, graph)
    rules = pilot.correlation_rules_from_dem(graph, dem)
    return graph, model, Native(library, graph, rules, model)
