"""Experimental DEM soft evidence; production UF is unchanged."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN = HERE.parent / 'parallel_uf_clustering_d7_d9_p003_100k'
VARIANTS = ('plain_uf', 'correlated_uf', 'prior_projection_uf', 'bp5_uf', 'bp20_uf')
FIELDS = ('prediction', 'cost', 'selected_edges', 'forest_edges', 'max_cluster',
          'epochs', 'edge_evaluations', 'frontier_visits', 'zero_weight_edges',
          'evidence_seconds', 'uf_seconds')
ITERATIONS = (5, 20)
DAMPING = 0.5
LLR_LIMIT = 30.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


frozen = load_module('uf_evidence_frozen', FROZEN / 'experiment.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [*HERE.glob('*.py'), HERE/'kernel.cc', HERE/'README.md',
             FROZEN/'kernel.cc', FROZEN/'experiment.py']
    paths += [p for p in (REPO/'src').rglob('*.py') if not p.name.endswith('_test.py')]
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def xor_probability(a, b):
    return a*(1-b) + b*(1-a)


@dataclass
class FaultModel:
    probabilities: np.ndarray
    fault_offsets: np.ndarray
    detectors: np.ndarray
    edge_offsets: np.ndarray
    edges: np.ndarray
    nd: int
    ne: int
    raw_mechanisms: int

    @property
    def nf(self):
        return len(self.probabilities)

    @property
    def incidence_faults(self):
        return np.repeat(np.arange(self.nf), np.diff(self.fault_offsets))

    @classmethod
    def from_dem(cls, dem, graph):
        edge_ids = {(u, None) if v is None else tuple(sorted((u, v))): e
                    for e, (u, v, _, _) in enumerate(graph.edges)}
        mechanisms = {}
        raw = 0
        for instruction in dem.flattened():
            if instruction.type != 'error':
                continue
            probability, = instruction.args_copy()
            if not 0 < probability < 0.5:
                if probability == 0:
                    continue
                raise ValueError('This experiment requires mechanism priors in (0, 0.5)')
            raw += 1
            components = [[]]
            detector_set = set()
            mask = 0
            for target in instruction.targets_copy():
                if target.is_separator():
                    components.append([])
                elif target.is_relative_detector_id():
                    components[-1].append(target.val)
                    detector_set.symmetric_difference_update([target.val])
                elif target.is_logical_observable_id():
                    mask ^= 1 << target.val
                else:
                    raise ValueError('Unexpected DEM target')
            component_edges = []
            for detectors in components:
                if not detectors:
                    continue
                if len(detectors) not in (1, 2):
                    raise ValueError('Expected graphlike components of the decomposed DEM')
                key = (detectors[0], None) if len(detectors) == 1 else tuple(sorted(detectors))
                component_edges.append(edge_ids[key])
            if len(set(component_edges)) != len(component_edges):
                raise ValueError('Repeated component edge in a mechanism')
            reconstructed_detectors, reconstructed_mask = set(), 0
            for e in component_edges:
                u, v, _, label = graph.edges[e]
                reconstructed_detectors.symmetric_difference_update([u])
                if v is not None:
                    reconstructed_detectors.symmetric_difference_update([v])
                reconstructed_mask ^= label
            if reconstructed_detectors != detector_set or reconstructed_mask != mask:
                raise ValueError('Fault incidence or logical label disagrees with graph decomposition')
            if not detector_set or not component_edges:
                raise ValueError('Detectorless mechanisms are outside this experiment')
            key = (tuple(sorted(detector_set)), tuple(sorted(component_edges)))
            mechanisms[key] = xor_probability(mechanisms.get(key, 0.), probability)
        probabilities, detectors, edges = [], [], []
        detector_offsets, edge_offsets = [0], [0]
        for (ds, es), probability in sorted(mechanisms.items()):
            probabilities.append(probability)
            detectors.extend(ds)
            edges.extend(es)
            detector_offsets.append(len(detectors))
            edge_offsets.append(len(edges))
        model = cls(np.array(probabilities, dtype=np.float64),
                    np.array(detector_offsets, dtype=np.int32), np.array(detectors, dtype=np.int32),
                    np.array(edge_offsets, dtype=np.int32), np.array(edges, dtype=np.int32),
                    graph.num_detectors, len(graph.edges), raw)
        # This is a prior identity, not an independence claim about BP posteriors.
        marginal = np.zeros(model.ne)
        for f, probability in enumerate(model.probabilities):
            es = model.edges[model.edge_offsets[f]:model.edge_offsets[f+1]]
            marginal[es] = xor_probability(marginal[es], probability)
        expected = 1 / (1 + np.exp([edge[2] for edge in graph.edges]))
        np.testing.assert_allclose(marginal, expected, rtol=2e-11, atol=1e-14)
        return model

    def project(self, posterior):
        component_probabilities = np.repeat(posterior, np.diff(self.edge_offsets))
        support = np.bincount(self.edges, weights=component_probabilities, minlength=self.ne)
        return -np.log(np.clip(support, 1e-15, 1.))

    def describe(self):
        check_degrees = np.bincount(self.detectors, minlength=self.nd)
        return dict(detectors=self.nd, graph_edges=self.ne, raw_dem_mechanisms=self.raw_mechanisms,
                    grouped_mechanisms=self.nf, incidences=len(self.detectors),
                    graph_component_incidences=len(self.edges),
                    maximum_check_degree=int(check_degrees.max()),
                    directed_messages_per_iteration=2*len(self.detectors),
                    prior_marginal_identity_verified=True)


def python_bp(model, syndrome, iterations, damping=DAMPING):
    """Independent log-product check update, distinct from native prefix products."""
    if iterations < 0 or not 0 < damping <= 1:
        raise ValueError('Invalid BP parameters')
    syndrome = np.asarray(syndrome)
    if syndrome.shape != (model.nd,) or not np.isin(syndrome, (0, 1)).all():
        raise ValueError('Invalid syndrome')
    prior = np.log1p(-model.probabilities) - np.log(model.probabilities)
    fault = model.incidence_faults
    detector = model.detectors
    q = np.clip(prior[fault], -LLR_LIMIT, LLR_LIMIT)
    r = np.zeros(len(q))
    total = prior.copy()
    cap = np.tanh(LLR_LIMIT/2)
    for _ in range(iterations):
        t = np.tanh(q/2)
        zeros = t == 0
        logs = np.log(np.where(zeros, 1., np.abs(t)))
        log_products = np.bincount(detector, weights=logs, minlength=model.nd)
        zero_counts = np.bincount(detector, weights=zeros, minlength=model.nd)
        negative_counts = np.bincount(detector, weights=t < 0, minlength=model.nd)
        magnitude = np.exp(np.minimum(0., log_products[detector] - logs))
        magnitude[zero_counts[detector] - zeros > 0] = 0
        sign = 1 - 2*((negative_counts[detector] - (t < 0) + syndrome[detector]) % 2)
        product = np.clip(sign*magnitude, -cap, cap)
        incoming = np.log1p(product) - np.log1p(-product)
        r = (1-damping)*r + damping*incoming
        total = prior + np.bincount(fault, weights=r, minlength=model.nf)
        q = np.clip(total[fault] - r, -LLR_LIMIT, LLR_LIMIT)
    posterior = 1/(1+np.exp(np.clip(total, -LLR_LIMIT, LLR_LIMIT)))
    return posterior, model.project(posterior)


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'evidence.so'
    command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp', '-ffp-contract=off',
               str(HERE/'kernel.cc'), '-o', str(target)]
    subprocess.run(command, check=True)
    return target, command


def ptr(array):
    return ctypes.c_void_p(array.ctypes.data)


class Native:
    def __init__(self, library, graph, rules, model):
        self.graph, self.model = graph, model
        if model.nd != graph.num_detectors or model.ne != len(graph.edges):
            raise ValueError('Model dimensions disagree with graph')
        self.base = frozen.Native(library, graph, rules)
        self.lib = self.base.lib
        self.lib.evidence_create.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p]*5
        self.lib.evidence_create.restype = ctypes.c_void_p
        self.lib.evidence_destroy.argtypes = [ctypes.c_void_p]
        self.lib.evidence_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.evidence_decode.restype = ctypes.c_int
        self.lib.evidence_weights.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.evidence_weights.restype = ctypes.c_int
        self.handle = self.lib.evidence_create(self.base.handle, model.nf, ptr(model.probabilities),
                                                ptr(model.fault_offsets), ptr(model.detectors),
                                                ptr(model.edge_offsets), ptr(model.edges))
        if not self.handle:
            self.base.close()
            raise RuntimeError('Could not initialize evidence model')

    def close(self):
        if self.handle:
            self.lib.evidence_destroy(self.handle)
            self.handle = None
        self.base.close()

    def decode(self, packed, threads=1, audit=False):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if threads < 1 or packed.ndim != 2 or packed.shape[1] != (self.graph.num_detectors+7)//8:
            raise ValueError('Invalid threads or packed syndrome shape')
        out = np.empty((len(packed), len(VARIANTS), len(FIELDS)), dtype=np.float64)
        flags = np.zeros((len(packed), len(self.graph.edges)), dtype=np.uint16) if audit else None
        failures = self.lib.evidence_decode(self.handle, len(packed), packed.shape[1], ptr(packed),
                                            ptr(out), threads, ptr(flags) if audit else None)
        if failures or not np.isfinite(out).all():
            raise RuntimeError(f'{failures} evidence-decoder shots failed validation')
        return out, flags

    def weights(self, syndrome, iterations, damping=DAMPING):
        syndrome = np.asarray(syndrome)
        if syndrome.shape != (self.model.nd,) or not np.isin(syndrome, (0, 1)).all():
            raise ValueError('Invalid syndrome')
        if iterations < 0 or not 0 < damping <= 1:
            raise ValueError('Invalid BP parameters')
        packed = np.packbits(syndrome.astype(np.uint8), bitorder='little')
        posterior, weights = np.empty(self.model.nf), np.empty(self.model.ne)
        failure = self.lib.evidence_weights(self.handle, ptr(packed), iterations, damping,
                                            ptr(posterior), ptr(weights))
        if failure or not np.isfinite(weights).all() or not np.isfinite(posterior).all():
            raise RuntimeError('BP weights failed validation')
        return posterior, weights


def setup(dem, library):
    graph = DecodingGraph.from_dem(dem)
    model = FaultModel.from_dem(dem, graph)
    return graph, model, Native(library, graph, correlation_rules_from_dem(graph, dem), model)
