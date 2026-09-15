"""Test support: a small six-patch yoked fixture.

Distance 3 with 4d = 12 rounds keeps end-to-end tests at a few seconds while
exercising both yokes, all six patches, and correlated mechanisms. Library
code never imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.hierarchical._patch_graphs import PatchGraphs

NUM_PATCHES = 6
"""The fixture always uses the experiment's six patches and two yokes."""


@dataclass(frozen=True)
class YokedFixture:
    """A sampled six-patch circuit and its per-patch graphs.

    Fields: ``circuit`` and ``dem`` as built by the repository; ``patches``
    from ``PatchGraphs.from_yoked_dem``; ``detectors`` of shape (shots, n_d)
    and ``actual`` of shape (shots, 12), both boolean.
    """
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    patches: PatchGraphs
    detectors: np.ndarray
    actual: np.ndarray


def yoked_fixture(*, distance: int = 3, shots: int = 64, seed: int = 7, p: float = 0.003) -> YokedFixture:
    circuit = yoked_magic_memory_circuit(
        patch_diameter=distance, rounds=4 * distance, noise=gen.NoiseModel.si1000(p),
        style='cz', yokes=2, num_patches=NUM_PATCHES,
    )
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    detectors, actual = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True, bit_packed=False,
    )
    return YokedFixture(circuit, dem, PatchGraphs.from_yoked_dem(dem, num_patches=NUM_PATCHES),
                        detectors.astype(bool), actual.astype(bool))
