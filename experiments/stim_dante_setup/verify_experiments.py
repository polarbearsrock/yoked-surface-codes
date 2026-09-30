"""Check a newly built Stim against the frozen seven-decoder study."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pymatching
import stim

WORKSPACE = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_probe(path):
    circuit = stim.Circuit.from_file(path)
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    detectors, truth = circuit.compile_detector_sampler(seed=29).sample(
        shots=4, separate_observables=True, bit_packed=True)
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    predictions = matching.decode_batch(detectors, enable_correlations=True,
        bit_packed_shots=True, bit_packed_predictions=True)
    assert predictions.shape == truth.shape
    module = importlib.import_module(stim.Circuit.__module__)
    return dict(version=stim.__version__, path=stim.__file__, native_sha256=digest(module.__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = WORKSPACE/'results/bp_uf_single_patch_d9_d11_d13_p003_20260929T062831Z'
    d7 = WORKSPACE/'results/bp_uf_single_patch_d7_p003_20260929T054325Z'
    sys.path.insert(0, str(reference/'source_snapshot/experiments/bp_uf_single_patch_sweep'))
    import run as study
    configuration = json.loads((reference/'configuration.json').read_text())
    library = Path(configuration['compile_command'][-1])
    assert digest(library) == configuration['native_library_sha256']
    checks = dict(d7=study.verify_reference(d7, library))
    print('d=7: all seven decoders match saved pilot and confirmation rows.', flush=True)
    for distance in (9, 11, 13):
        circuit, dem, native = study.model(distance, library)
        retained = reference/f'd{distance}'
        assert str(circuit) == (retained/'circuit.stim').read_text()
        assert str(dem) == (retained/'model.dem').read_text()
        sample = json.loads((retained/'pilot/sample.json').read_text())
        assert digest(sample['sample_path']) == sample['raw_npz_sha256']
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        try:
            with np.load(sample['sample_path']) as raw, np.load(retained/'pilot/results.npz') as saved:
                rows = np.unique(np.linspace(0, len(raw['detectors'])-1, 32, dtype=int))
                packed = raw['detectors'][rows]
                output, _ = native.decode(packed, 32)
                np.testing.assert_array_equal(output[:, :, study.DETERMINISTIC],
                    saved['fields'][rows][:, :, study.DETERMINISTIC])
                prediction = matching.decode_batch(packed, enable_correlations=True,
                    bit_packed_shots=True, bit_packed_predictions=True)[:, 0]
                np.testing.assert_array_equal(prediction, saved['predictions'][rows, 6])
            # Exercise fresh sampling and verify each native physical correction.
            packed, truth = circuit.compile_detector_sampler(seed=distance).sample(
                shots=16, separate_observables=True, bit_packed=True)
            output, flags = native.decode(packed, 32, audit=True)
            syndromes = np.unpackbits(packed, axis=1, count=circuit.num_detectors, bitorder='little')
            for syndrome, values, selected in zip(syndromes, output, flags):
                for decoder in range(6):
                    residual, observable = syndrome.copy(), 0
                    for edge in np.flatnonzero(selected & (1 << decoder)):
                        u, v, _, label = native.graph.edges[edge]
                        residual[u] ^= 1
                        if v is not None:
                            residual[v] ^= 1
                        observable ^= label
                    assert not residual.any() and observable == int(values[decoder, 0])
            prediction = matching.decode_batch(packed, enable_correlations=True,
                bit_packed_shots=True, bit_packed_predictions=True)
            assert prediction.shape == truth.shape == (16, 1)
            checks[f'd{distance}'] = dict(circuit_and_dem_identical=True,
                saved_rows=rows.tolist(), all_seven_predictions_identical=True,
                all_native_deterministic_fields_identical=True, fresh_shots=16,
                physical_corrections_verified=96)
            print(f'd={distance}: saved rows and fresh-sample correction checks passed.', flush=True)
        finally:
            native.close()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        workers = list(pool.map(worker_probe, [str(reference/'d9/circuit.stim')]*2))
    native_module = importlib.import_module(stim.Circuit.__module__)
    expected = dict(version=stim.__version__, path=stim.__file__, native_sha256=digest(native_module.__file__))
    assert all(worker == expected for worker in workers)
    summary = dict(status='passed', stim=expected, python=sys.version,
        checks=checks, spawned_worker_checks=workers, native_decoder_sha256=digest(library),
        command=[sys.executable, *sys.argv],
        limitations=['Small compatibility checks, not a new logical-error-rate study.',
            'New sample streams need not match older Stim versions.'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(status='passed', output=str(args.output))))


if __name__ == '__main__':
    main()
