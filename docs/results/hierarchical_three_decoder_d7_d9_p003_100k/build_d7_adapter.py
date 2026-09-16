#!/usr/bin/env python3
"""Verify and adapt the saved legacy d=7 run to the current recorded-run layout."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from yoked.hierarchical._baselines import packed_prediction_sha256
from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._provenance import packed_sample_hash, sha256_file

REPO = Path('/data2/s2chitni/yoked-surface-codes')
SAMPLE = REPO / 'out/union_find_correlated_comparison_d7_p003_100k_seed42'
FOUR = Path('/data2/s2chitni/.tmp/ysc-four-decoders-d7-p003-100k-46deb37u')
OUT = Path('/data2/s2chitni/.tmp/ysc-four-decoders-d7-p003-100k-adapter')
STEMS = ('mwpm', 'uf', 'correlated_mwpm', 'correlated_uf')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def raw_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    sample_manifest = read(SAMPLE / 'manifest.json')
    four_manifest = read(FOUR / 'manifest.json')
    results = read(FOUR / 'results.json')
    parameters = dict(sample_manifest['parameters'])
    parameters['style'] = parameters.pop('gates')

    # Verify the original saved inputs under both the legacy payload convention and
    # their declared file hashes before copying a byte.
    assert sha256_file(SAMPLE / 'circuit.stim') == sample_manifest['input_sha256']['circuit']
    assert sha256_file(SAMPLE / 'model.dem') == sample_manifest['input_sha256']['dem']
    detectors = np.load(SAMPLE / 'detectors_packed.npy', mmap_mode='r', allow_pickle=False)
    actual = np.load(SAMPLE / 'actual_observables_packed.npy', mmap_mode='r', allow_pickle=False)
    assert raw_sha(detectors) == sample_manifest['input_sha256']['packed_detectors']
    assert raw_sha(actual) == sample_manifest['input_sha256']['packed_actual_observables']
    payload = packed_sample_hash(detectors, actual)
    assert payload == sample_manifest['input_sha256']['packed_detectors_then_actual_observables']
    for filename in ('circuit.stim', 'model.dem', 'detectors_packed.npy',
                     'actual_observables_packed.npy'):
        assert sha256_file(SAMPLE / filename) == sample_manifest['files'][filename]['sha256']

    # The four-decoder bundle declares that it reused this exact sample and provenance.
    assert four_manifest['parameters'] == sample_manifest['parameters']
    assert four_manifest['input_sha256'] == sample_manifest['input_sha256']
    assert four_manifest['versions'] == sample_manifest['versions']
    for name, digest in sample_manifest['source_sha256'].items():
        assert four_manifest['source_sha256'][name] == digest
    assert results['parameters'] == four_manifest['parameters']
    assert results['validation']['baseline_versions_and_sources_match'] is True
    assert sha256_file(FOUR / 'predictions.npz')

    with np.load(FOUR / 'predictions.npz', allow_pickle=False) as archive:
        predictions = {stem: np.asarray(archive[stem]) for stem in STEMS}
    for stem, prediction in predictions.items():
        assert prediction.shape == (parameters['shots'], 2 * parameters['patches'])
        assert prediction.dtype == np.bool_
        assert packed_prediction_sha256(prediction) == \
            results['decoders'][stem]['prediction_packed_sha256']
    # The separately checkpointed correlated-UF output is another byte-for-byte witness.
    np.testing.assert_array_equal(
        predictions['correlated_uf'],
        np.load(FOUR / 'correlated_uf_predictions.npy', mmap_mode='r', allow_pickle=False))

    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ('circuit.stim', 'model.dem', 'detectors_packed.npy',
                     'actual_observables_packed.npy'):
        shutil.copy2(SAMPLE / filename, OUT / filename)
    for stem, prediction in predictions.items():
        np.save(OUT / f'{stem}_predictions.npy', prediction, allow_pickle=False)
    shutil.copy2(FOUR / 'results.json', OUT / 'results.json')

    manifest = {
        'parameters': parameters,
        'normalization': four_manifest['normalization'],
        'paired_samples': True,
        'input_sha256': {
            'circuit.stim': sha256_file(OUT / 'circuit.stim'),
            'model.dem': sha256_file(OUT / 'model.dem'),
            'packed_detectors_payload': raw_sha(detectors),
            'packed_observables_payload': raw_sha(actual),
            'packed_detectors_then_observables_payload': payload,
        },
        'versions': four_manifest['versions'],
        'source_sha256': four_manifest['source_sha256'],
        # The correlated-UF legacy run did not record a single Git commit. Preserve
        # that fact rather than misattributing its predictions to the baseline commit.
        'code_commit': None,
        'adapter': {
            'kind': 'verified-layout-adapter; no resampling or decoding',
            'source_sample_directory': str(SAMPLE),
            'source_four_decoder_directory': str(FOUR),
            'source_sample_manifest_sha256': sha256_file(SAMPLE / 'manifest.json'),
            'source_four_decoder_manifest_sha256': sha256_file(FOUR / 'manifest.json'),
            'source_results_sha256': sha256_file(FOUR / 'results.json'),
            'source_predictions_npz_sha256': sha256_file(FOUR / 'predictions.npz'),
            'baseline_code_commit': sample_manifest['code_commit'],
            'sampling_code_commit': sample_manifest['sampling_code_commit'],
            'legacy_source_sha256': sample_manifest['source_sha256'],
            'four_decoder_source_sha256': four_manifest['source_sha256'],
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n',
                                       encoding='utf-8')

    # Exercise both current public gates over the finished adapter.
    sample = SampleSet.load_recorded_run(OUT)
    from yoked.hierarchical._baselines import load_recorded_baselines
    baselines = load_recorded_baselines(OUT)
    print(json.dumps({
        'adapter': str(OUT),
        'manifest_sha256': sha256_file(OUT / 'manifest.json'),
        'results_sha256': sha256_file(OUT / 'results.json'),
        'payload_sha256': sample.payload_sha256,
        'sample_identities': dict(sample.identities),
        'prediction_sha256': dict(baselines.prediction_sha256),
        'verified_baselines': list(baselines.predictions),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
