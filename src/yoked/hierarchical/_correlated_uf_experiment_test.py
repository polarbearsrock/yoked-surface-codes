"""Paired statistics, artifact validation, and CLI checks without native MPP."""
import json
import subprocess
import sys

import numpy as np
import pytest

from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._correlated_uf_experiment import load_baseline, stage_correlated_uf, summarize_accuracy
from yoked.hierarchical._provenance import REPOSITORY_ROOT, sha256_file


def test_paired_metrics_allow_different_l1_references():
    truth = np.zeros((4, 4), dtype=bool)
    candidate, baseline = truth.copy(), truth.copy()
    candidate[[1, 2], 0] = True
    baseline[[0, 2], 0] = True
    arrays = {'actual': truth, 'reference': candidate, 'prediction': candidate,
              'mwpm_reference': baseline, 'gap_prediction': baseline, 'mpp_prediction': baseline,
              'outer_tied': np.zeros((4, 2), dtype=bool), 'dijkstra_states': np.ones((4, 4), dtype=int)}
    result = summarize_accuracy(arrays, replicates=100)
    pair = result['paired_against']['correlated_mwpm_complementary_gap']
    assert pair['candidate_repairs'] == pair['candidate_regressions'] == 1
    assert pair['both_fail'] == pair['both_succeed'] == 1
    assert pair['candidate_to_baseline_ratio'] == 1
    assert pair['l1_reference_bit_disagreements'] == 2
    arrays['gap_prediction'] = truth
    assert summarize_accuracy(arrays, replicates=100)['paired_against'][
        'correlated_mwpm_complementary_gap']['candidate_to_baseline_ratio'] is None


def _baseline_fixture(sample, directory):
    """Make a tiny format fixture; its predictions are only for I/O tests."""
    sample.save(directory / 'sample')
    _, actual = sample.rows(np.arange(sample.shots))
    np.savez_compressed(directory / 'predictions.npz', row_ids=np.arange(sample.shots),
                        actual=actual, reference=actual, gap_prediction=actual, mpp_prediction=actual)
    (directory / 'results.json').write_text(json.dumps({'parameters': sample.parameters.to_json()}))
    manifest = {'schema': 'mpp-sharded-comparison/1', 'sample_identities': dict(sample.identities),
                'provenance': {'fixture': True},
                'artifacts': {name: sha256_file(directory / name)
                              for name in ('results.json', 'predictions.npz')}}
    (directory / 'manifest.json').write_text(json.dumps(manifest))


def test_paired_cli_and_verified_saved_artifacts(tmp_path):
    sample = SampleSet.sample(CircuitParameters(distance=3, rounds=12, p=0.003), shots=6, seed=993)
    baseline = tmp_path / 'baseline'
    _baseline_fixture(sample, baseline)
    output = tmp_path / 'candidate'
    subprocess.run([sys.executable, str(REPOSITORY_ROOT / 'tools/correlated_uf_experiment'),
                    '--baseline-run', str(baseline), '--out', str(output), '--rows', '1:5',
                    '--batch-size', '2', '--bootstrap-replicates', '100'], check=True)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['sample']['identities'] == dict(sample.identities)
    assert manifest['baseline']['manifest_sha256'] == sha256_file(baseline / 'manifest.json')
    for name, expected in manifest['artifacts'].items():
        assert sha256_file(output / name) == expected
    with np.load(output / 'predictions.npz') as arrays:
        np.testing.assert_array_equal(arrays['row_ids'], np.arange(1, 5))
        np.testing.assert_array_equal(arrays['prediction'], arrays['reference'] ^ arrays['residual'])
        recomputed = summarize_accuracy(arrays, replicates=100)
    stored = json.loads((output / 'results.json').read_text())
    assert stored['configurations'] == recomputed['configurations']
    assert stored['paired_against'] == recomputed['paired_against']
    with pytest.raises(FileExistsError):
        stage_correlated_uf(sample, output)


def test_baseline_hash_and_sample_identity_must_match(tmp_path):
    sample = SampleSet.sample(CircuitParameters(distance=3, rounds=12, p=0.003), shots=4, seed=19)
    baseline = tmp_path / 'baseline'
    _baseline_fixture(sample, baseline)
    other = SampleSet.sample(sample.parameters, shots=4, seed=20)
    with pytest.raises(ValueError, match='identical saved sample'):
        load_baseline(other, baseline)
    with (baseline / 'predictions.npz').open('ab') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_baseline(sample, baseline)
