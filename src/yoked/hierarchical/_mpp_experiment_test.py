"""Paired metrics and experiment publication, including the fixed-reference gate."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.hierarchical._collect import CircuitParameters, SampleSet
from yoked.hierarchical._mpp import MppHierarchicalDecoder
from yoked.hierarchical._mpp_experiment import compare_accuracy, stage_mpp_compare
from yoked.hierarchical._provenance import REPOSITORY_ROOT, sha256_file


def test_paired_counts_intervals_and_undefined_misattribution():
    truth = np.zeros((4, 4), dtype=bool)
    gap, mpp = truth.copy(), truth.copy()
    gap[[0, 2], 0] = True
    mpp[[1, 2], 0] = True
    result = compare_accuracy(truth, truth, gap, mpp, replicates=1000)
    assert result['complementary_gap']['block_failures'] == 2
    assert result['mpp']['block_failures'] == 2
    paired = result['paired']
    assert paired['mpp_repairs'] == paired['mpp_regressions'] == 1
    assert paired['both_fail'] == paired['both_succeed'] == 1
    assert paired['block_failure_difference'] == 0
    assert paired['pooled_misattribution_difference'] is None
    assert paired['pooled_misattribution_difference_ci95'] is None
    assert paired['zero_eligible_bootstrap_replicates'] == 1000
    assert result == compare_accuracy(truth, truth, gap, mpp, replicates=1000)


def test_pooled_misattribution_counts_both_sectors_per_shot():
    truth = np.zeros((2, 4), dtype=bool)
    reference = truth.copy()
    reference[0, :2] = True  # two eligible sectors in this shot
    reference[1, 0] = True  # one in this shot
    gap, mpp = truth.copy(), reference.copy()
    result = compare_accuracy(reference, truth, gap, mpp, replicates=50)
    assert result['mpp']['single_error_misattribution']['pooled'] == dict(count=3, total=3, value=1.)
    assert result['paired']['pooled_misattribution_difference'] == 1
    assert result['paired']['pooled_misattribution_difference_ci95'] == [1., 1.]


native = pytest.mark.skipif(not os.environ.get('YOKED_MPP_BUILD'), reason='optional MPP build not selected')


@native
def test_saved_sample_cli_and_hashed_artifacts(tmp_path):
    sample = SampleSet.sample(CircuitParameters(distance=3, rounds=12, p=0.003), shots=8, seed=993)
    sample.save(tmp_path / 'sample')
    output = tmp_path / 'comparison'
    subprocess.run([sys.executable, str(REPOSITORY_ROOT / 'tools/mpp_experiment'),
                    '--sample', str(tmp_path / 'sample'), '--out', str(output),
                    '--method', 'incremental', '--verify-scores', '--rows', '1:7',
                    '--batch-size', '2', '--bootstrap-replicates', '50'], check=True)
    manifest = json.loads((output / 'manifest.json').read_text())
    result = json.loads((output / 'results.json').read_text())
    assert result['shots'] == 6
    assert result['reference_disagreements'] == result['native_weight_disagreements'] == 0
    assert manifest['sample']['identities'] == dict(sample.identities)
    assert manifest['native_build']['dependencies']['pymatching'] == '6f63b2b9474ba0fa7e511fe52bffdce858a06984'
    for name, digest in manifest['artifacts'].items():
        assert sha256_file(output / name) == digest
    with np.load(output / 'predictions.npz') as data:
        np.testing.assert_array_equal(data['row_ids'], np.arange(1, 7))
        np.testing.assert_array_equal(data['mpp_prediction'], data['reference'] ^ data['mpp_residual'])
    with pytest.raises(FileExistsError):
        stage_mpp_compare(sample, output)


@native
def test_reference_disagreement_prevents_publication(tmp_path, monkeypatch):
    sample = SampleSet.sample(CircuitParameters(distance=3, rounds=12, p=0.003), shots=2, seed=17)
    decode = MppHierarchicalDecoder.decode_with_scores_batch

    def wrong_reference(self, detectors):
        result = decode(self, detectors)
        arrays = result.arrays()
        arrays['reference'] = ~result.reference
        return SimpleNamespace(**arrays)

    monkeypatch.setattr(MppHierarchicalDecoder, 'decode_with_scores_batch', wrong_reference)
    with pytest.raises(ValueError, match='reference predictions'):
        stage_mpp_compare(sample, tmp_path / 'rejected', replicates=10)
    assert not (tmp_path / 'rejected/manifest.json').exists()
