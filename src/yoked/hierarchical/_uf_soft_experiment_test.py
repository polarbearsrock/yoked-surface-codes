"""Saved-shot identity gates and an end-to-end UF-confidence experiment."""
import numpy as np
import pytest

from yoked.hierarchical._collect import CircuitParameters
from yoked.hierarchical._provenance import read_json
from yoked.hierarchical._record import load_record, by_sector
from yoked.hierarchical._stages import CollectRequest, stage_collect
from yoked.hierarchical._uf_soft_analysis import analyze, compare_predictions
from yoked.hierarchical._uf_soft_collect import collect_features, load_features


def test_saved_shot_pipeline_and_integrity(tmp_path):
    parameters = CircuitParameters(distance=3, rounds=12, p=.01)
    for seed, role in enumerate(('calibration', 'evaluation'), start=7):
        stage_collect(CollectRequest(out_dir=tmp_path / role, role=role,
                                    parameters=parameters, seed=seed, shots=8, chunk_size=4))
        features = tmp_path / f'{role}_features'
        collect_features(tmp_path / role, features, chunk_size=4)
        # A completed rerun must reuse every verified checkpoint.
        collect_features(tmp_path / role, features, chunk_size=4)
        with pytest.raises(ValueError, match='request changed'):
            collect_features(tmp_path / role, features, chunk_size=4, gap_cap=1.)
    result = analyze(tmp_path / 'calibration', tmp_path / 'evaluation',
                     tmp_path / 'calibration_features', tmp_path / 'evaluation_features', tmp_path / 'analysis')
    assert result['work']['l1_matching_solves'] == 0
    assert result['evaluation_shots'] == 8
    assert 'correlated_uf_gap_4bit' in result['methods']
    evaluation = load_record(tmp_path / 'evaluation')
    with np.load(tmp_path / 'analysis' / 'predictions.npz') as saved:
        for prediction in saved.values():
            np.testing.assert_array_equal(by_sector(prediction).sum(axis=2) % 2, evaluation.record.yoke)
    with pytest.raises(ValueError, match='roles'):
        analyze(tmp_path / 'calibration', tmp_path / 'calibration',
                tmp_path / 'calibration_features', tmp_path / 'calibration_features', tmp_path / 'bad')
    with pytest.raises(ValueError, match='another record'):
        load_features(tmp_path / 'calibration_features', evaluation)
    with (tmp_path / 'evaluation_features' / 'features.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_features(tmp_path / 'evaluation_features', evaluation)
    assert read_json(tmp_path / 'analysis' / 'manifest.json')['schema'] == 'UFSoftAnalysis/1'


def test_paired_bootstrap_preserves_identical_decoder_outcomes():
    actual = np.zeros((10, 2), dtype=bool)
    predicted = actual.copy()
    predicted[:4, 0] = True
    result = compare_predictions({'joint_correlated_uf': predicted, 'same': predicted.copy()},
                                 actual, pieces=72, replicates=100)
    assert result['same']['paired_ratio_ci95'] == [1., 1.]
    assert result['same']['paired_block_difference_ci95'] == [0., 0.]
