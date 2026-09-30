"""Exercise saved samples, retained provenance, batching, and the public CLI."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from yoked.hierarchical import CircuitParameters, SampleSet, stage_paper_decode
from yoked.hierarchical._provenance import canonical_json, sha256_bytes, sha256_file


def _sample():
    return SampleSet.sample(CircuitParameters(distance=3, rounds=3, p=0.003, patches=2), seed=17, shots=6)


def test_stage_retains_native_results_and_verified_sample_provenance(tmp_path):
    sample = _sample()
    result = stage_paper_decode(sample, tmp_path / 'run', rows=[1, 3, 5], batch_size=2, gap_scale=0.9)
    manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert result['shots'] == 3 and result['gap_scale'] == 0.9
    assert manifest['sample']['identities'] == dict(sample.identities)
    assert manifest['calibration'] == 'none; direct gap weights'
    for artifact, expected in manifest['artifacts'].items():
        assert sha256_file(tmp_path / 'run' / artifact) == expected
    identity_input = {key: value for key, value in manifest.items()
                      if key not in ('identity', 'code_commit', 'created_utc', 'artifacts')}
    assert manifest['identity'] == sha256_bytes(canonical_json(identity_input).encode())
    with np.load(tmp_path / 'run/predictions.npz') as data:
        np.testing.assert_array_equal(data['row_ids'], [1, 3, 5])
        assert result['block_failures'] == np.any(data['prediction'] != data['actual'], axis=1).sum()
        np.testing.assert_array_equal(data['outer_weights'], data['complementary_gap'] * 0.9)
    saved = SampleSet.load(tmp_path / 'run/sample')
    assert saved.payload_sha256 == sample.payload_sha256
    with pytest.raises(FileExistsError, match='new path'):
        stage_paper_decode(saved, tmp_path / 'run')


def test_saved_sample_batching_and_row_subsets_are_identical(tmp_path):
    sample = SampleSet.load(_sample().save(tmp_path / 'sample'))
    stage_paper_decode(sample, tmp_path / 'full', batch_size=1)
    stage_paper_decode(sample, tmp_path / 'subset', rows=[1, 2, 4], batch_size=3)
    # Reusing an existing verified sample should not duplicate the full payload.
    assert not (tmp_path / 'subset/sample').exists()
    with np.load(tmp_path / 'full/predictions.npz') as full, np.load(tmp_path / 'subset/predictions.npz') as subset:
        for name in full.files:
            np.testing.assert_array_equal(full[name][[1, 2, 4]], subset[name])


def test_paper_command_generates_then_reuses_a_sample(tmp_path):
    tool = Path(__file__).resolve().parents[3] / 'tools/hierarchical_experiment'
    command = [sys.executable, str(tool), 'paper', '--out', str(tmp_path / 'generated'),
               '--distance', '3', '--rounds', '3', '--p', '0.003', '--patches', '2',
               '--shots', '4', '--seed', '17', '--gap-scale', '0.9']
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    assert 'native correlated L1' in completed.stdout
    subprocess.run([sys.executable, str(tool), 'paper', '--out', str(tmp_path / 'reused'),
                    '--sample', str(tmp_path / 'generated/sample'), '--rows', '1:3', '--gap-scale', '0.9'],
                   text=True, capture_output=True, check=True)
    with np.load(tmp_path / 'generated/predictions.npz') as full, np.load(tmp_path / 'reused/predictions.npz') as subset:
        np.testing.assert_array_equal(full['prediction'][1:3], subset['prediction'])
    invalid = subprocess.run([sys.executable, str(tool), 'paper', '--out', str(tmp_path / 'invalid'),
                              '--sample', str(tmp_path / 'generated/sample'), '--shots', '2'],
                             text=True, capture_output=True)
    assert invalid.returncode != 0 and 'choose exactly one' in invalid.stderr
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('rows,batch_size', [([], 2), ([2, 1], 2), ([0, 6], 2), ([0], 0)])
def test_invalid_requests_do_not_publish(tmp_path, rows, batch_size):
    with pytest.raises(ValueError):
        stage_paper_decode(_sample(), tmp_path / 'invalid', rows=rows, batch_size=batch_size)
    assert not (tmp_path / 'invalid/manifest.json').exists()
