#!/usr/bin/env python3
"""Audit saved UF experiment artifacts and check a few rows through both APIs.

This is validation work, outside experiment timing. It reloads verified samples,
checks saved statistics, and compares the soft-output path with the existing
hard correlated-UF API. Use the same source revision as the saved run.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(os.environ['DANTE_REPO']) / 'src'))

import numpy as np
import stim

from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._correlated_uf import CorrelatedUFHierarchicalDecoder
from yoked.hierarchical._correlated_uf_experiment import SOURCES, summarize_accuracy
from yoked.hierarchical._provenance import sha256_file, source_hashes, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--check-rows', type=int, default=2)
    args = parser.parse_args()
    if args.check_rows < 1:
        parser.error('--check-rows must be positive')
    root = args.run.resolve()
    configuration = json.loads((root / 'run.json').read_text())
    completion = json.loads((root / 'completion.json').read_text())
    assert completion['run_sha256'] == sha256_file(root / 'run.json')
    assert completion['summary_sha256'] == sha256_file(root / 'summary.json')
    assert configuration['provenance']['source_sha256'] == source_hashes(SOURCES)
    summary = json.loads((root / 'summary.json').read_text())

    # Import the run's saved recipe to use its shard verifier, not a potentially
    # newer workspace recipe. Its hash was recorded before workers launched.
    assert configuration['provenance']['recipe_sha256'] == sha256_file(root / 'recipe.py')
    spec = importlib.util.spec_from_file_location('saved_uf_recipe', root / 'recipe.py')
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    report = {}
    for distance in configuration['distances']:
        directory = root / f'd{distance}'
        assert completion['distance_manifests'][str(distance)] == sha256_file(directory / 'manifest.json')
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, expected in manifest['artifacts'].items():
            assert sha256_file(directory / name) == expected
        sample = SampleSet.load(Path(configuration['baseline_root']) / f'd{distance}' / 'sample')
        with np.load(directory / 'predictions.npz') as stored:
            arrays = {name: stored[name] for name in stored.files}
        recomputed = summarize_accuracy(arrays)
        for key, value in recomputed.items():
            assert summary[str(distance)][key] == value
        count = min(args.check_rows, len(arrays['row_ids']))
        rows = arrays['row_ids'][:count]
        detectors, actual = sample.rows(rows)
        decoder = CorrelatedUFHierarchicalDecoder(stim.DetectorErrorModel(sample.dem_text))
        decoded = decoder.decode_with_gaps_batch(detectors)
        for name, value in decoded.arrays().items():
            np.testing.assert_array_equal(value, arrays[name][:count])
        np.testing.assert_array_equal(actual, arrays['actual'][:count])
        for index, local_decoder in enumerate(decoder.decoders):
            local = decoder.patches[index].local_syndromes(detectors)
            np.testing.assert_array_equal(local_decoder.decode_batch(local),
                                          decoded.reference[:, 2*index:2*index+2])
        for relative, expected in manifest['shard_manifests'].items():
            shard = directory / relative
            assert sha256_file(shard / 'manifest.json') == expected
            start, stop = map(int, shard.name.split('-'))
            recipe.verified_shard(shard, sample, np.arange(start, stop), configuration, distance)
        report[str(distance)] = {'shots': len(arrays['row_ids']), 'rows_redecoded': count,
                                 'hard_and_soft_references_agree': True,
                                 'statistics_and_artifacts_verified': True}
        print(f'd={distance}: artifacts, statistics and {count} independent hard-API checks passed', flush=True)

    # Exercise the resume guard on a temporary copy; saved results stay intact.
    # Only one small shard is needed to verify that corruption is rejected.
    import shutil
    with tempfile.TemporaryDirectory(prefix='uf-audit-', dir=os.environ['TMPDIR']) as temporary:
        damaged = Path(temporary) / 'shard'
        shutil.copytree(shard, damaged)
        with (damaged / 'predictions.npz').open('ab') as handle:
            handle.write(b'injected corruption for validation')
        try:
            recipe.verified_shard(damaged, sample, np.arange(start, stop), configuration, distance)
        except ValueError as error:
            assert 'artifact mismatch' in str(error)
        else:
            raise AssertionError('Corrupted shard was accepted')
    write_json_atomic(root / 'audit.json', {'distances': report, 'corrupt_shard_rejected': True})


if __name__ == '__main__':
    main()
