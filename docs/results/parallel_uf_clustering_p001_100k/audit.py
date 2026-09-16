"""Independently recheck every observed failure in the low-noise experiment."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pymatching
import stim

import collect
from experiment import build, masks, unpack_results, write_json
from verify import check_case
from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.work_dir.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve()):
        parser.error('The work directory must be under TMPDIR')
    library, command = build(args.work_dir/'build')
    checks = {}
    for d in (7, 9, 11, 13):
        root = collect.HERE/f'd{d}'
        request = json.loads((root/'request.json').read_text())
        saved = json.loads((root/'summary.json').read_text())
        assert sha256_file(root/'results.npz') == saved['validation']['results_sha256']
        with np.load(root/'results.npz') as file:
            arrays = dict(file)
        rows = np.flatnonzero(np.any(arrays['predictions'] != arrays['actual'][:, None], axis=1)
                              | (arrays['matching'] != arrays['actual']))
        sample = SampleSet.load(request['sample']['directory'])
        assert sample.payload_sha256 == request['sample']['payload_sha256']
        if len(rows):
            dem = stim.DetectorErrorModel(sample.dem_text)
            graph = DecodingGraph.from_dem(dem)
            rules = correlation_rules_from_dem(graph, dem)
            syndromes = np.unpackbits(sample.detectors_packed[rows], axis=1,
                                      count=sample.num_detectors, bitorder='little')
            checked = np.array([check_case(library, graph, rules, syndrome, scan_normalized=True)
                                for syndrome in syndromes])
            for name, value in unpack_results(checked).items():
                np.testing.assert_array_equal(value, arrays[name][rows])
            matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            np.testing.assert_array_equal(masks(matching.decode_batch(syndromes, enable_correlations=True)),
                                          arrays['matching'][rows])
        checks[d] = dict(rows=rows.tolist(), shots=len(rows), results_sha256=sha256_file(root/'results.npz'),
                         payload_sha256=sample.payload_sha256, passed=True)
        print(f'd={d}: all {len(rows)} observed failure rows passed independent checks', flush=True)
    write_json(collect.HERE/'failure_audit.json', dict(
        distances=checks, compile_command=command, sources={**collect.sources(),
        str(Path(__file__).relative_to(collect.REPO)): sha256_file(Path(__file__))},
        selection='All rows where any of the seven decoders failed; validation only, after decoding. No decoder parameters or predictions are changed.'))


if __name__ == '__main__':
    main()
