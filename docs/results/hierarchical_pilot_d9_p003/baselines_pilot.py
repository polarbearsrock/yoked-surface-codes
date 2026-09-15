import json
import os
from pathlib import Path

import numpy as np
import sinter

from yoked.hierarchical._provenance import (
    packed_sample_hash, sha256_bytes, sha256_file, write_json_atomic,
)
from yoked.hierarchical._record import load_record

root, run = Path(os.environ['OUT']), Path(os.environ['RUN'])
record = load_record(root / 'evaluation').record
collection = json.loads((root / 'evaluation/manifest.json').read_text())
history = json.loads((run / 'manifest.json').read_text())
results = json.loads((run / 'results.json').read_text())
for name in ('circuit.stim', 'model.dem'):
    assert sha256_file(run / name) == history['input_sha256'][name]
    assert sha256_file(root / 'evaluation/sample' / name) == history['input_sha256'][name]
detectors = np.load(run / 'detectors_packed.npy', mmap_mode='r', allow_pickle=False)
actual_packed = np.load(run / 'actual_observables_packed.npy', mmap_mode='r', allow_pickle=False)
payload_hash = packed_sample_hash(detectors, actual_packed)
assert payload_hash == history['input_sha256']['packed_detectors_then_observables_payload']
assert payload_hash == collection['parent_payload_sha256']
actual = np.unpackbits(actual_packed[record.rows], axis=1, bitorder='little')[:, :12].astype(bool)
np.testing.assert_array_equal(actual, record.actual)
audit = dict(parent_payload_sha256=payload_hash, rows=record.rows.tolist(),
             source_sha256=history['source_sha256'], versions=history['versions'],
             code_commit=history['code_commit'], results_sha256=sha256_file(run / 'results.json'), decoders={})
for name in ('mwpm', 'uf', 'correlated_mwpm', 'correlated_uf'):
    path = run / f'{name}_predictions.npy'
    predicted = np.load(path, mmap_mode='r', allow_pickle=False)
    assert predicted.shape == (history['parameters']['shots'], 12)
    assert np.isin(predicted, (0, 1)).all()
    prediction_hash = sha256_bytes(np.packbits(predicted, axis=1, bitorder='little').tobytes())
    assert prediction_hash == results['decoders'][name]['prediction_packed_sha256']
    failures = (predicted[record.rows] != actual).any(axis=1)
    block = float(failures.mean())
    audit['decoders'][name] = dict(prediction_packed_sha256=prediction_hash,
        file_sha256=sha256_file(path), errors=int(failures.sum()), shots=record.shots,
        block_failure=block,
        normalized_ler=float(sinter.shot_error_rate_to_piece_error_rate(block, pieces=216, values=8)))
write_json_atomic(root / 'baselines_pilot.json', audit)
print(json.dumps(audit['decoders'], indent=2))
