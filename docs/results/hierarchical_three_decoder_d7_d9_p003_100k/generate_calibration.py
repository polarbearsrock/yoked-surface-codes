"""Sample a separate calibration call from the exact saved d7 circuit bytes."""
import os
import shutil
from pathlib import Path
import numpy as np
import stim
from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._provenance import package_versions, packed_sample_hash, sha256_file, utc_now, write_json_atomic

root = Path(os.environ['TMPDIR']) / 'hier-three-decoders-d7-d9-p003-100k-3MxG5S'
source = Path(os.environ['TMPDIR']) / 'ysc-four-decoders-d7-p003-100k-adapter'
out = root / 'd7/calibration_source'
if out.exists():
    raise ValueError(f'Refusing to replace {out}')
sample = SampleSet.load_recorded_run(source)
assert sample.model_versions == package_versions(('stim',))
seed, shots = 142, 50000
circuit = stim.Circuit(sample.circuit_text)
detectors, actual = circuit.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True, bit_packed=True)
out.mkdir(parents=True)
for name in ('circuit.stim', 'model.dem'):
    shutil.copyfile(source / name, out / name)
np.save(out / 'detectors_packed.npy', detectors)
np.save(out / 'actual_observables_packed.npy', actual)
write_json_atomic(out / 'manifest.json', {
    'parameters': {**sample.parameters.to_json(), 'seed': seed, 'shots': shots},
    'input_sha256': {'circuit.stim': sample.circuit_sha256, 'model.dem': sample.dem_sha256,
                     'packed_detectors_then_observables_payload': packed_sample_hash(detectors, actual)},
    'versions': package_versions(('stim', 'numpy')),
    'sampling': {'seed': seed, 'sample_calls': 1, 'shots_per_call': shots, 'bit_packed': True, 'separate_observables': True},
    'circuit_source': {'directory': str(source), 'manifest_sha256': sha256_file(source / 'manifest.json')},
    'generation_script_sha256': sha256_file(__file__), 'created_utc': utc_now(),
})
verified = SampleSet.load_recorded_run(out)
assert verified.identities['model'] == sample.identities['model']
assert verified.identities['sampling_family'] != sample.identities['sampling_family']
assert verified.identities['parent_sample'] != sample.identities['parent_sample']
print({'directory': str(out), 'shots': verified.shots, 'payload_sha256': verified.payload_sha256, 'model': verified.identities['model']})
