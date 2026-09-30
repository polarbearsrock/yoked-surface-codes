"""Check the reviewed 01_Baseline Python noise builder against installed Stim.

Only extracts the reviewed parameter definitions and circuit-construction block.
It never runs Dongwhee's experiment launcher, C++ sampler, or saved task states.
This probes his standard surface-code circuit; it is not a port to our CZ circuit.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pymatching
import stim

EXPECTED_RUN_SHA256 = None  # Recorded in the review provenance before execution.
MESSAGE = 'https://pncel.slack.com/archives/D0BNPSPPZ27/p1790662621419269'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named_assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def extract_builder(source):
    tree = ast.parse(source.read_text())
    devices = next(ast.literal_eval(n.value) for n in tree.body if named_assignment(n, 'DEVICE_PARAMS'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('compute_pauli_twirl', 'compute_n_dd_pulses')]
    assert len(functions) == 2
    namespace = dict(math=math, stim=stim)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    pipeline = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_pipeline')
    body = next(n.body for n in pipeline.body if isinstance(n, ast.Try))
    first = next(i for i, n in enumerate(body) if named_assignment(n, 'ideal'))
    last = next(i for i, n in enumerate(body) if named_assignment(n, 'dem'))
    construction = body[first:last+1]
    # Protect the reviewed boundary against unintentionally including orchestration.
    forbidden = {'os', 'sys', 'subprocess', 'task_state', 'multiprocessing', 'open', 'exec', 'eval', '__import__'}
    assert not any(isinstance(n, ast.Name) and n.id in forbidden
                   for node in construction for n in ast.walk(node))
    code = compile(ast.Module(body=construction, type_ignores=[]), str(source), 'exec')
    return devices, namespace, code, dict(first_line=body[first].lineno, last_line=body[last].end_lineno)


def build(device_name, distance, rounds, memory, devices, namespace, code):
    dev = devices[device_name]
    t_data = dev['QEC_cycle'] - 4*dev['t_2Q']
    t_ancilla = max(0., dev['QEC_cycle'] - (dev['M_dur']+dev['R_dur']+2*dev['t_1Q']+4*dev['t_2Q']))
    scope = dict(namespace, d=distance, r=rounds, mem_type=memory,
        device_name=device_name, dev=dev, pid='compatibility-probe',
        T1_NS=dev['T1_ns'], T2_NS=dev['T2_ns'], T_DATA_NS=t_data, T_ANCILLA_NS=t_ancilla,
        p1=0.0001, p2=0.001, pm=dev['p_meas_at_Lb'], pr=0.001)
    exec(code, scope)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--decoder-reference', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    reference = output/'source_snapshot/01_Baseline'
    reference.mkdir(parents=True)
    for name in ('run.py', 'simulation.cpp', 'simulation.hpp', 'main.cpp', 'Makefile',
                 'README.md', 'task_state.py', 'monitoring.py', 'ideal_surface_code_d5.stim'):
        shutil.copyfile(args.source/name, reference/name)
    shutil.copyfile(Path(__file__).resolve(), output/'probe.py')
    devices, namespace, code, span = extract_builder(reference/'run.py')
    records = []
    selected_scope = None
    for index, device in enumerate(devices):
        for basis in ('x', 'z'):
            scope = build(device, 3, 3, f'rotated_memory_{basis}', devices, namespace, code)
            circuit, dem = scope['circuit'], scope['dem']
            assert circuit.num_observables == 1
            assert len(scope['data_qubits']) == 9 and len(scope['ancilla_qubits']) == 8
            assert scope['current_MR'] == 3
            detectors, actual = circuit.compile_detector_sampler(seed=202609290700+2*index+(basis == 'z')).sample(
                shots=128, separate_observables=True)
            matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            prediction = matching.decode_batch(detectors, enable_correlations=True)
            assert prediction.shape == actual.shape == (128, 1)
            prefix = f'{device}_d3_r3_{basis}'
            (output/f'{prefix}.stim').write_text(str(circuit))
            (output/f'{prefix}.dem').write_text(str(dem))
            records.append(dict(device=device, distance=3, rounds=3, basis=basis,
                detectors=circuit.num_detectors, observables=1, sampled_shots=128,
                exact_dem_constructed=True, correlated_mwpm_decoded=True,
                data_idle_ns=scope['T_DATA_NS'], ancilla_idle_ns=scope['T_ANCILLA_NS'],
                data_pauli_probabilities=[scope['pXY_data']]*2+[scope['pZ_data']],
                dd_pulses=scope['n_dd_pulses']))
    # Verify the decoder-side interface using the same frozen native implementation
    # as the completed study, on the new reference circuit's single observable.
    snapshot = args.decoder_reference/'source_snapshot'
    sys.path.insert(0, str(snapshot/'experiments/bp_uf_single_patch_sweep'))
    import run as decoder_study
    config = json.loads((args.decoder_reference/'configuration.json').read_text())
    library = Path(config['compile_command'][-1])
    assert sha256(library) == config['native_library_sha256']
    scope = build('google_willow', 7, 28, 'rotated_memory_x', devices, namespace, code)
    circuit, dem = scope['circuit'], scope['dem']
    graph = decoder_study.evidence.DecodingGraph.from_dem(dem)
    faults = decoder_study.evidence.FaultModel.from_dem(dem, graph)
    native = decoder_study.Native(library, graph, decoder_study.evidence.correlation_rules_from_dem(graph, dem), faults)
    detectors, actual = circuit.compile_detector_sampler(seed=202609290800).sample(
        shots=32, separate_observables=True, bit_packed=True)
    try:
        fields, flags = native.decode(detectors, 1, audit=True)
        syndromes = np.unpackbits(detectors, axis=1, count=circuit.num_detectors, bitorder='little')
        for syndrome, outputs, selected in zip(syndromes, fields, flags):
            for k in range(6):
                residual, mask = syndrome.copy(), 0
                for edge in np.flatnonzero(selected & (1 << k)):
                    u, v, _, label = graph.edges[edge]
                    residual[u] ^= 1
                    if v is not None:
                        residual[v] ^= 1
                    mask ^= label
                assert not residual.any() and mask == int(outputs[k, 0])
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        matched = matching.decode_batch(detectors, bit_packed_shots=True,
            bit_packed_predictions=True, enable_correlations=True)
        assert matched.shape == actual.shape == (32, 1)
        (output/'google_willow_d7_r28_x.stim').write_text(str(circuit))
        (output/'google_willow_d7_r28_x.dem').write_text(str(dem))
        np.savez_compressed(output/'interface_probe.npz', detectors=detectors, actual=actual,
            native_fields=fields, mwpm_prediction=matched, native_variants=np.array(decoder_study.NATIVE_VARIANTS))
    finally:
        native.close()
    summary = dict(status='passed', slack_message=MESSAGE, slack_file_id='F0C56JBPRGS',
        archive_sha256=sha256(args.archive), source_run_sha256=sha256(reference/'run.py'),
        construction_span=span, reference_tests=records,
        seven_decoder_interface=dict(device='google_willow', distance=7, rounds=28,
            memory='rotated_memory_x', observables=1, shots=32,
            native_physical_corrections_verified=192, fixed_budgets=[1, 2, 5, 10],
            matching_correlations_enabled=True, library_sha256=sha256(library)),
        stim_version=stim.__version__, stim_path=stim.__file__, pymatching_version=pymatching.__version__,
        command=[sys.executable, *sys.argv], created_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['Compatibility checks only; no statistical performance claim.',
            'Original Python builder preserved, including first-reset placement of initial decoherence.',
            'Custom C++ sampler has not been built or statistically cross-validated.',
            'No port to the original CZ magic-boundary memory circuit has been made.',
            'Device baseline gate rates are p2=0.001; these are not SI1000 p=0.003.'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    manifest = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    (output/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='passed', output=str(output), reference_cases=len(records),
                         seven_decoder_interface=summary['seven_decoder_interface']), indent=2))


if __name__ == '__main__':
    main()
