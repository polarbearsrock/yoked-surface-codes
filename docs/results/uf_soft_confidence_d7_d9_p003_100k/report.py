"""Verify the completed UF-confidence experiment and export its report artifacts."""
from pathlib import Path
import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
import sinter

from yoked.hierarchical._provenance import read_json, sha256_file, source_hashes, write_json_atomic
from yoked.hierarchical._record import load_record
from yoked.hierarchical._stages import _load_result, _verified_replay_manifest, _require_same_record

METHODS = (
    ('joint_mwpm_recorded', 'Joint MWPM'),
    ('joint_uf', 'Joint UF'),
    ('joint_correlated_mwpm', 'Correlated MWPM'),
    ('joint_correlated_uf', 'Correlated UF'),
    ('cluster_gap', 'HUF + cluster gap'),
    ('bounded_cluster', 'HUF + bounded cluster gap'),
    ('correlated_bounded_cluster', 'HUF + correlated bounded gap'),
    ('uf_gap', 'HUF + UF class-cost gap'),
    ('correlated_uf_gap', 'HUF + correlated UF class-cost gap'),
    ('correlated_uf_gap_4bit', 'HUF + 16-bin correlated UF gap'),
    ('correlated_matching_gap', 'HUF + correlated matching gap'),
)


def verify(root, distance):
    path = root / f'd{distance}' / 'analysis'
    manifest = read_json(path / 'manifest.json')
    for name, digest in manifest['artifacts'].items():
        assert sha256_file(path / name) == digest, name
    for name, digest in manifest['inputs'].items():
        assert sha256_file(name) == digest, name
    data = read_json(path / 'results.json')
    assert source_hashes(data['analysis_sources']) == data['analysis_sources']
    evaluation = load_record(data['evaluation_record'])
    calibration = load_record(data['calibration_record'])
    assert evaluation.record.shots == 100000 and calibration.record.shots == 50000
    assert evaluation.manifest['seed'] == 42 and calibration.manifest['seed'] == 142
    assert dict(evaluation.manifest['parameters']) == data['parameters']
    assert data['parameters']['distance'] == distance
    assert data['parameters']['rounds'] == 4 * distance
    assert data['parameters']['p'] == .003 and data['parameters']['patches'] == 6
    with np.load(path / 'predictions.npz') as saved:
        predictions = dict(saved)
    for name, prediction in predictions.items():
        count = int(np.any(prediction != evaluation.record.actual, axis=1).sum())
        row = data['methods'][name]
        assert count == row['failures']
        assert row['normalized_ler'] == sinter.shot_error_rate_to_piece_error_rate(
            count / 100000, pieces=6 * 4 * distance, values=8)
    # The re-fit expensive endpoint must reproduce the preceding MWPM-L2 experiment.
    replay = root.parent / 'hier-mwpm-l2-d7-d9-1f0KIx' / f'd{distance}/replay_full'
    replay_manifest = _verified_replay_manifest(replay)
    _require_same_record(evaluation, replay_manifest['record'], where=replay / 'replay_manifest.json')
    previous, _ = _load_result(replay / 'uf_cluster_gap_to_gap_correlated_all_refined_mixed')
    assert np.array_equal(previous.final, predictions['correlated_matching_gap'])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {str(d): verify(args.root, d) for d in (7, 9)}
    benchmark = read_json(args.root / 'benchmark.json')
    verification = dict(all_counts_and_ler_recomputed=True, previous_matching_endpoint_bit_identical=True,
                        all_artifact_and_input_hashes_verified=True, current_analysis_sources_verified=True)
    combined = dict(distances=results, benchmark=benchmark,
                    benchmark_environment=read_json(args.root / 'benchmark_environment.json'),
                    verification=verification)
    write_json_atomic(out / 'comparison.json', combined)
    write_json_atomic(out / 'verification.json', verification)
    for d in (7, 9):
        destination = out / f'd{d}'
        destination.mkdir(exist_ok=True)
        for name in ('results.json', 'manifest.json', 'calibrators.json', 'confidence_rom.json'):
            shutil.copyfile(args.root / f'd{d}/analysis' / name, destination / name)
    for name in ('run.py', 'benchmark.py', 'benchmark.json', 'benchmark_environment.json', 'report.py'):
        shutil.copyfile(args.root / name, out / name)
    lines = ['| Decoder | d=7 failures | d=7 normalized LER | d=9 failures | d=9 normalized LER |',
             '|---|---:|---:|---:|---:|']
    for name, label in METHODS:
        rows = [results[str(d)]['methods'][name] for d in (7, 9)]
        lines.append(f"| {label} | {rows[0]['failures']:,} | {rows[0]['normalized_ler']:.8g} | "
                     f"{rows[1]['failures']:,} | {rows[1]['normalized_ler']:.8g} |")
    lines += ['', 'All denominators are 100,000 shots. HUF uses a fixed plain-UF L1 reference and MWPM at L2.', '']
    for d in (7, 9):
        lines += [f'## d={d}: uncertainty and comparison with correlated UF', '',
                  '| Decoder | Normalized LER (95% CI) | LER ratio to correlated UF (paired 95% CI) |',
                  '|---|---:|---:|']
        for name, label in METHODS:
            row = results[str(d)]['methods'][name]
            lo, hi = row['ler_ci95']
            rl, rh = row['paired_ratio_ci95']
            lines.append(f"| {label} | {row['normalized_ler']:.8g} ({lo:.8g}, {hi:.8g}) | "
                         f"{row['ler_ratio_to_correlated_uf']:.5f} ({rl:.5f}, {rh:.5f}) |")
        lines.append('')
    (out / 'tables.md').write_text('\n'.join(lines))
    fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')
    positions = np.arange(len(METHODS))
    for d, shift, color in ((7, -.12, '#2166ac'), (9, .12, '#d95f02')):
        rows = [results[str(d)]['methods'][name] for name, _ in METHODS]
        values = np.array([r['normalized_ler'] for r in rows])
        intervals = np.array([r['ler_ci95'] for r in rows])
        ax.errorbar(values, positions + shift,
                    xerr=np.stack([values - intervals[:, 0], intervals[:, 1] - values]),
                    fmt='o', markersize=4, capsize=2, label=f'd = {d}', color=color)
    ax.set_yticks(positions, [label for _, label in METHODS])
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Normalized logical error rate (95% evaluation-shot intervals)')
    ax.set_title('UF confidence alternatives: p = 0.003, 6 patches, rounds = 4d\n100,000 reused evaluation shots per distance; MWPM at L2')
    ax.grid(axis='x', alpha=.25, which='both')
    ax.legend()
    fig.savefig(out / 'comparison.png', dpi=180)
    fig.savefig(out / 'comparison.svg')
    print('\n'.join(lines[:len(METHODS) + 4]))


if __name__ == '__main__':
    main()
