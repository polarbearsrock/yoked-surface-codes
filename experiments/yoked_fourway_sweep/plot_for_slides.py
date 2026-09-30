"""Reproduce three presentation figures from the completed six-patch sweep."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np
import sinter

NAMES = ['correlated_mwpm_complementary_gap', 'correlated_mwpm_cluster_gap',
         'correlated_uf_cluster_gap', 'bp5_weighted_uf_cluster_gap']
LABELS = ['Corr. MWPM + complementary gap', 'Corr. MWPM + cluster gap',
          'Corr. UF + cluster gap', 'BP5 + weighted UF + cluster gap']
COLORS = ['#202020', '#398a8c', '#b17b14', '#a17fc5']
MARKERS = ['x', 's', 'D', 'o']
STYLES = [':', '-', '--', '-']
DISTANCES = [7, 9, 11, 13, 15]


def main(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    raw = (source / 'results.json').read_bytes()
    results = json.loads(raw)
    assert results['state'] == 'complete'
    data = results['distances']
    rows = []
    relative = {}
    hashes = {'results.json': hashlib.sha256(raw).hexdigest()}
    for d in DISTANCES:
        saved = data[str(d)]
        assert saved['shots'] == 100000
        assert saved['parameters']['patches'] == 6 and saved['rounds'] == 4*d
        assert saved['parameters']['p'] == .003 and saved['parameters']['noise'] == 'si1000'
        assert saved['latency']['protocol']['shots'] == 1000
        path = source / f'd{d}/paired.npz'
        hashes[f'd{d}/paired.npz'] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[f'd{d}/paired.npz'] == saved['paired_arrays_sha256']
        with np.load(path) as arrays:
            np.testing.assert_array_equal(arrays['variant_names'], NAMES)
            failures = arrays['block_failures']
        # Reuse the exact whole-shot bootstrap draws from the retained analysis.
        patterns, counts = np.unique(failures, axis=0, return_counts=True)
        draws = np.random.default_rng(saved['bootstrap']['seed']).multinomial(
            saved['shots'], counts/saved['shots'], size=saved['bootstrap']['replicates'])
        rates = (draws @ patterns.astype(np.int64)) / saved['shots']
        normal = np.array([[sinter.shot_error_rate_to_piece_error_rate(
            float(p), pieces=24*d, values=8) for p in row] for row in rates])
        assert (normal[:, 0] > 0).all()
        relative[d] = {}
        baseline = saved['variants'][NAMES[0]]['normalized_ler']
        for k, name in enumerate(NAMES):
            v = saved['variants'][name]
            assert int(failures[:, k].sum()) == v['block_failures']
            gap = 100*(v['normalized_ler']/baseline - 1)
            interval = np.percentile(100*(normal[:, k]/normal[:, 0]-1), [2.5, 97.5])
            relative[d][name] = dict(percent=gap, ci95=interval.tolist())
            if k:
                prior = saved['paired'][name+'__versus__'+NAMES[0]]
                np.testing.assert_allclose(np.percentile(normal[:, k]-normal[:, 0], [2.5, 97.5]),
                                           prior['normalized_difference_ci95'], rtol=1e-12, atol=1e-15)
            timing = saved['latency']['variants'][name]
            rows.append(dict(distance=d, rounds=4*d, patches=6, p=.003, shots=100000,
                variant=name, failures=v['block_failures'], normalized_ler=v['normalized_ler'],
                normalized_ci95_low=v['normalized_ci95'][0], normalized_ci95_high=v['normalized_ci95'][1],
                relative_percent=gap, relative_ci95_low=interval[0], relative_ci95_high=interval[1],
                timing_samples=1000, threads=1, median_ms=timing['median_ms'],
                mean_ms=timing['mean_ms'], p95_ms=timing['p95_ms'], p99_ms=timing['p99_ms']))

    # Same aspect ratio and plot family as the reference slides' original recipe.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18.5,
        'axes.labelsize': 19, 'xtick.labelsize': 18.5, 'ytick.labelsize': 18.5,
        'legend.fontsize': 14.4, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 1.2, 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})

    def canvas(footer):
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = fig.add_axes([.165, .225, .79, .535])
        ax.grid(axis='y', color='#e4e4e4', linewidth=.9)
        ax.set_axisbelow(True)
        ax.set_xticks(DISTANCES)
        ax.set_xlim(6.6, 15.4)
        ax.set_xlabel('Code distance, d', labelpad=7)
        fig.text(.54, .985, '6 patches  ·  SI1000 p = 0.3%  ·  4d rounds',
                 ha='center', va='top', fontsize=15)
        fig.text(.53, .025, footer, ha='center', va='bottom', fontsize=13.3, color='#555555')
        return fig, ax

    def finish(fig, ax, stem):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .905),
                   ncol=2, frameon=False, columnspacing=.95, handlelength=1.4,
                   handletextpad=.4, labelspacing=.45)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output/f'{stem}.{ext}', dpi=260)
        plt.close(fig)

    fig, ax = canvas('100k paired shots / d  ·  exact 95% intervals  ·  log scale')
    for k, name in enumerate(NAMES):
        y = np.array([data[str(d)]['variants'][name]['normalized_ler'] for d in DISTANCES])
        ci = np.array([data[str(d)]['variants'][name]['normalized_ci95'] for d in DISTANCES]).T
        ax.errorbar(DISTANCES, y, yerr=[y-ci[0], ci[1]-y], label=LABELS[k],
            color=COLORS[k], marker=MARKERS[k], linestyle=STYLES[k], linewidth=2,
            markersize=6.5, capsize=3, elinewidth=1.1)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1.8e-3)
    ax.set_ylabel('Effective LER\nper patch per round', labelpad=7)
    finish(fig, ax, 'normalized_ler')

    fig, ax = canvas('Normalized LER ratio  ·  95% paired bootstrap intervals')
    ax.axhline(0, color=COLORS[0], linestyle=STYLES[0], linewidth=1.5, label=LABELS[0])
    for k, name in enumerate(NAMES[1:], 1):
        y = np.array([relative[d][name]['percent'] for d in DISTANCES])
        ci = np.array([relative[d][name]['ci95'] for d in DISTANCES]).T
        ax.errorbar(DISTANCES, y, yerr=[y-ci[0], ci[1]-y], label=LABELS[k],
            color=COLORS[k], marker=MARKERS[k], linestyle=STYLES[k], linewidth=2.2,
            markersize=7, capsize=3, elinewidth=1.1)
    ax.set_ylim(-25, 178)
    ax.set_yticks([0, 50, 100, 150])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel('LER increase over baseline (%)', labelpad=7)
    finish(fig, ax, 'relative_ler')

    fig, ax = canvas('1,000 paired blocks / d  ·  one thread  ·  full six-patch block')
    for k, name in enumerate(NAMES):
        y = [data[str(d)]['latency']['variants'][name]['median_ms'] for d in DISTANCES]
        ax.plot(DISTANCES, y, label=LABELS[k], color=COLORS[k], marker=MARKERS[k],
                linestyle=STYLES[k], linewidth=2, markersize=6.5)
    ax.set_yscale('log')
    ax.set_ylim(2, 2100)
    ax.set_yticks([10, 100, 1000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel('Median serial decode time (ms)', labelpad=7)
    finish(fig, ax, 'serial_latency')

    with (output/'plot_values.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(source=str(source), source_sha256=hashes, distances=DISTANCES, variants=NAMES,
        labels=LABELS, colors=COLORS, relative=relative,
        normalization='sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=6*r, values=8)',
        accuracy_intervals='Exact binomial intervals transformed through the normalization',
        relative_intervals='10000 whole paired shot bootstrap replicates; same draws and seeds as original analysis',
        latency='Median warm single-thread API time per six-patch block, 1000 samples; includes L1, confidence and L2, Python adapters and built-in validation; shared host; excludes sampling, setup and external prediction comparison',
        d7_accuracy='Reused from prior corrected 100k-shot experiment',
        software='Dante Stim 1.17.dev0; PyMatching 2.4.0 including pinned native MPP engine',
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'plot_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'figures': 3, 'rows': len(rows),
                      'bootstrap_verified_against_retained_analysis': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
