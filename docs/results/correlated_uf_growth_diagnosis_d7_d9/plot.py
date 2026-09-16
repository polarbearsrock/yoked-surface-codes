"""Plot measured growth traces and the competing logical correction costs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.7),
                             gridspec_kw={'width_ratios': [1.65, 1]})
    for index, (name, label) in enumerate([
            ('trace_44875', 'd=7 · reduced 12-fault example'),
            ('full_d9_34624', 'd=9 · complete saved shot 34624')]):
        data = json.loads((args.root/f'{name}.json').read_text())
        blocked = data['summary']['blocked_MWPM_edges'][0]
        edge = blocked['edge']
        changes = [change for event in data['history'] for change in
                   event['changes'] if change['edge'] == edge]
        times = [0.] + [event['time'] for event in data['history']
                        if any(c['edge'] == edge for c in event['changes'])]
        remaining = [blocked['weight']] + [c['after']['remaining'] for c in changes]
        ax = axes[index, 0]
        ax.plot(times, remaining, '-o', color='#246791', linewidth=2.4, markersize=4)
        stop = times[-1]
        residual = remaining[-1]
        ax.axhline(0, color='#999999', linewidth=.8)
        ax.plot([stop, stop+residual], [residual, 0], '--', color='#b45331', linewidth=2)
        ax.axvline(stop, color='#b45331', alpha=.35, linewidth=1)
        ax.scatter([stop], [residual], color='#b45331', zorder=5, s=45)
        ax.annotate(f'UF stops\n{residual:.3f} still needed', xy=(stop, residual),
                    xytext=(stop*.54, blocked['weight']*.48),
                    arrowprops={'arrowstyle': '->', 'color': '#b45331'},
                    color='#8d3c20', fontsize=11)
        ax.set_title(label, loc='left', fontweight='bold', pad=12)
        ax.set_ylabel('Remaining edge growth (nats)')
        ax.set_xlabel('UF growth time (algorithm units, not runtime)')
        ax.set_ylim(-blocked['weight']*.12, blocked['weight']*1.12)
        ax.set_xlim(-stop*.025, stop+max(residual, .07*stop))
        ax.grid(axis='y', color='#eeeeee')
        ax.set_axisbelow(True)
        comp, = data['summary']['logical_components']
        ax = axes[index, 1]
        costs = [comp['UF_cost'], comp['MWPM_cost']]
        ax.barh([1, 0], costs, color=['#b45331', '#246791'], height=.45)
        ax.set_yticks([1, 0], ['UF', 'MWPM'])
        for y, cost in zip([1, 0], costs):
            ax.text(cost+.4, y, f'{cost:.3f}', va='center')
        ax.set_xlim(0, max(costs)*1.23)
        ax.set_ylim(-.8, 1.8)
        ax.set_xlabel('Cost on the differing logical path (nats)')
        ax.set_title('Same weights, different logical answer', loc='left', fontsize=11, pad=12)
        ax.text(.04, .91, f'MWPM is cheaper by {costs[0]-costs[1]:.3f}',
                transform=ax.transAxes, color='#246791')
    fig.suptitle('UF stops at a valid correction before exposing a cheaper logical alternative',
                 fontsize=15, fontweight='bold', y=.985)
    fig.text(.04, .013, 'Solid lines: measured growth. Dashed: completion if this edge kept growing at unit rate.\n'
             'Adding this one edge and optimizing the resulting forest repairs each example; the edge choice here uses MWPM.',
             fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, .07, 1, .965), h_pad=2)
    fig.savefig(args.root/'growth_and_cost.png', dpi=180)
    fig.savefig(args.root/'growth_and_cost.svg')


if __name__ == '__main__':
    main()
