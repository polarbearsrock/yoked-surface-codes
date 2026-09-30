"""Summarize individual CZ deletions and plot the 15-type family ablation."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import stim

from analyze import base
from yoked.decoders import DecodingGraph


def load_events(directory):
    path = directory / 'single_events.json'
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress((directory / 'single_events.json.gz').read_bytes()))


def outcome(event, key='removed'):
    v = event[key]
    return int(v['UF'] != v['actual']) + 2 * int(v['MWPM'] != v['actual'])


def component_category(event):
    components = event['direct_graph_components']
    if components is None:
        return 'not_directly_mapped'
    if not components:
        return 'no_components'
    u = all(c['UF_selected'] for c in components)
    m = all(c['MWPM_selected'] for c in components)
    return ('both_select_all' if u and m else 'MWPM_only_selects_all' if m else
            'UF_only_selects_all' if u else 'neither_selects_all')


def counts(events):
    return dict(events=len(events),
        outcomes_after_deletion=dict(Counter(str(outcome(e)) for e in events)),
        both_correct_after_deletion=sum(outcome(e) == 0 for e in events),
        any_unfired_footprint=sum(bool(e['unfired_footprint_detectors']) for e in events),
        any_blocked_direct_component=sum(e['direct_graph_components'] is not None and
            any(not c['UF_same_cluster'] for c in e['direct_graph_components']) for e in events),
        component_selection=dict(Counter(component_category(e) for e in events)),
        footprint_sizes=dict(Counter(str(len(e['detectors'])) for e in events)),
        affected_shots=len({e['shot'] for e in events}))


def paired_association(events, category, labels, draws, shots):
    """Resample shots, preserving dependence among all deletions in one shot."""
    position = {shot: k for k, shot in enumerate(shots)}
    total = np.zeros((len(shots), 2), dtype=int)
    repaired = np.zeros_like(total)
    for event in events:
        group = category(event)
        if group is None:
            continue
        k = position[event['shot']]
        total[k, group] += 1
        repaired[k, group] += outcome(event) == 0
    boot = (draws @ repaired) / (draws @ total)
    rates = repaired.sum(axis=0) / total.sum(axis=0)
    return dict(groups=labels, events=total.sum(axis=0).tolist(),
        repairs=repaired.sum(axis=0).tolist(), repair_rates=rates.tolist(),
        repair_rate_ci95=np.quantile(boot, [.025, .975], axis=0).T.tolist(),
        second_minus_first_rate=float(rates[1]-rates[0]),
        difference_ci95=np.quantile(boot[:, 1]-boot[:, 0], [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = dict(outcome_codes={'0': 'both correct', '1': 'UF alone fails',
        '2': 'MWPM alone fails', '3': 'both fail'}, distances={},
        interpretation='Conditional sample of UF-only failures. Events within one shot '
            'are dependent; deletion effects overlap. Counts are not shares of the LER gap.')
    pauli_summaries = []
    for distance in (7, 9):
        directory = args.work_dir / f'd{distance}'
        data = load_events(directory)
        events = data['events']
        selected = json.loads((directory / 'single_event_selection.json').read_text())
        assert {e['shot'] for e in events} == set(selected['shots'])
        assert all(outcome(e, 'alone') == 0 for e in events)
        selection = json.loads((directory / 'selection.json').read_text())
        graph = DecodingGraph.from_dem(stim.DetectorErrorModel.from_file(Path(selection['source'])/'model.dem'))
        mapped = 0
        for event in events:
            components = event['direct_graph_components']
            if components is None:
                continue
            observed, detectors = 0, set()
            for component in components:
                u, v, _, label = graph.edges[component['edge']]
                observed ^= label
                detectors.symmetric_difference_update([u] if v is None else [u, v])
            assert sorted(detectors) == event['detectors']
            assert observed == sum(1 << k for k in event['observables'])
            mapped += 1
        repairs = [e for e in events if outcome(e) == 0]
        shots = selected['shots']
        rng = np.random.default_rng(2026091723 + distance)
        draws = rng.multinomial(len(shots), np.full(len(shots), 1/len(shots)), size=20000)
        row = dict(selected_shots=len(data['baseline']), all_events=counts(events),
            repairing_events=counts(repairs), all_events_correct_in_isolation=True,
            directly_mapped_event_responses_independently_verified=mapped,
            by_pauli={p: counts([e for e in events if e['pauli'] == p]) for p in sorted({e['pauli'] for e in events})},
            by_CZ_slot={str(slot): counts([e for e in events if e['CZ_slot'] == slot]) for slot in range(1, 5)},
            by_footprint_size={str(n): counts([e for e in events if len(e['detectors']) == n])
                               for n in sorted({len(e['detectors']) for e in events})},
            by_cancellation={str(value): counts([e for e in events if bool(e['unfired_footprint_detectors']) == value])
                             for value in (False, True)},
            exploratory_associations=dict(
                detector_cancellation=paired_association(events,
                    lambda e: int(bool(e['unfired_footprint_detectors'])),
                    ['no cancelled footprint detector', 'at least one cancelled footprint detector'], draws, shots),
                footprint_size=paired_association(events,
                    lambda e: {2: 0, 4: 1}.get(len(e['detectors'])),
                    ['two detectors', 'four detectors'], draws, shots),
                data_Y=paired_association(events, lambda e: int(e['pauli'][0] == 'Y'),
                    ['data Pauli I/X/Z', 'data Pauli Y'], draws, shots),
                method='20,000 paired bootstrap resamples of whole shots; marginal exploratory '
                    '95% intervals conditional on original UF-only failures. '
                    'These associations do not isolate causal mechanisms.'))
        summary['distances'][str(distance)] = row
        print(json.dumps(dict(distance=distance, all_events=row['all_events'], repairing_events=row['repairing_events']), indent=2))
        pauli_summaries.append(json.loads((directory / 'summary_pauli.json').read_text()))
    base.write_json(args.work_dir / 'single_event_summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.8), sharey=True)
    colors = ['#2166ac', '#b35806']
    for ax, data, color in zip(axes, pauli_summaries, colors, strict=True):
        rows = data['rows'][1:-1]
        y = np.arange(len(rows))
        centers = np.array([r['fraction_of_gap_removed']*100 for r in rows])
        bounds = np.array([r['fraction_of_gap_removed_ci95'] for r in rows])*100
        ax.errorbar(centers, y, xerr=np.stack([centers-bounds[:, 0], bounds[:, 1]-centers]),
                    fmt='o', color=color, markersize=5, capsize=3, linewidth=1.4)
        ax.axvline(0, color='#777777', linestyle='--', linewidth=1)
        ax.set_yticks(y, [r['variant'][3:] for r in rows])
        ax.set_title(f'd = {data["distance"]}')
        ax.set_xlabel('Reduction in UF–MWPM shot-failure gap (%)')
        ax.grid(axis='x', alpha=.2)
        ax.set_xlim(-45, 55)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].set_ylabel('CZ fault Pauli: data, then ancilla')
    fig.suptitle('Delete one CZ Pauli family; keep all other recorded faults', fontsize=13)
    fig.text(.5, .015, '1,024 stratified shots per distance; marginal 95% paired bootstrap intervals.\n'
             'Effects overlap. The comparison does not establish a unique leading Pauli type.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    for extension in ('png', 'svg'):
        fig.savefig(args.work_dir / f'pauli_gap.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
