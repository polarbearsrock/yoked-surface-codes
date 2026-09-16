"""Check cohort labels, aggregates, traced corrections, and source provenance."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pymatching
import stim

from trace import Analysis, PREVIOUS, bitmask, write
from algorithms import minimum_cost_forest
from analyze import SOURCES, sha


def verify_cohort(a, root):
    directory = root/f'd{a.distance}'
    rows = json.loads((directory/'rows.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    previous = json.loads((PREVIOUS/f'd{a.distance}/selection.json').read_text())
    old_rows = {r['shot']: r for r in json.loads((PREVIOUS/f'd{a.distance}/rows.json').read_text())}
    assert selection['failures'] == previous['sample']
    u_ok = np.all(a.record.baselines['joint_correlated_uf'] == a.record.actual, axis=1)
    m_ok = np.all(a.record.baselines['joint_correlated_mwpm'] == a.record.actual, axis=1)
    expected = sorted(np.random.default_rng(selection['control_seed']).choice(
        np.flatnonzero(u_ok & m_ok), size=128, replace=False).tolist())
    assert expected == selection['both_correct_controls']
    assert len(rows) == 256
    for cohort, key in [('UF_only_failure','failures'), ('both_correct','both_correct_controls')]:
        selected = [r for r in rows if r['cohort'] == cohort]
        assert [r['shot'] for r in selected] == selection[key]
        for row in selected:
            shot = row['shot']
            assert row['actual'] == bitmask(a.record.actual[shot])
            assert row['baseline_UF'] == bitmask(a.record.baselines['joint_correlated_uf'][shot])
            assert row['tree_DP_original_mask'] == row['baseline_UF']
            if cohort == 'UF_only_failure':
                assert not u_ok[shot] and m_ok[shot]
                assert row['same_weight_MWPM_correct'] == (not old_rows[shot]['wrong']['UM'])
                assert old_rows[shot]['partition_logical_rank'] == 0
            else:
                assert u_ok[shot] and m_ok[shot]
            cost, edge_count = row['tree_DP_original_cost'], 0
            for cap in selection['caps']:
                v = row['variants'][cap]
                assert v['correct'] == (v['mask'] == row['actual'])
                assert math.isfinite(v['cost']) and v['cost'] <= cost + 1e-8
                assert v['added_edges'] >= edge_count
                cost, edge_count = v['cost'], v['added_edges']
    failures = [r for r in rows if r['cohort'] == 'UF_only_failure']
    controls = [r for r in rows if r['cohort'] == 'both_correct']
    same = [r for r in failures if r['same_weight_MWPM_correct']]
    assert summary['same_weight_MWPM_repairs'] == len(same)
    categories = Counter('boundary_only' if set(r['missing_edges_by_kind']) == {'boundary'}
                         else 'ordinary_or_yoke' for r in same)
    assert summary['missing_connection_categories'] == dict(categories)
    assert summary['single_missing_boundary'] == sum(r['missing_edges_by_kind'] == {'boundary':1} for r in same)
    for cap in selection['caps']:
        expected = dict(failure_repairs=sum(r['variants'][cap]['correct'] for r in failures),
            control_regressions=sum(not r['variants'][cap]['correct'] for r in controls),
            ordinary_peel_failure_repairs=sum(r['variants'][cap]['ordinary_peel_correct'] for r in failures),
            mean_added_edges=float(np.mean([r['variants'][cap]['added_edges'] for r in rows])))
        assert summary['variants'][cap] == expected
    assert summary['independent_tree_cost_checks'] == sum(r.get('independent_tree_cost_checks',0) for r in rows) == 24
    return dict(distance=a.distance, rows_verified=len(rows), input_record=str(a.loaded.directory)
                if hasattr(a.loaded,'directory') else str(a.directory), inputs=dict(a.run))


def verify_cases(a, root, checks):
    checked = 0
    for row in checks['cases']:
        if row['distance'] != a.distance:
            continue
        trace = json.loads((root/f"{row['case']}.json").read_text())
        if row['reduced']:
            source = json.loads((PREVIOUS/f"reduced_{trace['summary']['shot']}.json").read_text())
            syndrome = np.zeros(a.graph.num_detectors, dtype=bool)
            syndrome[source['defect_ids']] = True
            assert source['reduced']['actual_mask'] == row['actual']
        else:
            syndrome = a.syndrome(trace['summary']['shot'])
            assert bitmask(a.record.actual[trace['summary']['shot']]) == row['actual']
        assert a.mask(trace['UF_selected'], syndrome) == row['UF_mask'] != row['actual']
        assert a.mask(trace['MWPM_selected'], syndrome) == row['actual']
        first = a.uf._decode(syndrome)
        graph = a.regraph(a.weights(first.selected_edges))
        selected, cost = minimum_cost_forest(graph, syndrome,
            trace['UF_forest'] + [row['single_missing_edge']])
        assert a.mask(selected, syndrome) == row['augmented_DP_mask'] == row['actual']
        assert math.isclose(cost, row['augmented_DP_cost'], abs_tol=1e-8)
        assert math.isclose(cost, row['independent_matching_cost'], abs_tol=1e-5)
        assert len(row['tie_order_checks']) == 17
        assert all(t['same_partition'] and t['mask'] == row['UF_mask'] for t in row['tie_order_checks'])
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    checks = json.loads((args.root/'case_checks.json').read_text())
    inputs, checked = [], 0
    for distance in (7,9):
        a = Analysis(distance)
        old = json.loads((PREVIOUS/f'd{distance}/summary.json').read_text())
        for path in SOURCES:
            assert sha(path) == old['source_hashes'][path]
        inputs.append(verify_cohort(a, args.root))
        checked += verify_cases(a, args.root, checks)
    assert checked == 4
    # Parse every archived analysis source without creating bytecode files.
    for path in Path(__file__).parent.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    provenance = dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        production_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        production_source_hashes={p:sha(p) for p in SOURCES},
        versions=dict(stim=stim.__version__,pymatching=pymatching.__version__,numpy=np.__version__),
        inputs=inputs, cohort_rows_verified=512, case_interventions_rechecked=checked,
        recorded_tie_order_checks=68, recorded_independent_tree_cost_checks=52,
        production_source_matches_previous_investigation=True)
    write(args.root/'verification.json', provenance)
    print(json.dumps(dict(cohort_rows_verified=512,case_interventions_rechecked=checked,
                         source_hashes_match=True)), flush=True)
    files={str(p.relative_to(args.root)):sha(p) for p in sorted(args.root.rglob('*'))
           if p.is_file() and p.suffix in ('.json','.py','.png','.svg') and p.name != 'artifact_manifest.json'}
    write(args.root/'artifact_manifest.json', dict(sha256=files))


if __name__ == '__main__':
    main()
