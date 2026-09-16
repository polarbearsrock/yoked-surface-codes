"""Verify archived pilot estimates and add finite-population uncertainty bounds."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    helper = root.parent / 'physical_fault_ablation_d7_d9/audit.py'
    spec = importlib.util.spec_from_file_location('finite_population_intervals', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for distance in (7, 9):
        directory = root / f'd{distance}'
        report = json.loads((directory / 'summary.json').read_text())
        request = json.loads((directory / 'request.json').read_text())
        data = dict(np.load(directory / 'results.npz'))
        assert hashlib.sha256((directory / 'results.npz').read_bytes()).hexdigest() == report['validation']['results_sha256']
        assert hashlib.sha256((root / 'probe.py').read_bytes()).hexdigest() == request['analysis_sha256']
        strata = np.array(request['original_selection']['strata'])
        populations = np.array(report['populations'])
        np.testing.assert_array_equal(data['predictions'][:, 1], data['baseline'])
        np.testing.assert_array_equal((data['baseline'] != data['actual']).astype(int)
                                     + 2 * (data['matching'] != data['actual']), strata)
        for row in report['rows']:
            step = row['pass_number'] - 1
            lower = upper = 0
            estimate = 0.
            for code, population in enumerate(populations):
                cohort = strata == code
                original_fail = code in (1, 3)
                failures = data['predictions'][cohort, step] != data['actual'][cohort]
                estimate += population / populations.sum() * failures.mean()
                changed = int(np.count_nonzero(failures != original_fail))
                expected = row['repairs_by_stratum'][code] if original_fail else row['regressions_by_stratum'][code]
                assert changed == expected
                lo, hi = module.count_interval(changed, int(cohort.sum()), int(population), .05 / 8)
                if original_fail:
                    lower += lo
                    upper += hi
                else:
                    lower -= hi
                    upper -= lo
            np.testing.assert_allclose(estimate, row['estimated_shot_failure_probability'], atol=1e-12)
            row['conservative_reduction_ci95'] = [0, 0] if step == 1 else [lower / int(populations.sum()), upper / int(populations.sum())]
        varying = np.zeros(len(strata), dtype=bool)
        for shot, period in enumerate(data['cycle_period']):
            if period > 1:
                first = data['first_seen_pass'][shot] - 1
                varying[shot] = len(set(data['predictions'][shot, first:first + period])) > 1
        equal_23 = data['predictions'][:, 1] == data['predictions'][:, 2]
        later_changes = np.any(data['predictions'][:, 3:] != data['predictions'][:, 1:2], axis=1)
        correct = data['predictions'] == data['actual'][:, None]
        early_correct = np.any(correct[:, :2], axis=1)
        later_correct = np.any(correct[:, 2:], axis=1)
        uf_only = strata == 1
        report['additional_diagnostics'] = dict(
            logical_variation_within_cycle_counts_by_stratum=[int(varying[strata == code].sum()) for code in range(4)],
            estimated_logical_variation_within_cycle_probability=float(sum(
                populations[code] / populations.sum() * varying[strata == code].mean() for code in range(4))),
            same_logical_prediction_in_passes_2_and_3_but_later_changes_by_stratum=[
                int((equal_23 & later_changes & (strata == code)).sum()) for code in range(4)],
            pass8_relative_ler_reduction=1 - report['rows'][7]['estimated_normalized_ler'] / report['rows'][1]['estimated_normalized_ler'],
            uf_only_correct_at_pass1=int(np.count_nonzero(uf_only & correct[:, 0])),
            uf_only_new_correct_answer_beyond_first_two_passes=int(np.count_nonzero(uf_only & ~early_correct & later_correct)),
            uf_only_any_correct_answer_across_all_eight_passes=int(np.count_nonzero(uf_only & (early_correct | later_correct))),
        )
        report['conservative_interval_method'] = (
            'Exact finite-population hypergeometric inversion for repair/regression Bernoulli counts '
            'in four strata, with a Bonferroni union bound; marginal 95% per fixed pass count.')
        (directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Pilot hashes, baseline labels, weighted estimates and conservative intervals verified.')


if __name__ == '__main__':
    main()
