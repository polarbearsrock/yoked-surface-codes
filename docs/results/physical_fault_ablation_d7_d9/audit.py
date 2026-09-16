"""Check archived estimates and bound rare, unseen counterfactual outcomes.

Invert the finite-population hypergeometric distribution for each stratum's
UF-only and MWPM-only outcome counts. A Bonferroni union bound over eight
intervals gives a conservative 95% interval for each variant's population gap.
Unlike the bootstrap, these intervals allow outcomes absent from the sample.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom


def count_interval(successes, sampled, population, tail):
    minimum, maximum = successes, population - sampled + successes
    lo, hi = minimum, maximum
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom.sf(successes - 1, population, mid, sampled) >= tail:
            hi = mid
        else:
            lo = mid + 1
    lower = lo
    lo, hi = minimum, maximum
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(successes, population, mid, sampled) >= tail:
            lo = mid
        else:
            hi = mid - 1
    return lower, lo


def main():
    root = Path(__file__).resolve().parent
    report = dict(method=__doc__, distances={})
    for distance in (7, 9):
        directory = root / f'd{distance}'
        selection = json.loads((directory / 'selection.json').read_text())
        population = np.array(selection['population_counts'])
        strata = np.array(selection['strata'])
        original_gap = (population[1] - population[2]) / population.sum()
        bounds = {}
        for label in ('primary', 'detail'):
            path = directory / f'decoded_{label}.npz'
            validation = json.loads((directory / f'decode_validation_{label}.json').read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == validation['result_sha256']
            decoded = dict(np.load(path))
            baseline = dict(np.load(directory / 'baseline_labels.npz'))
            np.testing.assert_array_equal(decoded['predictions'][:, 0, 0], baseline['uf'])
            np.testing.assert_array_equal(decoded['predictions'][:, 0, 1], baseline['mwpm'])
            np.testing.assert_array_equal(decoded['actual'][:, 0], baseline['actual'])
            summaries = json.loads((directory / f'summary_{label}.json').read_text())['rows']
            for j, name in enumerate(decoded['variants']):
                failed = (decoded['predictions'][:, j] != decoded['actual'][:, j, None]).any(axis=2)
                outcome = failed[:, 0].astype(int) + 2 * failed[:, 1]
                gap, lower_total, upper_total = 0., 0, 0
                for code, size in enumerate(population):
                    sample = outcome[strata == code]
                    counts = np.bincount(sample, minlength=4)
                    assert counts.tolist() == summaries[j]['outcomes_by_stratum'][code]
                    gap += size / population.sum() * (counts[1] - counts[2]) / len(sample)
                    plus = count_interval(int(counts[1]), len(sample), int(size), .05 / 16)
                    minus = count_interval(int(counts[2]), len(sample), int(size), .05 / 16)
                    lower_total += plus[0] - minus[1]
                    upper_total += plus[1] - minus[0]
                np.testing.assert_allclose(gap, summaries[j]['gap_after'], atol=1e-12)
                if name == 'baseline':
                    np.testing.assert_allclose(gap, original_gap)
                    continue  # the full-population baseline is already known exactly
                interval = [original_gap - upper_total / population.sum(),
                            original_gap - lower_total / population.sum()]
                bounds[str(name)] = dict(gap_reduction_ci95=interval,
                    fraction_of_gap_removed_ci95=[x / original_gap for x in interval])
        report['distances'][str(distance)] = bounds
    report['all_archived_predictions_and_weighted_estimates_checked'] = True
    (root / 'finite_population_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Archived results verified; conservative finite-population intervals saved.')


if __name__ == '__main__':
    main()
