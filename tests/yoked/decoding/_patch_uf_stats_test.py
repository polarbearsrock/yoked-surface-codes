from dataclasses import asdict
import json

import pytest

from yoked.decoding._patch_uf_stats import (
    ClusterSizeBin,
    ShotClusterRecord,
    bootstrap_cluster_sizes,
    bootstrap_workload_coverage,
    canonical_sparse_histogram,
    clopper_pearson_interval,
    merge_sparse_histograms,
    sparse_histogram_total,
    sparse_type7_quantile,
    summarize_cluster_sizes,
    summarize_paired_accuracy,
    summarize_workload_coverage,
    tango_paired_risk_difference_interval,
    validate_hrlk_histogram,
)
from yoked.decoding._promatch_stats import PairedContingency


def test_paired_accuracy_golden_table_and_exact_intervals():
    summary = summarize_paired_accuracy(
        a=970,
        b=20,
        c=10,
        d=0,
        prediction_agreements=975,
    )
    assert (summary.shots, summary.global_failures, summary.treatment_failures) == (
        1000,
        10,
        20,
    )
    assert summary.global_failure_rate == pytest.approx(0.01)
    assert summary.treatment_failure_rate == pytest.approx(0.02)
    assert summary.risk_difference == pytest.approx(0.01)
    assert summary.discordant == 30
    assert summary.discordance_rate == pytest.approx(0.03)
    assert summary.prediction_agreement_rate == pytest.approx(0.975)
    assert summary.global_failure_interval.lower == pytest.approx(
        0.004805510691049308
    )
    assert summary.global_failure_interval.upper == pytest.approx(
        0.01831324305511245
    )
    assert summary.treatment_failure_interval.lower == pytest.approx(
        0.012258267972406329
    )
    assert summary.treatment_failure_interval.upper == pytest.approx(
        0.03072003268260911
    )
    assert summary.risk_difference_interval.lower == pytest.approx(
        -0.0007833657002193004,
        abs=1e-11,
    )
    assert summary.risk_difference_interval.upper == pytest.approx(
        0.021817602464888862,
        abs=1e-11,
    )


def test_exact_interval_zero_and_all_event_boundaries():
    zero = clopper_pearson_interval(successes=0, trials=100)
    all_events = clopper_pearson_interval(successes=100, trials=100)
    assert zero.lower == 0
    assert zero.upper == pytest.approx(0.03621669264517646)
    assert all_events.lower == pytest.approx(0.9637833073548235)
    assert all_events.upper == 1

    tango = tango_paired_risk_difference_interval(
        PairedContingency(100, 0, 0, 0)
    )
    assert tango.lower == pytest.approx(-0.036993498207053714, abs=1e-12)
    assert tango.upper == pytest.approx(0.036993498207053714, abs=1e-12)


def test_paired_accuracy_validation_is_fail_closed():
    with pytest.raises(ValueError, match="at least one shot"):
        summarize_paired_accuracy(a=0, b=0, c=0, d=0)
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_paired_accuracy(a=1, b=-1, c=0, d=0)
    with pytest.raises(TypeError, match="integer"):
        summarize_paired_accuracy(a=True, b=0, c=0, d=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        summarize_paired_accuracy(
            a=1, b=0, c=0, d=0, prediction_agreements=2
        )


def test_sparse_histograms_and_type7_quantiles_are_exact():
    histogram = {4: 1, 1: 1, 2: 2}
    assert canonical_sparse_histogram(histogram) == ((1, 1), (2, 2), (4, 1))
    assert sparse_histogram_total(histogram) == 4
    assert merge_sparse_histograms(({1: 2}, ((2, 1),), {1: 3})) == (
        (1, 5),
        (2, 1),
    )
    assert sparse_type7_quantile(histogram, 0) == 1
    assert sparse_type7_quantile(histogram, 0.25) == pytest.approx(1.75)
    assert sparse_type7_quantile(histogram, 0.5) == 2
    assert sparse_type7_quantile(histogram, 0.9) == pytest.approx(3.4)
    assert sparse_type7_quantile(histogram, 0.99) == pytest.approx(3.94)
    assert sparse_type7_quantile(histogram, 1) == 4

    with pytest.raises(ValueError, match="duplicate"):
        canonical_sparse_histogram(((1, 1), (1, 2)))
    with pytest.raises(ValueError, match="positive"):
        canonical_sparse_histogram({1: 0})
    with pytest.raises(ValueError, match="must not be empty"):
        sparse_type7_quantile({}, 0.5)


def _mixed_cluster_records():
    return [
        ShotClusterRecord(0, True, {1: 2, 3: 1}, 3),
        ShotClusterRecord(1, False, {2: 1}, None),
        ShotClusterRecord(2, True, {}, 0),
    ]


def test_cluster_views_keep_completed_rows_from_incomplete_shots():
    summary = summarize_cluster_sizes(_mixed_cluster_records())
    assert summary.shots == 3
    assert summary.complete_cluster_summary_shots == 2
    assert summary.censored_cluster_summary_shots == 1
    assert summary.censor_rate == pytest.approx(1 / 3)
    assert summary.completed_components.histogram == ((1, 2), (2, 1), (3, 1))
    assert summary.completed_components.denominator == 4
    assert summary.completed_components.p50 == pytest.approx(1.5)
    assert summary.completed_components.maximum == 3
    assert summary.complete_shot_maxima.histogram == ((0, 1), (3, 1))
    assert summary.complete_shot_maxima.denominator == 2
    assert summary.complete_shot_maxima.p90 == pytest.approx(2.7)


def test_cluster_record_and_group_validation():
    with pytest.raises(ValueError, match="does not match"):
        ShotClusterRecord(0, True, {1: 1}, 2)
    with pytest.raises(ValueError, match="must be null"):
        ShotClusterRecord(0, False, {1: 1}, 1)
    duplicate = [
        ShotClusterRecord(4, True, {}, 0),
        ShotClusterRecord(4, True, {}, 0),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cluster_sizes(duplicate)


def test_cluster_bootstrap_is_seeded_grouped_and_has_golden_endpoints():
    records = [
        ShotClusterRecord(10, True, {1: 2}, 1),
        ShotClusterRecord(20, True, {2: 1}, 2),
        ShotClusterRecord(30, True, {4: 1}, 4),
    ]
    bins = [ClusterSizeBin("small", 1, 3), ClusterSizeBin("large", 3, None)]
    result = bootstrap_cluster_sizes(
        records=records,
        component_bins=bins,
        replicates=250,
        seed=111,
    )
    assert result == bootstrap_cluster_sizes(
        records=reversed(records),
        component_bins=bins,
        replicates=250,
        seed=111,
    )
    shot = dict(result.shot_quantiles)
    component = dict(result.component_quantiles)
    proportions = {item.label: endpoint for item, endpoint in result.component_bin_proportions}
    assert shot[0.5].estimate.value == 2
    assert shot[0.9].estimate.value == pytest.approx(3.6)
    assert shot[0.5].estimable_replicates == 250
    assert (shot[0.5].interval.lower, shot[0.5].interval.upper) == (1, 4)
    assert component[0.5].estimate.value == pytest.approx(1.5)
    assert component[0.9].estimate.value == pytest.approx(3.4)
    assert (component[0.5].interval.lower, component[0.5].interval.upper) == (
        1,
        4,
    )
    assert proportions["small"].estimate.value == pytest.approx(0.75)
    assert (proportions["small"].interval.lower, proportions["small"].interval.upper) == (
        0,
        1,
    )


def test_cluster_bootstrap_zero_denominator_and_all_replicates_rule():
    partial = [
        ShotClusterRecord(0, True, {1: 1}, 1),
        ShotClusterRecord(1, False, {}, None),
    ]
    result = bootstrap_cluster_sizes(records=partial, replicates=20, seed=0)
    shot = dict(result.shot_quantiles)[0.5]
    component = dict(result.component_quantiles)[0.5]
    assert shot.estimate.value == component.estimate.value == 1
    assert shot.estimable_replicates == component.estimable_replicates == 13
    assert shot.status == component.status == "interval-not-estimable"
    assert shot.interval is component.interval is None

    empty = bootstrap_cluster_sizes(
        records=[ShotClusterRecord(0, False, {}, None)],
        replicates=20,
        seed=0,
    )
    empty_shot = dict(empty.shot_quantiles)[0.5]
    empty_component = dict(empty.component_quantiles)[0.5]
    assert empty_shot.status == empty_component.status == "not-estimable"
    assert empty_shot.estimate.value is empty_component.estimate.value is None
    assert empty_shot.estimable_replicates == empty_component.estimable_replicates == 0


def _workload_histogram():
    return {(4, 3, 3, 1): 2, (0, 0, 0, 0): 1, (6, 4, 5, 2): 1}


def test_hrlk_reconciliation_and_workload_coverage_estimands():
    summary = summarize_workload_coverage(_workload_histogram())
    assert summary.shots == 4
    assert (
        summary.original_total,
        summary.residual_total,
        summary.lane_owned_total,
        summary.committed_total,
    ) == (14, 10, 11, 4)
    assert summary.original_mean == pytest.approx(3.5)
    assert summary.residual_mean == pytest.approx(2.5)
    assert summary.workload_mean_difference == pytest.approx(-1)
    assert summary.workload_ratio.value == pytest.approx(10 / 14)
    assert summary.frontend_coverage.value == pytest.approx(4 / 14)
    assert summary.lane_owned_coverage.value == pytest.approx(4 / 11)
    assert summary.frontend_coverage.value == pytest.approx(
        1 - summary.workload_ratio.value
    )

    with pytest.raises(ValueError, match="0 <= K <= L <= H"):
        validate_hrlk_histogram({(2, 0, 1, 2): 1})
    with pytest.raises(ValueError, match="R = H - K"):
        validate_hrlk_histogram({(2, 2, 2, 1): 1})


def test_hrlk_multinomial_bootstrap_golden_and_order_invariant():
    joint = {(4, 3, 3, 1): 2, (6, 4, 5, 2): 1}
    result = bootstrap_workload_coverage(
        joint_counts=joint,
        replicates=500,
        seed=54321,
    )
    assert result == bootstrap_workload_coverage(
        joint_counts=dict(reversed(tuple(joint.items()))),
        replicates=500,
        seed=54321,
    )
    assert result.workload_ratio.interval.lower == pytest.approx(2 / 3)
    assert result.workload_ratio.interval.upper == pytest.approx(3 / 4)
    assert result.workload_mean_difference.interval.lower == pytest.approx(-2)
    assert result.workload_mean_difference.interval.upper == pytest.approx(-1)
    assert result.frontend_coverage.interval.lower == pytest.approx(1 / 4)
    assert result.frontend_coverage.interval.upper == pytest.approx(1 / 3)
    assert result.lane_owned_coverage.interval.lower == pytest.approx(1 / 3)
    assert result.lane_owned_coverage.interval.upper == pytest.approx(2 / 5)


def test_hrlk_bootstrap_zero_denominators_never_emit_nonfinite_json():
    partial = bootstrap_workload_coverage(
        joint_counts=_workload_histogram(),
        replicates=500,
        seed=54321,
    )
    assert partial.workload_ratio.estimable_replicates == 497
    assert partial.workload_ratio.status == "interval-not-estimable"
    assert partial.workload_ratio.interval is None
    assert partial.frontend_coverage.estimable_replicates == 497
    assert partial.lane_owned_coverage.estimable_replicates == 497
    assert partial.workload_mean_difference.interval.lower == pytest.approx(-1.75)
    assert partial.workload_mean_difference.interval.upper == pytest.approx(-0.25)

    zero = bootstrap_workload_coverage(
        joint_counts={(0, 0, 0, 0): 3},
        replicates=20,
        seed=1,
    )
    for endpoint in (
        zero.workload_ratio,
        zero.frontend_coverage,
        zero.lane_owned_coverage,
    ):
        assert endpoint.status == "not-estimable"
        assert endpoint.estimate.value is None
        assert endpoint.estimable_replicates == 0
        assert endpoint.interval is None
    assert zero.workload_mean_difference.interval.lower == 0
    assert zero.workload_mean_difference.interval.upper == 0
    json.dumps(asdict(zero), allow_nan=False)
