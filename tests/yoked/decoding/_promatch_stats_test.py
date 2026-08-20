import hashlib
import json
import math
import struct

import numpy as np
import pytest

from yoked.decoding._promatch_stats import (
    BatchSpec,
    PairedContingency,
    canonical_json_bytes,
    clopper_pearson_lower,
    clopper_pearson_upper,
    confirmatory_sample_size,
    derive_stim_batch_seed,
    digest_array,
    empirical_type7_quantile,
    hierarchical_timing_bootstrap,
    manifest_experiment_id,
    paired_geometric_mean_ratio,
    paired_workload_bootstrap,
    paired_workload_histogram_bootstrap,
    paired_workload_ratio,
    simulate_tango_noninferiority_power,
    tango_paired_risk_difference_upper,
    validate_batch_schedule,
    validate_process_count,
    validate_protocol_manifest,
)


def test_paired_contingency_known_table_and_swap_symmetry():
    table = PairedContingency.from_failures(
        baseline_failed=[False, False, True, True, False],
        treatment_failed=[False, True, False, True, False],
    )
    assert table == PairedContingency(
        both_correct=2,
        regressions=1,
        recoveries=1,
        both_wrong=1,
    )
    assert table.shots == 5
    assert table.baseline_failures == table.treatment_failures == 2
    assert table.delta == 0

    asymmetric = PairedContingency(80, 12, 3, 5)
    swapped = PairedContingency(80, 3, 12, 5)
    assert asymmetric.delta == -swapped.delta


def test_paired_contingency_rejects_bad_counts_and_shapes():
    with pytest.raises(ValueError, match="nonnegative"):
        PairedContingency(1, -1, 0, 0)
    with pytest.raises(TypeError, match="integer"):
        PairedContingency(True, 0, 0, 0)
    with pytest.raises(ValueError, match="identical"):
        PairedContingency.from_failures(
            baseline_failed=[False], treatment_failed=[False, True]
        )
    with pytest.raises(ValueError, match="empty"):
        _ = PairedContingency(0, 0, 0, 0).delta


def test_clopper_pearson_known_values_and_boundaries():
    assert clopper_pearson_upper(successes=0, trials=100) == pytest.approx(
        0.029513049607039932
    )
    assert clopper_pearson_upper(successes=100, trials=100) == 1
    assert clopper_pearson_lower(successes=0, trials=100) == 0
    assert clopper_pearson_lower(successes=100, trials=100) == pytest.approx(
        0.05 ** (1 / 100)
    )
    assert clopper_pearson_upper(successes=10, trials=100) < clopper_pearson_upper(
        successes=11, trials=100
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        clopper_pearson_upper(successes=2, trials=1)


def test_confirmatory_sample_size_is_exactly_rounded_and_capped():
    design = confirmatory_sample_size(
        pilot_discordant=100,
        pilot_shots=200_000,
        delta_noninferiority=0.0001,
    )
    assert design.discordance_upper == pytest.approx(0.0005903696766590368)
    assert design.raw_shots == 620_327
    assert design.rounded_shots == 630_000
    assert design.fits_resource_cap
    capped = confirmatory_sample_size(
        pilot_discordant=100,
        pilot_shots=200_000,
        delta_noninferiority=0.0001,
        max_shots=620_000,
    )
    assert not capped.fits_resource_cap


@pytest.mark.parametrize(
    "table, expected",
    [
        (PairedContingency(980, 10, 10, 0), 0.009533434250454312),
        (PairedContingency(970, 20, 10, 0), 0.021817602464888862),
        (PairedContingency(90, 0, 10, 0), -0.05522913706060792),
        (PairedContingency(90, 10, 0, 0), 0.1743656615049395),
    ],
)
def test_tango_fixed_tables(table: PairedContingency, expected: float):
    assert tango_paired_risk_difference_upper(table) == pytest.approx(
        expected, abs=1e-11
    )


def test_tango_swap_relation_and_boundary_behavior():
    table = PairedContingency(960, 25, 10, 5)
    swapped = PairedContingency(960, 10, 25, 5)
    upper = tango_paired_risk_difference_upper(table)
    swapped_upper = tango_paired_risk_difference_upper(swapped)
    # Both estimates lie the same positive distance below their respective
    # upper bounds for a symmetric table swap only at delta=0; in general the
    # useful invariant is ordering around the sign-flipped point estimate.
    assert upper > table.delta
    assert swapped_upper > swapped.delta
    assert upper > swapped_upper
    z = 1.959963984540054
    assert tango_paired_risk_difference_upper(
        PairedContingency(100, 0, 0, 0)
    ) == pytest.approx(z * z / (100 + z * z), abs=1e-12)


def test_tango_exact_small_multinomial_coverage():
    # Exhaustive coverage calculation, avoiding Monte Carlo flakiness.  Here
    # p(regression)=p(recovery)=0.15 and the true Delta is zero.
    n = 20
    p_regression = p_recovery = 0.15
    coverage = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            other = n - b - c
            probability = (
                math.factorial(n)
                / math.factorial(b)
                / math.factorial(c)
                / math.factorial(other)
                * p_regression**b
                * p_recovery**c
                * (1 - p_regression - p_recovery) ** other
            )
            table = PairedContingency(other, b, c, 0)
            if tango_paired_risk_difference_upper(table) >= 0:
                coverage += probability
    assert coverage == pytest.approx(0.9758662300820893)
    assert coverage >= 0.975


def test_simulated_power_is_seeded_and_reports_exact_lower_bound():
    kwargs = dict(
        shots=500,
        discordance_probability=0.1,
        delta_noninferiority=0.04,
        replicates=200,
        seed=1234,
    )
    first = simulate_tango_noninferiority_power(**kwargs)
    second = simulate_tango_noninferiority_power(**kwargs)
    assert first == second
    assert first.trials == 200
    assert first.passes <= first.trials
    assert first.lower_bound <= first.estimate


def test_process_cap_is_fail_closed():
    assert validate_process_count(1) == 1
    assert validate_process_count(32) == 32
    with pytest.raises(ValueError, match="cap"):
        validate_process_count(33)
    with pytest.raises(ValueError, match="immutable"):
        validate_process_count(64, cap=64)
    with pytest.raises(ValueError, match="positive"):
        validate_process_count(0)


def test_counter_seed_matches_literal_sha256_rule():
    root = bytes(32)
    expected = int.from_bytes(
        hashlib.sha256(root + b"stim-batch" + struct.pack("<Q", 0)).digest()[:8],
        "little",
    )
    assert expected == 14_925_846_306_565_298_278
    assert derive_stim_batch_seed(seed_root=root, batch_id=0) == expected
    assert (
        derive_stim_batch_seed(seed_root="00" * 32, batch_id=1)
        == 1_582_663_946_662_125_067
    )
    with pytest.raises(ValueError, match="64 hexadecimal"):
        derive_stim_batch_seed(seed_root="00", batch_id=0)


def test_array_digest_is_content_order_shape_and_dtype_explicit():
    base = np.arange(12, dtype=np.uint8).reshape(3, 4)
    noncontiguous = base[:, ::-1]
    copied = np.ascontiguousarray(noncontiguous)
    assert digest_array(noncontiguous) == digest_array(copied)
    digest = digest_array(base)
    assert digest.shape == (3, 4)
    assert digest.dtype == "|u1"
    assert len(digest.sha256) == 64


def test_batch_schedule_rejects_duplicates_gaps_overlaps_and_wrong_total():
    valid = validate_batch_schedule(
        [
            BatchSpec(batch_id=10, shot_start=0, shots=10_000),
            BatchSpec(batch_id=11, shot_start=10_000, shots=2_345),
        ],
        expected_shots=12_345,
    )
    assert valid[-1].shot_stop == 12_345
    with pytest.raises(ValueError, match="duplicate"):
        validate_batch_schedule([BatchSpec(1, 0, 10_000), BatchSpec(1, 10_000, 1)])
    with pytest.raises(ValueError, match="gap"):
        validate_batch_schedule([BatchSpec(1, 1, 1)])
    with pytest.raises(ValueError, match="overlap"):
        validate_batch_schedule([BatchSpec(1, 0, 10_000), BatchSpec(2, 9_999, 1)])
    with pytest.raises(ValueError, match="expected"):
        validate_batch_schedule([BatchSpec(1, 0, 10)], expected_shots=11)


def _sample_manifest():
    return {
        "schema_version": 1,
        "clean_worktree": True,
        "repository_commit": "a" * 40,
        "stim_version": "1.16.0",
        "sample_batch_size": 10_000,
        "processes": 32,
        "sampler_seed_roots": {
            "pilot": "11" * 32,
            "holdout": "22" * 32,
        },
        "expected_shots": {"pilot": 10_001, "holdout": 10_000},
        "batch_schedules": {
            "pilot": [
                {"batch_id": 0, "shot_start": 0, "shots": 10_000},
                {"batch_id": 1, "shot_start": 10_000, "shots": 1},
            ],
            # Batch IDs are local to independently seeded corpora and may start
            # at zero in each split.
            "holdout": [{"batch_id": 0, "shot_start": 0, "shots": 10_000}],
        },
    }


def test_manifest_canonical_hash_and_self_validation():
    manifest = _sample_manifest()
    experiment_id = validate_protocol_manifest(
        manifest, required_fields=["repository_commit", "stim_version"]
    )
    reordered = dict(reversed(list(manifest.items())))
    assert canonical_json_bytes(reordered) == canonical_json_bytes(manifest)
    assert manifest_experiment_id(reordered) == experiment_id
    with_id = {**manifest, "experiment_id": experiment_id}
    assert validate_protocol_manifest(with_id) == experiment_id
    with pytest.raises(ValueError, match="expected experiment ID"):
        validate_protocol_manifest(manifest, expected_experiment_id="0" * 64)
    with pytest.raises(ValueError, match="embedded"):
        validate_protocol_manifest({**manifest, "experiment_id": "0" * 64})
    with pytest.raises(ValueError, match="missing"):
        validate_protocol_manifest(manifest, required_fields=["not_present"])


def test_manifest_rejects_noncanonical_or_integrity_breaking_values():
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_json_bytes({"bad": math.nan})
    with pytest.raises(TypeError, match="non-string"):
        canonical_json_bytes({1: "bad"})
    manifest = _sample_manifest()
    with pytest.raises(ValueError, match="cap"):
        validate_protocol_manifest({**manifest, "processes": 33})
    duplicate_roots = {
        **manifest,
        "sampler_seed_roots": {"pilot": "11" * 32, "holdout": "11" * 32},
    }
    with pytest.raises(ValueError, match="unique"):
        validate_protocol_manifest(duplicate_roots)
    broken = json.loads(json.dumps(manifest))
    broken["batch_schedules"]["pilot"][1]["shot_start"] = 9_999
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol_manifest(broken)


def test_paired_workload_ratio_and_seeded_bootstrap():
    original = np.array([0, 2, 4, 6, 8])
    residual = np.array([0, 1, 2, 3, 4])
    assert (
        paired_workload_ratio(original_events=original, residual_events=residual) == 0.5
    )
    first = paired_workload_bootstrap(
        original_events=original,
        residual_events=residual,
        replicates=1_000,
        seed=123,
    )
    second = paired_workload_bootstrap(
        original_events=original,
        residual_events=residual,
        replicates=1_000,
        seed=123,
    )
    assert first == second
    assert first.estimate == 0.5
    assert first.upper_bound == 0.5
    with pytest.raises(ValueError, match="zero"):
        paired_workload_ratio(original_events=[0, 0], residual_events=[0, 0])


def test_histogram_workload_bootstrap_is_exact_and_memory_bounded():
    histogram = {(0, 0): 5, (2, 1): 7, (4, 2): 3}
    first = paired_workload_histogram_bootstrap(
        joint_counts=histogram,
        replicates=1_001,
        seed=456,
        chunk_size=17,
    )
    second = paired_workload_histogram_bootstrap(
        joint_counts=histogram,
        replicates=1_001,
        seed=456,
        chunk_size=17,
    )
    assert first == second
    assert first.estimate == 0.5
    assert first.upper_bound == 0.5
    with pytest.raises(ValueError, match="positive"):
        paired_workload_histogram_bootstrap(
            joint_counts={(1, 1): 0}, replicates=10, seed=1
        )
    with pytest.raises(ValueError, match="zero"):
        paired_workload_histogram_bootstrap(
            joint_counts={(0, 0): 2}, replicates=10, seed=1
        )


def test_explicit_type7_quantile_and_geometric_ratio():
    values = np.array([1, 2, 3, 4], dtype=float)
    assert empirical_type7_quantile(values, 0.25) == 1.75
    assert empirical_type7_quantile(values, 0.99) == pytest.approx(
        np.quantile(values, 0.99, method="linear")
    )
    assert paired_geometric_mean_ratio(
        np.array([2.0, 8.0]), np.array([1.0, 4.0])
    ) == pytest.approx(2)


def test_hierarchical_timing_bootstrap_preserves_restart_block_clusters():
    denominator = np.arange(1, 1 + 2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    numerator = denominator * 0.8
    result = hierarchical_timing_bootstrap(
        numerator_calls=numerator,
        denominator_calls=denominator,
        replicates=250,
        seed=999,
    )
    assert result.geometric_ratio == pytest.approx(0.8)
    assert result.geometric_ratio_upper == pytest.approx(0.8)
    assert result.p99_ratio == pytest.approx(0.8)
    assert result.p99_ratio_upper == pytest.approx(0.8)
