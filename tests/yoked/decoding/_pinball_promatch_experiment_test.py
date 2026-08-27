from __future__ import annotations

import copy
import dataclasses
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing

import pytest

import yoked.decoding._pinball_promatch_experiment as experiment
from yoked.decoding._pinball_promatch_experiment import (
    PINBALL_CONFIG,
    PROMATCH_CONFIG,
    REPLAY_POLICY,
    collect_prepared_batch,
    prepare_cell,
    validate_ledger_row,
)
from yoked.decoding._promatch_stats import BatchSpec, canonical_json_bytes


CELL = {
    "cell_id": "three-arm-smoke",
    "d": 3,
    "r": 3,
    "p": 0.002,
    "patches": 2,
    "yokes": 2,
}
DEM_OPTIONS = {
    "decompose_errors": True,
    "approximate_disjoint_errors": True,
}
SEED_ROOT = "12" * 32


@pytest.fixture(scope="module")
def prepared():
    return prepare_cell(
        CELL,
        promatch_config=PROMATCH_CONFIG,
        pinball_config=PINBALL_CONFIG,
        dem_options=DEM_OPTIONS,
        verify_hashes=False,
    )


@pytest.fixture(scope="module")
def batch() -> BatchSpec:
    return BatchSpec(batch_id=7, shot_start=70, shots=24)


@pytest.fixture(scope="module")
def ledger(prepared, batch):
    return collect_prepared_batch(
        prepared,
        batch=batch,
        seed_root=SEED_ROOT,
        experiment_id="three-arm-test",
        phase="smoke",
        replay_policy=REPLAY_POLICY,
        microbatch_size=5,
    )


def validate(row, prepared, batch) -> None:
    validate_ledger_row(
        row,
        experiment_id="three-arm-test",
        phase="smoke",
        cell=CELL,
        batch=batch,
        seed_root=SEED_ROOT,
        expected_provenance=prepared.provenance,
        replay_policy=REPLAY_POLICY,
    )


def test_prepared_cell_freezes_common_and_per_arm_provenance(prepared) -> None:
    provenance = prepared.provenance
    assert provenance["num_detectors"] == prepared.dem.num_detectors
    assert provenance["num_observables"] == prepared.dem.num_observables
    assert len(provenance["circuit_sha256"]) == 64
    assert len(provenance["dem_sha256"]) == 64
    assert len(provenance["promatch_layout_fingerprint"]) == 64
    assert len(provenance["promatch_graph_fingerprint"]) == 64
    assert len(provenance["pinball_layout_fingerprint"]) == 64
    assert len(provenance["pinball_graph_fingerprint"]) == 64
    assert len(provenance["pinball_schedule_fingerprint"]) == 64
    assert provenance["arms"]["promatch"] == PROMATCH_CONFIG
    assert provenance["arms"]["pinball"] == PINBALL_CONFIG
    # U0 is an independently compiled direct matching, not either treatment's
    # compiled graph object.
    assert prepared.matcher_u0 is not prepared.compiled_promatch.graph.matcher
    assert prepared.matcher_u0 is not prepared.compiled_pinball.graph.matcher


def test_prepare_cell_authenticates_all_arm_hashes(prepared) -> None:
    frozen = {**CELL, **{k: v for k, v in prepared.provenance.items() if k not in {"arms", "num_detectors", "num_observables"}}}
    authenticated = prepare_cell(
        frozen,
        promatch_config=PROMATCH_CONFIG,
        pinball_config=PINBALL_CONFIG,
        dem_options=DEM_OPTIONS,
        verify_hashes=True,
    )
    assert authenticated.provenance == prepared.provenance
    frozen["pinball_schedule_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="pinball_schedule_fingerprint mismatch"):
        prepare_cell(
            frozen,
            promatch_config=PROMATCH_CONFIG,
            pinball_config=PINBALL_CONFIG,
            dem_options=DEM_OPTIONS,
            verify_hashes=True,
        )


def test_native_configs_fail_closed() -> None:
    changed = dict(PROMATCH_CONFIG)
    changed["domain_mode"] = "fullhistory"
    with pytest.raises(ValueError, match="frozen native"):
        prepare_cell(
            CELL,
            promatch_config=changed,
            pinball_config=PINBALL_CONFIG,
            dem_options=DEM_OPTIONS,
            verify_hashes=False,
        )
    changed_pb = dict(PINBALL_CONFIG)
    changed_pb["transaction"] = "whole-shot"
    with pytest.raises(ValueError, match="frozen native"):
        prepare_cell(
            CELL,
            promatch_config=PROMATCH_CONFIG,
            pinball_config=changed_pb,
            dem_options=DEM_OPTIONS,
            verify_hashes=False,
        )


def test_collector_streams_decoder_records_in_microbatches(
    prepared, batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_pm: list[int] = []
    seen_pb: list[int] = []
    original_pm = prepared.compiled_promatch.predecode_shots
    original_pb = prepared.compiled_pinball.predecode_shots

    def pm(shots):
        seen_pm.append(len(shots))
        return original_pm(shots)

    def pb(shots):
        seen_pb.append(len(shots))
        return original_pb(shots)

    monkeypatch.setattr(prepared.compiled_promatch, "predecode_shots", pm)
    monkeypatch.setattr(prepared.compiled_pinball, "predecode_shots", pb)
    row = collect_prepared_batch(
        prepared,
        batch=batch,
        seed_root=SEED_ROOT,
        experiment_id="three-arm-test",
        phase="smoke",
        microbatch_size=7,
    )
    assert seen_pm == seen_pb == [7, 7, 7, 3]
    assert row["telemetry"]["common"]["shots"] == 24


def test_ledger_has_failure_cube_all_pairwise_tables_and_agreement(
    ledger, prepared, batch
) -> None:
    validate(ledger, prepared, batch)
    assert tuple(ledger["correctness_cube"]) == tuple(f"{i:03b}" for i in range(8))
    assert sum(ledger["correctness_cube"].values()) == batch.shots
    assert tuple(ledger["pairwise_contingencies"]) == experiment.PAIR_ORDER
    assert tuple(ledger["prediction_agreement"]) == experiment.PAIR_ORDER
    for pair in experiment.PAIR_ORDER:
        assert sum(ledger["pairwise_contingencies"][pair].values()) == batch.shots
        assert sum(ledger["prediction_agreement"][pair].values()) == batch.shots


def test_collection_is_bit_exact_for_same_seed(prepared, batch, ledger) -> None:
    repeated = collect_prepared_batch(
        prepared,
        batch=batch,
        seed_root=SEED_ROOT,
        experiment_id="three-arm-test",
        phase="smoke",
        replay_policy=REPLAY_POLICY,
        microbatch_size=5,
    )
    assert repeated == ledger


def test_sorted_json_ledger_round_trip_remains_valid(prepared, batch, ledger) -> None:
    # Scientific ledgers are serialized with sort_keys=True, which must not
    # make semantic validation depend on the insertion order of nested maps.
    reloaded = json.loads(json.dumps(ledger, sort_keys=True))
    validate(reloaded, prepared, batch)


def test_shared_unpacked_input_mutation_is_detected(
    prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = prepared.compiled_promatch.predecode_shots

    def mutating(shots):
        result = original(shots)
        shots[0, 0] ^= 1
        return result

    monkeypatch.setattr(prepared.compiled_promatch, "predecode_shots", mutating)
    with pytest.raises(AssertionError, match="ProMatch mutated"):
        collect_prepared_batch(
            prepared,
            batch=BatchSpec(batch_id=99, shot_start=0, shots=2),
            seed_root=SEED_ROOT,
            experiment_id="three-arm-test",
            phase="smoke",
            microbatch_size=1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(extra=1), "top-level"),
        (lambda row: row["correctness_cube"].update({"000": -1}), "cube"),
        (lambda row: row["pairwise_contingencies"]["pinball_minus_u0"].update(regressions=999), "contingencies"),
        (lambda row: row["prediction_agreement"]["promatch_minus_u0"].update(disagree=999), "agreement"),
        (lambda row: row["telemetry"]["pinball"].pop("complex_shots"), "telemetry fields"),
        (lambda row: row["detectors"].update(sha256="g" * 64), "SHA-256"),
    ],
)
def test_strict_ledger_validator_rejects_tampering(
    ledger, prepared, batch, mutation, message
) -> None:
    changed = copy.deepcopy(ledger)
    mutation(changed)
    with pytest.raises(ValueError, match=message):
        validate(changed, prepared, batch)


def test_replay_rows_are_bounded_deterministic_and_contain_all_predictions(
    ledger, prepared, batch
) -> None:
    validate(ledger, prepared, batch)
    counts = {category: 0 for category in experiment.REPLAY_CATEGORIES}
    previous = {category: "" for category in experiment.REPLAY_CATEGORIES}
    for sample in ledger["replay_samples"]:
        category = sample["category"]
        counts[category] += 1
        assert previous[category] <= sample["selection_sha256"]
        previous[category] = sample["selection_sha256"]
        assert set((
            "u0_prediction_hex",
            "promatch_prediction_hex",
            "pinball_prediction_hex",
        )) <= set(sample)
    cap = REPLAY_POLICY["maximum_candidate_rows_per_category_per_batch_ledger"]
    assert all(value <= cap for value in counts.values())


def test_worker_task_round_trip_uses_fixed_batch_seed(
    prepared, batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "_WORKER_CACHE", {CELL["cell_id"]: prepared})
    task = experiment._collection_task(
        cell=CELL,
        batch=batch,
        promatch_config=PROMATCH_CONFIG,
        pinball_config=PINBALL_CONFIG,
        dem_options=DEM_OPTIONS,
        verify_hashes=False,
        seed_root=SEED_ROOT,
        experiment_id="three-arm-test",
        phase="smoke",
        replay_policy=REPLAY_POLICY,
        microbatch_size=6,
    )
    row = experiment._worker_collect(task)
    assert row["stim_seed"] == experiment.derive_stim_batch_seed(
        seed_root=SEED_ROOT, batch_id=batch.batch_id
    )
    validate(row, prepared, batch)


def test_forked_worker_uses_parent_preload_without_recompiling(
    prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = BatchSpec(batch_id=101, shot_start=202, shots=2)
    task = experiment._collection_task(
        cell=CELL,
        batch=batch,
        promatch_config=PROMATCH_CONFIG,
        pinball_config=PINBALL_CONFIG,
        dem_options=DEM_OPTIONS,
        verify_hashes=False,
        seed_root=SEED_ROOT,
        experiment_id="three-arm-test",
        phase="smoke",
        replay_policy=REPLAY_POLICY,
        microbatch_size=1,
        require_preload=True,
    )

    def forbidden_prepare(*_args, **_kwargs):
        raise AssertionError("forked child recompiled a preloaded cell")

    experiment._preload_worker_cell(prepared)
    monkeypatch.setattr(experiment, "prepare_cell", forbidden_prepare)
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            initializer=experiment.configure_single_thread_runtime,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            row = executor.submit(experiment._worker_collect, task).result()
    finally:
        experiment._clear_worker_preload()
    validate(row, prepared, batch)


def test_required_preload_fails_closed_instead_of_recompiling() -> None:
    task = experiment._collection_task(
        cell=CELL,
        batch=BatchSpec(batch_id=102, shot_start=204, shots=1),
        promatch_config=PROMATCH_CONFIG,
        pinball_config=PINBALL_CONFIG,
        dem_options=DEM_OPTIONS,
        verify_hashes=False,
        seed_root=SEED_ROOT,
        experiment_id="three-arm-test",
        phase="smoke",
        replay_policy=REPLAY_POLICY,
        require_preload=True,
    )
    experiment._clear_worker_preload()
    with pytest.raises(RuntimeError, match="required parent preload"):
        experiment._worker_collect(task)


def test_fork_preload_and_fresh_compile_produce_canonical_equal_ledgers(
    prepared,
) -> None:
    batch = BatchSpec(batch_id=103, shot_start=206, shots=4)
    common = {
        "cell": CELL,
        "batch": batch,
        "promatch_config": PROMATCH_CONFIG,
        "pinball_config": PINBALL_CONFIG,
        "dem_options": DEM_OPTIONS,
        "verify_hashes": False,
        "seed_root": SEED_ROOT,
        "experiment_id": "three-arm-test",
        "phase": "smoke",
        "replay_policy": REPLAY_POLICY,
        "microbatch_size": 2,
    }
    context = multiprocessing.get_context("fork")
    experiment._preload_worker_cell(prepared)
    try:
        preload_task = experiment._collection_task(
            **common,
            require_preload=True,
        )
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            preload_row = executor.submit(
                experiment._worker_collect,
                preload_task,
            ).result()
    finally:
        experiment._clear_worker_preload()

    fresh_task = experiment._collection_task(
        **common,
        require_preload=False,
    )
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        fresh_row = executor.submit(
            experiment._worker_collect,
            fresh_task,
        ).result()

    assert canonical_json_bytes(preload_row) == canonical_json_bytes(fresh_row)


def test_batch_spec_is_serialized_exactly(batch) -> None:
    assert dataclasses.asdict(batch) == {
        "batch_id": 7,
        "shot_start": 70,
        "shots": 24,
    }
