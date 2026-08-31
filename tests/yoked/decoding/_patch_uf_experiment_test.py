from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from yoked.decoding._patch_uf_experiment import (
    CHARACTERIZATION_STAGE,
    COMPONENT_SCHEMA,
    PROTOCOL_SCHEMA,
    RANGE_COUNT,
    SEED_DERIVATION,
    SHOT_SCHEMA,
    PreparedCell,
    ShotRange,
    VerifiedCollection,
    canonical_protocol_self_sha256,
    collect_prepared_range,
    derive_named_seed,
    fixed_worker_ranges,
    run_collection,
    verify_collection,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


@dataclasses.dataclass(frozen=True)
class _FakeComponent:
    component_index: int
    cluster_defect_count: int
    exact_margin: object | None = None
    event_batch_ids: tuple[int, ...] = (0,)
    event_batch_times: tuple[int, ...] = (0,)
    last_membership_event_time: int = 0
    maximum_incident_half_edge_charge: int = 1


@dataclasses.dataclass(frozen=True)
class _FakeLane:
    lane_id: int
    last_complete_batch_id: int | None
    completed_components: tuple[object, ...]
    censored_components: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True)
class _FakeShotCorrection:
    lane_outcomes: tuple[_FakeLane, ...]
    original_detector_count: int
    residual_detector_count: int
    lane_owned_detector_count: int
    committed_defect_count: int
    cluster_summary_complete: bool = True
    durable_support_count: int = 1
    durable_boundary_count: int = 1
    durable_frame_weight: int = 0

    def component_durable_decision(
        self, lane_id: int, component_index: int
    ) -> tuple[bool, str]:
        assert lane_id == 0
        assert component_index == 0
        return True, "committed"


class _FakeSampler:
    def __init__(self, seed: int):
        self.seed = seed
        self.calls = 0

    def sample(self, *, shots: int, separate_observables: bool, bit_packed: bool):
        assert separate_observables is True
        assert bit_packed is True
        self.calls += 1
        rng = np.random.default_rng(self.seed)
        detectors = rng.integers(0, 256, size=(shots, 2), dtype=np.uint8)
        detectors[:, 0] |= 1
        detectors[:, 1] &= 1  # Nine detector bits; reject unused tail noise.
        observables = np.bitwise_and(detectors[:, :1], np.uint8(7))
        return detectors, observables


class _FakeCircuit:
    def __init__(self):
        self.seeds: list[int] = []
        self.samplers: list[_FakeSampler] = []

    def compile_detector_sampler(self, *, seed: int):
        self.seeds.append(seed)
        sampler = _FakeSampler(seed)
        self.samplers.append(sampler)
        return sampler


class _FakeGlobal:
    def decode_batch(
        self,
        packed: np.ndarray,
        *,
        bit_packed_shots: bool,
        bit_packed_predictions: bool,
    ) -> np.ndarray:
        assert bit_packed_shots and bit_packed_predictions
        return np.bitwise_and(packed[:, :1], np.uint8(7))


class _FakePassThrough:
    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        return np.bitwise_and(bit_packed_detection_event_data[:, :1], np.uint8(7))


class _FakeTreatment:
    def __init__(self, *, bad_component: bool = False):
        self.bad_component = bad_component

    @staticmethod
    def _prediction(packed: np.ndarray) -> np.ndarray:
        prediction = np.bitwise_and(packed[:, :1], np.uint8(7)).copy()
        prediction[:, 0] ^= np.right_shift(
            np.bitwise_and(packed[:, 0], np.uint8(8)), 3
        )
        return prediction

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        return self._prediction(bit_packed_detection_event_data)

    def decode_shots_bit_packed_with_telemetry(
        self, *, bit_packed_detection_event_data: np.ndarray
    ):
        packed = bit_packed_detection_event_data
        corrections = []
        unpacked = np.unpackbits(packed, axis=1, count=9, bitorder="little")
        for row in unpacked:
            original = int(np.count_nonzero(row))
            component: object
            if self.bad_component:
                component = {
                    "component_index": 0,
                    "cluster_defect_count": 1,
                    # Normative event/charge fields deliberately omitted.
                }
            else:
                component = _FakeComponent(0, 1)
            lanes = (
                _FakeLane(0, 0, (component,)),
                *tuple(_FakeLane(index, 0, ()) for index in range(1, 12)),
            )
            corrections.append(
                _FakeShotCorrection(
                    lane_outcomes=lanes,
                    original_detector_count=original,
                    residual_detector_count=original - 1,
                    lane_owned_detector_count=original,
                    committed_defect_count=1,
                )
            )
        return self._prediction(packed), tuple(corrections)


def _provenance():
    return {
        "circuit_sha256": "01" * 32,
        "dem_sha256": "02" * 32,
        "num_detectors": 9,
        "num_observables": 3,
    }


def _protocol(*, stage: str = CHARACTERIZATION_STAGE, shots: int = 32):
    value = {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "status": "DRAFT",
        "frozen": False,
        "experiment_id": "11" * 32,
        "source_identity": {
            "implementation_commit": "a" * 40,
            "config_commit": "b" * 40,
        },
        "selected_cell": {
            "cell_id": "d7-n6-y2-r28-p0.003",
            "d": 7,
            "r": 28,
            "p": 0.003,
            "patches": 6,
            "yokes": 2,
            "style": "cz",
            "noise": "si1000",
            "remove_x_yoke": False,
            "provenance": _provenance(),
        },
        "sampling": {
            "range_count": 32,
            "seed_derivation": SEED_DERIVATION,
            "stages": {stage: {"shots": shots, "seed_root": "22" * 32}},
        },
        "dem_options": {
            "decompose_errors": True,
            "approximate_disjoint_errors": True,
        },
        "decoder": {"fake": True},
        "collection_limits": {
            "expected_lanes_per_shot": 12,
            "maximum_component_records_per_shot": 8,
            "maximum_metric_bytes_per_range": 1_000_000,
        },
    }
    value["protocol_self_sha256"] = canonical_protocol_self_sha256(value)
    return value


def _prepared(*, bad_component: bool = False):
    return PreparedCell(
        cell=_protocol()["selected_cell"],
        circuit=_FakeCircuit(),
        dem=object(),
        global_decoder=_FakeGlobal(),
        treatment_decoder=_FakeTreatment(bad_component=bad_component),
        control_decoder=_FakePassThrough(),
        shadow_decoder=_FakePassThrough(),
        provenance=_provenance(),
    )


def test_fixed_ranges_match_the_frozen_floor_partition():
    ranges = fixed_worker_ranges(1000)
    assert len(ranges) == RANGE_COUNT
    assert ranges[0] == ShotRange(0, 0, 31)
    assert ranges[-1] == ShotRange(31, 968, 1000)
    widths = [item.shots for item in ranges]
    assert widths.count(31) == 24
    assert widths.count(32) == 8
    assert all(left.shot_stop == right.shot_start for left, right in zip(ranges, ranges[1:]))
    with pytest.raises(ValueError, match="at least 32"):
        fixed_worker_ranges(31)


def test_named_seed_has_a_fixed_golden_value_and_uses_every_identity():
    shot_range = ShotRange(3, 93, 125)
    kwargs = dict(
        seed_root="22" * 32,
        experiment_id="11" * 32,
        stage=CHARACTERIZATION_STAGE,
        cell_id="d7-n6-y2-r28-p0.003",
        shot_range=shot_range,
        purpose="stim-sample",
    )
    seed = derive_named_seed(**kwargs)
    assert seed == 1_338_884_782_362_936_335
    assert derive_named_seed(**kwargs) == seed
    assert derive_named_seed(**{**kwargs, "purpose": "probe"}) != seed
    with pytest.raises(ValueError, match="64 lowercase"):
        derive_named_seed(**{**kwargs, "seed_root": "AA" * 32})


def test_collect_prepared_range_is_deterministic_paired_and_bounded():
    protocol = _protocol()
    prepared = _prepared()
    shot_range = fixed_worker_ranges(32)[0]
    first = collect_prepared_range(
        prepared,
        protocol=protocol,
        stage=CHARACTERIZATION_STAGE,
        shot_range=shot_range,
    )
    second = collect_prepared_range(
        prepared,
        protocol=protocol,
        stage=CHARACTERIZATION_STAGE,
        shot_range=shot_range,
    )
    assert first.component_bytes == second.component_bytes
    assert first.shot_bytes == second.shot_bytes
    assert first.component_payload["schema"] == COMPONENT_SCHEMA
    assert first.shot_payload["schema"] == SHOT_SCHEMA
    assert first.component_payload["lane_count"] == 12
    assert first.component_payload["component_count"] == 1
    assert first.component_payload["shot_index"] == [[0, 0, 12, 0, 1]]
    assert first.shot_payload["shot_index"] == first.component_payload["shot_index"]
    assert first.shot_payload["component_file"]["sha256"]
    assert sum(first.shot_payload["paired_contingency"].values()) == 1
    assert first.shot_payload["hrlk_joint_histogram"][0][-1] == 1
    assert all(
        item["mismatches"] == 0
        for item in first.shot_payload["outside_timer_checks"].values()
    )
    assert len(prepared.circuit.seeds) == 2
    assert all(sampler.calls == 1 for sampler in prepared.circuit.samplers)
    assert gzip.decompress(first.component_bytes) == canonical_json_bytes(
        first.component_payload
    )


def test_collector_rejects_metrics_missing_normative_component_trace():
    with pytest.raises(ValueError, match="normative fields"):
        collect_prepared_range(
            _prepared(bad_component=True),
            protocol=_protocol(),
            stage=CHARACTERIZATION_STAGE,
            shot_range=fixed_worker_ranges(32)[0],
        )


def test_run_verify_resume_and_characterization_corpus(tmp_path: Path):
    protocol = _protocol()
    out = tmp_path / "collection"
    summary = run_collection(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        out=out,
        processes=2,
        scientific=False,
        prepared=_prepared(),
    )
    assert summary["shots"] == 32
    assert summary["ranges"] == 32
    assert summary["lane_records"] == 384
    assert summary["component_records"] == 32
    assert summary["cluster_summary"]["complete_shots"] == 32
    assert summary["control_equality"]["global_vs_uf_shadow"]["mismatches"] == 0
    assert (out / "corpus" / "detectors.bitpack").stat().st_size == 64
    assert (out / "corpus" / "observables.bitpack").stat().st_size == 32

    verified = verify_collection(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        out=out,
        processes=2,
        scientific=False,
    )
    assert isinstance(verified, VerifiedCollection)
    assert verified.summary == summary
    assert len(verified.shot_rows) == 32
    assert len(verified.lane_rows) == 384
    assert len(verified.component_rows) == 32
    assert all(
        row["adapter"]["exact_margin"] == "infinity"
        for row in verified.component_rows
    )
    assert len(verified.cluster_records) == 32
    assert all(record.completed_component_size_histogram == ((1, 1),) for record in verified.cluster_records)
    assert verified.corpus_identity == summary["corpus"]

    resumed = run_collection(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        out=out,
        processes=2,
        scientific=False,
        prepared=_prepared(),
    )
    assert resumed == summary


def test_valid_orphan_component_is_regenerated_before_shot_commit(tmp_path: Path):
    protocol = _protocol(stage="smoke", shots=32)
    prepared = _prepared()
    # Prepared cell identity is stage-independent, but its stored mapping came
    # from the characterization helper and has the same selected-cell content.
    artifact = collect_prepared_range(
        prepared,
        protocol=protocol,
        stage="smoke",
        shot_range=fixed_worker_ranges(32)[0],
    )
    out = tmp_path / "orphan"
    out.mkdir()
    (out / "protocol.json").write_bytes(canonical_json_bytes(protocol))
    component = out / artifact.shot_payload["component_file"]["path"]
    component.parent.mkdir(parents=True)
    component.write_bytes(artifact.component_bytes)

    summary = run_collection(
        protocol,
        stage="smoke",
        out=out,
        processes=2,
        scientific=False,
        prepared=prepared,
    )
    assert summary["shots"] == 32
    assert component.read_bytes() == artifact.component_bytes
    shot_path = out / "collection" / "shards" / component.name
    assert shot_path.is_file()


def test_tamper_and_missing_corpus_fail_closed_without_repair(tmp_path: Path):
    protocol = _protocol()
    out = tmp_path / "tamper"
    run_collection(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        out=out,
        processes=2,
        scientific=False,
        prepared=_prepared(),
    )
    detectors = out / "corpus" / "detectors.bitpack"
    detectors.unlink()
    with pytest.raises(ValueError, match="corpus detectors"):
        verify_collection(
            protocol,
            stage=CHARACTERIZATION_STAGE,
            out=out,
            processes=2,
            scientific=False,
        )
    assert not detectors.exists(), "read-only verification must not repair artifacts"

    # Restore in a separate complete collection, then corrupt one committed
    # component file.  The paired verifier must reject it before analysis.
    out2 = tmp_path / "tamper-gzip"
    run_collection(
        protocol,
        stage=CHARACTERIZATION_STAGE,
        out=out2,
        processes=2,
        scientific=False,
        prepared=_prepared(),
    )
    component = sorted((out2 / "collection" / "component_metrics").iterdir())[0]
    data = bytearray(component.read_bytes())
    data[-1] ^= 1
    component.write_bytes(data)
    with pytest.raises(ValueError):
        verify_collection(
            protocol,
            stage=CHARACTERIZATION_STAGE,
            out=out2,
            processes=2,
            scientific=False,
        )


def test_fixed_n_process_and_scientific_cell_guards(monkeypatch, tmp_path: Path):
    protocol = _protocol()
    monkeypatch.setenv("MAX_ERRORS", "")
    with pytest.raises(ValueError, match="must remain unset"):
        run_collection(
            protocol,
            stage=CHARACTERIZATION_STAGE,
            out=tmp_path / "max-errors",
            processes=1,
            scientific=False,
            prepared=_prepared(),
        )
    monkeypatch.delenv("MAX_ERRORS")
    with pytest.raises(ValueError, match="exceeds 32"):
        run_collection(
            protocol,
            stage=CHARACTERIZATION_STAGE,
            out=tmp_path / "too-many",
            processes=33,
            scientific=False,
            prepared=_prepared(),
        )

    scientific = _protocol(stage="engineering-shakeout", shots=1000)
    scientific["status"] = "FROZEN"
    scientific["frozen"] = True
    scientific["protocol_self_sha256"] = canonical_protocol_self_sha256(scientific)
    with pytest.raises(ValueError, match="exactly 32"):
        run_collection(
            scientific,
            stage="engineering-shakeout",
            out=tmp_path / "31p",
            processes=31,
            scientific=True,
            prepared=_prepared(),
        )
    wrong_cell = json.loads(json.dumps(scientific))
    wrong_cell["selected_cell"]["d"] = 5
    wrong_cell["protocol_self_sha256"] = canonical_protocol_self_sha256(wrong_cell)
    with pytest.raises(ValueError, match="selected cell d"):
        run_collection(
            wrong_cell,
            stage="engineering-shakeout",
            out=tmp_path / "wrong-cell",
            processes=32,
            scientific=True,
            prepared=_prepared(),
        )
