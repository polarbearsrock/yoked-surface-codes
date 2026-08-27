"""Three-arm paired batches for the native YSC ProMatch/Pinball comparison.

This module is deliberately additive.  It does not change, resume, or import
state from the frozen two-arm ProMatch campaigns.  A batch is sampled once and
the same immutable packed corpus is decoded by direct PyMatching (U0), the
registered native ProMatch policy, and Pinball V2.  Per-shot decoder records
are consumed in bounded microbatches instead of being retained for the batch.
"""

from __future__ import annotations

from collections import Counter
import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._pinball_v2 import (
    PINBALL_V2_BASIS_PROFILE,
    PINBALL_V2_GEOMETRY_PROFILE,
    PINBALL_V2_SOURCE_COMMIT,
    PINBALL_V2_STAGE_ORDER,
)
from yoked.decoding._pinball_v2_decoder import (
    PINBALL_V2_DECODER_NAME,
    PinballV2Decoder,
)
from yoked.decoding._promatch_decoder import PromatchDecoder
from yoked.decoding._promatch_experiment import (
    _atomic_json_write,
    _canonical_file_hash,
    configure_single_thread_runtime,
    current_execution_environment,
    current_software_versions,
    repository_state,
)
from yoked.decoding._promatch_stats import (
    ArrayDigest,
    BatchSpec,
    PairedContingency,
    derive_stim_batch_seed,
    digest_array,
)


GENERATOR = "yoked._yoked_memory_circuits:yoked_magic_memory_circuit"
SEED_DERIVATION = "sha256-root+stim-batch+uint64le-first8-uint64le"
LEDGER_SCHEMA = "pinball-promatch-ysc-paired-batch-v1"
ARM_ORDER = ("u0", "promatch", "pinball")
PAIR_ORDER = (
    "pinball_minus_promatch",
    "pinball_minus_u0",
    "promatch_minus_u0",
)
REPLAY_CATEGORIES = (
    "promatch_correct_pinball_wrong",
    "pinball_correct_promatch_wrong",
    "both_wrong_prediction_disagreement",
    "promatch_rollback",
    "pinball_complex",
)
REPLAY_POLICY: dict[str, Any] = {
    "categories": list(REPLAY_CATEGORIES),
    "maximum_candidate_rows_per_category_per_batch_ledger": 1,
    "selection_key": "SHA256_ASCII(cell_id:batch_id:shot_index:category)",
    "batch_selection": "lowest_selection_sha256_within_batch_and_category",
}
PROMATCH_CONFIG: dict[str, Any] = {
    "decoder_name": (
        "promatch-l1-v1-windowd-hw10-stages1234-"
        "noboundary-zeroframe-pymatching"
    ),
    "residual_hw_limit": 10,
    "domain_mode": "windowd",
    "boundary_policy": "disabled",
    "observable_policy": "zero-frame",
}
PINBALL_CONFIG: dict[str, Any] = {
    "decoder_name": PINBALL_V2_DECODER_NAME,
    "source_commit": PINBALL_V2_SOURCE_COMMIT,
    "geometry_profile": PINBALL_V2_GEOMETRY_PROFILE,
    "basis_profile": PINBALL_V2_BASIS_PROFILE,
    "stage_order": list(PINBALL_V2_STAGE_ORDER),
    "transaction": "fullhistory-patch-basis-domain-atomic",
}


@dataclasses.dataclass(frozen=True)
class PreparedCell:
    """One common physical cell with all three compiled native arms."""

    cell: dict[str, Any]
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    matcher_u0: pymatching.Matching
    compiled_promatch: Any
    compiled_pinball: Any
    provenance: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable_digest(value: ArrayDigest) -> dict[str, Any]:
    return {
        "sha256": value.sha256,
        "shape": list(value.shape),
        "dtype": value.dtype,
    }


def _counter_json(value: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _validate_config(
    value: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    result = dict(expected if value is None else value)
    if result != dict(expected):
        raise ValueError(f"{name} must be exactly the frozen native configuration")
    return result


def _validate_replay_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if set(result) != set(REPLAY_POLICY):
        raise ValueError("replay_policy has incorrect fields")
    if result["categories"] != list(REPLAY_CATEGORIES):
        raise ValueError("replay_policy has incorrect categories")
    cap = result["maximum_candidate_rows_per_category_per_batch_ledger"]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("replay cap must be a nonnegative integer")
    for key in ("selection_key", "batch_selection"):
        if result[key] != REPLAY_POLICY[key]:
            raise ValueError(f"replay_policy has incorrect {key}")
    return result


def _validate_cell(cell: Mapping[str, Any], *, verify_hashes: bool) -> None:
    required = {"cell_id", "d", "r", "p", "patches", "yokes"}
    missing = required - set(cell)
    if missing:
        raise ValueError(f"cell is missing fields {sorted(missing)}")
    cell_id = cell["cell_id"]
    if (
        not isinstance(cell_id, str)
        or not cell_id
        or cell_id in {".", ".."}
        or "/" in cell_id
        or "\\" in cell_id
    ):
        raise ValueError("cell_id is not a safe nonempty path component")
    for key in ("d", "r", "patches"):
        if isinstance(cell[key], bool) or not isinstance(cell[key], int) or cell[key] <= 0:
            raise ValueError(f"cell {key} must be a positive integer")
    if cell["r"] < 2 or cell["yokes"] not in (0, 1, 2):
        raise ValueError("cell has unsupported rounds or yoke count")
    if not isinstance(cell["p"], (int, float)) or not 0 <= cell["p"] <= 1:
        raise ValueError("cell p must lie in [0, 1]")
    optional_fixed = {
        "generator": GENERATOR,
        "noise": "si1000",
        "style": "cz",
        "remove_x_yoke": False,
    }
    for key, expected in optional_fixed.items():
        if key in cell and cell[key] != expected:
            raise ValueError(f"cell {key} must be {expected!r}")
    if verify_hashes:
        for key in (
            "circuit_sha256",
            "dem_sha256",
            "promatch_layout_fingerprint",
            "promatch_graph_fingerprint",
            "pinball_layout_fingerprint",
            "pinball_graph_fingerprint",
            "pinball_schedule_fingerprint",
        ):
            value = cell.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"cell {key} must be a SHA-256 digest")


def prepare_cell(
    cell: Mapping[str, Any],
    *,
    promatch_config: Mapping[str, Any] | None = None,
    pinball_config: Mapping[str, Any] | None = None,
    dem_options: Mapping[str, bool],
    verify_hashes: bool,
) -> PreparedCell:
    """Materialize and optionally authenticate one three-arm cell."""

    _validate_cell(cell, verify_hashes=verify_hashes)
    pm_config = _validate_config(promatch_config, PROMATCH_CONFIG, name="promatch_config")
    pb_config = _validate_config(pinball_config, PINBALL_CONFIG, name="pinball_config")
    circuit = yoked_magic_memory_circuit(
        patch_diameter=int(cell["d"]),
        rounds=int(cell["r"]),
        noise=gen.NoiseModel.si1000(float(cell["p"])),
        style="cz",
        yokes=int(cell["yokes"]),
        num_patches=int(cell["patches"]),
        remove_x_yoke=False,
    )
    dem = circuit.detector_error_model(**dict(dem_options))
    matcher_u0 = pymatching.Matching.from_detector_error_model(dem)
    matcher_u0.ensure_num_fault_ids(dem.num_observables)
    compiled_pm = PromatchDecoder(
        residual_hw_limit=pm_config["residual_hw_limit"],
        domain_mode=pm_config["domain_mode"],
        boundary_policy=pm_config["boundary_policy"],
        observable_policy=pm_config["observable_policy"],
    ).compile_decoder_for_dem(dem=dem)
    compiled_pb = PinballV2Decoder().compile_decoder_for_dem(dem=dem)
    provenance = {
        "circuit_sha256": _sha256_bytes(str(circuit).encode()),
        "dem_sha256": _sha256_bytes(str(dem).encode()),
        "promatch_layout_fingerprint": compiled_pm.graph.layout.fingerprint,
        "promatch_graph_fingerprint": compiled_pm.graph.fingerprint,
        "pinball_layout_fingerprint": compiled_pb.graph.layout.fingerprint,
        "pinball_graph_fingerprint": compiled_pb.graph.fingerprint,
        "pinball_schedule_fingerprint": compiled_pb.schedule.fingerprint,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "arms": {
            "u0": {
                "decoder": "uncorrelated-pymatching-from-common-dem",
                "construction": "pymatching.Matching.from_detector_error_model",
            },
            "promatch": dict(pm_config),
            "pinball": dict(pb_config),
        },
    }
    if verify_hashes:
        for key, actual in provenance.items():
            if key in {"num_detectors", "num_observables", "arms"}:
                continue
            if cell[key] != actual:
                raise ValueError(
                    f"cell {cell['cell_id']!r} {key} mismatch: "
                    f"protocol={cell[key]!r}, actual={actual!r}"
                )
    return PreparedCell(
        cell=dict(cell),
        circuit=circuit,
        dem=dem,
        matcher_u0=matcher_u0,
        compiled_promatch=compiled_pm,
        compiled_pinball=compiled_pb,
        provenance=provenance,
    )


def _prediction_failed(prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if prediction.shape != actual.shape:
        raise AssertionError("prediction and observable shapes differ")
    return np.any(np.bitwise_xor(prediction, actual) != 0, axis=1)


def _decode_residual(compiled: Any, residual: np.ndarray, frames: np.ndarray) -> np.ndarray:
    packed = np.packbits(residual, axis=1, bitorder="little")
    prediction = np.asarray(
        compiled.graph.matcher.decode_batch(
            packed, bit_packed_shots=True, bit_packed_predictions=True
        ),
        dtype=np.uint8,
    )
    if compiled.num_observables:
        prediction ^= np.packbits(frames, axis=1, bitorder="little")
        if compiled.num_observables % 8:
            prediction[:, -1] &= (1 << (compiled.num_observables % 8)) - 1
    return prediction


class _Telemetry:
    def __init__(self, prepared: PreparedCell) -> None:
        self.prepared = prepared
        self.common = Counter()
        self.common_hist: dict[str, Counter[Any]] = {
            "original_hw_histogram": Counter(),
        }
        self.pm = Counter()
        self.pm_hist: dict[str, Counter[Any]] = {
            key: Counter()
            for key in (
                "residual_hw_histogram",
                "original_residual_hw_joint_histogram",
                "domain_initial_hw_histogram",
                "domain_attempted_hw_histogram",
                "domain_final_hw_histogram",
                "domain_status_counts",
                "fallback_reason_counts",
                "decision_weight_histogram_float_hex",
                "xor_support_weight_histogram_float_hex",
                "committed_path_length_histogram",
            )
        }
        self.pm_attempted_stages = np.zeros(4, dtype=np.int64)
        self.pm_committed_stages = np.zeros(4, dtype=np.int64)
        self.pb = Counter()
        self.pb_hist: dict[str, Counter[Any]] = {
            key: Counter()
            for key in (
                "residual_hw_histogram",
                "original_residual_hw_joint_histogram",
                "domain_initial_hw_histogram",
                "domain_tentative_hw_histogram",
                "domain_final_hw_histogram",
                "domain_status_counts",
                "committed_edge_count_histogram",
                "tentative_edge_count_histogram",
                "observable_frame_hw_histogram",
                "physical_correction_weight_histogram",
            )
        }
        self.pb_stage_family = Counter()

    def add_common(self, original: np.ndarray) -> None:
        hw = np.count_nonzero(original, axis=1)
        self.common["shots"] += len(original)
        self.common["original_event_sum"] += int(hw.sum())
        self.common_hist["original_hw_histogram"].update(map(int, hw))
        layout = self.prepared.compiled_promatch.graph.layout
        self.common["terminal_event_sum"] += int(
            np.count_nonzero(original[:, layout.terminal_detector_ids])
        )
        self.common["yoke_event_sum"] += int(
            np.count_nonzero(original[:, layout.yoke_detector_ids])
        )

    def add_promatch(self, original: np.ndarray, residual: np.ndarray, results: tuple[Any, ...]) -> None:
        before = np.count_nonzero(original, axis=1)
        after = np.count_nonzero(residual, axis=1)
        self.pm["shots"] += len(results)
        self.pm["residual_event_sum"] += int(after.sum())
        layout = self.prepared.compiled_promatch.graph.layout
        self.pm["residual_terminal_event_sum"] += int(
            np.count_nonzero(residual[:, layout.terminal_detector_ids])
        )
        self.pm["residual_yoke_event_sum"] += int(
            np.count_nonzero(residual[:, layout.yoke_detector_ids])
        )
        self.pm_hist["residual_hw_histogram"].update(map(int, after))
        self.pm_hist["original_residual_hw_joint_histogram"].update(
            f"{int(a)},{int(b)}" for a, b in zip(before, after)
        )
        for result in results:
            self.pm_hist["decision_weight_histogram_float_hex"][float(result.decision_weight).hex()] += 1
            self.pm_hist["xor_support_weight_histogram_float_hex"][float(result.xor_support_weight).hex()] += 1
            for path in result.paths:
                self.pm_hist["committed_path_length_histogram"][len(path.edge_ids)] += 1
            activated = rollback = success = False
            for item in result.domain_stats.values():
                self.pm_hist["domain_initial_hw_histogram"][item.initial_hw] += 1
                self.pm_hist["domain_attempted_hw_histogram"][item.attempted_residual_hw] += 1
                self.pm_hist["domain_final_hw_histogram"][item.final_residual_hw] += 1
                self.pm_hist["domain_status_counts"][item.status] += 1
                if item.fallback_reason is not None:
                    self.pm_hist["fallback_reason_counts"][item.fallback_reason.value] += 1
                self.pm_attempted_stages += np.asarray(item.attempted_stage_counts)
                self.pm_committed_stages += np.asarray(item.committed_stage_counts)
                self.pm["attempted_matches"] += item.attempted_matches
                self.pm["committed_matches"] += item.committed_matches
                self.pm["boundary_added_domains"] += int(item.boundary_was_added)
                self.pm["boundary_used_domains"] += int(item.boundary_was_used)
                self.pm["boundary_discarded_domains"] += int(item.boundary_discarded_unused)
                activated |= item.status != "below-limit"
                rollback |= item.status == "rollback"
                success |= item.status == "success"
            self.pm["activated_shots"] += int(activated)
            self.pm["rollback_shots"] += int(rollback)
            self.pm["success_shots"] += int(success)

    def add_pinball(self, original: np.ndarray, residual: np.ndarray, results: tuple[Any, ...]) -> None:
        before = np.count_nonzero(original, axis=1)
        after = np.count_nonzero(residual, axis=1)
        self.pb["shots"] += len(results)
        self.pb["residual_event_sum"] += int(after.sum())
        layout = self.prepared.compiled_pinball.graph.layout
        self.pb["residual_terminal_event_sum"] += int(
            np.count_nonzero(residual[:, layout.terminal_detector_ids])
        )
        self.pb["residual_yoke_event_sum"] += int(
            np.count_nonzero(residual[:, layout.yoke_detector_ids])
        )
        self.pb_hist["residual_hw_histogram"].update(map(int, after))
        self.pb_hist["original_residual_hw_joint_histogram"].update(
            f"{int(a)},{int(b)}" for a, b in zip(before, after)
        )
        schedules = self.prepared.compiled_pinball.schedule.stages
        for result in results:
            simple_domains = 0
            committed = len(result.edge_support)
            tentative = len(result.tentative_edge_support)
            self.pb_hist["committed_edge_count_histogram"][committed] += 1
            self.pb_hist["tentative_edge_count_histogram"][tentative] += 1
            frame_hw = int(np.count_nonzero(result.observable_frame))
            self.pb_hist["observable_frame_hw_histogram"][frame_hw] += 1
            self.pb_hist["physical_correction_weight_histogram"][len(result.physical_correction)] += 1
            self.pb["committed_edge_sum"] += committed
            self.pb["tentative_edge_sum"] += tentative
            self.pb["committed_physical_correction_sum"] += len(result.physical_correction)
            self.pb["tentative_physical_correction_sum"] += len(result.tentative_physical_correction)
            self.pb["observable_frame_event_sum"] += frame_hw
            self.pb["no_commit_shots"] += int(committed == 0)
            for domain in result.domain_results.values():
                status = "complex" if domain.complex else "simple"
                self.pb_hist["domain_status_counts"][status] += 1
                self.pb_hist["domain_initial_hw_histogram"][domain.initial_hw] += 1
                self.pb_hist["domain_tentative_hw_histogram"][domain.tentative_residual_hw] += 1
                self.pb_hist["domain_final_hw_histogram"][domain.final_residual_hw] += 1
                self.pb["complex_domains"] += int(domain.complex)
                self.pb["simple_domains"] += int(not domain.complex)
                simple_domains += int(not domain.complex)
            domains = len(result.domain_results)
            self.pb["all_simple_shots"] += int(simple_domains == domains)
            self.pb["all_complex_shots"] += int(simple_domains == 0)
            self.pb["mixed_domain_shots"] += int(0 < simple_domains < domains)
            self.pb["complex_shots"] += int(result.complex)
            for schedule, count in zip(schedules, result.stage_match_counts):
                self.pb_stage_family[schedule.stage] += int(count)

    def finish(self) -> dict[str, Any]:
        common = {key: int(value) for key, value in self.common.items()}
        common.update({key: _counter_json(value) for key, value in self.common_hist.items()})
        pm = {key: int(value) for key, value in self.pm.items()}
        pm.update({key: _counter_json(value) for key, value in self.pm_hist.items()})
        pm["attempted_stage_counts"] = [int(v) for v in self.pm_attempted_stages]
        pm["committed_stage_counts"] = [int(v) for v in self.pm_committed_stages]
        pb = {key: int(value) for key, value in self.pb.items()}
        pb.update({key: _counter_json(value) for key, value in self.pb_hist.items()})
        pb["stage_match_counts_by_family"] = _counter_json(self.pb_stage_family)
        return {"common": common, "promatch": pm, "pinball": pb}


def _selection_sha256(*, cell_id: str, batch_id: int, shot_index: int, category: str) -> str:
    return _sha256_bytes(f"{cell_id}:{batch_id}:{shot_index}:{category}".encode("ascii"))


def _retain_candidate(store: dict[str, list[dict[str, Any]]], sample: dict[str, Any], *, cap: int) -> None:
    category = sample["category"]
    values = store[category]
    values.append(sample)
    values.sort(key=lambda item: item["selection_sha256"])
    del values[cap:]


def _replay_sample(
    *, prepared: PreparedCell, batch: BatchSpec, seed: int, category: str,
    offset: int, det: np.ndarray, obs: np.ndarray, u0: np.ndarray,
    pm: np.ndarray, pb: np.ndarray, promatch_rollback: bool, pinball_complex: bool,
) -> dict[str, Any]:
    shot_index = batch.shot_start + offset
    return {
        "selection_sha256": _selection_sha256(
            cell_id=prepared.cell["cell_id"], batch_id=batch.batch_id,
            shot_index=shot_index, category=category,
        ),
        "category": category,
        "batch_id": batch.batch_id,
        "shot_offset": offset,
        "shot_index": shot_index,
        "stim_seed": seed,
        "detection_events_hex": bytes(det).hex(),
        "observables_hex": bytes(obs).hex(),
        "u0_prediction_hex": bytes(u0).hex(),
        "promatch_prediction_hex": bytes(pm).hex(),
        "pinball_prediction_hex": bytes(pb).hex(),
        "promatch_rollback": promatch_rollback,
        "pinball_complex": pinball_complex,
    }


def collect_prepared_batch(
    prepared: PreparedCell,
    *,
    batch: BatchSpec,
    seed_root: str,
    experiment_id: str,
    phase: str,
    replay_policy: Mapping[str, Any] = REPLAY_POLICY,
    microbatch_size: int = 32,
) -> dict[str, Any]:
    """Collect one fixed-seed same-shot batch with bounded result retention."""

    policy = _validate_replay_policy(replay_policy)
    cap = policy["maximum_candidate_rows_per_category_per_batch_ledger"]
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size <= 0:
        raise ValueError("microbatch_size must be a positive integer")
    seed = derive_stim_batch_seed(seed_root=seed_root, batch_id=batch.batch_id)
    dets, obs = prepared.circuit.compile_detector_sampler(seed=seed).sample(
        shots=batch.shots, separate_observables=True, bit_packed=True
    )
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    dets.setflags(write=False)
    obs.setflags(write=False)
    det_digest = digest_array(dets)
    obs_digest = digest_array(obs)

    cube: Counter[str] = Counter()
    agreement = {name: Counter() for name in PAIR_ORDER}
    telemetry = _Telemetry(prepared)
    replay: dict[str, list[dict[str, Any]]] = {category: [] for category in REPLAY_CATEGORIES}

    for start in range(0, batch.shots, microbatch_size):
        stop = min(batch.shots, start + microbatch_size)
        packed = dets[start:stop]
        actual = obs[start:stop]
        u0 = np.asarray(
            prepared.matcher_u0.decode_batch(
                packed, bit_packed_shots=True, bit_packed_predictions=True
            ), dtype=np.uint8,
        )
        if prepared.dem.num_observables % 8 and u0.shape[1]:
            u0[:, -1] &= (1 << (prepared.dem.num_observables % 8)) - 1
        unpacked = np.unpackbits(
            packed, axis=1, count=prepared.dem.num_detectors, bitorder="little"
        )
        unpacked_digest = digest_array(unpacked)
        pm_residual, pm_frames, pm_results = prepared.compiled_promatch.predecode_shots(unpacked)
        if digest_array(unpacked) != unpacked_digest:
            raise AssertionError("ProMatch mutated the shared unpacked syndrome")
        if np.count_nonzero(pm_frames):
            raise AssertionError("frozen zero-frame ProMatch emitted a frame")
        pm = _decode_residual(prepared.compiled_promatch, pm_residual, pm_frames)
        pb_residual, pb_frames, pb_results = prepared.compiled_pinball.predecode_shots(unpacked)
        if digest_array(unpacked) != unpacked_digest:
            raise AssertionError("Pinball mutated the shared unpacked syndrome")
        pb = _decode_residual(prepared.compiled_pinball, pb_residual, pb_frames)
        failures = {
            "u0": _prediction_failed(u0, actual),
            "promatch": _prediction_failed(pm, actual),
            "pinball": _prediction_failed(pb, actual),
        }
        for values in zip(failures["u0"], failures["promatch"], failures["pinball"]):
            cube["".join("1" if value else "0" for value in values)] += 1
        pair_arrays = {
            "pinball_minus_promatch": (pm, pb),
            "pinball_minus_u0": (u0, pb),
            "promatch_minus_u0": (u0, pm),
        }
        for name, (left, right) in pair_arrays.items():
            equal = np.all(left == right, axis=1)
            agreement[name]["agree"] += int(np.count_nonzero(equal))
            agreement[name]["disagree"] += int(len(equal) - np.count_nonzero(equal))
        telemetry.common["promatch_pinball_both_wrong_prediction_disagreement_shots"] += int(
            np.count_nonzero(
                failures["promatch"]
                & failures["pinball"]
                & np.any(pm != pb, axis=1)
            )
        )
        telemetry.add_common(unpacked)
        telemetry.add_promatch(unpacked, pm_residual, pm_results)
        telemetry.add_pinball(unpacked, pb_residual, pb_results)

        for local in range(stop - start):
            offset = start + local
            pm_failed = bool(failures["promatch"][local])
            pb_failed = bool(failures["pinball"][local])
            predictions_differ = not np.array_equal(pm[local], pb[local])
            pm_rollback = any(
                item.status == "rollback"
                for item in pm_results[local].domain_stats.values()
            )
            pb_complex = bool(pb_results[local].complex)
            categories = []
            if not pm_failed and pb_failed:
                categories.append("promatch_correct_pinball_wrong")
            if pm_failed and not pb_failed:
                categories.append("pinball_correct_promatch_wrong")
            if pm_failed and pb_failed and predictions_differ:
                categories.append("both_wrong_prediction_disagreement")
            if pm_rollback:
                categories.append("promatch_rollback")
            if pb_complex:
                categories.append("pinball_complex")
            for category in categories:
                _retain_candidate(
                    replay,
                    _replay_sample(
                        prepared=prepared, batch=batch, seed=seed, category=category,
                        offset=offset, det=packed[local], obs=actual[local], u0=u0[local],
                        pm=pm[local], pb=pb[local], promatch_rollback=pm_rollback,
                        pinball_complex=pb_complex,
                    ),
                    cap=cap,
                )

    if digest_array(dets) != det_digest or digest_array(obs) != obs_digest:
        raise AssertionError("a decoder mutated the shared paired-shot corpus")
    complete_cube = {key: int(cube[key]) for key in (f"{i:03b}" for i in range(8))}
    failed_arrays_from_cube = None  # documents that tables below derive from the cube.
    del failed_arrays_from_cube
    def table(baseline_index: int, treatment_index: int) -> dict[str, int]:
        return dataclasses.asdict(PairedContingency(
            both_correct=sum(v for k, v in complete_cube.items() if k[baseline_index] == "0" and k[treatment_index] == "0"),
            regressions=sum(v for k, v in complete_cube.items() if k[baseline_index] == "0" and k[treatment_index] == "1"),
            recoveries=sum(v for k, v in complete_cube.items() if k[baseline_index] == "1" and k[treatment_index] == "0"),
            both_wrong=sum(v for k, v in complete_cube.items() if k[baseline_index] == "1" and k[treatment_index] == "1"),
        ))
    pairs = {
        "pinball_minus_promatch": table(1, 2),
        "pinball_minus_u0": table(0, 2),
        "promatch_minus_u0": table(0, 1),
    }
    return {
        "schema": LEDGER_SCHEMA,
        "experiment_id": experiment_id,
        "phase": phase,
        "cell_id": prepared.cell["cell_id"],
        "batch": dataclasses.asdict(batch),
        "stim_seed": seed,
        "microbatch_size": microbatch_size,
        "detectors": _jsonable_digest(det_digest),
        "observables": _jsonable_digest(obs_digest),
        "provenance": prepared.provenance,
        "correctness_cube": complete_cube,
        "pairwise_contingencies": pairs,
        "prediction_agreement": {
            key: {"agree": int(value["agree"]), "disagree": int(value["disagree"])}
            for key, value in agreement.items()
        },
        "telemetry": telemetry.finish(),
        "replay_samples": [sample for category in REPLAY_CATEGORIES for sample in replay[category]],
    }


_LEDGER_KEYS = {
    "schema", "experiment_id", "phase", "cell_id", "batch", "stim_seed",
    "microbatch_size", "detectors", "observables", "provenance",
    "correctness_cube", "pairwise_contingencies", "prediction_agreement",
    "telemetry", "replay_samples",
}
_COMMON_TELEMETRY_KEYS = {
    "shots", "original_event_sum", "terminal_event_sum", "yoke_event_sum",
    "promatch_pinball_both_wrong_prediction_disagreement_shots",
    "original_hw_histogram",
}
_PROMATCH_TELEMETRY_KEYS = {
    "shots", "residual_event_sum", "residual_terminal_event_sum",
    "residual_yoke_event_sum", "attempted_matches", "committed_matches",
    "boundary_added_domains", "boundary_used_domains", "boundary_discarded_domains",
    "activated_shots", "rollback_shots", "success_shots",
    "residual_hw_histogram", "original_residual_hw_joint_histogram",
    "domain_initial_hw_histogram", "domain_attempted_hw_histogram",
    "domain_final_hw_histogram", "domain_status_counts", "fallback_reason_counts",
    "decision_weight_histogram_float_hex", "xor_support_weight_histogram_float_hex",
    "committed_path_length_histogram", "attempted_stage_counts",
    "committed_stage_counts",
}
_PINBALL_TELEMETRY_KEYS = {
    "shots", "residual_event_sum", "residual_terminal_event_sum",
    "residual_yoke_event_sum", "committed_edge_sum", "tentative_edge_sum",
    "committed_physical_correction_sum", "tentative_physical_correction_sum",
    "observable_frame_event_sum", "no_commit_shots", "complex_domains",
    "simple_domains", "all_simple_shots", "all_complex_shots", "mixed_domain_shots",
    "complex_shots", "residual_hw_histogram",
    "original_residual_hw_joint_histogram", "domain_initial_hw_histogram",
    "domain_tentative_hw_histogram", "domain_final_hw_histogram",
    "domain_status_counts", "committed_edge_count_histogram",
    "tentative_edge_count_histogram", "observable_frame_hw_histogram",
    "physical_correction_weight_histogram", "stage_match_counts_by_family",
}


def _decode_hex(value: Any, *, width: int, name: str) -> bytes:
    if not isinstance(value, str) or value.lower() != value or len(value) != 2 * width:
        raise ValueError(f"replay {name} has invalid width or spelling")
    try:
        return bytes.fromhex(value)
    except ValueError as ex:
        raise ValueError(f"replay {name} is not hexadecimal") from ex


def _expected_tables(cube: Mapping[str, int]) -> dict[str, dict[str, int]]:
    def table(b: int, t: int) -> dict[str, int]:
        return {
            "both_correct": sum(v for k, v in cube.items() if k[b] == "0" and k[t] == "0"),
            "regressions": sum(v for k, v in cube.items() if k[b] == "0" and k[t] == "1"),
            "recoveries": sum(v for k, v in cube.items() if k[b] == "1" and k[t] == "0"),
            "both_wrong": sum(v for k, v in cube.items() if k[b] == "1" and k[t] == "1"),
        }
    return {
        "pinball_minus_promatch": table(1, 2),
        "pinball_minus_u0": table(0, 2),
        "promatch_minus_u0": table(0, 1),
    }


def _validate_additive_tree(value: Any, *, path: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{path} contains a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{path} contains a negative count")
        return
    if isinstance(value, list):
        if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value):
            raise ValueError(f"{path} contains a non-count list")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string key")
        for key, item in value.items():
            _validate_additive_tree(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-additive value")


def validate_ledger_row(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    phase: str,
    cell: Mapping[str, Any],
    batch: BatchSpec,
    seed_root: str,
    expected_provenance: Mapping[str, Any],
    replay_policy: Mapping[str, Any] = REPLAY_POLICY,
) -> None:
    """Fail closed on every field of one resumable three-arm ledger."""

    policy = _validate_replay_policy(replay_policy)
    if set(row) != _LEDGER_KEYS:
        raise ValueError("ledger has incorrect top-level fields")
    if row["schema"] != LEDGER_SCHEMA or row["experiment_id"] != experiment_id or row["phase"] != phase:
        raise ValueError("ledger protocol identity mismatch")
    if row["cell_id"] != cell["cell_id"] or row["batch"] != dataclasses.asdict(batch):
        raise ValueError("ledger cell/batch identity mismatch")
    expected_seed = derive_stim_batch_seed(seed_root=seed_root, batch_id=batch.batch_id)
    if row["stim_seed"] != expected_seed:
        raise ValueError("ledger Stim seed mismatch")
    if row["provenance"] != dict(expected_provenance):
        raise ValueError("ledger provenance mismatch")
    microbatch = row["microbatch_size"]
    if isinstance(microbatch, bool) or not isinstance(microbatch, int) or microbatch <= 0:
        raise ValueError("ledger microbatch_size is invalid")
    widths = {
        "detectors": (expected_provenance["num_detectors"] + 7) // 8,
        "observables": (expected_provenance["num_observables"] + 7) // 8,
    }
    for key, width in widths.items():
        digest = row[key]
        if not isinstance(digest, Mapping) or set(digest) != {"sha256", "shape", "dtype"}:
            raise ValueError(f"ledger {key} digest is malformed")
        if (
            not isinstance(digest["sha256"], str)
            or digest["sha256"].lower() != digest["sha256"]
            or len(digest["sha256"]) != 64
        ):
            raise ValueError(f"ledger {key} SHA-256 is malformed")
        try:
            bytes.fromhex(digest["sha256"])
        except ValueError as ex:
            raise ValueError(f"ledger {key} SHA-256 is malformed") from ex
        if digest["shape"] != [batch.shots, width] or digest["dtype"] != "|u1":
            raise ValueError(f"ledger {key} digest metadata mismatch")
    cube = row["correctness_cube"]
    cube_keys = {f"{i:03b}" for i in range(8)}
    if not isinstance(cube, Mapping) or set(cube) != cube_keys:
        raise ValueError("ledger correctness cube is malformed")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in cube.values()) or sum(cube.values()) != batch.shots:
        raise ValueError("ledger correctness cube does not reconcile")
    if row["pairwise_contingencies"] != _expected_tables(cube):
        raise ValueError("ledger paired contingencies disagree with correctness cube")
    agreements = row["prediction_agreement"]
    if not isinstance(agreements, Mapping) or set(agreements) != set(PAIR_ORDER):
        raise ValueError("ledger prediction agreement has incorrect pairs/order")
    for pair in PAIR_ORDER:
        item = agreements[pair]
        if not isinstance(item, Mapping) or set(item) != {"agree", "disagree"}:
            raise ValueError("ledger prediction agreement is malformed")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in item.values()) or sum(item.values()) != batch.shots:
            raise ValueError("ledger prediction agreement does not reconcile")
        table = row["pairwise_contingencies"][pair]
        if item["disagree"] < table["regressions"] + table["recoveries"]:
            raise ValueError("prediction disagreement misses correctness-discordant shots")
    telemetry = row["telemetry"]
    if not isinstance(telemetry, Mapping) or set(telemetry) != {
        "common",
        "promatch",
        "pinball",
    }:
        raise ValueError("ledger telemetry has incorrect arms/order")
    _validate_additive_tree(telemetry, path="telemetry")
    expected_telemetry_keys = {
        "common": _COMMON_TELEMETRY_KEYS,
        "promatch": _PROMATCH_TELEMETRY_KEYS,
        "pinball": _PINBALL_TELEMETRY_KEYS,
    }
    for arm in ("common", "promatch", "pinball"):
        if set(telemetry[arm]) != expected_telemetry_keys[arm]:
            raise ValueError(f"ledger {arm} telemetry fields are incorrect")
        if telemetry[arm].get("shots") != batch.shots:
            raise ValueError(f"ledger {arm} telemetry shot count mismatch")
    if sum(telemetry["common"]["original_hw_histogram"].values()) != batch.shots:
        raise ValueError("ledger common HW histogram does not reconcile")
    for arm in ("promatch", "pinball"):
        if sum(telemetry[arm]["residual_hw_histogram"].values()) != batch.shots:
            raise ValueError(f"ledger {arm} residual HW histogram does not reconcile")
        if sum(telemetry[arm]["original_residual_hw_joint_histogram"].values()) != batch.shots:
            raise ValueError(f"ledger {arm} joint HW histogram does not reconcile")
    if len(telemetry["promatch"]["attempted_stage_counts"]) != 4 or len(telemetry["promatch"]["committed_stage_counts"]) != 4:
        raise ValueError("ledger ProMatch stage telemetry has wrong width")
    pb = telemetry["pinball"]
    if pb["all_simple_shots"] + pb["all_complex_shots"] + pb["mixed_domain_shots"] != batch.shots:
        raise ValueError("ledger Pinball shot classes do not reconcile")
    if pb["complex_shots"] != pb["all_complex_shots"] + pb["mixed_domain_shots"]:
        raise ValueError("ledger Pinball complex shots do not reconcile")
    if pb["simple_domains"] + pb["complex_domains"] != sum(pb["domain_status_counts"].values()):
        raise ValueError("ledger Pinball domain classes do not reconcile")
    replay = row["replay_samples"]
    if not isinstance(replay, list):
        raise ValueError("ledger replay_samples must be an array")
    cap = policy["maximum_candidate_rows_per_category_per_batch_ledger"]
    counts = Counter()
    identities: set[tuple[str, int]] = set()
    last_hash: dict[str, str] = {}
    det_width, obs_width = widths["detectors"], widths["observables"]
    required_sample = {
        "selection_sha256", "category", "batch_id", "shot_offset", "shot_index",
        "stim_seed", "detection_events_hex", "observables_hex", "u0_prediction_hex",
        "promatch_prediction_hex", "pinball_prediction_hex", "promatch_rollback",
        "pinball_complex",
    }
    for sample in replay:
        if not isinstance(sample, Mapping) or set(sample) != required_sample:
            raise ValueError("ledger replay sample is malformed")
        category = sample["category"]
        if category not in REPLAY_CATEGORIES:
            raise ValueError("ledger replay category is unsupported")
        offset = sample["shot_offset"]
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < batch.shots:
            raise ValueError("ledger replay offset is invalid")
        if sample["batch_id"] != batch.batch_id or sample["shot_index"] != batch.shot_start + offset or sample["stim_seed"] != expected_seed:
            raise ValueError("ledger replay identity mismatch")
        expected_selection = _selection_sha256(
            cell_id=cell["cell_id"], batch_id=batch.batch_id,
            shot_index=sample["shot_index"], category=category,
        )
        if sample["selection_sha256"] != expected_selection:
            raise ValueError("ledger replay selection hash mismatch")
        identity = (category, offset)
        if identity in identities:
            raise ValueError("ledger has duplicate replay sample")
        identities.add(identity)
        if category in last_hash and expected_selection < last_hash[category]:
            raise ValueError("ledger replay samples are not hash ordered")
        last_hash[category] = expected_selection
        counts[category] += 1
        if counts[category] > cap:
            raise ValueError("ledger exceeds replay cap")
        _decode_hex(sample["detection_events_hex"], width=det_width, name="detectors")
        actual = _decode_hex(sample["observables_hex"], width=obs_width, name="observables")
        predictions = {
            "u0": _decode_hex(sample["u0_prediction_hex"], width=obs_width, name="u0 prediction"),
            "promatch": _decode_hex(sample["promatch_prediction_hex"], width=obs_width, name="promatch prediction"),
            "pinball": _decode_hex(sample["pinball_prediction_hex"], width=obs_width, name="pinball prediction"),
        }
        failed = {key: value != actual for key, value in predictions.items()}
        if category == "promatch_correct_pinball_wrong" and not (not failed["promatch"] and failed["pinball"]):
            raise ValueError("replay is not a ProMatch win")
        if category == "pinball_correct_promatch_wrong" and not (not failed["pinball"] and failed["promatch"]):
            raise ValueError("replay is not a Pinball win")
        if category == "both_wrong_prediction_disagreement" and not (failed["promatch"] and failed["pinball"] and predictions["promatch"] != predictions["pinball"]):
            raise ValueError("replay is not a both-wrong prediction disagreement")
        if not isinstance(sample["promatch_rollback"], bool) or not isinstance(sample["pinball_complex"], bool):
            raise ValueError("replay control-flow flags are malformed")
        if category == "promatch_rollback" and not sample["promatch_rollback"]:
            raise ValueError("replay is not a ProMatch rollback")
        if category == "pinball_complex" and not sample["pinball_complex"]:
            raise ValueError("replay is not Pinball-complex")
    category_populations = {
        "promatch_correct_pinball_wrong": sum(
            value for key, value in cube.items() if key[1:] == "01"
        ),
        "pinball_correct_promatch_wrong": sum(
            value for key, value in cube.items() if key[1:] == "10"
        ),
        "both_wrong_prediction_disagreement": telemetry["common"][
            "promatch_pinball_both_wrong_prediction_disagreement_shots"
        ],
        "promatch_rollback": telemetry["promatch"]["rollback_shots"],
        "pinball_complex": telemetry["pinball"]["complex_shots"],
    }
    for category, population in category_populations.items():
        if counts[category] != min(cap, population):
            raise ValueError("ledger replay retention is incomplete")


def _collection_task(
    *, cell: Mapping[str, Any], batch: BatchSpec, promatch_config: Mapping[str, Any],
    pinball_config: Mapping[str, Any], dem_options: Mapping[str, bool],
    verify_hashes: bool, seed_root: str, experiment_id: str, phase: str,
    replay_policy: Mapping[str, Any], microbatch_size: int = 32,
    require_preload: bool = False,
) -> dict[str, Any]:
    return {
        "cell": dict(cell), "batch": dataclasses.asdict(batch),
        "promatch_config": dict(promatch_config), "pinball_config": dict(pinball_config),
        "dem_options": dict(dem_options), "verify_hashes": verify_hashes,
        "seed_root": seed_root, "experiment_id": experiment_id, "phase": phase,
        "replay_policy": dict(replay_policy), "microbatch_size": microbatch_size,
        "require_preload": require_preload,
    }


_WORKER_CACHE: dict[str, PreparedCell] = {}


def _preload_worker_cell(prepared: PreparedCell) -> None:
    """Install one authenticated cell before a fork-based worker pool starts.

    With the campaign's mandatory ``fork`` context, children inherit this
    read-mostly compiled state through copy-on-write and therefore do not each
    compile a multi-gigabyte Pinball/ProMatch cell.  The campaign owns the
    lifecycle and clears the preload after its per-cell pool exits.
    """

    if not isinstance(prepared, PreparedCell):
        raise TypeError("worker preload must be a PreparedCell")
    cell_id = str(prepared.cell["cell_id"])
    _WORKER_CACHE.clear()
    _WORKER_CACHE[cell_id] = prepared


def _clear_worker_preload() -> None:
    """Release parent-owned compiled state after a per-cell pool exits."""

    _WORKER_CACHE.clear()


def _worker_collect(task: Mapping[str, Any]) -> dict[str, Any]:
    configure_single_thread_runtime()
    cell = task["cell"]
    cache_key = str(cell["cell_id"])
    prepared = _WORKER_CACHE.get(cache_key)
    if prepared is None:
        if task.get("require_preload", False):
            raise RuntimeError(
                f"required parent preload for cell {cache_key!r} was not inherited"
            )
        _WORKER_CACHE.clear()
        prepared = prepare_cell(
            cell, promatch_config=task["promatch_config"], pinball_config=task["pinball_config"],
            dem_options=task["dem_options"], verify_hashes=task["verify_hashes"],
        )
        _WORKER_CACHE[cache_key] = prepared
    return collect_prepared_batch(
        prepared, batch=BatchSpec(**task["batch"]), seed_root=task["seed_root"],
        experiment_id=task["experiment_id"], phase=task["phase"],
        replay_policy=task["replay_policy"], microbatch_size=task["microbatch_size"],
    )


def _default_source_paths(root: Path | None = None) -> list[str]:
    del root
    return [
        "src/yoked/decoding/_pinball_promatch_experiment.py",
        "src/yoked/decoding/_pinball_v2.py",
        "src/yoked/decoding/_pinball_v2_decoder.py",
        "src/yoked/decoding/_promatch.py",
        "src/yoked/decoding/_promatch_decoder.py",
        "src/yoked/decoding/_promatch_graph.py",
        "src/yoked/decoding/_promatch_layout.py",
        "src/yoked/_yoked_memory_circuits.py",
    ]


__all__ = [
    "ARM_ORDER", "GENERATOR", "LEDGER_SCHEMA", "PAIR_ORDER", "PINBALL_CONFIG",
    "PROMATCH_CONFIG", "PreparedCell", "REPLAY_CATEGORIES", "REPLAY_POLICY",
    "SEED_DERIVATION", "_atomic_json_write", "_canonical_file_hash",
    "_clear_worker_preload", "_collection_task", "_default_source_paths",
    "_preload_worker_cell", "_worker_collect",
    "collect_prepared_batch", "configure_single_thread_runtime",
    "current_execution_environment", "current_software_versions", "prepare_cell",
    "repository_state", "validate_ledger_row",
]
