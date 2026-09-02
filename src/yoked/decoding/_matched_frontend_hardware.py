"""Exploratory hardware-work replay for the matched frontend corpus.

This module is additive and decision-neutral.  It replays the already frozen
matched detector corpus through the existing ProMatch-style and Pinball-style
frontends, imports the already recorded Union-Find decision, and observes the
hardware proxy counters produced after each decision.  It never samples new
shots and none of its counters participate in decoding.

The modeled cycle fields are deliberately algorithm specific:

* ProMatch uses the paper-inspired raw edge/path selection-round proxy.
* Pinball reports an ideal nine-stage streaming lower bound and, separately,
  an optional offline full-history residual OR-tree depth.
* Union-Find uses the Helios-style depth proxy from
  :mod:`yoked.decoding._patch_uf_hw_proxy`: growth iterations from each
  lane's terminal event time at the caller's weight resolution, plus merge
  flooding charged by the diameter of the clusters that merge.  Diameters
  missing from older corpora are rebuilt from the retained forest edge ids.

Consequently these fields are architecture studies, not measured latency and
not interchangeable estimates of a common RTL implementation.  Transport,
clock frequency, routing, memory arbitration, L2 queueing, and residual-MWPM
latency are outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from fractions import Fraction
import hashlib
import io
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np

from yoked.decoding._artifact_io import install_bytes_atomic, load_json_strict
from yoked.decoding._patch_uf_experiment import ShotRange, fixed_worker_ranges
from yoked.decoding._patch_uf_hw_proxy import (
    UFParallelDepthAssumptions,
    derive_uf_shot_hardware_proxy,
)
from yoked.decoding._patch_uf_reference import forest_diameter_hops
from yoked.decoding._pinball_promatch_experiment import _decode_residual
from yoked.decoding._pinball_promatch_matched_accuracy import (
    MatchedCorpus,
    validate_prepared_cell,
)
from yoked.decoding._promatch_stats import canonical_json_bytes


SCHEMA = "yoked.matched-frontends-hardware-proxy-replay-v1"
ANALYSIS_SCHEMA = "yoked.matched-frontends-hardware-proxy-analysis-v2"
PROVENANCE_SCHEMA = "yoked.matched-frontends-hardware-proxy-provenance-v2"
CLAIM_STATUS = "exploratory-non-claim-bearing-dirty-tree-replay"
ARM_ORDER = ("global", "promatch", "pinball", "union_find")
FRONTEND_ARMS = ("promatch", "pinball", "union_find")
ARCHITECTURES = (
    "fully_parallel_12_lane",
    "patch_shared_6_engine",
    "fully_shared_1_engine",
)
CYCLE_BUDGETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

# Exact identities of the already published 10k matched characterization.
# These are validation anchors, not statistical targets or fresh hypotheses.
FROZEN_EXPECTED_FAILURES = {
    "global": 3413,
    "promatch": 6641,
    "pinball": 3436,
    "union_find": 3989,
}
FROZEN_EXPECTED_RESIDUAL_EVENTS = {
    "global": 7_667_494,
    "promatch": 4_622_842,
    "pinball": 7_346_187,
    "union_find": 2_906_529,
}

_PRIMARY_CYCLE_FIELDS = {
    "promatch": "promatch_cycles",
    "pinball": "pinball_stream_offline_or_cycles",
    "union_find": "union_find_cycles",
}
_MAIN_WORK_FIELDS = {
    "promatch": "promatch_cycles_fully_parallel_12_lane",
    "pinball": "pinball_fired_primitive_count",
    "union_find": "union_find_cycles_fully_parallel_12_lane",
}

# Stable analysis names, source-array names, user-facing labels, and units.
# The Pinball source array predates the terminology clarification: its
# ``activation_bit_read`` spelling counts logical primitive operands, not
# physical RAM/register-file transactions.
RAW_WORK_METRICS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "promatch": (
        (
            "selection_rounds",
            "promatch_selection_round_count",
            "Selection rounds",
            "rounds",
        ),
        (
            "sum_active_eligible_subgraph_edges_over_selection_rounds",
            "promatch_eligible_edge_check_count",
            "Sum of active eligible-subgraph edge counts over selection rounds",
            "edge-round incidences",
        ),
        (
            "stage3_invoked_selection_rounds",
            "promatch_stage3_round_count",
            "Selection rounds invoking Stage 3",
            "rounds",
        ),
        (
            "stage3_ordered_path_candidates",
            "promatch_stage3_path_candidate_count",
            "Stage-3 ordered singleton-to-active path candidates",
            "candidates",
        ),
        (
            "stage3_shortest_path_checks",
            "promatch_stage3_path_check_count",
            "Stage-3 shortest-path checks after the no-new-singleton test",
            "checks",
        ),
        (
            "peak_selection_round_proxy_work",
            "promatch_peak_selection_round_work",
            "Peak raw edge/path proxy work in one selection round",
            "proxy work items",
        ),
        (
            "rollback_domains",
            "promatch_rollback_domain_count",
            "Rolled-back patch/basis/window domains",
            "domains",
        ),
    ),
    "pinball": (
        (
            "scheduled_primitive_evaluations",
            "pinball_primitive_evaluation_count",
            "Scheduled primitive evaluations",
            "primitive evaluations",
        ),
        (
            "activation_operand_uses",
            "pinball_activation_bit_read_count",
            "Primitive activation-operand uses (not physical memory reads)",
            "logical operand uses",
        ),
        (
            "fired_primitives",
            "pinball_fired_primitive_count",
            "Fired primitives, including rolled-back work",
            "primitives",
        ),
        (
            "tentative_detector_xor_writes",
            "pinball_tentative_detector_xor_write_count",
            "Tentative detector-boundary XOR writes",
            "logical XOR writes",
        ),
        (
            "committed_detector_xor_writes",
            "pinball_committed_detector_xor_write_count",
            "Committed detector-boundary XOR writes",
            "logical XOR writes",
        ),
        (
            "tentative_physical_target_toggles",
            "pinball_tentative_physical_target_toggle_count",
            "Tentative physical-target toggles",
            "logical target toggles",
        ),
        (
            "committed_physical_target_toggles",
            "pinball_committed_physical_target_toggle_count",
            "Committed physical-target toggles",
            "logical target toggles",
        ),
        (
            "maximum_parallel_primitive_width",
            "pinball_maximum_parallel_primitive_width",
            "Maximum conflict-free primitive width in one stage slot",
            "parallel primitives",
        ),
        (
            "complex_domains",
            "pinball_complex_domain_count",
            "Complex patch/basis domains",
            "domains",
        ),
    ),
    "union_find": (
        (
            "growth_iterations",
            "union_find_growth_iteration_count",
            "Helios growth iterations (slowest lane)",
            "iterations",
        ),
        (
            "growth_depth_milli_weight_units",
            "union_find_growth_depth_milli_weight_units",
            "Growth depth, slowest lane (milli weight-units)",
            "milli-weight-units",
        ),
        (
            "maximum_forest_diameter_hops",
            "union_find_maximum_forest_diameter_hops",
            "Largest component forest diameter",
            "hops",
        ),
        (
            "merge_depth_cycles",
            "union_find_merge_depth_cycles",
            "Merge flooding cycles (slowest lane)",
            "cycles",
        ),
        (
            "synchronous_event_batch_work",
            "union_find_synchronous_event_batch_work",
            "Synchronous event batches summed across lanes (raw software work)",
            "lane-batches",
        ),
        (
            "saturated_growth_events",
            "union_find_saturated_growth_event_work",
            "Saturated growth events",
            "growth events",
        ),
        (
            "union_merge_attempts",
            "union_find_union_merge_attempt_work",
            "Union/merge attempts",
            "attempts",
        ),
        (
            "successful_union_merges",
            "union_find_successful_union_merge_work",
            "Successful union/merge operations",
            "merges",
        ),
        (
            "redundant_union_merges",
            "union_find_redundant_union_merge_work",
            "Redundant/failed union attempts",
            "attempts",
        ),
        (
            "forest_edges",
            "union_find_forest_edge_work",
            "Forest edges retained for peeling",
            "edges",
        ),
        (
            "peel_operations",
            "union_find_peel_operation_work",
            "Peel operations",
            "operations",
        ),
        (
            "completed_components",
            "union_find_completed_component_count",
            "Completed components",
            "components",
        ),
        (
            "maximum_lane_synchronous_event_batches",
            "union_find_maximum_lane_synchronous_event_batches",
            "Maximum synchronous-event batches in one lane",
            "batches",
        ),
        (
            "maximum_component_event_batches",
            "union_find_maximum_component_event_batch_count",
            "Maximum synchronous-event batches in one component",
            "batches",
        ),
        (
            "residual_boundary_update_work",
            "union_find_residual_boundary_update_work",
            "Durable residual-boundary bit updates",
            "logical bit updates",
        ),
    ),
}


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_json(value: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(dict(value)))


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "sha256": _sha256(array.tobytes(order="C")),
        "shape": [int(item) for item in array.shape],
        "dtype": array.dtype.str,
    }


def load_matched_hardware_protocol(path: str | Path) -> dict[str, Any]:
    """Load and self-authenticate the frozen matched d=7 protocol."""

    protocol = load_json_strict(Path(path), description="matched frontend protocol")
    if protocol.get("schema") != "yoked.matched-frontends-d7-p003-protocol-v1":
        raise ValueError("matched frontend protocol schema differs")
    if protocol.get("schema_version") != 1:
        raise ValueError("matched frontend protocol version differs")
    recorded = protocol.get("protocol_self_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_self_sha256", None)
    if recorded != _digest_json(unsigned):
        raise ValueError("matched frontend protocol self hash differs")
    accuracy = protocol.get("accuracy")
    if not isinstance(accuracy, Mapping):
        raise ValueError("matched frontend accuracy protocol is malformed")
    if dict(accuracy) != {
        "arm_order": list(ARM_ORDER),
        "microbatch_size": 32,
        "processes": 32,
        "ranges": 32,
        "shots": 10_000,
    }:
        raise ValueError("matched frontend accuracy protocol differs")
    if protocol.get("status") != "FROZEN" or protocol.get("frozen") is not True:
        raise ValueError("hardware replay requires the frozen matched protocol")
    return protocol


def validate_protocol_corpus(
    protocol: Mapping[str, Any], corpus: MatchedCorpus
) -> None:
    """Bind an authenticated compact corpus to the matched protocol."""

    if not isinstance(corpus, MatchedCorpus):
        raise TypeError("corpus must be a MatchedCorpus")
    source = protocol.get("source")
    cell = protocol.get("cell")
    if not isinstance(source, Mapping) or not isinstance(cell, Mapping):
        raise ValueError("matched protocol source/cell is malformed")
    expected_identity = {
        "experiment_id": source.get("patch_uf_experiment_id"),
        "protocol_self_sha256": source.get("patch_uf_protocol_self_sha256"),
        "collection_summary_payload_sha256": source.get(
            "characterization_collection_summary_payload_sha256"
        ),
        "detector_corpus_sha256": source.get("detector_corpus_sha256"),
        "observable_corpus_sha256": source.get("observables_corpus_sha256"),
        "corpus_index_payload_sha256": source.get("corpus_index_payload_sha256"),
        "shots": 10_000,
        "num_detectors": cell.get("num_detectors"),
        "num_observables": cell.get("num_observables"),
        "cell_id": cell.get("cell_id"),
    }
    for name, expected in expected_identity.items():
        if corpus.source_identity.get(name) != expected:
            raise ValueError(f"matched corpus {name} differs from protocol")
    for name in ("circuit_sha256", "dem_sha256", "num_detectors", "num_observables"):
        if corpus.source_provenance.get(name) != cell.get(name):
            raise ValueError(f"matched corpus provenance {name} differs")


def validate_verified_uf_rows(
    protocol: Mapping[str, Any],
    corpus: MatchedCorpus,
    *,
    uf_summary: Mapping[str, Any],
    uf_shot_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Bind fully verified retained UF rows to the compact matched corpus."""

    source = protocol["source"]
    expected_summary = source["characterization_collection_summary_payload_sha256"]
    if uf_summary.get("payload_sha256") != expected_summary:
        raise ValueError("verified UF summary differs from matched protocol")
    if uf_summary.get("shots") != corpus.shots:
        raise ValueError("verified UF shot count differs from matched corpus")
    if uf_summary.get("cell_id") != corpus.source_identity["cell_id"]:
        raise ValueError("verified UF cell differs from matched corpus")
    rows = tuple(uf_shot_rows)
    if len(rows) != corpus.shots:
        raise ValueError("verified UF rows do not cover the matched corpus")
    for expected_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("global_shot_id") != expected_id:
            raise ValueError("verified UF rows are not in canonical shot order")
        if row.get("global_failed") != bool(corpus.global_failures[expected_id]):
            raise ValueError("verified UF Global outcome differs from compact corpus")
        if row.get("treatment_failed") != bool(
            corpus.union_find_failures[expected_id]
        ):
            raise ValueError("verified UF treatment outcome differs from compact corpus")
    return rows


def _lane_architectures(
    lane_costs: Mapping[tuple[int, str], int], *, patches: int
) -> dict[str, int]:
    expected = {(patch, basis) for patch in range(patches) for basis in ("X", "Z")}
    if set(lane_costs) != expected:
        raise ValueError("frontend lane costs do not contain exact patch/X-Z lanes")
    costs = [int(lane_costs[key]) for key in sorted(lane_costs)]
    if any(value < 0 for value in costs):
        raise ValueError("frontend lane costs must be nonnegative")
    patch_costs = [
        int(lane_costs[(patch, "X")]) + int(lane_costs[(patch, "Z")])
        for patch in range(patches)
    ]
    return {
        "fully_parallel_12_lane": max(costs, default=0),
        "patch_shared_6_engine": max(patch_costs, default=0),
        "fully_shared_1_engine": sum(costs),
    }


def _pair_order_architectures(lane_costs: Sequence[int], *, patches: int) -> dict[str, int]:
    values = tuple(int(value) for value in lane_costs)
    if len(values) != 2 * patches or any(value < 0 for value in values):
        raise ValueError("UF lane costs must contain two nonnegative lanes per patch")
    patch_costs = [values[2 * patch] + values[2 * patch + 1] for patch in range(patches)]
    return {
        "fully_parallel_12_lane": max(values, default=0),
        "patch_shared_6_engine": max(patch_costs, default=0),
        "fully_shared_1_engine": sum(values),
    }


def _promatch_shot_metrics(result: Any, *, patches: int) -> dict[str, int]:
    lane_cycles = {(patch, basis): 0 for patch in range(patches) for basis in ("X", "Z")}
    selection_rounds = edge_checks = stage3_rounds = 0
    path_candidates = path_checks = peak_round = rollback_domains = 0
    for domain, stats in result.domain_stats.items():
        key = (int(domain.patch_id), str(domain.check_basis))
        if key not in lane_cycles:
            raise ValueError("ProMatch result contains an unexpected domain")
        proxy = stats.hardware_proxy
        if proxy is None:
            if stats.status == "below-limit":
                continue
            raise ValueError("active ProMatch domain omits opted-in hardware proxy")
        lane_cycles[key] += int(proxy.paper_cycle_proxy_total)
        selection_rounds += int(proxy.selection_rounds)
        edge_checks += sum(map(int, proxy.active_eligible_edge_checks_per_round))
        stage3_rounds += sum(map(bool, proxy.stage3_invoked_per_round))
        path_candidates += sum(map(int, proxy.stage3_path_candidates_per_round))
        path_checks += sum(map(int, proxy.stage3_path_checks_per_round))
        peak_round = max(peak_round, int(proxy.peak_round_work))
        rollback_domains += int(stats.status == "rollback")
    architecture = _lane_architectures(lane_cycles, patches=patches)
    return {
        **{f"promatch_cycles_{name}": value for name, value in architecture.items()},
        "promatch_selection_round_count": selection_rounds,
        "promatch_eligible_edge_check_count": edge_checks,
        "promatch_stage3_round_count": stage3_rounds,
        "promatch_stage3_path_candidate_count": path_candidates,
        "promatch_stage3_path_check_count": path_checks,
        "promatch_peak_selection_round_work": peak_round,
        "promatch_rollback_domain_count": rollback_domains,
    }


def _pinball_shot_metrics(result: Any, *, patches: int) -> dict[str, int]:
    stream: dict[tuple[int, str], int] = {}
    stream_offline_or: dict[tuple[int, str], int] = {}
    post_final_input_tail: dict[tuple[int, str], int] = {}
    totals = {
        "pinball_primitive_evaluation_count": 0,
        "pinball_activation_bit_read_count": 0,
        "pinball_fired_primitive_count": 0,
        "pinball_tentative_detector_xor_write_count": 0,
        "pinball_tentative_physical_target_toggle_count": 0,
        "pinball_committed_primitive_count": 0,
        "pinball_committed_detector_xor_write_count": 0,
        "pinball_committed_physical_target_toggle_count": 0,
        "pinball_scheduled_stage_slot_count": 0,
        "pinball_nonempty_stage_slot_count": 0,
        "pinball_streamed_block_count": 0,
        "pinball_maximum_parallel_primitive_width": 0,
        "pinball_maximum_residual_or_tree_depth": 0,
        "pinball_complex_domain_count": 0,
    }
    for domain, domain_result in result.domain_results.items():
        key = (int(domain.patch_id), str(domain.check_basis))
        if key in stream:
            raise ValueError("Pinball result contains a duplicate domain")
        proxy = domain_result.hardware_proxies
        if proxy is None:
            raise ValueError("Pinball domain omits opted-in hardware proxy")
        stream[key] = int(proxy.ideal_stream_cycle_lower_bound)
        stream_offline_or[key] = int(
            proxy.ideal_stream_cycle_lower_bound
            + proxy.residual_or_reduction_tree_depth
        )
        post_final_input_tail[key] = int(
            proxy.streamed_terminal_flush_block_count
            + proxy.pipeline_drain_cycle_lower_bound
        )
        for field, proxy_field in (
            ("pinball_primitive_evaluation_count", "primitive_evaluation_count"),
            ("pinball_activation_bit_read_count", "activation_bit_read_count"),
            ("pinball_fired_primitive_count", "fired_primitive_count"),
            (
                "pinball_tentative_detector_xor_write_count",
                "tentative_detector_xor_write_count",
            ),
            (
                "pinball_tentative_physical_target_toggle_count",
                "tentative_physical_target_toggle_count",
            ),
            ("pinball_committed_primitive_count", "committed_primitive_count"),
            (
                "pinball_committed_detector_xor_write_count",
                "committed_detector_xor_write_count",
            ),
            (
                "pinball_committed_physical_target_toggle_count",
                "committed_physical_target_toggle_count",
            ),
            ("pinball_scheduled_stage_slot_count", "scheduled_stage_slot_count"),
            ("pinball_nonempty_stage_slot_count", "nonempty_stage_slot_count"),
        ):
            totals[field] += int(getattr(proxy, proxy_field))
        totals["pinball_streamed_block_count"] += int(
            proxy.streamed_full_block_count
            + proxy.streamed_terminal_flush_block_count
        )
        totals["pinball_maximum_parallel_primitive_width"] = max(
            totals["pinball_maximum_parallel_primitive_width"],
            int(proxy.maximum_parallel_primitive_width),
        )
        totals["pinball_maximum_residual_or_tree_depth"] = max(
            totals["pinball_maximum_residual_or_tree_depth"],
            int(proxy.residual_or_reduction_tree_depth),
        )
        totals["pinball_complex_domain_count"] += int(domain_result.complex)
    stream_arch = _lane_architectures(stream, patches=patches)
    stream_offline_or_arch = _lane_architectures(
        stream_offline_or, patches=patches
    )
    tail_arch = _lane_architectures(post_final_input_tail, patches=patches)
    return {
        **{
            f"pinball_stream_cycles_{name}": value
            for name, value in stream_arch.items()
        },
        **{
            f"pinball_stream_offline_or_cycles_{name}": value
            for name, value in stream_offline_or_arch.items()
        },
        **{
            f"pinball_post_final_input_tail_cycles_{name}": value
            for name, value in tail_arch.items()
        },
        **totals,
    }


def _flatten_uf_lanes(shot: Mapping[str, Any], *, patches: int) -> tuple[Any, ...]:
    metrics = shot.get("adapter_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("UF shot omits adapter_metrics")
    patch_rows = metrics.get("patch_outcomes")
    if not isinstance(patch_rows, Sequence) or isinstance(patch_rows, (str, bytes)):
        raise ValueError("UF shot patch outcomes are malformed")
    indexed: dict[int, Sequence[Any]] = {}
    for patch in patch_rows:
        if not isinstance(patch, Mapping):
            raise ValueError("UF patch outcome is malformed")
        patch_id = patch.get("patch_id")
        lanes = patch.get("lane_outcomes")
        if (
            isinstance(patch_id, bool)
            or not isinstance(patch_id, int)
            or not isinstance(lanes, Sequence)
            or isinstance(lanes, (str, bytes))
            or len(lanes) != 2
            or patch_id in indexed
        ):
            raise ValueError("UF patch/lane layout is malformed")
        indexed[int(patch_id)] = lanes
    if set(indexed) != set(range(patches)):
        raise ValueError("UF shot does not contain the exact patch set")
    return tuple(lane for patch in range(patches) for lane in indexed[patch])


def _component_forest_diameter(
    component: Mapping[str, Any],
    edge_endpoints: Mapping[int, tuple[int, int | None]],
) -> int:
    raw_ids = component.get("forest_edge_ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("completed component omits forest_edge_ids")
    tree_edges: list[tuple[int, int]] = []
    vertices: set[int] = set()
    for raw in raw_ids:
        edge_id = _positive_int(raw, name="forest edge id") if raw else 0
        if edge_id not in edge_endpoints:
            raise ValueError(f"forest edge {edge_id} is not a canonical edge")
        source, target = edge_endpoints[edge_id]
        vertices.add(int(source))
        if target is None:
            continue  # true-boundary incidence: a virtual leaf, not a tree edge
        vertices.add(int(target))
        tree_edges.append((int(source), int(target)))
    absorbed = component.get("absorbed_vertex_count")
    if absorbed is None:
        absorbed = len(component.get("absorbed_vertices", ()))
    if not tree_edges:
        if int(absorbed) > 1:
            raise ValueError(
                "multi-vertex component has no correction forest edges"
            )
        return 0
    if len(vertices) != int(absorbed):
        raise ValueError(
            "correction forest endpoints do not span the absorbed vertices"
        )
    return forest_diameter_hops(vertices, tree_edges)


def annotate_forest_diameters(
    shot: Mapping[str, Any],
    edge_endpoints: Mapping[int, tuple[int, int | None]],
) -> dict[str, Any]:
    """Return a copy of a retained UF shot row with every completed component
    carrying ``forest_diameter_hops``.

    Corpora collected before the engine recorded the diameter retain each
    component's canonical ``forest_edge_ids``; the diameter is rebuilt from
    those edges' endpoints.  Values already present are kept unchanged.
    """

    result = copy.deepcopy(dict(shot))
    metrics = result.get("adapter_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("UF shot omits adapter_metrics")
    for patch in metrics.get("patch_outcomes", ()):
        for lane in patch.get("lane_outcomes", ()):
            for component in lane.get("completed_components", ()) or ():
                if "forest_diameter_hops" in component:
                    continue
                component["forest_diameter_hops"] = _component_forest_diameter(
                    component, edge_endpoints
                )
    return result


def _uf_shot_metrics(
    shot: Mapping[str, Any],
    *,
    patches: int,
    assumptions: UFParallelDepthAssumptions,
    edge_endpoints: Mapping[int, tuple[int, int | None]],
) -> dict[str, int]:
    if not isinstance(assumptions, UFParallelDepthAssumptions):
        raise TypeError("assumptions must be UFParallelDepthAssumptions")
    shot = annotate_forest_diameters(shot, edge_endpoints)
    lanes = _flatten_uf_lanes(shot, patches=patches)
    proxy = derive_uf_shot_hardware_proxy(shot, lane_rows=lanes, assumptions=assumptions)
    if proxy.global_shot_id != shot.get("global_shot_id"):
        raise ValueError("UF proxy shot identity differs")
    if proxy.lane_count != 2 * patches or proxy.patch_count != patches:
        raise ValueError("UF proxy has the wrong lane/patch count")
    lane_arch = _pair_order_architectures(
        [lane.conservative_parallel_depth_cycles for lane in proxy.lanes],
        patches=patches,
    )
    if proxy.residual_update_depth_cycles is None:
        raise ValueError("UF proxy omits residual-update depth")
    residual_depth = int(proxy.residual_update_depth_cycles)
    explicit_architecture = (
        proxy.parallel_lane_cores_per_patch_depth_cycles,
        proxy.serial_basis_patch_engines_depth_cycles,
        proxy.fully_shared_frontend_engine_depth_cycles,
    )
    if any(value is None for value in explicit_architecture):
        raise ValueError("UF telemetry omits exact per-patch architecture depth")
    architecture = dict(zip(ARCHITECTURES, map(int, explicit_architecture)))
    depth_milli = proxy.growth_depth_weight * 1000
    return {
        **{f"union_find_cycles_{name}": value for name, value in architecture.items()},
        "union_find_growth_iteration_count": int(proxy.growth_iteration_count),
        "union_find_growth_depth_milli_weight_units": int(
            depth_milli.numerator // depth_milli.denominator
        ),
        "union_find_maximum_forest_diameter_hops": int(
            proxy.maximum_forest_diameter_hops
        ),
        "union_find_merge_depth_cycles": max(
            (lane.merge_depth_cycles for lane in proxy.lanes), default=0
        ),
        "union_find_lane_core_cycles_fully_parallel_12_lane": lane_arch[
            "fully_parallel_12_lane"
        ],
        "union_find_lane_core_cycles_patch_shared_6_engine": lane_arch[
            "patch_shared_6_engine"
        ],
        "union_find_lane_core_cycles_fully_shared_1_engine": lane_arch[
            "fully_shared_1_engine"
        ],
        "union_find_residual_update_depth_cycles": residual_depth,
        "union_find_residual_boundary_update_work": int(
            proxy.residual_boundary_update_work or 0
        ),
        "union_find_active_lane_count": int(proxy.active_lane_count),
        "union_find_censored_lane_count": int(proxy.censored_lane_count),
        "union_find_synchronous_event_batch_work": int(
            proxy.synchronous_event_batch_work
        ),
        "union_find_saturated_growth_event_work": int(
            proxy.saturated_growth_event_work
        ),
        "union_find_union_merge_attempt_work": int(proxy.union_merge_attempt_work),
        "union_find_successful_union_merge_work": int(
            proxy.successful_union_merge_work
        ),
        "union_find_redundant_union_merge_work": int(
            proxy.redundant_union_merge_work
        ),
        "union_find_forest_edge_work": int(proxy.forest_edge_work),
        "union_find_peel_operation_work": int(proxy.peel_operation_work),
        "union_find_completed_component_count": int(proxy.completed_component_count),
        "union_find_censored_component_count": int(proxy.censored_component_count),
        "union_find_maximum_lane_synchronous_event_batches": int(
            proxy.maximum_lane_synchronous_event_batches
        ),
        "union_find_maximum_completed_component_defect_count": int(
            proxy.maximum_completed_component_defect_count
        ),
        "union_find_maximum_censored_component_defect_lower_bound": int(
            proxy.maximum_censored_component_defect_lower_bound
        ),
        "union_find_maximum_observed_component_defect_count": int(
            proxy.maximum_observed_component_defect_count
        ),
        "union_find_maximum_absorbed_vertex_count": int(
            proxy.maximum_absorbed_vertex_count
        ),
        "union_find_maximum_component_event_batch_count": int(
            proxy.maximum_component_event_batch_count
        ),
    }


def _failure(prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if prediction.shape != actual.shape:
        raise ValueError("prediction and actual observable shapes differ")
    return np.any(np.bitwise_xor(prediction, actual) != 0, axis=1)


def replay_hardware_range(
    prepared: Any,
    corpus: MatchedCorpus,
    uf_shot_rows: Sequence[Mapping[str, Any]],
    *,
    shot_range: ShotRange,
    uf_assumptions: UFParallelDepthAssumptions,
    microbatch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Replay one exact matched range and return decision-aligned proxies."""

    validate_prepared_cell(prepared, corpus)
    if not isinstance(uf_assumptions, UFParallelDepthAssumptions):
        raise TypeError("uf_assumptions must be UFParallelDepthAssumptions")
    edge_endpoints = canonical_edge_endpoints(prepared)
    if shot_range not in fixed_worker_ranges(corpus.shots):
        raise ValueError("hardware replay requires one exact matched range")
    microbatch_size = _positive_int(microbatch_size, name="microbatch_size")
    if len(uf_shot_rows) != corpus.shots:
        raise ValueError("UF shot rows do not cover the corpus")
    patches = _positive_int(prepared.cell["patches"], name="patches")
    parts: dict[str, list[int]] = {"shot_id": []}

    def append_metrics(values: Mapping[str, int]) -> None:
        existing = set(parts) - {"shot_id"}
        if existing and set(values) != existing:
            raise AssertionError("per-shot hardware metric fields changed during replay")
        for name, value in values.items():
            parts.setdefault(name, []).append(int(value))

    for batch_start in range(
        shot_range.shot_start, shot_range.shot_stop, microbatch_size
    ):
        batch_stop = min(shot_range.shot_stop, batch_start + microbatch_size)
        packed = corpus.detectors[batch_start:batch_stop]
        actual = corpus.actual_observables[batch_start:batch_stop]
        computed_global = np.asarray(
            prepared.matcher_u0.decode_batch(
                packed,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
        imported_global = corpus.global_predictions[batch_start:batch_stop]
        if not np.array_equal(computed_global, imported_global):
            raise ValueError("recomputed Global prediction differs from matched corpus")
        unpacked = np.unpackbits(
            packed,
            axis=1,
            count=corpus.num_detectors,
            bitorder="little",
        )
        original = unpacked.copy()
        pm_residual, pm_frames, pm_results = prepared.compiled_promatch.predecode_shots(
            unpacked,
            collect_hardware_proxies=True,
        )
        if not np.array_equal(unpacked, original):
            raise AssertionError("ProMatch mutated the shared syndrome")
        pm_prediction = _decode_residual(
            prepared.compiled_promatch, pm_residual, pm_frames
        )
        pb_residual, pb_frames, pb_results = prepared.compiled_pinball.predecode_shots(
            unpacked,
            collect_hardware_proxies=True,
        )
        if not np.array_equal(unpacked, original):
            raise AssertionError("Pinball mutated the shared syndrome")
        pb_prediction = _decode_residual(
            prepared.compiled_pinball, pb_residual, pb_frames
        )
        failures = {
            "global": _failure(computed_global, actual),
            "promatch": _failure(pm_prediction, actual),
            "pinball": _failure(pb_prediction, actual),
            "union_find": corpus.union_find_failures[batch_start:batch_stop],
        }
        if not np.array_equal(
            failures["global"], corpus.global_failures[batch_start:batch_stop]
        ):
            raise ValueError("recomputed Global failures differ from matched corpus")
        original_hw = np.count_nonzero(original, axis=1)
        pm_hw = np.count_nonzero(pm_residual, axis=1)
        pb_hw = np.count_nonzero(pb_residual, axis=1)
        uf_hw = corpus.union_find_residual_hw[batch_start:batch_stop]
        if np.any(uf_hw < 0):
            raise ValueError("matched UF residual workload is incomplete")
        for local, shot_id in enumerate(range(batch_start, batch_stop)):
            row = uf_shot_rows[shot_id]
            if row.get("global_shot_id") != shot_id:
                raise ValueError("UF shot rows are not in canonical order")
            values: dict[str, int] = {
                "global_failed": int(failures["global"][local]),
                "promatch_failed": int(failures["promatch"][local]),
                "pinball_failed": int(failures["pinball"][local]),
                "union_find_failed": int(failures["union_find"][local]),
                "original_detector_event_count": int(original_hw[local]),
                "global_residual_detector_event_count": int(original_hw[local]),
                "promatch_residual_detector_event_count": int(pm_hw[local]),
                "pinball_residual_detector_event_count": int(pb_hw[local]),
                "union_find_residual_detector_event_count": int(uf_hw[local]),
            }
            values.update(_promatch_shot_metrics(pm_results[local], patches=patches))
            values.update(_pinball_shot_metrics(pb_results[local], patches=patches))
            values.update(
                _uf_shot_metrics(
                    row,
                    patches=patches,
                    assumptions=uf_assumptions,
                    edge_endpoints=edge_endpoints,
                )
            )
            parts["shot_id"].append(shot_id)
            append_metrics(values)
    arrays = {
        name: np.ascontiguousarray(values, dtype=np.int64)
        for name, values in parts.items()
    }
    expected_ids = np.arange(
        shot_range.shot_start, shot_range.shot_stop, dtype=np.int64
    )
    if not np.array_equal(arrays["shot_id"], expected_ids):
        raise AssertionError("range output shot IDs are incomplete")
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def canonical_edge_endpoints(prepared: Any) -> dict[int, tuple[int, int | None]]:
    """Map canonical edge ids to detector endpoints from the prepared cell."""

    graph = getattr(getattr(prepared, "compiled_promatch", None), "graph", None)
    edges = getattr(graph, "edges", None)
    if edges is None:
        raise ValueError("prepared cell does not expose the canonical edge table")
    return {
        int(edge.edge_id): (int(edge.source), None if edge.target is None else int(edge.target))
        for edge in edges
    }


def maximum_canonical_edge_weight(prepared: Any) -> Fraction:
    """Exact maximum canonical edge weight of the prepared cell's graph."""

    graph = getattr(getattr(prepared, "compiled_promatch", None), "graph", None)
    edges = getattr(graph, "edges", None)
    if not edges:
        raise ValueError("prepared cell does not expose the canonical edge table")
    return max(Fraction.from_float(float(edge.weight)) for edge in edges)


def helios_assumptions(
    prepared: Any, *, weight_resolution: int
) -> UFParallelDepthAssumptions:
    """Default Helios-style assumptions at an integer weight resolution ``w_max``."""

    resolution = _positive_int(weight_resolution, name="weight_resolution")
    if resolution < 2:
        raise ValueError("weight_resolution must be at least 2")
    return UFParallelDepthAssumptions(
        growth_quantum_weight=maximum_canonical_edge_weight(prepared) / resolution
    )


_WORKER_PRELOAD: (
    tuple[Any, MatchedCorpus, tuple[Mapping[str, Any], ...], UFParallelDepthAssumptions]
    | None
) = None


def preload_hardware_replay_worker(
    prepared: Any,
    corpus: MatchedCorpus,
    uf_shot_rows: Sequence[Mapping[str, Any]],
    uf_assumptions: UFParallelDepthAssumptions,
) -> None:
    """Install parent-owned state for inheritance by explicit-fork workers."""

    global _WORKER_PRELOAD
    validate_prepared_cell(prepared, corpus)
    if not isinstance(uf_assumptions, UFParallelDepthAssumptions):
        raise TypeError("uf_assumptions must be UFParallelDepthAssumptions")
    rows = tuple(uf_shot_rows)
    if len(rows) != corpus.shots:
        raise ValueError("UF shot rows do not cover the matched corpus")
    _WORKER_PRELOAD = (prepared, corpus, rows, uf_assumptions)


def clear_hardware_replay_worker() -> None:
    global _WORKER_PRELOAD
    _WORKER_PRELOAD = None


def hardware_replay_tasks(
    corpus: MatchedCorpus, *, microbatch_size: int = 32
) -> tuple[dict[str, Any], ...]:
    microbatch_size = _positive_int(microbatch_size, name="microbatch_size")
    return tuple(
        {"range": shot_range.as_json(), "microbatch_size": microbatch_size}
        for shot_range in fixed_worker_ranges(corpus.shots)
    )


def worker_replay_hardware_range(task: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Fork-worker entry point using only inherited compiled/corpus state."""

    if _WORKER_PRELOAD is None:
        raise RuntimeError("hardware replay worker lacks inherited state")
    if not isinstance(task, Mapping) or set(task) != {"range", "microbatch_size"}:
        raise ValueError("hardware replay worker task is malformed")
    raw = task["range"]
    if not isinstance(raw, Mapping) or set(raw) != {
        "range_id",
        "shot_start",
        "shot_stop",
        "shots",
    }:
        raise ValueError("hardware replay range task is malformed")
    shot_range = ShotRange(
        range_id=int(raw["range_id"]),
        shot_start=int(raw["shot_start"]),
        shot_stop=int(raw["shot_stop"]),
    )
    if raw["shots"] != shot_range.shots:
        raise ValueError("hardware replay range size differs")
    prepared, corpus, rows, uf_assumptions = _WORKER_PRELOAD
    return replay_hardware_range(
        prepared,
        corpus,
        rows,
        shot_range=shot_range,
        uf_assumptions=uf_assumptions,
        microbatch_size=int(task["microbatch_size"]),
    )


def concatenate_hardware_ranges(
    range_results: Sequence[Mapping[str, np.ndarray]], *, shots: int
) -> dict[str, np.ndarray]:
    """Concatenate 32 range results and enforce exact shot coverage."""

    if len(range_results) != 32:
        raise ValueError("hardware replay requires exactly 32 range results")
    fields = set(range_results[0])
    if not fields or "shot_id" not in fields:
        raise ValueError("hardware range results omit shot_id")
    if any(set(row) != fields for row in range_results):
        raise ValueError("hardware range metric fields differ")
    result = {
        name: np.ascontiguousarray(
            np.concatenate([np.asarray(row[name]) for row in range_results])
        )
        for name in sorted(fields)
    }
    if not np.array_equal(result["shot_id"], np.arange(shots, dtype=np.int64)):
        raise ValueError("hardware ranges do not give canonical complete shot coverage")
    if any(array.shape != (shots,) for array in result.values()):
        raise ValueError("hardware replay array shape differs")
    for array in result.values():
        array.setflags(write=False)
    return result


def _distribution(value: np.ndarray) -> dict[str, int | float]:
    array = np.asarray(value)
    if array.ndim != 1 or not len(array):
        raise ValueError("distribution input must be a nonempty vector")
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p90": float(np.percentile(array, 90, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": int(np.max(array)),
    }


def _raw_hardware_work_distributions(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return per-shot distributions of observed work, before cycle modeling."""

    result: dict[str, Any] = {
        "definition": (
            "decision-neutral integer work counters observed after each frontend "
            "decision; these counters do not assume a clock, schedule, or common "
            "hardware architecture"
        ),
        "arms": {},
    }
    for arm, metrics in RAW_WORK_METRICS.items():
        rows: dict[str, Any] = {}
        for analysis_name, source_array, label, unit in metrics:
            values = np.asarray(arrays[source_array])
            rows[analysis_name] = {
                "label": label,
                "unit": unit,
                "source_array": source_array,
                "total_over_shots": int(np.sum(values)),
                "nonzero_shots": int(np.count_nonzero(values)),
                "per_shot_distribution": _distribution(values),
            }
        result["arms"][arm] = rows
    return result


def _histogram(value: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(np.asarray(value), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts)}


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _masked_distribution(value: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(value)[np.asarray(mask, dtype=np.bool_)]
    return {
        "shots": int(len(selected)),
        "distribution": None if not len(selected) else _distribution(selected),
    }


def _strata(
    global_failure: np.ndarray,
    treatment_failure: np.ndarray,
) -> dict[str, np.ndarray]:
    g = np.asarray(global_failure, dtype=np.bool_)
    t = np.asarray(treatment_failure, dtype=np.bool_)
    return {
        "both_correct": ~g & ~t,
        "regression": ~g & t,
        "recovery": g & ~t,
        "both_wrong": g & t,
    }


def _tail_analysis(
    metric: np.ndarray,
    *,
    global_failure: np.ndarray,
    treatment_failure: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    regressions = ~global_failure & treatment_failure
    for percentile in (90, 95, 99):
        threshold = int(np.percentile(metric, percentile, method="higher"))
        selected = metric >= threshold
        selected_shots = int(np.count_nonzero(selected))
        selected_regressions = int(np.count_nonzero(selected & regressions))
        selected_failures = int(np.count_nonzero(selected & treatment_failure))
        result[f"p{percentile}_and_above"] = {
            "inclusive_threshold": threshold,
            "shots": selected_shots,
            "corpus_fraction": selected_shots / len(metric),
            "regressions": selected_regressions,
            "regression_rate": _rate(selected_regressions, selected_shots),
            "treatment_failures": selected_failures,
            "treatment_failure_rate": _rate(selected_failures, selected_shots),
        }
    return result


def _deadline_policy_row(
    metric: np.ndarray,
    *,
    threshold: int,
    arm_failure: np.ndarray,
    global_failure: np.ndarray,
    arm_residual: np.ndarray,
    original_detector_events: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one inclusive frontend budget with strict and bypass policies."""

    if threshold < 0:
        raise ValueError("deadline threshold must be nonnegative")
    timeout = metric > threshold
    strict_failure = arm_failure | timeout
    bypass_failure = np.where(timeout, global_failure, arm_failure)
    bypass_residual = np.where(timeout, original_detector_events, arm_residual)
    original_total = int(np.sum(original_detector_events))
    bypass_total = int(np.sum(bypass_residual))
    ratio = bypass_total / original_total
    return {
        "inclusive_cycle_budget": threshold,
        "timeout_shots": int(np.count_nonzero(timeout)),
        "timeout_rate": float(np.mean(timeout)),
        "strict_failure_rate": float(np.mean(strict_failure)),
        "strict_failure_percentage": 100 * float(np.mean(strict_failure)),
        "global_bypass_failure_rate": float(np.mean(bypass_failure)),
        "global_bypass_failure_percentage": 100 * float(np.mean(bypass_failure)),
        "global_bypass_residual_detector_events": bypass_total,
        "global_bypass_residual_over_original_ratio": ratio,
        "global_bypass_detector_event_reduction_fraction": 1 - ratio,
        "global_bypass_detector_event_reduction_percentage": 100 * (1 - ratio),
    }


def _validate_hardware_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    required = {
        "shot_id",
        *(f"{arm}_failed" for arm in ARM_ORDER),
        "original_detector_event_count",
        *(f"{arm}_residual_detector_event_count" for arm in ARM_ORDER),
        *(
            source_array
            for metrics in RAW_WORK_METRICS.values()
            for _, source_array, _, _ in metrics
        ),
        *(
            f"promatch_cycles_{architecture}"
            for architecture in ARCHITECTURES
        ),
        *(
            f"pinball_stream_cycles_{architecture}"
            for architecture in ARCHITECTURES
        ),
        *(
            f"pinball_stream_offline_or_cycles_{architecture}"
            for architecture in ARCHITECTURES
        ),
        *(
            f"pinball_post_final_input_tail_cycles_{architecture}"
            for architecture in ARCHITECTURES
        ),
        *(
            f"union_find_cycles_{architecture}"
            for architecture in ARCHITECTURES
        ),
        "pinball_fired_primitive_count",
        "union_find_maximum_observed_component_defect_count",
        "union_find_maximum_completed_component_defect_count",
        "union_find_maximum_censored_component_defect_lower_bound",
        "union_find_maximum_absorbed_vertex_count",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"hardware replay arrays omit {sorted(missing)}")
    shot_id = np.asarray(arrays["shot_id"])
    if shot_id.ndim != 1 or not len(shot_id):
        raise ValueError("hardware replay shot_id is malformed")
    shots = len(shot_id)
    if not np.array_equal(shot_id, np.arange(shots, dtype=shot_id.dtype)):
        raise ValueError("hardware replay shot IDs are not canonical")
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.shape != (shots,):
            raise ValueError(f"hardware replay {name} has the wrong shape")
        if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError(f"hardware replay {name} must be nonnegative integer")
    for arm in ARM_ORDER:
        if np.any(np.asarray(arrays[f"{arm}_failed"]) > 1):
            raise ValueError(f"hardware replay {arm} failures are not binary")
    return shots


def _correctness_cube(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    failures = [np.asarray(arrays[f"{arm}_failed"], dtype=np.bool_) for arm in ARM_ORDER]
    values = sum(
        failure.astype(np.uint8) << (len(ARM_ORDER) - index - 1)
        for index, failure in enumerate(failures)
    )
    counts = np.bincount(values, minlength=1 << len(ARM_ORDER))
    return {
        f"{value:0{len(ARM_ORDER)}b}": int(counts[value])
        for value in range(1 << len(ARM_ORDER))
    }


def _pairwise_contingencies(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for baseline_index, baseline in enumerate(ARM_ORDER):
        baseline_failure = np.asarray(arrays[f"{baseline}_failed"], dtype=np.bool_)
        for treatment in ARM_ORDER[baseline_index + 1 :]:
            treatment_failure = np.asarray(
                arrays[f"{treatment}_failed"], dtype=np.bool_
            )
            result[f"{treatment}_minus_{baseline}"] = {
                "both_correct": int(
                    np.count_nonzero(~baseline_failure & ~treatment_failure)
                ),
                "regressions": int(
                    np.count_nonzero(~baseline_failure & treatment_failure)
                ),
                "recoveries": int(
                    np.count_nonzero(baseline_failure & ~treatment_failure)
                ),
                "both_wrong": int(
                    np.count_nonzero(baseline_failure & treatment_failure)
                ),
            }
    return result


def validate_reference_results(
    arrays: Mapping[str, np.ndarray],
    reference_analysis: Mapping[str, Any],
    *,
    enforce_frozen_expected: bool = True,
) -> dict[str, Any]:
    """Require replay accuracy/workload to equal the prior matched analysis."""

    shots = _validate_hardware_arrays(arrays)
    if reference_analysis.get("shots") != shots:
        raise ValueError("reference analysis shot count differs")
    accuracy = reference_analysis.get("accuracy")
    raw = reference_analysis.get("raw_inputs")
    if not isinstance(accuracy, Mapping) or not isinstance(raw, Mapping):
        raise ValueError("reference matched analysis is malformed")
    marginals = accuracy.get("marginals")
    telemetry = raw.get("telemetry")
    if not isinstance(marginals, Mapping) or not isinstance(telemetry, Mapping):
        raise ValueError("reference matched results are malformed")
    actual_failures: dict[str, int] = {}
    actual_residual: dict[str, int] = {}
    residual_histograms: dict[str, dict[str, int]] = {}
    for arm in ARM_ORDER:
        actual_failures[arm] = int(np.sum(arrays[f"{arm}_failed"]))
        actual_residual[arm] = int(
            np.sum(arrays[f"{arm}_residual_detector_event_count"])
        )
        expected_failure = marginals.get(arm, {}).get("failures")
        expected_residual = telemetry.get(arm, {}).get("residual_event_sum")
        if actual_failures[arm] != expected_failure:
            raise ValueError(f"replayed {arm} failures differ from reference")
        if actual_residual[arm] != expected_residual:
            raise ValueError(f"replayed {arm} residual workload differs from reference")
        residual_histograms[arm] = _histogram(
            np.asarray(arrays[f"{arm}_residual_detector_event_count"])
        )
        expected_histogram = telemetry.get(arm, {}).get("residual_hw_histogram")
        if expected_histogram is not None and residual_histograms[arm] != expected_histogram:
            raise ValueError(f"replayed {arm} residual histogram differs from reference")
    cube = _correctness_cube(arrays)
    expected_cube = accuracy.get("correctness_cube")
    if cube != expected_cube:
        raise ValueError("replayed four-arm correctness cube differs from reference")
    pairwise = _pairwise_contingencies(arrays)
    expected_pairwise = raw.get("pairwise_contingencies")
    if pairwise != expected_pairwise:
        raise ValueError("replayed pairwise contingencies differ from reference")
    if enforce_frozen_expected and (
        actual_failures != FROZEN_EXPECTED_FAILURES
        or actual_residual != FROZEN_EXPECTED_RESIDUAL_EVENTS
    ):
        raise ValueError("replay differs from frozen 10k validation anchors")
    return {
        "failures": actual_failures,
        "residual_events": actual_residual,
        "residual_hw_histograms_exact": True,
        "correctness_cube": cube,
        "pairwise_contingencies": pairwise,
    }


def analyze_hardware_replay(
    arrays: Mapping[str, np.ndarray],
    *,
    source_identity: Mapping[str, Any],
    reference_analysis: Mapping[str, Any],
    enforce_frozen_expected: bool = True,
) -> dict[str, Any]:
    """Summarize hardware proxies without treating them as measured latency."""

    shots = _validate_hardware_arrays(arrays)
    validation = validate_reference_results(
        arrays,
        reference_analysis,
        enforce_frozen_expected=enforce_frozen_expected,
    )
    failures = {
        arm: np.asarray(arrays[f"{arm}_failed"], dtype=np.bool_)
        for arm in ARM_ORDER
    }
    original = np.asarray(arrays["original_detector_event_count"])
    original_sum = int(np.sum(original))
    accuracy = {
        arm: {
            "failures": validation["failures"][arm],
            "shots": shots,
            "failure_rate": validation["failures"][arm] / shots,
            "failure_percentage": 100 * validation["failures"][arm] / shots,
            "difference_from_global_percentage_points": 100
            * (validation["failures"][arm] - validation["failures"]["global"])
            / shots,
        }
        for arm in ARM_ORDER
    }
    workload = {}
    for arm in ARM_ORDER:
        residual = np.asarray(arrays[f"{arm}_residual_detector_event_count"])
        residual_sum = int(np.sum(residual))
        workload[arm] = {
            "original_detector_events": original_sum,
            "residual_detector_events": residual_sum,
            "residual_over_original_ratio": residual_sum / original_sum,
            "detector_event_reduction_fraction": 1 - residual_sum / original_sum,
            "detector_event_reduction_percentage": 100
            * (1 - residual_sum / original_sum),
            "per_shot_residual_distribution": _distribution(residual),
        }

    cycles: dict[str, Any] = {
        "promatch": {
            architecture: _distribution(
                np.asarray(arrays[f"promatch_cycles_{architecture}"])
            )
            for architecture in ARCHITECTURES
        },
        "pinball": {
            "ideal_stream_cycle_lower_bound": {
                architecture: _distribution(
                    np.asarray(arrays[f"pinball_stream_cycles_{architecture}"])
                )
                for architecture in ARCHITECTURES
            },
            "stream_plus_offline_full_history_or_depth_proxy": {
                architecture: _distribution(
                    np.asarray(
                        arrays[f"pinball_stream_offline_or_cycles_{architecture}"]
                    )
                )
                for architecture in ARCHITECTURES
            },
            "post_final_input_tail_lower_bound": {
                architecture: _distribution(
                    np.asarray(
                        arrays[
                            f"pinball_post_final_input_tail_cycles_{architecture}"
                        ]
                    )
                )
                for architecture in ARCHITECTURES
            },
        },
        "union_find": {
            architecture: _distribution(
                np.asarray(arrays[f"union_find_cycles_{architecture}"])
            )
            for architecture in ARCHITECTURES
        },
    }

    deadlines: dict[str, Any] = {}
    for arm in FRONTEND_ARMS:
        prefix = _PRIMARY_CYCLE_FIELDS[arm]
        deadlines[arm] = {}
        for architecture in ARCHITECTURES:
            metric = np.asarray(arrays[f"{prefix}_{architecture}"])
            arm_residual = np.asarray(
                arrays[f"{arm}_residual_detector_event_count"]
            )
            rows: dict[str, Any] = {}
            for budget in CYCLE_BUDGETS:
                rows[str(budget)] = _deadline_policy_row(
                    metric,
                    threshold=budget,
                    arm_failure=failures[arm],
                    global_failure=failures["global"],
                    arm_residual=arm_residual,
                    original_detector_events=original,
                )
            quantile_rows: dict[str, Any] = {}
            for label, percentile in (
                ("p50", 50),
                ("p90", 90),
                ("p95", 95),
                ("p99", 99),
                ("max", 100),
            ):
                raw_quantile = float(
                    np.percentile(metric, percentile, method="linear")
                )
                threshold = int(np.ceil(raw_quantile))
                quantile_rows[label] = {
                    "source_quantile": label,
                    "raw_quantile_cycles": raw_quantile,
                    **_deadline_policy_row(
                        metric,
                        threshold=threshold,
                        arm_failure=failures[arm],
                        global_failure=failures["global"],
                        arm_residual=arm_residual,
                        original_detector_events=original,
                    ),
                }
            p99 = float(np.percentile(metric, 99, method="linear"))
            p99_ceiling = int(np.ceil(p99))
            next_power_of_two = (
                1 if p99_ceiling <= 1 else 1 << (p99_ceiling - 1).bit_length()
            )
            envelope_policy = _deadline_policy_row(
                metric,
                threshold=next_power_of_two,
                arm_failure=failures[arm],
                global_failure=failures["global"],
                arm_residual=arm_residual,
                original_detector_events=original,
            )
            deadlines[arm][architecture] = {
                "budget_rows": rows,
                "quantile_policy_rows": quantile_rows,
                "p99_envelope": {
                    "p99_cycles": p99,
                    "next_power_of_two_budget": next_power_of_two,
                    "budget_is_in_standard_table": next_power_of_two
                    in CYCLE_BUDGETS,
                    **envelope_policy,
                },
            }

    strata: dict[str, Any] = {}
    tails: dict[str, Any] = {}
    for arm in FRONTEND_ARMS:
        metric_name = _MAIN_WORK_FIELDS[arm]
        metric = np.asarray(arrays[metric_name])
        arm_strata = _strata(failures["global"], failures[arm])
        strata[arm] = {
            "main_work_metric": metric_name,
            "groups": {
                name: _masked_distribution(metric, mask)
                for name, mask in arm_strata.items()
            },
        }
        if arm == "union_find":
            component_size = np.asarray(
                arrays["union_find_maximum_observed_component_defect_count"]
            )
            for name, mask in arm_strata.items():
                strata[arm]["groups"][name]["maximum_component_defect_count"] = (
                    _masked_distribution(component_size, mask)
                )
        tails[arm] = {
            "main_work_metric": metric_name,
            "inclusive_tails": _tail_analysis(
                metric,
                global_failure=failures["global"],
                treatment_failure=failures[arm],
            ),
        }

    maximum_observed = np.asarray(
        arrays["union_find_maximum_observed_component_defect_count"]
    )
    cluster_sizes = {
        "definition": (
            "per-shot maximum defect count across completed components and "
            "censored-component lower bounds"
        ),
        "maximum_observed_component_defect_count": {
            "distribution": _distribution(maximum_observed),
            "histogram": _histogram(maximum_observed),
        },
        "maximum_completed_component_defect_count": {
            "distribution": _distribution(
                np.asarray(
                    arrays["union_find_maximum_completed_component_defect_count"]
                )
            ),
            "histogram": _histogram(
                np.asarray(
                    arrays["union_find_maximum_completed_component_defect_count"]
                )
            ),
        },
        "maximum_censored_component_defect_lower_bound": {
            "distribution": _distribution(
                np.asarray(
                    arrays[
                        "union_find_maximum_censored_component_defect_lower_bound"
                    ]
                )
            ),
            "histogram": _histogram(
                np.asarray(
                    arrays[
                        "union_find_maximum_censored_component_defect_lower_bound"
                    ]
                )
            ),
        },
        "maximum_absorbed_vertex_count": {
            "distribution": _distribution(
                np.asarray(arrays["union_find_maximum_absorbed_vertex_count"])
            ),
            "histogram": _histogram(
                np.asarray(arrays["union_find_maximum_absorbed_vertex_count"])
            ),
        },
    }
    raw_work = _raw_hardware_work_distributions(arrays)

    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "claim_status": CLAIM_STATUS,
        "shots": shots,
        "source_identity": dict(source_identity),
        "validation": {
            "reference_results_exact": True,
            "frozen_validation_anchors_enforced": bool(enforce_frozen_expected),
            **validation,
        },
        "accuracy": accuracy,
        "residual_detector_event_workload": workload,
        "raw_hardware_work_distributions": raw_work,
        "cycle_proxy_distributions": cycles,
        "illustrative_deadline_sensitivity": deadlines,
        "paired_global_failure_strata": strata,
        "work_proxy_tails": tails,
        "union_find_cluster_sizes": cluster_sizes,
        "architecture_definitions": {
            "fully_parallel_12_lane": (
                "one independent X/Z engine per patch; shot depth is the slowest lane"
            ),
            "patch_shared_6_engine": (
                "one engine per patch; X/Z lane work is serial within each patch and "
                "patches operate concurrently"
            ),
            "fully_shared_1_engine": (
                "one engine serializes all twelve patch/basis lanes"
            ),
        },
        "deadline_time_origin": (
            "counterfactual all-data-ready full-block service/occupancy: t=0 is "
            "after the complete shot has been presented; these values are not "
            "streaming-response latency"
        ),
        "cycle_proxy_time_bases": {
            "promatch": (
                "sum of d-wide window selection-round occupancy within each "
                "patch/basis lane"
            ),
            "pinball_ideal_stream": (
                "ideal all-block makespan N+terminal-flush+pipeline-drain; the "
                "separately reported post-final-input tail is the nine-stage "
                "critical tail for one lane"
            ),
            "pinball_offline_full_history_or": (
                "optional whole-history buffered residual OR-tree added after the "
                "ideal stream; this is not Pinball's streaming per-layer OR or "
                "accumulated complex flag"
            ),
            "union_find": (
                "Helios-style schedule: growth iterations from each lane's "
                "terminal event time at the configured weight resolution, merge "
                "flooding charged by merging-cluster forest diameter, then "
                "serialized peel, confidence-check, transaction, and exact "
                "per-patch update work"
            ),
        },
        "proxy_caveats": [
            "Cycle values are algorithm-specific architecture proxies, not measured latency.",
            "The illustrative cycle budgets do not imply a clock frequency or hardware deadline and must not be used as a cross-arm ranking.",
            "Strict deadline policy treats every proxy overrun as failure; Global bypass uses the recorded Global MWPM outcome on overrun.",
            "Pinball stream and stream-plus-offline-full-history-OR fields are reported separately; the latter assumes all full-history residual bits are buffered and reduced after streaming.",
            "Union-Find architecture fields use exact per-patch residual-boundary update counts retained by the frontend telemetry.",
            "Residual detector-event reduction is an L2 input-size metric, not an L2 latency measurement.",
            "The UF proxy charges one serialized confidence cycle per completed component; detailed confidence arithmetic and confidence-payload transport are outside the model.",
            "UF growth iterations are the slowest lane's terminal event time divided by the growth quantum (maximum canonical edge weight over the weight resolution), rounded up; the quantum is recorded in provenance.",
            "UF merge flooding uses each component's final forest diameter for every iteration in which it had an event, an upper bound on Helios's cluster-id and parity convergence over saturated edges; synchronous event batches are reported as raw software work only.",
            "Transport, queueing, clock frequency, routing, and residual-MWPM execution are outside the frontend cycle proxies.",
        ],
    }
    analysis["payload_sha256"] = _digest_json(analysis)
    return analysis


def render_hardware_report(analysis: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the exact JSON analysis."""

    lines = [
        "# Matched frontend hardware-proxy replay",
        "",
        f"Claim status: `{analysis['claim_status']}`.",
        "",
        "This is a decision-identical replay of the authenticated 10,000-shot "
        "d=7, p=0.3% corpus. Cycle fields are architecture proxies, not measured "
        "latency.",
        "",
        "## Reproduced accuracy and residual workload",
        "",
        "| Arm | Failures | Failure | Accuracy delta vs Global | Residual events | Reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    accuracy = analysis["accuracy"]
    workload = analysis["residual_detector_event_workload"]
    names = {
        "global": "Global MWPM baseline",
        "promatch": "ProMatch-style frontend + residual MWPM",
        "pinball": "Pinball-style frontend + residual MWPM",
        "union_find": "Union-Find frontend + residual MWPM",
    }
    for arm in ARM_ORDER:
        a = accuracy[arm]
        w = workload[arm]
        lines.append(
            f"| {names[arm]} | {a['failures']:,} | {a['failure_percentage']:.2f}% "
            f"| {a['difference_from_global_percentage_points']:+.2f} pp "
            f"| {w['residual_detector_events']:,} "
            f"| {w['detector_event_reduction_percentage']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Raw hardware-work distributions",
            "",
            "These are observed integer work counters and do not assume a cycle "
            "schedule. They are the primary hardware-proxy result.",
        ]
    )
    raw = analysis["raw_hardware_work_distributions"]["arms"]
    raw_arm_names = {
        "promatch": "ProMatch-style frontend",
        "pinball": "Pinball-style frontend",
        "union_find": "Union-Find frontend",
    }
    for arm in FRONTEND_ARMS:
        lines.extend(
            [
                "",
                f"### {raw_arm_names[arm]}",
                "",
                "| Raw work counter | p50 | p90 | p99 | max | Total |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in raw[arm].values():
            row = metric["per_shot_distribution"]
            lines.append(
                f"| {metric['label']} | {row['p50']:.1f} | {row['p90']:.1f} "
                f"| {row['p99']:.1f} | {row['max']:,} "
                f"| {metric['total_over_shots']:,} |"
            )

    lines.extend(
        [
            "",
            "## Secondary synthetic cycle/depth models",
            "",
            "| Arm / proxy | Architecture | p50 | p90 | p95 | p99 | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    cycle_rows = [
        ("ProMatch raw edge/path rounds", analysis["cycle_proxy_distributions"]["promatch"]),
        (
            "Pinball ideal stream lower bound",
            analysis["cycle_proxy_distributions"]["pinball"][
                "ideal_stream_cycle_lower_bound"
            ],
        ),
        (
            "Pinball stream + offline full-history OR-depth proxy",
            analysis["cycle_proxy_distributions"]["pinball"][
                "stream_plus_offline_full_history_or_depth_proxy"
            ],
        ),
        (
            "Pinball post-final-input tail lower bound",
            analysis["cycle_proxy_distributions"]["pinball"][
                "post_final_input_tail_lower_bound"
            ],
        ),
        (
            "Union-Find synthetic parameterized depth proxy",
            analysis["cycle_proxy_distributions"]["union_find"],
        ),
    ]
    for label, rows in cycle_rows:
        for architecture in ARCHITECTURES:
            row = rows[architecture]
            lines.append(
                f"| {label} | {architecture} | {row['p50']:.1f} | {row['p90']:.1f} "
                f"| {row['p95']:.1f} | {row['p99']:.1f} | {row['max']:,} |"
            )

    lines.extend(
        [
            "",
            "### p99 cycle-budget envelopes",
            "",
            "| Arm | Architecture | p99 | Next power-of-two budget | Timeout | Bypass failure | Bypass workload reduction |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    deadline_names = {
        "promatch": "ProMatch",
        "pinball": "Pinball offline-OR option",
        "union_find": "Union-Find",
    }
    deadline_rows = analysis["illustrative_deadline_sensitivity"]
    for arm in FRONTEND_ARMS:
        for architecture in ARCHITECTURES:
            envelope = deadline_rows[arm][architecture]["p99_envelope"]
            lines.append(
                f"| {deadline_names[arm]} | {architecture} "
                f"| {envelope['p99_cycles']:.1f} "
                f"| {envelope['next_power_of_two_budget']:,} "
                f"| {100 * envelope['timeout_rate']:.2f}% "
                f"| {envelope['global_bypass_failure_percentage']:.2f}% "
                f"| {envelope['global_bypass_detector_event_reduction_percentage']:.2f}% |"
            )

    cluster = analysis["union_find_cluster_sizes"][
        "maximum_observed_component_defect_count"
    ]["distribution"]
    lines.extend(
        [
            "",
            "## Union-Find cluster sizes",
            "",
            "The per-shot maximum observed component uses completed-component defect "
            "counts and censored-component lower bounds. Its distribution is "
            f"p50={cluster['p50']:.1f}, p90={cluster['p90']:.1f}, "
            f"p95={cluster['p95']:.1f}, p99={cluster['p99']:.1f}, "
            f"max={cluster['max']:,}.",
            "",
            "## How to read the deadline tables",
            "",
            "The time bases differ by arm: ProMatch is summed d-wide-window engine "
            "occupancy; Pinball is ideal all-block makespan (with a separately reported "
            "post-final-input pipeline tail); Union-Find is a synthetic full-shot "
            "schedule. The Pinball offline full-history OR-tree option is not the "
            "paper's streaming per-layer OR / accumulated-complex-flag path.",
            "",
            "`analysis.json` contains all 64 through 16,384 power-of-two cycle "
            "sensitivity "
            "tables for all three resource-sharing architectures. The strict policy "
            "counts a timeout as a failure. The Global-bypass policy substitutes the "
            "recorded Global MWPM outcome on timeout. These are illustrative policies, "
            "not timing closure results. Their time origin is counterfactual all-data-ready "
            "full-block service/occupancy, and the common budgets must not be interpreted "
            "as a cross-arm ranking.",
            "Quantile-derived inclusive policy rows at p50/p90/p95/p99/max are also "
            "included. Every policy row jointly reports timeout rate, strict and "
            "Global-bypass accuracy, and Global-bypass residual detector-event "
            "workload.",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["proxy_caveats"])
    return "\n".join(lines) + "\n"


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        **{name: np.ascontiguousarray(value) for name, value in sorted(arrays.items())},
    )
    return stream.getvalue()


_SAFE_NPZ_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_MAX_NPZ_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


def load_hardware_replay_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a bounded no-pickle proxy archive and validate every array."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError("hardware replay NPZ must be a regular non-symlink file")
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if not members:
                raise ValueError("hardware replay NPZ is empty")
            array_names: list[str] = []
            total_uncompressed = 0
            for member in members:
                name = member.filename
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or not name.endswith(".npy")
                    or "/" in name
                    or "\\" in name
                ):
                    raise ValueError("hardware replay NPZ has an unsafe member")
                array_name = name[:-4]
                if _SAFE_NPZ_ARRAY_NAME.fullmatch(array_name) is None:
                    raise ValueError("hardware replay NPZ has an unsafe array name")
                array_names.append(array_name)
                total_uncompressed += int(member.file_size)
            if len(array_names) != len(set(array_names)):
                raise ValueError("hardware replay NPZ has duplicate arrays")
            if total_uncompressed > _MAX_NPZ_UNCOMPRESSED_BYTES:
                raise ValueError("hardware replay NPZ exceeds the safe size limit")
    except zipfile.BadZipFile as ex:
        raise ValueError("hardware replay NPZ is not a valid ZIP archive") from ex

    try:
        with np.load(source, allow_pickle=False) as loaded:
            if set(loaded.files) != set(array_names):
                raise ValueError("hardware replay NPZ member inventory differs")
            arrays = {
                name: np.ascontiguousarray(np.asarray(loaded[name]))
                for name in sorted(loaded.files)
            }
    except (OSError, ValueError) as ex:
        raise ValueError("hardware replay NPZ contains an unsafe NumPy array") from ex
    _validate_hardware_arrays(arrays)
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def validate_hardware_npz_provenance(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a loaded NPZ byte-for-byte to its original replay provenance."""

    source = Path(path).absolute()
    unsigned = dict(provenance)
    recorded_payload = unsigned.pop("payload_sha256", None)
    if recorded_payload != _digest_json(unsigned):
        raise ValueError("source hardware provenance payload hash differs")
    descriptor = provenance.get("per_shot_npz")
    if not isinstance(descriptor, Mapping):
        raise ValueError("source hardware provenance omits per_shot_npz")
    source_bytes = source.read_bytes()
    if descriptor.get("file_sha256") != _sha256(source_bytes):
        raise ValueError("source hardware NPZ file hash differs from provenance")
    array_descriptors = descriptor.get("arrays")
    if not isinstance(array_descriptors, Mapping) or set(array_descriptors) != set(
        arrays
    ):
        raise ValueError("source hardware NPZ array inventory differs")
    for name, array in arrays.items():
        if array_descriptors.get(name) != _array_digest(np.asarray(array)):
            raise ValueError(f"source hardware NPZ array {name} differs")
    return {
        "path": str(source),
        "file_sha256": _sha256(source_bytes),
        "source_provenance_payload_sha256": recorded_payload,
        "arrays": {
            name: _array_digest(np.asarray(array))
            for name, array in sorted(arrays.items())
        },
    }


def write_hardware_reanalysis_artifacts(
    output: str | Path,
    *,
    analysis: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    """Write analysis/report/provenance while leaving the source NPZ untouched."""

    source = provenance.get("source_per_shot_npz")
    if not isinstance(source, Mapping) or not isinstance(
        source.get("file_sha256"), str
    ):
        raise ValueError("reanalysis provenance must bind a source NPZ")
    root = Path(output).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("hardware reanalysis output must be a fresh absent path")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise ValueError("hardware reanalysis output parent is unsafe")
    root.mkdir()
    analysis_payload = canonical_json_bytes(dict(analysis)) + b"\n"
    report = render_hardware_report(analysis).encode("utf-8")
    enriched: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        **dict(provenance),
        "claim_status": CLAIM_STATUS,
        "analysis": {
            "path": "analysis.json",
            "file_sha256": _sha256(analysis_payload),
            "payload_sha256": analysis.get("payload_sha256"),
        },
        "report": {"path": "report.md", "file_sha256": _sha256(report)},
    }
    enriched["payload_sha256"] = _digest_json(enriched)
    install_bytes_atomic(
        root / "analysis.json",
        analysis_payload,
        prefix="matched-hardware-reanalysis-",
        overwrite=False,
    )
    install_bytes_atomic(
        root / "report.md",
        report,
        prefix="matched-hardware-reanalysis-report-",
        overwrite=False,
    )
    install_bytes_atomic(
        root / "provenance.json",
        canonical_json_bytes(enriched) + b"\n",
        prefix="matched-hardware-reanalysis-provenance-",
        overwrite=False,
    )
    return {
        "analysis": str(root / "analysis.json"),
        "report": str(root / "report.md"),
        "provenance": str(root / "provenance.json"),
    }


def write_hardware_replay_artifacts(
    output: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    analysis: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    """Write one fresh non-resumable exploratory artifact directory."""

    _validate_hardware_arrays(arrays)
    root = Path(output).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("hardware replay output must be a fresh absent path")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise ValueError("hardware replay output parent is unsafe")
    root.mkdir()
    npz = _npz_bytes(arrays)
    analysis_payload = canonical_json_bytes(dict(analysis)) + b"\n"
    report = render_hardware_report(analysis).encode("utf-8")
    enriched_provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        **dict(provenance),
        "claim_status": CLAIM_STATUS,
        "per_shot_npz": {
            "path": "per_shot.npz",
            "file_sha256": _sha256(npz),
            "arrays": {
                name: _array_digest(np.asarray(value))
                for name, value in sorted(arrays.items())
            },
        },
        "analysis": {
            "path": "analysis.json",
            "file_sha256": _sha256(analysis_payload),
            "payload_sha256": analysis.get("payload_sha256"),
        },
        "report": {"path": "report.md", "file_sha256": _sha256(report)},
    }
    enriched_provenance["payload_sha256"] = _digest_json(enriched_provenance)
    provenance_payload = canonical_json_bytes(enriched_provenance) + b"\n"
    install_bytes_atomic(
        root / "per_shot.npz",
        npz,
        prefix="matched-hardware-npz-",
        overwrite=False,
    )
    install_bytes_atomic(
        root / "analysis.json",
        analysis_payload,
        prefix="matched-hardware-analysis-",
        overwrite=False,
    )
    install_bytes_atomic(
        root / "report.md",
        report,
        prefix="matched-hardware-report-",
        overwrite=False,
    )
    install_bytes_atomic(
        root / "provenance.json",
        provenance_payload,
        prefix="matched-hardware-provenance-",
        overwrite=False,
    )
    return {
        "per_shot": str(root / "per_shot.npz"),
        "analysis": str(root / "analysis.json"),
        "report": str(root / "report.md"),
        "provenance": str(root / "provenance.json"),
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARCHITECTURES",
    "ARM_ORDER",
    "CLAIM_STATUS",
    "CYCLE_BUDGETS",
    "FROZEN_EXPECTED_FAILURES",
    "FROZEN_EXPECTED_RESIDUAL_EVENTS",
    "RAW_WORK_METRICS",
    "SCHEMA",
    "analyze_hardware_replay",
    "clear_hardware_replay_worker",
    "concatenate_hardware_ranges",
    "hardware_replay_tasks",
    "load_hardware_replay_npz",
    "load_matched_hardware_protocol",
    "preload_hardware_replay_worker",
    "render_hardware_report",
    "replay_hardware_range",
    "validate_protocol_corpus",
    "validate_hardware_npz_provenance",
    "validate_reference_results",
    "validate_verified_uf_rows",
    "worker_replay_hardware_range",
    "write_hardware_reanalysis_artifacts",
    "write_hardware_replay_artifacts",
]
