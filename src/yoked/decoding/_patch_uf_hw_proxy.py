"""Hardware-work proxies for the confidence-gated patch-UF frontend.

This module is deliberately observational.  It consumes an already completed
``ShotCorrection``/``LaneOutcome`` tree, or the equivalent normalized mapping
records retained by the experiment collector.  It never participates in
growth, gating, correction selection, or residual construction.

The raw counters are exact properties of the software policy and are the
primary output.  The cycle figures follow the stage model of the Helios
distributed Union-Find decoder (Liyanage, Wu, Deters, Zhong, 2023):

* Every cluster in a lane grows in lockstep, one integer weight unit per
  iteration.  The lane's terminal event time is the radius of its slowest
  cluster in log-likelihood weight units, so the number of growth iterations
  is that radius divided by the configured quantum, rounded up.  The quantum
  is the caller's hardware weight resolution (Helios's ``w_max``) expressed
  in weight units; it has no default.
* Each iteration is charged a fixed cost: the one-cycle Growing stage, the
  controller's stage-transition wait, and one settle cycle for the concurrent
  Merging/Checking pass.
* Merging additionally floods the cluster id and converges the parity across
  the merged cluster one hop per cycle, so an iteration in which a cluster
  merges is charged ``merge_cycles_per_hop`` times the diameter of the largest
  cluster that had an event in that iteration.  The component's final forest
  diameter stands in for its diameter at every one of its events, which makes
  this an upper bound; a censored snapshot has no forest diameter and uses the
  chain bound ``absorbed_vertex_count - 1``.
* Peeling, confidence checks, the patch transaction, and residual bit updates
  remain serialized inside their owning engine, as before.

Synchronous event batches, the number of distinct exact event times, remain
reported as raw software work but no longer enter any cycle figure: they are a
property of exact arithmetic on many distinct weights, not of a quantized
grid.  Routing, memory, clock-domain, transport, and L2-decoder costs are
outside the model.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping, Sequence
from fractions import Fraction


_MISSING = object()


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name=name)


def _exact_value(value: object, *, name: str) -> Fraction:
    """Converts live exact values or their serialized forms to a Fraction."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool")
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value)
    if isinstance(value, Mapping):
        if set(value) != {"mantissa", "exponent"}:
            raise ValueError(f"{name} serialized value must have mantissa/exponent")
        mantissa = _integer(value["mantissa"], name=f"{name}.mantissa", minimum=-(1 << 62))
        exponent = _integer(value["exponent"], name=f"{name}.exponent", minimum=-(1 << 62))
        if exponent >= 0:
            return Fraction(mantissa << exponent)
        return Fraction(mantissa, 1 << -exponent)
    method = getattr(value, "as_fraction", None)
    if method is not None:
        result = method()
        if not isinstance(result, Fraction):
            raise TypeError(f"{name}.as_fraction() must return Fraction")
        return result
    raise TypeError(f"{name} must be an exact value, got {type(value)!r}")


def iteration_index(time: object, quantum: object) -> int:
    """Returns the Helios growth iteration in which an event at ``time`` lands.

    Growth advances one quantum per iteration from time zero, so an event at
    time t belongs to iteration ceil(t / quantum); time zero is iteration 0.
    The division is exact.
    """

    exact_time = _exact_value(time, name="event time")
    exact_quantum = _exact_value(quantum, name="growth_quantum_weight")
    if exact_quantum <= 0:
        raise ValueError("growth_quantum_weight must be positive")
    if exact_time < 0:
        raise ValueError("event time must not be negative")
    return int(-((-exact_time) // exact_quantum))


def _get(value: object, name: str, default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"telemetry record omits {name}")


def _unwrap_adapter(value: object) -> object:
    if isinstance(value, Mapping) and "adapter" in value:
        adapter = value["adapter"]
        if not isinstance(adapter, Mapping):
            raise TypeError("serialized adapter field must be an object")
        return adapter
    return value


def _records(value: object, *, name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


@dataclasses.dataclass(frozen=True)
class UFParallelDepthAssumptions:
    """Explicit coefficients for the Helios-style depth schedule.

    ``growth_quantum_weight`` is the weight-unit advance of every growing
    boundary per iteration: the cell's maximum edge weight divided by the
    hardware weight resolution ``w_max``.  It has no default because it is a
    hardware choice.  The remaining coefficients are cycles.  ``lanes_per_patch``
    describes the dense lane ordering in the frozen patch-UF projection.
    """

    growth_quantum_weight: object
    syndrome_load_cycles: int = 1
    growing_cycles_per_iteration: int = 1
    controller_cycles_per_iteration: int = 2
    merge_settle_cycles_per_iteration: int = 1
    merge_cycles_per_hop: int = 1
    peel_cycles_per_operation: int = 1
    confidence_cycles_per_completed_component: int = 1
    patch_transaction_cycles: int = 1
    residual_update_cycles_per_boundary_event: int = 1
    lanes_per_patch: int = 2
    basis_lanes_parallel: bool = True

    def __post_init__(self) -> None:
        quantum = _exact_value(self.growth_quantum_weight, name="growth_quantum_weight")
        if quantum <= 0:
            raise ValueError("growth_quantum_weight must be positive")
        object.__setattr__(self, "growth_quantum_weight", quantum)
        for field in dataclasses.fields(self):
            if field.name == "growth_quantum_weight":
                continue
            value = getattr(self, field.name)
            if field.name == "basis_lanes_parallel":
                if not isinstance(value, bool):
                    raise TypeError("basis_lanes_parallel must be bool")
                continue
            minimum = 1 if field.name == "lanes_per_patch" else 0
            _integer(value, name=field.name, minimum=minimum)

    @property
    def fixed_cycles_per_iteration(self) -> int:
        return (
            self.growing_cycles_per_iteration
            + self.controller_cycles_per_iteration
            + self.merge_settle_cycles_per_iteration
        )


@dataclasses.dataclass(frozen=True)
class UFLaneHardwareProxy:
    """Raw work and Helios-style modeled depth for one basis lane.

    ``growth_depth_weight`` is the lane's terminal event time, the radius of
    its slowest cluster.  ``growth_iteration_count`` is that radius in
    quanta, rounded up.  ``maximum_forest_diameter_hops`` is exact for
    completed components and the chain bound for censored snapshots.
    ``synchronous_event_batch_count`` is retained as raw software work only.
    """

    lane_offset: int | None
    status: str
    synchronous_event_batch_count: int
    saturated_growth_event_count: int
    union_merge_attempt_count: int
    successful_union_merge_count: int
    redundant_union_merge_count: int
    forest_edge_count: int
    peel_operation_count: int
    completed_component_count: int
    censored_component_count: int
    maximum_completed_component_defect_count: int
    maximum_censored_component_defect_lower_bound: int
    maximum_observed_component_defect_count: int
    maximum_absorbed_vertex_count: int
    maximum_component_event_batch_count: int
    growth_depth_weight: Fraction
    growth_iteration_count: int
    maximum_forest_diameter_hops: int
    growth_depth_cycles: int
    merge_depth_cycles: int
    peel_depth_cycles: int
    confidence_depth_cycles: int
    conservative_parallel_depth_cycles: int

    @property
    def active(self) -> bool:
        return self.status != "empty"


@dataclasses.dataclass(frozen=True)
class UFShotHardwareProxy:
    """Raw aggregate work and modeled L1 depths for one shot.

    ``growth_depth_weight``, ``growth_iteration_count``, and
    ``maximum_forest_diameter_hops`` are maxima over lanes: with every lane and
    cluster growing in parallel, the slowest lane sets the critical path.  The
    exact completed-component maximum and censored-component lower-bound
    maximum are separate fields; ``maximum_observed_component_defect_count``
    is their numeric maximum.

    When exact patch transactions are present, three additional schedules are
    reported.  ``parallel_lane_cores_per_patch_depth_cycles`` models one X and
    one Z lane core plus a residual updater per patch, with every core/patch
    concurrent.  ``serial_basis_patch_engines_depth_cycles`` models one engine
    per patch that runs its X and Z lanes serially while patches remain
    concurrent.  ``fully_shared_frontend_engine_depth_cycles`` serializes all
    lane cores, all patch transactions, and all residual updates.  The first
    two are null when patch-local boundary counts are unavailable; the shared
    schedule needs only the exact shot total and can remain estimable.
    """

    global_shot_id: int | None
    lane_count: int
    patch_count: int
    active_lane_count: int
    censored_lane_count: int
    synchronous_event_batch_work: int
    saturated_growth_event_work: int
    union_merge_attempt_work: int
    successful_union_merge_work: int
    redundant_union_merge_work: int
    forest_edge_work: int
    peel_operation_work: int
    completed_component_count: int
    censored_component_count: int
    maximum_lane_synchronous_event_batches: int
    maximum_completed_component_defect_count: int
    maximum_censored_component_defect_lower_bound: int
    maximum_observed_component_defect_count: int
    maximum_absorbed_vertex_count: int
    maximum_component_event_batch_count: int
    growth_depth_weight: Fraction
    growth_iteration_count: int
    maximum_forest_diameter_hops: int
    residual_boundary_update_work: int | None
    lane_core_critical_path_cycles: int
    patch_transaction_depth_cycles: int
    residual_update_depth_cycles: int | None
    conservative_parallel_depth_cycles: int | None
    per_patch_residual_boundary_update_work: tuple[int, ...] | None
    per_patch_parallel_basis_depth_cycles: tuple[int, ...] | None
    per_patch_serial_basis_depth_cycles: tuple[int, ...] | None
    parallel_lane_cores_per_patch_depth_cycles: int | None
    serial_basis_patch_engines_depth_cycles: int | None
    fully_shared_frontend_engine_depth_cycles: int | None
    assumptions: UFParallelDepthAssumptions
    lanes: tuple[UFLaneHardwareProxy, ...]


def _component_state_kind(value: object) -> str:
    if isinstance(value, Mapping):
        collection = value.get("state_collection")
        if collection == "completed_components":
            return "completed"
        if collection == "censored_components":
            return "censored"
        state = value.get("state_kind")
        if state in ("completed", "censored"):
            return str(state)
    adapter = _unwrap_adapter(value)
    if _get(adapter, "cluster_defect_count", None) is not None or _get(
        adapter, "original_defects", None
    ) is not None:
        return "completed"
    if _get(adapter, "partial_cluster_defect_lower_bound", None) is not None or _get(
        adapter, "current_defects", None
    ) is not None:
        return "censored"
    raise ValueError("component telemetry does not identify completed/censored state")


@dataclasses.dataclass(frozen=True)
class _ComponentValues:
    kind: str
    size: int
    absorbed: int
    batches: int
    event_times: tuple[Fraction, ...]
    diameter_hops: int


def _component_values(value: object) -> _ComponentValues:
    kind = _component_state_kind(value)
    adapter = _unwrap_adapter(value)
    if kind == "completed":
        size_value = _get(adapter, "cluster_defect_count", None)
        if size_value is None:
            size_value = len(
                _records(_get(adapter, "original_defects"), name="original_defects")
            )
        size = _integer(size_value, name="cluster_defect_count", minimum=1)
    else:
        size_value = _get(adapter, "partial_cluster_defect_lower_bound", None)
        if size_value is None:
            size_value = len(
                _records(_get(adapter, "current_defects"), name="current_defects")
            )
        size = _integer(
            size_value, name="partial_cluster_defect_lower_bound", minimum=1
        )
    absorbed_value = _get(adapter, "absorbed_vertex_count", None)
    if absorbed_value is None:
        absorbed_value = len(
            _records(_get(adapter, "absorbed_vertices"), name="absorbed_vertices")
        )
    absorbed = _integer(absorbed_value, name="absorbed_vertex_count", minimum=1)
    if absorbed < size:
        raise ValueError(
            "absorbed_vertex_count must cover the reported component defects"
        )
    batches_value = _get(adapter, "simultaneous_event_batch_count", None)
    event_batch_ids = _get(adapter, "event_batch_ids", None)
    if batches_value is None:
        batches_value = len(_records(event_batch_ids, name="event_batch_ids"))
    batches = _integer(batches_value, name="component simultaneous_event_batch_count")
    if event_batch_ids is not None and batches != len(
        _records(event_batch_ids, name="event_batch_ids")
    ):
        raise ValueError("component event-batch count does not match event_batch_ids")
    raw_times = _get(adapter, "event_batch_times", None)
    if raw_times is None:
        raise ValueError("component telemetry omits event_batch_times")
    event_times = tuple(
        _exact_value(item, name="event_batch_times")
        for item in _records(raw_times, name="event_batch_times")
    )
    if len(event_times) != batches:
        raise ValueError("component event_batch_times do not match its batch count")
    raw_diameter = _get(adapter, "forest_diameter_hops", None)
    if raw_diameter is None:
        if kind == "completed":
            raise ValueError("completed component telemetry omits forest_diameter_hops")
        diameter = absorbed - 1
    else:
        diameter = _integer(raw_diameter, name="forest_diameter_hops")
        if diameter >= absorbed:
            raise ValueError("forest_diameter_hops exceeds the absorbed vertex count")
    return _ComponentValues(kind, size, absorbed, batches, event_times, diameter)


def derive_uf_lane_hardware_proxy(
    lane_outcome: object,
    *,
    assumptions: UFParallelDepthAssumptions,
    components: Sequence[object] | None = None,
    lane_offset: int | None = None,
) -> UFLaneHardwareProxy:
    """Derive exact lane work and the configured Helios-style depth proxy.

    ``lane_outcome`` may be a live ``LaneOutcome``, its ``dataclasses.asdict``
    representation, or a normalized collector row with an ``adapter`` field.
    For collector rows, pass that lane's separately retained component rows via
    ``components``.
    """

    if not isinstance(assumptions, UFParallelDepthAssumptions):
        raise TypeError("assumptions must be UFParallelDepthAssumptions")
    wrapper = lane_outcome
    adapter = _unwrap_adapter(wrapper)
    if lane_offset is None:
        raw_offset = _get(wrapper, "lane_offset", None)
        lane_offset = _optional_integer(raw_offset, name="lane_offset")
    else:
        lane_offset = _integer(lane_offset, name="lane_offset")
    status = _get(adapter, "status")
    if status not in ("empty", "completed", "censored"):
        raise ValueError(f"unsupported UF lane status {status!r}")
    counters = _unwrap_adapter(_get(adapter, "counters"))
    batches = _integer(
        _get(counters, "simultaneous_event_batch_count"),
        name="simultaneous_event_batch_count",
    )
    growth_events = _integer(
        _get(counters, "growth_event_count"), name="growth_event_count"
    )
    attempts = _integer(
        _get(counters, "union_attempt_count"), name="union_attempt_count"
    )
    successes = _integer(
        _get(counters, "successful_union_count"), name="successful_union_count"
    )
    failures = _integer(
        _get(counters, "failed_union_count"), name="failed_union_count"
    )
    if attempts != successes + failures:
        raise ValueError(
            "union attempts do not equal successful plus failed union counts"
        )
    forest_edges = _integer(
        _get(counters, "forest_edge_count"), name="forest_edge_count"
    )
    peel_operations = _integer(
        _get(counters, "peel_operation_count"), name="peel_operation_count"
    )

    if components is None:
        completed = _records(
            _get(adapter, "completed_components", ()), name="completed_components"
        )
        censored = _records(
            _get(adapter, "censored_components", ()), name="censored_components"
        )
        component_records = tuple(completed) + tuple(censored)
    else:
        component_records = _records(components, name="components")

    values = [_component_values(component) for component in component_records]
    completed_sizes = [v.size for v in values if v.kind == "completed"]
    censored_sizes = [v.size for v in values if v.kind == "censored"]
    if status == "empty":
        if component_records:
            raise ValueError("empty lane cannot contain component telemetry")
        if any(
            (batches, growth_events, attempts, successes, failures, forest_edges, peel_operations)
        ):
            raise ValueError("empty lane cannot contain UF operation work")
    elif status == "completed":
        if censored_sizes:
            raise ValueError("completed lane cannot contain censored components")
        if not completed_sizes:
            raise ValueError(
                "completed active lane requires its completed component telemetry"
            )
    else:
        if completed_sizes:
            raise ValueError("censored lane cannot contain completed components")
        if not censored_sizes:
            raise ValueError("censored lane requires its partial component telemetry")
    maximum_component_batches = max((v.batches for v in values), default=0)
    if maximum_component_batches > batches:
        raise ValueError("component event-batch count exceeds its lane total")

    quantum = assumptions.growth_quantum_weight
    if status == "empty":
        depth_weight = Fraction(0)
    else:
        raw_terminal = _get(adapter, "terminal_event_time", None)
        if raw_terminal is None:
            raise ValueError("active lane telemetry omits terminal_event_time")
        depth_weight = _exact_value(raw_terminal, name="terminal_event_time")
        if depth_weight < 0:
            raise ValueError("terminal_event_time must not be negative")
    iterations = iteration_index(depth_weight, quantum)

    # Merge flooding: each iteration with a merge event is charged the diameter
    # of the largest cluster that had an event in it.
    diameter_by_iteration: dict[int, int] = {}
    for v in values:
        for time in v.event_times:
            if time > depth_weight:
                raise ValueError(
                    "component event time exceeds the lane terminal event time"
                )
            index = iteration_index(time, quantum)
            if index < 1:
                raise ValueError("component events cannot precede the first iteration")
            diameter_by_iteration[index] = max(
                diameter_by_iteration.get(index, 0), v.diameter_hops
            )
    merge_depth = assumptions.merge_cycles_per_hop * sum(diameter_by_iteration.values())
    growth_depth = iterations * assumptions.fixed_cycles_per_iteration
    peel_depth = peel_operations * assumptions.peel_cycles_per_operation
    confidence_depth = (
        len(completed_sizes) * assumptions.confidence_cycles_per_completed_component
    )
    total_depth = (
        assumptions.syndrome_load_cycles
        + growth_depth
        + merge_depth
        + peel_depth
        + confidence_depth
    )
    maximum_completed = max(completed_sizes, default=0)
    maximum_censored = max(censored_sizes, default=0)
    return UFLaneHardwareProxy(
        lane_offset=lane_offset,
        status=str(status),
        synchronous_event_batch_count=batches,
        saturated_growth_event_count=growth_events,
        union_merge_attempt_count=attempts,
        successful_union_merge_count=successes,
        redundant_union_merge_count=failures,
        forest_edge_count=forest_edges,
        peel_operation_count=peel_operations,
        completed_component_count=len(completed_sizes),
        censored_component_count=len(censored_sizes),
        maximum_completed_component_defect_count=maximum_completed,
        maximum_censored_component_defect_lower_bound=maximum_censored,
        maximum_observed_component_defect_count=max(maximum_completed, maximum_censored),
        maximum_absorbed_vertex_count=max((v.absorbed for v in values), default=0),
        maximum_component_event_batch_count=maximum_component_batches,
        growth_depth_weight=depth_weight,
        growth_iteration_count=iterations,
        maximum_forest_diameter_hops=max((v.diameter_hops for v in values), default=0),
        growth_depth_cycles=growth_depth,
        merge_depth_cycles=merge_depth,
        peel_depth_cycles=peel_depth,
        confidence_depth_cycles=confidence_depth,
        conservative_parallel_depth_cycles=total_depth,
    )


def _component_lane_offset(value: object) -> int:
    raw = _get(value, "lane_offset", None)
    if raw is None:
        raw = _get(_unwrap_adapter(value), "lane_offset", None)
    if raw is None:
        raise ValueError("serialized component row omits lane_offset")
    return _integer(raw, name="component lane_offset")


def _shot_metric_source(value: object) -> object:
    if isinstance(value, Mapping) and "shot" in value:
        return value["shot"]
    return value


def _candidate_residual_work(value: object) -> int | None:
    sources = [value]
    if isinstance(value, Mapping):
        if "adapter_metrics" in value:
            sources.append(value["adapter_metrics"])
        if "shot" in value:
            sources.append(value["shot"])
    candidates: list[int] = []
    for source in sources:
        source = _unwrap_adapter(source)
        for name in ("durable_boundary_count", "committed_defect_count"):
            raw = _get(source, name, None)
            if raw is not None:
                candidates.append(_integer(raw, name=name))
        original = _get(source, "original_detector_count", None)
        residual = _get(source, "residual_detector_count", None)
        if original is not None and residual is not None:
            h = _integer(original, name="original_detector_count")
            r = _integer(residual, name="residual_detector_count")
            if r > h:
                raise ValueError("residual detector count exceeds original count")
            candidates.append(h - r)
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise ValueError("residual boundary-work metrics do not reconcile")
    return candidates[0]


def _patch_outcome_records(value: object) -> tuple[object, ...] | None:
    sources = [value]
    if isinstance(value, Mapping):
        if "adapter_metrics" in value:
            sources.append(value["adapter_metrics"])
        if "shot" in value:
            sources.append(value["shot"])
    found: list[tuple[object, ...]] = []
    for source in sources:
        source = _unwrap_adapter(source)
        raw = _get(source, "patch_outcomes", None)
        if raw is not None:
            found.append(_records(raw, name="patch_outcomes"))
    if not found:
        return None
    first = found[0]
    if any(candidate != first for candidate in found[1:]):
        raise ValueError("duplicate patch_outcomes telemetry does not reconcile")
    return first


def _patch_boundary_work(
    value: object,
    *,
    lane_count: int,
    lanes_per_patch: int,
) -> tuple[int, ...] | None:
    records = _patch_outcome_records(value)
    if records is None:
        return None
    if lane_count % lanes_per_patch:
        raise ValueError(
            "per-patch telemetry requires a complete dense lane group per patch"
        )
    expected_patches = lane_count // lanes_per_patch
    indexed: dict[int, int] = {}
    for position, raw in enumerate(records):
        adapter = _unwrap_adapter(raw)
        patch_id = _integer(_get(adapter, "patch_id", position), name="patch_id")
        if patch_id in indexed:
            raise ValueError(f"duplicate patch_id {patch_id}")
        lane_outcomes = _get(adapter, "lane_outcomes", None)
        if lane_outcomes is not None and len(
            _records(lane_outcomes, name="patch lane_outcomes")
        ) != lanes_per_patch:
            raise ValueError(
                f"patch {patch_id} does not contain exactly {lanes_per_patch} lanes"
            )
        counts: list[int] = []
        direct = _get(adapter, "durable_boundary_count", None)
        if direct is not None:
            counts.append(_integer(direct, name="patch durable_boundary_count"))
        boundary = _get(adapter, "durable_detector_boundary", None)
        if boundary is not None:
            detector_ids = tuple(
                _integer(item, name="durable boundary detector ID")
                for item in _records(boundary, name="patch durable_detector_boundary")
            )
            if detector_ids != tuple(sorted(set(detector_ids))):
                raise ValueError(
                    "patch durable detector boundary must be sorted and unique"
                )
            counts.append(len(detector_ids))
        if not counts:
            raise ValueError(f"patch {patch_id} omits its durable residual-boundary work")
        if len(set(counts)) != 1:
            raise ValueError(f"patch {patch_id} durable boundary metrics do not reconcile")
        indexed[patch_id] = counts[0]
    if tuple(sorted(indexed)) != tuple(range(expected_patches)):
        raise ValueError(
            "patch IDs must be dense and match the number of complete lane groups"
        )
    # The selected YSC cell has six physical patches and two basis lanes.  The
    # generic path remains usable for smaller unit graphs, but whenever either
    # side exposes the production cardinality, require the exact 6x2 layout.
    if lane_count == 12 and (expected_patches != 6 or lanes_per_patch != 2):
        raise ValueError("the 12-lane YSC layout must be six dense two-lane patches")
    return tuple(indexed[patch_id] for patch_id in range(expected_patches))


def derive_uf_shot_hardware_proxy(
    shot: object,
    *,
    assumptions: UFParallelDepthAssumptions,
    lane_rows: Sequence[object] | None = None,
    component_rows: Sequence[object] | None = None,
    residual_boundary_update_work: int | None = None,
) -> UFShotHardwareProxy:
    """Derive one shot's raw work and Helios-style L1 depth proxy.

    Live ``ShotCorrection`` objects and ``dataclasses.asdict`` mappings expose
    ``lane_outcomes`` directly.  A normalized artifact bundle may instead use
    ``{"shot": ..., "lanes": [...], "components": [...]}``, or callers may
    pass the two row sequences explicitly.  Artifact rows must all belong to
    one shot and use dense lane offsets.
    """

    if not isinstance(assumptions, UFParallelDepthAssumptions):
        raise TypeError("assumptions must be UFParallelDepthAssumptions")
    metric_source = _shot_metric_source(shot)
    if lane_rows is None:
        embedded_lanes = _get(shot, "lane_outcomes", None)
        if embedded_lanes is None and isinstance(shot, Mapping):
            embedded_lanes = shot.get("lanes")
        if embedded_lanes is None:
            embedded_lanes = _get(metric_source, "lane_outcomes", None)
        if embedded_lanes is None:
            raise ValueError("shot telemetry omits lane outcomes/rows")
        resolved_lane_rows = _records(embedded_lanes, name="lane_rows")
    else:
        resolved_lane_rows = _records(lane_rows, name="lane_rows")
    if component_rows is None and isinstance(shot, Mapping):
        component_rows = shot.get("components")
    resolved_components = _records(component_rows, name="component_rows")

    shot_ids: set[int] = set()
    direct_shot_id = _get(metric_source, "global_shot_id", None)
    if direct_shot_id is not None:
        shot_ids.add(_integer(direct_shot_id, name="global_shot_id"))
    indexed_lanes: dict[int, object] = {}
    for position, row in enumerate(resolved_lane_rows):
        row_shot = _get(row, "global_shot_id", None)
        if row_shot is not None:
            shot_ids.add(_integer(row_shot, name="lane global_shot_id"))
        raw_offset = _get(row, "lane_offset", None)
        offset = position if raw_offset is None else _integer(raw_offset, name="lane_offset")
        if offset in indexed_lanes:
            raise ValueError(f"duplicate lane_offset {offset}")
        indexed_lanes[offset] = row
    if not indexed_lanes:
        raise ValueError("shot hardware proxy requires at least one lane")
    if tuple(sorted(indexed_lanes)) != tuple(range(len(indexed_lanes))):
        raise ValueError("lane offsets must be dense from zero")

    components_by_lane: dict[int, list[object]] = {offset: [] for offset in indexed_lanes}
    for row in resolved_components:
        row_shot = _get(row, "global_shot_id", None)
        if row_shot is not None:
            shot_ids.add(_integer(row_shot, name="component global_shot_id"))
        offset = _component_lane_offset(row)
        if offset not in components_by_lane:
            raise ValueError(f"component references absent lane_offset {offset}")
        components_by_lane[offset].append(row)
    if len(shot_ids) > 1:
        raise ValueError("serialized rows belong to more than one global shot")

    lanes = tuple(
        derive_uf_lane_hardware_proxy(
            indexed_lanes[offset],
            assumptions=assumptions,
            components=components_by_lane[offset] if resolved_components else None,
            lane_offset=offset,
        )
        for offset in range(len(indexed_lanes))
    )

    patch_depths: list[int] = []
    width = assumptions.lanes_per_patch
    for start in range(0, len(lanes), width):
        group = lanes[start : start + width]
        depths = [lane.conservative_parallel_depth_cycles for lane in group]
        patch_depths.append(max(depths) if assumptions.basis_lanes_parallel else sum(depths))
    lane_critical = max(patch_depths, default=0)

    per_patch_residual_work = _patch_boundary_work(
        metric_source, lane_count=len(lanes), lanes_per_patch=width
    )
    if per_patch_residual_work is None and metric_source is not shot:
        per_patch_residual_work = _patch_boundary_work(
            shot, lane_count=len(lanes), lanes_per_patch=width
        )
    inferred_residual_work = _candidate_residual_work(metric_source)
    if inferred_residual_work is None:
        inferred_residual_work = _candidate_residual_work(shot)
    if per_patch_residual_work is not None:
        patch_total = sum(per_patch_residual_work)
        if inferred_residual_work is not None and patch_total != inferred_residual_work:
            raise ValueError(
                "per-patch residual boundary work does not reconcile with shot total"
            )
        inferred_residual_work = patch_total
    if residual_boundary_update_work is not None:
        explicit = _integer(residual_boundary_update_work, name="residual_boundary_update_work")
        if inferred_residual_work is not None and explicit != inferred_residual_work:
            raise ValueError("explicit residual boundary work disagrees with telemetry")
        inferred_residual_work = explicit
    residual_depth = (
        None
        if inferred_residual_work is None
        else inferred_residual_work * assumptions.residual_update_cycles_per_boundary_event
    )
    total_depth = (
        None
        if residual_depth is None
        else lane_critical + assumptions.patch_transaction_cycles + residual_depth
    )
    if per_patch_residual_work is None:
        parallel_patch_depths = None
        serial_patch_depths = None
        parallel_lane_cores_depth = None
        serial_basis_engines_depth = None
    else:
        parallel_values: list[int] = []
        serial_values: list[int] = []
        for patch_id, boundary_work in enumerate(per_patch_residual_work):
            group = lanes[patch_id * width : (patch_id + 1) * width]
            lane_depths = [lane.conservative_parallel_depth_cycles for lane in group]
            residual_patch_depth = (
                boundary_work * assumptions.residual_update_cycles_per_boundary_event
            )
            common_tail = assumptions.patch_transaction_cycles + residual_patch_depth
            parallel_values.append(max(lane_depths) + common_tail)
            serial_values.append(sum(lane_depths) + common_tail)
        parallel_patch_depths = tuple(parallel_values)
        serial_patch_depths = tuple(serial_values)
        parallel_lane_cores_depth = max(parallel_patch_depths, default=0)
        serial_basis_engines_depth = max(serial_patch_depths, default=0)
    shared_engine_depth = (
        None
        if residual_depth is None
        else sum(lane.conservative_parallel_depth_cycles for lane in lanes)
        + len(patch_depths) * assumptions.patch_transaction_cycles
        + residual_depth
    )
    maximum_completed = max(
        (lane.maximum_completed_component_defect_count for lane in lanes), default=0
    )
    maximum_censored = max(
        (lane.maximum_censored_component_defect_lower_bound for lane in lanes), default=0
    )
    return UFShotHardwareProxy(
        global_shot_id=next(iter(shot_ids), None),
        lane_count=len(lanes),
        patch_count=len(patch_depths),
        active_lane_count=sum(lane.active for lane in lanes),
        censored_lane_count=sum(lane.status == "censored" for lane in lanes),
        synchronous_event_batch_work=sum(lane.synchronous_event_batch_count for lane in lanes),
        saturated_growth_event_work=sum(lane.saturated_growth_event_count for lane in lanes),
        union_merge_attempt_work=sum(lane.union_merge_attempt_count for lane in lanes),
        successful_union_merge_work=sum(lane.successful_union_merge_count for lane in lanes),
        redundant_union_merge_work=sum(lane.redundant_union_merge_count for lane in lanes),
        forest_edge_work=sum(lane.forest_edge_count for lane in lanes),
        peel_operation_work=sum(lane.peel_operation_count for lane in lanes),
        completed_component_count=sum(lane.completed_component_count for lane in lanes),
        censored_component_count=sum(lane.censored_component_count for lane in lanes),
        maximum_lane_synchronous_event_batches=max(
            (lane.synchronous_event_batch_count for lane in lanes), default=0
        ),
        maximum_completed_component_defect_count=maximum_completed,
        maximum_censored_component_defect_lower_bound=maximum_censored,
        maximum_observed_component_defect_count=max(maximum_completed, maximum_censored),
        maximum_absorbed_vertex_count=max(
            (lane.maximum_absorbed_vertex_count for lane in lanes), default=0
        ),
        maximum_component_event_batch_count=max(
            (lane.maximum_component_event_batch_count for lane in lanes), default=0
        ),
        growth_depth_weight=max((lane.growth_depth_weight for lane in lanes), default=Fraction(0)),
        growth_iteration_count=max((lane.growth_iteration_count for lane in lanes), default=0),
        maximum_forest_diameter_hops=max(
            (lane.maximum_forest_diameter_hops for lane in lanes), default=0
        ),
        residual_boundary_update_work=inferred_residual_work,
        lane_core_critical_path_cycles=lane_critical,
        patch_transaction_depth_cycles=assumptions.patch_transaction_cycles,
        residual_update_depth_cycles=residual_depth,
        conservative_parallel_depth_cycles=total_depth,
        per_patch_residual_boundary_update_work=per_patch_residual_work,
        per_patch_parallel_basis_depth_cycles=parallel_patch_depths,
        per_patch_serial_basis_depth_cycles=serial_patch_depths,
        parallel_lane_cores_per_patch_depth_cycles=parallel_lane_cores_depth,
        serial_basis_patch_engines_depth_cycles=serial_basis_engines_depth,
        fully_shared_frontend_engine_depth_cycles=shared_engine_depth,
        assumptions=assumptions,
        lanes=lanes,
    )


__all__ = [
    "UFLaneHardwareProxy",
    "UFParallelDepthAssumptions",
    "UFShotHardwareProxy",
    "derive_uf_lane_hardware_proxy",
    "derive_uf_shot_hardware_proxy",
    "iteration_index",
]
