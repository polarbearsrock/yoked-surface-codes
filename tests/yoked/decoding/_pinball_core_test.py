import dataclasses

import numpy as np
import pytest

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._pinball import (
    PINBALL_STAGE_ORDER,
    CompiledPinballSchedule,
    PinballPrimitive,
    PinballResult,
    PinballStageSchedule,
    compile_pinball_schedule,
    predecode_pinball,
)
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1TerminalDetector,
    compile_layout,
)


@pytest.fixture(scope="module")
def maintained_graph():
    circuit = yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=3,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=2,
        num_patches=2,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    layout = compile_layout(dem, mode="fullhistory")
    return compile_matching_graph(dem, layout, require_zero_frame=False)


@pytest.fixture(scope="module")
def maintained_schedule(maintained_graph):
    return compile_pinball_schedule(maintained_graph)


def _stage_key(stage):
    return (
        stage.patch_id,
        stage.check_basis,
        stage.sweep_time,
        PINBALL_STAGE_ORDER.index(stage.stage),
    )


def _primitive_for(schedule, stage_name):
    return next(
        primitive
        for stage in schedule.stages
        if stage.stage == stage_name
        for primitive in stage.primitives
    )


def _single_mechanism_shot(graph, primitive):
    shot = np.zeros(graph.num_detectors, dtype=np.uint8)
    shot[primitive.source] = 1
    if primitive.target is not None:
        shot[primitive.target] = 1
    return shot


def test_real_d3_schedule_is_deterministic_and_uses_nine_stage_order(
    maintained_graph, maintained_schedule
):
    second = compile_pinball_schedule(maintained_graph)
    assert second == maintained_schedule
    assert second.fingerprint == maintained_schedule.fingerprint
    assert maintained_schedule.graph_fingerprint == maintained_graph.fingerprint
    assert len(maintained_schedule.fingerprint) == 64
    assert PINBALL_STAGE_ORDER == (
        "M",
        "B1",
        "B2",
        "B3",
        "B4",
        "ST1",
        "ST2",
        "H",
        "E",
    )
    assert [_stage_key(stage) for stage in maintained_schedule.stages] == sorted(
        _stage_key(stage) for stage in maintained_schedule.stages
    )

    d = maintained_graph.layout.rounds
    for patch_id in range(maintained_graph.layout.num_patches):
        for basis in ("X", "Z"):
            for sweep_time in range(d + 1):
                stages = [
                    stage.stage
                    for stage in maintained_schedule.stages
                    if stage.patch_id == patch_id
                    and stage.check_basis == basis
                    and stage.sweep_time == sweep_time
                ]
                assert tuple(stages) == PINBALL_STAGE_ORDER
            terminal_flush = [
                stage.stage
                for stage in maintained_schedule.stages
                if stage.patch_id == patch_id
                and stage.check_basis == basis
                and stage.sweep_time == d + 1
            ]
            assert terminal_flush == ["E"]

    assert {primitive.stage for primitive in maintained_schedule.primitives} == {
        "M",
        "B1",
        "B2",
        "ST1",
        "ST2",
        "H",
        "E",
    }


def test_real_d5_schedule_exercises_all_stages_and_normalized_b_orientations():
    circuit = yoked_magic_memory_circuit(
        patch_diameter=5,
        rounds=2,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=2,
        num_patches=2,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    graph = compile_matching_graph(
        dem,
        compile_layout(dem, mode="fullhistory"),
        require_zero_frame=False,
    )
    schedule = compile_pinball_schedule(graph)

    assert compile_pinball_schedule(graph).fingerprint == schedule.fingerprint
    assert {primitive.stage for primitive in schedule.primitives} == set(
        PINBALL_STAGE_ORDER
    )
    for basis in ("X", "Z"):
        assert {
            primitive.stage
            for primitive in schedule.primitives
            if primitive.check_basis == basis
        } == set(PINBALL_STAGE_ORDER)

    expected_delta = {
        "B1": (+1, +1),
        "B2": (+1, -1),
        "B3": (-1, -1),
        "B4": (-1, +1),
    }
    observed_by_basis = {"X": set(), "Z": set()}
    for primitive in schedule.primitives:
        if primitive.stage not in expected_delta:
            continue
        source_role = graph.layout.role_of(primitive.source)
        target_role = graph.layout.role_of(primitive.target)
        assert isinstance(source_role, (L1BodyDetector, L1TerminalDetector))
        assert isinstance(target_role, (L1BodyDetector, L1TerminalDetector))
        source_x, source_y, *_ = graph.layout.coordinates[primitive.source]
        target_x, target_y, *_ = graph.layout.coordinates[primitive.target]
        source_local_x = source_x - source_role.patch_id * graph.layout.pitch
        target_local_x = target_x - target_role.patch_id * graph.layout.pitch
        if primitive.check_basis == "X":
            source_u, source_v = source_local_x, source_y
            target_u, target_v = target_local_x, target_y
        else:
            source_u, source_v = source_y, source_local_x
            target_u, target_v = target_y, target_local_x
        if round(source_u - 0.5) % 2 == 0:
            delta = (round(target_u - source_u), round(target_v - source_v))
        else:
            assert round(target_u - 0.5) % 2 == 0
            delta = (round(source_u - target_u), round(source_v - target_v))
        assert delta == expected_delta[primitive.stage]
        observed_by_basis[primitive.check_basis].add((primitive.stage, delta))

    expected_mapping = set(expected_delta.items())
    assert observed_by_basis == {
        "X": expected_mapping,
        "Z": expected_mapping,
    }


def test_compiled_dataclasses_are_frozen(maintained_schedule):
    primitive = maintained_schedule.primitives[0]
    stage = maintained_schedule.stages[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        primitive.edge_id = -1
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage.sweep_time = -1
    with pytest.raises(dataclasses.FrozenInstanceError):
        maintained_schedule.fingerprint = "changed"

    assert isinstance(primitive, PinballPrimitive)
    assert isinstance(stage, PinballStageSchedule)
    assert isinstance(maintained_schedule, CompiledPinballSchedule)


@pytest.mark.parametrize("stage_name", ["M", "B1", "ST1", "H", "E"])
def test_representative_single_mechanism_patterns_commit(
    maintained_graph, maintained_schedule, stage_name
):
    primitive = _primitive_for(maintained_schedule, stage_name)
    shot = _single_mechanism_shot(maintained_graph, primitive)
    original = shot.copy()

    result = predecode_pinball(maintained_graph, maintained_schedule, shot)

    assert isinstance(result, PinballResult)
    assert not result.complex
    assert np.array_equal(shot, original)
    assert not np.any(result.residual_syndrome)
    assert not np.any(result.tentative_residual_syndrome)
    assert result.edge_support == (primitive.edge_id,)
    assert result.tentative_edge_support == (primitive.edge_id,)
    assert sum(result.stage_match_counts) == 1
    stage_index = next(
        index
        for index, stage in enumerate(maintained_schedule.stages)
        if primitive in stage.primitives
    )
    assert result.stage_match_counts[stage_index] == 1


def test_input_and_returned_arrays_are_immutable_owned_values(
    maintained_graph, maintained_schedule
):
    primitive = _primitive_for(maintained_schedule, "M")
    shot = _single_mechanism_shot(maintained_graph, primitive).astype(bool)
    original = shot.copy()
    result = predecode_pinball(maintained_graph, maintained_schedule, shot)

    assert np.array_equal(shot, original)
    for array in (
        result.residual_syndrome,
        result.observable_frame,
        result.tentative_residual_syndrome,
        result.tentative_observable_frame,
    ):
        assert array.dtype == np.uint8
        assert array.flags.owndata
        assert not array.flags.writeable


def test_interior_singleton_makes_all_tentative_work_roll_back(
    maintained_graph, maintained_schedule
):
    boundary_sources = {
        primitive.source
        for primitive in maintained_schedule.primitives
        if primitive.stage == "E"
    }
    interior = next(
        detector_id
        for detector_id in maintained_graph.layout.body_detector_ids
        if detector_id not in boundary_sources
        and maintained_graph.layout.role_of(detector_id).time == 0
    )
    terminal_edge = next(
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.stage == "E"
        and isinstance(
            maintained_graph.layout.role_of(primitive.source), L1TerminalDetector
        )
    )
    shot = np.zeros(maintained_graph.num_detectors, dtype=np.uint8)
    shot[interior] = 1
    shot[terminal_edge.source] = 1
    original = shot.copy()

    result = predecode_pinball(maintained_graph, maintained_schedule, shot)

    assert result.complex
    assert np.array_equal(result.residual_syndrome, original)
    assert not np.any(result.observable_frame)
    assert result.edge_support == ()
    assert not result.tentative_residual_syndrome[terminal_edge.source]
    assert result.tentative_residual_syndrome[interior]
    assert result.tentative_edge_support == (terminal_edge.edge_id,)
    assert sum(result.stage_match_counts) == 1


def test_gf2_boundary_and_nonzero_observable_frame_are_exact(
    maintained_graph, maintained_schedule
):
    primitive = next(
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.stage == "B1" and primitive.patch_id == 0
    )
    old_edge = maintained_graph.edges[primitive.edge_id]
    new_edge = dataclasses.replace(old_edge, observable_mask=b"\x01")
    edges = list(maintained_graph.edges)
    edges[primitive.edge_id] = new_edge
    graph = dataclasses.replace(
        maintained_graph,
        edges=tuple(edges),
        fingerprint="pinball-frame-algebra-test",
    )
    schedule = compile_pinball_schedule(graph)
    selected = next(p for p in schedule.primitives if p.edge_id == primitive.edge_id)
    shot = _single_mechanism_shot(graph, selected)

    result = predecode_pinball(graph, schedule, shot)

    expected_boundary = np.zeros(graph.num_detectors, dtype=np.uint8)
    expected_boundary[selected.source] = 1
    expected_boundary[selected.target] = 1
    assert np.array_equal(shot ^ result.residual_syndrome, expected_boundary)
    assert result.observable_frame.tolist() == [1, 0, 0, 0]
    assert np.array_equal(
        result.observable_frame, result.tentative_observable_frame
    )


def test_schedule_covers_terminal_measurement_edges_and_flushes_terminal_e(
    maintained_graph, maintained_schedule
):
    rounds = maintained_graph.layout.rounds
    terminal_ids = set(maintained_graph.layout.terminal_detector_ids)
    terminal_m = [
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.stage == "M" and primitive.target in terminal_ids
    ]
    assert len(terminal_m) == len(terminal_ids)
    terminal_e = [
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.stage == "E" and primitive.source in terminal_ids
    ]
    assert terminal_e
    assert {primitive.sweep_time for primitive in terminal_e} == {rounds + 1}


def test_topology_validation_rejects_wrong_compilation_modes(maintained_graph):
    with pytest.raises(ValueError, match="fullhistory"):
        compile_pinball_schedule(
            dataclasses.replace(
                maintained_graph,
                layout=dataclasses.replace(maintained_graph.layout, mode="windowd"),
            )
        )
    with pytest.raises(ValueError, match="require_zero_frame=False"):
        compile_pinball_schedule(
            dataclasses.replace(maintained_graph, require_zero_frame=True)
        )
    with pytest.raises(ValueError, match="odd code distance"):
        compile_pinball_schedule(
            dataclasses.replace(
                maintained_graph,
                layout=dataclasses.replace(maintained_graph.layout, distance=4),
            )
        )


def test_topology_validation_rejects_duplicate_boundary_edge(maintained_graph):
    boundary = next(edge for edge in maintained_graph.edges if edge.target is None)
    duplicate = dataclasses.replace(boundary, edge_id=len(maintained_graph.edges))
    graph = dataclasses.replace(
        maintained_graph,
        edges=maintained_graph.edges + (duplicate,),
        fingerprint="pinball-duplicate-boundary-test",
    )
    with pytest.raises(ValueError, match="not conflict-free|more than one"):
        compile_pinball_schedule(graph)


def test_topology_validation_rejects_measurement_edge_observable_mask(
    maintained_graph, maintained_schedule
):
    measurement = _primitive_for(maintained_schedule, "M")
    edges = list(maintained_graph.edges)
    edges[measurement.edge_id] = dataclasses.replace(
        edges[measurement.edge_id], observable_mask=b"\x01"
    )
    graph = dataclasses.replace(
        maintained_graph,
        edges=tuple(edges),
        fingerprint="pinball-nonzero-measurement-frame-test",
    )
    with pytest.raises(ValueError, match="M edge.*all-zero observable mask"):
        compile_pinball_schedule(graph)


@pytest.mark.parametrize(
    "bad_shot,exception,message",
    [
        (np.zeros((1, 1), dtype=np.uint8), ValueError, "one-dimensional"),
        (np.zeros(1, dtype=np.uint8), ValueError, "length"),
        (np.asarray(["0"] * 66), TypeError, "boolean or integer"),
    ],
)
def test_predecode_input_validation(
    maintained_graph, maintained_schedule, bad_shot, exception, message
):
    with pytest.raises(exception, match=message):
        predecode_pinball(maintained_graph, maintained_schedule, bad_shot)


def test_predecode_rejects_schedule_from_other_graph(
    maintained_graph, maintained_schedule
):
    graph = dataclasses.replace(maintained_graph, fingerprint="another-graph")
    shot = np.zeros(graph.num_detectors, dtype=np.uint8)
    with pytest.raises(ValueError, match="different graph"):
        predecode_pinball(graph, maintained_schedule, shot)
