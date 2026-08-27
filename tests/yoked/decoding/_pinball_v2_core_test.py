import dataclasses

import numpy as np
import pytest

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._pinball_v2 import (
    PINBALL_V2_STAGE_ORDER,
    CompiledPinballV2Schedule,
    PinballV2PauliTarget,
    PinballV2Result,
    compile_pinball_v2_schedule,
    predecode_pinball_v2,
)
from yoked.decoding._pinball_reference import PinballReference
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1FullHistoryDomain,
    L1TerminalDetector,
    YokeDetector,
    compile_layout,
)


def _graph(*, d: int = 3, rounds: int = 3, yokes: int = 2):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=d,
        rounds=rounds,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=yokes,
        num_patches=2,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    return compile_matching_graph(
        dem,
        compile_layout(dem, mode="fullhistory"),
        require_zero_frame=False,
    )


@pytest.fixture(scope="module")
def maintained_graph():
    return _graph()


@pytest.fixture(scope="module")
def maintained_schedule(maintained_graph):
    return compile_pinball_v2_schedule(maintained_graph)


def _local(graph, detector_id):
    role = graph.layout.role_of(detector_id)
    assert isinstance(role, (L1BodyDetector, L1TerminalDetector))
    x, y, *_ = graph.layout.coordinates[detector_id]
    local_x = x - role.patch_id * graph.layout.pitch
    if role.check_basis == "X":
        return local_x, y, role.time
    return y, local_x, role.time


@pytest.mark.parametrize("d", [3, 5, 7])
def test_exact_signed_profiles_and_two_e_kinds_at_maintained_distances(d):
    graph = _graph(d=d, rounds=2)
    schedule = compile_pinball_v2_schedule(graph)
    expected_e = (d + 1) // 2

    assert isinstance(schedule, CompiledPinballV2Schedule)
    assert compile_pinball_v2_schedule(graph).fingerprint == schedule.fingerprint
    expected_stages = set(PINBALL_V2_STAGE_ORDER)
    if d == 3:
        expected_stages -= {"B3", "B4"}
    assert {p.stage for p in schedule.primitives} == expected_stages

    e_counts = {}
    for primitive in schedule.primitives:
        edge = graph.edges[primitive.edge_id]
        if primitive.stage in ("ST1", "ST2", "H"):
            a_u, a_v, a_t = _local(graph, edge.source)
            b_u, b_v, b_t = _local(graph, edge.target)
            observed = (b_u - a_u, b_v - a_v, b_t - a_t)
            expected = {
                "ST1": (+1, +1, +1),
                "ST2": (-1, +1, +1),
                "H": (0, +2, +1),
            }[primitive.stage]
            assert observed == expected
        if primitive.stage != "E":
            continue

        assert len(primitive.activation_detectors) == 1
        inner = primitive.activation_detectors[0]
        u, _, time = _local(graph, inner)
        assert inner in primitive.detector_boundary
        if edge.target is None:
            kind = "true"
            assert primitive.detector_boundary == (inner,)
            assert u == d - 1.5
            assert not any(edge.observable_mask)
        else:
            kind = "yoke"
            assert len(primitive.detector_boundary) == 2
            assert any(
                isinstance(graph.layout.role_of(k), YokeDetector)
                for k in primitive.detector_boundary
            )
            assert u == 0.5
            observable = 2 * primitive.domain.patch_id + (
                0 if primitive.domain.check_basis == "X" else 1
            )
            assert edge.observable_mask == bytes([1 << observable])
        key = (primitive.domain, time, kind)
        e_counts[key] = e_counts.get(key, 0) + 1

    for domain in schedule.domains:
        for time in range(graph.layout.rounds + 1):
            assert e_counts[(domain, time, "yoke")] == expected_e
            assert e_counts[(domain, time, "true")] == expected_e


@pytest.mark.parametrize("initial_yoke", [0, 1])
def test_inner_yoke_e_uses_inner_activation_but_full_boundary_and_frame(
    maintained_graph, maintained_schedule, initial_yoke
):
    primitive = next(
        p
        for p in maintained_schedule.primitives
        if p.stage == "E" and len(p.detector_boundary) == 2
    )
    inner = primitive.activation_detectors[0]
    yoke = next(k for k in primitive.detector_boundary if k != inner)
    shot = np.zeros(maintained_graph.num_detectors, dtype=np.uint8)
    shot[inner] = 1
    shot[yoke] = initial_yoke
    original = shot.copy()

    result = predecode_pinball_v2(maintained_graph, maintained_schedule, shot)

    assert isinstance(result, PinballV2Result)
    assert not result.complex
    assert np.array_equal(shot, original)
    assert result.edge_support == (primitive.edge_id,)
    assert result.residual_syndrome[inner] == 0
    assert result.residual_syndrome[yoke] == 1 - initial_yoke
    expected_frame = np.unpackbits(
        np.frombuffer(
            maintained_graph.edges[primitive.edge_id].observable_mask,
            dtype=np.uint8,
        ),
        bitorder="little",
        count=maintained_graph.num_observables,
    )
    assert np.array_equal(result.observable_frame, expected_frame)
    for array in (
        result.residual_syndrome,
        result.observable_frame,
        result.tentative_residual_syndrome,
        result.tentative_observable_frame,
    ):
        assert array.dtype == np.uint8
        assert array.flags.owndata
        assert not array.flags.writeable
    with pytest.raises(TypeError):
        result.domain_results[primitive.domain] = result.domain_results[
            primitive.domain
        ]


def test_domains_commit_and_roll_back_independently():
    maintained_graph = _graph(d=5)
    maintained_schedule = compile_pinball_v2_schedule(maintained_graph)
    simple_domain = L1FullHistoryDomain(0, "X")
    complex_domain = L1FullHistoryDomain(1, "Z")
    simple_e = next(
        p
        for p in maintained_schedule.primitives
        if p.domain == simple_domain
        and p.stage == "E"
        and len(p.detector_boundary) == 2
    )
    complex_ids = np.asarray(
        [
            detector_id
            for detector_id, role in enumerate(maintained_graph.layout.roles)
            if isinstance(role, (L1BodyDetector, L1TerminalDetector))
            and L1FullHistoryDomain(role.patch_id, role.check_basis) == complex_domain
        ],
        dtype=np.int64,
    )
    # A fixed, topology-derived high-weight pattern with both tentative work
    # and an uncleared residual.  Advancing once makes the selected pattern
    # independent of detector numbering beyond the domain's stable ordering.
    rng = np.random.default_rng(1)
    rng.random(len(complex_ids))
    complex_pattern = complex_ids[rng.random(len(complex_ids)) < 0.2]
    shot = np.zeros(maintained_graph.num_detectors, dtype=np.uint8)
    shot[simple_e.activation_detectors[0]] = 1
    shot[complex_pattern] = 1
    original = shot.copy()

    result = predecode_pinball_v2(maintained_graph, maintained_schedule, shot)

    assert result.complex
    assert simple_e.edge_id in result.edge_support
    simple_yoke = next(
        detector_id
        for detector_id in simple_e.detector_boundary
        if isinstance(maintained_graph.layout.role_of(detector_id), YokeDetector)
    )
    assert result.residual_syndrome[simple_yoke] == 1
    expected_simple_observable = 2 * simple_domain.patch_id
    assert result.observable_frame[expected_simple_observable] == 1
    assert set(simple_e.physical_support) <= set(result.physical_correction)
    assert result.domain_results[complex_domain].tentative_edge_support
    assert not set(result.domain_results[complex_domain].tentative_edge_support) & set(
        result.edge_support
    )
    assert result.residual_syndrome[simple_e.activation_detectors[0]] == 0
    assert np.array_equal(
        result.residual_syndrome[complex_ids], original[complex_ids]
    )
    assert result.domain_results[simple_domain].complex is False
    assert result.domain_results[simple_domain].edge_support == (simple_e.edge_id,)
    assert (
        result.domain_results[simple_domain].physical_correction
        == simple_e.physical_support
    )
    assert result.domain_results[complex_domain].complex is True
    assert result.domain_results[complex_domain].edge_support == ()
    assert result.domain_results[complex_domain].physical_correction == ()
    assert (
        result.domain_results[complex_domain].final_residual_hw
        == len(complex_pattern)
    )
    assert np.array_equal(shot, original)


def test_two_patch_e_commits_cancel_shared_yoke_but_keep_owned_frames(
    maintained_graph, maintained_schedule
):
    first = next(
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.domain == L1FullHistoryDomain(0, "X")
        and primitive.stage == "E"
        and len(primitive.detector_boundary) == 2
    )
    shared_yoke = next(
        detector_id
        for detector_id in first.detector_boundary
        if isinstance(maintained_graph.layout.role_of(detector_id), YokeDetector)
    )
    second = next(
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.domain == L1FullHistoryDomain(1, "X")
        and primitive.sweep_time == first.sweep_time
        and shared_yoke in primitive.detector_boundary
    )
    shot = np.zeros(maintained_graph.num_detectors, dtype=np.uint8)
    shot[first.activation_detectors[0]] = 1
    shot[second.activation_detectors[0]] = 1

    result = predecode_pinball_v2(maintained_graph, maintained_schedule, shot)

    assert not result.complex
    assert set(result.edge_support) == {first.edge_id, second.edge_id}
    assert result.residual_syndrome[first.activation_detectors[0]] == 0
    assert result.residual_syndrome[second.activation_detectors[0]] == 0
    assert result.residual_syndrome[shared_yoke] == 0
    np.testing.assert_array_equal(
        result.observable_frame,
        np.asarray([1, 0, 1, 0], dtype=np.uint8),
    )
    assert set(result.physical_correction) == set(
        first.physical_support + second.physical_support
    )


@pytest.mark.parametrize("unsupported_yokes", [0, 1])
def test_exact_e_validation_rejects_bad_mask_and_yoke_profiles(
    maintained_graph, unsupported_yokes
):
    schedule = compile_pinball_v2_schedule(maintained_graph)
    yoke_e = next(
        p
        for p in schedule.primitives
        if p.stage == "E" and len(p.detector_boundary) == 2
    )
    edges = list(maintained_graph.edges)
    edges[yoke_e.edge_id] = dataclasses.replace(
        edges[yoke_e.edge_id],
        observable_mask=bytes(len(edges[yoke_e.edge_id].observable_mask)),
    )
    bad_mask_graph = dataclasses.replace(
        maintained_graph, edges=tuple(edges), fingerprint="bad-inner-yoke-mask"
    )
    with pytest.raises(
        ValueError,
        match="physical support disagrees|exact inner-yoke E",
    ):
        compile_pinball_v2_schedule(bad_mask_graph)

    with pytest.raises(ValueError, match="exactly two yoke detectors"):
        compile_pinball_v2_schedule(_graph(yokes=unsupported_yokes))


def test_every_compiled_order_one_primitive_is_simple_and_exact(
    maintained_graph, maintained_schedule
):
    for primitive in maintained_schedule.primitives:
        shot = np.zeros(maintained_graph.num_detectors, dtype=np.uint8)
        shot[np.asarray(primitive.activation_detectors, dtype=np.int64)] = 1

        result = predecode_pinball_v2(maintained_graph, maintained_schedule, shot)

        expected_residual = shot.copy()
        expected_residual[
            np.asarray(primitive.detector_boundary, dtype=np.int64)
        ] ^= 1
        assert not result.complex, primitive
        assert result.edge_support == (primitive.edge_id,), primitive
        expected_correction = tuple(sorted(primitive.physical_support))
        assert result.physical_correction == expected_correction, primitive
        assert (
            result.tentative_physical_correction == expected_correction
        ), primitive
        domain_result = result.domain_results[primitive.domain]
        assert domain_result.physical_correction == expected_correction, primitive
        assert (
            domain_result.tentative_physical_correction == expected_correction
        ), primitive
        assert np.array_equal(result.residual_syndrome, expected_residual), primitive
        assert all(not item.complex for item in result.domain_results.values())


@pytest.mark.parametrize("d", [3, 5, 7])
def test_every_primitive_has_exact_physical_pauli_support_and_frame(d):
    graph = _graph(d=d, rounds=2)
    schedule = compile_pinball_v2_schedule(graph)

    for primitive in schedule.primitives:
        expected_size = 0 if primitive.stage == "M" else 2 if primitive.stage == "H" else 1
        assert len(primitive.physical_support) == expected_size, primitive
        for target in primitive.physical_support:
            assert isinstance(target, PinballV2PauliTarget)
            assert target.patch_id == primitive.domain.patch_id
            assert 0 <= target.local_x < d
            assert 0 <= target.y < d
            assert target.pauli == ("Z" if primitive.domain.check_basis == "X" else "X")

        expected_mask = bytearray((graph.num_observables + 7) // 8)
        for target in primitive.physical_support:
            if target.pauli == "Z" and target.local_x == 0:
                observable = 2 * target.patch_id
            elif target.pauli == "X" and target.y == 0:
                observable = 2 * target.patch_id + 1
            else:
                continue
            expected_mask[observable // 8] ^= 1 << (observable % 8)
        assert graph.edges[primitive.edge_id].observable_mask == bytes(expected_mask)

        normalized_targets = tuple(
            (target.local_x, target.y)
            if primitive.domain.check_basis == "X"
            else (target.y, target.local_x)
            for target in primitive.physical_support
        )
        detector_sites = [_local(graph, k)[:2] for k in primitive.activation_detectors]
        if primitive.stage in ("B1", "B2", "B3", "B4", "ST1", "ST2"):
            expected = (
                int(round((detector_sites[0][0] + detector_sites[1][0]) / 2)),
                int(round((detector_sites[0][1] + detector_sites[1][1]) / 2)),
            )
            assert normalized_targets == (expected,), primitive
        elif primitive.stage == "H":
            later = max(detector_sites, key=lambda site: site[1])
            data_u = later[0] + (0.5 if primitive.domain.check_basis == "X" else -0.5)
            assert normalized_targets == (
                (int(round(data_u)), int(round(later[1] - 0.5))),
                (int(round(data_u)), int(round(later[1] - 1.5))),
            ), primitive
        elif primitive.stage == "E":
            u, v = detector_sites[0]
            if u == 0.5:
                data_u = 0
                data_v = v + (0.5 if primitive.domain.check_basis == "X" else -0.5)
            else:
                assert u == d - 1.5
                data_u = d - 1
                data_v = v + (-0.5 if primitive.domain.check_basis == "X" else 0.5)
            assert normalized_targets == ((data_u, int(round(data_v))),), primitive


def test_physical_correction_is_xor_reduced_in_result_and_domain_telemetry(
    maintained_graph, maintained_schedule
):
    domain = L1FullHistoryDomain(0, "X")
    candidates = [
        primitive
        for primitive in maintained_schedule.primitives
        if primitive.domain == domain
        and primitive.stage == "E"
        and len(primitive.detector_boundary) == 2
    ]
    first = candidates[0]
    last = next(
        primitive
        for primitive in reversed(candidates)
        if primitive.physical_support == first.physical_support
        and primitive.sweep_time >= first.sweep_time + 2
    )
    shot = np.zeros(maintained_graph.num_detectors, dtype=np.uint8)
    shot[first.activation_detectors[0]] = 1
    shot[last.activation_detectors[0]] = 1

    result = predecode_pinball_v2(maintained_graph, maintained_schedule, shot)

    assert set(result.edge_support) == {first.edge_id, last.edge_id}
    assert result.physical_correction == ()
    assert result.tentative_physical_correction == ()
    assert result.domain_results[domain].physical_correction == ()
    assert result.domain_results[domain].tentative_physical_correction == ()


def test_schedule_validation_rejects_tampered_physical_support(
    maintained_graph, maintained_schedule
):
    stage_index = next(
        index
        for index, stage in enumerate(maintained_schedule.stages)
        if stage.primitives and stage.primitives[0].physical_support
    )
    stage = maintained_schedule.stages[stage_index]
    primitives = list(stage.primitives)
    primitives[0] = dataclasses.replace(primitives[0], physical_support=())
    stages = list(maintained_schedule.stages)
    stages[stage_index] = dataclasses.replace(stage, primitives=tuple(primitives))
    bad_schedule = dataclasses.replace(maintained_schedule, stages=tuple(stages))

    with pytest.raises(ValueError, match="wrong physical support"):
        predecode_pinball_v2(
            maintained_graph,
            bad_schedule,
            np.zeros(maintained_graph.num_detectors, dtype=np.uint8),
        )


def test_z_domain_randomized_syndrome_complex_and_physical_differential():
    d = 5
    rounds = 2
    graph = _graph(d=d, rounds=rounds)
    schedule = compile_pinball_v2_schedule(graph)
    domain = L1FullHistoryDomain(0, "Z")
    cols = (d - 1) // 2
    batch_size = rounds + 1
    detector_by_reference_site = {}
    for detector_id, role in enumerate(graph.layout.roles):
        if not isinstance(role, (L1BodyDetector, L1TerminalDetector)):
            continue
        if L1FullHistoryDomain(role.patch_id, role.check_basis) != domain:
            continue
        u, v, time = _local(graph, detector_id)
        row = int(round(v + 0.5))
        col = int(round((u - 0.5 - ((row + 1) % 2)) / 2))
        detector_by_reference_site[(time, row, col)] = detector_id

    assert [
        stage.stage
        for stage in schedule.stages
        if stage.domain == domain and stage.sweep_time == 0
    ] == ["M", "B2", "B1", "B4", "B3", "ST2", "ST1", "H", "E"]

    reference = PinballReference(distance=d, batch_size=batch_size)
    rng = np.random.default_rng(918273)
    for _ in range(129):
        original_batch = (rng.random((batch_size, (d + 1) * cols)) < 0.22).astype(
            np.uint8
        )
        shot = np.zeros(graph.num_detectors, dtype=np.uint8)
        for time in range(batch_size):
            for row in range(d + 1):
                for col in range(cols):
                    shot[detector_by_reference_site[(time, row, col)]] = (
                        original_batch[time, row * cols + col]
                    )

        reference_batch = original_batch.copy()
        reference_correction, reference_complex = reference.decode_batch(
            reference_batch
        )
        result = predecode_pinball_v2(graph, schedule, shot)
        domain_result = result.domain_results[domain]

        actual_batch = np.zeros_like(reference_batch)
        for time in range(batch_size):
            for row in range(d + 1):
                for col in range(cols):
                    actual_batch[time, row * cols + col] = (
                        result.tentative_residual_syndrome[
                            detector_by_reference_site[(time, row, col)]
                        ]
                    )
        assert np.array_equal(actual_batch, reference_batch)
        assert domain_result.complex == reference_complex

        expected_targets = tuple(
            PinballV2PauliTarget(
                patch_id=0,
                local_x=data_index // d,
                y=data_index % d,
                pauli="X",
            )
            for data_index in np.flatnonzero(reference_correction)
        )
        assert domain_result.tentative_physical_correction == expected_targets
        assert domain_result.physical_correction == (
            () if reference_complex else expected_targets
        )


def test_signed_classifier_rejects_time_reversed_st_edge(
    maintained_graph, maintained_schedule
):
    primitive = next(p for p in maintained_schedule.primitives if p.stage == "ST1")
    old = maintained_graph.edges[primitive.edge_id]
    assert old.target is not None
    reversed_edge = dataclasses.replace(
        old,
        source=old.target,
        target=old.source,
        source_role=old.target_role,
        target_role=old.source_role,
    )
    edges = list(maintained_graph.edges)
    edges[primitive.edge_id] = reversed_edge
    graph = dataclasses.replace(
        maintained_graph, edges=tuple(edges), fingerprint="time-reversed-st"
    )
    with pytest.raises(ValueError, match="canonically oriented|unsupported signed"):
        compile_pinball_v2_schedule(graph)


def test_input_and_schedule_validation(maintained_graph, maintained_schedule):
    with pytest.raises(ValueError, match="one-dimensional"):
        predecode_pinball_v2(
            maintained_graph,
            maintained_schedule,
            np.zeros((1, maintained_graph.num_detectors), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="binary"):
        predecode_pinball_v2(
            maintained_graph,
            maintained_schedule,
            np.full(maintained_graph.num_detectors, 2, dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="another graph"):
        predecode_pinball_v2(
            dataclasses.replace(maintained_graph, fingerprint="other"),
            maintained_schedule,
            np.zeros(maintained_graph.num_detectors, dtype=np.uint8),
        )
