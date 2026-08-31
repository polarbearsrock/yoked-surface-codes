import hashlib
import pickle

import pytest

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_graph import (
    ExactDyadic,
    PatchUFLaneKey,
    compile_patch_uf_projection,
    replay_support,
)
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1TerminalDetector,
    YokeDetector,
    compile_layout,
)


def _compiled(*, d=3, rounds=6, patches=2, yokes=2, p=0.003):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=d,
        rounds=rounds,
        noise=gen.NoiseModel.si1000(p),
        style="cz",
        yokes=yokes,
        num_patches=patches,
        remove_x_yoke=False,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    layout = compile_layout(dem, mode="fullhistory")
    graph = compile_matching_graph(
        dem,
        layout,
        require_zero_frame=False,
        retain_cross_lane_edges=True,
    )
    projection = compile_patch_uf_projection(dem, graph)
    return circuit, dem, graph, projection


def test_exact_dyadic_is_normalized_and_adds_without_float_rounding() -> None:
    assert ExactDyadic.from_float(0.5) == ExactDyadic(1, -1)
    assert ExactDyadic.from_float(-1.5) == ExactDyadic(-3, -1)
    assert ExactDyadic.from_float(0.5) + ExactDyadic.from_float(0.25) == ExactDyadic(
        3, -2
    )
    with pytest.raises(ValueError, match="nonfinite"):
        ExactDyadic.from_float(float("inf"))


def test_projection_is_terminal_inclusive_deterministic_and_pickleable() -> None:
    _, dem, graph, first = _compiled()
    second = compile_patch_uf_projection(dem, graph)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert pickle.loads(pickle.dumps(first)) == first
    assert len(first.lanes) == 4
    assert first.patch_lane_ids == ((0, 1), (2, 3))
    assert first.canonical_graph_fingerprint == graph.fingerprint
    assert len(first.support_edges) == len(graph.edges)
    assert len(first.edge_owner_kind) == len(graph.edges)

    for detector_id, role in enumerate(graph.layout.roles):
        assert first.detector_role_kind[detector_id] == (
            "body"
            if isinstance(role, L1BodyDetector)
            else "terminal"
            if isinstance(role, L1TerminalDetector)
            else "yoke"
        )
        lane_id = first.detector_lane_id[detector_id]
        local_index = first.detector_local_index[detector_id]
        if isinstance(role, (L1BodyDetector, L1TerminalDetector)):
            assert lane_id is not None
            assert local_index is not None
            lane = first.lanes[lane_id]
            assert lane.global_detector_ids[local_index] == detector_id
            assert lane.key == PatchUFLaneKey(role.patch_id, role.check_basis)
            if isinstance(role, L1TerminalDetector):
                assert role.time in lane.times
        else:
            assert isinstance(role, YokeDetector)
            assert lane_id is None
            assert local_index is None
            assert first.detector_lane_id_array[detector_id] == -1
            assert first.detector_local_index_array[detector_id] == -1


def test_projection_includes_cross_window_terminal_boundary_and_yoke_context() -> None:
    _, _, graph, projection = _compiled()
    saw_cross_window = False
    saw_terminal = False
    saw_boundary = False
    saw_yoke_port = False

    for lane in projection.lanes:
        assert len(lane.incidence_offsets) == len(lane.global_detector_ids) + 1
        assert lane.incidence_offsets[-1] == len(lane.incidence_indices)
        for local_vertex in range(len(lane.global_detector_ids)):
            assert all(
                incidence.local_vertex == local_vertex
                for incidence in lane.incident(local_vertex)
            )
        for local_edge in lane.internal_correction_edges:
            edge = graph.edges[local_edge.edge_id]
            roles = edge.source_role, edge.target_role
            assert edge.target is not None
            assert not any(edge.observable_mask)
            if any(isinstance(role, L1TerminalDetector) for role in roles):
                saw_terminal = True
            if all(isinstance(role, L1BodyDetector) for role in roles):
                a, b = roles
                assert isinstance(a, L1BodyDetector)
                assert isinstance(b, L1BodyDetector)
                saw_cross_window |= a.window_id != b.window_id
        for boundary in lane.true_boundary_edges:
            edge = graph.edges[boundary.edge_id]
            assert edge.target is None
            assert not any(edge.observable_mask)
            saw_boundary = True
        for port in lane.guard_ports:
            edge = graph.edges[port.edge_id]
            assert projection.edge_owner_kind[port.edge_id] == "global-port"
            assert projection.edge_owner_lane[port.edge_id] is None
            assert projection.edge_owner_lane_array[port.edge_id] == -1
            if port.port_kind == "yoke":
                assert isinstance(graph.layout.role_of(port.remote_detector_id), YokeDetector)
                saw_yoke_port = True

    assert saw_cross_window
    assert saw_terminal
    assert saw_boundary
    assert saw_yoke_port


def test_cross_lane_global_edge_has_one_owner_and_two_local_incidences() -> None:
    _, _, graph, projection = _compiled(yokes=1)
    port_rows = {}
    for lane in projection.lanes:
        for port in lane.guard_ports:
            if port.port_kind == "cross-lane":
                port_rows.setdefault(port.edge_id, []).append(port)

    assert port_rows
    for edge_id, rows in port_rows.items():
        assert len(rows) == 2
        assert rows[0].lane_id != rows[1].lane_id
        assert {row.remote_lane_id for row in rows} == {
            rows[0].lane_id,
            rows[1].lane_id,
        }
        assert projection.edge_owner_kind[edge_id] == "global-port"
        assert projection.edge_owner_lane[edge_id] is None
        assert graph.edges[edge_id].target is not None


def test_support_replay_reconstructs_boundary_frame_weight_and_ownership() -> None:
    _, _, _, projection = _compiled()
    lane = projection.lanes[0]
    correction = lane.internal_correction_edges[0]
    boundary = lane.true_boundary_edges[0]

    correction_replay = replay_support(
        projection,
        (correction.edge_id,),
        expected_owner_lane=lane.lane_id,
    )
    assert correction_replay.detector_boundary == tuple(
        sorted(
            (
                lane.global_detector_ids[correction.local_source],
                lane.global_detector_ids[correction.local_target],
            )
        )
    )
    assert not any(correction_replay.observable_mask)
    assert correction_replay.exact_weight == projection.exact_weights[
        correction.exact_weight_index
    ]

    boundary_replay = replay_support(
        projection,
        (boundary.edge_id,),
        expected_owner_lane=lane.lane_id,
    )
    assert boundary_replay.detector_boundary == (
        lane.global_detector_ids[boundary.local_vertex],
    )

    with pytest.raises(ValueError, match="not correction-eligible"):
        replay_support(projection, (lane.guard_ports[0].edge_id,))
    with pytest.raises(ValueError, match="sorted and unique"):
        replay_support(projection, (correction.edge_id, correction.edge_id))


@pytest.fixture(scope="module")
def selected_cell():
    return _compiled(d=7, rounds=28, patches=6, yokes=2, p=0.003)


def test_selected_d7_cell_matches_authenticated_census(selected_cell) -> None:
    circuit, dem, graph, projection = selected_cell

    assert hashlib.sha256(str(circuit).encode()).hexdigest() == (
        "8cfa9bb9eaf6db86dfc9ffcfefa4582eb29932d11dc1fd911239ef1425841ff9"
    )
    assert hashlib.sha256(str(dem).encode()).hexdigest() == (
        "9b06141668bef9b334df78e4700853f60505a71f3c4185746d0261e7e3790e0a"
    )
    assert graph.layout.fingerprint == (
        "0af0345e3e778552f0b2392b20e59efa2b4143b4847d3ed1808a84b939003a1a"
    )
    assert graph.fingerprint == (
        "5f4639248bec80d73cd1a4cf85006188e964cfa23498d9ba6af41a49afc6e0cb"
    )
    assert dem.num_detectors == 8_354
    assert dem.num_observables == 12
    assert len(graph.edges) == 40_836
    assert len(projection.lanes) == 12
    assert projection.edge_owner_kind.count("local-correction") == 39_444
    assert projection.edge_owner_kind.count("global-port") == 1_392
    assert all(
        (
            len(lane.global_detector_ids),
            len(lane.internal_correction_edges),
            len(lane.true_boundary_edges),
            len(lane.guard_ports),
        )
        == (696, 3_171, 116, 116)
        for lane in projection.lanes
    )
    assert projection.validated_catalog_fingerprint == (
        "98f6bb301951114e4d8422040ff49747c2af38aefc584a532ef154f4bb50855c"
    )
    assert projection.fingerprint == (
        "43a2dbc7f08278697816dabf2f55d00f266b16033fe4b13146c333cbecca7498"
    )
