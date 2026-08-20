import pytest

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding import _promatch_graph
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1TerminalDetector,
    YokeDetector,
    compile_layout,
)


def _dem(*, d=3, rounds=6, patches=2, yokes=2):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=d,
        rounds=rounds,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=yokes,
        num_patches=patches,
    )
    return circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )


def test_compile_matching_graph_is_domain_local_and_deterministic():
    dem = _dem()
    layout = compile_layout(dem)
    compiled1 = compile_matching_graph(dem, layout)
    compiled2 = compile_matching_graph(dem, layout)

    assert compiled1.fingerprint == compiled2.fingerprint
    assert compiled1.num_detectors == dem.num_detectors
    assert compiled1.num_observables == dem.num_observables
    assert len(compiled1.edge_by_id) == compiled1.matcher.num_edges
    assert [edge.edge_id for edge in compiled1.edges] == list(
        range(len(compiled1.edges))
    )

    for domain, domain_graph in compiled1.domain_graphs.items():
        assert domain_graph.domain == domain
        assert set(domain_graph.detector_ids) == set(domain_graph.adjacency)
        for edge in domain_graph.edges:
            assert not any(edge.observable_mask)
            assert edge.source in domain_graph.detector_ids
            if edge.target is not None:
                assert edge.target in domain_graph.detector_ids
                assert layout.domain_of(edge.source) == domain
                assert layout.domain_of(edge.target) == domain
            else:
                assert edge in domain_graph.boundary_edges
        for detector_id, neighbors in domain_graph.neighbors.items():
            assert tuple(sorted(set(neighbors))) == neighbors
            assert all(layout.domain_of(other) == domain for other in neighbors)


def test_cross_window_terminal_and_yoke_edges_remain_only_global():
    dem = _dem()
    layout = compile_layout(dem)
    compiled = compile_matching_graph(dem, layout)
    domain_edge_ids = {
        edge.edge_id
        for domain_graph in compiled.domain_graphs.values()
        for edge in domain_graph.edges
    }

    saw_cross_window = False
    saw_terminal = False
    saw_yoke = False
    for edge in compiled.edges:
        if edge.target is None:
            continue
        source_role = edge.source_role
        target_role = edge.target_role
        if isinstance(source_role, YokeDetector) or isinstance(
            target_role, YokeDetector
        ):
            saw_yoke = True
            assert edge.edge_id not in domain_edge_ids
        elif isinstance(source_role, L1TerminalDetector) or isinstance(
            target_role, L1TerminalDetector
        ):
            saw_terminal = True
            assert edge.edge_id not in domain_edge_ids
        elif (
            isinstance(source_role, L1BodyDetector)
            and isinstance(target_role, L1BodyDetector)
            and source_role.window_id != target_role.window_id
        ):
            saw_cross_window = True
            assert edge.edge_id not in domain_edge_ids
    assert saw_cross_window
    assert saw_terminal
    assert saw_yoke


def test_one_yoke_cross_basis_edges_are_withheld_not_rejected():
    dem = _dem(yokes=1)
    layout = compile_layout(dem)
    compiled = compile_matching_graph(dem, layout)
    domain_edge_ids = {
        edge.edge_id
        for graph in compiled.domain_graphs.values()
        for edge in graph.edges
    }
    cross_basis = []
    for edge in compiled.edges:
        if edge.target is None:
            continue
        a = edge.source_role
        b = edge.target_role
        if (
            isinstance(a, L1BodyDetector)
            and isinstance(b, L1BodyDetector)
            and a.patch_id == b.patch_id
            and a.check_basis != b.check_basis
        ):
            cross_basis.append(edge)
            assert edge.edge_id not in domain_edge_ids
    assert cross_basis


class _FakeMatching:
    fake_edges = ()

    @classmethod
    def from_detector_error_model(cls, dem):
        return cls()

    def ensure_num_fault_ids(self, count):
        pass

    def edges(self):
        return list(self.fake_edges)


def _patch_fake_matching(monkeypatch, edges):
    fake = type("FakeMatching", (_FakeMatching,), {"fake_edges": tuple(edges)})
    monkeypatch.setattr(
        _promatch_graph.pymatching,
        "Matching",
        fake,
    )


def _same_domain_pair(layout):
    detector_ids = next(iter(layout.domain_detector_ids.values()))
    return detector_ids[0], detector_ids[1]


@pytest.mark.parametrize(
    "edge_data, message",
    [
        ({"weight": -1.0, "fault_ids": set()}, "negative weight"),
        ({"weight": float("nan"), "fault_ids": set()}, "nonfinite weight"),
        ({"weight": 1.0, "fault_ids": {999}}, "observable ID"),
    ],
)
def test_rejects_invalid_eligible_edges(monkeypatch, edge_data, message):
    dem = _dem()
    layout = compile_layout(dem)
    a, b = _same_domain_pair(layout)
    _patch_fake_matching(monkeypatch, [(a, b, edge_data)])
    with pytest.raises(ValueError, match=message):
        compile_matching_graph(dem, layout)


def test_rejects_nonzero_frame_domain_candidate(monkeypatch):
    dem = _dem()
    layout = compile_layout(dem)
    a, b = _same_domain_pair(layout)
    _patch_fake_matching(
        monkeypatch,
        [(a, b, {"weight": 1.0, "fault_ids": {0}})],
    )
    with pytest.raises(ValueError, match="nonzero observable mask"):
        compile_matching_graph(dem, layout, require_zero_frame=True)

    compiled = compile_matching_graph(dem, layout, require_zero_frame=False)
    assert (
        compiled.domain_graphs[layout.domain_of(a)].edges[0].observable_mask == b"\x01"
    )


@pytest.mark.parametrize("target_is_boundary", [False, True])
def test_rejects_foreign_patch_observable_on_local_correction(
    monkeypatch, target_is_boundary
):
    dem = _dem()
    layout = compile_layout(dem)
    a, b = _same_domain_pair(layout)
    target = None if target_is_boundary else b
    # Observable 2 is owned by patch 1, while the selected domain is patch 0.
    _patch_fake_matching(
        monkeypatch,
        [(a, target, {"weight": 1.0, "fault_ids": {2}})],
    )
    with pytest.raises(ValueError, match="owned by patch 1"):
        compile_matching_graph(dem, layout, require_zero_frame=False)


def test_rejects_duplicate_normalized_edges(monkeypatch):
    dem = _dem()
    layout = compile_layout(dem)
    a, b = _same_domain_pair(layout)
    data = {"weight": 1.0, "fault_ids": set()}
    _patch_fake_matching(monkeypatch, [(a, b, data), (b, a, data)])
    with pytest.raises(ValueError, match="duplicate normalized"):
        compile_matching_graph(dem, layout)


def test_large_dem_count_and_coordinate_metadata_are_materialized_once(monkeypatch):
    real_dem = _dem()
    layout = compile_layout(real_dem)

    class CountingDem:
        def __init__(self):
            self.num_detector_reads = 0
            self.num_observable_reads = 0
            self.coordinate_reads = 0

        @property
        def num_detectors(self):
            self.num_detector_reads += 1
            return real_dem.num_detectors

        @property
        def num_observables(self):
            self.num_observable_reads += 1
            return real_dem.num_observables

        def get_detector_coordinates(self):
            self.coordinate_reads += 1
            return real_dem.get_detector_coordinates()

    fake_dem = CountingDem()
    monkeypatch.setattr(_promatch_graph.stim, "DetectorErrorModel", CountingDem)
    _patch_fake_matching(monkeypatch, [])

    compiled = compile_matching_graph(fake_dem, layout)
    assert compiled.num_detectors == real_dem.num_detectors
    assert compiled.num_observables == real_dem.num_observables
    assert fake_dem.num_detector_reads == 1
    assert fake_dem.num_observable_reads == 1
    assert fake_dem.coordinate_reads == 1
