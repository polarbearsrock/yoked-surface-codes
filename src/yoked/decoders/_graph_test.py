import dataclasses
import math

import numpy as np
import pymatching
import pytest
import stim

from yoked.decoders import DecodingGraph


def test_direct_graph_is_immutable_and_preserves_parallel_edges_and_counts():
    edges = [(0, 1, 1, 1), (0, 1, 2, 2), (1, None, 0, 0), (1, None, 3, 0)]
    graph = DecodingGraph(np.int64(4), np.int64(3), edges)
    edges.clear()
    assert len(graph.edges) == 4
    assert graph.endpoints == ((0, 1), (0, 1), (1, 4), (1, 5))
    assert graph.adjacency == ((0, 1), (0, 1, 2, 3), (), (), (2,), (3,))
    assert graph.num_detectors == 4
    assert graph.num_observables == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        graph.num_detectors = 5


@pytest.mark.parametrize('nd,no,edges', [
    (-1, 0, []), (1, -1, []), (1.0, 0, []), (True, 0, []),
    (2, 1, [(0, 0, 1, 0)]),
    (2, 1, [(-1, 1, 1, 0)]), (2, 1, [(0, 2, 1, 0)]),
    (2, 1, [(0.5, 1, 1, 0)]),
    (2, 1, [(0, 1, 1, -1)]), (2, 1, [(0, 1, 1, 2)]),
    (2, 1, [(0, 1, 1, 0.0)]), (2, 1, [(0, 1, 1)]),
    (2, 1, [(0, 1, -1, 0)]), (2, 1, [(0, 1, float('nan'), 0)]),
    (2, 1, [(0, 1, float('inf'), 0)]),
])
def test_invalid_direct_graph(nd, no, edges):
    with pytest.raises(ValueError):
        DecodingGraph(nd, no, edges)


def test_dem_repeat_shift_export_order_and_unused_columns():
    dem = stim.DetectorErrorModel('''
        detector D6
        logical_observable L4
        error(0.1) D1 D0 L1
        repeat 2 {
            error(0.2) D0 L0
            shift_detectors 2
        }
    ''')
    graph = DecodingGraph.from_dem(dem)
    assert graph.num_detectors == 7
    assert graph.num_observables == 5
    assert [(u, v) for u, v, _, _ in graph.edges] == [(1, 0), (0, None), (2, None)]
    assert [mask for _, _, _, mask in graph.edges] == [2, 1, 1]
    np.testing.assert_allclose([e[2] for e in graph.edges], [math.log(9), math.log(4), math.log(4)])
    assert not graph.adjacency[6]


def test_dem_decomposition_and_parallel_first_label():
    graph = DecodingGraph.from_dem(stim.DetectorErrorModel('''
        error(0.1) D0 D1 L0 ^ D2
        error(0.2) D1 D0 L1
    '''))
    assert len(graph.edges) == 2
    assert graph.edges[0][:2] == (0, 1)
    assert graph.edges[0][3] == 1
    assert graph.edges[0][2] == pytest.approx(math.log(0.74 / 0.26))
    assert graph.edges[1] == pytest.approx((2, None, math.log(9), 0))


def test_dem_zero_weight_and_zero_probability():
    graph = DecodingGraph.from_dem(stim.DetectorErrorModel('''
        error(0.5) D0 L0
        error(0) D0 D1 D2 L3
    '''))
    assert graph.edges == ((0, None, 0.0, 1),)
    assert (graph.num_detectors, graph.num_observables) == (3, 4)


@pytest.mark.parametrize('text,message', [
    ('error(0.1) D0 D1 D2', 'at most two'),
    ('error(0.1) L0', 'Observable-only'),
    ('error(0.1) D0 D0 D1', 'Repeated detector'),
    ('error(0.1) D1\nerror(0.2) D0 D0 D1', 'Repeated detector'),
    ('error(0.1) D0 D0', 'Repeated detector'),
    ('error(0.75) D0', 'nonnegative'),
    ('error(1) D0', 'nonfinite'),
])
def test_rejected_dem(text, message):
    with pytest.raises(ValueError, match=message):
        DecodingGraph.from_dem(stim.DetectorErrorModel(text))


@pytest.mark.parametrize('change', ['drop', 'label', 'weight', 'duplicate'])
def test_export_mismatch_is_rejected(monkeypatch, change):
    dem = stim.DetectorErrorModel('error(0.1) D0 D1 L0')
    exported = pymatching.Matching.from_detector_error_model(dem).edges()
    if change == 'drop':
        exported.clear()
    elif change == 'label':
        exported[0][2]['fault_ids'] = set()
    elif change == 'weight':
        exported[0][2]['weight'] += 1
    elif change == 'duplicate':
        exported.append(exported[0])

    class Export:
        def edges(self):
            return exported

    monkeypatch.setattr(pymatching.Matching, 'from_detector_error_model', lambda _: Export())
    with pytest.raises(ValueError, match='audit and graph export disagree'):
        DecodingGraph.from_dem(dem)
