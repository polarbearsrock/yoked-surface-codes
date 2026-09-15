"""Tests for the per-patch graph splitter (_patch_graphs.py).

Checks: a two-patch synthetic DEM splits into per-patch local DEMs and graphs
with observable-flipping boundary edges re-targeted to the right check
vertex; per-patch local syndromes are gathered by global detector id;
malformed DEMs (component shape, yoke membership, cross-patch mechanisms,
and component-count mismatches) are rejected with a message naming the
violated check; the split is lossless on the distance-3, six-patch fixture;
and sampled yoke detector bits equal the sampled parity of each sector's
observables.
"""
import collections

import numpy as np
import pytest
import stim

from yoked.decoders import DecodingGraph
from yoked.hierarchical._fixtures import yoked_fixture
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraphs

# Two patches. Patch 0: X sector D0 D1 (observable 0), Z sector D2 D3 (observable 1).
# Patch 1: X sector D4 D5 (observable 2), Z sector D6 D7 (observable 3).
# Yokes: D8 (X, parity of observables 0 and 2), D9 (Z, parity of 1 and 3).
TWO_PATCH_DEM = '''
    error(0.10) D0 D1
    error(0.05) D0 D8 L0
    error(0.20) D1
    error(0.10) D2 D3
    error(0.05) D2 D9 L1
    error(0.20) D3
    error(0.10) D4 D5 ^ D6 D7
    error(0.05) D4 D8 L2
    error(0.20) D5
    error(0.05) D6 D9 L3
    error(0.20) D7
'''


def _two_patches():
    return PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(TWO_PATCH_DEM), num_patches=2)


def test_two_patch_dem_splits_into_local_dems_and_graphs():
    patches = _two_patches()
    assert patches.yoke_detector_ids == (8, 9)
    assert (patches.num_detectors, patches.num_observables) == (10, 4)
    assert [p.global_detector_ids for p in patches] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert [p.observable_ids for p in patches] == [(0, 1), (2, 3)]
    for patch in patches:
        assert patch.local_dem.num_detectors == 4 and patch.local_dem.num_observables == 2
        assert patch.graph.num_detectors == 4 and patch.check_graph.num_detectors == 6
        assert patch.check_vertices == (4, 5)
        assert len(patch.graph.edges) == len(patch.check_graph.edges) == 6
    # The observable-flipping boundary edges now end at the sector's check vertex.
    check_edges = {(u, v) for u, v, _, mask in patches[0].check_graph.edges if mask}
    assert check_edges == {(0, 4), (2, 5)}
    plain_edges = {(u, v) for u, v, _, mask in patches[0].check_graph.edges if not mask}
    assert plain_edges == {(0, 1), (1, None), (2, 3), (3, None)}
    # The correlated mechanism of patch 1 survives as one two-component instruction.
    mechanism = next(inst for inst in patches[1].local_dem if inst.type == 'error')
    assert mechanism.args_copy()[0] == pytest.approx(0.1)
    assert mechanism.target_groups() == [
        [stim.target_relative_detector_id(0), stim.target_relative_detector_id(1)],
        [stim.target_relative_detector_id(2), stim.target_relative_detector_id(3)],
    ]


def test_local_syndromes_gather_by_global_detector_id():
    patches = _two_patches()
    shots = np.zeros((2, 10), dtype=bool)
    shots[0, [1, 6]] = True
    shots[1, [4, 9]] = True
    np.testing.assert_array_equal(patches[0].local_syndromes(shots), [[0, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(patches[1].local_syndromes(shots), [[0, 0, 1, 0], [1, 0, 0, 0]])
    stacked = patches.local_syndromes(shots)
    assert stacked.shape == (2, 2, 4)
    np.testing.assert_array_equal(stacked[1, 1], [1, 0, 0, 0])


@pytest.mark.parametrize('bad, message', [
    ('error(0.05) D8 L0', 'physical detector'),                     # yoke and observable only, no physical detector
    ('error(0.05) D0 D9 L0', 'yoke membership'),                    # X observable with the Z yoke
    ('error(0.05) D0 L0', 'yoke membership'),                       # X observable with no yoke at all
    ('error(0.10) D0 D1 ^ D4 D5', 'more than one patch'),           # one mechanism in two patches
    ('error(0.10) D1 D4', 'component'),                             # joins two patches: one component, two observables
    ('error(0.05) D0 D1 D8 L0', 'one physical detector'),           # observable-flipping component keeps two physical
                                                                     # detectors; DecodingGraph.from_dem's first-label
                                                                     # merge would otherwise hide this behind the
                                                                     # existing plain D0 D1 edge
    ('error(0.05) D0 D1 D2', 'one or two physical detectors'),      # component keeps three physical detectors
    ('error(0.05) D0 D8 D9 L0 L1', 'more than one observable'),     # component flips two observables
])
def test_malformed_dems_are_rejected(bad, message):
    dem = stim.DetectorErrorModel(TWO_PATCH_DEM + bad)
    with pytest.raises(ValueError, match=message):
        PatchGraphs.from_yoked_dem(dem, num_patches=2)


def test_observable_count_must_match_patch_count():
    with pytest.raises(ValueError, match='observables'):
        PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(TWO_PATCH_DEM), num_patches=3)


def test_component_count_mismatch_without_observable_conflict_is_rejected():
    # Drop the D0 D1 union: D0 keeps its observable-flipping component, but
    # D1 becomes its own component with no observable. That yields 5
    # yoke-free components where 4 (one per observable) are expected, with no
    # two components ever disagreeing on an observable, so this exercises the
    # component-count check on its own account rather than the observable
    # conflict check exercised by the D1 D4 case above.
    missing_union = TWO_PATCH_DEM.replace('    error(0.10) D0 D1\n', '', 1)
    dem = stim.DetectorErrorModel(missing_union)
    with pytest.raises(ValueError, match='components after removing the yoke'):
        PatchGraphs.from_yoked_dem(dem, num_patches=2)


def test_split_is_lossless_on_the_distance_3_fixture():
    fx = yoked_fixture(shots=1)
    joint = DecodingGraph.from_dem(fx.dem)

    def key(u, v, mask):
        endpoints = (u, None) if v is None else (min(u, v), max(u, v))
        return endpoints, mask

    expected = collections.defaultdict(list)
    for u, v, weight, mask in joint.edges:
        expected[key(u, v, mask)].append(weight)
    rebuilt = collections.defaultdict(list)
    for patch in fx.patches:
        to_global = dict(enumerate(patch.global_detector_ids))
        to_global.update(zip(patch.check_vertices, fx.patches.yoke_detector_ids))
        for u, v, weight, mask in patch.check_graph.edges:
            global_mask = sum(1 << patch.observable_ids[s] for s in range(NUM_SECTORS) if (mask >> s) & 1)
            rebuilt[key(to_global[u], None if v is None else to_global[v], global_mask)].append(weight)
    assert rebuilt.keys() == expected.keys()
    for group, weights in expected.items():
        assert len(rebuilt[group]) == len(weights)
        np.testing.assert_allclose(sorted(rebuilt[group]), sorted(weights), atol=1e-9, rtol=0)
    assert len(fx.patches) == 6
    assert all(len(p.global_detector_ids) == (fx.dem.num_detectors - 2) // 6 for p in fx.patches)


def test_sampled_yoke_bits_are_sector_parities():
    fx = yoked_fixture(shots=200)
    yoke = fx.detectors[:, list(fx.patches.yoke_detector_ids)]
    for sector in range(NUM_SECTORS):
        parity = fx.actual[:, sector::NUM_SECTORS].sum(axis=1) % 2
        np.testing.assert_array_equal(yoke[:, sector], parity.astype(bool))
