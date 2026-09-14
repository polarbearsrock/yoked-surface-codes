# Hierarchical L1/L2 Decoding, M1 and Pilot: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the patch-local L1 layer (UF reference plus cluster gap, forced matching weights plain and correlated), the exact L2 outer decoder, isotonic calibration, resumable collection with validation, endpoint replay with core metrics, and run the 2,000-shot d=9 pilot that decides whether the full experiment proceeds.

**Architecture:** A new package `src/yoked/hierarchical/` with one module per concept, tests beside each module. The six-patch DEM is split into per-patch graphs by connected components with the yoke hub replaced by per-patch check vertices. L1 decoders run on those graphs and never see a yoke bit. Everything L1 produces is stored once in an `L1Record`; calibration and replay are offline stages over that record. A thin CLI in `tools/hierarchical_experiment` drives the stages.

**Tech Stack:** Python 3.14 in `.venv`, NumPy, SciPy sparse, Stim 1.16, PyMatching 2.4 (edge-list construction through `from_check_matrix`), sinter for LER normalization, pytest. Run everything from the repository root with `PYTHONPATH=src`.

**Spec:** `docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md`. This plan implements spec sections 4, 5, 6, 7, the `initial_only` and `all_refined` rows of section 8, the primary metric, strata, block failure and bootstrap of section 9, validation tests 1 to 9 of section 10, and milestone M1 of section 12. Sections 8 (selective policies, random controls, work accounting), the rest of 9, tests 10 to 15, and milestones M2 to M4 belong to a second plan written after the pilot decision.

## Global Constraints

- Notation follows spec section 2 exactly: patch `i`, sector `s` (0 = X, 1 = Z), observable `o(i, s) = 2i + s`, record column `2i + s`, internal sector-major layout `(shots, 2, 6)`.
- Yoke detectors are the last two detectors, X then Z. Check vertices in a patch's check graph are local ids `n` and `n + 1` where `n` is the patch's detector count.
- All weights, gaps, and scores are in nats. Nothing converts to decibels.
- L1 code never reads a sampled yoke bit. Forced decodes use synthetic check bits only.
- Calibrators are fit on the calibration sample only; the driver refuses to calibrate and evaluate on the same sample or subset (spec test 9).
- Every module starts with a docstring stating the object it computes in spec notation and naming the invariants its tests check. Records crossing module boundaries are frozen dataclasses with a docstring per field including units and shapes. Named constants carry a comment saying why they hold their value. Seeds are explicit parameters.
- Temporary and run outputs go under `$TMPDIR` (`/data2/s2chitni/.tmp`), never under the home directory or `/tmp`.
- Tests run with `PYTHONPATH=src .venv/bin/pytest <path> -q`. The existing suite in `src/yoked/decoders` must keep passing after every task.
- Commit after every task with the attribution lines from the session reminder.

## File Structure

| File | Responsibility |
|---|---|
| `src/yoked/decoders/_correlations.py` | Correlation rules from a DEM, indexing by source, applying rules to weights. Moved out of the correlated UF module so matching and UF share one implementation. |
| `src/yoked/decoders/_correlated_union_find.py` | Modified to import from `_correlations`. Behavior unchanged. |
| `src/yoked/hierarchical/__init__.py` | Public names. |
| `src/yoked/hierarchical/_patch_graphs.py` | `PatchGraph`, `PatchGraphs`: hub split into local DEMs, check-free and check graphs, local syndromes. |
| `src/yoked/hierarchical/_fixtures.py` | Test support: a small six-patch yoked fixture (distance 3) shared by test modules. |
| `src/yoked/hierarchical/_cluster_gap.py` | `ClusterGapUnionFindDecoder`: UF reference bits plus cluster gap by parity-augmented Dijkstra. |
| `src/yoked/hierarchical/_matching_gaps.py` | `MatchingGaps`, `ForcedWeights`, `signed_gaps`: forced class weights, plain and correlated, from check-matrix matchers. |
| `src/yoked/hierarchical/_outer_decoder.py` | `exact_outer_map`, `exact_outer_map_batch`, `frame_adjusted_syndrome`. |
| `src/yoked/hierarchical/_calibration.py` | `IsotonicCalibrator`: PAV fit, interpolation, clipping, JSON. |
| `src/yoked/hierarchical/_provenance.py` | SHA-256 helpers, atomic JSON writes, package versions, source hashes. |
| `src/yoked/hierarchical/_record.py` | `L1Record`: the stored L1 outputs, save and load, sector-major views. |
| `src/yoked/hierarchical/_collect.py` | Sampling or loading a sample set, per-row L1 evaluation, chunked parallel collection with resume and row subsets, record validation. |
| `src/yoked/hierarchical/_policies.py` | `NoRefinement`, `RefineAll`, and the deterministic policy interface later policies implement. |
| `src/yoked/hierarchical/_replay.py` | Estimator scores, `ReplayConfig`, `replay`, `ReplayResult`. |
| `src/yoked/hierarchical/_metrics.py` | Residual errors, strata, sector and block failure, misattribution, paired bootstrap, normalized LER. |
| `tools/hierarchical_experiment` | CLI: `collect`, `calibrate`, `replay`, `summarize`. |
| `docs/hierarchical_decoding.md` | Usage, mirroring `docs/union_find_usage.md`. |
| `docs/results/hierarchical_pilot_d9_p003.md` | Pilot report and go/no-go record. |

Deviations from the spec's file list, all additive: `_fixtures.py` (test support), `_provenance.py` and `_record.py` (split out of `_replay.py` so that collection and replay share record I/O without importing each other), and `_collect.py` (stage logic the spec requires to be reachable without the CLI). The `report` subcommand of the spec is delivered as `summarize` here, producing the pilot tables; the full figure-producing `report` belongs to the second plan.

---

### Task 1: Shared correlation rules module

The correlated UF decoder holds the only implementation of DEM-derived correlation rules. The matching gaps need the same rules and the same application semantics (lowest implied weight wins, weights never rise). Move the code to a shared module without changing behavior.

**Files:**
- Create: `src/yoked/decoders/_correlations.py`
- Create: `src/yoked/decoders/_correlations_test.py`
- Modify: `src/yoked/decoders/_correlated_union_find.py` (whole file; it shrinks to the decoder class)

**Interfaces:**
- Consumes: `DecodingGraph`, `_index` from `yoked.decoders._graph`.
- Produces:
  - `CorrelationRule = tuple[int, int, float]` (source edge, target edge, implied weight in nats).
  - `RulesBySource = tuple[tuple[tuple[int, float], ...], ...]`, indexed by source edge id.
  - `correlation_rules_from_dem(graph: DecodingGraph, dem: stim.DetectorErrorModel) -> list[CorrelationRule]`
  - `index_rules_by_source(graph: DecodingGraph, rules: Iterable[CorrelationRule]) -> RulesBySource`
  - `apply_correlation_rules(weights: Sequence[float], rules_by_source: RulesBySource, selected_edges: Iterable[int]) -> list[float] | None` returning `None` when no rule lowers a weight.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/decoders/_correlations_test.py`:

```python
import math

import pytest
import stim

from yoked.decoders import DecodingGraph
from yoked.decoders._correlations import (
    apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)


def test_apply_rules_keeps_the_lowest_weight_and_never_raises_one():
    graph = DecodingGraph(3, 0, [(0, 1, 1.0, 0), (1, 2, 2.0, 0), (2, None, 3.0, 0)])
    rules = index_rules_by_source(graph, [(0, 1, 0.5), (2, 1, 1.5), (0, 2, 4.0)])
    weights = [1.0, 2.0, 3.0]
    assert apply_correlation_rules(weights, rules, []) is None
    # Sources 0 and 2 both support edge 1; the lower implied weight wins.
    # Source 0 also names edge 2 with weight 4.0 > 3.0, which must not raise it.
    assert apply_correlation_rules(weights, rules, [0, 2]) == [1.0, 0.5, 3.0]
    assert apply_correlation_rules(weights, rules, [2]) == [1.0, 1.5, 3.0]
    assert weights == [1.0, 2.0, 3.0]


def test_index_rules_validates_and_preserves_order():
    graph = DecodingGraph(2, 0, [(0, 1, 1.0, 0), (1, None, 1.0, 0)])
    with pytest.raises(ValueError, match='out-of-range'):
        index_rules_by_source(graph, [(0, 5, 0.0)])
    with pytest.raises(ValueError, match='different edges'):
        index_rules_by_source(graph, [(0, 0, 0.0)])
    with pytest.raises(ValueError, match='nonnegative'):
        index_rules_by_source(graph, [(0, 1, -1.0)])
    assert index_rules_by_source(graph, [(1, 0, 0.25), (1, 0, 0.5)]) == ((), ((0, 0.25), (0, 0.5)))


def test_rules_from_dem_reproduce_the_documented_implied_weights():
    dem = stim.DetectorErrorModel('''
        error(0.1) D1 D0 ^ D2 ^ D3
        error(0.2) D0 D1
        error(0.05) D0 D1 ^ D2
        error(0.3) D3
    ''')
    graph = DecodingGraph.from_dem(dem)
    rules = {(source, target): weight for source, target, weight in correlation_rules_from_dem(graph, dem)}
    # Edge 0 = D0D1 (marginal .284), edge 1 = D2 (.14), edge 2 = D3 (.34).
    # Shared mechanism probabilities: 01 -> .14, 02 -> .1, 12 -> .1.
    assert rules[(0, 1)] == pytest.approx(math.log(0.144 / 0.14))
    assert rules[(0, 2)] == pytest.approx(math.log(0.184 / 0.1))
    assert rules[(1, 0)] == 0 and rules[(1, 2)] == 0
    assert rules[(2, 0)] == pytest.approx(math.log(0.24 / 0.1))
    assert rules[(2, 1)] == pytest.approx(math.log(0.24 / 0.1))


def test_rules_from_dem_reject_a_repeated_edge_across_components():
    dem = stim.DetectorErrorModel('error(0.1) D0 D1 ^ D1 D0')
    with pytest.raises(ValueError, match='repeats a graph edge'):
        correlation_rules_from_dem(DecodingGraph.from_dem(dem), dem)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/decoders/_correlations_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.decoders._correlations'`

- [ ] **Step 3: Create the shared module**

Create `src/yoked/decoders/_correlations.py`. The body of `correlation_rules_from_dem` and `_xor_probability` is moved verbatim from `_correlated_union_find.py`; `index_rules_by_source` is the validation loop from that class's `__init__`; `apply_correlation_rules` is the reweighting loop from its `_decode`.

```python
"""Pairwise correlation rules derived from a decomposed DEM.

A DEM error mechanism that decomposes into several graph edges makes those
edges correlated: selecting one edge in a first decoding pass is evidence
that its partners occurred too. Following PyMatching's pairwise
approximation, every ordered pair (source, target) of edges that share at
least one mechanism yields a rule ``(source, target, implied_weight)``.
When ``source`` is selected, the second pass lowers ``target``'s weight to
``implied_weight = ln((1 - p) / p)`` with ``p = min(1/2, P(shared) / P(source))``.

Rules never raise a weight. When several selected sources support one
target, the lowest implied weight wins. ``_correlations_test.py`` checks the
compiled weights on hand-written DEMs and the application rule; the
correlated UF tests cover the decoder that uses them.
"""
from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Iterable, Sequence

from yoked.decoders._graph import DecodingGraph, _index

if TYPE_CHECKING:
    import stim

CorrelationRule = tuple[int, int, float]
"""``(source edge id, target edge id, implied weight in nats)``."""

RulesBySource = tuple[tuple[tuple[int, float], ...], ...]
"""``rules_by_source[source]`` lists ``(target, implied_weight)`` pairs in rule order."""


def correlation_rules_from_dem(
        graph: DecodingGraph, dem: stim.DetectorErrorModel,
) -> list[CorrelationRule]:
    """Compile PyMatching's pairwise implied-weight approximation once."""
    edge_ids = {
        (u, None) if v is None else tuple(sorted((u, v))): e
        for e, (u, v, _, _) in enumerate(graph.edges)
    }
    marginal = [0.0] * len(graph.edges)
    shared: dict[tuple[int, int], float] = {}
    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        probability, = instruction.args_copy()
        if probability == 0:
            continue

        # Separators split one error mechanism into its graph components.
        # The graph audit has already checked detector counts and labels.
        components: list[list[int]] = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                components.append([])
            elif target.is_relative_detector_id():
                components[-1].append(target.val)
        edges = [
            edge_ids[(ds[0], None) if len(ds) == 1 else tuple(sorted(ds))]
            for ds in components if ds
        ]
        if len(set(edges)) != len(edges):
            raise ValueError('A DEM error repeats a graph edge across components')
        for e in edges:
            marginal[e] = _xor_probability(marginal[e], probability)
        for a, b in itertools.combinations(sorted(edges), 2):
            shared[a, b] = _xor_probability(shared.get((a, b), 0.0), probability)

    rules = []
    for (a, b), probability in shared.items():
        if probability == 0:
            continue
        for source, target in ((a, b), (b, a)):
            # This is an implied probability from shared mechanisms, not a
            # full posterior over physical faults. Clipping keeps weights
            # nonnegative, including zero for strongly supported partners.
            implied_probability = min(0.5, probability / marginal[source])
            weight = math.log1p(-implied_probability) - math.log(implied_probability)
            if weight < graph.edges[target][2]:
                rules.append((source, target, weight))
    return rules


def index_rules_by_source(graph: DecodingGraph, rules: Iterable[CorrelationRule]) -> RulesBySource:
    """Validate rules against ``graph`` and group them by source edge, keeping order."""
    by_source: list[list[tuple[int, float]]] = [[] for _ in graph.edges]
    for source, target, weight in rules:
        source = _index(source, 'Correlation source edge')
        target = _index(target, 'Correlation target edge')
        weight = float(weight)
        if not 0 <= source < len(graph.edges) or not 0 <= target < len(graph.edges):
            raise ValueError('Correlation rule has an out-of-range edge ID')
        if source == target:
            raise ValueError('Correlation rules must connect different edges')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Implied weights must be finite and nonnegative')
        by_source[source].append((target, weight))
    return tuple(tuple(targets) for targets in by_source)


def apply_correlation_rules(
        weights: Sequence[float], rules_by_source: RulesBySource, selected_edges: Iterable[int],
) -> list[float] | None:
    """Lower target weights supported by the selected edges; ``None`` if nothing changes.

    Evidence comes from the selected correction only, never from every grown
    edge. Several sources can support one target; the lowest weight is kept.
    """
    adjusted = list(weights)
    changed = False
    for source in selected_edges:
        for target, implied_weight in rules_by_source[source]:
            if implied_weight < adjusted[target]:
                adjusted[target] = implied_weight
                changed = True
    return adjusted if changed else None


def _xor_probability(a: float, b: float) -> float:
    # Independent mechanisms toggle an edge: two occurrences cancel.
    return a * (1 - b) + b * (1 - a)
```

- [ ] **Step 4: Point the correlated UF decoder at the shared module**

Replace the whole of `src/yoked/decoders/_correlated_union_find.py` with:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from yoked.decoders._correlations import (
    CorrelationRule, apply_correlation_rules, correlation_rules_from_dem, index_rules_by_source,
)
from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import UnionFindDecoder, _Correction

if TYPE_CHECKING:
    import stim


class CorrelatedUnionFindDecoder(UnionFindDecoder):
    """Two UF passes, with edge correlations informing the second pass.

    Each correlation rule is (source_edge, target_edge, implied_weight).
    Selecting the source in the first correction lowers the target's weight
    for the second pass. Edge IDs refer to positions in ``graph.edges``.

    Use ``from_dem`` to derive the rules from shared DEM error mechanisms,
    or supply them explicitly for a graph. Both passes use the original
    syndrome and the repository's weighted growth and peeling algorithm.
    The rule compilation and application live in ``_correlations`` so that
    the hierarchical matching gaps share them.
    """

    def __init__(self, graph: DecodingGraph, *, correlation_rules: Iterable[CorrelationRule]):
        super().__init__(graph)
        # Own an immutable copy, just as DecodingGraph owns its edge list.
        self._correlation_rules = index_rules_by_source(graph, correlation_rules)

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> CorrelatedUnionFindDecoder:
        """Import a decomposed DEM and its pairwise correlation rules.

        Graph import performs the usual component/weight/label audit. A DEM
        error that repeats an edge across components is additionally rejected:
        the repeated detector incidences cancel instead of forming partners.
        PyMatching is used for graph export only; neither UF pass runs MWPM.
        """
        graph = DecodingGraph.from_dem(dem)
        return cls(graph, correlation_rules=correlation_rules_from_dem(graph, dem))

    def _decode(self, syndrome) -> _Correction:
        first = super()._decode(syndrome)
        weights = apply_correlation_rules(
            [weight for _, _, weight, _ in self.graph.edges],
            self._correlation_rules,
            first.selected_edges,
        )
        if weights is None:
            return first

        # Keep edge IDs, endpoints, and observable labels unchanged. A fresh
        # graph keeps these shot-specific weights out of subsequent calls.
        adjusted = DecodingGraph(
            self.graph.num_detectors,
            self.graph.num_observables,
            [(u, v, weights[e], mask) for e, (u, v, _, mask) in enumerate(self.graph.edges)],
        )
        # Restart growth on the original syndrome. The first correction was
        # evidence only; the second correction supplies the complete answer.
        return UnionFindDecoder(adjusted)._decode(syndrome)
```

- [ ] **Step 5: Run the new and existing decoder tests**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/decoders -q`
Expected: all tests pass, including `_correlated_union_find_test.py` (its `_correlation_rules` assertions still hold because `index_rules_by_source` returns the same tuple-of-tuples structure in the same order).

- [ ] **Step 6: Commit**

```bash
git add src/yoked/decoders/_correlations.py src/yoked/decoders/_correlations_test.py src/yoked/decoders/_correlated_union_find.py
git commit -m "Move correlation rules into a shared decoders module

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 2: Patch graphs: splitting the hub

**Files:**
- Create: `src/yoked/hierarchical/__init__.py`
- Create: `src/yoked/hierarchical/_patch_graphs.py`
- Create: `src/yoked/hierarchical/_patch_graphs_test.py`
- Create: `src/yoked/hierarchical/_fixtures.py`

**Interfaces:**
- Consumes: `DecodingGraph.from_dem`, `yoked_magic_memory_circuit`, `gen.NoiseModel.si1000`.
- Produces:
  - `PatchGraph` frozen dataclass: `patch_index: int`, `global_detector_ids: tuple[int, ...]`, `observable_ids: tuple[int, int]`, `local_dem: stim.DetectorErrorModel`, `graph: DecodingGraph`, `check_graph: DecodingGraph`; properties `num_detectors`, `check_vertices`; method `local_syndromes(global_syndromes) -> np.ndarray`.
  - `PatchGraphs` frozen dataclass: `patches`, `yoke_detector_ids: tuple[int, int]`, `num_detectors`, `num_observables`; `from_yoked_dem(dem, *, num_patches)`, `__len__`, `__getitem__`, `__iter__`, `local_syndromes(global_syndromes) -> (shots, patches, n_local)`.
  - `NUM_SECTORS = 2`.
  - `yoked_fixture(*, distance=3, shots=64, seed=7, p=0.003) -> YokedFixture` with fields `circuit`, `dem`, `patches`, `detectors (shots, n_d) bool`, `actual (shots, 12) bool`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_patch_graphs_test.py`:

```python
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
    text = str(patches[1].local_dem)
    assert 'error(0.1) D0 D1 ^ D2 D3' in text


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
])
def test_malformed_dems_are_rejected(bad, message):
    dem = stim.DetectorErrorModel(TWO_PATCH_DEM + bad)
    with pytest.raises(ValueError, match=message):
        PatchGraphs.from_yoked_dem(dem, num_patches=2)


def test_observable_count_must_match_patch_count():
    with pytest.raises(ValueError, match='observables'):
        PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(TWO_PATCH_DEM), num_patches=3)


def test_split_is_lossless_on_the_distance_3_fixture():
    fx = yoked_fixture(shots=1)
    joint = DecodingGraph.from_dem(fx.dem)

    def key(u, v, weight, mask):
        endpoints = (u, None) if v is None else (min(u, v), max(u, v))
        return endpoints, round(weight, 9), mask

    expected = collections.Counter(key(*edge) for edge in joint.edges)
    rebuilt = collections.Counter()
    for patch in fx.patches:
        to_global = dict(enumerate(patch.global_detector_ids))
        to_global.update(zip(patch.check_vertices, fx.patches.yoke_detector_ids))
        for u, v, weight, mask in patch.check_graph.edges:
            global_mask = sum(1 << patch.observable_ids[s] for s in range(NUM_SECTORS) if (mask >> s) & 1)
            rebuilt[key(to_global[u], None if v is None else to_global[v], weight, global_mask)] += 1
    assert rebuilt == expected
    assert len(fx.patches) == 6
    assert all(len(p.global_detector_ids) == (fx.dem.num_detectors - 2) // 6 for p in fx.patches)


def test_sampled_yoke_bits_are_sector_parities():
    fx = yoked_fixture(shots=200)
    yoke = fx.detectors[:, list(fx.patches.yoke_detector_ids)]
    for sector in range(NUM_SECTORS):
        parity = fx.actual[:, sector::NUM_SECTORS].sum(axis=1) % 2
        np.testing.assert_array_equal(yoke[:, sector], parity.astype(bool))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_patch_graphs_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical'`

- [ ] **Step 3: Create the package, the fixture, and the splitter**

Create `src/yoked/hierarchical/__init__.py`:

```python
"""Hierarchical L1/L2 decoding of the 1D yoked surface code.

See docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md.
"""
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph, PatchGraphs

__all__ = ['NUM_SECTORS', 'PatchGraph', 'PatchGraphs']
```

Create `src/yoked/hierarchical/_fixtures.py`:

```python
"""Test support: a small six-patch yoked fixture.

Distance 3 with 4d = 12 rounds keeps end-to-end tests at a few seconds while
exercising both yokes, all six patches, and correlated mechanisms. Library
code never imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.hierarchical._patch_graphs import PatchGraphs

NUM_PATCHES = 6
"""The fixture always uses the experiment's six patches and two yokes."""


@dataclass(frozen=True)
class YokedFixture:
    """A sampled six-patch circuit and its per-patch graphs.

    Fields: ``circuit`` and ``dem`` as built by the repository; ``patches``
    from ``PatchGraphs.from_yoked_dem``; ``detectors`` of shape (shots, n_d)
    and ``actual`` of shape (shots, 12), both boolean.
    """
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    patches: PatchGraphs
    detectors: np.ndarray
    actual: np.ndarray


def yoked_fixture(*, distance: int = 3, shots: int = 64, seed: int = 7, p: float = 0.003) -> YokedFixture:
    circuit = yoked_magic_memory_circuit(
        patch_diameter=distance, rounds=4 * distance, noise=gen.NoiseModel.si1000(p),
        style='cz', yokes=2, num_patches=NUM_PATCHES,
    )
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    detectors, actual = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True, bit_packed=False,
    )
    return YokedFixture(circuit, dem, PatchGraphs.from_yoked_dem(dem, num_patches=NUM_PATCHES),
                        detectors.astype(bool), actual.astype(bool))
```

Create `src/yoked/hierarchical/_patch_graphs.py`:

```python
"""Split the six-patch yoked decoding problem into per-patch graphs.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 4.

The six-patch DEM has two yoke detectors, the last two, each equal to the
parity of one sector's six observables (X: even observable indices, Z: odd).
Every error component that flips observable o(i, s) also contains the yoke
detector of sector s, so the yoke is a hub joining all patches. Removing the
yoke detectors leaves one connected component per (patch, sector), each
carrying exactly one observable.

For each patch i this module builds
  * ``local_dem``:   the patch's error mechanisms with local detector ids and
                     local observable ids (0 = X, 1 = Z), yoke targets dropped,
                     component structure and probabilities kept so that
                     correlation rules can be derived from it;
  * ``graph``:       DecodingGraph.from_dem(local_dem), the check-free graph
                     that L1 decodes; it has one component per sector;
  * ``check_graph``: the same edge ids and weights, with every
                     observable-flipping boundary edge re-targeted to the
                     sector's check vertex (local ids n and n + 1). Setting a
                     check bit forces a matching into that logical class, as
                     the repository's gap circuits do with their check detector.

Invariants checked by ``_patch_graphs_test.py``: the split is lossless
(merging check vertices back into the yoke vertices reproduces the six-patch
graph), every component keeps a physical detector, a component's yoke
membership equals its observable's sector, an observable-flipping edge is a
boundary edge, and one mechanism never spans two patches.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import stim

from yoked.decoders._graph import DecodingGraph

NUM_SECTORS = 2
"""Sector 0 is X (observable 2i), sector 1 is Z (observable 2i + 1)."""


@dataclass(frozen=True)
class _Component:
    """One graphlike piece of a DEM error mechanism, global ids, yokes included."""
    detectors: tuple[int, ...]
    observables: tuple[int, ...]   # flipped an odd number of times


@dataclass(frozen=True)
class _Instruction:
    """One DEM error mechanism after flattening."""
    probability: float
    components: tuple[_Component, ...]


@dataclass(frozen=True)
class PatchGraph:
    """Everything L1 needs for one surface-code patch.

    Fields:
      patch_index: i in 0..num_patches - 1.
      global_detector_ids: sorted global ids of the patch's physical detectors;
        local id k is global id ``global_detector_ids[k]``.
      observable_ids: (o(i, X), o(i, Z)) as global observable indices.
      local_dem: the patch's mechanisms with local detector and observable ids.
      graph: check-free DecodingGraph with two connected components (X, Z).
      check_graph: DecodingGraph whose vertices n and n + 1 (n = detector count)
        are the X and Z check vertices. Edge ids match ``graph`` one to one.
    """
    patch_index: int
    global_detector_ids: tuple[int, ...]
    observable_ids: tuple[int, int]
    local_dem: stim.DetectorErrorModel
    graph: DecodingGraph
    check_graph: DecodingGraph

    @property
    def num_detectors(self) -> int:
        return len(self.global_detector_ids)

    @property
    def check_vertices(self) -> tuple[int, int]:
        """Local ids of the X and Z check vertices in ``check_graph``."""
        return self.num_detectors, self.num_detectors + 1

    def local_syndromes(self, global_syndromes: np.ndarray) -> np.ndarray:
        """Gather this patch's detector bits from (n_d,) or (shots, n_d) input."""
        return np.asarray(global_syndromes)[..., list(self.global_detector_ids)]


@dataclass(frozen=True)
class PatchGraphs:
    """The per-patch view of one six-patch yoked DEM.

    Fields: ``patches`` in patch order; ``yoke_detector_ids`` = (X yoke, Z yoke)
    global ids; ``num_detectors`` and ``num_observables`` of the six-patch DEM.
    """
    patches: tuple[PatchGraph, ...]
    yoke_detector_ids: tuple[int, int]
    num_detectors: int
    num_observables: int

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> PatchGraph:
        return self.patches[index]

    def __iter__(self) -> Iterator[PatchGraph]:
        return iter(self.patches)

    def local_syndromes(self, global_syndromes: np.ndarray) -> np.ndarray:
        """Stack every patch's local syndrome: shape (..., patches, n_local)."""
        return np.stack([patch.local_syndromes(global_syndromes) for patch in self.patches], axis=-2)

    @classmethod
    def from_yoked_dem(cls, dem: stim.DetectorErrorModel, *, num_patches: int) -> PatchGraphs:
        num_detectors, num_observables = dem.num_detectors, dem.num_observables
        if num_observables != NUM_SECTORS * num_patches:
            raise ValueError(
                f'Expected {NUM_SECTORS * num_patches} observables for {num_patches} patches, found {num_observables}')
        # The circuit appends the X yoke detector first, then the Z yoke detector.
        yoke = (num_detectors - 2, num_detectors - 1)
        instructions = _parse_errors(dem)
        labels = _component_labels(instructions, num_physical=num_detectors - 2, yoke=yoke)
        observable_of_label = _observable_of_label(instructions, labels, yoke, num_observables)
        label_of_observable = {observable: label for label, observable in observable_of_label.items()}
        patches = []
        for i in range(num_patches):
            sector_labels = tuple(label_of_observable[NUM_SECTORS * i + s] for s in range(NUM_SECTORS))
            detector_ids = tuple(d for d in range(num_detectors - 2) if labels[d] in sector_labels)
            patches.append(_build_patch(i, detector_ids, instructions, labels, sector_labels, yoke))
        return cls(tuple(patches), yoke, num_detectors, num_observables)


def _parse_errors(dem: stim.DetectorErrorModel) -> list[_Instruction]:
    instructions = []
    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        probability, = instruction.args_copy()
        if probability == 0:
            continue
        components = []
        for group in instruction.target_groups():
            detectors = tuple(t.val for t in group if t.is_relative_detector_id())
            if len(set(detectors)) != len(detectors):
                raise ValueError('Repeated detector targets in a DEM component are unsupported')
            counts = collections.Counter(t.val for t in group if t.is_logical_observable_id())
            observables = tuple(sorted(o for o, n in counts.items() if n % 2))
            components.append(_Component(detectors, observables))
        instructions.append(_Instruction(probability, tuple(components)))
    return instructions


def _physical(component: _Component, yoke: tuple[int, int]) -> tuple[int, ...]:
    return tuple(d for d in component.detectors if d not in yoke)


def _validate_component(component: _Component, yoke: tuple[int, int]) -> tuple[int, ...]:
    """Return the physical detectors of a component whose shape the split supports."""
    physical = _physical(component, yoke)
    if not 1 <= len(physical) <= 2:
        raise ValueError(f'Component {component} must keep one or two physical detectors after removing the yoke')
    if len(component.observables) > 1:
        raise ValueError(f'Component {component} flips more than one observable')
    present = {d for d in component.detectors if d in yoke}
    implied = {yoke[o % NUM_SECTORS] for o in component.observables}
    if present != implied:
        raise ValueError(f'Component {component} has yoke membership {present} but its observables imply {implied}')
    return physical


def _component_labels(instructions: list[_Instruction], *, num_physical: int, yoke: tuple[int, int]) -> list[int]:
    """Label physical detectors by connected component of the yoke-free graph."""
    parent = list(range(num_physical))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for instruction in instructions:
        for component in instruction.components:
            physical = _validate_component(component, yoke)
            if len(physical) == 2:
                parent[find(physical[0])] = find(physical[1])
    return [find(v) for v in range(num_physical)]


def _observable_of_label(
        instructions: list[_Instruction], labels: list[int], yoke: tuple[int, int], num_observables: int,
) -> dict[int, int]:
    """Map each component label to the single observable it carries."""
    observable_of_label: dict[int, int] = {}
    for instruction in instructions:
        for component in instruction.components:
            if not component.observables:
                continue
            observable, = component.observables
            label = labels[_physical(component, yoke)[0]]
            if observable_of_label.setdefault(label, observable) != observable:
                raise ValueError(f'One component carries observables {observable_of_label[label]} and {observable}')
    if len(set(labels)) != num_observables:
        raise ValueError(f'Expected {num_observables} components after removing the yoke, found {len(set(labels))}')
    if set(observable_of_label.values()) != set(range(num_observables)) or len(observable_of_label) != num_observables:
        raise ValueError('Each observable must belong to exactly one component')
    return observable_of_label


def _build_patch(
        patch_index: int, detector_ids: tuple[int, ...], instructions: list[_Instruction],
        labels: list[int], sector_labels: tuple[int, ...], yoke: tuple[int, int],
) -> PatchGraph:
    local_id = {d: k for k, d in enumerate(detector_ids)}
    local_dem = stim.DetectorErrorModel()
    for instruction in instructions:
        owned = {labels[d] in sector_labels for component in instruction.components
                 for d in _physical(component, yoke)}
        if owned == {False}:
            continue
        if owned != {True}:
            raise ValueError('A DEM error mechanism spans more than one patch')
        targets = []
        for k, component in enumerate(instruction.components):
            if k:
                targets.append(stim.DemTarget.separator())
            targets.extend(stim.DemTarget.relative_detector_id(local_id[d]) for d in _physical(component, yoke))
            targets.extend(stim.DemTarget.logical_observable_id(o % NUM_SECTORS) for o in component.observables)
        local_dem.append('error', instruction.probability, targets)
    # Declare the counts so that a detector or observable absent from every
    # mechanism still exists in the local DEM.
    local_dem.append('detector', [], [stim.DemTarget.relative_detector_id(len(detector_ids) - 1)])
    local_dem.append('logical_observable', [], [stim.DemTarget.logical_observable_id(NUM_SECTORS - 1)])
    graph = DecodingGraph.from_dem(local_dem)
    observable_ids = (NUM_SECTORS * patch_index, NUM_SECTORS * patch_index + 1)
    return PatchGraph(patch_index, detector_ids, observable_ids, local_dem, graph, _with_check_vertices(graph))


def _with_check_vertices(graph: DecodingGraph) -> DecodingGraph:
    """Re-target observable-flipping boundary edges to the sector's check vertex."""
    check = (graph.num_detectors, graph.num_detectors + 1)
    edges = []
    for u, v, weight, mask in graph.edges:
        if mask == 0:
            edges.append((u, v, weight, mask))
            continue
        if v is not None:
            raise ValueError('An observable-flipping edge must be a boundary edge')
        if mask not in (1, 2):
            raise ValueError('An edge may flip only one of the two patch observables')
        edges.append((u, check[mask.bit_length() - 1], weight, mask))
    return DecodingGraph(graph.num_detectors + NUM_SECTORS, graph.num_observables, edges)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_patch_graphs_test.py -q`
Expected: all pass. The `D1 D4` case is rejected by `_observable_of_label` because one component then carries observables 0 and 2; its message contains the word `component`, as does the component-count message, so the match covers both.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/__init__.py src/yoked/hierarchical/_patch_graphs.py src/yoked/hierarchical/_patch_graphs_test.py src/yoked/hierarchical/_fixtures.py
git commit -m "Add per-patch graph split for hierarchical decoding

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 3: UF reference with the cluster gap

**Files:**
- Create: `src/yoked/hierarchical/_cluster_gap.py`
- Create: `src/yoked/hierarchical/_cluster_gap_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export `ClusterGapUnionFindDecoder`, `ClusterGapResult`)

**Interfaces:**
- Consumes: `DecodingGraph`, and from `yoked.decoders._union_find` the private `_Growth`, `_peel`, `_validate_syndromes` (already used by the decoder tests; this is the growth engine the spec says to reuse).
- Produces:
  - `ClusterGapResult` frozen dataclass: `prediction (num_observables,) bool`, `cluster_gap (num_observables,) float64`, `dijkstra_states (num_observables,) int64`, `selected_edges tuple[int, ...]`.
  - `ClusterGapUnionFindDecoder(graph).decode_with_gaps(syndrome) -> ClusterGapResult`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_cluster_gap_test.py`:

```python
import collections
import math

import numpy as np
import pytest

from yoked.decoders import DecodingGraph, UnionFindDecoder
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder, _remaining_costs
from yoked.decoders._union_find import _Growth
from yoked.hierarchical._fixtures import yoked_fixture


def _brute_force_odd_walk(graph, costs, observable):
    """Minimum cost over all simple paths in the state graph from (B, 0) to (B, 1)."""
    boundary = graph.num_detectors
    adjacency = collections.defaultdict(list)
    for e, (u, v, _, _) in enumerate(graph.edges):
        other = boundary if v is None else v
        adjacency[u].append((e, other))
        adjacency[other].append((e, u))
    best = math.inf

    def walk(vertex, parity, cost, visited):
        nonlocal best
        if (vertex, parity) == (boundary, 1):
            best = min(best, cost)
            return
        for e, other in adjacency[vertex]:
            flips = (graph.edges[e][3] >> observable) & 1
            state = (other, parity ^ flips)
            if state not in visited:
                walk(other, parity ^ flips, cost + costs[e], visited | {state})

    walk(boundary, 0, 0.0, {(boundary, 0)})
    return best


def _path_graph(boundary_weights=(1.0, 1.0)):
    # B -(flips)- 0 - 1 - 2 -(plain)- B, unit interior weights.
    w0, w2 = boundary_weights
    return DecodingGraph(3, 1, [(0, 1, 1.0, 0), (1, 2, 1.0, 0), (0, None, w0, 1), (2, None, w2, 0)])


def _assert_valid_correction(graph, syndrome, result):
    reconstructed = np.zeros(graph.num_detectors, dtype=np.uint8)
    mask = 0
    for e in result.selected_edges:
        u, v, _, label = graph.edges[e]
        reconstructed[u] ^= 1
        if v is not None:
            reconstructed[v] ^= 1
        mask ^= label
    np.testing.assert_array_equal(reconstructed, syndrome)
    assert [(mask >> k) & 1 for k in range(graph.num_observables)] == result.prediction.tolist()


def test_empty_syndrome_gap_is_the_cheapest_logical_path():
    graph = _path_graph()
    result = ClusterGapUnionFindDecoder(graph).decode_with_gaps([0, 0, 0])
    assert result.prediction.tolist() == [False]
    assert result.selected_edges == ()
    # No cluster exists, so the walk B-0-1-2-B costs every edge in full.
    assert result.cluster_gap[0] == pytest.approx(4.0)
    assert result.dijkstra_states[0] > 0


def test_cluster_spanning_both_boundaries_has_zero_gap():
    graph = _path_graph()
    syndrome = np.array([0, 1, 0], dtype=bool)
    result = ClusterGapUnionFindDecoder(graph).decode_with_gaps(syndrome)
    _assert_valid_correction(graph, syndrome, result)
    # Growth from detector 1 reaches both boundary terminals in one tied batch,
    # so both logical classes are free inside the final cluster.
    assert result.cluster_gap[0] == pytest.approx(0.0)


def test_partially_grown_edges_are_charged_their_remaining_growth():
    graph = _path_graph(boundary_weights=(3.0, 0.5))
    syndrome = np.array([1, 0, 0], dtype=bool)
    growth = _Growth(graph, syndrome)
    growth.run()
    costs = _remaining_costs(graph, growth)
    # Growth from detector 0: edges 0-1 (t=1) and 1-2 (t=2) complete, then the
    # cheap boundary at 2 (t=2.5) stops the cluster. The flipping boundary edge at 0
    # has grown 2.5 of its weight 3, leaving 0.5; internal edges cost nothing.
    np.testing.assert_allclose(costs, [0.0, 0.0, 0.5, 0.0])
    result = ClusterGapUnionFindDecoder(graph).decode_with_gaps(syndrome)
    _assert_valid_correction(graph, syndrome, result)
    assert result.cluster_gap[0] == pytest.approx(0.5)
    assert result.cluster_gap[0] == pytest.approx(_brute_force_odd_walk(graph, costs, 0))


@pytest.mark.parametrize('seed', range(6))
def test_dijkstra_agrees_with_brute_force_on_random_small_graphs(seed):
    rng = np.random.default_rng(seed)
    n = 5
    # A path keeps every detector connected to a boundary; extra chords add cycles.
    edges = [(u, u + 1, float(rng.integers(1, 5)), 0) for u in range(n - 1)]
    edges += [(u, v, float(rng.integers(1, 5)), 0) for u in range(n) for v in range(u + 2, n) if rng.random() < 0.4]
    edges += [(0, None, float(rng.integers(1, 5)), 1), (n - 1, None, float(rng.integers(1, 5)), 0)]
    graph = DecodingGraph(n, 1, edges)
    decoder = ClusterGapUnionFindDecoder(graph)
    for _ in range(4):
        syndrome = rng.random(n) < 0.4
        growth = _Growth(graph, syndrome)
        growth.run()
        costs = _remaining_costs(graph, growth)
        result = decoder.decode_with_gaps(syndrome)
        _assert_valid_correction(graph, syndrome, result)
        assert result.cluster_gap[0] == pytest.approx(_brute_force_odd_walk(graph, costs, 0))


def test_observable_without_flipping_edges_is_rejected():
    with pytest.raises(ValueError, match='exactly one component'):
        ClusterGapUnionFindDecoder(DecodingGraph(2, 1, [(0, 1, 1.0, 0), (1, None, 1.0, 0)]))


def test_reference_bits_equal_plain_uf_on_the_fixture():
    fx = yoked_fixture(shots=32)
    patch = fx.patches[2]
    local = patch.local_syndromes(fx.detectors)
    expected = UnionFindDecoder(patch.graph).decode_batch(local)
    decoder = ClusterGapUnionFindDecoder(patch.graph)
    for shot in range(len(local)):
        result = decoder.decode_with_gaps(local[shot])
        _assert_valid_correction(patch.graph, local[shot], result)
        np.testing.assert_array_equal(result.prediction, expected[shot])
        assert np.isfinite(result.cluster_gap).all() and (result.cluster_gap >= 0).all()
        assert (result.dijkstra_states > 0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_cluster_gap_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._cluster_gap'`

- [ ] **Step 3: Implement the decoder**

Create `src/yoked/hierarchical/_cluster_gap.py`:

```python
"""Union-Find reference decoding with the cluster-gap soft output.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 5.1.

For a patch graph, run the repository's weighted UF (growth then peeling) to
obtain the reference bits r and a validated correction. Then compute, per
observable k, the cluster gap phi_k of Meister, Pattison and Preskill
(arXiv:2405.07433, Definition 9) on the terminated growth state:

    cost(e) = 0                   if both endpoints lie in one cluster,
    cost(e) = max(0, w_e - g_e)   otherwise: the growth the edge still needs,

    phi_k = minimum cost of a walk from the boundary B back to B whose
            edges flip observable k an odd number of times.

Every boundary terminal is merged into one vertex B. The walk is found by
Dijkstra over states (vertex, parity) from (B, 0) to (B, 1); traversing an
edge toggles the parity when its mask has bit k. The search is restricted
to the connected component that carries observable k. The number of settled
states is recorded as the soft-output work proxy.

``_cluster_gap_test.py`` checks phi against a brute-force enumeration of odd
closed walks on small graphs, with and without partial growth, and checks
that the reference bits equal plain UF's on the distance-3 fixture.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import _Growth, _peel, _validate_syndromes


@dataclass(frozen=True)
class ClusterGapResult:
    """L1 output of one patch for one shot.

    Fields: ``prediction`` (num_observables,) bool, the reference bits r;
    ``cluster_gap`` (num_observables,) float64 in nats, ``inf`` if no odd walk
    exists; ``dijkstra_states`` (num_observables,) int64 settled states;
    ``selected_edges`` edge ids of the validated correction.
    """
    prediction: np.ndarray
    cluster_gap: np.ndarray
    dijkstra_states: np.ndarray
    selected_edges: tuple[int, ...]


class ClusterGapUnionFindDecoder:
    """Repository UF plus the cluster gap of every observable of the graph."""

    def __init__(self, graph: DecodingGraph):
        self.graph = graph
        # One extra vertex id stands for every boundary terminal at once.
        self._boundary = graph.num_detectors
        self._adjacency = tuple(self._sector_adjacency(k) for k in range(graph.num_observables))

    def _sector_adjacency(self, observable: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Adjacency of the component carrying ``observable``, terminals merged into B."""
        parent = list(range(self.graph.num_detectors))

        def find(v: int) -> int:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for u, v, _, _ in self.graph.edges:
            if v is not None:
                parent[find(u)] = find(v)
        labels = {find(u) for u, _, _, mask in self.graph.edges if (mask >> observable) & 1}
        if len(labels) != 1:
            raise ValueError(f'Observable {observable} must be flipped by edges of exactly one component')
        label, = labels
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.graph.num_detectors + 1)]
        for e, (u, v, _, _) in enumerate(self.graph.edges):
            if find(u) != label:
                continue
            other = self._boundary if v is None else v
            adjacency[u].append((e, other))
            adjacency[other].append((e, u))
        return tuple(tuple(neighbours) for neighbours in adjacency)

    def decode_with_gaps(self, syndrome: np.ndarray) -> ClusterGapResult:
        syndrome = _validate_syndromes(syndrome, self.graph.num_detectors, 1)
        growth = _Growth(self.graph, syndrome)
        growth.run()
        selected, mask = _peel(self.graph, syndrome, growth.forest)
        costs = _remaining_costs(self.graph, growth)
        gaps, states = [], []
        for observable in range(self.graph.num_observables):
            gap, settled = self._shortest_odd_walk(costs, observable)
            gaps.append(gap)
            states.append(settled)
        prediction = np.array([(mask >> k) & 1 for k in range(self.graph.num_observables)], dtype=bool)
        return ClusterGapResult(prediction, np.array(gaps, dtype=np.float64),
                                np.array(states, dtype=np.int64), selected)

    def _shortest_odd_walk(self, costs: np.ndarray, observable: int) -> tuple[float, int]:
        """Dijkstra over states 2 * vertex + parity from (B, 0) to (B, 1)."""
        adjacency = self._adjacency[observable]
        start, target = 2 * self._boundary, 2 * self._boundary + 1
        distance = {start: 0.0}
        heap = [(0.0, start)]
        settled = 0
        while heap:
            d, state = heapq.heappop(heap)
            if d > distance.get(state, math.inf):
                continue  # a stale entry superseded by a shorter one
            settled += 1
            if state == target:
                return d, settled
            vertex, parity = divmod(state, 2)
            for e, other in adjacency[vertex]:
                flips = (self.graph.edges[e][3] >> observable) & 1
                next_state = 2 * other + (parity ^ flips)
                candidate = d + costs[e]
                if candidate < distance.get(next_state, math.inf):
                    distance[next_state] = candidate
                    heapq.heappush(heap, (candidate, next_state))
        return math.inf, settled


def _remaining_costs(graph: DecodingGraph, growth: _Growth) -> np.ndarray:
    """Edge costs on the terminated growth state: 0 inside a cluster, else remaining growth."""
    costs = np.empty(len(graph.edges), dtype=np.float64)
    for e, (u, terminal) in enumerate(graph.endpoints):
        # At termination every rate is zero, so settling adds nothing; it keeps
        # this function correct if the growth engine ever stops early.
        growth._settle(e)
        if growth.find(u) == growth.find(terminal):
            costs[e] = 0.0
        else:
            costs[e] = max(0.0, graph.edges[e][2] - growth.grown[e])
    return costs
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._cluster_gap import ClusterGapResult, ClusterGapUnionFindDecoder
```

and extend `__all__` with `'ClusterGapResult', 'ClusterGapUnionFindDecoder'`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_cluster_gap_test.py -q`
Expected: all pass. If `test_partially_grown_edges_are_charged_their_remaining_growth` reports a cost other than 0.5 on the flipping boundary edge, print `growth.grown` and `growth.time`; the growth engine's batch tolerance can settle the edge at the batch time, so a mismatch indicates a misunderstanding of the fixture rather than of the cost rule, and the fixture weights should be adjusted so that the boundary at 2 completes strictly before the boundary at 0.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_cluster_gap.py src/yoked/hierarchical/_cluster_gap_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add UF reference decoding with the cluster-gap soft output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 4: Matching gaps: forced weights, plain and correlated

**Files:**
- Create: `src/yoked/hierarchical/_matching_gaps.py`
- Create: `src/yoked/hierarchical/_matching_gaps_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export `MatchingGaps`, `ForcedWeights`, `signed_gaps`)

**Interfaces:**
- Consumes: `PatchGraph` (Task 2), `index_rules_by_source`, `apply_correlation_rules`, `correlation_rules_from_dem` (Task 1), PyMatching `Matching.from_check_matrix`, `decode_to_edges_array`, `decode_batch(return_weights=True)`.
- Produces:
  - `ForcedWeights` frozen dataclass: `plain (2, 2) float64`, `correlated (2, 2) float64`, `first_pass (2,) bool`, `correlated_prediction (2,) bool`, `rules_fired bool`.
  - `MatchingGaps(patch, correlation_rules).forced_weights(local_syndrome) -> ForcedWeights`.
  - `signed_gaps(forced, reference) -> np.ndarray` for `forced` of shape `(..., 2, 2)` and `reference` of shape `(..., 2)`, returning `(..., 2)`.
  - `CHECK_PATTERNS`, the four `(c_X, c_Z)` rows.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_matching_gaps_test.py`:

```python
import math

import numpy as np
import pymatching
import pytest
import stim

from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._fixtures import yoked_fixture
from yoked.hierarchical._matching_gaps import MatchingGaps, signed_gaps
from yoked.hierarchical._patch_graphs import PatchGraphs

ADDITIVITY_TOLERANCE = 1e-9  # spec test 4: sector weights add exactly up to float rounding


def _gaps_for(patch):
    return MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))


def test_signed_gaps_index_the_forced_weights_by_reference_class():
    forced = np.array([[0.0, 1.0], [2.0, 3.5]])  # W[c_X, c_Z]
    np.testing.assert_allclose(signed_gaps(forced, [0, 0]), [2.0, 1.0])
    np.testing.assert_allclose(signed_gaps(forced, [1, 1]), [1.0 - 3.5, 2.0 - 3.5])
    batch = np.stack([forced, forced])
    np.testing.assert_allclose(signed_gaps(batch, [[0, 0], [1, 0]]), [[2.0, 1.0], [-2.0, 1.5]])


def test_forced_weights_agree_with_pymatching_on_the_fixture():
    fx = yoked_fixture(shots=48)
    patch = fx.patches[0]
    gaps = _gaps_for(patch)
    reference = pymatching.Matching.from_detector_error_model(patch.local_dem)
    local = patch.local_syndromes(fx.detectors)
    expected_prediction, expected_weight = reference.decode_batch(local.astype(np.uint8), return_weights=True)
    fired = 0
    for shot in range(len(local)):
        f = gaps.forced_weights(local[shot])
        # The unforced optimum is the cheapest class, and sectors add up (spec test 4).
        assert f.plain.min() == pytest.approx(expected_weight[shot], abs=ADDITIVITY_TOLERANCE)
        assert f.plain[0, 0] + f.plain[1, 1] == pytest.approx(f.plain[0, 1] + f.plain[1, 0], abs=ADDITIVITY_TOLERANCE)
        plain_gap = signed_gaps(f.plain, f.first_pass)
        assert (plain_gap >= -ADDITIVITY_TOLERANCE).all()
        for s in range(2):
            if plain_gap[s] > ADDITIVITY_TOLERANCE:
                assert f.first_pass[s] == bool(expected_prediction[shot, s])
        # Under one reweighted model the same two invariants hold (spec test 5).
        assert f.correlated[0, 0] + f.correlated[1, 1] == pytest.approx(
            f.correlated[0, 1] + f.correlated[1, 0], abs=ADDITIVITY_TOLERANCE)
        assert (signed_gaps(f.correlated, f.correlated_prediction) >= -ADDITIVITY_TOLERANCE).all()
        if not f.rules_fired:
            np.testing.assert_array_equal(f.correlated, f.plain)
            np.testing.assert_array_equal(f.correlated_prediction, f.first_pass)
        fired += f.rules_fired
    assert fired > 0  # SI1000 mechanisms decompose, so some shots must exercise the correlated path


def test_hand_built_correlation_lowers_the_partner_and_widens_the_gap():
    # One patch. X sector: D0 - D1 with a flipping boundary at D0 and a plain boundary at D1.
    # Z sector: D2 - D3 likewise. D4 and D5 are the yokes. Each detector has one boundary edge,
    # as in the real graphs, so the importer merges nothing.
    dem = stim.DetectorErrorModel('''
        error(0.45) D0 D4 L0 ^ D2 D5 L1
        error(0.10) D0 D1
        error(0.30) D1
        error(0.10) D2 D3
        error(0.30) D3
    ''')
    patch = PatchGraphs.from_yoked_dem(dem, num_patches=1)[0]
    gaps = _gaps_for(patch)
    flip = math.log(0.55 / 0.45)                          # the observable-flipping boundary edge
    path = math.log(0.90 / 0.10) + math.log(0.70 / 0.30)  # the detour through the plain boundary
    f = gaps.forced_weights(np.array([1, 0, 1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(f.first_pass, [True, True])
    np.testing.assert_allclose(f.plain, [[2 * path, path + flip], [path + flip, 2 * flip]])
    # Selecting D0's flipping edge implies D2's with probability min(1/2, .45/.45) = 1/2, weight 0,
    # and symmetrically, so under the reweighted model both flipping edges cost nothing.
    assert f.rules_fired
    np.testing.assert_allclose(f.correlated, [[2 * path, path], [path, 0.0]], atol=1e-12)
    np.testing.assert_array_equal(f.correlated_prediction, [True, True])
    np.testing.assert_allclose(signed_gaps(f.plain, f.first_pass), [path - flip, path - flip])
    np.testing.assert_allclose(signed_gaps(f.correlated, f.first_pass), [path, path])


def test_syndrome_shape_is_validated():
    fx = yoked_fixture(shots=1)
    with pytest.raises(ValueError, match='detector bits'):
        _gaps_for(fx.patches[0]).forced_weights(np.zeros(3, dtype=np.uint8))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_matching_gaps_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._matching_gaps'`

- [ ] **Step 3: Implement the module**

Create `src/yoked/hierarchical/_matching_gaps.py`:

```python
"""Minimum-weight matching in each logical class: forced weights and gaps.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 5.2.

For a patch, W(c_X, c_Z) is the minimum matching weight on the check graph
with the X check bit c_X and the Z check bit c_Z. Because the two sectors
are disconnected, W(c_X, c_Z) = W_X(c_X) + W_Z(c_Z). The signed gap of
sector X relative to reference bits (r_X, r_Z) is

    delta_X = W(1 - r_X, r_Z) - W(r_X, r_Z),

and symmetrically for Z; a negative gap means the matcher prefers the
complement of the reference.

Two variants share one code path. Plain: the original weights. Correlated:
an unforced first pass on the check-free graph selects edges; the DEM
correlation rules lower the weights of their partners; the four forced
decodes then run on the check graph with those weights, so both classes are
compared under one reweighted model. The unforced second pass on the
reweighted check-free graph is recorded for validation only.

Matchers are built from DecodingGraph edge lists through a check matrix so
that adjusted weights can be supplied per shot. ``_matching_gaps_test.py``
checks equivalence with PyMatching's DEM import, sector additivity,
consistency of the forced argmin with the unforced predictions, and a
hand-built mechanism whose correlation changes a gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pymatching
import scipy.sparse

from yoked.decoders._correlations import CorrelationRule, apply_correlation_rules, index_rules_by_source
from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph

CHECK_PATTERNS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
"""Rows are (c_X, c_Z) in the order that reshapes to an array indexed [c_X, c_Z]."""


@dataclass(frozen=True)
class ForcedWeights:
    """Forced class weights of one patch for one shot.

    Fields: ``plain`` and ``correlated`` are (2, 2) float64 arrays in nats
    indexed [c_X, c_Z]; ``first_pass`` (2,) bool is the unforced plain
    prediction; ``correlated_prediction`` (2,) bool is the unforced second
    pass under the reweighted model; ``rules_fired`` records whether any
    correlation rule lowered a weight (when false the correlated arrays
    equal the plain ones).
    """
    plain: np.ndarray
    correlated: np.ndarray
    first_pass: np.ndarray
    correlated_prediction: np.ndarray
    rules_fired: bool


class _CheckMatrixGraph:
    """A DecodingGraph as PyMatching check and fault matrices, for per-shot matchers."""

    def __init__(self, graph: DecodingGraph):
        rows, cols = [], []
        for e, (u, v, _, _) in enumerate(graph.edges):
            rows.append(u)
            cols.append(e)
            if v is not None:
                rows.append(v)
                cols.append(e)
        # A column with a single entry is a boundary edge in PyMatching.
        self.check_matrix = scipy.sparse.csc_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(graph.num_detectors, len(graph.edges)))
        fault_rows, fault_cols = [], []
        for e, (_, _, _, mask) in enumerate(graph.edges):
            for k in range(graph.num_observables):
                if (mask >> k) & 1:
                    fault_rows.append(k)
                    fault_cols.append(e)
        self.faults_matrix = scipy.sparse.csc_matrix(
            (np.ones(len(fault_rows)), (fault_rows, fault_cols)), shape=(graph.num_observables, len(graph.edges)))
        self.weights = np.array([weight for _, _, weight, _ in graph.edges], dtype=np.float64)
        self.masks = [mask for _, _, _, mask in graph.edges]
        self.edge_ids = {
            (u, -1) if v is None else (min(u, v), max(u, v)): e for e, (u, v, _, _) in enumerate(graph.edges)
        }

    def matcher(self, weights: Sequence[float] | None = None) -> pymatching.Matching:
        weights = self.weights if weights is None else np.asarray(weights, dtype=np.float64)
        return pymatching.Matching.from_check_matrix(
            self.check_matrix, weights=weights, faults_matrix=self.faults_matrix, merge_strategy='disallow')

    def edge_ids_of(self, edges_array: np.ndarray) -> list[int]:
        """Map PyMatching's [u, v] rows (v = -1 for a boundary) to edge ids."""
        return [
            self.edge_ids[(int(u), -1) if v < 0 else (min(int(u), int(v)), max(int(u), int(v)))]
            for u, v in edges_array
        ]


class MatchingGaps:
    """Forced class weights of one patch, plain and correlated."""

    def __init__(self, patch: PatchGraph, correlation_rules: Sequence[CorrelationRule]):
        self.patch = patch
        self._rules = index_rules_by_source(patch.graph, correlation_rules)
        self._free = _CheckMatrixGraph(patch.graph)
        self._check = _CheckMatrixGraph(patch.check_graph)
        self._plain_free = self._free.matcher()
        self._plain_check = self._check.matcher()

    def forced_weights(self, local_syndrome: np.ndarray) -> ForcedWeights:
        syndrome = np.asarray(local_syndrome, dtype=np.uint8)
        if syndrome.shape != (self.patch.num_detectors,):
            raise ValueError(f'Expected {self.patch.num_detectors} detector bits, got shape {syndrome.shape}')
        selected = self._free.edge_ids_of(self._plain_free.decode_to_edges_array(syndrome))
        first_pass = self._prediction_of(selected)
        plain = self._forced(self._plain_check, syndrome)
        adjusted = apply_correlation_rules(self._free.weights, self._rules, selected)
        if adjusted is None:
            return ForcedWeights(plain, plain.copy(), first_pass, first_pass.copy(), False)
        correlated_prediction = self._free.matcher(adjusted).decode(syndrome).astype(bool)
        correlated = self._forced(self._check.matcher(adjusted), syndrome)
        return ForcedWeights(plain, correlated, first_pass, correlated_prediction, True)

    def _prediction_of(self, selected: Sequence[int]) -> np.ndarray:
        mask = 0
        for e in selected:
            mask ^= self._free.masks[e]
        return np.array([(mask >> k) & 1 for k in range(NUM_SECTORS)], dtype=bool)

    def _forced(self, matcher: pymatching.Matching, syndrome: np.ndarray) -> np.ndarray:
        """Decode the four check patterns at once; returns W indexed [c_X, c_Z]."""
        rows = np.concatenate(
            [np.broadcast_to(syndrome, (len(CHECK_PATTERNS), len(syndrome))), CHECK_PATTERNS], axis=1)
        _, weights = matcher.decode_batch(rows, return_weights=True)
        return np.asarray(weights, dtype=np.float64).reshape(NUM_SECTORS, NUM_SECTORS)


def signed_gaps(forced: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Gaps relative to reference bits, sector by sector, for any leading shape.

    ``forced`` has shape (..., 2, 2) indexed [c_X, c_Z]; ``reference`` has
    shape (..., 2). The result (..., 2) holds delta_X and delta_Z.
    """
    forced = np.asarray(forced, dtype=np.float64)
    reference = np.asarray(reference).astype(np.intp)
    r_x, r_z = reference[..., 0], reference[..., 1]

    def pick(c_x: np.ndarray, c_z: np.ndarray) -> np.ndarray:
        by_x = np.take_along_axis(forced, c_x[..., None, None], axis=-2)[..., 0, :]
        return np.take_along_axis(by_x, c_z[..., None], axis=-1)[..., 0]

    base = pick(r_x, r_z)
    return np.stack([pick(1 - r_x, r_z) - base, pick(r_x, 1 - r_z) - base], axis=-1)
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._matching_gaps import CHECK_PATTERNS, ForcedWeights, MatchingGaps, signed_gaps
```

and extend `__all__` with `'CHECK_PATTERNS', 'ForcedWeights', 'MatchingGaps', 'signed_gaps'`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_matching_gaps_test.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_matching_gaps.py src/yoked/hierarchical/_matching_gaps_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add forced matching weights and signed gaps for patch graphs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 5: Exact L2 outer decoder

**Files:**
- Create: `src/yoked/hierarchical/_outer_decoder.py`
- Create: `src/yoked/hierarchical/_outer_decoder_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export `OuterDecision`, `exact_outer_map`, `exact_outer_map_batch`, `frame_adjusted_syndrome`, `TIE_TOLERANCE`)

**Interfaces:**
- Consumes: NumPy only.
- Produces:
  - `OuterDecision(pattern: np.ndarray (n,) bool, tied: bool)`.
  - `exact_outer_map(q, parity, candidates=None) -> OuterDecision`.
  - `exact_outer_map_batch(q (shots, n), parity (shots,), candidates (shots, n) | None) -> tuple[patterns (shots, n) bool, tied (shots,) bool]`.
  - `frame_adjusted_syndrome(yoke (..., 2) bool, reference (..., 2 * patches) bool) -> (..., 2) bool`.
  - `TIE_TOLERANCE = 1e-9`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_outer_decoder_test.py`:

```python
import numpy as np
import pytest

from yoked.hierarchical._outer_decoder import (
    TIE_TOLERANCE, exact_outer_map, exact_outer_map_batch, frame_adjusted_syndrome,
)


def _analytic_rule(q, parity, candidates=None):
    """Spec section 7: threshold at one half, then the cheapest single toggle fixes parity."""
    q = np.asarray(q, dtype=float)
    n = len(q)
    candidates = np.ones(n, dtype=bool) if candidates is None else np.asarray(candidates, dtype=bool)
    lam = np.log((1 - q) / q)
    b = (q > 0.5) & candidates
    cost = np.where(candidates, np.abs(lam), np.inf)
    zero_cost = candidates & np.isclose(lam, 0.0, atol=1e-12)
    if b.sum() % 2 == parity:
        # Toggling a zero-cost (q = 1/2) bit yields an equal-weight pattern of higher binary value.
        return b, bool(zero_cost.any())
    if not np.isfinite(cost).any():
        raise ValueError('no candidate')
    best = cost[np.isfinite(cost)].min()
    tied_bits = np.flatnonzero(np.isclose(cost, best, atol=TIE_TOLERANCE))
    set_bits = [i for i in tied_bits if b[i]]
    # Lowest binary value: clearing the largest set bit beats setting any clear bit.
    i = max(set_bits) if set_bits else int(tied_bits.min())
    x = b.copy()
    x[i] ^= True
    return x, len(tied_bits) > 1


GRID = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98])


@pytest.mark.parametrize('seed', range(20))
def test_enumeration_matches_the_analytic_rule(seed):
    rng = np.random.default_rng(seed)
    for n in (1, 2, 6, 8):
        for _ in range(25):
            q = rng.choice(GRID, size=n)
            parity = int(rng.integers(2))
            candidates = None if rng.random() < 0.5 else rng.random(n) < 0.6
            if candidates is not None and parity == 1 and not candidates.any():
                with pytest.raises(ValueError, match='no candidate'):
                    exact_outer_map(q, parity, candidates)
                continue
            expected, expected_tie = _analytic_rule(q, parity, candidates)
            decision = exact_outer_map(q, parity, candidates)
            np.testing.assert_array_equal(decision.pattern, expected)
            assert decision.tied == expected_tie


def test_two_error_rescue_with_unfired_yoke():
    q = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    assert exact_outer_map(q, 0).pattern.tolist() == [1, 1, 0, 0, 0, 0]
    assert exact_outer_map(q, 1).pattern.tolist() == [1, 0, 0, 0, 0, 0]


def test_rank_preserving_transformation_can_change_the_map_pattern():
    q = np.array([0.45, 0.4, 0.1, 0.1, 0.1, 0.1])
    assert not exact_outer_map(q, 0).pattern.any()
    stretched = np.sqrt(q)   # strictly increasing, so rankings are unchanged
    assert exact_outer_map(stretched, 0).pattern.tolist() == [1, 1, 0, 0, 0, 0]


def test_ties_break_to_the_lowest_binary_value_and_are_reported():
    decision = exact_outer_map([0.5, 0.5, 0.1], 0)
    assert decision.pattern.tolist() == [0, 0, 0] and decision.tied
    decision = exact_outer_map([0.1, 0.1, 0.1], 1)
    assert decision.pattern.tolist() == [1, 0, 0] and decision.tied
    decision = exact_outer_map([0.7, 0.3, 0.1], 0)   # clearing bit 0 costs the same as setting bit 1
    assert decision.pattern.tolist() == [0, 0, 0] and decision.tied


def test_candidate_restriction_fixes_non_candidates_to_zero():
    q = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    decision = exact_outer_map(q, 1, candidates=[False, True, True, True, True, True])
    assert decision.pattern.tolist() == [0, 1, 0, 0, 0, 0] and decision.tied
    decision = exact_outer_map(q, 0, candidates=[False, True, True, True, True, True])
    assert not decision.pattern.any() and not decision.tied
    with pytest.raises(ValueError, match='no candidate'):
        exact_outer_map(q, 1, candidates=[False] * 6)


def test_batch_and_single_agree_and_validate_probabilities():
    rng = np.random.default_rng(3)
    q = rng.choice(GRID, size=(40, 6))
    parity = rng.integers(2, size=40)
    candidates = rng.random((40, 6)) < 0.7
    candidates[parity == 1, 0] = True
    patterns, tied = exact_outer_map_batch(q, parity, candidates)
    for shot in range(40):
        single = exact_outer_map(q[shot], parity[shot], candidates[shot])
        np.testing.assert_array_equal(patterns[shot], single.pattern)
        assert tied[shot] == single.tied
    with pytest.raises(ValueError, match='inside'):
        exact_outer_map([0.0, 0.5], 0)
    with pytest.raises(ValueError, match='inside'):
        exact_outer_map([1.0, 0.5], 0)


def test_frame_adjusted_syndrome_is_the_residual_parity():
    yoke = np.array([[1, 0], [0, 1]], dtype=bool)
    reference = np.zeros((2, 12), dtype=bool)
    reference[0, [0, 2]] = True      # two X-sector reference flips: parity 0 -> sigma_X = 1
    reference[1, [1]] = True         # one Z-sector reference flip: parity 1 -> sigma_Z = 0
    np.testing.assert_array_equal(frame_adjusted_syndrome(yoke, reference), [[1, 0], [0, 0]])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_outer_decoder_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._outer_decoder'`

- [ ] **Step 3: Implement the module**

Create `src/yoked/hierarchical/_outer_decoder.py`:

```python
"""Exact L2 decoding for the factorized patch model.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 7.

Per sector, n residual-error probabilities q_i and the frame-adjusted
syndrome sigma (the parity of the residual errors) define the weight

    w(x) = prod_i q_i^{x_i} (1 - q_i)^{1 - x_i}

over residual patterns x in {0,1}^n with parity(x) = sigma. L2 returns the
maximum-weight feasible pattern. With a candidate mask, non-candidates are
fixed to x_i = 0. Log weights within TIE_TOLERANCE of the maximum tie; ties
are broken toward the lowest binary value (bit i is patch i) and reported.
Multiple flips are allowed, including for sigma = 0.

The frame-adjusted syndrome is sigma[s] = y[s] XOR parity(r[:, s]).

``_outer_decoder_test.py`` checks the enumeration against an analytic
threshold-and-parity rule, the fixtures named in the spec, candidate
restrictions, and the infeasible case.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

TIE_TOLERANCE = 1e-9
"""Log-weight differences below this are ties: far below any calibrated probability's resolution."""

MAX_PATCHES = 16
"""Enumeration builds 2**n patterns per shot; 16 keeps a batch to 65,536 columns."""


@dataclass(frozen=True)
class OuterDecision:
    """L2's answer for one sector: ``pattern`` (n,) bool and whether it was ``tied``."""
    pattern: np.ndarray
    tied: bool


def frame_adjusted_syndrome(yoke: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """sigma[s] = y[s] XOR parity of r over patches, for yoke (..., 2) and reference (..., 2 * patches)."""
    yoke = np.asarray(yoke, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    by_patch = reference.reshape(reference.shape[:-1] + (-1, 2))   # (..., patches, sectors)
    return yoke ^ (by_patch.sum(axis=-2) % 2).astype(bool)


def exact_outer_map(q: np.ndarray, parity: int, candidates: np.ndarray | None = None) -> OuterDecision:
    """Maximum-weight residual pattern of the required parity for one sector."""
    batch_candidates = None if candidates is None else np.asarray(candidates, dtype=bool)[None]
    patterns, tied = exact_outer_map_batch(np.asarray(q, dtype=np.float64)[None], np.asarray([parity]), batch_candidates)
    return OuterDecision(patterns[0], bool(tied[0]))


def exact_outer_map_batch(
        q: np.ndarray, parity: np.ndarray, candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``exact_outer_map``: q (shots, n), parity (shots,), candidates (shots, n) or None."""
    q = np.asarray(q, dtype=np.float64)
    parity = np.asarray(parity).astype(bool)
    if q.ndim != 2 or parity.shape != (q.shape[0],):
        raise ValueError('Expected q of shape (shots, patches) and parity of shape (shots,)')
    if q.shape[1] > MAX_PATCHES:
        raise ValueError(f'At most {MAX_PATCHES} patches are supported')
    if not ((q > 0) & (q < 1)).all():
        raise ValueError('Probabilities must lie strictly inside (0, 1)')
    patterns = _patterns(q.shape[1])                                             # (P, n)
    log_weight = np.log(q) @ patterns.T.astype(np.float64) + np.log1p(-q) @ (~patterns).T.astype(np.float64)
    feasible = ((patterns.sum(axis=1) % 2) == 1)[None, :] == parity[:, None]     # (shots, P)
    if candidates is not None:
        candidates = np.asarray(candidates, dtype=bool)
        if candidates.shape != q.shape:
            raise ValueError('candidates must have the same shape as q')
        feasible &= ~(patterns[None, :, :] & ~candidates[:, None, :]).any(axis=2)
    if not feasible.any(axis=1).all():
        raise ValueError('Odd parity with no candidate patch: no feasible pattern')
    log_weight = np.where(feasible, log_weight, -np.inf)
    best = log_weight.max(axis=1, keepdims=True)
    near = log_weight >= best - TIE_TOLERANCE
    chosen = near.argmax(axis=1)   # the first tie in pattern order has the lowest binary value
    return patterns[chosen], near.sum(axis=1) > 1


@functools.lru_cache(maxsize=None)
def _patterns(n: int) -> np.ndarray:
    """All 2**n patterns as a (2**n, n) bool array whose row p has bit i equal to (p >> i) & 1."""
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._outer_decoder import (
    TIE_TOLERANCE, OuterDecision, exact_outer_map, exact_outer_map_batch, frame_adjusted_syndrome,
)
```

and extend `__all__` accordingly.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_outer_decoder_test.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_outer_decoder.py src/yoked/hierarchical/_outer_decoder_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add exact L2 outer decoder for the factorized patch model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 6: Isotonic calibration

**Files:**
- Create: `src/yoked/hierarchical/_calibration.py`
- Create: `src/yoked/hierarchical/_calibration_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export `IsotonicCalibrator`, `CLIP`)

**Interfaces:**
- Consumes: NumPy only.
- Produces:
  - `IsotonicCalibrator` frozen dataclass with `direction: str`, `centers: np.ndarray`, `probabilities: np.ndarray`, `num_samples: int`; `fit(scores, outcomes, *, direction) -> IsotonicCalibrator`; `probability(scores) -> np.ndarray`; `to_json() -> dict`; `from_json(data) -> IsotonicCalibrator`.
  - `CLIP = 1e-6`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_calibration_test.py`:

```python
import json

import numpy as np
import pytest

from yoked.hierarchical._calibration import CLIP, IsotonicCalibrator


def test_recovers_a_decreasing_step_function():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 10, 20000)
    truth = np.where(scores < 5, 0.4, 0.05)
    outcomes = rng.random(20000) < truth
    calibrator = IsotonicCalibrator.fit(scores, outcomes, direction='decreasing')
    assert calibrator.num_samples == 20000
    assert calibrator.probability([1.0]) == pytest.approx(0.4, abs=0.03)
    assert calibrator.probability([9.0]) == pytest.approx(0.05, abs=0.02)
    probabilities = calibrator.probability(np.linspace(-1, 11, 500))
    assert (np.diff(probabilities) <= 1e-12).all()   # monotone, including constant extrapolation


def test_decreasing_fit_interpolates_between_block_centres():
    calibrator = IsotonicCalibrator.fit([1, 2, 3, 4], [1, 1, 0, 0], direction='decreasing')
    np.testing.assert_allclose(calibrator.centers, [1.5, 3.5])
    np.testing.assert_allclose(calibrator.probabilities, [1 - CLIP, CLIP])
    assert calibrator.probability([2.5]) == pytest.approx(0.5, abs=1e-6)
    assert calibrator.probability([0.0]) == pytest.approx(1 - CLIP)
    assert calibrator.probability([9.0]) == pytest.approx(CLIP)


def test_increasing_direction_pools_equal_scores():
    calibrator = IsotonicCalibrator.fit([1, 1, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 0, 1, 1, 1], direction='increasing')
    np.testing.assert_allclose(calibrator.probability([1, 2, 3]), [0.5, 0.5, 1 - CLIP])
    assert calibrator.probability([0]) == pytest.approx(0.5)


def test_violators_pool_into_one_block():
    calibrator = IsotonicCalibrator.fit([1, 2, 3, 4], [1, 1, 0, 0], direction='increasing')
    assert len(calibrator.centers) == 1
    np.testing.assert_allclose(calibrator.probability([0, 1, 2.5, 4, 9]), 0.5)


def test_clipping_keeps_log_odds_finite():
    calibrator = IsotonicCalibrator.fit([0.0, 1.0, 2.0], [0, 0, 0], direction='decreasing')
    p = calibrator.probability([1.0])
    assert p == pytest.approx(CLIP) and np.isfinite(np.log(p / (1 - p)))


def test_rejects_bad_input():
    with pytest.raises(ValueError, match='direction'):
        IsotonicCalibrator.fit([1.0], [0], direction='sideways')
    with pytest.raises(ValueError, match='finite'):
        IsotonicCalibrator.fit([np.inf, 1.0], [0, 1], direction='decreasing')
    with pytest.raises(ValueError, match='0 or 1'):
        IsotonicCalibrator.fit([1.0, 2.0], [0, 2], direction='decreasing')
    with pytest.raises(ValueError, match='length'):
        IsotonicCalibrator.fit([1.0, 2.0], [0], direction='decreasing')
    with pytest.raises(ValueError, match='empty'):
        IsotonicCalibrator.fit([], [], direction='decreasing')
    calibrator = IsotonicCalibrator.fit([1.0, 2.0], [0, 1], direction='increasing')
    with pytest.raises(ValueError, match='finite'):
        calibrator.probability([np.nan])


def test_json_round_trip():
    calibrator = IsotonicCalibrator.fit(np.arange(50) % 7, (np.arange(50) % 3) == 0, direction='decreasing')
    restored = IsotonicCalibrator.from_json(json.loads(json.dumps(calibrator.to_json())))
    grid = np.linspace(-2, 9, 200)
    np.testing.assert_array_equal(restored.probability(grid), calibrator.probability(grid))
    assert restored.direction == 'decreasing' and restored.num_samples == 50
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_calibration_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._calibration'`

- [ ] **Step 3: Implement the module**

Create `src/yoked/hierarchical/_calibration.py`:

```python
"""Isotonic calibration of soft-output scores to residual-error probabilities.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 6.

A calibrator maps a score to P(e = 1 | score) with a monotone map fit by
pool-adjacent-violators (PAV) on the calibration sample only. The direction
is fixed by the score's definition: 'decreasing' for gaps, since a larger
gap means a smaller error probability. Equal scores are pooled before the
fit. Between block centres the map is linear; beyond the fitted range it is
constant. Outputs are clipped to [CLIP, 1 - CLIP] so that log-odds stay
finite. Probabilities above one half are allowed: a refined signed gap can
favor reversing the fixed reference.

``_calibration_test.py`` checks that the fit recovers a monotone step
function, is monotone in both directions including flat blocks and
clipping, pools equal scores, rejects non-finite input, and round-trips
through JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIP = 1e-6
"""Keeps log-odds finite; far below any residual-error rate this experiment can resolve."""

DIRECTIONS = ('increasing', 'decreasing')


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A fitted monotone map from score to residual-error probability.

    Fields: ``direction`` in DIRECTIONS; ``centers`` (blocks,) increasing
    scores, each the weighted mean score of one PAV block; ``probabilities``
    (blocks,) the block means clipped to [CLIP, 1 - CLIP], monotone in
    ``direction``; ``num_samples`` used for the fit.
    """
    direction: str
    centers: np.ndarray
    probabilities: np.ndarray
    num_samples: int

    @classmethod
    def fit(cls, scores, outcomes, *, direction: str) -> IsotonicCalibrator:
        if direction not in DIRECTIONS:
            raise ValueError(f'direction must be one of {DIRECTIONS}')
        scores = np.asarray(scores, dtype=np.float64).ravel()
        outcomes = np.asarray(outcomes, dtype=np.float64).ravel()
        if len(scores) != len(outcomes):
            raise ValueError('scores and outcomes must have the same length')
        if len(scores) == 0:
            raise ValueError('Cannot fit a calibrator on an empty sample')
        if not np.isfinite(scores).all():
            raise ValueError('Scores must be finite')
        if not np.isin(outcomes, (0.0, 1.0)).all():
            raise ValueError('Outcomes must be 0 or 1')
        unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=outcomes) / counts
        order = slice(None) if direction == 'increasing' else slice(None, None, -1)
        # PAV fits a non-decreasing sequence; for a decreasing map, fit it on scores in reverse order.
        centers, values = _pool_adjacent_violators(unique[order], means[order], counts[order])
        return cls(direction, centers[order], np.clip(values[order], CLIP, 1 - CLIP), int(len(scores)))

    def probability(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('Scores must be finite')
        return np.interp(scores, self.centers, self.probabilities)

    def to_json(self) -> dict:
        return dict(direction=self.direction, centers=self.centers.tolist(),
                    probabilities=self.probabilities.tolist(), num_samples=self.num_samples)

    @classmethod
    def from_json(cls, data: dict) -> IsotonicCalibrator:
        return cls(data['direction'], np.asarray(data['centers'], dtype=np.float64),
                   np.asarray(data['probabilities'], dtype=np.float64), int(data['num_samples']))


def _pool_adjacent_violators(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted non-decreasing fit of y against the order of x.

    Returns the weighted mean x and the fitted value of each block. Adjacent
    blocks whose values violate the order are merged until none do.
    """
    centers: list[float] = []
    values: list[float] = []
    weights: list[float] = []
    for xi, yi, wi in zip(x, y, w):
        centers.append(float(xi))
        values.append(float(yi))
        weights.append(float(wi))
        while len(values) > 1 and values[-2] > values[-1]:
            total = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total
            centers[-2] = (centers[-2] * weights[-2] + centers[-1] * weights[-1]) / total
            weights[-2] = total
            del values[-1], centers[-1], weights[-1]
    return np.array(centers), np.array(values)
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._calibration import CLIP, IsotonicCalibrator
```

and extend `__all__` with `'CLIP', 'IsotonicCalibrator'`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_calibration_test.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_calibration.py src/yoked/hierarchical/_calibration_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add isotonic calibration for soft-output scores

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 7: Provenance helpers and the L1 record

**Files:**
- Create: `src/yoked/hierarchical/_provenance.py`
- Create: `src/yoked/hierarchical/_record.py`
- Create: `src/yoked/hierarchical/_record_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export `L1Record`, `by_sector`, `to_columns`)

**Interfaces:**
- Consumes: NumPy, `importlib.metadata`, `hashlib`, `subprocess` (git).
- Produces, in `_provenance.py`:
  - `REPO_ROOT: Path`, `SOURCE_FILES: tuple[str, ...]` (repo-relative paths hashed into manifests).
  - `sha256_bytes(data: bytes) -> str`, `sha256_file(path) -> str`.
  - `sample_hash(detectors: np.ndarray, actual: np.ndarray) -> str` (SHA-256 of little-endian packed detector bytes followed by packed observable bytes, the convention of the recorded runs).
  - `write_json_atomic(path, value) -> None`, `read_json(path) -> dict`.
  - `package_versions() -> dict[str, str]`, `source_hashes() -> dict[str, str]`, `git_commit() -> str | None`, `utc_now() -> str`.
- Produces, in `_record.py`:
  - `L1Record` frozen dataclass with fields `actual (shots, 2P) bool`, `yoke (shots, 2) bool`, `uf_reference`, `mwpm_reference`, `correlated_prediction`, `joint_mwpm` (each `(shots, 2P) bool`), `cluster_gap (shots, 2P) float64`, `dijkstra_states (shots, 2P) int64`, `forced_plain (shots, P, 2, 2) float64`, `forced_correlated (shots, P, 2, 2) float64`, `rows (shots,) int64`, `baselines: dict[str, np.ndarray]`; properties `shots`, `num_patches`; methods `reference(name) -> np.ndarray` for `'uf' | 'mwpm'`, `save(path)`, `load(path)` (classmethod), `subset(indices) -> L1Record`, `concatenate(records) -> L1Record` (staticmethod, sorted by `rows`, disjoint rows required).
  - `by_sector(columns (..., 2P)) -> (..., 2, P)` and `to_columns(sector_major (..., 2, P)) -> (..., 2P)`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_record_test.py`:

```python
import json

import numpy as np
import pytest

from yoked.hierarchical._provenance import sample_hash, sha256_bytes, write_json_atomic, read_json
from yoked.hierarchical._record import L1Record, by_sector, to_columns


def _synthetic_record(shots=5, patches=6, seed=0, rows=None):
    rng = np.random.default_rng(seed)
    columns = 2 * patches
    bits = lambda: rng.random((shots, columns)) < 0.3
    return L1Record(
        actual=bits(), yoke=rng.random((shots, 2)) < 0.5,
        uf_reference=bits(), mwpm_reference=bits(), correlated_prediction=bits(), joint_mwpm=bits(),
        cluster_gap=rng.uniform(0, 5, (shots, columns)),
        dijkstra_states=rng.integers(1, 100, (shots, columns)),
        forced_plain=rng.uniform(0, 5, (shots, patches, 2, 2)),
        forced_correlated=rng.uniform(0, 5, (shots, patches, 2, 2)),
        rows=np.arange(shots) if rows is None else np.asarray(rows),
    )


def test_sector_views_round_trip_and_follow_the_column_convention():
    columns = np.arange(24).reshape(2, 12)
    sectors = by_sector(columns)
    assert sectors.shape == (2, 2, 6)
    np.testing.assert_array_equal(sectors[0, 0], [0, 2, 4, 6, 8, 10])   # X sector: even columns 2i
    np.testing.assert_array_equal(sectors[0, 1], [1, 3, 5, 7, 9, 11])   # Z sector: odd columns 2i + 1
    np.testing.assert_array_equal(to_columns(sectors), columns)


def test_record_validates_shapes_and_dtypes():
    record = _synthetic_record()
    assert record.shots == 5 and record.num_patches == 6
    np.testing.assert_array_equal(record.reference('uf'), record.uf_reference)
    with pytest.raises(ValueError, match='reference'):
        record.reference('joint')
    bad = dict(record.__dict__)
    bad['yoke'] = np.zeros((5, 3), dtype=bool)
    with pytest.raises(ValueError, match='yoke'):
        L1Record(**bad)
    bad = dict(record.__dict__)
    bad['forced_plain'] = np.zeros((5, 6, 2), dtype=float)
    with pytest.raises(ValueError, match='forced_plain'):
        L1Record(**bad)


def test_record_save_load_subset_and_concatenate(tmp_path):
    record = _synthetic_record(rows=[10, 11, 12, 13, 14])
    record.baselines['joint_uf'] = np.zeros((5, 12), dtype=bool)
    record.save(tmp_path / 'record.npz')
    loaded = L1Record.load(tmp_path / 'record.npz')
    for name, value in record.__dict__.items():
        if name == 'baselines':
            np.testing.assert_array_equal(loaded.baselines['joint_uf'], value['joint_uf'])
        else:
            np.testing.assert_array_equal(getattr(loaded, name), value)
    first, second = record.subset([0, 1]), record.subset([2, 3, 4])
    merged = L1Record.concatenate([second, first])
    np.testing.assert_array_equal(merged.rows, record.rows)
    np.testing.assert_array_equal(merged.cluster_gap, record.cluster_gap)
    with pytest.raises(ValueError, match='disjoint'):
        L1Record.concatenate([first, first])


def test_sample_hash_follows_the_recorded_run_convention():
    detectors = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 1]], dtype=bool)
    actual = np.array([[0, 1]], dtype=bool)
    packed = np.packbits(detectors, axis=1, bitorder='little').tobytes()
    packed += np.packbits(actual, axis=1, bitorder='little').tobytes()
    assert sample_hash(detectors, actual) == sha256_bytes(packed)


def test_json_is_written_atomically(tmp_path):
    path = tmp_path / 'manifest.json'
    write_json_atomic(path, dict(a=1))
    assert read_json(path) == dict(a=1)
    assert not (tmp_path / 'manifest.pending').exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_record_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._provenance'`

- [ ] **Step 3: Implement the two modules**

Create `src/yoked/hierarchical/_provenance.py`:

```python
"""Provenance helpers: hashes, atomic JSON, versions, source snapshots.

Every stage writes a manifest naming its inputs by SHA-256 so that later
stages can verify what they read (spec sections 3 and 5.3). Sample hashes
follow the recorded comparison runs: the SHA-256 of the little-endian
bit-packed detector bytes followed by the packed observable bytes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
"""src/yoked/hierarchical/_provenance.py sits three levels below the repository root."""

SOURCE_FILES = (
    'src/yoked/decoders/_graph.py',
    'src/yoked/decoders/_union_find.py',
    'src/yoked/decoders/_correlations.py',
    'src/yoked/decoders/_correlated_union_find.py',
    'src/yoked/hierarchical/_patch_graphs.py',
    'src/yoked/hierarchical/_cluster_gap.py',
    'src/yoked/hierarchical/_matching_gaps.py',
    'src/yoked/hierarchical/_outer_decoder.py',
    'src/yoked/hierarchical/_calibration.py',
    'src/yoked/hierarchical/_record.py',
    'src/yoked/hierarchical/_collect.py',
    'src/yoked/hierarchical/_policies.py',
    'src/yoked/hierarchical/_replay.py',
    'src/yoked/hierarchical/_metrics.py',
)
"""Modules whose content determines a record, a calibrator, or a replay result."""

PACKAGES = ('stim', 'numpy', 'pymatching', 'sinter', 'scipy')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sample_hash(detectors: np.ndarray, actual: np.ndarray) -> str:
    """SHA-256 of packed detectors then packed observables, as in the recorded runs."""
    digest = hashlib.sha256(np.packbits(np.asarray(detectors, dtype=bool), axis=1, bitorder='little').tobytes())
    digest.update(np.packbits(np.asarray(actual, dtype=bool), axis=1, bitorder='little').tobytes())
    return digest.hexdigest()


def write_json_atomic(path: Path, value) -> None:
    """Write to a sibling .pending file and rename, so readers never see a partial file."""
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    pending.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPO_ROOT / name) for name in SOURCE_FILES if (REPO_ROOT / name).exists()}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def utc_now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
```

Create `src/yoked/hierarchical/_record.py`:

```python
"""The stored L1 outputs of one sample set.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 5.3.

Column 2i + s of a (shots, 2P) array holds patch i, sector s (0 = X, 1 = Z).
Forced-weight arrays are patch-major, (shots, P, 2, 2), indexed [c_X, c_Z].
``rows`` are the indices of the record's shots inside their parent sample,
so that pilot subsets and resumed collections keep their provenance.
``by_sector`` and ``to_columns`` convert between the column layout and the
sector-major (shots, 2, P) layout used by replay.

``_record_test.py`` checks the layout convention, shape validation, and
that save, load, subset, and concatenate preserve every array.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

REFERENCE_NAMES = ('uf', 'mwpm')
BASELINE_PREFIX = 'baseline_'


def by_sector(columns: np.ndarray) -> np.ndarray:
    """(..., 2P) columns 2i + s  ->  (..., 2, P) indexed [sector, patch]."""
    columns = np.asarray(columns)
    return columns.reshape(columns.shape[:-1] + (-1, 2)).swapaxes(-1, -2)


def to_columns(sector_major: np.ndarray) -> np.ndarray:
    """(..., 2, P) indexed [sector, patch]  ->  (..., 2P) columns 2i + s."""
    sector_major = np.asarray(sector_major)
    return sector_major.swapaxes(-1, -2).reshape(sector_major.shape[:-2] + (-1,))


@dataclass(frozen=True)
class L1Record:
    """Everything the offline stages need, evaluated once per shot.

    Fields (shots = number of collected shots, P = patches):
      actual: (shots, 2P) bool, sampled observable flips a[i, s].
      yoke: (shots, 2) bool, sampled yoke bits y[s], X then Z.
      uf_reference, mwpm_reference: (shots, 2P) bool reference bits r[i, s].
      correlated_prediction: (shots, 2P) bool, unforced second pass under the
        reweighted model, validation only.
      joint_mwpm: (shots, 2P) bool, joint PyMatching on the hub DEM, validation only.
      cluster_gap: (shots, 2P) float64, nats.
      dijkstra_states: (shots, 2P) int64, cluster-gap work proxy.
      forced_plain, forced_correlated: (shots, P, 2, 2) float64, W(c_X, c_Z) in nats.
      rows: (shots,) int64, row index of each shot in its parent sample.
      baselines: optional historical predictions, name -> (shots, 2P) bool.
    """
    actual: np.ndarray
    yoke: np.ndarray
    uf_reference: np.ndarray
    mwpm_reference: np.ndarray
    correlated_prediction: np.ndarray
    joint_mwpm: np.ndarray
    cluster_gap: np.ndarray
    dijkstra_states: np.ndarray
    forced_plain: np.ndarray
    forced_correlated: np.ndarray
    rows: np.ndarray
    baselines: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shots, columns = np.asarray(self.actual).shape
        patches = columns // 2
        expected = dict(
            actual=((shots, columns), bool), yoke=((shots, 2), bool),
            uf_reference=((shots, columns), bool), mwpm_reference=((shots, columns), bool),
            correlated_prediction=((shots, columns), bool), joint_mwpm=((shots, columns), bool),
            cluster_gap=((shots, columns), np.float64), dijkstra_states=((shots, columns), np.int64),
            forced_plain=((shots, patches, 2, 2), np.float64), forced_correlated=((shots, patches, 2, 2), np.float64),
            rows=((shots,), np.int64),
        )
        if columns % 2:
            raise ValueError('actual must have an even number of columns (two sectors per patch)')
        for name, (shape, dtype) in expected.items():
            value = np.ascontiguousarray(np.asarray(getattr(self, name)).astype(dtype, copy=False))
            if value.shape != shape:
                raise ValueError(f'{name} must have shape {shape}, got {value.shape}')
            object.__setattr__(self, name, value)
        for name, value in self.baselines.items():
            value = np.asarray(value, dtype=bool)
            if value.shape != (shots, columns):
                raise ValueError(f'baseline {name} must have shape {(shots, columns)}, got {value.shape}')
            self.baselines[name] = value

    @property
    def shots(self) -> int:
        return self.actual.shape[0]

    @property
    def num_patches(self) -> int:
        return self.actual.shape[1] // 2

    def reference(self, name: str) -> np.ndarray:
        if name not in REFERENCE_NAMES:
            raise ValueError(f'reference must be one of {REFERENCE_NAMES}, got {name!r}')
        return self.uf_reference if name == 'uf' else self.mwpm_reference

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = {name: getattr(self, name) for name in ARRAY_FIELDS}
        arrays.update({BASELINE_PREFIX + name: value for name, value in self.baselines.items()})
        return arrays

    def save(self, path: Path) -> None:
        path = Path(path)
        pending = path.with_suffix('.pending.npz')
        np.savez_compressed(pending, **self._arrays())
        pending.replace(path)

    @classmethod
    def load(cls, path: Path) -> L1Record:
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        baselines = {name[len(BASELINE_PREFIX):]: arrays.pop(name)
                     for name in list(arrays) if name.startswith(BASELINE_PREFIX)}
        return cls(**arrays, baselines=baselines)

    def subset(self, indices: Sequence[int]) -> L1Record:
        indices = np.asarray(indices)
        return L1Record(**{name: value[indices] for name, value in self._arrays().items()
                           if not name.startswith(BASELINE_PREFIX)},
                        baselines={name: value[indices] for name, value in self.baselines.items()})

    @staticmethod
    def concatenate(records: Sequence[L1Record]) -> L1Record:
        """Merge records of one parent sample, ordered by row; rows must be disjoint."""
        rows = np.concatenate([record.rows for record in records])
        if len(np.unique(rows)) != len(rows):
            raise ValueError('Records to concatenate must have disjoint rows')
        order = np.argsort(rows, kind='stable')
        names = set.intersection(*(set(record.baselines) for record in records)) if records else set()
        merged = {name: np.concatenate([getattr(record, name) for record in records])[order] for name in ARRAY_FIELDS}
        baselines = {name: np.concatenate([record.baselines[name] for record in records])[order] for name in names}
        return L1Record(**merged, baselines=baselines)


ARRAY_FIELDS = (
    'actual', 'yoke', 'uf_reference', 'mwpm_reference', 'correlated_prediction', 'joint_mwpm',
    'cluster_gap', 'dijkstra_states', 'forced_plain', 'forced_correlated', 'rows',
)
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._record import L1Record, by_sector, to_columns
```

and extend `__all__` with `'L1Record', 'by_sector', 'to_columns'`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_record_test.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_provenance.py src/yoked/hierarchical/_record.py src/yoked/hierarchical/_record_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add the L1 record format and provenance helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 8: Single-process collection and record validation

**Files:**
- Create: `src/yoked/hierarchical/_collect.py`
- Create: `src/yoked/hierarchical/_collect_test.py`

**Interfaces:**
- Consumes: `PatchGraphs`, `ClusterGapUnionFindDecoder`, `MatchingGaps`, `correlation_rules_from_dem`, `signed_gaps`, `exact_outer_map_batch`, `frame_adjusted_syndrome`, `L1Record`, `by_sector`, `to_columns`, `sample_hash`, `yoked_magic_memory_circuit`, `gen.NoiseModel`.
- Produces:
  - `CircuitParameters(distance, rounds, p, patches=6, yokes=2, style='cz', noise='si1000')` with `circuit()`, `dem()`, `to_json()`, `from_json(data)`.
  - `SampleSet(parameters, seed, detectors (shots, n_d) bool, actual (shots, 2P) bool)` with `shots`, `hash`, `sample(parameters, *, seed, shots)` (classmethod, one Stim call), `save(directory)`, `load(directory)` (classmethod), `load_recorded_run(directory)` (classmethod, the four-decoder run layout).
  - `L1Context(dem, num_patches)` with `patches`, `uf`, `gaps`, `joint`; `from_dem_text(text, num_patches)` (classmethod).
  - `collect_rows(context, detectors, actual, rows) -> L1Record`.
  - `RecordChecks` frozen dataclass and `check_record(record) -> RecordChecks` with `passed` and `to_json()`; constants `WEIGHT_TOLERANCE = 1e-9`, `COST_TOLERANCE = 1e-6`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_collect_test.py`:

```python
import numpy as np
import pytest

from yoked.hierarchical._collect import (
    CircuitParameters, L1Context, RecordChecks, SampleSet, check_record, collect_rows,
)
from yoked.hierarchical._fixtures import yoked_fixture
from yoked.hierarchical._record import by_sector

PARAMETERS = CircuitParameters(distance=3, rounds=12, p=0.003)


def test_sample_set_is_one_stim_call_with_a_stable_hash(tmp_path):
    sample = SampleSet.sample(PARAMETERS, seed=5, shots=20)
    assert sample.shots == 20 and sample.detectors.dtype == bool and sample.actual.shape == (20, 12)
    again = SampleSet.sample(PARAMETERS, seed=5, shots=20)
    assert sample.hash == again.hash
    sample.save(tmp_path)
    loaded = SampleSet.load(tmp_path)
    assert loaded.hash == sample.hash and loaded.parameters == PARAMETERS and loaded.seed == 5
    np.testing.assert_array_equal(loaded.detectors, sample.detectors)


def test_recorded_run_layout_is_loaded_and_hash_verified(tmp_path):
    sample = SampleSet.sample(PARAMETERS, seed=5, shots=8)
    np.save(tmp_path / 'detectors_packed.npy', np.packbits(sample.detectors, axis=1, bitorder='little'))
    np.save(tmp_path / 'actual_observables_packed.npy', np.packbits(sample.actual, axis=1, bitorder='little'))
    manifest = dict(parameters=dict(distance=3, rounds=12, p=0.003, patches=6, yokes=2, style='cz',
                                    noise='si1000', shots=8, seed=5),
                    input_sha256=dict(packed_detectors_then_observables_payload=sample.hash))
    (tmp_path / 'manifest.json').write_text(__import__('json').dumps(manifest))
    loaded = SampleSet.load_recorded_run(tmp_path)
    assert loaded.hash == sample.hash and loaded.parameters == PARAMETERS
    manifest['input_sha256']['packed_detectors_then_observables_payload'] = '0' * 64
    (tmp_path / 'manifest.json').write_text(__import__('json').dumps(manifest))
    with pytest.raises(ValueError, match='hash'):
        SampleSet.load_recorded_run(tmp_path)


def test_collect_rows_fills_a_valid_record_that_passes_the_checks():
    fx = yoked_fixture(shots=40, seed=11)
    context = L1Context(fx.dem, num_patches=6)
    record = collect_rows(context, fx.detectors, fx.actual, rows=np.arange(40))
    assert record.shots == 40 and record.num_patches == 6
    np.testing.assert_array_equal(record.actual, fx.actual)
    np.testing.assert_array_equal(record.yoke, fx.detectors[:, -2:])
    assert (record.cluster_gap >= 0).all() and np.isfinite(record.cluster_gap).all()
    assert (record.dijkstra_states > 0).all()
    # Every reference satisfies the yoke parity? No: patch-local references need not. Joint MWPM does.
    joint_parity = by_sector(record.joint_mwpm).sum(axis=2) % 2
    np.testing.assert_array_equal(joint_parity.astype(bool), record.yoke)
    checks = check_record(record)
    assert isinstance(checks, RecordChecks)
    assert checks.passed, checks.to_json()
    assert checks.check_parity_agreement == 1.0
    assert checks.plain_additivity_max_error <= 1e-9
    assert checks.joint_agreement_fraction > 0.5   # ties are common at distance 3; the next line is the real test
    assert checks.joint_disagreements_unexplained == 0


def test_context_from_dem_text_matches_direct_construction():
    fx = yoked_fixture(shots=4, seed=2)
    direct = L1Context(fx.dem, num_patches=6)
    from_text = L1Context.from_dem_text(str(fx.dem), num_patches=6)
    a = collect_rows(direct, fx.detectors, fx.actual, rows=np.arange(4))
    b = collect_rows(from_text, fx.detectors, fx.actual, rows=np.arange(4))
    np.testing.assert_array_equal(a.forced_correlated, b.forced_correlated)
    np.testing.assert_array_equal(a.uf_reference, b.uf_reference)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_collect_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._collect'`

- [ ] **Step 3: Implement the module**

Create `src/yoked/hierarchical/_collect.py`:

```python
"""Collect L1 outputs for a sample set into an L1Record, and validate them.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md,
sections 3, 5.3, and 10 (tests 2 to 6).

Per shot and patch, collection records the UF reference bits and cluster
gaps (ClusterGapUnionFindDecoder), the plain and correlated forced weights
with their unforced predictions (MatchingGaps), and joint PyMatching on
the hub DEM for validation. A sample set is one Stim sampling call. This
module holds the single-process building blocks; chunked parallel
collection with resume lives in the same module under ``collect_sample``
(Task 9).

``check_record`` evaluates the spec's record-level invariants:
  2. per-patch actual flips XOR to the yoke bit;
  3. every UF correction satisfied H c = s and L c = r (enforced during
     collection, so a record can only exist if it held);
  4. plain forced weights are additive across sectors and their argmin is
     the unforced prediction wherever the class weights differ;
  5. the same two properties under the correlated model;
  6. the MWPM-reference pipeline with the uncalibrated logistic map
     reproduces joint PyMatching, up to ties of equal total cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pymatching
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._matching_gaps import MatchingGaps, signed_gaps
from yoked.hierarchical._outer_decoder import exact_outer_map_batch, frame_adjusted_syndrome
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraphs
from yoked.hierarchical._provenance import read_json, sample_hash, write_json_atomic
from yoked.hierarchical._record import L1Record, by_sector, to_columns

WEIGHT_TOLERANCE = 1e-9
"""Sector weights add exactly; 1e-9 nats absorbs float summation order."""

COST_TOLERANCE = 1e-6
"""Two joint predictions with total forced cost within this are one tie (spec test 6)."""

LOGISTIC_FLOOR = 1e-300
"""The uncalibrated logistic map 1 / (1 + exp(gap)) is clipped here only to keep logs finite."""


@dataclass(frozen=True)
class CircuitParameters:
    """The yoked memory circuit of one experiment cell."""
    distance: int
    rounds: int
    p: float
    patches: int = 6
    yokes: int = 2
    style: str = 'cz'
    noise: str = 'si1000'

    def circuit(self) -> stim.Circuit:
        if self.noise != 'si1000':
            raise ValueError('Only SI1000 noise is supported')
        return yoked_magic_memory_circuit(
            patch_diameter=self.distance, rounds=self.rounds, noise=gen.NoiseModel.si1000(self.p),
            style=self.style, yokes=self.yokes, num_patches=self.patches)

    def dem(self) -> stim.DetectorErrorModel:
        return self.circuit().detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> CircuitParameters:
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SampleSet:
    """One Stim sampling call: ``detectors`` (shots, n_d) bool and ``actual`` (shots, 2P) bool."""
    parameters: CircuitParameters
    seed: int
    detectors: np.ndarray
    actual: np.ndarray

    @property
    def shots(self) -> int:
        return self.detectors.shape[0]

    @property
    def hash(self) -> str:
        return sample_hash(self.detectors, self.actual)

    @classmethod
    def sample(cls, parameters: CircuitParameters, *, seed: int, shots: int) -> SampleSet:
        # One call fixes the sampling schedule; the packed bytes are what the hash covers.
        sampler = parameters.circuit().compile_detector_sampler(seed=seed)
        detectors, actual = sampler.sample(shots=shots, separate_observables=True, bit_packed=True)
        num_detectors = parameters.circuit().num_detectors
        return cls(parameters, seed,
                   np.unpackbits(detectors, axis=1, count=num_detectors, bitorder='little').astype(bool),
                   np.unpackbits(actual, axis=1, count=2 * parameters.patches, bitorder='little').astype(bool))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'detectors_packed.npy', np.packbits(self.detectors, axis=1, bitorder='little'))
        np.save(directory / 'actual_observables_packed.npy', np.packbits(self.actual, axis=1, bitorder='little'))
        write_json_atomic(directory / 'sample.json', dict(
            parameters=self.parameters.to_json(), seed=self.seed, shots=self.shots,
            num_detectors=self.detectors.shape[1], hash=self.hash))

    @classmethod
    def load(cls, directory: Path) -> SampleSet:
        directory = Path(directory)
        info = read_json(directory / 'sample.json')
        sample = cls._from_packed(CircuitParameters.from_json(info['parameters']), info['seed'], directory,
                                  info['num_detectors'], info['shots'])
        if sample.hash != info['hash']:
            raise ValueError('Sample hash does not match sample.json')
        return sample

    @classmethod
    def load_recorded_run(cls, directory: Path) -> SampleSet:
        """Load the layout of the recorded four-decoder runs and verify their payload hash."""
        directory = Path(directory)
        manifest = read_json(directory / 'manifest.json')
        parameters = CircuitParameters.from_json(manifest['parameters'])
        num_detectors = parameters.circuit().num_detectors
        sample = cls._from_packed(parameters, manifest['parameters']['seed'], directory, num_detectors,
                                  manifest['parameters']['shots'])
        if sample.hash != manifest['input_sha256']['packed_detectors_then_observables_payload']:
            raise ValueError('Recorded sample hash does not match its manifest')
        return sample

    @classmethod
    def _from_packed(cls, parameters, seed, directory, num_detectors, shots) -> SampleSet:
        detectors = np.load(directory / 'detectors_packed.npy')
        actual = np.load(directory / 'actual_observables_packed.npy')
        sample = cls(parameters, int(seed),
                     np.unpackbits(detectors, axis=1, count=num_detectors, bitorder='little').astype(bool),
                     np.unpackbits(actual, axis=1, count=2 * parameters.patches, bitorder='little').astype(bool))
        if sample.shots != shots:
            raise ValueError(f'Expected {shots} shots, found {sample.shots}')
        return sample


class L1Context:
    """Per-process decoders for every patch, plus joint PyMatching for validation."""

    def __init__(self, dem: stim.DetectorErrorModel, num_patches: int):
        self.patches = PatchGraphs.from_yoked_dem(dem, num_patches=num_patches)
        self.uf = [ClusterGapUnionFindDecoder(patch.graph) for patch in self.patches]
        self.gaps = [MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))
                     for patch in self.patches]
        self.joint = pymatching.Matching.from_detector_error_model(dem)

    @classmethod
    def from_dem_text(cls, text: str, num_patches: int) -> L1Context:
        return cls(stim.DetectorErrorModel(text), num_patches)


def collect_rows(context: L1Context, detectors: np.ndarray, actual: np.ndarray, rows: np.ndarray) -> L1Record:
    """Evaluate L1 on the given shots; raises if any UF correction is invalid."""
    detectors = np.asarray(detectors, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    shots, patches = detectors.shape[0], len(context.patches)
    columns = NUM_SECTORS * patches
    uf_reference = np.zeros((shots, columns), dtype=bool)
    mwpm_reference = np.zeros((shots, columns), dtype=bool)
    correlated_prediction = np.zeros((shots, columns), dtype=bool)
    cluster_gap = np.zeros((shots, columns), dtype=np.float64)
    dijkstra_states = np.zeros((shots, columns), dtype=np.int64)
    forced_plain = np.zeros((shots, patches, 2, 2), dtype=np.float64)
    forced_correlated = np.zeros((shots, patches, 2, 2), dtype=np.float64)
    local = context.patches.local_syndromes(detectors)   # (shots, patches, n_local)
    for shot in range(shots):
        for i, patch in enumerate(context.patches):
            syndrome = local[shot, i]
            uf = context.uf[i].decode_with_gaps(syndrome)
            _assert_valid_correction(patch.graph, syndrome, uf.selected_edges, uf.prediction, rows[shot], i)
            forced = context.gaps[i].forced_weights(syndrome)
            span = slice(NUM_SECTORS * i, NUM_SECTORS * i + NUM_SECTORS)
            uf_reference[shot, span] = uf.prediction
            cluster_gap[shot, span] = uf.cluster_gap
            dijkstra_states[shot, span] = uf.dijkstra_states
            mwpm_reference[shot, span] = forced.first_pass
            correlated_prediction[shot, span] = forced.correlated_prediction
            forced_plain[shot, i] = forced.plain
            forced_correlated[shot, i] = forced.correlated
    joint_mwpm = context.joint.decode_batch(detectors.astype(np.uint8)).astype(bool)
    return L1Record(
        actual=actual, yoke=detectors[:, list(context.patches.yoke_detector_ids)],
        uf_reference=uf_reference, mwpm_reference=mwpm_reference,
        correlated_prediction=correlated_prediction, joint_mwpm=joint_mwpm,
        cluster_gap=cluster_gap, dijkstra_states=dijkstra_states,
        forced_plain=forced_plain, forced_correlated=forced_correlated, rows=np.asarray(rows, dtype=np.int64),
    )


def _assert_valid_correction(graph, syndrome, selected_edges, prediction, row, patch_index) -> None:
    """Spec test 3: H c = s over GF(2) and L c = r, checked on every collected shot."""
    reconstructed = np.zeros(len(graph.adjacency), dtype=np.int64)
    mask = 0
    for e in selected_edges:
        u, terminal = graph.endpoints[e]
        reconstructed[u] ^= 1
        reconstructed[terminal] ^= 1
        mask ^= graph.edges[e][3]
    if not np.array_equal(reconstructed[:graph.num_detectors].astype(bool), np.asarray(syndrome, dtype=bool)):
        raise AssertionError(f'UF correction violates H c = s at row {row}, patch {patch_index}')
    if [(mask >> k) & 1 for k in range(graph.num_observables)] != list(prediction.astype(int)):
        raise AssertionError(f'UF prediction differs from L c at row {row}, patch {patch_index}')


@dataclass(frozen=True)
class RecordChecks:
    """Outcome of the record-level invariants of spec section 10."""
    shots: int
    check_parity_agreement: float
    plain_additivity_max_error: float
    plain_argmin_disagreements: int
    correlated_additivity_max_error: float
    correlated_sign_disagreements: int
    joint_agreement_fraction: float
    joint_disagreements: int
    joint_disagreements_unexplained: int

    @property
    def passed(self) -> bool:
        return (self.check_parity_agreement == 1.0
                and self.plain_additivity_max_error <= WEIGHT_TOLERANCE
                and self.plain_argmin_disagreements == 0
                and self.correlated_additivity_max_error <= WEIGHT_TOLERANCE
                and self.correlated_sign_disagreements == 0
                and self.joint_disagreements_unexplained == 0)

    def to_json(self) -> dict:
        return dict(asdict(self), passed=self.passed)


def check_record(record: L1Record) -> RecordChecks:
    patches = record.num_patches
    # Test 2: per-patch actual flips XOR to the yoke bit.
    parity = (by_sector(record.actual).sum(axis=2) % 2).astype(bool)
    parity_agreement = float(np.mean((parity == record.yoke).all(axis=1)))
    # Tests 4 and 5: additivity and argmin consistency under both models.
    plain_error, plain_disagreements = _model_consistency(record.forced_plain, record.mwpm_reference)
    correlated_error, correlated_disagreements = _model_consistency(
        record.forced_correlated, record.correlated_prediction)
    # Test 6: MWPM reference plus plain gaps through the logistic map reproduces joint MWPM.
    reference = record.mwpm_reference
    gaps = to_columns(signed_gaps(record.forced_plain, reference.reshape(record.shots, patches, 2)).swapaxes(1, 2))
    q = np.clip(1.0 / (1.0 + np.exp(by_sector(gaps))), LOGISTIC_FLOOR, 1 - 1e-16)   # (shots, 2, P)
    sigma = frame_adjusted_syndrome(record.yoke, reference)
    x = np.stack([exact_outer_map_batch(q[:, s], sigma[:, s])[0] for s in range(NUM_SECTORS)], axis=1)
    final = reference ^ to_columns(x)
    disagree = (final != record.joint_mwpm).any(axis=1)
    cost_final = _total_forced_cost(record.forced_plain, final)
    cost_joint = _total_forced_cost(record.forced_plain, record.joint_mwpm)
    unexplained = int(np.sum(disagree & (np.abs(cost_final - cost_joint) > COST_TOLERANCE)))
    return RecordChecks(
        shots=record.shots, check_parity_agreement=parity_agreement,
        plain_additivity_max_error=plain_error, plain_argmin_disagreements=plain_disagreements,
        correlated_additivity_max_error=correlated_error, correlated_sign_disagreements=correlated_disagreements,
        joint_agreement_fraction=float(1 - disagree.mean()), joint_disagreements=int(disagree.sum()),
        joint_disagreements_unexplained=unexplained,
    )


def _model_consistency(forced: np.ndarray, prediction: np.ndarray) -> tuple[float, int]:
    """Max additivity error and count of sectors whose forced argmin contradicts the prediction."""
    additivity = np.abs(forced[:, :, 0, 0] + forced[:, :, 1, 1] - forced[:, :, 0, 1] - forced[:, :, 1, 0])
    patches = forced.shape[1]
    gaps = signed_gaps(forced, prediction.reshape(-1, patches, 2))   # (shots, P, 2), >= 0 when consistent
    disagreements = int(np.sum(gaps < -WEIGHT_TOLERANCE))
    return float(additivity.max(initial=0.0)), disagreements


def _total_forced_cost(forced: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """sum_i W_i(f[i, X], f[i, Z]) per shot, the joint matching cost of a full prediction."""
    patches = forced.shape[1]
    bits = prediction.reshape(-1, patches, 2).astype(np.intp)
    shots = np.arange(forced.shape[0])[:, None]
    return forced[shots, np.arange(patches)[None, :], bits[:, :, 0], bits[:, :, 1]].sum(axis=1)
```

Note on the gap layout in `check_record`: `signed_gaps` returns `(shots, P, 2)` for patch-major input; `.swapaxes(1, 2)` gives `(shots, 2, P)` which `to_columns` turns into columns `2i + s`, and `by_sector` returns to `(shots, 2, P)` for L2. Keep both calls so the intent reads as "columns in, sectors out".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_collect_test.py -q`
Expected: all pass. `joint_disagreements_unexplained == 0` is the assertion that matters; the agreement fraction is only a floor because equal-weight ties are common at distance 3.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_collect.py src/yoked/hierarchical/_collect_test.py
git commit -m "Add single-process L1 collection and record validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 9: Chunked parallel collection with resume and manifests

**Files:**
- Modify: `src/yoked/hierarchical/_collect.py` (append the section below)
- Modify: `src/yoked/hierarchical/_collect_test.py` (append tests)

**Interfaces:**
- Consumes: Task 8's `SampleSet`, `L1Context`, `collect_rows`, `check_record`; `L1Record`, `ARRAY_FIELDS` from `_record`; `_provenance` helpers.
- Produces:
  - `ROLES = ('evaluation', 'calibration', 'confirmation')`.
  - `CollectionSettings(role, rows, workers=1, chunk_size=64, max_chunks=None)` with `rows_summary() -> dict`.
  - `collect_sample(sample_dir, out_dir, settings) -> L1Record | None`: writes `collection.json`, `partial.npz`, `completed.npy`, `progress.json` while running, and `record.npz` plus `manifest.json` when every row is done; returns `None` when stopped by `max_chunks`; is idempotent once complete; refuses a directory whose recorded sample, rows, role, or source hashes differ.

- [ ] **Step 1: Write the failing tests**

Append to `src/yoked/hierarchical/_collect_test.py`:

```python
from dataclasses import replace

from yoked.hierarchical._collect import CollectionSettings, collect_sample
from yoked.hierarchical._provenance import read_json
from yoked.hierarchical._record import L1Record


def test_settings_validate_and_summarize_rows():
    settings = CollectionSettings(role='evaluation', rows=[5, 3, 3, 9])
    np.testing.assert_array_equal(settings.rows, [3, 5, 9])
    summary = settings.rows_summary()
    assert (summary['start'], summary['stop'], summary['count']) == (3, 10, 3) and len(summary['sha256']) == 64
    with pytest.raises(ValueError, match='role'):
        CollectionSettings(role='training', rows=[0])
    with pytest.raises(ValueError, match='empty'):
        CollectionSettings(role='evaluation', rows=[])


def test_parallel_collection_matches_single_process_and_resumes(tmp_path):
    sample = SampleSet.sample(PARAMETERS, seed=9, shots=48)
    sample.save(tmp_path / 'sample')
    settings = CollectionSettings(role='calibration', rows=np.arange(48), workers=2, chunk_size=8, max_chunks=3)
    assert collect_sample(tmp_path / 'sample', tmp_path / 'out', settings) is None
    assert np.load(tmp_path / 'out' / 'completed.npy').sum() == 24
    assert not (tmp_path / 'out' / 'record.npz').exists()

    record = collect_sample(tmp_path / 'sample', tmp_path / 'out', replace(settings, max_chunks=None))
    expected = collect_rows(L1Context(PARAMETERS.dem(), num_patches=6), sample.detectors, sample.actual, np.arange(48))
    for name in ('uf_reference', 'mwpm_reference', 'cluster_gap', 'dijkstra_states',
                 'forced_plain', 'forced_correlated', 'joint_mwpm', 'rows', 'actual', 'yoke'):
        np.testing.assert_array_equal(getattr(record, name), getattr(expected, name), err_msg=name)
    manifest = read_json(tmp_path / 'out' / 'manifest.json')
    assert manifest['role'] == 'calibration' and manifest['checks']['passed'] and manifest['shots'] == 48
    assert manifest['sample_hash'] == sample.hash and manifest['rows']['count'] == 48
    assert not (tmp_path / 'out' / 'partial.npz').exists()
    reloaded = L1Record.load(tmp_path / 'out' / 'record.npz')
    np.testing.assert_array_equal(reloaded.cluster_gap, record.cluster_gap)

    # Complete directories are idempotent, and a different row selection is refused.
    again = collect_sample(tmp_path / 'sample', tmp_path / 'out', replace(settings, max_chunks=None))
    np.testing.assert_array_equal(again.rows, record.rows)
    with pytest.raises(ValueError, match='rows'):
        collect_sample(tmp_path / 'sample', tmp_path / 'out', CollectionSettings(role='calibration', rows=np.arange(10)))


def test_row_subset_collects_only_those_rows(tmp_path):
    sample = SampleSet.sample(PARAMETERS, seed=9, shots=20)
    sample.save(tmp_path / 'sample')
    settings = CollectionSettings(role='evaluation', rows=np.arange(5, 13), workers=1, chunk_size=3)
    record = collect_sample(tmp_path / 'sample', tmp_path / 'out', settings)
    np.testing.assert_array_equal(record.rows, np.arange(5, 13))
    np.testing.assert_array_equal(record.actual, sample.actual[5:13])
    with pytest.raises(ValueError, match='exceed'):
        collect_sample(tmp_path / 'sample', tmp_path / 'out2', CollectionSettings(role='evaluation', rows=[25]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_collect_test.py -q -k "settings or parallel or subset"`
Expected: FAIL with `ImportError: cannot import name 'CollectionSettings'`

- [ ] **Step 3: Append the parallel collection to `_collect.py`**

Add these imports at the top of `src/yoked/hierarchical/_collect.py`:

```python
import multiprocessing
import time
from yoked.hierarchical._provenance import (
    git_commit, package_versions, sha256_bytes, sha256_file, source_hashes, utc_now,
)
from yoked.hierarchical._record import ARRAY_FIELDS
```

and append:

```python
ROLES = ('evaluation', 'calibration', 'confirmation')

SAVE_INTERVAL_SECONDS = 60.0
"""Partial results are flushed at most this often: a lost minute is cheap, a large write per chunk is not."""


@dataclass(frozen=True)
class CollectionSettings:
    """Which rows of a sample to collect, and how.

    Fields: ``role`` in ROLES, the dataset role recorded in the manifest;
    ``rows`` sorted unique indices into the parent sample; ``workers``
    processes; ``chunk_size`` rows per work item; ``max_chunks`` stops after
    that many chunks (controlled runs and tests), leaving a resumable state.
    """
    role: str
    rows: np.ndarray
    workers: int = 1
    chunk_size: int = 64
    max_chunks: int | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f'role must be one of {ROLES}, got {self.role!r}')
        rows = np.unique(np.asarray(self.rows, dtype=np.int64))
        if len(rows) == 0:
            raise ValueError('rows must not be empty')
        if self.workers < 1 or self.chunk_size < 1:
            raise ValueError('workers and chunk_size must be positive')
        object.__setattr__(self, 'rows', rows)

    def rows_summary(self) -> dict:
        return dict(start=int(self.rows[0]), stop=int(self.rows[-1]) + 1, count=int(len(self.rows)),
                    sha256=sha256_bytes(self.rows.tobytes()))


def collect_sample(sample_dir: Path, out_dir: Path, settings: CollectionSettings) -> L1Record | None:
    """Collect, or resume collecting, ``settings.rows`` of the saved sample into ``out_dir``."""
    sample_dir, out_dir = Path(sample_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = read_json(sample_dir / 'sample.json')
    parameters = CircuitParameters.from_json(info['parameters'])
    if settings.rows[-1] >= info['shots']:
        raise ValueError(f'rows exceed the sample of {info["shots"]} shots')
    _verify_or_create_collection(out_dir, info, settings)
    if (out_dir / 'record.npz').exists():
        return L1Record.load(out_dir / 'record.npz')
    arrays, completed = _load_or_create_partial(out_dir, settings, parameters.patches)

    pending = np.flatnonzero(~completed)
    chunks = [pending[k:k + settings.chunk_size] for k in range(0, len(pending), settings.chunk_size)]
    if settings.max_chunks is not None:
        chunks = chunks[:settings.max_chunks]
    started = last_save = time.monotonic()
    if chunks:
        dem_text = str(parameters.dem())
        initargs = (dem_text, parameters.patches, str(sample_dir), info['num_detectors'])
        with multiprocessing.get_context('forkserver').Pool(settings.workers, _init_worker, initargs) as pool:
            jobs = [(positions, settings.rows[positions]) for positions in chunks]
            for positions, record in pool.imap_unordered(_collect_positions, jobs):
                for name in ARRAY_FIELDS:
                    if name != 'rows':
                        arrays[name][positions] = getattr(record, name)
                completed[positions] = True
                _write_progress(out_dir, completed, started)
                if time.monotonic() - last_save > SAVE_INTERVAL_SECONDS:
                    _save_partial(out_dir, arrays, completed)
                    last_save = time.monotonic()
    _save_partial(out_dir, arrays, completed)
    if not completed.all():
        return None

    record = L1Record(**arrays)
    record.save(out_dir / 'record.npz')
    checks = check_record(record)
    write_json_atomic(out_dir / 'manifest.json', dict(
        stage='collect', role=settings.role, parameters=parameters.to_json(), seed=info['seed'],
        shots=record.shots, rows=settings.rows_summary(), sample_hash=info['hash'],
        sample_directory=str(sample_dir.resolve()), workers=settings.workers, chunk_size=settings.chunk_size,
        versions=package_versions(), source_sha256=source_hashes(), code_commit=git_commit(),
        checks=checks.to_json(), record_sha256=sha256_file(out_dir / 'record.npz'),
        seconds_this_run=time.monotonic() - started, created_utc=utc_now(),
    ))
    (out_dir / 'partial.npz').unlink(missing_ok=True)
    return record


def _verify_or_create_collection(out_dir: Path, info: dict, settings: CollectionSettings) -> None:
    """Pin the directory to one sample, row set, role, and source snapshot."""
    expected = dict(sample_hash=info['hash'], parameters=info['parameters'], seed=info['seed'],
                    role=settings.role, rows=settings.rows_summary(), source_sha256=source_hashes())
    path = out_dir / 'collection.json'
    if not path.exists():
        write_json_atomic(path, dict(expected, created_utc=utc_now()))
        return
    recorded = read_json(path)
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(f'Existing collection in {out_dir} differs in {key}; use a new output directory')


def _load_or_create_partial(out_dir: Path, settings: CollectionSettings, patches: int):
    if (out_dir / 'partial.npz').exists():
        with np.load(out_dir / 'partial.npz') as data:
            arrays = {name: data[name] for name in data.files}
        return arrays, np.load(out_dir / 'completed.npy')
    shots, columns = len(settings.rows), NUM_SECTORS * patches
    arrays = dict(
        actual=np.zeros((shots, columns), dtype=bool), yoke=np.zeros((shots, 2), dtype=bool),
        uf_reference=np.zeros((shots, columns), dtype=bool), mwpm_reference=np.zeros((shots, columns), dtype=bool),
        correlated_prediction=np.zeros((shots, columns), dtype=bool), joint_mwpm=np.zeros((shots, columns), dtype=bool),
        cluster_gap=np.zeros((shots, columns), dtype=np.float64), dijkstra_states=np.zeros((shots, columns), dtype=np.int64),
        forced_plain=np.zeros((shots, patches, 2, 2), dtype=np.float64),
        forced_correlated=np.zeros((shots, patches, 2, 2), dtype=np.float64),
        rows=settings.rows.copy(),
    )
    return arrays, np.zeros(shots, dtype=bool)


def _save_partial(out_dir: Path, arrays: dict, completed: np.ndarray) -> None:
    pending = out_dir / 'partial.pending.npz'
    np.savez_compressed(pending, **arrays)
    pending.replace(out_dir / 'partial.npz')
    np.save(out_dir / 'completed.pending.npy', completed)
    (out_dir / 'completed.pending.npy').replace(out_dir / 'completed.npy')


def _write_progress(out_dir: Path, completed: np.ndarray, started: float) -> None:
    done, total = int(completed.sum()), int(len(completed))
    elapsed = time.monotonic() - started
    write_json_atomic(out_dir / 'progress.json', dict(completed=done, total=total, seconds=elapsed))
    print(f'collected {done}/{total} rows in {elapsed:.0f} s', flush=True)


_WORKER: dict = {}
"""Per-process state filled by ``_init_worker``: decoders and memory-mapped sample arrays."""


def _init_worker(dem_text: str, num_patches: int, sample_dir: str, num_detectors: int) -> None:
    _WORKER['context'] = L1Context.from_dem_text(dem_text, num_patches)
    _WORKER['detectors'] = np.load(Path(sample_dir) / 'detectors_packed.npy', mmap_mode='r')
    _WORKER['actual'] = np.load(Path(sample_dir) / 'actual_observables_packed.npy', mmap_mode='r')
    _WORKER['num_detectors'] = num_detectors
    _WORKER['num_observables'] = NUM_SECTORS * num_patches


def _collect_positions(job: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, L1Record]:
    positions, rows = job
    worker = _WORKER
    detectors = np.unpackbits(worker['detectors'][rows], axis=1, count=worker['num_detectors'], bitorder='little')
    actual = np.unpackbits(worker['actual'][rows], axis=1, count=worker['num_observables'], bitorder='little')
    return positions, collect_rows(worker['context'], detectors.astype(bool), actual.astype(bool), rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_collect_test.py -q`
Expected: all pass. The parallel test takes a few seconds because each worker rebuilds the distance-3 context. If workers fail to import `yoked`, confirm `PYTHONPATH=src` is exported in the environment rather than only prefixed, since `forkserver` children inherit the environment but not the parent's `sys.path` edits.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_collect.py src/yoked/hierarchical/_collect_test.py
git commit -m "Add resumable parallel L1 collection with manifests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 10: Policies, replay, and core metrics

**Files:**
- Create: `src/yoked/hierarchical/_policies.py`
- Create: `src/yoked/hierarchical/_replay.py`
- Create: `src/yoked/hierarchical/_replay_test.py`
- Create: `src/yoked/hierarchical/_metrics.py`
- Create: `src/yoked/hierarchical/_metrics_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export the public names below)

**Interfaces:**
- Consumes: `L1Record`, `by_sector`, `to_columns` (Task 7); `signed_gaps` (Task 4); `IsotonicCalibrator` (Task 6); `exact_outer_map_batch`, `frame_adjusted_syndrome` (Task 5); `sinter.shot_error_rate_to_piece_error_rate`.
- Produces, in `_policies.py`: `DeterministicPolicy` protocol with `name: str` and `select(q0 (shots, 2, P), sigma (shots, 2)) -> M (shots, 2, P) bool`; `NoRefinement` (name `initial_only`), `RefineAll` (name `all_refined`); `policy_from_name(name)`.
- Produces, in `_replay.py`: `REFERENCES`, `SCORES`, `OUTER_RULES`; `Estimator(reference, score)` with `name`, `direction`, `parse(text)`; `estimator_scores(record, estimator) -> (shots, 2P)`; `residual_errors(record, reference) -> (shots, 2P) bool`; `Calibrators = dict[str, tuple[IsotonicCalibrator, IsotonicCalibrator]]`; `fit_calibrators(record, estimators) -> Calibrators`; `calibrated_probabilities(record, estimator, calibrators) -> (shots, 2, P)`; `ReplayConfig(initial, refined, policy, outer='mixed')` with `name`; `ReplayResult(config, final, refined, refined_patches, ties, above_half)`; `replay(record, calibrators, config, *, chunk_size=20000) -> ReplayResult`.
- Produces, in `_metrics.py`: `Rate(count, total)` with `value`, `to_json()`; `sector_failures`, `block_failures`, `residual_failure_counts`, `misattribution`, `sector_failure_by_stratum`, `PairedDifference`, `paired_bootstrap(...)`, `normalized_ler(rate, *, pieces, values=8)`, `summarize_result(record, result, reference, *, pieces) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_replay_test.py`:

```python
import numpy as np
import pytest

from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._policies import NoRefinement, RefineAll, policy_from_name
from yoked.hierarchical._record import L1Record, by_sector
from yoked.hierarchical._replay import (
    Estimator, ReplayConfig, calibrated_probabilities, estimator_scores, fit_calibrators, replay, residual_errors,
)


def _linear_calibrator():
    # Score 0 -> 0.9, score 10 -> 0.05, linear in between, constant outside.
    return IsotonicCalibrator('decreasing', np.array([0.0, 10.0]), np.array([0.9, 0.05]), 100)


def _additive_forced(delta_x, delta_z):
    """W[c_X, c_Z] = delta_x * c_X + delta_z * c_Z: gaps relative to reference (0, 0) are the deltas."""
    return np.array([[0.0, delta_z], [delta_x, delta_x + delta_z]])


def _two_patch_record():
    # Two patches, columns (0X, 0Z, 1X, 1Z). Shot 0: X yoke fires; shot 1: nothing fires.
    shots = 2
    zeros = np.zeros((shots, 4), dtype=bool)
    actual = zeros.copy()
    actual[0, 0] = True                      # patch 0's X observable really flipped in shot 0
    yoke = np.array([[1, 0], [0, 0]], dtype=bool)
    cluster_gap = np.array([[8.0, 9.0, 1.0, 9.0], [9.0, 9.0, 9.0, 9.0]])   # patch 1 looks uncertain in shot 0
    forced_correlated = np.array([
        [_additive_forced(0.5, 9.0), _additive_forced(9.0, 9.0)],          # shot 0: patch 0 now looks uncertain
        [_additive_forced(9.0, 9.0), _additive_forced(9.0, 9.0)],
    ])
    forced_plain = np.array([[_additive_forced(6.0, 9.0), _additive_forced(6.0, 9.0)]] * shots)
    return L1Record(
        actual=actual, yoke=yoke, uf_reference=zeros.copy(), mwpm_reference=zeros.copy(),
        correlated_prediction=zeros.copy(), joint_mwpm=zeros.copy(), cluster_gap=cluster_gap,
        dijkstra_states=np.ones((shots, 4), dtype=np.int64), forced_plain=forced_plain,
        forced_correlated=forced_correlated, rows=np.arange(shots),
    )


def test_estimator_parsing_and_validation():
    assert Estimator.parse('uf:gap_correlated') == Estimator('uf', 'gap_correlated')
    assert Estimator('uf', 'cluster_gap').name == 'uf:cluster_gap'
    assert Estimator('mwpm', 'gap_plain').direction == 'decreasing'
    with pytest.raises(ValueError, match='cluster_gap'):
        Estimator('mwpm', 'cluster_gap')
    with pytest.raises(ValueError, match='reference'):
        Estimator('joint', 'gap_plain')
    with pytest.raises(ValueError, match='score'):
        Estimator.parse('uf:entropy')


def test_estimator_scores_are_gaps_relative_to_the_chosen_reference():
    record = _two_patch_record()
    np.testing.assert_array_equal(estimator_scores(record, Estimator('uf', 'cluster_gap')), record.cluster_gap)
    expected = signed_gaps(record.forced_correlated, record.uf_reference.reshape(2, 2, 2)).reshape(2, 4)
    np.testing.assert_allclose(estimator_scores(record, Estimator('uf', 'gap_correlated')), expected)
    np.testing.assert_allclose(estimator_scores(record, Estimator('uf', 'gap_correlated'))[0], [0.5, 9.0, 9.0, 9.0])
    np.testing.assert_array_equal(residual_errors(record, 'uf'), record.actual)


def test_fit_calibrators_pools_patches_per_sector():
    record = _two_patch_record()
    calibrators = fit_calibrators(record, [Estimator('uf', 'cluster_gap')])
    x_calibrator, z_calibrator = calibrators['uf:cluster_gap']
    assert x_calibrator.num_samples == 4 and z_calibrator.num_samples == 4   # 2 shots x 2 patches per sector
    assert x_calibrator.direction == 'decreasing'


def test_replay_endpoints_follow_the_hand_worked_decisions():
    record = _two_patch_record()
    calibrator = _linear_calibrator()
    calibrators = {'uf:cluster_gap': (calibrator, calibrator), 'uf:gap_correlated': (calibrator, calibrator)}
    initial, refined = Estimator('uf', 'cluster_gap'), Estimator('uf', 'gap_correlated')
    q0 = calibrated_probabilities(record, initial, calibrators)
    np.testing.assert_allclose(q0[0, 0], np.interp([8.0, 1.0], [0, 10], [0.9, 0.05]))

    initial_only = replay(record, calibrators, ReplayConfig(initial, refined, NoRefinement()))
    # Shot 0, X sector fires: the least confident initial patch is patch 1, so it is flipped.
    np.testing.assert_array_equal(initial_only.final, [[0, 0, 1, 0], [0, 0, 0, 0]])
    assert not initial_only.refined.any() and not initial_only.refined_patches.any() and not initial_only.ties.any()

    all_refined = replay(record, calibrators, ReplayConfig(initial, refined, RefineAll()))
    # With refined scores patch 0 is the uncertain one, which is also the true failure.
    np.testing.assert_array_equal(all_refined.final, [[1, 0, 0, 0], [0, 0, 0, 0]])
    assert all_refined.refined.all() and all_refined.refined_patches.all()
    assert all_refined.config == 'uf:cluster_gap->gap_correlated:all_refined:mixed'
    np.testing.assert_array_equal(all_refined.above_half[0, 0], [True, False])

    restricted = replay(record, calibrators, ReplayConfig(initial, refined, RefineAll(), outer='restricted'))
    np.testing.assert_array_equal(restricted.final, all_refined.final)
    with pytest.raises(ValueError, match='no candidate'):
        replay(record, calibrators, ReplayConfig(initial, refined, NoRefinement(), outer='restricted'))


def test_replay_config_validation_and_policy_names():
    with pytest.raises(ValueError, match='same reference'):
        ReplayConfig(Estimator('uf', 'cluster_gap'), Estimator('mwpm', 'gap_plain'), NoRefinement())
    with pytest.raises(ValueError, match='outer'):
        ReplayConfig(Estimator('uf', 'cluster_gap'), Estimator('uf', 'gap_plain'), NoRefinement(), outer='loose')
    assert policy_from_name('initial_only').name == 'initial_only'
    assert policy_from_name('all_refined').name == 'all_refined'
    with pytest.raises(ValueError, match='Unknown policy'):
        policy_from_name('top_k_given_yoke:2')
```

Create `src/yoked/hierarchical/_metrics_test.py`:

```python
import numpy as np
import pytest
import sinter

from yoked.hierarchical._metrics import (
    Rate, block_failures, misattribution, normalized_ler, paired_bootstrap, residual_failure_counts,
    sector_failure_by_stratum, sector_failures,
)


def test_rates_and_strata_on_a_small_example():
    actual = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    reference = np.zeros_like(actual)
    final = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=bool)
    counts = residual_failure_counts(reference, actual)
    np.testing.assert_array_equal(counts, [[1, 0], [2, 0], [0, 0], [0, 1]])
    np.testing.assert_array_equal(sector_failures(final, actual), [[0, 0], [1, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(block_failures(final, actual), [0, 1, 1, 1])
    rates = misattribution(sector_failures(final, actual), counts)
    assert rates['X'] == Rate(0, 1) and rates['Z'] == Rate(1, 1) and rates['pooled'] == Rate(1, 2)
    by_stratum = sector_failure_by_stratum(sector_failures(final, actual), counts)
    assert by_stratum['multiple']['X'] == Rate(1, 1) and by_stratum['zero']['X'] == Rate(1, 2)
    assert Rate(0, 0).value != Rate(0, 0).value   # nan for an empty stratum, never zero
    assert Rate(1, 4).to_json() == dict(count=1, total=4, value=0.25)


def test_paired_bootstrap_is_deterministic_and_brackets_a_known_difference():
    rng = np.random.default_rng(1)
    denominator = np.ones(5000)
    a = (rng.random(5000) < 0.30).astype(float)
    b = a.copy()
    b[:500] = 0.0                       # b fails on 500 fewer shots
    first = paired_bootstrap(a, b, denominator, denominator, replicates=2000, seed=43)
    second = paired_bootstrap(a, b, denominator, denominator, replicates=2000, seed=43)
    assert first == second
    assert first.estimate_a == pytest.approx(a.mean()) and first.estimate_b == pytest.approx(b.mean())
    assert first.low < first.difference < first.high < 0
    assert first.zero_denominator_replicates == 0
    empty = paired_bootstrap(a, b, np.zeros(5000), np.zeros(5000), replicates=10, seed=1)
    assert empty.zero_denominator_replicates == 10 and np.isnan(empty.difference)


def test_normalized_ler_uses_the_sinter_piece_conversion():
    assert normalized_ler(0.2566, pieces=216) == pytest.approx(
        sinter.shot_error_rate_to_piece_error_rate(0.2566, pieces=216, values=8))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_replay_test.py src/yoked/hierarchical/_metrics_test.py -q`
Expected: FAIL with `ModuleNotFoundError` for `_policies`, `_replay`, and `_metrics`.

- [ ] **Step 3: Implement the three modules**

Create `src/yoked/hierarchical/_policies.py`:

```python
"""Refinement policies: which patch-sectors receive refined scores.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 8.

A policy sees only the initial probabilities q0, shape (shots, 2, P), and
the frame-adjusted syndromes sigma, shape (shots, 2); never q1, the actual
flips, or outcomes. It returns the request mask M, shape (shots, 2, P):
the refined probability replaces the initial one exactly where M is set.

This plan ships the two endpoints. Selective policies and exact random
controls follow in the next plan and implement the same ``select`` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DeterministicPolicy(Protocol):
    name: str

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class NoRefinement:
    """The initial-only endpoint: no refined score is ever requested."""
    name: str = 'initial_only'

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.zeros(np.shape(q0), dtype=bool)


@dataclass(frozen=True)
class RefineAll:
    """The all-refined endpoint: every patch-sector is requested, whatever sigma is."""
    name: str = 'all_refined'

    def select(self, q0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.ones(np.shape(q0), dtype=bool)


def policy_from_name(name: str) -> DeterministicPolicy:
    if name == 'initial_only':
        return NoRefinement()
    if name == 'all_refined':
        return RefineAll()
    raise ValueError(f'Unknown policy {name!r}; available: initial_only, all_refined')
```

Create `src/yoked/hierarchical/_replay.py`:

```python
"""Offline replay: estimators, calibrated probabilities, policy selection, L2.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md,
sections 6 and 8.

An estimator is a pair (reference, score). Reference bits r come from the
record; a score is the cluster gap, or a signed matching gap relative to r.
All three scores grow with confidence in r, so every calibrator is
'decreasing'. Calibrators are fit per sector, pooling the six patches.

Replay builds q0 from the initial estimator and q1 from the refined one,
asks the policy for the request mask M, substitutes q1 where M is set, and
runs the exact L2 per sector: 'mixed' lets every patch flip, 'restricted'
lets only requested patches flip. The final prediction is r XOR x.

``_replay_test.py`` checks scores against hand-computed gaps, calibrator
pooling, and hand-worked L2 decisions for both endpoints and both rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._matching_gaps import signed_gaps
from yoked.hierarchical._outer_decoder import exact_outer_map_batch, frame_adjusted_syndrome
from yoked.hierarchical._patch_graphs import NUM_SECTORS
from yoked.hierarchical._policies import DeterministicPolicy
from yoked.hierarchical._record import REFERENCE_NAMES, L1Record, by_sector, to_columns

REFERENCES = REFERENCE_NAMES
SCORES = ('cluster_gap', 'gap_plain', 'gap_correlated')
OUTER_RULES = ('mixed', 'restricted')

Calibrators = dict[str, tuple[IsotonicCalibrator, IsotonicCalibrator]]
"""Estimator name -> (X-sector calibrator, Z-sector calibrator)."""


@dataclass(frozen=True)
class Estimator:
    """A reference decoder and a confidence score defined relative to it."""
    reference: str
    score: str

    def __post_init__(self) -> None:
        if self.reference not in REFERENCES:
            raise ValueError(f'reference must be one of {REFERENCES}, got {self.reference!r}')
        if self.score not in SCORES:
            raise ValueError(f'score must be one of {SCORES}, got {self.score!r}')
        if self.score == 'cluster_gap' and self.reference != 'uf':
            raise ValueError('cluster_gap is defined for the uf reference only')

    @property
    def name(self) -> str:
        return f'{self.reference}:{self.score}'

    @property
    def direction(self) -> str:
        return 'decreasing'   # every score grows with confidence in the reference bit

    @classmethod
    def parse(cls, text: str) -> Estimator:
        reference, _, score = text.partition(':')
        return cls(reference, score)


def residual_errors(record: L1Record, reference: str) -> np.ndarray:
    """e[i, s] = a[i, s] XOR r[i, s], in column layout."""
    return record.actual ^ record.reference(reference)


def estimator_scores(record: L1Record, estimator: Estimator) -> np.ndarray:
    """Scores in column layout (shots, 2P)."""
    if estimator.score == 'cluster_gap':
        return record.cluster_gap
    forced = record.forced_plain if estimator.score == 'gap_plain' else record.forced_correlated
    reference = record.reference(estimator.reference).reshape(record.shots, record.num_patches, NUM_SECTORS)
    return signed_gaps(forced, reference).reshape(record.shots, -1)   # patch-major (i, s) is the column order


def fit_calibrators(record: L1Record, estimators: Sequence[Estimator]) -> Calibrators:
    calibrators: Calibrators = {}
    for estimator in estimators:
        scores = by_sector(estimator_scores(record, estimator))
        outcomes = by_sector(residual_errors(record, estimator.reference))
        calibrators[estimator.name] = tuple(
            IsotonicCalibrator.fit(scores[:, s].ravel(), outcomes[:, s].ravel(), direction=estimator.direction)
            for s in range(NUM_SECTORS)
        )
    return calibrators


def calibrated_probabilities(record: L1Record, estimator: Estimator, calibrators: Calibrators) -> np.ndarray:
    """Residual-error probabilities, sector-major (shots, 2, P)."""
    scores = by_sector(estimator_scores(record, estimator))
    x_calibrator, z_calibrator = calibrators[estimator.name]
    return np.stack([x_calibrator.probability(scores[:, 0]), z_calibrator.probability(scores[:, 1])], axis=1)


@dataclass(frozen=True)
class ReplayConfig:
    """One replay cell: initial and refined estimators, the policy, and the L2 rule."""
    initial: Estimator
    refined: Estimator
    policy: DeterministicPolicy
    outer: str = 'mixed'

    def __post_init__(self) -> None:
        if self.initial.reference != self.refined.reference:
            raise ValueError('initial and refined estimators must share the same reference decoder')
        if self.outer not in OUTER_RULES:
            raise ValueError(f'outer must be one of {OUTER_RULES}, got {self.outer!r}')

    @property
    def name(self) -> str:
        return f'{self.initial.name}->{self.refined.score}:{self.policy.name}:{self.outer}'


@dataclass(frozen=True)
class ReplayResult:
    """Per-shot outputs of one replay cell.

    Fields: ``config`` name; ``final`` (shots, 2P) bool predictions f = r XOR x;
    ``refined`` (shots, 2P) bool, the request mask M in column layout;
    ``refined_patches`` (shots, P) bool, U[i] = M[i, X] or M[i, Z];
    ``ties`` (shots, 2) bool, L2 ties per sector; ``above_half`` (shots, 2, P)
    bool, which mixed probabilities exceeded one half.
    """
    config: str
    final: np.ndarray
    refined: np.ndarray
    refined_patches: np.ndarray
    ties: np.ndarray
    above_half: np.ndarray


def replay(record: L1Record, calibrators: Calibrators, config: ReplayConfig, *, chunk_size: int = 20000) -> ReplayResult:
    reference = record.reference(config.initial.reference)
    sigma = frame_adjusted_syndrome(record.yoke, reference)                         # (shots, 2)
    q0 = calibrated_probabilities(record, config.initial, calibrators)             # (shots, 2, P)
    q1 = calibrated_probabilities(record, config.refined, calibrators)
    requested = config.policy.select(q0, sigma)                                     # M
    q = np.where(requested, q1, q0)
    patterns = np.zeros(q.shape, dtype=bool)
    ties = np.zeros(sigma.shape, dtype=bool)
    for s in range(NUM_SECTORS):
        for start in range(0, record.shots, chunk_size):
            block = slice(start, start + chunk_size)
            candidates = requested[block, s] if config.outer == 'restricted' else None
            patterns[block, s], ties[block, s] = exact_outer_map_batch(q[block, s], sigma[block, s], candidates)
    return ReplayResult(
        config=config.name, final=reference ^ to_columns(patterns), refined=to_columns(requested),
        refined_patches=requested.any(axis=1), ties=ties, above_half=q > 0.5,
    )
```

Create `src/yoked/hierarchical/_metrics.py`:

```python
"""Accuracy metrics for replay results.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md,
section 9: the primary misattribution metric, reference-failure strata,
sector and block failure, paired bootstrap over whole shots, and the
normalized logical error rate. Every per-shot array is aligned to the
record's rows, so two configurations on one record are paired by position.

``_metrics_test.py`` checks each rate on a hand-built example, bootstrap
determinism and interval placement, and the sinter conversion.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import sinter

from yoked.hierarchical._record import L1Record, by_sector

SECTOR_NAMES = ('X', 'Z')
STRATA = dict(zero=lambda counts: counts == 0, one=lambda counts: counts == 1, multiple=lambda counts: counts >= 2)
"""Reference-failure strata by number of residual reference failures in a sector."""

DEFAULT_REPLICATES = 10000
DEFAULT_SEED = 43
"""The bootstrap replicate count and RNG seed of the existing four-decoder comparison."""

DEFAULT_VALUES = 8
"""Logical values per shot for the [[6, 4, 2]] outer code: 2 sectors x (6 - 2) logical qubits."""


@dataclass(frozen=True)
class Rate:
    count: int
    total: int

    @property
    def value(self) -> float:
        return self.count / self.total if self.total else math.nan

    def to_json(self) -> dict:
        return dict(count=self.count, total=self.total, value=self.value)


def sector_failures(final: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(shots, 2) bool: a sector fails when any of its patch predictions is wrong."""
    return (by_sector(final) != by_sector(actual)).any(axis=2)


def block_failures(final: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(shots,) bool: the existing comparison's block failure, any observable wrong."""
    return (np.asarray(final) != np.asarray(actual)).any(axis=1)


def residual_failure_counts(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(shots, 2) int: number of patches with a residual reference failure per sector."""
    return by_sector(np.asarray(reference) != np.asarray(actual)).sum(axis=2)


def _rate(mask_numerator: np.ndarray, mask_denominator: np.ndarray) -> Rate:
    return Rate(int(np.sum(mask_numerator & mask_denominator)), int(np.sum(mask_denominator)))


def misattribution(sector_failure: np.ndarray, counts: np.ndarray) -> dict[str, Rate]:
    """Failure rate among sectors with exactly one residual reference failure."""
    eligible = counts == 1
    rates = {name: _rate(sector_failure[:, s], eligible[:, s]) for s, name in enumerate(SECTOR_NAMES)}
    rates['pooled'] = _rate(sector_failure, eligible)
    return rates


def sector_failure_by_stratum(sector_failure: np.ndarray, counts: np.ndarray) -> dict[str, dict[str, Rate]]:
    result = {}
    for stratum, predicate in STRATA.items():
        eligible = predicate(counts)
        result[stratum] = {name: _rate(sector_failure[:, s], eligible[:, s]) for s, name in enumerate(SECTOR_NAMES)}
        result[stratum]['pooled'] = _rate(sector_failure, eligible)
    return result


@dataclass(frozen=True)
class PairedDifference:
    """Bootstrap of statistic(b) - statistic(a), resampling whole shots with replacement."""
    estimate_a: float
    estimate_b: float
    difference: float
    low: float
    high: float
    replicates: int
    seed: int
    zero_denominator_replicates: int

    def to_json(self) -> dict:
        return asdict(self)


def paired_bootstrap(
        numerator_a: np.ndarray, numerator_b: np.ndarray,
        denominator_a: np.ndarray, denominator_b: np.ndarray,
        *, replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED,
) -> PairedDifference:
    """Per-shot numerators and denominators (shots,); the statistic is sum(num) / sum(den)."""
    arrays = [np.asarray(a, dtype=np.float64) for a in (numerator_a, numerator_b, denominator_a, denominator_b)]
    shots = len(arrays[0])
    rng = np.random.default_rng(seed)
    differences = np.full(replicates, np.nan)
    for k in range(replicates):
        weights = np.bincount(rng.integers(0, shots, shots), minlength=shots).astype(np.float64)
        den_a, den_b = weights @ arrays[2], weights @ arrays[3]
        if den_a > 0 and den_b > 0:
            differences[k] = (weights @ arrays[1]) / den_b - (weights @ arrays[0]) / den_a
    valid = differences[np.isfinite(differences)]
    estimate_a = arrays[0].sum() / arrays[2].sum() if arrays[2].sum() else math.nan
    estimate_b = arrays[1].sum() / arrays[3].sum() if arrays[3].sum() else math.nan
    low, high = (np.percentile(valid, [2.5, 97.5]) if len(valid) else (math.nan, math.nan))
    return PairedDifference(float(estimate_a), float(estimate_b), float(estimate_b - estimate_a),
                            float(low), float(high), replicates, seed, int(replicates - len(valid)))


def normalized_ler(block_failure_rate: float, *, pieces: int, values: int = DEFAULT_VALUES) -> float:
    """LER per patch per round by the repository's sinter piece conversion."""
    return float(sinter.shot_error_rate_to_piece_error_rate(block_failure_rate, pieces=pieces, values=values))


def summarize_result(record: L1Record, final: np.ndarray, reference: str, *, pieces: int,
                     ties: np.ndarray | None = None, above_half: np.ndarray | None = None) -> dict:
    """The per-configuration numbers of the pilot tables, as JSON-ready values."""
    counts = residual_failure_counts(record.reference(reference), record.actual)
    failures = sector_failures(final, record.actual)
    block = block_failures(final, record.actual)
    summary = dict(
        shots=record.shots,
        block_failure=Rate(int(block.sum()), record.shots).to_json(),
        normalized_ler=normalized_ler(block.mean(), pieces=pieces),
        sector_failure={name: Rate(int(failures[:, s].sum()), record.shots).to_json()
                        for s, name in enumerate(SECTOR_NAMES)},
        misattribution={name: rate.to_json() for name, rate in misattribution(failures, counts).items()},
        by_stratum={stratum: {name: rate.to_json() for name, rate in rates.items()}
                    for stratum, rates in sector_failure_by_stratum(failures, counts).items()},
        stratum_counts={stratum: int(predicate(counts).sum()) for stratum, predicate in STRATA.items()},
    )
    if ties is not None:
        summary['ties'] = int(np.sum(ties))
    if above_half is not None:
        summary['above_half_fraction'] = float(np.mean(above_half))
    return summary
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._metrics import (
    PairedDifference, Rate, block_failures, misattribution, normalized_ler, paired_bootstrap,
    residual_failure_counts, sector_failure_by_stratum, sector_failures, summarize_result,
)
from yoked.hierarchical._policies import DeterministicPolicy, NoRefinement, RefineAll, policy_from_name
from yoked.hierarchical._replay import (
    Calibrators, Estimator, ReplayConfig, ReplayResult, calibrated_probabilities, estimator_scores,
    fit_calibrators, replay, residual_errors,
)
```

and extend `__all__` with every imported name.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical -q`
Expected: all pass across the package.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_policies.py src/yoked/hierarchical/_replay.py src/yoked/hierarchical/_replay_test.py src/yoked/hierarchical/_metrics.py src/yoked/hierarchical/_metrics_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add endpoint policies, offline replay, and core metrics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 11: Stages, the CLI, and the usage doc

**Files:**
- Create: `src/yoked/hierarchical/_stages.py`
- Create: `src/yoked/hierarchical/_stages_test.py`
- Create: `tools/hierarchical_experiment` (executable)
- Create: `docs/hierarchical_decoding.md`
- Modify: `README.md` (one paragraph pointing at the usage doc, after the "Union Find decoder" section)

**Interfaces:**
- Consumes: everything from Tasks 7 to 10.
- Produces, in `_stages.py`:
  - `CollectRequest(out_dir, role, rows=None, workers=1, chunk_size=64, max_chunks=None, recorded_run=None, parameters=None, seed=None, shots=None)`.
  - `stage_collect(request) -> L1Record | None`: writes `out_dir/sample/` on first use, then delegates to `collect_sample`.
  - `stage_calibrate(record_dir, out_path, estimators) -> Calibrators`: requires role `calibration`, writes a calibrators JSON with provenance.
  - `load_calibrators(path) -> tuple[Calibrators, dict]`.
  - `parse_config(text) -> ReplayConfig` for `initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed`.
  - `config_directory_name(config) -> str`.
  - `stage_replay(record_dir, calibrators_path, out_dir, configs) -> dict[str, ReplayResult]`: refuses when the calibration sample hash equals the record's sample hash; writes per-config `results.json` and `final.npy`, and `replay_manifest.json`.
  - `stage_summarize(replay_dirs, out_path, *, replicates=10000, seed=43) -> str`: markdown tables with paired differences against each estimator pair's `initial_only` cell.

- [ ] **Step 1: Write the failing tests**

Create `src/yoked/hierarchical/_stages_test.py`:

```python
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from yoked.hierarchical._collect import CircuitParameters
from yoked.hierarchical._provenance import REPO_ROOT, read_json
from yoked.hierarchical._replay import Estimator
from yoked.hierarchical._stages import (
    CollectRequest, config_directory_name, load_calibrators, parse_config, stage_calibrate, stage_collect,
    stage_replay, stage_summarize,
)

PARAMETERS = CircuitParameters(distance=3, rounds=12, p=0.003)
ESTIMATORS = [Estimator('uf', 'cluster_gap'), Estimator('uf', 'gap_correlated'), Estimator('mwpm', 'gap_plain'),
              Estimator('mwpm', 'gap_correlated')]
CONFIGS = [
    'initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed',
    'initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed',
    'initial=mwpm:gap_plain,refined=gap_correlated,policy=initial_only,outer=mixed',
    'initial=mwpm:gap_plain,refined=gap_correlated,policy=all_refined,outer=mixed',
]


def test_parse_config_and_directory_names():
    config = parse_config(CONFIGS[1])
    assert config.name == 'uf:cluster_gap->gap_correlated:all_refined:mixed'
    assert config_directory_name(config) == 'uf_cluster_gap_to_gap_correlated_all_refined_mixed'
    with pytest.raises(ValueError, match='initial'):
        parse_config('refined=gap_plain,policy=all_refined')


def test_stages_run_end_to_end_and_refuse_to_evaluate_on_the_calibration_sample(tmp_path):
    calibration = stage_collect(CollectRequest(
        tmp_path / 'calibration', role='calibration', parameters=PARAMETERS, seed=1, shots=64, workers=2, chunk_size=16))
    evaluation = stage_collect(CollectRequest(
        tmp_path / 'evaluation', role='evaluation', parameters=PARAMETERS, seed=2, shots=64, workers=2, chunk_size=16))
    assert calibration.shots == 64 and evaluation.shots == 64
    assert (tmp_path / 'calibration' / 'sample' / 'sample.json').exists()

    stage_calibrate(tmp_path / 'calibration', tmp_path / 'calibrators.json', ESTIMATORS)
    calibrators, provenance = load_calibrators(tmp_path / 'calibrators.json')
    assert set(calibrators) == {e.name for e in ESTIMATORS}
    assert provenance['sample_hash'] == read_json(tmp_path / 'calibration' / 'manifest.json')['sample_hash']
    with pytest.raises(ValueError, match='calibration'):
        stage_calibrate(tmp_path / 'evaluation', tmp_path / 'wrong.json', ESTIMATORS)

    with pytest.raises(ValueError, match='calibration sample'):
        stage_replay(tmp_path / 'calibration', tmp_path / 'calibrators.json', tmp_path / 'bad', [parse_config(CONFIGS[0])])
    results = stage_replay(tmp_path / 'evaluation', tmp_path / 'calibrators.json', tmp_path / 'replay',
                           [parse_config(text) for text in CONFIGS])
    assert len(results) == 4
    for config in CONFIGS:
        directory = tmp_path / 'replay' / config_directory_name(parse_config(config))
        assert (directory / 'results.json').exists() and (directory / 'final.npy').exists()
        assert np.load(directory / 'final.npy').shape == (64, 12)
    manifest = read_json(tmp_path / 'replay' / 'replay_manifest.json')
    assert manifest['record']['role'] == 'evaluation' and len(manifest['configs']) == 4

    text = stage_summarize([tmp_path / 'replay'], tmp_path / 'summary.md', replicates=200, seed=43)
    assert 'uf:cluster_gap' in text and 'all_refined' in text and 'misattribution' in text.lower()
    assert (tmp_path / 'summary.md').read_text() == text


def test_cli_runs_the_pipeline(tmp_path):
    tool = REPO_ROOT / 'tools' / 'hierarchical_experiment'
    env = dict(PYTHONPATH=str(REPO_ROOT / 'src'), TMPDIR=str(tmp_path), PATH='/usr/bin:/bin')

    def run(*args):
        subprocess.run([sys.executable, str(tool), *args], check=True, env=env, cwd=REPO_ROOT)

    common = ['--distance', '3', '--rounds', '12', '--p', '0.003', '--shots', '32', '--workers', '1', '--chunk-size', '8']
    run('collect', '--out', str(tmp_path / 'cal'), '--role', 'calibration', '--seed', '1', *common)
    run('collect', '--out', str(tmp_path / 'eva'), '--role', 'evaluation', '--seed', '2', '--rows', '0:16', *common)
    run('calibrate', '--record', str(tmp_path / 'cal'), '--out', str(tmp_path / 'cal.json'),
        '--estimators', 'uf:cluster_gap', 'uf:gap_correlated')
    run('replay', '--record', str(tmp_path / 'eva'), '--calibrators', str(tmp_path / 'cal.json'),
        '--out', str(tmp_path / 'rep'), '--config', CONFIGS[0], '--config', CONFIGS[1])
    run('summarize', '--replays', str(tmp_path / 'rep'), '--out', str(tmp_path / 'summary.md'), '--replicates', '100')
    assert (tmp_path / 'summary.md').exists()
    assert read_json(tmp_path / 'eva' / 'manifest.json')['shots'] == 16
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_stages_test.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'yoked.hierarchical._stages'`

- [ ] **Step 3: Implement the stages**

Create `src/yoked/hierarchical/_stages.py`:

```python
"""Driver stages: collect, calibrate, replay, summarize.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, section 11.

Each stage reads only files that earlier stages wrote, verifies their
manifests, and writes its own outputs atomically. The command-line tool
``tools/hierarchical_experiment`` is a thin argparse layer over these
functions, so every behavior here is reachable from tests without it.

Directory layout produced by the stages:

    <collect out>/sample/           packed sample arrays and sample.json
    <collect out>/record.npz        the L1Record
    <collect out>/manifest.json     provenance and record checks
    <calibrators path>              one JSON file, estimators -> per-sector calibrators
    <replay out>/<config>/          results.json, final.npy, refined.npy, ties.npy
    <replay out>/replay_manifest.json

``_stages_test.py`` runs the stages end to end on the distance-3 fixture
and checks the held-out rule: a record cannot be evaluated with calibrators
fit on the same sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from yoked.hierarchical._calibration import IsotonicCalibrator
from yoked.hierarchical._collect import CircuitParameters, CollectionSettings, SampleSet, collect_sample
from yoked.hierarchical._metrics import paired_bootstrap, residual_failure_counts, sector_failures, summarize_result
from yoked.hierarchical._policies import policy_from_name
from yoked.hierarchical._provenance import (
    git_commit, package_versions, read_json, sha256_file, source_hashes, utc_now, write_json_atomic,
)
from yoked.hierarchical._record import L1Record
from yoked.hierarchical._replay import Calibrators, Estimator, ReplayConfig, ReplayResult, fit_calibrators, replay


@dataclass(frozen=True)
class CollectRequest:
    """Inputs of the collect stage: where to write, which rows, and where the sample comes from."""
    out_dir: Path
    role: str
    rows: np.ndarray | None = None
    workers: int = 1
    chunk_size: int = 64
    max_chunks: int | None = None
    recorded_run: Path | None = None
    parameters: CircuitParameters | None = None
    seed: int | None = None
    shots: int | None = None


def stage_collect(request: CollectRequest) -> L1Record | None:
    out_dir = Path(request.out_dir)
    sample_dir = out_dir / 'sample'
    if not (sample_dir / 'sample.json').exists():
        if request.recorded_run is not None:
            sample = SampleSet.load_recorded_run(request.recorded_run)
        elif request.parameters is not None and request.seed is not None and request.shots is not None:
            sample = SampleSet.sample(request.parameters, seed=request.seed, shots=request.shots)
        else:
            raise ValueError('Provide either recorded_run or parameters, seed, and shots')
        sample.save(sample_dir)
    info = read_json(sample_dir / 'sample.json')
    rows = np.arange(info['shots']) if request.rows is None else np.asarray(request.rows)
    settings = CollectionSettings(request.role, rows, request.workers, request.chunk_size, request.max_chunks)
    return collect_sample(sample_dir, out_dir, settings)


def _load_record(record_dir: Path) -> tuple[L1Record, dict]:
    record_dir = Path(record_dir)
    manifest = read_json(record_dir / 'manifest.json')
    if sha256_file(record_dir / 'record.npz') != manifest['record_sha256']:
        raise ValueError(f'record.npz in {record_dir} does not match its manifest')
    return L1Record.load(record_dir / 'record.npz'), manifest


def stage_calibrate(record_dir: Path, out_path: Path, estimators: Sequence[Estimator]) -> Calibrators:
    record, manifest = _load_record(record_dir)
    if manifest['role'] != 'calibration':
        raise ValueError(f'Calibrators must be fit on a calibration record, not {manifest["role"]!r}')
    calibrators = fit_calibrators(record, estimators)
    write_json_atomic(Path(out_path), dict(
        stage='calibrate', sample_hash=manifest['sample_hash'], record_sha256=manifest['record_sha256'],
        record_directory=str(Path(record_dir).resolve()), rows=manifest['rows'], shots=record.shots,
        estimators={name: dict(X=x.to_json(), Z=z.to_json()) for name, (x, z) in calibrators.items()},
        versions=package_versions(), source_sha256=source_hashes(), code_commit=git_commit(), created_utc=utc_now(),
    ))
    return calibrators


def load_calibrators(path: Path) -> tuple[Calibrators, dict]:
    data = read_json(path)
    calibrators = {name: (IsotonicCalibrator.from_json(entry['X']), IsotonicCalibrator.from_json(entry['Z']))
                   for name, entry in data['estimators'].items()}
    return calibrators, data


def parse_config(text: str) -> ReplayConfig:
    """``initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed``."""
    fields = dict(part.split('=', 1) for part in text.split(',') if part)
    for key in ('initial', 'refined', 'policy'):
        if key not in fields:
            raise ValueError(f'config needs {key}=...: {text!r}')
    initial = Estimator.parse(fields['initial'])
    return ReplayConfig(initial, Estimator(initial.reference, fields['refined']),
                        policy_from_name(fields['policy']), fields.get('outer', 'mixed'))


def config_directory_name(config: ReplayConfig) -> str:
    return config.name.replace('->', '_to_').replace(':', '_')


def stage_replay(record_dir: Path, calibrators_path: Path, out_dir: Path,
                 configs: Sequence[ReplayConfig]) -> dict[str, ReplayResult]:
    record, manifest = _load_record(record_dir)
    calibrators, calibration = load_calibrators(calibrators_path)
    if calibration['sample_hash'] == manifest['sample_hash']:
        raise ValueError('Refusing to evaluate on the calibration sample or a subset of it (spec test 9)')
    out_dir = Path(out_dir)
    pieces = manifest['parameters']['patches'] * manifest['parameters']['rounds']
    results = {}
    for config in configs:
        result = replay(record, calibrators, config)
        directory = out_dir / config_directory_name(config)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'final.npy', result.final)
        np.save(directory / 'refined.npy', result.refined)
        np.save(directory / 'ties.npy', result.ties)
        summary = summarize_result(record, result.final, config.initial.reference, pieces=pieces,
                                   ties=result.ties, above_half=result.above_half)
        write_json_atomic(directory / 'results.json', dict(
            config=config.name, initial=config.initial.name, refined=config.refined.name,
            policy=config.policy.name, outer=config.outer, reference=config.initial.reference,
            pieces=pieces, summary=summary))
        results[config.name] = result
    write_json_atomic(out_dir / 'replay_manifest.json', dict(
        stage='replay', record=dict(directory=str(Path(record_dir).resolve()), role=manifest['role'],
                                    sample_hash=manifest['sample_hash'], record_sha256=manifest['record_sha256'],
                                    rows=manifest['rows'], parameters=manifest['parameters']),
        calibrators=dict(path=str(Path(calibrators_path).resolve()), sample_hash=calibration['sample_hash'],
                         record_sha256=calibration['record_sha256']),
        configs=[config.name for config in configs], versions=package_versions(), source_sha256=source_hashes(),
        code_commit=git_commit(), created_utc=utc_now(),
    ))
    return results


def stage_summarize(replay_dirs: Sequence[Path], out_path: Path, *, replicates: int = 10000, seed: int = 43) -> str:
    """Markdown tables per estimator pair, with paired differences against its initial_only cell."""
    lines = ['# Hierarchical replay summary', '']
    for replay_dir in replay_dirs:
        replay_dir = Path(replay_dir)
        manifest = read_json(replay_dir / 'replay_manifest.json')
        record = L1Record.load(Path(manifest['record']['directory']) / 'record.npz')
        cells = {name: _load_cell(replay_dir / config_directory_name(parse_config(_config_text(name))))
                 for name in manifest['configs']}
        lines += [f'## Record: {manifest["record"]["directory"]}',
                  f'Role {manifest["record"]["role"]}, shots {record.shots}, rows {manifest["record"]["rows"]["start"]}'
                  f' to {manifest["record"]["rows"]["stop"]}.', '']
        for pair in sorted({(cell['initial'], cell['refined']) for cell in cells.values()}):
            group = {name: cell for name, cell in cells.items() if (cell['initial'], cell['refined']) == pair}
            baseline = next((cell for cell in group.values() if cell['policy'] == 'initial_only'), None)
            lines += [f'### Initial {pair[0]}, refined {pair[1]}', '',
                      '| Configuration | Block failure | Normalized LER | Misattribution (pooled) '
                      '| Difference vs initial_only [95% CI] | Sectors zero / one / multiple | Ties |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for name, cell in group.items():
                lines.append(_row(record, cell, baseline, replicates, seed))
            lines.append('')
    text = '\n'.join(lines) + '\n'
    Path(out_path).write_text(text)
    return text


def _config_text(name: str) -> str:
    initial, rest = name.split('->')
    refined, policy, outer = rest.split(':')
    return f'initial={initial},refined={refined},policy={policy},outer={outer}'


def _load_cell(directory: Path) -> dict:
    cell = read_json(directory / 'results.json')
    cell['final'] = np.load(directory / 'final.npy')
    return cell


def _row(record: L1Record, cell: dict, baseline: dict | None, replicates: int, seed: int) -> str:
    summary = cell['summary']
    counts = residual_failure_counts(record.reference(cell['reference']), record.actual)
    eligible = (counts == 1).astype(float)
    failures = sector_failures(cell['final'], record.actual).astype(float)
    if baseline is None or baseline is cell:
        difference = 'reference cell' if baseline is cell else 'no initial_only cell'
    else:
        base_failures = sector_failures(baseline['final'], record.actual).astype(float)
        paired = paired_bootstrap((base_failures * eligible).sum(axis=1), (failures * eligible).sum(axis=1),
                                  eligible.sum(axis=1), eligible.sum(axis=1), replicates=replicates, seed=seed)
        difference = f'{paired.difference:+.4f} [{paired.low:+.4f}, {paired.high:+.4f}]'
    block, mis, strata = summary['block_failure'], summary['misattribution']['pooled'], summary['stratum_counts']
    return (f'| {cell["policy"]} ({cell["outer"]}) | {block["count"]}/{block["total"]} = {block["value"]:.4f} '
            f'| {summary["normalized_ler"]:.3e} | {mis["count"]}/{mis["total"]} = {mis["value"]:.4f} '
            f'| {difference} | {strata["zero"]} / {strata["one"]} / {strata["multiple"]} | {summary.get("ties", 0)} |')
```

- [ ] **Step 4: Create the CLI**

Create `tools/hierarchical_experiment` and make it executable (`chmod +x`):

```python
#!/usr/bin/env python3
"""Hierarchical L1/L2 experiment driver: collect | calibrate | replay | summarize.

Run from the repository root. Every subcommand is a thin wrapper over
yoked.hierarchical._stages; see docs/hierarchical_decoding.md for the
workflow and docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md
for the definitions.
"""
import argparse
import pathlib
import sys

import numpy as np

src_path = pathlib.Path(__file__).parent.parent / 'src'
assert src_path.exists()
sys.path.append(str(src_path))

from yoked.hierarchical._collect import ROLES, CircuitParameters  # noqa: E402
from yoked.hierarchical._replay import Estimator  # noqa: E402
from yoked.hierarchical._stages import (  # noqa: E402
    CollectRequest, parse_config, stage_calibrate, stage_collect, stage_replay, stage_summarize,
)


def parse_rows(text):
    """'START:STOP' -> row indices START..STOP-1; None means every row."""
    if text is None:
        return None
    start, _, stop = text.partition(':')
    return np.arange(int(start or 0), int(stop))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    collect = commands.add_parser('collect', help='evaluate L1 on a sample set and write an L1 record')
    collect.add_argument('--out', type=pathlib.Path, required=True)
    collect.add_argument('--role', choices=ROLES, required=True)
    collect.add_argument('--recorded-run', type=pathlib.Path, help='directory of a saved four-decoder run')
    collect.add_argument('--distance', type=int)
    collect.add_argument('--rounds', type=int)
    collect.add_argument('--p', type=float)
    collect.add_argument('--patches', type=int, default=6)
    collect.add_argument('--seed', type=int)
    collect.add_argument('--shots', type=int)
    collect.add_argument('--rows', type=str, help='START:STOP subset of the sample, e.g. 0:2000')
    collect.add_argument('--workers', type=int, default=1)
    collect.add_argument('--chunk-size', type=int, default=64)
    collect.add_argument('--max-chunks', type=int)

    calibrate = commands.add_parser('calibrate', help='fit per-sector calibrators on a calibration record')
    calibrate.add_argument('--record', type=pathlib.Path, required=True)
    calibrate.add_argument('--out', type=pathlib.Path, required=True)
    calibrate.add_argument('--estimators', nargs='+', required=True, help='e.g. uf:cluster_gap uf:gap_correlated')

    replay = commands.add_parser('replay', help='replay configurations on a record with fitted calibrators')
    replay.add_argument('--record', type=pathlib.Path, required=True)
    replay.add_argument('--calibrators', type=pathlib.Path, required=True)
    replay.add_argument('--out', type=pathlib.Path, required=True)
    replay.add_argument('--config', action='append', required=True,
                        help='initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed')

    summarize = commands.add_parser('summarize', help='write markdown tables for replay directories')
    summarize.add_argument('--replays', nargs='+', type=pathlib.Path, required=True)
    summarize.add_argument('--out', type=pathlib.Path, required=True)
    summarize.add_argument('--replicates', type=int, default=10000)
    summarize.add_argument('--seed', type=int, default=43)

    args = parser.parse_args()
    if args.command == 'collect':
        parameters = None
        if args.recorded_run is None:
            parameters = CircuitParameters(distance=args.distance, rounds=args.rounds, p=args.p, patches=args.patches)
        record = stage_collect(CollectRequest(
            out_dir=args.out, role=args.role, rows=parse_rows(args.rows), workers=args.workers,
            chunk_size=args.chunk_size, max_chunks=args.max_chunks, recorded_run=args.recorded_run,
            parameters=parameters, seed=args.seed, shots=args.shots))
        print('collection complete' if record is not None else 'collection paused; rerun to resume')
    elif args.command == 'calibrate':
        stage_calibrate(args.record, args.out, [Estimator.parse(text) for text in args.estimators])
        print(f'calibrators written to {args.out}')
    elif args.command == 'replay':
        results = stage_replay(args.record, args.calibrators, args.out, [parse_config(text) for text in args.config])
        print(f'{len(results)} configurations replayed into {args.out}')
    elif args.command == 'summarize':
        stage_summarize(args.replays, args.out, replicates=args.replicates, seed=args.seed)
        print(f'summary written to {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_stages_test.py -q`
Expected: all pass. The CLI test takes up to a minute because it runs four collections through the forkserver pool.

- [ ] **Step 6: Write the usage doc and the README pointer**

Create `docs/hierarchical_decoding.md`:

```markdown
# Hierarchical L1/L2 decoding

Run from the repository root with `PYTHONPATH=src`. Definitions and the
experiment design live in
[the spec](superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md);
this page shows the code paths.

**Split a yoked DEM into patch graphs.** The two yoke detectors are the last
two detectors. Removing them leaves one component per patch and sector.

```python
import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.hierarchical import PatchGraphs

circuit = yoked_magic_memory_circuit(
    patch_diameter=9, rounds=36, noise=gen.NoiseModel.si1000(0.003),
    style='cz', yokes=2, num_patches=6,
)
dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
patches = PatchGraphs.from_yoked_dem(dem, num_patches=6)
patch = patches[0]
patch.graph            # check-free DecodingGraph, two components (X, Z)
patch.check_graph      # the same edges, observable-flipping boundary edges routed to check vertices
patch.local_dem        # the patch's mechanisms with local ids, for correlation rules
```

**L1: UF reference bits and cluster gaps.** The decoder never sees a yoke bit.

```python
from yoked.hierarchical import ClusterGapUnionFindDecoder

detectors = circuit.compile_detector_sampler(seed=42).sample(shots=1)[0]
result = ClusterGapUnionFindDecoder(patch.graph).decode_with_gaps(patch.local_syndromes(detectors))
result.prediction, result.cluster_gap, result.dijkstra_states
```

**L1: forced matching weights.** `W[c_X, c_Z]` for all four check patterns,
plain and under the correlated reweighting; `signed_gaps` turns them into
gaps relative to any reference bits.

```python
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical import MatchingGaps, signed_gaps

gaps = MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))
forced = gaps.forced_weights(patch.local_syndromes(detectors))
signed_gaps(forced.correlated, reference=result.prediction)   # (delta_X, delta_Z) in nats
```

**L2: exact outer decoder.** Probabilities strictly inside (0, 1), one
sector at a time; `candidates` restricts which patches may flip.

```python
from yoked.hierarchical import exact_outer_map

exact_outer_map([0.9, 0.8, 0.1, 0.1, 0.1, 0.1], parity=0).pattern   # [1, 1, 0, 0, 0, 0]
```

**Stages.** Collect once per sample set, calibrate on the calibration
record only, replay on another record, summarize.

```bash
OUT=$TMPDIR/hier-d9-p003
tools/hierarchical_experiment collect --out $OUT/calibration --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --rows 0:2000 --workers 16 --chunk-size 50
tools/hierarchical_experiment collect --out $OUT/evaluation --role evaluation \
    --recorded-run /data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha --rows 0:2000 --workers 16 --chunk-size 50
tools/hierarchical_experiment calibrate --record $OUT/calibration --out $OUT/calibrators.json \
    --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated
tools/hierarchical_experiment replay --record $OUT/evaluation --calibrators $OUT/calibrators.json --out $OUT/replay \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed
tools/hierarchical_experiment summarize --replays $OUT/replay --out $OUT/summary.md
```

A collection can be stopped with `--max-chunks` or by interrupting it, and
resumes when rerun with the same arguments. Every stage writes a manifest
naming its inputs by SHA-256, and replay refuses a record whose sample is
the one the calibrators were fit on.

**Output layout.**

```text
<collect out>/sample/            detectors_packed.npy, actual_observables_packed.npy, sample.json
<collect out>/record.npz         the L1 record (spec section 5.3)
<collect out>/manifest.json      parameters, rows, hashes, record checks
<calibrators>.json               estimators -> per-sector calibrators, with provenance
<replay out>/<config>/           results.json, final.npy, refined.npy, ties.npy
<replay out>/replay_manifest.json
```
```

Add to `README.md`, after the "Union Find decoder" section:

```markdown
## Hierarchical L1/L2 decoding

The `yoked.hierarchical` package splits the six-patch yoked decoding graph
into per-patch graphs, decodes them with soft outputs, and recombines the
patches with an exact outer decoder over the yoke syndrome. See
[hierarchical decoding](docs/hierarchical_decoding.md) for the code paths
and the driver, and the
[design spec](docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md)
for definitions.
```

- [ ] **Step 7: Run the whole suite and commit**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/decoders src/yoked/hierarchical -q`
Expected: all pass.

```bash
chmod +x tools/hierarchical_experiment
git add src/yoked/hierarchical/_stages.py src/yoked/hierarchical/_stages_test.py tools/hierarchical_experiment docs/hierarchical_decoding.md README.md
git commit -m "Add hierarchical experiment stages, CLI, and usage doc

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

### Task 12: The d=9 pilot and its report

This task runs code, not writes it. It produces the pilot records, the endpoint replays, and `docs/results/hierarchical_pilot_d9_p003.md`, and it records the go/no-go decision of spec section 12, milestone M1.

**Files:**
- Create: `docs/results/hierarchical_pilot_d9_p003.md`
- Run outputs (not committed): `$TMPDIR/hier-d9-p003/...`

**Interfaces:**
- Consumes: the CLI from Task 11 and the saved d=9 run at `/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha`.
- Produces: the pilot report and the decision on whether M2 starts.

- [ ] **Step 1: Preflight 50 rows of the saved evaluation sample and read the record checks**

```bash
cd /data2/s2chitni/yoked-surface-codes
export PYTHONPATH=src
OUT=$TMPDIR/hier-d9-p003
RUN=/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha
time tools/hierarchical_experiment collect --out $OUT/preflight_evaluation --role evaluation \
    --recorded-run $RUN --rows 0:50 --workers 2 --chunk-size 25
python -c "import json; m=json.load(open('$OUT/preflight_evaluation/manifest.json')); print(json.dumps(m['checks'], indent=1)); print('seconds', m['seconds_this_run'])"
```

Expected: `checks.passed` is `true`, `joint_disagreements_unexplained` is 0, `check_parity_agreement` is 1.0. Note the wall time: two workers on 50 rows gives the per-row cost; the 2,000-row collections below run 16 workers, so estimate their duration as `seconds * (2000 / 50) / 8` and the full 100,000-row collection as 50 times that. Write both estimates into the report. If any check fails, stop here and debug the failing invariant on the preflight record before collecting more; the record and manifest are in `$OUT/preflight_evaluation`.

- [ ] **Step 2: Sample the calibration set and collect the two 2,000-row pilot subsets**

The calibration sample is one 50,000-shot call at seed 142; only rows 0 to 1,999 are decoded now. The evaluation pilot is rows 0 to 1,999 of the saved seed-42 sample.

```bash
tools/hierarchical_experiment collect --out $OUT/calibration --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --rows 0:2000 --workers 16 --chunk-size 25
tools/hierarchical_experiment collect --out $OUT/evaluation --role evaluation \
    --recorded-run $RUN --rows 0:2000 --workers 16 --chunk-size 25
for role in calibration evaluation; do
  python -c "import json; m=json.load(open('$OUT/$role/manifest.json')); print('$role', m['checks']['passed'], m['shots'], m['sample_hash'][:12], round(m['seconds_this_run']))"
done
```

Expected: both manifests report `passed` true and 2,000 shots. The evaluation sample hash must equal `d55da8f4c9b8287fa0af64f8382de755a243a774030499e108c348b665e102f5`, the recorded run's payload hash.

- [ ] **Step 3: Calibrate on the calibration pilot only, then replay the endpoints on the evaluation pilot**

```bash
tools/hierarchical_experiment calibrate --record $OUT/calibration --out $OUT/calibrators_pilot.json \
    --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated
tools/hierarchical_experiment replay --record $OUT/evaluation --calibrators $OUT/calibrators_pilot.json \
    --out $OUT/replay_pilot \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=all_refined,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=all_refined,outer=mixed
tools/hierarchical_experiment summarize --replays $OUT/replay_pilot --out $OUT/summary_pilot.md
cat $OUT/summary_pilot.md
```

Expected: three tables, one per estimator pair, each with an `initial_only` reference row and an `all_refined` row carrying a paired difference with a 95% interval. If replay raises about the calibration sample, the two collections point at the same sample directory and Step 2 must be redone with distinct output directories.

- [ ] **Step 4: Compute the saved joint-decoder baselines on the same 2,000 rows**

The full baseline import is part of the next plan; for the pilot, quote the saved predictions directly so the tables have the joint decoders beside them. Save the snippet as `$OUT/baselines_pilot.py` and run it with `python $OUT/baselines_pilot.py`:

```python
import numpy as np
import sinter

RUN = '/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha'
actual = np.unpackbits(np.load(f'{RUN}/actual_observables_packed.npy'), axis=1, bitorder='little')[:2000, :12].astype(bool)
for name in ['mwpm', 'uf', 'correlated_mwpm', 'correlated_uf']:
    predicted = np.load(f'{RUN}/{name}_predictions.npy')[:2000].astype(bool)
    block = (predicted != actual).any(axis=1).mean()
    ler = sinter.shot_error_rate_to_piece_error_rate(block, pieces=216, values=8)
    print(f'{name:16s} block failure {block:.4f}  normalized LER {ler:.3e}')
```

- [ ] **Step 5: Write the pilot report**

Create `docs/results/hierarchical_pilot_d9_p003.md` in the style of `docs/results/decoder_comparison_d9_p003_100k.md`: tables and reproduction details, no interpretation prose beyond the decision. Fill every `<...>` from the outputs above.

```markdown
# Hierarchical L1/L2 pilot at d=9, p=0.003: 2,000-shot endpoints

Exploratory pilot of spec milestone M1. The evaluation rows are the first
2,000 shots of the saved seed-42 sample; the calibration rows are the first
2,000 shots of a fresh 50,000-shot seed-142 sample. Both remain part of
their full sets. Nothing here is a confirmation measurement.

**Configuration.**

| Parameter | Value |
|---|---|
| Circuit | 1D yoked magic-memory circuit, CZ style, d=9, 36 rounds, six patches, two yokes |
| Physical noise | SI1000, p=0.003 |
| Evaluation rows | 0 to 1,999 of seed 42 (payload hash d55da8f4...) |
| Calibration rows | 0 to 1,999 of seed 142 (payload hash <from manifest>) |
| L1 reference | repository UF on check-free patch graphs; patch-local MWPM as control |
| Initial score | cluster gap (UF); plain gap (MWPM control) |
| Refined score | correlated gap; plain gap as a secondary cell |
| L2 | exact factorized MAP, mixed rule |
| Code commit | <git rev-parse HEAD> |

**Record checks (spec section 10).**

| Record | Check parity | Plain additivity max error | Plain argmin disagreements | Correlated sign disagreements | Joint agreement | Unexplained joint disagreements |
|---|---:|---:|---:|---:|---:|---:|
| evaluation | <...> | <...> | <...> | <...> | <...> | <...> |
| calibration | <...> | <...> | <...> | <...> | <...> | <...> |

**Endpoints on the evaluation pilot.** Paste the three tables from
`summary_pilot.md` here unchanged.

**Saved joint decoders on the same 2,000 rows.**

| Decoder | Block failure | Normalized LER |
|---|---:|---:|
| Joint MWPM | <...> | <...> |
| Repository UF | <...> | <...> |
| Correlated MWPM | <...> | <...> |
| Correlated UF | <...> | <...> |

**Collection cost.**

| Run | Rows | Workers | Wall seconds | Rows per worker-second |
|---|---:|---:|---:|---:|
| preflight | 50 | 2 | <...> | <...> |
| calibration pilot | 2,000 | 16 | <...> | <...> |
| evaluation pilot | 2,000 | 16 | <...> | <...> |

Projected full collection at 16 workers: <...> hours for 100,000 evaluation
rows and <...> hours for the remaining 48,000 calibration rows.

**Decision.** <Proceed to M2 | Diagnose | Stop>. Proceed requires the
all-refined minus initial-only paired difference on the pooled
misattribution rate of the UF reference pair to be negative with a 95%
interval excluding zero. Otherwise record the observed difference and
interval and the diagnosis planned.

**Reproduction.** Commands as in Task 12 of
`docs/superpowers/plans/2026-09-14-hierarchical-l1-l2-m1-pilot.md`. Run
directories: `$TMPDIR/hier-d9-p003/{preflight_evaluation,calibration,evaluation,replay_pilot}`;
calibrators `$TMPDIR/hier-d9-p003/calibrators_pilot.json`. Package versions
and source hashes are in each manifest.
```

- [ ] **Step 6: Commit the report**

```bash
git add docs/results/hierarchical_pilot_d9_p003.md
git commit -m "Record the d=9 hierarchical L1/L2 pilot endpoints and decision

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RCFuXzCmozp8C3y72SmFdM"
```

---

## What the second plan covers

Written after the pilot decision, against the same spec: selective policies (`top_k_given_yoke`, `top_k_uncertain`) and exact random controls with `ExpectedReplayResult` (section 8); work accounting for requested patch-sectors, distinct patches, and matching calls; coverage, transitions with the magnitude-versus-reversal split, recovery fraction η(k), queried-population reliability and residual marginals (sections 6 and 9); historical baseline import with hash and row verification; the figure-producing `report` stage; the freeze manifest; full collection of the remaining calibration and evaluation rows with `L1Record.concatenate`; confirmation sampling and replay (milestones M2 to M4); and the d=7 replication.

## Plan self-review

Spec coverage for M1: section 4 is Task 2; 5.1 is Task 3; 5.2 is Task 4 plus Task 1; 5.3 is Tasks 7 to 9; 6 is Task 6 and the per-sector pooling in Task 10; 7 is Task 5; the two endpoint rows of section 8 are Task 10; the primary metric, strata, block failure, bootstrap, and normalized LER of section 9 are Task 10; validation tests 1 to 9 map to Tasks 2, 8, 8, 4 and 8, 4 and 8, 8, 5, 3, and 6 plus 11; milestone M1's pilot is Task 12. Deferred items are listed above.

Type consistency: `signed_gaps` takes patch-major `(..., 2, 2)` and `(..., 2)` inputs everywhere it is called (Tasks 4, 8, 10). `by_sector` and `to_columns` are the only layout converters. `ARRAY_FIELDS` in `_record.py` is the public field tuple used by Task 9. `Estimator.direction` is always `'decreasing'`, matching `IsotonicCalibrator.fit`'s `direction` argument. `policy_from_name` raises for the selective names until the second plan adds them.
