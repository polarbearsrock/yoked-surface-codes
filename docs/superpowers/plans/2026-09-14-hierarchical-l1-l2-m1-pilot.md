# Hierarchical L1/L2 Decoding, M1 and Pilot: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the patch-local L1 layer (UF reference plus cluster gap, forced matching weights plain and correlated), the exact L2 outer decoder, isotonic calibration, resumable collection with validation, endpoint replay with core metrics, and run the 2,000-shot d=9 pilot that decides whether the full experiment proceeds.

**Architecture:** A new package `src/yoked/hierarchical/` with one module per concept, tests beside each module. The six-patch DEM is split into per-patch graphs by connected components with the yoke hub replaced by per-patch check vertices. L1 decoders run on those graphs and never see a yoke bit. Everything L1 produces is stored once in an `L1Record`; calibration and replay are offline stages over that record. A thin CLI in `tools/hierarchical_experiment` drives the stages.

**Tech Stack:** Python 3.14 in `.venv`, NumPy, SciPy sparse, Stim 1.16, PyMatching 2.4 (edge-list construction through `from_check_matrix`), sinter for LER normalization, pytest. Run everything from the repository root with `PYTHONPATH=src`.

**Spec:** `docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md`. This plan implements spec sections 4, 5, 6, 7, the `initial_only` and `all_refined` rows of section 8, the primary metric, strata, block failure and bootstrap of section 9, validation tests 1 to 9 of section 10, and milestone M1 of section 12. M1 also includes endpoint work accounting, paired block-failure intervals, d=9 graph equivalence and imported-edge statistics, deterministic endpoint replay, and the applicable collection/provenance checks from tests 10, 11, 13, and 15. Selective policies, exact random controls, their diagnostics, and milestones M2 to M4 belong to a second plan written after the pilot decision. Confirmation is not an accepted dataset role in this plan.

## Global Constraints

- Notation follows spec section 2 exactly: patch `i`, sector `s` (0 = X, 1 = Z), observable `o(i, s) = 2i + s`, record column `2i + s`, internal sector-major layout `(shots, 2, 6)`.
- Yoke detectors are the last two detectors, X then Z. Check vertices in a patch's check graph are local ids `n` and `n + 1` where `n` is the patch's detector count.
- All weights, gaps, and scores are in nats. Nothing converts to decibels.
- L1 code never reads a sampled yoke bit. Forced decodes use synthetic check bits only.
- Calibrators are fit on the calibration sample only; the driver refuses to calibrate and evaluate on the same sample or subset (spec test 9).
- Every module starts with a docstring stating the object it computes in spec notation and naming the invariants its tests check. Records crossing module boundaries are frozen dataclasses with a docstring per field including units and shapes. Array fields own read-only storage; mappings are copied and made immutable. Validate values before casting. Mutable collection buffers stay inside the collector. Named constants carry a comment saying why they hold their value. Seeds are explicit parameters.
- Keep interfaces narrow: one UF method returns the correction and settled edge costs; hierarchical code never accesses `_Growth`, `_peel`, or growth-engine storage. Re-export only intended public entry points, not every helper.
- Collection identity covers the saved sample, model, rows, role, and decoder implementation. Calibration and replay have separate identities. Changing a policy or a report must not invalidate reusable L1 outputs.
- Use small functions and concrete records. The stage contracts and failure tests below are requirements; no generic workflow engine, plugin registry, configurable serialization framework, or new service is needed.
- Temporary and run outputs go under `$TMPDIR` (`/data2/s2chitni/.tmp`), never under the home directory or `/tmp`.
- Tests run with `PYTHONPATH=src .venv/bin/pytest <path> -q`. The existing suite in `src/yoked/decoders` must keep passing after every task.
- Commit after each completed task using a description of the actual change. Attribution must reflect the people and tools that did the work; do not copy historical session metadata.

## File Structure

| File | Responsibility |
|---|---|
| `src/yoked/decoders/_correlations.py` | Correlation rules from a DEM, indexing by source, applying rules to weights. Moved out of the correlated UF module so matching and UF share one implementation. |
| `src/yoked/decoders/_correlated_union_find.py` | Modified to import from `_correlations`. Behavior unchanged. |
| `src/yoked/decoders/_union_find.py` | Shared growth-and-peeling path plus immutable `GrowthDecodeResult`; existing predictions unchanged. |
| `src/yoked/hierarchical/__init__.py` | Public entry points only. |
| `src/yoked/hierarchical/_arrays.py` | Small internal helper for owned, read-only arrays. |
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
| `src/yoked/hierarchical/_stages.py` | Verified stage boundaries and publication; no decoder algorithms. |
| `tools/hierarchical_experiment` | CLI: `collect`, `calibrate`, `replay`, `summarize`. |
| `docs/hierarchical_decoding.md` | Usage, mirroring `docs/union_find_usage.md`. |
| `docs/results/hierarchical_pilot_d9_p003.md` | Pilot report and go/no-go record. |

Deviations from the spec's file list: `_fixtures.py` is test support; `_arrays.py` is a small internal ownership helper; `_provenance.py` and `_record.py` separate artifact and record I/O from replay; `_collect.py` owns collection and validation; `_stages.py` owns the stage boundaries. The existing UF module gains one immutable-result method. The `report` subcommand of the spec is delivered as `summarize` here, producing the pilot tables; the full figure-producing `report` belongs to the second plan.

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
git commit -m "Move correlation rules into a shared decoders module"
```

---

### Task 2: Patch graphs: splitting the hub

**Files:**
- Create: `src/yoked/hierarchical/__init__.py`
- Create: `src/yoked/hierarchical/_patch_graphs.py`
- Create: `src/yoked/hierarchical/_patch_graphs_test.py`
- Create: `src/yoked/hierarchical/_fixtures.py`
- Create: `src/yoked/hierarchical/_arrays.py` and `_arrays_test.py`

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

Create `src/yoked/hierarchical/_arrays.py`:

```python
"""Owned array storage for immutable mathematical results.

Callers validate shapes and values before converting to the documented dtype.
A frozen dataclass alone does not prevent changes through a NumPy alias.
"""
import numpy as np


def readonly_array(value, *, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order='C', copy=True)
    result.setflags(write=False)
    return result
```

Add `_arrays_test.py`: changing the input after construction must not change
the result, assignment through the result raises, and the requested dtype and
shape are preserved. Use this helper for result arrays throughout the package;
keep it internal. Add both files to this task's checks and commit.

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
git add src/yoked/hierarchical/_arrays.py src/yoked/hierarchical/_arrays_test.py src/yoked/hierarchical/__init__.py src/yoked/hierarchical/_patch_graphs.py src/yoked/hierarchical/_patch_graphs_test.py src/yoked/hierarchical/_fixtures.py
git commit -m "Add per-patch graph split for hierarchical decoding"
```

---

### Task 3: UF reference with the cluster gap

**Files:**
- Modify: `src/yoked/decoders/_union_find.py` and its tests (one shared decode path and immutable growth-cost result)
- Create: `src/yoked/hierarchical/_cluster_gap.py`
- Create: `src/yoked/hierarchical/_cluster_gap_test.py`
- Modify: `src/yoked/hierarchical/__init__.py` (export `ClusterGapUnionFindDecoder`, `ClusterGapResult`)

**Interfaces:**
- Consumes: `DecodingGraph` and `UnionFindDecoder.decode_with_growth_costs(syndrome) -> GrowthDecodeResult`. Growth-engine state stays in the existing decoder module.
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
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
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
    costs = UnionFindDecoder(graph).decode_with_growth_costs(syndrome).remaining_costs
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
        costs = UnionFindDecoder(graph).decode_with_growth_costs(syndrome).remaining_costs
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

First add the following result to `src/yoked/decoders/_union_find.py`:

```python
@dataclass(frozen=True)
class GrowthDecodeResult:
    """UF correction and the settled costs needed by a soft-output consumer.

    selected_edges: correction edge ids in graph order.
    observable_mask: XOR of selected edge masks, an integer bit mask.
    remaining_costs: one cost per graph edge in nats, zero within a cluster.
    """
    selected_edges: tuple[int, ...]
    observable_mask: int
    remaining_costs: tuple[float, ...]
```

Factor the current `_decode` body into `_decode_state`, returning its existing
`_Correction` and the per-call `_Growth` inside this module. `_decode` returns
only the correction, preserving the existing public API and avoiding an edge
scan for plain UF. Add this method to `UnionFindDecoder`:

```python
def decode_with_growth_costs(self, syndrome: np.ndarray) -> GrowthDecodeResult:
    correction, growth = self._decode_state(syndrome)
    costs = []
    for edge_id, (u, v) in enumerate(self.graph.endpoints):
        growth._settle(edge_id)
        cost = (0.0 if growth.find(u) == growth.find(v)
                else max(0.0, self.graph.edges[edge_id][2] - growth.grown[edge_id]))
        costs.append(cost)
    return GrowthDecodeResult(correction.selected_edges, correction.observable_mask, tuple(costs))
```

The two entry points share validation, growth, and peeling exactly once. Keep
`_Growth` and `_decode_state` private; no mutable state escapes. Existing UF,
correlated UF, and adapter tests must pass. Add tests that both entry points
choose identical corrections, returned tuples survive another decode unchanged,
and malformed syndromes fail identically. The partial-growth fixture in Step 1
checks the new method's cost semantics against hand-computed values.

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
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._arrays import readonly_array


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

    def __post_init__(self) -> None:
        for name, dtype in (('prediction', bool), ('cluster_gap', np.float64), ('dijkstra_states', np.int64)):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=dtype))


class ClusterGapUnionFindDecoder:
    """Repository UF plus the cluster gap of every observable of the graph."""

    def __init__(self, graph: DecodingGraph):
        self.graph = graph
        self._uf = UnionFindDecoder(graph)
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
        decoded = self._uf.decode_with_growth_costs(syndrome)
        selected, mask = decoded.selected_edges, decoded.observable_mask
        costs = decoded.remaining_costs
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

```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._cluster_gap import ClusterGapResult, ClusterGapUnionFindDecoder
```

and extend `__all__` with `'ClusterGapResult', 'ClusterGapUnionFindDecoder'`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_cluster_gap_test.py -q`
Expected: all pass, including existing decoder regressions. If the partial-growth cost differs from the hand-computed 0.5, investigate settling and batch timing inside the UF module. Preserve the asserted cost convention; do not adjust the fixture merely to make an unexpected result pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/decoders/_union_find.py src/yoked/decoders/_union_find_test.py src/yoked/hierarchical/_cluster_gap.py src/yoked/hierarchical/_cluster_gap_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add UF reference decoding with the cluster-gap soft output"
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
from yoked.hierarchical._arrays import readonly_array

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

    def __post_init__(self) -> None:
        for name, dtype in (('plain', np.float64), ('correlated', np.float64),
                            ('first_pass', bool), ('correlated_prediction', bool)):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=dtype))


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
        syndrome = np.asarray(local_syndrome)
        if syndrome.shape != (self.patch.num_detectors,):
            raise ValueError(f'Expected {self.patch.num_detectors} detector bits, got shape {syndrome.shape}')
        if syndrome.dtype.kind not in 'buif' or not np.isin(syndrome, (0, 1)).all():
            raise ValueError('Expected binary detector bits')
        syndrome = syndrome.astype(np.uint8, copy=False)
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
from yoked.hierarchical._matching_gaps import ForcedWeights, MatchingGaps, signed_gaps
```

and extend `__all__` with `'ForcedWeights', 'MatchingGaps', 'signed_gaps'`. Keep `CHECK_PATTERNS` internal.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_matching_gaps_test.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_matching_gaps.py src/yoked/hierarchical/_matching_gaps_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add forced matching weights and signed gaps for patch graphs"
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
  - `exact_outer_map_batch(q (shots, n), parity (shots,), candidates (shots, n) | None) -> BatchOuterDecision`, with read-only `patterns (shots, n) bool` and `tied (shots,) bool` fields.
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
    """Threshold/parity minima, with conditional minima resolving absolute-tolerance ties.

    No patterns are enumerated. Fixing some bits leaves the same analytic
    problem: take the preferred free bits and, if needed, the cheapest toggle.
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    candidates = np.ones(n, dtype=bool) if candidates is None else np.asarray(candidates, dtype=bool)
    b = (q > 0.5) & candidates
    cost = np.abs(np.log1p(-q) - np.log(q))

    def minimum_with(fixed):
        if np.any((fixed == 1) & ~candidates):
            return np.inf
        assigned = fixed >= 0
        preferred = b.copy()
        preferred[assigned] = fixed[assigned].astype(bool)
        result = cost[assigned & (preferred != b)].sum()
        free = ~assigned & candidates
        if preferred.sum() % 2 != parity:
            result += cost[free].min(initial=np.inf)
        return result

    fixed = np.full(n, -1, dtype=np.int8)
    best = minimum_with(fixed)
    if not np.isfinite(best):
        raise ValueError('no candidate')
    limit = best + TIE_TOLERANCE
    # Highest bit first: prefer zero whenever an admissible completion exists.
    for bit in reversed(range(n)):
        fixed[bit] = 0
        if minimum_with(fixed) > limit:
            fixed[bit] = 1
    pattern = fixed.astype(bool)
    # Another admissible pattern must differ at at least one candidate bit.
    tied = False
    for bit in np.flatnonzero(candidates):
        alternative = np.full(n, -1, dtype=np.int8)
        alternative[bit] = 1 - fixed[bit]
        tied |= minimum_with(alternative) <= limit
    return pattern, bool(tied)


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


def test_one_half_bit_does_not_create_a_parity_preserving_tie():
    for parity in (0, 1):
        result = exact_outer_map([0.5], parity)
        assert result.pattern.tolist() == [bool(parity)] and not result.tied
        expected, tied = _analytic_rule([0.5], parity)
        np.testing.assert_array_equal(result.pattern, expected)
        assert result.tied == tied


def test_tie_tolerance_is_absolute_and_allows_multiple_small_toggles():
    q = 1 / (1 + np.exp(np.array([1.0, 1.0 + 5e-7])))
    result = exact_outer_map(q, 1)
    assert result.pattern.tolist() == [True, False] and not result.tied
    expected, tied = _analytic_rule(q, 1)
    np.testing.assert_array_equal(result.pattern, expected)
    assert result.tied == tied
    q = np.full(3, 1 / (1 + np.exp(-0.2 * TIE_TOLERANCE)))
    result = exact_outer_map(q, 0)
    assert result.pattern.tolist() == [False, False, False] and result.tied
    expected, tied = _analytic_rule(q, 0)
    np.testing.assert_array_equal(result.pattern, expected)
    assert result.tied == tied


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
    batch = exact_outer_map_batch(q, parity, candidates)
    patterns, tied = batch.patterns, batch.tied
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

from yoked.hierarchical._arrays import readonly_array

TIE_TOLERANCE = 1e-9
"""Log-weight differences below this are ties: far below any calibrated probability's resolution."""

MAX_PATCHES = 16
"""Enumeration builds 2**n patterns per shot; 16 keeps a batch to 65,536 columns."""


@dataclass(frozen=True)
class OuterDecision:
    """L2's answer for one sector: ``pattern`` (n,) bool and whether it was ``tied``."""
    pattern: np.ndarray
    tied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, 'pattern', readonly_array(self.pattern, dtype=bool))


@dataclass(frozen=True)
class BatchOuterDecision:
    """Read-only L2 outputs: patterns (shots, n) bool and tied (shots,) bool."""
    patterns: np.ndarray
    tied: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'patterns', readonly_array(self.patterns, dtype=bool))
        object.__setattr__(self, 'tied', readonly_array(self.tied, dtype=bool))


def frame_adjusted_syndrome(yoke: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """sigma[s] = y[s] XOR parity of r over patches, for yoke (..., 2) and reference (..., 2 * patches)."""
    yoke, reference = np.asarray(yoke), np.asarray(reference)
    if (yoke.ndim < 1 or reference.ndim < 1 or yoke.shape[-1] != 2
            or reference.shape[-1] % 2 or yoke.shape[:-1] != reference.shape[:-1]):
        raise ValueError('Expected matching leading shapes, two yoke bits, and two bits per patch')
    for values in (yoke, reference):
        if values.dtype.kind not in 'buif' or not np.isin(values, (0, 1)).all():
            raise ValueError('Expected binary values')
    yoke, reference = yoke.astype(bool), reference.astype(bool)
    by_patch = reference.reshape(reference.shape[:-1] + (-1, 2))   # (..., patches, sectors)
    return yoke ^ (by_patch.sum(axis=-2) % 2).astype(bool)


def exact_outer_map(q: np.ndarray, parity: int, candidates: np.ndarray | None = None) -> OuterDecision:
    """Maximum-weight residual pattern of the required parity for one sector."""
    batch_candidates = None if candidates is None else np.asarray(candidates)[None]
    result = exact_outer_map_batch(np.asarray(q, dtype=np.float64)[None], np.asarray([parity]), batch_candidates)
    return OuterDecision(result.patterns[0], bool(result.tied[0]))


def exact_outer_map_batch(
        q: np.ndarray, parity: np.ndarray, candidates: np.ndarray | None = None,
) -> BatchOuterDecision:
    """Vectorized ``exact_outer_map``: q (shots, n), parity (shots,), candidates (shots, n) or None."""
    q = np.asarray(q, dtype=np.float64)
    parity = np.asarray(parity)
    if parity.dtype.kind not in 'buif' or not np.isin(parity, (0, 1)).all():
        raise ValueError('Parity must contain binary values')
    parity = parity.astype(bool, copy=False)
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
        candidates = np.asarray(candidates)
        if candidates.shape != q.shape:
            raise ValueError('candidates must have the same shape as q')
        if candidates.dtype.kind not in 'buif' or not np.isin(candidates, (0, 1)).all():
            raise ValueError('Candidates must contain binary values')
        candidates = candidates.astype(bool, copy=False)
        feasible &= ~(patterns[None, :, :] & ~candidates[:, None, :]).any(axis=2)
    if not feasible.any(axis=1).all():
        raise ValueError('Odd parity with no candidate patch: no feasible pattern')
    log_weight = np.where(feasible, log_weight, -np.inf)
    best = log_weight.max(axis=1, keepdims=True)
    near = log_weight >= best - TIE_TOLERANCE
    chosen = near.argmax(axis=1)   # the first tie in pattern order has the lowest binary value
    return BatchOuterDecision(patterns[chosen], near.sum(axis=1) > 1)


@functools.lru_cache(maxsize=None)
def _patterns(n: int) -> np.ndarray:
    """All 2**n patterns as a (2**n, n) bool array whose row p has bit i equal to (p >> i) & 1."""
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
```

Add to `src/yoked/hierarchical/__init__.py`:

```python
from yoked.hierarchical._outer_decoder import (
    TIE_TOLERANCE, OuterDecision, BatchOuterDecision, exact_outer_map, exact_outer_map_batch, frame_adjusted_syndrome,
)
```

and extend `__all__` accordingly.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest src/yoked/hierarchical/_outer_decoder_test.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yoked/hierarchical/_outer_decoder.py src/yoked/hierarchical/_outer_decoder_test.py src/yoked/hierarchical/__init__.py
git commit -m "Add exact L2 outer decoder for the factorized patch model"
```

---

### Task 6: Isotonic calibration

**Knot convention:** pool repeated input scores before PAV. Merge adjacent
blocks only for a strict monotonicity violation. Equal-valued adjacent blocks
remain separate interpolation knots; do not coalesce them after fitting.
Each knot is its block's count-weighted mean score and fitted error rate.
Interpolate linearly through those knots, with constant extrapolation and
clipping. This fixes one deterministic interpretation of the spec's block
centres; merging equal plateaus would define a different estimator.

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
    np.testing.assert_allclose(calibrator.centers, [1, 2, 3, 4])
    np.testing.assert_allclose(calibrator.probabilities, [1 - CLIP, 1 - CLIP, CLIP, CLIP])
    np.testing.assert_allclose(calibrator.probability([1.5, 2, 3, 3.5]), [1 - CLIP, 1 - CLIP, CLIP, CLIP])
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
fit. PAV merges strict violations only; equal-valued adjacent blocks stay as
separate knots. Between block centres the map is linear; beyond the fitted range it is
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

from yoked.hierarchical._arrays import readonly_array

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

    def __post_init__(self) -> None:
        centers = readonly_array(self.centers, dtype=np.float64)
        probabilities = readonly_array(self.probabilities, dtype=np.float64)
        if self.direction not in DIRECTIONS:
            raise ValueError('Invalid direction')
        if centers.ndim != 1 or not len(centers) or probabilities.shape != centers.shape:
            raise ValueError('Expected nonempty, equally sized knot arrays')
        if not np.isfinite(centers).all() or not (np.diff(centers) > 0).all():
            raise ValueError('Knot centers must be finite and strictly increasing')
        if not ((probabilities >= CLIP) & (probabilities <= 1 - CLIP)).all():
            raise ValueError('Probabilities must be finite and inside the clipping range')
        differences = np.diff(probabilities)
        violates_order = (differences < 0).any() if self.direction == 'increasing' else (differences > 0).any()
        if violates_order:
            raise ValueError('Probabilities must be monotone')
        if (isinstance(self.num_samples, (bool, np.bool_))
                or not isinstance(self.num_samples, (int, np.integer)) or self.num_samples < 1):
            raise ValueError('num_samples must be a positive integer')
        object.__setattr__(self, 'centers', centers)
        object.__setattr__(self, 'probabilities', probabilities)

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
                   np.asarray(data['probabilities'], dtype=np.float64), data['num_samples'])


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
git commit -m "Add isotonic calibration for soft-output scores"
```

---

### Task 7: Immutable records, artifact identities, and small I/O helpers

**Files:** create `_record.py`, `_record_test.py`, `_provenance.py`, and
`_provenance_test.py` under `src/yoked/hierarchical/`.

**Deliverable:** implement the contracts below as concrete records and small
functions. Tasks 7 to 11 specify interfaces, invariants, and failure tests
instead of prescribing complete module bodies. Every requirement is exercised
by a test; avoid duplicate validation logic in the CLI.

- [ ] **Step 1: Define ownership and the record schema.**

`L1Record` is a frozen dataclass containing the arrays in spec section 5.3,
plus `rows (shots,) int64` and `reweighted_patches (shots, P) bool` recording
`ForcedWeights.rules_fired` for actual collection-work accounting. Use column
`2i + s`, forced-weight layout `(shots, P, 2, 2)`, and units of nats. Optional
historical baselines are a `Mapping[str, np.ndarray]`, copied into an immutable
mapping; each value is an owned, read-only `(shots, 2P)` boolean array.

All array fields use `_arrays.readonly_array` **after** shape/value checks.
Reject nonbinary bits, fractional/negative row indices, duplicate rows,
negative state counts, malformed shapes, and invalid numeric values before
casting. A complete pilot record requires finite nonnegative gaps and weights.
Do not allow `astype(bool)` or `astype(int64)` to conceal invalid input.

Expose `shots`, `num_patches`, `reference(name)`, `subset(positions)`,
`by_sector(columns)`, and `to_columns(sectors)`. Record construction and subset
creation preserve ownership. Attach baselines at construction or with
`dataclasses.replace`, never by mutating an existing record. Add a small
`LoadedRecord` frozen dataclass carrying the record and its verified identity.
Low-level array serialization is private; downstream stages use only
`load_record(record_dir) -> LoadedRecord`, the verified loader from Task 9.

Keep raw-buffer allocation private to collection. A worker returns an immutable
chunk result; because immutable mapping wrappers need not be pickleable, send
plain array payloads across a process queue and reconstruct the checked result
at the receiving boundary. Do not add a generic serialization registry.

Full-set concatenation is used in M2. When provided, it must consume verified
`LoadedRecord` values, require equal parent sample/model/decoder/role identities
and identical baseline columns, reject overlapping rows, and preserve sorted
parent row ids. It must not silently take the intersection of baseline names.

- [ ] **Step 2: Define identities independently of filesystem locations.**

Use canonical JSON and SHA-256, with explicit schema/layout versions. Keep the
historical payload hash convention: packed detector bytes followed by packed
observable bytes, little bit order; this differs from hashing `.npy` files.
Record both payload and file hashes with unambiguous names. Provide
`packed_sample_hash(detectors_packed, actual_packed)`, `sha256_bytes`,
`sha256_file`, and `write_json_atomic` for these concrete operations. Stream
packed payload bytes in detector-then-observable order, without `.npy` headers.

| Identity | Inputs that determine it |
|---|---|
| Model | Circuit parameters, exact saved `circuit.stim` and `model.dem` hashes, detector/observable counts, and bit/layout conventions |
| Parent sample | Model identity, sampling seed, full sampling-call shot count, sampling versions, and complete packed payload hash; row subsets retain this identity |
| Sampling family | Circuit hash, seed, and sampling versions; prevents fitting/evaluating on separate same-seed calls with potentially overlapping streams |
| Decoder | Graph import, UF, correlation compiler, splitter, cluster-gap, matching-gap, array/record conventions, and their relevant package versions |
| Collection | Parent sample, decoder, dataset role, and exact ordered row ids |
| Calibration | Verified calibration record and rows, model/decoder identities, estimator definitions, knot convention, calibration sources and versions |
| Replay | Verified record and calibrator artifacts, configurations, tie rule, work-count convention, replay/metric sources and versions |

Implement explicit source groups, not one global `SOURCE_FILES` list. Policy,
plotting, reporting, and calibration changes do not change the decoder identity.
Changes to an L1 algorithm or its dependencies do. Validation sources, including
the outer decoder, have a separate recorded check identity: changed validation
can recheck an existing record without rerunning L1. Record `_stages.py`, the
CLI, circuit-generation sources, package versions, and the git commit for
auditability; the saved circuit and DEM bytes determine the model actually used.
Required source files must exist; never silently omit missing files from a hash.

Calibration compatibility requires equal model and decoder identities and
identical score/reference conventions. It does not require equal sample ids.
Held-out checks reject equal parent sample ids **or** equal sampling families,
regardless of path or dataset-role labels. A subset cannot acquire a new parent
identity merely by being saved to another directory.

- [ ] **Step 3: Implement narrowly scoped atomic I/O helpers.**

Use temporary sibling files beneath the run directory in `$TMPDIR`, close the
file, and replace its destination atomically. Use `allow_pickle=False` on
loads. Provide helpers for canonical JSON, file/payload hashing, JSON writes,
and array-container writes. A stage publishes its completion manifest last.
It never overwrites a completed stage with different inputs: require a new
output directory. Missing completion manifests mean incomplete work, even when
some final-looking arrays exist.

A completed collection manifest includes `schema_version`, `status='complete'`, `role`,
`parameters`, `seed`, `parent_shots`, `shots`, the row summary (count, start,
stop, and hash of the exact ids stored in the record), `parent_payload_sha256`,
`identities` (model, parent_sample, sampling_family, decoder, collection),
`artifacts` (relative filename to hash), check identity/results, versions,
source hashes, `collection_work`, and timing (`seconds_this_run` plus cumulative
collection time across resumptions). `checks` has `passed`, `graph`, and `record`
entries; the last two serialize the concrete check records. A complete sample
manifest identifies all four input files and uses the same named model/parent
identities and payload-hash convention. Loaders verify the fields relevant
to their stage,
including `checks.passed is True`, before exposing data. JSON serialization
represents undefined statistics as `null` with their counts, not nonstandard
`NaN` literals.

- [ ] **Step 4: Verify ownership, identity, and persistence.**

Test input-alias mutation, attempted writes through every record field and
baseline, save/load/subset ownership, and all invalid-value cases above. Test
that a policy/report change leaves collection identity unchanged, a decoder
change changes it, a changed DEM with unchanged parameters changes model
identity, and rows/roles alter collection identity. Verify the historical hash
convention against a hand-packed example. Test missing required source files,
corrupt files, failed checks, missing completion markers, and schema mismatch.
Run the new tests and the existing decoder suite; commit the completed task.

---

### Task 8: Verified samples, single-process collection, and graph/record checks

**Files:** create `_collect.py` and `_collect_test.py`; use the Task 7 helpers.

**Interfaces:**

- `CircuitParameters(distance, rounds, p, patches=6, yokes=2, style='cz', noise='si1000')`.
- `SampleSet.sample(parameters, *, seed, shots)` creates one full packed sample
  plus its exact circuit and DEM; `save(directory)` publishes its sample manifest.
- `SampleSet.load(directory)` verifies and opens the saved sample;
  `load_recorded_run(directory)` imports the existing benchmark format.
- `L1Context.from_dem_text(text, num_patches)` owns reusable per-process decoders.
- `collect_rows(context, detectors, actual, rows)` returns a checked `L1Record`.
- `check_graphs(dem, patches) -> GraphChecks` and
  `check_record(record) -> RecordChecks`, both frozen result records.

- [ ] **Step 1: Preserve the actual sampling model.**

A sample directory contains `circuit.stim`, `model.dem`,
`detectors_packed.npy`, `actual_observables_packed.npy`, and `sample.json`.
Generate the circuit once and derive the DEM from it with the exact options
in the design. Sample once with explicit seed/full shot count; unpack only
requested chunks during collection. Store sampling versions, hashes, dimensions,
parameters, parent sample id, and sampling-family id. Retain read-only packed
memory maps; do not unpack an entire 50,000- or 100,000-shot sample just to use
its first 2,000 rows.

Import the saved benchmark's **original** circuit and DEM. Verify their hashes
against `manifest.json['input_sha256']` along with the saved packed payloads,
array shapes, counts, and conventions before copying/referencing them. Never
regenerate the decoding model from today's circuit generator for imported
shots. Check the declared circuit/model dimensions and parameters for consistency.

Verify actual saved bytes on first use and on resume, once per parent process,
not once per chunk. Workers read the verified saved DEM and read-only sample
arrays. A changed dependency version or decoder source cannot silently join an
existing partial collection.

- [ ] **Step 2: Make graph validation a collection prerequisite.**

Build the imported joint `DecodingGraph` and the patch graphs. Remap patch check
vertices to the two global yokes, detector ids to global ids, and observable
masks to global masks. Compare edge multiplicities grouped by endpoint/mask;
within each group sort weights and compare with absolute tolerance `1e-9`,
`rtol=0`. Rounding weights into dictionary keys is not a tolerance comparison.
Reject missing/extra edges, changed labels, or out-of-tolerance weights.

Run this check on every distinct sample/model, including the saved d=9 model,
before launching workers. Record imported-edge counts and adjacency degrees.
For the recorded d=9 fixture verify two yoke degrees of 1,110, median detector
degree 11, and maximum non-yoke degree 12. Derive statistics for other distances;
never substitute raw DEM target multiplicities for graph degrees.

Test equivalent graphs and deliberately changed edge multiplicity, mask, and
weight. Include a near-rounding-boundary weight difference smaller than `1e-9`
that passes, and a larger difference that fails.

- [ ] **Step 3: Collect each row through one implementation path.**

For each patch, run the shared UF result method through the cluster-gap decoder,
validate its selected correction against `H c = s` and `L c = r`, and collect
plain/correlated forced weights and the two unforced validation predictions.
Persist `rules_fired` in `reweighted_patches`. Joint plain MWPM runs on the full
saved DEM for validation. L1 never reads sampled yoke bits; only record assembly
and validation use them.

Use the same `collect_rows` function for serial and parallel collection.
A `workers=1` run calls it directly, with no process pool, so failures are easy
to reproduce. Validate binary values and row ids before narrowing dtypes.

Record actual collection work separately from replay work. For the Task 4
implementation, per row there are `P` UF decodes, `2P` Dijkstra searches, `P`
unforced plain calls, `4P` plain forced calls, and `P` reweight attempts. If `R`
patches have `rules_fired=True`, there are `4R` correlated forced calls and `R`
unforced correlated validation calls; unchanged weights reuse the plain results.
There is also one joint MWPM validation decode per row. A matching count means
one decoded syndrome: a four-row `decode_batch` contributes four forced decodes.
Include settled Dijkstra states and setup work. Retained-row counts can be
reconstructed from the record; retries and repeated setup are recorded separately.
If interruption loses attempted-work telemetry, label that telemetry incomplete
instead of claiming that retained-row counts cover every attempted decode. Test these counts with instrumented decoder calls so an
implementation change cannot silently make the formulas stale.

- [ ] **Step 4: Enforce every applicable record invariant.**

`RecordChecks.passed` requires check parity on every row; valid correction
checks on every patch; finite/nonnegative collected values; plain and correlated
additivity and preferred-class consistency within `1e-9`; and zero unexplained
joint-MWPM disagreements. Do not encode an agreement-percentage acceptance floor.

For the joint comparison, take signed plain gaps relative to the MWPM reference,
apply the uncalibrated logistic, compute the frame-adjusted syndrome, and use
`BatchOuterDecision.patterns`. Convert patch-major gaps to sector-major once;
avoid round trips through column layout. Both the reconstructed final prediction
and joint MWPM must obey yoke parity, including on disagreements. Compare total
forced costs on every disagreement within `1e-6` nats. Record tie differences.

A failing graph, correction, or record invariant raises and prevents publication.
Keep diagnostics with row/patch identifiers; diagnosing a failed collection
never requires accepting it as completed data.

- [ ] **Step 5: Run the distance-3 integration checks.**

Use a few hundred shared shots to exercise both correlation branches, graph
checks, valid corrections, matching additivity, and joint-optimum equivalence.
Deliberately corrupt a yoke bit, reference, forced cost, final parity, and graph
edge; verify each appropriate gate fails. Compare single-call and partitioned
serial collection by every array and parent row id. Test a fresh sample round
trip and the imported benchmark format with small fixtures. Run the decoder
suite and commit.

---

### Task 9: Atomic checkpointing, resume, and validated completion

**Files:** extend `_collect.py` and `_collect_test.py`; put shared verified record
loading in `_record.py` using `_provenance.py` helpers.

**Interfaces:**

- `CollectionSettings(role, rows, workers=1, chunk_size=64, max_chunks=None)`;
  roles are only `('calibration', 'evaluation')` in M1.
- `collect_sample(sample_dir, out_dir, settings) -> LoadedRecord | None`.
- `load_record(record_dir) -> LoadedRecord`, the only public completed-record loader.

- [ ] **Step 1: Validate requests and establish collection identity.**

Require a nonempty one-dimensional integer row array with unique indices in
`[0, parent_shots)`. Reject duplicates, negative values, floats, and booleans;
normalize valid rows into increasing order and hash those exact ids. Validate
worker/chunk counts and an explicitly supplied `max_chunks` as positive integers.

Verify the parent sample and saved model, then compare collection identity with
`collection.json`. Changing rows, role, model, parent, or decoder identity
requires a separate collection directory. Worker/chunk settings may change on
resume; they do not affect the decoded result. M2 can collect remaining rows
separately and concatenate verified disjoint records with the pilot.

- [ ] **Step 2: Use one checkpoint file for data and progress.**

`checkpoint.npz` contains all in-progress array buffers, exact row ids,
`completed (shots,) bool`, schema, and collection identity encoded without
pickle. Write one temporary sibling file and atomically replace the checkpoint.
There is no separate `completed.npy`. Validate the checkpoint identity, keys,
dtypes/shapes, row ids, and completion mask before resuming.

Only the coordinator changes buffers. A row becomes completed after all its
arrays and collection-work metadata are installed. Dispatch only unfinished
rows; key results by their original positions, reject unexpected/duplicate
positions, and make final output independent of worker completion order.
`max_chunks` stops scheduling after that many chunks and writes a checkpoint.
An interrupted decode may lose work since the last successful checkpoint but
cannot report an unfinished row as completed. Document a named flush interval.

- [ ] **Step 3: Publish only validated complete data.**

Follow this order:

```text
verify input bytes + model + request + existing collection identity
if completion manifest exists:
    require complete status, passing checks, correct identities and artifact hashes
    load the validated record (or revalidate if only check-code identity changed)
else:
    load a valid checkpoint or allocate private buffers
    compute pending chunks and atomically checkpoint
    if rows remain: return None
    construct immutable record and run all graph/correction/record checks
    if any check fails: retain checkpoint + diagnostics, raise; publish no completion marker
    atomically write record.npz and its artifact hash
    atomically publish manifest.json with status='complete' and checks.passed=true
    remove the checkpoint only after successful publication
```

The manifest is the commit point. An orphan `record.npz` never means completion;
resume from the checkpoint, revalidate, and replace the orphan as necessary.
If a completion manifest exists but a hash/check/identity fails, raise instead
of silently treating the damaged result as either valid or a new run. A change
to validation code can explicitly revalidate verified stored arrays and publish
a new check identity without redoing L1. A decoder mismatch cannot take this path.

- [ ] **Step 4: Add failure-injection and resume tests.**

Compare uninterrupted serial, partitioned serial, parallel, and resumed runs on
the same distance-3 sample, including retained-row work counts and row ids. Inject interruption
before checkpoint replacement, after replacement, after final record replacement,
and before completion-manifest publication. Each restart either uses the previous
complete checkpoint or the new complete checkpoint and produces the same result.
Test failed `RecordChecks`, missing/corrupt manifests, corrupt records/checkpoints,
changed sample bytes despite unchanged JSON, dependency/model changes, and altered
rows/roles. Verify downstream loading rejects failed and incomplete collections.
Run the new tests and decoder regressions; commit.

---

### Task 10: Endpoint replay, paired metrics, and work accounting

**Files:** create `_policies.py`, `_replay.py`, `_metrics.py`, and adjacent tests.

**Interfaces:**

- `Estimator(reference, score)`: references `uf`/`mwpm`; scores `cluster_gap`,
  `gap_plain`, `gap_correlated`; cluster gap requires UF. All maps decrease.
- `NoRefinement` and `RefineAll`, implementing
  `select(q0 (shots, 2, P), sigma (shots, 2)) -> request_mask`.
- `ReplayConfig(initial, refined, policy, outer='mixed')`, requiring one reference.
- `fit_calibrators(record, estimators)` and
  `calibrated_probabilities(record, estimator, calibrators)`.
- `replay(record, calibrators, config) -> ReplayResult`, with immutable final bits,
  request mask `M`, distinct-patch mask `U`, ties, probabilities-above-half flags,
  and a named per-shot `WorkCounts` record.
- `Rate`, `PairedDifference`, rate/stratum functions, `paired_bootstrap`, and
  `summarize_result`. Arrays and mappings crossing boundaries follow Task 7.

- [ ] **Step 1: Implement the fixed-reference replay.**

Fit one calibrator per estimator/sector on calibration rows, pooling patches.
Scores are cluster gaps or signed forced-weight differences relative to the
chosen reference; outcomes are `actual XOR reference`. Build `q0`, `q1`, and
`sigma = yoke XOR parity(reference)`. Policies see only `q0` and `sigma`.
Set `q = where(M, q1, q0)`; `U = M.any(axis=1)`. Use exact L2 per sector and
return `final = reference XOR pattern`. Validate final yoke parity on every row.

Implement mixed and candidate-restricted L2 for the two endpoints. Restricted
odd parity with no requested candidate raises explicitly. Selective policy names
remain unsupported until M3; do not add stubs that silently act like endpoints.
Calibrators and records must remain unchanged by replay; use a bounded batch
size for enumeration. Do not rerun a decoder downstream of collection.

- [ ] **Step 2: Implement endpoint work counts now.**

`WorkCounts` contains per-shot integer counts with explicit units: requested
patch-sectors, distinct patches refined, initial UF calls, initial Dijkstra
searches and settled states, initial unforced/forced plain matching calls, and
incremental unforced plain calls, plain forced calls, reweight passes, and
correlated forced calls. Every returned array is read-only. Derive the counts
from `M`, `U`, the estimator pair, and stored Dijkstra states.

| Estimator pair | Initial work per patch | Increment per distinct requested patch |
|---|---|---|
| UF cluster gap -> plain gap | 1 UF, 2 Dijkstra searches | 4 plain forced calls |
| UF cluster gap -> correlated gap | 1 UF, 2 Dijkstra searches | 1 unforced plain call, 1 reweight pass, 4 correlated forced calls |
| MWPM plain gap -> correlated gap | 1 unforced plain call, 4 plain forced calls | 1 reweight pass, 4 correlated forced calls |

For six patches, `initial_only` has `sum(M)=sum(U)=0`; `all_refined` has
`sum(M)=12`, `sum(U)=6`. Both sector requests share one patch refinement.
These are counts for the specified replay procedure, separately labeled from
actual collection calls, validation calls, graph setup, and wall time. Offline
reuse of already collected forced weights does not make replay work zero.
Report initial and incremental totals separately; do not claim speedups.

- [ ] **Step 3: Implement accuracy and paired intervals.**

Compute sector/block failures; X/Z/pooled single-failure misattribution;
zero/one/multiple-reference-failure strata with denominators; tie counts;
above-half frequencies; and the existing sinter normalized LER conversion
(`pieces=patches*rounds`, `values=8` for this six-patch experiment).

Bootstrap paired whole shots with 10,000 replicates and explicit seed 43.
Pooled misattribution resamples each shot's eligible-sector count and its
misattributed eligible-sector count together. Block failure uses per-shot
failure indicators and denominator one. Report per-endpoint rate intervals
and paired `all_refined - initial_only` differences/95% intervals for **both**
primary misattribution and block failure. Retain cross-sector dependence.
Report zero-denominator replicate counts and undefined rates as unavailable;
do not substitute zero or require a fixed agreement fraction.

The pilot does not estimate recovery fraction eta, selective coverage, or
random-policy expectations; those belong to M3.

- [ ] **Step 4: Test behavior with hand-worked examples.**

Verify signed score indexing, patch pooling, endpoint final bits, mixed versus
restricted behavior, `q > 1/2` reversals, deterministic ties, and unchanged
inputs. Test each estimator pair's counts at no/all refinement, and use a small
request-mask fixture to show X and Z on one patch cost one refinement while
requests on two patches cost two. Verify unrequested `q1` values are ignored.
Check actual collection counts separately against Task 8's instrumentation.

Use explicit per-shot numerator/denominator fixtures for both bootstrap metrics,
including empty strata and cross-sector dependence. Check reproducibility and
a hand-computed paired point estimate. Run replay twice and compare all results.
Run the package and decoder suites; commit.

---

### Task 11: Verified stages, thin CLI, and usage documentation

**Files:** create `_stages.py`, `_stages_test.py`, `tools/hierarchical_experiment`,
and `docs/hierarchical_decoding.md`; add a short README pointer.

**Interfaces:**

- `CollectRequest(out_dir, role, rows=None, workers=1, chunk_size=64,
  max_chunks=None, recorded_run=None, parameters=None, seed=None, shots=None)`.
- `stage_collect(request) -> LoadedRecord | None`.
- `stage_calibrate(record_dir, out_path, estimators)` and verified calibrator loading.
- `parse_config(text) -> ReplayConfig` and a deterministic directory-name helper.
- `stage_replay(record_dir, calibrators_path, out_dir, configs)`.
- `stage_summarize(replay_dirs, out_path, *, replicates, seed)`.

- [ ] **Step 1: Enforce collection requests even when outputs already exist.**

A request supplies either a recorded run or the complete parameters/seed/full
shot-count triple, not both. Validate it before reading existing outputs. If a
sample already exists, compare the supplied request with its verified identity:
parameters, seed, full shot count, and exact circuit/DEM/payload for an imported
run. Compare a generated request's model to the saved model; a changed generator
must not silently choose a new model for old shots. Paths alone are not identities.
A mismatch raises a diagnostic naming the changed field. Never ignore new
arguments just because `sample.json` exists. Then call Task 9's collector.

- [ ] **Step 2: Enforce calibration and replay compatibility.**

Calibration uses only a completed, passing calibration record. Store the exact
input record and manifest hashes, parent/family ids, fitted row ids, model and
decoder identities, estimator names/directions, knot convention, clip constant,
and calibration sources/versions with the fitted knots.

Replay uses a completed, passing evaluation record and validated calibrators.
Reject calibration-role evaluation, equal parent sample or sampling-family ids,
changed/missing estimators, incompatible model or decoder identities, unsupported
schemas, nonmonotone/invalid knots, and incorrect declared conventions. Verify
artifacts before decoding. Confirmation is unavailable until freeze verification
is implemented in the later plan.

For each configuration, write final predictions, request/distinct-patch masks,
ties, work arrays, and results JSON. Hash these actual output files and the exact
calibrator artifact in `replay_manifest.json`; an input record hash alone does
not identify the fitted calibration. Publish the completed replay manifest last,
after parity and result checks pass. Reusing an output directory requires the
same verified inputs/configurations and valid output hashes; otherwise require
a new directory. Interrupted publication can be resumed without accepting
partially replaced artifacts as completed results.

- [ ] **Step 3: Summarize only verified artifacts.**

The summary loader verifies the replay completion manifest, every consumed
array/JSON hash, and the referenced record/manifest identity and exact row order.
It must reject a record replaced after replay or altered prediction files. Pair
configurations only on identical sample/row ids and reference strata. Persist
bootstrap inputs/settings and results with the summary's provenance.

Produce endpoint tables with both rate intervals and both paired differences,
stratum denominators, ties/above-half frequencies, and a separate work table
with initial/incremental counts. Undefined intervals are displayed explicitly
with eligible counts. Write the report and its completion manifest atomically.

- [ ] **Step 4: Add the CLI and user documentation.**

Keep argparse parsing in the executable and all behavior in library functions.
Subcommands are `collect`, `calibrate`, `replay`, and `summarize`. Preserve the
configuration syntax used in Task 12. Accept only calibration/evaluation roles.
Validate `START:STOP` row ranges and mutually exclusive sample sources.
`summarize` records `--replicates 10000 --seed 43` explicitly in the pilot commands.

Document the fixed-reference model, read-only records, verified saved model,
serial/debug path, full-call sampling versus subset collection, checkpoint and
completion semantics, and separate actual/replay work counts. Show the existing
decoder examples with named results, including `BatchOuterDecision.patterns`.
Document these outputs under `$TMPDIR`:

```text
<collect out>/sample/       circuit.stim, model.dem, packed arrays, sample.json
<collect out>/collection.json
<collect out>/checkpoint.npz        only while incomplete
<collect out>/record.npz
<collect out>/manifest.json         completion marker, hashes, graph/record checks
<calibrators>.json                  knots, verified parents and compatibility
<replay out>/<config>/              prediction/mask/tie/work arrays, results.json
<replay out>/replay_manifest.json    completion marker and all artifact hashes
<summary>.md and <summary>.manifest.json
```

- [ ] **Step 5: Exercise the stage boundaries end to end.**

Run a small distance-3 calibration/evaluation pipeline both through functions
and through the CLI. Verify valid resume and reject changed seed, shot count,
parameters, recorded-run identity, raw data, decoder version, or model; reusing
the same output path cannot suppress those checks. Test calibration reuse on
its own parent/subset and same sampling family, incompatible calibration from
another distance/noise setting, failed collection checks, invalid knots, altered
calibrator/prediction files, mismatched rows, and a missing completion manifest.
Inject interrupted replay publication and verify clean recovery. Reject
confirmation requests. Verify that reporting/policy changes preserve collection
reuse. Avoid duplicating each mathematical unit test through every CLI command.

Run `PYTHONPATH=src .venv/bin/pytest src/yoked/decoders src/yoked/hierarchical -q`.
Commit the stages, CLI, and documentation after the relevant checks pass.

---

### Task 12: The d=9 pilot and its report

This task runs the implemented pipeline after its checks pass. It produces
`docs/results/hierarchical_pilot_d9_p003.md` and the M1 decision. All run data
stays under `$TMPDIR`; confirmation is neither sampled nor collected.

- [ ] **Step 1: Preflight 50 saved evaluation rows.**

```bash
cd /data2/s2chitni/yoked-surface-codes
export PYTHONPATH=src
export OUT="$TMPDIR/hier-d9-p003"
export RUN=/data2/s2chitni/.tmp/ysc-four-decoders-d9-p003-100k-7efr23ha
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/preflight_evaluation" --role evaluation \
    --recorded-run "$RUN" --rows 0:50 --workers 2 --chunk-size 25
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
manifest = json.loads((Path(os.environ['OUT']) / 'preflight_evaluation/manifest.json').read_text())
print(json.dumps(manifest['checks'], indent=2))
print(json.dumps(manifest['collection_work'], indent=2))
print('seconds', manifest['seconds_this_run'])
assert manifest['status'] == 'complete' and manifest['checks']['passed']
PY
```

Require d=9 graph equivalence and the imported-edge statistics, perfect check
and final parity, valid corrections, and zero unexplained joint disagreements.
If any invariant fails, collection raises, leaves its checkpoint/diagnostics,
and publishes no completed manifest. Diagnose the failing row/graph before
collecting more. Do not turn a failing invariant into a percentage threshold.

Use the measured preflight time for a preliminary full-run estimate, explicitly
labeling any assumed worker scaling. Update the estimate using the actual
16-worker pilot throughput; initialization and I/O need not scale linearly.

- [ ] **Step 2: Collect the two 2,000-row pilot subsets.**

The calibration sample is one 50,000-shot call at seed 142; only rows 0 to 1,999
are decoded. Evaluation uses those rows from the saved 100,000-shot seed-42
sample and its saved circuit/DEM. Both pilots retain their full parent ids.

```bash
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/calibration" --role calibration \
    --distance 9 --rounds 36 --p 0.003 --seed 142 --shots 50000 --rows 0:2000 --workers 16 --chunk-size 25
.venv/bin/python tools/hierarchical_experiment collect --out "$OUT/evaluation" --role evaluation \
    --recorded-run "$RUN" --rows 0:2000 --workers 16 --chunk-size 25
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['OUT'])
for role in ('calibration', 'evaluation'):
    m = json.loads((root / role / 'manifest.json').read_text())
    assert m['status'] == 'complete' and m['checks']['passed'] and m['shots'] == 2000
    print(role, m['parent_payload_sha256'], m['seconds_this_run'])
    if role == 'evaluation':
        assert m['parent_payload_sha256'] == 'd55da8f4c9b8287fa0af64f8382de755a243a774030499e108c348b665e102f5'
PY
```

A resume with a changed seed, parameter, parent, or decoder must raise. Use a
new directory for a different experiment; do not relabel an existing sample.

- [ ] **Step 3: Calibrate and replay all pilot endpoint pairs.**

```bash
.venv/bin/python tools/hierarchical_experiment calibrate --record "$OUT/calibration" --out "$OUT/calibrators_pilot.json" \
    --estimators uf:cluster_gap uf:gap_plain uf:gap_correlated mwpm:gap_plain mwpm:gap_correlated
.venv/bin/python tools/hierarchical_experiment replay --record "$OUT/evaluation" --calibrators "$OUT/calibrators_pilot.json" \
    --out "$OUT/replay_pilot" \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_correlated,policy=all_refined,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=initial_only,outer=mixed \
    --config initial=uf:cluster_gap,refined=gap_plain,policy=all_refined,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=initial_only,outer=mixed \
    --config initial=mwpm:gap_plain,refined=gap_correlated,policy=all_refined,outer=mixed
.venv/bin/python tools/hierarchical_experiment summarize --replays "$OUT/replay_pilot" \
    --out "$OUT/summary_pilot.md" --replicates 10000 --seed 43
```

Require three endpoint-pair tables with rate intervals and paired differences
for both pooled misattribution and block failure, strata/denominators, ties,
above-half frequencies, and the work table. The primary comparison is UF cluster
gap to correlated gap; UF to plain gap is secondary and MWPM is the control.

- [ ] **Step 4: Verify historical baseline inputs before quoting them.**

Full baseline integration into records remains in M2. For M1, save this small
analysis script as `$OUT/baselines_pilot.py` and run it using `.venv/bin/python`.
It uses the same parent rows and verifies the historical prediction **packed
payload** hashes from `results.json`, not just the current `.npy` file hashes.

```python
import json
import os
from pathlib import Path

import numpy as np
import sinter

from yoked.hierarchical._provenance import (
    packed_sample_hash, sha256_bytes, sha256_file, write_json_atomic,
)
from yoked.hierarchical._record import load_record

root, run = Path(os.environ['OUT']), Path(os.environ['RUN'])
record = load_record(root / 'evaluation').record
collection = json.loads((root / 'evaluation/manifest.json').read_text())
history = json.loads((run / 'manifest.json').read_text())
results = json.loads((run / 'results.json').read_text())
for name in ('circuit.stim', 'model.dem'):
    assert sha256_file(run / name) == history['input_sha256'][name]
    assert sha256_file(root / 'evaluation/sample' / name) == history['input_sha256'][name]
detectors = np.load(run / 'detectors_packed.npy', mmap_mode='r', allow_pickle=False)
actual_packed = np.load(run / 'actual_observables_packed.npy', mmap_mode='r', allow_pickle=False)
payload_hash = packed_sample_hash(detectors, actual_packed)
assert payload_hash == history['input_sha256']['packed_detectors_then_observables_payload']
assert payload_hash == collection['parent_payload_sha256']
actual = np.unpackbits(actual_packed[record.rows], axis=1, bitorder='little')[:, :12].astype(bool)
np.testing.assert_array_equal(actual, record.actual)
audit = dict(parent_payload_sha256=payload_hash, rows=record.rows.tolist(),
             source_sha256=history['source_sha256'], versions=history['versions'],
             code_commit=history['code_commit'], results_sha256=sha256_file(run / 'results.json'), decoders={})
for name in ('mwpm', 'uf', 'correlated_mwpm', 'correlated_uf'):
    path = run / f'{name}_predictions.npy'
    predicted = np.load(path, mmap_mode='r', allow_pickle=False)
    assert predicted.shape == (history['parameters']['shots'], 12)
    assert np.isin(predicted, (0, 1)).all()
    prediction_hash = sha256_bytes(np.packbits(predicted, axis=1, bitorder='little').tobytes())
    assert prediction_hash == results['decoders'][name]['prediction_packed_sha256']
    failures = (predicted[record.rows] != actual).any(axis=1)
    block = float(failures.mean())
    audit['decoders'][name] = dict(prediction_packed_sha256=prediction_hash,
        file_sha256=sha256_file(path), errors=int(failures.sum()), shots=record.shots,
        block_failure=block,
        normalized_ler=float(sinter.shot_error_rate_to_piece_error_rate(block, pieces=216, values=8)))
write_json_atomic(root / 'baselines_pilot.json', audit)
print(json.dumps(audit['decoders'], indent=2))
```

- [ ] **Step 5: Write the pilot report from verified outputs.**

Use the style of `docs/results/decoder_comparison_d9_p003_100k.md`. Include:

1. **Configuration/provenance:** d=9, 36 rounds, six patches, two ideal yokes,
   CZ, SI1000 p=0.003; parent samples/seeds and exact row ids; model/decoder ids;
   original circuit/DEM hashes; calibration artifact/knot convention; code and
   package versions. Label every result exploratory, not confirmation.
2. **Graph and record checks:** original/rebuilt edge counts, maximum weight
   discrepancy, graph equivalence, yoke and non-yoke degree statistics;
   correction/check/final parity checks; plain and correlated additivity/sign
   checks; joint agreement, tie differences, and unexplained disagreements.
3. **Endpoint tables:** X/Z/pooled primary rates and block rates with 95% intervals;
   paired all-refined minus initial-only differences/intervals for both pooled
   misattribution and block failure; normalized LER, stratum denominators,
   ties/above-half frequencies, and any undefined bootstrap results.
4. **Replay work:** requested patch-sectors and distinct patches; initial UF,
   Dijkstra/search-state and plain-matching counts; incremental plain matching,
   reweighting and correlated matching counts. Show per-shot means and totals.
5. **Actual collection work:** measured calls, validation/setup work, wall time,
   rows/workers, and measured throughput. Identify retries or incomplete
   attempted-work telemetry after interruptions. Keep this separate from the replay
   procedure counts. No latency or speedup claim follows from these counts.
6. **Historical baselines:** block failure and normalized LER on the same rows,
   citing the verified `baselines_pilot.json` and prediction hashes.
7. **Projected full collection cost and decision:** Proceed to M2, Diagnose, or
   Stop. The plan's proceed criterion is a negative paired primary UF-reference
   misattribution difference with its 95% interval excluding zero; also report
   the block-failure effect and work. Otherwise record the effect, interval,
   eligible counts, and planned diagnosis. The pilot remains exploratory.
8. **Reproduction:** exact commands, all artifact paths and manifests, bootstrap
   replicate count/seed, and any resumptions. Use the actual pilot throughput
   for full evaluation and remaining-calibration time estimates.

Do not hand-edit rates after copying them from verified summaries. Keep the
machine-readable summaries and baseline audit with the report's provenance.

- [ ] **Step 6: Commit the report and decision.**

Only the report is committed; samples, checkpoints, predictions, and intermediate
outputs remain under `$TMPDIR`. A failed or inconclusive pilot still produces
an honest report and diagnosis rather than starting full collection automatically.

---

## What the second plan covers

M2 covers the remaining calibration/evaluation rows and verified concatenation,
full-set refitting/endpoints, and reusable historical-baseline import. M3 adds
selective policies, exact random controls, both outer rules for those policies,
coverage, transitions, recovery fraction eta, queried-population reliability,
residual marginals, figures, and the analysis freeze. It extends the existing
work accounting rather than introducing it for the first time. M4 adds
confirmation-role support only together with freeze verification for sampling,
collection, replay, and reporting. The d=7 replication follows separately.

## Plan consistency and acceptance checklist

- [ ] Tasks 1 to 6 provide shared correlation rules, a narrow UF result method,
  graph splitting, soft outputs, an independent parity-aware L2 oracle, and an
  explicit PAV knot convention. Result arrays are owned/read-only.
- [ ] Tasks 7 to 9 provide immutable records, saved model/sample verification,
  stage-specific identities, atomic checkpoints, and validated completion.
  Tests exercise failure and interruption paths as well as successful resume.
- [ ] Tasks 10 to 11 provide deterministic endpoint replay, initial/incremental
  work accounting, rate intervals and paired intervals for primary and block
  failure, verified calibration compatibility, and verified report inputs.
- [ ] Task 12 repeats graph/correction invariants on d=9, checks imported-edge
  statistics, reports endpoints and work, and records an exploratory M1 decision.
- [ ] Selective/random-policy checks and confirmation/freeze work are explicitly
  deferred. M1 refuses confirmation requests.
- [ ] Existing decoder regressions and all new applicable checks pass before the
  pilot. No test is weakened to fit a result; investigate invariant failures.
