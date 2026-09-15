"""L1 evaluation of one chunk of parent rows into record arrays and work counts.

Spec: docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md, sections 5.1, 5.2, 5.3.

The object this module computes is ``collect_rows(context, detectors, actual, rows) ->
CollectedRows``: for every parent row of the chunk and every patch ``i``, the
cluster-gap UF reference bits and gaps, the settled-state counts, the plain and
correlated forced matching weights and their unforced predictions, and the joint
PyMatching validation decode, laid into the ``L1Record`` arrays together with the
``CollectionWork`` counters describing the decoder calls that actually ran.

Three objects support it. ``L1Context`` owns the per-process decoders: the patch
split, one ``ClusterGapUnionFindDecoder`` and one ``MatchingGaps`` per patch, and the
joint matcher used only for the validation decode. ``CollectionWork`` is the frozen
counter record, ``WORK_FIELDS`` its field names in declaration order, and
``CollectedRows`` pairs the filled record with the work it cost.

This is the whole of the per-row decoder path, which is why ``_provenance.py`` lists
this file in ``DECODER_SOURCES``: every number a stored record holds is produced here,
so a change to it changes what the record means and must not be merged into a
collection decoded by an earlier version. Sampling, the graph and record gates, and
the checkpointed coordinator live in ``_collect.py`` and are identified separately.

Invariants ``_l1_test.py`` checks:

  * Every UF correction reproduces its own syndrome, ``H c = s``, and flips exactly
    the observables its reference bits name, ``L c = r``; a correction that does not
    raises naming the offending parent row and patch rather than being stored.
  * Column ``2i + s`` of every ``(shots, 2P)`` array holds sector ``s`` of patch
    ``i``: each sector block is pinned to an independent recomputation from the patch
    graph, so a transposition of sectors or patches fails.
  * The work formulas: one UF decode, one unforced plain matching and one reweight
    attempt per row and patch; ``NUM_SECTORS`` Dijkstra searches and their settled
    states per UF decode; ``FORCED_CALLS_PER_DECODE`` plain forced matchings per row
    and patch; the correlated forced and validation matchings only where a rule
    fired; one joint decode per row. Chunk work adds, so a partitioned collection
    reports the same totals as one call.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pymatching
import stim

from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._matching_gaps import CHECK_PATTERNS, MatchingGaps
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph, PatchGraphs
from yoked.hierarchical._record import L1Record

FORCED_CALLS_PER_DECODE = len(CHECK_PATTERNS)
"""Forced decodes per patch and variant. A matching count means one decoded syndrome,
so the four-row ``decode_batch`` behind ``forced_weights`` contributes four."""


# --- shared value checks -----------------------------------------------------

def _whole(value, name: str, *, minimum: int) -> int:
    """A Python int from a whole, in-range value; ``True`` is a flag, not a count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}, got {value}')
    return int(value)


def _binary(value, name: str, *, columns: int) -> np.ndarray:
    """An (rows, columns) boolean array; 0.5 is rejected rather than cast to True."""
    array = np.asarray(value)
    if array.dtype.kind not in 'buif':
        raise ValueError(f'{name} must be numeric, got dtype {array.dtype}')
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f'{name} must have shape (shots, {columns}), got {array.shape}')
    if array.dtype.kind != 'b' and not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain only 0 and 1')
    return array.astype(bool, copy=False)


def _parent_rows(value, shots: int) -> np.ndarray:
    """The parent sample row of each collected shot: whole, nonnegative, and increasing."""
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise ValueError(f'rows must be an integer array of parent row ids, got dtype {array.dtype}')
    if array.ndim != 1 or len(array) != shots:
        raise ValueError(f'rows must have shape ({shots},), got {array.shape}')
    if not np.isfinite(array).all() or (array.dtype.kind == 'f' and not (array == np.floor(array)).all()):
        raise ValueError('rows must be whole numbers')
    if (array < 0).any():
        raise ValueError('rows must be nonnegative')
    if shots > 1 and not (np.diff(array) > 0).all():
        raise ValueError('rows must be unique and increasing')
    return array.astype(np.int64)


# --- the per-process decoders ------------------------------------------------

class L1Context:
    """The decoders one process reuses across every chunk it collects.

    Attributes: ``dem`` the joint model; ``patches`` its ``PatchGraphs`` split;
    ``decoders`` one ``ClusterGapUnionFindDecoder`` per patch; ``matchers`` one
    ``MatchingGaps`` per patch; ``joint`` the joint PyMatching matcher, used only for
    the validation decode that ``check_record`` compares against.

    Building one costs a model parse, the split, the correlation-rule compilation, and
    the joint matcher, which is why a worker builds it once per process rather than once
    per chunk. It deliberately does not run ``check_graphs``: that is a prerequisite the
    coordinator runs once per distinct model before any worker starts.
    """

    def __init__(self, dem: stim.DetectorErrorModel, num_patches: int):
        self.dem = dem
        self.patches = PatchGraphs.from_yoked_dem(dem, num_patches=num_patches)
        self.decoders = tuple(ClusterGapUnionFindDecoder(patch.graph) for patch in self.patches)
        self.matchers = tuple(
            MatchingGaps(patch, correlation_rules_from_dem(patch.graph, patch.local_dem))
            for patch in self.patches)
        self.joint = pymatching.Matching.from_detector_error_model(dem)

    @classmethod
    def from_dem_text(cls, text: str, num_patches: int) -> L1Context:
        """Build a context from the sample's saved model text."""
        return cls(stim.DetectorErrorModel(text), num_patches)

    @property
    def num_patches(self) -> int:
        return len(self.patches)


# --- the work one chunk cost -------------------------------------------------

@dataclass(frozen=True)
class CollectionWork:
    """Decoder work that actually ran while collecting a set of rows.

    Every field is a whole nonnegative count over the rows collected. Setup (building an
    ``L1Context``) and retries are recorded by the collection stage, not here.

    - ``rows``: parent rows collected.
    - ``uf_decodes``: syndromes decoded by the cluster-gap UF decoder (calls).
    - ``dijkstra_searches``: shortest-odd-walk searches run (calls), two per UF decode.
    - ``dijkstra_states``: states settled by those searches (states), the soft-output
      cost proxy.
    - ``unforced_plain_calls``: unforced first-pass matchings on the check-free graph (calls).
    - ``plain_forced_calls``: forced matchings under the plain weights (calls); a
      four-row ``decode_batch`` counts as four decoded syndromes.
    - ``reweight_attempts``: attempts to apply the correlation rules (calls), one per
      patch and row whether or not a rule fired.
    - ``correlated_forced_calls``: forced matchings under reweighted weights (calls),
      run only for the patches where a rule fired.
    - ``correlated_validation_calls``: unforced second-pass matchings under reweighted
      weights (calls), same condition.
    - ``joint_decodes``: joint PyMatching validation decodes on the full model (calls).
    """
    rows: int
    uf_decodes: int
    dijkstra_searches: int
    dijkstra_states: int
    unforced_plain_calls: int
    plain_forced_calls: int
    reweight_attempts: int
    correlated_forced_calls: int
    correlated_validation_calls: int
    joint_decodes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name, _whole(getattr(self, field.name), field.name, minimum=0))

    def __add__(self, other) -> CollectionWork:
        """Totals of two chunks, so a parallel collection sums what its workers reported."""
        if not isinstance(other, CollectionWork):
            return NotImplemented
        return CollectionWork(**{name: getattr(self, name) + getattr(other, name) for name in WORK_FIELDS})

    def to_json(self) -> dict:
        return {name: getattr(self, name) for name in WORK_FIELDS}


WORK_FIELDS = tuple(field.name for field in dataclasses.fields(CollectionWork))
"""The work counters in declaration order."""


@dataclass(frozen=True)
class CollectedRows:
    """What one ``collect_rows`` call produced: the checked ``record`` and its ``work``."""
    record: L1Record
    work: CollectionWork

    def __post_init__(self) -> None:
        if not isinstance(self.record, L1Record):
            raise TypeError(f'record must be an L1Record, got {type(self.record).__name__}')
        if not isinstance(self.work, CollectionWork):
            raise TypeError(f'work must be a CollectionWork, got {type(self.work).__name__}')


# --- one chunk of rows -------------------------------------------------------

def _validate_correction(patch: PatchGraph, result, syndrome: np.ndarray, *, row: int, index: int) -> None:
    """Check ``H c = s`` and ``L c = r`` for one patch's UF correction.

    The graph's endpoints give every boundary edge its own terminal beyond the real
    detectors, so flipping both endpoints of each selected edge and then reading back
    only the detector entries is exactly ``H c``.
    """
    graph = patch.graph
    parity = np.zeros(len(graph.adjacency), dtype=bool)
    mask = 0
    for edge_id in result.selected_edges:
        u, v = graph.endpoints[edge_id]
        parity[u] ^= True
        parity[v] ^= True
        mask ^= graph.edges[edge_id][3]
    if not np.array_equal(parity[:graph.num_detectors], syndrome):
        raise ValueError(f'row {row}, patch {index}: the UF correction does not reproduce its syndrome')
    predicted = np.array([(mask >> k) & 1 for k in range(graph.num_observables)], dtype=bool)
    if not np.array_equal(predicted, result.prediction):
        raise ValueError(f'row {row}, patch {index}: the UF correction flips {predicted.tolist()}, '
                         f'its reference bits are {result.prediction.tolist()}')


def collect_rows(context: L1Context, detectors, actual, rows) -> CollectedRows:
    """Run L1 on a chunk of parent rows and return its record and the work that ran.

    ``detectors`` (k, n_d) and ``actual`` (k, 2P) are the unpacked bits of the parent
    rows named by ``rows``, which must be whole, nonnegative, and increasing. The patch
    decoders never see the yoke bits: they enter the record as the ``yoke`` column and
    are read again only by ``check_record``.
    """
    patches = context.patches
    detectors = _binary(detectors, 'detectors', columns=patches.num_detectors)
    actual = _binary(actual, 'actual', columns=patches.num_observables)
    if len(actual) != len(detectors):
        raise ValueError(f'detectors holds {len(detectors)} rows and actual {len(actual)}')
    rows = _parent_rows(rows, len(detectors))
    shots, num_patches = len(rows), len(patches)
    columns = NUM_SECTORS * num_patches

    uf_reference = np.zeros((shots, columns), dtype=bool)
    mwpm_reference = np.zeros((shots, columns), dtype=bool)
    correlated_prediction = np.zeros((shots, columns), dtype=bool)
    cluster_gap = np.zeros((shots, columns), dtype=np.float64)
    settled_states = np.zeros((shots, columns), dtype=np.int64)
    forced_plain = np.zeros((shots, num_patches, NUM_SECTORS, NUM_SECTORS), dtype=np.float64)
    forced_correlated = np.zeros_like(forced_plain)
    reweighted_patches = np.zeros((shots, num_patches), dtype=bool)

    # Gathered once per patch so the per-row loop only indexes: each patch reads its own
    # detector ids out of the chunk.
    local = [patch.local_syndromes(detectors) for patch in patches]
    joint_mwpm = np.asarray(context.joint.decode_batch(detectors.astype(np.uint8))).astype(bool)

    counts = dict.fromkeys(WORK_FIELDS, 0)
    counts['rows'] = shots
    counts['joint_decodes'] = shots   # decode_batch decodes one syndrome per row
    for position in range(shots):
        row = int(rows[position])
        for index, patch in enumerate(patches):
            syndrome = local[index][position]
            result = context.decoders[index].decode_with_gaps(syndrome)
            _validate_correction(patch, result, syndrome, row=row, index=index)
            forced = context.matchers[index].forced_weights(syndrome)
            sectors = slice(NUM_SECTORS * index, NUM_SECTORS * (index + 1))
            uf_reference[position, sectors] = result.prediction
            cluster_gap[position, sectors] = result.cluster_gap
            settled_states[position, sectors] = result.dijkstra_states
            mwpm_reference[position, sectors] = forced.first_pass
            correlated_prediction[position, sectors] = forced.correlated_prediction
            forced_plain[position, index] = forced.plain
            forced_correlated[position, index] = forced.correlated
            reweighted_patches[position, index] = forced.rules_fired
            counts['uf_decodes'] += 1
            counts['dijkstra_searches'] += len(result.dijkstra_states)
            counts['dijkstra_states'] += int(result.dijkstra_states.sum())
            counts['unforced_plain_calls'] += 1
            counts['plain_forced_calls'] += FORCED_CALLS_PER_DECODE
            counts['reweight_attempts'] += 1
            if forced.rules_fired:
                counts['correlated_forced_calls'] += FORCED_CALLS_PER_DECODE
                counts['correlated_validation_calls'] += 1

    record = L1Record(
        actual=actual, yoke=detectors[:, list(patches.yoke_detector_ids)], uf_reference=uf_reference,
        mwpm_reference=mwpm_reference, correlated_prediction=correlated_prediction,
        joint_mwpm=joint_mwpm, cluster_gap=cluster_gap, dijkstra_states=settled_states,
        forced_plain=forced_plain, forced_correlated=forced_correlated,
        reweighted_patches=reweighted_patches, rows=rows)
    return CollectedRows(record, CollectionWork(**counts))
