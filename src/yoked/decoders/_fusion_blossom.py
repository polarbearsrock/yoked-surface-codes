import numpy as np

from yoked.decoders._graph import DecodingGraph
from yoked.decoders._union_find import InvalidSyndromeError, _validate_syndromes


class FusionBlossomUnionFindDecoder:
    """Optional Fusion Blossom UF variant, using max_tree_size=0.

    This uses blossom cluster contraction and shortest-path reconstruction.
    Weights are rounded to even integers. Each instance reuses a native solver
    and must not be called concurrently.
    """

    def __init__(self, graph: DecodingGraph, *, weight_scale: int = 1000):
        try:
            import fusion_blossom as fb
        except ImportError as ex:
            raise ImportError(
                'Fusion Blossom UF is optional. Install it with '
                'uv pip install --python .venv/bin/python fusion-blossom==0.2.13'
            ) from ex
        if isinstance(weight_scale, bool) or not isinstance(weight_scale, int) or weight_scale <= 0:
            raise ValueError('weight_scale must be a positive integer')
        if len({tuple(sorted(pair)) for pair in graph.endpoints}) != len(graph.edges):
            raise ValueError('Fusion Blossom UF does not support parallel detector edges; import a DEM to merge them')
        try:
            weights = [2 * round(weight * weight_scale) for _, _, weight, _ in graph.edges]
        except OverflowError as ex:
            raise ValueError('Graph weights exceed the Fusion Blossom integer range; reduce weight_scale') from ex
        # The published wheel uses signed 32-bit weights. Include headroom for
        # intermediate sums as well as simple paths and corrections.
        if 2 * sum(weights) >= 2**31:
            raise ValueError('Graph weights exceed the Fusion Blossom integer range; reduce weight_scale')
        self.graph = graph
        self.weight_scale = weight_scale
        initializer = fb.SolverInitializer(
            len(graph.adjacency),
            [(u, v, weight) for (u, v), weight in zip(graph.endpoints, weights)],
            list(range(graph.num_detectors, len(graph.adjacency))),
        )
        self._solver = fb.SolverSerial(initializer, max_tree_size=0)
        self._syndrome_pattern = fb.SyndromePattern
        self._closed_components = _closed_components(graph)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Return predicted observables, rejecting impossible syndromes."""
        syndrome = _validate_syndromes(syndrome, self.graph.num_detectors, 1)
        for component in self._closed_components:
            if np.count_nonzero(syndrome[component]) % 2:
                raise InvalidSyndromeError('An odd detector component has no boundary')
        self._solver.clear()
        mask = 0
        if syndrome.any():
            self._solver.solve(self._syndrome_pattern(np.flatnonzero(syndrome).tolist()))
            for edge in self._solver.subgraph():
                mask ^= self.graph.edges[edge][3]
        return np.fromiter(
            ((mask >> k) & 1 for k in range(self.graph.num_observables)),
            dtype=np.bool_, count=self.graph.num_observables,
        )

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a binary (shots, num_detectors) array without changing it."""
        syndromes = _validate_syndromes(syndromes, self.graph.num_detectors, 2)
        predictions = np.empty((len(syndromes), self.graph.num_observables), dtype=np.bool_)
        for k, syndrome in enumerate(syndromes):
            predictions[k] = self.decode(syndrome)
        return predictions


def _closed_components(graph: DecodingGraph) -> list[np.ndarray]:
    """Cache components requiring even parity before entering the native solver."""
    seen = set()
    closed = []
    for start in range(graph.num_detectors):
        if start in seen:
            continue
        seen.add(start)
        pending = [start]
        detectors = []
        has_boundary = False
        while pending:
            vertex = pending.pop()
            if vertex < graph.num_detectors:
                detectors.append(vertex)
            else:
                has_boundary = True
            for edge in graph.adjacency[vertex]:
                u, v = graph.endpoints[edge]
                other = v if u == vertex else u
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        if not has_boundary:
            closed.append(np.array(detectors, dtype=np.intp))
    return closed
