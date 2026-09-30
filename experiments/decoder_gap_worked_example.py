#!/usr/bin/env python3
"""Reproduce the tutorial's class-cost table with the real matching helpers.

This is a tiny illustrative graph, not an additional noise-model experiment.
Each sector has an ordinary-boundary route and a logical-boundary route.
Their costs make every forced-class calculation easy to check by hand.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(os.environ['DANTE_REPO'])
sys.path.insert(0, str(REPOSITORY / 'src'))

import numpy as np
import pymatching

from yoked.decoders._graph import DecodingGraph
from yoked.hierarchical._matching_gaps import CHECK_PATTERNS, _CheckMatrixGraph, signed_gaps
from yoked.hierarchical._outer_mwpm import mwpm_outer_log_odds_batch
from yoked.hierarchical._patch_graphs import _with_check_vertices


def main() -> None:
    # Edge fields are (detector, other detector or boundary, weight, logical mask).
    # X sector: ordinary route 3 + 5 = 8, logical route 5.
    # Z sector: ordinary route 3 + 4 = 7, logical route 11.
    edges = [
        (0, 1, 3.0, 0), (1, None, 5.0, 0), (0, None, 5.0, 1),
        (2, 3, 3.0, 0), (3, None, 4.0, 0), (2, None, 11.0, 2),
    ]
    graph = DecodingGraph(4, 2, edges)
    syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
    forced_matcher = _CheckMatrixGraph(_with_check_vertices(graph)).matcher()
    forced_syndromes = np.concatenate(
        [np.tile(syndrome, (4, 1)), CHECK_PATTERNS], axis=1)
    classes, costs = forced_matcher.decode_batch(forced_syndromes, return_weights=True)
    reference, native_cost = _CheckMatrixGraph(graph).matcher().decode(
        syndrome, return_weight=True)
    table = costs.reshape(2, 2)
    gaps = signed_gaps(table, reference)

    # These expectations come from summing the two displayed route costs.
    np.testing.assert_array_equal(classes, CHECK_PATTERNS)
    np.testing.assert_array_equal(table, [[15, 19], [12, 16]])
    np.testing.assert_array_equal(reference, [1, 0])
    np.testing.assert_array_equal(gaps, [3, 4])
    assert native_cost == 12
    assert table[0, 0] + table[1, 1] == table[0, 1] + table[1, 0]

    # The L2 example intentionally supplies two different confidence orderings
    # for the same hard predictions. No truth label is supplied to the decoder.
    references = np.array([[1, 0, 0, 1, 0, 0]], dtype=np.uint8)
    yoke = np.array([1], dtype=np.uint8)
    sigma = yoke ^ (references.sum(axis=1).astype(np.uint8) % 2)
    outer_results = {}
    for name, scores, expected_patch in (
        ('complementary_gap', [3, 9, 1, 7, 4, 8], 3),
        ('cluster_score', [0.5, 4, 0.8, 3, 2, 5], 1),
    ):
        result = mwpm_outer_log_odds_batch(np.array([scores]), sigma)
        selected = (np.flatnonzero(result.patterns[0]) + 1).tolist()
        assert selected == [expected_patch]
        prediction = references ^ result.patterns
        np.testing.assert_array_equal(prediction.sum(axis=1) % 2, yoke)
        outer_results[name] = {
            'scores': scores, 'reversed_patches_one_based': selected,
            'final_prediction': prediction[0].astype(int).tolist(),
        }

    output = Path(os.environ['DANTE_WORKSPACE']) / 'results/decoder_gap_explainer_detailed_20260924'
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        'purpose': 'Illustrative arithmetic; not SI1000 accuracy data',
        'pymatching_version': pymatching.__version__,
        'recipe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'graph_edges': edges, 'syndrome': syndrome.tolist(),
        'check_patterns': CHECK_PATTERNS.tolist(),
        'forced_predictions': classes.tolist(), 'class_costs': table.tolist(),
        'reference': reference.tolist(), 'reference_weight': float(native_cost),
        'complementary_gaps': gaps.tolist(),
        'sector_x_costs': [8, 5], 'sector_z_costs': [7, 11],
        'additivity_verified': True,
        'outer_reference': references[0].tolist(), 'outer_yoke': int(yoke[0]),
        'outer_residual_parity': int(sigma[0]), 'outer_results': outer_results,
        'illustrative_coupled_table': [[0, 5], [6, 1]],
        'illustrative_coupled_interaction': -10,
    }
    path = output / 'worked_example.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Verified the four class costs, both gaps, and both L2 decisions: {path}')


if __name__ == '__main__':
    main()
