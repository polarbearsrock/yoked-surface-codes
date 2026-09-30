#!/usr/bin/env python3
"""Replay four patch cases to separate UF conditioning from final decoding.

This is a small diagnostic ablation, not a new Monte Carlo experiment. It
reads two recorded d=15 syndromes and checks the original results before
changing one ingredient at a time. Private graph adapters are used only after
their source hashes are checked against the saved experiment manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pymatching
import stim

from yoked.decoders._correlated_union_find import CorrelatedUnionFindDecoder
from yoked.decoders._correlations import correlation_rules_from_dem
from yoked.decoders._union_find import UnionFindDecoder
from yoked.hierarchical._cluster_gap import ClusterGapUnionFindDecoder
from yoked.hierarchical._matching_gaps import CHECK_PATTERNS, _CheckMatrixGraph, signed_gaps
from yoked.hierarchical._patch_graphs import PatchGraphs


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bits(mask):
    return np.array([bool(mask & 1), bool(mask & 2)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.analysis / "analysis.json").read_text())
    d15 = next(row for row in report["distances"] if row["distance"] == 15)
    examples = d15["examples"]
    row_ids = np.array([e["row_id"] for e in examples])
    base = args.result / "run/d15/mwpm/d15"
    sample = base / "sample"
    metadata = json.loads((sample / "sample.json").read_text())
    manifest = json.loads((args.result / "distances/d15/manifest.json").read_text())
    assert metadata["identities"] == manifest["sample_identities"]
    assert digest(sample / "model.dem") == metadata["files"]["model.dem"]
    sources = {**manifest["baseline"]["provenance"]["source_sha256"],
               **manifest["provenance"]["source_sha256"]}
    for name, expected in sources.items():
        assert digest(args.repo / name) == expected, f"Source changed: {name}"

    # Memory-map the packed payload; unpack only the selected rows. The large
    # sample was verified when collected. Here the model hash, row truth, and
    # independently reproduced decoder outputs are checked again.
    packed = np.load(sample / "detectors_packed.npy", mmap_mode="r")
    detectors = np.unpackbits(packed[row_ids], axis=1, bitorder="little",
                              count=metadata["num_detectors"])
    packed_truth = np.load(sample / "actual_observables_packed.npy", mmap_mode="r")
    actual = np.unpackbits(packed_truth[row_ids], axis=1, bitorder="little", count=12)
    with np.load(args.result / "distances/d15/predictions.npz") as archive:
        saved_uf = archive["reference"][row_ids]
        saved_scores = archive["cluster_gap"][row_ids]
        saved_mwpm = archive["mwpm_reference"][row_ids]
        np.testing.assert_array_equal(actual, archive["actual"][row_ids])
    model = stim.DetectorErrorModel.from_file(sample / "model.dem")
    patches = PatchGraphs.from_yoked_dem(model, num_patches=6)
    cases = []

    for index, example in enumerate(examples):
        # Inspect the actual wrong patch and the wrongly selected alternative.
        relevant = sorted(set(example["uf_cluster_gap"]["wrong_l1_patches"])
                          | set(example["uf_cluster_gap"]["flipped_patches"]))
        for patch_id in relevant:
            patch = patches[patch_id]
            syndrome = patch.local_syndromes(detectors[index]).astype(np.uint8)
            free = _CheckMatrixGraph(patch.graph)
            check = _CheckMatrixGraph(patch.check_graph)
            native = pymatching.Matching.from_detector_error_model(
                patch.local_dem, enable_correlations=True)
            native_first = free.edge_ids_of(
                native.decode_to_edges_array(syndrome, enable_correlations=False))
            native_reference = native.decode(syndrome, enable_correlations=True).astype(bool)
            uf_first = UnionFindDecoder(patch.graph).decode_to_edge_ids(syndrome)
            correlated = CorrelatedUnionFindDecoder(
                patch.graph, correlation_rules=correlation_rules_from_dem(patch.graph, patch.local_dem))
            search = ClusterGapUnionFindDecoder(patch.graph)

            # Hold UF's first-pass evidence fixed, and compare its final UF
            # correction with a matching solve on exactly those float weights.
            second = correlated._second_pass_decoder(uf_first) or UnionFindDecoder(patch.graph)
            uf_result = second.decode_with_growth_costs(syndrome)
            uf_reference = bits(uf_result.observable_mask)
            uf_gaps, _ = search.gaps_from_costs(uf_result.remaining_costs)
            weights = np.array([edge[2] for edge in second.graph.edges])
            mwpm_after_uf = free.matcher(weights).decode(syndrome).astype(bool)
            forced_syndromes = np.concatenate([
                np.broadcast_to(syndrome, (4, len(syndrome))), CHECK_PATTERNS], axis=1)
            _, forced_costs = check.matcher(weights).decode_batch(
                forced_syndromes, return_weights=True)
            forced_costs = forced_costs.reshape(2, 2)

            # Hold the final UF algorithm and its float correlation rules fixed,
            # but feed it the original MWPM first-pass correction as evidence.
            hybrid = correlated._second_pass_decoder(native_first) or UnionFindDecoder(patch.graph)
            hybrid_result = hybrid.decode_with_growth_costs(syndrome)
            hybrid_gaps, _ = search.gaps_from_costs(hybrid_result.remaining_costs)
            hybrid_weights = np.array([edge[2] for edge in hybrid.graph.edges])

            columns = slice(2 * patch_id, 2 * patch_id + 2)
            np.testing.assert_array_equal(uf_reference, saved_uf[index, columns])
            np.testing.assert_allclose(uf_gaps, saved_scores[index, columns], rtol=0, atol=1e-9)
            np.testing.assert_array_equal(native_reference, saved_mwpm[index, columns])
            first_mask = 0
            for edge_id in uf_first:
                first_mask ^= patch.graph.edges[edge_id][3]
            cases.append(dict(
                row_id=int(row_ids[index]), patch_id=patch_id, sector=example["sector"],
                actual_XZ=actual[index, columns].tolist(),
                uf_first_pass_reference_XZ=bits(first_mask).astype(int).tolist(),
                native_correlated_mwpm_XZ=native_reference.astype(int).tolist(),
                original_correlated_uf_XZ=uf_reference.astype(int).tolist(),
                original_uf_cluster_gaps_XZ=uf_gaps.tolist(),
                mwpm_on_uf_conditioned_weights_XZ=mwpm_after_uf.astype(int).tolist(),
                uf_with_mwpm_first_pass_XZ=bits(hybrid_result.observable_mask).astype(int).tolist(),
                uf_with_mwpm_first_pass_gaps_XZ=hybrid_gaps.tolist(),
                first_pass_edge_symmetric_difference=len(set(uf_first) ^ set(native_first)),
                different_conditioned_edge_weights=int(np.count_nonzero(np.abs(weights - hybrid_weights) > 1e-9)),
                uf_conditioned_forced_class_costs=forced_costs.tolist(),
                uf_conditioned_signed_matching_gaps_at_uf_reference=signed_gaps(forced_costs, uf_reference).tolist(),
                uf_correction_cost=float(weights[list(uf_result.selected_edges)].sum()),
                minimum_matching_cost=float(forced_costs.min()),
                minimum_matching_cost_in_uf_class=float(forced_costs[tuple(uf_reference.astype(int))]),
            ))
            print(f"Replayed row {row_ids[index]}, patch {patch_id}: saved outputs reproduced", flush=True)

    output = dict(
        cases=cases,
        note="Four illustrative patch cases; not a population-wide causal estimate. Matching counterfactuals use PyMatching's float-weight import and its internal quantization.",
        validation=dict(original_outputs_reproduced=True, model_sha256=digest(sample / "model.dem"),
                        source_hashes_verified=sources,
                        selected_packed_rows_sha256=hashlib.sha256(packed[row_ids].tobytes()).hexdigest(),
                        full_packed_sample_hash_recomputed=False),
    )
    (args.analysis / "example_replay.json").write_text(json.dumps(output, indent=2) + "\n")
    shutil.copy2(__file__, args.analysis / Path(__file__).name)


if __name__ == "__main__":
    main()
