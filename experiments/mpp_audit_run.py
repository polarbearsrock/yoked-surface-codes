#!/usr/bin/env python3
"""Audit a completed paired MPP run against its saved samples and manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(os.environ["DANTE_REPO"]) / "src"))
from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._outer_decoder import frame_adjusted_syndrome


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"Audit failed: {description}")


def audit_distance(root, distance, run):
    directory = root / f"d{distance}"
    manifest = read(directory / "manifest.json")
    for name, expected in manifest["artifacts"].items():
        require_equal(digest(directory / name), expected, name)
    for name, expected in manifest["shard_manifests"].items():
        shard = directory / name
        require_equal(digest(shard / "manifest.json"), expected, name)
        record = read(shard / "manifest.json")
        for key in ("source_sha256", "native_build", "versions"):
            require_equal(record[key], run["provenance"][key], f"shard {key}")
        require_equal(record["sample"]["identities"], manifest["sample_identities"], "shard sample identity")
        for artifact, expected in record["artifacts"].items():
            require_equal(digest(shard / artifact), expected, artifact)
        result = read(shard / "results.json")
        require_equal(result["reference_disagreements"], 0, "L1 reference agreement")
        require_equal(result["native_weight_disagreements"], 0, "L1 matching weight agreement")

    sample = SampleSet.load(directory / "sample")
    summary = read(root / "summary.json")[str(distance)]
    require_equal(read(directory / "results.json"), summary, "distance summary")
    require_equal(sample.parameters.to_json(), summary["parameters"], "sample parameters")
    if "rounds_by_distance" in run:
        require_equal(sample.parameters.rounds, run["rounds_by_distance"][str(distance)], "configured rounds")
        schedule = run["round_schedule"]
        expected_rounds = schedule["per_distance"] * distance if "per_distance" in schedule else schedule["fixed"]
        require_equal(sample.parameters.rounds, expected_rounds, "round schedule")
    require_equal(dict(sample.identities), manifest["sample_identities"], "sample identity")
    with np.load(directory / "predictions.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    np.testing.assert_array_equal(arrays["row_ids"], np.arange(run["shots_per_distance"]))
    truth = np.unpackbits(sample.actual_packed, axis=1, count=sample.num_observables, bitorder="little").astype(bool)
    np.testing.assert_array_equal(arrays["actual"], truth)

    # Check full shot coverage, XOR corrections, and measured yoke parity in
    # bounded batches rather than unpacking every detector in the run at once.
    for start in range(0, len(truth), 2500):
        stop = min(start + 2500, len(truth))
        detectors, _ = sample.rows(np.arange(start, stop))
        yokes = detectors[:, -2:]
        reference = arrays["reference"][start:stop]
        np.testing.assert_array_equal(arrays["sigma"][start:stop], frame_adjusted_syndrome(yokes, reference))
        for name in ("gap", "mpp"):
            prediction = arrays[f"{name}_prediction"][start:stop]
            np.testing.assert_array_equal(prediction, reference ^ arrays[f"{name}_residual"][start:stop])
            parity = prediction.reshape(stop - start, sample.parameters.patches, 2).sum(axis=1) % 2
            np.testing.assert_array_equal(parity, yokes)

    gap = np.any(arrays["gap_prediction"] != truth, axis=1)
    mpp = np.any(arrays["mpp_prediction"] != truth, axis=1)
    counts = {"both_succeed": int((~gap & ~mpp).sum()), "both_fail": int((gap & mpp).sum()),
              "mpp_repairs": int((gap & ~mpp).sum()), "mpp_regressions": int((~gap & mpp).sum())}
    require_equal(int(gap.sum()), summary["complementary_gap"]["block_failures"], "gap failure count")
    require_equal(int(mpp.sum()), summary["mpp"]["block_failures"], "MPP failure count")
    for name, count in counts.items():
        require_equal(count, summary["paired"][name], name)
    reference_errors = arrays["reference"] != truth
    sector_counts = reference_errors.reshape(len(truth), sample.parameters.patches, 2).sum(axis=1)
    return {
        "parameters": sample.parameters.to_json(), "shots": len(truth),
        "complete_unique_row_coverage": True, "sample_truth_matches": True,
        "artifact_and_source_hashes_verified": True, "residual_and_yoke_parity_verified": True,
        "gap_block_failures": int(gap.sum()), "mpp_block_failures": int(mpp.sum()), "paired": counts,
        "reference_error_blocks": int(np.any(reference_errors, axis=1).sum()),
        "single_error_sectors": int((sector_counts == 1).sum()),
        "multiple_error_sectors": int((sector_counts > 1).sum()),
        "final_prediction_vector_disagreements": int(np.any(arrays["gap_prediction"] != arrays["mpp_prediction"], axis=1).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    run, completion = read(root / "run.json"), read(root / "completion.json")
    for name in ("run", "summary"):
        require_equal(digest(root / f"{name}.json"), completion[f"{name}_sha256"], name)
    require_equal(digest(root / "recipe.py"), run["provenance"]["recipe_sha256"], "recipe")
    for name, expected in run["provenance"]["source_sha256"].items():
        require_equal(digest(root / "source_snapshot" / name), expected, name)
    audits = {}
    for distance in run["distances"]:
        require_equal(digest(root / f"d{distance}/manifest.json"),
                      completion["distance_manifests"][str(distance)], "distance manifest")
        audits[str(distance)] = audit_distance(root, distance, run)
    (root / "outcome_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
