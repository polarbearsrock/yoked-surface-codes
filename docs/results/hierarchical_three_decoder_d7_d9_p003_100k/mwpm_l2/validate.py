#!/usr/bin/env python3
"""Validate a new outer-MWPM replay against the frozen enumerating replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yoked.hierarchical._metrics import block_failures, normalized_ler
from yoked.hierarchical._record import load_record
from yoked.hierarchical._replay import Estimator, by_sector, calibrated_probabilities
from yoked.hierarchical._stages import (
    REPLAY_MANIFEST, _load_result, _require_same_record, _verified_replay_manifest,
    load_calibrators,
)

CONFIG_DIR = "uf_cluster_gap_to_gap_correlated_all_refined_mixed"
CONFIG = "uf:cluster_gap->gap_correlated:all_refined:mixed"
TOLERANCE = 1e-9


def _verified_result(record, replay_dir: Path):
    manifest = _verified_replay_manifest(replay_dir)
    _require_same_record(record, manifest["record"], where=replay_dir / REPLAY_MANIFEST)
    result, metadata = _load_result(replay_dir / CONFIG_DIR)
    if metadata["config"] != CONFIG:
        raise ValueError(f"{replay_dir / CONFIG_DIR} is {metadata['config']!r}, expected {CONFIG!r}")
    return manifest, result


def _log_weight(q: np.ndarray, reference: np.ndarray, final: np.ndarray) -> np.ndarray:
    residual = by_sector(reference ^ final).astype(float)
    return (residual * np.log(q) + (1 - residual) * np.log1p(-q)).sum(axis=2)


def _parity(final: np.ndarray, patches: int) -> np.ndarray:
    return (final.reshape(len(final), patches, 2).sum(axis=1) % 2).astype(bool)


def compare(distance: int, root: Path, old_dir: Path, new_dir: Path, calibrators_path: Path) -> dict:
    loaded = load_record(root / "evaluation")
    if loaded.manifest["parameters"]["distance"] != distance:
        raise ValueError(f"{root} is not d={distance}")
    old_manifest, old = _verified_result(loaded, old_dir)
    new_manifest, new = _verified_result(loaded, new_dir)
    calibrators, artifact = load_calibrators(calibrators_path)
    if old_manifest["calibrators"]["identity"] != artifact["identity"] \
            or new_manifest["calibrators"]["identity"] != artifact["identity"]:
        raise ValueError("old replay, new replay, and supplied calibrators do not share an identity")
    record = loaded.record
    estimator = Estimator.parse("uf:gap_correlated")
    q = calibrated_probabilities(record, estimator, calibrators)
    reference = record.uf_reference
    old_weight = _log_weight(q, reference, old.final)
    new_weight = _log_weight(q, reference, new.final)
    delta = new_weight - old_weight
    bit_diff = old.final != new.final
    row_diff = bit_diff.any(axis=1)
    tie_diff = old.ties != new.ties
    old_parity = _parity(old.final, record.num_patches)
    new_parity = _parity(new.final, record.num_patches)
    old_failed = block_failures(old.final, record.actual)
    new_failed = block_failures(new.final, record.actual)
    pieces = record.num_patches * loaded.manifest["parameters"]["rounds"]
    q_above = q > 0.5
    near_half = np.abs(np.log(q) - np.log1p(-q)) <= TOLERANCE
    near_per_sector = near_half.sum(axis=2)
    return {
        "distance": distance,
        "shots": record.shots,
        "patches": record.num_patches,
        "pieces": pieces,
        "old_replay_identity": old_manifest["identity"],
        "new_replay_identity": new_manifest["identity"],
        "calibrator_identity": artifact["identity"],
        "final": {
            "different_bits": int(bit_diff.sum()),
            "different_rows": int(row_diff.sum()),
            "first_different_rows": record.rows[np.flatnonzero(row_diff)[:20]].tolist(),
        },
        "ties": {
            "old_count": int(old.ties.sum()), "new_count": int(new.ties.sum()),
            "different_flags": int(tie_diff.sum()),
            "different_rows": int(tie_diff.any(axis=1).sum()),
        },
        "map_log_weight_new_minus_old": {
            "minimum": float(delta.min()), "maximum": float(delta.max()),
            "max_absolute": float(np.abs(delta).max()),
            "sectors_below_minus_tolerance": int((delta < -TOLERANCE).sum()),
            "sectors_above_tolerance": int((delta > TOLERANCE).sum()),
        },
        "parity": {
            "old_violations": int((old_parity != record.yoke).sum()),
            "new_violations": int((new_parity != record.yoke).sum()),
        },
        "accuracy": {
            "old_block_failures": int(old_failed.sum()),
            "new_block_failures": int(new_failed.sum()),
            "changed_failure_status_rows": int((old_failed != new_failed).sum()),
            "old_normalized_ler": normalized_ler(float(old_failed.mean()), pieces=pieces),
            "new_normalized_ler": normalized_ler(float(new_failed.mean()), pieces=pieces),
        },
        "calibrated_probabilities": {
            "values": int(q.size), "above_half": int(q_above.sum()),
            "shot_sectors_with_any_above_half": int(q_above.any(axis=2).sum()),
            "within_tie_tolerance_of_half": int(near_half.sum()),
            "shot_sectors_with_any_near_half": int((near_per_sector >= 1).sum()),
            "shot_sectors_with_two_or_more_near_half": int((near_per_sector >= 2).sum()),
            "minimum_abs_log_odds": float(np.abs(np.log(q) - np.log1p(-q)).min()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--d7-new", type=Path, required=True)
    parser.add_argument("--d9-new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old = {d: args.run_root / f"d{d}" / "replay_full" for d in (7, 9)}
    calibrators = {
        7: args.run_root / "d7" / "calibrators.json",
        9: Path(__import__("os").environ["TMPDIR"]) / "hier-d9-p003-m2" / "calibrators_full.json",
    }
    result = {
        "comparison": "new outer MWPM backend minus frozen enumerating backend",
        "tie_tolerance": TOLERANCE,
        "distances": {
            f"d{d}": compare(d, args.run_root / f"d{d}", old[d], new, calibrators[d])
            for d, new in ((7, args.d7_new), (9, args.d9_new))
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
