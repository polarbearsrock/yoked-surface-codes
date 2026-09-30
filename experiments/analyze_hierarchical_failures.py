#!/usr/bin/env python3
"""Attribute saved hierarchical-decoder failures without running a new sweep.

The current experiment has one ideal parity check per X/Z sector and six
patches. With nonnegative outer weights, L2 flips the least-confident patch
for odd parity and does nothing for even parity. We verify that rule against
every saved decision before using it to classify failures.

Counts of sectors and counts of whole blocks are kept separate: a block has
two sectors and can fail in both. The shared-reference comparison controls
for L1's *logical output*, not its internal clusters or correlation evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIE_TOLERANCE = 1e-9  # The tolerance used by the archived L2 implementation.
METHODS = ("mwpm_mpp", "uf_cluster_gap")
LABELS = ("Correlated MWPM + MPP", "Correlated UF + cluster gap")
COLORS = ("#278f83", "#d55e00")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_arrays(folder: Path, keys: tuple[str, ...]):
    """Validate the saved container before loading only the arrays we need."""
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in ("results.json", "predictions.npz"):
        if sha256(folder / name) != manifest["artifacts"][name]:
            raise ValueError(f"Artifact hash mismatch: {folder / name}")
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in keys}
    return arrays, manifest


def by_sector(array: np.ndarray) -> np.ndarray:
    """Convert patch-major columns to (shots, X/Z sector, patch)."""
    return array.reshape(len(array), -1, 2).transpose(0, 2, 1)


def count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def paired_counts(baseline, candidate, mask=None):
    """A paired success/failure table, using the same eligible rows on both sides."""
    if mask is None:
        mask = np.ones_like(baseline, dtype=bool)
    regressions = count(mask & candidate & ~baseline)
    repairs = count(mask & baseline & ~candidate)
    return dict(
        eligible=count(mask), baseline_failures=count(mask & baseline),
        candidate_failures=count(mask & candidate),
        both_fail=count(mask & baseline & candidate),
        both_succeed=count(mask & ~baseline & ~candidate),
        candidate_only_fail=regressions, baseline_only_fail=repairs,
        net_excess_failures=regressions - repairs,
    )


def audit_method(actual, reference, prediction, scores, sigma, saved_ties):
    """Check L2, then partition observed failures by the number of L1 errors."""
    actual, reference, prediction, scores = map(
        by_sector, (actual, reference, prediction, scores))
    assert np.isfinite(scores).all() and (scores >= 0).all()
    errors = actual ^ reference
    error_count = errors.sum(axis=2)
    flips = reference ^ prediction
    failures = (actual ^ prediction).any(axis=2)
    odd = error_count % 2 == 1
    np.testing.assert_array_equal(sigma, odd)

    # Independent vectorized solution of the nonnegative single-parity model.
    # The first acceptable patch gives the lowest binary correction on a tie.
    minimum = scores.min(axis=2, keepdims=True)
    chosen = (scores <= minimum + TIE_TOLERANCE).argmax(axis=2)
    expected = np.zeros_like(flips)
    shot, sector = np.nonzero(odd)
    expected[shot, sector, chosen[shot, sector]] = True
    np.testing.assert_array_equal(flips, expected)
    np.testing.assert_array_equal((actual ^ prediction).sum(axis=2) % 2, 0)
    ordered_scores = np.sort(scores, axis=2)
    expected_ties = np.where(
        odd, ordered_scores[:, :, 1] - ordered_scores[:, :, 0] <= TIE_TOLERANCE,
        ordered_scores[:, :, 1] + ordered_scores[:, :, 0] <= TIE_TOLERANCE)
    np.testing.assert_array_equal(saved_ties, expected_ties)

    single = error_count == 1
    wrong_patch = single & failures
    even_multiple = (error_count >= 2) & ~odd
    odd_multiple = (error_count >= 3) & odd
    multiple = even_multiple | odd_multiple
    np.testing.assert_array_equal(failures, wrong_patch | multiple)

    # Each block is counted once in this disjoint partition. The sector
    # partition above remains available for X/Z-specific interpretation.
    any_wrong = wrong_patch.any(axis=1)
    any_multiple = multiple.any(axis=1)
    block_failures = failures.any(axis=1)
    block_partition = dict(
        single_error_misranking_only=count(any_wrong & ~any_multiple),
        multiple_errors_only=count(any_multiple & ~any_wrong),
        both=count(any_multiple & any_wrong),
    )
    assert sum(block_partition.values()) == count(block_failures)

    # Rank only on sectors with exactly one wrong L1 bit, where there is an
    # unambiguous true target. Near ties use the same tolerance as L2.
    true_patch = errors.argmax(axis=2)
    true_score = np.take_along_axis(scores, true_patch[..., None], axis=2)[..., 0]
    ahead = scores < true_score[..., None] - TIE_TOLERANCE
    near = np.abs(scores - true_score[..., None]) <= TIE_TOLERANCE
    ahead |= near & (np.arange(scores.shape[2]) < true_patch[..., None])
    rank = 1 + ahead.sum(axis=2)
    np.testing.assert_array_equal((rank > 1)[single], failures[single])

    metrics = dict(
        shots=len(actual), sector_trials=2 * len(actual),
        l1_block_errors=count(errors.any(axis=(1, 2))),
        l1_wrong_bits=count(errors),
        l1_error_count_histogram=np.bincount(error_count.ravel(), minlength=7).tolist(),
        single_error_sectors=count(single), wrong_patch_sectors=count(wrong_patch),
        wrong_patch_rate=count(wrong_patch) / count(single),
        even_multiple_error_sectors=count(even_multiple),
        odd_multiple_error_sectors=count(odd_multiple),
        multiple_error_sectors=count(multiple),
        final_failed_sectors=count(failures), final_failed_blocks=count(block_failures),
        block_failure_partition=block_partition,
        blocks_with_multiple_errors=count(any_multiple),
        failed_blocks_with_no_outer_syndrome=count(block_failures & ~odd.any(axis=1)),
        outer_tied_sectors=count(saved_ties),
        zero_score_patch_bits=count(scores == 0),
        true_error_rank_histogram=np.bincount(rank[single], minlength=7)[1:].tolist(),
        failure_sectors_by_basis=dict(X=count(failures[:, 0]), Z=count(failures[:, 1])),
        l2_rule_mismatches=0, final_parity_violations=0,
    )
    state = dict(reference=reference, scores=scores, errors=errors, k=error_count,
                 failures=failures, block_failures=block_failures, chosen=chosen,
                 flips=flips, odd=odd)
    return metrics, state


def example_row(row, basis, actual, states, kind):
    """Retain all six patch scores and bits for a concrete, reproducible failure."""
    result = dict(kind=kind, row_id=int(row), sector="XZ"[basis],
                  actual=by_sector(actual)[row, basis].astype(int).tolist(),
                  patch_ids=list(range(6)))
    for name, state in states.items():
        result[name] = dict(
            reference=state["reference"][row, basis].astype(int).tolist(),
            wrong_l1_patches=np.flatnonzero(state["errors"][row, basis]).tolist(),
            scores=state["scores"][row, basis].tolist(),
            outer_syndrome=int(state["odd"][row, basis]),
            flipped_patches=np.flatnonzero(state["flips"][row, basis]).tolist(),
            final_wrong_patches=np.flatnonzero(
                state["errors"][row, basis] ^ state["flips"][row, basis]).tolist(),
        )
    return result


def analyze_distance(root: Path, distance: int):
    combined = root / "distances" / f"d{distance}"
    mwpm = root / "run" / f"d{distance}" / "mwpm" / f"d{distance}"
    uf, uf_manifest = checked_arrays(combined, (
        "row_ids", "actual", "reference", "prediction", "cluster_gap", "sigma",
        "outer_tied", "mwpm_reference", "mpp_prediction"))
    matching, matching_manifest = checked_arrays(mwpm, (
        "row_ids", "actual", "reference", "mpp_prediction", "cluster_score",
        "sigma", "mpp_outer_tied"))
    assert uf_manifest["sample_identities"] == matching_manifest["sample_identities"]
    np.testing.assert_array_equal(uf["row_ids"], np.arange(1_000_000))
    for left, right in (("row_ids", "row_ids"), ("actual", "actual"),
                        ("mwpm_reference", "reference"), ("mpp_prediction", "mpp_prediction")):
        np.testing.assert_array_equal(uf[left], matching[right])
    saved = json.loads((combined / "results.json").read_text())
    assert saved["parameters"] == dict(distance=distance, noise="si1000", p=0.003,
                                      patches=6, rounds=4 * distance, style="cz", yokes=2)
    metrics, states = {}, {}
    metrics[METHODS[0]], states[METHODS[0]] = audit_method(
        matching["actual"], matching["reference"], matching["mpp_prediction"],
        matching["cluster_score"], matching["sigma"], matching["mpp_outer_tied"])
    metrics[METHODS[1]], states[METHODS[1]] = audit_method(
        uf["actual"], uf["reference"], uf["prediction"], uf["cluster_gap"], uf["sigma"], uf["outer_tied"])
    for name, saved_name in zip(METHODS, ("correlated_mwpm_mpp", "correlated_uf_cluster_gap")):
        assert metrics[name]["final_failed_blocks"] == saved["configurations"][saved_name]["block_failures"]

    m, u = (states[name] for name in METHODS)
    same = (m["reference"] == u["reference"]).all(axis=2)
    comparisons = dict(
        all_blocks=paired_counts(m["block_failures"], u["block_failures"]),
        identical_l1_blocks=paired_counts(m["block_failures"], u["block_failures"], same.all(axis=1)),
        different_l1_blocks=paired_counts(m["block_failures"], u["block_failures"], ~same.all(axis=1)),
        identical_l1_single_error_sectors=paired_counts(m["failures"], u["failures"], same & (m["k"] == 1)),
        different_l1_sectors=paired_counts(m["failures"], u["failures"], ~same),
    )

    # Symmetric algebraic decomposition; this is descriptive, not a causal
    # swap experiment. The two single-error populations contain different shots.
    mm, uu = (metrics[name] for name in METHODS)
    n_m, n_u = mm["single_error_sectors"], uu["single_error_sectors"]
    q_m, q_u = mm["wrong_patch_rate"], uu["wrong_patch_rate"]
    decomposition = dict(
        excess_multiple_error_sectors=uu["multiple_error_sectors"] - mm["multiple_error_sectors"],
        extra_single_error_exposure=(n_u - n_m) * (q_u + q_m) / 2,
        changed_conditional_misranking=(q_u - q_m) * (n_u + n_m) / 2,
    )
    assert np.isclose(sum(decomposition.values()), uu["final_failed_sectors"] - mm["final_failed_sectors"])

    examples = []
    # Select the first eligible shot, avoiding retrospective selection of the
    # most spectacular score error. Patch and row IDs are zero-based.
    if distance == 15:
        pure_score = same.all(axis=1) & (m["k"].sum(axis=1) == 1)
        pure_score &= u["block_failures"] & ~m["block_failures"]
        row = int(np.flatnonzero(pure_score)[0])
        examples.append(example_row(row, int(m["k"][row].argmax()), uf["actual"], states,
                                    "same L1 output; UF confidence picks the wrong patch"))
        parity_cancel = (m["k"].sum(axis=1) == 0) & (u["k"].max(axis=1) == 2)
        parity_cancel &= (u["k"].sum(axis=1) == 2)
        row = int(np.flatnonzero(parity_cancel)[0])
        examples.append(example_row(row, int(u["k"][row].argmax()), uf["actual"], states,
                                    "two UF L1 errors cancel in the outer parity check"))

    return dict(distance=distance, parameters=saved["parameters"], methods=metrics,
                paired=comparisons, descriptive_sector_decomposition=decomposition,
                examples=examples, sources=dict(
                    uf=dict(directory=str(combined.resolve()), artifacts=uf_manifest["artifacts"]),
                    mwpm=dict(directory=str(mwpm.resolve()), artifacts=matching_manifest["artifacts"]),
                    sample_identities=uf_manifest["sample_identities"]))


def draw_diagnostics(rows, out):
    """Separate how often L1 errs from how well confidence locates an error."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.4))
    fig.subplots_adjust(left=.06, right=.99, bottom=.21, top=.73, wspace=.3)
    distances = [r["distance"] for r in rows]
    for name, label, color, marker in zip(METHODS, LABELS, COLORS, ("D", "^")):
        values = [r["methods"][name] for r in rows]
        axes[0].plot(distances, [100*v["l1_block_errors"]/v["shots"] for v in values],
                     label=label, color=color, marker=marker, linewidth=2)
        axes[1].plot(distances, [100*v["wrong_patch_rate"] for v in values],
                     color=color, marker=marker, linewidth=2)
        key = "baseline_failures" if name == METHODS[0] else "candidate_failures"
        common = [r["paired"]["identical_l1_single_error_sectors"] for r in rows]
        axes[2].plot(distances, [100*v[key]/v["eligible"] for v in common],
                     color=color, marker=marker, linewidth=2)
    titles = ("L1 error burden", "Wrong-patch choices", "Same L1 output, different confidence")
    labels = ("Blocks with an L1 error (%)", "Failed single-error sectors (%)",
              "Failed shared single-error sectors (%)")
    for axis, title, label in zip(axes, titles, labels):
        axis.set(title=title, xlabel="Code distance, d", ylabel=label)
        axis.set_xticks(distances)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dce1e7", linewidth=.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .89), frameon=False, fontsize=12)
    fig.text(.5, .975, "Why the two hierarchies separate", ha="center", va="top", fontsize=18)
    fig.text(.5, .06, "SI1000 p = 0.3%  ·  4d rounds  ·  1M paired blocks per distance  ·  6 patches",
             ha="center", fontsize=11)
    fig.text(.5, .025, "Middle: each decoder’s own single-error cases. Right: identical L1 bits on the same sector-shots.",
             ha="center", fontsize=10, color="#4b5563")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"failure_diagnostics.{ext}", dpi=250, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=[7, 9, 11, 13, 15])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for distance in args.distances:
        row = analyze_distance(args.result, distance)
        rows.append(row)
        print(f"d={distance}: verified all paired decisions and failure categories", flush=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  recipe_sha256=sha256(Path(__file__)), distances=rows,
                  units="Counts are whole blocks or X/Z sector-shots, explicitly labelled; not per-round LER.")
    (args.out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "summary.csv").open("w", newline="") as stream:
        columns = ["distance", "method", "l1_block_errors", "single_error_sectors",
                   "wrong_patch_sectors", "wrong_patch_rate", "multiple_error_sectors",
                   "final_failed_blocks", "final_failed_sectors", "outer_tied_sectors"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for name in METHODS:
                metrics = row["methods"][name]
                writer.writerow(dict(distance=row["distance"], method=name,
                                     **{key: metrics[key] for key in columns[2:]}))
    draw_diagnostics(rows, args.out)
    shutil.copy2(__file__, args.out / "analyze_hierarchical_failures.py")
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
