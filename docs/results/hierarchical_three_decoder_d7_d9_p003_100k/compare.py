#!/usr/bin/env python3
"""Compare three decoders on verified d=7 and d=9 hierarchical records.

The bootstrap compresses each distance's shots into the eight joint failure
states of the three decoders. Drawing their counts from the empirical
multinomial distribution is distributionally identical to resampling whole
shots with replacement, while preserving all within-shot decoder dependence.
Distances are resampled independently.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sinter

from yoked.hierarchical._provenance import sha256_file
from yoked.hierarchical._record import load_record
from yoked.hierarchical._stages import (
    REPLAY_MANIFEST,
    _load_result,
    _require_same_record,
    _verified_replay_manifest,
    load_calibrators,
)

DECODERS = (
    ("correlated_mwpm", "Correlated MWPM"),
    ("correlated_uf", "Correlated UF"),
    ("hierarchical_uf", "Hierarchical UF + correlated gap"),
)
HIER_CONFIG_DIR = "uf_cluster_gap_to_gap_correlated_all_refined_mixed"
PERCENTILES = (2.5, 97.5)


def _json_ready(value):
    """Recursively turn verified read-only mappings and NumPy scalars into JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ler(rate: np.ndarray | float, pieces: int) -> np.ndarray | float:
    values = np.asarray(rate, dtype=float)
    converted = np.array([
        sinter.shot_error_rate_to_piece_error_rate(float(v), pieces=pieces, values=8)
        for v in values.ravel()
    ]).reshape(values.shape)
    return float(converted) if converted.ndim == 0 else converted


def _interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, PERCENTILES)]


def _load(root: Path, distance: int) -> tuple[dict, np.ndarray]:
    record_dir = root / "evaluation"
    replay_dir = root / "replay_full"
    loaded = load_record(record_dir)
    if loaded.record.shots != 100_000:
        raise ValueError(f"{record_dir} has {loaded.record.shots} shots, expected 100000")
    parameters = loaded.manifest["parameters"]
    expected = {"distance": distance, "p": 0.003, "rounds": 4 * distance,
                "patches": 6, "yokes": 2}
    for key, value in expected.items():
        if parameters[key] != value:
            raise ValueError(f"{record_dir} parameter {key}={parameters[key]!r}, expected {value!r}")
    sample_inputs = loaded.manifest["sample_identity_inputs"]
    expected_rows = {"start": 0, "stop": 100_000, "count": 100_000}
    if loaded.manifest["role"] != "evaluation" \
            or {key: loaded.manifest["rows"][key] for key in expected_rows} != expected_rows \
            or sample_inputs["seed"] != 42 \
            or sample_inputs["parent_shots"] != 100_000:
        raise ValueError(f"{record_dir} is not all 100000 parent rows of the seed-42 evaluation")
    manifest = _verified_replay_manifest(replay_dir)
    _require_same_record(loaded, manifest["record"], where=replay_dir / REPLAY_MANIFEST)
    calibrator_path = Path(manifest["calibrators"]["path"])
    if sha256_file(calibrator_path) != manifest["calibrators"]["sha256"]:
        raise ValueError(f"{calibrator_path} no longer has the hash named by the replay")
    _, calibration = load_calibrators(calibrator_path)
    if calibration["identity"] != manifest["calibrators"]["identity"]:
        raise ValueError("the verified calibrator identity differs from the replay's identity")
    calibration_record = load_record(calibration["record"]["directory"])
    _require_same_record(calibration_record, calibration["record"], where=calibrator_path)
    cal_inputs = calibration_record.manifest["sample_identity_inputs"]
    cal_rows = calibration_record.manifest["rows"]
    if calibration_record.manifest["role"] != "calibration" \
            or {key: cal_rows[key] for key in expected_rows} != {"start": 0, "stop": 50_000, "count": 50_000} \
            or cal_inputs["seed"] != 142 or cal_inputs["parent_shots"] != 50_000:
        raise ValueError(f"{calibration_record.directory} is not all 50000 rows of seed-142 calibration")
    if calibration_record.identities["sampling_family"] == loaded.identities["sampling_family"]:
        raise ValueError("calibration and evaluation unexpectedly share a sampling family")
    if calibration_record.identities["model"] != loaded.identities["model"]:
        raise ValueError("calibration and evaluation do not share a model identity")
    result, metadata = _load_result(replay_dir / HIER_CONFIG_DIR)
    expected_config = "uf:cluster_gap->gap_correlated:all_refined:mixed"
    if metadata["config"] != expected_config:
        raise ValueError(f"hierarchical result is {metadata['config']!r}, expected {expected_config!r}")
    record = loaded.record
    required = ("joint_correlated_mwpm", "joint_correlated_uf")
    missing = [name for name in required if name not in record.baselines]
    if missing:
        raise ValueError(f"{record_dir} lacks verified baselines {missing}")
    predictions = np.stack([
        record.baselines["joint_correlated_mwpm"],
        record.baselines["joint_correlated_uf"],
        result.final,
    ])
    failures = np.any(predictions != record.actual[None, :, :], axis=2).T
    provenance = {
        "root": str(root.resolve()),
        "record_manifest_sha256": sha256_file(record_dir / "manifest.json"),
        "record_sha256": loaded.manifest["artifacts"]["record.npz"],
        "replay_manifest_sha256": sha256_file(replay_dir / REPLAY_MANIFEST),
        "replay_identity": manifest["identity"],
        "collection_identity": loaded.identities["collection"],
        "evaluation": {
            "seed": sample_inputs["seed"], "parent_shots": sample_inputs["parent_shots"],
            "rows": dict(loaded.manifest["rows"]), "identities": dict(loaded.identities),
            "sample_identity_inputs": dict(sample_inputs),
            "source_sha256": dict(loaded.manifest["source_sha256"]),
            "versions": dict(loaded.manifest["versions"]),
            "baseline_attachment": dict(loaded.manifest["baselines"]),
        },
        "calibration": {
            "path": str(calibrator_path.resolve()), "sha256": manifest["calibrators"]["sha256"],
            "identity": calibration["identity"], "payload_sha256": calibration["payload_sha256"],
            "record": dict(calibration["record"]), "seed": cal_inputs["seed"],
            "parent_shots": cal_inputs["parent_shots"],
            "sample_identity_inputs": dict(cal_inputs),
            "record_source_sha256": dict(calibration_record.manifest["source_sha256"]),
            "record_versions": dict(calibration_record.manifest["versions"]),
            "source_sha256": dict(calibration["source_sha256"]),
            "versions": dict(calibration["versions"]), "code_commit": calibration["code_commit"],
        },
        "replay": {
            "identity": manifest["identity"], "configurations": list(manifest["configurations"]),
            "source_sha256": dict(manifest["source_sha256"]), "versions": dict(manifest["versions"]),
            "code_commit": manifest["code_commit"], "calibrators": dict(manifest["calibrators"]),
            "record": dict(manifest["record"]), "artifacts": dict(manifest["artifacts"]),
        },
        "shots": record.shots,
        "parameters": {key: parameters[key] for key in expected},
    }
    return _json_ready(provenance), failures


def _state_counts(failures: np.ndarray) -> np.ndarray:
    codes = failures[:, 0].astype(np.uint8) | (failures[:, 1].astype(np.uint8) << 1) \
        | (failures[:, 2].astype(np.uint8) << 2)
    return np.bincount(codes, minlength=8)


def _bootstrap(counts: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    states = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(float)
    sampled = rng.multinomial(int(counts.sum()), counts / counts.sum(), size=replicates)
    return sampled @ states / counts.sum()


def _distance_summary(distance: int, provenance: dict, failures: np.ndarray,
                      boot_rates: np.ndarray, replicates: int, seed: int) -> dict:
    shots = failures.shape[0]
    pieces = 6 * 4 * distance
    rates = failures.mean(axis=0)
    boot_ler = _ler(boot_rates, pieces)
    entries = {}
    for index, (name, label) in enumerate(DECODERS):
        entries[name] = {
            "label": label,
            "block_failures": int(failures[:, index].sum()),
            "shots": shots,
            "block_failure_rate": float(rates[index]),
            "block_failure_95_ci": _interval(boot_rates[:, index]),
            "normalized_ler": _ler(rates[index], pieces),
            "normalized_ler_95_ci": _interval(boot_ler[:, index]),
        }
    h = 2
    for base in (0, 1):
        name = DECODERS[base][0]
        diff = boot_rates[:, h] - boot_rates[:, base]
        ratio = boot_ler[:, h] / boot_ler[:, base]
        discordant = {
            "baseline_succeeds_hierarchical_fails": int((~failures[:, base] & failures[:, h]).sum()),
            "baseline_fails_hierarchical_succeeds": int((failures[:, base] & ~failures[:, h]).sum()),
            "both_succeed": int((~failures[:, base] & ~failures[:, h]).sum()),
            "both_fail": int((failures[:, base] & failures[:, h]).sum()),
        }
        entries["hierarchical_uf"][f"paired_block_difference_vs_{name}"] = {
            "estimate": float(rates[h] - rates[base]), "95_ci": _interval(diff),
            "joint_outcomes": discordant}
        entries["hierarchical_uf"][f"paired_ler_ratio_vs_{name}"] = {
            "estimate": float(_ler(rates[h], pieces) / _ler(rates[base], pieces)),
            "95_ci": _interval(ratio)}
    return {"distance": distance, "pieces": pieces, "values": 8,
            "bootstrap": {"replicates": replicates, "seed": seed,
                          "method": "empirical multinomial over 8 joint failure states"},
            "provenance": provenance, "joint_failure_state_counts": _state_counts(failures).tolist(),
            "decoders": entries}


def _fmt_ci(ci: list[float], digits: int = 6) -> str:
    return f"({ci[0]:.{digits}f}, {ci[1]:.{digits}f})"


def _markdown(payload: dict) -> str:
    settings = payload["bootstrap"]
    lines = ["# Correlated and hierarchical decoder comparison", "",
             "Configuration: SI1000 `p=0.003`, six patches, two ideal yokes, and `rounds=4d`. "
             "The hierarchical row is `uf:cluster_gap -> gap_correlated`, `all_refined`, "
             "with the mixed exact outer rule.", "",
             "Evaluation uses all 100,000 saved seed-42 shots at each distance; calibration uses "
             "a separate 50,000-shot seed-142 call at that distance. All comparisons use the "
             "same evaluation shots within a distance. The two distances are separate sampling "
             "calls and are resampled independently; their row numbers are not paired. "
             f"Intervals use {settings['replicates']:,} empirical whole-shot bootstrap "
             f"replicates at seed {settings['seed']}; the eight joint decoder-failure states "
             "preserve pairing. Calibration "
             "is held fixed, so these intervals quantify evaluation-shot uncertainty conditional "
             "on the fitted calibrators. Block failure means any of the 12 observables is wrong; "
             "normalized LER is Sinter's aggregate conversion of that block rate, not a directly "
             "counted per-patch error rate.", ""]
    for key in ("d7", "d9"):
        item = payload["distances"][key]
        lines += [f"## d={item['distance']}", "",
                  "| Decoder | Failed shots | Block rate (95% CI) | Normalized LER (95% CI) |",
                  "|---|---:|---:|---:|"]
        for name, label in DECODERS:
            d = item["decoders"][name]
            lines.append(f"| {label} | {d['block_failures']:,} / {d['shots']:,} | "
                         f"{d['block_failure_rate']:.6f} {_fmt_ci(d['block_failure_95_ci'])} | "
                         f"{d['normalized_ler']:.8g} {_fmt_ci(d['normalized_ler_95_ci'], 8)} |")
        h = item["decoders"]["hierarchical_uf"]
        lines += ["", "Paired hierarchical comparisons:", "",
                  "| Baseline | Block-rate difference (95% CI) | Normalized-LER ratio (95% CI) |",
                  "|---|---:|---:|"]
        for name, label in DECODERS[:2]:
            diff = h[f"paired_block_difference_vs_{name}"]
            ratio = h[f"paired_ler_ratio_vs_{name}"]
            lines.append(f"| {label} | {diff['estimate']:.6f} {_fmt_ci(diff['95_ci'])} | "
                         f"{ratio['estimate']:.6f} {_fmt_ci(ratio['95_ci'])} |")
        lines += ["", "Discordant and concordant shot counts:", "",
                  "| Baseline | Baseline succeeds / hierarchy fails | Baseline fails / hierarchy succeeds | Both succeed | Both fail |",
                  "|---|---:|---:|---:|---:|"]
        for name, label in DECODERS[:2]:
            outcomes = h[f"paired_block_difference_vs_{name}"]["joint_outcomes"]
            lines.append(f"| {label} | {outcomes['baseline_succeeds_hierarchical_fails']:,} | "
                         f"{outcomes['baseline_fails_hierarchical_succeeds']:,} | "
                         f"{outcomes['both_succeed']:,} | {outcomes['both_fail']:,} |")
        lines.append("")
    lines += ["## Distance scaling", "",
              "Ratios are d=7 normalized LER divided by d=9 normalized LER, with the two "
              "distances bootstrapped independently.", "",
              "| Decoder | d7 / d9 normalized-LER ratio (95% CI) |", "|---|---:|"]
    for name, label in DECODERS:
        ratio = payload["distance_ratios"][name]
        lines.append(f"| {label} | {ratio['estimate']:.6f} {_fmt_ci(ratio['95_ci'])} |")
    lines += ["", "## Cost scope", "",
              "`all_refined` is a cost-unconstrained endpoint, not a selective-refinement policy. "
              "The stored work counters count decoder operations rather than elapsed time, and no "
              "comparable end-to-end latency benchmark was run, so these results support no latency "
              "or speedup claim."]
    return "\n".join(lines) + "\n"


def _plot(payload: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    x = np.arange(2)
    width = 0.23
    for i, (name, label) in enumerate(DECODERS):
        ys, lo, hi = [], [], []
        for key in ("d7", "d9"):
            d = payload["distances"][key]["decoders"][name]
            ys.append(d["normalized_ler"])
            lo.append(d["normalized_ler_95_ci"][0])
            hi.append(d["normalized_ler_95_ci"][1])
        ys = np.array(ys)
        err = np.vstack([ys - lo, np.array(hi) - ys])
        ax.errorbar(x + (i - 1) * width, ys, yerr=err, fmt="o", capsize=3, label=label)
    ax.set_xticks(x, ["d=7", "d=9"])
    ax.set_yscale("log")
    ax.set_ylabel("Normalized logical error rate")
    ax.set_title("p = 0.003 · 6 patches · rounds = 4d\n100,000 evaluation shots per distance · 95% intervals", fontsize=11)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d7-root", type=Path, required=True)
    parser.add_argument("--d9-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.replicates < 1 or args.seed < 0:
        parser.error("replicates must be positive and seed nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = {"d7": _load(args.d7_root, 7), "d9": _load(args.d9_root, 9)}
    rng = np.random.default_rng(args.seed)
    boots = {key: _bootstrap(_state_counts(failures), args.replicates, rng)
             for key, (_, failures) in loaded.items()}
    distances = {key: _distance_summary(distance, *loaded[key], boots[key], args.replicates, args.seed)
                 for key, distance in (("d7", 7), ("d9", 9))}
    ratios = {}
    for index, (name, _) in enumerate(DECODERS):
        d7 = distances["d7"]["decoders"][name]["normalized_ler"]
        d9 = distances["d9"]["decoders"][name]["normalized_ler"]
        draw = _ler(boots["d7"][:, index], distances["d7"]["pieces"]) / \
            _ler(boots["d9"][:, index], distances["d9"]["pieces"])
        ratios[name] = {"estimate": d7 / d9, "95_ci": _interval(draw)}
    payload = {"bootstrap": {"replicates": args.replicates, "seed": args.seed},
               "distances": distances, "distance_ratios": ratios}
    (args.out / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out / "comparison.md").write_text(_markdown(payload))
    _plot(payload, args.out / "comparison.png")


if __name__ == "__main__":
    main()
