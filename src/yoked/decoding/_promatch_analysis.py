"""Fail-closed analysis for paired L1 ProMatch experiment ledgers."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from scipy.stats import binomtest

from yoked.decoding._artifact_io import is_lowercase_hex, load_json_strict
from yoked.decoding._promatch_experiment import (
    ANALYSIS_CELL_FIELDS,
    ANALYSIS_COMMON_TOP_LEVEL_FIELDS,
    NONINFERIORITY_MARGIN_FRACTION,
    PILOT_SELECTION_ROW_FIELDS,
    PILOT_SELECTION_TOP_LEVEL_FIELDS,
    PROTOCOL_SCHEMA,
    SUMMARY_SCHEMA,
    _ledger_path,
    _phase_cells,
    _phase_schedules,
    _protocol_split,
    _validate_ledger_row,
    _verify_scientific_regeneration_without_writes,
    normalize_protocol,
    prepare_cell,
    summarize_ledgers,
    validate_experiment_protocol,
)
from yoked.decoding._promatch_stats import (
    PairedContingency,
    canonical_json_bytes,
    confirmatory_sample_size,
    paired_workload_histogram_bootstrap,
    simulate_tango_noninferiority_power,
    tango_paired_risk_difference_upper,
)


ANALYSIS_SCHEMA = "promatch-l1-analysis-v1"
PILOT_SELECTION_SCHEMA = "promatch-l1-pilot-selection-v1"

# Smoke is non-claim-bearing and uses these explicit conservative defaults
# only for implementation diagnostics, never for a scientific conclusion.
_SMOKE_ANALYSIS_CONFIG = {
    "alpha_one_sided": 0.025,
    "workload_bootstrap_replicates": 100,
    "workload_alpha_one_sided": 0.025,
    "workload_ratio_upper_threshold": 0.9,
}


# Artifact authentication and normalized inputs.


@dataclasses.dataclass(frozen=True)
class _VerificationProvenance:
    experiment_id: str
    phase: str
    deterministic_regeneration_passed: bool
    summary_sha256: str
    manifest_sha256: str


class _VerifiedSummary(dict[str, Any]):
    """Private marker carried only by the fail-closed artifact verifier."""

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        provenance: _VerificationProvenance,
    ) -> None:
        super().__init__(value)
        self.provenance = provenance


def _validate_pilot_selection_artifact(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(
        PILOT_SELECTION_TOP_LEVEL_FIELDS
    ):
        raise ValueError("generated pilot selection has incorrect top-level fields")
    rows = value.get("cells")
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) or set(row) != set(PILOT_SELECTION_ROW_FIELDS)
        for row in rows
    ):
        raise ValueError("generated pilot selection rows have incorrect fields")
    selected = value.get("selected")
    if selected is not None and (
        not isinstance(selected, Mapping)
        or set(selected) != set(PILOT_SELECTION_ROW_FIELDS)
        or dict(selected) not in [dict(row) for row in rows]
    ):
        raise ValueError("generated pilot selection has an invalid selected row")
    without_hash = dict(value)
    recorded_hash = without_hash.pop("selection_sha256", None)
    expected_hash = hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest()
    if recorded_hash != expected_hash:
        raise ValueError("generated pilot selection hash does not reconcile")


def validate_generated_analysis_artifact(value: Any) -> None:
    """Fail closed if an emitted analysis drifts from the V3 output contract."""

    if not isinstance(value, Mapping):
        raise ValueError("generated analysis must be an object")
    phase = value.get("phase")
    expected_top = set(ANALYSIS_COMMON_TOP_LEVEL_FIELDS)
    if phase == "pilot":
        expected_top.add("blinded_selection")
    if set(value) != expected_top:
        raise ValueError("generated analysis has incorrect top-level fields")
    cells = value.get("cells")
    if (
        not isinstance(cells, list)
        or not cells
        or any(
            not isinstance(cell, Mapping) or set(cell) != set(ANALYSIS_CELL_FIELDS)
            for cell in cells
        )
    ):
        raise ValueError("generated analysis cells have incorrect fields")
    if phase == "pilot":
        _validate_pilot_selection_artifact(value.get("blinded_selection"))
    without_hash = dict(value)
    recorded_hash = without_hash.pop("analysis_sha256", None)
    expected_hash = hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest()
    if recorded_hash != expected_hash:
        raise ValueError("generated analysis hash does not reconcile")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pilot_result_digest(input_directory: Path) -> str:
    """Digest only authoritative raw pilot artifacts in stable path order."""

    paths = [
        input_directory / "experiment.json",
        input_directory / "protocol.json",
        input_directory / "summary.json",
        *sorted((input_directory / "batches").glob("*/*.json")),
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError("pilot result directory is missing authoritative artifacts")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(input_directory).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _table(value: Mapping[str, Any], *, shots: int) -> PairedContingency:
    expected = {"both_correct", "regressions", "recoveries", "both_wrong"}
    if set(value) != expected:
        raise ValueError("paired contingency has incorrect fields")
    table = PairedContingency(**{key: value[key] for key in expected})
    if table.shots != shots:
        raise ValueError(
            f"paired contingency covers {table.shots} shots, expected {shots}"
        )
    return table


def _joint_workload_histogram(
    telemetry: Mapping[str, Any],
) -> dict[tuple[int, int], int]:
    raw = telemetry.get("original_residual_hw_joint_histogram")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "telemetry is missing original_residual_hw_joint_histogram; "
            "paired workload inference is impossible"
        )
    result: dict[tuple[int, int], int] = {}
    for key, count in raw.items():
        if not isinstance(key, str):
            raise ValueError("joint workload histogram keys must be strings")
        pieces = key.split(",")
        if len(pieces) != 2:
            raise ValueError(f"invalid joint workload key {key!r}")
        try:
            pair = int(pieces[0]), int(pieces[1])
        except ValueError as ex:
            raise ValueError(f"invalid joint workload key {key!r}") from ex
        if (
            min(pair) < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(
                "joint workload cells must be nonnegative with positive counts"
            )
        if pair in result:
            raise ValueError(f"duplicate normalized joint workload key {pair}")
        result[pair] = count
    return result


def _bootstrap_seed(manifest: Mapping[str, Any], *, cell_id: str) -> int:
    roots = _require_mapping(
        manifest.get("sampler_seed_roots"), name="sampler_seed_roots"
    )
    root = (
        roots.get("workload_bootstrap")
        or roots.get("timing_bootstrap")
        or roots.get(str(manifest.get("phase")))
        or roots.get("smoke")
    )
    if not isinstance(root, str) or len(root) != 64:
        raise ValueError("protocol needs a literal workload/timing bootstrap seed root")
    try:
        root_bytes = bytes.fromhex(root)
    except ValueError as ex:
        raise ValueError("bootstrap seed root must be hexadecimal") from ex
    digest = hashlib.sha256(
        root_bytes + b"workload-bootstrap" + cell_id.encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _analysis_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = manifest.get("analysis_config")
    if not isinstance(raw, Mapping):
        if manifest.get("phase") == "smoke":
            return dict(_SMOKE_ANALYSIS_CONFIG)
        raise ValueError("frozen protocol is missing analysis_config")
    workload = _require_mapping(raw.get("workload_protocol"), name="workload_protocol")
    if workload.get("quantile_method") != "empirical_type_7":
        raise ValueError("workload protocol requires empirical_type_7 quantiles")
    if (
        workload.get("bootstrap_unit")
        != "paired_shot_via_exact_joint_histogram_multinomial"
    ):
        raise ValueError("workload protocol has an unsupported bootstrap unit")
    phase = manifest.get("phase")
    if phase == "pilot":
        design = _require_mapping(
            raw.get("statistical_design"), name="statistical_design"
        )
        selection = _require_mapping(raw.get("selection_gates"), name="selection_gates")
        power = _require_mapping(
            design.get("power_verification"), name="power_verification"
        )
        pilot_cells = manifest.get("cells")
        if not isinstance(pilot_cells, list):
            raise ValueError("pilot manifest cells must be an array")
        return {
            "alpha_one_sided": design.get("noninferiority_alpha_one_sided"),
            "workload_bootstrap_replicates": workload.get(
                "paired_bootstrap_replicates"
            ),
            "workload_alpha_one_sided": workload.get("bootstrap_alpha_one_sided"),
            "workload_ratio_upper_threshold": workload.get(
                "workload_ratio_upper_threshold"
            ),
            "pilot_selection_gates": {
                "minimum_activation_fraction": selection.get(
                    "minimum_activation_fraction"
                ),
                "minimum_u0_failures": selection.get("minimum_u0_direct_failures"),
                "minimum_discordant_pairs": selection.get("minimum_discordant_pairs"),
                "require_integrity_checks": selection.get("require_integrity_checks"),
            },
            "ordered_pilot_cell_ids": [cell.get("cell_id") for cell in pilot_cells],
            "noninferiority_margin_fraction": NONINFERIORITY_MARGIN_FRACTION,
            "maximum_confirmatory_shots": selection.get(
                "maximum_confirmatory_paired_shots"
            ),
            "power_simulation_replicates": power.get("replicates"),
            "power_simulation_seed": power.get("seed"),
            "power_bound_alpha": power.get("power_bound_alpha"),
            "minimum_power_lower_bound": power.get("accept_if_lower_bound_at_least"),
        }
    if phase == "confirm":
        accuracy = _require_mapping(
            raw.get("accuracy_protocol"), name="accuracy_protocol"
        )
        selection = _require_mapping(raw.get("selection"), name="selection")
        return {
            "alpha_one_sided": accuracy.get("alpha_one_sided"),
            "delta_noninferiority": selection.get("delta_noninferiority"),
            "workload_bootstrap_replicates": workload.get(
                "paired_bootstrap_replicates"
            ),
            "workload_alpha_one_sided": workload.get("bootstrap_alpha_one_sided"),
            "workload_ratio_upper_threshold": workload.get(
                "workload_ratio_upper_threshold"
            ),
        }
    return dict(_SMOKE_ANALYSIS_CONFIG)


# Per-cell endpoints and frozen pilot selection.


def analyze_cell(
    cell: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    delta_noninferiority: float | None,
) -> dict[str, Any]:
    """Computes paired accuracy and workload endpoints for one summary cell.

    Denominators and telemetry are reconciled before any statistic is emitted.
    ``delta_noninferiority=None`` marks non-confirmatory phases explicitly.
    """

    shots = cell.get("shots")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("cell summary has invalid shots")
    table = _table(
        _require_mapping(cell.get("paired_contingency"), name="paired_contingency"),
        shots=shots,
    )
    telemetry = _require_mapping(cell.get("telemetry"), name="telemetry")
    if telemetry.get("shots") != shots:
        raise ValueError("telemetry shots do not reconcile")
    original_events = telemetry.get("original_event_sum")
    residual_events = telemetry.get("residual_event_sum")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (original_events, residual_events)
    ):
        raise ValueError("workload event sums must be nonnegative integers")
    if original_events == 0:
        workload_ratio = None
        workload_bootstrap = None
    else:
        workload_ratio = residual_events / original_events
        analysis_config = _analysis_config(manifest)
        replicates = int(analysis_config["workload_bootstrap_replicates"])
        alpha = float(analysis_config["workload_alpha_one_sided"])
        histogram = _joint_workload_histogram(telemetry)
        if sum(histogram.values()) != shots:
            raise ValueError("joint workload histogram does not reconcile to shots")
        workload_bootstrap = paired_workload_histogram_bootstrap(
            joint_counts=histogram,
            replicates=replicates,
            seed=_bootstrap_seed(manifest, cell_id=str(cell["cell_id"])),
            alpha=alpha,
        )

    analysis_config = _analysis_config(manifest)
    alpha = float(analysis_config["alpha_one_sided"])
    upper = tango_paired_risk_difference_upper(table, alpha=alpha)
    discordant = table.discordant
    mcnemar_p = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                table.regressions,
                discordant,
                p=0.5,
                alternative="less",
            ).pvalue
        )
    )
    activated = telemetry.get("activated_shots")
    rollback = telemetry.get("rollback_shots")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= shots
        for value in (activated, rollback)
    ):
        raise ValueError("activation/rollback telemetry does not reconcile")
    ni_passed = None if delta_noninferiority is None else upper < delta_noninferiority
    superiority_passed = None if ni_passed is None else bool(ni_passed and upper < 0)
    workload_threshold = float(analysis_config["workload_ratio_upper_threshold"])
    return {
        "cell_id": cell["cell_id"],
        "shots": shots,
        "paired_contingency": dataclasses.asdict(table),
        "u0_failure_rate": table.baseline_failures / shots,
        "pu_failure_rate": table.treatment_failures / shots,
        "delta_pu_minus_u0": table.delta,
        "tango_upper_one_sided": upper,
        "alpha_one_sided": alpha,
        "delta_noninferiority": delta_noninferiority,
        "noninferiority_passed": ni_passed,
        "ordered_superiority_passed": superiority_passed,
        "exact_mcnemar_superiority_p": mcnemar_p,
        "activation_fraction": activated / shots,
        "rollback_fraction": rollback / shots,
        "original_detector_events": original_events,
        "residual_detector_events": residual_events,
        "workload_ratio": workload_ratio,
        "workload_ratio_upper_one_sided": (
            None if workload_bootstrap is None else workload_bootstrap.upper_bound
        ),
        "workload_bootstrap_replicates": (
            None if workload_bootstrap is None else workload_bootstrap.replicates
        ),
        "workload_improvement_passed": (
            None
            if workload_bootstrap is None
            else workload_bootstrap.upper_bound < workload_threshold
        ),
        "workload_ratio_upper_threshold": workload_threshold,
    }


def _power_verified_size(
    *,
    pilot_discordant: int,
    pilot_shots: int,
    delta_noninferiority: float,
    config: Mapping[str, Any],
) -> tuple[Any, Any]:
    max_shots = int(config["maximum_confirmatory_shots"])
    design = confirmatory_sample_size(
        pilot_discordant=pilot_discordant,
        pilot_shots=pilot_shots,
        delta_noninferiority=delta_noninferiority,
        max_shots=max_shots,
    )
    if not design.fits_resource_cap:
        return design, None
    shots = design.rounded_shots
    while shots <= max_shots:
        power = simulate_tango_noninferiority_power(
            shots=shots,
            discordance_probability=design.discordance_upper,
            delta_noninferiority=delta_noninferiority,
            replicates=int(config["power_simulation_replicates"]),
            seed=int(config["power_simulation_seed"]),
            decision_alpha=float(config["alpha_one_sided"]),
            power_bound_alpha=float(config["power_bound_alpha"]),
        )
        if power.lower_bound >= float(config["minimum_power_lower_bound"]):
            return dataclasses.replace(design, rounded_shots=shots), power
        shots += 10_000
    return dataclasses.replace(
        design, rounded_shots=shots, fits_resource_cap=False
    ), power


def select_pilot_cell(
    summary: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the blinded fixed-order pilot rule without computing signed b-c."""

    provenance = getattr(summary, "provenance", None)
    if (
        not isinstance(summary, _VerifiedSummary)
        or not isinstance(provenance, _VerificationProvenance)
        or provenance.experiment_id != manifest.get("experiment_id")
        or provenance.phase != "pilot"
        or provenance.deterministic_regeneration_passed is not True
        or provenance.summary_sha256
        != hashlib.sha256(canonical_json_bytes(dict(summary))).hexdigest()
        or provenance.manifest_sha256
        != hashlib.sha256(canonical_json_bytes(dict(manifest))).hexdigest()
    ):
        raise ValueError(
            "pilot selection requires a scientifically verified summary from "
            "load_verified_summary"
        )

    config = _analysis_config(manifest)
    gates = _require_mapping(
        config.get("pilot_selection_gates"), name="pilot_selection_gates"
    )
    cells = summary.get("cells")
    if not isinstance(cells, list):
        raise ValueError("summary cells must be an array")
    by_id = {cell.get("cell_id"): cell for cell in cells if isinstance(cell, Mapping)}
    ordered_ids = config.get("ordered_pilot_cell_ids")
    if not isinstance(ordered_ids, list) or set(ordered_ids) != set(by_id):
        raise ValueError("pilot summary does not exactly cover the frozen ordered grid")
    rows = []
    selected = None
    for cell_id in ordered_ids:
        cell = by_id[cell_id]
        shots = int(cell["shots"])
        table = _table(
            _require_mapping(cell["paired_contingency"], name="paired_contingency"),
            shots=shots,
        )
        telemetry = _require_mapping(cell["telemetry"], name="telemetry")
        if telemetry.get("shots") != shots:
            raise ValueError("pilot telemetry shots do not reconcile")
        workload_histogram = _joint_workload_histogram(telemetry)
        if sum(workload_histogram.values()) != shots:
            raise ValueError("pilot workload histogram does not reconcile")
        integrity_checks_passed = True
        activation_fraction = int(telemetry["activated_shots"]) / shots
        baseline_failures = table.baseline_failures
        discordant_pairs = table.discordant
        p_u0_design = baseline_failures / shots
        delta_ni = float(config["noninferiority_margin_fraction"]) * p_u0_design
        if delta_ni > 0:
            design, power = _power_verified_size(
                pilot_discordant=discordant_pairs,
                pilot_shots=shots,
                delta_noninferiority=delta_ni,
                config=config,
            )
        else:
            design = power = None
        # Truth table, identical to the previous conditional expression:
        #   require_integrity is True  -> integrity_checks_passed
        #   require_integrity is False -> True
        #   anything else (missing, None, 1, "yes", ...) -> False (fail closed)
        require_integrity = gates.get("require_integrity_checks")
        integrity_gate_ok = require_integrity is False or (
            require_integrity is True and integrity_checks_passed
        )
        passed = (
            activation_fraction >= float(gates["minimum_activation_fraction"])
            and baseline_failures >= int(gates["minimum_u0_failures"])
            and discordant_pairs >= int(gates["minimum_discordant_pairs"])
            and integrity_gate_ok
            and design is not None
            and design.fits_resource_cap
            and power is not None
            and power.lower_bound >= float(config["minimum_power_lower_bound"])
        )
        row = {
            "cell_id": cell_id,
            "shots": shots,
            "activation_fraction": activation_fraction,
            "u0_failures": baseline_failures,
            "discordant_pairs": discordant_pairs,
            "integrity_checks_passed": integrity_checks_passed,
            "p_u0_design": p_u0_design,
            "delta_noninferiority": delta_ni,
            "discordance_upper": None if design is None else design.discordance_upper,
            "normal_rule_raw_shots": None if design is None else design.raw_shots,
            "confirmatory_shots": None if design is None else design.rounded_shots,
            "power_estimate": None if power is None else power.estimate,
            "power_lower_bound": None if power is None else power.lower_bound,
            "passed": bool(passed),
        }
        rows.append(row)
        if selected is None and passed:
            selected = row
    result = {
        "schema": PILOT_SELECTION_SCHEMA,
        "experiment_id": summary["experiment_id"],
        "selection_used_signed_difference": False,
        "cells": rows,
        "selected": selected,
        "status": "selected" if selected is not None else "confirmation-infeasible",
    }
    result["selection_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    _validate_pilot_selection_artifact(result)
    return result


def construct_confirmatory_draft_from_pilot(
    draft: Mapping[str, Any],
    *,
    pilot_manifest: Mapping[str, Any],
    pilot_protocol_path: Path,
    pilot_input_directory: Path,
) -> dict[str, Any]:
    """Derive every adaptive confirmatory literal from verified pilot ledgers.

    This is the only supported route for filling the first-round confirmatory
    template.  It reruns the deterministic pilot verification through
    :func:`load_verified_summary`, applies the blinded fixed-order selector,
    and embeds the complete unsigned selection log into the draft.
    """

    if draft.get("protocol_kind") != "first_round_confirmatory":
        raise ValueError("confirmatory construction requires the first-round template")
    pilot_protocol_path = pilot_protocol_path.resolve()
    pilot_input_directory = pilot_input_directory.resolve()
    normalized_pilot = normalize_protocol(pilot_manifest)
    if normalized_pilot.get("phase") != "pilot":
        raise ValueError("--pilot-protocol must be a frozen pilot protocol")
    summary = load_verified_summary(
        manifest=normalized_pilot,
        input_directory=pilot_input_directory,
        scientific=True,
    )
    pilot_analysis = analyze_summary(summary, manifest=normalized_pilot)
    selection_log = pilot_analysis.get("blinded_selection")
    if not isinstance(selection_log, Mapping):
        raise ValueError("verified pilot analysis did not produce a selection log")
    selected = selection_log.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("pilot did not identify a feasible confirmatory cell")
    cell_id = selected.get("cell_id")
    matches = [cell for cell in normalized_pilot["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ValueError("selected pilot cell is not unique in the frozen manifest")
    cell = matches[0]
    n_confirm = selected.get("confirmatory_shots")
    if (
        isinstance(n_confirm, bool)
        or not isinstance(n_confirm, int)
        or n_confirm <= 0
        or n_confirm % 10_000
    ):
        raise ValueError("pilot selection produced an invalid confirmatory shot count")

    result = copy.deepcopy(dict(draft))
    result_roots = result.get("sampler_seed_roots")
    pilot_roots = normalized_pilot.get("sampler_seed_roots")
    if not isinstance(result_roots, dict) or not isinstance(pilot_roots, Mapping):
        raise ValueError("pilot and confirmatory sampler_seed_roots must be objects")
    pilot_seed_root = pilot_roots.get("pilot")
    if not is_lowercase_hex(pilot_seed_root, length=64):
        raise ValueError("frozen pilot has an invalid pilot seed root")
    result_roots["pilot"] = pilot_seed_root
    nested_cell = {
        "cell_id": cell["cell_id"],
        "d": cell["d"],
        "patches": cell["patches"],
        "yokes": cell["yokes"],
        "rounds": cell["r"],
        "si1000_p": cell["p"],
    }
    selection = result["selection"]
    selection.update(
        {
            "selected_cell_id": cell_id,
            "selected_cell": nested_cell,
            "gate_inputs": {
                "activation_fraction": selected["activation_fraction"],
                "u0_direct_failures": selected["u0_failures"],
                "discordant_pairs": selected["discordant_pairs"],
                "integrity_checks_passed": selected["integrity_checks_passed"],
                "resource_gate_passed": True,
            },
            "p_u0_design": selected["p_u0_design"],
            "delta_noninferiority": selected["delta_noninferiority"],
            "discordance_clopper_pearson_upper": selected["discordance_upper"],
            "normal_rule_raw_shots": selected["normal_rule_raw_shots"],
            "power_verified_confirmatory_shots": n_confirm,
            "n_confirm": n_confirm,
        }
    )
    result["expected_shots"]["confirmatory_holdout"] = n_confirm
    result["batch_schedules"]["confirmatory_holdout"] = {
        "batch_id_start": 0,
        "batch_id_end_inclusive": n_confirm // 10_000 - 1,
        "shots_per_batch": 10_000,
    }
    result["workload_protocol"]["selected_cell_shots"] = n_confirm
    result["generator_contract"]["selected_cell_metadata"] = nested_cell
    result["generator_contract"]["selected_cell_circuit_sha256"] = cell[
        "circuit_sha256"
    ]
    result["dem"]["selected_cell_dem_sha256"] = cell["dem_sha256"]
    result["dem"]["selected_cell_compiled_graph_fingerprint"] = cell[
        "graph_fingerprint"
    ]

    provenance = result["pilot_provenance"]
    provenance.update(
        {
            "pilot_protocol_path": str(pilot_protocol_path),
            "pilot_protocol_experiment_id": normalized_pilot["experiment_id"],
            "pilot_protocol_sha256": _sha256_file(pilot_protocol_path),
            "raw_pilot_result_sha256": _pilot_result_digest(pilot_input_directory),
            "pilot_analysis_sha256": pilot_analysis["analysis_sha256"],
            "selection_log_sha256": selection_log["selection_sha256"],
            "selection_did_not_read_signed_difference": True,
            "selection_log": copy.deepcopy(dict(selection_log)),
            "pilot_source_hashes": copy.deepcopy(normalized_pilot["source_hashes"]),
        }
    )
    return result


def analyze_summary(
    summary: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Builds and authenticates the deterministic analysis artifact.

    Pilot analysis additionally runs the frozen blinded selector; confirmatory
    thresholds are read only from the normalized protocol.
    """

    manifest = normalize_protocol(manifest)
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("unsupported paired summary schema")
    if summary.get("experiment_id") != manifest.get("experiment_id"):
        raise ValueError("summary and protocol experiment IDs differ")
    phase = summary.get("phase")
    if phase not in {"pilot", "confirm", "target", "smoke"}:
        raise ValueError("summary has invalid phase")
    cells = summary.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("summary must contain at least one cell")
    config = _analysis_config(manifest)
    delta_ni = None
    if phase == "confirm":
        delta_ni = float(config["delta_noninferiority"])
        if not math.isfinite(delta_ni) or delta_ni <= 0:
            raise ValueError("confirmatory delta_noninferiority must be positive")
    result = {
        "schema": ANALYSIS_SCHEMA,
        "experiment_id": summary["experiment_id"],
        "phase": phase,
        "claim_bearing": bool(
            manifest.get("claim_bearing") is True and phase in {"confirm", "target"}
        ),
        "collection_scope": summary.get("collection_scope"),
        "accuracy_claim_scope": (
            "descriptive-only-not-powered"
            if phase == "target"
            else "paired-confirmatory"
            if phase == "confirm"
            else "non-confirmatory"
        ),
        "cells": [
            analyze_cell(cell, manifest=manifest, delta_noninferiority=delta_ni)
            for cell in cells
        ],
    }
    if phase == "pilot":
        result["blinded_selection"] = select_pilot_cell(summary, manifest=manifest)
    result["analysis_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    validate_generated_analysis_artifact(result)
    return result


# Whole-corpus authentication and human-readable output.


def load_verified_summary(
    *, manifest: Mapping[str, Any], input_directory: Path, scientific: bool = True
) -> dict[str, Any]:
    """Authenticate a complete collection and regenerate it without writes."""

    manifest = normalize_protocol(manifest)
    input_candidate = Path(input_directory).absolute()
    if input_candidate.is_symlink() or not input_candidate.is_dir():
        raise ValueError("collection input must be a regular non-symlink directory")
    input_directory = input_candidate.resolve()
    identity_path = input_directory / "experiment.json"
    protocol_path = input_directory / "protocol.json"
    summary_path = input_directory / "summary.json"
    batch_root = input_directory / "batches"
    expected_top_level = {
        identity_path,
        protocol_path,
        summary_path,
        batch_root,
    }
    actual_top_level = set(input_directory.iterdir())
    if actual_top_level != expected_top_level:
        missing = sorted(expected_top_level - actual_top_level)
        extras = sorted(actual_top_level - expected_top_level)
        raise ValueError(
            "collection artifact set differs from the exact output contract: "
            f"missing={missing[:8]}, unexpected={extras[:8]}"
        )
    if batch_root.is_symlink() or not batch_root.is_dir():
        raise ValueError("collection batches must be a regular non-symlink directory")

    identity = load_json_strict(identity_path, description="collection experiment.json")
    if not isinstance(identity, Mapping) or set(identity) != {
        "schema",
        "experiment_id",
        "phase",
    }:
        raise ValueError("experiment.json has incorrect fields")
    phase = str(identity.get("phase"))
    if phase not in {"pilot", "confirm", "target", "smoke"}:
        raise ValueError("experiment.json has an invalid phase")
    if (
        load_json_strict(protocol_path, description="collection protocol.json")
        != manifest
    ):
        raise ValueError("result protocol.json differs from the supplied manifest")
    recorded = load_json_strict(summary_path, description="collection summary.json")

    experiment_id = validate_experiment_protocol(
        manifest,
        phase=phase,
        scientific=scientific,
        processes=int(manifest["processes"]),
    )
    if identity != {
        "schema": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "phase": phase,
    }:
        raise ValueError("experiment.json does not match the supplied protocol")

    cells = _phase_cells(manifest, phase)
    schedules = _phase_schedules(manifest, phase, cells)
    split = _protocol_split(manifest, phase)
    decoder = manifest["decoder"]
    dem_options = manifest["dem_options"]
    expected_paths = {
        _ledger_path(
            input_directory,
            cell_id=cell["cell_id"],
            batch_id=batch.batch_id,
        )
        for cell in cells
        for batch in schedules[cell["cell_id"]]
    }
    expected_cell_directories = {path.parent for path in expected_paths}
    actual_cell_directories: set[Path] = set()
    actual_paths: set[Path] = set()
    for cell_directory in batch_root.iterdir():
        if cell_directory.is_symlink() or not cell_directory.is_dir():
            raise ValueError(
                "collection batches contain an unsafe non-directory entry: "
                f"{cell_directory}"
            )
        actual_cell_directories.add(cell_directory)
        for artifact in cell_directory.iterdir():
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError(
                    "collection batches contain an unsafe non-regular artifact: "
                    f"{artifact}"
                )
            actual_paths.add(artifact)
    if actual_cell_directories != expected_cell_directories:
        missing = sorted(expected_cell_directories - actual_cell_directories)
        extras = sorted(actual_cell_directories - expected_cell_directories)
        raise ValueError(
            "batch cell directory set differs from the frozen schedule: "
            f"missing={missing[:8]}, unexpected={extras[:8]}"
        )
    missing = sorted(expected_paths - actual_paths)
    extras = sorted(actual_paths - expected_paths)
    if missing or extras:
        raise ValueError(
            "batch ledger set differs from the frozen schedule: "
            f"missing={missing[:8]}, unexpected={extras[:8]}"
        )

    rows: list[Mapping[str, Any]] = []
    for cell in cells:
        prepared = prepare_cell(
            cell,
            decoder_config=decoder,
            dem_options=dem_options,
            verify_hashes=scientific,
        )
        for batch in schedules[cell["cell_id"]]:
            path = _ledger_path(
                input_directory,
                cell_id=cell["cell_id"],
                batch_id=batch.batch_id,
            )
            row = load_json_strict(path, description="paired batch ledger")
            _validate_ledger_row(
                row,
                experiment_id=experiment_id,
                phase=phase,
                cell=cell,
                batch=batch,
                seed_root=manifest["sampler_seed_roots"][split],
                expected_provenance=prepared.provenance,
                replay_policy=manifest["replay_policy"],
            )
            rows.append(row)
    recomputed = summarize_ledgers(
        rows,
        experiment_id=experiment_id,
        phase=phase,
        replay_policy=manifest["replay_policy"],
    )
    if recorded != recomputed:
        raise ValueError("summary.json does not exactly reconcile with batch ledgers")
    if scientific:
        _verify_scientific_regeneration_without_writes(
            manifest,
            phase=phase,
            recorded_rows=rows,
            processes=int(manifest["processes"]),
        )
    return _VerifiedSummary(
        recomputed,
        provenance=_VerificationProvenance(
            experiment_id=experiment_id,
            phase=phase,
            deterministic_regeneration_passed=scientific,
            summary_sha256=hashlib.sha256(canonical_json_bytes(recomputed)).hexdigest(),
            manifest_sha256=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        ),
    )


def render_markdown(analysis: Mapping[str, Any]) -> str:
    """Renders a compact deterministic human summary of an analysis artifact."""

    lines = [
        "# ProMatch L1 analysis",
        "",
        f"Experiment: `{analysis['experiment_id']}`",
        "",
        f"Phase: `{analysis['phase']}`",
        "",
        f"Claim-bearing: `{analysis.get('claim_bearing', False)}`",
        "",
    ]
    for cell in analysis["cells"]:
        lines.extend(
            [
                f"## {cell['cell_id']}",
                "",
                f"- Paired shots: {cell['shots']}",
                f"- U0 failure rate: {cell['u0_failure_rate']:.8g}",
                f"- PU failure rate: {cell['pu_failure_rate']:.8g}",
                f"- PU-U0 risk difference: {cell['delta_pu_minus_u0']:.8g}",
                f"- One-sided Tango upper bound: {cell['tango_upper_one_sided']:.8g}",
                f"- Activation fraction: {cell['activation_fraction']:.8g}",
                f"- Rollback fraction: {cell['rollback_fraction']:.8g}",
                f"- Workload ratio: {cell['workload_ratio']}",
                f"- Workload upper bound: {cell['workload_ratio_upper_one_sided']}",
                f"- Accuracy non-inferiority passed: {cell['noninferiority_passed']}",
                f"- Ordered accuracy superiority passed: {cell['ordered_superiority_passed']}",
                "",
            ]
        )
    if "blinded_selection" in analysis:
        selection = analysis["blinded_selection"]
        selected = selection["selected"]
        lines.extend(
            [
                "## Blinded pilot selection",
                "",
                "The selector did not compute or emit the signed paired difference.",
                "",
                f"Selected cell: `{None if selected is None else selected['cell_id']}`",
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_SCHEMA",
    "PILOT_SELECTION_SCHEMA",
    "analyze_cell",
    "analyze_summary",
    "construct_confirmatory_draft_from_pilot",
    "load_verified_summary",
    "render_markdown",
    "select_pilot_cell",
    "validate_generated_analysis_artifact",
]
