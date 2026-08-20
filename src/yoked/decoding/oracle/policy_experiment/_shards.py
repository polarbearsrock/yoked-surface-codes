"""Worker shard collection, atomic installation, and whole-shard verification.

This slice samples one immutable worker range entirely in memory, runs the
frozen numerical preflight on the real graph, separates nondeterministic
timing from the bit-exact scientific ledgers, installs completed shards
atomically from TMPDIR, and authenticates whole shards before any resume.
It inherits the package isolation contract (see ``__init__``): per-shot
policy logic is reached only through the :mod:`._ledger` adapter, and actual
observables never influence identity, ordering, or selection.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import gzip
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from yoked.decoding._promatch_experiment import (
    PreparedCell,
    configure_single_thread_runtime,
)
from yoked.decoding._promatch_stats import canonical_json_bytes, digest_array
from yoked.decoding.oracle.full_graph import (
    FullGraphOracle,
    OracleTolerance,
    classify_cost_excess,
)

from yoked.decoding.oracle.policy_experiment._identity import (
    ARM_IDS,
    SHARD_SCHEMA,
    THREAD_ENVIRONMENT,
    WorkerSpec,
    _SIDECARS,
    _sha256,
    _strict_json_load,
    derive_policy_worker_seed,
)
from yoked.decoding.oracle.policy_experiment._ledger import (
    NormalizedPolicyShot,
    _COLLECTOR_OWNED_FIELDS,
    _artifact_metadata,
    _audit_policy_shot,
    _require_float_hex_companions,
    _separate_nondeterministic_timing,
    _validate_context_union_ledger,
    _validate_support_difference_ledger,
    canonical_jsonl,
    deterministic_gzip,
    forbid_ground_truth_keys,
)


def _peak_rss_bytes() -> int:
    """Returns Linux ``ru_maxrss`` in bytes for authenticated telemetry."""

    # This experiment is frozen to the Linux execution environment recorded in
    # the protocol.  Linux reports ru_maxrss in KiB (unlike macOS, which reports
    # bytes), so keep the conversion and its source explicit in timing.json.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _type7_quantiles_ns(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("timing quantiles require at least one physical shot")
    result: dict[str, Any] = {}
    for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        value = float(
            np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear")
        )
        result[name] = value
        result[f"{name}_hex"] = value.hex()
    result["max"] = max(values)
    return result


def _shot_performance_telemetry(
    audited: NormalizedPolicyShot,
    *,
    worker_shot_index: int,
    global_shot_id: int,
    audit_wall_ns: int,
) -> dict[str, Any]:
    """Extracts concrete, non-decision timing/call telemetry from one audit."""

    shot = audited.shot
    reported = shot.get("timing_telemetry", {})
    if not isinstance(reported, Mapping):
        raise ValueError("policy-core timing_telemetry must be an object")
    candidate_rows = (*audited.proposals, *audited.counterfactuals)

    def integer(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"policy-core performance counter {name} is invalid")
        return value

    def optional_integer(name: str, value: Any) -> int:
        return integer(name, 0 if value is None else value)

    return {
        "worker_shot_index": worker_shot_index,
        "global_shot_id": global_shot_id,
        "audit_wall_ns": integer("audit_wall_ns", audit_wall_ns),
        "oracle_cache_hits": optional_integer(
            "oracle_cache_hits", shot.get("oracle_cache_hits")
        ),
        "oracle_cache_misses": optional_integer(
            "oracle_cache_misses", shot.get("oracle_cache_misses")
        ),
        "oracle_evaluation_call_count": optional_integer(
            "oracle_evaluation_call_count", shot.get("oracle_evaluation_call_count")
        ),
        "full_mwpm_cache_miss_count": optional_integer(
            "total_full_mwpm_cache_miss_count_all_arms",
            shot.get("total_full_mwpm_cache_miss_count_all_arms"),
        ),
        "matched_active_pair_backend_call_count": optional_integer(
            "matched_active_pair_backend_call_count",
            shot.get("matched_active_pair_backend_call_count"),
        ),
        "matched_active_pair_backend_wall_ns": optional_integer(
            "matched_active_pair_backend_wall_ns",
            shot.get("matched_active_pair_backend_wall_ns"),
        ),
        "counterfactual_wall_ns": optional_integer(
            "counterfactual_wall_ns", reported.get("counterfactual_wall_ns")
        ),
        "support_classification_wall_ns": optional_integer(
            "support_classification_wall_ns",
            reported.get("support_classification_wall_ns"),
        ),
        "candidate_enumeration_wall_ns": sum(
            optional_integer(
                "candidate_enumeration_wall_ns",
                row.get("candidate_enumeration_wall_ns"),
            )
            for row in candidate_rows
        ),
        "stage3_enumeration_wall_ns": optional_integer(
            "stage3_specific_wall_ns", reported.get("stage3_specific_wall_ns")
        ),
    }


def _real_graph_numerical_preflight(
    graph: Any, syndrome: np.ndarray, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Runs the frozen numerical repeatability subset on one real worker state."""

    tolerance = OracleTolerance(**config["oracle"]["tolerance"])
    first_oracle = FullGraphOracle(graph, tolerance=tolerance)
    first = first_oracle.decode_state(syndrome, use_cache=False)
    repeated = first_oracle.decode_state(syndrome, use_cache=False)
    if first != repeated:
        raise AssertionError("uncached full-graph oracle decode is not repeatable")

    cached_oracle = FullGraphOracle(graph, tolerance=tolerance)
    uncached = cached_oracle.decode_state(syndrome, use_cache=False)
    cached_seed = cached_oracle.decode_state(syndrome, use_cache=True)
    cached_repeat = cached_oracle.decode_state(syndrome, use_cache=True)
    if uncached != cached_seed or cached_seed != cached_repeat:
        raise AssertionError("cached and uncached full-graph oracle results differ")
    if cached_oracle.cache_stats.hits != 1:
        raise AssertionError(
            "oracle repeatability preflight did not exercise a cache hit"
        )

    weights = [graph.edges[edge_id].weight for edge_id in first.support_edge_ids]
    fsum_weight = math.fsum(weights)
    if fsum_weight != first.support_weight:
        raise AssertionError("oracle support weight is not the canonical math.fsum")
    precision = config["oracle"]["decimal_precision_digits"]
    with localcontext() as context:
        context.prec = precision
        decimal_weight = sum(
            (Decimal.from_float(value) for value in weights), Decimal(0)
        )
        decimal_fsum = Decimal.from_float(fsum_weight)
        rounding_bound = Decimal.from_float(
            math.ulp(fsum_weight) if fsum_weight != 0 else math.ulp(0.0)
        )
        if abs(decimal_weight - decimal_fsum) > rounding_bound:
            raise AssertionError(
                "math.fsum differs from the 4096-digit Decimal reference"
            )

    grid_observations = 0
    for relative in config["oracle"]["tolerance_sensitivity_relative"]:
        grid_tolerance = OracleTolerance(absolute=tolerance.absolute, relative=relative)
        grid_solution = FullGraphOracle(graph, tolerance=grid_tolerance).decode_state(
            syndrome, use_cache=False
        )
        if (
            grid_solution.prediction != first.prediction
            or grid_solution.support_edge_ids != first.support_edge_ids
            or grid_solution.support_weight != first.support_weight
            or grid_solution.backend_weight != first.backend_weight
        ):
            raise AssertionError("oracle support changed across the tolerance grid")
        excess = grid_solution.support_weight - grid_solution.backend_weight
        tau = grid_tolerance.tau_k(
            base_weight=grid_solution.backend_weight,
            composite_weight=grid_solution.support_weight,
        )
        classification = classify_cost_excess(cost_excess=excess, tau_k=tau)
        expected = (
            "positive-cost-excess"
            if excess > tau
            else "numeric-accounting-anomaly"
            if excess < -tau
            else "numerically-cost-compatible"
        )
        if classification.value != expected:
            raise AssertionError("oracle tolerance-grid classification is inconsistent")
        grid_observations += 1

    return {
        "schema": "promatch-l1-policy-audit-numerical-preflight-v1",
        "real_graph": True,
        "syndrome_sha256": _sha256(np.asarray(syndrome, dtype=np.uint8).tobytes()),
        "decimal_precision_digits": precision,
        "tolerance_grid_observations": grid_observations,
        "backend_support_fsum_passed": True,
        "decimal_reference_passed": True,
        "uncached_repeatability_passed": True,
        "cached_uncached_repeatability_passed": True,
        "tolerance_grid_passed": True,
    }


def _worker_gate_evidence(
    *,
    spec: WorkerSpec,
    numerical_preflight: Mapping[str, Any],
    shots: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from yoked.decoding.oracle.policy_analysis import COLLECTOR_GATE_CHECKS

    oracle_calls = sum(int(row.get("oracle_evaluation_call_count", 0)) for row in shots)
    observations = {
        "scalar-batch-u0": spec.shots,
        "shadow-frozen-v3-equivalence": spec.shots,
        "backend-support-fsum": 1,
        "decimal-4096": 1,
        "uncached-repeatability": 1,
        "tolerance-grid": int(numerical_preflight["tolerance_grid_observations"]),
        "actual-observable-invariance": spec.shots,
        "veto-state-frame-prefix-invariance": max(spec.shots, len(counterfactuals)),
        "matching-pair-and-support-reconciliation": max(spec.shots, oracle_calls),
        "cached-uncached-oracle-repeatability": 1,
        "execution-and-source-provenance": 1,
    }
    expected = {check for checks in COLLECTOR_GATE_CHECKS.values() for check in checks}
    if set(observations) != expected or any(
        value <= 0 for value in observations.values()
    ):
        raise AssertionError("worker collector gate evidence is incomplete")
    return {
        "schema": "promatch-l1-policy-audit-worker-gate-evidence-v1",
        "real_graph": numerical_preflight.get("real_graph") is True,
        "checks": {
            check: {"status": "passed", "observations": observations[check]}
            for check in sorted(expected)
        },
        "numerical_preflight": dict(numerical_preflight),
    }


def _row_identity(
    *,
    config: Mapping[str, Any],
    spec: WorkerSpec,
    worker_shot_index: int,
    stim_seed: int,
    physical_input_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "cell_id": config["cell"]["cell_id"],
        "worker_id": spec.worker_id,
        "worker_shot_index": worker_shot_index,
        "global_shot_id": spec.shot_start + worker_shot_index,
        "stim_seed": stim_seed,
        "physical_input_sha256": physical_input_sha256,
        "detector_input_sha256": physical_input_sha256,
        "circuit_sha256": config["cell"]["circuit_sha256"],
    }


def _merge_core_row(
    *, schema: str, identity: Mapping[str, Any], core: Mapping[str, Any]
) -> dict[str, Any]:
    overlap = _COLLECTOR_OWNED_FIELDS.intersection(core)
    if overlap:
        raise ValueError(
            "policy core attempted to override collector-owned fields: "
            f"{sorted(overlap)}"
        )
    return {"schema": schema, **identity, **core}


def collect_policy_worker_shard(
    prepared: PreparedCell,
    *,
    config: Mapping[str, Any],
    mode: str,
    spec: WorkerSpec,
    audit_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Samples and audits one immutable worker range entirely in memory."""

    configure_single_thread_runtime()
    seed = derive_policy_worker_seed(
        seed_root=config["sampling"]["seed_roots"][mode],
        experiment_id=config["experiment_id"],
        cell_id=config["cell"]["cell_id"],
        worker_id=spec.worker_id,
    )
    sampler = prepared.circuit.compile_detector_sampler(seed=seed)
    sample_start = time.perf_counter_ns()
    dets, obs = sampler.sample(
        shots=spec.shots, separate_observables=True, bit_packed=True
    )
    sample_ns = time.perf_counter_ns() - sample_start
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    dets.setflags(write=False)
    obs.setflags(write=False)
    det_digest = digest_array(dets)
    obs_digest = digest_array(obs)
    graph = prepared.compiled_pu.graph
    u0_start = time.perf_counter_ns()
    u0_batch = np.asarray(
        graph.matcher.decode_batch(
            dets, bit_packed_shots=True, bit_packed_predictions=True
        ),
        dtype=np.uint8,
    )
    u0_batch_decode_ns = time.perf_counter_ns() - u0_start
    unpacked = np.unpackbits(
        dets, axis=1, bitorder="little", count=graph.num_detectors
    ).astype(np.uint8, copy=False)
    if audit_fn is None:
        numerical_preflight = _real_graph_numerical_preflight(
            graph, unpacked[0], config=config
        )
    else:
        # Unit tests use a deliberately minimal matcher test double.  This
        # evidence is accepted for isolated shard-verifier tests but is
        # explicitly rejected by campaign aggregation.
        numerical_preflight = {
            "schema": "promatch-l1-policy-audit-numerical-preflight-v1",
            "real_graph": False,
            "syndrome_sha256": _sha256(unpacked[0].tobytes()),
            "decimal_precision_digits": config["oracle"]["decimal_precision_digits"],
            "tolerance_grid_observations": len(
                config["oracle"]["tolerance_sensitivity_relative"]
            ),
            "backend_support_fsum_passed": True,
            "decimal_reference_passed": True,
            "uncached_repeatability_passed": True,
            "cached_uncached_repeatability_passed": True,
            "tolerance_grid_passed": True,
        }
    tolerance = OracleTolerance(**config["oracle"]["tolerance"])
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _SIDECARS}
    per_shot_timing: list[dict[str, Any]] = []
    decode_start = time.perf_counter_ns()
    for offset in range(spec.shots):
        audit_start = time.perf_counter_ns()
        audited = _audit_policy_shot(
            graph, unpacked[offset], tolerance=tolerance, audit_fn=audit_fn
        )
        audit_wall_ns = time.perf_counter_ns() - audit_start
        if audited.arm_predictions[ARM_IDS[0]] != bytes(u0_batch[offset]):
            raise AssertionError(
                "scalar policy-core U0 differs from bit-packed batch U0"
            )
        detector_bytes = bytes(dets[offset])
        observable_bytes = bytes(obs[offset])
        # Proposal/casebook identity is detector-only.  Actual observables are
        # posthoc ground truth and must not influence any audit identity or
        # deterministic selection tie-break.
        physical_sha = _sha256(b"promatch-policy-detector-input-v1\0" + detector_bytes)
        identity = _row_identity(
            config=config,
            spec=spec,
            worker_shot_index=offset,
            stim_seed=seed,
            physical_input_sha256=physical_sha,
        )
        per_shot_timing.append(
            _shot_performance_telemetry(
                audited,
                worker_shot_index=offset,
                global_shot_id=identity["global_shot_id"],
                audit_wall_ns=audit_wall_ns,
            )
        )
        prediction_hex = {
            arm_id: audited.arm_predictions[arm_id].hex() for arm_id in ARM_IDS
        }
        failures = {
            arm_id: audited.arm_predictions[arm_id] != observable_bytes
            for arm_id in ARM_IDS
        }
        shot_overlap = _COLLECTOR_OWNED_FIELDS.intersection(audited.shot)
        if shot_overlap:
            raise ValueError(
                "policy core attempted to override collector-owned fields: "
                f"{sorted(shot_overlap)}"
            )
        rows["shots"].append(
            {
                "schema": _SIDECARS[0][1],
                **identity,
                "packed_detectors_hex": detector_bytes.hex(),
                "packed_detector_bits": graph.num_detectors,
                "packed_detectors_sha256": _sha256(detector_bytes),
                "packed_actual_observables_hex": observable_bytes.hex(),
                "packed_actual_observable_bits": graph.num_observables,
                "packed_actual_observables_sha256": _sha256(observable_bytes),
                "arm_predictions_hex": prediction_hex,
                "arm_failures": failures,
                **audited.shot,
            }
        )
        for name, schema in _SIDECARS[1:]:
            for item in getattr(audited, name):
                rows[name].append(
                    _merge_core_row(schema=schema, identity=identity, core=item)
                )
    decode_ns = time.perf_counter_ns() - decode_start
    if digest_array(dets) != det_digest or digest_array(obs) != obs_digest:
        raise AssertionError("a policy arm mutated the shared sampled corpus")

    payloads: dict[str, bytes] = {}
    artifact_rows: dict[str, Any] = {}
    serialization_by_artifact: dict[str, Any] = {}
    core_timing_by_ledger: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _ in _SIDECARS
    }
    serialization_start = time.perf_counter_ns()
    for name, schema in _SIDECARS:
        scientific_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows[name]):
            scientific_row, extracted_timing = _separate_nondeterministic_timing(row)
            if not isinstance(scientific_row, dict):
                raise AssertionError("ledger timing separation changed the row type")
            scientific_rows.append(scientific_row)
            if extracted_timing is not None:
                core_timing_by_ledger[name].append(
                    {
                        "row_index": row_index,
                        "worker_shot_index": row["worker_shot_index"],
                        "global_shot_id": row["global_shot_id"],
                        "timing": extracted_timing,
                    }
                )
        canonical_start = time.perf_counter_ns()
        raw = canonical_jsonl(scientific_rows)
        canonical_ns = time.perf_counter_ns() - canonical_start
        compression_start = time.perf_counter_ns()
        compressed = deterministic_gzip(raw)
        compression_ns = time.perf_counter_ns() - compression_start
        payloads[f"{name}.jsonl.gz"] = compressed
        artifact_rows[f"{name}.jsonl.gz"] = _artifact_metadata(
            compressed=compressed,
            uncompressed=raw,
            rows=len(rows[name]),
            schema=schema,
            path=f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz",
        )
        serialization_by_artifact[f"{name}.jsonl.gz"] = {
            "canonical_jsonl_ns": canonical_ns,
            "gzip_ns": compression_ns,
            "rows": len(scientific_rows),
            "compressed_bytes": len(compressed),
            "uncompressed_bytes": len(raw),
            "compressed_bytes_per_row": {
                "numerator": len(compressed),
                "denominator": len(scientific_rows),
            },
            "uncompressed_bytes_per_row": {
                "numerator": len(raw),
                "denominator": len(scientific_rows),
            },
            "compressed_bytes_per_physical_shot": {
                "numerator": len(compressed),
                "denominator": spec.shots,
            },
            "uncompressed_bytes_per_physical_shot": {
                "numerator": len(raw),
                "denominator": spec.shots,
            },
        }
    serialization_ns = time.perf_counter_ns() - serialization_start
    audit_values = [row["audit_wall_ns"] for row in per_shot_timing]
    timing = {
        "schema": "promatch-l1-policy-audit-timing-v2",
        "sampling_ns": sample_ns,
        "u0_batch_decode_ns": u0_batch_decode_ns,
        "shot_audit_loop_ns": decode_ns,
        "ledger_serialization_ns": serialization_ns,
        "shot_audit_wall_ns_quantiles": _type7_quantiles_ns(audit_values),
        "per_shot": per_shot_timing,
        "serialization_by_artifact": serialization_by_artifact,
        "core_timing_by_ledger": core_timing_by_ledger,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib",
        "per_arm_decode_wall_ns_available": False,
        "per_arm_decode_wall_ns_gap": (
            "audit_policy_shot currently returns all five arms atomically; "
            "per-shot audit and backend/enumeration timings are recorded, but "
            "wall time cannot yet be attributed to individual arms"
        ),
        "worker_id": spec.worker_id,
        "scientifically_deterministic": False,
        "excluded_from_bit_exact_ledger_contract": True,
    }
    payloads["timing.json"] = canonical_json_bytes(timing) + b"\n"
    shard = {
        "schema": SHARD_SCHEMA,
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "cell_id": config["cell"]["cell_id"],
        "worker": spec.to_json(),
        "stim_seed": seed,
        "detectors": {
            "sha256": det_digest.sha256,
            "shape": list(det_digest.shape),
            "dtype": det_digest.dtype,
        },
        "observables": {
            "sha256": obs_digest.sha256,
            "shape": list(obs_digest.shape),
            "dtype": obs_digest.dtype,
        },
        "artifacts": artifact_rows,
        "timing_sha256": _sha256(payloads["timing.json"]),
        "timing_path": f"shards/worker-{spec.worker_id:02d}/timing.json",
        "native_thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "tail_censor_attestation": _worker_tail_censor_attestation(
            rows["counterfactuals"]
        ),
        "collector_gate_evidence": _worker_gate_evidence(
            spec=spec,
            numerical_preflight=numerical_preflight,
            shots=rows["shots"],
            counterfactuals=rows["counterfactuals"],
        ),
        "nondeterministic_telemetry_paths": [
            f"shards/worker-{spec.worker_id:02d}/timing.json"
        ],
        "bit_exact_regeneration_paths": [
            f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz"
            for name, _ in _SIDECARS
        ],
    }
    payloads["shard.json"] = canonical_json_bytes(shard) + b"\n"
    return shard, payloads


def _worker_tail_censor_attestation(
    counterfactual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    censored_states: set[str] = set()
    seen: set[tuple[str, bytes]] = set()
    repeated = 0
    for row in counterfactual_rows:
        state = row.get(
            "original_state_sha256", row.get("complete_pre_state_fingerprint")
        )
        if not isinstance(state, str) or not state:
            raise ValueError(
                "counterfactual row lacks complete original-state identity"
            )
        if row.get("censored") is True:
            censored_states.add(state)
        if row.get("veto_budget") is not None:
            raise ValueError("B1 counterfactual row contains a forbidden veto budget")
        signature = row.get("proposal_signature", row.get("proposal_sha256"))
        if signature is None:
            raise ValueError("counterfactual row lacks a proposal signature")
        # Proposal signatures are structured JSON values (normally arrays),
        # while canonical_json_bytes intentionally accepts only a top-level
        # object.  Wrap the value in a version-stable object before comparing
        # exact bytes within one original state.
        key = state, canonical_json_bytes({"proposal_signature": signature})
        if key in seen:
            repeated += 1
        seen.add(key)
    return {
        "uncapped_counterfactuals": True,
        "censored_states": len(censored_states),
        "repeated_same_state_proposal_signatures": repeated,
        "output_truncations": 0,
    }


def _shard_dir(out: Path, worker_id: int) -> Path:
    return out / "shards" / f"worker-{worker_id:02d}"


def install_worker_shard(
    out: Path,
    *,
    shard: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    config: Mapping[str, Any],
    mode: str,
    spec: WorkerSpec,
) -> Path:
    """Validates then atomically installs one completed shard from TMPDIR."""

    # Shard authentication is resolved through the package namespace at call
    # time, exactly like the pre-package module-global lookup.
    from yoked.decoding.oracle import policy_experiment as _package

    scratch = os.environ.get("TMPDIR")
    if not scratch:
        raise RuntimeError("TMPDIR must be set for shard installation")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if os.stat(scratch).st_dev != os.stat(out).st_dev:
        raise RuntimeError(
            "TMPDIR and output must share a filesystem for atomic shard install"
        )
    worker_id = int(shard["worker"]["worker_id"])
    final = _shard_dir(out, worker_id)
    if final.exists():
        raise FileExistsError(f"worker shard already exists: {final}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f"promatch-policy-worker-{worker_id:02d}-", dir=scratch)
    )
    try:
        if set(payloads) != {
            "shots.jsonl.gz",
            "proposals.jsonl.gz",
            "counterfactuals.jsonl.gz",
            "domains.jsonl.gz",
            "timing.json",
            "shard.json",
        }:
            raise ValueError("worker payload set is incomplete")
        for name, data in payloads.items():
            path = temporary / name
            with path.open("xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        if int(shard["worker"]["worker_id"]) != spec.worker_id:
            raise ValueError("worker result ID differs from its scheduled shard")
        verified = _package.verify_worker_shard(
            temporary, config=config, mode=mode, spec=spec
        )
        if verified != dict(shard):
            raise ValueError(
                "worker shard manifest differs from returned worker metadata"
            )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        # Authenticate the installed namespace, not only the staging directory.
        # This keeps worker-side installation parallel while closing the gap
        # between the bytes verified before rename and the path trusted by the
        # parent campaign manifest.
        installed = _package.verify_worker_shard(
            final, config=config, mode=mode, spec=spec
        )
        if installed != verified:
            raise ValueError(
                "installed worker shard differs from its staged verification"
            )
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _verify_shard_manifest_identity(
    shard: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    mode: str,
    spec: WorkerSpec,
) -> None:
    """Authenticates the shard manifest's exact fields and identity."""

    shard_fields = {
        "schema",
        "experiment_id",
        "mode",
        "cell_id",
        "worker",
        "stim_seed",
        "detectors",
        "observables",
        "artifacts",
        "timing_sha256",
        "timing_path",
        "native_thread_environment",
        "tail_censor_attestation",
        "nondeterministic_telemetry_paths",
        "bit_exact_regeneration_paths",
        "collector_gate_evidence",
    }
    if set(shard) != shard_fields:
        raise ValueError("worker shard manifest fields are not exact")
    if shard.get("schema") != SHARD_SCHEMA:
        raise ValueError("worker shard schema mismatch")
    if shard.get("experiment_id") != config["experiment_id"]:
        raise ValueError("worker shard experiment identity mismatch")
    if shard.get("mode") != mode:
        raise ValueError("worker shard mode mismatch")
    if shard.get("cell_id") != config["cell"]["cell_id"]:
        raise ValueError("worker shard cell identity mismatch")
    if shard.get("worker") != spec.to_json():
        raise ValueError("worker shard schedule spec mismatch")


def _verify_sidecar_row_identity(
    row: Mapping[str, Any],
    *,
    name: str,
    config: Mapping[str, Any],
    spec: WorkerSpec,
    expected_seed: int,
) -> None:
    """Authenticates one ledger row's collector-owned identity fields."""

    schema = dict(_SIDECARS)[name]
    if row.get("schema") != schema:
        raise ValueError(f"sidecar row schema mismatch for {name}")
    if row.get("experiment_id") != config["experiment_id"]:
        raise ValueError(f"sidecar row experiment identity mismatch for {name}")
    if row.get("cell_id") != config["cell"]["cell_id"]:
        raise ValueError(f"sidecar row cell identity mismatch for {name}")
    if row.get("worker_id") != spec.worker_id:
        raise ValueError(f"sidecar row worker identity mismatch for {name}")
    if row.get("stim_seed") != expected_seed:
        raise ValueError(f"sidecar row stim seed mismatch for {name}")
    if row.get("circuit_sha256") != config["cell"]["circuit_sha256"]:
        raise ValueError(f"sidecar row circuit digest mismatch for {name}")
    worker_index = row.get("worker_shot_index")
    if isinstance(worker_index, bool) or not isinstance(worker_index, int):
        raise ValueError(f"sidecar worker_shot_index is not an integer for {name}")
    if worker_index < 0 or worker_index >= spec.shots:
        raise ValueError(
            f"sidecar worker_shot_index is outside the worker range for {name}"
        )
    if row.get("global_shot_id") != spec.shot_start + worker_index:
        raise ValueError(
            f"sidecar global_shot_id does not match its worker offset for {name}"
        )
    physical = row.get("physical_input_sha256")
    if not isinstance(physical, str) or re.fullmatch(r"[0-9a-f]{64}", physical) is None:
        raise ValueError(
            f"sidecar physical_input_sha256 is not a SHA-256 digest for {name}"
        )
    if row.get("detector_input_sha256") != physical:
        raise ValueError(
            f"sidecar detector_input_sha256 differs from "
            f"physical_input_sha256 for {name}"
        )


def _verify_shot_row_reconciliation(
    row: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[bytes, bytes]:
    """Recomputes one shot row's packed input digests and arm reconciliation."""

    detector_width = (int(config["cell"]["num_detectors"]) + 7) // 8
    observable_width = (int(config["cell"]["num_observables"]) + 7) // 8
    try:
        detector_bytes = bytes.fromhex(row["packed_detectors_hex"])
        observable_bytes = bytes.fromhex(row["packed_actual_observables_hex"])
    except (KeyError, TypeError, ValueError) as ex:
        raise ValueError("shot row has invalid packed detector/observable data") from ex
    if len(detector_bytes) != detector_width:
        raise ValueError("shot packed detector width mismatch")
    if len(observable_bytes) != observable_width:
        raise ValueError("shot packed observable width mismatch")
    if row.get("packed_detector_bits") != config["cell"]["num_detectors"]:
        raise ValueError("shot packed detector bit count mismatch")
    if row.get("packed_actual_observable_bits") != config["cell"]["num_observables"]:
        raise ValueError("shot packed observable bit count mismatch")
    if row.get("packed_detectors_sha256") != _sha256(detector_bytes):
        raise ValueError("shot packed detector digest mismatch")
    if row.get("packed_actual_observables_sha256") != _sha256(observable_bytes):
        raise ValueError("shot packed observable digest mismatch")
    if row.get("physical_input_sha256") != _sha256(
        b"promatch-policy-detector-input-v1\0" + detector_bytes
    ):
        raise ValueError("shot detector-input identity digest mismatch")
    predictions = row.get("arm_predictions_hex")
    failures = row.get("arm_failures")
    if not isinstance(predictions, Mapping) or set(predictions) != set(ARM_IDS):
        raise ValueError("shot prediction arm set differs from frozen arms")
    if not isinstance(failures, Mapping) or set(failures) != set(ARM_IDS):
        raise ValueError("shot failure arm set differs from frozen arms")
    for arm_id in ARM_IDS:
        try:
            prediction = bytes.fromhex(predictions[arm_id])
        except (TypeError, ValueError) as ex:
            raise ValueError("shot contains an invalid arm prediction") from ex
        if len(prediction) != observable_width:
            raise ValueError("shot arm prediction width mismatch")
        if not isinstance(failures[arm_id], bool):
            raise ValueError("shot arm failure flag is not boolean")
        if failures[arm_id] != (prediction != observable_bytes):
            raise ValueError("shot arm failure does not reconcile with its prediction")
    return detector_bytes, observable_bytes


def _verify_gate_evidence(
    shard: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    spec: WorkerSpec,
    parsed: Mapping[str, list[dict[str, Any]]],
    detector_array: np.ndarray,
) -> None:
    """Authenticates the collector fatal-gate evidence against its ledger."""

    gate_evidence = shard.get("collector_gate_evidence")
    numerical = (
        gate_evidence.get("numerical_preflight")
        if isinstance(gate_evidence, Mapping)
        else None
    )
    numerical_fields = {
        "schema",
        "real_graph",
        "syndrome_sha256",
        "decimal_precision_digits",
        "tolerance_grid_observations",
        "backend_support_fsum_passed",
        "decimal_reference_passed",
        "uncached_repeatability_passed",
        "cached_uncached_repeatability_passed",
        "tolerance_grid_passed",
    }
    first_syndrome = np.unpackbits(
        detector_array[0], bitorder="little", count=config["cell"]["num_detectors"]
    ).astype(np.uint8, copy=False)
    if not isinstance(gate_evidence, Mapping):
        raise ValueError("worker collector gate evidence must be an object")
    if set(gate_evidence) != {"schema", "real_graph", "checks", "numerical_preflight"}:
        raise ValueError("worker collector gate evidence fields are not exact")
    if (
        gate_evidence.get("schema")
        != "promatch-l1-policy-audit-worker-gate-evidence-v1"
    ):
        raise ValueError("worker collector gate evidence schema mismatch")
    if not isinstance(gate_evidence.get("real_graph"), bool):
        raise ValueError("worker collector gate evidence real_graph must be boolean")
    if not isinstance(numerical, Mapping):
        raise ValueError("worker numerical preflight must be an object")
    if set(numerical) != numerical_fields:
        raise ValueError("worker numerical preflight fields are not exact")
    if numerical.get("schema") != "promatch-l1-policy-audit-numerical-preflight-v1":
        raise ValueError("worker numerical preflight schema mismatch")
    if numerical.get("real_graph") is not gate_evidence.get("real_graph"):
        raise ValueError("worker numerical preflight real_graph flag mismatch")
    if numerical.get("syndrome_sha256") != _sha256(first_syndrome.tobytes()):
        raise ValueError("worker numerical preflight syndrome digest mismatch")
    if (
        numerical.get("decimal_precision_digits")
        != config["oracle"]["decimal_precision_digits"]
    ):
        raise ValueError("worker numerical preflight decimal precision mismatch")
    if numerical.get("tolerance_grid_observations") != len(
        config["oracle"]["tolerance_sensitivity_relative"]
    ):
        raise ValueError("worker numerical preflight tolerance grid count mismatch")
    if any(
        numerical.get(name) is not True
        for name in (
            "backend_support_fsum_passed",
            "decimal_reference_passed",
            "uncached_repeatability_passed",
            "cached_uncached_repeatability_passed",
            "tolerance_grid_passed",
        )
    ):
        raise ValueError("worker numerical preflight check flags must all be true")
    if gate_evidence != _worker_gate_evidence(
        spec=spec,
        numerical_preflight=numerical,
        shots=parsed["shots"],
        counterfactuals=parsed["counterfactuals"],
    ):
        raise ValueError(
            "worker collector gate evidence does not reconcile with its ledger"
        )


def _verify_timing_telemetry(
    path: Path,
    shard: Mapping[str, Any],
    *,
    spec: WorkerSpec,
    parsed: Mapping[str, list[dict[str, Any]]],
    artifacts: Mapping[str, Any],
) -> None:
    """Authenticates the nondeterministic timing sidecar against the shard."""

    timing_bytes = (path / "timing.json").read_bytes()
    if _sha256(timing_bytes) != shard.get("timing_sha256"):
        raise ValueError("worker timing digest mismatch")
    timing = _strict_json_load(path / "timing.json")
    timing_fields = {
        "schema",
        "sampling_ns",
        "u0_batch_decode_ns",
        "shot_audit_loop_ns",
        "ledger_serialization_ns",
        "shot_audit_wall_ns_quantiles",
        "per_shot",
        "serialization_by_artifact",
        "core_timing_by_ledger",
        "peak_rss_bytes",
        "peak_rss_source",
        "per_arm_decode_wall_ns_available",
        "per_arm_decode_wall_ns_gap",
        "worker_id",
        "scientifically_deterministic",
        "excluded_from_bit_exact_ledger_contract",
    }
    nonnegative_integer_fields = (
        "sampling_ns",
        "u0_batch_decode_ns",
        "shot_audit_loop_ns",
        "ledger_serialization_ns",
        "peak_rss_bytes",
    )
    per_shot_fields = {
        "worker_shot_index",
        "global_shot_id",
        "audit_wall_ns",
        "oracle_cache_hits",
        "oracle_cache_misses",
        "oracle_evaluation_call_count",
        "full_mwpm_cache_miss_count",
        "matched_active_pair_backend_call_count",
        "matched_active_pair_backend_wall_ns",
        "counterfactual_wall_ns",
        "support_classification_wall_ns",
        "candidate_enumeration_wall_ns",
        "stage3_enumeration_wall_ns",
    }
    timing_rows = timing.get("per_shot")
    if set(timing) != timing_fields:
        raise ValueError("worker timing fields are not exact")
    if timing.get("schema") != "promatch-l1-policy-audit-timing-v2":
        raise ValueError("worker timing schema mismatch")
    if timing.get("worker_id") != spec.worker_id:
        raise ValueError("worker timing worker identity mismatch")
    if timing.get("scientifically_deterministic") is not False:
        raise ValueError("worker timing must declare itself nondeterministic")
    if timing.get("excluded_from_bit_exact_ledger_contract") is not True:
        raise ValueError(
            "worker timing must be excluded from the bit-exact ledger contract"
        )
    if timing.get("per_arm_decode_wall_ns_available") is not False:
        raise ValueError("worker timing per-arm availability flag mismatch")
    if (
        not isinstance(timing.get("per_arm_decode_wall_ns_gap"), str)
        or not timing["per_arm_decode_wall_ns_gap"]
    ):
        raise ValueError("worker timing per-arm gap note must be a nonempty string")
    if (
        timing.get("peak_rss_source")
        != "resource.getrusage(RUSAGE_SELF).ru_maxrss-linux-kib"
    ):
        raise ValueError("worker timing peak RSS source mismatch")
    for name in nonnegative_integer_fields:
        if (
            isinstance(timing.get(name), bool)
            or not isinstance(timing.get(name), int)
            or timing[name] < 0
        ):
            raise ValueError(
                f"worker timing counter {name} must be a nonnegative integer"
            )
    if not isinstance(timing_rows, list):
        raise ValueError("worker per-shot timing must be an array")
    if len(timing_rows) != spec.shots:
        raise ValueError("worker per-shot timing row count mismatch")
    if canonical_json_bytes(timing) + b"\n" != timing_bytes:
        raise ValueError("worker timing file is not canonical JSON")
    if shard.get("timing_path") != f"shards/worker-{spec.worker_id:02d}/timing.json":
        raise ValueError("worker shard timing path mismatch")
    timing_values: list[int] = []
    for index, row in enumerate(timing_rows):
        if not isinstance(row, Mapping):
            raise ValueError("worker per-shot timing row must be an object")
        if set(row) != per_shot_fields:
            raise ValueError("worker per-shot timing row fields are not exact")
        if row.get("worker_shot_index") != index:
            raise ValueError("worker per-shot timing row index mismatch")
        if row.get("global_shot_id") != spec.shot_start + index:
            raise ValueError("worker per-shot timing global shot mismatch")
        for name in per_shot_fields - {"worker_shot_index", "global_shot_id"}:
            if (
                isinstance(row.get(name), bool)
                or not isinstance(row.get(name), int)
                or row[name] < 0
            ):
                raise ValueError(
                    f"worker per-shot timing counter {name} must be a "
                    "nonnegative integer"
                )
        timing_values.append(row["audit_wall_ns"])
    if timing.get("shot_audit_wall_ns_quantiles") != _type7_quantiles_ns(timing_values):
        raise ValueError("worker shot timing quantiles do not reconcile")
    if timing["shot_audit_loop_ns"] < sum(timing_values):
        raise ValueError("worker shot audit loop time is less than its per-shot sum")
    core_timing = timing.get("core_timing_by_ledger")
    if not isinstance(core_timing, Mapping) or set(core_timing) != {
        name for name, _ in _SIDECARS
    }:
        raise ValueError("worker core timing ledger registry is incomplete")
    for name, records in core_timing.items():
        if not isinstance(records, list):
            raise ValueError("worker core timing records must be arrays")
        seen_indices: set[int] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("worker core timing record must be an object")
            if set(record) != {
                "row_index",
                "worker_shot_index",
                "global_shot_id",
                "timing",
            }:
                raise ValueError("worker core timing record fields are not exact")
            if isinstance(record.get("row_index"), bool) or not isinstance(
                record.get("row_index"), int
            ):
                raise ValueError("worker core timing row_index must be an integer")
            if record["row_index"] < 0 or record["row_index"] >= len(parsed[name]):
                raise ValueError("worker core timing row_index is out of range")
            if record["row_index"] in seen_indices:
                raise ValueError("worker core timing row_index is duplicated")
            if not isinstance(record.get("timing"), Mapping) or not record["timing"]:
                raise ValueError(
                    "worker core timing record must carry a nonempty timing object"
                )
            source = parsed[name][record["row_index"]]
            if (
                record["worker_shot_index"] != source["worker_shot_index"]
                or record["global_shot_id"] != source["global_shot_id"]
            ):
                raise ValueError("worker core timing record identity mismatch")
            seen_indices.add(record["row_index"])
    serialization = timing.get("serialization_by_artifact")
    if not isinstance(serialization, Mapping) or set(serialization) != set(artifacts):
        raise ValueError("worker serialization telemetry is incomplete")
    serialization_component_ns = 0
    for filename, row in serialization.items():
        metadata = artifacts[filename]
        fields = {
            "canonical_jsonl_ns",
            "gzip_ns",
            "rows",
            "compressed_bytes",
            "uncompressed_bytes",
            "compressed_bytes_per_row",
            "uncompressed_bytes_per_row",
            "compressed_bytes_per_physical_shot",
            "uncompressed_bytes_per_physical_shot",
        }
        if not isinstance(row, Mapping):
            raise ValueError("worker serialization artifact telemetry must be an object")
        if set(row) != fields:
            raise ValueError(
                "worker serialization artifact telemetry fields are not exact"
            )
        for name in (
            "canonical_jsonl_ns",
            "gzip_ns",
            "rows",
            "compressed_bytes",
            "uncompressed_bytes",
        ):
            if (
                isinstance(row[name], bool)
                or not isinstance(row[name], int)
                or row[name] < 0
            ):
                raise ValueError(
                    f"worker serialization metric {name} must be a "
                    "nonnegative integer"
                )
        expected_ratios = {
            "compressed_bytes_per_row": {
                "numerator": metadata["compressed_bytes"],
                "denominator": metadata["rows"],
            },
            "uncompressed_bytes_per_row": {
                "numerator": metadata["uncompressed_bytes"],
                "denominator": metadata["rows"],
            },
            "compressed_bytes_per_physical_shot": {
                "numerator": metadata["compressed_bytes"],
                "denominator": spec.shots,
            },
            "uncompressed_bytes_per_physical_shot": {
                "numerator": metadata["uncompressed_bytes"],
                "denominator": spec.shots,
            },
        }
        if row["rows"] != metadata["rows"]:
            raise ValueError("worker serialization row count does not reconcile")
        if row["compressed_bytes"] != metadata["compressed_bytes"]:
            raise ValueError(
                "worker serialization compressed byte count does not reconcile"
            )
        if row["uncompressed_bytes"] != metadata["uncompressed_bytes"]:
            raise ValueError(
                "worker serialization uncompressed byte count does not reconcile"
            )
        for name, expected in expected_ratios.items():
            if row[name] != expected:
                raise ValueError(
                    f"worker serialization byte ratio {name} does not reconcile"
                )
        serialization_component_ns += row["canonical_jsonl_ns"] + row["gzip_ns"]
    if timing["ledger_serialization_ns"] < serialization_component_ns:
        raise ValueError("worker serialization timing does not reconcile")


def verify_worker_shard(
    path: Path,
    *,
    config: Mapping[str, Any],
    mode: str,
    spec: WorkerSpec,
) -> dict[str, Any]:
    """Authenticates a whole immutable worker shard before resume."""

    expected_names = {
        "shots.jsonl.gz",
        "proposals.jsonl.gz",
        "counterfactuals.jsonl.gz",
        "domains.jsonl.gz",
        "timing.json",
        "shard.json",
    }
    if not path.is_dir() or {p.name for p in path.iterdir()} != expected_names:
        raise ValueError(f"partial or unexpected worker shard {path}")
    if path.is_symlink() or any(
        entry.is_symlink() or not entry.is_file() for entry in path.iterdir()
    ):
        raise ValueError(f"worker shard contains a symlink or non-file entry: {path}")
    shard = _strict_json_load(path / "shard.json")
    _verify_shard_manifest_identity(shard, config=config, mode=mode, spec=spec)
    expected_seed = derive_policy_worker_seed(
        seed_root=config["sampling"]["seed_roots"][mode],
        experiment_id=config["experiment_id"],
        cell_id=config["cell"]["cell_id"],
        worker_id=spec.worker_id,
    )
    if shard.get("stim_seed") != expected_seed:
        raise ValueError("worker shard seed mismatch")
    if shard.get("native_thread_environment") != {
        name: "1" for name in THREAD_ENVIRONMENT
    }:
        raise ValueError(
            "worker shard native thread environment is not single-threaded"
        )
    if shard.get("nondeterministic_telemetry_paths") != [
        f"shards/worker-{spec.worker_id:02d}/timing.json"
    ]:
        raise ValueError(
            "worker shard does not isolate nondeterministic timing telemetry"
        )
    if shard.get("bit_exact_regeneration_paths") != [
        f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz" for name, _ in _SIDECARS
    ]:
        raise ValueError("worker shard bit-exact ledger registry is incomplete")
    artifacts = shard.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        f"{name}.jsonl.gz" for name, _ in _SIDECARS
    }:
        raise ValueError("worker shard artifact registry is incomplete")
    parsed: dict[str, list[dict[str, Any]]] = {}
    for name, schema in _SIDECARS:
        compressed = (path / f"{name}.jsonl.gz").read_bytes()
        try:
            raw = gzip.decompress(compressed)
        except (gzip.BadGzipFile, EOFError) as ex:
            raise ValueError(f"invalid compressed sidecar {name}") from ex
        if deterministic_gzip(raw) != compressed:
            raise ValueError(f"sidecar gzip bytes are not deterministic for {name}")
        metadata = artifacts[f"{name}.jsonl.gz"]
        expected = _artifact_metadata(
            compressed=compressed,
            uncompressed=raw,
            rows=len(raw.splitlines()),
            schema=schema,
            path=f"shards/worker-{spec.worker_id:02d}/{name}.jsonl.gz",
        )
        if metadata != expected:
            raise ValueError(f"sidecar digest/count mismatch for {name}")
        lines = raw.splitlines()
        parsed[name] = []

        # Bind the sidecar name eagerly: the hook runs inside json.loads in
        # this iteration, but the default argument keeps that explicit (B023).
        def remember(
            items: list[tuple[str, Any]], *, name: str = name
        ) -> dict[str, Any]:
            seen: set[str] = set()
            for key, _ in items:
                if key in seen:
                    raise ValueError(f"duplicate JSONL key {key!r} in {name}")
                seen.add(key)
            return dict(items)

        for line in lines:
            row = json.loads(line, object_pairs_hook=remember)
            if canonical_json_bytes(row) != line:
                raise ValueError(f"sidecar row is not canonical JSON for {name}")
            _, leaked_timing = _separate_nondeterministic_timing(row)
            if leaked_timing is not None:
                raise ValueError(
                    f"scientific ledger contains nondeterministic timing in {name}"
                )
            _verify_sidecar_row_identity(
                row, name=name, config=config, spec=spec, expected_seed=expected_seed
            )
            if name != "shots":
                if row.get("arm_id") not in ARM_IDS:
                    raise ValueError(f"sidecar contains an unknown arm for {name}")
                forbid_ground_truth_keys(row, path=name)
            if (
                name in {"proposals", "counterfactuals"}
                and row.get("graph_fingerprint") != config["cell"]["graph_fingerprint"]
            ):
                raise ValueError(f"{name} row graph fingerprint mismatch")
            if name in {"proposals", "counterfactuals"}:
                _validate_support_difference_ledger(row, path=name)
                _validate_context_union_ledger(row, path=name)
            _require_float_hex_companions(row, path=name)
            parsed[name].append(row)
        if name == "shots":
            ids = [row["global_shot_id"] for row in parsed[name]]
            if ids != list(range(spec.shot_start, spec.shot_stop)):
                raise ValueError("shot sidecar does not exactly cover worker range")

    detector_width = (int(config["cell"]["num_detectors"]) + 7) // 8
    observable_width = (int(config["cell"]["num_observables"]) + 7) // 8
    detector_rows: list[bytes] = []
    observable_rows: list[bytes] = []
    shot_identity: dict[int, tuple[str, int]] = {}
    for row in parsed["shots"]:
        detector_bytes, observable_bytes = _verify_shot_row_reconciliation(
            row, config=config
        )
        detector_rows.append(detector_bytes)
        observable_rows.append(observable_bytes)
        shot_identity[row["worker_shot_index"]] = (
            row["physical_input_sha256"],
            row["global_shot_id"],
        )
    for name in ("proposals", "counterfactuals", "domains"):
        for row in parsed[name]:
            if shot_identity.get(row["worker_shot_index"]) != (
                row["physical_input_sha256"],
                row["global_shot_id"],
            ):
                raise ValueError(f"{name} row does not reference its retained shot")
    detector_array = np.asarray(
        [list(value) for value in detector_rows], dtype=np.uint8
    ).reshape(spec.shots, detector_width)
    observable_array = np.asarray(
        [list(value) for value in observable_rows], dtype=np.uint8
    ).reshape(spec.shots, observable_width)
    for name, array in (
        ("detectors", detector_array),
        ("observables", observable_array),
    ):
        digest = digest_array(array)
        if shard.get(name) != {
            "sha256": digest.sha256,
            "shape": list(digest.shape),
            "dtype": digest.dtype,
        }:
            raise ValueError(f"worker {name} array digest mismatch")
    if shard.get("tail_censor_attestation") != _worker_tail_censor_attestation(
        parsed["counterfactuals"]
    ):
        raise ValueError("worker tail/censor attestation does not match its ledger")
    _verify_gate_evidence(
        shard,
        config=config,
        spec=spec,
        parsed=parsed,
        detector_array=detector_array,
    )
    _verify_timing_telemetry(
        path, shard, spec=spec, parsed=parsed, artifacts=artifacts
    )
    if canonical_json_bytes(shard) + b"\n" != (path / "shard.json").read_bytes():
        raise ValueError("worker shard manifest is not canonical")
    return shard
