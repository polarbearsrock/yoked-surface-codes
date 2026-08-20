"""Deterministic schedules, frozen constants, and experiment identity.

This is the first protocol-order slice of the B1 policy-audit collector: the
frozen schema names, seed derivation, worker schedules, source-path scopes,
and canonical hashing/IO primitives that every later slice builds on.  It
inherits the package isolation contract (see ``__init__``): protocol and
provenance only, never per-shot policy logic or ground truth.
"""

from __future__ import annotations

# Native-thread limits intentionally execute before numerical/package imports.
# ruff: noqa: E402

# These limits must be in force before importing NumPy/PyMatching indirectly.
import os

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_name in THREAD_ENVIRONMENT:
    os.environ[_thread_name] = "1"

import dataclasses
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any, Mapping

from yoked.decoding._artifact_io import install_bytes_atomic
from yoked.decoding._promatch_stats import canonical_json_bytes


PROTOCOL_SCHEMA = "promatch-l1-policy-audit-protocol-v1"
EXPERIMENT_SCHEMA = "promatch-l1-policy-audit-experiment-v1"
MANIFEST_SCHEMA = "promatch-l1-policy-audit-manifest-v1"
SHARD_SCHEMA = "promatch-l1-policy-audit-shard-v1"
SHOT_SCHEMA = "promatch-l1-policy-audit-shot-v1"
PROPOSAL_SCHEMA = "promatch-l1-policy-audit-proposal-v1"
COUNTERFACTUAL_SCHEMA = "promatch-l1-policy-audit-counterfactual-v1"
DOMAIN_SCHEMA = "promatch-l1-policy-audit-domain-v1"
PROBE_ATTESTATION_SCHEMA = "promatch-l1-policy-audit-probe-attestation-v1"
SEED_DERIVATION = (
    "sha256-root+promatch-policy-worker-v1+experiment-id+cell-id+worker-id-"
    "first8-uint64le"
)
SCIENTIFIC_SHOTS = 20_000
SCIENTIFIC_WORKERS = 32
SCIENTIFIC_SHOTS_PER_WORKER = 625
GZIP_LEVEL = 9
ARM_IDS = (
    "u0-joint-y2",
    "pu-v3-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2-shadow",
    "pu-ocost-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2",
    "pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-tx-joint-y2",
    "pu-oframe-window-hw10-boundary-disabled-observable-zero-frame-partial-joint-y2",
)
# Frozen V1/V2 manifests record the pre-cleanup layout. Keep this tuple exact
# so those artifacts remain readable for analysis; scientific execution still
# rejects a checkout whose commit or source hashes differ from the manifest.
LEGACY_POLICY_SOURCE_PATHS = (
    "requirements.txt",
    "src/yoked/_yoked_memory_circuits.py",
    "src/yoked/decoding/__init__.py",
    "src/yoked/decoding/_promatch.py",
    "src/yoked/decoding/_promatch_decoder.py",
    "src/yoked/decoding/_promatch_experiment.py",
    "src/yoked/decoding/_promatch_graph.py",
    "src/yoked/decoding/_promatch_layout.py",
    "src/yoked/decoding/_promatch_oracle.py",
    "src/yoked/decoding/_promatch_oracle_test.py",
    "src/yoked/decoding/_promatch_oracle_replay.py",
    "src/yoked/decoding/_promatch_oracle_replay_test.py",
    "src/yoked/decoding/_promatch_stats.py",
    "src/yoked/decoding/_promatch_stepper_test.py",
    "src/yoked/decoding/_promatch_policy_audit.py",
    "src/yoked/decoding/_promatch_policy_audit_test.py",
    "src/yoked/decoding/_promatch_policy_casebook.py",
    "src/yoked/decoding/_promatch_policy_casebook_test.py",
    "src/yoked/decoding/_promatch_policy_experiment.py",
    "src/yoked/decoding/_promatch_policy_experiment_test.py",
    "src/yoked/decoding/_promatch_policy_analysis.py",
    "src/yoked/decoding/_promatch_policy_analysis_test.py",
    "experiments/PROMATCH_L1_POLICY_AUDIT_20K.md",
    "tools/benchmark_promatch_policy_audit",
    "tools/analyze_promatch_policy_audit",
)

# Newly materialized protocols authenticate the reader-facing package layout.
POLICY_SOURCE_PATHS = (
    "requirements.txt",
    "pytest.ini",
    "src/yoked/_yoked_memory_circuits.py",
    "src/yoked/decoding/__init__.py",
    "src/yoked/decoding/_artifact_io.py",
    "src/yoked/decoding/_promatch.py",
    "src/yoked/decoding/_promatch_decoder.py",
    "src/yoked/decoding/_promatch_experiment.py",
    "src/yoked/decoding/_promatch_graph.py",
    "src/yoked/decoding/_promatch_layout.py",
    "src/yoked/decoding/oracle/__init__.py",
    "src/yoked/decoding/oracle/full_graph.py",
    "tests/yoked/decoding/oracle/full_graph_test.py",
    "src/yoked/decoding/oracle/replay.py",
    "tests/yoked/decoding/oracle/replay_test.py",
    "src/yoked/decoding/_promatch_stats.py",
    "tests/yoked/decoding/_promatch_stepper_test.py",
    "src/yoked/decoding/oracle/policy_audit.py",
    "tests/yoked/decoding/oracle/policy_audit_test.py",
    "src/yoked/decoding/oracle/policy_casebook.py",
    "tests/yoked/decoding/oracle/policy_casebook_test.py",
    "src/yoked/decoding/oracle/policy_experiment/__init__.py",
    "src/yoked/decoding/oracle/policy_experiment/_attestation.py",
    "src/yoked/decoding/oracle/policy_experiment/_collection.py",
    "src/yoked/decoding/oracle/policy_experiment/_identity.py",
    "src/yoked/decoding/oracle/policy_experiment/_ledger.py",
    "src/yoked/decoding/oracle/policy_experiment/_protocol.py",
    "src/yoked/decoding/oracle/policy_experiment/_shards.py",
    "tests/yoked/decoding/oracle/policy_experiment_test.py",
    "src/yoked/decoding/oracle/policy_analysis/__init__.py",
    "src/yoked/decoding/oracle/policy_analysis/_artifacts.py",
    "src/yoked/decoding/oracle/policy_analysis/_casebook.py",
    "src/yoked/decoding/oracle/policy_analysis/_contract.py",
    "src/yoked/decoding/oracle/policy_analysis/_corpus.py",
    "src/yoked/decoding/oracle/policy_analysis/_fields.py",
    "src/yoked/decoding/oracle/policy_analysis/_plots.py",
    "src/yoked/decoding/oracle/policy_analysis/_report.py",
    "src/yoked/decoding/oracle/policy_analysis/_rows.py",
    "src/yoked/decoding/oracle/policy_analysis/_stats.py",
    "src/yoked/decoding/oracle/policy_analysis/_tables.py",
    "tests/yoked/decoding/oracle/policy_analysis_test.py",
    "experiments/PROMATCH_L1_POLICY_AUDIT_20K.md",
    "tools/benchmark_promatch_policy_audit",
    "tools/analyze_promatch_policy_audit",
)
ANALYSIS_TABLE_NAMES = (
    "overview",
    "paired_outcomes",
    "event_and_transaction_summary",
    "residual_hw_distributions",
    "domain_terminal_summary",
    "certificate_by_stage",
    "certificate_by_domain",
    "unsafe_fraction_by_stage",
    "counterfactual_terminal_action",
    "first_safe_rank",
    "stage_transition",
    "context_views",
    "visibility_summary",
    "association_by_unsafe_count",
    "unsafe_count_distribution",
    "association_by_first",
    "first_conflict_discordant",
    "continuous_distributions",
    "local_competitor_summary",
    "cost_excess_ecdf_by_stage",
    "cost_excess_ecdf_by_certificate",
    "cost_excess_ecdf_by_context",
    "original_vs_alternative",
    "risk_heatmaps",
    "veto_chain_tails",
    "fatal_gates",
    "interpretation_checkpoints",
)
ANALYSIS_PLOT_NAMES = (
    "certificate-flow",
    "unsafe-fraction-by-stage",
    "first-conflict-stage-context",
    "cost-excess-ecdf",
    "first-safe-action-rank",
    "stage-transition-matrix",
    "original-versus-alternative",
    "risk-heatmaps",
    "disagreement-association",
    "event-relief",
    "veto-chain-tails",
)
_ARM_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SIDECARS = (
    ("shots", SHOT_SCHEMA),
    ("proposals", PROPOSAL_SCHEMA),
    ("counterfactuals", COUNTERFACTUAL_SCHEMA),
    ("domains", DOMAIN_SCHEMA),
)


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """One immutable worker-owned half-open range of physical shots."""

    worker_id: int
    shot_start: int
    shots: int

    @property
    def shot_stop(self) -> int:
        return self.shot_start + self.shots

    def to_json(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json_load(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f, object_pairs_hook=unique)
    except json.JSONDecodeError as ex:
        raise ValueError(f"invalid JSON file {path}: {ex}") from ex
    if not isinstance(value, dict):
        raise ValueError(f"JSON file {path} must contain one object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    install_bytes_atomic(
        path, canonical_json_bytes(value) + b"\n", prefix="promatch-policy-json-"
    )


def policy_worker_schedule(mode: str) -> tuple[WorkerSpec, ...]:
    """Returns the exact deterministic worker-owned shot ranges."""

    if mode == "scientific":
        counts = [SCIENTIFIC_SHOTS_PER_WORKER] * SCIENTIFIC_WORKERS
    elif mode == "smoke":
        counts = [1] * SCIENTIFIC_WORKERS
    elif mode == "probe":
        counts = [4] * 4 + [3] * 28
    else:
        raise ValueError(f"unsupported policy-audit mode {mode!r}")
    rows: list[WorkerSpec] = []
    cursor = 0
    for worker_id, shots in enumerate(counts):
        rows.append(WorkerSpec(worker_id, cursor, shots))
        cursor += shots
    expected = {"scientific": 20_000, "smoke": 32, "probe": 100}[mode]
    if cursor != expected or len(rows) != SCIENTIFIC_WORKERS:
        raise AssertionError("internal B1 worker schedule is inconsistent")
    return tuple(rows)


def derive_policy_worker_seed(
    *, seed_root: str, experiment_id: str, cell_id: str, worker_id: int
) -> int:
    """Derives a schedule-independent uint64 Stim seed for one worker."""

    if not re.fullmatch(r"[0-9a-f]{64}", seed_root):
        raise ValueError("sampling seed root must be 64 lowercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{64}", experiment_id):
        raise ValueError("experiment_id must be a lowercase SHA-256 digest")
    if not isinstance(cell_id, str) or not cell_id or not cell_id.isascii():
        raise ValueError("cell_id must be nonempty ASCII")
    if isinstance(worker_id, bool) or not isinstance(worker_id, int):
        raise TypeError("worker_id must be an integer")
    if worker_id < 0 or worker_id >= SCIENTIFIC_WORKERS:
        raise ValueError("worker_id must be in [0, 32)")
    digest = hashlib.sha256(
        bytes.fromhex(seed_root)
        + b"promatch-policy-worker-v1\0"
        + bytes.fromhex(experiment_id)
        + b"\0"
        + cell_id.encode("ascii")
        + b"\0"
        + struct.pack("<Q", worker_id)
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _semantic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(config)))
    result.pop("experiment_id", None)
    result.pop("config_self_sha256", None)
    return result


def policy_config_self_sha256(config: Mapping[str, Any]) -> str:
    """Hashes canonical protocol semantics with a config-specific domain tag."""

    return _sha256(
        b"promatch-policy-config-v1\0" + canonical_json_bytes(_semantic_config(config))
    )


def policy_experiment_id(config: Mapping[str, Any]) -> str:
    """Derives the experiment identity from canonical semantic configuration."""

    return _sha256(
        b"promatch-policy-experiment-v1\0"
        + canonical_json_bytes(_semantic_config(config))
    )


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    value: Any, required: set[str], *, name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != required:
        raise ValueError(
            f"{name} fields must be exactly {sorted(required)}; got {sorted(value)}"
        )
    return value
