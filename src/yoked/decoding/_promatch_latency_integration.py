"""Real yoked-circuit workloads for the controlled ProMatch latency runner.

The generic runner in :mod:`yoked.decoding._promatch_latency` deliberately
knows nothing about circuits or decoders.  This module is the narrow bridge
from a canonical paired-experiment manifest to that runner.  A
:class:`YokedPromatchLatencyFactory` is pickle-friendly and reconstructs all
scientific state inside each fresh restart worker.  Circuit generation,
detector sampling, decoder compilation, and residual-corpus generation all
finish before the runner starts warmup or reads its clock.

Both backend diagnostics use ordinary, uncorrelated PyMatching.  Correlated
matching is intentionally not available through this integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pymatching

from yoked.decoding._promatch_decoder import (
    CompiledIdentityWrappedPyMatchingDecoder,
    CompiledPromatchDecoder,
    IdentityWrappedPyMatchingDecoder,
)
from yoked.decoding._promatch_experiment import (
    PROTOCOL_SCHEMA,
    _decoder_config,
    _dem_options,
    prepare_cell,
)
from yoked.decoding._promatch_latency import LatencyProtocol, LatencyWorkload
from yoked.decoding._promatch_stats import (
    canonical_json_bytes,
    derive_stim_batch_seed,
    manifest_experiment_id,
)


__all__ = [
    "TinyLatencySmokeConfig",
    "YokedPromatchLatencyFactory",
    "latency_protocol_from_manifest",
]


_SCIENTIFIC_COUNTS = {
    "process_restarts": 10,
    "warmup_calls_per_decoder_per_restart": 1_000,
    "paired_blocks_per_restart": 100,
    "calls_per_decoder_per_block": 100,
}
_SCIENTIFIC_GATES = {
    "backend_geometric_ratio_upper": 0.9,
    "total_geometric_ratio_upper": 0.95,
    "total_p99_ratio_upper": 1.05,
}
_SCIENTIFIC_PRIMARY_BATCH = 1
_SCIENTIFIC_SECONDARY_BATCHES = (64, 1024)
_SCIENTIFIC_CORPUS_SHOTS_PER_RESTART = 10_000
_SCIENTIFIC_PRIMARY_INTERVAL = "adapter entry through packed prediction return"
_SCIENTIFIC_DIAGNOSTIC_INTERVALS = ["total", "backend"]
_SCIENTIFIC_DECODER = {
    "residual_hw_limit": 10,
    "domain_mode": "windowd",
    "boundary_policy": "disabled",
    "observable_policy": "zero-frame",
}
_TIMING_CORPUS_ROOT = "timing_corpus"
_TIMING_SCHEDULE_ROOT = "timing_bootstrap"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _seed_root(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a literal 256-bit hexadecimal seed root")
    try:
        bytes.fromhex(value)
    except ValueError as ex:
        raise ValueError(
            f"{name} must be a literal 256-bit hexadecimal seed root"
        ) from ex
    return value.lower()


def _timing_protocol(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    analysis = manifest.get("analysis_config")
    if not isinstance(analysis, Mapping):
        raise ValueError("normalized manifest must contain analysis_config")
    timing = analysis.get("timing_protocol")
    if not isinstance(timing, Mapping):
        raise ValueError("normalized manifest must contain timing_protocol")
    return timing


def _timing_roots(manifest: Mapping[str, Any]) -> tuple[str, str]:
    roots = manifest.get("sampler_seed_roots")
    if not isinstance(roots, Mapping):
        raise ValueError("normalized manifest must contain sampler_seed_roots")
    return (
        _seed_root(roots.get(_TIMING_CORPUS_ROOT), name=_TIMING_CORPUS_ROOT),
        _seed_root(roots.get(_TIMING_SCHEDULE_ROOT), name=_TIMING_SCHEDULE_ROOT),
    )


@dataclass(frozen=True)
class TinyLatencySmokeConfig:
    """Explicit, non-claim-bearing replacement for the frozen timing counts.

    No tiny configuration is selected implicitly.  A caller must construct
    and pass this object together with ``scientific=False``.
    """

    restarts: int = 1
    blocks_per_restart: int = 2
    calls_per_block: int = 1
    warmup_calls_per_variant: int = 1
    batch_sizes: tuple[int, ...] = (1,)
    schedule_seed: int = 0x534D4F4B45
    corpus_batches: int = 2
    timing_corpus_seed_root: str = "5c" * 32

    def validate(self) -> None:
        _positive_int(self.restarts, name="smoke.restarts")
        blocks = _positive_int(
            self.blocks_per_restart, name="smoke.blocks_per_restart"
        )
        if blocks % 2:
            raise ValueError("smoke.blocks_per_restart must be even")
        _positive_int(self.calls_per_block, name="smoke.calls_per_block")
        _positive_int(
            self.warmup_calls_per_variant,
            name="smoke.warmup_calls_per_variant",
        )
        if not isinstance(self.batch_sizes, tuple) or not self.batch_sizes:
            raise TypeError("smoke.batch_sizes must be a nonempty tuple")
        for value in self.batch_sizes:
            _positive_int(value, name="smoke.batch_size")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("smoke.batch_sizes must not contain duplicates")
        if isinstance(self.schedule_seed, bool) or not isinstance(
            self.schedule_seed, int
        ):
            raise TypeError("smoke.schedule_seed must be an integer")
        if not 0 <= self.schedule_seed < 2**256:
            raise ValueError("smoke.schedule_seed must lie in [0, 2**256)")
        _positive_int(self.corpus_batches, name="smoke.corpus_batches")
        _seed_root(
            self.timing_corpus_seed_root,
            name="smoke.timing_corpus_seed_root",
        )

    def protocol(self) -> LatencyProtocol:
        self.validate()
        result = LatencyProtocol(
            restarts=self.restarts,
            blocks_per_restart=self.blocks_per_restart,
            calls_per_block=self.calls_per_block,
            warmup_calls_per_variant=self.warmup_calls_per_variant,
            batch_sizes=self.batch_sizes,
            schedule_seed=self.schedule_seed,
        )
        result.validate(scientific=False)
        return result


def latency_protocol_from_manifest(
    manifest: Mapping[str, Any],
    *,
    scientific: bool,
    smoke: TinyLatencySmokeConfig | None = None,
) -> LatencyProtocol:
    """Extract and validate the latency runner protocol.

    Scientific extraction is fail-closed: all preregistered counts, the
    primary/secondary batch sizes, and the practical claim gates must match the
    round-one design exactly.  The literal ``timing_bootstrap`` root becomes
    the schedule seed.  Tiny counts are possible only through an explicit
    :class:`TinyLatencySmokeConfig` and are always non-claim-bearing.
    """

    if smoke is not None:
        if scientific:
            raise ValueError("a tiny smoke protocol cannot be claim-bearing")
        return smoke.protocol()
    if not scientific:
        raise ValueError(
            "non-scientific latency requires an explicit TinyLatencySmokeConfig"
        )

    timing = _timing_protocol(manifest)
    for field, expected in _SCIENTIFIC_COUNTS.items():
        if timing.get(field) != expected:
            raise ValueError(
                f"scientific timing_protocol requires {field}={expected}"
            )
    if timing.get("primary_mode_batch_size") != _SCIENTIFIC_PRIMARY_BATCH:
        raise ValueError("scientific timing_protocol requires primary batch size 1")
    secondary = timing.get("secondary_batch_sizes")
    if secondary != list(_SCIENTIFIC_SECONDARY_BATCHES):
        raise ValueError(
            "scientific timing_protocol requires secondary batch sizes [64, 1024]"
        )
    if timing.get("input_generation") != "pregenerated_outside_timed_regions":
        raise ValueError("scientific timing inputs must be pregenerated")
    if timing.get("clock") != "time.perf_counter_ns":
        raise ValueError("scientific timing requires time.perf_counter_ns")
    if timing.get("diagnostic_intervals") != _SCIENTIFIC_DIAGNOSTIC_INTERVALS:
        raise ValueError(
            "scientific timing requires diagnostic_intervals=['total', 'backend']"
        )
    if (
        manifest.get("phase") == "confirm"
        and timing.get("primary_interval") != _SCIENTIFIC_PRIMARY_INTERVAL
    ):
        raise ValueError(
            "scientific confirmatory timing requires the frozen adapter interval"
        )
    if timing.get("block_order") != "randomized_balanced_AB_BA":
        raise ValueError("scientific timing requires randomized balanced AB/BA blocks")
    if (
        timing.get("timing_corpus_shots_per_restart")
        != _SCIENTIFIC_CORPUS_SHOTS_PER_RESTART
        or timing.get("timing_corpus_reuse_policy")
        != "cycle_pregenerated_natural_noise_corpus"
    ):
        raise ValueError(
            "scientific timing requires a frozen 10000-shot natural-noise corpus per restart"
        )
    if timing.get("restart_concurrency") != 1:
        raise ValueError("scientific timing restarts must be serialized")

    gate_key = "claim_gates" if "claim_gates" in timing else "gates"
    gates = timing.get(gate_key)
    if gates != _SCIENTIFIC_GATES:
        raise ValueError(
            f"scientific timing_protocol requires exact {gate_key}={_SCIENTIFIC_GATES}"
        )
    _, schedule_root = _timing_roots(manifest)
    result = LatencyProtocol(
        restarts=_SCIENTIFIC_COUNTS["process_restarts"],
        blocks_per_restart=_SCIENTIFIC_COUNTS["paired_blocks_per_restart"],
        calls_per_block=_SCIENTIFIC_COUNTS["calls_per_decoder_per_block"],
        warmup_calls_per_variant=_SCIENTIFIC_COUNTS[
            "warmup_calls_per_decoder_per_restart"
        ],
        batch_sizes=(
            _SCIENTIFIC_PRIMARY_BATCH,
            *_SCIENTIFIC_SECONDARY_BATCHES,
        ),
        # Keeping the conventional big-endian hexadecimal interpretation makes
        # LatencyProtocol.to_json reproduce the literal committed seed root.
        schedule_seed=int(schedule_root, 16),
    )
    result.validate(scientific=True)
    return result


@dataclass(frozen=True)
class _MatcherCall:
    """Uninstrumented packed call into an uncorrelated matcher."""

    matcher: pymatching.Matching
    num_observables: int

    def __call__(self, packed_detection_events: np.ndarray) -> np.ndarray:
        result = np.asarray(
            self.matcher.decode_batch(
                packed_detection_events,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
        expected = (packed_detection_events.shape[0], (self.num_observables + 7) // 8)
        if result.shape != expected:
            raise AssertionError(
                f"uncorrelated matcher returned shape {result.shape}, expected {expected}"
            )
        if self.num_observables % 8 and result.shape[1]:
            result[:, -1] &= (1 << (self.num_observables % 8)) - 1
        return result


@dataclass(frozen=True)
class _IdentityWrapperCall:
    decoder: CompiledIdentityWrappedPyMatchingDecoder

    def __call__(self, packed_detection_events: np.ndarray) -> np.ndarray:
        return self.decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed_detection_events
        )


@dataclass(frozen=True)
class _PromatchCall:
    decoder: CompiledPromatchDecoder

    def __call__(self, packed_detection_events: np.ndarray) -> np.ndarray:
        return self.decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed_detection_events
        )


def _config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def _manifest_latency_cells(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the disjoint accuracy and performance timing scopes."""

    result: list[Mapping[str, Any]] = []
    for field in ("cells", "performance_cells"):
        raw = manifest.get(field, [])
        if not isinstance(raw, list):
            raise ValueError(f"normalized manifest {field} must be an array")
        if any(not isinstance(cell, Mapping) for cell in raw):
            raise ValueError(f"normalized manifest {field} entries must be objects")
        result.extend(raw)
    ids = [cell.get("cell_id") for cell in result]
    if len(ids) != len(set(ids)):
        raise ValueError("accuracy/performance latency cell IDs must be disjoint")
    return result


@dataclass(frozen=True)
class YokedPromatchLatencyFactory:
    """Pickle-friendly restart factory backed by canonical JSON state."""

    manifest_json: str
    cell_id: str
    batch_sizes: tuple[int, ...]
    timing_corpus_seed_root: str
    corpus_batches: int = 1
    corpus_shots_per_restart: int | None = None
    verify_hashes: bool = True

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        cell_id: str,
        scientific: bool,
        smoke: TinyLatencySmokeConfig | None = None,
    ) -> "YokedPromatchLatencyFactory":
        """Freeze one normalized manifest cell into a restart factory."""

        if manifest.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError(
                "latency factory requires a canonical normalized protocol manifest"
            )
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a nonempty string")
        protocol = latency_protocol_from_manifest(
            manifest,
            scientific=scientific,
            smoke=smoke,
        )
        cells = _manifest_latency_cells(manifest)
        matches = [cell for cell in cells if cell.get("cell_id") == cell_id]
        if len(matches) != 1:
            raise ValueError(f"cell_id {cell_id!r} must identify exactly one cell")
        if scientific:
            if manifest.get("status") != "FROZEN" or manifest.get("frozen") is not True:
                raise ValueError("scientific latency requires a frozen manifest")
            if _decoder_config(manifest) != _SCIENTIFIC_DECODER:
                raise ValueError("scientific latency requires the primary PU-window decoder")
            required_hashes = {
                "circuit_sha256",
                "dem_sha256",
                "layout_fingerprint",
                "graph_fingerprint",
            }
            if not required_hashes <= set(matches[0]):
                raise ValueError("scientific latency cell is missing frozen hashes")
            embedded_id = manifest.get("experiment_id")
            if embedded_id != manifest_experiment_id(manifest):
                raise ValueError("scientific latency manifest has a stale experiment_id")
            corpus_root, _ = _timing_roots(manifest)
            corpus_batches = 1
            corpus_shots_per_restart = _SCIENTIFIC_CORPUS_SHOTS_PER_RESTART
        else:
            if smoke is None:
                raise ValueError("non-scientific latency requires explicit smoke config")
            smoke.validate()
            corpus_root = _seed_root(
                smoke.timing_corpus_seed_root,
                name="smoke.timing_corpus_seed_root",
            )
            corpus_batches = smoke.corpus_batches
            corpus_shots_per_restart = None

        # A canonical JSON string is immutable, pickle-stable, and prevents
        # later caller mutation from changing a queued restart task.
        manifest_json = canonical_json_bytes(manifest).decode("utf-8")
        return cls(
            manifest_json=manifest_json,
            cell_id=cell_id,
            batch_sizes=protocol.batch_sizes,
            timing_corpus_seed_root=corpus_root,
            corpus_batches=corpus_batches,
            corpus_shots_per_restart=corpus_shots_per_restart,
            verify_hashes=scientific,
        )

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.manifest_json)
        except (TypeError, json.JSONDecodeError) as ex:
            raise ValueError("manifest_json must be canonical JSON") from ex
        if canonical_json_bytes(manifest).decode("utf-8") != self.manifest_json:
            raise ValueError("manifest_json is not canonical")
        if not isinstance(self.batch_sizes, tuple) or not self.batch_sizes:
            raise TypeError("batch_sizes must be a nonempty tuple")
        for batch_size in self.batch_sizes:
            _positive_int(batch_size, name="batch_size")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates")
        _positive_int(self.corpus_batches, name="corpus_batches")
        if self.corpus_shots_per_restart is not None:
            _positive_int(
                self.corpus_shots_per_restart,
                name="corpus_shots_per_restart",
            )
        _seed_root(self.timing_corpus_seed_root, name="timing_corpus_seed_root")

    @property
    def suite_identity(self) -> Mapping[str, Any]:
        """Identity bound into latency ledgers before any resume is accepted."""

        manifest = json.loads(self.manifest_json)
        cells = _manifest_latency_cells(manifest)
        matches = [cell for cell in cells if cell.get("cell_id") == self.cell_id]
        if len(matches) != 1:
            raise ValueError(f"cell_id {self.cell_id!r} must identify exactly one cell")
        cell = matches[0]
        hashes = {
            key: cell.get(key)
            for key in (
                "circuit_sha256",
                "dem_sha256",
                "layout_fingerprint",
                "graph_fingerprint",
            )
        }
        if self.verify_hashes and any(value is None for value in hashes.values()):
            raise ValueError("scientific latency suite identity is missing frozen hashes")
        result = {
            "experiment_id": manifest.get("experiment_id"),
            "cell_id": self.cell_id,
            "decoder_config_sha256": _config_digest(_decoder_config(manifest)),
            "dem_options_sha256": _config_digest(_dem_options(manifest)),
            "timing_corpus_seed_root_sha256": hashlib.sha256(
                bytes.fromhex(self.timing_corpus_seed_root)
            ).hexdigest(),
            "corpus_batches": self.corpus_batches,
            "corpus_shots_per_restart": self.corpus_shots_per_restart,
            "batch_sizes": list(self.batch_sizes),
            "verify_hashes": self.verify_hashes,
        }
        if self.verify_hashes:
            result["cell_hashes"] = hashes
        return result

    def __call__(self, restart_index: int, batch_size: int) -> LatencyWorkload:
        if isinstance(restart_index, bool) or not isinstance(restart_index, int):
            raise TypeError("restart_index must be an integer")
        if restart_index < 0:
            raise ValueError("restart_index must be nonnegative")
        if batch_size not in self.batch_sizes:
            raise ValueError(f"batch_size={batch_size} is not declared by the factory")

        manifest = json.loads(self.manifest_json)
        cells = _manifest_latency_cells(manifest)
        cell = next(item for item in cells if item["cell_id"] == self.cell_id)
        decoder_config = _decoder_config(manifest)
        dem_options = _dem_options(manifest)

        # Each restart rebuilds and, for scientific runs, verifies the exact
        # circuit, DEM, layout, and graph before producing any timing input.
        prepared = prepare_cell(
            cell,
            decoder_config=decoder_config,
            dem_options=dem_options,
            verify_hashes=self.verify_hashes,
        )
        compiled_pu = prepared.compiled_pu
        if not isinstance(compiled_pu, CompiledPromatchDecoder):
            raise TypeError("prepare_cell did not compile the production ProMatch adapter")

        # The counter is injective for the declared (restart, batch-size)
        # pairs, so every timing corpus has a distinct deterministic Stim seed.
        batch_index = self.batch_sizes.index(batch_size)
        corpus_batch_id = restart_index * len(self.batch_sizes) + batch_index
        if corpus_batch_id >= 2**64:
            raise ValueError("timing corpus batch ID exceeds uint64")
        stim_seed = derive_stim_batch_seed(
            seed_root=self.timing_corpus_seed_root,
            batch_id=corpus_batch_id,
        )
        shots = (
            batch_size * self.corpus_batches
            if self.corpus_shots_per_restart is None
            else max(batch_size, self.corpus_shots_per_restart)
        )
        sampler = prepared.circuit.compile_detector_sampler(seed=stim_seed)
        packed = np.asarray(
            sampler.sample(shots=shots, bit_packed=True),
            dtype=np.uint8,
        )
        packed = np.ascontiguousarray(packed)
        packed.setflags(write=False)

        # Compile all three production paths independently.  U0-direct and the
        # backend diagnostic use this ordinary uncorrelated matcher.  PU owns
        # its separately compiled but equivalent uncorrelated residual matcher.
        direct_matcher = pymatching.Matching.from_detector_error_model(prepared.dem)
        direct_call = _MatcherCall(direct_matcher, prepared.dem.num_observables)
        compiled_wrap = IdentityWrappedPyMatchingDecoder(
            domain_mode=decoder_config["domain_mode"]
        ).compile_decoder_for_dem(dem=prepared.dem)
        wrap_call = _IdentityWrapperCall(compiled_wrap)
        pu_call = _PromatchCall(compiled_pu)

        # Generate the backend's paired residual corpus via the production
        # non-retaining path.  No PrematchResult tuple survives construction.
        unpacked = np.unpackbits(
            packed,
            axis=1,
            count=prepared.dem.num_detectors,
            bitorder="little",
        )
        residual, frames, retained_results = compiled_pu._predecode_shots(
            unpacked,
            retain_results=False,
        )
        if retained_results:
            raise AssertionError("latency residual generation retained shot telemetry")
        del frames, retained_results, unpacked
        residual_packed = np.ascontiguousarray(
            np.packbits(residual, axis=1, bitorder="little")
        )
        del residual
        residual_packed.setflags(write=False)

        expected_prediction_shape = (
            shots,
            (prepared.dem.num_observables + 7) // 8,
        )
        direct_prediction = direct_call(packed)
        wrap_prediction = wrap_call(packed)
        pu_prediction = pu_call(packed)
        if direct_prediction.shape != expected_prediction_shape:
            raise AssertionError("U0-direct prediction shape mismatch")
        if wrap_prediction.shape != expected_prediction_shape:
            raise AssertionError("U0-wrap prediction shape mismatch")
        if pu_prediction.shape != expected_prediction_shape:
            raise AssertionError("PU prediction shape mismatch")
        if not np.array_equal(direct_prediction, wrap_prediction):
            raise AssertionError("U0-wrap differs from U0-direct")
        del direct_prediction, wrap_prediction, pu_prediction

        decoder_digest = _config_digest(decoder_config)
        provenance = {
            **prepared.provenance,
            "cell_id": self.cell_id,
            "experiment_id": manifest.get("experiment_id"),
            "decoder_config_sha256": decoder_digest,
            "decoder_config": decoder_config,
            "dem_options": dem_options,
            "restart_index": restart_index,
            "batch_size": batch_size,
            "corpus_batches": self.corpus_batches,
            "corpus_shots_per_restart": self.corpus_shots_per_restart,
            "complete_timing_batches": shots // batch_size,
            "corpus_batch_id": corpus_batch_id,
            "stim_seed": stim_seed,
            "timing_corpus_seed_root_sha256": hashlib.sha256(
                bytes.fromhex(self.timing_corpus_seed_root)
            ).hexdigest(),
            "u0_backend": "pymatching-uncorrelated",
            "residual_backend": "pymatching-uncorrelated",
            "correlated_matching": False,
            "residual_generation_retained_shot_telemetry": False,
        }
        return LatencyWorkload(
            total_corpus=packed,
            u0_direct=direct_call,
            u0_wrap=wrap_call,
            pu_window=pu_call,
            backend_original_corpus=packed,
            backend_residual_corpus=residual_packed,
            backend_original=direct_call,
            backend_residual=direct_call,
            provenance=provenance,
        )
