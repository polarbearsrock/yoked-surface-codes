"""Decoders and decoder-building utilities for yoked surface codes.

The package groups four layers of the ProMatch-L1 experiment:

* Layout/graph compilation (``compile_layout``, ``compile_matching_graph``)
  and the sinter-facing decoders (``PromatchDecoder``,
  ``IdentityWrappedPyMatchingDecoder``); ``custom_decoders`` is the
  ``yoked.decoding:custom_decoders`` sinter entry point.
* The deterministic predecoding core (``predecode``,
  ``DomainProposalStepper``, ``PrematchResult``,
  ``apply_detector_boundary``).
* The paired fixed-shot experiment harness (``run_collection``,
  ``freeze_protocol``, ``analyze_summary``, ...) and its statistics helpers
  (``PairedContingency``, ``tango_paired_risk_difference_upper``, ...).
* The latency benchmark (``run_latency_benchmark``,
  ``analyze_latency_suite``, ``YokedPromatchLatencyFactory``, ...).

Not eagerly imported: the diagnostic full-graph oracle and policy-audit
experiments live in :mod:`yoked.decoding.oracle` (import its submodules
directly; some lazily pull in Matplotlib on rendering paths), and the
JSON-artifact I/O infrastructure shared by the ``tools/`` scripts lives in
:mod:`yoked.decoding._artifact_io`.
"""

import sinter

from yoked.decoding._promatch import (
    CommitProposal,
    DomainPrematchStats,
    DomainProposalStepper,
    DomainStepOutcome,
    FallbackReason,
    PrematchResult,
    PrematchedPath,
    apply_detector_boundary,
    predecode,
)
from yoked.decoding._promatch_analysis import (
    analyze_summary,
    construct_confirmatory_draft_from_pilot,
    load_verified_summary,
    render_markdown,
    validate_generated_analysis_artifact,
)
from yoked.decoding._promatch_decoder import (
    IdentityWrappedPyMatchingDecoder,
    PromatchDecoder,
)
from yoked.decoding._promatch_experiment import (
    PreparedCell,
    default_smoke_protocol,
    freeze_protocol,
    inspect_protocol,
    normalize_protocol,
    prepare_cell,
    run_collection,
    validate_experiment_protocol,
)
from yoked.decoding._promatch_graph import (
    CompiledPromatchGraph,
    DomainGraph,
    Edge,
    compile_matching_graph,
)
from yoked.decoding._promatch_latency import run_latency_benchmark
from yoked.decoding._promatch_latency_analysis import (
    TinyLatencyAnalysisConfig,
    analyze_latency_suite,
    render_latency_markdown,
)
from yoked.decoding._promatch_latency_integration import (
    TinyLatencySmokeConfig,
    YokedPromatchLatencyFactory,
    latency_protocol_from_manifest,
)
from yoked.decoding._promatch_layout import (
    DetectorRole,
    L1BodyDetector,
    L1DomainKey,
    L1FullHistoryDomain,
    L1TerminalDetector,
    L1WindowDomain,
    PromatchLayout,
    YokeDetector,
    compile_layout,
)
from yoked.decoding._promatch_stats import (
    ArrayDigest,
    PairedContingency,
    clopper_pearson_lower,
    clopper_pearson_upper,
    confirmatory_sample_size,
    derive_stim_batch_seed,
    digest_array,
    simulate_tango_noninferiority_power,
    tango_paired_risk_difference_upper,
    validate_process_count,
)


def custom_decoders() -> dict[str, sinter.Decoder]:
    """Returns the frozen first-round ProMatch decoder configurations."""

    return {
        "promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching": PromatchDecoder(
            residual_hw_limit=10,
            domain_mode="windowd",
            boundary_policy="disabled",
            observable_policy="zero-frame",
        ),
        "promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching": PromatchDecoder(
            residual_hw_limit=10,
            domain_mode="windowd",
            boundary_policy="odd-parity",
            observable_policy="zero-frame",
        ),
        "promatch-l1-v1-fullhistory-hw10-stages1234-noboundary-zeroframe-pymatching": PromatchDecoder(
            residual_hw_limit=10,
            domain_mode="fullhistory",
            boundary_policy="disabled",
            observable_policy="zero-frame",
        ),
        "pymatching-u0-wrap-v1-windowd": IdentityWrappedPyMatchingDecoder(
            domain_mode="windowd"
        ),
    }


__all__ = [
    "ArrayDigest",
    "CommitProposal",
    "CompiledPromatchGraph",
    "DetectorRole",
    "DomainGraph",
    "DomainPrematchStats",
    "DomainProposalStepper",
    "DomainStepOutcome",
    "Edge",
    "FallbackReason",
    "IdentityWrappedPyMatchingDecoder",
    "L1BodyDetector",
    "L1DomainKey",
    "L1FullHistoryDomain",
    "L1TerminalDetector",
    "L1WindowDomain",
    "PairedContingency",
    "PrematchResult",
    "PrematchedPath",
    "PreparedCell",
    "PromatchDecoder",
    "PromatchLayout",
    "TinyLatencyAnalysisConfig",
    "TinyLatencySmokeConfig",
    "YokeDetector",
    "YokedPromatchLatencyFactory",
    "analyze_latency_suite",
    "analyze_summary",
    "apply_detector_boundary",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "compile_layout",
    "compile_matching_graph",
    "confirmatory_sample_size",
    "construct_confirmatory_draft_from_pilot",
    "custom_decoders",
    "default_smoke_protocol",
    "derive_stim_batch_seed",
    "digest_array",
    "freeze_protocol",
    "inspect_protocol",
    "latency_protocol_from_manifest",
    "load_verified_summary",
    "normalize_protocol",
    "predecode",
    "prepare_cell",
    "render_latency_markdown",
    "render_markdown",
    "run_collection",
    "run_latency_benchmark",
    "simulate_tango_noninferiority_power",
    "tango_paired_risk_difference_upper",
    "validate_experiment_protocol",
    "validate_generated_analysis_artifact",
    "validate_process_count",
]
