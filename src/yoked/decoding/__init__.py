"""Decoders and decoder-building utilities for yoked surface codes."""

import sinter

from yoked.decoding._promatch_decoder import (
    IdentityWrappedPyMatchingDecoder,
    PromatchDecoder,
)

from yoked.decoding._promatch_graph import (
    CompiledPromatchGraph,
    DomainGraph,
    Edge,
    compile_matching_graph,
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
    "CompiledPromatchGraph",
    "DetectorRole",
    "DomainGraph",
    "Edge",
    "L1BodyDetector",
    "L1DomainKey",
    "L1FullHistoryDomain",
    "L1TerminalDetector",
    "L1WindowDomain",
    "IdentityWrappedPyMatchingDecoder",
    "PromatchDecoder",
    "PromatchLayout",
    "YokeDetector",
    "compile_layout",
    "compile_matching_graph",
    "custom_decoders",
]
