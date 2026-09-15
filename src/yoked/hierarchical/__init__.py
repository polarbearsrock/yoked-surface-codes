"""Hierarchical L1/L2 decoding of the 1D yoked surface code.

See docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md.
"""
from yoked.hierarchical._calibration import CLIP, IsotonicCalibrator
from yoked.hierarchical._cluster_gap import ClusterGapResult, ClusterGapUnionFindDecoder
from yoked.hierarchical._collect import (
    ROLES, CircuitParameters, CollectionSettings, GraphChecks, RecordChecks, SampleSet,
    check_graphs, check_record, collect_sample,
)
from yoked.hierarchical._l1 import CollectedRows, CollectionWork, L1Context, collect_rows
from yoked.hierarchical._matching_gaps import ForcedWeights, MatchingGaps, signed_gaps
from yoked.hierarchical._metrics import (
    PairedDifference, Rate, compare_endpoints, normalized_ler, paired_bootstrap, summarize_result,
)
from yoked.hierarchical._outer_decoder import (
    TIE_TOLERANCE, BatchOuterDecision, OuterDecision, exact_outer_map, exact_outer_map_batch,
    frame_adjusted_syndrome,
)
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph, PatchGraphs
from yoked.hierarchical._policies import NoRefinement, RefineAll, policy_from_name
from yoked.hierarchical._record import L1Record, LoadedRecord, by_sector, load_record, to_columns
from yoked.hierarchical._replay import (
    Calibrators, Estimator, ReplayConfig, ReplayResult, WorkCounts, calibrated_probabilities,
    estimator_scores, fit_calibrators, replay, residual_errors,
)
from yoked.hierarchical._stages import (
    CollectRequest, config_directory_name, load_calibrators, parse_config, stage_calibrate,
    stage_collect, stage_replay, stage_summarize,
)

__all__ = [
    'NUM_SECTORS', 'PatchGraph', 'PatchGraphs',
    'CLIP', 'IsotonicCalibrator',
    'ClusterGapResult', 'ClusterGapUnionFindDecoder',
    'CircuitParameters', 'SampleSet', 'L1Context', 'CollectedRows', 'CollectionWork', 'collect_rows',
    'ROLES', 'CollectionSettings', 'collect_sample',
    'GraphChecks', 'RecordChecks', 'check_graphs', 'check_record',
    'ForcedWeights', 'MatchingGaps', 'signed_gaps',
    'TIE_TOLERANCE', 'OuterDecision', 'BatchOuterDecision', 'exact_outer_map', 'exact_outer_map_batch',
    'frame_adjusted_syndrome',
    'L1Record', 'LoadedRecord', 'by_sector', 'load_record', 'to_columns',
    'NoRefinement', 'RefineAll', 'policy_from_name',
    'Estimator', 'ReplayConfig', 'ReplayResult', 'WorkCounts', 'Calibrators',
    'fit_calibrators', 'calibrated_probabilities', 'estimator_scores', 'residual_errors', 'replay',
    'Rate', 'PairedDifference', 'paired_bootstrap', 'normalized_ler',
    'summarize_result', 'compare_endpoints',
    'CollectRequest', 'stage_collect', 'stage_calibrate', 'load_calibrators',
    'parse_config', 'config_directory_name', 'stage_replay', 'stage_summarize',
]
