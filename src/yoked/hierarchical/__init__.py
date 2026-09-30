"""Hierarchical L1/L2 decoding of the 1D yoked surface code.

See docs/superpowers/specs/2026-09-14-hierarchical-l1-l2-design.md.
"""
from yoked.hierarchical._baselines import (
    BASELINE_DECODERS, RecordedBaselines, attach_baselines, load_recorded_baselines,
)
from yoked.hierarchical._calibration import CLIP, IsotonicCalibrator
from yoked.hierarchical._cluster_gap import ClusterGapResult, ClusterGapUnionFindDecoder
from yoked.hierarchical._correlated_matching_gap import ComplementaryGapResult, CorrelatedMatchingGapDecoder
from yoked.hierarchical._correlated_uf import (
    CORRELATED_UF_CONFIGURATION, CorrelatedUFDecodeResult, CorrelatedUFHierarchicalDecoder,
)
from yoked.hierarchical._correlated_uf_experiment import stage_correlated_uf
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
from yoked.hierarchical._outer_mwpm import mwpm_outer_map, mwpm_outer_map_batch, mwpm_outer_log_odds_batch
from yoked.hierarchical._paper_decoder import (
    PAPER_CONFIGURATION, PaperDecodeResult, PaperHierarchicalDecoder, SinterPaperHierarchicalDecoder,
)
from yoked.hierarchical._paper_stage import stage_paper_decode
from yoked.hierarchical._mpp import MPP_CONFIGURATION, MppDecodeResult, MppHierarchicalDecoder
from yoked.hierarchical._mpp_experiment import stage_mpp_compare
from yoked.hierarchical._patch_graphs import NUM_SECTORS, PatchGraph, PatchGraphs
from yoked.hierarchical._policies import NoRefinement, RefineAll, policy_from_name
from yoked.hierarchical._record import L1Record, LoadedRecord, by_sector, load_record, to_columns
from yoked.hierarchical._replay import (
    Calibrators, Estimator, ReplayConfig, ReplayResult, WorkCounts, calibrated_probabilities,
    estimator_scores, fit_calibrators, replay, residual_errors,
)
from yoked.hierarchical._reproduction import SubsetCheck, subset_reproduction
from yoked.hierarchical._stages import (
    CollectRequest, config_directory_name, load_calibrators, parse_config, stage_calibrate,
    stage_collect, stage_import_baselines, stage_replay, stage_summarize, stage_verify_subset,
)

__all__ = [
    'NUM_SECTORS', 'PatchGraph', 'PatchGraphs',
    'CLIP', 'IsotonicCalibrator',
    'ClusterGapResult', 'ClusterGapUnionFindDecoder',
    'ComplementaryGapResult', 'CorrelatedMatchingGapDecoder',
    'CORRELATED_UF_CONFIGURATION', 'CorrelatedUFDecodeResult', 'CorrelatedUFHierarchicalDecoder',
    'stage_correlated_uf',
    'PAPER_CONFIGURATION', 'PaperDecodeResult', 'PaperHierarchicalDecoder', 'SinterPaperHierarchicalDecoder',
    'stage_paper_decode',
    'MPP_CONFIGURATION', 'MppDecodeResult', 'MppHierarchicalDecoder', 'stage_mpp_compare',
    'CircuitParameters', 'SampleSet', 'L1Context', 'CollectedRows', 'CollectionWork', 'collect_rows',
    'ROLES', 'CollectionSettings', 'collect_sample',
    'GraphChecks', 'RecordChecks', 'check_graphs', 'check_record',
    'ForcedWeights', 'MatchingGaps', 'signed_gaps',
    'TIE_TOLERANCE', 'OuterDecision', 'BatchOuterDecision', 'exact_outer_map', 'exact_outer_map_batch',
    'mwpm_outer_map', 'mwpm_outer_map_batch',
    'mwpm_outer_log_odds_batch',
    'frame_adjusted_syndrome',
    'L1Record', 'LoadedRecord', 'by_sector', 'load_record', 'to_columns',
    'BASELINE_DECODERS', 'RecordedBaselines', 'load_recorded_baselines', 'attach_baselines',
    'SubsetCheck', 'subset_reproduction',
    'NoRefinement', 'RefineAll', 'policy_from_name',
    'Estimator', 'ReplayConfig', 'ReplayResult', 'WorkCounts', 'Calibrators',
    'fit_calibrators', 'calibrated_probabilities', 'estimator_scores', 'residual_errors', 'replay',
    'Rate', 'PairedDifference', 'paired_bootstrap', 'normalized_ler',
    'summarize_result', 'compare_endpoints',
    'CollectRequest', 'stage_collect', 'stage_verify_subset', 'stage_import_baselines',
    'stage_calibrate', 'load_calibrators', 'parse_config', 'config_directory_name',
    'stage_replay', 'stage_summarize',
]
