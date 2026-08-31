import dataclasses
import gc
import pickle
import weakref
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from yoked.decoding import _patch_uf_decoder as decoder
from yoked.decoding._patch_uf import BudgetLimits, UFPolicy
from yoked.decoding._patch_uf_decoder import (
    AdapterControlDecoder,
    CaptureMode,
    CompiledAdapterControlDecoder,
    CompiledGlobalMWPMDecoder,
    CompiledPatchUFTreatmentDecoder,
    CompiledUFShadowDecoder,
    GlobalMWPMDecoder,
    PatchUFTreatmentDecoder,
    UFShadowDecoder,
    compile_patch_uf_lane_graphs,
)
from yoked.decoding._patch_uf_graph import (
    ExactDyadic,
    PatchUFLaneKey,
    PatchUFLaneProjection,
    PatchUFProjection,
    PatchUFSupportEdge,
    TrueBoundaryIncidence,
)


def _limits(value: int | None) -> BudgetLimits:
    return BudgetLimits(
        growth_event_count=value,
        simultaneous_event_batch_count=value,
        union_attempt_count=value,
        successful_union_count=value,
        forest_edge_count=value,
        absorbed_vertex_count=value,
        peel_operation_count=value,
        heap_push_count=value,
        heap_pop_count=value,
        heap_operation_count=value,
        peak_heap_size=value,
        temporary_memory_units=value,
    )


def _policy(*, limit: int | None = None) -> UFPolicy:
    limits = _limits(limit)
    return UFPolicy(tau=Fraction(0), semantic_limits=limits, production_limits=limits)


def _lane(lane_id: int, detector_id: int, edge_id: int) -> PatchUFLaneProjection:
    return PatchUFLaneProjection(
        lane_id=lane_id,
        key=PatchUFLaneKey(0, "X" if lane_id == 0 else "Z"),
        global_detector_ids=(detector_id,),
        local_x2=(0,),
        y2=(0,),
        times=(0,),
        internal_correction_edges=(),
        true_boundary_edges=(TrueBoundaryIncidence(edge_id, 0, 0),),
        guard_ports=(),
        incidences=(),
        incidence_offsets=(0, 0),
        incidence_indices=(),
    )


def _projection() -> PatchUFProjection:
    lanes = (_lane(0, 0, 0), _lane(1, 1, 1))
    support = (
        PatchUFSupportEdge(0, 0, None, b"\x00", 0, "local-correction", 0),
        PatchUFSupportEdge(1, 1, None, b"\x00", 0, "local-correction", 1),
    )
    return PatchUFProjection(
        canonical_graph_fingerprint="graph",
        validated_catalog_fingerprint="catalog",
        num_detectors=3,
        num_observables=3,
        num_patches=1,
        lanes=lanes,
        patch_lane_ids=((0, 1),),
        detector_lane_id=(0, 1, None),
        detector_local_index=(0, 0, None),
        detector_role_kind=("body", "terminal", "yoke"),
        detector_lane_id_array=(0, 1, -1),
        detector_local_index_array=(0, 0, -1),
        support_edges=support,
        edge_owner_kind=("local-correction", "local-correction"),
        edge_owner_lane=(0, 1),
        edge_owner_lane_array=(0, 1),
        exact_weights=(ExactDyadic(1, 0),),
        fingerprint="projection",
    )


class _MatcherSpy:
    def __init__(self, *, mutate: bool = False) -> None:
        self.calls = 0
        self.inputs = []
        self.mutate = mutate

    def decode_batch(self, shots, **kwargs):
        self.calls += 1
        assert kwargs == {
            "bit_packed_shots": True,
            "bit_packed_predictions": True,
        }
        self.inputs.append(np.array(shots, copy=True))
        if self.mutate:
            shots[...] = 0xFF
        return np.full((len(shots), 1), 0xFF, dtype=np.uint8)


def _compiled(compiled_type, matcher, *, policy=None):
    projection = _projection()
    if compiled_type is CompiledGlobalMWPMDecoder:
        return compiled_type(
            graph=SimpleNamespace(matcher=matcher),
            num_detectors=projection.num_detectors,
            num_observables=projection.num_observables,
        )
    return compiled_type(
        graph=SimpleNamespace(matcher=matcher),
        projection=projection,
        compiled_lanes=compile_patch_uf_lane_graphs(
            projection, _policy() if policy is None else policy
        ),
        policy=_policy() if policy is None else policy,
        num_detectors=projection.num_detectors,
        num_observables=projection.num_observables,
    )


def test_compile_time_lane_adapter_preserves_canonical_ids_and_exact_weights() -> None:
    policy = _policy()
    compiled = compile_patch_uf_lane_graphs(_projection(), policy)

    assert tuple(value.lane_id for value in compiled) == (0, 1)
    assert tuple(value.graph.num_vertices for value in compiled) == (1, 1)
    assert tuple(edge.edge_id for edge in compiled[0].graph.edges) == (0,)
    assert compiled[0].graph.edges[0].kind == "boundary"
    assert compiled[0].graph.edges[0].weight == 1
    assert compiled[0].engine.graph is compiled[0].graph
    assert compiled[0].engine.policy == policy


def test_direct_control_shadow_and_no_durable_treatment_bitmatch_once_per_batch(
    monkeypatch,
) -> None:
    packed_storage = np.zeros((4, 2), dtype=np.uint8)
    packed = packed_storage[:, ::2]
    assert not packed.flags.c_contiguous
    before = packed.copy()

    direct_spy = _MatcherSpy()
    direct = _compiled(CompiledGlobalMWPMDecoder, direct_spy)
    expected = direct.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    assert direct_spy.calls == 1

    for compiled_type in (
        CompiledAdapterControlDecoder,
        CompiledUFShadowDecoder,
        CompiledPatchUFTreatmentDecoder,
    ):
        spy = _MatcherSpy()
        compiled = _compiled(compiled_type, spy)
        if compiled_type is CompiledUFShadowDecoder:
            with monkeypatch.context() as scoped:
                scoped.setattr(
                    decoder,
                    "apply_shot_correction",
                    lambda *_args, **_kwargs: pytest.fail(
                        "shadow must not construct or apply a residual"
                    ),
                )
                actual, telemetry = compiled.decode_shots_bit_packed_with_telemetry(
                    bit_packed_detection_event_data=packed
                )
        else:
            actual, telemetry = compiled.decode_shots_bit_packed_with_telemetry(
                bit_packed_detection_event_data=packed
            )
        np.testing.assert_array_equal(actual, expected)
        assert spy.calls == 1
        assert len(telemetry) == len(packed)
        assert all(len(item.lane_outcomes) == 2 for item in telemetry)
        assert all(item.original_detector_count == 0 for item in telemetry)
        assert all(item.residual_detector_count == 0 for item in telemetry)

    np.testing.assert_array_equal(packed, before)


@pytest.mark.parametrize(
    "compiled_type",
    (
        CompiledAdapterControlDecoder,
        CompiledUFShadowDecoder,
        CompiledPatchUFTreatmentDecoder,
    ),
)
def test_none_capture_streams_and_releases_each_correction_before_next_shot(
    compiled_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    packed = np.array([[0b011], [0], [0b001], [0b010]], dtype=np.uint8)
    retained_spy = _MatcherSpy()
    retained = _compiled(compiled_type, retained_spy)
    expected, telemetry = retained.decode_shots_bit_packed_with_capture(
        bit_packed_detection_event_data=packed,
        capture=CaptureMode.METRICS,
    )
    assert len(telemetry) == len(packed)
    assert retained_spy.calls == 1

    streaming_spy = _MatcherSpy()
    streaming = _compiled(compiled_type, streaming_spy)
    original_plan = streaming.plan_shot
    correction_refs: list[weakref.ReferenceType[object]] = []

    def recording_plan(
        shot: np.ndarray, *, capture: CaptureMode
    ) -> object:
        assert capture is CaptureMode.NONE
        gc.collect()
        assert all(reference() is None for reference in correction_refs)
        correction = original_plan(shot, capture=capture)
        correction_refs.append(weakref.ref(correction))
        return correction

    monkeypatch.setattr(streaming, "plan_shot", recording_plan)
    original_backend = streaming_spy.decode_batch

    def checking_backend(shots: np.ndarray, **kwargs: object) -> np.ndarray:
        gc.collect()
        assert all(reference() is None for reference in correction_refs)
        return original_backend(shots, **kwargs)

    monkeypatch.setattr(streaming_spy, "decode_batch", checking_backend)
    actual, corrections = streaming.decode_shots_bit_packed_with_capture(
        bit_packed_detection_event_data=packed,
        capture=CaptureMode.NONE,
    )

    assert corrections == ()
    assert len(correction_refs) == len(packed)
    gc.collect()
    assert all(reference() is None for reference in correction_refs)
    assert streaming_spy.calls == 1
    np.testing.assert_array_equal(streaming_spy.inputs, retained_spy.inputs)
    np.testing.assert_array_equal(actual, expected)


def test_treatment_plans_patch_atomically_applies_support_and_retains_counts() -> None:
    spy = _MatcherSpy()
    compiled = _compiled(CompiledPatchUFTreatmentDecoder, spy)
    packed = np.array([[0b011]], dtype=np.uint8)

    residual, frames, corrections = compiled.precompute_residual_batch(
        packed, capture=CaptureMode.TRACE
    )
    correction = corrections[0]

    np.testing.assert_array_equal(residual, np.array([[0]], dtype=np.uint8))
    np.testing.assert_array_equal(frames, np.zeros((1, 1), dtype=np.uint8))
    assert spy.calls == 0
    assert correction.durable_support_edge_ids == (0, 1)
    assert correction.durable_detector_boundary == (0, 1)
    assert correction.durable_support_count == 2
    assert correction.durable_boundary_count == 2
    assert correction.durable_frame_weight == 0
    assert correction.original_detector_count == 2
    assert correction.lane_owned_detector_count == 2
    assert correction.residual_detector_count == 0
    assert correction.lane_original_detector_counts == (1, 1)
    assert correction.lane_residual_detector_counts == (0, 0)
    assert (
        correction.original_body_detector_count,
        correction.original_terminal_detector_count,
        correction.original_yoke_detector_count,
    ) == (1, 1, 0)
    assert (
        correction.residual_body_detector_count,
        correction.residual_terminal_detector_count,
        correction.residual_yoke_detector_count,
    ) == (0, 0, 0)
    assert correction.committed_defect_count == 2
    assert correction.cluster_summary_complete
    assert correction.completed_final_component_count == 2
    assert correction.completed_component_size_histogram == ((1, 2),)
    assert correction.maximum_final_component_defect_count == 1
    assert correction.component_durable_decision(0, 0) == (True, "committed")
    assert correction.patch_outcomes[0].status == "durable"

    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    assert spy.calls == 1
    np.testing.assert_array_equal(spy.inputs[0], np.array([[0]], dtype=np.uint8))
    np.testing.assert_array_equal(predictions, np.array([[0b111]], dtype=np.uint8))


def test_censored_sibling_aborts_entire_patch_and_treatment_bitmatches_direct() -> None:
    policy = _policy(limit=0)
    direct_spy = _MatcherSpy()
    treatment_spy = _MatcherSpy()
    direct = _compiled(CompiledGlobalMWPMDecoder, direct_spy)
    treatment = _compiled(
        CompiledPatchUFTreatmentDecoder, treatment_spy, policy=policy
    )
    packed = np.array([[0b011]], dtype=np.uint8)

    expected = direct.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    actual, corrections = treatment.decode_shots_bit_packed_with_telemetry(
        bit_packed_detection_event_data=packed
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(treatment_spy.inputs[0], packed)
    correction = corrections[0]
    assert correction.patch_outcomes[0].status == "aborted"
    assert correction.patch_outcomes[0].abort_reason == (
        "budget-exhaustion-patch-abort"
    )
    assert correction.durable_support_edge_ids == ()
    assert correction.residual_detector_count == correction.original_detector_count
    assert correction.lane_original_detector_counts == (1, 1)
    assert correction.lane_residual_detector_counts == (1, 1)
    assert not correction.cluster_summary_complete
    assert correction.maximum_final_component_defect_count is None


def test_capture_modes_share_identical_immutable_core_outcomes() -> None:
    compiled = _compiled(CompiledPatchUFTreatmentDecoder, _MatcherSpy())
    shot = np.array([1, 1, 0], dtype=np.uint8)

    corrections = tuple(
        compiled.plan_shot(shot, capture=mode) for mode in CaptureMode
    )

    semantic = [
        dataclasses.replace(
            value,
            capture=CaptureMode.NONE,
            lane_original_detector_counts=None,
            lane_residual_detector_counts=None,
            original_body_detector_count=None,
            residual_body_detector_count=None,
            original_terminal_detector_count=None,
            residual_terminal_detector_count=None,
            original_yoke_detector_count=None,
            residual_yoke_detector_count=None,
        )
        for value in corrections
    ]
    assert semantic[0] == semantic[1] == semantic[2]


def test_strict_packing_empty_scalar_tail_output_and_input_immutability() -> None:
    spy = _MatcherSpy(mutate=True)
    compiled = _compiled(CompiledGlobalMWPMDecoder, spy)

    empty = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.zeros((0, 1), dtype=np.uint8)
    )
    assert empty.shape == (0, 1)
    assert spy.calls == 0

    row = np.array([0b001], dtype=np.uint8)
    before = row.copy()
    prediction = compiled.decode_shot_bit_packed(row)
    np.testing.assert_array_equal(row, before)
    np.testing.assert_array_equal(prediction, np.array([0b111], dtype=np.uint8))
    assert spy.calls == 1

    with pytest.raises(ValueError, match="dtype uint8"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, 1), dtype=np.int8)
        )
    with pytest.raises(ValueError, match="must be 2-D"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros(1, dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="width mismatch"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, 2), dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="unused tail bits"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.array([[0b1000]], dtype=np.uint8)
        )
    with pytest.raises(TypeError, match="actual_observables"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, 1), dtype=np.uint8),
            actual_observables=np.zeros((1, 1), dtype=np.uint8),
        )


def test_prevalidated_backend_helper_is_exactly_one_matcher_invocation() -> None:
    spy = _MatcherSpy()
    compiled = _compiled(CompiledUFShadowDecoder, spy)
    packed = np.zeros((2, 1), dtype=np.uint8)

    raw = compiled.invoke_backend_prevalidated(packed)

    assert spy.calls == 1
    np.testing.assert_array_equal(raw, np.full((2, 1), 0xFF, dtype=np.uint8))


def test_factories_are_top_level_pickleable_and_require_explicit_policy() -> None:
    policy = _policy()
    factories = (
        GlobalMWPMDecoder(),
        AdapterControlDecoder(policy),
        UFShadowDecoder(policy),
        PatchUFTreatmentDecoder(policy),
    )
    for factory in factories:
        assert pickle.loads(pickle.dumps(factory)) == factory

    with pytest.raises(TypeError):
        AdapterControlDecoder()
    with pytest.raises(TypeError):
        UFShadowDecoder()
    with pytest.raises(TypeError):
        PatchUFTreatmentDecoder()

    assert decoder.PATCH_UF_TREATMENT_DECODER_NAME == (
        "weighted-uf-fullhistory-patchlocal-zeroframe-residual-global-mwpm-v1"
    )
    assert decoder.PATCH_UF_V1_POLICY.tau == 0
    assert decoder.PATCH_UF_V1_POLICY.semantic_limits.growth_event_count == 3_403
    assert decoder.PATCH_UF_V1_POLICY.production_limits.heap_operation_count == (
        8_388_608
    )
