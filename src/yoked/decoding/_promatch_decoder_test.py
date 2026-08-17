import dataclasses
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import pymatching
import sinter
import stim

import gen
from yoked.decoding import custom_decoders
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._promatch_decoder import (
    IdentityWrappedPyMatchingDecoder,
    PromatchDecoder,
    _pack_observable_frames,
    _validate_packed_input,
)


@pytest.mark.parametrize("value", [True, False, 1.5, "10"])
def test_promatch_decoder_rejects_noninteger_limit_before_compilation(value) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        PromatchDecoder(residual_hw_limit=value)


def _small_circuit(*, yokes=2, noise_strength=1e-3):
    return yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=12,
        noise=gen.NoiseModel.si1000(noise_strength),
        style="cz",
        yokes=yokes,
        num_patches=2,
    )


def _decode_with_factory_in_spawned_worker(
    decoder: sinter.Decoder,
    dem_text: str,
    packed_detection_events: np.ndarray,
) -> np.ndarray:
    """Top-level worker proving that a decoder factory crosses spawn pickling."""

    dem = stim.DetectorErrorModel(dem_text)
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    return compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_detection_events
    )


@pytest.mark.parametrize("yokes", [0, 1, 2])
def test_identity_wrapper_matches_direct_pymatching(yokes: int) -> None:
    circuit = _small_circuit(yokes=yokes)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    dets, _ = circuit.compile_detector_sampler(seed=1234).sample(
        shots=40,
        bit_packed=True,
        separate_observables=True,
    )
    wrapped = IdentityWrappedPyMatchingDecoder().compile_decoder_for_dem(dem=dem)
    actual = wrapped.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets,
    )
    matcher = pymatching.Matching.from_detector_error_model(dem)
    matcher.ensure_num_fault_ids(dem.num_observables)
    expected = matcher.decode_batch(
        dets,
        bit_packed_shots=True,
        bit_packed_predictions=True,
    )
    np.testing.assert_array_equal(actual, expected)

    # Sinter's implementation is the authoritative U0-direct baseline used by
    # collection. Comparing only with a directly constructed Matching object
    # would not detect an adapter-setting mismatch in Sinter.
    sinter_expected = sinter.BUILT_IN_DECODERS[
        "pymatching"
    ].compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )
    np.testing.assert_array_equal(actual, sinter_expected)


def test_explicit_correlated_two_pass_matches_sinter_builtin() -> None:
    circuit = _small_circuit(yokes=2, noise_strength=0.01)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    dets, _ = circuit.compile_detector_sampler(seed=4_289_011).sample(
        shots=80,
        bit_packed=True,
        separate_observables=True,
    )

    explicit = pymatching.Matching.from_detector_error_model(
        dem,
        enable_correlations=True,
    )
    explicit.ensure_num_fault_ids(dem.num_observables)
    explicit_predictions = explicit.decode_batch(
        dets,
        bit_packed_shots=True,
        bit_packed_predictions=True,
        enable_correlations=True,
    )
    builtin_predictions = sinter.BUILT_IN_DECODERS[
        "pymatching-correlated"
    ].compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )
    np.testing.assert_array_equal(explicit_predictions, builtin_predictions)


@pytest.mark.parametrize("yokes", [0, 1, 2])
def test_promatch_pass_through_does_not_mutate_input(yokes: int) -> None:
    circuit = _small_circuit(yokes=yokes)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    dets, _ = circuit.compile_detector_sampler(seed=5678).sample(
        shots=20,
        bit_packed=True,
        separate_observables=True,
    )
    before = dets.copy()
    compiled = PromatchDecoder(residual_hw_limit=10**9).compile_decoder_for_dem(
        dem=dem
    )
    actual = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets,
    )
    np.testing.assert_array_equal(dets, before)

    direct = pymatching.Matching.from_detector_error_model(dem)
    direct.ensure_num_fault_ids(dem.num_observables)
    expected = direct.decode_batch(
        dets,
        bit_packed_shots=True,
        bit_packed_predictions=True,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("yokes", [0, 1, 2])
def test_active_real_dem_preserves_global_gf2_algebra(yokes: int) -> None:
    circuit = _small_circuit(yokes=yokes, noise_strength=0.02)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    packed, _ = circuit.compile_detector_sampler(seed=10_000 + yokes).sample(
        shots=24,
        bit_packed=True,
        separate_observables=True,
    )
    unpacked = np.unpackbits(
        packed,
        axis=1,
        count=dem.num_detectors,
        bitorder="little",
    )
    compiled = PromatchDecoder(residual_hw_limit=0).compile_decoder_for_dem(dem=dem)
    residual, frames, results = compiled.predecode_shots(unpacked)

    assert len(results) == len(unpacked)
    assert any(result.paths for result in results)
    edge_by_id = {edge.edge_id: edge for edge in compiled.graph.edges}
    protected = (
        *compiled.graph.layout.terminal_detector_ids,
        *compiled.graph.layout.yoke_detector_ids,
    )
    for shot, residual_shot, frame, result in zip(
        unpacked, residual, frames, results
    ):
        boundary = np.zeros(dem.num_detectors, dtype=np.uint8)
        for path in result.paths:
            for edge_id in path.edge_ids:
                edge = edge_by_id[edge_id]
                boundary[edge.source] ^= 1
                if edge.target is not None:
                    boundary[edge.target] ^= 1
        np.testing.assert_array_equal(shot ^ residual_shot, boundary)
        np.testing.assert_array_equal(residual_shot[list(protected)], shot[list(protected)])
        assert not np.any(frame)


def test_production_adapter_calls_residual_matcher_once_and_retains_no_results() -> None:
    circuit = _small_circuit(noise_strength=0.01)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    packed, _ = circuit.compile_detector_sampler(seed=2468).sample(
        shots=12,
        bit_packed=True,
        separate_observables=True,
    )
    compiled = PromatchDecoder(residual_hw_limit=0).compile_decoder_for_dem(dem=dem)

    class CountingMatcher:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def decode_batch(self, *args, **kwargs):
            self.calls += 1
            return self.inner.decode_batch(*args, **kwargs)

    matcher = CountingMatcher(compiled.graph.matcher)
    compiled.graph = dataclasses.replace(compiled.graph, matcher=matcher)
    output = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed,
    )
    assert matcher.calls == 1
    assert output.shape == (12, (dem.num_observables + 7) // 8)


def test_noncontiguous_packed_batch_is_supported() -> None:
    circuit = _small_circuit()
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    packed, _ = circuit.compile_detector_sampler(seed=97531).sample(
        shots=10,
        bit_packed=True,
        separate_observables=True,
    )
    storage = np.zeros((len(packed), packed.shape[1] * 2), dtype=np.uint8)
    storage[:, ::2] = packed
    view = storage[:, ::2]
    assert not view.flags.c_contiguous
    compiled = PromatchDecoder(residual_hw_limit=10**9).compile_decoder_for_dem(dem=dem)
    actual = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=view)
    direct = pymatching.Matching.from_detector_error_model(dem)
    direct.ensure_num_fault_ids(dem.num_observables)
    expected = direct.decode_batch(
        packed,
        bit_packed_shots=True,
        bit_packed_predictions=True,
    )
    np.testing.assert_array_equal(actual, expected)


def test_promatch_empty_batch_shape() -> None:
    circuit = _small_circuit()
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    compiled = PromatchDecoder().compile_decoder_for_dem(dem=dem)
    dets = np.zeros((0, (dem.num_detectors + 7) // 8), dtype=np.uint8)
    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets,
    )
    assert result.dtype == np.uint8
    assert result.shape == (0, (dem.num_observables + 7) // 8)


def test_identity_wrapper_empty_batch_shape() -> None:
    circuit = _small_circuit()
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    compiled = IdentityWrappedPyMatchingDecoder().compile_decoder_for_dem(dem=dem)
    dets = np.zeros((0, (dem.num_detectors + 7) // 8), dtype=np.uint8)
    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets,
    )
    assert result.dtype == np.uint8
    assert result.shape == (0, (dem.num_observables + 7) // 8)


def test_custom_decoder_names_are_frozen() -> None:
    assert set(custom_decoders()) == {
        "promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching",
        "promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching",
        "promatch-l1-v1-fullhistory-hw10-stages1234-noboundary-zeroframe-pymatching",
        "pymatching-u0-wrap-v1-windowd",
    }


def test_decoder_factories_pickle_and_primary_factory_decodes_under_spawn() -> None:
    factories = custom_decoders()
    round_tripped = pickle.loads(pickle.dumps(factories))
    assert set(round_tripped) == set(factories)

    circuit = _small_circuit(yokes=2)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    dets, _ = circuit.compile_detector_sampler(seed=9_177_331).sample(
        shots=8,
        bit_packed=True,
        separate_observables=True,
    )
    name = (
        "promatch-l1-v1-windowd-hw10-stages1234-"
        "noboundary-zeroframe-pymatching"
    )
    decoder = round_tripped[name]
    expected = decoder.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )

    # This is a one-worker unit smoke, not a simulation experiment. It uses the
    # strictest start method so the decoder object must actually be pickled.
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        actual = pool.submit(
            _decode_with_factory_in_spawned_worker,
            decoder,
            str(dem),
            dets,
        ).result(timeout=30)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bit_count", [0, 1, 7, 8, 9, 16])
def test_packing_shapes_bit_order_and_unused_bits(bit_count: int) -> None:
    frames = np.zeros((3, bit_count), dtype=np.uint8)
    if bit_count:
        frames[0, 0] = 1
        frames[1, bit_count - 1] = 1
    packed = _pack_observable_frames(frames, num_observables=bit_count)
    assert packed.dtype == np.uint8
    assert packed.shape == (3, (bit_count + 7) // 8)
    unpacked = np.unpackbits(
        packed,
        axis=1,
        count=bit_count,
        bitorder="little",
    )
    np.testing.assert_array_equal(unpacked, frames)
    if bit_count % 8 and packed.shape[1]:
        assert np.all(packed[:, -1] >> (bit_count % 8) == 0)


@pytest.mark.parametrize("detector_count", [0, 1, 7, 8, 9, 16])
def test_packed_detector_width_validation(detector_count: int) -> None:
    value = np.zeros((2, (detector_count + 7) // 8), dtype=np.uint8)
    assert _validate_packed_input(value, num_detectors=detector_count) is value
    with pytest.raises(ValueError, match="width mismatch"):
        _validate_packed_input(
            np.zeros((2, value.shape[1] + 1), dtype=np.uint8),
            num_detectors=detector_count,
        )
