import pickle
from types import SimpleNamespace

import numpy as np
import pymatching
import pytest

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding import (
    PINBALL_DECODER_NAME,
    PINBALL_V2_DECODER_NAME,
    CompiledPinballDecoder,
    IdentityWrappedPyMatchingDecoder,
    PinballDecoder,
    PinballV2Decoder,
    PromatchDecoder,
    custom_decoders,
)
from yoked.decoding import _pinball_decoder as pinball_decoder


def _circuit(*, rounds: int = 7, noise_strength: float = 0.01):
    return yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=rounds,
        noise=gen.NoiseModel.si1000(noise_strength),
        style="cz",
        yokes=2,
        num_patches=2,
    )


def _dem_and_shots(*, rounds: int = 7, shots: int = 40, seed: int = 1234):
    circuit = _circuit(rounds=rounds)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    packed, _ = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots,
        bit_packed=True,
        separate_observables=True,
    )
    return dem, packed


def _direct_predictions(dem, packed: np.ndarray) -> np.ndarray:
    matcher = pymatching.Matching.from_detector_error_model(dem)
    matcher.ensure_num_fault_ids(dem.num_observables)
    return matcher.decode_batch(
        packed,
        bit_packed_shots=True,
        bit_packed_predictions=True,
    )


def test_registry_adds_pinball_without_changing_frozen_promatch_factories() -> None:
    factories = custom_decoders()
    assert set(factories) == {
        "promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching",
        "promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching",
        "promatch-l1-v1-fullhistory-hw10-stages1234-noboundary-zeroframe-pymatching",
        "pymatching-u0-wrap-v1-windowd",
        PINBALL_DECODER_NAME,
        PINBALL_V2_DECODER_NAME,
    }
    assert factories[
        "promatch-l1-v1-windowd-hw10-stages1234-noboundary-zeroframe-pymatching"
    ] == PromatchDecoder(
        residual_hw_limit=10,
        domain_mode="windowd",
        boundary_policy="disabled",
        observable_policy="zero-frame",
    )
    assert factories[
        "promatch-l1-v1-windowd-hw10-stages1234-parityboundary-zeroframe-pymatching"
    ] == PromatchDecoder(
        residual_hw_limit=10,
        domain_mode="windowd",
        boundary_policy="odd-parity",
        observable_policy="zero-frame",
    )
    assert factories[
        "promatch-l1-v1-fullhistory-hw10-stages1234-noboundary-zeroframe-pymatching"
    ] == PromatchDecoder(
        residual_hw_limit=10,
        domain_mode="fullhistory",
        boundary_policy="disabled",
        observable_policy="zero-frame",
    )
    assert factories["pymatching-u0-wrap-v1-windowd"] == (
        IdentityWrappedPyMatchingDecoder(domain_mode="windowd")
    )
    assert factories[PINBALL_DECODER_NAME] == PinballDecoder()
    assert factories[PINBALL_V2_DECODER_NAME] == PinballV2Decoder()


def test_factory_is_parameter_free_picklable_and_supports_arbitrary_rounds() -> None:
    factory = pickle.loads(pickle.dumps(PinballDecoder()))
    assert factory == PinballDecoder()
    dem, _ = _dem_and_shots(rounds=5, shots=1)
    compiled = factory.compile_decoder_for_dem(dem=dem)
    assert compiled.schedule.graph_fingerprint == compiled.graph.fingerprint
    assert compiled.schedule.num_detectors == dem.num_detectors
    assert compiled.schedule.num_observables == dem.num_observables


def test_single_primitive_composes_residual_and_observable_frame() -> None:
    dem, _ = _dem_and_shots(shots=1)
    compiled = PinballDecoder().compile_decoder_for_dem(dem=dem)
    primitive = compiled.schedule.primitives[0]
    shot = np.zeros((1, dem.num_detectors), dtype=np.uint8)
    shot[0, primitive.source] = 1
    if primitive.target is not None:
        shot[0, primitive.target] = 1

    residual, frames, results = compiled.predecode_shots(shot)

    assert len(results) == 1
    assert not results[0].complex
    np.testing.assert_array_equal(residual, 0)
    expected_frame = np.unpackbits(
        np.frombuffer(
            compiled.graph.edges[primitive.edge_id].observable_mask,
            dtype=np.uint8,
        ),
        count=dem.num_observables,
        bitorder="little",
    )
    np.testing.assert_array_equal(frames[0], expected_frame)
    assert results[0].edge_support == (primitive.edge_id,)


def test_complex_rollback_adapter_is_bit_exact_with_direct_pymatching() -> None:
    circuit = _circuit(rounds=9, noise_strength=0.03)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    packed, _ = circuit.compile_detector_sampler(seed=8_675_309).sample(
        shots=128,
        bit_packed=True,
        separate_observables=True,
    )
    compiled = PinballDecoder().compile_decoder_for_dem(dem=dem)
    unpacked = np.unpackbits(
        packed,
        axis=1,
        count=dem.num_detectors,
        bitorder="little",
    )
    residual, frames, results = compiled.predecode_shots(unpacked)
    complex_indices = [k for k, result in enumerate(results) if result.complex]
    assert complex_indices
    np.testing.assert_array_equal(residual[complex_indices], unpacked[complex_indices])
    np.testing.assert_array_equal(frames[complex_indices], 0)

    complex_packed = packed[complex_indices]
    before = complex_packed.copy()
    actual = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=complex_packed,
    )
    np.testing.assert_array_equal(complex_packed, before)
    np.testing.assert_array_equal(actual, _direct_predictions(dem, complex_packed))


def test_noncontiguous_and_empty_packed_batches() -> None:
    dem, packed = _dem_and_shots(shots=12, seed=97531)
    compiled = PinballDecoder().compile_decoder_for_dem(dem=dem)
    storage = np.zeros((len(packed), packed.shape[1] * 2), dtype=np.uint8)
    storage[:, ::2] = packed
    view = storage[:, ::2]
    assert not view.flags.c_contiguous
    expected = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    before = view.copy()
    np.testing.assert_array_equal(
        compiled.decode_shots_bit_packed(bit_packed_detection_event_data=view),
        expected,
    )
    np.testing.assert_array_equal(view, before)

    empty = np.zeros((0, packed.shape[1]), dtype=np.uint8)
    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=empty,
    )
    assert result.dtype == np.uint8
    assert result.shape == (0, (dem.num_observables + 7) // 8)


def test_packed_input_dtype_rank_and_width_are_validated() -> None:
    compiled = CompiledPinballDecoder(
        graph=SimpleNamespace(matcher=None),
        schedule=None,
        num_detectors=9,
        num_observables=2,
    )
    with pytest.raises(ValueError, match="dtype uint8"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, 2), dtype=np.int8)
        )
    with pytest.raises(ValueError, match="must be 2-D"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros(2, dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="width mismatch"):
        compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, 1), dtype=np.uint8)
        )


def test_adapter_xors_frame_masks_high_bits_and_calls_matcher_once(monkeypatch) -> None:
    class CountingMatcher:
        def __init__(self) -> None:
            self.calls = 0

        def decode_batch(self, shots, **kwargs):
            self.calls += 1
            assert kwargs == {
                "bit_packed_shots": True,
                "bit_packed_predictions": True,
            }
            assert shots.shape == (2, 1)
            return np.array([[0x04, 0xFF], [0x08, 0xFC]], dtype=np.uint8)

    def fake_predecode(_graph, _schedule, shot):
        frame = np.zeros(10, dtype=np.uint8)
        frame[0] = shot[0]
        frame[9] = shot[1]
        return SimpleNamespace(
            residual_syndrome=shot.copy(),
            observable_frame=frame,
        )

    matcher = CountingMatcher()
    compiled = CompiledPinballDecoder(
        graph=SimpleNamespace(matcher=matcher),
        schedule=None,
        num_detectors=5,
        num_observables=10,
    )
    monkeypatch.setattr(pinball_decoder, "predecode_pinball", fake_predecode)
    packed = np.array([[0b00001], [0b00010]], dtype=np.uint8)
    before = packed.copy()
    actual = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed,
    )

    assert matcher.calls == 1
    np.testing.assert_array_equal(packed, before)
    np.testing.assert_array_equal(
        actual,
        np.array([[0x05, 0x03], [0x08, 0x02]], dtype=np.uint8),
    )


def test_predecode_shots_retains_results_but_batch_path_does_not(monkeypatch) -> None:
    sentinel = SimpleNamespace(
        residual_syndrome=np.array([1, 0], dtype=np.uint8),
        observable_frame=np.array([1], dtype=np.uint8),
    )
    monkeypatch.setattr(
        pinball_decoder,
        "predecode_pinball",
        lambda _graph, _schedule, _shot: sentinel,
    )
    compiled = CompiledPinballDecoder(
        graph=SimpleNamespace(matcher=None),
        schedule=None,
        num_detectors=2,
        num_observables=1,
    )
    shots = np.zeros((3, 2), dtype=np.uint8)
    _, _, retained = compiled.predecode_shots(shots)
    _, _, discarded = compiled._predecode_shots(shots, retain_results=False)
    assert retained == (sentinel, sentinel, sentinel)
    assert discarded == ()
