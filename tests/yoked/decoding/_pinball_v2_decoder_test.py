import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding import _pinball_v2_decoder as pinball_v2_decoder
from yoked.decoding._pinball_v2_decoder import (
    PINBALL_V2_DECODER_NAME,
    CompiledPinballV2Decoder,
    PinballV2Decoder,
)


def _dem_and_packed_shots(*, shots: int = 12, seed: int = 1234):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=3,
        noise=gen.NoiseModel.si1000(0.002),
        style="cz",
        yokes=2,
        num_patches=2,
    )
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


def _packed_frames(frames: np.ndarray, *, num_observables: int) -> np.ndarray:
    result = np.packbits(frames, axis=1, bitorder="little")
    if num_observables % 8 and result.shape[1]:
        result[:, -1] &= (1 << (num_observables % 8)) - 1
    return result


def test_versioned_factory_is_registerable_picklable_and_compiles_real_dem() -> None:
    assert PINBALL_V2_DECODER_NAME == (
        "pinball-ysc-v2-cz-fullhistory-nine-stage-domainatomic-yokeedge-pymatching"
    )
    factory = pickle.loads(pickle.dumps(PinballV2Decoder()))
    assert factory == PinballV2Decoder()

    dem, _ = _dem_and_packed_shots(shots=1)
    compiled = factory.compile_decoder_for_dem(dem=dem)

    assert compiled.num_detectors == dem.num_detectors
    assert compiled.num_observables == dem.num_observables
    assert compiled.schedule.graph_fingerprint == compiled.graph.fingerprint
    assert compiled.schedule.num_detectors == dem.num_detectors
    assert compiled.schedule.num_observables == dem.num_observables
    assert compiled.schedule.domains
    assert compiled.schedule.stages


def test_unmerged_catalog_rejects_parallel_frame_ambiguity() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.2) D0 L0
        detector(0, 0, 0) D0
        """
    )
    graph = SimpleNamespace(
        num_detectors=1,
        num_observables=1,
        edges=(
            SimpleNamespace(source=0, target=None, observable_mask=b"\x00"),
        ),
    )

    with pytest.raises(ValueError, match="ambiguous observable frames"):
        pinball_v2_decoder._validate_unmerged_dem_catalog(dem, graph)


def test_unmerged_catalog_preserves_separator_components_and_frames() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        """
    )
    graph = SimpleNamespace(
        num_detectors=2,
        num_observables=2,
        edges=(
            SimpleNamespace(source=0, target=None, observable_mask=b"\x01"),
            SimpleNamespace(source=1, target=None, observable_mask=b"\x02"),
        ),
    )

    pinball_v2_decoder._validate_unmerged_dem_catalog(dem, graph)


@pytest.mark.parametrize(
    ("dem_text", "graph", "message"),
    [
        (
            "error(0.1) L0",
            SimpleNamespace(num_detectors=0, num_observables=1, edges=()),
            "detector-free logical component",
        ),
        (
            "error(0.1) D0 D1 D2",
            SimpleNamespace(num_detectors=3, num_observables=0, edges=()),
            "at most two detectors",
        ),
        (
            "error(0.1) D0\ndetector(0, 0, 0) D0",
            SimpleNamespace(num_detectors=1, num_observables=0, edges=()),
            "does not preserve",
        ),
        (
            "error(0.1) D0 L0\ndetector(0, 0, 0) D0",
            SimpleNamespace(
                num_detectors=1,
                num_observables=1,
                edges=(
                    SimpleNamespace(
                        source=0,
                        target=None,
                        observable_mask=b"\x00",
                    ),
                ),
            ),
            "does not preserve",
        ),
    ],
)
def test_unmerged_catalog_rejects_unsafe_component_shapes(
    dem_text, graph, message
) -> None:
    dem = stim.DetectorErrorModel(dem_text)

    with pytest.raises(ValueError, match=message):
        pinball_v2_decoder._validate_unmerged_dem_catalog(dem, graph)


def test_real_batch_composes_global_residual_mwpm_and_retains_telemetry() -> None:
    dem, packed = _dem_and_packed_shots(shots=24, seed=8675309)
    compiled = PinballV2Decoder().compile_decoder_for_dem(dem=dem)
    unpacked = np.unpackbits(
        packed,
        axis=1,
        count=dem.num_detectors,
        bitorder="little",
    )

    residual, frames, results = compiled.predecode_shots(unpacked)
    residual_packed = np.packbits(residual, axis=1, bitorder="little")
    expected = np.asarray(
        compiled.graph.matcher.decode_batch(
            residual_packed,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        ),
        dtype=np.uint8,
    )
    expected ^= _packed_frames(frames, num_observables=dem.num_observables)
    if dem.num_observables % 8:
        expected[:, -1] &= (1 << (dem.num_observables % 8)) - 1

    before = packed.copy()
    actual, packed_results = compiled.decode_shots_bit_packed_with_telemetry(
        bit_packed_detection_event_data=packed
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(packed, before)
    assert len(results) == len(packed)
    assert len(packed_results) == len(packed)
    assert all(result.domain_results for result in results)
    assert all(
        len(result.stage_match_counts) == len(compiled.schedule.stages)
        for result in results
    )


def test_frame_xor_tail_mask_one_matcher_call_and_explicit_telemetry(
    monkeypatch,
) -> None:
    class CountingMatcher:
        def __init__(self) -> None:
            self.calls = 0

        def decode_batch(self, shots, **kwargs):
            self.calls += 1
            assert kwargs == {
                "bit_packed_shots": True,
                "bit_packed_predictions": True,
            }
            np.testing.assert_array_equal(shots, np.array([[1], [2]], dtype=np.uint8))
            return np.array([[0x04, 0xFF], [0x08, 0xFC]], dtype=np.uint8)

    sentinels = []

    def fake_predecode(_graph, _schedule, shot, **kwargs):
        assert kwargs == {
            "collect_hardware_proxies": False,
            "_schedule_is_validated": True,
        }
        frame = np.zeros(10, dtype=np.uint8)
        frame[0] = shot[0]
        frame[9] = shot[1]
        result = SimpleNamespace(
            residual_syndrome=shot.copy(),
            observable_frame=frame,
            domain_results={"domain": SimpleNamespace(initial_hw=int(np.sum(shot)))},
            stage_match_counts=(0,),
        )
        sentinels.append(result)
        return result

    matcher = CountingMatcher()
    compiled = CompiledPinballV2Decoder(
        graph=SimpleNamespace(matcher=matcher),
        schedule=SimpleNamespace(stages=(object(),)),
        num_detectors=5,
        num_observables=10,
    )
    monkeypatch.setattr(pinball_v2_decoder, "predecode_pinball_v2", fake_predecode)
    packed = np.array([[0b00001], [0b00010]], dtype=np.uint8)

    actual, telemetry = compiled.decode_shots_bit_packed_with_telemetry(
        bit_packed_detection_event_data=packed
    )

    assert matcher.calls == 1
    assert telemetry == tuple(sentinels)
    np.testing.assert_array_equal(
        actual,
        np.array([[0x05, 0x03], [0x08, 0x02]], dtype=np.uint8),
    )


def test_explicit_predecode_forwards_hardware_proxy_opt_in(monkeypatch) -> None:
    seen = []

    def fake_predecode(_graph, _schedule, shot, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            residual_syndrome=shot.copy(),
            observable_frame=np.zeros(1, dtype=np.uint8),
        )

    compiled = CompiledPinballV2Decoder(
        graph=SimpleNamespace(),
        schedule=SimpleNamespace(),
        num_detectors=2,
        num_observables=1,
    )
    monkeypatch.setattr(pinball_v2_decoder, "predecode_pinball_v2", fake_predecode)

    _, _, results = compiled.predecode_shots(
        np.zeros((2, 2), dtype=np.uint8),
        collect_hardware_proxies=True,
    )

    assert len(results) == 2
    assert seen == [
        {
            "collect_hardware_proxies": True,
            "_schedule_is_validated": True,
        }
    ] * 2
    with pytest.raises(TypeError, match="collect_hardware_proxies must be bool"):
        compiled.predecode_shots(
            np.zeros((1, 2), dtype=np.uint8),
            collect_hardware_proxies=1,
        )


def test_standard_sinter_path_discards_results(monkeypatch) -> None:
    class ZeroMatcher:
        def decode_batch(self, shots, **_kwargs):
            return np.zeros((len(shots), 1), dtype=np.uint8)

    sentinel = SimpleNamespace(
        residual_syndrome=np.zeros(2, dtype=np.uint8),
        observable_frame=np.zeros(1, dtype=np.uint8),
    )
    monkeypatch.setattr(
        pinball_v2_decoder,
        "predecode_pinball_v2",
        lambda _graph, _schedule, _shot, **_kwargs: sentinel,
    )
    compiled = CompiledPinballV2Decoder(
        graph=SimpleNamespace(matcher=ZeroMatcher()),
        schedule=None,
        num_detectors=2,
        num_observables=1,
    )
    shots = np.zeros((3, 2), dtype=np.uint8)

    _, _, retained = compiled.predecode_shots(shots)
    _, _, discarded = compiled._predecode_shots(shots, retain_results=False)
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.zeros((3, 1), dtype=np.uint8)
    )

    assert retained == (sentinel, sentinel, sentinel)
    assert discarded == ()
    assert predictions.shape == (3, 1)


def test_empty_noncontiguous_and_invalid_packed_batches(monkeypatch) -> None:
    class FailingMatcher:
        def decode_batch(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("empty batches must not call the matcher")

    compiled = CompiledPinballV2Decoder(
        graph=SimpleNamespace(matcher=FailingMatcher()),
        schedule=None,
        num_detectors=9,
        num_observables=2,
    )
    empty = np.zeros((0, 2), dtype=np.uint8)
    predictions, telemetry = compiled.decode_shots_bit_packed_with_telemetry(
        bit_packed_detection_event_data=empty
    )
    assert predictions.shape == (0, 1)
    assert telemetry == ()

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

    monkeypatch.setattr(
        pinball_v2_decoder,
        "predecode_pinball_v2",
        lambda _graph, _schedule, shot, **_kwargs: SimpleNamespace(
            residual_syndrome=shot.copy(),
            observable_frame=np.zeros(2, dtype=np.uint8),
        ),
    )

    class ZeroMatcher:
        def decode_batch(self, shots, **_kwargs):
            return np.zeros((len(shots), 1), dtype=np.uint8)

    compiled.graph = SimpleNamespace(matcher=ZeroMatcher())
    storage = np.zeros((4, 4), dtype=np.uint8)
    storage[:, ::2] = np.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.uint8)
    view = storage[:, ::2]
    assert not view.flags.c_contiguous
    before = view.copy()
    assert compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=view
    ).shape == (4, 1)
    np.testing.assert_array_equal(view, before)
