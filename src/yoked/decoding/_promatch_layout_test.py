import itertools

import pytest
import stim

import gen
from yoked._squareberg_circuits import squareberg_magic_memory_circuit
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._promatch_layout import (
    L1BodyDetector,
    L1FullHistoryDomain,
    L1TerminalDetector,
    L1WindowDomain,
    YokeDetector,
    compile_layout,
)


def _dem(*, d=3, rounds=6, patches=2, yokes=2, remove_x_yoke=False):
    circuit = yoked_magic_memory_circuit(
        patch_diameter=d,
        rounds=rounds,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        yokes=yokes,
        num_patches=patches,
        remove_x_yoke=remove_x_yoke,
    )
    return circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )


@pytest.mark.parametrize("d,yokes", list(itertools.product([3, 4], [0, 1, 2])))
def test_compile_window_layout(d: int, yokes: int):
    layout = compile_layout(_dem(d=d, rounds=2 * d, yokes=yokes))

    assert layout.mode == "windowd"
    assert layout.distance == d
    assert layout.rounds == 2 * d
    assert layout.pitch == d + 1
    assert layout.num_patches == 2
    assert layout.num_windows == 2
    assert len(layout.roles) == layout.num_detectors
    assert len(layout.domains) == 2 * 2 * 2
    assert len(layout.yoke_detector_ids) == (2 if yokes == 2 else 1)
    assert [layout.observable_owner(k) for k in range(4)] == [0, 0, 1, 1]

    for detector_id in layout.body_detector_ids:
        role = layout.role_of(detector_id)
        assert isinstance(role, L1BodyDetector)
        assert role.time < layout.rounds
        assert role.window_id == role.time // d
        assert layout.domain_of(detector_id) == L1WindowDomain(
            role.patch_id, role.check_basis, role.window_id
        )
        x, y, *_ = layout.coordinates[detector_id]
        local_x = x - role.patch_id * layout.pitch
        expected_basis = "X" if round(local_x + y) % 2 == 0 else "Z"
        assert role.check_basis == expected_basis

    for detector_id in layout.terminal_detector_ids:
        assert isinstance(layout.role_of(detector_id), L1TerminalDetector)
        assert layout.domain_of(detector_id) is None
    for detector_id in layout.yoke_detector_ids:
        assert isinstance(layout.role_of(detector_id), YokeDetector)
        assert layout.domain_of(detector_id) is None


def test_compile_full_history_layout():
    layout = compile_layout(_dem(d=3, rounds=7), mode="fullhistory")
    assert layout.mode == "fullhistory"
    assert layout.rounds == 7
    assert layout.num_windows == 3
    assert layout.domains == (
        L1FullHistoryDomain(0, "X"),
        L1FullHistoryDomain(0, "Z"),
        L1FullHistoryDomain(1, "X"),
        L1FullHistoryDomain(1, "Z"),
    )
    assert all(
        isinstance(layout.domain_of(k), L1FullHistoryDomain)
        for k in layout.body_detector_ids
    )


def test_window_layout_rejects_partial_window():
    dem = _dem(d=3, rounds=7)
    with pytest.raises(ValueError, match="rounds divisible by distance"):
        compile_layout(dem)
    assert compile_layout(dem, mode="fullhistory").rounds == 7


def test_layout_rejects_remove_x_yoke():
    with pytest.raises(ValueError, match="two observables per patch"):
        compile_layout(_dem(remove_x_yoke=True))


def test_layout_rejects_missing_coordinates():
    dem = stim.DetectorErrorModel("error(0.1) D0\nlogical_observable L0")
    with pytest.raises(ValueError, match="at least"):
        compile_layout(dem)


def test_layout_rejects_duplicate_and_nonstandard_spatial_sites():
    dem_text = str(_dem())
    duplicate = stim.DetectorErrorModel(
        dem_text.replace(
            "detector(-0.5, 1.5, 0) D0",
            "detector(0.5, -0.5, 0) D0",
            1,
        )
    )
    with pytest.raises(ValueError, match="duplicate inner detector coordinate"):
        compile_layout(duplicate)

    nonstandard = stim.DetectorErrorModel(
        dem_text.replace(
            "detector(-0.5, 1.5, 0) D0",
            "detector(-0.5, 0.5, 0) D0",
            1,
        )
    )
    with pytest.raises(ValueError, match="maintained stabilizer layout"):
        compile_layout(nonstandard)


def test_layout_rejects_squareberg():
    circuit = squareberg_magic_memory_circuit(
        patch_diameter=2,
        rounds=2,
        noise=gen.NoiseModel.si1000(1e-3),
        style="cz",
        num_patches=16,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    with pytest.raises(ValueError, match="yoke"):
        compile_layout(dem)


def test_layout_fingerprint_is_deterministic_and_mode_specific():
    dem = _dem()
    window1 = compile_layout(dem)
    window2 = compile_layout(dem)
    full = compile_layout(dem, mode="fullhistory")
    assert window1.fingerprint == window2.fingerprint
    assert window1.fingerprint != full.fingerprint
