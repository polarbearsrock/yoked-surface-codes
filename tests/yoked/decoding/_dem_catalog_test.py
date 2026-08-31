from types import SimpleNamespace

import pytest
import stim

import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._dem_catalog import (
    parse_dem_mechanism_catalog,
    validate_uf_dem_catalog,
    validate_unmerged_dem_catalog,
)
from yoked.decoding._pinball_v2_decoder import (
    _validate_unmerged_dem_catalog as pinball_validate_unmerged_dem_catalog,
)
from yoked.decoding._promatch_graph import compile_matching_graph
from yoked.decoding._promatch_layout import compile_layout


def _small_dem():
    circuit = yoked_magic_memory_circuit(
        patch_diameter=3,
        rounds=3,
        noise=gen.NoiseModel.si1000(0.002),
        style="cz",
        yokes=2,
        num_patches=2,
    )
    return circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )


def test_catalog_preserves_separator_order_probability_multiplicity_and_masks() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1
        error(0.2) D0 L0
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        """
    )

    first = parse_dem_mechanism_catalog(dem)
    second = parse_dem_mechanism_catalog(dem)

    assert first == second
    assert [component.detector_boundary for component in first.components] == [
        (0, None),
        (1, None),
        (0, None),
    ]
    assert [component.component_index for component in first.components] == [0, 1, 0]
    assert [component.probability_hex for component in first.components] == [
        (0.1).hex(),
        (0.1).hex(),
        (0.2).hex(),
    ]
    assert [component.observable_mask for component in first.components] == [
        b"\x01",
        b"\x02",
        b"\x01",
    ]


def test_extracted_compatibility_validator_matches_pinball_on_real_dem() -> None:
    dem = _small_dem()
    layout = compile_layout(dem, mode="fullhistory")
    graph = compile_matching_graph(dem, layout, require_zero_frame=False)

    pinball_validate_unmerged_dem_catalog(dem, graph)
    validate_unmerged_dem_catalog(dem, graph)


def test_extracted_and_pinball_validators_both_reject_frame_ambiguity() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.2) D0 L1
        detector(0, 0, 0) D0
        """
    )
    graph = SimpleNamespace(
        num_detectors=1,
        num_observables=2,
        edges=(
            SimpleNamespace(source=0, target=None, observable_mask=b"\x01"),
        ),
    )

    for validator in (pinball_validate_unmerged_dem_catalog, validate_unmerged_dem_catalog):
        with pytest.raises(ValueError, match="ambiguous observable frames"):
            validator(dem, graph)


def test_uf_catalog_reconciles_parallel_probability_and_canonical_weight() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.2) D0 L0
        detector(0, 0, 0) D0
        """
    )
    import pymatching

    matcher = pymatching.Matching.from_detector_error_model(dem)
    raw_source, raw_target, data = next(iter(matcher.edges()))
    edge = SimpleNamespace(
        edge_id=0,
        source=raw_source,
        target=raw_target,
        weight=data["weight"],
        observable_mask=b"\x01",
    )
    graph = SimpleNamespace(
        num_detectors=1,
        num_observables=1,
        edges=(edge,),
    )

    validated = validate_uf_dem_catalog(dem, graph)

    assert len(validated.catalog.components) == 2
    assert len(validated.merge_records) == 1
    record = validated.merge_records[0]
    assert record.component_indices == (0, 1)
    assert float.fromhex(record.effective_probability_hex) == pytest.approx(0.26)
    assert record.canonical_weight_hex == float(data["weight"]).hex()
    assert len(validated.fingerprint) == 64

    bad_graph = SimpleNamespace(
        num_detectors=1,
        num_observables=1,
        edges=(
            SimpleNamespace(
                edge_id=0,
                source=0,
                target=None,
                weight=float(data["weight"]) + 0.25,
                observable_mask=b"\x01",
            ),
        ),
    )
    with pytest.raises(ValueError, match="weight disagrees"):
        validate_uf_dem_catalog(dem, bad_graph)


@pytest.mark.parametrize(
    ("dem_text", "message"),
    [
        ("error(0.1) L0", "detector-free logical component"),
        ("error(0.1) D0 D1 D2", "at most two detectors"),
    ],
)
def test_catalog_rejects_non_graphlike_or_detector_free_logical_components(
    dem_text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_dem_mechanism_catalog(stim.DetectorErrorModel(dem_text))
