from gen._layers._qubit_coord_annotation_layer import QubitCoordAnnotationLayer
from gen._layers._shift_coord_annotation_layer import ShiftCoordAnnotationLayer


def test_shift_coord_annotation_layer_copy_is_independent():
    layer = ShiftCoordAnnotationLayer(shift=[1, 2])
    copied = layer.copy()
    copied.offset_by([10, 20, 30])
    assert layer.shift == [1, 2]
    assert copied.shift == [11, 22, 30]


def test_qubit_coord_annotation_layer_copy_is_independent():
    layer = QubitCoordAnnotationLayer(coords={5: [1, 2]})
    copied = layer.copy()
    copied.offset_by([10, 20])
    assert layer.coords == {5: [1, 2]}
    assert copied.coords == {5: [11, 22]}
