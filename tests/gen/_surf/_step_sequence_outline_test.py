import pytest

import gen._surf._viz_sequence_3d
from gen._surf._step_sequence_outline import StepSequenceOutline


def test_write_gltf_does_not_print_before_saving(tmp_path, capsys, monkeypatch):
    class FailingModel:
        def save_json(self, path):
            raise RuntimeError('save failed')

    monkeypatch.setattr(
        gen._surf._viz_sequence_3d,
        'patch_sequence_to_model',
        lambda seq: FailingModel(),
    )
    seq = StepSequenceOutline([])
    with pytest.raises(RuntimeError):
        seq.write_gltf(tmp_path / 'model.gltf')
    # The "wrote file://..." message must not appear when saving failed.
    assert capsys.readouterr().out == ''
