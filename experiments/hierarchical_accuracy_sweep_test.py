"""Exercise restart safety and progress accounting before a long production run."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_recipe(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mpp = load_recipe('mpp_recipe', 'mpp_confidence_100k.py')
sweep = load_recipe('accuracy_sweep', 'hierarchical_accuracy_sweep.py')


def test_failed_mwpm_worker_keeps_log_but_publishes_no_partial_shard(tmp_path, monkeypatch):
    (tmp_path / 'd7/shards').mkdir(parents=True)
    (tmp_path / 'logs').mkdir()

    def failed_worker(command, **kwargs):
        output = Path(command[command.index('--out') + 1])
        output.mkdir()
        (output / 'partial-data').write_text('unfinished')
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(mpp.subprocess, 'run', failed_worker)
    with pytest.raises(RuntimeError, match='failed; inspect'):
        mpp.decode_shard(tmp_path, 7, 0, 16)
    assert not list((tmp_path / 'd7/shards').iterdir())
    assert (tmp_path / 'logs/d7-000000-000016.log').is_file()


def test_completed_mwpm_shard_is_published_and_reused(tmp_path, monkeypatch):
    (tmp_path / 'd7/shards').mkdir(parents=True)
    (tmp_path / 'logs').mkdir()
    calls = []

    def successful_worker(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index('--out') + 1])
        output.mkdir()
        (output / 'manifest.json').write_text('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mpp.subprocess, 'run', successful_worker)
    assert mpp.decode_shard(tmp_path, 7, 0, 16)[-1] == 'decoded'
    assert mpp.decode_shard(tmp_path, 7, 0, 16)[-1] == 'reused'
    assert len(calls) == 1
    assert [p.name for p in (tmp_path / 'd7/shards').iterdir()] == ['000000-000016']


def test_progress_never_counts_a_saved_shard_twice(tmp_path):
    for name in ('000000-001250', '.working-interrupted'):
        directory = tmp_path / 'd7/shards' / name
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text('{}')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'd7-000000-001250.log').write_text('1250/1250 paired shots decoded\n')
    (logs / 'd7-001250-002500.log').write_text('32/1250 shots decoded\n64/1250 shots decoded\n')
    progress = sweep.stage_progress(tmp_path, 7)
    assert progress == dict(saved_shots=1250, decoded_shots=1314,
                            completed_shards=1, unfinished_worker_logs=1)


def test_stage_plan_uses_one_pool_and_identical_distance_settings(tmp_path):
    args = SimpleNamespace(out=tmp_path, shots=1_000_000, workers=32,
                           shard_size=1250, p=0.003, seed_base=2026092400)
    stages = list(sweep.stage_commands(args, 15))
    assert [name for name, _, _ in stages] == ['mwpm', 'uf']
    for _, _, command in stages:
        for option, value in (('--distances', '15'), ('--shots', '1000000'),
                              ('--workers', '32'), ('--rounds-per-distance', '4')):
            assert command[command.index(option) + 1] == value
    uf_command = stages[1][2]
    assert uf_command[uf_command.index('--baseline-root') + 1] == str(stages[0][1])
    assert args.shots // args.shard_size % args.workers == 0
