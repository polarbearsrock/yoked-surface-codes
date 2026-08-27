from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import REPO_ROOT
import yoked.decoding._pinball_promatch_fig8_sweep as sweep


def _fake_provenance(index: int) -> dict[str, object]:
    hashes = sorted(sweep.HASH_PROVENANCE_FIELDS)
    result: dict[str, object] = {
        key: f"{index * len(hashes) + offset + 1:064x}"
        for offset, key in enumerate(hashes)
    }
    result["num_detectors"] = 100 + index
    result["num_observables"] = 12
    result["arms"] = {
        "u0": {
            "decoder": "uncorrelated-pymatching-from-common-dem",
            "construction": "pymatching.Matching.from_detector_error_model",
        },
        "promatch": dict(sweep.PROMATCH_DECODER),
        "pinball": dict(sweep.PINBALL_DECODER),
    }
    return result


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    repo.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    source_paths = ["source.py", *sorted(sweep.REQUIRED_SWEEP_SOURCE_PATHS)]
    for relative in source_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source\n", encoding="utf-8")
    monkeypatch.setattr(
        sweep,
        "repository_state",
        lambda _: {"repository_commit": "a" * 40, "clean_worktree": True},
    )
    monkeypatch.setattr(sweep, "current_software_versions", lambda: {"python": "x"})
    monkeypatch.setattr(
        sweep, "current_execution_environment", lambda: {"machine": "x"}
    )
    monkeypatch.setattr(sweep, "_source_paths", lambda _: source_paths)
    monkeypatch.setattr(sweep, "_canonical_file_hash", lambda _: "b" * 64)
    calls: list[dict] = []

    def prepare(cell, **kwargs):
        assert kwargs == {
            "promatch_config": sweep.PROMATCH_DECODER,
            "pinball_config": sweep.PINBALL_DECODER,
            "dem_options": sweep.DEM_OPTIONS,
            "verify_hashes": False,
        }
        result = SimpleNamespace(provenance=_fake_provenance(len(calls)))
        calls.append(cell)
        return result

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep.secrets, "token_hex", lambda _: "c" * 64)
    out = tmp_path / "campaign"
    frozen = sweep.create_campaign(out, shots_per_cell=2_001, root=repo)
    assert len(calls) == 16
    return out, frozen


def test_create_freezes_native_three_arm_grid_and_complete_schedules(
    campaign: tuple[Path, dict],
) -> None:
    out, frozen = campaign
    assert set(frozen) == sweep.CAMPAIGN_FIELDS
    assert frozen["p"] == sweep.FIXED_P == 0.002
    assert frozen["claim_bearing"] is False
    assert frozen["promatch_decoder"] == sweep.PROMATCH_DECODER
    assert frozen["pinball_decoder"] == sweep.PINBALL_DECODER
    assert (
        frozen["pinball_decoder"]["stage_order"]
        is not sweep.PINBALL_DECODER["stage_order"]
    )
    assert frozen["processes"] == 32
    assert frozen["threads_per_process"] == 1
    assert len(frozen["cells"]) == 16
    assert {
        (cell["d"], cell["patches"], cell["r"] // cell["d"], cell["p"])
        for cell in frozen["cells"]
    } == {
        (d, patches, multiplier, 0.002)
        for d in sweep.DISTANCES
        for patches in sweep.PATCH_COUNTS
        for multiplier in sweep.ROUND_MULTIPLIERS
    }
    ids: list[int] = []
    for cell in frozen["cells"]:
        rows = frozen["cell_batch_schedules"][cell["cell_id"]]
        assert [row["shots"] for row in rows] == [1_000, 1_000, 1]
        ids.extend(row["batch_id"] for row in rows)
    assert ids == list(range(48))
    assert out.joinpath("campaign.json").is_file()
    assert list(out.joinpath("collection").iterdir()) == []
    assert list(out.parent.joinpath("scratch").iterdir()) == []
    assert sweep.load_campaign(out) == frozen


def test_campaign_directory_install_is_atomic_and_retryable_after_interrupt(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen = campaign
    destination = tmp_path / "retry-campaign"
    scratch = Path(os.environ["TMPDIR"])
    real_publish = sweep._rename_directory_noreplace

    def interrupt_publication(_source: Path, _target: Path) -> None:
        raise KeyboardInterrupt("simulated interruption")

    with monkeypatch.context() as context:
        context.setattr(sweep, "_rename_directory_noreplace", interrupt_publication)
        with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
            sweep._install_new_campaign(destination, frozen)

    assert not destination.exists()
    assert not list(scratch.glob("pinball-campaign-retry-campaign-*"))

    observed_sources: list[Path] = []

    def observe_publication(source: Path, target: Path) -> None:
        observed_sources.append(source)
        real_publish(source, target)

    with monkeypatch.context() as context:
        context.setattr(sweep, "_rename_directory_noreplace", observe_publication)
        sweep._install_new_campaign(destination, frozen)

    assert len(observed_sources) == 1
    assert observed_sources[0].parent == scratch
    assert sweep.load_campaign(destination) == frozen
    assert not list(scratch.glob("pinball-campaign-retry-campaign-*"))


def test_campaign_install_requires_canonical_same_filesystem_tmpdir(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen = campaign

    monkeypatch.delenv("TMPDIR")
    with pytest.raises(ValueError, match="TMPDIR must be set"):
        sweep._install_new_campaign(tmp_path / "missing-tmpdir", frozen)
    assert not tmp_path.joinpath("missing-tmpdir").exists()

    canonical_scratch = tmp_path / "scratch"
    scratch_link = tmp_path / "scratch-link"
    scratch_link.symlink_to(canonical_scratch, target_is_directory=True)
    monkeypatch.setenv("TMPDIR", str(scratch_link))
    with pytest.raises(ValueError, match="absolute, real directory"):
        sweep._install_new_campaign(tmp_path / "symlinked-tmpdir", frozen)
    assert not tmp_path.joinpath("symlinked-tmpdir").exists()

    scratch = tmp_path / "other-scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(sweep, "_same_filesystem", lambda *_: False)
    with pytest.raises(ValueError, match="share a filesystem"):
        sweep._install_new_campaign(tmp_path / "other-filesystem", frozen)
    assert not tmp_path.joinpath("other-filesystem").exists()
    assert list(scratch.iterdir()) == []


def test_campaign_install_cleans_stage_after_manifest_write_failure(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen = campaign
    destination = tmp_path / "write-failure"
    scratch = Path(os.environ["TMPDIR"])
    monkeypatch.setattr(
        sweep,
        "_atomic_json_write",
        lambda *_: (_ for _ in ()).throw(OSError("simulated write failure")),
    )

    with pytest.raises(OSError, match="simulated write failure"):
        sweep._install_new_campaign(destination, frozen)

    assert not destination.exists()
    assert not list(scratch.glob("pinball-campaign-write-failure-*"))


def test_atomic_publication_preserves_competing_empty_directory(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen = campaign
    destination = tmp_path / "competing-directory"
    real_publish = sweep._rename_directory_noreplace

    def create_competitor_then_publish(source: Path, target: Path) -> None:
        target.mkdir()
        real_publish(source, target)

    monkeypatch.setattr(
        sweep, "_rename_directory_noreplace", create_competitor_then_publish
    )
    with pytest.raises(FileExistsError, match="atomic publication"):
        sweep._install_new_campaign(destination, frozen)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_noreplace_publication_allows_exactly_one_complete_winner(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first-stage"
    second = tmp_path / "second-stage"
    destination = tmp_path / "winner"
    first.mkdir()
    second.mkdir()
    first.joinpath("marker").write_text("first\n", encoding="utf-8")
    second.joinpath("marker").write_text("second\n", encoding="utf-8")

    sweep._rename_directory_noreplace(first, destination)
    with pytest.raises(FileExistsError, match="atomic publication"):
        sweep._rename_directory_noreplace(second, destination)

    assert destination.joinpath("marker").read_text() == "first\n"
    assert second.joinpath("marker").read_text() == "second\n"


@pytest.mark.parametrize("link_kind", ["dangling", "directory"])
def test_atomic_publication_preserves_competing_symlink(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    _, frozen = campaign
    destination = tmp_path / f"competing-{link_kind}-link"
    target = tmp_path / f"{link_kind}-target"
    if link_kind == "directory":
        target.mkdir()
        target.joinpath("marker").write_text("untouched\n", encoding="utf-8")
    real_publish = sweep._rename_directory_noreplace

    def create_competitor_then_publish(source: Path, output: Path) -> None:
        output.symlink_to(target, target_is_directory=True)
        real_publish(source, output)

    monkeypatch.setattr(
        sweep, "_rename_directory_noreplace", create_competitor_then_publish
    )
    with pytest.raises(FileExistsError, match="atomic publication"):
        sweep._install_new_campaign(destination, frozen)

    assert destination.is_symlink()
    assert destination.readlink() == target
    if link_kind == "directory":
        assert target.joinpath("marker").read_text() == "untouched\n"
    else:
        assert not target.exists()


def test_campaign_install_fails_closed_without_noreplace_support(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen = campaign
    destination = tmp_path / "unsupported-publication"
    scratch = Path(os.environ["TMPDIR"])
    monkeypatch.setattr(
        sweep,
        "_rename_directory_noreplace",
        lambda *_: (_ for _ in ()).throw(RuntimeError("renameat2 unavailable")),
    )

    with pytest.raises(RuntimeError, match="renameat2 unavailable"):
        sweep._install_new_campaign(destination, frozen)

    assert not destination.exists()
    assert not list(scratch.glob("pinball-campaign-unsupported-publication-*"))


def test_post_publication_interrupt_leaves_complete_campaign(
    campaign: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen = campaign
    destination = tmp_path / "post-publication-interrupt"
    real_fsync = sweep._fsync_directory

    def interrupt_parent_fsync(directory: Path) -> None:
        if directory == destination.parent:
            raise KeyboardInterrupt("simulated post-publication interruption")
        real_fsync(directory)

    monkeypatch.setattr(sweep, "_fsync_directory", interrupt_parent_fsync)
    with pytest.raises(KeyboardInterrupt, match="post-publication"):
        sweep._install_new_campaign(destination, frozen)

    assert sweep.load_campaign(destination) == frozen


def test_dirty_create_fails_before_compilation_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    destination = tmp_path / "campaign"
    monkeypatch.setattr(
        sweep,
        "repository_state",
        lambda _: {"repository_commit": "a" * 40, "clean_worktree": False},
    )
    monkeypatch.setattr(
        sweep,
        "prepare_cell",
        lambda *_args, **_kwargs: pytest.fail("dirty create compiled a cell"),
    )

    with pytest.raises(ValueError, match="clean worktree"):
        sweep.create_campaign(destination, shots_per_cell=1, root=repo)

    assert not destination.exists()


def test_create_validates_atomic_location_before_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    destination = tmp_path / "campaign"
    monkeypatch.setattr(
        sweep,
        "repository_state",
        lambda _: {"repository_commit": "a" * 40, "clean_worktree": True},
    )
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(
        sweep,
        "prepare_cell",
        lambda *_args, **_kwargs: pytest.fail("invalid location compiled a cell"),
    )

    with pytest.raises(ValueError, match="TMPDIR must be set"):
        sweep.create_campaign(destination, shots_per_cell=1, root=repo)

    assert not destination.exists()


def test_one_million_shot_schedule_has_16000_unique_complete_batches() -> None:
    expected, schedules = sweep._build_schedules(
        sweep._unfrozen_cells(), shots_per_cell=1_000_000
    )
    assert set(expected.values()) == {1_000_000}
    assert all(len(rows) == 1_000 for rows in schedules.values())
    assert all(
        row["shots"] == 1_000
        for cell_rows in schedules.values()
        for row in cell_rows
    )
    ids = [
        row["batch_id"]
        for cell_rows in schedules.values()
        for row in cell_rows
    ]
    assert ids == list(range(16_000))


def test_source_set_freezes_all_scientific_entrypoints(
    campaign: tuple[Path, dict],
) -> None:
    assert {
        "aws/run_pinball_promatch_fig8",
        "gcp/run_pinball_promatch_fig8",
        "experiments/PINBALL_PROMATCH_FIG8_PAIRED_32.md",
    } <= sweep.REQUIRED_SWEEP_SOURCE_PATHS
    assert sweep.REQUIRED_SWEEP_SOURCE_PATHS <= set(sweep._source_paths(REPO_ROOT))
    _, frozen = campaign
    changed = copy.deepcopy(frozen)
    missing = "tools/benchmark_pinball_promatch_fig8"
    del changed["source_hashes"][missing]
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="required sweep sources"):
        sweep.validate_campaign(changed)


def test_campaign_rejects_parameter_and_provenance_drift(
    campaign: tuple[Path, dict],
) -> None:
    _, frozen = campaign
    for mutate, message in (
        (lambda value: value.__setitem__("p", 0.001), "frozen p"),
        (
            lambda value: value["cells"][0].__setitem__("p", 0.001),
            "Figure-8 cell p",
        ),
        (
            lambda value: value["cells"][0].__setitem__(
                "pinball_schedule_fingerprint", "bad"
            ),
            "pinball_schedule_fingerprint",
        ),
    ):
        changed = copy.deepcopy(frozen)
        mutate(changed)
        changed["experiment_id"] = sweep.manifest_experiment_id(changed)
        with pytest.raises(ValueError, match=message):
            sweep.validate_campaign(changed)


def test_campaign_rejects_incomplete_or_overlarge_batch_schedule(
    campaign: tuple[Path, dict],
) -> None:
    _, frozen = campaign
    changed = copy.deepcopy(frozen)
    cell_id = changed["cells"][0]["cell_id"]
    changed["cell_batch_schedules"][cell_id][0]["shots"] = 1_001
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="exceeds"):
        sweep.validate_campaign(changed)


def test_status_counts_only_validated_ledgers(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first = frozen["cells"][0]["cell_id"]
    fake_row = {"batch": {"batch_id": 0, "shot_start": 0, "shots": 1_000}}
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: ({(first, 0): fake_row}, out / "collection"),
    )
    status = sweep.campaign_status(out)
    assert status["complete"] is False
    assert status["p"] == 0.002
    assert status["totals"] == {
        "expected_shots": 16 * 2_001,
        "completed_shots": 1_000,
        "expected_batches": 48,
        "completed_batches": 1,
    }


def test_loader_orders_rows_and_can_require_completeness(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first = frozen["cells"][0]["cell_id"]
    batch_0 = {"batch": {"batch_id": 0}}
    batch_1 = {"batch": {"batch_id": 1}}
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: (
            {(first, 1): batch_1, (first, 0): batch_0},
            out / "collection",
        ),
    )
    loaded, rows = sweep.load_validated_collection(out)
    assert loaded == frozen
    assert rows == (batch_0, batch_1)
    with pytest.raises(ValueError, match="incomplete"):
        sweep.load_validated_collection(out, require_complete=True)


def test_collection_requires_exactly_32_processes_before_runtime_work(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = campaign
    monkeypatch.setattr(sweep, "_validate_runtime", lambda *_, **__: None)
    with pytest.raises(ValueError, match="exactly 32"):
        sweep.run_campaign(out, processes=31)


def test_runtime_requires_one_native_thread_and_fixed_shot_collection(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    repo = out.parent / "repo"
    monkeypatch.setattr(sweep, "validate_analysis_runtime", lambda *_, **__: None)
    monkeypatch.setattr(
        sweep, "current_execution_environment", lambda: frozen["execution_environment"]
    )
    for name in sweep.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("MAX_ERRORS", raising=False)
    sweep._validate_runtime(frozen, processes=32, root=repo)

    monkeypatch.setenv("MAX_ERRORS", "1")
    with pytest.raises(ValueError, match="MAX_ERRORS"):
        sweep._validate_runtime(frozen, processes=32, root=repo)
    monkeypatch.delenv("MAX_ERRORS")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        sweep._validate_runtime(frozen, processes=32, root=repo)


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ("dirty", "clean worktree"),
        ("commit", "repository commit"),
        ("software", "software versions"),
        ("source", "source hash"),
    ],
)
def test_analysis_runtime_rejects_identity_mismatches(
    campaign: tuple[Path, dict],
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
    message: str,
) -> None:
    out, frozen = campaign
    repo = out.parent / "repo"
    if gate == "dirty":
        monkeypatch.setattr(
            sweep,
            "repository_state",
            lambda _: {"repository_commit": "a" * 40, "clean_worktree": False},
        )
    elif gate == "commit":
        monkeypatch.setattr(
            sweep,
            "repository_state",
            lambda _: {"repository_commit": "d" * 40, "clean_worktree": True},
        )
    elif gate == "software":
        monkeypatch.setattr(
            sweep, "current_software_versions", lambda: {"python": "y"}
        )
    else:
        monkeypatch.setattr(sweep, "_canonical_file_hash", lambda _: "d" * 64)

    with pytest.raises(ValueError, match=message):
        sweep.validate_analysis_runtime(frozen, root=repo)


def test_status_remains_available_without_consulting_checkout_identity(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = campaign
    monkeypatch.setattr(
        sweep,
        "repository_state",
        lambda _: pytest.fail("status consulted checkout identity"),
    )

    status = sweep.campaign_status(out)

    assert status["complete"] is False
    assert status["totals"]["completed_batches"] == 0


def test_run_rejects_dirty_checkout_before_collection_work(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    repo = out.parent / "repo"
    for name in sweep.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("MAX_ERRORS", raising=False)
    monkeypatch.setattr(
        sweep,
        "repository_state",
        lambda _: {"repository_commit": "a" * 40, "clean_worktree": False},
    )
    monkeypatch.setattr(
        sweep,
        "current_execution_environment",
        lambda: frozen["execution_environment"],
    )
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: pytest.fail("dirty run inspected collection ledgers"),
    )
    monkeypatch.setattr(
        sweep,
        "prepare_cell",
        lambda *_args, **_kwargs: pytest.fail("dirty run prepared a cell"),
    )

    with pytest.raises(ValueError, match="clean worktree"):
        sweep.run_campaign(out, processes=32, root=repo)


def test_run_resumes_and_atomically_installs_only_missing_batches(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first_cell = frozen["cells"][0]["cell_id"]
    collection = out / "collection"
    existing_key = (first_cell, 0)
    prepared_cells: list[str] = []
    submitted: list[dict] = []
    pool_cell_ids: list[str] = []
    preloaded: list[str] = []
    clear_count = 0
    validated: list[tuple[str, int]] = []
    installed: list[tuple[Path, object]] = []
    sentinel = {"complete": True}

    monkeypatch.setattr(sweep, "_validate_runtime", lambda *_, **__: None)
    monkeypatch.setattr(sweep, "configure_single_thread_runtime", lambda: None)
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: ({existing_key: {"already": "valid"}}, collection),
    )

    def prepare(cell, **kwargs):
        assert kwargs["verify_hashes"] is True
        prepared_cells.append(cell["cell_id"])
        return SimpleNamespace(cell=cell)

    def collection_task(**kwargs):
        batch = kwargs["batch"]
        return {
            **kwargs,
            "batch": {
                "batch_id": batch.batch_id,
                "shot_start": batch.shot_start,
                "shots": batch.shots,
            },
        }

    def run_pool(tasks, *, processes, install):
        assert processes == 32
        tasks = list(tasks)
        assert {task["cell"]["cell_id"] for task in tasks} == {
            tasks[0]["cell"]["cell_id"]
        }
        pool_cell_ids.append(tasks[0]["cell"]["cell_id"])
        submitted.extend(tasks)
        for task in tasks:
            install(task, {"installed_batch": task["batch"]["batch_id"]})

    def preload(prepared):
        preloaded.append(prepared.cell["cell_id"])

    def clear():
        nonlocal clear_count
        clear_count += 1

    def validate(_row, **kwargs):
        validated.append((kwargs["cell"]["cell_id"], kwargs["batch"].batch_id))

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep, "_collection_task", collection_task)
    monkeypatch.setattr(sweep, "_run_bounded_pool", run_pool)
    monkeypatch.setattr(sweep, "_preload_worker_cell", preload)
    monkeypatch.setattr(sweep, "_clear_worker_preload", clear)
    monkeypatch.setattr(sweep, "validate_ledger_row", validate)
    monkeypatch.setattr(
        sweep, "_atomic_json_write", lambda path, row: installed.append((path, row))
    )
    monkeypatch.setattr(sweep, "campaign_status", lambda _: sentinel)

    assert sweep.run_campaign(out, processes=32, root=out.parent / "repo") is sentinel
    assert prepared_cells == [cell["cell_id"] for cell in frozen["cells"]]
    assert pool_cell_ids == prepared_cells
    assert preloaded == prepared_cells
    assert clear_count == len(prepared_cells)
    assert len(submitted) == len(validated) == len(installed) == 47
    assert all(
        not (
            task["cell"]["cell_id"] == existing_key[0]
            and task["batch"]["batch_id"] == existing_key[1]
        )
        for task in submitted
    )
    for path, row in installed:
        assert path.name == f"batch-{row['installed_batch']:08d}.json"


def test_bounded_pool_never_queues_more_than_twice_the_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending_sizes: list[int] = []
    installed: list[int] = []
    parent_checked = False

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            return self.task

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["mp_context"].get_start_method() == "fork"
            assert kwargs["initializer"] is sweep.configure_single_thread_runtime

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def submit(self, _worker, task):
            assert parent_checked
            return Future(task)

    def wait_one(pending, **_):
        pending_sizes.append(len(pending))
        first = next(iter(pending))
        return {first}, set(pending) - {first}

    monkeypatch.setattr(sweep, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(sweep, "wait", wait_one)

    def require_single_threaded_parent() -> None:
        nonlocal parent_checked
        parent_checked = True

    monkeypatch.setattr(
        sweep,
        "_require_single_threaded_parent",
        require_single_threaded_parent,
    )
    sweep._run_bounded_pool(
        ({"index": index} for index in range(9)),
        processes=2,
        install=lambda _task, result: installed.append(result["index"]),
    )
    assert parent_checked
    assert max(pending_sizes) == 4
    assert sorted(installed) == list(range(9))


def test_single_threaded_parent_accepts_one_linux_task(tmp_path: Path) -> None:
    task_directory = tmp_path / "task"
    task_directory.mkdir()
    task_directory.joinpath("101").mkdir()

    sweep._require_single_threaded_parent(
        task_directory,
        expected_tid=101,
    )


def test_single_threaded_parent_rejects_multiple_linux_tasks(tmp_path: Path) -> None:
    task_directory = tmp_path / "task"
    task_directory.mkdir()
    task_directory.joinpath("101").mkdir()
    task_directory.joinpath("202").mkdir()

    with pytest.raises(RuntimeError, match="exactly one parent native task"):
        sweep._require_single_threaded_parent(
            task_directory,
            expected_tid=101,
        )


def test_run_skips_preload_and_pool_for_a_fully_completed_cell(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first_cell = frozen["cells"][0]
    first_id = first_cell["cell_id"]
    existing = {
        (first_id, row["batch_id"]): {"already": "valid"}
        for row in frozen["cell_batch_schedules"][first_id]
    }
    prepared: list[str] = []
    pooled: list[str] = []
    preloaded: list[str] = []
    clears = 0

    monkeypatch.setattr(sweep, "_validate_runtime", lambda *_, **__: None)
    monkeypatch.setattr(sweep, "configure_single_thread_runtime", lambda: None)
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: (existing, out / "collection"),
    )

    def prepare(cell, **_kwargs):
        prepared.append(cell["cell_id"])
        return SimpleNamespace(cell=cell)

    def task(**kwargs):
        batch = kwargs["batch"]
        return {
            **kwargs,
            "batch": {
                "batch_id": batch.batch_id,
                "shot_start": batch.shot_start,
                "shots": batch.shots,
            },
        }

    def pool(tasks, **_kwargs):
        tasks = list(tasks)
        pooled.append(tasks[0]["cell"]["cell_id"])

    def clear():
        nonlocal clears
        clears += 1

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep, "_collection_task", task)
    monkeypatch.setattr(sweep, "_run_bounded_pool", pool)
    monkeypatch.setattr(
        sweep,
        "_preload_worker_cell",
        lambda item: preloaded.append(item.cell["cell_id"]),
    )
    monkeypatch.setattr(sweep, "_clear_worker_preload", clear)
    monkeypatch.setattr(sweep, "campaign_status", lambda _: {"complete": False})

    sweep.run_campaign(out, processes=32, root=out.parent / "repo")
    expected = [cell["cell_id"] for cell in frozen["cells"][1:]]
    assert prepared == pooled == preloaded == expected
    assert clears == len(expected)


@pytest.mark.parametrize("shots", [0, 1_000_001, True])
def test_shot_bound_is_exact(shots: int) -> None:
    with pytest.raises((TypeError, ValueError), match="shots_per_cell"):
        sweep._validate_shots(shots)
