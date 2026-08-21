from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import REPO_ROOT
import yoked.decoding._fig8_paired_sweep as sweep


def _fake_provenance(index: int) -> dict[str, object]:
    return {
        "circuit_sha256": f"{index + 1:064x}",
        "dem_sha256": f"{index + 17:064x}",
        "layout_fingerprint": f"{index + 33:064x}",
        "graph_fingerprint": f"{index + 49:064x}",
        "num_detectors": 100 + index,
        "num_observables": 12,
    }


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    repo.mkdir()
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
    calls = 0

    def prepare(cell, **_):
        nonlocal calls
        result = SimpleNamespace(provenance=_fake_provenance(calls))
        calls += 1
        return result

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep.secrets, "token_hex", lambda _: "c" * 64)
    out = tmp_path / "campaign"
    frozen = sweep.create_campaign(out, p=0.001, shots_per_cell=2_001, root=repo)
    assert calls == 16
    return out, frozen


def test_create_freezes_exact_grid_and_globally_unique_1000_shot_batches(
    campaign: tuple[Path, dict],
) -> None:
    out, frozen = campaign
    assert set(frozen) == sweep.CAMPAIGN_FIELDS
    assert len(frozen["cells"]) == 16
    assert [cell["d"] for cell in frozen["cells"]] == [
        d for d in sweep.DISTANCES for _ in range(4)
    ]
    assert {
        (cell["d"], cell["patches"], cell["r"] // cell["d"]) for cell in frozen["cells"]
    } == {
        (d, patches, multiplier)
        for d in sweep.DISTANCES
        for patches in sweep.PATCH_COUNTS
        for multiplier in sweep.ROUND_MULTIPLIERS
    }
    schedules = frozen["cell_batch_schedules"]
    ids = []
    for cell in frozen["cells"]:
        rows = schedules[cell["cell_id"]]
        assert [row["shots"] for row in rows] == [1_000, 1_000, 1]
        ids.extend(row["batch_id"] for row in rows)
    assert ids == list(range(48))
    assert out.joinpath("campaign.json").is_file()
    assert list(out.joinpath("collection").iterdir()) == []
    assert sweep.load_campaign(out) == frozen


def test_frozen_source_set_includes_collector_and_plotter(
    campaign: tuple[Path, dict],
) -> None:
    paths = sweep._source_paths(REPO_ROOT)
    assert "tools/benchmark_fig8_paired" in paths
    assert "tools/plot_fig8_paired" in paths
    assert "src/yoked/decoding/_fig8_paired_sweep.py" in paths

    _, frozen = campaign
    changed = copy.deepcopy(frozen)
    del changed["source_hashes"]["tools/plot_fig8_paired"]
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="required sweep sources"):
        sweep.validate_campaign(changed)


def test_campaign_rejects_grid_or_batch_contract_drift(
    campaign: tuple[Path, dict],
) -> None:
    _, frozen = campaign
    changed = copy.deepcopy(frozen)
    changed["cells"][0]["yokes"] = 1
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="incorrect Figure-8 cell"):
        sweep.validate_campaign(changed)

    changed = copy.deepcopy(frozen)
    cell_id = changed["cells"][0]["cell_id"]
    changed["cell_batch_schedules"][cell_id][0]["shots"] = 1_001
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="exceeds"):
        sweep.validate_campaign(changed)


def test_status_is_compact_and_counts_only_validated_ledgers(
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
    assert status["totals"] == {
        "expected_shots": 16 * 2_001,
        "completed_shots": 1_000,
        "expected_batches": 48,
        "completed_batches": 1,
    }
    assert status["cells"][0]["completed_batches"] == 1
    assert status["cells"][1]["completed_batches"] == 0


def test_status_allows_plot_outputs_outside_strict_collection(
    campaign: tuple[Path, dict],
) -> None:
    out, _ = campaign
    plots = out / "plots"
    plots.mkdir()
    plots.joinpath("figure.png").write_bytes(b"plot output")

    status = sweep.campaign_status(out)
    assert status["complete"] is False
    assert status["totals"]["completed_batches"] == 0


def test_public_collection_loader_orders_rows_and_can_require_completeness(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first = frozen["cells"][0]["cell_id"]
    second_batch = {"batch": {"batch_id": 1}}
    first_batch = {"batch": {"batch_id": 0}}
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: (
            {(first, 1): second_batch, (first, 0): first_batch},
            out / "collection",
        ),
    )

    loaded, rows = sweep.load_validated_collection(out)
    assert loaded == frozen
    assert rows == (first_batch, second_batch)
    with pytest.raises(ValueError, match="incomplete"):
        sweep.load_validated_collection(out, require_complete=True)


def test_run_requires_exact_32_processes_before_runtime_work(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = campaign
    monkeypatch.setattr(sweep, "_validate_runtime", lambda *_, **__: None)
    with pytest.raises(ValueError, match="exactly 32"):
        sweep.run_campaign(out, processes=31)


def test_analysis_runtime_checks_code_identity_but_not_collection_environment(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    repo = out.parent / "repo"
    monkeypatch.setenv("MAX_ERRORS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    monkeypatch.setattr(
        sweep, "current_execution_environment", lambda: {"machine": "different"}
    )

    sweep.validate_analysis_runtime(frozen, root=repo)


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
        monkeypatch.setattr(sweep, "current_software_versions", lambda: {"python": "y"})
    else:
        monkeypatch.setattr(sweep, "_canonical_file_hash", lambda _: "d" * 64)

    with pytest.raises(ValueError, match=message):
        sweep.validate_analysis_runtime(frozen, root=repo)


def test_collection_runtime_enforces_collection_only_gates(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    repo = out.parent / "repo"
    for name in sweep.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("MAX_ERRORS", raising=False)
    monkeypatch.setattr(sweep, "validate_analysis_runtime", lambda *_, **__: None)

    monkeypatch.setenv("MAX_ERRORS", "10")
    with pytest.raises(ValueError, match="MAX_ERRORS"):
        sweep._validate_runtime(frozen, processes=32, root=repo)
    monkeypatch.delenv("MAX_ERRORS")

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        sweep._validate_runtime(frozen, processes=32, root=repo)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    monkeypatch.setattr(
        sweep,
        "validate_analysis_runtime",
        lambda *_, **__: (_ for _ in ()).throw(ValueError("analysis mismatch")),
    )
    with pytest.raises(ValueError, match="analysis mismatch"):
        sweep._validate_runtime(frozen, processes=32, root=repo)

    monkeypatch.setattr(sweep, "validate_analysis_runtime", lambda *_, **__: None)
    monkeypatch.setattr(
        sweep, "current_execution_environment", lambda: {"machine": "wrong"}
    )
    with pytest.raises(ValueError, match="execution environment"):
        sweep._validate_runtime(frozen, processes=32, root=repo)


def test_run_campaign_resumes_and_atomically_installs_missing_batches(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first_cell = frozen["cells"][0]["cell_id"]
    existing_key = (first_cell, 0)
    collection = out / "collection"
    prepared_cells: list[str] = []
    installed: list[tuple[Path, object]] = []
    validated: list[tuple[str, int]] = []
    submitted: list[dict] = []
    sentinel_status = {"complete": True}

    monkeypatch.setattr(sweep, "_validate_runtime", lambda *_, **__: None)
    monkeypatch.setattr(sweep, "configure_single_thread_runtime", lambda: None)
    monkeypatch.setattr(
        sweep,
        "_validate_existing_ledgers",
        lambda *_: ({existing_key: {"already": "validated"}}, collection),
    )

    def prepare(cell, **kwargs):
        assert kwargs["verify_hashes"] is True
        prepared_cells.append(cell["cell_id"])
        return object()

    def validate(row, **kwargs):
        validated.append((kwargs["cell"]["cell_id"], kwargs["batch"].batch_id))

    def run_pool(tasks, *, processes, install):
        assert processes == 32
        submitted.extend(tasks)
        for task in submitted:
            install(task, {"installed_batch": task["batch"]["batch_id"]})

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep, "_validate_ledger_row", validate)
    monkeypatch.setattr(sweep, "_run_bounded_pool", run_pool)
    monkeypatch.setattr(
        sweep, "_atomic_json_write", lambda path, row: installed.append((path, row))
    )
    monkeypatch.setattr(sweep, "campaign_status", lambda _: sentinel_status)

    assert (
        sweep.run_campaign(out, processes=32, root=out.parent / "repo")
        is sentinel_status
    )
    assert prepared_cells == [cell["cell_id"] for cell in frozen["cells"]]
    assert len(submitted) == 47
    assert all(
        not (
            task["cell"]["cell_id"] == existing_key[0]
            and task["batch"]["batch_id"] == existing_key[1]
        )
        for task in submitted
    )
    assert len(validated) == len(installed) == 47
    for path, row in installed:
        assert path.parent.parent.parent == collection
        assert path.name == f"batch-{row['installed_batch']:08d}.json"


def test_bounded_pool_never_has_more_than_twice_processes_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending_sizes: list[int] = []
    installed: list[int] = []

    class Future:
        def __init__(self, task: dict) -> None:
            self.task = task

        def result(self) -> dict:
            return self.task

    class Executor:
        def __init__(self, **kwargs) -> None:
            assert kwargs["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def submit(self, _worker, task):
            return Future(task)

    def wait_one(pending, **_):
        pending_sizes.append(len(pending))
        first = next(iter(pending))
        return {first}, set(pending) - {first}

    monkeypatch.setattr(sweep, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(sweep, "wait", wait_one)
    sweep._run_bounded_pool(
        ({"index": index} for index in range(9)),
        processes=2,
        install=lambda task, result: installed.append(result["index"]),
    )
    assert max(pending_sizes) == 4
    assert all(size <= 4 for size in pending_sizes)
    assert sorted(installed) == list(range(9))


@pytest.mark.parametrize("shots", [0, 1_000_001, True])
def test_shot_bound_is_exact(shots: int) -> None:
    with pytest.raises((TypeError, ValueError), match="shots_per_cell"):
        sweep._validate_shots(shots)


@pytest.mark.parametrize("p", [0, 0.2000000001, 1, -0.1, float("inf"), float("nan")])
def test_si1000_probability_must_be_in_frozen_domain(
    p: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"0 < p <= 0\.2"):
        sweep._validate_probability(p)


@pytest.mark.parametrize("p", [1e-15, 0.001, 0.2])
def test_si1000_probability_model_validator_accepts_domain_values(p: float) -> None:
    assert sweep._validate_probability(p) == p


def test_si1000_probability_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="number"):
        sweep._validate_probability(True)
