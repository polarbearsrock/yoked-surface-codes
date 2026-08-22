from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

import yoked.decoding._fig8_paired_aws_sweep as sweep


def _identity() -> dict[str, str]:
    return {
        "instance_type": "c8a.48xlarge",
        "region": "us-east-1",
        "availability_zone": "us-east-1a",
        "lifecycle": "spot",
    }


def _topology() -> dict:
    return {
        "visible_cpus": list(range(192)),
        "physical_core_count": 192,
        "threads_per_core": 1,
        "memory_total_kib": 384 * 1024 * 1024,
        "nodes": [
            {
                "node_id": node,
                "cpus": list(range(node * 96, (node + 1) * 96)),
                "memory_total_kib": 190 * 1024 * 1024,
            }
            for node in range(2)
        ],
    }


def _provenance(index: int) -> dict[str, object]:
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
    source_paths = sorted(
        {"source.py", *sweep.base.REQUIRED_SWEEP_SOURCE_PATHS, *sweep.REQUIRED_AWS_SOURCE_PATHS}
    )
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
    monkeypatch.setattr(sweep, "current_execution_environment", lambda: {"machine": "x"})
    monkeypatch.setattr(sweep, "_source_paths", lambda _: source_paths)
    monkeypatch.setattr(sweep, "_canonical_file_hash", lambda _: "b" * 64)
    monkeypatch.setattr(
        sweep,
        "validate_aws_host",
        lambda: {
            "schema": sweep.HOST_CHECK_SCHEMA,
            "valid": True,
            "aws_identity": _identity(),
            "numa_topology": _topology(),
            "worker_layout": sweep._worker_layout(_topology()),
        },
    )
    calls = 0

    def prepare(_cell, **_kwargs):
        nonlocal calls
        result = SimpleNamespace(provenance=_provenance(calls))
        calls += 1
        return result

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep.secrets, "token_hex", lambda _: "c" * 64)
    out = tmp_path / "campaign"
    frozen = sweep.create_campaign(out, p=0.001, shots_per_cell=2_001, root=repo)
    assert calls == 16
    return out, frozen


def test_create_freezes_separate_192_worker_2x96_profile(
    campaign: tuple[Path, dict],
) -> None:
    out, frozen = campaign
    assert set(frozen) == sweep.CAMPAIGN_FIELDS
    assert frozen["schema"] == sweep.CAMPAIGN_SCHEMA
    assert frozen["processes"] == 192
    assert frozen["worker_layout"]["pool_count"] == 2
    assert frozen["worker_layout"]["workers_per_pool"] == 96
    assert [pool["cpus"] for pool in frozen["worker_layout"]["pools"]] == [
        list(range(96)),
        list(range(96, 192)),
    ]
    assert sweep.load_campaign(out) == frozen


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda identity, topology: identity.update(instance_type="c7a.48xlarge"), "instance_type"),
        (lambda identity, topology: identity.update(lifecycle="on-demand"), "lifecycle"),
        (lambda identity, topology: topology.update(physical_core_count=96), "physical cores"),
        (lambda identity, topology: topology.update(threads_per_core=2), "hardware thread"),
        (lambda identity, topology: topology.update(memory_total_kib=100), "350 GiB"),
        (
            lambda identity, topology: topology["nodes"][0].update(memory_total_kib=100),
            "175 GiB",
        ),
    ],
)
def test_host_check_fails_closed_on_wrong_aws_shape(mutation, message: str) -> None:
    identity = _identity()
    topology = _topology()
    mutation(identity, topology)
    with pytest.raises(ValueError, match=message):
        sweep.validate_aws_host(identity=identity, topology=topology)


def test_imdsv2_uses_token_then_authenticated_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class Response:
        def __init__(self, value: bytes):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.value

    responses = iter((Response(b"token-value"), Response(b"c8a.48xlarge")))

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return next(responses)

    monkeypatch.setattr(sweep.urllib.request, "urlopen", urlopen)
    assert sweep._imds_get("instance-type") == "c8a.48xlarge"
    assert requests[0][0].method == "PUT"
    assert requests[0][0].full_url.endswith("/latest/api/token")
    assert requests[1][0].full_url.endswith("/latest/meta-data/instance-type")
    assert requests[1][0].get_header("X-aws-ec2-metadata-token") == "token-value"
    assert [timeout for _, timeout in requests] == [2.0, 2.0]


def test_imdsv2_failure_is_reported_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sweep.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("no IMDS")),
    )
    with pytest.raises(RuntimeError, match="IMDSv2"):
        sweep._imds_get("instance-type")


def test_campaign_rejects_worker_layout_or_aws_contract_drift(
    campaign: tuple[Path, dict],
) -> None:
    _, frozen = campaign
    changed = copy.deepcopy(frozen)
    changed["worker_layout"]["workers_per_pool"] = 95
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="worker layout"):
        sweep.validate_campaign(changed)

    changed = copy.deepcopy(frozen)
    changed["aws_identity"]["lifecycle"] = "on-demand"
    changed["experiment_id"] = sweep.manifest_experiment_id(changed)
    with pytest.raises(ValueError, match="lifecycle"):
        sweep.validate_campaign(changed)


def test_collection_runtime_rejects_aws_identity_or_topology_drift(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    for name in sweep.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("MAX_ERRORS", raising=False)
    monkeypatch.setattr(sweep, "validate_analysis_runtime", lambda *_, **__: None)
    monkeypatch.setattr(
        sweep, "current_execution_environment", lambda: frozen["execution_environment"]
    )
    wrong_identity = copy.deepcopy(frozen["aws_identity"])
    wrong_identity["availability_zone"] = "us-east-1b"
    monkeypatch.setattr(
        sweep,
        "validate_aws_host",
        lambda: {
            "aws_identity": wrong_identity,
            "numa_topology": frozen["numa_topology"],
        },
    )
    with pytest.raises(ValueError, match="AWS identity"):
        sweep._validate_runtime(frozen, processes=192, root=out.parent / "repo")

    wrong_topology = copy.deepcopy(frozen["numa_topology"])
    wrong_topology["memory_total_kib"] += 1
    monkeypatch.setattr(
        sweep,
        "validate_aws_host",
        lambda: {
            "aws_identity": frozen["aws_identity"],
            "numa_topology": wrong_topology,
        },
    )
    with pytest.raises(ValueError, match="NUMA topology"):
        sweep._validate_runtime(frozen, processes=192, root=out.parent / "repo")


def _task(batch_id: int) -> dict:
    return {
        "batch": {"batch_id": batch_id, "shot_start": batch_id * 1_000, "shots": 1_000}
    }


def test_batch_id_modulo_partition_is_complete_disjoint_and_deterministic() -> None:
    tasks = [_task(batch_id) for batch_id in range(17)]
    first, second = sweep._partition_tasks(tasks)
    assert [task["batch"]["batch_id"] for task in first] == list(range(0, 17, 2))
    assert [task["batch"]["batch_id"] for task in second] == list(range(1, 17, 2))
    assert sorted(id(task) for task in first + second) == sorted(id(task) for task in tasks)


def test_numa_pool_uses_two_96_worker_executors_and_bounds_each_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors = []
    pending_seen = []
    installed = []

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            return self.task

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 96
            assert kwargs["initializer"] is sweep._initialize_numa_worker
            self.cpus = kwargs["initargs"][0]
            self.shutdown_calls = []
            executors.append(self)

        def submit(self, _worker, task):
            return Future(task)

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    def wait_one(pending, **_kwargs):
        by_pool = [0, 0]
        for future in pending:
            by_pool[future.task["batch"]["batch_id"] % 2] += 1
        pending_seen.append(tuple(by_pool))
        first = next(iter(pending))
        return {first}, set(pending) - {first}

    monkeypatch.setattr(sweep, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(sweep, "wait", wait_one)
    topology = _topology()
    sweep._run_numa_pools(
        [_task(batch_id) for batch_id in range(450)],
        layout=sweep._worker_layout(topology),
        install=lambda task, row: installed.append((task, row)),
    )
    assert len(executors) == 2
    assert executors[0].cpus == tuple(range(96))
    assert executors[1].cpus == tuple(range(96, 192))
    assert max(count for pair in pending_seen for count in pair) <= 192
    assert len(installed) == 450
    assert [executor.shutdown_calls for executor in executors] == [
        [{"wait": True}],
        [{"wait": True}],
    ]


def test_worker_sets_affinity_before_configuring_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    current = set(range(192))

    def set_affinity(_pid, cpus):
        calls.append("affinity")
        current.clear()
        current.update(cpus)

    def configure():
        calls.append("threads")

    class Cache(dict):
        def clear(self):
            calls.append("cache")
            super().clear()

    monkeypatch.setattr(sweep.os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(sweep.os, "sched_getaffinity", lambda _pid: current)
    monkeypatch.setattr(sweep, "configure_single_thread_runtime", configure)
    monkeypatch.setattr(sweep, "_WORKER_CACHE", Cache())
    sweep._initialize_numa_worker(tuple(range(96)))
    assert calls == ["affinity", "threads", "cache"]


def test_campaign_lock_rejects_a_second_coordinator(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with sweep._exclusive_campaign_lock(campaign):
        with pytest.raises(RuntimeError, match="another coordinator"):
            with sweep._exclusive_campaign_lock(campaign):
                pytest.fail("second coordinator acquired the lock")


def test_pool_failure_cancels_queued_work(monkeypatch: pytest.MonkeyPatch) -> None:
    executors = []

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            if self.task["batch"]["batch_id"] == 0:
                raise RuntimeError("worker failed")
            return self.task

    class Executor:
        def __init__(self, **_kwargs):
            self.shutdown_calls = []
            executors.append(self)

        def submit(self, _worker, task):
            return Future(task)

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(sweep, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        sweep,
        "wait",
        lambda pending, **_kwargs: ({next(iter(pending))}, set()),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        sweep._run_numa_pools(
            [_task(batch_id) for batch_id in range(4)],
            layout=sweep._worker_layout(_topology()),
            install=lambda *_args: None,
        )
    assert [executor.shutdown_calls for executor in executors] == [
        [{"wait": False, "cancel_futures": True}],
        [{"wait": False, "cancel_futures": True}],
    ]


def test_resume_validates_then_atomically_installs_only_missing_batches(
    campaign: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, frozen = campaign
    first_cell = frozen["cells"][0]["cell_id"]
    existing_key = (first_cell, 0)
    collection = out / "collection"
    prepared_cells = []
    submitted = []
    validated = []
    installed = []
    sentinel = {"complete": True}

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

    def validate(_row, **kwargs):
        validated.append((kwargs["cell"]["cell_id"], kwargs["batch"].batch_id))

    def run_pools(tasks, *, layout, install):
        assert layout == frozen["worker_layout"]
        submitted.extend(tasks)
        for task in submitted:
            install(task, {"installed_batch": task["batch"]["batch_id"]})

    monkeypatch.setattr(sweep, "prepare_cell", prepare)
    monkeypatch.setattr(sweep, "_validate_ledger_row", validate)
    monkeypatch.setattr(sweep, "_run_numa_pools", run_pools)
    monkeypatch.setattr(
        sweep, "_atomic_json_write", lambda path, row: installed.append((path, row))
    )
    monkeypatch.setattr(sweep, "campaign_status", lambda _: sentinel)

    assert (
        sweep._run_campaign_locked(out, processes=192, root=out.parent / "repo")
        is sentinel
    )
    assert prepared_cells == [cell["cell_id"] for cell in frozen["cells"]]
    assert len(submitted) == 47
    assert all(
        (task["cell"]["cell_id"], task["batch"]["batch_id"]) != existing_key
        for task in submitted
    )
    assert len(validated) == len(installed) == 47
    for path, row in installed:
        assert path.parent.parent.parent == collection
        assert path.name == f"batch-{row['installed_batch']:08d}.json"


def test_run_rejects_any_process_count_other_than_192(
    campaign: tuple[Path, dict],
) -> None:
    out, _ = campaign
    with pytest.raises(ValueError, match="exactly 192"):
        sweep.run_campaign(out, processes=96)
