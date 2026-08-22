from __future__ import annotations

import copy
from pathlib import Path

import pytest

import yoked.decoding._fig8_paired_aws_ondemand_continuation as continuation


def _identity(lifecycle: str) -> dict[str, str]:
    return {
        "instance_type": "c8a.48xlarge",
        "region": "us-east-1",
        "availability_zone": "us-east-1c",
        "lifecycle": lifecycle,
    }


def _topology(*, memory_delta: int = 0) -> dict:
    return {
        "visible_cpus": list(range(192)),
        "physical_core_count": 192,
        "threads_per_core": 1,
        "memory_total_kib": 384 * 1024 * 1024 + memory_delta,
        "nodes": [
            {
                "node_id": node,
                "cpus": list(range(node * 96, (node + 1) * 96)),
                "memory_total_kib": 190 * 1024 * 1024 + memory_delta,
            }
            for node in range(2)
        ],
    }


def _campaign() -> dict:
    return {
        "experiment_id": "e" * 64,
        "repository_commit": "a" * 40,
        "source_hashes": {"frozen.py": "f" * 64},
        "software_versions": {"python": "x"},
        "execution_environment": {"kernel": "x"},
        "aws_identity": _identity("spot"),
        "numa_topology": _topology(),
        "cell_batch_schedules": {
            "cell": [
                {"batch_id": 0, "shot_start": 0, "shots": 1000},
                {"batch_id": 1, "shot_start": 1000, "shots": 1000},
            ]
        },
    }


def test_ondemand_host_accepts_only_lifecycle_and_memory_report_difference() -> None:
    frozen_identity = _identity("spot")
    assert continuation._validate_ondemand_identity(
        _identity("on-demand"), frozen=frozen_identity
    )["lifecycle"] == "on-demand"
    assert continuation._validate_compatible_topology(
        _topology(memory_delta=-1024), frozen=_topology()
    )["memory_total_kib"] == _topology(memory_delta=-1024)["memory_total_kib"]

    wrong_identity = _identity("on-demand")
    wrong_identity["availability_zone"] = "us-east-1b"
    with pytest.raises(ValueError, match="availability_zone"):
        continuation._validate_ondemand_identity(
            wrong_identity, frozen=frozen_identity
        )

    wrong_topology = _topology()
    wrong_topology["nodes"][1]["cpus"][-1] = 0
    with pytest.raises(ValueError, match="NUMA node"):
        continuation._validate_compatible_topology(
            wrong_topology, frozen=_topology()
        )


@pytest.fixture
def prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict, dict]:
    campaign = _campaign()
    directory = tmp_path / "campaign"
    ledger = directory / "collection/batches/cell/batch-00000000.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"batch":{"shots":1000}}\n', encoding="utf-8")
    (directory / "campaign.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    monkeypatch.setattr(continuation.spot, "load_campaign", lambda _: campaign)
    monkeypatch.setattr(
        continuation.spot,
        "_validate_existing_ledgers",
        lambda *_: (
            {("cell", 0): {"batch": {"shots": 1000}}},
            directory / "collection",
        ),
    )
    monkeypatch.setattr(
        continuation,
        "_validate_repository_for_prepare",
        lambda *_args, **_kwargs: {
            "repository_commit": "b" * 40,
            "clean_worktree": True,
        },
    )
    monkeypatch.setattr(
        continuation, "current_execution_environment", lambda: {"kernel": "x"}
    )
    monkeypatch.setattr(
        continuation, "current_software_versions", lambda: {"python": "x"}
    )
    topology = _topology(memory_delta=-1024)
    monkeypatch.setattr(
        continuation,
        "_current_host",
        lambda _: {
            "aws_identity": _identity("on-demand"),
            "instance_id": "i-ondemand",
            "ami_id": "ami-test",
            "numa_topology": topology,
            "worker_layout": continuation.spot._worker_layout(topology),
        },
    )
    source_hashes = {
        relative: str(index + 1).zfill(64)
        for index, relative in enumerate(sorted(continuation.CONTINUATION_SOURCE_PATHS))
    }
    monkeypatch.setattr(
        continuation, "_continuation_source_hashes", lambda _: source_hashes
    )

    campaign_before = (directory / "campaign.json").read_bytes()
    record = continuation.prepare_continuation(directory, root=tmp_path / "repo")
    assert (directory / "campaign.json").read_bytes() == campaign_before
    return directory, campaign, record


def test_prepare_freezes_boundary_without_mutating_campaign(
    prepared: tuple[Path, dict, dict],
) -> None:
    directory, campaign, record = prepared
    assert record["campaign_experiment_id"] == campaign["experiment_id"]
    assert record["baseline_completed_batches"] == 1
    assert record["baseline_completed_shots"] == 1000
    assert record["missing_batches_at_prepare"] == 1
    assert list(record["baseline_ledger_hashes"]) == [
        "collection/batches/cell/batch-00000000.json"
    ]
    assert continuation.continuation_record_path(directory).is_file()


def test_baseline_mutation_is_detected(
    prepared: tuple[Path, dict, dict],
) -> None:
    directory, _, record = prepared
    ledger = directory / next(iter(record["baseline_ledger_hashes"]))
    ledger.write_text('{"batch":{"shots":999}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="baseline ledger changed"):
        continuation._validate_baseline_ledgers(directory, record)


def test_run_reuses_original_collector_after_continuation_validation(
    prepared: tuple[Path, dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, campaign, record = prepared
    root = directory.parent / "repo"
    monkeypatch.setattr(
        continuation,
        "_validate_runtime",
        lambda *_args, **_kwargs: (campaign, record),
    )
    original_validator = continuation.spot._validate_runtime
    calls = []

    def run_original(_directory, *, processes, root):
        continuation.spot._validate_runtime(
            campaign, processes=processes, root=root
        )
        calls.append((campaign["experiment_id"], processes))
        return {
            "schema": "original-status",
            "experiment_id": campaign["experiment_id"],
            "complete": True,
            "totals": {},
            "cells": [],
        }

    monkeypatch.setattr(continuation.spot, "_run_campaign_locked", run_original)
    result = continuation.run_continuation(
        directory, processes=192, root=root
    )
    assert calls == [(campaign["experiment_id"], 192)]
    assert result["experiment_id"] == campaign["experiment_id"]
    assert result["continuation"]["baseline_completed_batches"] == 1
    assert continuation.spot._validate_runtime is original_validator


def test_continuation_record_rejects_changed_experiment_id(
    prepared: tuple[Path, dict, dict],
) -> None:
    directory, campaign, record = prepared
    changed = copy.deepcopy(record)
    changed["campaign_experiment_id"] = "d" * 64
    changed["continuation_id"] = continuation._continuation_id(changed)
    with pytest.raises(ValueError, match="experiment ID"):
        continuation.validate_continuation(
            changed, campaign=campaign, directory=directory
        )
