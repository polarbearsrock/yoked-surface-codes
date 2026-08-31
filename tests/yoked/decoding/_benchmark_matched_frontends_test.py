from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "benchmark_matched_frontends"
DRAFT = ROOT / "docs" / "MATCHED_FRONTENDS_D7_P003_DRAFT.json"


@pytest.fixture(scope="module")
def tool() -> dict[str, object]:
    return runpy.run_path(str(TOOL), run_name="matched_frontends_cli_test_module")


def test_draft_protocol_is_valid_for_import_but_not_collection(tool) -> None:
    load = tool["_load_protocol"]
    value = load(DRAFT, require_frozen=False)
    assert value["cell"]["d"] == 7
    assert value["cell"]["p"] == 0.003
    assert value["accuracy"] == {
        "arm_order": ["global", "promatch", "pinball", "union_find"],
        "microbatch_size": 32,
        "processes": 32,
        "ranges": 32,
        "shots": 10_000,
    }
    with pytest.raises(ValueError, match="frozen protocol"):
        load(DRAFT, require_frozen=True)


def test_frozen_protocol_requires_exact_decoder_configs(tool, tmp_path: Path) -> None:
    value = json.loads(DRAFT.read_text())
    value.update(
        {
            "status": "FROZEN",
            "frozen": True,
            "implementation_commit": "1" * 40,
        }
    )
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(value))
    assert tool["_load_protocol"](path, require_frozen=True)["frozen"] is True
    changed = copy.deepcopy(value)
    changed["promatch_config"]["residual_hw_limit"] = 9
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="ProMatch config"):
        tool["_load_protocol"](path, require_frozen=False)


def test_status_is_read_only_and_reports_absent_outputs(
    tool, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main = tool["main"]
    code = main(
        [
            "status",
            "--protocol",
            str(DRAFT),
            "--accuracy-collection",
            str(tmp_path / "accuracy"),
            "--latency-collection",
            str(tmp_path / "latency"),
        ]
    )
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "DRAFT"
    assert output["accuracy_ranges"] == 0
    assert output["accuracy_complete"] is False
    assert output["latency_restarts"] == 0
    assert output["latency_complete"] is False


def test_process_count_gate_precedes_compilation(tool, tmp_path: Path) -> None:
    namespace = tool["argparse"].Namespace(
        protocol=DRAFT,
        corpus=tmp_path / "missing-corpus",
        out=tmp_path / "out",
        processes=31,
    )
    with pytest.raises(ValueError, match="exactly 32"):
        tool["_cmd_collect_accuracy"](namespace)
