"""Canonical, atomic analysis installation and bit-exact verification.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` writes the
analysis products (tables, plot data, plots, report, manifest, casebook
selection, ANALYSIS_READY) atomically and verifies an installed analysis by
recomputing every canonical byte.  It inherits the package's downstream-only
contract: it never imports circuit generation, sampling, matching, or
decoding code, which is why the local atomic-install helper mirrors, rather
than imports, :func:`yoked.decoding._artifact_io.install_bytes_atomic`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._contract import (
    ANALYSIS_MANIFEST_SCHEMA,
    ANALYSIS_READY_SCHEMA,
    HUMAN_REPORT_FILE,
    PolicyAnalysisError,
    _sha256,
    canonical_json_bytes,
)
from ._corpus import PolicyAuditCorpus, _load_json
from ._fields import _at
from ._plots import _plot_payloads, _render_plots
from ._report import policy_human_report_bytes


def _write_canonical_json(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    return _sha256(data)


def _install_bytes_atomic(
    data: bytes, destination: Path, *, scratch_root: str, prefix: str
) -> None:
    """Atomically installs ``data`` at ``destination`` (tmp + fsync + replace).

    Sync note: :func:`yoked.decoding._artifact_io.install_bytes_atomic` is
    this helper's infrastructure twin.  The package's downstream-only
    contract forbids importing that decoding-infrastructure module, so the
    mkstemp/fsync/replace dance is mirrored here with the caller-validated
    ``scratch_root`` (already required to share a filesystem with the audit
    root) instead of the twin's TMPDIR/EXDEV restaging.  Keep the
    write/fsync/replace semantics of the two implementations aligned.
    """

    fd, temporary_name = tempfile.mkstemp(prefix=prefix, dir=scratch_root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_policy_analysis(
    corpus: PolicyAuditCorpus,
    analysis: Mapping[str, Any],
    *,
    render_plots: bool = True,
) -> dict[str, Any]:
    """Installs analysis products and writes ANALYSIS_READY last.

    ``COMPLETE`` is deliberately not written here.  It belongs to the later
    casebook-expansion/finalization stage after exhaustive sidecars verify.
    """

    root = corpus.root
    gate_rows = _at(analysis, "tables.fatal_gates", required=True)
    passing_statuses = {"passed-ledger-recomputed", "collector-attested"}
    if not isinstance(gate_rows, list):
        raise PolicyAnalysisError(
            "tables.fatal_gates must be an array; ANALYSIS_READY refused"
        )
    if [row.get("gate") for row in gate_rows if isinstance(row, Mapping)] != list(
        range(1, 19)
    ):
        raise PolicyAnalysisError(
            "tables.fatal_gates gate numbering is not exactly 1..18; ANALYSIS_READY refused"
        )
    for row in gate_rows:
        if not isinstance(row, Mapping):
            raise PolicyAnalysisError(
                "fatal gate row must be an object; ANALYSIS_READY refused"
            )
        if row.get("status") not in passing_statuses:
            raise PolicyAnalysisError(
                f"fatal gate {row.get('gate')} status is not a passing status; "
                "ANALYSIS_READY refused"
            )
        if not isinstance(row.get("evidence"), Mapping):
            raise PolicyAnalysisError(
                f"fatal gate {row.get('gate')} evidence must be an object; "
                "ANALYSIS_READY refused"
            )
        if not row["evidence"].get("authenticated_by"):
            raise PolicyAnalysisError(
                f"fatal gate {row.get('gate')} evidence lacks authenticated_by; "
                "ANALYSIS_READY refused"
            )
    if (root / "analysis").exists() or (root / "ANALYSIS_READY").exists():
        raise PolicyAnalysisError(
            "analysis output already exists; immutable rerun refused"
        )
    scratch_root = os.environ.get("TMPDIR")
    if not scratch_root:
        raise PolicyAnalysisError("TMPDIR must be set for atomic analysis installation")
    if os.stat(scratch_root).st_dev != os.stat(root).st_dev:
        raise PolicyAnalysisError("TMPDIR and audit root must share a filesystem")
    temporary = Path(
        tempfile.mkdtemp(prefix="promatch-policy-analysis-", dir=scratch_root)
    )
    installed = False
    try:
        tables_dir = temporary / "tables"
        plot_data_dir = temporary / "plot-data"
        plot_dir = temporary / "plots"
        tables_dir.mkdir()
        plot_data_dir.mkdir()
        plot_dir.mkdir()
        table_hashes = {}
        tables = analysis["tables"]
        if not isinstance(tables, Mapping):
            raise PolicyAnalysisError("analysis tables must be an object")
        for name, table in sorted(tables.items()):
            table_hashes[f"tables/{name}.json"] = _write_canonical_json(
                tables_dir / f"{name}.json", table
            )
        plot_payloads = _plot_payloads(tables)
        plot_data_hashes = {}
        for name, payload in sorted(plot_payloads.items()):
            plot_data_hashes[f"plot-data/{name}.json"] = _write_canonical_json(
                plot_data_dir / f"{name}.json", payload
            )
        rendered = _render_plots(plot_dir, plot_payloads) if render_plots else []
        summary_hash = _write_canonical_json(temporary / "summary.json", analysis)
        report_data = policy_human_report_bytes(analysis)
        (temporary / HUMAN_REPORT_FILE).write_bytes(report_data)
        report_hash = _sha256(report_data)
        manifest = {
            "schema": ANALYSIS_MANIFEST_SCHEMA,
            "experiment_id": analysis["experiment_id"],
            "analysis_sha256": analysis["analysis_sha256"],
            "summary_file_sha256": summary_hash,
            "report_file_sha256": report_hash,
            "source_hashes": corpus.source_hashes,
            "table_file_hashes": dict(sorted(table_hashes.items())),
            "plot_data_file_hashes": dict(sorted(plot_data_hashes.items())),
            "plot_images": [f"plots/{name}" for name in sorted(rendered)],
            "plot_images_scientifically_digested": False,
        }
        manifest_hash = _write_canonical_json(temporary / "manifest.json", manifest)

        casebook_dir = root / "casebook"
        casebook_dir.mkdir(exist_ok=True)
        selection_path = casebook_dir / "selection.json"
        selection_data = canonical_json_bytes(analysis["casebook_selection"]) + b"\n"
        if selection_path.exists():
            if selection_path.read_bytes() != selection_data:
                raise PolicyAnalysisError("existing casebook selection differs")
        else:
            _install_bytes_atomic(
                selection_data,
                selection_path,
                scratch_root=scratch_root,
                prefix="promatch-policy-casebook-",
            )

        os.replace(temporary, root / "analysis")
        installed = True
        ready = {
            "schema": ANALYSIS_READY_SCHEMA,
            "experiment_id": analysis["experiment_id"],
            "analysis_manifest_sha256": manifest_hash,
            "casebook_selection_sha256": _sha256(selection_data),
            "report_file_sha256": report_hash,
            "plots_rendered": render_plots,
            "casebook_exhaustive_expansion_required_before_complete": True,
        }
        _install_bytes_atomic(
            canonical_json_bytes(ready) + b"\n",
            root / "ANALYSIS_READY",
            scratch_root=scratch_root,
            prefix="promatch-policy-analysis-ready-",
        )
        return manifest
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def verify_existing_policy_analysis(
    corpus: PolicyAuditCorpus, analysis: Mapping[str, Any]
) -> dict[str, Any]:
    """Recomputes and authenticates an already-installed offline analysis."""

    root = corpus.root
    analysis_dir = root / "analysis"
    if not analysis_dir.is_dir():
        raise PolicyAnalysisError("audit root has no installed analysis")
    marker = root / "ANALYSIS_READY"
    if not marker.is_file():
        raise PolicyAnalysisError(
            "installed analysis has no ANALYSIS_READY marker; COMPLETE is not an analysis substitute"
        )
    expected_summary_bytes = canonical_json_bytes(analysis) + b"\n"
    summary_path = analysis_dir / "summary.json"
    if (
        not summary_path.is_file()
        or summary_path.read_bytes() != expected_summary_bytes
    ):
        raise PolicyAnalysisError(
            "installed summary bytes differ from recomputed canonical analysis"
        )
    analysis_without_digest = dict(analysis)
    claimed_analysis_digest = analysis_without_digest.pop("analysis_sha256", None)
    if claimed_analysis_digest != _sha256(
        canonical_json_bytes(analysis_without_digest)
    ):
        raise PolicyAnalysisError("recomputed analysis self digest is invalid")
    manifest = _load_json(analysis_dir / "manifest.json")
    manifest_fields = {
        "schema",
        "experiment_id",
        "analysis_sha256",
        "summary_file_sha256",
        "report_file_sha256",
        "source_hashes",
        "table_file_hashes",
        "plot_data_file_hashes",
        "plot_images",
        "plot_images_scientifically_digested",
    }
    if set(manifest) != manifest_fields:
        raise PolicyAnalysisError("installed analysis manifest has unexpected fields")
    if manifest.get("schema") != ANALYSIS_MANIFEST_SCHEMA:
        raise PolicyAnalysisError("installed analysis manifest has the wrong schema")
    if manifest.get("experiment_id") != analysis.get("experiment_id"):
        raise PolicyAnalysisError(
            "installed analysis manifest has the wrong experiment identity"
        )
    if manifest.get("analysis_sha256") != claimed_analysis_digest:
        raise PolicyAnalysisError("installed manifest has the wrong analysis digest")
    if manifest.get("summary_file_sha256") != _sha256(expected_summary_bytes):
        raise PolicyAnalysisError("installed analysis summary digest mismatch")
    expected_report_bytes = policy_human_report_bytes(analysis)
    report_path = analysis_dir / HUMAN_REPORT_FILE
    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or report_path.read_bytes() != expected_report_bytes
    ):
        raise PolicyAnalysisError(
            "installed human report bytes differ from recomputation"
        )
    if manifest.get("report_file_sha256") != _sha256(expected_report_bytes):
        raise PolicyAnalysisError("installed human report digest mismatch")
    if manifest.get("source_hashes") != corpus.source_hashes:
        raise PolicyAnalysisError(
            "installed analysis manifest has the wrong source hashes"
        )
    if manifest.get("plot_images_scientifically_digested") is not False:
        raise PolicyAnalysisError("plot images must remain explicitly non-scientific")

    tables = analysis.get("tables")
    if not isinstance(tables, Mapping):
        raise PolicyAnalysisError("recomputed analysis tables must be an object")
    expected_tables = {
        f"tables/{name}.json": canonical_json_bytes(table) + b"\n"
        for name, table in sorted(tables.items())
    }
    plot_payloads = _plot_payloads(tables)
    expected_plot_data = {
        f"plot-data/{name}.json": canonical_json_bytes(payload) + b"\n"
        for name, payload in sorted(plot_payloads.items())
    }
    for group, expected_files in (
        ("table_file_hashes", expected_tables),
        ("plot_data_file_hashes", expected_plot_data),
    ):
        records = manifest.get(group)
        expected_hashes = {
            relative: _sha256(data) for relative, data in expected_files.items()
        }
        if records != expected_hashes:
            raise PolicyAnalysisError(
                f"installed {group} differs from recomputed artifacts"
            )
        for relative, expected_bytes in expected_files.items():
            path = analysis_dir / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != expected_bytes
            ):
                raise PolicyAnalysisError(
                    f"installed analysis bytes differ from recomputation: {relative}"
                )
    plot_images = manifest.get("plot_images")
    if not isinstance(plot_images, list) or any(
        not isinstance(item, str) for item in plot_images
    ):
        raise PolicyAnalysisError(
            "installed analysis manifest has invalid plot image paths"
        )
    for relative in plot_images:
        raw_path = analysis_dir / relative
        path = raw_path.resolve()
        if (
            analysis_dir.resolve() not in path.parents
            or raw_path.is_symlink()
            or not path.is_file()
        ):
            raise PolicyAnalysisError(
                f"installed plot image is missing or unsafe: {relative}"
            )

    ready = _load_json(marker)
    expected_ready_fields = {
        "schema",
        "experiment_id",
        "analysis_manifest_sha256",
        "casebook_selection_sha256",
        "report_file_sha256",
        "plots_rendered",
        "casebook_exhaustive_expansion_required_before_complete",
    }
    if set(ready) != expected_ready_fields:
        raise PolicyAnalysisError("ANALYSIS_READY has unexpected fields")
    if ready.get("schema") != ANALYSIS_READY_SCHEMA:
        raise PolicyAnalysisError("ANALYSIS_READY has the wrong schema")
    if ready.get("experiment_id") != analysis.get("experiment_id"):
        raise PolicyAnalysisError("ANALYSIS_READY has the wrong experiment identity")
    if ready.get("analysis_manifest_sha256") != _sha256(
        (analysis_dir / "manifest.json").read_bytes()
    ):
        raise PolicyAnalysisError("ANALYSIS_READY manifest digest mismatch")
    if ready.get("plots_rendered") is not bool(plot_images):
        raise PolicyAnalysisError("ANALYSIS_READY plot-rendering flag is inconsistent")
    if ready.get("casebook_exhaustive_expansion_required_before_complete") is not True:
        raise PolicyAnalysisError(
            "ANALYSIS_READY omitted the required expansion contract"
        )
    selection = root / "casebook" / "selection.json"
    expected_selection = canonical_json_bytes(analysis["casebook_selection"]) + b"\n"
    if not selection.is_file() or selection.read_bytes() != expected_selection:
        raise PolicyAnalysisError("installed casebook selection differs")
    if ready.get("casebook_selection_sha256") != _sha256(expected_selection):
        raise PolicyAnalysisError("ANALYSIS_READY selection digest mismatch")
    if ready.get("report_file_sha256") != _sha256(expected_report_bytes):
        raise PolicyAnalysisError("ANALYSIS_READY report digest mismatch")
    return manifest
