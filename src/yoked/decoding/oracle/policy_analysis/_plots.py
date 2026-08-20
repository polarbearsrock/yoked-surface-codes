"""Frozen plot payload construction and non-scientific figure rendering.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` freezes the plot
payload tables and renders the fixed set of descriptive figures from them;
rendering is a final, separate step so no presentation code can influence
the measured records.  It inherits the package's downstream-only contract:
it never imports circuit generation, sampling, matching, or decoding code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._contract import (
    CERTIFICATE_CLASSES,
    CONTEXT_PRIORITY,
    PLOT_TABLE_SCHEMA,
    PolicyAnalysisError,
)


def _plot_payloads(tables: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "certificate-flow": {
            "schema": PLOT_TABLE_SCHEMA,
            "certificate_by_stage": tables["certificate_by_stage"],
            "terminal_action": tables["counterfactual_terminal_action"],
        },
        "unsafe-fraction-by-stage": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["unsafe_fraction_by_stage"],
        },
        "first-conflict-stage-context": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["first_conflict_discordant"],
            "context_views": tables["context_views"],
        },
        "cost-excess-ecdf": {
            "schema": PLOT_TABLE_SCHEMA,
            "by_stage": tables["cost_excess_ecdf_by_stage"],
            "by_certificate": tables["cost_excess_ecdf_by_certificate"],
            "by_context": tables["cost_excess_ecdf_by_context"],
            "tolerance_tau_k": tables["continuous_distributions"][
                "cost_tolerance_tau_k"
            ],
        },
        "first-safe-action-rank": {
            "schema": PLOT_TABLE_SCHEMA,
            "actions": tables["counterfactual_terminal_action"],
            "ranks": tables["first_safe_rank"],
        },
        "stage-transition-matrix": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["stage_transition"],
        },
        "original-versus-alternative": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["original_vs_alternative"],
        },
        "risk-heatmaps": {
            "schema": PLOT_TABLE_SCHEMA,
            "tables": tables["risk_heatmaps"],
        },
        "disagreement-association": {
            "schema": PLOT_TABLE_SCHEMA,
            "rows": tables["association_by_unsafe_count"],
        },
        "event-relief": {
            "schema": PLOT_TABLE_SCHEMA,
            "summary": tables["event_and_transaction_summary"],
            "distributions": tables["residual_hw_distributions"],
        },
        "veto-chain-tails": {
            "schema": PLOT_TABLE_SCHEMA,
            "metrics": tables["veto_chain_tails"],
        },
    }


def _save_plot(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(
        path, dpi=180, bbox_inches="tight", metadata={"Software": "yoked-policy-audit"}
    )
    import matplotlib.pyplot as plt

    plt.close(figure)


def _draw_certificate_flow(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["certificate_by_stage"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(4)
    for certificate in CERTIFICATE_CLASSES:
        values = [
            next(
                row["count"]
                for row in rows
                if row["stage"] == stage and row["certificate_class"] == certificate
            )
            for stage in range(1, 5)
        ]
        ax.bar(range(1, 5), values, bottom=bottom, label=certificate)
        bottom += values
    ax.set(
        xlabel="ProMatch stage",
        ylabel="durable shadow commitments",
        title="Certificate flow by original stage",
    )
    ax.legend(fontsize=8)
    return fig


def _draw_unsafe_fraction_by_stage(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(1, 5)
    y = np.array(
        [row["fraction"] if row["fraction"] is not None else np.nan for row in rows]
    )
    lower = np.array(
        [
            row["bootstrap"]["lower"]
            if row["bootstrap"]["lower"] is not None
            else np.nan
            for row in rows
        ]
    )
    upper = np.array(
        [
            row["bootstrap"]["upper"]
            if row["bootstrap"]["upper"] is not None
            else np.nan
            for row in rows
        ]
    )
    ax.errorbar(
        x,
        y,
        yerr=np.maximum(0, np.vstack((y - lower, upper - y))),
        marker="o",
        capsize=4,
    )
    ax.set(
        xlabel="ProMatch stage",
        ylabel="O-frame-unsafe fraction",
        title="Unsafe durable commitments by stage",
        xticks=x,
        ylim=(0, 1),
    )
    return fig


def _draw_first_conflict_stage_context(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    contexts = [
        label
        for label in (*CONTEXT_PRIORITY, "none")
        if any(row["first_unsafe_context"] == label for row in rows)
    ]
    bottom = np.zeros(4)
    for context in contexts:
        values = [
            sum(
                row["count"]
                for row in rows
                if row["first_unsafe_stage"] == stage
                and row["first_unsafe_context"] == context
            )
            for stage in range(1, 5)
        ]
        ax.bar(range(1, 5), values, bottom=bottom, label=context)
        bottom += values
    if int(bottom.sum()) != sum(row["count"] for row in rows):
        raise PolicyAnalysisError("rendered first-conflict stacks do not reconcile")
    ax.set(
        xlabel="first unsafe stage",
        ylabel="U0/PU-discordant shots",
        title="First conflict context (exclusive display label)",
        xticks=range(1, 5),
    )
    if contexts:
        ax.legend(fontsize=8)
    return fig


def _draw_cost_excess_ecdf(payload: Mapping[str, Any], plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for stage, points in payload["by_stage"].items():
        if points:
            ax.step(
                [point["value"] for point in points],
                [point["cumulative_fraction"] for point in points],
                where="post",
                label=f"stage {stage}",
            )
    tolerance_points = payload["tolerance_tau_k"]["ecdf"]
    max_tolerance = max((point["value"] for point in tolerance_points), default=0.0)
    ax.axvspan(
        -max_tolerance,
        max_tolerance,
        color="0.8",
        alpha=0.5,
        label="recorded ±tau_k band",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1e-9)
    ax.set(
        xlabel="cost excess (symlog)",
        ylabel="empirical CDF",
        title="Cost-excess distribution by stage",
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    return fig


def _draw_first_safe_action_rank(payload: Mapping[str, Any], plt: Any) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    action_rows = [
        row
        for row in payload["actions"]
        if row["terminal_action"] != "censored-invalid"
    ]
    axes[0].bar(range(len(action_rows)), [row["count"] for row in action_rows])
    axes[0].set_xticks(
        range(len(action_rows)),
        [row["terminal_action"] for row in action_rows],
        rotation=25,
        ha="right",
    )
    axes[0].set(ylabel="unsafe states", title="Terminal action")
    rank_rows = [row for row in payload["ranks"] if row["first_safe_rank"] is not None]
    axes[1].bar(
        [row["first_safe_rank"] for row in rank_rows],
        [row["count"] for row in rank_rows],
    )
    axes[1].set(
        xlabel="operational first-safe rank",
        ylabel="unsafe states",
        title="First safe alternative",
    )
    return fig


def _draw_stage_transition_matrix(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["rows"]
    columns: list[Any] = [1, 2, 3, 4, "abstain"]
    matrix = np.zeros((4, len(columns)))
    for row in rows:
        matrix[int(row["original_stage"]) - 1, columns.index(row["terminal_stage"])] = (
            row["count"]
        )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(columns)), [str(value) for value in columns])
    ax.set_yticks(range(4), [str(stage) for stage in range(1, 5)])
    ax.set(
        xlabel="first-safe stage / terminal",
        ylabel="original unsafe stage",
        title="Original-state counterfactual transition",
    )
    fig.colorbar(image, ax=ax, label="states")
    return fig


def _draw_original_versus_alternative(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    metrics = (
        ("decision_weight", "decision weight"),
        ("path_length", "path length"),
        ("weight_margin", "local weight margin"),
        ("events_removed", "immediate events removed"),
    )
    for ax, (metric, label) in zip(axes.flat, metrics):
        for row in rows:
            left, right = row[f"original_{metric}"], row[f"first_safe_{metric}"]
            if left is not None and right is not None:
                ax.plot((0, 1), (left, right), color="0.6", alpha=0.2)
        ax.set_xticks((0, 1), ("original", "first safe"))
        ax.set_ylabel(label)
    fig.suptitle("Original unsafe candidate versus first safe alternative")
    return fig


def _draw_risk_heatmaps(payload: Mapping[str, Any], plt: Any) -> Any:
    risk = payload["tables"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, rows) in zip(
        axes.flat,
        [
            (name, risk[name])
            for name in (
                "stage_by_window_offset",
                "stage_by_static_boundary_competition",
                "domain_hw_by_candidate_multiplicity",
                "stage_by_margin_decile",
            )
        ],
    ):
        if not rows:
            ax.set_axis_off()
            continue
        keys = [
            key
            for key in rows[0]
            if key not in {"unsafe", "denominator", "unsafe_fraction", "stratum_status"}
        ]
        ys = sorted({row[keys[0]] for row in rows}, key=str)
        xs = sorted({row[keys[1]] for row in rows}, key=str)
        values = np.full((len(ys), len(xs)), np.nan)
        for row in rows:
            values[ys.index(row[keys[0]]), xs.index(row[keys[1]])] = row[
                "unsafe_fraction"
            ]
        ax.imshow(values, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(
            range(len(xs)), [str(value) for value in xs], rotation=45, ha="right"
        )
        ax.set_yticks(range(len(ys)), [str(value) for value in ys])
        ax.set(xlabel=keys[1], ylabel=keys[0], title=name.replace("_", " "))
    fig.suptitle("Descriptive unsafe-fraction heatmaps (not policy thresholds)")
    return fig


def _draw_disagreement_association(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(rows))
    denominator = np.array([row["shots"] for row in rows], dtype=float)
    for name in (
        "prediction_disagreement",
        "regression",
        "recovery",
        "prediction_discordant_both_wrong",
    ):
        values = np.divide(
            [row[name] for row in rows],
            denominator,
            out=np.full(len(rows), np.nan),
            where=denominator != 0,
        )
        ax.plot(x, values, marker="o", label=name)
    ax.set_xticks(x, [row["unsafe_count_bin"] for row in rows])
    ax.set(
        xlabel="unsafe durable commitments per shot",
        ylabel="unconditional shot fraction",
        title="Association with final U0/PU outcome",
    )
    ax.legend(fontsize=8)
    return fig


def _draw_event_relief(payload: Mapping[str, Any], plt: Any) -> Any:
    rows = payload["summary"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    y = np.array(
        [row["R_event"] if row["R_event"] is not None else np.nan for row in rows]
    )
    lower = np.array(
        [
            row["R_event_bootstrap"]["lower"]
            if row["R_event_bootstrap"]["lower"] is not None
            else np.nan
            for row in rows
        ]
    )
    upper = np.array(
        [
            row["R_event_bootstrap"]["upper"]
            if row["R_event_bootstrap"]["upper"] is not None
            else np.nan
            for row in rows
        ]
    )
    ax.errorbar(
        x,
        y,
        yerr=np.maximum(0, np.vstack((y - lower, upper - y))),
        fmt="o",
        capsize=4,
    )
    ax.set_xticks(x, [row["arm"] for row in rows], rotation=25, ha="right")
    ax.set(ylabel="R_event (ratio of sums)", title="Durable detector-event relief")
    return fig


def _draw_veto_chain_tails(payload: Mapping[str, Any], plt: Any) -> Any:
    metrics = payload["metrics"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, summary in metrics.items():
        if summary["ecdf"]:
            ax.step(
                [row["value"] for row in summary["ecdf"]],
                [row["cumulative_fraction"] for row in summary["ecdf"]],
                where="post",
                label=name,
            )
    ax.set(
        xlabel="count / elapsed ns",
        ylabel="empirical CDF",
        title="Counterfactual and Stage-3 tails",
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    return fig


# One spec per rendered figure: (payload name, payload extraction, draw
# function).  The payload name doubles as the PNG file stem, and the spec
# order preserves the pre-split back-to-back rendering order exactly.
_PLOT_SPECS: tuple[
    tuple[str, Callable[[Mapping[str, Any]], Mapping[str, Any]], Callable[..., Any]],
    ...,
] = (
    (
        "certificate-flow",
        lambda payloads: payloads["certificate-flow"],
        _draw_certificate_flow,
    ),
    (
        "unsafe-fraction-by-stage",
        lambda payloads: payloads["unsafe-fraction-by-stage"],
        _draw_unsafe_fraction_by_stage,
    ),
    (
        "first-conflict-stage-context",
        lambda payloads: payloads["first-conflict-stage-context"],
        _draw_first_conflict_stage_context,
    ),
    (
        "cost-excess-ecdf",
        lambda payloads: payloads["cost-excess-ecdf"],
        _draw_cost_excess_ecdf,
    ),
    (
        "first-safe-action-rank",
        lambda payloads: payloads["first-safe-action-rank"],
        _draw_first_safe_action_rank,
    ),
    (
        "stage-transition-matrix",
        lambda payloads: payloads["stage-transition-matrix"],
        _draw_stage_transition_matrix,
    ),
    (
        "original-versus-alternative",
        lambda payloads: payloads["original-versus-alternative"],
        _draw_original_versus_alternative,
    ),
    (
        "risk-heatmaps",
        lambda payloads: payloads["risk-heatmaps"],
        _draw_risk_heatmaps,
    ),
    (
        "disagreement-association",
        lambda payloads: payloads["disagreement-association"],
        _draw_disagreement_association,
    ),
    (
        "event-relief",
        lambda payloads: payloads["event-relief"],
        _draw_event_relief,
    ),
    (
        "veto-chain-tails",
        lambda payloads: payloads["veto-chain-tails"],
        _draw_veto_chain_tails,
    ),
)


def _render_plots(plot_dir: Path, payloads: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rendered: list[str] = []
    for name, extract, draw in _PLOT_SPECS:
        figure = draw(extract(payloads), plt)
        filename = f"{name}.png"
        _save_plot(figure, plot_dir / filename)
        rendered.append(filename)
    return rendered
