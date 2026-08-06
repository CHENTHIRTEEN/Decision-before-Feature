from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_evaluate import _json_default


EVAL_DOMAINS = ("all_validation", "changed_algorithm_validation", "same_algorithm_reference")
SELECTED_DECISION_POINTS = {
    ("lightgbm", "zero"),
    ("xgboost", "zero"),
    ("random_forest", "zero"),
    ("random_forest", "train_utility"),
    ("kernel_regression", "train_utility"),
}
POLICY_LABELS = {
    "no_ela_sbs": "No ELA / SBS",
    "always_ela_traditional_aas": "Always ELA",
    "random_analysis_p50": "Random p=0.5",
    "best_observed_analysis_action": "Best observed action",
    "decision_before_feature": "Decision-before-Feature",
}


def run_pareto_plot(
    *,
    ablation_policy_summary_path: Path,
    output_dir: Path,
    dataset_name: str,
) -> dict[str, Any]:
    frame = _read_ablation_summary(ablation_policy_summary_path)
    points = _pareto_points(frame, dataset_name)
    frontier = _frontier_rows(points)

    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "pareto_points.parquet"
    frontier_path = output_dir / "pareto_frontier.parquet"
    png_path = output_dir / "cost_performance_pareto.png"
    pdf_path = output_dir / "cost_performance_pareto.pdf"
    svg_path = output_dir / "cost_performance_pareto.svg"
    summary_path = output_dir / "cost_performance_pareto_summary.json"

    pq.write_table(pa.Table.from_pandas(points, preserve_index=False), points_path)
    pq.write_table(pa.Table.from_pandas(frontier, preserve_index=False), frontier_path)
    _draw_pareto(points, frontier, png_path, pdf_path, svg_path, dataset_name)

    summary = {
        "experiment": "min_support_cost_performance_pareto_plot",
        "dataset_name": dataset_name,
        "input": str(ablation_policy_summary_path),
        "cost_axis": {
            "primary": "ela_call_rate",
            "secondary_panel": "runtime_mean_seconds",
        },
        "performance_axis": "utility_mean",
        "outputs": {
            "points": str(points_path),
            "frontier": str(frontier_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
            "svg": str(svg_path),
            "summary": str(summary_path),
        },
        "frontier_by_eval_domain": frontier.to_dict(orient="records"),
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "original_utility_labels_modified": False,
            "decision_input_uses_ela_features": False,
            "formal_phase1_configs_modified": False,
        },
        "notes": [
            "Utility is plotted as relative utility against no_ela_sbs; larger values are better.",
            "Final performance is a minimization loss in the source data, so utility_mean is used as the vertical quality measure.",
            "Best observed action is shown as an unattainable reference and is excluded from the deployable Pareto frontier.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote Pareto points to {points_path}")
    print(f"wrote Pareto frontier to {frontier_path}")
    print(f"wrote Pareto plot to {png_path}")
    print(f"wrote Pareto summary to {summary_path}")
    return summary


def _read_ablation_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing ablation policy summary: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "policy_name",
        "policy_category",
        "model_family",
        "threshold_mode",
        "eval_domain",
        "layer",
        "ela_call_rate",
        "utility_mean",
        "utility_sum",
        "runtime_mean_seconds",
        "positive_row_capture_rate",
        "utility_capture_rate",
        "unhelpful_call_cost_sum",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ablation summary is missing required columns: {missing}")
    return frame


def _pareto_points(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    overall = frame[(frame["layer"] == "overall") & (frame["eval_domain"].isin(EVAL_DOMAINS))].copy()
    selected = overall[
        (overall["policy_category"] != "proposed")
        | overall.apply(lambda row: (str(row["model_family"]), str(row["threshold_mode"])) in SELECTED_DECISION_POINTS, axis=1)
    ].copy()
    selected["dataset_name"] = dataset_name
    selected["plot_label"] = selected.apply(_plot_label, axis=1)
    selected["deployable_policy"] = selected["policy_name"] != "best_observed_analysis_action"
    return selected[
        [
            "dataset_name",
            "eval_domain",
            "policy_name",
            "policy_category",
            "model_family",
            "threshold_mode",
            "plot_label",
            "deployable_policy",
            "ela_call_rate",
            "runtime_mean_seconds",
            "utility_mean",
            "utility_sum",
            "positive_row_capture_rate",
            "utility_capture_rate",
            "unhelpful_call_cost_sum",
        ]
    ].reset_index(drop=True)


def _plot_label(row: pd.Series) -> str:
    if row["policy_name"] != "decision_before_feature":
        return POLICY_LABELS.get(str(row["policy_name"]), str(row["policy_name"]))
    return f"DBF {row['model_family']} {row['threshold_mode']}"


def _frontier_rows(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eval_domain, group in points[points["deployable_policy"]].groupby("eval_domain", sort=True):
        frontier = _non_dominated(group, cost_column="ela_call_rate", quality_column="utility_mean")
        frontier = frontier.assign(frontier_cost_axis="ela_call_rate")
        rows.append(frontier)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _non_dominated(frame: pd.DataFrame, cost_column: str, quality_column: str) -> pd.DataFrame:
    ordered = frame.sort_values([cost_column, quality_column], ascending=[True, False]).copy()
    keep = []
    best_quality = -np.inf
    for index, row in ordered.iterrows():
        quality = float(row[quality_column])
        if quality > best_quality:
            keep.append(index)
            best_quality = quality
    return ordered.loc[keep].copy()


def _draw_pareto(
    points: pd.DataFrame,
    frontier: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
    dataset_name: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    colors = {
        "baseline": "#4C78A8",
        "proposed": "#F58518",
        "reference_upper_bound": "#54A24B",
    }
    markers = {
        "baseline": "o",
        "proposed": "s",
        "reference_upper_bound": "*",
    }
    titles = {
        "all_validation": "All validation",
        "changed_algorithm_validation": "Changed algorithm rows",
        "same_algorithm_reference": "Same algorithm reference",
    }
    for col, eval_domain in enumerate(EVAL_DOMAINS):
        domain_points = points[points["eval_domain"] == eval_domain]
        _draw_panel(
            ax=axes[0, col],
            domain_points=domain_points,
            frontier=frontier[frontier["eval_domain"] == eval_domain],
            x_column="ela_call_rate",
            y_column="utility_mean",
            x_label="ELA call rate",
            title=titles[eval_domain],
            colors=colors,
            markers=markers,
            draw_frontier=True,
        )
        _draw_panel(
            ax=axes[1, col],
            domain_points=domain_points,
            frontier=frontier[frontier["eval_domain"] == eval_domain],
            x_column="runtime_mean_seconds",
            y_column="utility_mean",
            x_label="Mean runtime (s)",
            title="",
            colors=colors,
            markers=markers,
            draw_frontier=False,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"Cost-performance Pareto diagnostic ({dataset_name})", y=1.02, fontsize=13)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _draw_panel(
    *,
    ax: plt.Axes,
    domain_points: pd.DataFrame,
    frontier: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    title: str,
    colors: dict[str, str],
    markers: dict[str, str],
    draw_frontier: bool,
) -> None:
    frontier_labels = set(frontier["plot_label"].astype(str)) if not frontier.empty else set()
    for _, row in domain_points.iterrows():
        category = str(row["policy_category"])
        ax.scatter(
            float(row[x_column]),
            float(row[y_column]),
            s=120 if category == "reference_upper_bound" else 58,
            marker=markers[category],
            color=colors[category],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9 if row["deployable_policy"] else 0.55,
            label=str(row["plot_label"]),
        )
        if category != "proposed" or str(row["plot_label"]) in frontier_labels:
            ax.annotate(
                str(row["plot_label"]),
                (float(row[x_column]), float(row[y_column])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )
    if draw_frontier and not frontier.empty:
        front = frontier.sort_values(x_column)
        ax.plot(front[x_column], front[y_column], color="#222222", linewidth=1.4, linestyle="--", label="Deployable frontier")
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean relative utility")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot min-support cost-performance Pareto curves from ablation output.")
    parser.add_argument(
        "--ablation-policy-summary",
        type=Path,
        default=Path("results/decision/min_support/ablation_comparison/ablation_policy_summary.parquet"),
    )
    parser.add_argument("--dataset-name", default="fe_transition_model_sensitivity")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/pareto_curve"),
    )
    args = parser.parse_args()
    run_pareto_plot(
        ablation_policy_summary_path=args.ablation_policy_summary,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
