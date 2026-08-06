from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_diagnostics import _corr, _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default, _read_labels


GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "fe_ratio": ["FE_ratio"],
    "problem_id": ["problem_id"],
}


def run_label_source_check(
    *,
    labels_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    labels = _read_labels(labels_path)
    _check_target(target_column)
    labels = labels.copy()
    labels["label_source"] = np.where(
        labels["selected_algorithm"].astype(str) == labels["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    labels["utility_gt_zero"] = labels[target_column] > 0.0
    labels["performance_gain_gt_zero"] = labels["performance_gain_norm"] > 0.0

    source_summary = _source_summary_table(labels, target_column)
    layered_summary = _layered_source_summary(labels, target_column)
    positive_rows = labels[labels["utility_gt_zero"]].copy()
    positive_summary = _positive_summary(labels, positive_rows, target_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary_path = output_dir / "label_source_summary.parquet"
    layered_summary_path = output_dir / "label_source_layered_summary.parquet"
    positive_rows_path = output_dir / "positive_utility_rows.parquet"
    summary_path = output_dir / "label_source_check_summary.json"
    pq.write_table(pa.Table.from_pandas(source_summary, preserve_index=False), source_summary_path)
    pq.write_table(pa.Table.from_pandas(layered_summary, preserve_index=False), layered_summary_path)
    pq.write_table(pa.Table.from_pandas(positive_rows, preserve_index=False), positive_rows_path)

    summary = {
        "experiment": "min_support_label_source_check",
        "labels": str(labels_path),
        "target_column": target_column,
        "rows": int(len(labels)),
        "positive_utility_rows": int(len(positive_rows)),
        "source_counts": labels["label_source"].value_counts().sort_index().to_dict(),
        "positive_utility_source_counts": positive_rows["label_source"].value_counts().sort_index().to_dict(),
        "positive_summary": positive_summary,
        "outputs": {
            "source_summary": str(source_summary_path),
            "layered_summary": str(layered_summary_path),
            "positive_utility_rows": str(positive_rows_path),
            "summary": str(summary_path),
        },
        "interpretation_note": (
            "Rows where selected_algorithm equals default_algorithm indicate same-algorithm paired continuations; "
            "positive utility there should not be attributed to selector-driven algorithm changes."
        ),
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote label source summary to {source_summary_path}")
    print(f"wrote layered label source summary to {layered_summary_path}")
    print(f"wrote positive utility rows to {positive_rows_path}")
    print(f"wrote label source check summary to {summary_path}")
    return summary


def _source_summary_table(labels: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = [_summary_rows(labels, ["label_source"], target_column)]
    frames.append(_summary_rows(labels, ["label_source", "default_algorithm", "selected_algorithm"], target_column))
    return pd.concat(frames, ignore_index=True)


def _layered_source_summary(labels: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = []
    for layer, columns in GROUP_LAYERS.items():
        frames.append(_summary_rows(labels, ["label_source", *columns], target_column, layer=layer))
    return pd.concat(frames, ignore_index=True)


def _summary_rows(
    labels: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
    layer: str = "source",
) -> pd.DataFrame:
    rows = []
    for group_values, subset in labels.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_summary_row(subset, group, target_column, layer))
    return pd.DataFrame(rows)


def _summary_row(subset: pd.DataFrame, group: dict[str, Any], target_column: str, layer: str) -> dict[str, Any]:
    gain = subset["performance_gain_norm"].to_numpy(dtype=float)
    cost = subset["time_cost_norm"].to_numpy(dtype=float)
    utility = subset[target_column].to_numpy(dtype=float)
    positive = utility > 0.0
    return {
        "layer": layer,
        "group": _group_label({key: value for key, value in group.items() if key != "label_source"}),
        "label_source": group.get("label_source"),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "default_algorithm": group.get("default_algorithm"),
        "selected_algorithm": group.get("selected_algorithm"),
        "rows": int(len(subset)),
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)),
        "performance_gain_gt_zero_rate": float(np.mean(gain > 0.0)),
        "mean_performance_gain_norm": float(np.mean(gain)),
        "median_performance_gain_norm": float(np.median(gain)),
        "mean_time_cost_norm": float(np.mean(cost)),
        "median_time_cost_norm": float(np.median(cost)),
        "mean_utility": float(np.mean(utility)),
        "median_utility": float(np.median(utility)),
        "sum_positive_utility": float(np.sum(utility[positive])),
        "mean_positive_utility": float(np.mean(utility[positive])) if np.any(positive) else 0.0,
        "corr_gain_utility": _corr(gain, utility),
        "corr_time_cost_utility": _corr(cost, utility),
    }


def _positive_summary(labels: pd.DataFrame, positive_rows: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if positive_rows.empty:
        return {
            "same_algorithm_positive_rate": 0.0,
            "changed_algorithm_positive_rate": 0.0,
            "same_algorithm_positive_utility_share": 0.0,
            "changed_algorithm_positive_utility_share": 0.0,
        }
    total_positive_utility = float(positive_rows[target_column].sum())
    source = positive_rows.groupby("label_source")[target_column].agg(["size", "sum", "mean"]).reset_index()
    rows = {str(row["label_source"]): row for _, row in source.iterrows()}
    same = rows.get("same_algorithm")
    changed = rows.get("changed_algorithm")
    return {
        "same_algorithm_positive_rows": int(same["size"]) if same is not None else 0,
        "changed_algorithm_positive_rows": int(changed["size"]) if changed is not None else 0,
        "same_algorithm_positive_row_share": float((same["size"] if same is not None else 0) / len(positive_rows)),
        "changed_algorithm_positive_row_share": float((changed["size"] if changed is not None else 0) / len(positive_rows)),
        "same_algorithm_positive_utility_sum": float(same["sum"]) if same is not None else 0.0,
        "changed_algorithm_positive_utility_sum": float(changed["sum"]) if changed is not None else 0.0,
        "same_algorithm_positive_utility_share": float((same["sum"] if same is not None else 0.0) / total_positive_utility),
        "changed_algorithm_positive_utility_share": float((changed["sum"] if changed is not None else 0.0) / total_positive_utility),
        "same_algorithm_positive_mean_utility": float(same["mean"]) if same is not None else 0.0,
        "changed_algorithm_positive_mean_utility": float(changed["mean"]) if changed is not None else 0.0,
        "same_algorithm_total_rows": int((labels["label_source"] == "same_algorithm").sum()),
        "changed_algorithm_total_rows": int((labels["label_source"] == "changed_algorithm").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check minimum-support label sources by selected/default algorithm match.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--output-dir", type=Path, default=Path("results/decision/min_support/label_source_check"))
    args = parser.parse_args()
    run_label_source_check(labels_path=args.labels, output_dir=args.output_dir, target_column=args.target_column)


if __name__ == "__main__":
    main()
