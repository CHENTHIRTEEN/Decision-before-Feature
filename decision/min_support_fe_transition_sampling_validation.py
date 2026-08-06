from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _json_default, _read_labels
from utility_labels.fields import UTILITY_VALUE_COLUMNS


GROUP_LAYERS: dict[str, list[str]] = {
    "FE_ratio": ["FE_ratio"],
    "label_source_FE_ratio": ["label_source", "FE_ratio"],
    "family_FE_ratio": ["label_source", "family", "FE_ratio"],
    "dimension_FE_ratio": ["label_source", "dimension", "FE_ratio"],
    "problem_id_FE_ratio": ["label_source", "problem_id", "FE_ratio"],
    "family_dimension_FE_ratio": ["label_source", "family", "dimension", "FE_ratio"],
}


def run_fe_transition_sampling_validation(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
    original_train_labels_path: Path | None,
    original_validation_labels_path: Path | None,
) -> dict[str, Any]:
    _check_target_column(target_column)
    labels = pd.concat(
        [
            _with_dataset(_read_labels(train_labels_path), "train_fe_transition"),
            _with_dataset(_read_labels(validation_labels_path), "validation_fe_transition"),
        ],
        ignore_index=True,
    )
    labels = _with_label_source(labels)

    layered_summary = pd.concat(
        [
            _summarize_layer(labels, dataset="new_fe_transition", layer=layer, group_columns=columns, target_column=target_column)
            for layer, columns in GROUP_LAYERS.items()
        ],
        ignore_index=True,
    )
    dataset_summary = _summarize_layer(
        labels,
        dataset="new_fe_transition",
        layer="dataset_label_source_FE_ratio",
        group_columns=["dataset", "label_source", "FE_ratio"],
        target_column=target_column,
    )
    rows_of_interest = labels[labels[target_column] > 0.0].copy()

    comparison_summary = _original_vs_new_summary(
        labels=labels,
        target_column=target_column,
        original_train_labels_path=original_train_labels_path,
        original_validation_labels_path=original_validation_labels_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    layered_path = output_dir / "fe_transition_layered_label_summary.parquet"
    dataset_path = output_dir / "fe_transition_dataset_label_summary.parquet"
    rows_path = output_dir / "fe_transition_u_gt_zero_rows.parquet"
    comparison_path = output_dir / "fe_transition_original_vs_new_summary.parquet"
    summary_path = output_dir / "fe_transition_sampling_summary.json"
    pq.write_table(pa.Table.from_pandas(layered_summary, preserve_index=False), layered_path)
    pq.write_table(pa.Table.from_pandas(dataset_summary, preserve_index=False), dataset_path)
    pq.write_table(pa.Table.from_pandas(rows_of_interest, preserve_index=False), rows_path)
    pq.write_table(pa.Table.from_pandas(comparison_summary, preserve_index=False), comparison_path)

    summary = {
        "experiment": "min_support_fe_transition_sampling_validation",
        "research_question": (
            "Does denser min-support checkpoint sampling between FE_ratio 0.30 and 0.60 reveal "
            "where U_ELA > 0 concentrates under the unchanged shared-prefix utility-label protocol?"
        ),
        "train_labels": str(train_labels_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "new_checkpoint_ratios_for_labels": sorted(float(value) for value in labels["FE_ratio"].unique()),
        "rows": {
            "train_fe_transition": int((labels["dataset"] == "train_fe_transition").sum()),
            "validation_fe_transition": int((labels["dataset"] == "validation_fe_transition").sum()),
        },
        "u_gt_zero": _compact_rate_summary(labels, target_column),
        "changed_algorithm_u_gt_zero_by_FE_ratio": _rate_records(
            labels[labels["label_source"] == "changed_algorithm"],
            group_columns=["dataset", "FE_ratio"],
            target_column=target_column,
        ),
        "same_algorithm_reference_u_gt_zero_by_FE_ratio": _rate_records(
            labels[labels["label_source"] == "same_algorithm"],
            group_columns=["dataset", "FE_ratio"],
            target_column=target_column,
        ),
        "excluded_from_decision_input": [
            "ELA feature columns",
            "function_id",
            "family",
            "problem_id",
            "dimension",
            "prefix_algorithm",
            "selected_algorithm",
            "default_algorithm",
        ],
        "outputs": {
            "layered_label_summary": str(layered_path),
            "dataset_label_summary": str(dataset_path),
            "u_gt_zero_rows": str(rows_path),
            "original_vs_new_summary": str(comparison_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "decision_input_uses_ela_features": False,
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote FE-transition layered label summary to {layered_path}")
    print(f"wrote FE-transition dataset label summary to {dataset_path}")
    print(f"wrote FE-transition U_ELA > 0 rows to {rows_path}")
    print(f"wrote FE-transition original-vs-new summary to {comparison_path}")
    print(f"wrote FE-transition sampling summary to {summary_path}")
    return summary


def _check_target_column(target_column: str) -> None:
    if target_column not in UTILITY_VALUE_COLUMNS:
        raise ValueError(f"target column must be one of {list(UTILITY_VALUE_COLUMNS)}")


def _with_dataset(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "dataset", dataset)
    return result


def _with_label_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label_source"] = np.where(
        result["selected_algorithm"].astype(str) == result["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return result


def _summarize_layer(
    frame: pd.DataFrame,
    *,
    dataset: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group[target_column].to_numpy(dtype=float)
        gt_zero = values > 0.0
        row = {
            "summary_dataset": dataset,
            "layer": layer,
            "rows": int(len(group)),
            "u_gt_zero_count": int(gt_zero.sum()),
            "u_gt_zero_rate": float(gt_zero.mean()) if len(group) else 0.0,
            "u_mean": float(np.mean(values)) if len(group) else 0.0,
            "u_median": float(np.median(values)) if len(group) else 0.0,
            "u_sum": float(np.sum(values)) if len(group) else 0.0,
            "u_gt_zero_sum": float(np.sum(values[gt_zero])) if gt_zero.any() else 0.0,
            "performance_gain_norm_mean": float(group["performance_gain_norm"].mean()),
            "time_cost_norm_mean": float(group["time_cost_norm"].mean()),
            "changed_algorithm_rate": float((group["label_source"] == "changed_algorithm").mean()),
        }
        row.update({column: value for column, value in zip(group_columns, group_key, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def _original_vs_new_summary(
    *,
    labels: pd.DataFrame,
    target_column: str,
    original_train_labels_path: Path | None,
    original_validation_labels_path: Path | None,
) -> pd.DataFrame:
    frames = [
        labels.assign(sampling_design="new_fe_transition"),
    ]
    if original_train_labels_path is not None and original_train_labels_path.exists():
        frames.append(_with_label_source(_with_dataset(_read_labels(original_train_labels_path), "train_original")).assign(sampling_design="original"))
    if original_validation_labels_path is not None and original_validation_labels_path.exists():
        frames.append(
            _with_label_source(_with_dataset(_read_labels(original_validation_labels_path), "validation_original")).assign(
                sampling_design="original"
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    return _summarize_layer(
        combined,
        dataset="original_vs_new",
        layer="sampling_design_dataset_label_source_FE_ratio",
        group_columns=["sampling_design", "dataset", "label_source", "FE_ratio"],
        target_column=target_column,
    )


def _compact_rate_summary(frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    result = {
        "all_rows": _single_rate(frame, target_column),
    }
    for dataset, group in frame.groupby("dataset", sort=True):
        result[str(dataset)] = _single_rate(group, target_column)
    return result


def _single_rate(frame: pd.DataFrame, target_column: str) -> dict[str, float | int]:
    values = frame[target_column].to_numpy(dtype=float)
    gt_zero = values > 0.0
    return {
        "rows": int(len(frame)),
        "u_gt_zero_count": int(gt_zero.sum()),
        "u_gt_zero_rate": float(gt_zero.mean()) if len(frame) else 0.0,
        "u_mean": float(values.mean()) if len(frame) else 0.0,
    }


def _rate_records(frame: pd.DataFrame, *, group_columns: list[str], target_column: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    summary = _summarize_layer(
        frame,
        dataset="compact",
        layer="_".join(group_columns),
        group_columns=group_columns,
        target_column=target_column,
    )
    return summary.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate min_support FE-transition checkpoint sampling with utility labels.")
    parser.add_argument(
        "--train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train_fe_transition/utility_labels.parquet"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation_fe_transition/utility_labels.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/fe_transition_sampling"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--original-train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train/utility_labels.parquet"),
    )
    parser.add_argument(
        "--original-validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    args = parser.parse_args()
    run_fe_transition_sampling_validation(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
        original_train_labels_path=args.original_train_labels,
        original_validation_labels_path=args.original_validation_labels,
    )


if __name__ == "__main__":
    main()
