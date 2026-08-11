from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_query_lamT_1"
AUXILIARY_LABEL_COLUMN = "need_query_lamT_1"
METADATA_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    "query_id",
    "query_protocol",
    "sample_design_id",
    "default_algorithm",
    "selection_reference_default_algorithm",
    "selection_reference_protocol",
    "selector_prediction_source",
    "selected_algorithm",
    "selected_action",
    "selected_equals_default",
    "selected_equals_prefix",
    "best_observed_algorithm",
    "selected_matches_best_observed",
    "potential_gain_raw",
    "selector_regret_raw",
    "skip_switches_from_prefix",
    "no_query_transition_mode",
    "query_transition_mode",
    "label_source",
)
FORBIDDEN_X_COLUMNS = {
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    AUXILIARY_LABEL_COLUMN,
    "algorithm",
    "function_id",
    "FE_total",
    "FE_prefix",
    "FE_query",
    "FE_no_query_optimization",
    "FE_query_optimization",
    "p_skip",
    "p_query",
    "performance_gain_raw",
    "performance_gain_norm",
    "runtime_query",
    "runtime_selection",
    "runtime_no_query_optimization",
    "runtime_query_optimization",
    "time_cost_norm",
    "memory_cost_norm",
    "u_query_lamT_0",
    "u_query_lamT_025",
    "u_query_lamT_05",
    "u_query_lamT_2",
    "need_query_lamT_0",
    "need_query_lamT_025",
    "need_query_lamT_05",
    "need_query_lamT_2",
}
FORBIDDEN_X_NAME_FRAGMENTS = (
    "query",
    "function",
    "algorithm",
    "selected",
    "default",
    "family",
    "problem",
    "dimension",
)
EPS = 1e-12


def check_training_pipeline_contract(
    *,
    query_id: str,
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    spec = get_query_spec(query_id)
    if schema.get("query_id") != query_id or schema.get("query_protocol") != spec.protocol:
        raise ValueError("Decision schema does not match the requested query protocol")

    input_columns = _input_columns(schema)
    _check_columns(dataset, schema, input_columns)
    split_summary = _split_summary(dataset)
    train = dataset[dataset["split"] == TRAIN_SPLIT].copy()
    validation = dataset[dataset["split"] == VALIDATION_SPLIT].copy()
    _check_splits(train, validation)

    x_legality = _x_legality_summary(input_columns)
    if not bool(x_legality["passed"].all()):
        failed = x_legality[~x_legality["passed"]]
        raise ValueError(f"Decision X legality check failed: {failed.to_dict(orient='records')}")

    x_train = train[input_columns]
    x_validation = validation[input_columns]
    y_train = train[TARGET_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    _check_target(y_train, y_validation)

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    x_validation_imputed = imputer.transform(x_validation)
    imputer_summary = _imputer_summary(imputer, x_train, x_validation, input_columns)
    if not bool(imputer_summary["median_matches_train_split"].all()):
        raise ValueError("imputer statistics must match train split medians")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_validation_scaled = scaler.transform(x_validation_imputed)
    scaler_summary = _scaler_summary(scaler, x_train_imputed, input_columns)
    if int(_n_samples_seen(scaler)) != len(train):
        raise ValueError("scaler must be fit on train split rows only")

    transform_summary = _transform_summary(
        x_train_imputed=x_train_imputed,
        x_validation_imputed=x_validation_imputed,
        x_train_scaled=x_train_scaled,
        x_validation_scaled=x_validation_scaled,
        train_rows=len(train),
        validation_rows=len(validation),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(split_summary, output_dir / "split_summary")
    _write_frame(x_legality, output_dir / "x_legality_summary")
    _write_frame(imputer_summary, output_dir / "imputer_fit_summary")
    _write_frame(scaler_summary, output_dir / "scaler_fit_summary")
    _write_frame(transform_summary, output_dir / "preprocessing_transform_summary")

    summary = {
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "rows": int(len(dataset)),
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "input_columns": input_columns,
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "preprocessing": {
            "imputer": "SimpleImputer(strategy='median')",
            "imputer_fit_split": TRAIN_SPLIT,
            "imputer_fit_rows": int(len(train)),
            "scaler": "StandardScaler()",
            "scaler_fit_split": TRAIN_SPLIT,
            "scaler_fit_rows": int(_n_samples_seen(scaler)),
            "validation_rows_used_for_fit": 0,
        },
        "model_training_executed": False,
        "all_checks_passed": True,
    }
    summary_path = output_dir / "training_pipeline_contract_summary.json"
    report_path = output_dir / "training_pipeline_contract_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            split_summary=split_summary,
            x_legality=x_legality,
            imputer_summary=imputer_summary,
            scaler_summary=scaler_summary,
            transform_summary=transform_summary,
        ),
        encoding="utf-8",
    )

    print(f"wrote training pipeline contract summary to {summary_path}")
    print(f"wrote training pipeline contract report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "training_pipeline_contract_summary.json",
        output_dir / "training_pipeline_contract_report.md",
        output_dir / "split_summary.csv",
        output_dir / "split_summary.parquet",
        output_dir / "x_legality_summary.csv",
        output_dir / "x_legality_summary.parquet",
        output_dir / "imputer_fit_summary.csv",
        output_dir / "imputer_fit_summary.parquet",
        output_dir / "scaler_fit_summary.csv",
        output_dir / "scaler_fit_summary.parquet",
        output_dir / "preprocessing_transform_summary.csv",
        output_dir / "preprocessing_transform_summary.parquet",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"training pipeline contract outputs already exist; pass --overwrite: {existing[0]}")


def _input_columns(schema: dict[str, Any]) -> list[str]:
    columns = list(schema.get("input_columns", []))
    if columns != list(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("schema input_columns must exactly equal BEHAVIOR_FEATURE_COLUMNS")
    return columns


def _check_columns(dataset: pd.DataFrame, schema: dict[str, Any], input_columns: list[str]) -> None:
    required = set(METADATA_COLUMNS) | {TARGET_COLUMN, AUXILIARY_LABEL_COLUMN, *input_columns}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"materialized dataset missing required columns: {missing}")
    schema_target = schema.get("target_column")
    if schema_target != TARGET_COLUMN:
        raise ValueError(f"schema target_column must be {TARGET_COLUMN}, got {schema_target}")
    schema_metadata = list(schema.get("metadata_columns", []))
    if schema_metadata != list(METADATA_COLUMNS):
        raise ValueError("schema metadata_columns must match the frozen metadata columns")


def _split_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in dataset.groupby("split", dropna=False):
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "target_null_count": int(group[TARGET_COLUMN].isna().sum()),
                "target_finite": bool(np.isfinite(group[TARGET_COLUMN].to_numpy(dtype=float)).all()),
                "u_gt_zero_rate": float((group[TARGET_COLUMN].to_numpy(dtype=float) > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _check_splits(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if train.empty:
        raise ValueError(f"missing train split rows: {TRAIN_SPLIT}")
    if validation.empty:
        raise ValueError(f"missing validation split rows: {VALIDATION_SPLIT}")
    for name, frame in ((TRAIN_SPLIT, train), (VALIDATION_SPLIT, validation)):
        if not (frame["prefix_algorithm"].astype(str) == frame["default_algorithm"].astype(str)).all():
            raise ValueError(f"{name} must use the train-derived SBS as both prefix and default")
        if frame["skip_switches_from_prefix"].astype(bool).any():
            raise ValueError(f"{name} must not switch algorithms on the no-query path")
        invalid_no_action = frame["selected_equals_prefix"].astype(bool) & (frame[TARGET_COLUMN].astype(float) > 0.0)
        if invalid_no_action.any():
            raise ValueError(f"{name} contains positive utility without an optimizer action change")


def _x_legality_summary(input_columns: list[str]) -> pd.DataFrame:
    exact_forbidden = sorted(set(input_columns).intersection(FORBIDDEN_X_COLUMNS))
    name_forbidden = [
        column
        for column in input_columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_X_NAME_FRAGMENTS)
    ]
    return pd.DataFrame(
        [
            {
                "check": "x_columns_equal_behavior_feature_columns",
                "passed": input_columns == list(BEHAVIOR_FEATURE_COLUMNS),
                "detail": ",".join(input_columns),
            },
            {
                "check": "metadata_columns_absent_from_x",
                "passed": len(set(input_columns).intersection(METADATA_COLUMNS)) == 0,
                "detail": ",".join(sorted(set(input_columns).intersection(METADATA_COLUMNS))),
            },
            {
                "check": "target_columns_absent_from_x",
                "passed": TARGET_COLUMN not in input_columns and AUXILIARY_LABEL_COLUMN not in input_columns,
                "detail": ",".join([column for column in (TARGET_COLUMN, AUXILIARY_LABEL_COLUMN) if column in input_columns]),
            },
            {
                "check": "forbidden_exact_columns_absent_from_x",
                "passed": len(exact_forbidden) == 0,
                "detail": ",".join(exact_forbidden),
            },
            {
                "check": "forbidden_name_fragments_absent_from_x",
                "passed": len(name_forbidden) == 0,
                "detail": ",".join(name_forbidden),
            },
        ]
    )


def _check_target(y_train: pd.Series, y_validation: pd.Series) -> None:
    for name, values in ((TRAIN_SPLIT, y_train), (VALIDATION_SPLIT, y_validation)):
        if values.isna().any():
            raise ValueError(f"{TARGET_COLUMN} contains null values in {name}")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{TARGET_COLUMN} contains non-finite values in {name}")


def _imputer_summary(
    imputer: SimpleImputer,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    input_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for index, column in enumerate(input_columns):
        train_values = pd.to_numeric(x_train[column], errors="coerce")
        validation_values = pd.to_numeric(x_validation[column], errors="coerce")
        train_median = float(train_values.median())
        statistic = float(imputer.statistics_[index])
        rows.append(
            {
                "feature": column,
                "fit_split": TRAIN_SPLIT,
                "fit_rows": int(len(x_train)),
                "validation_rows_used_for_fit": 0,
                "strategy": str(imputer.strategy),
                "imputer_statistic": statistic,
                "train_raw_median": train_median,
                "validation_raw_median": _float_or_none(validation_values.median()),
                "median_matches_train_split": bool(abs(statistic - train_median) <= EPS * max(1.0, abs(train_median))),
                "train_null_count": int(x_train[column].isna().sum()),
                "validation_null_count": int(x_validation[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _scaler_summary(scaler: StandardScaler, x_train_imputed: np.ndarray, input_columns: list[str]) -> pd.DataFrame:
    train_mean = x_train_imputed.mean(axis=0)
    train_var = x_train_imputed.var(axis=0)
    rows = []
    for index, column in enumerate(input_columns):
        scaler_mean = float(scaler.mean_[index])
        scaler_var = float(scaler.var_[index])
        rows.append(
            {
                "feature": column,
                "fit_split": TRAIN_SPLIT,
                "fit_rows": int(_n_samples_seen(scaler)),
                "validation_rows_used_for_fit": 0,
                "scaler_mean": scaler_mean,
                "train_imputed_mean": float(train_mean[index]),
                "mean_matches_train_split": bool(abs(scaler_mean - train_mean[index]) <= EPS * max(1.0, abs(float(train_mean[index])))),
                "scaler_var": scaler_var,
                "train_imputed_var": float(train_var[index]),
                "var_matches_train_split": bool(abs(scaler_var - train_var[index]) <= EPS * max(1.0, abs(float(train_var[index])))),
                "scaler_scale": float(scaler.scale_[index]),
            }
        )
    return pd.DataFrame(rows)


def _transform_summary(
    *,
    x_train_imputed: np.ndarray,
    x_validation_imputed: np.ndarray,
    x_train_scaled: np.ndarray,
    x_validation_scaled: np.ndarray,
    train_rows: int,
    validation_rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "imputer_transform_train",
                "rows": int(train_rows),
                "finite_values": bool(np.isfinite(x_train_imputed).all()),
                "remaining_nan_count": int(np.isnan(x_train_imputed).sum()),
            },
            {
                "stage": "imputer_transform_validation",
                "rows": int(validation_rows),
                "finite_values": bool(np.isfinite(x_validation_imputed).all()),
                "remaining_nan_count": int(np.isnan(x_validation_imputed).sum()),
            },
            {
                "stage": "scaler_transform_train",
                "rows": int(train_rows),
                "finite_values": bool(np.isfinite(x_train_scaled).all()),
                "remaining_nan_count": int(np.isnan(x_train_scaled).sum()),
            },
            {
                "stage": "scaler_transform_validation",
                "rows": int(validation_rows),
                "finite_values": bool(np.isfinite(x_validation_scaled).all()),
                "remaining_nan_count": int(np.isnan(x_validation_scaled).sum()),
            },
        ]
    )


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _markdown_report(
    *,
    summary: dict[str, Any],
    split_summary: pd.DataFrame,
    x_legality: pd.DataFrame,
    imputer_summary: pd.DataFrame,
    scaler_summary: pd.DataFrame,
    transform_summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Decision Model training pipeline contract check",
            "",
            "## Scope",
            "",
            "- This check loads the materialized Decision dataset and validates preprocessing contract only.",
            "- No Decision Model estimator was trained.",
            f"- X columns: `{', '.join(summary['input_columns'])}`.",
            f"- Target column: `{summary['target_column']}`.",
            "",
            "## Split summary",
            "",
            _markdown_table(split_summary),
            "",
            "## X legality",
            "",
            _markdown_table(x_legality),
            "",
            "## Imputer fit contract",
            "",
            f"- Imputer: `{summary['preprocessing']['imputer']}`.",
            f"- Fit split: `{summary['preprocessing']['imputer_fit_split']}`.",
            f"- Fit rows: `{summary['preprocessing']['imputer_fit_rows']}`.",
            "- Validation rows used for fit: `0`.",
            "",
            _markdown_table(
                imputer_summary[
                    [
                        "feature",
                        "fit_split",
                        "fit_rows",
                        "validation_rows_used_for_fit",
                        "strategy",
                        "median_matches_train_split",
                        "train_null_count",
                        "validation_null_count",
                    ]
                ]
            ),
            "",
            "## Scaler fit contract",
            "",
            f"- Scaler: `{summary['preprocessing']['scaler']}`.",
            f"- Fit split: `{summary['preprocessing']['scaler_fit_split']}`.",
            f"- Fit rows: `{summary['preprocessing']['scaler_fit_rows']}`.",
            "- Validation rows used for fit: `0`.",
            "",
            _markdown_table(
                scaler_summary[
                    [
                        "feature",
                        "fit_split",
                        "fit_rows",
                        "validation_rows_used_for_fit",
                        "mean_matches_train_split",
                        "var_matches_train_split",
                        "scaler_scale",
                    ]
                ]
            ),
            "",
            "## Transform checks",
            "",
            _markdown_table(transform_summary),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    headers = list(frame.columns)
    rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _n_samples_seen(scaler: StandardScaler) -> int:
    seen = scaler.n_samples_seen_
    if np.isscalar(seen):
        return int(seen)
    return int(np.asarray(seen).max())


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Decision Model training pipeline preprocessing contract without model training.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    materialized = Path("results/decision") / args.query_id / "materialized_training_data"
    check_training_pipeline_contract(
        query_id=args.query_id,
        dataset_path=args.dataset or materialized / "decision_dataset.parquet",
        schema_path=args.schema or materialized / "decision_dataset_schema.json",
        output_dir=args.output_dir or Path("results/decision") / args.query_id / "training_pipeline_contract_check",
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
