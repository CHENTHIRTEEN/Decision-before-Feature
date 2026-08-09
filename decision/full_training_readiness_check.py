from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_FEATURE_COLUMNS


DEFAULT_DATASET_PATH = Path("results/decision/phase1_refined_sampling/materialized_training_data/decision_dataset.parquet")
DEFAULT_SCHEMA_PATH = Path("results/decision/phase1_refined_sampling/materialized_training_data/decision_dataset_schema.json")
DEFAULT_MATERIALIZED_DIR = Path("results/decision/phase1_refined_sampling/materialized_training_data")
DEFAULT_TRAINING_CONTRACT_DIR = Path("results/decision/phase1_refined_sampling/training_pipeline_contract_check")
DEFAULT_CHANGED_SAME_DIR = Path("results/decision/phase1_refined_sampling/changed_same_learnability_diagnostic")
DEFAULT_STACKING_DIR = Path("results/decision/phase1_refined_sampling/algorithm_partition_stacking_diagnostic")
DEFAULT_UTILITY_ROOT = Path("results/utility_labels/phase1_refined_sampling")
DEFAULT_BEHAVIOR_ROOT = Path("results/phase1_refined_sampling")
DEFAULT_OUTPUT_DIR = Path("results/decision/phase1_refined_sampling/full_training_readiness")

TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_ela_lamT_1"
AUXILIARY_LABEL_COLUMN = "need_ela_lamT_1"
EXPECTED_TOTAL_ROWS = 259_200
EXPECTED_TRAIN_ROWS = 194_400
EXPECTED_VALIDATION_ROWS = 64_800
EXPECTED_UTILITY_SHARDS = 72
EXPECTED_BEHAVIOR_SHARDS = 72
EXPECTED_ALGORITHMS = ("cmaes", "de", "pso", "shade")
EXPECTED_DIMENSIONS = (10, 20, 40)
METADATA_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    "default_algorithm",
    "selected_algorithm",
    "label_source",
)
FORBIDDEN_X_COLUMNS = {
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    AUXILIARY_LABEL_COLUMN,
    "algorithm",
    "function_id",
    "function",
    "FE_total",
    "FE_prefix",
    "FE_analysis",
    "FE_skip_optimization",
    "FE_ela_optimization",
    "p_skip",
    "p_ela",
    "performance_gain_raw",
    "performance_gain_norm",
    "runtime_analysis",
    "runtime_selection",
    "runtime_skip_optimization",
    "runtime_ela_optimization",
    "time_cost_norm",
    "memory_cost_norm",
    "u_ela_lamT_0",
    "u_ela_lamT_025",
    "u_ela_lamT_05",
    "u_ela_lamT_2",
    "need_ela_lamT_0",
    "need_ela_lamT_025",
    "need_ela_lamT_05",
    "need_ela_lamT_2",
}
FORBIDDEN_X_NAME_FRAGMENTS = (
    "ela",
    "function",
    "algorithm",
    "selected",
    "default",
    "family",
    "problem",
    "dimension",
)
EPS = 1e-12


def run_full_training_readiness_check(
    *,
    dataset_path: Path,
    schema_path: Path,
    materialized_dir: Path,
    training_contract_dir: Path,
    changed_same_dir: Path,
    stacking_dir: Path,
    utility_root: Path,
    behavior_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)

    check_rows: list[dict[str, Any]] = []
    dataset = _read_dataset(dataset_path, check_rows)
    schema = _read_json(schema_path, "schema_json_readable", check_rows)
    input_columns = _schema_input_columns(schema, check_rows)

    _check_dataset_contract(dataset, schema, input_columns, check_rows)
    split_summary = _split_summary(dataset)
    feature_summary = _feature_summary(dataset, input_columns)
    label_source_summary = _label_source_summary(dataset)

    _check_shards(utility_root, behavior_root, check_rows)
    _check_materialization_outputs(materialized_dir, check_rows)
    _check_training_contract(training_contract_dir, dataset, input_columns, check_rows)
    _check_changed_same_outputs(changed_same_dir, check_rows)
    _check_stacking_outputs(stacking_dir, check_rows)

    readiness_checks = pd.DataFrame(check_rows)
    all_blocking_passed = bool(readiness_checks.loc[readiness_checks["severity"] == "blocking", "passed"].all())
    warnings_passed = bool(readiness_checks.loc[readiness_checks["severity"] == "warning", "passed"].all())
    can_start_full_training = all_blocking_passed

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(readiness_checks, output_dir / "readiness_checks")
    _write_frame(split_summary, output_dir / "split_readiness_summary")
    _write_frame(feature_summary, output_dir / "feature_readiness_summary")
    _write_frame(label_source_summary, output_dir / "label_source_readiness_summary")

    summary = {
        "readiness_scope": "phase1_refined_sampling_full_decision_training_preflight",
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "rows": int(len(dataset)),
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "train_rows": int((dataset["split"].astype(str) == TRAIN_SPLIT).sum()),
        "validation_rows": int((dataset["split"].astype(str) == VALIDATION_SPLIT).sum()),
        "input_columns": input_columns,
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "approved_training_data": str(dataset_path),
        "approved_main_training_protocol": {
            "fit_rows": "all bbob_train rows",
            "validation_rows": "all bbob_validation rows",
            "x_columns": input_columns,
            "y_column": TARGET_COLUMN,
            "imputer": "SimpleImputer(strategy='median'), fit on bbob_train only",
            "scaler": "StandardScaler(), fit on train-imputed matrix only",
            "metadata_usage": "reporting, split, and error analysis only; never model input",
            "thresholding": "predicted_u_ela > 0 plus train-derived thresholds/ranking; validation never used for threshold fitting",
        },
        "diagnostic_boundaries": {
            "changed_same_learnability": "baseline diagnostic only; may guide reporting strata",
            "algorithm_partition_stacking": (
                "diagnostic ablation only; prefix_algorithm is not approved as a main Decision Model input"
            ),
            "old_min_support_entries": "not approved as formal full-training entrypoints for the materialized phase1 dataset",
        },
        "blocking_checks_passed": all_blocking_passed,
        "warning_checks_passed": warnings_passed,
        "can_start_full_training": can_start_full_training,
        "model_training_executed": False,
    }
    summary_path = output_dir / "full_training_readiness_summary.json"
    report_path = output_dir / "full_training_readiness_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            readiness_checks=readiness_checks,
            split_summary=split_summary,
            feature_summary=feature_summary,
            label_source_summary=label_source_summary,
        ),
        encoding="utf-8",
    )

    print(f"wrote full training readiness summary to {summary_path}")
    print(f"wrote full training readiness report to {report_path}")
    if not can_start_full_training:
        failed = readiness_checks[(readiness_checks["severity"] == "blocking") & (~readiness_checks["passed"])]
        raise ValueError(f"full training readiness check failed: {failed.to_dict(orient='records')}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "full_training_readiness_summary.json",
        output_dir / "full_training_readiness_report.md",
        output_dir / "readiness_checks.csv",
        output_dir / "readiness_checks.parquet",
        output_dir / "split_readiness_summary.csv",
        output_dir / "split_readiness_summary.parquet",
        output_dir / "feature_readiness_summary.csv",
        output_dir / "feature_readiness_summary.parquet",
        output_dir / "label_source_readiness_summary.csv",
        output_dir / "label_source_readiness_summary.parquet",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"readiness outputs already exist; pass --overwrite: {existing[0]}")


def _read_dataset(dataset_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    exists = dataset_path.exists()
    _add_check(rows, "dataset_file_exists", exists, "blocking", str(dataset_path))
    if not exists:
        raise FileNotFoundError(dataset_path)
    dataset = pq.read_table(dataset_path).to_pandas()
    _add_check(rows, "dataset_parquet_readable", True, "blocking", f"rows={len(dataset)}")
    return dataset


def _read_json(path: Path, check_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    exists = path.exists()
    _add_check(rows, f"{check_name}_file_exists", exists, "blocking", str(path))
    if not exists:
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _add_check(rows, check_name, True, "blocking", str(path))
    return payload


def _schema_input_columns(schema: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    input_columns = list(schema.get("input_columns", []))
    expected = list(BEHAVIOR_FEATURE_COLUMNS)
    _add_check(
        rows,
        "schema_input_columns_equal_behavior_features",
        input_columns == expected,
        "blocking",
        ",".join(input_columns),
    )
    return input_columns


def _check_dataset_contract(
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    input_columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    expected_columns = list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN] + list(BEHAVIOR_FEATURE_COLUMNS)
    _add_check(
        rows,
        "dataset_columns_match_schema_order",
        list(dataset.columns) == expected_columns and list(schema.get("column_order", [])) == expected_columns,
        "blocking",
        ",".join(dataset.columns),
    )
    _add_check(rows, "dataset_total_rows_expected", len(dataset) == EXPECTED_TOTAL_ROWS, "blocking", f"rows={len(dataset)}")

    split_counts = dataset["split"].astype(str).value_counts().to_dict() if "split" in dataset.columns else {}
    _add_check(
        rows,
        "train_validation_row_counts_expected",
        split_counts.get(TRAIN_SPLIT, 0) == EXPECTED_TRAIN_ROWS
        and split_counts.get(VALIDATION_SPLIT, 0) == EXPECTED_VALIDATION_ROWS
        and set(split_counts) == {TRAIN_SPLIT, VALIDATION_SPLIT},
        "blocking",
        json.dumps(split_counts, sort_keys=True),
    )

    exact_forbidden = sorted(set(input_columns).intersection(FORBIDDEN_X_COLUMNS))
    name_forbidden = [
        column
        for column in input_columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_X_NAME_FRAGMENTS)
    ]
    _add_check(rows, "x_has_no_forbidden_exact_columns", not exact_forbidden, "blocking", ",".join(exact_forbidden))
    _add_check(rows, "x_has_no_forbidden_name_fragments", not name_forbidden, "blocking", ",".join(name_forbidden))

    target = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    _add_check(rows, "target_non_null", not dataset[TARGET_COLUMN].isna().any(), "blocking", TARGET_COLUMN)
    _add_check(rows, "target_finite", bool(np.isfinite(target.to_numpy(dtype=float)).all()), "blocking", TARGET_COLUMN)
    auxiliary_expected = target.to_numpy(dtype=float) > 0.0
    auxiliary_observed = dataset[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool)
    _add_check(
        rows,
        "auxiliary_label_matches_target_gt_zero",
        bool(np.array_equal(auxiliary_expected, auxiliary_observed)),
        "blocking",
        f"{AUXILIARY_LABEL_COLUMN} == {TARGET_COLUMN} > 0",
    )

    label_expected = np.where(
        dataset["selected_algorithm"].astype(str) == dataset["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    _add_check(
        rows,
        "label_source_matches_algorithm_equality",
        bool(np.array_equal(dataset["label_source"].to_numpy(dtype=str), label_expected)),
        "blocking",
        "same_algorithm iff selected_algorithm == default_algorithm",
    )

    algorithms = tuple(sorted(dataset["prefix_algorithm"].astype(str).unique().tolist()))
    dimensions = tuple(sorted(int(value) for value in dataset["dimension"].unique().tolist()))
    _add_check(rows, "expected_prefix_algorithm_domains", algorithms == EXPECTED_ALGORITHMS, "blocking", ",".join(algorithms))
    _add_check(
        rows,
        "expected_dimension_domains",
        dimensions == EXPECTED_DIMENSIONS,
        "blocking",
        ",".join(str(value) for value in dimensions),
    )

    non_finite_features = []
    for column in input_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        invalid = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            non_finite_features.append(f"{column}:{int(invalid.sum())}")
    _add_check(rows, "non_null_behavior_features_are_finite", not non_finite_features, "blocking", ",".join(non_finite_features))


def _check_shards(
    utility_root: Path,
    behavior_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    utility_paths = sorted(utility_root.glob("*/*/dimension_*/utility_labels.parquet"))
    behavior_paths = sorted(behavior_root.glob("*/*/dimension_*/behavior.parquet"))
    _add_check(
        rows,
        "utility_shard_count_expected",
        len(utility_paths) == EXPECTED_UTILITY_SHARDS,
        "blocking",
        f"found={len(utility_paths)} expected={EXPECTED_UTILITY_SHARDS}",
    )
    _add_check(
        rows,
        "behavior_shard_count_expected",
        len(behavior_paths) == EXPECTED_BEHAVIOR_SHARDS,
        "blocking",
        f"found={len(behavior_paths)} expected={EXPECTED_BEHAVIOR_SHARDS}",
    )


def _check_materialization_outputs(materialized_dir: Path, rows: list[dict[str, Any]]) -> None:
    required_files = (
        "decision_dataset_materialization_report.md",
        "join_coverage_summary.csv",
        "input_legality_summary.csv",
        "feature_null_summary.parquet",
        "target_summary.parquet",
        "label_source_summary.parquet",
    )
    missing = [name for name in required_files if not (materialized_dir / name).exists()]
    _add_check(rows, "materialization_output_files_exist", not missing, "blocking", ",".join(missing))
    if missing:
        return

    join_summary = pd.read_csv(materialized_dir / "join_coverage_summary.csv")
    join_row = join_summary.iloc[0].to_dict()
    join_ok = (
        abs(float(join_row["join_coverage"]) - 1.0) <= EPS
        and int(join_row["unmatched_rows"]) == 0
        and int(join_row["utility_key_duplicate_rows"]) == 0
        and int(join_row["behavior_key_duplicate_rows"]) == 0
        and int(join_row["fe_ratio_mismatch_count"]) == 0
    )
    _add_check(rows, "materialization_join_coverage_clean", join_ok, "blocking", json.dumps(join_row, sort_keys=True))

    input_legality = pd.read_csv(materialized_dir / "input_legality_summary.csv")
    legality_ok = bool(input_legality["passed"].astype(bool).all())
    _add_check(rows, "materialization_input_legality_passed", legality_ok, "blocking", "")


def _check_training_contract(
    training_contract_dir: Path,
    dataset: pd.DataFrame,
    input_columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    summary_path = training_contract_dir / "training_pipeline_contract_summary.json"
    report_path = training_contract_dir / "training_pipeline_contract_report.md"
    missing = [str(path) for path in (summary_path, report_path) if not path.exists()]
    _add_check(rows, "training_pipeline_contract_outputs_exist", not missing, "blocking", ",".join(missing))
    if missing:
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    preprocessing = dict(summary.get("preprocessing", {}))
    train_rows = int((dataset["split"].astype(str) == TRAIN_SPLIT).sum())
    contract_ok = (
        bool(summary.get("all_checks_passed")) is True
        and bool(summary.get("model_training_executed")) is False
        and list(summary.get("input_columns", [])) == input_columns
        and summary.get("target_column") == TARGET_COLUMN
        and preprocessing.get("imputer_fit_split") == TRAIN_SPLIT
        and int(preprocessing.get("imputer_fit_rows", -1)) == train_rows
        and preprocessing.get("scaler_fit_split") == TRAIN_SPLIT
        and int(preprocessing.get("scaler_fit_rows", -1)) == train_rows
        and int(preprocessing.get("validation_rows_used_for_fit", -1)) == 0
    )
    _add_check(rows, "training_pipeline_contract_clean", contract_ok, "blocking", json.dumps(summary, sort_keys=True))


def _check_changed_same_outputs(changed_same_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_path = changed_same_dir / "changed_same_learnability_summary.json"
    report_path = changed_same_dir / "changed_same_learnability_report.md"
    missing = [str(path) for path in (summary_path, report_path) if not path.exists()]
    _add_check(rows, "changed_same_learnability_outputs_exist", not missing, "warning", ",".join(missing))
    if missing:
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostic_ok = (
        bool(summary.get("main_complex_model_trained")) is False
        and bool(summary.get("metadata_used_as_input")) is False
        and bool(summary.get("algorithm_labels_used_as_input")) is False
        and list(summary.get("feature_columns", [])) == list(BEHAVIOR_FEATURE_COLUMNS)
    )
    _add_check(rows, "changed_same_learnability_boundary_clean", diagnostic_ok, "warning", json.dumps(summary, sort_keys=True))


def _check_stacking_outputs(stacking_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_path = stacking_dir / "algorithm_partition_stacking_summary.json"
    report_path = stacking_dir / "algorithm_partition_stacking_report.md"
    missing = [str(path) for path in (summary_path, report_path) if not path.exists()]
    _add_check(rows, "algorithm_partition_stacking_outputs_exist", not missing, "warning", ",".join(missing))
    if missing:
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostic_ok = (
        bool(summary.get("main_complex_model_trained")) is False
        and bool(summary.get("model_files_saved")) is False
        and bool(summary.get("metadata_used_as_model_input")) is False
        and bool(summary.get("algorithm_labels_used_as_global_or_meta_input")) is False
        and int(summary.get("validation_rows_used_for_fit", -1)) == 0
        and list(summary.get("feature_columns", [])) == list(BEHAVIOR_FEATURE_COLUMNS)
    )
    _add_check(rows, "algorithm_partition_stacking_boundary_clean", diagnostic_ok, "warning", json.dumps(summary, sort_keys=True))


def _split_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in dataset.groupby("split", dropna=False):
        target = group[TARGET_COLUMN].astype(float)
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "u_gt_zero_rows": int((target > 0.0).sum()),
                "u_gt_zero_rate": float((target > 0.0).mean()),
                "mean_u": float(target.mean()),
                "median_u": float(target.median()),
                "target_null_count": int(group[TARGET_COLUMN].isna().sum()),
                "target_finite": bool(np.isfinite(target.to_numpy(dtype=float)).all()),
            }
        )
    return pd.DataFrame(rows)


def _feature_summary(dataset: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    rows = []
    for split, group in dataset.groupby("split", dropna=False):
        for column in input_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            non_null = values.notna()
            finite = non_null & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
            rows.append(
                {
                    "split": str(split),
                    "feature": column,
                    "rows": int(len(group)),
                    "null_count": int(group[column].isna().sum()),
                    "null_rate": float(group[column].isna().mean()),
                    "finite_count": int(finite.sum()),
                    "finite_rate_nonnull": float(finite.sum() / max(int(non_null.sum()), 1)),
                }
            )
    return pd.DataFrame(rows)


def _label_source_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, label_source), group in dataset.groupby(["split", "label_source"], dropna=False):
        target = group[TARGET_COLUMN].astype(float)
        split_rows = int((dataset["split"] == split).sum())
        rows.append(
            {
                "split": str(split),
                "label_source": str(label_source),
                "rows": int(len(group)),
                "row_share_within_split": float(len(group) / max(split_rows, 1)),
                "u_gt_zero_rows": int((target > 0.0).sum()),
                "u_gt_zero_rate": float((target > 0.0).mean()),
                "mean_u": float(target.mean()),
                "median_u": float(target.median()),
                "positive_utility_sum": float(target[target > 0.0].sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _add_check(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    severity: str,
    detail: str,
) -> None:
    rows.append(
        {
            "check": check,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def _markdown_report(
    *,
    summary: dict[str, Any],
    readiness_checks: pd.DataFrame,
    split_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    label_source_summary: pd.DataFrame,
) -> str:
    failed = readiness_checks[~readiness_checks["passed"]]
    feature_compact = (
        feature_summary.groupby("feature", as_index=False)
        .agg(
            max_null_rate=("null_rate", "max"),
            min_finite_rate_nonnull=("finite_rate_nonnull", "min"),
        )
        .sort_values("feature")
    )
    return "\n".join(
        [
            "# Full Decision training readiness report",
            "",
            "## Scope",
            "",
            "- This report checks preconditions for full Decision Model training on phase1 refined sampling data.",
            "- It does not train the main Decision Model and does not save model files.",
            f"- Approved dataset: `{summary['approved_training_data']}`.",
            f"- Can start full training: `{summary['can_start_full_training']}`.",
            "",
            "## Approved main training contract",
            "",
            f"- X columns: `{', '.join(summary['input_columns'])}`.",
            f"- Target: `{summary['target_column']}`.",
            "- Fit split: `bbob_train`; validation split: `bbob_validation`.",
            "- Median imputer and scaler must be fit on train rows only.",
            "- Metadata, algorithm identifiers, ELA fields, selector fields, costs, targets, and diagnostics must not enter X.",
            "- `prefix_algorithm` may be used only for stratified reporting or diagnostic ablation boundaries.",
            "",
            "## Readiness checks",
            "",
            _markdown_table(readiness_checks[["check", "passed", "severity"]]),
            "",
            "## Failed checks",
            "",
            _markdown_table(failed[["check", "severity", "detail"]]) if len(failed) else "No failed checks.",
            "",
            "## Split target summary",
            "",
            _markdown_table(split_summary),
            "",
            "## Feature readiness summary",
            "",
            _markdown_table(feature_compact),
            "",
            "## label_source summary",
            "",
            _markdown_table(label_source_summary),
            "",
            "## Training boundary",
            "",
            "- Use the materialized dataset for formal full training; do not use older min-support train/validation label paths as the formal entrypoint.",
            "- The changed/same learnability and algorithm-partition stacking outputs are diagnostic context only.",
            "- Validation rows must not be used for imputer, scaler, model, or threshold fitting.",
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

    if frame.empty:
        return ""
    headers = list(frame.columns)
    table_rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check readiness for full Decision Model training without training a model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--materialized-dir", type=Path, default=DEFAULT_MATERIALIZED_DIR)
    parser.add_argument("--training-contract-dir", type=Path, default=DEFAULT_TRAINING_CONTRACT_DIR)
    parser.add_argument("--changed-same-dir", type=Path, default=DEFAULT_CHANGED_SAME_DIR)
    parser.add_argument("--stacking-dir", type=Path, default=DEFAULT_STACKING_DIR)
    parser.add_argument("--utility-root", type=Path, default=DEFAULT_UTILITY_ROOT)
    parser.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_full_training_readiness_check(
        dataset_path=args.dataset,
        schema_path=args.schema,
        materialized_dir=args.materialized_dir,
        training_contract_dir=args.training_contract_dir,
        changed_same_dir=args.changed_same_dir,
        stacking_dir=args.stacking_dir,
        utility_root=args.utility_root,
        behavior_root=args.behavior_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
