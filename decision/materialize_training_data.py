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


DEFAULT_UTILITY_ROOT = Path("results/utility_labels/phase1_refined_sampling")
DEFAULT_BEHAVIOR_ROOT = Path("results/phase1_refined_sampling")
DEFAULT_OUTPUT_DIR = Path("results/decision/phase1_refined_sampling/materialized_training_data")
TARGET_COLUMN = "u_ela_lamT_1"
AUXILIARY_LABEL_COLUMN = "need_ela_lamT_1"
EXPECTED_UTILITY_SHARDS = 72
EXPECTED_BEHAVIOR_SHARDS = 72
JOIN_KEY_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
)
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
DATASET_COLUMNS = METADATA_COLUMNS + (TARGET_COLUMN, AUXILIARY_LABEL_COLUMN) + BEHAVIOR_FEATURE_COLUMNS
FORBIDDEN_INPUT_COLUMNS = {
    "split",
    "problem_id",
    "family",
    "dimension",
    "algorithm",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    "default_algorithm",
    "selected_algorithm",
    "label_source",
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
    "u_ela_lamT_1",
    "u_ela_lamT_2",
    "need_ela_lamT_0",
    "need_ela_lamT_025",
    "need_ela_lamT_05",
    "need_ela_lamT_1",
    "need_ela_lamT_2",
}
FORBIDDEN_INPUT_NAME_FRAGMENTS = (
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


def materialize_decision_training_data(
    *,
    utility_root: Path,
    behavior_root: Path,
    output_dir: Path,
    expected_utility_shards: int,
    expected_behavior_shards: int,
    overwrite: bool,
) -> dict[str, Any]:
    utility_paths = sorted(utility_root.glob("*/*/dimension_*/utility_labels.parquet"))
    behavior_paths = sorted(behavior_root.glob("*/*/dimension_*/behavior.parquet"))
    _check_shard_counts(
        utility_paths=utility_paths,
        behavior_paths=behavior_paths,
        expected_utility_shards=expected_utility_shards,
        expected_behavior_shards=expected_behavior_shards,
    )
    _check_output_paths(output_dir, overwrite)

    utility = _read_parquet_shards(utility_paths, root_marker=utility_root.name)
    behavior = _read_parquet_shards(behavior_paths, root_marker=behavior_root.name).rename(columns={"algorithm": "prefix_algorithm"})
    _check_required_columns(utility, behavior)
    _check_input_legality()

    utility_duplicates = int(utility.duplicated(list(JOIN_KEY_COLUMNS)).sum())
    behavior_duplicates = int(behavior.duplicated(list(JOIN_KEY_COLUMNS)).sum())
    if utility_duplicates:
        raise ValueError(f"utility label join keys must be unique; duplicate rows: {utility_duplicates}")
    if behavior_duplicates:
        raise ValueError(f"behavior join keys must be unique; duplicate rows: {behavior_duplicates}")

    joined = utility.merge(
        behavior[list(JOIN_KEY_COLUMNS) + ["FE_ratio", *BEHAVIOR_FEATURE_COLUMNS]],
        on=list(JOIN_KEY_COLUMNS),
        how="left",
        suffixes=("_utility", "_behavior"),
        indicator=True,
    )
    matched = joined["_merge"].eq("both")
    join_coverage = float(matched.mean()) if len(joined) else 0.0
    unmatched_rows = int((~matched).sum())
    if join_coverage != 1.0:
        raise ValueError(f"utility-to-behavior join coverage must be 1.0; observed {join_coverage:.12g}")

    fe_ratio_delta = (joined["FE_ratio_utility"] - joined["FE_ratio_behavior"]).abs()
    fe_ratio_mismatch_count = int((fe_ratio_delta > EPS).sum())
    if fe_ratio_mismatch_count:
        raise ValueError(f"FE_ratio mismatch after join for {fe_ratio_mismatch_count} rows")

    dataset = _materialized_dataset(joined)
    _check_targets(dataset)
    _check_feature_values(dataset)
    _check_label_source(dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "decision_dataset.parquet"
    schema_path = output_dir / "decision_dataset_schema.json"
    report_path = output_dir / "decision_dataset_materialization_report.md"

    pq.write_table(pa.Table.from_pandas(dataset, preserve_index=False), dataset_path)
    feature_null_summary = _feature_null_summary(dataset)
    target_summary = _target_summary(dataset)
    label_source_summary = _label_source_summary(dataset)
    label_source_by_dimension = _label_source_by_dimension_summary(dataset)
    join_summary = _join_summary(
        utility_rows=len(utility),
        behavior_rows=len(behavior),
        joined_rows=len(joined),
        unmatched_rows=unmatched_rows,
        join_coverage=join_coverage,
        utility_duplicates=utility_duplicates,
        behavior_duplicates=behavior_duplicates,
        fe_ratio_mismatch_count=fe_ratio_mismatch_count,
    )
    input_legality = _input_legality_summary()

    _write_frame(feature_null_summary, output_dir / "feature_null_summary")
    _write_frame(target_summary, output_dir / "target_summary")
    _write_frame(label_source_summary, output_dir / "label_source_summary")
    _write_frame(label_source_by_dimension, output_dir / "label_source_by_dimension_summary")
    _write_frame(join_summary, output_dir / "join_coverage_summary")
    _write_frame(input_legality, output_dir / "input_legality_summary")

    schema_path.write_text(
        json.dumps(_schema_payload(dataset), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        _markdown_report(
            dataset=dataset,
            join_summary=join_summary,
            input_legality=input_legality,
            feature_null_summary=feature_null_summary,
            target_summary=target_summary,
            label_source_summary=label_source_summary,
            label_source_by_dimension=label_source_by_dimension,
            utility_shards=len(utility_paths),
            behavior_shards=len(behavior_paths),
            dataset_path=dataset_path,
            schema_path=schema_path,
        ),
        encoding="utf-8",
    )

    print(f"wrote materialized Decision dataset to {dataset_path}")
    print(f"wrote materialization schema summary to {schema_path}")
    print(f"wrote materialization report to {report_path}")
    return {
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "report": str(report_path),
        "rows": int(len(dataset)),
        "join_coverage": join_coverage,
        "utility_shards": len(utility_paths),
        "behavior_shards": len(behavior_paths),
    }


def _check_shard_counts(
    *,
    utility_paths: list[Path],
    behavior_paths: list[Path],
    expected_utility_shards: int,
    expected_behavior_shards: int,
) -> None:
    if len(utility_paths) != expected_utility_shards:
        raise ValueError(f"expected {expected_utility_shards} utility label shards, found {len(utility_paths)}")
    if len(behavior_paths) != expected_behavior_shards:
        raise ValueError(f"expected {expected_behavior_shards} behavior shards, found {len(behavior_paths)}")


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    output_paths = (
        output_dir / "decision_dataset.parquet",
        output_dir / "decision_dataset_schema.json",
        output_dir / "decision_dataset_materialization_report.md",
        output_dir / "feature_null_summary.csv",
        output_dir / "feature_null_summary.parquet",
        output_dir / "input_legality_summary.csv",
        output_dir / "input_legality_summary.parquet",
        output_dir / "join_coverage_summary.csv",
        output_dir / "join_coverage_summary.parquet",
        output_dir / "target_summary.csv",
        output_dir / "target_summary.parquet",
        output_dir / "label_source_summary.csv",
        output_dir / "label_source_summary.parquet",
        output_dir / "label_source_by_dimension_summary.csv",
        output_dir / "label_source_by_dimension_summary.parquet",
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"materialization outputs already exist; pass --overwrite to replace: {existing[0]}")


def _read_parquet_shards(paths: list[Path], *, root_marker: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pq.read_table(path).to_pandas()
        if "split" not in frame.columns:
            frame.insert(0, "split", _split_from_path(path, root_marker))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _split_from_path(path: Path, root_marker: str) -> str:
    parts = path.parts
    index = parts.index(root_marker)
    return parts[index + 1]


def _check_required_columns(utility: pd.DataFrame, behavior: pd.DataFrame) -> None:
    utility_required = set(JOIN_KEY_COLUMNS) | {
        "FE_ratio",
        "default_algorithm",
        "selected_algorithm",
        TARGET_COLUMN,
        AUXILIARY_LABEL_COLUMN,
    }
    behavior_required = set(JOIN_KEY_COLUMNS) | {"FE_ratio", *BEHAVIOR_FEATURE_COLUMNS}
    missing_utility = sorted(utility_required.difference(utility.columns))
    missing_behavior = sorted(behavior_required.difference(behavior.columns))
    if missing_utility:
        raise ValueError(f"utility labels missing required columns: {missing_utility}")
    if missing_behavior:
        raise ValueError(f"behavior features missing required columns: {missing_behavior}")


def _check_input_legality() -> None:
    input_columns = list(BEHAVIOR_FEATURE_COLUMNS)
    exact_forbidden = sorted(set(input_columns).intersection(FORBIDDEN_INPUT_COLUMNS))
    pattern_forbidden = [
        column
        for column in input_columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_INPUT_NAME_FRAGMENTS)
    ]
    if exact_forbidden:
        raise ValueError(f"Decision input contains forbidden columns: {exact_forbidden}")
    if pattern_forbidden:
        raise ValueError(f"Decision input contains forbidden name fragments: {pattern_forbidden}")


def _materialized_dataset(joined: pd.DataFrame) -> pd.DataFrame:
    dataset = joined[
        [
            "split",
            "problem_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio_utility",
            "default_algorithm",
            "selected_algorithm",
            TARGET_COLUMN,
            AUXILIARY_LABEL_COLUMN,
            *[f"{column}_behavior" for column in BEHAVIOR_FEATURE_COLUMNS],
        ]
    ].copy()
    dataset = dataset.rename(
        columns={
            "FE_ratio_utility": "FE_ratio",
            **{f"{column}_behavior": column for column in BEHAVIOR_FEATURE_COLUMNS},
        }
    )
    dataset["label_source"] = np.where(
        dataset["selected_algorithm"].astype(str) == dataset["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return dataset[list(DATASET_COLUMNS)]


def _check_targets(dataset: pd.DataFrame) -> None:
    target = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError(f"target column {TARGET_COLUMN} must not contain null values")
    if not np.isfinite(target).all():
        raise ValueError(f"target column {TARGET_COLUMN} must contain only finite values")
    expected_label = dataset[TARGET_COLUMN].to_numpy(dtype=float) > 0.0
    observed_label = dataset[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool)
    if not np.array_equal(expected_label, observed_label):
        raise ValueError(f"{AUXILIARY_LABEL_COLUMN} must equal {TARGET_COLUMN} > 0")


def _check_feature_values(dataset: pd.DataFrame) -> None:
    invalid = []
    for column in BEHAVIOR_FEATURE_COLUMNS:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        non_finite = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        count = int(non_finite.sum())
        if count:
            invalid.append({"feature": column, "non_finite_non_null_rows": count})
    if invalid:
        raise ValueError(f"non-null behavior feature values must be finite: {invalid}")


def _check_label_source(dataset: pd.DataFrame) -> None:
    expected = np.where(
        dataset["selected_algorithm"].astype(str) == dataset["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    if not np.array_equal(dataset["label_source"].to_numpy(dtype=str), expected):
        raise ValueError("label_source must match selected_algorithm == default_algorithm")


def _join_summary(
    *,
    utility_rows: int,
    behavior_rows: int,
    joined_rows: int,
    unmatched_rows: int,
    join_coverage: float,
    utility_duplicates: int,
    behavior_duplicates: int,
    fe_ratio_mismatch_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "utility_rows": int(utility_rows),
                "behavior_rows": int(behavior_rows),
                "joined_rows": int(joined_rows),
                "unmatched_rows": int(unmatched_rows),
                "join_coverage": float(join_coverage),
                "utility_key_duplicate_rows": int(utility_duplicates),
                "behavior_key_duplicate_rows": int(behavior_duplicates),
                "fe_ratio_mismatch_count": int(fe_ratio_mismatch_count),
            }
        ]
    )


def _input_legality_summary() -> pd.DataFrame:
    input_columns = list(BEHAVIOR_FEATURE_COLUMNS)
    exact_forbidden = sorted(set(input_columns).intersection(FORBIDDEN_INPUT_COLUMNS))
    pattern_forbidden = [
        column
        for column in input_columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_INPUT_NAME_FRAGMENTS)
    ]
    return pd.DataFrame(
        [
            {
                "check": "input_columns_equal_behavior_feature_columns",
                "passed": input_columns == list(BEHAVIOR_FEATURE_COLUMNS),
                "detail": ",".join(input_columns),
            },
            {
                "check": "forbidden_exact_columns_absent_from_inputs",
                "passed": len(exact_forbidden) == 0,
                "detail": ",".join(exact_forbidden),
            },
            {
                "check": "forbidden_name_fragments_absent_from_inputs",
                "passed": len(pattern_forbidden) == 0,
                "detail": ",".join(pattern_forbidden),
            },
        ]
    )


def _feature_null_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_frame in _group_with_overall(dataset, ["split"]):
        rows.extend(_feature_rows(split_frame, split=split, fe_ratio=None))
    for (split, fe_ratio), group in dataset.groupby(["split", "FE_ratio"], dropna=False):
        rows.extend(_feature_rows(group, split=str(split), fe_ratio=float(fe_ratio)))
    return pd.DataFrame(rows)


def _feature_rows(frame: pd.DataFrame, *, split: str, fe_ratio: float | None) -> list[dict[str, Any]]:
    rows = []
    for column in BEHAVIOR_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        null_count = int(frame[column].isna().sum())
        non_null = values.notna()
        finite = non_null & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        finite_count = int(finite.sum())
        non_null_count = int(non_null.sum())
        finite_values = values[finite]
        rows.append(
            {
                "split": split,
                "FE_ratio": fe_ratio,
                "feature": column,
                "rows": int(len(frame)),
                "null_count": null_count,
                "null_rate": float(null_count / max(len(frame), 1)),
                "finite_count": finite_count,
                "finite_rate_nonnull": float(finite_count / max(non_null_count, 1)),
                "min": _float_or_none(finite_values.min()) if finite_count else None,
                "median": _float_or_none(finite_values.median()) if finite_count else None,
                "max": _float_or_none(finite_values.max()) if finite_count else None,
            }
        )
    return rows


def _target_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = [_target_row(dataset, split="overall")]
    for split, group in dataset.groupby("split", dropna=False):
        rows.append(_target_row(group, split=str(split)))
    return pd.DataFrame(rows)


def _target_row(frame: pd.DataFrame, *, split: str) -> dict[str, Any]:
    values = frame[TARGET_COLUMN].astype(float)
    return {
        "split": split,
        "rows": int(len(frame)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "q95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "u_gt_zero_rows": int((values > 0.0).sum()),
        "u_gt_zero_rate": float((values > 0.0).mean()),
        "abs_u_le_1e_3_rate": float((values.abs() <= 1e-3).mean()),
        "abs_u_le_1e_2_rate": float((values.abs() <= 1e-2).mean()),
    }


def _label_source_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, label_source), group in dataset.groupby(["split", "label_source"], dropna=False):
        split_rows = int((dataset["split"] == split).sum())
        values = group[TARGET_COLUMN].astype(float)
        rows.append(
            {
                "split": str(split),
                "label_source": str(label_source),
                "rows": int(len(group)),
                "row_share_within_split": float(len(group) / max(split_rows, 1)),
                "u_gt_zero_rate": float((values > 0.0).mean()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "u_gt_zero_utility_sum": float(values[values > 0.0].sum()),
            }
        )
    return pd.DataFrame(rows)


def _label_source_by_dimension_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, label_source, dimension), group in dataset.groupby(["split", "label_source", "dimension"], dropna=False):
        values = group[TARGET_COLUMN].astype(float)
        rows.append(
            {
                "split": str(split),
                "label_source": str(label_source),
                "dimension": int(dimension),
                "rows": int(len(group)),
                "u_gt_zero_rate": float((values > 0.0).mean()),
                "mean": float(values.mean()),
                "median": float(values.median()),
            }
        )
    return pd.DataFrame(rows)


def _group_with_overall(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, pd.DataFrame]]:
    groups = [("overall", frame)]
    for key, group in frame.groupby(columns, dropna=False):
        if isinstance(key, tuple):
            key_text = "|".join(str(item) for item in key)
        else:
            key_text = str(key)
        groups.append((key_text, group))
    return groups


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _schema_payload(dataset: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": "phase1_refined_sampling_decision_training_data",
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "input_columns": list(BEHAVIOR_FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "column_order": list(DATASET_COLUMNS),
        "column_dtypes": {column: str(dtype) for column, dtype in dataset.dtypes.items()},
        "label_source_rule": {
            "same_algorithm": "selected_algorithm == default_algorithm",
            "changed_algorithm": "selected_algorithm != default_algorithm",
        },
        "materialization_rules": {
            "missing_behavior_features": "preserved in materialized dataset; train-split median imputation belongs to the training pipeline",
            "finite_extreme_values": "preserved without clipping, scaling, or replacement",
            "non_null_infinite_values": "not allowed",
            "row_inclusion": "all joined rows with finite target are retained, including early FE checkpoints",
        },
        "excluded_from_decision_input": sorted(FORBIDDEN_INPUT_COLUMNS),
    }


def _markdown_report(
    *,
    dataset: pd.DataFrame,
    join_summary: pd.DataFrame,
    input_legality: pd.DataFrame,
    feature_null_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    label_source_summary: pd.DataFrame,
    label_source_by_dimension: pd.DataFrame,
    utility_shards: int,
    behavior_shards: int,
    dataset_path: Path,
    schema_path: Path,
) -> str:
    overall_features = feature_null_summary[feature_null_summary["FE_ratio"].isna()]
    return "\n".join(
        [
            "# Decision Model training data materialization report",
            "",
            "## Scope",
            "",
            "- Data source: formal phase1 refined sampling utility labels joined to formal behavior features.",
            f"- Utility shards: {utility_shards}; behavior shards: {behavior_shards}.",
            f"- Materialized rows: {len(dataset)}.",
            f"- Dataset output: `{dataset_path}`.",
            f"- Schema summary output: `{schema_path}`.",
            "- No model training, imputation, clipping, scaling, or threshold calibration was run.",
            "",
            "## Decision inputs and target",
            "",
            f"- Input columns: `{', '.join(BEHAVIOR_FEATURE_COLUMNS)}`.",
            f"- Main regression target: `{TARGET_COLUMN}`.",
            f"- Auxiliary decision label: `{AUXILIARY_LABEL_COLUMN}`.",
            "- Metadata and stratification columns are retained only for reporting, splitting, and error analysis.",
            "",
            "## Join coverage",
            "",
            _markdown_table(join_summary),
            "",
            "## Input legality",
            "",
            _markdown_table(input_legality),
            "",
            "## Feature null and finite summary",
            "",
            _markdown_table(
                overall_features[
                    [
                        "split",
                        "feature",
                        "rows",
                        "null_count",
                        "null_rate",
                        "finite_count",
                        "finite_rate_nonnull",
                        "min",
                        "median",
                        "max",
                    ]
                ]
            ),
            "",
            "## Target distribution",
            "",
            _markdown_table(target_summary),
            "",
            "## same_algorithm / changed_algorithm summary",
            "",
            _markdown_table(label_source_summary),
            "",
            "## same_algorithm / changed_algorithm by dimension",
            "",
            _markdown_table(label_source_by_dimension),
            "",
            "## Processing rules",
            "",
            "- Missing `bf_*` values are preserved in the materialized table.",
            "- Training code must fit median imputation on the train split only.",
            "- Finite extreme feature values are preserved without clipping or scaling.",
            "- Non-null infinite feature values and non-finite target values fail materialization.",
            "- `label_source` is `same_algorithm` only when `selected_algorithm == default_algorithm`.",
            "- ELA features, function id, algorithm id, optimizer parameters, selector fields, cost ledger fields, and utility component fields are excluded from Decision inputs.",
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


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize phase1 Decision Model training data without model training.")
    parser.add_argument("--utility-root", type=Path, default=DEFAULT_UTILITY_ROOT)
    parser.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-utility-shards", type=int, default=EXPECTED_UTILITY_SHARDS)
    parser.add_argument("--expected-behavior-shards", type=int, default=EXPECTED_BEHAVIOR_SHARDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    materialize_decision_training_data(
        utility_root=args.utility_root,
        behavior_root=args.behavior_root,
        output_dir=args.output_dir,
        expected_utility_shards=args.expected_utility_shards,
        expected_behavior_shards=args.expected_behavior_shards,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
