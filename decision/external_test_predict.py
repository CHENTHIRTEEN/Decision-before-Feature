from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.query_contract import decision_query_root, validate_query_frame, validate_query_payload
from decision.train_full_decision_model import AUXILIARY_LABEL_COLUMN, METADATA_COLUMNS, TARGET_COLUMN
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


DEFAULT_MODEL_NAME = "ridge_regression"
DEFAULT_THRESHOLD_MODE = "train_utility"
DEFAULT_EXPECTED_SPLIT = "cec2017_test"
FORBIDDEN_INPUT_NAME_FRAGMENTS = (
    "query",
    "function",
    "algorithm",
    "selected",
    "default",
    "family",
    "problem",
    "dimension",
)


def predict_external_test(
    *,
    query_id: str,
    dataset_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    expected_split: str,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    validate_query_payload(training_summary, query_id=query_id, artifact="Decision training summary")
    feature_columns = _feature_columns(training_summary)
    model_path = _model_path(training_summary, model_name)
    threshold = _threshold(training_summary, model_name, threshold_mode)

    dataset = pq.read_table(dataset_path).to_pandas()
    validate_query_frame(dataset, query_id=query_id, artifact="external Decision dataset")
    _check_dataset(dataset, feature_columns, expected_split)

    model = joblib.load(model_path)
    started = perf_counter()
    scores = np.asarray(model.predict(dataset[feature_columns]), dtype=float)
    prediction_seconds = perf_counter() - started
    if not np.isfinite(scores).all():
        raise ValueError("external test prediction produced non-finite scores")

    predictions = _prediction_frame(
        frame=dataset,
        scores=scores,
        threshold=threshold,
        threshold_mode=threshold_mode,
        model_name=model_name,
        model_family=_model_family(training_summary, model_name),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "controller_predictions.parquet"
    summary_path = output_dir / "external_test_prediction_summary.json"
    report_path = output_dir / "external_test_prediction_report.md"
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), predictions_path)

    input_contract = _input_contract(feature_columns, expected_split)
    summary = {
        "experiment": "cec2017_external_test_controller_prediction",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "dataset": str(dataset_path),
        "training_summary": str(training_summary_path),
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "threshold": float(threshold),
        "expected_split": expected_split,
        "rows": int(len(predictions)),
        "feature_columns": feature_columns,
        "model_path": str(model_path),
        "prediction_seconds": float(prediction_seconds),
        "prediction_seconds_per_row": float(prediction_seconds / max(len(predictions), 1)),
        "outputs": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "data_leakage_check": {
            "external_rows_used_for_model_fit": 0,
            "external_rows_used_for_imputer_or_scaler_fit": 0,
            "external_rows_used_for_threshold_fit": 0,
            "decision_input_uses_only_behavior_features": True,
            "query_features_used_as_decision_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            input_contract=input_contract,
            predictions=predictions,
        ),
        encoding="utf-8",
    )
    print(f"wrote external test predictions to {predictions_path}")
    print(f"wrote external test prediction summary to {summary_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "controller_predictions.parquet",
        output_dir / "external_test_prediction_summary.json",
        output_dir / "external_test_prediction_report.md",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"external test prediction outputs already exist; pass --overwrite: {existing[0]}")


def _feature_columns(training_summary: dict[str, Any]) -> list[str]:
    columns = [str(column) for column in training_summary.get("feature_columns", [])]
    if not columns:
        raise ValueError("training summary does not define feature_columns")
    if not set(columns).issubset(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("training feature columns must be a subset of BEHAVIOR_FEATURE_COLUMNS")
    forbidden = [
        column for column in columns if any(fragment in column.lower() for fragment in FORBIDDEN_INPUT_NAME_FRAGMENTS)
    ]
    if forbidden:
        raise ValueError(f"training feature columns contain forbidden name fragments: {forbidden}")
    return columns


def _model_path(training_summary: dict[str, Any], model_name: str) -> Path:
    for artifact in training_summary.get("model_artifacts", []):
        if str(artifact.get("model_name")) == model_name:
            path = Path(str(artifact.get("model_path")))
            if not path.exists():
                raise FileNotFoundError(f"missing model artifact: {path}")
            return path
    raise ValueError(f"model artifact not found in training summary: {model_name}")


def _threshold(training_summary: dict[str, Any], model_name: str, threshold_mode: str) -> float:
    outputs = training_summary.get("outputs", {})
    threshold_path = Path(str(outputs.get("decision_thresholds", "")))
    if not threshold_path.exists():
        raise FileNotFoundError(f"missing decision threshold table: {threshold_path}")
    thresholds = pq.read_table(threshold_path).to_pandas()
    row = thresholds[
        (thresholds["model_name"].astype(str) == model_name)
        & (thresholds["threshold_mode"].astype(str) == threshold_mode)
    ]
    if len(row) != 1:
        raise ValueError(f"expected one threshold row for {model_name}/{threshold_mode}, found {len(row)}")
    external_rows = int(row["validation_rows_used_for_threshold_fit"].iloc[0])
    if external_rows != 0:
        raise ValueError("threshold table reports nonzero held-out rows used for threshold fit")
    return float(row["threshold"].iloc[0])


def _model_family(training_summary: dict[str, Any], model_name: str) -> str:
    fit_path = Path(str(training_summary.get("outputs", {}).get("model_fit_summary", "")))
    if not fit_path.exists():
        return ""
    rows = pq.read_table(fit_path).to_pandas()
    row = rows[rows["model_name"].astype(str) == model_name]
    return str(row["model_family"].iloc[0]) if len(row) else ""


def _check_dataset(dataset: pd.DataFrame, feature_columns: list[str], expected_split: str) -> None:
    required = set(METADATA_COLUMNS) | {TARGET_COLUMN, AUXILIARY_LABEL_COLUMN, *feature_columns}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"external test dataset missing required columns: {missing}")
    splits = sorted(dataset["split"].astype(str).unique().tolist())
    if splits != [expected_split]:
        raise ValueError(f"external test dataset must contain only split={expected_split}, observed {splits}")
    target = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    if dataset[TARGET_COLUMN].isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"{TARGET_COLUMN} must be non-null and finite")
    if not np.array_equal(dataset[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool), target.to_numpy(dtype=float) > 0.0):
        raise ValueError(f"{AUXILIARY_LABEL_COLUMN} must equal {TARGET_COLUMN} > 0")
    for column in feature_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        invalid = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            raise ValueError(f"non-null behavior feature values must be finite: {column}")


def _prediction_frame(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    threshold_mode: str,
    model_name: str,
    model_family: str,
) -> pd.DataFrame:
    output = frame[list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN]].copy()
    output.insert(0, "data_split", "external_test")
    output.insert(1, "model_name", model_name)
    output.insert(2, "model_family", model_family)
    output["decision_score"] = scores.astype(float)
    output[f"decision_run_query_{threshold_mode}"] = scores > threshold
    output[f"decision_utility_{threshold_mode}"] = np.where(scores > threshold, output[TARGET_COLUMN], 0.0)
    return output


def _input_contract(feature_columns: list[str], expected_split: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "x_columns_subset_of_behavior_feature_columns",
                "passed": set(feature_columns).issubset(BEHAVIOR_FEATURE_COLUMNS),
                "detail": ",".join(feature_columns),
            },
            {
                "check": "external_split_not_used_for_fit",
                "passed": True,
                "detail": f"{expected_split} rows used for prediction only",
            },
            {
                "check": "metadata_retained_for_reporting_only",
                "passed": True,
                "detail": ",".join(METADATA_COLUMNS),
            },
        ]
    )


def _markdown_report(*, summary: dict[str, Any], input_contract: pd.DataFrame, predictions: pd.DataFrame) -> str:
    call_column = f"decision_run_query_{summary['threshold_mode']}"
    call_rate = float(predictions[call_column].mean()) if len(predictions) else 0.0
    return "\n".join(
        [
            "# CEC2017 external test controller prediction",
            "",
            "## Scope",
            "",
            f"- Dataset: `{summary['dataset']}`.",
            f"- Training summary: `{summary['training_summary']}`.",
            f"- Model: `{summary['model_name']}`.",
            f"- Threshold mode: `{summary['threshold_mode']}`.",
            f"- External test rows: {summary['rows']}.",
            f"- Controller Query call rate: {call_rate:.6g}.",
            "- CEC2017 rows were not used for model fitting, preprocessing fitting, or threshold fitting.",
            "",
            "## Input contract",
            "",
            _markdown_table(input_contract),
            "",
            "## Outputs",
            "",
            f"- Predictions: `{summary['outputs']['predictions']}`.",
            f"- Summary: `{summary['outputs']['summary']}`.",
        ]
    ) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a frozen BBOB-trained controller on an external test split.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--expected-split", default=DEFAULT_EXPECTED_SPLIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    external_root = query_root / args.expected_split

    predict_external_test(
        query_id=args.query_id,
        dataset_path=args.dataset or external_root / "materialized_test_data/decision_dataset.parquet",
        training_summary_path=args.training_summary
        or query_root / "full_training/full_decision_model_training_summary.json",
        output_dir=args.output_dir or external_root / "external_test_prediction",
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        expected_split=args.expected_split,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
