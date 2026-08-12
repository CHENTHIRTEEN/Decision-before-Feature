from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.query_contract import decision_query_root, validate_query_frame, validate_query_payload
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_query_lamT_1"
DOMAIN_SPECS = {
    "all": None,
    "handoff_required=true": True,
    "handoff_required=false": False,
}
TOP_K_FRACTIONS = (0.05, 0.10, 0.20)
EPS = 1e-12


def run_handoff_learnability_diagnostic(
    *,
    query_id: str,
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    overwrite: bool,
    random_seed: int,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_query_payload(schema, query_id=query_id, artifact="Decision schema")
    validate_query_frame(dataset, query_id=query_id, artifact="Decision dataset")
    feature_columns = _feature_columns(schema)
    _check_dataset(dataset, feature_columns)

    domain_summary = _domain_summary(dataset)
    regression_rows = []
    ranking_rows = []
    train_score_rows = []
    validation_score_rows = []

    for domain, handoff_required in DOMAIN_SPECS.items():
        train = _domain_frame(dataset, TRAIN_SPLIT, handoff_required)
        validation = _domain_frame(dataset, VALIDATION_SPLIT, handoff_required)
        _check_domain(domain, train, validation)

        for model_name, model in _model_specs(random_seed).items():
            fitted = _fit_baseline(model, train, feature_columns)
            train_scores = _predict_baseline(fitted, train, feature_columns)
            validation_scores = _predict_baseline(fitted, validation, feature_columns)
            train_score_rows.append(_score_frame(train, train_scores, domain, model_name, "train"))
            validation_score_rows.append(_score_frame(validation, validation_scores, domain, model_name, "validation"))
            regression_rows.append(
                _regression_row(
                    domain=domain,
                    model_name=model_name,
                    train=train,
                    validation=validation,
                    train_scores=train_scores,
                    validation_scores=validation_scores,
                )
            )
            ranking_rows.extend(
                _ranking_rows(
                    domain=domain,
                    model_name=model_name,
                    validation=validation,
                    validation_scores=validation_scores,
                )
            )

    regression_summary = pd.DataFrame(regression_rows)
    ranking_summary = pd.DataFrame(ranking_rows)
    train_scores = pd.concat(train_score_rows, ignore_index=True)
    validation_scores = pd.concat(validation_score_rows, ignore_index=True)
    signal_summary = _signal_summary(regression_summary, ranking_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(domain_summary, output_dir / "domain_summary")
    _write_frame(regression_summary, output_dir / "baseline_regression_summary")
    _write_frame(ranking_summary, output_dir / "baseline_ranking_summary")
    _write_frame(signal_summary, output_dir / "behavior_signal_summary")
    pq.write_table(pa.Table.from_pandas(train_scores, preserve_index=False), output_dir / "baseline_train_scores.parquet")
    pq.write_table(pa.Table.from_pandas(validation_scores, preserve_index=False), output_dir / "baseline_validation_scores.parquet")

    summary = {
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "experiment": "handoff_behavior_learnability_diagnostic",
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "domains": DOMAIN_SPECS,
        "models": sorted(regression_summary["model_name"].unique().tolist()),
        "top_k_fractions": list(TOP_K_FRACTIONS),
        "model_training_scope": "baseline diagnostic only",
        "main_complex_model_trained": False,
        "metadata_used_as_input": False,
        "algorithm_labels_used_as_input": False,
        "outputs": {
            "domain_summary": str(output_dir / "domain_summary.parquet"),
            "baseline_regression_summary": str(output_dir / "baseline_regression_summary.parquet"),
            "baseline_ranking_summary": str(output_dir / "baseline_ranking_summary.parquet"),
            "behavior_signal_summary": str(output_dir / "behavior_signal_summary.parquet"),
            "train_scores": str(output_dir / "baseline_train_scores.parquet"),
            "validation_scores": str(output_dir / "baseline_validation_scores.parquet"),
            "report": str(output_dir / "handoff_learnability_report.md"),
        },
    }
    summary_path = output_dir / "handoff_learnability_summary.json"
    report_path = output_dir / "handoff_learnability_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            domain_summary=domain_summary,
            regression_summary=regression_summary,
            ranking_summary=ranking_summary,
            signal_summary=signal_summary,
            feature_columns=feature_columns,
        ),
        encoding="utf-8",
    )

    print(f"wrote handoff learnability summary to {summary_path}")
    print(f"wrote handoff learnability report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "handoff_learnability_summary.json",
        output_dir / "handoff_learnability_report.md",
        output_dir / "domain_summary.csv",
        output_dir / "domain_summary.parquet",
        output_dir / "baseline_regression_summary.csv",
        output_dir / "baseline_regression_summary.parquet",
        output_dir / "baseline_ranking_summary.csv",
        output_dir / "baseline_ranking_summary.parquet",
        output_dir / "behavior_signal_summary.csv",
        output_dir / "behavior_signal_summary.parquet",
        output_dir / "baseline_train_scores.parquet",
        output_dir / "baseline_validation_scores.parquet",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"learnability diagnostic outputs already exist; pass --overwrite: {existing[0]}")


def _feature_columns(schema: dict[str, Any]) -> list[str]:
    columns = list(schema.get("input_columns", []))
    if columns != list(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("schema input_columns must exactly equal BEHAVIOR_FEATURE_COLUMNS")
    forbidden_fragments = ("query", "function", "algorithm", "selected", "default", "family", "problem", "dimension")
    forbidden = [column for column in columns if any(fragment in column.lower() for fragment in forbidden_fragments)]
    if forbidden:
        raise ValueError(f"Decision input contains forbidden name fragments: {forbidden}")
    return columns


def _check_dataset(dataset: pd.DataFrame, feature_columns: list[str]) -> None:
    required = {
        "split",
        "handoff_required",
        "problem_id",
        "family",
        "dimension",
        "prefix_algorithm",
        "default_algorithm",
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "skip_switches_from_prefix",
        "no_query_transition_mode",
        "query_transition_mode",
        "handoff_type",
        "seed",
        "FE",
        "FE_ratio",
        TARGET_COLUMN,
        *feature_columns,
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"materialized dataset missing required columns: {missing}")
    if set(dataset["split"].astype(str).unique()) != {TRAIN_SPLIT, VALIDATION_SPLIT}:
        raise ValueError(f"expected splits {TRAIN_SPLIT} and {VALIDATION_SPLIT}")
    target = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    if dataset[TARGET_COLUMN].isna().any() or not np.isfinite(target).all():
        raise ValueError(f"{TARGET_COLUMN} must be non-null and finite")
    selected_equals_default = (
        dataset["selected_algorithm"].astype(str) == dataset["default_algorithm"].astype(str)
    ).to_numpy(dtype=bool)
    selected_equals_prefix = (
        dataset["selected_algorithm"].astype(str) == dataset["prefix_algorithm"].astype(str)
    ).to_numpy(dtype=bool)
    expected_action = dataset["selected_algorithm"].astype(str).where(
        ~selected_equals_prefix,
        "continue_current",
    )
    expected_handoff = ~selected_equals_prefix
    if not np.array_equal(dataset["selected_equals_default"].to_numpy(dtype=bool), selected_equals_default):
        raise ValueError("selected_equals_default is inconsistent")
    if not np.array_equal(dataset["selected_equals_prefix"].to_numpy(dtype=bool), selected_equals_prefix):
        raise ValueError("selected_equals_prefix is inconsistent")
    if not np.array_equal(dataset["selected_action"].astype(str).to_numpy(), expected_action.to_numpy()):
        raise ValueError("selected_action is inconsistent")
    if not np.array_equal(dataset["handoff_required"].to_numpy(dtype=bool), expected_handoff):
        raise ValueError("handoff_required is inconsistent")
    if not np.array_equal(
        dataset["handoff_required"].to_numpy(dtype=bool),
        dataset["handoff_type"].astype(str).eq("population_transfer_initialization").to_numpy(dtype=bool),
    ):
        raise ValueError("handoff_required must match handoff_type")
    expected_transition = np.where(
        expected_handoff,
        "population_transfer_initialization",
        "native_optimizer_state",
    )
    if not np.array_equal(dataset["query_transition_mode"].astype(str).to_numpy(), expected_transition):
        raise ValueError("query_transition_mode is inconsistent")
    if not np.array_equal(dataset["handoff_type"].astype(str).to_numpy(), expected_transition):
        raise ValueError("handoff_type is inconsistent")
    for column in feature_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        invalid = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            raise ValueError(f"non-null feature values must be finite: {column}")


def _domain_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, handoff_required in DOMAIN_SPECS.items():
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT):
            frame = _domain_frame(dataset, split, handoff_required)
            values = frame[TARGET_COLUMN].to_numpy(dtype=float)
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "handoff_required_filter": "all" if handoff_required is None else bool(handoff_required),
                    "rows": int(len(frame)),
                    "u_gt_zero_rows": int(np.sum(values > 0.0)),
                    "u_gt_zero_rate": float(np.mean(values > 0.0)),
                    "mean_u": float(np.mean(values)),
                    "median_u": float(np.median(values)),
                    "positive_utility_sum": float(np.sum(values[values > 0.0])),
                }
            )
    return pd.DataFrame(rows)


def _domain_frame(dataset: pd.DataFrame, split: str, handoff_required: bool | None) -> pd.DataFrame:
    frame = dataset[dataset["split"] == split].copy()
    if handoff_required is not None:
        frame = frame[frame["handoff_required"].astype(bool) == handoff_required].copy()
    return frame


def _check_domain(domain: str, train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if train.empty:
        raise ValueError(f"empty train rows for domain {domain}")
    if validation.empty:
        raise ValueError(f"empty validation rows for domain {domain}")


def _model_specs(random_seed: int) -> dict[str, Any]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge_linear": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
            ]
        ),
    }


def _fit_baseline(model: Any, train: pd.DataFrame, feature_columns: list[str]) -> Any:
    model.fit(train[feature_columns], train[TARGET_COLUMN])
    return model


def _predict_baseline(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict(frame[feature_columns]).astype(float)


def _score_frame(frame: pd.DataFrame, scores: np.ndarray, domain: str, model_name: str, data_split: str) -> pd.DataFrame:
    result = frame[
        [
            "problem_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio",
            "handoff_required",
            TARGET_COLUMN,
        ]
    ].copy()
    result.insert(0, "data_split", data_split)
    result.insert(0, "model_name", model_name)
    result.insert(0, "domain", domain)
    result["decision_score"] = scores
    return result


def _regression_row(
    *,
    domain: str,
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
) -> dict[str, Any]:
    train_y = train[TARGET_COLUMN].to_numpy(dtype=float)
    validation_y = validation[TARGET_COLUMN].to_numpy(dtype=float)
    return {
        "domain": domain,
        "model_name": model_name,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_u_gt_zero_rate": float(np.mean(train_y > 0.0)),
        "validation_u_gt_zero_rate": float(np.mean(validation_y > 0.0)),
        "train_mae": float(mean_absolute_error(train_y, train_scores)),
        "validation_mae": float(mean_absolute_error(validation_y, validation_scores)),
        "train_rmse": float(mean_squared_error(train_y, train_scores) ** 0.5),
        "validation_rmse": float(mean_squared_error(validation_y, validation_scores) ** 0.5),
        "train_r2": _finite_or_none(lambda: r2_score(train_y, train_scores)),
        "validation_r2": _finite_or_none(lambda: r2_score(validation_y, validation_scores)),
        "train_pearson": _correlation(train_y, train_scores, method="pearson"),
        "validation_pearson": _correlation(validation_y, validation_scores, method="pearson"),
        "train_spearman": _correlation(train_y, train_scores, method="spearman"),
        "validation_spearman": _correlation(validation_y, validation_scores, method="spearman"),
    }


def _ranking_rows(
    *,
    domain: str,
    model_name: str,
    validation: pd.DataFrame,
    validation_scores: np.ndarray,
) -> list[dict[str, Any]]:
    observed = validation[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    positive_utility_sum = float(np.sum(observed[positive]))
    score_unique_count = int(np.unique(validation_scores).size)
    ranking_status = "informative_score_order" if score_unique_count > 1 else "constant_score_tie_reference"
    order = np.lexsort(
        (
            validation["FE"].to_numpy(dtype=int),
            validation["seed"].to_numpy(dtype=int),
            validation["problem_id"].astype(str).to_numpy(),
            -validation_scores,
        )
    )
    rows = []
    for fraction in TOP_K_FRACTIONS:
        top_k_rows = max(1, int(np.ceil(len(validation) * fraction)))
        top_index = order[:top_k_rows]
        top_positive = positive[top_index]
        top_observed = observed[top_index]
        rows.append(
            {
                "domain": domain,
                "model_name": model_name,
                "ranking_status": ranking_status,
                "score_unique_count": score_unique_count,
                "top_k_fraction": float(fraction),
                "top_k_rows": int(top_k_rows),
                "validation_rows": int(len(validation)),
                "validation_u_gt_zero_rows": int(np.sum(positive)),
                "validation_u_gt_zero_rate": float(np.mean(positive)),
                "top_k_u_gt_zero_rows": int(np.sum(top_positive)),
                "top_k_positive_row_rate": float(np.mean(top_positive)),
                "top_k_positive_row_lift": float(np.mean(top_positive) / max(float(np.mean(positive)), EPS)),
                "top_k_positive_row_capture_rate": float(np.sum(top_positive) / max(int(np.sum(positive)), 1)),
                "top_k_positive_utility_sum": float(np.sum(top_observed[top_positive])),
                "top_k_utility_capture_rate": (
                    float(np.sum(top_observed[top_positive]) / positive_utility_sum)
                    if positive_utility_sum > 0.0
                    else 0.0
                ),
                "top_k_observed_u_mean": float(np.mean(top_observed)),
            }
        )
    return rows


def _signal_summary(regression_summary: pd.DataFrame, ranking_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_SPECS:
        dummy = regression_summary[
            (regression_summary["domain"] == domain) & (regression_summary["model_name"] == "dummy_mean")
        ].iloc[0]
        ridge = regression_summary[
            (regression_summary["domain"] == domain) & (regression_summary["model_name"] == "ridge_linear")
        ].iloc[0]
        dummy_top10 = ranking_summary[
            (ranking_summary["domain"] == domain)
            & (ranking_summary["model_name"] == "dummy_mean")
            & (ranking_summary["top_k_fraction"] == 0.10)
        ].iloc[0]
        ridge_top10 = ranking_summary[
            (ranking_summary["domain"] == domain)
            & (ranking_summary["model_name"] == "ridge_linear")
            & (ranking_summary["top_k_fraction"] == 0.10)
        ].iloc[0]
        rows.append(
            {
                "domain": domain,
                "validation_rows": int(ridge["validation_rows"]),
                "validation_u_gt_zero_rate": float(ridge["validation_u_gt_zero_rate"]),
                "mae_improvement_vs_dummy": float(dummy["validation_mae"] - ridge["validation_mae"]),
                "rmse_improvement_vs_dummy": float(dummy["validation_rmse"] - ridge["validation_rmse"]),
                "ridge_validation_r2": float(ridge["validation_r2"]),
                "ridge_validation_spearman": float(ridge["validation_spearman"]),
                "top10_positive_row_lift_vs_base_rate": float(ridge_top10["top_k_positive_row_lift"]),
                "top10_positive_row_lift_delta_vs_dummy": float(
                    ridge_top10["top_k_positive_row_lift"] - dummy_top10["top_k_positive_row_lift"]
                ),
                "top10_utility_capture_rate": float(ridge_top10["top_k_utility_capture_rate"]),
            }
        )
    return pd.DataFrame(rows)


def _correlation(observed: np.ndarray, predicted: np.ndarray, *, method: str) -> float:
    if np.unique(predicted).size <= 1 or np.unique(observed).size <= 1:
        return 0.0
    return float(pd.Series(observed).corr(pd.Series(predicted), method=method))


def _finite_or_none(fn: Any) -> float | None:
    try:
        value = float(fn())
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _markdown_report(
    *,
    domain_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    signal_summary: pd.DataFrame,
    feature_columns: list[str],
) -> str:
    top10 = ranking_summary[ranking_summary["top_k_fraction"] == 0.10].copy()
    return "\n".join(
        [
            "# Handoff behavior learnability diagnostic",
            "",
            "## Scope",
            "",
            "- Baseline diagnostic only: `dummy_mean` and `ridge_linear`.",
            "- No main complex Decision Model was trained.",
            f"- Decision inputs: `{', '.join(feature_columns)}`.",
            f"- Target: `{TARGET_COLUMN}`.",
            "- This diagnostic uses the primary SBS-prefix dataset and separates native continuation from population-transfer actions with `handoff_required`.",
            "- Metadata and algorithm labels are not used as inputs; `handoff_required` is used only to define diagnostic domains.",
            "",
            "## Domain distribution",
            "",
            _markdown_table(domain_summary),
            "",
            "## Regression summary",
            "",
            _markdown_table(regression_summary),
            "",
            "## Top 10% validation ranking summary",
            "",
            _markdown_table(top10),
            "",
            "## Behavior signal summary",
            "",
            _markdown_table(signal_summary),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose handoff-stratified behavior-feature learnability with baseline models.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    materialized = query_root / "materialized_training_data"

    run_handoff_learnability_diagnostic(
        query_id=args.query_id,
        dataset_path=args.dataset or materialized / "decision_dataset.parquet",
        schema_path=args.schema or materialized / "decision_dataset_schema.json",
        output_dir=args.output_dir or query_root / "handoff_learnability_diagnostic",
        overwrite=args.overwrite,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
