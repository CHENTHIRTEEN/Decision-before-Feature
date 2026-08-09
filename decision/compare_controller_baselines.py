from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PREDICTIONS_PATH = Path(
    "results/decision/phase1_refined_sampling/feature_group_ablation/"
    "primary_with_maturity/validation_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = Path("results/decision/phase1_refined_sampling/controller_baseline_comparison")
DEFAULT_MODEL_NAME = "ridge_regression"
DEFAULT_THRESHOLD_MODE = "train_utility"
DEFAULT_EXPECTED_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_ela_lamT_1"
GROUP_LAYERS = {
    "overall": [],
    "label_source": ["label_source"],
    "FE_ratio": ["FE_ratio"],
    "dimension": ["dimension"],
    "prefix_algorithm": ["prefix_algorithm"],
    "family": ["family"],
}


def compare_controller_baselines(
    *,
    predictions_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    random_ela_probability: float,
    random_repetitions: int,
    random_seed: int,
    expected_split: str,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    predictions = _read_predictions(predictions_path, model_name, threshold_mode, expected_split)
    policies = _policy_frames(
        predictions=predictions,
        threshold_mode=threshold_mode,
        random_ela_probability=random_ela_probability,
        random_seed=random_seed,
    )
    policy_summary = _policy_summary(policies)
    relative_summary = _relative_summary(policy_summary)
    best_policy_summary = _best_policy_summary(policy_summary)
    random_repetition_summary = _random_repetition_summary(
        predictions=predictions,
        random_ela_probability=random_ela_probability,
        random_seed=random_seed,
        random_repetitions=random_repetitions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(policy_summary, output_dir / "controller_baseline_policy_summary")
    _write_frame(relative_summary, output_dir / "controller_baseline_relative_summary")
    _write_frame(best_policy_summary, output_dir / "controller_baseline_best_policy")
    _write_frame(random_repetition_summary, output_dir / "controller_baseline_random_repetition_summary")

    summary = {
        "experiment": "phase1_refined_sampling_controller_baseline_comparison",
        "research_question": (
            "How does the current Decision-before-Feature controller compare with SBS/no-ELA, "
            "always-ELA, and random-ELA under the existing validation U_ELA labels?"
        ),
        "predictions_path": str(predictions_path),
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "expected_split": expected_split,
        "random_ela_probability": random_ela_probability,
        "random_repetitions": random_repetitions,
        "random_seed": random_seed,
        "rows": int(len(predictions)),
        "policies": sorted(policies["policy_name"].unique().tolist()),
        "outputs": {
            "policy_summary": str(output_dir / "controller_baseline_policy_summary.parquet"),
            "relative_summary": str(output_dir / "controller_baseline_relative_summary.parquet"),
            "best_policy": str(output_dir / "controller_baseline_best_policy.parquet"),
            "random_repetition_summary": str(output_dir / "controller_baseline_random_repetition_summary.parquet"),
            "report": str(output_dir / "controller_baseline_comparison_report.md"),
            "summary": str(output_dir / "controller_baseline_comparison_summary.json"),
        },
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "expected_split_rows_used_for_controller_fit_or_threshold": False,
            "ela_features_used_as_decision_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
        },
        "scope_notes": [
            "The comparison is expressed in the current U_ELA label space.",
            "sbs_skip_reference and no_ela have identical utility under the current phase1 materialized dataset because "
            "default_algorithm is CMA-ES, matching the selection_reference SBS algorithm.",
            "A distinct optimizer-level SBS performance comparison requires materializing P_sbs or final-performance "
            "columns; those fields are not present in the current controller prediction table.",
        ],
    }
    summary_path = output_dir / "controller_baseline_comparison_summary.json"
    report_path = output_dir / "controller_baseline_comparison_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            policy_summary=policy_summary,
            relative_summary=relative_summary,
            best_policy_summary=best_policy_summary,
            random_repetition_summary=random_repetition_summary,
            model_name=model_name,
            threshold_mode=threshold_mode,
            random_ela_probability=random_ela_probability,
            random_repetitions=random_repetitions,
            expected_split=expected_split,
        ),
        encoding="utf-8",
    )
    print(f"wrote controller baseline policy summary to {output_dir / 'controller_baseline_policy_summary.parquet'}")
    print(f"wrote controller baseline relative summary to {output_dir / 'controller_baseline_relative_summary.parquet'}")
    print(f"wrote controller baseline report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "controller_baseline_policy_summary.csv",
        output_dir / "controller_baseline_policy_summary.parquet",
        output_dir / "controller_baseline_relative_summary.csv",
        output_dir / "controller_baseline_relative_summary.parquet",
        output_dir / "controller_baseline_best_policy.csv",
        output_dir / "controller_baseline_best_policy.parquet",
        output_dir / "controller_baseline_random_repetition_summary.csv",
        output_dir / "controller_baseline_random_repetition_summary.parquet",
        output_dir / "controller_baseline_comparison_report.md",
        output_dir / "controller_baseline_comparison_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"controller baseline comparison outputs already exist; pass --overwrite: {existing[0]}")


def _read_predictions(path: Path, model_name: str, threshold_mode: str, expected_split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
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
        TARGET_COLUMN,
        f"decision_run_ela_{threshold_mode}",
        f"decision_utility_{threshold_mode}",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction table is missing required columns: {missing}")
    frame = frame[frame["model_name"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"no prediction rows for model_name={model_name}")
    if frame["split"].nunique() != 1 or str(frame["split"].iloc[0]) != expected_split:
        raise ValueError(f"controller baseline comparison expects {expected_split} prediction rows")
    if not np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError(f"{TARGET_COLUMN} contains non-finite values")
    return frame.reset_index(drop=True)


def _policy_frames(
    *,
    predictions: pd.DataFrame,
    threshold_mode: str,
    random_ela_probability: float,
    random_seed: int,
) -> pd.DataFrame:
    if random_ela_probability < 0.0 or random_ela_probability > 1.0:
        raise ValueError("random_ela_probability must be in [0, 1]")
    observed = predictions[TARGET_COLUMN].to_numpy(dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), 20260809, len(predictions)]))
    random_call = rng.random(len(predictions)) < random_ela_probability
    current_call = predictions[f"decision_run_ela_{threshold_mode}"].to_numpy(dtype=bool)
    policy_specs = (
        ("sbs_skip_reference", "baseline", np.zeros(len(predictions), dtype=bool)),
        ("no_ela", "baseline", np.zeros(len(predictions), dtype=bool)),
        ("always_ela", "baseline", np.ones(len(predictions), dtype=bool)),
        (f"random_ela_p{int(round(random_ela_probability * 100)):02d}", "baseline", random_call),
        ("current_controller", "controller", current_call),
    )
    frames = []
    metadata = predictions[
        [
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
            TARGET_COLUMN,
        ]
    ].copy()
    for policy_name, policy_category, call in policy_specs:
        frame = metadata.copy()
        frame.insert(0, "policy_name", policy_name)
        frame.insert(1, "policy_category", policy_category)
        frame["run_ela"] = call
        frame["policy_utility"] = np.where(call, observed, 0.0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _policy_summary(policies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, group_columns in GROUP_LAYERS.items():
        if not group_columns:
            for (_, _), frame in policies.groupby(["policy_name", "policy_category"], sort=True):
                rows.append(_policy_row(frame, layer=layer, group={}))
            continue
        for (_, _), policy_frame in policies.groupby(["policy_name", "policy_category"], sort=True):
            for group_values, frame in policy_frame.groupby(group_columns, dropna=False, sort=True):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                rows.append(_policy_row(frame, layer=layer, group=dict(zip(group_columns, group_values, strict=True))))
    result = pd.DataFrame(rows)
    sort_columns = ["layer", "group", "policy_category", "policy_name"]
    return result.sort_values(sort_columns).reset_index(drop=True)


def _policy_row(frame: pd.DataFrame, *, layer: str, group: dict[str, Any]) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    utility = frame["policy_utility"].to_numpy(dtype=float)
    calls = frame["run_ela"].to_numpy(dtype=bool)
    positive = observed > 0.0
    captured_positive = positive & calls
    unhelpful_calls = (~positive) & calls
    positive_utility_sum = float(np.sum(observed[positive]))
    captured_positive_utility_sum = float(np.sum(observed[captured_positive]))
    call_rows = int(np.sum(calls))
    return {
        "policy_name": str(frame["policy_name"].iloc[0]),
        "policy_category": str(frame["policy_category"].iloc[0]),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "prefix_algorithm": group.get("prefix_algorithm"),
        "label_source": group.get("label_source"),
        "rows": int(len(frame)),
        "observed_utility_gt_zero_rows": int(np.sum(positive)),
        "observed_utility_gt_zero_rate": float(np.mean(positive)),
        "ela_call_rows": call_rows,
        "ela_call_rate": float(np.mean(calls)),
        "mean_observed_utility_under_calls": float(np.mean(observed[calls])) if call_rows else 0.0,
        "positive_row_capture_rate": float(np.sum(captured_positive) / max(np.sum(positive), 1)),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_rows, 1)),
        "unhelpful_call_rows": int(np.sum(unhelpful_calls)),
        "unhelpful_call_rate_within_calls": float(np.sum(unhelpful_calls) / max(call_rows, 1)),
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful_calls])),
        "utility_sum": float(np.sum(utility)),
        "utility_mean": float(np.mean(utility)),
        "utility_median": float(np.median(utility)),
        "best_observed_analysis_action_mean_utility": float(np.mean(np.maximum(observed, 0.0))),
    }


def _relative_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    overall = policy_summary[policy_summary["layer"] == "overall"].copy()
    baselines = overall[overall["policy_category"] == "baseline"][
        [
            "policy_name",
            "utility_mean",
            "utility_sum",
            "ela_call_rate",
            "utility_capture_rate",
            "precision_u_gt_zero_under_calls",
            "unhelpful_call_cost_sum",
        ]
    ].rename(
        columns={
            "policy_name": "baseline_policy",
            "utility_mean": "baseline_utility_mean",
            "utility_sum": "baseline_utility_sum",
            "ela_call_rate": "baseline_ela_call_rate",
            "utility_capture_rate": "baseline_utility_capture_rate",
            "precision_u_gt_zero_under_calls": "baseline_precision_u_gt_zero_under_calls",
            "unhelpful_call_cost_sum": "baseline_unhelpful_call_cost_sum",
        }
    )
    controller = overall[overall["policy_name"] == "current_controller"][
        [
            "utility_mean",
            "utility_sum",
            "ela_call_rate",
            "utility_capture_rate",
            "precision_u_gt_zero_under_calls",
            "unhelpful_call_cost_sum",
        ]
    ].rename(
        columns={
            "utility_mean": "controller_utility_mean",
            "utility_sum": "controller_utility_sum",
            "ela_call_rate": "controller_ela_call_rate",
            "utility_capture_rate": "controller_utility_capture_rate",
            "precision_u_gt_zero_under_calls": "controller_precision_u_gt_zero_under_calls",
            "unhelpful_call_cost_sum": "controller_unhelpful_call_cost_sum",
        }
    )
    if controller.empty:
        raise ValueError("current_controller row missing from policy summary")
    result = baselines.copy()
    for column, value in controller.iloc[0].items():
        result[column] = value
    result["utility_mean_delta_vs_baseline"] = result["controller_utility_mean"] - result["baseline_utility_mean"]
    result["utility_sum_delta_vs_baseline"] = result["controller_utility_sum"] - result["baseline_utility_sum"]
    result["ela_call_rate_delta_vs_baseline"] = result["controller_ela_call_rate"] - result["baseline_ela_call_rate"]
    return result.reset_index(drop=True)


def _random_repetition_summary(
    *,
    predictions: pd.DataFrame,
    random_ela_probability: float,
    random_seed: int,
    random_repetitions: int,
) -> pd.DataFrame:
    if random_repetitions <= 0:
        raise ValueError("random_repetitions must be positive")
    observed = predictions[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    positive_utility_sum = float(np.sum(observed[positive]))
    rows = []
    for repetition in range(random_repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(random_seed), 20260810, len(predictions), int(repetition)])
        )
        call = rng.random(len(predictions)) < random_ela_probability
        captured_positive = positive & call
        utility = np.where(call, observed, 0.0)
        call_rows = int(np.sum(call))
        rows.append(
            {
                "repetition": repetition,
                "rows": int(len(predictions)),
                "ela_call_rate": float(np.mean(call)),
                "utility_mean": float(np.mean(utility)),
                "utility_sum": float(np.sum(utility)),
                "utility_capture_rate": (
                    float(np.sum(observed[captured_positive]) / positive_utility_sum)
                    if positive_utility_sum > 0.0
                    else 0.0
                ),
                "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_rows, 1)),
            }
        )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for metric in (
        "ela_call_rate",
        "utility_mean",
        "utility_sum",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
    ):
        values = raw[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "policy_name": f"random_ela_p{int(round(random_ela_probability * 100)):02d}",
                "metric": metric,
                "repetitions": random_repetitions,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if random_repetitions > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def _best_policy_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    overall = policy_summary[policy_summary["layer"] == "overall"].copy()
    return overall.sort_values(["utility_mean", "utility_sum"], ascending=[False, False]).reset_index(drop=True)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _markdown_report(
    *,
    policy_summary: pd.DataFrame,
    relative_summary: pd.DataFrame,
    best_policy_summary: pd.DataFrame,
    random_repetition_summary: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    random_ela_probability: float,
    random_repetitions: int,
    expected_split: str,
) -> str:
    overall_columns = [
        "policy_name",
        "policy_category",
        "rows",
        "utility_mean",
        "utility_sum",
        "ela_call_rate",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "unhelpful_call_cost_sum",
    ]
    relative_columns = [
        "baseline_policy",
        "baseline_utility_mean",
        "controller_utility_mean",
        "utility_mean_delta_vs_baseline",
        "baseline_ela_call_rate",
        "controller_ela_call_rate",
        "ela_call_rate_delta_vs_baseline",
    ]
    label_columns = [
        "policy_name",
        "group",
        "rows",
        "utility_mean",
        "utility_sum",
        "ela_call_rate",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
    ]
    overall = best_policy_summary[overall_columns]
    label_source = policy_summary[policy_summary["layer"] == "label_source"][label_columns].sort_values(
        ["group", "utility_mean"], ascending=[True, False]
    )
    lines = [
        "# Controller Baseline Comparison",
        "",
        "## 摘要",
        "",
        f"- 当前 controller：`{model_name}`，阈值口径为 `{threshold_mode}`。",
        f"- Random-ELA baseline 使用 `p={random_ela_probability:.3f}`，随机流由显式 `SeedSequence` 生成。",
        f"- Random-ELA 额外报告 `{random_repetitions}` 个独立随机流的均值、标准差和范围。",
        f"- 指标在 `{expected_split}` 上计算，口径是当前 materialized dataset 中的 `U_ELA`。",
        "- `sbs_skip_reference` 与 `no_ela` 在当前 phase1 表中数值相同：都表示不调用 ELA，相对 utility 为 0。",
        "",
        "## Overall Policies",
        "",
        _markdown_table(overall),
        "",
        "## Controller Relative To Baselines",
        "",
        _markdown_table(relative_summary[relative_columns]),
        "",
        "## Random-ELA Repetition Summary",
        "",
        _markdown_table(random_repetition_summary),
        "",
        "## Label-Source Breakdown",
        "",
        _markdown_table(label_source),
        "",
        "## 解释",
        "",
        "当 controller 的 mean utility 大于 0 时，它优于 `sbs_skip_reference` 和 `no_ela`。"
        "与 `always_ela` 的比较用于判断选择性调用 ELA 是否减少了无效调用。"
        "Random-ELA 用于检查当前结果是否只是由调用频率变化带来的随机效应。",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current phase1 controller with ELA-call baselines.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--random-ela-probability", type=float, default=0.5)
    parser.add_argument("--random-repetitions", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--expected-split", default=DEFAULT_EXPECTED_SPLIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compare_controller_baselines(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        random_ela_probability=args.random_ela_probability,
        random_repetitions=args.random_repetitions,
        random_seed=args.random_seed,
        expected_split=args.expected_split,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
