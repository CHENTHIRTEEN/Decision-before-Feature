from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_diagnostics import GROUP_LAYERS, _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _json_default
from decision.min_support_model_sensitivity import EVALUATION_DOMAINS


BASELINE_POLICIES = (
    {
        "policy_name": "no_ela_sbs",
        "source_policy": "never_ela_sbs",
        "policy_category": "baseline",
        "call_column": None,
        "call_rate": 0.0,
    },
    {
        "policy_name": "always_ela_traditional_aas",
        "source_policy": "always_ela_traditional_aas",
        "policy_category": "baseline",
        "call_column": None,
        "call_rate": 1.0,
    },
    {
        "policy_name": "random_analysis_p50",
        "source_policy": "random_analysis",
        "policy_category": "baseline",
        "call_column": "random_analysis_run_ela",
        "call_rate": None,
    },
    {
        "policy_name": "best_observed_analysis_action",
        "source_policy": "best_observed_analysis_action",
        "policy_category": "reference_upper_bound",
        "call_column": "best_observed_analysis_run_ela",
        "call_rate": None,
    },
)


def run_ablation_comparison(
    *,
    predictions_path: Path,
    output_dir: Path,
    target_column: str,
    dataset_name: str,
    selection_reference_path: Path | None,
) -> dict[str, Any]:
    predictions = _read_predictions(predictions_path, target_column)
    baseline_source = _baseline_source_frame(predictions)
    policy_summary = _policy_summary(predictions, baseline_source, target_column)
    relative_summary = _relative_summary(policy_summary)
    best_policy_summary = _best_policy_summary(policy_summary)
    vbs_reference = _selection_reference_summary(selection_reference_path)
    conclusion = _diagnostic_conclusion(policy_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "ablation_policy_summary.parquet"
    relative_path = output_dir / "ablation_relative_summary.parquet"
    best_path = output_dir / "ablation_best_policy_by_domain.parquet"
    json_path = output_dir / "ablation_comparison_summary.json"

    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), policy_path)
    pq.write_table(pa.Table.from_pandas(relative_summary, preserve_index=False), relative_path)
    pq.write_table(pa.Table.from_pandas(best_policy_summary, preserve_index=False), best_path)

    summary = {
        "experiment": "min_support_ablation_comparison",
        "research_question": (
            "How do no-ELA/SBS, always-ELA, random analysis, and model-based Decision-before-Feature policies "
            "compare under the unchanged min-support utility-label protocol?"
        ),
        "dataset_name": dataset_name,
        "predictions": str(predictions_path),
        "target_column": target_column,
        "policies": {
            "baselines": [policy["policy_name"] for policy in BASELINE_POLICIES],
            "proposed": "decision_before_feature for each existing model_family and threshold_mode",
            "vbs_reference": "selection_reference only; paired VBS utility labels are not generated in this diagnostic",
        },
        "outputs": {
            "policy_summary": str(policy_path),
            "relative_summary": str(relative_path),
            "best_policy_by_domain": str(best_path),
            "summary": str(json_path),
        },
        "vbs_reference": vbs_reference,
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "original_utility_labels_modified": False,
            "decision_input_uses_ela_features": False,
            "formal_phase1_configs_modified": False,
        },
        "notes": [
            "no_ela_sbs uses the existing utility_never_ela_sbs baseline, whose relative utility is zero by definition.",
            "always_ela_traditional_aas uses the observed U_ELA value for every row.",
            "best_observed_analysis_action is an unattainable row-level upper bound, not a deployable policy.",
            "SBS/VBS optimizer-level final-performance comparisons are not regenerated; VBS is reported only from selection_reference metadata when supplied.",
        ],
    }
    json_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote ablation policy summary to {policy_path}")
    print(f"wrote ablation relative summary to {relative_path}")
    print(f"wrote ablation best-policy summary to {best_path}")
    print(f"wrote ablation comparison summary to {json_path}")
    return summary


def _read_predictions(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing model_sensitivity predictions: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "threshold_mode",
        "label_source",
        "decision_run_ela",
        "random_analysis_run_ela",
        "best_observed_analysis_run_ela",
        target_column,
    }
    for policy in [*BASELINE_POLICIES, {"source_policy": "decision_before_feature"}]:
        source_policy = policy["source_policy"]
        required.update(
            {
                f"utility_{source_policy}",
                f"final_performance_{source_policy}",
                f"runtime_{source_policy}",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"predictions file is missing required columns: {missing}")
    return frame


def _baseline_source_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["model_name", "threshold_mode"]
    first_key = predictions[key_columns].drop_duplicates().sort_values(key_columns).iloc[0]
    return predictions[
        (predictions["model_name"] == first_key["model_name"])
        & (predictions["threshold_mode"] == first_key["threshold_mode"])
    ].copy()


def _policy_summary(predictions: pd.DataFrame, baseline_source: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for eval_domain, label_source in EVALUATION_DOMAINS.items():
        baseline_domain = _domain_frame(baseline_source, label_source)
        for layer, group_columns in GROUP_LAYERS.items():
            rows.extend(_baseline_policy_rows(baseline_domain, eval_domain, layer, group_columns, target_column))

        for (model_name, model_family, score_semantics, threshold_mode), model_frame in predictions.groupby(
            ["model_name", "model_family", "score_semantics", "threshold_mode"],
            sort=True,
        ):
            model_domain = _domain_frame(model_frame, label_source)
            for layer, group_columns in GROUP_LAYERS.items():
                rows.extend(
                    _grouped_policy_rows(
                        frame=model_domain,
                        policy_name="decision_before_feature",
                        source_policy="decision_before_feature",
                        policy_category="proposed",
                        call_column="decision_run_ela",
                        fixed_call_rate=None,
                        eval_domain=eval_domain,
                        layer=layer,
                        group_columns=group_columns,
                        target_column=target_column,
                        model_name=str(model_name),
                        model_family=str(model_family),
                        score_semantics=str(score_semantics),
                        threshold_mode=str(threshold_mode),
                    )
                )
    return pd.DataFrame(rows)


def _domain_frame(frame: pd.DataFrame, label_source: str | None) -> pd.DataFrame:
    if label_source is None:
        return frame
    return frame[frame["label_source"] == label_source]


def _baseline_policy_rows(
    frame: pd.DataFrame,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in BASELINE_POLICIES:
        rows.extend(
            _grouped_policy_rows(
                frame=frame,
                policy_name=policy["policy_name"],
                source_policy=policy["source_policy"],
                policy_category=policy["policy_category"],
                call_column=policy["call_column"],
                fixed_call_rate=policy["call_rate"],
                eval_domain=eval_domain,
                layer=layer,
                group_columns=group_columns,
                target_column=target_column,
                model_name="",
                model_family="",
                score_semantics="",
                threshold_mode="",
            )
        )
    return rows


def _grouped_policy_rows(
    *,
    frame: pd.DataFrame,
    policy_name: str,
    source_policy: str,
    policy_category: str,
    call_column: str | None,
    fixed_call_rate: float | None,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
    model_name: str,
    model_family: str,
    score_semantics: str,
    threshold_mode: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if not group_columns:
        return [
            _policy_row(
                frame=frame,
                policy_name=policy_name,
                source_policy=source_policy,
                policy_category=policy_category,
                call_column=call_column,
                fixed_call_rate=fixed_call_rate,
                eval_domain=eval_domain,
                layer=layer,
                group={},
                target_column=target_column,
                model_name=model_name,
                model_family=model_family,
                score_semantics=score_semantics,
                threshold_mode=threshold_mode,
            )
        ]
    rows: list[dict[str, Any]] = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        rows.append(
            _policy_row(
                frame=subset,
                policy_name=policy_name,
                source_policy=source_policy,
                policy_category=policy_category,
                call_column=call_column,
                fixed_call_rate=fixed_call_rate,
                eval_domain=eval_domain,
                layer=layer,
                group=dict(zip(group_columns, group_values, strict=True)),
                target_column=target_column,
                model_name=model_name,
                model_family=model_family,
                score_semantics=score_semantics,
                threshold_mode=threshold_mode,
            )
        )
    return rows


def _policy_row(
    *,
    frame: pd.DataFrame,
    policy_name: str,
    source_policy: str,
    policy_category: str,
    call_column: str | None,
    fixed_call_rate: float | None,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
    model_name: str,
    model_family: str,
    score_semantics: str,
    threshold_mode: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    utility = frame[f"utility_{source_policy}"].to_numpy(dtype=float)
    performance = frame[f"final_performance_{source_policy}"].to_numpy(dtype=float)
    runtime = frame[f"runtime_{source_policy}"].to_numpy(dtype=float)
    run_ela = _run_ela_array(frame, call_column, fixed_call_rate)
    observed_help = observed > 0.0
    captured_positive = observed[observed_help & run_ela]
    unhelpful_call = observed[(~observed_help) & run_ela]
    positive_utility_sum = float(np.sum(observed[observed_help]))
    captured_positive_sum = float(np.sum(captured_positive))
    never_performance = frame["final_performance_never_ela_sbs"].to_numpy(dtype=float)
    always_performance = frame["final_performance_always_ela_traditional_aas"].to_numpy(dtype=float)
    return {
        "policy_name": policy_name,
        "policy_category": policy_category,
        "model_name": model_name,
        "model_family": model_family,
        "score_semantics": score_semantics,
        "threshold_mode": threshold_mode,
        "eval_domain": eval_domain,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(frame)),
        "observed_utility_gt_zero_rows": int(np.sum(observed_help)),
        "observed_utility_gt_zero_rate": float(np.mean(observed_help)),
        "ela_call_count": int(np.sum(run_ela)),
        "ela_call_rate": float(np.mean(run_ela)),
        "positive_row_capture_rate": float(np.mean(run_ela[observed_help])) if np.any(observed_help) else 0.0,
        "utility_capture_rate": captured_positive_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0,
        "captured_positive_utility_sum": captured_positive_sum,
        "unhelpful_call_count": int(np.sum((~observed_help) & run_ela)),
        "unhelpful_call_utility_sum": float(np.sum(unhelpful_call)),
        "unhelpful_call_cost_sum": float(-np.sum(unhelpful_call)),
        "utility_sum": float(np.sum(utility)),
        "utility_mean": float(np.mean(utility)),
        "utility_median": float(np.median(utility)),
        "final_performance_mean": float(np.mean(performance)),
        "final_performance_median": float(np.median(performance)),
        "final_performance_delta_vs_no_ela_mean": float(np.mean(performance - never_performance)),
        "final_performance_delta_vs_always_ela_mean": float(np.mean(performance - always_performance)),
        "runtime_mean_seconds": float(np.mean(runtime)),
        "runtime_median_seconds": float(np.median(runtime)),
    }


def _run_ela_array(frame: pd.DataFrame, call_column: str | None, fixed_call_rate: float | None) -> np.ndarray:
    if call_column is not None:
        return frame[call_column].to_numpy(dtype=bool)
    if fixed_call_rate == 1.0:
        return np.ones(len(frame), dtype=bool)
    if fixed_call_rate == 0.0:
        return np.zeros(len(frame), dtype=bool)
    raise ValueError("policy must define either call_column or fixed_call_rate")


def _relative_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    overall = policy_summary[policy_summary["layer"] == "overall"].copy()
    key_columns = ["eval_domain"]
    baselines = overall[overall["policy_category"] != "proposed"][
        [*key_columns, "policy_name", "utility_mean", "utility_sum", "ela_call_rate", "runtime_mean_seconds"]
    ].rename(
        columns={
            "policy_name": "baseline_policy",
            "utility_mean": "baseline_utility_mean",
            "utility_sum": "baseline_utility_sum",
            "ela_call_rate": "baseline_ela_call_rate",
            "runtime_mean_seconds": "baseline_runtime_mean_seconds",
        }
    )
    proposed = overall[overall["policy_category"] == "proposed"][
        [
            *key_columns,
            "model_name",
            "model_family",
            "threshold_mode",
            "utility_mean",
            "utility_sum",
            "ela_call_rate",
            "runtime_mean_seconds",
            "positive_row_capture_rate",
            "utility_capture_rate",
            "unhelpful_call_cost_sum",
        ]
    ].rename(
        columns={
            "utility_mean": "proposed_utility_mean",
            "utility_sum": "proposed_utility_sum",
            "ela_call_rate": "proposed_ela_call_rate",
            "runtime_mean_seconds": "proposed_runtime_mean_seconds",
        }
    )
    result = proposed.merge(baselines, on=key_columns, how="left")
    result["utility_mean_delta_vs_baseline"] = result["proposed_utility_mean"] - result["baseline_utility_mean"]
    result["utility_sum_delta_vs_baseline"] = result["proposed_utility_sum"] - result["baseline_utility_sum"]
    result["ela_call_rate_delta_vs_baseline"] = result["proposed_ela_call_rate"] - result["baseline_ela_call_rate"]
    result["runtime_mean_delta_vs_baseline"] = (
        result["proposed_runtime_mean_seconds"] - result["baseline_runtime_mean_seconds"]
    )
    return result


def _best_policy_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    overall = policy_summary[policy_summary["layer"] == "overall"].copy()
    sort_columns = ["eval_domain", "utility_sum", "utility_mean"]
    ranked = overall.sort_values(sort_columns, ascending=[True, False, False]).copy()
    return ranked.groupby("eval_domain", as_index=False).head(8).reset_index(drop=True)


def _selection_reference_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided"}
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    frame = pq.read_table(path).to_pandas()
    if frame.empty:
        return {"status": "empty", "path": str(path)}
    result = {
        "status": "available",
        "path": str(path),
        "rows": int(len(frame)),
        "selected_matches_sbs_rate": float((frame["selected_algorithm"] == frame["sbs_algorithm"]).mean())
        if "sbs_algorithm" in frame.columns
        else None,
        "selected_matches_vbs_rate": float((frame["selected_algorithm"] == frame["vbs_algorithm"]).mean())
        if "vbs_algorithm" in frame.columns
        else None,
    }
    return result


def _diagnostic_conclusion(policy_summary: pd.DataFrame) -> dict[str, Any]:
    overall = policy_summary[
        (policy_summary["layer"] == "overall")
        & (policy_summary["eval_domain"].isin(["all_validation", "changed_algorithm_validation", "same_algorithm_reference"]))
    ].copy()
    best_by_domain = (
        overall.sort_values(["eval_domain", "utility_sum"], ascending=[True, False])
        .groupby("eval_domain", as_index=False)
        .head(1)
    )
    changed = overall[overall["eval_domain"] == "changed_algorithm_validation"]
    proposed_changed = changed[changed["policy_category"] == "proposed"].sort_values("utility_sum", ascending=False)
    baseline_changed = changed[changed["policy_category"] != "proposed"].sort_values("utility_sum", ascending=False)
    return {
        "best_policy_by_eval_domain": best_by_domain[
            ["eval_domain", "policy_name", "model_family", "threshold_mode", "utility_sum", "utility_mean", "ela_call_rate"]
        ].to_dict(orient="records"),
        "best_proposed_on_changed_algorithm_validation": (
            proposed_changed[
                [
                    "policy_name",
                    "model_family",
                    "threshold_mode",
                    "utility_sum",
                    "utility_mean",
                    "ela_call_rate",
                    "positive_row_capture_rate",
                    "utility_capture_rate",
                    "unhelpful_call_cost_sum",
                ]
            ]
            .head(3)
            .to_dict(orient="records")
        ),
        "best_baseline_on_changed_algorithm_validation": (
            baseline_changed[
                ["policy_name", "utility_sum", "utility_mean", "ela_call_rate", "positive_row_capture_rate", "utility_capture_rate"]
            ]
            .head(3)
            .to_dict(orient="records")
        ),
        "interpretation": (
            "The deployable Decision-before-Feature policies should be judged against no_ela_sbs, "
            "always_ela_traditional_aas, and random_analysis; best_observed_analysis_action is included only as an "
            "unattainable row-level reference. Positive utility means improvement relative to no_ela_sbs under the "
            "existing U_ELA definition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare min-support Decision-before-Feature policies with baseline ablations.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/decision/min_support/fe_transition_model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--dataset-name", default="fe_transition_model_sensitivity")
    parser.add_argument("--selection-reference", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/ablation_comparison"),
    )
    args = parser.parse_args()
    run_ablation_comparison(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        target_column=args.target_column,
        dataset_name=args.dataset_name,
        selection_reference_path=args.selection_reference,
    )


if __name__ == "__main__":
    main()
