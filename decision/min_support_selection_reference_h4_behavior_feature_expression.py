from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default
from decision.min_support_f024_behavior_feature_candidates import CANDIDATE_FEATURE_COLUMNS, _candidate_features_from_trajectories
from decision.min_support_f024_behavior_separability import _annotate, _load_config, _target_holdout_pairs, _target_problem_ids


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "existing_behavior": tuple(BEHAVIOR_FEATURE_COLUMNS),
    "candidate_behavior": tuple(CANDIDATE_FEATURE_COLUMNS),
    "existing_plus_candidate": tuple(BEHAVIOR_FEATURE_COLUMNS) + tuple(CANDIDATE_FEATURE_COLUMNS),
}


def run_h4_behavior_feature_expression(
    *,
    validation_labels_path: Path,
    extension_train_labels_path: Path,
    extension_config_path: Path,
    validation_trajectory_root: Path,
    extension_trajectory_root: Path,
    output_dir: Path,
    target_column: str,
    random_seed: int,
) -> dict[str, Any]:
    _check_target(target_column)
    config = _load_config(extension_config_path)
    target_problem_ids = _target_problem_ids(config)
    target_holdout_pairs = _target_holdout_pairs(config)

    validation_labels = _annotate(
        pd.read_parquet(validation_labels_path),
        target_problem_ids=target_problem_ids,
        target_holdout_pairs=target_holdout_pairs,
        data_split="validation",
    )
    extension_labels = _annotate(
        pd.read_parquet(extension_train_labels_path),
        target_problem_ids=target_problem_ids,
        target_holdout_pairs=target_holdout_pairs,
        data_split="extension_train",
    )

    validation_candidates = _candidate_features_from_trajectories(_trajectory_paths(validation_trajectory_root))
    extension_candidates = _candidate_features_from_trajectories(_trajectory_paths(extension_trajectory_root))
    validation = _merge_candidate_features(validation_labels, validation_candidates)
    extension_train = _merge_candidate_features(extension_labels, extension_candidates)

    train_frame = _training_frame(extension_train)
    eval_frame = _evaluation_frame(validation)
    prediction_rows, model_summary = _fit_predict_feature_sets(
        train_frame=train_frame,
        eval_frame=eval_frame,
        target_column=target_column,
        random_seed=random_seed,
    )
    metric_summary = _metric_summary(prediction_rows, target_column)
    score_rank_summary = _score_rank_summary(prediction_rows, target_column)
    feature_completeness = _feature_completeness_table(train_frame, eval_frame)
    conclusion = _diagnostic_conclusion(metric_summary, model_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "h4_feature_set_prediction_rows.parquet"
    metric_summary_path = output_dir / "h4_feature_set_metric_summary.parquet"
    score_rank_path = output_dir / "h4_feature_set_score_rank_summary.parquet"
    model_summary_path = output_dir / "h4_feature_set_model_summary.parquet"
    feature_completeness_path = output_dir / "h4_feature_completeness.parquet"
    summary_path = output_dir / "h4_behavior_feature_expression_summary.json"
    report_path = output_dir / "h4_behavior_feature_expression_report.md"

    pq.write_table(pa.Table.from_pandas(prediction_rows, preserve_index=False), prediction_path)
    pq.write_table(pa.Table.from_pandas(metric_summary, preserve_index=False), metric_summary_path)
    pq.write_table(pa.Table.from_pandas(score_rank_summary, preserve_index=False), score_rank_path)
    pq.write_table(pa.Table.from_pandas(model_summary, preserve_index=False), model_summary_path)
    pq.write_table(pa.Table.from_pandas(feature_completeness, preserve_index=False), feature_completeness_path)

    summary = {
        "experiment": "selection_reference_h4_behavior_feature_expression_diagnostic",
        "research_question": (
            "Do candidate algorithm-agnostic population/fitness behavior features improve f024 target-holdout "
            "U_ELA>0 capture without increasing non-target validation calls?"
        ),
        "target_column": target_column,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "domains": {
            "extension_train_f024_target_changed_late": "diagnostic training rows from f024 followup extension",
            "validation_f024_target_holdout_changed_late": "held-out target f024 problem/seed rows, changed_algorithm, FE_ratio >= 0.5",
            "validation_non_target_changed_late": "validation changed_algorithm late-stage rows outside the target holdout set",
        },
        "inputs": {
            "validation_labels": str(validation_labels_path),
            "extension_train_labels": str(extension_train_labels_path),
            "extension_config": str(extension_config_path),
            "validation_trajectory_root": str(validation_trajectory_root),
            "extension_trajectory_root": str(extension_trajectory_root),
        },
        "outputs": {
            "prediction_rows": str(prediction_path),
            "metric_summary": str(metric_summary_path),
            "score_rank_summary": str(score_rank_path),
            "model_summary": str(model_summary_path),
            "feature_completeness": str(feature_completeness_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
            "formal_feature_extractor_modified": False,
            "utility_labels_regenerated": False,
            "candidate_features_written_to_formal_behavior_files": False,
            "diagnostic_models_trained": True,
            "diagnostic_models_written_as_formal_phase1": False,
            "ela_features_used_as_decision_input": False,
            "function_id_used_as_decision_input": False,
            "problem_id_used_only_for_domain_definition": True,
            "prefix_algorithm_used_only_for_trajectory_label_alignment": True,
        },
        "notes": [
            "Candidate behavior features are computed from existing trajectory population and fitness values.",
            "Thresholds are selected only on the diagnostic extension_train f024 rows.",
            "P_ELA and U_ELA are read from existing utility labels for rows selected by the diagnostic decision rule.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown_report(summary, metric_summary, score_rank_summary, model_summary), encoding="utf-8")

    print(f"wrote H4 feature-set prediction rows to {prediction_path}")
    print(f"wrote H4 feature-set metric summary to {metric_summary_path}")
    print(f"wrote H4 feature-set score-rank summary to {score_rank_path}")
    print(f"wrote H4 model summary to {model_summary_path}")
    print(f"wrote H4 feature completeness to {feature_completeness_path}")
    print(f"wrote H4 summary to {summary_path}")
    print(f"wrote H4 report to {report_path}")
    return summary


def _trajectory_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("bbob_f*/dimension_*/trajectories.parquet"))
    if not paths:
        raise FileNotFoundError(f"no trajectory shards under {root}")
    return paths


def _merge_candidate_features(labels: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
    merged = labels.merge(candidates[key_columns + list(CANDIDATE_FEATURE_COLUMNS)], on=key_columns, how="left", validate="many_to_one")
    return merged


def _training_frame(extension_train: pd.DataFrame) -> pd.DataFrame:
    mask = (
        extension_train["is_target_problem"].to_numpy(dtype=bool)
        & (extension_train["label_source"].to_numpy() == "changed_algorithm")
        & (extension_train["FE_ratio"].to_numpy(dtype=float) >= 0.5)
    )
    return extension_train.loc[mask].copy().assign(evaluation_domain="extension_train_f024_target_changed_late")


def _evaluation_frame(validation: pd.DataFrame) -> pd.DataFrame:
    changed_late = validation[
        (validation["label_source"] == "changed_algorithm")
        & (validation["FE_ratio"].astype(float) >= 0.5)
    ].copy()
    target_holdout = changed_late["is_target_problem"].to_numpy(dtype=bool) & changed_late["is_target_holdout_seed"].to_numpy(dtype=bool)
    changed_late["evaluation_domain"] = np.where(
        target_holdout,
        "validation_f024_target_holdout_changed_late",
        "validation_non_target_changed_late",
    )
    return changed_late


def _fit_predict_feature_sets(
    *,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target_column: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    model_rows = []
    for feature_set_name, feature_columns in FEATURE_SETS.items():
        model, model_status = _fit_classifier(train_frame, list(feature_columns), target_column, random_seed)
        train_scores = _predict_scores(model, model_status, train_frame, list(feature_columns))
        threshold, threshold_metrics = _select_threshold(train_scores, train_frame[target_column].to_numpy(dtype=float))

        for domain_frame in (train_frame, eval_frame):
            frame = domain_frame.copy()
            frame["feature_set"] = feature_set_name
            frame["model_status"] = model_status
            frame["decision_score"] = _predict_scores(model, model_status, frame, list(feature_columns))
            frame["decision_threshold"] = float(threshold)
            frame["decision_run_ela"] = frame["decision_score"] >= float(threshold)
            frame["utility_gt_zero"] = frame[target_column] > 0.0
            prediction_frames.append(frame)

        model_rows.append(
            {
                "feature_set": feature_set_name,
                "model_status": model_status,
                "feature_count": int(len(feature_columns)),
                "train_rows": int(len(train_frame)),
                "train_positive_rows": int((train_frame[target_column] > 0.0).sum()),
                "threshold": float(threshold),
                **{f"train_threshold_{key}": value for key, value in threshold_metrics.items()},
            }
        )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(model_rows)


def _fit_classifier(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_seed: int,
) -> tuple[Pipeline | float, str]:
    available = frame[list(feature_columns)].notna().all(axis=1)
    train = frame.loc[available].copy()
    y = (train[target_column] > 0.0).to_numpy(dtype=bool)
    if len(train) == 0:
        return 0.0, "fallback_no_complete_rows"
    if len(np.unique(y)) < 2:
        return float(np.mean(y)), "constant_single_class"
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_seed,
                    class_weight="balanced",
                    min_samples_leaf=1,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[feature_columns], y)
    return model, "random_forest_classifier"


def _predict_scores(model: Pipeline | float, model_status: str, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if len(frame) == 0:
        return np.asarray([], dtype=float)
    complete = frame[feature_columns].notna().all(axis=1).to_numpy(dtype=bool)
    scores = np.full(len(frame), np.nan, dtype=float)
    if model_status in {"fallback_no_complete_rows", "constant_single_class"}:
        scores[complete] = float(model)
        return scores
    if not isinstance(model, Pipeline):
        scores[complete] = float(model)
        return scores
    probabilities = model.predict_proba(frame.loc[complete, feature_columns])
    classes = model.named_steps["classifier"].classes_
    positive_index = int(np.where(classes == True)[0][0])
    scores[complete] = probabilities[:, positive_index]
    return scores


def _select_threshold(scores: np.ndarray, observed: np.ndarray) -> tuple[float, dict[str, Any]]:
    finite = np.isfinite(scores)
    if not np.any(finite):
        return float("inf"), _classification_metrics(np.zeros_like(observed, dtype=bool), observed)
    candidates = np.unique(scores[finite])
    best_threshold = float(candidates[0])
    best_metrics = None
    for threshold in candidates:
        calls = scores >= threshold
        metrics = _classification_metrics(calls, observed)
        if best_metrics is None or _threshold_sort_key(metrics) > _threshold_sort_key(best_metrics):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = _classification_metrics(np.zeros_like(observed, dtype=bool), observed)
    return best_threshold, best_metrics


def _metric_summary(prediction_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for (feature_set, domain), group in prediction_rows.groupby(["feature_set", "evaluation_domain"], sort=True):
        rows.extend(_policy_metric_rows(feature_set, domain, "overall", group, target_column))
        if domain == "validation_non_target_changed_late":
            for family, family_rows in group.groupby("family", sort=True):
                rows.extend(_policy_metric_rows(feature_set, domain, f"family={family}", family_rows, target_column))
    return pd.DataFrame(rows)


def _policy_metric_rows(
    feature_set: str,
    evaluation_domain: str,
    group_label: str,
    frame: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    return [
        _metric_summary_row(
            feature_set,
            evaluation_domain,
            group_label,
            "threshold_selected_on_extension",
            frame,
            frame["decision_run_ela"].fillna(False).to_numpy(dtype=bool),
            target_column,
        ),
        _metric_summary_row(
            feature_set,
            evaluation_domain,
            group_label,
            "top20pct_score",
            frame,
            _top_fraction_calls(frame, fraction=0.20),
            target_column,
        ),
    ]


def _metric_summary_row(
    feature_set: str,
    evaluation_domain: str,
    group_label: str,
    policy_mode: str,
    frame: pd.DataFrame,
    calls: np.ndarray,
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    labels = observed > 0.0
    metrics = _classification_metrics(calls, observed)
    called = frame[calls]
    return {
        "feature_set": feature_set,
        "evaluation_domain": evaluation_domain,
        "group": group_label,
        "policy_mode": policy_mode,
        "rows": int(len(frame)),
        "positive_rows": int(labels.sum()),
        "complete_feature_rows": int(np.isfinite(frame["decision_score"].to_numpy(dtype=float)).sum()),
        "decision_call_rows": int(calls.sum()),
        "decision_call_rate": float(calls.mean()) if len(frame) else 0.0,
        "positive_capture_rows": metrics["true_positive_rows"],
        "u_ela_gt_zero_capture_rate": metrics["recall"],
        "precision": metrics["precision"],
        "false_call_rows": metrics["false_positive_rows"],
        "missed_positive_rows": metrics["false_negative_rows"],
        "called_p_ela_mean": float(called["p_ela"].mean()) if len(called) else None,
        "called_u_ela_mean": float(called[target_column].mean()) if len(called) else None,
        "decision_policy_u_ela_mean": float(np.mean(np.where(calls, observed, 0.0))) if len(frame) else None,
        "decision_policy_u_ela_sum": float(np.sum(np.where(calls, observed, 0.0))) if len(frame) else 0.0,
        "original_p_ela_mean": float(frame["p_ela"].mean()) if len(frame) else None,
        "original_u_ela_mean": float(frame[target_column].mean()) if len(frame) else None,
        "selected_algorithm_counts": _value_counts_string(frame["selected_algorithm"]),
        "call_selected_algorithm_counts": _value_counts_string(called["selected_algorithm"]) if len(called) else "",
    }


def _top_fraction_calls(frame: pd.DataFrame, *, fraction: float) -> np.ndarray:
    calls = np.zeros(len(frame), dtype=bool)
    if frame.empty:
        return calls
    scores = frame["decision_score"].to_numpy(dtype=float)
    finite_positions = np.where(np.isfinite(scores))[0]
    if finite_positions.size == 0:
        return calls
    k = max(1, int(np.ceil(len(frame) * float(fraction))))
    ordered = finite_positions[np.argsort(scores[finite_positions])[::-1]]
    calls[ordered[: min(k, len(ordered))]] = True
    return calls


def _score_rank_summary(prediction_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for (feature_set, domain), group in prediction_rows.groupby(["feature_set", "evaluation_domain"], sort=True):
        ranked = group.sort_values("decision_score", ascending=False).copy()
        labels = ranked[target_column].to_numpy(dtype=float) > 0.0
        row = {
            "feature_set": feature_set,
            "evaluation_domain": domain,
            "rows": int(len(ranked)),
            "positive_rows": int(labels.sum()),
            "positive_mean_score": float(ranked.loc[labels, "decision_score"].mean()) if labels.any() else None,
            "non_positive_mean_score": float(ranked.loc[~labels, "decision_score"].mean()) if (~labels).any() else None,
        }
        for fraction in (0.05, 0.10, 0.20):
            k = max(1, int(np.ceil(len(ranked) * fraction)))
            top = np.zeros(len(ranked), dtype=bool)
            top[:k] = True
            row[f"top_{int(fraction * 100)}pct_k"] = int(k)
            row[f"top_{int(fraction * 100)}pct_positive_capture_rows"] = int((top & labels).sum())
            row[f"top_{int(fraction * 100)}pct_positive_capture_rate"] = float((top & labels).sum() / labels.sum()) if labels.sum() else 0.0
            row[f"top_{int(fraction * 100)}pct_precision"] = float((top & labels).sum() / top.sum()) if top.sum() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_completeness_table(train_frame: pd.DataFrame, eval_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frames = [train_frame, eval_frame]
    combined = pd.concat(frames, ignore_index=True)
    for feature_set, columns in FEATURE_SETS.items():
        for domain, group in combined.groupby("evaluation_domain", sort=True):
            complete = group[list(columns)].notna().all(axis=1)
            rows.append(
                {
                    "feature_set": feature_set,
                    "evaluation_domain": domain,
                    "rows": int(len(group)),
                    "complete_rows": int(complete.sum()),
                    "complete_rate": float(complete.mean()) if len(group) else 0.0,
                    "feature_count": int(len(columns)),
                }
            )
    return pd.DataFrame(rows)


def _classification_metrics(calls: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    labels = observed > 0.0
    finite_calls = np.asarray(calls, dtype=bool)
    tp = int(np.sum(finite_calls & labels))
    fp = int(np.sum(finite_calls & ~labels))
    fn = int(np.sum(~finite_calls & labels))
    tn = int(np.sum(~finite_calls & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    policy_u = np.where(finite_calls, observed, 0.0)
    positive_sum = float(np.sum(observed[labels]))
    captured_positive_sum = float(np.sum(observed[finite_calls & labels]))
    return {
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "false_negative_rows": fn,
        "true_negative_rows": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(observed)) if len(observed) else 0.0,
        "captured_positive_utility_sum": captured_positive_sum,
        "positive_utility_capture_rate": captured_positive_sum / positive_sum if positive_sum > 0.0 else 0.0,
        "decision_policy_u_ela_mean": float(np.mean(policy_u)) if len(policy_u) else None,
        "decision_policy_u_ela_sum": float(np.sum(policy_u)) if len(policy_u) else 0.0,
    }


def _threshold_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["f1"]),
        float(metrics["decision_policy_u_ela_mean"] or 0.0),
        float(metrics["positive_utility_capture_rate"]),
        -float(metrics["false_positive_rows"]),
    )


def _diagnostic_conclusion(metric_summary: pd.DataFrame, model_summary: pd.DataFrame) -> dict[str, Any]:
    target = metric_summary[
        (metric_summary["evaluation_domain"] == "validation_f024_target_holdout_changed_late")
        & (metric_summary["group"] == "overall")
        & (metric_summary["policy_mode"] == "threshold_selected_on_extension")
    ].set_index("feature_set")
    non_target = metric_summary[
        (metric_summary["evaluation_domain"] == "validation_non_target_changed_late")
        & (metric_summary["group"] == "overall")
        & (metric_summary["policy_mode"] == "threshold_selected_on_extension")
    ].set_index("feature_set")
    target_top20 = metric_summary[
        (metric_summary["evaluation_domain"] == "validation_f024_target_holdout_changed_late")
        & (metric_summary["group"] == "overall")
        & (metric_summary["policy_mode"] == "top20pct_score")
    ].set_index("feature_set")
    return {
        "target_holdout_capture_rate": {
            feature_set: float(target.loc[feature_set, "u_ela_gt_zero_capture_rate"]) for feature_set in target.index
        },
        "target_holdout_top20_capture_rate": {
            feature_set: float(target_top20.loc[feature_set, "u_ela_gt_zero_capture_rate"]) for feature_set in target_top20.index
        },
        "target_holdout_top20_precision": {
            feature_set: float(target_top20.loc[feature_set, "precision"]) for feature_set in target_top20.index
        },
        "target_holdout_precision": {
            feature_set: float(target.loc[feature_set, "precision"]) for feature_set in target.index
        },
        "target_holdout_policy_u_mean": {
            feature_set: float(target.loc[feature_set, "decision_policy_u_ela_mean"]) for feature_set in target.index
        },
        "non_target_call_rate": {
            feature_set: float(non_target.loc[feature_set, "decision_call_rate"]) for feature_set in non_target.index
        },
        "non_target_policy_u_mean": {
            feature_set: float(non_target.loc[feature_set, "decision_policy_u_ela_mean"]) for feature_set in non_target.index
        },
        "model_status": {
            str(row["feature_set"]): str(row["model_status"]) for _, row in model_summary.iterrows()
        },
        "interpretation": (
            "The H4 diagnostic compares feature expression only through diagnostic classifiers trained on f024 extension rows. "
            "Improved target-holdout capture is useful only if non-target call rate and policy U_ELA do not deteriorate."
        ),
    }


def _markdown_report(
    summary: dict[str, Any],
    metric_summary: pd.DataFrame,
    score_rank_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    main = metric_summary[
        metric_summary["evaluation_domain"].isin(
            [
                "extension_train_f024_target_changed_late",
                "validation_f024_target_holdout_changed_late",
                "validation_non_target_changed_late",
            ]
        )
        & (metric_summary["group"] == "overall")
    ].sort_values(["evaluation_domain", "feature_set", "policy_mode"])
    lines = [
        "# selection_reference 泛化失败 H4 最小诊断",
        "",
        "本报告只使用当前项目内已有 utility labels 与 trajectories；candidate behavior 只在诊断脚本内计算，未写回正式 feature extractor。",
        "",
        "## Feature-set Decision Metrics",
        "",
        "| domain | feature_set | policy | rows | positives | calls | capture | precision | called P_ELA | called U_ELA | policy U_ELA |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        called_p = "NA" if pd.isna(row["called_p_ela_mean"]) else f"{row['called_p_ela_mean']:.6f}"
        called_u = "NA" if pd.isna(row["called_u_ela_mean"]) else f"{row['called_u_ela_mean']:.6f}"
        lines.append(
            f"| `{row['evaluation_domain']}` | `{row['feature_set']}` | `{row['policy_mode']}` | {int(row['rows'])} | "
            f"{int(row['positive_rows'])} | {int(row['decision_call_rows'])} | "
            f"{row['u_ela_gt_zero_capture_rate']:.6f} | {row['precision']:.6f} | "
            f"{called_p} | {called_u} | {row['decision_policy_u_ela_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Training Thresholds",
            "",
            "| feature_set | status | threshold | train F1 | train policy U_ELA |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in model_summary.sort_values("feature_set").iterrows():
        lines.append(
            f"| `{row['feature_set']}` | `{row['model_status']}` | {row['threshold']:.6f} | "
            f"{row['train_threshold_f1']:.6f} | {row['train_threshold_decision_policy_u_ela_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 诊断边界",
            "",
            "- `P_ELA` 与 `U_ELA` 来自原始 utility label 行；这里只改变诊断调用规则，不生成新 continuation label。",
            "- `non-target validation` 定义为 validation changed_algorithm late-stage rows 中不属于 f024 target holdout 的行。",
            "- 该结果不能直接作为正式 behavior extractor 变更依据，只用于判断 candidate feature 表达是否值得进入下一轮稳健性检查。",
            "",
            "## 输出文件",
            "",
            f"- prediction rows: `{summary['outputs']['prediction_rows']}`",
            f"- metric summary: `{summary['outputs']['metric_summary']}`",
            f"- score-rank summary: `{summary['outputs']['score_rank_summary']}`",
            f"- model summary: `{summary['outputs']['model_summary']}`",
            f"- summary: `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _value_counts_string(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H4 behavior feature expression diagnostic for selection_reference generalization.")
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument(
        "--extension-train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train_late_stage_f024_followup/utility_labels_fe050.parquet"),
    )
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=Path("configs/min_support_bbob_train_late_stage_f024_followup.yaml"),
    )
    parser.add_argument(
        "--validation-trajectory-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_validation"),
    )
    parser.add_argument(
        "--extension-trajectory-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_train_late_stage_f024_followup"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/selection_reference_generalization_data_quality/h4_behavior_feature_expression"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--random-seed", type=int, default=1701)
    args = parser.parse_args()
    run_h4_behavior_feature_expression(
        validation_labels_path=args.validation_labels,
        extension_train_labels_path=args.extension_train_labels,
        extension_config_path=args.extension_config,
        validation_trajectory_root=args.validation_trajectory_root,
        extension_trajectory_root=args.extension_trajectory_root,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
