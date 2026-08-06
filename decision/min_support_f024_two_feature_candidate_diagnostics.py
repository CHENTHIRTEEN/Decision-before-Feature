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
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default
from decision.min_support_f024_behavior_feature_candidates import CANDIDATE_FEATURE_COLUMNS


PRIMARY_FEATURE = "cf_elite_centroid_shift_norm"
PARTNER_FEATURES = (
    "bf_diversity_change_w05",
    "bf_convergence_rate_w10",
    "cf_elite_fitness_gap_norm",
)
FOCUS_DOMAIN = "validation_target_holdout_changed_late_fe050"
EXISTING_FOCUS_DOMAIN = "validation_target_holdout_fe050"
GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "FE_ratio": ["FE_ratio"],
    "problem_id": ["problem_id"],
    "family_dimension_FE_ratio_problem_id": ["family", "dimension", "FE_ratio", "problem_id"],
}


def run_f024_two_feature_candidate_diagnostics(
    *,
    candidate_dir: Path,
    separability_dir: Path,
    validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    target_rows = _load_target_holdout_rows(
        candidate_values_path=candidate_dir / "f024_candidate_behavior_feature_values.parquet",
        validation_labels_path=validation_labels_path,
        target_column=target_column,
    )
    thresholds = _load_thresholds(
        candidate_thresholds_path=candidate_dir / "f024_candidate_single_feature_thresholds.parquet",
        existing_thresholds_path=separability_dir / "f024_single_feature_threshold_separability.parquet",
    )
    failure_rows = _load_failure_rows(
        candidate_failure_path=candidate_dir / "f024_candidate_score_failure_values.parquet",
        existing_failure_path=separability_dir / "f024_score_ranking_failure_samples.parquet",
        target_column=target_column,
    )

    rule_rows = _build_rule_rows(thresholds)
    observed_useful_descriptors = _observed_useful_rule_descriptors(target_rows, rule_rows, target_column)
    rule_performance = _rule_performance_table(target_rows, rule_rows, observed_useful_descriptors, target_column)
    layered_performance = _layered_performance_table(target_rows, rule_rows, observed_useful_descriptors, target_column)
    segment_table = _quadrant_segment_table(target_rows, rule_rows, target_column)
    failure_coverage = _failure_coverage_table(failure_rows, rule_rows, observed_useful_descriptors, target_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    rules_path = output_dir / "f024_two_feature_rules.parquet"
    performance_path = output_dir / "f024_two_feature_rule_performance.parquet"
    layered_path = output_dir / "f024_two_feature_layered_performance.parquet"
    segments_path = output_dir / "f024_two_feature_quadrant_segments.parquet"
    failure_path = output_dir / "f024_two_feature_failure_coverage.parquet"
    summary_path = output_dir / "f024_two_feature_candidate_diagnostic_summary.json"

    _write_parquet(rule_rows, rules_path)
    _write_parquet(rule_performance, performance_path)
    _write_parquet(layered_performance, layered_path)
    _write_parquet(segment_table, segments_path)
    _write_parquet(failure_coverage, failure_path)

    summary = {
        "experiment": "min_support_f024_two_feature_candidate_diagnostic",
        "research_question": (
            "Can two-feature threshold combinations involving cf_elite_centroid_shift_norm improve "
            "U_ELA>0 capture or reduce unhelpful calls on f024 target holdout FE_ratio=0.50 changed_algorithm rows?"
        ),
        "target_column": target_column,
        "primary_feature": PRIMARY_FEATURE,
        "partner_features": list(PARTNER_FEATURES),
        "inputs": {
            "candidate_dir": str(candidate_dir),
            "separability_dir": str(separability_dir),
            "validation_labels": str(validation_labels_path),
        },
        "rows": {
            "target_holdout_fe050_changed_algorithm_rows": int(len(target_rows)),
            "target_holdout_fe050_u_ela_gt_zero_rows": int((target_rows[target_column] > 0.0).sum()),
            "failure_sample_rows": int(len(failure_rows)),
        },
        "best_rules": _best_rule_summary(rule_performance),
        "interpretation": _interpretation(rule_performance, segment_table),
        "data_leakage_check": {
            "formal_models_retrained": False,
            "formal_phase1_configs_modified": False,
            "original_utility_labels_modified": False,
            "ela_features_used_as_decision_input": False,
            "function_id_used_as_decision_input": False,
            "algorithm_id_used_as_decision_input": False,
            "algorithm_internal_parameters_used": False,
            "uses_existing_behavior_and_candidate_feature_values": True,
            "uses_existing_single_feature_threshold_tables": True,
            "observed_useful_quadrants_are_diagnostic_not_formal_threshold_selection": True,
            "metadata_used_only_for_grouping": True,
        },
        "outputs": {
            "rules": str(rules_path),
            "rule_performance": str(performance_path),
            "layered_performance": str(layered_path),
            "quadrant_segments": str(segments_path),
            "failure_coverage": str(failure_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, default=_json_default, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote f024 two-feature rules to {rules_path}")
    print(f"wrote f024 two-feature rule performance to {performance_path}")
    print(f"wrote f024 two-feature layered performance to {layered_path}")
    print(f"wrote f024 two-feature quadrant segments to {segments_path}")
    print(f"wrote f024 two-feature failure coverage to {failure_path}")
    print(f"wrote f024 two-feature diagnostic summary to {summary_path}")
    return summary


def _load_target_holdout_rows(
    *,
    candidate_values_path: Path,
    validation_labels_path: Path,
    target_column: str,
) -> pd.DataFrame:
    if not candidate_values_path.exists():
        raise FileNotFoundError(candidate_values_path)
    if not validation_labels_path.exists():
        raise FileNotFoundError(validation_labels_path)
    candidates = pd.read_parquet(candidate_values_path)
    labels = pd.read_parquet(validation_labels_path)
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
    existing_features = [feature for feature in BEHAVIOR_FEATURE_COLUMNS if feature in labels.columns]
    frame = candidates.merge(
        labels[key_columns + existing_features],
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    mask = (
        (frame["analysis_data_split"].astype(str) == "validation")
        & (frame["label_source"].astype(str) == "changed_algorithm")
        & frame["is_target_holdout_seed"].astype(bool)
        & (frame["FE_ratio"].astype(float).round(6) == 0.5)
    )
    required = [PRIMARY_FEATURE, *PARTNER_FEATURES, target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return frame.loc[mask].copy()


def _load_thresholds(*, candidate_thresholds_path: Path, existing_thresholds_path: Path) -> pd.DataFrame:
    if not candidate_thresholds_path.exists():
        raise FileNotFoundError(candidate_thresholds_path)
    if not existing_thresholds_path.exists():
        raise FileNotFoundError(existing_thresholds_path)
    candidate = pd.read_parquet(candidate_thresholds_path)
    candidate = candidate[
        (candidate["data_split"] == FOCUS_DOMAIN)
        & (candidate["layer"] == "overall")
        & (candidate["feature"].isin([PRIMARY_FEATURE, "cf_elite_fitness_gap_norm"]))
    ].copy()
    candidate["threshold_source"] = "candidate_single_feature_threshold_table"

    existing = pd.read_parquet(existing_thresholds_path)
    existing = existing[
        (existing["data_split"] == EXISTING_FOCUS_DOMAIN)
        & (existing["layer"] == "overall")
        & (existing["feature"].isin(["bf_diversity_change_w05", "bf_convergence_rate_w10"]))
    ].copy()
    existing["threshold_source"] = "existing_behavior_single_feature_threshold_table"
    thresholds = pd.concat([candidate, existing], ignore_index=True, sort=False)
    required_features = {PRIMARY_FEATURE, *PARTNER_FEATURES}
    available_features = set(thresholds["feature"].astype(str))
    missing = sorted(required_features - available_features)
    if missing:
        raise ValueError(f"missing threshold rows for {missing}")
    return thresholds


def _load_failure_rows(
    *,
    candidate_failure_path: Path,
    existing_failure_path: Path,
    target_column: str,
) -> pd.DataFrame:
    if not candidate_failure_path.exists() or not existing_failure_path.exists():
        return pd.DataFrame()
    candidate = pd.read_parquet(candidate_failure_path)
    existing = pd.read_parquet(existing_failure_path)
    key_columns = [
        "failure_role",
        "eval_domain",
        "training_dataset",
        "model_name",
        "problem_id",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
    ]
    existing_features = [
        feature
        for feature in ("bf_diversity_change_w05", "bf_convergence_rate_w10")
        if feature in existing.columns
    ]
    merged = candidate.merge(
        existing[key_columns + existing_features],
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    return merged[
        (merged["eval_domain"] == "target_holdout_changed_late_fe050")
        & (merged["FE_ratio"].astype(float).round(6) == 0.5)
    ].copy()


def _build_rule_rows(thresholds: pd.DataFrame) -> pd.DataFrame:
    threshold_by_feature = {str(row["feature"]): row for _, row in thresholds.iterrows()}
    primary = threshold_by_feature[PRIMARY_FEATURE]
    rows = []
    for partner_feature in PARTNER_FEATURES:
        partner = threshold_by_feature[partner_feature]
        base = {
            "primary_feature": PRIMARY_FEATURE,
            "primary_direction": str(primary["direction"]),
            "primary_threshold": float(primary["threshold"]),
            "primary_threshold_source": str(primary["threshold_source"]),
            "partner_feature": partner_feature,
            "partner_direction": str(partner["direction"]),
            "partner_threshold": float(partner["threshold"]),
            "partner_threshold_source": str(partner["threshold_source"]),
        }
        rows.extend(
            [
                {**base, "rule_name": "primary_only", "rule_type": "single_feature_reference"},
                {**base, "rule_name": "partner_only", "rule_type": "single_feature_reference"},
                {**base, "rule_name": "primary_and_partner", "rule_type": "and"},
                {**base, "rule_name": "primary_or_partner", "rule_type": "or"},
                {**base, "rule_name": "quadrant_both_call", "rule_type": "segmented_quadrant"},
                {**base, "rule_name": "quadrant_primary_only", "rule_type": "segmented_quadrant"},
                {**base, "rule_name": "quadrant_partner_only", "rule_type": "segmented_quadrant"},
                {**base, "rule_name": "quadrant_neither_call_reference", "rule_type": "segmented_quadrant_reference"},
            ]
        )
    return pd.DataFrame(rows)


def _rule_performance_table(
    frame: pd.DataFrame,
    rules: pd.DataFrame,
    observed_useful_descriptors: list[dict[str, Any]],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for _, rule in rules.iterrows():
        calls = _rule_calls(frame, rule)
        rows.append({**rule.to_dict(), **_metrics(frame[target_column].to_numpy(dtype=float), calls)})
    useful_rows = _observed_useful_quadrant_rules(frame, observed_useful_descriptors, target_column)
    rows.extend(useful_rows)
    return pd.DataFrame(rows)


def _layered_performance_table(
    frame: pd.DataFrame,
    rules: pd.DataFrame,
    observed_useful_descriptors: list[dict[str, Any]],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    all_rules = pd.concat([rules, pd.DataFrame(observed_useful_descriptors)], ignore_index=True)
    for _, rule in all_rules.iterrows():
        for layer, group_columns in GROUP_LAYERS.items():
            grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False)
            for group_values, subset in grouped:
                group = _group_dict(group_columns, group_values)
                calls = _rule_calls(subset, rule)
                rows.append(
                    {
                        **rule.to_dict(),
                        "layer": layer,
                        "group": _group_label(group),
                        "family": group.get("family"),
                        "dimension": group.get("dimension"),
                        "FE_ratio": group.get("FE_ratio"),
                        "problem_id": group.get("problem_id"),
                        **_metrics(subset[target_column].to_numpy(dtype=float), calls),
                    }
                )
    return pd.DataFrame(rows)


def _quadrant_segment_table(frame: pd.DataFrame, rules: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for _, rule in rules.drop_duplicates("partner_feature").iterrows():
        primary_call, partner_call = _base_calls(frame, rule)
        segments = {
            "both_call": primary_call & partner_call,
            "primary_only": primary_call & ~partner_call,
            "partner_only": ~primary_call & partner_call,
            "neither_call": ~primary_call & ~partner_call,
        }
        observed = frame[target_column].to_numpy(dtype=float)
        for segment_name, mask in segments.items():
            rows.append(
                {
                    "primary_feature": PRIMARY_FEATURE,
                    "partner_feature": rule["partner_feature"],
                    "segment": segment_name,
                    **_segment_metrics(observed, mask),
                }
            )
    return pd.DataFrame(rows)


def _failure_coverage_table(
    failure_rows: pd.DataFrame,
    rules: pd.DataFrame,
    observed_useful_descriptors: list[dict[str, Any]],
    target_column: str,
) -> pd.DataFrame:
    if failure_rows.empty:
        return pd.DataFrame()
    rows = []
    all_rules = pd.concat([rules, pd.DataFrame(observed_useful_descriptors)], ignore_index=True)
    for _, rule in all_rules.iterrows():
        calls = _rule_calls(failure_rows, rule)
        for (failure_role, training_dataset, model_name), subset in failure_rows.groupby(
            ["failure_role", "training_dataset", "model_name"],
            dropna=False,
        ):
            positions = failure_rows.index.get_indexer(subset.index)
            local_calls = calls[positions]
            observed = subset[target_column].to_numpy(dtype=float)
            utility_gt_zero = observed > 0.0
            rows.append(
                {
                    **rule.to_dict(),
                    "failure_role": failure_role,
                    "training_dataset": training_dataset,
                    "model_name": model_name,
                    "failure_rows": int(len(subset)),
                    "utility_gt_zero_rows": int(np.sum(utility_gt_zero)),
                    "rule_call_rows": int(np.sum(local_calls)),
                    "rule_call_rate": float(np.mean(local_calls)) if len(local_calls) else None,
                    "rule_capture_utility_gt_zero_rows": int(np.sum(local_calls & utility_gt_zero)),
                    "rule_call_unhelpful_rows": int(np.sum(local_calls & ~utility_gt_zero)),
                }
            )
    return pd.DataFrame(rows)


def _observed_useful_quadrant_rules(
    frame: pd.DataFrame,
    observed_useful_descriptors: list[dict[str, Any]],
    target_column: str,
) -> list[dict[str, Any]]:
    rows = []
    for descriptor in observed_useful_descriptors:
        calls = _rule_calls(frame, pd.Series(descriptor))
        rows.append({**descriptor, **_metrics(frame[target_column].to_numpy(dtype=float), calls)})
    return rows


def _observed_useful_rule_descriptors(frame: pd.DataFrame, rules: pd.DataFrame, target_column: str) -> list[dict[str, Any]]:
    descriptors = []
    observed = frame[target_column].to_numpy(dtype=float)
    for _, rule in rules.drop_duplicates("partner_feature").iterrows():
        primary_call, partner_call = _base_calls(frame, rule)
        segments = {
            "both_call": primary_call & partner_call,
            "primary_only": primary_call & ~partner_call,
            "partner_only": ~primary_call & partner_call,
        }
        useful_segments = []
        for segment_name, mask in segments.items():
            if not np.any(mask):
                continue
            if float(np.mean(observed[mask])) > 0.0:
                useful_segments.append(segment_name)
        descriptors.append(
            {
                **rule.to_dict(),
                "rule_name": "observed_useful_quadrants",
                "rule_type": "diagnostic_label_observed_segmented",
                "observed_useful_segments": ",".join(useful_segments),
            }
        )
    return descriptors


def _rule_calls(frame: pd.DataFrame, rule: pd.Series) -> np.ndarray:
    primary_call, partner_call = _base_calls(frame, rule)
    rule_name = str(rule["rule_name"])
    if rule_name == "primary_only":
        return primary_call
    if rule_name == "partner_only":
        return partner_call
    if rule_name == "primary_and_partner":
        return primary_call & partner_call
    if rule_name == "primary_or_partner":
        return primary_call | partner_call
    if rule_name == "quadrant_both_call":
        return primary_call & partner_call
    if rule_name == "quadrant_primary_only":
        return primary_call & ~partner_call
    if rule_name == "quadrant_partner_only":
        return ~primary_call & partner_call
    if rule_name == "quadrant_neither_call_reference":
        return ~primary_call & ~partner_call
    if rule_name == "observed_useful_quadrants":
        selected = set(str(rule.get("observed_useful_segments", "")).split(",")) - {""}
        calls = np.zeros(len(frame), dtype=bool)
        if "both_call" in selected:
            calls |= primary_call & partner_call
        if "primary_only" in selected:
            calls |= primary_call & ~partner_call
        if "partner_only" in selected:
            calls |= ~primary_call & partner_call
        return calls
    raise ValueError(f"unknown rule_name: {rule_name}")


def _base_calls(frame: pd.DataFrame, rule: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    primary = _threshold_call(
        frame[str(rule["primary_feature"])].to_numpy(dtype=float),
        str(rule["primary_direction"]),
        float(rule["primary_threshold"]),
    )
    partner = _threshold_call(
        frame[str(rule["partner_feature"])].to_numpy(dtype=float),
        str(rule["partner_direction"]),
        float(rule["partner_threshold"]),
    )
    return primary, partner


def _threshold_call(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    if direction == "greater":
        return finite & (values > threshold)
    if direction == "less_equal":
        return finite & (values <= threshold)
    raise ValueError(f"unknown threshold direction: {direction}")


def _metrics(observed: np.ndarray, calls: np.ndarray) -> dict[str, float | int]:
    labels = observed > 0.0
    true_run = int(np.sum(calls & labels))
    unhelpful = int(np.sum(calls & ~labels))
    missed = int(np.sum(~calls & labels))
    skipped_unhelpful = int(np.sum(~calls & ~labels))
    positive_sum = float(np.sum(observed[labels]))
    captured_sum = float(np.sum(observed[calls & labels]))
    precision = true_run / (true_run + unhelpful) if true_run + unhelpful else 0.0
    recall = true_run / (true_run + missed) if true_run + missed else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(observed)),
        "positive_rows": int(np.sum(labels)),
        "decision_call_rows": int(np.sum(calls)),
        "true_run_ela_rows": true_run,
        "unhelpful_call_rows": unhelpful,
        "missed_positive_rows": missed,
        "skip_when_unhelpful_rows": skipped_unhelpful,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "utility_capture_rate": captured_sum / positive_sum if positive_sum > 0.0 else 0.0,
        "captured_positive_utility_sum": captured_sum,
        "unhelpful_call_cost_sum": float(-np.sum(observed[calls & ~labels])),
        "decision_mean_utility": float(np.mean(np.where(calls, observed, 0.0))) if len(observed) else None,
    }


def _segment_metrics(observed: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    labels = observed > 0.0
    segment_values = observed[mask]
    segment_labels = labels[mask]
    positive_sum = float(np.sum(observed[labels]))
    segment_positive_sum = float(np.sum(segment_values[segment_values > 0.0]))
    return {
        "rows": int(len(observed)),
        "segment_rows": int(np.sum(mask)),
        "segment_positive_rows": int(np.sum(segment_labels)),
        "segment_unhelpful_rows": int(np.sum(mask & ~labels)),
        "segment_precision_if_called": float(np.mean(segment_labels)) if len(segment_labels) else None,
        "segment_utility_sum": float(np.sum(segment_values)),
        "segment_mean_utility": float(np.mean(segment_values)) if len(segment_values) else None,
        "segment_utility_capture_rate_if_called": segment_positive_sum / positive_sum if positive_sum > 0.0 else 0.0,
        "segment_unhelpful_call_cost_if_called": float(-np.sum(segment_values[segment_values <= 0.0])),
    }


def _best_rule_summary(rule_performance: pd.DataFrame) -> dict[str, Any]:
    if rule_performance.empty:
        return {}
    sort_columns = ["f1", "decision_mean_utility", "utility_capture_rate"]
    best_f1 = rule_performance.sort_values(sort_columns, ascending=False).head(5)
    best_mean = rule_performance.sort_values(["decision_mean_utility", "f1"], ascending=False).head(5)
    return {
        "best_by_f1": _compact_rules(best_f1),
        "best_by_decision_mean_utility": _compact_rules(best_mean),
    }


def _compact_rules(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "rule_name",
        "rule_type",
        "primary_feature",
        "partner_feature",
        "observed_useful_segments",
        "decision_call_rows",
        "true_run_ela_rows",
        "unhelpful_call_rows",
        "precision",
        "recall",
        "f1",
        "utility_capture_rate",
        "unhelpful_call_cost_sum",
        "decision_mean_utility",
    ]
    return _clean_records(frame[[column for column in columns if column in frame.columns]].to_dict(orient="records"))


def _interpretation(rule_performance: pd.DataFrame, segment_table: pd.DataFrame) -> dict[str, Any]:
    if rule_performance.empty:
        return {"summary": "No rule rows were available."}
    primary = rule_performance[
        (rule_performance["rule_name"] == "primary_only")
        & (rule_performance["partner_feature"] == "bf_diversity_change_w05")
    ]
    and_or = rule_performance[rule_performance["rule_type"].isin(["and", "or"])]
    best_and_or = and_or.sort_values(["decision_mean_utility", "f1"], ascending=False).head(1)
    useful_segments = segment_table[segment_table["segment_mean_utility"].fillna(0.0) > 0.0]
    return {
        "primary_reference": _compact_rules(primary),
        "best_and_or_by_decision_mean_utility": _compact_rules(best_and_or),
        "observed_useful_quadrant_count": int(len(useful_segments)),
        "observed_useful_quadrants": useful_segments[
            ["partner_feature", "segment", "segment_rows", "segment_positive_rows", "segment_mean_utility"]
        ].pipe(lambda frame: _clean_records(frame.to_dict(orient="records"))),
        "summary": (
            "AND rules test whether partner thresholds can reduce unhelpful calls from cf_elite_centroid_shift_norm; "
            "OR rules test missed-positive recovery. Quadrant rows are diagnostic summaries of the two-threshold "
            "partition and are not formal threshold selection."
        ),
    }


def _clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _clean_value(value) for key, value in row.items()} for row in records]


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def _group_dict(group_columns: list[str], group_values: Any) -> dict[str, Any]:
    if not group_columns:
        return {}
    if not isinstance(group_values, tuple):
        group_values = (group_values,)
    return dict(zip(group_columns, group_values, strict=True))


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "overall"
    return "|".join(f"{key}={value}" for key, value in group.items())


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run f024 two-feature candidate diagnostics for min-support outputs.")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_feature_candidates"),
    )
    parser.add_argument(
        "--separability-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_separability"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_two_feature_candidate_diagnostic"),
    )
    args = parser.parse_args()
    run_f024_two_feature_candidate_diagnostics(
        candidate_dir=args.candidate_dir,
        separability_dir=args.separability_dir,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
