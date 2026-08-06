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
from decision.min_support_f024_two_feature_candidate_diagnostics import (
    PRIMARY_FEATURE,
    _clean_records,
    _group_dict,
    _group_label,
    _metrics,
    _rule_calls,
)


GUARD_PARTNERS = ("bf_diversity_change_w05", "bf_convergence_rate_w10")
RULE_NAME = "primary_and_partner"
RULE_TYPE = "and"
GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "problem_id": ["problem_id"],
    "family_dimension_problem_id": ["family", "dimension", "problem_id"],
}


def run_f024_two_feature_guard_stability(
    *,
    candidate_values_path: Path,
    validation_labels_path: Path,
    extension_labels_path: Path,
    two_feature_rules_path: Path,
    candidate_failure_path: Path,
    existing_failure_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    rules = _load_guard_rules(two_feature_rules_path)
    rows = _load_stability_rows(
        candidate_values_path=candidate_values_path,
        validation_labels_path=validation_labels_path,
        extension_labels_path=extension_labels_path,
        target_column=target_column,
    )
    failure_rows = _load_failure_rows(
        candidate_failure_path=candidate_failure_path,
        existing_failure_path=existing_failure_path,
        target_column=target_column,
    )

    domain_performance = _domain_performance_table(rows, rules, target_column)
    layered_performance = _layered_performance_table(rows, rules, target_column)
    failure_coverage = _failure_coverage_table(failure_rows, rules, target_column)
    stability_gap = _stability_gap_table(domain_performance)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "f024_two_feature_guard_stability_rows.parquet"
    domain_path = output_dir / "f024_two_feature_guard_domain_performance.parquet"
    layered_path = output_dir / "f024_two_feature_guard_layered_performance.parquet"
    failure_path = output_dir / "f024_two_feature_guard_failure_coverage.parquet"
    gap_path = output_dir / "f024_two_feature_guard_stability_gap.parquet"
    summary_path = output_dir / "f024_two_feature_guard_stability_summary.json"

    _write_parquet(rows, rows_path)
    _write_parquet(domain_performance, domain_path)
    _write_parquet(layered_performance, layered_path)
    _write_parquet(failure_coverage, failure_path)
    _write_parquet(stability_gap, gap_path)

    summary = {
        "experiment": "min_support_f024_two_feature_guard_stability_diagnostic",
        "research_question": (
            "Do the cf_elite_centroid_shift_norm AND existing-behavior guard rules remain useful across "
            "target holdout, target non-holdout, and extension_train f024 target rows?"
        ),
        "target_column": target_column,
        "primary_feature": PRIMARY_FEATURE,
        "guard_partners": list(GUARD_PARTNERS),
        "inputs": {
            "candidate_values": str(candidate_values_path),
            "validation_labels": str(validation_labels_path),
            "extension_labels": str(extension_labels_path),
            "two_feature_rules": str(two_feature_rules_path),
            "candidate_failure_values": str(candidate_failure_path),
            "existing_failure_samples": str(existing_failure_path),
        },
        "rows": {
            "stability_rows": int(len(rows)),
            "target_holdout_rows": int((rows["stability_domain"] == "target_holdout_fe050").sum()),
            "target_non_holdout_rows": int((rows["stability_domain"] == "target_non_holdout_fe050").sum()),
            "extension_train_target_rows": int((rows["stability_domain"] == "extension_train_target_fe050").sum()),
            "failure_rows": int(len(failure_rows)),
        },
        "best_domain_summary": _best_domain_summary(domain_performance),
        "stability_interpretation": _interpretation(domain_performance, stability_gap),
        "data_leakage_check": {
            "formal_models_retrained": False,
            "formal_phase1_configs_modified": False,
            "original_utility_labels_modified": False,
            "ela_features_used_as_decision_input": False,
            "function_id_used_as_decision_input": False,
            "algorithm_id_used_as_decision_input": False,
            "algorithm_internal_parameters_used": False,
            "uses_existing_behavior_and_candidate_feature_values": True,
            "uses_existing_two_feature_threshold_rules": True,
            "extension_train_failure_samples_available": False,
            "metadata_used_only_for_grouping": True,
        },
        "outputs": {
            "stability_rows": str(rows_path),
            "domain_performance": str(domain_path),
            "layered_performance": str(layered_path),
            "failure_coverage": str(failure_path),
            "stability_gap": str(gap_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, default=_json_default, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote f024 two-feature guard stability rows to {rows_path}")
    print(f"wrote f024 two-feature guard domain performance to {domain_path}")
    print(f"wrote f024 two-feature guard layered performance to {layered_path}")
    print(f"wrote f024 two-feature guard failure coverage to {failure_path}")
    print(f"wrote f024 two-feature guard stability gap to {gap_path}")
    print(f"wrote f024 two-feature guard stability summary to {summary_path}")
    return summary


def _load_guard_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rules = pd.read_parquet(path)
    rules = rules[
        (rules["rule_name"] == RULE_NAME)
        & (rules["rule_type"] == RULE_TYPE)
        & (rules["primary_feature"] == PRIMARY_FEATURE)
        & (rules["partner_feature"].isin(GUARD_PARTNERS))
    ].copy()
    if set(rules["partner_feature"]) != set(GUARD_PARTNERS):
        missing = sorted(set(GUARD_PARTNERS) - set(rules["partner_feature"]))
        raise ValueError(f"missing guard rules for {missing}")
    return rules.reset_index(drop=True)


def _load_stability_rows(
    *,
    candidate_values_path: Path,
    validation_labels_path: Path,
    extension_labels_path: Path,
    target_column: str,
) -> pd.DataFrame:
    if not candidate_values_path.exists():
        raise FileNotFoundError(candidate_values_path)
    candidate_values = pd.read_parquet(candidate_values_path)
    validation = _merge_behavior_features(candidate_values, pd.read_parquet(validation_labels_path), "validation")
    extension = _merge_behavior_features(candidate_values, pd.read_parquet(extension_labels_path), "extension_train")
    rows = pd.concat([validation, extension], ignore_index=True, sort=False)

    rows["stability_domain"] = None
    validation_mask = (
        (rows["analysis_data_split"] == "validation")
        & (rows["label_source"] == "changed_algorithm")
        & (rows["FE_ratio"].astype(float).round(6) == 0.5)
    )
    rows.loc[validation_mask & rows["is_target_holdout_seed"].astype(bool), "stability_domain"] = "target_holdout_fe050"
    rows.loc[validation_mask & ~rows["is_target_holdout_seed"].astype(bool), "stability_domain"] = "target_non_holdout_fe050"

    extension_mask = (
        (rows["analysis_data_split"] == "extension_train")
        & (rows["label_source"] == "changed_algorithm")
        & (rows["FE_ratio"].astype(float).round(6) == 0.5)
    )
    rows.loc[extension_mask, "stability_domain"] = "extension_train_target_fe050"

    required = [PRIMARY_FEATURE, *GUARD_PARTNERS, target_column]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    result = rows[rows["stability_domain"].notna()].copy()
    return result


def _merge_behavior_features(candidate_values: pd.DataFrame, labels: pd.DataFrame, data_split: str) -> pd.DataFrame:
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
    behavior_columns = [feature for feature in BEHAVIOR_FEATURE_COLUMNS if feature in labels.columns]
    frame = candidate_values[candidate_values["analysis_data_split"] == data_split].copy()
    return frame.merge(labels[key_columns + behavior_columns], on=key_columns, how="left", validate="many_to_one")


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
    existing_features = [feature for feature in GUARD_PARTNERS if feature in existing.columns]
    merged = candidate.merge(existing[key_columns + existing_features], on=key_columns, how="left", validate="many_to_one")
    fe050 = merged[merged["FE_ratio"].astype(float).round(6) == 0.5].copy()
    fe050["stability_domain"] = None
    target_problem = fe050["eval_domain"].astype(str) == "target_problem_changed_late_fe050"
    target_holdout = fe050["eval_domain"].astype(str) == "target_holdout_changed_late_fe050"
    fe050.loc[target_problem & ~fe050["is_target_holdout_seed"].astype(bool), "stability_domain"] = "target_non_holdout_fe050"
    fe050.loc[target_holdout, "stability_domain"] = "target_holdout_fe050"
    return fe050[fe050["stability_domain"].notna()].copy()


def _domain_performance_table(rows: pd.DataFrame, rules: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result_rows = []
    for _, rule in rules.iterrows():
        for domain, subset in rows.groupby("stability_domain", dropna=False):
            calls = _rule_calls(subset, rule)
            result_rows.append(
                {
                    **rule.to_dict(),
                    "stability_domain": domain,
                    **_metrics(subset[target_column].to_numpy(dtype=float), calls),
                }
            )
    return pd.DataFrame(result_rows)


def _layered_performance_table(rows: pd.DataFrame, rules: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result_rows = []
    for _, rule in rules.iterrows():
        for domain, domain_frame in rows.groupby("stability_domain", dropna=False):
            for layer, group_columns in GROUP_LAYERS.items():
                grouped = [((), domain_frame)] if not group_columns else domain_frame.groupby(group_columns, dropna=False)
                for group_values, subset in grouped:
                    group = _group_dict(group_columns, group_values)
                    calls = _rule_calls(subset, rule)
                    result_rows.append(
                        {
                            **rule.to_dict(),
                            "stability_domain": domain,
                            "layer": layer,
                            "group": _group_label(group),
                            "family": group.get("family"),
                            "dimension": group.get("dimension"),
                            "problem_id": group.get("problem_id"),
                            **_metrics(subset[target_column].to_numpy(dtype=float), calls),
                        }
                    )
    return pd.DataFrame(result_rows)


def _failure_coverage_table(failure_rows: pd.DataFrame, rules: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if failure_rows.empty:
        return pd.DataFrame()
    result_rows = []
    for _, rule in rules.iterrows():
        for (domain, failure_role, training_dataset, model_name), subset in failure_rows.groupby(
            ["stability_domain", "failure_role", "training_dataset", "model_name"],
            dropna=False,
        ):
            calls = _rule_calls(subset, rule)
            observed = subset[target_column].to_numpy(dtype=float)
            utility_gt_zero = observed > 0.0
            result_rows.append(
                {
                    **rule.to_dict(),
                    "stability_domain": domain,
                    "failure_role": failure_role,
                    "training_dataset": training_dataset,
                    "model_name": model_name,
                    "failure_rows": int(len(subset)),
                    "utility_gt_zero_rows": int(np.sum(utility_gt_zero)),
                    "rule_call_rows": int(np.sum(calls)),
                    "rule_call_rate": float(np.mean(calls)) if len(calls) else None,
                    "rule_capture_utility_gt_zero_rows": int(np.sum(calls & utility_gt_zero)),
                    "rule_call_unhelpful_rows": int(np.sum(calls & ~utility_gt_zero)),
                }
            )
    return pd.DataFrame(result_rows)


def _stability_gap_table(domain_performance: pd.DataFrame) -> pd.DataFrame:
    holdout = domain_performance[domain_performance["stability_domain"] == "target_holdout_fe050"]
    rows = []
    for _, reference in holdout.iterrows():
        key = str(reference["partner_feature"])
        for _, candidate in domain_performance[domain_performance["partner_feature"] == key].iterrows():
            rows.append(
                {
                    "partner_feature": key,
                    "comparison_domain": candidate["stability_domain"],
                    "holdout_recall": float(reference["recall"]),
                    "comparison_recall": float(candidate["recall"]),
                    "recall_delta_vs_holdout": float(candidate["recall"] - reference["recall"]),
                    "holdout_unhelpful_call_cost_sum": float(reference["unhelpful_call_cost_sum"]),
                    "comparison_unhelpful_call_cost_sum": float(candidate["unhelpful_call_cost_sum"]),
                    "unhelpful_cost_delta_vs_holdout": float(
                        candidate["unhelpful_call_cost_sum"] - reference["unhelpful_call_cost_sum"]
                    ),
                    "holdout_f1": float(reference["f1"]),
                    "comparison_f1": float(candidate["f1"]),
                    "f1_delta_vs_holdout": float(candidate["f1"] - reference["f1"]),
                    "holdout_decision_mean_utility": float(reference["decision_mean_utility"]),
                    "comparison_decision_mean_utility": float(candidate["decision_mean_utility"]),
                    "decision_mean_utility_delta_vs_holdout": float(
                        candidate["decision_mean_utility"] - reference["decision_mean_utility"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _best_domain_summary(domain_performance: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "partner_feature",
        "stability_domain",
        "rows",
        "positive_rows",
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
    return _clean_records(domain_performance[columns].sort_values(["stability_domain", "partner_feature"]).to_dict(orient="records"))


def _interpretation(domain_performance: pd.DataFrame, stability_gap: pd.DataFrame) -> dict[str, Any]:
    holdout = domain_performance[domain_performance["stability_domain"] == "target_holdout_fe050"]
    extension = domain_performance[domain_performance["stability_domain"] == "extension_train_target_fe050"]
    non_holdout = domain_performance[domain_performance["stability_domain"] == "target_non_holdout_fe050"]
    return {
        "holdout_reference": _best_domain_summary(holdout),
        "non_holdout_check": _best_domain_summary(non_holdout),
        "extension_train_check": _best_domain_summary(extension),
        "largest_f1_drop_vs_holdout": _clean_records(
            stability_gap.sort_values("f1_delta_vs_holdout").head(1).to_dict(orient="records")
        ),
        "summary": (
            "A stable guard should keep useful-row capture reasonably close to the target holdout result while "
            "not increasing unhelpful call cost. This diagnostic applies the same two thresholds to each domain; "
            "it does not tune thresholds on non-holdout or extension_train rows."
        ),
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run f024 two-feature guard stability diagnostics.")
    parser.add_argument(
        "--candidate-values",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_feature_candidates/f024_candidate_behavior_feature_values.parquet"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument(
        "--extension-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train_late_stage_f024_followup/utility_labels_fe050.parquet"),
    )
    parser.add_argument(
        "--two-feature-rules",
        type=Path,
        default=Path("results/decision/min_support/f024_two_feature_candidate_diagnostic/f024_two_feature_rules.parquet"),
    )
    parser.add_argument(
        "--candidate-failure-values",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_feature_candidates/f024_candidate_score_failure_values.parquet"),
    )
    parser.add_argument(
        "--existing-failure-samples",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_separability/f024_score_ranking_failure_samples.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_two_feature_guard_stability"),
    )
    args = parser.parse_args()
    run_f024_two_feature_guard_stability(
        candidate_values_path=args.candidate_values,
        validation_labels_path=args.validation_labels,
        extension_labels_path=args.extension_labels,
        two_feature_rules_path=args.two_feature_rules,
        candidate_failure_path=args.candidate_failure_values,
        existing_failure_path=args.existing_failure_samples,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
