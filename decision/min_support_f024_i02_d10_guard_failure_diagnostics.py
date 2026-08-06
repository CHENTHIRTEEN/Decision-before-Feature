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
from decision.min_support_f024_two_feature_candidate_diagnostics import PRIMARY_FEATURE, _clean_records, _rule_calls


TARGET_PROBLEM_ID = "bbob_f024_i02_d10"
GUARD_PARTNERS = ("bf_diversity_change_w05", "bf_convergence_rate_w10")
ROLE_ORDER = (
    "captured_u_ela_gt_zero",
    "missed_u_ela_gt_zero",
    "guard_called_unhelpful",
)
ROLE_PAIR_ORDER = (
    ("guard_called_unhelpful", "captured_u_ela_gt_zero"),
    ("missed_u_ela_gt_zero", "captured_u_ela_gt_zero"),
    ("guard_called_unhelpful", "missed_u_ela_gt_zero"),
)


def run_f024_i02_d10_guard_failure_diagnostics(
    *,
    stability_rows_path: Path,
    two_feature_rules_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    rows = _load_target_rows(stability_rows_path, target_column)
    rules = _load_guard_rules(two_feature_rules_path)

    role_rows = _role_rows(rows, rules, target_column)
    feature_distributions = _feature_distribution_table(role_rows, target_column)
    role_contrasts = _role_feature_contrast_table(role_rows)
    threshold_margins = _threshold_margin_table(role_rows)
    coverage = _coverage_table(role_rows, target_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    role_rows_path = output_dir / "f024_i02_d10_guard_failure_rows.parquet"
    distribution_path = output_dir / "f024_i02_d10_guard_failure_feature_distributions.parquet"
    contrast_path = output_dir / "f024_i02_d10_guard_failure_role_feature_contrasts.parquet"
    margin_path = output_dir / "f024_i02_d10_guard_failure_threshold_margins.parquet"
    coverage_path = output_dir / "f024_i02_d10_guard_failure_coverage.parquet"
    summary_path = output_dir / "f024_i02_d10_guard_failure_summary.json"

    _write_parquet(role_rows, role_rows_path)
    _write_parquet(feature_distributions, distribution_path)
    _write_parquet(role_contrasts, contrast_path)
    _write_parquet(threshold_margins, margin_path)
    _write_parquet(coverage, coverage_path)

    summary = {
        "experiment": "min_support_f024_i02_d10_guard_failure_diagnostic",
        "research_question": (
            "For bbob_f024_i02_d10, do guard failures mainly come from threshold migration, limited sample "
            "coverage, or insufficient behavior-feature separability?"
        ),
        "target_column": target_column,
        "target_problem_id": TARGET_PROBLEM_ID,
        "guard_rules": _rule_summary(rules),
        "inputs": {
            "stability_rows": str(stability_rows_path),
            "two_feature_rules": str(two_feature_rules_path),
        },
        "rows": {
            "target_rows": int(len(rows)),
            "diagnostic_role_rows": int(len(role_rows)),
            "domains": sorted(rows["stability_domain"].astype(str).unique().tolist()),
        },
        "coverage_summary": _coverage_summary(coverage),
        "interpretation": _interpretation(coverage, threshold_margins, role_contrasts),
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
            "metadata_used_only_for_grouping": True,
        },
        "outputs": {
            "role_rows": str(role_rows_path),
            "feature_distributions": str(distribution_path),
            "role_feature_contrasts": str(contrast_path),
            "threshold_margins": str(margin_path),
            "coverage": str(coverage_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, default=_json_default, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote f024_i02_d10 guard failure rows to {role_rows_path}")
    print(f"wrote f024_i02_d10 guard failure feature distributions to {distribution_path}")
    print(f"wrote f024_i02_d10 guard failure role-feature contrasts to {contrast_path}")
    print(f"wrote f024_i02_d10 guard failure threshold margins to {margin_path}")
    print(f"wrote f024_i02_d10 guard failure coverage to {coverage_path}")
    print(f"wrote f024_i02_d10 guard failure summary to {summary_path}")
    return summary


def _load_target_rows(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_parquet(path)
    target_rows = rows[rows["problem_id"].astype(str) == TARGET_PROBLEM_ID].copy()
    if target_rows.empty:
        raise ValueError(f"no rows found for {TARGET_PROBLEM_ID}")
    required = [target_column, PRIMARY_FEATURE, *GUARD_PARTNERS, *BEHAVIOR_FEATURE_COLUMNS, *CANDIDATE_FEATURE_COLUMNS]
    missing = [column for column in required if column not in target_rows.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return target_rows


def _load_guard_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rules = pd.read_parquet(path)
    selected = rules[
        (rules["rule_name"] == "primary_and_partner")
        & (rules["rule_type"] == "and")
        & (rules["primary_feature"] == PRIMARY_FEATURE)
        & (rules["partner_feature"].isin(GUARD_PARTNERS))
    ].copy()
    missing = sorted(set(GUARD_PARTNERS) - set(selected["partner_feature"].astype(str)))
    if missing:
        raise ValueError(f"missing guard rules for {missing}")
    return selected.reset_index(drop=True)


def _role_rows(rows: pd.DataFrame, rules: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = []
    for _, rule in rules.iterrows():
        frame = rows.copy()
        observed = frame[target_column].to_numpy(dtype=float)
        utility_gt_zero = observed > 0.0
        calls = _rule_calls(frame, rule)
        frame["partner_feature"] = str(rule["partner_feature"])
        frame["primary_threshold"] = float(rule["primary_threshold"])
        frame["partner_threshold"] = float(rule["partner_threshold"])
        frame["primary_direction"] = str(rule["primary_direction"])
        frame["partner_direction"] = str(rule["partner_direction"])
        frame["guard_call"] = calls
        frame["utility_gt_zero"] = utility_gt_zero
        frame["guard_role"] = "skipped_unhelpful_reference"
        frame.loc[calls & utility_gt_zero, "guard_role"] = "captured_u_ela_gt_zero"
        frame.loc[~calls & utility_gt_zero, "guard_role"] = "missed_u_ela_gt_zero"
        frame.loc[calls & ~utility_gt_zero, "guard_role"] = "guard_called_unhelpful"
        frame["primary_signed_margin"] = _signed_margin(
            frame[PRIMARY_FEATURE].to_numpy(dtype=float),
            str(rule["primary_direction"]),
            float(rule["primary_threshold"]),
        )
        frame["partner_signed_margin"] = _signed_margin(
            frame[str(rule["partner_feature"])].to_numpy(dtype=float),
            str(rule["partner_direction"]),
            float(rule["partner_threshold"]),
        )
        frame["combined_min_signed_margin"] = np.minimum(frame["primary_signed_margin"], frame["partner_signed_margin"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined[combined["guard_role"].isin(ROLE_ORDER)].copy()


def _feature_distribution_table(role_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    feature_columns = _feature_columns(role_rows)
    result_rows = []
    for (partner, domain, role), subset in role_rows.groupby(["partner_feature", "stability_domain", "guard_role"], dropna=False):
        observed = subset[target_column].to_numpy(dtype=float)
        for feature in feature_columns:
            values = _finite_values(subset[feature])
            result_rows.append(
                {
                    "partner_feature": partner,
                    "stability_domain": domain,
                    "guard_role": role,
                    "feature": feature,
                    "feature_source": "candidate_behavior" if feature.startswith("cf_") else "existing_behavior",
                    "rows": int(len(subset)),
                    "finite_rows": int(len(values)),
                    "utility_sum": float(np.sum(observed)),
                    "utility_mean": float(np.mean(observed)) if len(observed) else None,
                    **_stats(values),
                }
            )
    return pd.DataFrame(result_rows)


def _role_feature_contrast_table(role_rows: pd.DataFrame) -> pd.DataFrame:
    feature_columns = _feature_columns(role_rows)
    result_rows = []
    for (partner, domain), subset in role_rows.groupby(["partner_feature", "stability_domain"], dropna=False):
        for left_role, right_role in ROLE_PAIR_ORDER:
            left = subset[subset["guard_role"] == left_role]
            right = subset[subset["guard_role"] == right_role]
            for feature in feature_columns:
                left_values = _finite_values(left[feature])
                right_values = _finite_values(right[feature])
                left_stats = _stats(left_values)
                right_stats = _stats(right_values)
                auc = _rank_auc(left_values, right_values)
                result_rows.append(
                    {
                        "partner_feature": partner,
                        "stability_domain": domain,
                        "left_role": left_role,
                        "right_role": right_role,
                        "feature": feature,
                        "feature_source": "candidate_behavior" if feature.startswith("cf_") else "existing_behavior",
                        "left_rows": int(len(left)),
                        "right_rows": int(len(right)),
                        "left_finite_rows": int(len(left_values)),
                        "right_finite_rows": int(len(right_values)),
                        "left_median": left_stats["median"],
                        "right_median": right_stats["median"],
                        "median_gap_abs": _abs_gap(left_stats["median"], right_stats["median"]),
                        "iqr_overlap_ratio": _interval_overlap_ratio(
                            left_stats["q25"],
                            left_stats["q75"],
                            right_stats["q25"],
                            right_stats["q75"],
                        ),
                        "range_overlap_ratio": _interval_overlap_ratio(
                            left_stats["min"],
                            left_stats["max"],
                            right_stats["min"],
                            right_stats["max"],
                        ),
                        "rank_auc_left_greater": auc,
                        "rank_separation": None if auc is None else float(abs(auc - 0.5) * 2.0),
                    }
                )
    return pd.DataFrame(result_rows)


def _threshold_margin_table(role_rows: pd.DataFrame) -> pd.DataFrame:
    margin_columns = ("primary_signed_margin", "partner_signed_margin", "combined_min_signed_margin")
    result_rows = []
    for (partner, domain, role), subset in role_rows.groupby(["partner_feature", "stability_domain", "guard_role"], dropna=False):
        for margin in margin_columns:
            values = _finite_values(subset[margin])
            stats = _stats(values)
            result_rows.append(
                {
                    "partner_feature": partner,
                    "stability_domain": domain,
                    "guard_role": role,
                    "margin": margin,
                    "rows": int(len(subset)),
                    "finite_rows": int(len(values)),
                    "pass_side_rows": int(np.sum(values > 0.0)),
                    "pass_side_rate": float(np.mean(values > 0.0)) if len(values) else None,
                    "near_threshold_rows_abs_margin_le_001": int(np.sum(np.abs(values) <= 0.01)),
                    "near_threshold_rate_abs_margin_le_001": float(np.mean(np.abs(values) <= 0.01)) if len(values) else None,
                    **stats,
                }
            )
    return pd.DataFrame(result_rows)


def _coverage_table(role_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result_rows = []
    for (partner, domain, role), subset in role_rows.groupby(["partner_feature", "stability_domain", "guard_role"], dropna=False):
        result_rows.append(
            {
                "partner_feature": partner,
                "stability_domain": domain,
                "guard_role": role,
                "rows": int(len(subset)),
                "seeds": sorted(int(seed) for seed in subset["seed"].unique()),
                "prefix_algorithms": sorted(str(value) for value in subset["prefix_algorithm"].unique()),
                "utility_sum": float(subset[target_column].sum()),
                "utility_mean": float(subset[target_column].mean()) if len(subset) else None,
                "guard_call_rows": int(subset["guard_call"].sum()),
                "utility_gt_zero_rows": int(subset["utility_gt_zero"].sum()),
            }
        )
    return pd.DataFrame(result_rows)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [feature for feature in BEHAVIOR_FEATURE_COLUMNS if feature in frame.columns]
    columns.extend(feature for feature in CANDIDATE_FEATURE_COLUMNS if feature in frame.columns)
    return columns


def _signed_margin(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "greater":
        return values - threshold
    if direction == "less_equal":
        return threshold - values
    raise ValueError(f"unknown direction: {direction}")


def _finite_values(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "q25": None, "q75": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _rank_auc(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    comparisons = left[:, None] - right[None, :]
    greater = np.sum(comparisons > 0.0)
    ties = np.sum(comparisons == 0.0)
    return float((greater + 0.5 * ties) / comparisons.size)


def _interval_overlap_ratio(a_low: float | None, a_high: float | None, b_low: float | None, b_high: float | None) -> float | None:
    if None in (a_low, a_high, b_low, b_high):
        return None
    a_width = max(float(a_high) - float(a_low), 0.0)
    b_width = max(float(b_high) - float(b_low), 0.0)
    denom = min(a_width, b_width)
    if denom <= 0.0:
        return None
    overlap = max(0.0, min(float(a_high), float(b_high)) - max(float(a_low), float(b_low)))
    return float(overlap / denom)


def _abs_gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(abs(left - right))


def _rule_summary(rules: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "primary_feature",
        "primary_direction",
        "primary_threshold",
        "partner_feature",
        "partner_direction",
        "partner_threshold",
        "rule_name",
        "rule_type",
    ]
    return _clean_records(rules[columns].to_dict(orient="records"))


def _coverage_summary(coverage: pd.DataFrame) -> list[dict[str, Any]]:
    return _clean_records(coverage.to_dict(orient="records"))


def _interpretation(coverage: pd.DataFrame, threshold_margins: pd.DataFrame, role_contrasts: pd.DataFrame) -> dict[str, Any]:
    role_counts = coverage.pivot_table(
        index=["partner_feature", "stability_domain"],
        columns="guard_role",
        values="rows",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for role in ROLE_ORDER:
        if role not in role_counts.columns:
            role_counts[role] = 0

    combined_margins = threshold_margins[threshold_margins["margin"] == "combined_min_signed_margin"].copy()
    called_unhelpful_margins = combined_margins[combined_margins["guard_role"] == "guard_called_unhelpful"]
    missed_margins = combined_margins[combined_margins["guard_role"] == "missed_u_ela_gt_zero"]
    strong_unhelpful = called_unhelpful_margins[
        (called_unhelpful_margins["median"].fillna(0.0) > 0.02)
        & (called_unhelpful_margins["rows"] > 0)
    ]
    near_missed = missed_margins[
        (missed_margins["near_threshold_rate_abs_margin_le_001"].fillna(0.0) >= 0.5)
        & (missed_margins["rows"] > 0)
    ]

    contrast_focus = role_contrasts[
        role_contrasts["left_role"].isin(["guard_called_unhelpful", "missed_u_ela_gt_zero"])
        & (role_contrasts["right_role"] == "captured_u_ela_gt_zero")
    ].copy()
    estimable = contrast_focus[contrast_focus["rank_separation"].notna()].copy()
    weak_feature_fraction = None
    if not estimable.empty:
        weak_feature_fraction = float((estimable["rank_separation"] < 0.35).mean())

    limited_domains = role_counts[
        (role_counts["captured_u_ela_gt_zero"] + role_counts["missed_u_ela_gt_zero"]) <= 1
    ]
    if len(limited_domains) >= 2:
        main_source = "sample_coverage_insufficient_with_behavior_feature_insufficiency"
    elif not strong_unhelpful.empty and (weak_feature_fraction is not None and weak_feature_fraction >= 0.5):
        main_source = "behavior_feature_insufficiency"
    elif not near_missed.empty:
        main_source = "threshold_migration"
    else:
        main_source = "mixed_or_inconclusive"

    return {
        "main_instability_source": main_source,
        "role_counts": _clean_records(role_counts.to_dict(orient="records")),
        "limited_useful_row_domains": _clean_records(limited_domains.to_dict(orient="records")),
        "strong_called_unhelpful_margin_rows": _clean_records(
            strong_unhelpful[
                ["partner_feature", "stability_domain", "guard_role", "rows", "median", "q25", "q75"]
            ].to_dict(orient="records")
        ),
        "near_threshold_missed_positive_rows": _clean_records(
            near_missed[
                [
                    "partner_feature",
                    "stability_domain",
                    "guard_role",
                    "rows",
                    "median",
                    "near_threshold_rate_abs_margin_le_001",
                ]
            ].to_dict(orient="records")
        ),
        "weak_feature_contrast_fraction": weak_feature_fraction,
        "reason": (
            "The diagnostic treats sample coverage as limiting when a domain has at most one U_ELA>0 row; "
            "threshold migration is supported when missed U_ELA>0 rows sit close to the combined threshold; "
            "behavior-feature insufficiency is supported when called-unhelpful rows pass the guard with clear "
            "margins and role contrasts remain weak across existing and candidate behavior features."
        ),
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose f024_i02_d10 two-feature guard failures.")
    parser.add_argument(
        "--stability-rows",
        type=Path,
        default=Path("results/decision/min_support/f024_two_feature_guard_stability/f024_two_feature_guard_stability_rows.parquet"),
    )
    parser.add_argument(
        "--two-feature-rules",
        type=Path,
        default=Path("results/decision/min_support/f024_two_feature_candidate_diagnostic/f024_two_feature_rules.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_i02_d10_guard_failure"),
    )
    args = parser.parse_args()
    run_f024_i02_d10_guard_failure_diagnostics(
        stability_rows_path=args.stability_rows,
        two_feature_rules_path=args.two_feature_rules,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
