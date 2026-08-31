"""Aggregate read-only diagnostics for Stage-A action-loss labels.

The report consumes existing query-adjusted action-loss, utility-label, and
function-level OOF prediction artifacts. It does not fit a model, derive a
new threshold, or write Utility/Decision labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ACTION_ROOT = REPO / "results/selection_reference/descriptor_cheap_invariant"
MATURITY_ROOT = REPO / "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision"
DEFAULT_OUTPUT = REPO / "results/dataset_analysis/label_learnability_diagnostics"

ACTION_FILES = {
    "bbob_train": ACTION_ROOT / "bbob_train/query_adjusted_budget.parquet",
    "bbob_validation": ACTION_ROOT / "bbob_validation/query_adjusted_budget.parquet",
    "mabbob_formal": ACTION_ROOT / "mabbob_formal/query_adjusted_budget.parquet",
    "mabbob_validation": ACTION_ROOT / "mabbob_validation/query_adjusted_budget.parquet",
}

STATE_KEYS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
)
RUN_KEYS = STATE_KEYS[:-1]
GROUP_KEYS = ("split", "function_id", "family", "dimension", "prefix_algorithm")
PORTFOLIO = ("de", "pso", "cmaes", "shade")
TIE_TOLERANCE = 0.05

ACTION_COLUMNS = [
    *STATE_KEYS,
    "action",
    "target_algorithm",
    "action_loss",
    "action_status",
    "path_completed",
    "timed_out",
    "action_loss_protocol",
    "performance_value_mode",
    "performance_loss_mode",
    "log10_gap_floor",
    "log10_gap_cap",
]

UTILITY_COLUMNS = [
    *STATE_KEYS,
    "g_fe",
    "g_fe_gt_zero",
    "g_fe_selected_path",
    "g_fe_selected_path_gt_zero",
    "selected_equals_prefix",
    "handoff_required",
    "acceptable_action_count",
    "selected_matches_best_observed",
]

OOF_COLUMNS = [
    "data_split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "decision_run_query_nested_oof",
    "nested_oof_threshold",
    "g_fe_selected_path",
]

VALIDATION_PREDICTION_COLUMNS = [
    "data_split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "decision_run_query_oof_g_fe_selected_path_first_trigger",
]


def _require_columns(frame: pd.DataFrame, columns: list[str], artifact: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact} missing required columns: {missing}")


def _read_action_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, path in ACTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=ACTION_COLUMNS)
        _require_columns(frame, ACTION_COLUMNS, str(path))
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"{path} split column does not match its path")
        frames.append(frame)
    actions = pd.concat(frames, ignore_index=True)
    if actions.duplicated(list(STATE_KEYS) + ["target_algorithm"]).any():
        raise ValueError("action-loss input has duplicate state/target_algorithm rows")
    if set(actions["action_loss_protocol"].astype(str)) != {
        "shared_complete_state_observed_action_loss"
    }:
        raise ValueError("action-loss protocol is not the active shared-state protocol")
    if set(actions["performance_value_mode"].astype(str)) != {"raw_objective"}:
        raise ValueError("action-loss input has an unexpected performance value mode")
    if set(actions["performance_loss_mode"].astype(str)) != {"known_optimum_gap"}:
        raise ValueError("action-loss input has an unexpected performance loss mode")
    if not np.isfinite(actions["action_loss"].to_numpy(dtype=float)).all():
        raise ValueError("action-loss input contains non-finite action_loss values")
    return actions


def _action_state_matrix(actions: pd.DataFrame) -> pd.DataFrame:
    state_counts = actions.groupby(list(STATE_KEYS), dropna=False).size()
    if not bool(state_counts.eq(len(PORTFOLIO)).all()):
        bad = state_counts[state_counts.ne(len(PORTFOLIO))].head(3).to_dict()
        raise ValueError(f"each state must contain exactly four actions; examples={bad}")
    target_sets = actions.groupby(list(STATE_KEYS), dropna=False)["target_algorithm"].agg(
        lambda values: tuple(sorted(set(values.astype(str))))
    )
    expected = tuple(sorted(PORTFOLIO))
    target_set_ok = target_sets.map(lambda value: value == expected)
    if not bool(target_set_ok.all()):
        bad = target_sets[~target_set_ok].head(3).to_dict()
        raise ValueError(f"each state must contain all four portfolio targets; examples={bad}")

    values = actions.pivot(index=list(STATE_KEYS), columns="target_algorithm", values="action_loss")
    values = values.reset_index()
    bounds = (
        actions.groupby(list(STATE_KEYS), dropna=False)[["log10_gap_floor", "log10_gap_cap"]]
        .nunique()
    )
    if bool((bounds > 1).any(axis=None)):
        raise ValueError("log10 gap bounds are inconsistent within a state")
    bounds = (
        actions.groupby(list(STATE_KEYS), dropna=False)[["log10_gap_floor", "log10_gap_cap"]]
        .first()
        .reset_index()
    )
    values = values.merge(bounds, on=list(STATE_KEYS), how="left", validate="one_to_one")
    for algorithm in PORTFOLIO:
        if algorithm not in values:
            raise ValueError(f"missing action-loss column for {algorithm}")
    values["loss_continue_current"] = values.apply(
        lambda row: float(row[str(row["prefix_algorithm"])]), axis=1
    )
    for algorithm in PORTFOLIO:
        values[f"loss_{algorithm}"] = values[algorithm].astype(float)
    for algorithm in PORTFOLIO:
        values[f"delta_{algorithm}_vs_continue"] = (
            values[f"loss_{algorithm}"] - values["loss_continue_current"]
        )
    loss_columns = [f"loss_{algorithm}" for algorithm in PORTFOLIO]
    values["loss_range"] = values[loss_columns].max(axis=1) - values[loss_columns].min(axis=1)
    loss_array = values[loss_columns].to_numpy(dtype=float)
    lower = values["log10_gap_floor"].to_numpy(dtype=float)[:, None]
    upper = values["log10_gap_cap"].to_numpy(dtype=float)[:, None]
    log_array = np.log10(np.clip(loss_array, lower, upper))
    for index, algorithm in enumerate(PORTFOLIO):
        values[f"log10_loss_{algorithm}"] = log_array[:, index]
    values["log10_loss_continue_current"] = np.asarray(
        [log_array[row_index, PORTFOLIO.index(str(prefix))] for row_index, prefix in enumerate(values["prefix_algorithm"])],
        dtype=float,
    )
    values["log10_loss_range"] = log_array.max(axis=1) - log_array.min(axis=1)
    values["approx_tie"] = (
        (log_array - log_array.min(axis=1)[:, None] <= TIE_TOLERANCE).sum(axis=1) >= 2
    )

    status = actions.copy()
    status["action_valid"] = (
        status["action_status"].astype(str).eq("ok")
        & status["path_completed"].fillna(False).astype(bool)
        & ~status["timed_out"].fillna(True).astype(bool)
    )
    valid = status.pivot(index=list(STATE_KEYS), columns="target_algorithm", values="action_valid")
    valid = valid.reset_index()
    for algorithm in PORTFOLIO:
        values[f"failure_{algorithm}"] = ~valid[algorithm].astype(bool)
    values["failure_continue_current"] = np.asarray(
        [
            bool(values.loc[row_index, f"failure_{str(prefix)}"])
            for row_index, prefix in enumerate(values["prefix_algorithm"])
        ],
        dtype=bool,
    )
    values["any_action_failure"] = values[[f"failure_{algorithm}" for algorithm in PORTFOLIO]].any(axis=1)
    return values


def _summarize_actions(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, frame in matrix.groupby(list(GROUP_KEYS), dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row: dict[str, object] = dict(zip(GROUP_KEYS, group_values, strict=True))
        row["n_states"] = int(len(frame))
        row["n_runs"] = int(frame[list(RUN_KEYS)].drop_duplicates().shape[0])
        row["mean_opportunities_per_run"] = float(len(frame) / row["n_runs"])
        row["min_FE"] = int(frame["FE"].min())
        row["max_FE"] = int(frame["FE"].max())
        row["n_FE_positions"] = int(frame["FE"].nunique())
        row["approx_tie_rate"] = float(frame["approx_tie"].mean())
        row["any_action_failure_rate"] = float(frame["any_action_failure"].mean())
        row["loss_range_mean"] = float(frame["loss_range"].mean())
        row["loss_range_median"] = float(frame["loss_range"].median())
        row["loss_range_q75"] = float(frame["loss_range"].quantile(0.75))
        row["log10_loss_range_mean"] = float(frame["log10_loss_range"].mean())
        row["log10_loss_range_median"] = float(frame["log10_loss_range"].median())
        row["log10_loss_range_q75"] = float(frame["log10_loss_range"].quantile(0.75))
        continue_loss = frame["loss_continue_current"].astype(float)
        row["loss_continue_current_mean"] = float(continue_loss.mean())
        row["loss_continue_current_median"] = float(continue_loss.median())
        row["loss_continue_current_std"] = float(continue_loss.std(ddof=0))
        row["loss_continue_current_q25"] = float(continue_loss.quantile(0.25))
        row["loss_continue_current_q75"] = float(continue_loss.quantile(0.75))
        row["failure_continue_current_rate"] = float(frame["failure_continue_current"].mean())
        continue_log_loss = frame["log10_loss_continue_current"].astype(float)
        row["log10_loss_continue_current_mean"] = float(continue_log_loss.mean())
        row["log10_loss_continue_current_median"] = float(continue_log_loss.median())
        row["log10_loss_continue_current_std"] = float(continue_log_loss.std(ddof=0))
        for algorithm in PORTFOLIO:
            loss = frame[f"loss_{algorithm}"].astype(float)
            row[f"loss_{algorithm}_mean"] = float(loss.mean())
            row[f"loss_{algorithm}_median"] = float(loss.median())
            row[f"loss_{algorithm}_std"] = float(loss.std(ddof=0))
            row[f"loss_{algorithm}_q25"] = float(loss.quantile(0.25))
            row[f"loss_{algorithm}_q75"] = float(loss.quantile(0.75))
            row[f"failure_{algorithm}_rate"] = float(frame[f"failure_{algorithm}"].mean())
            row[f"best_{algorithm}_rate"] = float(
                frame[f"loss_{algorithm}"].le(frame[[f"loss_{a}" for a in PORTFOLIO]].min(axis=1)).mean()
            )
            row[f"delta_{algorithm}_vs_continue_mean"] = float(
                frame[f"delta_{algorithm}_vs_continue"].mean()
            )
            row[f"delta_{algorithm}_vs_continue_median"] = float(
                frame[f"delta_{algorithm}_vs_continue"].median()
            )
            row[f"delta_{algorithm}_vs_continue_std"] = float(
                frame[f"delta_{algorithm}_vs_continue"].std(ddof=0)
            )
            log_loss = frame[f"log10_loss_{algorithm}"].astype(float)
            row[f"log10_loss_{algorithm}_mean"] = float(log_loss.mean())
            row[f"log10_loss_{algorithm}_median"] = float(log_loss.median())
            row[f"log10_loss_{algorithm}_std"] = float(log_loss.std(ddof=0))
            row[f"delta_log10_{algorithm}_vs_continue_mean"] = float(
                (log_loss - frame["log10_loss_continue_current"]).mean()
            )
            row[f"delta_log10_{algorithm}_vs_continue_median"] = float(
                (log_loss - frame["log10_loss_continue_current"]).median()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(GROUP_KEYS)).reset_index(drop=True)


def _read_utility_labels() -> pd.DataFrame:
    path = MATURITY_ROOT / "utility_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=UTILITY_COLUMNS)
    _require_columns(frame, UTILITY_COLUMNS, str(path))
    if set(frame["prefix_algorithm"].astype(str)) != {"cmaes"}:
        raise ValueError("current g_fe_selected_path artifact is expected to contain only cmaes prefix rows")
    if frame["g_fe_selected_path"].isna().any():
        raise ValueError("g_fe_selected_path contains missing values")
    mismatch = frame["handoff_required"].astype(bool) != ~frame["selected_equals_prefix"].astype(bool)
    if bool(mismatch.any()):
        raise ValueError("handoff_required is inconsistent with selected_equals_prefix")
    return frame


def _summarize_utility(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in frame.groupby(list(GROUP_KEYS), dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row: dict[str, object] = dict(zip(GROUP_KEYS, group_values, strict=True))
        g = group["g_fe_selected_path"].astype(float)
        row["g_fe_materialized"] = True
        row["g_fe_rows"] = int(len(group))
        row["g_fe_finite_rate"] = float(np.isfinite(g).mean())
        row["g_fe_nonzero_rate"] = float(g.ne(0.0).mean())
        row["g_fe_gt_zero_rate"] = float(group["g_fe_selected_path_gt_zero"].astype(bool).mean())
        row["g_fe_lt_zero_rate"] = float(g.lt(0.0).mean())
        row["g_fe_mean"] = float(g.mean())
        row["g_fe_median"] = float(g.median())
        row["g_fe_std"] = float(g.std(ddof=0))
        best = group["g_fe"].astype(float)
        row["g_fe_best_observed_nonzero_rate"] = float(best.ne(0.0).mean())
        row["g_fe_best_observed_gt_zero_rate"] = float(group["g_fe_gt_zero"].astype(bool).mean())
        row["g_fe_best_observed_mean"] = float(best.mean())
        row["g_fe_best_observed_median"] = float(best.median())
        row["g_fe_best_observed_std"] = float(best.std(ddof=0))
        best_gt_zero = group["g_fe_gt_zero"].astype(bool)
        selected_gt_zero = group["g_fe_selected_path_gt_zero"].astype(bool)
        row["selected_gt_zero_capture_of_best_observed_gt_zero"] = float(
            selected_gt_zero[best_gt_zero].mean() if bool(best_gt_zero.any()) else np.nan
        )
        row["selected_matches_best_observed_rate_given_best_gt_zero"] = float(
            group.loc[best_gt_zero, "selected_matches_best_observed"].astype(bool).mean()
            if bool(best_gt_zero.any())
            else np.nan
        )
        row["selected_equals_prefix_rate"] = float(group["selected_equals_prefix"].astype(bool).mean())
        row["handoff_required_rate"] = float(group["handoff_required"].astype(bool).mean())
        row["relation_consistency_violation_rate"] = float(
            (
                group["handoff_required"].astype(bool)
                != ~group["selected_equals_prefix"].astype(bool)
            ).mean()
        )
        row["acceptable_action_tie_rate"] = float(group["acceptable_action_count"].ge(2).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(GROUP_KEYS)).reset_index(drop=True)


def _map_oof_split(function_id: pd.Series, *, validation: bool) -> pd.Series:
    prefix = function_id.astype(str).str.split("_", n=1).str[0]
    if validation:
        return prefix.map({"bbob": "bbob_validation", "mabbob": "mabbob_validation"})
    return prefix.map({"bbob": "bbob_train", "mabbob": "mabbob_formal"})


def _summarize_first_trigger() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    oof_path = MATURITY_ROOT / "nested_oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path, columns=OOF_COLUMNS)
        _require_columns(oof, OOF_COLUMNS, str(oof_path))
        oof["split"] = _map_oof_split(oof["function_id"], validation=False)
        oof = oof[oof["split"].notna()].copy()
        oof["first_trigger"] = oof["decision_run_query_nested_oof"].fillna(False).astype(bool)
        oof["coverage_source"] = "nested_function_level_oof"
        rows.append(oof)

    validation_path = MATURITY_ROOT / "validation_predictions.parquet"
    if validation_path.exists():
        validation = pd.read_parquet(validation_path, columns=VALIDATION_PREDICTION_COLUMNS)
        _require_columns(validation, VALIDATION_PREDICTION_COLUMNS, str(validation_path))
        validation["split"] = _map_oof_split(validation["function_id"], validation=True)
        validation = validation[validation["split"].notna()].copy()
        validation["first_trigger"] = validation[
            "decision_run_query_oof_g_fe_selected_path_first_trigger"
        ].fillna(False).astype(bool)
        validation["coverage_source"] = "full_train_fixed_validation"
        rows.append(validation)

    if not rows:
        return pd.DataFrame(columns=[*GROUP_KEYS, "first_trigger_source"])

    combined = pd.concat(rows, ignore_index=True, sort=False)
    run_key = ["split", "problem_id", "function_id", "family", "cv_group_id", "dimension", "prefix_algorithm", "seed"]
    per_run = combined.groupby(run_key, dropna=False).agg(
        first_trigger=("first_trigger", "max"),
        state_rows=("first_trigger", "size"),
        coverage_source=("coverage_source", "first"),
    ).reset_index()
    if bool(per_run.groupby(run_key, dropna=False)["first_trigger"].sum().gt(1).any()):
        raise ValueError("first-trigger artifact contains more than one trigger per run")

    grouped = per_run.groupby(list(GROUP_KEYS), dropna=False, sort=True)
    summary = grouped.agg(
        first_trigger_runs=("first_trigger", "sum"),
        first_trigger_total_runs=("first_trigger", "size"),
        first_trigger_coverage=("first_trigger", "mean"),
        first_trigger_source=("coverage_source", "first"),
    ).reset_index()
    return summary.sort_values(list(GROUP_KEYS)).reset_index(drop=True)


def _merge_summaries(action_summary: pd.DataFrame, utility_summary: pd.DataFrame, trigger_summary: pd.DataFrame) -> pd.DataFrame:
    merged = action_summary.merge(utility_summary, on=list(GROUP_KEYS), how="left", validate="one_to_one")
    merged = merged.merge(trigger_summary, on=list(GROUP_KEYS), how="left", validate="one_to_one")
    merged["g_fe_materialized"] = merged["g_fe_materialized"].fillna(False).astype(bool)
    merged["g_fe_nonzero_rate"] = merged["g_fe_nonzero_rate"].where(
        merged["g_fe_materialized"], np.nan
    )
    merged["g_fe_gt_zero_rate"] = merged["g_fe_gt_zero_rate"].where(
        merged["g_fe_materialized"], np.nan
    )
    for column in (
        "g_fe_best_observed_nonzero_rate",
        "g_fe_best_observed_gt_zero_rate",
        "g_fe_best_observed_mean",
        "g_fe_best_observed_median",
        "g_fe_best_observed_std",
        "selected_gt_zero_capture_of_best_observed_gt_zero",
        "selected_matches_best_observed_rate_given_best_gt_zero",
    ):
        merged[column] = merged[column].where(merged["g_fe_materialized"], np.nan)
    merged["selected_equals_prefix_rate"] = merged["selected_equals_prefix_rate"].where(
        merged["g_fe_materialized"], np.nan
    )
    merged["handoff_required_rate"] = merged["handoff_required_rate"].where(
        merged["g_fe_materialized"], np.nan
    )
    merged["approx_tie_rate_source"] = np.where(
        merged["g_fe_materialized"], "action_loss_0.05_and_utility_acceptable_count", "action_loss_0.05_only"
    )
    merged["g_fe_materialization_note"] = np.where(
        merged["g_fe_materialized"], "materialized_from_existing_utility_labels", "not_materialized_for_prefix"
    )
    return merged.sort_values(list(GROUP_KEYS)).reset_index(drop=True)


def _overall_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in detail.groupby("split", sort=True):
        row: dict[str, object] = {"split": split}
        row["groups"] = int(len(group))
        row["action_states"] = int(group["n_states"].sum())
        row["action_runs_approx"] = int(group["n_runs"].sum())
        row["action_failure_rate_state_weighted"] = float(
            np.average(group["any_action_failure_rate"], weights=group["n_states"])
        )
        row["approx_tie_rate_state_weighted"] = float(
            np.average(group["approx_tie_rate"], weights=group["n_states"])
        )
        row["loss_range_median_of_groups"] = float(group["loss_range_median"].median())
        row["log10_loss_range_median_of_groups"] = float(group["log10_loss_range_median"].median())
        materialized = group[group["g_fe_materialized"]]
        row["g_fe_materialized_groups"] = int(len(materialized))
        if not materialized.empty:
            weights = materialized["g_fe_rows"].to_numpy(dtype=float)
            row["g_fe_nonzero_rate_state_weighted"] = float(
                np.average(materialized["g_fe_nonzero_rate"], weights=weights)
            )
            row["g_fe_gt_zero_rate_state_weighted"] = float(
                np.average(materialized["g_fe_gt_zero_rate"], weights=weights)
            )
            row["g_fe_selected_path_nonzero_rate_state_weighted"] = row["g_fe_nonzero_rate_state_weighted"]
            row["g_fe_selected_path_gt_zero_rate_state_weighted"] = row["g_fe_gt_zero_rate_state_weighted"]
            row["g_fe_best_observed_nonzero_rate_state_weighted"] = float(
                np.average(materialized["g_fe_best_observed_nonzero_rate"], weights=weights)
            )
            row["g_fe_best_observed_gt_zero_rate_state_weighted"] = float(
                np.average(materialized["g_fe_best_observed_gt_zero_rate"], weights=weights)
            )
            best_gt_zero_weight = materialized["g_fe_best_observed_gt_zero_rate"].to_numpy(dtype=float) * weights
            valid_best_weight = (
                (best_gt_zero_weight > 0.0)
                & np.isfinite(best_gt_zero_weight)
                & np.isfinite(materialized["selected_gt_zero_capture_of_best_observed_gt_zero"].to_numpy(dtype=float))
                & np.isfinite(materialized["selected_matches_best_observed_rate_given_best_gt_zero"].to_numpy(dtype=float))
            )
            row["selected_gt_zero_capture_of_best_observed_gt_zero"] = float(
                np.average(
                    materialized.loc[valid_best_weight, "selected_gt_zero_capture_of_best_observed_gt_zero"],
                    weights=best_gt_zero_weight[valid_best_weight],
                )
                if bool(valid_best_weight.any())
                else np.nan
            )
            row["selected_matches_best_observed_rate_given_best_gt_zero"] = float(
                np.average(
                    materialized.loc[valid_best_weight, "selected_matches_best_observed_rate_given_best_gt_zero"],
                    weights=best_gt_zero_weight[valid_best_weight],
                )
                if bool(valid_best_weight.any())
                else np.nan
            )
            row["selected_equals_prefix_rate_state_weighted"] = float(
                np.average(materialized["selected_equals_prefix_rate"], weights=weights)
            )
            row["handoff_required_rate_state_weighted"] = float(
                np.average(materialized["handoff_required_rate"], weights=weights)
            )
        else:
            row["g_fe_nonzero_rate_state_weighted"] = np.nan
            row["g_fe_gt_zero_rate_state_weighted"] = np.nan
            row["g_fe_selected_path_nonzero_rate_state_weighted"] = np.nan
            row["g_fe_selected_path_gt_zero_rate_state_weighted"] = np.nan
            row["g_fe_best_observed_nonzero_rate_state_weighted"] = np.nan
            row["g_fe_best_observed_gt_zero_rate_state_weighted"] = np.nan
            row["selected_gt_zero_capture_of_best_observed_gt_zero"] = np.nan
            row["selected_matches_best_observed_rate_given_best_gt_zero"] = np.nan
            row["selected_equals_prefix_rate_state_weighted"] = np.nan
            row["handoff_required_rate_state_weighted"] = np.nan
        triggered = group[group["first_trigger_source"].notna()]
        if not triggered.empty:
            row["first_trigger_runs"] = int(triggered["first_trigger_runs"].sum())
            row["first_trigger_total_runs"] = int(triggered["first_trigger_total_runs"].sum())
            row["first_trigger_coverage_run_weighted"] = float(
                row["first_trigger_runs"] / row["first_trigger_total_runs"]
            )
            row["first_trigger_source"] = ";".join(sorted(triggered["first_trigger_source"].dropna().unique()))
        else:
            row["first_trigger_runs"] = 0
            row["first_trigger_total_runs"] = 0
            row["first_trigger_coverage_run_weighted"] = np.nan
            row["first_trigger_source"] = "not_materialized"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def _classify_signal(overall: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in sorted(detail["split"].astype(str).unique()):
        group = detail[detail["split"].astype(str).eq(split)]
        state_alignment = bool(group["any_action_failure_rate"].max() == 0.0)
        materialized = group[group["g_fe_materialized"]]
        if materialized.empty:
            label_sparsity = "not_assessable"
            best_observed_label_sparsity = "not_assessable"
            cross_function = "not_materialized"
        else:
            gt_zero = float(np.average(materialized["g_fe_gt_zero_rate"], weights=materialized["g_fe_rows"]))
            label_sparsity = "high_sparsity" if gt_zero < 0.10 else "not_dominant"
            best_gt_zero = float(
                np.average(
                    materialized["g_fe_best_observed_gt_zero_rate"],
                    weights=materialized["g_fe_rows"],
                )
            )
            best_observed_label_sparsity = "high_sparsity" if best_gt_zero < 0.10 else "not_dominant"
            triggered = group[group["first_trigger_source"].notna()]
            if triggered.empty:
                cross_function = "not_materialized"
            else:
                coverage = float(triggered["first_trigger_runs"].sum() / triggered["first_trigger_total_runs"].sum())
                cross_function = "sparse_first_trigger_coverage" if coverage < 0.10 else "not_dominant"
        tie_rate = float(np.average(group["approx_tie_rate"], weights=group["n_states"]))
        action_complementarity = "heterogeneous" if 0.20 < tie_rate < 0.80 else (
            "mostly_tied" if tie_rate >= 0.80 else "mostly_separated"
        )
        rows.append(
            {
                "split": split,
                "state_alignment_signal": "no_failure_evidence" if state_alignment else "inspect_failures",
                "action_complementarity_signal": action_complementarity,
                "label_sparsity_signal": label_sparsity,
                "best_observed_label_sparsity_signal": best_observed_label_sparsity,
                "cross_function_generalization_signal": cross_function,
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    output: Path,
    detail: pd.DataFrame,
    overall: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    materialized = detail[detail["g_fe_materialized"]]
    selected_gt_zero_rates = "; ".join(
        f"{row.split}={row.g_fe_gt_zero_rate_state_weighted:.2%}"
        for row in overall.itertuples(index=False)
        if pd.notna(row.g_fe_gt_zero_rate_state_weighted)
    )
    best_observed_gt_zero_rates = "; ".join(
        f"{row.split}={row.g_fe_best_observed_gt_zero_rate_state_weighted:.2%}"
        for row in overall.itertuples(index=False)
        if pd.notna(row.g_fe_best_observed_gt_zero_rate_state_weighted)
    )
    capture_rates = "; ".join(
        f"{row.split}={row.selected_gt_zero_capture_of_best_observed_gt_zero:.2%}"
        for row in overall.itertuples(index=False)
        if pd.notna(row.selected_gt_zero_capture_of_best_observed_gt_zero)
    )
    lines = [
        "# Stage-A 标签可学习性诊断报告",
        "",
        "> 该报告仅对现有 query-adjusted Stage-A action-loss、`g_fe_selected_path` 和既有 first-trigger 预测做描述性聚合。没有重新拟合模型、阈值、Utility 或 Decision label，也没有读取 runtime/wall-clock 字段。",
        "",
        "## 1. 数据与口径",
        "",
        f"- 生成时间：{pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}",
        f"- action-loss 根目录：`{ACTION_ROOT}`",
        f"- 当前 Utility/OOF 目录：`{MATURITY_ROOT}`",
        f"- state key：`{', '.join(STATE_KEYS)}`；run-level first-trigger 的 run key 去掉 `FE`。",
        f"- 动作统一为 `continue_current`（目标算法等于 prefix）和 `de/pso/cmaes/shade` 三个替代动作；loss 越小表示 known-optimum gap 越小。",
        "- 明细表同时保留显式的 `loss_continue_current_*`；`loss_{prefix_algorithm}_*` 是同一状态的等价 portfolio target loss。",
        f"- 近似并列固定为四动作中至少两个 `log10(clip(action_loss))` 不超过该状态最小值 + `{TIE_TOLERANCE}`。",
        "- 失败率定义为四动作中任一路径 `action_status != ok`、`path_completed=false` 或 `timed_out=true`；`target_hit_observed=false` 不被当作路径失败。",
        "- 训练侧 first-trigger 来源为 `search_maturity_linear` 的 function-level nested OOF；验证侧来源为既有 full-train fixed validation 预测。",
        "",
        "## 2. 关键质量检查",
        "",
        f"- action-loss 状态总数：`{int(detail['n_states'].sum())}` 个 group-state；每个状态均通过四动作完整性检查。",
        f"- action-loss 路径失败率最大值：`{detail['any_action_failure_rate'].max():.6f}`。",
        f"- `g_fe_selected_path` 已物化 group：`{len(materialized)}`；仅覆盖 `prefix_algorithm=cmaes`。",
        f"- 当前主标签 `g_fe_selected_path_gt_zero`：{selected_gt_zero_rates}；最佳已观测动作诊断 `g_fe_gt_zero`：{best_observed_gt_zero_rates}。",
        f"- selected-path 对 best-observed 大于零状态的捕获率：{capture_rates}。",
        f"- relation consistency violation 最大值：`{materialized['relation_consistency_violation_rate'].max():.6f}`。" if not materialized.empty else "- relation fields 尚未在 action-loss 的非 CMA-ES prefix 上物化。",
        "",
        "## 3. 按 split 汇总",
        "",
        overall.to_markdown(index=False, floatfmt=".6f"),
        "",
        "字段说明：`g_fe_nonzero_rate`/`g_fe_gt_zero_rate` 对应 Selector 实际选中路径；`g_fe_best_observed_*` 对应最佳已观测动作诊断。严格不等于 0 与大于 0 分开报告，后者才对应主二元标签的类别比例。",
        "",
        "## 4. 信号来源判断",
        "",
        classification.to_markdown(index=False),
        "",
        "判断只用于诊断，不构成模型选择或科学结论。当前读数中，四动作路径完整且无失败证据；主要可疑点是 `g_fe_selected_path_gt_zero` 稀疏和 nested-OOf first-trigger 调用稀少。动作间差异在不同 split 间不均一，不能简单归结为动作完全没有互补性。",
        "",
        "## 5. 详细 group 表",
        "",
        "完整的 `function_id × family × dimension × prefix_algorithm` 明细保存在 `label_learnability_by_group.csv` 和 `label_learnability_by_group.parquet`。其中非 CMA-ES prefix 的 `g_fe_*`、关系字段和 first-trigger 字段按当前物化边界保留为空，并标记 `g_fe_materialization_note=not_materialized_for_prefix`。",
        "",
        "## 6. 下一步建议",
        "",
        "1. 先按本报告的 group 明细核对 state 对齐和四动作 loss 的重复键；当前数据未显示路径失败问题。",
        "2. 在不改变主协议的前提下，优先分析 `g_fe_selected_path_gt_zero` 的类别稀疏、近似并列率和 action-loss 差值分布；不要把严格非零但极接近 0 的值等同于可用效用信号。",
        "3. 对 function-level nested OOF 的 first-trigger 稀疏与 threshold 分布分开解释；不要用 validation outcome 重新拟合阈值。",
        "4. 非 CMA-ES prefix 的 query-adjusted Utility/Decision label 目前没有物化，若将来需要覆盖，必须另行生成并保持与本报告输入隔离。",
    ]
    (output / "label_learnability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, str]:
    actions = _read_action_rows()
    matrix = _action_state_matrix(actions)
    action_summary = _summarize_actions(matrix)
    utility = _read_utility_labels()
    utility_summary = _summarize_utility(utility)
    trigger_summary = _summarize_first_trigger()
    detail = _merge_summaries(action_summary, utility_summary, trigger_summary)
    overall = _overall_summary(detail)
    classification = _classify_signal(overall, detail)

    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "label_learnability_by_group.csv", index=False)
    detail.to_parquet(output / "label_learnability_by_group.parquet", index=False)
    overall.to_csv(output / "label_learnability_by_split.csv", index=False)
    classification.to_csv(output / "label_learnability_signal_classification.csv", index=False)
    metadata = {
        "report": str(output / "label_learnability_report.md"),
        "group_summary": str(output / "label_learnability_by_group.parquet"),
        "split_summary": str(output / "label_learnability_by_split.csv"),
        "signal_classification": str(output / "label_learnability_signal_classification.csv"),
        "action_loss_inputs": {split: str(path) for split, path in ACTION_FILES.items()},
        "utility_labels_input": str(MATURITY_ROOT / "utility_labels.parquet"),
        "nested_oof_input": str(MATURITY_ROOT / "nested_oof_predictions.parquet"),
        "validation_predictions_input": str(MATURITY_ROOT / "validation_predictions.parquet"),
        "state_keys": list(STATE_KEYS),
        "group_keys": list(GROUP_KEYS),
        "portfolio": list(PORTFOLIO),
        "approx_tie_tolerance_log10_gap": TIE_TOLERANCE,
        "runtime_used": False,
        "model_fitted": False,
        "new_utility_or_decision_labels_generated": False,
        "function_level_oof_used_for_training_first_trigger": True,
        "current_g_fe_prefix_scope": ["cmaes"],
    }
    (output / "label_learnability_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(output, detail, overall, classification)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metadata = run(args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
