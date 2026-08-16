from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from decision.cluster_weighting import (
    CLUSTER_BALANCED_FIT,
    WeightedMedianImputer,
    cluster_balanced_row_weights,
    fit_pipeline_with_weights,
)
from landscape_queries.specs import QUERY_PREPROCESSING_VERSION, LandscapeQuerySpec, get_query_spec
from selection_reference.action_losses import (
    ACTION_LOSS_PROTOCOL,
    BEHAVIOR_ONLY_FULL_BUDGET,
    FULL_REMAINING_BUDGET,
    EXECUTION_ORDER_PROTOCOL,
    NOT_APPLICABLE,
    PRE_RUN_QUERY_ADJUSTED_BUDGET,
    QUERY_ADJUSTED_BUDGET,
    STATE_KEY_COLUMNS,
    TIMING_REPETITIONS,
)
from trajectory.sampling import SAMPLING_METADATA_COLUMNS


SELECTION_REFERENCE_PROTOCOL = "statewise_observed_action_loss_regression"
QUERY_ONLY_SELECTION_REFERENCE_PROTOCOL = "query_only_observed_action_loss_regression"
BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL = "behavior_only_observed_action_loss_regression"
PRE_RUN_SELECTION_REFERENCE_PROTOCOL = "pre_run_query_only_observed_algorithm_loss_regression"
SELECTOR_TARGET_TRANSFORM = "clipped_log10_gap_advantage_vs_continue_current"
SELECTOR_TARGET_SENSITIVITY_TRANSFORM = "statewise_minmax_observed_action_loss"
PRE_RUN_SELECTOR_TARGET_TRANSFORM = "clipped_log10_observed_action_loss"
QUERY_FULL_INPUT = "query_full"
STATE_ONLY_INPUT = "state_only"
QUERY_ONLY_INPUT = "query_only"
BEHAVIOR_ONLY_FULL_BUDGET_INPUT = "behavior_only_full_budget"
PRE_RUN_QUERY_ONLY_INPUT = "pre_run_query_only"
SELECTOR_INPUT_MODES = (
    QUERY_FULL_INPUT,
    STATE_ONLY_INPUT,
    QUERY_ONLY_INPUT,
    BEHAVIOR_ONLY_FULL_BUDGET_INPUT,
    PRE_RUN_QUERY_ONLY_INPUT,
)
EPS = 1e-12
FROZEN_PORTFOLIO_ORDER = ("de", "pso", "cmaes", "shade")
PRE_RUN_STATE_KEY_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "seed",
    "FE",
)


def selector_target_transform_for_mode(selector_input_mode: str) -> str:
    if selector_input_mode not in SELECTOR_INPUT_MODES:
        raise ValueError(f"unsupported Selector input mode: {selector_input_mode}")
    if selector_input_mode == PRE_RUN_QUERY_ONLY_INPUT:
        return PRE_RUN_SELECTOR_TARGET_TRANSFORM
    return SELECTOR_TARGET_TRANSFORM


@dataclass
class StatewiseSelectorModel:
    model: Pipeline
    target_algorithms: tuple[str, ...]
    feature_columns: tuple[str, ...]
    default_algorithm: str
    query_id: str
    query_protocol: str
    query_preprocessing_id: str
    sample_design_id: str
    query_feature_columns: tuple[str, ...]
    selector_input_mode: str = QUERY_FULL_INPUT
    action_budget_mode: str = QUERY_ADJUSTED_BUDGET
    selector_target_transform: str = SELECTOR_TARGET_TRANSFORM
    fit_weight_mode: str = CLUSTER_BALANCED_FIT
    protocol: str = SELECTION_REFERENCE_PROTOCOL

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"selector input is missing columns: {sorted(missing)}")
        values = np.asarray(self.model.predict(frame[list(self.feature_columns)]), dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] != len(self.target_algorithms):
            raise ValueError("selector prediction width does not match target algorithms")
        return values

    def select_one(self, features: dict[str, Any]) -> tuple[str, dict[str, float], float]:
        started = perf_counter()
        frame = pd.DataFrame([{column: features[column] for column in self.feature_columns}])
        scores = self.predict_scores(frame)[0]
        score_map = dict(zip(self.target_algorithms, scores.astype(float), strict=True))
        selected = min(self.target_algorithms, key=lambda algorithm: score_map[algorithm])
        return selected, score_map, float(perf_counter() - started)


def _frozen_portfolio_order(values: pd.Series | np.ndarray) -> tuple[str, ...]:
    observed = {str(value) for value in values}
    if observed != set(FROZEN_PORTFOLIO_ORDER):
        raise ValueError(
            "portfolio algorithms must be exactly de,pso,cmaes,shade; "
            f"observed={sorted(observed)}"
        )
    return FROZEN_PORTFOLIO_ORDER


def selector_feature_columns(
    query_spec: LandscapeQuerySpec,
    selector_input_mode: str,
) -> tuple[str, ...]:
    """Return the fixed input columns for one Selector information condition."""
    if selector_input_mode not in SELECTOR_INPUT_MODES:
        raise ValueError(f"unsupported Selector input mode: {selector_input_mode}")
    if selector_input_mode == QUERY_FULL_INPUT:
        return (
            *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
            *query_spec.feature_columns,
            "remaining_budget_ratio",
        )
    if selector_input_mode in {STATE_ONLY_INPUT, BEHAVIOR_ONLY_FULL_BUDGET_INPUT}:
        return (*SELECTOR_BEHAVIOR_FEATURE_COLUMNS, "remaining_budget_ratio")
    return (*query_spec.feature_columns, "remaining_budget_ratio")


def read_action_loss_data(paths: list[Path]) -> pd.DataFrame:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("action_losses.parquet")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(f"missing state-action loss input: {path}")
    if not files:
        raise ValueError("no state-action loss parquet files found")
    return pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)


def read_behavior_data(paths: list[Path]) -> pd.DataFrame:
    files: list[tuple[Path, Path | None]] = []
    for path in paths:
        if path.is_dir():
            files.extend((file, path) for file in sorted(path.rglob("behavior.parquet")))
        elif path.exists():
            files.append((path, None))
        else:
            raise FileNotFoundError(f"missing behavior input: {path}")
    if not files:
        raise ValueError("no behavior parquet files found")
    frames = []
    for file, input_root in files:
        frame = pq.read_table(file).to_pandas()
        if "split" not in frame.columns:
            frame.insert(0, "split", _behavior_split_from_path(file, input_root))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if "algorithm" in frame.columns and "prefix_algorithm" not in frame.columns:
        frame = frame.rename(columns={"algorithm": "prefix_algorithm"})
    return frame


def _behavior_split_from_path(file: Path, input_root: Path | None) -> str:
    candidates: list[str] = []
    if input_root is not None:
        relative = file.relative_to(input_root)
        if len(relative.parts) > 1:
            candidates.append(relative.parts[0])
        candidates.append(input_root.name)
    candidates.extend(file.parts)
    supported = {"bbob_train", "bbob_validation", "cec2017_test", "cec2022_test"}
    matches = [value for value in candidates if value in supported]
    if not matches:
        raise ValueError(
            "behavior input has no split column and its path does not identify a supported split: "
            f"{file}"
        )
    if len(set(matches)) != 1:
        raise ValueError(f"behavior input path has ambiguous split components: {file}")
    return matches[0]


def read_query_feature_data(paths: list[Path]) -> pd.DataFrame:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("features.parquet")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(f"missing query feature input: {path}")
    if not files:
        raise ValueError("no query feature parquet files found")
    return pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)


def prepare_state_matrix(
    action_losses: pd.DataFrame,
    *,
    behavior: pd.DataFrame,
    query_features: pd.DataFrame | None,
    query_spec: LandscapeQuerySpec,
    action_budget_mode: str = QUERY_ADJUSTED_BUDGET,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if action_budget_mode not in {QUERY_ADJUSTED_BUDGET, FULL_REMAINING_BUDGET}:
        raise ValueError(f"unsupported action budget mode: {action_budget_mode}")
    if action_budget_mode == QUERY_ADJUSTED_BUDGET and query_features is None:
        raise ValueError("query-adjusted state preparation requires query features")
    required = {
        *STATE_KEY_COLUMNS,
        "prefix_scope",
        "action_budget_mode",
        "FE_ratio",
        "FE_total",
        *SAMPLING_METADATA_COLUMNS,
        "sample_design_id",
        "sample_design_protocol",
        "FE_prefix",
        "FE_query",
        "FE_no_query_optimization",
        "FE_action_optimization",
        "remaining_budget_ratio",
        "benchmark_reference_value",
        "success_gap_target",
        "failure_loss_cap",
        "log10_gap_floor",
        "log10_gap_cap",
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "runtime_no_query_handoff",
        "runtime_no_query_optimization",
        "runtime_no_query_handoff_repetitions",
        "runtime_no_query_optimization_repetitions",
        "no_query_transition_mode",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "skip_timed_out",
        "skip_path_completed",
        "no_query_algorithm",
        "action",
        "target_algorithm",
        "transition_mode",
        "action_status",
        "action_loss",
        "action_loss_raw",
        "action_loss_norm",
        "log10_action_loss",
        "selector_target_loss",
        "loss_gap_raw",
        "loss_gap_norm",
        "runtime_handoff",
        "runtime_action_optimization",
        "runtime_handoff_repetitions",
        "runtime_action_optimization_repetitions",
        "prefix_first_hit_FE",
        "continuation_first_hit_FE",
        "observed_first_hit_FE",
        "target_hit_observed",
        "target_hit_before_failure",
        "endpoint_success",
        "first_hit_FE",
        "success",
        "planned_FE",
        "effective_FE",
        "timed_out",
        "path_completed",
        "action_outcome_execution_count",
        "action_runtime_role",
        "timing_repetitions",
        "timing_repetition_indices",
        "timing_order_protocol",
        "execution_order_repetitions",
        "skip_execution_order_repetitions",
        "best_observed_algorithm",
        "best_observed_loss",
        "action_loss_protocol",
        "performance_value_mode",
        "performance_loss_mode",
    }
    missing = required.difference(action_losses.columns)
    if missing:
        raise ValueError(f"state-action loss input is missing columns: {sorted(missing)}")
    if set(action_losses["action_loss_protocol"].astype(str)) != {ACTION_LOSS_PROTOCOL}:
        raise ValueError("state-action loss inputs use an unsupported continuation protocol")
    if set(action_losses["prefix_scope"].astype(str)) != {"all_portfolio"}:
        raise ValueError("formal Selector fitting requires all-prefix action outcomes")
    if set(action_losses["action_budget_mode"].astype(str)) != {action_budget_mode}:
        raise ValueError("state-action loss input uses the wrong action budget mode")
    if action_budget_mode == QUERY_ADJUSTED_BUDGET:
        if set(action_losses["sample_design_id"].astype(str)) != {query_spec.sample_design_id}:
            raise ValueError(f"{query_spec.query_id} is paired with the wrong query-FE action-loss table")
        if set(action_losses["sample_design_protocol"].astype(str)) != {query_spec.sample_design.protocol}:
            raise ValueError(f"{query_spec.query_id} is paired with an unsupported sample protocol")
    else:
        if set(action_losses["sample_design_id"].astype(str)) != {NOT_APPLICABLE}:
            raise ValueError("full-remaining action losses must use sample_design_id=not_applicable")
        if (action_losses["FE_query"].astype(int) != 0).any():
            raise ValueError("full-remaining action losses must use FE_query=0")
    key = list(STATE_KEY_COLUMNS)
    if action_losses.duplicated(key + ["target_algorithm"]).any():
        raise ValueError("state-action loss input contains duplicate state/algorithm rows")
    numeric_losses = action_losses[
        [
            "action_loss",
            "action_loss_norm",
            "log10_action_loss",
            "selector_target_loss",
            "best_observed_loss",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_losses).all():
        raise ValueError("state-action losses and diagnostics must be finite")
    runtime_columns = (
        "runtime_no_query_handoff",
        "runtime_no_query_optimization",
        "runtime_handoff",
        "runtime_action_optimization",
    )
    runtime_values = action_losses[list(runtime_columns)].to_numpy(dtype=float)
    if not np.isfinite(runtime_values).all() or (runtime_values < 0.0).any():
        raise ValueError("state-action runtimes must be finite and non-negative")
    _validate_action_timing_repetitions(action_losses)
    _validate_action_ert_fields(action_losses)
    portfolio = _frozen_portfolio_order(action_losses["target_algorithm"].astype(str).unique())
    if len(portfolio) != 4:
        raise ValueError("state-action loss input must contain exactly four unique portfolio algorithms")
    if not action_losses["prefix_algorithm"].astype(str).isin(portfolio).all():
        raise ValueError("every prefix algorithm must belong to the four-action portfolio")
    counts = action_losses.groupby(key, dropna=False)["target_algorithm"].nunique()
    if not bool((counts == len(portfolio)).all()):
        raise ValueError("every shared state must contain one observed loss for every portfolio algorithm")
    observed_sets = action_losses.groupby(key, dropna=False)["target_algorithm"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    if not bool((observed_sets == frozenset(portfolio)).all()):
        raise ValueError("portfolio algorithms must be identical across shared states")
    expected_actions = action_losses["target_algorithm"].astype(str).where(
        action_losses["target_algorithm"].astype(str) != action_losses["prefix_algorithm"].astype(str),
        "continue_current",
    )
    if not bool((action_losses["action"].astype(str) == expected_actions).all()):
        raise ValueError("action must be continue_current exactly when target_algorithm equals prefix_algorithm")
    expected_transition = np.where(
        action_losses["target_algorithm"].astype(str) == action_losses["prefix_algorithm"].astype(str),
        "native_optimizer_state",
        "population_transfer_initialization",
    )
    if not np.array_equal(action_losses["transition_mode"].astype(str).to_numpy(), expected_transition):
        raise ValueError("transition_mode does not match target_algorithm and prefix_algorithm")
    expected_zero_handoff = action_losses["transition_mode"].astype(str) == "native_optimizer_state"
    if not bool((action_losses.loc[expected_zero_handoff, "runtime_handoff"].astype(float) == 0.0).all()):
        raise ValueError("native continuation must have zero handoff runtime")
    if not bool(
        (
            action_losses["no_query_algorithm"].astype(str)
            == action_losses["prefix_algorithm"].astype(str)
        ).all()
    ):
        raise ValueError("raw no_query_algorithm must equal prefix_algorithm")
    if action_losses["runtime_no_query_handoff"].astype(float).ne(0.0).any():
        raise ValueError("raw native no-query continuation must have zero handoff runtime")
    if set(action_losses["no_query_transition_mode"].astype(str)) != {"native_optimizer_state"}:
        raise ValueError("raw no-query continuation must preserve native optimizer state")
    expected_fe_ratio = action_losses["FE"].astype(float) / action_losses["FE_total"].astype(float)
    if not np.array_equal(
        action_losses["FE_ratio"].to_numpy(dtype=float),
        expected_fe_ratio.to_numpy(dtype=float),
    ):
        raise ValueError("action-loss FE_ratio must equal the actual FE / FE_total")
    if not np.array_equal(
        action_losses["FE_prefix"].to_numpy(dtype=int),
        action_losses["FE"].to_numpy(dtype=int),
    ):
        raise ValueError("FE_prefix must equal the sampled trajectory FE")
    expected_no_query_budget = (
        action_losses["FE_total"].to_numpy(dtype=int)
        - action_losses["FE_prefix"].to_numpy(dtype=int)
    )
    if not np.array_equal(
        action_losses["FE_no_query_optimization"].to_numpy(dtype=int),
        expected_no_query_budget,
    ):
        raise ValueError("FE_no_query_optimization must equal FE_total - FE_prefix")
    expected_action_budget = expected_no_query_budget - action_losses["FE_query"].to_numpy(dtype=int)
    if not np.array_equal(
        action_losses["FE_action_optimization"].to_numpy(dtype=int),
        expected_action_budget,
    ):
        raise ValueError("FE_action_optimization must equal FE_total - FE_prefix - FE_query")
    expected_remaining_ratio = expected_action_budget / action_losses["FE_total"].to_numpy(dtype=float)
    if not np.allclose(
        action_losses["remaining_budget_ratio"].to_numpy(dtype=float),
        expected_remaining_ratio,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("remaining_budget_ratio must represent the action optimization budget")

    metadata_columns = [
        *key,
        "prefix_scope",
        "FE_ratio",
        "FE_total",
        *SAMPLING_METADATA_COLUMNS,
        "action_budget_mode",
        "sample_design_id",
        "sample_design_protocol",
        "FE_prefix",
        "FE_query",
        "FE_no_query_optimization",
        "FE_action_optimization",
        "remaining_budget_ratio",
        "benchmark_reference_value",
        "success_gap_target",
        "failure_loss_cap",
        "log10_gap_floor",
        "log10_gap_cap",
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "runtime_no_query_handoff",
        "runtime_no_query_optimization",
        "runtime_no_query_handoff_repetitions",
        "runtime_no_query_optimization_repetitions",
        "no_query_transition_mode",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "skip_timed_out",
        "skip_path_completed",
        "prefix_first_hit_FE",
        "no_query_algorithm",
        "best_observed_algorithm",
        "best_observed_loss",
        "action_outcome_execution_count",
        "action_runtime_role",
        "timing_repetitions",
        "timing_repetition_indices",
        "timing_order_protocol",
        "skip_execution_order_repetitions",
    ]
    _validate_state_metadata(action_losses, key=key, metadata_columns=metadata_columns)
    states = action_losses.drop_duplicates(key)[metadata_columns].copy()
    portfolio_index = {algorithm: index for index, algorithm in enumerate(portfolio)}
    ordered = action_losses.assign(
        _portfolio_index=action_losses["target_algorithm"].astype(str).map(portfolio_index)
    ).sort_values([*key, "action_loss", "_portfolio_index"])
    computed_best = ordered.drop_duplicates(key)[key + ["target_algorithm", "action_loss"]].rename(
        columns={
            "target_algorithm": "computed_best_observed_algorithm",
            "action_loss": "computed_best_observed_loss",
        }
    )
    states = states.merge(computed_best, on=key, how="inner", validate="one_to_one")
    if not bool(
        (
            states["best_observed_algorithm"].astype(str)
            == states["computed_best_observed_algorithm"].astype(str)
        ).all()
    ):
        raise ValueError("best_observed_algorithm does not match the minimum observed action loss")
    if not np.allclose(
        states["best_observed_loss"].to_numpy(dtype=float),
        states["computed_best_observed_loss"].to_numpy(dtype=float),
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("best_observed_loss does not match the minimum observed action loss")
    state_min = action_losses.groupby(key, dropna=False)["action_loss"].transform("min").to_numpy(dtype=float)
    state_max = action_losses.groupby(key, dropna=False)["action_loss"].transform("max").to_numpy(dtype=float)
    expected_normalized = (
        action_losses["action_loss"].to_numpy(dtype=float) - state_min
    ) / np.maximum(state_max - state_min, EPS)
    if not np.allclose(
        action_losses["action_loss_norm"].to_numpy(dtype=float),
        expected_normalized,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("action_loss_norm is inconsistent with the statewise observed loss range")
    log_floor = action_losses["log10_gap_floor"].to_numpy(dtype=float)
    log_cap = action_losses["log10_gap_cap"].to_numpy(dtype=float)
    if (
        not np.isfinite(log_floor).all()
        or not np.isfinite(log_cap).all()
        or bool((log_floor <= 0.0).any())
        or bool((log_cap <= log_floor).any())
    ):
        raise ValueError("Selector log10 gap bounds must be finite, positive, and ordered")
    expected_log_loss = np.log10(
        np.minimum(
            np.maximum(action_losses["action_loss"].to_numpy(dtype=float), log_floor),
            log_cap,
        )
    )
    if not np.allclose(
        action_losses["log10_action_loss"].to_numpy(dtype=float),
        expected_log_loss,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("log10_action_loss is inconsistent with the configured gap bounds")
    continue_rows = action_losses["action"].astype(str).eq("continue_current")
    continue_count = continue_rows.groupby(
        [action_losses[column] for column in key],
        dropna=False,
    ).sum()
    if not bool((continue_count == 1).all()):
        raise ValueError("every shared state must contain exactly one continue_current action")
    continue_log_loss = pd.Series(expected_log_loss, index=action_losses.index).where(
        continue_rows
    ).groupby(
        [action_losses[column] for column in key],
        dropna=False,
    ).transform("max")
    expected_selector_target = expected_log_loss - continue_log_loss.to_numpy(dtype=float)
    if not np.allclose(
        action_losses["selector_target_loss"].to_numpy(dtype=float),
        expected_selector_target,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError(
            "selector_target_loss must be the clipped log10 gap advantage relative to continue_current"
        )
    if not np.allclose(
        action_losses.loc[continue_rows, "selector_target_loss"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("continue_current Selector target must equal zero")
    if not np.allclose(
        action_losses["action_loss"].to_numpy(dtype=float),
        np.minimum(
            np.maximum(
                action_losses["action_loss_raw"].to_numpy(dtype=float)
                - action_losses["benchmark_reference_value"].to_numpy(dtype=float),
                0.0,
            ),
            action_losses["failure_loss_cap"].to_numpy(dtype=float),
        ),
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("action_loss must equal the known-optimum gap derived from action_loss_raw")
    if not np.allclose(
        action_losses["loss_gap_raw"].to_numpy(dtype=float),
        action_losses["action_loss"].to_numpy(dtype=float),
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("loss_gap_raw must equal action_loss")
    if not np.allclose(
        action_losses["loss_skip"].to_numpy(dtype=float),
        action_losses["p_skip"].to_numpy(dtype=float),
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("loss_skip must equal p_skip")
    if not np.allclose(
        action_losses["p_skip"].to_numpy(dtype=float),
        np.minimum(
            np.maximum(
                action_losses["p_skip_raw"].to_numpy(dtype=float)
                - action_losses["benchmark_reference_value"].to_numpy(dtype=float),
                0.0,
            ),
            action_losses["failure_loss_cap"].to_numpy(dtype=float),
        ),
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("p_skip must equal the known-optimum gap derived from p_skip_raw")
    if set(action_losses["performance_value_mode"].astype(str)) != {"raw_objective"}:
        raise ValueError("performance_value_mode must be raw_objective")
    if set(action_losses["performance_loss_mode"].astype(str)) != {"known_optimum_gap"}:
        raise ValueError("performance_loss_mode must be known_optimum_gap")
    states = states.drop(columns=["computed_best_observed_algorithm", "computed_best_observed_loss"])
    raw = action_losses.pivot(index=key, columns="target_algorithm", values="action_loss").reset_index()
    raw_loss = action_losses.pivot(index=key, columns="target_algorithm", values="action_loss_raw").reset_index()
    selector_target = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="selector_target_loss",
    ).reset_index()
    runtimes = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="runtime_action_optimization",
    ).reset_index()
    handoff_runtimes = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="runtime_handoff",
    ).reset_index()
    action_runtime_repetitions = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="runtime_action_optimization_repetitions",
    ).reset_index()
    handoff_runtime_repetitions = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="runtime_handoff_repetitions",
    ).reset_index()
    execution_order_repetitions = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="execution_order_repetitions",
    ).reset_index()
    first_hit = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="first_hit_FE",
    ).reset_index()
    continuation_first_hit = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="continuation_first_hit_FE",
    ).reset_index()
    success = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="success",
    ).reset_index()
    planned_fe = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="planned_FE",
    ).reset_index()
    effective_fe = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="effective_FE",
    ).reset_index()
    timed_out = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="timed_out",
    ).reset_index()
    path_completed = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="path_completed",
    ).reset_index()
    raw = raw.rename(columns={algorithm: f"observed_loss_{algorithm}" for algorithm in portfolio})
    raw_loss = raw_loss.rename(columns={algorithm: f"observed_loss_raw_{algorithm}" for algorithm in portfolio})
    selector_target = selector_target.rename(
        columns={algorithm: f"target_selector_loss_{algorithm}" for algorithm in portfolio}
    )
    runtimes = runtimes.rename(columns={algorithm: f"runtime_action_optimization_{algorithm}" for algorithm in portfolio})
    handoff_runtimes = handoff_runtimes.rename(
        columns={algorithm: f"runtime_handoff_{algorithm}" for algorithm in portfolio}
    )
    action_runtime_repetitions = action_runtime_repetitions.rename(
        columns={
            algorithm: f"runtime_action_optimization_repetitions_{algorithm}"
            for algorithm in portfolio
        }
    )
    handoff_runtime_repetitions = handoff_runtime_repetitions.rename(
        columns={
            algorithm: f"runtime_handoff_repetitions_{algorithm}"
            for algorithm in portfolio
        }
    )
    execution_order_repetitions = execution_order_repetitions.rename(
        columns={
            algorithm: f"execution_order_repetitions_{algorithm}"
            for algorithm in portfolio
        }
    )
    first_hit = first_hit.rename(
        columns={algorithm: f"first_hit_FE_{algorithm}" for algorithm in portfolio}
    )
    continuation_first_hit = continuation_first_hit.rename(
        columns={
            algorithm: f"continuation_first_hit_FE_{algorithm}"
            for algorithm in portfolio
        }
    )
    success = success.rename(
        columns={algorithm: f"success_{algorithm}" for algorithm in portfolio}
    )
    planned_fe = planned_fe.rename(
        columns={algorithm: f"planned_FE_{algorithm}" for algorithm in portfolio}
    )
    effective_fe = effective_fe.rename(
        columns={algorithm: f"effective_FE_{algorithm}" for algorithm in portfolio}
    )
    timed_out = timed_out.rename(
        columns={algorithm: f"timed_out_{algorithm}" for algorithm in portfolio}
    )
    path_completed = path_completed.rename(
        columns={algorithm: f"path_completed_{algorithm}" for algorithm in portfolio}
    )
    states = states.merge(raw, on=key, how="inner", validate="one_to_one")
    states = states.merge(raw_loss, on=key, how="inner", validate="one_to_one")
    states = states.merge(selector_target, on=key, how="inner", validate="one_to_one")
    states = states.merge(runtimes, on=key, how="inner", validate="one_to_one")
    states = states.merge(handoff_runtimes, on=key, how="inner", validate="one_to_one")
    states = states.merge(
        action_runtime_repetitions,
        on=key,
        how="inner",
        validate="one_to_one",
    )
    states = states.merge(
        handoff_runtime_repetitions,
        on=key,
        how="inner",
        validate="one_to_one",
    )
    states = states.merge(
        execution_order_repetitions,
        on=key,
        how="inner",
        validate="one_to_one",
    )
    states = states.merge(first_hit, on=key, how="inner", validate="one_to_one")
    states = states.merge(
        continuation_first_hit,
        on=key,
        how="inner",
        validate="one_to_one",
    )
    states = states.merge(success, on=key, how="inner", validate="one_to_one")
    states = states.merge(planned_fe, on=key, how="inner", validate="one_to_one")
    states = states.merge(effective_fe, on=key, how="inner", validate="one_to_one")
    states = states.merge(timed_out, on=key, how="inner", validate="one_to_one")
    states = states.merge(path_completed, on=key, how="inner", validate="one_to_one")
    states = _join_selector_inputs(
        states=states,
        behavior=behavior,
        query_features=query_features,
        query_spec=query_spec,
    )
    states["default_algorithm"] = _compute_problem_level_sbs(states, action_losses)
    return states.sort_values(key).reset_index(drop=True), portfolio


def _compute_problem_level_sbs(
    states: pd.DataFrame,
    action_losses: pd.DataFrame,
) -> pd.Series:
    """Derive the single-best algorithm across all training problems from mean p_skip."""
    continue_rows = action_losses["action"].astype(str).eq("continue_current")
    mean_p_skip = (
        action_losses.loc[continue_rows]
        .groupby("prefix_algorithm")["p_skip"]
        .mean()
        .sort_values()
    )
    sbs = str(mean_p_skip.index[0])
    return pd.Series(sbs, index=states.index)


def prepare_pre_run_state_matrix(
    action_losses: pd.DataFrame,
    *,
    query_features: pd.DataFrame,
    query_spec: LandscapeQuerySpec,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Prepare the independent FE=0 Traditional-AAS outcome matrix."""
    key = list(PRE_RUN_STATE_KEY_COLUMNS)
    required = {
        *key,
        "FE_prefix",
        "FE_ratio",
        "FE_total",
        "action_budget_mode",
        "sample_design_id",
        "sample_design_protocol",
        "FE_query",
        "FE_action_optimization",
        "remaining_budget_ratio",
        "benchmark_reference_value",
        "success_gap_target",
        "failure_loss_cap",
        "action",
        "target_algorithm",
        "transition_mode",
        "action_status",
        "action_loss",
        "action_loss_raw",
        "action_loss_norm",
        "log10_gap_floor",
        "log10_gap_cap",
        "log10_action_loss",
        "selector_target_loss",
        "runtime_handoff",
        "runtime_fresh_initialization",
        "runtime_action_optimization",
        "runtime_handoff_repetitions",
        "runtime_fresh_initialization_repetitions",
        "runtime_action_optimization_repetitions",
        "execution_order_repetitions",
        "timing_repetitions",
        "timing_repetition_indices",
        "timing_order_protocol",
        "observed_first_hit_FE",
        "target_hit_observed",
        "target_hit_before_failure",
        "endpoint_success",
        "first_hit_FE",
        "success",
        "planned_FE",
        "effective_FE",
        "timed_out",
        "path_completed",
        "action_outcome_execution_count",
        "action_runtime_role",
        "best_observed_algorithm",
        "best_observed_loss",
        "action_loss_protocol",
    }
    missing = sorted(required.difference(action_losses.columns))
    if missing:
        raise ValueError(f"pre-run action outcomes are missing columns: {missing}")
    if set(action_losses["action_budget_mode"].astype(str)) != {
        PRE_RUN_QUERY_ADJUSTED_BUDGET
    }:
        raise ValueError("pre-run outcomes use the wrong action budget mode")
    if set(action_losses["sample_design_id"].astype(str)) != {
        query_spec.sample_design_id
    }:
        raise ValueError("pre-run outcomes use the wrong query sample design")
    if set(action_losses["sample_design_protocol"].astype(str)) != {
        query_spec.sample_design.protocol
    }:
        raise ValueError("pre-run outcomes use the wrong query sample protocol")
    if not action_losses["FE"].astype(int).eq(0).all() or not action_losses[
        "FE_prefix"
    ].astype(int).eq(0).all():
        raise ValueError("pre-run outcomes must use FE=FE_prefix=0")
    expected_budget = (
        action_losses["FE_total"].to_numpy(dtype=int)
        - action_losses["FE_query"].to_numpy(dtype=int)
    )
    if not np.array_equal(
        action_losses["FE_action_optimization"].to_numpy(dtype=int),
        expected_budget,
    ):
        raise ValueError("pre-run action budget must equal FE_total - FE_query")
    if set(action_losses["transition_mode"].astype(str)) != {
        "fresh_optimizer_initialization"
    }:
        raise ValueError("pre-run actions must use fresh optimizer initialization")
    if not np.allclose(
        action_losses["runtime_handoff"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("FE=0 fresh initialization must not be recorded as handoff runtime")
    fresh_runtime = action_losses["runtime_fresh_initialization"].to_numpy(dtype=float)
    if not np.isfinite(fresh_runtime).all() or bool((fresh_runtime < 0.0).any()):
        raise ValueError("pre-run fresh-initialization runtime must be finite and non-negative")
    if not (
        action_losses["action"].astype(str)
        == action_losses["target_algorithm"].astype(str)
    ).all():
        raise ValueError("pre-run action must equal target_algorithm")
    if set(action_losses["timing_repetitions"].astype(int)) != {TIMING_REPETITIONS}:
        raise ValueError("pre-run action outcomes must be executed exactly once")
    _validate_action_runtime_role(action_losses)
    if set(action_losses["timing_order_protocol"].astype(str)) != {
        EXECUTION_ORDER_PROTOCOL
    }:
        raise ValueError("pre-run actions use the wrong single-execution order protocol")
    if action_losses.duplicated(key + ["target_algorithm"]).any():
        raise ValueError("pre-run outcomes contain duplicate state/algorithm rows")
    _validate_pre_run_ert_fields(action_losses)
    pre_run_floor = action_losses["log10_gap_floor"].to_numpy(dtype=float)
    pre_run_cap = action_losses["log10_gap_cap"].to_numpy(dtype=float)
    if (
        not np.isfinite(pre_run_floor).all()
        or not np.isfinite(pre_run_cap).all()
        or bool((pre_run_floor <= 0.0).any())
        or bool((pre_run_cap <= pre_run_floor).any())
    ):
        raise ValueError("pre-run Selector log10 bounds must be finite, positive, and ordered")
    expected_pre_run_log_loss = np.log10(
        np.minimum(
            np.maximum(
                action_losses["action_loss"].to_numpy(dtype=float),
                pre_run_floor,
            ),
            pre_run_cap,
        )
    )
    if not np.allclose(
        action_losses["log10_action_loss"].to_numpy(dtype=float),
        expected_pre_run_log_loss,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("pre-run log10_action_loss is inconsistent with configured bounds")
    if not np.allclose(
        action_losses["selector_target_loss"].to_numpy(dtype=float),
        expected_pre_run_log_loss,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("pre-run Selector target must equal absolute clipped log10 action loss")
    portfolio = _frozen_portfolio_order(action_losses["target_algorithm"].astype(str).unique())
    if len(portfolio) != 4:
        raise ValueError("pre-run outcomes require exactly four portfolio algorithms")
    counts = action_losses.groupby(key, dropna=False)["target_algorithm"].nunique()
    if not bool((counts == len(portfolio)).all()):
        raise ValueError("every pre-run state must contain all four algorithm outcomes")
    for column in (
        "runtime_handoff_repetitions",
        "runtime_fresh_initialization_repetitions",
        "runtime_action_optimization_repetitions",
        "execution_order_repetitions",
    ):
        if not action_losses[column].map(lambda values: len(values) == TIMING_REPETITIONS).all():
            raise ValueError(f"{column} must contain exactly one value")

    metadata_columns = [
        *key,
        "FE_prefix",
        "FE_ratio",
        "FE_total",
        "action_budget_mode",
        "sample_design_id",
        "sample_design_protocol",
        "FE_query",
        "FE_action_optimization",
        "remaining_budget_ratio",
        "benchmark_reference_value",
        "success_gap_target",
        "failure_loss_cap",
        "log10_gap_floor",
        "log10_gap_cap",
        "action_outcome_execution_count",
        "action_runtime_role",
        "timing_repetitions",
        "timing_repetition_indices",
        "timing_order_protocol",
        "best_observed_algorithm",
        "best_observed_loss",
    ]
    _validate_state_metadata(action_losses, key=key, metadata_columns=metadata_columns)
    states = action_losses.drop_duplicates(key)[metadata_columns].copy()
    value_columns = {
        "action_loss": "observed_loss",
        "action_loss_raw": "observed_loss_raw",
        "selector_target_loss": "target_selector_loss",
        "runtime_handoff": "runtime_handoff",
        "runtime_fresh_initialization": "runtime_fresh_initialization",
        "runtime_action_optimization": "runtime_action_optimization",
        "runtime_handoff_repetitions": "runtime_handoff_repetitions",
        "runtime_fresh_initialization_repetitions": (
            "runtime_fresh_initialization_repetitions"
        ),
        "runtime_action_optimization_repetitions": "runtime_action_optimization_repetitions",
        "execution_order_repetitions": "execution_order_repetitions",
        "first_hit_FE": "first_hit_FE",
        "observed_first_hit_FE": "observed_first_hit_FE",
        "target_hit_observed": "target_hit_observed",
        "target_hit_before_failure": "target_hit_before_failure",
        "endpoint_success": "endpoint_success",
        "success": "success",
        "planned_FE": "planned_FE",
        "effective_FE": "effective_FE",
        "timed_out": "timed_out",
        "path_completed": "path_completed",
    }
    for source, prefix in value_columns.items():
        pivoted = action_losses.pivot(
            index=key,
            columns="target_algorithm",
            values=source,
        ).reset_index()
        pivoted = pivoted.rename(
            columns={algorithm: f"{prefix}_{algorithm}" for algorithm in portfolio}
        )
        states = states.merge(pivoted, on=key, how="inner", validate="one_to_one")
    states["default_algorithm"] = NOT_APPLICABLE
    states = _join_pre_run_query_inputs(
        states=states,
        query_features=query_features,
        query_spec=query_spec,
    )
    return states.sort_values(key).reset_index(drop=True), portfolio


def _join_pre_run_query_inputs(
    *,
    states: pd.DataFrame,
    query_features: pd.DataFrame,
    query_spec: LandscapeQuerySpec,
) -> pd.DataFrame:
    query_key = ["split", "problem_id", "function_id", "family", "dimension"]
    required = {
        *query_key,
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "sample_design_id",
        "benchmark_reference_value",
        "success_gap_target",
        "query_success",
        "query_first_hit_offset",
        "query_best_gap",
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_query_feature_computation",
        "runtime_query",
        "query_feature_columns",
        *query_spec.feature_columns,
    }
    missing = sorted(required.difference(query_features.columns))
    if missing:
        raise ValueError(f"pre-run query features are missing columns: {missing}")
    if query_features.duplicated(query_key).any():
        raise ValueError("pre-run query features contain duplicate problem keys")
    if set(query_features["query_id"].astype(str)) != {query_spec.query_id}:
        raise ValueError("pre-run query features use the wrong query_id")
    columns = list(required)
    joined = states.merge(
        query_features[columns],
        on=query_key,
        how="left",
        validate="many_to_one",
        suffixes=("", "_query"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("pre-run outcomes and query features do not have complete coverage")
    joined = joined.drop(columns="_merge")
    for column in ("sample_design_id", "benchmark_reference_value", "success_gap_target"):
        query_column = f"{column}_query"
        if column == "sample_design_id":
            matches = joined[column].astype(str) == joined[query_column].astype(str)
        else:
            matches = np.isclose(
                joined[column].to_numpy(dtype=float),
                joined[query_column].to_numpy(dtype=float),
                rtol=0.0,
                atol=EPS,
            )
        if not bool(np.all(matches)):
            raise ValueError(f"pre-run outcomes and query features disagree on {column}")
        joined = joined.drop(columns=query_column)
    return joined


def fit_selector_with_cross_family_predictions(
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    query_spec: LandscapeQuerySpec,
    *,
    selector_input_mode: str = QUERY_FULL_INPUT,
) -> tuple[StatewiseSelectorModel, np.ndarray, str]:
    defaults = tuple(sorted(states["default_algorithm"].astype(str).unique()))
    if len(defaults) != 1:
        raise ValueError("selector training states must use one train-derived SBS default")
    feature_columns = selector_feature_columns(query_spec, selector_input_mode)
    x = states[list(feature_columns)]
    all_missing = [column for column in feature_columns if x[column].isna().all()]
    if all_missing:
        raise ValueError(f"selector training has entirely missing input columns: {all_missing}")
    y = states[[f"target_selector_loss_{algorithm}" for algorithm in portfolio]].to_numpy(
        dtype=float
    )
    group_column = "cv_group_id" if "cv_group_id" in states.columns else "function_id"
    unique_groups = states[group_column].astype(str).nunique()
    if unique_groups >= 2:
        cross_predictions = predict_with_main_prefix_cross_family_fits(
            training_states=states,
            prediction_states=states,
            portfolio=portfolio,
            query_spec=query_spec,
            selector_input_mode=selector_input_mode,
        )
        prediction_source = "cross_cv_group"
    else:
        diagnostic_model = _make_model()
        fit_pipeline_with_weights(
            diagnostic_model,
            x,
            y,
            cluster_balanced_row_weights(states),
        )
        cross_predictions = np.asarray(diagnostic_model.predict(x), dtype=float)
        prediction_source = "in_sample_insufficient_cv_groups"
    if not np.isfinite(cross_predictions).all():
        raise ValueError("cross-CV-group selector predictions contain missing or non-finite values")
    final_model = _make_model()
    fit_pipeline_with_weights(
        final_model,
        x,
        y,
        cluster_balanced_row_weights(states),
    )
    observed_budget_modes = tuple(sorted(states["action_budget_mode"].astype(str).unique()))
    observed_sample_designs = tuple(sorted(states["sample_design_id"].astype(str).unique()))
    if len(observed_budget_modes) != 1 or len(observed_sample_designs) != 1:
        raise ValueError("Selector states must contain one action-budget and sample-design condition")
    if selector_input_mode == BEHAVIOR_ONLY_FULL_BUDGET_INPUT:
        selector_protocol = BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL
        model_query_id = NOT_APPLICABLE
        model_query_protocol = NOT_APPLICABLE
        model_preprocessing_id = NOT_APPLICABLE
    elif selector_input_mode == PRE_RUN_QUERY_ONLY_INPUT:
        selector_protocol = PRE_RUN_SELECTION_REFERENCE_PROTOCOL
        model_query_id = query_spec.query_id
        model_query_protocol = query_spec.protocol
        model_preprocessing_id = query_spec.preprocessing_id
    else:
        selector_protocol = (
            QUERY_ONLY_SELECTION_REFERENCE_PROTOCOL
            if selector_input_mode == QUERY_ONLY_INPUT
            else SELECTION_REFERENCE_PROTOCOL
        )
        model_query_id = query_spec.query_id
        model_query_protocol = query_spec.protocol
        model_preprocessing_id = query_spec.preprocessing_id
    selector_model = StatewiseSelectorModel(
        model=final_model,
        target_algorithms=portfolio,
        feature_columns=feature_columns,
        default_algorithm=defaults[0],
        query_id=model_query_id,
        query_protocol=model_query_protocol,
        query_preprocessing_id=model_preprocessing_id,
        sample_design_id=observed_sample_designs[0],
        query_feature_columns=(
            query_spec.feature_columns
            if selector_input_mode
            in {QUERY_FULL_INPUT, QUERY_ONLY_INPUT, PRE_RUN_QUERY_ONLY_INPUT}
            else ()
        ),
        selector_input_mode=selector_input_mode,
        action_budget_mode=observed_budget_modes[0],
        selector_target_transform=selector_target_transform_for_mode(selector_input_mode),
        fit_weight_mode=CLUSTER_BALANCED_FIT,
        protocol=selector_protocol,
    )
    return selector_model, cross_predictions, prediction_source


def predict_with_main_prefix_cross_family_fits(
    *,
    training_states: pd.DataFrame,
    prediction_states: pd.DataFrame,
    portfolio: tuple[str, ...],
    query_spec: LandscapeQuerySpec,
    selector_input_mode: str = QUERY_FULL_INPUT,
) -> np.ndarray:
    """Predict each state with a model that excludes its complete CV group.

    The CV grouping field is ``cv_group_id`` (set equal to ``function_id`` in the
    main experiment), not ``family``.  This ensures the selector is evaluated
    on unseen functions rather than unseen landscape families.
    """
    group_column = "cv_group_id" if "cv_group_id" in training_states.columns else "function_id"
    if group_column not in prediction_states.columns:
        raise ValueError(f"prediction states are missing the CV group column: {group_column}")
    if training_states.empty or prediction_states.empty:
        raise ValueError("cross-CV-group selector prediction requires non-empty state tables")
    feature_columns = selector_feature_columns(query_spec, selector_input_mode)
    missing_training = set(feature_columns).difference(training_states.columns)
    missing_prediction = set(feature_columns).difference(prediction_states.columns)
    if missing_training:
        raise ValueError(f"selector training states are missing columns: {sorted(missing_training)}")
    if missing_prediction:
        raise ValueError(f"selector prediction states are missing columns: {sorted(missing_prediction)}")
    target_columns = [f"target_selector_loss_{algorithm}" for algorithm in portfolio]
    missing_targets = set(target_columns).difference(training_states.columns)
    if missing_targets:
        raise ValueError(f"selector training states are missing targets: {sorted(missing_targets)}")

    training_groups = training_states[group_column].astype(str).to_numpy()
    prediction_groups = prediction_states[group_column].astype(str).to_numpy()
    unique_training_groups = np.unique(training_groups)
    if len(unique_training_groups) < 2:
        raise ValueError("cross-CV-group selector prediction requires at least two training groups")
    unknown_prediction_groups = sorted(
        set(prediction_groups).difference(unique_training_groups)
    )
    if unknown_prediction_groups:
        raise ValueError(
            "cross-CV-group prediction states contain groups absent from training: "
            f"{unknown_prediction_groups}"
        )

    x_training = training_states[list(feature_columns)]
    y_training = training_states[target_columns].to_numpy(dtype=float)
    x_prediction = prediction_states[list(feature_columns)]
    predictions = np.full((len(prediction_states), len(portfolio)), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=min(len(unique_training_groups), len(unique_training_groups)))
    for fit_indices, held_indices in splitter.split(
        x_training,
        y_training,
        groups=training_groups,
    ):
        fit_groups = set(training_groups[fit_indices])
        held_groups = set(training_groups[held_indices])
        if fit_groups.intersection(held_groups):
            raise RuntimeError("selector CV-group cross-fitting fold contains group overlap")
        prediction_indices = np.flatnonzero(
            np.isin(prediction_groups, tuple(sorted(held_groups)))
        )
        if len(prediction_indices) == 0:
            continue
        fold_model = _make_model()
        fit_pipeline_with_weights(
            fold_model,
            x_training.iloc[fit_indices],
            y_training[fit_indices],
            cluster_balanced_row_weights(training_states.iloc[fit_indices]),
        )
        predictions[prediction_indices] = fold_model.predict(
            x_prediction.iloc[prediction_indices]
        )
    if not np.isfinite(predictions).all():
        raise ValueError("cross-CV-group selector predictions contain missing or non-finite values")
    return predictions


def selection_rows(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    predictions: np.ndarray,
    prediction_source: str,
    runtime_selection: float,
    selector_input_mode: str = QUERY_FULL_INPUT,
    selection_reference_protocol: str = SELECTION_REFERENCE_PROTOCOL,
) -> pd.DataFrame:
    if predictions.shape != (len(states), len(portfolio)):
        raise ValueError("selector prediction matrix has an unexpected shape")
    output = states[
        [
            *STATE_KEY_COLUMNS,
            "FE_ratio",
            "FE_total",
            *SAMPLING_METADATA_COLUMNS,
            "prefix_scope",
            "action_budget_mode",
            "sample_design_id",
            "FE_prefix",
            "FE_query",
            "FE_no_query_optimization",
            "FE_action_optimization",
            "remaining_budget_ratio",
            "benchmark_reference_value",
            "p_skip",
            "p_skip_raw",
            "loss_skip",
            "runtime_no_query_handoff",
            "runtime_no_query_optimization",
            "runtime_no_query_handoff_repetitions",
            "runtime_no_query_optimization_repetitions",
            "timing_repetitions",
            "timing_repetition_indices",
            "timing_order_protocol",
            "skip_execution_order_repetitions",
            "prefix_first_hit_FE",
            "skip_status",
            "skip_failure_type",
            "skip_failure_message",
            "skip_prefix_first_hit_FE",
            "skip_continuation_first_hit_FE",
            "skip_observed_first_hit_FE",
            "skip_target_hit_observed",
            "skip_target_hit_before_failure",
            "skip_endpoint_success",
            "skip_first_hit_FE",
            "skip_success",
            "skip_planned_FE",
            "skip_effective_FE",
            "skip_timed_out",
            "skip_path_completed",
            "no_query_transition_mode",
            "runtime_query_sampling",
            "runtime_query_evaluation",
            "runtime_query_feature_computation",
            "runtime_query",
            "success_gap_target",
            "query_success",
            "query_first_hit_offset",
            "query_best_gap",
            "feature_status",
            "feature_failure",
            "default_algorithm",
            "no_query_algorithm",
            "best_observed_algorithm",
            "best_observed_loss",
        ]
    ].copy()
    output["skip_continuation_planned_FE"] = output["skip_planned_FE"].astype(int)
    output["skip_continuation_effective_FE"] = output["skip_effective_FE"].astype(int)
    output["skip_planned_FE"] = output["FE_total"].astype(int)
    output["skip_effective_FE"] = (
        output["FE_prefix"].astype(int)
        + output["skip_continuation_effective_FE"].astype(int)
    )
    output["sbs_algorithm"] = output["default_algorithm"].astype(str)
    for index, algorithm in enumerate(portfolio):
        output[f"predicted_selector_target_{algorithm}"] = predictions[:, index]
        output[f"observed_loss_{algorithm}"] = states[f"observed_loss_{algorithm}"].to_numpy(dtype=float)
    selected_algorithms = []
    selected_losses = []
    selected_raw_losses: list[float] = []
    selected_scores = []
    selected_runtimes = []
    selected_handoff_runtimes = []
    selected_runtime_repetitions: list[list[float]] = []
    selected_handoff_repetitions: list[list[float]] = []
    selected_order_repetitions: list[list[int]] = []
    selected_first_hits: list[int | None] = []
    selected_continuation_first_hits: list[int | None] = []
    selected_success: list[bool] = []
    selected_planned_fe: list[int] = []
    selected_effective_fe: list[int] = []
    selected_timed_out: list[bool] = []
    selected_path_completed: list[bool] = []
    best_losses = output["best_observed_loss"].to_numpy(dtype=float)
    worst_losses = states[[f"observed_loss_{algorithm}" for algorithm in portfolio]].max(axis=1).to_numpy(dtype=float)
    for row_index in range(len(output)):
        selected = min(
            portfolio,
            key=lambda algorithm: float(
                output.at[row_index, f"predicted_selector_target_{algorithm}"]
            ),
        )
        selected_algorithms.append(selected)
        selected_scores.append(
            float(output.at[row_index, f"predicted_selector_target_{selected}"])
        )
        selected_losses.append(float(output.at[row_index, f"observed_loss_{selected}"]))
        selected_raw_losses.append(float(states.at[row_index, f"observed_loss_raw_{selected}"]))
        selected_runtimes.append(float(states.at[row_index, f"runtime_action_optimization_{selected}"]))
        selected_handoff_runtimes.append(float(states.at[row_index, f"runtime_handoff_{selected}"]))
        selected_runtime_repetitions.append(
            [
                float(value)
                for value in states.at[
                    row_index,
                    f"runtime_action_optimization_repetitions_{selected}",
                ]
            ]
        )
        selected_handoff_repetitions.append(
            [
                float(value)
                for value in states.at[row_index, f"runtime_handoff_repetitions_{selected}"]
            ]
        )
        selected_order_repetitions.append(
            [
                int(value)
                for value in states.at[row_index, f"execution_order_repetitions_{selected}"]
            ]
        )
        first_hit_value = states.at[row_index, f"first_hit_FE_{selected}"]
        continuation_hit_value = states.at[
            row_index,
            f"continuation_first_hit_FE_{selected}",
        ]
        selected_first_hits.append(None if pd.isna(first_hit_value) else int(first_hit_value))
        selected_continuation_first_hits.append(
            None if pd.isna(continuation_hit_value) else int(continuation_hit_value)
        )
        selected_success.append(bool(states.at[row_index, f"success_{selected}"]))
        selected_planned_fe.append(int(states.at[row_index, f"planned_FE_{selected}"]))
        selected_effective_fe.append(int(states.at[row_index, f"effective_FE_{selected}"]))
        selected_timed_out.append(bool(states.at[row_index, f"timed_out_{selected}"]))
        selected_path_completed.append(
            bool(states.at[row_index, f"path_completed_{selected}"])
        )
    output["selected_algorithm"] = selected_algorithms
    output["selected_action"] = np.where(
        output["selected_algorithm"].astype(str) == output["prefix_algorithm"].astype(str),
        "continue_current",
        output["selected_algorithm"].astype(str),
    )
    output["selected_equals_default"] = (
        output["selected_algorithm"].astype(str) == output["default_algorithm"].astype(str)
    )
    output["selected_equals_prefix"] = (
        output["selected_algorithm"].astype(str) == output["prefix_algorithm"].astype(str)
    )
    output["selected_transition_mode"] = np.where(
        output["selected_equals_prefix"].astype(bool),
        "native_optimizer_state",
        "population_transfer_initialization",
    )
    output["handoff_required"] = ~output["selected_equals_prefix"].astype(bool)
    output["handoff_type"] = output["selected_transition_mode"].astype(str)
    output["selected_action_loss"] = selected_losses
    output["selected_action_loss_raw"] = np.asarray(selected_raw_losses, dtype=float)
    output["runtime_selected_action_optimization"] = selected_runtimes
    output["runtime_handoff"] = selected_handoff_runtimes
    output["runtime_handoff_repetitions"] = selected_handoff_repetitions
    output["runtime_selected_action_optimization_repetitions"] = selected_runtime_repetitions
    output["selected_execution_order_repetitions"] = selected_order_repetitions
    selected_hit_observed = np.asarray(
        [value is not None for value in selected_first_hits], dtype=bool
    )
    selected_completed = np.asarray(selected_path_completed, dtype=bool)
    output["selected_action_observed_first_hit_FE"] = selected_first_hits
    output["selected_action_target_hit_observed"] = selected_hit_observed
    output["selected_action_target_hit_before_failure"] = (
        selected_hit_observed & ~selected_completed
    )
    output["selected_action_endpoint_success"] = selected_hit_observed & selected_completed
    output["selected_action_first_hit_FE"] = selected_first_hits
    output["selected_continuation_first_hit_FE"] = selected_continuation_first_hits
    output["selected_action_success"] = selected_hit_observed
    output["selected_action_planned_FE"] = selected_planned_fe
    output["selected_action_effective_FE"] = selected_effective_fe
    output["selected_action_timed_out"] = selected_timed_out
    output["selected_action_path_completed"] = selected_path_completed
    output["selected_predicted_selector_target"] = selected_scores
    output["selector_regret_raw"] = output["selected_action_loss"].astype(float) - output[
        "best_observed_loss"
    ].astype(float)
    output["selector_regret_norm"] = output["selector_regret_raw"].to_numpy(dtype=float) / np.maximum(
        worst_losses - best_losses,
        1e-12,
    )
    output["selected_matches_best_observed"] = (
        output["selected_algorithm"].astype(str) == output["best_observed_algorithm"].astype(str)
    )
    output["selector_prediction_source"] = prediction_source
    output["selector_input_mode"] = selector_input_mode
    output["selector_status"] = "random_forest_action_loss_regression"
    output["selector_target_transform"] = selector_target_transform_for_mode(
        selector_input_mode
    )
    output["selector_fit_weight_mode"] = CLUSTER_BALANCED_FIT
    output["selection_reference_protocol"] = selection_reference_protocol
    output["query_preprocessing_id"] = states["query_preprocessing_id"].astype(str).to_numpy()
    output["performance_value_mode"] = "raw_objective"
    output["performance_loss_mode"] = "known_optimum_gap"
    continuation_gap = output["selected_action_loss"].to_numpy(dtype=float)
    query_sample_gap = output["query_best_gap"].to_numpy(dtype=float)
    selected_completed = output["selected_action_path_completed"].to_numpy(dtype=bool)
    operational_query_gap = np.where(
        selected_completed,
        np.minimum(continuation_gap, query_sample_gap),
        continuation_gap,
    )
    output["continuation_only_gap"] = continuation_gap
    output["query_sample_best_gap"] = query_sample_gap
    output["query_sample_improved_terminal"] = selected_completed & (
        query_sample_gap < continuation_gap
    )
    output["query_sample_terminal_gap_improvement"] = np.where(
        selected_completed,
        continuation_gap - operational_query_gap,
        0.0,
    )
    output["p_query_raw"] = (
        output["benchmark_reference_value"].to_numpy(dtype=float) + operational_query_gap
    )
    output["loss_skip"] = output["p_skip"].astype(float).to_numpy()
    output["loss_query"] = operational_query_gap
    output["p_query"] = output["loss_query"].astype(float)
    output["performance_gain_gap_raw"] = output["loss_skip"] - output["loss_query"]
    output["performance_gain_norm_gap"] = output["performance_gain_gap_raw"] / np.maximum(
        np.maximum(output["loss_skip"], output["loss_query"]),
        1e-12,
    )
    query_first_hit: list[int | None] = []
    for _, row in output.iterrows():
        candidates: list[int] = []
        prefix_hit = row["prefix_first_hit_FE"]
        if not pd.isna(prefix_hit):
            candidates.append(int(prefix_hit))
        if bool(row["query_success"]):
            offset = row["query_first_hit_offset"]
            if pd.isna(offset):
                raise ValueError("query_success requires query_first_hit_offset")
            candidates.append(int(row["FE_prefix"]) + int(offset))
        continuation_hit = row["selected_continuation_first_hit_FE"]
        if not pd.isna(continuation_hit):
            candidates.append(int(continuation_hit))
        hit = min(candidates) if candidates else None
        query_first_hit.append(hit)
    query_target_hit = np.asarray([value is not None for value in query_first_hit], dtype=bool)
    query_completed = output["selected_action_path_completed"].to_numpy(dtype=bool)
    output["query_path_observed_first_hit_FE"] = query_first_hit
    output["query_path_target_hit_observed"] = query_target_hit
    output["query_path_target_hit_before_failure"] = query_target_hit & ~query_completed
    output["query_path_endpoint_success"] = query_target_hit & query_completed
    # Compatibility aliases retain the formal ERT semantics.
    output["query_path_first_hit_FE"] = query_first_hit
    output["query_path_success"] = query_target_hit
    output["query_path_timed_out"] = output["selected_action_timed_out"].astype(bool)
    output["query_path_completed"] = output[
        "selected_action_path_completed"
    ].astype(bool)
    output["query_path_planned_FE"] = output["FE_total"].astype(int)
    output["query_path_effective_FE"] = (
        output["FE_prefix"].astype(int)
        + output["FE_query"].astype(int)
        + output["selected_action_effective_FE"].astype(int)
    )
    query_ids = tuple(sorted(states["query_id"].astype(str).unique()))
    query_protocols = tuple(sorted(states["query_protocol"].astype(str).unique()))
    preprocessing_ids = tuple(sorted(states["query_preprocessing_id"].astype(str).unique()))
    if len(query_ids) != 1 or len(query_protocols) != 1 or len(preprocessing_ids) != 1:
        raise ValueError("selection rows must contain one query protocol")
    output["query_id"] = query_ids[0]
    output["query_protocol"] = query_protocols[0]
    output["query_preprocessing_id"] = preprocessing_ids[0]
    output["query_feature_columns"] = states["query_feature_columns"].astype(str).to_numpy()
    if not np.isfinite(runtime_selection) or runtime_selection < 0.0:
        raise ValueError("runtime_selection must be finite and non-negative")
    output["runtime_selection"] = float(runtime_selection)
    return output


def sampling_only_continue_current_rows(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
) -> pd.DataFrame:
    """Materialize query sampling followed by native continuation without descriptors."""
    portfolio_index = {algorithm: index for index, algorithm in enumerate(portfolio)}
    scores = np.ones((len(states), len(portfolio)), dtype=float)
    for row_index, prefix_algorithm in enumerate(states["prefix_algorithm"].astype(str)):
        if prefix_algorithm not in portfolio_index:
            raise ValueError("sampling-only prefix algorithm is outside the frozen portfolio")
        scores[row_index, portfolio_index[prefix_algorithm]] = 0.0
    rows = selection_rows(
        states=states,
        portfolio=portfolio,
        predictions=scores,
        prediction_source="prespecified_sampling_only_continue_current",
        runtime_selection=0.0,
        selector_input_mode=STATE_ONLY_INPUT,
        selection_reference_protocol=SELECTION_REFERENCE_PROTOCOL,
    )
    if not rows["selected_equals_prefix"].astype(bool).all():
        raise RuntimeError("sampling-only path must select continue_current for every state")
    rows["selector_input_mode"] = NOT_APPLICABLE
    rows["selector_prediction_source"] = "not_applicable_sampling_only"
    rows["runtime_query_feature_computation"] = 0.0
    rows["runtime_query"] = (
        rows["runtime_query_sampling"].to_numpy(dtype=float)
        + rows["runtime_query_evaluation"].to_numpy(dtype=float)
    )
    rows["runtime_selection"] = 0.0
    rows["descriptor_computation_required"] = False
    return rows


def behavior_only_selection_rows(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    predictions: np.ndarray,
    prediction_source: str,
    runtime_selection: float,
) -> pd.DataFrame:
    """Materialize observed choices for the no-query, full-remaining-budget Selector."""
    if set(states["action_budget_mode"].astype(str)) != {FULL_REMAINING_BUDGET}:
        raise ValueError("behavior-only Selector requires full-remaining action outcomes")
    if set(states["sample_design_id"].astype(str)) != {NOT_APPLICABLE}:
        raise ValueError("behavior-only Selector outcomes must be independent of query sample design")
    if predictions.shape != (len(states), len(portfolio)):
        raise ValueError("behavior-only Selector prediction matrix has an unexpected shape")
    if not np.isfinite(runtime_selection) or runtime_selection < 0.0:
        raise ValueError("measured Selector runtime must be finite and non-negative")
    common_columns = [
        *STATE_KEY_COLUMNS,
        "FE_ratio",
        "FE_total",
        *SAMPLING_METADATA_COLUMNS,
        "prefix_scope",
        "action_budget_mode",
        "sample_design_id",
        "FE_prefix",
        "FE_query",
        "FE_no_query_optimization",
        "FE_action_optimization",
        "remaining_budget_ratio",
        "benchmark_reference_value",
        "success_gap_target",
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "runtime_no_query_handoff",
        "runtime_no_query_optimization",
        "runtime_no_query_handoff_repetitions",
        "runtime_no_query_optimization_repetitions",
        "timing_repetitions",
        "timing_repetition_indices",
        "timing_order_protocol",
        "skip_execution_order_repetitions",
        "prefix_first_hit_FE",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "skip_timed_out",
        "skip_path_completed",
        "no_query_transition_mode",
        "default_algorithm",
        "no_query_algorithm",
        "best_observed_algorithm",
        "best_observed_loss",
    ]
    missing = sorted(set(common_columns).difference(states.columns))
    if missing:
        raise ValueError(f"behavior-only Selector states are missing columns: {missing}")
    output = states[common_columns].copy()
    output["skip_continuation_planned_FE"] = output["skip_planned_FE"].astype(int)
    output["skip_continuation_effective_FE"] = output["skip_effective_FE"].astype(int)
    output["skip_planned_FE"] = output["FE_total"].astype(int)
    output["skip_effective_FE"] = (
        output["FE_prefix"].astype(int)
        + output["skip_continuation_effective_FE"].astype(int)
    )
    for index, algorithm in enumerate(portfolio):
        output[f"predicted_selector_target_{algorithm}"] = predictions[:, index]
        output[f"observed_loss_{algorithm}"] = states[f"observed_loss_{algorithm}"].to_numpy(dtype=float)
    selected_algorithms: list[str] = []
    selected_losses: list[float] = []
    selected_raw_losses: list[float] = []
    selected_scores: list[float] = []
    selected_runtimes: list[float] = []
    selected_handoff_runtimes: list[float] = []
    selected_runtime_repetitions: list[list[float]] = []
    selected_handoff_repetitions: list[list[float]] = []
    selected_order_repetitions: list[list[int]] = []
    selected_first_hits: list[int | None] = []
    selected_continuation_first_hits: list[int | None] = []
    selected_success: list[bool] = []
    selected_planned_fe: list[int] = []
    selected_effective_fe: list[int] = []
    selected_timed_out: list[bool] = []
    selected_path_completed: list[bool] = []
    for row_index in range(len(output)):
        selected = min(
            portfolio,
            key=lambda algorithm: float(
                output.at[row_index, f"predicted_selector_target_{algorithm}"]
            ),
        )
        selected_algorithms.append(selected)
        selected_scores.append(
            float(output.at[row_index, f"predicted_selector_target_{selected}"])
        )
        selected_losses.append(float(states.at[row_index, f"observed_loss_{selected}"]))
        selected_raw_losses.append(float(states.at[row_index, f"observed_loss_raw_{selected}"]))
        selected_runtimes.append(float(states.at[row_index, f"runtime_action_optimization_{selected}"]))
        selected_handoff_runtimes.append(float(states.at[row_index, f"runtime_handoff_{selected}"]))
        selected_runtime_repetitions.append(
            [
                float(value)
                for value in states.at[
                    row_index,
                    f"runtime_action_optimization_repetitions_{selected}",
                ]
            ]
        )
        selected_handoff_repetitions.append(
            [
                float(value)
                for value in states.at[row_index, f"runtime_handoff_repetitions_{selected}"]
            ]
        )
        selected_order_repetitions.append(
            [
                int(value)
                for value in states.at[row_index, f"execution_order_repetitions_{selected}"]
            ]
        )
        first_hit_value = states.at[row_index, f"first_hit_FE_{selected}"]
        continuation_hit_value = states.at[
            row_index,
            f"continuation_first_hit_FE_{selected}",
        ]
        selected_first_hits.append(None if pd.isna(first_hit_value) else int(first_hit_value))
        selected_continuation_first_hits.append(
            None if pd.isna(continuation_hit_value) else int(continuation_hit_value)
        )
        selected_success.append(bool(states.at[row_index, f"success_{selected}"]))
        selected_planned_fe.append(int(states.at[row_index, f"planned_FE_{selected}"]))
        selected_effective_fe.append(int(states.at[row_index, f"effective_FE_{selected}"]))
        selected_timed_out.append(bool(states.at[row_index, f"timed_out_{selected}"]))
        selected_path_completed.append(
            bool(states.at[row_index, f"path_completed_{selected}"])
        )
    output["selected_algorithm"] = selected_algorithms
    output["selected_action"] = np.where(
        output["selected_algorithm"].astype(str) == output["prefix_algorithm"].astype(str),
        "continue_current",
        output["selected_algorithm"].astype(str),
    )
    output["selected_equals_default"] = (
        output["selected_algorithm"].astype(str) == output["default_algorithm"].astype(str)
    )
    output["selected_equals_prefix"] = (
        output["selected_algorithm"].astype(str) == output["prefix_algorithm"].astype(str)
    )
    output["handoff_required"] = ~output["selected_equals_prefix"].astype(bool)
    output["handoff_type"] = np.where(
        output["handoff_required"].astype(bool),
        "population_transfer_initialization",
        "native_optimizer_state",
    )
    output["selected_action_loss"] = np.asarray(selected_losses, dtype=float)
    output["selected_action_loss_raw"] = np.asarray(selected_raw_losses, dtype=float)
    output["selected_predicted_selector_target"] = np.asarray(selected_scores, dtype=float)
    output["runtime_selected_action_optimization"] = np.asarray(selected_runtimes, dtype=float)
    output["runtime_handoff"] = np.asarray(selected_handoff_runtimes, dtype=float)
    output["runtime_handoff_repetitions"] = selected_handoff_repetitions
    output["runtime_selected_action_optimization_repetitions"] = selected_runtime_repetitions
    output["selected_execution_order_repetitions"] = selected_order_repetitions
    selected_hit_observed = np.asarray(
        [value is not None for value in selected_first_hits], dtype=bool
    )
    selected_completed = np.asarray(selected_path_completed, dtype=bool)
    output["selected_action_observed_first_hit_FE"] = selected_first_hits
    output["selected_action_target_hit_observed"] = selected_hit_observed
    output["selected_action_target_hit_before_failure"] = (
        selected_hit_observed & ~selected_completed
    )
    output["selected_action_endpoint_success"] = selected_hit_observed & selected_completed
    output["selected_action_first_hit_FE"] = selected_first_hits
    output["selected_continuation_first_hit_FE"] = selected_continuation_first_hits
    output["selected_action_success"] = selected_hit_observed
    output["selected_action_planned_FE"] = selected_planned_fe
    output["selected_action_effective_FE"] = selected_effective_fe
    output["selected_action_timed_out"] = selected_timed_out
    output["selected_action_path_completed"] = selected_path_completed
    output["p_behavior"] = output["selected_action_loss"].to_numpy(dtype=float)
    output["behavior_path_observed_first_hit_FE"] = selected_first_hits
    output["behavior_path_target_hit_observed"] = selected_hit_observed
    output["behavior_path_target_hit_before_failure"] = (
        selected_hit_observed & ~selected_completed
    )
    output["behavior_path_endpoint_success"] = selected_hit_observed & selected_completed
    # Compatibility aliases retain the formal ERT semantics.
    output["behavior_path_first_hit_FE"] = selected_first_hits
    output["behavior_path_success"] = selected_hit_observed
    output["behavior_path_timed_out"] = output["selected_action_timed_out"].astype(bool)
    output["behavior_path_completed"] = output[
        "selected_action_path_completed"
    ].astype(bool)
    output["behavior_path_planned_FE"] = output["FE_total"].astype(int)
    output["behavior_path_effective_FE"] = (
        output["FE_prefix"].astype(int)
        + output["selected_action_effective_FE"].astype(int)
    )
    output["selector_prediction_source"] = str(prediction_source)
    output["selector_input_mode"] = BEHAVIOR_ONLY_FULL_BUDGET_INPUT
    output["selector_target_transform"] = selector_target_transform_for_mode(
        BEHAVIOR_ONLY_FULL_BUDGET_INPUT
    )
    output["selector_fit_weight_mode"] = CLUSTER_BALANCED_FIT
    output["selection_reference_protocol"] = BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL
    output["runtime_selection"] = float(runtime_selection)
    return output


def pre_run_selection_rows(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    predictions: np.ndarray,
    prediction_source: str,
    runtime_selection: float,
) -> pd.DataFrame:
    """Materialize the independent FE=0 query-only Traditional-AAS choices."""
    if set(states["action_budget_mode"].astype(str)) != {
        PRE_RUN_QUERY_ADJUSTED_BUDGET
    }:
        raise ValueError("pre-run Selector requires FE=0 fresh-initialization outcomes")
    if predictions.shape != (len(states), len(portfolio)):
        raise ValueError("pre-run Selector prediction matrix has an unexpected shape")
    columns = [
        *PRE_RUN_STATE_KEY_COLUMNS,
        "FE_prefix",
        "FE_ratio",
        "FE_total",
        "action_budget_mode",
        "sample_design_id",
        "FE_query",
        "FE_action_optimization",
        "remaining_budget_ratio",
        "benchmark_reference_value",
        "success_gap_target",
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "query_feature_columns",
        "query_success",
        "query_first_hit_offset",
        "query_best_gap",
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_query_feature_computation",
        "runtime_query",
        "best_observed_algorithm",
        "best_observed_loss",
        "timing_repetitions",
        "timing_repetition_indices",
        "timing_order_protocol",
        "default_algorithm",
        "no_query_algorithm",
    ]
    missing = sorted(set(columns).difference(states.columns))
    if missing:
        raise ValueError(f"pre-run Selector states are missing columns: {missing}")
    output = states[columns].copy()
    for index, algorithm in enumerate(portfolio):
        output[f"predicted_selector_target_{algorithm}"] = predictions[:, index]
        output[f"observed_loss_{algorithm}"] = states[
            f"observed_loss_{algorithm}"
        ].to_numpy(dtype=float)
    selected_algorithms: list[str] = []
    selected_gaps: list[float] = []
    selected_raw: list[float] = []
    selected_first_hits: list[int | None] = []
    selected_success: list[bool] = []
    selected_planned: list[int] = []
    selected_effective: list[int] = []
    selected_timed_out: list[bool] = []
    selected_path_completed: list[bool] = []
    selected_runtime_repetitions: list[list[float]] = []
    selected_initializations: list[float] = []
    selected_initialization_repetitions: list[list[float]] = []
    selected_order_repetitions: list[list[int]] = []
    for row_index in range(len(output)):
        selected = min(
            portfolio,
            key=lambda algorithm: float(
                output.at[row_index, f"predicted_selector_target_{algorithm}"]
            ),
        )
        selected_algorithms.append(selected)
        selected_gaps.append(float(states.at[row_index, f"observed_loss_{selected}"]))
        selected_raw.append(float(states.at[row_index, f"observed_loss_raw_{selected}"]))
        first_hit = states.at[row_index, f"first_hit_FE_{selected}"]
        selected_first_hits.append(None if pd.isna(first_hit) else int(first_hit))
        selected_success.append(bool(states.at[row_index, f"success_{selected}"]))
        selected_planned.append(int(states.at[row_index, f"planned_FE_{selected}"]))
        selected_effective.append(int(states.at[row_index, f"effective_FE_{selected}"]))
        selected_timed_out.append(bool(states.at[row_index, f"timed_out_{selected}"]))
        selected_path_completed.append(
            bool(states.at[row_index, f"path_completed_{selected}"])
        )
        selected_runtime_repetitions.append(
            [
                float(value)
                for value in states.at[
                    row_index,
                    f"runtime_action_optimization_repetitions_{selected}",
                ]
            ]
        )
        selected_initializations.append(
            float(states.at[row_index, f"runtime_fresh_initialization_{selected}"])
        )
        selected_initialization_repetitions.append(
            [
                float(value)
                for value in states.at[
                    row_index,
                    f"runtime_fresh_initialization_repetitions_{selected}",
                ]
            ]
        )
        selected_order_repetitions.append(
            [
                int(value)
                for value in states.at[row_index, f"execution_order_repetitions_{selected}"]
            ]
        )
    output["selected_algorithm"] = selected_algorithms
    output["selected_action"] = selected_algorithms
    output["sbs_algorithm"] = output["default_algorithm"].astype(str)
    output["prefix_algorithm"] = output["selected_algorithm"].astype(str)
    output["selected_equals_default"] = (
        output["selected_algorithm"].astype(str)
        == output["default_algorithm"].astype(str)
    )
    output["selected_equals_prefix"] = True
    output["handoff_required"] = False
    output["handoff_type"] = "fresh_optimizer_initialization"
    output["selected_action_loss"] = np.asarray(selected_gaps, dtype=float)
    output["selected_action_loss_raw"] = np.asarray(selected_raw, dtype=float)
    output["selected_action_first_hit_FE"] = selected_first_hits
    selected_hit_observed = np.asarray(
        [value is not None for value in selected_first_hits], dtype=bool
    )
    selected_completed = np.asarray(selected_path_completed, dtype=bool)
    output["selected_action_observed_first_hit_FE"] = selected_first_hits
    output["selected_action_target_hit_observed"] = selected_hit_observed
    output["selected_action_target_hit_before_failure"] = (
        selected_hit_observed & ~selected_completed
    )
    output["selected_action_endpoint_success"] = selected_hit_observed & selected_completed
    output["selected_action_success"] = selected_hit_observed
    output["selected_action_planned_FE"] = selected_planned
    output["selected_action_effective_FE"] = selected_effective
    output["selected_action_timed_out"] = selected_timed_out
    output["selected_action_path_completed"] = selected_path_completed
    output["runtime_selected_action_optimization_repetitions"] = selected_runtime_repetitions
    output["runtime_handoff"] = 0.0
    output["runtime_handoff_repetitions"] = [
        [0.0] * TIMING_REPETITIONS for _ in range(len(output))
    ]
    output["runtime_fresh_initialization"] = selected_initializations
    output["runtime_fresh_initialization_repetitions"] = selected_initialization_repetitions
    output["selected_execution_order_repetitions"] = selected_order_repetitions
    continuation_gap = output["selected_action_loss"].to_numpy(dtype=float)
    query_gap = output["query_best_gap"].to_numpy(dtype=float)
    selected_completed = output["selected_action_path_completed"].to_numpy(dtype=bool)
    output["continuation_only_gap"] = continuation_gap
    output["pre_run_terminal_gap"] = np.where(
        selected_completed,
        np.minimum(continuation_gap, query_gap),
        continuation_gap,
    )
    output["query_sample_improved_terminal"] = selected_completed & (
        query_gap < continuation_gap
    )
    first_hits: list[int | None] = []
    for _, row in output.iterrows():
        candidates: list[int] = []
        if bool(row["query_success"]):
            candidates.append(int(row["query_first_hit_offset"]))
        if not pd.isna(row["selected_action_first_hit_FE"]):
            candidates.append(int(row["selected_action_first_hit_FE"]))
        first_hits.append(min(candidates) if candidates else None)
    pre_run_target_hit = np.asarray([value is not None for value in first_hits], dtype=bool)
    pre_run_completed = output["selected_action_path_completed"].to_numpy(dtype=bool)
    output["pre_run_observed_first_hit_FE"] = first_hits
    output["pre_run_target_hit_observed"] = pre_run_target_hit
    output["pre_run_target_hit_before_failure"] = pre_run_target_hit & ~pre_run_completed
    output["pre_run_endpoint_success"] = pre_run_target_hit & pre_run_completed
    # Compatibility aliases retain the formal ERT semantics.
    output["pre_run_first_hit_FE"] = first_hits
    output["pre_run_success"] = pre_run_target_hit
    output["pre_run_timed_out"] = output["selected_action_timed_out"].astype(bool)
    output["pre_run_path_completed"] = output[
        "selected_action_path_completed"
    ].astype(bool)
    output["pre_run_planned_FE"] = output["FE_total"].astype(int)
    output["pre_run_effective_FE"] = (
        output["FE_query"].astype(int)
        + output["selected_action_effective_FE"].astype(int)
    )
    output["selector_prediction_source"] = str(prediction_source)
    output["selector_input_mode"] = PRE_RUN_QUERY_ONLY_INPUT
    output["selector_target_transform"] = PRE_RUN_SELECTOR_TARGET_TRANSFORM
    output["selector_fit_weight_mode"] = CLUSTER_BALANCED_FIT
    output["selection_reference_protocol"] = PRE_RUN_SELECTION_REFERENCE_PROTOCOL
    output["runtime_selection"] = float(runtime_selection)
    return output


def measure_online_selection_runtime(
    selector_model: StatewiseSelectorModel,
    states: pd.DataFrame,
    *,
    max_states: int = 32,
) -> float:
    """Measure deployed one-state model inference and action selection, not batch throughput."""
    if states.empty:
        raise ValueError("cannot measure selector runtime on an empty state table")
    sample = states.iloc[: min(len(states), max_states)]
    runtimes = []
    for _, row in sample.iterrows():
        features = {column: row[column] for column in selector_model.feature_columns}
        _, _, elapsed = selector_model.select_one(features)
        runtimes.append(float(elapsed))
    runtime = float(np.median(np.asarray(runtimes, dtype=float)))
    if not np.isfinite(runtime) or runtime < 0.0:
        raise ValueError("measured selector runtime must be finite and non-negative")
    return runtime


def make_selector_features(
    *,
    behavior_features: dict[str, Any],
    query_features: dict[str, Any],
    query_feature_columns: tuple[str, ...],
    remaining_budget_ratio: float,
) -> dict[str, float]:
    return {
        **{
            column: np.nan if behavior_features[column] is None else float(behavior_features[column])
            for column in SELECTOR_BEHAVIOR_FEATURE_COLUMNS
        },
        **{
            column: np.nan if query_features[column] is None else float(query_features[column])
            for column in query_feature_columns
        },
        "remaining_budget_ratio": float(remaining_budget_ratio),
    }


def save_selector_model(selector_model: StatewiseSelectorModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selector_model, path, compress=3)


def load_selector_model(path: Path) -> StatewiseSelectorModel:
    if not path.exists():
        raise FileNotFoundError(f"missing statewise selector model: {path}")
    selector_model = joblib.load(path)
    if not isinstance(selector_model, StatewiseSelectorModel):
        raise ValueError("selector model artifact has an unexpected type")
    required_attributes = (
        "protocol",
        "query_id",
        "query_protocol",
        "sample_design_id",
        "query_feature_columns",
        "feature_columns",
        "selector_target_transform",
        "fit_weight_mode",
        "selector_input_mode",
        "action_budget_mode",
    )
    missing_attributes = [
        attribute for attribute in required_attributes if not hasattr(selector_model, attribute)
    ]
    if missing_attributes:
        raise ValueError(
            "selector model artifact predates the landscape-query protocol; "
            f"missing fields: {missing_attributes}"
        )
    supported_protocols = {
        SELECTION_REFERENCE_PROTOCOL,
        QUERY_ONLY_SELECTION_REFERENCE_PROTOCOL,
        BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL,
        PRE_RUN_SELECTION_REFERENCE_PROTOCOL,
    }
    if selector_model.protocol not in supported_protocols:
        raise ValueError("selector model artifact uses an unsupported protocol")
    expected_target_transform = selector_target_transform_for_mode(
        selector_model.selector_input_mode
    )
    if selector_model.selector_target_transform != expected_target_transform:
        raise ValueError("selector model artifact uses an unsupported action-loss target transform")
    if selector_model.fit_weight_mode != CLUSTER_BALANCED_FIT:
        raise ValueError("selector model artifact uses an unsupported fit-weight mode")
    if selector_model.selector_input_mode == BEHAVIOR_ONLY_FULL_BUDGET_INPUT:
        if selector_model.protocol != BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL:
            raise ValueError("behavior-only Selector uses the wrong protocol")
        if selector_model.action_budget_mode != FULL_REMAINING_BUDGET:
            raise ValueError("behavior-only Selector uses the wrong action budget")
        if selector_model.sample_design_id != NOT_APPLICABLE:
            raise ValueError("behavior-only Selector must not depend on a query sample design")
        if tuple(selector_model.query_feature_columns):
            raise ValueError("behavior-only Selector must not contain query feature columns")
        expected_model_columns = (*SELECTOR_BEHAVIOR_FEATURE_COLUMNS, "remaining_budget_ratio")
    else:
        spec = get_query_spec(selector_model.query_id)
        if selector_model.query_protocol != spec.protocol:
            raise ValueError("selector model query protocol is inconsistent with the active spec")
        if selector_model.sample_design_id != spec.sample_design_id:
            raise ValueError("selector model sample design is inconsistent with the active query spec")
        if getattr(selector_model, "query_preprocessing_id", QUERY_PREPROCESSING_VERSION) != QUERY_PREPROCESSING_VERSION:
            raise ValueError("selector model preprocessing contract is inconsistent with the active query spec")
        expected_query_columns = (
            spec.feature_columns
            if selector_model.selector_input_mode
            in {QUERY_FULL_INPUT, QUERY_ONLY_INPUT, PRE_RUN_QUERY_ONLY_INPUT}
            else ()
        )
        if tuple(selector_model.query_feature_columns) != expected_query_columns:
            raise ValueError("selector model feature columns are inconsistent with its input mode")
        if selector_model.selector_input_mode == PRE_RUN_QUERY_ONLY_INPUT:
            if selector_model.protocol != PRE_RUN_SELECTION_REFERENCE_PROTOCOL:
                raise ValueError("pre-run Selector uses the wrong protocol")
            if selector_model.action_budget_mode != PRE_RUN_QUERY_ADJUSTED_BUDGET:
                raise ValueError("pre-run Selector uses the wrong action budget")
        elif selector_model.selector_input_mode == QUERY_ONLY_INPUT:
            if selector_model.protocol != QUERY_ONLY_SELECTION_REFERENCE_PROTOCOL:
                raise ValueError("query-only Selector uses the wrong protocol")
        elif selector_model.protocol != SELECTION_REFERENCE_PROTOCOL:
            raise ValueError("statewise Selector uses the wrong protocol")
        if (
            selector_model.selector_input_mode != PRE_RUN_QUERY_ONLY_INPUT
            and selector_model.action_budget_mode != QUERY_ADJUSTED_BUDGET
        ):
            raise ValueError("query-specific Selector uses the wrong action budget")
        expected_model_columns = selector_feature_columns(spec, selector_model.selector_input_mode)
    if tuple(selector_model.feature_columns) != expected_model_columns:
        raise ValueError("selector model input columns are inconsistent with the active input contract")
    return selector_model


def _join_selector_inputs(
    *,
    states: pd.DataFrame,
    behavior: pd.DataFrame,
    query_features: pd.DataFrame | None,
    query_spec: LandscapeQuerySpec,
) -> pd.DataFrame:
    joined = _join_behavior_inputs(states=states, behavior=behavior)
    if query_features is None:
        return joined
    query_key = ["split", "problem_id", "function_id", "family", "dimension"]
    query_required = {
        *query_key,
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "sample_design_id",
        "benchmark_reference_value",
        "success_gap_target",
        "query_success",
        "query_first_hit_offset",
        "query_best_gap",
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_query_feature_computation",
        "runtime_query",
        "feature_status",
        "feature_failure",
        "feature_group_status",
        "feature_nonfinite",
        "additional_function_evaluations",
        "query_feature_columns",
        *query_spec.feature_columns,
    }
    missing_query = query_required.difference(query_features.columns)
    if missing_query:
        raise ValueError(f"query feature input is missing columns: {sorted(missing_query)}")
    if query_features.duplicated(query_key).any():
        raise ValueError("query feature input contains duplicate problem keys")
    if set(query_features["query_id"].astype(str)) != {query_spec.query_id}:
        raise ValueError("query feature input uses the wrong query_id")
    if set(query_features["query_protocol"].astype(str)) != {query_spec.protocol}:
        raise ValueError("query feature input uses the wrong query_protocol")
    if set(query_features["sample_design_id"].astype(str)) != {query_spec.sample_design_id}:
        raise ValueError("query feature input uses the wrong sample design")
    if set(query_features["query_preprocessing_id"].astype(str)) != {QUERY_PREPROCESSING_VERSION}:
        raise ValueError("query feature input uses the wrong preprocessing contract")
    expected_feature_columns = json.dumps(list(query_spec.feature_columns), ensure_ascii=False)
    if set(query_features["query_feature_columns"].astype(str)) != {expected_feature_columns}:
        raise ValueError("query feature input does not use the frozen feature-column list")
    if (query_features["additional_function_evaluations"].astype(int) != 0).any():
        raise ValueError("query feature input reports additional objective evaluations")
    query_runtime_columns = (
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_query_feature_computation",
        "runtime_query",
    )
    query_runtimes = query_features[list(query_runtime_columns)].to_numpy(dtype=float)
    if not np.isfinite(query_runtimes).all() or (query_runtimes < 0.0).any():
        raise ValueError("query feature runtimes must be finite and non-negative")
    expected_runtime_query = query_runtimes[:, 0] + query_runtimes[:, 1] + query_runtimes[:, 2]
    if not np.allclose(query_runtimes[:, 3], expected_runtime_query, rtol=0.0, atol=EPS):
        raise ValueError("runtime_query must equal query sampling, evaluation, and feature computation")
    for row in query_features.to_dict(orient="records"):
        group_status = json.loads(str(row["feature_group_status"]))
        if set(group_status) != set(query_spec.feature_groups):
            raise ValueError("query feature group status does not cover the frozen groups")
        has_group_failure = any(str(status.get("status")) != "ok" for status in group_status.values())
        expected_status = "failed" if has_group_failure else "ok"
        if str(row["feature_status"]) != expected_status:
            raise ValueError("query feature_status is inconsistent with group-level status")
    bbob = query_features[query_features["split"].astype(str).isin({"bbob_train", "bbob_validation"})]
    if not bbob.empty and (bbob["feature_status"].astype(str) != "ok").any():
        raise ValueError("BBOB train/validation cannot contain group-level query extraction failures")
    joined = joined.merge(
        query_features[list(query_required)],
        on=query_key,
        how="left",
        validate="many_to_one",
        indicator="_query_join",
        suffixes=("", "_query"),
    )
    if not joined["_query_join"].eq("both").all():
        raise ValueError("action-loss to query-feature join coverage must be 1.0")
    joined = joined.drop(columns="_query_join")
    if not (joined["sample_design_id"].astype(str) == joined["sample_design_id_query"].astype(str)).all():
        raise ValueError("action-loss and query-feature sample designs do not match")
    joined = joined.drop(columns="sample_design_id_query")
    if not np.allclose(
        joined["benchmark_reference_value"].to_numpy(dtype=float),
        joined["benchmark_reference_value_query"].to_numpy(dtype=float),
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("query samples and action outcomes use different benchmark reference values")
    joined = joined.drop(columns="benchmark_reference_value_query")
    return joined


def _join_behavior_inputs(*, states: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    behavior_key = list(STATE_KEY_COLUMNS)
    behavior_required = {
        *behavior_key,
        *SAMPLING_METADATA_COLUMNS,
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
    }
    missing_behavior = behavior_required.difference(behavior.columns)
    if missing_behavior:
        raise ValueError(f"behavior input is missing columns: {sorted(missing_behavior)}")
    if behavior.duplicated(behavior_key).any():
        raise ValueError("behavior input contains duplicate state keys")
    behavior_columns = behavior[
        behavior_key + list(SAMPLING_METADATA_COLUMNS) + list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    ].rename(columns={column: f"{column}_behavior" for column in SAMPLING_METADATA_COLUMNS})
    joined = states.merge(
        behavior_columns,
        on=behavior_key,
        how="left",
        validate="one_to_one",
        indicator="_behavior_join",
    )
    if not joined["_behavior_join"].eq("both").all():
        raise ValueError("action-loss to behavior join coverage must be 1.0")
    joined = joined.drop(columns="_behavior_join")
    for column in SAMPLING_METADATA_COLUMNS:
        behavior_column = f"{column}_behavior"
        if not all(
            _metadata_values_equal(left, right)
            for left, right in zip(joined[column], joined[behavior_column], strict=True)
        ):
            raise ValueError(
                f"action-loss and behavior dynamic-sampling metadata do not match: {column}"
            )
    return joined.drop(
        columns=[f"{column}_behavior" for column in SAMPLING_METADATA_COLUMNS]
    )


def _validate_state_metadata(
    action_losses: pd.DataFrame,
    *,
    key: list[str],
    metadata_columns: list[str],
) -> None:
    for _, group in action_losses.groupby(key, dropna=False, sort=False):
        first = group.iloc[0]
        for column in metadata_columns:
            if not all(
                _metadata_values_equal(first[column], value)
                for value in group[column].iloc[1:]
            ):
                raise ValueError(
                    f"state metadata are inconsistent across candidate actions: {column}"
                )


def _validate_action_timing_repetitions(action_losses: pd.DataFrame) -> None:
    _validate_action_runtime_role(action_losses)
    if set(action_losses["timing_repetitions"].astype(int)) != {TIMING_REPETITIONS}:
        raise ValueError("each candidate action outcome must be executed exactly once")
    if set(action_losses["timing_order_protocol"].astype(str)) != {
        EXECUTION_ORDER_PROTOCOL
    }:
        raise ValueError("action outcomes use an unsupported single-execution order protocol")
    list_columns = (
        "timing_repetition_indices",
        "runtime_no_query_handoff_repetitions",
        "runtime_no_query_optimization_repetitions",
        "runtime_handoff_repetitions",
        "runtime_action_optimization_repetitions",
        "execution_order_repetitions",
        "skip_execution_order_repetitions",
    )
    for column in list_columns:
        lengths = action_losses[column].map(lambda values: len(list(values)))
        if not bool((lengths == TIMING_REPETITIONS).all()):
            raise ValueError(f"{column} must contain exactly one entry per action row")
    expected_indices = tuple(range(TIMING_REPETITIONS))
    if not action_losses["timing_repetition_indices"].map(
        lambda values: tuple(int(value) for value in values) == expected_indices
    ).all():
        raise ValueError("action outcome execution indices must be exactly [0]")
    scalar_repeat_pairs = (
        ("runtime_no_query_handoff", "runtime_no_query_handoff_repetitions"),
        ("runtime_no_query_optimization", "runtime_no_query_optimization_repetitions"),
        ("runtime_handoff", "runtime_handoff_repetitions"),
        ("runtime_action_optimization", "runtime_action_optimization_repetitions"),
    )
    for scalar_column, repetitions_column in scalar_repeat_pairs:
        repetitions = action_losses[repetitions_column].map(
            lambda values: np.asarray(values, dtype=float)
        )
        if not repetitions.map(
            lambda values: np.isfinite(values).all() and bool((values >= 0.0).all())
        ).all():
            raise ValueError(f"{repetitions_column} must be finite and non-negative")
        medians = repetitions.map(lambda values: float(np.median(values))).to_numpy(dtype=float)
        if not np.allclose(
            action_losses[scalar_column].to_numpy(dtype=float),
            medians,
            rtol=0.0,
            atol=EPS,
        ):
            raise ValueError(f"{scalar_column} must equal its single recorded observation")
    for _, state in action_losses.groupby(list(STATE_KEY_COLUMNS), sort=False, dropna=False):
        skip_orders = tuple(int(value) for value in state.iloc[0]["skip_execution_order_repetitions"])
        action_orders = [
            tuple(int(value) for value in values)
            for values in state["execution_order_repetitions"]
        ]
        for repetition_index in range(TIMING_REPETITIONS):
            positions = {skip_orders[repetition_index]}
            positions.update(values[repetition_index] for values in action_orders)
            if positions != set(range(len(state) + 1)):
                raise ValueError(
                    "each timing repetition must execute Skip and every action at unique order positions"
                )


def _validate_action_runtime_role(action_losses: pd.DataFrame) -> None:
    if set(action_losses["action_outcome_execution_count"].astype(int)) != {1}:
        raise ValueError("every candidate action outcome must be executed exactly once")
    if set(action_losses["action_runtime_role"].astype(str)) != {
        "diagnostic_not_utility"
    }:
        raise ValueError(
            "action component runtimes are diagnostic only and cannot enter Utility"
        )


def _validate_action_ert_fields(action_losses: pd.DataFrame) -> None:
    if not np.isfinite(action_losses["success_gap_target"].to_numpy(dtype=float)).all():
        raise ValueError("success_gap_target must be finite")
    if bool((action_losses["success_gap_target"].to_numpy(dtype=float) <= 0.0).any()):
        raise ValueError("success_gap_target must be positive")
    action_completed = action_losses["path_completed"].to_numpy(dtype=bool)
    action_ok = action_losses["action_status"].astype(str).eq("ok").to_numpy(dtype=bool)
    skip_completed = action_losses["skip_path_completed"].to_numpy(dtype=bool)
    skip_ok = action_losses["skip_status"].astype(str).eq("ok").to_numpy(dtype=bool)
    if not np.array_equal(action_completed, action_ok):
        raise ValueError("candidate path_completed must equal action_status == ok")
    if not np.array_equal(skip_completed, skip_ok):
        raise ValueError("Skip path_completed must equal skip_status == ok")
    if bool((action_losses["timed_out"].to_numpy(dtype=bool) & action_completed).any()):
        raise ValueError("a timed-out candidate action cannot be path_completed")
    if bool((action_losses["skip_timed_out"].to_numpy(dtype=bool) & skip_completed).any()):
        raise ValueError("a timed-out Skip path cannot be path_completed")
    _validate_observed_hit_contract(
        action_losses,
        observed_first_hit_column="observed_first_hit_FE",
        target_hit_column="target_hit_observed",
        hit_before_failure_column="target_hit_before_failure",
        endpoint_success_column="endpoint_success",
        compatibility_first_hit_column="first_hit_FE",
        compatibility_success_column="success",
        completed=action_completed,
        label="candidate action",
    )
    _validate_observed_hit_contract(
        action_losses,
        observed_first_hit_column="skip_observed_first_hit_FE",
        target_hit_column="skip_target_hit_observed",
        hit_before_failure_column="skip_target_hit_before_failure",
        endpoint_success_column="skip_endpoint_success",
        compatibility_first_hit_column="skip_first_hit_FE",
        compatibility_success_column="skip_success",
        completed=skip_completed,
        label="Skip",
    )
    action_expected_hits = action_losses["prefix_first_hit_FE"].where(
        action_losses["prefix_first_hit_FE"].notna(),
        action_losses["continuation_first_hit_FE"],
    )
    skip_expected_hits = action_losses["skip_prefix_first_hit_FE"].where(
        action_losses["skip_prefix_first_hit_FE"].notna(),
        action_losses["skip_continuation_first_hit_FE"],
    )
    if not _nullable_integer_series_equal(
        action_losses["observed_first_hit_FE"], action_expected_hits
    ):
        raise ValueError("candidate observed_first_hit_FE does not match prefix/continuation hits")
    if not _nullable_integer_series_equal(
        action_losses["skip_observed_first_hit_FE"], skip_expected_hits
    ):
        raise ValueError("Skip observed_first_hit_FE does not match prefix/continuation hits")
    for planned_column, effective_column, status_column in (
        ("planned_FE", "effective_FE", "action_status"),
        ("skip_planned_FE", "skip_effective_FE", "skip_status"),
    ):
        planned = action_losses[planned_column].to_numpy(dtype=int)
        effective = action_losses[effective_column].to_numpy(dtype=int)
        if bool((planned <= 0).any()) or bool((effective < 0).any()) or bool((effective > planned).any()):
            raise ValueError(f"{planned_column}/{effective_column} FE accounting is inconsistent")
        completed_for_path = action_losses[status_column].astype(str).eq("ok").to_numpy(
            dtype=bool
        )
        if not np.array_equal(effective[completed_for_path], planned[completed_for_path]):
            raise ValueError(f"completed {status_column} rows must execute their planned FE")
    failure_cap = action_losses["failure_loss_cap"].to_numpy(dtype=float)
    reference = action_losses["benchmark_reference_value"].to_numpy(dtype=float)
    if not np.isfinite(failure_cap).all() or bool((failure_cap <= 0.0).any()):
        raise ValueError("failure_loss_cap must be finite and positive")
    incomplete_checks = (
        (
            ~action_completed,
            action_losses["action_loss"].to_numpy(dtype=float),
            action_losses["action_loss_raw"].to_numpy(dtype=float),
            "candidate action",
        ),
        (
            ~skip_completed,
            action_losses["p_skip"].to_numpy(dtype=float),
            action_losses["p_skip_raw"].to_numpy(dtype=float),
            "Skip",
        ),
    )
    for incomplete, capped_gap, capped_raw, label in incomplete_checks:
        if not np.allclose(capped_gap[incomplete], failure_cap[incomplete], rtol=0.0, atol=EPS):
            raise ValueError(f"incomplete {label} rows must use failure_loss_cap")
        if not np.allclose(
            capped_raw[incomplete],
            reference[incomplete] + failure_cap[incomplete],
            rtol=0.0,
            atol=EPS,
        ):
            raise ValueError(f"incomplete {label} raw losses must use reference + failure cap")
    for column in ("observed_first_hit_FE", "skip_observed_first_hit_FE"):
        observed = action_losses[column].dropna().to_numpy(dtype=float)
        if bool((observed <= 0.0).any()):
            raise ValueError(f"{column} must be positive when present")
        if bool(
            (
                action_losses.loc[action_losses[column].notna(), column].to_numpy(dtype=float)
                > action_losses.loc[action_losses[column].notna(), "FE_total"].to_numpy(dtype=float)
            ).any()
        ):
            raise ValueError(f"{column} cannot exceed FE_total")


def _validate_pre_run_ert_fields(action_losses: pd.DataFrame) -> None:
    completed = action_losses["path_completed"].to_numpy(dtype=bool)
    ok = action_losses["action_status"].astype(str).eq("ok").to_numpy(dtype=bool)
    if not np.array_equal(completed, ok):
        raise ValueError("pre-run path_completed must equal action_status == ok")
    if bool((action_losses["timed_out"].to_numpy(dtype=bool) & completed).any()):
        raise ValueError("a timed-out pre-run path cannot be path_completed")
    _validate_observed_hit_contract(
        action_losses,
        observed_first_hit_column="observed_first_hit_FE",
        target_hit_column="target_hit_observed",
        hit_before_failure_column="target_hit_before_failure",
        endpoint_success_column="endpoint_success",
        compatibility_first_hit_column="first_hit_FE",
        compatibility_success_column="success",
        completed=completed,
        label="pre-run action",
    )
    planned = action_losses["planned_FE"].to_numpy(dtype=int)
    effective = action_losses["effective_FE"].to_numpy(dtype=int)
    if bool((planned <= 0).any()) or bool((effective < 0).any()) or bool((effective > planned).any()):
        raise ValueError("pre-run planned/effective FE accounting is inconsistent")
    if not np.array_equal(effective[completed], planned[completed]):
        raise ValueError("completed pre-run paths must execute their complete planned FE")
    failure_cap = action_losses["failure_loss_cap"].to_numpy(dtype=float)
    reference = action_losses["benchmark_reference_value"].to_numpy(dtype=float)
    incomplete = ~completed
    if not np.isfinite(failure_cap).all() or bool((failure_cap <= 0.0).any()):
        raise ValueError("pre-run failure_loss_cap must be finite and positive")
    if not np.allclose(
        action_losses.loc[incomplete, "action_loss"].to_numpy(dtype=float),
        failure_cap[incomplete],
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("incomplete pre-run paths must use failure_loss_cap")
    if not np.allclose(
        action_losses.loc[incomplete, "action_loss_raw"].to_numpy(dtype=float),
        reference[incomplete] + failure_cap[incomplete],
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("incomplete pre-run raw losses must use reference + failure cap")


def _validate_observed_hit_contract(
    frame: pd.DataFrame,
    *,
    observed_first_hit_column: str,
    target_hit_column: str,
    hit_before_failure_column: str,
    endpoint_success_column: str,
    compatibility_first_hit_column: str,
    compatibility_success_column: str,
    completed: np.ndarray,
    label: str,
) -> None:
    observed = frame[observed_first_hit_column]
    target_hit = observed.notna().to_numpy(dtype=bool)
    if not np.array_equal(frame[target_hit_column].to_numpy(dtype=bool), target_hit):
        raise ValueError(f"{label} target_hit_observed must equal observed_first_hit_FE is not null")
    if not _nullable_integer_series_equal(
        observed, frame[compatibility_first_hit_column]
    ):
        raise ValueError(f"{label} first_hit_FE must alias observed_first_hit_FE")
    if not np.array_equal(
        frame[compatibility_success_column].to_numpy(dtype=bool), target_hit
    ):
        raise ValueError(f"{label} success must alias target_hit_observed")
    if not np.array_equal(
        frame[hit_before_failure_column].to_numpy(dtype=bool),
        target_hit & ~completed,
    ):
        raise ValueError(f"{label} target_hit_before_failure is inconsistent")
    if not np.array_equal(
        frame[endpoint_success_column].to_numpy(dtype=bool),
        target_hit & completed,
    ):
        raise ValueError(f"{label} endpoint_success is inconsistent")


def _nullable_integer_series_equal(left: pd.Series, right: pd.Series) -> bool:
    return bool(
        np.array_equal(
            left.fillna(-1).to_numpy(dtype=np.int64),
            right.fillna(-1).to_numpy(dtype=np.int64),
        )
    )


def _metadata_values_equal(left: Any, right: Any) -> bool:
    if _metadata_value_is_null(left) or _metadata_value_is_null(right):
        return _metadata_value_is_null(left) and _metadata_value_is_null(right)
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(
        right, (list, tuple, np.ndarray)
    ):
        if not isinstance(left, (list, tuple, np.ndarray)) or not isinstance(
            right, (list, tuple, np.ndarray)
        ):
            return False
        return tuple(left) == tuple(right)
    return bool(left == right)


def _metadata_value_is_null(value: Any) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    return bool(pd.isna(value))


def _make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", WeightedMedianImputer()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=1701,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
