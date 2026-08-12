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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from landscape_queries.specs import LandscapeQuerySpec, get_query_spec
from selection_reference.action_losses import ACTION_LOSS_PROTOCOL, STATE_KEY_COLUMNS


SELECTION_REFERENCE_PROTOCOL = "query_specific_statewise_action_loss_regression_v3"
SELECTOR_TARGET_TRANSFORM = "statewise_minmax_observed_action_loss"
EPS = 1e-12


@dataclass
class StatewiseSelectorModel:
    model: Pipeline
    target_algorithms: tuple[str, ...]
    feature_columns: tuple[str, ...]
    default_algorithm: str
    query_id: str
    query_protocol: str
    sample_design_id: str
    query_feature_columns: tuple[str, ...]
    selector_target_transform: str = SELECTOR_TARGET_TRANSFORM
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
        selected = min(self.target_algorithms, key=lambda algorithm: (score_map[algorithm], algorithm))
        return selected, score_map, float(perf_counter() - started)


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
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("behavior.parquet")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(f"missing behavior input: {path}")
    if not files:
        raise ValueError("no behavior parquet files found")
    frame = pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)
    if "algorithm" in frame.columns and "prefix_algorithm" not in frame.columns:
        frame = frame.rename(columns={"algorithm": "prefix_algorithm"})
    return frame


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
    query_features: pd.DataFrame,
    query_spec: LandscapeQuerySpec,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    required = {
        *STATE_KEY_COLUMNS,
        "FE_ratio",
        "FE_total",
        "sample_design_id",
        "sample_design_protocol",
        "FE_query",
        "FE_no_query_optimization",
        "FE_query_optimization",
        "remaining_budget_ratio",
        "p_skip",
        "runtime_no_query_optimization",
        "no_query_transition_mode",
        "default_algorithm",
        "no_query_algorithm",
        "action",
        "target_algorithm",
        "transition_mode",
        "action_loss",
        "action_loss_norm",
        "runtime_action_optimization",
        "best_observed_algorithm",
        "best_observed_loss",
        "action_loss_protocol",
    }
    missing = required.difference(action_losses.columns)
    if missing:
        raise ValueError(f"state-action loss input is missing columns: {sorted(missing)}")
    if set(action_losses["action_loss_protocol"].astype(str)) != {ACTION_LOSS_PROTOCOL}:
        raise ValueError("state-action loss inputs use an unsupported continuation protocol")
    if set(action_losses["sample_design_id"].astype(str)) != {query_spec.sample_design_id}:
        raise ValueError(f"{query_spec.query_id} is paired with the wrong query-FE action-loss table")
    key = list(STATE_KEY_COLUMNS)
    if action_losses.duplicated(key + ["target_algorithm"]).any():
        raise ValueError("state-action loss input contains duplicate state/algorithm rows")
    numeric_losses = action_losses[["action_loss", "action_loss_norm", "best_observed_loss"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_losses).all():
        raise ValueError("state-action losses and diagnostics must be finite")
    portfolio = tuple(sorted(action_losses["target_algorithm"].astype(str).unique()))
    if len(portfolio) != 4:
        raise ValueError("state-action loss input must contain exactly four unique portfolio algorithms")
    if not action_losses["prefix_algorithm"].astype(str).isin(portfolio).all():
        raise ValueError("every prefix algorithm must belong to the four-action portfolio")
    counts = action_losses.groupby(key, dropna=False)["target_algorithm"].nunique()
    if not bool((counts == len(portfolio)).all()):
        raise ValueError("every shared state must contain one observed loss for every portfolio algorithm")
    observed_sets = action_losses.groupby(key, dropna=False)["target_algorithm"].agg(
        lambda values: tuple(sorted(str(value) for value in values))
    )
    if not bool((observed_sets == portfolio).all()):
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
    if not bool(
        (action_losses["no_query_algorithm"].astype(str) == action_losses["default_algorithm"].astype(str)).all()
    ):
        raise ValueError("no_query_algorithm must equal default_algorithm")

    metadata_columns = [
        *key,
        "FE_ratio",
        "FE_total",
        "sample_design_id",
        "sample_design_protocol",
        "FE_query",
        "FE_no_query_optimization",
        "FE_query_optimization",
        "remaining_budget_ratio",
        "p_skip",
        "runtime_no_query_optimization",
        "no_query_transition_mode",
        "default_algorithm",
        "no_query_algorithm",
        "best_observed_algorithm",
        "best_observed_loss",
    ]
    states = action_losses[metadata_columns].drop_duplicates()
    if states.duplicated(key).any():
        raise ValueError("state metadata are inconsistent across candidate actions")
    ordered = action_losses.sort_values([*key, "action_loss", "target_algorithm"])
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
    states = states.drop(columns=["computed_best_observed_algorithm", "computed_best_observed_loss"])
    raw = action_losses.pivot(index=key, columns="target_algorithm", values="action_loss").reset_index()
    normalized = action_losses.pivot(index=key, columns="target_algorithm", values="action_loss_norm").reset_index()
    runtimes = action_losses.pivot(
        index=key,
        columns="target_algorithm",
        values="runtime_action_optimization",
    ).reset_index()
    raw = raw.rename(columns={algorithm: f"observed_loss_{algorithm}" for algorithm in portfolio})
    normalized = normalized.rename(columns={algorithm: f"target_loss_norm_{algorithm}" for algorithm in portfolio})
    runtimes = runtimes.rename(columns={algorithm: f"runtime_action_optimization_{algorithm}" for algorithm in portfolio})
    states = states.merge(raw, on=key, how="inner", validate="one_to_one")
    states = states.merge(normalized, on=key, how="inner", validate="one_to_one")
    states = states.merge(runtimes, on=key, how="inner", validate="one_to_one")
    states = _join_selector_inputs(
        states=states,
        behavior=behavior,
        query_features=query_features,
        query_spec=query_spec,
    )
    return states.sort_values(key).reset_index(drop=True), portfolio


def fit_selector_with_cross_family_predictions(
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    query_spec: LandscapeQuerySpec,
) -> tuple[StatewiseSelectorModel, np.ndarray, str]:
    defaults = tuple(sorted(states["default_algorithm"].astype(str).unique()))
    if len(defaults) != 1:
        raise ValueError("selector training states must use one train-derived SBS default")
    feature_columns = BEHAVIOR_FEATURE_COLUMNS + query_spec.feature_columns + ("remaining_budget_ratio",)
    x = states[list(feature_columns)]
    all_missing = [column for column in feature_columns if x[column].isna().all()]
    if all_missing:
        raise ValueError(f"selector training has entirely missing input columns: {all_missing}")
    y = states[[f"target_loss_norm_{algorithm}" for algorithm in portfolio]].to_numpy(dtype=float)
    families = states["family"].astype(str).to_numpy()
    unique_families = np.unique(families)
    cross_predictions = np.full_like(y, np.nan, dtype=float)
    if len(unique_families) >= 2:
        splitter = GroupKFold(n_splits=min(5, len(unique_families)))
        for train_indices, held_indices in splitter.split(x, y, groups=families):
            fit_families = set(families[train_indices])
            held_families = set(families[held_indices])
            if fit_families.intersection(held_families):
                raise RuntimeError("selector function-family cross-fitting fold contains family overlap")
            fold_model = _make_model()
            fold_model.fit(x.iloc[train_indices], y[train_indices])
            cross_predictions[held_indices] = fold_model.predict(x.iloc[held_indices])
        prediction_source = "cross_family"
    else:
        diagnostic_model = _make_model()
        diagnostic_model.fit(x, y)
        cross_predictions[:] = diagnostic_model.predict(x)
        prediction_source = "in_sample_insufficient_families"
    if not np.isfinite(cross_predictions).all():
        raise ValueError("cross-family selector predictions contain missing or non-finite values")
    final_model = _make_model()
    final_model.fit(x, y)
    selector_model = StatewiseSelectorModel(
        model=final_model,
        target_algorithms=portfolio,
        feature_columns=feature_columns,
        default_algorithm=defaults[0],
        query_id=query_spec.query_id,
        query_protocol=query_spec.protocol,
        sample_design_id=query_spec.sample_design_id,
        query_feature_columns=query_spec.feature_columns,
        selector_target_transform=SELECTOR_TARGET_TRANSFORM,
    )
    return selector_model, cross_predictions, prediction_source


def selection_rows(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    predictions: np.ndarray,
    prediction_source: str,
    runtime_selection: float,
) -> pd.DataFrame:
    if predictions.shape != (len(states), len(portfolio)):
        raise ValueError("selector prediction matrix has an unexpected shape")
    output = states[
        [
            *STATE_KEY_COLUMNS,
            "FE_ratio",
            "FE_total",
            "sample_design_id",
            "FE_query",
            "FE_no_query_optimization",
            "FE_query_optimization",
            "remaining_budget_ratio",
            "p_skip",
            "runtime_no_query_optimization",
            "no_query_transition_mode",
            "runtime_query",
            "feature_status",
            "feature_failure",
            "default_algorithm",
            "no_query_algorithm",
            "best_observed_algorithm",
            "best_observed_loss",
        ]
    ].copy()
    output["sbs_algorithm"] = output["default_algorithm"].astype(str)
    for index, algorithm in enumerate(portfolio):
        output[f"predicted_loss_norm_{algorithm}"] = predictions[:, index]
        output[f"observed_loss_{algorithm}"] = states[f"observed_loss_{algorithm}"].to_numpy(dtype=float)
    selected_algorithms = []
    selected_losses = []
    selected_scores = []
    selected_runtimes = []
    best_losses = output["best_observed_loss"].to_numpy(dtype=float)
    worst_losses = states[[f"observed_loss_{algorithm}" for algorithm in portfolio]].max(axis=1).to_numpy(dtype=float)
    for row_index in range(len(output)):
        selected = min(
            portfolio,
            key=lambda algorithm: (float(output.at[row_index, f"predicted_loss_norm_{algorithm}"]), algorithm),
        )
        selected_algorithms.append(selected)
        selected_scores.append(float(output.at[row_index, f"predicted_loss_norm_{selected}"]))
        selected_losses.append(float(output.at[row_index, f"observed_loss_{selected}"]))
        selected_runtimes.append(float(states.at[row_index, f"runtime_action_optimization_{selected}"]))
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
    output["runtime_selected_action_optimization"] = selected_runtimes
    output["selected_predicted_loss_norm"] = selected_scores
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
    output["selector_status"] = "random_forest_action_loss_regression"
    output["selector_target_transform"] = SELECTOR_TARGET_TRANSFORM
    output["selection_reference_protocol"] = SELECTION_REFERENCE_PROTOCOL
    query_ids = tuple(sorted(states["query_id"].astype(str).unique()))
    query_protocols = tuple(sorted(states["query_protocol"].astype(str).unique()))
    if len(query_ids) != 1 or len(query_protocols) != 1:
        raise ValueError("selection rows must contain one query protocol")
    output["query_id"] = query_ids[0]
    output["query_protocol"] = query_protocols[0]
    output["query_feature_columns"] = states["query_feature_columns"].astype(str).to_numpy()
    if not np.isfinite(runtime_selection) or runtime_selection < 0.0:
        raise ValueError("runtime_selection must be finite and non-negative")
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
            for column in BEHAVIOR_FEATURE_COLUMNS
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
    )
    missing_attributes = [
        attribute for attribute in required_attributes if not hasattr(selector_model, attribute)
    ]
    if missing_attributes:
        raise ValueError(
            "selector model artifact predates the landscape-query protocol; "
            f"missing fields: {missing_attributes}"
        )
    if selector_model.protocol != SELECTION_REFERENCE_PROTOCOL:
        raise ValueError("selector model artifact uses an unsupported protocol")
    if selector_model.selector_target_transform != SELECTOR_TARGET_TRANSFORM:
        raise ValueError("selector model artifact uses an unsupported action-loss target transform")
    spec = get_query_spec(selector_model.query_id)
    if selector_model.query_protocol != spec.protocol:
        raise ValueError("selector model query protocol is inconsistent with the frozen spec")
    if selector_model.sample_design_id != spec.sample_design_id:
        raise ValueError("selector model sample design is inconsistent with the frozen query spec")
    if tuple(selector_model.query_feature_columns) != spec.feature_columns:
        raise ValueError("selector model feature columns are inconsistent with the frozen query whitelist")
    expected_model_columns = BEHAVIOR_FEATURE_COLUMNS + spec.feature_columns + ("remaining_budget_ratio",)
    if tuple(selector_model.feature_columns) != expected_model_columns:
        raise ValueError("selector model input columns are inconsistent with the frozen query contract")
    return selector_model


def _join_selector_inputs(
    *,
    states: pd.DataFrame,
    behavior: pd.DataFrame,
    query_features: pd.DataFrame,
    query_spec: LandscapeQuerySpec,
) -> pd.DataFrame:
    behavior_key = list(STATE_KEY_COLUMNS)
    behavior_required = {*behavior_key, *BEHAVIOR_FEATURE_COLUMNS}
    missing_behavior = behavior_required.difference(behavior.columns)
    if missing_behavior:
        raise ValueError(f"behavior input is missing columns: {sorted(missing_behavior)}")
    if behavior.duplicated(behavior_key).any():
        raise ValueError("behavior input contains duplicate state keys")
    query_key = ["split", "problem_id", "family", "dimension"]
    query_required = {
        *query_key,
        "query_id",
        "query_protocol",
        "sample_design_id",
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
    expected_feature_columns = json.dumps(list(query_spec.feature_columns), ensure_ascii=False)
    if set(query_features["query_feature_columns"].astype(str)) != {expected_feature_columns}:
        raise ValueError("query feature input does not use the frozen feature-column list")
    if (query_features["additional_function_evaluations"].astype(int) != 0).any():
        raise ValueError("query feature input reports additional objective evaluations")
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
    joined = states.merge(
        behavior[behavior_key + list(BEHAVIOR_FEATURE_COLUMNS)],
        on=behavior_key,
        how="left",
        validate="one_to_one",
        indicator="_behavior_join",
    )
    if not joined["_behavior_join"].eq("both").all():
        raise ValueError("action-loss to behavior join coverage must be 1.0")
    joined = joined.drop(columns="_behavior_join")
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
    return joined


def _make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=1701,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
