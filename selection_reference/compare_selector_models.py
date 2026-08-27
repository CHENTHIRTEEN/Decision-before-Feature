from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from decision.cluster_weighting import (
    CLUSTER_BALANCED_FIT,
    WeightedMedianImputer,
    cluster_balanced_row_weights,
    fit_pipeline_with_weights,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from selection_reference.build import FORMAL_SELECTOR_TRAINING_SPLITS
from selection_reference.model import (
    QUERY_FULL_INPUT,
    SELECTOR_TARGET_TRANSFORM,
    QUERY_ONLY_INPUT,
    STATE_ONLY_INPUT,
    StatewiseSelectorModel,
    prepare_state_matrix,
    read_action_loss_data,
    read_behavior_data,
    read_query_feature_data,
    selector_feature_columns,
    selector_target_transform_for_mode,
)


SENSITIVITY_PROTOCOL = "selector_model_sensitivity_v1"
MODEL_ROLE = "sensitivity_only_not_formal_selection_reference"
RANDOM_STATE = 1701
DEFAULT_MODELS = (
    "rf_predefined",
    "gradient_boosting",
    "hist_gradient_boosting",
)
SENSITIVITY_SELECTOR_INPUT_MODES = (QUERY_FULL_INPUT, STATE_ONLY_INPUT, QUERY_ONLY_INPUT)
PORTFOLIO_ORDER = ("de", "pso", "cmaes", "shade")


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], Pipeline]
    available: bool = True
    unavailable_reason: str = ""


def _rf_predefined() -> Pipeline:
    return Pipeline(
        [
            ("imputer", WeightedMedianImputer()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _multioutput(estimator: RegressorMixin) -> Pipeline:
    return Pipeline(
        [
            ("imputer", WeightedMedianImputer()),
            ("regressor", MultiOutputRegressor(estimator, n_jobs=1)),
        ]
    )


def _gradient_boosting() -> Pipeline:
    return _multioutput(
        GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
    )


def _hist_gradient_boosting() -> Pipeline:
    return _multioutput(
        HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )
    )


def _optional_lightgbm() -> ModelSpec:
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:  # pragma: no cover - optional dependency
        return ModelSpec(
            name="lightgbm",
            factory=_gradient_boosting,
            available=False,
            unavailable_reason=f"lightgbm is not installed: {exc}",
        )

    def factory() -> Pipeline:
        return _multioutput(
            LGBMRegressor(
                objective="regression",
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            )
        )

    return ModelSpec(name="lightgbm", factory=factory)


def _optional_xgboost() -> ModelSpec:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - optional dependency
        return ModelSpec(
            name="xgboost",
            factory=_gradient_boosting,
            available=False,
            unavailable_reason=f"xgboost is not installed: {exc}",
        )

    def factory() -> Pipeline:
        return _multioutput(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        )

    return ModelSpec(name="xgboost", factory=factory)


def model_specs(include_optional: bool) -> dict[str, ModelSpec]:
    specs = {
        "rf_predefined": ModelSpec("rf_predefined", _rf_predefined),
        "gradient_boosting": ModelSpec("gradient_boosting", _gradient_boosting),
        "hist_gradient_boosting": ModelSpec("hist_gradient_boosting", _hist_gradient_boosting),
    }
    if include_optional:
        for spec in (_optional_lightgbm(), _optional_xgboost()):
            specs[spec.name] = spec
    return specs


def build_training_states(
    *,
    query_id: str,
    action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_paths: list[Path],
):
    query_spec = get_query_spec(query_id)
    action_losses = read_action_loss_data(action_loss_paths)
    behavior = read_behavior_data(behavior_paths)
    query_features = read_query_feature_data(query_feature_paths)
    states, portfolio = prepare_state_matrix(
        action_losses,
        behavior=behavior,
        query_features=query_features,
        query_spec=query_spec,
    )
    if portfolio != PORTFOLIO_ORDER:
        raise ValueError(f"unexpected portfolio order: {portfolio}")
    states["no_query_algorithm"] = states["default_algorithm"].astype(str)
    main_prefix = (
        states["prefix_algorithm"].astype(str)
        == states["default_algorithm"].astype(str)
    )
    train_states = states.loc[main_prefix].reset_index(drop=True)
    _validate_training_scope(train_states)
    return query_spec, train_states, portfolio


def sample_states_per_cv_group(states: pd.DataFrame, max_states_per_cv_group: int | None) -> pd.DataFrame:
    if max_states_per_cv_group is None:
        return states
    if max_states_per_cv_group <= 0:
        raise ValueError("--max-states-per-cv-group must be positive when provided")
    groups = []
    for _, group in states.groupby("cv_group_id", sort=True):
        groups.append(
            group.sample(
                n=min(len(group), int(max_states_per_cv_group)),
                random_state=RANDOM_STATE,
            )
        )
    sampled = (
        pd.concat(groups, ignore_index=True)
        .sort_values(
            ["split", "cv_group_id", "problem_id", "dimension", "seed", "FE"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return sampled


def _validate_training_scope(states: pd.DataFrame) -> None:
    splits = set(states["split"].astype(str))
    allowed = set(FORMAL_SELECTOR_TRAINING_SPLITS)
    if not splits.issubset(allowed):
        raise ValueError(
            "selector model sensitivity fitting only supports formal training splits; "
            f"observed={sorted(splits)}"
        )
    if "bbob_train" not in splits:
        raise ValueError("selector model sensitivity requires bbob_train rows")
    defaults = states["default_algorithm"].astype(str)
    if defaults.nunique() != 1:
        raise ValueError("selector model sensitivity requires one train-derived SBS default")
    if states["cv_group_id"].astype(str).nunique() < 2:
        raise ValueError("selector model sensitivity requires at least two CV groups")


def cross_cv_group_predictions(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    feature_columns: tuple[str, ...],
    factory: Callable[[], Pipeline],
    cv_splits: int | None,
) -> np.ndarray:
    target_columns = [f"target_selector_loss_{algorithm}" for algorithm in portfolio]
    x = states[list(feature_columns)]
    y = states[target_columns].to_numpy(dtype=float)
    groups = states["cv_group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    predictions = np.full((len(states), len(portfolio)), np.nan, dtype=float)
    n_splits = len(unique_groups) if cv_splits is None else min(int(cv_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError("--cv-splits must be at least 2 when provided")
    splitter = GroupKFold(n_splits=n_splits)
    for fit_indices, held_indices in splitter.split(x, y, groups=groups):
        model = factory()
        fit_pipeline_with_weights(
            model,
            x.iloc[fit_indices],
            y[fit_indices],
            cluster_balanced_row_weights(states.iloc[fit_indices]),
        )
        predictions[held_indices] = model.predict(x.iloc[held_indices])
    if not np.isfinite(predictions).all():
        raise ValueError("selector sensitivity predictions contain non-finite values")
    return predictions


def evaluate_predictions(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    predictions: np.ndarray,
    model_name: str,
    selector_input_mode: str,
    evaluation_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = states[[f"target_selector_loss_{algorithm}" for algorithm in portfolio]].to_numpy(dtype=float)
    observed = states[[f"observed_loss_{algorithm}" for algorithm in portfolio]].to_numpy(dtype=float)
    best_loss = states["best_observed_loss"].to_numpy(dtype=float)
    best_algorithm = states["best_observed_algorithm"].astype(str).to_numpy()
    selected_indices = np.argmin(predictions, axis=1)
    selected_algorithms = np.asarray(portfolio, dtype=object)[selected_indices]
    selected_loss = observed[np.arange(len(states)), selected_indices]
    worst_loss = observed.max(axis=1)
    regret_raw = selected_loss - best_loss
    regret_norm = regret_raw / np.maximum(worst_loss - best_loss, 1e-12)

    log_floor = states["log10_gap_floor"].to_numpy(dtype=float)
    log_cap = states["log10_gap_cap"].to_numpy(dtype=float)
    log_losses = np.log10(np.minimum(np.maximum(observed, log_floor[:, None]), log_cap[:, None]))
    best_log = log_losses.min(axis=1)
    selected_log = log_losses[np.arange(len(states)), selected_indices]
    regret_log10 = selected_log - best_log
    acceptable = log_losses - best_log[:, None] <= 0.05
    selected_acceptable = acceptable[np.arange(len(states)), selected_indices]

    prediction_rows = states[
        [
            "split",
            "problem_id",
            "function_id",
            "family",
            "cv_group_id",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "default_algorithm",
            "best_observed_algorithm",
            "best_observed_loss",
        ]
    ].copy()
    prediction_rows["selector_model"] = model_name
    prediction_rows["selector_model_role"] = MODEL_ROLE
    prediction_rows["selector_input_mode"] = selector_input_mode
    prediction_rows["evaluation_role"] = evaluation_role
    prediction_rows["selected_algorithm"] = selected_algorithms
    prediction_rows["selected_equals_default"] = (
        prediction_rows["selected_algorithm"].astype(str)
        == prediction_rows["default_algorithm"].astype(str)
    )
    prediction_rows["selected_equals_prefix"] = (
        prediction_rows["selected_algorithm"].astype(str)
        == prediction_rows["prefix_algorithm"].astype(str)
    )
    prediction_rows["handoff_required"] = ~prediction_rows["selected_equals_prefix"].astype(bool)
    prediction_rows["selected_matches_best_observed"] = (
        prediction_rows["selected_algorithm"].astype(str).to_numpy() == best_algorithm
    )
    prediction_rows["selected_loss"] = selected_loss
    prediction_rows["selector_regret_raw"] = regret_raw
    prediction_rows["selector_regret_norm"] = regret_norm
    prediction_rows["selector_regret_log10_gap"] = regret_log10
    prediction_rows["selected_is_acceptable_action"] = selected_acceptable
    for index, algorithm in enumerate(portfolio):
        prediction_rows[f"predicted_selector_target_{algorithm}"] = predictions[:, index]
        prediction_rows[f"observed_loss_{algorithm}"] = observed[:, index]

    summary = pd.DataFrame(
        [
            {
                "selector_model": model_name,
                "selector_model_role": MODEL_ROLE,
                "selector_input_mode": selector_input_mode,
                "evaluation_role": evaluation_role,
                "rows": int(len(states)),
                "cv_groups": int(states["cv_group_id"].astype(str).nunique()),
                "splits": ",".join(sorted(states["split"].astype(str).unique())),
                "rmse_target": float(np.sqrt(np.mean((predictions - target) ** 2))),
                "mean_selected_loss": float(np.mean(selected_loss)),
                "mean_best_observed_loss": float(np.mean(best_loss)),
                "mean_selector_regret_raw": float(np.mean(regret_raw)),
                "median_selector_regret_raw": float(np.median(regret_raw)),
                "mean_selector_regret_norm": float(np.mean(regret_norm)),
                "mean_selector_regret_log10_gap": float(np.mean(regret_log10)),
                "selected_matches_best_observed_rate": float(
                    np.mean(selected_algorithms == best_algorithm)
                ),
                "selected_is_acceptable_action_rate": float(np.mean(selected_acceptable)),
                "handoff_rate": float(np.mean(prediction_rows["handoff_required"].to_numpy(dtype=bool))),
                "selected_de_rate": float(np.mean(selected_algorithms == "de")),
                "selected_pso_rate": float(np.mean(selected_algorithms == "pso")),
                "selected_cmaes_rate": float(np.mean(selected_algorithms == "cmaes")),
                "selected_shade_rate": float(np.mean(selected_algorithms == "shade")),
                "protocol": SENSITIVITY_PROTOCOL,
                "selector_target_transform": SELECTOR_TARGET_TRANSFORM,
                "fit_weight_mode": CLUSTER_BALANCED_FIT,
            }
        ]
    )
    return summary, prediction_rows


def fit_final_model(
    *,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    feature_columns: tuple[str, ...],
    query_id: str,
    query_spec: Any,
    selector_input_mode: str,
    factory: Callable[[], Pipeline],
) -> StatewiseSelectorModel:
    target_columns = [f"target_selector_loss_{algorithm}" for algorithm in portfolio]
    x = states[list(feature_columns)]
    y = states[target_columns].to_numpy(dtype=float)
    model = factory()
    fit_pipeline_with_weights(model, x, y, cluster_balanced_row_weights(states))
    defaults = tuple(sorted(states["default_algorithm"].astype(str).unique()))
    sample_designs = tuple(sorted(states["sample_design_id"].astype(str).unique()))
    budget_modes = tuple(sorted(states["action_budget_mode"].astype(str).unique()))
    if len(defaults) != 1 or len(sample_designs) != 1 or len(budget_modes) != 1:
        raise ValueError("selector sensitivity final fit requires one default, sample design and budget mode")
    return StatewiseSelectorModel(
        model=model,
        target_algorithms=portfolio,
        feature_columns=feature_columns,
        default_algorithm=defaults[0],
        query_id=query_id,
        query_protocol=query_spec.protocol,
        query_preprocessing_id=query_spec.preprocessing_id,
        sample_design_id=sample_designs[0],
        query_feature_columns=(
            tuple(query_spec.feature_columns)
            if selector_input_mode in {QUERY_FULL_INPUT, QUERY_ONLY_INPUT}
            else ()
        ),
        selector_input_mode=selector_input_mode,
        action_budget_mode=budget_modes[0],
        selector_target_transform=selector_target_transform_for_mode(selector_input_mode),
        fit_weight_mode=CLUSTER_BALANCED_FIT,
        protocol=SENSITIVITY_PROTOCOL,
    )


def compare_selector_models(
    *,
    query_id: str,
    action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_paths: list[Path],
    output_dir: Path,
    selector_input_mode: str,
    model_names: list[str],
    include_optional: bool,
    max_states_per_cv_group: int | None,
    cv_splits: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"selector sensitivity output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    query_spec, states, portfolio = build_training_states(
        query_id=query_id,
        action_loss_paths=action_loss_paths,
        behavior_paths=behavior_paths,
        query_feature_paths=query_feature_paths,
    )
    full_rows = int(len(states))
    states = sample_states_per_cv_group(states, max_states_per_cv_group)
    _validate_training_scope(states)
    feature_columns = selector_feature_columns(query_spec, selector_input_mode)
    missing = sorted(set(feature_columns).difference(states.columns))
    if missing:
        raise ValueError(f"selector sensitivity input is missing columns: {missing}")

    requested_optional = bool({"lightgbm", "xgboost"}.intersection(model_names))
    specs = model_specs(include_optional=include_optional or requested_optional)
    unknown = sorted(set(model_names).difference(specs))
    if unknown:
        raise ValueError(f"unknown selector sensitivity models: {unknown}")

    summaries: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for name in model_names:
        spec = specs[name]
        if not spec.available:
            skipped.append({"selector_model": name, "reason": spec.unavailable_reason})
            continue
        oof = cross_cv_group_predictions(
            states=states,
            portfolio=portfolio,
            feature_columns=feature_columns,
            factory=spec.factory,
            cv_splits=cv_splits,
        )
        summary, rows = evaluate_predictions(
            states=states,
            portfolio=portfolio,
            predictions=oof,
            model_name=name,
            selector_input_mode=selector_input_mode,
            evaluation_role="cross_cv_group_oof",
        )
        summaries.append(summary)
        predictions.append(rows)
        final_model = fit_final_model(
            states=states,
            portfolio=portfolio,
            feature_columns=feature_columns,
            query_id=query_id,
            query_spec=query_spec,
            selector_input_mode=selector_input_mode,
            factory=spec.factory,
        )
        joblib.dump(final_model, model_dir / f"{name}.joblib")

    if not summaries:
        raise ValueError("no selector sensitivity models were available to fit")
    summary_frame = pd.concat(summaries, ignore_index=True).sort_values(
        ["mean_selector_regret_log10_gap", "mean_selector_regret_raw", "selector_model"],
        kind="mergesort",
    )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(summary_frame, preserve_index=False), output_dir / "selector_model_summary.parquet")
    pq.write_table(pa.Table.from_pandas(prediction_frame, preserve_index=False), output_dir / "selector_model_predictions.parquet")
    summary_frame.to_csv(output_dir / "selector_model_summary.csv", index=False)
    prediction_frame.to_csv(output_dir / "selector_model_predictions.csv", index=False)
    manifest = {
        "protocol": SENSITIVITY_PROTOCOL,
        "selector_model_role": MODEL_ROLE,
        "query_id": query_id,
        "selector_input_mode": selector_input_mode,
        "selector_target_transform": SELECTOR_TARGET_TRANSFORM,
        "fit_weight_mode": CLUSTER_BALANCED_FIT,
        "rows": int(len(states)),
        "full_rows_before_sampling": full_rows,
        "max_states_per_cv_group": max_states_per_cv_group,
        "cv_splits": cv_splits,
        "cv_groups": int(states["cv_group_id"].astype(str).nunique()),
        "models_requested": model_names,
        "models_fitted": summary_frame["selector_model"].astype(str).tolist(),
        "models_skipped": skipped,
        "outputs": {
            "summary_csv": str(output_dir / "selector_model_summary.csv"),
            "summary_parquet": str(output_dir / "selector_model_summary.parquet"),
            "predictions_parquet": str(output_dir / "selector_model_predictions.parquet"),
            "models": str(model_dir),
        },
        "formal_use_warning": (
            "Sensitivity outputs must not replace the predefined RF Selection Reference, "
            "Decision Model candidate selection, thresholds, Utility labels or policy endpoints."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False))
    if skipped:
        print("skipped optional models:")
        for item in skipped:
            print(f"- {item['selector_model']}: {item['reason']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit cross-CV-group downstream Selector model sensitivities."
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--action-losses", type=Path, action="append", required=True)
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--query-features", type=Path, action="append", required=True)
    parser.add_argument("--selector-input-mode", choices=SENSITIVITY_SELECTOR_INPUT_MODES, default=QUERY_FULL_INPUT)
    parser.add_argument("--model", choices=sorted(model_specs(include_optional=True)), action="append", default=None)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--max-states-per-cv-group", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    models = args.model or list(DEFAULT_MODELS)
    output_dir = args.output_dir or Path("results/selection_reference_sensitivity") / args.query_id / args.selector_input_mode
    compare_selector_models(
        query_id=args.query_id,
        action_loss_paths=args.action_losses,
        behavior_paths=args.behavior,
        query_feature_paths=args.query_features,
        output_dir=output_dir,
        selector_input_mode=args.selector_input_mode,
        model_names=models,
        include_optional=args.include_optional,
        max_states_per_cv_group=args.max_states_per_cv_group,
        cv_splits=args.cv_splits,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
