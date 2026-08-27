from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from selection_reference.model import (
    SELECTION_REFERENCE_PROTOCOL,
    SELECTOR_TARGET_TRANSFORM,
    fit_selector_with_cross_family_predictions,
    measure_online_selection_runtime,
    predict_with_main_prefix_cross_family_fits,
    prepare_state_matrix,
    read_action_loss_data,
    read_behavior_data,
    read_query_feature_data,
    save_selector_model,
    selection_rows,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from trajectory.sampling import SAMPLING_METADATA_COLUMNS


FORMAL_SELECTOR_TRAINING_SPLITS = ("bbob_train", "mabbob_formal")


def build_selection_reference(
    *,
    query_id: str,
    train_action_loss_paths: list[Path],
    predict_action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_paths: list[Path],
    output_path: Path,
    model_output_path: Path,
    pairwise_sensitivity_output_path: Path | None,
    formal_baseline_output_path: Path | None,
    evaluation_summary_output_path: Path | None,
    overwrite: bool,
) -> dict[str, int | str]:
    sensitivity_output = pairwise_sensitivity_output_path or output_path.with_name(
        "pairwise_aggregation_sensitivity.parquet"
    )
    baseline_output = formal_baseline_output_path or output_path.with_name(
        "formal_multioutput_rf_baseline.parquet"
    )
    evaluation_output = evaluation_summary_output_path or output_path.with_name(
        "selector_evaluation_summary.parquet"
    )
    existing_outputs = [
        path
        for path in (
            output_path,
            model_output_path,
            sensitivity_output,
            baseline_output,
            evaluation_output,
        )
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "selection-reference output already exists; pass --overwrite: "
            f"{existing_outputs[0]}"
        )
    query_spec = get_query_spec(query_id)
    behavior = read_behavior_data(behavior_paths)
    query_features = read_query_feature_data(query_feature_paths)
    train_action_losses = read_action_loss_data(train_action_loss_paths)
    all_train_states, portfolio = prepare_state_matrix(
        train_action_losses,
        behavior=behavior,
        query_features=query_features,
        query_spec=query_spec,
    )
    main_prefix_mask = (
        all_train_states["prefix_algorithm"].astype(str)
        == all_train_states["default_algorithm"].astype(str)
    )
    all_train_states["no_query_algorithm"] = all_train_states["default_algorithm"].astype(str)
    train_states = all_train_states.loc[main_prefix_mask].reset_index(drop=True)
    cross_probe_states = all_train_states.loc[~main_prefix_mask].reset_index(drop=True)
    _validate_training_scope(train_states)
    (
        selector_model,
        cross_predictions,
        train_prediction_source,
        cross_prediction_batch,
    ) = fit_selector_with_cross_family_predictions(train_states, portfolio, query_spec)
    outputs = [
        selection_rows(
            states=train_states,
            portfolio=portfolio,
            predictions=cross_predictions,
            prediction_source=train_prediction_source,
            runtime_selection=measure_online_selection_runtime(selector_model, train_states),
            prediction_batch=cross_prediction_batch,
        )
    ]
    sensitivity_outputs = [
        selection_rows(
            states=train_states,
            portfolio=portfolio,
            predictions=cross_predictions,
            prediction_source=train_prediction_source,
            runtime_selection=measure_online_selection_runtime(selector_model, train_states),
            prediction_batch=cross_prediction_batch.pairwise_only(),
        )
    ]
    baseline_outputs = [
        selection_rows(
            states=train_states,
            portfolio=portfolio,
            predictions=cross_predictions,
            prediction_source=train_prediction_source,
            runtime_selection=measure_online_selection_runtime(selector_model, train_states),
            prediction_batch=cross_prediction_batch.formal_only(portfolio),
        )
    ]
    if not cross_probe_states.empty:
        cross_probe_prediction_batch = predict_with_main_prefix_cross_family_fits(
            training_states=train_states,
            prediction_states=cross_probe_states,
            portfolio=portfolio,
            query_spec=query_spec,
        )
        cross_probe_predictions = cross_probe_prediction_batch.predicted_targets
        outputs.append(
            selection_rows(
                states=cross_probe_states,
                portfolio=portfolio,
                predictions=cross_probe_predictions,
                prediction_source="cross_cv_group_main_prefix",
                runtime_selection=measure_online_selection_runtime(
                    selector_model,
                    cross_probe_states,
                ),
                prediction_batch=cross_probe_prediction_batch,
            )
        )
        sensitivity_outputs.append(
            selection_rows(
                states=cross_probe_states,
                portfolio=portfolio,
                predictions=cross_probe_predictions,
                prediction_source="cross_cv_group_main_prefix",
                runtime_selection=measure_online_selection_runtime(
                    selector_model,
                    cross_probe_states,
                ),
                prediction_batch=cross_probe_prediction_batch.pairwise_only(),
            )
        )
        baseline_outputs.append(
            selection_rows(
                states=cross_probe_states,
                portfolio=portfolio,
                predictions=cross_probe_predictions,
                prediction_source="cross_cv_group_main_prefix",
                runtime_selection=measure_online_selection_runtime(
                    selector_model,
                    cross_probe_states,
                ),
                prediction_batch=cross_probe_prediction_batch.formal_only(portfolio),
            )
        )

    if predict_action_loss_paths:
        predict_action_losses = read_action_loss_data(predict_action_loss_paths)
        predict_states, predict_portfolio = prepare_state_matrix(
            predict_action_losses,
            behavior=behavior,
            query_features=query_features,
            query_spec=query_spec,
        )
        if predict_portfolio != portfolio:
            raise ValueError("training and prediction action-loss files use different algorithm portfolios")
        predict_states["no_query_algorithm"] = predict_states["default_algorithm"].astype(str)
        _validate_no_training_overlap(all_train_states, predict_states)
        prediction_batch = selector_model.predict_decisions(predict_states)
        predictions = prediction_batch.predicted_targets
        outputs.append(
            selection_rows(
                states=predict_states,
                portfolio=portfolio,
                predictions=predictions,
                prediction_source="train_fit",
                runtime_selection=measure_online_selection_runtime(selector_model, predict_states),
                prediction_batch=prediction_batch,
            )
        )
        sensitivity_outputs.append(
            selection_rows(
                states=predict_states,
                portfolio=portfolio,
                predictions=predictions,
                prediction_source="train_fit",
                runtime_selection=measure_online_selection_runtime(
                    selector_model,
                    predict_states,
                ),
                prediction_batch=prediction_batch.pairwise_only(),
            )
        )
        baseline_outputs.append(
            selection_rows(
                states=predict_states,
                portfolio=portfolio,
                predictions=predictions,
                prediction_source="train_fit",
                runtime_selection=measure_online_selection_runtime(
                    selector_model,
                    predict_states,
                ),
                prediction_batch=prediction_batch.formal_only(portfolio),
            )
        )

    reference = pd.concat(outputs, ignore_index=True).sort_values(
        ["split", "problem_id", "dimension", "prefix_algorithm", "seed", "FE"]
    ).reset_index(drop=True)
    _validate_reference(reference, portfolio)
    pairwise_sensitivity = pd.concat(sensitivity_outputs, ignore_index=True).sort_values(
        ["split", "problem_id", "dimension", "prefix_algorithm", "seed", "FE"]
    ).reset_index(drop=True)
    _validate_reference(pairwise_sensitivity, portfolio)
    formal_baseline = pd.concat(baseline_outputs, ignore_index=True).sort_values(
        ["split", "problem_id", "dimension", "prefix_algorithm", "seed", "FE"]
    ).reset_index(drop=True)
    _validate_reference(formal_baseline, portfolio)
    evaluation_summary = _selector_evaluation_summary(
        pd.concat(
            [reference, pairwise_sensitivity, formal_baseline],
            ignore_index=True,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(reference, preserve_index=False), output_path)
    sensitivity_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pairwise_sensitivity, preserve_index=False),
        sensitivity_output,
    )
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(formal_baseline, preserve_index=False),
        baseline_output,
    )
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(evaluation_summary, preserve_index=False),
        evaluation_output,
    )
    save_selector_model(selector_model, model_output_path)
    print(f"wrote {len(reference)} statewise selection-reference rows to {output_path}")
    print(f"wrote statewise selector model to {model_output_path}")
    print(f"wrote pairwise aggregation sensitivity to {sensitivity_output}")
    print(f"wrote formal multi-output RF baseline to {baseline_output}")
    print(f"wrote Selector evaluation summary to {evaluation_output}")
    print(f"SBS/default optimizer: {selector_model.default_algorithm}")
    return {
        "rows": int(len(reference)),
        "training_rows": int(len(train_states)),
        "cross_probe_training_rows": int(len(cross_probe_states)),
        "output": str(output_path),
        "model": str(model_output_path),
        "pairwise_sensitivity": str(sensitivity_output),
        "formal_baseline": str(baseline_output),
        "evaluation_summary": str(evaluation_output),
        "default_algorithm": selector_model.default_algorithm,
        "training_splits": ",".join(sorted(train_states["split"].astype(str).unique())),
        "query_id": query_spec.query_id,
        "query_protocol": query_spec.protocol,
        "query_feature_columns": ",".join(query_spec.feature_columns),
        "protocol": SELECTION_REFERENCE_PROTOCOL,
        "selector_target_transform": SELECTOR_TARGET_TRANSFORM,
    }


def _selector_evaluation_summary(reference: pd.DataFrame) -> pd.DataFrame:
    by_dimension = reference.copy()
    by_dimension["dimension_stratum"] = by_dimension["dimension"].map(
        lambda value: f"dimension_{int(value)}"
    )
    all_dimensions = reference.copy()
    all_dimensions["dimension_stratum"] = "all_dimensions"
    evaluation = pd.concat([by_dimension, all_dimensions], ignore_index=True)
    rows: list[dict[str, object]] = []
    group_columns = [
        "selector_type",
        "selector_prediction_source",
        "split",
        "dimension_stratum",
    ]
    for keys, group in evaluation.groupby(group_columns, dropna=False, sort=True):
        regret = group["selector_regret_log10_gap"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "rows": int(len(group)),
                "runs": int(
                    group[
                        ["problem_id", "prefix_algorithm", "seed"]
                    ].drop_duplicates().shape[0]
                ),
                "mean_selector_regret_log10_gap": float(np.mean(regret)),
                "median_selector_regret_log10_gap": float(np.median(regret)),
                "p90_selector_regret_log10_gap": float(np.quantile(regret, 0.90)),
                "selected_matches_best_observed_rate": float(
                    group["selected_matches_best_observed"].astype(bool).mean()
                ),
                "selected_is_acceptable_action_rate": float(
                    group["selected_is_acceptable_action"].astype(bool).mean()
                ),
                "handoff_required_rate": float(
                    group["handoff_required"].astype(bool).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _validate_training_scope(states: pd.DataFrame) -> None:
    splits = set(states["split"].astype(str))
    allowed = set(FORMAL_SELECTOR_TRAINING_SPLITS)
    if not splits.issubset(allowed):
        raise ValueError(
            "formal selector fitting only supports BBOB-train and the selected "
            f"MA-BBOB formal augmentation; observed={sorted(splits)}"
        )
    if "bbob_train" not in splits:
        raise ValueError("formal selector fitting requires the bbob_train split")
    defaults = states["default_algorithm"].astype(str)
    if defaults.nunique() != 1:
        raise ValueError("formal selector training requires one train-derived SBS default")
    if not bool((states["prefix_algorithm"].astype(str) == defaults).all()):
        raise ValueError("formal selector training must use only main-protocol prefix=default states")
    if states["family"].astype(str).nunique() < 2:
        raise ValueError(
            "formal selector training requires at least two landscape families "
            "for cross-family predictions"
        )


def _validate_no_training_overlap(train_states: pd.DataFrame, predict_states: pd.DataFrame) -> None:
    train_splits = set(train_states["split"].astype(str))
    predict_splits = set(predict_states["split"].astype(str))
    if train_splits.intersection(predict_splits):
        raise ValueError(
            "prediction action-loss inputs overlap the training split; omit them because training rows are already emitted"
        )
    train_functions = set(train_states["function_id"].astype(str))
    predict_functions = set(predict_states["function_id"].astype(str))
    overlap = sorted(train_functions.intersection(predict_functions))
    if overlap:
        raise ValueError(f"training and held-out prediction function IDs overlap: {overlap}")


def _validate_reference(reference: pd.DataFrame, portfolio: tuple[str, ...]) -> None:
    key = [
        "split",
        "problem_id",
        "function_id",
        "family",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
    ]
    if reference.empty:
        raise ValueError("selection reference contains no rows")
    if len(portfolio) != 4 or len(set(portfolio)) != 4:
        raise ValueError("selection reference requires exactly four unique portfolio algorithms")
    if reference.duplicated(key).any():
        raise ValueError("selection reference contains duplicate shared-state keys")
    missing_sampling = set(SAMPLING_METADATA_COLUMNS).difference(reference.columns)
    if missing_sampling:
        raise ValueError(
            "selection reference is missing dynamic-sampling metadata: "
            f"{sorted(missing_sampling)}"
        )
    expected_fe_ratio = reference["FE"].astype(float) / reference["FE_total"].astype(float)
    if not np.array_equal(
        reference["FE_ratio"].to_numpy(dtype=float),
        expected_fe_ratio.to_numpy(dtype=float),
    ):
        raise ValueError("selection-reference FE_ratio must equal the actual FE / FE_total")
    if set(reference["selected_algorithm"].astype(str)).difference(portfolio):
        raise ValueError("selection reference selected an algorithm outside the portfolio")
    selector_columns = {
        "selector_type",
        "selector_component",
        "selector_status",
        "selected_selector_ranking_score",
        *{f"selector_ranking_score_{algorithm}" for algorithm in portfolio},
        *{f"pairwise_votes_{algorithm}" for algorithm in portfolio},
        *{f"pairwise_probability_sum_{algorithm}" for algorithm in portfolio},
    }
    missing_selector_columns = selector_columns.difference(reference.columns)
    if missing_selector_columns:
        raise ValueError(
            "selection reference is missing Selector routing fields: "
            f"{sorted(missing_selector_columns)}"
        )
    selector_types = set(reference["selector_type"].astype(str))
    supported_selector_types = {
        "dimension_aware_hybrid_selector",
        "pairwise_aggregation_rf_classifier",
        "formal_multioutput_rf",
    }
    if not selector_types.issubset(supported_selector_types):
        raise ValueError(f"selection reference uses unsupported Selector types: {selector_types}")
    if not bool(
        (
            reference["selector_status"].astype(str)
            == reference["selector_type"].astype(str)
        ).all()
    ):
        raise ValueError("selector_status must identify the deployed Selector type")
    expected_components = np.where(
        reference["dimension"].astype(int).eq(40),
        "pairwise_aggregation_rf_classifier",
        "formal_multioutput_rf",
    )
    pairwise_only = reference["selector_type"].astype(str).eq(
        "pairwise_aggregation_rf_classifier"
    )
    formal_only = reference["selector_type"].astype(str).eq("formal_multioutput_rf")
    expected_components = np.where(
        pairwise_only,
        "pairwise_aggregation_rf_classifier",
        expected_components,
    )
    expected_components = np.where(
        formal_only,
        "formal_multioutput_rf",
        expected_components,
    )
    if not np.array_equal(
        reference["selector_component"].astype(str).to_numpy(),
        expected_components,
    ):
        raise ValueError("Selector component does not match the dimension routing rule")
    predicted_targets = reference[
        [f"predicted_selector_target_{algorithm}" for algorithm in portfolio]
    ].to_numpy(dtype=float)
    ranking_scores = reference[
        [f"selector_ranking_score_{algorithm}" for algorithm in portfolio]
    ].to_numpy(dtype=float)
    selected_indices = np.asarray(
        [portfolio.index(value) for value in reference["selected_algorithm"].astype(str)],
        dtype=int,
    )
    row_indices = np.arange(len(reference))
    if not np.allclose(
        reference["selected_predicted_selector_target"].to_numpy(dtype=float),
        predicted_targets[row_indices, selected_indices],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("selected predicted target is inconsistent")
    if not np.allclose(
        reference["selected_selector_ranking_score"].to_numpy(dtype=float),
        ranking_scores[row_indices, selected_indices],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("selected Selector ranking score is inconsistent")
    if not np.array_equal(np.argmin(ranking_scores, axis=1), selected_indices):
        raise ValueError("selected algorithm does not minimize the deployed ranking score")
    formal_rows = reference["selector_component"].astype(str).eq("formal_multioutput_rf")
    if formal_rows.any() and not np.allclose(
        ranking_scores[formal_rows],
        predicted_targets[formal_rows],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("formal multi-output rows must rank by predicted target")
    pairwise_rows = ~formal_rows
    if pairwise_rows.any():
        votes = reference.loc[
            pairwise_rows,
            [f"pairwise_votes_{algorithm}" for algorithm in portfolio],
        ].to_numpy(dtype=float)
        probability_sums = reference.loc[
            pairwise_rows,
            [f"pairwise_probability_sum_{algorithm}" for algorithm in portfolio],
        ].to_numpy(dtype=float)
        expected_scores = -(votes * float(len(portfolio)) + probability_sums)
        if not np.allclose(
            ranking_scores[pairwise_rows],
            expected_scores,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("pairwise ranking scores do not match votes and probability sums")
    if not (reference["no_query_algorithm"].astype(str) == reference["default_algorithm"].astype(str)).all():
        raise ValueError("selection reference no_query_algorithm must equal default_algorithm")
    expected_action = reference["selected_algorithm"].astype(str).where(
        reference["selected_algorithm"].astype(str) != reference["prefix_algorithm"].astype(str),
        "continue_current",
    )
    if not bool((reference["selected_action"].astype(str) == expected_action).all()):
        raise ValueError("selected_action does not match selected_algorithm and prefix_algorithm")
    expected_equals_default = (
        reference["selected_algorithm"].astype(str) == reference["default_algorithm"].astype(str)
    )
    expected_equals_prefix = (
        reference["selected_algorithm"].astype(str) == reference["prefix_algorithm"].astype(str)
    )
    if not bool((reference["selected_equals_default"].astype(bool) == expected_equals_default).all()):
        raise ValueError("selected_equals_default is inconsistent")
    if not bool((reference["selected_equals_prefix"].astype(bool) == expected_equals_prefix).all()):
        raise ValueError("selected_equals_prefix is inconsistent")
    expected_handoff = ~expected_equals_prefix
    if not bool((reference["handoff_required"].astype(bool) == expected_handoff).all()):
        raise ValueError("handoff_required is inconsistent")
    if not bool(
        (
            reference["handoff_required"].astype(bool)
            == reference["selected_action"].astype(str).ne("continue_current")
        ).all()
    ):
        raise ValueError("handoff_required must match selected_action")
    if not bool((reference["handoff_type"].astype(str) == reference["selected_transition_mode"].astype(str)).all()):
        raise ValueError("handoff_type must equal selected_transition_mode")
    if not bool(
        (
            reference["handoff_required"].astype(bool)
            == reference["handoff_type"].astype(str).eq("population_transfer_initialization")
        ).all()
    ):
        raise ValueError("handoff_required must match the selected transition mode")
    if not bool(
        (reference.loc[~reference["handoff_required"].astype(bool), "runtime_handoff"].astype(float) == 0.0).all()
    ):
        raise ValueError("native selected actions must have zero handoff runtime")
    regret = reference["action_loss"].astype(float) - reference["best_observed_loss"].astype(float)
    if bool((regret < -1e-12).any()):
        raise ValueError("selector regret cannot be smaller than zero")
    acceptable_columns = {
        "acceptable_action_tolerance_log10_gap",
        "acceptable_action_set",
        "selected_is_acceptable_action",
        "acceptable_action_count",
        "selector_regret_log10_gap",
        "log10_gap_floor",
        "log10_gap_cap",
    }
    missing_acceptable = acceptable_columns.difference(reference.columns)
    if missing_acceptable:
        raise ValueError(f"selection reference is missing acceptable-action fields: {sorted(missing_acceptable)}")
    if not np.allclose(
        reference["acceptable_action_tolerance_log10_gap"].to_numpy(dtype=float),
        0.05,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("acceptable_action_tolerance_log10_gap must equal 0.05")
    observed_losses = reference[[f"observed_loss_{algorithm}" for algorithm in portfolio]].to_numpy(dtype=float)
    log_floor = reference["log10_gap_floor"].to_numpy(dtype=float)
    log_cap = reference["log10_gap_cap"].to_numpy(dtype=float)
    log_losses = np.log10(np.minimum(np.maximum(observed_losses, log_floor[:, None]), log_cap[:, None]))
    best_log_losses = log_losses.min(axis=1)
    selected_algorithms = reference["selected_algorithm"].astype(str).to_numpy()
    expected_log_regrets = np.empty(len(reference), dtype=float)
    expected_acceptable_counts = np.empty(len(reference), dtype=int)
    expected_selected_acceptable = np.empty(len(reference), dtype=bool)
    expected_acceptable_sets: list[set[str]] = []
    for row_index, selected_algorithm in enumerate(selected_algorithms):
        acceptable = {
            algorithm
            for algorithm_index, algorithm in enumerate(portfolio)
            if log_losses[row_index, algorithm_index] - best_log_losses[row_index] <= 0.05
        }
        expected_acceptable_sets.append(acceptable)
        selected_index = portfolio.index(selected_algorithm)
        expected_log_regrets[row_index] = log_losses[row_index, selected_index] - best_log_losses[row_index]
        expected_acceptable_counts[row_index] = len(acceptable)
        expected_selected_acceptable[row_index] = selected_algorithm in acceptable
    observed_acceptable_sets = [
        {str(value) for value in values}
        for values in reference["acceptable_action_set"]
    ]
    if observed_acceptable_sets != expected_acceptable_sets:
        raise ValueError("acceptable_action_set is inconsistent")
    if not np.allclose(
        reference["selector_regret_log10_gap"].to_numpy(dtype=float),
        expected_log_regrets,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("selector_regret_log10_gap is inconsistent")
    if not np.array_equal(
        reference["acceptable_action_count"].to_numpy(dtype=int),
        expected_acceptable_counts,
    ):
        raise ValueError("acceptable_action_count is inconsistent")
    if not np.array_equal(
        reference["selected_is_acceptable_action"].to_numpy(dtype=bool),
        expected_selected_acceptable,
    ):
        raise ValueError("selected_is_acceptable_action is inconsistent")
    if not bool((reference["selection_reference_protocol"].astype(str) == SELECTION_REFERENCE_PROTOCOL).all()):
        raise ValueError("selection reference protocol field is inconsistent")
    if not bool((reference["selector_target_transform"].astype(str) == SELECTOR_TARGET_TRANSFORM).all()):
        raise ValueError("selection reference target transform field is inconsistent")
    selected_completed = reference["selected_action_path_completed"].to_numpy(
        dtype=bool
    )
    selected_timed_out = reference["selected_action_timed_out"].to_numpy(dtype=bool)
    if bool((selected_completed & selected_timed_out).any()):
        raise ValueError("a timed-out selected action cannot be completed")
    if not np.array_equal(
        reference["query_path_completed"].to_numpy(dtype=bool),
        selected_completed,
    ):
        raise ValueError("query_path_completed must match the selected continuation")
    if not np.array_equal(
        reference["query_path_timed_out"].to_numpy(dtype=bool),
        selected_timed_out,
    ):
        raise ValueError("query_path_timed_out must match the selected continuation")
    query_target_hit = reference["query_path_target_hit_observed"].to_numpy(
        dtype=bool
    )
    query_first_hit_present = reference["query_path_first_hit_FE"].notna().to_numpy()
    if not np.array_equal(query_target_hit, query_first_hit_present):
        raise ValueError(
            "query_path_target_hit_observed must equal query_path_first_hit_FE is not null"
        )
    if not np.array_equal(
        reference["query_path_success"].to_numpy(dtype=bool),
        query_target_hit,
    ):
        raise ValueError(
            "query_path_success compatibility alias must equal query_path_target_hit_observed"
        )
    if not np.array_equal(
        reference["query_path_target_hit_before_failure"].to_numpy(dtype=bool),
        query_target_hit & ~selected_completed,
    ):
        raise ValueError(
            "query_path_target_hit_before_failure must retain hits on incomplete paths"
        )
    if not np.array_equal(
        reference["query_path_endpoint_success"].to_numpy(dtype=bool),
        query_target_hit & selected_completed,
    ):
        raise ValueError(
            "query_path_endpoint_success must require both an observed hit and path completion"
        )
    incomplete = ~selected_completed
    if not np.allclose(
        reference.loc[incomplete, "p_query"].to_numpy(dtype=float),
        reference.loc[incomplete, "action_loss"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("an incomplete Stage-A Query path must retain the failure-capped action loss")
    if bool(reference.loc[incomplete, "query_sample_improved_terminal"].astype(bool).any()):
        raise ValueError("query sample best cannot overwrite an incomplete Stage-A path")
    if reference["query_id"].astype(str).nunique() != 1:
        raise ValueError("selection reference must contain exactly one query_id")
    training_splits = set(FORMAL_SELECTOR_TRAINING_SPLITS)
    train = reference[reference["split"].astype(str).isin(training_splits)]
    main_prefix = train[
        train["prefix_algorithm"].astype(str) == train["default_algorithm"].astype(str)
    ]
    cross_probe = train[
        train["prefix_algorithm"].astype(str) != train["default_algorithm"].astype(str)
    ]
    held_out = reference[~reference["split"].astype(str).isin(training_splits)]
    if main_prefix.empty or set(main_prefix["selector_prediction_source"].astype(str)) != {"cross_cv_group"}:
        raise ValueError("formal training main-prefix rows must use cross-CV-group selector predictions")
    if not cross_probe.empty and set(cross_probe["selector_prediction_source"].astype(str)) != {
        "cross_cv_group_main_prefix"
    }:
        raise ValueError(
            "formal training cross-probe rows must use main-prefix fits that exclude their CV group"
        )
    if not held_out.empty and set(held_out["selector_prediction_source"].astype(str)) != {"train_fit"}:
        raise ValueError("held-out selector rows must use the complete formal-training fit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a continuous-budget statewise action-loss selection reference."
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--train-action-losses", type=Path, action="append", required=True)
    parser.add_argument("--predict-action-losses", type=Path, action="append", default=None)
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--query-features", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--pairwise-sensitivity-output", type=Path, default=None)
    parser.add_argument("--formal-baseline-output", type=Path, default=None)
    parser.add_argument("--evaluation-summary-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("results/selection_reference") / args.query_id / "selection_reference.parquet"
    model_output = args.model_output or Path("results/selection_reference") / args.query_id / "statewise_selector.joblib"
    build_selection_reference(
        query_id=args.query_id,
        train_action_loss_paths=args.train_action_losses,
        predict_action_loss_paths=args.predict_action_losses or [],
        behavior_paths=args.behavior,
        query_feature_paths=args.query_features,
        output_path=output,
        model_output_path=model_output,
        pairwise_sensitivity_output_path=args.pairwise_sensitivity_output,
        formal_baseline_output_path=args.formal_baseline_output,
        evaluation_summary_output_path=args.evaluation_summary_output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
