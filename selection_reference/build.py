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


def build_selection_reference(
    *,
    query_id: str,
    train_action_loss_paths: list[Path],
    predict_action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_paths: list[Path],
    output_path: Path,
    model_output_path: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    existing_outputs = [path for path in (output_path, model_output_path) if path.exists()]
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
    selector_model, cross_predictions, train_prediction_source = fit_selector_with_cross_family_predictions(
        train_states,
        portfolio,
        query_spec,
    )
    outputs = [
        selection_rows(
            states=train_states,
            portfolio=portfolio,
            predictions=cross_predictions,
            prediction_source=train_prediction_source,
            runtime_selection=measure_online_selection_runtime(selector_model, train_states),
        )
    ]
    if not cross_probe_states.empty:
        cross_probe_predictions = predict_with_main_prefix_cross_family_fits(
            training_states=train_states,
            prediction_states=cross_probe_states,
            portfolio=portfolio,
            query_spec=query_spec,
        )
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
        _validate_no_training_overlap(all_train_states, predict_states)
        predictions = selector_model.predict_scores(predict_states)
        outputs.append(
            selection_rows(
                states=predict_states,
                portfolio=portfolio,
                predictions=predictions,
                prediction_source="train_fit",
                runtime_selection=measure_online_selection_runtime(selector_model, predict_states),
            )
        )

    reference = pd.concat(outputs, ignore_index=True).sort_values(
        ["split", "problem_id", "dimension", "prefix_algorithm", "seed", "FE"]
    ).reset_index(drop=True)
    _validate_reference(reference, portfolio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(reference, preserve_index=False), output_path)
    save_selector_model(selector_model, model_output_path)
    print(f"wrote {len(reference)} statewise selection-reference rows to {output_path}")
    print(f"wrote statewise selector model to {model_output_path}")
    print(f"SBS/default optimizer: {selector_model.default_algorithm}")
    return {
        "rows": int(len(reference)),
        "training_rows": int(len(train_states)),
        "cross_probe_training_rows": int(len(cross_probe_states)),
        "output": str(output_path),
        "model": str(model_output_path),
        "default_algorithm": selector_model.default_algorithm,
        "query_id": query_spec.query_id,
        "query_protocol": query_spec.protocol,
        "query_feature_columns": ",".join(query_spec.feature_columns),
        "protocol": SELECTION_REFERENCE_PROTOCOL,
        "selector_target_transform": SELECTOR_TARGET_TRANSFORM,
    }


def _validate_training_scope(states: pd.DataFrame) -> None:
    if set(states["split"].astype(str)) != {"bbob_train"}:
        raise ValueError("formal selector fitting must use only the bbob_train split")
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
    regret = reference["selected_action_loss"].astype(float) - reference["best_observed_loss"].astype(float)
    if bool((regret < -1e-12).any()):
        raise ValueError("selector regret cannot be smaller than zero")
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
        reference.loc[incomplete, "selected_action_loss"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("an incomplete Stage-A Query path must retain the failure-capped action loss")
    if bool(reference.loc[incomplete, "query_sample_improved_terminal"].astype(bool).any()):
        raise ValueError("query sample best cannot overwrite an incomplete Stage-A path")
    if reference["query_id"].astype(str).nunique() != 1:
        raise ValueError("selection reference must contain exactly one query_id")
    train = reference[reference["split"].astype(str) == "bbob_train"]
    main_prefix = train[
        train["prefix_algorithm"].astype(str) == train["default_algorithm"].astype(str)
    ]
    cross_probe = train[
        train["prefix_algorithm"].astype(str) != train["default_algorithm"].astype(str)
    ]
    held_out = reference[reference["split"].astype(str) != "bbob_train"]
    if main_prefix.empty or set(main_prefix["selector_prediction_source"].astype(str)) != {"cross_cv_group"}:
        raise ValueError("BBOB-train main-prefix rows must use cross-CV-group selector predictions")
    if not cross_probe.empty and set(cross_probe["selector_prediction_source"].astype(str)) != {
        "cross_cv_group_main_prefix"
    }:
        raise ValueError(
            "BBOB-train cross-probe rows must use main-prefix fits that exclude their CV group"
        )
    if not held_out.empty and set(held_out["selector_prediction_source"].astype(str)) != {"train_fit"}:
        raise ValueError("held-out selector rows must use the complete BBOB-train fit")


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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
