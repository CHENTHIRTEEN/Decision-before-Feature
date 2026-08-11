from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from selection_reference.model import (
    SELECTION_REFERENCE_PROTOCOL,
    fit_selector_with_cross_family_predictions,
    measure_online_selection_runtime,
    prepare_state_matrix,
    read_action_loss_data,
    read_behavior_data,
    read_query_feature_data,
    save_selector_model,
    selection_rows,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


def build_selection_reference(
    *,
    query_id: str,
    train_action_loss_paths: list[Path],
    predict_action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_paths: list[Path],
    output_path: Path,
    model_output_path: Path,
) -> dict[str, int | str]:
    query_spec = get_query_spec(query_id)
    behavior = read_behavior_data(behavior_paths)
    query_features = read_query_feature_data(query_feature_paths)
    train_action_losses = read_action_loss_data(train_action_loss_paths)
    train_states, portfolio = prepare_state_matrix(
        train_action_losses,
        behavior=behavior,
        query_features=query_features,
        query_spec=query_spec,
    )
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
        _validate_no_training_overlap(train_states, predict_states)
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

    reference = pd.concat(outputs, ignore_index=True)
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
        "output": str(output_path),
        "model": str(model_output_path),
        "default_algorithm": selector_model.default_algorithm,
        "query_id": query_spec.query_id,
        "query_protocol": query_spec.protocol,
        "query_feature_columns": ",".join(query_spec.feature_columns),
        "protocol": SELECTION_REFERENCE_PROTOCOL,
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
        raise ValueError("formal selector training requires at least two function families for cross-family predictions")


def _validate_no_training_overlap(train_states: pd.DataFrame, predict_states: pd.DataFrame) -> None:
    train_splits = set(train_states["split"].astype(str))
    predict_splits = set(predict_states["split"].astype(str))
    if train_splits.intersection(predict_splits):
        raise ValueError(
            "prediction action-loss inputs overlap the training split; omit them because training rows are already emitted"
        )
    train_families = set(train_states["family"].astype(str))
    predict_families = set(predict_states["family"].astype(str))
    overlap = sorted(train_families.intersection(predict_families))
    if overlap:
        raise ValueError(f"training and held-out prediction families overlap: {overlap}")


def _validate_reference(reference: pd.DataFrame, portfolio: tuple[str, ...]) -> None:
    key = ["split", "problem_id", "family", "dimension", "prefix_algorithm", "seed", "FE"]
    if reference.empty:
        raise ValueError("selection reference contains no rows")
    if reference.duplicated(key).any():
        raise ValueError("selection reference contains duplicate shared-state keys")
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
    if not bool((reference["handoff_type"].astype(str) == reference["selected_transition_mode"].astype(str)).all()):
        raise ValueError("handoff_type must equal selected_transition_mode")
    regret = reference["selected_action_loss"].astype(float) - reference["best_observed_loss"].astype(float)
    if bool((regret < -1e-12).any()):
        raise ValueError("selector regret cannot be smaller than zero")
    if not bool((reference["selection_reference_protocol"].astype(str) == SELECTION_REFERENCE_PROTOCOL).all()):
        raise ValueError("selection reference protocol field is inconsistent")
    if reference["query_id"].astype(str).nunique() != 1:
        raise ValueError("selection reference must contain exactly one query_id")


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
    )


if __name__ == "__main__":
    main()
