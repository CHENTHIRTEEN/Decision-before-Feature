from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from landscape_queries.specs import LandscapeQuerySpec, get_query_spec
from selection_reference.action_losses import (
    FULL_REMAINING_BUDGET,
    QUERY_ADJUSTED_BUDGET,
    STATE_KEY_COLUMNS,
)
from selection_reference.common import single_best_solver
from selection_reference.model import (
    BEHAVIOR_ONLY_FULL_BUDGET_INPUT,
    BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL,
    QUERY_FULL_INPUT,
    QUERY_ONLY_INPUT,
    QUERY_ONLY_SELECTION_REFERENCE_PROTOCOL,
    SELECTION_REFERENCE_PROTOCOL,
    STATE_ONLY_INPUT,
    PRE_RUN_QUERY_ONLY_INPUT,
    PRE_RUN_STATE_KEY_COLUMNS,
    StatewiseSelectorModel,
    behavior_only_selection_rows,
    fit_selector_with_cross_family_predictions,
    measure_online_selection_runtime,
    prepare_state_matrix,
    prepare_pre_run_state_matrix,
    pre_run_selection_rows,
    sampling_only_continue_current_rows,
    read_action_loss_data,
    read_behavior_data,
    read_query_feature_data,
    selection_rows,
)
from utility_labels.generation import paired_utility_label_view
from utility_labels.fields import TIMING_REPLAY_STATUSES


TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
FOLD_SELECTOR_MODES = (QUERY_FULL_INPUT, STATE_ONLY_INPUT, QUERY_ONLY_INPUT)
EPS = 1e-12
COMPLETE_PATH_TIMING_SOURCE = "measured_complete_policy_path"
COMPLETE_PATH_TIMING_ORIGIN = "decision_state_to_terminal"
COMPLETE_PATH_REPLAY_PROTOCOL = "fold_specific_selected_complete_policy_paths_v1"
COMPLETE_PATHS = (
    "skip",
    "query_joint",
    "query_matched_state_only",
    "sampling_only_continue_current",
    "behavior_only_full_budget",
)
TIMING_REPETITIONS = 3
TIMING_ORDER_PROTOCOL = "cyclic_complete_path_v1"


@dataclass(frozen=True)
class PreparedNestedLearningInputs:
    query_spec: LandscapeQuerySpec
    performance: pd.DataFrame
    query_adjusted_states: pd.DataFrame
    behavior_only_states: pd.DataFrame
    portfolio: tuple[str, ...]
    complete_path_timings: pd.DataFrame | None
    log10_gap_floor: float
    log10_gap_cap: float
    pre_run_states: pd.DataFrame | None = None


@dataclass
class FoldLearningViews:
    sbs_algorithm: str
    fit_labels: pd.DataFrame
    holdout_labels: pd.DataFrame
    query_selector: StatewiseSelectorModel
    behavior_only_selector: StatewiseSelectorModel
    query_only_selector: StatewiseSelectorModel
    state_only_selector: StatewiseSelectorModel
    selector_summary: pd.DataFrame
    query_state_only_fit_rows: pd.DataFrame
    query_state_only_holdout_rows: pd.DataFrame
    pre_run_query_only_selector: StatewiseSelectorModel | None = None
    pre_run_fit_rows: pd.DataFrame | None = None
    pre_run_holdout_rows: pd.DataFrame | None = None
    replay_plan: pd.DataFrame | None = None


@dataclass
class FoldSelectorViews:
    fold_role: str
    fit_split: str
    holdout_split: str
    fit_families: tuple[str, ...]
    holdout_families: tuple[str, ...]
    sbs_algorithm: str
    query_selector: StatewiseSelectorModel
    behavior_only_selector: StatewiseSelectorModel
    query_only_selector: StatewiseSelectorModel
    state_only_selector: StatewiseSelectorModel
    selector_summary: pd.DataFrame
    query_fit_states: pd.DataFrame
    query_holdout_states: pd.DataFrame
    query_fit_rows: pd.DataFrame
    query_holdout_rows: pd.DataFrame
    behavior_fit_rows: pd.DataFrame
    behavior_holdout_rows: pd.DataFrame
    state_only_fit_rows: pd.DataFrame
    state_only_holdout_rows: pd.DataFrame
    sampling_only_fit_rows: pd.DataFrame
    sampling_only_holdout_rows: pd.DataFrame
    replay_plan: pd.DataFrame
    pre_run_query_only_selector: StatewiseSelectorModel | None = None
    pre_run_fit_rows: pd.DataFrame | None = None
    pre_run_holdout_rows: pd.DataFrame | None = None


def prepare_nested_learning_inputs(
    *,
    query_id: str,
    performance: pd.DataFrame,
    query_action_loss_paths: list[Path],
    behavior_action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_paths: list[Path],
    complete_path_timing_paths: list[Path] | None,
    log10_gap_floor: float,
    log10_gap_cap: float,
    pre_run_action_loss_paths: list[Path] | None = None,
) -> PreparedNestedLearningInputs:
    query_spec = get_query_spec(query_id)
    behavior = read_behavior_data(behavior_paths)
    query_features = read_query_feature_data(query_feature_paths)
    query_states, query_portfolio = prepare_state_matrix(
        read_action_loss_data(query_action_loss_paths),
        behavior=behavior,
        query_features=query_features,
        query_spec=query_spec,
        action_budget_mode=QUERY_ADJUSTED_BUDGET,
    )
    behavior_states, behavior_portfolio = prepare_state_matrix(
        read_action_loss_data(behavior_action_loss_paths),
        behavior=behavior,
        query_features=None,
        query_spec=query_spec,
        action_budget_mode=FULL_REMAINING_BUDGET,
    )
    if query_portfolio != behavior_portfolio:
        raise ValueError("query-adjusted and behavior-only outcomes use different portfolios")
    _check_all_prefix_coverage(query_states, query_portfolio, "query-adjusted")
    _check_all_prefix_coverage(behavior_states, behavior_portfolio, "behavior-only")
    _check_outcome_state_alignment(query_states, behavior_states)
    _check_performance_coverage(performance, query_states, query_portfolio)
    complete_path_timings = (
        _read_complete_path_timings(complete_path_timing_paths)
        if complete_path_timing_paths
        else None
    )
    pre_run_states: pd.DataFrame | None = None
    if pre_run_action_loss_paths:
        pre_run_states, pre_run_portfolio = prepare_pre_run_state_matrix(
            read_action_loss_data(pre_run_action_loss_paths),
            query_features=query_features,
            query_spec=query_spec,
        )
        if pre_run_portfolio != query_portfolio:
            raise ValueError("pre-run and online outcome matrices use different portfolios")
    return PreparedNestedLearningInputs(
        query_spec=query_spec,
        performance=performance.copy(),
        query_adjusted_states=query_states,
        behavior_only_states=behavior_states,
        portfolio=query_portfolio,
        complete_path_timings=complete_path_timings,
        log10_gap_floor=float(log10_gap_floor),
        log10_gap_cap=float(log10_gap_cap),
        pre_run_states=pre_run_states,
    )


def build_fold_learning_views(
    *,
    inputs: PreparedNestedLearningInputs,
    fit_families: tuple[str, ...],
    holdout_families: tuple[str, ...],
    fit_split: str = TRAIN_SPLIT,
    holdout_split: str = TRAIN_SPLIT,
    fold_role: str,
) -> FoldLearningViews:
    selector_views = fit_fold_selectors(
        inputs=inputs,
        fit_families=fit_families,
        holdout_families=holdout_families,
        fit_split=fit_split,
        holdout_split=holdout_split,
        fold_role=fold_role,
    )
    return build_fold_utility_labels(inputs=inputs, selector_views=selector_views)


def fit_fold_selectors(
    *,
    inputs: PreparedNestedLearningInputs,
    fit_families: tuple[str, ...],
    holdout_families: tuple[str, ...],
    fit_split: str = TRAIN_SPLIT,
    holdout_split: str = TRAIN_SPLIT,
    fold_role: str,
) -> FoldSelectorViews:
    """Fit fold-specific selectors using ``cv_group_id`` as the grouping field.

    The ``fit_families`` / ``holdout_families`` parameter names are retained
    for backward compatibility but now carry CV group IDs (function IDs).
    """
    fit_family_set = set(str(value) for value in fit_families)
    holdout_family_set = set(str(value) for value in holdout_families)
    if not fit_family_set or not holdout_family_set:
        raise ValueError("fold-specific learning requires non-empty fit and holdout CV groups")
    if fit_split == holdout_split and fit_family_set.intersection(holdout_family_set):
        raise ValueError("fold-specific fit and holdout CV groups overlap")
    group_column = "cv_group_id" if "cv_group_id" in inputs.performance.columns else "family"
    performance_fit = inputs.performance[
        inputs.performance[group_column].astype(str).isin(fit_family_set)
    ].copy()
    sbs_algorithm = single_best_solver(performance_fit, portfolio_order=inputs.portfolio)
    query_fit = _fold_state_view(
        inputs.query_adjusted_states,
        split=fit_split,
        families=fit_family_set,
        sbs_algorithm=sbs_algorithm,
    )
    query_holdout = _fold_state_view(
        inputs.query_adjusted_states,
        split=holdout_split,
        families=holdout_family_set,
        sbs_algorithm=sbs_algorithm,
    )
    behavior_fit = _fold_state_view(
        inputs.behavior_only_states,
        split=fit_split,
        families=fit_family_set,
        sbs_algorithm=sbs_algorithm,
    )
    behavior_holdout = _fold_state_view(
        inputs.behavior_only_states,
        split=holdout_split,
        families=holdout_family_set,
        sbs_algorithm=sbs_algorithm,
    )
    _check_selected_prefix_alignment(query_fit, behavior_fit, "fit")
    _check_selected_prefix_alignment(query_holdout, behavior_holdout, "holdout")

    query_outputs: dict[str, tuple[StatewiseSelectorModel, pd.DataFrame, pd.DataFrame]] = {}
    selector_summaries: list[pd.DataFrame] = []
    for selector_input_mode in FOLD_SELECTOR_MODES:
        selector_model, cross_predictions, prediction_source = (
            fit_selector_with_cross_family_predictions(
                query_fit,
                inputs.portfolio,
                inputs.query_spec,
                selector_input_mode=selector_input_mode,
            )
        )
        runtime_selection = measure_online_selection_runtime(selector_model, query_fit)
        selector_protocol = (
            QUERY_ONLY_SELECTION_REFERENCE_PROTOCOL
            if selector_input_mode == QUERY_ONLY_INPUT
            else SELECTION_REFERENCE_PROTOCOL
        )
        fit_rows = selection_rows(
            states=query_fit,
            portfolio=inputs.portfolio,
            predictions=cross_predictions,
            prediction_source=f"{fold_role}_fit_cross_cv_group",
            runtime_selection=runtime_selection,
            selector_input_mode=selector_input_mode,
            selection_reference_protocol=selector_protocol,
        )
        holdout_rows = selection_rows(
            states=query_holdout,
            portfolio=inputs.portfolio,
            predictions=selector_model.predict_scores(query_holdout),
            prediction_source=f"{fold_role}_fit",
            runtime_selection=runtime_selection,
            selector_input_mode=selector_input_mode,
            selection_reference_protocol=selector_protocol,
        )
        query_outputs[selector_input_mode] = (selector_model, fit_rows, holdout_rows)
        selector_summaries.append(
            _selector_performance_summary(
                fit_rows,
                selector_input_mode=selector_input_mode,
                evaluation_role="fit_cross_cv_group",
                fold_role=fold_role,
                sbs_algorithm=sbs_algorithm,
            )
        )
        selector_summaries.append(
            _selector_performance_summary(
                holdout_rows,
                selector_input_mode=selector_input_mode,
                evaluation_role="holdout",
                fold_role=fold_role,
                sbs_algorithm=sbs_algorithm,
            )
        )

    behavior_model, behavior_cross_predictions, _ = fit_selector_with_cross_family_predictions(
        behavior_fit,
        inputs.portfolio,
        inputs.query_spec,
        selector_input_mode=BEHAVIOR_ONLY_FULL_BUDGET_INPUT,
    )
    behavior_runtime = measure_online_selection_runtime(behavior_model, behavior_fit)
    behavior_fit_rows = behavior_only_selection_rows(
        states=behavior_fit,
        portfolio=inputs.portfolio,
        predictions=behavior_cross_predictions,
        prediction_source=f"{fold_role}_fit_cross_cv_group",
        runtime_selection=behavior_runtime,
    )
    behavior_holdout_rows = behavior_only_selection_rows(
        states=behavior_holdout,
        portfolio=inputs.portfolio,
        predictions=behavior_model.predict_scores(behavior_holdout),
        prediction_source=f"{fold_role}_fit",
        runtime_selection=behavior_runtime,
    )
    selector_summaries.append(
        _selector_performance_summary(
            behavior_fit_rows,
            selector_input_mode=BEHAVIOR_ONLY_FULL_BUDGET_INPUT,
            evaluation_role="fit_cross_cv_group",
            fold_role=fold_role,
            sbs_algorithm=sbs_algorithm,
        )
    )
    selector_summaries.append(
        _selector_performance_summary(
            behavior_holdout_rows,
            selector_input_mode=BEHAVIOR_ONLY_FULL_BUDGET_INPUT,
            evaluation_role="holdout",
            fold_role=fold_role,
            sbs_algorithm=sbs_algorithm,
        )
    )

    query_model, query_fit_rows, query_holdout_rows = query_outputs[QUERY_FULL_INPUT]
    state_only_fit_rows = query_outputs[STATE_ONLY_INPUT][1]
    state_only_holdout_rows = query_outputs[STATE_ONLY_INPUT][2]
    sampling_only_fit_rows = sampling_only_continue_current_rows(
        states=query_fit,
        portfolio=inputs.portfolio,
    )
    sampling_only_holdout_rows = sampling_only_continue_current_rows(
        states=query_holdout,
        portfolio=inputs.portfolio,
    )
    pre_run_model: StatewiseSelectorModel | None = None
    pre_run_fit_rows: pd.DataFrame | None = None
    pre_run_holdout_rows: pd.DataFrame | None = None
    if inputs.pre_run_states is not None:
        pre_run_fit = _pre_run_fold_view(
            inputs.pre_run_states,
            split=fit_split,
            families=fit_family_set,
            sbs_algorithm=sbs_algorithm,
        )
        pre_run_holdout = _pre_run_fold_view(
            inputs.pre_run_states,
            split=holdout_split,
            families=holdout_family_set,
            sbs_algorithm=sbs_algorithm,
        )
        pre_run_model, pre_run_cross_predictions, _ = fit_selector_with_cross_family_predictions(
            pre_run_fit,
            inputs.portfolio,
            inputs.query_spec,
            selector_input_mode=PRE_RUN_QUERY_ONLY_INPUT,
        )
        pre_run_runtime = measure_online_selection_runtime(pre_run_model, pre_run_fit)
        pre_run_fit_rows = pre_run_selection_rows(
            states=pre_run_fit,
            portfolio=inputs.portfolio,
            predictions=pre_run_cross_predictions,
            prediction_source=f"{fold_role}_fit_cross_cv_group",
            runtime_selection=pre_run_runtime,
        )
        pre_run_holdout_rows = pre_run_selection_rows(
            states=pre_run_holdout,
            portfolio=inputs.portfolio,
            predictions=pre_run_model.predict_scores(pre_run_holdout),
            prediction_source=f"{fold_role}_fit",
            runtime_selection=pre_run_runtime,
        )
        selector_summaries.append(
            _selector_performance_summary(
                pre_run_fit_rows,
                selector_input_mode=PRE_RUN_QUERY_ONLY_INPUT,
                evaluation_role="fit_cross_cv_group",
                fold_role=fold_role,
                sbs_algorithm=sbs_algorithm,
            )
        )
        selector_summaries.append(
            _selector_performance_summary(
                pre_run_holdout_rows,
                selector_input_mode=PRE_RUN_QUERY_ONLY_INPUT,
                evaluation_role="holdout",
                fold_role=fold_role,
                sbs_algorithm=sbs_algorithm,
            )
        )

    replay_plan = pd.concat(
        [
            _selected_complete_path_replay_plan(
                query_rows=query_fit_rows,
                behavior_rows=behavior_fit_rows,
                state_only_rows=state_only_fit_rows,
                sampling_only_rows=sampling_only_fit_rows,
                learning_fold_role=f"{fold_role}_fit_cross_cv_group",
            ),
            _selected_complete_path_replay_plan(
                query_rows=query_holdout_rows,
                behavior_rows=behavior_holdout_rows,
                state_only_rows=state_only_holdout_rows,
                sampling_only_rows=sampling_only_holdout_rows,
                learning_fold_role=f"{fold_role}_holdout",
            ),
        ],
        ignore_index=True,
    )
    _check_replay_plan(replay_plan)
    return FoldSelectorViews(
        fold_role=str(fold_role),
        fit_split=str(fit_split),
        holdout_split=str(holdout_split),
        fit_families=tuple(sorted(fit_family_set)),
        holdout_families=tuple(sorted(holdout_family_set)),
        sbs_algorithm=sbs_algorithm,
        query_selector=query_model,
        behavior_only_selector=behavior_model,
        query_only_selector=query_outputs[QUERY_ONLY_INPUT][0],
        state_only_selector=query_outputs[STATE_ONLY_INPUT][0],
        selector_summary=pd.concat(selector_summaries, ignore_index=True),
        query_fit_states=query_fit,
        query_holdout_states=query_holdout,
        query_fit_rows=query_fit_rows,
        query_holdout_rows=query_holdout_rows,
        behavior_fit_rows=behavior_fit_rows,
        behavior_holdout_rows=behavior_holdout_rows,
        state_only_fit_rows=state_only_fit_rows,
        state_only_holdout_rows=state_only_holdout_rows,
        sampling_only_fit_rows=sampling_only_fit_rows,
        sampling_only_holdout_rows=sampling_only_holdout_rows,
        replay_plan=replay_plan,
        pre_run_query_only_selector=pre_run_model,
        pre_run_fit_rows=pre_run_fit_rows,
        pre_run_holdout_rows=pre_run_holdout_rows,
    )


def build_fold_utility_labels(
    *,
    inputs: PreparedNestedLearningInputs,
    selector_views: FoldSelectorViews,
) -> FoldLearningViews:
    if inputs.complete_path_timings is None:
        raise ValueError(
            "fold Utility labels require measured decision-state-to-terminal complete-path "
            "timings; first emit and execute selector_views.replay_plan"
        )
    fit_family_set = set(selector_views.fit_families)
    holdout_family_set = set(selector_views.holdout_families)
    fit_labels = paired_utility_label_view(
        query_selection=selector_views.query_fit_rows,
        behavior_selection=selector_views.behavior_fit_rows,
        query_adjusted_behavior_selection=selector_views.state_only_fit_rows,
        sampling_only_selection=selector_views.sampling_only_fit_rows,
        complete_path_timings=_fold_timing_view(
            inputs.complete_path_timings,
            learning_fold_role=f"{selector_views.fold_role}_fit_cross_cv_group",
            split=selector_views.fit_split,
            families=fit_family_set,
        ),
        query_id=inputs.query_spec.query_id,
        log10_gap_floor=inputs.log10_gap_floor,
        log10_gap_cap=inputs.log10_gap_cap,
    )
    holdout_labels = paired_utility_label_view(
        query_selection=selector_views.query_holdout_rows,
        behavior_selection=selector_views.behavior_holdout_rows,
        query_adjusted_behavior_selection=selector_views.state_only_holdout_rows,
        sampling_only_selection=selector_views.sampling_only_holdout_rows,
        complete_path_timings=_fold_timing_view(
            inputs.complete_path_timings,
            learning_fold_role=f"{selector_views.fold_role}_holdout",
            split=selector_views.holdout_split,
            families=holdout_family_set,
        ),
        query_id=inputs.query_spec.query_id,
        log10_gap_floor=inputs.log10_gap_floor,
        log10_gap_cap=inputs.log10_gap_cap,
    )
    fit_labels = _attach_decision_features(fit_labels, selector_views.query_fit_states)
    holdout_labels = _attach_decision_features(
        holdout_labels,
        selector_views.query_holdout_states,
    )
    _check_label_family_role(
        fit_labels,
        split=selector_views.fit_split,
        families=fit_family_set,
        role="fit",
    )
    _check_label_family_role(
        holdout_labels,
        split=selector_views.holdout_split,
        families=holdout_family_set,
        role="holdout",
    )
    return FoldLearningViews(
        sbs_algorithm=selector_views.sbs_algorithm,
        fit_labels=fit_labels,
        holdout_labels=holdout_labels,
        query_selector=selector_views.query_selector,
        behavior_only_selector=selector_views.behavior_only_selector,
        query_only_selector=selector_views.query_only_selector,
        state_only_selector=selector_views.state_only_selector,
        selector_summary=selector_views.selector_summary,
        query_state_only_fit_rows=selector_views.state_only_fit_rows,
        query_state_only_holdout_rows=selector_views.state_only_holdout_rows,
        pre_run_query_only_selector=selector_views.pre_run_query_only_selector,
        pre_run_fit_rows=selector_views.pre_run_fit_rows,
        pre_run_holdout_rows=selector_views.pre_run_holdout_rows,
        replay_plan=selector_views.replay_plan,
    )


def family_fold_partitions(
    *,
    families: tuple[str, ...],
    requested_folds: int,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Create grouped folds that hold out complete landscape families.

    Retained for the leave-one-landscape-family-out secondary robustness
    analysis.  The main experiment uses :func:`cv_group_fold_partitions`.
    """
    ordered_families = tuple(sorted(set(str(value) for value in families)))
    n_splits = min(int(requested_folds), len(ordered_families))
    if n_splits < 2:
        raise ValueError("landscape-family folds require at least two families")
    family_frame = pd.DataFrame({"family": ordered_families})
    groups = family_frame["family"].to_numpy(dtype=str)
    partitions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fit_index, holdout_index in splitter.split(family_frame, groups=groups):
        fit = tuple(sorted(family_frame.iloc[fit_index]["family"].astype(str)))
        holdout = tuple(sorted(family_frame.iloc[holdout_index]["family"].astype(str)))
        if set(fit).intersection(holdout):
            raise RuntimeError("landscape-family fold partition overlaps")
        partitions.append((fit, holdout))
    return tuple(partitions)


def cv_group_fold_partitions(
    *,
    cv_groups: tuple[str, ...],
    requested_folds: int,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Create grouped folds that hold out complete CV groups (function IDs)."""
    ordered_groups = tuple(sorted(set(str(value) for value in cv_groups)))
    n_splits = min(int(requested_folds), len(ordered_groups))
    if n_splits < 2:
        raise ValueError("CV-group folds require at least two groups")
    group_frame = pd.DataFrame({"cv_group_id": ordered_groups})
    groups = group_frame["cv_group_id"].to_numpy(dtype=str)
    partitions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fit_index, holdout_index in splitter.split(group_frame, groups=groups):
        fit = tuple(sorted(group_frame.iloc[fit_index]["cv_group_id"].astype(str)))
        holdout = tuple(sorted(group_frame.iloc[holdout_index]["cv_group_id"].astype(str)))
        if set(fit).intersection(holdout):
            raise RuntimeError("CV-group fold partition overlaps")
        partitions.append((fit, holdout))
    return tuple(partitions)


def build_required_replay_plan(
    *,
    inputs: PreparedNestedLearningInputs,
    outer_folds: int,
    inner_folds: int,
) -> pd.DataFrame:
    group_column = (
        "cv_group_id"
        if "cv_group_id" in inputs.query_adjusted_states.columns
        else "family"
    )
    train_families = tuple(
        sorted(
            set(
                inputs.query_adjusted_states.loc[
                    inputs.query_adjusted_states["split"].astype(str).eq(TRAIN_SPLIT),
                    group_column,
                ].astype(str)
            )
        )
    )
    validation_families = tuple(
        sorted(
            set(
                inputs.query_adjusted_states.loc[
                    inputs.query_adjusted_states["split"].astype(str).eq(VALIDATION_SPLIT),
                    group_column,
                ].astype(str)
            )
        )
    )
    plans = [
        fit_fold_selectors(
            inputs=inputs,
            fit_families=train_families,
            holdout_families=validation_families,
            fit_split=TRAIN_SPLIT,
            holdout_split=VALIDATION_SPLIT,
            fold_role="full_train_final",
        ).replay_plan
    ]
    outer_partitions = cv_group_fold_partitions(
        cv_groups=train_families,
        requested_folds=outer_folds,
    )
    for outer_fold, (outer_fit_families, outer_holdout_families) in enumerate(
        outer_partitions
    ):
        outer_role = f"train_outer_{len(outer_partitions)}_fold_{outer_fold}"
        plans.append(
            fit_fold_selectors(
                inputs=inputs,
                fit_families=outer_fit_families,
                holdout_families=outer_holdout_families,
                fold_role=outer_role,
            ).replay_plan
        )
        inner_partitions = cv_group_fold_partitions(
            cv_groups=outer_fit_families,
            requested_folds=inner_folds,
        )
        for inner_fold, (inner_fit_families, inner_holdout_families) in enumerate(
            inner_partitions
        ):
            inner_role = (
                f"{outer_role}_inner_{len(inner_partitions)}_fold_{inner_fold}"
            )
            plans.append(
                fit_fold_selectors(
                    inputs=inputs,
                    fit_families=inner_fit_families,
                    holdout_families=inner_holdout_families,
                    fold_role=inner_role,
                ).replay_plan
            )
    replay_plan = pd.concat(plans, ignore_index=True)
    _check_replay_plan(replay_plan)
    duplicate_key = ["learning_fold_role", *STATE_KEY_COLUMNS, "path"]
    if replay_plan.duplicated(duplicate_key).any():
        raise RuntimeError("required replay plan contains duplicate fold/state/path rows")
    path_order = {path: index for index, path in enumerate(COMPLETE_PATHS)}
    replay_plan = replay_plan.assign(
        _path_order=replay_plan["path"].astype(str).map(path_order).astype(int)
    )
    return replay_plan.sort_values(
        ["learning_fold_role", *STATE_KEY_COLUMNS, "_path_order"],
        kind="mergesort",
    ).drop(columns="_path_order").reset_index(drop=True)


def _selected_complete_path_replay_plan(
    *,
    query_rows: pd.DataFrame,
    behavior_rows: pd.DataFrame,
    state_only_rows: pd.DataFrame,
    sampling_only_rows: pd.DataFrame,
    learning_fold_role: str,
) -> pd.DataFrame:
    key = list(STATE_KEY_COLUMNS)
    query_required = {
        *key,
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "sample_design_id",
        "FE_total",
        "default_algorithm",
        "no_query_algorithm",
        "no_query_transition_mode",
        "p_skip",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "skip_timed_out",
        "skip_path_completed",
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "p_query",
        "query_path_first_hit_FE",
        "query_path_success",
        "query_path_planned_FE",
        "query_path_effective_FE",
        "query_path_timed_out",
        "query_path_completed",
        "selector_prediction_source",
        "action_budget_mode",
    }
    behavior_required = {
        *key,
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "p_behavior",
        "behavior_path_first_hit_FE",
        "behavior_path_success",
        "behavior_path_planned_FE",
        "behavior_path_effective_FE",
        "behavior_path_timed_out",
        "behavior_path_completed",
        "selector_prediction_source",
        "action_budget_mode",
    }
    missing_query = sorted(query_required.difference(query_rows.columns))
    missing_behavior = sorted(behavior_required.difference(behavior_rows.columns))
    missing_state_only = sorted(query_required.difference(state_only_rows.columns))
    missing_sampling_only = sorted(query_required.difference(sampling_only_rows.columns))
    if missing_query or missing_behavior or missing_state_only or missing_sampling_only:
        raise ValueError(
            "cannot emit complete-path replay plan; missing Selector columns: "
            f"query={missing_query}, behavior={missing_behavior}, "
            f"state_only={missing_state_only}, sampling_only={missing_sampling_only}"
        )
    if any(
        frame.duplicated(key).any()
        for frame in (query_rows, behavior_rows, state_only_rows, sampling_only_rows)
    ):
        raise ValueError("replay planning requires unique state keys for every path")
    query_keys = query_rows[key].sort_values(key).reset_index(drop=True)
    for label, frame in (
        ("behavior", behavior_rows),
        ("state-only", state_only_rows),
        ("sampling-only", sampling_only_rows),
    ):
        observed_keys = frame[key].sort_values(key).reset_index(drop=True)
        if not query_keys.equals(observed_keys):
            raise ValueError(f"query and {label} paths are not aligned for replay planning")

    identity_columns = [
        *key,
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "sample_design_id",
        "FE_total",
        "default_algorithm",
    ]
    identity = query_rows[identity_columns].copy()

    skip = identity.copy()
    skip["path"] = "skip"
    skip["canonical_path_order"] = 0
    skip["selected_algorithm"] = query_rows["no_query_algorithm"].astype(str).to_numpy()
    skip["selected_action"] = np.where(
        skip["selected_algorithm"].astype(str) == skip["prefix_algorithm"].astype(str),
        "continue_current",
        skip["selected_algorithm"].astype(str),
    )
    skip["selected_equals_default"] = (
        skip["selected_algorithm"].astype(str) == skip["default_algorithm"].astype(str)
    )
    skip["selected_equals_prefix"] = (
        skip["selected_algorithm"].astype(str) == skip["prefix_algorithm"].astype(str)
    )
    skip["handoff_required"] = ~skip["selected_equals_prefix"].astype(bool)
    skip["handoff_type"] = query_rows["no_query_transition_mode"].astype(str).to_numpy()
    skip["action_budget_mode"] = "native_remaining_budget"
    skip["selector_prediction_source"] = "not_applicable_skip"
    skip["selector_artifact_role"] = "not_applicable"
    skip["descriptor_computation_required"] = False
    skip["expected_terminal_gap"] = query_rows["p_skip"].to_numpy(dtype=float)
    skip["expected_first_hit_FE"] = query_rows["skip_first_hit_FE"].to_numpy()
    skip["expected_success"] = query_rows["skip_success"].to_numpy(dtype=bool)
    skip["expected_planned_FE"] = query_rows["skip_planned_FE"].to_numpy(dtype=int)
    skip["expected_effective_FE"] = query_rows["skip_effective_FE"].to_numpy(dtype=int)
    skip["expected_timed_out"] = query_rows["skip_timed_out"].to_numpy(dtype=bool)
    skip["expected_path_completed"] = query_rows["skip_path_completed"].to_numpy(
        dtype=bool
    )

    def query_acquisition_path(
        rows: pd.DataFrame,
        *,
        path: str,
        canonical_order: int,
        descriptor_computation_required: bool,
    ) -> pd.DataFrame:
        frame = identity.copy()
        frame["path"] = path
        frame["canonical_path_order"] = int(canonical_order)
        for column in (
            "selected_algorithm",
            "selected_action",
            "selected_equals_default",
            "selected_equals_prefix",
            "handoff_required",
            "handoff_type",
            "selector_prediction_source",
            "action_budget_mode",
        ):
            frame[column] = rows[column].to_numpy()
        frame["selector_artifact_role"] = rows["selector_prediction_source"].astype(
            str
        ).to_numpy()
        frame["descriptor_computation_required"] = bool(
            descriptor_computation_required
        )
        frame["expected_terminal_gap"] = rows["p_query"].to_numpy(dtype=float)
        frame["expected_first_hit_FE"] = rows["query_path_first_hit_FE"].to_numpy()
        frame["expected_success"] = rows["query_path_success"].to_numpy(dtype=bool)
        frame["expected_planned_FE"] = rows["query_path_planned_FE"].to_numpy(dtype=int)
        frame["expected_effective_FE"] = rows["query_path_effective_FE"].to_numpy(
            dtype=int
        )
        frame["expected_timed_out"] = rows["query_path_timed_out"].to_numpy(dtype=bool)
        frame["expected_path_completed"] = rows["query_path_completed"].to_numpy(
            dtype=bool
        )
        return frame

    query = query_acquisition_path(
        query_rows,
        path="query_joint",
        canonical_order=1,
        descriptor_computation_required=True,
    )
    state_only = query_acquisition_path(
        state_only_rows,
        path="query_matched_state_only",
        canonical_order=2,
        descriptor_computation_required=True,
    )
    sampling_only = query_acquisition_path(
        sampling_only_rows,
        path="sampling_only_continue_current",
        canonical_order=3,
        descriptor_computation_required=False,
    )
    if not sampling_only["selected_equals_prefix"].astype(bool).all():
        raise ValueError("sampling-only replay path must continue the prefix algorithm")

    behavior_columns = [
        *key,
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "selector_prediction_source",
        "action_budget_mode",
        "p_behavior",
        "behavior_path_first_hit_FE",
        "behavior_path_success",
        "behavior_path_planned_FE",
        "behavior_path_effective_FE",
        "behavior_path_timed_out",
        "behavior_path_completed",
    ]
    behavior = identity.merge(
        behavior_rows[behavior_columns],
        on=key,
        how="left",
        validate="one_to_one",
    )
    behavior["path"] = "behavior_only_full_budget"
    behavior["canonical_path_order"] = 4
    behavior["selector_artifact_role"] = behavior[
        "selector_prediction_source"
    ].astype(str)
    behavior["descriptor_computation_required"] = False
    behavior["expected_terminal_gap"] = behavior.pop("p_behavior").to_numpy(dtype=float)
    behavior["expected_first_hit_FE"] = behavior.pop("behavior_path_first_hit_FE")
    behavior["expected_success"] = behavior.pop("behavior_path_success").to_numpy(dtype=bool)
    behavior["expected_planned_FE"] = behavior.pop("behavior_path_planned_FE").to_numpy(dtype=int)
    behavior["expected_effective_FE"] = behavior.pop("behavior_path_effective_FE").to_numpy(dtype=int)
    behavior["expected_timed_out"] = behavior.pop(
        "behavior_path_timed_out"
    ).to_numpy(dtype=bool)
    behavior["expected_path_completed"] = behavior.pop(
        "behavior_path_completed"
    ).to_numpy(dtype=bool)

    plan = pd.concat(
        [skip, query, state_only, sampling_only, behavior],
        ignore_index=True,
    )
    plan.insert(0, "replay_protocol", COMPLETE_PATH_REPLAY_PROTOCOL)
    plan.insert(1, "learning_fold_role", str(learning_fold_role))
    plan["timing_origin"] = COMPLETE_PATH_TIMING_ORIGIN
    plan["timing_repetitions"] = TIMING_REPETITIONS
    plan["timing_order_protocol"] = TIMING_ORDER_PROTOCOL
    plan["required_timing_source"] = COMPLETE_PATH_TIMING_SOURCE
    plan["shared_prefix_cost_treatment"] = "sunk_before_decision_state"
    plan["physical_execution_scope"] = "state_action_continuation"
    return plan


def _check_replay_plan(replay_plan: pd.DataFrame) -> None:
    required = {
        "replay_protocol",
        "learning_fold_role",
        *STATE_KEY_COLUMNS,
        "path",
        "canonical_path_order",
        "selected_algorithm",
        "selected_action",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "expected_terminal_gap",
        "expected_planned_FE",
        "expected_effective_FE",
        "expected_timed_out",
        "expected_path_completed",
        "timing_origin",
        "timing_repetitions",
        "timing_order_protocol",
        "required_timing_source",
    }
    missing = sorted(required.difference(replay_plan.columns))
    if missing:
        raise ValueError(f"complete-path replay plan is missing columns: {missing}")
    if replay_plan.empty:
        raise ValueError("complete-path replay plan must not be empty")
    if set(replay_plan["replay_protocol"].astype(str)) != {COMPLETE_PATH_REPLAY_PROTOCOL}:
        raise ValueError("complete-path replay plan uses an unsupported protocol")
    if set(replay_plan["timing_origin"].astype(str)) != {COMPLETE_PATH_TIMING_ORIGIN}:
        raise ValueError("nested Utility timing must start at the saved decision state")
    if set(replay_plan["required_timing_source"].astype(str)) != {
        COMPLETE_PATH_TIMING_SOURCE
    }:
        raise ValueError("replay plan must require measured complete policy paths")
    if set(replay_plan["timing_repetitions"].astype(int)) != {TIMING_REPETITIONS}:
        raise ValueError("replay plan must require exactly three real repetitions")
    if set(replay_plan["timing_order_protocol"].astype(str)) != {
        TIMING_ORDER_PROTOCOL
    }:
        raise ValueError("replay plan uses the wrong cyclic timing order")
    path_order = {path: index for index, path in enumerate(COMPLETE_PATHS)}
    observed_order = replay_plan["path"].astype(str).map(path_order)
    if observed_order.isna().any() or not np.array_equal(
        observed_order.to_numpy(dtype=int),
        replay_plan["canonical_path_order"].to_numpy(dtype=int),
    ):
        raise ValueError("replay plan paths do not match the canonical path order")
    group_key = ["learning_fold_role", *STATE_KEY_COLUMNS]
    paths = replay_plan.groupby(group_key, dropna=False)["path"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    if not bool((paths == frozenset(COMPLETE_PATHS)).all()):
        raise ValueError("every replay-plan state must contain exactly the five policy paths")
    selected_equals_prefix = (
        replay_plan["selected_algorithm"].astype(str)
        == replay_plan["prefix_algorithm"].astype(str)
    ).to_numpy(dtype=bool)
    if not np.array_equal(
        replay_plan["selected_equals_prefix"].to_numpy(dtype=bool),
        selected_equals_prefix,
    ):
        raise ValueError("replay-plan selected_equals_prefix is inconsistent")
    if not np.array_equal(
        replay_plan["handoff_required"].to_numpy(dtype=bool),
        ~selected_equals_prefix,
    ):
        raise ValueError("replay-plan handoff_required is inconsistent")
    expected_handoff = replay_plan["handoff_type"].astype(str).eq(
        "population_transfer_initialization"
    ).to_numpy(dtype=bool)
    if not np.array_equal(expected_handoff, ~selected_equals_prefix):
        raise ValueError("replay-plan handoff_type is inconsistent")
    gaps = replay_plan["expected_terminal_gap"].to_numpy(dtype=float)
    planned = replay_plan["expected_planned_FE"].to_numpy(dtype=int)
    effective = replay_plan["expected_effective_FE"].to_numpy(dtype=int)
    timed_out = replay_plan["expected_timed_out"].to_numpy(dtype=bool)
    completed = replay_plan["expected_path_completed"].to_numpy(dtype=bool)
    if not np.isfinite(gaps).all() or bool((gaps < 0.0).any()):
        raise ValueError("replay-plan terminal gaps must be finite and non-negative")
    if bool((planned <= 0).any()) or bool((effective < 0).any()) or bool(
        (effective > planned).any()
    ):
        raise ValueError("replay-plan planned/effective FE accounting is inconsistent")
    if bool((timed_out & completed).any()):
        raise ValueError("a timed-out Stage-A path cannot be marked completed")
    if not np.array_equal(effective[completed], planned[completed]):
        raise ValueError("completed Stage-A replay-plan paths must consume planned FE")


def _fold_state_view(
    states: pd.DataFrame,
    *,
    split: str,
    families: set[str],
    sbs_algorithm: str,
) -> pd.DataFrame:
    """Filter states by split, SBS algorithm, and CV group membership.

    The ``families`` parameter carries CV group IDs.  Filtering uses
    ``cv_group_id`` when present, falling back to ``family`` for legacy data.
    """
    group_column = "cv_group_id" if "cv_group_id" in states.columns else "family"
    view = states[
        states["split"].astype(str).eq(str(split))
        & states[group_column].astype(str).isin(families)
        & states["prefix_algorithm"].astype(str).eq(str(sbs_algorithm))
    ].copy()
    if view.empty:
        raise ValueError(
            f"no all-prefix states remain for split={split}, SBS={sbs_algorithm}, CV groups={sorted(families)}"
        )
    if set(view[group_column].astype(str)) != families:
        missing = sorted(families.difference(set(view[group_column].astype(str))))
        raise ValueError(f"fold-specific SBS view is missing CV groups: {missing}")
    view["default_algorithm"] = str(sbs_algorithm)
    view["no_query_algorithm"] = str(sbs_algorithm)
    if not view["prefix_algorithm"].astype(str).eq(str(sbs_algorithm)).all():
        raise RuntimeError("fold-specific view does not use the fold-derived SBS prefix")
    if not np.allclose(
        view["runtime_no_query_handoff"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=EPS,
    ):
        raise ValueError("fold-specific no-query path must be native continuation")
    if set(view["no_query_transition_mode"].astype(str)) != {"native_optimizer_state"}:
        raise ValueError("fold-specific no-query path must preserve native optimizer state")
    return view.sort_values(list(STATE_KEY_COLUMNS)).reset_index(drop=True)


def _check_all_prefix_coverage(
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
    label: str,
) -> None:
    state_without_prefix = [column for column in STATE_KEY_COLUMNS if column != "prefix_algorithm"]
    observed = states.groupby(state_without_prefix, dropna=False)["prefix_algorithm"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    if observed.empty or not bool((observed == frozenset(portfolio)).all()):
        raise ValueError(f"{label} outcomes must cover every portfolio prefix at every state")


def _check_outcome_state_alignment(query_states: pd.DataFrame, behavior_states: pd.DataFrame) -> None:
    key = list(STATE_KEY_COLUMNS)
    query_keys = query_states[key].drop_duplicates().sort_values(key).reset_index(drop=True)
    behavior_keys = behavior_states[key].drop_duplicates().sort_values(key).reset_index(drop=True)
    if not query_keys.equals(behavior_keys):
        raise ValueError("query-adjusted and behavior-only outcomes have different state keys")


def _check_performance_coverage(
    performance: pd.DataFrame,
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
) -> None:
    required = {
        "function_id",
        "family",
        "problem_id",
        "algorithm",
        "seed",
        "FE",
        "FE_total",
        "best_fitness",
    }
    missing = sorted(required.difference(performance.columns))
    if missing:
        raise ValueError(f"SBS performance table is missing columns: {missing}")
    perf_group_column = "cv_group_id" if "cv_group_id" in performance.columns else "family"
    states_group_column = "cv_group_id" if "cv_group_id" in states.columns else "family"
    train_families = set(states.loc[states["split"].astype(str) == TRAIN_SPLIT, states_group_column].astype(str))
    if set(performance[perf_group_column].astype(str)) != train_families:
        raise ValueError("SBS performance CV groups must equal BBOB-train outcome CV groups")
    train_functions = set(
        states.loc[
            states["split"].astype(str) == TRAIN_SPLIT,
            "function_id",
        ].astype(str)
    )
    if set(performance["function_id"].astype(str)) != train_functions:
        raise ValueError("SBS performance function IDs must equal BBOB-train outcome functions")
    if set(performance["algorithm"].astype(str)) != set(portfolio):
        raise ValueError("SBS performance table uses a different algorithm portfolio")


def _check_selected_prefix_alignment(
    query_states: pd.DataFrame,
    behavior_states: pd.DataFrame,
    role: str,
) -> None:
    key = list(STATE_KEY_COLUMNS)
    query_keys = query_states[key].sort_values(key).reset_index(drop=True)
    behavior_keys = behavior_states[key].sort_values(key).reset_index(drop=True)
    if not query_keys.equals(behavior_keys):
        raise ValueError(f"{role} query and behavior-only fold views are not state aligned")


def _selector_performance_summary(
    rows: pd.DataFrame,
    *,
    selector_input_mode: str,
    evaluation_role: str,
    fold_role: str,
    sbs_algorithm: str,
) -> pd.DataFrame:
    selected_loss = rows["selected_action_loss"].to_numpy(dtype=float)
    best_loss = rows["best_observed_loss"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "fold_role": fold_role,
                "evaluation_role": evaluation_role,
                "selector_input_mode": selector_input_mode,
                "action_budget_mode": str(rows["action_budget_mode"].iloc[0]),
                "sbs_algorithm": sbs_algorithm,
                "rows": int(len(rows)),
                "families": ",".join(sorted(set(rows["family"].astype(str)))),
                "mean_observed_action_loss": float(np.mean(selected_loss)),
                "mean_selector_regret_raw": float(np.mean(selected_loss - best_loss)),
                "selected_matches_best_observed_rate": float(
                    np.mean(rows["selected_algorithm"].astype(str) == rows["best_observed_algorithm"].astype(str))
                ),
                "validation_rows_used_for_fit": 0,
            }
        ]
    )


def _check_label_family_role(
    labels: pd.DataFrame,
    *,
    split: str,
    families: set[str],
    role: str,
) -> None:
    group_column = "cv_group_id" if "cv_group_id" in labels.columns else "family"
    if set(labels["split"].astype(str)) != {split}:
        raise RuntimeError(f"{role} Utility label view uses the wrong split")
    if set(labels[group_column].astype(str)) != families:
        raise RuntimeError(f"{role} Utility label view uses the wrong CV groups")


def _read_complete_path_timings(paths: list[Path]) -> pd.DataFrame:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("complete_path_timings.parquet")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(f"missing complete-path timing input: {path}")
    if not files:
        raise ValueError("nested learning requires complete-path timing inputs")
    frame = pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)
    required = {
        *STATE_KEY_COLUMNS,
        "learning_fold_role",
        "path",
        "repetition_index",
        "order_position",
        "runtime_seconds",
        "timing_repetitions",
        "timing_order_protocol",
        "timing_source",
        "timing_origin",
        "timing_environment_id",
        "thread_count",
        "selected_algorithm",
        "terminal_gap",
        "first_hit_FE",
        "success",
        "planned_FE",
        "effective_FE",
        "timed_out",
        "path_completed",
        "timing_replay_status",
        "timing_replay_timeout_seconds",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"complete-path timing input is missing columns: {missing}")
    duplicate_key = [
        *STATE_KEY_COLUMNS,
        "learning_fold_role",
        "path",
        "repetition_index",
    ]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("complete-path timing inputs contain duplicate fold/state/path repetitions")
    if set(frame["timing_source"].astype(str)) != {COMPLETE_PATH_TIMING_SOURCE}:
        raise ValueError(
            "complete-path timing must come from measured complete policy paths; "
            "component-runtime sums are not accepted"
        )
    if set(frame["timing_origin"].astype(str)) != {COMPLETE_PATH_TIMING_ORIGIN}:
        raise ValueError(
            "nested Utility complete-path timing must run from the saved decision state "
            "to the terminal budget; the shared prefix is a sunk cost"
        )
    if bool((frame["thread_count"].astype(int) <= 0).any()):
        raise ValueError("complete-path timing thread_count must be positive")
    if frame["timing_environment_id"].isna().any() or bool(
        frame["timing_environment_id"].astype(str).str.len().eq(0).any()
    ):
        raise ValueError("complete-path timing requires timing_environment_id")
    planned = frame["planned_FE"].to_numpy(dtype=int)
    effective = frame["effective_FE"].to_numpy(dtype=int)
    if bool((planned <= 0).any()) or bool((effective < 0).any()) or bool((effective > planned).any()):
        raise ValueError("complete-path planned/effective FE accounting is inconsistent")
    completed = frame["path_completed"].to_numpy(dtype=bool)
    timed_out = frame["timed_out"].to_numpy(dtype=bool)
    statuses = frame["timing_replay_status"].astype(str)
    if not set(statuses).issubset(TIMING_REPLAY_STATUSES):
        raise ValueError(
            "timing_replay_status must be completed, timed_out, or failed"
        )
    if not np.array_equal(
        completed,
        statuses.eq("completed").to_numpy(dtype=bool),
    ) or not np.array_equal(
        timed_out,
        statuses.eq("timed_out").to_numpy(dtype=bool),
    ):
        raise ValueError("complete-path replay status and timeout/completion flags disagree")
    if bool((timed_out & completed).any()):
        raise ValueError("timed-out complete policy paths cannot be marked completed")
    replay_timeout = frame["timing_replay_timeout_seconds"].to_numpy(dtype=float)
    if not np.isfinite(replay_timeout).all() or bool((replay_timeout <= 0.0).any()):
        raise ValueError("timing replay timeout must be finite and positive")
    if not np.isfinite(frame["terminal_gap"].to_numpy(dtype=float)).all() or bool(
        (frame["terminal_gap"].to_numpy(dtype=float) < 0.0).any()
    ):
        raise ValueError("complete-path terminal gaps must be finite and non-negative")
    return frame


def _fold_timing_view(
    timings: pd.DataFrame,
    *,
    learning_fold_role: str,
    split: str,
    families: set[str],
) -> pd.DataFrame:
    group_column = "cv_group_id" if "cv_group_id" in timings.columns else "family"
    view = timings[
        timings["learning_fold_role"].astype(str).eq(str(learning_fold_role))
        & timings["split"].astype(str).eq(str(split))
        & timings[group_column].astype(str).isin(families)
    ].copy()
    if view.empty:
        raise ValueError(
            "missing fold-specific complete-path timings for "
            f"role={learning_fold_role}, split={split}, CV groups={sorted(families)}"
        )
    if set(view[group_column].astype(str)) != families:
        raise ValueError("fold-specific complete-path timings do not cover every requested CV group")
    return view.drop(columns="learning_fold_role").reset_index(drop=True)


def _attach_decision_features(labels: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    key = list(STATE_KEY_COLUMNS)
    columns = [*key, *SELECTOR_BEHAVIOR_FEATURE_COLUMNS]
    missing = sorted(set(columns).difference(states.columns))
    if missing:
        raise ValueError(f"fold states are missing Decision behavior features: {missing}")
    features = states[columns].copy()
    if features.duplicated(key).any():
        raise ValueError("fold Decision behavior features contain duplicate state keys")
    joined = labels.merge(features, on=key, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("fold Utility labels and Decision behavior features are not aligned")
    return joined.drop(columns="_merge")


def _pre_run_fold_view(
    states: pd.DataFrame,
    *,
    split: str,
    families: set[str],
    sbs_algorithm: str,
) -> pd.DataFrame:
    group_column = "cv_group_id" if "cv_group_id" in states.columns else "family"
    view = states[
        states["split"].astype(str).eq(str(split))
        & states[group_column].astype(str).isin(families)
    ].copy()
    if view.empty or set(view[group_column].astype(str)) != families:
        raise ValueError(
            f"pre-run outcomes do not cover split={split}, CV groups={sorted(families)}"
        )
    view["default_algorithm"] = str(sbs_algorithm)
    view["no_query_algorithm"] = str(sbs_algorithm)
    return view.sort_values(list(PRE_RUN_STATE_KEY_COLUMNS)).reset_index(drop=True)
