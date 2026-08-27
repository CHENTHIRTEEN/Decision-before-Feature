from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_FEATURE_GROUPS
from decision.model_protocol import PREDEFINED_THRESHOLD_MODE
from decision.query_contract import decision_query_root, validate_query_frame, validate_query_payload
from decision.sampling_opportunities import (
    STATE_KEY_COLUMNS,
    assert_aligned_decision_opportunities,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


FEATURE_GROUP_ORDER = ("T0", "B1", "B2", "B2+Motion", "B2+Maturity", "B3")
EXPECTED_FEATURE_COUNTS = {
    "T0": 1,
    "B1": 19,
    "B2": 25,
    "B2+Motion": 28,
    "B2+Maturity": 28,
    "B3": 31,
}
TOP_K_FRACTION = 0.10
PRIMARY_OPPORTUNITY_SCOPE = "milestone_only"
SELECTION_OPPORTUNITY_SCOPE = "all_accepted"
RUN_KEY_COLUMNS = STATE_KEY_COLUMNS[:-1]


def compare_feature_group_training(
    *,
    query_id: str,
    input_root: Path,
    b3_selection_summary_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    group_payloads = [_read_group_outputs(input_root, group, query_id) for group in FEATURE_GROUP_ORDER]
    selection_summary = _read_b3_selection_summary(b3_selection_summary_path, query_id)
    selected_model_name = str(selection_summary["selected_model_name"])
    _check_group_comparability(group_payloads, selected_model_name=selected_model_name)

    regression = pd.concat([payload["regression"] for payload in group_payloads], ignore_index=True)
    score = pd.concat([payload["score"] for payload in group_payloads], ignore_index=True)
    decision = pd.concat([payload["decision"] for payload in group_payloads], ignore_index=True)
    ranking = pd.concat([payload["ranking"] for payload in group_payloads], ignore_index=True)
    formal_regression = regression[regression["model_name"].astype(str) == selected_model_name].copy()
    formal_score = score[score["model_name"].astype(str) == selected_model_name].copy()
    formal_decision = decision[decision["model_name"].astype(str) == selected_model_name].copy()
    formal_ranking = ranking[ranking["model_name"].astype(str) == selected_model_name].copy()
    if any(frame.empty for frame in (formal_score, formal_decision, formal_ranking)):
        raise ValueError("every formal feature group must contain the B3-selected model")
    for name, frame in (
        ("score", formal_score),
        ("decision", formal_decision),
        ("ranking", formal_ranking),
    ):
        if set(frame["feature_group"].astype(str)) != set(FEATURE_GROUP_ORDER):
            raise ValueError(
                f"{name} outputs do not cover the six canonical feature groups for the B3-selected model"
            )
    feature_groups = _feature_group_summary(group_payloads)
    rq2_primary_contrast = _rq2_primary_contrast_rows(
        group_payloads=group_payloads,
        selected_model_name=selected_model_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(feature_groups, output_dir / "feature_group_inputs")
    _write_frame(formal_regression, output_dir / "feature_group_regression_summary")
    _write_frame(formal_score, output_dir / "feature_group_score_summary")
    _write_frame(formal_decision, output_dir / "feature_group_decision_summary")
    _write_frame(formal_ranking, output_dir / "feature_group_ranking_summary")
    _write_frame(regression, output_dir / "feature_group_candidate_diagnostic_regression")
    _write_frame(score, output_dir / "feature_group_candidate_diagnostic_score")
    _write_frame(decision, output_dir / "feature_group_candidate_diagnostic_decision")
    _write_frame(ranking, output_dir / "feature_group_candidate_diagnostic_ranking")
    _write_frame(rq2_primary_contrast, output_dir / "rq2_milestone_b3_minus_t0_run_rows")

    summary = {
        "experiment": "phase1_refined_sampling_feature_group_ablation",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "research_question": (
            "Does milestone-only B3 improve first-trigger Decision Utility over milestone-only "
            "T0 when both use identical decision opportunities and the model family selected by "
            "B3/all_accepted on BBOB-train?"
        ),
        "input_root": str(input_root),
        "b3_selection_summary_path": str(b3_selection_summary_path),
        "feature_groups": FEATURE_GROUP_ORDER,
        "formal_comparison_model_name": selected_model_name,
        "formal_comparison_model_source": (
            "B3/all_accepted nested landscape-family OOF selection on BBOB-train"
        ),
        "primary_rq2_contrast": {
            "name": "milestone_only_B3_minus_milestone_only_T0",
            "opportunity_scope": PRIMARY_OPPORTUNITY_SCOPE,
            "decision_opportunities_identical": True,
            "comparison_model_name": selected_model_name,
            "model_source_opportunity_scope": SELECTION_OPPORTUNITY_SCOPE,
            "confirmatory_evaluation_split": "bbob_validation",
            "train_oof_role": "development_diagnostic",
            "run_rows_output": str(
                output_dir / "rq2_milestone_b3_minus_t0_run_rows.parquet"
            ),
        },
        "time_only_baseline": {
            "mathematical_input": ["FE_ratio"],
            "implementation_input": ["bf_fe_ratio"],
            "equality_contract": "bf_fe_ratio equals FE_ratio row by row",
            "research_question": "whether Decision performance is explained by optimization stage alone",
        },
        "top_k_fraction": TOP_K_FRACTION,
        "outputs": {
            "feature_group_inputs": str(output_dir / "feature_group_inputs.parquet"),
            "regression": str(output_dir / "feature_group_regression_summary.parquet"),
            "score": str(output_dir / "feature_group_score_summary.parquet"),
            "decision": str(output_dir / "feature_group_decision_summary.parquet"),
            "ranking": str(output_dir / "feature_group_ranking_summary.parquet"),
            "candidate_diagnostic_regression": str(output_dir / "feature_group_candidate_diagnostic_regression.parquet"),
            "candidate_diagnostic_score": str(output_dir / "feature_group_candidate_diagnostic_score.parquet"),
            "candidate_diagnostic_decision": str(output_dir / "feature_group_candidate_diagnostic_decision.parquet"),
            "candidate_diagnostic_ranking": str(output_dir / "feature_group_candidate_diagnostic_ranking.parquet"),
            "rq2_primary_contrast_run_rows": str(
                output_dir / "rq2_milestone_b3_minus_t0_run_rows.parquet"
            ),
            "report": str(output_dir / "feature_group_ablation_report.md"),
            "summary": str(output_dir / "feature_group_ablation_summary.json"),
        },
        "data_leakage_check": {
            "same_fold_specific_upstream_inputs_used_for_all_groups": True,
            "same_model_candidates_and_random_seed_used_for_all_groups": True,
            "same_threshold_modes_used_for_all_groups": True,
            "all_feature_groups_use_milestone_only_opportunities": True,
            "all_feature_groups_use_identical_decision_opportunities": True,
            "all_feature_groups_use_identical_utility_and_action_relation_labels": True,
            "feature_groups_drawn_from_behavior_features_only": True,
            "metadata_used_as_input": False,
            "algorithm_identifier_used_as_input": False,
            "query_features_used_as_input": False,
        },
    }
    summary_path = output_dir / "feature_group_ablation_summary.json"
    report_path = output_dir / "feature_group_ablation_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            feature_groups=feature_groups,
            regression=formal_regression,
            score=formal_score,
            decision=formal_decision,
            ranking=formal_ranking,
            selected_model_name=selected_model_name,
        ),
        encoding="utf-8",
    )

    print(f"wrote feature group input summary to {output_dir / 'feature_group_inputs.parquet'}")
    print(f"wrote feature group regression summary to {output_dir / 'feature_group_regression_summary.parquet'}")
    print(f"wrote feature group decision summary to {output_dir / 'feature_group_decision_summary.parquet'}")
    print(f"wrote feature group ranking summary to {output_dir / 'feature_group_ranking_summary.parquet'}")
    print(f"wrote feature group ablation report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "feature_group_inputs.csv",
        output_dir / "feature_group_inputs.parquet",
        output_dir / "feature_group_regression_summary.csv",
        output_dir / "feature_group_regression_summary.parquet",
        output_dir / "feature_group_score_summary.csv",
        output_dir / "feature_group_score_summary.parquet",
        output_dir / "feature_group_decision_summary.csv",
        output_dir / "feature_group_decision_summary.parquet",
        output_dir / "feature_group_ranking_summary.csv",
        output_dir / "feature_group_ranking_summary.parquet",
        output_dir / "feature_group_candidate_diagnostic_regression.csv",
        output_dir / "feature_group_candidate_diagnostic_regression.parquet",
        output_dir / "feature_group_candidate_diagnostic_score.csv",
        output_dir / "feature_group_candidate_diagnostic_score.parquet",
        output_dir / "feature_group_candidate_diagnostic_decision.csv",
        output_dir / "feature_group_candidate_diagnostic_decision.parquet",
        output_dir / "feature_group_candidate_diagnostic_ranking.csv",
        output_dir / "feature_group_candidate_diagnostic_ranking.parquet",
        output_dir / "rq2_milestone_b3_minus_t0_run_rows.csv",
        output_dir / "rq2_milestone_b3_minus_t0_run_rows.parquet",
        output_dir / "feature_group_ablation_report.md",
        output_dir / "feature_group_ablation_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"feature group comparison outputs already exist; pass --overwrite: {existing[0]}")


def _read_b3_selection_summary(path: Path, query_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    validate_query_payload(summary, query_id=query_id, artifact="B3 model-selection summary")
    expected = {
        "feature_group": "B3",
        "opportunity_scope": SELECTION_OPPORTUNITY_SCOPE,
    }
    mismatch = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatch:
        raise ValueError(f"B3 model-selection summary is inconsistent: {mismatch}")
    selected_model_name = str(summary.get("selected_model_name", ""))
    if not selected_model_name:
        raise ValueError("B3/all_accepted summary must identify the nested-OOF selected model")
    if selected_model_name not in tuple(summary.get("models_trained", [])):
        raise ValueError("B3/all_accepted selected model is absent from models_trained")
    return summary


def _read_group_outputs(input_root: Path, group: str, query_id: str) -> dict[str, Any]:
    group_dir = input_root / group / PRIMARY_OPPORTUNITY_SCOPE
    summary_path = group_dir / "full_decision_model_training_summary.json"
    regression_path = group_dir / "validation_regression_summary.parquet"
    score_path = group_dir / "validation_score_summary.parquet"
    decision_path = group_dir / "validation_decision_summary.parquet"
    ranking_path = group_dir / "validation_ranking_summary.parquet"
    train_oof_predictions_path = group_dir / "train_oof_predictions.parquet"
    validation_predictions_path = group_dir / "validation_predictions.parquet"
    for path in (
        summary_path,
        regression_path,
        score_path,
        decision_path,
        ranking_path,
        train_oof_predictions_path,
        validation_predictions_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_query_payload(summary, query_id=query_id, artifact=f"{group} training summary")
    if summary.get("feature_group") != group:
        raise ValueError(f"training summary feature_group mismatch for {group}: {summary.get('feature_group')}")
    if summary.get("opportunity_scope") != PRIMARY_OPPORTUNITY_SCOPE:
        raise ValueError(
            f"formal feature-group comparison requires {group}/{PRIMARY_OPPORTUNITY_SCOPE}, "
            f"observed opportunity_scope={summary.get('opportunity_scope')!r}"
        )
    feature_columns = list(summary.get("feature_columns", []))
    expected_columns = list(BEHAVIOR_FEATURE_GROUPS[group])
    if feature_columns != expected_columns:
        raise ValueError(f"training summary feature columns mismatch for {group}")

    regression = pq.read_table(regression_path).to_pandas()
    regression = regression[regression["layer"] == "all_validation"].copy()
    score = pq.read_table(score_path).to_pandas()
    score = score[score["layer"] == "all_validation"].copy()
    decision = pq.read_table(decision_path).to_pandas()
    decision = decision[decision["layer"] == "all_validation"].copy()
    ranking = pq.read_table(ranking_path).to_pandas()
    ranking = ranking[(ranking["layer"] == "all_validation") & np.isclose(ranking["top_k_fraction"], TOP_K_FRACTION)].copy()
    train_oof_predictions = pq.read_table(train_oof_predictions_path).to_pandas()
    validation_predictions = pq.read_table(validation_predictions_path).to_pandas()
    validate_query_frame(
        train_oof_predictions,
        query_id=query_id,
        artifact=f"{group} milestone-only train OOF predictions",
    )
    validate_query_frame(
        validation_predictions,
        query_id=query_id,
        artifact=f"{group} milestone-only validation predictions",
    )

    for frame in (regression, score, decision, ranking):
        frame.insert(0, "feature_group", group)
        frame.insert(1, "feature_count", len(feature_columns))
    return {
        "group": group,
        "summary": summary,
        "regression": regression,
        "score": score,
        "decision": decision,
        "ranking": ranking,
        "train_oof_predictions": train_oof_predictions,
        "validation_predictions": validation_predictions,
    }


def _check_group_comparability(
    group_payloads: list[dict[str, Any]],
    *,
    selected_model_name: str,
) -> None:
    if not group_payloads:
        raise ValueError("feature-group comparison requires at least one group")
    fields = (
        "training_input_mode",
        "query_protocol",
        "sample_design_id",
        "target_column",
        "auxiliary_label_column",
        "train_split",
        "validation_split",
        "rows",
        "models_trained",
        "threshold_modes",
        "random_seed",
        "opportunity_scope",
    )
    actual_groups = tuple(str(payload["group"]) for payload in group_payloads)
    if actual_groups != FEATURE_GROUP_ORDER:
        raise ValueError(f"formal feature-group order must be {FEATURE_GROUP_ORDER}")
    column_sets: dict[str, set[str]] = {}
    for payload in group_payloads:
        group = str(payload["group"])
        columns = list(payload["summary"].get("feature_columns", []))
        if len(columns) != EXPECTED_FEATURE_COUNTS[group] or len(set(columns)) != len(columns):
            raise ValueError(
                f"formal feature group {group} must contain "
                f"{EXPECTED_FEATURE_COUNTS[group]} unique columns"
            )
        column_sets[group] = set(columns)
    if len({tuple(sorted(values)) for values in column_sets.values()}) != len(column_sets):
        raise ValueError("formal feature groups must have distinct input-column sets")
    if not (
        column_sets["T0"] < column_sets["B1"] < column_sets["B2"]
        and column_sets["B2"] < column_sets["B2+Motion"] < column_sets["B3"]
        and column_sets["B2"] < column_sets["B2+Maturity"] < column_sets["B3"]
    ):
        raise ValueError(
            "formal feature groups must follow T0<B1<B2, with Motion and Maturity as distinct siblings nested in B3"
        )

    reference = group_payloads[0]["summary"]
    for field in fields:
        if field not in reference:
            raise ValueError(f"feature-group training summary is missing comparability field: {field}")
    for payload in group_payloads[1:]:
        summary = payload["summary"]
        for field in fields:
            if summary.get(field) != reference[field]:
                raise ValueError(
                    f"feature-group outputs are not comparable on {field}: "
                    f"T0={reference[field]!r}, {payload['group']}={summary.get(field)!r}"
                )
    for split_key in ("train_oof_predictions", "validation_predictions"):
        reference_predictions = _selected_prediction_rows(
            group_payloads[0][split_key],
            selected_model_name=selected_model_name,
            artifact=f"T0/{PRIMARY_OPPORTUNITY_SCOPE} {split_key}",
        )
        for payload in group_payloads[1:]:
            group = str(payload["group"])
            candidate_predictions = _selected_prediction_rows(
                payload[split_key],
                selected_model_name=selected_model_name,
                artifact=f"{group}/{PRIMARY_OPPORTUNITY_SCOPE} {split_key}",
            )
            assert_aligned_decision_opportunities(
                reference_predictions,
                candidate_predictions,
                reference_artifact=f"T0/{PRIMARY_OPPORTUNITY_SCOPE} {split_key}",
                candidate_artifact=f"{group}/{PRIMARY_OPPORTUNITY_SCOPE} {split_key}",
            )
            _assert_shared_prediction_labels(
                reference_predictions,
                candidate_predictions,
                target_column=str(reference["target_column"]),
                reference_artifact=f"T0/{PRIMARY_OPPORTUNITY_SCOPE} {split_key}",
                candidate_artifact=f"{group}/{PRIMARY_OPPORTUNITY_SCOPE} {split_key}",
            )


def _selected_prediction_rows(
    frame: pd.DataFrame,
    *,
    selected_model_name: str,
    artifact: str,
) -> pd.DataFrame:
    required = {
        "model_name",
        *STATE_KEY_COLUMNS,
        "is_budget_milestone",
        f"decision_run_query_{PREDEFINED_THRESHOLD_MODE}",
        f"decision_utility_{PREDEFINED_THRESHOLD_MODE}",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")
    selected = frame.loc[frame["model_name"].astype(str) == selected_model_name].copy()
    if selected.empty:
        raise ValueError(f"{artifact} has no rows for B3-selected model {selected_model_name}")
    if not bool(selected["is_budget_milestone"].to_numpy(dtype=bool).all()):
        raise ValueError(f"{artifact} contains non-milestone decision opportunities")
    return selected.sort_values(list(STATE_KEY_COLUMNS)).reset_index(drop=True)


def _assert_shared_prediction_labels(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    target_column: str,
    reference_artifact: str,
    candidate_artifact: str,
) -> None:
    string_columns = (
        "default_algorithm",
        "no_query_algorithm",
        "selected_algorithm",
        "selected_action",
        "handoff_type",
        "selector_target_transform",
    )
    boolean_columns = (
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
    )
    required = {target_column, *string_columns, *boolean_columns}
    for artifact, frame in (
        (reference_artifact, reference),
        (candidate_artifact, candidate),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{artifact} is missing shared-label columns: {missing}")
    if not np.allclose(
        reference[target_column].to_numpy(dtype=float),
        candidate[target_column].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{reference_artifact} and {candidate_artifact} use different {target_column} labels"
        )
    for column in string_columns:
        if not np.array_equal(
            reference[column].astype(str).to_numpy(),
            candidate[column].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"{reference_artifact} and {candidate_artifact} disagree on {column}"
            )
    for column in boolean_columns:
        if not np.array_equal(
            reference[column].to_numpy(dtype=bool),
            candidate[column].to_numpy(dtype=bool),
        ):
            raise ValueError(
                f"{reference_artifact} and {candidate_artifact} disagree on {column}"
            )


def _rq2_primary_contrast_rows(
    *,
    group_payloads: list[dict[str, Any]],
    selected_model_name: str,
) -> pd.DataFrame:
    payload_by_group = {str(payload["group"]): payload for payload in group_payloads}
    rows: list[pd.DataFrame] = []
    split_roles = (
        ("train_oof_predictions", "bbob_train_oof", "development_diagnostic"),
        ("validation_predictions", "bbob_validation", "predefined_evaluation"),
    )
    for prediction_key, split_name, evidence_role in split_roles:
        group_run_rows: dict[str, pd.DataFrame] = {}
        for group in ("T0", "B3"):
            selected = _selected_prediction_rows(
                payload_by_group[group][prediction_key],
                selected_model_name=selected_model_name,
                artifact=f"{group}/{PRIMARY_OPPORTUNITY_SCOPE} {prediction_key}",
            )
            group_run_rows[group] = _first_trigger_run_rows(selected, feature_group=group)
        paired = group_run_rows["T0"].merge(
            group_run_rows["B3"],
            on=list(RUN_KEY_COLUMNS),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not bool(paired["_merge"].eq("both").all()):
            raise ValueError(
                f"RQ2 {split_name} B3 and T0 do not cover identical trajectory keys"
            )
        paired = paired.drop(columns="_merge")
        if not np.array_equal(
            paired["T0_opportunity_count"].to_numpy(dtype=int),
            paired["B3_opportunity_count"].to_numpy(dtype=int),
        ):
            raise ValueError(
                f"RQ2 {split_name} B3 and T0 have different milestone counts within trajectories"
            )
        paired.insert(0, "comparison", "milestone_only_B3_minus_milestone_only_T0")
        paired.insert(1, "evidence_split", split_name)
        paired.insert(2, "evidence_role", evidence_role)
        paired.insert(3, "opportunity_scope", PRIMARY_OPPORTUNITY_SCOPE)
        paired.insert(4, "model_name", selected_model_name)
        paired["B3_minus_T0_policy_utility"] = (
            paired["B3_policy_utility"].to_numpy(dtype=float)
            - paired["T0_policy_utility"].to_numpy(dtype=float)
        )
        paired["identical_decision_opportunities"] = True
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def _first_trigger_run_rows(
    frame: pd.DataFrame,
    *,
    feature_group: str,
) -> pd.DataFrame:
    call_column = f"decision_run_query_{PREDEFINED_THRESHOLD_MODE}"
    utility_column = f"decision_utility_{PREDEFINED_THRESHOLD_MODE}"
    rows: list[dict[str, Any]] = []
    for run_key, run_frame in frame.groupby(list(RUN_KEY_COLUMNS), sort=True, dropna=False):
        if not isinstance(run_key, tuple):
            run_key = (run_key,)
        order_columns = ["FE"]
        if "decision_opportunity_index" in run_frame.columns:
            if run_frame["decision_opportunity_index"].isna().any():
                raise ValueError("decision_opportunity_index contains missing values")
            order_columns.append("decision_opportunity_index")
        elif run_frame["FE"].duplicated().any():
            raise ValueError(
                "multiple feature-group opportunities share one FE without decision_opportunity_index"
            )
        ordered = run_frame.sort_values(order_columns, kind="mergesort")
        calls = ordered[call_column].to_numpy(dtype=bool)
        if int(np.sum(calls)) > 1:
            raise ValueError(
                f"{feature_group} first-trigger predictions call more than once in one trajectory"
            )
        utility = ordered[utility_column].to_numpy(dtype=float)
        if not np.isfinite(utility).all():
            raise ValueError(f"{feature_group} first-trigger Utility contains non-finite values")
        if bool(np.any(np.abs(utility[~calls]) > 1e-12)):
            raise ValueError(
                f"{feature_group} first-trigger Utility must be zero outside its trigger row"
            )
        called = bool(np.any(calls))
        row = dict(zip(RUN_KEY_COLUMNS, run_key, strict=True))
        row.update(
            {
                f"{feature_group}_opportunity_count": int(len(ordered)),
                f"{feature_group}_called": called,
                f"{feature_group}_trigger_FE": (
                    int(ordered.loc[calls, "FE"].iloc[0]) if called else None
                ),
                f"{feature_group}_policy_utility": float(np.sum(utility)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_group_summary(group_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in group_payloads:
        group = str(payload["group"])
        feature_columns = list(payload["summary"]["feature_columns"])
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(feature_columns),
                "feature_columns": ",".join(feature_columns),
                "contains_maturity_features": any("maturity" in column or "explore_exploit" in column for column in feature_columns),
                "contains_diagnostic_features": any(
                    column
                    in {
                        "bf_fitness_diversity",
                        "bf_population_overlap_w05",
                        "bf_best_distance_fitness_corr",
                    }
                    for column in feature_columns
                ),
                "is_time_only_baseline": group == "T0",
            }
        )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _markdown_report(
    *,
    feature_groups: pd.DataFrame,
    regression: pd.DataFrame,
    score: pd.DataFrame,
    decision: pd.DataFrame,
    ranking: pd.DataFrame,
    selected_model_name: str,
) -> str:
    oof_threshold = decision[decision["threshold_mode"] == PREDEFINED_THRESHOLD_MODE].copy()
    return "\n".join(
        [
            "# Feature group ablation report",
            "",
            "## Feature groups",
            "",
            _markdown_table(
                feature_groups[
                    [
                        "feature_group",
                        "feature_count",
                        "contains_maturity_features",
                        "contains_diagnostic_features",
                        "is_time_only_baseline",
                    ]
                ]
            ),
            "",
            f"Formal cross-group comparison model: `{selected_model_name}` (selected by B3/all-accepted nested landscape-family OOF on BBOB-train).",
            "",
            "## Primary RQ2 contrast",
            "",
            "- The sole primary RQ2 contrast is milestone-only B3 minus milestone-only T0.",
            "- Both policies are fitted, thresholded, and evaluated on the same twelve milestone opportunities; exact state-key, sampling-metadata, Utility-label, and action-relation alignment is checked for train OOF and BBOB-validation predictions.",
            "- The run-level paired rows are written to `rq2_milestone_b3_minus_t0_run_rows.parquet`; BBOB-train OOF rows are development diagnostics and predefined BBOB-validation rows provide the prespecified evaluation.",
            "- Dynamic all-accepted B3 versus milestone-only T0 is not part of this RQ2 output and cannot identify the incremental contribution of Behavior features.",
            "",
            "## All-validation auxiliary score metrics for the B3-selected model",
            "",
            _markdown_table(
                score[
                    [
                        "feature_group",
                        "feature_count",
                        "model_name",
                        "objective",
                        "auroc",
                        "average_precision",
                        "spearman",
                        "rmse",
                        "rmse_applicable",
                    ]
                ].sort_values(["feature_group", "model_name"])
            ),
            "",
            "## All-validation continuous Utility regression (Ridge only)",
            "",
            _markdown_table(regression[["feature_group", "feature_count", "model_name", "rmse", "r2", "spearman"]].sort_values(["feature_group", "rmse"])),
            "",
            "## Predefined OOF-threshold decision",
            "",
            _markdown_table(
                oof_threshold[
                    [
                        "feature_group",
                        "feature_count",
                        "model_name",
                        "decision_query_call_rate",
                        "decision_mean_utility",
                        "positive_run_capture_rate",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                    ]
                ].sort_values(["feature_group", "decision_mean_utility"], ascending=[True, False])
            ),
            "",
            f"## Top {TOP_K_FRACTION:.0%} ranking",
            "",
            _markdown_table(
                ranking[
                    [
                        "feature_group",
                        "feature_count",
                        "model_name",
                        "top_k_u_gt_zero_rate",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "top_k_mean_observed_utility",
                    ]
                ].sort_values(["feature_group", "utility_capture_rate"], ascending=[True, False])
            ),
            "",
            "## Protocol",
            "",
            "- All groups are loaded only from `{feature_group}/milestone_only`, use the same mandatory-milestone rows and target column, and exclude event-only rows.",
            "- All groups use the same three fixed model candidates, random seed, nested landscape-family OOF selection, and fixed train-OOF threshold modes.",
            "- Formal feature-group conclusions compare the single model family selected by B3/all-accepted; per-candidate rows are retained only as model diagnostics.",
            "- BBOB-validation metrics are descriptive predefined evaluations only; this report does not rank or select a feature group from validation.",
            "- Feature groups are selected from `BEHAVIOR_FEATURE_GROUPS` only.",
            "- `T0` implements mathematical input `X={FE_ratio}` through `bf_fe_ratio`, which is checked row by row against trajectory `FE_ratio` during behavior validation and Decision materialization.",
            "- The formal milestone-only ablation is T0/B1/B2/B2+Motion/B2+Maturity/B3 with 1/19/25/28/28/31 inputs; the two 28-field groups are prespecified siblings.",
            "- Compare `T0` with behavior groups to test whether Decision performance is explained by optimization stage alone; a non-zero T0 result is not evidence that search behavior adds information.",
            "- Metadata, function identifiers, algorithm identifiers, optimizer internals, and Query features are not used as Decision Model input.",
        ]
    ) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    headers = list(frame.columns)
    rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the six feature groups on identical milestone-only opportunities and "
            "materialize the primary RQ2 B3-minus-T0 paired run rows."
        )
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument(
        "--b3-selection-summary",
        type=Path,
        default=None,
        help="B3/all-accepted training summary that freezes the model family used by every milestone group.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    compare_feature_group_training(
        query_id=args.query_id,
        input_root=args.input_root or query_root / "feature_group_ablation",
        b3_selection_summary_path=args.b3_selection_summary
        or query_root
        / "feature_group_ablation/B3/all_accepted/full_decision_model_training_summary.json",
        output_dir=args.output_dir or query_root / "feature_group_ablation_summary",
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
