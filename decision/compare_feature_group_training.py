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
from decision.model_protocol import FROZEN_THRESHOLD_MODE
from decision.query_contract import decision_query_root, validate_query_payload
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


FEATURE_GROUP_ORDER = ("T0", "B1", "B2", "B3")
EXPECTED_FEATURE_COUNTS = {"T0": 1, "B1": 19, "B2": 25, "B3": 31}
TOP_K_FRACTION = 0.10


def compare_feature_group_training(
    *,
    query_id: str,
    input_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    group_payloads = [_read_group_outputs(input_root, group, query_id) for group in FEATURE_GROUP_ORDER]
    _check_group_comparability(group_payloads)
    selected_model_name = str(group_payloads[-1]["summary"].get("selected_model_name", ""))
    if not selected_model_name:
        raise ValueError("B3 training summary must identify the nested-OOF selected model")

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
                f"{name} outputs do not cover T0/B1/B2/B3 for the B3-selected model"
            )
    feature_groups = _feature_group_summary(group_payloads)

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

    summary = {
        "experiment": "phase1_refined_sampling_feature_group_ablation",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "input_root": str(input_root),
        "feature_groups": FEATURE_GROUP_ORDER,
        "formal_comparison_model_name": selected_model_name,
        "formal_comparison_model_source": "B3 nested function-family OOF selection on BBOB-train",
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
            "report": str(output_dir / "feature_group_ablation_report.md"),
            "summary": str(output_dir / "feature_group_ablation_summary.json"),
        },
        "data_leakage_check": {
            "same_materialized_dataset_used_for_all_groups": True,
            "same_model_candidates_and_random_seed_used_for_all_groups": True,
            "same_threshold_modes_used_for_all_groups": True,
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
        output_dir / "feature_group_ablation_report.md",
        output_dir / "feature_group_ablation_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"feature group comparison outputs already exist; pass --overwrite: {existing[0]}")


def _read_group_outputs(input_root: Path, group: str, query_id: str) -> dict[str, Any]:
    group_dir = input_root / group
    summary_path = group_dir / "full_decision_model_training_summary.json"
    regression_path = group_dir / "validation_regression_summary.parquet"
    score_path = group_dir / "validation_score_summary.parquet"
    decision_path = group_dir / "validation_decision_summary.parquet"
    ranking_path = group_dir / "validation_ranking_summary.parquet"
    for path in (summary_path, regression_path, score_path, decision_path, ranking_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_query_payload(summary, query_id=query_id, artifact=f"{group} training summary")
    if summary.get("feature_group") != group:
        raise ValueError(f"training summary feature_group mismatch for {group}: {summary.get('feature_group')}")
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
    }


def _check_group_comparability(group_payloads: list[dict[str, Any]]) -> None:
    if not group_payloads:
        raise ValueError("feature-group comparison requires at least one group")
    fields = (
        "dataset",
        "schema",
        "target_column",
        "auxiliary_label_column",
        "train_split",
        "validation_split",
        "rows",
        "models_trained",
        "threshold_modes",
        "random_seed",
    )
    actual_groups = tuple(str(payload["group"]) for payload in group_payloads)
    if actual_groups != FEATURE_GROUP_ORDER:
        raise ValueError(f"formal feature-group order must be {FEATURE_GROUP_ORDER}")
    column_sets = []
    previous_columns: list[str] = []
    for payload in group_payloads:
        group = str(payload["group"])
        columns = list(payload["summary"].get("feature_columns", []))
        if len(columns) != EXPECTED_FEATURE_COUNTS[group] or len(set(columns)) != len(columns):
            raise ValueError(
                f"formal feature group {group} must contain "
                f"{EXPECTED_FEATURE_COUNTS[group]} unique columns"
            )
        if previous_columns and columns[: len(previous_columns)] != previous_columns:
            raise ValueError("formal feature groups must be strictly ordered nested prefixes")
        column_sets.append(frozenset(columns))
        previous_columns = columns
    if len(set(column_sets)) != len(column_sets):
        raise ValueError("formal feature groups must have distinct input-column sets")

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
    oof_threshold = decision[decision["threshold_mode"] == FROZEN_THRESHOLD_MODE].copy()
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
            f"Formal cross-group comparison model: `{selected_model_name}` (selected by B3 nested function-family OOF on BBOB-train).",
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
            "## Frozen OOF-threshold decision",
            "",
            _markdown_table(
                oof_threshold[
                    [
                        "feature_group",
                        "feature_count",
                        "model_name",
                        "decision_query_call_rate",
                        "decision_mean_utility",
                        "positive_row_capture_rate",
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
            "- All groups use the same materialized dataset and target column.",
            "- All groups use the same three fixed model candidates, random seed, nested family-OOF selection, and frozen train-OOF threshold modes.",
            "- Formal T0/B1/B2/B3 conclusions compare the single model selected by B3; per-candidate rows are retained only as model diagnostics.",
            "- BBOB-validation metrics are descriptive frozen evaluations only; this report does not rank or select a feature group from validation.",
            "- Feature groups are selected from `BEHAVIOR_FEATURE_GROUPS` only.",
            "- `T0` implements mathematical input `X={FE_ratio}` through `bf_fe_ratio`, which is checked row by row against trajectory `FE_ratio` during behavior validation and Decision materialization.",
            "- The formal ablation is exactly T0/B1/B2/B3 with 1/19/25/31 strictly nested, distinct inputs.",
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
    parser = argparse.ArgumentParser(description="Compare phase1 refined sampling feature group training outputs.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    compare_feature_group_training(
        query_id=args.query_id,
        input_root=args.input_root or query_root / "feature_group_ablation",
        output_dir=args.output_dir or query_root / "feature_group_ablation_summary",
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
