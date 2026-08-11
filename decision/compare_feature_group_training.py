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


FEATURE_GROUP_ORDER = ("time_only", "base", "primary", "primary_with_maturity", "all_candidates")
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

    regression = pd.concat([payload["regression"] for payload in group_payloads], ignore_index=True)
    score = pd.concat([payload["score"] for payload in group_payloads], ignore_index=True)
    decision = pd.concat([payload["decision"] for payload in group_payloads], ignore_index=True)
    ranking = pd.concat([payload["ranking"] for payload in group_payloads], ignore_index=True)
    best = _best_summary(score, decision, ranking)
    feature_groups = _feature_group_summary(group_payloads)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(feature_groups, output_dir / "feature_group_inputs")
    _write_frame(regression, output_dir / "feature_group_regression_summary")
    _write_frame(score, output_dir / "feature_group_score_summary")
    _write_frame(decision, output_dir / "feature_group_decision_summary")
    _write_frame(ranking, output_dir / "feature_group_ranking_summary")
    _write_frame(best, output_dir / "feature_group_best_summary")

    summary = {
        "experiment": "phase1_refined_sampling_feature_group_ablation",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "input_root": str(input_root),
        "feature_groups": FEATURE_GROUP_ORDER,
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
            "best": str(output_dir / "feature_group_best_summary.parquet"),
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
            regression=regression,
            score=score,
            decision=decision,
            ranking=ranking,
            best=best,
        ),
        encoding="utf-8",
    )

    print(f"wrote feature group input summary to {output_dir / 'feature_group_inputs.parquet'}")
    print(f"wrote feature group regression summary to {output_dir / 'feature_group_regression_summary.parquet'}")
    print(f"wrote feature group decision summary to {output_dir / 'feature_group_decision_summary.parquet'}")
    print(f"wrote feature group ranking summary to {output_dir / 'feature_group_ranking_summary.parquet'}")
    print(f"wrote feature group best summary to {output_dir / 'feature_group_best_summary.parquet'}")
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
        output_dir / "feature_group_best_summary.csv",
        output_dir / "feature_group_best_summary.parquet",
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
                    f"time_only={reference[field]!r}, {payload['group']}={summary.get(field)!r}"
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
                "contains_diagnostic_features": any(column in {"bf_population_overlap_w05", "bf_best_distance_fitness_corr"} for column in feature_columns),
                "is_time_only_baseline": group == "time_only",
            }
        )
    return pd.DataFrame(rows)


def _best_summary(score: pd.DataFrame, decision: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(_best_row(score[score["rmse_applicable"].astype(bool)], "lowest_rmse", "rmse", ascending=True))
    rows.append(_best_row(score, "highest_spearman", "spearman", ascending=False))
    rows.append(_best_row(score, "highest_auroc", "auroc", ascending=False))
    rows.append(_best_row(score, "highest_average_precision", "average_precision", ascending=False))
    oof_threshold = decision[decision["threshold_mode"] == FROZEN_THRESHOLD_MODE].copy()
    zero_threshold = decision[decision["threshold_mode"] == "zero"].copy()
    rows.append(_best_row(oof_threshold, "highest_oof_threshold_decision_mean_utility", "decision_mean_utility", ascending=False))
    rows.append(_best_row(zero_threshold, "highest_zero_threshold_decision_mean_utility", "decision_mean_utility", ascending=False))
    rows.append(_best_row(ranking, "highest_top10_utility_capture_rate", "utility_capture_rate", ascending=False))
    return pd.DataFrame(rows)


def _best_row(frame: pd.DataFrame, criterion: str, metric: str, *, ascending: bool) -> dict[str, Any]:
    finite = frame[np.isfinite(pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float, na_value=np.nan))].copy()
    if finite.empty:
        return {"criterion": criterion, "metric": metric, "status": "unavailable"}
    row = finite.sort_values(metric, ascending=ascending).iloc[0].to_dict()
    selected = {
        "criterion": criterion,
        "metric": metric,
        "status": "available",
        "feature_group": row.get("feature_group"),
        "feature_count": row.get("feature_count"),
        "model_name": row.get("model_name"),
        "model_family": row.get("model_family"),
        "threshold_mode": row.get("threshold_mode"),
        metric: row.get(metric),
    }
    for optional in (
        "decision_mean_utility",
        "decision_query_call_rate",
        "positive_row_capture_rate",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "top_k_u_gt_zero_rate",
    ):
        if optional in row and optional not in selected:
            selected[optional] = row.get(optional)
    return selected


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
    best: pd.DataFrame,
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
            "## Best rows",
            "",
            _markdown_table(best),
            "",
            "## All-validation auxiliary score metrics",
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
            "- Feature groups are selected from `BEHAVIOR_FEATURE_GROUPS` only.",
            "- `time_only` implements mathematical input `X={FE_ratio}` through `bf_fe_ratio`, which is checked row by row against trajectory `FE_ratio` during behavior validation and Decision materialization.",
            "- Compare `time_only` with behavior groups to test whether Decision performance is explained by optimization stage alone; a non-zero time-only result is not evidence that search behavior adds information.",
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
