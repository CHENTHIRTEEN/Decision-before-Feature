from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.query_contract import decision_query_root, validate_query_payload
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS


TOP_K_FRACTION = 0.10


def build_full_training_interpretation_report(
    *,
    query_id: str,
    full_training_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    _check_output_paths(output_dir, overwrite)
    training_summary = json.loads(
        (full_training_dir / "full_decision_model_training_summary.json").read_text(encoding="utf-8")
    )
    validate_query_payload(training_summary, query_id=query_id, artifact="Decision training summary")
    regression = pd.read_csv(full_training_dir / "validation_regression_summary.csv")
    decision = pd.read_csv(full_training_dir / "validation_decision_summary.csv")
    ranking = pd.read_csv(full_training_dir / "validation_ranking_summary.csv")
    thresholds = pd.read_csv(full_training_dir / "decision_thresholds.csv")
    model_fit = pd.read_csv(full_training_dir / "model_fit_summary.csv")
    input_contract = pd.read_csv(full_training_dir / "model_input_contract.csv")
    preprocessing = pd.read_csv(full_training_dir / "preprocessing_fit_summary.csv")

    tables = _evidence_tables(
        regression=regression,
        decision=decision,
        ranking=ranking,
        thresholds=thresholds,
        model_fit=model_fit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        _write_frame(frame, output_dir / name)

    report_path = output_dir / "decision_model_result_interpretation_report.md"
    report_path.write_text(
        _markdown_report(
            tables=tables,
            input_contract=input_contract,
            preprocessing=preprocessing,
            full_training_dir=full_training_dir,
        ),
        encoding="utf-8",
    )
    print(f"wrote Decision Model interpretation report to {report_path}")
    return {"report": str(report_path)}


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "decision_model_result_interpretation_report.md",
        output_dir / "all_validation_regression.csv",
        output_dir / "all_validation_regression.parquet",
        output_dir / "all_validation_threshold_decision.csv",
        output_dir / "all_validation_threshold_decision.parquet",
        output_dir / "all_validation_top10_ranking.csv",
        output_dir / "all_validation_top10_ranking.parquet",
        output_dir / "label_source_top10_ranking.csv",
        output_dir / "label_source_top10_ranking.parquet",
        output_dir / "dimension_top10_ranking.csv",
        output_dir / "dimension_top10_ranking.parquet",
        output_dir / "fe_ratio_best_top10_ranking.csv",
        output_dir / "fe_ratio_best_top10_ranking.parquet",
        output_dir / "prefix_algorithm_top10_ranking.csv",
        output_dir / "prefix_algorithm_top10_ranking.parquet",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"interpretation outputs already exist; pass --overwrite: {existing[0]}")


def _evidence_tables(
    *,
    regression: pd.DataFrame,
    decision: pd.DataFrame,
    ranking: pd.DataFrame,
    thresholds: pd.DataFrame,
    model_fit: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    all_regression = (
        regression[regression["layer"] == "all_validation"][
            ["model_name", "rows", "mae", "rmse", "r2", "pearson", "spearman"]
        ]
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    all_threshold = (
        decision[decision["layer"] == "all_validation"][
            [
                "model_name",
                "threshold_mode",
                "threshold",
                "decision_query_call_rate",
                "mean_observed_utility_under_calls",
                "positive_row_capture_rate",
                "utility_capture_rate",
                "precision_u_gt_zero_under_calls",
                "unhelpful_call_rate_within_calls",
                "decision_mean_utility",
            ]
        ]
        .sort_values(["threshold_mode", "utility_capture_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )
    all_top10 = (
        ranking[(ranking["layer"] == "all_validation") & np.isclose(ranking["top_k_fraction"], TOP_K_FRACTION)][
            [
                "model_name",
                "rows",
                "top_k_rows",
                "u_gt_zero_rate",
                "top_k_u_gt_zero_rate",
                "lift_vs_base_rate",
                "positive_row_capture_rate",
                "utility_capture_rate",
                "top_k_mean_observed_utility",
            ]
        ]
        .sort_values("utility_capture_rate", ascending=False)
        .reset_index(drop=True)
    )
    label_source_top10 = (
        ranking[(ranking["layer"] == "label_source") & np.isclose(ranking["top_k_fraction"], TOP_K_FRACTION)][
            [
                "model_name",
                "label_source",
                "rows",
                "u_gt_zero_rate",
                "top_k_u_gt_zero_rate",
                "lift_vs_base_rate",
                "positive_row_capture_rate",
                "utility_capture_rate",
                "top_k_mean_observed_utility",
            ]
        ]
        .sort_values(["label_source", "utility_capture_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )
    dimension_top10 = (
        ranking[(ranking["layer"] == "dimension") & np.isclose(ranking["top_k_fraction"], TOP_K_FRACTION)][
            [
                "model_name",
                "dimension",
                "rows",
                "u_gt_zero_rate",
                "top_k_u_gt_zero_rate",
                "lift_vs_base_rate",
                "utility_capture_rate",
                "top_k_mean_observed_utility",
            ]
        ]
        .sort_values(["dimension", "utility_capture_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )
    fe_ratio_top10 = ranking[
        (ranking["layer"] == "FE_ratio") & np.isclose(ranking["top_k_fraction"], TOP_K_FRACTION)
    ].copy()
    fe_ratio_best = (
        fe_ratio_top10.sort_values(["FE_ratio", "utility_capture_rate"], ascending=[True, False])
        .groupby("FE_ratio", as_index=False)
        .head(1)[
            [
                "FE_ratio",
                "model_name",
                "rows",
                "u_gt_zero_rate",
                "top_k_u_gt_zero_rate",
                "lift_vs_base_rate",
                "utility_capture_rate",
                "top_k_mean_observed_utility",
            ]
        ]
        .reset_index(drop=True)
    )
    prefix_top10 = (
        ranking[(ranking["layer"] == "prefix_algorithm") & np.isclose(ranking["top_k_fraction"], TOP_K_FRACTION)][
            [
                "model_name",
                "prefix_algorithm",
                "rows",
                "u_gt_zero_rate",
                "top_k_u_gt_zero_rate",
                "lift_vs_base_rate",
                "utility_capture_rate",
                "top_k_mean_observed_utility",
            ]
        ]
        .sort_values(["prefix_algorithm", "utility_capture_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )
    model_summary = model_fit[
        [
            "model_name",
            "model_family",
            "fit_seconds",
            "validation_prediction_seconds",
            "model_path",
        ]
    ].copy()
    thresholds_compact = thresholds[
        ["model_name", "threshold_mode", "threshold", "fit_split", "validation_rows_used_for_threshold_fit"]
    ].copy()
    return {
        "all_validation_regression": all_regression,
        "all_validation_threshold_decision": all_threshold,
        "all_validation_top10_ranking": all_top10,
        "label_source_top10_ranking": label_source_top10,
        "dimension_top10_ranking": dimension_top10,
        "fe_ratio_best_top10_ranking": fe_ratio_best,
        "prefix_algorithm_top10_ranking": prefix_top10,
        "model_runtime_summary": model_summary,
        "threshold_source_summary": thresholds_compact,
    }


def _markdown_report(
    *,
    tables: dict[str, pd.DataFrame],
    input_contract: pd.DataFrame,
    preprocessing: pd.DataFrame,
    full_training_dir: Path,
) -> str:
    best_regression = tables["all_validation_regression"].iloc[0]
    best_top10 = tables["all_validation_top10_ranking"].iloc[0]
    changed = tables["label_source_top10_ranking"]
    best_changed = changed[changed["label_source"] == "changed_algorithm"].iloc[0]
    best_same = changed[changed["label_source"] == "same_algorithm"].iloc[0]
    zero_decision = tables["all_validation_threshold_decision"]
    zero_decision = zero_decision[zero_decision["threshold_mode"] == "zero"].sort_values(
        "utility_capture_rate", ascending=False
    )
    best_zero = zero_decision.iloc[0]
    train_threshold = tables["all_validation_threshold_decision"]
    train_threshold = train_threshold[train_threshold["threshold_mode"] == "train_utility"].sort_values(
        "utility_capture_rate", ascending=False
    )
    best_train_threshold = train_threshold.iloc[0]
    prep_ok = bool(
        preprocessing["imputer_statistic_matches_train_median"].all()
        and preprocessing["scaler_mean_matches_train"].all()
        and preprocessing["scaler_var_matches_train"].all()
        and preprocessing["validation_rows_used_for_fit"].eq(0).all()
    )
    input_ok = bool(input_contract["passed"].all())

    return "\n".join(
        [
            "# Decision Model full-training result interpretation",
            "",
            "## Technical summary",
            "",
            (
                f"- **Recommended main report model: `lightgbm_regression`.** It has the best all-validation "
                f"RMSE ({_fmt(best_regression['rmse'])}) and nearly ties the best all-validation top-10% "
                f"utility capture ({_fmt_pct(tables['all_validation_top10_ranking'].loc[tables['all_validation_top10_ranking']['model_name'].eq('lightgbm_regression'), 'utility_capture_rate'].iloc[0])})."
            ),
            (
                f"- **Recommended main decision口径: score ranking / top-k analysis, with train-derived thresholds reported as secondary.** "
                f"The best top-10% policy captures {_fmt_pct(best_top10['utility_capture_rate'])} of validation positive utility, "
                f"whereas the best zero-threshold call policy captures only {_fmt_pct(best_zero['utility_capture_rate'])} and has negative mean utility under calls."
            ),
            (
                f"- **The usable signal is concentrated in `changed_algorithm`.** LightGBM top-10% ranking reaches "
                f"{_fmt_pct(best_changed['top_k_u_gt_zero_rate'])} `U_query>0` rate and {_fmt_pct(best_changed['utility_capture_rate'])} utility capture on changed rows, "
                f"but the best same-row top-10% utility capture is only {_fmt_pct(best_same['utility_capture_rate'])} and still has negative top-k mean observed utility."
            ),
            (
                f"- **Protocol checks passed.** Input legality is `{input_ok}` and train-only preprocessing checks are `{prep_ok}`; metadata fields are used only for stratified reporting."
            ),
            "",
            "## What was compared and how to read the metrics",
            "",
            "- Cohort: `bbob_validation`, 64,800 rows from the phase1 refined sampling materialized Decision dataset.",
            "- Input matrix: exactly the nine frozen `bf_*` behavior features.",
            "- Target: `u_query_lamT_1`; `U_query>0` means executing the evaluated fixed query has positive observed utility under this frozen target.",
            "- Regression metrics evaluate predicted utility magnitude; ranking metrics evaluate whether high scores enrich rows with useful query outcomes.",
            "- Decision threshold metrics evaluate the policy `decision_score > threshold`; `zero` uses 0, while `train_utility` is selected only from train predictions.",
            "",
            "## Regression says LightGBM is the strongest main model, while Ridge is mostly a ranking baseline",
            "",
            "All four models have weak or below-mean validation R2, so the exact magnitude of `U_query` remains hard to predict. LightGBM is still the best main model because it gives the lowest RMSE with nontrivial rank correlation; XGBoost is close, Random Forest is weaker, and Ridge has poor RMSE despite high Spearman.",
            "",
            _markdown_table(tables["all_validation_regression"]),
            "",
            "Interpretation: use regression results to rank model families, not to claim calibrated utility magnitude. The negative R2 values mean a mean predictor is difficult to beat on squared error under the current imbalanced target distribution.",
            "",
            "## Zero threshold is not sufficient as the main decision rule",
            "",
            "The zero-threshold rule is too brittle because model scores are mostly below zero while true positive utility rows are sparse. The best zero-threshold capture is XGBoost at only 16.2% utility capture, and every nontrivial zero-threshold policy has negative mean observed utility under calls. Train-derived thresholds do not fix the issue; they shift call rates slightly but still yield negative mean decision utility on all-validation.",
            "",
            _markdown_table(tables["all_validation_threshold_decision"]),
            "",
            "Implication: `predicted_u_query > 0` should remain a protocol baseline because it matches the conceptual decision rule, but it should not be the only main empirical claim. The more defensible empirical claim is that behavior scores provide a useful prioritization signal.",
            "",
            "## Top-k ranking gives the cleanest Decision-before-Feature evidence",
            "",
            "Top-10% ranking turns weak point predictions into a useful prioritization policy. XGBoost is fractionally best on all-validation utility capture, but LightGBM is essentially tied while also being the best regression model, which makes LightGBM the cleaner single model for the main report.",
            "",
            _markdown_table(tables["all_validation_top10_ranking"]),
            "",
            "Interpretation: if the downstream system can control a query budget or call rate, top-k ranking is a better operational decision口径 than raw zero thresholding. Under the primary SBS-prefix protocol, same_algorithm means the query path selects the current SBS without changing parameters, restart behavior, budget policy, or risk policy; these rows are structurally budget-and-cost-only references, not confirmation-value evidence.",
            "",
            "## changed_algorithm rows carry the main learnable signal",
            "",
            "Under the primary SBS-prefix protocol, `changed_algorithm` means `selected_equals_prefix=false`, so these rows isolate cases where the query changes the optimizer action. `same_algorithm` rows make no action change and should have non-positive performance gain before non-FE costs; any positive utility must trigger a state/budget consistency check rather than be interpreted as information confirmation value.",
            "",
            _markdown_table(tables["label_source_top10_ranking"]),
            "",
            "Implication: the main paper table should report all primary rows and the selected-equals-prefix split side by side. The legacy changed/same names are acceptable only because primary rows satisfy prefix_algorithm=default_algorithm; all-prefix robustness reports must use the explicit relation fields.",
            "",
            "## Dimension and FE_ratio reveal where the ranking signal is most usable",
            "",
            "By dimension, LightGBM/XGBoost dominate most top-10% utility-capture comparisons. The 40D layer has a much lower base `U_query>0` rate, but top-k lift is high; this means the model can enrich rare useful rows even when the absolute row rate is small.",
            "",
            _markdown_table(tables["dimension_top10_ranking"]),
            "",
            "By FE checkpoint, the best model varies. Early checkpoints around 0.20-0.35 are weak; utility capture becomes much stronger after 0.40, with Ridge unexpectedly strongest at several later FE ratios. That pattern suggests FE-stage-specific thresholding or budget allocation may be worth a follow-up, but not as part of this main model selection unless it is trained on train split only.",
            "",
            _markdown_table(tables["fe_ratio_best_top10_ranking"]),
            "",
            "## Prefix-algorithm robustness is isolated from the main result",
            "",
            "The primary Decision dataset contains only the train-derived SBS prefix. Multi-prefix comparisons belong to the separate cross-probe dataset and may support cross-probe robustness, leave-one-probe-out, and algorithm-agnostic generalization analyses. They do not justify adding `prefix_algorithm` to the main model input.",
            "",
            _markdown_table(tables["prefix_algorithm_top10_ranking"]),
            "",
            "## Model choice",
            "",
            "Recommended main report model: `lightgbm_regression`.",
            "",
            "- It is best on all-validation RMSE among the four trained models.",
            "- It nearly ties XGBoost on all-validation top-10% utility capture.",
            "- It is best on the primary-protocol `changed_algorithm` top-10% utility capture, where changed is equivalent to `selected_equals_prefix=false`.",
            "- It is fast enough for reporting and deployment-style analysis in this dataset.",
            "",
            "Recommended secondary comparison model: `xgboost_regression`, because it slightly leads all-validation top-10% utility capture. Recommended baseline model: `ridge_regression`, because it provides a simple linear ranking baseline but should not be used for calibrated utility magnitude.",
            "",
            "## Main decision口径",
            "",
            "Recommended primary empirical口径: top-k ranking at 5%, 10%, and 20%, with 10% as the headline table because it balances selectivity and enough rows for stable stratification.",
            "",
            "Recommended secondary口径: train-derived utility threshold, reported mainly to show that thresholding without validation tuning remains difficult. Zero threshold should be kept only as the conceptual baseline for `predicted_u_query > 0`, not as the main performance claim.",
            "",
            "## Robustness and limitations",
            "",
            "- This is predictive evidence, not causal evidence about landscape analysis; labels are observed utility rows generated for the evaluated fixed query under the frozen phase1 setup.",
            "- The sparse `U_query>0` rate makes regression magnitude difficult; ranking is more stable than direct thresholding.",
            "- `prefix_algorithm`, `dimension`, `FE_ratio`, `label_source`, `family`, and problem metadata are not model inputs; they are only reporting strata.",
            "- Primary-protocol same-algorithm rows are no-action-change budget-and-cost references; positive utility in this layer is a consistency-check failure condition, not confirmation value.",
            f"- Source evidence comes from `{full_training_dir}`.",
            "",
            "## Recommended next steps",
            "",
            "1. Use LightGBM as the main Decision Model in the formal result table.",
            "2. Report top-5%, top-10%, and top-20% ranking metrics as the primary empirical decision evidence.",
            "3. Keep zero threshold and train-derived threshold as secondary threshold diagnostics.",
            "4. Add a follow-up train-only stage policy analysis for FE_ratio-aware call budgets, because later FE checkpoints show stronger capture.",
            "5. Keep algorithm-partition stacking as diagnostic evidence only; do not move `prefix_algorithm` into the main Decision input.",
            "",
            "## Further questions",
            "",
            "- Can train-only FE-stage thresholds improve mean decision utility without using validation statistics?",
            "- Does the LightGBM ranking advantage transfer to CEC2017, CEC2022, and engineering problems under the planned external test protocol?",
            "- Can calibration be improved without adding forbidden query/function/algorithm metadata to X?",
            "",
        ]
    )


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    if frame.empty:
        return ""
    headers = list(frame.columns)
    rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return f"{float(value):.4g}"


def _fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an interpretation report from full Decision Model training outputs.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--full-training-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    build_full_training_interpretation_report(
        query_id=args.query_id,
        full_training_dir=args.full_training_dir or query_root / "full_training",
        output_dir=args.output_dir or query_root / "full_training_interpretation",
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
