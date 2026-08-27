"""Align complete Decision opportunities across BBOB-train OOF and CEC.

This is an independent diagnostic.  It keeps the deployed Decision model,
preprocessing, threshold, and online protocol unchanged.  It reconstructs the
complete CMA-ES behavior opportunity stream for CEC Native and Unit-cube
coordinates, joins the fixed model score, and compares both CEC streams with
the BBOB-train OOF opportunity table.

The cross-suite comparison preserves the unique within-run key
``function x dimension x seed x FE``.  Since different suites do not share
function identifiers or every exact FE value, aggregate comparisons use the
common ``FE_ratio`` and predefined sampling phases after that key check.
"""

from __future__ import annotations

import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from benchmarks import make_problem
from behavior.features import DECISION_BEHAVIOR_FEATURE_COLUMNS
from behavior.streaming import StreamingBehaviorState
from decision.model_protocol import decision_scores
from experiments.cli.cec2017_normalized_online_compare import _make_unit_cube_problem, _unit_cube_runtime
from experiments.phase1_batch_common import load_config
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/dataset_analysis/cec2017_opportunity_alignment"
CEC_CONFIG = REPO / "configs/cec2017_representative_online_compare.yaml"
DECISION_DIR = REPO / "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision"
NATIVE_REFERENCE = REPO / "results/dataset_analysis/cec2017_decision_diagnostics/cec_cmaes_decision_scores.parquet"
NATIVE_ONLINE = REPO / "outputs/cec2017_representative_online_compare/online_comparison_run_metrics.parquet"
UNIT_ONLINE = REPO / "outputs/cec2017_representative_online_compare_unit_cube/online_comparison_run_metrics.parquet"

CEC_FUNCTIONS = (1, 5, 9, 20, 24)
CEC_DIMENSIONS = (10, 20, 30, 50)
CEC_SEEDS = (1, 2, 3, 4, 5)
POPULATION_SIZE = 40
SAMPLING_PROTOCOL = "phase1_dynamic_budget_event_v1"
USER_THRESHOLD = 0.2997557291
FEATURES = tuple(
    column
    for column in DECISION_BEHAVIOR_FEATURE_COLUMNS
    if column not in {"bf_search_maturity", "bf_explore_exploit_ratio"}
)
KEY_COLUMNS = ("problem_id", "function_id", "dimension", "seed", "FE")
META_COLUMNS = (
    "source",
    "condition",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "seed",
    "FE",
    "FE_ratio",
    "FE_total",
    "sampling_phase",
    "opportunity_index",
    "decision_score",
)

PALETTE = {
    "bbob": "#2F5D8C",
    "native": "#657A3A",
    "unit": "#C96B27",
    "charcoal": "#2F3136",
    "grey": "#8A929B",
    "grid": "#D9DDE2",
    "paper": "#FBFBFA",
}
SOURCE_LABELS = {
    "bbob_train_oof": "BBOB-train OOF",
    "cec_native": "CEC Native",
    "cec_unit_cube": "CEC Unit-cube",
}


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PALETTE["paper"],
            "figure.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["grey"],
            "axes.labelcolor": PALETTE["charcoal"],
            "xtick.color": PALETTE["charcoal"],
            "ytick.color": PALETTE["charcoal"],
            "text.color": PALETTE["charcoal"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "savefig.dpi": 220,
        }
    )


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _dimension_reference(dimension: int) -> int:
    dimension = int(dimension)
    return dimension if dimension in (10, 20) else 40


def _dimension_group(source: str, dimension: int) -> str:
    if str(source) == "bbob_train_oof":
        return "BBOB 10/20D" if int(dimension) in (10, 20) else "BBOB 40D"
    return "CEC 10/20D" if int(dimension) in (10, 20) else "CEC 30/50D"


def _phase_bin(ratio: float) -> str:
    ratio = float(ratio)
    if ratio <= 0.30 + 1e-12:
        return "early"
    if ratio < 0.50:
        return "mid"
    return "late"


def _validate_feature_contract() -> None:
    if len(FEATURES) != 29 or not all(column.startswith("bf_") for column in FEATURES):
        raise ValueError(f"expected 29 bf_* Decision features, got {FEATURES}")


def _read_model() -> tuple[object, float, dict[str, Any]]:
    summary_path = DECISION_DIR / "full_decision_model_training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed = tuple(str(value) for value in summary.get("feature_columns", []))
    if observed != FEATURES:
        raise ValueError("deployed model feature order does not match DECISION_BEHAVIOR_FEATURE_COLUMNS")
    if summary.get("feature_group") != "B2+Motion+SearchMaturityLinear":
        raise ValueError(f"unexpected feature group: {summary.get('feature_group')}")
    if summary.get("selected_model_name") != "random_forest_regressor":
        raise ValueError(f"unexpected model: {summary.get('selected_model_name')}")
    threshold_table = pd.read_parquet(DECISION_DIR / "decision_thresholds.parquet")
    rows = threshold_table[threshold_table["threshold_mode"].astype(str).eq("oof_g_fe_selected_path_first_trigger")]
    if len(rows) != 1:
        raise ValueError("expected one predefined threshold row")
    threshold = float(rows.iloc[0]["threshold"])
    if not np.isclose(threshold, USER_THRESHOLD, rtol=0.0, atol=2e-10):
        raise ValueError(f"unexpected threshold: {threshold}")
    return joblib.load(DECISION_DIR / "models/random_forest_regressor.joblib"), threshold, summary


def _problem_config(function: int, dimension: int, boundary_handling: str) -> dict[str, Any]:
    return {
        "suite": "cec2017",
        "function": int(function),
        "instance": 1,
        "dimension": int(dimension),
        "boundary_handling": str(boundary_handling),
    }


def _collect_cec_opportunities(
    *,
    config: dict[str, Any],
    function: int,
    dimension: int,
    seed: int,
    condition: str,
) -> list[dict[str, Any]]:
    fe_total = int(config["FE_total_by_dimension"][dimension])
    boundary_handling = str(config.get("boundary_handling", "reflect"))
    if condition == "cec_unit_cube":
        problem = _make_unit_cube_problem(_problem_config(function, dimension, boundary_handling))
    elif condition == "cec_native":
        problem = make_problem(_problem_config(function, dimension, boundary_handling))
    else:
        raise ValueError(f"unknown CEC condition: {condition}")

    settings = OptimizerSettings(population_size=POPULATION_SIZE, checkpoint_ratios=(1.0,))
    stream = StreamingBehaviorState(
        problem_id=problem.problem_id,
        function_id=problem.function_id,
        family=problem.family,
        dimension=problem.dimension,
        algorithm="cmaes",
        seed=int(seed),
        fe_total=fe_total,
        sampling_protocol=SAMPLING_PROTOCOL,
    )
    rows: list[dict[str, Any]] = []
    try:
        state = initialize_optimizer_state(
            algorithm="cmaes",
            problem=problem,
            seed=int(seed),
            settings=settings,
        )
        stream.observe(
            fe=int(state.evaluations),
            native_updates=int(state.generation),
            population=state.population,
            fitness=state.fitness,
            best_fitness=state.best_fitness,
        )
        current_fe = int(state.evaluations)
        opportunity_index = 0
        while stream.next_monitor_ratio is not None:
            delta = min(POPULATION_SIZE, fe_total - current_fe)
            if delta <= 0:
                break
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=delta,
                on_native_update=lambda updated: stream.observe(
                    fe=int(updated.evaluations),
                    native_updates=int(updated.generation),
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                ),
            )
            current_fe = int(state.evaluations)
            behavior = stream.sample_dynamic()
            if behavior is None:
                continue
            opportunity_index += 1
            row = {
                "source": condition,
                "condition": condition,
                "problem_id": str(problem.problem_id),
                "function_id": str(problem.function_id),
                "family": str(problem.family),
                "cv_group_id": str(problem.cv_group_id),
                "dimension": int(dimension),
                "seed": int(seed),
                "FE": int(behavior["FE"]),
                "FE_ratio": float(behavior["FE_ratio"]),
                "FE_total": int(fe_total),
                "sampling_phase": str(behavior["sampling_phase"]),
                "opportunity_index": int(opportunity_index),
                **{feature: behavior[feature] for feature in FEATURES},
            }
            rows.append(row)
        # The predefined dynamic sampling protocol has no Decision
        # opportunities after the 0.60 budget milestone.  The online policy
        # continues the optimizer to FE_total after this point, but there are
        # no further behavior rows to align.
    finally:
        problem.close()
    return rows


def _collect_cec_matrix(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(CEC_FUNCTIONS) * len(CEC_DIMENSIONS) * len(CEC_SEEDS) * 2
    completed = 0
    for condition in ("cec_native", "cec_unit_cube"):
        # The unit-cube online runner also redirects the bounds used by the
        # feature and window-statistics modules.  Reuse that same routing here
        # so the reconstructed behavior features are identical to online.
        with (_unit_cube_runtime() if condition == "cec_unit_cube" else nullcontext()):
            for function in CEC_FUNCTIONS:
                for dimension in CEC_DIMENSIONS:
                    for seed in CEC_SEEDS:
                        print(f"{condition} F{function:02d} D{dimension} seed={seed}", flush=True)
                        rows.extend(
                            _collect_cec_opportunities(
                                config=config,
                                function=function,
                                dimension=dimension,
                                seed=seed,
                                condition=condition,
                            )
                        )
                        completed += 1
                        print(f"opportunity replay {completed}/{total}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("CEC opportunity matrix is empty")
    return frame


def _read_bbob_oof() -> pd.DataFrame:
    decision_path = DECISION_DIR / "decision_dataset.parquet"
    decision_columns = [
        "split",
        "dataset_role",
        "problem_id",
        "function_id",
        "family",
        "cv_group_id",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "FE_total",
        "sampling_phase",
        *FEATURES,
    ]
    decision = pd.read_parquet(decision_path, columns=decision_columns)
    decision = decision[
        decision["split"].astype(str).eq("bbob_train")
        & decision["dataset_role"].astype(str).eq("train")
        & decision["prefix_algorithm"].astype(str).eq("cmaes")
    ].copy()
    prediction = pd.read_parquet(
        DECISION_DIR / "train_oof_predictions.parquet",
        columns=[
            "problem_id",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "model_name",
            "decision_score",
        ],
    )
    prediction = prediction[
        prediction["model_name"].astype(str).eq("random_forest_regressor")
        & prediction["problem_id"].astype(str).str.startswith("bbob_")
    ].copy()
    keys = ["problem_id", "dimension", "prefix_algorithm", "seed", "FE"]
    if decision.duplicated(keys).any() or prediction.duplicated(keys).any():
        raise ValueError("BBOB train opportunity key is not unique")
    merged = decision.merge(
        prediction[keys + ["decision_score"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged["decision_score"].isna().any():
        raise ValueError("BBOB-train OOF scores are missing for some opportunities")
    merged["source"] = "bbob_train_oof"
    merged["condition"] = "bbob_train_oof"
    merged["opportunity_index"] = merged.sort_values(keys).groupby(
        ["problem_id", "dimension", "seed"], sort=False
    ).cumcount() + 1
    merged = merged.rename(columns={"prefix_algorithm": "algorithm"})
    return merged[
        list(META_COLUMNS[:-1]) + ["decision_score"] + list(FEATURES)
    ].copy()


def _score_cec(cec: pd.DataFrame, model: object) -> pd.DataFrame:
    scored = cec.copy()
    values = scored.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scored["decision_score"] = np.asarray(decision_scores(model, values), dtype=float)
    return scored


def _validate_opportunities(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != len(frame.drop_duplicates(["source", *KEY_COLUMNS])):
        raise ValueError("duplicate source-specific opportunity key")
    if frame[list(FEATURES)].isna().all(axis=1).any():
        raise ValueError("some opportunities have no behavior features")
    counts = frame.groupby("source").size().to_dict()
    run_counts = frame.groupby("source")[list(KEY_COLUMNS[:4])].apply(lambda x: len(x.drop_duplicates())).to_dict()
    return {
        "rows_by_source": {str(k): int(v) for k, v in counts.items()},
        "runs_by_source": {str(k): int(v) for k, v in run_counts.items()},
        "key_columns": list(KEY_COLUMNS),
        "feature_count": len(FEATURES),
    }


def _support_and_score_decomposition(frame: pd.DataFrame, bbob: pd.DataFrame, model: object, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cec = frame[frame["source"].isin(["cec_native", "cec_unit_cube"])].copy()
    rows: list[dict[str, Any]] = []
    enriched: list[pd.DataFrame] = []
    for (condition, dimension), group in cec.groupby(["condition", "dimension"], sort=True):
        reference_dimension = _dimension_reference(int(dimension))
        reference = bbob[bbob["dimension"].astype(int).eq(reference_dimension)]
        if reference.empty:
            raise ValueError(f"no BBOB reference for CEC dimension {dimension}")
        lower = reference.loc[:, FEATURES].quantile(0.01)
        upper = reference.loc[:, FEATURES].quantile(0.99)
        median = reference.loc[:, FEATURES].median()
        iqr = reference.loc[:, FEATURES].quantile(0.75) - reference.loc[:, FEATURES].quantile(0.25)
        scale = iqr.mask(~np.isfinite(iqr) | (iqr <= 1e-12), 1.0)
        values = group.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").copy()
        outside = values.lt(lower, axis="columns") | values.gt(upper, axis="columns")
        robust_z = (values - median) / scale
        clipped_values = values.clip(lower=lower, upper=upper, axis="columns")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clipped_score = np.asarray(decision_scores(model, clipped_values.to_numpy(dtype=float)), dtype=float)
        item = group.copy()
        item["reference_dimension"] = reference_dimension
        item["n_features_outside_reference_q01_q99"] = outside.sum(axis=1).to_numpy(dtype=int)
        item["max_abs_robust_z"] = np.max(np.abs(robust_z.to_numpy(dtype=float)), axis=1)
        item["score_clipped_to_reference_q01_q99"] = clipped_score
        item["score_delta_clipped"] = clipped_score - item["decision_score"].to_numpy(dtype=float)
        enriched.append(item)
        original = item["decision_score"].to_numpy(dtype=float)
        delta = item["score_delta_clipped"].to_numpy(dtype=float)
        clipped = item["score_clipped_to_reference_q01_q99"].to_numpy(dtype=float)
        outside_count = item["n_features_outside_reference_q01_q99"].to_numpy(dtype=int)
        rows.append(
            {
                "condition": condition,
                "dimension": int(dimension),
                "reference_dimension": reference_dimension,
                "n_opportunities": int(len(item)),
                "any_feature_outside_q01_q99_rate": float(np.mean(outside_count >= 1)),
                "at_least_5_features_outside_rate": float(np.mean(outside_count >= 5)),
                "at_least_10_features_outside_rate": float(np.mean(outside_count >= 10)),
                "median_features_outside_q01_q99": float(np.median(outside_count)),
                "q95_max_abs_robust_z": float(np.quantile(item["max_abs_robust_z"], 0.95)),
                "score_median": float(np.median(original)),
                "score_q95": float(np.quantile(original, 0.95)),
                "score_max": float(np.max(original)),
                "score_gt_zero_rate": float(np.mean(original > 0.0)),
                "score_gt_threshold_rate": float(np.mean(original > threshold)),
                "score_clipped_median": float(np.median(clipped)),
                "score_delta_clipped_median": float(np.median(delta)),
                "score_delta_clipped_q95": float(np.quantile(delta, 0.95)),
                "score_clipped_gt_zero_rate": float(np.mean(clipped > 0.0)),
                "score_clipped_gt_threshold_rate": float(np.mean(clipped > threshold)),
                "clipped_cross_zero_rate": float(np.mean((original <= 0.0) & (clipped > 0.0))),
                "clipped_cross_threshold_rate": float(np.mean((original <= threshold) & (clipped > threshold))),
            }
        )
    enriched_frame = pd.concat(enriched, ignore_index=True)
    return pd.DataFrame(rows), enriched_frame


def _feature_distribution_summary(frame: pd.DataFrame, bbob: pd.DataFrame) -> pd.DataFrame:
    cec = frame[frame["source"].isin(["cec_native", "cec_unit_cube"])].copy()
    rows: list[dict[str, Any]] = []
    for (condition, dimension, phase), group in cec.groupby(
        ["condition", "dimension", "sampling_phase"], sort=True
    ):
        reference_dimension = _dimension_reference(int(dimension))
        reference = bbob[
            bbob["dimension"].astype(int).eq(reference_dimension)
            & bbob["sampling_phase"].astype(str).eq(str(phase))
        ]
        if len(reference) < 2:
            reference = bbob[bbob["dimension"].astype(int).eq(reference_dimension)]
        for feature in FEATURES:
            ref_values = _finite(reference[feature])
            cec_values = _finite(group[feature])
            if len(ref_values) < 2 or len(cec_values) < 2:
                continue
            ref_q01, ref_q99 = np.quantile(ref_values, [0.01, 0.99])
            ref_iqr = float(np.quantile(ref_values, 0.75) - np.quantile(ref_values, 0.25))
            scale = ref_iqr if np.isfinite(ref_iqr) and ref_iqr > 1e-12 else 1.0
            ks = ks_2samp(ref_values, cec_values, alternative="two-sided", mode="auto")
            rows.append(
                {
                    "condition": condition,
                    "dimension": int(dimension),
                    "phase": str(phase),
                    "reference_dimension": reference_dimension,
                    "feature": feature,
                    "n_cec": int(len(cec_values)),
                    "n_bbob_reference": int(len(ref_values)),
                    "cec_median": float(np.median(cec_values)),
                    "bbob_median": float(np.median(ref_values)),
                    "median_delta_cec_minus_bbob": float(np.median(cec_values) - np.median(ref_values)),
                    "robust_median_shift_iqr": float((np.median(cec_values) - np.median(ref_values)) / scale),
                    "cec_q01": float(np.quantile(cec_values, 0.01)),
                    "cec_q99": float(np.quantile(cec_values, 0.99)),
                    "bbob_q01": float(ref_q01),
                    "bbob_q99": float(ref_q99),
                    "outside_bbob_q01_q99_rate": float(np.mean((cec_values < ref_q01) | (cec_values > ref_q99))),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein_over_bbob_iqr": float(wasserstein_distance(ref_values, cec_values) / scale),
                }
            )
    return pd.DataFrame(rows)


def _score_distribution_summary(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    working = frame.copy()
    working["phase"] = working["FE_ratio"].map(_phase_bin)
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(["source", "dimension", "phase"], sort=True):
        source, dimension, phase = keys
        scores = _finite(group["decision_score"])
        if len(scores) == 0:
            continue
        rows.append(
            {
                "source": source,
                "source_label": SOURCE_LABELS.get(str(source), str(source)),
                "dimension": int(dimension),
                "phase": str(phase),
                "n": int(len(scores)),
                "score_mean": float(np.mean(scores)),
                "score_median": float(np.median(scores)),
                "score_q05": float(np.quantile(scores, 0.05)),
                "score_q95": float(np.quantile(scores, 0.95)),
                "score_max": float(np.max(scores)),
                "rate_score_gt_zero": float(np.mean(scores > 0.0)),
                "rate_score_gt_threshold": float(np.mean(scores > threshold)),
            }
        )
    for source, group in working.groupby("source", sort=True):
        scores = _finite(group["decision_score"])
        rows.append(
            {
                "source": source,
                "source_label": SOURCE_LABELS.get(str(source), str(source)),
                "dimension": 0,
                "phase": "overall",
                "n": int(len(scores)),
                "score_mean": float(np.mean(scores)),
                "score_median": float(np.median(scores)),
                "score_q05": float(np.quantile(scores, 0.05)),
                "score_q95": float(np.quantile(scores, 0.95)),
                "score_max": float(np.max(scores)),
                "rate_score_gt_zero": float(np.mean(scores > 0.0)),
                "rate_score_gt_threshold": float(np.mean(scores > threshold)),
            }
        )
    return pd.DataFrame(rows)


def _score_calibration_summary(frame: pd.DataFrame, bbob: pd.DataFrame, threshold: float) -> pd.DataFrame:
    bbob_scores = _finite(bbob["decision_score"])
    rows: list[dict[str, Any]] = []
    for source, group in frame[frame["source"].isin(["cec_native", "cec_unit_cube"])].groupby("source", sort=True):
        scores = _finite(group["decision_score"])
        ranks = np.searchsorted(np.sort(bbob_scores), scores, side="right") / len(bbob_scores)
        rows.append(
            {
                "source": source,
                "source_label": SOURCE_LABELS.get(str(source), str(source)),
                "n": int(len(scores)),
                "bbob_train_oof_reference_n": int(len(bbob_scores)),
                "score_median": float(np.median(scores)),
                "bbob_score_percentile_median": float(np.median(ranks)),
                "bbob_score_percentile_q05": float(np.quantile(ranks, 0.05)),
                "bbob_score_percentile_q95": float(np.quantile(ranks, 0.95)),
                "score_gt_zero_rate": float(np.mean(scores > 0.0)),
                "score_gt_threshold_rate": float(np.mean(scores > threshold)),
                "bbob_reference_score_median": float(np.median(bbob_scores)),
                "bbob_reference_score_q95": float(np.quantile(bbob_scores, 0.95)),
                "bbob_reference_score_max": float(np.max(bbob_scores)),
            }
        )
    return pd.DataFrame(rows)


def _cec_native_unit_paired_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Summarize exact within-CEC Native/Unit-cube opportunity pairs.

    The two coordinate parameterizations can emit different event opportunities,
    so the comparison is an inner join on the requested function/dimension/seed/FE
    key.  BBOB is a separate suite and is therefore not included in this exact
    cross-condition join.
    """
    key = ["function_id", "dimension", "seed", "FE"]
    native = frame[frame["source"].eq("cec_native")].set_index(key)
    unit = frame[frame["source"].eq("cec_unit_cube")].set_index(key)
    common = native.index.intersection(unit.index)
    rows: list[dict[str, Any]] = []
    for feature in ["decision_score", *FEATURES]:
        native_values = pd.to_numeric(native.loc[common, feature], errors="coerce").to_numpy(dtype=float)
        unit_values = pd.to_numeric(unit.loc[common, feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(native_values) & np.isfinite(unit_values)
        native_values = native_values[finite]
        unit_values = unit_values[finite]
        delta = unit_values - native_values
        if len(delta) == 0:
            continue
        rows.append(
            {
                "feature": feature,
                "n_exact_pairs": int(len(delta)),
                "median_unit_minus_native": float(np.median(delta)),
                "mean_unit_minus_native": float(np.mean(delta)),
                "q05_unit_minus_native": float(np.quantile(delta, 0.05)),
                "q95_unit_minus_native": float(np.quantile(delta, 0.95)),
                "median_abs_difference": float(np.median(np.abs(delta))),
                "q95_abs_difference": float(np.quantile(np.abs(delta), 0.95)),
                "unit_greater_than_native_rate": float(np.mean(delta > 0.0)),
                "exact_equal_rate_at_1e12": float(np.mean(np.isclose(delta, 0.0, rtol=0.0, atol=1.0e-12))),
            }
        )
    counts = {
        "exact_paired_rows": int(len(common)),
        "native_only_rows": int(len(native.index.difference(unit.index))),
        "unit_only_rows": int(len(unit.index.difference(native.index))),
    }
    return pd.DataFrame(rows), counts


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_heatmap(feature_summary: pd.DataFrame, output: Path) -> None:
    overall = feature_summary[feature_summary["phase"].astype(str).eq("early")].copy()
    overall["comparison"] = overall["condition"].map({"cec_native": "Native", "cec_unit_cube": "Unit-cube"})
    overall["dimension_group"] = np.where(overall["dimension"].isin([10, 20]), "CEC 10/20D", "CEC 30/50D")
    overall = overall[overall["dimension"].isin([10, 20, 30, 50])]
    overall["sort_value"] = overall["robust_median_shift_iqr"].abs()
    order = (
        overall.groupby("feature")["sort_value"].max().sort_values(ascending=False).head(29).index.tolist()
    )
    pivot = overall.groupby(["feature", "comparison"], sort=False)["robust_median_shift_iqr"].mean().unstack("comparison")
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(8.2, 10.5))
    matrix = pivot.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(matrix))), 1.0)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value).removeprefix("bf_").replace("_", " ") for value in pivot.index], fontsize=8)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("CEC coordinate condition")
    ax.set_title("CEC behavior-feature shift vs BBOB-train")
    ax.set_ylabel("Decision behavior feature")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("robust median shift / BBOB IQR")
    _save_figure(fig, output / "fig01_feature_shift_heatmap.png")


def _plot_score_distributions(frame: pd.DataFrame, threshold: float, output: Path) -> None:
    groups = [
        ("bbob_train_oof", "BBOB-train OOF", PALETTE["bbob"]),
        ("cec_native", "CEC Native", PALETTE["native"]),
        ("cec_unit_cube", "CEC Unit-cube", PALETTE["unit"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, dimensions, title in zip(axes, [(10, 20), (30, 40, 50)], ["10/20D", "30/40/50D"]):
        data = []
        labels = []
        colors = []
        for source, label, color in groups:
            values = frame[frame["source"].eq(source) & frame["dimension"].isin(dimensions)]["decision_score"]
            values = _finite(values)
            if len(values):
                data.append(values)
                labels.append(label)
                colors.append(color)
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor(PALETTE["charcoal"])
        ax.axhline(0.0, color=PALETTE["charcoal"], linewidth=1.0, linestyle="--", label="score = 0")
        ax.axhline(threshold, color=PALETTE["unit"], linewidth=1.0, linestyle=":", label="configured threshold")
        ax.set_title(f"{title}: complete opportunity scores")
        ax.set_ylabel("Decision score")
        ax.tick_params(axis="x", labelrotation=18)
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Decision-score distributions: BBOB-train OOF vs CEC", x=0.06, ha="left", fontsize=14)
    _save_figure(fig, output / "fig02_score_distributions.png")


def _plot_support_score(enriched: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, condition, title, color in [
        (axes[0], "cec_native", "CEC Native", PALETTE["native"]),
        (axes[1], "cec_unit_cube", "CEC Unit-cube", PALETTE["unit"]),
    ]:
        data = enriched[enriched["condition"].eq(condition)]
        ax.scatter(
            data["n_features_outside_reference_q01_q99"],
            data["decision_score"],
            s=8,
            alpha=0.25,
            color=color,
            edgecolors="none",
        )
        ax.axhline(0.0, color=PALETTE["charcoal"], linewidth=1.0, linestyle="--")
        ax.axhline(USER_THRESHOLD, color=PALETTE["unit"], linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("features outside BBOB q01–q99")
        ax.set_xlim(left=-0.5)
    axes[0].set_ylabel("Decision score")
    fig.suptitle("Out-of-support feature count vs Decision score", x=0.06, ha="left", fontsize=14)
    _save_figure(fig, output / "fig03_support_count_vs_score.png")


def _plot_score_by_phase(frame: pd.DataFrame, threshold: float, output: Path) -> None:
    working = frame.copy()
    working["phase"] = working["FE_ratio"].map(_phase_bin)
    phase_order = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, dimensions, title in zip(axes, [(10, 20), (30, 40, 50)], ["10/20D", "30/40/50D"]):
        for source, label, color, linestyle in [
            ("bbob_train_oof", "BBOB-train OOF", PALETTE["bbob"], "-"),
            ("cec_native", "CEC Native", PALETTE["native"], "--"),
            ("cec_unit_cube", "CEC Unit-cube", PALETTE["unit"], ":"),
        ]:
            grouped = working[working["source"].eq(source) & working["dimension"].isin(dimensions)].groupby("phase")["decision_score"].median()
            values = [float(grouped.get(phase, np.nan)) for phase in phase_order]
            ax.plot(phase_order, values, marker="o", linewidth=1.8, label=label, color=color, linestyle=linestyle)
        ax.axhline(0.0, color=PALETTE["charcoal"], linewidth=1.0, linestyle="--")
        ax.axhline(threshold, color=PALETTE["unit"], linewidth=1.0, linestyle=":")
        ax.set_title(f"{title}: median Decision score")
        ax.set_xlabel("common FE-ratio phase")
    axes[0].set_ylabel("Decision score median")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Decision score by common FE-ratio phase", x=0.06, ha="left", fontsize=14)
    _save_figure(fig, output / "fig04_score_by_phase.png")


def _write_report(
    output: Path,
    validation: dict[str, Any],
    support: pd.DataFrame,
    calibration: pd.DataFrame,
    paired: pd.DataFrame,
    paired_counts: dict[str, int],
    enriched: pd.DataFrame,
) -> None:
    def fmt(value: Any, digits: int = 5) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NA"
        return f"{value:.{digits}g}" if np.isfinite(value) else "NA"

    cec_support = support.groupby("condition", sort=True).agg(
        n_opportunities=("n_opportunities", "sum"),
        any_outside=("any_feature_outside_q01_q99_rate", "mean"),
        at_least_5=("at_least_5_features_outside_rate", "mean"),
        median_score=("score_median", "median"),
        score_max=("score_max", "max"),
        clip_delta=("score_delta_clipped_median", "median"),
        clip_cross_zero=("clipped_cross_zero_rate", "mean"),
    ).reset_index()
    lines = [
        "# CEC2017 完整 Decision opportunity 对齐诊断",
        "",
        "> 仅用于诊断当前固定 Decision 模型在 CEC 上的输入域与 score 分布；不重新拟合模型、预处理或阈值，不改变主 online 测评结果。",
        "",
        "## 研究问题",
        "",
        "在相同的 within-run `function × dimension × seed × FE` opportunity 粒度下，判断 CEC 不触发主要来自：",
        "",
        "1. CMA-ES 搜索轨迹的 29 个行为特征没有进入 BBOB-train 的经验支持范围；或",
        "2. 特征已经接近训练域，但固定模型/稀有标签/预先指定阈值使 score 整体低于触发边界。",
        "",
        "## 数据与对齐",
        "",
        f"- BBOB-train OOF：{validation['rows_by_source'].get('bbob_train_oof', 0)} 个 opportunity，score 为固定模型的 train OOF 输出。",
        f"- CEC Native：{validation['rows_by_source'].get('cec_native', 0)} 个 opportunity；CEC Unit-cube：{validation['rows_by_source'].get('cec_unit_cube', 0)} 个 opportunity。",
        f"- within-run 唯一键：`{', '.join(KEY_COLUMNS)}`；行为特征：29 个 `bf_*`。",
        "- 跨套件不强行匹配函数编号；在唯一键核对后，用共同 `FE_ratio` 和 sampling phase 做分布汇总。CEC 10/20D 使用同维度 BBOB 参考，CEC 30/50D 使用 BBOB 40D 作为最近的高维参考。",
        "",
        "## 主要结果",
        "",
        "| condition | opportunities | any feature outside BBOB q01–q99 | ≥5 features outside | score median | score max | median score shift after clipping | clipped cross score=0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cec_support.iterrows():
        lines.append(
            f"| {row['condition']} | {int(row['n_opportunities'])} | {fmt(row['any_outside'] * 100, 4)}% | {fmt(row['at_least_5'] * 100, 4)}% | {fmt(row['median_score'], 6)} | {fmt(row['score_max'], 6)} | {fmt(row['clip_delta'], 6)} | {fmt(row['clip_cross_zero'] * 100, 4)}% |"
        )
    lines.extend(
        [
            "",
            "## Native 与 Unit-cube 的精确 opportunity 配对",
            "",
            f"- 按 `function_id × dimension × seed × FE` 做 inner join，得到 {paired_counts['exact_paired_rows']} 对；Native 独有 {paired_counts['native_only_rows']} 行，Unit-cube 独有 {paired_counts['unit_only_rows']} 行。独有行来自动态事件采样在两种坐标参数化下产生了不同的实际 opportunity FE。",
            "- BBOB-train 属于不同 benchmark suite，不能把 `function_id` 与 CEC 强行视为同一函数；因此 BBOB 只作为同维度/共同 phase 的经验参考，不进入该精确配对。",
            "",
            "| feature | exact pairs | median Unit−Native | q05 | q95 | median absolute difference |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    paired_display = paired.sort_values("median_abs_difference", ascending=False)
    score_row = paired[paired["feature"].eq("decision_score")]
    display_rows = pd.concat([score_row, paired_display[~paired_display["feature"].eq("decision_score")].head(5)], ignore_index=True)
    for _, row in display_rows.iterrows():
        lines.append(
            f"| `{row['feature']}` | {int(row['n_exact_pairs'])} | {fmt(row['median_unit_minus_native'], 6)} | {fmt(row['q05_unit_minus_native'], 6)} | {fmt(row['q95_unit_minus_native'], 6)} | {fmt(row['median_abs_difference'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## 支持范围分层与 score",
            "",
            "如果某行 29 个行为特征都位于对应 BBOB 参考的 q01–q99 内，但 score 仍不超过 0，则‘未进入训练域’不是触发失败的必要条件。以下统计直接按 opportunity 行计算。",
            "",
            "| condition | rows with no feature outside | median score (no feature outside) | max score (no feature outside) | median score (any feature outside) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, group in enriched.groupby("condition", sort=True):
        outside = pd.to_numeric(group["n_features_outside_reference_q01_q99"], errors="coerce").to_numpy(dtype=float) > 0.0
        inside_scores = pd.to_numeric(group.loc[~outside, "decision_score"], errors="coerce").dropna()
        outside_scores = pd.to_numeric(group.loc[outside, "decision_score"], errors="coerce").dropna()
        lines.append(
            f"| {condition} | {fmt(100.0 * np.mean(~outside), 5)}% | {fmt(inside_scores.median() if len(inside_scores) else np.nan, 6)} | {fmt(inside_scores.max() if len(inside_scores) else np.nan, 6)} | {fmt(outside_scores.median() if len(outside_scores) else np.nan, 6)} |"
        )
    lines.extend(
        [
            "",
            "## 解释口径",
            "",
            "- `outside q01–q99` 衡量 CEC 行为输入是否落在 BBOB-train 的边缘支持范围外；它是分布诊断，不是模型错误的直接证明。",
            "- `score_clipped_to_reference_q01_q99` 是把 CEC 每个输入单独截到 BBOB 参考的 1%–99% 区间后重新计算固定模型 score 的敏感性检查。若截断后 score 大幅上升或跨过 0，说明输入域偏移可能是重要因素；若变化很小而 score 仍低，则更符合输出校准/标签稀有性与阈值迁移问题。该检查不代表重新运行优化器。",
            f"- 两个触发边界分别是 `score > 0` 与 `score > {USER_THRESHOLD}`；严格比较使用数值大于，而不是大于等于。",
            "",
            "## 固定模型 score 参照",
            "",
            calibration.to_markdown(index=False),
            "",
            "## 输出",
            "",
            "- `opportunity_rows.parquet`：BBOB-train OOF、CEC Native、CEC Unit-cube 的完整 opportunity 与 29 个行为特征。",
            "- `feature_distribution_summary.csv`：按 CEC condition、维度、phase 的特征分布位移。",
            "- `support_score_summary.csv`：训练支持范围、score 和 clipping 敏感性摘要。",
            "- `score_distribution_summary.csv`：三方 score 的总体与阶段分布。",
            "- `score_calibration_summary.csv`：CEC score 在 BBOB-train OOF score 经验分布中的百分位。",
            "- `cec_native_unit_paired_summary.csv`：Native 与 Unit-cube 精确 key 配对后的 29 个特征及 score 差异。",
            "- `fig01_feature_shift_heatmap.png` 至 `fig04_score_by_phase.png`：分布与 score 可视化。",
            "",
            "## 下一步建议",
            "",
            "若 clipping 后的 score 仍几乎不跨过 0，优先做固定模型下的阈值可迁移性敏感性分析；若 clipping 能显著提高 score，再单独检查造成位移的高重要性行为特征及其与边界处理的关系。两者都不应直接替换当前主模型或阈值。",
        ]
    )
    (output / "cec2017_opportunity_alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(output_dir: Path, overwrite: bool, refresh_opportunities: bool = False) -> dict[str, Any]:
    _validate_feature_contract()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, threshold, model_summary = _read_model()
    config = load_config(CEC_CONFIG)
    raw_cec_path = output_dir / "cec_opportunities_unscored.parquet"
    if raw_cec_path.exists() and not refresh_opportunities:
        cec_raw = pd.read_parquet(raw_cec_path)
    else:
        cec_raw = _collect_cec_matrix(config)
        cec_raw.to_parquet(raw_cec_path, index=False)
    cec = _score_cec(cec_raw, model)
    bbob = _read_bbob_oof()
    bbob["decision_score"] = pd.to_numeric(bbob["decision_score"], errors="coerce")
    frame = pd.concat([bbob, cec], ignore_index=True, sort=False)
    validation = _validate_opportunities(frame)
    support, enriched_cec = _support_and_score_decomposition(frame, bbob, model, threshold)
    feature_summary = _feature_distribution_summary(frame, bbob)
    score_summary = _score_distribution_summary(frame, threshold)
    calibration = _score_calibration_summary(frame, bbob, threshold)
    paired, paired_counts = _cec_native_unit_paired_summary(frame)

    frame.to_parquet(output_dir / "opportunity_rows.parquet", index=False)
    enriched_cec.to_parquet(output_dir / "cec_support_score_rows.parquet", index=False)
    feature_summary.to_csv(output_dir / "feature_distribution_summary.csv", index=False)
    support.to_csv(output_dir / "support_score_summary.csv", index=False)
    score_summary.to_csv(output_dir / "score_distribution_summary.csv", index=False)
    calibration.to_csv(output_dir / "score_calibration_summary.csv", index=False)
    paired.to_csv(output_dir / "cec_native_unit_paired_summary.csv", index=False)

    _set_plot_style()
    _plot_feature_heatmap(feature_summary, output_dir)
    _plot_score_distributions(frame, threshold, output_dir)
    _plot_support_score(enriched_cec, output_dir)
    _plot_score_by_phase(frame, threshold, output_dir)

    native_reference_check: dict[str, Any] = {"status": "reference_missing"}
    if NATIVE_REFERENCE.exists():
        reference = pd.read_parquet(NATIVE_REFERENCE)
        keys = ["problem_id", "function_id", "dimension", "seed", "FE"]
        left = cec[cec["source"].eq("cec_native")][keys + ["decision_score"]]
        right = reference[keys + ["decision_score"]].rename(columns={"decision_score": "reference_score"})
        merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
        native_reference_check = {
            "status": "matched",
            "matched_rows": int(len(merged)),
            "direct_native_rows": int(len(left)),
            "reference_rows": int(len(right)),
            "max_abs_score_difference": float(np.max(np.abs(merged["decision_score"] - merged["reference_score"]))) if len(merged) else None,
        }

    online_checks = []
    for label, path, condition in [
        ("native_online", NATIVE_ONLINE, "cec_native"),
        ("unit_online", UNIT_ONLINE, "cec_unit_cube"),
    ]:
        if not path.exists():
            continue
        online = pd.read_parquet(path, columns=["policy_name", "function", "dimension", "seed", "decision_score"])
        online = online[online["policy_name"].astype(str).isin(["predicted_G_FE_gt_0", "predicted_g_fe_selected_path_gt_0.2997557291"])]
        direct = cec[cec["condition"].eq(condition)].sort_values(["function_id", "dimension", "seed", "FE"]).groupby(["function_id", "dimension", "seed"], sort=False).tail(1)
        direct = direct.assign(function=direct["function_id"].astype(str).str.extract(r"f(\d+)")[0].astype(int))["function dimension seed decision_score".split()]
        online_last = online.sort_values(["function", "dimension", "seed"]).groupby(
            ["policy_name", "function", "dimension", "seed"], sort=False
        ).tail(1)
        merged = online_last.merge(direct, on=["function", "dimension", "seed"], suffixes=("_online", "_direct"))
        online_checks.append(
            {
                "source": label,
                "condition": condition,
                "matched_rows": int(len(merged)),
                "max_abs_score_difference": float(np.max(np.abs(merged["decision_score_online"] - merged["decision_score_direct"]))) if len(merged) else None,
            }
        )
    summary = {
        "status": "ok",
        "analysis": "cec2017_opportunity_alignment",
        "model": "random_forest_regressor",
        "feature_group": model_summary.get("feature_group"),
        "feature_count": len(FEATURES),
        "threshold": threshold,
        "alignment": validation,
        "native_reference_check": native_reference_check,
        "online_score_checks": online_checks,
        "cec_native_unit_exact_pairing": paired_counts,
        "sources": {
            "bbob_train_oof": str(DECISION_DIR / "train_oof_predictions.parquet"),
            "decision_features": str(DECISION_DIR / "decision_dataset.parquet"),
            "native_reference": str(NATIVE_REFERENCE),
            "native_online": str(NATIVE_ONLINE),
            "unit_online": str(UNIT_ONLINE),
        },
        "outputs": {
            "opportunity_rows": str(output_dir / "opportunity_rows.parquet"),
            "cec_support_score_rows": str(output_dir / "cec_support_score_rows.parquet"),
            "feature_distribution_summary": str(output_dir / "feature_distribution_summary.csv"),
            "support_score_summary": str(output_dir / "support_score_summary.csv"),
            "score_distribution_summary": str(output_dir / "score_distribution_summary.csv"),
            "score_calibration_summary": str(output_dir / "score_calibration_summary.csv"),
            "cec_native_unit_paired_summary": str(output_dir / "cec_native_unit_paired_summary.csv"),
            "report": str(output_dir / "cec2017_opportunity_alignment_report.md"),
        },
        "data_leakage_check": {
            "cec_rows_used_for_model_fit": 0,
            "cec_rows_used_for_threshold_fit": 0,
            "model_refit": False,
            "threshold_refit": False,
        },
    }
    (output_dir / "cec2017_opportunity_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(output_dir, validation, support, calibration, paired, paired_counts, enriched_cec)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh-opportunities", action="store_true")
    args = parser.parse_args()
    result = run_analysis(args.output_dir, args.overwrite, args.refresh_opportunities)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
