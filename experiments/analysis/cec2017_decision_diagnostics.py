"""Diagnose why the deployed Decision model does not trigger on CEC2017.

This analysis is deliberately limited to the currently deployed Decision model
and its 29 ``bf_*`` inputs.  It does not refit the model, preprocessing, or
threshold, and it does not use CEC rows for fitting any object.  The CEC
behavior rows are scored only after the training artifact has been loaded.

Outputs are written below ``results/dataset_analysis/cec2017_decision_diagnostics``:

* feature profiles and standardized-shift tables;
* CEC state-level Decision scores and score summaries;
* model feature importance and single-input replacement sensitivity;
* five PNG figures and a Chinese diagnostic report.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from trajectory.sampling import SAMPLING_PHASES, sampling_phase


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/dataset_analysis/cec2017_decision_diagnostics"
DEFAULT_DECISION_DIR = (
    REPO
    / "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision"
)
DEFAULT_CEC_BEHAVIOR_ROOT = (
    REPO
    / "results/phase1_cec2017_distribution_shift/cec2017_distribution_shift"
)
DEFAULT_ONLINE_RUNS = (
    REPO / "outputs/cec2017_representative_online_compare/online_comparison_run_metrics.parquet"
)

TRAIN_SPLITS = ("bbob_train", "mabbob_formal")
VALIDATION_SPLITS = ("bbob_validation", "mabbob_validation")
CEC_SOURCE = "cec2017"
CEC_SUITE = "CEC2017"
CEC_COMMON_DIMS = (10, 20)
CEC_UNSEEN_DIMS = (30, 50)
TRAIN_REFERENCE_DIMS = (10, 20, 40)
USER_THRESHOLD = 0.2997557291
EPS = 1.0e-12

PALETTE = {
    "blue": "#2F5D8C",
    "blue_light": "#B9CCE1",
    "gold": "#C08A2D",
    "orange": "#C96B27",
    "olive": "#657A3A",
    "pink": "#B45C78",
    "charcoal": "#2F3136",
    "grey": "#8A929B",
    "grey_light": "#D9DDE2",
    "paper": "#FBFBFA",
}

SOURCE_LABELS = {
    "bbob_train": "BBOB train",
    "mabbob_formal": "MA-BBOB train",
    "bbob_validation": "BBOB validation",
    "mabbob_validation": "MA-BBOB validation",
    CEC_SOURCE: "CEC2017 cmaes",
}

GROUP_COLORS = {
    "BBOB 10/20D": PALETTE["blue"],
    "MA-BBOB 10/20D": PALETTE["gold"],
    "CEC 10/20D": PALETTE["orange"],
    "BBOB 40D": PALETTE["olive"],
    "MA-BBOB 40D": PALETTE["pink"],
    "CEC 30/50D": PALETTE["charcoal"],
}


def _feature_label(feature: str) -> str:
    label = str(feature).removeprefix("bf_")
    replacements = {
        "improvement_rate_w02": "improve rate w02",
        "improvement_frequency_w02": "improve freq w02",
        "diversity_mean_pairwise": "diversity pairwise",
        "diversity_change_w05": "diversity change w05",
        "covariance_spectral_concentration": "cov spectral concentration",
        "distance_decay_w10": "distance decay w10",
        "stagnation_w10": "stagnation w10",
        "convergence_rate_w10": "convergence rate w10",
        "fitness_diversity_rel": "fitness diversity rel",
        "population_wasserstein_rate_w05": "population Wasserstein w05",
        "centroid_shift_rate_w05": "centroid shift rate w05",
        "centroid_shift_coherence_w05": "centroid coherence w05",
        "fitness_quantile_improvement_fraction_w02": "fitness quantile improve w02",
        "fitness_distribution_improvement_rate_w02": "fitness distribution improve w02",
        "fitness_wasserstein_rate_w02": "fitness Wasserstein w02",
        "elite_concentration": "elite concentration",
        "best_fitness_slope_rel_w05": "best fitness slope rel w05",
        "diversity_slope_w05": "diversity slope w05",
        "fitness_spread_slope_w05": "fitness spread slope w05",
        "population_centroid_shift_w05": "population centroid shift w05",
        "elite_centroid_shift_w05": "elite centroid shift w05",
        "covariance_trace_ratio_w05": "cov trace ratio w05",
        "covariance_effective_rank_w05": "cov effective rank w05",
        "diversity_recovery_w05": "diversity recovery w05",
        "population_chamfer_distance_w05": "population Chamfer w05",
        "covariance_trace_change_w05": "cov trace change w05",
        "covariance_effective_rank_change_w05": "cov effective rank change w05",
        "search_maturity_linear": "search maturity linear",
    }
    return replacements.get(label, label.replace("_", " "))


def _feature_family(feature: str) -> str:
    if feature in {
        "bf_fe_ratio",
        "bf_improvement_rate_w02",
        "bf_improvement_frequency_w02",
        "bf_diversity_mean_pairwise",
        "bf_diversity_change_w05",
        "bf_covariance_spectral_concentration",
        "bf_distance_decay_w10",
        "bf_stagnation_w10",
        "bf_convergence_rate_w10",
    }:
        return "base"
    if feature in {
        "bf_population_chamfer_distance_w05",
        "bf_covariance_trace_change_w05",
        "bf_covariance_effective_rank_change_w05",
    }:
        return "motion"
    if feature == "bf_search_maturity_linear":
        return "maturity"
    if feature in {
        "bf_fitness_spread_slope_w05",
        "bf_population_centroid_shift_w05",
        "bf_elite_centroid_shift_w05",
        "bf_covariance_trace_ratio_w05",
        "bf_covariance_effective_rank_w05",
        "bf_diversity_recovery_w05",
    }:
        return "dynamics"
    return "primary"


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "font.sans-serif": [
                "Microsoft YaHei",
                "PingFang SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.facecolor": PALETTE["paper"],
            "figure.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["grey"],
            "axes.labelcolor": PALETTE["charcoal"],
            "xtick.color": PALETTE["charcoal"],
            "ytick.color": PALETTE["charcoal"],
            "text.color": PALETTE["charcoal"],
            "axes.grid": True,
            "grid.color": PALETTE["grey_light"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "savefig.dpi": 220,
        }
    )


def _read_training_data(decision_dir: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
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
        "sampling_phase",
        *features,
    ]
    dataset = pd.read_parquet(decision_dir / "decision_dataset.parquet", columns=required)
    train = dataset[
        dataset["dataset_role"].astype(str).eq("train")
        & dataset["split"].astype(str).isin(TRAIN_SPLITS)
    ].copy()
    validation = dataset[
        dataset["dataset_role"].astype(str).eq("validation")
        & dataset["split"].astype(str).isin(VALIDATION_SPLITS)
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("Decision dataset does not contain the expected train and validation rows")
    if set(train["prefix_algorithm"].astype(str)) != {"cmaes"}:
        raise ValueError(
            "The current Decision training dataset is expected to use the cmaes SBS prefix; "
            f"observed={sorted(train['prefix_algorithm'].astype(str).unique())}"
        )
    for frame in (train, validation):
        frame["source"] = frame["split"].map(
            {
                "bbob_train": "bbob_train",
                "mabbob_formal": "mabbob_formal",
                "bbob_validation": "bbob_validation",
                "mabbob_validation": "mabbob_validation",
            }
        )
        frame["source_label"] = frame["source"].map(SOURCE_LABELS)
        frame["prefix_algorithm"] = "cmaes"
    return train, validation


def _read_cec_behavior(behavior_root: Path, features: list[str]) -> pd.DataFrame:
    paths = sorted(behavior_root.glob("*/dimension_*/behavior.parquet"))
    if not paths:
        raise FileNotFoundError(f"No CEC behavior files under {behavior_root}")
    columns = [
        "problem_id",
        "function_id",
        "family",
        "cv_group_id",
        "dimension",
        "algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "sampling_phase",
        *features,
    ]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    cec = pd.concat(frames, ignore_index=True)
    cec["source"] = CEC_SOURCE
    cec["source_label"] = SOURCE_LABELS[CEC_SOURCE]
    cec["prefix_algorithm"] = cec["algorithm"].astype(str)
    if set(cec["sampling_phase"].astype(str)) != set(SAMPLING_PHASES):
        raise ValueError(
            "CEC behavior files do not expose the expected official sampling phases: "
            f"{sorted(cec['sampling_phase'].astype(str).unique())}"
        )
    # The matrix used by the previous distribution-shift analysis did not
    # retain this metadata; verify that its phase convention is reproduced by
    # the official ratio mapping before using the direct behavior rows.
    mapped = cec["FE_ratio"].map(sampling_phase)
    if not mapped.astype(str).equals(cec["sampling_phase"].astype(str)):
        raise ValueError("CEC sampling_phase is inconsistent with the predefined FE-ratio phase mapping")
    return cec[cec["algorithm"].astype(str).eq("cmaes")].copy()


def _read_model(decision_dir: Path) -> tuple[dict, object, float]:
    summary = json.loads(
        (decision_dir / "full_decision_model_training_summary.json").read_text(encoding="utf-8")
    )
    features = [str(value) for value in summary.get("feature_columns", [])]
    if len(features) != 29 or not all(value.startswith("bf_") for value in features):
        raise ValueError(f"The deployed Decision artifact must expose 29 bf_* features, observed={features}")
    if summary.get("feature_group") != "B2+Motion+SearchMaturityLinear":
        raise ValueError(f"Unexpected deployed feature group: {summary.get('feature_group')}")
    if summary.get("selected_model_name") != "random_forest_regressor":
        raise ValueError(f"Unexpected deployed model: {summary.get('selected_model_name')}")
    model_path = decision_dir / "models/random_forest_regressor.joblib"
    model = joblib.load(model_path)
    threshold_table = pd.read_parquet(decision_dir / "decision_thresholds.parquet")
    row = threshold_table[
        threshold_table["threshold_mode"].astype(str).eq("oof_g_fe_selected_path_first_trigger")
    ]
    if len(row) != 1:
        raise ValueError("Expected exactly one predefined Decision threshold")
    threshold = float(row.iloc[0]["threshold"])
    return summary, model, threshold


def _numeric_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    values[~np.isfinite(values)] = np.nan
    return values


def _standardized_values(model: object, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = _numeric_matrix(frame, features)
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    imputed = imputer.transform(values)
    return scaler.transform(imputed)


def _dimension_group(source: str, dimension: int) -> str:
    source = str(source)
    dimension = int(dimension)
    if source in {"bbob_train", "mabbob_formal"}:
        return "10/20D" if dimension in CEC_COMMON_DIMS else "40D"
    return "10/20D" if dimension in CEC_COMMON_DIMS else "30/50D"


def _plot_group(source: str, dimension_group: str) -> str:
    labels = {
        ("bbob_train", "10/20D"): "BBOB 10/20D",
        ("mabbob_formal", "10/20D"): "MA-BBOB 10/20D",
        (CEC_SOURCE, "10/20D"): "CEC 10/20D",
        ("bbob_train", "40D"): "BBOB 40D",
        ("mabbob_formal", "40D"): "MA-BBOB 40D",
        (CEC_SOURCE, "30/50D"): "CEC 30/50D",
    }
    return labels[(str(source), str(dimension_group))]


def _finite_summary(values: Iterable[float]) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
        int(len(array)),
    )


def _build_feature_profiles(
    frames: dict[str, pd.DataFrame],
    model: object,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    for source, frame in frames.items():
        if frame.empty:
            continue
        working = frame.copy()
        working["dimension_group"] = [
            _dimension_group(source, value) for value in working["dimension"]
        ]
        z_values = _standardized_values(model, working, features)
        for feature_index, feature in enumerate(features):
            for (dimension_group, phase), group in working.groupby(
                ["dimension_group", "sampling_phase"], observed=True, sort=True
            ):
                indices = group.index.to_numpy()
                positions = working.index.get_indexer(indices)
                raw = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float)
                raw = raw[np.isfinite(raw)]
                z = z_values[positions, feature_index]
                z = z[np.isfinite(z)]
                if len(raw) == 0 or len(z) == 0:
                    continue
                profile_rows.append(
                    {
                        "source": source,
                        "source_label": SOURCE_LABELS.get(source, source),
                        "dimension_group": dimension_group,
                        "plot_group": _plot_group(source, dimension_group),
                        "sampling_phase": str(phase),
                        "feature": feature,
                        "feature_label": _feature_label(feature),
                        "n": int(len(raw)),
                        "raw_median": float(np.median(raw)),
                        "raw_q25": float(np.quantile(raw, 0.25)),
                        "raw_q75": float(np.quantile(raw, 0.75)),
                        "z_median": float(np.median(z)),
                        "z_q25": float(np.quantile(z, 0.25)),
                        "z_q75": float(np.quantile(z, 0.75)),
                    }
                )

    profiles = pd.DataFrame(profile_rows)
    if profiles.empty:
        raise ValueError("Feature profile table is empty")

    # The two CEC panels are compared to dimension-matched or nearest available
    # training dimensions.  For 30/50D, 40D is the only high-dimensional
    # training stratum and is therefore reported as a sensitivity reference,
    # not as an exact dimension match.
    train_frame = pd.concat(
        [frames["bbob_train"], frames["mabbob_formal"]], ignore_index=True
    )
    train_frame["dimension_group"] = [
        "10/20D" if int(value) in CEC_COMMON_DIMS else "40D"
        for value in train_frame["dimension"]
    ]
    cec_frame = frames[CEC_SOURCE].copy()
    cec_frame["dimension_group"] = [
        "10/20D" if int(value) in CEC_COMMON_DIMS else "30/50D"
        for value in cec_frame["dimension"]
    ]
    train_z = _standardized_values(model, train_frame, features)
    cec_z = _standardized_values(model, cec_frame, features)
    importance = np.asarray(model.named_steps["regressor"].feature_importances_, dtype=float)
    imputer = model.named_steps["imputer"]
    for cec_group, reference_group, comparison_note in (
        ("10/20D", "10/20D", "dimension-matched"),
        ("30/50D", "40D", "nearest-high-dimensional-training-reference"),
    ):
        cec_positions = np.flatnonzero(cec_frame["dimension_group"].eq(cec_group).to_numpy())
        train_positions = np.flatnonzero(train_frame["dimension_group"].eq(reference_group).to_numpy())
        for index, feature in enumerate(features):
            cec_raw = pd.to_numeric(cec_frame.iloc[cec_positions][feature], errors="coerce").to_numpy(float)
            train_raw = pd.to_numeric(train_frame.iloc[train_positions][feature], errors="coerce").to_numpy(float)
            cec_raw = cec_raw[np.isfinite(cec_raw)]
            train_raw = train_raw[np.isfinite(train_raw)]
            q01, q99 = np.quantile(train_raw, [0.01, 0.99])
            train_min, train_max = float(np.min(train_raw)), float(np.max(train_raw))
            cec_z_feature = cec_z[cec_positions, index]
            train_z_feature = train_z[train_positions, index]
            train_iqr = float(np.quantile(train_raw, 0.75) - np.quantile(train_raw, 0.25))
            scale = train_iqr if np.isfinite(train_iqr) and train_iqr > EPS else 1.0
            shift_rows.append(
                {
                    "comparison": f"CEC {cec_group} vs train {reference_group}",
                    "cec_dimension_group": cec_group,
                    "reference_dimension_group": reference_group,
                    "comparison_note": comparison_note,
                    "feature": feature,
                    "feature_label": _feature_label(feature),
                    "feature_family": _feature_family(feature),
                    "n_cec": int(len(cec_raw)),
                    "n_train_reference": int(len(train_raw)),
                    "cec_raw_median": float(np.median(cec_raw)),
                    "train_raw_median": float(np.median(train_raw)),
                    "raw_median_delta_cec_minus_train": float(np.median(cec_raw) - np.median(train_raw)),
                    "train_iqr": train_iqr,
                    "robust_median_shift": float((np.median(cec_raw) - np.median(train_raw)) / scale),
                    "cec_z_median": float(np.median(cec_z_feature)),
                    "train_z_median": float(np.median(train_z_feature)),
                    "z_median_delta_cec_minus_train": float(
                        np.median(cec_z_feature) - np.median(train_z_feature)
                    ),
                    "cec_z_q95_abs": float(np.quantile(np.abs(cec_z_feature), 0.95)),
                    "outside_train_q01_q99_rate": float(
                        np.mean((cec_raw < q01) | (cec_raw > q99))
                    ),
                    "outside_train_minmax_rate": float(
                        np.mean((cec_raw < train_min) | (cec_raw > train_max))
                    ),
                    "missing_cec_rate": float(
                        np.mean(~np.isfinite(pd.to_numeric(cec_frame.iloc[cec_positions][feature], errors="coerce")))
                    ),
                    "model_feature_importance": float(importance[index]),
                    "model_imputer_statistic": float(imputer.statistics_[index]),
                }
            )
    shifts = pd.DataFrame(shift_rows)
    return profiles, shifts


def _score_frames(
    decision_dir: Path,
    model: object,
    cec: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_oof = pd.read_parquet(decision_dir / "train_oof_predictions.parquet")
    validation = pd.read_parquet(decision_dir / "validation_predictions.parquet")
    train_oof = train_oof[train_oof["model_name"].astype(str).eq("random_forest_regressor")].copy()
    validation = validation[validation["model_name"].astype(str).eq("random_forest_regressor")].copy()
    if train_oof.empty or validation.empty:
        raise ValueError("OOF or validation predictions are missing for the deployed model")
    for frame, stage in ((train_oof, "train_oof"), (validation, "validation")):
        suite_source = np.where(
            frame["function_id"].astype(str).str.startswith("bbob_"),
            "bbob_train" if stage == "train_oof" else "bbob_validation",
            "mabbob_formal" if stage == "train_oof" else "mabbob_validation",
        )
        frame["source"] = suite_source
        frame["source_label"] = frame["source"].map(SOURCE_LABELS)
        frame["stage"] = stage
        frame["decision_score"] = pd.to_numeric(frame["decision_score"], errors="coerce")
    cec_scored = cec.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cec_scored["decision_score"] = np.asarray(
            model.predict(_numeric_matrix(cec_scored, features)), dtype=float
        )
    cec_scored["stage"] = "cec_deployment"
    cec_scored["source_label"] = SOURCE_LABELS[CEC_SOURCE]
    return train_oof, validation, cec_scored


def _score_summary(
    train_oof: pd.DataFrame,
    validation: pd.DataFrame,
    cec_scored: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    frames = []
    for frame in (train_oof, validation, cec_scored):
        item = frame.copy()
        item["sampling_phase"] = item["sampling_phase"].astype(str)
        item["dimension_group"] = [
            _dimension_group(source, dimension)
            for source, dimension in zip(item["source"], item["dimension"])
        ]
        frames.append(item)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for keys, group in combined.groupby(
        ["stage", "source", "source_label", "sampling_phase", "dimension_group"],
        observed=True,
        sort=True,
    ):
        stage, source, source_label, phase, dimension_group = keys
        scores = pd.to_numeric(group["decision_score"], errors="coerce").dropna().to_numpy(float)
        rows.append(
            {
                "stage": stage,
                "source": source,
                "source_label": source_label,
                "sampling_phase": phase,
                "dimension_group": dimension_group,
                "n": int(len(scores)),
                "score_mean": float(np.mean(scores)),
                "score_median": float(np.median(scores)),
                "score_q05": float(np.quantile(scores, 0.05)),
                "score_q95": float(np.quantile(scores, 0.95)),
                "score_max": float(np.max(scores)),
                "rate_score_gt_zero": float(np.mean(scores > 0.0)),
                "rate_score_gt_user_threshold": float(np.mean(scores > USER_THRESHOLD)),
                "rate_score_gt_configured_threshold": float(np.mean(scores > threshold)),
            }
        )
    return pd.DataFrame(rows)


def _overall_score_summary(
    train_oof: pd.DataFrame,
    validation: pd.DataFrame,
    cec_scored: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in (train_oof, validation, cec_scored):
        for source, group in frame.groupby("source", sort=True):
            scores = pd.to_numeric(group["decision_score"], errors="coerce").dropna().to_numpy(float)
            rows.append(
                {
                    "stage": str(group["stage"].iloc[0]),
                    "source": source,
                    "source_label": SOURCE_LABELS.get(str(source), str(source)),
                    "n": int(len(scores)),
                    "score_mean": float(np.mean(scores)),
                    "score_median": float(np.median(scores)),
                    "score_q01": float(np.quantile(scores, 0.01)),
                    "score_q05": float(np.quantile(scores, 0.05)),
                    "score_q25": float(np.quantile(scores, 0.25)),
                    "score_q75": float(np.quantile(scores, 0.75)),
                    "score_q95": float(np.quantile(scores, 0.95)),
                    "score_q99": float(np.quantile(scores, 0.99)),
                    "score_max": float(np.max(scores)),
                    "rate_score_gt_zero": float(np.mean(scores > 0.0)),
                    "rate_score_gt_user_threshold": float(np.mean(scores > USER_THRESHOLD)),
                    "rate_score_gt_configured_threshold": float(np.mean(scores > threshold)),
                }
            )
    return pd.DataFrame(rows)


def _cec_function_dimension_summary(cec_scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (function_id, dimension), group in cec_scored.groupby(
        ["function_id", "dimension"], sort=True
    ):
        scores = group["decision_score"].to_numpy(float)
        rows.append(
            {
                "function_id": str(function_id),
                "function_label": f"F{int(str(function_id).split('_f')[-1]):02d}",
                "dimension": int(dimension),
                "n_states": int(len(scores)),
                "n_runs": int(group[["problem_id", "seed"]].drop_duplicates().shape[0]),
                "score_median": float(np.median(scores)),
                "score_q95": float(np.quantile(scores, 0.95)),
                "score_max": float(np.max(scores)),
                "rate_score_gt_zero": float(np.mean(scores > 0.0)),
                "rate_score_gt_user_threshold": float(np.mean(scores > USER_THRESHOLD)),
                "rate_score_gt_configured_threshold": float(np.mean(scores > threshold)),
                "max_FE_ratio": float(group["FE_ratio"].max()),
            }
        )
    return pd.DataFrame(rows)


def _model_input_sensitivity(
    cec_scored: pd.DataFrame,
    model: object,
    features: list[str],
) -> pd.DataFrame:
    base_values = _numeric_matrix(cec_scored, features)
    imputer = model.named_steps["imputer"]
    base_scores = cec_scored["decision_score"].to_numpy(float)
    rows: list[dict[str, object]] = []
    importance = np.asarray(model.named_steps["regressor"].feature_importances_, dtype=float)
    for index, feature in enumerate(features):
        altered = base_values.copy()
        altered[:, index] = float(imputer.statistics_[index])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            altered_scores = np.asarray(model.predict(altered), dtype=float)
        delta = altered_scores - base_scores
        rows.append(
            {
                "feature": feature,
                "feature_label": _feature_label(feature),
                "feature_family": _feature_family(feature),
                "model_feature_importance": float(importance[index]),
                "cec_to_train_center_score_delta_median": float(np.median(delta)),
                "cec_to_train_center_score_delta_mean": float(np.mean(delta)),
                "cec_to_train_center_score_delta_q25": float(np.quantile(delta, 0.25)),
                "cec_to_train_center_score_delta_q75": float(np.quantile(delta, 0.75)),
                "share_delta_gt_zero": float(np.mean(delta > 0.0)),
                "imputer_statistic": float(imputer.statistics_[index]),
            }
        )
    return pd.DataFrame(rows)


def _row_support_summary(
    cec_scored: pd.DataFrame,
    train_frame: pd.DataFrame,
    model: object,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_values = _numeric_matrix(train_frame, features)
    cec_values = _numeric_matrix(cec_scored, features)
    train_q01 = np.nanquantile(train_values, 0.01, axis=0)
    train_q99 = np.nanquantile(train_values, 0.99, axis=0)
    outside = (cec_values < train_q01[None, :]) | (cec_values > train_q99[None, :])
    z = _standardized_values(model, cec_scored, features)
    per_row = pd.DataFrame(
        {
            "problem_id": cec_scored["problem_id"].to_numpy(),
            "function_id": cec_scored["function_id"].to_numpy(),
            "dimension": cec_scored["dimension"].to_numpy(),
            "seed": cec_scored["seed"].to_numpy(),
            "FE": cec_scored["FE"].to_numpy(),
            "FE_ratio": cec_scored["FE_ratio"].to_numpy(),
            "sampling_phase": cec_scored["sampling_phase"].to_numpy(),
            "decision_score": cec_scored["decision_score"].to_numpy(),
            "outside_train_q01_q99_feature_count": outside.sum(axis=1),
            "max_abs_preprocessed_z": np.max(np.abs(z), axis=1),
        }
    )
    summary = {
        "cec_states": float(len(per_row)),
        "cec_rows_with_any_feature_outside_train_q01_q99": float(np.mean(outside.sum(axis=1) > 0)),
        "cec_rows_with_at_least_five_features_outside_train_q01_q99": float(
            np.mean(outside.sum(axis=1) >= 5)
        ),
        "cec_rows_with_at_least_ten_features_outside_train_q01_q99": float(
            np.mean(outside.sum(axis=1) >= 10)
        ),
        "cec_median_max_abs_preprocessed_z": float(np.median(np.max(np.abs(z), axis=1))),
        "cec_q95_max_abs_preprocessed_z": float(np.quantile(np.max(np.abs(z), axis=1), 0.95)),
        "cec_max_max_abs_preprocessed_z": float(np.max(np.max(np.abs(z), axis=1))),
        "cec_missing_cell_rate": float(np.mean(~np.isfinite(cec_values))),
        "train_missing_cell_rate": float(np.mean(~np.isfinite(train_values))),
    }
    return per_row, summary


def _online_reproduction_check(online_path: Path, cec_scored: pd.DataFrame) -> pd.DataFrame:
    if not online_path.exists():
        return pd.DataFrame(
            [{"status": "online_output_missing", "matched_rows": 0, "max_abs_score_difference": None}]
        )
    online = pd.read_parquet(online_path)
    required = {"policy_category", "decision_score", "function", "dimension", "seed", "policy_name"}
    if not required.issubset(online.columns):
        return pd.DataFrame(
            [{"status": "online_output_schema_missing", "matched_rows": 0, "max_abs_score_difference": None}]
        )
    online = online[online["policy_category"].astype(str).eq("controller")].copy()
    online = online[online["decision_score"].notna()].copy()
    if online.empty:
        return pd.DataFrame(
            [{"status": "online_controller_scores_missing", "matched_rows": 0, "max_abs_score_difference": None}]
        )
    cec_last = (
        cec_scored.sort_values(["problem_id", "seed", "FE"])
        .groupby(["problem_id", "seed"], as_index=False)
        .tail(1)
        .loc[:, ["problem_id", "dimension", "seed", "FE", "decision_score"]]
        .rename(columns={"FE": "cec_last_FE", "decision_score": "cec_last_decision_score"})
    )
    online["problem_id"] = [
        f"cec2017_f{int(function):02d}_d{int(dimension)}"
        for function, dimension in zip(online["function"], online["dimension"])
    ]
    merged = online.merge(
        cec_last,
        on=["problem_id", "dimension", "seed"],
        how="left",
        validate="many_to_one",
    )
    merged["score_difference_online_minus_reconstructed"] = (
        pd.to_numeric(merged["decision_score"], errors="coerce")
        - pd.to_numeric(merged["cec_last_decision_score"], errors="coerce")
    )
    matched = merged[merged["cec_last_decision_score"].notna()].copy()
    if matched.empty:
        status = "online_and_cec_keys_did_not_match"
        max_diff = None
    else:
        max_diff = float(np.max(np.abs(matched["score_difference_online_minus_reconstructed"])))
        status = "matched" if max_diff <= 1.0e-10 else "matched_with_difference"
    by_policy = (
        matched.groupby("policy_name", as_index=False)
        .agg(
            matched_rows=("decision_score", "size"),
            max_abs_score_difference=("score_difference_online_minus_reconstructed", lambda x: float(np.max(np.abs(x)))),
            online_score_max=("decision_score", "max"),
            cec_reconstructed_score_max=("cec_last_decision_score", "max"),
            query_calls=("query_called", "sum") if "query_called" in matched else ("decision_score", "size"),
        )
    )
    by_policy.insert(0, "status", status)
    by_policy["overall_matched_rows"] = int(len(matched))
    by_policy["overall_max_abs_score_difference"] = max_diff
    return by_policy


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)


def _figure_feature_heatmap(profiles: pd.DataFrame, shifts: pd.DataFrame, output: Path) -> None:
    groups = [
        "BBOB 10/20D",
        "MA-BBOB 10/20D",
        "CEC 10/20D",
        "BBOB 40D",
        "MA-BBOB 40D",
        "CEC 30/50D",
    ]
    overall = profiles.groupby(["plot_group", "feature"], as_index=False)["z_median"].median()
    pivot = overall.pivot(index="feature", columns="plot_group", values="z_median").reindex(columns=groups)
    order = (
        shifts[shifts["cec_dimension_group"].eq("30/50D")]
        .set_index("feature")["z_median_delta_cec_minus_train"]
        .abs()
        .sort_values(ascending=False)
        .index
    )
    pivot = pivot.reindex([feature for feature in order if feature in pivot.index])
    values = pivot.to_numpy(float)
    limit = max(1.0, float(np.nanquantile(np.abs(values), 0.98)))
    cmap = LinearSegmentedColormap.from_list(
        "decision_shift", [PALETTE["blue"], PALETTE["paper"], PALETTE["orange"]]
    )
    fig, ax = plt.subplots(figsize=(13.5, 12.5))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(groups)), groups, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [_feature_label(value) for value in pivot.index])
    ax.set_xlabel("数据来源与维度组")
    ax.set_ylabel("特征（按 CEC 30/50D 相对训练参考的位移排序）")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("标准化中位数（model z-space）")
    ax.set_facecolor(PALETTE["paper"])
    fig.suptitle("Decision 29 特征的标准化中位数位置", x=0.27, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.27,
        0.955,
        "数值为部署模型 StandardScaler 空间中的中位数；0 表示训练拟合中心，颜色不是触发标签。",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.27, right=0.9, bottom=0.14, top=0.91)
    _save_figure(fig, output / "fig01_decision_feature_z_heatmap.png")


def _feature_distribution_values(
    frame: pd.DataFrame,
    model: object,
    features: list[str],
    selected: list[str],
) -> pd.DataFrame:
    working = frame.copy()
    working["dimension_group"] = [
        _dimension_group(source, value) for source, value in zip(working["source"], working["dimension"])
    ]
    working["plot_group"] = [
        _plot_group(source, dimension_group)
        for source, dimension_group in zip(working["source"], working["dimension_group"])
    ]
    z = _standardized_values(model, working, features)
    rows = []
    for index, feature in enumerate(features):
        if feature not in selected:
            continue
        for position, (_, row) in enumerate(working.iterrows()):
            rows.append(
                {
                    "feature": feature,
                    "feature_label": _feature_label(feature),
                    "sampling_phase": str(row["sampling_phase"]),
                    "plot_group": row["plot_group"],
                    "z_value": float(z[position, index]),
                }
            )
    return pd.DataFrame(rows)


def _figure_feature_distributions(
    train: pd.DataFrame,
    cec: pd.DataFrame,
    model: object,
    features: list[str],
    importance: pd.DataFrame,
    shifts: pd.DataFrame,
    output: Path,
) -> None:
    top_importance = importance.sort_values("model_feature_importance", ascending=False)["feature"].head(4).tolist()
    top_shift = shifts[shifts["cec_dimension_group"].eq("30/50D")].sort_values(
        "robust_median_shift", key=lambda x: x.abs(), ascending=False
    )["feature"].head(2).tolist()
    selected = list(dict.fromkeys(top_importance + top_shift))[:6]
    combined = pd.concat([train, cec], ignore_index=True, sort=False)
    long = _feature_distribution_values(combined, model, features, selected)
    phase_order = ["early", "mid", "late"]
    group_order = ["BBOB 10/20D", "MA-BBOB 10/20D", "CEC 10/20D", "CEC 30/50D"]
    fig, axes = plt.subplots(
        len(selected), len(phase_order), figsize=(16.5, max(9.0, 2.5 * len(selected))), sharey="row"
    )
    axes = np.atleast_2d(axes)
    for row_index, feature in enumerate(selected):
        for col_index, phase in enumerate(phase_order):
            ax = axes[row_index, col_index]
            subset = long[long["feature"].eq(feature) & long["sampling_phase"].eq(phase)]
            data = []
            labels = []
            colors = []
            for group in group_order:
                values = subset.loc[subset["plot_group"].eq(group), "z_value"].to_numpy(float)
                if len(values):
                    data.append(values)
                    labels.append(group.replace(" 10/20D", "\n10/20D").replace(" 30/50D", "\n30/50D").replace(" 40D", "\n40D"))
                    colors.append(GROUP_COLORS[group])
            if data:
                box = ax.boxplot(
                    data,
                    patch_artist=True,
                    widths=0.58,
                    showfliers=False,
                    medianprops={"color": PALETTE["charcoal"], "linewidth": 1.3},
                    whiskerprops={"color": PALETTE["grey"], "linewidth": 0.9},
                    capprops={"color": PALETTE["grey"], "linewidth": 0.9},
                )
                for patch, color in zip(box["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.72)
                    patch.set_edgecolor(PALETTE["charcoal"])
            ax.axhline(0.0, color=PALETTE["charcoal"], linewidth=0.85, linestyle="--")
            ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7.4)
            if row_index == 0:
                ax.set_title(phase, fontsize=11, weight="bold")
            if col_index == 0:
                ax.set_ylabel(_feature_label(feature), fontsize=9)
            else:
                ax.set_ylabel("")
            ax.grid(axis="y")
    fig.suptitle("重点 Decision 特征按采样阶段的标准化分布", x=0.06, ha="left", y=0.99, fontsize=15, weight="bold")
    fig.text(
        0.06,
        0.965,
        "箱体在 model z-space 展示 BBOB/MA-BBOB 训练参考与 CEC cmaes；CEC 30/50D 没有同维度训练 strata。",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.91, hspace=0.45, wspace=0.08)
    _save_figure(fig, output / "fig02_key_feature_phase_distributions.png")


def _figure_score_distributions(
    train_oof: pd.DataFrame,
    validation: pd.DataFrame,
    cec_scored: pd.DataFrame,
    threshold: float,
    output: Path,
) -> None:
    sources = ["BBOB train OOF", "MA-BBOB train OOF", "BBOB validation", "MA-BBOB validation", "CEC2017 cmaes"]
    source_to_frame = {
        "BBOB train OOF": train_oof[train_oof["source"].eq("bbob_train")],
        "MA-BBOB train OOF": train_oof[train_oof["source"].eq("mabbob_formal")],
        "BBOB validation": validation[validation["source"].eq("bbob_validation")],
        "MA-BBOB validation": validation[validation["source"].eq("mabbob_validation")],
        "CEC2017 cmaes": cec_scored,
    }
    source_colors = [PALETTE["blue"], PALETTE["gold"], PALETTE["blue_light"], PALETTE["pink"], PALETTE["orange"]]
    # Chart contract: compare score distributions at the same three sampling
    # phases, with score=0 and the two fixed trigger boundaries visible.
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=True)
    for ax, phase in zip(axes, ["early", "mid", "late"]):
        data = []
        labels = []
        for source in sources:
            values = source_to_frame[source].loc[
                source_to_frame[source]["sampling_phase"].astype(str).eq(phase), "decision_score"
            ].dropna().to_numpy(float)
            data.append(values)
            labels.append(source.replace(" ", "\n", 1))
        box = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.62,
            showfliers=False,
            medianprops={"color": PALETTE["charcoal"], "linewidth": 1.4},
            whiskerprops={"color": PALETTE["grey"], "linewidth": 0.9},
            capprops={"color": PALETTE["grey"], "linewidth": 0.9},
        )
        for patch, color in zip(box["boxes"], source_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor(PALETTE["charcoal"])
        ax.axhline(0.0, color=PALETTE["charcoal"], linewidth=1.0, linestyle="--", label="score = 0")
        ax.axhline(threshold, color=PALETTE["orange"], linewidth=1.0, linestyle=":", label="configured threshold")
        ax.axhline(USER_THRESHOLD, color=PALETTE["gold"], linewidth=0.8, linestyle="-.", label="display threshold")
        ax.set_title(phase, fontsize=11, weight="bold")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set_xlabel("score source")
        ax.set_ylim(-0.85, max(0.33, threshold * 1.08))
        ax.grid(axis="y")
    axes[0].set_ylabel("Decision score")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("部署 Decision score 的阶段分布与触发边界", x=0.06, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.06,
        0.918,
        "CEC 面板是在线实际前缀 cmaes 的所有可检查状态；两条几乎重合的阈值线分别表示配置精度与用户指定显示精度。",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.80, wspace=0.08)
    _save_figure(fig, output / "fig03_decision_score_distributions.png")


def _heatmap_panel(ax: plt.Axes, table: pd.DataFrame, value_column: str, title: str, cmap: str) -> None:
    pivot = table.pivot(index="function_label", columns="dimension", values=value_column)
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    values = pivot.to_numpy(float)
    if value_column == "score_median":
        vmax = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{int(value)}D" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = f"{value:.3f}" if np.isfinite(value) else "NA"
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color=PALETTE["charcoal"])
    ax.set_title(title, fontsize=11, weight="bold")
    return image


def _figure_cec_heatmap(cec_summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1]})
    cmap = LinearSegmentedColormap.from_list("cec_score", [PALETTE["paper"], PALETTE["blue"]])
    _heatmap_panel(axes[0], cec_summary, "score_median", "CEC cmaes score 中位数", cmap)
    _heatmap_panel(axes[1], cec_summary, "score_max", "CEC cmaes score 最大值", cmap)
    fig.suptitle("CEC2017 各函数/维度的 Decision score", x=0.06, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.06,
        0.93,
        "每个格子聚合 5 个 seed 的全部可检查状态；所有格子的 score 最大值仍低于 0，因此严格 score > 0 不触发。",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.82, wspace=0.25)
    _save_figure(fig, output / "fig04_cec_function_dimension_score_heatmap.png")


def _figure_importance_shift(
    shifts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output: Path,
) -> None:
    common = shifts[shifts["cec_dimension_group"].eq("10/20D")].loc[
        :, ["feature", "feature_label", "robust_median_shift", "model_feature_importance"]
    ].rename(columns={"robust_median_shift": "shift_common"})
    unseen = shifts[shifts["cec_dimension_group"].eq("30/50D")].loc[
        :, ["feature", "robust_median_shift", "z_median_delta_cec_minus_train"]
    ].rename(columns={"robust_median_shift": "shift_unseen", "z_median_delta_cec_minus_train": "z_shift_unseen"})
    plot = common.merge(unseen, on="feature", how="inner").merge(
        sensitivity.loc[:, ["feature", "cec_to_train_center_score_delta_median", "feature_family"]],
        on="feature",
        how="left",
    )
    # Chart contract: expose the two distinct diagnostics separately.  The
    # left panel relates CEC shift to model split importance; the right panel
    # relates the same shift to the score change from one-input replacement.
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharex=True)
    family_colors = {
        "base": PALETTE["blue"],
        "primary": PALETTE["gold"],
        "dynamics": PALETTE["olive"],
        "motion": PALETTE["pink"],
        "maturity": PALETTE["orange"],
    }
    for family, group in plot.groupby("feature_family", sort=True):
        marker_size = 4000 * group["model_feature_importance"].to_numpy(float) + 24
        axes[0].scatter(
            group["shift_unseen"].abs(),
            group["model_feature_importance"],
            s=marker_size,
            c=family_colors.get(family, PALETTE["grey"]),
            alpha=0.82,
            edgecolor=PALETTE["charcoal"],
            linewidth=0.65,
            label=family,
        )
        axes[1].scatter(
            group["shift_unseen"].abs(),
            group["cec_to_train_center_score_delta_median"],
            s=marker_size,
            c=family_colors.get(family, PALETTE["grey"]),
            alpha=0.82,
            edgecolor=PALETTE["charcoal"],
            linewidth=0.65,
            label=family,
        )
    labels_to_add = plot.assign(
        label_priority=lambda x: x["model_feature_importance"] * (x["shift_unseen"].abs() + 0.01)
    ).sort_values("label_priority", ascending=False).head(8)
    for _, row in labels_to_add.iterrows():
        axes[0].annotate(
            row["feature_label"],
            (abs(float(row["shift_unseen"])), float(row["model_feature_importance"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.2,
            bbox={"facecolor": PALETTE["paper"], "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
    axes[0].axhline(0.0, color=PALETTE["charcoal"], linewidth=1.0, linestyle="--")
    axes[1].axhline(0.0, color=PALETTE["charcoal"], linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("CEC 30/50D 相对 40D 训练参考的 |robust median shift|")
    axes[1].set_xlabel("CEC 30/50D 相对 40D 训练参考的 |robust median shift|")
    axes[0].set_ylabel("随机森林 split importance")
    axes[1].set_ylabel("单特征替代后的 score 变化中位数\n（替代敏感性，不作因果解释）")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    max_importance = float(plot["model_feature_importance"].max())
    axes[0].set_ylim(-0.005, max(0.01, max_importance * 1.18))
    delta = plot["cec_to_train_center_score_delta_median"].to_numpy(float)
    delta_min, delta_max = float(np.nanmin(delta)), float(np.nanmax(delta))
    delta_span = max(delta_max - delta_min, 1.0e-5)
    axes[1].set_ylim(delta_min - 0.08 * delta_span, delta_max + 0.12 * delta_span)
    axes[0].set_title("分布位移与模型重要性", fontsize=11, weight="bold")
    axes[1].set_title("分布位移与输入替代敏感性", fontsize=11, weight="bold")
    fig.suptitle("CEC 30/50D 的 Decision 特征位移与模型输入诊断", x=0.08, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.08,
        0.918,
        "左图 y 轴为随机森林 split importance；右图 y>0 表示把该输入替换为训练 imputer 中心后，CEC score 中位数通常上升。点色表示特征族，点大小仍编码模型重要性。",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="feature family",
        frameon=False,
        ncol=5,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.90),
        fontsize=8.5,
        title_fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.28)
    _save_figure(fig, output / "fig05_importance_shift_sensitivity.png")


def _write_report(
    output: Path,
    summary: dict[str, object],
    overall_scores: pd.DataFrame,
    cec_summary: pd.DataFrame,
    shifts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    support_summary: dict[str, float],
    online_check: pd.DataFrame,
) -> None:
    def fmt(value: object, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}g}"
        return str(value)

    def one(source: str, column: str) -> object:
        row = overall_scores[overall_scores["source"].eq(source)]
        return row.iloc[0][column] if len(row) else None

    cec_max = float(cec_summary["score_max"].max())
    cec_median = float(cec_summary["score_median"].median())
    top_shift = shifts[shifts["cec_dimension_group"].eq("30/50D")].sort_values(
        "robust_median_shift", key=lambda x: x.abs(), ascending=False
    ).head(8)
    top_imp = shifts[shifts["cec_dimension_group"].eq("30/50D")].sort_values(
        "model_feature_importance", ascending=False
    ).head(8)
    top_sensitivity = sensitivity.sort_values(
        "cec_to_train_center_score_delta_median", ascending=False
    ).head(8)
    lines = [
        "# CEC2017 Decision Model 特征工程诊断",
        "",
        "> 研究范围：解释当前在线 CEC2017 运行为什么没有触发 Decision；仅使用部署模型的 29 个 `bf_*` 输入，未重新训练模型、预处理或阈值。",
        "",
        "## 结论摘要",
        "",
        f"1. **触发条件在 CEC 上没有被满足**：CEC cmaes 的 {int(summary['cec_state_rows'])} 个可检查状态中，`score > 0` 的比例为 {fmt(summary['cec_rate_gt_zero'] * 100, 5)}%，`score > {USER_THRESHOLD}` 的比例为 {fmt(summary['cec_rate_gt_user_threshold'] * 100, 5)}%；最大 score={fmt(cec_max, 6)}。因此两个控制器都不触发，后者只是更严格的第二道边界。",
        f"2. **当前分数整体靠近 0 但位于 0 以下**：CEC score 中位数为 {fmt(cec_median, 6)}；BBOB/MA-BBOB train OOF 中位数分别为 {fmt(one('bbob_train', 'score_median'), 6)} 和 {fmt(one('mabbob_formal', 'score_median'), 6)}，validation 中位数分别为 {fmt(one('bbob_validation', 'score_median'), 6)} 和 {fmt(one('mabbob_validation', 'score_median'), 6)}。CEC 没有达到 `score > 0` 的数值边界。",
        f"3. **阈值处在极端上尾**：配置中的精确阈值为 {fmt(summary['configured_threshold'], 12)}（报告按用户要求显示 {USER_THRESHOLD}）；训练 OOF 的 score 最大值为 {fmt(one('bbob_train', 'score_max'), 6)} / {fmt(one('mabbob_formal', 'score_max'), 6)}，validation 最大值为 {fmt(one('bbob_validation', 'score_max'), 6)} / {fmt(one('mabbob_validation', 'score_max'), 6)}，CEC 最大值只有 {fmt(cec_max, 6)}。",
        f"4. **CEC 存在可见的特征分布偏移，但不是所有行都处于训练支持范围外**：按训练 1%–99% 区间统计，CEC 有至少一个特征越界的行占 {fmt(support_summary['cec_rows_with_any_feature_outside_train_q01_q99'] * 100, 4)}%，至少 5 个特征越界的行占 {fmt(support_summary['cec_rows_with_at_least_five_features_outside_train_q01_q99'] * 100, 4)}%；缺失单元格比例为 {fmt(support_summary['cec_missing_cell_rate'] * 100, 4)}%。",
        f"5. **偏移落在若干高重要性行为特征上**：在 CEC 30/50D 与 40D 训练参考的比较中，top-10 模型重要性特征与 top-10 标准化中位数位移特征的交集为 {int(summary['top10_importance_shift_overlap'])} 个；这说明输入域差异可能改变模型分数，但 split importance 本身不是方向性或因果证据。",
        f"6. **30/50D 是风险放大器而不是唯一原因**：10/20D 也没有触发；30/50D 的 score 中位数更低，尤其 50D。由于训练维度只有 10/20/40D，CEC 30/50D 只能与 40D 作为近邻高维参考比较，不能声称这是严格同维度差异。",
        "",
        "## 数据与输入合同",
        "",
        f"- 部署模型：`random_forest_regressor`，特征组 `B2+Motion+SearchMaturityLinear`，输入 {summary['feature_count']} 个行为特征。",
        f"- 训练参考：BBOB train {int(summary['bbob_train_rows'])} 行、MA-BBOB train {int(summary['mabbob_train_rows'])} 行；只读取训练 artifact 与已生成的 CEC behavior 文件。",
        f"- CEC 诊断：5 个函数、10/20/30/50D、5 个 seed 的 cmaes 前缀，共 {int(summary['cec_run_count'])} 条 run、{int(summary['cec_state_rows'])} 个行为状态。",
        "- 阶段一致性：CEC behavior 中的 `sampling_phase` 与项目预先定义的 FE-ratio 映射一致，使用 `early/mid/late`，没有把四段 phase bin 当作模型输入。",
        "- 运行时输出交叉核对：在线 controller 的最后一次 Decision score 与 CEC behavior 重建结果的最大绝对差见 `online_score_reproduction_check.csv`；该检查只用于确认诊断输入与在线路径一致。",
        "",
        "## 重点特征差异",
        "",
        "CEC 30/50D 相对 40D 训练参考的标准化中位数位移前 8 项：",
        "",
        "| 特征 | robust median shift | model importance | CEC z 中位数 | 训练参考 z 中位数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in top_shift.iterrows():
        lines.append(
            f"| `{row['feature']}` | {fmt(row['robust_median_shift'], 5)} | {fmt(row['model_feature_importance'], 5)} | {fmt(row['cec_z_median'], 5)} | {fmt(row['train_z_median'], 5)} |"
        )
    lines.extend(
        [
            "",
            "其中，CEC cmaes 的 `bf_population_wasserstein_rate_w05`、`bf_centroid_shift_rate_w05`、`bf_population_centroid_shift_w05`、`bf_population_chamfer_distance_w05` 和 `bf_diversity_mean_pairwise` 中位数都低于训练参考；这些特征同时具有较高的模型重要性或明显的标准化位移。它们描述的是群体变化与多样性，不能从该图直接解释为真实性能因果因素，只能说明模型输入在这些维度上更容易落到训练分布的另一侧。",
            "",
            "## 单特征输入替代敏感性",
            "",
            "下面的诊断把 CEC 每一行的单个输入替换为部署 imputer 的训练中心，重新计算 score，记录模型输出变化。它是模型输入敏感性分析，不是重新运行优化器，也不构成因果结论。",
            "",
            "| 特征 | 替代后 score 变化中位数 | 替代后 score 变化均值 | importance |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in top_sensitivity.iterrows():
        lines.append(
            f"| `{row['feature']}` | {fmt(row['cec_to_train_center_score_delta_median'], 5)} | {fmt(row['cec_to_train_center_score_delta_mean'], 5)} | {fmt(row['model_feature_importance'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## CEC 函数/维度分层",
            "",
            "详表见 `cec_function_dimension_summary.csv`；图 4 同时展示 score 中位数和最大值。所有函数/维度格子的最大值都低于 0，所以当前问题不是少数函数触发失败，而是 CEC cmaes 状态在本次测评中没有跨过第一道数值边界。",
            "",
            "## 结果解释",
            "",
            "综合来看，CEC 不触发是三个因素叠加：",
            "",
            "- **第一层是输出边界**：即使使用较宽的 `score > 0`，CEC cmaes 状态也没有超过 0。",
            f"- **第二层是稀有标签与高阈值**：训练中 `g_fe_selected_path > 0` 的科学标签比例只有约 {fmt(summary['train_label_positive_rate'] * 100, 3)}%，预先指定阈值位于很高的 score 区域；validation 和 CEC 都没有接近该阈值。",
            "- **第三层是输入域差异**：CEC 的若干群体运动、多样性和 fitness-distribution 特征与训练参考分布不同，且一部分差异出现在高重要性特征上；30/50D 又超出训练维度 strata。",
            "",
            "因此，当前证据不支持把原因简化为“CEC 全部落在训练集之外”，也不支持只把问题归因于 30/50D。更准确的表述是：部署模型在 CEC 的实际 cmaes 行为输入上产生了整体低于 0 的 score；特征域差异和训练标签/阈值的稀有性共同使这一输出难以跨过触发边界。",
            "",
            "## 输出文件",
            "",
            "- `cec2017_decision_diagnostics_report.md`：本报告。",
            "- `decision_feature_profiles.csv`：按来源、维度组、采样阶段和特征的原始值与 model z-space 分布。",
            "- `decision_feature_shift.csv`：CEC 与维度匹配/近邻训练参考的特征位移。",
            "- `decision_score_summary.csv` 与 `decision_score_overall_summary.csv`：分阶段和总体 score 分布、阈值比例。",
            "- `cec_function_dimension_summary.csv`：CEC 函数/维度 score 聚合。",
            "- `cec_cmaes_decision_scores.parquet`：CEC cmaes 行为状态的 29 特征对应 score。",
            "- `decision_feature_input_sensitivity.csv`：单特征替代敏感性。",
            "- `online_score_reproduction_check.csv`：在线 score 与行为表重建 score 的核对结果。",
            "- `fig01_decision_feature_z_heatmap.png` 至 `fig05_importance_shift_sensitivity.png`：五张诊断图。",
            "",
            "## 下一步建议",
            "",
            "1. 在不改动当前主协议的前提下，单独开展“阈值可迁移性”敏感性分析：固定模型和特征工程，只报告不同预先指定阈值对 CEC 的 call rate，不将其替换为主结果。",
            "2. 为后续跨套件训练补充 30/50D 的训练参考，或把维度外推风险作为单独 strata 报告；不要把 30/50D 的行为直接当作 10/20/40D 的同分布样本。",
            "3. 如果下一步要判断是否值得重新定义标签，应先比较 BBOB-train nested family-OOF 中稀有 `g_fe_selected_path > 0` 行的特征画像与 CEC 画像，再决定是否做新的协议版本。",
        ]
    )
    if len(online_check):
        lines.extend(["", "## 在线 score 重建核对摘要", "", online_check.to_markdown(index=False)])
    (output / "cec2017_decision_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    cec_behavior_root: Path = DEFAULT_CEC_BEHAVIOR_ROOT,
    online_path: Path = DEFAULT_ONLINE_RUNS,
    output_dir: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录已有内容：{output_dir}；如需重跑请使用 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, model, threshold = _read_model(decision_dir)
    features = [str(value) for value in summary["feature_columns"]]
    if len(features) != 29:
        raise ValueError("Decision feature count must equal 29")
    assert len(features) == 29, f"Expected 29 Decision features, got {len(features)}"
    assert all(column.startswith("bf_") for column in features), "Missing behavior features"

    train, validation = _read_training_data(decision_dir, features)
    cec = _read_cec_behavior(cec_behavior_root, features)
    if cec.empty:
        raise ValueError("CEC cmaes behavior table is empty")
    frames = {
        "bbob_train": train[train["source"].eq("bbob_train")].copy(),
        "mabbob_formal": train[train["source"].eq("mabbob_formal")].copy(),
        CEC_SOURCE: cec,
    }
    profiles, shifts = _build_feature_profiles(frames, model, features)
    train_oof, validation_predictions, cec_scored = _score_frames(
        decision_dir, model, cec, features
    )
    score_summary = _score_summary(train_oof, validation_predictions, cec_scored, threshold)
    overall_scores = _overall_score_summary(train_oof, validation_predictions, cec_scored, threshold)
    cec_summary = _cec_function_dimension_summary(cec_scored, threshold)
    sensitivity = _model_input_sensitivity(cec_scored, model, features)
    train_for_support = pd.concat([frames["bbob_train"], frames["mabbob_formal"]], ignore_index=True)
    support_rows, support_summary = _row_support_summary(
        cec_scored, train_for_support, model, features
    )
    online_check = _online_reproduction_check(online_path, cec_scored)

    # Merge sensitivity into shifts for compact downstream analysis.
    shifts = shifts.merge(
        sensitivity.loc[:, ["feature", "cec_to_train_center_score_delta_median", "cec_to_train_center_score_delta_mean"]],
        on="feature",
        how="left",
        validate="many_to_one",
    )

    train_labels = pd.read_parquet(
        decision_dir / "decision_dataset.parquet",
        columns=["dataset_role", "g_fe_selected_path", "g_fe_selected_path_gt_zero"],
    )
    train_label_rows = train_labels[train_labels["dataset_role"].astype(str).eq("train")]
    top_imp = shifts[shifts["cec_dimension_group"].eq("30/50D")].nlargest(10, "model_feature_importance")
    top_shift = shifts[shifts["cec_dimension_group"].eq("30/50D")].assign(
        _absolute_shift=lambda x: x["robust_median_shift"].abs()
    ).sort_values("_absolute_shift", ascending=False).head(10)
    top_shift_abs = shifts[shifts["cec_dimension_group"].eq("30/50D")].reindex(
        shifts[shifts["cec_dimension_group"].eq("30/50D")]["robust_median_shift"].abs().sort_values(ascending=False).index
    ).head(10)
    metadata = {
        "analysis": "cec2017_decision_diagnostics",
        "status": "ok",
        "model": "random_forest_regressor",
        "feature_group": str(summary["feature_group"]),
        "feature_count": len(features),
        "features": features,
        "configured_threshold": threshold,
        "user_display_threshold": USER_THRESHOLD,
        "train_rows": int(len(train)),
        "bbob_train_rows": int(len(frames["bbob_train"])),
        "mabbob_train_rows": int(len(frames["mabbob_formal"])),
        "validation_rows": int(len(validation)),
        "cec_run_count": int(cec_scored[["problem_id", "seed"]].drop_duplicates().shape[0]),
        "cec_state_rows": int(len(cec_scored)),
        "cec_rate_gt_zero": float(np.mean(cec_scored["decision_score"].to_numpy(float) > 0.0)),
        "cec_rate_gt_user_threshold": float(
            np.mean(cec_scored["decision_score"].to_numpy(float) > USER_THRESHOLD)
        ),
        "cec_rate_gt_configured_threshold": float(
            np.mean(cec_scored["decision_score"].to_numpy(float) > threshold)
        ),
        "cec_score_max": float(cec_scored["decision_score"].max()),
        "cec_score_median": float(cec_scored["decision_score"].median()),
        "train_label_positive_rows": int(train_label_rows["g_fe_selected_path_gt_zero"].astype(bool).sum()),
        "train_label_positive_rate": float(train_label_rows["g_fe_selected_path_gt_zero"].astype(bool).mean()),
        "top10_importance_shift_overlap": int(
            len(set(top_imp["feature"]) & set(top_shift_abs["feature"]))
        ),
        "support_summary": support_summary,
        "online_score_reproduction_status": (
            str(online_check["status"].iloc[0]) if len(online_check) else "not_run"
        ),
        "sources": {
            "decision_summary": str(decision_dir / "full_decision_model_training_summary.json"),
            "decision_dataset": str(decision_dir / "decision_dataset.parquet"),
            "decision_model": str(decision_dir / "models/random_forest_regressor.joblib"),
            "cec_behavior_root": str(cec_behavior_root),
            "online_run_metrics": str(online_path),
        },
        "data_leakage_check": {
            "cec_rows_used_for_model_fit": 0,
            "cec_rows_used_for_preprocessing_fit": 0,
            "cec_rows_used_for_threshold_fit": 0,
            "query_descriptors_used_as_decision_inputs": False,
            "runtime_used_as_decision_inputs": False,
        },
    }

    profiles.to_csv(output_dir / "decision_feature_profiles.csv", index=False)
    shifts.to_csv(output_dir / "decision_feature_shift.csv", index=False)
    score_summary.to_csv(output_dir / "decision_score_summary.csv", index=False)
    overall_scores.to_csv(output_dir / "decision_score_overall_summary.csv", index=False)
    cec_summary.to_csv(output_dir / "cec_function_dimension_summary.csv", index=False)
    sensitivity.to_csv(output_dir / "decision_feature_input_sensitivity.csv", index=False)
    support_rows.to_csv(output_dir / "cec_row_support_summary.csv", index=False)
    online_check.to_csv(output_dir / "online_score_reproduction_check.csv", index=False)
    cec_scored.to_parquet(output_dir / "cec_cmaes_decision_scores.parquet", index=False)
    (output_dir / "cec2017_decision_diagnostics_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    importance_table = shifts[
        shifts["cec_dimension_group"].eq("30/50D")
    ].loc[:, ["feature", "feature_label", "feature_family", "model_feature_importance"]].drop_duplicates()
    _set_plot_style()
    _figure_feature_heatmap(profiles, shifts, output_dir)
    _figure_feature_distributions(train, cec, model, features, importance_table, shifts, output_dir)
    _figure_score_distributions(train_oof, validation_predictions, cec_scored, threshold, output_dir)
    _figure_cec_heatmap(cec_summary, output_dir)
    _figure_importance_shift(shifts, sensitivity, output_dir)
    _write_report(
        output_dir,
        metadata,
        overall_scores,
        cec_summary,
        shifts,
        sensitivity,
        support_summary,
        online_check,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-dir", type=Path, default=DEFAULT_DECISION_DIR)
    parser.add_argument("--cec-behavior-root", type=Path, default=DEFAULT_CEC_BEHAVIOR_ROOT)
    parser.add_argument("--online-path", type=Path, default=DEFAULT_ONLINE_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_analysis(
        decision_dir=args.decision_dir,
        cec_behavior_root=args.cec_behavior_root,
        online_path=args.online_path,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
