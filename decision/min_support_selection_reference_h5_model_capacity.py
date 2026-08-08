from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_family_split, _check_target, _json_default, _read_labels
from decision.min_support_f024_behavior_separability import _load_config, _target_holdout_pairs, _target_problem_ids
from ela.features import ELA_FEATURE_COLUMNS


TARGET_FAMILIES = ("bbob_f005", "bbob_f019", "bbob_f024")
FE_KEY_DECIMALS = 6
RF_GRID: tuple[tuple[int | None, int], ...] = (
    (None, 1),
    (None, 3),
    (3, 1),
    (3, 3),
    (5, 1),
    (5, 3),
    (8, 1),
    (8, 3),
    (8, 5),
)


def run_h5_model_capacity_diagnostic(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    extension_train_labels_path: Path,
    selection_reference_path: Path,
    extension_selection_reference_path: Path,
    train_ela_features_path: Path,
    validation_ela_features_path: Path,
    extension_ela_features_path: Path,
    train_trajectory_root: Path,
    validation_trajectory_root: Path,
    extension_trajectory_root: Path,
    extension_config_path: Path,
    output_dir: Path,
    target_column: str,
    random_seed: int,
) -> dict[str, Any]:
    _check_target(target_column)
    train_labels = _read_labels(train_labels_path).assign(dataset_role="train")
    validation_labels = _read_labels(validation_labels_path).assign(dataset_role="validation")
    _check_family_split(train_labels, validation_labels)

    config = _load_config(extension_config_path)
    target_problem_ids = _target_problem_ids(config)
    target_holdout_pairs = _target_holdout_pairs(config)

    extension_labels = pd.read_parquet(extension_train_labels_path).assign(dataset_role="extension_train")
    labels = pd.concat([train_labels, validation_labels, extension_labels], ignore_index=True)
    labels = _annotate_label_rows(labels, target_problem_ids=target_problem_ids, target_holdout_pairs=target_holdout_pairs, target_column=target_column)

    base_selection = _read_selection_reference(selection_reference_path)
    extension_selection = _read_selection_reference(extension_selection_reference_path, forced_dataset_role="extension_train")
    selection = pd.concat([base_selection, extension_selection], ignore_index=True)

    features = pd.concat(
        [
            _read_ela_features(train_ela_features_path, dataset_role="train"),
            _read_ela_features(validation_ela_features_path, dataset_role="validation"),
            _read_ela_features(extension_ela_features_path, dataset_role="extension_train"),
        ],
        ignore_index=True,
    )
    train_samples = _training_samples(selection, features)
    predictions, model_summary = _variant_predictions(
        train_samples=train_samples,
        selection=selection,
        features=features,
        random_seed=random_seed,
    )
    metric_rows = _label_metric_rows(
        labels=labels,
        predictions=predictions,
        trajectory_roots={
            "train": train_trajectory_root,
            "validation": validation_trajectory_root,
            "extension_train": extension_trajectory_root,
        },
        target_column=target_column,
    )
    problem_stage_summary = _problem_stage_summary(predictions)
    label_metric_summary = _label_metric_summary(metric_rows, target_column)
    conclusion = _diagnostic_conclusion(problem_stage_summary, label_metric_summary, model_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples_path = output_dir / "h5_training_samples.parquet"
    predictions_path = output_dir / "h5_selector_predictions.parquet"
    model_summary_path = output_dir / "h5_selector_model_summary.parquet"
    metric_rows_path = output_dir / "h5_label_metric_rows.parquet"
    problem_stage_summary_path = output_dir / "h5_problem_stage_summary.parquet"
    label_metric_summary_path = output_dir / "h5_label_metric_summary.parquet"
    summary_path = output_dir / "h5_model_capacity_summary.json"
    report_path = output_dir / "h5_model_capacity_report.md"

    pq.write_table(pa.Table.from_pandas(train_samples, preserve_index=False), train_samples_path)
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), predictions_path)
    pq.write_table(pa.Table.from_pandas(model_summary, preserve_index=False), model_summary_path)
    pq.write_table(pa.Table.from_pandas(metric_rows, preserve_index=False), metric_rows_path)
    pq.write_table(pa.Table.from_pandas(problem_stage_summary, preserve_index=False), problem_stage_summary_path)
    pq.write_table(pa.Table.from_pandas(label_metric_summary, preserve_index=False), label_metric_summary_path)

    summary = {
        "experiment": "selection_reference_h5_model_capacity_diagnostic",
        "research_question": (
            "Do RF depth/min_samples_leaf, logistic, nearest-neighbor bucket, and stage-wise majority selectors "
            "show different generalization patterns on the H1-H4 evaluation domains?"
        ),
        "target_column": target_column,
        "selector_variants": _selector_variant_descriptions(),
        "evaluation_domains": {
            "problem_stage": [
                "train_all",
                "validation_all",
                "validation_bbob_f005",
                "validation_bbob_f019",
                "validation_bbob_f024",
                "extension_train_f024_followup",
            ],
            "label_rows": [
                "train_all",
                "validation_all",
                "validation_bbob_f005",
                "validation_bbob_f019",
                "validation_bbob_f024",
                "extension_train_f024_followup",
                "extension_train_f024_target_changed_late",
                "validation_f024_target_holdout_changed_late",
                "validation_non_target_changed_late",
            ],
        },
        "inputs": {
            "train_labels": str(train_labels_path),
            "validation_labels": str(validation_labels_path),
            "extension_train_labels": str(extension_train_labels_path),
            "selection_reference": str(selection_reference_path),
            "extension_selection_reference": str(extension_selection_reference_path),
            "train_ela_features": str(train_ela_features_path),
            "validation_ela_features": str(validation_ela_features_path),
            "extension_ela_features": str(extension_ela_features_path),
            "train_trajectory_root": str(train_trajectory_root),
            "validation_trajectory_root": str(validation_trajectory_root),
            "extension_trajectory_root": str(extension_trajectory_root),
            "extension_config": str(extension_config_path),
        },
        "outputs": {
            "training_samples": str(train_samples_path),
            "selector_predictions": str(predictions_path),
            "selector_model_summary": str(model_summary_path),
            "label_metric_rows": str(metric_rows_path),
            "problem_stage_summary": str(problem_stage_summary_path),
            "label_metric_summary": str(label_metric_summary_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "original_utility_labels_modified": False,
            "selection_reference_modified": False,
            "formal_feature_extractor_modified": False,
            "formal_phase1_configs_modified": False,
            "utility_labels_regenerated": False,
            "diagnostic_selector_models_trained": True,
            "diagnostic_models_written_as_formal_phase1": False,
            "decision_input_uses_ela_features": True,
            "ela_features_used_only_for_fixed_downstream_selector_diagnostic": True,
            "function_id_used_as_decision_input": False,
            "problem_id_used_only_for_domain_definition": True,
        },
        "notes": [
            "selected=VBS is computed on the problem-stage selection_reference grain.",
            "U_ELA>0 capture and precision use bucket_proxy_u > 0 as the diagnostic selector call because alternate utility-label rows are not regenerated.",
            "bucket_proxy_p_ela and bucket_proxy_u are computed from existing trajectory performance buckets and existing time_cost_norm.",
            "Observed P_ELA/U_ELA are reported only when the diagnostic selector chooses the same algorithm as the original utility-label row.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown_report(summary, problem_stage_summary, label_metric_summary, model_summary), encoding="utf-8")

    print(f"wrote H5 training samples to {train_samples_path}")
    print(f"wrote H5 selector predictions to {predictions_path}")
    print(f"wrote H5 selector model summary to {model_summary_path}")
    print(f"wrote H5 label metric rows to {metric_rows_path}")
    print(f"wrote H5 problem-stage summary to {problem_stage_summary_path}")
    print(f"wrote H5 label metric summary to {label_metric_summary_path}")
    print(f"wrote H5 summary to {summary_path}")
    print(f"wrote H5 report to {report_path}")
    return summary


def _read_selection_reference(path: Path, forced_dataset_role: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing selection_reference: {path}")
    frame = pq.read_table(path).to_pandas()
    if forced_dataset_role is None:
        frame["dataset_role"] = np.where(frame["split"].astype(str).str.contains("validation"), "validation", "train")
    else:
        frame["dataset_role"] = forced_dataset_role
    frame["remaining_budget_key"] = frame["remaining_budget_ratio"].round(FE_KEY_DECIMALS)
    return frame


def _read_ela_features(path: Path, *, dataset_role: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing ELA features: {path}")
    frame = pq.read_table(path).to_pandas()
    failed = frame[frame["feature_status"] != "ok"]
    if not failed.empty:
        raise ValueError(f"ELA feature file contains failed rows: {path}")
    frame["dataset_role"] = dataset_role
    return frame


def _training_samples(selection: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train_selection = selection[selection["dataset_role"] == "train"].copy()
    train_features = features[features["dataset_role"] == "train"].copy()
    samples = train_selection.merge(
        train_features[["dataset_role", "split", "problem_id", "family", "dimension", *ELA_FEATURE_COLUMNS]],
        on=["dataset_role", "split", "problem_id", "family", "dimension"],
        how="left",
        validate="many_to_one",
    )
    missing_features = samples[list(ELA_FEATURE_COLUMNS)].isna().any(axis=1)
    if missing_features.any():
        missing = samples.loc[missing_features, ["split", "problem_id", "dimension"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"training samples missing ELA features: {missing}")
    missing_vbs = samples["vbs_algorithm"].fillna("").astype(str) == ""
    if missing_vbs.any():
        missing = samples.loc[missing_vbs, ["split", "problem_id", "dimension", "remaining_budget_key"]].to_dict(orient="records")
        raise ValueError(f"training samples missing VBS targets: {missing}")
    return samples


def _variant_predictions(
    *,
    train_samples: pd.DataFrame,
    selection: pd.DataFrame,
    features: pd.DataFrame,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_base = selection.merge(
        features[["dataset_role", "split", "problem_id", "family", "dimension", *ELA_FEATURE_COLUMNS]],
        on=["dataset_role", "split", "problem_id", "family", "dimension"],
        how="left",
        validate="many_to_one",
    )
    if prediction_base[list(ELA_FEATURE_COLUMNS)].isna().any(axis=1).any():
        missing = prediction_base[prediction_base[list(ELA_FEATURE_COLUMNS)].isna().any(axis=1)][
            ["dataset_role", "split", "problem_id", "dimension"]
        ].drop_duplicates()
        raise ValueError(f"prediction rows are missing ELA features: {missing.to_dict(orient='records')}")

    prediction_frames = [_existing_selection_predictions(prediction_base)]
    model_rows = [
        {
            "remaining_budget_key": np.nan,
            "selector_variant": "existing_selection_reference",
            "selector_family": "existing_selection_reference",
            "diagnostic_selector_status": "existing_selection_reference",
            "train_samples_available": int(len(train_samples)),
            "train_samples_used": 0,
            "target_algorithm_counts": "",
            "selected_algorithm_counts_train_prediction": "",
        }
    ]

    variant_specs = _selector_variant_specs()
    fallback = _global_sbs_algorithm(train_samples)
    for remaining_budget, stage_train in train_samples.groupby("remaining_budget_key", sort=True):
        stage_predict = prediction_base[prediction_base["remaining_budget_key"] == remaining_budget].copy()
        if stage_predict.empty:
            continue
        for spec in variant_specs:
            selected, status = _predict_stage_variant(
                spec=spec,
                stage_train=stage_train,
                stage_predict=stage_predict,
                fallback=fallback,
                random_seed=random_seed,
            )
            variant_frame = _prediction_frame(stage_predict, selector_variant=spec["name"], status=status, selected=selected)
            prediction_frames.append(variant_frame)
            model_rows.append(
                {
                    "remaining_budget_key": float(remaining_budget),
                    "selector_variant": spec["name"],
                    "selector_family": spec["family"],
                    "diagnostic_selector_status": status,
                    "train_samples_available": int(len(stage_train)),
                    "train_samples_used": int(_train_rows_used(stage_train, spec)),
                    "target_algorithm_counts": _value_counts_string(stage_train["vbs_algorithm"]),
                    "selected_algorithm_counts_train_prediction": "",
                    "rf_max_depth": spec.get("max_depth"),
                    "rf_min_samples_leaf": spec.get("min_samples_leaf"),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_summary = pd.DataFrame(model_rows)
    train_pred_counts = (
        predictions[predictions["dataset_role"] == "train"]
        .groupby(["remaining_budget_key", "selector_variant"])["diagnostic_selected_algorithm"]
        .apply(_value_counts_string)
        .reset_index(name="selected_algorithm_counts_train_prediction")
    )
    model_summary = model_summary.drop(columns=["selected_algorithm_counts_train_prediction"]).merge(
        train_pred_counts,
        on=["remaining_budget_key", "selector_variant"],
        how="left",
    )
    return predictions, model_summary


def _existing_selection_predictions(prediction_base: pd.DataFrame) -> pd.DataFrame:
    return _prediction_frame(
        prediction_base,
        selector_variant="existing_selection_reference",
        status="existing_selection_reference",
        selected=prediction_base["selected_algorithm"].astype(str).to_numpy(dtype=object),
    )


def _prediction_frame(stage_predict: pd.DataFrame, *, selector_variant: str, status: str, selected: np.ndarray) -> pd.DataFrame:
    frame = stage_predict[
        [
            "dataset_role",
            "split",
            "problem_id",
            "family",
            "dimension",
            "remaining_budget_ratio",
            "remaining_budget_key",
            "performance_bucket_ratio",
            "default_algorithm",
            "sbs_algorithm",
            "vbs_algorithm",
            "selector_status",
        ]
    ].copy()
    frame["selector_variant"] = selector_variant
    frame["diagnostic_selector_status"] = status
    frame["diagnostic_selected_algorithm"] = selected
    frame["diagnostic_selected_differs_from_default"] = (
        frame["diagnostic_selected_algorithm"].astype(str) != frame["default_algorithm"].astype(str)
    )
    frame["diagnostic_selected_matches_vbs"] = (
        frame["diagnostic_selected_algorithm"].astype(str) == frame["vbs_algorithm"].astype(str)
    )
    return frame


def _selector_variant_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "stage_wise_majority", "family": "stage_wise_majority"},
        {"name": "nearest_neighbor_bucket", "family": "nearest_neighbor_bucket"},
        {"name": "logistic_multinomial", "family": "logistic"},
    ]
    for max_depth, min_samples_leaf in RF_GRID:
        depth_name = "none" if max_depth is None else str(max_depth)
        specs.append(
            {
                "name": f"rf_depth_{depth_name}_leaf_{min_samples_leaf}",
                "family": "random_forest",
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
        )
    return specs


def _selector_variant_descriptions() -> dict[str, str]:
    descriptions = {
        "existing_selection_reference": "existing selection_reference selected_algorithm, read only",
        "stage_wise_majority": "per remaining_budget_ratio majority VBS algorithm from train problem-stage rows",
        "nearest_neighbor_bucket": "per remaining_budget_ratio 1-nearest-neighbor classifier on ELA features",
        "logistic_multinomial": "per remaining_budget_ratio multinomial logistic classifier on ELA features",
    }
    for max_depth, min_samples_leaf in RF_GRID:
        depth_name = "none" if max_depth is None else str(max_depth)
        descriptions[f"rf_depth_{depth_name}_leaf_{min_samples_leaf}"] = (
            f"per remaining_budget_ratio RandomForestClassifier(max_depth={max_depth}, min_samples_leaf={min_samples_leaf})"
        )
    return descriptions


def _predict_stage_variant(
    *,
    spec: dict[str, Any],
    stage_train: pd.DataFrame,
    stage_predict: pd.DataFrame,
    fallback: str,
    random_seed: int,
) -> tuple[np.ndarray, str]:
    if spec["family"] == "stage_wise_majority":
        selected = _majority_algorithm(stage_train["vbs_algorithm"], fallback=fallback)
        return np.full(len(stage_predict), selected, dtype=object), "stage_wise_majority"
    model, status = _fit_stage_model(spec=spec, stage_train=stage_train, fallback=fallback, random_seed=random_seed)
    selected = _predict_stage_model(model, status=status, stage_predict=stage_predict, fallback=fallback)
    return selected, status


def _fit_stage_model(
    *,
    spec: dict[str, Any],
    stage_train: pd.DataFrame,
    fallback: str,
    random_seed: int,
) -> tuple[Pipeline | str, str]:
    fit_rows = stage_train.copy()
    y = fit_rows["vbs_algorithm"].fillna("").astype(str)
    valid = y != ""
    fit_rows = fit_rows.loc[valid].copy()
    y = y.loc[valid]
    if fit_rows.empty:
        return fallback, "fallback_no_vbs_targets"
    if len(set(y)) <= 1:
        return str(y.iloc[0]), "constant_single_class"

    if spec["family"] == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
            max_depth=spec["max_depth"],
            min_samples_leaf=int(spec["min_samples_leaf"]),
            n_jobs=-1,
        )
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])
    elif spec["family"] == "logistic":
        classifier = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    elif spec["family"] == "nearest_neighbor_bucket":
        classifier = KNeighborsClassifier(n_neighbors=1, weights="uniform")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    else:
        raise ValueError(f"unknown selector family: {spec['family']}")
    model.fit(fit_rows[list(ELA_FEATURE_COLUMNS)], y)
    return model, str(spec["family"])


def _predict_stage_model(model: Pipeline | str, *, status: str, stage_predict: pd.DataFrame, fallback: str) -> np.ndarray:
    if stage_predict.empty:
        return np.asarray([], dtype=object)
    if status in {"fallback_no_vbs_targets", "constant_single_class"}:
        return np.full(len(stage_predict), str(model), dtype=object)
    if not isinstance(model, Pipeline):
        return np.full(len(stage_predict), fallback, dtype=object)
    return model.predict(stage_predict[list(ELA_FEATURE_COLUMNS)]).astype(str)


def _train_rows_used(stage_train: pd.DataFrame, spec: dict[str, Any]) -> int:
    if spec["family"] == "stage_wise_majority":
        return int(len(stage_train))
    return int((stage_train["vbs_algorithm"].fillna("").astype(str) != "").sum())


def _global_sbs_algorithm(train_samples: pd.DataFrame) -> str:
    if "sbs_algorithm" in train_samples.columns:
        selected = _majority_algorithm(train_samples["sbs_algorithm"], fallback="")
        if selected:
            return selected
    return _majority_algorithm(train_samples["default_algorithm"], fallback="cmaes")


def _majority_algorithm(series: pd.Series, *, fallback: str) -> str:
    counts = series.fillna("").astype(str)
    counts = counts[counts != ""].value_counts()
    if counts.empty:
        return fallback
    best_count = counts.max()
    tied = sorted(str(index) for index, value in counts.items() if value == best_count)
    return tied[0] if tied else fallback


def _annotate_label_rows(
    labels: pd.DataFrame,
    *,
    target_problem_ids: set[str],
    target_holdout_pairs: set[tuple[str, int]],
    target_column: str,
) -> pd.DataFrame:
    frame = labels.copy()
    frame["remaining_budget_key"] = (frame["FE_ela_optimization"] / frame["FE_total"]).round(FE_KEY_DECIMALS)
    frame["label_source"] = np.where(
        frame["selected_algorithm"].astype(str) == frame["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    frame["utility_gt_zero"] = frame[target_column] > 0.0
    frame["stage"] = np.where(frame["FE_ratio"].astype(float) >= 0.5, "late_stage", "not_late_stage")
    frame["is_target_problem"] = frame["problem_id"].astype(str).isin(target_problem_ids)
    pairs = list(zip(frame["problem_id"].astype(str), frame["seed"].astype(int), strict=False))
    frame["is_target_holdout_seed"] = [pair in target_holdout_pairs for pair in pairs]
    return frame


def _label_metric_rows(
    *,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    trajectory_roots: dict[str, Path],
    target_column: str,
) -> pd.DataFrame:
    metric_rows = labels.merge(
        predictions[
            [
                "selector_variant",
                "dataset_role",
                "split",
                "problem_id",
                "family",
                "dimension",
                "remaining_budget_key",
                "performance_bucket_ratio",
                "diagnostic_selector_status",
                "diagnostic_selected_algorithm",
                "diagnostic_selected_matches_vbs",
                "diagnostic_selected_differs_from_default",
                "vbs_algorithm",
            ]
        ],
        on=["dataset_role", "split", "problem_id", "family", "dimension", "remaining_budget_key"],
        how="inner",
        validate="many_to_many",
    )
    metric_rows["observed_label_available"] = (
        metric_rows["diagnostic_selected_algorithm"].astype(str) == metric_rows["selected_algorithm"].astype(str)
    )
    metric_rows["observed_p_ela_for_diagnostic_selector"] = np.where(
        metric_rows["observed_label_available"],
        metric_rows["p_ela"],
        np.nan,
    )
    metric_rows["observed_u_for_diagnostic_selector"] = np.where(
        metric_rows["observed_label_available"],
        metric_rows[target_column],
        np.nan,
    )
    proxy = _trajectory_proxy_table(labels=labels, trajectory_roots=trajectory_roots)
    metric_rows = metric_rows.merge(
        proxy,
        left_on=[
            "dataset_role",
            "problem_id",
            "dimension",
            "performance_bucket_ratio",
            "diagnostic_selected_algorithm",
            "default_algorithm",
        ],
        right_on=["dataset_role", "problem_id", "dimension", "performance_bucket_ratio", "selected_algorithm_proxy", "default_algorithm"],
        how="left",
        validate="many_to_one",
    )
    metric_rows["bucket_proxy_performance_gain_norm"] = (
        (metric_rows["bucket_proxy_p_skip"] - metric_rows["bucket_proxy_p_ela"])
        / np.maximum(np.abs(metric_rows["bucket_proxy_p_skip"]), 1e-12)
    )
    metric_rows["bucket_proxy_u"] = metric_rows["bucket_proxy_performance_gain_norm"] - metric_rows["time_cost_norm"]
    metric_rows["bucket_proxy_u_gt_zero"] = metric_rows["bucket_proxy_u"] > 0.0
    return metric_rows


def _trajectory_proxy_table(*, labels: pd.DataFrame, trajectory_roots: dict[str, Path]) -> pd.DataFrame:
    keys = labels[["dataset_role", "family", "problem_id", "dimension"]].drop_duplicates()
    frames = []
    for dataset_role, root in trajectory_roots.items():
        role_keys = keys[keys["dataset_role"] == dataset_role]
        if role_keys.empty:
            continue
        for (family, dimension), _ in role_keys.groupby(["family", "dimension"], sort=True):
            path = root / str(family) / f"dimension_{int(dimension)}" / "trajectories.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing trajectory shard: {path}")
            table = pq.read_table(path, columns=["problem_id", "dimension", "algorithm", "FE_ratio", "best_fitness"])
            frame = table.to_pandas()
            frame["dataset_role"] = dataset_role
            frames.append(frame)
    if not frames:
        raise ValueError("no trajectory rows available for proxy utility")
    performance = pd.concat(frames, ignore_index=True)
    perf = (
        performance.groupby(["dataset_role", "problem_id", "dimension", "algorithm", "FE_ratio"], as_index=False)["best_fitness"]
        .mean()
        .rename(columns={"best_fitness": "mean_best_fitness", "FE_ratio": "performance_bucket_ratio"})
    )
    selected = perf.rename(
        columns={"algorithm": "selected_algorithm_proxy", "mean_best_fitness": "bucket_proxy_p_ela"}
    )
    default = perf.rename(columns={"algorithm": "default_algorithm", "mean_best_fitness": "bucket_proxy_p_skip"})[
        ["dataset_role", "problem_id", "dimension", "performance_bucket_ratio", "default_algorithm", "bucket_proxy_p_skip"]
    ]
    return selected.merge(
        default,
        on=["dataset_role", "problem_id", "dimension", "performance_bucket_ratio"],
        how="inner",
        validate="many_to_many",
    )


def _problem_stage_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for selector_variant, variant_rows in predictions.groupby("selector_variant", sort=True):
        for domain_name, domain_rows in _problem_stage_domains(variant_rows):
            rows.append(_problem_stage_summary_row(selector_variant, domain_name, domain_rows))
    return pd.DataFrame(rows)


def _problem_stage_domains(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    validation = frame[frame["dataset_role"] == "validation"]
    return [
        ("train_all", frame[frame["dataset_role"] == "train"]),
        ("validation_all", validation),
        ("validation_bbob_f005", validation[validation["family"] == "bbob_f005"]),
        ("validation_bbob_f019", validation[validation["family"] == "bbob_f019"]),
        ("validation_bbob_f024", validation[validation["family"] == "bbob_f024"]),
        ("extension_train_f024_followup", frame[frame["dataset_role"] == "extension_train"]),
    ]


def _problem_stage_summary_row(selector_variant: str, evaluation_domain: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "selector_variant": selector_variant,
        "evaluation_domain": evaluation_domain,
        "problem_stage_rows": int(len(frame)),
        "selected_matches_vbs_rows": int(frame["diagnostic_selected_matches_vbs"].sum()),
        "selected_matches_vbs_rate": float(frame["diagnostic_selected_matches_vbs"].mean()) if len(frame) else 0.0,
        "changed_selection_rows": int(frame["diagnostic_selected_differs_from_default"].sum()),
        "changed_selection_rate": float(frame["diagnostic_selected_differs_from_default"].mean()) if len(frame) else 0.0,
        "diagnostic_selected_algorithm_counts": _value_counts_string(frame["diagnostic_selected_algorithm"]),
        "vbs_algorithm_counts": _value_counts_string(frame["vbs_algorithm"]),
        "diagnostic_selector_status_counts": _value_counts_string(frame["diagnostic_selector_status"]),
    }


def _label_metric_summary(metric_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for selector_variant, variant_rows in metric_rows.groupby("selector_variant", sort=True):
        for domain_name, domain_rows in _label_domains(variant_rows):
            rows.append(_label_metric_summary_row(selector_variant, domain_name, "all", domain_rows, target_column))
            for label_source in ("same_algorithm", "changed_algorithm"):
                source_rows = domain_rows[domain_rows["label_source"] == label_source]
                rows.append(_label_metric_summary_row(selector_variant, domain_name, label_source, source_rows, target_column))
    return pd.DataFrame(rows)


def _label_domains(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    train = frame[frame["dataset_role"] == "train"]
    validation = frame[frame["dataset_role"] == "validation"]
    extension = frame[frame["dataset_role"] == "extension_train"]
    changed_late_validation = validation[
        (validation["label_source"] == "changed_algorithm")
        & (validation["FE_ratio"].astype(float) >= 0.5)
    ]
    target_holdout = (
        changed_late_validation["is_target_problem"].to_numpy(dtype=bool)
        & changed_late_validation["is_target_holdout_seed"].to_numpy(dtype=bool)
    )
    extension_target_changed_late = extension[
        (extension["is_target_problem"].to_numpy(dtype=bool))
        & (extension["label_source"].to_numpy() == "changed_algorithm")
        & (extension["FE_ratio"].to_numpy(dtype=float) >= 0.5)
    ]
    return [
        ("train_all", train),
        ("validation_all", validation),
        ("validation_bbob_f005", validation[validation["family"] == "bbob_f005"]),
        ("validation_bbob_f019", validation[validation["family"] == "bbob_f019"]),
        ("validation_bbob_f024", validation[validation["family"] == "bbob_f024"]),
        ("extension_train_f024_followup", extension),
        ("extension_train_f024_target_changed_late", extension_target_changed_late),
        ("validation_f024_target_holdout_changed_late", changed_late_validation[target_holdout]),
        ("validation_non_target_changed_late", changed_late_validation[~target_holdout]),
    ]


def _label_metric_summary_row(
    selector_variant: str,
    evaluation_domain: str,
    label_source: str,
    frame: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    original_positive = frame[target_column] > 0.0
    proxy_positive = frame["bucket_proxy_u_gt_zero"].fillna(False)
    true_positive = original_positive & proxy_positive
    observed = frame[frame["observed_label_available"]]
    called = frame[proxy_positive]
    return {
        "selector_variant": selector_variant,
        "evaluation_domain": evaluation_domain,
        "label_source": label_source,
        "rows": int(len(frame)),
        "problem_count": int(frame["problem_id"].nunique()) if len(frame) else 0,
        "utility_gt_zero_rows": int(original_positive.sum()),
        "diagnostic_proxy_u_gt_zero_rows": int(proxy_positive.sum()),
        "u_ela_gt_zero_capture_rows": int(true_positive.sum()),
        "u_ela_gt_zero_capture_rate": float(true_positive.sum() / original_positive.sum()) if original_positive.sum() else 0.0,
        "precision": float(true_positive.sum() / proxy_positive.sum()) if proxy_positive.sum() else 0.0,
        "selected_matches_vbs_rate": float(frame["diagnostic_selected_matches_vbs"].mean()) if len(frame) else 0.0,
        "observed_label_rows": int(len(observed)),
        "observed_label_coverage_rate": float(len(observed) / len(frame)) if len(frame) else 0.0,
        "p_ela_mean_observed_when_available": float(observed["p_ela"].mean()) if len(observed) else None,
        "u_mean_observed_when_available": float(observed[target_column].mean()) if len(observed) else None,
        "bucket_proxy_missing_rows": int(frame["bucket_proxy_u"].isna().sum()) if len(frame) else 0,
        "bucket_proxy_p_ela_mean": float(frame["bucket_proxy_p_ela"].mean()) if len(frame) else None,
        "bucket_proxy_u_mean": float(frame["bucket_proxy_u"].mean()) if len(frame) else None,
        "called_bucket_proxy_p_ela_mean": float(called["bucket_proxy_p_ela"].mean()) if len(called) else None,
        "called_bucket_proxy_u_mean": float(called["bucket_proxy_u"].mean()) if len(called) else None,
        "original_p_ela_mean": float(frame["p_ela"].mean()) if len(frame) else None,
        "original_u_mean": float(frame[target_column].mean()) if len(frame) else None,
        "diagnostic_selected_algorithm_counts": _value_counts_string(frame["diagnostic_selected_algorithm"]),
        "original_selected_algorithm_counts": _value_counts_string(frame["selected_algorithm"]),
    }


def _diagnostic_conclusion(
    problem_stage_summary: pd.DataFrame,
    label_metric_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> dict[str, Any]:
    validation_problem = problem_stage_summary[problem_stage_summary["evaluation_domain"] == "validation_all"].set_index("selector_variant")
    validation_metrics = label_metric_summary[
        (label_metric_summary["evaluation_domain"] == "validation_all")
        & (label_metric_summary["label_source"] == "all")
    ].set_index("selector_variant")
    f024_holdout = label_metric_summary[
        (label_metric_summary["evaluation_domain"] == "validation_f024_target_holdout_changed_late")
        & (label_metric_summary["label_source"] == "all")
    ].set_index("selector_variant")
    best_validation_vbs = validation_problem["selected_matches_vbs_rate"].astype(float).sort_values(ascending=False).head(5)
    best_validation_u = validation_metrics["bucket_proxy_u_mean"].astype(float).sort_values(ascending=False).head(5)
    best_f024_capture = f024_holdout["u_ela_gt_zero_capture_rate"].astype(float).sort_values(ascending=False).head(5)
    train_perfect_variants = problem_stage_summary[
        (problem_stage_summary["evaluation_domain"] == "train_all")
        & (problem_stage_summary["selected_matches_vbs_rate"] >= 0.999999)
    ]["selector_variant"].astype(str).tolist()
    rf_status_counts = model_summary[model_summary["selector_family"] == "random_forest"]["diagnostic_selector_status"].value_counts().to_dict()
    return {
        "top_validation_selected_vbs": {str(key): float(value) for key, value in best_validation_vbs.items()},
        "top_validation_bucket_proxy_u": {str(key): float(value) for key, value in best_validation_u.items()},
        "top_f024_holdout_capture": {str(key): float(value) for key, value in best_f024_capture.items()},
        "train_perfect_selected_vbs_variants": train_perfect_variants,
        "random_forest_status_counts": {str(key): int(value) for key, value in rf_status_counts.items()},
        "interpretation": (
            "H5 isolates model capacity and inductive bias by retraining diagnostic ELA selectors on the same "
            "train problem-stage targets. If simpler selectors improve selected=VBS without improving bucket proxy "
            "U_ELA, the failure should not be attributed to capacity alone."
        ),
    }


def _markdown_report(
    summary: dict[str, Any],
    problem_stage_summary: pd.DataFrame,
    label_metric_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    key_variants = [
        "existing_selection_reference",
        "rf_depth_none_leaf_1",
        "rf_depth_3_leaf_3",
        "rf_depth_5_leaf_3",
        "rf_depth_8_leaf_5",
        "logistic_multinomial",
        "nearest_neighbor_bucket",
        "stage_wise_majority",
    ]
    problem = problem_stage_summary[
        problem_stage_summary["evaluation_domain"].isin(
            ["train_all", "validation_all", "validation_bbob_f005", "validation_bbob_f019", "validation_bbob_f024"]
        )
        & problem_stage_summary["selector_variant"].isin(key_variants)
    ].sort_values(["evaluation_domain", "selector_variant"])
    labels = label_metric_summary[
        label_metric_summary["evaluation_domain"].isin(
            [
                "validation_all",
                "validation_bbob_f005",
                "validation_bbob_f019",
                "validation_bbob_f024",
                "extension_train_f024_target_changed_late",
                "validation_f024_target_holdout_changed_late",
                "validation_non_target_changed_late",
            ]
        )
        & (label_metric_summary["label_source"] == "all")
        & label_metric_summary["selector_variant"].isin(key_variants)
    ].sort_values(["evaluation_domain", "selector_variant"])
    rf_grid = label_metric_summary[
        (label_metric_summary["evaluation_domain"] == "validation_all")
        & (label_metric_summary["label_source"] == "all")
        & (label_metric_summary["selector_variant"].str.startswith("rf_depth_"))
    ].sort_values(["bucket_proxy_u_mean", "selected_matches_vbs_rate"], ascending=[False, False])

    lines = [
        "# selection_reference 泛化失败 H5 最小诊断",
        "",
        "本报告只使用当前项目内已有 `utility_labels`、`selection_reference`、ELA features 和 trajectories；未修改原始结果，未重训正式 selector，未改正式 feature extractor 或 phase1 配置。",
        "",
        "## Problem-stage selected=VBS",
        "",
        "| domain | selector_variant | selected=VBS | changed selection rate | selected counts |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in problem.iterrows():
        lines.append(
            f"| `{row['evaluation_domain']}` | `{row['selector_variant']}` | "
            f"{row['selected_matches_vbs_rate']:.6f} | {row['changed_selection_rate']:.6f} | "
            f"`{row['diagnostic_selected_algorithm_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Label-row Utility Metrics",
            "",
            "`capture` 与 `precision` 使用 `bucket_proxy_u > 0` 作为诊断 selector 的正效用判断；`P_ELA/U_ELA` 为 trajectory bucket proxy，不是新生成的 utility label。",
            "",
            "| domain | selector_variant | rows | selected=VBS | U>0 capture | precision | proxy P_ELA | proxy U_ELA | observed coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in labels.iterrows():
        p_ela = "NA" if pd.isna(row["bucket_proxy_p_ela_mean"]) else f"{row['bucket_proxy_p_ela_mean']:.6f}"
        u_ela = "NA" if pd.isna(row["bucket_proxy_u_mean"]) else f"{row['bucket_proxy_u_mean']:.6f}"
        lines.append(
            f"| `{row['evaluation_domain']}` | `{row['selector_variant']}` | {int(row['rows'])} | "
            f"{row['selected_matches_vbs_rate']:.6f} | {row['u_ela_gt_zero_capture_rate']:.6f} | "
            f"{row['precision']:.6f} | {p_ela} | {u_ela} | {row['observed_label_coverage_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## RF Grid on validation_all",
            "",
            "| selector_variant | selected=VBS | U>0 capture | precision | proxy P_ELA | proxy U_ELA |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in rf_grid.iterrows():
        lines.append(
            f"| `{row['selector_variant']}` | {row['selected_matches_vbs_rate']:.6f} | "
            f"{row['u_ela_gt_zero_capture_rate']:.6f} | {row['precision']:.6f} | "
            f"{row['bucket_proxy_p_ela_mean']:.6f} | {row['bucket_proxy_u_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Model Status",
            "",
            "| selector_family | status | stages |",
            "|---|---|---:|",
        ]
    )
    status_counts = (
        model_summary.groupby(["selector_family", "diagnostic_selector_status"], dropna=False)
        .size()
        .reset_index(name="stages")
        .sort_values(["selector_family", "diagnostic_selector_status"])
    )
    for _, row in status_counts.iterrows():
        lines.append(f"| `{row['selector_family']}` | `{row['diagnostic_selector_status']}` | {int(row['stages'])} |")
    lines.extend(
        [
            "",
            "## 诊断边界",
            "",
            "- H5 的 RF/logistic/kNN/stage-wise majority 均为诊断训练副本，只读取现有 train problem-stage VBS target。",
            "- `observed coverage` 表示诊断 selector 选择的算法与原始 utility label 行一致的比例；只有这些行有真实观测 `P_ELA/U_ELA`。",
            "- 其他行的 `P_ELA/U_ELA` 使用已有 trajectory bucket proxy，因此只能用于同口径模型比较。",
            "",
            "## 输出文件",
            "",
            f"- selector predictions: `{summary['outputs']['selector_predictions']}`",
            f"- label metric rows: `{summary['outputs']['label_metric_rows']}`",
            f"- problem-stage summary: `{summary['outputs']['problem_stage_summary']}`",
            f"- label metric summary: `{summary['outputs']['label_metric_summary']}`",
            f"- model summary: `{summary['outputs']['selector_model_summary']}`",
            f"- summary: `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _value_counts_string(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H5 model capacity diagnostic for selection_reference generalization.")
    parser.add_argument(
        "--train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train/utility_labels.parquet"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument(
        "--extension-train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train_late_stage_f024_followup/utility_labels_fe050.parquet"),
    )
    parser.add_argument(
        "--selection-reference",
        type=Path,
        default=Path("results/selection_reference/min_support_bbob_train/selection_reference.parquet"),
    )
    parser.add_argument(
        "--extension-selection-reference",
        type=Path,
        default=Path("results/selection_reference/min_support_bbob_train_late_stage_f024_followup/selection_reference.parquet"),
    )
    parser.add_argument(
        "--train-ela-features",
        type=Path,
        default=Path("results/ela/min_support_bbob_train/features.parquet"),
    )
    parser.add_argument(
        "--validation-ela-features",
        type=Path,
        default=Path("results/ela/min_support_bbob_validation/features.parquet"),
    )
    parser.add_argument(
        "--extension-ela-features",
        type=Path,
        default=Path("results/ela/min_support_bbob_train_late_stage_f024_followup/features.parquet"),
    )
    parser.add_argument(
        "--train-trajectory-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_train"),
    )
    parser.add_argument(
        "--validation-trajectory-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_validation"),
    )
    parser.add_argument(
        "--extension-trajectory-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_train_late_stage_f024_followup"),
    )
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=Path("configs/min_support_bbob_train_late_stage_f024_followup.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/selection_reference_generalization_data_quality/h5_model_capacity"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--random-seed", type=int, default=1701)
    args = parser.parse_args()
    run_h5_model_capacity_diagnostic(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        extension_train_labels_path=args.extension_train_labels,
        selection_reference_path=args.selection_reference,
        extension_selection_reference_path=args.extension_selection_reference,
        train_ela_features_path=args.train_ela_features,
        validation_ela_features_path=args.validation_ela_features,
        extension_ela_features_path=args.extension_ela_features,
        train_trajectory_root=args.train_trajectory_root,
        validation_trajectory_root=args.validation_trajectory_root,
        extension_trajectory_root=args.extension_trajectory_root,
        extension_config_path=args.extension_config,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
