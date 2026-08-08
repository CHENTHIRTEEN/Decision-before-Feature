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
from sklearn.pipeline import Pipeline

from decision.min_support_diagnostics import _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_family_split, _check_target, _json_default, _read_labels
from ela.features import ELA_FEATURE_COLUMNS


TARGET_FAMILIES = ("bbob_f005", "bbob_f019", "bbob_f024")
SELECTOR_VARIANTS = ("full_label", "changed_only", "changed_weighted")
CHANGED_WEIGHT = 5.0
FE_KEY_DECIMALS = 6


def run_h3_label_source_diagnostic(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    selection_reference_path: Path,
    train_ela_features_path: Path,
    validation_ela_features_path: Path,
    train_trajectory_root: Path,
    validation_trajectory_root: Path,
    output_dir: Path,
    target_column: str,
    changed_weight: float,
    random_seed: int,
) -> dict[str, Any]:
    _check_target(target_column)
    train_labels = _read_labels(train_labels_path).assign(dataset_role="train")
    validation_labels = _read_labels(validation_labels_path).assign(dataset_role="validation")
    _check_family_split(train_labels, validation_labels)

    selection = _read_selection_reference(selection_reference_path)
    features = pd.concat(
        [
            _read_ela_features(train_ela_features_path).assign(dataset_role="train"),
            _read_ela_features(validation_ela_features_path).assign(dataset_role="validation"),
        ],
        ignore_index=True,
    )
    label_support = _label_support(train_labels, validation_labels, target_column)
    train_samples = _training_samples(selection, features, label_support)
    predictions, model_summary = _variant_predictions(
        train_samples=train_samples,
        selection=selection,
        features=features,
        changed_weight=changed_weight,
        random_seed=random_seed,
    )

    metric_rows = _label_metric_rows(
        labels=pd.concat([train_labels, validation_labels], ignore_index=True),
        predictions=predictions,
        train_trajectory_root=train_trajectory_root,
        validation_trajectory_root=validation_trajectory_root,
        target_column=target_column,
    )
    problem_stage_summary = _problem_stage_summary(predictions)
    label_metric_summary = _label_metric_summary(metric_rows, target_column)
    conclusion = _diagnostic_conclusion(problem_stage_summary, label_metric_summary, model_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples_path = output_dir / "h3_training_samples.parquet"
    predictions_path = output_dir / "h3_selector_predictions.parquet"
    model_summary_path = output_dir / "h3_selector_model_summary.parquet"
    metric_rows_path = output_dir / "h3_label_metric_rows.parquet"
    problem_stage_summary_path = output_dir / "h3_problem_stage_summary.parquet"
    label_metric_summary_path = output_dir / "h3_label_metric_summary.parquet"
    summary_path = output_dir / "h3_label_source_selector_diagnostic_summary.json"
    report_path = output_dir / "h3_label_source_selector_diagnostic_report.md"

    pq.write_table(pa.Table.from_pandas(train_samples, preserve_index=False), train_samples_path)
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), predictions_path)
    pq.write_table(pa.Table.from_pandas(model_summary, preserve_index=False), model_summary_path)
    pq.write_table(pa.Table.from_pandas(metric_rows, preserve_index=False), metric_rows_path)
    pq.write_table(pa.Table.from_pandas(problem_stage_summary, preserve_index=False), problem_stage_summary_path)
    pq.write_table(pa.Table.from_pandas(label_metric_summary, preserve_index=False), label_metric_summary_path)

    summary = {
        "experiment": "selection_reference_h3_label_source_selector_diagnostic",
        "research_question": (
            "Does separating same_algorithm and changed_algorithm label sources alter the apparent "
            "selection_reference generalization quality?"
        ),
        "target_column": target_column,
        "selector_variants": {
            "full_label": "stage-wise RF selector trained with every train problem-stage VBS target",
            "changed_only": "stage-wise RF or constant selector trained only on train problem-stages where selected_algorithm differs from default_algorithm",
            "changed_weighted": f"stage-wise RF selector trained on every train problem-stage with changed_algorithm sample weight {changed_weight:g}",
        },
        "inputs": {
            "train_labels": str(train_labels_path),
            "validation_labels": str(validation_labels_path),
            "selection_reference": str(selection_reference_path),
            "train_ela_features": str(train_ela_features_path),
            "validation_ela_features": str(validation_ela_features_path),
            "train_trajectory_root": str(train_trajectory_root),
            "validation_trajectory_root": str(validation_trajectory_root),
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
            "formal_phase1_configs_modified": False,
            "utility_labels_regenerated": False,
            "diagnostic_selector_models_trained": True,
            "diagnostic_models_written_as_formal_phase1": False,
            "decision_input_uses_ela_features": True,
            "ela_features_used_only_for_fixed_downstream_selector_diagnostic": True,
        },
        "notes": [
            "P_ELA and U_ELA are exact observed label values only when a diagnostic selector chooses the same algorithm as the existing utility label row.",
            "For all rows, bucket_proxy_p_ela and bucket_proxy_u use existing trajectory-bucket performance only; no alternate utility labels are generated.",
            "same_algorithm rows are retained as continuation randomness control and are reported separately from changed_algorithm rows.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown_report(summary, problem_stage_summary, label_metric_summary, model_summary), encoding="utf-8")

    print(f"wrote H3 training samples to {train_samples_path}")
    print(f"wrote H3 selector predictions to {predictions_path}")
    print(f"wrote H3 selector model summary to {model_summary_path}")
    print(f"wrote H3 label metric rows to {metric_rows_path}")
    print(f"wrote H3 problem-stage summary to {problem_stage_summary_path}")
    print(f"wrote H3 label metric summary to {label_metric_summary_path}")
    print(f"wrote H3 summary to {summary_path}")
    print(f"wrote H3 report to {report_path}")
    return summary


def _read_selection_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing selection_reference: {path}")
    frame = pq.read_table(path).to_pandas()
    frame["dataset_role"] = np.where(frame["split"].astype(str).str.contains("validation"), "validation", "train")
    frame["remaining_budget_key"] = frame["remaining_budget_ratio"].round(FE_KEY_DECIMALS)
    return frame


def _read_ela_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing ELA features: {path}")
    frame = pq.read_table(path).to_pandas()
    failed = frame[frame["feature_status"] != "ok"]
    if not failed.empty:
        raise ValueError(f"ELA feature file contains failed rows: {path}")
    return frame


def _label_support(train_labels: pd.DataFrame, validation_labels: pd.DataFrame, target_column: str) -> pd.DataFrame:
    labels = pd.concat([train_labels, validation_labels], ignore_index=True).copy()
    labels["remaining_budget_key"] = (labels["FE_ela_optimization"] / labels["FE_total"]).round(FE_KEY_DECIMALS)
    labels["label_source"] = np.where(
        labels["selected_algorithm"].astype(str) == labels["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    labels["utility_gt_zero"] = labels[target_column] > 0.0
    grouped = []
    key_columns = ["dataset_role", "split", "problem_id", "family", "dimension", "remaining_budget_key"]
    for key, group in labels.groupby(key_columns, sort=True, dropna=False):
        values = group[target_column].to_numpy(dtype=float)
        changed_rows = group[group["label_source"] == "changed_algorithm"]
        same_rows = group[group["label_source"] == "same_algorithm"]
        grouped.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "label_rows": int(len(group)),
                "same_algorithm_rows": int(len(same_rows)),
                "changed_algorithm_rows": int(len(changed_rows)),
                "changed_algorithm_rate": float(len(changed_rows) / len(group)) if len(group) else 0.0,
                "training_label_source": "changed_algorithm" if len(changed_rows) > 0 else "same_algorithm",
                "utility_gt_zero_rows": int((values > 0.0).sum()),
                "utility_gt_zero_rate": float((values > 0.0).mean()) if len(group) else 0.0,
                "utility_sum": float(values.sum()) if len(group) else 0.0,
                "positive_utility_sum": float(values[values > 0.0].sum()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(grouped)


def _training_samples(selection: pd.DataFrame, features: pd.DataFrame, label_support: pd.DataFrame) -> pd.DataFrame:
    train_selection = selection[selection["dataset_role"] == "train"].copy()
    train_features = features[features["dataset_role"] == "train"].copy()
    samples = train_selection.merge(
        train_features[["problem_id", "dimension", *ELA_FEATURE_COLUMNS]],
        on=["problem_id", "dimension"],
        how="left",
        validate="many_to_one",
    )
    samples = samples.merge(
        label_support[
            [
                "split",
                "problem_id",
                "dimension",
                "remaining_budget_key",
                "label_rows",
                "same_algorithm_rows",
                "changed_algorithm_rows",
                "changed_algorithm_rate",
                "training_label_source",
                "utility_gt_zero_rows",
                "utility_gt_zero_rate",
                "positive_utility_sum",
            ]
        ],
        on=["split", "problem_id", "dimension", "remaining_budget_key"],
        how="left",
        validate="one_to_one",
    )
    missing_features = samples[list(ELA_FEATURE_COLUMNS)].isna().any(axis=1)
    if missing_features.any():
        missing = samples.loc[missing_features, ["problem_id", "dimension"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"training samples missing ELA features: {missing}")
    samples["training_label_source"] = samples["training_label_source"].fillna("same_algorithm")
    samples["label_rows"] = samples["label_rows"].fillna(0).astype(int)
    samples["same_algorithm_rows"] = samples["same_algorithm_rows"].fillna(0).astype(int)
    samples["changed_algorithm_rows"] = samples["changed_algorithm_rows"].fillna(0).astype(int)
    samples["changed_algorithm_rate"] = samples["changed_algorithm_rate"].fillna(0.0)
    return samples


def _variant_predictions(
    *,
    train_samples: pd.DataFrame,
    selection: pd.DataFrame,
    features: pd.DataFrame,
    changed_weight: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    model_rows = []
    feature_lookup = features[["dataset_role", "split", "problem_id", "family", "dimension", *ELA_FEATURE_COLUMNS]].copy()
    prediction_base = selection.merge(
        feature_lookup,
        on=["dataset_role", "split", "problem_id", "family", "dimension"],
        how="left",
        validate="many_to_one",
    )
    if prediction_base[list(ELA_FEATURE_COLUMNS)].isna().any(axis=1).any():
        raise ValueError("prediction rows are missing ELA features")

    sbs_algorithm = _sbs_algorithm(train_samples)
    for remaining_budget, stage_train in train_samples.groupby("remaining_budget_key", sort=True):
        stage_predict = prediction_base[prediction_base["remaining_budget_key"] == remaining_budget].copy()
        for variant in SELECTOR_VARIANTS:
            fit_rows, sample_weight = _fit_rows(stage_train, variant, changed_weight)
            model, status = _fit_selector_variant(
                fit_rows=fit_rows,
                sample_weight=sample_weight,
                fallback=sbs_algorithm,
                random_seed=random_seed,
            )
            selected = _predict_selector_variant(model, status, stage_predict, fallback=sbs_algorithm)
            variant_frame = stage_predict[
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
            variant_frame["selector_variant"] = variant
            variant_frame["diagnostic_selector_status"] = status
            variant_frame["diagnostic_selected_algorithm"] = selected
            variant_frame["diagnostic_selected_differs_from_default"] = (
                variant_frame["diagnostic_selected_algorithm"].astype(str) != variant_frame["default_algorithm"].astype(str)
            )
            variant_frame["diagnostic_selected_matches_vbs"] = (
                variant_frame["diagnostic_selected_algorithm"].astype(str) == variant_frame["vbs_algorithm"].astype(str)
            )
            prediction_frames.append(variant_frame)
            model_rows.append(
                {
                    "remaining_budget_key": float(remaining_budget),
                    "selector_variant": variant,
                    "diagnostic_selector_status": status,
                    "train_samples_available": int(len(stage_train)),
                    "train_samples_used": int(len(fit_rows)),
                    "train_changed_samples_available": int((stage_train["training_label_source"] == "changed_algorithm").sum()),
                    "train_same_samples_available": int((stage_train["training_label_source"] == "same_algorithm").sum()),
                    "target_algorithm_counts": _value_counts_string(fit_rows["vbs_algorithm"]) if len(fit_rows) else "",
                    "selected_algorithm_counts_train_prediction": "",
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
        train_pred_counts, on=["remaining_budget_key", "selector_variant"], how="left"
    )
    return predictions, model_summary


def _sbs_algorithm(train_samples: pd.DataFrame) -> str:
    counts = train_samples["default_algorithm"].astype(str).value_counts()
    if counts.empty:
        return "cmaes"
    return str(counts.index[0])


def _fit_rows(stage_train: pd.DataFrame, variant: str, changed_weight: float) -> tuple[pd.DataFrame, np.ndarray | None]:
    if variant == "full_label":
        return stage_train.copy(), None
    if variant == "changed_only":
        return stage_train[stage_train["training_label_source"] == "changed_algorithm"].copy(), None
    if variant == "changed_weighted":
        rows = stage_train.copy()
        weights = np.where(rows["training_label_source"] == "changed_algorithm", float(changed_weight), 1.0)
        return rows, weights.astype(float)
    raise ValueError(f"unknown selector variant: {variant}")


def _fit_selector_variant(
    *,
    fit_rows: pd.DataFrame,
    sample_weight: np.ndarray | None,
    fallback: str,
    random_seed: int,
) -> tuple[Pipeline | str, str]:
    if fit_rows.empty:
        return fallback, "fallback_no_training_rows"
    original_index = fit_rows.index
    y = fit_rows["vbs_algorithm"].fillna("").astype(str)
    y = y[y != ""]
    fit_rows = fit_rows.loc[y.index]
    if fit_rows.empty:
        return fallback, "fallback_no_vbs_targets"
    if len(set(y)) <= 1:
        return str(y.iloc[0]), "constant"
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_seed,
                    class_weight="balanced",
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    if sample_weight is None:
        model.fit(fit_rows[list(ELA_FEATURE_COLUMNS)], y)
    else:
        weights = pd.Series(sample_weight, index=original_index).loc[fit_rows.index].to_numpy(dtype=float)
        model.fit(fit_rows[list(ELA_FEATURE_COLUMNS)], y, classifier__sample_weight=weights)
    return model, "random_forest"


def _predict_selector_variant(model: Pipeline | str, status: str, rows: pd.DataFrame, fallback: str) -> np.ndarray:
    if rows.empty:
        return np.asarray([], dtype=str)
    if status in {"constant", "fallback_no_training_rows", "fallback_no_vbs_targets"}:
        return np.full(len(rows), str(model), dtype=object)
    if not isinstance(model, Pipeline):
        return np.full(len(rows), fallback, dtype=object)
    return model.predict(rows[list(ELA_FEATURE_COLUMNS)]).astype(str)


def _label_metric_rows(
    *,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    train_trajectory_root: Path,
    validation_trajectory_root: Path,
    target_column: str,
) -> pd.DataFrame:
    labels = labels.copy()
    labels["remaining_budget_key"] = (labels["FE_ela_optimization"] / labels["FE_total"]).round(FE_KEY_DECIMALS)
    labels["label_source"] = np.where(
        labels["selected_algorithm"].astype(str) == labels["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
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
        metric_rows["observed_label_available"], metric_rows["p_ela"], np.nan
    )
    metric_rows["observed_u_for_diagnostic_selector"] = np.where(
        metric_rows["observed_label_available"], metric_rows[target_column], np.nan
    )
    proxy = _trajectory_proxy_table(
        labels=labels,
        train_trajectory_root=train_trajectory_root,
        validation_trajectory_root=validation_trajectory_root,
    )
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
    metric_rows["target_family"] = metric_rows["family"].isin(TARGET_FAMILIES)
    return metric_rows


def _trajectory_proxy_table(
    *,
    labels: pd.DataFrame,
    train_trajectory_root: Path,
    validation_trajectory_root: Path,
) -> pd.DataFrame:
    keys = labels[["dataset_role", "family", "problem_id", "dimension"]].drop_duplicates()
    frames = []
    for dataset_role, root in (("train", train_trajectory_root), ("validation", validation_trajectory_root)):
        role_keys = keys[keys["dataset_role"] == dataset_role]
        for (family, dimension), _ in role_keys.groupby(["family", "dimension"], sort=True):
            path = root / str(family) / f"dimension_{int(dimension)}" / "trajectories.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing trajectory shard: {path}")
            table = pq.read_table(path, columns=["problem_id", "dimension", "algorithm", "FE_ratio", "best_fitness"])
            frame = table.to_pandas()
            frame["dataset_role"] = dataset_role
            frames.append(frame)
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
        for domain_name, domain_rows in _evaluation_domains(variant_rows):
            rows.append(_problem_stage_summary_row(selector_variant, domain_name, "all", domain_rows))
            for family, family_rows in domain_rows.groupby("family", sort=True):
                if domain_name == "validation_all" and family in TARGET_FAMILIES:
                    rows.append(_problem_stage_summary_row(selector_variant, f"validation_{family}", "family", family_rows))
    return pd.DataFrame(rows)


def _problem_stage_summary_row(
    selector_variant: str,
    evaluation_domain: str,
    layer: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "selector_variant": selector_variant,
        "evaluation_domain": evaluation_domain,
        "layer": layer,
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
        for domain_name, domain_rows in _evaluation_domains(variant_rows):
            for label_source in ("all", "same_algorithm", "changed_algorithm"):
                source_rows = domain_rows if label_source == "all" else domain_rows[domain_rows["label_source"] == label_source]
                rows.append(_label_metric_summary_row(selector_variant, domain_name, label_source, source_rows, target_column))
            if domain_name == "validation_all":
                for family in TARGET_FAMILIES:
                    family_rows = domain_rows[domain_rows["family"] == family]
                    rows.append(_label_metric_summary_row(selector_variant, f"validation_{family}", "all", family_rows, target_column))
                    for label_source in ("same_algorithm", "changed_algorithm"):
                        rows.append(
                            _label_metric_summary_row(
                                selector_variant,
                                f"validation_{family}",
                                label_source,
                                family_rows[family_rows["label_source"] == label_source],
                                target_column,
                            )
                        )
    return pd.DataFrame(rows)


def _label_metric_summary_row(
    selector_variant: str,
    evaluation_domain: str,
    label_source: str,
    frame: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    observed = frame[frame["observed_label_available"]]
    return {
        "selector_variant": selector_variant,
        "evaluation_domain": evaluation_domain,
        "label_source": label_source,
        "rows": int(len(frame)),
        "problem_count": int(frame["problem_id"].nunique()) if len(frame) else 0,
        "observed_label_rows": int(len(observed)),
        "observed_label_coverage_rate": float(len(observed) / len(frame)) if len(frame) else 0.0,
        "selected_matches_vbs_rate": float(frame["diagnostic_selected_matches_vbs"].mean()) if len(frame) else 0.0,
        "p_ela_mean_observed_when_available": float(observed["p_ela"].mean()) if len(observed) else None,
        "u_mean_observed_when_available": float(observed[target_column].mean()) if len(observed) else None,
        "u_gt_zero_rate_observed_when_available": float((observed[target_column] > 0.0).mean()) if len(observed) else None,
        "bucket_proxy_p_ela_mean": float(frame["bucket_proxy_p_ela"].mean()) if len(frame) else None,
        "bucket_proxy_u_mean": float(frame["bucket_proxy_u"].mean()) if len(frame) else None,
        "bucket_proxy_u_gt_zero_rate": float((frame["bucket_proxy_u"] > 0.0).mean()) if len(frame) else 0.0,
        "original_p_ela_mean": float(frame["p_ela"].mean()) if len(frame) else None,
        "original_u_mean": float(frame[target_column].mean()) if len(frame) else None,
        "diagnostic_selected_algorithm_counts": _value_counts_string(frame["diagnostic_selected_algorithm"]),
        "original_selected_algorithm_counts": _value_counts_string(frame["selected_algorithm"]),
    }


def _evaluation_domains(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("train_all", frame[frame["dataset_role"] == "train"]),
        ("validation_all", frame[frame["dataset_role"] == "validation"]),
    ]


def _diagnostic_conclusion(
    problem_stage_summary: pd.DataFrame,
    label_metric_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> dict[str, Any]:
    validation_problem = problem_stage_summary[
        (problem_stage_summary["evaluation_domain"] == "validation_all") & (problem_stage_summary["layer"] == "all")
    ].set_index("selector_variant")
    validation_labels = label_metric_summary[
        (label_metric_summary["evaluation_domain"] == "validation_all") & (label_metric_summary["label_source"] == "changed_algorithm")
    ].set_index("selector_variant")
    changed_only_fallback_stages = int(
        (
            (model_summary["selector_variant"] == "changed_only")
            & (model_summary["diagnostic_selector_status"].astype(str).str.startswith("fallback"))
        ).sum()
    )
    return {
        "validation_selected_matches_vbs_rate": {
            variant: float(validation_problem.loc[variant, "selected_matches_vbs_rate"]) for variant in validation_problem.index
        },
        "validation_changed_algorithm_bucket_proxy_u_mean": {
            variant: float(validation_labels.loc[variant, "bucket_proxy_u_mean"]) for variant in validation_labels.index
        },
        "validation_changed_algorithm_observed_label_coverage_rate": {
            variant: float(validation_labels.loc[variant, "observed_label_coverage_rate"]) for variant in validation_labels.index
        },
        "changed_only_fallback_stage_count": changed_only_fallback_stages,
        "interpretation": (
            "The H3 diagnostic separates label sources without changing utility labels. Differences between "
            "full_label, changed_only, and changed_weighted indicate how much same_algorithm continuation rows "
            "affect the fixed ELA selector training signal. Bucket proxy utility is used when the diagnostic "
            "selector chooses an algorithm not present in the original utility-label row."
        ),
    }


def _markdown_report(
    summary: dict[str, Any],
    problem_stage_summary: pd.DataFrame,
    label_metric_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    problem = problem_stage_summary[
        (problem_stage_summary["evaluation_domain"].isin(["train_all", "validation_all"]))
        & (problem_stage_summary["layer"] == "all")
    ].sort_values(["evaluation_domain", "selector_variant"])
    validation_changed = label_metric_summary[
        (label_metric_summary["evaluation_domain"] == "validation_all")
        & (label_metric_summary["label_source"] == "changed_algorithm")
    ].sort_values("selector_variant")
    target = label_metric_summary[
        label_metric_summary["evaluation_domain"].isin([f"validation_{family}" for family in TARGET_FAMILIES])
        & (label_metric_summary["label_source"] == "all")
    ].sort_values(["evaluation_domain", "selector_variant"])
    fallback_count = int(
        (
            (model_summary["selector_variant"] == "changed_only")
            & (model_summary["diagnostic_selector_status"].astype(str).str.startswith("fallback"))
        ).sum()
    )
    lines = [
        "# selection_reference 泛化失败 H3 最小诊断",
        "",
        "本报告只使用当前项目内已有 `utility_labels`、`selection_reference`、ELA features 和 trajectories；未修改原始结果，未重训正式模型，未生成新的 utility labels。",
        "",
        "## Problem-stage selected=VBS",
        "",
        "| selector_variant | domain | selected=VBS | changed selection rate | selected counts |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in problem.iterrows():
        lines.append(
            f"| `{row['selector_variant']}` | `{row['evaluation_domain']}` | "
            f"{row['selected_matches_vbs_rate']:.6f} | {row['changed_selection_rate']:.6f} | "
            f"`{row['diagnostic_selected_algorithm_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Validation changed_algorithm label rows",
            "",
            "| selector_variant | observed coverage | observed U mean | bucket proxy P_ELA | bucket proxy U | bucket proxy U>0 rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in validation_changed.iterrows():
        observed_u = "NA" if pd.isna(row["u_mean_observed_when_available"]) else f"{row['u_mean_observed_when_available']:.6f}"
        lines.append(
            f"| `{row['selector_variant']}` | {row['observed_label_coverage_rate']:.6f} | {observed_u} | "
            f"{row['bucket_proxy_p_ela_mean']:.6f} | {row['bucket_proxy_u_mean']:.6f} | "
            f"{row['bucket_proxy_u_gt_zero_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Target validation families",
            "",
            "| family domain | selector_variant | rows | selected=VBS | bucket proxy P_ELA | bucket proxy U |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in target.iterrows():
        lines.append(
            f"| `{row['evaluation_domain']}` | `{row['selector_variant']}` | {int(row['rows'])} | "
            f"{row['selected_matches_vbs_rate']:.6f} | {row['bucket_proxy_p_ela_mean']:.6f} | "
            f"{row['bucket_proxy_u_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 诊断边界",
            "",
            f"- changed_only fallback stages: `{fallback_count}`",
            "- `observed U mean` 只在诊断 selector 选择的算法与原始 utility label 行一致时可解释为真实观测。",
            "- `bucket proxy P_ELA/U` 是现有 trajectory bucket 上的性能代理，不是重新生成的 utility label。",
            "",
            "## 输出文件",
            "",
            f"- selector predictions: `{summary['outputs']['selector_predictions']}`",
            f"- label metric summary: `{summary['outputs']['label_metric_summary']}`",
            f"- problem-stage summary: `{summary['outputs']['problem_stage_summary']}`",
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
    parser = argparse.ArgumentParser(description="Run H3 same/changed label-source diagnostic for selection_reference selector.")
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
        "--selection-reference",
        type=Path,
        default=Path("results/selection_reference/min_support_bbob_train/selection_reference.parquet"),
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
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/selection_reference_generalization_data_quality/h3_label_source"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--changed-weight", type=float, default=CHANGED_WEIGHT)
    parser.add_argument("--random-seed", type=int, default=1701)
    args = parser.parse_args()
    run_h3_label_source_diagnostic(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        selection_reference_path=args.selection_reference,
        train_ela_features_path=args.train_ela_features,
        validation_ela_features_path=args.validation_ela_features,
        train_trajectory_root=args.train_trajectory_root,
        validation_trajectory_root=args.validation_trajectory_root,
        output_dir=args.output_dir,
        target_column=args.target_column,
        changed_weight=args.changed_weight,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
