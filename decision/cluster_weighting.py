from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline


CLUSTER_BALANCED_FIT = "cluster_balanced_fit"
ROW_WEIGHTED_FIT = "row_weighted_fit"
EPS = 1.0e-12


def cluster_balanced_row_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give equal mass to function, dimension, problem, run, then run states."""
    required = {"function_id", "dimension", "problem_id", "seed"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cluster-balanced fit is missing grouping columns: {missing}")
    if frame.empty:
        raise ValueError("cluster-balanced fit requires at least one row")

    function_count = int(frame["function_id"].astype(str).nunique())
    dimensions_per_function = frame.groupby("function_id", dropna=False)["dimension"].transform(
        "nunique"
    )
    problem_group = ["function_id", "dimension"]
    problems_per_stratum = frame.groupby(problem_group, dropna=False)["problem_id"].transform(
        "nunique"
    )
    run_key = ["function_id", "dimension", "problem_id"]
    if "prefix_algorithm" in frame.columns:
        run_key.append("prefix_algorithm")
    run_key.append("seed")
    first_row_of_run = ~frame.duplicated(run_key)
    runs_per_problem = first_row_of_run.groupby(
        [frame["function_id"], frame["dimension"], frame["problem_id"]],
        dropna=False,
    ).transform("sum")
    states_per_run = frame.groupby(run_key, dropna=False)["function_id"].transform("size")

    denominators = (
        float(function_count)
        * dimensions_per_function.to_numpy(dtype=float)
        * problems_per_stratum.to_numpy(dtype=float)
        * runs_per_problem.to_numpy(dtype=float)
        * states_per_run.to_numpy(dtype=float)
    )
    if not np.isfinite(denominators).all() or bool((denominators <= 0.0).any()):
        raise ValueError("cluster-balanced fit produced invalid hierarchy denominators")
    weights = 1.0 / denominators
    weights /= float(np.mean(weights))
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("cluster-balanced row weights must be finite and positive")
    return weights


def row_weighted_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        raise ValueError("row-weighted fit requires at least one row")
    return np.ones(len(frame), dtype=float)


class WeightedMedianImputer(TransformerMixin, BaseEstimator):
    """Columnwise median imputation under the fit-fold scientific row weights."""

    strategy = "weighted_median"

    def fit(
        self,
        x: Any,
        y: Any = None,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedMedianImputer":
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("weighted median imputation requires a two-dimensional matrix")
        weights = (
            np.ones(values.shape[0], dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if len(weights) != values.shape[0]:
            raise ValueError("imputation sample weights do not match the training rows")
        if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
            raise ValueError("imputation sample weights must be finite and positive")
        statistics: list[float] = []
        for column_index in range(values.shape[1]):
            column = values[:, column_index]
            observed = np.isfinite(column)
            if not bool(observed.any()):
                raise ValueError(
                    f"fit-fold input column {column_index} is entirely missing or non-finite"
                )
            observed_values = column[observed]
            observed_weights = weights[observed]
            order = np.argsort(observed_values, kind="mergesort")
            sorted_values = observed_values[order]
            sorted_weights = observed_weights[order]
            cutoff = 0.5 * float(np.sum(sorted_weights))
            median_index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
            statistics.append(float(sorted_values[min(median_index, len(sorted_values) - 1)]))
        self.statistics_ = np.asarray(statistics, dtype=float)
        self.n_features_in_ = int(values.shape[1])
        return self

    def transform(self, x: Any) -> np.ndarray:
        values = np.asarray(x, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("imputation input width does not match the fitted feature width")
        missing = ~np.isfinite(values)
        if bool(missing.any()):
            values[missing] = np.take(self.statistics_, np.nonzero(missing)[1])
        return values


class WeightedLinearDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Binary LDA with fit-fold sample weights in priors, means, and covariance."""

    def __init__(self, covariance_regularization: float = 1.0e-9) -> None:
        self.covariance_regularization = float(covariance_regularization)

    def fit(
        self,
        x: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedLinearDiscriminantAnalysis":
        values = np.asarray(x, dtype=float)
        labels = np.asarray(y).reshape(-1)
        weights = (
            np.ones(len(values), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if values.ndim != 2 or len(labels) != len(values) or len(weights) != len(values):
            raise ValueError("weighted LDA inputs have inconsistent shapes")
        if not np.isfinite(values).all() or not np.isfinite(weights).all():
            raise ValueError("weighted LDA inputs must be finite after imputation")
        if bool((weights <= 0.0).any()):
            raise ValueError("weighted LDA sample weights must be positive")
        classes = np.unique(labels)
        if not np.array_equal(classes, np.asarray([0, 1])):
            raise ValueError("weighted LDA requires both binary classes 0 and 1")

        class_weights = np.asarray(
            [float(np.sum(weights[labels == value])) for value in classes],
            dtype=float,
        )
        means = np.vstack(
            [
                np.average(values[labels == value], axis=0, weights=weights[labels == value])
                for value in classes
            ]
        )
        covariance = np.zeros((values.shape[1], values.shape[1]), dtype=float)
        for class_index, value in enumerate(classes):
            centered = values[labels == value] - means[class_index]
            covariance += (centered * weights[labels == value, None]).T @ centered
        covariance /= max(float(np.sum(weights)), EPS)
        scale = max(float(np.trace(covariance)) / max(values.shape[1], 1), 1.0)
        covariance += np.eye(values.shape[1], dtype=float) * (
            self.covariance_regularization * scale
        )
        inverse_covariance = np.linalg.pinv(covariance, hermitian=True)
        coefficient = inverse_covariance @ (means[1] - means[0])
        priors = class_weights / float(np.sum(class_weights))
        intercept = -0.5 * float((means[1] + means[0]) @ coefficient) + float(
            np.log(priors[1] / priors[0])
        )

        self.classes_ = classes
        self.coef_ = coefficient.reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=float)
        self.means_ = means
        self.priors_ = priors
        self.covariance_ = covariance
        self.n_features_in_ = int(values.shape[1])
        return self

    def decision_function(self, x: Any) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        return values @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, x: Any) -> np.ndarray:
        scores = self.decision_function(x)
        positive = np.empty_like(scores, dtype=float)
        nonnegative = scores >= 0.0
        positive[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
        exponential = np.exp(scores[~nonnegative])
        positive[~nonnegative] = exponential / (1.0 + exponential)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: Any) -> np.ndarray:
        return (self.decision_function(x) >= 0.0).astype(int)


def fit_pipeline_with_weights(
    model: Pipeline,
    x: Any,
    y: Any,
    sample_weight: np.ndarray,
) -> Pipeline:
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(weights) != len(x):
        raise ValueError("fit weights do not match the training rows")
    fit_parameters: dict[str, np.ndarray] = {}
    if "imputer" in model.named_steps:
        fit_parameters["imputer__sample_weight"] = weights
    if "scaler" in model.named_steps:
        fit_parameters["scaler__sample_weight"] = weights
    final_step = tuple(model.named_steps)[-1]
    fit_parameters[f"{final_step}__sample_weight"] = weights
    model.fit(x, y, **fit_parameters)
    return model
