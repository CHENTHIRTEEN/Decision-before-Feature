from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision.cluster_weighting import (
    WeightedLinearDiscriminantAnalysis,
    WeightedMedianImputer,
)


ACTIVE_MODEL_NAMES = (
    "lda_classifier",
    "logistic_regression_classifier",
    "ridge_regression",
)
MODEL_SELECTION_METRIC = "nested_family_oof_first_trigger_decision_mean_utility"
SELECTED_MODEL_ALIAS = "selected"
FROZEN_THRESHOLD_MODE = "oof_utility_first_trigger"
BEHAVIOR_FROZEN_THRESHOLD_MODE = "oof_behavior_utility_first_trigger"
OUTER_OOF_FOLDS = 5
INNER_OOF_FOLDS = 4
FULL_TRAIN_OOF_FOLDS = 5
THRESHOLD_NEIGHBORHOOD_QUANTILE = 0.10


@dataclass(frozen=True)
class DecisionModelSpec:
    model_name: str
    model_family: str
    estimator_name: str
    estimator: BaseEstimator
    objective: str

    @property
    def supports_utility_rmse(self) -> bool:
        return self.objective == "regression"


def active_model_specs(random_seed: int) -> tuple[DecisionModelSpec, ...]:
    logistic_seed = int(
        np.random.SeedSequence([int(random_seed), 20260811, 31]).generate_state(1, dtype=np.uint32)[0]
    )
    specs = (
        DecisionModelSpec(
            model_name="lda_classifier",
            model_family="lda",
            estimator_name="LinearDiscriminantAnalysis(weighted fit-fold estimates)",
            estimator=Pipeline(
                [
                    ("imputer", WeightedMedianImputer()),
                    ("scaler", StandardScaler()),
                    ("classifier", WeightedLinearDiscriminantAnalysis()),
                ]
            ),
            objective="classification",
        ),
        DecisionModelSpec(
            model_name="logistic_regression_classifier",
            model_family="logistic_regression",
            estimator_name="LogisticRegression(class_weight=None,C=1.0)",
            estimator=Pipeline(
                [
                    ("imputer", WeightedMedianImputer()),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0,
                            class_weight=None,
                            max_iter=1000,
                            solver="lbfgs",
                            random_state=logistic_seed,
                        ),
                    ),
                ]
            ),
            objective="classification",
        ),
        DecisionModelSpec(
            model_name="ridge_regression",
            model_family="ridge",
            estimator_name="Ridge(alpha=1.0)",
            estimator=Pipeline(
                [
                    ("imputer", WeightedMedianImputer()),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0)),
                ]
            ),
            objective="regression",
        ),
    )
    observed = tuple(spec.model_name for spec in specs)
    if observed != ACTIVE_MODEL_NAMES:
        raise RuntimeError(f"active Decision model order does not match the frozen protocol: {observed}")
    return specs


def decision_scores(model: BaseEstimator, features: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(getattr(model, "classes_", []))
        positive_index = np.flatnonzero(classes == 1)
        if probabilities.ndim != 2 or len(positive_index) != 1:
            raise ValueError("classification Decision model must expose exactly one positive-class probability")
        scores = probabilities[:, int(positive_index[0])]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
    else:
        scores = np.asarray(model.predict(features), dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Decision model produced non-finite scores")
    return scores


def resolve_model_name(training_summary: dict[str, Any], requested_model_name: str) -> str:
    requested = str(requested_model_name)
    if requested == SELECTED_MODEL_ALIAS:
        requested = str(training_summary.get("selected_model_name", ""))
        if not requested:
            raise ValueError("Decision training summary does not define selected_model_name")
    trained = tuple(str(name) for name in training_summary.get("models_trained", []))
    if trained != ACTIVE_MODEL_NAMES:
        raise ValueError(
            "Decision training summary does not match the active three-model protocol: "
            f"expected={ACTIVE_MODEL_NAMES}, observed={trained}"
        )
    if requested not in ACTIVE_MODEL_NAMES:
        raise ValueError(f"model is not an active Decision candidate: {requested}")
    return requested
