from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    "random_forest_classifier",
)
MODEL_SELECTION_METRIC = "nested_cv_group_oof_first_trigger_mean_g_fe"
SELECTED_MODEL_ALIAS = "selected"
FROZEN_THRESHOLD_MODE = "oof_g_fe_first_trigger"
BEHAVIOR_FROZEN_THRESHOLD_MODE = "oof_behavior_g_fe_first_trigger"
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
        DecisionModelSpec(
            model_name="random_forest_classifier",
            model_family="random_forest",
            estimator_name="RandomForestClassifier(n_estimators=200,max_depth=8,max_features=sqrt)",
            estimator=Pipeline(
                [
                    ("imputer", WeightedMedianImputer()),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=8,
                            max_features="sqrt",
                            random_state=int(random_seed),
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            objective="classification",
        ),
    )
    observed = tuple(spec.model_name for spec in specs)
    if observed != ACTIVE_MODEL_NAMES:
        raise RuntimeError(f"active Decision model order does not match the frozen protocol: {observed}")
    return specs


def extended_model_specs(random_seed: int) -> tuple[DecisionModelSpec, ...]:
    """Return all model candidates for ablation / model comparison experiments."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import LinearRegression

    xgb_seed = int(np.random.SeedSequence([int(random_seed), 20260811, 99]).generate_state(1, dtype=np.uint32)[0])

    base_specs = list(active_model_specs(random_seed))

    extended = base_specs + [
        DecisionModelSpec(
            model_name="random_forest_regressor",
            model_family="random_forest",
            estimator_name="RandomForestRegressor(n_estimators=200,max_depth=8)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(
                    n_estimators=200, max_depth=8, max_features="sqrt",
                    random_state=xgb_seed, n_jobs=1,
                )),
            ]),
            objective="regression",
        ),
        DecisionModelSpec(
            model_name="mlp_classifier",
            model_family="mlp",
            estimator_name="MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500,
                    random_state=xgb_seed, early_stopping=True,
                )),
            ]),
            objective="classification",
        ),
        DecisionModelSpec(
            model_name="mlp_regressor",
            model_family="mlp",
            estimator_name="MLPRegressor(hidden_layer_sizes=(64,32),max_iter=500)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", MLPRegressor(
                    hidden_layer_sizes=(64, 32), max_iter=500,
                    random_state=xgb_seed, early_stopping=True,
                )),
            ]),
            objective="regression",
        ),
        DecisionModelSpec(
            model_name="svm_rbf_classifier",
            model_family="svm",
            estimator_name="SVC(kernel=rbf,C=1.0,gamma=scale)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=xgb_seed)),
            ]),
            objective="classification",
        ),
        DecisionModelSpec(
            model_name="svm_rbf_regressor",
            model_family="svm",
            estimator_name="SVR(kernel=rbf,C=1.0,gamma=scale)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", SVR(kernel="rbf", C=1.0, gamma="scale")),
            ]),
            objective="regression",
        ),
        DecisionModelSpec(
            model_name="knn_classifier",
            model_family="knn",
            estimator_name="KNeighborsClassifier(n_neighbors=7)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]),
            objective="classification",
        ),
        DecisionModelSpec(
            model_name="knn_regressor",
            model_family="knn",
            estimator_name="KNeighborsRegressor(n_neighbors=7)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", KNeighborsRegressor(n_neighbors=7, weights="distance")),
            ]),
            objective="regression",
        ),
        DecisionModelSpec(
            model_name="linear_regression",
            model_family="linear_regression",
            estimator_name="LinearRegression()",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]),
            objective="regression",
        ),
        DecisionModelSpec(
            model_name="kernel_ridge_regressor",
            model_family="kernel_ridge",
            estimator_name="KernelRidge(kernel=rbf,alpha=1.0,gamma=scale)",
            estimator=Pipeline([
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", KernelRidge(kernel="rbf", alpha=1.0, gamma="scale")),
            ]),
            objective="regression",
        ),
    ]

    # Try XGBoost (optional dependency)
    try:
        from xgboost import XGBClassifier, XGBRegressor
        extended += (
            DecisionModelSpec(
                model_name="xgboost_classifier",
                model_family="xgboost",
                estimator_name="XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1)",
                estimator=Pipeline([
                    ("imputer", WeightedMedianImputer()),
                    ("scaler", StandardScaler()),
                    ("classifier", XGBClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        random_state=xgb_seed, n_jobs=1, eval_metric="logloss",
                        use_label_encoder=False,
                    )),
                ]),
                objective="classification",
            ),
            DecisionModelSpec(
                model_name="xgboost_regressor",
                model_family="xgboost",
                estimator_name="XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1)",
                estimator=Pipeline([
                    ("imputer", WeightedMedianImputer()),
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        random_state=xgb_seed, n_jobs=1,
                    )),
                ]),
                objective="regression",
            ),
        )
    except ImportError:
        pass

    return tuple(extended)


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
            "Decision training summary does not match the active four-candidate protocol: "
            f"expected={ACTIVE_MODEL_NAMES}, observed={trained}"
        )
    if requested not in ACTIVE_MODEL_NAMES:
        raise ValueError(f"model is not an active Decision candidate: {requested}")
    return requested
