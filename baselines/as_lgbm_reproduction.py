"""复现 Guo 等（2025）AS-LGBM 的静态算法选择基线。

该模块只处理论文式的静态问题实例表：每个实例一行，输入为 61 个低成本
ELA 特征，标签由 10 个算法各 30 次运行的结果通过 Soft-ERT 离线生成。
它不读取 Decision-before-Feature 的 Decision dataset，也不参与主 Decision
Model 的候选模型或下游 Selector。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ranksums
from sklearn.model_selection import KFold, train_test_split


PAPER_ALGORITHMS = (
    "ABC",
    "ACO",
    "CMA-ES",
    "CSO",
    "DE",
    "FEP",
    "GA",
    "PSO",
    "SA",
    "RAND",
)
PAPER_FEATURE_COUNT = 61
PAPER_REPEATS = 30
SOFT_ERT_TIMEOUT_PENALTY = 10001.0
RANK_SUM_ACCEPTANCE_P = 0.05
FEATURE_VALUE_LIMIT = 1.0e8
SPLIT_STREAM_CODE = 2401
MODEL_STREAM_CODE = 2402


@dataclass
class StaticDataset:
    features: np.ndarray
    performance_runs: np.ndarray
    source_row_index: np.ndarray


@dataclass
class FoldResult:
    fold: int
    train_rows: int
    evaluation_rows: int
    best_iteration: int
    exact_accuracy: float
    acceptable_accuracy: float
    model_path: str


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")
    if path.is_dir():
        parts = sorted(path.glob("part-*.parquet"))
        if not parts:
            raise ValueError(f"input directory has no part-*.parquet files: {path}")
        return pd.concat(
            [pd.read_parquet(part) for part in parts],
            ignore_index=True,
        )
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, header=None)
    raise ValueError(f"unsupported input format: {path.suffix}; use CSV or Parquet")


def _parse_column_list(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    columns = tuple(item.strip() for item in value.split(",") if item.strip())
    if not columns:
        raise ValueError("column list is empty")
    return columns


def _numeric_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"input contains no numeric columns: {path}")
    return numeric


def _select_matrix_columns(
    frame: pd.DataFrame,
    *,
    path: Path,
    requested: tuple[str, ...] | None,
    expected_width: int,
    semantic_prefixes: tuple[str, ...],
) -> np.ndarray:
    if requested is not None:
        missing = sorted(set(requested).difference(frame.columns))
        if missing:
            raise ValueError(f"requested columns are missing from {path}: {missing}")
        selected = frame.loc[:, list(requested)]
        if selected.shape[1] != expected_width:
            raise ValueError(
                f"requested columns in {path} have width {selected.shape[1]}, "
                f"expected {expected_width}"
            )
    else:
        semantic = [
            column
            for column in frame.columns
            if any(str(column).startswith(prefix) for prefix in semantic_prefixes)
        ]
        numeric = _numeric_frame(frame, path)
        if len(semantic) == expected_width:
            selected = frame.loc[:, semantic]
        elif len(numeric.columns) == expected_width:
            selected = numeric
        elif len(numeric.columns) == expected_width + 1:
            selected = numeric.iloc[:, 1:]
        else:
            raise ValueError(
                f"cannot infer {expected_width} numeric columns from {path}; "
                f"found {len(numeric.columns)}. Use an explicit column list."
            )
    try:
        values = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"selected columns in {path} are not numeric") from exc
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(f"matrix from {path} has unexpected shape {values.shape}")
    return values


def load_static_dataset(
    *,
    feature_path: Path,
    performance_path: Path,
    feature_columns: tuple[str, ...] | None = None,
    performance_columns: tuple[str, ...] | None = None,
    feature_count: int = PAPER_FEATURE_COUNT,
    repeats: int = PAPER_REPEATS,
    algorithm_count: int = len(PAPER_ALGORITHMS),
) -> StaticDataset:
    feature_frame = _read_table(feature_path)
    performance_frame = _read_table(performance_path)
    features = _select_matrix_columns(
        feature_frame,
        path=feature_path,
        requested=feature_columns,
        expected_width=feature_count,
        semantic_prefixes=("feature_", "ela_", "f_"),
    )
    performance_runs = _select_matrix_columns(
        performance_frame,
        path=performance_path,
        requested=performance_columns,
        expected_width=algorithm_count * repeats,
        semantic_prefixes=("performance_", "perf_", "run_", "rt_"),
    )
    if len(features) != len(performance_runs):
        raise ValueError(
            "feature and performance tables have different row counts: "
            f"{len(features)} versus {len(performance_runs)}"
        )
    if not np.isfinite(features).all():
        raise ValueError("feature table contains NaN or infinite values")
    if not np.isfinite(performance_runs).all():
        raise ValueError("performance table contains NaN or infinite values")
    return StaticDataset(
        features=features,
        performance_runs=performance_runs,
        source_row_index=np.arange(len(features), dtype=int),
    )


def screen_dataset(
    dataset: StaticDataset,
    *,
    feature_value_limit: float = FEATURE_VALUE_LIMIT,
) -> StaticDataset:
    """按仓库 ``Benchmarks.data_screening`` 的规则筛选有效行。"""
    features = dataset.features
    performances = dataset.performance_runs
    keep = (
        np.sum(features, axis=1) != 0.0
    )
    keep &= np.isfinite(features).all(axis=1)
    keep &= (np.abs(features) <= feature_value_limit).all(axis=1)
    keep &= np.sum(performances, axis=1) != 0.0
    return StaticDataset(
        features=features[keep],
        performance_runs=performances[keep],
        source_row_index=dataset.source_row_index[keep],
    )


def soft_ert(
    performance_runs: np.ndarray,
    *,
    algorithm_count: int,
    repeats: int = PAPER_REPEATS,
    timeout_penalty: float = SOFT_ERT_TIMEOUT_PENALTY,
) -> np.ndarray:
    """计算论文和仓库实现使用的 Soft-ERT 矩阵。"""
    values = np.asarray(performance_runs, dtype=float)
    expected_width = algorithm_count * repeats
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            f"performance matrix must have shape (n, {expected_width}), got {values.shape}"
        )
    runs = values.reshape(len(values), algorithm_count, repeats)
    uncensored = np.sum(runs != -1.0, axis=2)
    return (
        np.sum(runs, axis=2) + np.sum(runs == -1.0, axis=2) * timeout_penalty
    ) / np.maximum(uncensored, 1)


def best_algorithm_labels(
    performance_runs: np.ndarray,
    *,
    algorithm_count: int,
    repeats: int = PAPER_REPEATS,
) -> tuple[np.ndarray, np.ndarray]:
    ert = soft_ert(
        performance_runs,
        algorithm_count=algorithm_count,
        repeats=repeats,
    )
    return np.argmin(ert, axis=1).astype(int), ert


def sanitized_performance_runs(performance_runs: np.ndarray) -> np.ndarray:
    """按仓库标签函数将失败运行值 -1 替换为 10000。"""
    return np.where(np.asarray(performance_runs, dtype=float) == -1.0, 10000.0, performance_runs)


def _derived_seed(root_seed: int, stream_code: int, unit_number: int) -> int:
    sequence = np.random.SeedSequence([int(root_seed), int(stream_code), int(unit_number)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def acceptable_accuracy(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    performance_runs: np.ndarray,
    *,
    repeats: int = PAPER_REPEATS,
    p_threshold: float = RANK_SUM_ACCEPTANCE_P,
) -> float:
    """计算仓库中使用的 acceptable accuracy。"""
    true_values = np.asarray(true_labels, dtype=int)
    predicted_values = np.asarray(predicted_labels, dtype=int)
    runs = sanitized_performance_runs(performance_runs).reshape(
        len(performance_runs), -1, repeats
    )
    if len(true_values) != len(predicted_values) or len(true_values) != len(runs):
        raise ValueError("accuracy inputs have different row counts")
    accepted = 0
    for row, (truth, predicted) in enumerate(zip(true_values, predicted_values, strict=True)):
        if truth == predicted:
            accepted += 1
            continue
        p_value = float(ranksums(runs[row, predicted], runs[row, truth]).pvalue)
        if p_value > p_threshold:
            accepted += 1
    return float(accepted / len(true_values)) if len(true_values) else float("nan")


def _fit_booster(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    evaluation_y: np.ndarray,
    *,
    algorithm_count: int,
    learning_rate: float,
    max_depth: int,
    num_rounds: int,
    early_stopping_rounds: int,
    lambda_l1: float,
    lambda_l2: float,
    seed: int,
):
    import lightgbm as lgb

    train_data = lgb.Dataset(train_x, label=train_y)
    evaluation_data = lgb.Dataset(evaluation_x, label=evaluation_y, reference=train_data)
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_class": algorithm_count,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }
    return lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, evaluation_data],
        valid_names=["train", "evaluation"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _evaluate_fold(
    *,
    fold: int,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    dataset: StaticDataset,
    labels: np.ndarray,
    algorithm_names: Sequence[str],
    repeats: int,
    output_dir: Path,
    learning_rate: float,
    max_depth: int,
    num_rounds: int,
    early_stopping_rounds: int,
    lambda_l1: float,
    lambda_l2: float,
    seed: int,
) -> tuple[FoldResult, pd.DataFrame]:
    model = _fit_booster(
        dataset.features[train_indices],
        labels[train_indices],
        dataset.features[evaluation_indices],
        labels[evaluation_indices],
        algorithm_count=len(algorithm_names),
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_rounds=num_rounds,
        early_stopping_rounds=early_stopping_rounds,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        seed=_derived_seed(seed, MODEL_STREAM_CODE, fold),
    )
    model_path = output_dir / f"selector_fold_{fold}.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    probabilities = np.asarray(model.predict(dataset.features[evaluation_indices]), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(algorithm_names):
        raise ValueError("LightGBM prediction matrix has an unexpected shape")
    predictions = np.argmax(probabilities, axis=1).astype(int)
    true_values = labels[evaluation_indices]
    evaluation_runs = dataset.performance_runs[evaluation_indices]
    prediction_frame = pd.DataFrame(
        {
            "fold": fold,
            "source_row_index": dataset.source_row_index[evaluation_indices],
            "true_label": true_values,
            "predicted_label": predictions,
            "true_algorithm": [algorithm_names[index] for index in true_values],
            "predicted_algorithm": [algorithm_names[index] for index in predictions],
            "exact_match": true_values == predictions,
            "acceptable_match": _acceptable_flags(
                true_values,
                predictions,
                evaluation_runs,
                repeats=repeats,
            ),
        }
    )
    exact = float(np.mean(true_values == predictions))
    acceptable = float(prediction_frame["acceptable_match"].mean())
    result = FoldResult(
        fold=fold,
        train_rows=int(len(train_indices)),
        evaluation_rows=int(len(evaluation_indices)),
        best_iteration=int(model.best_iteration or num_rounds),
        exact_accuracy=exact,
        acceptable_accuracy=acceptable,
        model_path=str(model_path),
    )
    return result, prediction_frame


def _acceptable_flags(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    performance_runs: np.ndarray,
    *,
    repeats: int,
    p_threshold: float = RANK_SUM_ACCEPTANCE_P,
) -> np.ndarray:
    runs = sanitized_performance_runs(performance_runs).reshape(
        len(performance_runs), -1, repeats
    )
    flags = np.zeros(len(true_labels), dtype=bool)
    for row, (truth, predicted) in enumerate(zip(true_labels, predicted_labels, strict=True)):
        flags[row] = truth == predicted
        if not flags[row]:
            flags[row] = float(ranksums(runs[row, predicted], runs[row, truth]).pvalue) > p_threshold
    return flags


def run_reproduction(
    *,
    dataset: StaticDataset,
    algorithm_names: Sequence[str] = PAPER_ALGORITHMS,
    repeats: int = PAPER_REPEATS,
    evaluation_mode: str = "paper_holdout",
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    learning_rate: float = 0.01,
    max_depth: int = 5,
    num_rounds: int = 2000,
    early_stopping_rounds: int = 200,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    output_dir: Path = Path("outputs/as_lgbm_paper_baseline"),
) -> tuple[dict[str, object], pd.DataFrame]:
    names = tuple(str(name) for name in algorithm_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("algorithm_names must contain at least two unique names")
    if dataset.performance_runs.shape[1] != len(names) * repeats:
        raise ValueError(
            "performance width does not match algorithm_names and repeats: "
            f"{dataset.performance_runs.shape[1]} versus {len(names) * repeats}"
        )
    labels, ert = best_algorithm_labels(
        dataset.performance_runs,
        algorithm_count=len(names),
        repeats=repeats,
    )
    if len(labels) < 2:
        raise ValueError("at least two valid instances are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[FoldResult] = []
    prediction_frames: list[pd.DataFrame] = []

    if evaluation_mode == "paper_holdout":
        indices = np.arange(len(labels), dtype=int)
        split_seed = _derived_seed(random_state, SPLIT_STREAM_CODE, 0)
        train_indices, evaluation_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=split_seed,
        )
        result, frame = _evaluate_fold(
            fold=0,
            train_indices=np.asarray(train_indices, dtype=int),
            evaluation_indices=np.asarray(evaluation_indices, dtype=int),
            dataset=dataset,
            labels=labels,
            algorithm_names=names,
            repeats=repeats,
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_rounds=num_rounds,
            early_stopping_rounds=early_stopping_rounds,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            seed=random_state,
        )
        fold_results.append(result)
        prediction_frames.append(frame)
    elif evaluation_mode == "five_fold":
        if cv_folds < 2 or cv_folds > len(labels):
            raise ValueError("cv_folds must be between 2 and the number of rows")
        split_seed = _derived_seed(random_state, SPLIT_STREAM_CODE, cv_folds)
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=split_seed)
        for fold, (train_indices, evaluation_indices) in enumerate(splitter.split(dataset.features), 1):
            result, frame = _evaluate_fold(
                fold=fold,
                train_indices=train_indices,
                evaluation_indices=evaluation_indices,
                dataset=dataset,
                labels=labels,
                algorithm_names=names,
                repeats=repeats,
                output_dir=output_dir,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_rounds=num_rounds,
                early_stopping_rounds=early_stopping_rounds,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                seed=random_state,
            )
            fold_results.append(result)
            prediction_frames.append(frame)
    else:
        raise ValueError(f"unsupported evaluation_mode: {evaluation_mode}")

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "source_row_index"]
    ).reset_index(drop=True)
    summary = {
        "method": "AS-LGBM",
        "evaluation_mode": evaluation_mode,
        "feature_count": int(dataset.features.shape[1]),
        "instance_count_before_screening": int(len(dataset.source_row_index)),
        "instance_count_after_screening": int(len(labels)),
        "algorithm_names": list(names),
        "algorithm_count": len(names),
        "repeats_per_algorithm": repeats,
        "soft_ert_timeout_penalty": SOFT_ERT_TIMEOUT_PENALTY,
        "rank_sum_acceptance_p": RANK_SUM_ACCEPTANCE_P,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_rounds": num_rounds,
        "early_stopping_rounds": early_stopping_rounds,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "random_state": random_state,
        "folds": [result.__dict__ for result in fold_results],
        "mean_exact_accuracy": float(np.mean([result.exact_accuracy for result in fold_results])),
        "mean_acceptable_accuracy": float(
            np.mean([result.acceptable_accuracy for result in fold_results])
        ),
        "label_distribution": {
            names[index]: int(np.sum(labels == index)) for index in range(len(names))
        },
        "soft_ert_shape": list(ert.shape),
        "evaluation_data_used_for_early_stopping": True,
        "evaluation_protocol_note": (
            "The public notebook uses the evaluation partition for LightGBM early stopping; "
            "this field is retained to prevent interpreting the score as an independent estimate."
        ),
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, predictions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-columns")
    parser.add_argument("--performance-columns")
    parser.add_argument("--algorithm-names", default=",".join(PAPER_ALGORITHMS))
    parser.add_argument("--repeats", type=int, default=PAPER_REPEATS)
    parser.add_argument(
        "--evaluation-mode",
        choices=("paper_holdout", "five_fold"),
        default="paper_holdout",
        help="paper_holdout mirrors the public repository notebook; five_fold follows the paper text.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    algorithm_names = tuple(
        name.strip() for name in args.algorithm_names.split(",") if name.strip()
    )
    raw = load_static_dataset(
        feature_path=args.features,
        performance_path=args.performance,
        feature_columns=_parse_column_list(args.feature_columns),
        performance_columns=_parse_column_list(args.performance_columns),
        feature_count=PAPER_FEATURE_COUNT,
        repeats=args.repeats,
        algorithm_count=len(algorithm_names),
    )
    screened = screen_dataset(raw)
    summary, _ = run_reproduction(
        dataset=screened,
        algorithm_names=algorithm_names,
        repeats=args.repeats,
        evaluation_mode=args.evaluation_mode,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_rounds=args.num_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
