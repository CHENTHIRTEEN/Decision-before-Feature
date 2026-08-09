from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ela.features import ELA_FEATURE_COLUMNS
from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    fe_total_for_dimension,
    load_config,
    make_shards,
    selected_dimensions,
    selected_functions,
)


SELECTION_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "remaining_budget_ratio",
    "performance_bucket_ratio",
    "selected_algorithm",
    "default_algorithm",
    "sbs_algorithm",
    "vbs_algorithm",
    "selector_status",
    "runtime_selection",
)


def build_selection_reference(
    *,
    train_config_path: Path,
    output_path: Path,
    ela_root: Path,
    predict_config_paths: list[Path] | None,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> dict[str, int | str]:
    train_config = load_config(train_config_path)
    predict_configs = [load_config(path) for path in (predict_config_paths or [train_config_path])]
    train_features = _read_feature_file(_ela_feature_path(train_config, ela_root))
    train_performance = _read_performance(train_config, only_functions, only_dimensions)
    train_budget_map = _checkpoint_budget_map(train_config, train_performance, only_dimensions)
    sbs_algorithm = _single_best_solver(train_performance)
    buckets = sorted({remaining for values in train_budget_map.values() for remaining, _ in values})

    rows = []
    for remaining_ratio in buckets:
        train_bucket_ratio_by_dimension = _bucket_ratio_by_dimension(train_budget_map, remaining_ratio)
        target = _best_algorithm_by_problem(train_performance, train_bucket_ratio_by_dimension)
        model, status = _fit_selector(train_features, target)
        for config in predict_configs:
            features = _read_feature_file(_ela_feature_path(config, ela_root))
            performance = _read_performance(config, only_functions, only_dimensions)
            split = _split_name(config)
            predict_budget_map = _checkpoint_budget_map(config, performance, only_dimensions)
            predict_bucket_ratio_by_dimension = _bucket_ratio_by_dimension(predict_budget_map, remaining_ratio)
            predicted = _predict_algorithms(model, status, features, sbs_algorithm)
            vbs = _best_algorithm_by_problem(performance, predict_bucket_ratio_by_dimension)
            for _, feature_row in features.iterrows():
                problem_id = str(feature_row["problem_id"])
                dimension = int(feature_row["dimension"])
                started = perf_counter()
                selected_algorithm = predicted[problem_id]
                runtime_selection = perf_counter() - started
                rows.append(
                    {
                        "split": split,
                        "problem_id": problem_id,
                        "family": str(feature_row["family"]),
                        "dimension": dimension,
                        "remaining_budget_ratio": float(remaining_ratio),
                        "performance_bucket_ratio": float(predict_bucket_ratio_by_dimension[dimension]),
                        "selected_algorithm": selected_algorithm,
                        "default_algorithm": sbs_algorithm,
                        "sbs_algorithm": sbs_algorithm,
                        "vbs_algorithm": vbs.get(problem_id, ""),
                        "selector_status": status,
                        "runtime_selection": float(runtime_selection),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), output_path)
    print(f"wrote {len(rows)} selection reference rows to {output_path}")
    print(f"SBS default optimizer: {sbs_algorithm}")
    return {"rows": len(rows), "output": str(output_path)}


def _ela_feature_path(config: dict, ela_root: Path) -> Path:
    split = _split_name(config)
    return ela_root / split / "features.parquet"


def _read_feature_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing ELA feature file: {path}")
    frame = pq.read_table(path).to_pandas()
    failed = frame[frame["feature_status"] != "ok"]
    if not failed.empty:
        raise ValueError(f"ELA feature file contains failed rows: {path}")
    return frame


def _read_performance(
    config: dict,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> pd.DataFrame:
    frames = []
    for shard in make_shards(config, only_functions, only_dimensions):
        if not shard.output_path.exists():
            raise FileNotFoundError(f"missing trajectory shard: {shard.output_path}")
        table = pq.read_table(
            shard.output_path,
            columns=["problem_id", "family", "dimension", "algorithm", "seed", "FE", "FE_ratio", "best_fitness"],
        )
        frames.append(table.to_pandas())
    if not frames:
        raise ValueError("no trajectory rows available for selection reference")
    return pd.concat(frames, ignore_index=True)


def _checkpoint_budget_map(
    config: dict,
    performance: pd.DataFrame,
    only_dimensions: list[int] | None,
) -> dict[int, list[tuple[float, float]]]:
    result = {}
    for dimension in selected_dimensions(config, only_dimensions):
        fe_total = fe_total_for_dimension(config, dimension)
        fe_analysis = int(0.05 * fe_total)
        subset = performance[performance["dimension"].astype(int) == int(dimension)]
        values = []
        for _, row in subset[["FE", "FE_ratio"]].drop_duplicates().iterrows():
            prefix_ratio = float(row["FE_ratio"])
            fe_prefix = int(row["FE"])
            if prefix_ratio >= 0.12 and fe_prefix + fe_analysis < fe_total:
                remaining_ratio = round((fe_total - fe_prefix - fe_analysis) / fe_total, 6)
                values.append((remaining_ratio, round(prefix_ratio, 6)))
        if not values:
            raise ValueError(f"no eligible checkpoint ratios for dimension {dimension}")
        result[int(dimension)] = sorted(set(values))
    return result


def _bucket_ratio_by_dimension(
    budget_map: dict[int, list[tuple[float, float]]],
    remaining_ratio: float,
) -> dict[int, float]:
    return {
        dimension: min(values, key=lambda item: (abs(item[0] - remaining_ratio), item[0]))[1]
        for dimension, values in budget_map.items()
    }


def _single_best_solver(performance: pd.DataFrame) -> str:
    final = performance.sort_values("FE_ratio").groupby(["problem_id", "algorithm"], as_index=False).tail(1)
    means = final.groupby(["problem_id", "algorithm"], as_index=False)["best_fitness"].mean()
    means["rank"] = means.groupby("problem_id")["best_fitness"].rank(method="average", ascending=True)
    ranks = means.groupby("algorithm")["rank"].mean().sort_values()
    return str(ranks.index[0])


def _best_algorithm_by_problem(performance: pd.DataFrame, bucket_ratio_by_dimension: dict[int, float]) -> dict[str, str]:
    frames = []
    for dimension, ratio in bucket_ratio_by_dimension.items():
        subset = performance[(performance["dimension"] == dimension) & (performance["FE_ratio"].round(6) == round(ratio, 6))]
        frames.append(subset)
    selected = pd.concat(frames, ignore_index=True)
    means = selected.groupby(["problem_id", "algorithm"], as_index=False)["best_fitness"].mean()
    ordered = means.sort_values(["problem_id", "best_fitness", "algorithm"])
    best = ordered.groupby("problem_id", as_index=False).first()
    return dict(zip(best["problem_id"], best["algorithm"], strict=False))


def _fit_selector(train_features: pd.DataFrame, target: dict[str, str]) -> tuple[Pipeline | str, str]:
    rows = train_features[train_features["problem_id"].isin(target)].copy()
    y = rows["problem_id"].map(target).astype(str)
    if rows.empty:
        raise ValueError("selection reference training target has no matching ELA feature rows")
    if len(set(y)) <= 1:
        return str(y.iloc[0]), "constant"
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=1701,
                    class_weight="balanced",
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    model.fit(rows[list(ELA_FEATURE_COLUMNS)], y)
    return model, "random_forest"


def _predict_algorithms(model: Pipeline | str, status: str, features: pd.DataFrame, fallback: str) -> dict[str, str]:
    if status == "constant":
        return {str(problem_id): str(model) for problem_id in features["problem_id"]}
    if not isinstance(model, Pipeline):
        return {str(problem_id): fallback for problem_id in features["problem_id"]}
    values = model.predict(features[list(ELA_FEATURE_COLUMNS)])
    return dict(zip(features["problem_id"].astype(str), values.astype(str), strict=False))


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("split", pa.string()),
            ("problem_id", pa.string()),
            ("family", pa.string()),
            ("dimension", pa.int32()),
            ("remaining_budget_ratio", pa.float64()),
            ("performance_bucket_ratio", pa.float64()),
            ("selected_algorithm", pa.string()),
            ("default_algorithm", pa.string()),
            ("sbs_algorithm", pa.string()),
            ("vbs_algorithm", pa.string()),
            ("selector_status", pa.string()),
            ("runtime_selection", pa.float64()),
        ]
    )


def _split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    return Path(config["output"]).stem.removesuffix("_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only selection reference for utility label generation.")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--predict-config", type=Path, action="append", default=None)
    parser.add_argument("--ela-root", type=Path, default=Path("results/ela"))
    parser.add_argument("--output", type=Path, default=Path("results/selection_reference/bbob_train/selection_reference.parquet"))
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    args = parser.parse_args()
    predict_configs = args.predict_config or [
        Path("configs/phase1_bbob_train.yaml"),
        Path("configs/phase1_bbob_validation.yaml"),
    ]
    build_selection_reference(
        train_config_path=args.train_config,
        output_path=args.output,
        ela_root=args.ela_root,
        predict_config_paths=predict_configs,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
    )


if __name__ == "__main__":
    main()
