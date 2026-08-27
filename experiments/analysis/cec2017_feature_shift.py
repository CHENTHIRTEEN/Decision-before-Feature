"""Compare CEC2017 feature distributions with the BBOB/MA-BBOB training reference.

The script reuses the existing trajectory behavior extractor and the primary
14-dimensional landscape descriptor.  It writes one merged state-level feature
matrix, run-phase profiles for distribution checks, univariate shift tables,
and grouped out-of-fold domain-classifier diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from behavior.features import (
    BEHAVIOR_FEATURE_COLUMNS,
    BEHAVIOR_FEATURE_GROUPS,
    SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
)
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/dataset_analysis/cec2017_feature_shift"
DESCRIPTOR_COLUMNS = tuple(f"descriptor_{name.removeprefix('descriptor_')}" for name in DESCRIPTOR_CHEAP_COLUMNS)
B3_COLUMNS = tuple(BEHAVIOR_FEATURE_GROUPS["B3"])
DECISION_COLUMNS = tuple(BEHAVIOR_FEATURE_GROUPS["B2+Motion+SearchMaturityLinear"])
SELECTOR_COLUMNS = tuple(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
SELECTOR_FEATURE_COLUMNS = SELECTOR_COLUMNS + DESCRIPTOR_COLUMNS + ("remaining_budget_ratio",)
DECISION_FEATURE_COLUMNS = DECISION_COLUMNS
MATRIX_FEATURE_COLUMNS = B3_COLUMNS + DESCRIPTOR_COLUMNS + (
    "FE_prefix",
    "FE_total",
    "remaining_budget_ratio",
)
COMMON_DIMENSIONS = (10, 20)
ANALYSIS_SEED = 20260826
CEC_SOURCE = "cec2017_distribution_shift"
TRAIN_SOURCES = ("bbob_train", "mabbob_formal")


@dataclass
class SourceSpec:
    name: str
    dataset: str
    suite: str
    trajectory_root: Path
    descriptor_split: str


SOURCE_SPECS = (
    SourceSpec(
        name="bbob_train",
        dataset="train_reference",
        suite="BBOB",
        trajectory_root=REPO / "results/phase1_refined_sampling/bbob_train",
        descriptor_split="bbob_train",
    ),
    SourceSpec(
        name="mabbob_formal",
        dataset="train_reference",
        suite="MA-BBOB",
        trajectory_root=REPO / "results/phase1_mabbob/mabbob_formal",
        descriptor_split="mabbob_formal",
    ),
    SourceSpec(
        name="cec2017_distribution_shift",
        dataset="cec2017",
        suite="CEC2017",
        trajectory_root=REPO / "results/phase1_cec2017_distribution_shift/cec2017_distribution_shift",
        descriptor_split="cec2017_distribution_shift",
    ),
)


def _finite_seed() -> int:
    sequence = np.random.SeedSequence([ANALYSIS_SEED, 731, 1])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _read_source(spec: SourceSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    shards = sorted(spec.trajectory_root.glob("*/dimension_*/trajectories.parquet"))
    if not shards:
        raise FileNotFoundError(f"no trajectory shards found under {spec.trajectory_root}")
    behavior_parts: list[pd.DataFrame] = []
    for trajectory_path in shards:
        behavior_path = trajectory_path.with_name("behavior.parquet")
        if not behavior_path.exists():
            raise FileNotFoundError(f"missing behavior shard: {behavior_path}")
        frame = pd.read_parquet(
            behavior_path,
            columns=[
                "problem_id",
                "function_id",
                "family",
                "cv_group_id",
                "dimension",
                "algorithm",
                "seed",
                "FE",
                "FE_ratio",
                *BEHAVIOR_FEATURE_COLUMNS,
            ],
        )
        trajectory_meta = pd.read_parquet(
            trajectory_path,
            columns=["problem_id", "algorithm", "seed", "FE", "FE_total"],
        )
        frame = frame.merge(
            trajectory_meta,
            on=["problem_id", "algorithm", "seed", "FE"],
            how="left",
            validate="one_to_one",
        )
        if frame["FE_total"].isna().any():
            raise ValueError(f"behavior rows have no trajectory FE_total: {behavior_path}")
        frame["source"] = spec.name
        frame["dataset"] = spec.dataset
        frame["suite"] = spec.suite
        behavior_parts.append(frame)
    behavior = pd.concat(behavior_parts, ignore_index=True)

    descriptor_path = (
        REPO
        / "results/landscape_queries/features/descriptor_cheap_invariant"
        / spec.descriptor_split
        / "features.parquet"
    )
    if not descriptor_path.exists():
        raise FileNotFoundError(f"missing descriptor feature table: {descriptor_path}")
    descriptor = pd.read_parquet(
        descriptor_path,
        columns=["problem_id", "dimension", *DESCRIPTOR_COLUMNS],
    )
    descriptor["source"] = spec.name
    descriptor["dataset"] = spec.dataset
    descriptor["suite"] = spec.suite
    return behavior, descriptor


def _build_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior_parts: list[pd.DataFrame] = []
    descriptor_parts: list[pd.DataFrame] = []
    for spec in SOURCE_SPECS:
        behavior, descriptor = _read_source(spec)
        behavior_parts.append(behavior)
        descriptor_parts.append(descriptor)
    behavior = pd.concat(behavior_parts, ignore_index=True)
    descriptors = pd.concat(descriptor_parts, ignore_index=True)
    descriptor_function = behavior[
        ["source", "problem_id", "dimension", "function_id"]
    ].drop_duplicates()
    if descriptor_function.duplicated(["source", "problem_id", "dimension"]).any():
        raise ValueError("behavior table has inconsistent function_id per descriptor key")
    descriptors = descriptors.merge(
        descriptor_function,
        on=["source", "problem_id", "dimension"],
        how="left",
        validate="one_to_one",
    )
    if descriptors["function_id"].isna().any():
        raise ValueError("descriptor table has no matched function_id")
    descriptor_keys = ["source", "problem_id", "dimension"]
    if descriptors.duplicated(descriptor_keys).any():
        raise ValueError("descriptor table has duplicate source/problem/dimension keys")

    descriptor_values = descriptors[descriptor_keys + list(DESCRIPTOR_COLUMNS)]
    matrix = behavior.merge(
        descriptor_values,
        on=descriptor_keys,
        how="left",
        validate="many_to_one",
    )
    if matrix[list(DESCRIPTOR_COLUMNS)].isna().all(axis=1).any():
        missing = matrix.loc[
            matrix[list(DESCRIPTOR_COLUMNS)].isna().all(axis=1),
            ["source", "problem_id", "dimension"],
        ].drop_duplicates()
        raise ValueError(f"behavior rows have no matched descriptor rows: {missing.to_dict('records')[:5]}")
    matrix["FE_prefix"] = matrix["FE"].astype(float)
    matrix["remaining_budget_ratio"] = (
        (matrix["FE_total"].astype(float) - matrix["FE"].astype(float))
        / matrix["FE_total"].astype(float)
    )
    if len(B3_COLUMNS) != 31:
        raise ValueError(f"expected 31 B3 behavior features, got {len(B3_COLUMNS)}")
    if len(SELECTOR_COLUMNS) != 28:
        raise ValueError(f"expected 28 selector behavior features, got {len(SELECTOR_COLUMNS)}")
    if len(DESCRIPTOR_COLUMNS) != 14:
        raise ValueError(f"expected 14 descriptor features, got {len(DESCRIPTOR_COLUMNS)}")
    if len(SELECTOR_FEATURE_COLUMNS) != 43:
        raise ValueError(f"expected 43 selector features, got {len(SELECTOR_FEATURE_COLUMNS)}")
    if len(DECISION_FEATURE_COLUMNS) != 29:
        raise ValueError(f"expected 29 decision features, got {len(DECISION_FEATURE_COLUMNS)}")
    if not any(column.startswith("bf_") for column in SELECTOR_FEATURE_COLUMNS):
        raise ValueError("missing behavior features in selector feature contract")
    if not any(column.startswith("descriptor_") for column in SELECTOR_FEATURE_COLUMNS):
        raise ValueError("missing descriptor features in selector feature contract")
    return matrix, descriptors


def _profile_states(matrix: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.25, 0.50, 0.75, 1.0 + 1e-12]
    labels = ["early", "mid_early", "mid_late", "late"]
    profile = matrix.copy()
    profile["phase_bin"] = pd.cut(
        profile["FE_ratio"].astype(float),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    keys = [
        "source",
        "dataset",
        "suite",
        "problem_id",
        "function_id",
        "algorithm",
        "seed",
        "dimension",
        "phase_bin",
    ]
    values = list(SELECTOR_FEATURE_COLUMNS) + list(DECISION_FEATURE_COLUMNS)
    values = list(dict.fromkeys(values))
    grouped = profile.groupby(keys, observed=True, sort=True)[values].mean().reset_index()
    return grouped


def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _iqr(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _univariate_shift(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    feature_set: str,
    scope: str,
    dimensions: tuple[int, ...] | None,
) -> list[dict[str, object]]:
    subset = frame if dimensions is None else frame[frame["dimension"].isin(dimensions)]
    train = subset[subset["dataset"] == "train_reference"]
    cec = subset[subset["dataset"] == "cec2017"]
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        train_values = _finite_values(train, column)
        cec_values = _finite_values(cec, column)
        if len(train_values) < 2 or len(cec_values) < 2:
            continue
        train_median = float(np.median(train_values))
        cec_median = float(np.median(cec_values))
        train_iqr = _iqr(train_values)
        median_delta = cec_median - train_median
        scale = train_iqr if np.isfinite(train_iqr) and train_iqr > 1e-12 else 1.0
        pooled = np.concatenate([train_values, cec_values])
        labels = np.concatenate([np.zeros(len(train_values)), np.ones(len(cec_values))])
        auc = float(roc_auc_score(labels, pooled))
        ks = ks_2samp(train_values, cec_values, alternative="two-sided", mode="auto")
        rows.append(
            {
                "scope": scope,
                "feature_set": feature_set,
                "feature": column,
                "train_n": int(len(train_values)),
                "cec_n": int(len(cec_values)),
                "train_median": train_median,
                "cec_median": cec_median,
                "median_delta_cec_minus_train": median_delta,
                "train_iqr": train_iqr,
                "standardized_median_shift": float(median_delta / scale),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue_diagnostic": float(ks.pvalue),
                "univariate_auc": auc,
                "directional_auc": float(max(auc, 1.0 - auc)),
            }
        )
    return rows


def _domain_classifier_auc(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    feature_set: str,
    scope: str,
    dimensions: tuple[int, ...] | None,
    reference_sources: tuple[str, ...] = TRAIN_SOURCES,
) -> dict[str, object]:
    subset = frame[frame["source"].isin((*reference_sources, CEC_SOURCE))]
    if dimensions is not None:
        subset = subset[subset["dimension"].isin(dimensions)]
    subset = subset.reset_index(drop=True)
    y = (subset["source"].astype(str) == CEC_SOURCE).astype(int).to_numpy()
    groups = subset["problem_id"].astype(str).to_numpy()
    reference_name = "+".join(reference_sources)
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 5:
        return {
            "scope": scope,
            "feature_set": feature_set,
            "reference_source": reference_name,
            "dimensions": "all" if dimensions is None else "+".join(map(str, dimensions)),
            "rows": int(len(subset)),
            "problems": int(len(np.unique(groups))),
            "cec_rows": int(y.sum()),
            "train_rows": int((y == 0).sum()),
            "oof_auc": None,
            "oof_directional_auc": None,
            "fold_auc_mean": None,
            "fold_directional_auc_mean": None,
            "fold_auc_sd": None,
            "status": "insufficient_class_or_group_coverage",
        }
    x = subset.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=_finite_seed(),
    )
    oof = np.full(len(subset), np.nan, dtype=float)
    fold_auc: list[float] = []
    for train_index, test_index in splitter.split(x, y, groups):
        if len(np.unique(y[train_index])) < 2 or len(np.unique(y[test_index])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
            ),
        )
        model.fit(x.iloc[train_index], y[train_index])
        probabilities = model.predict_proba(x.iloc[test_index])[:, 1]
        oof[test_index] = probabilities
        fold_auc.append(float(roc_auc_score(y[test_index], probabilities)))
    valid = np.isfinite(oof)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        status = "no_valid_group_folds"
        oof_auc = None
    else:
        status = "ok"
        oof_auc = float(roc_auc_score(y[valid], oof[valid]))
    return {
        "scope": scope,
        "feature_set": feature_set,
        "reference_source": reference_name,
        "dimensions": "all" if dimensions is None else "+".join(map(str, dimensions)),
        "rows": int(len(subset)),
        "problems": int(len(np.unique(groups))),
        "cec_rows": int(y.sum()),
        "train_rows": int((y == 0).sum()),
        "oof_auc": oof_auc,
        "oof_directional_auc": float(max(oof_auc, 1.0 - oof_auc)) if oof_auc is not None else None,
        "fold_auc_mean": float(np.mean(fold_auc)) if fold_auc else None,
        "fold_directional_auc_mean": (
            float(np.mean([max(value, 1.0 - value) for value in fold_auc]))
            if fold_auc
            else None
        ),
        "fold_auc_sd": float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else None,
        "status": status,
    }


def _stratified_domain_scopes() -> tuple[tuple[str, tuple[int, ...] | None, tuple[str, ...]], ...]:
    """Return pooled and suite-specific comparisons with shared dimensions."""
    return (
        ("all", None, TRAIN_SOURCES),
        ("common_10_20", COMMON_DIMENSIONS, TRAIN_SOURCES),
        ("dimension_10", (10,), TRAIN_SOURCES),
        ("dimension_20", (20,), TRAIN_SOURCES),
        ("dimension_10_bbob", (10,), ("bbob_train",)),
        ("dimension_20_bbob", (20,), ("bbob_train",)),
        ("dimension_10_mabbob", (10,), ("mabbob_formal",)),
        ("dimension_20_mabbob", (20,), ("mabbob_formal",)),
    )


def _top_common_features(shifts: pd.DataFrame, per_set: int = 5) -> dict[str, tuple[str, ...]]:
    selected: dict[str, tuple[str, ...]] = {}
    for feature_set in ("descriptor_14", "decision_29", "selector_43"):
        subset = shifts[
            (shifts["scope"] == "common_10_20")
            & (shifts["feature_set"] == feature_set)
        ].sort_values(["directional_auc", "ks_statistic"], ascending=False)
        selected[feature_set] = tuple(subset["feature"].head(per_set).tolist())
    return selected


def _trio_distribution_table(
    frame_by_set: dict[str, pd.DataFrame],
    selected_features: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Summarize BBOB, MA-BBOB and CEC distributions at shared dimensions."""
    rows: list[dict[str, object]] = []
    scopes = (
        ("common_10_20", "all_common", COMMON_DIMENSIONS),
        ("dimension_10", "10", (10,)),
        ("dimension_20", "20", (20,)),
    )
    for feature_set, features in selected_features.items():
        frame = frame_by_set[feature_set]
        for scope, dimension_label, dimensions in scopes:
            scoped = frame[frame["dimension"].isin(dimensions)]
            for feature in features:
                for source, group in scoped.groupby("source", sort=True):
                    values = _finite_values(group, feature)
                    if len(values) == 0:
                        continue
                    q25, q75 = np.percentile(values, [25, 75])
                    rows.append(
                        {
                            "scope": scope,
                            "dimension": dimension_label,
                            "feature_set": feature_set,
                            "feature": feature,
                            "source": source,
                            "suite": str(group["suite"].iloc[0]),
                            "n": int(len(values)),
                            "median": float(np.median(values)),
                            "q25": float(q25),
                            "q75": float(q75),
                            "iqr": float(q75 - q25),
                        }
                    )
    return pd.DataFrame(rows)


def _cec_function_distribution_table(
    frame_by_set: dict[str, pd.DataFrame],
    selected_features: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Summarize selected CEC feature distributions by function and dimension."""
    rows: list[dict[str, object]] = []
    for feature_set, features in selected_features.items():
        frame = frame_by_set[feature_set]
        cec = frame[frame["source"].eq(CEC_SOURCE)]
        for (function_id, dimension), group in cec.groupby(
            ["function_id", "dimension"], sort=True
        ):
            for feature in features:
                values = _finite_values(group, feature)
                if len(values) == 0:
                    continue
                q25, q75 = np.percentile(values, [25, 75])
                rows.append(
                    {
                        "feature_set": feature_set,
                        "function_id": str(function_id),
                        "dimension": int(dimension),
                        "feature": feature,
                        "n": int(len(values)),
                        "median": float(np.median(values)),
                        "q25": float(q25),
                        "q75": float(q75),
                        "iqr": float(q75 - q25),
                    }
                )
    return pd.DataFrame(rows)


def _profile_table(matrix: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in matrix.groupby("source", sort=True):
        rows.append(
            {
                "source": source,
                "dataset": str(group["dataset"].iloc[0]),
                "suite": str(group["suite"].iloc[0]),
                "state_rows": int(len(group)),
                "runs": int(group[["problem_id", "algorithm", "seed"]].drop_duplicates().shape[0]),
                "problems": int(group[["problem_id", "dimension"]].drop_duplicates().shape[0]),
                "dimensions": "+".join(map(str, sorted(group["dimension"].unique()))),
                "nonfinite_matrix_cells": int(
                    (~np.isfinite(group.loc[:, list(MATRIX_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce"))).sum().sum()
                ),
            }
        )
    for source, group in descriptors.groupby("source", sort=True):
        rows.append(
            {
                "source": f"{source}:descriptors",
                "dataset": str(group["dataset"].iloc[0]),
                "suite": str(group["suite"].iloc[0]),
                "state_rows": None,
                "runs": None,
                "problems": int(len(group)),
                "dimensions": "+".join(map(str, sorted(group["dimension"].unique()))),
                "nonfinite_matrix_cells": int(
                    (~np.isfinite(group.loc[:, list(DESCRIPTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce"))).sum().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(
    output_dir: Path,
    profile: pd.DataFrame,
    domain: pd.DataFrame,
    shifts: pd.DataFrame,
    matrix: pd.DataFrame,
    descriptors: pd.DataFrame,
    trio_distribution: pd.DataFrame,
    cec_function_distribution: pd.DataFrame,
) -> Path:
    overall = domain[domain["scope"].eq("all")].copy()
    valid_overall = overall[overall["status"].eq("ok")].copy()
    common = domain[domain["scope"].eq("common_10_20")].copy()
    valid_common = common[common["status"].eq("ok")].copy()
    best_common_auc = (
        float(valid_common["oof_directional_auc"].max())
        if not valid_common.empty
        else float("nan")
    )
    if np.isfinite(best_common_auc) and best_common_auc >= 0.80:
        interpretation = "至少一个特征组具有较强的 CEC/训练域可分性；应把 CEC 视为明显分布偏移的外部域。"
    elif np.isfinite(best_common_auc) and best_common_auc >= 0.60:
        interpretation = "共同 10/20D 上至少一个特征组具有中等的 CEC/训练域可分性；建议报告域偏移并做稳健性分析。"
    else:
        interpretation = "共同 10/20D 上未观察到强的 CEC/训练域可分性；仍需结合单特征结果解释。"
    top = shifts.sort_values(
        ["scope", "feature_set", "directional_auc", "ks_statistic"],
        ascending=[True, True, False, False],
    ).head(20)
    domain_display = domain[
        [
            "scope",
            "feature_set",
            "reference_source",
            "dimensions",
            "rows",
            "problems",
            "oof_auc",
            "oof_directional_auc",
            "fold_auc_mean",
            "fold_directional_auc_mean",
            "fold_auc_sd",
            "status",
        ]
    ].copy()
    stratified_display = domain[domain["scope"].str.startswith("dimension_")].copy()
    stratified_display = stratified_display[
        [
            "scope",
            "feature_set",
            "reference_source",
            "dimensions",
            "rows",
            "problems",
            "oof_auc",
            "oof_directional_auc",
            "fold_auc_mean",
            "fold_directional_auc_mean",
            "fold_auc_sd",
            "status",
        ]
    ]
    trio_display = trio_distribution[trio_distribution["scope"].eq("common_10_20")].copy()
    function_display = cec_function_distribution[
        (cec_function_distribution["feature_set"] == "descriptor_14")
        & (cec_function_distribution["dimension"] == 20)
    ].copy()
    lines = [
        "# CEC2017 与 BBOB/MA-BBOB 训练参考的特征分布比较",
        "",
        "## 结论摘要",
        "",
        f"- {interpretation}",
        "- 域分类器只用于诊断特征分布是否可分，不用于重新拟合 Decision/Selector，也不改变任何实验阈值。",
        "- all scope 汇总了全部维度；common_10_20 只比较 CEC 与训练参考共同拥有的 10D/20D，减少维度组成差异。",
        "- 报告原始 OOF AUC，并以 directional AUC=max(AUC, 1-AUC) 汇总可分性；0.5 表示无可分性，0.60/0.80 仅为本报告的解释性分界，不是项目主实验阈值。",
        "",
        "## 采样与特征工程协议",
        "",
        "- CEC2017：F1/F5/F9/F20/F24，10/20/30/50D，每个函数-维度使用 seed 1–5。",
        "- Query sampling：`lhs_50d`，`sample_base_seed=0`；与离线 BBOB/MA-BBOB descriptor 表使用相同采样设计。",
        "- Trajectory：`phase1_dynamic_budget_event_v1`；同一 34 列原生 behavior、31 列 B3、29 列 Decision 和 43 列 Selector 特征合同。",
        "",
        "## 数据粒度",
        "",
        profile.to_markdown(index=False),
        "",
        "## 特征合同",
        "",
        f"- B3 behavior：{len(B3_COLUMNS)} 列；固定 Decision 组：{len(DECISION_FEATURE_COLUMNS)} 列。",
        f"- Selector 组：{len(SELECTOR_COLUMNS)} 个 behavior + {len(DESCRIPTOR_COLUMNS)} 个 descriptor + remaining_budget_ratio，共 {len(SELECTOR_FEATURE_COLUMNS)} 列。",
        f"- state-level feature matrix：{len(matrix):,} 行；problem-level descriptor rows：{len(descriptors):,} 行。",
        "",
        "## 域分类器 OOF AUC",
        "",
        "共同维度的结论优先于 all scope；原始 AUC 小于 0.5 时，directional AUC 用于避免把分类方向反转误读为无偏移。",
        "",
        domain_display.to_markdown(index=False),
        "",
        "## 按维度和参考套件拆分的 OOF AUC",
        "",
        "10D/20D 是训练参考和 CEC 的共同维度；`reference_source` 表示与 CEC 进行比较的训练参考套件。",
        "",
        stratified_display.to_markdown(index=False),
        "",
        "## 单变量偏移最大的特征（按 directional AUC 排序）",
        "",
        top.to_markdown(index=False),
        "",
        "## 偏移特征的三方分布（共同 10/20D）",
        "",
        "表中报告 BBOB-train、MA-BBOB formal 和 CEC2017 的中位数及 IQR；特征按共同 10/20D 的单变量偏移排序，每组取前 5 个。",
        "",
        trio_display.to_markdown(index=False),
        "",
        "## CEC 函数级摘要（20D descriptor）",
        "",
        "该表用于查看 F1/F5/F9/F20/F24 的函数内异质性；完整 10/20/30/50D、descriptor/behavior 结果见函数级 CSV。",
        "",
        function_display.to_markdown(index=False),
        "",
        "## 解释边界",
        "",
        "- 这里的 train_reference 是当前 BBOB-train 与 MA-BBOB formal 的合并参考域；CEC2017 没有参与特征标准化、模型训练或阈值拟合。",
        "- CEC 的 30D/50D 在训练参考中没有同维度 strata，因此 all scope 结果不能单独解释为函数族偏移；common_10_20 是更保守的比较。",
        "- CEC 每个函数-维度只有 instance 1 的 descriptor 行，因此函数级 descriptor 表中的 n=1/IQR=0 只表示当前采样设计下的一条问题级记录，不表示 descriptor 没有采样不确定性。",
        "- 高 AUC 说明特征空间能够区分两个域，不等价于 Decision 在 CEC 上一定失效；需要与在线优化结果或迁移评价联合解释。",
        "",
        "## 产物",
        "",
        "- `cec2017_feature_shift_matrix.parquet`：同一 B3+descriptor+budget 特征工程矩阵。",
        "- `cec2017_feature_shift_profiles.parquet`：按 run×phase 聚合的分布诊断表。",
        "- `cec2017_feature_shift_univariate.csv`：KS、IQR 标准化中位数差和单变量 AUC。",
        "- `cec2017_feature_shift_domain_auc.csv`：分组 OOF 域分类 AUC。",
        "- `cec2017_feature_shift_domain_auc_stratified.csv`：按维度及 BBOB/MA-BBOB 参考套件拆分的 OOF 域分类 AUC。",
        "- `cec2017_feature_shift_trio_distributions.csv`：偏移特征的 BBOB、MA-BBOB、CEC 三方中位数/IQR。",
        "- `cec2017_feature_shift_cec_function_distributions.csv`：CEC 各函数/维度的 selected feature 中位数/IQR。",
    ]
    path = output_dir / "cec2017_feature_shift_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, descriptors = _build_matrix()
    profiles = _profile_states(matrix)
    profile = _profile_table(matrix, descriptors)

    feature_sets = {
        "descriptor_14": DESCRIPTOR_COLUMNS,
        "decision_29": DECISION_FEATURE_COLUMNS,
        "selector_43": SELECTOR_FEATURE_COLUMNS,
    }
    frame_by_set = {
        "descriptor_14": descriptors,
        "decision_29": profiles,
        "selector_43": profiles,
    }
    shift_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    for feature_set, columns in feature_sets.items():
        descriptor_frame = descriptors
        behavior_frame = profiles
        shift_rows.extend(_univariate_shift(descriptor_frame, DESCRIPTOR_COLUMNS, "descriptor_14", "all", None)) if feature_set == "descriptor_14" else None
        if feature_set == "descriptor_14":
            shift_rows.extend(_univariate_shift(descriptor_frame, DESCRIPTOR_COLUMNS, feature_set, "common_10_20", COMMON_DIMENSIONS))
            domain_rows.append(
                _domain_classifier_auc(
                    descriptor_frame,
                    columns,
                    feature_set,
                    "all",
                    None,
                    TRAIN_SOURCES,
                )
            )
            domain_rows.append(
                _domain_classifier_auc(
                    descriptor_frame,
                    columns,
                    feature_set,
                    "common_10_20",
                    COMMON_DIMENSIONS,
                    TRAIN_SOURCES,
                )
            )
        else:
            shift_rows.extend(_univariate_shift(behavior_frame, columns, feature_set, "all", None))
            shift_rows.extend(_univariate_shift(behavior_frame, columns, feature_set, "common_10_20", COMMON_DIMENSIONS))
            domain_rows.append(
                _domain_classifier_auc(
                    behavior_frame,
                    columns,
                    feature_set,
                    "all",
                    None,
                    TRAIN_SOURCES,
                )
            )
            domain_rows.append(
                _domain_classifier_auc(
                    behavior_frame,
                    columns,
                    feature_set,
                    "common_10_20",
                    COMMON_DIMENSIONS,
                    TRAIN_SOURCES,
                )
            )
    shifts = pd.DataFrame(shift_rows)
    for feature_set, columns in feature_sets.items():
        frame = frame_by_set[feature_set]
        for scope, dimensions, references in _stratified_domain_scopes()[2:]:
            domain_rows.append(
                _domain_classifier_auc(
                    frame,
                    columns,
                    feature_set,
                    scope,
                    dimensions,
                    references,
                )
            )
    domain = pd.DataFrame(domain_rows)
    selected_features = _top_common_features(shifts)
    trio_distribution = _trio_distribution_table(frame_by_set, selected_features)
    cec_function_distribution = _cec_function_distribution_table(
        frame_by_set,
        selected_features,
    )

    matrix.to_parquet(output_dir / "cec2017_feature_shift_matrix.parquet", index=False)
    profiles.to_parquet(output_dir / "cec2017_feature_shift_profiles.parquet", index=False)
    profile.to_csv(output_dir / "cec2017_feature_shift_profile.csv", index=False)
    shifts.to_csv(output_dir / "cec2017_feature_shift_univariate.csv", index=False)
    domain.to_csv(output_dir / "cec2017_feature_shift_domain_auc.csv", index=False)
    stratified_domain = domain[domain["scope"].str.startswith("dimension_")].copy()
    stratified_domain.to_csv(
        output_dir / "cec2017_feature_shift_domain_auc_stratified.csv",
        index=False,
    )
    trio_distribution.to_csv(
        output_dir / "cec2017_feature_shift_trio_distributions.csv",
        index=False,
    )
    cec_function_distribution.to_csv(
        output_dir / "cec2017_feature_shift_cec_function_distributions.csv",
        index=False,
    )
    report_path = _write_markdown(
        output_dir,
        profile,
        domain,
        shifts,
        matrix,
        descriptors,
        trio_distribution,
        cec_function_distribution,
    )

    valid_domain = domain[domain["status"].eq("ok")]
    summary = {
        "status": "ok",
        "analysis": "cec2017_feature_shift",
        "sources": [spec.name for spec in SOURCE_SPECS],
        "training_reference_sources": ["bbob_train", "mabbob_formal"],
        "cec_source": "cec2017_distribution_shift",
        "cec_functions": [1, 5, 9, 20, 24],
        "cec_dimensions": [10, 20, 30, 50],
        "common_dimensions": list(COMMON_DIMENSIONS),
        "sample_design": "lhs_50d",
        "sample_base_seed": 0,
        "trajectory_sampling_protocol": "phase1_dynamic_budget_event_v1",
        "trajectory_algorithms": sorted(matrix["algorithm"].astype(str).unique().tolist()),
        "trajectory_seeds": sorted(matrix["seed"].astype(int).unique().tolist()),
        "behavior_feature_count": len(BEHAVIOR_FEATURE_COLUMNS),
        "b3_behavior_feature_count": len(B3_COLUMNS),
        "decision_feature_count": len(DECISION_FEATURE_COLUMNS),
        "selector_feature_count": len(SELECTOR_FEATURE_COLUMNS),
        "descriptor_feature_count": len(DESCRIPTOR_COLUMNS),
        "state_rows": int(len(matrix)),
        "profile_rows": int(len(profiles)),
        "descriptor_rows": int(len(descriptors)),
        "nonfinite_matrix_cells": int(
            (~np.isfinite(matrix.loc[:, list(MATRIX_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce"))).sum().sum()
        ),
        "nonfinite_selector_cells": int(
            (~np.isfinite(matrix.loc[:, list(SELECTOR_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce"))).sum().sum()
        ),
        "nonfinite_decision_cells": int(
            (~np.isfinite(matrix.loc[:, list(DECISION_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce"))).sum().sum()
        ),
        "domain_auc": domain.to_dict(orient="records"),
        "stratified_domain_auc": stratified_domain.to_dict(orient="records"),
        "top_common_shift_features": {
            key: list(value) for key, value in selected_features.items()
        },
        "cec_function_distribution_rows": int(len(cec_function_distribution)),
        "max_valid_oof_auc_all_scope": (
            float(valid_domain[valid_domain["scope"].eq("all")]["oof_auc"].max())
            if not valid_domain[valid_domain["scope"].eq("all")].empty
            else None
        ),
        "max_valid_directional_oof_auc_all_scope": (
            float(
                valid_domain[valid_domain["scope"].eq("all")][
                    "oof_directional_auc"
                ].max()
            )
            if not valid_domain[valid_domain["scope"].eq("all")].empty
            else None
        ),
        "max_valid_directional_oof_auc_common_10_20": (
            float(
                valid_domain[valid_domain["scope"].eq("common_10_20")][
                    "oof_directional_auc"
                ].max()
            )
            if not valid_domain[valid_domain["scope"].eq("common_10_20")].empty
            else None
        ),
        "outputs": {
            "matrix": str(output_dir / "cec2017_feature_shift_matrix.parquet"),
            "profiles": str(output_dir / "cec2017_feature_shift_profiles.parquet"),
            "profile": str(output_dir / "cec2017_feature_shift_profile.csv"),
            "univariate": str(output_dir / "cec2017_feature_shift_univariate.csv"),
            "domain_auc": str(output_dir / "cec2017_feature_shift_domain_auc.csv"),
            "stratified_domain_auc": str(
                output_dir / "cec2017_feature_shift_domain_auc_stratified.csv"
            ),
            "trio_distributions": str(
                output_dir / "cec2017_feature_shift_trio_distributions.csv"
            ),
            "cec_function_distributions": str(
                output_dir / "cec2017_feature_shift_cec_function_distributions.csv"
            ),
            "report": str(report_path),
        },
    }
    (output_dir / "cec2017_feature_shift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CEC2017 and BBOB/MA-BBOB feature distributions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
