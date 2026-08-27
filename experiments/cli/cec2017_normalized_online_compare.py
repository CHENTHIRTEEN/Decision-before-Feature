"""Run an independent CEC2017 online comparison with unit-cube coordinates.

This sensitivity runner keeps the Decision model, threshold, Selector, FE
budget, query design, seeds, functions, and policies unchanged. Only the
coordinate parameterization seen by the optimizers and the online query is
changed: u lies in [0, 1]^d and the benchmark objective is evaluated at
lower + (upper - lower) * u. The native CEC online result directory is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import behavior.features as behavior_features_module
import benchmarks.factory as benchmark_factory
import decision.online_controller_evaluate as online_module
import experiments.cli.cec2017_representative_online_compare as native_module
import trajectory.window_statistics as window_statistics_module
from benchmarks import make_problem as factory_make_problem
from benchmarks.core import Problem


REPO = Path(__file__).resolve().parents[2]
NATIVE_OUTPUT = REPO / "outputs/cec2017_representative_online_compare"
NORMALIZED_OUTPUT = REPO / "outputs/cec2017_representative_online_compare_unit_cube"
NATIVE_CONFIG = REPO / "configs/cec2017_representative_online_compare.yaml"
NATIVE_TRAINING_SUMMARY = REPO / native_module.DEFAULT_TRAINING_SUMMARY
NATIVE_SELECTOR_MODEL = REPO / native_module.DEFAULT_SELECTOR_MODEL

KEY_COLUMNS = ["policy_name", "function", "dimension", "seed"]
COMPARISON_COLUMNS = [
    "query_called",
    "trigger_FE_ratio",
    "decision_score",
    "decision_check_count",
    "FE_used",
    "final_gap",
    "log10_gap",
    "target_hit_observed",
    "endpoint_success",
    "ert_FE_contribution",
    "time_ert_seconds_contribution",
    "runtime_full_run_wall_clock_median",
    "time_to_target_seconds_median",
    "selected_algorithm",
    "path_status",
]


def _unit_bounds_for_problem(problem_id: str) -> tuple[np.ndarray, np.ndarray]:
    text = str(problem_id)
    if "cec2017_" not in text or "_d" not in text:
        raise ValueError(f"unit-cube bounds only support CEC2017 problem IDs, got {problem_id!r}")
    try:
        dimension = int(text.rsplit("_d", 1)[1])
    except ValueError as exc:
        raise ValueError(f"cannot parse dimension from CEC2017 problem ID: {problem_id!r}") from exc
    if dimension <= 0:
        raise ValueError(f"problem dimension must be positive, got {dimension}")
    return np.zeros(dimension, dtype=float), np.ones(dimension, dtype=float)


def _make_unit_cube_problem(config: dict[str, Any]) -> Problem:
    base_problem = factory_make_problem(config)
    if not str(base_problem.problem_id).startswith("cec2017_"):
        base_problem.close()
        raise ValueError("unit-cube CEC runner received a non-CEC problem")
    native_lower = np.asarray(base_problem.lower_bounds, dtype=float).copy()
    native_upper = np.asarray(base_problem.upper_bounds, dtype=float).copy()
    native_span = native_upper - native_lower
    if np.any(native_span <= 0.0) or not np.isfinite(native_span).all():
        base_problem.close()
        raise ValueError("CEC problem bounds must have finite positive spans")
    unit_lower, unit_upper = _unit_bounds_for_problem(base_problem.problem_id)

    def objective(unit_population: np.ndarray) -> np.ndarray:
        values = np.asarray(unit_population, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        native_population = native_lower[None, :] + values * native_span[None, :]
        return base_problem.evaluate(native_population)

    return Problem(
        problem_id=base_problem.problem_id,
        function_id=base_problem.function_id,
        family=base_problem.family,
        dimension=base_problem.dimension,
        suite_code=base_problem.suite_code,
        function_number=base_problem.function_number,
        instance_number=base_problem.instance_number,
        bounds=np.column_stack([unit_lower, unit_upper]),
        objective=objective,
        reference_value=base_problem.reference_value,
        close_callback=base_problem.close,
        cv_group_id=base_problem.cv_group_id,
        boundary_handling=base_problem.boundary_handling,
    )


@contextmanager
def _unit_cube_runtime() -> Iterator[None]:
    """Temporarily route all CEC online components through unit coordinates."""

    original_bindings = {
        "native_make_problem": native_module.make_problem,
        "online_make_problem": online_module.make_problem,
        "factory_problem_bounds": benchmark_factory.problem_bounds,
        "behavior_problem_bounds": behavior_features_module.problem_bounds,
        "window_problem_bounds": window_statistics_module.problem_bounds,
    }
    native_module.make_problem = _make_unit_cube_problem
    online_module.make_problem = _make_unit_cube_problem
    benchmark_factory.problem_bounds = _unit_bounds_for_problem
    behavior_features_module.problem_bounds = _unit_bounds_for_problem
    window_statistics_module.problem_bounds = _unit_bounds_for_problem
    try:
        yield
    finally:
        native_module.make_problem = original_bindings["native_make_problem"]
        online_module.make_problem = original_bindings["online_make_problem"]
        benchmark_factory.problem_bounds = original_bindings["factory_problem_bounds"]
        behavior_features_module.problem_bounds = original_bindings["behavior_problem_bounds"]
        window_statistics_module.problem_bounds = original_bindings["window_problem_bounds"]


def _read_run_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(KEY_COLUMNS + COMPARISON_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"run metrics are missing columns: {missing}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"run metrics contain duplicate keys: {path}")
    return frame


def _compare_run_metrics(native: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    native_keys = set(map(tuple, native[KEY_COLUMNS].itertuples(index=False, name=None)))
    normalized_keys = set(map(tuple, normalized[KEY_COLUMNS].itertuples(index=False, name=None)))
    if native_keys != normalized_keys:
        raise ValueError(
            f"native and normalized run coverage differs: native={len(native_keys)}, normalized={len(normalized_keys)}"
        )
    native_view = native[KEY_COLUMNS + COMPARISON_COLUMNS].rename(
        columns={column: f"{column}_native" for column in COMPARISON_COLUMNS}
    )
    normalized_view = normalized[KEY_COLUMNS + COMPARISON_COLUMNS].rename(
        columns={column: f"{column}_unit_cube" for column in COMPARISON_COLUMNS}
    )
    comparison = normalized_view.merge(
        native_view, on=KEY_COLUMNS, how="inner", validate="one_to_one"
    )
    for column in (
        "trigger_FE_ratio",
        "decision_score",
        "decision_check_count",
        "FE_used",
        "final_gap",
        "log10_gap",
        "ert_FE_contribution",
        "time_ert_seconds_contribution",
        "runtime_full_run_wall_clock_median",
        "time_to_target_seconds_median",
    ):
        comparison[f"delta_{column}"] = (
            pd.to_numeric(comparison[f"{column}_unit_cube"], errors="coerce")
            - pd.to_numeric(comparison[f"{column}_native"], errors="coerce")
        )
    comparison["delta_query_called"] = (
        comparison["query_called_unit_cube"].astype(bool).astype(int)
        - comparison["query_called_native"].astype(bool).astype(int)
    )
    comparison["delta_target_hit_observed"] = (
        comparison["target_hit_observed_unit_cube"].astype(bool).astype(int)
        - comparison["target_hit_observed_native"].astype(bool).astype(int)
    )
    return comparison.sort_values(KEY_COLUMNS, ignore_index=True)


def _safe_ert(frame: pd.DataFrame, column: str) -> float | None:
    hits = frame["target_hit_observed"].astype(bool)
    hit_count = int(hits.sum())
    if hit_count == 0:
        return None
    # ERT includes the full-budget contribution of unsuccessful runs.
    # Filtering to hit rows would report the conditional mean hit FE rather
    # than the run-level expected running time used by the main summary.
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.sum() / hit_count)


def _summarize_policy(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        called = group["query_called"].astype(bool)
        row.update(
            {
                "runs": int(len(group)),
                "query_call_rate": float(called.mean()),
                "endpoint_success_rate": float(group["endpoint_success"].astype(bool).mean()),
                "target_hit_observed_rate": float(group["target_hit_observed"].astype(bool).mean()),
                "ERT_FE": _safe_ert(group, "ert_FE_contribution"),
                "ERT_time_seconds": _safe_ert(group, "time_ert_seconds_contribution"),
                "mean_log10_gap": float(pd.to_numeric(group["log10_gap"], errors="coerce").mean()),
                "mean_runtime_seconds": float(
                    pd.to_numeric(group["runtime_full_run_wall_clock_median"], errors="coerce").mean()
                ),
                "mean_trigger_FE_ratio_called": float(
                    pd.to_numeric(group.loc[called, "trigger_FE_ratio"], errors="coerce").mean()
                ),
                "mean_decision_check_count": float(
                    pd.to_numeric(group["decision_check_count"], errors="coerce").mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_policy_comparison(native: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    native_summary = _summarize_policy(native, ["policy_name"]).add_suffix("_native")
    normalized_summary = _summarize_policy(normalized, ["policy_name"]).add_suffix("_unit_cube")
    result = normalized_summary.merge(
        native_summary,
        left_on="policy_name_unit_cube",
        right_on="policy_name_native",
        how="inner",
        validate="one_to_one",
    )
    result["policy_name"] = result["policy_name_unit_cube"]
    for column in (
        "query_call_rate",
        "endpoint_success_rate",
        "target_hit_observed_rate",
        "ERT_FE",
        "ERT_time_seconds",
        "mean_log10_gap",
        "mean_runtime_seconds",
        "mean_trigger_FE_ratio_called",
        "mean_decision_check_count",
    ):
        result[f"delta_{column}"] = (
            pd.to_numeric(result[f"{column}_unit_cube"], errors="coerce")
            - pd.to_numeric(result[f"{column}_native"], errors="coerce")
        )
    columns = ["policy_name"]
    columns.extend(
        column
        for column in result.columns
        if column not in {"policy_name", "policy_name_native", "policy_name_unit_cube"}
    )
    return result[columns].sort_values("policy_name", ignore_index=True)


def _fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _write_report(
    output: Path,
    policy_comparison: pd.DataFrame,
) -> None:
    lines = [
        "# CEC2017 单位立方体搜索坐标 online 敏感性测评",
        "",
        "> 该测评仅改变优化器内部的坐标参数化，不修改 Decision 模型、阈值、Selector、query 设计、CEC 目标函数、FE 总预算或主实验结果。",
        "",
        "## 实验条件",
        "",
        "- Native 条件使用 CEC 原始坐标；Unit-cube 条件使用 u∈[0,1]^d，并以 x=lower+(upper-lower)u 评价同一个 CEC 目标函数。",
        "- 两个条件均使用当前固定线形成熟度模型、固定 Selector、五个策略、F01/F05/F09/F20/F24、10/20/30/50D 和 seed 1--5。",
        "- 主 query 仍为 descriptor_cheap_invariant / lhs_50d，query FE 仍为总 FE 的 5%；Unit-cube 条件重新生成与单位坐标一致的 query 样本，并在线实测完整路径时间。",
        "- 每条 trajectory 仍按最早满足阈值的机会执行最多一次 query；未触发 run 不执行 query。",
        "",
        "## 汇总比较",
        "",
        "| policy | query call rate native | query call rate unit-cube | Δ call rate | endpoint success native | endpoint success unit-cube | ERT FE native | ERT FE unit-cube | ERT time native (s) | ERT time unit-cube (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in policy_comparison.iterrows():
        lines.append(
            "| {policy} | {native_call} | {unit_call} | {delta_call} | {native_success} | {unit_success} | {native_ert} | {unit_ert} | {native_time} | {unit_time} |".format(
                policy=row["policy_name"],
                native_call=_fmt(row["query_call_rate_native"]),
                unit_call=_fmt(row["query_call_rate_unit_cube"]),
                delta_call=_fmt(row["delta_query_call_rate"]),
                native_success=_fmt(row["endpoint_success_rate_native"]),
                unit_success=_fmt(row["endpoint_success_rate_unit_cube"]),
                native_ert=_fmt(row["ERT_FE_native"]),
                unit_ert=_fmt(row["ERT_FE_unit_cube"]),
                native_time=_fmt(row["ERT_time_seconds_native"]),
                unit_time=_fmt(row["ERT_time_seconds_unit_cube"]),
            )
        )
    decision_rows = policy_comparison[
        policy_comparison["policy_name"].isin(
            ["predicted_G_FE_gt_0", "predicted_g_fe_selected_path_gt_0.2997557291"]
        )
    ]
    if not decision_rows.empty:
        native_call = float(decision_rows["query_call_rate_native"].mean())
        unit_call = float(decision_rows["query_call_rate_unit_cube"].mean())
        call_delta = unit_call - native_call
        interpretation = (
            "单位坐标条件提高了两个决策器策略的平均触发率。"
            if call_delta > 1e-12
            else "单位坐标条件没有提高两个决策器策略的平均触发率。"
        )
    else:
        interpretation = "当前结果中没有两个决策器策略的可比较行。"
    lines.extend(
        [
            "",
            "## 解释",
            "",
            f"在本次结果中，{interpretation} 这只能说明优化器坐标参数化改变后的在线行为是否更接近当前 Decision 触发区域，不能单独证明 CEC 与 BBOB 的性能差异完全由边界尺度造成。",
            "若 Unit-cube 条件的 query call rate、trigger FE ratio、ERT FE 或 ERT time 同时改善，应理解为边界单位与优化器步长/内部参数的交互影响；若只有触发率改变而 ERT 未改善，则说明决策器触发域变化不等于优化性能改善。",
            "",
            "## 输出",
            "",
            f"- Unit-cube online 结果目录：{output}",
            "- online_comparison_run_metrics.parquet：单位坐标条件的 run-level online 结果。",
            "- coordinate_normalization_run_comparison.csv：与 native 结果逐 run 对齐的差异。",
            "- coordinate_normalization_policy_comparison.csv：按 policy 汇总的 call rate、ERT、终点和 runtime 差异。",
            "- online_comparison_report.md：入口生成的单位坐标运行报告。",
            "",
            "## 下一步建议",
            "",
            "若单位坐标条件仍未触发，应继续按同一 opportunity 对齐 Decision score 与 29 个行为特征，区分是行为轨迹未进入训练域，还是阈值/score 校准造成触发不足；不重新拟合当前模型或阈值。",
        ]
    )
    (output / "coordinate_normalization_sensitivity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_analysis(output_dir: Path, overwrite: bool) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录已有内容：{output_dir}；请使用 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _unit_cube_runtime():
        normalized_summary = native_module.run_experiment(
            config_path=NATIVE_CONFIG,
            training_summary_path=NATIVE_TRAINING_SUMMARY,
            selector_model_path=NATIVE_SELECTOR_MODEL,
            output_dir=output_dir,
            only_functions=None,
            only_dimensions=None,
            only_seeds=None,
            overwrite=overwrite,
        )

    native_metrics = _read_run_metrics(NATIVE_OUTPUT / "online_comparison_run_metrics.parquet")
    normalized_metrics = _read_run_metrics(output_dir / "online_comparison_run_metrics.parquet")
    run_comparison = _compare_run_metrics(native_metrics, normalized_metrics)
    policy_comparison = _make_policy_comparison(native_metrics, normalized_metrics)
    run_comparison.to_csv(output_dir / "coordinate_normalization_run_comparison.csv", index=False)
    policy_comparison.to_csv(output_dir / "coordinate_normalization_policy_comparison.csv", index=False)

    normalized_summary.update(
        {
            "coordinate_space_mode": "unit_cube_internal_coordinates",
            "coordinate_mapping": "x_native = lower + (upper - lower) * u",
            "native_comparison_output": str(NATIVE_OUTPUT),
            "decision_model_modified": False,
            "threshold_modified": False,
            "selector_modified": False,
            "query_fe_fraction_modified": False,
            "objective_function_modified": False,
            "run_level_comparison_rows": int(len(run_comparison)),
            "policy_comparison_rows": int(len(policy_comparison)),
            "outputs": {
                **dict(normalized_summary.get("outputs", {})),
                "run_comparison": str(output_dir / "coordinate_normalization_run_comparison.csv"),
                "policy_comparison": str(output_dir / "coordinate_normalization_policy_comparison.csv"),
                "sensitivity_report": str(output_dir / "coordinate_normalization_sensitivity_report.md"),
            },
        }
    )
    (output_dir / "online_comparison_summary.json").write_text(
        json.dumps(normalized_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir, policy_comparison)
    return normalized_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=NORMALIZED_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_analysis(args.output_dir, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
