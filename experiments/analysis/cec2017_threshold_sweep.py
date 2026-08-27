"""Run a fixed-model CEC2017 Decision threshold sensitivity analysis.

The Decision model, feature columns, preprocessing, query design, Selector,
optimizer portfolio, and FE budgets are kept unchanged.  Only the scalar
threshold applied to the already fitted Decision score varies.  Threshold 0
reuses the previously measured online path; the other thresholds are measured
with three cyclic complete-path timing replays for both CEC Native and
Unit-cube coordinates.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision.online_controller_evaluate import _load_controller, _online_timing_base_order
from decision.model_protocol import PREDEFINED_THRESHOLD_MODE, SELECTED_MODEL_ALIAS
from experiments.analysis.cec2017_opportunity_alignment import _read_bbob_oof
from experiments.cli.cec2017_normalized_online_compare import _unit_cube_runtime
from experiments.cli.cec2017_representative_online_compare import (
    OnlineSelector,
    _aggregate_policy_rows,
    _run_policy_once,
)
from experiments.cli.cec2017_representative_online_compare import (
    _validate_feature_contract,
)
from experiments.phase1_batch_common import (
    load_config,
    selected_dimensions,
    selected_functions,
    validate_dynamic_collection_config,
)
from landscape_queries.specs import MAIN_QUERY_ID, get_query_spec
from selection_reference.model import load_selector_model


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/cec2017_representative_online_compare.yaml"
DEFAULT_TRAINING_SUMMARY = REPO / (
    "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision/"
    "full_decision_model_training_summary.json"
)
DEFAULT_SELECTOR_MODEL = REPO / (
    "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision/"
    "models/selector__query_full.joblib"
)
NATIVE_EXISTING = REPO / "outputs/cec2017_representative_online_compare"
UNIT_EXISTING = REPO / "outputs/cec2017_representative_online_compare_unit_cube"
DEFAULT_OUTPUT = REPO / "results/dataset_analysis/cec2017_threshold_sweep"
TIMING_REPETITIONS = 3

THRESHOLD_LABELS = {
    0.0: "threshold_0",
    -0.005: "threshold_minus_0.005",
    -0.01: "threshold_minus_0.01",
}

PALETTE = {"native": "#657A3A", "unit": "#C96B27", "charcoal": "#2F3136", "grid": "#D9DDE2", "paper": "#FBFBFA"}


def _threshold_specs() -> list[dict[str, Any]]:
    bbob = _read_bbob_oof()
    q95 = float(pd.to_numeric(bbob["decision_score"], errors="coerce").quantile(0.95))
    return [
        {"label": THRESHOLD_LABELS[0.0], "threshold": 0.0, "definition": "0"},
        {"label": THRESHOLD_LABELS[-0.005], "threshold": -0.005, "definition": "-0.005"},
        {"label": THRESHOLD_LABELS[-0.01], "threshold": -0.01, "definition": "-0.01"},
        {"label": "threshold_bbob_q95", "threshold": q95, "definition": "BBOB-train OOF q95"},
    ]


def _load_online_zero(condition: str, label: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = NATIVE_EXISTING if condition == "cec_native" else UNIT_EXISTING
    run_path = base / "online_comparison_run_metrics.parquet"
    timing_path = base / "online_comparison_timing_replays.parquet"
    run = pd.read_parquet(run_path)
    timing = pd.read_parquet(timing_path)
    run = run[run["policy_name"].astype(str).eq("predicted_G_FE_gt_0")].copy()
    timing = timing[timing["policy_name"].astype(str).eq("predicted_G_FE_gt_0")].copy()
    if len(run) != 100 or len(timing) != 300:
        raise ValueError(f"threshold=0 reuse coverage is invalid for {condition}: {len(run)}/{len(timing)}")
    for frame in (run, timing):
        frame["policy_name"] = label
        frame["threshold_label"] = label
        frame["threshold_value"] = float(threshold)
        frame["coordinate_condition"] = condition
        frame["threshold_source"] = "reused_verified_threshold_zero_online_replay"
    return run, timing


def _run_custom_condition(
    *,
    condition: str,
    config: dict[str, Any],
    functions: list[int],
    dimensions: list[int],
    seeds: list[int],
    controller: Any,
    selector: Any,
    query_spec: Any,
    threshold_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    custom_specs = [spec for spec in threshold_specs if spec["threshold"] != 0.0]
    run_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    context = _unit_cube_runtime() if condition == "cec_unit_cube" else nullcontext()
    with context:
        # Query inputs are part of the online protocol and must use the same
        # coordinate bounds as the policy path in the Unit-cube condition.
        from experiments.cli.cec2017_representative_online_compare import _prepare_online_query_inputs

        query_features, query_samples = _prepare_online_query_inputs(
            config=config,
            query_id=MAIN_QUERY_ID,
            query_spec=query_spec,
            functions=functions,
            dimensions=dimensions,
        )
        for function in functions:
            for dimension in dimensions:
                for seed in seeds:
                    print(
                        f"threshold sweep {condition} F{function:02d} D{dimension} seed={seed}",
                        flush=True,
                    )
                    fe_total = int(config["FE_total_by_dimension"][dimension])
                    stage_a: dict[str, dict[str, Any]] = {}
                    replay_rows: list[dict[str, Any]] = []
                    policy_specs: dict[str, dict[str, Any]] = {}
                    for spec in custom_specs:
                        custom_controller = replace(
                            controller,
                            threshold=float(spec["threshold"]),
                            threshold_mode=f"fixed_sensitivity_{spec['label']}",
                        )
                        policy_specs[spec["label"]] = {
                            "kind": "online",
                            "controller": custom_controller,
                            "policy_spec": {"policy_name": "current_controller"},
                        }
                    order = _online_timing_base_order(
                        path_count=len(custom_specs),
                        function=function,
                        dimension=dimension,
                        seed=seed,
                        random_repetitions=1,
                    )
                    for spec in custom_specs:
                        label = spec["label"]
                        result = _run_policy_once(
                            policy_name=label,
                            policy_spec=policy_specs[label],
                            config=config,
                            function=function,
                            dimension=dimension,
                            seed=seed,
                            fe_total=fe_total,
                            selector=selector,
                            controller=policy_specs[label]["controller"],
                            query_feature_row=query_features[(function, dimension)],
                            query_sample_row=query_samples[(function, dimension)],
                        )
                        result.update(
                            {
                                "stage": "stage_a_scientific",
                                "timing_repetition": None,
                                "timing_order_position": int(order[list(s["label"] for s in custom_specs).index(label)]),
                                "function": int(function),
                                "dimension": int(dimension),
                                "seed": int(seed),
                                "threshold_label": label,
                                "threshold_value": float(spec["threshold"]),
                                "coordinate_condition": condition,
                                "threshold_source": "new_online_replay_fixed_model_custom_threshold",
                            }
                        )
                        stage_a[label] = result
                    for timing_repetition in range(TIMING_REPETITIONS):
                        rotated = order[timing_repetition:] + order[:timing_repetition]
                        for order_position, policy_index in enumerate(rotated):
                            spec = custom_specs[int(policy_index)]
                            label = spec["label"]
                            result = _run_policy_once(
                                policy_name=label,
                                policy_spec=policy_specs[label],
                                config=config,
                                function=function,
                                dimension=dimension,
                                seed=seed,
                                fe_total=fe_total,
                                selector=selector,
                                controller=policy_specs[label]["controller"],
                                query_feature_row=query_features[(function, dimension)],
                                query_sample_row=query_samples[(function, dimension)],
                            )
                            result.update(
                                {
                                    "stage": "stage_b_timing_replay",
                                    "timing_source": "measured_complete_policy_path",
                                    "timing_repetition": int(timing_repetition),
                                    "timing_order_position": int(order_position),
                                    "function": int(function),
                                    "dimension": int(dimension),
                                    "seed": int(seed),
                                    "threshold_label": label,
                                    "threshold_value": float(spec["threshold"]),
                                    "coordinate_condition": condition,
                                    "threshold_source": "new_online_replay_fixed_model_custom_threshold",
                                }
                            )
                            replay_rows.append(result)
                    for spec in custom_specs:
                        label = spec["label"]
                        replays = [row for row in replay_rows if row["policy_name"] == label]
                        aggregated = _aggregate_policy_rows(
                            scientific=stage_a[label],
                            replays=replays,
                            policy_name=label,
                            function=function,
                            dimension=dimension,
                            seed=seed,
                            random_target=None,
                        )
                        aggregated.update(
                            {
                                "threshold_label": label,
                                "threshold_value": float(spec["threshold"]),
                                "coordinate_condition": condition,
                                "threshold_source": "new_online_replay_fixed_model_custom_threshold",
                            }
                        )
                        run_rows.append(aggregated)
                    timing_rows.extend(replay_rows)
    return pd.DataFrame(run_rows), pd.DataFrame(timing_rows)


def _summarize(run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, label, threshold), group in run.groupby(
        ["coordinate_condition", "threshold_label", "threshold_value"], sort=True
    ):
        hits = group["target_hit_observed"].astype(bool)
        queried = group[group["query_called"].astype(bool)]
        n_hits = int(hits.sum())
        rows.append(
            {
                "coordinate_condition": condition,
                "threshold_label": label,
                "threshold_value": float(threshold),
                "runs": int(len(group)),
                "query_count": int(group["query_called"].astype(bool).sum()),
                "query_call_rate": float(group["query_called"].astype(bool).mean()),
                "query_handoff_count": int(queried["handoff_required"].astype(bool).sum()),
                "query_selected_prefix_rate": float(queried["selected_equals_prefix"].astype(bool).mean()) if len(queried) else np.nan,
                "query_selected_algorithm_counts": "; ".join(
                    f"{str(name)}={int(count)}" for name, count in queried["selected_algorithm"].value_counts().sort_index().items()
                ) if len(queried) else "",
                "decision_score_median": float(pd.to_numeric(group["decision_score"], errors="coerce").median()),
                "decision_check_count_mean": float(pd.to_numeric(group["decision_check_count"], errors="coerce").mean()),
                "target_hits": n_hits,
                "target_hit_rate": float(hits.mean()),
                "endpoint_success_rate": float(group["endpoint_success"].astype(bool).mean()),
                "ERT_FE": float(group["ert_FE_contribution"].astype(float).sum() / n_hits) if n_hits else np.nan,
                "ERT_time_seconds": float(group["time_ert_seconds_contribution"].astype(float).sum() / n_hits) if n_hits else np.nan,
                "median_time_to_target_seconds_among_hits": float(
                    pd.to_numeric(group.loc[hits, "time_to_target_seconds_median"], errors="coerce").dropna().median()
                ) if hits.any() else np.nan,
                "mean_log10_gap": float(pd.to_numeric(group["log10_gap"], errors="coerce").mean()),
                "median_log10_gap": float(pd.to_numeric(group["log10_gap"], errors="coerce").median()),
                "median_full_run_wall_clock_seconds": float(
                    pd.to_numeric(group["full_run_wall_clock_seconds"], errors="coerce").median()
                ),
                "all_timing_replays_measured": bool(
                    group["timing_source"].astype(str).eq("measured_complete_policy_path").all()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["coordinate_condition", "threshold_value"]).reset_index(drop=True)


def _plot_summary(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PALETTE["paper"],
            "figure.facecolor": PALETTE["paper"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "savefig.dpi": 220,
        }
    )
    order = summary[["threshold_label", "threshold_value"]].drop_duplicates().sort_values("threshold_value")
    labels = order["threshold_label"].tolist()
    tick_label_map = {
        "threshold_0": "0",
        "threshold_minus_0.005": "−0.005",
        "threshold_minus_0.01": "−0.01",
        "threshold_bbob_q95": "BBOB q95",
    }
    tick_labels = [tick_label_map[label] for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    panels = [
        ("query_call_rate", "Query call rate", "rate"),
        ("query_count", "Total query count", "count"),
        ("ERT_FE", "ERT (FE)", "ert"),
        ("ERT_time_seconds", "ERT (seconds)", "ert"),
    ]
    for ax, (column, title, kind) in zip(axes.ravel(), panels):
        for condition, color, label in [
            ("cec_native", PALETTE["native"], "CEC Native"),
            ("cec_unit_cube", PALETTE["unit"], "CEC Unit-cube"),
        ]:
            group = summary[summary["coordinate_condition"].eq(condition)].set_index("threshold_label").reindex(labels)
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            if kind == "rate":
                values = values * 100.0
            ax.plot(x, values, marker="o", linewidth=1.8, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_title(title)
        ax.set_xlabel("Decision threshold")
        if kind == "rate":
            ax.set_ylabel("percent")
        elif kind == "count":
            ax.set_ylabel("calls / 100 runs")
        elif column == "ERT_FE":
            ax.set_ylabel("FE")
            ax.set_yscale("log")
        else:
            ax.set_ylabel("seconds")
            ax.set_yscale("log")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("CEC2017 fixed-model Decision threshold sweep", x=0.06, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "fig_threshold_sweep.png", bbox_inches="tight")
    plt.close(fig)


def _write_report(output: Path, summary: pd.DataFrame, threshold_specs: list[dict[str, Any]], run_rows: int, timing_rows: int) -> None:
    q95 = next(spec for spec in threshold_specs if spec["label"] == "threshold_bbob_q95")
    table = summary.copy()
    table["threshold"] = table["threshold_value"].map(lambda value: f"{value:.12g}")
    table["query_call_rate"] = (100.0 * table["query_call_rate"]).map(lambda value: f"{value:.2f}%")
    table["query_selected_prefix_rate"] = (100.0 * table["query_selected_prefix_rate"]).map(
        lambda value: "NA" if not np.isfinite(value) else f"{value:.2f}%"
    )
    table["target_hit_rate"] = (100.0 * table["target_hit_rate"]).map(lambda value: f"{value:.2f}%")
    table["ERT_FE"] = table["ERT_FE"].map(lambda value: "NA" if not np.isfinite(value) else f"{value:.2f}")
    table["ERT_time_seconds"] = table["ERT_time_seconds"].map(lambda value: "NA" if not np.isfinite(value) else f"{value:.6g}")
    table = table[
        ["coordinate_condition", "threshold_label", "threshold", "query_count", "query_call_rate", "query_handoff_count", "query_selected_prefix_rate", "query_selected_algorithm_counts", "target_hit_rate", "ERT_FE", "ERT_time_seconds", "decision_score_median"]
    ].rename(
        columns={
            "coordinate_condition": "condition",
            "threshold_label": "threshold label",
            "query_count": "query count",
            "query_call_rate": "query call rate",
            "query_handoff_count": "query handoffs",
            "query_selected_prefix_rate": "prefix selected rate",
            "query_selected_algorithm_counts": "selected algorithms among queries",
            "target_hit_rate": "target hit rate",
            "ERT_FE": "ERT FE",
            "ERT_time_seconds": "ERT seconds",
            "decision_score_median": "score median",
        }
    )
    lines = [
        "# CEC2017 固定 Decision 模型阈值敏感性分析",
        "",
        "> 主模型、29 个行为特征、预处理、Selector、query 设计和 FE 预算均保持不变；仅改变部署时使用的标量 Decision threshold。该分析不替换主测评阈值。",
        "",
        "## 技术摘要",
        "",
        "- 目标是区分‘阈值位置过高’与‘模型排序/score 上尾不足’：降低阈值后若 query 增加且 ERT 改善，说明阈值位置至少是必要因素；若大量触发但 ERT 不改善，则说明排序能力或选择后的路径收益不足。",
        f"- 四个阈值为 `0`、`-0.005`、`-0.01` 和 BBOB-train OOF score q95=`{q95['threshold']:.12g}`；后者由固定 BBOB-train OOF score 的 95% 分位数计算，不使用 CEC 数据。",
        f"- 结果覆盖 {run_rows} 条科学 run 行和 {timing_rows} 条真实 timing replay 行；每个新阈值×坐标条件×function×dimension×seed 有 3 次完整路径 replay。",
        "",
        "## 指标定义与比较基准",
        "",
        "- query count 是 100 条 run 中 `query_called=True` 的次数；由于每条 trajectory 最多 query 一次，等于调用次数。query call rate 是该次数除以 100。",
        "- `ERT_FE` 与 `ERT seconds` 沿用当前 online runner 的 run-level 首次命中定义：命中 run 使用首次命中 FE/秒数，未命中 run 贡献完整路径 FE/秒数，再除以命中 run 数。",
        "- 时间指标来自真实 `measured_complete_policy_path` replay，不由 component runtime 拼接。threshold=0 使用此前已经完成并核对过的同协议真实 replay。",
        "",
        "## Threshold sweep 结果",
        "",
        table.to_markdown(index=False),
        "",
        "## 结果解释",
        "",
        "图中 query call rate 和 query count 描述阈值是否让控制器开始行动；两项 ERT 描述这些新增调用是否转化为实际优化收益。应同时看命中率、ERT_FE 和 ERT seconds，不能只依据调用次数判断改善。",
        "",
        "## 稳健性与限制",
        "",
        "- CEC Native 与 Unit-cube 分开报告；Unit-cube 只改变优化器内部坐标参数化，并使用相同的固定模型与 Selector。",
        "- BBOB q95 是 score 分布位置的敏感性参照，不是重新拟合的 CEC 校准阈值；本分析没有用 CEC 结果拟合任何参数。",
        "- 若低阈值触发后 Selector 仍频繁选择 `continue_current`，或 ERT 与 Always/No-query 基线接近，则降低阈值只解决了调用门槛，不能证明模型有足够的动作排序能力。",
        "",
        "## 输出",
        "",
        "- `threshold_sweep_run_metrics.parquet`：每个阈值、条件和 run 的科学端点。",
        "- `threshold_sweep_timing_replays.parquet`：3 次 cyclic complete-path 真实 timing replay。",
        "- `threshold_sweep_summary.csv`：按条件和阈值汇总的 query、命中率与 ERT。",
        "- `fig_threshold_sweep.png`：阈值与 query/ERT 对比图。",
        "",
        "## 下一步建议",
        "",
        "若低阈值显著增加 query 但 ERT 不改善，下一步应分析触发后 Selector 的动作选择和 query-adjusted action loss；若低阈值同步改善 ERT，再把阈值位置作为独立部署敏感性因素报告，不直接改写主模型阈值。",
    ]
    (output / "cec2017_threshold_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(output_dir: Path, overwrite: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(DEFAULT_CONFIG)
    validate_dynamic_collection_config(config)
    functions = selected_functions(config, None)
    dimensions = selected_dimensions(config, None)
    seeds = [int(seed) for seed in config["seeds"]]
    threshold_specs = _threshold_specs()
    controller = _load_controller(DEFAULT_TRAINING_SUMMARY, SELECTED_MODEL_ALIAS, PREDEFINED_THRESHOLD_MODE)
    selector = OnlineSelector(load_selector_model(DEFAULT_SELECTOR_MODEL))
    _validate_feature_contract(controller, selector)
    query_spec = get_query_spec(MAIN_QUERY_ID)
    run_frames: list[pd.DataFrame] = []
    timing_frames: list[pd.DataFrame] = []
    for condition in ("cec_native", "cec_unit_cube"):
        zero_spec = threshold_specs[0]
        zero_run, zero_timing = _load_online_zero(condition, zero_spec["label"], zero_spec["threshold"])
        run_frames.append(zero_run)
        timing_frames.append(zero_timing)
        custom_run, custom_timing = _run_custom_condition(
            condition=condition,
            config=config,
            functions=functions,
            dimensions=dimensions,
            seeds=seeds,
            controller=controller,
            selector=selector,
            query_spec=query_spec,
            threshold_specs=threshold_specs,
        )
        run_frames.append(custom_run)
        timing_frames.append(custom_timing)
    run = pd.concat(run_frames, ignore_index=True, sort=False)
    timing = pd.concat(timing_frames, ignore_index=True, sort=False)
    expected_runs = len(functions) * len(dimensions) * len(seeds) * len(threshold_specs) * 2
    if len(run) != expected_runs:
        raise ValueError(f"threshold sweep run rows={len(run)}, expected={expected_runs}")
    if len(timing) != expected_runs * TIMING_REPETITIONS:
        raise ValueError(f"threshold sweep timing rows={len(timing)}, expected={expected_runs * TIMING_REPETITIONS}")
    if not timing["timing_source"].astype(str).eq("measured_complete_policy_path").all():
        raise ValueError("all threshold-sweep timing rows must be measured complete policy paths")
    summary = _summarize(run)
    run.to_parquet(output_dir / "threshold_sweep_run_metrics.parquet", index=False)
    timing.to_parquet(output_dir / "threshold_sweep_timing_replays.parquet", index=False)
    summary.to_csv(output_dir / "threshold_sweep_summary.csv", index=False)
    _plot_summary(summary, output_dir)
    _write_report(output_dir, summary, threshold_specs, len(run), len(timing))
    result = {
        "status": "ok",
        "experiment": "cec2017_fixed_model_threshold_sweep",
        "functions": functions,
        "dimensions": dimensions,
        "seeds": seeds,
        "thresholds": threshold_specs,
        "model_name": controller.model_name,
        "feature_group": controller.feature_group,
        "feature_count": len(controller.feature_columns),
        "selector_model": str(DEFAULT_SELECTOR_MODEL),
        "run_rows": int(len(run)),
        "timing_rows": int(len(timing)),
        "timing_repetitions": TIMING_REPETITIONS,
        "data_leakage_check": {
            "cec_rows_used_for_model_fit": 0,
            "cec_rows_used_for_threshold_fit": 0,
            "model_refit": False,
            "threshold_fit_on_cec": False,
        },
        "outputs": {
            "run_metrics": str(output_dir / "threshold_sweep_run_metrics.parquet"),
            "timing_replays": str(output_dir / "threshold_sweep_timing_replays.parquet"),
            "summary": str(output_dir / "threshold_sweep_summary.csv"),
            "report": str(output_dir / "cec2017_threshold_sweep_report.md"),
            "figure": str(output_dir / "fig_threshold_sweep.png"),
        },
    }
    (output_dir / "cec2017_threshold_sweep_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_analysis(args.output_dir.resolve(), args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
