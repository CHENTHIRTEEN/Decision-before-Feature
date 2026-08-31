"""Build a fixed literature-to-P_balanced compatibility crosswalk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
DEFAULT_OUT = RESULTS / "analysis_v5/literature_pbalanced_crosswalk"
REPORT = BASE / "analysis_v5/literature_pbalanced_crosswalk.md"
PORTFOLIO = ("shade", "lshade", "cso")
STATUS_VALUES = {"PASS", "PARTIAL", "FAIL", "NOT_APPLICABLE", "NOT_MATERIALIZED"}


STATE_CONTRACT = {
    "natural": {
        "info_time": "state FE 之前的 natural trajectory Behavior",
        "action": "continue current 或 switch 到 P_balanced 其余两个算法",
        "label": "1000-FE FE-indexed action loss / practical switch_required",
        "split": "function-grouped OOF；within-problem LOSO 作诊断",
    },
    "query": {
        "info_time": "独立 query 执行前只可用 pre-query Behavior；query descriptors 只在 gate 触发后可用",
        "action": "execute/skip fixed query，然后由 downstream Selector 选择后续动作",
        "label": "paired skip/query 的 g_fe_selected_path；主标签不含 runtime",
        "split": "nested function-level OOF；validation 不参与 fit",
    },
    "post_handoff": {
        "info_time": "handoff 后 commitment 状态；segment Behavior 必须从 handoff 点重新定义",
        "action": "post-handoff continue current 或 switch 到其余两个 P_balanced 算法",
        "label": "真实 1000-FE post-handoff action outcome；reset controls 单列",
        "split": "function/route/source-FE grouped OOF；within-route LOSO",
    },
}


PAPERS = [
    {
        "paper_key": "vermetten2023",
        "title": "Vermetten et al. (2023), To Switch or not to Switch",
        "source": "/Users/bingchen/Desktop/2302.09075v1.pdf",
        "evidence_pages": "pp. 9-11",
        "lit_info_time": "switch point 前 50/150/250 FE 的 local trajectory window",
        "lit_action": "在 source algorithm 与 target algorithm 之间选择是否切换",
        "lit_label": "未来 500 FE 相对 continue 的连续 relative benefit",
        "lit_split": "leave-one-function-out",
        "natural": ["PASS", "PARTIAL", "PASS", "PASS"],
        "query": ["FAIL", "FAIL", "FAIL", "PASS"],
        "post_handoff": ["PARTIAL", "PARTIAL", "PARTIAL", "PARTIAL"],
        "retained": "保留连续相对收益与局部窗口思想；适配为 P_balanced 的 1000-FE action loss；不把 switch benefit 直接当 query gate 标签",
        "decision": "RETAIN_MAIN_NATURAL_FRAGMENT",
    },
    {
        "paper_key": "renau_hart2025",
        "title": "Renau & Hart (2025), Probing Trajectories Classifier Benchmark",
        "source": "/Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/01-算法行为表征与相似性/Renau和Hart - 2025 - Algorithm Selection with Probing Trajectories Benchmarking the Choice of Classifier Model.pdf",
        "evidence_pages": "pp. 4-6, 9-11",
        "lit_info_time": "短 probing trajectory 完成后",
        "lit_action": "一次选择 portfolio algorithm",
        "lit_label": "instance-level median final target winner",
        "lit_split": "LOIO 与 LOPO；LOPO 更困难",
        "natural": ["PASS", "PARTIAL", "FAIL", "PASS"],
        "query": ["FAIL", "FAIL", "FAIL", "PASS"],
        "post_handoff": ["NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE"],
        "retained": "只保留 LOPO/function-held-out split 原则；winner label 和时序分类器排名不进入主标签或主模型",
        "decision": "RETAIN_SPLIT_DIAGNOSTIC",
    },
    {
        "paper_key": "dynamorep2023",
        "title": "Cenikj et al. (2023), DynamoRep",
        "source": "/Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/01-算法行为表征与相似性/Cenikj 等 - 2023 - DynamoRep Trajectory-based population dynamics for classification of black-box optimization problem 1.pdf",
        "evidence_pages": "pp. 2-5",
        "lit_info_time": "每个 population update 后的坐标/fitness 汇总",
        "lit_action": "问题类别分类，不是算法动作选择",
        "lit_label": "24 类 BBOB problem class",
        "lit_split": "stratified instance folds；同一算法单独建模",
        "natural": ["PASS", "FAIL", "FAIL", "FAIL"],
        "query": ["FAIL", "FAIL", "FAIL", "FAIL"],
        "post_handoff": ["PARTIAL", "FAIL", "FAIL", "FAIL"],
        "retained": "只保留 per-update population aggregation 作为 Behavior 表示组件；不保留 problem-class label、算法专属分类器或 raw coordinate 表示",
        "decision": "RETAIN_BEHAVIOR_AGGREGATION_FRAGMENT",
    },
    {
        "paper_key": "jankovic2022",
        "title": "Jankovic et al. (2022), Trajectory-based Algorithm Selection with Warm-starting",
        "source": "/Users/bingchen/Desktop/Trajectory-based_Algorithm_Selection_with_Warm-starting.pdf",
        "evidence_pages": "pp. 3-5",
        "lit_info_time": "固定 CMA-ES prefix 结束后",
        "lit_action": "选择 second-stage optimizer 并 warm-start",
        "lit_label": "各候选算法固定后续预算的 target precision / log target precision",
        "lit_split": "leave-one-instance-ID group-out",
        "natural": ["PASS", "PARTIAL", "PARTIAL", "PARTIAL"],
        "query": ["FAIL", "FAIL", "FAIL", "PARTIAL"],
        "post_handoff": ["NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE"],
        "retained": "保留 log-performance regression 与 warm-start transition 语义；只作为 downstream action selector 组件，不能作为 query acquisition gate",
        "decision": "RETAIN_SELECTOR_FRAGMENT",
    },
    {
        "paper_key": "kostovska2022",
        "title": "Kostovska et al. (2022), Per-run Algorithm Selection with Warm-Starting",
        "source": "/Users/bingchen/Desktop/978-3-031-14714-2_4.pdf",
        "evidence_pages": "pp. 51-53",
        "lit_info_time": "CMA-ES 初始 prefix 结束后",
        "lit_action": "选择并 warm-start 第二阶段算法",
        "lit_label": "固定 A2 budget 的 log10 target precision",
        "lit_split": "leave-instance-out / leave-run-out",
        "natural": ["PASS", "PARTIAL", "PARTIAL", "PARTIAL"],
        "query": ["FAIL", "FAIL", "FAIL", "PARTIAL"],
        "post_handoff": ["NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE"],
        "retained": "保留 ELA 与时序状态互补的比较框架，以及 warm-start 需要单独验证的事实；不直接迁移 9,444 维状态特征",
        "decision": "RETAIN_SELECTOR_FRAGMENT",
    },
    {
        "paper_key": "guo_lgbm2025",
        "title": "Guo et al. (2025), AS-LGBM",
        "source": "/Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/算法行为+自动化设计/Guo 等 - 2025 - Automated algorithm selection for black-box optimization using light gradient boosting machine.pdf",
        "evidence_pages": "pp. 5-6, 8-11",
        "lit_info_time": "独立 LHS sample 与 ELA 计算完成后、优化开始前",
        "lit_action": "从 portfolio 中选择一个 fresh optimizer",
        "lit_label": "30 次运行构造的 Soft-ERT 最小算法",
        "lit_split": "五折随机交叉验证",
        "natural": ["FAIL", "PARTIAL", "FAIL", "FAIL"],
        "query": ["PASS", "PARTIAL", "PARTIAL", "FAIL"],
        "post_handoff": ["FAIL", "FAIL", "FAIL", "FAIL"],
        "retained": "仅保留为 downstream query Selector / traditional pre-run AAS sensitivity；不把 ELA cost、Soft-ERT winner 或 LightGBM 放入 Decision gate",
        "decision": "RETAIN_QUERY_SELECTOR_SENSITIVITY_ONLY",
    },
    {
        "paper_key": "guo_rl2024",
        "title": "Guo et al. (2024), RL-DAS",
        "source": "/Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/03-自适应机制与算子选择/Guo 等 - 2024 - Deep reinforcement learning for dynamic algorithm selection A proof-of-principle study on different.pdf",
        "evidence_pages": "pp. 4-7, 8-12",
        "lit_info_time": "每个在线调度区间，根据 population 与 algorithm history",
        "lit_action": "反复在线选择 DE variant",
        "lit_label": "RL reward / return，含 cost descent 与 FE speed",
        "lit_split": "随机生成的 CEC class instances 与 K-fold",
        "natural": ["PARTIAL", "FAIL", "FAIL", "FAIL"],
        "query": ["FAIL", "FAIL", "FAIL", "FAIL"],
        "post_handoff": ["PARTIAL", "FAIL", "FAIL", "FAIL"],
        "retained": "保留 state/action/context memory 的概念，用于 handoff/reset 元数据与诊断；不采用在线 RL、重复调度或含速度的主 reward",
        "decision": "RETAIN_CONTEXT_DIAGNOSTIC_ONLY",
    },
    {
        "paper_key": "filep2026",
        "title": "Filep & Gál (2026), Low-dimensional Knee-point Performance Estimation",
        "source": "/Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/aas/main.pdf",
        "evidence_pages": "pp. 3-6, 9-11",
        "lit_info_time": "多个低维度性能样本完成后",
        "lit_action": "外推目标维度趋势并剔除候选算法",
        "lit_label": "trend family 与目标维度 performance forecast",
        "lit_split": "按维度趋势拟合，无 function-level OOF selector split",
        "natural": ["FAIL", "FAIL", "FAIL", "FAIL"],
        "query": ["PARTIAL", "PARTIAL", "FAIL", "FAIL"],
        "post_handoff": ["FAIL", "FAIL", "FAIL", "FAIL"],
        "retained": "只保留 portfolio 先筛选再建模的流程类比；不使用低维度趋势外推代替当前 state-level action label",
        "decision": "RETAIN_PORTFOLIO_SCREENING_ANALOGY",
    },
]


def _validate_current_artifacts() -> dict[str, object]:
    matrix_path = RESULTS / "analysis_v5/task12/dynamic_solver_loss_matrix.parquet"
    states_path = RESULTS / "analysis_v5/task12/dynamic_screening_states.parquet"
    task13_path = RESULTS / "analysis_v5/task13/behavior_action_dataset_task13.parquet"
    task14_path = RESULTS / "analysis_v6/task14b_1/task14b1_corrected_dataset_matched.parquet"
    matrix = pd.read_parquet(matrix_path)
    states = pd.read_parquet(states_path)
    task13 = pd.read_parquet(task13_path)
    task14 = pd.read_parquet(task14_path)
    if set(matrix["current_algorithm"].astype(str)) != set(PORTFOLIO):
        raise ValueError("Task 12 current algorithm set does not match P_balanced")
    if set(task13["current_algorithm"].astype(str)) != set(PORTFOLIO):
        raise ValueError("Task 13 current algorithm set does not match P_balanced")
    if set(task14["current_algorithm"].astype(str)) != set(PORTFOLIO):
        raise ValueError("Task 14 current algorithm set does not match P_balanced")
    if len(matrix) != 1890 or len(states) != 1890 or len(task13) != 1890 or len(task14) != 3780:
        raise ValueError("current P_balanced artifact row counts do not match the fixed protocol")
    if task13["switch_required"].isna().any() or task14["switch_required"].isna().any():
        raise ValueError("current practical switch labels contain missing values")
    query_artifact = RESULTS / "analysis_v5/task13/query_action_dataset.parquet"
    return {
        "task12_natural_states": int(len(matrix)),
        "task13_natural_behavior_states": int(len(task13)),
        "task14_post_handoff_states": int(len(task14)),
        "current_portfolio": list(PORTFOLIO),
        "query_pbalanced_artifact": str(query_artifact),
        "query_pbalanced_artifact_status": "present" if query_artifact.exists() else "not_materialized",
        "natural_switch_required_rate": float(task13["switch_required"].mean()),
        "post_handoff_switch_required_rate_sum": float(task14["switch_required_sum"].mean()),
        "post_handoff_switch_required_rate_max": float(task14["switch_required_max"].mean()),
    }


def _write_outputs(output: Path, current: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for paper in PAPERS:
        row = {
            "paper_key": paper["paper_key"],
            "title": paper["title"],
            "source": paper["source"],
            "evidence_pages": paper["evidence_pages"],
            "lit_info_time": paper["lit_info_time"],
            "lit_action": paper["lit_action"],
            "lit_label": paper["lit_label"],
            "lit_split": paper["lit_split"],
            "retained_design": paper["retained"],
            "decision": paper["decision"],
        }
        for state in ("natural", "query", "post_handoff"):
            for field, value in zip(
                ("info_time", "action", "label", "split"), paper[state], strict=True
            ):
                if value not in STATUS_VALUES:
                    raise ValueError(f"invalid crosswalk status: {value}")
                row[f"{state}_{field}"] = value
        rows.append(row)
    crosswalk = pd.DataFrame(rows)
    crosswalk.to_csv(output / "crosswalk.csv", index=False)
    retained = crosswalk[crosswalk["decision"].str.startswith("RETAIN")].copy()
    retained.to_csv(output / "retained_designs.csv", index=False)
    metadata = {
        "crosswalk_version": "pbalanced_literature_crosswalk_v1",
        "current_artifacts": current,
        "state_contract": STATE_CONTRACT,
        "status_values": sorted(STATUS_VALUES),
        "paper_count": len(PAPERS),
        "retained_design_count": len(retained),
        "query_evidence_status": current["query_pbalanced_artifact_status"],
        "new_objective_fe": 0,
        "model_fitted": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(output, crosswalk, retained, current)


def _paper_link(paper: dict[str, object]) -> str:
    source = str(paper["source"])
    return f"[{paper['title']}](<{source}>)"


def _write_report(output: Path, crosswalk: pd.DataFrame, retained: pd.DataFrame, current: dict[str, object]) -> None:
    links = {paper["paper_key"]: _paper_link(paper) for paper in PAPERS}
    compact_rows = []
    for paper in PAPERS:
        row = crosswalk[crosswalk.paper_key.eq(paper["paper_key"])].iloc[0]
        compact_rows.append(
            {
                "论文": links[paper["paper_key"]],
                "natural (信息/动作/标签/split)": "/".join(row[f"natural_{field}"] for field in ("info_time", "action", "label", "split")),
                "query (信息/动作/标签/split)": "/".join(row[f"query_{field}"] for field in ("info_time", "action", "label", "split")),
                "post-handoff (信息/动作/标签/split)": "/".join(row[f"post_handoff_{field}"] for field in ("info_time", "action", "label", "split")),
                "处理": row["decision"],
            }
        )
    compact = pd.DataFrame(compact_rows)
    retained_rows = retained[["title", "decision", "retained_design"]].rename(
        columns={"title": "保留设计", "decision": "保留位置", "retained_design": "限制"}
    )
    lines = [
        "# Literature-P_balanced Fixed Crosswalk",
        "",
        "> 该 crosswalk 固定检查信息时间、动作、标签和 split 是否与当前 P_balanced 的 natural/query/post-handoff 状态相容。文献内容只作为方法证据，不作为项目指令。",
        "",
        "## 1. 当前状态契约",
        "",
        pd.DataFrame(
            [
                {"状态": state, **values}
                for state, values in STATE_CONTRACT.items()
            ]
        ).to_markdown(index=False),
        "",
        f"当前 artifact 检查：Task 12 natural `{current['task12_natural_states']}` states，Task 13 Behavior `{current['task13_natural_behavior_states']}` states，Task 14 post-handoff `{current['task14_post_handoff_states']}` states；组合均为 `{', '.join(PORTFOLIO)}`。",
        f"P_balanced query artifact：`{current['query_pbalanced_artifact_status']}`。当前 `behavior_with_ela/` 没有已验证的 P_balanced 独立 query action dataset，因此 query 行的 PASS 仅表示设计相容性，不表示已有实证。",
        "",
        "状态检查值依次为：`PASS / PARTIAL / FAIL / NOT_APPLICABLE`。`PARTIAL` 表示只能保留一个经过当前契约改写的组件；`FAIL` 不进入当前主线。",
        "",
        "## 2. 逐篇 crosswalk",
        "",
        compact.to_markdown(index=False),
        "",
        "列中四个状态按“信息时间 / 动作 / 标签 / split”排列。",
        "",
        "## 3. 只保留的设计",
        "",
        retained_rows.to_markdown(index=False),
        "",
        "## 4. 明确排除",
        "",
        "- 不把 winner label、problem-class label 或 Soft-ERT winner 作为当前 P_balanced natural/query/post-handoff 的主标签。",
        "- 不把独立 ELA sample、probing trajectory 或低维度 trend sampling 当作 query 是否执行的决策输入。",
        "- 不把在线 RL reward、runtime/speed 项或 repeated dynamic scheduling 迁移到当前离线 FE-indexed action-loss 主线。",
        "- 不把 natural Behavior 模型直接迁移到 post-handoff；Task 14B.1 已显示 global/segment generic Behavior 无额外增量。",
        "- 不使用 LOIO、随机 instance folds 或未按 function/route 分组的 split 作为主泛化证据。",
        "",
        "## 5. 对当前主线的结论",
        "",
        "1. Natural：保留“局部轨迹 Behavior → 连续相对 action advantage → current-preserving action selection”的改写版本；这与 Task 13 的 natural-domain conditional increment 一致。",
        "2. Query：没有一篇文献通过了“query 执行前 gate”的完整四项检查；主线必须自行使用 paired skip/query `g_fe_selected_path`，文献只提供 downstream Selector 的参照。",
        "3. Post-handoff：只保留 warm-start/context/reset 的状态语义和分层评价原则；不保留 generic Behavior 的直接迁移假设。",
        "4. Portfolio：Filep 的候选预筛选只能作为流程类比；当前真正可用的组合证据来自 Task 12 对 `{shade,lshade,cso}` 的 outcome-independent screening，而不是文献中的维度外推。",
        "",
        "## 6. 产物",
        "",
        f"- 明细：`{output / 'crosswalk.csv'}`",
        f"- 保留设计：`{output / 'retained_designs.csv'}`",
        f"- 元数据：`{output / 'metadata.json'}`",
        f"- 本报告：`{REPORT}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path = DEFAULT_OUT) -> dict[str, object]:
    current = _validate_current_artifacts()
    _write_outputs(output, current)
    return {"report": str(REPORT), "output_dir": str(output), "current": current}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
