# 01 · Task 9A：Per-prefix deployability 分析（v2 策略按起始算法拆解）

- 日期：2026-08-29
- 问题：`behavior_action_loss_regression_v2`（阈值 0.145087，practical delta 1.4639）的整体策略行为，在 PSO-start / SHADE-start / CMAES-start 三个 prefix 下到底发生了什么；整体 switch rate = 2/3 是否几乎完全由 prefix 身份决定。
- 方法：完全复用已有产物 `results/analysis_v2/task1/{train,validation}_first_trigger_runs.parquet`（train grouped-family OOF 与未触碰的 BBOB validation），不执行任何 objective evaluation。有害切换按三套口径统计：`G < 0`、`G < -delta_50`、`G < -delta_95`；`delta_50 = 0.0564`、`delta_95 = 1.4639` 由 action repetition 数据按函数平衡分位数重算（`results/analysis_v3/noise_deltas.json`）。
- 产物：`analysis_v3/task1/{per_prefix_policy_summary,per_prefix_target_distribution,per_prefix_harmful_rates}.parquet` + `summary.json`。

## 1. 按 prefix 的策略行为（函数平衡）

| split | prefix | runs | switch rate | 选中 target | 平均选中增益 | median 增益 | terminal log10 loss | 归一化 regret | G<0 | G<-d50 | G<-d95 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| train OOF | pso | 540 | **1.000** | cmaes(538)/shade(2) | +2.4611 | +0.3680 | -3.4310 | 0.2173 | 0.2056 | 0.1759 | 0.0000 |
| train OOF | shade | 540 | **1.000** | cmaes(540) | +1.4810 | +0.2760 | -4.9710 | 0.4234 | 0.2556 | 0.2370 | 0.0926 |
| train OOF | cmaes | 540 | **0.000** | —（全部续跑） | 0.0000 | 0.0000 | -5.1375 | 0.1285 | 0 | 0 | 0 |
| validation | pso | 180 | **1.000** | cmaes(180) | +4.2286 | +4.2396 | -4.1808 | 0.1197 | 0.0889 | 0.0667 | 0.0000 |
| validation | shade | 180 | **1.000** | cmaes(180) | +1.3226 | +0.7911 | -4.4888 | 0.2992 | 0.1167 | 0.0722 | 0.0000 |
| validation | cmaes | 180 | **0.000** | —（全部续跑） | 0.0000 | 0.0000 | -5.0497 | 0.0720 | 0 | 0 | 0 |

选中 FE 分布：两个 split、全部发生切换的 prefix 上，p10 = p50 = p90 = **2000（0.2B）**，即首个决策机会。

## 2. 对工作单问题 A/B/C 的回答

**A. switch rate = 2/3 是否几乎完全由 prefix 身份构成？是，精确成立。**
两个 split 上均为：pso-start → 100% 切换，shade-start → 100% 切换，cmaes-start → 0% 切换。整体 2/3 恰是"2/3 的 run 以弱算法开局"。策略在行为空间等价于规则：*只要 prefix ≠ cmaes，就在 0.2B 切到 cmaes；prefix = cmaes 则永远继续*。

**B. CMAES-start 下 v2 是否找到任何有价值的切换机会？没有。**
1,080 个 cmaes-start run（train 540 + validation 540）中切换次数为 0。在训练 OOF 上模型为 cmaes-prefix 学到的候选增益从未超过阈值 0.1451（validation 上同样）。这与 Task 3 的诊断一致：CMA-ES→pso / CMA-ES→shade 的真实 improve 基线率接近 0（0.000/0.007），"提前离开 CMA-ES 几乎总是有害"被模型正确学习为"从不切换"。

**C. PSO/SHADE-start 的高增益是否主要来自 switch-to-CMAES？是。**
切换目标中 cmaes 占 1,078/1,080（train OOF，其余 2 次为 pso→shade）与 360/360（validation）。pso-start 的增益最大（validation 平均 +4.23），但这反映的是"PSO 前 20% 预算相对 CMA-ES 的机会成本"，不是状态相关信息的贡献。

## 3. 有害切换的结构（修正认知）

- harmful（G<-d50）全部集中在弱算法起始的切换上：train OOF pso 0.176 / shade 0.237；validation pso 0.067 / shade 0.072；cmaes-start 恒为 0。
- train OOF 上 shade-start 出现 G<-d95 的深度有害切换 50/540（9.3%），validation 上为 0——这是 Task 3 识别的 SHADE→CMAES 难例族内记忆的又一表现：训练族内模型对 shade→cmaes 过度自信，在 held-out 函数上出现了少量深度损失。
- 由于 cmaes-start 从不切换，**v2 的全部收益与全部风险都只在"从弱算法开局"的场景中存在**。

## 4. deployability 含义

1. 当前 v2 的策略行为不是"逐状态的动态调度"，而是一个确定性的 prefix→target 映射 + 固定 0.2B 时点。这与后缀报告 02 的 Prefix-only / Fixed-0.20 对照结果互为印证。
2. 真实部署场景（初始算法 = SBS = cmaes）下，v2 的行为与 SBS 完全重合：不产生增益，也不产生有害切换（见 03 报告的配对分解）。
3. 论文表述不应再使用"Behavior 驱动的动态切换"来描述当前策略；准确的表述是"从任意求解器轨迹出发的、以 Behavior 回归分数为实现载体的 prefix 条件切换规则"。

## 5. 复现命令

```bash
.venv/bin/python behavior_with_ela/analysis_v3/task1_per_prefix.py
```
