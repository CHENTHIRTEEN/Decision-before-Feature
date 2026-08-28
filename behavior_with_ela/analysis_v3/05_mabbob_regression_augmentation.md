# 05 · Task 9E：selected MA-BBOB 增广在回归载体上的重新验证

- 日期：2026-08-29
- 问题：Task 7 证明的是三分类载体上的增广价值；正式主载体已是 `behavior_action_loss_regression_v2`，必须在回归载体上重新验证 MA-BBOB 增广是否仍有价值。
- 设计：
  - **R-B**：仅 BBOB train 行，28 维 Behavior，与 v2 完全相同的多输出 `RandomForestRegressor(200, depth 8, sqrt)`、相同 family-OOF 折、相同阈值协议、相同 practical delta（1.4639）、同一 validation。
  - **R-BM**：BBOB + selected MA-BBOB（即当前 v2，数字从 `analysis_v2/task1` 已定稿产物复述，阈值 0.1451；R-B 的 OOF 阈值为 0.1752）。
  - 附加：MA-BBOB definition 分组 5 折 OOF 回归诊断（折分配沿用 `analysis_v2/task7/definition_fold_assignment.json`，保证与旧诊断可比）。
- 无新 objective evaluation。产物：`analysis_v3/task5/`（策略汇总、分数汇总、逐候选回归质量、definition OOF 表），run 级在 `results/analysis_v3/task5/`。

## 1. 策略级对比（函数平衡）

| 模型 | split | gain | terminal loss | 归一化 regret | switch rate | G<0 | G<-d50 | G<-d95 | success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R-B（BBOB only） | train OOF | 1.1074 | -4.3048 | 0.2911 | 0.676 | 0.2648 | 0.2347 | 0.0502 | 0.306 |
| **R-BM（BBOB+MA，v2）** | train OOF | **1.3141** | -4.5132 | **0.2564** | 0.667 | **0.2306** | **0.2065** | **0.0463** | 0.323 |
| R-B | validation | 1.8504 | -4.5731 | 0.1636 | 0.667 | 0.1028 | 0.0694 | 0.0000 | 0.337 |
| **R-BM** | validation | 1.8504 | -4.5731 | 0.1636 | 0.667 | 0.1028 | 0.0694 | 0.0000 | 0.337 |

validation 上两模型的 540 个 run 的决策与端点**完全一致**（gain / regret / harmful / success 全部相同），尽管阈值不同（0.1752 vs 0.1451）。

## 2. 分数级对比（train family OOF）

| 模型 | pairwise 排序正确率 | top-1 动作命中率 | 逐候选增益 Spearman（pso / shade / cmaes） |
|---|---:|---:|---|
| R-B | 0.7193 | 0.5105 | 见 `rb_per_candidate_regression_quality_oof.csv` |
| **R-BM** | **0.7607** | **0.6453** | +0.32 / +0.49 / -0.26（Task 1 复述） |

## 3. 与 SBS 的关系（validation 配对）

| 模型 | all-prefix 配对增益 vs SBS | cmaes-start 配对增益 vs SBS |
|---|---:|---:|
| R-B | -0.4766 | 0.0000 |
| R-BM | -0.4766 | 0.0000 |

两模型相对 SBS 的位置完全相同（参见 03 报告）：增广不改变 deployability 结论。

## 4. MA-BBOB definition 分组 OOF（回归诊断，分数级）

| fold | loss MAE | 增益 Spearman |
|---:|---:|---:|
| 0 | 2.465 | 0.855 |
| 1 | 2.157 | 0.864 |
| 2 | 2.174 | 0.794 |
| 3 | 3.036 | **0.285** |
| 4 | 3.167 | 0.841 |

5 折中 4 折增益 Spearman > 0.79，fold 3（定义子集）明显偏弱（0.285）——MA 域内可学习性总体良好但存在对定义子集的敏感性，与 Task 7 在分类载体上观察到的"fold 3 最弱"一致。

## 5. 结论（按预指定措辞）

**MA augmentation helps regression**：train family OOF 上策略增益 +0.207（1.107→1.314）、pairwise 排序 +0.041、top-1 命中 +0.135，三类有害切换率同时下降；held-out BBOB validation 上与 R-B 决策完全一致（**不低于** R-B，无负贡献）。增广的价值与 Task 7 的分类载体结论同向，且现在有了回归载体上的直接证据。按工作单要求，不能使用 validation 反向调参；此处 validation 仅作固定程序评价。
