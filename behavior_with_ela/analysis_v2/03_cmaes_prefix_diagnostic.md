# Task 3：CMA-ES 候选弱点诊断与 prefix-aware 消融

- 日期：2026-08-28
- 现象：三分类模型 CMA-ES 候选 improve AP ≈ 0.251，低于其 improve prevalence ≈ 0.291。
- 方法：按 prefix × candidate 拆解 Model U（Behavior-only，Phase 1 OOF）；新增 Model P（Behavior 28 维 + prefix one-hot 3 维，31 特征）做诊断性消融。无新 objective 运行。

## 1. CMA-ES 弱点的来源：prefix 不对称

Model U 的 OOF（按 prefix 拆解 CMA-ES 候选行）：

| prefix → candidate | 行数 | improve 基线率 | improve AP | mean true gain |
|---|---:|---:|---:|---:|
| PSO → CMAES | 8,894 | 0.2861 | **0.3401** | +1.5430 |
| SHADE → CMAES | 8,370 | 0.2963 | **0.2092** | +1.1778 |

**总体 CMA-ES AP 低于基线率完全由 SHADE→CMAES 一对驱动**；PSO→CMAES 的 AP（0.340）实际上高于其基线率。附带发现：CMA-ES→pso 与 CMA-ES→shade 的 improve 基线率接近 0（0.000/0.007），提前离开 CMA-ES 几乎总是有害的。

## 2. prefix-aware 消融（Model U vs Model P）

| 指标 | U（28 维） | P（31 维） |
|---|---:|---:|
| train OOF gain | 0.8467 | **0.9942** |
| train OOF 归一化 regret | 0.4023 | 0.2920 |
| train OOF AP | 0.2642 | 0.2711 |
| train OOF balanced accuracy | 0.5683 | 0.5657 |
| **validation gain** | 1.4481 | 1.4493 |
| **validation 归一化 regret** | 0.2561 | 0.2360 |

Model P 的明细表见 `prefix_candidate_diagnostic_p.csv`；CMA-ES 按 FE 阶段（early/mid/late）× prefix 的分解见 `cmaes_phase_diagnostic.csv`。

## 3. 解读

1. 加入 prefix 身份使 train OOF gain 提升约 +0.15，但 **validation gain 几乎不变（1.449 vs 1.448）**——prefix 带来的增量主要是对训练族内切换模式（尤其 SHADE→CMAES 的难例）的记忆，不迁移到 held-out 函数。
2. SHADE→CMAES 是共同难点：两个模型在该对上都低于基线率（U 0.209、P 0.212）。诊断性结论：**行为表示可以近似 algorithm-agnostic，但"当前求解器上下文"信息在训练族内有价值、在跨函数时不稳定**。
3. 按工作单要求，prefix-aware 模型仅作为 diagnostic ablation 记录，不替换主模型（其验证集增益未证明迁移价值，且增加 3 个与算法池耦合的输入维度）。

产物：`analysis_v2/task3/`（model_summary.csv、prefix_candidate_diagnostic_{p,u}.csv、cmaes_phase_diagnostic.csv）、`results/analysis_v2/task3/`（Model P OOF 与验证集 run 明细）。
