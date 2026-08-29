# 10 · Three-Layer Oracle Headroom（Task 12L）

- 日期：2026-08-30
- 定义（solver 语义损失矩阵，1000-FE horizon，log10 损失）：$L_{SBS}$=单算法策略（fb 最优单一 solver）；$L_{problem\text{-}static}$=每 problem 固定最优 solver；$L_{problem+FE}$=每 (problem, FE) 固定最优 solver；$L_{statewise}$=逐状态最优。三层 headroom：$\Delta_{portfolio}=L_{SBS}-L_{statewise}$、$\Delta_{problem}=L_{problem\text{-}static}-L_{statewise}$、$\Delta_{dynamic}=L_{problem+FE}-L_{statewise}$（**Behavior-relevant headroom**）。函数级 bootstrap 95% CI（2000 次）。
- 产物：`analysis_v5/task12/{dynamic_oracle_headroom,oracle_headroom_bootstrap}.parquet`。

## 1. 主表

| suite | SBS | $L_{SBS}$ | $L_{prob}$ | $L_{prob{+}FE}$ | $L_{statewise}$ | $\Delta_{portfolio}$ | $\Delta_{problem}$ | $\Delta_{dynamic}$ | δ95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BBOB | lshade | −1.550 | −1.608 | −1.645 | −1.759 | **+0.209** [0.126, 0.306] | +0.152 [0.100, 0.206] | **+0.115 [0.074, 0.163]** | 0.088 |
| MA | lshade | −4.496 | −4.531 | −4.605 | −4.690 | **+0.195** [0.122, 0.280] | +0.158 [0.102, 0.224] | **+0.085 [0.056, 0.119]** | 0.084 |

## 2. 判读

1. **$\Delta_{portfolio}=+0.21/+0.19$**：组合的 VBS–SBS headroom 明显存在且 CI 远离 0——与 Task 11（{pso,shade,cmaes} 上 0.013）形成数量级对比。
2. **problem-static oracle 已吃掉大部分 headroom**（Δ_problem=0.15/0.16）：多数价值在"为每个问题选对 solver"——这是 **static/contextual AS** 的价值。
3. **但 $\Delta_{dynamic}=+0.115/+0.085$ 仍然非零**：BBOB 上 CI [0.074, 0.163] 整体高于噪声 δ95=0.088 的点值（CI 下界略低于 δ95，点估计高 31%）；MA 上 CI [0.056, 0.119] 显著为正但点估计 ≈ δ95（0.084）——**MA 处于噪声边界**，必须如实标注。
4. 结合 08 报告的条件熵（problem+FE 后仍剩 0.97 bits）与 permutation null（观测切换低于 tie 噪声水平）：**dynamic 价值是结构性的，但幅度为中等（≈0.1 log10），且在 MA 上贴近噪声边缘**。

## 3. 对"是否 Behavior-driven"的初步定位

$\Delta_{dynamic}>0$ 说明存在 problem+FE 之外的状态依赖选择价值——这正是 Behavior 特征（含 segment Behavior）理论上可以捕获的部分；但注意 $L_{problem+FE}$ oracle 的动作只需要 (problem, FE) 两个离散变量即可复现，**Behavior 增量测试必须证明 Behavior 能比 (problem, current, FE) 离散基线更好地预测这 0.09–0.12 的状态级残差**（工作单 §42 的门）。
