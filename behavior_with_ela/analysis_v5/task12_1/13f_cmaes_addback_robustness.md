# 13f · CMA-ES One-Step Add-Back 稳健性重分析（Task 12.1S/T）

- 日期：2026-08-29
- 数据：Task 12 隔离 add-back 分支（1890 states × switch-to-cmaes，replicate 0 + 206 重复 cell ×R=3）。零新增 objective evaluations。
- 产物：`cmaes_addback_robustness.parquet`、`cmaes_action_noise.parquet`。

## 1. 新语义下的 4 动作空间（$P_{balanced}\cup\{\text{cmaes}\}$，pairwise conservative δ）

| 量 | BBOB | MA |
|---|---:|---:|
| $L_{SBS}^{(4)}$（cmaes 为最强单项） | −1.8277 | −4.8365 |
| $L_{current+FE}^{OOF,(4)}$ | −1.8237 | −4.8273 |
| $L_{problem+current+FE}^{desc,(4)}$ | −1.9223 | −4.8927 |
| $L_{statewise}^{(4)}$ | −1.9669 | −4.9317 |
| $\Delta_{deploy-residual}^{(4)}$ | 0.1432 | 0.1044 |
| $\Delta_{context-residual}^{(4)}$ | 0.0446 | 0.0390 |
| $P(\text{cmaes}\in A_{ND})$ | 0.799 | 0.798 |
| $P(A_{ND}=\{\text{cmaes}\})$ | 0.236 | 0.354 |
| switch-required rate（4 动作） | 0.414 | 0.511 |
| switch-required rate（3 动作对照） | 0.258 | 0.265 |
| $H(A_{op}^{(4)})$（max 2.0 bits） | 1.563 | 1.586 |
| $E|A_{ND}|$ | 2.37 | 2.04 |
| cmaes $\delta_{95}$（add-back 重复标定） | 0.1626 | 0.1112 |
| residual ratio（$\Delta_{deploy}^{(4)}/\Delta_{deploy}^{(3)}$） | 0.826 | 0.651 |

## 2. Verdict（预注册规则：singleton-cmaes ≥0.85 或 ratio ≤0.2 → STRONG；≥0.55 或 ratio ≤0.6 → PARTIAL；否则 NO）

$$
\boxed{\text{BBOB: ONE-STEP ADD-BACK: NO COLLAPSE}\qquad \text{MA: ONE-STEP ADD-BACK: NO COLLAPSE}}
$$

- cmaes 进入候选集后 switch-required 从 26% 升到 41–51%（cmaes 在额外 15–25% 的 states 上严格超出 practical δ 支配 current），但 current-preserving continue 份额仍有 0.49–0.59，$\Delta_{deploy-residual}$ 保留 65–83%；
- cmaes 自身噪声最大（δ95 0.163/0.111），其支配多为小幅、接近噪声尺度；
- 无 $A_{ND}$ 空集，operational 熵上升（1.22→1.56–1.59 bits）。

## 3. 解释边界（必须在所有引用处保留）

1. Task 12 states 的 current 仅覆盖 {shade, lshade, cso}，cmaes 从未作为 current 生成过成熟重复状态——**本结论只是 one-step add-back**；
2. repeated CMAES-current collapse 与否**仍 unresolved**；若未来把 cmaes 正式纳入 sequential 组合，必须先补 CMAES-current 自然轨迹 + 重复分支，再重做同型审计；
3. 正式组合是否加入 cmaes 属于 deployment 目标下的设计决策，本轮不改 $P_{balanced}$。

## 4. Transfer / Restart 混淆对照设计（Task 12.1T：只设计，不执行）

SHADE↔L-SHADE 的切换收益混有 memory/archive reset 效应。后续 confirmatory control（在 Behavior Incremental Test 通过之后才执行）：

- SHADE-current：① native SHADE continue；② population-preserving reset-SHADE（保留种群、重置 SHADE memory/archive/RNG）；③ switch L-SHADE；
- L-SHADE-current：① native L-SHADE continue；② population-preserving reset-L-SHADE（保留缩减后种群、重置 reduction schedule）；③ switch SHADE。

三者同一 checkpoint、同一 1000-FE horizon；②−① 给出纯 reset 效应，③−② 给出净化后的算法切换效应。
