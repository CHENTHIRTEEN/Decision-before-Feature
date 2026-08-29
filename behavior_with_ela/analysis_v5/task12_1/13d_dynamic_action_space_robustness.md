# 13d · Dynamic Action Space 稳健性重分析与 DCM 复核（Task 12.1H/I）

- 日期：2026-08-29
- 数据：同 13c（set-valued、current-preserving，主判据 = pairwise conservative δ）。
- 产物：`switch_required_distribution.parquet`、`current_preserving_transition_matrix.parquet`、`within_trajectory_switch_variation.parquet`、`dynamic_dcm_robustness.parquet`、行级 `results/analysis_v5/task12_1/{practical_action_sets, switch_required_strata}.parquet`。

## 1. Current-preserving 转移结构（主语义）

| suite | current | P(continue) | →shade | →lshade | →cso |
|---|---|---:|---:|---:|---:|
| BBOB | shade | 0.742 | — | 0.655\* | 0.345\* |
| BBOB | lshade | 0.747 | 0.360\* | — | 0.640\* |
| BBOB | cso | 0.738 | 0.398\* | 0.602\* | — |
| MA | shade | 0.750 | — | 0.400\* | 0.600\* |
| MA | lshade | 0.706 | 0.302\* | — | 0.698\* |
| MA | cso | 0.750 | 0.356\* | 0.644\* | — |

\* 为 switch-required 条件下的 target 份额（joint 份额见 parquet）。**切换结构保持方向性**：cso→{lshade,shade}（早期专家交棒）、lshade→cso（MA）/shade（BBOB）为主方向，与 Task 12 的定性结论一致，但总体切换频次从旧口径的"practical best≠current ≈40–78%"收缩到 **26%**。

## 2. 分层 switch-required rate（主语义）

| 分层 | BBOB | MA |
|---|---:|---:|
| FE=2000 | 0.349 | 0.433 |
| FE=4000 | 0.224 | 0.194 |
| FE=6000 | 0.200 | 0.167 |
| current=shade / lshade / cso | 0.258 / 0.253 / 0.262 | 0.250 / 0.294 / 0.250 |
| family 最低–最高 | 0.196–0.307 | 0.265（单 family） |

切换需求集中在**早期（FE=2000）**，current 之间几乎均匀——"何时切"比"谁在跑时切"更具结构。

## 3. 熵（H1）：从单标签熵改为 switch-status 熵

| 量（bits） | BBOB | MA |
|---|---:|---:|
| $H(A_{op})$ operational（continue/2 targets，max 1.585） | 1.224 | 1.239 |
| $H(Z)$ | 0.823 | 0.834 |
| $H(Z\mid current)$ | 0.823 | 0.832 |
| $H(Z\mid current,FE)$ | 0.798 | 0.761 |
| $H(Z\mid problem,current,FE)$ | **0.483** | **0.439** |

Task 12 的表述"$H(A^\star\mid problem,FE)=0.97$，故 76% 变异是 state 级"就此退役：condition on current 之后，正确的问题是 $H(Z\mid problem,current,FE)$——仍有 0.44–0.48 bits 的 switch-status 不确定性无法由完整 simple context 解释；但其**价值**含义以 13b 的 $\Delta_{context-residual}$（≈0.05，且与 winner's-curse 不可区分）为准，熵与价值两个口径都已如实收窄。

## 4. 轨迹内变异（H2）

- switch-status 在 FE=2000/4000/6000 间翻转的轨迹占比：**BBOB 0.398 / MA 0.461**；
- switch-required 轨迹内 target 随 FE 变化占比：0.438 / 0.475。

切换需求与 target 都随阶段轮替（结构性），而非固定标签。

## 5. DCM 对 noise semantics 的稳健性（I）

| suite | pair | legacy DCM | pairwise DCM（主） | pooled DCM | 双向性（pairwise 两方向 P） |
|---|---|---:|---:|---:|---|
| BBOB | shade↔lshade | 0.297 | 0.304 | 0.294 | 0.196 / 0.267 ✓ |
| BBOB | shade↔cso | 0.333 | 0.333 | 0.353 | 0.399 / 0.167 ✓ |
| BBOB | lshade↔cso | 0.366 | 0.370 | 0.378 | 0.424 / 0.130 ✓ |
| MA | shade↔lshade | 0.259 | 0.254 | 0.254 | 0.246 / 0.259 ✓ |
| MA | shade↔cso | 0.298 | 0.309 | 0.313 | 0.441 / 0.191 ✓ |
| MA | lshade↔cso | 0.307 | 0.326 | 0.330 | 0.426 / 0.174 ✓ |

**Task 12 "所有 pair 双向互补"的结论对 noise semantics 稳健**：三种 δ 语义下 DCM 变化 ≤0.02，无任何 pair 退化为单边支配；最小方向概率 0.130（BBOB lshade↔cso）仍远离 0。

## 6. 结论

在取消 tie 偏置、使用 pair-specific 保守 δ、current-preserving 语义之后：

1. 动作空间**非退化**（switch-required ≈26%、$H(A_{op})$≈1.22–1.24 bits、无 $A_{ND}$ 空集、无近全支配）；
2. 切换结构方向明确且集中在早期；
3. 三对互补性对噪声语义稳健；
4. 所有数值不再依赖 `tied[0]`（三语义一致性互证）。

→ 支撑总报告 verdict 的条件 1/2/3/7；残余风险只在 oracle 乐观偏差（13b §3）。
