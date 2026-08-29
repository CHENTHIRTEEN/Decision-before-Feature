# 14b · Pairwise Delta 零 FE 敏感性（Task 13B）：max / quadrature / sum

- 日期：2026-08-29；零新增 objective evaluations。
- 动机：Task 12.1 的 fallback $\delta_{ij}^{cons}=\max(\delta_i,\delta_j)$ 未必是统计意义上的保守 pairwise bound。按工作单预注册补充两个更保守的组合规则：
$$
\text{quad: }\sqrt{\delta_i^2+\delta_j^2},\qquad \text{sum: }\delta_i+\delta_j,
$$
其中 cell 级 $\delta_{95}$ 仍取 Task 12 (suite × action) 函数平衡标定（state-aware：cell 的 raw action 为 continue 或 switch）。
- 预注册判定：**STABLE** 若 sum 下 $P(\text{switch-required})\ge0.10$ 且三 pair 双向（双向 := 两方向概率均 ≥0.05）；**MODERATE** 若 quad 非退化但 sum 接近退化；**FRAGILE** 若 quad 已近退化（<0.05）→ STOP，不得进入 Behavior 训练。
- 产物：`delta_sensitivity_max_quad_sum.parquet`、`delta_sensitivity_variation.parquet`、`delta_sensitivity_verdict.parquet`、行级 `results/analysis_v5/task13/practical_action_sets_{max,quadrature,sum}.parquet`。

## 1. 三种规则下的动作空间

| rule | suite | switch-required | $P(c\in A_{ND})$ | $P(|A_{ND}|{>}1)$ | $P(A_{ND}{=}\varnothing)$ | $H(A_{op})$ | optional-switch |
|---|---|---:|---:|---:|---:|---:|---:|
| max（12.1 主） | BBOB | 0.2578 | 0.7422 | 0.5319 | 0 | 1.2239 | 0.4644 |
| max | MA | 0.2648 | 0.7352 | 0.4870 | 0 | 1.2391 | 0.4296 |
| quadrature | BBOB | 0.2259 | 0.7741 | 0.5800 | 0 | 1.1211 | 0.5178 |
| quadrature | MA | 0.2519 | 0.7481 | 0.5333 | 0 | 1.2000 | 0.4685 |
| sum（最保守） | BBOB | **0.1867** | 0.8133 | 0.6393 | 0 | 0.9817 | 0.5844 |
| sum | MA | **0.2111** | 0.7889 | 0.5944 | 0 | 1.0619 | 0.5407 |

## 2. Pairwise DCM 与双向性

| rule | suite | shade↔lshade | shade↔cso | lshade↔cso | 最小方向概率 |
|---|---|---:|---:|---:|---:|
| max | BBOB | 0.304 | 0.333 | 0.370 | 0.130 |
| max | MA | 0.254 | 0.309 | 0.326 | 0.174 |
| quadrature | BBOB | 0.324 | 0.347 | 0.377 | 0.123 |
| quadrature | MA | 0.274 | 0.317 | 0.335 | 0.165 |
| sum | BBOB | 0.357 | 0.367 | 0.394 | 0.106 |
| sum | MA | 0.315 | 0.333 | 0.359 | 0.141 |

δ 变保守后 tie 份额上升、DCM 值反而略升（互补质量未被单边支配替代）；**全部 6 个 (suite×pair) 组合在任何规则下都保持双向**。

## 3. 轨迹内变异（sum 口径）

switch-status 随 FE 翻转的轨迹占比：BBOB 0.327 / MA 0.394；switch-required 轨迹内 target 变化：0.153 / 0.152——结构变异在最强保守规则下仍非平凡。

## 4. 判定

$$
\boxed{\text{Task 13B verdict：STABLE（sum 下 switch-required 0.187/0.211}\ge0.10\text{，双向性保持）}}
$$

连续性验证：max 规则的 set 构造与 Task 12.1 `practical_action_sets.parquet` 的 switch_required 逐 state 一致（断言通过）。按工作单 §6：**允许进入 Behavior State Reconstruction Replay**；本轮以 max 规则为 practical 主语义、sum 规则为保守敏感性。
