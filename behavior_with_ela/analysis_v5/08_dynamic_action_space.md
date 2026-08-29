# 08 · Dynamic Action Space（Task 12K）

- 日期：2026-08-30
- 数据：1,890 candidate-current states × 3 solvers（solver 语义损失矩阵：current=X 时取 continue，否则取 switch-to-X）。
- 产物：`analysis_v5/task12/{dynamic_action_entropy,dynamic_best_action_distribution}.parquet`、`results/analysis_v5/task12/dynamic_solver_loss_matrix.parquet`。

## 1. Dynamic best-action 分布（practical，δ95 suite 口径）

| current | stay（continue） | → shade | → lshade | → cso |
|---|---:|---:|---:|---:|
| shade | 0.733 | — | **0.173** | 0.094 |
| lshade | 0.400 | **0.430** | — | 0.170 |
| cso | 0.125 | **0.776** | 0.098 | — |

（"stay"列即 continue；表按 current 分行列出全部 practical best 份额。）

**核心观察：跨求解器切换结构真实存在且方向明确**——
- cso 轨迹在 77.6% 的状态上应交棒给 shade（早期专家 → 中后期主力）；
- lshade 轨迹在 43.0% 的状态上应交棒给 shade（种群缩减后期 shade 更稳）；
- shade 轨迹 73.3% 应续跑，但仍有 17.3% 应切 lshade。

## 2. Dynamic practical entropy（bits，最大 $\log_2 3=1.585$）

| 量 | practical | raw |
|---|---:|---:|
| $H(A^\star)$ | 1.272 | 1.532 |
| $H(A^\star\mid current)$ | 1.187 | 1.468 |
| $H(A^\star\mid problem)$ | 1.152 | 1.355 |
| $H(A^\star\mid problem,FE)$ | **0.974** | 1.163 |
| $H(A^\star\mid current,problem,FE)$ | 0.599 | 0.730 |

**与 Task 11 的对照是本质性的**：Task 11（{pso,shade,cmaes} mature states）的 $H(A^\star)\le0.45$ bits 且 problem+FE 条件化后几乎归零方向；本轮 $H(A^\star)=1.27$ bits，且**知道 problem+FE 后仍剩 0.97 bits（76% 的变异是 state 级的）**。practical best 不是由 problem 身份 + 阶段决定的。

## 3. Permutation null 敏感性（工作单 §34）

- 组内置换（打乱组内标签）：对 $H(A^\star\mid problem,FE)$ 恒等（构造使然），改用于 $P(V_{traj}=1)$ null——observed 0.596/0.578 vs null 0.70/0.75（p95 0.72/0.78）：**观测到的轨迹内切换显著低于随机 tie 噪声水平**，即切换是结构性的（随 FE 推进的 winner 轮替），不是并列噪声。
- 全局置换：observed $H(A^\star\mid problem,FE)=1.00/0.91$ vs null 1.15/1.20（p05 1.13/1.17）——组标签与动作的关联显著强于随机。

## 4. 结论

动态 action space **非退化**：高且非噪声的条件熵、明确方向的跨求解器切换结构、以及 10 报告的 Δ_dynamic>0 三者互证。当前不存在"problem ID → solver"的静态化退化模式。
