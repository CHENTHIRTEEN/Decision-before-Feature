# 06 · Source-History Effect 与 Transfer-History Control（Task 11L/M）

- 日期：2026-08-30
- 设计：三条 route 的 current algorithm 都是 CMAES，因此在固定 FE 下可直接比较 native / PSO→CMAES / SHADE→CMAES 的状态与动作结构。
- 产物：`analysis_v4/task11/{route_by_fe_summary,transfer_vs_native_bootstrap}.parquet`。

## 1. 同 FE 下的 route 对比（1000-FE horizon，每格 390 states）

| route | FE | current log10 gap（均值） | continue 分支损失 | escape rate (δ95) | practical best = continue 占比 |
|---|---:|---:|---:|---:|---:|
| R0 native | 3000 | −1.58 | −2.34 | 0.064 | 0.887 |
| R1 pso→cmaes | 3000 | **−0.33** | −1.04 | 0.064 | 0.846 |
| R2 shade→cmaes | 3000 | −0.68 | −1.99 | 0.049 | 0.880 |
| R0 native | 6000 | −3.75 | −4.44 | 0.021 | 0.962 |
| R1 pso→cmaes | 6000 | −2.26 | −2.83 | 0.015 | 0.956 |
| R2 shade→cmaes | 6000 | −3.82 | −4.55 | 0.013 | 0.956 |

（500/terminal horizon 同型，见 `route_by_fe_summary.parquet`。）

**读法**：source history 强烈改变**状态位置**——PSO→CMAES 在 FE=3000 的 gap（−0.33）远差于 native（−1.58），即 20% 预算的 pso 起点造成持久的机会成本；但**动作结构几乎不变**：escape rate（R1@3000 0.064 = R0@3000 0.064；更晚的 FE 上 transfer 甚至更低）与 practical-best=continue 占比（0.846–0.956）跨 route 接近。

## 2. RQ4 答案

$P(a^\star\mid current=\text{cmaes},FE)$ 对 source history 的依赖**主要不是动作偏好层的**：固定 current+FE 后，不同 source 的 best-action 分布、escape rate、practical winner 结构的差异都小于δ95 噪声尺度内的波动；真正随 source 变化的是 current gap 本身（可由 history 解释的位置差）。因此：

- "current solver identity + FE 不足以描述 repeated state"在**位置/轨迹层面成立**（segment Behavior 与 gap 明显不同），但在**下一动作选择层面基本不成立**——不存在需要 source history 才能预测的动作偏好差异（与 04 报告的退化结论一致：本来就几乎没有可预测的动作差异）。
- 严格地说：在本 portfolio 下，把 post-handoff 状态压缩成 current+FE 不会丢失动作选择信息（因为没有信息可丢）；这不应被解读为"current+FE 是充分状态表示"。

## 3. Transfer-history control（Task 11M）：post-transfer ≠ native？

同 (cv_group, FE) 配对的分支 gain 差（transfer − native，函数级 bootstrap 95% CI，2000 次重抽）：

| horizon | contrast | action | mean Δ | CI |
|---|---|---|---:|---|
| 1000 | R1−R0 | pso | +0.131 | [−0.115, +0.187] |
| 1000 | R1−R0 | shade | +0.120 | [−0.018, +0.174] |
| 1000 | R2−R0 | pso | **−0.230** | [−0.383, −0.140] |
| 1000 | R2−R0 | shade | **−0.169** | [−0.257, −0.088] |
| terminal | R1−R0 | pso | −0.154 | [−0.945, −0.121] |
| terminal | R2−R0 | shade | +0.195 | [−0.115, +0.648] |

**post-transfer 状态分布 ≠ native 状态分布成立**：SHADE→CMAES 状态在 1000 horizon 上的两个 switch 分支 gain 显著低于 native（CI 不含 0，约 −0.17～−0.23 log10）——转移刚完成后替代算法的相对机会比 native 状态更差（与 Task 10 的 transfer-restart confound 方向一致）。但该差异属于"替代动作更差"而非"continue 可被替代"，因此不产生选择价值，只再次强化 continue 主导。

## 4. 结论

- RQ4：**动作偏好层面 source history 无实质影响**（在退化前提下）；位置层面影响巨大。
- RQ-M：post-transfer ≠ native 成立（方向：transfer 后的 switch 机会更差）。
- 对下一阶段：若未来扩 portfolio 后出现非退化 action space，source/segment history（$B^{segment}$）仍值得作为特征测试；本轮数据不足以支持或否定其特征价值，因为不存在可预测的动作差异。
