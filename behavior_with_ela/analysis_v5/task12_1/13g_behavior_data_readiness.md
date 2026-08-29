# 13g · Behavior 数据就绪核查（Task 12.1Q/R）

- 日期：2026-08-29
- 性质：本轮不训练 Behavior；只确认后续首次 Behavior test 能否复用现有 states。

## 1. 现有 states 的 Behavior 字段状态（Q1）

- `dynamic_screening_states.parquet` **不含**任何 `bg_*` / `bs_*` 列（13a #6/#7）；
- Stage-1 仅记录每 1000-FE 的 log-gap mark，**无法**从现有 artifacts 恢复 optimizer 运行历史 → Behavior 特征不可离线重建；
- 本轮 $\ell_t$ 恢复**未**触发 replay（Stage-1 marks 逐位覆盖），故 13e 的 progress 审计也未产生 Behavior。**当前不存在任何已记录的 $B_t^{global}$。**

## 2. Segment Behavior 不可伪造（Q2）

Stage-2 states 全部为 natural trajectories（segment_start=0，13a #1/#2），因此 $B_t^{segment}\approx B_t^{global}$ 是构造使然。本轮与后续测试均不得：

- 把 global Behavior 复制一份命名为 segment Behavior；
- 声称比较过 bg vs bs；
- 声称 segment Behavior 有任何增量。

$$
\boxed{\text{True segment-Behavior value remains untested until post-handoff states exist.}}
$$

## 3. 后续首次 Behavior test 的合法特征组（Q3）

| 组 | 特征 | 现在是否可运行 |
|---|---|---|
| M0 | `current + FE`（simple baseline） | **是**——本轮已给出其 grouped-OOF 参考值 $L_{current+FE}^{OOF}$=−1.5856/−4.5298（13b） |
| M1 | $B^{global}$（bg_28） | 否——需先做状态重建 |
| M2 | `[current, FE, B^{global}]` | 否——同上 |

判定基线必须是 M0 的 **OOF** 版本（部署口径），而非任何 descriptive/oracle 量；正式目标为

$$\text{Performance}(current+FE+B)>\text{Performance}(current+FE).$$

## 4. 状态重建 replay 的前提与账目（R）

后续若 GO，Behavior test 需先执行一次确定性自然 replay（3 solvers × 42 problems × 5 seeds × 6000 FE ≈ **3.78M FE 上限**，属 state-reconstruction FE，须在资源账本单列，不得与 action-label FE 混算），并：

1. 启用当前正式 recorder 记录各 checkpoint 的 $B_t^{global}$（同一跑内完成，避免二次重建）；
2. 保存 `replay_checkpoint_state.parquet` / `replay_alignment.parquet` / `behavior_global_replayed.parquet`；
3. 执行对齐审计：suite/problem/instance/seed/current/FE/current best/确定性 RNG/population size/L-SHADE 实际种群 一一对应。

**对齐可行性已被本轮证明**：Stage-1 与 Stage-2 的自然轨迹在 terminal（FE=10000）与 continue（FE=t+1000）两条恒等式上 1890/1890 逐位相等（max diff = 0.0，`replay_alignment.parquet`）——即同初始化语义的 replay 必然精确复现原 state。若未来 replay 出现任何非零失配，按预注册规则 STOP，不得拼接。

## 5. 结论

- Task 12 states 目前**只能**支持 M0 基线（已就绪）；
- M1/M2 需要一次带正式 recorder 的确定性状态重建（下一轮的先行步骤，账目单列）；
- 真正 segment Behavior 留给 sequential post-handoff 实验，本轮数据不支持任何相关声明。
