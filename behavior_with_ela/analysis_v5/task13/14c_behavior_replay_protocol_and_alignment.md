# 14c · Behavior State Reconstruction Replay 协议与逐位对齐审计（Task 13C/13D）

- 日期：2026-08-29。实现：`analysis_v5/task13/task13_replay.py`；产物：`results/analysis_v5/task13/{behavior_replay_checkpoints, behavior_replay_alignment, behavior_global_features}.parquet`、`analysis_v5/task13/replay_summary.json`。

## 1. Replay 协议（13C）

- 范围：Task 12 Stage-2 全部 natural states 的源轨迹：42 problems（BBOB 30 + selected MA 12）× seeds 1–5 × {SHADE, L-SHADE, CSO}，从 FE=0 跑到 **FE=6000**，checkpoint FE∈{2000,4000,6000}；
- **FE 账目：恰好 3,780,000 FE = 预注册上限**（state-reconstruction FE，单列账本，无任何 action-label FE）；
- 与 Task 12 的同一性：同一 `configs/behavior_with_ela_train.yaml`（10D、population_size=40、boundary=reflect、fe_total=10000）、同一 `make_experiment_problem`、同一 `OptimizerSettings`、同一 `initialize_optimizer_state` / `advance_optimizer_state` 调用模式（fe_budget=min(40, remaining) 分块）、同 seed 的 RNG 派生；问题集先经程序断言与 Stage-2 states 完全一致；
- Behavior 记录采用正式流式语义（与 `StreamingBehaviorState`/Phase-1 提取一致）：`NativeUpdateWindowRecorder` 在**每次完整原生 update** 后观察（初始种群作首个快照），checkpoint 处 `build(fe_total=10000, ...)` 生成 w02/w05/w10 窗口统计与 native-update 历史，再经 `TrajectoryRecord.from_arrays` + `extract_behavior_rows` 得到正式 bf_* 特征（28 selector 列 + 3 maturity + 3 diagnostic）；
- 窗口 anchor 满足 AGENTS.md 契约：不晚于目标位置的最近完整 update，窗口不小于名义窗口、偏差小于一次 population update（见 §3 的实现修正）；record 的 FE = 最后一次完整 update 的 FE（提取契约要求 native history 终点=record FE），checkpoint 表同时保存 checkpoint FE 与 snapshot_fe/effective window 元数据；**只生成 $B_t^{global}$，不存在任何 segment 字段**。

## 2. 对齐审计（13D）：1890/1890 全部通过，STOP 未触发

| 检查项 | 结果 |
|---|---|
| state_id 集合（suite/problem/instance/seed/current/FE） | 1890/1890 一一对应 |
| checkpoint log10-gap vs Task 12.1 恢复值（容差 1e-12） | **max abs diff = 0.0（逐位相等）** |
| best_fitness / log10 gap / population size / L-SHADE 实际种群 | 已记录（L-SHADE snapshot NP = {33, 26, 18}，线性缩减可见） |
| 确定性 RNG / native updates | snapshot_native_updates 全部单调；与 Stage-1 marks 的两条恒等式（Task 12.1）互证 |

结论：replay 轨迹与 Task 12 action outcomes 的源轨迹**逐位同一**，Behavior 特征与旧标签的拼接合法。

## 3. 正式 recorder 的收缩种群最小扩展（必须在论文方法学中声明）

Phase-1 以来的 Behavior 提取只覆盖 NP 恒定的求解器（pso/shade/cmaes），正式窗口模块存在两处隐含的恒定-NP 假设，L-SHADE（NP 40→4 线性缩减）首次触到：

1. **窗口量化容差**：原实现用"当前 NP"约束 anchor 偏差；协议语义是"偏差小于一次 population update"（anchor 处的实际 update 间隔）。已改为 anchor 邻域的局部 update 间隔（`trajectory/window_statistics.py`）。**对 NP 恒定的求解器两者数值恒等，历史特征值不受影响**——回归验证：对已提交的 `bbob_f001` Phase-1 轨迹重新提取，1274 行 × 全部 bf_* 列 diff = 0.0；
2. **窗口端点种群规模**：w02/w05/w10 的 fitness 分位数特征原本要求端点等长（sorted 逐位相减）。端点等长时保留原路径（逐位不变）；端点不等长时（仅收缩种群会出现）改用共同概率水平网格（n_max 个水平）比较两个分位函数。Wasserstein/质心/精英/协方差/overlap/chamfer 本就支持矩形矩阵，无需改动。

该扩展只影响 L-SHADE（及未来收缩种群算法）的提取；所有既有 artifacts 经回归证明逐位不变。L-SHADE 的 w 系列分位数特征使用网格估计器一事已如实记录，供 13E 特征审计与论文方法学部分引用。

## 4. 边界

- replay 只到 FE=6000（checkpoint 覆盖所需），不重建 6000–10000 段；
- 本轮依旧没有 post-handoff / segment 语义（states 全部 natural，`handoff=false`）；
- replay 的 wall time 约 3 分钟（8 进程并行），已在资源账本注明为近似值。
