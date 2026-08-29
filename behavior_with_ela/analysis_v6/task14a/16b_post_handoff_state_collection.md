# 16b · Post-Handoff State 采集汇总（Task 14A Collection）

- 产物：`post_handoff_states.parquet`（3780 行）、`post_handoff_action_outcomes_1000.parquet`（13,624 行）、`reset_control_outcomes.parquet`（2,520 行）、`post_handoff_behavior.parquet`（7,560 行 = 3780 B_global + 3780 B_segment）。

## 1. 状态覆盖

| 维度 | 值 |
|---|---|
| handoff directions | **6/6 全保留**，每方向 630 states（42 problems × 5 seeds × 3 source FE） |
| suites | bbob 2700 / mabbob 1080 |
| source FE ∈ {2000,4000,6000} | 各 1260 states |
| segment_start / segment_age | = source FE / 恒为 1000 FE |

## 2. Post-handoff attained gap

| source FE | 2000 | 4000 | 6000 |
|---|---:|---:|---:|
| fb log10 gap | −0.726 | −2.323 | −3.612 |

route 均值：lshade→shade −2.880（最好）、shade→cso −1.737（最差）——与 Task 12 natural 强度排序一致。

## 3. L-SHADE 缩减状态保持验证

post-handoff states 中 current=L-SHADE 的实际 NP 记录（如 source_fe=6000 时 NP=36→33→18 系列）证明转移后的 L-SHADE 沿用全局 FE 驱动的缩减阶段；reset 分支的 `population_size_preserved` 与 `reduction_max_fe`/`schedule_max_evaluations` 字段逐行记录（15g/16g 引用），**未出现恢复 NP=40 或 schedule 重启**。

## 4. Behavior 记录

- **B_global**：全局 recorder 自 0 起连续累积（跨 handoff），fe_total=10000 的正式窗口；
- **B_segment**：segment recorder 于 handoff 重建（首观察 fe=segment_start），窗口为 **segment 相对**（fe_total=1000 → w02/w05/w10 = 20/50/100 FE）。协议说明：segment 年龄恰为 1000 FE 时，全局口径的 w10 anchor 会落在 segment 起点之外；segment 相对窗口使全部 anchor 位于段内，14B 沿用同一定义；
- 两类特征均为正式 `extract_behavior_rows` 输出（28 selector + 3 maturity + 3 diagnostic），本轮只记录、不训练。
