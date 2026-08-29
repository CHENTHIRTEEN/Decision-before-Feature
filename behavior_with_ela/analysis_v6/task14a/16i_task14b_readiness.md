# 16i · Task 14B 就绪判定（Readiness）

- 日期：2026-08-29。Gate：action-space verdict ∈ {A1, A2}（Task 14A §27）。

## 1. Verdict 汇总

| Verdict | 结果 |
|---|---|
| Action-space | **A1 POST-HANDOFF ACTION SPACE ROBUST**（conservative sum 口径 switch-required 0.157–0.183 明显非零；无 route 全退化 continue；多 current 双向偏好；Δ_post=+0.105 ≫ 噪声；complementarity 未坍塌） |
| Reset-confound | **RC1 SOLVER-SPECIFIC EFFECT**（reset 有害、无法解释切换增益；schedule 混杂已排除） |

$$
\boxed{\text{Task 14B GO}}
$$

## 2. 14B 数据与协议就绪清单

| 项 | 状态 |
|---|---|
| post-handoff states（3780，6 方向 × 630） | ✅ `post_handoff_states.parquet` |
| B_global 28 列正式特征 | ✅ 3780 条（`post_handoff_behavior.parquet`, kind=global） |
| B_segment 28 列正式特征（segment 相对窗口） | ✅ 3780 条（kind=segment）；本轮未训练任何 bg/bs 模型 |
| 1000-FE next-action 真实标签 | ✅ 13,624 行（含 10%×R=3 重复） |
| post-handoff 噪声标定 | ✅ pooled solver δ95：shade 0.087 / lshade 0.095 / cso 0.048 |
| 分层字段 | ✅ suite / route / current / source FE / cv_group_id |

## 3. 14B 的注册比较（工作单 §33）

- $M_0=[current,FE,dwell]$；$M_G=[current,FE,dwell,B^{global}]$；$M_S=[\dots,B^{segment}]$；$M_{GS}=$ 并集；
- 核心问题：**$B^{segment}$ 是否超越 $B^{global}$**；
- 载体与 OOF 协议沿用 Task 13（正式 RF + Ridge 对照、leave-cv_group-out、fb paired bootstrap）；
- 基线注意：16h 显示 natural 域冻结策略在本域增益≈0，故 14B 的 M0 必须在本域 OOF 内重建，不得沿用 natural 域模型；
- dwell=1000 恒定 → dwell 列无变异（保留字段以兼容后续多 dwell 轮）。

## 4. 封存项

- ProgressForecast@1000FE：**PG3 NO-GO 不变**；
- CEC2017 formal = PAUSED；CEC2022 = HELD OUT；
- seeds 6–10：**保持 confirmation status**（本轮未读取/未分析），待 14B development protocol 预先确定后单独执行；
- κ=0.5/1.0 两个 pre-fixed candidate 维持，14A/14B 均不得重选。
