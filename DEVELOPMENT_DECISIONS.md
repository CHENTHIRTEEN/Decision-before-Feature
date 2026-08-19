# Decision-before-Feature 开发前裁决

本文仅保留当前仍生效的项目级裁决。已废弃的历史推导、事故复盘长文和过时口径已删除。

## 1. 最高优先级

- `AGENTS.md` 是最高优先级活动规范。
- `PROJECT_HANDOFF.md` 仅作当前状态摘要。
- 若 `docs/` 与活动规范冲突，以 `AGENTS.md` 和本文件为准。

## 2. 当前生效裁决

- `G_FE` 是主功效；`runtime` / `wall-clock` 仅作为独立资源与计时维度。
- `action_loss` 统一解释为严格等总 FE 预算下的 FE-indexed optimization loss。
- 任何科学标签不得由 wall-clock time 定义。
- 活动 Decision Model 只使用当前协议冻结的特征、标签与 threshold 口径。
- Selection Reference 的 action-loss 回归、Utility labels、Decision labels 与在线评估必须共享一致的数据契约。
- 所有 runtime / wall-clock 对比必须使用真实 replay，不得用 component runtime 合成完整路径时间。
- 所有正式实验前必须完成数据契约、字段一致性、replay plan、artifact 路由与 failure materialization 的核对。
- 旧 `u_query_joint_lamT_*`、`performance_gain_norm`、`time_cost_norm` 和其他时间主标签口径仅可作为诊断，不得作为活动目标。

## 3. 已废弃内容

以下内容已从活动裁决中删除，不再保留为当前工作依据：

- 早期重建式 continuation 口径
- 以时间加权 Utility 作为主标签的旧方案
- 逐状态 threshold 的旧模型选择方式
- 旧版 validation 参与选模或阈值拟合的口径
- 用合成 timings 替代实测 replay 的做法
- 任何将 runtime 当作主科学标签组成项的旧表述
