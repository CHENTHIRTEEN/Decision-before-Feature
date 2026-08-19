# Decision-before-Feature 项目交接记录

本文仅保留当前活动研究协议、仓库状态与后续工作指引。旧版本推导、已历史结果、过时口径和历史长篇说明已删除。

## 当前状态

- 主协议已切换为方案 A：`G_FE` 作为主功效，`runtime` 仅作为独立资源/计时维度。
- `action_loss` 统一按严格等总 FE 预算下的 FE-indexed optimization loss 解释，不得用 wall-clock 定义科学标签。
- 当前工作区保持干净。

## 当前活动模块

- `trajectory/`：轨迹与最终性能底层数据。
- `behavior/`：行为特征与提取逻辑。
- `landscape_queries/`：query 采样与特征提取。
- `selection_reference/`：statewise action-loss 参考与 selector 训练。
- `utility_labels/`：utility / efficacy 标签生成与校验。
- `decision/`：模型协议、训练、评估与控制器。
- `experiments/cli/`：所有实验 CLI 入口。
- `benchmarks/`：BBOB / CEC benchmark 适配。

## 仍需完成的事项

- 继续闭合 offline decision-state-to-terminal runner 与 replay consumer。
- 继续完成 BBOB-validation、CEC2017 / CEC2022 和工程集合的正式评估闭环。
- 继续清理历史文档中残留的联合效用 / 操作性增量命名、`performance_gain_norm`、`time_cost_norm` 与时间主标签口径。

## 文档指引

- 活动协议以 `AGENTS.md` 为最高优先级。
- 设计与训练协议以 `docs/10_protocols/Decision-before-Feature Decision Model设计与训练协议.md` 为准。
- 最小字段规范以 `docs/10_protocols/Decision-before-Feature_最小ActionLoss字段规范.md` 为准。
- 结果概览见 `docs/30_results/phase1_current_results.md`。
