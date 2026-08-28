# Task 0：协议一致性检查报告（behavior_with_ela Phase 1 / Phase 2）

- 日期：2026-08-28
- 范围：`behavior_with_ela/model.py`、`baselines.py`、`phase2.py`、`phase2_action_loss_increment.py`、`local_landscape.py`、`action_dataset.py`、`configs/behavior_with_ela_{train,validation}.yaml` 与 `behavior_with_ela/results/` 现有产物。
- 方法：对真实结果文件运行时断言（脚本 `analysis_v2/task0_check.py`，证据 `results/analysis_v2/task0/task0_findings.json`），辅以代码路径核查。不因代码"看起来合理"下结论。
- 命名说明：按项目开发规范（AGENTS.md 0.3，普通一致性检查不得命名为 audit/审计），本报告命名为 consistency check。

## 检查结论总表

| # | 检查项 | 结论 | 证据摘要 |
|---|---|---|---|
| 1 | state-action 键唯一性 | PASS | 135,417 行无重复键 |
| 2 | 每 state 动作矩阵完整 | PASS | 45,139 个 state 恰有 3 个候选 {pso, shade, cmaes} |
| 3 | continue-current 为同 run 原生续跑 | PASS | 45,139 条 continue 行全部 `handoff_type=native_optimizer_state`，且 `action=continue_current` |
| 4 | 跨算法切换统一 population transfer | PASS | 90,278 条 switch 行全部 `handoff_type=population_transfer_initialization` |
| 5 | 重复配对正确 | PASS | 135,417 个 state-action 的 replicate_id 为 1 或 3 且无空洞；计数与 `action_repetitions` 计划一致 |
| 6 | family OOF 真 leave-family-out | PASS | 5 折；每条 OOF 预测行的 `family == heldout_family`；50,938 行恰好覆盖 bbob_train 全部 switch 行 |
| 7 | MA-BBOB component 泄漏防护 | PASS | 24 个 selected definition 无一携带被扣函数族的函数身份；重叠组件的 MA 行按折被 `_ma_overlaps_heldout` 排除 |
| 8 | validation 完全未参与 fit/threshold | PASS | `validation_rows_used_for_model_fit = 0`、`validation_rows_used_for_threshold_fit = 0` |
| 9 | function balancing 语义 | PASS | 手工计算 mean-of-per-family-means 与汇总一致（0.846671865） |
| 10 | first-trigger 每 run 至多一次、最早越过 | PASS | 1,620 runs 恰 1,620 行；触发的 1,243 个 BBOB run 的 `selected_FE` 全部等于该阈值下最早越过 state |
| 11 | action-loss 回归目标矩阵列序稳定 | PASS | 目标矩阵 45,139×3；25 个随机 state 逐列核对 [pso, shade, cmaes] 顺序 |
| 12 | A1 vs A3 validation 决策一致非代码问题 | PASS | 验证集 selected action/FE 完全一致（重核）；模型预测分数确实不同（相关 0.959，平均差 0.398）；train OOF state 级 top-1 分歧率 1.60%（25,469 states） |
| 13 | train OOF 策略指标存在阈值选择乐观性 | WARNING | 所选阈值 0.2445 是在 40,060 个候选上最大化同一 OOF 指标得到（候选中位数 0.5709）；缓解：主要结论同时在未触碰的 BBOB validation 上报告 |

## 补充说明

1. **阈值乐观性（唯一 WARNING）**：train OOF 上的策略指标按构造偏乐观；本报告后续所有 delta 敏感性、prefix-aware 等诊断均明确区分 train OOF 与 validation，最终方法裁决以 validation 为独立确认（不参与任何选择）。
2. **A1 vs A3 一致性**：验证集 first-trigger 决策 100% 相同不是数据复制错误——两组模型预测分数不同，但在验证集上分数符号一致率 100%，且阈值扫描最终决策重合；train OOF 上 top-1 分歧率 1.60%，且分歧处强制采用 A3 反而变差（见 05 报告）。
3. 本轮检查未发现任何 FAIL；未发现需要重新生成 Phase 1/2 数据的问题。
4. 全部检查可复现：`.venv/bin/python behavior_with_ela/analysis_v2/task0_check.py`。
