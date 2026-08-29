# 13a · Task 12 Stage-2 数据语义核查（Task 12.1A）

- 日期：2026-08-29
- 性质：纯数据/代码语义核查，不含任何统计结论；零新增 objective evaluations。
- 数据：`results/analysis_v5/task12/{dynamic_screening_states, dynamic_action_outcomes_1000, dynamic_solver_loss_matrix}.parquet`、`results/portfolio_screening/task12/stage2/shards/{branches,addback}.parquet`、`task12_stage2.py` / `task12_stage2_analysis.py` / `task12_addback.py` 源码。
- 机器可读版：`12a_data_semantics_checklist.json`。

## 1. 14 项核查结果

| # | 项目 | 结果 | 证据 |
|---|---|---|---|
| 1 | Stage 2 states 是否全部为 natural-current states | **是** | `route == "current_"+current`、`source==current` 全部 1890/1890 成立 |
| 2 | 是否没有真实 post-handoff states | **是（没有）** | `handoff_performed` 全 False，`dwell_FE == FE`，无任何 handoff 路径 |
| 3 | `source_algorithm == current_algorithm` 恒成立 | **是** | 1890/1890 |
| 4 | `handoff_performed == False` 恒成立 | **是** | 1890/1890 |
| 5 | states table 是否保存 checkpoint current log-gap（$\ell_t$） | **否（原本缺失）** | 列表中无该字段；本轮已从 Task 12 Stage-1 自然跑 marks 恢复（见 13e 与 `states_with_checkpoint_gap.parquet`） |
| 6 | 是否保存 bg_*（global Behavior） | **否** | states 无任何 `bg_` 前缀列 |
| 7 | 是否保存 bs_*（segment Behavior） | **否** | states 无任何 `bs_` 前缀列 |
| 8 | practical best 是否使用 `tied[0]` | **是** | `task12_stage2_analysis.py`：`best_practical[state_id] = tied[0]` |
| 9 | candidate order 是否固定 `shade, lshade, cso` | **是** | `CANDIDATES = ("shade","lshade","cso")` 即 tie 序 |
| 10 | oracle 是否缺少 `current` 条件 | **是** | Task 12 oracle 仅 condition on `problem` / `problem+FE` |
| 11 | add-back 是否没有 CMAES-current states | **是** | cmaes 仅作为动作分支出现，从不作为 `current_algorithm` |
| 12 | repetition 是否约 10% state-action | **是：9.82%** | 557/5670 cell 被 SampleSequence 确定性抽样 ×R=3 |
| 13 | continue branch 是否为真实 $t{+}1000$ current-solver outcome | **是（逐位验证）** | Stage-2 continue 分支 loss 与 Stage-1 同轨迹 FE=t+1000 mark 全部 1890 项 diff=0.0（`replay_alignment.parquet`） |
| 14 | 1000-FE 是否精确执行 1000 evaluations | **是** | `_run_horizon_branch` 内 `evaluations != remaining` 即 raise；采集期无异常 |

## 2. tie-breaking 偏置的量化（#8/#9 的后果）

在 Task 12 的 suite δ95 语义下：

- **50.3%** 的 states 其 practical tie set（$\{a: L_a \le L_{\min}+\delta\}$）元素数 >1；
- **24.2%** 的 states 被 `tied[0]` 指到了**不是 raw argmin** 的 solver。

即 Task 12 报告的 practical best 分布、转移表、practical entropy、dominance 与 within-trajectory variation 在约四分之一的 states 上受固定 candidate 顺序影响。这些指标以本轮 13c/13d 的 set-valued、current-preserving 重算为准；Task 12 的 `best_action_practical` 列此后仅作 legacy 对照，不再作为主结论依据。

## 3. 两个缺失字段的恢复/不可恢复声明

1. **checkpoint log-gap $\ell_t$**：Stage-2 表未存。本轮通过两条逐位恒等式验证 Stage-1 marks 与 Stage-2 轨迹完全一致后，直接取自 Stage-1（未执行任何 replay、未消耗任何 FE）。明细：`results/analysis_v5/task12_1/states_with_checkpoint_gap.parquet`。
2. **Behavior 字段（bg_*/bs_*）**：不可从现有 artifacts 恢复（Stage-1 只记录每 1000-FE 的 log-gap mark，无 optimizer 状态/运行历史）。后续如需 Behavior 必须做一次带正式 recorder 的确定性状态重建；本账目见 13g。

## 4. 对后续轮次的约束

- Stage-2 states 全部为 natural-current（segment_start=0），$B^{segment}\approx B^{global}$ 是构造使然：**Task 12 数据不能证明任何 post-handoff segment 增量**；
- `dynamic_solver_loss_matrix.parquet` 的 `best_action_practical`（tied[0]）自本轮起降级为 legacy 敏感性列；
- add-back 相关结论一律加 "ONE-STEP" 前缀（无 CMAES-current 重复状态）。
