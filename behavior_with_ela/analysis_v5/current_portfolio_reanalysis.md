# behavior_with_ela 当前组合重分析

> 本报告只使用 `behavior_with_ela/` 内 Task 12、Task 13、Task 14 的当前结果。旧 Phase 1 的 `results/actions/`（`{pso, shade, cmaes}`）不纳入分析。

## 1. 当前实验对象

- 当前组合：`{shade, lshade, cso}`；CMA-ES 仅作为隔离的 add-back control，不作为当前组合动作。
- Task 12 natural action space：1,890 states = 42 problems × 5 seeds × 3 current algorithms × 3 FE checkpoints（2,000/4,000/6,000）。
- 主动态 horizon：1,000 FE；Task 13 使用 28 列 Behavior，主 carrier 为 RF；Task 14 使用真实 post-handoff states。
- 本文件夹当前结果不使用 `g_fe` / `g_fe_selected_path` 字段；对应指标是 `loss_*`、`gain_vs_continue`、`switch_required` 和 grouped-OOF policy loss。
- 本次仅做读取和聚合：new objective FE = 0；没有重新训练模型、重选组合或修改标签。

## 2. Natural action space

严格收益定义为 `continue_loss - target_loss > 0`；practical `switch_required` 使用 Task 13 的 set-valued action 规则，不把 within-delta 平局强行压成单一 winner。

| suite   |   states |   strict_any_alt_gt_zero_rate |   practical_switch_required_rate |   mean_best_alt_gain |
|:--------|---------:|------------------------------:|---------------------------------:|---------------------:|
| bbob    |     1350 |                        0.443  |                           0.2578 |              -0.0244 |
| mabbob  |      540 |                        0.4926 |                           0.2648 |               0.0196 |

按 current 与 FE 的完整表见 `natural_action_summary.csv`；切换目标条件分布见 `natural_transition_summary.csv`。

| suite   | current_algorithm   | switch_target   |   states |   conditional_share |
|:--------|:--------------------|:----------------|---------:|--------------------:|
| bbob    | cso                 | lshade          |       71 |              0.6017 |
| bbob    | cso                 | shade           |       47 |              0.3983 |
| bbob    | lshade              | cso             |       73 |              0.6404 |
| bbob    | lshade              | shade           |       41 |              0.3596 |
| bbob    | shade               | cso             |       40 |              0.3448 |
| bbob    | shade               | lshade          |       76 |              0.6552 |
| mabbob  | cso                 | lshade          |       29 |              0.6444 |
| mabbob  | cso                 | shade           |       16 |              0.3556 |
| mabbob  | lshade              | cso             |       37 |              0.6981 |
| mabbob  | lshade              | shade           |       16 |              0.3019 |
| mabbob  | shade               | cso             |       27 |              0.6    |
| mabbob  | shade               | lshade          |       18 |              0.4    |

当前正确的方向来自六个 transition：`shade↔lshade`、`shade↔cso`、`lshade↔cso`。没有 PSO→CMA-ES 或 SHADE→CMA-ES 的当前组合结论。

## 3. Task 13 Behavior 增量

Task 13 的对象是 natural P_balanced states 上的 1,000-FE action selection，不是独立 query 是否值得执行。

| model                  | carrier   | suite   |   functions |   function_balanced_realized_loss |   function_balanced_gain_vs_continue |   function_balanced_switch_rate |   function_balanced_harmful_rate |
|:-----------------------|:----------|:--------|------------:|----------------------------------:|-------------------------------------:|--------------------------------:|---------------------------------:|
| M0_current_FE          | rf        | bbob    |          10 |                           -1.5634 |                              -0.0421 |                          0.2556 |                           0.0652 |
| M1_behavior            | rf        | bbob    |          10 |                           -1.6054 |                              -0.0001 |                          0.6711 |                           0.1489 |
| M2_current_FE_behavior | rf        | bbob    |          10 |                           -1.6107 |                               0.0052 |                          0.6422 |                           0.1356 |
| M3_current_only        | rf        | bbob    |          10 |                           -1.5767 |                              -0.0288 |                          0.1    |                           0.0393 |
| M4_FE_only             | rf        | bbob    |          10 |                           -1.5212 |                              -0.0843 |                          0.6667 |                           0.2052 |
| M0_current_FE          | rf        | mabbob  |          12 |                           -4.5219 |                              -0.0063 |                          0.2685 |                           0.0444 |
| M1_behavior            | rf        | mabbob  |          12 |                           -4.5732 |                               0.0451 |                          0.6463 |                           0.1704 |
| M2_current_FE_behavior | rf        | mabbob  |          12 |                           -4.5731 |                               0.045  |                          0.5907 |                           0.137  |
| M3_current_only        | rf        | mabbob  |          12 |                           -4.5068 |                              -0.0214 |                          0.0833 |                           0.0426 |
| M4_FE_only             | rf        | mabbob  |          12 |                           -4.4756 |                              -0.0526 |                          0.6667 |                           0.2167 |

| carrier   | suite   |      W0 |      W1 |     W2 |   delta_W0_minus_W1 |   delta_W0_minus_W2 |
|:----------|:--------|--------:|--------:|-------:|--------------------:|--------------------:|
| rf        | bbob    | -1.6573 | -1.6768 | -1.676 |              0.0195 |              0.0188 |
| rf        | mabbob  | -4.5859 | -4.6003 | -4.602 |              0.0144 |              0.0161 |

解释：RF 的 Behavior policy 在 natural 域相对 `current+FE` 的点估计增量为 BBOB +0.047、MA +0.051；MA 的 grouped-OOF 区间不跨 0，BBOB 区间较宽。within-problem LOSO 的 RF 增量在两个 suite 均为正，说明存在固定 problem/current/FE 后的 state 区分信号。

风险同时上升：raw Behavior policy 的 harmful rate 约为 BBOB 0.136、MA 0.137，不能只报告平均 policy loss。

## 4. Post-handoff Task 14

Task 14A/14B.1 的 3,780 个 states 仍使用 `{shade,lshade,cso}`；它们不是 Task 12 natural states 的简单重复，而是 handoff 后 commitment 状态。

| suite   | policy      |   realized_fb_loss |   n_states |
|:--------|:------------|-------------------:|-----------:|
| bbob    | Continue    |            -1.9692 |       2700 |
| bbob    | lookup      |            -1.9847 |       2700 |
| bbob    | M0          |            -1.9835 |       2700 |
| bbob    | MG          |            -1.9543 |       2700 |
| bbob    | MS_old      |            -1.9421 |       2700 |
| bbob    | MS_matched  |            -1.94   |       2700 |
| bbob    | MGS_old     |            -1.9413 |       2700 |
| bbob    | MGS_matched |            -1.946  |       2700 |
| mabbob  | Continue    |            -5.0322 |       1080 |
| mabbob  | lookup      |            -5.0471 |       1080 |
| mabbob  | M0          |            -5.0431 |       1080 |
| mabbob  | MG          |            -5.019  |       1080 |
| mabbob  | MS_old      |            -5.0058 |       1080 |
| mabbob  | MS_matched  |            -5.0149 |       1080 |
| mabbob  | MGS_old     |            -5.0239 |       1080 |
| mabbob  | MGS_matched |            -5.0214 |       1080 |

| comparison            | suite   |   fb_mean |   ci_low |   ci_high |
|:----------------------|:--------|----------:|---------:|----------:|
| MGS_matched_vs_lookup | bbob    |   -0.0387 |  -0.0628 |   -0.0179 |
| MGS_matched_vs_lookup | mabbob  |   -0.0257 |  -0.0596 |    0.0213 |
| MGS_matched_vs_M0     | bbob    |   -0.0376 |  -0.0585 |   -0.0176 |
| MGS_matched_vs_M0     | mabbob  |   -0.0217 |  -0.0595 |    0.0368 |
| MGS_matched_vs_MG     | bbob    |   -0.0083 |  -0.0232 |    0.0008 |
| MGS_matched_vs_MG     | mabbob  |    0.0024 |  -0.0095 |    0.016  |

Task 14B.1 的结论是 post-handoff generic Behavior 没有超过 M0/lookup，segment 相对 global 也没有增量；该结论不否定 Task 13 在 natural 域的条件增量，而是说明增量不能直接迁移到 handoff 后域。

## 5. 更正后的结论

1. 当前组合的动态动作空间确实是 `{shade,lshade,cso}`，且 practical `switch_required` 约为 26%，方向分布在六个组合之间。
2. Natural 域中 Behavior 对 1,000-FE action selection 有条件增量（Task 13：A2 CONDITIONAL；within-problem：B1 GENUINE STATE VALUE），但切换风险更高。
3. Post-handoff 域中，generic global/segment Behavior 未提供额外 policy 增量（Task 14B.1：最终 NO-GO）。
4. 因此，之前的 `PSO→CMA-ES`、`SHADE→CMA-ES`、`CMA-ES→继续` 只能归入旧 Phase 1 历史结果，不能用于解释当前 `{shade,lshade,cso}` 实验。

## 6. 产物

- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/natural_action_summary.csv`
- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/natural_transition_summary.csv`
- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task13_oof_summary.csv`
- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task13_within_problem_summary.csv`
- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task14b1_policy_summary.csv`
- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task14b1_pairwise_bootstrap.csv`
