# Decision-before-Feature：CEC 前部署性与行为贡献确认报告（analysis_v3 总报告）

- 日期：2026-08-29
- 范围：Task 9A–9H 全部完成；未运行 CEC2017 / CEC2022 的任何 objective evaluation（唯一例外为此前已提交的三函数开发性 sanity check，见 `analysis_v2/09_quick_cec_transfer_check.md`，本轮未新增）。所有数字从仓库真实产物重新计算；与旧报告不一致处以真实产物为准（差异清单见 §8）。
- 术语说明（Task 9G）：旧报告中的 `best_observed_one_switch`（"83.8% oracle gain captured" 一类表述）统一改称 **prefix 条件单次切换上界**（`prefix_conditioned_one_switch_upper_bound` / `conditional_one_switch_upper_bound`）。它是"给定 prefix 轨迹后逐状态最佳单次切换动作"的上界，不是全局优化 oracle；历史 parquet 字段名不变，报告与论文层一律使用新名称。相应地，旧表述"捕获 oracle 增益的 83.8%"改写为"捕获 prefix 条件单次切换可得增益的 83.8%"（1.8504 / 2.2080 = 0.838，本轮复核一致）。

---

## 1. 七个研究问题的答案

### RQ1. Behavior v2 是否优于 Prefix-only？

**否。** 三行查表（仅 prefix 身份，train 函数平衡平均增益学得：pso→cmaes、shade→cmaes、cmaes→不切）+ 同一 first-trigger 阈值协议，在 train OOF 上 1.3137 vs v2 1.3141（逐 run 配对差 +0.0003，由 v2 在 6 个 run 上的零星非 0.2B 切换造成），在 validation 上 1.8504 vs 1.8504（540 个 run 决策完全一致，差值为 0.0000）。

### RQ2. Behavior v2 是否优于完全匹配的 Fixed-0.20？

**否。** Fixed-0.20 learned mapping（仅 prefix 输入、仅 FE=2000 决策、映射由训练数据学得而非手写）与 v2 的差异同样是 train +0.0003 / validation 0.0000。timing 完全匹配后，Behavior 输入没有带来任何可测增量。

### RQ3. 多个 decision opportunities 是否优于 Behavior@0.20 单次决策？

**否。** 只在 0.2B 决策一次的 Behavior@0.20 与 full dynamic v2 决策完全一致（train 配对差 +0.0003 / validation 0.0000）。v2 的 OOF 分数在首个机会越过阈值后，后续机会从不再次触发切换；"动态调度"在当前数据上没有发生过。

### RQ4. 从 SBS=CMAES 开始时，v2 是否仍产生正 terminal gain？

**否，但也无损失。** cmaes-start 场景（1,080 runs）中 v2 从不切换，端点与 SBS 逐 run 完全相同：配对增益 $G_{v2|SBS}=0.0000$（train 与 validation 一致，success rate 相同 0.398/0.333）。v2 在强求解器轨迹上没有识别出任何有利转移机会；同时 03 报告显示 all-prefix 诊断口径下 v2 **显著劣于 SBS**（validation 配对增益 -0.477，95% CI [-0.861, -0.216]，6/6 函数为负）。

### RQ5. MA-BBOB augmentation 是否对最终 regression 有价值？

**有（helps regression），且无负贡献。** train family OOF 上 R-BM vs R-B：增益 1.3141 vs 1.1074、pairwise 排序 0.7607 vs 0.7193、top-1 命中 0.6453 vs 0.5105、三类有害切换率同时下降；held-out validation 上两者决策完全一致（不低于 R-B）。MA definition 分组 OOF 回归诊断 5 折中 4 折增益 Spearman > 0.79（fold 3 为 0.285，存在定义子集敏感性）。

### RQ6. 修正后的 OOF action uncertainty 是否能识别 harmful switch？

**能，但结构有边界。** 修正两处实现问题（风险标签从"真实最佳切换增益"改为"模型实际选中动作的真实增益"；树级不确定性从 full-train 集成改为 family-OOF 折模型）后：harmful H0/H50 由预测选中切换增益边际排序（AUC 0.818/0.805；底部三个十分位风险率 0%，顶部十分位 47.0%）；harmful H95（深度有害）集中在"高树间一致性 + 中等预测边际"状态（一致性特征 AUC 0.85，方向与旧 Task 6 的"一致地错"结论一致）；missed helpful 由高树分歧标记（AUC 0.78）。修正本身把 H95 有害切换从 744 修为 829（+85，旧标签系统性低估）。基线率低（H95 占 state 3.3%）且增益边际与 prefix 高度混杂，做门控的代价需专门评估；本轮仍不实现 Query Gate。

### RQ7. Behavior 与 Local Landscape 的信息重叠在 grouped OOF 后是否仍成立？

**仅函数内部分成立；跨函数不成立。** state 去重 + 分组预测后：leave-BBOB-family-out 下 lf→bf 中位 R² 从旧报告的 0.475 缩水到 0.118（R²>0.5 目标占比 46.4%→10.7%），bf→lf 中位 0.186 且无任何目标超过 0.5；按 problem_id 分组时 lf→bf 中位 0.391，头部目标全部是多样性/散布类（`bf_diversity_mean_pairwise` 0.81、`bf_population_chamfer_distance_w05` 0.77 等），improvement/stagnation 类几乎不可预测。旧 0.475 确认为 state 重复行泄漏造成的乐观值，**予以撤回**。更直接的 actionable-overlap 证据不变：A1 vs A3 的 OOF top-1 分歧率 1.60%，分歧处强制 A3 的真实增益差 mean -2.58（68:338 劣于 A1）。

---

## 2. 部署性判据核对（工作单 §11）

| # | 判据 | 结果 |
|---|---|---|
| 1 | protocol 无新增 FAIL | **成立**。本轮 0 FAIL；新增校验全部通过（OOF 不确定性折模型与 Task 1 OOF 逐行一致 <1e-9；SBS 配对参照与独立 static summary 差 0.012–0.054，来源已解释；各 split 覆盖断言通过） |
| 2 | v2 明显优于 Prefix-only 或 Fixed-0.20 之一 | **不成立**（RQ1/RQ2：完全等价） |
| 3 | v2 优势不能完全由 PSO/SHADE→CMAES 解释 | **不成立**（Task 9A：切换 100% 集中于弱起始→cmaes，2/3 切换率精确等于弱起始占比） |
| 4 | CMAES-start 无明显灾难性恶化 | **成立**（与 SBS 逐 run 相同，零有害切换） |
| 5 | MA augmentation 对 regression 无负贡献 | **成立**（RQ5） |
| 6 | online regression adapter 已实现 | **成立**（`online.py` 回归分支已在 09 报告验证；本轮把策略名 `behavior_action_loss_regression_v2` 正式注册进 `online_comparison.py`，含 bundle 校验与 `--v2-regression-model` 参数，import 冒烟通过） |
| 7 | CEC 前模型与阈值保持不变 | **成立**（28 维特征、RF(200,8,sqrt)、阈值 0.145087、delta 1.4639、portfolio [pso,shade,cmaes]、one-switch first-trigger 规则均未改动；本轮 0 次重训练主模型、0 次用 validation 调规则） |

判据 2、3 不满足，触发工作单的预定分支：**v2 ≈ Fixed-0.20，可以进入 CEC，但研究定位必须修改**。

## 3. 最终判定：CONDITIONAL GO（定位修改后进入 CEC2017）

**不得再使用的主张**：
- "strongly dynamic behavior scheduling" / "Behavior 驱动的动态切换调度"；
- "v2 优于 SBS 的 deployable DAS"；
- 把 all-prefix 口径的 gain-over-continue（validation 1.8504）表述为相对 SBS 的策略优势。

**修改后的研究定位**（与 02/03 报告的证据一致）：
1. **trajectory-conditioned early algorithm selection**：给定任意起始求解器的前缀轨迹，在 0.2B 处做一次继续/切换决策；当前实现的行为等价于"prefix 条件的早期切换规则"，Behavior 回归分数是其实现载体。
2. **behavior-based recovery / continuation selection from arbitrary solver trajectories**：价值主张是"从弱/未知求解器轨迹中恢复"（validation 上相对 continue 增益 1.85、捕获 prefix 条件单次切换可得增益的 83.8%），而非"超越静态最优"。
3. CMAES-start（=SBS 起始）是默认部署场景，其预期行为是"与 SBS 相同且无有害切换"；若 CEC2017 上出现 cmaes-start 的切换触发，将是对 Behavior 状态依赖性的首个真实证据，值得单独报告。
4. action uncertainty（修正后）作为后续 abstention / optional-query 的候选机制保留，但不进入本轮 CEC 评价。

## 4. CEC2017 精确命令（按工作单要求仅列出，不执行）

正式评价分两个初始算法口径；策略面板已含 SBS/continue/fixed/random-matched/time-only/三分类/v2 回归/traditional AAS（Query 类 bundle 不存在，不选入）：

```bash
# (a) all-prefix 口径（诊断主口径，与离线分析对齐）
.venv/bin/python -m behavior_with_ela.online_comparison \
  --config configs/behavior_with_ela_cec2017.yaml \
  --output behavior_with_ela/results/online/cec2017_v3_all_prefix \
  --initial-algorithm all --workers 8 --overwrite \
  --policy continue_current --policy sbs --policy traditional_aas \
  --policy random_one_switch --policy random_matched_switch_rate \
  --policy fixed_030_transition --policy time_only_action_gain \
  --policy behavior_action_loss_rf --policy to_switch_style_rf \
  --policy phase1_action_gain --policy behavior_action_loss_regression_v2 \
  --phase1-model behavior_with_ela/results/model/behavior_action_gain/models.joblib \
  --baseline-model behavior_with_ela/results/baselines/phase1/baseline_models.joblib \
  --traditional-aas-model behavior_with_ela/results/model/traditional_aas/traditional_aas_models.joblib \
  --v2-regression-model behavior_with_ela/results/analysis_v2/task9_quick_cec/v2_online_bundle.joblib

# (b) SBS-start 口径（deployable 主口径：初始算法 = 训练派生 SBS = cmaes）
.venv/bin/python -m behavior_with_ela.online_comparison \
  --config configs/behavior_with_ela_cec2017.yaml \
  --output behavior_with_ela/results/online/cec2017_v3_sbs_start \
  --initial-algorithm sbs --workers 8 --overwrite \
  （--policy / 模型路径参数与 (a) 完全相同）
```

约后生成汇总报告写入 `behavior_with_ela/analysis_v3/07_cec2017_transfer.md`。CEC2017 只评价、不训练；不得据其结果回调模型或阈值后重报同一套结果。

## 5. 交付物清单

- 报告：`analysis_v3/01–06` 六份 + 本总报告
- 分析代码：`analysis_v3/common.py`、`task1_per_prefix.py`、`task2_prefix_time_baselines.py`、`task3_v2_vs_sbs.py`、`task4_oof_uncertainty.py`、`task5_mabbob_regression.py`、`task6_grouped_redundancy.py`
- 轻量表：`analysis_v3/task1–task6/`（策略面板、配对差、逐函数增益、AUC 表、十分位风险表、汇总 json）
- 重产物（不入 git）：`results/analysis_v3/task1–task6/`（run 级明细、OOF 状态表、阈值表）
- 代码改动：`online_comparison.py` 注册 `behavior_action_loss_regression_v2` 策略（默认 v2 bundle 指向 quick-CEC 已验证产物）

## 6. 本轮零消耗声明

新 objective evaluations = 0；新 FE 消耗 = 0；模型重训练仅限消融所需的 OOF 折模型（R-B、OOF 不确定性折、MA definition 折），未触碰主模型与阈值；未删除任何 Phase 1/2 阴性结果。

## 7. 与旧报告的数字差异清单

| 旧表述 | 本轮修正 |
|---|---|
| Task 5：lf→bf R² 中位 0.475、46.4% 目标 R²>0.5 | **撤回**（行级划分的 state 重复泄漏）；grouped 口径为 0.118（跨族）/0.391（同函数） |
| Task 6：harmful switch rate 0.0291（best-switch 标签 + 训练集树方差） | 修正标签后 H95 为 829/25,469 = 3.26%（state 级），旧值漏掉 85 个"选中动作有害"状态；AUC 方向结论不变、数值更保守 |
| "83.8% oracle gain captured" | 改称"捕获 prefix 条件单次切换可得增益的 83.8%"（数值复核一致） |
| v2 gain 1.3141 / 1.8504、阈值 0.1451、delta 1.464 | 复核一致，无变化 |

## 8. 下一步建议

1. 按本报告 §4 命令执行 CEC2017 正式评价（两个初始口径），生成 07 报告；
2. 论文叙事按 §3 的定位修改重写方法与贡献部分；
3. 后续研究线索（不进入本轮）：uncertainty-aware abstention 的代价评估、per-opportunity 规则下的 dwell/滞回协议（Repeated DAS）、以及"prefix 条件切换规则的 CEC 迁移是否保持"这一核心问题。

### 下一步 prompt（可直接复制开新对话）

```
你正在继续 GitHub 项目 Decision-before-Feature（工作目录 behavior_with_ela/）。
先阅读 behavior_with_ela/analysis_v3/ 下全部 6 份编号报告与
Decision-before-Feature_CEC前部署性与行为贡献确认报告.md，
并遵守仓库根目录 AGENTS.md（实验隔离、禁止机制、术语与输出规范）。

当前状态：analysis_v3 已完成 Task 9A-9H，主候选 behavior_action_loss_regression_v2
已被证明与 Prefix-only / Fixed-0.20 完全等价，CEC 判定为 CONDITIONAL GO，
研究定位已修改为 trajectory-conditioned early algorithm selection +
behavior-based recovery from arbitrary solver trajectories。
模型与阈值（0.145087）保持不变，本轮禁止重训或调参。

本轮任务：执行 CEC2017 正式在线评价并生成报告。
1. 运行总报告 §4 的两条命令（all-prefix 与 sbs-start 两个口径），
   输出到 behavior_with_ela/results/online/cec2017_v3_all_prefix 与 cec2017_v3_sbs_start；
2. 校验 complete_path_timings 全部由实测生成（timing_source=
   measured_complete_policy_path，3 次重复 cyclic order，无 failed run）；
3. 汇总两个口径的函数平衡指标：mean log10 gap、相对 continue 增益、相对 SBS
   配对增益、switch rate、切换时点、目标分布、success rate、gap closed fraction；
4. 重点回答：(a) prefix 条件切换规则是否迁移到 CEC2017；(b) sbs-start 口径下
   是否出现任何 cmaes-start 切换触发（Behavior 状态依赖性的真实证据）；
   (c) v2 与三分类载体在 CEC 上是否保持离线排序；
5. 将结果写入 behavior_with_ela/analysis_v3/07_cec2017_transfer.md，
   更新 GO 判定，并给出论文级的定位表述建议；
6. 禁止：CEC2022、工程问题、20D/40D、新增 ELA/Query、重训模型、
   据 CEC 结果回调阈值后重报同一套结果、删除历史阴性结果。
```
