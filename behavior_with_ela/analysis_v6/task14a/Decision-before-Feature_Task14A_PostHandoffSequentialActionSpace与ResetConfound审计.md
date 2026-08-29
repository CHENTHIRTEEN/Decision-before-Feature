# Decision-before-Feature：Task 14A 真实换挡后序贯动作空间 × 重置混杂审计 总报告

- 日期：2026-08-29。前置门：Task 13.1-H hygiene = H1 NEGLIGIBLE → **GO 确认**。
- 成本：新 objective FE **23,704,000**（source natural 3.78M / commitment 3.78M / next-action branch 13.62M（含 10%×R=3 重复）/ reset control 2.52M，分类账本 `task14a_collection_ledger.parquet`）；wall ≈195 s @8 workers。
- 报告组：`analysis_v6/task14a/16a–16i`；行级表 `results/analysis_v6/task14a/`；代码 `analysis_v6/task14a_{collect,analysis}.py`。
- 预先固定项：portfolio、6 方向、$H_a$=1000FE、current-preserving practical 语义、κ=0.5/1.0 两个 pre-fixed candidate（不重选）、ProgressForecast PG3、CEC 暂停。

## 一、两个核心问题的回答

**RQ1（换挡成熟后是否仍有 next-action 空间）——是，且幅度与自然状态相当。**

| 量 | natural（Task 12.1） | post-handoff（本轮，max 口径） |
|---|---:|---:|
| switch-required（bbob / ma） | 0.258 / 0.265 | **0.230 / 0.210**（pooled 0.224；sum 口径 0.183/0.157） |
| DCM | 0.254–0.370 全双向 | **0.331–0.347 全双向** |
| 状态级残差 | Δ_dynamic 0.114 / 0.085 | **Δ_post = route+srcFE − statewise = +0.105 / +0.107** |
| route+srcFE 吃掉的比例 | — | 仅 ~15%（P0→P3 = 0.018） |

**RQ3（SHADE/L-SHADE 切换收益来自 solver identity 还是 reset）**——成熟换挡后状态上**切换收益本身消失**（G_switch pooled = −0.019/−0.105，continue 占优），且 **reset 是有害操作**（G_reset = −0.044/−0.089，重置成熟记忆/Archive 使性能变差）；SHADE-current 上 Δ_solver-specific = **+0.025 [+0.008, +0.041]**（离开 SHADE 时换 L-SHADE 优于原地重置）。结论：natural 状态的切换增益不可能是 reset 伪影；**不得在成熟换挡后域沿用自然域的方向性互补**。

$$
\boxed{\text{Action-space Verdict：A1 POST-HANDOFF ACTION SPACE ROBUST}}
$$
$$
\boxed{\text{Reset-Confound Verdict：RC1 SOLVER-SPECIFIC EFFECT}}
$$
$$
\boxed{\text{Task 14B GO}}
$$

## 二、§32 的 32 个问题逐条回答

| # | 问题 | 回答 |
|---|---|---|
| 1 | 6 方向 states 数 | 各 **630**（共 3780） |
| 2 | commitment 后 attained gap | source FE 2000/4000/6000 → mean log10 gap −0.726/−2.323/−3.612；route 上 lshade→shade 最好（−2.880）、shade→cso 最差（−1.737） |
| 3 | post-handoff switch-required | **0.2243**（pooled max）；bbob 0.230 / ma 0.210 |
| 4 | 与 natural 26% 相比 | 轻度下降（−0.03 左右），仍明显非零 |
| 5 | sum 保守口径下 | **0.176**（bbob 0.183 / ma 0.157） |
| 6 | 三对 DCM 是否仍双向 | **是**（max：0.331/0.336/0.347；方向概率 ≥0.15） |
| 7 | 哪些 current 有吸收风险 | 无（最大 lshade 0.880 < 0.90） |
| 8 | 最退化 route | cso→lshade（switch-required 0.102，P∈A_ND 0.898，未越阈） |
| 9 | post-handoff statewise 残差 | Δ_post = **+0.106**（pooled；bbob 0.105 / ma 0.107） |
| 10 | route+FE 吃掉多少 | 仅 ~0.018（P0→P3），≈15% |
| 11 | 是否仍有 Behavior 可用残差 | **是**（0.105，fb CI 远离 0，且 P1 退化为 Always Continue ⇒ 残差全在 state 级） |
| 12 | SHADE-current：reset vs switch | switch 更好：Δ = **+0.025 [+0.008, +0.041]**（reset-SHADE 差于 continue −0.044） |
| 13 | L-SHADE-current：reset vs switch | 无显著差：Δ = −0.016 [−0.034, +0.004]；两者均差于 continue（−0.089/−0.105） |
| 14 | solver-specific 是否超过 reset effect | **是**（SHADE-current Δ CI>0；reset 全线负增益） |
| 15 | 是否存在 schedule-reset 混杂 | **否**（缩减阶段由保留的 evaluations 驱动；逐行记录 preserved NP 与 schedule 字段） |
| 16 | L-SHADE reset 是否正确保留 NP 与全局缩减相位 | **是**（post NP 36/26/18 系列保持；reset 行 `population_size_preserved` 一致） |
| 17 | natural complementarity 是否 survives handoff | **是**（16e 对照表） |
| 18 | κ=0.5/1.0 在 post-handoff 上表现 | 风险面可迁移（harmful 0.166→0.097→0.074），性能增益不可迁移（pooled gain −0.029/−0.009/−0.003；bbob κ≥0.5 勉强 +0.003/+0.006） |
| 19 | 是否允许重选 κ | **NO**（工作单 §24 禁止；κ=0.5/1.0 pre-fixed 地位不变） |
| 20 | B_global 是否记录完成 | **是**（3780 条正式提取） |
| 21 | true segment Behavior 是否记录完成 | **是**（3780 条；segment 相对窗口；handoff 时 recorder 重建，segment_start=t） |
| 22 | 本轮是否训练 segment model | **NO**（只记录） |
| 23 | Action-space verdict | **A1 POST-HANDOFF ACTION SPACE ROBUST** |
| 24 | Reset-confound verdict | **RC1 SOLVER-SPECIFIC EFFECT** |
| 25 | 是否允许进入 Task 14B | **是（GO）** |
| 26 | seeds 6–10 confirmation status | **保持**（本轮未读取/未分析） |
| 27 | 正式 CEC 是否继续暂停 | **YES**（CEC2017 PAUSED / CEC2022 HELD OUT） |
| 28 | ProgressForecast 是否仍 PG3 NO-GO | **YES** |

（工作单问题中"unique practical winner rate"= 0.453、"unnecessary switch"等全量指标见 `post_handoff_practical_action_sets.parquet` 与行级表。）

## 三、实现与协议声明

1. transfer/clone/RNG 语义与 Task 12 完全同源（`initialize_transferred_optimizer_state` / `clone_optimizer_state` / `make_event_rng`）；
2. reset 经 `dataclasses.replace` 实现：保留 population/fitness/best/evaluations/schedule（initial/min NP、max_evaluations、reduction_max_fe），重置 memory_f/cr（回 0.5 初值）、archive（清空）、memory_index、RNG（fresh 语义事件），清除 partial-generation 状态（最多吸收一个未完成 generation 的 FE，账目不变）；
3. segment 窗口为 **segment 相对定义**（fe_total=1000；w02/w05/w10=20/50/100 FE）——segment 年龄恰为 1000 FE 时全局口径的 anchor 会越出段首，故 14B 的 B_segment 一律沿用本定义；
4. margin 确认使用 §25 允许的 frozen cross-group carrier（Task 13 正式协议在全开发域拟合一次），仅作诊断。

## 四、停止声明

按工作单 §35 链条全部执行：hygiene 门（H1）→ 6 方向冻结 → source checkpoints → A→B transfer → 1000FE commitment → checkpoint+bg/bs 保存 → next-action fork → outcome-blind repetitions → 必做 reset controls → 噪声/实用动作分析 → natural-vs-post 对照 → 吸收态审计 → reset 混杂分析 → A/RC verdicts → 14B readiness → **STOP**。未自动执行 Task 14B 训练、κ 选择、seeds 6–10 分析、validation 或 CEC。

## 五、下一步建议

执行 **Task 14B：B_global vs B_segment 增量测试**（A1 GO）。要点：M0=[current,FE,dwell] 在本域 OOF 重建（不得沿用 natural 域模型，16h 已证迁移失败）；M_G/M_S/M_GS 沿用 Task 13 载体与 leave-cv_group-out；主问题 $B^{segment} \stackrel{?}{>} B^{global}$；14B 通过后方可在 post-handoff 域重新校准 margin 策略（κ 网格重跑、fold-local 语义）。

### 下一步 prompt（可直接复制开新对话）

```
你正在继续 GitHub 项目 Decision-before-Feature（目录 behavior_with_ela/）。
先读 AGENTS.md、analysis_v6/task14a/ 总报告与 16a-16i、analysis_v5/task13 与
task13_1 总报告。当前状态：Task 14A verdict A1 + RC1 → Task 14B GO；
post-handoff 3780 states（6 方向）、B_global/B_segment 特征与 1000-FE
真实标签已就绪（results/analysis_v6/task14a/）；16h 证明 natural 域冻结
策略在 post-handoff 域增益≈0，M0 必须本域重建。

本轮任务：Task 14B —— B_segment 增量测试（零新增 objective FE）。
1. 数据：post_handoff_behavior.parquet（kind=global/segment 各 3780）+
   post_handoff_action_outcomes_1000.parquet（replicate 0 为主标签）+
   post_handoff_states.parquet 元数据；
2. 特征组：M0=[current,FE,dwell]（dwell 恒 1000，仅保留字段）、
   M_G=[M0+B_global 28]、M_S=[M0+B_segment 28]、M_GS=并集；
   载体：正式 WeightedMedianImputer→StandardScaler→RF(200,8,sqrt,fixed seed)
   + Ridge 对照；leave-cv_group-out；BBOB/MA 分列；
3. 主比较：M_S vs M_G 与 M_S vs M0 的 fb policy loss 差（真实 1000-FE
   outcome），paired cv_group bootstrap 5000；辅助：vs continue、harmful、
   switch precision/recall（post-handoff 噪声 max 口径）；
4. 判定：M_S 或 M_GS 须在两 suite 同时优于 M_G 且 CI>0 才算
   SEGMENT INCREMENT CONFIRMED；否则 NO SEGMENT INCREMENT；
5. 组内置换对照（segment 与 global 各 100 次，RF 载体）；
6. 输出 analysis_v6/task14b/ 报告组 + 轻量表 + 资源账本；
7. 禁止：新 FE、κ 重选、seeds 6-10 分析、validation、CEC、ProgressForecast、
   segment 模型之外的任何新模型。
```
