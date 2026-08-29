# Decision-before-Feature：Task 13.1 Behavior 策略鲁棒性与风险控制复核 总报告

- 日期：2026-08-29；HEAD = `c55563f`（Task 13 提交）。
- 定位：Task 13 与 post-handoff sequential confirmation 之间的**零新增 FE** 方法复核（new objective FE = 0）；不训练新模型、不调参、不新增阈值候选、不引入 uncertainty gate、不复活 ProgressForecast。
- 报告组：`analysis_v5/task13_1/15a–15h`；轻量表 `analysis_v5/task13_1/`；行级表 `results/analysis_v5/task13_1/`；代码 `analysis_v5/task13_1/{task13_1_analysis, task13_1_perm}.py`。

## 一、一句话结论

$$
\boxed{
\text{Behavior 信号经受住强基线与置换检验（S2 + W1），}
}
$$
$$
\boxed{
\text{且一个预注册的 current-preserving margin 阈值（}\kappa{=}0.5\text{）同时把过度切换与有害切换减半而不损失平均性能（R1）。}
}
$$

Task 13 的"Beta 有信息"结论升级为：**信息可转化为风险合理的 switching policy 形态**——但 κ 的最终确定仍需 Task 14A 确认数据。

## 二、三层 Verdict

$$
\boxed{\text{Verdict 1：S2 CONDITIONAL}}\quad
\boxed{\text{Verdict 2：R1 RISK-CONTROL FEASIBLE}}\quad
\boxed{\text{Verdict 3：W1 ROBUST}}
$$

$$
\boxed{\text{Task 14A GO（S2∈\{S1,S2\} 且 W1∈\{W1,W2\}）}}
$$

## 三、强基线阶梯与配对比较（15b）

| 基线（fb loss） | BBOB | MA |
|---|---:|---:|
| B0 Always Continue | **−1.6055** | −4.5282 |
| B1 Empirical current+FE lookup | −1.5856 | −4.5298 |
| B2 RF-M0 | −1.5634 | −4.5219 |
| B3 M1 Behavior-only | −1.6054 | −4.5732 |
| B4 Raw M2 | −1.6107 | −4.5731 |

配对比较（5000 draws）：Continue−M2 = +0.0052 [−0.098, +0.146] / +0.0450 [−0.0067, +0.104]；**Lookup−M2 = +0.0251 [−0.073, +0.159] / +0.0433 [+0.0042, +0.085]**；M0−M2 = +0.047/+0.051；Lookup−M1 = +0.020/+0.043。

**分解**：Task 13 的 Δ_B=+0.047/+0.051 中，BBOB 有 **0.022（约 47%）** 来自 RF-M0 弱于 empirical lookup；MA 主体保留（lookup−M2=+0.043，CI>0）。

## 四、Margin 校准与 κ 网格（15c/15d）

- 校准：pooled decile 上 harmful 概率从 0.56（bin0）单调降至 0.11–0.18（bin8–9），realized gain 自 bin6–7 转正；两 suite 同型（bbob 顶箱小样本回落已记录）——**margin 有真实排序意义**；
- κ 网格（max scale，全部点报告）：

| κ | switch rate (bbob/ma) | harmful rate | fb loss | gain vs Continue | gain vs Lookup |
|---:|---|---:|---|---:|---:|
| 0.0（raw） | 0.642 / 0.591 | 0.136 / 0.137 | −1.6107 / −4.5731 | +0.005 / +0.045 | +0.025 / +0.043 |
| 0.5 | 0.416 / 0.398 | 0.071 / 0.078 | −1.6271 / −4.5741 | +0.022 / +0.046 | +0.041 / +0.044 |
| 1.0 | 0.322 / 0.261 | 0.050 / 0.043 | −1.6203 / −4.5594 | +0.015 / +0.031 | +0.035 / +0.030 |
| 1.5 | 0.256 / 0.191 | 0.037 / 0.032 | −1.6120 / −4.5586 | +0.007 / +0.031 | +0.026 / +0.029 |
| 2.0 | 0.195 / 0.119 | 0.030 / 0.020 | −1.6083 / −4.5476 | +0.003 / +0.019 | +0.023 / +0.018 |

- sum-scale 同型（更保守）；κ=0.5 相对 raw M2 的损失差 CI：bbob +0.016 [−0.037, +0.065]、MA +0.001 [−0.015, +0.015]——**无显著劣化**；
- **Pareto**：raw M2（κ=0）在两 suite 均不在前沿上，被 κ=0.5 支配；harmful mass 同步下降（0.087→0.045 / 0.047→0.034）；
- M1 vs M2 的 margin–risk 曲线形态一致（M2 全 κ 略优）——margin 门的价值主要来自 Behavior 信号本身；显式叠加 current/FE 在该载体下只提供边际改进（15d/14f 的 M1≈M2 按本句口径解读）。

## 五、Within-Problem 100 次 permutation（15e）

| suite | 观测 Δ_within | null mean | null q97.5 | empirical p |
|---|---:|---:|---:|---:|
| BBOB | +0.01877 | +0.00107 | +0.00928 | **0.0099** |
| MA | +0.01609 | +0.00007 | +0.00908 | **0.0099** |

观测超过全部 100 个 null（2×q97.5）；Task 13 B1 在正式协议下成立。措辞边界：**"Behavior contains within-problem state-discriminative signal under the fixed RF carrier"**——不声称"每个 problem 内已学到普遍可泛化的 state policy"。实现修正声明见 15e §4（首轮 slim W0 泄漏已修复并全量重跑，Task 13 artifacts 不受影响）。

## 六、Problem-ID 诊断降级（15f）

旧 `[problem,current,FE]` one-hot 诊断在 leave-cv_group-out 下，held-out problem 的 dummy 列在训练内恒为 0——**不能解释为"模型知道测试 problem identity"**。旧结果标记 `INVALID AS KNOWN-PROBLEM DIAGNOSTIC / LEGACY ONLY`（artifact 保留不删）；正式替代证据 = within-problem LOSO + 本轮 100 次 permutation。

## 七、Transition 风险分层（15g）

harmful 集中于 shade-current 两个方向（shade→lshade 0.30/0.51、shade→cso 0.29/0.31）与 bbob lshade→shade（0.32）；高价值/高精度方向为 →cso（lshade→cso 增益 +0.44/+0.28、precision 0.45/0.53）与 cso→shade（+0.11/+0.04、harmful 0.04/0.00）。**6 方向全部保留进入 Task 14A 预注册**；分层仅作为 14A 分方向报告模板。

## 八、§32 的 30 个问题逐条回答

| # | 问题 | 回答 |
|---|---|---|
| 1 | Task 13 OOF 逐 state 复现？ | **是**（M2 预测 diff=0.0、selected/realized 1890/1890 一致） |
| 2 | lookup 复现 Task 12.1？ | **是**（全精度 diff=0.0：−1.585625/−4.529830） |
| 3 | Always Continue fb loss | −1.6055 / −4.5282（强基线：bbob 上优于 RF-M0） |
| 4 | Raw M2 vs Continue | +0.0052 [−0.098, +0.146] / +0.0450 [−0.0067, +0.104] |
| 5 | Raw M2 vs lookup | +0.0251 [−0.073, +0.159] / +0.0433 [+0.0042, +0.085] |
| 6 | M1 vs lookup | +0.0198 [−0.071, +0.146] / +0.0434 [+0.0023, +0.089] |
| 7 | +0.047/+0.051 中 RF-M0 偏弱的成分 | BBOB 0.022/0.047（≈47%）；MA 0.008/0.051（≈16%） |
| 8 | Verdict 1 | **S2 CONDITIONAL** |
| 9 | margin 与 realized switch gain 单调？ | 基本是：bin0–3 增益为负、bin6–8 转正（+0.08～+0.12）；bbob 顶箱小样本回落 |
| 10 | margin 越大 harmful 越低？ | 是：harmful 概率 0.56→0.11–0.18（bbob 0.58→0.10，MA 0.52→0.15） |
| 11 | κ=0/0.5/1.0/1.5/2.0 switch rate | 0.642/0.416/0.322/0.256/0.195（bbob）；0.591/0.398/0.261/0.191/0.119（MA） |
| 12 | harmful rate | 0.136/0.071/0.050/0.037/0.030（bbob）；0.137/0.078/0.043/0.032/0.020（MA） |
| 13 | realized fb loss | 见 §四表（κ=0.5 bbob 最佳 −1.6271；MA −4.5741） |
| 14 | gain vs Continue | κ=0.5：+0.022/+0.046（κ=2.0 收缩至 +0.003/+0.019） |
| 15 | gain vs lookup | κ=0.5：+0.041/+0.044（不低于 raw M2 的 +0.025/+0.043） |
| 16 | 是否存在降险不损性能的预注册 policy | **是**：κ=0.5 与 κ=1.0（max）——双 suite harmful/switch 双降、增益保持、对 raw M2 无显著损失 |
| 17 | Risk-Control verdict | **R1 RISK-CONTROL FEASIBLE** |
| 18 | max vs sum 结论一致？ | 一致（同型曲线，sum 更保守；κ=0.5 sum：switch 0.345/0.296、harmful 0.053/0.050） |
| 19 | transition harmful 集中？ | 是：shade→lshade（0.30/0.51）、shade→cso（0.29/0.31）、bbob lshade→shade（0.32）；→cso 方向最优质；6 方向不删 |
| 20 | 100 次 permutation empirical p | **0.0099 / 0.0099**（=1/101，两 suite 均超过全部 null） |
| 21 | B1 在 100 次下成立？ | **是（W1 ROBUST）** |
| 22 | 旧 Problem-ID 为何不能那样解释 | held-out problem 的 dummy 列训练内恒 0，模型从未见过该身份（15f） |
| 23 | 正式替代证据 | within-problem LOSO + 100 次 permutation（15e） |
| 24 | M1/M2 margin-risk 曲线是否基本相同 | **是**（同型，M2 全 κ 略优）——margin 门价值主要来自 Behavior 本身 |
| 25 | 是否允许进入 Task 14A | **是（GO）** |
| 26 | 是否建议 seeds 6–10 | **是**（confirmation seeds，train-domain 不变，避开 seeds1–5 迭代污染） |
| 27 | reset controls 是否必须执行 | **是**（SHADE-current / L-SHADE-current population-preserving reset，与 handoff 分支同轮） |
| 28 | Task 14A 是否允许直接训练 segment Behavior | **NO**（14A 只做 action-space / reset 审计；segment 留给 14B 且需 action space non-degenerate 前提） |
| 29 | ProgressForecast 是否仍 PG3 NO-GO | **YES**（margin gate ≠ progress gate） |
| 30 | 正式 CEC 是否继续暂停 | **YES**（CEC2017 PAUSED；CEC2022 HELD OUT） |

## 九、成本账本

| 项 | 值 |
|---|---:|
| new objective FE | **0** |
| 分析 wall time | 主分析 ≈4 min；permutation 100 次 ≈35 min（8 进程） |
| 产物 | 11 个核心 parquet + 行级表（`results/analysis_v5/task13_1/`）+ 资源账本 `task13_1_resource_ledger.parquet` |

## 十、科学边界与停止声明

**"Behavior has predictive value" 不自动等于 "deployment-ready"**——本轮补上的正是后者所需的三块证据：强基线对照（S2）、风险控制可行性（R1：κ=0.5/1.0 max 同时降 switch/harmful 而不显著损失性能）、置换稳健性（W1）。仍缺的是未见过 seeds 的确认：所有数字都来自 seeds 1–5 的开发域。

按工作单 §34 链条全部执行完毕：复现核查（双双 diff=0）→ 强基线阶梯 → 配对比较 → pooled 尺度 → margin 网格 → 校准 → Pareto → 100 次 permutation → Problem-ID 降级 → transition 分层 → 三层 verdict → **STOP**。未自动执行 Task 14A、post-handoff 采集、reset-control FE、segment Behavior、ProgressForecast、validation 或 CEC。

## 十一、下一步建议

执行 **Task 14A：Post-Handoff Sequential Action-Space + Reset Audit**（GO 已满足）：seeds 6–10、6 方向、1000-FE commitment 后 fork、reset controls 同轮、零 segment 训练；分方向报告并以 15g 为模板。可复制下一段 prompt 开新对话。

### 下一步 prompt（可直接复制）

```
你正在继续 GitHub 项目 Decision-before-Feature（目录 behavior_with_ela/）。
先读 AGENTS.md、analysis_v5/task13 与 task13_1 两份总报告及 15a-15h。
当前状态：Task13 verdict A2+B1；Task13.1 verdict S2+R1+W1（margin κ=0.5 max
支配 raw M2：harmful 减半、switch 0.64→0.42、损失不劣化；within-problem
100-perm p=0.0099 双 suite）→ Task 14A GO。

本轮任务：Post-Handoff Sequential Action-Space + Reset Audit。
1. seeds 6-10（未参与 Task 12/13 开发；train-domain 不变，不碰 validation/CEC）；
2. Stage-2 同一 problem 集，6 方向（shade↔lshade, shade↔cso, lshade↔cso）
   population-transfer handoff，先 1000-FE commitment 形成 post-handoff
   checkpoint，再 fork {continue, switch other two}（各 1000-FE 分支）；
3. 同轮预注册执行 reset controls：SHADE-current / L-SHADE-current 的
   population-preserving reset 分支（分离 solver identity vs memory/restart）；
4. 正式 recorder 记录 handoff 后 checkpoint 的 B_global 与真实 segment 语义
   B_segment（segment_start=handoff 点）；不得用 global 复制冒充；
5. 分析：post-handoff action space 非退化性（current-preserving、pairwise
   max-delta 口径 + quad/sum 敏感性）、6 方向分列的 practical switch 结构、
   reset vs native continue 的损失差；B_global/B_segment 训练一律禁止（14B 才做）；
6. 资源账本单列分支 FE；输出 analysis_v5/task14/ 报告组与轻量表；
7. 禁止：ProgressForecast（PG3）、portfolio 重选、调参、CEC、validation、
   删除方向或阴性结果、预先确定任何部署阈值。
```
