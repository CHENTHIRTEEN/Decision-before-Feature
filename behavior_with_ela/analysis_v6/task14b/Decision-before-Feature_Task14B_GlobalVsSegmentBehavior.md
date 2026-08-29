# Decision-before-Feature：Task 14B Global vs Segment Behavior 总报告

- 日期：2026-08-29；HEAD 基线 = Task 14A（post-handoff 3780 states 已采集）。
- 成本：**new objective FE = 0**（全部分析复用 Task 14A 采集；账本 `task14b_resource_ledger.parquet`）。
- 报告组：`analysis_v6/task14b/17a–17i`；行级表 `results/analysis_v6/task14b/`；代码 `analysis_v6/task14b/task14b_analysis.py`。

## 一、一句话结论

$$
\boxed{
\text{在真实 post-handoff 成熟状态上，Behavior（global、segment 或并集）}
}
$$
$$
\boxed{
\text{不能超过简单的 route+sourceFE 上下文，segment 相对 global 无增量——Repeated Behavior learning NO-GO。}
}
$$

全部三个 Behavior 模型在 grouped-OOF 真实 policy 评价中**劣于** M0（bbob 上显著负增量：MG −0.029 [−0.047,−0.011]、MS −0.041 [−0.071,−0.013]、MGS −0.042 [−0.067,−0.021]），且显著劣于 empirical route+sourceFE lookup（MGS −0.043 [−0.073,−0.021]）；within-route LOSO 下 Δ 为负（−0.044/−0.033）且 100 次 permutation 显示观测值落在 null 中心（p=0.13–0.34）。

## 二、三层 Verdict

$$
\boxed{\text{Verdict A：A3 NO-GO（Post-Handoff Behavior Increment）}}
$$
MGS 不超过 lookup/M0；bbob 显著反向，ma 无增量——"无增量且方向冲突"。

$$
\boxed{\text{Verdict B：B4 BEHAVIOR NO-GO（Segment Increment）}}
$$
MG/MS/MGS 均不超过 M0/lookup；segment 增量 MGS−MG ≈ 0（bbob −0.013 CI 含 0 / ma +0.005 CI 含 0）；within-route 置换 p=0.13–0.34 不支持。**注意（§44）**：B3/B4 的语义边界——若未来 M_G>M_0 成立而 MGS≈MG，只能说"在固定 1000-FE segment-age 设定下 Global Behavior 已充分"，不能说"Segment Behavior 一般无用"；本轮连 M_G>M_0 也不成立，故为 B4。

$$
\boxed{\text{Verdict C：C2 TRADEOFF（Risk-Control）}}
$$
post-handoff 域内重训的 MGS margin 策略：κ 阈值仍单调降低 harmful（0.170→0.109→0.046）与 switch（0.637→0.199），但平均性能不保留（gain 全负，至多 bbob κ=1.0 ≈ 打平）——风险下降、性能受损的权衡；margin 排序意义弱（Spearman 0.12–0.14，顶三分位增益未转正）。

## 三、按工作单 §43 逐条回答（36 问）

| # | 问题 | 回答 |
|---|---|---|
| 1 | 3780 states 全部完成 bg/bs/label join？ | **是**（1:1，无丢/重） |
| 2 | bg/bs 无 leakage？ | **是**（segment 窗口全部位于 [t,t+1000] 段内；global 用 [0,t+1000]；无 outcome/未来信息） |
| 3 | M0 OOF loss | −1.9835 / −5.0431（pooled −3.6524） |
| 4 | lookup loss | **−1.8980 / −5.0007**（post-handoff 域最强 simple baseline） |
| 5 | MG OOF loss | −1.9543 / −5.0190 |
| 6 | MS OOF loss | −1.9421 / −5.0058 |
| 7 | MGS OOF loss | −1.9413 / −5.0239 |
| 8 | MGS vs lookup gain/CI | −0.0433 [−0.073,−0.021] / −0.0232 [−0.058,+0.021] |
| 9 | MG vs M0 gain/CI | −0.0292 [−0.047,−0.011] / −0.0241 [−0.058,+0.023] |
| 10 | MS vs M0 gain/CI | −0.0414 [−0.071,−0.013] / −0.0373 [−0.081,+0.019] |
| 11 | MGS vs MG（segment increment）/CI | −0.0129 [−0.035,+0.002] / +0.0049 [−0.009,+0.018] |
| 12 | Segment 是否提供 Global 之外增量 | **否** |
| 13 | BBOB/MA 一致？ | 一致（均无正增量；bbob 证据更强） |
| 14 | 6 route 方向一致？ | **是**（全方向 MG/MS/MGS 无正增益） |
| 15 | source FE 2000/4000/6000 增量 | MGS gain −0.024/−0.044/−0.041——**任何阶段都不成立** |
| 16 | within-route WG/WS/WGS 增量 | Δ_global −0.044/−0.033；Δ_segment-only −0.044/−0.033；Δ_segment −0.000/−0.001 |
| 17 | Segment permutation p | 0.337 / 0.228（观测≈null 中心） |
| 18 | Global permutation p | 0.267 / 0.129 |
| 19 | bg→bs / bs→bg reconstructability | OOF Ridge R² 剧烈为负——诊断在本域退化为 UNINFORMATIVE（长尾×组间漂移），不做推断（17f） |
| 20 | MG vs MS 谁强 | MS 点估计略优（两者皆劣于 M0） |
| 21 | MGS 是否优于二者 | **否** |
| 22 | MGS margin 有排序意义？ | 弱（Spearman 0.12–0.14；顶三分位增益未转正）——不足以支撑正增益策略 |
| 23 | κ=0.5 降 harmful/switch？ | **是**（harmful 0.170/0.157→0.109/0.119；switch 0.637/0.603→0.363/0.367） |
| 24 | κ=1.0 降 harmful/switch？ | **是**（harmful 0.046/0.064；switch 0.199/0.207） |
| 25 | 两 κ 保留 performance？ | **否**（gain 全负，bbob κ=1.0 至多打平 +0.002） |
| 26 | 是否允许重选 κ？ | **NO**（κ=0.5/1.0 pre-fixed 维持） |
| 27 | Post-Handoff Behavior verdict | **A3 NO-GO** |
| 28 | Segment verdict | **B4 BEHAVIOR NO-GO** |
| 29 | Risk verdict | **C2 TRADEOFF** |
| 30 | 是否允许 seeds 6–10 confirmation | **否——无确认对象**（post-handoff selector 已 NO-GO；seeds 6–10 继续封存备用） |
| 31 | 是否允许 closed-loop repeated DAS | **否**（Gate A 未过） |
| 32 | ProgressForecast 仍 PG3 NO-GO？ | **YES** |
| 33 | CEC2017 仍暂停？ | **YES** |
| 34 | CEC2022 仍 held-out？ | **YES** |
| 35 | 下一阶段准确是什么 | **无自动下一阶段**——项目定位回落（见 §五）；任何新假设须先经用户批准并另行预注册 |
| 36 | （科学边界 §44） | 见下 |

## 四、科学边界（§44 逐条落实）

1. 本轮不能写"Segment Behavior proves recent history causes switching decisions"——本轮连预测增量都不存在；
2. 由于 M_G 劣于 M0（而非 ≈），也不能写"Global Behavior is sufficient"——正确的写法是：**在成熟 post-handoff 状态（segment age 恒 1000FE）上，被测 Behavior 表征与载体不能捕获 route+sourceFE 之外的 next-action value，且使用它们会因过度切换而损害性能**；
3. $M_G, M_S, M_{GS} \le M_0$ 全部成立 ⇒

$$
\boxed{\text{Repeated Behavior learning NO-GO}}
$$

项目定位回落到 **trajectory-conditioned one-step algorithm selection**（Task 13 的 natural 域 A2+B1 结论保持域限定有效）。

## 五、成本账本与停止声明

| 项 | 值 |
|---|---:|
| new objective FE | **0** |
| wall time | 主分析+200 次 permutation ≈3.3 h（8 进程） |
| 资源账本 | `results/analysis_v6/task14b/task14b_resource_ledger.parquet` |

停止条件链全部执行：数据集构建 → 一致性/泄漏审计 → 强基线 → grouped-OOF → 真实 policy 评价 → within-route LOSO → P1/P2 100 次 permutation → 冗余诊断（标注 UNINFORMATIVE）→ fold-local 噪声 → 固定 κ={0,0.5,1.0} margin → route/phase/suite 分层 → A/B/C verdicts → **STOP**。未自动执行 seeds 6–10、任何新 FE、closed-loop、CEC、validation、portfolio 变更或 ProgressForecast。

## 六、对论文的含义

1. **保留主张**：$P_{balanced}$ 的静态+上下文互补（Task 12）；natural 轨迹上的 Behavior one-step 增量（Task 13 A2+B1，域限定、置换检验 p≈0.01）；margin 风险控制语义（natural 域 R1）；
2. **新增负结果主张**（有完整对照链）：换挡成熟后的 next-action 空间仍非退化（14A A1），但 **route+sourceFE 之外的残差不被 Behavior 特征捕获，且 Behavior 策略因过度切换而劣于 continue/lookup**（14B A3+B4+C2）；reset 混杂被排除（14A RC1：reset 有害）；
3. **方法学资产**：fold-local 校准语义（13.1-H）、post-handoff 噪声 route 条件化标定（14A 16c）、within-route permutation 协议（13.1/14B）。
