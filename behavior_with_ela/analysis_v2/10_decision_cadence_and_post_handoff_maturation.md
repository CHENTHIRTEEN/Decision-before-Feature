# 10 · Decision Cadence × Post-Handoff Maturation 诊断（在线调度开发性实验）

- 日期：2026-08-29
- 性质：**scheduler development diagnostic only**。仅使用三个 CEC2017 开发函数（F1/F10/F29 × 10 seeds × 3 初始算法 = 90 runs/策略），用于回答在线切换调度问题；不得当作正式 benchmark 结论。
- 不变量：两个在线载体（`behavior_action_loss_regression_v2` 与 `action_gain_classifier`）的模型、28 维 Behavior、阈值、portfolio 完全不变；未训练任何模型、未新增 BBOB/MA-BBOB 数据、未运行全量 CEC。
- 实现：`online.py::run_one_switch_policy` 将三个时钟拆分——Observation Clock（每次原生更新记录轨迹/Behavior/全局 FE）、Decision Check Clock（沿用既有 budget milestones + event states）、Switch Eligibility Clock（新增 `minimum_dwell_fe` 与 classifier 专用 `hysteresis_margin`，预测照常计算，但 eligibility 不满足时必须 continue）。同时新增 segment 记录与 gap checkpoint（按"首个 $\ge$ 检查点的 40-FE 更新边界"记录，两条分支使用同一规则，等 FE 可比）。
- 产物：`results/online/cec2017_quick3_scheduler_ablation/<carrier>__<policy>/`（outcomes/opportunities/segments/resources）；聚合表在 `results/analysis_v2/task10/`；历史目录 `cec2017_quick3*` 与 09/09b 报告未改动。

## 1. Motivation

09b 的 per-opportunity 测试发现：classifier 载体在无 dwell、无滞回、无切换上限的规则下出现多次切换，部分第二次切换距第一次仅 80–200 FE（population 40 时约 2–5 次原生更新），且切换链出现 shade↔cmaes 振荡。population transfer 会重初始化新算法内部状态，刚切换后的短时间属于 adaptation transient。因此不能把 repeated switching 的改善直接解释为"模型发现了阶段转换"。

## 2. Current pathology（P1 无约束逐机会策略的病理）

| 指标 | classifier P1（raw） |
|---|---:|
| 相邻切换间隔 min / p25 / median / p75 / p90 | 80 / 120 / 200 / 200 / 200 FE |
| 间隔 <200 / <500 / <1000 占比 | 0.356 / **0.978** / 0.978 |
| 反转事件（A→B→A，间隔<1000） | 24 次；15/90 runs（0.167） |
| 反转算法对 | **全部为 shade↔cmaes**（shade→cmaes 18 次 median 160 FE，min 80；cmaes→shade 7 次 median 200 FE，min 120） |
| segment 生存期 p10 / median | **200 / 2000 FE** |
| 平均切换次数 | 1.20 |

即：约 2–5 个原生更新就被再次切走的段占 segment 总量的十分之一，且振荡只发生在 shade 与 cmaes 之间——"算法阶段转换"与"chattering"在该策略下混杂。

## 3. Experimental protocol

| 策略 | 规则 | 载体 |
|---|---|---|
| P0 first_trigger | 每 run 至多一次切换（既有 reference） | 双载体 |
| P1 raw_per_opportunity | 逐机会触发，无 dwell/滞回/上限 | 双载体 |
| P2 dwell_500 | minimum_dwell_FE = 500 | 双载体 |
| P3 dwell_1000 | minimum_dwell_FE = 1000 | 双载体 |
| P4 dwell_1500 | minimum_dwell_FE = 1500 | 双载体 |
| P5 dwell_1000 + hysteresis | P3 + top1−top2 概率边际 > 0.10（复用 `repeated_das` 定义） | classifier 专用 |

P1–P4 均不设切换上限。v2 回归载体的 score 是预测损失优势（非概率），按工作单不给它硬造 0.10 滞回；实测 v2 在 per-opportunity 下从不二次切换，**hysteresis is not applicable / not activated** under the current observed policy behavior。

## 4. Dwell ablation（函数平衡）

| 载体 | 策略 | mean terminal log10 gap | gain vs continue | 平均切换次数 | 首次/二次切换 FE (median) | success |
|---|---|---:|---:|---:|---|---:|
| v2 | P0–P4（全部相同） | −1.6061 | 1.8679 | 0.667 | 2000 / — | 0.200 |
| clf | P0 | −1.1581 | 1.4198 | 0.700 | 2000 / — | 0.200 |
| clf | P1 | −1.5788 | 1.8405 | 1.200 | 2000 / 2200 | 0.200 |
| clf | P2 | −1.6440 | 1.9058 | 0.911 | 2000 / 2600 | 0.200 |
| clf | **P3** | **−1.6545** | **1.9163** | 0.911 | 2000 / 3000 | **0.211** |
| clf | P4 | −1.4011 | 1.6629 | 0.822 | 2000 / 3800 | 0.211 |
| clf | P5 | −1.5541 | 1.8159 | 0.889 | 2000 / 3400 | 0.211 |

- **v2 回归载体对调度完全无感**：P0–P4 逐 run 相同（从不二次切换），证实 09b 结论。
- classifier 载体：dwell 不仅没有损失 per-opportunity 的收益，反而进一步改善——P3 比 P1 好 **+0.076**（1.9163 vs 1.8405），比 P0 好 **+0.50**。
- **dwell_1500 过度保守**（比 P3 差 0.25），丢失的是有价值的中后期切换（二次切换 median 被 3800 FE 推迟）。
- 逐函数（P1→P3）：F1 −8.546→**−8.766**（+0.22，success 0.60→0.633）；F29 3.305→**3.284**（+0.021）；F10 0.504→0.519（−0.014，小幅退化）。逐 prefix：收益集中在 pso 起始（−0.397→**−0.615**）；shade/cmaes 起始几乎不变。

## 5. Hysteresis ablation（classifier only）

P5（dwell_1000 + 滞回 0.10）terminal −1.5541，**比纯 P3 差 0.10**：滞回把切换次数从 0.911 压到 0.889 的同时拦截了部分有效切换（F1 −8.57 vs P3 −8.77）。滞回能把 chattering 归零（P2 残留 0.033 的 run 率），但 P3 已经归零，滞回的额外收益为负。

## 6. Switch interval analysis

| 载体=clf | P1 | P2 | P3 | P4 | P5 |
|---|---:|---:|---:|---:|---:|
| 间隔 min / median / p90 | 80 / 200 / 200 | 520 / 600 / 1400 | 1000 / 1000 / 1440 | 1520 / 1800 / 1800 | 1000 / 1100 / 2136 |
| 间隔<200 / <500 / <1000 | .356 / .978 / .978 | 0 / 0 / .842 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

P1 的二次切换几乎全部发生在 80–200 FE（= 2–5 个原生更新）；dwell=500 使最小间隔升到 520 FE；dwell=1000 使全部间隔 ≥1000 FE。

## 7. Chattering analysis

- P1：反转事件 25 次（shade↔cmaes），run 率 0.167（H=1000），平均反转间隔 234 FE。
- P2：5 次（run 率 0.033）；**P3/P4/P5：0 次**。
- 即 dwell_1000 完全消除 CMAES↔SHADE 振荡；滞回也能消除但对 terminal 有代价（§5）。

## 8. Segment lifetime analysis（classifier）

| 策略 | segments | 生存期 median / p10 / p90 (FE) | 段内 log10 改善 median / mean |
|---|---:|---|---:|
| P1 | 198 | 2000 / **200** / 10000 | 3.744 / 5.147 |
| P2 | 172 | 6160 / 1440 / 10000 | 4.097 / 5.959 |
| P3 | 172 | 5620 / 1620 / 10000 | 4.097 / 5.965 |
| P4 | 164 | 6200 / 2000 / 10000 | 4.136 / 6.117 |

dwell 消灭了 200-FE 级的短命段；存续段的平均进展更高。v2 载体在所有策略下都是同样的 150 个段（median 8000 FE）。

## 9. Post-handoff maturation：等 FE 强制驻留分支对比

方法：从 classifier P1 中取 44 个"二次切换间隔 <1000 FE"的案例（其中间隔 80/120/200 FE 的 41 个，280–400 FE 的 3 个）。对每个案例：raw 重放（与记录逐 FE 一致，终端 gap 差异 <1e-12 校验）与 dwell-h 分支（同 checkpoint 规则）在 $t_0+h$ 处取 $\ell$，$M_h=\ell_{raw}(t_0+h)-\ell_{forced\text{-}B}(t_0+h)$，正值表示"强制让刚接管的算法成熟"更好；另记录 resume 后的 terminal（即 dwell-h 策略的自然终点）。

| h | cases | mean $M_h$ | median $M_h$ | $M_h>0$ 占比 | resume terminal 相对 raw |
|---:|---:|---:|---:|---:|---:|
| 500 | 44 | **−0.070** | −0.006 | 0.409 | **+0.081** |
| 1000 | 44 | **−0.183** | −0.084 | 0.227 | **+0.127** |
| 1500 | 44 | −0.310 | −0.101 | 0.227 | −0.385 |

按被强制算法拆分：forced **shade** 的 $M_{1000}$ = **−0.267**（仅 16.7% 为正）——SHADE post-handoff 强制成熟明确更差；forced **cmaes** 的 $M_{1000}$ = −0.001（≈中性）。按间隔拆分：80–220 FE 的快速回切案例 $M_h$ 全部为负（−0.076/−0.205/−0.344）。

**解读**：等 FE 下，80–200 FE 就被切走的算法并不存在"再给 500–1000 FE 就会反超"的延迟成熟效应——快速再评估本身是局部合理甚至更优的。但 resume 后 terminal 更好（+0.08/+0.13）说明 dwell 的收益不在"成熟"，而在**改变后续决策路径、消除振荡**（与 §4–§8 一致）。

## 10. Post-switch progress curves

对每个真实切换 A→B（budget 允许时）记录 $\ell_t$ 与 $\ell_{t+h}$，$P_h=\ell_t-\ell_{t+h}$：

| 载体→目标 | switches | mean $P_{200}$ | mean $P_{500}$ | mean $P_{1000}$ | $P_{500}>P_{200}$ 占比 |
|---|---:|---:|---:|---:|---:|
| clf→cmaes | 78 | 0.191 | 0.555 | 1.120 | 0.859 |
| clf→shade | 30 | 0.080 | 0.365 | 0.905 | 0.833 |
| v2→cmaes | 60 | 0.186 | 0.560 | 1.218 | 0.817 |

切换后的进展确实呈"前 200 FE 慢 → 500–1000 FE 明显加快"的形状（约 85% 的切换 $P_{500}>P_{200}$）——adaptation transient 在绝对意义上存在；但 §9 表明这种延迟进展不足以让"强制驻留"在等 FE 比较中胜过快速再评估。两个事实并存：**transient 存在，但不是性能解释的主导项**。

## 11. Transfer/restart confound

对 5 个 A→B→A 案例（回放与记录逐 FE 校验一致）：同一 $t_1$ 检查点分叉——(a) 按 raw 实际执行的 population transfer 重新初始化 A；(b) 恢复离开 A 时保存的原生 state（`clone_optimizer_state`，含 RNG）继续。restoration gain = $\ell_{transfer}-\ell_{restore}$：**−1.91 / +0.43 / +0.31 / −0.03 / +0.08**。方向不一致、幅度可观：reset/restart 效应真实存在但个案依赖，不能把 repeated switching 的收益单一归因于隐式重启扰动，也不能排除其在个别 run 中的作用。该 confound 在 80–200 FE 级别的切换上不可忽略，进入正式协议前必须用更大样本量化（本轮仅 5 例诊断，不引入主策略）。

## 12. Implication for ProgressForecast（协议映射，不做训练）

三个参数必须分开预指定：
- **Check interval** $\Delta_{check}$：沿用既有 200-FE milestone 决策机会（本轮事件点与 milestone 均作为 decision check，不改采样协议）。
- **Minimum dwell** $H_{dwell}$：候选 500/1000/1500；本轮证据支持 **1000 FE** 为默认（500 为可接受的下限），1500 过保守。
- **Progress horizon** $H_g$：500/1000 FE。$P_h$ 曲线（§10）显示切换后 1000 FE 内进展呈加速形状，与 $H_{dwell}=H_g=1000$ FE 的组合一致；但是否采用应等 ProgressForecast 的 OOF 证据（Behavior 是否优于 prefix+time）决定，本轮不据此直接固定。

## 13. Final scheduling recommendation 与决策规则

**结论：C. DWELL-1000 PREFERRED（DEVELOPMENTAL EVIDENCE ONLY）。**

- 相对 A（NO-DWELL INVALID）：P1 的收益在加入 dwell 后不仅保持而且增强（1.841→1.916），chattering 归零——no-dwell 的收益不是假的，但其 80–200 FE 二次切换带有可消除的振荡成本与不可忽略的 transfer-restart confound（§11），正式协议必须加 dwell。
- 相对 B（DWELL-500 SUFFICIENT）：500 已消除 <500 FE 的快速回切并把 terminal 提到 1.906，但残留 3 个 run 的 chattering 且 F1/F29 上略逊于 1000。
- 相对 D（DWELL-1500 REQUIRED）：1500 明确过度保守（−0.25 terminal），排除。
- 逐函数分歧（F10 上 dwell 略负）与 3 函数 × 10 seeds 的规模限制：以上仅是开发性证据，正式结论必须由全量 benchmark 或扩展诊断支持。

## 14. Q1–Q9 逐条回答

- **Q1（80–200 FE 二次切换是否 premature？）**：作为等 FE 决策，不是——强制成熟分支在 $t_0+h$ 处反而更差（$M_h<0$）；作为调度行为，是——它构成 shade↔cmaes 振荡并携带 restart confound，去除后整体更优。
- **Q2（500 FE dwell 是否足够？）**：基本足够（消除 <500 回切，terminal +0.065 vs raw），但不是最优。
- **Q3（1000 是否比 500 更稳定？）**：是：chattering 完全归零、terminal 再 +0.011、success +0.011，二次切换 median 恰为 3000 FE。
- **Q4（1500 是否过度保守？）**：是：terminal 掉 0.25，错失中后期有效转换。
- **Q5（滞回能否进一步减少振荡？）**：能归零振荡，但相对 P3 有 −0.10 的 terminal 代价；在 dwell=1000 下滞回不增值。
- **Q6（raw 收益在加 dwell 后是否保持？）**：保持且增强（1.841→1.916，比 P0 高 0.50）。
- **Q7（若收益消失是否来自 restart？）**：不适用（收益未消失）；但 §11 表明 restart 效应真实存在、方向个案依赖，仍是未完全解决的 confound（unresolved transfer-restart confound，已量化 5 例）。
- **Q8（post-handoff 是否存在 maturation pattern？）**：绝对进展意义上存在（85% 的切换 $P_{500}>P_{200}$，$P_{1000}$ 最大）；但等 FE 强制驻留不占优（shade 段明确更差），故 maturation 不能作为 dwell 收益的机制解释。
- **Q9（$H_{dwell}=H_g=1000$ FE 作为 ProgressForecast 主协议？）**：本轮证据支持把 1000 FE 作为两者的工作默认（§12），但按工作单要求不在此固定；待 ProgressForecast Stage-A 的 OOF 证据出具后一并预先指定。

## 15. 复现

```bash
.venv/bin/python behavior_with_ela/analysis_v2/task10_cadence_maturation.py --step ablation
.venv/bin/python behavior_with_ela/analysis_v2/task10_cadence_maturation.py --step analysis
.venv/bin/python behavior_with_ela/analysis_v2/task10_cadence_maturation.py --step maturation
.venv/bin/python behavior_with_ela/analysis_v2/task10_cadence_maturation.py --step curves
.venv/bin/python behavior_with_ela/analysis_v2/task10_cadence_maturation.py --step confound
```

聚合表：`results/analysis_v2/task10/{policy_comparison,switch_interval_chattering,segment_summary,segment_lifetime,policy_by_function,policy_by_prefix,maturation_branches,maturation_summary,maturation_by_target_algorithm,maturation_by_transition_pair,post_switch_progress_curves,progress_curve_summary,transfer_restart_confound}`。 历史 `cec2017_quick3*` 结果与 09/09b 报告未做任何改动。
