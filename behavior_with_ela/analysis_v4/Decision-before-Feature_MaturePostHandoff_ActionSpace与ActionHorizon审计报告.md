# Decision-before-Feature：Mature Post-Handoff Action-Space 与 Action-Horizon 审计总报告

- 日期：2026-08-30
- 协议：`mature_post_handoff_action_horizon_audit_v1`
- 范围：Task 11A（零 FE 重分析）+ Task 11B–O（新增 objective evaluations ≈121.96M FE：BBOB train 18 函数×3 instances + selected MA-BBOB 24 definitions，seeds 1–5，3 条 route，4,680 个 mature checkpoint × 3 动作 × {500,1000,terminal} 分支，10% × R=3 重复）。
- 原则核对：先证明 repeated state-action 问题本身存在，再谈模型——本轮零模型训练，全部结论来自真实标签。
- 报告组：`analysis_v4/01–08`；轻量表 `analysis_v4/task11/`；重表 `results/analysis_v4/task11/`；原始分片 `results/post_handoff/task11/`。

## 一、逐条回答（工作单 §32）

1. **Task 10 的 reversal event 正确数量**：P1（raw per-opportunity）A→B→A 反转任意间隔 **25 次**，其中间隔 <1000 FE **24 次**——两个数字是同一数据的不同口径（01 报告 A3）；Task 10 verdict 依赖的 <1000 口径正确，"25 次"未注明口径已修正。
2. **窗口速率修正后是否仍有 post-switch acceleration？** 弱化：仅 62–70% 的切换满足 $r_{200:500}>r_{0:200}$，73–83% 满足 $r_{500:1000}>r_{0:200}$，50–60% 满足 $r_{500:1000}>r_{200:500}$；shade 段中位 $r_{0:200}=0$。是"多数温和加速、少数减速"，**撤回**累计口径的"普遍加速"表述。
3. **native 与 PSO/SHADE→CMAES 的 state distribution 是否不同？** 是——位置差异巨大（FE=3000 时 current log10 gap：native −1.58 vs R1 −0.33 vs R2 −0.68），transfer 后 switch 分支的相对机会更差（R2−R0 的 1000-FE switch gain 差 −0.17～−0.23，CI 不含 0）；但动作结构层面差异在噪声内（06 报告）。
4/5/6. **500/1000/terminal 的 best-action 分布**：continue 占 91.0–93.1%（500）、91.7–93.1%（1000）、84.4–87.2%（terminal），pso/shade 合计 7–16% 且大量为并列零改善状态（04 报告 I1）。
7. **CMAES practical escape rate**：500：3.1–3.9%；**1000：2.9–3.9%**；terminal：4.7–7.4%（δ95 口径，跨 route 几乎相同）。
8. **best@1000 ≠ best@terminal 比例**：16.4%（raw）；practical 6.3–7.6%。
9. **disagreement 是否跨 problem/family 稳定**：switch@1000→continue@T 的 pattern 出现在 19/42 个 function group、4.3% 的状态——分布广但密度低。
10. **1000 horizon 的 action entropy 是否高于 terminal？** 否：0.454 vs 0.741 bits（terminal 更高，但同样不含可利用价值）。
11/12. **route+FE 能解释多少 1000-FE action value？state-wise 上界还剩多少？** 全部信息层级（current/route/route+FE）的经验最优动作都是 continue；state-wise oracle 相对 always-continue 的剩余增益仅 **0.013 log10（1000）**（500：0.009；terminal：0.042），比噪声 δ95（≈0.098）低约 7 倍。
13. **source history 是否在固定 current+FE 后改变动作偏好？** 否（动作偏好层面）；它改变的是位置（gap/segment behavior），而那不转化为动作选择差异。
14. **Action-space verdict：C. DEGENERATE**（判据逐条满足，见 04 报告）。
15. **Action-horizon verdict：H2. MODERATE HORIZON DEPENDENCE**（16% raw / 6–8% practical 分歧，但两边可得增益都被封顶在 ≈0.01–0.04，性能影响有限，见 05 报告）。
16. **repeated Selector 下一阶段**：**暂停**当前三算法 portfolio 上的 segment action selector（Case 3）——主标签换成 $G_{1000}$ 也不改变上界。
17. **是否生成 mature SHADE post-handoff routes？** 否：真实 $G_{1000}$ escape 仅 3.3%，无跨 family 的稳定 CMAES→SHADE 模式；扩展该 route 是无价值密度的 FE 消耗。
18. **是否允许 Behavior incremental test？** 否（无动作差异可供增量解释）。
19. **是否允许 ProgressForecast full pipeline？** 否（GO 条件 A/B 全部失败；且 Progress Gate 的下游收益同样被 0.013 的动作上界封顶）。
20. **是否转入 Portfolio Sufficiency Pilot？** **是**——这是唯一与证据一致的方向（Case 3）。
21. **正式 CEC2017**：**继续暂停**（当前方法没有可评价的 repeated action 声明；dwell=1000 仍保留为 Task 10 的调度开发性结论）。

## 二、双 verdict 与决策矩阵

$$
\boxed{\text{Verdict A：1000-FE repeated action space = DEGENERATE}}
$$
$$
\boxed{\text{Verdict B：Action horizon = MODERATE HORIZON DEPENDENCE（影响有限）}}
$$

→ **Case 3**：停止当前三算法 repeated Selector / ProgressForecast 扩展，下一阶段进入 **Portfolio Sufficiency Pilot**：
- 候选沿用预定义：**lbest-PSO、L-SHADE、IPOP-CMAES**（不新增）；
- 设计沿用本轮基础设施：同 route 基座 + checkpoint 分支 + 多 horizon + 短程噪声标定，把新算法作为第四动作加入 fork；
- Pilot 判据（预定义，不因结果改动）：marginal value $\Delta(a'|P)>0$、exclusive-win 率非平凡、跨多 family 复现、提升 $H(A^\star|prefix)$、非重复 CMAES 支配；
- WEAKLY 情形的"扩至 10 seeds"不适用（本轮是明确的 DEGENERATE，非 borderline）。

## 三、对 Task 10 结论的最终定位（本轮修正后）

1. dwell=1000 仍是有效的调度开发性结论（最小承诺/时间正则，不是成熟期——$M_h<0$ 且本轮证明 mature 段无 maturation 可兑现）；
2. repeated switching 在三个 CEC 开发函数上的收益**不能**归因于训练域上的 segment-level complementarity（RQ5 否定）；
3. task 9（initial routing 退化为查表）与 task 11（mature repeated routing 无信号）合并后的完整图景：**当前三算法组合在 BBOB/MA 训练域上，无论 initial 还是 mature 状态，动作空间都不承载超越 prefix 身份与噪声的状态依赖价值**。要使"动态算法选择"成为真问题，必须先改变问题（扩 portfolio 创造互补动作区），而不是继续改进模型。

## 四、成本账本（工作单 §27）

| 项 | 值 |
|---|---:|
| base route FE | 7,020,000 |
| branch FE（含 repetition 2,760 行） | 114,936,000 |
| 总新增 FE | ≈121,956,000 |
| BBOB / MA 分占比 | 3,240 / 1,440 states |
| states / checkpoints | 4,680 |
| branches | 16,800（base 14,040） |
| wall time | 4,018 core-秒（8 workers） |
| peak RSS | ≈160 MB/worker |
| 未来 progress-triggered 分支节省 | bottom20/30 触发率 0.20 → 分支成本 −80%；bottom40 → −60%（理论估计，未训练 Gate） |

## 五、本轮停止声明

按工作单 §31：Task 11A→O 与双 verdict 已完成，**STOP**。未训练 Behavior Selector / ProgressForecast，未扩 portfolio，未运行 validation/全量 CEC/CEC2022；Task 9/10 产物未改动（仅 01 报告对 Task 10 文本口径做出修正记录）。

## 六、下一阶段建议（唯一推荐路径）

**Portfolio Sufficiency Pilot**：在 BBOB train（5 seeds 可先行）+ selected MA-BBOB 上，对 R0/R1/R2 mature checkpoints 增加 lbest-PSO / L-SHADE / IPOP-CMAES 第四动作分支，按 Pilot 判据逐一评估 marginal value、exclusive-win、跨 family 稳定性与 $H(A^\star|prefix)$ 变化；只有 Pilot 建立了非退化动作空间，才重启 segment Selector 与 ProgressForecast 线。

### 下一步 prompt（可直接复制开新对话）

```
你正在继续 GitHub 项目 Decision-before-Feature（目录 behavior_with_ela/）。
先读 analysis_v4/ 全部 01-08 报告与总报告
Decision-before-Feature_MaturePostHandoff_ActionSpace与ActionHorizon审计报告.md，
遵守 AGENTS.md。当前已判定：三算法 mature post-handoff action space DEGENERATE
（oracle 增益 0.013 log10 << δ95≈0.098），horizon 为 MODERATE；决策矩阵为 Case 3。

本轮任务：执行 Portfolio Sufficiency Pilot（只做 pilot，不扩正式 portfolio）。
1. 复用 behavior_with_ela/post_handoff_audit.py 的 route 基座，
   在 R0/R1/R2 的 checkpoint 上为每状态增加三个新动作分支：
   lbest-PSO、L-SHADE、IPOP-CMAES（新优化器实现若不存在需先按项目
   optimizer 接口实现并单测，population transfer 语义与现有一致）；
2. 域：BBOB train 18 函数 × 3 instances × seeds 1-5（先 5 seeds）；
   checkpoints FE ∈ {3000,4000,5000,6000}；horizon {1000, terminal}；
   动作集 = {continue cmaes, pso, shade, lbest-pso, l-shade, ipop-cmaes}；
3. 对每个新算法 a' 计算 Δ(a'|P) = E[L_P*(s) − L_{P+a'}*(s)]、
   exclusive-win 率 U(a')、跨 family 分布、以及加入前后 H(A*|prefix) 变化；
   短程噪声按 Task 11H 协议重新标定（10% × R=3）；
4. 判据（预先固定）：positive marginal value、exclusive-win 非平凡、
   跨多 family 复现、提升条件动作多样性、非重复 CMAES 支配；
5. 输出 analysis_v4/09_portfolio_sufficiency_pilot.md 与轻量表，
   给出每个候选 KEEP / REJECT 建议与是否扩至 10 seeds；
6. 禁止：训练 selector/progress 模型、改动 v2/classifier、跑 CEC、
   使用 validation 选规则、删除任何阴性结果。
```
