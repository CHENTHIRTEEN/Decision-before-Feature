# Decision-before-Feature：Task 12 互补性驱动动态组合筛选 总报告

- 日期：2026-08-30
- 协议：`complementarity_driven_dynamic_portfolio_screening_v1`；总成本 ≈41.65M FE（分两阶段，较 all-to-all 直接分支节约 ≈48%）。
- 报告组：`analysis_v5/01–12`；轻量表 `analysis_v5/task12/`；重表 `results/analysis_v5/task12/`；原始分片 `results/portfolio_screening/task12/`。
- 定位声明（按工作单 §45）：The original three-solver portfolio was empirically shown to possess negligible dynamic action headroom due to solver dominance. Task 12 therefore performs outcome-independent complementarity screening over a pre-registered candidate pool, while retaining CMA-ES as a strong external and add-back control. Guo et al. (SWEVO 2025) 的 DE–GA 静态互补性仅作为外部假设被独立检验，未预设在我们的设定中成立。

## 一、§44 逐条回答

1. **7 候选 standalone strength**：terminal fb log10 gap 排序 lshade（−5.68）< shade（−5.46）< de（−3.16）< cso（−2.81）< pso（−2.95 前段弱）≈ lbestpso（−1.97）< ga（+0.46）；早期（FE=2000）cso 最强（−1.20）、ga 最弱。
2. **明显 weak**：ga（全域 exclusive win=0）、lbestpso（≤0.5%）、pso（≤1.8%）、de（≤0.5%）。
3. **redundant**：无（shade/lshade DCM 0.44 且双方 leave-one-out marginal VBS 均显著为正）。
4. **有真实 practical exclusive-win region**：shade（6.7–14.6%，3 families+MA）、lshade（12.6–15.6%，4 families+MA）、cso（FE=2000 时 28.2%，2 families+MA——早期区）。
5. **最大 exclusive gain mass**：FE=10000 lshade 0.761 > shade 0.490 >> cso 0.004；FE=2000 cso 0.392 独占鳌头。
6. **Stage 1 最互补 pair**：shade↔cso（natural DCM 0.385，最低）。
7. **DE–GA 是否复现 Guo 2025**：**NOT REPLICATED**——DE 对 GA 单边支配（P(GA≻DE)=0 处处成立），GA 全域 exclusive win=0（04 报告，含调优边界声明）。
8. **BBOB 与 MA 互补性是否一致**：一致（DCM 矩阵同型、MA 略更低；oracle headroom 两 suite 同向）。
9/10. **Pareto front 上的 3/4 元子集**：KEEP 集恰 3 个 → 唯一 3 元子集 {shade, lshade, cso} 即 P4；4 元子集不存在（禁止保留弱算法凑数）。P1/P2/P3 因成员 REJECT-WEAK 不可构造，处置已记录未删除。
11. **Stage 2 是否 single solver dominance**：无——practical 支配率 max 0.65（BBOB）/0.64（MA），且三 solver 各有 12–43% 的 practical best 份额。
12. **VBS–SBS headroom**：Δ_portfolio = **+0.209**（BBOB）/ **+0.195**（MA），bootstrap CI 远离 0。
13. **problem-static oracle 相对 SBS**：+0.152 / +0.158（选用 lshade 为全局 solver 损失最大）。
14. **problem+FE oracle 相对 problem-static**：+0.038 / +0.050（Δ_dynamic−Δ_problem）。
15. **state-wise 相对 problem+FE 再提升**：见 Δ_dynamic 本身（下条）。
16. **Δ_dynamic 是否明显为正**：**是**——BBOB +0.115 [0.074, 0.163]（点估计 > δ95=0.088）；MA +0.085 [0.056, 0.119]（**点估计 ≈ δ95=0.084，处于噪声边界**，CI>0，已如实标注）。
17. **H(A*|problem,FE) 是否明显非 0**：是——practical 口径 0.97 bits（上限 2 bits 的 49%）；相对边际熵 1.27 减少 0.30 bits（24%），其余 76% 为 state 级变异。
18/19. **同一 problem 轨迹 best action 是否变化、是否超过噪声**：P(varies)=0.60/0.58（≥2 个不同 practical best 的轨迹占比）；组内置换 null 为 0.70/0.75（p95 0.72/0.78）——观测切换**低于**随机 tie 噪声水平，即变化是结构性的（随 FE 的 winner 轮替）而非噪声并列。
20. **最终 balanced portfolio**：$P_{balanced}=\{\text{SHADE},\ \text{L-SHADE},\ \text{CSO}\}$（3 个，已预先固定）。
21. **保留理由**：shade=中后期独占区+marginal VBS +0.86；lshade=最强 standalone+最大 marginal VBS +1.21（支配率 0.65 未失控）；cso=唯一早期区（FE=2000 win 28.2%）+与 shade DCM 全矩阵最低 0.385+dynamic 下 16.5% 独占。
22. **淘汰理由**：见 05 报告裁决表（pso/de/lbestpso/ga 全部 exclusive-win≈0、gain-mass≈0）。
23. **CMAES add-back 是否 collapse**：**NO COLLAPSE（BBOB 与 MA 一致）**——cmaes practical win 27.1%/37.4%、Δ_dynamic 保持（0.094/0.083，比值 0.83/0.97）、practical entropy 上升至 1.74/1.75 bits。
24. **是否允许训练 Behavior Selector**：**允许进入 Behavior Incremental Test**（P1 路径的下一步）：须证明 Behavior（含 segment Behavior）优于 problem+current+FE 离散基线，对象为 Δ_dynamic≈0.09–0.11 的状态级残差。
25. **是否恢复 ProgressForecast**：本轮结束时**仍禁止**；仅在 Behavior Incremental Test 通过后按 §42 恢复。
26. **正式 CEC**：**继续 PAUSED**（CEC2022 held out；quick F1/F10/F29 仅历史 sanity check，未参与任何筛选/DCM/阈值）。
27. **下一阶段**：Behavior Incremental Test on P_balanced（主标签 $G_{1000}$；基线 = problem+current+FE 离散策略；特征 = bg_* / bs_* / 两者并集）。

## 二、最终 verdict

$$
\boxed{\text{Verdict P1：BALANCED DYNAMIC PORTFOLIO FOUND}}
$$
$$
\boxed{\text{CMAES ADD-BACK：NO COLLAPSE}}
$$

满足条件：practical VBS–SBS headroom 明显（+0.19~+0.21）；Δ_dynamic 明显为正（BBOB 0.115>δ95；MA 0.085≈δ95 为唯一边界项，已标注）；within-problem action variation 非平凡且结构性（0.60/0.58，低于 tie-null）；无近全支配（max 0.65）；3-algorithm subset 稳定。若采取保守读法（以 MA 贴边界为由降为 P3），下一步动作相同：扩至 10 seeds 而非训练模型——两种读法都不允许跳过 Behavior Incremental Test。

## 三、与 Task 9–11 结论的衔接

- Task 9（initial routing 退化为 prefix 查表）与本轮不冲突：initial 任务的退化源于 cmaes 对"从弱 solver 的 0.2B 状态出发"的单边支配；
- Task 11（mature cmaes-current 状态上 action space 退化）与本轮不冲突：**支配随状态分布走**——本轮状态由互补候选占据，cmaes 只作 add-back 时也只是诸强之一（12 报告）；
- Task 10 的 dwell=1000（最小承诺）继续有效，且本轮的 P_balanced 动态结构（cso→shade 的早期交棒、lshade→shade 的后期交棒）为 repeated DAS 提供了真实的动作序列假设。

## 四、成本账本

| 项 | FE |
|---|---:|
| Stage 1 natural runs | 32,560,000 |
| Stage 2 分支（含 1,890 重复分支） | 6,784,000 |
| CMA-ES add-back 分支（含重复） | 2,302,000 |
| **总计** | **≈41,646,000** |
| 节约对照（7 候选 all-to-all 直接分支 ≈80.2M） | ≈48% |

## 五、停止声明

按工作单 §46：筛选→dc→DE-GA→Pareto→Stage 2→噪声→DCM→headroom→variation→选组→预先固定→add-back→verdict 全部完成，**STOP**。未训练任何 Selector/ProgressForecast，未跑 validation/正式 CEC，未新增候选，Task 9–11 产物未改动。

## 六、下一阶段建议

**Behavior Incremental Test on P_balanced**：在同一 Stage 2 域上（可复用本轮全部标签，零新增 objective）训练 `problem+current+FE` 离散基线 vs Behavior RF（bg_28 / bs_28 / bg+bs），grouped OOF（按 cv_group），指标为 1000-FE 动作选择的 fb policy loss 与对 always-continue/lshade-SBS 的实际策略增益；只有 Behavior 增量成立，才进入 ProgressForecast 设计。

### 下一步 prompt（可直接复制开新对话）

```
你正在继续 GitHub 项目 Decision-before-Feature（目录 behavior_with_ela/）。
先读 analysis_v5/ 全部 01-12 报告与总报告
Decision-before-Feature_Task12_ComplementarityDrivenPortfolioScreening.md，
遵守 AGENTS.md。当前状态：P_balanced = {shade, lshade, cso} 已预先固定，
Δ_portfolio=+0.21/+0.19，Δ_dynamic=+0.115/+0.085（MA 贴噪声边界），
CMAES add-back NO COLLAPSE，verdict = P1。

本轮任务：Behavior Incremental Test（零新增 objective evaluations）。
1. 复用 results/analysis_v5/task12/dynamic_solver_loss_matrix.parquet 与
   dynamic_screening_states.parquet 的全部 1000-FE 标签；
2. 基线：(a) always-continue；(b) problem+current+FE 离散经验最优
   （function-balanced 拟合，leave-cv_group-out）；
3. 模型：RandomForestRegressor(200, depth 8, sqrt)（项目正式参数），
   特征组 M1=bg_28、M2=bs_28、M3=bg_28+bs_28、M4=problem+current+FE one-hot；
   grouped OOF 按 cv_group；主标签 = 1000-FE 动作损失（3 solver 矩阵）；
4. 评价：fb policy loss（按 OOF 预测选动作）、对 always-continue 的实际
   函数平衡策略增益、对 lshade-SBS 的增益、practical 达标率（δ95=0.084/0.088）；
5. 判定：Behavior 任一特征组须在两个 suite 同时优于 M4 基线且增量超过
   短程噪声，才允许进入 ProgressForecast 设计；否则记录 NO INCREMENT；
6. 输出 analysis_v5/13_behavior_incremental_test.md 与轻量表；
7. 禁止：调参搜索、新特征工程、CEC、validation 参与、删除阴性结果。
```
